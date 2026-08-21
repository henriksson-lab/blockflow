// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Two images, one answer.** Before this the crate could combine two volumes
// only with a Boolean connective (`ops::voxelwise::Logic`) or with the one
// subtraction written for one consumer (`ops::background::DifferenceCombine`),
// and it could convolve only with a Gaussian. This file is the acceptance suite
// for what closes that: `ArithmeticCombine` — add, subtract, multiply, divide,
// per-voxel minimum and maximum — and `ConvolveOp`, a general kernel with a
// named sense and a named boundary convention.
//
// What it pins, in the order the claims depend on each other:
//
//  1. **A composed filter is the hand-written reference, bit for bit.** A Sobel
//     gradient magnitude built out of two convolutions, two multiplications, an
//     addition and a square root, against a direct triple loop written in this
//     file from the definition. Nothing in the reference calls the op.
//  2. **It is decomposition-invariant**: byte-identical at every block edge,
//     including cuts that divide no axis of the extent and a **one-voxel**
//     block, where every voxel of the answer comes from its own halo.
//  3. **The same for the combines against a supplied image** — the second
//     operand being an array the caller handed the run (`ImageId::supplied(0)`,
//     `ArrayEnvironment::with_inputs`) rather than a branch of the plan.
//  4. **The negative controls.** Four programs that differ from the right one by
//     one thing each — the kernel flipped (correlation for convolution), the
//     boundary convention changed, the anchor moved by one voxel, the operands
//     of a subtraction swapped — each producing plausible output and a different
//     answer, with the number of voxels that moved asserted.
//  5. **The liveness partners**, and there are four of them because building
//     this file turned up two that are not obvious:
//     * a **symmetric** kernel cannot tell correlation from convolution;
//     * a **constant** field cannot tell any of the four controls apart;
//     * a kernel that reaches **one voxel** cannot tell `Clamp` from `Reflect`,
//       because at that distance the two are the same function — which is why
//       the boundary control below uses a kernel that reaches two;
//     * a **gradient magnitude** cannot tell correlation from convolution
//       either. A Sobel kernel is antisymmetric, so the flip negates each
//       derivative, and squaring cancels it. That is the most natural fixture
//       anyone would reach for to test a convolution and it is blind; the
//       composed filter this file demonstrates is therefore deliberately *not*
//       the fixture its negative controls run on.
//     Each is asserted, so that a suite passing on those fixtures would be known
//     to be saying nothing.
//  6. **The refusals**, by name and before a block runs: an arithmetic combine
//     over an integer chain, and a supplied array that is not in image 0's
//     coordinate space. The rest of the refusals are algebra rather than
//     assembly and are pinned where they live — branches of different extents,
//     a third branch handed to a subtraction, a `ClippedStart` element handed to
//     a kernel — in `ops::voxelwise` and `ops::convolve`'s own test modules.
//
// No assertion here is on wall-clock time.

use ndarray::Array3;

use blockflow::assemble::ImageId;
use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, Chain};
use blockflow::ops::{
    Arithmetic, ArithmeticCombine, Boundary, ConvolveOp, Gaussian, Kernel, Sense, SmoothOp,
    VoxelwiseMapOp,
};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

/// Small, and **prime on two axes**, so that a cubic block edge divides no axis
/// and every grid below has ragged blocks at the high faces.
const VOLUME: [usize; 3] = [9, 7, 11];

// ------------------------------------------------------------- fixtures --

/// A field with no symmetry, no plateau and no axis it is constant along.
///
/// Every one of those would make one of the negative controls below agree with
/// the right answer for a reason that has nothing to do with the op: a field
/// symmetric under negation cannot tell a flipped kernel from an unflipped one,
/// and a field constant near a face cannot tell one boundary convention from
/// another.
fn field() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        let ramp = (3 * i + 5 * j + 7 * k) as f64 * 0.125;
        let step = if i * 2 + j > k * 3 { 4.0 } else { -1.5 };
        let ripple = ((i * 13 + j * 5 + k * 3) % 7) as f64 * 0.5;
        ramp + step + ripple
    })
}

/// A second field, for the two-image cases. Strictly positive, so that a
/// division by it is finite everywhere and the test is about the arithmetic
/// rather than about the infinities (which have their own test in
/// `ops::voxelwise`).
fn reference_field() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        1.0 + ((i * 5 + j * 3 + k) % 11) as f64 * 0.25
    })
}

/// The fixture that can distinguish nothing, kept so the suite can say so.
fn constant_field() -> Array3<f64> {
    Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 2.75)
}

// ------------------------------------------------------- the two kernels --

/// The 3x3 Sobel derivative along **axis 1**, flat on axis 0.
///
/// The weights are in the element's own order — ascending on axis 0, then 1,
/// then 2 — which for this element is the `(d1, d2)` box read row-major.
fn sobel_axis_1() -> Kernel {
    Kernel::from_sides(
        [0, 1, 1],
        [0, 1, 1],
        vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
    )
    .expect("a kernel")
}

/// The 3x3 Sobel derivative along **axis 2**.
fn sobel_axis_2() -> Kernel {
    Kernel::from_sides(
        [0, 1, 1],
        [0, 1, 1],
        vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
    )
    .expect("a kernel")
}

// ------------------------------------------------ the hand-written oracle --

/// A direct correlation of a flat 3x3 kernel with the array's edge **clamped**,
/// written from the definition and calling nothing in `ops`.
///
/// The taps are summed in the same order the element enumerates them — `d1`
/// outer, `d2` inner — because floating-point addition does not associate and a
/// reference that summed them in another order would be a different number in
/// the last bit. That is not a concession: it is the statement that the op's sum
/// order is part of its contract, which is exactly what makes the answer
/// decomposition-invariant.
fn correlate_by_hand(input: &Array3<f64>, weights: &[f64; 9]) -> Array3<f64> {
    let extent = [VOLUME[0] as isize, VOLUME[1] as isize, VOLUME[2] as isize];
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        let mut total = 0.0f64;
        let mut which = 0usize;
        for d1 in -1isize..=1 {
            for d2 in -1isize..=1 {
                let jj = (j as isize + d1).clamp(0, extent[1] - 1) as usize;
                let kk = (k as isize + d2).clamp(0, extent[2] - 1) as usize;
                total += weights[which] * input[[i, jj, kk]];
                which += 1;
            }
        }
        total
    })
}

/// The gradient magnitude, by hand: `sqrt(g1 * g1 + g2 * g2)`.
///
/// The expression order matters for the same reason the tap order does, and it
/// is the order the chain evaluates in — branch 0's square, then branch 1's,
/// then their sum, then the root.
fn sobel_magnitude_by_hand(input: &Array3<f64>) -> Array3<f64> {
    let g1 = correlate_by_hand(input, &[-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0]);
    let g2 = correlate_by_hand(input, &[-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]);
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |at| {
        (g1[at] * g1[at] + g2[at] * g2[at]).sqrt()
    })
}

// -------------------------------------------------------- the assemblies --

/// One convolution, as a chain node.
fn convolution(name: &'static str, kernel: Kernel, sense: Sense, boundary: Boundary) -> Chain {
    Chain::op(ConvolveOp::new(name, kernel, sense, boundary))
}

/// `x * x` where `x` is one convolution: a two-branch diamond whose sink is a
/// [`Arithmetic::Multiply`].
///
/// **Both branches compute the same convolution**, which is redundant work and
/// is deliberate here: squaring through a `VoxelwiseMapOp` would be cheaper and
/// would demonstrate nothing about a two-image multiply, which is the thing
/// under test.
fn square_of(name: &'static str, kernel: Kernel, sense: Sense, boundary: Boundary) -> Chain {
    Chain::parallel(
        vec![
            convolution(name, kernel.clone(), sense, boundary),
            convolution(name, kernel, sense, boundary),
        ],
        Box::new(ArithmeticCombine::new("square", Arithmetic::Multiply)),
    )
    .expect("a fan-in of two")
}

/// The whole filter: two squared derivatives, added, rooted.
fn sobel_magnitude(sense: Sense, boundary: Boundary, one: Kernel, two: Kernel) -> Chain {
    let magnitude = Chain::parallel(
        vec![
            square_of("gradient.1", one, sense, boundary),
            square_of("gradient.2", two, sense, boundary),
        ],
        Box::new(ArithmeticCombine::new("sum_of_squares", Arithmetic::Add)),
    )
    .expect("a fan-in of two");
    Chain::sequence(vec![
        magnitude,
        Chain::op(VoxelwiseMapOp::new("root", f64::sqrt)),
    ])
}

/// The one this suite is about: correlation, clamped, the anchor centred.
fn the_filter() -> Chain {
    sobel_magnitude(
        Sense::Correlate,
        Boundary::Clamp,
        sobel_axis_1(),
        sobel_axis_2(),
    )
}

// ----------------------------------------------------------- the harness --

fn workflow(chain: Chain) -> Workflow {
    Workflow::new(chain, VOLUME, Dtype::F64)
}

/// One phase holding the whole chain, at a stated block extent.
///
/// The reach is the chain's **own**, so nothing here can hide one that is wrong:
/// an op that under-declared would be short of a halo and the comparison against
/// the whole-volume reference would say so.
fn plan(workflow: &Workflow, block: [usize; 3]) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let reach = workflow.chain.reach3(&VOLUME);
    let grid = BlockGrid::new(VOLUME, block).expect("a grid");
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid);
    Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: reach,
    }
}

/// Block extents that between them exercise every case the bar names: the whole
/// volume, cuts that divide no axis, a slab on each axis alone, and one voxel.
fn blocks() -> Vec<[usize; 3]> {
    vec![
        VOLUME,
        [4, 4, 4],
        [2, 3, 2],
        [5, 5, 5],
        [3, 7, 5],
        [9, 2, 5],
        [9, 7, 2],
        [1, 1, 1],
    ]
}

fn run(chain: Chain, input: &Array3<f64>) -> Vec<(String, Array3<f64>)> {
    let workflow = workflow(chain);
    blocks()
        .into_iter()
        .map(|block| {
            let decomposition = plan(&workflow, block);
            let env =
                ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4])
                    .expect("an environment");
            execute("filter", &workflow, &decomposition, &Hints::default(), &env).expect("a run");
            (
                format!("{block:?}"),
                env.output().view::<f64>().unwrap().to_owned(),
            )
        })
        .collect()
}

/// The chain applied once, over the whole array, with no plan and no blocking.
fn resident(chain: &Chain, input: &Array3<f64>) -> Array3<f64> {
    let source: Voxels = input.clone().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).expect("a buffer");
    chain
        .apply(&source, &mut out, &Anchor::whole(VOLUME))
        .expect("the whole-volume reference must run");
    out.view::<f64>().unwrap().to_owned()
}

fn moved(left: &Array3<f64>, right: &Array3<f64>) -> usize {
    left.iter()
        .zip(right.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

// ------------------------------------- 1 and 2. the filter, and invariance --

/// **The acceptance test.** A gradient magnitude composed from a general
/// convolution and two arithmetic combines is the hand-written reference, bit
/// for bit, at every block edge including one voxel.
#[test]
fn a_composed_gradient_magnitude_is_the_hand_written_answer_under_every_decomposition() {
    let input = field();
    let wanted = sobel_magnitude_by_hand(&input);

    // The oracle is not degenerate: it varies, it is not the input, and it is
    // not zero anywhere near everywhere.
    assert!(distinct(&wanted) > 20, "the oracle barely varies");
    assert_ne!(wanted, input);
    assert!(wanted.iter().any(|&value| value > 1.0));

    for (block, got) in run(the_filter(), &input) {
        assert_eq!(got, wanted, "block {block}");
    }
}

/// The decomposed answers are the resident one too, which is the same claim from
/// the other side and is what a reader comparing this file against
/// `ops::background` will look for.
#[test]
fn the_resident_chain_and_every_decomposition_agree() {
    let input = field();
    let wanted = resident(&the_filter(), &input);
    assert_eq!(wanted, sobel_magnitude_by_hand(&input));
    for (block, got) in run(the_filter(), &input) {
        assert_eq!(got, wanted, "block {block}");
    }
}

/// Every one of the six operations is decomposition-invariant on its own, over
/// two branches of one plan.
///
/// The branches are two different Gaussians, so the two operands really differ
/// and a combine that read one of them twice could not pass.
#[test]
fn every_arithmetic_combine_is_decomposition_invariant() {
    let input = field();
    for op in [
        Arithmetic::Add,
        Arithmetic::Subtract,
        Arithmetic::Multiply,
        Arithmetic::Divide,
        Arithmetic::Minimum,
        Arithmetic::Maximum,
    ] {
        let chain = || {
            Chain::parallel(
                vec![
                    Chain::op(SmoothOp::new(
                        "narrow",
                        Gaussian::isotropic(0.8, 3.0).unwrap(),
                    )),
                    Chain::op(SmoothOp::new(
                        "wide",
                        Gaussian::isotropic(2.0, 3.0).unwrap(),
                    )),
                ],
                Box::new(ArithmeticCombine::new("op", op)),
            )
            .expect("a fan-in of two")
        };
        let wanted = resident(&chain(), &input);
        // not a constant, and not either operand
        assert!(
            wanted
                .iter()
                .map(|value| value.to_bits())
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                > 1,
            "{op:?} produced a constant"
        );
        for (block, got) in run(chain(), &input) {
            assert_eq!(got, wanted, "{op:?}, block {block}");
        }
    }
}

/// A **difference of Gaussians**, which is the composition the survey names
/// first and the one `ops::background`'s diamond was already shaped for.
#[test]
fn a_difference_of_gaussians_is_the_two_blurs_subtracted() {
    let input = field();
    let chain = || {
        Chain::parallel(
            vec![
                Chain::op(SmoothOp::new(
                    "narrow",
                    Gaussian::isotropic(0.8, 3.0).unwrap(),
                )),
                Chain::op(SmoothOp::new(
                    "wide",
                    Gaussian::isotropic(2.0, 3.0).unwrap(),
                )),
            ],
            Box::new(ArithmeticCombine::new("difference", Arithmetic::Subtract)),
        )
        .expect("a fan-in of two")
    };
    // The reference: each blur taken on its own, then subtracted by hand.
    let narrow = resident(
        &Chain::op(SmoothOp::new(
            "narrow",
            Gaussian::isotropic(0.8, 3.0).unwrap(),
        )),
        &input,
    );
    let wide = resident(
        &Chain::op(SmoothOp::new(
            "wide",
            Gaussian::isotropic(2.0, 3.0).unwrap(),
        )),
        &input,
    );
    let wanted = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |at| {
        narrow[at] - wide[at]
    });
    assert!(wanted.iter().any(|&value| value > 0.0));
    assert!(wanted.iter().any(|&value| value < 0.0));
    for (block, got) in run(chain(), &input) {
        assert_eq!(got, wanted, "block {block}");
    }
}

// ------------------------------------------ 3. against a **supplied** image --

/// The chain for the second half of the gap: one arm is the phase's own input,
/// the other is an array the caller handed the run.
fn against_supplied(op: Arithmetic, reversed: bool) -> Chain {
    let kept = Chain::op(VoxelwiseMapOp::new("kept", |value: f64| value));
    let supplied = Chain::source(ImageId::supplied(0), Dtype::F64);
    let branches = if reversed {
        vec![supplied, kept]
    } else {
        vec![kept, supplied]
    };
    Chain::parallel(branches, Box::new(ArithmeticCombine::new("against", op)))
        .expect("a fan-in of two")
}

fn supplied_plan(chain: &Chain, block: [usize; 3]) -> Decomposition {
    let slots = chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let reach = chain.reach3(&VOLUME);
    let grid = BlockGrid::new(VOLUME, block).expect("a grid");
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid);
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![phase],
        chain_reach: reach,
    };
    plan.declare_dtypes(chain).expect("element types");
    plan.declare_source_images(chain).expect("source images");
    plan
}

fn run_against_supplied(
    op: Arithmetic,
    reversed: bool,
    input: &Array3<f64>,
    other: &Array3<f64>,
    block: [usize; 3],
) -> Array3<f64> {
    let chain = against_supplied(op, reversed);
    let decomposition = supplied_plan(&chain, block);
    let workflow = Workflow::new(against_supplied(op, reversed), VOLUME, Dtype::F64);
    let env = ArrayEnvironment::with_inputs(
        input.clone().into(),
        vec![other.clone().into()],
        &decomposition,
        [4, 4, 4],
    )
    .expect("an environment");
    execute(
        "supplied",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .expect("a run");
    env.output().view::<f64>().unwrap().to_owned()
}

/// **Arithmetic against an array the run did not compute**, which is what G5
/// made reachable and what nothing had used for it yet. Every one of the six, at
/// every block edge, against the arithmetic written out.
#[test]
fn arithmetic_against_a_supplied_image_is_the_resident_answer_under_every_decomposition() {
    let input = field();
    let other = reference_field();
    for op in [
        Arithmetic::Add,
        Arithmetic::Subtract,
        Arithmetic::Multiply,
        Arithmetic::Divide,
        Arithmetic::Minimum,
        Arithmetic::Maximum,
    ] {
        let wanted = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |at| {
            op.apply(input[at], other[at])
        });
        assert!(wanted.iter().all(|value| value.is_finite()));
        assert_ne!(wanted, input, "{op:?} gave the input back");
        assert_ne!(wanted, other, "{op:?} gave the operand back");
        for block in blocks() {
            let got = run_against_supplied(op, false, &input, &other, block);
            assert_eq!(got, wanted, "{op:?}, block {block:?}");
        }
    }
}

// ------------------------------------------- a kernel that discriminates --

/// A **5x5 kernel with no symmetry whatever**, flat on axis 0.
///
/// The Sobel pair above cannot be the fixture for the negative controls, and
/// finding that out is half of what this file is for — see
/// [`the_gradient_magnitude_is_blind_to_the_flip`]. This one reaches two voxels,
/// so the two boundary conventions differ on it, and it is not the negation of
/// itself, so the two senses differ on it.
fn wide_lo() -> [usize; 3] {
    [0, 2, 2]
}
fn wide_hi() -> [usize; 3] {
    [0, 2, 2]
}
fn wide_weights() -> Vec<f64> {
    (0..25)
        .map(|which: usize| ((which * 7) % 11) as f64 - 5.0 + which as f64 * 0.25)
        .collect()
}
fn wide_kernel() -> Kernel {
    Kernel::from_sides(wide_lo(), wide_hi(), wide_weights()).expect("a kernel")
}

/// One convolution and nothing else, as a chain.
fn one_convolution(kernel: Kernel, sense: Sense, boundary: Boundary) -> Chain {
    convolution("wide", kernel, sense, boundary)
}

/// The boundary rules, written out rather than borrowed from the crate.
///
/// `Reflect` is the **half-sample** mirror — the mirror sits between `-1` and
/// `0`, so `-1` reads `0` — folded repeatedly by the period `2 * extent`, which
/// is what `ops::ridge::Boundary` documents and what a reference has to
/// reproduce independently to be worth anything.
fn resolve(boundary: Boundary, position: isize, extent: usize) -> usize {
    let extent = extent as isize;
    if extent <= 1 {
        return 0;
    }
    match boundary {
        Boundary::Clamp => position.clamp(0, extent - 1) as usize,
        Boundary::Reflect => {
            let period = 2 * extent;
            let mut folded = position % period;
            if folded < 0 {
                folded += period;
            }
            (if folded >= extent {
                period - 1 - folded
            } else {
                folded
            }) as usize
        } // No catch-all: `Boundary` has exactly these two today, and a third
          // one should fail to compile here rather than silently fall through to
          // whichever of them a wildcard picked.
    }
}

/// A direct linear filter over a dense box kernel, in either sense, with either
/// boundary rule — written from the definition and calling nothing in `ops`.
///
/// The taps are enumerated in the element's own order (axis 0 outer, axis 2
/// inner), which is the order the sum has to be taken in for the answer to be
/// the same bits.
fn filter_by_hand(
    input: &Array3<f64>,
    lo: [usize; 3],
    hi: [usize; 3],
    weights: &[f64],
    sense: Sense,
    boundary: Boundary,
) -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        let anchor = [i as isize, j as isize, k as isize];
        let mut total = 0.0f64;
        let mut which = 0usize;
        for d0 in -(lo[0] as isize)..=hi[0] as isize {
            for d1 in -(lo[1] as isize)..=hi[1] as isize {
                for d2 in -(lo[2] as isize)..=hi[2] as isize {
                    let delta = match sense {
                        Sense::Correlate => [d0, d1, d2],
                        Sense::Convolve => [-d0, -d1, -d2],
                    };
                    let at = [
                        resolve(boundary, anchor[0] + delta[0], VOLUME[0]),
                        resolve(boundary, anchor[1] + delta[1], VOLUME[1]),
                        resolve(boundary, anchor[2] + delta[2], VOLUME[2]),
                    ];
                    total += weights[which] * input[at];
                    which += 1;
                }
            }
        }
        total
    })
}

fn distinct(values: &Array3<f64>) -> usize {
    values
        .iter()
        .map(|value| value.to_bits())
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

// ----------------------------------------------------- 4. negative controls --

/// The wide kernel is the hand-written answer in **all four** combinations of
/// sense and boundary — which is what makes each of the controls below a
/// statement about the op rather than about the fixture.
#[test]
fn a_general_kernel_is_the_hand_written_answer_in_every_sense_and_convention() {
    let input = field();
    for sense in [Sense::Correlate, Sense::Convolve] {
        for boundary in [Boundary::Clamp, Boundary::Reflect] {
            let wanted = filter_by_hand(
                &input,
                wide_lo(),
                wide_hi(),
                &wide_weights(),
                sense,
                boundary,
            );
            let got = resident(&one_convolution(wide_kernel(), sense, boundary), &input);
            assert_eq!(got, wanted, "{sense:?} {boundary:?}");
        }
    }
}

/// …and under every decomposition, at a **real halo of two**.
#[test]
fn a_general_kernel_is_decomposition_invariant() {
    let input = field();
    for sense in [Sense::Correlate, Sense::Convolve] {
        let wanted = filter_by_hand(
            &input,
            wide_lo(),
            wide_hi(),
            &wide_weights(),
            sense,
            Boundary::Clamp,
        );
        for (block, got) in run(
            one_convolution(wide_kernel(), sense, Boundary::Clamp),
            &input,
        ) {
            assert_eq!(got, wanted, "{sense:?}, block {block}");
        }
    }
}

/// **Correlation is not convolution.** The same program with the sense changed
/// produces a plausible filtered volume and a different one — and the count is
/// asserted, because a control that moved three voxels in a corner would not be
/// a control.
#[test]
fn flipping_the_kernel_moves_the_answer() {
    let input = field();
    let right = resident(
        &one_convolution(wide_kernel(), Sense::Correlate, Boundary::Clamp),
        &input,
    );
    let flipped = resident(
        &one_convolution(wide_kernel(), Sense::Convolve, Boundary::Clamp),
        &input,
    );
    assert!(flipped.iter().all(|value| value.is_finite()));
    let count = moved(&right, &flipped);
    assert_eq!(
        count,
        right.len(),
        "a flipped kernel moved {count} of {} voxels",
        right.len()
    );
    // …and the flip is a reflection of the kernel rather than a second
    // algorithm: convolving with the reflected kernel is correlating with it.
    let reflected = wide_kernel().reflected().expect("a reflection");
    assert_eq!(
        resident(
            &one_convolution(reflected, Sense::Convolve, Boundary::Clamp),
            &input
        ),
        right
    );
}

/// **A wrong boundary convention** moves the faces and nothing else, which is
/// the sharpest form the control can take: it says both that the convention is
/// real and that it is confined to where it should be.
#[test]
fn changing_the_boundary_convention_moves_the_faces_and_only_the_faces() {
    let input = field();
    let right = resident(
        &one_convolution(wide_kernel(), Sense::Correlate, Boundary::Clamp),
        &input,
    );
    let reflected = resident(
        &one_convolution(wide_kernel(), Sense::Correlate, Boundary::Reflect),
        &input,
    );
    let count = moved(&right, &reflected);
    assert!(count > 0, "the two boundary conventions agreed everywhere");
    let mut interior = 0usize;
    for i in 0..VOLUME[0] {
        for j in 2..VOLUME[1] - 2 {
            for k in 2..VOLUME[2] - 2 {
                assert_eq!(
                    right[[i, j, k]].to_bits(),
                    reflected[[i, j, k]].to_bits(),
                    "the interior moved at {i} {j} {k}"
                );
                interior += 1;
            }
        }
    }
    assert!(
        count <= right.len() - interior,
        "{count} moved but only {} voxels are outside the interior",
        right.len() - interior
    );
    // The **outermost** layer is where the two conventions really part: at a
    // distance of one the mirror and the clamp agree (`-1` reads `0` under
    // both), so the second layer in need not move and the first must.
    let mut outermost = 0usize;
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                if j == 0 || j == VOLUME[1] - 1 || k == 0 || k == VOLUME[2] - 1 {
                    assert_ne!(
                        right[[i, j, k]].to_bits(),
                        reflected[[i, j, k]].to_bits(),
                        "the outermost layer did not move at {i} {j} {k}"
                    );
                    outermost += 1;
                }
            }
        }
    }
    // And it is *exactly* the outermost layer: 288 of 693 on this fixture, with
    // the 189-voxel interior and the second layer in both untouched.
    assert_eq!(
        count, outermost,
        "{count} moved and {outermost} are on the outermost layer"
    );
}

/// **An off-by-one anchor.** The same twenty-five weights over a
/// twenty-five-voxel box, moved one voxel along axis 2 — the classic
/// transcription slip, and one that leaves the output looking exactly like the
/// filter it is not.
#[test]
fn moving_the_anchor_by_one_voxel_moves_the_answer() {
    let input = field();
    let right = resident(
        &one_convolution(wide_kernel(), Sense::Correlate, Boundary::Clamp),
        &input,
    );
    let shifted = Kernel::from_sides([0, 2, 1], [0, 2, 3], wide_weights()).expect("a kernel");
    let got = resident(
        &one_convolution(shifted, Sense::Correlate, Boundary::Clamp),
        &input,
    );
    assert!(got.iter().all(|value| value.is_finite()));
    let count = moved(&right, &got);
    assert_eq!(
        count,
        right.len(),
        "an anchor moved by one voxel moved only {count} of {} voxels",
        right.len()
    );
    // and it is really the same weights, so the control changed one thing
    assert_eq!(
        Kernel::from_sides([0, 2, 1], [0, 2, 3], wide_weights())
            .unwrap()
            .weights(),
        wide_kernel().weights()
    );
}

/// **The operands of a subtraction are not interchangeable**, and neither are a
/// division's. A combine that folded its branches in the wrong order would still
/// produce a well-formed volume.
#[test]
fn swapping_the_operands_moves_the_answer_for_the_two_that_do_not_commute() {
    let input = field();
    let other = reference_field();
    for op in [Arithmetic::Subtract, Arithmetic::Divide] {
        let right = run_against_supplied(op, false, &input, &other, [4, 4, 4]);
        let swapped = run_against_supplied(op, true, &input, &other, [4, 4, 4]);
        let count = moved(&right, &swapped);
        assert!(
            count > right.len() / 2,
            "{op:?} moved only {count} voxels when its operands were swapped"
        );
    }
    // …and the four that do commute are unmoved by the swap, which is what
    // makes the two above a property of those two rather than of the harness.
    for op in [
        Arithmetic::Add,
        Arithmetic::Multiply,
        Arithmetic::Minimum,
        Arithmetic::Maximum,
    ] {
        let right = run_against_supplied(op, false, &input, &other, [4, 4, 4]);
        let swapped = run_against_supplied(op, true, &input, &other, [4, 4, 4]);
        assert_eq!(right, swapped, "{op:?} is commutative on this fixture");
    }
}

// ---------------------------------------------------- 5. liveness partners --

/// **A symmetric kernel cannot tell the two senses apart**, so a suite that only
/// ever tested a box blur would be silent about the distinction the crate went
/// to the trouble of naming.
#[test]
fn a_symmetric_kernel_cannot_distinguish_correlation_from_convolution() {
    let input = field();
    let symmetric = || Kernel::from_sides([0, 2, 2], [0, 2, 2], vec![1.0; 25]).unwrap();
    let correlated = resident(
        &one_convolution(symmetric(), Sense::Correlate, Boundary::Clamp),
        &input,
    );
    let convolved = resident(
        &one_convolution(symmetric(), Sense::Convolve, Boundary::Clamp),
        &input,
    );
    assert_eq!(correlated, convolved);
    assert!(distinct(&correlated) > 20, "a live fixture");
}

/// **A kernel that reaches one voxel cannot tell the two boundary conventions
/// apart**, because at that distance they are the same function: `-1` resolves
/// to `0` and `n` to `n - 1` under both. `ops::ridge::hessian_at` takes no
/// convention for exactly this reason and says so; this is the same fact from
/// the other side, and it is why the boundary control above uses a kernel that
/// reaches two.
#[test]
fn a_kernel_that_reaches_one_voxel_cannot_distinguish_the_boundary_conventions() {
    let input = field();
    let narrow = || sobel_axis_2();
    let clamped = resident(
        &one_convolution(narrow(), Sense::Correlate, Boundary::Clamp),
        &input,
    );
    let reflected = resident(
        &one_convolution(narrow(), Sense::Correlate, Boundary::Reflect),
        &input,
    );
    assert_eq!(clamped, reflected);
    assert!(distinct(&clamped) > 20, "a live fixture");
}

/// **The gradient magnitude itself cannot tell the two senses apart**, and that
/// is worth a test of its own because it is a trap rather than a curiosity.
///
/// A Sobel kernel is *antisymmetric* under negation, so flipping it negates the
/// derivative; the magnitude squares each derivative before adding, so the sign
/// cancels and the whole filter is invariant. Anybody testing a convolution
/// implementation on a gradient magnitude — which is the most natural thing to
/// reach for — would find correlation and convolution indistinguishable and
/// would conclude the wrong thing about their code. The composed filter this
/// file demonstrates is therefore **not** the fixture its negative controls run
/// on, and this test is the reason.
#[test]
fn the_gradient_magnitude_is_blind_to_the_flip() {
    let input = field();
    let correlated = resident(&the_filter(), &input);
    let convolved = resident(
        &sobel_magnitude(
            Sense::Convolve,
            Boundary::Clamp,
            sobel_axis_1(),
            sobel_axis_2(),
        ),
        &input,
    );
    assert_eq!(correlated, convolved);
    assert!(distinct(&correlated) > 20, "a live fixture");
    // The blindness is the magnitude's, not the convolution's: one derivative on
    // its own is negated by the flip, everywhere it is not zero.
    let one = resident(
        &convolution("g", sobel_axis_1(), Sense::Correlate, Boundary::Clamp),
        &input,
    );
    let other = resident(
        &convolution("g", sobel_axis_1(), Sense::Convolve, Boundary::Clamp),
        &input,
    );
    let negated = moved(&one, &other);
    assert!(negated > one.len() / 2, "only {negated} derivatives moved");
    for (a, b) in one.iter().zip(other.iter()) {
        assert_eq!(a.to_bits(), (-b).to_bits(), "the flip is a negation");
    }
}

/// **A constant field distinguishes none of the controls.** Flip the kernel,
/// change the boundary, move the anchor: a linear filter of a constant is the
/// weight sum times the constant under every one of them, so a suite built on a
/// flat fixture would pass with every control in place.
#[test]
fn a_constant_field_distinguishes_none_of_the_controls() {
    let flat = constant_field();
    let programs = [
        one_convolution(wide_kernel(), Sense::Correlate, Boundary::Clamp),
        one_convolution(wide_kernel(), Sense::Convolve, Boundary::Clamp),
        one_convolution(wide_kernel(), Sense::Correlate, Boundary::Reflect),
        one_convolution(
            Kernel::from_sides([0, 2, 1], [0, 2, 3], wide_weights()).unwrap(),
            Sense::Correlate,
            Boundary::Clamp,
        ),
    ];
    let first = resident(&programs[0], &flat);
    assert_eq!(distinct(&first), 1, "a constant in gives a constant out");
    for (which, program) in programs.iter().enumerate().skip(1) {
        assert_eq!(
            resident(program, &flat),
            first,
            "control {which} was distinguished by a constant field"
        );
    }
    // …and every one of them is distinguished by the live fixture, which is the
    // half that makes this a partner rather than a curiosity.
    let live = field();
    let right = resident(&programs[0], &live);
    for (which, program) in programs.iter().enumerate().skip(1) {
        assert!(
            moved(&right, &resident(program, &live)) > 0,
            "control {which} was not distinguished by the live fixture either"
        );
    }
}

// ------------------------------------------------------------ 6. refusals --

/// An arithmetic combine over an integer chain is refused when the chain is
/// built, not when a block runs.
#[test]
fn arithmetic_over_an_integer_chain_is_refused_by_name() {
    let integers = || {
        Chain::parallel(
            vec![
                Chain::source(ImageId::supplied(0), Dtype::U16),
                Chain::source(ImageId::supplied(1), Dtype::U16),
            ],
            Box::new(ArithmeticCombine::new("add", Arithmetic::Add)),
        )
        .expect("the chain is built; `produces` is where it is judged")
    };
    let refusal = integers().produces(Dtype::U16).expect_err("a refusal");
    let message = refusal.to_string();
    assert!(message.contains("add"), "{message}");
    assert!(message.contains("uint16"), "{message}");

    // …and the two selections are accepted over exactly the same chain, which is
    // what makes the refusal a statement about the arithmetic rather than about
    // the integers.
    let selected = Chain::parallel(
        vec![
            Chain::source(ImageId::supplied(0), Dtype::U16),
            Chain::source(ImageId::supplied(1), Dtype::U16),
        ],
        Box::new(ArithmeticCombine::new("maximum", Arithmetic::Maximum)),
    )
    .expect("a fan-in of two");
    assert_eq!(selected.produces(Dtype::U16).expect("a type"), Dtype::U16);
}

/// A supplied array that is not in image 0's coordinate space is refused at
/// prepare time, by name. **This is G2's residue and not this combine's**, and
/// the message says so: a second image at another extent is not something a
/// voxelwise join could have paired up.
#[test]
fn a_supplied_operand_on_another_lattice_is_refused_by_name() {
    let input = field();
    let chain = against_supplied(Arithmetic::Add, false);
    let decomposition = supplied_plan(&chain, [4, 4, 4]);
    // half the extent on one axis: the same field at another binning
    let binned: Array3<f64> = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2] / 2 + 1), 1.0);
    let message = match ArrayEnvironment::with_inputs(
        input.clone().into(),
        vec![binned.into()],
        &decomposition,
        [4, 4, 4],
    ) {
        Ok(_) => panic!("a supplied array on another lattice must be refused"),
        Err(refusal) => refusal.to_string(),
    };
    assert!(message.contains("supplied input 0"), "{message}");
    assert!(message.contains("coordinate space"), "{message}");
}

/// The cost the planner is told is the kernel's, per tap, and it really varies
/// with the kernel — a flat figure would price a 27-tap stencil and a 343-tap
/// one the same.
#[test]
fn a_convolutions_cost_follows_its_tap_count() {
    use blockflow::op::BlockOp;
    let small = ConvolveOp::new(
        "small",
        Kernel::from_radius([1, 1, 1], vec![0.5; 27]).unwrap(),
        Sense::Correlate,
        Boundary::Clamp,
    );
    let large = ConvolveOp::new(
        "large",
        Kernel::from_radius([3, 3, 3], vec![0.5; 343]).unwrap(),
        Sense::Correlate,
        Boundary::Clamp,
    );
    assert!(small.cost_per_voxel() > 0.0);
    let ratio = large.cost_per_voxel() / small.cost_per_voxel();
    assert!((ratio - 343.0 / 27.0).abs() < 1e-9, "ratio {ratio}");
}
