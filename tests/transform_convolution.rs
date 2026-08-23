// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A frequency-domain operation that is a phase.**
//
// The ops survey's G3 row says a spectrum has no element type to live in, and
// reads that as the reason `ops::fft` "declines to be an op at all". This file
// is the counter-example: `ops::convolve::TransformConvolveOp` computes a linear
// filter through the Fourier transform, is an ordinary `BlockOp` with an
// ordinary bounded reach, writes `f64` voxels, and needs no `Dtype::Complex*`
// because the spectrum never leaves the inside of one `apply`.
//
// What it pins, in the order the claims depend on each other:
//
//  1. **It computes the filter.** Against `ConvolveOp` — the direct gather, the
//     op this crate already accepts — to a rounding, in both senses and both
//     boundary conventions, on a kernel whose two sides differ on every axis so
//     that a swapped `lo`/`hi` cannot pass.
//  2. **It is decomposition-invariant to the bit**, across block extents that
//     are asserted to be genuinely distinct and to cut the volume genuinely
//     differently, against the whole-volume reference.
//  3. **The tile is load-bearing in the arithmetic.** Two tiles give the same
//     answer to a rounding and *different bits*, so (2) is a property of the
//     global tile anchoring and not of an op that ignores its tile.
//  4. **The anchor is load-bearing.** The same block computed at two different
//     anchors is a different set of tiles; an op that ignored `Anchor` would
//     pass (1) and fail (2), so this is asserted directly.
//  5. **The negative controls**, each differing from the right program by one
//     thing: the sense flipped, the boundary convention changed, the tile moved.
//     Each with its liveness partner — a symmetric kernel cannot tell the senses
//     apart, a constant field cannot tell any of them apart, and a kernel that
//     reaches one voxel cannot tell `Clamp` from `Reflect`.
//  6. **The refusals**, by name and before a block runs.
//
// No assertion here is on wall-clock time.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::convolve::TransformConvolveOp;
use blockflow::ops::{Boundary, ConvolveOp, Kernel, Sense};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

/// Prime on two axes, so a cubic block edge divides no axis and every grid below
/// has ragged blocks at its high faces. Not a multiple of the tile either, which
/// is the case the alignment slack in the halo exists for.
const VOLUME: [usize; 3] = [9, 7, 11];

/// Asymmetric on every axis and asymmetric *differently* on each, so a `lo` and
/// a `hi` that were swapped, or two axes that were transposed, both show.
const TILE: [usize; 3] = [3, 3, 3];

// -------------------------------------------------------------- fixtures --

/// A field with no symmetry, no plateau, and no axis it is constant along.
fn field() -> Array3<f64> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(a, b, c)| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let noise = (state >> 11) as f64 / (1u64 << 53) as f64;
        noise + 0.31 * a as f64 - 0.17 * b as f64 + 0.09 * c as f64
    })
}

fn constant_field() -> Array3<f64> {
    Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 0.75)
}

/// Sides that differ on every axis, and weights that are not symmetric under any
/// reflection.
fn lopsided() -> Kernel {
    let lo = [0usize, 1, 2];
    let hi = [2usize, 1, 0];
    let size = (lo[0] + hi[0] + 1) * (lo[1] + hi[1] + 1) * (lo[2] + hi[2] + 1);
    let weights: Vec<f64> = (0..size)
        .map(|which| (which as f64 + 1.0) / size as f64 - 0.4)
        .collect();
    Kernel::from_sides(lo, hi, weights).expect("a kernel")
}

/// The same box, with weights symmetric under negation of every offset. The
/// liveness partner for the sense control: this one cannot tell the two apart.
fn symmetric() -> Kernel {
    let radius = [1usize, 1, 1];
    let size = 27;
    let weights: Vec<f64> = (0..size)
        .map(|which| {
            let mirrored = size - 1 - which;
            (which.min(mirrored) as f64 + 1.0) / size as f64
        })
        .collect();
    Kernel::from_radius(radius, weights).expect("a kernel")
}

/// A kernel reaching exactly one voxel: the liveness partner for the boundary
/// control, which it cannot tell apart, because at that distance `Clamp` and
/// `Reflect` are the same function.
fn narrow() -> Kernel {
    let weights: Vec<f64> = (0..27).map(|which| (which as f64 + 1.0) / 27.0).collect();
    Kernel::from_radius([1, 1, 1], weights).expect("a kernel")
}

fn transform_op(
    kernel: Kernel,
    sense: Sense,
    boundary: Boundary,
    tile: [usize; 3],
) -> TransformConvolveOp {
    TransformConvolveOp::new("transform", kernel, sense, boundary, tile).expect("an op")
}

// --------------------------------------------------------------- harness --

fn workflow(chain: Chain) -> Workflow {
    Workflow::new(chain, VOLUME, Dtype::F64)
}

/// One phase holding the op, at a stated block extent, with the chain's **own**
/// reach — so an op that under-declared would be short of a halo and the
/// comparison against the whole-volume reference would say so.
///
/// **`reach_spec` and not `reach3`, and that is load-bearing here.** `reach3`
/// flattens to a symmetric triple, which for this op is the *unaligned* worst
/// case on every lattice; the full [`blockflow::reach::Reach`] carries
/// `AxisReach::Aligned`, so `PhaseDecomposition::derive` resolves it against the
/// grid and a tile-aligned edge is handed the smaller halo. Half the block
/// extents below are tile-aligned on at least one axis, so the invariance test
/// runs over both resolutions and byte-identity has to hold across the two.
fn plan(workflow: &Workflow, block: [usize; 3]) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let reach = workflow.chain.reach_spec(VOLUME).expect("a reach");
    let grid = BlockGrid::new(VOLUME, block).expect("a grid");
    let phase = PhaseDecomposition::derive(
        (0..slots.len()).collect(),
        names,
        reach.clone(),
        reach,
        grid,
    );
    Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: workflow.chain.reach3(&VOLUME),
    }
}

/// Block extents that between them cut every axis, divide none of them evenly,
/// and are never a multiple of [`TILE`] on all three axes at once.
fn blocks() -> Vec<[usize; 3]> {
    vec![
        VOLUME,
        [4, 4, 4],
        [2, 3, 2],
        [5, 5, 5],
        [3, 7, 5],
        [9, 2, 5],
        [9, 7, 2],
        [2, 2, 2],
    ]
}

fn run(op: TransformConvolveOp, input: &Array3<f64>) -> Vec<(String, Array3<f64>)> {
    let workflow = workflow(Chain::op(op));
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

/// The op applied once, over the whole array, with no plan and no blocking.
fn resident(op: &dyn BlockOp, input: &Array3<f64>) -> Array3<f64> {
    let source: Voxels = input.clone().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).expect("a buffer");
    op.apply(&source, &mut out, &Anchor::whole(VOLUME))
        .expect("the whole-volume reference must run");
    out.view::<f64>().unwrap().to_owned()
}

fn moved(left: &Array3<f64>, right: &Array3<f64>) -> usize {
    left.iter()
        .zip(right.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

fn worst(left: &Array3<f64>, right: &Array3<f64>) -> f64 {
    let mut seen = 0.0f64;
    for (a, b) in left.iter().zip(right.iter()) {
        let apart = (a - b).abs();
        if apart.total_cmp(&seen).is_gt() {
            seen = apart;
        }
    }
    seen
}

// ------------------------------------------------- 1. it is the filter --

/// **The acceptance test.** The transform path computes the same filter the
/// direct gather does, in both senses and both boundary conventions.
#[test]
fn the_transform_path_is_the_direct_convolution_to_a_rounding() {
    let input = field();
    let scale = input.iter().fold(0.0f64, |seen, value| {
        if value.abs().total_cmp(&seen).is_gt() {
            value.abs()
        } else {
            seen
        }
    });
    for sense in [Sense::Correlate, Sense::Convolve] {
        for boundary in [Boundary::Clamp, Boundary::Reflect] {
            let direct = ConvolveOp::new("direct", lopsided(), sense, boundary);
            let transform = transform_op(lopsided(), sense, boundary, TILE);
            let expected = resident(&direct, &input);
            let got = resident(&transform, &input);
            let apart = worst(&expected, &got);
            println!("{sense:?} {boundary:?}: worst deviation {apart:e}");
            assert!(
                apart < 1.0e-12 * scale,
                "{sense:?}/{boundary:?}: the transform path deviates from the direct one by \
                 {apart:e}, which is not a rounding"
            );
            // **Liveness for the tolerance.** A bound this loose would also be
            // met by an op that answered a *different* filter if the two filters
            // happened to be close; the wrong sense is the nearest such filter
            // and it must fail by orders.
            let other = match sense {
                Sense::Correlate => Sense::Convolve,
                Sense::Convolve => Sense::Correlate,
            };
            let wrong = resident(
                &ConvolveOp::new("wrong", lopsided(), other, boundary),
                &input,
            );
            let far = worst(&wrong, &got);
            assert!(
                far > 1.0e-6 * scale,
                "the wrong sense is within {far:e} of the right answer, so the tolerance \
                 above is not evidence of anything"
            );
        }
    }
}

// --------------------------------------- 2, 3 and 4. the decomposition --

/// **The bar.** Byte-identical at every block extent, against the whole-volume
/// reference — and the extents are asserted to be genuinely distinct
/// decompositions rather than eight names for one.
#[test]
fn the_transform_path_is_decomposition_invariant() {
    let input = field();
    let reference = resident(
        &transform_op(lopsided(), Sense::Correlate, Boundary::Clamp, TILE),
        &input,
    );

    let counts: Vec<usize> = blocks()
        .iter()
        .map(|block| BlockGrid::new(VOLUME, *block).expect("a grid").n_blocks())
        .collect();
    assert!(
        counts.iter().filter(|&&count| count == 1).count() == 1,
        "exactly one of these extents may be the whole volume; the block counts are {counts:?}"
    );
    assert!(
        counts.iter().copied().max().unwrap() >= 100,
        "no extent here cuts the volume finely; the block counts are {counts:?}"
    );
    let mut distinct = counts.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() >= 5,
        "the extents collapse to {distinct:?} distinct decompositions, which is too few for \
         this to be measuring invariance"
    );

    for (label, got) in run(
        transform_op(lopsided(), Sense::Correlate, Boundary::Clamp, TILE),
        &input,
    ) {
        assert_eq!(
            moved(&reference, &got),
            0,
            "block {label} disagrees with the whole-volume reference in {} voxels",
            moved(&reference, &got)
        );
    }
}

/// **The tile is in the arithmetic.** If it were not — if the op transformed
/// whatever buffer it was handed — the test above would be measuring nothing,
/// because every grid would agree with every other for the wrong reason. Two
/// tiles must therefore agree to a rounding and differ in bits.
#[test]
fn changing_the_tile_changes_the_bits_and_not_the_answer() {
    let input = field();
    let three = resident(
        &transform_op(lopsided(), Sense::Correlate, Boundary::Clamp, [3, 3, 3]),
        &input,
    );
    let four = resident(
        &transform_op(lopsided(), Sense::Correlate, Boundary::Clamp, [4, 5, 4]),
        &input,
    );
    let apart = worst(&three, &four);
    let scale = input.iter().fold(0.0f64, |seen, value| {
        if value.abs().total_cmp(&seen).is_gt() {
            value.abs()
        } else {
            seen
        }
    });
    assert!(
        apart < 1.0e-12 * scale,
        "two tiles must compute the same filter, and these differ by {apart:e}"
    );
    let bits = moved(&three, &four);
    assert!(
        bits > 0,
        "two different tiles produced bit-identical output, so the tile is not in the \
         arithmetic and the invariance test above proves nothing about anchoring"
    );
    println!(
        "tiles [3,3,3] and [4,5,4] differ in {bits} voxels of {}",
        three.len()
    );
}

/// **The anchor is in the arithmetic.** The same buffer computed as if it sat at
/// two different places in the volume must give different answers; an op that
/// ignored `Anchor` would pass the agreement test and silently fail invariance
/// on any volume whose blocks were not tile-aligned.
#[test]
fn moving_the_anchor_moves_the_tiles() {
    let input = field();
    let op = transform_op(lopsided(), Sense::Correlate, Boundary::Clamp, TILE);
    let source: Voxels = input.clone().into();
    let mut at_origin = Voxels::zeros(Dtype::F64, VOLUME).expect("a buffer");
    let mut shifted = Voxels::zeros(Dtype::F64, VOLUME).expect("a buffer");
    op.apply(&source, &mut at_origin, &Anchor::whole(VOLUME))
        .expect("a block");
    // A volume twice as long on axis 0 that this buffer sits one voxel into: the
    // tile boundaries now fall in different places relative to the data.
    op.apply(
        &source,
        &mut shifted,
        &Anchor::new([1, 0, 0], [2 * VOLUME[0], VOLUME[1], VOLUME[2]]),
    )
    .expect("a block");
    let left = at_origin.view::<f64>().unwrap().to_owned();
    let right = shifted.view::<f64>().unwrap().to_owned();
    let bits = moved(&left, &right);
    assert!(
        bits > 0,
        "the same buffer at two anchors produced identical output, so `Anchor` is ignored \
         and the tile grid is not global"
    );
    println!("one voxel of anchor moved {bits} voxels of {}", left.len());
}

// ------------------------------------------------- 5. negative controls --

#[test]
fn flipping_the_sense_moves_the_answer() {
    let input = field();
    let correlate = resident(
        &transform_op(lopsided(), Sense::Correlate, Boundary::Clamp, TILE),
        &input,
    );
    let convolve = resident(
        &transform_op(lopsided(), Sense::Convolve, Boundary::Clamp, TILE),
        &input,
    );
    let bits = moved(&correlate, &convolve);
    assert!(
        bits > correlate.len() / 2,
        "flipping the kernel moved only {bits} voxels of {}",
        correlate.len()
    );
}

#[test]
fn a_symmetric_kernel_cannot_distinguish_the_two_senses() {
    // The liveness partner for the control above: on this fixture the control
    // would pass for no reason, which is why the control does not use it.
    let input = field();
    let correlate = resident(
        &transform_op(symmetric(), Sense::Correlate, Boundary::Clamp, TILE),
        &input,
    );
    let convolve = resident(
        &transform_op(symmetric(), Sense::Convolve, Boundary::Clamp, TILE),
        &input,
    );
    assert_eq!(
        moved(&correlate, &convolve),
        0,
        "a kernel symmetric under negation must give the same answer in both senses"
    );
}

#[test]
fn a_constant_field_cannot_distinguish_anything() {
    let input = constant_field();
    let one = resident(
        &transform_op(lopsided(), Sense::Correlate, Boundary::Clamp, TILE),
        &input,
    );
    for other in [
        resident(
            &transform_op(lopsided(), Sense::Convolve, Boundary::Clamp, TILE),
            &input,
        ),
        resident(
            &transform_op(lopsided(), Sense::Correlate, Boundary::Reflect, TILE),
            &input,
        ),
    ] {
        let apart = worst(&one, &other);
        assert!(
            apart < 1.0e-14,
            "a constant field distinguished two programs by {apart:e}, so it is not the \
             blind fixture this asserts it is"
        );
    }
}

#[test]
fn changing_the_boundary_convention_moves_the_faces() {
    let input = field();
    let clamp = resident(
        &transform_op(lopsided(), Sense::Correlate, Boundary::Clamp, TILE),
        &input,
    );
    let reflect = resident(
        &transform_op(lopsided(), Sense::Correlate, Boundary::Reflect, TILE),
        &input,
    );
    let bits = moved(&clamp, &reflect);
    assert!(bits > 0, "the boundary convention changed nothing");
    // Only voxels within the kernel's reach of a face may move: the two
    // conventions are the same function everywhere the window is inside the
    // volume.
    for a in 0..VOLUME[0] {
        for b in 0..VOLUME[1] {
            for c in 0..VOLUME[2] {
                let interior = a >= 2
                    && a + 2 < VOLUME[0]
                    && b >= 1
                    && b + 1 < VOLUME[1]
                    && c >= 2
                    && c + 2 < VOLUME[2];
                if interior {
                    let apart = (clamp[[a, b, c]] - reflect[[a, b, c]]).abs();
                    assert!(
                        apart < 1.0e-12,
                        "an interior voxel {a},{b},{c} moved by {apart:e} when only the \
                         boundary convention changed"
                    );
                }
            }
        }
    }
}

/// The liveness partner for the control above — **and it does not hold to the
/// bit here, where it does for `ConvolveOp`, and that difference is a property
/// of a transform worth having written down.**
///
/// At a distance of one, `Clamp` and `Reflect` resolve every index the *answer*
/// depends on to the same sample, so the two conventions are the same function.
/// A direct gather therefore produces identical bits. This op does not: a tile's
/// transform is a sum over the whole window, and the window contains samples
/// that no output of that tile depends on — the ones feeding positions past the
/// volume's face, which are computed and thrown away. Those samples *do* differ
/// between the two conventions, and a Fourier coefficient is a sum over every
/// element of its input, so they perturb every output of that tile in the last
/// place.
///
/// **This is not a decomposition hazard**: the window is a function of the
/// global tile and of the volume's faces, both of which every lattice agrees on,
/// which is why the invariance test above is byte-exact. It is a statement about
/// what "the same function" means once the arithmetic stops being local.
#[test]
fn a_kernel_reaching_one_voxel_cannot_distinguish_the_two_boundary_conventions() {
    let input = field();
    let clamp = resident(
        &transform_op(narrow(), Sense::Correlate, Boundary::Clamp, TILE),
        &input,
    );
    let reflect = resident(
        &transform_op(narrow(), Sense::Correlate, Boundary::Reflect, TILE),
        &input,
    );
    let apart = worst(&clamp, &reflect);
    let scale = clamp.iter().fold(0.0f64, |seen, value| {
        if value.abs().total_cmp(&seen).is_gt() {
            value.abs()
        } else {
            seen
        }
    });
    assert!(
        apart < 1.0e-14 * scale,
        "at a reach of one the two conventions are the same function and these differ by \
         {apart:e} against an answer of scale {scale:e}"
    );
    // The same fixture through the direct gather, which *is* bit-identical —
    // stated here so the clause above is a measured difference between the two
    // paths and not an excuse for a loose bound.
    let direct_clamp = resident(
        &ConvolveOp::new("direct", narrow(), Sense::Correlate, Boundary::Clamp),
        &input,
    );
    let direct_reflect = resident(
        &ConvolveOp::new("direct", narrow(), Sense::Correlate, Boundary::Reflect),
        &input,
    );
    assert_eq!(
        moved(&direct_clamp, &direct_reflect),
        0,
        "the direct gather must be bit-identical under the two conventions at a reach of one"
    );
    println!(
        "reach one: the transform path moves {} voxels in the last place, the direct gather 0",
        moved(&clamp, &reflect)
    );
}

// ------------------------------------------------------- 6. the refusals --

#[test]
fn an_empty_tile_is_refused_by_name() {
    for tile in [[0, 3, 3], [3, 0, 3], [3, 3, 0]] {
        let message = TransformConvolveOp::new(
            "transform",
            lopsided(),
            Sense::Correlate,
            Boundary::Clamp,
            tile,
        )
        .unwrap_err()
        .to_string();
        assert!(
            message.contains("non-empty"),
            "an empty tile must be refused by name, got {message}"
        );
    }
    assert!(TransformConvolveOp::new(
        "transform",
        lopsided(),
        Sense::Correlate,
        Boundary::Clamp,
        [1, 1, 1]
    )
    .is_ok());
}

#[test]
fn what_it_accepts_is_a_list_and_half_precision_is_not_on_it() {
    let op = transform_op(lopsided(), Sense::Correlate, Boundary::Clamp, TILE);
    for dtype in [
        Dtype::Bool,
        Dtype::U8,
        Dtype::U16,
        Dtype::U32,
        Dtype::U64,
        Dtype::I8,
        Dtype::I16,
        Dtype::I32,
        Dtype::I64,
        Dtype::F32,
        Dtype::F64,
    ] {
        assert!(
            op.accepts(dtype),
            "{dtype:?} holds a real number and must be accepted"
        );
    }
    assert!(!op.accepts(Dtype::F16), "no buffer holds half-precision");
    assert_eq!(op.produces(Dtype::U8), Dtype::F64);
}

#[test]
fn the_declared_halo_is_the_kernel_plus_the_tiles_alignment_slack() {
    let op = transform_op(lopsided(), Sense::Correlate, Boundary::Clamp, TILE);
    let spec = op.reach_spec(VOLUME);
    // `lopsided` reaches `lo = [0, 1, 2]`, `hi = [2, 1, 0]`, and the tile is
    // three on every axis, so the slack is two per side.
    for (axis, expected) in [(0usize, (2usize, 4usize)), (1, (3, 3)), (2, (4, 2))] {
        assert_eq!(
            spec.axis(axis).bound(VOLUME[axis]),
            expected,
            "axis {axis}'s declared halo"
        );
    }
    // The direct op's halo is the kernel's alone; the difference between the two
    // is exactly what this op's header calls the price of a global tile grid.
    let direct = ConvolveOp::new("direct", lopsided(), Sense::Correlate, Boundary::Clamp);
    for (axis, expected) in [(0usize, (0usize, 2usize)), (1, (1, 1)), (2, (2, 0))] {
        assert_eq!(
            direct.reach_spec(VOLUME).axis(axis).bound(VOLUME[axis]),
            expected
        );
    }
}

#[test]
fn there_is_no_constant_short_circuit_and_the_uniform_answer_is_still_right() {
    let op = transform_op(lopsided(), Sense::Correlate, Boundary::Clamp, TILE);
    assert!(
        op.constant_maps_to(0.75).is_none(),
        "the transform path may not declare a constant it does not compute to the bit"
    );
    // And the declaration's absence is not hiding a wrong answer: a uniform
    // block really does map to the weighted sum, to a rounding.
    let uniform = resident(&op, &constant_field());
    let total: f64 = lopsided().weights().iter().sum::<f64>() * 0.75;
    let apart = worst(&uniform, &Array3::from_elem(uniform.dim(), total));
    assert!(
        apart < 1.0e-14,
        "a uniform field maps to {total} and the op deviates by {apart:e}"
    );
    // Liveness: a kernel summing to zero would make the assertion above pass on
    // any op that wrote zeros.
    assert!(
        total.abs() > 0.1,
        "the fixture's weighted sum is {total}, too near zero to be evidence"
    );
}

/// **The declared halo is load-bearing**, and this is the assertion that says so
/// rather than the one that assumes it.
///
/// The invariance test above would pass just as well if the halo were larger
/// than the op needs — a plan that over-fetches computes the right answer and
/// says nothing about the declaration. So this builds a plan whose halo is the
/// *kernel's* alone, which is what `ConvolveOp` would need and what a reader who
/// missed the alignment slack would write, and asserts that the framework
/// notices. Whichever way it notices — a refusal at plan time or a different
/// answer at run time — is recorded here, because the two are different facts
/// about where the guard lives.
///
/// **Measured: it refuses**, and it refuses for the right reason and by name —
/// "valid regions do not tile the volume exactly (6 of 693 voxels covered) ...
/// 18 block(s) lost part of their core ... A halo below the phase reach is the
/// usual cause." So the alignment slack is enforced by the tiling check that
/// already existed, and a caller who writes the kernel's halo gets an error
/// rather than a plausible wrong volume. That is the branch this test asserts;
/// the other branch is kept because a *future* change that made the guard
/// permissive would fall into it, and a test that had assumed the refusal would
/// then be asserting the wrong thing quietly.
#[test]
fn a_plan_with_only_the_kernels_halo_does_not_quietly_produce_the_right_answer() {
    let input = field();
    let op = transform_op(lopsided(), Sense::Correlate, Boundary::Clamp, TILE);
    let reference = resident(&op, &input);

    let workflow = workflow(Chain::op(transform_op(
        lopsided(),
        Sense::Correlate,
        Boundary::Clamp,
        TILE,
    )));
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let reach = workflow.chain.reach3(&VOLUME);
    // The kernel's own two sides, with the tile's alignment slack left out.
    let short =
        ConvolveOp::new("direct", lopsided(), Sense::Correlate, Boundary::Clamp).reach_spec(VOLUME);
    let block = [4, 4, 4];
    let grid = BlockGrid::new(VOLUME, block).expect("a grid");
    let phase = PhaseDecomposition::derive(
        (0..slots.len()).collect(),
        names,
        reach,
        short.clone(),
        grid,
    );
    let decomposition = Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: reach,
    };
    let env = ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4])
        .expect("an environment");
    match execute("filter", &workflow, &decomposition, &Hints::default(), &env) {
        Err(error) => {
            println!("an under-halo'd plan is refused at run time: {error}");
        }
        Ok(_) => {
            let got = env.output().view::<f64>().unwrap().to_owned();
            let bits = moved(&reference, &got);
            assert!(
                bits > 0,
                "a plan carrying only the kernel's halo produced the right answer to the bit, \
                 which would mean the tile's alignment slack is not needed and this op \
                 over-declares its reach"
            );
            println!("an under-halo'd plan runs and moves {bits} voxels");
        }
    }
    // Liveness: the two halos really are different, or the case above is not
    // the case it says it is.
    let declared = op.reach_spec(VOLUME);
    assert_ne!(
        short.axis(0).bound(VOLUME[0]),
        declared.axis(0).bound(VOLUME[0]),
        "the short halo equals the declared one, so nothing was tested"
    );
}

// ------------------------------- 7. what the tile's alignment slack costs --

/// **The price of the missing stride constraint, in read amplification.**
///
/// The op's halo is the kernel's two sides **plus `tile - 1` per side**, because
/// a block cannot know where its tiles begin. That slack is *pure waste* on any
/// lattice whose block edge is a whole number of tiles — and `BlockGrid::cores`
/// builds `start = index * block`, so a block edge that is a multiple of the
/// tile makes **every** block start tile-aligned and the true halo exactly the
/// kernel's. The planner's own candidate ladder is powers of two; a
/// power-of-two tile therefore lands on most of it. The waste is not a corner
/// case, it is the ordinary case, and it is not expressible away today.
///
/// This prints the two amplifications side by side over the crate's own ladders.
///
/// **It is the optimistic half of the cost.** Amplification is what an axis pays
/// when it survives `decomposition::cuttable_axes` at all; that function drops an
/// axis whose `edge + lo + hi` is not less than the extent, so on a volume that
/// is not large against the slack the axis is dropped instead and the phase
/// becomes one block. `1024^3` is large enough that every row below is a real
/// amplification rather than a degeneration;
/// `a_planner_prices_and_runs_the_aligned_lattice_and_gets_the_same_volume`
/// measures the other regime on `96^3`.
#[test]
#[ignore = "a measurement, not an assertion"]
fn what_the_alignment_slack_costs() {
    println!("{}", slack_report());
}

/// The amplification a lattice of `edge` pays under `halo`, using the crate's
/// own grid arithmetic rather than a formula written here.
fn amplification(volume: [usize; 3], edge: usize, halo: &blockflow::reach::Reach) -> Option<f64> {
    let grid = BlockGrid::along(volume, &[0, 1, 2], edge).ok()?;
    Some(
        grid.mean_read_voxels(halo) * grid.n_blocks() as f64
            / volume.iter().product::<usize>() as f64,
    )
}

/// `(declared, aligned)` — what ships, and what a tile-aligned lattice would
/// need if the op could demand one.
fn halos(tile: usize, radius: usize) -> (blockflow::reach::Reach, blockflow::reach::Reach) {
    use blockflow::reach::Reach;
    let slack = tile - 1 + radius;
    (
        Reach::symmetric([slack, slack, slack]),
        Reach::symmetric([radius, radius, radius]),
    )
}

fn slack_report() -> String {
    let volume = [1024usize, 1024, 1024];
    let mut out = String::from(
        "volume 1024^3, all three axes cuttable\n\
         tile  radius  edge  aligned?  declared halo  read x (declared)  read x (aligned)  ratio\n",
    );
    for (tile, radius) in [(32usize, 4usize), (32, 8), (16, 4), (64, 4)] {
        let (declared, aligned) = halos(tile, radius);
        for edge in [16usize, 24, 32, 48, 64, 96, 128, 256, 512] {
            let (Some(a), Some(b)) = (
                amplification(volume, edge, &declared),
                amplification(volume, edge, &aligned),
            ) else {
                continue;
            };
            out.push_str(&format!(
                "{tile:4}  {radius:6}  {edge:4}  {:8}  {:13}  {a:17.3}  {b:16.3}  {:5.2}\n",
                if edge % tile == 0 { "yes" } else { "no" },
                tile - 1 + radius,
                a / b
            ));
        }
    }
    out
}

/// **The slack is real, it is large, and it is paid exactly where the planner
/// cuts.** The assertion, so that the measurement above is pinned by something
/// that fails if the cost ever stops being there.
#[test]
fn the_alignment_slack_is_paid_on_lattices_that_do_not_need_it() {
    let volume = [1024usize, 1024, 1024];
    let tile = 32usize;
    let radius = 4usize;
    let (declared, aligned) = halos(tile, radius);
    // The crate's own coarse ladder. Three of its four rungs are multiples of a
    // 32-voxel tile, so on three of four the declared slack buys nothing at all.
    let ladder = [16usize, 32, 64, 128];
    let multiples: Vec<usize> = ladder.iter().copied().filter(|e| e % tile == 0).collect();
    assert_eq!(
        multiples,
        vec![32, 64, 128],
        "the coarse ladder's rungs that are whole tiles"
    );
    for edge in multiples {
        let declared_x = amplification(volume, edge, &declared).expect("a grid");
        let aligned_x = amplification(volume, edge, &aligned).expect("a grid");
        assert!(
            declared_x > aligned_x * 1.5,
            "at edge {edge} the declared halo reads {declared_x:.3}x and an aligned lattice \
             would read {aligned_x:.3}x — less than a 1.5x gap, so the missing constraint is \
             not worth what this file claims for it"
        );
    }
    // **Liveness.** The gap closes as the block grows, so an assertion made at a
    // large enough edge would pass for no reason. At 1024 the volume is one
    // block and the two halos are the same fetch — asserted, so that the choice
    // of ladder above is a choice and not an accident.
    let whole_declared = amplification(volume, 1024, &declared).expect("a grid");
    let whole_aligned = amplification(volume, 1024, &aligned).expect("a grid");
    assert!(
        (whole_declared - whole_aligned).abs() < 1.0e-9,
        "at one block the two halos must cost the same ({whole_declared} against \
         {whole_aligned}); if they differ, this test is measuring something else"
    );
}

/// **The discount reaches the plan, and only where the lattice earns it.**
///
/// The pair is the point: the *same* grid is planned twice, once with the op's
/// own [`blockflow::reach::Reach`] — which carries `AxisReach::Aligned` — and
/// once with **the same reach flattened to its own worst case**, which is what
/// every question asked without a lattice gets and is exactly what this op
/// declared before the variant existed. On a tile-aligned edge the two must
/// differ and the aligned one must fetch less; on an unaligned edge they must be
/// **identical**, which is the liveness partner and is what makes the first half
/// evidence of a *discount* rather than of two different reaches.
///
/// The baseline is derived from the op rather than written here, because a
/// hand-written triple is a second statement of the same quantity — and the
/// first attempt at this test used `reach3`, the *symmetric* fold, which is
/// wider than the worst case on an asymmetric kernel and made the unaligned
/// halves differ for a reason that had nothing to do with alignment.
#[test]
fn a_tile_aligned_lattice_is_planned_with_the_smaller_halo() {
    use blockflow::reach::Reach;

    let workflow = workflow(Chain::op(transform_op(
        lopsided(),
        Sense::Correlate,
        Boundary::Clamp,
        TILE,
    )));
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let stated = workflow.chain.reach_spec(VOLUME).expect("a reach");
    let worst = Reach::asymmetric([
        stated.axis(0).bound(VOLUME[0]),
        stated.axis(1).bound(VOLUME[1]),
        stated.axis(2).bound(VOLUME[2]),
    ]);
    assert_ne!(
        worst, stated,
        "the flattened worst case must differ from the stated reach, or this test compares a \
         thing with itself"
    );

    let fetched = |block: [usize; 3], halo: &Reach| -> usize {
        let grid = BlockGrid::new(VOLUME, block).expect("a grid");
        PhaseDecomposition::derive(
            (0..slots.len()).collect(),
            names.clone(),
            halo.clone(),
            halo.clone(),
            grid,
        )
        .blocks
        .iter()
        .map(|geometry| geometry.read.shape.iter().product::<usize>())
        .sum()
    };

    // `TILE` is three on every axis, so this edge is a whole number of tiles on
    // all three and the discount is available on all three.
    let aligned_edge = [3usize, 3, 3];
    let with_discount = fetched(aligned_edge, &stated);
    let without = fetched(aligned_edge, &worst);
    assert!(
        with_discount < without,
        "on a tile-aligned lattice the stated reach must fetch less than its own worst case: \
         {with_discount} against {without}"
    );
    println!(
        "aligned edge {aligned_edge:?}: {with_discount} voxels fetched against {without}, {:.2}x",
        without as f64 / with_discount as f64
    );

    // **Liveness.** Four is not a multiple of three on any axis, so there is
    // nothing to discount and the two must agree exactly. Without this the test
    // above would pass for an `Aligned` that simply reported smaller numbers
    // than its worst case everywhere, which would be an under-halo and not a
    // discount.
    let unaligned_edge = [4usize, 4, 4];
    assert_eq!(
        fetched(unaligned_edge, &stated),
        fetched(unaligned_edge, &worst),
        "on a lattice that earns no discount the stated reach must be its own worst case"
    );
    // And the mixed case, which is the one a real volume gives: three divides
    // the first axis and not the other two, so exactly one axis is discounted.
    let mixed_edge = [3usize, 4, 4];
    assert!(
        fetched(mixed_edge, &stated) < fetched(mixed_edge, &worst),
        "one aligned axis is still a discount"
    );
}

/// **The discount survives a real planner — and what it buys there is not a
/// smaller halo, it is a phase that can be blocked at all.**
///
/// Everything above builds its own `PhaseDecomposition`. This one hands the
/// workflow to `Greedy` and lets it plan, and the first attempt at it asserted
/// the wrong thing: that the aligned lattice *fetches less in total*. It does
/// not, and cannot — total fetch is minimised by a single block, which reads the
/// volume exactly once. What the measurement actually showed is sharper.
///
/// `decomposition::cuttable_axes` drops an axis whose `edge + lo + hi` is not
/// less than the extent, and it runs **before** anything is priced. On `96^3`
/// with a 32-voxel tile the unresolved halo is 33 a side, so `32 + 66` exceeds
/// 96 on every axis, every axis is dropped, and the phase degenerates to **one
/// block reading the whole volume** — the exact cost `docs/design/barriers.md`
/// exists to remove, arrived at from the other end. With the reach resolved
/// against the candidate edge the halo is 2 a side, `32 + 4` is comfortably
/// under 96, and the phase cuts into blocks.
///
/// So the assertions are: the same volume byte for byte on both lattices, and
/// the three `cuttable_axes` answers that isolate *alignment* as the cause.
#[test]
fn a_planner_prices_and_runs_the_aligned_lattice_and_gets_the_same_volume() {
    use blockflow::decomposition::{cuttable_axes, Constraints, CostModel};
    use blockflow::reach::Reach;
    use blockflow::strategy::{Greedy, Strategy};

    const SIDE: [usize; 3] = [96, 96, 96];
    let tile = [32usize, 32, 32];
    let weights: Vec<f64> = (0..125)
        .map(|which| (which as f64 + 1.0) / 125.0 - 0.3)
        .collect();
    let kernel = Kernel::from_radius([2, 2, 2], weights).expect("a kernel");

    let mut state = 0x1234_5678_9ABC_DEF1u64;
    let input = Array3::from_shape_fn((SIDE[0], SIDE[1], SIDE[2]), |(a, b, c)| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 + 0.01 * (a + b + c) as f64
    });

    let build = || {
        Chain::op(
            TransformConvolveOp::new(
                "transform",
                kernel.clone(),
                Sense::Correlate,
                Boundary::Clamp,
                tile,
            )
            .expect("an op"),
        )
    };

    // ---- the three answers that isolate alignment as the cause ----
    let stated = build().reach_spec(SIDE).expect("a reach");
    let worst = Reach::asymmetric([
        stated.axis(0).bound(SIDE[0]),
        stated.axis(1).bound(SIDE[1]),
        stated.axis(2).bound(SIDE[2]),
    ]);
    let axes = [0usize, 1, 2];
    assert_eq!(
        cuttable_axes(&axes, &stated, SIDE, 32).len(),
        3,
        "32 is a whole number of tiles, so every axis is cuttable"
    );
    assert!(
        cuttable_axes(&axes, &worst, SIDE, 32).is_empty(),
        "without the discount the same edge leaves no axis cuttable, and the phase becomes one \
         block reading the whole volume"
    );
    // **Liveness, and it is the assertion that makes this about alignment.** The
    // stated reach is not simply smaller everywhere: at an edge the tile does
    // not divide it is its own worst case, and the axes go away again.
    assert!(
        cuttable_axes(&axes, &stated, SIDE, 48).is_empty(),
        "at an edge that is not a whole number of tiles the stated reach must be its own worst \
         case; if it is cuttable here the discount is being taken where it is not earned"
    );

    // ---- and the answer does not move, on either lattice ----
    let reference = {
        let source: Voxels = input.clone().into();
        let mut out = Voxels::zeros(Dtype::F64, SIDE).expect("a buffer");
        build()
            .apply(&source, &mut out, &Anchor::whole(SIDE))
            .expect("the whole-volume reference must run");
        out.view::<f64>().unwrap().to_owned()
    };
    let run_with = |edge: usize| -> (Array3<f64>, usize) {
        let workflow = Workflow::new(build(), SIDE, Dtype::F64);
        let constraints = Constraints {
            budget_bytes: None,
            expected_concurrency: 1,
            model: CostModel::default(),
            block_candidates: vec![edge],
            split_axes: axes.to_vec(),
            ..Default::default()
        };
        let decomposition = Greedy::default()
            .decompose(&workflow, &constraints)
            .expect("a plan");
        decomposition.check().expect("the plan must check");
        let blocks = decomposition.phases[0].blocks.len();
        let env = ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [8, 8, 8])
            .expect("an environment");
        execute("filter", &workflow, &decomposition, &Hints::default(), &env).expect("a run");
        (env.output().view::<f64>().unwrap().to_owned(), blocks)
    };

    let (aligned, aligned_blocks) = run_with(32);
    let (unaligned, unaligned_blocks) = run_with(48);
    assert!(
        aligned_blocks > 1,
        "the aligned lattice must actually cut the volume, got {aligned_blocks} block(s)"
    );
    assert_eq!(
        unaligned_blocks, 1,
        "the unaligned lattice degenerates to one block, which is the cost this measures"
    );
    assert_eq!(
        moved(&reference, &aligned),
        0,
        "the aligned plan disagrees with the whole-volume reference"
    );
    assert_eq!(
        moved(&reference, &unaligned),
        0,
        "the unaligned plan disagrees with the whole-volume reference"
    );
    println!(
        "planner on 96^3, tile 32: edge 32 gives {aligned_blocks} blocks, edge 48 gives \
         {unaligned_blocks}"
    );
}

/// **Fusing the op with one that reaches nothing must not cost the discount**,
/// and this is the assertion that says so because the first version of the
/// mechanism failed it.
///
/// A reach fold that flattened an `AxisReach::Aligned` lost the whole discount
/// to the most ordinary fusion there is — a transform convolution followed by a
/// voxelwise map — because a phase's reach is its ops' reaches added, and adding
/// a reach of nothing was not the identity. Measured on `96^3` at candidate edge
/// 32: **27 blocks alone against one when fused**, which is the entire feature
/// and, worse, is a phase reading the whole volume per block.
///
/// So: the fused chain must plan into the same number of blocks as the op alone,
/// and must produce the same volume the fused chain produces resident.
#[test]
fn fusing_with_an_op_that_reaches_nothing_keeps_the_discount() {
    use blockflow::decomposition::{Constraints, CostModel};
    use blockflow::ops::VoxelwiseMapOp;
    use blockflow::strategy::{Greedy, Strategy};

    const SIDE: [usize; 3] = [96, 96, 96];
    let tile = [32usize, 32, 32];
    let weights: Vec<f64> = (0..125)
        .map(|which| (which as f64 + 1.0) / 125.0 - 0.3)
        .collect();
    let kernel = Kernel::from_radius([2, 2, 2], weights).expect("a kernel");

    let mut state = 0x0FED_CBA9_8765_4321u64;
    let input = Array3::from_shape_fn((SIDE[0], SIDE[1], SIDE[2]), |(a, b, c)| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 + 0.02 * (a + b + c) as f64
    });

    fn doubled(value: f64) -> f64 {
        value * 2.0
    }
    let transform = || {
        TransformConvolveOp::new(
            "transform",
            Kernel::from_radius(
                [2, 2, 2],
                (0..125)
                    .map(|which| (which as f64 + 1.0) / 125.0 - 0.3)
                    .collect(),
            )
            .expect("a kernel"),
            Sense::Correlate,
            Boundary::Clamp,
            tile,
        )
        .expect("an op")
    };
    let _ = &kernel;

    let plan_blocks = |chain: Chain| -> (usize, Array3<f64>) {
        let workflow = Workflow::new(chain, SIDE, Dtype::F64);
        let constraints = Constraints {
            budget_bytes: None,
            expected_concurrency: 1,
            model: CostModel::default(),
            block_candidates: vec![32],
            split_axes: vec![0, 1, 2],
            ..Default::default()
        };
        let decomposition = Greedy::default()
            .decompose(&workflow, &constraints)
            .expect("a plan");
        decomposition.check().expect("the plan must check");
        let blocks: usize = decomposition.phases.iter().map(|p| p.blocks.len()).sum();
        let env = ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [8, 8, 8])
            .expect("an environment");
        execute("filter", &workflow, &decomposition, &Hints::default(), &env).expect("a run");
        (blocks, env.output().view::<f64>().unwrap().to_owned())
    };

    let (alone_blocks, _) = plan_blocks(Chain::op(transform()));
    let fused = Chain::sequence(vec![
        Chain::op(transform()),
        Chain::op(VoxelwiseMapOp::new("double", doubled)),
    ]);
    let (fused_blocks, fused_out) = plan_blocks(fused);

    assert!(
        alone_blocks > 1,
        "the op alone must cut the volume, got {alone_blocks}"
    );
    assert_eq!(
        fused_blocks, alone_blocks,
        "fusing with a reach of nothing must not change the lattice: {fused_blocks} blocks \
         against {alone_blocks}. A fold that flattened the aligned reach gave 1 here."
    );

    // The answer, against the same chain applied resident.
    let reference = {
        let source: Voxels = input.clone().into();
        let mut out = Voxels::zeros(Dtype::F64, SIDE).expect("a buffer");
        Chain::sequence(vec![
            Chain::op(transform()),
            Chain::op(VoxelwiseMapOp::new("double", doubled)),
        ])
        .apply(&source, &mut out, &Anchor::whole(SIDE))
        .expect("the whole-volume reference must run");
        out.view::<f64>().unwrap().to_owned()
    };
    assert_eq!(
        moved(&reference, &fused_out),
        0,
        "the fused plan disagrees with the whole-volume reference"
    );

    // **Liveness.** The equality above would hold for a chain whose second op
    // did nothing, and then it would not be evidence that a *reaching* fusion
    // was folded rather than dropped. The map must really have run.
    let unmapped = {
        let source: Voxels = input.clone().into();
        let mut out = Voxels::zeros(Dtype::F64, SIDE).expect("a buffer");
        Chain::op(transform())
            .apply(&source, &mut out, &Anchor::whole(SIDE))
            .expect("a run");
        out.view::<f64>().unwrap().to_owned()
    };
    assert!(
        moved(&unmapped, &fused_out) > unmapped.len() / 2,
        "the map in the fused chain did not change the answer, so this test does not \
         distinguish a fused chain from a bare one"
    );
    println!("fused with a reach-nothing map: {fused_blocks} blocks, same as {alone_blocks} alone");
}
