// SPDX-License-Identifier: MIT
//
// The **interpolation convention** on the lattice interpolation: which samples a
// fine voxel is blended from, as a declared parameter rather than a fact about
// the lattice.
//
// | test | what it settles |
// |---|---|
// | `a_hand_computed_case_pins_both_conventions` | the arithmetic, against numbers derived on paper |
// | `the_two_conventions_disagree_wherever_there_is_anything_to_disagree_about` | the parameter is observable, and where it stops being |
// | `the_default_is_the_convention_this_op_already_had` | nothing moved for a caller who says nothing |
// | `a_single_sample_axis_is_the_sample_everywhere` | the degenerate axis, under both |
// | `a_single_voxel_axis_is_the_first_sample` | the other end of the same guard |
// | `the_pinned_convention_is_decomposition_invariant` | byte-identical across block sizes, end blocks included |
// | `the_composition_is_decomposition_invariant_in_both_grids` | and so is the pair of phases |
// | `the_fetch_is_exactly_the_samples_the_convention_reads` | the geometry claim, checked rather than assumed |
// | `the_convention_changes_no_declared_volume_or_reach` | and the rest of the declaration is untouched |
// | `the_convention_is_part_of_what_the_op_is` | declared, not inferred |
//
// The same convention reaches the **fused** op — the one that evaluates the
// samples and interpolates them in a single pass — through the same
// `Alignment::bracket`, and that half is settled below:
//
// | test | what it settles |
// |---|---|
// | `a_hand_computed_case_pins_the_fused_op` | the fused arithmetic, against the same numbers derived on paper |
// | `the_fused_default_is_the_convention_it_already_had` | nothing moved for a caller who says nothing |
// | `the_fused_and_split_paths_are_the_same_function` | two independent paths, one rule, byte for byte |
// | `the_fused_op_is_decomposition_invariant_when_the_ends_are_pinned` | byte-identical across block sizes, end blocks included |
// | `the_threshold_carries_the_convention_as_well` | and so does the op composed after it |
// | `the_fused_reach_grows_with_the_convention` | the one declaration that *does* move, and why it must |
// | `the_granted_halo_covers_the_samples_each_block_reads` | and the tighter per-block table moves with it |
//
// **Why a hand-computed case is the first test.** The two conventions agree
// wherever a voxel sits exactly on a sample, so a lattice of every voxel — and
// any case with a whole-number ratio — cannot tell them apart. A test that
// merely asserted the code agrees with a second copy of the same formula would
// pass just as well against the wrong formula. The numbers in
// `a_hand_computed_case_pins_both_conventions` were worked out from the
// definitions, on an axis chosen so that every weight is a power of two and
// every product is exact in binary, so the assertion is on equality and not on a
// tolerance.

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::{
    interpolate_block_edge, lattice_interpolate_phase, lattice_statistic_phase,
    statistic_block_edge, AdaptiveThresholdOp, Alignment, ElementShape, LatticeInterpolateOp,
    LatticeStatisticOp, LocalStatistic, LocalStatisticOp, SampleLattice, Sampling, Statistic,
    StructuringElement, Total,
};
use blockflow::reach::Space;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;
use ndarray::Array3;

/// Not a round number on any axis, so no spacing and no block edge divides one
/// unless it was chosen to.
const VOLUME: [usize; 3] = [30, 24, 18];

fn texture(shape: [usize; 3]) -> Array3<f64> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        ((i * 7919 + j * 104729 + k * 1299709) % 1013) as f64 + 1.0
    })
}

fn lattice(volume: [usize; 3], step: [usize; 3]) -> SampleLattice {
    SampleLattice::of(&Sampling::every(step), volume).unwrap()
}

fn interpolate_op(
    volume: [usize; 3],
    step: [usize; 3],
    convention: Alignment,
) -> LatticeInterpolateOp {
    LatticeInterpolateOp::new("interpolate", lattice(volume, step))
        .unwrap()
        .with_alignment(convention)
}

fn run(chain: Chain, input: &Voxels, decomposition: &Decomposition) -> Voxels {
    let workflow = Workflow::new(chain, input.shape(), input.dtype());
    let env = ArrayEnvironment::for_decomposition(input.clone(), decomposition, [4, 4, 4]).unwrap();
    execute("lattice", &workflow, decomposition, &Hints::default(), &env).unwrap();
    env.output()
}

/// A plan for the interpolation alone, cut at `edge` on every axis.
fn interpolate_plan(op: &LatticeInterpolateOp, edge: usize) -> (Decomposition, usize) {
    let fine = op.lattice().volume();
    let mut block = fine;
    for axis in 0..3 {
        block[axis] = interpolate_block_edge(op.lattice(), axis, edge).unwrap();
    }
    let grid = BlockGrid::new(fine, block).unwrap();
    let blocks = grid.cores().len();
    let phase = lattice_interpolate_phase(vec![0], vec!["i".to_string()], op, grid).unwrap();
    (
        Decomposition {
            volume: op.lattice().lattice_volume(),
            dtype: Dtype::F64,
            phases: vec![phase],
            chain_reach: [0, 0, 0],
        },
        blocks,
    )
}

// ------------------------------------------------------- the two oracles --

/// The **existing** convention, written out: a voxel placed against the sample
/// coordinates that bracket it, in fine voxels.
///
/// This is `reference_interpolate` from `lattice_split.rs`, repeated here so
/// that "the default is unchanged" is asserted against a form of the rule that
/// does not go through `Alignment` at all.
fn reference_positions(coarse: &Array3<f64>, lattice: &SampleLattice) -> Array3<f64> {
    let volume = lattice.volume();
    Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(i, j, k)| {
        let (a0, b0, t0) = lattice.bracket(0, i);
        let (a1, b1, t1) = lattice.bracket(1, j);
        let (a2, b2, t2) = lattice.bracket(2, k);
        blend(coarse, [a0, a1, a2], [b0, b1, b2], [t0, t1, t2])
    })
}

/// The **new** convention, written out from its definition and from nothing
/// else: sample coordinate `v * (n - 1) / (len - 1)`, clamped at the top, split
/// by `floor` and `ceil`.
///
/// Deliberately not a call into the crate. It is the oracle the invariance tests
/// compare against, and an oracle that asked the implementation what the answer
/// was would only be asserting that the code agrees with itself.
fn pinned_coordinate(voxel: usize, samples: usize, extent: usize) -> f64 {
    if samples <= 1 || extent <= 1 {
        return 0.0;
    }
    voxel as f64 * (samples - 1) as f64 / (extent - 1) as f64
}

fn pinned_bracket(voxel: usize, samples: usize, extent: usize) -> (usize, usize, f64) {
    let at = pinned_coordinate(voxel, samples, extent).min((samples - 1) as f64);
    let low = at.floor() as usize;
    (low, at.ceil() as usize, at - low as f64)
}

fn reference_pinned(coarse: &Array3<f64>, lattice: &SampleLattice) -> Array3<f64> {
    let volume = lattice.volume();
    let counts = lattice.lattice_volume();
    Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(i, j, k)| {
        let (a0, b0, t0) = pinned_bracket(i, counts[0], volume[0]);
        let (a1, b1, t1) = pinned_bracket(j, counts[1], volume[1]);
        let (a2, b2, t2) = pinned_bracket(k, counts[2], volume[2]);
        blend(coarse, [a0, a1, a2], [b0, b1, b2], [t0, t1, t2])
    })
}

/// The samples themselves, written out: the mean over `element` clamped to the
/// volume, at each lattice position.
///
/// The half of the answer the convention has **no** opinion about — both
/// conventions evaluate the same statistic at the same places, and differ only
/// in which of the results a voxel is blended from. Computed here from the
/// definition so that a whole-volume reference for either path can be built
/// without asking either path what it does.
fn reference_samples(
    fine: &Array3<f64>,
    element: &StructuringElement,
    lattice: &SampleLattice,
) -> Array3<f64> {
    let volume = lattice.volume();
    let counts = lattice.lattice_volume();
    Array3::from_shape_fn((counts[0], counts[1], counts[2]), |(p, q, r)| {
        let centre = [
            lattice.centre(0, p) as isize,
            lattice.centre(1, q) as isize,
            lattice.centre(2, r) as isize,
        ];
        let mut window: Vec<Total> = Vec::new();
        for offset in element.offsets() {
            let mut index = [0usize; 3];
            let mut inside = true;
            for axis in 0..3 {
                let at = centre[axis] + offset[axis];
                if at < 0 || at >= volume[axis] as isize {
                    inside = false;
                    break;
                }
                index[axis] = at as usize;
            }
            if inside {
                window.push(Total(fine[index]));
            }
        }
        Statistic::Mean.reduce(&mut window, element.len())
    })
}

/// The trilinear blend, in the crate's own nesting and its own `a + t * (b - a)`
/// form. Shared by the two oracles so that a difference between them is a
/// difference of *convention* and not of summation order.
fn blend(coarse: &Array3<f64>, a: [usize; 3], b: [usize; 3], t: [f64; 3]) -> f64 {
    let lerp = |x: f64, y: f64, w: f64| x + w * (y - x);
    let edge = |p: usize, q: usize| lerp(coarse[[p, q, a[2]]], coarse[[p, q, b[2]]], t[2]);
    let face = |p: usize| lerp(edge(p, a[1]), edge(p, b[1]), t[1]);
    lerp(face(a[0]), face(b[0]), t[0])
}

// ----------------------------------------------------- 1. the arithmetic --

/// The numbers, worked out on paper.
///
/// An axis of **nine** voxels carrying **three** samples, at a spacing of three.
/// The lattice leaves a margin of one voxel at each end, so its samples sit at
/// volume coordinates `1, 4, 7` — and the two conventions then say different
/// things about every voxel that is not one of those three.
///
/// *Between the samples' own positions*, the gap is three voxels:
///
/// | voxel | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
/// |---|---|---|---|---|---|---|---|---|---|
/// | samples | 0 | 0 | 0,1 | 0,1 | 1 | 1,2 | 1,2 | 2 | 2 |
/// | weight | — | — | 1/3 | 2/3 | — | 1/3 | 2/3 | — | — |
///
/// Voxels 0 and 8 lie outside the sampled span and read the nearest sample, so
/// each end is a plateau two voxels wide.
///
/// *With both ends pinned*, the coordinate is `v * (3 - 1) / (9 - 1)`, which is
/// `v / 4` — so voxel 0 lands on sample 0, voxel 4 on sample 1 and voxel 8 on
/// sample 2, and the weights are quarters:
///
/// | voxel | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
/// |---|---|---|---|---|---|---|---|---|---|
/// | coordinate | 0 | ¼ | ½ | ¾ | 1 | 1¼ | 1½ | 1¾ | 2 |
/// | samples | 0 | 0,1 | 0,1 | 0,1 | 1 | 1,2 | 1,2 | 1,2 | 2 |
///
/// There is no plateau at either end, which is the visible difference.
///
/// The sample values are `8, 16, 40` — gaps of 8 and 24 — so every quarter-step
/// is a whole number and the pinned column below is exact in binary. The
/// positions column is written as the thirds it is, because a third is not.
#[test]
fn a_hand_computed_case_pins_both_conventions() {
    let volume = [9usize, 1, 1];
    let step = [3usize, 1, 1];
    let lattice = lattice(volume, step);

    // The lattice this case is argued from, checked before anything rests on it.
    assert_eq!(lattice.lattice_volume(), [3, 1, 1]);
    assert_eq!(lattice.positions(0), &[1, 4, 7]);

    let mut coarse = Array3::zeros((3, 1, 1));
    coarse[[0, 0, 0]] = 8.0;
    coarse[[1, 0, 0]] = 16.0;
    coarse[[2, 0, 0]] = 40.0;
    let input: Voxels = coarse.into();

    let third = 1.0f64 / 3.0;
    let expected = [
        // (sample positions, both ends pinned)
        (8.0, 8.0),
        (8.0, 10.0),
        (8.0 + third * 8.0, 12.0),
        (8.0 + (2.0 * third) * 8.0, 14.0),
        (16.0, 16.0),
        (16.0 + third * 24.0, 22.0),
        (16.0 + (2.0 * third) * 24.0, 28.0),
        (40.0, 34.0),
        (40.0, 40.0),
    ];

    for (convention, pick) in [
        (Alignment::SamplePositions, 0usize),
        (Alignment::PinnedEnds, 1),
    ] {
        let op = interpolate_op(volume, step, convention);
        let (plan, _) = interpolate_plan(&op, 9);
        let got = run(
            Chain::op(interpolate_op(volume, step, convention)),
            &input,
            &plan,
        );
        let got = got.view::<f64>().unwrap();
        for voxel in 0..9 {
            let want = if pick == 0 {
                expected[voxel].0
            } else {
                expected[voxel].1
            };
            assert_eq!(
                got[[voxel, 0, 0]],
                want,
                "{convention:?} at voxel {voxel}: the hand-computed value is {want}"
            );
        }
    }
}

/// The parameter is observable — and the case in which it is not is named.
///
/// Half the point of a convention is that somebody could pick the wrong one and
/// not notice. On a coarse lattice the two disagree on most voxels; at a spacing
/// of one every voxel is its own sample and the two are the same function, which
/// is the degenerate case the two arms of a caller can sit in without ever
/// finding out which convention they wanted.
#[test]
fn the_two_conventions_disagree_wherever_there_is_anything_to_disagree_about() {
    for step in [[4usize, 4, 2], [7, 7, 7], [3, 3, 3], [5, 5, 5]] {
        let here = lattice(VOLUME, step);
        let grid = texture(here.lattice_volume());
        let positions = reference_positions(&grid, &here);
        let pinned = reference_pinned(&grid, &here);
        let differing = positions
            .iter()
            .zip(pinned.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            differing > positions.len() / 2,
            "spacing {step:?}: only {differing} of {} voxels differ between the two \
             conventions, so a test at this spacing would prove little",
            positions.len()
        );
    }

    // And the degenerate case, stated rather than left to be discovered.
    let every = lattice(VOLUME, [1, 1, 1]);
    let grid = texture(every.lattice_volume());
    assert_eq!(
        reference_positions(&grid, &every),
        reference_pinned(&grid, &every),
        "at a spacing of one every voxel is its own sample and the choice cannot be seen"
    );
}

// ------------------------------------------------- 2. nothing else moved --

/// A caller who says nothing gets exactly the numbers it got before.
///
/// Asserted against `reference_positions`, which is the rule written out rather
/// than the op asked what it does, and asserted **byte for byte** — this is an
/// interpolation, so "close" would hide a convention change everywhere except at
/// the ends.
#[test]
fn the_default_is_the_convention_this_op_already_had() {
    assert_eq!(Alignment::default(), Alignment::SamplePositions);

    for step in [[2usize, 2, 2], [4, 4, 2], [3, 3, 3], [5, 5, 5], [7, 7, 7]] {
        let plain = LatticeInterpolateOp::new("interpolate", lattice(VOLUME, step)).unwrap();
        assert_eq!(
            plain.alignment(),
            Alignment::SamplePositions,
            "the constructor must not have acquired an opinion"
        );

        let grid = texture(plain.lattice().lattice_volume());
        let coarse: Voxels = grid.clone().into();
        let reference: Voxels = reference_positions(&grid, plain.lattice()).into();

        for edge in [7usize, 11, 30] {
            let (plan, _) = interpolate_plan(&plain, edge);
            let got = run(
                Chain::op(LatticeInterpolateOp::new("interpolate", lattice(VOLUME, step)).unwrap()),
                &coarse,
                &plan,
            );
            assert_eq!(got, reference, "spacing {step:?} at block edge {edge}");
        }
    }
}

// --------------------------------------------------- 3. degenerate axes --

/// One sample on an axis: every voxel of that axis reads it, under both
/// conventions.
///
/// The case the pinned rule's ratio would divide by zero over if it were written
/// the other way up. It is not undefined and it is not refused — with one sample
/// there is nothing to interpolate *between*, so the sample coordinate is zero
/// everywhere and the answer is the sample. The existing convention reaches the
/// same place by a different route (there is no second sample to bracket
/// against), which is why this is one assertion for both.
#[test]
fn a_single_sample_axis_is_the_sample_everywhere() {
    let volume = [5usize, 6, 7];
    // A spacing as long as the axis leaves exactly one sample on it.
    let step = [5usize, 6, 7];
    let here = lattice(volume, step);
    assert_eq!(here.lattice_volume(), [1, 1, 1], "the case is set up wrong");

    let mut coarse = Array3::zeros((1, 1, 1));
    coarse[[0, 0, 0]] = 17.5;
    let input: Voxels = coarse.into();

    for convention in [Alignment::SamplePositions, Alignment::PinnedEnds] {
        let op = interpolate_op(volume, step, convention);
        let (plan, _) = interpolate_plan(&op, 3);
        let got = run(
            Chain::op(interpolate_op(volume, step, convention)),
            &input,
            &plan,
        );
        let expected: Voxels = Array3::from_elem((volume[0], volume[1], volume[2]), 17.5).into();
        assert_eq!(got, expected, "{convention:?}");
    }
}

/// One *voxel* on an axis: the other clause of the same guard, and the one that
/// is a division by zero in the ratio itself.
///
/// An axis of one voxel can hold no more than one sample, so this arrives with
/// the previous case rather than instead of it — but the guard is written with
/// both clauses because the expression has both a numerator and a denominator,
/// and reading only one of them is how a rule gets reproduced half-way.
#[test]
fn a_single_voxel_axis_is_the_first_sample() {
    let volume = [1usize, 9, 9];
    let step = [1usize, 3, 3];
    let here = lattice(volume, step);
    assert_eq!(here.volume()[0], 1);
    assert_eq!(here.lattice_volume(), [1, 3, 3]);

    let grid = texture(here.lattice_volume());
    let coarse: Voxels = grid.clone().into();
    for convention in [Alignment::SamplePositions, Alignment::PinnedEnds] {
        let op = interpolate_op(volume, step, convention);
        let (plan, _) = interpolate_plan(&op, 4);
        let got = run(
            Chain::op(interpolate_op(volume, step, convention)),
            &coarse,
            &plan,
        );
        let got = got.view::<f64>().unwrap();
        // Axis 0 contributes the first (and only) sample with weight zero; the
        // other two axes interpolate normally, so this is not a vacuous case.
        for j in 0..9 {
            for k in 0..9 {
                let (a1, b1, t1) = pinned_bracket(j, 3, 9);
                let (a2, b2, t2) = pinned_bracket(k, 3, 9);
                if convention == Alignment::PinnedEnds {
                    assert_eq!(
                        got[[0, j, k]],
                        blend(&grid, [0, a1, a2], [0, b1, b2], [0.0, t1, t2]),
                        "{convention:?} at {j},{k}"
                    );
                }
            }
        }
    }
}

// ------------------------------------------- 4. decomposition invariance --

/// The new convention, byte-identical to the whole-volume answer at every block
/// size.
///
/// **The block edges are chosen so the end blocks genuinely differ in span.**
/// `VOLUME` is `[30, 24, 18]` and none of the edges below divides any of its
/// axes, so the last block on each axis is short — and short by an amount that
/// is not a whole number of sample gaps, which is the case `op::Placement` made
/// expressible. An edge that divided the volume would pass for the wrong reason,
/// so each blocking is also asserted to have produced more than one block.
///
/// The reference is `reference_pinned`, computed once over the whole volume from
/// the rule's own definition. What the pinned convention makes easy to get wrong
/// under decomposition is precisely what it makes easy to get right: the
/// coordinate is a function of the **volume's** extent and the **global** voxel
/// index, so a block that measured either against its own buffer would be wrong
/// on every voxel it owns rather than only near a seam.
#[test]
fn the_pinned_convention_is_decomposition_invariant() {
    for step in [[4usize, 4, 2], [7, 7, 7], [3, 3, 3], [5, 5, 5]] {
        let op = interpolate_op(VOLUME, step, Alignment::PinnedEnds);
        let grid = texture(op.lattice().lattice_volume());
        let coarse: Voxels = grid.clone().into();
        let reference: Voxels = reference_pinned(&grid, op.lattice()).into();
        assert_eq!(reference.shape(), VOLUME);

        for edge in [4usize, 7, 8, 11, 16, 30] {
            let (plan, blocks) = interpolate_plan(&op, edge);
            let got = run(
                Chain::op(interpolate_op(VOLUME, step, Alignment::PinnedEnds)),
                &coarse,
                &plan,
            );
            assert_eq!(got, reference, "spacing {step:?} at block edge {edge}");
            if edge < VOLUME[2] {
                assert!(
                    blocks > 1,
                    "spacing {step:?} edge {edge} planned {blocks} block(s)"
                );
            }
        }
    }
}

/// And the same for the pair of phases: the statistic on a blocked lattice, then
/// the interpolation on a blocked fine grid, at the new convention.
///
/// The statistic half is untouched by the convention — it writes samples and
/// interpolates nothing — so what this adds over the test above is that the
/// coarse level is *materialised* between the two, and the interpolation reads a
/// level whose shape is neither the volume's nor its own output's while still
/// mapping global fine coordinates.
#[test]
fn the_composition_is_decomposition_invariant_in_both_grids() {
    let fine = texture(VOLUME);
    let input: Voxels = fine.clone().into();
    let size = [5usize, 5, 3];

    for step in [[4usize, 4, 2], [7, 7, 7]] {
        let statistic = || {
            LatticeStatisticOp::new(
                "statistic",
                StructuringElement::from_size(ElementShape::Box, size).unwrap(),
                &Sampling::every(step),
                Statistic::Mean,
                VOLUME,
            )
            .unwrap()
        };
        let interpolate = || interpolate_op(VOLUME, step, Alignment::PinnedEnds);

        // The whole-volume reference: the statistic's definition, then the
        // pinned rule's definition. Neither goes through a plan.
        let here = lattice(VOLUME, step);
        let element = StructuringElement::from_size(ElementShape::Box, size).unwrap();
        let counts = here.lattice_volume();
        let coarse = reference_samples(&fine, &element, &here);
        let reference: Voxels = reference_pinned(&coarse, &here).into();

        for coarse_edge in [1usize, 2, 3] {
            for fine_edge in [7usize, 11, 16, 30] {
                let first_op = statistic();
                let mut coarse_block = counts;
                for axis in 0..3 {
                    coarse_block[axis] =
                        statistic_block_edge(&first_op, axis, coarse_edge).unwrap();
                }
                let first = lattice_statistic_phase(
                    vec![0],
                    vec!["statistic".to_string()],
                    &first_op,
                    BlockGrid::new(counts, coarse_block).unwrap(),
                )
                .unwrap();

                let second_op = interpolate();
                let mut fine_block = VOLUME;
                for axis in 0..3 {
                    fine_block[axis] =
                        interpolate_block_edge(second_op.lattice(), axis, fine_edge).unwrap();
                }
                let second = lattice_interpolate_phase(
                    vec![1],
                    vec!["interpolate".to_string()],
                    &second_op,
                    BlockGrid::new(VOLUME, fine_block).unwrap(),
                )
                .unwrap();

                let plan = Decomposition {
                    volume: VOLUME,
                    dtype: Dtype::F64,
                    phases: vec![first, second],
                    chain_reach: [0, 0, 0],
                };
                let chain = Chain::sequence(vec![Chain::op(statistic()), Chain::op(interpolate())]);
                let got = run(chain, &input, &plan);
                assert_eq!(
                    got, reference,
                    "spacing {step:?} at coarse edge {coarse_edge} and fine edge {fine_edge}"
                );
            }
        }
    }
}

// -------------------------------------------------- 5. what is declared --

/// Every block's fetch is exactly the samples its own voxels read under the
/// convention it was given — no wider, and never short.
///
/// **The claim the invariance tests cannot make on their own.** They would still
/// pass if the fetch were three samples too wide on every block: the answers
/// would be right and the plan would merely be dearer than it says. This asks
/// the fetch region directly and compares it against the union of the brackets
/// the block's voxels take, computed from the rule's own definition — so a fetch
/// derived under one convention while the kernel reads under another is caught
/// here rather than showing up as a refusal on some later volume.
#[test]
fn the_fetch_is_exactly_the_samples_the_convention_reads() {
    for step in [[4usize, 4, 2], [7, 7, 7], [3, 3, 3]] {
        for convention in [Alignment::SamplePositions, Alignment::PinnedEnds] {
            let op = interpolate_op(VOLUME, step, convention);
            let counts = op.lattice().lattice_volume();
            for edge in [4usize, 7, 11] {
                let mut block = VOLUME;
                for axis in 0..3 {
                    block[axis] = interpolate_block_edge(op.lattice(), axis, edge).unwrap();
                }
                let grid = BlockGrid::new(VOLUME, block).unwrap();
                let phase =
                    lattice_interpolate_phase(vec![0], vec!["i".to_string()], &op, grid).unwrap();
                for geometry in &phase.blocks {
                    let fetched = &geometry.source;
                    for axis in 0..3 {
                        let from = geometry.read.start[axis];
                        let extent = geometry.read.shape[axis];
                        if extent == 0 {
                            continue;
                        }
                        // The samples this block's voxels actually read, taken
                        // one voxel at a time so that nothing rests on the
                        // monotonicity the op assumes.
                        let mut lowest = usize::MAX;
                        let mut highest = 0usize;
                        for voxel in from..from + extent {
                            let (a, b, _) = match convention {
                                Alignment::SamplePositions => op.lattice().bracket(axis, voxel),
                                Alignment::PinnedEnds => {
                                    pinned_bracket(voxel, counts[axis], VOLUME[axis])
                                }
                            };
                            lowest = lowest.min(a);
                            highest = highest.max(b);
                        }
                        assert_eq!(
                            (fetched.start[axis], fetched.shape[axis]),
                            (lowest, highest - lowest + 1),
                            "{convention:?} spacing {step:?} edge {edge}, block {:?} axis {axis}: \
                             its voxels read samples {lowest}..={highest}",
                            geometry.index
                        );
                    }
                }
            }
        }
    }
}

/// The convention changes which samples are read and **nothing else** that is
/// declared: not the output volume, not the reach, not its space.
///
/// Worth asserting rather than assuming, because the two conventions read
/// different source positions and a parameter that quietly moved a declared
/// geometry would be a plan that no longer describes the op.
#[test]
fn the_convention_changes_no_declared_volume_or_reach() {
    for step in [[4usize, 4, 2], [7, 7, 7]] {
        let positions = interpolate_op(VOLUME, step, Alignment::SamplePositions);
        let pinned = interpolate_op(VOLUME, step, Alignment::PinnedEnds);
        let counts = positions.lattice().lattice_volume();

        assert_eq!(
            positions.geometry(counts).output_volume(),
            pinned.geometry(counts).output_volume()
        );
        assert_eq!(pinned.geometry(counts).output_volume(), VOLUME);
        assert_eq!(pinned.output_shape(counts), VOLUME);
        assert_eq!(pinned.reach_spec(VOLUME), positions.reach_spec(VOLUME));
        assert_eq!(pinned.reach_spec(VOLUME).space(), Space::source_index());
        assert_eq!(pinned.cost_per_voxel(), positions.cost_per_voxel());
        // A uniform coarse level is still exactly uniform after the blend, under
        // either convention: every tap is `a + t * (b - a)` with `a == b`.
        assert_eq!(pinned.constant_maps_to(3.5), Some(3.5));
    }
}

/// The choice is part of what the op **is**, so two ops differing only in it are
/// different ops.
///
/// Declared, not inferred: nothing looks at the lattice and decides. That
/// matters because the two conventions coincide on any lattice whose ratio is a
/// whole number, so an op that guessed would be right on the cases where the
/// difference does not show and wrong on the ones where it does.
#[test]
fn the_convention_is_part_of_what_the_op_is() {
    let positions = interpolate_op(VOLUME, [4, 4, 2], Alignment::SamplePositions);
    let pinned = interpolate_op(VOLUME, [4, 4, 2], Alignment::PinnedEnds);

    assert_eq!(positions.alignment(), Alignment::SamplePositions);
    assert_eq!(pinned.alignment(), Alignment::PinnedEnds);
    assert_ne!(positions, pinned);
    assert_eq!(positions, positions.clone());
    assert_eq!(
        pinned,
        interpolate_op(VOLUME, [4, 4, 2], Alignment::SamplePositions)
            .with_alignment(Alignment::PinnedEnds)
    );

    // And it survives the other builder, so the two cannot be ordered wrongly.
    let dear = pinned.clone().with_cost(9.0);
    assert_eq!(dear.alignment(), Alignment::PinnedEnds);
    assert_eq!(dear.cost_per_voxel(), 9.0);

    // A lattice of every voxel is the case where the two agree; they are still
    // not the same op, because equality is of the declaration and not of the
    // numbers one particular lattice happens to produce.
    assert_ne!(
        interpolate_op(VOLUME, [1, 1, 1], Alignment::SamplePositions),
        interpolate_op(VOLUME, [1, 1, 1], Alignment::PinnedEnds)
    );
}

// -------------------------------------------------------- 6. the fused op --
//
// `LocalStatisticOp` evaluates the samples and interpolates them back in one
// pass, so it has to be told the same thing the pair of ops above is told. Every
// test in this section is the *same claim* as one made for the split form; what
// they add is that the fused arm reaches the convention at all, and that the two
// arms are one function rather than two implementations of one idea.

/// A window of a single voxel, so the statistic is the identity and the samples
/// are the input's own values at the lattice positions.
///
/// That is what makes a hand-computed case possible for an op that also computes
/// the samples: the sample values are numbers chosen by the test rather than
/// means of something.
fn point_element() -> StructuringElement {
    StructuringElement::from_size(ElementShape::Box, [1, 1, 1]).unwrap()
}

fn fused_op(
    element: StructuringElement,
    step: [usize; 3],
    convention: Alignment,
) -> LocalStatisticOp {
    LocalStatisticOp::new(
        "fused",
        LocalStatistic::sampled(element, Sampling::every(step), Statistic::Mean).unwrap(),
    )
    .with_alignment(convention)
}

/// The whole-volume answer, with no plan in it at all.
fn whole(op: &LocalStatisticOp, input: &Voxels) -> Voxels {
    let shape = input.shape();
    let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
    op.apply(input, &mut out, &Anchor::whole(shape)).unwrap();
    out
}

/// The numbers from `a_hand_computed_case_pins_both_conventions`, arrived at
/// through the op that computes the samples as well as the blend.
///
/// **Same paper, other path.** The axis is nine voxels carrying three samples at
/// a spacing of three, so the lattice positions are `1, 4, 7`; the window is a
/// single voxel, so the samples *are* the input at those three positions, and
/// the input is written so that they are `8, 16, 40` — the same three values the
/// split case blends. The expected columns are therefore the same two columns,
/// and they were derived from the definitions rather than from either
/// implementation.
///
/// The six voxels that are not sampled hold a value no expected number contains,
/// so a window that read one of them would be visible rather than absorbed.
#[test]
fn a_hand_computed_case_pins_the_fused_op() {
    let volume = [9usize, 1, 1];
    let step = [3usize, 1, 1];

    let mut fine = Array3::from_elem((9, 1, 1), 1_000_000.0);
    fine[[1, 0, 0]] = 8.0;
    fine[[4, 0, 0]] = 16.0;
    fine[[7, 0, 0]] = 40.0;
    let input: Voxels = fine.into();

    // The lattice this case is argued from, checked before anything rests on it.
    assert_eq!(lattice(volume, step).positions(0), &[1, 4, 7]);

    let third = 1.0f64 / 3.0;
    let expected = [
        // (sample positions, both ends pinned)
        (8.0, 8.0),
        (8.0, 10.0),
        (8.0 + third * 8.0, 12.0),
        (8.0 + (2.0 * third) * 8.0, 14.0),
        (16.0, 16.0),
        (16.0 + third * 24.0, 22.0),
        (16.0 + (2.0 * third) * 24.0, 28.0),
        (40.0, 34.0),
        (40.0, 40.0),
    ];

    for (convention, pick) in [
        (Alignment::SamplePositions, 0usize),
        (Alignment::PinnedEnds, 1),
    ] {
        let got = whole(&fused_op(point_element(), step, convention), &input);
        let got = got.view::<f64>().unwrap();
        for voxel in 0..9 {
            let want = if pick == 0 {
                expected[voxel].0
            } else {
                expected[voxel].1
            };
            assert_eq!(
                got[[voxel, 0, 0]],
                want,
                "{convention:?} at voxel {voxel}: the hand-computed value is {want}"
            );
        }
    }
}

/// A caller who says nothing gets exactly the numbers it got before.
///
/// Asserted against the rule written out — `reference_samples` then
/// `reference_positions`, neither of which goes through `Alignment` — and
/// asserted byte for byte. The reach is checked too, because the fused op's
/// reach *does* move with the convention and a default that had picked up the
/// wider one would re-plan every existing pipeline.
#[test]
fn the_fused_default_is_the_convention_it_already_had() {
    let fine = texture(VOLUME);
    let input: Voxels = fine.clone().into();
    let size = [5usize, 5, 3];
    let element = StructuringElement::from_size(ElementShape::Box, size).unwrap();

    for step in [[4usize, 4, 2], [7, 7, 7], [3, 3, 3], [5, 5, 5]] {
        let plain = LocalStatisticOp::new(
            "fused",
            LocalStatistic::sampled(element.clone(), Sampling::every(step), Statistic::Mean)
                .unwrap(),
        );
        assert_eq!(
            plain.alignment(),
            Alignment::SamplePositions,
            "the constructor must not have acquired an opinion"
        );

        let here = lattice(VOLUME, step);
        let reference: Voxels =
            reference_positions(&reference_samples(&fine, &element, &here), &here).into();
        assert_eq!(
            whole(&plain, &input),
            reference,
            "spacing {step:?}: the default moved"
        );

        // And the declaration a plan is built from is the one it was.
        let stated = fused_op(element.clone(), step, Alignment::SamplePositions);
        assert_eq!(plain.reach_spec(VOLUME), stated.reach_spec(VOLUME));
        for axis in 0..3 {
            assert_eq!(
                plain.reach(axis, VOLUME[axis]),
                Sampling::every(step).max_distance(axis, VOLUME[axis]) + element.sides(axis).0
            );
        }
    }
}

/// The strongest check available: **two independent paths, one rule.**
///
/// The fused op and the pair of split ops share nothing but `Alignment::bracket`
/// — the fused form holds a `Sampling` and materialises its lattice per call
/// from the anchor, the split pair binds a `SampleLattice` into each op and
/// passes a coarse level between two phases — and they are asserted equal on the
/// bits under both conventions, at four spacings none of which divides any axis
/// of `VOLUME`.
///
/// Under `SamplePositions` this restates what `lattice_split.rs` already pins;
/// under `PinnedEnds` it is the claim this work exists to make.
#[test]
fn the_fused_and_split_paths_are_the_same_function() {
    let input: Voxels = texture(VOLUME).into();
    let size = [5usize, 5, 3];
    let element = StructuringElement::from_size(ElementShape::Box, size).unwrap();

    for convention in [Alignment::SamplePositions, Alignment::PinnedEnds] {
        for step in [[4usize, 4, 2], [7, 7, 7], [3, 3, 3], [5, 5, 5]] {
            let fused = whole(&fused_op(element.clone(), step, convention), &input);

            let statistic = LatticeStatisticOp::new(
                "statistic",
                element.clone(),
                &Sampling::every(step),
                Statistic::Mean,
                VOLUME,
            )
            .unwrap();
            let interpolate = interpolate_op(VOLUME, step, convention);
            let counts = statistic.lattice().lattice_volume();

            let mut coarse_block = counts;
            let mut fine_block = VOLUME;
            for axis in 0..3 {
                coarse_block[axis] = statistic_block_edge(&statistic, axis, 2).unwrap();
                fine_block[axis] = interpolate_block_edge(interpolate.lattice(), axis, 11).unwrap();
            }
            let plan = Decomposition {
                volume: VOLUME,
                dtype: Dtype::F64,
                phases: vec![
                    lattice_statistic_phase(
                        vec![0],
                        vec!["statistic".to_string()],
                        &statistic,
                        BlockGrid::new(counts, coarse_block).unwrap(),
                    )
                    .unwrap(),
                    lattice_interpolate_phase(
                        vec![1],
                        vec!["interpolate".to_string()],
                        &interpolate,
                        BlockGrid::new(VOLUME, fine_block).unwrap(),
                    )
                    .unwrap(),
                ],
                chain_reach: [0, 0, 0],
            };
            let chain = Chain::sequence(vec![
                Chain::op(
                    LatticeStatisticOp::new(
                        "statistic",
                        element.clone(),
                        &Sampling::every(step),
                        Statistic::Mean,
                        VOLUME,
                    )
                    .unwrap(),
                ),
                Chain::op(interpolate_op(VOLUME, step, convention)),
            ]);
            let split = run(chain, &input, &plan);

            assert_eq!(
                split, fused,
                "{convention:?} spacing {step:?}: the fused op and the split pair must be one \
                 function, not two implementations of one idea"
            );
        }
    }
}

/// A plan for the fused op alone, cut at `edge` on every axis.
///
/// The halo granted is the reach itself, which is what a generic planner hands a
/// stencil phase. `LocalStatistic::halo` is the tighter per-block table and is
/// not a halo a phase can be *derived* with — a phase whose halo is below its
/// reach loses part of every core — so it is checked directly, in
/// `the_granted_halo_covers_the_samples_each_block_reads`.
fn fused_plan(op: &LocalStatisticOp, edge: usize) -> (Decomposition, usize) {
    let grid = BlockGrid::along(VOLUME, &[0, 1, 2], edge).unwrap();
    let blocks = grid.cores().len();
    let statistic = op.statistic();
    let phase = PhaseDecomposition::derive(
        vec![0],
        vec!["fused".to_string()],
        statistic.reach_spec(VOLUME),
        statistic.reach_spec(VOLUME),
        grid,
    );
    (
        Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases: vec![phase],
            chain_reach: [
                statistic.reach(0, VOLUME[0]),
                statistic.reach(1, VOLUME[1]),
                statistic.reach(2, VOLUME[2]),
            ],
        },
        blocks,
    )
}

/// The fused op under the pinned convention is byte-identical to the
/// whole-volume answer at every block size.
///
/// **The block edges are chosen so the end blocks genuinely differ in span.**
/// None of them divides any axis of `VOLUME`, so the last block on each axis is
/// short — and short by an amount that is not a whole number of sample gaps.
///
/// The reference is the whole-volume answer of the same op, which is the form
/// with no plan in it at all. What the pinned convention makes easy to get wrong
/// under decomposition is precisely what it makes easy to get right: the
/// coordinate is a function of the **volume's** extent and the **global** voxel
/// index, so a block that measured either against its own buffer would be wrong
/// on every voxel it owns rather than only near a seam.
#[test]
fn the_fused_op_is_decomposition_invariant_when_the_ends_are_pinned() {
    let input: Voxels = texture(VOLUME).into();
    let element = StructuringElement::from_size(ElementShape::Box, [5, 5, 3]).unwrap();

    for step in [[4usize, 4, 2], [7, 7, 7], [3, 3, 3], [5, 5, 5]] {
        let op = fused_op(element.clone(), step, Alignment::PinnedEnds);
        let reference = whole(&op, &input);

        for edge in [7usize, 11, 13, 16] {
            let (plan, blocks) = fused_plan(&op, edge);
            let got = run(
                Chain::op(fused_op(element.clone(), step, Alignment::PinnedEnds)),
                &input,
                &plan,
            );
            assert_eq!(got, reference, "spacing {step:?} at block edge {edge}");
            assert!(
                blocks > 1,
                "spacing {step:?} edge {edge} planned {blocks} block(s)"
            );
        }
    }
}

/// The threshold carries the convention too, and carries it into the level it
/// compares against rather than into the comparison.
///
/// Checked against the fused statistic's own output put through the same affine
/// adjustment and the same comparison — so what is asserted is that the
/// parameter arrives, not that thresholding works.
#[test]
fn the_threshold_carries_the_convention_as_well() {
    let fine = texture(VOLUME);
    let input: Voxels = fine.clone().into();
    let element = StructuringElement::from_size(ElementShape::Box, [5, 5, 3]).unwrap();
    let step = [7usize, 7, 7];
    let (scale, offset) = (1.25f64, -3.5f64);

    let threshold = AdaptiveThresholdOp::new(
        "threshold",
        LocalStatistic::sampled(element.clone(), Sampling::every(step), Statistic::Mean).unwrap(),
        scale,
        offset,
    )
    .with_alignment(Alignment::PinnedEnds);
    assert_eq!(threshold.alignment(), Alignment::PinnedEnds);

    let mut got = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    threshold
        .apply(&input, &mut got, &Anchor::whole(VOLUME))
        .unwrap();

    let level = whole(&fused_op(element, step, Alignment::PinnedEnds), &input);
    let level = level.view::<f64>().unwrap();
    let want = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        let against = scale * level[[i, j, k]] + offset;
        if fine[[i, j, k]] > against {
            1.0
        } else {
            0.0
        }
    });
    assert_eq!(got, want.into());

    // And the default is still what it was.
    assert_eq!(
        AdaptiveThresholdOp::new(
            "threshold",
            LocalStatistic::new(
                StructuringElement::from_size(ElementShape::Box, [3, 3, 3]).unwrap(),
                step,
                Statistic::Mean
            )
            .unwrap(),
            scale,
            offset,
        )
        .alignment(),
        Alignment::SamplePositions
    );
}

/// The one declaration the convention **does** move, and the direction it moves
/// in.
///
/// The split interpolation declares `Reach::none()` in the source-index space
/// and states its dependency per block as a fetch region, so its reach cannot
/// change. The fused op has no second level to fetch from: its dependency is on
/// fine voxels of its own input and is stated as a halo. Pinning the ends maps a
/// voxel near an end past the lattice's unsampled margin, so it reads a sample
/// that sits further away than any voxel reaches under the other convention —
/// and the halo has to cover it or every block would be truncating a window the
/// whole volume would not have truncated.
///
/// Checked against the samples actually read, taken one voxel at a time from the
/// rule's own definition rather than from the crate's.
#[test]
fn the_fused_reach_grows_with_the_convention() {
    let element = StructuringElement::from_size(ElementShape::Box, [5, 5, 3]).unwrap();

    for step in [[4usize, 4, 2], [7, 7, 7], [5, 5, 5]] {
        let here = lattice(VOLUME, step);
        let counts = here.lattice_volume();
        let pinned = fused_op(element.clone(), step, Alignment::PinnedEnds);
        let positions = fused_op(element.clone(), step, Alignment::SamplePositions);

        for axis in 0..3 {
            // The furthest a voxel is from a sample it blends, under the pinned
            // rule written out here.
            let mut furthest = 0usize;
            for voxel in 0..VOLUME[axis] {
                let (low, high, _) = pinned_bracket(voxel, counts[axis], VOLUME[axis]);
                for index in [low, high] {
                    furthest = furthest.max(here.positions(axis)[index].abs_diff(voxel));
                }
            }
            let (lo, hi) = element.sides(axis);
            assert_eq!(
                pinned.statistic().reach_sides(axis, VOLUME[axis]),
                (furthest + lo, furthest + hi),
                "spacing {step:?} axis {axis}"
            );
            assert!(
                pinned.reach(axis, VOLUME[axis]) > positions.reach(axis, VOLUME[axis]),
                "spacing {step:?} axis {axis}: a lattice with an unsampled margin at each end \
                 must reach further when the ends are pinned"
            );
        }

        // Nothing else in the declaration moved.
        assert_eq!(pinned.output_shape(VOLUME), positions.output_shape(VOLUME));
        assert_eq!(pinned.cost_per_voxel(), positions.cost_per_voxel());
        assert_eq!(
            pinned.constant_maps_to(3.5),
            positions.constant_maps_to(3.5)
        );
    }
}

/// The tighter per-block halo table follows the convention as well.
///
/// `LocalStatistic::halo` says what a block *needs* — the window below the
/// lowest sample its core voxels read and above the highest — rather than the
/// worst case over the volume, and it is the number a planner that wants to pay
/// less would grant. Derived under one mapping and read under another it would
/// be short on exactly the blocks near the volume's ends, which is where a
/// pinned mapping sends a voxel furthest from its samples.
///
/// Compared against the table written out from the pinned rule's own definition,
/// per block and per side.
#[test]
fn the_granted_halo_covers_the_samples_each_block_reads() {
    let element = StructuringElement::from_size(ElementShape::Box, [5, 5, 3]).unwrap();

    for step in [[4usize, 4, 2], [7, 7, 7], [5, 5, 5]] {
        let here = lattice(VOLUME, step);
        let counts = here.lattice_volume();
        let statistic =
            LocalStatistic::sampled(element.clone(), Sampling::every(step), Statistic::Mean)
                .unwrap()
                .with_alignment(Alignment::PinnedEnds);

        for edge in [7usize, 11, 13] {
            let grid = BlockGrid::along(VOLUME, &[0, 1, 2], edge).unwrap();
            let halo = statistic.halo(&grid).unwrap();
            let block = grid.block();
            for axis in 0..3 {
                let (window_lo, window_hi) = element.sides(axis);
                let mut index = 0usize;
                let mut core_lo = 0usize;
                while core_lo < VOLUME[axis] {
                    let core_hi = (core_lo + block[axis]).min(VOLUME[axis]);
                    let low = pinned_bracket(core_lo, counts[axis], VOLUME[axis]).0;
                    let high = pinned_bracket(core_hi - 1, counts[axis], VOLUME[axis]).1;
                    let lowest = here.positions(axis)[low].saturating_sub(window_lo);
                    let highest = here.positions(axis)[high] + window_hi;
                    assert_eq!(
                        halo.at(axis, index, VOLUME[axis]),
                        (
                            core_lo.saturating_sub(lowest),
                            highest.saturating_sub(core_hi - 1)
                        ),
                        "spacing {step:?} edge {edge} axis {axis} block {index}"
                    );
                    index += 1;
                    core_lo = core_hi;
                }
            }
        }
    }
}
