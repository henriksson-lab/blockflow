// SPDX-License-Identifier: MIT
//
// The lattice statistic and the interpolation back, as **two phases**, against
// the fused op that does both.
//
// | test | what it settles |
// |---|---|
// | `the_statistic_reproduces_the_whole_volume_answer` | the coarse phase is decomposition-invariant on its own |
// | `the_interpolation_is_the_lattices_own_brackets` | the fine half computes what the lattice says, at every spacing |
// | `the_two_phases_reproduce_the_fused_op_exactly` | the split is the *same function* as `LocalStatisticOp`, byte for byte |
// | `the_statistic_half_stays_exact_when_the_lattice_is_blocked` | and stays so when the coarse phase is blocked and the fine one is not |
// | `a_multi_block_interpolation_is_the_whole_volume_answer` | the fine half is decomposition-invariant too, now that a block knows where it is |
// | `the_two_phase_split_is_decomposition_invariant_in_both_grids` | and so is the composition, at every pair of block sizes |
// | `a_fetch_short_of_the_samples_a_block_brackets_is_refused` | the check that replaces the one the placement gives up |
// | `the_split_fetches_strictly_less_than_the_fused_form` | the reason the split exists, measured rather than argued |
// | `an_unevenly_spread_lattice_is_refused_by_name` | the case neither op can invert, refused where it is stated |
// | `a_lattice_block_too_small_for_one_window_is_refused` | the statistic's one grid constraint |
//
// **The third test is the load-bearing one.** `LocalStatisticOp` already
// evaluates a statistic on a lattice and interpolates it back, in one op, and it
// is checked against a whole-volume reference elsewhere. So the split has an
// oracle that is not "a second implementation of the same idea": it has to
// reproduce an op that already exists, exactly, and any disagreement is a defect
// in the split rather than a difference of convention.

use blockflow::decomposition::Decomposition;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, Chain};
use blockflow::ops::{
    interpolate_block_edge, lattice_interpolate_phase, lattice_statistic_phase,
    statistic_block_edge, ElementShape, LatticeInterpolateOp, LatticeStatisticOp, LocalStatistic,
    LocalStatisticOp, Rank, SampleLattice, Sampling, Statistic, StructuringElement, Total,
};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;
use ndarray::Array3;
use std::sync::atomic::Ordering;

/// Not a round number on any axis, so a spacing divides none of them evenly
/// unless it was chosen to.
const VOLUME: [usize; 3] = [30, 24, 18];

/// Structure at several scales, so a statistic has something to vary over and a
/// seam has something to get wrong. Deliberately not smooth: a ramp would agree
/// with itself under a wrong sample map.
fn texture(shape: [usize; 3]) -> Array3<f64> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        ((i * 7919 + j * 104729 + k * 1299709) % 1013) as f64 + 1.0
    })
}

fn element(size: [usize; 3]) -> StructuringElement {
    StructuringElement::from_size(ElementShape::Box, size).unwrap()
}

fn spacing(spacing: [usize; 3]) -> Sampling {
    Sampling::every(spacing)
}

fn statistic_op(size: [usize; 3], step: [usize; 3]) -> LatticeStatisticOp {
    LatticeStatisticOp::new(
        "lattice statistic",
        element(size),
        &spacing(step),
        Statistic::Mean,
        VOLUME,
    )
    .unwrap()
}

fn interpolate_op(step: [usize; 3]) -> LatticeInterpolateOp {
    LatticeInterpolateOp::new(
        "lattice interpolate",
        SampleLattice::of(&spacing(step), VOLUME).unwrap(),
    )
    .unwrap()
}

/// The whole-volume reference for the statistic, **written out rather than
/// asked of the op**.
///
/// A cross-grid op cannot be its own reference the way a stencil op can:
/// `apply`ing it to the whole array is not the whole-volume case, because the
/// region a block of the whole lattice is handed is the constant window the op
/// declares and not the array. So the reference here is the definition — for
/// each sample, gather the element around it from the whole volume, clamped at
/// the volume's edges, and reduce — and it shares no code with the geometry
/// under test. A fetch mapping that is subtly wrong cannot agree with this by
/// agreeing with itself.
fn reference_statistic(
    input: &Array3<f64>,
    element: &StructuringElement,
    lattice: &SampleLattice,
    statistic: &Statistic,
) -> Array3<f64> {
    let counts = lattice.lattice_volume();
    let volume = lattice.volume();
    Array3::from_shape_fn((counts[0], counts[1], counts[2]), |(p, q, r)| {
        let centre = [
            lattice.centre(0, p) as isize,
            lattice.centre(1, q) as isize,
            lattice.centre(2, r) as isize,
        ];
        let mut window: Vec<Total> = Vec::new();
        for step in element.offsets() {
            let mut index = [0usize; 3];
            let mut inside = true;
            for axis in 0..3 {
                let at = centre[axis] + step[axis];
                if at < 0 || at >= volume[axis] as isize {
                    inside = false;
                    break;
                }
                index[axis] = at as usize;
            }
            if inside {
                window.push(Total(input[index]));
            }
        }
        statistic.reduce(&mut window, element.len())
    })
}

/// The whole-volume reference for the interpolation, likewise written out: every
/// fine voxel, trilinearly blended from the brackets the lattice gives it.
fn reference_interpolate(coarse: &Array3<f64>, lattice: &SampleLattice) -> Array3<f64> {
    let volume = lattice.volume();
    Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(i, j, k)| {
        let (a0, b0, t0) = lattice.bracket(0, i);
        let (a1, b1, t1) = lattice.bracket(1, j);
        let (a2, b2, t2) = lattice.bracket(2, k);
        let lerp = |a: f64, b: f64, t: f64| a + t * (b - a);
        let edge = |p: usize, q: usize| lerp(coarse[[p, q, a2]], coarse[[p, q, b2]], t2);
        let face = |p: usize| lerp(edge(p, a1), edge(p, b1), t1);
        lerp(face(a0), face(b0), t0)
    })
}

/// The whole-volume reference for an ordinary same-shape chain.
fn whole(chain: &Chain, input: &Voxels) -> Voxels {
    let shape = input.shape();
    let mut out = Voxels::zeros(
        chain.produces(input.dtype()).unwrap(),
        chain.output_shape(shape).unwrap(),
    )
    .unwrap();
    chain.apply(input, &mut out, &Anchor::whole(shape)).unwrap();
    out
}

fn run(chain: Chain, input: &Voxels, decomposition: &Decomposition) -> (Voxels, u64) {
    let workflow = Workflow::new(chain, input.shape(), input.dtype());
    let env = ArrayEnvironment::for_decomposition(input.clone(), decomposition, [4, 4, 4]).unwrap();
    execute("lattice", &workflow, decomposition, &Hints::default(), &env).unwrap();
    let read = env.counters().read_voxels.load(Ordering::SeqCst);
    (env.output(), read)
}

/// A one-phase plan for the statistic, cut on the **lattice**.
fn statistic_plan(op: &LatticeStatisticOp, edge: usize, axes: &[usize]) -> Decomposition {
    // Every axis goes through `statistic_block_edge`, including the ones this
    // case does not split: a full-axis block is exactly the one a lattice phase
    // usually cannot have, so "unsplit" means "as large as the window allows"
    // rather than "the whole axis".
    let counts = op.lattice().lattice_volume();
    let mut block = counts;
    for axis in 0..3 {
        let wanted = if axes.contains(&axis) {
            edge
        } else {
            counts[axis]
        };
        block[axis] = statistic_block_edge(op, axis, wanted).unwrap();
    }
    let grid = BlockGrid::new(counts, block).unwrap();
    let phase =
        lattice_statistic_phase(vec![0], vec!["lattice statistic".to_string()], op, grid).unwrap();
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![phase],
        chain_reach: [0, 0, 0],
    }
}

/// The one-block plan for the interpolation, which is the only one it has.
fn interpolate_plan(op: &LatticeInterpolateOp) -> Decomposition {
    let volume = op.lattice().volume();
    let grid = BlockGrid::new(volume, volume).unwrap();
    let phase =
        lattice_interpolate_phase(vec![0], vec!["lattice interpolate".to_string()], op, grid)
            .unwrap();
    Decomposition {
        volume: op.lattice().lattice_volume(),
        dtype: Dtype::F64,
        phases: vec![phase],
        chain_reach: [0, 0, 0],
    }
}

// ------------------------------------------------ 1. each phase alone --

/// Every decomposition of the lattice reproduces the whole-volume answer.
///
/// The block edges include 1 and 2, which are smaller than any element, so a
/// block's fetch is dominated by the window rather than by its cores — the case
/// in which a fetch that was derived from the core alone would be short.
#[test]
fn the_statistic_reproduces_the_whole_volume_answer() {
    let fine = texture(VOLUME);
    let input: Voxels = fine.clone().into();
    for (size, step) in [
        ([3usize, 3, 3], [2usize, 2, 2]),
        ([5, 5, 3], [4, 4, 2]),
        ([7, 5, 5], [3, 3, 3]),
        // A spacing that divides no axis of the volume evenly.
        ([5, 5, 5], [7, 7, 7]),
        // An even extent: the window is off-centre and the reach is asymmetric.
        ([4, 4, 4], [3, 3, 3]),
    ] {
        let op = statistic_op(size, step);
        let reference: Voxels =
            reference_statistic(&fine, op.element(), op.lattice(), &Statistic::Mean).into();
        for edge in [1usize, 2, 3, 5] {
            for axes in [&[0usize][..], &[0, 1][..], &[0, 1, 2][..]] {
                let plan = statistic_plan(&op, edge, axes);
                let (split, _) = run(Chain::op(statistic_op(size, step)), &input, &plan);
                assert_eq!(
                    split, reference,
                    "element {size:?} spacing {step:?} edge {edge} axes {axes:?}"
                );
            }
        }
    }
}

/// The interpolation computes the lattice's own brackets, at every spacing.
///
/// One block, which is the case with no seams at all —
/// `a_multi_block_interpolation_is_the_whole_volume_answer` is where the blocked
/// ones are. What this settles is the arithmetic:
/// the op agrees with `SampleLattice::bracket` applied voxel by voxel, including
/// in the unsampled margins at each end, where a voxel has no bracket to sit
/// between and reads the nearest sample instead.
#[test]
fn the_interpolation_is_the_lattices_own_brackets() {
    for step in [[2usize, 2, 2], [4, 4, 2], [3, 3, 3], [5, 5, 5], [7, 7, 7]] {
        let op = interpolate_op(step);
        let grid = texture(op.lattice().lattice_volume());
        let coarse: Voxels = grid.clone().into();
        let reference: Voxels = reference_interpolate(&grid, op.lattice()).into();
        assert_eq!(reference.shape(), VOLUME);

        let plan = interpolate_plan(&op);
        let (ours, _) = run(Chain::op(interpolate_op(step)), &coarse, &plan);
        assert_eq!(ours, reference, "spacing {step:?}");

        // The margins are not empty, or the claim about them is vacuous.
        assert!(
            op.lattice().centre(0, 0) > 0,
            "spacing {step:?} leaves no unsampled margin to check"
        );
    }
}

// ----------------------------------------- 2. the split is the fused op --

/// The two phases, composed, are `LocalStatisticOp` — exactly.
///
/// This is the claim the split has to earn. `LocalStatisticOp` evaluates the
/// statistic at every lattice point and interpolates back in one pass; the two
/// ops here do the same two things through two levels. If they were merely
/// *similar* the difference would show up as a handful of voxels near the
/// unsampled margins, which is where the two would most easily disagree about a
/// convention.
#[test]
fn the_two_phases_reproduce_the_fused_op_exactly() {
    let fine = texture(VOLUME);
    let input: Voxels = fine.clone().into();
    for (size, step, rank) in [
        ([3usize, 3, 3], [2usize, 2, 2], None),
        ([5, 5, 3], [4, 4, 2], None),
        ([5, 5, 5], [7, 7, 7], None),
        ([4, 4, 4], [3, 3, 3], None),
        // A rank statistic as well as a mean: the interpolation of a selected
        // value is where `a + t * (b - a)` has to be exact rather than close.
        ([3, 3, 3], [4, 4, 4], Some(Rank::Nth(13))),
    ] {
        let statistic = rank.map_or(Statistic::Mean, Statistic::Rank);
        let op_statistic = statistic.clone();
        let fused = LocalStatisticOp::new(
            "fused",
            LocalStatistic::sampled(element(size), spacing(step), statistic.clone()).unwrap(),
        );
        let theirs = whole(&Chain::op(fused), &input);

        let op =
            LatticeStatisticOp::new("s", element(size), &spacing(step), statistic, VOLUME).unwrap();
        let coarse = reference_statistic(&fine, op.element(), op.lattice(), &op_statistic);
        let ours: Voxels = reference_interpolate(&coarse, op.lattice()).into();

        assert_eq!(
            ours, theirs,
            "element {size:?} spacing {step:?}: the split must be the fused op, not a variant of it"
        );
    }
}

/// The same, with the **coarse** phase blocked, through the executor.
///
/// The two-phase plan is the shape a caller actually writes, and it exercises
/// what one phase alone cannot: the coarse level is *materialised* between the
/// two, so the interpolation reads a level whose shape is neither the volume's
/// nor its own output's. The fine phase is one block, which is all it can be —
/// see `a_multi_block_interpolation_is_refused_and_says_what_is_missing`.
#[test]
fn the_statistic_half_stays_exact_when_the_lattice_is_blocked() {
    let input: Voxels = texture(VOLUME).into();
    for (size, step) in [
        ([3usize, 3, 3], [2usize, 2, 2]),
        ([5, 5, 3], [4, 4, 2]),
        ([5, 5, 5], [7, 7, 7]),
    ] {
        let fused = LocalStatisticOp::new(
            "fused",
            LocalStatistic::sampled(element(size), spacing(step), Statistic::Mean).unwrap(),
        );
        let reference = whole(&Chain::op(fused), &input);

        for coarse_edge in [1usize, 2, 3, 5] {
            let statistic = statistic_op(size, step);
            let interpolate = interpolate_op(step);
            let counts = statistic.lattice().lattice_volume();

            let mut coarse_block = counts;
            for axis in 0..3 {
                coarse_block[axis] = statistic_block_edge(&statistic, axis, coarse_edge).unwrap();
            }
            let first = lattice_statistic_phase(
                vec![0],
                vec!["statistic".to_string()],
                &statistic,
                BlockGrid::new(counts, coarse_block).unwrap(),
            )
            .unwrap();
            let second = lattice_interpolate_phase(
                vec![1],
                vec!["interpolate".to_string()],
                &interpolate,
                BlockGrid::new(VOLUME, VOLUME).unwrap(),
            )
            .unwrap();

            let decomposition = Decomposition {
                volume: VOLUME,
                dtype: Dtype::F64,
                phases: vec![first, second],
                chain_reach: [0, 0, 0],
            };
            let chain = Chain::sequence(vec![
                Chain::op(statistic_op(size, step)),
                Chain::op(interpolate_op(step)),
            ]);
            let (split, _) = run(chain, &input, &decomposition);
            assert_eq!(
                split, reference,
                "element {size:?} spacing {step:?} at coarse edge {coarse_edge}"
            );
        }
    }
}

/// The thing this split could not do, now that it can: a **multi-block**
/// interpolation, byte-identical to the whole-volume answer at every blocking.
///
/// This test replaces the one that pinned the refusal, and the replacement is
/// the same assertion read the other way round. A block of the fine phase writes
/// a span that depends on where it sits — the first sample is half a gap into
/// the volume and the volume's extent is not a whole number of gaps — while
/// `BlockOp::output_shape` sees only the shape of the buffer handed in. What
/// changed is not the lattice but where the extent comes from: `op::Placement`
/// carries the block's position in *both* spaces and the extent the plan cut,
/// and `BlockOp::placed_output_shape` takes it from there.
///
/// **The block edges are chosen to make the end blocks genuinely different.**
/// `VOLUME` is `[30, 24, 18]` and none of the edges below divides its axis, so
/// the last block on each axis is short — and short by an amount that is *not* a
/// whole number of gaps, which is the exact case the old constant-width fetch
/// could not state. An edge that divided the volume would pass this test for the
/// wrong reason.
#[test]
fn a_multi_block_interpolation_is_the_whole_volume_answer() {
    for step in [[4usize, 4, 2], [7, 7, 7], [3, 3, 3], [5, 5, 5]] {
        let op = interpolate_op(step);
        let grid_values = texture(op.lattice().lattice_volume());
        let coarse: Voxels = grid_values.clone().into();
        let reference: Voxels = reference_interpolate(&grid_values, op.lattice()).into();

        for edge in [4usize, 7, 8, 11, 16] {
            let mut block = VOLUME;
            for axis in 0..3 {
                block[axis] = interpolate_block_edge(op.lattice(), axis, edge).unwrap();
            }
            let grid = BlockGrid::new(VOLUME, block).unwrap();
            let blocks = grid.cores().len();
            let phase =
                lattice_interpolate_phase(vec![0], vec!["i".to_string()], &op, grid).unwrap();
            let plan = Decomposition {
                volume: op.lattice().lattice_volume(),
                dtype: Dtype::F64,
                phases: vec![phase],
                chain_reach: [0, 0, 0],
            };
            let (ours, _) = run(Chain::op(interpolate_op(step)), &coarse, &plan);
            assert!(
                blocks > 1,
                "spacing {step:?} edge {edge} planned {blocks} block(s), so it proves nothing"
            );
            assert_eq!(ours, reference, "spacing {step:?} at block edge {edge:?}");
        }
    }
}

/// And the same for the **composition**: the statistic on a blocked lattice,
/// then the interpolation on a blocked fine grid, against the fused op.
///
/// The point the previous test cannot make on its own. Each half is now
/// decomposition-invariant separately; what a caller writes is the pair, through
/// two levels, with the coarse level materialised between them — and the claim
/// worth having is that *no* choice of the two block sizes changes a bit.
#[test]
fn the_two_phase_split_is_decomposition_invariant_in_both_grids() {
    let input: Voxels = texture(VOLUME).into();
    for (size, step) in [
        ([3usize, 3, 3], [2usize, 2, 2]),
        ([5, 5, 3], [4, 4, 2]),
        ([5, 5, 5], [7, 7, 7]),
        ([4, 4, 4], [3, 3, 3]),
    ] {
        let fused = LocalStatisticOp::new(
            "fused",
            LocalStatistic::sampled(element(size), spacing(step), Statistic::Mean).unwrap(),
        );
        let reference = whole(&Chain::op(fused), &input);

        for coarse_edge in [1usize, 2, 3, 5] {
            for fine_edge in [5usize, 8, 13, 30] {
                let statistic = statistic_op(size, step);
                let interpolate = interpolate_op(step);
                let counts = statistic.lattice().lattice_volume();

                let mut coarse_block = counts;
                let mut fine_block = VOLUME;
                for axis in 0..3 {
                    coarse_block[axis] =
                        statistic_block_edge(&statistic, axis, coarse_edge).unwrap();
                    fine_block[axis] =
                        interpolate_block_edge(interpolate.lattice(), axis, fine_edge).unwrap();
                }
                let first = lattice_statistic_phase(
                    vec![0],
                    vec!["statistic".to_string()],
                    &statistic,
                    BlockGrid::new(counts, coarse_block).unwrap(),
                )
                .unwrap();
                let second = lattice_interpolate_phase(
                    vec![1],
                    vec!["interpolate".to_string()],
                    &interpolate,
                    BlockGrid::new(VOLUME, fine_block).unwrap(),
                )
                .unwrap();

                let decomposition = Decomposition {
                    volume: VOLUME,
                    dtype: Dtype::F64,
                    phases: vec![first, second],
                    chain_reach: [0, 0, 0],
                };
                let chain = Chain::sequence(vec![
                    Chain::op(statistic_op(size, step)),
                    Chain::op(interpolate_op(step)),
                ]);
                let (split, _) = run(chain, &input, &decomposition);
                assert_eq!(
                    split, reference,
                    "element {size:?} spacing {step:?} at coarse edge {coarse_edge} and fine \
                     edge {fine_edge}"
                );
            }
        }
    }
}

/// A block handed fewer samples than its voxels bracket is refused **by name**,
/// rather than clamped to the buffer's edge.
///
/// The check that replaces the one `placed_output_shape` gives up. Taking the
/// write extent from the plan means the executor's "declared shape against
/// derived read extent" comparison compares the plan with itself, so the op owes
/// a check against the buffer it was actually handed — which is a check against
/// data rather than against a declaration, and therefore the stronger one. Here
/// the fetch is deliberately one sample short of what the block reads.
#[test]
fn a_fetch_short_of_the_samples_a_block_brackets_is_refused() {
    use blockflow::op::{Anchor, BlockOp, Placement, SourceInputs};

    let op = interpolate_op([4, 4, 2]);
    let counts = op.lattice().lattice_volume();
    // A block writing fine voxels 8..20 of axis 0, handed samples starting at 1
    // when its first voxel brackets from sample 1 — and one sample too few at
    // the top.
    let short = [2usize, counts[1], counts[2]];
    let samples = Voxels::zeros(Dtype::F64, short).unwrap();
    let mut out = Voxels::zeros(Dtype::F64, [12, VOLUME[1], VOLUME[2]]).unwrap();
    let at = Placement::new(
        Anchor::new([1, 0, 0], counts),
        Anchor::new([8, 0, 0], VOLUME),
    )
    .writing([12, VOLUME[1], VOLUME[2]]);
    let error = op
        .apply_placed(&samples, SourceInputs::none(), &mut out, &at)
        .unwrap_err()
        .to_string();
    assert!(error.contains("brackets samples"), "{error}");
    assert!(error.contains("was handed samples"), "{error}");
}

// --------------------------------------------- 3. why the split exists --

/// The split reads strictly fewer voxels than the fused op at the same
/// parameters, and the saving grows with the spacing.
///
/// **Measured through the environment's own counters**, not derived from the
/// reach the ops declare — a claim about cost that is computed from the same
/// numbers the plan was built from would be comparing a number against itself.
#[test]
fn the_split_fetches_strictly_less_than_the_fused_form() {
    let input: Voxels = texture(VOLUME).into();
    let size = [5usize, 5, 3];
    let step = [4usize, 4, 2];
    let edge = 6;

    // The fused form: one phase over the fine volume, halo = spacing + radius.
    let fused = LocalStatistic::sampled(element(size), spacing(step), Statistic::Mean).unwrap();
    let grid = BlockGrid::along(VOLUME, &[0, 1, 2], edge).unwrap();
    let fused_phase = blockflow::decomposition::PhaseDecomposition::derive(
        vec![0],
        vec!["fused".to_string()],
        fused.reach_spec(VOLUME),
        // The reach itself, granted: what a generic planner hands a stencil
        // phase. `LocalStatistic::halo` is the tighter per-block table and is
        // not what is being compared here — the point of the measurement is the
        // *fetch*, and giving the fused form the tighter halo would understate
        // it in the direction that flatters this test.
        fused.reach_spec(VOLUME),
        grid,
    );
    let fused_plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![fused_phase],
        chain_reach: [
            fused.reach(0, VOLUME[0]),
            fused.reach(1, VOLUME[1]),
            fused.reach(2, VOLUME[2]),
        ],
    };
    let (_, fused_counters) = run(
        Chain::op(LocalStatisticOp::new("fused", fused)),
        &input,
        &fused_plan,
    );

    // The split: the same two computations through two phases.
    let statistic = statistic_op(size, step);
    let interpolate = interpolate_op(step);
    let counts = statistic.lattice().lattice_volume();
    let mut coarse_block = counts;
    for axis in 0..3 {
        coarse_block[axis] = statistic_block_edge(&statistic, axis, 2).unwrap();
    }
    let coarse_grid = BlockGrid::new(counts, coarse_block).unwrap();
    let first =
        lattice_statistic_phase(vec![0], vec!["s".to_string()], &statistic, coarse_grid).unwrap();
    let second = lattice_interpolate_phase(
        vec![1],
        vec!["i".to_string()],
        &interpolate,
        BlockGrid::new(VOLUME, VOLUME).unwrap(),
    )
    .unwrap();
    let split_plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![first, second],
        chain_reach: [0, 0, 0],
    };
    let (_, split_counters) = run(
        Chain::sequence(vec![
            Chain::op(statistic_op(size, step)),
            Chain::op(interpolate_op(step)),
        ]),
        &input,
        &split_plan,
    );

    // The split reads the fine level once with an element-sized fetch and the
    // coarse level once; the fused form reads the fine level with a fetch that
    // also carries the spacing. The second read is of a level `4*4*2` times
    // smaller, so it does not undo the saving.
    assert!(
        split_counters < fused_counters,
        "split read {split_counters} voxels and the fused form read {fused_counters}"
    );
}

// ----------------------------------------------------- 4. the refusals --

/// An unevenly spread lattice has no constant fetch width, and both ops say so
/// where the lattice is stated rather than at the first block.
#[test]
fn an_unevenly_spread_lattice_is_refused_by_name() {
    let uneven = Sampling::At {
        positions: [vec![2, 5, 11], vec![2, 8, 14], vec![1, 5, 9]],
    };
    let error = LatticeStatisticOp::new(
        "statistic",
        element([3, 3, 3]),
        &uneven,
        Statistic::Mean,
        VOLUME,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("unevenly spread"), "{error}");
    assert!(error.contains("LocalStatisticOp"), "{error}");

    let lattice = SampleLattice::of(&uneven, VOLUME).unwrap();
    let error = LatticeInterpolateOp::new("interpolate", lattice)
        .unwrap_err()
        .to_string();
    assert!(error.contains("unevenly spread"), "{error}");
}

/// A lattice block whose constant window does not fit the axis is refused, and
/// the message says what to do about it.
#[test]
fn a_lattice_block_too_small_for_one_window_is_refused() {
    // An element wider than the whole volume's axis leaves no window that fits.
    let op = LatticeStatisticOp::new(
        "statistic",
        element([29, 3, 3]),
        &spacing([4, 4, 2]),
        Statistic::Mean,
        VOLUME,
    )
    .unwrap();
    let grid = BlockGrid::along(op.lattice().lattice_volume(), &[0], 1).unwrap();
    let error = lattice_statistic_phase(vec![0], vec!["s".to_string()], &op, grid)
        .unwrap_err()
        .to_string();
    assert!(error.contains("no constant fetch window"), "{error}");
}

/// The statistic's output volume is the lattice's, and its reach is a *marked*
/// nothing rather than a silent one.
#[test]
fn the_statistic_declares_the_lattice_as_its_output_space() {
    use blockflow::op::BlockOp;
    use blockflow::reach::Space;

    let op = statistic_op([5, 5, 3], [4, 4, 2]);
    let geometry = op.geometry(VOLUME);
    assert_eq!(geometry.output_volume(), op.lattice().lattice_volume());
    assert_ne!(geometry.output_volume(), VOLUME);
    assert_eq!(op.reach_spec(VOLUME).space(), Space::source_index());

    // And it is a reach of nothing in that space, which is the statement the
    // per-block fetch regions are the rest of.
    assert!(!op.reach_spec(VOLUME).space().converts_to_voxels());
}

/// An op that answers from **where it is**: `out = in + the global coordinate`.
///
/// Position-dependent on purpose, and reach zero so it can share a phase with
/// anything. It also checks the volume it is anchored to, so a slot handed the
/// coarse anchor fails loudly rather than merely differing.
struct CoordinateOp {
    volume: [usize; 3],
}

impl blockflow::op::BlockOp for CoordinateOp {
    fn name(&self) -> &'static str {
        "coordinate"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<(), blockflow::Error> {
        if at.volume != self.volume {
            return Err(blockflow::Error::InvalidArgument(format!(
                "coordinate op is over {:?} and was anchored to {:?}",
                self.volume, at.volume
            )));
        }
        let input = input.view::<f64>()?;
        let mut out = out.view_mut::<f64>()?;
        for i in 0..input.shape()[0] {
            for j in 0..input.shape()[1] {
                for k in 0..input.shape()[2] {
                    out[[i, j, k]] = input[[i, j, k]] + (at.offset[0] + i) as f64;
                }
            }
        }
        Ok(())
    }

    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        // Position-dependent, so a uniform block is *not* a uniform answer and
        // the short circuit must not fire — which would otherwise skip the very
        // slot this test is about.
        None
    }
}

/// A phase that **fuses** the interpolation with a position-dependent slot,
/// which is what makes the placement per *slot* rather than per phase.
///
/// The fold this test exists for. A phase's slots are a run of chains, each
/// handed what the one before produced, and every one of them used to be given
/// the same `Anchor` — the phase's fetch, in the level that was read. That is
/// right exactly as long as no slot changes the grid. Here the first slot does:
/// it reads the coarse level and writes the fine one, so the slot after it is
/// anchored in a space the phase's fetch does not describe. `CoordinateOp` reads
/// both halves of that anchor, so a slot given the coarse one fails on the
/// volume and a slot given the coarse *offset* would differ on every voxel.
///
/// `op::place_parts` resolves it from the two ends the plan holds, and the
/// interesting half is the **backward** pass: the interpolation's own output
/// anchor is not derivable from its input, so it comes from the slot after it,
/// which does keep its grid. Nothing else in this suite exercises that
/// direction, because every other lattice phase is one slot long.
#[test]
fn a_phase_fusing_the_interpolation_with_a_placed_slot_places_both() {
    let input: Voxels = texture(VOLUME).into();
    let (size, step) = ([5usize, 5, 3], [4usize, 4, 2]);
    let statistic = statistic_op(size, step);
    let coarse = reference_statistic(
        &texture(VOLUME),
        statistic.element(),
        statistic.lattice(),
        &Statistic::Mean,
    );
    let interpolated = reference_interpolate(&coarse, statistic.lattice());
    let reference: Voxels =
        Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
            interpolated[[i, j, k]] + i as f64
        })
        .into();

    let counts = statistic.lattice().lattice_volume();
    let mut coarse_block = counts;
    for axis in 0..3 {
        coarse_block[axis] = statistic_block_edge(&statistic, axis, 2).unwrap();
    }
    let first = lattice_statistic_phase(
        vec![0],
        vec!["statistic".to_string()],
        &statistic,
        BlockGrid::new(counts, coarse_block).unwrap(),
    )
    .unwrap();

    let interpolate = interpolate_op(step);
    for fine_edge in [7usize, 11, 30] {
        let mut fine_block = VOLUME;
        for axis in 0..3 {
            fine_block[axis] =
                interpolate_block_edge(interpolate.lattice(), axis, fine_edge).unwrap();
        }
        let second = lattice_interpolate_phase(
            vec![1, 2],
            vec!["interpolate".to_string(), "coordinate".to_string()],
            &interpolate,
            BlockGrid::new(VOLUME, fine_block).unwrap(),
        )
        .unwrap();
        let decomposition = Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases: vec![first.clone(), second],
            chain_reach: [0, 0, 0],
        };
        let chain = Chain::sequence(vec![
            Chain::op(statistic_op(size, step)),
            Chain::op(interpolate_op(step)),
            Chain::op(CoordinateOp { volume: VOLUME }),
        ]);
        let (fused, _) = run(chain, &input, &decomposition);
        assert_eq!(fused, reference, "fine edge {fine_edge}");
    }
}
