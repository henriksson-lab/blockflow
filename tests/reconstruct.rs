// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `ops::reconstruct`: grey morphological
// reconstruction, and the h-maxima transform built on it.
//
// This is the first op in the crate that is an **iteration** rather than a pass,
// and it is the operation `crate::iterate` was built for. So the bar has three
// parts rather than two.
//
// 1. **The bar every op here meets.** Byte-identical output under several block
//    decompositions — including one block, and including cuts that leave a
//    partial block on every axis — against a whole-volume reference that runs the
//    same kernel in a loop.
// 2. **The bar an iteration adds.** Propagation is the point: a ridge at
//    constant height crossing several blocks, seeded at one end, must arrive at
//    the far end, and the **substage count must grow with the ridge's length**.
//    The second half is what distinguishes propagation from a local computation
//    that happened to be right; without it a bounded-reach implementation could
//    pass the whole suite on data whose ridges all fit inside one block.
// 3. **The bar this operation adds.** `h` is a *prominence threshold*, and
//    nothing in parts 1 and 2 would notice if it were off by a factor or applied
//    to the wrong quantity. So there is a scene of peaks of **known** prominence
//    and an assertion that exactly the ones below `h` are removed and exactly the
//    ones above survive, truncated by `h`.
//
// And the reason the op exists at all is the composition at the end:
// `regional_maxima(HMAX_h(f))` is the extended-maxima transform, which is a
// prominence-filtered maximum finder. The count of maxima is asserted at each
// `h` against a known number rather than merely asserted to be decreasing, so a
// change that removed the wrong peaks would fail rather than pass monotonically.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::fragment::PhaseWork;
use blockflow::geometry::BlockGrid;
use blockflow::iterate::{iterative_phase, substage_reach, IterativeOp, Substage, SubstageLimit};
use blockflow::log::Stats;
use blockflow::op::{Anchor, Chain};
use blockflow::ops::element::{ElementShape, StructuringElement};
use blockflow::ops::reconstruct::{flooding_bound, h_extrema, HExtremaOp, Reconstruction};
use blockflow::ops::regional::{
    regional_maxima, regional_phases, LabelPlateauxOp, RegionalMaximaOp,
};
use blockflow::ops::skeleton::PassLimit;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [24, 10, 6];

// ------------------------------------------------------------- the scenes --

/// The full 26-neighbourhood. Used everywhere except the propagation test, which
/// says there why it wants a flat one.
fn element() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, [1, 1, 1])
}

/// A volume with two things in it, each there to catch something different.
///
/// * **Eight isolated spikes of prominence 1 to 8**, each a single voxel over a
///   flat background of zero, three voxels apart so that no two are neighbours
///   under any element this suite uses. Their prominence is exactly their height,
///   which is what makes `h` assertable against known numbers.
/// * **A ridge at 5.0 running the whole length of axis 0**, with a single 9.0
///   voxel at one end of it. The ridge is one flat plateau, so nothing but the
///   9.0 can raise it, and raising its far end means crossing every block on that
///   axis. Without it a decomposition test would be comparing two answers that
///   both fit inside a block.
fn scene() -> Array3<f64> {
    let mut values = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 0.0);
    for (n, x) in [1usize, 4, 7, 10, 13, 16, 19, 22].into_iter().enumerate() {
        values[[x, 6, 3]] = (n + 1) as f64;
    }
    for x in 0..VOLUME[0] {
        values[[x, 1, 1]] = 5.0;
    }
    values[[0, 1, 1]] = 9.0;
    values
}

/// Where the spikes are and how tall each one is.
fn spikes() -> Vec<([usize; 3], f64)> {
    [1usize, 4, 7, 10, 13, 16, 19, 22]
        .into_iter()
        .enumerate()
        .map(|(n, x)| ([x, 6, 3], (n + 1) as f64))
        .collect()
}

/// The scene for part 3 and for the composition: **only** the graded spikes, on
/// a flat background.
///
/// Separate from [`scene`] because the maxima are counted exactly, and a ridge in
/// the volume would put its own contribution into every count.
fn graded_spikes() -> Array3<f64> {
    let mut values = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 0.0);
    for (at, height) in spikes() {
        values[at] = height;
    }
    values
}

// ----------------------------------------------------------------- the run --

fn op(h: f64, limit: SubstageLimit) -> HExtremaOp {
    HExtremaOp::maxima("h-maxima", element(), h, limit).expect("a non-negative h")
}

fn generous() -> SubstageLimit {
    flooding_bound(VOLUME, &element())
}

fn plan_with(op: &HExtremaOp, block: [usize; 3], volume: [usize; 3]) -> Decomposition {
    let grid = BlockGrid::new(volume, block).expect("a lattice");
    Decomposition {
        volume,
        dtype: Dtype::F64,
        phases: vec![iterative_phase(op, grid).expect("an iterative phase")],
        chain_reach: substage_reach(op),
    }
}

fn plan(op: &HExtremaOp, block: [usize; 3]) -> Decomposition {
    plan_with(op, block, VOLUME)
}

/// An iterative phase owns no chain slot, so the workflow it runs under has none.
fn empty_workflow(volume: [usize; 3]) -> Workflow {
    Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::F64)
}

fn run_plan(
    plan: &Decomposition,
    source: Voxels,
    op: &HExtremaOp,
) -> blockflow::Result<(Stats, ArrayEnvironment)> {
    let volume = plan.volume;
    let env = ArrayEnvironment::for_decomposition(source, plan, [4, 4, 4]).expect("an environment");
    let stats = execute_phases(
        "reconstruct",
        &empty_workflow(volume),
        plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Iterate(op)],
    )?;
    Ok((stats, env))
}

fn run(op: &HExtremaOp, values: &Array3<f64>, block: [usize; 3]) -> (Array3<f64>, usize) {
    let volume = [values.shape()[0], values.shape()[1], values.shape()[2]];
    let plan = plan_with(op, block, volume);
    let (stats, env) = run_plan(&plan, values.clone().into(), op).expect("a run");
    (
        env.output().view::<f64>().expect("f64").to_owned(),
        stats.substages[0],
    )
}

/// The whole-volume reference: **the op's own substage**, in a loop, over the
/// whole array with no blocks and no halo at all.
///
/// The loop is the executor's own written out — the running operand starts as the
/// input, the fixed operand is the input at every substage, and it stops when a
/// substage changes nothing. Not a second implementation of the transform: a
/// disagreement between this and a decomposed run is a decomposition bug, which
/// is the only thing the comparison is for.
fn reference(op: &HExtremaOp, values: &Array3<f64>) -> (Array3<f64>, usize) {
    let volume = [values.shape()[0], values.shape()[1], values.shape()[2]];
    let at = Anchor::whole(volume);
    let fixed: Voxels = values.clone().into();
    let mut current = fixed.clone();
    let mut ran = 0usize;
    loop {
        let mut out = Voxels::zeros(Dtype::F64, volume).expect("a buffer");
        {
            let operands: [&Voxels; 2] = [&current, &fixed];
            op.substage(&Substage::new(ran, &operands, &at), &mut out)
                .expect("a substage");
        }
        ran += 1;
        let changed = out != current;
        current = out;
        if !changed {
            break;
        }
    }
    (current.view::<f64>().expect("f64").to_owned(), ran)
}

/// The decompositions the suite sweeps: one block — no seams at all — several
/// shapes, and three that leave a partial block on an axis, one of them on every
/// axis at once.
fn blockings() -> Vec<[usize; 3]> {
    vec![
        VOLUME,
        [8, 10, 6],
        [12, 5, 3],
        [8, 4, 2],
        [7, 4, 4],
        [5, 10, 6],
    ]
}

// ------------------------------------------------------------ the scene --

/// Every other test here is worthless if the scene does not hold what it claims,
/// so it is asserted first.
#[test]
fn the_scene_holds_peaks_of_known_prominence_and_a_ridge_that_crosses_every_block() {
    let values = scene();
    for (at, height) in spikes() {
        assert_eq!(values[at], height);
        // isolated: its whole 26-neighbourhood is background, so its prominence
        // above its surroundings is exactly its height
        for a in -1isize..=1 {
            for b in -1isize..=1 {
                for c in -1isize..=1 {
                    if [a, b, c] == [0, 0, 0] {
                        continue;
                    }
                    let neighbour = [
                        (at[0] as isize + a) as usize,
                        (at[1] as isize + b) as usize,
                        (at[2] as isize + c) as usize,
                    ];
                    assert_eq!(
                        values[neighbour], 0.0,
                        "the spike at {at:?} is not isolated"
                    );
                }
            }
        }
    }
    for x in 1..VOLUME[0] {
        assert_eq!(values[[x, 1, 1]], 5.0);
    }
    assert_eq!(values[[0, 1, 1]], 9.0);
    // and the ridge really is cut by every blocking the suite sweeps except the
    // one-block case, or the propagation those runs test would be block-local
    for block in blockings() {
        if block == VOLUME {
            continue;
        }
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        assert!(
            grid.blocks_per_axis()[0] > 1,
            "cut {block:?} leaves the ridge inside one block, so it tests nothing about seams"
        );
    }
}

// ------------------------------------------------ 1. decomposition invariance --

/// The answer is the whole-volume answer, byte-identical, under every cut — and
/// the substage count is the whole volume's too.
#[test]
fn an_h_maxima_phase_is_decomposition_invariant() {
    let values = scene();
    let op = op(3.0, generous());
    let (expected, expected_substages) = reference(&op, &values);

    // The reference is not the input: an op that did nothing at all would pass a
    // comparison against itself.
    assert_ne!(expected, values);
    // and it really does iterate, or the invariance would be a property of one
    // pass rather than of an iteration
    assert!(
        expected_substages > 20,
        "the reference converged in {expected_substages} substages, which is too few for this \
         scene to be testing propagation"
    );

    let mut answers = Vec::new();
    for block in blockings() {
        let (got, substages) = run(&op, &values, block);
        assert_eq!(
            got, expected,
            "cut {block:?} disagrees with the whole volume"
        );
        assert_eq!(
            substages, expected_substages,
            "cut {block:?} took a different number of substages than the whole volume"
        );
        answers.push(got);
    }
    for answer in &answers[1..] {
        assert_eq!(answer, &answers[0], "two decompositions disagree");
    }
}

/// The whole-volume convenience is the same composition the phase is, including
/// the way it counts substages. Asserted so that the tests below may use the
/// cheap path and still be talking about the op.
#[test]
fn the_whole_volume_transform_agrees_with_the_phase_it_models() {
    let values = scene();
    for h in [0.0, 1.0, 2.5, 4.0] {
        let op = op(h, generous());
        let (expected, expected_substages) = reference(&op, &values);
        let (got, substages) = h_extrema(
            values.view(),
            &element(),
            Reconstruction::ByDilation,
            h,
            generous(),
        )
        .expect("a whole-volume run");
        assert_eq!(got, expected, "h = {h}");
        assert_eq!(substages, expected_substages, "h = {h}");

        let (blocked, blocked_substages) = run(&op, &values, [8, 4, 2]);
        assert_eq!(blocked, expected, "h = {h}");
        assert_eq!(blocked_substages, expected_substages, "h = {h}");
    }
}

// ------------------------------------------------------- 2. what h means --

/// **The defining property**, and the one test without which this op is
/// unverified however green the rest of the suite is: `h` is a prominence
/// threshold in intensity units.
///
/// Peaks of prominence 1 to 8 over a flat background. At each `h`, exactly the
/// peaks whose prominence is at most `h` are gone — flattened to the level of the
/// base they stand on — and exactly the ones above it survive, each truncated by
/// `h` and by nothing else.
#[test]
fn exactly_the_peaks_below_h_are_removed_and_the_ones_above_survive_truncated_by_h() {
    let values = graded_spikes();
    for h in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0] {
        let op = op(h, generous());
        let (got, _) = run(&op, &values, [8, 4, 2]);
        for (at, height) in spikes() {
            if height <= h {
                assert_eq!(
                    got[at], 0.0,
                    "a peak of prominence {height} survived h = {h}: it stands {height} above \
                     its base and {h} is at least that, so it must be flattened to the base"
                );
            } else {
                assert_eq!(
                    got[at],
                    height - h,
                    "a peak of prominence {height} under h = {h} must survive, truncated by h \
                     and by nothing else"
                );
            }
        }
        // and the base itself is untouched wherever it is reachable from a
        // surviving peak, which is what "flooded from its own base" means
        if h < 8.0 {
            assert_eq!(got[[0, 0, 0]], 0.0, "h = {h}");
        }
    }
}

/// `HMAX_0` is the identity, **exactly**, and every `h` is bounded above by the
/// input pointwise. The first is the boundary case a rounding error would break;
/// the second is what makes the transform a *lowering* rather than an arbitrary
/// filter.
#[test]
fn h_zero_is_the_identity_and_every_h_is_bounded_above_by_the_input() {
    let values = scene();

    let identity = op(0.0, generous());
    let (got, substages) = run(&identity, &values, [8, 4, 2]);
    assert_eq!(got, values, "HMAX_0 is not the identity");
    assert_eq!(
        substages, 1,
        "the seed is the input, and substage 0 both writes it and answers the convergence \
         question — so the identity costs one pass rather than a pass and a check"
    );

    for h in [0.5, 1.0, 3.0, 9.0, 100.0] {
        let lowered = op(h, generous());
        let (got, _) = run(&lowered, &values, [8, 4, 2]);
        for (slot, source) in got.iter().zip(values.iter()) {
            assert!(
                slot <= source,
                "HMAX_{h} rose above the input: {slot} > {source}"
            );
        }
    }
}

// ------------------------------------- 3. reconstruction's own properties --

/// The answer is between the seed and the mask, and it is a **fixed point**: one
/// more substage moves nothing. All three, because two of them hold of the seed
/// itself and only the third says the iteration finished.
#[test]
fn the_answer_lies_between_the_seed_and_the_mask_and_one_more_substage_changes_nothing() {
    let values = scene();
    let h = 3.0;
    let op = op(h, generous());
    let (got, _) = run(&op, &values, [8, 4, 2]);

    for (slot, source) in got.iter().zip(values.iter()) {
        assert!(
            *slot >= source - h,
            "below the seed: {slot} < {source} - {h}"
        );
        assert!(slot <= source, "above the mask: {slot} > {source}");
    }

    // one more substage, run by hand through the op's own kernel
    let at = Anchor::whole(VOLUME);
    let running: Voxels = got.clone().into();
    let mask: Voxels = values.clone().into();
    let mut again = Voxels::zeros(Dtype::F64, VOLUME).expect("a buffer");
    let operands: [&Voxels; 2] = [&running, &mask];
    op.substage(&Substage::new(9, &operands, &at), &mut again)
        .expect("a substage");
    assert_eq!(
        again.view::<f64>().expect("f64").to_owned(),
        got,
        "the answer is not a fixed point of the step it is defined by"
    );
}

/// The dual, over the same scene turned upside down. `HMIN_h(-f) == -HMAX_h(f)`,
/// which is what "the dual" means and is a property no shared code path can fake:
/// the two differ by two comparisons and by the sign of the seed's offset.
#[test]
fn h_minima_is_the_dual_of_h_maxima_under_negation() {
    let values = scene();
    let negated = values.mapv(|value| -value);
    let h = 3.0;

    let (maxima, up) = h_extrema(
        values.view(),
        &element(),
        Reconstruction::ByDilation,
        h,
        generous(),
    )
    .expect("a run");
    let (minima, down) = h_extrema(
        negated.view(),
        &element(),
        Reconstruction::ByErosion,
        h,
        generous(),
    )
    .expect("a run");

    assert_eq!(minima, maxima.mapv(|value| -value));
    assert_eq!(up, down);
}

// --------------------------------------------------------- 4. propagation --

/// The volume the propagation test uses: long on axis 0 so a ridge crosses many
/// blocks.
const LINE: [usize; 3] = [40, 4, 4];

/// A ridge at constant height along axis 0, `length` voxels of it, seeded by one
/// higher voxel at its near end and stopped by one deep voxel at its far end.
///
/// **The element used with this scene is flat on axes 1 and 2**, and that is the
/// point of the scene rather than a saving. A flood spreads through everything
/// its element connects, so with a solid element the background around the ridge
/// floods too and the substage count becomes the maximum of two distances. Flat
/// on the other two axes, nothing but this one line can move, so the count is a
/// function of the ridge's length and of nothing else — which is the quantity the
/// test is about.
///
/// The deep voxel is what stops the flood: the seed arrives, is capped at the
/// deep voxel's own value, and the ordinary background beyond it is above that,
/// so nothing further changes. Without it the flood would run to the end of the
/// volume whatever `length` was, and the count would stop depending on it.
fn ridge(length: usize) -> Array3<f64> {
    let mut values = Array3::from_elem((LINE[0], LINE[1], LINE[2]), 0.0);
    for x in 1..=length {
        values[[x, 1, 1]] = 5.0;
    }
    values[[0, 1, 1]] = 10.0;
    values[[length + 1, 1, 1]] = -10.0;
    values
}

fn flat_element() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, [1, 0, 0])
}

/// **The test that fails for any bounded-reach implementation.** A ridge at
/// constant height crossing several blocks, seeded at one end, arrives at the
/// other — and the substage count grows with its length, which is what
/// distinguishes propagation from a local computation that happened to be right.
#[test]
fn a_ridge_propagates_its_whole_length_and_the_substage_count_grows_with_it() {
    let mut counts = Vec::new();
    for length in [4usize, 12, 20, 30] {
        let values = ridge(length);
        let op = HExtremaOp::maxima(
            "h-maxima",
            flat_element(),
            1.0,
            flooding_bound(LINE, &flat_element()),
        )
        .expect("a positive h");
        let (got, substages) = run(&op, &values, [8, 4, 4]);

        // The whole ridge is at the mask's own level, all the way to the far end,
        // which is five blocks from the voxel that raised it. An implementation
        // that read a fixed halo would leave the far end at the seed's 4.0.
        for x in 1..=length {
            assert_eq!(
                got[[x, 1, 1]],
                5.0,
                "the ridge stops at x = {x} of {length}; the flood did not travel"
            );
        }
        assert_eq!(got[[0, 1, 1]], 9.0, "the seeding voxel is truncated by h");
        assert_eq!(
            got[[length + 1, 1, 1]],
            -10.0,
            "the deep voxel caps the flood at its own value"
        );

        // and the block-decomposed run agrees with the whole volume, count and all
        let (expected, expected_substages) = reference(&op, &values);
        assert_eq!(got, expected, "length {length}");
        assert_eq!(substages, expected_substages, "length {length}");
        counts.push(substages);
    }

    // The count is a function of the data, and this is what "grows with the
    // length" means: strictly increasing, and by the length's own increments.
    // `length` substages to travel — one voxel each — plus the substage that
    // derives the seed and the one that observes that nothing moved. The deep
    // voxel costs nothing extra: it is pulled down to its own value by the first
    // substage, since everything beside it is above it.
    assert_eq!(counts, vec![6, 14, 22, 32]);
    for (count, length) in counts.iter().zip([4usize, 12, 20, 30]) {
        assert_eq!(*count, length + 2);
    }
    for pair in counts.windows(2) {
        assert!(pair[1] > pair[0], "{counts:?} is not increasing");
    }
}

/// The external reach is one substage's, whatever the count — the property the
/// whole iterative shape exists for, stated here for this op because it is what
/// makes the test above affordable.
#[test]
fn the_external_reach_is_the_elements_radius_and_does_not_grow_with_the_substage_count() {
    let op = op(1.0, generous());
    assert_eq!(substage_reach(&op), [1, 1, 1]);
    let plan = plan(&op, [8, 4, 2]);
    assert_eq!(plan.phases[0].reach, [1, 1, 1]);
    assert_eq!(plan.phases[0].halo, [1, 1, 1]);

    // A wider element reaches wider, and nothing else does: no factor, no
    // configured halo, no substage count.
    let wide = HExtremaOp::maxima(
        "h-maxima",
        StructuringElement::from_radius(ElementShape::Box, [3, 2, 1]),
        1.0,
        generous(),
    )
    .expect("a positive h");
    assert_eq!(substage_reach(&wide), [3, 2, 1]);

    // The same plan over two datasets that iterate very differently: the
    // fingerprint and the reach are the same object either way.
    let brief = ridge(1);
    let deep = ridge(30);
    let flat = HExtremaOp::maxima(
        "h-maxima",
        flat_element(),
        1.0,
        flooding_bound(LINE, &flat_element()),
    )
    .expect("a positive h");
    let plan = plan_with(&flat, [8, 4, 4], LINE);
    let (short, _) = run_plan(&plan, brief.into(), &flat).expect("a run");
    let (long, _) = run_plan(&plan, deep.into(), &flat).expect("a run");
    assert!(long.substages[0] > 4 * short.substages[0]);
    assert_eq!(plan.phases[0].halo, [1, 0, 0]);
    assert_eq!(
        short.decomposition_fingerprint,
        long.decomposition_fingerprint
    );
}

// --------------------------------------------------------------- 5. limits --

/// The volume the maze lives in: a slab, so that a corridor in it can be far
/// longer than the slab's diameter.
const SLAB: [usize; 3] = [16, 16, 1];

/// A serpentine corridor at constant height with one higher voxel at its start.
///
/// Eight rows spanning the slab, joined alternately at each end: a corridor about
/// 135 voxels long inside a slab whose L1 diameter is 30. **This is the case
/// [`flooding_bound`] does not cover and says it does not cover** — the bound is
/// a shortest-path length and a mask can force a detour — so it is here, as a
/// test, rather than only as a sentence in the header.
fn maze() -> Array3<f64> {
    let mut values = Array3::from_elem((SLAB[0], SLAB[1], SLAB[2]), 0.0);
    for row in 0..8usize {
        let y = row * 2;
        for x in 0..SLAB[0] {
            values[[x, y, 0]] = 5.0;
        }
        // the turn: the last column on even rows, the first on odd ones
        if row + 1 < 8 {
            let turn = if row % 2 == 0 { SLAB[0] - 1 } else { 0 };
            values[[turn, y + 1, 0]] = 5.0;
        }
    }
    values[[0, 0, 0]] = 10.0;
    values
}

/// The in-plane four-neighbourhood, so that the corridor's one-voxel walls cannot
/// be stepped over diagonally.
fn cross() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Ellipsoid, [1, 1, 0])
}

/// **The runaway guard fires, names the op, and writes nothing** — on a case that
/// genuinely needs more substages than its limit allows rather than on an
/// artificially small number, so the message it produces is the message a caller
/// would actually meet.
#[test]
fn the_limit_fires_by_name_on_a_path_longer_than_the_geometry_predicts() {
    let bound = flooding_bound(SLAB, &cross());
    assert_eq!(bound.substages(), 32, "15 + 15 crossings, plus the two");

    let op = HExtremaOp::maxima("h-maxima", cross(), 1.0, bound).expect("a positive h");
    let plan = plan_with(&op, [8, 8, 1], SLAB);
    let env = ArrayEnvironment::for_decomposition(maze().into(), &plan, [4, 4, 1])
        .expect("an environment");
    let error = execute_phases(
        "reconstruct",
        &empty_workflow(SLAB),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Iterate(&op)],
    )
    .expect_err("a corridor four times the slab's diameter cannot converge in its diameter");
    let message = error.to_string();
    assert!(message.contains("h-maxima"), "{message}");
    assert!(message.contains("32 substage(s)"), "{message}");
    assert!(message.contains("deliberately not written"), "{message}");

    // A partially flooded volume is plausible, well-formed and wrong, so none of
    // it reaches the output: the level is still the unwritten sentinel.
    let out = env.output().view::<f64>().expect("f64").to_owned();
    assert!(
        out.iter().all(|value| value.is_nan()),
        "a truncated answer reached the output level"
    );

    // And raising the limit is the fix the message names. The corridor then floods
    // its whole length, which is what the bound was too small for.
    let raised = HExtremaOp::maxima(
        "h-maxima",
        cross(),
        1.0,
        SubstageLimit::of(SLAB[0] * SLAB[1] + 2).expect("a positive limit"),
    )
    .expect("a positive h");
    let (got, substages) = run(&raised, &maze(), [8, 8, 1]);
    assert!(
        substages > bound.substages(),
        "{substages} substages is within the bound, so this scene proves nothing"
    );
    assert_eq!(
        got[[0, 14, 0]],
        5.0,
        "the far end of the corridor never rose to the mask"
    );
    assert_eq!(got[[0, 0, 0]], 9.0);
}

/// **The peeling bound would refuse a correct run**, which is why this op derives
/// its own rather than taking `PassLimit::for_volume`.
///
/// The two numbers are put beside each other because the mistake is not obvious
/// from either one alone: half the shortest axis is a bound on how far a *surface*
/// can eat inward, and it says nothing whatever about how far a value can travel
/// along a path.
#[test]
fn the_peeling_bound_would_refuse_a_run_this_op_needs() {
    let peeling = PassLimit::for_volume(VOLUME).passes();
    let flooding = flooding_bound(VOLUME, &element()).substages();
    assert_eq!(peeling, 5, "half the shortest axis of {VOLUME:?}, plus two");
    assert_eq!(flooding, 39, "23 + 9 + 5 crossings, plus the two");

    let values = scene();
    let (_, needed) = reference(&op(3.0, generous()), &values);
    assert!(
        needed > peeling,
        "this scene converges in {needed} substages, which the peeling bound of {peeling} \
         would have allowed — the test needs a scene that travels further"
    );
    assert!(
        needed <= flooding,
        "{needed} substages is beyond the flooding bound of {flooding}, which would make the \
         bound wrong rather than the peeling one"
    );

    // And it is not merely a smaller number: a run under it fails.
    let refused = HExtremaOp::maxima(
        "h-maxima",
        element(),
        3.0,
        SubstageLimit::of(peeling).expect("a positive limit"),
    )
    .expect("a positive h");
    let plan = plan(&refused, [8, 4, 2]);
    assert!(
        run_plan(&plan, values.into(), &refused).is_err(),
        "the peeling bound admitted a run that needs more than it allows"
    );
}

// --------------------------------------------------------- 6. non-finite --

/// A mask holding a NaN is refused **by name**, rather than iterating to the
/// limit and reporting a failure to converge.
///
/// The reason is in the module header and is a property of the framework rather
/// than of the operation: the convergence test is `==` on what a substage wrote
/// against what it read, and a NaN is not equal to itself. The refusal is what
/// turns that into a message a caller can act on.
#[test]
fn a_mask_holding_a_nan_is_refused_by_name_rather_than_failing_to_converge() {
    let mut values = scene();
    values[[5, 5, 2]] = f64::NAN;
    let op = op(1.0, generous());
    let plan = plan(&op, [8, 4, 2]);
    let message = run_plan(&plan, values.into(), &op)
        .err()
        .expect("a NaN cannot converge under an equality test")
        .to_string();
    assert!(message.contains("h-maxima"), "{message}");
    assert!(message.contains("NaN"), "{message}");
    assert!(message.contains("not equal to itself"), "{message}");
}

/// The infinities need no rule and are asserted to need none: `+inf` caps
/// nothing, `-inf` floods nothing, and neither disturbs its neighbours.
#[test]
fn the_infinities_behave_as_the_order_says_and_need_no_special_case() {
    let mut values = scene();
    values[[5, 5, 2]] = f64::INFINITY;
    values[[9, 5, 2]] = f64::NEG_INFINITY;
    let op = op(1.0, generous());
    let (got, _) = run(&op, &values, [8, 4, 2]);

    assert_eq!(
        got[[5, 5, 2]],
        f64::INFINITY,
        "an infinite peak lowered by a finite h is still infinite"
    );
    assert_eq!(
        got[[9, 5, 2]],
        f64::NEG_INFINITY,
        "a bottomless voxel caps the flood at its own value"
    );
    // and their neighbours are unharmed: the background around them still floods
    // to its own level from the ridge
    assert_eq!(got[[5, 5, 1]], 0.0);
    assert_eq!(got[[9, 5, 1]], 0.0);
}

// ------------------------------------------------------- 7. composition --

/// A three-phase plan: the h-maxima transform, then the two phases the regional
/// maxima are found with. One run, one environment, three levels.
fn extended_maxima_plan(
    h: f64,
    block: [usize; 3],
    stream: &str,
) -> (Decomposition, HExtremaOp, LabelPlateauxOp, RegionalMaximaOp) {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let hmax = HExtremaOp::maxima("h-maxima", element(), h, generous()).expect("a positive h");
    let label = LabelPlateauxOp::new("label", stream.to_string(), Lifecycle::DeleteOnExit);
    // The faces are written by phase **1** now, not phase 0: the transform is in
    // front of them. A stream is addressed by the phase that wrote it, so this is
    // the one number a caller composing the two has to move.
    let maxima = RegionalMaximaOp::new("maxima", stream.to_string(), 1, Dtype::Bool, &grid);
    let regional =
        regional_phases(grid.clone(), Dtype::F64, &label, &maxima).expect("the regional phases");
    let mut phases: Vec<PhaseDecomposition> =
        vec![iterative_phase(&hmax, grid).expect("an iterative phase")];
    phases.extend(regional.phases);
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: substage_reach(&hmax),
    };
    plan.check().expect("a three-phase plan");
    (plan, hmax, label, maxima)
}

/// **The reason this op exists.** `regional_maxima(HMAX_h(f))` is the extended-
/// maxima transform: a maximum finder with a prominence threshold on it.
///
/// The scene holds eight peaks of prominence 1 to 8, each a single voxel, so the
/// number of flagged voxels at each `h` is a *known number* — the count of peaks
/// whose prominence is above `h` — rather than merely a smaller one. That is the
/// difference between testing the transform and testing that something decreases.
///
/// At `h = 8` no peak survives, the volume is one flat plateau with nothing above
/// it, and **all of it** is maximal: 1440 voxels rather than none. That step is
/// asserted too, because it is the one a reader would get wrong, and because it
/// is the honest answer rather than an edge case to be suppressed.
#[test]
fn extended_maxima_are_the_maxima_of_the_h_maxima_and_the_count_is_known_at_each_h() {
    let values = graded_spikes();
    let whole = VOLUME[0] * VOLUME[1] * VOLUME[2];
    for (h, want) in [
        (0.0, 8usize),
        (1.0, 7),
        (2.0, 6),
        (3.0, 5),
        (5.0, 3),
        (8.0, whole),
    ] {
        let (plan, hmax, label, maxima) =
            extended_maxima_plan(h, [8, 4, 2], &format!("reconstruct.faces.{h}"));
        let env = ArrayEnvironment::for_decomposition(values.clone().into(), &plan, [4, 4, 2])
            .expect("an environment");
        execute_phases(
            "extended maxima",
            &empty_workflow(VOLUME),
            &plan,
            &Hints::default(),
            &env,
            &[],
            &[
                PhaseWork::Iterate(&hmax),
                PhaseWork::Fragments(&label),
                PhaseWork::Fragments(&maxima),
            ],
        )
        .expect("a run");
        let found = env.output().view::<bool>().expect("bool").to_owned();
        let count = found.iter().filter(|&&set| set).count();
        assert_eq!(
            count, want,
            "h = {h} left {count} maximal voxels and the scene has {want} above it"
        );

        // and they are the *right* ones: exactly the peaks above h, each still a
        // single voxel, plus the whole-volume plateau in the degenerate case
        if h < 8.0 {
            for (at, height) in spikes() {
                assert_eq!(
                    found[at],
                    height > h,
                    "the peak of prominence {height} under h = {h}"
                );
            }
            assert!(
                !found[[0, 0, 0]],
                "the base is not a maximum while a peak stands on it"
            );
        } else {
            assert!(
                found.iter().all(|&set| set),
                "with every peak flattened the volume is one plateau, and all of it is maximal"
            );
        }
    }
}

/// The composition against the same two ops run separately over the whole volume:
/// the three-phase plan is not computing something of its own.
#[test]
fn the_composed_plan_agrees_with_the_two_transforms_run_by_hand() {
    let values = graded_spikes();
    for h in [1.0, 3.0] {
        let (transformed, _) = h_extrema(
            values.view(),
            &element(),
            Reconstruction::ByDilation,
            h,
            generous(),
        )
        .expect("a whole-volume transform");
        let expected = regional_maxima(transformed.view()).expect("the whole-volume maxima");

        let (plan, hmax, label, maxima) =
            extended_maxima_plan(h, [12, 5, 3], &format!("reconstruct.pair.{h}"));
        let env = ArrayEnvironment::for_decomposition(values.clone().into(), &plan, [4, 4, 2])
            .expect("an environment");
        execute_phases(
            "extended maxima",
            &empty_workflow(VOLUME),
            &plan,
            &Hints::default(),
            &env,
            &[],
            &[
                PhaseWork::Iterate(&hmax),
                PhaseWork::Fragments(&label),
                PhaseWork::Fragments(&maxima),
            ],
        )
        .expect("a run");
        assert_eq!(
            env.output().view::<bool>().expect("bool").to_owned(),
            expected,
            "h = {h}"
        );
    }
}
