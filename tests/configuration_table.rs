// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `ops::configuration`: a mask rewritten by a table
// indexed on the 3x3x3 neighbourhood, in both of its shells.
//
// The bar is `tests/image_ops.rs`'s — **decomposition invariance against a
// whole-volume reference** — and this op is where that bar is interesting,
// because its reach is not a property of the op at all in one shell and is the
// whole parameter in the other:
//
// | shell | reach | what a wrong halo would look like |
// |---|---|---|
// | [`ConfigurationPassOp`] at `n` passes | `n` | correct everywhere except within `n` voxels of a seam, which is a thin shell of plausible values |
// | [`ConfigurationFixedPointOp`] | 1, at any depth | the same, but the depth is discovered from the data and no plan mentions it |
//
// So the properties here are:
//
// 1. **Both shells are decomposition invariant**, at several block edges and
//    split axes, byte-identical to the same kernel run once over everything.
// 2. **The pass count is really in the reach.** Understating it produces a wrong
//    volume that still tiles — the silent failure — which is what makes
//    `reach = passes` a fact rather than a safe over-declaration. The
//    understatement is searched for rather than assumed, and on this fixture the
//    smallest one that shows is **two** voxels, not one; a decomposition leaves
//    a voxel of slack somewhere, and the test says the measured number rather
//    than the hoped-for one.
// 3. **The fixed point's reach does not grow with its depth**, and the depth
//    follows the data rather than the plan.
// 4. **The two shells meet**: a fixed point is what a stated pass count settles
//    on once the count is past it.
// 5. **A table that does not converge ends at the limit**, naming the op, rather
//    than at a plausible partially-settled volume.
//
// And beside every one of them, a **liveness** figure: how many voxels the table
// actually moved. A table that moves nothing is decomposition invariant, agrees
// with every reference, and proves nothing at all, so each test that compares
// asserts the comparison had something to say.

use std::sync::Arc;

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::fragment::PhaseWork;
use blockflow::geometry::BlockGrid;
use blockflow::iterate::{iterative_phase, substage_reach, SubstageLimit};
use blockflow::op::Chain;
use blockflow::ops::configuration::{
    configuration_passes_into, configuration_to_fixed_point, ConfigurationFixedPointOp,
    ConfigurationPassOp, ConfigurationTable, ConfigurationTemplate,
};
use blockflow::strategy::{execute, execute_phases, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::Dtype;

const VOLUME: [usize; 3] = [32, 24, 20];

// ------------------------------------------------------------- fixtures --

/// A mask with structure at several scales **and speckle on top of it**, so a
/// neighbourhood rule has plenty to do and a seam has plenty to get wrong.
///
/// The speckle is the part that matters for liveness: a rule that tidies
/// boundaries moves a few hundred voxels of a smooth mask and tens of thousands
/// of a noisy one, and a comparison is only worth as much as the number of
/// voxels it had an opinion about.
fn mask() -> Array3<bool> {
    let scene = Scene::new(
        SceneSpec::new(VOLUME, 20250811)
            .with_objects(40)
            .with_radius(1.5, 4.0)
            .with_noise(0.02),
    )
    .unwrap();
    let rendered = scene.render();
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        // A deterministic generator, written out so the fixture does not depend
        // on a crate that might reseed itself between versions.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let flip = state % 10 == 0;
        (rendered.intensity[[i, j, k]] > 0.35) != flip
    })
}

/// **Majority over the seven-voxel face neighbourhood**: the voxel and its six
/// face neighbours, set where four or more of the seven are.
///
/// The workhorse of the invariance tests. It moves a great many voxels of a
/// speckled mask on the first pass and keeps moving them for several more, and
/// it is stated as 128 templates — one per pattern of the seven — so the table
/// is complete by construction rather than by an argument about what the
/// remaining configurations do.
fn majority_faces() -> ConfigurationTable {
    let positions = face_neighbourhood();
    let mut table = ConfigurationTable::zeros();
    for pattern in 0u32..(1 << positions.len()) {
        if pattern.count_ones() < 4 {
            continue;
        }
        let mut template = ConfigurationTemplate::any();
        for (bit, offset) in positions.iter().enumerate() {
            template = template.with(*offset, pattern >> bit & 1 == 1).unwrap();
        }
        table.assign_matching(&template, true);
    }
    table
}

/// The voxel and its six face neighbours, in a fixed order.
fn face_neighbourhood() -> [[isize; 3]; 7] {
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
}

/// **A shadow cast along axis 0**: a clear voxel becomes set when the voxel
/// below it on axis 0 is set.
///
/// The table the fixed-point tests use, and it was chosen for three properties
/// no smoothing rule has all of:
///
/// * it is **monotone** — a pass only ever sets voxels — so the sequence
///   increases, is bounded by the volume, and therefore terminates. That is a
///   property of this table rather than of the op, which is the point
///   [`ConfigurationFixedPointOp::new`] makes by refusing to derive a limit;
/// * its fixed point is **deep and controllable**: information travels one voxel
///   per pass along axis 0, so the depth is the distance from the highest set
///   voxel to the far face and a test can set it by placing one voxel;
/// * its fixed point is **known independently** — the running maximum along axis
///   0 — so the comparison is against the answer rather than only against the
///   same kernel run differently.
fn shadow() -> ConfigurationTable {
    let mut table = ConfigurationTable::identity();
    table.assign_matching(
        &ConfigurationTemplate::any()
            .with_clear([0, 0, 0])
            .unwrap()
            .with_set([-1, 0, 0])
            .unwrap(),
        true,
    );
    table
}

/// **A spread in all six face directions**: a clear voxel becomes set when any
/// face neighbour is set.
///
/// The table the reach test uses, and it is the right one for that job because
/// it moves information one voxel per pass **on every axis**. A rule that
/// happens to be insensitive on some axis would let an understated halo pass
/// unnoticed there, and the property being tested is about the op's declaration
/// rather than about any one table's sensitivities.
fn dilate_faces() -> ConfigurationTable {
    let mut table = ConfigurationTable::identity();
    for offset in face_neighbourhood().into_iter().skip(1) {
        table.assign_matching(
            &ConfigurationTemplate::any()
                .with_clear([0, 0, 0])
                .unwrap()
                .with_set(offset)
                .unwrap(),
            true,
        );
    }
    table
}

/// What [`shadow`] settles on, stated without running it: every voxel at or
/// beyond a set voxel along axis 0.
fn shadow_answer(input: &Array3<bool>) -> Array3<bool> {
    let mut out = input.clone();
    for i in 1..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                out[[i, j, k]] = out[[i, j, k]] || out[[i - 1, j, k]];
            }
        }
    }
    out
}

/// Every configuration maps to the complement of its centre: period two on any
/// non-empty volume, and the reason a limit is required rather than derived.
fn alternating() -> ConfigurationTable {
    let mut table = ConfigurationTable::identity();
    let centre = ConfigurationTemplate::any().with_set([0, 0, 0]).unwrap();
    table.assign_matching(&centre, false);
    table.assign_matching(&centre.inverted(), true);
    table
}

fn moved(before: &Array3<bool>, after: &Array3<bool>) -> usize {
    before
        .iter()
        .zip(after.iter())
        .filter(|(a, b)| a != b)
        .count()
}

// ------------------------------------------------------ the pass shell --

fn pass_workflow(table: &Arc<ConfigurationTable>, passes: usize) -> Workflow {
    Workflow::new(
        Chain::op(ConfigurationPassOp::new(
            "configuration",
            table.clone(),
            passes,
        )),
        VOLUME,
        Dtype::Bool,
    )
}

fn plan_with_reach(
    workflow: &Workflow,
    block: usize,
    split_axes: &[usize],
    reach: [usize; 3],
) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(VOLUME, split_axes, block).unwrap();
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid)
        .with_dtype(Dtype::Bool);
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::Bool,
        phases: vec![phase],
        chain_reach: reach,
    }
}

fn plan(workflow: &Workflow, block: usize, split_axes: &[usize]) -> Decomposition {
    plan_with_reach(workflow, block, split_axes, workflow.chain.reach3(&VOLUME))
}

fn run_pass(
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Array3<bool>,
) -> Array3<bool> {
    let env = ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4])
        .expect("an environment");
    execute(
        "configuration",
        workflow,
        decomposition,
        &Hints::default(),
        &env,
    )
    .expect("a run");
    env.output().view::<bool>().unwrap().to_owned()
}

/// The whole-volume reference: the same kernel, once, over everything.
fn reference(table: &ConfigurationTable, input: &Array3<bool>, passes: usize) -> Array3<bool> {
    let mut out = Array3::from_elem(input.raw_dim(), false);
    configuration_passes_into(input.view(), table, passes, out.view_mut()).expect("a reference");
    out
}

/// Property 1, for the pass shell, and the liveness figure beside it.
#[test]
fn a_stated_pass_count_is_decomposition_invariant() {
    let table = Arc::new(majority_faces());
    let input = mask();
    let set = input.iter().filter(|value| **value).count();
    assert!(set > 1000, "the fixture must have something in it: {set}");

    for passes in [1usize, 2, 3, 6] {
        let want = reference(&table, &input, passes);
        let changed = moved(&input, &want);
        assert!(
            changed > 100,
            "at {passes} pass(es) the table moved {changed} voxels, which is too few for the \
             comparisons below to mean anything"
        );
        println!(
            "passes {passes}: {changed} of {} voxels moved ({} set before, {} after)",
            input.len(),
            set,
            want.iter().filter(|value| **value).count()
        );

        let workflow = pass_workflow(&table, passes);
        assert_eq!(workflow.chain.reach3(&VOLUME), [passes, passes, passes]);
        for block in [8usize, 12, 16, 20] {
            for axes in [&[0usize][..], &[0, 1][..], &[0, 1, 2][..]] {
                let decomposition = plan(&workflow, block, axes);
                let got = run_pass(&workflow, &decomposition, &input);
                assert_eq!(
                    got,
                    want,
                    "at {passes} pass(es), block {block}, axes {axes:?}: \
                     {} voxels differ from the whole-volume answer",
                    moved(&got, &want)
                );
            }
        }
    }
}

/// Property 2. **The reach is `passes` and not a safe over-declaration**: a plan
/// one voxel short still tiles, still runs, and is wrong near the seams.
///
/// This is the failure the derived reach exists to prevent, and a guard nobody
/// has watched fail is not known to work.
#[test]
fn understating_the_reach_produces_a_wrong_volume_that_still_tiles() {
    let table = Arc::new(dilate_faces());
    let input = mask();
    let passes = 4usize;
    let want = reference(&table, &input, passes);
    let workflow = pass_workflow(&table, passes);

    // The smallest understatement that shows, searched for rather than assumed:
    // `tests/image_ops.rs` searches the same way, because a block edge that
    // happens to fall where the data is quiet hides a real shortfall.
    let mut smallest = None;
    'search: for short_by in 1..=passes {
        for axis in 0..3 {
            for block in [8usize, 12] {
                let mut short = [passes; 3];
                short[axis] = passes - short_by;
                let lying = plan_with_reach(&workflow, block, &[axis], short);
                lying
                    .check()
                    .expect("an understated reach still tiles — that is the danger");
                let got = run_pass(&workflow, &lying, &input);
                let differing = moved(&got, &want);
                if differing > 0 {
                    assert!(
                        differing < got.len(),
                        "everything differs on axis {axis}, so this is not a seam effect"
                    );
                    println!(
                        "reach {passes} short by {short_by} on axis {axis}, block {block}: \
                         {differing} voxels wrong"
                    );
                    smallest = Some(short_by);
                    break 'search;
                }
            }
        }
    }
    let smallest = smallest.unwrap_or_else(|| {
        panic!(
            "no understatement of a reach of {passes} produced a wrong value under any \
             decomposition here, so nothing this op reads matters and its reach describes \
             something else"
        )
    });
    assert!(
        smallest <= 2,
        "the reach of {passes} had to be cut by {smallest} before it mattered, which is more \
         slack than a per-pass derivation should leave"
    );
}

// ----------------------------------------------- the fixed-point shell --

fn iterate_plan(
    op: &ConfigurationFixedPointOp,
    block: usize,
    split_axes: &[usize],
) -> Decomposition {
    let grid = BlockGrid::along(VOLUME, split_axes, block).unwrap();
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::Bool,
        phases: vec![iterative_phase(op, grid).expect("an iterative phase")],
        chain_reach: [1, 1, 1],
    }
}

/// An iterative phase owns no chain slot, so the workflow it runs under has none.
fn empty_workflow() -> Workflow {
    Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::Bool)
}

fn run_iterate(
    op: &ConfigurationFixedPointOp,
    decomposition: &Decomposition,
    input: &Array3<bool>,
) -> (Array3<bool>, usize) {
    let env = ArrayEnvironment::for_decomposition(input.clone().into(), decomposition, [8, 4, 4])
        .expect("an environment");
    let stats = execute_phases(
        "configuration",
        &empty_workflow(),
        decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Iterate(op)],
    )
    .expect("a run");
    (
        env.output().view::<bool>().unwrap().to_owned(),
        stats.substages[0],
    )
}

fn generous() -> SubstageLimit {
    SubstageLimit::of(200).expect("a positive limit")
}

/// Properties 1 and 3 for the fixed-point shell: invariant across
/// decompositions, and its reach is one pass whatever depth the data asks for.
#[test]
fn the_fixed_point_is_decomposition_invariant_at_the_reach_of_one_pass() {
    let table = Arc::new(shadow());
    let input = mask();
    let op = ConfigurationFixedPointOp::new("configuration", table.clone(), generous());

    assert_eq!(
        substage_reach(&op),
        [1, 1, 1],
        "one pass reads the 3x3x3 neighbourhood, and the substage count multiplies nothing"
    );

    let (want, depth) = configuration_to_fixed_point(input.view(), &table, generous())
        .expect("a monotone table settles");
    // The answer, stated independently of the iteration that produced it.
    assert_eq!(want, shadow_answer(&input));
    let changed = moved(&input, &want);
    assert!(
        depth > 2,
        "the fixed point was reached in {depth} pass(es), which is too shallow to say \
         anything about depth"
    );
    assert!(changed > 100, "the table moved {changed} voxels");
    println!(
        "fixed point: {depth} passes, {changed} of {} voxels moved",
        input.len()
    );

    for block in [8usize, 12, 16, 20] {
        for axes in [&[0usize][..], &[0, 1][..], &[0, 1, 2][..]] {
            let decomposition = iterate_plan(&op, block, axes);
            // The plan names no substage count, and its halo is one pass's.
            assert_eq!(decomposition.phases[0].reach, [1, 1, 1]);
            assert_eq!(decomposition.phases[0].halo, [1, 1, 1]);
            let (got, substages) = run_iterate(&op, &decomposition, &input);
            assert_eq!(
                got,
                want,
                "block {block}, axes {axes:?}: {} voxels differ from the whole-volume answer",
                moved(&got, &want)
            );
            assert_eq!(
                substages, depth,
                "block {block}, axes {axes:?}: the depth is a fact about the data and the \
                 whole-volume run found {depth}"
            );
        }
    }
}

/// Property 3's other half: the depth follows the data, and the plan does not
/// change when it does.
///
/// One set voxel, moved along axis 0. The shadow travels one voxel per pass, so
/// how far it has to travel is how deep the iteration is — and the plan is the
/// same object in both runs.
#[test]
fn the_substage_count_follows_the_data_and_not_the_plan() {
    let table = Arc::new(shadow());
    let op = ConfigurationFixedPointOp::new("configuration", table.clone(), generous());
    let decomposition = iterate_plan(&op, 12, &[0]);

    let mut depths = Vec::new();
    for start in [VOLUME[0] - 3, 1] {
        let mut input = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
        input[[start, 12, 10]] = true;
        let (got, substages) = run_iterate(&op, &decomposition, &input);
        assert_eq!(got, shadow_answer(&input), "from {start}");
        depths.push(substages);
    }
    assert!(
        depths[1] > depths[0] + 10,
        "a shadow starting at 1 took {} passes and one starting near the far face {}; the \
         depth is supposed to be a fact about the data",
        depths[1],
        depths[0]
    );
    println!("depth by starting voxel: {depths:?}");
}

/// Property 4. The two shells meet: iterating to a fixed point is what a stated
/// count settles on, and stating a larger count changes nothing.
#[test]
fn a_fixed_point_is_what_a_stated_pass_count_settles_on() {
    let table = Arc::new(shadow());
    let input = mask();
    let (settled, depth) = configuration_to_fixed_point(input.view(), &table, generous())
        .expect("a monotone table settles");

    // One pass short is not there yet, and every count past it is.
    let short = reference(&table, &input, depth - 2);
    assert_ne!(
        short, settled,
        "the fixed point must be later than {depth} - 2 passes"
    );
    for passes in [depth - 1, depth, depth + 3] {
        assert_eq!(
            reference(&table, &input, passes),
            settled,
            "at {passes} pass(es)"
        );
    }
}

/// Property 5. A table with no fixed point ends at the limit, naming the op,
/// rather than at a partially settled volume.
#[test]
fn a_table_that_does_not_converge_is_refused_by_name_rather_than_truncated() {
    let table = Arc::new(alternating());
    let limit = SubstageLimit::of(9).expect("a positive limit");
    let op = ConfigurationFixedPointOp::new("alternating", table, limit);
    let decomposition = iterate_plan(&op, 12, &[0]);
    let input = mask();

    let env = ArrayEnvironment::for_decomposition(input.into(), &decomposition, [8, 4, 4])
        .expect("an environment");
    let error = execute_phases(
        "configuration",
        &empty_workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Iterate(&op)],
    )
    .expect_err("a table with period two cannot converge")
    .to_string();
    assert!(error.contains("alternating"), "{error}");
    assert!(error.contains("9 substage"), "{error}");
}

/// The convergence predicate is **global**, and this is what that buys.
///
/// One region of the volume settles at once and another takes many passes. Every
/// block runs every substage until the slowest region is done — so the answer in
/// the quiet region is the same as if it had been the only thing in the volume,
/// which is what "no voxel anywhere changed" means and what a per-block fixed
/// point would get wrong: a block that stopped early would freeze its own edge
/// and starve its neighbour of the values the neighbour is still spreading.
#[test]
fn convergence_is_decided_over_the_whole_volume_and_not_per_block() {
    let table = Arc::new(shadow());
    let op = ConfigurationFixedPointOp::new("configuration", table.clone(), generous());

    // One line that is finished after two passes, and one that takes the length
    // of the axis. Both are cut across several blocks.
    let mut input = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    input[[VOLUME[0] - 2, 6, 10]] = true;
    input[[1, 18, 10]] = true;

    let (want, depth) = configuration_to_fixed_point(input.view(), &table, generous())
        .expect("a monotone table settles");
    assert_eq!(want, shadow_answer(&input));
    assert!(depth > 20, "the slow line must be genuinely slow: {depth}");

    let decomposition = iterate_plan(&op, 8, &[0]);
    let (got, substages) = run_iterate(&op, &decomposition, &input);
    assert_eq!(substages, depth);
    assert_eq!(
        got,
        want,
        "{} voxels differ; a per-block fixed point would show up here",
        moved(&got, &want)
    );

    // The quiet line stopped moving after two passes and the run went on for
    // `depth`, which is what makes the predicate global rather than local: the
    // blocks that hold it kept being run, and kept agreeing.
    let quiet =
        |volume: &Array3<bool>| -> usize { (0..VOLUME[0]).filter(|&i| volume[[i, 6, 10]]).count() };
    assert_eq!(quiet(&reference(&table, &input, 2)), quiet(&want));
    assert_eq!(quiet(&want), 2);
}
