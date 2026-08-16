// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The framework's acceptance tests. `docs/design/BLOCK_OPS.md` asks for four
// things, and they are separated here on purpose because each catches a
// different class of bug:
//
// 1. **Planner decisions** — synthetic reaches and costs whose optimal
//    partition is computable by hand.
// 2. **Identity ops as a whole-framework oracle** — if every op is the
//    identity, output must equal input *exactly*, swept across block sizes,
//    halos, partitions, iteration orders and worker counts. No reference
//    implementation; the expected answer is the input.
// 3. **A window-sum op to detect an insufficient halo**, failing *both*
//    structurally (the tiling check fires) and behaviourally (values diverge).
//    A guard that has never been seen to fire is not known to work — the
//    `zarrs` chunk race matched digests by luck until someone provoked it.
// 4. **The plan was honoured**, asserted from the event log: every block
//    affected by every op, in chain order, once each; plus counts of ops,
//    blocks, materialisations and reads through the counting loader.
//
// And the conformance suite the design asks for on top: every strategy's output
// equals `Trivial`'s, and one strategy's `run` against another's `decompose`.

use ndarray::Array3;

use crate::dtype::Dtype;
use crate::region::Region;

use super::decomposition::{Constraints, CostModel, Decomposition, PhaseDecomposition};
use super::env::{AccountingEnvironment, ArrayEnvironment, Environment};
use super::geometry::BlockGrid;
use super::graph::TaskGraph;
use super::log::Event;
use super::op::Chain;
use super::probes::{AffineOp, IdentityOp, OpaqueOp, WindowSumOp};
use super::strategy::{
    execute, Enumerating, Greedy, Hints, PartitionSearch, SchedulePriority, Strategy, Trivial,
    Workflow,
};

// ------------------------------------------------------------- fixtures --

fn ramp(shape: [usize; 3]) -> Array3<f64> {
    let mut array = Array3::zeros((shape[0], shape[1], shape[2]));
    for (flat, value) in array.iter_mut().enumerate() {
        *value = ((flat * 7919) % 1013) as f64 + 1.0;
    }
    array
}

/// The last level as `f64`, which is what every test in this file works in.
///
/// A level's element type is a tag now, so a test that wants `f64` says so
/// once here rather than at every assertion.
fn output(env: &ArrayEnvironment) -> Array3<f64> {
    env.output()
        .view::<f64>()
        .expect("these tests run f64 levels")
        .to_owned()
}

fn noop(name: &'static str, reach: usize, cost: f64) -> Chain {
    Chain::op(IdentityOp::new(name, [reach, reach, reach]).with_cost(cost))
}

fn workflow(chain: Chain, shape: [usize; 3]) -> Workflow {
    Workflow::new(chain, shape, Dtype::F64)
}

fn constraints(split_axes: Vec<usize>, candidates: Vec<usize>) -> Constraints {
    Constraints {
        budget_bytes: None,
        expected_concurrency: 1,
        model: CostModel::default(),
        block_candidates: candidates,
        split_axes,
    }
}

/// The chain order the log must reproduce: `(slot index, op name)`.
fn expected_sequence(chain: &Chain) -> Vec<(usize, String)> {
    chain
        .slots()
        .iter()
        .enumerate()
        .map(|(slot, sub)| (slot, sub.display_name()))
        .collect()
}

// ================================================================ 1. plan ==

/// The design's own adversarial case: an op with huge reach between two with
/// none. The cut should isolate it.
///
/// The neighbours are given real compute cost, and that is not test rigging —
/// redundancy is a *multiplier on compute*, so isolating a wide op saves in
/// proportion to the work it would otherwise multiply. With free neighbours
/// there is nothing to save and fusing is genuinely right; that half of the
/// property is asserted in `decomposition::tests`.
#[test]
fn the_planner_isolates_a_high_reach_op_between_two_expensive_ones() {
    let chain = Chain::sequence(vec![
        noop("a", 0, 5.0),
        noop("b", 50, 1.0),
        noop("c", 0, 5.0),
    ]);
    let workflow = workflow(chain, [1024, 64, 64]);
    let decomposition = Enumerating::default()
        .decompose(&workflow, &constraints(vec![0], vec![64]))
        .unwrap();
    assert_eq!(
        decomposition
            .phases
            .iter()
            .map(|phase| phase.slots.clone())
            .collect::<Vec<_>>(),
        vec![vec![0], vec![1], vec![2]],
        "expected the 50-reach op isolated, got {:?}",
        decomposition.op_names_in_order()
    );
    assert_eq!(decomposition.phases[0].reach, [0, 0, 0]);
    assert_eq!(decomposition.phases[1].reach, [50, 50, 50]);
    // only axis 0 is split, so only axis 0's reach costs anything
    assert_eq!(decomposition.phases[1].grid.split_axes(), &[0]);
}

/// All reaches zero: there is no redundancy to avoid, so every cut is a pure
/// extra read and write. Fusing everything wins.
#[test]
fn the_planner_fuses_everything_when_no_op_has_reach() {
    let chain = Chain::sequence(vec![
        noop("a", 0, 1.0),
        noop("b", 0, 1.0),
        noop("c", 0, 1.0),
        noop("d", 0, 1.0),
    ]);
    let workflow = workflow(chain, [512, 64, 64]);
    let decomposition = Enumerating::default()
        .decompose(&workflow, &constraints(vec![0], vec![64]))
        .unwrap();
    assert_eq!(decomposition.n_phases(), 1);
    assert_eq!(decomposition.phases[0].slots, vec![0, 1, 2, 3]);
}

/// Every op with a large reach: fusing multiplies the redundancy (reaches add),
/// so cutting everywhere wins despite paying three materialisations.
#[test]
fn the_planner_cuts_everywhere_when_every_op_has_a_large_reach() {
    let chain = Chain::sequence(vec![
        noop("a", 40, 1.0),
        noop("b", 40, 1.0),
        noop("c", 40, 1.0),
    ]);
    let workflow = workflow(chain, [2048, 64, 64]);
    let decomposition = Enumerating::default()
        .decompose(&workflow, &constraints(vec![0], vec![64]))
        .unwrap();
    assert_eq!(decomposition.n_phases(), 3);
}

/// A full-reach op is a **planning barrier**, so the planner segments at it
/// rather than fusing across it.
///
/// The measured defect this pins, over a `[32, 4, 4]` volume with candidates
/// `[8, 16, 32]`: `identity(0) > identity(volume) > identity(0)` decomposed into
/// **one** phase, block `[32, 4, 4]`, reach `[32, 4, 4]`, modelled total
/// **2560** — both cheap local ops dragged into a single-block phase whose
/// working set is the whole volume, priced at redundancy **1.0** because
/// `BlockGrid` drops an axis from `split_axes` when the block spans the volume.
///
/// Two things changed, and the test asserts both because either alone would
/// leave a hole: the partition is now constrained *structurally*
/// (`is_planning_barrier`), and the price of the fused partition is no longer a
/// lie — that same one-phase plan now costs **55808** against **31232** for the
/// three-phase one, so the cost model and the structure agree rather than the
/// structure overruling it.
#[test]
fn the_planner_segments_at_a_full_reach_op_rather_than_fusing_across_it() {
    let volume = [32usize, 4, 4];
    let chain = Chain::sequence(vec![
        noop("before", 0, 1.0),
        Chain::op(IdentityOp::new("global", volume)),
        noop("after", 0, 1.0),
    ]);
    let workflow = workflow(chain, volume);
    let decomposition = Enumerating::default()
        .decompose(&workflow, &constraints(vec![0], vec![8, 16, 32]))
        .unwrap();

    assert_eq!(
        decomposition
            .phases
            .iter()
            .map(|phase| phase.slots.clone())
            .collect::<Vec<_>>(),
        vec![vec![0], vec![1], vec![2]],
        "expected the full-reach op alone in its phase, got {:?}",
        decomposition.op_names_in_order()
    );
    // the barrier phase must span the volume — any smaller block reads less
    // than the op needs on every voxel — and the local phases keep their own
    assert_eq!(decomposition.phases[1].reach, volume);
    assert_eq!(decomposition.phases[1].grid.n_blocks(), 1);
    assert_eq!(decomposition.phases[0].reach, [0, 0, 0]);
    decomposition.check().unwrap();

    // and the price of what it rejected, in the model's own numbers
    let model = CostModel::default();
    let whole = BlockGrid::whole(volume).unwrap();
    let price = |reach: [usize; 3], compute: f64, materialised: bool| {
        super::decomposition::price_phase(
            &whole,
            &reach.into(),
            compute,
            1,
            materialised,
            8.0,
            &model,
            1.0,
        )
        .cost_per_block
    };
    let fused = price(volume, 3.0, false);
    let segmented =
        price([0, 0, 0], 1.0, true) + price(volume, 1.0, true) + price([0, 0, 0], 1.0, false);
    assert!(
        segmented < fused,
        "fusing across the barrier priced {fused}, segmenting {segmented}"
    );
    assert_eq!(
        (fused, segmented),
        (55808.0, 31232.0),
        "the recorded figures moved"
    );
}

/// The over-firing this must not do: a **bounded** reach is not a barrier
/// however large it is, and still fuses where fusing is right.
///
/// `docs/design/GRAPH_MIGRATION.md` measures the case: of seven merge steps in
/// one real chain, two reach a single voxel and are not barriers at all. A rule
/// of the form "a reach above some fraction of the volume is a barrier" would
/// segment here — 512 is a large absolute reach, an eighth of the volume, and
/// the fused phase's halo is real — and it would be wrong: fusing is 1.22x
/// cheaper in the model, and the phase is still cut into blocks rather than
/// forced to span the volume.
#[test]
fn a_large_but_bounded_reach_is_not_a_barrier_and_still_fuses() {
    let volume = [4096usize, 4, 4];
    let bounded = |reach: usize| {
        let chain = Chain::sequence(vec![
            noop("before", 0, 1.0),
            Chain::op(IdentityOp::new("wide", [reach, 0, 0])),
            noop("after", 0, 1.0),
        ]);
        Enumerating::default()
            .decompose(&workflow(chain, volume), &constraints(vec![0], vec![1024]))
            .unwrap()
    };

    for reach in [1usize, 512] {
        let decomposition = bounded(reach);
        assert_eq!(
            decomposition.n_phases(),
            1,
            "a bounded reach of {reach} over {volume:?} was segmented into {:?}",
            decomposition
                .phases
                .iter()
                .map(|phase| phase.names.clone())
                .collect::<Vec<_>>()
        );
        assert!(decomposition.phases[0].grid.n_blocks() > 1);
        assert_eq!(decomposition.phases[0].grid.split_axes(), &[0]);
    }

    // the same chain with the reach taken to the volume: now it segments, and
    // the difference between the two is one voxel of reach, not a threshold
    let full = Chain::sequence(vec![
        noop("before", 0, 1.0),
        Chain::op(IdentityOp::new("global", [volume[0], 0, 0])),
        noop("after", 0, 1.0),
    ]);
    let segmented = Enumerating::default()
        .decompose(&workflow(full, volume), &constraints(vec![0], vec![1024]))
        .unwrap();
    assert_eq!(segmented.n_phases(), 3);
    assert_eq!(segmented.phases[1].grid.n_blocks(), 1);

    let nearly = Chain::sequence(vec![
        noop("before", 0, 1.0),
        Chain::op(IdentityOp::new("nearly", [volume[0] - 1, 0, 0])),
        noop("after", 0, 1.0),
    ]);
    let nearly = Enumerating::default()
        .decompose(&workflow(nearly, volume), &constraints(vec![0], vec![1024]))
        .unwrap();
    // priced out of fusing, but not *forbidden* from it: the phase carrying it
    // may still be cut, which is the whole difference from a barrier
    assert_eq!(nearly.phases[1].grid.split_axes(), &[0]);
}

/// The second planner implements the same position, by its own route.
#[test]
fn greedy_also_segments_at_a_full_reach_op() {
    let volume = [64usize, 8, 8];
    let chain = Chain::sequence(vec![
        noop("before", 0, 1.0),
        Chain::op(IdentityOp::new("global", volume)),
        noop("after", 0, 1.0),
        noop("also_after", 0, 1.0),
    ]);
    let decomposition = Greedy::default()
        .decompose(
            &workflow(chain, volume),
            &constraints(vec![0], vec![16, 32]),
        )
        .unwrap();
    assert_eq!(
        decomposition
            .phases
            .iter()
            .map(|phase| phase.slots.clone())
            .collect::<Vec<_>>(),
        vec![vec![0], vec![1], vec![2, 3]],
        "greedy fused across the barrier: {:?}",
        decomposition.op_names_in_order()
    );
    // the local phases are blocked; the barrier phase spans the volume
    assert!(decomposition.phases[0].grid.n_blocks() > 1);
    assert_eq!(decomposition.phases[1].grid.n_blocks(), 1);
    decomposition.check().unwrap();
}

/// Ties must resolve the same way every time, or no two runs are comparable.
#[test]
fn ties_are_broken_deterministically() {
    let chain = || {
        Chain::sequence(vec![
            noop("a", 10, 1.0),
            noop("b", 10, 1.0),
            noop("c", 10, 1.0),
        ])
    };
    let first = Enumerating::default()
        .decompose(
            &workflow(chain(), [512, 64, 64]),
            &constraints(vec![0], vec![64]),
        )
        .unwrap();
    for _ in 0..8 {
        let again = Enumerating::default()
            .decompose(
                &workflow(chain(), [512, 64, 64]),
                &constraints(vec![0], vec![64]),
            )
            .unwrap();
        assert_eq!(first.fingerprint(), again.fingerprint());
        assert_eq!(first, again);
    }
}

/// The budget is what makes full fusion infeasible, and the planner must cut
/// rather than plan something that will not fit.
#[test]
fn a_memory_budget_forces_cuts_the_cost_model_would_not_choose() {
    let chain = Chain::sequence(vec![
        noop("a", 8, 1.0),
        noop("b", 8, 1.0),
        noop("c", 8, 1.0),
    ]);
    let workflow = workflow(chain, [1024, 32, 32]);

    let unbounded = Enumerating::default()
        .decompose(&workflow, &constraints(vec![0], vec![128]))
        .unwrap();
    assert_eq!(unbounded.n_phases(), 1, "unbounded, the cost model fuses");

    // One phase of three ops reaches 24, so a block of 128 reads 176 planes:
    // 176 x 32 x 32 x 8 bytes x 2 buffers = 2.88 MB. A single op reaches 8 and
    // reads 144 planes: 2.36 MB. Budget between them.
    let mut tight = constraints(vec![0], vec![128]);
    tight.budget_bytes = Some(2_500_000);
    let bounded = Enumerating::default().decompose(&workflow, &tight).unwrap();
    assert!(
        bounded.n_phases() > 1,
        "the budget should have forced a cut, got {} phase(s)",
        bounded.n_phases()
    );
    for phase in &bounded.phases {
        let read: usize = phase.blocks.iter().map(|b| b.read.voxels()).max().unwrap();
        assert!((read * 8 * 2) as u64 <= 2_500_000);
    }
}

#[test]
fn an_impossible_budget_is_refused_with_a_number_to_act_on() {
    let chain = Chain::sequence(vec![noop("a", 4, 1.0)]);
    let workflow = workflow(chain, [256, 64, 64]);
    let mut impossible = constraints(vec![0], vec![64]);
    impossible.budget_bytes = Some(1024);
    let err = Enumerating::default()
        .decompose(&workflow, &impossible)
        .unwrap_err();
    let message = err.to_string();
    assert!(message.contains("byte budget"), "got: {message}");
    assert!(message.contains("block candidates"), "got: {message}");
}

/// The traversal-disagreement signal only bites when a measurement backs it, so
/// the penalty defaults to zero and must be turned on explicitly.
#[test]
fn traversal_disagreement_becomes_a_cut_only_when_the_penalty_is_set() {
    let chain = || {
        Chain::sequence(vec![
            Chain::op(IdentityOp::new("xy", [0, 0, 0]).with_order([2, 0, 1])),
            Chain::op(IdentityOp::new("z", [0, 0, 0]).with_order([0, 1, 2])),
        ])
    };
    let shape = [512, 64, 64];

    let neutral = Enumerating::default()
        .decompose(&workflow(chain(), shape), &constraints(vec![0], vec![64]))
        .unwrap();
    assert_eq!(neutral.n_phases(), 1, "no measurement, no cut");

    let mut penalised = constraints(vec![0], vec![64]);
    penalised.model.order_conflict_penalty = 100.0;
    let cut = Enumerating::default()
        .decompose(&workflow(chain(), shape), &penalised)
        .unwrap();
    assert_eq!(cut.n_phases(), 2);
    assert_eq!(cut.phases[0].slots, vec![0]);
    assert_eq!(cut.phases[1].slots, vec![1]);
}

/// The infinite-grid assumption must err towards over-reading, never under.
#[test]
fn the_cost_model_never_predicts_fewer_reads_than_the_run_performs() {
    let chain = Chain::sequence(vec![noop("a", 3, 1.0), noop("b", 7, 2.0)]);
    let workflow = workflow(chain, [200, 40, 40]);
    let decomposition = Enumerating::default()
        .decompose(&workflow, &constraints(vec![0], vec![32]))
        .unwrap();

    let modelled: f64 = decomposition
        .phases
        .iter()
        .map(|phase| {
            let mut redundancy = 1.0;
            let reach = phase.reach.bound(phase.volume());
            for &axis in phase.grid.split_axes() {
                redundancy *= (phase.grid.block()[axis] as f64 + 2.0 * reach[axis] as f64)
                    / phase.grid.block()[axis] as f64;
            }
            phase.grid.core_voxels() * redundancy * phase.grid.n_blocks() as f64
        })
        .sum();

    let env = AccountingEnvironment::new([200, 40, 40], [32, 40, 40], 8);
    let stats = Enumerating::default()
        .run(&workflow, &decomposition, &env)
        .unwrap();
    assert!(
        modelled >= stats.read_voxels as f64,
        "model predicted {modelled} reads, the run performed {}",
        stats.read_voxels
    );
    // and the recorded exact prediction must match the run to the voxel
    let exact: usize = decomposition.exact_read_voxels().iter().sum();
    assert_eq!(exact as u64, stats.read_voxels);
}

// =========================================================== 2. geometry ==

/// The whole-framework oracle: identity ops, so the output *is* the input,
/// swept across everything a schedule can vary.
#[test]
fn identity_ops_reproduce_the_input_exactly_across_the_whole_sweep() {
    let shape = [37, 12, 11];
    let input = ramp(shape);

    for reach in [0usize, 1, 3, 5] {
        for block in [4usize, 7, 64] {
            for split_axes in [vec![0], vec![2], vec![0, 1, 2]] {
                for &priority in &[SchedulePriority::PhaseMajor, SchedulePriority::BlockMajor] {
                    for &concurrency in &[1usize, 3] {
                        for &visit_order in &[None, Some([2, 1, 0]), Some([1, 2, 0])] {
                            let chain = Chain::sequence(vec![
                                noop("a", reach, 1.0),
                                noop("b", reach, 1.0),
                                noop("c", 0, 1.0),
                            ]);
                            let expected = expected_sequence(&chain);
                            let workflow = workflow(chain, shape);
                            for partition in [0u32, 1, 2, 3] {
                                let decomposition =
                                    manual_partition(&workflow, partition, block, &split_axes);
                                let env = ArrayEnvironment::new(
                                    input.clone().into(),
                                    decomposition.n_phases(),
                                    [4, 4, 4],
                                )
                                .unwrap();
                                let hints = Hints {
                                    visit_order,
                                    priority,
                                    concurrency,
                                    prefetch_depth: 0,
                                    ..Hints::default()
                                };
                                let stats =
                                    execute("sweep", &workflow, &decomposition, &hints, &env)
                                        .unwrap();
                                assert_eq!(
                                    output(&env),
                                    input,
                                    "identity chain changed the volume: reach {reach}, block \
                                     {block}, axes {split_axes:?}, {priority:?}, workers \
                                     {concurrency}, order {visit_order:?}, partition {partition}"
                                );
                                stats
                                    .log
                                    .check_coverage_and_order(
                                        &expected,
                                        decomposition.phases[0].blocks.len(),
                                    )
                                    .unwrap();
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Build a decomposition by hand from a cut mask, so the sweep can vary the
/// partition independently of any planner.
fn manual_partition(
    workflow: &Workflow,
    mask: u32,
    block: usize,
    split_axes: &[usize],
) -> Decomposition {
    let slots = workflow.chain.slots();
    let volume = workflow.shape;
    let groups = super::decomposition::groups_for(mask, slots.len());
    let phases = groups
        .iter()
        .map(|group| {
            let (reach, _, names, _) =
                super::decomposition::summarise_slots(&slots, group, volume).unwrap();
            let grid = BlockGrid::along(volume, split_axes, block).unwrap();
            PhaseDecomposition::derive(group.clone(), names, reach.clone(), reach, grid)
        })
        .collect();
    let decomposition = Decomposition {
        volume,
        dtype: workflow.dtype,
        phases,
        chain_reach: workflow.chain.reach3(&volume),
    };
    decomposition.check().unwrap();
    decomposition
}

/// A voxelwise chain with a known closed form, so the sweep also proves values
/// land at the right *offsets* rather than merely surviving.
#[test]
fn an_affine_chain_lands_at_the_right_offsets_under_every_schedule() {
    let shape = [23, 9, 7];
    let input = ramp(shape);
    let mut want = input.clone();
    want.iter_mut()
        .for_each(|value| *value = *value * 2.0 + 3.0);

    for block in [3usize, 8, 32] {
        for &concurrency in &[1usize, 4] {
            let chain = Chain::sequence(vec![
                Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
                Chain::op(AffineOp::new("plus3", 1.0, 3.0, [0, 0, 0])),
            ]);
            let workflow = workflow(chain, shape);
            let decomposition = manual_partition(&workflow, 1, block, &[0, 1, 2]);
            let env =
                ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [2, 2, 2])
                    .unwrap();
            execute(
                "affine",
                &workflow,
                &decomposition,
                &Hints {
                    concurrency,
                    priority: SchedulePriority::BlockMajor,
                    ..Hints::default()
                },
                &env,
            )
            .unwrap();
            assert_eq!(output(&env), want, "block {block}, workers {concurrency}");
        }
    }
}

// ------------------------------------------- a phase that changes shape --

/// A two-phase plan whose second phase is cut from a **smaller** volume and
/// reads a window of the level below.
///
/// This is the plan that could not exist: `Decomposition::check` refused any
/// phase whose grid was over a different volume, so `input grid != output grid`
/// was not expensive or unpriced but inexpressible, and the only way to run one
/// was to hide the mapping inside an `Environment` where nothing prices it.
///
/// Two limits are asserted here rather than left to be discovered:
///
/// * the mapping may **move** an extent and not resize it, because
///   `BlockOp::apply` writes an output the shape of its input. The executor says
///   so; the test provokes it.
/// * `BlockGeometry::derive` treats a read clamped at the phase's own volume
///   edge as trustworthy, which is exactly right when that edge is a real edge
///   of the array — and a cropping phase's edges are *not* edges of the level
///   below. So the reach is zero here. A non-zero reach across a crop seam needs
///   the reach to be stated in the source space, which is the pending reach
///   work; nothing in this change makes it wrong, and nothing in it makes it
///   right either.
fn crop_plan(input: [usize; 3], keep: usize, first: usize, second: usize) -> Decomposition {
    let output = [keep, input[1], input[2]];
    let offset = input[0] - keep;
    let phases = vec![
        PhaseDecomposition::derive(
            vec![0],
            vec!["a".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            BlockGrid::along(input, &[0], first).unwrap(),
        ),
        PhaseDecomposition::derive(
            vec![1],
            vec!["b".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            BlockGrid::along(output, &[0], second).unwrap(),
        )
        .with_sources(move |block| {
            Region::new(
                &[
                    block.read.start[0] + offset,
                    block.read.start[1],
                    block.read.start[2],
                ],
                &block.read.shape.clone(),
            )
        }),
    ];
    let plan = Decomposition {
        volume: input,
        dtype: Dtype::F64,
        phases,
        chain_reach: [0, 0, 0],
    };
    plan.check().unwrap();
    plan
}

#[test]
fn a_phase_may_read_one_volume_and_write_another_across_several_decompositions() {
    let shape = [32, 6, 5];
    let keep = 12;
    let input = ramp(shape);
    let want = input
        .slice_axis(
            ndarray::Axis(0),
            ndarray::Slice::from((shape[0] - keep) as isize..),
        )
        .to_owned();

    for first in [4usize, 8, 32] {
        for second in [3usize, 4, 12] {
            for &concurrency in &[1usize, 3] {
                for &priority in &[SchedulePriority::PhaseMajor, SchedulePriority::BlockMajor] {
                    let chain = Chain::sequence(vec![noop("a", 0, 1.0), noop("b", 0, 1.0)]);
                    let workflow = workflow(chain, shape);
                    let plan = crop_plan(shape, keep, first, second);
                    let env =
                        ArrayEnvironment::for_decomposition(input.clone().into(), &plan, [2, 2, 2])
                            .unwrap();
                    let stats = execute(
                        "crop",
                        &workflow,
                        &plan,
                        &Hints {
                            concurrency,
                            priority,
                            ..Hints::default()
                        },
                        &env,
                    )
                    .unwrap();
                    assert_eq!(
                        output(&env),
                        want,
                        "blocks {first}/{second}, workers {concurrency}, {priority:?}"
                    );
                    // The plan's own read accounting is what the run performed,
                    // which is the property `source` exists for: the fetch is in
                    // the plan, so it is counted.
                    let exact: usize = plan.exact_read_voxels().iter().sum();
                    assert_eq!(exact as u64, stats.read_voxels);
                }
            }
        }
    }
}

/// The environment that holds one shape per level refuses the plan it cannot
/// hold, and the one built from the plan holds it.
#[test]
fn an_environment_says_whether_it_can_host_a_plan_that_changes_shape() {
    let shape = [32, 6, 5];
    let input = ramp(shape);
    let plan = crop_plan(shape, 12, 8, 4);

    let flat = ArrayEnvironment::new(input.clone().into(), plan.n_phases(), [2, 2, 2]).unwrap();
    let error = flat.prepare(&plan).unwrap_err().to_string();
    assert!(error.contains("holds level 2"), "{error}");

    let counting = AccountingEnvironment::new(shape, [2, 2, 2], 8);
    let error = counting.prepare(&plan).unwrap_err().to_string();
    assert!(error.contains("changes shape between levels"), "{error}");

    ArrayEnvironment::for_decomposition(input.into(), &plan, [2, 2, 2])
        .unwrap()
        .prepare(&plan)
        .unwrap();
}

/// A fetch that resizes has nowhere to land, and the executor says so instead
/// of writing a well-formed wrong volume.
#[test]
fn a_fetch_that_is_not_the_shape_of_the_read_extent_is_refused_by_the_executor() {
    let shape = [32, 6, 5];
    let input = ramp(shape);
    let mut plan = crop_plan(shape, 12, 8, 4);
    plan.phases[1] = plan.phases[1].clone().with_sources(|block| {
        Region::new(
            &[block.read.start[0], 0, 0],
            &[block.read.shape[0] + 1, 6, 5],
        )
    });
    plan.check().unwrap();

    let chain = Chain::sequence(vec![noop("a", 0, 1.0), noop("b", 0, 1.0)]);
    let workflow = workflow(chain, shape);
    let env = ArrayEnvironment::for_decomposition(input.into(), &plan, [2, 2, 2]).unwrap();
    let error = execute("crop", &workflow, &plan, &Hints::default(), &env)
        .unwrap_err()
        .to_string();
    assert!(error.contains("has nowhere to land"), "{error}");
}

/// The guard, provoked on the phase that changed the volume.
///
/// `with_forced_halo` is the provocation, and it must reach a phase whose valid
/// regions are tiled against **its own** volume — a path that did not exist
/// while every phase shared one.
#[test]
fn the_tiling_guard_fires_on_the_phase_that_changed_the_volume() {
    let shape = [32, 6, 5];
    let mut plan = crop_plan(shape, 12, 8, 4);
    // give the second phase a reach, then a halo that cannot cover it
    plan.phases[1].reach = [3, 0, 0].into();
    let short = plan.with_forced_halo([1, 0, 0]);
    let message = short.check().unwrap_err().to_string();
    assert!(
        message.contains("phase 1") && message.contains("do not tile the volume exactly"),
        "{message}"
    );
    // and the fetch regions survived the provocation, so it changed one thing
    assert!(short.phases[1].reads_across_grids());
}

// =============================================================== 3. halo ==

/// A window sum computed whole-volume, for the halo tests to diverge from.
fn window_sum_reference(input: &Array3<f64>, radius: [usize; 3]) -> Array3<f64> {
    use super::op::{Anchor, BlockOp};
    let op = WindowSumOp::new("reference", radius);
    let shape = input.shape();
    let shape = [shape[0], shape[1], shape[2]];
    let source: crate::voxels::Voxels = input.clone().into();
    let mut out = crate::voxels::Voxels::zeros(Dtype::F64, shape).unwrap();
    op.apply(&source, &mut out, &Anchor::whole(shape)).unwrap();
    out.view::<f64>().unwrap().to_owned()
}

#[test]
fn a_sufficient_halo_reproduces_the_whole_volume_window_sum() {
    let shape = [29, 13, 11];
    let input = ramp(shape);
    let radius = [2usize, 1, 2];
    let want = window_sum_reference(&input, radius);

    for block in [4usize, 5, 9, 32] {
        let chain = Chain::op(WindowSumOp::new("window", radius));
        let workflow = workflow(chain, shape);
        let decomposition = manual_partition(&workflow, 0, block, &[0, 1, 2]);
        let env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
        execute("window", &workflow, &decomposition, &Hints::default(), &env).unwrap();
        assert_eq!(output(&env), want, "block {block}");
    }
}

/// The provoked guard. Declaring reach `r` while providing `h < r` must fail
/// **both** ways.
#[test]
fn a_short_halo_fails_structurally_and_behaviourally() {
    let shape = [29, 13, 11];
    let input = ramp(shape);
    let radius = [3usize, 0, 0];
    let want = window_sum_reference(&input, radius);

    let chain = Chain::op(WindowSumOp::new("window", radius));
    let workflow = workflow(chain, shape);
    let honest = manual_partition(&workflow, 0, 8, &[0]);
    honest.check().expect("the honest decomposition must tile");

    let short = honest.with_forced_halo([1, 0, 0]);

    // (a) structural: the existing tiling check fires.
    let err = short.check().unwrap_err().to_string();
    assert!(
        err.contains("do not tile the volume exactly"),
        "expected the tiling guard, got: {err}"
    );
    assert!(err.contains("halo [1, 0, 0]"), "got: {err}");

    // and the executor refuses the same decomposition, for the same reason
    let env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
    let err = execute("short", &workflow, &short, &Hints::default(), &env)
        .unwrap_err()
        .to_string();
    assert!(err.contains("do not tile the volume exactly"), "got: {err}");

    // (b) behavioural: force the values out anyway, by lying about the reach so
    // the geometry looks fine, and watch them diverge at the block seams.
    let lying = Chain::op(WindowSumOp::under_declaring("window", radius, [1, 0, 0]));
    let lying_workflow = workflow_from(lying, shape);
    let lying_decomposition = manual_partition(&lying_workflow, 0, 8, &[0]);
    lying_decomposition
        .check()
        .expect("an under-declared reach still tiles — that is the danger");
    let env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
    execute(
        "under-declared",
        &lying_workflow,
        &lying_decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();
    let got = output(&env);
    assert_ne!(
        got, want,
        "an under-declared reach must change the values, or the window-sum probe \
         is not sensitive to the halo at all"
    );
    let differing = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
    assert!(differing > 0);
    // divergence is at block seams, not everywhere: interior voxels are fine
    assert!(
        differing < got.len(),
        "everything differs, so this is not a seam effect"
    );
}

fn workflow_from(chain: Chain, shape: [usize; 3]) -> Workflow {
    Workflow::new(chain, shape, Dtype::F64)
}

/// Over-fetching must be free of consequence beyond reads.
#[test]
fn a_generous_halo_changes_nothing_but_the_read_count() {
    let shape = [29, 13, 11];
    let input = ramp(shape);
    let radius = [2usize, 0, 0];
    let want = window_sum_reference(&input, radius);

    let chain = Chain::op(WindowSumOp::new("window", radius));
    let workflow = workflow(chain, shape);
    let tight = manual_partition(&workflow, 0, 8, &[0]);
    let generous = tight.with_forced_halo([6, 0, 0]);
    generous.check().unwrap();

    let mut reads = Vec::new();
    for decomposition in [&tight, &generous] {
        let env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
        let stats = execute("halo", &workflow, decomposition, &Hints::default(), &env).unwrap();
        assert_eq!(output(&env), want);
        reads.push(stats.read_voxels);
    }
    assert!(
        reads[1] > reads[0],
        "a generous halo should read more: {reads:?}"
    );
}

// ================================================== 4. the plan honoured ==

/// The acceptance criterion, asserted from the log at a scale no array could
/// reach: every block affected by every op, in chain order, once each.
#[test]
fn every_block_is_affected_by_every_op_in_order_at_scale() {
    // 2048 x 2048 x 2048 = 8.6 Gvoxel. Nothing is allocated.
    let shape = [2048, 2048, 2048];
    let chain = Chain::sequence(vec![
        noop("median", 4, 1.0),
        noop("deconvolve", 6, 2.0),
        noop("equalize", 12, 1.0),
        noop("adaptive", 3, 1.0),
        noop("combine", 0, 0.5),
    ]);
    let expected = expected_sequence(&chain);
    let workflow = workflow(chain, shape);
    let decomposition = Enumerating::default()
        .decompose(&workflow, &constraints(vec![2], vec![256]))
        .unwrap();
    let blocks = decomposition.phases[0].blocks.len();
    assert_eq!(blocks, 8);

    let env = AccountingEnvironment::new(shape, [128, 128, 128], 8);
    let stats = Greedy { concurrency: 4 }
        .run(&workflow, &decomposition, &env)
        .unwrap();

    stats
        .log
        .check_coverage_and_order(&expected, blocks)
        .unwrap();
    assert!(stats.log.duplicate_applications().is_empty());
    assert_eq!(stats.tasks, decomposition.n_tasks());
    assert_eq!(stats.ops_applied, blocks * 5);
    assert_eq!(stats.blocks_visited, blocks);
    assert_eq!(
        stats.materialisations,
        blocks * (decomposition.n_phases() - 1)
    );
}

/// A plan saying "fuse these three" and an executor that materialises between
/// them agree on output while disagreeing on everything the plan controls. So
/// the counts, not the output, are what prove the partition was honoured.
#[test]
fn the_partition_is_visible_in_the_counts_not_in_the_output() {
    let shape = [512, 64, 64];
    let build = || {
        Chain::sequence(vec![
            noop("a", 4, 1.0),
            noop("b", 4, 1.0),
            noop("c", 4, 1.0),
        ])
    };

    let fused_workflow = workflow(build(), shape);
    let fused = manual_partition(&fused_workflow, 0b00, 64, &[0]);
    let split = manual_partition(&fused_workflow, 0b11, 64, &[0]);
    assert_eq!(fused.n_phases(), 1);
    assert_eq!(split.n_phases(), 3);

    let mut measured = Vec::new();
    for decomposition in [&fused, &split] {
        let env = AccountingEnvironment::new(shape, [64, 64, 64], 8);
        let stats = execute(
            "counts",
            &fused_workflow,
            decomposition,
            &Hints::default(),
            &env,
        )
        .unwrap();
        stats
            .log
            .check_coverage_and_order(&expected_sequence(&build()), 8)
            .unwrap();
        // one loader read and one loader write per task, no more: the plan
        // controls how many times data is fetched, and a run that fetched more
        // would be doing something the plan did not describe
        assert_eq!(stats.reads, stats.tasks as u64);
        assert_eq!(stats.writes, stats.tasks as u64);
        measured.push((
            stats.tasks,
            stats.materialisations,
            stats.read_voxels,
            stats.ops_applied,
        ));
    }

    // Same ops applied, same coverage; everything else differs.
    assert_eq!(measured[0].3, measured[1].3, "op applications must match");
    assert_eq!(measured[0].0, 8);
    assert_eq!(measured[1].0, 24);
    assert_eq!(measured[0].1, 0, "one phase materialises nothing");
    assert_eq!(measured[1].1, 16, "two boundaries x 8 blocks");
    // fusing pays the summed reach once; splitting pays each reach separately
    assert!(
        measured[1].2 > measured[0].2,
        "splitting should read more in total here: {measured:?}"
    );
}

#[test]
fn the_visit_order_hint_is_visible_in_the_log_and_changes_nothing_else() {
    let shape = [32, 32, 32];
    let input = ramp(shape);
    let chain = Chain::sequence(vec![noop("a", 0, 1.0)]);
    let workflow = workflow(chain, shape);
    let decomposition = manual_partition(&workflow, 0, 8, &[0, 1, 2]);

    let mut orders = Vec::new();
    for visit_order in [Some([0, 1, 2]), Some([2, 1, 0])] {
        let env = ArrayEnvironment::new(input.clone().into(), 1, [8, 8, 8]).unwrap();
        let stats = execute(
            "order",
            &workflow,
            &decomposition,
            &Hints {
                visit_order,
                concurrency: 1,
                ..Hints::default()
            },
            &env,
        )
        .unwrap();
        assert_eq!(output(&env), input);
        orders.push(stats.log.visit_order(0));
    }
    assert_ne!(orders[0], orders[1], "the hint had no observable effect");
    assert_eq!(orders[0].len(), orders[1].len());
    assert_eq!(orders[0][1], [0, 0, 1]);
    assert_eq!(orders[1][1], [1, 0, 0]);
}

// ================================================= empty-block short cut ==

#[test]
fn an_all_constant_block_and_halo_short_circuits_to_the_same_answer() {
    let shape = [32, 8, 8];
    // left half constant, right half not: only the constant blocks may skip
    let mut input = Array3::from_elem((shape[0], shape[1], shape[2]), 0.0);
    for i in 16..32 {
        for j in 0..8 {
            for k in 0..8 {
                input[[i, j, k]] = (i + j + k) as f64;
            }
        }
    }

    let chain = Chain::sequence(vec![
        Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
        Chain::op(AffineOp::new("plus1", 1.0, 1.0, [0, 0, 0])),
    ]);
    let expected = expected_sequence(&chain);
    let workflow = workflow(chain, shape);
    let decomposition = manual_partition(&workflow, 0, 8, &[0]);

    let env = ArrayEnvironment::new(input.clone().into(), 1, [8, 8, 8]).unwrap();
    let stats = execute("empty", &workflow, &decomposition, &Hints::default(), &env).unwrap();

    let mut want = input.clone();
    want.iter_mut()
        .for_each(|value| *value = *value * 2.0 + 1.0);
    assert_eq!(
        output(&env),
        want,
        "a skipped block must equal a computed one"
    );

    assert!(stats.tasks_short_circuited > 0, "nothing was skipped");
    assert!(
        stats.tasks_short_circuited < stats.tasks,
        "everything was skipped, so the predicate is not discriminating"
    );
    // and a short-circuited block still counts as covered by every op
    stats.log.check_coverage_and_order(&expected, 4).unwrap();
}

#[test]
fn one_op_without_a_declared_constant_disables_the_short_circuit() {
    let shape = [32, 8, 8];
    let input = Array3::from_elem((shape[0], shape[1], shape[2]), 0.0);
    let chain = Chain::sequence(vec![
        Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
        Chain::op(OpaqueOp::new("opaque", [0, 0, 0])),
    ]);
    let workflow = workflow(chain, shape);
    let decomposition = manual_partition(&workflow, 0, 8, &[0]);
    let env = ArrayEnvironment::new(input.into(), 1, [8, 8, 8]).unwrap();
    let stats = execute("opaque", &workflow, &decomposition, &Hints::default(), &env).unwrap();
    assert_eq!(stats.tasks_short_circuited, 0);
    assert_eq!(stats.ops_applied, stats.tasks * 2);
}

/// The halo is part of the predicate, not an afterthought: a block whose core
/// is empty but whose halo is not must still be computed.
#[test]
fn a_block_with_an_empty_core_but_a_non_empty_halo_is_not_skipped() {
    let shape = [24, 4, 4];
    let mut input = Array3::from_elem((shape[0], shape[1], shape[2]), 0.0);
    // one non-zero plane, just outside block 1's core but inside its halo
    for j in 0..4 {
        for k in 0..4 {
            input[[7, j, k]] = 5.0;
        }
    }
    let chain = Chain::op(WindowSumOp::new("window", [2, 0, 0]));
    let workflow = workflow(chain, shape);
    let decomposition = manual_partition(&workflow, 0, 8, &[0]);
    let env = ArrayEnvironment::new(input.clone().into(), 1, [8, 4, 4]).unwrap();
    let stats = execute(
        "halo-empty",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();

    let want = window_sum_reference(&input, [2, 0, 0]);
    assert_eq!(output(&env), want);
    // block 1's core is all zero but its halo is not, so it must not have been
    // skipped — and the output at plane 8 proves it
    assert!(output(&env)[[8, 0, 0]] > 0.0);
    let skipped_indices: Vec<[usize; 3]> = stats
        .log
        .events()
        .into_iter()
        .filter_map(|event| match event {
            Event::BlockShortCircuited { index, .. } => Some(index),
            _ => None,
        })
        .collect();
    assert!(!skipped_indices.contains(&[1, 0, 0]), "{skipped_indices:?}");
}

// ============================================== conformance across pairs ==

fn conformance_chain() -> Chain {
    Chain::sequence(vec![
        Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
        Chain::op(WindowSumOp::new("window", [2, 1, 0])),
        Chain::op(AffineOp::new("plus7", 1.0, 7.0, [0, 0, 0])),
    ])
}

#[test]
fn every_strategy_reproduces_the_trivial_strategys_output() {
    let shape = [27, 11, 9];
    let input = ramp(shape);
    let constraints = constraints(vec![0, 1, 2], vec![4, 8, 16]);

    let oracle_workflow = workflow(conformance_chain(), shape);
    let oracle_decomposition = Trivial.decompose(&oracle_workflow, &constraints).unwrap();
    let oracle_env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
    Trivial
        .run(&oracle_workflow, &oracle_decomposition, &oracle_env)
        .unwrap();
    let want = output(&oracle_env);
    assert_eq!(oracle_decomposition.n_tasks(), 1);

    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(Trivial),
        Box::new(Enumerating {
            concurrency: 3,
            priority: SchedulePriority::PhaseMajor,
            ..Enumerating::default()
        }),
        // and the exhaustive search, which must plan the same chain the same
        // way — the conformance suite is the coarse end of what
        // `tests/partition_search.rs` asserts partition for partition.
        Box::new(Enumerating {
            concurrency: 1,
            priority: SchedulePriority::BlockMajor,
            search: PartitionSearch::Exhaustive,
        }),
        Box::new(Greedy { concurrency: 4 }),
    ];

    for strategy in &strategies {
        let workflow = workflow(conformance_chain(), shape);
        let decomposition = strategy.decompose(&workflow, &constraints).unwrap();
        let env = ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4])
            .unwrap();
        strategy.run(&workflow, &decomposition, &env).unwrap();
        assert_eq!(
            output(&env),
            want,
            "{} disagreed with the trivial oracle",
            strategy.name()
        );
    }
}

/// The property the merge into one trait could otherwise erode: a `run` that
/// only works with its own `decompose` has quietly made the binding half
/// advisory.
#[test]
fn a_strategys_run_honours_a_foreign_decomposition() {
    let shape = [27, 11, 9];
    let input = ramp(shape);
    let constraints = constraints(vec![0, 1, 2], vec![4, 8, 16]);

    let oracle_workflow = workflow(conformance_chain(), shape);
    let trivial_decomposition = Trivial.decompose(&oracle_workflow, &constraints).unwrap();
    let enumerating = Enumerating::default();
    let enumerating_decomposition = enumerating
        .decompose(&oracle_workflow, &constraints)
        .unwrap();
    assert_ne!(
        trivial_decomposition.fingerprint(),
        enumerating_decomposition.fingerprint(),
        "the two decompositions must actually differ for this test to mean anything"
    );

    let env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
    Trivial
        .run(&oracle_workflow, &trivial_decomposition, &env)
        .unwrap();
    let want = output(&env);

    // Greedy::run against Trivial::decompose
    let greedy_workflow = workflow(conformance_chain(), shape);
    let env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
    let stats = Greedy { concurrency: 4 }
        .run(&greedy_workflow, &trivial_decomposition, &env)
        .unwrap();
    assert_eq!(output(&env), want);
    assert_eq!(
        stats.decomposition_fingerprint,
        trivial_decomposition.fingerprint(),
        "greedy ran something other than the decomposition it was handed"
    );
    assert_eq!(stats.tasks, 1);

    // Trivial::run against Enumerating::decompose
    let trivial_workflow = workflow(conformance_chain(), shape);
    let env = ArrayEnvironment::new(
        input.clone().into(),
        enumerating_decomposition.n_phases(),
        [4, 4, 4],
    )
    .unwrap();
    let stats = Trivial
        .run(&trivial_workflow, &enumerating_decomposition, &env)
        .unwrap();
    assert_eq!(output(&env), want);
    assert_eq!(
        stats.decomposition_fingerprint,
        enumerating_decomposition.fingerprint()
    );
    assert_eq!(stats.tasks, enumerating_decomposition.n_tasks());
}

/// The conformance pairs again, over a chain the planner now segments: the
/// oracle fuses everything into one block, the others cut the chain in three
/// around the barrier, and all four must produce the same voxels.
///
/// Worth its own test rather than a case in the sweep above, because segmenting
/// adds two materialisations to a chain the oracle runs in one pass — the plans
/// disagree about everything except the answer, which is the property.
#[test]
fn every_strategy_agrees_on_a_chain_that_contains_a_barrier() {
    let shape = [21, 7, 5];
    let input = ramp(shape);
    let constraints = constraints(vec![0, 1, 2], vec![4, 8, 16]);
    let chain = || {
        Chain::sequence(vec![
            Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
            Chain::op(IdentityOp::new("global", shape)),
            Chain::op(AffineOp::new("plus7", 1.0, 7.0, [0, 0, 0])),
        ])
    };

    let oracle_workflow = workflow(chain(), shape);
    let oracle = Trivial.decompose(&oracle_workflow, &constraints).unwrap();
    let oracle_env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
    Trivial.run(&oracle_workflow, &oracle, &oracle_env).unwrap();
    let want = output(&oracle_env);

    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(Enumerating::default()),
        Box::new(Enumerating {
            concurrency: 3,
            priority: SchedulePriority::BlockMajor,
            search: PartitionSearch::Exhaustive,
        }),
        Box::new(Greedy { concurrency: 4 }),
    ];
    for strategy in &strategies {
        let planned = workflow(chain(), shape);
        let decomposition = strategy.decompose(&planned, &constraints).unwrap();
        assert_eq!(
            decomposition.n_phases(),
            3,
            "{} did not segment at the barrier",
            strategy.name()
        );
        assert_eq!(decomposition.phases[1].grid.n_blocks(), 1);
        let env = ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4])
            .unwrap();
        strategy.run(&planned, &decomposition, &env).unwrap();
        assert_eq!(
            output(&env),
            want,
            "{} disagreed with the trivial oracle across a barrier",
            strategy.name()
        );

        // and the foreign-decomposition property, which segmenting could
        // plausibly have broken: the oracle's executor against this plan
        let foreign_workflow = workflow(chain(), shape);
        let env = ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4])
            .unwrap();
        let stats = Trivial
            .run(&foreign_workflow, &decomposition, &env)
            .unwrap();
        assert_eq!(output(&env), want);
        assert_eq!(stats.decomposition_fingerprint, decomposition.fingerprint());
    }
}

/// Same decomposition, wildly different schedules, identical output — the
/// invariant that would catch a scheduler having quietly acquired a
/// correctness dependency.
#[test]
fn the_same_decomposition_under_different_schedules_gives_the_same_output() {
    let shape = [31, 13, 7];
    let input = ramp(shape);
    let workflow = workflow(conformance_chain(), shape);
    let decomposition = manual_partition(&workflow, 0b01, 6, &[0, 1]);

    let mut outputs = Vec::new();
    for priority in [SchedulePriority::PhaseMajor, SchedulePriority::BlockMajor] {
        for concurrency in [1usize, 2, 5] {
            for visit_order in [None, Some([1, 0, 2]), Some([2, 1, 0])] {
                let env = ArrayEnvironment::new(
                    input.clone().into(),
                    decomposition.n_phases(),
                    [3, 3, 3],
                )
                .unwrap();
                execute(
                    "schedule",
                    &workflow,
                    &decomposition,
                    &Hints {
                        visit_order,
                        priority,
                        concurrency,
                        prefetch_depth: 0,
                        ..Hints::default()
                    },
                    &env,
                )
                .unwrap();
                outputs.push(output(&env));
            }
        }
    }
    for output in &outputs[1..] {
        assert_eq!(output, &outputs[0]);
    }
}

// ================================================================= graph ==

#[test]
fn the_task_graph_is_a_dag_over_block_phase_pairs_with_real_dependencies() {
    let shape = [64, 8, 8];
    let chain = Chain::sequence(vec![noop("a", 4, 1.0), noop("b", 4, 1.0)]);
    let workflow = workflow(chain, shape);
    let decomposition = manual_partition(&workflow, 0b1, 16, &[0]);
    let graph = TaskGraph::build(&decomposition);

    assert_eq!(graph.n_phases(), 2);
    assert_eq!(graph.len(), 8);
    graph.dependencies_cover_reads(&decomposition).unwrap();
    // phase 0 has no dependencies; phase 1 depends on its own block and its
    // neighbours, because its read extent is wider than its core
    assert!(graph.tasks_in_phase(0).iter().all(|t| t.deps.is_empty()));
    let phase_one = graph.tasks_in_phase(1);
    assert_eq!(phase_one[0].deps.len(), 2);
    assert_eq!(phase_one[1].deps.len(), 3);
    assert_eq!(phase_one[3].deps.len(), 2);
}

/// The simulator is what makes keeping several strategies cheap: predicted
/// stats for a full-scale volume with nothing allocated.
#[test]
fn a_full_scale_workflow_can_be_simulated_without_allocating_anything() {
    let shape = [2094, 13316, 3369];
    let chain = Chain::sequence(vec![
        noop("median", 2, 1.0),
        noop("deconvolve", 6, 3.0),
        noop("equalize", 16, 1.0),
        noop("adaptive", 4, 1.0),
        noop("combine", 0, 0.2),
    ]);
    let workflow = workflow(chain, shape);
    // Split XY as well as Z. With `axes = [2]` alone a block spans the full
    // 2094 x 13316 plane, which is the 58-119 GB block
    // `docs/design/XY_BLOCK_SPLITTING.md` exists to remove — and the planner
    // says so by refusing the budget rather than by planning something that
    // will not fit.
    let mut z_only = constraints(vec![2], vec![64, 128, 256]);
    z_only.budget_bytes = Some(32 * 1024 * 1024 * 1024);
    z_only.expected_concurrency = 8;
    let refusal = Enumerating::default()
        .decompose(&workflow, &z_only)
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("byte budget"), "got: {refusal}");

    let mut constraints = constraints(vec![0, 1, 2], vec![256, 512]);
    constraints.budget_bytes = Some(32 * 1024 * 1024 * 1024);
    constraints.expected_concurrency = 8;

    let ranked: Vec<(String, f64, u64)> = [
        (
            "trivial-ish",
            Box::new(Enumerating::default()) as Box<dyn Strategy>,
        ),
        ("greedy", Box::new(Greedy { concurrency: 8 })),
    ]
    .into_iter()
    .map(|(name, strategy)| {
        let decomposition = strategy.decompose(&workflow, &constraints).unwrap();
        let env = AccountingEnvironment::new(shape, [128, 128, 128], 2).with_emptiness(0.39, 0.0);
        let stats = strategy.run(&workflow, &decomposition, &env).unwrap();
        assert!(
            stats.tasks_short_circuited > 0,
            "emptiness was not exploited"
        );
        (name.to_string(), stats.estimated_work, stats.read_voxels)
    })
    .collect();

    assert_eq!(ranked.len(), 2);
    for (name, work, reads) in &ranked {
        assert!(*work > 0.0, "{name} predicted no work at all");
        assert!(*reads > 0);
    }
}

// ============================================================ regression ==

/// A read extent that leaves the volume is a bug in the geometry, not
/// something to clamp away silently at the sink.
#[test]
fn reads_and_writes_stay_inside_the_volume() {
    let shape = [17, 5, 3];
    let chain = Chain::op(WindowSumOp::new("window", [4, 2, 2]));
    let workflow = workflow(chain, shape);
    let decomposition = manual_partition(&workflow, 0, 4, &[0, 1, 2]);
    for phase in &decomposition.phases {
        for block in &phase.blocks {
            for axis in 0..3 {
                assert!(block.read.start[axis] + block.read.shape[axis] <= shape[axis]);
                assert!(block.valid.start[axis] + block.valid.shape[axis] <= shape[axis]);
            }
        }
    }
    let env = ArrayEnvironment::new(ramp(shape).into(), 1, [2, 2, 2]).unwrap();
    execute("bounds", &workflow, &decomposition, &Hints::default(), &env).unwrap();
}

#[test]
fn an_alternative_branch_budgets_for_the_max_and_runs_the_taken_one() {
    let shape = [24, 6, 6];
    let input = ramp(shape);
    let chain = Chain::alternative(
        vec![
            Chain::op(WindowSumOp::new("wide", [5, 0, 0])),
            Chain::op(AffineOp::new("cheap", 3.0, 0.0, [0, 0, 0])),
        ],
        1,
    )
    .unwrap();
    let workflow = workflow(chain, shape);
    let decomposition = manual_partition(&workflow, 0, 8, &[0]);
    // the halo is budgeted for the *wide* branch even though the cheap one runs
    assert_eq!(decomposition.phases[0].reach, [5, 0, 0]);
    assert_eq!(decomposition.phases[0].halo, [5, 0, 0]);

    let env = ArrayEnvironment::new(input.clone().into(), 1, [8, 6, 6]).unwrap();
    execute(
        "alternative",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();
    let mut want = input;
    want.iter_mut().for_each(|value| *value *= 3.0);
    assert_eq!(output(&env), want);
}

#[test]
fn a_region_helper_agrees_with_the_geometry_it_describes() {
    let grid = BlockGrid::along([100, 10, 10], &[0], 32).unwrap();
    assert_eq!(grid.n_blocks(), 4);
    let cores = grid.cores();
    assert_eq!(cores[3].core, Region::new(&[96, 0, 0], &[4, 10, 10]));
    let total: usize = cores.iter().map(|core| core.core.voxels()).sum();
    assert_eq!(total, 100 * 10 * 10);
}

// ------------------------------------------------- non-pixel block output --
//
// A sidecar is bytes keyed by `(stream, phase, block)`, so it has to work
// wherever a block index does — including in the simulated environment, whose
// whole job is to run a strategy over a volume nobody could allocate. If a
// fragment only worked against real arrays, a strategy that produced one could
// not be simulated, and the design's requirement that the *same* strategy code
// runs in both worlds would quietly stop holding.

#[test]
fn sidecar_traffic_is_counted_by_the_environment_that_carried_it() {
    use crate::sidecar::Lifecycle;

    let env = AccountingEnvironment::new([2094, 13316, 3369], [128, 128, 128], 2);
    env.declare_sidecar("fragments", Lifecycle::DeleteOnExit)
        .unwrap();
    for block in 0..4usize {
        env.write_sidecar("fragments", 0, [block, 0, 0], &[7u8; 16])
            .unwrap();
    }
    let (writes, reads, written, read) = env.counters().sidecar_snapshot();
    assert_eq!((writes, written), (4, 64));
    assert_eq!((reads, read), (0, 0));

    // Reading counts too, and a fragment that is not there is a counted read
    // of zero bytes rather than an uncounted nothing.
    assert!(env
        .read_sidecar("fragments", 0, [0, 0, 0])
        .unwrap()
        .is_some());
    assert!(env
        .read_sidecar("fragments", 0, [9, 0, 0])
        .unwrap()
        .is_none());
    let (_, reads, _, read) = env.counters().sidecar_snapshot();
    assert_eq!((reads, read), (2, 16));

    // And the pixel counters are untouched: a fragment is not a region, and
    // folding it into `writes` would corrupt every cost the model derives from
    // that number.
    let (pixel_reads, pixel_writes, ..) = env.counters().snapshot();
    assert_eq!((pixel_reads, pixel_writes), (0, 0));

    // Reading the whole stream is the merge side's call, and it is counted the
    // same way.
    let fragments = env.sidecar_fragments("fragments").unwrap();
    assert_eq!(fragments.len(), 4);
    let (_, reads, _, read) = env.counters().sidecar_snapshot();
    assert_eq!((reads, read), (6, 16 + 64));

    let report = env.discard_sidecars().unwrap();
    assert_eq!(report.fragments(), 4);
    assert_eq!(report.bytes(), 64);
    assert!(env.sidecar_keys("fragments").unwrap().is_empty());
}

#[test]
fn sidecar_writes_reach_the_listeners_a_run_registered() {
    use crate::listener::EventListener;
    use crate::sidecar::Lifecycle;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Seen(Mutex<Vec<Event>>);
    impl EventListener for Seen {
        fn on_event(&self, event: &Event) {
            self.0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(event.clone());
        }
    }

    let env = ArrayEnvironment::new(ramp([8, 4, 4]).into(), 1, [4, 4, 4]).unwrap();
    let seen = Arc::new(Seen::default());
    env.sidecars()
        .expect("this environment has a sidecar store")
        .attach(seen.clone());
    env.declare_sidecar("fragments", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("fragments", 0, [1, 0, 0], b"abc")
        .unwrap();

    let events = seen.0.lock().unwrap().clone();
    assert_eq!(
        events,
        vec![Event::SidecarWritten {
            stream: "fragments".to_string(),
            phase: 0,
            index: [1, 0, 0],
            bytes: 3,
            // the only field a test cannot pin
            duration_ns: match &events[0] {
                Event::SidecarWritten { duration_ns, .. } => *duration_ns,
                other => panic!("{other:?}"),
            },
        }]
    );
}

#[test]
fn a_fragment_is_bytes_and_the_store_never_looks_inside_one() {
    use crate::sidecar::Lifecycle;

    // Three payloads a format-aware store would have an opinion about: invalid
    // UTF-8, an embedded NUL, and something that looks like JSON but is not.
    let payloads: Vec<Vec<u8>> = vec![
        vec![0xff, 0xfe, 0x00, 0x80],
        b"a\0b".to_vec(),
        b"{not really".to_vec(),
    ];
    let env = ArrayEnvironment::new(ramp([4, 4, 4]).into(), 1, [4, 4, 4]).unwrap();
    env.declare_sidecar("opaque", Lifecycle::Persistent)
        .unwrap();
    for (block, payload) in payloads.iter().enumerate() {
        env.write_sidecar("opaque", 0, [block, 0, 0], payload)
            .unwrap();
    }
    for (block, payload) in payloads.iter().enumerate() {
        assert_eq!(
            env.read_sidecar("opaque", 0, [block, 0, 0])
                .unwrap()
                .as_ref(),
            Some(payload)
        );
    }
}
