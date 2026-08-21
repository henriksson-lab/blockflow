// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The three shapes `apply(input, out, at)` cannot express, executed:
//
// 1. `volume -> fragments` — read pixels, emit one fragment per block;
// 2. `fragments -> fragments` — read your own fragment and, where you say so,
//    your neighbours'; emit another. **No pixel IO at all**;
// 3. `fragments -> volume` — read every fragment, write pixels. The global
//    reduction, expressed as a full-reach phase rather than as a fan-in the DAG
//    does not have.
//
// What each assertion is for
// --------------------------
// * **Coverage twice, independently.** Once from the event stream, which is
//   what a distributed run merges and therefore the only thing available there;
//   once from what the store actually holds, which is what a merging reader
//   would see. Two sources agreeing is worth more than either alone, because
//   the failure being guarded against is a phase that *reports* covering the
//   lattice.
// * **The fetch count, measured against the analytic one.** A reach is a
//   declaration; the number of fragments fetched is the behaviour. A zero-reach
//   phase that quietly read its neighbours would still produce the right answer
//   here and be wrong at scale, so the count is asserted rather than the result.
// * **Pixel counters at zero.** The claim "no pixel array involved" is only
//   worth making if it is checked, and it is checked against an environment
//   that *has* real arrays — so a stray read would show up rather than being
//   impossible.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use blockflow::decomposition::{
    price_phase, Constraints, CostModel, Decomposition, PhaseDecomposition,
};
use blockflow::dtype::Dtype;
use blockflow::env::{AccountingEnvironment, ArrayEnvironment, Environment};
use blockflow::fragment::{
    append_fragment_phase, check_fragment_coverage, fold_fragments, fragment_only, neighbourhood,
    neighbourhood_size, BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput, PhaseWork,
};
use blockflow::geometry::BlockGrid;
use blockflow::log::{Event, ExecutionLog};
use blockflow::op::Chain;
use blockflow::probes::{BlockSummaryOp, FragmentReduceOp, IdentityOp, NeighbourFoldOp};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Enumerating, Hints, Strategy, Workflow};
use ndarray::Array3;

const VOLUME: [usize; 3] = [32, 4, 4];
const BLOCK: [usize; 3] = [8, 4, 4];
const BLOCKS: usize = 4;
const LATTICE: [usize; 3] = [BLOCKS, 1, 1];

fn ramp() -> blockflow::Voxels {
    let mut array = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for (flat, value) in array.iter_mut().enumerate() {
        *value = flat as f64;
    }
    array.into()
}

fn grid() -> BlockGrid {
    BlockGrid::new(VOLUME, BLOCK).expect("a lattice")
}

/// One pixel phase over the lattice, as a starting point for appending
/// fragment phases. Built by hand rather than planned, so the block count the
/// analytic fetch counts rest on is stated rather than chosen by a cost model.
fn one_pixel_phase() -> Decomposition {
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![PhaseDecomposition::derive(
            vec![0],
            vec!["identity".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            grid(),
        )],
        chain_reach: [0, 0, 0],
    }
}

fn identity_workflow() -> Workflow {
    Workflow::new(
        Chain::op(IdentityOp::new("identity", [0, 0, 0])),
        VOLUME,
        Dtype::F64,
    )
}

fn empty_workflow() -> Workflow {
    Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::F64)
}

/// The store's own event stream, which is where fragment IO is reported: the
/// executor's `Dispatch` is not a listener, so a fragment write reaches the run
/// through the listeners a caller attached to the store. A distributed worker
/// does exactly this (`distributed/worker.rs`), so the test does too.
fn watch(env: &dyn Environment) -> Arc<ExecutionLog> {
    let log = Arc::new(ExecutionLog::new());
    env.sidecars()
        .expect("this environment has a sidecar store")
        .attach(log.clone());
    log
}

fn written_blocks(log: &ExecutionLog, stream: &str, phase: usize) -> BTreeSet<[usize; 3]> {
    log.events()
        .into_iter()
        .filter_map(|event| match event {
            Event::SidecarWritten {
                stream: name,
                phase: at,
                index,
                ..
            } if name == stream && at == phase => Some(index),
            _ => None,
        })
        .collect()
}

fn reads_per_block(log: &ExecutionLog, stream: &str, phase: usize) -> BTreeMap<[usize; 3], usize> {
    let mut out = BTreeMap::new();
    for event in log.events() {
        if let Event::SidecarRead {
            stream: name,
            phase: at,
            index,
            ..
        } = event
        {
            if name == stream && at == phase {
                *out.entry(index).or_insert(0) += 1;
            }
        }
    }
    out
}

fn stored_blocks(env: &dyn Environment, stream: &str, phase: usize) -> BTreeSet<[usize; 3]> {
    env.sidecar_keys(stream)
        .expect("the keys")
        .into_iter()
        .filter(|key| key.phase == phase)
        .map(|key| key.block)
        .collect()
}

fn lattice_blocks() -> BTreeSet<[usize; 3]> {
    grid().cores().into_iter().map(|core| core.index).collect()
}

// ------------------------------------------------- 1. volume -> fragments --

#[test]
fn a_volume_to_fragments_phase_leaves_one_fragment_per_block_and_they_reduce_to_the_volume() {
    let plan = one_pixel_phase();
    let summary = BlockSummaryOp::new("summary", "fragments", Lifecycle::DeleteOnExit);
    let plan = append_fragment_phase(plan, &summary).expect("a fragment phase");
    let workflow = identity_workflow();
    let env = ArrayEnvironment::new(ramp(), plan.n_phases(), [4, 4, 4]).expect("an environment");
    let log = watch(&env);

    let stats = execute_phases(
        "fragments",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&summary)],
    )
    .expect("a run");
    assert_eq!(stats.phases, 2);
    assert_eq!(stats.tasks, 2 * BLOCKS);

    // Coverage from the event stream...
    assert_eq!(
        written_blocks(&log, "fragments", 1),
        lattice_blocks(),
        "the fragment writes on the event stream do not cover the lattice"
    );
    // ...and, independently, from what the store actually holds.
    assert_eq!(stored_blocks(&env, "fragments", 1), lattice_blocks());

    // Accounting sees the fragment traffic, and the pixel counters are the
    // pixel phase's alone: two images of 512 voxels read and written.
    let (writes, reads, bytes_written, bytes_read) = env.counters().sidecar_snapshot();
    assert_eq!(writes, BLOCKS as u64);
    assert_eq!(reads, 0, "nothing read a fragment in this run");
    assert_eq!(bytes_written, BLOCKS as u64 * 48, "six words per fragment");
    assert_eq!(bytes_read, 0);

    // The reduction, streamed: one fragment resident at a time.
    let mut voxels = 0usize;
    let mut sum = 0.0_f64;
    let mut blocks = BTreeSet::new();
    let seen = fold_fragments(&env, "fragments", &mut |key, bytes| {
        let (block, block_voxels, block_sum, _) = BlockSummaryOp::read(bytes)?;
        assert_eq!(block, key.block, "a payload disagrees with its key");
        blocks.insert(block);
        voxels += block_voxels;
        sum += block_sum;
        Ok(())
    })
    .expect("a merge");
    assert_eq!(seen, BLOCKS);
    assert_eq!(blocks, lattice_blocks());
    // Two facts about the whole volume that no single fragment holds.
    assert_eq!(voxels, VOLUME.iter().product::<usize>());
    assert_eq!(sum, ramp().view::<f64>().unwrap().iter().sum::<f64>());

    // The merge read them, and the accounting says so.
    let (_, reads, _, bytes_read) = env.counters().sidecar_snapshot();
    assert_eq!(reads, BLOCKS as u64);
    assert_eq!(bytes_read, BLOCKS as u64 * 48);
    let _ = bytes_written;
}

// ---------------------------------------------- 2. fragments -> fragments --

/// Produces a fragment from **nothing**: no pixels, no inputs. The seed for a
/// run with no pixel volume behind it at all.
struct SeedOp;

impl FragmentOp for SeedOp {
    fn name(&self) -> &'static str {
        "seed"
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            "seeds",
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> blockflow::error::Result<BlockOutput> {
        assert!(!at.has_pixels(), "the seed op declared no pixel access");
        Ok(BlockOutput::fragment(
            "seeds",
            blockflow::fragment::pack_u64(&[
                at.index[0] as u64,
                at.index[1] as u64,
                at.index[2] as u64,
                at.valid.voxels() as u64,
                0,
                0,
            ]),
        ))
    }
}

/// The measured fetch count against the analytic one, for every reach.
fn fetch_counts_for(reach: [usize; 3]) -> (usize, usize) {
    let seed = SeedOp;
    let fold = NeighbourFoldOp::new("fold", "seeds", 0, reach, "folded", Lifecycle::DeleteOnExit);
    let plan = fragment_only(VOLUME, BLOCK, Dtype::F64, &[&seed, &fold]).expect("a plan");
    let env = ArrayEnvironment::new(ramp(), plan.n_phases(), [4, 4, 4]).expect("an environment");
    let log = watch(&env);

    execute_phases(
        "fragments-only",
        &empty_workflow(),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&seed), PhaseWork::Fragments(&fold)],
    )
    .expect("a run");

    // Not one voxel moved, in an environment that holds real arrays.
    let (reads, writes, read_voxels, write_voxels, chunks, _, _) = env.counters().snapshot();
    assert_eq!(
        (reads, writes, read_voxels, write_voxels, chunks),
        (0, 0, 0, 0, 0),
        "a fragments -> fragments run touched pixels"
    );

    let measured: usize = reads_per_block(&log, "seeds", 0).values().sum();
    let analytic: usize = (0..BLOCKS)
        .map(|block| neighbourhood_size([block, 0, 0], reach, LATTICE))
        .sum();

    // Every block that was fetched is one the declaration asks for, and no
    // other: a total alone would pass if a block read the wrong neighbour.
    for block in 0..BLOCKS {
        let wanted = neighbourhood([block, 0, 0], reach, LATTICE);
        let (_, seen, voxels, _) = NeighbourFoldOp::read(
            &env.read_sidecar("folded", 1, [block, 0, 0])
                .expect("a fragment")
                .expect("block wrote one"),
        )
        .expect("a fold");
        assert_eq!(seen, wanted.len(), "block {block} folded the wrong count");
        assert_eq!(
            voxels,
            wanted.len() * BLOCK.iter().product::<usize>(),
            "block {block} folded the wrong blocks"
        );
    }
    assert_eq!(stored_blocks(&env, "folded", 1), lattice_blocks());
    (measured, analytic)
}

#[test]
fn a_zero_reach_fragment_phase_reads_exactly_its_own_block_and_no_neighbour() {
    let (measured, analytic) = fetch_counts_for([0, 0, 0]);
    assert_eq!(analytic, BLOCKS, "the analytic count for zero reach");
    assert_eq!(
        measured, analytic,
        "a zero-reach phase fetched {measured} fragments for {BLOCKS} blocks"
    );
}

#[test]
fn a_declared_reach_reads_exactly_the_neighbours_it_should() {
    let (measured, analytic) = fetch_counts_for([1, 0, 0]);
    // 4 blocks in a line: 2 + 3 + 3 + 2.
    assert_eq!(analytic, 10);
    assert_eq!(measured, analytic, "reach 1 fetched {measured}, wanted 10");

    let (measured, analytic) = fetch_counts_for([2, 0, 0]);
    // 3 + 4 + 4 + 3, clamped at both ends.
    assert_eq!(analytic, 14);
    assert_eq!(measured, analytic, "reach 2 fetched {measured}, wanted 14");
}

#[test]
fn a_fragments_only_run_simulates_with_no_data_at_all() {
    let seed = SeedOp;
    let fold = NeighbourFoldOp::new(
        "fold",
        "seeds",
        0,
        [1, 0, 0],
        "folded",
        Lifecycle::DeleteOnExit,
    );
    let plan = fragment_only(VOLUME, BLOCK, Dtype::F64, &[&seed, &fold]).expect("a plan");
    let env = AccountingEnvironment::new(VOLUME, [4, 4, 4], 8);
    execute_phases(
        "simulated",
        &empty_workflow(),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&seed), PhaseWork::Fragments(&fold)],
    )
    .expect("a simulated run");

    // A fragment's bytes are the caller's and cannot be fabricated from a
    // region, so the simulated environment carries them for real. That is why
    // the fold's answer is the same one the real run gave.
    let (writes, reads, _, _) = env.counters().sidecar_snapshot();
    assert_eq!(writes, 2 * BLOCKS as u64);
    assert_eq!(reads, 10);
    let (pixel_reads, pixel_writes, ..) = env.counters().snapshot();
    assert_eq!((pixel_reads, pixel_writes), (0, 0));
}

// ------------------------------------------------- 3. fragments -> volume --

#[test]
fn a_full_reach_fragments_to_volume_phase_reduces_and_writes_the_answer_back() {
    let summary = BlockSummaryOp::new("summary", "fragments", Lifecycle::DeleteOnExit);
    let reduce = FragmentReduceOp::new("reduce", "fragments", 0, LATTICE);
    let plan = fragment_only(VOLUME, BLOCK, Dtype::F64, &[&summary, &reduce]).expect("a plan");

    // The derivation, checked: with a full reach only halo == volume tiles.
    assert_eq!(plan.phases[1].reach, VOLUME);
    assert_eq!(plan.phases[1].halo, VOLUME);
    for block in &plan.phases[1].blocks {
        assert_eq!(block.read.shape, VOLUME.to_vec(), "every block reads all");
        assert_eq!(block.valid, block.core);
    }
    plan.check().expect("a full-reach plan tiles");

    // And with a short halo it does not, loudly. `with_forced_halo` is the
    // crate's own way of provoking a guard that has never been seen to fire.
    let short = plan.with_forced_halo([8, 4, 4]);
    let error = short
        .check()
        .expect_err("a short halo under a full reach must fail the tiling check")
        .to_string();
    assert!(error.contains("lost part of their core"), "{error}");

    let env = ArrayEnvironment::new(ramp(), plan.n_phases(), [4, 4, 4]).expect("an environment");
    let log = watch(&env);
    execute_phases(
        "reduce",
        &empty_workflow(),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Fragments(&summary),
            PhaseWork::Fragments(&reduce),
        ],
    )
    .expect("a run");

    // Every voxel of the output holds the reduction, which is knowable without
    // a reference implementation: the blocks tile, so the voxels sum to the
    // volume.
    let total = VOLUME.iter().product::<usize>() as f64;
    let output = env.output();
    assert!(
        output
            .view::<f64>()
            .unwrap()
            .iter()
            .all(|&value| value == total),
        "the reduced volume is not uniformly {total}"
    );

    // Full reach means every block read every fragment: 4 x 4.
    let measured: usize = reads_per_block(&log, "fragments", 0).values().sum();
    assert_eq!(
        measured,
        BLOCKS * BLOCKS,
        "a full-reach phase read {measured} fragments for {BLOCKS} blocks"
    );
    // ...and it streamed them rather than gathering: the op says `gathers() ==
    // false`, so residency is one fragment, not all of them. There is no
    // counter for that; the declaration is what the executor acts on.
    assert_eq!(stored_blocks(&env, "fragments", 0), lattice_blocks());
}

/// The planning consequence, measured rather than asserted: a full-reach phase
/// is priced N-fold redundant, so the cost model prefers one block.
#[test]
fn the_cost_model_prices_a_full_reach_phase_towards_one_block() {
    let model = CostModel::default();
    let mut costs = Vec::new();
    for edge in [8usize, 16, 32] {
        let lattice = BlockGrid::along(VOLUME, &[0], edge).expect("a lattice");
        let cost = price_phase(&lattice, &VOLUME.into(), 1.0, 1, false, 8.0, &model, 1.0);
        costs.push((
            edge,
            lattice.n_blocks(),
            cost.redundancy,
            cost.cost_per_block * lattice.n_blocks() as f64,
        ));
    }
    for (edge, blocks, redundancy, total) in &costs {
        println!(
            "full reach, edge {edge}: {blocks} block(s), redundancy {redundancy:.1}x, \
             total {total:.0}"
        );
    }
    let cheapest = costs
        .iter()
        .min_by(|left, right| left.3.total_cmp(&right.3))
        .expect("a cheapest");
    assert_eq!(
        cheapest.0, 32,
        "the cheapest full-reach block size was {} not the whole volume; costs {costs:?}",
        cheapest.0
    );
}

/// Does the planner *segment* at a full-reach op, or plan across it? Recorded
/// rather than asserted as a preference: what matters is knowing which.
#[test]
fn what_the_planner_does_with_a_full_reach_op_in_a_chain() {
    let workflow = Workflow::new(
        Chain::sequence(vec![
            Chain::op(IdentityOp::new("before", [0, 0, 0])),
            Chain::op(IdentityOp::new("global", VOLUME)),
            Chain::op(IdentityOp::new("after", [0, 0, 0])),
        ]),
        VOLUME,
        Dtype::F64,
    );
    let constraints = Constraints {
        block_candidates: vec![8, 16, 32],
        split_axes: vec![0],
        ..Default::default()
    };
    let plan = Enumerating::default()
        .decompose(&workflow, &constraints)
        .expect("a plan");
    println!(
        "a chain with a full-reach op in the middle decomposes into {} phase(s): {:?}",
        plan.n_phases(),
        plan.phases
            .iter()
            .map(|phase| (phase.names.clone(), phase.grid.block(), phase.reach.clone()))
            .collect::<Vec<_>>()
    );
    // The planner segments here. It did not when this test was written: it
    // fused all three ops into one phase, dragging the two cheap local ops into
    // a single-block phase whose working set was the whole volume, and priced
    // that at redundancy 1.0 — because `price_phase` charged redundancy only on
    // `grid.split_axes()`, and `BlockGrid::new` drops an axis from `split_axes`
    // when `block == volume`. The model charged nothing for the most expensive
    // configuration it could pick.
    //
    // Now a full-reach op is a barrier: `Enumerating` refuses to price a
    // partition that fuses across one, and redundancy is charged on split axes
    // *union* full-reach axes. The same fused plan prices at 55808 against
    // 31232 for the three-phase one, so structure and cost agree rather than
    // the cut resting on price alone.
    assert!(
        plan.n_phases() >= 2,
        "the chain fused across a full-reach op into {} phase(s)",
        plan.n_phases()
    );
    plan.check().expect("whatever it chose must tile");
    let global = plan
        .phases
        .iter()
        .find(|phase| phase.names.iter().any(|name| name == "global"))
        .expect("the full-reach op is in some phase");
    assert_eq!(
        global.grid.n_blocks(),
        1,
        "the phase carrying a full-reach op was cut into {} blocks",
        global.grid.n_blocks()
    );
}

// ------------------------------------------------------ 4. the guard bites --

/// Declares every-block coverage and writes for one block only. The failure the
/// tiling check cannot see, because a fragment phase's valid regions are its
/// cores whatever it wrote.
struct HoleOp;

impl FragmentOp for HoleOp {
    fn name(&self) -> &'static str {
        "hole"
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            "holed",
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> blockflow::error::Result<BlockOutput> {
        if at.index == [0, 0, 0] {
            return Ok(BlockOutput::fragment("holed", vec![0u8; 8]));
        }
        Ok(BlockOutput::nothing())
    }
}

#[test]
fn a_stream_that_declares_every_block_and_leaves_a_hole_fails_the_run() {
    let hole = HoleOp;
    let plan = fragment_only(VOLUME, BLOCK, Dtype::F64, &[&hole]).expect("a plan");
    // The pixel-side guards both pass, which is the point.
    plan.check()
        .expect("the valid regions tile whatever it wrote");

    let env = ArrayEnvironment::new(ramp(), plan.n_phases(), [4, 4, 4]).expect("an environment");
    let error = execute_phases(
        "holed",
        &empty_workflow(),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&hole)],
    )
    .expect_err("a hole in an every-block stream must fail the run")
    .to_string();
    assert!(error.contains("every-block coverage"), "{error}");
    assert!(error.contains("wrote none"), "{error}");
}

/// A stream that is honestly sparse is allowed, and then only containment is
/// checked — but the phase must still have something checkable about it, so a
/// sparse-only phase that writes no image is refused before it runs.
#[test]
fn a_phase_with_nothing_checkable_about_it_is_refused_rather_than_run() {
    struct SparseOnly;

    impl FragmentOp for SparseOnly {
        fn name(&self) -> &'static str {
            "sparse"
        }

        fn outputs(&self) -> Vec<FragmentOutput> {
            vec![FragmentOutput::new(
                "sparse",
                Lifecycle::DeleteOnExit,
                Coverage::Sparse,
            )]
        }

        fn apply(&self, _at: &BlockView<'_>) -> blockflow::error::Result<BlockOutput> {
            Ok(BlockOutput::nothing())
        }
    }

    let op = SparseOnly;
    let plan = fragment_only(VOLUME, BLOCK, Dtype::F64, &[&op]).expect("a plan");
    let env = ArrayEnvironment::new(ramp(), plan.n_phases(), [4, 4, 4]).expect("an environment");
    let error = execute_phases(
        "sparse",
        &empty_workflow(),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&op)],
    )
    .expect_err("a phase with no checkable output must be refused")
    .to_string();
    assert!(error.contains("entirely unguarded"), "{error}");
}

/// The guard is also callable on its own, which is how a merging reader — or a
/// coordinator that ran no task — checks a distributed phase.
#[test]
fn the_coverage_guard_is_callable_by_a_reader_that_ran_nothing() {
    let plan = one_pixel_phase();
    let summary = BlockSummaryOp::new("summary", "fragments", Lifecycle::DeleteOnExit);
    let plan = append_fragment_phase(plan, &summary).expect("a fragment phase");
    let env = ArrayEnvironment::new(ramp(), plan.n_phases(), [4, 4, 4]).expect("an environment");
    execute_phases(
        "fragments",
        &identity_workflow(),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&summary)],
    )
    .expect("a run");

    let report = check_fragment_coverage(&env, &plan, 1, &summary).expect("the guard passes");
    assert_eq!(report.len(), 1);
    assert_eq!(report[0].blocks, BLOCKS);
    assert_eq!(report[0].lattice, BLOCKS);
    assert_eq!(report[0].declared, Coverage::EveryBlock);
    println!("{}", report[0].describe());
}
