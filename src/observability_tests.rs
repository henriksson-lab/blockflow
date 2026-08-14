// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance tests for observation. Separate from `tests.rs` because they
// assert a different kind of thing: `tests.rs` proves the *framework* is right,
// this file proves that watching it does not change it and that two watchers
// agree about what they saw.
//
// Four properties, each catching a distinct failure:
//
// 1. **Every event reaches every listener.** Asserted by running the existing
//    coverage/order checker against a *registered* listener rather than against
//    the built-in log — if dispatch dropped anything, the acceptance criterion
//    fails from the listener's point of view while still passing from the
//    executor's, which is exactly the bug worth catching.
// 2. **Two listeners agree.** `LatestOpPerChunk`'s final state must equal the
//    last op per block taken from the order log. They compute it by completely
//    different means — one keeps 8 bytes per block, the other keeps everything
//    — so agreement is evidence rather than tautology.
// 3. **A poll during a run does not block and never sees an impossible state.**
// 4. **A listener cannot change the result.** The same run, with and without
//    listeners, must produce identical counts and an identical output array.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ndarray::Array3;

use crate::dtype::Dtype;
use crate::error::Result;
use crate::voxels::Voxels;

use super::decomposition::{Constraints, CostModel, Decomposition, PhaseDecomposition};
use super::env::{AccountingEnvironment, ArrayEnvironment};
use super::export::{order_log_to_json, ExportMeta};
use super::geometry::BlockGrid;
use super::listener::{EventListener, LatestOpPerChunk, OrderLog, ProgressKind};
use super::log::Event;
use super::op::{Anchor, BlockOp, Chain};
use super::probes::{AffineOp, IdentityOp};
use super::strategy::{
    execute_observed, Enumerating, Greedy, Hints, SchedulePriority, Strategy, Workflow,
};

// -------------------------------------------------------------- fixtures --

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

fn expected_sequence(chain: &Chain) -> Vec<(usize, String)> {
    chain
        .slots()
        .iter()
        .enumerate()
        .map(|(slot, sub)| (slot, sub.display_name()))
        .collect()
}

/// A three-op workflow over a grid of 8 blocks, split into three phases.
///
/// The partition is stated rather than planned: these tests are about what the
/// executor *reports*, so the plan has to be fixed or a change to the cost
/// model would silently turn a multi-phase test into a single-phase one and
/// stop testing what it says it tests.
fn scenario() -> (Workflow, Decomposition, Vec<(usize, String)>) {
    let chain = Chain::sequence(vec![
        noop("median", 2, 1.0),
        noop("wide", 4, 1.0),
        noop("threshold", 1, 1.0),
    ]);
    let expected = expected_sequence(&chain);
    let workflow = workflow(chain, [512, 64, 64]);
    // 0b11: cut after every op, so three phases over the same 8-block grid.
    let decomposition = manual_partition(&workflow, 0b11, 64, &[0]);
    (workflow, decomposition, expected)
}

/// A decomposition with the phase boundaries chosen by hand. `mask` bit *i*
/// set means "cut after slot *i*".
fn manual_partition(
    workflow: &Workflow,
    mask: u32,
    block: usize,
    split_axes: &[usize],
) -> Decomposition {
    let slots = workflow.chain.slots();
    let volume = workflow.shape;
    let phases = super::decomposition::groups_for(mask, slots.len())
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

// ------------------------------------------------- 1. nothing is dropped --

/// The acceptance criterion, asserted from a **registered listener** instead of
/// from the executor's own log. Both must see the same run.
#[test]
fn every_event_the_executor_emits_reaches_every_registered_listener() {
    let (workflow, decomposition, expected) = scenario();
    let blocks = decomposition.phases[0].blocks.len();
    assert!(decomposition.n_phases() > 1, "want a multi-phase schedule");

    let first = Arc::new(OrderLog::new());
    let second = Arc::new(OrderLog::new());
    let listeners: Vec<Arc<dyn EventListener>> = vec![first.clone(), second.clone()];

    let env = AccountingEnvironment::new(workflow.shape, [64, 64, 64], 8);
    let stats = Greedy { concurrency: 4 }
        .run_observed(&workflow, &decomposition, &env, &listeners)
        .unwrap();

    assert_eq!(stats.listener_faults, 0);
    // The listener sees the criterion the executor's own log sees.
    first.check_coverage_and_order(&expected, blocks).unwrap();
    assert!(first.duplicate_applications().is_empty());
    second.check_coverage_and_order(&expected, blocks).unwrap();
    assert!(first.len() > 0);

    // Same events, same count, same per-block order — the three properties a
    // consumer may rely on. Not the same *global* interleaving: dispatch is
    // deliberately not serialised across workers, so two listeners may see two
    // different linearisations of the same concurrent run. See `Dispatch`.
    assert_eq!(first.len(), stats.log.len());
    assert_eq!(second.len(), stats.log.len());
    assert_eq!(
        first.op_sequence_per_block(),
        stats.log.op_sequence_per_block()
    );
    assert_eq!(
        second.op_sequence_per_block(),
        stats.log.op_sequence_per_block()
    );
    let multiset = |log: &OrderLog| {
        let mut all: Vec<String> = log
            .events()
            .iter()
            .map(|event| format!("{event:?}"))
            .collect();
        all.sort();
        all
    };
    assert_eq!(multiset(&first), multiset(&stats.log));
    assert_eq!(multiset(&second), multiset(&stats.log));
}

/// With one worker there is no interleaving to differ over, so a listener's
/// stream must be **identical** to the executor's own, event for event. This is
/// what separates "dispatch reorders under concurrency" from "dispatch drops or
/// reorders, full stop".
#[test]
fn with_one_worker_a_listener_sees_the_identical_sequence() {
    let (workflow, decomposition, _) = scenario();
    let mirror = Arc::new(OrderLog::new());
    let listeners: Vec<Arc<dyn EventListener>> = vec![mirror.clone()];
    let env = AccountingEnvironment::new(workflow.shape, [64, 64, 64], 8);
    let stats = execute_observed(
        "serial",
        &workflow,
        &decomposition,
        &Hints {
            concurrency: 1,
            ..Hints::default()
        },
        &env,
        &listeners,
    )
    .unwrap();
    assert_eq!(mirror.events(), stats.log.events());
}

/// The IO and scheduling layers are in the same stream as the op layer, and at
/// a rate bounded by the task count rather than by the chunk count.
#[test]
fn io_and_scheduling_events_share_the_stream_at_a_task_bounded_rate() {
    let (workflow, decomposition, _) = scenario();
    // A deliberately small chunk shape: each block spans many chunks, so a
    // per-chunk emission policy would show up here as an event explosion.
    let env = AccountingEnvironment::new(workflow.shape, [8, 8, 8], 8);
    let stats = Greedy { concurrency: 2 }
        .run(&workflow, &decomposition, &env)
        .unwrap();

    let mut region_reads = 0usize;
    let mut region_writes = 0usize;
    let mut admitted = 0usize;
    let mut materialised = 0usize;
    let mut chunks_per_read = 0u64;
    for event in stats.log.events() {
        match event {
            Event::RegionRead { chunks, .. } => {
                region_reads += 1;
                chunks_per_read = chunks_per_read.max(chunks);
            }
            Event::RegionWritten { .. } => region_writes += 1,
            Event::TaskAdmitted { .. } => admitted += 1,
            Event::Materialised { .. } => materialised += 1,
            _ => {}
        }
    }
    assert_eq!(
        region_reads, stats.tasks,
        "one read event per task, no more"
    );
    assert_eq!(region_writes, stats.tasks);
    assert_eq!(admitted, stats.tasks);
    assert_eq!(materialised, stats.phases, "one per phase");
    assert!(
        chunks_per_read > 8,
        "a block spans many chunks ({chunks_per_read}); the point is that this \
         is a count on one event, not that many events"
    );
    // The whole stream stays a small multiple of the task count.
    assert!(
        stats.log.len() < stats.tasks * 12,
        "{} events for {} tasks",
        stats.log.len(),
        stats.tasks
    );
}

// ------------------------------------------------- 2. two listeners agree --

#[test]
fn latest_op_per_chunk_agrees_with_the_order_log() {
    let (workflow, decomposition, expected) = scenario();
    let latest = Arc::new(LatestOpPerChunk::new());
    let listeners: Vec<Arc<dyn EventListener>> = vec![latest.clone()];

    let env = AccountingEnvironment::new(workflow.shape, [64, 64, 64], 8);
    let stats = Greedy { concurrency: 4 }
        .run_observed(&workflow, &decomposition, &env, &listeners)
        .unwrap();

    // The last op each block saw, from the full history...
    let from_log: std::collections::BTreeMap<[usize; 3], (usize, String)> = stats
        .log
        .op_sequence_per_block()
        .into_iter()
        .map(|(index, ops)| (index, ops.last().cloned().expect("a block with no ops")))
        .collect();
    // ...and from 8 bytes per block.
    let from_latest = latest.last_op_per_block();

    assert_eq!(from_latest, from_log);
    assert_eq!(latest.blocks_seen(), from_log.len());
    // and it really is the chain's last op
    let last = expected.last().unwrap().clone();
    for (index, seen) in &from_latest {
        assert_eq!(seen, &last, "block {index:?}");
    }
    // every block ends written, because the write is the last thing a task does
    for state in latest.snapshot() {
        assert_eq!(state.kind, ProgressKind::Written, "{:?}", state.index);
    }
}

/// The short circuit leaves a block holding what computing would have produced,
/// so the progress view must report it as having reached the last op — not as
/// stalled at the read.
#[test]
fn a_short_circuited_block_reports_progress_like_a_computed_one() {
    let shape = [32, 8, 8];
    let mut input = Array3::from_elem((shape[0], shape[1], shape[2]), 0.0);
    for value in input.iter_mut().take(8 * 8 * 8) {
        *value = 3.0;
    }
    let chain = Chain::sequence(vec![
        Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
        Chain::op(AffineOp::new("plus1", 1.0, 1.0, [0, 0, 0])),
    ]);
    let workflow = workflow(chain, shape);
    let decomposition = Enumerating::default()
        .decompose(&workflow, &constraints(vec![0], vec![8]))
        .unwrap();
    let latest = Arc::new(LatestOpPerChunk::new());
    let listeners: Vec<Arc<dyn EventListener>> = vec![latest.clone()];
    let env = ArrayEnvironment::new(input.into(), 1, [8, 8, 8]).unwrap();
    let stats = execute_observed(
        "short-circuit",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
        &listeners,
    )
    .unwrap();

    assert!(stats.tasks_short_circuited > 0, "nothing was skipped");
    let last_ops = latest.last_op_per_block();
    assert_eq!(last_ops.len(), 4);
    for (index, (slot, name)) in &last_ops {
        assert_eq!((*slot, name.as_str()), (1, "plus1"), "block {index:?}");
    }
}

// ------------------------------------------------------- 3. live polling --

/// A visualiser polls while the executor runs. Every state it sees must be one
/// the executor really produced — no block outside the grid, no slot outside
/// the chain, no slot and name that disagree.
///
/// Promptness is *not* asserted here. It used to be, as a 250 ms bound on the
/// slowest poll, which is a benchmark wearing a correctness test's clothes: it
/// failed twice under a concurrent build and passed alone. The property it was
/// reaching for is asserted structurally in
/// `a_poll_completes_while_the_executor_is_inside_an_op`.
#[test]
fn a_poll_concurrent_with_a_run_never_blocks_and_never_tears() {
    let (workflow, decomposition, expected) = scenario();
    let names: Vec<String> = expected.iter().map(|(_, name)| name.clone()).collect();
    let grid = [
        decomposition.phases[0].grid.blocks_per_axis()[0],
        decomposition.phases[0].grid.blocks_per_axis()[1],
        decomposition.phases[0].grid.blocks_per_axis()[2],
    ];

    let latest = Arc::new(LatestOpPerChunk::new());
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let polls = Arc::new(AtomicUsize::new(0));

    let poller = {
        let latest = latest.clone();
        let done = done.clone();
        let polls = polls.clone();
        std::thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                let snapshot = latest.snapshot();
                polls.fetch_add(1, Ordering::Relaxed);
                for state in snapshot {
                    for axis in 0..3 {
                        assert!(
                            state.index[axis] < grid[axis],
                            "a block outside the grid: {:?}",
                            state.index
                        );
                    }
                    // The slot is sticky, so any kind may carry one; what must
                    // never be observed is a slot that is not in the chain, or
                    // a slot and a name that disagree.
                    if let Some(slot) = state.slot {
                        assert!(slot < names.len(), "slot {slot} is not in the chain");
                        if let Some(op) = &state.op {
                            assert_eq!(op, &names[slot], "slot and name disagree");
                        }
                    }
                    assert!(state.phase < 3, "phase {} is not in the plan", state.phase);
                }
                std::thread::yield_now();
            }
        })
    };

    // Wait for the poller to be demonstrably alive before starting the run.
    // Without this the test asserts `polls > 0` about a thread that a loaded
    // machine may not schedule before a 20 ms run finishes — which is how it
    // flaked. Concurrency *during* the run is the separate, deterministic
    // property asserted by `a_poll_completes_while_the_executor_is_inside_an_op`.
    while polls.load(Ordering::Relaxed) == 0 {
        std::thread::yield_now();
    }

    let listeners: Vec<Arc<dyn EventListener>> = vec![latest.clone()];
    let env = AccountingEnvironment::new(workflow.shape, [64, 64, 64], 8);
    let stats = Greedy { concurrency: 4 }
        .run_observed(&workflow, &decomposition, &env, &listeners)
        .unwrap();
    done.store(true, Ordering::Relaxed);
    poller.join().unwrap();

    assert_eq!(stats.listener_faults, 0);
}

/// An op that stays inside `apply` for as long as it takes the poller to prove
/// it is not blocked, then returns.
///
/// It is a rendezvous rather than a sleep because a sleep makes the test a
/// race: on a loaded machine the poller may simply not be scheduled inside a
/// fixed window, and the test then fails for a reason that has nothing to do
/// with the property. Waiting *until* the poller succeeds is self-calibrating
/// — load lengthens the window instead of failing the test. The timeout exists
/// only so a genuine block fails rather than hangs.
struct HoldingOp {
    in_flight: Arc<AtomicUsize>,
    polled_during_work: Arc<AtomicUsize>,
    timeout: std::time::Duration,
}

impl BlockOp for HoldingOp {
    fn name(&self) -> &'static str {
        "holding"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        let deadline = std::time::Instant::now() + self.timeout;
        while self.polled_during_work.load(Ordering::SeqCst) == 0
            && std::time::Instant::now() < deadline
        {
            std::thread::yield_now();
        }
        out.assign(input)?;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }
}

/// A snapshot completes *while* the executor is inside `apply`.
///
/// This is the property the previous wall-clock bound was standing in for: a
/// poll must not queue behind the work. Stated structurally it does not depend
/// on how fast the machine is, which the 250 ms bound did — it failed twice
/// under nothing worse than a concurrent build.
#[test]
fn a_poll_completes_while_the_executor_is_inside_an_op() {
    let in_flight = Arc::new(AtomicUsize::new(0));
    let polled_during_work = Arc::new(AtomicUsize::new(0));

    let chain = Chain::op(HoldingOp {
        in_flight: in_flight.clone(),
        polled_during_work: polled_during_work.clone(),
        timeout: std::time::Duration::from_secs(10),
    });
    let workflow = workflow(chain, [512, 64, 64]);
    let decomposition = manual_partition(&workflow, 0, 64, &[0]);

    let latest = Arc::new(LatestOpPerChunk::new());
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let poller = {
        let latest = latest.clone();
        let done = done.clone();
        let in_flight = in_flight.clone();
        let polled_during_work = polled_during_work.clone();
        std::thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                // Both loads bracket the snapshot, so a counted poll began and
                // ended with an op inside `apply`. One load either side would
                // admit a poll that merely overlapped the window's edge.
                let busy_before = in_flight.load(Ordering::SeqCst) > 0;
                let snapshot = latest.snapshot();
                let busy_after = in_flight.load(Ordering::SeqCst) > 0;
                if busy_before && busy_after {
                    polled_during_work.fetch_add(1, Ordering::SeqCst);
                }
                drop(snapshot);
                std::thread::yield_now();
            }
        })
    };

    let listeners: Vec<Arc<dyn EventListener>> = vec![latest.clone()];
    // A real array environment, not the accounting one: `AccountingEnvironment`
    // is data-free and never calls `BlockOp::apply` (`env.rs:583`), so the op
    // would never enter the window this test is about.
    let input = Array3::<f64>::zeros((workflow.shape[0], workflow.shape[1], workflow.shape[2]));
    let env = ArrayEnvironment::new(input.into(), decomposition.n_phases(), [64, 64, 64]).unwrap();
    let stats = Greedy { concurrency: 4 }
        .run_observed(&workflow, &decomposition, &env, &listeners)
        .unwrap();
    done.store(true, Ordering::Relaxed);
    poller.join().unwrap();

    assert!(
        polled_during_work.load(Ordering::SeqCst) > 0,
        "no snapshot completed while an op was inside `apply`: the poll is \
         blocking behind the run"
    );
    assert_eq!(stats.listener_faults, 0);
}

// ------------------------------------ 4. observation cannot change a run --

#[test]
fn a_run_with_listeners_computes_exactly_what_a_run_without_them_computes() {
    let shape = [64, 16, 16];
    let mut input = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
    for (flat, value) in input.iter_mut().enumerate() {
        *value = ((flat * 7919) % 1013) as f64 + 1.0;
    }
    let input: crate::voxels::Voxels = input.into();
    let build = || {
        Chain::sequence(vec![
            Chain::op(AffineOp::new("double", 2.0, 0.0, [1, 1, 1])),
            Chain::op(AffineOp::new("plus7", 1.0, 7.0, [2, 0, 0])),
        ])
    };
    let workflow = workflow(build(), shape);
    let decomposition = Enumerating::default()
        .decompose(&workflow, &constraints(vec![0], vec![16]))
        .unwrap();

    let plain_env = ArrayEnvironment::new(input.clone(), 1, [16, 16, 16]).unwrap();
    let plain = Greedy { concurrency: 4 }
        .run(&workflow, &decomposition, &plain_env)
        .unwrap();

    struct Slow(AtomicUsize);
    impl EventListener for Slow {
        fn on_event(&self, _event: &Event) {
            // Deliberately not free: a listener that costs time must cost
            // throughput and nothing else.
            self.0.fetch_add(1, Ordering::SeqCst);
            std::hint::spin_loop();
        }
    }
    struct Exploding;
    impl EventListener for Exploding {
        fn on_event(&self, _event: &Event) {
            panic!("an observer with a bug in it");
        }
    }

    let slow = Arc::new(Slow(AtomicUsize::new(0)));
    let listeners: Vec<Arc<dyn EventListener>> = vec![
        Arc::new(Exploding),
        slow.clone(),
        Arc::new(LatestOpPerChunk::new()),
    ];
    let watched_env = ArrayEnvironment::new(input, 1, [16, 16, 16]).unwrap();
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let watched = Greedy { concurrency: 4 }
        .run_observed(&workflow, &decomposition, &watched_env, &listeners)
        .unwrap();
    std::panic::set_hook(previous);

    assert_eq!(
        watched.listener_faults, 1,
        "the broken listener was isolated"
    );
    assert!(slow.0.load(Ordering::SeqCst) > 0, "the slow listener ran");
    assert_eq!(
        watched_env.output(),
        plain_env.output(),
        "observation changed the result"
    );
    assert_eq!(watched.tasks, plain.tasks);
    assert_eq!(watched.ops_applied, plain.ops_applied);
    assert_eq!(watched.blocks_visited, plain.blocks_visited);
    assert_eq!(watched.reads, plain.reads);
    assert_eq!(watched.writes, plain.writes);
    assert_eq!(watched.log.len(), plain.log.len());
}

// --------------------------------------------------------------- export --

/// The exported document must describe *this* run: the same blocks, the same
/// per-block op order, and voxel geometry a Python consumer can use without
/// recomputing anything.
#[test]
fn the_export_describes_the_run_it_came_from() {
    let (workflow, decomposition, expected) = scenario();
    let env = AccountingEnvironment::new(workflow.shape, [64, 64, 64], 8);
    let stats = Greedy { concurrency: 4 }
        .run(&workflow, &decomposition, &env)
        .unwrap();

    let meta = ExportMeta::new("greedy", workflow.shape, decomposition.n_phases())
        .with_ops(expected.clone());
    let document = order_log_to_json(&stats.log, &meta);

    assert_eq!(document["version"], 1);
    assert_eq!(document["phases"], decomposition.n_phases());
    let grid = document["grid"].as_array().unwrap();
    assert_eq!(
        grid[0].as_u64().unwrap() as usize,
        decomposition.phases[0].grid.blocks_per_axis()[0]
    );
    let blocks = document["blocks"].as_array().unwrap();
    assert_eq!(blocks.len(), stats.blocks_visited);

    // The valid regions tile the volume: a consumer drawing them gets no gaps.
    let mut covered = 0u64;
    for block in blocks {
        let shape = block["valid"]["shape"].as_array().unwrap();
        covered += shape
            .iter()
            .map(|value| value.as_u64().unwrap())
            .product::<u64>();
        // and the read extent is at least the valid extent, because of the halo
        let read = block["read"]["shape"].as_array().unwrap();
        for axis in 0..3 {
            assert!(read[axis].as_u64().unwrap() >= shape[axis].as_u64().unwrap());
        }
    }
    assert_eq!(
        covered,
        (workflow.shape[0] * workflow.shape[1] * workflow.shape[2]) as u64
    );

    // Per-block op order in the document equals the chain order.
    let mut per_block: std::collections::BTreeMap<Vec<u64>, Vec<String>> = Default::default();
    for event in document["events"].as_array().unwrap() {
        if event["type"] == "op_applied" {
            let index: Vec<u64> = event["index"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap())
                .collect();
            per_block
                .entry(index)
                .or_default()
                .push(event["op"].as_str().unwrap().to_string());
        }
    }
    let want: Vec<String> = expected.iter().map(|(_, name)| name.clone()).collect();
    assert_eq!(per_block.len(), stats.blocks_visited);
    for (index, ops) in per_block {
        assert_eq!(ops, want, "block {index:?}");
    }
}

/// Two schedules of the *same* workflow must produce visibly different event
/// streams — that is the whole reason the animation is worth rendering. If
/// block-major and phase-major produced the same admission order, either the
/// hints do nothing or the log is not recording the schedule.
#[test]
fn block_major_and_phase_major_produce_different_admission_orders() {
    let (workflow, decomposition, _) = scenario();
    assert!(decomposition.n_phases() > 1);

    let admissions = |priority: SchedulePriority| {
        let hints = Hints {
            priority,
            concurrency: 1,
            ..Hints::default()
        };
        let env = AccountingEnvironment::new(workflow.shape, [64, 64, 64], 8);
        let stats =
            execute_observed("order", &workflow, &decomposition, &hints, &env, &[]).unwrap();
        stats
            .log
            .events()
            .into_iter()
            .filter_map(|event| match event {
                Event::TaskAdmitted { phase, index } => Some((phase, index)),
                _ => None,
            })
            .collect::<Vec<_>>()
    };

    let block_major = admissions(SchedulePriority::BlockMajor);
    let phase_major = admissions(SchedulePriority::PhaseMajor);
    assert_eq!(block_major.len(), phase_major.len());
    assert_ne!(
        block_major, phase_major,
        "the two schedules are indistinguishable in the log"
    );

    // Phase-major finishes every phase-0 task before starting phase 1;
    // block-major does not.
    let first_phase_change = |order: &[(usize, [usize; 3])]| {
        order
            .iter()
            .position(|(phase, _)| *phase > 0)
            .unwrap_or(order.len())
    };
    let phase_zero_tasks = decomposition.phases[0].blocks.len();
    assert_eq!(first_phase_change(&phase_major), phase_zero_tasks);
    assert!(
        first_phase_change(&block_major) < phase_zero_tasks,
        "block-major advanced no block ahead of the phase"
    );
}

/// The `Mutex`-backed order log is the only lock on the event path with
/// exclusive access; this pins the claim that a listener sees a *consistent*
/// prefix even while several workers emit.
#[test]
fn concurrent_workers_do_not_interleave_a_single_blocks_ops() {
    let (workflow, decomposition, expected) = scenario();
    let seen = Arc::new(Mutex::new(Vec::new()));
    struct Watch(Arc<Mutex<Vec<([usize; 3], usize)>>>);
    impl EventListener for Watch {
        fn on_event(&self, event: &Event) {
            if let Event::OpApplied { index, slot, .. } = event {
                self.0.lock().unwrap().push((*index, *slot));
            }
        }
    }
    let listeners: Vec<Arc<dyn EventListener>> = vec![Arc::new(Watch(seen.clone()))];
    let env = AccountingEnvironment::new(workflow.shape, [64, 64, 64], 8);
    Greedy { concurrency: 8 }
        .run_observed(&workflow, &decomposition, &env, &listeners)
        .unwrap();

    let mut per_block: std::collections::BTreeMap<[usize; 3], Vec<usize>> = Default::default();
    for (index, slot) in seen.lock().unwrap().iter() {
        per_block.entry(*index).or_default().push(*slot);
    }
    let want: Vec<usize> = expected.iter().map(|(slot, _)| *slot).collect();
    for (index, slots) in per_block {
        assert_eq!(slots, want, "block {index:?} saw its ops out of order");
    }
}
