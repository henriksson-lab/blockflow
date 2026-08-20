// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The verification the distribution design asks for, run the way it asks:
// **local multi-node mode, as separate processes.**
//
// Not threads. A thread-based fake shares one address space, one cache, one
// memory budget, one allocator and one set of file handles, so it would
// exercise the message shapes and none of the things that actually go wrong
// here — a worker reading an intermediate before the writer flushed it, two
// processes writing one file, an event stream that merges correctly only
// because both ends were the same object, a work list that stays ahead only
// because the "network" was a function call. Every test in this file starts
// real processes over real sockets against real shared files.
//
// The five claims, one test each
// ------------------------------
// 1. **N workers produce byte-identical output to a single-node run**, swept
//    over worker counts. The headline, and the one that would catch a wrong
//    seam, a missing flush or a mis-stitched block.
// 2. **Every block executed exactly once**, asserted by the crate's *existing*
//    `check_coverage_and_order` over the merged event stream — no
//    distribution-specific analysis, because a merged stream is an
//    `ExecutionLog` like any other.
// 3. **A worker dies and the job stops, naming what was lost.** The default:
//    node loss is not recovered from, and what to do about it is decided above
//    this crate. What must not happen is a silent hang.
// 4. **A worker dies under an explicit lease and its task is reissued.** The
//    same death with one field set. Reissue is opt-in now — see the module
//    header for the decision of 2026-08-17 — and this test is what keeps it
//    compiled, exercised and honest.
// 5. **The work list stays at least one task ahead**, so a worker never blocks
//    except on its own pull. No single-node test would catch this regressing.
//
// A note on 3 and 4 together, because the pair is the load-bearing part. Each
// is the other's counterexample: 4 alone cannot tell "expiry is off" from
// "expiry is broken", and 3 alone cannot tell "no lease" from "no claims". Run
// as a pair over the same fixture and the same death, they pin the default and
// the opt-in against each other.

#![cfg(feature = "distributed")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use blockflow::decomposition::Decomposition;
use blockflow::distributed::local::{self, Binaries, LocalOptions, LocalRun};
use blockflow::distributed::shared_volume::SharedVolumes;
use blockflow::distributed::spec::{
    probe_job_over, read_task_fragment, ChainSpec, FragmentPhaseSpec, JobSpec, OpSpec,
    ProbeWorkflows, SidecarSpec, StoreSpec,
};
use blockflow::distributed::{HandoutPolicy, WorkflowFactory};
use blockflow::env::Environment;
use blockflow::export::event_from_json;
use blockflow::fragment::neighbourhood_size;
use blockflow::graph::TaskGraph;
use blockflow::log::ExecutionLog;
use blockflow::probes::NeighbourFoldOp;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute, Hints, Workflow};
use ndarray::Array3;
use serde_json::Value;

const BLOCKS: usize = 16;

fn binaries() -> Binaries {
    Binaries {
        coordinator: PathBuf::from(env!("CARGO_BIN_EXE_blockflow-coordinator")),
        worker: PathBuf::from(env!("CARGO_BIN_EXE_blockflow-worker")),
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("blockflow-multinode-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// A volume whose every voxel is different, so a block written to the wrong
/// place is *visible* rather than plausible. Plausible-but-wrong is the failure
/// mode this whole crate is arranged against.
fn ramp(shape: [usize; 3]) -> Array3<f64> {
    let mut array = Array3::zeros((shape[0], shape[1], shape[2]));
    for (flat, value) in array.iter_mut().enumerate() {
        *value = flat as f64;
    }
    array
}

/// Build the job, lay down its input, and return everything needed to run it.
fn prepare(dir: &Path, phases: usize, lease: Option<Duration>) -> (JobSpec, Decomposition) {
    let volumes = dir.join("volumes");
    let (mut spec, decomposition) = probe_job_over(
        BLOCKS,
        phases,
        ChainSpec::identity(),
        StoreSpec::Files {
            dir: volumes.clone(),
        },
    );
    spec.policy = HandoutPolicy::NearestFirst;
    spec.lease = lease;
    let store = SharedVolumes::create(
        &volumes,
        spec.workflow.shape,
        spec.workflow.chunk,
        decomposition.n_phases(),
    )
    .expect("level files");
    store
        .write_level(0, &ramp(spec.workflow.shape))
        .expect("an input");
    (spec, decomposition)
}

fn options(dir: &Path, workers: usize) -> LocalOptions {
    let mut options = LocalOptions::new(dir, workers).expect("local options");
    options.binaries = binaries();
    options.timeout = Duration::from_secs(120);
    options
}

/// The final level's bytes, exactly as they are on disk.
fn output_bytes(dir: &Path, spec: &JobSpec, decomposition: &Decomposition) -> Vec<u8> {
    let store = SharedVolumes::open(
        &dir.join("volumes"),
        spec.workflow.shape,
        spec.workflow.chunk,
        decomposition.n_phases(),
    )
    .expect("the volumes");
    store
        .level_bytes(decomposition.n_phases())
        .expect("the output level")
}

/// A **single-node** run: one process, the crate's own scheduler, the same
/// decomposition, the same input.
///
/// This is the reference the distributed runs are compared against, and it is
/// deliberately not "the run with one worker": it goes through `execute` rather
/// than through a coordinator, so agreement says the distributed path produces
/// what the single-node path produces rather than merely being self-consistent.
fn single_node(dir: &Path, spec: &JobSpec, decomposition: &Decomposition) -> Vec<u8> {
    let volumes = dir.join("volumes");
    let store = SharedVolumes::create(
        &volumes,
        spec.workflow.shape,
        spec.workflow.chunk,
        decomposition.n_phases(),
    )
    .expect("level files");
    store
        .write_level(0, &ramp(spec.workflow.shape))
        .expect("an input");
    let chain = ProbeWorkflows.chain(&spec.workflow).expect("a chain");
    let workflow = Workflow::new(chain, spec.workflow.shape, spec.workflow.dtype);
    execute(
        "single-node",
        &workflow,
        decomposition,
        &Hints::default(),
        &store,
    )
    .expect("a single-node run");
    store
        .level_bytes(decomposition.n_phases())
        .expect("the output level")
}

/// The merged event stream, decoded back into an `ExecutionLog`.
fn merged_log(run: &LocalRun) -> ExecutionLog {
    let log = ExecutionLog::new();
    let events = run
        .report
        .get("log")
        .and_then(|document| document.get("events"))
        .and_then(Value::as_array)
        .expect("the coordinator's report carries the merged stream");
    for (at, event) in events.iter().enumerate() {
        if let Ok(Some(event)) = event_from_json(event, at) {
            log.push(event);
        }
    }
    log
}

fn expected_ops(decomposition: &Decomposition) -> Vec<(usize, String)> {
    decomposition
        .op_names_in_order()
        .into_iter()
        .enumerate()
        .collect()
}

// ------------------------------------------------------- 1. byte-identical --

#[test]
fn n_workers_produce_byte_identical_output_to_a_single_node_run() {
    let reference_dir = scratch("reference");
    let (spec, decomposition) = probe_job_over(
        BLOCKS,
        1,
        ChainSpec::identity(),
        StoreSpec::Files {
            dir: reference_dir.join("volumes"),
        },
    );
    let reference = single_node(&reference_dir, &spec, &decomposition);

    // The chain is identity followed by `2x + 1`, so the answer is knowable
    // without any reference at all. Checking it here means an agreement between
    // the two paths cannot be an agreement on something wrong.
    let expected: Vec<u8> = ramp(spec.workflow.shape)
        .iter()
        .flat_map(|value| (2.0 * value + 1.0).to_le_bytes())
        .collect();
    assert_eq!(
        reference, expected,
        "the single-node reference does not compute the chain it was given"
    );

    for workers in [1usize, 2, 3, 5] {
        let dir = scratch(&format!("workers-{workers}"));
        let (spec, decomposition) = prepare(&dir, 1, None);
        let run = local::run(&options(&dir, workers), &spec, &decomposition)
            .unwrap_or_else(|error| panic!("{workers} workers: {error}"));
        assert_eq!(
            run.status.done, run.status.tasks,
            "{workers} workers: {} of {} tasks",
            run.status.done, run.status.tasks
        );
        assert!(
            run.status.workers >= workers,
            "{workers} workers asked for, {} joined",
            run.status.workers
        );
        let produced = output_bytes(&dir, &spec, &decomposition);
        assert_eq!(
            produced.len(),
            reference.len(),
            "{workers} workers produced a different sized volume"
        );
        assert!(
            produced == reference,
            "{workers} workers produced {} differing bytes against the single-node run",
            produced
                .chunks(8)
                .zip(reference.chunks(8))
                .filter(|(left, right)| left != right)
                .count()
        );
        println!(
            "{workers} worker(s): {} tasks, {} events, {:?} — byte-identical",
            run.status.done, run.status.events, run.elapsed
        );
        std::fs::remove_dir_all(&dir).ok();
    }
    std::fs::remove_dir_all(&reference_dir).ok();
}

/// The same, across a **barrier** — a full-reach op, which resolves to a single
/// block, so one worker reads the whole of a level every other worker helped
/// write.
///
/// This is the phase locality-aware placement exists for, and it is the phase in
/// which placement can be *wrong*: the coordinator withholds the task from a
/// worker it models as badly placed, and the whole rule rests on that costing a
/// re-read and never a result. Asserted over processes rather than argued,
/// across the same worker counts as the headline, because a rule that changes
/// who runs the only task in a phase is exactly the kind of change that could
/// deadlock a run or seam it differently and pass every in-process test.
#[test]
fn a_barrier_phase_lands_on_one_worker_and_the_output_is_still_byte_identical() {
    // identity, then `2x + 1`, then an op reaching the whole of every axis.
    // `is_planning_barrier` is an exact `reach >= extent`, so the last op cannot
    // be fused with the others and its phase cannot be split.
    let chain = |shape: [usize; 3]| {
        let mut ops = ChainSpec::identity().0;
        ops.push(OpSpec::new("identity", "merge", shape));
        ChainSpec(ops)
    };
    let shape = [8 * BLOCKS, 8, 8];

    let reference_dir = scratch("barrier-reference");
    let (spec, decomposition) = probe_job_over(
        BLOCKS,
        1,
        chain(shape),
        StoreSpec::Files {
            dir: reference_dir.join("volumes"),
        },
    );
    let last = decomposition.n_phases() - 1;
    assert_eq!(
        decomposition.phases[last].blocks.len(),
        1,
        "a full-reach op should decompose to one block; got {:?}",
        decomposition
            .phases
            .iter()
            .map(|phase| phase.blocks.len())
            .collect::<Vec<_>>()
    );
    let reference = single_node(&reference_dir, &spec, &decomposition);
    // The barrier op is an identity, so the answer is still knowable outright.
    let expected: Vec<u8> = ramp(spec.workflow.shape)
        .iter()
        .flat_map(|value| (2.0 * value + 1.0).to_le_bytes())
        .collect();
    assert_eq!(reference, expected, "the single-node reference is wrong");

    for workers in [1usize, 2, 3, 5] {
        let dir = scratch(&format!("barrier-workers-{workers}"));
        let volumes = dir.join("volumes");
        let (mut spec, decomposition) = probe_job_over(
            BLOCKS,
            1,
            chain(shape),
            StoreSpec::Files {
                dir: volumes.clone(),
            },
        );
        spec.policy = HandoutPolicy::NearestFirst;
        let store = SharedVolumes::create(
            &volumes,
            spec.workflow.shape,
            spec.workflow.chunk,
            decomposition.n_phases(),
        )
        .expect("level files");
        store
            .write_level(0, &ramp(spec.workflow.shape))
            .expect("an input");
        drop(store);

        let run = local::run(&options(&dir, workers), &spec, &decomposition)
            .unwrap_or_else(|error| panic!("{workers} workers over a barrier: {error}"));
        assert_eq!(
            run.status.done, run.status.tasks,
            "{workers} workers: {} of {} tasks — a barrier that nobody claimed would \
             show up exactly here",
            run.status.done, run.status.tasks
        );
        let produced = output_bytes(&dir, &spec, &decomposition);
        assert!(
            produced == reference,
            "{workers} workers produced {} differing bytes across a barrier",
            produced
                .chunks(8)
                .zip(reference.chunks(8))
                .filter(|(left, right)| left != right)
                .count()
        );
        // Not `check_coverage_and_order`: that check takes **one** block count
        // for the whole run, and a barrier phase has a lattice of its own — one
        // block against the sixteen of the phase before it. It refuses a barrier
        // job whatever the worker count, single-node included, which is a
        // property of the check rather than of this run. What is asserted
        // instead is the part that is about distribution: the merged stream
        // shows the barrier op applied to its one block, exactly once, with
        // nobody having died.
        let log = merged_log(&run);
        let applications = log
            .op_sequence_per_block()
            .values()
            .flat_map(|sequence| sequence.iter())
            .filter(|(_, name)| name == "merge")
            .count();
        assert_eq!(
            applications, 1,
            "{workers} workers: the barrier op was applied {applications} times"
        );
        assert!(
            log.duplicate_applications().is_empty(),
            "{workers} workers: a block ran twice with nobody dying"
        );
        println!(
            "{workers} worker(s) over a barrier: {} tasks, {} events, {:?} — byte-identical",
            run.status.done, run.status.events, run.elapsed
        );
        std::fs::remove_dir_all(&dir).ok();
    }
    std::fs::remove_dir_all(&reference_dir).ok();
}

/// The same, across a phase boundary — where one worker reads an intermediate
/// another worker wrote, which is the case a single-node run cannot exercise at
/// all.
#[test]
fn several_workers_agree_with_one_node_across_a_phase_boundary() {
    let reference_dir = scratch("phases-reference");
    let (spec, decomposition) = probe_job_over(
        BLOCKS,
        2,
        ChainSpec::identity(),
        StoreSpec::Files {
            dir: reference_dir.join("volumes"),
        },
    );
    assert!(
        decomposition.n_phases() >= 2,
        "this test is about a phase boundary and there is not one"
    );
    let reference = single_node(&reference_dir, &spec, &decomposition);

    let dir = scratch("phases-distributed");
    let (spec, decomposition) = prepare(&dir, 2, None);
    let run = local::run(&options(&dir, 4), &spec, &decomposition).expect("a four-worker run");
    assert_eq!(run.status.done, run.status.tasks);
    assert_eq!(
        output_bytes(&dir, &spec, &decomposition),
        reference,
        "four workers across a phase boundary disagree with one node"
    );
    println!(
        "{} phases, {} tasks, 4 workers: byte-identical in {:?}",
        decomposition.n_phases(),
        run.status.tasks,
        run.elapsed
    );
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&reference_dir).ok();
}

// ---------------------------------------------------- 2. exactly-once --

#[test]
fn every_block_was_executed_exactly_once_across_the_merged_event_stream() {
    let dir = scratch("coverage");
    let (spec, decomposition) = prepare(&dir, 2, None);
    let run = local::run(&options(&dir, 4), &spec, &decomposition).expect("a four-worker run");

    // The coordinator ran the criterion itself, over what it holds.
    assert_eq!(
        run.report.get("coverage_ok").and_then(Value::as_bool),
        Some(true),
        "coverage: {:?}",
        run.report.get("coverage")
    );
    assert_eq!(
        run.report.get("unknown_events").and_then(Value::as_u64),
        Some(0),
        "the coordinator could not decode some of what its workers sent"
    );

    // And again here, from the exported stream, with the crate's own check.
    let log = merged_log(&run);
    log.check_coverage_and_order(
        &expected_ops(&decomposition),
        decomposition.phases[0].blocks.len(),
    )
    .expect("every block affected by every op, in the right order, once each");
    assert!(
        log.duplicate_applications().is_empty(),
        "with nobody dying, no block should have run twice: {:?}",
        log.duplicate_applications()
    );
    assert_eq!(run.status.reissued, 0, "nothing should have been reissued");
    println!(
        "{} tasks over {} workers: {} events merged, coverage and order hold",
        run.status.tasks, run.status.workers, run.status.events
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ------------------------------------------- 3. a death, by default --

/// Kill a worker mid-run, with the job exactly as a job comes: **no lease**.
///
/// The job must **stop**, and it must say what was lost. Not reissue the
/// claims — this deployment is 10-20 cooperating nodes and a lost one may have
/// held a great deal in memory, so re-running the tasks it had claimed restores
/// the claim table and not the position. And not hang, which is the outcome
/// this test exists to rule out: a run that never returns is the one failure
/// nobody can act on.
///
/// The detection is a **signal, not a timeout**. `local::run` started these
/// processes, so the operating system tells it when one dies; it relays that to
/// the coordinator, which names the worker and its claims and gives up. See the
/// `distributed` module header for why the coordinator cannot see this itself,
/// and why a duration would be the wrong instrument even if it could.
///
/// Timed, because "eventually" is not the claim: the run has to end in the
/// seconds after the kill, not at the harness timeout, and a hang that is
/// merely slow would otherwise pass.
#[test]
fn a_worker_that_is_killed_mid_run_aborts_the_job_by_default_and_names_what_was_lost() {
    let dir = scratch("loss");
    let (spec, decomposition) = prepare(&dir, 1, None);
    assert_eq!(spec.lease, None, "a job has no lease unless it asks");
    let mut options = options(&dir, 3);
    options.timeout = Duration::from_secs(60);
    options.kill_at_progress = vec![(0, 2)];
    let started = std::time::Instant::now();
    let outcome = local::run(&options, &spec, &decomposition);
    let took = started.elapsed();

    let error = outcome.err().unwrap_or_else(|| {
        panic!("a worker was killed and the run reported success; node loss is not recovered from")
    });
    let message = error.to_string();
    assert!(
        message.contains("worker-0"),
        "the failure does not say which worker was lost: {message}"
    );
    assert!(
        message.contains("aborting"),
        "the failure does not say the job gave up: {message}"
    );
    assert!(
        message.contains("does not recover"),
        "the failure does not say why it gave up: {message}"
    );
    // The whole point is that it *ended*. Generous against a loaded machine and
    // still nowhere near the timeout, which is what a hang would have cost.
    assert!(
        took < Duration::from_secs(45),
        "the run took {took:?}, which is close enough to the {:?} timeout to be a hang \
         rather than an abort",
        options.timeout
    );
    println!("a worker was killed two tasks in and the job aborted after {took:?}: {message}");
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------- 4. a death, under a lease --

/// Kill a worker mid-run and show the job completes with correct output.
///
/// `SIGKILL`, once the coordinator reports progress — not a timer, which would
/// race the job, and not a clean exit, which the worker would get to choose the
/// moment of. The process is taken wherever it is: possibly inside a task, with
/// a block part-written and nothing reported, and certainly holding the tasks
/// its work list was keeping ahead.
///
/// **Opt-in, and the explicit lease below is the documentation.** A job has no
/// lease by default, and the default answer to the death above is to abort. A
/// job that sets one is saying something different — that its tasks are cheap
/// enough to re-run and that it would rather carry on — and from the
/// coordinator the death then looks like what it always did: a claim that stops
/// being renewed, indistinguishable from a preemption, a spot reclamation or a
/// segfault. No failure detector and no membership protocol; the lease runs out
/// and the task goes to somebody else.
///
/// Any duplicate execution that follows is expected and harmless: both attempts
/// write the same values to the same valid region, so a reissue costs work and
/// never correctness.
#[test]
fn a_worker_that_is_killed_mid_run_has_its_tasks_reissued_and_the_output_is_still_right() {
    let reference_dir = scratch("death-reference");
    let (spec, decomposition) = probe_job_over(
        BLOCKS,
        1,
        ChainSpec::identity(),
        StoreSpec::Files {
            dir: reference_dir.join("volumes"),
        },
    );
    let reference = single_node(&reference_dir, &spec, &decomposition);

    let dir = scratch("death");
    // A finite lease, **set explicitly**, because no job has one by default.
    // That is the whole documentation value of this line: reissue is opt-in, a
    // default run holds a claim until it completes, and this test is the one
    // place in the suite that asks for the other thing. 400 ms so the test does
    // not sit out a long one; see the header for what that costs when nobody
    // has died.
    let (spec, decomposition) = prepare(&dir, 1, Some(Duration::from_millis(400)));
    let mut options = options(&dir, 3);
    options.kill_at_progress = vec![(0, 2)];
    let run = local::run(&options, &spec, &decomposition).expect("the survivors finish the job");

    assert_eq!(
        run.status.done, run.status.tasks,
        "the job did not finish after a worker died"
    );
    assert!(
        run.status.reissued >= 1,
        "a worker vanished holding work and nothing was reissued; either the lease never \
         expired or the claim was never made"
    );
    assert_eq!(
        output_bytes(&dir, &spec, &decomposition),
        reference,
        "the output after a death differs from a clean single-node run"
    );

    // Exactly the tasks that were reissued are the ones that appear twice —
    // which is what distinguishes "a block ran twice because a worker died"
    // from "a block ran twice because the coordinator lost track of it".
    //
    // Stated as a **subset**, not as a count. `duplicate_applications` is keyed
    // on `(block, op slot)`, so one block that ran twice contributes one entry
    // *per op in the chain*, while `reissued` is keyed on tasks; comparing the
    // two lengths compares (blocks x ops) against tasks and is wrong by the
    // chain length — with the two-op `identity` chain it demands that at least
    // half of every death's reissues were claims the dead worker had not yet
    // started, which is a property of how fast the machine happened to be and
    // not a property of the design. The subset is what the design does
    // guarantee, is independent of the chain length, and is strictly stronger:
    // it fails on a single block that ran twice without a recorded reissue.
    let reissued = run.reissued();
    assert!(!reissued.is_empty(), "no task was handed out twice");
    let graph = TaskGraph::build(&decomposition);
    let attempts: BTreeMap<[usize; 3], u64> = reissued
        .iter()
        .map(|&(task, attempts)| (graph.tasks[task].index, attempts))
        .collect();
    let log = merged_log(&run);
    let duplicated = log.duplicate_applications();
    let unexplained: BTreeSet<[usize; 3]> = duplicated
        .iter()
        .map(|&(index, _, _)| index)
        .filter(|index| !attempts.contains_key(index))
        .collect();
    assert!(
        unexplained.is_empty(),
        "block(s) {unexplained:?} were computed twice but their task was never handed out \
         twice: the coordinator lost track of work rather than reissuing it. Reissued: \
         {reissued:?}"
    );
    // And no block ran more times than its task was handed out. This is the
    // half that would catch a worker executing one assignment twice, which the
    // subset alone cannot see.
    for &(index, slot, count) in &duplicated {
        let handouts = attempts[&index];
        assert!(
            count as u64 <= handouts,
            "block {index:?} applied op {slot} {count} times on {handouts} handout(s)"
        );
    }
    let blocks: BTreeSet<[usize; 3]> = duplicated.iter().map(|&(index, _, _)| index).collect();
    println!(
        "a worker was killed two tasks in: {} claim(s) reissued {:?}, {} block(s) computed \
         twice {:?}, every one of them a task the coordinator had reissued, output \
         byte-identical to a clean single-node run",
        run.status.reissued,
        reissued,
        blocks.len(),
        blocks
    );
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&reference_dir).ok();
}

// ------------------------------------------------------- 5. no stalls --

/// A worker never blocks except on its own pull, and its work list stays ahead.
///
/// The regression this guards is specific and silent: a handout that only
/// answers when a worker is idle leaves the list empty at exactly the moment
/// prefetch needed to start, so **prefetch stops working multi-node while every
/// single-node test keeps passing**. `starved` counts a wait that happened
/// while the coordinator was known to have work, and it is the number that must
/// be zero.
#[test]
fn the_work_list_stays_at_least_one_task_ahead_of_what_is_being_computed() {
    let dir = scratch("pipeline");
    let (spec, decomposition) = prepare(&dir, 1, None);
    let workers = 2;
    let run = local::run(&options(&dir, workers), &spec, &decomposition).expect("a run");
    assert_eq!(run.status.done, run.status.tasks);
    assert_eq!(
        run.starved(),
        0,
        "a worker's list ran empty while the coordinator had work: {:?}",
        run.workers
    );
    // Every task after each worker's first should have started with the next
    // one already in hand.
    let waited: u64 = run
        .workers
        .iter()
        .filter_map(|report| report.get("started_after_waiting").and_then(Value::as_u64))
        .sum();
    assert!(
        waited <= workers as u64,
        "{waited} tasks started only after waiting, with {workers} workers; at most one \
         each — the first — is expected on a single-phase job"
    );
    assert!(
        run.started_ready() + waited as usize >= run.status.tasks,
        "the workers' own counters do not add up to the tasks the coordinator handed out"
    );
    println!(
        "{workers} workers, {} tasks: {} started with work in hand, {waited} after waiting, \
         {} starved",
        run.status.tasks,
        run.started_ready(),
        run.starved()
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// How deep should a worker's work list be? A measurement, not an assertion.
///
/// `ahead = 2` — one task being computed, one in hand — was chosen while the
/// coordinator was stamping a deadline on every claim, and under a lease a
/// deeper list is actively dangerous: the third task in a list is a claim that
/// has not been *started*, so it burns its lease sitting still, and the
/// contract `lease > (ahead + 1) x task duration` gets harder to satisfy with
/// every task added. **With no lease that constraint is gone**, and the depth
/// can be chosen for pipelining alone. So the question is open, and the honest
/// way to close it is to sweep it rather than to reason about it.
///
/// Two things make this measurable rather than merely printable.
///
/// **A noise control.** `ahead = 1` is not a configuration: the worker clamps
/// with `ahead.max(2)`, because a list of one is the list that cannot be ahead
/// of anything. So depth 1 and depth 2 run *identical code*, and the gap
/// between their two timings is this fixture's noise floor, measured under
/// exactly the conditions the rest of the sweep ran under. No difference
/// smaller than that floor means anything, and the first version of this test
/// reported a confident 40 % that was entirely inside it.
///
/// **Interleaving.** Depths are run round-robin within each repeat rather than
/// one depth at a time, so a busy minute on a shared machine is spread across
/// the whole curve instead of landing on whichever depth was unlucky.
///
/// What the numbers can and cannot say. This fixture is the friendliest
/// possible case for a deeper list: 8 x 8 x 8 blocks of `f64` through an
/// identity chain, so a task is microseconds of work behind a loopback round
/// trip, and handout latency is as large a share of the run as it can ever be.
/// If depth does not help *here* it will not help on real blocks, where a task
/// is seconds. If it does help here, the honest reading is "up to this much,
/// under this ratio", and not a new default.
///
/// What it said, 2026-08-17, `--release`, 40-core host, 9 interleaved runs per
/// depth, 256 tasks over 4 workers. Medians, against `ahead = 2`:
///
/// | ahead | median (ms) | vs 2 | starved |
/// |---|---|---|---|
/// | 1 *(control)* | 6991 | -0.3 % | 0 |
/// | 2 | 6968 | — | 0 |
/// | 4 | 7103 | -1.9 % | 0 |
/// | 8 | 7492 | -7.5 % | 0 |
/// | 16 | 7422 | -6.5 % | 0 |
/// | 32 | 7354 | -5.5 % | 0 |
/// | 64 | 7911 | -13.5 % | 0 |
///
/// **Deeper is not better; it is mildly worse, and the default stays at 2.**
/// The reason is in the last column and in `waited`: `starved` is already zero
/// at depth 2 and the only waits left are the unavoidable ones — a worker's
/// first task and a phase boundary — so there is no hunger for depth to feed.
/// What depth does instead is let one worker *claim* work it has not started,
/// which lengthens the tail: at 64 a worker can hold half a phase while another
/// runs dry at the end. The pipeline was already full at 2 and the only thing
/// left to move was the load balance, in the wrong direction.
///
/// Read it as a bound rather than a law: it is one fixture, and one whose tasks
/// are microseconds. It rules out the thing worth ruling out — that `ahead = 2`
/// was leaving pipelining on the table because a lease was watching — and it
/// says the lever is elsewhere.
///
/// Run it:
///
///     cargo test --release --features distributed --test local_multi_node -- \
///         --ignored --nocapture how_deep_the_work_list_should_be
#[test]
#[ignore = "a measurement, not an assertion; run with --release --ignored --nocapture"]
fn how_deep_the_work_list_should_be() {
    const BLOCKS_HERE: usize = 128;
    const PHASES: usize = 2;
    const WORKERS: usize = 4;
    const REPEATS: usize = 9;
    let depths = [1usize, 2, 4, 8, 16, 32, 64];

    let mut millis: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let mut ready: BTreeMap<usize, usize> = BTreeMap::new();
    let mut waited: BTreeMap<usize, u64> = BTreeMap::new();
    let mut starved: BTreeMap<usize, usize> = BTreeMap::new();
    let mut tasks = 0usize;

    for repeat in 0..REPEATS {
        for &ahead in &depths {
            let dir = scratch(&format!("ahead-{ahead}-{repeat}"));
            let volumes = dir.join("volumes");
            let (mut spec, decomposition) = probe_job_over(
                BLOCKS_HERE,
                PHASES,
                ChainSpec::identity(),
                StoreSpec::Files {
                    dir: volumes.clone(),
                },
            );
            spec.policy = HandoutPolicy::NearestFirst;
            let store = SharedVolumes::create(
                &volumes,
                spec.workflow.shape,
                spec.workflow.chunk,
                decomposition.n_phases(),
            )
            .expect("level files");
            store
                .write_level(0, &ramp(spec.workflow.shape))
                .expect("an input");
            let mut options = options(&dir, WORKERS);
            options.ahead = ahead;
            let run = local::run(&options, &spec, &decomposition).expect("a run");
            assert_eq!(run.status.done, run.status.tasks);
            // The coordinator's own clock, not the harness's: process spawn is
            // a fixed cost of the fixture and would flatten every difference
            // the sweep is looking for.
            let elapsed = run
                .report
                .get("elapsed_ms")
                .and_then(Value::as_u64)
                .expect("the coordinator times its own job") as f64;
            millis.entry(ahead).or_default().push(elapsed);
            *ready.entry(ahead).or_default() += run.started_ready();
            *waited.entry(ahead).or_default() += run
                .workers
                .iter()
                .filter_map(|report| report.get("started_after_waiting").and_then(Value::as_u64))
                .sum::<u64>();
            *starved.entry(ahead).or_default() += run.starved();
            tasks = run.status.tasks;
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    println!(
        "\n{BLOCKS_HERE} blocks x {PHASES} phases = {tasks} tasks over {WORKERS} workers, \
         {REPEATS} interleaved runs per depth, identity chain, 8x8x8 f64 blocks\n"
    );
    println!(
        "{:>6}  {:>9}  {:>9}  {:>9}  {:>10}  {:>8}  {:>8}",
        "ahead", "best (ms)", "med (ms)", "spread", "tasks/s", "waited", "starved"
    );
    let mut curve: Vec<(usize, f64, f64)> = Vec::new();
    for &ahead in &depths {
        let mut samples = millis[&ahead].clone();
        samples.sort_by(f64::total_cmp);
        let best = samples[0];
        let median = samples[samples.len() / 2];
        let worst = samples[samples.len() - 1];
        println!(
            "{ahead:>6}  {best:>9.0}  {median:>9.0}  {:>8.0} %  {:>10.0}  {:>8}  {:>8}",
            (worst - best) / best * 100.0,
            tasks as f64 / (median / 1000.0),
            waited[&ahead],
            starved[&ahead]
        );
        curve.push((ahead, median, best));
    }

    // The control, read out loud. Depth 1 and depth 2 are the same code, so
    // this is the smallest difference this fixture can honestly resolve.
    let one = curve[0].1;
    let two = curve[1].1;
    let floor = (one - two).abs() / two.max(one) * 100.0;
    println!(
        "\nnoise floor: ahead = 1 and ahead = 2 are the same configuration (the worker \
         clamps with max(2)) and their medians differ by {floor:.1} %. Nothing below that \
         is a result."
    );
    println!("\nagainst ahead = 2, on medians:");
    for &(ahead, median, _) in &curve {
        let change = (two - median) / two * 100.0;
        let verdict = if change.abs() <= floor {
            "inside the noise floor"
        } else if change > 0.0 {
            "faster"
        } else {
            "slower"
        };
        println!("  ahead = {ahead:>3}: {change:+6.1} %  ({verdict})");
    }
    // What a deeper list costs. The work list holds **descriptors, not data**:
    // an `Assignment` is a job id, four small integers and three regions, and
    // the block's voxels are read inside `execute_task_of` when the task runs,
    // one task at a time. So depth costs bytes per queued task and not blocks
    // per queued task — printed rather than asserted, because the number that
    // matters is how small it is.
    let assignment = std::mem::size_of::<blockflow::distributed::Assignment>();
    println!(
        "\nmemory: an Assignment is {assignment} bytes, so the deepest list swept ({}) costs \
         {} bytes per worker and {} across {WORKERS}. The queue holds descriptors and never \
         voxels — a block's inputs are read inside execute_task_of, one task at a time — so \
         a deeper list does not hold a block open.",
        depths[depths.len() - 1],
        assignment * depths[depths.len() - 1],
        assignment * depths[depths.len() - 1] * WORKERS,
    );
}

/// Observation never blocks the thing it observes.
///
/// The event stream is unbatched and sent as it happens, on its own connection,
/// by its own thread — so a coordinator that is slow to accept events delays
/// *events* and never work. The visible consequence is that a run's events all
/// arrive and its tasks all complete, with the event count far exceeding the
/// task count and no worker reporting a stall.
#[test]
fn events_are_sent_as_they_happen_without_holding_the_work_up() {
    let dir = scratch("events");
    let (spec, decomposition) = prepare(&dir, 1, None);
    let run = local::run(&options(&dir, 3), &spec, &decomposition).expect("a run");
    assert_eq!(run.status.done, run.status.tasks);
    // Every task emits exactly one `RegionRead`, one `BlockRead`, one
    // `OpApplied` per op, one `RegionWritten` and one `BlockWritten`. Asserting
    // the exact count rather than "more than the tasks" is what makes this a
    // check on the *completeness* of the merged record — and the record is what
    // the acceptance criterion is asserted from, so a stream that quietly lost
    // its tail would make every other check weaker without failing any of them.
    let per_task = 4 + spec.workflow.ops.len();
    assert_eq!(
        run.status.events,
        run.status.tasks * per_task,
        "{} events for {} tasks at {per_task} each: the merged stream is incomplete",
        run.status.events,
        run.status.tasks
    );
    assert_eq!(run.starved(), 0);
    let sent: u64 = run
        .workers
        .iter()
        .filter_map(|report| report.get("events").and_then(Value::as_u64))
        .sum();
    assert_eq!(
        sent as usize, run.status.events,
        "every event a worker sent should be in the merged stream"
    );
    assert_eq!(
        run.workers
            .iter()
            .filter_map(|report| report.get("listener_faults").and_then(Value::as_u64))
            .sum::<u64>(),
        0
    );
    println!(
        "{} events from 3 workers over {} tasks, all merged, none batched",
        run.status.events, run.status.tasks
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The two lifetimes are the same program.
///
/// A **persistent** coordinator is started with no job and outlives them; a
/// **per-run** one is the same binary with `--job` and `--exit-when-done`. This
/// exercises the persistent shape end to end — submit over HTTP, run workers
/// against it, and find it still there afterwards — because everything else in
/// this file exercises the per-run one.
#[test]
fn a_persistent_coordinator_accepts_a_job_over_http_and_outlives_it() {
    use blockflow::distributed::client::Client;
    use blockflow::distributed::coordinator::Coordinator;
    use blockflow::distributed::protocol::path;
    use blockflow::distributed::server::{self, Options};
    use blockflow::distributed::worker::{self, WorkerOptions};

    let dir = scratch("persistent");
    let (spec, decomposition) = prepare(&dir, 1, None);

    let coordinator = Arc::new(Coordinator::new(false));
    let handle = server::serve(
        coordinator.clone(),
        Options {
            bind: "127.0.0.1:0".parse().unwrap(),
            ..Default::default()
        },
    )
    .expect("a coordinator on loopback");
    let addr = handle.bound();

    // No jobs yet, and a worker would have nothing to join.
    let mut client = Client::new(addr);
    assert!(client.post(path::JOIN, &serde_json::json!({})).is_err());

    let submitted = client
        .post(path::SUBMIT, &spec.to_json(&decomposition).unwrap())
        .expect("a job may be submitted to a running coordinator");
    assert_eq!(
        submitted.get("tasks").and_then(Value::as_u64),
        Some(decomposition.n_tasks() as u64)
    );

    let mut options = WorkerOptions::new(addr);
    options.name = Some("in-process".to_string());
    let report = worker::run(options, &ProbeWorkflows).expect("a worker finishes the job");
    assert_eq!(report.tasks, decomposition.n_tasks());
    assert_eq!(report.starved, 0);

    // The job is finished; the coordinator is not.
    assert!(coordinator.all_finished());
    assert!(
        !coordinator.should_exit(),
        "a persistent coordinator must not exit because a job finished"
    );
    let jobs = client.get(path::JOBS).expect("the registry answers");
    assert_eq!(
        jobs.get("exit_when_done").and_then(Value::as_bool),
        Some(false)
    );
    // A second job on the same coordinator.
    let second_dir = scratch("persistent-second");
    let (mut second, second_decomposition) = prepare(&second_dir, 1, None);
    second.id = "second".to_string();
    client
        .post(
            path::SUBMIT,
            &second.to_json(&second_decomposition).unwrap(),
        )
        .expect("a second job");
    assert_eq!(coordinator.job_ids().len(), 2);

    handle.shutdown();
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&second_dir).ok();
}

/// A coordinator refuses to publish itself on a network address unless it was
/// asked to by name.
///
/// It is the one server in this crate that genuinely needs a public bind, and
/// that is the reason to keep the gate rather than to remove it: there is no
/// authentication, so a coordinator on `0.0.0.0` lets anyone on the network
/// read the job's plan and take work from it.
#[test]
fn a_coordinator_will_not_bind_a_public_address_by_accident() {
    use blockflow::distributed::coordinator::Coordinator;
    use blockflow::distributed::server::{self, Options};

    let refused = server::serve(
        Arc::new(Coordinator::new(false)),
        Options {
            bind: "0.0.0.0:0".parse().unwrap(),
            allow_public: false,
            ..Default::default()
        },
    );
    let error = match refused {
        Ok(_) => panic!("a public bind should need --allow-public"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("not a loopback address"), "{error}");
    assert!(error.contains("--allow-public"), "{error}");
}

// ------------------------------------------- 6. non-pixel per-block output --
//
// The claim: **fragments written by N worker processes are all readable by one
// merging reader.** It is the property that makes a block-keyed sidecar store
// worth building at all — a per-block value held in memory pins its stage to one
// node, and a per-block value on shared storage does not.
//
// The merge is deliberately a *global reduction* and deliberately outside the
// framework: it is not block-parallel, the task DAG has no fan-in node to
// express it with, and the caller is the one who knows what the bytes mean. So
// the test does what a caller does — run the block phase, read the fragments,
// reduce them.

/// The store, opened by a process that ran no tasks: the merging reader.
fn merging_reader(dir: &Path, spec: &JobSpec, decomposition: &Decomposition) -> SharedVolumes {
    SharedVolumes::open(
        &dir.join("volumes"),
        spec.workflow.shape,
        spec.workflow.chunk,
        decomposition.n_phases(),
    )
    .expect("the volumes")
}

/// Reduce every fragment into `phase -> (blocks, voxels)`. A global reduction
/// over per-block non-pixel output, which is exactly the shape the graph stages
/// have and exactly what could not be done across nodes before.
fn merge_fragments(
    store: &SharedVolumes,
    stream: &str,
) -> std::collections::BTreeMap<usize, (std::collections::BTreeSet<[usize; 3]>, usize)> {
    let mut merged: std::collections::BTreeMap<
        usize,
        (std::collections::BTreeSet<[usize; 3]>, usize),
    > = std::collections::BTreeMap::new();
    for (key, bytes) in store.sidecar_fragments(stream).expect("the fragments") {
        let (block, voxels) = read_task_fragment(&bytes).expect("a task fragment");
        assert_eq!(
            block, key.block,
            "a fragment's payload disagrees with the key it was stored under"
        );
        let entry = merged.entry(key.phase).or_default();
        entry.0.insert(block);
        entry.1 += voxels;
    }
    merged
}

#[test]
fn fragments_written_by_several_worker_processes_are_readable_by_one_merging_reader() {
    const PHASES: usize = 2;
    for workers in [1usize, 3, 5] {
        let dir = scratch(&format!("sidecar-workers-{workers}"));
        let (mut spec, decomposition) = prepare(&dir, PHASES, None);
        spec.workflow.sidecar = Some(SidecarSpec::new("fragments", Lifecycle::DeleteOnExit));
        let run = local::run(&options(&dir, workers), &spec, &decomposition)
            .unwrap_or_else(|error| panic!("{workers} workers: {error}"));
        assert_eq!(run.status.done, run.status.tasks);

        // Several *processes* wrote, not one — the whole point of the exercise.
        let per_worker = run.fragments_per_worker();
        assert_eq!(
            per_worker.iter().sum::<usize>(),
            decomposition.n_tasks(),
            "{workers} workers wrote {per_worker:?} fragments for {} tasks",
            decomposition.n_tasks()
        );
        if workers > 1 {
            assert!(
                per_worker.iter().filter(|count| **count > 0).count() > 1,
                "{workers} workers but only one produced fragments ({per_worker:?}); this run \
                 proves nothing about several processes"
            );
        }

        // One reader, which ran nothing, sees all of it.
        let reader = merging_reader(&dir, &spec, &decomposition);
        let merged = merge_fragments(&reader, "fragments");
        assert_eq!(
            merged.len(),
            PHASES,
            "{workers} workers: fragments arrived for phases {:?}",
            merged.keys().collect::<Vec<_>>()
        );
        let voxels: usize = spec.workflow.shape.iter().product();
        for (phase, (blocks, summed)) in &merged {
            assert_eq!(
                blocks.len(),
                decomposition.phases[*phase].grid.n_blocks(),
                "{workers} workers, phase {phase}: not every block left a fragment"
            );
            // The reduction says something true about the whole volume that no
            // single fragment says: the valid regions tile it exactly.
            assert_eq!(
                *summed, voxels,
                "{workers} workers, phase {phase}: the merged fragments cover {summed} of \
                 {voxels} voxels"
            );
        }

        // And the writes are in the merged event stream, having travelled from
        // every worker to the coordinator like any other event.
        let written = merged_log(&run)
            .events()
            .into_iter()
            .filter(|event| matches!(event, blockflow::log::Event::SidecarWritten { .. }))
            .count();
        assert_eq!(
            written,
            decomposition.n_tasks(),
            "{workers} workers: {written} sidecar writes on the merged stream"
        );

        // Lifecycle: the stream said delete-on-exit, so it goes, and the
        // removal reports itself rather than being taken on trust.
        let report = reader.discard_sidecars().expect("a discard");
        assert_eq!(report.removed.len(), 1);
        assert_eq!(report.removed[0].stream, "fragments");
        assert_eq!(report.removed[0].fragments, decomposition.n_tasks());
        assert_eq!(report.removed[0].bytes, decomposition.n_tasks() as u64 * 32);
        assert!(report.kept.is_empty());
        assert!(
            !SharedVolumes::sidecar_root(&dir.join("volumes"))
                .join("fragments")
                .exists(),
            "a delete-on-exit stream is still on disk after a discard that claimed to remove it"
        );
        println!(
            "{workers} worker(s): {} fragments merged from {per_worker:?}, then {}",
            decomposition.n_tasks(),
            report.describe()
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// The other lifecycle, and it is not a default: a stream declared persistent
/// survives the same call that removes a delete-on-exit one.
#[test]
fn a_persistent_sidecar_stream_survives_the_discard_that_removes_a_delete_on_exit_one() {
    let dir = scratch("sidecar-persistent");
    let (mut spec, decomposition) = prepare(&dir, 1, None);
    spec.workflow.sidecar = Some(SidecarSpec::new("keepsake", Lifecycle::Persistent));
    let run = local::run(&options(&dir, 3), &spec, &decomposition).expect("a run");
    assert_eq!(run.status.done, run.status.tasks);

    let reader = merging_reader(&dir, &spec, &decomposition);
    // A second stream, declared the other way by the process that will do the
    // cleaning up. Both are in one store; only one is meant to go.
    reader
        .declare_sidecar("scratchpad", Lifecycle::DeleteOnExit)
        .expect("a declaration");
    reader
        .write_sidecar("scratchpad", 0, [0, 0, 0], b"working notes")
        .expect("a fragment");

    let report = reader.discard_sidecars().expect("a discard");
    assert_eq!(
        report
            .removed
            .iter()
            .map(|entry| &entry.stream)
            .collect::<Vec<_>>(),
        vec!["scratchpad"]
    );
    assert_eq!(report.kept, vec!["keepsake".to_string()]);
    assert_eq!(
        reader.sidecar_keys("keepsake").expect("the keys").len(),
        decomposition.n_tasks(),
        "a persistent stream must be untouched by a discard"
    );
    assert!(reader
        .sidecar_keys("scratchpad")
        .expect("the keys")
        .is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

// ------------------------------------------- 7. fragment ops across nodes --
//
// The claim the sidecar store existed for but could not yet demonstrate: **a
// fragment written by one worker process is read by another, because the plan
// said so.**
//
// Test 5 above showed the *storage* half — many producers, one merging reader —
// with the fragments produced as a job-level side effect rather than by an op.
// This is the half that needed `fragment::FragmentOp`: a phase whose input is
// another phase's fragments, with a declared reach in blocks, so that a worker
// running block `b` must read blocks `b-1 .. b+1` and those blocks were run by
// whichever workers the coordinator gave them to. Nothing coordinates that but
// the task DAG and shared storage, which is exactly what is under test.
//
// How "it crossed a process boundary" is *measured* rather than assumed: the
// summary op stamps the producing process's id on every fragment it writes, and
// the fold records how many distinct stamps it saw. A block reporting two or
// more folded fragments from more than one process, and there is no way for
// that to happen inside one address space.

/// The chain's phases, plus a `volume -> fragments` phase and a
/// `fragments -> fragments` phase that reaches one block either way.
fn fragment_job(dir: &Path, reach: [usize; 3]) -> (JobSpec, Decomposition) {
    let volumes = dir.join("volumes");
    let (mut spec, pixels_only) = probe_job_over(
        BLOCKS,
        1,
        ChainSpec::identity(),
        StoreSpec::Files {
            dir: volumes.clone(),
        },
    );
    spec.policy = HandoutPolicy::NearestFirst;
    spec.lease = None;
    let summary_phase = pixels_only.n_phases();
    spec.workflow.fragment_phases = vec![
        FragmentPhaseSpec::summary("summary", "fragments", Lifecycle::DeleteOnExit),
        FragmentPhaseSpec::fold(
            "fold",
            "fragments",
            summary_phase,
            reach,
            "folded",
            Lifecycle::DeleteOnExit,
        ),
    ];
    let decomposition = blockflow::distributed::spec::decompose(&spec, 1)
        .expect("a job with fragment phases decomposes");
    let store = SharedVolumes::create(
        &volumes,
        spec.workflow.shape,
        spec.workflow.chunk,
        decomposition.n_phases(),
    )
    .expect("level files");
    store
        .write_level(0, &ramp(spec.workflow.shape))
        .expect("an input");
    (spec, decomposition)
}

#[test]
fn a_fragment_written_by_one_worker_is_read_by_another_when_the_reach_demands_it() {
    const REACH: [usize; 3] = [1, 0, 0];
    for workers in [1usize, 3, 5] {
        let dir = scratch(&format!("fragment-ops-{workers}"));
        let (spec, decomposition) = fragment_job(&dir, REACH);
        let phases = decomposition.n_phases();
        let summary_phase = phases - 2;
        let fold_phase = phases - 1;

        let run = local::run(&options(&dir, workers), &spec, &decomposition)
            .unwrap_or_else(|error| panic!("{workers} workers: {error}"));
        assert_eq!(run.status.done, run.status.tasks);

        let reader = merging_reader(&dir, &spec, &decomposition);
        let lattice = decomposition.phases[summary_phase].grid.blocks_per_axis();
        let blocks: std::collections::BTreeSet<[usize; 3]> = decomposition.phases[summary_phase]
            .grid
            .cores()
            .into_iter()
            .map(|core| core.index)
            .collect();

        // Coverage, from what the store holds...
        for (stream, phase) in [("fragments", summary_phase), ("folded", fold_phase)] {
            let held: std::collections::BTreeSet<[usize; 3]> = reader
                .sidecar_keys(stream)
                .expect("the keys")
                .into_iter()
                .filter(|key| key.phase == phase)
                .map(|key| key.block)
                .collect();
            assert_eq!(
                held, blocks,
                "{workers} workers: stream {stream:?} does not cover phase {phase}'s lattice"
            );
        }
        // ...and from the merged event stream, which travelled from every
        // worker to the coordinator like any other event.
        let log = merged_log(&run);
        let written: std::collections::BTreeSet<([usize; 3], usize)> = log
            .events()
            .into_iter()
            .filter_map(|event| match event {
                blockflow::log::Event::SidecarWritten {
                    stream,
                    phase,
                    index,
                    ..
                } if stream == "folded" => Some((index, phase)),
                _ => None,
            })
            .collect();
        assert_eq!(
            written.len(),
            blocks.len(),
            "{workers} workers: the merged stream carries {} fold writes for {} blocks",
            written.len(),
            blocks.len()
        );

        // The fragment-side guard, run by a process that executed no task —
        // which is where it has to run in a distributed job, because no single
        // worker sees a whole phase. `execute_phases` runs it automatically on
        // one node; here the merging reader is the only party that can.
        for (index, phase_spec) in spec.workflow.fragment_phases.iter().enumerate() {
            let op = phase_spec.build(0).expect("a fragment op");
            let report = blockflow::fragment::check_fragment_coverage(
                &reader,
                &decomposition,
                summary_phase + index,
                op.as_ref(),
            )
            .unwrap_or_else(|error| panic!("{workers} workers: {error}"));
            for stream in &report {
                assert_eq!(stream.blocks, stream.lattice, "{}", stream.describe());
            }
        }

        // The reach, per block, against the analytic neighbourhood — and the
        // number of distinct *processes* whose fragments each block folded.
        let mut crossings = 0usize;
        let mut folded_total = 0usize;
        for block in &blocks {
            let bytes = reader
                .read_sidecar("folded", fold_phase, *block)
                .expect("a fragment")
                .expect("every block folded");
            let (at, seen, voxels, producers) =
                NeighbourFoldOp::read(&bytes).expect("a fold payload");
            assert_eq!(at, *block);
            assert_eq!(
                seen,
                neighbourhood_size(*block, REACH, lattice),
                "{workers} workers: block {block:?} folded {seen} fragments"
            );
            folded_total += voxels;
            if producers > 1 {
                crossings += 1;
            }
        }
        // Every fragment in every neighbourhood, summed: each block's voxels
        // counted once per neighbour that read it.
        let per_block: usize = decomposition.phases[summary_phase]
            .grid
            .block()
            .iter()
            .product();
        let analytic: usize = blocks
            .iter()
            .map(|block| neighbourhood_size(*block, REACH, lattice) * per_block)
            .sum();
        assert_eq!(folded_total, analytic);

        if workers > 1 {
            assert!(
                crossings > 0,
                "{workers} workers: no block folded fragments from more than one process, so \
                 this run does not demonstrate cross-node fragment visibility. Fragments per \
                 worker: {:?}",
                run.fragments_per_worker()
            );
        }
        println!(
            "{workers} worker(s): {} of {} blocks folded fragments from more than one \
             process; fragments per worker {:?}",
            crossings,
            blocks.len(),
            run.fragments_per_worker()
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// The zero-reach case, across nodes: a phase that declares no reach must read
/// exactly its own block's fragment. Asserted from the fold's own count, which
/// is the measured number rather than the declared one.
#[test]
fn a_zero_reach_fragment_phase_reads_no_neighbour_across_nodes_either() {
    let dir = scratch("fragment-ops-zero-reach");
    let (spec, decomposition) = fragment_job(&dir, [0, 0, 0]);
    let fold_phase = decomposition.n_phases() - 1;
    let run = local::run(&options(&dir, 4), &spec, &decomposition).expect("a run");
    assert_eq!(run.status.done, run.status.tasks);

    let reader = merging_reader(&dir, &spec, &decomposition);
    for core in decomposition.phases[fold_phase].grid.cores() {
        let bytes = reader
            .read_sidecar("folded", fold_phase, core.index)
            .expect("a fragment")
            .expect("every block folded");
        let (at, seen, _, producers) = NeighbourFoldOp::read(&bytes).expect("a fold payload");
        assert_eq!(at, core.index);
        assert_eq!(
            seen, 1,
            "block {:?} read {seen} fragments at zero reach",
            at
        );
        assert_eq!(producers, 1, "one fragment cannot come from two processes");
    }
    std::fs::remove_dir_all(&dir).ok();
}
