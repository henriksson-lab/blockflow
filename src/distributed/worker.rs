// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The worker. It plans nothing.
//
// Its whole life is *connect to the coordinator, pull a task, execute it,
// report*. Every decision that could have been made here — which task, in what
// order, on which node, with what halo — was made by the coordinator or fixed by
// the decomposition, and the reason is not tidiness: **hints inform the
// coordinator's choice; workers receive certainty.** A worker that guessed would
// be a second scheduler, and two schedulers on one job is how a task gets run
// twice or not at all.
//
// The one thing a worker does have to get right
// ---------------------------------------------
// **Its work list must stay at least one task ahead of what it is computing.**
// That is what lets the prefetcher fetch block N+1 while block N is being
// computed, exactly as a single-node run does — the prefetcher is *unchanged*
// from single-node, because the design's argument for it (prefetch here is
// hint-driven rather than predictive, because the work is enumerated rather
// than guessed) holds whether the list was derived up front or delivered
// incrementally.
//
// The failure mode to avoid is a handout that only answers when a worker is
// idle. The list would then be empty at exactly the moment prefetch needed to
// start, and **prefetch would silently stop working multi-node while every
// single-node test continued to pass.** So this file counts it: `starved` is
// the number of times the loop had to wait for a task *while the coordinator
// was known to have work*, and it is asserted to be zero rather than assumed.
//
// Three threads, so that none of them can wait behind another
// -----------------------------------------------------------
// * the **executor** — pops a task, runs it through the crate's own
//   `execute_task`, flushes, reports it complete;
// * the **puller** — keeps the work list topped up, independently of whether
//   the executor is busy. This is what "one request deep, pipelined" means;
// * the **reporter** — drains events and sends them, one per message, as they
//   happen. It has its own connection, so a slow coordinator delays *events*
//   and never the work.
//
// The listener the executor sees does one thing: clone the event and push it
// onto a queue. Observation must never be able to slow down the thing it
// observes, and an event sender that blocked would be exactly that.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::json;

use crate::decomposition::Decomposition;
use crate::error::{Error, Result};
use crate::export::event_json;
use crate::graph::TaskGraph;
use crate::listener::EventListener;
use crate::log::Event;
use crate::fragment::{check_phase_work, PhaseWork};
use crate::strategy::execute_task_of;

use super::client::Client;
use super::protocol::{path, Assignment, Handout, Joined, PROTOCOL_VERSION};
use super::spec::{task_fragment, JobSpec, WorkflowFactory};
use super::wire::decomposition_from_json;

#[derive(Debug, Clone)]
pub struct WorkerOptions {
    pub addr: SocketAddr,
    /// Which job. `None` takes the only one there is, which is the per-run
    /// shape.
    pub job: Option<String>,
    /// A name, for logs. `None` lets the coordinator allocate one.
    pub name: Option<String>,
    /// How many tasks to keep in hand. Two — one being computed, one ready —
    /// is the design's "at least one ahead"; more only deepens the pipeline and
    /// makes a reissue after a death more expensive.
    pub ahead: usize,
    /// Stop after this many tasks, whatever the job says. Only for making a
    /// worker die on cue in a test; a real worker runs until the job is done.
    pub stop_after: Option<usize>,
    pub verbose: bool,
}

impl WorkerOptions {
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            job: None,
            name: None,
            ahead: 2,
            stop_after: None,
            verbose: false,
        }
    }
}

/// What one worker did, and how well the pipeline held up.
#[derive(Debug, Clone, Default)]
pub struct WorkerReport {
    pub worker: String,
    pub job: String,
    pub tasks: usize,
    pub short_circuited: usize,
    /// Sidecar fragments this worker wrote — per-block output that is not a
    /// pixel region. Zero unless the job declared a stream.
    pub fragments: usize,
    pub events: usize,
    /// Tasks that started with the next one already in hand. The healthy case.
    pub started_ready: usize,
    /// Tasks that started only after waiting for a handout. One of these is
    /// expected — the first — and a phase boundary can legitimately produce
    /// more.
    pub started_after_waiting: usize,
    /// Waits that happened **while the coordinator was known to have work
    /// available**. This is the regression the design warns about, and it is
    /// the number to assert zero.
    pub starved: usize,
    pub reads: u64,
    pub chunks_read: u64,
    pub elapsed: Duration,
    pub listener_faults: usize,
}

/// What the puller last heard, so the executor can tell a legitimate empty list
/// from a starved one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LastReply {
    Nothing,
    Work,
    Blocked,
    Finished,
}

struct Shared {
    queue: Mutex<VecDeque<Assignment>>,
    arrived: Condvar,
    last: Mutex<LastReply>,
    /// No more work to pull. Ends the puller and, once the list is empty, the
    /// executor.
    done: AtomicBool,
    /// The executor has stopped, so no further event can be produced.
    ///
    /// Separate from `done` on purpose: `done` can be set by the *puller* while
    /// the executor is still in the middle of a task, and a reporter that
    /// exited on `done` would drop that task's events. This flag is set by the
    /// executor and by nobody else.
    quiet: AtomicBool,
    events: Mutex<VecDeque<Event>>,
    events_ready: Condvar,
    sent: AtomicUsize,
}

/// The executor's only view of the event stream: clone and enqueue.
struct Outbox {
    shared: Arc<Shared>,
}

impl EventListener for Outbox {
    fn on_event(&self, event: &Event) {
        // No serialisation, no IO, no waiting on anything but an uncontended
        // push. Whatever the coordinator is doing, this returns.
        self.shared
            .events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push_back(event.clone());
        self.shared.events_ready.notify_one();
    }
}

fn guard<T>(lock: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Join, run until the job is finished, report.
pub fn run(options: WorkerOptions, factory: &dyn WorkflowFactory) -> Result<WorkerReport> {
    let started = Instant::now();
    let mut client = Client::new(options.addr);
    let joined = Joined::from_json(&client.post(
        path::JOIN,
        &json!({
            "job": options.job.clone().unwrap_or_default(),
            "worker": options.name.clone().unwrap_or_default(),
            "protocol": PROTOCOL_VERSION,
        }),
    )?)?;
    if joined.protocol != PROTOCOL_VERSION {
        return Err(Error::invalid(format!(
            "the coordinator speaks protocol {} and this worker speaks {PROTOCOL_VERSION}. \
             Run the same build on every node.",
            joined.protocol
        )));
    }

    let spec = JobSpec::from_json(&joined.spec)?;
    let decomposition = decomposition_from_json(
        joined
            .spec
            .get("decomposition")
            .ok_or_else(|| Error::invalid("the job carries no decomposition"))?,
    )?;
    // Binding, and honoured exactly: the graph is rebuilt from the
    // decomposition rather than derived from anything local, so every worker
    // and the coordinator agree on what task 4 011 is by construction.
    let graph = TaskGraph::build(&decomposition);
    let chain = factory.chain(&spec.workflow)?;
    // A tag that identifies *this process*, so a probe that stamps it on its
    // fragments makes cross-node visibility measurable rather than assumed.
    let fragment_ops = factory.fragment_ops(&spec.workflow, std::process::id() as u64)?;
    let pixel_phases = decomposition.n_phases().checked_sub(fragment_ops.len()).ok_or_else(|| {
        Error::invalid(format!(
            "the job names {} fragment phase(s) but its decomposition has only {} phase(s).              Both are built from the same spec, so this is a coordinator and a worker              disagreeing about the job.",
            fragment_ops.len(),
            decomposition.n_phases()
        ))
    })?;
    let work: Vec<PhaseWork> = (0..decomposition.n_phases())
        .map(|phase| match phase.checked_sub(pixel_phases) {
            None => PhaseWork::Pixels,
            Some(index) => PhaseWork::Fragments(fragment_ops[index].as_ref()),
        })
        .collect();
    // The same whole-plan guard `execute_phases` runs, run once here for the
    // same reason `Decomposition::check` is: a worker that would produce
    // something unreadable should say so before it starts, not per task.
    check_phase_work(&decomposition, &work)?;
    let environment = factory.environment(&spec.workflow, decomposition.n_phases())?;
    environment.prepare(&decomposition)?;
    // Declared by every worker, idempotently, exactly as the job's own sidecar
    // stream is: a worker that joins late or restarts declares the same thing.
    for entry in &work {
        if let PhaseWork::Fragments(op) = entry {
            for output in op.outputs() {
                environment.declare_sidecar(&output.stream, output.lifecycle)?;
            }
        }
    }
    // Declared here, by every worker, rather than once by the coordinator.
    // Declaration is idempotent for the same lifecycle precisely so that it
    // needs no coordination — a worker that joins late, or is restarted after
    // a death, declares the same thing and carries on.
    if let Some(sidecar) = &spec.workflow.sidecar {
        environment.declare_sidecar(&sidecar.stream, sidecar.lifecycle)?;
    }

    let shared = Arc::new(Shared {
        queue: Mutex::new(VecDeque::new()),
        arrived: Condvar::new(),
        last: Mutex::new(LastReply::Nothing),
        done: AtomicBool::new(false),
        quiet: AtomicBool::new(false),
        events: Mutex::new(VecDeque::new()),
        events_ready: Condvar::new(),
        sent: AtomicUsize::new(0),
    });

    let puller = spawn_puller(&options, &joined, shared.clone())?;
    let reporter = spawn_reporter(&options, &joined, shared.clone())?;

    let listeners: Vec<Arc<dyn EventListener>> = vec![Arc::new(Outbox {
        shared: shared.clone(),
    })];
    // The sidecar store is not on the executor's path, so it does not receive
    // the listener slice the way `execute_task` does. Attaching here is what
    // puts fragment writes in the same merged event stream as everything else,
    // rather than in a second record nobody merges.
    if let Some(store) = environment.sidecars() {
        store.attach_all(&listeners);
    }
    let mut report = WorkerReport {
        worker: joined.worker.clone(),
        job: joined.job.clone(),
        ..Default::default()
    };
    let mut completions = Client::new(options.addr);

    loop {
        let Some(assignment) = next_task(&shared, &mut report) else {
            break;
        };
        check_agreement(&graph, &assignment, &decomposition)?;
        let task = &graph.tasks[assignment.task];
        match execute_task_of(
            &chain,
            &decomposition,
            task,
            &work[task.phase],
            environment.as_ref(),
            &listeners,
        ) {
            Ok(outcome) => {
                // Durability before the completion, not a barrier after it: a
                // dependent task is released by the coordinator only once this
                // message arrives, so flushing first is what makes reading the
                // intermediate safe without any worker ever waiting for another.
                environment.finish(task.phase + 1)?;
                // The task's non-pixel output, keyed by the block it came from.
                //
                // Written before the completion for the same reason the level
                // is flushed before it, but note the barrier is *weaker* here
                // and legitimately so: no task ever reads a peer's fragment.
                // Fragments are consumed by a global merge after the job, so
                // they are never on a dependency path and a worker never waits
                // for another worker's fragment.
                if let Some(sidecar) = &spec.workflow.sidecar {
                    environment.write_sidecar(
                        &sidecar.stream,
                        task.phase,
                        task.index,
                        &task_fragment(task.index, outcome.valid.voxels()),
                    )?;
                }
                report.tasks += 1;
                report.listener_faults += outcome.listener_faults;
                if outcome.short_circuited {
                    report.short_circuited += 1;
                }
                completions.post(
                    path::COMPLETED,
                    &json!({
                        "job": joined.job,
                        "worker": joined.worker,
                        "task": assignment.task,
                    }),
                )?;
            }
            Err(error) => {
                // Say so rather than dying quietly. A reported failure returns
                // the task immediately instead of waiting out a lease.
                let _ = completions.post(
                    path::FAILED,
                    &json!({
                        "job": joined.job,
                        "worker": joined.worker,
                        "task": assignment.task,
                        "why": error.to_string(),
                    }),
                );
                shared.done.store(true, Ordering::Release);
                shared.quiet.store(true, Ordering::Release);
                shared.arrived.notify_all();
                shared.events_ready.notify_all();
                let _ = puller.join();
                let _ = reporter.join();
                return Err(error);
            }
        }
        if options
            .stop_after
            .is_some_and(|limit| report.tasks >= limit)
        {
            break;
        }
    }

    shared.done.store(true, Ordering::Release);
    shared.quiet.store(true, Ordering::Release);
    shared.arrived.notify_all();
    shared.events_ready.notify_all();
    let _ = puller.join();
    let _ = reporter.join();

    let (reads, _, _, _, chunks_read, _, _) = environment.counters().snapshot();
    // Every fragment this process wrote, however it was produced: the job's own
    // sidecar stream and any fragment op's outputs count the same way, because
    // both go through `Environment::write_sidecar`.
    let (fragments, _, _, _) = environment.counters().sidecar_snapshot();
    report.fragments = fragments as usize;
    report.reads = reads;
    report.chunks_read = chunks_read;
    report.events = shared.sent.load(Ordering::SeqCst);
    report.elapsed = started.elapsed();
    if report.starved > 0 {
        // The design asks for this to be loud, because no single-node test
        // would catch it: prefetch would keep working on one node and silently
        // stop working on many.
        eprintln!(
            "worker {}: the work list ran empty {} time(s) while the coordinator had work. \
             Prefetch cannot start a block until the block is in the list, so this is a \
             pipeline regression, not a scheduling detail.",
            report.worker, report.starved
        );
    }
    Ok(report)
}

/// Take the next task, recording whether it was already in hand.
fn next_task(shared: &Arc<Shared>, report: &mut WorkerReport) -> Option<Assignment> {
    {
        let mut queue = guard(&shared.queue);
        if let Some(assignment) = queue.pop_front() {
            report.started_ready += 1;
            return Some(assignment);
        }
    }
    // The list is empty. Whether that is a problem depends on *why*: at the
    // start there is nothing to be ahead of, at a phase boundary the
    // coordinator has nothing to give, and at the end there is nothing left.
    // Only "the coordinator had work and we still went hungry" is a regression.
    let known_available = *guard(&shared.last) == LastReply::Work;
    if known_available && report.started_ready + report.started_after_waiting > 0 {
        report.starved += 1;
    }
    let mut queue = guard(&shared.queue);
    loop {
        if let Some(assignment) = queue.pop_front() {
            report.started_after_waiting += 1;
            return Some(assignment);
        }
        if shared.done.load(Ordering::Acquire) {
            return None;
        }
        let (next, _) = shared
            .arrived
            .wait_timeout(queue, Duration::from_millis(20))
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        queue = next;
    }
}

/// A worker rebuilds the DAG from the decomposition it was given, and the
/// coordinator hands out ids into that same DAG. If the two ever disagreed the
/// worker would compute the right thing for the wrong block, which is the
/// silent, well-formed, wrong failure the whole crate is arranged to prevent —
/// so it is checked rather than trusted.
fn check_agreement(
    graph: &TaskGraph,
    assignment: &Assignment,
    decomposition: &Decomposition,
) -> Result<()> {
    let task = graph.tasks.get(assignment.task).ok_or_else(|| {
        Error::invalid(format!(
            "the coordinator handed out task {} of a graph with {} tasks",
            assignment.task,
            graph.len()
        ))
    })?;
    if task.phase != assignment.phase
        || task.index != assignment.index
        || task.geometry.read != assignment.read
        || task.geometry.valid != assignment.valid
    {
        return Err(Error::invalid(format!(
            "task {} is (phase {}, block {:?}, read {:?}) here and (phase {}, block {:?}, \
             read {:?}) at the coordinator, over a decomposition fingerprinting {}. The two \
             ends built different graphs from the same plan.",
            assignment.task,
            task.phase,
            task.index,
            task.geometry.read,
            assignment.phase,
            assignment.index,
            assignment.read,
            decomposition.fingerprint()
        )));
    }
    Ok(())
}

/// Keep the work list topped up, whatever the executor is doing.
fn spawn_puller(
    options: &WorkerOptions,
    joined: &Joined,
    shared: Arc<Shared>,
) -> Result<std::thread::JoinHandle<()>> {
    let addr = options.addr;
    let ahead = options.ahead.max(2);
    let job = joined.job.clone();
    let worker = joined.worker.clone();
    let verbose = options.verbose;
    std::thread::Builder::new()
        .name(format!("blockflow-pull-{worker}"))
        .spawn(move || {
            let mut client = Client::new(addr);
            let body = json!({"job": job, "worker": worker});
            while !shared.done.load(Ordering::Acquire) {
                let held = guard(&shared.queue).len();
                if held >= ahead {
                    std::thread::sleep(Duration::from_millis(2));
                    continue;
                }
                match client
                    .post(path::PULL, &body)
                    .map(|value| Handout::from_json(&value))
                {
                    Ok(Ok(Handout::Task(assignment))) => {
                        *guard(&shared.last) = LastReply::Work;
                        guard(&shared.queue).push_back(*assignment);
                        shared.arrived.notify_all();
                    }
                    Ok(Ok(Handout::Wait { after_ms, .. })) => {
                        *guard(&shared.last) = LastReply::Blocked;
                        shared.arrived.notify_all();
                        std::thread::sleep(Duration::from_millis(after_ms.clamp(1, 200)));
                    }
                    Ok(Ok(Handout::Finished)) => {
                        *guard(&shared.last) = LastReply::Finished;
                        shared.done.store(true, Ordering::Release);
                        shared.arrived.notify_all();
                        break;
                    }
                    Ok(Err(error)) | Err(error) => {
                        if verbose {
                            eprintln!("worker {worker}: pulling: {error}");
                        }
                        // A coordinator that has gone away ends the worker
                        // rather than spinning against a closed port; a
                        // transient failure has already been retried once by
                        // the client.
                        *guard(&shared.last) = LastReply::Finished;
                        shared.done.store(true, Ordering::Release);
                        shared.arrived.notify_all();
                        break;
                    }
                }
            }
            shared.arrived.notify_all();
        })
        .map_err(|err| Error::backend(format!("starting the pull thread: {err}")))
}

/// Send events as they happen, one per message.
fn spawn_reporter(
    options: &WorkerOptions,
    joined: &Joined,
    shared: Arc<Shared>,
) -> Result<std::thread::JoinHandle<()>> {
    let addr = options.addr;
    let job = joined.job.clone();
    let worker = joined.worker.clone();
    std::thread::Builder::new()
        .name(format!("blockflow-report-{worker}"))
        .spawn(move || {
            let mut client = Client::new(addr);
            loop {
                let event = {
                    let mut queue = guard(&shared.events);
                    while queue.is_empty() {
                        if shared.quiet.load(Ordering::Acquire) {
                            return;
                        }
                        let (next, _) = shared
                            .events_ready
                            .wait_timeout(queue, Duration::from_millis(20))
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        queue = next;
                    }
                    queue.pop_front()
                };
                let Some(event) = event else { continue };
                // Unbatched, deliberately. A whole run is a few events per
                // second across every worker combined; a batching window would
                // be machinery with nothing to buy and one more thing between
                // an event happening and somebody seeing it.
                let message = json!({
                    "job": job,
                    "worker": worker,
                    "event": event_json(&event),
                });
                if client.post(path::REPORT, &message).is_ok() {
                    shared.sent.fetch_add(1, Ordering::SeqCst);
                }
                // A failed report is dropped, not retried and never allowed to
                // fail the run. Events are observation; the coordinator's task
                // accounting comes from completions, which *are* retried by
                // being reissued.
            }
        })
        .map_err(|err| Error::backend(format!("starting the report thread: {err}")))
}
