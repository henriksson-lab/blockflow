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
// single-node test continued to pass.** So it is counted rather than assumed —
// in two places, because one end cannot see the whole question. `starved` here
// is a wait that ended with a task the coordinator had been holding all along;
// `Job::withheld` there is a pull the coordinator answered "wait" while its
// ready set was not empty. A handout answering only idle workers raises the
// second whatever this worker's timing does, which is why the deterministic
// half of the guard lives at the coordinator.
//
// **And the list is kept ahead by waiting, never by sleeping.** The puller used
// to poll its own queue every two milliseconds for room, which is not a poll
// interval but a *lag*: the list could not be refilled until the sleep ended, so
// any task shorter than it left the list one deeper down than `ahead` asked for,
// and a loaded machine turned that into an empty list. It is a condvar now,
// signalled by the executor on every pop. Measured: nine starved runs in forty
// became none, and putting the sleep back brought them straight back.
//
// Three threads, so that none of them can wait behind another
// -----------------------------------------------------------
// Three *long-lived* ones. `WorkerOptions::threads` above one adds transient
// ones **inside** the executor's task — the slabs of `crate::slab`, scoped to
// one block and joined before it reports — and those do compete with the puller
// and the reporter for cores. That is the trade the option is: a node with cores
// to spare has nothing else to spend them on, because the loop below computes
// one task at a time, and a node without them should leave the default alone.
// * the **executor** — pops a task, runs it through the crate's own
//   `execute_task`, flushes, reports it complete;
// * the **puller** — keeps the work list topped up, independently of whether
//   the executor is busy. This is what "one request deep, pipelined" means;
// * the **reporter** — drains events and sends them, one per message, as they
//   happen. It has its own connection — which is what the design intended by
//   "a slow coordinator delays *events* and never the work", and that sentence
//   was **false as implemented and is inverted here rather than removed**. Its
//   own connection buys nothing if the connection is never read, and measured
//   on this tree one worker's pull connection went unread for a whole job while
//   another worker's event stream was served: `ready_ms: 7`,
//   `first_pull_ms: 270`, with the coordinator reporting its slowest pull at
//   115 microseconds and its longest registry-lock wait at 17. The queue was
//   not in either process. See `distributed::client::Client` for what it was
//   and what now holds instead: a request per connection, so the server's pool
//   always drains and a connection waits one request rather than one job.
//
// The listener the executor sees does one thing: clone the event and push it
// onto a queue. Observation must never be able to slow down the thing it
// observes, and an event sender that blocked would be exactly that.

use std::collections::{BTreeMap, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::json;

use crate::decomposition::{Decomposition, SlabPolicy};
use crate::error::{Error, Result};
use crate::export::event_json;
use crate::fragment::{check_phase_work, PhaseWork};
use crate::graph::TaskGraph;
use crate::listener::EventListener;
use crate::log::Event;
use crate::strategy::{execute_task_with_reduction, reduce_phase};

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
    /// **Die** having computed this many tasks, before reporting the last of
    /// them: `abort(3)`, so the process is taken down by an unhandled signal
    /// with nothing flushed, nothing reported and every claim still held.
    ///
    /// The counterpart of `stop_after`, which is a *clean* exit and therefore
    /// the wrong shape for a death test — a worker that leaves tidily is not
    /// the failure the design is arranged against.
    ///
    /// # Why the worker kills itself rather than being killed
    ///
    /// The runner used to `SIGKILL` a worker once the coordinator reported a
    /// progress threshold, and its own comment called that deterministic
    /// because it was triggered by observed progress rather than by a timer.
    /// **It is not**: progress is observed by *polling* the coordinator over
    /// the same HTTP server the workers are hammering, so a loaded machine can
    /// take the job from "two tasks done" to "finished" between two samples.
    /// Measured on this tree, the runner killed a worker that had already gone
    /// home in about one run in twenty, and both death tests then failed for
    /// having no death in them. Sampling races the job exactly as a timer
    /// would.
    ///
    /// Counting here does not sample anything: this worker knows how many
    /// tasks it has computed. The one thing it cannot guarantee is that it gets
    /// a task at all, and a worker that never ran cannot die holding work — so
    /// the runner records whether a death actually happened and the tests check
    /// that before they check anything else.
    ///
    /// **Between tasks, not inside one**, and the difference is worth stating:
    /// the block whose completion is lost has already been *computed*, so it is
    /// the case that produces a duplicate execution after the reissue rather
    /// than the case that produces a half-written block. Dying inside
    /// `execute_task` would need a hook in the executor for a test's benefit,
    /// which is a seam in production code paid for by nothing else.
    pub abort_after: Option<usize>,
    pub verbose: bool,
    /// How many threads this worker may spend **inside one block**.
    ///
    /// **`1` is the default and is exactly what this worker has always done**:
    /// one task, on one thread, uncut. Above one, the block is offered a cut
    /// into that many slabs — capped by
    /// [`SlabPolicy::CAP`](crate::decomposition::SlabPolicy::CAP), because the
    /// slab halo overtakes the threads past it — and a chain that has not
    /// declared itself sliceable declines the offer and runs uncut.
    ///
    /// **Why this is the regime that wants it.** The loop below computes **one
    /// task at a time**: `ahead` deepens the *claim* pipeline, not the compute.
    /// So a worker on a machine with cores to spare leaves all but one parked,
    /// and unlike the single-node executor there is no block-level parallelism
    /// inside this process to spend them on instead.
    /// `docs/design/intra-block.md` §7 measured that row — one block, threads
    /// swept — at 4.6-5.4x, and §13.3 measured 3.1-3.6x through the executor at
    /// the cap.
    ///
    /// **A worker option rather than something carried in the job**, and the
    /// precedent is beside it: `WorkflowSpec::cache_bytes` says the coordinator
    /// *models* a per-worker budget and that "the worker's own budget is its own
    /// business". How many cores a node has, and how many worker processes are
    /// sharing them, are facts about the node and not about the job — a
    /// deployment running four workers per box wants `cores / 4` here, and
    /// nothing the coordinator knows could tell it that. It is also why no slab
    /// *policy* has to travel: `threads: 1` is the off switch, and it is the
    /// default.
    ///
    /// **It cannot change a barrier phase.** A hoisted reduction is derived from
    /// the fragment set rather than transported, so every worker must compute
    /// byte-identical bytes; slabs never reach a fragment phase, because
    /// `strategy::run_task` dispatches one before the offer exists. Raising this
    /// therefore cannot make two workers disagree.
    pub threads: usize,
}

impl WorkerOptions {
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            job: None,
            name: None,
            ahead: 2,
            stop_after: None,
            abort_after: None,
            verbose: false,
            // One, so that a worker built today runs exactly as it ran before
            // this field existed. See the field.
            threads: 1,
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
    /// Sidecar fragments this worker **read**, in fragments. The counterpart of
    /// `fragments` one line above, which counts the ones it wrote.
    ///
    /// Reported because it is the figure that tells a hoisted merge from a
    /// per-block one, and until now that check was available in-process only.
    /// With the merge in `FragmentOp::reduce` a worker reads the fragment set
    /// once for the phase, so across a job the workers' reads sum to
    /// `O(nodes x blocks)`; with the same merge left in each block's `apply`
    /// every block reads the whole set and they sum to `O(blocks^2)`, for the
    /// same plan and the same answer. See `Stats::sidecar_reads`, which is the
    /// single-node form of the same number.
    pub sidecar_reads: u64,
    /// Applications of a fragment op to a block by this worker, in
    /// **block-applications**: one per block of a fragment phase this worker was
    /// handed, plus one more for each block re-applied to check a
    /// `SeamFold::Unordered` claim.
    ///
    /// Not `tasks`: a task of a chain phase contributes nothing here, and a
    /// block of a fragment phase whose op declares `SeamFold::Unordered` and is
    /// handed more than one fragment contributes two. The distributed form of
    /// `Stats::fragment_applications`, and summing it over the workers of a job
    /// gives what one node would have reported for the same plan.
    pub fragment_applications: u64,
    pub events: usize,
    /// Tasks that started with the next one already in hand. The healthy case.
    pub started_ready: usize,
    /// Tasks that started only after waiting for a handout. One of these is
    /// expected — the first — and a phase boundary can legitimately produce
    /// more.
    pub started_after_waiting: usize,
    /// Waits that ended with a task the coordinator had been holding all
    /// along — the list ran empty while there was work to have. This is the
    /// regression the design warns about, and it is the number to assert zero.
    ///
    /// **Decided when the wait ends, not when it starts.** The obvious test —
    /// "was the puller's last reply `Work`?" — reads a reply that may already
    /// be stale, and the staleness window is exactly one handout round trip:
    /// the puller receives the last task of a phase, the executor drains the
    /// list before the puller's *next* reply lands, and a wait for work that
    /// does not exist is recorded as a wait for work that was withheld. It
    /// mattered: that misclassification is what made this counter, and the
    /// three tests asserting it, fail under load. So the wait is classified by
    /// what ended it — see [`Self::told_to_wait`] for the other outcome.
    ///
    /// # What zero here promises, which is not "never"
    ///
    /// The list is one task deep at `ahead = 2` — one being computed, one in
    /// hand — so it survives a handout round trip only while that round trip is
    /// shorter than a task. That is a **contract**, in the same shape as the
    /// documented `lease > (ahead + 1) x task duration`, and it is stated here
    /// because the depth was not chosen against it: the sweep that settled on 2
    /// ran ~180 ms tasks on an unloaded machine, where `starved` was zero with
    /// room to spare, and the probe fixtures here run ~7 ms tasks on a machine
    /// with forty other things on it. Deeper is not free — the same sweep
    /// measured it monotonically worse for makespan, because depth is claim
    /// hoarding — so this is a trade-off recorded rather than tuned. A residual
    /// of about one run in forty remains at that task size, and it is the
    /// scheduler rather than the handout.
    pub starved: usize,
    /// Waits the coordinator answered, at least once, with "nothing for you
    /// now" before the next task arrived.
    ///
    /// Not a starve and not a fault: a phase boundary, a dependency that has
    /// not landed and the tail of a job all look like this, and in every one of
    /// them there was nothing for this worker to be ahead of. It is counted
    /// rather than dropped because it is the evidence that a zero in
    /// [`Self::starved`] is a classification and not an absence — the two
    /// together account for every wait after the first.
    ///
    /// Whether such a refusal was *legitimate* is a question about the
    /// coordinator, not about this worker, and it is asked there: see
    /// `Job::withheld`, which counts the refusals it issued while it had ready
    /// work in hand.
    pub told_to_wait: usize,
    pub reads: u64,
    pub chunks_read: u64,
    /// Barrier phases this worker reduced. **One per barrier phase, not one per
    /// task** — that is the whole claim `FragmentOp::reduce` makes, and a
    /// distributed run is where it is easiest to get wrong, so the count is
    /// reported rather than assumed.
    pub reductions: usize,
    /// Bytes those reductions produced, summed. What shipping the blob would
    /// have cost this worker if it were shipped, measured against the fragment
    /// set it was derived from instead.
    pub reduced_bytes: u64,
    pub elapsed: Duration,
    /// How long this worker took to be admitted, from the first line of
    /// `run`.
    ///
    /// Reported because "this worker ran nothing" has two very different
    /// causes and no other number tells them apart: it arrived after the work
    /// was gone, or it was here all along and the coordinator gave it nothing.
    /// Both look identical in `tasks`, and the tests whose premise is several
    /// processes fail on the first while reading like the second.
    pub joined: Duration,
    /// How long until this worker first asked for work — admission plus the
    /// plan it has to rebuild before it can accept any: the decomposition it
    /// was given, its own copy of the task graph, and the store.
    pub ready: Duration,
    /// Milliseconds from the first line of `run` until this worker's **first
    /// pull was answered**, and how much of that its connect took.
    ///
    /// The pair that says where a worker that ran nothing spent its life. The
    /// coordinator records what it spent serving each pull; if that is
    /// microseconds and this is hundreds of milliseconds, the wait was not in
    /// either process.
    pub first_pull: Duration,
    /// How long the pull connection took to be accepted. See
    /// [`Self::first_pull`].
    pub pull_connect: Duration,
    /// Times the coordinator answered this worker's pull with "nothing for you
    /// now".
    ///
    /// The third number a worker that ran nothing needs, after `ready` and
    /// `tasks`: a worker refused many times was *answered* and had nothing
    /// coming, and a worker refused zero times with no tasks was never answered
    /// at all. Those are different faults and the counts are the only place the
    /// difference shows.
    pub refused: usize,
    pub listener_faults: usize,
}

/// What the puller last heard. One half of telling a legitimate empty list from
/// a starved one; the other half is [`Shared::refusals`], because this value
/// alone can be a round trip out of date.
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
    /// The executor took one off the list, so there is room for another.
    ///
    /// The puller waits on this rather than sleeping on a timer. A timer here
    /// is not a poll interval, it is a **lag**: the list cannot be refilled
    /// until the sleep ends, so every task shorter than the sleep leaves the
    /// list one deeper down than it should be, and a job whose tasks are
    /// shorter than the sleep runs it empty however deep `ahead` is.
    taken: Condvar,
    last: Mutex<LastReply>,
    /// How many times the coordinator has answered "nothing for you now".
    ///
    /// Monotonic, and read as a *change* rather than as a value: a wait that
    /// spans no increment of this is a wait the coordinator never said it had
    /// nothing for. That is what makes [`WorkerReport::starved`] answerable
    /// after the fact instead of guessed from a reply that may be stale.
    refusals: AtomicUsize,
    /// Microseconds until the puller's first reply, and of that how much its
    /// connect took. See [`WorkerReport::first_pull`].
    first_pull_us: AtomicU64,
    pull_connect_us: AtomicU64,
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
    // Joining is the only thing this connection is for, and a connection held
    // open is a connection the coordinator's server has permanently dispatched
    // a thread to — see `Client` for what that costs when the pool runs out of
    // them. Closed here rather than at the end of the function, which is where
    // it used to close and where it was doing nothing but occupy a slot.
    drop(client);
    let joined_after = started.elapsed();
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
        taken: Condvar::new(),
        last: Mutex::new(LastReply::Nothing),
        refusals: AtomicUsize::new(0),
        first_pull_us: AtomicU64::new(0),
        pull_connect_us: AtomicU64::new(0),
        done: AtomicBool::new(false),
        quiet: AtomicBool::new(false),
        events: Mutex::new(VecDeque::new()),
        events_ready: Condvar::new(),
        sent: AtomicUsize::new(0),
    });

    let ready_after = started.elapsed();
    let puller = spawn_puller(&options, &joined, shared.clone(), started)?;
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
        joined: joined_after,
        ready: ready_after,
        ..Default::default()
    };
    let mut completions = Client::new(options.addr);
    // **A barrier phase's reduction, computed here and never transported.**
    //
    // One blob per phase, held for the life of this job on this worker. The
    // coordinator does not hand out a task of a barrier phase until every
    // earlier phase's tasks have been *reported* complete, and a worker writes
    // its fragments before it reports — so the first task of such a phase to
    // arrive here is proof that the fragment set is whole, and
    // `strategy::reduce_phase` re-verifies that rather than trusting it.
    //
    // **Every worker computes the same bytes and none of them says so to
    // anybody.** The blob is derived from the fragment set, not observed: the
    // set is on storage every worker reads, and `PhaseView` walks the lattice in
    // an order that is a function of the plan. Two workers therefore agree by
    // construction, with no election, no upload, no download and nothing added
    // to a coordinator whose design is that it holds no data. What it costs is
    // one extra read of the fragment set and one extra fold **per worker** —
    // a multiplier set by how many machines a caller has, not by how finely they
    // cut, which is the whole difference from re-deriving it per block.
    //
    // Held on the worker rather than in the op, for `FragmentOp::reduce`'s own
    // reason: an op that cached its answer would answer a second lattice with
    // the first lattice's table. This map dies with the job.
    let mut reduced: BTreeMap<usize, Vec<u8>> = BTreeMap::new();

    // **The slab offer, one block's worth, computed once.**
    //
    // `n_blocks` is `1` and not the plan's block count, and that difference is
    // the whole of why this needed its own wiring rather than the executor's.
    // The rule `floor(workers / n_blocks)` is asking *how many blocks are in
    // flight on this machine*; in `execute_phases` a wave of tasks makes those
    // two the same number, and here they are not — the loop below runs one task
    // at a time however many blocks the plan has. Routing it through
    // `slabs_for` rather than clamping inline keeps the cap in the one place
    // that owns it.
    let slabs = SlabPolicy::FillIdleWorkers.slabs_for(options.threads, 1);

    loop {
        let Some(assignment) = next_task(&shared, &mut report) else {
            break;
        };
        check_agreement(&graph, &assignment, &decomposition)?;
        let task = &graph.tasks[assignment.task];
        if decomposition.phases[task.phase].barrier && !reduced.contains_key(&task.phase) {
            let blob = reduce_phase(&decomposition, task.phase, &work, environment.as_ref())?;
            report.reduced_bytes += blob.len() as u64;
            report.reductions += 1;
            reduced.insert(task.phase, blob);
        }
        let empty: Vec<u8> = Vec::new();
        let blob = reduced.get(&task.phase).unwrap_or(&empty);
        match execute_task_with_reduction(
            &chain,
            &decomposition,
            task,
            &work[task.phase],
            blob,
            environment.as_ref(),
            &listeners,
            slabs,
        ) {
            Ok(outcome) => {
                // Durability before the completion, not a barrier after it: a
                // dependent task is released by the coordinator only once this
                // message arrives, so flushing first is what makes reading the
                // intermediate safe without any worker ever waiting for another.
                environment.finish(task.phase + 1)?;
                // The task's non-pixel output, keyed by the block it came from.
                //
                // Written before the completion for the same reason the image is
                // flushed before it, and **that ordering is now load-bearing
                // rather than merely tidy.** It used to be said here that the
                // barrier was weaker for fragments than for images, because no
                // task ever read a peer's fragment and fragments were consumed
                // by a global merge after the job. A barrier phase reads them:
                // its blocks, and `FragmentOp::reduce` above, read every block's
                // fragment including peers'. So the two are now the same
                // strength — durable before the completion — and it is the
                // coordinator's barrier gate, released by those completions, that
                // turns "written before I reported" into "whole before anyone
                // reduces".
                //
                // `FileSidecars::put` is write-then-rename, so a fragment is
                // either absent or entirely there; there is no prefix for a peer
                // to read.
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
                if options
                    .abort_after
                    .is_some_and(|limit| report.tasks >= limit)
                {
                    // Computed and not reported. No unwinding, no destructors,
                    // no report file: from the coordinator this is exactly a
                    // node that stopped existing. See `WorkerOptions::abort_after`.
                    std::process::abort();
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
                shared.taken.notify_all();
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
    shared.taken.notify_all();
    shared.events_ready.notify_all();
    let _ = puller.join();
    let _ = reporter.join();

    let (reads, _, _, _, chunks_read, _, _) = environment.counters().snapshot();
    // Every fragment this process wrote, however it was produced: the job's own
    // sidecar stream and any fragment op's outputs count the same way, because
    // both go through `Environment::write_sidecar`.
    let (fragments, sidecar_reads, _, _) = environment.counters().sidecar_snapshot();
    report.fragments = fragments as usize;
    report.sidecar_reads = sidecar_reads;
    // Every application of a fragment op this process performed, whichever
    // phase it belonged to. Read from the counter rather than derived from
    // `report.tasks`, because the two differ by exactly the thing worth seeing:
    // a chain task adds a task and no application, and an `Unordered` block
    // adds one task and two applications.
    report.fragment_applications = environment
        .counters()
        .fragment_applications
        .load(Ordering::SeqCst);
    report.reads = reads;
    report.chunks_read = chunks_read;
    report.refused = shared.refusals.load(Ordering::SeqCst);
    report.first_pull = Duration::from_micros(shared.first_pull_us.load(Ordering::SeqCst));
    report.pull_connect = Duration::from_micros(shared.pull_connect_us.load(Ordering::SeqCst));
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
///
/// # Why the empty list is classified when the wait *ends*
///
/// The list being empty is not by itself a fault. At the start there is nothing
/// to be ahead of, at a phase boundary the coordinator has nothing to give, and
/// at the end there is nothing left. Only "the coordinator had work and we still
/// went hungry" is the regression [`WorkerReport::starved`] exists to catch.
///
/// Asking that question on entry cannot answer it. The only evidence to hand is
/// [`Shared::last`], the puller's *previous* reply, and the puller is a round
/// trip ahead of the executor by construction — it receives the last task of a
/// phase, pushes it, and asks again, and the answer to that question has not
/// arrived yet. If the executor drains the list in that window it reads `Work`
/// and records a starve for work that had already run out. Under load the
/// window is a whole round trip wide, which is why this counter, and the three
/// tests asserting it is zero, failed by load rather than by fault.
///
/// So the wait is classified by what ends it. The coordinator gets one more
/// chance to say "nothing for you now" — [`Shared::refusals`] — and if it takes
/// it, this was a wait for work that did not exist. If the wait ends with a task
/// and no refusal in between, the coordinator was holding work the whole time
/// and the list should not have been empty: that is the starve. If it ends with
/// the job finishing, there was nothing to be ahead of.
///
/// This is deliberately *not* a widening. The regression the counter guards —
/// a handout that answers only an idle worker — ends its waits with a task and
/// no refusal, so it still counts, and the case it stops counting is one where
/// the coordinator itself said it had nothing. Whether *that* was legitimate is
/// a question about the coordinator and is asked there, by `Job::withheld`.
fn next_task(shared: &Arc<Shared>, report: &mut WorkerReport) -> Option<Assignment> {
    {
        let mut queue = guard(&shared.queue);
        if let Some(assignment) = queue.pop_front() {
            report.started_ready += 1;
            shared.taken.notify_one();
            return Some(assignment);
        }
    }
    // The first task has nothing to be ahead of, so no wait before one has
    // started is ever a starve.
    let running = report.started_ready + report.started_after_waiting > 0;
    let believed_available = *guard(&shared.last) == LastReply::Work;
    let refusals_before = shared.refusals.load(Ordering::Acquire);
    let mut queue = guard(&shared.queue);
    loop {
        if let Some(assignment) = queue.pop_front() {
            report.started_after_waiting += 1;
            if running {
                // A starve is a wait the coordinator never said it could not
                // fill: it believed work was there when the list emptied, and
                // no refusal arrived before the next task did. Everything else
                // is the coordinator having had nothing to give — whether it
                // had already said so when the list emptied, or said so while
                // we waited. The two together account for every wait past a
                // worker's first, which is what the pipeline test checks.
                if believed_available && shared.refusals.load(Ordering::Acquire) == refusals_before
                {
                    report.starved += 1;
                } else {
                    report.told_to_wait += 1;
                }
            }
            shared.taken.notify_one();
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
    started: Instant,
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
                {
                    // Wait for room, do not sleep towards it. A fixed sleep is
                    // a fixed lag between the list draining and this thread
                    // noticing, so a task shorter than the sleep hands the
                    // executor a list one shallower than `ahead` asked for.
                    let mut queue = guard(&shared.queue);
                    while queue.len() >= ahead && !shared.done.load(Ordering::Acquire) {
                        let (next, _) = shared
                            .taken
                            .wait_timeout(queue, Duration::from_millis(20))
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        queue = next;
                    }
                }
                if shared.done.load(Ordering::Acquire) {
                    break;
                }
                let answer = client.post(path::PULL, &body);
                if shared.first_pull_us.load(Ordering::Acquire) == 0 {
                    shared
                        .first_pull_us
                        .store(started.elapsed().as_micros() as u64, Ordering::Release);
                    shared
                        .pull_connect_us
                        .store(client.connected_in().as_micros() as u64, Ordering::Release);
                }
                match answer.map(|value| Handout::from_json(&value)) {
                    Ok(Ok(Handout::Task(assignment))) => {
                        *guard(&shared.last) = LastReply::Work;
                        guard(&shared.queue).push_back(*assignment);
                        shared.arrived.notify_all();
                    }
                    Ok(Ok(Handout::Wait { after_ms, .. })) => {
                        *guard(&shared.last) = LastReply::Blocked;
                        // Recorded before the notify, so an executor woken by
                        // it already sees the refusal it is about to be
                        // classified by.
                        shared.refusals.fetch_add(1, Ordering::AcqRel);
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
            // The sender's own count, and the merged stream's exactly-once key.
            // Stamped before the post and *not* advanced by a retry — the retry
            // is inside `Client::post` — so a request the coordinator processed
            // and could not answer arrives a second time carrying the number it
            // arrived with the first time, and is dropped there. See
            // `Job::reported`.
            let mut seq: u64 = 0;
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
                seq += 1;
                let message = json!({
                    "job": job,
                    "worker": worker,
                    "seq": seq,
                    "event": event_json(&event),
                });
                if let Ok(answer) = client.post(path::REPORT, &message) {
                    // An older coordinator says nothing, and said nothing means
                    // it took the event — which is what it did.
                    if answer
                        .get("accepted")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(true)
                    {
                        shared.sent.fetch_add(1, Ordering::SeqCst);
                    }
                }
                // A failed report is dropped, not retried and never allowed to
                // fail the run. Events are observation; the coordinator's task
                // accounting comes from completions, which *are* retried by
                // being reissued.
            }
        })
        .map_err(|err| Error::backend(format!("starting the report thread: {err}")))
}
