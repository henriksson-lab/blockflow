// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The coordinator: it holds the decomposition and the task DAG, tracks claims,
// hands out tasks, and receives events. It is **its own program**, always — not
// "rank 0 behaves differently" — so a worker's entire life is *connect to the
// coordinator at X, pull a task, execute, report*, and only how `X` is found
// varies by environment.
//
// Two lifetimes, one implementation
// ---------------------------------
// This type is **job-oriented**, which is what lets one implementation serve
// both deployment shapes without a mode flag anywhere in the logic:
//
// * a **persistent** coordinator accepts jobs over time and outlives them. It
//   is started with no job and never exits by itself.
// * a **per-run** coordinator is *the same program* started with one job and
//   `--exit-when-done`.
//
// The only difference is a boolean read in the serving loop, and it is read
// exactly once per poll: `finished_and_should_exit`. Everything else — the job
// registry, the claim table, the handout, the merged log — is identical. There
// is deliberately no second code path to keep in step, because a per-run
// coordinator that diverged from the persistent one is how the shape somebody
// tests stops being the shape somebody runs.
//
// State that outlives a restart is explicitly **not decided** and not built
// here. Nothing in this file persists, and nothing assumes it will not.
//
// Why the coordinator is cheap
// ----------------------------
// It moves no image data. A whole run is 5 597 handouts and ~101 k events, so
// it is idle between requests and is a fraction of one core out of forty — which
// is what makes running it alongside a worker on the first allocated node a
// non-decision.
//
// What it does *not* do
// ---------------------
// It does not compute, and it does not wait for anybody. In particular the
// cache model it keeps of each worker is derived from assignments it made; see
// `cache_model` for why nothing about cache state may ever introduce a point
// where one side waits for the other.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::decomposition::Decomposition;
use crate::error::{Error, Result};
use crate::export::event_from_json;
use crate::graph::{Task, TaskGraph};
use crate::log::{Event, ExecutionLog};

use super::cache_model::{ChunkGrid, ModelledCache};
use super::handout::{self, HandoutPolicy, WorkerView};
use super::placement::{self, Residency};
use super::protocol::{Assignment, Handout, JobStatus, Joined, PROTOCOL_VERSION};
use super::spec::JobSpec;

/// How long a claim survives without a completion before it is reissued, for a
/// job that asks for a lease at all.
///
/// **Not a default.** [`JobSpec::lease`] is `None` unless somebody sets it, and
/// this constant exists only so that a caller opting in has a figure to start
/// from rather than inventing one. It has to exceed the time a worker may hold
/// a task before starting it — a worker keeps `ahead` tasks in hand, so the
/// real constraint is `lease > (ahead + 1) x task duration`, and a lease that
/// violates it reissues work that nobody lost. That contract being implicit is
/// exactly why no job has a lease unless it asks.
///
/// [`JobSpec::lease`]: super::spec::JobSpec::lease
pub const SUGGESTED_LEASE_MS: u64 = 30_000;
/// How long a per-run coordinator waits, after the last task, for the event
/// stream to go quiet.
///
/// Events are **observation**: they are fire-and-forget, on their own
/// connection, and deliberately not acknowledged, because acknowledging them
/// would put the coordinator on a worker's critical path. The cost of that is
/// that the last few events of a run are still in flight when the last task
/// completes — so a coordinator that exited the instant the job finished would
/// truncate its own record, and the acceptance criterion is asserted from that
/// record.
///
/// Waiting for *quiet* rather than for a fixed delay is what makes this
/// self-tuning: the coordinator leaves as soon as nothing has arrived for this
/// long, which is immediately on a job whose workers have already drained and
/// longer on one whose have not. It can never delay a worker, because no worker
/// waits for the coordinator to exit.
pub const DEFAULT_LINGER_MS: u64 = 250;
/// How long a worker is told to wait when everything ready is claimed.
const WAIT_MS: u64 = 20;
/// How recently a worker must have been heard from to count as a **contender**
/// for a scarce task — see `placement`.
///
/// Not a new synchronisation point and not a scheduling timer: it is the
/// coordinator's answer to "could I hand this to somebody else *instead*", and
/// the only cost of getting it wrong is that a task waits this long for a worker
/// that has died. An idle worker polls every [`WAIT_MS`], so this is twenty-five
/// poll intervals — generous enough that a scheduler hiccup does not disqualify
/// a live worker, short enough that a dead one stops holding a barrier back
/// almost immediately. A worker that is *busy* is excluded regardless of when it
/// was last seen, because it cannot take the task now however well it would suit
/// it.
const CONTENDER_SEEN_WITHIN_MS: u64 = WAIT_MS * 25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskState {
    /// Dependencies unmet, or met and waiting for somebody to ask.
    Pending,
    Claimed,
    Done,
}

struct Claim {
    worker: String,
    at: Instant,
    /// The deadline this claim was handed out under, or `None` for the default
    /// — which is that there is no deadline. Stamped from the spec at handout
    /// rather than read from the spec at expiry, so that a claim made under one
    /// policy is never judged by another.
    lease: Option<Duration>,
}

/// One task a worker was holding, in the terms a person reads.
///
/// Exists so that a lost node can be *named* rather than counted. "Three claims
/// outstanding" tells an operator nothing about which part of the volume has to
/// be redone; `(task 41, phase 0, block [2, 1, 3], held 4.2 s)` tells them
/// where the job stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldClaim {
    pub task: usize,
    pub phase: usize,
    pub index: [usize; 3],
    pub held: Duration,
}

/// Why a job stopped short of finishing.
///
/// A job is either running, finished, or aborted, and the third is a state and
/// not an error return: the coordinator has to keep serving long enough to tell
/// the surviving workers to stop and to write out its record, so "we are giving
/// up, and here is exactly what was lost" has to be something it *holds*.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aborted {
    /// The worker that went away.
    pub worker: String,
    /// How that was found out — an exit status, a launcher's message. The
    /// coordinator does not detect this itself; see [`Job::worker_lost`].
    pub why: String,
    /// Exactly what that worker was holding when it went. Empty is possible and
    /// is worth reporting as itself: a worker that died between tasks lost no
    /// work, and the job still stops, because the decision is about the node
    /// and not about the tasks.
    pub held: Vec<HeldClaim>,
}

impl Aborted {
    /// The message, in one place, so the stderr line and the report agree.
    pub fn message(&self) -> String {
        let held = if self.held.is_empty() {
            "no tasks (it died between claims)".to_string()
        } else {
            self.held
                .iter()
                .map(|claim| {
                    format!(
                        "task {} (phase {}, block {:?}, held {:.1} s)",
                        claim.task,
                        claim.phase,
                        claim.index,
                        claim.held.as_secs_f64()
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "worker {} is gone ({}). It was holding {}. This job is aborting: blockflow \
             does not recover from the loss of a node. A lost node may have held a great \
             deal of work in memory, and re-running only the tasks it had claimed is not \
             the same as recovering it — so the decision is deliberately left to whatever \
             started this job (a batch script, an orchestrator, a person) rather than \
             guessed at here.",
            self.worker, self.why, held
        )
    }
}

/// What the coordinator knows about one worker.
///
/// Every field is derived from messages the coordinator has already handled —
/// assignments it made and completions it was told about. Nothing here is
/// requested, so nothing here can stall a handout.
struct WorkerModel {
    id: String,
    /// The centre of the block this worker last completed, in voxels.
    anchor: Option<[f64; 3]>,
    /// Where it was seeded, before it had completed anything. Kept so that a
    /// third worker is seeded away from where the first two *started*, not only
    /// from where they have got to.
    seed: Option<[f64; 3]>,
    cache: ModelledCache,
    /// Chunks this worker **wrote**, modelled the same way and for the same
    /// reason: it is derived from assignments and never reported.
    ///
    /// A second tier rather than more entries in `cache`, because it is a claim
    /// about a different thing and the two must not be confused. `cache` models
    /// this crate's own chunk cache, which holds what was *read*. `produced`
    /// models whatever on the node holds what was recently *written* — the page
    /// cache under a filesystem store, a local scratch tier, a write-back cache
    /// if one is ever added — none of which this crate owns. It is kept separate
    /// so that the handout's own scoring, and the numbers already measured for
    /// it, are untouched; only `placement` reads it.
    ///
    /// It exists because without it the barrier case is unanswerable. Phase `p`
    /// reads image `p` and writes image `p + 1`, so the image a barrier at phase
    /// `p` reads is one that only ever got written — every worker misses all of
    /// it under a read-only model, no worker is better than any other, and the
    /// placement rule correctly declines to have an opinion. Being wrong about
    /// this costs a re-read, which is what the read would have cost anyway.
    produced: ModelledCache,
    assigned: usize,
    completed: usize,
    last_seen: Instant,
}

impl WorkerModel {
    fn position(&self) -> Option<[f64; 3]> {
        self.anchor.or(self.seed)
    }
}

/// One job: a decomposition, its DAG, and everything that has happened to it.
pub struct Job {
    pub spec: JobSpec,
    pub decomposition: Decomposition,
    graph: TaskGraph,
    chunks: ChunkGrid,
    /// How many tasks each phase has. The *plan's* shape rather than what is
    /// left, and the only thing `placement` treats as scarce — see rule 1 there,
    /// and the measurement that made the distinction necessary.
    phase_tasks: Vec<usize>,
    /// How many of each phase's tasks have reported done. The tally
    /// [`Job::barrier_is_open`] reads, and the only thing enforcing a barrier on
    /// this side.
    phase_done: Vec<usize>,
    state: Vec<TaskState>,
    attempts: Vec<u32>,
    remaining_deps: Vec<usize>,
    dependents: Vec<Vec<usize>>,
    claims: BTreeMap<usize, Claim>,
    workers: BTreeMap<String, WorkerModel>,
    /// Every event every worker reported, in arrival order. This is the merged
    /// stream the acceptance criterion is asserted from, and it is the same
    /// `ExecutionLog` a single-node run produces — which is what lets one check
    /// answer for both.
    log: ExecutionLog,
    /// The highest event sequence number accepted from each worker.
    ///
    /// The merged stream's **exactly-once** key, and it exists because delivery
    /// is at-least-once. `Client::request` retries a request once on a dead
    /// connection, which is right for a connection being reaped and wrong for a
    /// request the coordinator had already processed — the response is what was
    /// lost, and the retry then appends the same event a second time. Measured
    /// on a loaded machine: a sixteen-task job whose stream must hold ninety-six
    /// events held ninety-seven, which broke both the completeness check and the
    /// worker-against-coordinator count that guards it.
    ///
    /// A per-worker counter is enough because a worker posts its events from one
    /// thread over one connection, so they are ordered *within* a worker however
    /// the merge interleaves them. This is also the first half of the causal key
    /// `ExecutionLog::check_coverage_unordered` names as what recovering order
    /// across workers would need; the other half is the happens-before the task
    /// DAG already knows, and nothing here claims to have it.
    reported: BTreeMap<String, u64>,
    /// Events dropped for having been seen before. Not a fault — it is the
    /// retry above doing its job — and reported so that a stream which is short
    /// can be told from one that was deduplicated.
    duplicate_events: usize,
    unknown_events: usize,
    done: usize,
    reissued: usize,
    /// Pulls answered `Wait` **while the ready set was not empty** — a worker
    /// sent away with work in the room.
    ///
    /// Legitimate exactly once: `placement::entitled` withholds a *scarce*
    /// phase's task from an asker with a better-placed idle peer, and a barrier
    /// phase is scarce by construction. Outside that, this is the regression
    /// `DISTRIBUTION.md` names — a handout that answers only an idle worker
    /// would raise it on every ordinary phase's tail, and it is exactly the
    /// shape the momentary-ready-set version of `is_scarce` had.
    ///
    /// Counted here rather than inferred at the worker on purpose. A worker can
    /// see only that its list ran empty; whether the coordinator had anything to
    /// give is the coordinator's own state, and reading it here is a fact where
    /// reading it there was a race. See `WorkerReport::starved` for the half
    /// this does not answer: a list that ran empty with nobody refusing it.
    withheld: usize,
    failed: usize,
    started: Instant,
    last_event: Instant,
    /// When a task was last reported complete. See [`Job::quiet_for`].
    last_completion: Instant,
    /// Whether a scarce task may be withheld from a worker with a better-placed
    /// idle peer. On, because a barrier phase is the only placement decision in
    /// its phase; off is here so the two can be measured against each other over
    /// the same job, which is the only way "it helps" means anything.
    scarce_placement: bool,
    /// Set once, by [`Job::worker_lost`], and never cleared. See [`Aborted`].
    aborted: Option<Aborted>,
    /// Pulls this job answered, per worker.
    ///
    /// Diagnostic, and it exists to answer one question no other number can. A
    /// worker that ran nothing either never asked, or asked and was told there
    /// was nothing, or asked and was **never answered**. Its own report
    /// separates the first two — `ready`, `refused` — and only this separates
    /// the third, because a request that is never served leaves no trace at the
    /// end that sent it.
    pulls: BTreeMap<String, u64>,
}

impl Job {
    pub fn new(spec: JobSpec, decomposition: Decomposition) -> Result<Self> {
        decomposition.check()?;
        let graph = TaskGraph::build(&decomposition);
        graph
            .dependencies_cover_reads(&decomposition)
            .map_err(Error::InvalidArgument)?;
        let remaining_deps: Vec<usize> = graph.tasks.iter().map(Task::n_dependencies).collect();
        let dependents = graph.dependents();
        let n = graph.len();
        let chunks = ChunkGrid::new(decomposition.volume, spec.workflow.chunk);
        let mut phase_tasks = vec![0usize; decomposition.n_phases()];
        for task in &graph.tasks {
            if let Some(count) = phase_tasks.get_mut(task.phase) {
                *count += 1;
            }
        }
        Ok(Self {
            spec,
            decomposition,
            graph,
            chunks,
            phase_done: vec![0; phase_tasks.len()],
            phase_tasks,
            state: vec![TaskState::Pending; n],
            attempts: vec![0; n],
            remaining_deps,
            dependents,
            claims: BTreeMap::new(),
            workers: BTreeMap::new(),
            log: ExecutionLog::new(),
            reported: BTreeMap::new(),
            duplicate_events: 0,
            unknown_events: 0,
            done: 0,
            reissued: 0,
            withheld: 0,
            failed: 0,
            started: Instant::now(),
            last_event: Instant::now(),
            last_completion: Instant::now(),
            scarce_placement: true,
            aborted: None,
            pulls: BTreeMap::new(),
        })
    }

    /// Turn locality-aware placement of scarce tasks off — the baseline the
    /// barrier measurement is taken against. See `placement`.
    pub fn with_scarce_placement(mut self, on: bool) -> Self {
        self.scarce_placement = on;
        self
    }

    /// Chunks of `task`'s read set the coordinator models `worker` as already
    /// holding, out of how many the read touches.
    ///
    /// Diagnostic, and what the barrier measurement is taken from. The model is
    /// never authoritative about anything, this included.
    pub fn modelled_overlap(&self, worker: &str, task: usize) -> Option<(usize, usize)> {
        let model = self.workers.get(worker)?;
        let keys = placement::read_keys(&self.graph, &self.chunks, task);
        Some((self.residency(model).overlap(&keys), keys.len()))
    }

    pub fn graph(&self) -> &TaskGraph {
        &self.graph
    }

    pub fn log(&self) -> &ExecutionLog {
        &self.log
    }

    pub fn finished(&self) -> bool {
        self.done == self.graph.len()
    }

    /// Why this job gave up, if it did.
    pub fn aborted(&self) -> Option<&Aborted> {
        self.aborted.as_ref()
    }

    /// Finished or aborted — the two ways a job stops needing workers.
    ///
    /// Deliberately one predicate rather than two tested at every call site: a
    /// coordinator that exits on one and hangs on the other is precisely the
    /// silent stall this design rules out.
    pub fn over(&self) -> bool {
        self.finished() || self.aborted.is_some()
    }

    /// What `worker` is holding right now.
    ///
    /// The coordinator has always known this; it had nowhere to say it. Reads
    /// the claim table directly rather than the cache model, because this is
    /// about work that will not be finished and not about bytes somebody might
    /// have.
    pub fn claims_held_by(&self, worker: &str) -> Vec<HeldClaim> {
        let now = Instant::now();
        self.claims
            .iter()
            .filter(|(_, claim)| claim.worker == worker)
            .map(|(&task, claim)| HeldClaim {
                task,
                phase: self.graph.tasks[task].phase,
                index: self.graph.tasks[task].index,
                held: now.duration_since(claim.at),
            })
            .collect()
    }

    /// A worker is gone. Stop the job and say what was lost.
    ///
    /// **This is the answer to a node loss, and reissue is not.** The two are
    /// different beliefs about what a lost node costs: reissue believes the
    /// tasks it held are the work it had, so re-running them restores the
    /// position. That is not this deployment. A node here may hold a great deal
    /// in memory and in its own cache, so the honest choice is between building
    /// real recovery — a large piece of design, deliberately not attempted now
    /// — and stopping cleanly for somebody above to decide. This is the second.
    ///
    /// The coordinator **does not detect this itself**, and the parameter is
    /// the evidence rather than a guess: it has no signal for a process it did
    /// not start, and the only thing it could use instead is silence, which is
    /// a timeout, which is the thing this design removed. Whoever started the
    /// worker — `local::run`, `srun`, an orchestrator — sees the exit and says
    /// so. See the module header.
    ///
    /// Idempotent, and it has to be: two survivors can notice the same death,
    /// and the first account is the one worth keeping, because by the second
    /// the claim table has already been read.
    pub fn worker_lost(&mut self, worker: &str, why: &str) -> Aborted {
        if let Some(already) = &self.aborted {
            return already.clone();
        }
        let aborted = Aborted {
            worker: worker.to_string(),
            why: why.to_string(),
            held: self.claims_held_by(worker),
        };
        eprintln!("coordinator: {}", aborted.message());
        self.aborted = Some(aborted.clone());
        aborted
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    pub fn reissued(&self) -> usize {
        self.reissued
    }

    /// See [`Job::withheld`].
    pub fn withheld(&self) -> usize {
        self.withheld
    }

    /// See [`Job::pulls`].
    pub fn pulls_per_worker(&self) -> BTreeMap<String, u64> {
        self.pulls.clone()
    }

    /// Chunk keys the coordinator *modelled* each worker as holding. Diagnostic
    /// only; the model is never authoritative about anything.
    pub fn modelled_cache_size(&self, worker: &str) -> Option<usize> {
        self.workers.get(worker).map(|model| model.cache.len())
    }

    pub fn status(&self) -> JobStatus {
        JobStatus {
            job: self.spec.id.clone(),
            tasks: self.graph.len(),
            done: self.done,
            claimed: self.claims.len(),
            pending: self.graph.len() - self.done - self.claims.len(),
            workers: self.workers.len(),
            reissued: self.reissued,
            failed: self.failed,
            events: self.log.len(),
            finished: self.finished(),
        }
    }

    /// Register a worker, or find it again after a reconnect.
    fn admit(&mut self, worker: &str) {
        let budget = self.spec.workflow.cache_bytes;
        let chunk_bytes = self.chunks.chunk().iter().product::<usize>() as u64
            * self.decomposition.dtype.size_of() as u64;
        self.workers
            .entry(worker.to_string())
            .or_insert_with(|| WorkerModel {
                id: worker.to_string(),
                anchor: None,
                seed: None,
                cache: ModelledCache::new(budget, chunk_bytes),
                // The same budget, because it is the only figure there is: the
                // node's write-side residency is not this crate's to size, and
                // an invented second number would be worse than a reused one.
                // Over-estimating it costs a re-read; under-estimating it costs
                // the placement rule an opinion it could have had.
                produced: ModelledCache::new(budget, chunk_bytes),
                assigned: 0,
                completed: 0,
                last_seen: Instant::now(),
            })
            .last_seen = Instant::now();
    }

    /// Return every claim whose lease has expired to the pending set.
    ///
    /// **Off unless a job asks for it.** A claim with no lease — the default,
    /// see [`JobSpec::lease`] and the module header — can never appear in
    /// `expired`, so this is a scan of a table whose every entry declines, and
    /// the reissue path below is never entered. That is deliberate rather than
    /// vestigial: the mechanism stays compiled and tested, and a job that wants
    /// it sets a lease.
    ///
    /// What it does when a job does ask: a worker that is preempted, reclaimed,
    /// killed or simply wedged stops renewing, its claim expires, and the task
    /// goes to somebody else. No failure detector, no membership protocol, no
    /// second mechanism for the "worker is slow" case versus the "worker is
    /// gone" case — a slow worker's task is reissued and its eventual
    /// completion is accepted or ignored, which costs a duplicate execution and
    /// never a wrong result, because the two executions write the same values
    /// to the same valid region.
    ///
    /// The reason that is not the default is the reason it is worth reading
    /// twice: **a slow worker is indistinguishable from a dead one here**, and
    /// the coordinator issues a claim's deadline at handout while the worker
    /// keeps `ahead` tasks in hand, so a lease shorter than `(ahead + 1) x`
    /// task duration reissues work that nobody lost. Measured on the local
    /// multi-node fixture with **nobody killed**, a 400 ms lease reissued 13 of
    /// 16 tasks and recomputed 11 of 16 blocks — 69 % of a job duplicated with
    /// no fault at all. The output stayed byte-identical, so it was waste and
    /// not corruption, but it is waste bought with a number that has to be
    /// guessed right.
    ///
    /// [`JobSpec::lease`]: super::spec::JobSpec::lease
    fn expire_claims(&mut self) {
        let now = Instant::now();
        let expired: Vec<(usize, String)> = self
            .claims
            .iter()
            .filter(|(_, claim)| {
                claim
                    .lease
                    .is_some_and(|lease| now.duration_since(claim.at) > lease)
            })
            .map(|(&task, claim)| (task, claim.worker.clone()))
            .collect();
        for (task, worker) in expired {
            self.claims.remove(&task);
            if self.state[task] == TaskState::Claimed {
                self.state[task] = TaskState::Pending;
                self.reissued += 1;
                eprintln!(
                    "coordinator: worker {worker} did not report task {task} within its \
                     lease; reissuing it"
                );
            }
        }
    }

    /// Pending tasks whose dependencies are all done — **and**, for a barrier
    /// phase, whose earlier phases are all finished.
    ///
    /// # The second half is not optional and is not derivable from the first
    ///
    /// [`TaskGraph::barriers`] is a phase-level fact that the edges deliberately
    /// do not spell: a barrier phase's blocks fetch only their own cores, so
    /// their `remaining_deps` reach zero long before the phase below has
    /// finished. A coordinator that handed such a task out on the first half
    /// alone would have a worker compute from an incomplete fragment set and
    /// report a plausible wrong answer — and, being a distributed run, report it
    /// differently on different machines.
    ///
    /// It is the same gate `strategy::execute_phases` applies in-process and it
    /// is stated the same way: *every earlier phase's tasks are done*. Earlier
    /// phases rather than only `p-1`, because a `FragmentInput` may name a
    /// stream written further back.
    ///
    /// **It cannot deadlock**, on `strategy.rs`'s own argument: a task of phase
    /// `p` waits only on earlier phases, so holding phase `p` back blocks
    /// nothing phase `p` needs, and phase 0 is ready from the start.
    ///
    /// **Reissue is safe.** `phase_done` is bumped exactly where `done` is —
    /// behind the same `TaskState::Done` early return — so a duplicate
    /// completion counts once, and a task returned to `Pending` by a lost worker
    /// or an expired lease was never counted.
    ///
    /// A hoisted `FragmentOp::reduce` is a different matter and is refused
    /// rather than gated: see `strategy::execute_task_of`, which is what a
    /// worker runs a block through.
    fn ready(&self) -> Vec<usize> {
        (0..self.graph.len())
            .filter(|&task| {
                self.state[task] == TaskState::Pending
                    && self.remaining_deps[task] == 0
                    && self.barrier_is_open(self.graph.tasks[task].phase)
            })
            .collect()
    }

    /// Whether a task of `phase` may start: `true` unless the phase declares a
    /// barrier and some earlier phase still has work outstanding.
    fn barrier_is_open(&self, phase: usize) -> bool {
        if !self.graph.is_barrier(phase) {
            return true;
        }
        (0..phase).all(|earlier| self.phase_done[earlier] >= self.phase_tasks[earlier])
    }

    fn residency<'a>(&self, model: &'a WorkerModel) -> Residency<'a> {
        Residency {
            id: &model.id,
            resident: Some(&model.cache),
            produced: Some(&model.produced),
        }
    }

    /// The workers this task could be handed to **instead** of the one asking.
    ///
    /// Derived entirely from state the coordinator already has, so building this
    /// asks nobody anything and waits for nothing. Three filters, each of which
    /// is about whether a handout to that worker could happen *now*:
    ///
    /// * not the asker;
    /// * **idle** — a worker holding a claim cannot take this task now, however
    ///   well its cache would suit it, and counting it would let a busy worker
    ///   hold a barrier away from an idle one indefinitely;
    /// * **recently seen** — see [`CONTENDER_SEEN_WITHIN_MS`]. This is what
    ///   bounds the deferral: a worker that has died stops being a contender, so
    ///   the task stops being withheld, with no timer of its own.
    ///
    /// Empty under [`HandoutPolicy::Naive`], which is defined as ignoring who is
    /// asking; declining a worker on the strength of what it holds is the
    /// strongest form of not ignoring it. That keeps the baseline the other
    /// policies are measured against exactly what it was.
    fn contenders(&self, worker: &str) -> Vec<Residency<'_>> {
        if self.spec.policy == HandoutPolicy::Naive || !self.scarce_placement {
            return Vec::new();
        }
        let busy: std::collections::BTreeSet<&str> = self
            .claims
            .values()
            .map(|claim| claim.worker.as_str())
            .collect();
        let fresh = Duration::from_millis(CONTENDER_SEEN_WITHIN_MS);
        self.workers
            .values()
            .filter(|model| model.id != worker)
            .filter(|model| !busy.contains(model.id.as_str()))
            .filter(|model| model.last_seen.elapsed() <= fresh)
            .map(|model| self.residency(model))
            .collect()
    }

    /// Hand one task to one worker.
    ///
    /// One block per handout: at the planned block sizes a whole run is under
    /// one request per second across the whole cluster, so there is nothing to
    /// amortise and a batch would only add failure modes.
    pub fn pull(&mut self, worker: &str) -> Handout {
        *self.pulls.entry(worker.to_string()).or_insert(0) += 1;
        self.admit(worker);
        self.expire_claims();
        // A survivor asking for work after the job has been abandoned is told
        // to go home, in the same word it would hear on a clean finish. It is
        // the same thing from the worker's side — there is no more work for it
        // — and inventing a third answer would give every worker a new state to
        // get wrong. The *record* keeps the difference: `aborted` is set, the
        // report says so, and the coordinator exits non-zero.
        if self.over() {
            return Handout::Finished;
        }
        let ready = self.ready();
        if ready.is_empty() {
            // Nothing to give, so nothing was withheld: this is a worker
            // waiting on the plan, not on the handout.
            return Handout::Wait {
                after_ms: WAIT_MS,
                remaining: self.graph.len() - self.done,
            };
        }
        let seeds: Vec<[f64; 3]> = self
            .workers
            .values()
            .filter(|model| model.id != worker)
            .filter_map(|model| model.position())
            .collect();
        // The placement pipeline: `ready -> entitled -> choose`. Each stage
        // narrows the same set of task ids, which is the seam a **capability**
        // filter — which classes of node may run this op at all — would be added
        // to, in front of the locality score rather than tangled into it. See
        // `placement` for what that would need.
        let contenders = self.contenders(worker);
        let entitled = {
            let model = self.workers.get(worker).expect("admitted above");
            placement::entitled(
                &ready,
                &self.graph,
                &self.chunks,
                &self.phase_tasks,
                &self.residency(model),
                &contenders,
            )
        };
        let view = {
            let model = self.workers.get(worker).expect("admitted above");
            WorkerView {
                anchor: model.anchor,
                cache: Some(&model.cache),
            }
        };
        let Some(task) = handout::choose(
            self.spec.policy,
            &entitled,
            &self.graph,
            &self.chunks,
            &view,
            &seeds,
        ) else {
            // Everything ready has a better owner that is idle and asking. It
            // is the same answer to this worker as an empty ready set, and it
            // is answered immediately — a refused pull never blocks — but it is
            // not the same event, so it is counted apart. See `Job::withheld`.
            self.withheld += 1;
            return Handout::Wait {
                after_ms: WAIT_MS,
                remaining: self.graph.len() - self.done,
            };
        };

        let entry = &self.graph.tasks[task];
        let phase = entry.phase;
        let read = entry.geometry.read.clone();
        let assignment = Assignment {
            job: self.spec.id.clone(),
            task,
            phase,
            index: entry.index,
            core: entry.geometry.core.clone(),
            read: read.clone(),
            valid: entry.geometry.valid.clone(),
            attempt: self.attempts[task] + 1,
            lease: self.spec.lease,
        };
        let placed = handout::position(&self.graph, task);
        self.attempts[task] += 1;
        self.state[task] = TaskState::Claimed;
        self.claims.insert(
            task,
            Claim {
                worker: worker.to_string(),
                at: Instant::now(),
                lease: self.spec.lease,
            },
        );
        // The cache model is updated **here**, from the assignment, and never
        // from anything a worker says. See `cache_model`.
        let keys = self.chunks.keys(phase, &read);
        // What this task will write, on the same terms. Phase `p` writes image
        // `p + 1`, over the block's *valid* region — the part it owns, which is
        // what tiles the image exactly. Recorded at assignment rather than at
        // completion for the same reason the read is: the coordinator is
        // modelling what it caused, and a task that never finishes leaves the
        // model claiming residency the node does not have, which costs a
        // re-read.
        let written = self.chunks.keys(phase + 1, &assignment.valid);
        if let Some(model) = self.workers.get_mut(worker) {
            model.assigned += 1;
            model.cache.note_assigned(&keys);
            model.produced.note_assigned(&written);
            if model.seed.is_none() {
                model.seed = Some(placed);
            }
        }
        Handout::Task(Box::new(assignment))
    }

    /// A worker says it finished a task.
    ///
    /// Idempotent, and it has to be: a reissued task can be completed twice, by
    /// the worker that was thought dead and by the one that took over. The
    /// second report is accepted and ignored rather than refused, because both
    /// executions wrote the same values to the same valid region and there is
    /// nothing to reconcile.
    pub fn completed(&mut self, worker: &str, task: usize) -> Result<()> {
        if task >= self.graph.len() {
            return Err(Error::invalid(format!(
                "completion names task {task} of a job with {} tasks",
                self.graph.len()
            )));
        }
        self.admit(worker);
        self.last_completion = Instant::now();
        if let Some(model) = self.workers.get_mut(worker) {
            model.completed += 1;
        }
        let anchor = handout::position(&self.graph, task);
        if let Some(model) = self.workers.get_mut(worker) {
            model.anchor = Some(anchor);
        }
        self.claims.remove(&task);
        if self.state[task] == TaskState::Done {
            return Ok(());
        }
        self.state[task] = TaskState::Done;
        self.done += 1;
        // Behind the same early return as `done`, so a duplicate completion
        // counts once. `barrier_is_open` reads it.
        if let Some(count) = self.phase_done.get_mut(self.graph.tasks[task].phase) {
            *count += 1;
        }
        for &next in &self.dependents[task] {
            self.remaining_deps[next] = self.remaining_deps[next].saturating_sub(1);
        }
        Ok(())
    }

    /// A worker says it could not finish a task.
    ///
    /// Returned to the pending set immediately rather than waiting for the
    /// lease: a worker that reports a failure has told us what a timeout would
    /// have had to guess, and there is no reason to sit on the task for another
    /// thirty seconds.
    pub fn failed(&mut self, worker: &str, task: usize, why: &str) {
        self.admit(worker);
        self.failed += 1;
        if task < self.graph.len() && self.state[task] == TaskState::Claimed {
            self.state[task] = TaskState::Pending;
            self.claims.remove(&task);
            self.reissued += 1;
        }
        eprintln!(
            "coordinator: worker {worker} failed task {task} of job {}: {why}",
            self.spec.id
        );
    }

    /// One event, as it happened.
    ///
    /// An unfamiliar event type is counted and dropped, not refused: a worker
    /// from a newer build reporting something this coordinator has never heard
    /// of must not fail the run, for the same reason a listener that panics is
    /// isolated rather than propagated.
    /// Append one worker's event to the merged stream, once.
    ///
    /// `seq` is the sender's own count of the events it has posted, starting at
    /// one. `None` means the sender does not number them, in which case this
    /// appends whatever it is given — the behaviour before numbering existed,
    /// and the safe direction for a worker built against an older coordinator.
    ///
    /// Returns whether the event was accepted, so the sender can keep a count
    /// that matches this one. See [`Job::reported`].
    pub fn report(&mut self, worker: &str, seq: Option<u64>, event: &Value) -> bool {
        self.last_event = Instant::now();
        if let Some(seq) = seq {
            let last = self.reported.entry(worker.to_string()).or_insert(0);
            if seq <= *last {
                self.duplicate_events += 1;
                return false;
            }
            *last = seq;
        }
        let at = self.log.len();
        match event_from_json(event, at) {
            Ok(Some(event)) => self.log.push(event),
            Ok(None) => self.unknown_events += 1,
            Err(_) => self.unknown_events += 1,
        }
        true
    }

    pub fn unknown_events(&self) -> usize {
        self.unknown_events
    }

    /// See [`Job::duplicate_events`].
    pub fn duplicate_events(&self) -> usize {
        self.duplicate_events
    }

    /// How long since anything at all was reported — an event **or** a
    /// completion.
    ///
    /// Both, because `linger` is there to let the tail of the event stream
    /// arrive after the last task finishes, and measured from events alone it
    /// gives a job that has sent *no* events no grace at all: `last_event` is
    /// then the moment the job was created, the coordinator reads the stream as
    /// having been quiet since before it started, and it leaves the instant the
    /// work does — taking the whole record with it. Seen once here, on a machine
    /// loaded enough for a reporter's first request to hit the client's own
    /// timeout. It is not a fix for that stall, which is longer than any linger
    /// worth having; it is the difference between a job that is quiet because it
    /// has nothing more to say and one that has not managed to say anything yet.
    pub fn quiet_for(&self) -> Duration {
        self.last_event.max(self.last_completion).elapsed()
    }

    /// Which tasks were handed out more than once, and how often.
    ///
    /// The number that distinguishes "a block ran twice because a worker died"
    /// from "a block ran twice because the coordinator lost track of it". The
    /// first is expected; the second would be a bug, and without this the two
    /// look identical in the merged log.
    pub fn reissued_tasks(&self) -> Vec<(usize, u32)> {
        self.attempts
            .iter()
            .enumerate()
            .filter(|(_, &attempts)| attempts > 1)
            .map(|(task, &attempts)| (task, attempts))
            .collect()
    }

    /// The acceptance criterion, over the merged stream: every block affected by
    /// every op, once each.
    ///
    /// **In any order, and that is not a weakening.** It asked for chain order
    /// until an intermittent failure showed the question was unaskable here:
    /// every worker posts its events from its own reporter thread and this
    /// coordinator appends them as they arrive, so two workers running two phases
    /// of one block can land their `OpApplied` events in either order however
    /// strictly the work was ordered. The work's order is this state machine's,
    /// not the log's — `remaining_deps` and `barrier_is_open` are what order it,
    /// and `dependencies_cover_reads` is what checks the plan they come from.
    /// See `ExecutionLog::check_coverage_unordered` for what is still caught and
    /// what recovering the order would actually need.
    pub fn check_coverage(&self) -> Result<()> {
        let expected: Vec<(usize, String)> = self
            .decomposition
            .op_names_in_order()
            .into_iter()
            .enumerate()
            .collect();
        let blocks = self.decomposition.phases[0].blocks.len();
        self.log.check_coverage_unordered(&expected, blocks)
    }
}

/// The registry, the lock, and the two lifetimes.
pub struct Coordinator {
    jobs: Mutex<BTreeMap<String, Job>>,
    /// Job ids that were submitted, in order, so a worker that names no job
    /// gets the one there is.
    order: Mutex<Vec<String>>,
    next_worker: AtomicU64,
    /// The **only** difference between a persistent coordinator and a per-run
    /// one. Read by the serving loop; nothing in the job logic sees it.
    exit_when_done: bool,
    linger: Duration,
    serving: Serving,
}

/// What the request threads spent their time on.
///
/// Not a performance counter for its own sake. The coordinator holds **one**
/// mutex over the whole registry and every request takes it — a pull, a
/// completion and every single event report alike — so "observation cannot slow
/// the work down" is a claim about that lock and nothing else. These are the
/// numbers that check it: how long the longest acquisition waited, how many
/// waited more than a millisecond, and how long the slowest pull and the slowest
/// report spent inside this process.
#[derive(Debug, Default)]
pub struct Serving {
    pub pulls: AtomicU64,
    pub reports: AtomicU64,
    pub other: AtomicU64,
    /// Microseconds the slowest pull spent between arriving here and being
    /// answered. If a worker waited a whole job for a handout and this is
    /// small, the wait was **not** in this process.
    pub slowest_pull_us: AtomicU64,
    pub slowest_report_us: AtomicU64,
    /// Microseconds the longest wait for the registry mutex took.
    pub slowest_lock_us: AtomicU64,
    /// Acquisitions of the registry mutex that waited longer than a
    /// millisecond.
    pub slow_locks: AtomicU64,
}

impl Serving {
    fn to_json(&self) -> Value {
        json!({
            "pulls": self.pulls.load(Ordering::Relaxed),
            "reports": self.reports.load(Ordering::Relaxed),
            "other": self.other.load(Ordering::Relaxed),
            "slowest_pull_us": self.slowest_pull_us.load(Ordering::Relaxed),
            "slowest_report_us": self.slowest_report_us.load(Ordering::Relaxed),
            "slowest_lock_us": self.slowest_lock_us.load(Ordering::Relaxed),
            "slow_locks": self.slow_locks.load(Ordering::Relaxed),
        })
    }
}

impl Coordinator {
    pub fn new(exit_when_done: bool) -> Self {
        Self {
            jobs: Mutex::new(BTreeMap::new()),
            order: Mutex::new(Vec::new()),
            next_worker: AtomicU64::new(1),
            exit_when_done,
            serving: Serving::default(),
            linger: Duration::from_millis(DEFAULT_LINGER_MS),
        }
    }

    /// How long to wait for the event stream to go quiet before exiting. See
    /// [`DEFAULT_LINGER_MS`]; only a per-run coordinator ever uses it.
    pub fn with_linger(mut self, linger: Duration) -> Self {
        self.linger = linger;
        self
    }

    pub fn exit_when_done(&self) -> bool {
        self.exit_when_done
    }

    pub fn submit(&self, spec: JobSpec, decomposition: Decomposition) -> Result<String> {
        let id = spec.id.clone();
        let job = Job::new(spec, decomposition)?;
        let mut jobs = self.lock_jobs();
        if jobs.contains_key(&id) {
            return Err(Error::invalid(format!(
                "a job called {id:?} already exists"
            )));
        }
        jobs.insert(id.clone(), job);
        drop(jobs);
        self.order
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(id.clone());
        Ok(id)
    }

    /// The registry lock, timed.
    ///
    /// Every request in this process passes through here, so this is the one
    /// place a queue can form between a worker asking and a worker being
    /// answered. See [`Serving`].
    fn lock_jobs(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, Job>> {
        let waiting_since = Instant::now();
        let guard = self
            .jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let waited = waiting_since.elapsed().as_micros() as u64;
        self.serving
            .slowest_lock_us
            .fetch_max(waited, Ordering::Relaxed);
        if waited > 1_000 {
            self.serving.slow_locks.fetch_add(1, Ordering::Relaxed);
        }
        guard
    }

    /// Record what one request cost, by route. Called by the server.
    pub fn served(&self, route: &str, took: Duration) {
        let micros = took.as_micros() as u64;
        match route {
            super::protocol::path::PULL => {
                self.serving.pulls.fetch_add(1, Ordering::Relaxed);
                self.serving
                    .slowest_pull_us
                    .fetch_max(micros, Ordering::Relaxed);
            }
            super::protocol::path::REPORT => {
                self.serving.reports.fetch_add(1, Ordering::Relaxed);
                self.serving
                    .slowest_report_us
                    .fetch_max(micros, Ordering::Relaxed);
            }
            _ => {
                self.serving.other.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// See [`Serving`].
    pub fn serving_json(&self) -> Value {
        self.serving.to_json()
    }

    pub fn job_ids(&self) -> Vec<String> {
        self.order
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// The job a request means. A named one, or — for the per-run shape, where
    /// there is exactly one — the only one there is.
    fn resolve(&self, named: Option<&str>) -> Result<String> {
        if let Some(name) = named {
            if !name.is_empty() {
                return Ok(name.to_string());
            }
        }
        let ids = self.job_ids();
        match ids.len() {
            0 => Err(Error::invalid(
                "this coordinator has no jobs yet. A per-run coordinator is started with \
                 one (`--job SPEC`); a persistent one is sent them (`POST /job`)."
                    .to_string(),
            )),
            1 => Ok(ids[0].clone()),
            _ => Err(Error::invalid(format!(
                "this coordinator is running {} jobs, so a request must name one: {}",
                ids.len(),
                ids.join(", ")
            ))),
        }
    }

    pub fn join(&self, named: Option<&str>, hint: Option<&str>) -> Result<Joined> {
        let id = self.resolve(named)?;
        let worker = match hint {
            Some(name) if !name.is_empty() => name.to_string(),
            _ => format!("worker-{}", self.next_worker.fetch_add(1, Ordering::SeqCst)),
        };
        let mut jobs = self.lock_jobs();
        let job = jobs
            .get_mut(&id)
            .ok_or_else(|| Error::invalid(format!("no job called {id:?}")))?;
        job.admit(&worker);
        Ok(Joined {
            protocol: PROTOCOL_VERSION,
            job: id,
            worker,
            spec: job.spec.to_json(&job.decomposition)?,
        })
    }

    pub fn pull(&self, job: &str, worker: &str) -> Result<Handout> {
        self.with_job(job, |job| Ok(job.pull(worker)))
    }

    pub fn completed(&self, job: &str, worker: &str, task: usize) -> Result<JobStatus> {
        self.with_job(job, |job| {
            job.completed(worker, task)?;
            Ok(job.status())
        })
    }

    pub fn failed(&self, job: &str, worker: &str, task: usize, why: &str) -> Result<()> {
        self.with_job(job, |job| {
            job.failed(worker, task, why);
            Ok(())
        })
    }

    /// Tell the coordinator a worker is gone. See [`Job::worker_lost`].
    pub fn worker_lost(&self, named: Option<&str>, worker: &str, why: &str) -> Result<Aborted> {
        let id = self.resolve(named)?;
        self.with_job(&id, |job| Ok(job.worker_lost(worker, why)))
    }

    /// Whether any job gave up, and the first account of why.
    pub fn aborted(&self) -> Option<Aborted> {
        self.lock_jobs()
            .values()
            .find_map(|job| job.aborted().cloned())
    }

    pub fn report(&self, job: &str, worker: &str, seq: Option<u64>, event: &Value) -> Result<bool> {
        self.with_job(job, |job| Ok(job.report(worker, seq, event)))
    }

    pub fn status(&self, named: Option<&str>) -> Result<JobStatus> {
        let id = self.resolve(named)?;
        self.with_job(&id, |job| Ok(job.status()))
    }

    /// Run something against a job while holding the registry lock.
    ///
    /// One lock over the whole registry, deliberately. Everything it protects is
    /// a few pointer-chases — a claim table update and a scan of the ready set —
    /// against a request rate under one per second per cluster, so a finer lock
    /// would be complexity bought with no measurement behind it.
    pub fn with_job<T>(&self, id: &str, act: impl FnOnce(&mut Job) -> Result<T>) -> Result<T> {
        let mut jobs = self.lock_jobs();
        let job = jobs
            .get_mut(id)
            .ok_or_else(|| Error::invalid(format!("no job called {id:?}")))?;
        act(job)
    }

    pub fn inspect<T>(&self, id: &str, look: impl FnOnce(&Job) -> T) -> Result<T> {
        let jobs = self.lock_jobs();
        let job = jobs
            .get(id)
            .ok_or_else(|| Error::invalid(format!("no job called {id:?}")))?;
        Ok(look(job))
    }

    /// Every job finished. The per-run coordinator's exit condition, and the
    /// persistent one's idle indicator.
    pub fn all_finished(&self) -> bool {
        let jobs = self.lock_jobs();
        !jobs.is_empty() && jobs.values().all(Job::finished)
    }

    /// The one place the two lifetimes differ.
    ///
    /// **Over** and quiet: the record has to be complete, because it is what
    /// the run is checked from. See [`DEFAULT_LINGER_MS`].
    ///
    /// Over rather than finished, so that an aborted job leaves as promptly as
    /// a completed one. A coordinator that only exited on success would turn
    /// every node loss into the hang the abort exists to prevent.
    pub fn should_exit(&self) -> bool {
        if !self.exit_when_done {
            return false;
        }
        let jobs = self.lock_jobs();
        !jobs.is_empty()
            && jobs
                .values()
                .all(|job| job.over() && job.quiet_for() >= self.linger)
    }

    /// The merged event stream, as an `ExecutionLog`.
    ///
    /// Deliberately the same type a single-node run produces, so
    /// `check_coverage_and_order`, `duplicate_applications` and
    /// `recomputed_margin_voxels` all answer for a distributed run with no
    /// distributed-specific analysis written for them.
    pub fn merged_log(&self, id: &str) -> Result<Vec<Event>> {
        self.inspect(id, |job| job.log().events())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::spec::{probe_job, ChainSpec};

    fn job(blocks: usize, phases: usize) -> Job {
        let (spec, decomposition) = probe_job(blocks, phases, ChainSpec::identity());
        Job::new(spec, decomposition).unwrap()
    }

    #[test]
    fn every_task_is_handed_out_exactly_once_when_nobody_dies() {
        let mut job = job(8, 2);
        let total = job.graph().len();
        let mut seen = Vec::new();
        loop {
            match job.pull("w1") {
                Handout::Task(assignment) => {
                    seen.push(assignment.task);
                    job.completed("w1", assignment.task).unwrap();
                }
                Handout::Wait { .. } => panic!("one worker, no concurrency: nothing to wait for"),
                Handout::Finished => break,
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..total).collect::<Vec<_>>());
        assert_eq!(job.status().done, total);
        assert!(job.reissued_tasks().is_empty());
    }

    #[test]
    fn a_second_phase_is_not_handed_out_before_its_dependencies_are_written() {
        let mut job = job(4, 2);
        let first_phase = job.graph().tasks_in_phase(0).len();
        let mut claimed = Vec::new();
        // Claim everything available without completing any of it.
        for worker in 0..first_phase + 4 {
            match job.pull(&format!("w{worker}")) {
                Handout::Task(assignment) => claimed.push(assignment.task),
                Handout::Wait { .. } => break,
                Handout::Finished => panic!("nothing has completed"),
            }
        }
        assert_eq!(claimed.len(), first_phase);
        for &task in &claimed {
            assert_eq!(job.graph().tasks[task].phase, 0, "phase 1 handed out early");
        }
        assert!(matches!(job.pull("late"), Handout::Wait { .. }));
    }

    /// The default, and the assertion that encodes its absence.
    ///
    /// A claim with no lease does not expire, however long it is held. Stated
    /// over a *deliberately* long hold, because the failure this guards is a
    /// default that quietly became finite again — `u64::MAX` milliseconds, a
    /// "sensible" fallback in a `from_json`, a `unwrap_or(30_000)` — and every
    /// one of those would pass a test that only waited a moment.
    ///
    /// Its liveness counterpart is the test directly below: the same claim,
    /// under an explicit finite lease, *is* taken away. The pair is the point.
    /// Neither half alone distinguishes "expiry is off" from "expiry is
    /// broken".
    #[test]
    fn a_claim_with_no_lease_is_never_taken_away_however_long_it_is_held() {
        let (spec, decomposition) = probe_job(6, 1, ChainSpec::identity());
        assert_eq!(spec.lease, None, "a job has no lease unless it asks");
        let mut job = Job::new(spec, decomposition).unwrap();
        let Handout::Task(held) = job.pull("holder") else {
            panic!("expected work")
        };
        assert_eq!(held.lease, None, "the handout advertised a deadline");
        // Long enough that any plausible finite default has passed, and the
        // claim is re-examined on every pull in between.
        for _ in 0..50 {
            std::thread::sleep(Duration::from_millis(2));
            let _ = job.pull("other");
        }
        assert!(
            job.reissued_tasks().is_empty(),
            "a claim was reissued by a job that has no lease: {:?}",
            job.reissued_tasks()
        );
        assert_eq!(job.status().reissued, 0);
        assert_eq!(
            job.claims_held_by("holder")
                .iter()
                .map(|claim| claim.task)
                .collect::<Vec<_>>(),
            vec![held.task],
            "the holder should still be holding exactly what it took"
        );
    }

    #[test]
    fn an_expired_claim_is_reissued_and_the_job_still_completes() {
        let mut spec = probe_job(6, 1, ChainSpec::identity());
        spec.0.lease = Some(Duration::from_millis(1));
        let mut job = Job::new(spec.0, spec.1).unwrap();
        // A worker takes one task and dies with it.
        let Handout::Task(abandoned) = job.pull("doomed") else {
            panic!("expected work")
        };
        std::thread::sleep(Duration::from_millis(5));
        let mut done = 0;
        loop {
            match job.pull("survivor") {
                Handout::Task(assignment) => {
                    job.completed("survivor", assignment.task).unwrap();
                    done += 1;
                }
                Handout::Wait { .. } => std::thread::sleep(Duration::from_millis(2)),
                Handout::Finished => break,
            }
        }
        assert_eq!(done, job.graph().len());
        assert_eq!(job.reissued_tasks(), vec![(abandoned.task, 2)]);
        assert!(job.finished());
    }

    #[test]
    fn a_completion_from_a_worker_that_was_thought_dead_is_accepted_and_ignored() {
        let mut spec = probe_job(4, 1, ChainSpec::identity());
        spec.0.lease = Some(Duration::from_millis(1));
        let mut job = Job::new(spec.0, spec.1).unwrap();
        let Handout::Task(first) = job.pull("slow") else {
            panic!("expected work")
        };
        std::thread::sleep(Duration::from_millis(5));
        // Somebody else picks it up and finishes it.
        job.expire_claims();
        job.completed("fast", first.task).unwrap();
        let done_before = job.status().done;
        // The original then reports in. Two executions wrote the same values to
        // the same region, so there is nothing to reconcile.
        job.completed("slow", first.task).unwrap();
        assert_eq!(job.status().done, done_before);
    }

    /// A node loss stops the job, and the message says enough to act on.
    ///
    /// Three separate things, and each is a way this could be useless: it has
    /// to *stop* (a job that keeps handing work to a fleet it cannot finish
    /// with is the silent hang), it has to **name the worker**, and it has to
    /// name **what that worker was holding** — because "one claim outstanding"
    /// tells an operator nothing about which part of the volume has to be
    /// redone.
    #[test]
    fn a_lost_worker_aborts_the_job_and_the_message_names_it_and_its_tasks() {
        let mut job = job(6, 1);
        let Handout::Task(taken) = job.pull("doomed") else {
            panic!("expected work")
        };
        assert!(job.aborted().is_none(), "nothing has gone wrong yet");
        let aborted = job.worker_lost("doomed", "the process exited with signal 9");
        assert_eq!(aborted.worker, "doomed");
        assert_eq!(
            aborted
                .held
                .iter()
                .map(|claim| claim.task)
                .collect::<Vec<_>>(),
            vec![taken.task],
            "the abort did not record what the lost worker was holding"
        );
        let message = aborted.message();
        for wanted in [
            "doomed",
            "signal 9",
            "aborting",
            &format!("task {}", taken.task),
        ] {
            assert!(
                message.contains(wanted),
                "the abort message does not mention {wanted:?}: {message}"
            );
        }
        // It stops. A survivor asking for work is told to go home rather than
        // being handed the five tasks that are still pending.
        assert!(matches!(job.pull("survivor"), Handout::Finished));
        assert!(!job.finished(), "the job did not finish; it gave up");
        assert!(job.over(), "an abandoned job is over");
        // And reissue is not what happened, which is the whole distinction.
        assert!(job.reissued_tasks().is_empty());
    }

    /// Two survivors can notice the same death; the first account is kept.
    #[test]
    fn a_second_report_of_the_same_loss_does_not_rewrite_the_first() {
        let mut job = job(6, 1);
        let Handout::Task(taken) = job.pull("doomed") else {
            panic!("expected work")
        };
        let first = job.worker_lost("doomed", "exited with 137");
        // By now the claim table has been read and the survivors told to stop,
        // so a second look would find nothing held and would report a loss
        // that cost no work — which is the opposite of the truth.
        let second = job.worker_lost("doomed", "noticed again, later");
        assert_eq!(first, second);
        assert_eq!(second.held.len(), 1);
        assert_eq!(second.held[0].task, taken.task);
    }

    /// A coordinator that only left on success would turn every node loss into
    /// the hang the abort exists to prevent.
    #[test]
    fn a_per_run_coordinator_leaves_when_a_job_is_abandoned_as_well_as_when_it_finishes() {
        let (spec, decomposition) = probe_job(6, 1, ChainSpec::identity());
        let coordinator = Coordinator::new(true).with_linger(Duration::from_millis(0));
        let id = coordinator.submit(spec, decomposition).unwrap();
        coordinator.pull(&id, "doomed").unwrap();
        assert!(!coordinator.should_exit(), "the job has barely started");
        coordinator
            .worker_lost(Some(&id), "doomed", "the process exited with signal 9")
            .unwrap();
        assert!(
            coordinator.should_exit(),
            "the coordinator would have served an abandoned job forever"
        );
        assert_eq!(coordinator.aborted().unwrap().worker, "doomed");
    }

    #[test]
    fn a_reported_failure_returns_the_task_without_waiting_for_the_lease() {
        let mut job = job(4, 1);
        let Handout::Task(assignment) = job.pull("w1") else {
            panic!("expected work")
        };
        job.failed("w1", assignment.task, "out of memory");
        let ready = job.ready();
        assert!(ready.contains(&assignment.task), "{ready:?}");
        assert_eq!(job.status().failed, 1);
    }

    #[test]
    fn a_worker_that_never_answers_does_not_hold_up_another_worker() {
        // The no-stall property, at the coordinator: a handout consults only
        // state the coordinator already has, so an unresponsive worker cannot
        // be on any other worker's critical path.
        let mut job = job(8, 1);
        let Handout::Task(_) = job.pull("silent") else {
            panic!("expected work")
        };
        // `silent` now never reports anything, ever. Everybody else carries on.
        for _ in 0..4 {
            assert!(matches!(job.pull("busy"), Handout::Task(_)));
        }
    }

    #[test]
    fn workers_are_seeded_apart_and_then_grow_towards_each_other() {
        let mut job = job(16, 1);
        let Handout::Task(first) = job.pull("a") else {
            panic!()
        };
        let Handout::Task(second) = job.pull("b") else {
            panic!()
        };
        let apart = (first.core.start[0] as i64 - second.core.start[0] as i64).abs();
        let volume = job.decomposition.volume[0] as i64;
        assert!(
            apart > volume / 2,
            "seeded {apart} apart in a volume of {volume}"
        );
        // and each then works outwards from where it is
        job.completed("a", first.task).unwrap();
        let Handout::Task(next) = job.pull("a") else {
            panic!()
        };
        let step = (next.core.start[0] as i64 - first.core.start[0] as i64).abs();
        assert_eq!(step, job.decomposition.phases[0].grid.block()[0] as i64);
    }

    #[test]
    fn one_registry_serves_a_job_it_was_started_with_and_jobs_sent_later() {
        // The two lifetimes, on one implementation. A per-run coordinator is
        // this with `exit_when_done` and one submit; a persistent one is this
        // with several and no exit.
        let per_run = Coordinator::new(true).with_linger(Duration::ZERO);
        let (spec, decomposition) = probe_job(4, 1, ChainSpec::identity());
        let id = per_run.submit(spec, decomposition).unwrap();
        assert!(!per_run.should_exit());
        let joined = per_run.join(None, None).unwrap();
        assert_eq!(joined.job, id);
        loop {
            match per_run.pull(&id, &joined.worker).unwrap() {
                Handout::Task(assignment) => {
                    per_run
                        .completed(&id, &joined.worker, assignment.task)
                        .unwrap();
                }
                Handout::Wait { .. } => {}
                Handout::Finished => break,
            }
        }
        assert!(per_run.should_exit());

        let persistent = Coordinator::new(false).with_linger(Duration::ZERO);
        for name in ["one", "two"] {
            let (mut spec, decomposition) = probe_job(2, 1, ChainSpec::identity());
            spec.id = name.to_string();
            persistent.submit(spec, decomposition).unwrap();
        }
        assert_eq!(persistent.job_ids(), vec!["one", "two"]);
        // With more than one job a request has to say which.
        assert!(persistent.join(None, None).is_err());
        assert!(persistent.join(Some("two"), None).is_ok());
        assert!(!persistent.should_exit());
    }

    #[test]
    fn a_duplicate_job_id_is_refused_rather_than_overwriting_a_running_job() {
        let coordinator = Coordinator::new(false);
        for _ in 0..2 {
            let (mut spec, decomposition) = probe_job(2, 1, ChainSpec::identity());
            spec.id = "same".to_string();
            let outcome = coordinator.submit(spec, decomposition);
            if outcome.is_err() {
                return;
            }
        }
        panic!("the second submission of an existing job id should have been refused");
    }

    #[test]
    fn an_unfamiliar_event_is_counted_rather_than_failing_the_run() {
        let mut job = job(2, 1);
        job.report(
            "a",
            Some(1),
            &serde_json::json!({"type": "something_new", "phase": 0}),
        );
        job.report(
            "a",
            Some(2),
            &serde_json::json!({"type": "phase_started", "phase": 0}),
        );
        assert_eq!(job.unknown_events(), 1);
        assert_eq!(job.log().len(), 1);
    }

    /// Delivery is at-least-once and the merged stream is exactly-once, and the
    /// distance between those two is one number. See `Job::reported`.
    #[test]
    fn an_event_delivered_twice_is_appended_once() {
        let mut job = job(2, 1);
        let event = serde_json::json!({"type": "phase_started", "phase": 0});
        assert!(job.report("a", Some(1), &event), "the first delivery");
        assert!(
            !job.report("a", Some(1), &event),
            "the retry of the same one"
        );
        assert_eq!(job.log().len(), 1);
        assert_eq!(job.duplicate_events(), 1);
        // The negative control: the same worker, the same event, the next
        // number — a second thing that happened, not a second delivery.
        assert!(job.report("a", Some(2), &event));
        assert_eq!(job.log().len(), 2);
        assert_eq!(job.duplicate_events(), 1);
        // And numbering is per worker, so two workers' first events are two
        // events rather than a duplicate.
        assert!(job.report("b", Some(1), &event));
        assert_eq!(job.log().len(), 3);
        assert_eq!(job.duplicate_events(), 1);
        // A sender that numbers nothing is taken at its word, which is what a
        // worker built against an older coordinator does.
        assert!(job.report("c", None, &event));
        assert!(job.report("c", None, &event));
        assert_eq!(job.log().len(), 5);
    }

    // ------------------------------------------- placing a scarce task ------

    /// A job of eight blocks and then one that reads all of them.
    fn barrier_job() -> Job {
        let shape = [8 * 8, 8, 8];
        let chain = crate::distributed::spec::ChainSpec(vec![
            crate::distributed::spec::OpSpec::new("identity", "first", [0, 0, 0]),
            crate::distributed::spec::OpSpec::new("identity", "merge", shape),
        ]);
        let (spec, decomposition) = probe_job(8, 2, chain);
        let last = decomposition.n_phases() - 1;
        assert_eq!(decomposition.phases[last].blocks.len(), 1);
        Job::new(spec, decomposition).unwrap()
    }

    /// Give `heavy` most of the first phase and `light` the rest, and stop just
    /// before the barrier is handed out.
    fn run_up_to_the_barrier(job: &mut Job, heavy: &str, light: &str) -> usize {
        let blocks = job.graph().tasks_in_phase(0).len();
        for at in 0..blocks {
            let worker = if at + 1 < blocks { heavy } else { light };
            let Handout::Task(assignment) = job.pull(worker) else {
                panic!("expected work")
            };
            job.completed(worker, assignment.task).unwrap();
        }
        let ready: Vec<usize> = (0..job.graph().len())
            .filter(|&task| job.state[task] == TaskState::Pending)
            .collect();
        assert_eq!(ready.len(), 1, "the barrier should be the only task left");
        ready[0]
    }

    /// The rule, at the coordinator: the worker that produced most of the image
    /// gets the barrier, whoever asks first.
    #[test]
    fn a_barrier_is_withheld_from_a_worker_a_better_placed_peer_could_take() {
        let mut job = barrier_job();
        let barrier = run_up_to_the_barrier(&mut job, "heavy", "light");
        let (heavy_overlap, total) = job.modelled_overlap("heavy", barrier).unwrap();
        let (light_overlap, _) = job.modelled_overlap("light", barrier).unwrap();
        assert!(
            heavy_overlap > light_overlap,
            "the fixture did not make an unequal cluster: {heavy_overlap} against \
             {light_overlap} of {total}"
        );
        assert_eq!(job.withheld(), 0, "nothing has been refused yet");
        assert!(
            matches!(job.pull("light"), Handout::Wait { .. }),
            "the barrier went to the worker holding less of it"
        );
        // The refusal is *recorded*, and this is the only place in the suite
        // that makes `withheld` non-zero on purpose. It is what keeps the
        // `withheld == 0` assertion in `tests/local_multi_node.rs` a statement
        // about an ordinary phase rather than a number that cannot move.
        assert_eq!(
            job.withheld(),
            1,
            "a task was withheld and the coordinator did not say so"
        );
        let Handout::Task(assignment) = job.pull("heavy") else {
            panic!("the best-placed worker must be able to take it")
        };
        assert_eq!(assignment.task, barrier);
        assert_eq!(job.withheld(), 1, "a handout is not a refusal");
    }

    /// The same, with the rule off: first come, first served. This is the
    /// baseline the measurement in `distributed::tests` is taken against, and it
    /// is asserted here so that "with and without" names two different
    /// behaviours rather than one.
    #[test]
    fn without_the_rule_the_barrier_goes_to_whoever_asks() {
        let mut job = barrier_job().with_scarce_placement(false);
        let barrier = run_up_to_the_barrier(&mut job, "heavy", "light");
        let Handout::Task(assignment) = job.pull("light") else {
            panic!("claim order should have given it to the first asker")
        };
        assert_eq!(assignment.task, barrier);
        // The negative control for the counter above: the same fixture and the
        // same pull, with the one rule that can withhold turned off.
        assert_eq!(
            job.withheld(),
            0,
            "nothing may be withheld with the rule off"
        );
    }

    /// A better-placed worker that is **busy** must not hold a scarce task away
    /// from an idle one. It cannot take the task now however well it would suit
    /// it, and counting it would trade a re-read for an idle cluster.
    ///
    /// Asserted on the filter itself rather than through a scenario, because the
    /// scenario is hard to stage honestly: a barrier is ready only once every
    /// task before it is *done*, so a peer holding a claim and a ready barrier
    /// rarely coexist. The filter is what has to be right, and it is what is
    /// relied on by the phase *before* the barrier, where the last few tasks are
    /// scarce while workers are still busy.
    #[test]
    fn a_busy_peer_is_not_a_contender() {
        let mut job = barrier_job();
        // "heavy" takes a block and does not report it.
        let Handout::Task(_) = job.pull("heavy") else {
            panic!("expected work")
        };
        // "light" takes one and finishes, so it is idle and has been seen.
        let Handout::Task(assignment) = job.pull("light") else {
            panic!("expected work")
        };
        job.completed("light", assignment.task).unwrap();
        let names: Vec<&str> = job.contenders("light").iter().map(|one| one.id).collect();
        assert!(
            !names.contains(&"heavy"),
            "a worker holding a claim was offered as an alternative owner: {names:?}"
        );
        let names: Vec<&str> = job.contenders("heavy").iter().map(|one| one.id).collect();
        assert_eq!(names, vec!["light"], "an idle peer should be a contender");
    }

    /// Under the baseline policy the coordinator has no contenders at all, so a
    /// task can never be withheld — which is what keeps `Naive` the thing the
    /// other policies are measured against.
    #[test]
    fn the_naive_policy_offers_no_alternative_owner() {
        let (mut spec, decomposition) = probe_job(8, 2, ChainSpec::identity());
        spec.policy = HandoutPolicy::Naive;
        let mut job = Job::new(spec, decomposition).unwrap();
        let Handout::Task(assignment) = job.pull("a") else {
            panic!("expected work")
        };
        job.completed("a", assignment.task).unwrap();
        assert!(job.contenders("b").is_empty());
    }

    /// The liveness bound, which is **not** a timer of its own: a worker stops
    /// being a contender when it stops looking alive, so a barrier withheld for
    /// a worker that has died is handed out once that worker goes stale.
    ///
    /// The wait is the cost of getting this wrong, and it is bounded by
    /// [`CONTENDER_SEEN_WITHIN_MS`] rather than by a lease or by anything a
    /// worker has to say.
    #[test]
    fn a_worker_that_stops_looking_alive_stops_holding_a_barrier_back() {
        let mut job = barrier_job();
        let barrier = run_up_to_the_barrier(&mut job, "heavy", "light");
        assert!(
            matches!(job.pull("light"), Handout::Wait { .. }),
            "the better-placed worker should be preferred while it looks alive"
        );
        // "heavy" is now killed: it never pulls, never completes, never reports.
        std::thread::sleep(Duration::from_millis(CONTENDER_SEEN_WITHIN_MS + 50));
        let Handout::Task(assignment) = job.pull("light") else {
            panic!("a dead worker held the barrier back indefinitely")
        };
        assert_eq!(assignment.task, barrier);
    }
}
