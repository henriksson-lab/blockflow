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

use serde_json::Value;

use crate::decomposition::Decomposition;
use crate::error::{Error, Result};
use crate::export::event_from_json;
use crate::graph::TaskGraph;
use crate::log::{Event, ExecutionLog};

use super::cache_model::{ChunkGrid, ModelledCache};
use super::handout::{self, HandoutPolicy, WorkerView};
use super::placement::{self, Residency};
use super::protocol::{Assignment, Handout, JobStatus, Joined, PROTOCOL_VERSION};
use super::spec::JobSpec;

/// How long a claim survives without a completion before it is reissued.
pub const DEFAULT_LEASE_MS: u64 = 30_000;
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
    lease: Duration,
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
    /// reads level `p` and writes level `p + 1`, so the level a barrier at phase
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
    unknown_events: usize,
    done: usize,
    reissued: usize,
    failed: usize,
    started: Instant,
    last_event: Instant,
    /// Whether a scarce task may be withheld from a worker with a better-placed
    /// idle peer. On, because a barrier phase is the only placement decision in
    /// its phase; off is here so the two can be measured against each other over
    /// the same job, which is the only way "it helps" means anything.
    scarce_placement: bool,
}

impl Job {
    pub fn new(spec: JobSpec, decomposition: Decomposition) -> Result<Self> {
        decomposition.check()?;
        let graph = TaskGraph::build(&decomposition);
        graph
            .dependencies_cover_reads(&decomposition)
            .map_err(Error::InvalidArgument)?;
        let remaining_deps: Vec<usize> = graph.tasks.iter().map(|task| task.deps.len()).collect();
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
            phase_tasks,
            state: vec![TaskState::Pending; n],
            attempts: vec![0; n],
            remaining_deps,
            dependents,
            claims: BTreeMap::new(),
            workers: BTreeMap::new(),
            log: ExecutionLog::new(),
            unknown_events: 0,
            done: 0,
            reissued: 0,
            failed: 0,
            started: Instant::now(),
            last_event: Instant::now(),
            scarce_placement: true,
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

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    pub fn reissued(&self) -> usize {
        self.reissued
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
    /// This is the whole of fault tolerance, and it is why the handout is a
    /// pull. A worker that is preempted, reclaimed, killed or simply wedged
    /// stops renewing, its claim expires, and the task goes to somebody else.
    /// No failure detector, no membership protocol, no second mechanism for the
    /// "worker is slow" case versus the "worker is gone" case — a slow worker's
    /// task is reissued and its eventual completion is accepted or ignored,
    /// which costs a duplicate execution and never a wrong result, because the
    /// two executions write the same values to the same valid region.
    fn expire_claims(&mut self) {
        let now = Instant::now();
        let expired: Vec<(usize, String)> = self
            .claims
            .iter()
            .filter(|(_, claim)| now.duration_since(claim.at) > claim.lease)
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

    fn ready(&self) -> Vec<usize> {
        (0..self.graph.len())
            .filter(|&task| {
                self.state[task] == TaskState::Pending && self.remaining_deps[task] == 0
            })
            .collect()
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
        self.admit(worker);
        self.expire_claims();
        if self.finished() {
            return Handout::Finished;
        }
        let ready = self.ready();
        if ready.is_empty() {
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
            // Either nothing is ready or everything ready has a better owner
            // that is idle and asking. Both are the same answer to this worker,
            // and it is answered immediately — a refused pull never blocks.
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
            lease_ms: self.spec.lease_ms,
        };
        let placed = handout::position(&self.graph, task);
        self.attempts[task] += 1;
        self.state[task] = TaskState::Claimed;
        self.claims.insert(
            task,
            Claim {
                worker: worker.to_string(),
                at: Instant::now(),
                lease: Duration::from_millis(self.spec.lease_ms),
            },
        );
        // The cache model is updated **here**, from the assignment, and never
        // from anything a worker says. See `cache_model`.
        let keys = self.chunks.keys(phase, &read);
        // What this task will write, on the same terms. Phase `p` writes level
        // `p + 1`, over the block's *valid* region — the part it owns, which is
        // what tiles the level exactly. Recorded at assignment rather than at
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
    pub fn report(&mut self, event: &Value) {
        self.last_event = Instant::now();
        let at = self.log.len();
        match event_from_json(event, at) {
            Ok(Some(event)) => self.log.push(event),
            Ok(None) => self.unknown_events += 1,
            Err(_) => self.unknown_events += 1,
        }
    }

    pub fn unknown_events(&self) -> usize {
        self.unknown_events
    }

    /// How long since anything at all was reported.
    pub fn quiet_for(&self) -> Duration {
        self.last_event.elapsed()
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
    /// every op, in the right order, once each.
    pub fn check_coverage(&self) -> Result<()> {
        let expected: Vec<(usize, String)> = self
            .decomposition
            .op_names_in_order()
            .into_iter()
            .enumerate()
            .collect();
        let blocks = self.decomposition.phases[0].blocks.len();
        self.log.check_coverage_and_order(&expected, blocks)
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
}

impl Coordinator {
    pub fn new(exit_when_done: bool) -> Self {
        Self {
            jobs: Mutex::new(BTreeMap::new()),
            order: Mutex::new(Vec::new()),
            next_worker: AtomicU64::new(1),
            exit_when_done,
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

    fn lock_jobs(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, Job>> {
        self.jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
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

    pub fn report(&self, job: &str, event: &Value) -> Result<()> {
        self.with_job(job, |job| {
            job.report(event);
            Ok(())
        })
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
    /// Finished **and quiet**: the record has to be complete, because it is
    /// what the run is checked from. See [`DEFAULT_LINGER_MS`].
    pub fn should_exit(&self) -> bool {
        if !self.exit_when_done {
            return false;
        }
        let jobs = self.lock_jobs();
        !jobs.is_empty()
            && jobs
                .values()
                .all(|job| job.finished() && job.quiet_for() >= self.linger)
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

    #[test]
    fn an_expired_claim_is_reissued_and_the_job_still_completes() {
        let mut spec = probe_job(6, 1, ChainSpec::identity());
        spec.0.lease_ms = 1;
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
        spec.0.lease_ms = 1;
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
        job.report(&serde_json::json!({"type": "something_new", "phase": 0}));
        job.report(&serde_json::json!({"type": "phase_started", "phase": 0}));
        assert_eq!(job.unknown_events(), 1);
        assert_eq!(job.log().len(), 1);
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

    /// The rule, at the coordinator: the worker that produced most of the level
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
        assert!(
            matches!(job.pull("light"), Handout::Wait { .. }),
            "the barrier went to the worker holding less of it"
        );
        let Handout::Task(assignment) = job.pull("heavy") else {
            panic!("the best-placed worker must be able to take it")
        };
        assert_eq!(assignment.task, barrier);
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
