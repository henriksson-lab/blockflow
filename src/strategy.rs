// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// `docs/design/BLOCK_OPS.md` §"Workflow -> Planner -> Executor, with the
// contract in the types". One trait, two methods — deliberately **not** a
// `Planner` trait and an `Executor` trait, because the dynamic half *is*
// planning: a greedy scheduler adapting to observed emptiness and compression
// makes planning decisions continuously, and a trait boundary between them
// would be crossed constantly or force the scheduler to re-derive what the
// planner already knew.
//
// The contract survives the merge because it lives in the *method* contracts
// and in the `Decomposition`/`Hints` split, not in the trait count:
//
// * `decompose` is **binding**. Deterministic, hashable, data-blind. Everything
//   parity-visible — block sizes, halos, valid regions, the phase partition —
//   is decided here and recorded. `run` must honour it exactly.
// * `run` is **dynamic**. It may choose visit order, concurrency, placement,
//   prefetch depth and whether a boundary lands in memory or in storage, and
//   may short-circuit a block only where an op declared `constant_maps_to`. It
//   must not alter block boundaries, halos or valid regions, skip an op absent
//   a declared algebraic property, or reorder the chain.
//
// Everything NP-hard is on the `run` side, and it is the safe side: wrong means
// slow, not wrong. Everything that must be deterministic is on the `decompose`
// side, and that part is easy — pick a block size under the budget, derive
// halos from reach, record both.
//
// One executor, many policies
// ---------------------------
// `execute` is the only thing in this crate that moves a block. Every strategy
// calls it with a different `Hints`, so the loop-nest question ("fuse or
// materialise") is a *priority over the task DAG* rather than a second code
// path: `BlockMajor` advances a block through phases (fusion), `PhaseMajor`
// finishes a phase across blocks (materialisation). Neither can change a voxel,
// because neither touches the decomposition.
//
// That also means a strategy cannot accidentally make its `run` depend on its
// own `decompose` — `execute` takes the decomposition as data, and the
// conformance suite runs `Greedy::run` against `Trivial::decompose` to prove
// it.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use rayon::prelude::*;

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::region::Region;
use crate::tiling::boxes_tile_exactly;

use super::decomposition::{
    check_block_constraints, check_dtypes, constraint_for, groups_for, is_planning_barrier,
    price_phase, region_to_ranges, splittable_axes, Constraints, Decomposition, PhaseDecomposition,
};
use super::env::block_shape;
use super::env::Environment;
use super::fragment::{check_phase_work, neighbourhood, BlockView, PhaseWork};
use super::geometry::{chunks_touched, BlockGrid};
use super::graph::TaskGraph;
use super::listener::{Dispatch, EventListener};
use super::log::{Event, Stats};
use super::op::{Anchor, Chain, Output};

/// Names an array the injected `Environment` resolves.
///
/// Deliberately not `io::SourceSpec`: the point of injection is that a workflow
/// carries no IO handles, so the same workflow can be run against real storage
/// and against a loader that only counts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArrayRef(pub String);

impl ArrayRef {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

/// Declarative. Contains **no** execution decisions.
pub struct Workflow {
    pub chain: Chain,
    pub input: ArrayRef,
    pub output: ArrayRef,
    pub shape: [usize; 3],
    pub dtype: Dtype,
}

impl Workflow {
    pub fn new(chain: Chain, shape: [usize; 3], dtype: Dtype) -> Self {
        Self {
            chain,
            input: ArrayRef::new("input"),
            output: ArrayRef::new("output"),
            shape,
            dtype,
        }
    }

    /// Every array this workflow writes: the primary output first, then the
    /// side outputs its ops declare.
    ///
    /// **Derived, not stored**, and for the reason `op` states about reach: one
    /// structure, not two. A list on the `Workflow` beside a declaration on the
    /// op is two places to say the same thing, and the one that is not walked by
    /// the executor is the one that goes stale. Folding it off the chain means
    /// an op cannot be added to execution and forgotten in the accounting.
    ///
    /// The primary's shape is **level 0's**, which is the output's too unless a
    /// phase changes it; `Decomposition::output_volume` is the authority there,
    /// and it is the plan the executor allocates from rather than this.
    pub fn outputs(&self) -> Vec<Output> {
        let mut outputs = vec![Output::new(self.output.0.clone(), self.dtype, &self.shape)];
        outputs.extend(self.chain.side_outputs(self.shape));
        outputs
    }
}

/// Which way to walk the task DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulePriority {
    /// Every block through phase 1, then phase 2. Phase-major materialisation.
    PhaseMajor,
    /// Advance one block as far through the phases as its dependencies allow.
    /// Fusion, and the smaller working set.
    BlockMajor,
}

/// **Advisory.** Performance-only: any strategy may ignore, override or
/// recompute all of it and still be correct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hints {
    /// Axes slowest- to fastest-varying, as the chain's ops prefer to be
    /// traversed. Ranks block visits and, later, prefetch. A wrong answer costs
    /// locality, never correctness, for the same reason a wrong halo costs
    /// reads: what is fetched is determined by what is asked for, not by the
    /// order it is asked in.
    pub visit_order: Option<[usize; 3]>,
    pub priority: SchedulePriority,
    pub concurrency: usize,
    /// Reserved for `MULTISLAB_IO.md` §4's hint-driven prefetcher. Recorded so
    /// a strategy can express it before there is a prefetcher to consume it.
    pub prefetch_depth: usize,
}

impl Default for Hints {
    fn default() -> Self {
        Self {
            visit_order: None,
            priority: SchedulePriority::PhaseMajor,
            concurrency: 1,
            prefetch_depth: 0,
        }
    }
}

/// The binding half and the advisory half, kept apart by type.
///
/// Ignoring `hints` is legitimate. Ignoring `decomposition` is a misuse, and
/// `execute` re-derives nothing from it — it reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub decomposition: Decomposition,
    pub hints: Hints,
}

/// One trait, two methods.
pub trait Strategy: Sync {
    fn name(&self) -> &'static str;

    /// BINDING: deterministic, hashable, data-blind.
    fn decompose(&self, workflow: &Workflow, constraints: &Constraints) -> Result<Decomposition>;

    /// What this strategy would advise. Advisory throughout.
    fn hints(&self, _workflow: &Workflow, _decomposition: &Decomposition) -> Hints {
        Hints::default()
    }

    /// DYNAMIC: must honour `decomposition` exactly; may otherwise adapt.
    fn run(
        &self,
        workflow: &Workflow,
        decomposition: &Decomposition,
        env: &dyn Environment,
    ) -> Result<Stats> {
        let hints = self.hints(workflow, decomposition);
        execute(self.name(), workflow, decomposition, &hints, env)
    }

    /// As `run`, with observers attached. Listeners are advisory throughout:
    /// see `listener::EventListener` for why one cannot change the result.
    fn run_observed(
        &self,
        workflow: &Workflow,
        decomposition: &Decomposition,
        env: &dyn Environment,
        listeners: &[Arc<dyn EventListener>],
    ) -> Result<Stats> {
        let hints = self.hints(workflow, decomposition);
        execute_observed(self.name(), workflow, decomposition, &hints, env, listeners)
    }

    fn plan(&self, workflow: &Workflow, constraints: &Constraints) -> Result<Plan> {
        let decomposition = self.decompose(workflow, constraints)?;
        let hints = self.hints(workflow, &decomposition);
        Ok(Plan {
            decomposition,
            hints,
        })
    }
}

// -------------------------------------------------------------- executor --

/// Walk the task DAG. The only code in this crate that moves a block.
pub fn execute(
    strategy: &'static str,
    workflow: &Workflow,
    decomposition: &Decomposition,
    hints: &Hints,
    env: &dyn Environment,
) -> Result<Stats> {
    execute_observed(strategy, workflow, decomposition, hints, env, &[])
}

/// `execute`, with observers attached.
///
/// The listener set is **fixed for the run**: it is a slice, not a registry, so
/// there is nothing to lock on the event path and a listener cannot be added
/// half way through a run and then disagree with the log about what happened.
/// A listener that panics is disabled and counted in `Stats::listener_faults`;
/// a listener can never fail the run. See `listener` for the argument.
pub fn execute_observed(
    strategy: &'static str,
    workflow: &Workflow,
    decomposition: &Decomposition,
    hints: &Hints,
    env: &dyn Environment,
    listeners: &[Arc<dyn EventListener>],
) -> Result<Stats> {
    let work = vec![PhaseWork::Pixels; decomposition.n_phases()];
    execute_phases(
        strategy,
        workflow,
        decomposition,
        hints,
        env,
        listeners,
        &work,
    )
}

/// `execute_observed`, with each phase saying what it runs.
///
/// The one executor, over both kinds of phase. A phase that runs a
/// [`crate::fragment::FragmentOp`] is scheduled by the same ready heap, admitted
/// by the same priority key, ordered by the same task DAG and reported on the
/// same event stream — the only thing that differs is what one task does with
/// its block, which is the whole reason a fragment op is a separate trait rather
/// than a wider `BlockOp`.
///
/// `work` has one entry per phase and there is no default: a plan whose phases
/// do not all say what they run is a plan somebody has not finished writing.
pub fn execute_phases(
    strategy: &'static str,
    workflow: &Workflow,
    decomposition: &Decomposition,
    hints: &Hints,
    env: &dyn Environment,
    listeners: &[Arc<dyn EventListener>],
    work: &[PhaseWork<'_>],
) -> Result<Stats> {
    // The guard, on a decomposition this executor may not have chosen.
    decomposition.check()?;
    check_phase_work(decomposition, work)?;

    let slots = workflow.chain.slots();
    let order = decomposition.slot_order();
    if order != (0..slots.len()).collect::<Vec<_>>() {
        return Err(Error::InvalidArgument(format!(
            "decomposition partitions the chain into slot order {order:?}, which is not the \
             chain's own order 0..{}. A decomposition may partition; it may never reorder or \
             drop an op.",
            slots.len()
        )));
    }
    // Level 0 only. What the *later* levels are shaped like is the phases' own
    // business now, and an environment that cannot host a phase which changes
    // shape says so from `prepare` — where it knows what it allocated — rather
    // than here, where the old whole-plan equality made every such plan
    // unrunnable for every environment.
    if decomposition.volume != workflow.shape || decomposition.volume != env.volume() {
        return Err(Error::InvalidArgument(format!(
            "volume disagreement: workflow {:?}, decomposition {:?}, environment {:?}",
            workflow.shape,
            decomposition.volume,
            env.volume()
        )));
    }
    // The ops' own guard on the plan. It is here rather than in
    // `Decomposition::check` because this is the first place that holds both the
    // plan and the implementations; see `check_block_constraints`.
    check_block_constraints(&workflow.chain, decomposition)?;
    // The same arrangement for the element type: declared by the op, folded by
    // the chain, and re-checked here because a plan whose levels are the wrong
    // width would otherwise be discovered one block at a time.
    check_dtypes(&workflow.chain, decomposition)?;
    env.prepare(decomposition)?;
    // Once, before any task, and by the executor rather than by each op: a
    // stream must exist before a block writes to it, and a declaration inside
    // `apply` would be a declaration repeated once per block on the hot path.
    // The *lifecycle* is still the op's, which is the part that is a decision.
    for entry in work {
        if let PhaseWork::Fragments(op) = entry {
            for output in op.outputs() {
                env.declare_sidecar(&output.stream, output.lifecycle)?;
            }
        }
    }
    // The same arrangement for the arrays an op writes beside its result: the
    // array exists before the first block writes into it, and it is declared
    // once rather than once per block.
    let side_outputs = side_outputs_per_phase(decomposition, &slots, work)?;
    for phase in &side_outputs {
        for output in phase {
            env.declare_side_output(output)?;
        }
    }
    let declared_side = distinct_side_outputs(&side_outputs)?;

    let graph = TaskGraph::build(decomposition);
    graph
        .dependencies_cover_reads(decomposition)
        .map_err(Error::InvalidArgument)?;

    let events = Dispatch::new(listeners);
    let n_phases = decomposition.n_phases();
    let concurrency = hints.concurrency.max(1);
    // Serial when one worker is asked for, and a *shared* pool otherwise.
    // Building a `rayon::ThreadPool` spawns and joins that many OS threads, and
    // `execute` is called thousands of times by the conformance sweep, so a
    // fresh pool per call was the dominant cost of the suite.
    let pool = (concurrency > 1)
        .then(|| worker_pool(concurrency))
        .transpose()?;

    let mut indegree: Vec<usize> = graph.tasks.iter().map(|task| task.deps.len()).collect();
    let dependents = graph.dependents();
    // A heap rather than a sorted vector: the ready set is re-ranked on every
    // wave, and re-sorting it was O(waves x ready x log ready) — around 3 x 10^8
    // comparisons at full scale, which would have made the *scheduler* the
    // bottleneck of a simulation whose whole point is to be free.
    let mut ready: BinaryHeap<Reverse<([usize; 5], usize)>> = (0..graph.len())
        .filter(|&id| indegree[id] == 0)
        .map(|id| Reverse((priority_key(&graph.tasks[id], hints), id)))
        .collect();
    let mut phase_remaining: Vec<usize> = (0..n_phases)
        .map(|phase| graph.tasks_in_phase(phase).len())
        .collect();
    let mut phase_started = vec![false; n_phases];
    let mut written: Vec<Vec<Vec<(usize, usize)>>> = vec![Vec::new(); n_phases];
    // Beside `written`, and checked the same way: what a block contributed to
    // each side output. Keyed by name because a side output is addressed by
    // name and may be written by more than one phase.
    let mut side_written: BTreeMap<String, Vec<Vec<(usize, usize)>>> = declared_side
        .iter()
        .map(|output| (output.name.clone(), Vec::new()))
        .collect();
    let mut phase_bytes: Vec<u64> = vec![0; n_phases];
    let mut done = 0usize;
    let mut short_circuited = 0usize;

    while done < graph.len() {
        if ready.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "task graph stalled after {done} of {} tasks; this is a dependency cycle, \
                 which a phase-layered DAG cannot have unless the graph was built wrongly",
                graph.len()
            )));
        }
        let take = concurrency.min(ready.len());
        let wave: Vec<usize> = (0..take)
            .map(|_| ready.pop().expect("checked non-empty").0 .1)
            .collect();

        for &id in &wave {
            let phase = graph.tasks[id].phase;
            if !phase_started[phase] {
                phase_started[phase] = true;
                events.emit(Event::PhaseStarted { phase });
            }
            events.emit(Event::TaskAdmitted {
                phase,
                index: graph.tasks[id].index,
            });
        }

        let run = |id: usize| {
            run_task(
                &graph.tasks[id],
                decomposition,
                &slots,
                &work[graph.tasks[id].phase],
                env,
                &events,
                n_phases,
            )
        };
        let outcomes: Vec<Result<TaskOutcome>> = match &pool {
            None => wave.iter().map(|&id| run(id)).collect(),
            Some(pool) => pool.install(|| wave.par_iter().map(|&id| run(id)).collect()),
        };

        for (&id, outcome) in wave.iter().zip(outcomes) {
            let outcome = outcome?;
            let phase = graph.tasks[id].phase;
            for (name, region) in &outcome.side_written {
                side_written
                    .get_mut(name)
                    .expect("every side output a task writes was declared")
                    .push(region.ranges());
            }
            written[phase].push(region_to_ranges(&outcome.valid));
            phase_bytes[phase] +=
                outcome.valid.voxels() as u64 * decomposition.dtype_at(phase + 1).size_of() as u64;
            if outcome.short_circuited {
                short_circuited += 1;
            }
            phase_remaining[phase] -= 1;
            if phase_remaining[phase] == 0 {
                if work[phase].writes_a_level() {
                    // A phase that wrote no level has nothing to flush and
                    // nothing to materialise, and saying otherwise would put a
                    // byte count on the stream for bytes never written.
                    env.finish(phase + 1)?;
                    events.emit(Event::Materialised {
                        phase,
                        level: phase + 1,
                        bytes: phase_bytes[phase],
                        intermediate: phase + 1 < n_phases,
                    });
                }
                if let PhaseWork::Fragments(op) = &work[phase] {
                    // The guard on the side this phase's output is actually on.
                    // The tiling check below runs over valid regions, which for
                    // a fragment phase are the cores and therefore tile
                    // whatever happened; this is the check that can fail.
                    super::fragment::check_fragment_coverage(env, decomposition, phase, *op)?;
                }
            }
            for &next in &dependents[id] {
                indegree[next] -= 1;
                if indegree[next] == 0 {
                    ready.push(Reverse((priority_key(&graph.tasks[next], hints), next)));
                }
            }
            done += 1;
        }
    }

    // The guard again, on what was *actually* written rather than on what the
    // decomposition promised. A decomposition that tiles and an executor that
    // wrote something else would otherwise agree. Against **each phase's own**
    // volume, which is the level it wrote.
    for (phase, boxes) in written.iter().enumerate() {
        boxes_tile_exactly(boxes, &decomposition.phases[phase].volume()).map_err(|err| {
            Error::InvalidArgument(format!(
                "phase {phase}: the regions the executor actually wrote do not tile the \
                 volume: {err}"
            ))
        })?;
    }

    // And the same guard on the side outputs, which is what makes
    // `BlockOp::side_region` safe to default: a mapping of the wrong rank, one
    // that leaves a hole, or one that lands two blocks on top of each other
    // fails the run rather than half-filling an array nobody looks at until
    // later. The predicate is rank-generic, so a rank-2 table is checked exactly
    // as a rank-3 volume is.
    for output in &declared_side {
        let boxes = &side_written[&output.name];
        boxes_tile_exactly(boxes, &output.shape).map_err(|err| {
            Error::InvalidArgument(format!(
                "side output {:?} ({:?}, {}): the regions the executor actually wrote do not \
                 tile it: {err}",
                output.name,
                output.shape,
                output.dtype.numpy_name()
            ))
        })?;
    }

    let (reads, writes, read_voxels, write_voxels, chunks_read, estimated_work, peak) =
        env.counters().snapshot();
    let (side_writes, _, side_bytes_written) = env.counters().side_snapshot();
    let blocks_visited = events.log().op_sequence_per_block().len();
    let listener_faults = events.faults();
    let log = events.into_log();
    Ok(Stats {
        strategy,
        decomposition_fingerprint: decomposition.fingerprint(),
        phases: n_phases,
        tasks: graph.len(),
        tasks_short_circuited: short_circuited,
        ops_applied: env
            .counters()
            .ops_applied
            .load(std::sync::atomic::Ordering::SeqCst) as usize,
        blocks_visited,
        materialisations: written
            .iter()
            .take(n_phases.saturating_sub(1))
            .map(|boxes| boxes.len())
            .sum(),
        reads,
        writes,
        read_voxels,
        write_voxels,
        chunks_read,
        side_writes,
        side_bytes_written,
        peak_resident_bytes: peak,
        estimated_work,
        listener_faults,
        log,
    })
}

/// Worker pools, shared per thread count.
fn worker_pool(threads: usize) -> Result<Arc<rayon::ThreadPool>> {
    static POOLS: OnceLock<Mutex<BTreeMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let pools = POOLS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut pools = pools
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(pool) = pools.get(&threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|err| Error::InvalidArgument(err.to_string()))?,
    );
    pools.insert(threads, Arc::clone(&pool));
    Ok(pool)
}

/// What running one `(block, phase)` task produced.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskOutcome {
    pub valid: Region,
    pub short_circuited: bool,
    /// `(side output name, region)` for every side output this task wrote.
    ///
    /// Reported rather than re-derived, on the same argument the `valid` region
    /// is: the post-run coverage guard has to run over what the executor
    /// *actually* wrote, or a plan that tiles and an executor that wrote
    /// something else would agree.
    pub side_written: Vec<(String, Region)>,
    /// Registered listeners that panicked during this task and were disabled.
    pub listener_faults: usize,
}

/// Run exactly one task of a decomposition, through the executor's own code.
///
/// The distributed worker's entry point, and it is a *thin* one on purpose.
/// `execute` above is a scheduler wrapped around a loop over `run_task`; a
/// worker that is handed its tasks one at a time by a coordinator needs the
/// loop body and none of the scheduler, so this exposes exactly that and
/// nothing else. Everything that makes a task correct — the read extent, the
/// short-circuit predicate over the halo, the op sequence, where the valid
/// region is written — is the same function either way, so a distributed run
/// cannot drift from a single-node one by having a second implementation.
///
/// **What this does not do**, and what therefore has to be done by whoever is
/// handing out tasks:
///
/// * `Decomposition::check` and `dependencies_cover_reads` — whole-plan guards,
///   and running them per task would be running them thousands of times over.
/// * `Environment::prepare` — once per environment, not once per task.
/// * the post-run check that the written regions tile the volume. A worker
///   writes a *subset* of the volume by construction, so the check is only
///   meaningful over the merged output of every worker, and the merged event
///   stream is where it belongs.
/// * scheduling: dependency order is the caller's to respect. A task whose
///   dependencies have not been written will read an intermediate that does not
///   hold what it should.
pub fn execute_task(
    chain: &Chain,
    decomposition: &Decomposition,
    task: &super::graph::Task,
    env: &dyn Environment,
    listeners: &[Arc<dyn EventListener>],
) -> Result<TaskOutcome> {
    execute_task_of(
        chain,
        decomposition,
        task,
        &PhaseWork::Pixels,
        env,
        listeners,
    )
}

/// `execute_task`, for a task whose phase may run a fragment op.
///
/// Same contract, same omissions. A distributed worker rebuilds the phase's
/// work locally from the job spec exactly as it rebuilds the chain, and hands
/// the entry for *this* task's phase.
pub fn execute_task_of(
    chain: &Chain,
    decomposition: &Decomposition,
    task: &super::graph::Task,
    work: &PhaseWork<'_>,
    env: &dyn Environment,
    listeners: &[Arc<dyn EventListener>],
) -> Result<TaskOutcome> {
    let slots = chain.slots();
    let order = decomposition.slot_order();
    if order != (0..slots.len()).collect::<Vec<_>>() {
        return Err(Error::InvalidArgument(format!(
            "decomposition partitions the chain into slot order {order:?}, which is not the \
             chain's own order 0..{}. A decomposition may partition; it may never reorder or \
             drop an op.",
            slots.len()
        )));
    }
    if task.phase >= decomposition.n_phases() {
        return Err(Error::InvalidArgument(format!(
            "task names phase {} of a decomposition with {} phases",
            task.phase,
            decomposition.n_phases()
        )));
    }
    let events = Dispatch::new(listeners);
    let mut outcome = run_task(
        task,
        decomposition,
        &slots,
        work,
        env,
        &events,
        decomposition.n_phases(),
    )?;
    outcome.listener_faults = events.faults();
    Ok(outcome)
}

#[allow(clippy::too_many_arguments)]
fn run_task(
    task: &super::graph::Task,
    decomposition: &Decomposition,
    slots: &[&Chain],
    work: &PhaseWork<'_>,
    env: &dyn Environment,
    events: &Dispatch,
    n_phases: usize,
) -> Result<TaskOutcome> {
    if let PhaseWork::Fragments(op) = work {
        return run_fragment_task(task, decomposition, *op, env, events, n_phases);
    }
    let phase = &decomposition.phases[task.phase];
    // Two regions, in two coordinate spaces, and which is which is the whole
    // content of the change that introduced `source`: `fetch` is asked of level
    // `task.phase` and is in that level's space; `read` is this phase's own read
    // extent, is what `valid` was derived from, and is the space the output goes
    // back in. They are the same region for every phase whose output grid is its
    // input grid.
    let fetch = &task.geometry.source;
    let read = &task.geometry.read;
    // What the phase's ops say they turn the fetched extent into, against the
    // read extent the plan derived. Those two must agree, because
    // `valid_within_read` slices the result at offsets measured in `read`.
    //
    // This used to be a flat refusal of any plan where `fetch.shape !=
    // read.shape`: `BlockOp::apply` wrote an output the shape of its input, so a
    // cross-grid fetch could translate but never resize. `output_shape` is what
    // closed it — an op *declares* what it produces, so a decimating or
    // upsampling phase is now a plan that either checks or is told exactly which
    // two extents disagree. For every phase whose ops keep their extent this
    // reduces to the old comparison, unchanged.
    let produced = phase_output_shape(slots, &phase.slots, block_shape(fetch)?)?;
    if produced != block_shape(read)? {
        return Err(Error::InvalidArgument(format!(
            "phase {} block {:?} fetches {:?}, its ops turn that into {produced:?}, and the \
             plan derived a read extent of {:?}. The valid region is measured inside the read \
             extent, so a result of any other shape has nowhere to land. Either the ops' \
             declared output shape or the phase's grid is wrong; they are the two things this \
             compares.",
            task.phase, task.index, fetch.shape, read.shape
        )));
    }
    let read_bytes_per_voxel = decomposition.dtype_at(task.phase).size_of() as u64;
    let write_bytes_per_voxel = decomposition.dtype_at(task.phase + 1).size_of() as u64;
    // The IO layer's own event is emitted here, around the environment call,
    // because `Environment` is the narrow waist every byte this executor moves
    // goes through — one seam rather than emission scattered through `io/`.
    let started = Instant::now();
    let mut buf = env.read(task.phase, fetch)?;
    let read_ns = started.elapsed().as_nanos() as u64;
    let read_chunks = chunks_touched(fetch, &env.chunk_shape());
    events.emit(Event::RegionRead {
        source: format!("level {}", task.phase),
        level: task.phase,
        index: Some(task.index),
        region: fetch.clone(),
        voxels: fetch.voxels(),
        bytes: fetch.voxels() as u64 * read_bytes_per_voxel,
        chunks: read_chunks,
        duration_ns: read_ns,
    });
    events.emit(Event::BlockRead {
        phase: task.phase,
        index: task.index,
        region: fetch.clone(),
        voxels: fetch.voxels(),
        chunks: read_chunks,
    });

    // The short circuit. It fires only when the block **and its halo** are
    // uniformly `value` — `read`, not `core` — and every op in the phase has
    // *declared* what that value maps to. A block whose core is empty but whose
    // halo is not can still produce non-empty output, which is why the
    // predicate is over the read extent.
    let mut short_circuited = false;
    if let Some(value) = env.uniform(&buf) {
        if let Some(constant) = fold_constant(phase, slots, value) {
            // Shaped and typed as the phase's *output*, not its input: a
            // short-circuited block skips the work and must still produce the
            // block the work would have produced.
            let replacement =
                env.constant(decomposition.dtype_at(task.phase + 1), read, constant)?;
            env.release(&buf);
            buf = replacement;
            short_circuited = true;
            events.emit(Event::BlockShortCircuited {
                phase: task.phase,
                index: task.index,
                from: value,
                to: constant,
                slots: phase.slots.clone(),
                names: phase.names.clone(),
            });
        }
    }

    let mut side_written: Vec<(String, Region)> = Vec::new();
    if !short_circuited {
        // The buffer holds `fetch`, so that is where the ops are anchored, and
        // the volume is the one `fetch` is a region of — the level that was
        // read. The op is handed the *volume* it belongs to, not the block,
        // which is what keeps a globally-anchored sample grid from moving with
        // the block.
        let at = Anchor::of_region(fetch, decomposition.volume_at(task.phase))?;
        for &slot in &phase.slots {
            let started = Instant::now();
            let next = env.apply(slots[slot], &buf, &at)?;
            let duration_ns = started.elapsed().as_nanos() as u64;
            // Side outputs, before the input buffer is released, because the op
            // is handed both what it read and what it produced. The regions come
            // from the *valid* region rather than the read extent: a side output
            // is written once per output voxel, and the halo is read twice.
            let declared = slots[slot].side_outputs(phase.volume());
            if !declared.is_empty() {
                let regions = declared
                    .iter()
                    .enumerate()
                    .map(|(which, _)| {
                        slots[slot].side_region(which, &task.geometry.valid, phase.volume())
                    })
                    .collect::<Result<Vec<Region>>>()?;
                let within = task.geometry.valid_within_read();
                let block = super::op::SideBlock {
                    at: &at,
                    within: &within,
                    regions: &regions,
                };
                let produced = env.apply_side(slots[slot], &buf, &next, &block)?;
                for ((output, region), extra) in declared.iter().zip(&regions).zip(produced.iter())
                {
                    env.write_side(output, task.phase, region, extra)?;
                    env.release_side(extra);
                    events.emit(Event::SideOutputWritten {
                        phase: task.phase,
                        index: task.index,
                        output: output.name.clone(),
                        region: region.clone(),
                        bytes: region.voxels() as u64 * output.dtype.size_of() as u64,
                    });
                    side_written.push((output.name.clone(), region.clone()));
                }
            }
            env.release(&buf);
            buf = next;
            events.emit(Event::OpApplied {
                phase: task.phase,
                index: task.index,
                slot,
                op: slots[slot].display_name(),
                over: read.clone(),
                duration_ns,
            });
        }
    }

    let within = task.geometry.valid_within_read();
    let valid = &task.geometry.valid;
    let started = Instant::now();
    env.write(task.phase + 1, &within, valid, &buf)?;
    let write_ns = started.elapsed().as_nanos() as u64;
    env.release(&buf);
    events.emit(Event::RegionWritten {
        sink: format!("level {}", task.phase + 1),
        level: task.phase + 1,
        index: Some(task.index),
        region: valid.clone(),
        voxels: valid.voxels(),
        bytes: valid.voxels() as u64 * write_bytes_per_voxel,
        chunks: chunks_touched(valid, &env.chunk_shape()),
        duration_ns: write_ns,
    });
    events.emit(Event::BlockWritten {
        phase: task.phase,
        index: task.index,
        valid: task.geometry.valid.clone(),
        materialised: task.phase + 1 < n_phases,
    });

    Ok(TaskOutcome {
        valid: task.geometry.valid.clone(),
        short_circuited,
        side_written,
        listener_faults: 0,
    })
}

/// One `(block, phase)` of a phase that runs a fragment op.
///
/// The shape of the body is deliberately the same as `run_task`'s — read, work,
/// write, report — because it is the same task in the same DAG. What differs:
///
/// * the pixel read happens only if the op declares it, so a `fragments ->
///   fragments` phase moves no voxel and touches no read counter;
/// * the inputs the op declared are gathered from the sidecar store, and
///   **nothing else is**: the neighbourhood comes from the declaration, so a
///   zero-reach phase fetches exactly one fragment per stream;
/// * the pixel write happens only if the op declares it, so a fragment-only
///   phase leaves the write counters at zero;
/// * there is no short circuit. It is licensed by `constant_maps_to`, which is
///   an algebra over pixel values, and a fragment's bytes are not pixel values.
///   An op that could skip cheaply skips inside `apply`.
#[allow(clippy::too_many_arguments)]
fn run_fragment_task(
    task: &super::graph::Task,
    decomposition: &Decomposition,
    op: &dyn super::fragment::FragmentOp,
    env: &dyn Environment,
    events: &Dispatch,
    n_phases: usize,
) -> Result<TaskOutcome> {
    let phase = &decomposition.phases[task.phase];
    // As in `run_task`: `fetch` is in the read level's space, `read` in this
    // phase's own.
    let fetch = &task.geometry.source;
    let read = &task.geometry.read;
    let read_bytes_per_voxel = decomposition.dtype_at(task.phase).size_of() as u64;
    let write_bytes_per_voxel = decomposition.dtype_at(task.phase + 1).size_of() as u64;

    let pixels = if op.reads_pixels() {
        let started = Instant::now();
        let buf = env.read(task.phase, fetch)?;
        let read_ns = started.elapsed().as_nanos() as u64;
        let read_chunks = chunks_touched(fetch, &env.chunk_shape());
        events.emit(Event::RegionRead {
            source: format!("level {}", task.phase),
            level: task.phase,
            index: Some(task.index),
            region: fetch.clone(),
            voxels: fetch.voxels(),
            bytes: fetch.voxels() as u64 * read_bytes_per_voxel,
            chunks: read_chunks,
            duration_ns: read_ns,
        });
        events.emit(Event::BlockRead {
            phase: task.phase,
            index: task.index,
            region: fetch.clone(),
            voxels: fetch.voxels(),
            chunks: read_chunks,
        });
        Some(buf)
    } else {
        None
    };

    let counts = phase.grid.blocks_per_axis();
    let mut wanted = BTreeMap::new();
    let mut gathered = BTreeMap::new();
    for input in op.inputs() {
        let blocks = neighbourhood(task.index, input.reach, counts);
        if op.gathers() {
            let mut found = Vec::with_capacity(blocks.len());
            for &block in &blocks {
                if let Some(bytes) = env.read_sidecar(&input.stream, input.phase, block)? {
                    found.push((
                        crate::sidecar::FragmentKey::new(&input.stream, input.phase, block),
                        bytes,
                    ));
                }
            }
            gathered.insert(input.stream.clone(), found);
        }
        wanted.insert(input.stream.clone(), (input.phase, blocks));
    }

    let at = Anchor::of_region(fetch, decomposition.volume_at(task.phase))?;
    let produced = {
        let view = BlockView::new(
            task.phase,
            task.index,
            &phase.grid,
            &task.geometry.core,
            read,
            &task.geometry.valid,
            at,
            decomposition.dtype_at(task.phase + 1),
            env,
            pixels.as_ref(),
            wanted,
            gathered,
        );
        op.apply(&view)?
    };
    if let Some(buf) = &pixels {
        env.release(buf);
    }

    let declared: Vec<String> = op
        .outputs()
        .into_iter()
        .map(|output| output.stream)
        .collect();
    for (stream, bytes) in &produced.fragments {
        if !declared.iter().any(|name| name == stream) {
            return Err(Error::InvalidArgument(format!(
                "fragment op {:?} wrote to stream {stream:?}, which it did not declare. \
                 Declared: {declared:?}. The declaration is what says a stream exists and \
                 what becomes of it, so an undeclared write has no lifecycle behind it.",
                op.name()
            )));
        }
        env.write_sidecar(stream, task.phase, task.index, bytes)?;
    }

    if op.writes_pixels() {
        let Some(buf) = &produced.pixels else {
            return Err(Error::InvalidArgument(format!(
                "fragment op {:?} declares `writes_pixels` and returned no buffer, so level \
                 {} would have a hole where this block's core is. Use \
                 `BlockView::output_buffer`, which works under a simulated environment too.",
                op.name(),
                task.phase + 1
            )));
        };
        let within = task.geometry.valid_within_read();
        let valid = &task.geometry.valid;
        let started = Instant::now();
        env.write(task.phase + 1, &within, valid, buf)?;
        let write_ns = started.elapsed().as_nanos() as u64;
        env.release(buf);
        events.emit(Event::RegionWritten {
            sink: format!("level {}", task.phase + 1),
            level: task.phase + 1,
            index: Some(task.index),
            region: valid.clone(),
            voxels: valid.voxels(),
            bytes: valid.voxels() as u64 * write_bytes_per_voxel,
            chunks: chunks_touched(valid, &env.chunk_shape()),
            duration_ns: write_ns,
        });
        events.emit(Event::BlockWritten {
            phase: task.phase,
            index: task.index,
            valid: valid.clone(),
            materialised: task.phase + 1 < n_phases,
        });
    } else if let Some(buf) = &produced.pixels {
        // An op that produced pixels it never declared would have them silently
        // dropped, which is the kind of quiet wrongness this crate is about.
        env.release(buf);
        return Err(Error::InvalidArgument(format!(
            "fragment op {:?} returned a pixel buffer but declares `writes_pixels() == \
             false`, so there is nowhere for it to go.",
            op.name()
        )));
    }

    Ok(TaskOutcome {
        valid: task.geometry.valid.clone(),
        short_circuited: false,
        // A fragment phase applies no slot of the chain, so it writes no side
        // output; a phase whose slots declare one is refused before any task
        // runs. See `side_outputs_per_phase`.
        side_written: Vec::new(),
        listener_faults: 0,
    })
}

/// The extent one phase turns `input` into, folding its slots in order.
///
/// The counterpart of [`fold_constant`] for shape, and the reason `run_task` can
/// now check a resizing phase instead of refusing every one: the answer comes
/// from what the ops *declared* rather than from what the buffer happened to be.
fn phase_output_shape(slots: &[&Chain], group: &[usize], input: [usize; 3]) -> Result<[usize; 3]> {
    let mut current = input;
    for &slot in group {
        current = slots[slot].output_shape(current)?;
    }
    Ok(current)
}

fn fold_constant(phase: &PhaseDecomposition, slots: &[&Chain], value: f64) -> Option<f64> {
    // A phase with a side output is never short-circuited. The short circuit is
    // licensed by `constant_maps_to`, which is an algebra over the *primary*
    // result and says nothing about the other arrays; skipping the block would
    // leave a hole in them, which the coverage guard would then report as a
    // failure of the op rather than of the skip. An op that can produce its side
    // outputs cheaply for a constant block skips inside `apply_side`.
    if phase
        .slots
        .iter()
        .any(|&slot| !slots[slot].side_outputs(phase.volume()).is_empty())
    {
        return None;
    }
    let mut current = value;
    for &slot in &phase.slots {
        current = slots[slot].constant_maps_to(current)?;
    }
    Some(current)
}

/// The side outputs each phase writes, in the order its slots write them.
///
/// A **fragment** phase whose slots declare side outputs is refused rather than
/// silently ignored: `run_fragment_task` does not apply the chain's slots at
/// all, so the arrays would be declared, allocated and never written, and the
/// coverage guard would then fire with a message about a hole rather than about
/// the phase being the wrong kind.
fn side_outputs_per_phase(
    decomposition: &Decomposition,
    slots: &[&Chain],
    work: &[PhaseWork<'_>],
) -> Result<Vec<Vec<Output>>> {
    let mut per_phase = Vec::with_capacity(decomposition.n_phases());
    for (index, phase) in decomposition.phases.iter().enumerate() {
        let declared: Vec<Output> = phase
            .slots
            .iter()
            .flat_map(|&slot| slots[slot].side_outputs(phase.volume()))
            .collect();
        if !declared.is_empty() && matches!(work[index], PhaseWork::Fragments(_)) {
            return Err(Error::InvalidArgument(format!(
                "phase {index} ({}) runs a fragment op, and its block ops declare {} side \
                 output(s) that nothing would write. A fragment phase applies no slot of the \
                 chain; put the op in a pixel phase, or emit the extra result as a fragment \
                 stream.",
                phase.names.join(">"),
                declared.len()
            )));
        }
        per_phase.push(declared);
    }
    Ok(per_phase)
}

/// One entry per distinct side output, and a refusal when two declarations of
/// one name disagree.
///
/// Two phases contributing to one array is legitimate — that is how a workflow
/// builds a table from several stages — but only if they agree about what the
/// array is. Two disagreeing declarations would make the array's shape depend on
/// which phase ran last.
fn distinct_side_outputs(per_phase: &[Vec<Output>]) -> Result<Vec<Output>> {
    let mut distinct: Vec<Output> = Vec::new();
    for phase in per_phase {
        for output in phase {
            match distinct.iter().find(|held| held.name == output.name) {
                None => distinct.push(output.clone()),
                Some(held) if held == output => {}
                Some(held) => {
                    return Err(Error::InvalidArgument(format!(
                        "side output {:?} is declared as {:?}/{} and as {:?}/{}",
                        output.name,
                        held.shape,
                        held.dtype.numpy_name(),
                        output.shape,
                        output.dtype.numpy_name()
                    )))
                }
            }
        }
    }
    Ok(distinct)
}

/// Where a task sits in the visit order the hints ask for.
///
/// A flat `[usize; 5]` so both policies are one comparable type: the *only*
/// difference between fusion and phase-major materialisation is where `phase`
/// sits in this key.
fn priority_key(task: &super::graph::Task, hints: &Hints) -> [usize; 5] {
    let axes = hints.visit_order.unwrap_or([0, 1, 2]);
    let spatial = [
        task.index[axes[0]],
        task.index[axes[1]],
        task.index[axes[2]],
    ];
    match hints.priority {
        SchedulePriority::PhaseMajor => {
            [task.phase, spatial[0], spatial[1], spatial[2], task.block]
        }
        SchedulePriority::BlockMajor => {
            [spatial[0], spatial[1], spatial[2], task.phase, task.block]
        }
    }
}

// ------------------------------------------------------------ strategies --

/// The oracle: one block, no seams, one phase, serial.
///
/// Obviously correct — there is no halo to get wrong and no seam to stitch — so
/// every other strategy's output must equal this one's. That is the whole
/// conformance suite, and it needs no reference implementation written by hand.
#[derive(Debug, Default, Clone, Copy)]
pub struct Trivial;

impl Strategy for Trivial {
    fn name(&self) -> &'static str {
        "trivial"
    }

    fn decompose(&self, workflow: &Workflow, _constraints: &Constraints) -> Result<Decomposition> {
        let grid = BlockGrid::whole(workflow.shape)?;
        let slots = workflow.chain.slots();
        let reach = workflow.chain.reach3(&workflow.shape);
        let names = slots.iter().map(|slot| slot.display_name()).collect();
        let mut decomposition = Decomposition {
            volume: workflow.shape,
            dtype: workflow.dtype,
            phases: vec![PhaseDecomposition::derive(
                (0..slots.len()).collect(),
                names,
                reach,
                [0, 0, 0],
                grid,
            )],
            chain_reach: reach,
        };
        decomposition.declare_dtypes(&workflow.chain)?;
        decomposition.check()?;
        // The oracle has exactly one plan to offer, so it consults the ops'
        // constraint by *checking* rather than by choosing: an op that mandates
        // anything other than the whole volume is told plainly that this
        // strategy has nothing for it, instead of being handed the one grid it
        // cannot use.
        check_block_constraints(&workflow.chain, &decomposition)?;
        Ok(decomposition)
    }

    fn hints(&self, _workflow: &Workflow, _decomposition: &Decomposition) -> Hints {
        Hints {
            visit_order: None,
            priority: SchedulePriority::PhaseMajor,
            concurrency: 1,
            prefetch_depth: 0,
        }
    }
}

/// Brute force over the `2^(n-1)` partitions, with the block size for each
/// phase chosen independently.
///
/// Enumeration rather than the `O(n^2)` DP because a chain is 5-8 ops, so
/// `2^(n-1) <= 128`: simpler to write, simpler to verify, and fast enough that
/// the schedule is recomputed whenever a parameter changes rather than cached
/// and trusted. The DP is the right structure if a chain ever gets long, and
/// `MAX_SLOTS` refuses rather than quietly taking minutes.
///
/// Per-phase block sizes are separable given the partition — the total is
/// `sum over phases of n_blocks_p x cost_per_block_p` and the budget binds each
/// phase independently — so choosing them costs `partitions x candidates`, not
/// `partitions x candidates^phases`.
#[derive(Debug, Clone)]
pub struct Enumerating {
    pub concurrency: usize,
    pub priority: SchedulePriority,
}

impl Default for Enumerating {
    fn default() -> Self {
        Self {
            concurrency: 1,
            priority: SchedulePriority::PhaseMajor,
        }
    }
}

/// Enumeration is `2^(n-1)`. Past this, reach for the DP.
pub const MAX_SLOTS: usize = 20;

impl Strategy for Enumerating {
    fn name(&self) -> &'static str {
        "enumerating"
    }

    fn decompose(&self, workflow: &Workflow, constraints: &Constraints) -> Result<Decomposition> {
        let slots = workflow.chain.slots();
        if slots.is_empty() {
            return Err(Error::InvalidArgument(
                "enumerating: the chain has no ops".to_string(),
            ));
        }
        if slots.len() > MAX_SLOTS {
            return Err(Error::InvalidArgument(format!(
                "enumerating: {} slots exceeds the enumeration limit of {MAX_SLOTS}; this is \
                 the case docs/design/BLOCK_OPS.md reserves the O(n^2) DP for",
                slots.len()
            )));
        }
        if constraints.block_candidates.is_empty() {
            return Err(Error::InvalidArgument(
                "enumerating: no block size candidates".to_string(),
            ));
        }
        let volume = workflow.shape;
        let bytes = workflow.dtype.size_of() as f64;
        let mut best: Option<(f64, usize, u32, Vec<PhaseDecomposition>)> = None;
        let mut budget_failures = 0usize;
        // Why a partition was dropped for a reason that is not the budget. Kept
        // so the final refusal can say "these two ops mandate different blocks"
        // rather than blaming a budget that was never the problem.
        let mut constraint_note: Option<String> = None;
        // Cuts the enumeration is not free to skip: a full-reach op is a
        // planning barrier, so it is its own phase whatever the cost model
        // thinks. See `is_planning_barrier`.
        let forced_cuts = barrier_cuts(&slots, volume);

        for mask in 0u32..(1u32 << (slots.len() - 1)) {
            if mask & forced_cuts != forced_cuts {
                continue;
            }
            let groups = groups_for(mask, slots.len());
            let mut total = 0.0_f64;
            let mut phases = Vec::with_capacity(groups.len());
            let mut feasible = true;

            for (position, group) in groups.iter().enumerate() {
                let (reach, compute, names, orders) =
                    super::decomposition::summarise_slots(&slots, group, volume);
                let is_materialised = position + 1 < groups.len();
                // What the ops in this group will accept. A conflict is a fact
                // about *this partition* — the same two ops in two phases are
                // fine — so it drops the partition and the search goes on.
                let mandated = match constraint_for(&slots, group, volume) {
                    Ok(found) => found,
                    Err(err) => {
                        constraint_note.get_or_insert_with(|| err.to_string());
                        feasible = false;
                        break;
                    }
                };
                let price = |grid: &BlockGrid| {
                    price_phase(
                        grid,
                        reach,
                        compute,
                        orders.len(),
                        is_materialised,
                        bytes,
                        &constraints.model,
                        constraints.model.materialise_cost_per_voxel,
                    )
                };
                let affordable = |cost: &super::decomposition::PhaseCost| {
                    constraints.budget_bytes.is_none_or(|budget| {
                        cost.working_set_bytes_per_block
                            * constraints.expected_concurrency.max(1) as f64
                            <= budget as f64
                    })
                };
                let mut chosen: Option<(f64, usize, BlockGrid)> = None;
                if let Some(constraint) = &mandated {
                    // A mandate replaces the candidate list rather than filtering
                    // it: `block_candidates` is a list of scalar edges and a
                    // mandated shape is anisotropic in general, so it is not
                    // expressible as a candidate at all. The budget still binds —
                    // a block that does not fit does not fit — but there is
                    // nothing to choose between.
                    match constraint.grid(volume) {
                        Some(grid) => {
                            let cost = price(&grid);
                            if affordable(&cost) {
                                chosen =
                                    Some((cost.cost_per_block * grid.n_blocks() as f64, 0, grid));
                            }
                        }
                        None => {
                            constraint_note.get_or_insert_with(|| {
                                format!(
                                    "the ops of phase {position} mandate {constraint:?}, which \
                                     no block grid produces: a grid's cores are `index * \
                                     block`, evenly strided and disjoint. A plan for it has to \
                                     state each block's fetch region explicitly."
                                )
                            });
                        }
                    }
                } else {
                    let axes = splittable_axes(&constraints.split_axes, reach, volume);
                    for &edge in &constraints.block_candidates {
                        let grid = match BlockGrid::along(volume, &axes, edge) {
                            Ok(grid) => grid,
                            Err(_) => continue,
                        };
                        let cost = price(&grid);
                        if !affordable(&cost) {
                            continue;
                        }
                        let phase_total = cost.cost_per_block * grid.n_blocks() as f64;
                        // deterministic: lower cost, then the larger block edge
                        let better = match &chosen {
                            None => true,
                            Some((best_cost, best_edge, _)) => {
                                (phase_total, std::cmp::Reverse(edge))
                                    < (*best_cost, std::cmp::Reverse(*best_edge))
                            }
                        };
                        if better {
                            chosen = Some((phase_total, edge, grid));
                        }
                    }
                }
                let Some((phase_total, _, grid)) = chosen else {
                    feasible = false;
                    break;
                };
                let phase = PhaseDecomposition::derive(group.clone(), names, reach, reach, grid);
                // The same check `execute` will run, run here: a mandated extent
                // and a non-zero reach are not jointly satisfiable, and the place
                // to discover that is the planner rather than the run.
                if let Some(constraint) = &mandated {
                    if let Err(err) = constraint.check(&phase.blocks, &format!("phase {position}"))
                    {
                        constraint_note.get_or_insert_with(|| err.to_string());
                        feasible = false;
                        break;
                    }
                }
                total += phase_total;
                phases.push(phase);
            }

            if !feasible {
                budget_failures += 1;
                continue;
            }
            let better = match &best {
                None => true,
                Some((best_cost, best_phases, best_mask, _)) => {
                    (total, phases.len(), mask) < (*best_cost, *best_phases, *best_mask)
                }
            };
            if better {
                best = Some((total, phases.len(), mask, phases));
            }
        }

        let (_, _, _, phases) = best.ok_or_else(|| {
            Error::InvalidArgument(format!(
                "enumerating: none of the {} partitions fits the {:?} byte budget at \
                 concurrency {} with block candidates {:?}. Reduce the concurrency, add a \
                 smaller block candidate, or raise the budget.{}",
                budget_failures,
                constraints.budget_bytes,
                constraints.expected_concurrency,
                constraints.block_candidates,
                match &constraint_note {
                    None => String::new(),
                    Some(note) => format!(
                        " At least one partition was dropped for a reason that is not the \
                         budget: {note}"
                    ),
                }
            ))
        })?;

        let mut decomposition = Decomposition {
            volume,
            dtype: workflow.dtype,
            phases,
            chain_reach: workflow.chain.reach3(&volume),
        };
        decomposition.declare_dtypes(&workflow.chain)?;
        decomposition.check()?;
        Ok(decomposition)
    }

    fn hints(&self, workflow: &Workflow, decomposition: &Decomposition) -> Hints {
        Hints {
            visit_order: consensus_order(workflow, decomposition),
            priority: self.priority,
            concurrency: self.concurrency,
            prefetch_depth: 1,
        }
    }
}

/// Cut where the ops disagree about traversal, fuse everywhere else; take the
/// largest block that fits the budget.
///
/// The decomposition is a heuristic — "disagreement on preferred order is a
/// candidate phase boundary" — and the run is the concurrent, block-major walk
/// of the DAG. It exists mainly so the conformance suite has a second,
/// genuinely different strategy to cross-pair with `Trivial`.
#[derive(Debug, Clone)]
pub struct Greedy {
    pub concurrency: usize,
}

impl Default for Greedy {
    fn default() -> Self {
        Self { concurrency: 4 }
    }
}

impl Strategy for Greedy {
    fn name(&self) -> &'static str {
        "greedy"
    }

    fn decompose(&self, workflow: &Workflow, constraints: &Constraints) -> Result<Decomposition> {
        let slots = workflow.chain.slots();
        if slots.is_empty() {
            return Err(Error::InvalidArgument(
                "greedy: the chain has no ops".to_string(),
            ));
        }
        let volume = workflow.shape;
        let bytes = workflow.dtype.size_of() as f64;

        // Cut at every full-reach op, and wherever the traversal preference
        // changes. The first is structural and the second is the heuristic:
        // fusing across a barrier is not a cost trade-off this strategy is
        // entitled to make. See `is_planning_barrier`.
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut current: Vec<usize> = Vec::new();
        let mut current_order: Option<[usize; 3]> = None;
        let mut current_mandate: Option<super::op::BlockConstraint> = None;
        let mut after_barrier = false;
        for (position, slot) in slots.iter().enumerate() {
            let order = slot.preferred_iterations().first().copied();
            let barrier = is_planning_barrier(slot, volume);
            // A third reason to cut, structural like the barrier rather than
            // heuristic like the order: two ops that mandate different blocks
            // cannot share a phase, and cutting between them is the plan that
            // runs. An op with no mandate joins whichever phase it lands in.
            let mandate = slot.block_constraint(volume)?;
            let order_changed =
                order.is_some() && current_order.is_some() && order != current_order;
            let mandate_changed =
                mandate.is_some() && current_mandate.is_some() && mandate != current_mandate;
            if !current.is_empty() && (barrier || after_barrier || order_changed || mandate_changed)
            {
                groups.push(std::mem::take(&mut current));
                current_order = order;
                current_mandate = mandate;
            } else {
                if current_order.is_none() {
                    current_order = order;
                }
                if current_mandate.is_none() {
                    current_mandate = mandate;
                }
            }
            current.push(position);
            after_barrier = barrier;
        }
        groups.push(current);

        let mut phases = Vec::with_capacity(groups.len());
        for (position, group) in groups.iter().enumerate() {
            let (reach, compute, names, orders) =
                super::decomposition::summarise_slots(&slots, group, volume);
            let is_materialised = position + 1 < groups.len();
            let mandated = constraint_for(&slots, group, volume)?;
            let mut grid = None;
            if let Some(constraint) = &mandated {
                // Mandated, so there is nothing to choose between; the budget
                // still binds. See `Enumerating` for why the candidate list is
                // replaced rather than filtered.
                let candidate = constraint.grid(volume).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "greedy: phase {position} mandates {constraint:?}, which no block grid \
                         produces — a grid's cores are `index * block`, evenly strided and \
                         disjoint. A plan for it has to state each block's fetch region \
                         explicitly, which this strategy does not do."
                    ))
                })?;
                let cost = price_phase(
                    &candidate,
                    reach,
                    compute,
                    orders.len(),
                    is_materialised,
                    bytes,
                    &constraints.model,
                    constraints.model.materialise_cost_per_voxel,
                );
                let fits = constraints.budget_bytes.is_none_or(|budget| {
                    cost.working_set_bytes_per_block
                        * constraints.expected_concurrency.max(1) as f64
                        <= budget as f64
                });
                if fits {
                    grid = Some(candidate);
                }
            } else {
                // largest candidate that fits
                let mut candidates = constraints.block_candidates.clone();
                candidates.sort_unstable_by(|a, b| b.cmp(a));
                let axes = splittable_axes(&constraints.split_axes, reach, volume);
                for edge in candidates {
                    let Ok(candidate) = BlockGrid::along(volume, &axes, edge) else {
                        continue;
                    };
                    let cost = price_phase(
                        &candidate,
                        reach,
                        compute,
                        orders.len(),
                        is_materialised,
                        bytes,
                        &constraints.model,
                        constraints.model.materialise_cost_per_voxel,
                    );
                    let fits = constraints.budget_bytes.is_none_or(|budget| {
                        cost.working_set_bytes_per_block
                            * constraints.expected_concurrency.max(1) as f64
                            <= budget as f64
                    });
                    if fits {
                        grid = Some(candidate);
                        break;
                    }
                }
            }
            let grid = grid.ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "greedy: no block candidate in {:?} fits the {:?} byte budget for phase \
                     {position} (reach {reach:?})",
                    constraints.block_candidates, constraints.budget_bytes
                ))
            })?;
            let phase = PhaseDecomposition::derive(group.clone(), names, reach, reach, grid);
            if let Some(constraint) = &mandated {
                constraint.check(&phase.blocks, &format!("greedy: phase {position}"))?;
            }
            phases.push(phase);
        }

        let mut decomposition = Decomposition {
            volume,
            dtype: workflow.dtype,
            phases,
            chain_reach: workflow.chain.reach3(&volume),
        };
        decomposition.declare_dtypes(&workflow.chain)?;
        decomposition.check()?;
        Ok(decomposition)
    }

    fn hints(&self, workflow: &Workflow, decomposition: &Decomposition) -> Hints {
        Hints {
            visit_order: consensus_order(workflow, decomposition),
            priority: SchedulePriority::BlockMajor,
            concurrency: self.concurrency,
            prefetch_depth: 2,
        }
    }
}

/// Cut mask bits the enumeration may not clear: one on each side of every
/// full-reach slot, so a barrier is always alone in its phase.
///
/// Stated as a mask over the same bit positions `groups_for` reads, so the
/// constraint costs one `&` per candidate partition and removes those
/// partitions from the search rather than pricing them out of it. A structural
/// fact should not depend on a weight.
fn barrier_cuts(slots: &[&Chain], volume: [usize; 3]) -> u32 {
    let barrier: Vec<bool> = slots
        .iter()
        .map(|slot| is_planning_barrier(slot, volume))
        .collect();
    let mut mask = 0u32;
    for slot in 1..slots.len() {
        if barrier[slot] || barrier[slot - 1] {
            mask |= 1 << (slot - 1);
        }
    }
    mask
}

/// The chain's traversal preference, if its ops agree. Disagreement yields
/// `None` — no preference is better than an arbitrary one.
fn consensus_order(workflow: &Workflow, _decomposition: &Decomposition) -> Option<[usize; 3]> {
    let orders = workflow.chain.preferred_iterations();
    (orders.len() == 1).then(|| orders[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probes::IdentityOp;

    fn workflow(chain: Chain, shape: [usize; 3]) -> Workflow {
        Workflow::new(chain, shape, Dtype::F64)
    }

    #[test]
    fn the_trivial_decomposition_has_one_block_and_one_phase() {
        let chain = Chain::sequence(vec![
            Chain::op(IdentityOp::new("a", [7, 7, 7])),
            Chain::op(IdentityOp::new("b", [9, 0, 0])),
        ]);
        let workflow = workflow(chain, [16, 8, 8]);
        let decomposition = Trivial
            .decompose(&workflow, &Constraints::default())
            .unwrap();
        assert_eq!(decomposition.n_phases(), 1);
        assert_eq!(decomposition.n_tasks(), 1);
        // reach is recorded honestly even though a single block cannot be short
        assert_eq!(decomposition.chain_reach, [16, 7, 7]);
        assert_eq!(decomposition.phases[0].halo, [0, 0, 0]);
        decomposition.check().unwrap();
    }

    #[test]
    fn a_decomposition_may_partition_but_never_reorder() {
        let chain = Chain::sequence(vec![
            Chain::op(IdentityOp::new("a", [0, 0, 0])),
            Chain::op(IdentityOp::new("b", [0, 0, 0])),
        ]);
        let workflow = workflow(chain, [8, 4, 4]);
        let mut decomposition = Trivial
            .decompose(&workflow, &Constraints::default())
            .unwrap();
        decomposition.phases[0].slots = vec![1, 0];
        let env = super::super::env::AccountingEnvironment::new([8, 4, 4], [4, 4, 4], 8);
        let err = execute("test", &workflow, &decomposition, &Hints::default(), &env).unwrap_err();
        assert!(err.to_string().contains("never reorder or drop an op"));
    }
}
