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
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use rayon::prelude::*;

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::region::Region;
use crate::tiling::boxes_tile_exactly;

use super::decomposition::{
    check_block_constraints, check_dtypes, check_output_shapes, check_source_levels,
    compute_per_voxel, constraint_for, groups_for, is_planning_barrier, price_phase,
    region_to_ranges, splittable_axes, Constraints, Decomposition, PhaseDecomposition, Visibility,
};
use super::env::{block_shape, BlockBuf, Environment};
use super::fragment::{check_phase_work, neighbourhood, BlockView, PhaseWork};
use super::geometry::{chunks_touched, BlockGrid};
use super::graph::{Task, TaskGraph};
use super::iterate::{IterativeOp, Operand};
use super::listener::{Dispatch, EventListener};
use super::log::{Event, Stats};
use super::op::{place_parts, Anchor, Chain, Output, Placement};
use super::reach::Reach;

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
    /// Internal levels to keep rather than free when their reader finishes.
    ///
    /// **Advisory, and it belongs here rather than in the plan for one reason:
    /// keeping an intermediate cannot change a voxel.** It changes what is on
    /// disk when the run ends, which is a debugging decision, so it sits with
    /// every other value a strategy may get wrong at no cost to the answer.
    ///
    /// Empty means "free every internal level as soon as its reader is done",
    /// which is the behaviour worth having by default: an `N`-phase chain then
    /// holds three levels at once instead of `N + 1`. Naming a level here is how
    /// a caller says "I want to look at that one" — the same choice the sidecar
    /// store spells `Lifecycle::Persistent`.
    ///
    /// Naming level 0 or the output level is harmless and does nothing; neither
    /// is ever freed.
    pub keep_levels: BTreeSet<usize>,
}

impl Default for Hints {
    fn default() -> Self {
        Self {
            visit_order: None,
            priority: SchedulePriority::PhaseMajor,
            concurrency: 1,
            prefetch_depth: 0,
            keep_levels: BTreeSet::new(),
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
    check_dtypes(&workflow.chain, decomposition, work)?;
    // And the same arrangement for the extent. It is not the per-block
    // comparison `run_task` makes — that one an op may answer out of the plan,
    // by taking `Placement::writes` — it is the same question at whole-volume
    // scale with the plan's answer withheld, which is the one form of it no op
    // can answer from the plan. See `check_output_shapes`.
    check_output_shapes(&workflow.chain, decomposition, work)?;
    // And the same arrangement for the levels a phase reads besides its own
    // input. **Before `prepare` and before the graph**: a forward reference is
    // a plan that is not a plan, and it is refused by name here rather than
    // becoming a missing dependency edge or a block asking for a level nothing
    // has written.
    check_source_levels(&workflow.chain, decomposition)?;
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

    let mut indegree: Vec<usize> = graph.tasks.iter().map(Task::n_dependencies).collect();
    let dependents = graph.dependents();
    // An iterative phase runs as a whole, so its tasks never enter the ready
    // heap: every block of substage `k+1` reads cores its neighbours wrote at
    // `k`, so the blocks advance in lockstep and there is nothing to interleave.
    // What is tracked instead is how many of the phase's tasks have become ready,
    // and the phase runs when all of them have.
    //
    // **This cannot deadlock.** A task of phase `p` waits only on phase `p-1`, so
    // holding phase `p`'s tasks back blocks nothing that phase `p`'s tasks need;
    // phase 0's tasks are ready from the start, and the property carries forward.
    let iterative: Vec<bool> = work.iter().map(|entry| entry.is_iterative()).collect();
    let mut iterative_ready: Vec<usize> = vec![0; n_phases];
    let mut iterative_run: Vec<bool> = vec![false; n_phases];
    // A heap rather than a sorted vector: the ready set is re-ranked on every
    // wave, and re-sorting it was O(waves x ready x log ready) — around 3 x 10^8
    // comparisons at full scale, which would have made the *scheduler* the
    // bottleneck of a simulation whose whole point is to be free.
    let mut ready: BinaryHeap<Reverse<([usize; 5], usize)>> = BinaryHeap::new();
    for id in 0..graph.len() {
        if indegree[id] == 0 {
            admit(
                &graph.tasks[id],
                hints,
                &iterative,
                &mut ready,
                &mut iterative_ready,
            );
        }
    }
    // How many substages each phase ran. Zero for every phase that is not an
    // iteration, which is what a phase with no loop in it took.
    let mut substages: Vec<usize> = vec![0; n_phases];
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
        // An iterative phase whose every task is ready runs now, as a whole,
        // before anything is popped: it is a barrier by construction and there is
        // no benefit to interleaving other work around it in a serial executor.
        let pending = (0..n_phases).find(|&phase| {
            iterative[phase]
                && !iterative_run[phase]
                && iterative_ready[phase] == graph.tasks_in_phase(phase).len()
        });
        let completed: Vec<(usize, TaskOutcome)> = match pending {
            Some(phase) => {
                iterative_run[phase] = true;
                if !phase_started[phase] {
                    phase_started[phase] = true;
                    events.emit(Event::PhaseStarted { phase });
                }
                for task in graph.tasks_in_phase(phase) {
                    events.emit(Event::TaskAdmitted {
                        phase,
                        index: task.index,
                    });
                }
                let PhaseWork::Iterate(op) = &work[phase] else {
                    unreachable!("`iterative` is derived from `work`");
                };
                let (ran, outcomes) =
                    run_iterative_phase(&graph, phase, decomposition, *op, env, &events, n_phases)?;
                substages[phase] = ran;
                graph
                    .tasks_in_phase(phase)
                    .iter()
                    .map(|task| task.id)
                    .zip(outcomes)
                    .collect()
            }
            None => {
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
                let mut gathered = Vec::with_capacity(wave.len());
                for (&id, outcome) in wave.iter().zip(outcomes) {
                    gathered.push((id, outcome?));
                }
                gathered
            }
        };

        for (id, outcome) in completed {
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
                // Every level whose **last** reader is this phase is now dead.
                //
                // This used to be the single level `phase` reads, on the
                // argument that exactly one phase reads a level. A source leaf
                // makes that a special case: a level read by a later phase has a
                // second reader, and freeing it here would free something still
                // wanted. `levels_dead_after` is the general statement — a level
                // dies after its last reader — and it answers `[phase]` for
                // every plan with no source leaf, so this is the same behaviour
                // stated in a way that stays true when there are two.
                //
                // The saving is the whole point of `Visibility`: without this an
                // `N`-phase chain holds `N + 1` full levels for the length of
                // the run, and only ever two of them are live.
                for level in decomposition.levels_dead_after(phase) {
                    if decomposition.level_visibility(level) == Visibility::Internal
                        && !hints.keep_levels.contains(&level)
                    {
                        // The phase goes with the level, so that a reader who
                        // wanted it back is told which `keep_levels` entry they
                        // needed rather than only that it is gone.
                        env.discard_level_after(level, phase)?;
                    }
                }
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
                    admit(
                        &graph.tasks[next],
                        hints,
                        &iterative,
                        &mut ready,
                        &mut iterative_ready,
                    );
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
        substages,
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

/// A task whose dependencies are all done joins the ready heap — unless its
/// phase is an iteration, in which case it is counted instead.
///
/// One function rather than the same three lines at the two places a task
/// becomes ready, because the two are exactly the same decision and a scheduler
/// that admitted an iterative task in one of them would run one block of a
/// lockstep phase on its own.
fn admit(
    task: &super::graph::Task,
    hints: &Hints,
    iterative: &[bool],
    ready: &mut BinaryHeap<Reverse<([usize; 5], usize)>>,
    iterative_ready: &mut [usize],
) {
    if iterative[task.phase] {
        iterative_ready[task.phase] += 1;
    } else {
        ready.push(Reverse((priority_key(task, hints), task.id)));
    }
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
    if let PhaseWork::Iterate(op) = work {
        // Refused rather than fallen through. An iterative phase owns no chain
        // slot, so the `Pixels` path below would apply nothing and copy the input
        // to the output — a complete, well-formed volume that is not the fixed
        // point. The scheduler keeps such a task out of the ready heap so that
        // the whole phase runs in lockstep; this is the guard that says so if
        // some other caller hands one over anyway.
        return Err(Error::InvalidArgument(format!(
            "task (phase {}, block {:?}) belongs to iterative op {:?}, which cannot be run one \
             block at a time: every block of substage k+1 reads cores its neighbours wrote at \
             substage k, so the phase advances in lockstep and `execute_phases` runs all of it \
             at once.",
            task.phase,
            task.index,
            op.name()
        )));
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
    // Where this block sits in **every** space the phase touches, which is two
    // regions the executor has had all along and passed one of: `fetch` is in
    // the level that was read, `read` is in the level being written. They are
    // the same region in the same volume for every phase whose output grid is
    // its input grid, which is why one anchor sufficed until a lattice phase
    // needed both. See `op::Placement`.
    //
    // The buffer holds `fetch`, so that is where the ops read from, and the
    // volume is the one `fetch` is a region of — the level that was read. An op
    // is handed the *volume* it belongs to, not the block, which is what keeps a
    // globally-anchored sample grid from moving with the block.
    //
    // `place_parts` then derives one placement **per slot** rather than handing
    // every slot the phase's own: a slot whose output grid is not its input grid
    // moves the space the slots after it are anchored in, and passing one anchor
    // to all of them was right only while no slot did that.
    let placement = Placement::new(
        Anchor::of_region(fetch, decomposition.volume_at(task.phase))?,
        Anchor::of_region(read, decomposition.volume_at(task.phase + 1))?,
    )
    .writing(block_shape(read)?);
    let places = place_parts(
        &phase
            .slots
            .iter()
            .map(|&slot| slots[slot])
            .collect::<Vec<_>>(),
        &placement,
        block_shape(fetch)?,
    );
    // The same fold `place_parts` above feeds, through the one derivation both
    // this and `check_output_shapes` use. **This comparison is the one an op may
    // answer out of the plan**: `placement` carries the read extent, so a slot
    // taking `Placement::writes` makes the two sides one number. That is what
    // the whole-volume guard is there to make non-silent; see
    // `decomposition::check_output_shapes`.
    let produced = crate::op::parts_output_shape(
        &phase
            .slots
            .iter()
            .map(|&slot| slots[slot])
            .collect::<Vec<_>>(),
        &placement,
        block_shape(fetch)?,
    )?;
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
        // The levels this phase's source leaves read, at **the same region**:
        // a source leaf has reach 0, so what it reads is what the block already
        // fetches, and `check_source_levels` is what makes those the same
        // integers by requiring the two levels to be on one lattice.
        //
        // Read here rather than beside the input read for one reason: a block
        // that short circuits has not looked at its input, and reading a second
        // array for it would be paying for an arm nothing consumed. (Today a
        // phase with a source leaf never short circuits — `Chain::Source`
        // declines `constant_maps_to` — so this is a property of the code
        // rather than of the plan, and worth keeping true by construction.)
        let mut sources: Vec<(usize, BlockBuf)> = Vec::with_capacity(phase.source_levels.len());
        for &level in &phase.source_levels {
            let started = Instant::now();
            let stored = env.read(level, fetch)?;
            let read_ns = started.elapsed().as_nanos() as u64;
            // Priced exactly like the input read, through the same event, so a
            // run that reads two arrays per block reports two arrays' worth of
            // bytes. A second arm that cost nothing in the counters would make
            // every measurement of this feature a measurement of the wrong plan.
            events.emit(Event::RegionRead {
                source: format!("level {level}"),
                level,
                index: Some(task.index),
                region: fetch.clone(),
                voxels: fetch.voxels(),
                bytes: fetch.voxels() as u64 * decomposition.dtype_at(level).size_of() as u64,
                chunks: chunks_touched(fetch, &env.chunk_shape()),
                duration_ns: read_ns,
            });
            sources.push((level, stored));
        }
        for (&slot, place) in phase.slots.iter().zip(&places) {
            let started = Instant::now();
            let next = env.apply(slots[slot], &buf, &sources, place)?;
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
                    at: &place.input,
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
        // After the last slot, not after the first: a phase may fuse several
        // slots and any of them may hold a source leaf naming the same level.
        for (_, stored) in &sources {
            env.release(stored);
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

/// One iterative phase: every block, every substage, until nothing changes.
///
/// Returns the substage count and one outcome per block, in the phase's block
/// order, so the caller's bookkeeping — the tiling guard, the byte counts, the
/// level lifetime, the dependents — is the same code that runs for every other
/// kind of phase.
///
/// **The trivial form, on purpose.** Every block runs every substage, and the
/// convergence test is "did any block's core come out different from what it went
/// in as". No per-block skip, no dirty set, no frontier: those are the
/// optimisation, and a trivial executor is correct by design, which is what makes
/// it the oracle they will be tested against.
///
/// **What a distributed run would need, and does not have here.** One barrier per
/// substage, and the two private buffers shared across workers rather than owned
/// by this function. The blocks of a single substage are independent — they read
/// `current` and write disjoint cores of `next` — so the parallelism is available;
/// it is the substage boundary that is a real exchange point, because every block
/// of substage `k+1` reads cores its neighbours wrote at `k`. Left unbuilt, and
/// this is the sentence that says where it goes.
#[allow(clippy::too_many_arguments)]
fn run_iterative_phase(
    graph: &TaskGraph,
    phase_index: usize,
    decomposition: &Decomposition,
    op: &dyn IterativeOp,
    env: &dyn Environment,
    events: &Dispatch,
    n_phases: usize,
) -> Result<(usize, Vec<TaskOutcome>)> {
    let phase = &decomposition.phases[phase_index];
    let tasks = graph.tasks_in_phase(phase_index);
    let volume = phase.volume();
    let dtype = decomposition.dtype_at(phase_index);
    let bytes_per_voxel = dtype.size_of() as u64;
    let whole = Region::whole(&volume);
    let operands = op.operands();
    let running_at = operands
        .iter()
        .position(|operand| operand.operand == Operand::Running)
        .ok_or_else(|| {
            Error::InvalidArgument(format!(
                "iterative op {:?} declares no running operand; `check_iterative` refuses such \
                 an op when the plan is built and again when it is run",
                op.name()
            ))
        })?;

    // **Two private buffers, whatever the substage count turns out to be.** The
    // buffer written at substage `k` already holds substage `k-2`'s output, which
    // nothing will read again, so live storage is `O(1)` in the substage count —
    // and a forty-substage phase costs exactly what a two-substage one costs. They
    // are allocated through the environment so that residency is booked the way it
    // books it, and owned here rather than by the environment because nothing
    // outside this phase can see them: the plan allocates no level for them and
    // `Visibility` has nothing to say about them.
    //
    // The fill value is never read. Every block writes its core at substage 0 and
    // the cores tile the volume, so the whole buffer is real data from the first
    // substage onwards.
    let mut current = env.constant(dtype, &whole, f64::NAN)?;
    let mut next = env.constant(dtype, &whole, f64::NAN)?;

    // Everything that can fail runs inside here, so the two buffers are released
    // on the way out however it ends. A failed run's residency counters are not
    // load-bearing, but a partial release would be a number that quietly means
    // nothing.
    let produced = (|| -> Result<(usize, Vec<TaskOutcome>)> {
        let limit = op.limit().substages();
        let mut ran = 0usize;
        loop {
            if ran == limit {
                return Err(Error::InvalidArgument(format!(
                    "iterative op {:?} did not converge in {limit} substage(s) over a {volume:?} \
                     volume. Either the limit is below what this data needs — raise it, from \
                     whatever bound the op's own behaviour gives — or the iteration does not \
                     converge at all, which would be a defect in the step rather than in the \
                     data. The partially converged volume is deliberately not written: it is a \
                     plausible, well-formed, wrong answer.",
                    op.name()
                )));
            }
            let mut changed = false;
            for task in tasks {
                let fetch = &task.geometry.source;
                let read = &task.geometry.read;
                let at = Anchor::of_region(fetch, decomposition.volume_at(phase_index))?;
                // One buffer per declared operand, every one of them over the
                // block's read extent — see `iterate::Substage` for why they share
                // an extent rather than each getting its own.
                let mut buffers = Vec::with_capacity(operands.len());
                for operand in &operands {
                    // The running operand comes off the level only at substage 0;
                    // after that it is what the previous substage wrote, and it is
                    // the neighbours' *cores* of that which make the reach stay at
                    // one substage's worth. A fixed operand comes off the level
                    // every time, which is the whole point of declaring it.
                    let from_level = operand.operand == Operand::Fixed || ran == 0;
                    if from_level {
                        let started = Instant::now();
                        let buf = env.read(phase_index, fetch)?;
                        let read_ns = started.elapsed().as_nanos() as u64;
                        let chunks = chunks_touched(fetch, &env.chunk_shape());
                        events.emit(Event::RegionRead {
                            source: format!("level {phase_index}"),
                            level: phase_index,
                            index: Some(task.index),
                            region: fetch.clone(),
                            voxels: fetch.voxels(),
                            bytes: fetch.voxels() as u64 * bytes_per_voxel,
                            chunks,
                            duration_ns: read_ns,
                        });
                        events.emit(Event::BlockRead {
                            phase: phase_index,
                            index: task.index,
                            region: fetch.clone(),
                            voxels: fetch.voxels(),
                            chunks,
                        });
                        buffers.push(buf);
                    } else {
                        // A private buffer is not a level: no `RegionRead`, because
                        // nothing was fetched from storage. The residency is still
                        // booked, because the block is still resident.
                        buffers.push(env.slice(&current, &whole, fetch)?);
                    }
                }

                let result = env.apply_substage(op, ran, &buffers, &at)?;
                let valid = &task.geometry.valid;
                let core = env.slice(&result, read, valid)?;
                // The convergence test, and it is the whole predicate: did this
                // block's core come out different from what it went in as. Against
                // the *running* operand, because that is what the next substage
                // will be handed.
                let before = env.slice(&buffers[running_at], fetch, valid)?;
                match env.same(&core, &before) {
                    Some(false) => changed = true,
                    Some(true) => {}
                    None => {
                        return Err(Error::InvalidArgument(format!(
                            "iterative op {:?} cannot be run under an environment that holds no \
                             data: an iteration runs to convergence, and whether anything \
                             changed is a question about values. Simulating one needs a stated \
                             substage count, which is a different thing from the count a real \
                             run discovers and is deliberately not invented here.",
                            op.name()
                        )));
                    }
                }
                env.place(&mut next, &whole, valid, &core)?;

                env.release(&before);
                env.release(&core);
                env.release(&result);
                for buf in &buffers {
                    env.release(buf);
                }
            }
            // The exchange point. Everything below reads `current`, so the swap is
            // the barrier; see this function's header for the distributed shape.
            std::mem::swap(&mut current, &mut next);
            ran += 1;
            if !changed {
                break;
            }
        }

        // The fixed point, written out block by block. This is the only write the
        // phase makes to a level, which is what makes the substage count invisible
        // to everything downstream.
        let mut outcomes = Vec::with_capacity(tasks.len());
        let write_bytes_per_voxel = decomposition.dtype_at(phase_index + 1).size_of() as u64;
        for task in tasks {
            let valid = &task.geometry.valid;
            let piece = env.slice(&current, &whole, valid)?;
            let within = Region::new(&[0, 0, 0], &valid.shape);
            let started = Instant::now();
            env.write(phase_index + 1, &within, valid, &piece)?;
            let write_ns = started.elapsed().as_nanos() as u64;
            env.release(&piece);
            events.emit(Event::RegionWritten {
                sink: format!("level {}", phase_index + 1),
                level: phase_index + 1,
                index: Some(task.index),
                region: valid.clone(),
                voxels: valid.voxels(),
                bytes: valid.voxels() as u64 * write_bytes_per_voxel,
                chunks: chunks_touched(valid, &env.chunk_shape()),
                duration_ns: write_ns,
            });
            events.emit(Event::BlockWritten {
                phase: phase_index,
                index: task.index,
                valid: valid.clone(),
                materialised: phase_index + 1 < n_phases,
            });
            outcomes.push(TaskOutcome {
                valid: valid.clone(),
                short_circuited: false,
                // An iterative phase applies no slot of the chain, so it writes no
                // side output — the same position a fragment phase is in.
                side_written: Vec::new(),
                listener_faults: 0,
            });
        }
        Ok((ran, outcomes))
    })();

    env.release(&current);
    env.release(&next);
    produced
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

// ------------------------------------------------- choosing between paths --

/// The block edge a phase running `chain` over `volume` would be given under
/// `constraints`: the largest candidate that fits, cut only on the axes the
/// reach admits.
///
/// The same arithmetic [`Greedy`] does, extracted so that a decision taken
/// *before* the partition — which of several ways of computing one result to
/// run — can be priced at the block the run will actually use rather than at a
/// block nobody chose. It is the whole chain's reach rather than a phase's,
/// because the partition is not known yet; that over-states the halo, which is
/// the direction the cost model is stated to be safe in.
pub fn planned_block(
    chain: &Chain,
    volume: [usize; 3],
    dtype: Dtype,
    constraints: &Constraints,
) -> Result<[usize; 3]> {
    let reach = chain.reach_spec(volume)?;
    let mut candidates = constraints.block_candidates.clone();
    candidates.sort_unstable_by(|a, b| b.cmp(a));
    for edge in candidates {
        let axes = splittable_axes(&constraints.split_axes, &reach, volume);
        let Ok(grid) = BlockGrid::along(volume, &axes, edge) else {
            continue;
        };
        let cost = price_phase(
            &grid,
            &reach,
            chain.cost_per_voxel_in(grid.block()),
            1,
            false,
            dtype.size_of() as f64,
            &constraints.model,
            constraints.model.materialise_cost_per_voxel,
        );
        let fits = constraints.budget_bytes.is_none_or(|budget| {
            cost.working_set_bytes_per_block * constraints.expected_concurrency.max(1) as f64
                <= budget as f64
        });
        if fits {
            return Ok(grid.block());
        }
    }
    // Nothing fits, and saying so is `decompose`'s job with `decompose`'s
    // message. The whole volume is the block a caller pricing a branch should
    // use in the meantime: it is the one grid that always exists.
    Ok(volume)
}

/// Resolve every [`Chain::Alternative`] in `chain` to its **cheapest** branch,
/// priced at `block`.
///
/// **This is a planning decision, and it is the one planning decision that could
/// not be taken until now.** `Chain::Alternative` has always been the way to say
/// "two ways of computing the same thing" — its own documentation puts it as
/// *reach is budgeted for every branch, execution runs one* — but `taken` was
/// whatever the chain's author wrote, so a chain carrying a fast path and a
/// general one ran whichever was named. The declarations needed to choose
/// between them are [`BlockOp::cost_per_voxel`] and, for a path whose advantage
/// depends on the block it is given, [`BlockOp::cost_per_voxel_in`]. Both are
/// declared by the op; nothing here knows what any branch does.
///
/// **Why choosing is safe, in the two senses that matter.**
///
/// * *It cannot invalidate a plan.* Every fold a plan is built from treats an
///   `Alternative` as the max over branches — reach, cost, and therefore the
///   halo, the valid regions and the budget — so a plan built for one branch is
///   a plan for every branch. That is not a coincidence, it is what `taken`
///   being an index into equally-planned branches means.
/// * *It cannot change an answer.* Branches of an `Alternative` are candidates
///   for one level: [`Chain::produces`] already refuses branches that write
///   different element types, and [`Chain::placed_output_shape`] already refuses
///   branches that write different extents. That the values agree is the chain
///   author's claim, exactly as it was before anything chose between them — what
///   changes here is *which* path computes, never what a path computes.
///
/// **The ordering, stated because it is a compromise.** The branch is chosen
/// before the partition, and the partition is what fixes the block; so `block`
/// is the block the *whole chain* would be given ([`planned_block`]) rather than
/// the block the phase this slot lands in will get. The two differ only when the
/// partition changes the reach enough to change the candidate, and the direction
/// of the error is known: a whole-chain reach is at least a phase's, so the
/// block is at most the phase's, so a path whose advantage grows with the block
/// is under-sold. Under-selling the fast path is the safe mistake — it keeps the
/// general one, which is the branch that is always correct.
///
/// A nested `Alternative` inside a chosen branch is resolved too, and one inside
/// a branch that is not chosen is left as it was: it is not going to run, and
/// rewriting it would put a decision in the plan about work that will not happen.
pub fn choose_branches(chain: Chain, block: [usize; 3]) -> Chain {
    match chain {
        Chain::Alternative { branches, taken } => {
            let mut chosen = taken;
            let mut best = f64::INFINITY;
            for (index, branch) in branches.iter().enumerate() {
                let cost = branch.cost_per_voxel_in(block);
                // Strictly cheaper, so ties keep the lower index and the choice
                // is a function of the branches rather than of their order in a
                // sort. A planner that is not deterministic is not a planner.
                if cost < best {
                    best = cost;
                    chosen = index;
                }
            }
            let branches: Vec<Chain> = branches
                .into_iter()
                .enumerate()
                .map(|(index, branch)| {
                    if index == chosen {
                        choose_branches(branch, block)
                    } else {
                        branch
                    }
                })
                .collect();
            Chain::Alternative {
                branches,
                taken: chosen,
            }
        }
        Chain::Sequence(children) => Chain::Sequence(
            children
                .into_iter()
                .map(|child| choose_branches(child, block))
                .collect(),
        ),
        Chain::Parallel { branches, combine } => Chain::Parallel {
            branches: branches
                .into_iter()
                .map(|branch| choose_branches(branch, block))
                .collect(),
            combine,
        },
        leaf => leaf,
    }
}

/// [`choose_branches`] at the block [`planned_block`] derives, which is the call
/// a caller who has a `Constraints` and no reason to name a block wants.
///
/// It is a free function rather than a step inside [`Strategy::decompose`]
/// because the chain belongs to the `Workflow` and `decompose` is handed it by
/// reference: the decision is recorded *in the chain*, where `taken` lives and
/// where the executor reads it, rather than in the `Decomposition`, which
/// records op names and not implementations. A plan carrying a branch index
/// would be a plan whose meaning depended on a chain it does not hold.
pub fn choose_paths(
    chain: Chain,
    volume: [usize; 3],
    dtype: Dtype,
    constraints: &Constraints,
) -> Result<Chain> {
    let block = planned_block(&chain, volume, dtype, constraints)?;
    Ok(choose_branches(chain, block))
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
        let reach = workflow.chain.reach_spec(workflow.shape)?;
        let chain_reach = reach.bound(workflow.shape);
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
            chain_reach,
        };
        decomposition.declare_dtypes(&workflow.chain)?;
        decomposition.declare_source_levels(&workflow.chain)?;
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
            keep_levels: BTreeSet::new(),
        }
    }
}

/// Choose the cuts by the `O(n^2)` dynamic program, with the block size for
/// each phase chosen independently. The `2^(n-1)` enumeration is retained and
/// selectable — see [`PartitionSearch`].
///
/// Per-phase block sizes are separable given the partition — the total is
/// `sum over phases of n_blocks_p x cost_per_block_p` and the budget binds each
/// phase independently — so choosing them is an inner loop over candidates
/// rather than a `candidates^phases` product.
#[derive(Debug, Clone)]
pub struct Enumerating {
    pub concurrency: usize,
    pub priority: SchedulePriority,
    /// Which search over contiguous partitions picks the cuts. Both give the
    /// same plan; see [`PartitionSearch`].
    pub search: PartitionSearch,
}

impl Default for Enumerating {
    fn default() -> Self {
        Self {
            concurrency: 1,
            priority: SchedulePriority::PhaseMajor,
            search: PartitionSearch::default(),
        }
    }
}

/// How [`Enumerating`] searches the space of contiguous partitions.
///
/// The two answer the same question, and where the precondition below holds
/// they return the **same partition** — not merely one of equal cost, which
/// matters because a plan's phase boundaries are asserted all over this crate.
/// `tests/partition_search.rs` sweeps random chains asserting exact agreement.
///
/// # The precondition is additivity
///
/// The dynamic program is licensed by exactly one property, and it is worth
/// writing as an equation rather than carrying as an intuition:
///
/// ```text
/// cost(partition) = sum over its groups of price(group)
/// ```
///
/// with `price(j..i)` a function of that group's slots and nothing else. It
/// holds here, and each clause of it is checkable:
///
/// * [`price_phase`] is charged per phase and the totals are summed — the same
///   arithmetic [`crate::decomposition::predicted_cost`] re-does over a
///   finished plan, one phase at a time;
/// * [`crate::decomposition::summarise_slots`] and [`constraint_for`] fold over
///   the group alone, so the reach, the traversal preferences and the mandate
///   are the group's own;
/// * [`compute_per_voxel`] is asked at the grid *this* phase chose, and the
///   budget is checked against that phase's own working set — so the block edge
///   is an inner loop, not a coupling between phases;
/// * and the one term that looks positional — `is_materialised`, "this phase
///   writes an intermediate rather than the workflow's output" — is `i < n`. It
///   depends on where the group *ends* and on nothing about the other groups.
///
/// # What would break it
///
/// This list is the specification for the heuristics that would have to replace
/// the DP, and the reason the enumeration is kept rather than deleted. Any one
/// of these makes `price` depend on more than its own group, at which point
/// `Exhaustive` is the correct search and the DP is not:
///
/// 1. **A materialisation charge that knows its consumer.** Today an
///    intermediate costs `materialise_cost_per_voxel` per core voxel whoever
///    reads it. Charge the *reader's* halo — the redundant re-read of a
///    boundary — and the price of group `j..i` depends on group `i..k`.
/// 2. **A cost of changing the block edge between phases.** Per-phase edges are
///    free of each other now. A rechunk penalty, or a cache that only survives
///    a boundary when the grids agree, couples adjacent phases. (Recoverable:
///    carry the chosen edge in the DP state, at `O(B^2 n^2)`.)
/// 3. **A budget that is global rather than per phase.** `budget_bytes` binds
///    one phase's working set at a time. A budget over the intermediates *alive
///    at once*, or a cap on total intermediate storage, is a knapsack across
///    groups and no prefix cost summarises it.
/// 4. **Any non-linear function of the phase count.** A fixed per-phase
///    overhead is additive and fine; a term in `n_phases^2`, or a hard cap on
///    phases, is not — though a cap is recoverable by adding the count to the
///    DP state.
/// 5. **Data-dependent savings priced at plan time.** The empty-block short
///    circuit fires only when *every remaining op* declares
///    `constant_maps_to`, which is a property of the whole suffix. Pricing it
///    would make a group's cost depend on every group after it.
/// 6. **Cost that is not a sum over a linear chain at all** — a fan-in whose
///    branches must be cut consistently, or fusing non-adjacent ops. Then the
///    plan is not a 1-D contiguous partition and neither search here applies.
/// 7. **A per-phase quantity folded from the phases before it.** The planner
///    prices every phase at `workflow.dtype` and at `workflow.shape`; an op
///    that changes the element type or the volume is priced as if it had not.
///    Were that fixed by folding the declared types and shapes, the folded
///    value would still be a function of the cut point `j` alone, so the DP
///    would survive — but only because the fold runs over a *prefix*. A fold
///    that depended on where the earlier cuts fell would not.
///
/// Note what is *not* on this list. Forced barrier cuts, budget-infeasible
/// groups and mandate conflicts are all local facts about one group, so they
/// are edges the DP simply does not take.
///
/// # A separate fact, for whoever prunes the inner scan: `price` is **not**
/// monotone
///
/// The obvious way to make the DP near-linear is an early exit: for a fixed
/// right end `i`, scan `j` leftwards and stop once `price(j..i)` alone exceeds
/// the best candidate already found for `i` — sound if prices are non-negative
/// (so `best[j] + price >= price`) **and** `price(j..i)` never falls as the
/// group widens. The first half holds. The second does not, and the
/// counterexample is a configuration this crate already has a test for.
///
/// Three ops each reaching 200 voxels on a 512-voxel axis, `split_axes = [0]`,
/// one candidate edge of 64:
///
/// ```text
/// slots 0..1   reach 200   splittable [0]   8 blocks   redundancy  7.25    507904
/// slots 0..2   reach 400   splittable [0]   8 blocks   redundancy 13.5    1359872
/// slots 0..3   reach 600   splittable  []   1 block    redundancy  3.34     471040
/// ```
///
/// The third widening is **cheaper than either of the first two, and cheaper
/// than the single slot**. Nothing has gone wrong: at a folded reach of 600 the
/// dependency spans the axis, [`splittable_axes`] stops offering to cut it, the
/// grid collapses to one block, and the amplification falls from 13.5x to
/// 3.3x. No forced cut prevents this group either — `barrier_cuts` fires on
/// slots that are *individually* full-reach, and none of these three is.
///
/// So the sources of non-monotonicity, all of them the same shape — **widening
/// changed which grid was chosen**:
///
/// * **[`splittable_axes`] dropping an axis** once the folded reach spans it,
///   as above. This is the big one, and it can cut the price by a factor.
/// * **A mandating op joining the group.** Its extent replaces the candidate
///   list entirely ([`crate::op::BlockConstraint::lattice`]), so the new grid
///   bears no relation to the old one and the price may go either way.
/// * **[`compute_per_voxel`] at a changed block**, for an op whose cost has a
///   block extent in its denominator.
///
/// And two conditions the argument quietly assumes: every model coefficient and
/// every `cost_per_voxel` is non-negative (a negative one breaks both halves at
/// once), and the exit must compare **strictly** greater — the DP's key is the
/// lexicographic triple, so a wider group whose price merely *equals* the
/// running best can still win on the phase count or the cut mask.
///
/// What *is* safe: hold the grid fixed and the price is monotone. The folded
/// reach only grows ([`crate::reach::Reach::add`] sums per side, and `All`
/// absorbs), so redundancy only grows; `distinct_orders` only grows;
/// infeasibility is absorbing in this direction, because the working set only
/// grows and a mandate or space conflict cannot be undone by adding a member.
/// A budget that forces a *smaller* edge raises the total too, since the read is
/// `volume x prod((B + lo + hi) / B)` and that falls with `B`. And the concern
/// that a wider group swallows a phase boundary and its write does **not**
/// apply: `is_materialised` is `i < n`, so for a fixed `i` it is the same for
/// every `j`, and the saved boundary is priced in `best[j]`, outside this term.
///
/// An exact prune is therefore still available — it has to restart its bound
/// whenever the chosen grid changes, which happens at most a few times per axis
/// as the reach grows. That is design work, not a one-line guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartitionSearch {
    /// `best[i] = min over j < i of best[j] + price(j..i)`, at `O(n^2)` priced
    /// groups. The default.
    #[default]
    Dp,
    /// Every one of the `2^(n-1)` contiguous partitions, priced whole.
    ///
    /// Kept because the DP's licence is the additivity above, and a cost model
    /// that breaks it needs a search that never assumed it. It is also the
    /// oracle the DP is tested against.
    Exhaustive,
}

/// The longest chain the `O(n^2)` DP will plan.
///
/// What bounds it is not the search — `O(n^2)` priced groups at `n = 32` is 528
/// of them — but the **cut mask**, a `u32` in which bit `i` means "cut between
/// slot `i` and slot `i + 1`". `barrier_cuts`, `groups_for` and the tie-break
/// key all read that mask, so a chain needs `n - 1 <= 32` bits to state its own
/// boundaries. Widening the mask is the change that raises this, and it is a
/// mechanical one.
pub const MAX_SLOTS: usize = 32;

/// The longest chain [`PartitionSearch::Exhaustive`] will plan.
///
/// Enumeration is `2^(n-1)` partitions — 524288 at `n = 20` — and past this it
/// refuses rather than quietly taking minutes. This is the limit the DP exists
/// to lift, and it is the only thing `Exhaustive` costs.
pub const MAX_EXHAUSTIVE_SLOTS: usize = 20;

// -- one contiguous run of slots, folded, priced, and searched over ---------

/// One contiguous run of slots, folded a slot at a time.
///
/// The incremental fold is what keeps the DP at `O(n^2)` group prices: for a
/// fixed start the run grows by one slot per step, and the reach, the names and
/// the traversal preferences grow with it, so extending `j..i` to `j..i+1`
/// costs one slot's work rather than the run's.
///
/// It reproduces [`crate::decomposition::summarise_slots`] exactly, *including*
/// which slot a refused fold blames: that function stops at the first slot that
/// cannot join, and so does this, so every longer run from the same start
/// reports the same message. `tests/partition_search.rs` asserts the agreement
/// over every contiguous run of randomly generated chains.
#[derive(Debug, Clone)]
struct GroupFold {
    start: usize,
    end: usize,
    reach: Option<Reach>,
    names: Vec<String>,
    orders: Vec<[usize; 3]>,
    /// The first slot that could not join, in `summarise_slots`' own words.
    refusal: Option<String>,
}

impl GroupFold {
    fn new(start: usize) -> Self {
        Self {
            start,
            end: start,
            reach: None,
            names: Vec::new(),
            orders: Vec::new(),
            refusal: None,
        }
    }

    /// Take in the next slot, so the fold covers `start..end + 1`.
    fn extend(&mut self, slots: &[&Chain], volume: [usize; 3]) {
        let chain = slots[self.end];
        self.end += 1;
        self.names.push(chain.display_name());
        for order in chain.preferred_iterations() {
            if !self.orders.contains(&order) {
                self.orders.push(order);
            }
        }
        if self.refusal.is_some() {
            return;
        }
        // The first slot's space is the run's; see `summarise_slots` for why
        // this is not `Reach::none()` plus an addition.
        let stated = match chain.reach_spec(volume) {
            Ok(stated) => stated,
            Err(err) => {
                self.refusal = Some(err.to_string());
                return;
            }
        };
        self.reach = match self.reach.take() {
            None => Some(stated),
            Some(so_far) => match so_far.add(&stated) {
                Ok(folded) => Some(folded),
                Err(err) => {
                    self.refusal = Some(err.to_string());
                    return;
                }
            },
        };
    }
}

/// A contiguous run of slots, priced at the block edge it chose.
///
/// The derived [`PhaseDecomposition`] is deliberately **not** held. Deriving one
/// materialises a `BlockGeometry` per block, and the search prices `O(n^2)` runs
/// of which it keeps one partition's worth; the parts a phase is derived *from*
/// are a grid, two reaches and a list of names, so the search carries those and
/// derives once, for the plan it returns.
#[derive(Debug, Clone)]
struct PricedGroup {
    total: f64,
    reach: Reach,
    halo: Reach,
    names: Vec<String>,
    grid: BlockGrid,
}

impl PricedGroup {
    fn into_phase(self, start: usize, end: usize) -> PhaseDecomposition {
        PhaseDecomposition::derive(
            (start..end).collect(),
            self.names,
            self.reach,
            self.halo,
            self.grid,
        )
    }
}

/// What pricing one contiguous run came to.
#[derive(Debug, Clone)]
enum GroupPrice {
    Priced(PricedGroup),
    /// Not usable as a phase. The string, when there is one, is a reason that is
    /// **not** the budget — two ops that cannot share a block, a reach in two
    /// coordinate spaces, a mandate no grid meets — kept so a final refusal can
    /// name it instead of blaming a budget that was never the problem.
    ///
    /// It is carried beside the price rather than raised, because *which*
    /// refusal a caller reports depends on the order it visits runs in, and the
    /// two searches visit in different orders.
    Refused(Option<String>),
}

/// Everything needed to price one contiguous run of slots.
struct PhasePricer<'a> {
    slots: &'a [&'a Chain],
    volume: [usize; 3],
    bytes: f64,
    constraints: &'a Constraints,
}

impl PhasePricer<'_> {
    /// Price the run `fold` covers, choosing its block edge.
    ///
    /// `is_materialised` is the whole of this function's dependence on the rest
    /// of the partition, and it is `fold.end < slots.len()` — see
    /// [`PartitionSearch`] on why that is what makes the DP legitimate.
    fn price(&self, fold: &GroupFold, is_materialised: bool) -> GroupPrice {
        // Reaches in two coordinate spaces cannot be folded without a grid, so a
        // run containing both is infeasible *as a run* — the same shape of
        // answer a block-shape conflict gives, and it drops the partition rather
        // than the plan.
        if let Some(refusal) = &fold.refusal {
            return GroupPrice::Refused(Some(refusal.clone()));
        }
        let group: Vec<usize> = (fold.start..fold.end).collect();
        let reach = fold.reach.clone().unwrap_or_default();
        // What the ops in this run will accept. A conflict is a fact about
        // *this partition* — the same two ops in two phases are fine — so it
        // drops the partition and the search goes on.
        let mandated = match constraint_for(self.slots, &group, self.volume) {
            Ok(found) => found,
            Err(err) => return GroupPrice::Refused(Some(err.to_string())),
        };
        // The compute figure is re-asked per candidate rather than taken from
        // the fold, because an op may declare a term whose denominator is a
        // block extent — see `decomposition::compute_per_voxel`. For every op
        // that does not, this is the same number by the same route.
        let price = |grid: &BlockGrid| {
            price_phase(
                grid,
                &reach,
                compute_per_voxel(self.slots, &group, grid.block()),
                fold.orders.len(),
                is_materialised,
                self.bytes,
                &self.constraints.model,
                self.constraints.model.materialise_cost_per_voxel,
            )
        };
        let affordable = |cost: &super::decomposition::PhaseCost| {
            self.constraints.budget_bytes.is_none_or(|budget| {
                cost.working_set_bytes_per_block
                    * self.constraints.expected_concurrency.max(1) as f64
                    <= budget as f64
            })
        };
        let mut chosen: Option<(f64, usize, BlockGrid)> = None;
        // The halo this phase grants. Equal to the reach unless an op mandates
        // an input extent, in which case it is the per-block window that hands
        // every block that extent — see `BlockConstraint::lattice`.
        let mut halo = reach.clone();
        let mut note: Option<String> = None;
        if let Some(constraint) = &mandated {
            // A mandate replaces the candidate list rather than filtering it:
            // `block_candidates` is a list of scalar edges and a mandated shape
            // is anisotropic in general, so it is not expressible as a candidate
            // at all. The budget still binds — a block that does not fit does
            // not fit — but there is nothing to choose between.
            match constraint.lattice(self.volume, &reach) {
                Ok(Some((grid, window))) => {
                    let cost = price(&grid);
                    if affordable(&cost) {
                        halo = window;
                        chosen = Some((cost.cost_per_block * grid.n_blocks() as f64, 0, grid));
                    }
                }
                Ok(None) => {
                    note = Some(format!(
                        "the ops of phase {}..{} mandate {constraint:?}, which no block grid \
                         produces: a grid's cores are `index * block`, evenly strided and \
                         disjoint. A plan for it has to state each block's fetch region \
                         explicitly.",
                        fold.start, fold.end
                    ));
                }
                Err(err) => note = Some(err.to_string()),
            }
        } else {
            // NOTE: `cuttable_axes` — the reach-derived floor — is deliberately
            // **not** called here. See its own docs for what it would change and
            // why that is a decision rather than a fix.
            let axes = splittable_axes(&self.constraints.split_axes, &reach, self.volume);
            for &edge in &self.constraints.block_candidates {
                let grid = match BlockGrid::along(self.volume, &axes, edge) {
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
        let Some((total, _, grid)) = chosen else {
            return GroupPrice::Refused(note);
        };
        let priced = PricedGroup {
            total,
            reach,
            halo,
            names: fold.names.clone(),
            grid,
        };
        // The same check `execute` will run, run here: a mandated extent and a
        // non-zero reach are not jointly satisfiable, and the place to discover
        // that is the planner rather than the run.
        if let Some(constraint) = &mandated {
            let phase = priced.clone().into_phase(fold.start, fold.end);
            let label = format!("phase {}..{}", fold.start, fold.end);
            if let Err(err) = constraint.check(&phase.blocks, &label) {
                return GroupPrice::Refused(Some(err.to_string()));
            }
        }
        GroupPrice::Priced(priced)
    }
}

/// Every contiguous run of slots either search is allowed to make a phase of,
/// priced once.
///
/// This is the whole of the `O(2^n)` to `O(n^2)` change: the enumeration used to
/// re-price a run once per partition that contained it, and `n^2 / 2` runs is
/// the number of *distinct* things there are to price. The exhaustive search
/// reads the same table, so it is now `O(2^n)` table lookups rather than
/// `O(2^n)` pricings, and neither search can disagree with the other about what
/// a run costs.
struct PriceTable {
    /// `runs[start][end - start - 1]` is the run `start..end`. Rows are short
    /// where a forced cut stops them: a run may not span a barrier, and every
    /// longer run from the same start spans it too.
    runs: Vec<Vec<GroupPrice>>,
    n: usize,
}

impl PriceTable {
    fn build(pricer: &PhasePricer, forced_cuts: u32) -> Self {
        let n = pricer.slots.len();
        let mut runs = Vec::with_capacity(n);
        for start in 0..n {
            let mut row = Vec::new();
            let mut fold = GroupFold::new(start);
            for end in start + 1..=n {
                // Growing to `end` crosses the boundary between slots `end - 2`
                // and `end - 1`, which is cut mask bit `end - 2`. A forced cut
                // there ends the row: neither this run nor any longer one from
                // this start is a legal phase.
                if end >= start + 2 && forced_cuts & (1u32 << (end - 2)) != 0 {
                    break;
                }
                fold.extend(pricer.slots, pricer.volume);
                row.push(pricer.price(&fold, end < n));
            }
            runs.push(row);
        }
        Self { runs, n }
    }

    /// The price of `start..end`, or `None` where a forced cut forbids the run.
    fn get(&self, start: usize, end: usize) -> Option<&GroupPrice> {
        self.runs.get(start)?.get(end.checked_sub(start + 1)?)
    }

    /// How many priced runs were refused, out of how many were priced, and the
    /// first reason that was not the budget — in `(start, end)` order, which is
    /// the DP's own.
    fn refusals(&self) -> (usize, usize, Option<String>) {
        let mut refused = 0;
        let mut priced = 0;
        let mut note = None;
        for row in &self.runs {
            for entry in row {
                priced += 1;
                if let GroupPrice::Refused(reason) = entry {
                    refused += 1;
                    if note.is_none() {
                        note.clone_from(reason);
                    }
                }
            }
        }
        (refused, priced, note)
    }
}

/// `best[i] = min over j < i of best[j] + price(j..i)`, in `O(n^2)`.
///
/// **The tie-break is the enumeration's, reproduced rather than reinvented.**
/// The enumeration keeps the candidate minimising `(total, n_phases, mask)`
/// lexicographically — it iterates masks upwards and compares that triple — and
/// all three components are additive over the groups of a partition: the total
/// by construction, the phase count as `+1` per group, and the mask because the
/// cuts of `0..j` occupy bits `0..j-1` while the run `j..i` contributes exactly
/// bit `j-1`, so prefix bits and suffix bits are disjoint and lower. A
/// lexicographic order on an additive triple is translation-invariant, so
/// minimising it over prefixes minimises it over whole partitions, and the DP
/// picks the same partition — not merely one of the same cost.
///
/// **The sum is accumulated in the same order too.** `best[j] + price(j..i)`
/// adds groups left to right, exactly as the enumeration's `total += ...` does,
/// so for any one partition the two produce the bit-identical `f64`. The one
/// place they could still part is a floating-point one: two prefixes whose costs
/// differ by less than an ulp of the eventual total compare `<` here and `==`
/// there, and the enumeration would then break the tie on the phase count. No
/// generated chain has produced it; it is stated because it is the only known
/// gap.
fn search_dp(table: &PriceTable) -> Option<Vec<(usize, usize)>> {
    let n = table.n;
    // (cost, phases, cut mask, the cut this state came from)
    let mut best: Vec<Option<(f64, usize, u32, usize)>> = vec![None; n + 1];
    best[0] = Some((0.0, 0, 0, 0));
    for end in 1..=n {
        for start in 0..end {
            let Some((prefix_cost, prefix_phases, prefix_mask, _)) = best[start] else {
                continue;
            };
            let Some(GroupPrice::Priced(priced)) = table.get(start, end) else {
                continue;
            };
            let cost = prefix_cost + priced.total;
            let phases = prefix_phases + 1;
            let mask = prefix_mask | if start == 0 { 0 } else { 1u32 << (start - 1) };
            let better = match best[end] {
                None => true,
                Some((best_cost, best_phases, best_mask, _)) => {
                    (cost, phases, mask) < (best_cost, best_phases, best_mask)
                }
            };
            if better {
                best[end] = Some((cost, phases, mask, start));
            }
        }
    }
    best[n]?;
    let mut spans = Vec::new();
    let mut end = n;
    while end > 0 {
        let (_, _, _, start) = best[end].expect("a reachable state names a reachable predecessor");
        spans.push((start, end));
        end = start;
    }
    spans.reverse();
    Some(spans)
}

/// Every contiguous partition, priced whole and compared as a triple.
///
/// Unchanged in what it chooses: the mask order, the forced-cut filter, the
/// `(total, n_phases, mask)` comparison and the order refusals are recorded in
/// are all what they were. What moved is that a run's price now comes from
/// [`PriceTable`] instead of being recomputed for every partition containing it.
fn search_exhaustive(
    table: &PriceTable,
    forced_cuts: u32,
    note: &mut Option<String>,
    refusals: &mut usize,
) -> Option<Vec<(usize, usize)>> {
    let n = table.n;
    let mut best: Option<(f64, usize, u32, Vec<(usize, usize)>)> = None;
    for mask in 0u32..(1u32 << (n - 1)) {
        if mask & forced_cuts != forced_cuts {
            continue;
        }
        let mut total = 0.0_f64;
        let mut spans = Vec::new();
        let mut feasible = true;
        for group in groups_for(mask, n) {
            let start = group[0];
            let end = group[group.len() - 1] + 1;
            match table.get(start, end) {
                Some(GroupPrice::Priced(priced)) => {
                    total += priced.total;
                    spans.push((start, end));
                }
                Some(GroupPrice::Refused(reason)) => {
                    if let Some(reason) = reason {
                        note.get_or_insert_with(|| reason.clone());
                    }
                    feasible = false;
                    break;
                }
                // Unreachable while the mask honours the forced cuts, which the
                // filter above guarantees; treated as infeasible rather than
                // asserted, because a partition is a candidate and not a claim.
                None => {
                    feasible = false;
                    break;
                }
            }
        }
        if !feasible {
            *refusals += 1;
            continue;
        }
        let better = match &best {
            None => true,
            Some((best_cost, best_phases, best_mask, _)) => {
                (total, spans.len(), mask) < (*best_cost, *best_phases, *best_mask)
            }
        };
        if better {
            best = Some((total, spans.len(), mask, spans));
        }
    }
    best.map(|(_, _, _, spans)| spans)
}

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
        let limit = match self.search {
            PartitionSearch::Dp => MAX_SLOTS,
            PartitionSearch::Exhaustive => MAX_EXHAUSTIVE_SLOTS,
        };
        if slots.len() > limit {
            return Err(Error::InvalidArgument(match self.search {
                PartitionSearch::Dp => format!(
                    "enumerating: {} slots exceeds the {MAX_SLOTS} the O(n^2) DP admits. The \
                     search is not what bounds it — the cut mask a plan is chosen by is a \
                     `u32`, and a chain this long cannot state its own boundaries in one.",
                    slots.len()
                ),
                PartitionSearch::Exhaustive => format!(
                    "enumerating: {} slots exceeds the exhaustive search's limit of \
                     {MAX_EXHAUSTIVE_SLOTS}; this is the case docs/design/BLOCK_OPS.md reserves \
                     the O(n^2) DP for, and `PartitionSearch::Dp` — the default — is it",
                    slots.len()
                ),
            }));
        }
        if constraints.block_candidates.is_empty() {
            return Err(Error::InvalidArgument(
                "enumerating: no block size candidates".to_string(),
            ));
        }
        let volume = workflow.shape;
        // Cuts neither search is free to skip: a full-reach op is a planning
        // barrier, so it is its own phase whatever the cost model thinks. See
        // `is_planning_barrier`.
        let forced_cuts = barrier_cuts(&slots, volume);
        let pricer = PhasePricer {
            slots: &slots,
            volume,
            bytes: workflow.dtype.size_of() as f64,
            constraints,
        };
        let table = PriceTable::build(&pricer, forced_cuts);

        // Why a partition was dropped for a reason that is not the budget. Kept
        // so the final refusal can say "these two ops mandate different blocks"
        // rather than blaming a budget that was never the problem.
        let mut constraint_note: Option<String> = None;
        let mut budget_failures = 0usize;
        let spans = match self.search {
            PartitionSearch::Dp => search_dp(&table),
            PartitionSearch::Exhaustive => search_exhaustive(
                &table,
                forced_cuts,
                &mut constraint_note,
                &mut budget_failures,
            ),
        };
        let spans = spans.ok_or_else(|| {
            let reason = |note: &Option<String>| match note {
                None => String::new(),
                Some(note) => format!(
                    " At least one partition was dropped for a reason that is not the budget: \
                     {note}"
                ),
            };
            Error::InvalidArgument(match self.search {
                PartitionSearch::Exhaustive => format!(
                    "enumerating: none of the {} partitions fits the {:?} byte budget at \
                     concurrency {} with block candidates {:?}. Reduce the concurrency, add a \
                     smaller block candidate, or raise the budget.{}",
                    budget_failures,
                    constraints.budget_bytes,
                    constraints.expected_concurrency,
                    constraints.block_candidates,
                    reason(&constraint_note),
                ),
                PartitionSearch::Dp => {
                    let (refused, priced, note) = table.refusals();
                    format!(
                        "enumerating: no partition of the {} slots fits the {:?} byte budget at \
                         concurrency {} with block candidates {:?} — {refused} of the {priced} \
                         contiguous slot runs the O(n^2) search priced could not be a phase. \
                         Reduce the concurrency, add a smaller block candidate, or raise the \
                         budget.{}",
                        slots.len(),
                        constraints.budget_bytes,
                        constraints.expected_concurrency,
                        constraints.block_candidates,
                        reason(&note),
                    )
                }
            })
        })?;
        let phases: Vec<PhaseDecomposition> = spans
            .into_iter()
            .map(|(start, end)| match table.get(start, end) {
                Some(GroupPrice::Priced(priced)) => priced.clone().into_phase(start, end),
                _ => unreachable!("the search returns only runs it priced"),
            })
            .collect();

        let mut decomposition = Decomposition {
            volume,
            dtype: workflow.dtype,
            phases,
            chain_reach: workflow.chain.reach3(&volume),
        };
        decomposition.declare_dtypes(&workflow.chain)?;
        decomposition.declare_source_levels(&workflow.chain)?;
        decomposition.check()?;
        Ok(decomposition)
    }

    fn hints(&self, workflow: &Workflow, decomposition: &Decomposition) -> Hints {
        Hints {
            visit_order: consensus_order(workflow, decomposition),
            priority: self.priority,
            concurrency: self.concurrency,
            prefetch_depth: 1,
            // Nothing kept: a strategy advising on speed has no reason to want
            // an intermediate afterwards. A caller who does overrides it.
            keep_levels: BTreeSet::new(),
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
            // `compute` is dropped here and re-asked per candidate grid; see
            // `decomposition::compute_per_voxel`.
            let (reach, _compute, names, orders) =
                super::decomposition::summarise_slots(&slots, group, volume)?;
            let is_materialised = position + 1 < groups.len();
            let mandated = constraint_for(&slots, group, volume)?;
            let mut grid = None;
            let mut halo = reach.clone();
            if let Some(constraint) = &mandated {
                // Mandated, so there is nothing to choose between; the budget
                // still binds. See `Enumerating` for why the candidate list is
                // replaced rather than filtered.
                let (candidate, window) = constraint.lattice(volume, &reach)?.ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "greedy: phase {position} mandates {constraint:?}, which no block grid \
                         produces — a grid's cores are `index * block`, evenly strided and \
                         disjoint. A plan for it has to state each block's fetch region \
                         explicitly, which this strategy does not do."
                    ))
                })?;
                halo = window;
                let cost = price_phase(
                    &candidate,
                    &reach,
                    compute_per_voxel(&slots, group, candidate.block()),
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
                // As in `Enumerating`: the reach-derived floor is not wired.
                let axes = splittable_axes(&constraints.split_axes, &reach, volume);
                for edge in candidates {
                    let Ok(candidate) = BlockGrid::along(volume, &axes, edge) else {
                        continue;
                    };
                    let cost = price_phase(
                        &candidate,
                        &reach,
                        compute_per_voxel(&slots, group, candidate.block()),
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
                     {position} (reach {reach})",
                    constraints.block_candidates, constraints.budget_bytes
                ))
            })?;
            let phase = PhaseDecomposition::derive(group.clone(), names, reach, halo, grid);
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
        decomposition.declare_source_levels(&workflow.chain)?;
        decomposition.check()?;
        Ok(decomposition)
    }

    fn hints(&self, workflow: &Workflow, decomposition: &Decomposition) -> Hints {
        Hints {
            visit_order: consensus_order(workflow, decomposition),
            priority: SchedulePriority::BlockMajor,
            concurrency: self.concurrency,
            prefetch_depth: 2,
            keep_levels: BTreeSet::new(),
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

#[cfg(test)]
mod block_floor_tests {
    use super::*;
    use crate::decomposition::{cuttable_axes, splittable_axes, CostModel};
    use crate::env::ArrayEnvironment;
    use crate::probes::WindowSumOp;
    use crate::reach::Reach;
    use crate::voxels::Voxels;

    /// The measured case, at the numbers it was measured at: a 716-offset
    /// element whose reach is 15 on a `[24, 20, 16]` volume.
    const VOLUME: [usize; 3] = [24, 20, 16];
    const RADIUS: [usize; 3] = [15, 15, 0];

    fn ramp(shape: [usize; 3]) -> Voxels {
        let n = shape[0] * shape[1] * shape[2];
        let data: Vec<f64> = (0..n).map(|i| (i % 37) as f64).collect();
        ndarray::Array3::from_shape_vec(shape, data).unwrap().into()
    }

    /// Total voxels fetched over the run, divided by the voxels in the volume.
    ///
    /// The number `forme.md` names as the one that justifies the floor: a
    /// small-reach stencil sits near `1.0`, and the distance from `1.0` *is* the
    /// cost of the decomposition.
    fn amplification(plan: &Decomposition) -> f64 {
        let read: usize = plan.exact_read_voxels().iter().sum();
        let volume: usize = VOLUME.iter().product();
        read as f64 / volume as f64
    }

    /// **What the floor is a floor against**, stated as arithmetic rather than
    /// as a claim: without it, the axes whose reach covers the volume are cut
    /// anyway, and every one of the resulting blocks reads all of them.
    #[test]
    fn cutting_an_axis_the_reach_already_spans_multiplies_the_read_and_saves_nothing() {
        let reach = Reach::from(RADIUS);
        let unfloored = splittable_axes(&[0, 1, 2], &reach, VOLUME);
        assert_eq!(unfloored, vec![0, 1, 2], "the old rule cuts all three");
        let grid = BlockGrid::along(VOLUME, &unfloored, 3).unwrap();
        assert_eq!(grid.n_blocks(), 336, "the plan that ran past 15 minutes");
        for axis in 0..2 {
            let (lo, hi) = reach.axis(axis).bound(VOLUME[axis]);
            assert!(
                3 + lo + hi >= VOLUME[axis],
                "axis {axis}: the read is clamped to the whole axis whatever the block edge is"
            );
        }

        // The floor keeps the one axis where a cut narrows the read and drops
        // the two where it does not. Not a refusal: a plan is still offered.
        let floored = cuttable_axes(&[0, 1, 2], &reach, VOLUME, 3);
        assert_eq!(floored, vec![2]);
        assert_eq!(
            BlockGrid::along(VOLUME, &floored, 3).unwrap().block(),
            [24, 20, 3]
        );
    }

    /// What the floor would buy, measured on the two plans it chooses between.
    ///
    /// **Not asserted through a planner**, because the floor is not wired into
    /// one; see `cuttable_axes`. What is asserted is the arithmetic the decision
    /// would rest on, so that whoever takes that decision has the number rather
    /// than the argument: 336 blocks reading **48.9x** the volume, against six
    /// reading 1.125x of it, for the same answer.
    ///
    /// The figure is `forme.md`'s *touched voxels / volume voxels* — the
    /// distance from `1.0` being the cost of the decomposition — taken from
    /// `Decomposition::exact_read_voxels`, which is the clamped geometry rather
    /// than the model's infinite-grid redundancy. 48.9 rather than 336 because
    /// axis 2's reach is zero, so the one axis a cut does narrow is narrowed.
    #[test]
    fn the_floor_would_take_the_amplification_from_49_to_1_125() {
        let reach = Reach::from(RADIUS);
        let phase = |axes: &[usize], edge: usize| {
            let plan = Decomposition {
                volume: VOLUME,
                dtype: Dtype::F64,
                phases: vec![PhaseDecomposition::derive(
                    vec![0],
                    vec!["w".to_string()],
                    reach.clone(),
                    reach.clone(),
                    BlockGrid::along(VOLUME, axes, edge).unwrap(),
                )],
                chain_reach: RADIUS,
            };
            (plan.phases[0].grid.n_blocks(), amplification(&plan))
        };

        let (blocks, amplified) = phase(&splittable_axes(&[0, 1, 2], &reach, VOLUME), 3);
        assert_eq!(blocks, 336);
        assert!(
            (amplified - 48.9375).abs() < 1e-9,
            "the number the decision rests on moved: {amplified}"
        );

        let (blocks, amplified) = phase(&cuttable_axes(&[0, 1, 2], &reach, VOLUME, 3), 3);
        assert_eq!(blocks, 6);
        assert!(amplified < 1.2, "{amplified}");
    }

    /// **And it changes no voxel.** The floored plan, the unfloored one it
    /// replaced, and the single-block oracle all produce the same volume, byte
    /// for byte. A block grid is how the answer is cut up, never what it is —
    /// which is what makes the floor a cost decision that is safe to take.
    #[test]
    fn the_floor_moves_the_grid_and_not_the_answer() {
        let workflow = Workflow::new(Chain::op(WindowSumOp::new("w", RADIUS)), VOLUME, Dtype::F64);
        let input = ramp(VOLUME);

        let run = |plan: &Decomposition| -> Voxels {
            let env = ArrayEnvironment::for_decomposition(input.clone(), plan, [4, 4, 4]).unwrap();
            execute("floor", &workflow, plan, &Hints::default(), &env).unwrap();
            env.level(plan.n_phases())
        };

        let oracle = run(&Trivial
            .decompose(&workflow, &Constraints::default())
            .unwrap());

        let reach = Reach::from(RADIUS);
        let plan = |axes: &[usize]| Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases: vec![PhaseDecomposition::derive(
                vec![0],
                vec!["w".to_string()],
                reach.clone(),
                reach.clone(),
                BlockGrid::along(VOLUME, axes, 3).unwrap(),
            )],
            chain_reach: RADIUS,
        };
        assert_eq!(
            run(&plan(&splittable_axes(&[0, 1, 2], &reach, VOLUME))),
            oracle,
            "the 336-block plan the floor would remove"
        );
        assert_eq!(
            run(&plan(&cuttable_axes(&[0, 1, 2], &reach, VOLUME, 3))),
            oracle,
            "the plan the floor would leave"
        );
    }

    /// It cannot make a plan infeasible: the resident set on a dropped axis was
    /// already clamped to the volume, so the budget sees the same number.
    #[test]
    fn the_floor_never_turns_an_affordable_plan_into_an_unaffordable_one() {
        let workflow = Workflow::new(Chain::op(WindowSumOp::new("w", RADIUS)), VOLUME, Dtype::F64);
        let reach = workflow.chain.reach_spec(VOLUME).unwrap();
        for edge in [1usize, 2, 3, 4, 8, 16, 32] {
            let before =
                BlockGrid::along(VOLUME, &splittable_axes(&[0, 1, 2], &reach, VOLUME), edge)
                    .unwrap();
            let after = BlockGrid::along(
                VOLUME,
                &cuttable_axes(&[0, 1, 2], &reach, VOLUME, edge),
                edge,
            )
            .unwrap();
            let price = |grid: &BlockGrid| {
                price_phase(grid, &reach, 1.0, 1, false, 8.0, &CostModel::default(), 1.0)
                    .working_set_bytes_per_block
            };
            assert!(
                price(&after) <= price(&before),
                "edge {edge}: the floor raised the resident set"
            );
        }
    }
}

#[cfg(test)]
mod fold_tests {
    use super::*;
    use crate::decomposition::summarise_slots;
    use crate::op::BlockOp;
    use crate::probes::{IdentityOp, WindowSumOp};
    use crate::reach::{AxisReach, Space};
    use crate::voxels::Voxels;

    /// An op that states whatever reach the test wants, including forms the
    /// shipped probes cannot: one-sided, whole-axis, and in another coordinate
    /// space.
    struct Stated {
        name: &'static str,
        reach: Reach,
    }

    impl BlockOp for Stated {
        fn name(&self) -> &'static str {
            self.name
        }

        /// Derived from the full statement, because `Chain::reach_spec` checks
        /// that this is a bound on it rather than trusting that it is.
        fn reach(&self, axis: usize, volume_len: usize) -> usize {
            match self.reach.axis(axis) {
                AxisReach::Bounded { lo, hi } => *lo.max(hi),
                _ => volume_len,
            }
        }

        fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
            self.reach.clone()
        }

        fn accepts(&self, _dtype: Dtype) -> bool {
            true
        }

        fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
            out.assign(input)
        }
    }

    /// [`GroupFold`] is [`summarise_slots`], incrementally — including which
    /// slot it blames.
    ///
    /// This assertion is not redundant with `tests/partition_search.rs`. Both
    /// searches read the fold, so a fold that diverged from `summarise_slots`
    /// would move the DP's plan and the enumeration's plan *together* and the
    /// agreement sweep would see nothing. This is the check that the refactor
    /// kept the enumeration's own answer, rather than only that the two searches
    /// still agree with each other.
    #[test]
    fn folding_a_run_of_slots_matches_summarising_it() {
        let volume = [32, 16, 8];
        let chain = Chain::sequence(vec![
            Chain::op(IdentityOp::new("a", [2, 0, 0]).with_order([0, 1, 2])),
            Chain::op(WindowSumOp::new("b", [1, 1, 0])),
            Chain::op(Stated {
                name: "c",
                reach: Reach::asymmetric([(3, 1), (0, 2), (0, 0)]),
            }),
            // A reach in another space: it cannot be folded with the ones above
            // it, so every run that spans it is refused — by both, in the same
            // words.
            Chain::op(Stated {
                name: "d",
                reach: Reach::none().in_space(Space::source_index()),
            }),
            Chain::op(IdentityOp::new("e", [0, 0, 4]).with_order([2, 1, 0])),
            Chain::op(Stated {
                name: "f",
                reach: Reach::per_axis([AxisReach::All, AxisReach::none(), AxisReach::none()]),
            }),
        ]);
        let slots = chain.slots();
        let mut refusals = 0;
        for start in 0..slots.len() {
            let mut fold = GroupFold::new(start);
            for end in start + 1..=slots.len() {
                fold.extend(&slots, volume);
                let group: Vec<usize> = (start..end).collect();
                match summarise_slots(&slots, &group, volume) {
                    Ok((reach, _compute, names, orders)) => {
                        assert!(fold.refusal.is_none(), "{start}..{end}: {:?}", fold.refusal);
                        assert_eq!(
                            fold.reach.clone().unwrap_or_default(),
                            reach,
                            "{start}..{end}"
                        );
                        assert_eq!(fold.names, names, "{start}..{end}");
                        assert_eq!(fold.orders, orders, "{start}..{end}");
                    }
                    Err(err) => {
                        refusals += 1;
                        assert_eq!(
                            fold.refusal.as_deref(),
                            Some(err.to_string().as_str()),
                            "{start}..{end}: the fold and the summary must refuse alike"
                        );
                    }
                }
            }
        }
        assert!(refusals >= 6, "the refusing runs were not exercised");
    }
}
