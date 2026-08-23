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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use rayon::prelude::*;

use crate::assemble::ImageId;
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::region::Region;
use crate::tiling::boxes_tile_exactly;
use crate::voxels::Voxels;

use super::decomposition::{
    check_block_constraints, check_dtypes, check_output_shapes, check_source_images,
    compute_per_voxel, constraint_for, cuttable_axes, groups_for, is_planning_barrier, price_phase,
    region_to_ranges, Constraints, Decomposition, PhaseDecomposition, SlabPolicy, Visibility,
};
use super::env::{block_shape, BlockBuf, Environment};
use super::fragment::{
    check_phase_work, neighbourhood, BlockOutput, BlockView, Coverage, PhaseWork, SeamFold,
    SourceBlocks,
};
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
    /// The primary's shape is **image 0's**, which is the output's too unless a
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
    /// Internal images to keep rather than free when their reader finishes.
    ///
    /// **Advisory, and it belongs here rather than in the plan for one reason:
    /// keeping an intermediate cannot change a voxel.** It changes what is on
    /// disk when the run ends, which is a debugging decision, so it sits with
    /// every other value a strategy may get wrong at no cost to the answer.
    ///
    /// Empty means "free every internal image as soon as its reader is done",
    /// which is the behaviour worth having by default: an `N`-phase chain then
    /// holds three images at once instead of `N + 1`. Naming an image here is how
    /// a caller says "I want to look at that one" — the same choice the sidecar
    /// store spells `Lifecycle::Persistent`.
    ///
    /// Naming image 0 or the output image is harmless and does nothing; neither
    /// is ever freed.
    pub keep_images: BTreeSet<ImageId>,
    /// Whether a block may be cut into slabs and run on several threads, and by
    /// what rule. See [`SlabPolicy`].
    ///
    /// **Advisory, and it belongs here for a stronger reason than the rest of
    /// this struct has.** Everything in `Hints` may be ignored at no cost to the
    /// answer; a slab count is the one where that is not a convention but a
    /// property somebody checks. `slab::apply_sliced`'s acceptance bar is
    /// bit-identity against the uncut block at every slab count, held by
    /// `tests/intra_block_slicing.rs`, so an executor that ignored this field
    /// entirely would produce the same bytes more slowly — which is exactly what
    /// "advisory" is supposed to mean and rarely gets to mean this literally.
    ///
    /// **What it is not: a second statement of the block count.** The rule reads
    /// `concurrency` above and the plan's own block count and answers `1`
    /// whenever the lattice already has work for every worker, which is the
    /// common case and the case this feature must not touch. See
    /// [`SlabPolicy::slabs_for`].
    ///
    /// **How a caller's [`Constraints::slab_policy`] reaches it.**
    /// [`Strategy::plan`] copies it, because that is the one method that holds
    /// the constraints and the hints at once. [`Strategy::run`] is handed a
    /// decomposition and no constraints, so it gets whatever the strategy's own
    /// [`Strategy::hints`] advises — the default. A caller who wants a run
    /// switched off takes the plan, or overrides the field.
    pub slab_policy: SlabPolicy,
}

impl Default for Hints {
    fn default() -> Self {
        Self {
            visit_order: None,
            priority: SchedulePriority::PhaseMajor,
            concurrency: 1,
            prefetch_depth: 0,
            keep_images: BTreeSet::new(),
            slab_policy: SlabPolicy::default(),
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

    /// The plan: the binding half and the advisory half, built together.
    ///
    /// **The one line that carries [`Constraints::slab_policy`] into the run**,
    /// and it is here rather than in [`Self::hints`] because this is the only
    /// method that holds a `Constraints` and a `Hints` at the same time. A
    /// strategy advises a slab policy the way it advises a concurrency — from
    /// its own fields — and a caller who has stated one overrides that advice,
    /// which is the same direction every other constraint travels.
    fn plan(&self, workflow: &Workflow, constraints: &Constraints) -> Result<Plan> {
        let decomposition = self.decompose(workflow, constraints)?;
        let hints = Hints {
            slab_policy: constraints.slab_policy,
            ..self.hints(workflow, &decomposition)
        };
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

/// [`execute`], with the run's **CPU-seconds against its wall-seconds**.
///
/// **The question wall time cannot answer.** A run that takes ten seconds on a
/// pool of forty tells a caller nothing about whether it used the forty;
/// `CpuLedger::mean_cores_busy` is the mean number of cores actually kept busy,
/// and against `Hints::concurrency` it is the acceptance bar for every threading
/// change this crate makes — including whether a block is worth cutting into
/// slabs at all, which is only ever worth it when this figure is *below* the
/// pool.
///
/// **A wrapper rather than a change to `execute`, and the two readings are at
/// the run boundary rather than per phase.** Two reasons, both about not
/// perturbing the thing being measured. The instrument costs one small file read
/// and two integer parses — nothing at a boundary, far too much on a block — and
/// `crate::cpu`'s header records what happened the last time this project
/// allocated inside a timed region. And a per-phase figure needs a phase
/// *finish*, which this executor's task graph does not have a single point for;
/// inventing one to hang an instrument on would be changing the schedule to
/// measure it. Per-phase is the natural next step and it needs that boundary
/// first.
///
/// The ledger reads zero ticks where `/proc` is absent, which
/// [`crate::cpu::CpuTime::now`] reports as `None` and this folds to a recorded
/// zero — so a caller on such a machine sees `mean_cores_busy` of zero rather
/// than a plausible wrong number.
pub fn execute_accounted(
    strategy: &'static str,
    workflow: &Workflow,
    decomposition: &Decomposition,
    hints: &Hints,
    env: &dyn Environment,
) -> Result<(Stats, crate::cpu::CpuLedger)> {
    let before = crate::cpu::CpuTime::now();
    let started = std::time::Instant::now();
    let stats = execute_observed(strategy, workflow, decomposition, hints, env, &[])?;
    // **Both readings taken before anything is formatted or allocated.** The
    // ledger's `record` is two atomic adds.
    let elapsed = started.elapsed();
    let after = crate::cpu::CpuTime::now();
    let ledger = crate::cpu::CpuLedger::new();
    let ticks = match (before, after) {
        (Some(before), Some(after)) => after.since(before),
        _ => 0,
    };
    ledger.record(ticks, elapsed.as_nanos().min(u128::from(u64::MAX)) as u64);
    Ok((stats, ledger))
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
    // Image 0 only. What the *later* images are shaped like is the phases' own
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
    // the chain, and re-checked here because a plan whose images are the wrong
    // width would otherwise be discovered one block at a time.
    check_dtypes(&workflow.chain, decomposition, work)?;
    // And the same arrangement for the extent. It is not the per-block
    // comparison `run_task` makes — that one an op may answer out of the plan,
    // by taking `Placement::writes` — it is the same question at whole-volume
    // scale with the plan's answer withheld, which is the one form of it no op
    // can answer from the plan. See `check_output_shapes`.
    check_output_shapes(&workflow.chain, decomposition, work)?;
    // And the same arrangement for the images a phase reads besides its own
    // input. **Before `prepare` and before the graph**: a forward reference is
    // a plan that is not a plan, and it is refused by name here rather than
    // becoming a missing dependency edge or a block asking for an image nothing
    // has written.
    check_source_images(&workflow.chain, decomposition)?;
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
    // A **barrier** phase is held back by the same mechanism, and here it is
    // holding the property up rather than merely tidying it away.
    //
    // **This is the only thing enforcing a barrier in this executor**, and that
    // is said plainly because it briefly was not: while a barrier was
    // `blocks x blocks` edges in the graph, the indegree above enforced it too,
    // and a correctness property enforced in one place that reads as though it
    // is enforced in two is worse than one honestly enforced once. The edges are
    // gone (`TaskGraph::barriers` has the measurement that removed them) and the
    // gate is here.
    //
    // A barrier phase's own `deps` are the ordinary region-derived ones — its
    // blocks fetch their own cores — so they go to zero long before the phase
    // below has finished. The condition below is therefore **stated rather than
    // derived**: every earlier phase's remaining count is zero. An earlier
    // version released when every task of the phase had become ready, which is
    // the same moment in any plan whose valid regions tile, but it arrived there
    // through an invariant proved elsewhere instead of through the sentence the
    // barrier is.
    //
    // **Earlier phases and not only `p-1`, and the difference is unobservable.**
    // A `FragmentInput` may name a stream written further back, so the sentence
    // has to be true of that stream too — but in any plan that passed
    // `Decomposition::check` the two conditions are the same moment, because
    // every task has a non-empty valid region and those regions tile, so every
    // task of phase `q` is a dependency of some task of `q+1` and all of `p-1`
    // done implies all of `p-2` done. Mutating this to `p-1` alone was tried
    // against a three-phase plan whose barrier reduces over phase 0's stream and
    // **nothing failed**, which is the induction working rather than a hole in
    // the test.
    //
    // It is written broadly anyway, because the cost is one comparison per phase
    // per wave and what it buys is that this line *says* the property instead of
    // depending on a proof that lives in another file. If the tiling invariant
    // is ever relaxed, this does not quietly become wrong.
    //
    // **This cannot deadlock**, on the argument already written above for an
    // iterative phase and unchanged by this: a task of phase `p` waits only on
    // earlier phases, so holding phase `p`'s tasks back blocks nothing that
    // phase `p`'s tasks need.
    let barrier: Vec<bool> = decomposition
        .phases
        .iter()
        .map(|phase| phase.barrier)
        .collect();
    let mut barrier_held: Vec<Vec<usize>> = vec![Vec::new(); n_phases];
    let mut barrier_released: Vec<bool> = vec![false; n_phases];
    // One blob per phase, empty for every phase whose op takes the default
    // `reduce` and for every phase that is not a barrier.
    let mut reduced: Vec<Vec<u8>> = vec![Vec::new(); n_phases];
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
                &barrier,
                &mut ready,
                &mut iterative_ready,
                &mut barrier_held,
                &barrier_released,
            );
        }
    }
    // How many substages each phase ran. Zero for every phase that is not an
    // iteration, which is what a phase with no loop in it took.
    let mut substages: Vec<usize> = vec![0; n_phases];
    // And how much each of them changed. Empty for the same phases, and for a
    // phase whose environment holds no values to difference; see `Stats`.
    let mut substage_changes: Vec<Vec<u64>> = vec![Vec::new(); n_phases];
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
        // **The gate, and the one thing that goes in the gap it creates.**
        // Every earlier phase has finished and no block of this phase has
        // started. That is the only moment at which the fragment set is complete
        // and untouched, so it is where `FragmentOp::reduce` runs: once, for the
        // phase, rather than once per block from inside `apply` where the op
        // would have nowhere to put the answer and would re-derive it `blocks`
        // times.
        //
        // Then the held tasks join the ready heap like any others. They are
        // ranked by the same priority key, popped in the same waves and run
        // concurrently with each other and with whatever else is ready: a
        // barrier constrains when a phase may *start*, not how it runs. Anything
        // admitted after this point goes straight to the heap, because `admit`
        // consults `barrier_released`.
        for phase in 0..n_phases {
            if !barrier[phase]
                || barrier_released[phase]
                || (0..phase).any(|earlier| phase_remaining[earlier] > 0)
            {
                continue;
            }
            reduced[phase] = reduce_phase(decomposition, phase, work, env)?;
            barrier_released[phase] = true;
            for id in std::mem::take(&mut barrier_held[phase]) {
                ready.push(Reverse((priority_key(&graph.tasks[id], hints), id)));
            }
        }
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
                let (ran, changes, outcomes) = run_iterative_phase(
                    &graph,
                    phase,
                    decomposition,
                    *op,
                    env,
                    &events,
                    n_phases,
                    pool.as_deref(),
                )?;
                substages[phase] = ran;
                substage_changes[phase] = changes;
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
                    let phase = graph.tasks[id].phase;
                    // **The planner's rule, evaluated per phase and nowhere
                    // else.** `n_blocks` is the phase's own lattice, from the
                    // plan rather than from how many tasks happen to be ready:
                    // the question the rule answers is "does this lattice leave
                    // workers parked", which is a property of the plan and must
                    // give the same answer whatever order the heap emptied in.
                    // On a plan with work for every worker this is `1` and the
                    // block runs exactly as it did before slabs existed.
                    //
                    // **One budget, not two, and the arithmetic is why.** The
                    // design note's §10 flags that threads spawned inside a task
                    // are not the pool's, so `n_blocks x slabs` could exceed the
                    // machine with nothing noticing. It cannot here: the wave is
                    // at most `min(concurrency, ready)` tasks, a phase of `n`
                    // blocks contributes at most `n` ready tasks, and each is
                    // allotted `floor(concurrency / n)` slabs — so one phase's
                    // wave spends at most `concurrency` threads. Two phases
                    // cannot both be one-block and both ready, because a
                    // one-block phase's task depends on the whole of the phase
                    // below it. The bound is the plan's shape rather than a
                    // clamp, which is why there is no clamp.
                    let slabs = hints
                        .slab_policy
                        .slabs_for(concurrency, decomposition.phases[phase].blocks.len());
                    run_task(
                        &graph.tasks[id],
                        decomposition,
                        &slots,
                        &work[phase],
                        env,
                        &events,
                        n_phases,
                        &reduced[phase],
                        slabs,
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
                // Every image whose **last** reader is this phase is now dead.
                //
                // This used to be the single image `phase` reads, on the
                // argument that exactly one phase reads an image. A source leaf
                // makes that a special case: an image read by a later phase has a
                // second reader, and freeing it here would free something still
                // wanted. `images_dead_after` is the general statement — an image
                // dies after its last reader — and it answers `[phase]` for
                // every plan with no source leaf, so this is the same behaviour
                // stated in a way that stays true when there are two.
                //
                // The saving is the whole point of `Visibility`: without this an
                // `N`-phase chain holds `N + 1` full images for the length of
                // the run, and only ever two of them are live.
                for image in decomposition.images_dead_after(phase) {
                    if decomposition.image_visibility(image) == Visibility::Internal
                        && !hints.keep_images.contains(&ImageId::from(image))
                    {
                        // The phase goes with the image, so that a reader who
                        // wanted it back is told which `keep_images` entry they
                        // needed rather than only that it is gone.
                        env.discard_image_after(image, phase)?;
                    }
                }
                if work[phase].writes_an_image() {
                    // A phase that wrote no image has nothing to flush and
                    // nothing to materialise, and saying otherwise would put a
                    // byte count on the stream for bytes never written.
                    env.finish(phase + 1)?;
                    events.emit(Event::Materialised {
                        phase,
                        image: phase + 1,
                        bytes: phase_bytes[phase],
                        intermediate: phase + 1 < n_phases,
                    });
                }
                if let PhaseWork::Fragments(op) = &work[phase] {
                    // The guard on the side this phase's output is actually on.
                    // The tiling check below runs over valid regions, which for
                    // a fragment phase are the cores and therefore tile
                    // whatever happened; this is the check that can fail.
                    //
                    // A later barrier that reduces over this phase's stream runs
                    // the same check again from `reduce_phase`, and that
                    // repetition is deliberate — see `reduce_phase`'s own doc
                    // for what it costs and why the alternatives are worse. This
                    // is the copy that runs for a phase *no* barrier reads, and
                    // the copy that fails at the phase which made the hole
                    // rather than at whatever comes after it.
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
                        &barrier,
                        &mut ready,
                        &mut iterative_ready,
                        &mut barrier_held,
                        &barrier_released,
                    );
                }
            }
            done += 1;
        }
    }

    // The guard again, on what was *actually* written rather than on what the
    // decomposition promised. A decomposition that tiles and an executor that
    // wrote something else would otherwise agree. Against **each phase's own**
    // volume, which is the image it wrote.
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
    let (sidecar_writes, sidecar_reads, sidecar_bytes_written, sidecar_bytes_read) =
        env.counters().sidecar_snapshot();
    let (sidecar_listings, sidecar_keys_listed) = env.counters().listing_snapshot();
    let blocks_visited = events.log().op_sequence_per_block().len();
    let blocks_admitted = events.log().blocks_admitted().len();
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
        blocks_admitted,
        fragment_applications: env
            .counters()
            .fragment_applications
            .load(std::sync::atomic::Ordering::SeqCst),
        materialisations: written
            .iter()
            .take(n_phases.saturating_sub(1))
            .map(|boxes| boxes.len())
            .sum(),
        substages,
        substage_changes,
        reads,
        writes,
        read_voxels,
        write_voxels,
        chunks_read,
        side_writes,
        side_bytes_written,
        slabs_run: env
            .counters()
            .slabs_run
            .load(std::sync::atomic::Ordering::SeqCst),
        blocks_sliced: env
            .counters()
            .blocks_sliced
            .load(std::sync::atomic::Ordering::SeqCst),
        sidecar_reads,
        sidecar_writes,
        sidecar_bytes_read,
        sidecar_bytes_written,
        sidecar_listings,
        sidecar_keys_listed,
        peak_resident_bytes: peak,
        estimated_work,
        listener_faults,
        log,
    })
}

/// A barrier phase's reduction: computed **once for the phase**, in the gap the
/// barrier creates.
///
/// Public because two schedulers need it and there must not be two copies.
/// `execute_phases` calls it when its gate opens; a distributed worker calls it
/// on the first task it is handed of a barrier phase. Both reach the same moment
/// — every earlier phase complete, no block of this one started — and both must
/// therefore reach the same bytes.
///
/// # Every node computes this, and nothing transports it
///
/// A distributed run does **not** ship the blob. It does not need to: the blob
/// is *derived* from the fragment set rather than observed, the fragment set is
/// already on storage every node can read, and
/// [`PhaseView`](crate::fragment::PhaseView) walks the lattice in an order that
/// is a function of the plan. So every worker that runs this over the same
/// complete set gets byte-identical bytes with no agreement protocol, no
/// election, and nothing added to a coordinator whose whole design is that it
/// holds no data. `docs/design/barriers.md` §9 has the measurement and the two
/// arms it was decided between.
///
/// What that costs is `nodes` reads of the fragment set and `nodes` folds
/// instead of one of each — and the count is the point: **`nodes` is set by the
/// machines a caller has, not by how finely they cut the volume.** The whole
/// case against re-deriving per block was that the multiplier was `blocks`,
/// which a caller raises to make a stage fit in memory. This multiplier does not
/// move when they do.
///
/// # The set is checked before it is reduced
///
/// A reduction over a partial fragment set is the plausible-wrong-answer shape
/// this feature exists to remove, so the completeness the barrier promises is
/// **verified** rather than assumed, for every declared input stream whose
/// producer said [`Coverage::EveryBlock`](crate::fragment::Coverage). In-process
/// that guard has always run at the end of the producing phase; in a distributed
/// run nothing ran it, because `execute_task_of` is per task and there is no
/// end-of-phase moment on a worker. Running it here closes that gap and catches
/// the one failure mode this design has: a sidecar store that is not in fact
/// shared between nodes, where each worker would otherwise reduce over its own
/// fragments and answer plausibly and differently on every machine.
///
/// In a single-node run that check is the *second* one on the same stream:
/// `execute_phases` runs the same `check_fragment_coverage` on a fragment
/// phase's outputs the moment that phase's last task completes, and every
/// producer named here is an earlier phase, so it has already completed and
/// already been checked. **That duplication is kept deliberately**, and what it
/// costs is one extra listing per producing phase, returning one key per block.
///
/// It is kept because the alternative is worse in the direction that matters.
/// The check cannot simply move to `reduce_phase`: `execute_phases` checks
/// *every* fragment phase, including those no barrier ever reduces over, and it
/// checks at the phase that made the hole rather than at whatever runs next, so
/// dropping it would let a doomed run keep going through the phases in between.
/// And it cannot be *conditionally* skipped by handing this function a
/// caller-supplied "already verified" set: that turns a guard against a
/// plausible-wrong-answer into something a caller can switch off by getting one
/// argument wrong, on the path — the distributed one — where nothing else runs
/// it at all.
///
/// The cost is bounded by the thing the standing rule asks about. `blocks` is
/// the multiplier a caller raises to make a stage fit in memory, and the extra
/// listing is `O(blocks)` keys against a phase that irreducibly writes `blocks`
/// fragments and reads at least `blocks` more — so the ratio is fixed and does
/// not move when they cut more finely. `Stats::sidecar_listings` and
/// `sidecar_keys_listed` report both figures, and `tests/fragment_stats.rs`
/// pins the listing count against the block count so a change that made it grow
/// would be caught rather than argued about.
///
/// Refuses a phase the plan does not mark as a barrier: without one there is no
/// moment at which the fragment set is complete, so there is nothing well
/// defined to compute.
pub fn reduce_phase(
    decomposition: &Decomposition,
    phase: usize,
    work: &[PhaseWork<'_>],
    env: &dyn Environment,
) -> Result<Vec<u8>> {
    if !decomposition
        .phases
        .get(phase)
        .map(|entry| entry.barrier)
        .unwrap_or(false)
    {
        return Err(Error::InvalidArgument(format!(
            "phase {phase} is not a barrier, so there is no moment at which its fragment \
             set is complete and nothing well defined for `FragmentOp::reduce` to be \
             computed over. A reduction taken at any other moment is taken over whatever \
             fragments happened to exist."
        )));
    }
    let PhaseWork::Fragments(op) = &work[phase] else {
        return Ok(Vec::new());
    };
    let inputs = op.inputs();
    // The completeness the barrier promises, verified. Only for a stream whose
    // producer declared every-block coverage: where a hole is legitimate there
    // is nothing to compare against, and the op decides what absence means.
    //
    // **Grouped by producing phase, and checked once per phase rather than once
    // per input.** `check_fragment_coverage` lists *every* output stream of the
    // op it is handed, so one call already covers each of that producer's
    // streams whichever input brought us here. Run per input it repeated the
    // whole check — and every listing in it — once for each stream this barrier
    // reads from that phase, and a listing returns one key per block. So the
    // duplicated cost was the product of two figures the caller sets: how many
    // streams the op joins and how finely the volume is cut. Grouping removes
    // the first factor outright.
    let mut by_producer: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
    for input in &inputs {
        let Some(PhaseWork::Fragments(producer)) = work.get(input.phase) else {
            continue;
        };
        let declares_every_block = producer
            .outputs()
            .iter()
            .any(|out| out.stream == input.stream && out.coverage == Coverage::EveryBlock);
        if !declares_every_block {
            continue;
        }
        by_producer
            .entry(input.phase)
            .or_default()
            .push(&input.stream);
    }
    for (from, streams) in &by_producer {
        let Some(PhaseWork::Fragments(producer)) = work.get(*from) else {
            continue;
        };
        super::fragment::check_fragment_coverage(env, decomposition, *from, *producer).map_err(
            |err| {
                Error::InvalidArgument(format!(
                    "phase {phase}: fragment op {:?} declares a barrier and is about to \
                     reduce over stream(s) {streams:?} from phase {from}, and that phase's \
                     fragment set is not complete: {err} A barrier is the statement that \
                     every block of the phase below has finished, so a hole here is either a \
                     scheduler that released this phase early or — in a distributed run — a \
                     sidecar store that is not actually shared between nodes, in which case \
                     every worker would reduce over its own fragments and answer plausibly \
                     and differently on each machine.",
                    op.name(),
                ))
            },
        )?;
    }
    let streams: BTreeMap<String, usize> = inputs
        .into_iter()
        .map(|input| (input.stream, input.phase))
        .collect();
    let view =
        super::fragment::PhaseView::new(phase, &decomposition.phases[phase].grid, env, streams);
    let answer = op.reduce(&view)?;
    // **`SeamFold::Unordered` is checked here too, and for the same reason it is
    // checked per block.** The claim is that the fold is a function of the *set*
    // of fragments rather than of their order; the lattice is walked row-major,
    // which is one order out of many, and two different lattices walk two
    // different ones. An `f64` accumulation over three or more fragments answers
    // differently and would make the phase's answer a property of how the volume
    // was cut. Skipped for a one-block lattice, which has no order.
    //
    // It is *not* what makes two nodes agree — they walk the same lattice, so
    // they see the same order and agree for any deterministic op, associative or
    // not. This is about two *lattices*, which is decomposition invariance.
    if op.seam_fold() == Some(SeamFold::Unordered) && view.blocks().len() > 1 {
        let again = op.reduce(&view.reversed())?;
        if again != answer {
            return Err(Error::InvalidArgument(format!(
                "fragment op {:?} declares `SeamFold::Unordered` — that its answer is a \
                 function of the set of fragments it is handed and not of their order — and \
                 its `reduce` over phase {phase}'s {} block(s) produced {} byte(s) walking \
                 the lattice forwards and {} byte(s) walking it backwards. A reduction is \
                 folded in the lattice's own order, and two lattices are two orders, so a \
                 fold that does not associate makes the phase's answer a property of how the \
                 volume was cut. Accumulate in a type where the combine associates — an \
                 integer or a fixed-point sum — or declare `SeamFold::OrderDependent` and \
                 give up decomposition invariance for this phase.",
                op.name(),
                view.blocks().len(),
                answer.len(),
                again.len(),
            )));
        }
    }
    Ok(answer)
}

/// A task whose dependencies are all done joins the ready heap — unless its
/// phase is an iteration, in which case it is counted instead, or a barrier
/// whose reduction has not run, in which case it is held until it has.
///
/// One function rather than the same three lines at the two places a task
/// becomes ready, because the two are exactly the same decision and a scheduler
/// that admitted an iterative task in one of them would run one block of a
/// lockstep phase on its own.
fn admit(
    task: &super::graph::Task,
    hints: &Hints,
    iterative: &[bool],
    barrier: &[bool],
    ready: &mut BinaryHeap<Reverse<([usize; 5], usize)>>,
    iterative_ready: &mut [usize],
    barrier_held: &mut [Vec<usize>],
    barrier_released: &[bool],
) {
    if iterative[task.phase] {
        iterative_ready[task.phase] += 1;
    } else if barrier[task.phase] && !barrier_released[task.phase] {
        barrier_held[task.phase].push(task.id);
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
    // **A hoisted reduction is refused on this entry point rather than handed
    // nothing.** `FragmentOp::reduce` is computed once for the phase and reaches
    // the blocks through `BlockView::reduced`; nothing in a single task's
    // arguments carries it, so running one block here would hand the op an empty
    // slice and get a plausible answer from an empty table in every block —
    // which `barriers.md` §7.7 says no guard could catch afterwards.
    //
    // **It is not refused because the blob cannot be had.** A caller with a
    // barrier phase computes it with [`reduce_phase`] and passes it to
    // [`execute_task_with_reduction`], which is exactly what
    // `distributed::worker` does — once per phase, from the same fragment set,
    // with no transport. This entry point is the one that has no blob, so it
    // refuses to pretend it does.
    if let PhaseWork::Fragments(op) = work {
        if op.barrier() {
            let probe = crate::fragment::PhaseView::at_plan_time(
                task.phase,
                &decomposition.phases[task.phase].grid,
            );
            let answered = op.reduce(&probe);
            if answered
                .as_ref()
                .map(|bytes| !bytes.is_empty())
                .unwrap_or(true)
            {
                return Err(Error::InvalidArgument(format!(
                    "task (phase {}, block {:?}) belongs to fragment op {:?}, which computes a \
                     phase reduction with `reduce`. That blob is computed once for the whole \
                     phase and handed to every block as `BlockView::reduced`, and this entry \
                     point takes no blob — so running one block through it would hand the op \
                     an empty reduction and get a plausible answer from an empty table. \
                     Compute it with `strategy::reduce_phase` once the barrier has opened and \
                     pass it to `strategy::execute_task_with_reduction`; nothing has to travel \
                     to do that, because the fragment set the reduction is derived from is \
                     already on storage every node reads.",
                    task.phase,
                    task.index,
                    op.name()
                )));
            }
        }
    }
    // **One slab**, which is what makes this the convenience layer: a caller
    // wanting threads inside a block is stating a machine's resources, and wants
    // the entry point that takes them.
    execute_task_with_reduction(chain, decomposition, task, work, &[], env, listeners, 1)
}

/// [`execute_task_of`], with the phase's reduction supplied and the threads the
/// caller will let one block have.
///
/// The entry point a distributed worker uses for a barrier phase whose op hoists
/// a reduction. `reduced` is what [`reduce_phase`] returned for `task.phase`,
/// computed once by that worker rather than shipped to it — see `reduce_phase`
/// for why the blob is derived rather than transported.
///
/// Pass `&[]` for any phase that is not a barrier, or whose op takes the default
/// `reduce`; that is what [`execute_task_of`] does, and it is the whole
/// difference between the two.
///
/// # `slabs`, and why it is a parameter here and nowhere else in this family
///
/// **`1` is today's behaviour exactly**: one block, on this thread, uncut. Above
/// one it is the most the block may be cut into, and it is an *offer* — a chain
/// that has not declared itself sliceable declines it and runs uncut, which is
/// every chain this crate shipped before the declarations existed. See
/// [`crate::slab::apply_at_most`].
///
/// **It is the regime the whole feature is for, and the only one in which this
/// crate has no other parallelism to offer.** `execute_phases` runs a wave of
/// blocks, and slabs are what it does with the workers a block lattice left
/// parked; a distributed worker's main loop computes **one task at a time**, so
/// on a node with cores to spare every one of them is parked and there is no
/// block-level alternative competing for them. `docs/design/intra-block.md` §7
/// measured that row at 4.6-5.4x, and §13.7 is the note that this was the last
/// place it had not reached.
///
/// **What the parameter costs, since this is a public entry point.** One call
/// site in this crate — `distributed::worker` — and a compile error naming the
/// line for anybody outside it, where `1` is the answer that keeps their
/// behaviour identical. It is a parameter rather than a fourth entry point
/// because this function is called *for a real run* and never as a convenience,
/// which is this crate's own test for when exhaustiveness is worth paying:
/// somebody has to decide how many threads a node spends on one block, and that
/// is exactly the decision that should not be inheritable from a default. The
/// two wrappers above are the convenience layer and keep their arity.
///
/// **It cannot disturb a hoisted reduction, and that is structural rather than
/// careful.** [`reduce_phase`] is computed by the caller before this is entered
/// and reads the fragment set; slabs are offered inside `run_task` only after it
/// has dispatched a [`PhaseWork::Fragments`] phase elsewhere, so a fragment
/// phase — the only kind that has a reduction — never reaches the offer at all.
/// `tests/intra_block_slicing.rs` asserts that rather than restating it.
#[allow(clippy::too_many_arguments)]
pub fn execute_task_with_reduction(
    chain: &Chain,
    decomposition: &Decomposition,
    task: &super::graph::Task,
    work: &PhaseWork<'_>,
    reduced: &[u8],
    env: &dyn Environment,
    listeners: &[Arc<dyn EventListener>],
    slabs: usize,
) -> Result<TaskOutcome> {
    let slots = chain.slots();
    let events = Dispatch::new(listeners);
    let mut outcome = run_task(
        task,
        decomposition,
        &slots,
        work,
        env,
        &events,
        decomposition.n_phases(),
        reduced,
        slabs,
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
    reduced: &[u8],
    slabs: usize,
) -> Result<TaskOutcome> {
    if let PhaseWork::Fragments(op) = work {
        return run_fragment_task(task, decomposition, *op, env, events, n_phases, reduced);
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
    // content of the change that introduced `source`: `fetch` is asked of image
    // `task.phase` and is in that image's space; `read` is this phase's own read
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
    // the image that was read, `read` is in the image being written. They are
    // the same region in the same volume for every phase whose output grid is
    // its input grid, which is why one anchor sufficed until a lattice phase
    // needed both. See `op::Placement`.
    //
    // The buffer holds `fetch`, so that is where the ops read from, and the
    // volume is the one `fetch` is a region of — the image that was read. An op
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
        image: task.phase,
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
        // The images this phase's source leaves read, at **the same region**:
        // a source leaf has reach 0, so what it reads is what the block already
        // fetches, and `check_source_images` is what makes those the same
        // integers by requiring the two images to be on one lattice.
        //
        // Read here rather than beside the input read for one reason: a block
        // that short circuits has not looked at its input, and reading a second
        // array for it would be paying for an arm nothing consumed. (Today a
        // phase with a source leaf never short circuits — `Chain::Source`
        // declines `constant_maps_to` — so this is a property of the code
        // rather than of the plan, and worth keeping true by construction.)
        let mut sources: Vec<(usize, BlockBuf)> = Vec::with_capacity(phase.source_images.len());
        for &image in &phase.source_images {
            let started = Instant::now();
            let stored = env.read(image, fetch)?;
            let read_ns = started.elapsed().as_nanos() as u64;
            // Priced exactly like the input read, through the same event, so a
            // run that reads two arrays per block reports two arrays' worth of
            // bytes. A second arm that cost nothing in the counters would make
            // every measurement of this feature a measurement of the wrong plan.
            events.emit(Event::RegionRead {
                source: format!("level {image}"),
                image,
                index: Some(task.index),
                region: fetch.clone(),
                voxels: fetch.voxels(),
                bytes: fetch.voxels() as u64 * decomposition.dtype_at(image).size_of() as u64,
                chunks: chunks_touched(fetch, &env.chunk_shape()),
                duration_ns: read_ns,
            });
            sources.push((image, stored));
        }
        for (&slot, place) in phase.slots.iter().zip(&places) {
            let started = Instant::now();
            // **The offer, not a demand.** `slabs` is what the policy allots
            // from the pool and the lattice; whether it is taken is the chain's
            // to say, and a chain that has not declared itself sliceable runs
            // uncut on this thread exactly as it always has. See
            // `slab::apply_at_most`, and `EnvCounters::blocks_sliced` for what
            // says afterwards which way it went.
            let (next, _ran) = env.apply_sliced(slots[slot], &buf, &sources, place, slabs)?;
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
                let produced = env.apply_side(slots[slot], &buf, &sources, &next, &block)?;
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
        // slots and any of them may hold a source leaf naming the same image.
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
        image: task.phase + 1,
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

/// What differs between two applications of one block, or `None` if nothing
/// does.
///
/// Compares what the executor would go on to *store*: the fragments, stream by
/// stream and byte for byte, and the pixel block when the run holds one. A
/// simulated run holds no array, so the pixel comparison is skipped there rather
/// than passing vacuously; the fragments are where a partial accumulator lives
/// and they are compared in both kinds of run.
fn order_disagreement(first: &BlockOutput, second: &BlockOutput) -> Option<String> {
    if first.fragments.len() != second.fragments.len() {
        return Some(format!(
            "{} fragment(s) rather than {}",
            second.fragments.len(),
            first.fragments.len()
        ));
    }
    for ((stream, bytes), (other, other_bytes)) in first.fragments.iter().zip(&second.fragments) {
        if stream != other {
            return Some(format!("stream {other:?} rather than {stream:?}"));
        }
        if bytes != other_bytes {
            return Some(format!("different bytes for stream {stream:?}"));
        }
    }
    match (&first.pixels, &second.pixels) {
        (Some(BlockBuf::Array(one)), Some(BlockBuf::Array(two))) if one != two => {
            Some("a different pixel block".to_string())
        }
        _ => None,
    }
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
#[allow(clippy::too_many_arguments)]
fn run_fragment_task(
    task: &super::graph::Task,
    decomposition: &Decomposition,
    op: &dyn super::fragment::FragmentOp,
    env: &dyn Environment,
    events: &Dispatch,
    n_phases: usize,
    reduced: &[u8],
) -> Result<TaskOutcome> {
    let phase = &decomposition.phases[task.phase];
    // As in `run_task`: `fetch` is in the read image's space, `read` in this
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
            image: task.phase,
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

    // The images this op declared besides the one it is handed, at the block's
    // **own fetch region** — the same arrangement `run_task` has for a chain's
    // source leaves, through the same event, so a fragment phase that reads two
    // arrays reports two arrays' worth of bytes. A second arm that cost nothing
    // in the counters would make every measurement of this feature a measurement
    // of the wrong plan.
    //
    // Read whether or not the op reads its own image: `reads_pixels` is about
    // image `p` and `source_inputs` is about every other image, so an op that
    // consults a stored array without wanting its own input pays for one array
    // rather than two.
    let mut sources: Vec<(usize, BlockBuf)> = Vec::with_capacity(phase.source_images.len());
    for &image in &phase.source_images {
        let started = Instant::now();
        let stored = env.read(image, fetch)?;
        let read_ns = started.elapsed().as_nanos() as u64;
        events.emit(Event::RegionRead {
            source: format!("level {image}"),
            image,
            index: Some(task.index),
            region: fetch.clone(),
            voxels: fetch.voxels(),
            bytes: fetch.voxels() as u64 * decomposition.dtype_at(image).size_of() as u64,
            chunks: chunks_touched(fetch, &env.chunk_shape()),
            duration_ns: read_ns,
        });
        sources.push((image, stored));
    }
    let borrowed: Vec<(usize, &BlockBuf)> =
        sources.iter().map(|(image, buf)| (*image, buf)).collect();

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

    // **`SeamFold::Unordered` is checked, not believed.** The claim is that this
    // block's output is a function of the *set* of fragments it was handed and
    // not of their order, so the way to test it is to hand them over twice in
    // opposite orders and compare the bytes. An `f64` accumulation over three or
    // more fragments fails it, which is the hazard the variant exists to catch;
    // an integer one cannot.
    //
    // Skipped when the neighbourhood holds at most one fragment, because a
    // one-element sequence has no order and the second application would cost a
    // block's work to assert nothing.
    //
    // The reversal is applied to `wanted` as well as to `gathered`, so it
    // reaches an op that declared `gathers() == false` and pulls with
    // `BlockView::stream_fragments` — that op pays a second pass of sidecar
    // reads, and the counters show it.
    let ordered: usize = wanted.values().map(|(_, blocks)| blocks.len()).sum();
    let verify_order = op.seam_fold() == Some(SeamFold::Unordered) && ordered > 1;
    let (reversed_wanted, reversed_gathered) = if verify_order {
        (
            wanted
                .iter()
                .map(|(stream, (phase, blocks))| {
                    let mut blocks = blocks.clone();
                    blocks.reverse();
                    (stream.clone(), (*phase, blocks))
                })
                .collect::<BTreeMap<_, _>>(),
            gathered
                .iter()
                .map(|(stream, found)| {
                    let mut found = found.clone();
                    found.reverse();
                    (stream.clone(), found)
                })
                .collect::<BTreeMap<_, _>>(),
        )
    } else {
        (BTreeMap::new(), BTreeMap::new())
    };

    let at = Anchor::of_region(fetch, decomposition.volume_at(task.phase))?;
    let produced = {
        let view = BlockView::new(
            task.phase,
            task.index,
            &phase.grid,
            &task.geometry.core,
            read,
            &task.geometry.valid,
            at.clone(),
            decomposition.dtype_at(task.phase + 1),
            env,
            pixels.as_ref(),
            wanted,
            gathered,
        )
        .with_reduced(reduced);
        // One atomic add per application, and nothing else: the counter must not
        // become part of what it measures. Nothing is formatted, nothing is
        // allocated, and the block's buffers are untouched.
        env.counters()
            .fragment_applications
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        op.apply_with(&view, SourceBlocks::new(&borrowed))?
    };
    if verify_order {
        let again = {
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
                reversed_wanted,
                reversed_gathered,
            )
            .with_reduced(reduced);
            // Counted, because it happened. The order check applies the op a
            // second time to the same block, and a `fragment_applications` that
            // quietly excluded it would report a cost the run did not have.
            env.counters()
                .fragment_applications
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            op.apply_with(&view, SourceBlocks::new(&borrowed))?
        };
        let disagreement = order_disagreement(&produced, &again);
        // Everything the second application allocated, and everything this block
        // holds, released before the refusal rather than after it: a run that
        // ends here still has to leave the residency counters meaning what they
        // say.
        if let Some(buf) = &again.pixels {
            env.release(buf);
        }
        if disagreement.is_some() {
            for (_, stored) in &sources {
                env.release(stored);
            }
            if let Some(buf) = &pixels {
                env.release(buf);
            }
            if let Some(buf) = &produced.pixels {
                env.release(buf);
            }
        }
        if let Some(what) = disagreement {
            return Err(Error::InvalidArgument(format!(
                "fragment op {:?} declares `SeamFold::Unordered` — that its output is a \
                 function of the set of fragments it is handed and not of their order — and \
                 block {:?} of phase {} produced {what} when the same {ordered} fragment(s) \
                 were handed over in the opposite order. That is a non-associative \
                 accumulator, and `f64` addition is the one that exists: a region straddling \
                 a seam would then be summed differently depending on how the volume was cut, \
                 so the answer is a property of the plan rather than of the data. Accumulate \
                 in a type where the combine associates — an integer or a fixed-point sum — \
                 or declare `SeamFold::OrderDependent` and give up decomposition invariance \
                 out loud.",
                op.name(),
                task.index,
                task.phase
            )));
        }
    }
    for (_, stored) in &sources {
        env.release(stored);
    }
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
                "fragment op {:?} declares `writes_pixels` and returned no buffer, so image \
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
            image: task.phase + 1,
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
/// image lifetime, the dependents — is the same code that runs for every other
/// kind of phase.
///
/// **The trivial form, on purpose.** Every block runs every substage, and the
/// convergence test is "did any block's core come out different from what it went
/// in as". No per-block skip, no dirty set, no frontier: those are the
/// optimisation, and a trivial executor is correct by design, which is what makes
/// it the oracle they will be tested against.
///
/// **The blocks of one substage run together.** They are independent — each reads
/// `current`, which nothing writes during the substage, and writes its own core of
/// `next`, and the cores are disjoint by construction — so the wave is the same
/// shape as the ready wave the rest of the executor runs, through the same pool at
/// the same `Hints::concurrency`. What is *not* parallel is the substage boundary,
/// and it must not be: every block of substage `k+1` reads cores its neighbours
/// wrote at `k`, so the swap below is a barrier and the loop is serial in the
/// substage index. That is the whole reason the depth is paid in private round
/// trips rather than in halo.
///
/// `next` is behind a lock rather than split into disjoint views because the
/// placement is a copy of a core and the substage around it is the arithmetic:
/// blocks contend for the length of a `memcpy` each, once per block per substage.
/// Splitting it would buy that back and needs an indexing scheme the private
/// buffer does not otherwise have.
///
/// **What a distributed run would need, and does not have here.** The two private
/// buffers shared across workers rather than owned by this function, and the
/// barrier below made explicit rather than implied by the end of a loop body.
#[allow(clippy::too_many_arguments)]
fn run_iterative_phase(
    graph: &TaskGraph,
    phase_index: usize,
    decomposition: &Decomposition,
    op: &dyn IterativeOp,
    env: &dyn Environment,
    events: &Dispatch,
    n_phases: usize,
    pool: Option<&rayon::ThreadPool>,
) -> Result<(usize, Vec<u64>, Vec<TaskOutcome>)> {
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
    // outside this phase can see them: the plan allocates no image for them and
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
    let produced = (|| -> Result<(usize, Vec<u64>, Vec<TaskOutcome>)> {
        let limit = op.limit().substages();
        let mut ran = 0usize;
        let mut changes: Vec<u64> = Vec::new();
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
            // The three things the wave shares. `next` is written by every block
            // and read by none of them until the swap below, so a lock over it is
            // the whole of the coordination a substage needs; the two counters are
            // a reduction over the blocks and are folded atomically for the same
            // reason the ready wave's outcomes are gathered rather than merged.
            let changed = AtomicBool::new(false);
            let changed_voxels = AtomicU64::new(0);
            {
                let sink = Mutex::new(&mut next);
                let one_block = |task: &Task| -> Result<()> {
                    let fetch = &task.geometry.source;
                    let read = &task.geometry.read;
                    let at = Anchor::of_region(fetch, decomposition.volume_at(phase_index))?;
                    // One buffer per declared operand, every one of them over the
                    // block's read extent — see `iterate::Substage` for why they share
                    // an extent rather than each getting its own.
                    let mut buffers = Vec::with_capacity(operands.len());
                    for operand in &operands {
                        // The running operand comes off the image only at substage 0;
                        // after that it is what the previous substage wrote, and it is
                        // the neighbours' *cores* of that which make the reach stay at
                        // one substage's worth. A fixed operand comes off the image
                        // every time, which is the whole point of declaring it.
                        let from_image = operand.operand == Operand::Fixed || ran == 0;
                        if from_image {
                            let started = Instant::now();
                            let buf = env.read(phase_index, fetch)?;
                            let read_ns = started.elapsed().as_nanos() as u64;
                            let chunks = chunks_touched(fetch, &env.chunk_shape());
                            events.emit(Event::RegionRead {
                                source: format!("level {phase_index}"),
                                image: phase_index,
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
                            // A private buffer is not an image: no `RegionRead`, because
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
                        Some(false) => {
                            changed.store(true, Ordering::SeqCst);
                            // Beside the decision and never in place of it: `same` is
                            // the environment's own override point and stays the sole
                            // thing that ends an iteration. This is the size of a
                            // difference it has already reported, and it can only be
                            // taken where `same` could answer at all.
                            changed_voxels.fetch_add(
                                differing_block_voxels(&core, &before)?.unwrap_or(0),
                                Ordering::SeqCst,
                            );
                        }
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
                    {
                        let mut held = sink.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        env.place(&mut held, &whole, valid, &core)?;
                    }

                    env.release(&before);
                    env.release(&core);
                    env.release(&result);
                    for buf in &buffers {
                        env.release(buf);
                    }
                    Ok(())
                };
                match pool {
                    None => tasks.iter().try_for_each(&one_block)?,
                    Some(pool) => pool.install(|| tasks.par_iter().try_for_each(&one_block))?,
                }
            }
            // The exchange point. Everything below reads `current`, so the swap is
            // the barrier; see this function's header for the distributed shape.
            std::mem::swap(&mut current, &mut next);
            ran += 1;
            changes.push(changed_voxels.load(Ordering::SeqCst));
            if !changed.load(Ordering::SeqCst) {
                break;
            }
        }

        // The fixed point, written out block by block. This is the only write the
        // phase makes to an image, which is what makes the substage count invisible
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
                image: phase_index + 1,
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
        Ok((ran, changes, outcomes))
    })();

    env.release(&current);
    env.release(&next);
    produced
}

/// How many voxels two buffers of one extent differ in, or `None` where there
/// are no values to difference.
///
/// **Derived beside [`Environment::same`] and never in place of it.** That method
/// is the environment's override point and stays the sole thing that decides an
/// iteration has converged; this is the *size* of a difference it has already
/// reported, which is what turns a substage count into a sequence a caller can
/// read a rate off. It answers exactly where `same` answers — both need real
/// arrays — so a run that can decide can also count, and one that cannot does
/// neither.
///
/// A free function rather than a method for the same reason: an environment has
/// nothing to say about it that it does not already say through `same`.
///
/// **`pub`, over [`Voxels`], and fallible — and each of those three was a
/// defect.** It was private, so thirteen consumer test files re-derived it in
/// three incompatible shapes; it took the executor's `BlockBuf`, which is not
/// what a consumer holds; and it answered `None` for mismatched extents, which
/// the one call site turned into **zero** with `unwrap_or(0)`. Ten of the
/// thirteen copies used a bare `zip`, which truncates — so two volumes of
/// *different* extents were reported as differing in a small number of voxels,
/// and in a parity suite that is the failure reading as a pass. A count that
/// cannot fail is a check that cannot fail.
///
/// So the extent and element-type mismatches are **refused by name** rather than
/// folded into an absence. The one thing that is legitimately absent — a
/// half-precision buffer, which no [`Voxels`] variant holds — is refused by name
/// too, for the same reason: `same` cannot answer there either, and saying so is
/// not the same as answering zero.
pub fn differing_voxels(left: &Voxels, right: &Voxels) -> Result<u64> {
    fn count<T: crate::voxels::VoxelElement>(left: &Voxels, right: &Voxels) -> Result<u64> {
        let left = left.view::<T>()?;
        let right = right.view::<T>()?;
        Ok(left
            .iter()
            .zip(right.iter())
            .filter(|(one, other)| one != other)
            .count() as u64)
    }
    if left.shape() != right.shape() {
        return Err(Error::ShapeMismatch {
            expected: left.shape().to_vec(),
            got: right.shape().to_vec(),
        });
    }
    if left.dtype() != right.dtype() {
        return Err(Error::InvalidArgument(format!(
            "counting differing voxels: one volume holds {} and the other holds {}. Two element \
             types are not comparable voxel by voxel, and answering a count for them would be a \
             number about nothing.",
            left.dtype().numpy_name(),
            right.dtype().numpy_name()
        )));
    }
    match left.dtype() {
        Dtype::Bool => count::<bool>(left, right),
        Dtype::U8 => count::<u8>(left, right),
        Dtype::U16 => count::<u16>(left, right),
        Dtype::U32 => count::<u32>(left, right),
        Dtype::U64 => count::<u64>(left, right),
        Dtype::I8 => count::<i8>(left, right),
        Dtype::I16 => count::<i16>(left, right),
        Dtype::I32 => count::<i32>(left, right),
        Dtype::I64 => count::<i64>(left, right),
        Dtype::F32 => count::<f32>(left, right),
        Dtype::F64 => count::<f64>(left, right),
        // No `Voxels` variant holds one, so there is nothing to view and
        // nothing to count. `same` is in the same position.
        Dtype::F16 => Err(Error::InvalidArgument(
            "counting differing voxels: no buffer holds half-precision, so there is nothing to \
             compare. `Environment::same` cannot answer here either."
                .to_string(),
        )),
    }
}

/// [`differing_voxels`] over the executor's own buffers.
///
/// `Ok(None)` is the one honest absence: an accounting environment holds no
/// values, so there is nothing to difference. A mismatch of extent or element
/// type is an **error** and is propagated, because the executor guarantees
/// neither can happen here and a zero in their place would hide the day one
/// does.
fn differing_block_voxels(left: &BlockBuf, right: &BlockBuf) -> Result<Option<u64>> {
    let (BlockBuf::Array(left), BlockBuf::Array(right)) = (left, right) else {
        return Ok(None);
    };
    differing_voxels(left, right).map(Some)
}

#[cfg(test)]
mod differing_voxels_tests {
    use super::*;

    /// **The count refuses two volumes of different extents, by name** — which
    /// is the whole of G21, because the shape it replaces did not.
    ///
    /// The fixture is chosen so that the wrong implementation returns a
    /// *plausible* answer rather than an obviously silly one: the two volumes
    /// agree everywhere the smaller one reaches, so a bare `zip` — which
    /// truncates to the shorter iterator — answers **zero differing voxels** for
    /// a pair that do not even have the same shape. In a parity suite that reads
    /// as a pass. The assertion below is that we refuse instead, and the
    /// liveness partner is the `zip` itself, run here so the number it would
    /// have produced is on the record rather than described.
    #[test]
    fn counting_differing_voxels_refuses_two_extents_by_name() {
        use crate::voxels::Voxels;
        use ndarray::Array3;

        let small: Voxels = Array3::<u16>::from_elem((2, 3, 4), 7).into();
        let large: Voxels = Array3::<u16>::from_elem((2, 3, 8), 7).into();

        let error = differing_voxels(&small, &large).unwrap_err().to_string();
        assert!(
            error.contains("2, 3, 4") && error.contains("2, 3, 8"),
            "the refusal must name both extents, got {error}"
        );
        assert!(
            differing_voxels(&large, &small).is_err(),
            "and in either order"
        );

        // **Liveness: what the shape this replaces would have answered.**
        let truncating = small
            .view::<u16>()
            .unwrap()
            .iter()
            .zip(large.view::<u16>().unwrap().iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            truncating, 0,
            "the fixture must be one a truncating count answers plausibly, or this test is not \
             about truncation"
        );

        // The element type is refused by name too, and for the same reason.
        let other: Voxels = Array3::<u8>::from_elem((2, 3, 4), 7).into();
        let error = differing_voxels(&small, &other).unwrap_err().to_string();
        assert!(
            error.contains("uint16") && error.contains("uint8"),
            "the refusal must name both element types, got {error}"
        );

        // And it still counts when it can: the ordinary case, and one voxel of
        // it, so that a function returning zero unconditionally would fail here.
        let mut changed = Array3::<u16>::from_elem((2, 3, 4), 7);
        changed[[1, 2, 3]] = 8;
        let changed: Voxels = changed.into();
        assert_eq!(differing_voxels(&small, &small).unwrap(), 0);
        assert_eq!(differing_voxels(&small, &changed).unwrap(), 1);
    }
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
    // The whole chain as one phase, which is what this function is asked about;
    // its slots are `0..n` and every source leaf in it is one this run reads.
    let slots = chain.slots();
    let group: Vec<usize> = (0..slots.len()).collect();
    let traffic = super::decomposition::PhaseTraffic {
        images_read: super::decomposition::images_read_by(&slots, &group, volume)?,
        writes_an_image: true,
    };
    let mut candidates = constraints.block_candidates.clone();
    candidates.sort_unstable_by(|a, b| b.cmp(a));
    for edge in candidates {
        // The same floor `Greedy` applies, so that a branch is priced at the
        // block the run will really use rather than at one the planner would
        // have refused to cut.
        let axes = cuttable_axes(&constraints.split_axes, &reach, volume, edge);
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
            traffic,
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
///   for one image: [`Chain::produces`] already refuses branches that write
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
        decomposition.declare_source_images(&workflow.chain)?;
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
            keep_images: BTreeSet::new(),
            // **The policy every strategy advises, and the one that answers
            // `1` under this strategy's own concurrency.** A slab count is
            // derived from the pool and the block lattice, so a strategy states
            // the rule and not a number; see `SlabPolicy` for why the rule is
            // "fill idle workers" rather than a tuning curve.
            slab_policy: SlabPolicy::default(),
        }
    }
}

/// Choose the cuts by the `O(n^2)` dynamic program, with the block size for
/// each phase chosen independently. The `2^(n-1)` enumeration is retained and
/// selectable — see [`PartitionSearch`].
///
/// Per-phase block sizes are separable given the partition — the total is
/// `sum over phases of makespan_p` and the budget binds each phase
/// independently — so choosing them is an inner loop over candidates rather
/// than a `candidates^phases` product.
///
/// # The objective is a makespan, and that is what makes the block choice real
///
/// The per-phase block edge has always been an inner loop here. What it lacked
/// was a reason to answer differently in different phases. The old objective was
/// the phase's **serial work**, `cost_per_block x n_blocks`, and that is
/// `volume x redundancy x per-voxel` — `n_blocks` cancels, `redundancy >= 1`
/// falls monotonically as the block grows, and so the sweep answered *"the
/// largest candidate"* in every phase of every chain. The freedom was on paper.
///
/// It has to be more than read volume, because a smaller block trades two things
/// against each other and read volume is only one of them:
///
/// * it **raises** the read, by `prod((B + lo + hi) / B)` per cut axis — that is
///   the term the old objective had, and the term that makes a fragment-and-join
///   op want one block;
/// * it **creates** the concurrency, because the unit of parallelism is the
///   block. At one block a phase has one task, and a pool of 40 workers runs it
///   on one thread with 39 parked. A local op — reach zero, redundancy `1.0` at
///   every grid — pays *nothing* for the cut and is the whole width of the pool
///   faster for it. Read volume alone cannot see that: it is the same number at
///   every edge, and the tie-break then takes the largest.
///
/// So the objective is the phase's predicted **wall clock**, and it is
/// [`phase_makespan`] — the larger of the two lower bounds a phase has:
///
/// ```text
/// makespan(phase) = max( cost_per_block x ceil(n / workers) ,  read x read_cost + core x write )
///                        \------------ the pool -----------/    \--------- the channel -------/
/// ```
///
/// summed over phases, which are sequential because a phase boundary is a
/// materialisation.
///
/// The **pool** bound's `ceil` is not a fudge: the blocks of one phase depend
/// only on the phase before, so they are independent and identically priced, and
/// `ceil(n / P)` is the exact makespan of `n` identical independent tasks on `P`
/// processors. It is honest about the quantisation too — 41 blocks on 40 workers
/// costs two rounds, and a search told so will not propose it.
///
/// The **channel** bound is there because the pool bound divides *everything* by
/// the pool, reads included, and workers do not multiply bandwidth. Without it
/// the search buys parallelism with read amplification — see [`phase_makespan`],
/// which records what that cost when it was measured. It is what makes the
/// search take the cut where it is free and refuse it where it is paid for in
/// traffic.
///
/// `workers` is [`Enumerating::concurrency`], the same number
/// [`Strategy::hints`] hands the executor. Nothing new is configured, and no
/// coefficient is invented: both bounds are built from
/// [`CostModel`](crate::decomposition::CostModel) as it already is.
///
/// **At `concurrency == 1` this is the old objective exactly** — `ceil(n / 1)`
/// is `n`, the pool bound is then the channel bound plus the compute and the
/// conflict so the `max` returns it, the expression is the one that was there,
/// and the `f64` is bit-identical. That is deliberate twice over: no plan built
/// before this moves, and the old objective stays reachable as the **negative
/// control**, a search that cannot see task count, takes one block everywhere,
/// and looks optimal on read volume while running on one thread.
///
/// # What it does not price
///
/// **Residency**, in two halves that go opposite ways and are worth separating.
///
/// * *The block half can only improve.* `budget_bytes` binds
///   `working_set_bytes_per_block x expected_concurrency` per phase and that
///   test is unchanged; a phase this objective moves to a *smaller* block has a
///   strictly smaller working set, and one it leaves alone has the same.
///   `tests/per_phase_block.rs` asserts the priced peak does not rise.
/// * *The partition half can rise.* More phases means more intermediate images
///   alive, and buying a cut in order to give one half its own grid is exactly
///   what this objective does. The cut is priced — it has to pay its own
///   `materialise_cost_per_voxel` — but it is priced in **time**, and a whole
///   intermediate image is a lot of bytes to buy with seconds. A caller whose
///   ceiling is bytes says so with `budget_bytes`, or leaves `concurrency` at
///   one. This is the one direction in which raising `concurrency` is not free,
///   and it is stated here rather than discovered at tile scale.
///
/// **Anything that would break the DP.** Both bounds read this phase's own grid,
/// this phase's own price and the pool width, so the objective is additive over
/// phases and local to one group — see [`PartitionSearch`], whose list of what
/// would break additivity this does not join.
#[derive(Debug, Clone)]
pub struct Enumerating {
    /// Blocks the run will hold in flight — handed to the executor by
    /// [`Strategy::hints`], and the `workers` of the makespan objective above.
    ///
    /// `1` is the negative control and the default: it makes the objective the
    /// serial work total, which is monotone in the block edge, so every phase
    /// takes the largest candidate that fits.
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
/// * [`price_phase`] is charged per phase and the phase makespans are summed —
///   the same arithmetic [`predicted_makespan`] re-does over a finished plan,
///   one phase at a time. [`phase_makespan`] turns a per-block cost into a
///   phase time out of that phase's own grid, that phase's own read, and the
///   pool width, so it is as local to one group as the price it multiplies. At
///   `concurrency == 1` it reduces to `cost_per_block x n_blocks` and this is
///   [`crate::decomposition::predicted_cost`] exactly;
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
/// * **[`rounds`] crossing a step**, once the objective is a makespan: a
///   widening that costs a phase one more round is a jump the fixed-grid
///   argument does not cover either. Same shape as the three above — widening
///   changed what was chosen — and it is one more reason the prune stays
///   unwritten.
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
/// A budget that forces a *smaller* edge raises the *work* total too, since the
/// read is `volume x prod((B + lo + hi) / B)` and that falls with `B` — though
/// under the makespan objective a smaller edge may still be cheaper, because it
/// buys rounds; that is a statement about which grid is chosen and not about the
/// monotonicity of the price at a fixed grid, which is what this paragraph is
/// establishing. And the concern
/// that a wider group swallows a phase boundary and its write does **not**
/// apply: `is_materialised` is `i < n`, so for a fixed `i` it is the same for
/// every `j`, and the saved boundary is priced in `best[j]`, outside this term.
///
/// An exact prune is therefore still available — it has to restart its bound
/// whenever the chosen grid changes, which happens at most a few times per axis
/// as the reach grows. That is design work, not a one-line guard.
/// **A parameter of the planner rather than of the answer**, and the first entry
/// in that space; [`crate::decomposition::BlockLadder`] is the second and
/// carries the space's definition and its acceptance bar. Every variant here
/// computes the same volume; what differs is which decomposition is chosen and
/// what the choosing costs.
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
    /// **No partition at all: every slot in one phase, and only the block edge
    /// is chosen.**
    ///
    /// Not an optimisation of the two above — it answers a different question,
    /// and the difference is who decides the fusion. A caller that has already
    /// decided its chain is one phase, for a reason the cost model cannot see,
    /// does not want a search that may cut it. The case this was built for is a
    /// consumer's thinning stage: it fuses `n` passes into one round because the
    /// phase it really runs is an [`crate::iterate::IterativeOp`] with **one**
    /// substage, and the chain it hands a planner is a stand-in with `n` slots.
    /// Asking the DP about that stand-in gets a correct answer to a question the
    /// stage cannot act on — measured there, it returns one phase at one pass
    /// per round, two at two and four at four, buying a materialisation per pass
    /// that an iterative op has no way to perform.
    ///
    /// So the group is given and the sweep over [`Constraints::block_candidates`]
    /// is the whole of the search. That sweep is not re-implemented: it is the
    /// same [`PhasePricer`], the same [`price_phase`], the same
    /// [`phase_makespan`] and the same tie-break — *lower makespan, then the
    /// larger edge* — which is the point, because a caller that hand-rolls it
    /// prices a grid the search would never offer the moment either drifts.
    ///
    /// **A barrier is refused rather than fused over.** A full-reach op must be
    /// alone in its phase ([`crate::decomposition::is_planning_barrier`]), and a
    /// caller asking for one group across one is asking for a plan that is not
    /// legal rather than one that is merely expensive. The two searches drop
    /// such a partition and carry on; this one has nothing to carry on to, so it
    /// says so by name.
    ///
    /// [`Constraints::block_candidates`]: crate::decomposition::Constraints::block_candidates
    SingleGroup,
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

/// What a plan predicts it will take on `workers`, under `model`.
///
/// [`crate::decomposition::predicted_cost`] is the same walk over the same
/// phases with the same [`price_phase`] arguments; the one difference is the
/// factor a phase's per-block cost is multiplied by — `n_blocks` there, and
/// [`rounds`] here. So `predicted_cost` is this at `workers == 1`, bit for bit,
/// and the relation is the whole of the change [`Enumerating`] documents.
///
/// It exists for the reason `predicted_cost` gives for itself: the search now
/// minimises a quantity, and a quantity nobody can read back off the plan is a
/// claim nobody can check. `tests/per_phase_block.rs` checks it — that the plan
/// the search returns is the plan that minimises this over the reachable ones.
///
/// The units are the model's: voxelwise maps under
/// [`CostModel::default`](crate::decomposition::CostModel::default),
/// nanoseconds under one calibrated from a [`crate::statistics::Snapshot`].
///
/// `work` is what [`crate::decomposition::predicted_cost`] takes it for, word
/// for word: a fragment or iterative phase owns no chain slot, and without it
/// such a phase prices at zero compute and at a read and a write it may not
/// perform. `&[]` is allowed for a plan that is all pixels and is **refused**
/// for one that is not.
pub fn predicted_makespan(
    chain: &Chain,
    decomposition: &Decomposition,
    work: &[crate::fragment::PhaseWork<'_>],
    model: &super::decomposition::CostModel,
    workers: usize,
) -> Result<f64> {
    let slots = chain.slots();
    let mut total = 0.0_f64;
    for (index, phase) in decomposition.phases.iter().enumerate() {
        if phase.slots.iter().any(|&slot| slot >= slots.len()) {
            return Err(Error::InvalidArgument(format!(
                "predicted_makespan: phase {index} names slot {:?}, and the chain has {}",
                phase.slots.iter().max(),
                slots.len()
            )));
        }
        let volume = decomposition.volume_at(index);
        let (_, _, _, orders) =
            super::decomposition::summarise_slots(&slots, &phase.slots, volume)?;
        let compute =
            super::decomposition::phase_compute_per_voxel(&slots, phase, work.get(index))?;
        let traffic = super::decomposition::phase_traffic(index, phase, work.get(index))?;
        let is_materialised = index + 1 < decomposition.phases.len();
        let cost = price_phase(
            &phase.grid,
            // The halo, not the reach: the two differ exactly where a granted
            // halo is wider than the ops asked for, and there the reach
            // under-charges by a factor that grows as the block shrinks. See
            // `price_phase`.
            &phase.halo,
            compute,
            orders.len(),
            is_materialised,
            decomposition.dtype_at(index).size_of() as f64,
            model,
            model.materialise_cost_per_voxel,
            traffic,
        );
        // Zero for a phase that writes no image, so that the channel bound below
        // counts the same bytes `price_phase` charged for. See `PhaseTraffic`.
        let write = if !traffic.writes_an_image {
            0.0
        } else if is_materialised {
            model.materialise_cost_per_voxel
        } else {
            model.write_cost_per_voxel
        };
        total += phase_makespan(&cost, &phase.grid, workers, model, write);
    }
    Ok(total)
}

/// What the partition search looked at, and what it threw away.
///
/// **A search that caps itself silently reads as a search that considered
/// everything.** Every number here is a count of something the search declined
/// to carry forward, and [`Enumerating::decompose_accounted`] is how a caller
/// reads them back off the plan it was given.
///
/// The space, stated so the counts have a denominator. For `n` slots the search
/// is over contiguous partitions **and** a block edge per phase. Those two are
/// separable — see [`Enumerating`] — so it is not `partitions x candidates^n`:
/// it is `n(n+1)/2` contiguous runs, each sweeping `block_candidates` once, and
/// then a search over partitions that only ever *looks up* a priced run.
///
/// What is pruned, in the order it happens:
///
/// * **barrier cuts** — `runs_forbidden_by_barrier` runs are never priced,
///   because a full-reach slot must be alone in its phase (`barrier_cuts`).
/// * **the reach-derived floor** — an axis a cut would not narrow is dropped
///   from the candidate's grid before it is priced ([`cuttable_axes`]). This is
///   not counted as a drop because the candidate survives; only its axes change.
/// * **candidates with no grid** — `candidates.no_grid`, an edge at which
///   [`BlockGrid::along`] produces nothing once the floor has taken its axes.
/// * **candidates over budget** — `candidates.over_budget`.
/// * **runs that cannot be a phase at all** — `runs_refused`: a mandate
///   conflict, two coordinate spaces, or no affordable candidate.
///
/// And one hard cap, which is the one worth shouting about: `slots` above
/// [`MAX_SLOTS`] is **refused**, not truncated. There is no silent cap in here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchAccount {
    pub slots: usize,
    /// The `workers` the makespan objective was taken against. `1` is the
    /// negative control — see [`Enumerating::concurrency`].
    pub workers: usize,
    /// `block_candidates.len()`, the width of the per-run inner sweep. A run
    /// with a mandated extent offers one grid instead; see
    /// [`CandidateTally::offered`].
    pub candidates_offered_per_run: usize,
    /// Contiguous runs priced, out of the `n(n+1)/2` that exist.
    pub runs_priced: usize,
    /// Of those, how many could not be a phase at all — a mandate conflict, two
    /// coordinate spaces, or no affordable candidate.
    pub runs_refused: usize,
    /// Runs never priced because a barrier cut forbids them.
    pub runs_forbidden_by_barrier: usize,
    /// The candidate sweep, summed over every run priced.
    pub candidates: CandidateTally,
    /// The chosen plan, phase by phase.
    ///
    /// This is the result the whole change exists for. Two entries with
    /// different blocks is a per-phase decision the caller can point at.
    pub chosen: Vec<ChosenPhase>,
}

/// One phase of the plan the search returned, as the search sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChosenPhase {
    /// The slot range this phase covers, `start..end`.
    pub slots: (usize, usize),
    pub block: [usize; 3],
    pub n_blocks: usize,
    /// `ceil(n_blocks / workers)` — see [`rounds`]. Equal to `n_blocks` under
    /// the negative control, which is what makes that control what it is.
    pub rounds: usize,
}

/// How many rounds of `workers` it takes to run `n_blocks` independent blocks.
///
/// **This is the term that makes the per-phase block choice a choice.** Blocks
/// inside a phase depend only on the phase before it — a boundary is a
/// materialisation — so the blocks of one phase are mutually independent and a
/// pool of `workers` runs them in `ceil(n / workers)` rounds. For identical
/// tasks that is not an approximation of the makespan, it *is* the makespan of
/// list scheduling, and the tasks are identical because [`price_phase`] charges
/// every block at the widest one.
///
/// At `workers == 1` it is `n_blocks`, so `cost_per_block * rounds` is the
/// literal expression the search used before this existed and every plan built
/// under it is bit-identical. See [`Enumerating::concurrency`].
pub fn rounds(n_blocks: usize, workers: usize) -> usize {
    n_blocks.div_ceil(workers.max(1))
}

/// What one phase is predicted to take on `workers`: a **roofline**, the larger
/// of the two lower bounds a phase has.
///
/// ```text
/// makespan = max( cost_per_block x ceil(n / workers) ,  read x n x read_cost + core x n x write )
///                 \------------- the pool ---------/    \------------ the channel ----------/
/// ```
///
/// # Why the second term has to be there
///
/// The first term alone says a phase can be made arbitrarily fast by cutting it
/// into more blocks, because it divides *everything* — the reads included — by
/// the pool. That is false and this crate has already paid for it once: a
/// 716-offset element on a `24 x 20` volume, cut into 336 blocks, ran past
/// fifteen minutes where one block read the volume once, and
/// [`cuttable_axes`] exists because of it. Blocks share a channel; workers do
/// not multiply bandwidth. Measured on this file's own probe before the term
/// existed, a phase whose reach denied every useful cut was handed 32 blocks at
/// **16x the read volume** for a predicted 2.5x — the same trade, re-derived.
///
/// So the phase cannot finish before its bytes have moved. That is a bound that
/// does not divide by anything, and the two together are what makes the search
/// buy parallelism where it is free (a local op reads the same total at every
/// grid) and refuse it where it is paid for in traffic (a wide reach reads
/// `(B + lo + hi) / B` times the volume, which is the whole term). Neither is a
/// tunable: both are built from coefficients
/// [`CostModel`](crate::decomposition::CostModel) already carries.
///
/// # It changes nothing at `workers == 1`
///
/// `cost_per_block` is `read x (read_cost + compute) + core x write + conflict`
/// and every one of those is non-negative — the assumption [`PartitionSearch`]
/// already states it searches under. So at `rounds == n` the first term is the
/// second plus the compute and the conflict, the `max` returns the first, and
/// the `f64` is the one the old objective produced. Bit for bit.
pub fn phase_makespan(
    cost: &super::decomposition::PhaseCost,
    grid: &BlockGrid,
    workers: usize,
    model: &super::decomposition::CostModel,
    write_cost_per_voxel: f64,
) -> f64 {
    let n = grid.n_blocks() as f64;
    let pool = cost.cost_per_block * rounds(grid.n_blocks(), workers) as f64;
    // `mean_core_voxels`, so that `core * n` is the volume the phase writes
    // rather than the volume plus the grid's padding. `price_phase` charges the
    // same core, and the channel bound is meant to be the same bytes counted a
    // second way — a bound stated in a different unit from the term it is
    // maxed against would not be one.
    let channel = cost.read_voxels_per_block * n * model.read_cost_per_voxel
        + grid.mean_core_voxels() * n * write_cost_per_voxel;
    // `total_cmp`, not `f64::max`: the crate's arithmetic never selects between
    // two `f64`s through a partial order.
    if pool.total_cmp(&channel).is_lt() {
        channel
    } else {
        pool
    }
}

/// What one contiguous run's sweep over the candidate edges came to.
///
/// Kept because a search that silently drops candidates reads as a search that
/// considered everything. Folded into [`SearchAccount`] as the search prices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CandidateTally {
    /// Candidate grids this run had to choose between. That is
    /// `block_candidates.len()` for an ordinary run and **one** for a run whose
    /// ops mandate a block extent — a mandate replaces the candidate list rather
    /// than filtering it, so there is nothing to choose between and nothing
    /// dropped.
    pub offered: usize,
    /// Dropped because [`BlockGrid::along`] produces no grid at that edge —
    /// after [`cuttable_axes`] has taken the reach-derived floor off the axes.
    pub no_grid: usize,
    /// Dropped because the working set times the concurrency exceeds the budget.
    pub over_budget: usize,
    /// Priced and compared. `offered - no_grid - over_budget`.
    pub priced: usize,
}

/// Everything needed to price one contiguous run of slots.
struct PhasePricer<'a> {
    slots: &'a [&'a Chain],
    volume: [usize; 3],
    bytes: f64,
    constraints: &'a Constraints,
    /// Blocks the run will hold in flight, from [`Enumerating::concurrency`] —
    /// the same number [`Strategy::hints`] hands the executor. See [`rounds`].
    workers: usize,
}

impl PhasePricer<'_> {
    /// The per-voxel write charge [`price_phase`] applies, re-derived so
    /// [`phase_makespan`] can put the same number on the channel bound.
    fn write_cost(&self, is_materialised: bool) -> f64 {
        if is_materialised {
            self.constraints.model.materialise_cost_per_voxel
        } else {
            self.constraints.model.write_cost_per_voxel
        }
    }

    /// Price the run `fold` covers, choosing its block edge.
    ///
    /// `is_materialised` is the whole of this function's dependence on the rest
    /// of the partition, and it is `fold.end < slots.len()` — see
    /// [`PartitionSearch`] on why that is what makes the DP legitimate.
    ///
    /// **The figure returned is a makespan, not a work total**, and the two
    /// differ by exactly [`rounds`]. See [`Enumerating`] for the argument.
    fn price(&self, fold: &GroupFold, is_materialised: bool) -> (GroupPrice, CandidateTally) {
        let mut tally = CandidateTally::default();
        // Reaches in two coordinate spaces cannot be folded without a grid, so a
        // run containing both is infeasible *as a run* — the same shape of
        // answer a block-shape conflict gives, and it drops the partition rather
        // than the plan.
        if let Some(refusal) = &fold.refusal {
            return (GroupPrice::Refused(Some(refusal.clone())), tally);
        }
        let group: Vec<usize> = (fold.start..fold.end).collect();
        let reach = fold.reach.clone().unwrap_or_default();
        // What the ops in this run will accept. A conflict is a fact about
        // *this partition* — the same two ops in two phases are fine — so it
        // drops the partition and the search goes on.
        let mandated = match constraint_for(self.slots, &group, self.volume) {
            Ok(found) => found,
            Err(err) => return (GroupPrice::Refused(Some(err.to_string())), tally),
        };
        // The compute figure is re-asked per candidate rather than taken from
        // the fold, because an op may declare a term whose denominator is a
        // block extent — see `decomposition::compute_per_voxel`. For every op
        // that does not, this is the same number by the same route.
        // Priced at the halo the grid would be *granted*, which is the reach
        // everywhere except under a mandate — see `price_phase` for why the
        // difference is not a rounding one.
        // One traversal per array the run reads: its own input image plus every
        // distinct image a `Chain::Source` leaf in it names. See
        // `images_read_by`.
        let images_read =
            match super::decomposition::images_read_by(self.slots, &group, self.volume) {
                Ok(count) => count,
                Err(err) => return (GroupPrice::Refused(Some(err.to_string())), tally),
            };
        let traffic = super::decomposition::PhaseTraffic {
            images_read,
            // A run of chain slots is a pixel phase, and a pixel phase writes
            // the image after it.
            writes_an_image: true,
        };
        let price = |grid: &BlockGrid, halo: &Reach| {
            price_phase(
                grid,
                halo,
                compute_per_voxel(self.slots, &group, grid.block()),
                fold.orders.len(),
                is_materialised,
                self.bytes,
                &self.constraints.model,
                self.constraints.model.materialise_cost_per_voxel,
                traffic,
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
                    let cost = price(&grid, &window);
                    tally.offered += 1;
                    if affordable(&cost) {
                        tally.priced += 1;
                        halo = window;
                        let makespan = phase_makespan(
                            &cost,
                            &grid,
                            self.workers,
                            &self.constraints.model,
                            self.write_cost(is_materialised),
                        );
                        chosen = Some((makespan, 0, grid));
                    } else {
                        tally.over_budget += 1;
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
            for &edge in &self.constraints.block_candidates {
                tally.offered += 1;
                // The reach-derived floor, per candidate: an axis is cut only
                // where the cut narrows what a block reads, which depends on the
                // edge and so cannot be hoisted out of this loop.
                let axes = cuttable_axes(&self.constraints.split_axes, &reach, self.volume, edge);
                let grid = match BlockGrid::along(self.volume, &axes, edge) {
                    Ok(grid) => grid,
                    Err(_) => {
                        tally.no_grid += 1;
                        continue;
                    }
                };
                let cost = price(&grid, &reach);
                if !affordable(&cost) {
                    tally.over_budget += 1;
                    continue;
                }
                tally.priced += 1;
                // **The objective.** Not `cost_per_block * n_blocks`, which is
                // the phase's serial *work* and falls monotonically as the block
                // grows — under it this loop always answers "the largest
                // candidate" and the per-phase freedom is freedom on paper. This
                // is the phase's predicted *wall clock*: the same per-block cost
                // over `ceil(n_blocks / workers)` rounds. At `workers == 1` the
                // two are the same expression and the same bits.
                let phase_total = phase_makespan(
                    &cost,
                    &grid,
                    self.workers,
                    &self.constraints.model,
                    self.write_cost(is_materialised),
                );
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
            return (GroupPrice::Refused(note), tally);
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
                return (GroupPrice::Refused(Some(err.to_string())), tally);
            }
        }
        (GroupPrice::Priced(priced), tally)
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
    /// What the sweep over runs and candidates cost, and what it threw away.
    account: SearchAccount,
}

impl PriceTable {
    fn build(pricer: &PhasePricer, forced_cuts: u32) -> Self {
        let n = pricer.slots.len();
        let mut runs = Vec::with_capacity(n);
        let mut account = SearchAccount {
            slots: n,
            workers: pricer.workers.max(1),
            candidates_offered_per_run: pricer.constraints.block_candidates.len(),
            ..SearchAccount::default()
        };
        for start in 0..n {
            let mut row = Vec::new();
            let mut fold = GroupFold::new(start);
            for end in start + 1..=n {
                // Growing to `end` crosses the boundary between slots `end - 2`
                // and `end - 1`, which is cut mask bit `end - 2`. A forced cut
                // there ends the row: neither this run nor any longer one from
                // this start is a legal phase.
                if end >= start + 2 && forced_cuts & (1u32 << (end - 2)) != 0 {
                    account.runs_forbidden_by_barrier += n + 1 - end;
                    break;
                }
                fold.extend(pricer.slots, pricer.volume);
                let (price, tally) = pricer.price(&fold, end < n);
                account.candidates.offered += tally.offered;
                account.candidates.no_grid += tally.no_grid;
                account.candidates.over_budget += tally.over_budget;
                account.candidates.priced += tally.priced;
                row.push(price);
            }
            runs.push(row);
        }
        let (refused, priced, _) = {
            let table = PriceTableRefusalView(&runs);
            table.refusals()
        };
        account.runs_priced = priced;
        account.runs_refused = refused;
        Self { runs, n, account }
    }

    /// The price of `start..end`, or `None` where a forced cut forbids the run.
    fn get(&self, start: usize, end: usize) -> Option<&GroupPrice> {
        self.runs.get(start)?.get(end.checked_sub(start + 1)?)
    }

    /// How many priced runs were refused, out of how many were priced, and the
    /// first reason that was not the budget — in `(start, end)` order, which is
    /// the DP's own.
    fn refusals(&self) -> (usize, usize, Option<String>) {
        PriceTableRefusalView(&self.runs).refusals()
    }
}

/// [`PriceTable::refusals`] over the rows alone, so `build` can tally them
/// before the table it is building exists.
struct PriceTableRefusalView<'a>(&'a [Vec<GroupPrice>]);

impl PriceTableRefusalView<'_> {
    fn refusals(&self) -> (usize, usize, Option<String>) {
        let mut refused = 0;
        let mut priced = 0;
        let mut note = None;
        for row in self.0 {
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
/// The whole chain as one phase, if that is a legal plan at all.
///
/// Two failures, distinguished because a caller can act on one and not the
/// other:
///
/// * **A barrier inside the group** is refused here with `Err`, naming the cut,
///   because no block candidate could ever fix it. The other two searches drop
///   the partition and keep going; this one has no other partition, and
///   answering "nothing fits the budget" would send the caller to widen a budget
///   that was never the problem.
/// * **No candidate fits** is `Ok(None)`, the same shape the other two return,
///   and the caller's message says a budget.
fn search_single_group(
    table: &PriceTable,
    forced_cuts: u32,
) -> Result<Option<Vec<(usize, usize)>>> {
    if forced_cuts != 0 {
        let first = forced_cuts.trailing_zeros() as usize;
        return Err(Error::InvalidArgument(format!(
            "enumerating: `PartitionSearch::SingleGroup` was asked for one phase over all \
             {} slots, and a full-reach op forces a cut between slots {first} and {}. A \
             planning barrier must be alone in its phase, so no block candidate makes this \
             one group legal — see `decomposition::is_planning_barrier`.",
            table.n,
            first + 1
        )));
    }
    Ok(match table.get(0, table.n) {
        Some(GroupPrice::Priced(_)) => Some(vec![(0, table.n)]),
        _ => None,
    })
}

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
        self.decompose_accounted(workflow, constraints)
            .map(|(decomposition, _)| decomposition)
    }

    fn hints(&self, workflow: &Workflow, decomposition: &Decomposition) -> Hints {
        Hints {
            visit_order: consensus_order(workflow, decomposition),
            priority: self.priority,
            concurrency: self.concurrency,
            prefetch_depth: 1,
            // Nothing kept: a strategy advising on speed has no reason to want
            // an intermediate afterwards. A caller who does overrides it.
            keep_images: BTreeSet::new(),
            slab_policy: SlabPolicy::default(),
        }
    }
}

impl Enumerating {
    /// [`Strategy::decompose`], with the [`SearchAccount`] it built the plan by.
    ///
    /// `decompose` is this and a `.0`. It exists separately because the trait
    /// returns a `Decomposition` and the counts are not part of the plan — they
    /// are a fact about the *search*, and a caller who wants to know what was
    /// dropped should not have to re-run it to find out.
    pub fn decompose_accounted(
        &self,
        workflow: &Workflow,
        constraints: &Constraints,
    ) -> Result<(Decomposition, SearchAccount)> {
        let slots = workflow.chain.slots();
        if slots.is_empty() {
            return Err(Error::InvalidArgument(
                "enumerating: the chain has no ops".to_string(),
            ));
        }
        let limit = match self.search {
            // No cut mask and no table of runs: one group is priced once, so
            // the `u32` mask that bounds the other two does not apply.
            PartitionSearch::SingleGroup => usize::MAX,
            PartitionSearch::Dp => MAX_SLOTS,
            PartitionSearch::Exhaustive => MAX_EXHAUSTIVE_SLOTS,
        };
        if slots.len() > limit {
            return Err(Error::InvalidArgument(match self.search {
                PartitionSearch::SingleGroup => {
                    unreachable!("SingleGroup has no slot limit, so this branch is unreachable")
                }
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
            workers: self.concurrency.max(1),
        };
        let table = PriceTable::build(&pricer, forced_cuts);

        // Why a partition was dropped for a reason that is not the budget. Kept
        // so the final refusal can say "these two ops mandate different blocks"
        // rather than blaming a budget that was never the problem.
        let mut constraint_note: Option<String> = None;
        let mut budget_failures = 0usize;
        let spans = match self.search {
            PartitionSearch::SingleGroup => search_single_group(&table, forced_cuts)?,
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
                PartitionSearch::SingleGroup => format!(
                    "enumerating: the one phase of all {} slots fits none of the block candidates                      {:?} within the {:?} byte budget at concurrency {}. `SingleGroup` has no                      other partition to fall back to — that is what asking for it means — so                      add a smaller candidate, raise the budget, or use `PartitionSearch::Dp` and                      let the search cut.{}",
                    slots.len(),
                    constraints.block_candidates,
                    constraints.budget_bytes,
                    constraints.expected_concurrency,
                    reason(&constraint_note),
                ),
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

        let mut account = table.account.clone();
        account.chosen = phases
            .iter()
            .map(|phase| {
                let n = phase.grid.n_blocks();
                let start = phase.slots.first().copied().unwrap_or(0);
                let end = phase.slots.last().map_or(0, |last| last + 1);
                ChosenPhase {
                    slots: (start, end),
                    block: phase.grid.block(),
                    n_blocks: n,
                    rounds: rounds(n, account.workers),
                }
            })
            .collect();

        let mut decomposition = Decomposition {
            volume,
            dtype: workflow.dtype,
            phases,
            chain_reach: workflow.chain.reach3(&volume),
        };
        decomposition.declare_dtypes(&workflow.chain)?;
        decomposition.declare_source_images(&workflow.chain)?;
        decomposition.check()?;
        Ok((decomposition, account))
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
            let is_materialised = position + 1 < groups.len();
            phases.push(phase_for_group(
                &slots,
                group,
                volume,
                bytes,
                is_materialised,
                constraints,
                "greedy",
                position,
            )?);
        }

        let mut decomposition = Decomposition {
            volume,
            dtype: workflow.dtype,
            phases,
            chain_reach: workflow.chain.reach3(&volume),
        };
        decomposition.declare_dtypes(&workflow.chain)?;
        decomposition.declare_source_images(&workflow.chain)?;
        decomposition.check()?;
        Ok(decomposition)
    }

    fn hints(&self, workflow: &Workflow, decomposition: &Decomposition) -> Hints {
        Hints {
            visit_order: consensus_order(workflow, decomposition),
            priority: SchedulePriority::BlockMajor,
            concurrency: self.concurrency,
            prefetch_depth: 2,
            keep_images: BTreeSet::new(),
            slab_policy: SlabPolicy::default(),
        }
    }
}

/// The phase a contiguous run of slots gets: its reach, its halo, and the
/// largest block grid the budget admits — or the one a mandate names.
///
/// Extracted because [`Greedy`] and [`Materialising`] differ in *where the cuts
/// go* and in nothing else once a group is in hand. Keeping one copy is not
/// tidiness: the budget test, the reach-derived floor and the mandate's lattice
/// are three places a second copy could drift, and a planner that prices a
/// candidate differently from its sibling makes the two incomparable, which is
/// exactly what a baseline exists to prevent.
///
/// `who` and `position` appear only in refusals, so each caller's message reads
/// as its own rather than as a shared internal's.
#[allow(clippy::too_many_arguments)]
fn phase_for_group(
    slots: &[&Chain],
    group: &[usize],
    volume: [usize; 3],
    bytes: f64,
    is_materialised: bool,
    constraints: &Constraints,
    who: &str,
    position: usize,
) -> Result<PhaseDecomposition> {
    // `compute` is dropped here and re-asked per candidate grid; see
    // `decomposition::compute_per_voxel`.
    let (reach, _compute, names, orders) =
        super::decomposition::summarise_slots(slots, group, volume)?;
    let mandated = constraint_for(slots, group, volume)?;
    // The same count `PhasePricer::price` takes; see `images_read_by`.
    let traffic = super::decomposition::PhaseTraffic {
        images_read: super::decomposition::images_read_by(slots, group, volume)?,
        writes_an_image: true,
    };
    let mut grid = None;
    let mut halo = reach.clone();
    if let Some(constraint) = &mandated {
        // Mandated, so there is nothing to choose between; the budget still
        // binds. See `Enumerating` for why the candidate list is replaced
        // rather than filtered.
        let (candidate, window) = constraint.lattice(volume, &reach)?.ok_or_else(|| {
            Error::InvalidArgument(format!(
                "{who}: phase {position} mandates {constraint:?}, which no block grid produces — \
                 a grid's cores are `index * block`, evenly strided and disjoint. A plan for it \
                 has to state each block's fetch region explicitly, which this strategy does not \
                 do."
            ))
        })?;
        halo = window;
        let cost = price_phase(
            &candidate,
            // The granted window, not the reach; see `price_phase`.
            &halo,
            compute_per_voxel(slots, group, candidate.block()),
            orders.len(),
            is_materialised,
            bytes,
            &constraints.model,
            constraints.model.materialise_cost_per_voxel,
            traffic,
        );
        let fits = constraints.budget_bytes.is_none_or(|budget| {
            cost.working_set_bytes_per_block * constraints.expected_concurrency.max(1) as f64
                <= budget as f64
        });
        if fits {
            grid = Some(candidate);
        }
    } else {
        // largest candidate that fits
        let mut candidates = constraints.block_candidates.clone();
        candidates.sort_unstable_by(|a, b| b.cmp(a));
        for edge in candidates {
            // As in `Enumerating`: the reach-derived floor, asked per candidate
            // because it is a question about this edge.
            let axes = cuttable_axes(&constraints.split_axes, &reach, volume, edge);
            let Ok(candidate) = BlockGrid::along(volume, &axes, edge) else {
                continue;
            };
            let cost = price_phase(
                &candidate,
                &reach,
                compute_per_voxel(slots, group, candidate.block()),
                orders.len(),
                is_materialised,
                bytes,
                &constraints.model,
                constraints.model.materialise_cost_per_voxel,
                traffic,
            );
            let fits = constraints.budget_bytes.is_none_or(|budget| {
                cost.working_set_bytes_per_block * constraints.expected_concurrency.max(1) as f64
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
            "{who}: no block candidate in {:?} fits the {:?} byte budget for phase {position} \
             (reach {reach})",
            constraints.block_candidates, constraints.budget_bytes
        ))
    })?;
    let phase = PhaseDecomposition::derive(group.to_vec(), names, reach, halo, grid);
    if let Some(constraint) = &mandated {
        constraint.check(&phase.blocks, &format!("{who}: phase {position}"))?;
    }
    Ok(phase)
}

/// One phase per slot: materialise everything, fuse nothing.
///
/// The missing corner of the strategy table. [`Trivial`] is maximally *fused*
/// and unblocked — one block, one phase — so it has no halo to re-read and
/// nothing to time separately. [`Greedy`] and [`Enumerating`] both fuse, by
/// heuristic and by search. Nothing ran the opposite extreme with real blocking
/// until this, and it is worth having for three reasons at once.
///
/// **1. It is the pessimistic baseline.** With nothing measured to trust,
/// materialising every stage is the conservative plan: no fusion, so no halo
/// recomputation beyond each op's own reach, and every intermediate paid for in
/// full. A fused plan that cannot beat it has bought nothing.
///
/// **2. It is the measurement instrument, which is the main point.** Per-op cost
/// attribution has exactly one obstacle: `Chain::Parallel` is a single `apply`
/// and a fused `Sequence` hides its members inside one phase, so *the measurable
/// unit is the slot*. Under this strategy every op **is** its own phase, so
/// `Event::OpApplied` times each one on its own and `statistics::Recorder`'s
/// per-slot attribution is exact — with no fusion that has to be broken in order
/// to measure it. That is how [`crate::decomposition::CostModel`] gets
/// *calibrated* rather than seeded.
///
/// **3. It is a free incumbent.** Its cost is `O(n)` in the slots — one phase
/// each, one candidate sweep each — and it is feasible whenever anything is, so
/// [`Materialising::incumbent_cost`] bounds a future pruned search from above
/// before that search starts. A partial partition already dearer than this
/// cannot be completed into a better plan.
///
/// **Why it is always feasible when anything is.** Fusing slots into one phase
/// can only *grow* a phase's reach, and a grown reach can only shrink the set of
/// cuttable axes and grow the resident set at a given block edge. So if the
/// singleton phase for slot `i` fits no candidate, no phase containing slot `i`
/// fits one either, and no partition at all is affordable. The refusal this
/// strategy gives is therefore the honest one and not an artefact of refusing to
/// fuse.
///
/// **Barriers fall out.** A full-reach op is a planning barrier and must be
/// alone in its phase; one phase per slot satisfies that without a special case,
/// and `barriers_do_not_merge_with_a_neighbour` in `tests/materialising.rs`
/// asserts that it is not merely an accident of the group-building loop.
///
/// **What it is not.** It is not a strategy to run production work with — it
/// writes every intermediate to an image and reads it back — and it does not
/// claim to be cheap. It claims to be *legible*.
#[derive(Debug, Clone)]
pub struct Materialising {
    pub concurrency: usize,
    pub priority: SchedulePriority,
}

impl Default for Materialising {
    /// **Serial and phase-major, and that is a measurement decision.** Two
    /// blocks in flight would overlap two ops' wall clocks, and the per-slot
    /// nanoseconds `Event::OpApplied` reports would then sum to more than the
    /// run took. At concurrency 1 the sum is a partition of the run's time and
    /// the accounted fraction means what it says. A caller wanting the baseline
    /// as a *plan* rather than as an instrument raises `concurrency` and loses
    /// nothing but the attribution.
    fn default() -> Self {
        Self {
            concurrency: 1,
            priority: SchedulePriority::PhaseMajor,
        }
    }
}

impl Strategy for Materialising {
    fn name(&self) -> &'static str {
        "materialising"
    }

    fn decompose(&self, workflow: &Workflow, constraints: &Constraints) -> Result<Decomposition> {
        let slots = workflow.chain.slots();
        if slots.is_empty() {
            return Err(Error::InvalidArgument(
                "materialising: the chain has no ops".to_string(),
            ));
        }
        let volume = workflow.shape;

        let mut phases = Vec::with_capacity(slots.len());
        // The element type the phase **reads**, folded slot by slot exactly as
        // `Decomposition::declare_dtypes` folds it and as
        // `decomposition::predicted_cost` reads it back with `dtype_at`.
        //
        // `Greedy` and `Enumerating` both hand `workflow.dtype` to every phase,
        // which prices a chain that binarizes halfway through as if the second
        // half still moved 8 bytes a voxel. That is one of the mispricings this
        // strategy exists to expose, and it is fixed *here* rather than in
        // `price_phase` because it is an argument a planner chooses, not
        // arithmetic the pricer does. One phase per slot is also where it bites
        // hardest: every dtype change is a phase boundary, so there is no fusion
        // hiding the discrepancy.
        //
        // It moves no voxel — `bytes` reaches only `working_set_bytes_per_block`,
        // which is a budget test.
        let mut reads = workflow.dtype;
        for position in 0..slots.len() {
            // Every phase but the last writes an intermediate. That is the whole
            // of "materialise everything": the test is the same one
            // `predicted_cost` makes, applied to a partition of singletons.
            let is_materialised = position + 1 < slots.len();
            phases.push(phase_for_group(
                &slots,
                &[position],
                volume,
                reads.size_of() as f64,
                is_materialised,
                constraints,
                "materialising",
                position,
            )?);
            reads = slots[position].produces(reads)?;
        }

        let mut decomposition = Decomposition {
            volume,
            dtype: workflow.dtype,
            phases,
            chain_reach: workflow.chain.reach3(&volume),
        };
        decomposition.declare_dtypes(&workflow.chain)?;
        decomposition.declare_source_images(&workflow.chain)?;
        decomposition.check()?;
        Ok(decomposition)
    }

    fn hints(&self, workflow: &Workflow, decomposition: &Decomposition) -> Hints {
        Hints {
            visit_order: consensus_order(workflow, decomposition),
            priority: self.priority,
            concurrency: self.concurrency,
            prefetch_depth: 1,
            keep_images: BTreeSet::new(),
            slab_policy: SlabPolicy::default(),
        }
    }
}

impl Materialising {
    /// What the fully materialised plan costs under `constraints.model`.
    ///
    /// **The incumbent bound, and it is free.** A branch-and-bound search over
    /// partitions needs an upper bound to prune against before it has explored
    /// anything, and this is one that is always available and always feasible —
    /// see the type's note on why refusing here means refusing everywhere. A
    /// partial partition whose priced prefix already exceeds this cannot be
    /// completed into a plan worth having.
    ///
    /// `O(n)` in the slots: one `phase_for_group` per slot, each a sweep over a
    /// candidate list that `Constraints` documents as short. No partition is
    /// enumerated and no table is built.
    ///
    /// It is deliberately **not** cached on the type. The cost is a function of
    /// the workflow and the constraints, neither of which this value holds, and a
    /// planner that memoised against the wrong one would be a planner whose
    /// answer depended on its history.
    ///
    /// **It is a work total, and [`Enumerating`] now minimises a makespan.** The
    /// two agree at `concurrency == 1` and part above it, in the direction that
    /// makes this an over-estimate: a makespan is never more than the work it is
    /// scheduled from. So it remains a valid *upper* bound for a pruned search
    /// over either objective — but a loose one at a wide pool, and anybody
    /// wiring branch-and-bound at `concurrency > 1` wants this strategy's plan
    /// through [`predicted_makespan`] instead, which is the same plan priced on
    /// the objective the search is minimising.
    pub fn incumbent_cost(&self, workflow: &Workflow, constraints: &Constraints) -> Result<f64> {
        let decomposition = self.decompose(workflow, constraints)?;
        // `&[]`: this strategy partitions a `Chain` and every phase it produces
        // owns slots of it, so there is no slotless phase for `work` to describe.
        super::decomposition::predicted_cost(
            &workflow.chain,
            &decomposition,
            &[],
            &constraints.model,
        )
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
    use crate::decomposition::{cuttable_axes, splittable_axes, CostModel, PhaseTraffic};
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

    /// What the floor buys, measured on the two plans it chooses between.
    ///
    /// The arithmetic underneath
    /// `both_planners_take_the_measured_case_from_336_blocks_to_6`, kept separate
    /// from it so that a planner change and a floor change fail different tests:
    /// 336 blocks reading **48.9x** the volume, against six reading it once, for
    /// the same answer.
    ///
    /// The figure is `forme.md`'s *touched voxels / volume voxels* — the
    /// distance from `1.0` being the cost of the decomposition — taken from
    /// `Decomposition::exact_read_voxels`, which is the clamped geometry rather
    /// than the model's infinite-grid redundancy. 48.9 rather than 336 because
    /// axis 2's reach is zero, so the one axis a cut does narrow is narrowed.
    #[test]
    fn the_floor_takes_the_amplification_from_49_to_1() {
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
        // exactly once: the one axis still cut carries no reach, so there is no
        // halo left to re-read. The `1.125` the motivating note recorded was
        // never asserted and does not reproduce.
        assert_eq!(amplified, 1.0);
    }

    /// **And it changes no voxel.** The floored plan, the unfloored one it
    /// replaced, and the single-block oracle all produce the same volume, byte
    /// for byte. A block grid is how the answer is cut up, never what it is —
    /// which is what makes the floor a cost decision that is safe to take. The
    /// same claim through the planners themselves is
    /// `neither_planner_moved_a_voxel_by_taking_the_floor`.
    #[test]
    fn the_floor_moves_the_grid_and_not_the_answer() {
        let workflow = Workflow::new(Chain::op(WindowSumOp::new("w", RADIUS)), VOLUME, Dtype::F64);
        let input = ramp(VOLUME);

        let run = |plan: &Decomposition| -> Voxels {
            let env = ArrayEnvironment::for_decomposition(input.clone(), plan, [4, 4, 4]).unwrap();
            execute("floor", &workflow, plan, &Hints::default(), &env).unwrap();
            env.image(plan.n_phases())
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
            "the 336-block plan the floor removes"
        );
        assert_eq!(
            run(&plan(&cuttable_axes(&[0, 1, 2], &reach, VOLUME, 3))),
            oracle,
            "the plan the floor leaves"
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
                price_phase(
                    grid,
                    &reach,
                    1.0,
                    1,
                    false,
                    8.0,
                    &CostModel::default(),
                    1.0,
                    PhaseTraffic::one_in_one_out(),
                )
                .working_set_bytes_per_block
            };
            assert!(
                price(&after) <= price(&before),
                "edge {edge}: the floor raised the resident set"
            );
        }
    }

    // ------------------------------------------------ through the planners --

    /// The floor, through the two planners that now call it.
    ///
    /// The measured case at the edge it was measured at: both searches used to
    /// offer 336 blocks reading **48.9x** the volume and now offer 6 reading it
    /// **once**, for the same answer. The 336-block grid is still constructed
    /// here, from `splittable_axes`, so the comparison is the two plans and not
    /// two recollections.
    ///
    /// One recorded figure is corrected in passing: the note that motivated this
    /// put the floored plan at `1.125x`. It is exactly `1.0` — axis 2 carries no
    /// reach at all, so the one axis still cut has no halo to re-read, and the
    /// only assertion that had been made on it was `< 1.2`.
    #[test]
    fn both_planners_take_the_measured_case_from_336_blocks_to_6() {
        let workflow = Workflow::new(Chain::op(WindowSumOp::new("w", RADIUS)), VOLUME, Dtype::F64);
        let constraints = Constraints {
            budget_bytes: None,
            expected_concurrency: 1,
            model: CostModel::default(),
            block_candidates: vec![3],
            split_axes: vec![0, 1, 2],
            ..Default::default()
        };
        let reach = Reach::from(RADIUS);
        let unfloored = Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases: vec![PhaseDecomposition::derive(
                vec![0],
                vec!["w".to_string()],
                reach.clone(),
                reach.clone(),
                BlockGrid::along(VOLUME, &splittable_axes(&[0, 1, 2], &reach, VOLUME), 3).unwrap(),
            )],
            chain_reach: RADIUS,
        };
        assert_eq!(unfloored.phases[0].grid.n_blocks(), 336);
        assert_eq!(amplification(&unfloored), 48.9375);

        for (name, plan) in [
            (
                "enumerating",
                Enumerating::default()
                    .decompose(&workflow, &constraints)
                    .unwrap(),
            ),
            (
                "greedy",
                Greedy::default()
                    .decompose(&workflow, &constraints)
                    .unwrap(),
            ),
        ] {
            assert_eq!(plan.phases[0].grid.block(), [24, 20, 3], "{name}");
            assert_eq!(plan.phases[0].grid.n_blocks(), 6, "{name}");
            assert_eq!(amplification(&plan), 1.0, "{name}");
            plan.check().unwrap();
        }
    }

    /// **And the planners' own plans compute the same volume.** The grid moved,
    /// so this is the assertion that matters: the floored plan, the 336-block one
    /// it replaced and the single-block oracle agree byte for byte.
    #[test]
    fn neither_planner_moved_a_voxel_by_taking_the_floor() {
        let workflow = Workflow::new(Chain::op(WindowSumOp::new("w", RADIUS)), VOLUME, Dtype::F64);
        let input = ramp(VOLUME);
        let run = |plan: &Decomposition| -> Voxels {
            let env = ArrayEnvironment::for_decomposition(input.clone(), plan, [4, 4, 4]).unwrap();
            execute("floor", &workflow, plan, &Hints::default(), &env).unwrap();
            env.image(plan.n_phases())
        };
        let oracle = run(&Trivial
            .decompose(&workflow, &Constraints::default())
            .unwrap());
        let constraints = Constraints {
            budget_bytes: None,
            expected_concurrency: 1,
            model: CostModel::default(),
            block_candidates: vec![3],
            split_axes: vec![0, 1, 2],
            ..Default::default()
        };
        assert_eq!(
            run(&Enumerating::default()
                .decompose(&workflow, &constraints)
                .unwrap()),
            oracle
        );
        assert_eq!(
            run(&Greedy::default()
                .decompose(&workflow, &constraints)
                .unwrap()),
            oracle
        );
    }

    /// **The floor is not the "large means full" rule the barrier predicate was
    /// measured against**, and the way to show it is to run it over the chain
    /// that measurement was taken on.
    ///
    /// `docs/design/GRAPH_MIGRATION.md` §6.5.1 tabulates seven merge steps: four
    /// reduce over the whole volume, two reach a single voxel, and one is
    /// unbounded. `reaches_whole_axis` is an exact comparison rather than a
    /// threshold because of them. The floor changes **nothing** on any of the
    /// seven, and for two separate reasons that are worth keeping apart:
    ///
    /// * on the five whole-volume steps `splittable_axes` has already taken the
    ///   axis away, so there is nothing left for the floor to refuse;
    /// * on the two 1-voxel steps `edge + 1 + 1 < extent` at every candidate that
    ///   cuts anything at all, so every axis stays.
    ///
    /// So the earlier argument does not defeat this one by measurement either,
    /// and not only by the difference between segmenting and cutting.
    #[test]
    fn the_floor_changes_nothing_on_the_chain_the_barrier_rule_was_measured_on() {
        let volume = [512usize, 512, 256];
        let steps: [(&str, Reach); 7] = [
            ("prefix_sum", Reach::all()),
            ("publish_shells", Reach::symmetric([1, 1, 1])),
            ("prefix_sum_kept", Reach::all()),
            ("kept_shells", Reach::symmetric([1, 1, 1])),
            ("centre_id_of", Reach::all()),
            ("join_fragments", Reach::all()),
            ("coordinate_gather", Reach::all()),
        ];
        for (name, reach) in steps {
            for edge in [16usize, 32, 64, 128] {
                assert_eq!(
                    cuttable_axes(&[0, 1, 2], &reach, volume, edge),
                    splittable_axes(&[0, 1, 2], &reach, volume),
                    "{name} at edge {edge}: the floor took an axis the barrier rule left"
                );
            }
        }
    }

    /// **The partition survives the floor, and it is the pricing that saves it.**
    ///
    /// The floor turns a near-full-reach phase into a single block. Left with the
    /// clamp discount that block prices at redundancy `1.0` — cheaper than any
    /// phase that is still cut — and the search fuses the whole chain into it:
    /// seven slots, one phase, one block over the volume, which is the plan with
    /// no parallelism and the largest resident set there is. Charging it on the
    /// infinite grid instead gives back the three phases the chain had before the
    /// floor, with the flanking runs still cut into blocks.
    ///
    /// Both halves are asserted: the partition, and the price that produces it.
    /// The discount would be `1.0` and the charge is strictly above it.
    #[test]
    fn the_partition_survives_the_floor_because_the_dropped_axis_is_still_charged() {
        use crate::probes::IdentityOp;
        let volume = [4096usize, 4, 4];
        let noop = |name: &'static str| Chain::op(IdentityOp::new(name, [0, 0, 0]).with_cost(1.0));
        let chain = Chain::sequence(vec![
            noop("b0"),
            noop("b1"),
            noop("b2"),
            Chain::op(IdentityOp::new("wide", [volume[0] - 1, 0, 0]).with_cost(1.0)),
            noop("a0"),
            noop("a1"),
            noop("a2"),
        ]);
        let constraints = Constraints {
            budget_bytes: None,
            expected_concurrency: 1,
            model: CostModel::default(),
            block_candidates: vec![1024],
            split_axes: vec![0],
            ..Default::default()
        };
        let plan = Enumerating::default()
            .decompose(&Workflow::new(chain, volume, Dtype::F64), &constraints)
            .unwrap();
        assert_eq!(
            plan.phases
                .iter()
                .map(|phase| phase.slots.clone())
                .collect::<Vec<_>>(),
            vec![vec![0, 1, 2], vec![3], vec![4, 5, 6]],
            "the chain fused into the phase the floor left uncut"
        );
        // the flanking phases keep the grid they had: only the phase whose reach
        // denies the cut loses it
        assert_eq!(plan.phases[0].grid.n_blocks(), 4);
        assert_eq!(plan.phases[1].grid.n_blocks(), 1);
        assert_eq!(plan.phases[2].grid.n_blocks(), 4);

        // and the price that keeps it there, in the model's own number
        let single = BlockGrid::along(volume, &[], 1024).unwrap();
        assert_eq!(single.n_blocks(), 1);
        let charged = price_phase(
            &single,
            &Reach::symmetric([volume[0] - 1, 0, 0]),
            1.0,
            1,
            false,
            8.0,
            &CostModel::default(),
            1.0,
            PhaseTraffic::one_in_one_out(),
        )
        .redundancy;
        assert!(
            charged > 1.0,
            "the axis the floor dropped was given the clamp discount: {charged}"
        );
        // a bounded reach is charged strictly under a barrier's 3, whatever its
        // size — `lo + hi < 2 * extent` by definition of bounded
        assert!(
            charged < 3.0,
            "a bounded reach priced as a barrier: {charged}"
        );
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

/// What the floor does to a chain of several ops, rather than to one op.
///
/// **This is where the number comes from, and it is bigger than the one-op case
/// suggests, because fusion adds reaches.** Five ops whose radii are transcribed
/// from the per-op measurement — 0, 1, 9, 5, and a disc of radius 15 across two
/// axes — fuse into one phase with a two-sided halo of 60 on two axes of a
/// 64-cube. That halo covers those axes at the benchmark's own block edge of 32,
/// so before the floor the planner cut all three anyway.
///
/// Measured through `Enumerating`, at `split_axes = [0, 1, 2]` and one candidate
/// edge, comparing the whole plan before wiring against the whole plan after —
/// *total blocks over every phase*, and `exact_read_voxels` summed over every
/// phase divided by the volume:
///
/// | volume | edge | before | after |
/// |---|---|---|---|
/// | `64^3` | 32 | 4 phases, 32 blocks, **6.90x** | 1 phase, 2 blocks, **1.47x** |
/// | `64^3` | 8 | 4 phases, 2048 blocks, **52.8x** | 2 phases, 520 blocks, **5.69x** |
/// | `64^3` | 3 | 5 phases, 53240 blocks, 451.8x | unchanged |
/// | `24 x 20 x 16` | 8 | 4 phases, 72 blocks, **28.1x** | 4 phases, 30 blocks, **7.15x** |
/// | `24 x 20 x 16` | 3 | 3 phases, 678 blocks, **11.1x** | 4 phases, 679 blocks, **7.12x** |
///
/// The third row is the floor declining to fire: at edge 3 no phase's halo covers
/// a 64 axis, every cut still narrows a read, and the plan is left alone. That is
/// the shape of the rule — it removes cuts that buy nothing and does not have an
/// opinion about the rest.
///
/// The row this test pins is the first, because it is the one at the blocking the
/// timing was taken at. It is asserted against the eight-block grid the same
/// phase would have been given, so the comparison is two grids and not a
/// recollection.
#[cfg(test)]
mod chain_floor_measurement {
    use super::*;
    use crate::decomposition::{splittable_axes, CostModel};
    use crate::probes::WindowSumOp;

    #[test]
    fn a_five_op_chain_reads_the_64_cube_1_47_times_where_it_read_it_6_9_times() {
        const VOLUME: [usize; 3] = [64, 64, 64];
        let radii: [(&str, [usize; 3]); 5] = [
            ("clip", [0, 0, 0]),
            ("median", [1, 1, 1]),
            ("deconvolve", [9, 9, 9]),
            ("adaptive", [5, 5, 5]),
            ("tubeness", [15, 15, 0]),
        ];
        let chain = Chain::sequence(
            radii
                .iter()
                .map(|(name, radius)| Chain::op(WindowSumOp::new(name, *radius)))
                .collect(),
        );
        let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
        let constraints = Constraints {
            budget_bytes: None,
            expected_concurrency: 1,
            model: CostModel::default(),
            block_candidates: vec![32],
            split_axes: vec![0, 1, 2],
            ..Default::default()
        };
        let amplification = |plan: &Decomposition| {
            let read: usize = plan.exact_read_voxels().iter().sum();
            read as f64 / VOLUME.iter().product::<usize>() as f64
        };

        let plan = Enumerating::default()
            .decompose(&workflow, &constraints)
            .unwrap();
        assert_eq!(plan.n_phases(), 1, "the chain fuses");
        assert_eq!(plan.phases[0].grid.block(), [64, 64, 32]);
        assert_eq!(plan.phases[0].grid.n_blocks(), 2);
        assert!(
            (amplification(&plan) - 1.469).abs() < 5e-4,
            "{}",
            amplification(&plan)
        );

        // the grid the same phase would have been given without the floor: the
        // fused halo is 60 on two axes of 64, so cutting them re-reads them
        let reach = &plan.phases[0].reach;
        let unfloored = Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases: vec![PhaseDecomposition::derive(
                plan.phases[0].slots.clone(),
                plan.phases[0].names.clone(),
                reach.clone(),
                plan.phases[0].halo.clone(),
                BlockGrid::along(VOLUME, &splittable_axes(&[0, 1, 2], reach, VOLUME), 32).unwrap(),
            )],
            chain_reach: plan.chain_reach,
        };
        assert_eq!(unfloored.phases[0].grid.n_blocks(), 8);
        assert!(
            (amplification(&unfloored) - 5.514).abs() < 5e-4,
            "{}",
            amplification(&unfloored)
        );
    }
}
