// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
//! **A discrete-event simulator for scheduler design.**
//!
//! # What this is for, and what it is not for
//!
//! It **ranks designs. It does not predict runtimes.** Every figure it produces
//! is meaningful only against another figure it produced under the same rates,
//! and a comparison against a wall clock is a category error. That is not
//! modesty about a first version — it follows from the calibration corpus,
//! where the coefficient relating bytes to seconds spans `0.31` to above `4`
//! depending on layout, warmth and codec. A simulator carrying one number for
//! that cannot be right about a duration. It can be right about **which of two
//! orderings finishes first**, because both pay the same wrong coefficient.
//!
//! So the acceptance test for anything here is not "does it match the tile
//! run". It is: *does a change that is known to be an improvement rank as one,
//! and does a change that is known to be neutral rank as neutral.*
//!
//! # Rich in structure, simple in behaviour
//!
//! The owner's constraint, and it decides every trade below: the simulator
//! must be rich enough that the real issues can be worked out in it, and no
//! richer.
//!
//! **Modelled**, because a scheduler decision turns on each:
//!
//! * the **real task DAG** — [`TaskGraph::build`], not a second graph. A
//!   simulator with its own notion of what depends on what would be a
//!   simulator of a different executor, and the divergence would be invisible;
//!   * per-phase **barriers**, which are the ordering the edges do not carry;
//! * **worker slots**, and therefore queueing;
//! * **image residency** — allocation at first write, freeing under the
//!   executor's own rule (`Internal` or released, minus kept), so that the
//!   held-and-dead distinction is visible to a scheduler;
//! * **block working set** — what each in-flight task holds;
//! * a **bounded cache** with LRU eviction over a chunk grid, so that ordering
//!   changes hit rate;
//! * **prefetch depth**, issued on plan rank;
//! * and a **pluggable [`Scheduler`]**, which is the whole point.
//!
//! **Not modelled**, deliberately and by instruction:
//!
//! * noise, jitter, or any distribution at all — one run, one answer;
//! * rates that change over time, thermal or otherwise;
//! * crashes, retries, stragglers, or workers leaving;
//! * storage physics — no seek, no queue depth, no readahead heuristics. What
//!   *is* modelled is a **single serial IO channel**: a byte fetched costs
//!   [`Rates::io_ns_per_byte`] of that one channel's time and a byte in cache
//!   costs nothing. That is the least structure under which prefetch is a
//!   trade rather than free money — without it, deeper prefetching improves
//!   every run without bound and a depth sweep is meaningless;
//! * NUMA, memory bandwidth contention, or any interaction between concurrent
//!   workers other than the slot count itself.
//!
//! Each of those is a place the simulator will be wrong. They are listed
//! because an unmodelled term that is *written down* is a known limit, and one
//! that is merely absent is a silent claim.
//!
//! # The one thing to be careful about
//!
//! Concurrent workers do not contend here, so **wall clock scales down with
//! worker count far more cleanly than a real machine's does.** The tile run
//! measured realised concurrency of `2.41x` against forty requested. A
//! scheduler tuned in here to exploit forty independent workers is tuned
//! against a machine that does not exist. Compare schedulers at a fixed worker
//! count; do not read the worker-count axis as a speed-up curve.
//!
//! `simulate` carries its own outer documentation on `pub mod simulate;` in
//! `lib.rs`, and a merged doc comment resolves its links in the scope of the
//! item rather than of this file, so these two are spelled from the crate root.
//!
//! [`Scheduler`]: crate::simulate::Scheduler
//! [`Rates::io_ns_per_byte`]: crate::simulate::Rates::io_ns_per_byte

use std::collections::BTreeSet;

use crate::assemble::ImageId;
use crate::decomposition::{Decomposition, Visibility};
use crate::distributed::cache_model::{ChunkGrid, ModelledCache};
use crate::error::Result;
use crate::fragment::PhaseWork;
use crate::graph::TaskGraph;

/// The machine, as the simulator understands one.
///
/// Every field is a **planner lever** — something a plan or a caller chooses —
/// rather than a property of the hardware, except `workers`, which is both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Machine {
    /// Slots. Tasks beyond this queue.
    pub workers: usize,
    /// The chunk cache's byte budget, shared across workers.
    ///
    /// Shared and not per-worker, which is the optimistic reading: two workers
    /// reading the same chunk pay for it once. The pessimistic reading is one
    /// cache per worker and no sharing at all. The truth is the page cache and
    /// is neither; this is the reading that makes ordering *matter*, which is
    /// what the simulator is for.
    pub cache_bytes: u64,
    /// How many blocks ahead the prefetcher runs, in plan rank.
    ///
    /// `0` disables it. See [`crate::prefetch::Prefetcher`], whose depth this
    /// mirrors.
    pub prefetch_depth: usize,
}

impl Default for Machine {
    fn default() -> Self {
        Self {
            workers: 1,
            cache_bytes: 0,
            prefetch_depth: 0,
        }
    }
}

/// The measured constants. **Every one of them is a stated parameter, not a
/// fitted one**, and none is trustworthy in absolute terms.
///
/// See the module header: the byte-to-seconds coefficient spans an order of
/// magnitude across layouts in the calibration corpus, so a single value here
/// is a *choice of regime*, and two simulations are comparable only when they
/// share it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rates {
    /// Compute time charged per voxel of a task's **read extent**, since that
    /// is what an op traverses.
    ///
    /// The fallback. [`simulate`] takes a per-phase slice that overrides it,
    /// and supplying one matters more than any other rate here: the tile run
    /// measured phases spanning **`3.541` to `201.397` ns per voxel, a factor
    /// of 57**. Under one uniform rate a throughput term is constant across
    /// every ready task and can discriminate nothing — see
    /// [`RateBasis::PerPhaseCost`], which is *correctly* inert in that case.
    pub compute_ns_per_voxel: f64,
    /// Fetch time per byte, charged only on a cache miss.
    pub io_ns_per_byte: f64,
    /// The stored chunk, which decides both the cache's granularity and how
    /// much a misaligned read over-fetches.
    ///
    /// **Alignment and not chunk count is the cost driver** — the corpus puts
    /// unaligned re-fetches well above aligned ones — and a chunk grid is the
    /// smallest model that can express that at all.
    pub chunk: [usize; 3],
    /// Bytes in one chunk.
    pub chunk_bytes: u64,
}

impl Default for Rates {
    fn default() -> Self {
        Self {
            // The tile run's own figure for a mid-cost stage, so that a default
            // simulation sits in the regime the measurements came from rather
            // than in a round-number one.
            compute_ns_per_voxel: 98.329,
            io_ns_per_byte: 1.0,
            chunk: [64, 64, 64],
            chunk_bytes: 64 * 64 * 64 * 8,
        }
    }
}

/// What one simulated run did.
///
/// **Read the ratios.** `makespan_ns` against another scheduler's is a finding;
/// `makespan_ns` on its own is an artefact of [`Rates`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Outcome {
    pub makespan_ns: u64,
    /// The worst simultaneous total of images plus in-flight block buffers.
    pub peak_bytes: u64,
    /// Bytes actually fetched — misses only. This is the **induced IO** an
    /// ordering causes, which is the quantity an IO penalty is made of.
    pub fetched_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    /// Slot-nanoseconds where a worker had no ready task. The scheduler's own
    /// waste, separated from the work.
    pub idle_slot_ns: u64,
    /// Bytes fetched ahead of demand. Part of [`Self::fetched_bytes`], not
    /// additional to it.
    pub prefetched_bytes: u64,
    /// Nanoseconds a worker spent waiting on the IO channel before its compute
    /// could start. **The quantity prefetch exists to reduce**, and the one a
    /// too-deep prefetch increases by queueing ahead of demand.
    pub io_wait_ns: u64,
    /// Tasks completed. **The conservation law**: it is a property of the plan
    /// and not of the schedule, so any two schedulers on one plan must agree on
    /// it. Cache misses are *not* such a quantity — ordering changes those,
    /// which is the whole reason a scheduler can matter — so this is the
    /// invariant to assert a scheduler against.
    pub tasks_run: u64,
}

impl Outcome {
    /// Worker-seconds of idleness as a fraction of the whole run. A scheduler
    /// that starves is visible here before it is visible in the makespan.
    pub fn idle_fraction(self, workers: usize) -> f64 {
        let total = self.makespan_ns.saturating_mul(workers.max(1) as u64);
        if total == 0 {
            return 0.0;
        }
        self.idle_slot_ns as f64 / total as f64
    }
}

/// What a [`Scheduler`] may look at when it chooses.
///
/// **Deliberately narrow.** Everything here is something the real coordinator
/// knows at handout time — elapsed time, what is running, what is resident,
/// what the cache holds. A field that the real thing could not know would make
/// a scheduler that cannot be shipped.
pub struct Decision<'a> {
    pub now_ns: u64,
    pub graph: &'a TaskGraph,
    pub decomposition: &'a Decomposition,
    /// Task ids that could start now, in ascending id order.
    pub ready: &'a [usize],
    /// Task ids currently occupying a slot.
    pub running: &'a [usize],
    /// Images alive right now, and their sizes.
    pub live_images: &'a [(usize, u64)],
    /// Bytes resident right now: images plus in-flight block buffers.
    pub resident_bytes: u64,
    /// The cache, for a scheduler that wants to prefer a warm task.
    pub cache: &'a ModelledCache,
    pub grid: &'a ChunkGrid,
    /// Compute nanoseconds per voxel, per phase. Always as long as the phase
    /// count — [`simulate`] fills it from [`Rates::compute_ns_per_voxel`] where
    /// the caller supplied nothing, so a scheduler never has to ask which it
    /// got.
    pub phase_ns_per_voxel: &'a [f64],
}

/// Choose which ready task runs next.
///
/// **One method, and it returns an index into `ready` rather than a task id**,
/// so a scheduler cannot return something that was not offered.
pub trait Scheduler {
    fn name(&self) -> &'static str;

    /// Which of `decision.ready` to start. `ready` is never empty.
    fn pick(&mut self, decision: &Decision<'_>) -> usize;
}

/// Plan order: the lowest ready task id.
///
/// **The baseline, and it is what the executor does today** — `strategy::
/// execute` walks the graph in id order as tasks become ready. Every other
/// scheduler is to be judged against this one, and a scheduler that cannot beat
/// it has not earned its complexity.
#[derive(Debug, Default, Clone, Copy)]
pub struct PlanOrder;

impl Scheduler for PlanOrder {
    fn name(&self) -> &'static str {
        "plan-order"
    }

    fn pick(&mut self, _decision: &Decision<'_>) -> usize {
        0
    }
}

/// Prefer the ready task whose reads the cache already holds.
///
/// **The tie-break form of the IO penalty, and deliberately not the additive
/// form.** The corpus can price *how many* bytes an ordering induces but not
/// what a byte costs — the coefficient moves by more than an order of magnitude
/// across layouts — so induced IO enters beneath the throughput term as a
/// tie-break and never as a weight summed into it.
#[derive(Debug, Default, Clone, Copy)]
pub struct WarmestFirst;

impl Scheduler for WarmestFirst {
    fn name(&self) -> &'static str {
        "warmest-first"
    }

    fn pick(&mut self, decision: &Decision<'_>) -> usize {
        let mut best = 0usize;
        let mut fewest = usize::MAX;
        for (slot, &id) in decision.ready.iter().enumerate() {
            let task = &decision.graph.tasks[id];
            let keys = decision.grid.keys(task.phase, &task.geometry.read);
            let misses = decision.cache.misses(&keys);
            // Strictly fewer, so ties keep plan order and the comparison
            // against `PlanOrder` isolates the cache term rather than mixing in
            // an arbitrary reordering of equals.
            if misses < fewest {
                fewest = misses;
                best = slot;
            }
        }
        best
    }
}

/// **What the executor actually does**, both of its policies.
///
/// `strategy::execute` pops a `BinaryHeap<Reverse<([usize; 5], usize)>>` keyed
/// by [`crate::strategy::priority_key`], so it dispatches in ascending key
/// order.
///
/// **That function is called here, not transcribed.** A transcription plus a
/// test comparing the two is a drift *detector*; sharing the definition is a
/// drift *impossibility*, and there is no reason to prefer the weaker one when
/// both types are in this crate. A simulator claiming to model the executor's
/// dispatch order must not carry its own copy of that order.
///
/// **These are the schedulers that matter**, because they are the only two a
/// caller can ask for today. Everything else in this module is a proposal.
#[derive(Debug, Clone, Copy)]
pub struct ExecutorOrder {
    /// `false` is `SchedulePriority::PhaseMajor`, which is `Hints::default()`.
    pub block_major: bool,
}

impl ExecutorOrder {
    /// `SchedulePriority::PhaseMajor` — every block through phase 1, then phase
    /// 2. **The shipped default.**
    pub fn phase_major() -> Self {
        Self { block_major: false }
    }

    /// `SchedulePriority::BlockMajor` — advance one block as far through the
    /// phases as its dependencies allow. Its own doc calls this "fusion, and
    /// the smaller working set".
    pub fn block_major() -> Self {
        Self { block_major: true }
    }

    /// The executor's own key. Ascending, because its heap is `Reverse`-wrapped
    /// and pops the smallest.
    ///
    /// `visit_order` is left at the default here: it permutes which axis is
    /// slowest-varying, which is a second lever, and mixing it into the
    /// phase-against-block comparison would make the result about two changes.
    pub fn key(self, task: &crate::graph::Task) -> [usize; 5] {
        crate::strategy::priority_key(
            task,
            &crate::strategy::Hints {
                priority: if self.block_major {
                    crate::strategy::SchedulePriority::BlockMajor
                } else {
                    crate::strategy::SchedulePriority::PhaseMajor
                },
                ..crate::strategy::Hints::default()
            },
        )
    }
}

impl Scheduler for ExecutorOrder {
    fn name(&self) -> &'static str {
        if self.block_major {
            "executor:block-major"
        } else {
            "executor:phase-major"
        }
    }

    fn pick(&mut self, decision: &Decision<'_>) -> usize {
        let mut best = 0usize;
        let mut best_key = [usize::MAX; 5];
        for (slot, &id) in decision.ready.iter().enumerate() {
            let key = self.key(&decision.graph.tasks[id]);
            if key < best_key {
                best_key = key;
                best = slot;
            }
        }
        best
    }
}

/// **Run as far ahead as the graph allows.** The adversary, not a proposal.
///
/// Prefers the ready task in the **highest** phase, so a worker starts phase
/// `p + 1` the instant one block of phase `p` unblocks it rather than finishing
/// the phase it is in. That allocates the next image early and holds the
/// previous one longer, which is the shape of every residency defect this
/// session chased.
///
/// It exists as a **control**. "Every scheduler reached the same peak" is a
/// finding only if some scheduler could have reached a different one; without a
/// deliberately bad one in the table, an inert peak measurement and a genuine
/// invariance look identical.
#[derive(Debug, Default, Clone, Copy)]
pub struct RunAhead;

impl Scheduler for RunAhead {
    fn name(&self) -> &'static str {
        "run-ahead"
    }

    fn pick(&mut self, decision: &Decision<'_>) -> usize {
        let mut best = 0usize;
        let mut deepest = 0usize;
        for (slot, &id) in decision.ready.iter().enumerate() {
            let phase = decision.graph.tasks[id].phase;
            if phase > deepest {
                deepest = phase;
                best = slot;
            }
        }
        best
    }
}

/// **Order so that memory can be released.**
///
/// The owner's Stage 4 requirement, made runnable: *it is the planner's job to
/// ensure that data can be released by doing things in a sensible order.*
///
/// # What it prefers, and why that is not "minimise peak"
///
/// An image is freed when its **last reader's phase completes** — every task of
/// it, not just the one that touched those voxels. So a phase that is the last
/// reader of a large image is worth *finishing*, and a scheduler that leaves
/// one task of it outstanding while starting a new phase holds a whole volume
/// for no reason. That image is **held and dead**: still allocated, read by
/// nothing.
///
/// The score is therefore *bytes this phase's completion would free, divided by
/// the tasks still standing between here and that completion* — the release per
/// unit of remaining work, which prefers finishing a nearly-done phase that
/// frees a lot over starting a fresh one that frees nothing.
///
/// **This is not peak minimisation and must not become it.** Memory that is
/// being read is a run going fast; only memory that is held and dead is a
/// defect. A scheduler that shrank the working set would be spending time to
/// buy nothing. This one shortens the interval between *last read* and *free*,
/// which costs nothing and is pure gain.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReleaseAware;

impl Scheduler for ReleaseAware {
    fn name(&self) -> &'static str {
        "release-aware"
    }

    fn pick(&mut self, decision: &Decision<'_>) -> usize {
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (slot, &id) in decision.ready.iter().enumerate() {
            let phase = decision.graph.tasks[id].phase;
            // What completing this phase would free: every live image whose
            // last reader is this phase.
            let freed: u64 = decision
                .live_images
                .iter()
                .filter(|&&(image, _)| {
                    decision.decomposition.readers_of_image(image).last() == Some(&phase)
                })
                .map(|&(_, bytes)| bytes)
                .sum();
            // Tasks of this phase still to be dispatched or finished. Counted
            // from the ready and running sets rather than tracked, so this
            // scheduler needs nothing the real coordinator would not have.
            let outstanding = decision
                .ready
                .iter()
                .chain(decision.running.iter())
                .filter(|&&other| decision.graph.tasks[other].phase == phase)
                .count()
                .max(1) as f64;
            let score = freed as f64 / outstanding;
            if score.total_cmp(&best_score) == std::cmp::Ordering::Greater {
                best = slot;
                best_score = score;
            }
        }
        best
    }
}

/// **Greedy throughput over a bounded horizon, with induced IO beneath it.**
///
/// The design §0.2 of the residency plan argues for, made runnable so it can be
/// measured rather than reasoned about.
///
/// # The two terms, and why they are not added together
///
/// The objective is total execution time; peak residency and cache size are
/// boundary conditions. Optimising that globally consumes the least trustworthy
/// part of the model — absolute magnitudes over a long horizon, compounded — so
/// this is greedy with a horizon, and **the horizon bounds the prediction
/// error**.
///
/// * **The throughput term** is voxels per nanosecond of compute. Work is
///   fixed, so this does not prefer cheap stages in any way that changes the
///   total; it prefers the task that keeps the most of the machine busy.
/// * **The IO term is a tie-break, not a summand.** The corpus can price *how
///   many* bytes an ordering induces but not what a byte costs — the
///   coefficient moves from `0.31` to above `4` across layouts — so folding it
///   into the objective would weight the ranking by the one number least worth
///   trusting. Beneath the throughput term it can only choose between tasks
///   the trustworthy term has already called equal.
///
/// # What the horizon is for
///
/// A bounded horizon makes a scheduler blind to any cost whose benefit lands
/// outside it, and IO is the clearest case: a fetch that pays off two tasks
/// later is invisible to a horizon shorter than two tasks. So the horizon has a
/// **derived lower bound** — long enough to contain the fetch it amortises —
/// and [`Self::new`] refuses one below it rather than silently scheduling
/// nonsense.
#[derive(Debug, Clone, Copy)]
pub struct BoundedHorizonThroughput {
    horizon_ns: u64,
    rate: RateBasis,
}

/// What the throughput term is computed over.
///
/// **The distinction is a measured finding, not a knob.** See
/// [`RateBasis::PerBlockReadExtent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateBasis {
    /// Output voxels over this block's own read extent.
    ///
    /// **The obvious reading, and it loses to doing nothing.** A read extent
    /// clamps at the volume boundary while a core does not, so `core / read` is
    /// *higher* for a face block than an interior one — an interior block of a
    /// `16^3` cut with a reach of one reads `18^3` for a `16^3` core, a face
    /// block reads less for the same core. The scheduler therefore prefers
    /// **boundary blocks**, which are exactly the blocks with the fewest
    /// neighbours to share a cached chunk with, and it scatters the traversal
    /// over the volume's surface.
    ///
    /// It is not wrong about any single task — a face block really does retire
    /// more output per voxel read. It is wrong about the run.
    /// `tests/simulate_ranks.rs` measures it inducing **7% more cache misses
    /// than plan order** and 16% more than the IO tie-break alone.
    ///
    /// Kept, and public, because a failure mode that is only described is a
    /// failure mode that comes back.
    PerBlockReadExtent,
    /// The **phase's** cost per voxel, which every block of a phase shares.
    ///
    /// The fix, and the reasoning is that the term should discriminate where
    /// the work genuinely differs — between phases, which cost different
    /// amounts per voxel — and not where it only appears to. Within a phase
    /// every block does the same work, so the clamping difference above is an
    /// artefact of geometry and the ranking should treat those blocks as equal,
    /// leaving the IO tie-break to choose. **The default.**
    PerPhaseCost,
}

impl BoundedHorizonThroughput {
    /// The shortest horizon that can contain one whole-block fetch, from the
    /// rates it will be simulated under.
    ///
    /// **Derived rather than chosen.** Below this the scheduler cannot see a
    /// fetch complete, so it cannot see a prefetch or a cache hit pay for
    /// itself, and it will happily order the run in a way that re-fetches
    /// everything — the "nonsense strategies that ignore cost of IO" a short
    /// horizon invites.
    pub fn floor_ns(rates: &Rates) -> u64 {
        ((rates.chunk_bytes as f64 * rates.io_ns_per_byte) as u64).max(1)
    }

    /// A horizon at or above [`Self::floor_ns`], on [`RateBasis::PerPhaseCost`].
    pub fn new(horizon_ns: u64, rates: &Rates) -> Result<Self> {
        Self::with_basis(horizon_ns, rates, RateBasis::PerPhaseCost)
    }

    /// The same, on a stated basis. See [`RateBasis`].
    pub fn with_basis(horizon_ns: u64, rates: &Rates, rate: RateBasis) -> Result<Self> {
        let floor = Self::floor_ns(rates);
        if horizon_ns < floor {
            return Err(crate::error::Error::InvalidArgument(format!(
                "a horizon of {horizon_ns} ns is shorter than the {floor} ns one chunk fetch \
                 takes at these rates. A scheduler that cannot see a fetch finish cannot see it \
                 pay for itself, and will order the run as though re-reading were free."
            )));
        }
        Ok(Self { horizon_ns, rate })
    }

    /// The horizon this was built with.
    pub fn horizon_ns(self) -> u64 {
        self.horizon_ns
    }
}

impl Scheduler for BoundedHorizonThroughput {
    fn name(&self) -> &'static str {
        "bounded-horizon-throughput"
    }

    fn pick(&mut self, decision: &Decision<'_>) -> usize {
        let mut best = 0usize;
        let mut best_rate = f64::NEG_INFINITY;
        let mut best_misses = usize::MAX;
        for (slot, &id) in decision.ready.iter().enumerate() {
            let task = &decision.graph.tasks[id];
            // Voxels this task retires, over what it costs to retire them. The
            // horizon enters as a cap: work beyond it is not credited, so a
            // very long task cannot win on volume alone.
            let voxels = task.geometry.core.voxels() as f64;
            let cost = match self.rate {
                RateBasis::PerBlockReadExtent => (task.geometry.read.voxels() as u64).max(1) as f64,
                // Every block of a phase shares it, so the term is constant
                // within a phase and discriminates only across phases — which
                // is where the work genuinely differs.
                RateBasis::PerPhaseCost => {
                    (voxels * decision.phase_ns_per_voxel[task.phase]).max(1.0)
                }
            };
            let rate = voxels / cost.min(self.horizon_ns as f64).max(1.0);
            let keys = decision.grid.keys(task.phase, &task.geometry.read);
            let misses = decision.cache.misses(&keys);
            // Strict on the throughput term; the IO term decides only what it
            // leaves equal. `total_cmp` rather than `>`, because this crate does
            // not select between two `f64`s through a partial order.
            let better = match rate.total_cmp(&best_rate) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Equal => misses < best_misses,
                std::cmp::Ordering::Less => false,
            };
            if better {
                best = slot;
                best_rate = rate;
                best_misses = misses;
            }
        }
        best
    }
}

/// Run the plan and report what it did.
///
/// # The event loop, and the one assumption in it
///
/// Time advances to the next completion. At each such point every freed slot is
/// filled, in a loop, before time advances again — so a scheduler is asked
/// repeatedly at one instant and sees each of its own choices reflected in
/// `running` before making the next. That is the property that lets a scheduler
/// reason about a *set* it is assembling rather than one task at a time.
///
/// `released` and `kept` are `Hints::release_images` and `Hints::keep_images`;
/// the image walk applies the executor's rule with them, exactly as
/// [`Decomposition::peak_image_bytes_with`] does.
pub fn simulate(
    decomposition: &Decomposition,
    work: &[PhaseWork<'_>],
    machine: &Machine,
    rates: &Rates,
    released: &BTreeSet<ImageId>,
    kept: &BTreeSet<ImageId>,
    phase_ns_per_voxel: &[f64],
    scheduler: &mut dyn Scheduler,
) -> Result<Outcome> {
    if !phase_ns_per_voxel.is_empty() && phase_ns_per_voxel.len() != decomposition.n_phases() {
        return Err(crate::error::Error::InvalidArgument(format!(
            "simulate: {} per-phase compute rates for a {}-phase plan. A partial list would \
             silently charge some phases the fallback rate, and which ones would depend on the \
             order the plan happened to be assembled in.",
            phase_ns_per_voxel.len(),
            decomposition.n_phases()
        )));
    }
    let phase_rates: Vec<f64> = if phase_ns_per_voxel.is_empty() {
        vec![rates.compute_ns_per_voxel; decomposition.n_phases()]
    } else {
        phase_ns_per_voxel.to_vec()
    };
    let graph = TaskGraph::build(decomposition);
    let dependents = graph.dependents();
    let mut indegree: Vec<usize> = graph.tasks.iter().map(|t| t.n_dependencies()).collect();

    let grid = ChunkGrid::new(decomposition.volume, rates.chunk);
    let mut cache = ModelledCache::new(machine.cache_bytes, rates.chunk_bytes);

    // --- image residency, on the executor's own rule -------------------------
    let bytes_of = |image: usize| -> u64 {
        let volume = decomposition.volume_at(image);
        volume.iter().product::<usize>() as u64 * decomposition.dtype_at(image).size_of() as u64
    };
    let mut live: Vec<(usize, u64)> = vec![(0, bytes_of(0))];
    for image in decomposition.supplied_input_images() {
        live.push((image, bytes_of(image)));
    }

    let mut outcome = Outcome::default();
    // **One serial IO channel.** Not storage physics — there is no seek, no
    // queue depth and no readahead — but a single shared resource with a
    // finite rate, which is the least that makes prefetch a *trade* rather
    // than free money. Without it, deeper prefetch would improve every run
    // without bound, and a scheduler tuned against that would be tuned against
    // a machine with infinite bandwidth.
    let mut io_free_at: u64 = 0;
    let mut now: u64 = 0;
    // (finish_ns, task id), kept sorted so the earliest completion is last.
    let mut running: Vec<(u64, usize)> = Vec::new();
    let mut block_bytes: Vec<u64> = vec![0; graph.tasks.len()];
    let mut in_flight_bytes: u64 = 0;
    let mut done_in_phase: Vec<usize> = vec![0; graph.n_phases()];
    let mut finished_phases: usize = 0;
    let mut remaining = graph.tasks.len();
    // Phases whose output image has been counted as allocated.
    let mut allocated: Vec<bool> = vec![false; decomposition.n_images() + 1];
    // Asked once per phase rather than once per task: it is a property of the
    // phase, and `phase_traffic` returns an error for a phase whose work is
    // missing, which is a thing to find out before the run rather than midway.
    let writes: Vec<bool> = decomposition
        .phases
        .iter()
        .enumerate()
        .map(|(index, phase)| {
            crate::decomposition::phase_traffic(index, phase, work.get(index))
                .map(|traffic| traffic.writes_an_image)
        })
        .collect::<Result<Vec<bool>>>()?;

    while remaining > 0 {
        // Which tasks may start: indegree zero, not started, and — for a
        // barrier phase — every earlier phase complete. The barrier is checked
        // here rather than encoded as edges because that is where the real
        // graph puts it; see `TaskGraph::barriers`.
        let ready: Vec<usize> = (0..graph.tasks.len())
            .filter(|&id| indegree[id] == 0)
            .filter(|&id| {
                let phase = graph.tasks[id].phase;
                !graph.is_barrier(phase) || finished_phases >= phase
            })
            .collect();

        if ready.is_empty() || running.len() >= machine.workers {
            // Advance to the next completion. Nothing can start before then.
            let Some(&(finish, _)) = running.last() else {
                // Nothing ready and nothing running: the graph cannot progress.
                // Reached only by a malformed decomposition, and returning the
                // partial outcome would report a run that did not happen.
                return Err(crate::error::Error::InvalidArgument(format!(
                    "simulate: {remaining} tasks remain, none is ready and none is running. The \
                     task graph has a cycle or a barrier that can never clear."
                )));
            };
            if ready.is_empty() && running.len() < machine.workers {
                outcome.idle_slot_ns += (finish - now) * (machine.workers - running.len()) as u64;
            }
            now = finish;
            while let Some(&(finish_ns, id)) = running.last() {
                if finish_ns != now {
                    break;
                }
                running.pop();
                in_flight_bytes -= block_bytes[id];
                remaining -= 1;
                outcome.tasks_run += 1;
                for &next in &dependents[id] {
                    indegree[next] -= 1;
                }
                let phase = graph.tasks[id].phase;
                done_in_phase[phase] += 1;
                if done_in_phase[phase] == graph.tasks_in_phase(phase).len() {
                    finished_phases = finished_phases.max(phase + 1);
                    // Free what the executor would free after this phase.
                    live.retain(|&(image, _)| {
                        let image_id = ImageId::from(image);
                        let freeable = (decomposition.image_visibility(image)
                            == Visibility::Internal
                            || released.contains(&image_id))
                            && !kept.contains(&image_id);
                        if !freeable {
                            return true;
                        }
                        match decomposition.readers_of_image(image).last() {
                            Some(&last) => last > phase,
                            None => false,
                        }
                    });
                }
            }
            continue;
        }

        let slot = {
            let running_ids: Vec<usize> = running.iter().map(|&(_, id)| id).collect();
            let decision = Decision {
                now_ns: now,
                graph: &graph,
                decomposition,
                ready: &ready,
                running: &running_ids,
                live_images: &live,
                resident_bytes: live.iter().map(|&(_, b)| b).sum::<u64>() + in_flight_bytes,
                cache: &cache,
                grid: &grid,
                phase_ns_per_voxel: &phase_rates,
            };
            scheduler.pick(&decision).min(ready.len() - 1)
        };
        let id = ready[slot];
        let task = &graph.tasks[id];

        // The image this phase writes is allocated when its first block starts.
        //
        // **`writes_an_image` is asked of the work, not assumed**, exactly as
        // `Decomposition::peak_image_bytes_with` asks it: a fragment phase may
        // write no image at all, and a walk that allocated one per phase would
        // over-count every plan with a fragment stage in it — which is every
        // plan this project actually runs.
        let written = task.phase + 1;
        if !allocated[written] && written < decomposition.n_images() && writes[task.phase] {
            allocated[written] = true;
            live.push((written, bytes_of(written)));
        }

        // What this task fetches, and what that costs. A demand fetch queues
        // behind whatever the channel is already carrying — including a
        // prefetch issued for a task that has not started, which is exactly how
        // a too-deep prefetch hurts.
        let keys = grid.keys(task.phase, &task.geometry.read);
        let misses = cache.misses(&keys) as u64;
        cache.note_assigned(&keys);
        outcome.cache_hits += keys.len() as u64 - misses;
        outcome.cache_misses += misses;
        let fetched = misses * rates.chunk_bytes;
        outcome.fetched_bytes += fetched;
        let transfer = (fetched as f64 * rates.io_ns_per_byte) as u64;
        let io_starts = now.max(io_free_at);
        let io_done = io_starts + transfer;
        io_free_at = io_done;
        outcome.io_wait_ns += io_done - now;

        let read_voxels = task.geometry.read.voxels() as u64;
        let bytes_per_voxel = decomposition.dtype_at(task.phase).size_of() as u64;
        // Input tile plus output tile, the same two-buffer figure
        // `PhaseCost::working_set_bytes_per_block` carries, and with the same
        // known gap: a phase reading three images is charged as if it read one.
        block_bytes[id] = read_voxels * bytes_per_voxel * 2;
        in_flight_bytes += block_bytes[id];

        let compute = (read_voxels as f64 * phase_rates[task.phase]) as u64;
        // Indegree is decremented on completion, so mark the task started by
        // making it un-ready. `usize::MAX` cannot be reached by decrementing.
        indegree[id] = usize::MAX;
        // Compute starts when the bytes have landed, not when the slot opened.
        let finish = io_done + compute.max(1);
        running.push((finish, id));
        // Descending by finish time, so the earliest completion is `last`.
        running.sort_by(|a, b| b.0.cmp(&a.0));

        // **Prefetch fills idle channel time and nothing else.**
        //
        // The rule is deliberately the conservative one: a fetch ahead is
        // issued only when the channel is free *now*, so it never displaces a
        // demand fetch that has already been asked for. It can still delay one
        // that arrives during the transfer, and that delay is the cost a depth
        // sweep is looking for — a serial channel is what makes deeper
        // prefetching stop paying rather than improve without bound.
        //
        // Ahead in **plan rank**, which is `TaskGraph`'s own task order and is
        // what `prefetch::Prefetcher` ranks on. That is the point made in the
        // bounded-horizon argument: the prefetcher is immune to the compute
        // scheduler's myopia precisely because it does not consult it.
        if machine.prefetch_depth > 0 {
            let mut issued = 0usize;
            for ahead in graph.tasks.iter().skip(id + 1) {
                if issued == machine.prefetch_depth {
                    break;
                }
                // **Only into idle channel time, and the test is taken once per
                // dispatch rather than once per fetch.** A prefetcher with a
                // queue issues a run of fetches when it finds the channel free;
                // it does not re-ask after each one, or `depth` would mean "at
                // most one outstanding fetch" and every depth above 1 would be
                // inert. That was this model's first form, and a sweep of
                // depths 1 to 64 returned the identical row seven times.
                if issued == 0 && io_free_at > now {
                    break;
                }
                if indegree[ahead.id] == usize::MAX {
                    continue;
                }
                let ahead_keys = grid.keys(ahead.phase, &ahead.geometry.read);
                let ahead_misses = cache.misses(&ahead_keys) as u64;
                if ahead_misses == 0 {
                    continue;
                }
                cache.note_assigned(&ahead_keys);
                let bytes = ahead_misses * rates.chunk_bytes;
                outcome.fetched_bytes += bytes;
                outcome.prefetched_bytes += bytes;
                outcome.cache_misses += ahead_misses;
                // **Queued, not restarted.** Each fetch begins when the one
                // before it ends, so a deep prefetch pushes the channel's free
                // moment far into the future — and the next *demand* fetch
                // queues behind all of it. That delay is the cost of depth, and
                // without accumulating here there is no cost and deeper would
                // be better without bound.
                io_free_at = io_free_at.max(now) + (bytes as f64 * rates.io_ns_per_byte) as u64;
                issued += 1;
            }
        }

        let resident = live.iter().map(|&(_, b)| b).sum::<u64>() + in_flight_bytes;
        outcome.peak_bytes = outcome.peak_bytes.max(resident);
        outcome.makespan_ns = outcome.makespan_ns.max(finish);
    }

    Ok(outcome)
}
