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
//! * **stored bytes**, both to the output and to an intermediate, priced
//!   separately and charged on the same channel the reads use — so a strategy
//!   whose payoff is fewer writes has somewhere to show it;
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
//! * storage physics — no seek, no request coalescing, no readahead heuristics.
//!   What *is* modelled is a finite set of per-node IO channels: fetched bytes
//!   and stored bytes both queue there, while cache hits cost no transfer. That
//!   is the least structure under which prefetch is a trade rather than free
//!   money — without it, deeper prefetching improves every run without bound
//!   and a depth sweep is meaningless;
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

use std::collections::{BTreeMap, BTreeSet};

use crate::assemble::ImageId;
use crate::decomposition::Decomposition;
use crate::distributed::cache_model::{ChunkGrid, ModelledCache};
use crate::error::Result;
use crate::fragment::{PhaseWork, SidecarSize};
use crate::geometry::product3;
use crate::graph::TaskGraph;
use crate::log::{Event, ExecutionLog, Stats};

/// The machine, as the simulator understands one.
///
/// Every field is a **planner lever** — something a plan or a caller chooses —
/// rather than a property of the hardware, except `workers`, which is both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Machine {
    /// **Computers.** `1` is one machine and is everything this simulator
    /// modelled before the field existed.
    ///
    /// # What a node is, and what it is not
    ///
    /// A node is a boundary that three things do not cross, and each of the
    /// three was previously modelled as if it did:
    ///
    /// * **the page cache.** Two computers reading the same chunk fetch it
    ///   twice, whatever either caches. `distributed::cache_model` calls that
    ///   duplicated fetch "the whole of what the handout and the placement
    ///   filter are entitled to lean on", and a simulator with one pool cannot
    ///   see it at all;
    /// * **the IO channel.** Each computer has its own link to storage, so `n`
    ///   nodes fetch on `n` sets of [`Self::io_channels`] rather than
    ///   contending for one;
    /// * **memory bandwidth.** [`Self::contention`] is a worker slowing for the
    ///   *other workers on its own machine*, which is what the coefficient was
    ///   fitted to.
    ///
    /// What it is not is a scheduling unit: [`Self::workers`] is still the total
    /// number of slots, and **worker `w` lives on node `w % nodes`**. Round
    /// robin rather than contiguous blocks so that a worker count which is not a
    /// multiple of the node count spreads rather than piling the remainder onto
    /// one machine, and so that `nodes == 1` leaves every worker where it was.
    ///
    /// The shape to have in mind is 2 to 4, and the field is not bounded: ten
    /// is a cluster this crate is meant to plan for and the arithmetic does not
    /// care.
    pub nodes: usize,
    /// Slots, **over all nodes**. Tasks beyond this queue.
    ///
    /// Unchanged in meaning by [`Self::nodes`], deliberately: every figure this
    /// crate has recorded is at some worker count, and a field that quietly
    /// became per-node would move all of them.
    pub workers: usize,
    /// The byte budget of whatever physically serves a re-read.
    ///
    /// # Which cache this is, decided
    ///
    /// It used to be `cache::ChunkCache`'s budget, and at the time that was a
    /// model of a component nobody constructed: what could physically serve a
    /// re-read on a node was the page cache, sized by free RAM, so the
    /// simulator's central mechanism — ordering changes hit rate — was
    /// parameterised by an axis that did not exist on the machine.
    ///
    /// **`ZarrEnvironment` caches by default now**, so that axis does exist for
    /// a run through storage. The decision below is unchanged and the reason is
    /// worth keeping: the page cache is still what serves a re-read for every
    /// other environment, it is still sized by free RAM rather than by anything
    /// this crate sets, and a simulator parameterised by the *smaller* and more
    /// variable of the two would model the machine less well, not more.
    ///
    /// **The decision taken here is to model what physically serves the
    /// re-read**, which today is the page cache. Two consequences follow and
    /// both matter:
    ///
    /// * it is **not a planner lever**. A plan cannot choose it, a strategy
    ///   cannot trade against it, and a scheduler tuned to a number the run
    ///   cannot set is tuned to nothing. [`Machine::with_page_cache`] sizes it
    ///   from free RAM, which is where it comes from.
    /// * it stays a *field* rather than a constant, because a sweep over it is
    ///   how one finds out how much the answer depends on it — which is a
    ///   different question from what to set it to.
    ///
    /// If `ChunkCache` ever acquires a construction site on the read path, this
    /// becomes that cache's budget and does become a lever. The doc moves then;
    /// the field does not.
    ///
    /// **Per node.** Each computer has its own memory, so `nodes` of them have
    /// `nodes` times this — not a share of it. At [`Self::nodes`] `== 1` that
    /// is the same sentence it always was.
    ///
    /// Shared across the workers *of a node*, which is the optimistic reading:
    /// two threads on one machine reading the same chunk pay for it once. The
    /// pessimistic reading is one cache per worker and no sharing at all, which
    /// is [`Self::cache_shared`] `== false`. The truth is the page cache and is
    /// neither; this is the reading that makes ordering *matter*, which is what
    /// the simulator is for.
    pub cache_bytes: u64,
    /// How many blocks ahead the prefetcher runs, in plan rank.
    ///
    /// `0` disables it. See [`crate::prefetch::Prefetcher`], whose depth this
    /// mirrors.
    pub prefetch_depth: usize,
    /// How many fetches the storage serves at once. `0` and `1` both mean one.
    ///
    /// **Per node**, because each computer has its own link to storage: `n`
    /// nodes fetch on `n` sets of these rather than contending for one.
    ///
    /// **The channel used to be singular and that was a statement about a
    /// device that does not exist.** One serial channel is the least structure
    /// that makes prefetch a trade rather than free money — that argument
    /// stands — but it also says concurrency never helps, which is false of
    /// every filesystem and emphatically false of object storage, where
    /// parallel requests are the only way bandwidth is reached at all.
    pub io_channels: usize,
    /// Whether the cache is one pool or one per worker.
    ///
    /// `true` — the default and the old behaviour — is the **optimistic**
    /// reading, and `Self::cache_bytes`'s own doc says so: two workers reading
    /// the same chunk pay for it once. That is right for threads on one machine
    /// and wrong for `distributed`, where a chunk two nodes both read costs two
    /// fetches whatever either caches. `distributed::cache_model` calls that
    /// duplicated fetch "the whole of what the handout and the placement filter
    /// are entitled to lean on" — and a simulator that shares one cache cannot
    /// see it at all, so it cannot rank a handout policy.
    ///
    /// **Within a node.** With [`Self::nodes`] above one there is a pool per
    /// node whatever this says; this decides whether the workers *of* a node
    /// share theirs. So the three arrangements are: one pool (one node,
    /// shared), one pool per computer (many nodes, shared — the physical one),
    /// and one pool per slot (not shared), which is the pessimistic reading.
    ///
    /// `false` gives each worker `cache_bytes / workers-per-node` and counts a
    /// chunk fetched by a second **pool** as [`Outcome::duplicated_fetches`].
    pub cache_shared: bool,
    /// The share of [`Self::cache_bytes`] held as **encoded** chunks, whose hits
    /// cost a decode. See [`Machine::with_encoded_fraction`].
    pub encoded_fraction: f64,
    /// Whether a phase waits for **all** of the one before it.
    ///
    /// `false` — the default, and everything this simulator has ever modelled —
    /// dispatches continuously: a task starts the moment its own dependencies
    /// are met, so phases overlap. `true` is what `strategy::execute` actually
    /// does: it pops a wave, runs it, and **joins the whole wave** before the
    /// next.
    ///
    /// # Why the difference is a field and not a detail
    ///
    /// `docs/design/planner-gaps.md` carries it as item **C** — "the simulator
    /// and the executor have different concurrency models, and neither states
    /// it" — and until there was a field, only one of them could be simulated.
    /// It is not benign: measured at **0.2%** of makespan when nothing contends
    /// and **47%** when something does, because overlapping phases put more
    /// workers on a node and every one of them slows the others through
    /// [`Self::contention`]. A plan the continuous model calls bad can be the
    /// plan the executor runs fastest.
    ///
    /// The mechanism is the one [`crate::graph::TaskGraph::barriers`] already
    /// has: a barrier phase waits for every earlier phase to finish. This makes
    /// **every** phase such a phase, which is why it costs nothing to model — the
    /// ready set already knew how to hold a task back and release it.
    pub wave_synchronous: bool,
    /// How much a worker's compute slows for each *other* worker running **on
    /// its own node**.
    ///
    /// Per node because that is what the coefficient measures: memory bandwidth
    /// and last-level cache are a computer's, and a worker on another computer
    /// takes none of either. At [`Self::nodes`] `== 1` this is the count it
    /// always was.
    ///
    /// `0.0` is the shipped default and is the old behaviour exactly: concurrent
    /// workers do not contend, and wall clock scales down with worker count far
    /// more cleanly than a real machine's does.
    ///
    /// **The number to put here is measured and is not small.** The tile run
    /// realised a concurrency of **`2.41x` against forty requested**; under
    /// Amdahl's form — a worker's duration scaled by `1 + a x (running - 1)` —
    /// that is `a` near [`MEASURED_CONTENTION`]. A scheduler tuned at `0.0`
    /// against forty independent workers is tuned against a machine nobody has.
    ///
    /// It stays off by default because every figure this crate has recorded
    /// about the simulator was taken without it, and a default that silently
    /// moved them would make the record unreadable. Turn it on deliberately.
    pub contention: f64,
    /// How many ready tasks a [`Scheduler`] is shown at one dispatch. **`0`
    /// means all of them**, and `0` is the default.
    ///
    /// # The term this bounds
    ///
    /// [`Decision::ready`] is *every* task that could start, and every
    /// scheduler in this crate walks the whole of it: [`ExecutorOrder`] keys
    /// each candidate, [`WarmestFirst`] asks the cache about each,
    /// [`BoundedHorizonThroughput`] walks each task's chunk keys, [`Handout`]
    /// hands the entire slice to `distributed::handout::choose`. One dispatch
    /// therefore costs `O(ready)` and a run of `T` tasks costs `O(T x R)`
    /// however cheap the loop around it is.
    ///
    /// That is the *second* quadratic term in the event loop. The first — the
    /// readiness scan, a `(0..tasks).filter(..)` per dispatch — is gone, and
    /// removing it exposed this one as the larger. Measured on a three-phase
    /// pixel chain over a `128^3` volume at four workers, release build, by
    /// `tests/planner_arena.rs::print_the_cost_of_the_dispatch_loop`:
    ///
    /// ```text
    ///      tasks   ExecutorOrder (ms)   an O(1) control (ms)   inside the scheduler
    ///      1 536                 13.9                    9.0                   35 %
    ///     12 288                424.4                   35.6                   92 %
    ///     98 304             36 298.9                  567.2                   98 %
    /// ```
    ///
    /// A window caps `R`. A dispatch costs `O(min(ready, window))`, so a run
    /// stops being quadratic the moment the ready set outgrows the window, and
    /// the saving is the whole of the 98% above minus what the window keeps.
    ///
    /// # Why this is the right place to cut, and not the `Scheduler` trait
    ///
    /// The alternative is to make each scheduler cheaper — an index, an
    /// incremental heap, a per-policy shortcut. That is one change per
    /// scheduler, it is a different change for each, and it would have to be
    /// re-argued for every policy added afterwards. A window is one change that
    /// bounds all of them, and it bounds the ones not written yet.
    ///
    /// It is also the constraint a **real** coordinator has. `Decision` is
    /// documented as deliberately narrow — "everything here is something the
    /// real coordinator knows at handout time" — and a coordinator that must
    /// rank every runnable block in a million-block plan before handing one out
    /// is not a coordinator anybody would ship. A window is that limit written
    /// down, and a policy measured under one is a policy that survives it.
    ///
    /// # `0` is the machine this crate has always simulated
    ///
    /// Bit for bit: at `0` the scheduler is handed exactly the slice it was
    /// handed before this field existed, so every figure recorded anywhere in
    /// this crate stands unmoved. That is the whole reason the default is `0`
    /// and not a number somebody liked: a default that quietly re-ordered runs
    /// would make the recorded corpus unreadable, and there would be no way to
    /// tell which figures were taken under which machine.
    ///
    /// The same argument as [`Self::contention`], which is off by default for
    /// exactly this reason and is not off because zero contention is realistic.
    ///
    /// # Which n, and why it must be stated
    ///
    /// The **first** `n` in **ascending task id**, which is the order the ready
    /// set is maintained in. Not a sample, not the most recently admitted, not
    /// the tail: a window whose membership depended on admission order or on a
    /// hash would make two runs of one plan schedule differently, and the
    /// simulator's whole use is comparing one run against another. Ascending id
    /// is also the order a truncation degrades *gracefully* in — it is plan
    /// order, which is what the executor does today, so the tasks a window hides
    /// are the ones the executor would have run last anyway.
    ///
    /// A scheduler is not told it is windowed and does not need to be: it sees
    /// a shorter `ready` and picks an index into it, which is the same contract.
    /// A term computed *over* `ready` — [`ReleaseAware`]'s count of a phase's
    /// outstanding tasks is the one in this crate — is computed over the window,
    /// which is the point rather than a defect: a windowed scheduler is one that
    /// reasons about the part of the machine it can see.
    ///
    /// # What it costs, in numbers
    ///
    /// `tests/candidate_window.rs::print_what_a_window_costs` is the
    /// measurement and carries the whole table. The three figures a caller
    /// setting this field needs, from a three-phase pixel chain over a `128^3`
    /// volume at 98 304 tasks, four computers of one worker:
    ///
    /// ```text
    ///            scheduler   window   wall (ms)     makespan   misses
    /// executor:phase-major     none     39293.2    807820792    35328   identical
    /// executor:phase-major      256       797.7    807820792    35328   identical
    ///        nearest-first     none     35935.8    621438131    12569   identical
    ///        nearest-first     4096      8000.5    616873687    12016   makespan  -0.73%
    ///        nearest-first     1024      1608.3    648578620    15889   makespan  +4.37%
    ///        nearest-first      256       812.0    737693332    26765   makespan +18.71%
    /// ```
    ///
    /// * **49x, for nothing**, on [`ExecutorOrder::phase_major`] — whose argmin
    ///   is the lowest ready id, which every prefix contains, so its schedule is
    ///   bit-identical at every window;
    /// * **the safe window is not a constant.** The same `256` improves the
    ///   12 288-task run by 4% and costs the 98 304-task run 19%, because a
    ///   window is a fraction of the ready set whether or not it is written as
    ///   one. That is the reason the default is `0` rather than a number: there
    ///   is no constant that is right at two plan sizes;
    /// * **a window is a locality prior, not an optimisation.** It confines a
    ///   scheduler to tasks adjacent in plan order, which improves the policies
    ///   that scatter — `block_major` and [`ReleaseAware`] gain ~30% of makespan
    ///   and ~80% of misses — and penalises the ones that were already local.
    ///   Two policies may therefore be compared only at the **same** window.
    ///
    /// `scenario::Scenario` deliberately does not carry this field through its
    /// JSON: the committed `costs/` files are compared byte for byte against
    /// what `Scenario::to_json` writes, every one of them was recorded
    /// unbounded, and a serialiser change would rewrite all of them to state the
    /// default.
    pub candidate_window: usize,
}

/// The contention coefficient the tile run implies, for callers who want the
/// measured machine rather than the ideal one.
///
/// From `2.41x` realised against forty requested: Amdahl's `S(n) = n / (1 + a x
/// (n - 1))` at `S(40) = 2.41` gives `a = (40 / 2.41 - 1) / 39`, near `0.40`.
/// **One parameter, fitted to one figure**, which is all the evidence there is —
/// it is not a model of caches, memory bandwidth or NUMA, and calling it one
/// would be claiming a shape nobody measured.
pub const MEASURED_CONTENTION: f64 = 0.40;

impl Machine {
    /// This machine's own free memory as the cache budget, because that is what
    /// serves a re-read here.
    ///
    /// Deliberately **not** `Default`: a figure taken from the machine the test
    /// happens to run on would make every recorded simulator number
    /// unreproducible, and the whole file is figures compared against other
    /// figures. A caller who wants the real machine asks for it.
    pub fn with_page_cache(self) -> Self {
        Self {
            cache_bytes: crate::budget::default_budget_bytes(),
            ..self
        }
    }

    /// How much of [`Self::cache_bytes`] holds **encoded** chunks.
    ///
    /// See [`Rates::decode_ns_per_byte`] and `cache::Tier`: the real cache has
    /// two, and `cache.rs` records an encoded hit at **962 us — ~100x a decoded
    /// hit**, still ~40x cheaper than storage. A simulator with one tier and
    /// free hits makes a cache-size sweep monotone by construction, where the
    /// real curve has a knee — more capacity buys more *encoded* residency and
    /// hits get two orders of magnitude dearer.
    ///
    /// `0.0` is the old behaviour: one tier, hits free.
    pub fn with_encoded_fraction(self, fraction: f64) -> Self {
        Self {
            encoded_fraction: fraction.clamp(0.0, 1.0),
            ..self
        }
    }
}

impl Default for Machine {
    fn default() -> Self {
        Self {
            nodes: 1,
            wave_synchronous: false,
            workers: 1,
            cache_bytes: 0,
            prefetch_depth: 0,
            io_channels: 1,
            cache_shared: true,
            encoded_fraction: 0.0,
            contention: 0.0,
            // Unbounded, so that a `Machine` nobody has configured is the one
            // every recorded figure was taken on. See the field.
            candidate_window: 0,
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
    /// Fixed cost of one fetch, whatever its size.
    ///
    /// `0.0` is the old behaviour: cost proportional to bytes and nothing else,
    /// under which a small chunk is free and a chunk-size sweep improves
    /// monotonically toward zero. On a filesystem, and far more so on object
    /// storage, the per-request cost is what decides the floor.
    ///
    /// Charged **per chunk fetched**, because that is what the store serves: a
    /// Zarr read of a block spanning `N` chunks is `N` objects, however few
    /// `Environment::read` calls the caller made.
    ///
    /// It is therefore finer than the executor's own observability — `RegionRead`
    /// is emitted once per `read`, carrying `chunks` as a count — which is why
    /// this is a rate a caller states rather than a figure
    /// `tests/simulator_against_the_executor.rs` can compare against a run.
    pub io_latency_ns: f64,
    /// Decode time per fetched byte, on the CPU, between the transfer and the
    /// compute.
    ///
    /// `zarr_env` reads gzip, so a fetched chunk costs a decode proportional to
    /// its bytes. `0.0` is the old behaviour, under which a codec's ratio is
    /// free and choosing one is not a trade.
    pub decode_ns_per_byte: f64,
    /// Store time per byte, for a write whose destination is the **workflow
    /// output**.
    ///
    /// Separate from [`Self::materialise_ns_per_byte`] for the reason
    /// `statistics::Term` keeps `Write` and `Materialise` apart: an intermediate
    /// compresses differently from an output, so one number over-values fusing
    /// late stages. The planner's `CostModel` has carried both since before this
    /// field existed; the simulator charged neither, which made every decision
    /// whose payoff is *fewer writes* — fusion against materialisation, keep
    /// against release, block-major against phase-major — invisible to it.
    pub write_ns_per_byte: f64,
    /// Store time per byte, for a write whose destination is an
    /// **intermediate**. See [`Self::write_ns_per_byte`].
    pub materialise_ns_per_byte: f64,
    /// The stored chunk, which decides both the cache's granularity and how
    /// much a misaligned read over-fetches.
    ///
    /// **Alignment and not chunk count is the cost driver** — the corpus puts
    /// unaligned re-fetches well above aligned ones — and a chunk grid is the
    /// smallest model that can express that at all.
    pub chunk: [usize; 3],
    /// Fallback bytes in one chunk.
    ///
    /// The simulator derives per-image transfer bytes from this chunk shape and
    /// each image's dtype. This scalar remains for callers that need one
    /// representative chunk size, such as bounded-horizon floors and tests that
    /// size a cache in chunks.
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
            io_latency_ns: 0.0,
            decode_ns_per_byte: 0.0,
            // The same seeds `CostModel` ships for `Write` and `Materialise`:
            // chosen for having the ordering right against the read, not the
            // scale. `Snapshot::calibrate` is what moves them.
            write_ns_per_byte: 1.0,
            materialise_ns_per_byte: 1.0,
            chunk: [64, 64, 64],
            chunk_bytes: 64 * 64 * 64 * 8,
        }
    }
}

impl Rates {
    /// Measured rates, from a recorded [`crate::statistics::Snapshot`].
    ///
    /// **The loop, closed.** `Snapshot::calibrate` has always refitted the
    /// planner's `CostModel` from real runs; nothing turned the same evidence
    /// into a `Rates`, so a simulation ran on constants somebody typed out of a
    /// table — `TILE_PHASE_RATES` in the acceptance suite was three numbers from
    /// a run dated `2026-08-23` — which will rot without anything failing.
    ///
    /// Each field falls back to the corresponding field of `seed` when the
    /// snapshot has no **believable** coefficient for it, on
    /// [`crate::statistics::Snapshot::believable`]'s definition: fewer than
    /// [`crate::statistics::REPRODUCTIONS`] runs is treated exactly as never
    /// having seen it, because "seen once" and "reproduced" are different
    /// claims. [`crate::statistics::Snapshot::provenance`] is how a caller finds
    /// out which it got, and this function deliberately does not fold that away.
    ///
    /// **`bytes_per_voxel` is an argument because a rate is per byte and the
    /// terms are per voxel.** `Term::ReadBytes` is the one exception and is used
    /// directly where it exists — it is documented as diagnostic-only, and it is
    /// exactly `io_ns_per_byte`. `Write` and `Materialise` are per voxel, so
    /// they are divided by the width of the element the run stored. A run whose
    /// images have different widths has no single answer, which is the same
    /// caveat `Term::ReadBytes`'s own doc records.
    pub fn from_snapshot(
        snapshot: &crate::statistics::Snapshot,
        seed: &Rates,
        bytes_per_voxel: f64,
    ) -> Self {
        use crate::statistics::Term;
        let per_byte = |term: Term, fallback: f64| -> f64 {
            match snapshot.believable(&term) {
                Some(c) if bytes_per_voxel > 0.0 && c.nanos_per_unit.is_finite() => {
                    c.nanos_per_unit / bytes_per_voxel
                }
                _ => fallback,
            }
        };
        Rates {
            compute_ns_per_voxel: snapshot
                .believable(&Term::Compute)
                .map(|c| c.nanos_per_unit)
                .filter(|n| n.is_finite())
                .unwrap_or(seed.compute_ns_per_voxel),
            io_ns_per_byte: snapshot
                .believable(&Term::ReadBytes)
                .map(|c| c.nanos_per_unit)
                .filter(|n| n.is_finite())
                .unwrap_or_else(|| per_byte(Term::Read, seed.io_ns_per_byte)),
            write_ns_per_byte: per_byte(Term::Write, seed.write_ns_per_byte),
            materialise_ns_per_byte: per_byte(Term::Materialise, seed.materialise_ns_per_byte),
            // Not measurements: the chunk geometry and the two terms a
            // snapshot has no coefficient for are carried from the seed.
            io_latency_ns: seed.io_latency_ns,
            decode_ns_per_byte: seed.decode_ns_per_byte,
            chunk: seed.chunk,
            chunk_bytes: seed.chunk_bytes,
        }
    }
}

/// Per-phase compute rates from the **per-op-family** coefficients a run
/// recorded.
///
/// `Term::ComputeOf` is keyed by slot name and documents itself as "recorded and
/// reported, **not used**" — because `CostModel` has one `compute_scale` and
/// nowhere to put a per-family correction. `simulate` is the consumer that does
/// have somewhere: it takes one rate per phase, and the tile run measured phases
/// spanning a factor of **57**, which one uniform rate cannot express at all.
///
/// A phase's rate is `sum over its slots of declared x measured`, where
/// `declared` is the slot's own `cost_per_voxel` — so the shipped constants'
/// absolute scale stays irrelevant, exactly as `Term::Compute` intends, while
/// their *ratios* carry through. A slot whose family has no believable
/// coefficient falls back to the run-wide `Term::Compute`, and if that is
/// missing too to `seed`.
pub fn phase_rates_from_snapshot(
    snapshot: &crate::statistics::Snapshot,
    decomposition: &Decomposition,
    slots: &[&crate::op::Chain],
    seed: f64,
) -> Vec<f64> {
    use crate::statistics::Term;
    let overall = snapshot
        .believable(&Term::Compute)
        .map(|c| c.nanos_per_unit)
        .filter(|n| n.is_finite())
        .unwrap_or(seed);
    decomposition
        .phases
        .iter()
        .map(|phase| {
            let mut rate = 0.0;
            for (position, &slot) in phase.slots.iter().enumerate() {
                let declared = slots
                    .get(slot)
                    .map(|chain| chain.cost_per_voxel())
                    .unwrap_or(1.0);
                let measured = phase
                    .names
                    .get(position)
                    .and_then(|name| snapshot.believable(&Term::ComputeOf(name.clone())))
                    .map(|c| c.nanos_per_unit)
                    .filter(|n| n.is_finite())
                    .unwrap_or(overall);
                rate += declared * measured;
            }
            // A phase with no chain slot — fragment, iterative — has no family
            // to look up, so it gets the run-wide figure rather than zero.
            if rate > 0.0 {
                rate
            } else {
                overall
            }
        })
        .collect()
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
    /// Bytes stored to the **workflow output**.
    ///
    /// **A property of the plan, not of the schedule**, like
    /// [`Self::tasks_run`]: every block's valid region is written exactly once
    /// whatever the order. Two schedulers on one plan must agree on it, which is
    /// what makes it an invariant to assert against rather than a finding.
    pub written_bytes: u64,
    /// Bytes stored to an **intermediate** image. The same invariant as
    /// [`Self::written_bytes`], and separate for the reason
    /// [`Rates::materialise_ns_per_byte`] is separate.
    pub materialised_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    /// Slot-nanoseconds where a worker had no ready task. The scheduler's own
    /// waste, separated from the work.
    pub idle_slot_ns: u64,
    /// Bytes fetched ahead of demand. Part of [`Self::fetched_bytes`], not
    /// additional to it.
    pub prefetched_bytes: u64,
    /// Nanoseconds a worker spent waiting on an IO channel before a transfer
    /// could start. **The read-side quantity prefetch exists to reduce**, and
    /// the one a too-deep prefetch increases by queueing ahead of demand.
    pub io_wait_ns: u64,
    /// Tasks completed. **The conservation law**: it is a property of the plan
    /// and not of the schedule, so any two schedulers on one plan must agree on
    /// it. Cache misses are *not* such a quantity — ordering changes those,
    /// which is the whole reason a scheduler can matter — so this is the
    /// invariant to assert a scheduler against.
    pub tasks_run: u64,
    /// Chunks one cache pool fetched that **another pool had already fetched**.
    ///
    /// Zero when there is only one pool — one node with a shared cache — because
    /// then the question does not arise. **The quantity a handout policy exists
    /// to reduce**, and the one
    /// `nearest_first_handout_costs_fewer_duplicated_fetches_than_naive_pull`
    /// measures on the real coordinator.
    ///
    /// **A re-fetch by the *same* pool is not one of these.** That is capacity
    /// pressure — the pool evicted it and had to go back — and it is already
    /// counted in [`Self::cache_misses`]. This counted it too until
    /// `Machine::nodes` existed, at which point the conflation stopped being
    /// harmless: on the fixture in `tests/multiple_computers.rs` a single node
    /// with a small cache reported 270 "duplicated" fetches, which would have
    /// made two thirds of the two-node figure eviction rather than duplication
    /// and the whole measurement unreadable.
    pub duplicated_fetches: u64,
    /// Sidecar bytes a fragment phase's blocks wrote, from the **declared**
    /// bound on each stream.
    ///
    /// Zero for a stream that declares `SidecarSize::Unstated`, which is most of
    /// them: an undeclared stream is one nothing can budget, and counting it as
    /// nothing is the visible form of that rather than a claim it is empty.
    pub sidecar_bytes_written: u64,
    /// The largest total a barrier held at once — every contributing block's
    /// fragment, resident together, which is the peak nothing was budgeting.
    pub sidecar_gather_peak: u64,
    /// Chunks served from the **encoded** tier: a hit, but one that paid a
    /// decode. Part of [`Self::cache_hits`], not additional to it.
    pub encoded_hits: u64,
    /// Tasks whose read extent was uniform, so the executor skipped their work.
    ///
    /// The counterpart of `Stats::tasks_short_circuited`. **Not** a conservation
    /// law: it is a property of the plan and the data, so two schedulers agree
    /// on it, but two *decompositions* of one volume do not.
    pub tasks_short_circuited: u64,
    /// The sum over phases of each phase's own **span**: from its first task
    /// starting to its last task finishing.
    ///
    /// **The quantity the planner's objective assumes it knows.**
    /// `strategy::phase_makespan` prices a phase on its own and the partition
    /// search adds the phases up, which is the wall clock only if no two phases
    /// are ever running at once. The `TaskGraph` says otherwise: a block of
    /// phase `p + 1` depends on the blocks of phase `p` that cover its read
    /// extent and on nothing else, so it starts while the rest of phase `p` is
    /// still going. `docs/design/planner-gaps.md` carries this as **G2**, and
    /// this field is what puts a number on it — see [`Outcome::phase_overlap`].
    ///
    /// A span is not a phase's busy time: it contains whatever idleness fell
    /// inside it. That is the right shape for the comparison, because the term
    /// it is being compared against — a phase priced alone — contains the same
    /// idleness by construction.
    pub phase_span_ns: u64,
}

impl Outcome {
    /// **How much the phases overlapped**: [`Self::phase_span_ns`] over the
    /// makespan.
    ///
    /// `1.0` is a run whose phases were strictly sequential, which is what the
    /// planner's objective assumes every run is; above `1.0` is a run that
    /// pipelined, and the excess is the wall clock the sequential-phase
    /// assumption over-charges. Below `1.0` is possible and means the spans did
    /// not cover the run — a plan that spent time between phases rather than
    /// inside one.
    ///
    /// `None` for a run of no length, where the ratio is not a number.
    pub fn phase_overlap(self) -> Option<f64> {
        (self.makespan_ns > 0).then(|| self.phase_span_ns as f64 / self.makespan_ns as f64)
    }

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

/// The per-phase figures that come from a **measurement**, not from the plan.
///
/// Both are functions of the data, and neither can be declared: a rate is a
/// property of the machine and the op together, and a substage count is a fixed
/// point over the volume. `statistics::Snapshot` is where they come from —
/// `Stats::substages` reports the second on every real run — and the two travel
/// together because they are consumed together, and because `simulate` was
/// already at the argument count clippy complains about.
#[derive(Debug, Clone, Copy, Default)]
pub struct PerPhase<'a> {
    /// Compute nanoseconds per voxel of a task's read extent, per phase.
    ///
    /// Empty falls back to [`Rates::compute_ns_per_voxel`] for every phase.
    /// Supplying one matters more than any other rate: the tile run measured
    /// phases spanning **`3.541` to `201.397` ns per voxel, a factor of 57**,
    /// and under one uniform rate a throughput term can discriminate nothing.
    pub ns_per_voxel: &'a [f64],
    /// Substages each phase ran, as [`crate::log::Stats::substages`] reports
    /// them. Empty, or a zero, means one.
    ///
    /// **Measured and not declared, and the crate has the evidence for why that
    /// is enough.** An iterative phase runs to convergence, so its count is in
    /// no reach, no image allocation and no phase structure —
    /// `IterativeOp::limit` is a *bound*, deliberately, and there is no method
    /// for the count. But `iterate`'s own sweep found the count **does not vary
    /// with the block edge**: thirteen lattices including `[1, 1, 1]`, four
    /// reaches and two data shapes, the whole-volume count every time. So it is
    /// constant across every lever a comparison varies, cancels between arms the
    /// way `Rates`'s wrong coefficient does, and needs no model of the op — one
    /// integer per phase, from a run.
    pub substages: &'a [usize],
    /// The fraction of a phase's blocks whose read extent is uniform, so that
    /// `BlockOp::constant_maps_to` lets the executor skip the work.
    ///
    /// **A model of the data, and it cannot be anything else.** Whether a block
    /// short-circuits depends on the volume *and* on the grid — a finer cut
    /// produces more uniform blocks — so no single measured number transfers
    /// between two decompositions, which is exactly the lever a block ladder
    /// sweeps. A phase whose ops decline `constant_maps_to` has a fraction of
    /// zero, and empty means zero everywhere, which is what this modelled
    /// before the field existed: every task charged in full, so the simulator
    /// could not see the thing `constant_maps_to` exists to buy and over-charged
    /// finer cuts systematically.
    ///
    /// **Which blocks** is a deterministic function of the block index, so two
    /// runs of one plan skip the same set and a scheduler cannot be rewarded for
    /// reordering into luck.
    pub constant_fraction: &'a [f64],
}

/// A configured simulator run.
///
/// This is a named-argument layer over [`simulate`]. The free function remains
/// the implementation; this type exists so callers can state only the machine,
/// rates, residency and per-phase measurements they mean to vary.
#[derive(Clone)]
pub struct Run<'a, 'work> {
    decomposition: &'a Decomposition,
    work: &'a [PhaseWork<'work>],
    machine: Machine,
    rates: Rates,
    released: BTreeSet<ImageId>,
    kept: BTreeSet<ImageId>,
    per_phase: PerPhase<'a>,
}

impl<'a, 'work> Run<'a, 'work> {
    pub fn new(decomposition: &'a Decomposition, work: &'a [PhaseWork<'work>]) -> Self {
        Self {
            decomposition,
            work,
            machine: Machine::default(),
            rates: Rates::default(),
            released: BTreeSet::new(),
            kept: BTreeSet::new(),
            per_phase: PerPhase::default(),
        }
    }

    pub fn machine(mut self, machine: Machine) -> Self {
        self.machine = machine;
        self
    }

    pub fn workers(mut self, workers: usize) -> Self {
        self.machine.workers = workers;
        self
    }

    pub fn rates(mut self, rates: Rates) -> Self {
        self.rates = rates;
        self
    }

    pub fn release_images(mut self, images: impl IntoIterator<Item = ImageId>) -> Self {
        self.released = images.into_iter().collect();
        self
    }

    pub fn keep_images(mut self, images: impl IntoIterator<Item = ImageId>) -> Self {
        self.kept = images.into_iter().collect();
        self
    }

    pub fn images(
        mut self,
        released: impl IntoIterator<Item = ImageId>,
        kept: impl IntoIterator<Item = ImageId>,
    ) -> Self {
        self.released = released.into_iter().collect();
        self.kept = kept.into_iter().collect();
        self
    }

    pub fn ns_per_voxel(mut self, rates: &'a [f64]) -> Self {
        self.per_phase.ns_per_voxel = rates;
        self
    }

    pub fn substages(mut self, substages: &'a [usize]) -> Self {
        self.per_phase.substages = substages;
        self
    }

    pub fn constant_fraction(mut self, fractions: &'a [f64]) -> Self {
        self.per_phase.constant_fraction = fractions;
        self
    }

    pub fn per_phase(mut self, per_phase: PerPhase<'a>) -> Self {
        self.per_phase = per_phase;
        self
    }

    pub fn go(self, scheduler: &mut dyn Scheduler) -> Result<Outcome> {
        simulate(
            self.decomposition,
            self.work,
            &self.machine,
            &self.rates,
            &self.released,
            &self.kept,
            self.per_phase,
            scheduler,
        )
    }
}

/// Owned per-phase simulator inputs derived from a real run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeasuredPerPhase {
    pub ns_per_voxel: Vec<f64>,
    pub substages: Vec<usize>,
    pub constant_fraction: Vec<f64>,
}

impl MeasuredPerPhase {
    /// Derive data-dependent simulator inputs from executor stats.
    pub fn from_stats(decomposition: &Decomposition, stats: &Stats) -> Self {
        let mut measured = Self::from_log(decomposition, &stats.log);
        measured.substages = stats.substages.clone();
        measured
    }

    /// Derive per-phase short-circuit fractions from the execution log.
    pub fn from_log(decomposition: &Decomposition, log: &ExecutionLog) -> Self {
        let n = decomposition.n_phases();
        let mut admitted = vec![0usize; n];
        let mut short = vec![0usize; n];
        for event in log.events() {
            match event {
                Event::TaskAdmitted { phase, .. } if phase < n => admitted[phase] += 1,
                Event::BlockShortCircuited { phase, .. } if phase < n => short[phase] += 1,
                _ => {}
            }
        }
        let constant_fraction = admitted
            .iter()
            .zip(short)
            .map(|(&tasks, skipped)| {
                if tasks == 0 {
                    0.0
                } else {
                    skipped as f64 / tasks as f64
                }
            })
            .collect();
        Self {
            ns_per_voxel: Vec::new(),
            substages: Vec::new(),
            constant_fraction,
        }
    }

    pub fn as_per_phase(&self) -> PerPhase<'_> {
        PerPhase {
            ns_per_voxel: &self.ns_per_voxel,
            substages: &self.substages,
            constant_fraction: &self.constant_fraction,
        }
    }
}

/// Whether a block short-circuits, from a fraction and its index.
///
/// A hash rather than a stride, so that the skipped set is not a plane or a
/// lattice a traversal order could exploit; deterministic, so the same plan
/// skips the same blocks in every run and under every scheduler.
fn short_circuits(index: [usize; 3], fraction: f64) -> bool {
    if fraction <= 0.0 {
        return false;
    }
    if fraction >= 1.0 {
        return true;
    }
    let mixed = (index[0].wrapping_mul(73_856_093)
        ^ index[1].wrapping_mul(19_349_663)
        ^ index[2].wrapping_mul(83_492_791)) as u64;
    (mixed % 10_000) < (fraction * 10_000.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decomposition::PhaseDecomposition;
    use crate::dtype::Dtype;
    use crate::geometry::BlockGrid;
    use crate::reach::Reach;

    #[test]
    fn machine_topology_normalizes_zero_workers_and_nodes() {
        let machine = Machine {
            nodes: 0,
            workers: 0,
            cache_shared: true,
            ..Machine::default()
        };

        let topology = MachineTopology::new(&machine);

        assert_eq!(topology.nodes, 1);
        assert_eq!(topology.workers, 1);
        assert_eq!(topology.node_of(0), 0);
        assert_eq!(topology.pool_of(0), 0);
        assert_eq!(topology.pools(), 1);
        assert_eq!(topology.workers_per_node(), 1);
        assert_eq!(topology.cache_pools_per_node(), 1);
    }

    #[test]
    fn measured_per_phase_derives_short_circuit_fraction_from_the_log() {
        let volume = [8, 4, 4];
        let grid = BlockGrid::along(volume, &[0], 4).unwrap();
        let decomposition = Decomposition {
            volume,
            dtype: Dtype::F64,
            phases: vec![
                PhaseDecomposition::derive(
                    vec![0],
                    vec!["a".to_string()],
                    Reach::from([0, 0, 0]),
                    Reach::from([0, 0, 0]),
                    grid.clone(),
                ),
                PhaseDecomposition::derive(
                    vec![1],
                    vec!["b".to_string()],
                    Reach::from([0, 0, 0]),
                    Reach::from([0, 0, 0]),
                    grid,
                ),
            ],
            chain_reach: [0, 0, 0],
        };
        let log = ExecutionLog::new();
        for phase in 0..2 {
            for x in 0..2 {
                log.push(Event::TaskAdmitted {
                    phase,
                    index: [x, 0, 0],
                });
            }
        }
        log.push(Event::BlockShortCircuited {
            phase: 1,
            index: [0, 0, 0],
            from: 0.0,
            to: 0.0,
            slots: vec![1],
            names: vec!["b".to_string()],
        });

        let measured = MeasuredPerPhase::from_log(&decomposition, &log);
        assert_eq!(measured.constant_fraction, vec![0.0, 0.5]);
        assert_eq!(measured.as_per_phase().constant_fraction, &[0.0, 0.5]);
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
    /// The same, restricted to **this worker's own computer**.
    ///
    /// What a node is about to have in its pool, which is a different thing from
    /// what every machine in the cluster is doing. A policy that wants a node's
    /// workers to converge on shared chunks needs this and not
    /// [`Self::running`]: see `distributed::handout::WorkerView::in_flight`,
    /// which is where it goes.
    pub node_running: &'a [usize],
    /// Which worker this choice is for.
    ///
    /// One node with a shared cache makes this uninteresting and it is `0`
    /// throughout; otherwise it is the identity a handout policy ranks against,
    /// and [`Decision::cache`] is that worker's own pool.
    pub worker: usize,
    /// How many computers there are; see [`Machine::nodes`].
    ///
    /// **The worker's node is `worker % nodes`**, and [`Self::node`] is that
    /// arithmetic written once. A policy that wants to keep a chunk on the
    /// machine that already holds it needs both: the cache it is handed is its
    /// node's, and which node that is decides what the *other* nodes are
    /// holding.
    pub nodes: usize,
    /// Where each worker last finished, for a policy that seeds workers apart.
    /// `None` for a worker that has finished nothing.
    pub anchors: &'a [Option<[f64; 3]>],
    /// The same, **per computer**: where any worker of each node last finished.
    ///
    /// **This is the one a locality policy wants, and the per-worker list is
    /// not.** What must be kept apart is the *nodes* — they cannot share a page
    /// cache, so two of them working the same region fetch every shared chunk
    /// twice. What must be kept *together* is the workers of one node, which
    /// share theirs and lose the sharing the moment they are scattered.
    ///
    /// Seeding by worker does both jobs at once and gets the second one
    /// backwards. Measured on a `96^3` plan in `16^3` chunks: at one worker per
    /// computer, farthest-point seeding is **1.24x** faster than plan order and
    /// fetches a third of the bytes; at ten workers per computer the same policy
    /// is **1.14x slower** than plan order, because it scatters each machine's
    /// own threads. The two are the same policy told to separate the wrong
    /// things.
    ///
    /// Identical to [`Self::anchors`] when there is one worker per node, which
    /// is why the good case stays exactly as good.
    pub node_anchors: &'a [Option<[f64; 3]>],
    /// Images alive right now, and their sizes.
    pub live_images: &'a [(usize, u64)],
    /// Bytes resident right now: images plus in-flight block buffers.
    pub resident_bytes: u64,
    read_footprints: &'a ReadFootprints,
    /// The cache, for a scheduler that wants to prefer a warm task.
    pub cache: &'a ModelledCache,
    /// One chunk grid per image the plan reads, keyed by image id.
    ///
    /// **A map and not one grid**, because the images of a plan do not share a
    /// volume — a resampling phase writes a different extent — and a scheduler
    /// asking about warmth has to ask on the right lattice. Most schedulers want
    /// [`Decision::chunks_of`] rather than this.
    pub grids: &'a BTreeMap<usize, ChunkGrid>,
    /// The images each phase's blocks fetch, by phase. See
    /// [`crate::decomposition::PhaseDecomposition::images_read`].
    pub images_read: &'a [Vec<usize>],
    /// Compute nanoseconds per voxel, per phase. Always as long as the phase
    /// count — [`simulate`] fills it from [`Rates::compute_ns_per_voxel`] where
    /// the caller supplied nothing, so a scheduler never has to ask which it
    /// got.
    pub phase_ns_per_voxel: &'a [f64],
    /// Substages each phase runs. Always as long as the phase count; see
    /// [`PerPhase::substages`].
    pub phase_substages: &'a [u64],
}

impl Decision<'_> {
    /// The computer this choice is for: [`Self::worker`] `%` [`Self::nodes`].
    pub fn node(&self) -> usize {
        self.worker % self.nodes.max(1)
    }

    /// Every chunk key one task fetches: each image the phase reads, at
    /// [`crate::geometry::BlockGeometry::source`], on that image's own grid.
    ///
    /// The same walk the event loop performs, exposed so that a scheduler
    /// reasoning about warmth asks the question the run will actually ask. A
    /// scheduler that assembled the keys itself would be a fourth statement of
    /// what a block fetches, and the first one to go stale.
    pub fn chunks_of(&self, task: &crate::graph::Task) -> Vec<u64> {
        self.read_footprints.chunks_of(task)
    }
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
            let keys = decision.chunks_of(task);
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
        (rates.io_latency_ns
            + rates.chunk_bytes as f64 * (rates.io_ns_per_byte + rates.decode_ns_per_byte))
            .ceil()
            .max(1.0) as u64
    }

    /// The shortest horizon that can contain the largest fetch any one task in
    /// `decomposition` may perform.
    ///
    /// This is the plan-aware form of [`Self::floor_ns`]. A task may read
    /// several chunks, and a source-leaf phase may read several images; the
    /// fallback one-chunk floor cannot see either. This walks the same
    /// per-image chunk grids that [`simulate`] and [`Decision::chunks_of`] use,
    /// then prices the miss path as latency per chunk plus transfer and decode
    /// per byte.
    pub fn floor_for_plan(decomposition: &Decomposition, rates: &Rates) -> u64 {
        let read_footprints = ReadFootprints::new(decomposition, rates);
        let graph = TaskGraph::build(decomposition);
        read_footprints.max_task_fetch_ns(&graph, rates)
    }

    /// A horizon at or above [`Self::floor_ns`], on [`RateBasis::PerPhaseCost`].
    pub fn new(horizon_ns: u64, rates: &Rates) -> Result<Self> {
        Self::with_basis(horizon_ns, rates, RateBasis::PerPhaseCost)
    }

    /// A horizon at or above [`Self::floor_for_plan`], on
    /// [`RateBasis::PerPhaseCost`].
    pub fn new_for_plan(
        horizon_ns: u64,
        rates: &Rates,
        decomposition: &Decomposition,
    ) -> Result<Self> {
        Self::with_basis_for_plan(horizon_ns, rates, decomposition, RateBasis::PerPhaseCost)
    }

    /// The same, on a stated basis. See [`RateBasis`].
    pub fn with_basis(horizon_ns: u64, rates: &Rates, rate: RateBasis) -> Result<Self> {
        let floor = Self::floor_ns(rates);
        Self::with_floor(horizon_ns, floor, rate)
    }

    /// [`Self::with_basis`], but using [`Self::floor_for_plan`] as the lower
    /// bound.
    pub fn with_basis_for_plan(
        horizon_ns: u64,
        rates: &Rates,
        decomposition: &Decomposition,
        rate: RateBasis,
    ) -> Result<Self> {
        let floor = Self::floor_for_plan(decomposition, rates);
        Self::with_floor(horizon_ns, floor, rate)
    }

    fn with_floor(horizon_ns: u64, floor: u64, rate: RateBasis) -> Result<Self> {
        if horizon_ns < floor {
            return Err(crate::error::Error::InvalidArgument(format!(
                "a horizon of {horizon_ns} ns is shorter than the {floor} ns fetch floor at \
                 these rates. A scheduler that cannot see a fetch finish cannot see it \
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
            let keys = decision.chunks_of(task);
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

/// The **real handout policy**, as a scheduler.
///
/// `distributed::handout::choose` is a free function over a `TaskGraph`, a
/// `ChunkGrid` and a `WorkerView`, so the simulator can *call* the coordinator's
/// policy rather than carry a second copy of it — the same move
/// `strategy::priority_key` makes for the executor's dispatch order, and for the
/// same reason: sharing the definition is a drift impossibility where a
/// transcription is only a drift detector.
///
/// **Only meaningful with `Machine::cache_shared` false.** A policy that ranks
/// on which worker already holds a chunk has nothing to rank when every worker
/// shares one pool, and `Outcome::duplicated_fetches` — the quantity it exists
/// to reduce — is zero by construction there.
pub struct Handout {
    pub policy: crate::distributed::handout::HandoutPolicy,
}

impl Handout {
    pub fn new(policy: crate::distributed::handout::HandoutPolicy) -> Self {
        Self { policy }
    }
}

impl Scheduler for Handout {
    fn name(&self) -> &'static str {
        self.policy.as_str()
    }

    fn pick(&mut self, decision: &Decision<'_>) -> usize {
        // The grid of the image the first ready task reads. `choose` takes one
        // grid and this crate now has one per image; on a plan whose phases
        // share a lattice they are the same grid, and where they are not the
        // policy is ranking distances rather than reading bytes, so the choice
        // of lattice moves nothing it decides.
        let first = decision.graph.tasks[decision.ready[0]].phase;
        let image = decision.images_read[first].first().copied().unwrap_or(0);
        let Some(grid) = decision.grids.get(&image) else {
            return 0;
        };
        // **Other computers, not other workers.** The seeds are what this choice
        // stays away from, and what it must stay away from is the machines that
        // cannot share its page cache — never the threads that can. See
        // `Decision::node_anchors` for the measurement that separates the two.
        let node = decision.node();
        let seeds: Vec<[f64; 3]> = decision
            .node_anchors
            .iter()
            .enumerate()
            .filter(|&(other, _)| other != node)
            .filter_map(|(_, anchor)| *anchor)
            .collect();
        let view = crate::distributed::handout::WorkerView {
            in_flight: decision.node_running,
            // The **node's** last block, so a thread with no history of its own
            // still follows its machine rather than seeding away from it — which
            // is what scattered a node's workers when this was per worker.
            anchor: decision
                .node_anchors
                .get(node)
                .copied()
                .flatten()
                .or(decision.anchors[decision.worker]),
            cache: Some(decision.cache),
        };
        let chosen = crate::distributed::handout::choose(
            self.policy,
            decision.ready,
            decision.graph,
            grid,
            &view,
            &seeds,
        );
        chosen
            .and_then(|task| decision.ready.iter().position(|&id| id == task))
            .unwrap_or(0)
    }
}

struct ReadFootprint {
    keys: Vec<u64>,
    chunk_bytes: u64,
    buffer_bytes: u64,
}

/// Per-task read footprints on each image's own chunk lattice.
///
/// Scheduler warmth, bounded-horizon floors, demand reads, prefetch reads and
/// block-buffer residency all ask this one owner what a task reads. The byte
/// meanings stay separate: `chunk_bytes` prices storage/cache transfers, while
/// `buffer_bytes` is the in-memory block slice the worker holds.
struct ReadFootprints {
    images_read: Vec<Vec<usize>>,
    grids: BTreeMap<usize, ChunkGrid>,
    chunk_bytes: BTreeMap<usize, u64>,
    element_bytes: BTreeMap<usize, u64>,
}

impl ReadFootprints {
    fn new(decomposition: &Decomposition, rates: &Rates) -> Self {
        // **A grid per image, not one per plan.** `ChunkGrid` is built from a
        // volume, and the images of a plan do not share one: a resampling phase
        // writes an image of a different extent, and a pyramid level is a
        // different extent by construction.
        //
        // `BTreeMap` rather than a `Vec` because a supplied input's id is not
        // an index into `0..n_images()`.
        let images_read: Vec<Vec<usize>> = decomposition
            .phases
            .iter()
            .enumerate()
            .map(|(index, phase)| phase.images_read(index))
            .collect();
        let mut grids: BTreeMap<usize, ChunkGrid> = BTreeMap::new();
        for images in &images_read {
            for &image in images {
                grids
                    .entry(image)
                    .or_insert_with(|| ChunkGrid::new(decomposition.volume_at(image), rates.chunk));
            }
        }
        let element_bytes: BTreeMap<usize, u64> = grids
            .keys()
            .map(|&image| (image, decomposition.dtype_at(image).size_of() as u64))
            .collect();
        let chunk_bytes: BTreeMap<usize, u64> = element_bytes
            .iter()
            .map(|(&image, &bytes)| {
                let chunk = product3(rates.chunk) as u64 * bytes;
                (image, chunk.max(1))
            })
            .collect();
        Self {
            images_read,
            grids,
            chunk_bytes,
            element_bytes,
        }
    }

    #[inline]
    fn images_read(&self) -> &[Vec<usize>] {
        &self.images_read
    }

    #[inline]
    fn grids(&self) -> &BTreeMap<usize, ChunkGrid> {
        &self.grids
    }

    #[inline]
    fn images_of(&self, phase: usize) -> &[usize] {
        &self.images_read[phase]
    }

    fn chunks_of(&self, task: &crate::graph::Task) -> Vec<u64> {
        self.images_of(task.phase)
            .iter()
            .flat_map(|&image| self.keys(image, task).unwrap_or_default())
            .collect()
    }

    fn demand_reads(&self, task: &crate::graph::Task, skipped: bool) -> Vec<ReadFootprint> {
        let images = self.images_of(task.phase);
        let fetches = if skipped {
            &images[..1.min(images.len())]
        } else {
            images
        };
        fetches
            .iter()
            .filter_map(|&image| self.footprint(image, task))
            .collect()
    }

    fn prefetch_reads(&self, task: &crate::graph::Task) -> Vec<(u64, u64)> {
        self.images_of(task.phase)
            .iter()
            .flat_map(|&image| {
                let size = self.chunk_bytes[&image];
                self.keys(image, task)
                    .unwrap_or_default()
                    .into_iter()
                    .map(move |key| (key, size))
            })
            .collect()
    }

    fn max_task_fetch_ns(&self, graph: &TaskGraph, rates: &Rates) -> u64 {
        graph
            .tasks
            .iter()
            .map(|task| {
                let mut chunks = 0u64;
                let mut bytes = 0u64;
                for read in self.demand_reads(task, false) {
                    let n = read.keys.len() as u64;
                    chunks += n;
                    bytes += n * read.chunk_bytes;
                }
                (chunks as f64 * rates.io_latency_ns
                    + bytes as f64 * (rates.io_ns_per_byte + rates.decode_ns_per_byte))
                    .ceil()
                    .max(1.0) as u64
            })
            .max()
            .unwrap_or_else(|| BoundedHorizonThroughput::floor_ns(rates))
    }

    fn footprint(&self, image: usize, task: &crate::graph::Task) -> Option<ReadFootprint> {
        let keys = self.keys(image, task)?;
        Some(ReadFootprint {
            keys,
            chunk_bytes: self.chunk_bytes[&image],
            buffer_bytes: task.geometry.source.voxels() as u64 * self.element_bytes[&image],
        })
    }

    fn keys(&self, image: usize, task: &crate::graph::Task) -> Option<Vec<u64>> {
        self.grids
            .get(&image)
            .map(|grid| grid.keys(image, &task.geometry.source))
    }
}

struct WriteFootprint {
    writes_image: bool,
    sidecars: Vec<SidecarSize>,
}

struct BlockStore {
    image_bytes: u64,
    image_kind: crate::decomposition::ImageKind,
    sidecar_bytes: u64,
}

/// Per-phase write and sidecar footprints.
///
/// Residency allocation, in-flight output buffers, store counters, store IO and
/// sidecar gather peaks all ask this one owner what a phase produces. The facts
/// still come from `phase_traffic` and `PhaseWork::outputs`; this type only
/// keeps the event loop from restating them at each accounting site.
struct WriteFootprints {
    phases: Vec<WriteFootprint>,
}

impl WriteFootprints {
    fn new(decomposition: &Decomposition, work: &[PhaseWork<'_>]) -> Result<Self> {
        let phases = decomposition
            .phases
            .iter()
            .enumerate()
            .map(|(index, phase)| {
                let traffic = crate::decomposition::phase_traffic(index, phase, work.get(index))?;
                let sidecars = match work.get(index) {
                    Some(crate::fragment::PhaseWork::Fragments(op)) => {
                        op.outputs().iter().map(|output| output.size).collect()
                    }
                    _ => Vec::new(),
                };
                Ok(WriteFootprint {
                    writes_image: traffic.writes_an_image,
                    sidecars,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { phases })
    }

    #[inline]
    fn writes_image(&self, phase: usize) -> bool {
        self.phases[phase].writes_image
    }

    #[inline]
    fn allocation(
        &self,
        task: &crate::graph::Task,
        decomposition: &Decomposition,
        bytes_of: impl FnOnce(usize) -> u64,
    ) -> Option<(usize, u64)> {
        let image = task.phase + 1;
        (image < decomposition.n_images() && self.writes_image(task.phase))
            .then(|| (image, bytes_of(image)))
    }

    #[inline]
    fn output_buffer_bytes(&self, task: &crate::graph::Task, decomposition: &Decomposition) -> u64 {
        if self.writes_image(task.phase) {
            task.geometry.read.voxels() as u64
                * decomposition.dtype_at(task.phase + 1).size_of() as u64
        } else {
            0
        }
    }

    fn sidecar_bytes(&self, task: &crate::graph::Task) -> u64 {
        self.phases[task.phase]
            .sidecars
            .iter()
            .filter_map(|size| {
                size.bytes_at_most(task.geometry.core.shape3(), task.geometry.read.shape3())
            })
            .sum()
    }

    fn gather_peak(&self, phase: usize, graph: &TaskGraph) -> u64 {
        if !graph.is_barrier(phase) && self.phases[phase].sidecars.is_empty() {
            return 0;
        }
        graph
            .tasks_in_phase(phase)
            .iter()
            .map(|task| self.sidecar_bytes(task))
            .sum()
    }

    fn store(&self, task: &crate::graph::Task, decomposition: &Decomposition) -> BlockStore {
        let image = task.phase + 1;
        let image_bytes = if self.writes_image(task.phase) && image < decomposition.n_images() {
            task.geometry.valid.voxels() as u64 * decomposition.dtype_at(image).size_of() as u64
        } else {
            0
        };
        BlockStore {
            image_bytes,
            image_kind: decomposition.image_kind(image),
            sidecar_bytes: self.sidecar_bytes(task),
        }
    }
}

/// Normalized machine topology for the simulator event loop.
///
/// `Machine` is the external contract; this is the internal view with at least
/// one worker, at least one node, round-robin worker placement, and one
/// cache-pool rule. Scheduler decisions, cache pools, node anchors, contention
/// and IO reservations all depend on these same facts.
#[derive(Debug, Clone, Copy)]
struct MachineTopology {
    nodes: usize,
    workers: usize,
    cache_shared: bool,
}

impl MachineTopology {
    fn new(machine: &Machine) -> Self {
        Self {
            nodes: machine.nodes.max(1),
            workers: machine.workers.max(1),
            cache_shared: machine.cache_shared,
        }
    }

    #[inline]
    fn node_of(self, worker: usize) -> usize {
        worker % self.nodes
    }

    #[inline]
    fn pool_of(self, worker: usize) -> usize {
        if self.cache_shared {
            self.node_of(worker)
        } else {
            worker
        }
    }

    #[inline]
    fn pools(self) -> usize {
        if self.cache_shared {
            self.nodes
        } else {
            self.workers
        }
    }

    #[inline]
    fn workers_per_node(self) -> usize {
        self.workers.div_ceil(self.nodes)
    }

    #[inline]
    fn cache_pools_per_node(self) -> u64 {
        if self.cache_shared {
            1
        } else {
            self.workers_per_node().max(1) as u64
        }
    }
}

/// Maintained task readiness, including tasks held behind phase barriers.
///
/// Dependency counts, started-task marking, and barrier-held admission live
/// here together. The visible ready slice is always sorted by task id,
/// preserving the order the old full scan handed to schedulers.
struct ReadySet {
    ready: Vec<usize>,
    held: Vec<Vec<usize>>,
    indegree: Vec<usize>,
}

impl ReadySet {
    fn new(graph: &TaskGraph, phase_open: impl Fn(usize) -> bool) -> Self {
        let mut this = Self {
            ready: Vec::new(),
            held: vec![Vec::new(); graph.n_phases()],
            indegree: graph
                .tasks
                .iter()
                .map(|task| task.n_dependencies())
                .collect(),
        };
        for id in 0..graph.tasks.len() {
            if this.indegree[id] == 0 {
                this.admit(id, graph.tasks[id].phase, &phase_open);
            }
        }
        this
    }

    #[inline]
    fn dependencies_complete(&self, id: usize) -> bool {
        self.indegree[id] == 0
    }

    #[inline]
    fn mark_started(&mut self, id: usize) {
        // `usize::MAX` cannot be reached by decrementing. It marks a task that
        // has left the ready set, so neither the debug oracle nor prefetch
        // treats it as ready again.
        self.indegree[id] = usize::MAX;
    }

    fn complete_dependencies(
        &mut self,
        completed: usize,
        graph: &TaskGraph,
        dependents: &[Vec<usize>],
        phase_open: impl Fn(usize) -> bool,
    ) {
        for &next in &dependents[completed] {
            self.indegree[next] -= 1;
            if self.indegree[next] == 0 {
                self.admit(next, graph.tasks[next].phase, &phase_open);
            }
        }
    }

    #[inline]
    fn as_slice(&self) -> &[usize] {
        &self.ready
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.ready.is_empty()
    }

    #[inline]
    fn push(&mut self, id: usize) {
        let at = self.ready.partition_point(|&other| other < id);
        self.ready.insert(at, id);
    }

    #[inline]
    fn admit(&mut self, id: usize, phase: usize, phase_open: impl Fn(usize) -> bool) {
        if phase_open(phase) {
            self.push(id);
        } else {
            self.hold(phase, id);
        }
    }

    #[inline]
    fn hold(&mut self, phase: usize, id: usize) {
        self.held[phase].push(id);
    }

    #[inline]
    fn release_through(&mut self, phase_limit: usize) {
        for phase in 0..phase_limit.min(self.held.len()) {
            let held = std::mem::take(&mut self.held[phase]);
            for id in held {
                self.push(id);
            }
        }
    }

    #[inline]
    fn remove(&mut self, slot: usize) -> usize {
        self.ready.remove(slot)
    }

    #[cfg(debug_assertions)]
    fn scanned(&self, graph: &TaskGraph, phase_open: impl Fn(usize) -> bool) -> Vec<usize> {
        (0..graph.tasks.len())
            .filter(|&id| self.indegree[id] == 0)
            .filter(|&id| phase_open(graph.tasks[id].phase))
            .collect()
    }
}

/// The prefix of the ready set visible to a scheduler at one dispatch.
///
/// `0` is the unbounded case. Every non-zero window is clamped to the current
/// ready length, and scheduler results are clamped back to the same visible
/// prefix. That keeps "what the scheduler was offered" and "what it can pick"
/// as one contract.
struct CandidateWindow<'a> {
    ready: &'a [usize],
}

impl<'a> CandidateWindow<'a> {
    fn new(ready: &'a ReadySet, limit: usize) -> Self {
        let ready = ready.as_slice();
        debug_assert!(
            !ready.is_empty(),
            "candidate window is only built when a task is ready"
        );
        let end = match limit {
            0 => ready.len(),
            limit => limit.min(ready.len()),
        };
        Self {
            ready: &ready[..end],
        }
    }

    #[inline]
    fn as_slice(&self) -> &'a [usize] {
        self.ready
    }

    #[inline]
    fn clamp_pick(&self, slot: usize) -> usize {
        slot.min(self.ready.len() - 1)
    }
}

struct CacheAccess {
    fetched_bytes: u64,
    misses: u64,
    encoded_hit_bytes: u64,
    hits: u64,
    encoded_hits: u64,
    duplicated_fetches: u64,
}

struct PrefetchAccess {
    fetched_bytes: u64,
    misses: u64,
}

struct DemandReadTransaction {
    fetches: Vec<ReadFootprint>,
    transfer: ChunkTransfer,
    encoded_hit_bytes: u64,
}

impl DemandReadTransaction {
    fn start(
        task: &crate::graph::Task,
        skipped: bool,
        read_footprints: &ReadFootprints,
        caches: &mut CachePools,
        pool: usize,
        outcome: &mut Outcome,
    ) -> Self {
        let fetches = read_footprints.demand_reads(task, skipped);
        let mut transfer = ChunkTransfer::default();
        let mut encoded_hit_bytes = 0u64;
        for read in &fetches {
            let access = caches.demand(pool, &read.keys, read.chunk_bytes);
            encoded_hit_bytes += access.encoded_hit_bytes;
            transfer.add(access.misses, access.fetched_bytes);
            outcome.cache_hits += access.hits;
            outcome.cache_misses += access.misses;
            outcome.encoded_hits += access.encoded_hits;
            outcome.duplicated_fetches += access.duplicated_fetches;
        }
        outcome.fetched_bytes += transfer.fetched_bytes;
        Self {
            fetches,
            transfer,
            encoded_hit_bytes,
        }
    }

    #[inline]
    fn transfer_ns(&self, rates: &Rates) -> u64 {
        self.transfer.read_ns(rates)
    }

    #[inline]
    fn needs_channel(&self) -> bool {
        !self.transfer.is_empty()
    }

    #[inline]
    fn decoded_at(&self, io_done: u64, rates: &Rates) -> u64 {
        let decoded_bytes = self.transfer.fetched_bytes + self.encoded_hit_bytes;
        io_done + (decoded_bytes as f64 * rates.decode_ns_per_byte) as u64
    }

    #[inline]
    fn input_buffer_bytes(&self) -> u64 {
        self.fetches.iter().map(|read| read.buffer_bytes).sum()
    }
}

struct StoreTransaction {
    store: BlockStore,
}

impl StoreTransaction {
    fn start(
        task: &crate::graph::Task,
        write_footprints: &WriteFootprints,
        decomposition: &Decomposition,
        outcome: &mut Outcome,
    ) -> Self {
        let store = write_footprints.store(task, decomposition);
        outcome.sidecar_bytes_written += store.sidecar_bytes;
        Self { store }
    }

    #[inline]
    fn finish_after(
        self,
        computed: u64,
        node: usize,
        io: &mut IoChannels,
        rates: &Rates,
        outcome: &mut Outcome,
    ) -> u64 {
        if self.store.image_bytes + self.store.sidecar_bytes == 0 {
            return computed;
        }

        let intermediate = self.store.image_kind == crate::decomposition::ImageKind::Intermediate;
        let rate = if intermediate {
            rates.materialise_ns_per_byte
        } else {
            rates.write_ns_per_byte
        };
        if intermediate {
            outcome.materialised_bytes += self.store.image_bytes;
        } else {
            outcome.written_bytes += self.store.image_bytes;
        }

        let transfer = (self.store.image_bytes as f64 * rate) as u64
            + (self.store.sidecar_bytes as f64 * rates.materialise_ns_per_byte) as u64;
        if transfer == 0 {
            return computed;
        }

        let (start, done) = io.reserve(node, computed, transfer);
        outcome.io_wait_ns += start - computed;
        done
    }
}

struct PrefetchIssuance;

impl PrefetchIssuance {
    #[allow(clippy::too_many_arguments)]
    fn issue_after_dispatch(
        depth: usize,
        dispatched_id: usize,
        now: u64,
        node: usize,
        pool: usize,
        graph: &TaskGraph,
        ready: &ReadySet,
        read_footprints: &ReadFootprints,
        caches: &mut CachePools,
        io: &mut IoChannels,
        rates: &Rates,
        outcome: &mut Outcome,
    ) {
        if depth == 0 {
            return;
        }
        let mut issued = 0usize;
        for ahead in graph.tasks.iter().skip(dispatched_id + 1) {
            if issued == depth {
                break;
            }
            // Test the idle-channel admission once per dispatch. A depth run
            // may queue several speculative reads once it has found idle time.
            if issued == 0 && io.all_busy_at(node, now) {
                break;
            }
            if !ready.dependencies_complete(ahead.id) {
                continue;
            }
            let ahead_keys = read_footprints.prefetch_reads(ahead);
            let access = caches.prefetch(pool, &ahead_keys);
            if access.misses == 0 {
                continue;
            }
            outcome.fetched_bytes += access.fetched_bytes;
            outcome.prefetched_bytes += access.fetched_bytes;
            outcome.cache_misses += access.misses;
            io.reserve(
                node,
                now,
                ChunkTransfer {
                    misses: access.misses,
                    fetched_bytes: access.fetched_bytes,
                }
                .read_ns(rates),
            );
            issued += 1;
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct ChunkTransfer {
    misses: u64,
    fetched_bytes: u64,
}

impl ChunkTransfer {
    #[inline]
    fn add(&mut self, misses: u64, fetched_bytes: u64) {
        self.misses += misses;
        self.fetched_bytes += fetched_bytes;
    }

    #[inline]
    fn is_empty(self) -> bool {
        self.misses == 0 && self.fetched_bytes == 0
    }

    #[inline]
    fn read_ns(self, rates: &Rates) -> u64 {
        (self.misses as f64 * rates.io_latency_ns
            + self.fetched_bytes as f64 * rates.io_ns_per_byte) as u64
    }
}

/// Demand and prefetch access to the simulator's decoded/encoded cache pools.
///
/// The wrapper keeps tier checks, LRU updates and duplicated-fetch accounting
/// together, because a prefetch and a demand read must agree on which chunks are
/// resident even though they charge different outcome fields.
struct CachePools {
    decoded: Vec<ModelledCache>,
    encoded: Vec<ModelledCache>,
    ever_fetched: BTreeMap<u64, usize>,
}

impl CachePools {
    fn new(pools: usize, decoded_bytes: u64, encoded_bytes: u64, chunk_bytes: u64) -> Self {
        Self {
            decoded: (0..pools)
                .map(|_| ModelledCache::new(decoded_bytes, chunk_bytes))
                .collect(),
            encoded: (0..pools)
                .map(|_| ModelledCache::new(encoded_bytes, chunk_bytes))
                .collect(),
            ever_fetched: BTreeMap::new(),
        }
    }

    #[inline]
    fn decoded(&self, pool: usize) -> &ModelledCache {
        &self.decoded[pool]
    }

    fn demand(&mut self, pool: usize, keys: &[u64], size: u64) -> CacheAccess {
        let sized_keys: Vec<(u64, u64)> = keys.iter().map(|&key| (key, size)).collect();
        let missed_decoded = self.decoded[pool].misses(keys) as u64;
        let not_in_either = keys
            .iter()
            .filter(|key| !self.decoded[pool].holds(**key) && !self.encoded[pool].holds(**key))
            .count() as u64;
        let mut duplicated_fetches = 0;
        for key in keys {
            if self.decoded[pool].holds(*key) || self.encoded[pool].holds(*key) {
                continue;
            }
            match self.ever_fetched.entry(*key) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(pool);
                }
                std::collections::btree_map::Entry::Occupied(first) => {
                    if *first.get() != pool {
                        duplicated_fetches += 1;
                    }
                }
            }
        }
        self.decoded[pool].note_assigned_sized(&sized_keys);
        self.encoded[pool].note_assigned_sized(&sized_keys);
        CacheAccess {
            fetched_bytes: not_in_either * size,
            misses: not_in_either,
            encoded_hit_bytes: (missed_decoded - not_in_either) * size,
            hits: keys.len() as u64 - not_in_either,
            encoded_hits: missed_decoded - not_in_either,
            duplicated_fetches,
        }
    }

    fn prefetch(&mut self, pool: usize, keys: &[(u64, u64)]) -> PrefetchAccess {
        let missed: Vec<(u64, u64)> = keys
            .iter()
            .filter(|(key, _)| !self.decoded[pool].holds(*key) && !self.encoded[pool].holds(*key))
            .copied()
            .collect();
        if missed.is_empty() {
            return PrefetchAccess {
                fetched_bytes: 0,
                misses: 0,
            };
        }
        let fetched_bytes = missed.iter().map(|(_, bytes)| *bytes).sum();
        self.decoded[pool].note_assigned_sized(keys);
        self.encoded[pool].note_assigned_sized(keys);
        PrefetchAccess {
            fetched_bytes,
            misses: missed.len() as u64,
        }
    }
}

/// Image residency and per-task in-flight buffers.
///
/// The simulator still asks [`Decomposition::images_freed_after`] when a phase
/// completes. This type owns the resulting bookkeeping: what images are live,
/// which phase outputs have already been allocated, and how many task-local
/// bytes are currently in flight.
struct Residency {
    live: Vec<(usize, u64)>,
    allocated: Vec<bool>,
    block_bytes: Vec<u64>,
    in_flight_bytes: u64,
}

impl Residency {
    fn new(decomposition: &Decomposition, tasks: usize, bytes_of: impl Fn(usize) -> u64) -> Self {
        let mut live = vec![(0, bytes_of(0))];
        for image in decomposition.supplied_input_images() {
            live.push((image, bytes_of(image)));
        }
        Self {
            live,
            allocated: vec![false; decomposition.n_images() + 1],
            block_bytes: vec![0; tasks],
            in_flight_bytes: 0,
        }
    }

    #[inline]
    fn live_images(&self) -> &[(usize, u64)] {
        &self.live
    }

    #[inline]
    fn resident_bytes(&self) -> u64 {
        self.live.iter().map(|&(_, bytes)| bytes).sum::<u64>() + self.in_flight_bytes
    }

    #[inline]
    fn allocate_once(&mut self, image: usize, bytes: u64) {
        if image < self.allocated.len() && !self.allocated[image] {
            self.allocated[image] = true;
            self.live.push((image, bytes));
        }
    }

    #[inline]
    fn release(&mut self, freed: &[usize]) {
        self.live.retain(|&(image, _)| !freed.contains(&image));
    }

    #[inline]
    fn start_task(&mut self, task: usize, bytes: u64) {
        self.block_bytes[task] = bytes;
        self.in_flight_bytes += bytes;
    }

    #[inline]
    fn finish_task(&mut self, task: usize) {
        self.in_flight_bytes -= self.block_bytes[task];
    }
}

/// Phase completion state and the barrier-open predicate derived from it.
///
/// The event loop needs four facts to agree: when a phase first started, when
/// its latest task finished, how many of its tasks are done, and which barrier
/// phases are now open. Keeping them together makes "phase p has drained" one
/// transition instead of repeated vector arithmetic at completion sites.
struct PhaseProgress {
    done: Vec<usize>,
    tasks: Vec<usize>,
    started: Vec<Option<u64>>,
    finished: Vec<u64>,
    finished_phases: usize,
}

impl PhaseProgress {
    fn new(graph: &TaskGraph) -> Self {
        Self {
            done: vec![0; graph.n_phases()],
            tasks: (0..graph.n_phases())
                .map(|phase| graph.tasks_in_phase(phase).len())
                .collect(),
            started: vec![None; graph.n_phases()],
            finished: vec![0; graph.n_phases()],
            finished_phases: 0,
        }
    }

    #[inline]
    fn phase_open(&self, phase: usize, waits_for_the_phase_below: impl Fn(usize) -> bool) -> bool {
        !waits_for_the_phase_below(phase) || self.finished_phases >= phase
    }

    #[inline]
    fn mark_started(&mut self, phase: usize, now: u64) {
        self.started[phase].get_or_insert(now);
    }

    #[inline]
    fn complete_task(&mut self, phase: usize, now: u64) -> bool {
        self.finished[phase] = self.finished[phase].max(now);
        self.done[phase] += 1;
        if self.done[phase] == self.tasks[phase] {
            self.finished_phases = self.finished_phases.max(phase + 1);
            true
        } else {
            false
        }
    }

    #[inline]
    fn release_limit(&self) -> usize {
        self.finished_phases + 1
    }

    fn phase_span_ns(&self) -> u64 {
        self.started
            .iter()
            .zip(self.finished.iter())
            .filter_map(|(started, finished)| {
                started.map(|started| finished.saturating_sub(started))
            })
            .sum()
    }
}

/// Worker timeline state: current time, running tasks, occupied slots, anchors,
/// and task conservation.
///
/// These values are one state machine. A task in `running` must occupy exactly
/// one worker slot, a completed task must free that slot before its anchor is
/// updated, and `remaining` must fall only with a real completion.
struct ExecutionTimeline {
    now: u64,
    running: Vec<(u64, usize)>,
    busy: Vec<Option<usize>>,
    anchors: Vec<Option<[f64; 3]>>,
    node_anchors: Vec<Option<[f64; 3]>>,
    remaining: usize,
}

impl ExecutionTimeline {
    fn new(graph: &TaskGraph, topology: MachineTopology) -> Self {
        Self {
            now: 0,
            running: Vec::new(),
            busy: vec![None; topology.workers],
            anchors: vec![None; topology.workers],
            node_anchors: vec![None; topology.nodes],
            remaining: graph.tasks.len(),
        }
    }

    #[inline]
    fn now(&self) -> u64 {
        self.now
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.remaining
    }

    #[inline]
    fn at_capacity(&self, topology: MachineTopology) -> bool {
        self.running.len() >= topology.workers
    }

    #[inline]
    fn has_free_worker(&self, topology: MachineTopology) -> bool {
        self.running.len() < topology.workers
    }

    fn idle_slots_until_next_finish(&self, ready_empty: bool, topology: MachineTopology) -> u64 {
        if ready_empty && self.has_free_worker(topology) {
            self.running
                .last()
                .map(|&(finish, _)| {
                    (finish - self.now) * (topology.workers - self.running.len()) as u64
                })
                .unwrap_or(0)
        } else {
            0
        }
    }

    #[inline]
    fn is_idle(&self) -> bool {
        self.running.is_empty()
    }

    fn advance_to_next_finish(
        &mut self,
        graph: &TaskGraph,
        topology: MachineTopology,
    ) -> Vec<usize> {
        let Some(&(finish, _)) = self.running.last() else {
            return Vec::new();
        };
        self.now = finish;
        let mut completed = Vec::new();
        while let Some(&(finish_ns, id)) = self.running.last() {
            if finish_ns != self.now {
                break;
            }
            self.running.pop();
            let slot = self
                .busy
                .iter()
                .position(|held| *held == Some(id))
                .expect("a running task occupies one worker slot");
            self.busy[slot] = None;
            let at = crate::distributed::handout::position(graph, id);
            self.anchors[slot] = Some(at);
            self.node_anchors[topology.node_of(slot)] = Some(at);
            self.remaining -= 1;
            completed.push(id);
        }
        self.debug_assert_consistent();
        completed
    }

    #[inline]
    fn free_worker(&self) -> usize {
        self.busy
            .iter()
            .position(|slot| slot.is_none())
            .expect("a free worker, since the loop only reaches here below the worker count")
    }

    fn running_ids(&self) -> Vec<usize> {
        self.running.iter().map(|&(_, id)| id).collect()
    }

    fn node_running(&self, node: usize, topology: MachineTopology) -> Vec<usize> {
        self.busy
            .iter()
            .enumerate()
            .filter(|&(slot, _)| topology.node_of(slot) == node)
            .filter_map(|(_, held)| *held)
            .collect()
    }

    #[inline]
    fn anchors(&self) -> &[Option<[f64; 3]>] {
        &self.anchors
    }

    #[inline]
    fn node_anchors(&self) -> &[Option<[f64; 3]>] {
        &self.node_anchors
    }

    fn running_on_node(&self, node: usize, topology: MachineTopology) -> usize {
        self.busy
            .iter()
            .enumerate()
            .filter(|&(slot, held)| held.is_some() && topology.node_of(slot) == node)
            .count()
    }

    fn start(&mut self, worker: usize, id: usize, finish: u64) {
        debug_assert!(
            self.busy[worker].is_none(),
            "worker slot is already occupied"
        );
        self.busy[worker] = Some(id);
        self.running.push((finish, id));
        // Descending by finish time, so the earliest completion is `last`.
        self.running
            .sort_by_key(|&(finish, _)| std::cmp::Reverse(finish));
        self.debug_assert_consistent();
    }

    #[inline]
    fn debug_assert_consistent(&self) {
        debug_assert_eq!(
            self.running.len(),
            self.busy.iter().filter(|slot| slot.is_some()).count(),
            "running tasks and occupied worker slots disagree"
        );
    }
}

/// Node-local IO channel reservations.
///
/// The simulator's IO model is deliberately small: a node owns `channels`
/// independent serial links, and reads, writes and prefetches all reserve the
/// earliest-free link on their own node. Keeping the flat table behind this
/// type makes "which channels belong to a node" one rule instead of arithmetic
/// repeated at demand, write and prefetch sites.
struct IoChannels {
    channels: usize,
    free_at: Vec<u64>,
}

impl IoChannels {
    fn new(nodes: usize, channels: usize) -> Self {
        let channels = channels.max(1);
        Self {
            channels,
            free_at: vec![0; channels * nodes.max(1)],
        }
    }

    #[inline]
    fn reserve(&mut self, node: usize, earliest: u64, duration: u64) -> (u64, u64) {
        let channel = self.earliest_channel(node);
        let start = earliest.max(self.free_at[channel]);
        let done = start + duration;
        self.free_at[channel] = done;
        (start, done)
    }

    #[inline]
    fn all_busy_at(&self, node: usize, now: u64) -> bool {
        self.node_channels(node).iter().all(|&free| free > now)
    }

    #[inline]
    fn earliest_channel(&self, node: usize) -> usize {
        let base = node * self.channels;
        self.node_channels(node)
            .iter()
            .enumerate()
            .min_by_key(|&(_, free)| *free)
            .map(|(index, _)| base + index)
            .unwrap_or(base)
    }

    #[inline]
    fn node_channels(&self, node: usize) -> &[u64] {
        let base = node * self.channels;
        &self.free_at[base..base + self.channels]
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
/// A scheduler is shown [`Machine::candidate_window`] of the ready set — all of
/// it at `0`, which is the default and is what every recorded figure was taken
/// under.
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
    per_phase: PerPhase<'_>,
    scheduler: &mut dyn Scheduler,
) -> Result<Outcome> {
    let n = decomposition.n_phases();
    for (what, len) in [
        ("compute rates", per_phase.ns_per_voxel.len()),
        ("substage counts", per_phase.substages.len()),
        ("constant fractions", per_phase.constant_fraction.len()),
    ] {
        if len != 0 && len != n {
            return Err(crate::error::Error::InvalidArgument(format!(
                "simulate: {len} per-phase {what} for a {n}-phase plan. A partial list would \
                 silently charge some phases the fallback, and which ones would depend on the \
                 order the plan happened to be assembled in."
            )));
        }
    }
    let phase_rates: Vec<f64> = if per_phase.ns_per_voxel.is_empty() {
        vec![rates.compute_ns_per_voxel; n]
    } else {
        per_phase.ns_per_voxel.to_vec()
    };
    let write_footprints = WriteFootprints::new(decomposition, work)?;
    let constant_fraction: Vec<f64> = if per_phase.constant_fraction.is_empty() {
        vec![0.0; n]
    } else {
        per_phase.constant_fraction.to_vec()
    };
    // Zero means one: a phase that is not an iteration reports `0` in
    // `Stats::substages`, and charging it nothing would be reading "not an
    // iteration" as "no work".
    let substages: Vec<u64> = if per_phase.substages.is_empty() {
        vec![1; n]
    } else {
        per_phase
            .substages
            .iter()
            .map(|&s| s.max(1) as u64)
            .collect()
    };
    let graph = TaskGraph::build(decomposition);
    let dependents = graph.dependents();

    let read_footprints = ReadFootprints::new(decomposition, rates);
    // **Two tiers, because the real cache has two.** The decoded tier is what a
    // hit used to be — free. The encoded tier holds more for the same bytes and
    // charges a decode for every hit, which is what gives a cache-size sweep the
    // knee the real one has instead of improving monotonically for ever.
    //
    // `cache.rs` sizes the trade: an encoded entry "survives roughly twenty
    // times longer" for the same bytes, and an encoded hit is **962 us, ~100x a
    // decoded hit** and still ~40x cheaper than storage. So the encoded tier's
    // capacity is its share of the budget times that ratio.
    const ENCODED_RESIDENCY: u64 = 20;
    //
    // **One pool, or one per worker.** Shared is the optimistic reading and the
    // old behaviour; per-worker is what `distributed` actually has, and is the
    // only arrangement in which a chunk two workers both read costs two fetches
    // — which is the quantity a handout policy is ranked on.
    // **A pool per node, and within a node one or one per slot.** The three
    // arrangements `Machine::cache_shared` names, and the middle one — a
    // computer's page cache, shared by its threads and by nobody else — is the
    // one that only exists once there are nodes.
    let topology = MachineTopology::new(machine);
    // **`cache_bytes` is per node**, so it is divided among the pools *of a
    // node* and not among all of them: four computers with 16 GiB each have 64,
    // not 16 shared four ways. At one node this is the expression it was.
    let per_pool = machine.cache_bytes / topology.cache_pools_per_node();
    let encoded_bytes = (per_pool as f64 * machine.encoded_fraction) as u64;
    let mut caches = CachePools::new(
        topology.pools(),
        per_pool - encoded_bytes,
        encoded_bytes * ENCODED_RESIDENCY,
        rates.chunk_bytes,
    );
    // --- image residency, on the executor's own rule -------------------------
    let bytes_of = |image: usize| -> u64 {
        let volume = decomposition.volume_at(image);
        product3(volume) as u64 * decomposition.dtype_at(image).size_of() as u64
    };
    let mut residency = Residency::new(decomposition, graph.tasks.len(), bytes_of);

    let mut outcome = Outcome::default();
    // **One serial IO channel.** Not storage physics — there is no seek, no
    // queue depth and no readahead — but a single shared resource with a
    // finite rate, which is the least that makes prefetch a *trade* rather
    // than free money. Without it, deeper prefetch would improve every run
    // without bound, and a scheduler tuned against that would be tuned against
    // a machine with infinite bandwidth.
    // **One free-at time per channel.** A fetch takes the earliest-free one, so
    // `channels == 1` is exactly the serial model this had and anything above it
    // lets concurrent fetches overlap the way real storage does.
    // **Channels per node**, flat: node `n`'s channels are
    // `n * channels .. (n + 1) * channels`. A fetch takes the earliest-free one
    // *on its own node*, so two computers never queue behind each other.
    let mut io = IoChannels::new(topology.nodes, machine.io_channels);
    let mut timeline = ExecutionTimeline::new(&graph, topology);
    let mut phases = PhaseProgress::new(&graph);
    // --- the ready set, maintained rather than rebuilt ----------------------
    //
    // **Which tasks may start**: indegree zero, not started, and — for a barrier
    // phase — every earlier phase complete. The barrier is checked here rather
    // than encoded as edges because that is where the real graph puts it; see
    // `TaskGraph::barriers`.
    //
    // This used to be a `(0..graph.tasks.len()).filter(..)` **per dispatch**,
    // which is `O(T)` predicate evaluations at every one of the `2T` events a
    // run has: fine at the `4^3` fixtures the simulator shipped with, and
    // `docs/design/planner-gaps.md` names it as the one thing G1 needed before
    // an arena could sweep. It is now maintained incrementally — a task is
    // admitted when its last dependency completes, or when the barrier its
    // phase waits on clears — so the per-event cost is the *ready* set rather
    // than the whole graph.
    //
    // **Ascending task id, exactly as the scan produced.** A `Scheduler` is
    // handed `Decision::ready` as a slice and several of them break a tie by the
    // first entry they see, so the order is part of the interface and not an
    // implementation detail: `PlanOrder` is documented as "the lowest ready task
    // id", and `CacheAware` returns the first of an equal-hit set. Every
    // insertion is therefore a `partition_point` and every removal takes the
    // element out in place. That the two agree is checked rather than argued —
    // see the debug assertion at the head of the loop.
    // **Whether a phase waits for the whole of the one before it.** A barrier
    // phase always does; under `Machine::wave_synchronous` every phase does,
    // which is the executor's own dispatch. One closure rather than three
    // spellings of it — the seed, the admission on completion and the debug
    // oracle all ask, and a topology that differed between them would be two
    // executors.
    let waits_for_the_phase_below =
        |phase: usize| machine.wave_synchronous || graph.is_barrier(phase);
    let mut ready = ReadySet::new(&graph, |phase| {
        phases.phase_open(phase, waits_for_the_phase_below)
    });

    while timeline.remaining() > 0 {
        // **The scan the maintained set replaced, kept as its oracle.** Every
        // test in this crate that runs a simulation runs this comparison — the
        // suite is built in the dev profile — and a release build pays nothing
        // for it. If the two ever part, the incremental admission has missed an
        // edge or a barrier and the finding is here rather than in a ranking
        // that quietly changed.
        #[cfg(debug_assertions)]
        {
            let scanned = ready.scanned(&graph, |phase| {
                phases.phase_open(phase, waits_for_the_phase_below)
            });
            debug_assert_eq!(
                ready.as_slice(),
                scanned.as_slice(),
                "the maintained ready set and the full scan disagree"
            );
        }

        if ready.is_empty() || timeline.at_capacity(topology) {
            // Advance to the next completion. Nothing can start before then.
            if timeline.is_idle() {
                // Nothing ready and nothing running: the graph cannot progress.
                // Reached only by a malformed decomposition, and returning the
                // partial outcome would report a run that did not happen.
                return Err(crate::error::Error::InvalidArgument(format!(
                    "simulate: {} tasks remain, none is ready and none is running. The \
                     task graph has a cycle or a barrier that can never clear.",
                    timeline.remaining()
                )));
            }
            outcome.idle_slot_ns +=
                timeline.idle_slots_until_next_finish(ready.is_empty(), topology);
            for id in timeline.advance_to_next_finish(&graph, topology) {
                residency.finish_task(id);
                outcome.tasks_run += 1;
                ready.complete_dependencies(id, &graph, &dependents, |phase| {
                    phases.phase_open(phase, waits_for_the_phase_below)
                });
                let phase = graph.tasks[id].phase;
                if phases.complete_task(phase, timeline.now()) {
                    // The barrier this completion cleared, for every phase it
                    // cleared it for. `finished_phases` only grows inside
                    // `PhaseProgress`, so a phase
                    // drained here is never held again.
                    //
                    // **`+ 1`, because the test is `finished_phases >= phase`
                    // and not `>`.** A phase whose barrier is cleared by the
                    // phase immediately before it is the ordinary case — its
                    // tasks are admitted during that phase's last completion,
                    // while `finished_phases` still reads the old value — and
                    // `take(finished_phases)` would leave exactly those held for
                    // ever. The debug oracle above catches it, but only on a
                    // fixture that has a barrier phase, which is why
                    // `simulate_ranks` grew one.
                    ready.release_through(phases.release_limit());
                    // **The gather.** A barrier reduces over every contributing
                    // block's fragment, which means holding them all at once —
                    // `n_blocks x payload` resident at one instant, with no term
                    // in `Residency` for it. Recorded here as a peak rather than
                    // added to `peak_bytes`, because a figure the byte budget
                    // does not yet know about must not silently start moving the
                    // number strategies are compared on.
                    let gathered = write_footprints.gather_peak(phase, &graph);
                    outcome.sidecar_gather_peak = outcome.sidecar_gather_peak.max(gathered);
                    // Free what the executor frees after this phase — by
                    // calling the executor's rule, not by restating it.
                    let freed = decomposition.images_freed_after(phase, released, kept);
                    residency.release(&freed);
                }
            }
            continue;
        }

        let worker = timeline.free_worker();
        let pool = topology.pool_of(worker);
        let node = topology.node_of(worker);
        let slot = {
            let running_ids = timeline.running_ids();
            // The tasks held by slots on this worker's own node, from the slot
            // table rather than from `running`, which is task ids and does not
            // say which machine holds one.
            let node_running = timeline.node_running(node, topology);
            let candidates = CandidateWindow::new(&ready, machine.candidate_window);
            let decision = Decision {
                now_ns: timeline.now(),
                nodes: topology.nodes,
                graph: &graph,
                decomposition,
                ready: candidates.as_slice(),
                running: &running_ids,
                node_running: &node_running,
                live_images: residency.live_images(),
                resident_bytes: residency.resident_bytes(),
                read_footprints: &read_footprints,
                cache: caches.decoded(pool),
                worker,
                anchors: timeline.anchors(),
                node_anchors: timeline.node_anchors(),
                grids: read_footprints.grids(),
                images_read: read_footprints.images_read(),
                phase_ns_per_voxel: &phase_rates,
                phase_substages: &substages,
            };
            candidates.clamp_pick(scheduler.pick(&decision))
        };
        // Out of the set as it starts. `slot` indexes the window, and the
        // window is a *prefix* of `ready`, so the two indices are the same
        // number and no translation is needed — which is the reason the window
        // is a prefix rather than a selection. `ReadySet::mark_started` keeps
        // the debug oracle and prefetch eligibility from seeing this task as
        // ready again.
        let id = ready.remove(slot);
        let task = &graph.tasks[id];
        // Time only advances at the head of the loop, so the first dispatch of a
        // phase is its earliest start.
        phases.mark_started(task.phase, timeline.now());

        // The image this phase writes is allocated when its first block starts.
        //
        // **`writes_an_image` is asked of the work, not assumed**, exactly as
        // `Decomposition::peak_image_bytes_with` asks it: a fragment phase may
        // write no image at all, and a walk that allocated one per phase would
        // over-count every plan with a fragment stage in it — which is every
        // plan this project actually runs.
        if let Some((image, bytes)) = write_footprints.allocation(task, decomposition, bytes_of) {
            residency.allocate_once(image, bytes);
        }

        // What this task fetches, and what that costs. A demand fetch queues
        // behind whatever the channel is already carrying — including a
        // prefetch issued for a task that has not started, which is exactly how
        // a too-deep prefetch hurts.
        //
        // **Every image the phase reads, at the region the executor reads it
        // at.** Two corrections in one, and both were charging the wrong thing
        // rather than charging too little of the right one:
        //
        // * `PhaseDecomposition::images_read` instead of the phase's own image
        //   alone. A phase with a `Chain::Source` arm really does traverse two
        //   arrays, and `PhaseTraffic::images_read` has counted them since
        //   before this loop existed.
        // * `BlockGeometry::source` instead of `read`. `source` is what the
        //   block fetches, in the read image's space; `read` is the phase's own
        //   extent. `strategy` reads at `source` — for the input image and for
        //   each source image alike — and the two differ exactly when a phase
        //   changes shape.
        // **A short-circuited block, by the executor's own sequence.** It still
        // fetches its own image — the uniformity test is `env.uniform(&buf)`, so
        // the bytes have to arrive before anything can be skipped — and it still
        // writes, because "the block the work would have produced" is a constant
        // block that must exist. What it skips is the **compute**, and the
        // **source images**, which `strategy` reads inside `if !short_circuited`.
        let skipped = short_circuits(task.index, constant_fraction[task.phase]);
        if skipped {
            outcome.tasks_short_circuited += 1;
        }
        let demand_read = DemandReadTransaction::start(
            task,
            skipped,
            &read_footprints,
            &mut caches,
            pool,
            &mut outcome,
        );
        // A request per chunk, plus the bytes. This is what puts a floor under a
        // small chunk: without it, halving the chunk halves the over-fetch and
        // nothing pays for the extra objects, so a chunk-size sweep improves
        // without bound toward zero.
        let transfer_ns = demand_read.transfer_ns(rates);
        // **A task that fetches nothing does not touch the channel**, and
        // therefore does not wait for it. This used to be `now.max(io_free_at)`
        // unconditionally, which was invisible while the only thing advancing
        // `io_free_at` was a fetch that a hit had already made rare — and became
        // a bug the moment writes began reserving it, because then every task
        // waited behind every prior task's store and the makespan stopped
        // responding to the worker count at all.
        let io_done = if demand_read.needs_channel() {
            let (_, done) = io.reserve(node, timeline.now(), transfer_ns);
            done
        } else {
            timeline.now()
        };
        outcome.io_wait_ns += io_done - timeline.now();
        // The decode is CPU work between the transfer and the compute, so it
        // does not occupy a channel and does not overlap with the fetch that
        // produced its bytes. An **encoded hit** pays the same decode without
        // the fetch, which is the whole of what the second tier trades.
        let decoded = demand_read.decoded_at(io_done, rates);

        let read_voxels = task.geometry.read.voxels() as u64;
        // One input tile per image the block fetches, at the extent it fetches
        // it at, plus the output tile at the extent the op writes — which is the
        // *read* extent, because `BlockOutput::pixels` is over the read extent
        // and the executor slices the valid sub-box out of it.
        //
        // This used to be `read_voxels x dtype x 2` — the same two-buffer figure
        // `PhaseCost::working_set_bytes_per_block` carries, and with the same
        // recorded gap: a phase reading three images was charged as if it read
        // one. That gap remains on `PhaseCost`; closing it there is a change to
        // what the planner *chooses* on and wants its own measurement.
        let input_bytes = demand_read.input_buffer_bytes();
        let output_bytes = write_footprints.output_buffer_bytes(task, decomposition);
        residency.start_task(id, input_bytes + output_bytes);

        // **`S x compute`, one fetch and one store.** The shape `iterate`'s own
        // header states is `S x (read + compute) + write`, and its `read` is the
        // block re-traversing its private buffers rather than the storage read:
        // the substages ping-pong two buffers and `run_iterative_phase` writes
        // only after the loop. So the traversal is inside the compute term, and
        // what repeats here is the compute alone.
        //
        // Pricing at `S == 1` — which this did, and which the planner still does
        // — over-weights the store against the rest by a residual that varies
        // with the block edge, and `iterate.rs` measured the resulting choice at
        // up to **1.125x** the right one.
        //
        // The count folds into the rate rather than multiplying the truncated
        // product, so that `S` substages cost exactly what `S` times the rate
        // costs. Truncating per substage and then multiplying differs by up to
        // one nanosecond per task — 8 ns over this suite's fixture — which is
        // nothing to a ranking and everything to an identity worth asserting.
        // **Contention: a worker's compute slows for each other worker running
        // on its own node.** One coefficient, fitted to the one figure there is
        // — see `MEASURED_CONTENTION`. Counted from `busy`, which is the slot
        // table, because `running` is task ids and a task does not say which
        // machine it is on; this slot has not been filled yet, so the `+ 1` is
        // this task. At one node it is `running.len() + 1`, the expression this
        // replaced, and `contention_counts_only_the_workers_of_one_node` is what
        // says so.
        let concurrent = (timeline.running_on_node(node, topology) + 1) as f64;
        let slowdown = 1.0 + machine.contention * (concurrent - 1.0);
        let compute = if skipped {
            0
        } else {
            (read_voxels as f64 * phase_rates[task.phase] * substages[task.phase] as f64 * slowdown)
                as u64
        };
        ready.mark_started(id);
        // Compute starts when the bytes have landed, not when the slot opened.
        let computed = decoded + compute.max(1);

        // **The write, on the same IO channels the reads use.**
        //
        // The extent and the element type are the executor's own accounting —
        // `strategy`'s `phase_bytes` is `outcome.valid.voxels() x
        // dtype_at(phase + 1)` — rather than a second opinion about what a block
        // stores. `writes_an_image` is asked of the work, so a fragment phase
        // that writes no image is charged nothing.
        //
        // The worker blocks on its store, and the store queues behind whatever
        // else is using the node's storage. This is the request queue the old
        // scalar model was missing: the arrival time is `computed`, not dispatch
        // time, so compute still overlaps compute while reads and writes
        // contend for the same finite IO channels.
        // The image bytes are counted as image bytes and the sidecar bytes as
        // sidecar bytes. They share the duration because the worker blocks on
        // both, but they do not share an outcome counter.
        let finish = StoreTransaction::start(task, &write_footprints, decomposition, &mut outcome)
            .finish_after(computed, node, &mut io, rates, &mut outcome);
        timeline.start(worker, id, finish);

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
        PrefetchIssuance::issue_after_dispatch(
            machine.prefetch_depth,
            id,
            timeline.now(),
            node,
            pool,
            &graph,
            &ready,
            &read_footprints,
            &mut caches,
            &mut io,
            rates,
            &mut outcome,
        );

        outcome.peak_bytes = outcome.peak_bytes.max(residency.resident_bytes());
        outcome.makespan_ns = outcome.makespan_ns.max(finish);
    }

    // The phase spans, summed. A phase that never started contributes nothing,
    // which is a phase with no tasks; `saturating_sub` rather than a subtraction
    // because a phase whose every task was short-circuited can finish in the
    // nanosecond it started.
    outcome.phase_span_ns = phases.phase_span_ns();

    Ok(outcome)
}
