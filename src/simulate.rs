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

use std::collections::{BTreeMap, BTreeSet};

use crate::assemble::ImageId;
use crate::decomposition::Decomposition;
use crate::distributed::cache_model::{ChunkGrid, ModelledCache};
use crate::error::Result;
use crate::fragment::{PhaseWork, SidecarSize};
use crate::graph::TaskGraph;

/// The machine, as the simulator understands one.
///
/// Every field is a **planner lever** — something a plan or a caller chooses —
/// rather than a property of the hardware, except `workers`, which is both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Machine {
    /// Slots. Tasks beyond this queue.
    pub workers: usize,
    /// The byte budget of whatever physically serves a re-read.
    ///
    /// # Which cache this is, decided
    ///
    /// It used to be `cache::ChunkCache`'s budget, and that was a model of a
    /// component nobody constructs: `distributed::cache_model` says so outright
    /// — "`cache::ChunkCache` … has **no non-test construction site**, so no
    /// `Environment::read` is served from one. What can physically serve a
    /// re-read on a node is the page cache, sized by free RAM." So the
    /// simulator's central mechanism — ordering changes hit rate — was
    /// parameterised by an axis that does not exist on the machine.
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
    /// Shared across workers, which is the optimistic reading.
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
    /// How many fetches the storage serves at once. `0` and `1` both mean one.
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
    /// `false` gives each worker `cache_bytes / workers` and counts a chunk
    /// fetched by a second worker as [`Outcome::duplicated_fetches`].
    pub cache_shared: bool,
    /// The share of [`Self::cache_bytes`] held as **encoded** chunks, whose hits
    /// cost a decode. See [`Machine::with_encoded_fraction`].
    pub encoded_fraction: f64,
    /// How much a worker's compute slows for each *other* worker running.
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
            workers: 1,
            cache_bytes: 0,
            prefetch_depth: 0,
            io_channels: 1,
            cache_shared: true,
            encoded_fraction: 0.0,
            contention: 0.0,
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
    /// Chunks fetched by a worker when another worker already held them.
    ///
    /// Always zero when [`Machine::cache_shared`] is true, because then there is
    /// only one cache and the question does not arise. **The quantity a handout
    /// policy exists to reduce**, and the one
    /// `nearest_first_handout_costs_fewer_duplicated_fetches_than_naive_pull`
    /// measures on the real coordinator.
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
    /// Which worker this choice is for.
    ///
    /// A shared cache makes this uninteresting and it is `0` throughout; with
    /// `Machine::cache_shared` false it is the identity a handout policy ranks
    /// against, and [`Decision::cache`] is that worker's own cache.
    pub worker: usize,
    /// Where each worker last finished, for a policy that seeds workers apart.
    /// `None` for a worker that has finished nothing.
    pub anchors: &'a [Option<[f64; 3]>],
    /// Images alive right now, and their sizes.
    pub live_images: &'a [(usize, u64)],
    /// Bytes resident right now: images plus in-flight block buffers.
    pub resident_bytes: u64,
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
    /// Every chunk key one task fetches: each image the phase reads, at
    /// [`crate::geometry::BlockGeometry::source`], on that image's own grid.
    ///
    /// The same walk the event loop performs, exposed so that a scheduler
    /// reasoning about warmth asks the question the run will actually ask. A
    /// scheduler that assembled the keys itself would be a fourth statement of
    /// what a block fetches, and the first one to go stale.
    pub fn chunks_of(&self, task: &crate::graph::Task) -> Vec<u64> {
        self.images_read[task.phase]
            .iter()
            .flat_map(|&image| {
                self.grids
                    .get(&image)
                    .map(|grid| grid.keys(image, &task.geometry.source))
                    .unwrap_or_default()
            })
            .collect()
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
        let seeds: Vec<[f64; 3]> = decision
            .anchors
            .iter()
            .enumerate()
            .filter(|&(worker, _)| worker != decision.worker)
            .filter_map(|(_, anchor)| *anchor)
            .collect();
        let view = crate::distributed::handout::WorkerView {
            anchor: decision.anchors[decision.worker],
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
    // What one block of each phase writes to its sidecar streams, from the
    // declaration. `work` carries the op, so this needs nothing the plan does not
    // already hand over; a phase that is not a fragment phase writes none.
    let sidecar_per_block: Vec<Vec<SidecarSize>> = (0..n)
        .map(|phase| match work.get(phase) {
            Some(crate::fragment::PhaseWork::Fragments(op)) => {
                op.outputs().iter().map(|output| output.size).collect()
            }
            _ => Vec::new(),
        })
        .collect();
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
    let mut indegree: Vec<usize> = graph.tasks.iter().map(|t| t.n_dependencies()).collect();

    // **A grid per image, not one per plan.** `ChunkGrid` is built from a
    // volume, and the images of a plan do not share one: a resampling phase
    // writes an image of a different extent, and a pyramid level is a different
    // extent by construction. One grid over `decomposition.volume` keyed every
    // image against the full-resolution lattice, so the hit rate reported for
    // any phase below the top was fiction.
    //
    // `BTreeMap` rather than a `Vec` because a supplied input's id is not an
    // index into `0..n_images()`.
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
    let pools = if machine.cache_shared {
        1
    } else {
        machine.workers.max(1)
    };
    let per_pool = machine.cache_bytes / pools as u64;
    let encoded_bytes = (per_pool as f64 * machine.encoded_fraction) as u64;
    let mut caches: Vec<ModelledCache> = (0..pools)
        .map(|_| ModelledCache::new(per_pool - encoded_bytes, rates.chunk_bytes))
        .collect();
    let mut encodeds: Vec<ModelledCache> = (0..pools)
        .map(|_| ModelledCache::new(encoded_bytes * ENCODED_RESIDENCY, rates.chunk_bytes))
        .collect();
    // Every chunk any worker has been assigned, for counting the duplicated
    // fetches a second worker causes. A *set*, not an eviction model: which
    // chunks a worker has read is something a coordinator genuinely knows, and
    // it is exactly what a duplicated fetch is made of.
    let mut ever_fetched: BTreeSet<u64> = BTreeSet::new();
    let mut anchors: Vec<Option<[f64; 3]>> = vec![None; machine.workers.max(1)];
    let mut busy: Vec<Option<usize>> = vec![None; machine.workers.max(1)];

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
    // **One free-at time per channel.** A fetch takes the earliest-free one, so
    // `channels == 1` is exactly the serial model this had and anything above it
    // lets concurrent fetches overlap the way real storage does.
    let mut io_free_at: Vec<u64> = vec![0; machine.io_channels.max(1)];
    let mut now: u64 = 0;
    // (finish_ns, task id), kept sorted so the earliest completion is last.
    let mut running: Vec<(u64, usize)> = Vec::new();
    let mut block_bytes: Vec<u64> = vec![0; graph.tasks.len()];
    let mut in_flight_bytes: u64 = 0;
    let mut done_in_phase: Vec<usize> = vec![0; graph.n_phases()];
    // When each phase's first task started and its last one finished, for
    // `Outcome::phase_span_ns`. `None` for a phase that never started, which is
    // a phase with no tasks — its span is nothing rather than zero-to-zero.
    let mut phase_started: Vec<Option<u64>> = vec![None; graph.n_phases()];
    let mut phase_finished: Vec<u64> = vec![0; graph.n_phases()];
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
    let mut ready: Vec<usize> = Vec::new();
    // Tasks whose dependencies are all done and whose phase is a barrier that
    // has not cleared. One list per phase, so a phase's tasks come back in the
    // order they went in — which is ascending, because that is the order they
    // are admitted in.
    let mut barrier_held: Vec<Vec<usize>> = vec![Vec::new(); graph.n_phases()];
    for id in 0..graph.tasks.len() {
        if indegree[id] == 0 {
            let phase = graph.tasks[id].phase;
            if !graph.is_barrier(phase) || finished_phases >= phase {
                ready.push(id);
            } else {
                barrier_held[phase].push(id);
            }
        }
    }

    while remaining > 0 {
        // **The scan the maintained set replaced, kept as its oracle.** Every
        // test in this crate that runs a simulation runs this comparison — the
        // suite is built in the dev profile — and a release build pays nothing
        // for it. If the two ever part, the incremental admission has missed an
        // edge or a barrier and the finding is here rather than in a ranking
        // that quietly changed.
        #[cfg(debug_assertions)]
        {
            let scanned: Vec<usize> = (0..graph.tasks.len())
                .filter(|&id| indegree[id] == 0)
                .filter(|&id| {
                    let phase = graph.tasks[id].phase;
                    !graph.is_barrier(phase) || finished_phases >= phase
                })
                .collect();
            debug_assert_eq!(
                ready, scanned,
                "the maintained ready set and the full scan disagree"
            );
        }

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
                if let Some(slot) = busy.iter().position(|held| *held == Some(id)) {
                    busy[slot] = None;
                    anchors[slot] = Some(crate::distributed::handout::position(&graph, id));
                }
                in_flight_bytes -= block_bytes[id];
                remaining -= 1;
                outcome.tasks_run += 1;
                for &next in &dependents[id] {
                    indegree[next] -= 1;
                    // Its last dependency: admit it, to the ready set or to the
                    // barrier its phase is still waiting on.
                    if indegree[next] == 0 {
                        let phase = graph.tasks[next].phase;
                        if !graph.is_barrier(phase) || finished_phases >= phase {
                            let at = ready.partition_point(|&other| other < next);
                            ready.insert(at, next);
                        } else {
                            barrier_held[phase].push(next);
                        }
                    }
                }
                let phase = graph.tasks[id].phase;
                phase_finished[phase] = phase_finished[phase].max(now);
                done_in_phase[phase] += 1;
                if done_in_phase[phase] == graph.tasks_in_phase(phase).len() {
                    finished_phases = finished_phases.max(phase + 1);
                    // The barrier this completion cleared, for every phase it
                    // cleared it for. `finished_phases` only grows, so a phase
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
                    let cleared = finished_phases + 1;
                    for held in barrier_held.iter_mut().take(cleared) {
                        for id in held.drain(..) {
                            let at = ready.partition_point(|&other| other < id);
                            ready.insert(at, id);
                        }
                    }
                    // **The gather.** A barrier reduces over every contributing
                    // block's fragment, which means holding them all at once —
                    // `n_blocks x payload` resident at one instant, with no term
                    // in `Residency` for it. Recorded here as a peak rather than
                    // added to `peak_bytes`, because a figure the byte budget
                    // does not yet know about must not silently start moving the
                    // number strategies are compared on.
                    if graph.is_barrier(phase) || !sidecar_per_block[phase].is_empty() {
                        let gathered: u64 = graph
                            .tasks_in_phase(phase)
                            .iter()
                            .map(|task| {
                                let core = task.geometry.core.shape3();
                                let read = task.geometry.read.shape3();
                                sidecar_per_block[phase]
                                    .iter()
                                    .filter_map(|size| size.bytes_at_most(core, read))
                                    .sum::<u64>()
                            })
                            .sum();
                        outcome.sidecar_gather_peak = outcome.sidecar_gather_peak.max(gathered);
                    }
                    // Free what the executor frees after this phase — by
                    // calling the executor's rule, not by restating it.
                    let freed = decomposition.images_freed_after(phase, released, kept);
                    live.retain(|&(image, _)| !freed.contains(&image));
                }
            }
            continue;
        }

        let worker = busy
            .iter()
            .position(|slot| slot.is_none())
            .expect("a free worker, since the loop only reaches here below the worker count");
        let pool = if machine.cache_shared { 0 } else { worker };
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
                cache: &caches[pool],
                worker,
                anchors: &anchors,
                grids: &grids,
                images_read: &images_read,
                phase_ns_per_voxel: &phase_rates,
                phase_substages: &substages,
            };
            scheduler.pick(&decision).min(ready.len() - 1)
        };
        // Out of the set as it starts. The scan's equivalent is `indegree[id] =
        // usize::MAX` below, which is still done — it is what the oracle above
        // compares against and what keeps a completed task's decrement from
        // re-admitting a started one.
        let id = ready.remove(slot);
        let task = &graph.tasks[id];
        // Time only advances at the head of the loop, so the first dispatch of a
        // phase is its earliest start.
        phase_started[task.phase].get_or_insert(now);

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
        let fetches: &[usize] = if skipped {
            &images_read[task.phase][..1.min(images_read[task.phase].len())]
        } else {
            &images_read[task.phase]
        };
        let mut misses = 0u64;
        let mut encoded_hits = 0u64;
        for &image in fetches {
            let keys = grids[&image].keys(image, &task.geometry.source);
            let missed_decoded = caches[pool].misses(&keys) as u64;
            // A chunk the decoded tier does not hold may still be in the encoded
            // one, where it is a hit that costs a decode rather than a fetch.
            let not_in_either = keys
                .iter()
                .filter(|key| !caches[pool].holds(**key) && !encodeds[pool].holds(**key))
                .count() as u64;
            encoded_hits += missed_decoded - not_in_either;
            // A chunk this worker must fetch that some worker has already
            // fetched is a duplicated fetch — the thing a handout policy exists
            // to avoid, and invisible under one shared pool.
            for key in &keys {
                if !caches[pool].holds(*key)
                    && !encodeds[pool].holds(*key)
                    && !ever_fetched.insert(*key)
                {
                    outcome.duplicated_fetches += 1;
                }
            }
            caches[pool].note_assigned(&keys);
            encodeds[pool].note_assigned(&keys);
            outcome.cache_hits += keys.len() as u64 - not_in_either;
            outcome.cache_misses += not_in_either;
            outcome.encoded_hits += missed_decoded - not_in_either;
            misses += not_in_either;
        }
        let fetched = misses * rates.chunk_bytes;
        outcome.fetched_bytes += fetched;
        // A request per chunk, plus the bytes. This is what puts a floor under a
        // small chunk: without it, halving the chunk halves the over-fetch and
        // nothing pays for the extra objects, so a chunk-size sweep improves
        // without bound toward zero.
        let transfer =
            (misses as f64 * rates.io_latency_ns + fetched as f64 * rates.io_ns_per_byte) as u64;
        // **A task that fetches nothing does not touch the channel**, and
        // therefore does not wait for it. This used to be `now.max(io_free_at)`
        // unconditionally, which was invisible while the only thing advancing
        // `io_free_at` was a fetch that a hit had already made rare — and became
        // a bug the moment writes began reserving it, because then every task
        // waited behind every prior task's store and the makespan stopped
        // responding to the worker count at all.
        let io_done = if transfer > 0 {
            let channel = io_free_at
                .iter()
                .enumerate()
                .min_by_key(|&(_, free)| *free)
                .map(|(index, _)| index)
                .unwrap_or(0);
            let done = now.max(io_free_at[channel]) + transfer;
            io_free_at[channel] = done;
            done
        } else {
            now
        };
        outcome.io_wait_ns += io_done - now;
        // The decode is CPU work between the transfer and the compute, so it
        // does not occupy a channel and does not overlap with the fetch that
        // produced its bytes. An **encoded hit** pays the same decode without
        // the fetch, which is the whole of what the second tier trades.
        let decoded_bytes = fetched + encoded_hits * rates.chunk_bytes;
        let decoded = io_done + (decoded_bytes as f64 * rates.decode_ns_per_byte) as u64;

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
        let fetch_voxels = task.geometry.source.voxels() as u64;
        let input_bytes: u64 = fetches
            .iter()
            .map(|&image| fetch_voxels * decomposition.dtype_at(image).size_of() as u64)
            .sum();
        let output_bytes = if writes[task.phase] {
            read_voxels * decomposition.dtype_at(task.phase + 1).size_of() as u64
        } else {
            0
        };
        block_bytes[id] = input_bytes + output_bytes;
        in_flight_bytes += block_bytes[id];

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
        // **Contention: a worker's compute slows for each other worker
        // running.** One coefficient, fitted to the one figure there is — see
        // `MEASURED_CONTENTION`. `running` has not yet been pushed, so the count
        // includes this task.
        let concurrent = (running.len() + 1) as f64;
        let slowdown = 1.0 + machine.contention * (concurrent - 1.0);
        let compute = if skipped {
            0
        } else {
            (read_voxels as f64 * phase_rates[task.phase] * substages[task.phase] as f64 * slowdown)
                as u64
        };
        // Indegree is decremented on completion, so mark the task started by
        // making it un-ready. `usize::MAX` cannot be reached by decrementing.
        indegree[id] = usize::MAX;
        // Compute starts when the bytes have landed, not when the slot opened.
        let computed = decoded + compute.max(1);

        // **The write, on the same channel the read came over.**
        //
        // The extent and the element type are the executor's own accounting —
        // `strategy`'s `phase_bytes` is `outcome.valid.voxels() x
        // dtype_at(phase + 1)` — rather than a second opinion about what a block
        // stores. `writes_an_image` is asked of the work, so a fragment phase
        // that writes no image is charged nothing.
        //
        // **The write costs the task its own time, and does not touch the
        // channel.** That is a smaller claim than it looks, and both halves were
        // arrived at by getting it wrong first.
        //
        // The worker really does block on its store — `strategy` writes inside
        // the task — so the duration belongs to the slot, and a plan that stores
        // more finishes later. That is the whole of what this item is for, and it
        // is enough for the ranking: makespan responds to write bytes, and
        // `written_bytes` and `materialised_bytes` record them exactly.
        //
        // Charging the store to the **serial channel** is the part left undone,
        // and deliberately. A store joins the channel when the compute ends, so
        // `io_free_at` would have to move past that compute — and then the next
        // task's *read* queues behind it, and the channel becomes a global lock
        // through which every task's compute is serialised. Measured on the
        // suite's own fixture: a plan that took `203.9 ms` on one worker took
        // `203.9 ms` on eight, to the nanosecond, which is a simulator with no
        // worker axis at all. Accounting it from dispatch instead avoids that
        // and breaks the other end — the channel is then never idle, and the
        // prefetcher, whose whole rule is "only into idle channel time", stops
        // issuing at every depth.
        //
        // Both failures are the same missing thing: a serial channel with an
        // arrival order needs a **request queue**, not another placement of one
        // scalar. That is the IO model being replaced rather than patched — see
        // the `IO latency and IO parallelism` item in
        // `docs/design/simulator-fidelity.md`, which is where read and write
        // contention become one question with one answer.
        let written = task.phase + 1;
        let write_bytes = if writes[task.phase] && written < decomposition.n_images() {
            task.geometry.valid.voxels() as u64 * decomposition.dtype_at(written).size_of() as u64
        } else {
            0
        };
        // The sidecar a fragment block writes, on the same terms as the image
        // write: the worker blocks on it, and the bytes are the declared bound.
        let sidecar_bytes: u64 = sidecar_per_block[task.phase]
            .iter()
            .filter_map(|size| {
                size.bytes_at_most(task.geometry.core.shape3(), task.geometry.read.shape3())
            })
            .sum();
        outcome.sidecar_bytes_written += sidecar_bytes;
        let finish = if write_bytes + sidecar_bytes > 0 {
            let intermediate =
                decomposition.image_kind(written) == crate::decomposition::ImageKind::Intermediate;
            let rate = if intermediate {
                rates.materialise_ns_per_byte
            } else {
                rates.write_ns_per_byte
            };
            // The image bytes are counted as image bytes and the sidecar bytes
            // as sidecar bytes — they share the *duration*, because the worker
            // blocks on both, and nothing else. Folding them into one counter
            // was the first version and it made `written_bytes` disagree with
            // the executor's `RegionWritten` by exactly the sidecar payload.
            if intermediate {
                outcome.materialised_bytes += write_bytes;
            } else {
                outcome.written_bytes += write_bytes;
            }
            // A sidecar is an intermediate by nature: nothing outside the run
            // reads one, and `Lifecycle` is how it goes.
            computed
                + (write_bytes as f64 * rate) as u64
                + (sidecar_bytes as f64 * rates.materialise_ns_per_byte) as u64
        } else {
            computed
        };
        busy[worker] = Some(id);
        running.push((finish, id));
        // Descending by finish time, so the earliest completion is `last`.
        running.sort_by_key(|&(finish, _)| std::cmp::Reverse(finish));

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
                if issued == 0 && io_free_at.iter().all(|&free| free > now) {
                    break;
                }
                // **Only a task whose dependencies are met.** `usize::MAX`
                // marks a task already started; `0` is the readiness test the
                // dispatcher itself uses. Anything else names a task whose
                // input image the producing phase has not written, and a fetch
                // of bytes that do not exist yet is not a prefetch — it is a
                // hit banked against data the run has not produced. Tasks are
                // laid out phase-major (`TaskGraph::build`), so without this
                // every depth past the end of the current phase was doing
                // exactly that.
                if indegree[ahead.id] != 0 {
                    continue;
                }
                let ahead_keys: Vec<u64> = images_read[ahead.phase]
                    .iter()
                    .flat_map(|&image| grids[&image].keys(image, &ahead.geometry.source))
                    .collect();
                let ahead_misses = ahead_keys
                    .iter()
                    .filter(|key| !caches[pool].holds(**key) && !encodeds[pool].holds(**key))
                    .count() as u64;
                if ahead_misses == 0 {
                    continue;
                }
                caches[pool].note_assigned(&ahead_keys);
                encodeds[pool].note_assigned(&ahead_keys);
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
                let channel = io_free_at
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, free)| *free)
                    .map(|(index, _)| index)
                    .unwrap_or(0);
                io_free_at[channel] = io_free_at[channel].max(now)
                    + (ahead_misses as f64 * rates.io_latency_ns
                        + bytes as f64 * rates.io_ns_per_byte) as u64;
                issued += 1;
            }
        }

        let resident = live.iter().map(|&(_, b)| b).sum::<u64>() + in_flight_bytes;
        outcome.peak_bytes = outcome.peak_bytes.max(resident);
        outcome.makespan_ns = outcome.makespan_ns.max(finish);
    }

    // The phase spans, summed. A phase that never started contributes nothing,
    // which is a phase with no tasks; `saturating_sub` rather than a subtraction
    // because a phase whose every task was short-circuited can finish in the
    // nanosecond it started.
    outcome.phase_span_ns = phase_started
        .iter()
        .zip(phase_finished.iter())
        .filter_map(|(started, finished)| started.map(|started| finished.saturating_sub(started)))
        .sum();

    Ok(outcome)
}
