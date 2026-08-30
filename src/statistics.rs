// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Measured cost coefficients, accumulated from real runs and fed back into
// planning.
//
// Why this exists at all
// ----------------------
// Every compute cost in `ops/` is a microbenchmark result, and two things went
// wrong with that in quick succession.
//
// **The unit moved.** Replacing a boxed closure with a monomorphic map made the
// voxelwise map 2.7x faster. Every constant denominated in `MAP_COST` — rank,
// morphology, local, smooth — instantly understated by that factor. The
// *ordering* survived, the absolute figures did not.
//
// **The bench is not reproducible.** Adding rows to `ops::cost::measure` moved
// the `gaussian smooth` row from 55 to 73 ns/voxel with `smooth.rs` untouched;
// at `-C codegen-units=1` both gave 78. The neighbourhood rows swing by about a
// third on codegen-unit partitioning alone, and *any* edit to the bench
// reshuffles a table four modules read their constants off.
//
// A microbenchmark measures a binary nobody runs. **A run measures the binary
// that did the work.** That is the whole argument for this module: the executor
// already times every op application and every region read and write, so the
// evidence is already being produced and thrown away.
//
// The shipped constants are the seed, not a fallback
// --------------------------------------------------
// There may be no first run — a cold planner has to plan *something* — so the
// store bootstraps from the constants in `ops/` and from `CostModel::default`.
// Those seeds do not have to be good, and they are not: the `MAP_COST` family
// understates by ~2.7x and the neighbourhood figures swing ~35% on codegen
// partitioning. **None of that matters for a seed.** What a cold planner needs
// is the *ordering* — a median costs more than a map, an opening more than the
// erosion inside it, a separable smooth far less than a dense element of the
// same reach — and the constants have that right whatever their absolute
// accuracy. Chasing precision in them is chasing a number that measurement will
// supply anyway, on the machine that will actually run the work.
//
// What is stored: coefficients, not timings
// -----------------------------------------
// `CostModel` is four coefficients, and the ops already state their cost
// *parametrically* — per element voxel for rank and morphology, per tap for
// smooth, per bin for isodata. So a stored table of "op X cost Y ns/voxel"
// would be both larger than necessary and unable to say anything about a filter
// size nobody has run.
//
// What is stored instead is **nanoseconds per unit of declared cost**. An op's
// `cost_per_voxel()` is its declared unit count; a run divides the time it
// really took by the units it declared, and the quotient is the coefficient.
// Because the declaration is already parametric, a 27-voxel median and a
// 343-voxel median inform *the same* coefficient — they declare 27 and 343
// units and are expected to take 27 and 343 times the unit cost. A coefficient
// of 1.0 means "the shipped constant was exactly right, in whatever unit the
// measurement is now denominated in".
//
// This is also why the existing constants do not need rescaling. If the map
// really is 2.7x faster than its constant claims, the measured coefficient
// absorbs it, and it does so without a table edit and without pretending the
// 35% codegen spread is signal.
//
// Everything lands in **nanoseconds**, which is the one unit that makes the
// planner's read, write and compute weights commensurable with each other and
// with a log. `CostModel`'s shipped defaults of `1.0` are not nanoseconds; they
// are "one voxelwise map's worth of work", which is what the seed means and is
// why a calibrated model is a wholesale change of unit rather than a nudge.
//
// Where it is stored, and why that is provisional
// -----------------------------------------------
// **JSON, by decision and not by neglect, and it is expected to be replaced.**
// The honest reason to start here is that nobody yet knows what wants storing:
// the set of terms above is a first guess, and a format that is a text file a
// person can read, diff and delete a line out of is the cheapest way to find out
// which of them earn their place. When the design settles, SQLite is the likely
// home.
//
// So the investment is deliberately thin. There is a `FORMAT`/`FORMAT_VERSION`
// pair and a refusal on anything else, because a coefficient read under the
// wrong interpretation is silently wrong rather than absently right — and that
// is *all* the care taken. There is no schema validation, no migration between
// versions, and no tuned layout. Those would be work spent on a format we intend
// to throw away.
//
// **What keeps the swap cheap is that the file's shape is not in the types.**
// `Statistics`, `Snapshot`, `Coefficient`, `Observed`, `Term` and `MachineKey`
// are ordinary Rust values, keyed by domain values rather than by stringified
// ones, and `to_json`/`from_json`/`load`/`save` are the only four functions in
// this module that know a file exists. Nothing the planner or the recorder
// touches has ever seen a `serde_json::Value`. Replacing the four is one
// afternoon; unpicking a `Value` out of a struct field would not have been.
//
// **There is one writer, and that is architecture rather than convention.**
// `save` reads the whole file, merges, and renames a complete replacement over
// it, which would be a poor story under concurrent savers — and there are none.
// A worker does not record: it emits events through `distributed::worker`'s
// `Outbox` and the coordinator receives them into its own `ExecutionLog`, so a
// measurement travels worker -> coordinator as an *observation*, by the same
// path as every other fact about a run. Whoever plans and executes is therefore
// the only process with a `Recorder` on it, by construction. The usual first
// reason to leave a text file — many writers contending on it — does not apply
// here and is not pending. What might still force a move is smaller and slower:
// rewrite cost, once the table is large enough that a run pays to rewrite
// coefficients it never touched, and wanting to *query* — "the coefficient for
// this term on this machine" — rather than parse everything to find out.
// Neither bites at a handful of terms per machine, eight records each, read once
// per process, which is why JSON is right today and will not be for ever.
//
// **What the coordinator path does cost is exactness, and it is inherited.**
// `distributed::Coordinator`'s header is explicit that events are
// fire-and-forget and deliberately unacknowledged, because acknowledging them
// would put the coordinator on a worker's critical path; the price is that the
// last events of a run are still in flight when the last task completes, and the
// coordinator waits for the stream to go *quiet* rather than for a fixed delay,
// which bounds the loss without making it zero. So:
//
// * **a coefficient is derived from most of a run, not provably all of it.**
//   That is fine for what it is — an average over hundreds of thousands of
//   events does not move because a few of the last ones did not arrive.
// * **the evidence figures on a `Coefficient` are approximate.** They are a
//   weight, telling a consumer roughly how much is behind a number. They are not
//   a ledger, and nothing should reconcile against them.
//
// Neither is a defect. Both are obvious today and mysterious in six months.
//
// What was added to the hot path
// ------------------------------
// **Nothing.** `Event::OpApplied` already carries `duration_ns` and the extent
// it was computed over; `strategy.rs` already brackets every `Environment::read`
// and `Environment::write` with `Instant::now()`; `EnvCounters` already counts
// voxels, bytes and chunks. `Recorder` is an ordinary `EventListener` reading
// the stream that a run emits whether or not anybody listens.
//
// Its per-event cost is two or three `fetch_add`s on preallocated atomics and no
// allocation, no lock and no map lookup — strictly cheaper than the built-in
// `ExecutionLog`, which every run already pays a mutex and a `Vec::push` for.
// The one thing it needs that the stream does not carry is each slot's
// *declared* cost, and it takes that from the `Chain` at construction rather
// than widening the event: adding an `f64` to `OpApplied` would have put a
// number on the wire to save a lookup the listener can do once.
//
// Two properties that are not negotiable
// --------------------------------------
// **A run without a store behaves exactly as it does today.** An empty or
// absent snapshot returns the caller's `CostModel` unchanged — the same value,
// not an equivalent one — so every plan, every fingerprint and every parity
// figure is untouched. See `Snapshot::calibrate`.
//
// **Reproducibility stays answerable.** A plan used to be a function of
// metadata alone, so the same workflow planned identically twice. With
// statistics in the loop it can plan differently after a run, which is the
// adaptive behaviour that is wanted — but it must not become *unanswerable*.
// So the evidence is frozen into a `Snapshot` before planning, the snapshot has
// a `fingerprint`, and `PlanIdentity` pairs that fingerprint with the
// decomposition's. A plan is reproducible given `(metadata, snapshot)`; see
// `PlanIdentity` for exactly what that guarantees and what it does not.
//
// The plan stays **data-blind** either way. A coefficient is a recorded
// observation about the machine and the binary, not a property of the array in
// front of the planner — the same argument that lets a run report substage
// counts without the plan depending on them.

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};

use crate::decomposition::{CostModel, Decomposition};
use crate::error::{Error, Result};
use crate::listener::EventListener;
use crate::log::{Event, ExecutionLog};
use crate::op::Chain;

/// The store's on-disk format name. Matched exactly by a reader, so it is
/// opaque and stable rather than descriptive.
pub const FORMAT: &str = "blockflow.statistics";

/// The on-disk format's version. Bumped when the *meaning* of a field changes;
/// a reader that does not recognise it refuses rather than guesses, because a
/// coefficient read under the wrong interpretation is worse than no coefficient.
pub const FORMAT_VERSION: u64 = 1;

/// How many runs' worth of evidence is kept per coefficient.
///
/// Bounded so the file cannot grow without limit, and small so that evidence
/// *ages out*: a machine that gets faster, a binary that gets rebuilt or a run
/// on an unrepresentative volume all stop mattering after this many runs
/// without anybody having to decide that they should. Eight is enough that one
/// odd run is diluted rather than decisive and few enough that a genuine change
/// is reflected within a working session.
pub const RETAINED_RUNS: usize = 8;

// ---------------------------------------------------------------- terms --

/// How many runs must have seen a term before its measurement is believed.
///
/// **Three, because three is where a majority starts**, and that is a derivation
/// rather than a choice: [`Statistics::snapshot`] takes the *weighted median* of
/// the retained runs, and a median needs three points to be a median rather than
/// an average of two. Three is the smallest number at which one contaminated run
/// can be outvoted. Two would leave the estimator a mean of two, where a single
/// stalled run still carries half the answer.
///
/// # Why a term needs protecting at all, measured
///
/// A `CostModel`'s four coefficients are **not** fitted from comparable amounts
/// of evidence, and it is not the unit counts that differ — it is the wall time.
/// The three-op chain in `tests/statistics.rs`, 1120 fits from eight concurrent
/// threads at a load average of 41:
///
/// ```text
///   term        |  p50 |  p99 |   max | max/p50 | wall time behind it
///   Read        | 2.89 | 8.66 | 54.11 |     19x |  0.40 ms
///   Write       | 2.71 | 6.51 | 64.80 |     24x |  0.18 ms
///   Materialise | 2.24 | 9.20 | 79.02 |     35x |  0.15 ms
///   Compute     | 5.44 |10.83 | 12.63 |    2.3x | 50.0   ms
/// ```
///
/// Every term in that plan is fitted from **two to four events**, so event count
/// is not the discriminator either. What separates them is that a scheduler
/// preemption costs about the same number of milliseconds whatever it
/// interrupts: it is a rounding error against 50 ms of compute and it is *the
/// entire measurement* against 0.15 ms of materialisation. So the three I/O
/// terms are each one stall away from being wrong by a factor of 30.
///
/// That would be a curiosity if the fragile terms were unimportant. They are the
/// opposite: `materialise` against `write` is exactly what pricing a **phase
/// cut** compares, so the least reproducible coefficient in the model is the one
/// that decides the shape of the plan. Measured, a fitted model re-planned
/// differently in 4 of 1080 fits, every one of them fusing two phases into one,
/// and every one with a materialise coefficient in the 63-83 band above.
///
/// # What this costs when it refuses something good, measured in both directions
///
/// Almost nothing, and that is the finding rather than the hope. A term without
/// a usable measurement does not fall back to its shipped *number* —
/// [`Snapshot::calibrate`] gives it the shipped **ratio** converted into
/// nanoseconds by the anchor, which is the path an absent term has always taken.
/// The anchor is work-weighted, so it is dominated by `Compute`: the one term
/// this rule has no cause to refuse is also the one that sets the unit.
///
/// 480 held-out predictions, eight concurrent threads, load average 47:
///
/// ```text
///   model                | p50    | p90    | max
///   shipped constants    | 5.38x  | 6.09x  | 10.66x
///   all four calibrated  | 1.03x  | 1.17x  |  2.04x
///   Compute alone        | 1.03x  | 1.15x  |  2.02x
/// ```
///
/// Refusing the three fragile terms is **free to two decimal places at the
/// median and slightly better at the ninetieth**, because what they were
/// contributing was mostly noise. The direction with a price on it is the other
/// one: believing a thin coefficient moved a consumer's chosen block edge by a
/// rung, and a stopwatch put the new rung **1.06x** slower.
///
/// The rule and the estimator are separate mechanisms and both are needed.
/// Under the old pooled quotient a fitted model re-planned differently in **4 of
/// 1080** fits; with the median and this rule, **0 of 480**, and a store holding
/// one or two runs cannot re-plan at all because it calibrates nothing.
pub const REPRODUCTIONS: usize = 3;

/// What a coefficient is *per*.
///
/// The four that `CostModel` can hold, plus two diagnostics it cannot. Each is
/// nanoseconds per one unit of the named thing.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Term {
    /// Nanoseconds per voxel fetched from an image, over every image a block
    /// reads. Feeds `CostModel::read_cost_per_voxel`.
    Read,
    /// Nanoseconds per voxel written to the **workflow output**. Feeds
    /// `CostModel::write_cost_per_voxel`.
    Write,
    /// Nanoseconds per voxel written to an **intermediate** image. Feeds
    /// `CostModel::materialise_cost_per_voxel`, which is a separate weight for
    /// exactly this reason: an intermediate compresses differently from an
    /// output, so one number over-values fusing late stages.
    Materialise,
    /// Nanoseconds per unit of declared compute — `voxels x cost_per_voxel`,
    /// summed over every op the run applied. Feeds `CostModel::compute_scale`,
    /// and is the term that makes the shipped `MAP_COST` family's absolute
    /// scale irrelevant.
    Compute,
    /// `Compute`, for one op family in isolation. **Recorded, reported and —
    /// since `CostModel::compute_of` existed to hold it — used**:
    /// `Snapshot::calibrate` turns each believable one into a dimensionless
    /// correction, its rate over the run-wide `Compute`, and the planner
    /// multiplies the family's declared cost by it. It was kept for a long time
    /// as the evidence for a gap it now fills; `Snapshot::families` is the raw
    /// reading and `Snapshot::family_spread` the size of what it corrects.
    ComputeOf(String),
    /// Nanoseconds per **byte** read. Diagnostic, for the same reason:
    /// `CostModel` prices reads per voxel, and a run whose images have
    /// different element types has no single bytes-per-voxel.
    ReadBytes,
    /// Nanoseconds per byte written, over both destinations. Diagnostic.
    WriteBytes,
}

impl Term {
    /// The string this term is stored and fingerprinted under.
    pub fn key(&self) -> String {
        match self {
            Term::Read => "read.ns_per_voxel".to_string(),
            Term::Write => "write.ns_per_voxel".to_string(),
            Term::Materialise => "materialise.ns_per_voxel".to_string(),
            Term::Compute => "compute.ns_per_unit".to_string(),
            Term::ComputeOf(name) => format!("compute.ns_per_unit/{name}"),
            Term::ReadBytes => "read.ns_per_byte".to_string(),
            Term::WriteBytes => "write.ns_per_byte".to_string(),
        }
    }

    /// The inverse of [`Term::key`]. `None` for a key this build does not know,
    /// which a reader treats as a term to carry through untouched rather than
    /// as a corrupt file.
    pub fn parse(key: &str) -> Option<Term> {
        Some(match key {
            "read.ns_per_voxel" => Term::Read,
            "write.ns_per_voxel" => Term::Write,
            "materialise.ns_per_voxel" => Term::Materialise,
            "compute.ns_per_unit" => Term::Compute,
            "read.ns_per_byte" => Term::ReadBytes,
            "write.ns_per_byte" => Term::WriteBytes,
            other => Term::ComputeOf(other.strip_prefix("compute.ns_per_unit/")?.to_string()),
        })
    }
}

// ---------------------------------------------------------- machine key --

/// What a coefficient is evidence *about*.
///
/// **A coefficient measured elsewhere is not evidence here**, so the store is
/// partitioned by this and a snapshot only ever reads the partition matching the
/// running machine. There is no fuzzy match and no nearest neighbour: a key that
/// differs in any field is a different machine as far as this file is concerned,
/// and its records sit in the file being no use to anybody, which is the right
/// failure — a shared store on a network filesystem must not quietly tell one
/// host what another host's cores can do.
///
/// **Why these fields.** Each one moves nanoseconds per voxel by more than the
/// spread of the measurement itself:
///
/// * `host` — the machine. Two boxes of the same architecture differ by memory
///   bandwidth, clock and thermal budget, and nothing cheap distinguishes them
///   but their name.
/// * `os`, `arch`, `pointer_width` — the target. A different instruction set
///   vectorises the inner loops differently; this is the axis along which the
///   ops' *relative* costs move, not just their scale.
/// * `cpus` — how much parallelism the run was competing for. A per-voxel figure
///   taken with 32 workers on 32 cores and one taken with 32 workers on 4 are
///   measurements of different systems. Taken from `available_parallelism`, so a
///   cgroup-limited container is correctly a different machine from its host.
/// * `profile` — debug or release. Worth 10-50x on this code, which dwarfs every
///   other entry here.
///
/// **What is deliberately *not* in the key: the binary's identity.** A rebuild
/// reshuffles codegen-unit partitioning and moves the neighbourhood figures by
/// about a third — the very instability that motivated this module. Keying on
/// the binary would be defensible and would also make the store useless, since
/// it would be invalidated by every `cargo build`. The bounded run history is
/// the answer instead: a stale binary's evidence is diluted by the next run and
/// gone within [`RETAINED_RUNS`]. Surviving a rebuild is exactly the thing a
/// microbenchmark could not offer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MachineKey {
    pub host: String,
    pub os: String,
    pub arch: String,
    pub pointer_width: u32,
    pub cpus: usize,
    /// `"debug"` or `"release"`, from `debug_assertions`.
    pub profile: String,
}

impl MachineKey {
    /// The machine this process is running on.
    pub fn detect() -> Self {
        Self {
            host: host_name(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            pointer_width: usize::BITS,
            cpus: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            profile: if cfg!(debug_assertions) {
                "debug".to_string()
            } else {
                "release".to_string()
            },
        }
    }

    /// A single string, for use as a map key and in the fingerprint.
    pub fn id(&self) -> String {
        format!(
            "{}/{}/{}/{}/{}/{}",
            self.host, self.os, self.arch, self.pointer_width, self.cpus, self.profile
        )
    }

    fn to_json(&self) -> Value {
        json!({
            "host": self.host,
            "os": self.os,
            "arch": self.arch,
            "pointer_width": self.pointer_width,
            "cpus": self.cpus,
            "profile": self.profile,
        })
    }

    fn from_json(value: &Value) -> Option<Self> {
        Some(Self {
            host: value.get("host")?.as_str()?.to_string(),
            os: value.get("os")?.as_str()?.to_string(),
            arch: value.get("arch")?.as_str()?.to_string(),
            pointer_width: value.get("pointer_width")?.as_u64()? as u32,
            cpus: value.get("cpus")?.as_u64()? as usize,
            profile: value.get("profile")?.as_str()?.to_string(),
        })
    }
}

/// The machine's name, without a dependency.
///
/// Four attempts in decreasing order of reliability, and `"unknown"` rather than
/// a failure at the end: a host that will not say its name should still be able
/// to accumulate statistics, it just shares a partition with every other
/// anonymous host — which is a wrong *number*, never a wrong voxel.
fn host_name() -> String {
    for path in ["/proc/sys/kernel/hostname", "/etc/hostname"] {
        if let Ok(text) = std::fs::read_to_string(path) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }
    for name in ["HOSTNAME", "COMPUTERNAME"] {
        if let Ok(value) = std::env::var(name) {
            if !value.is_empty() {
                return value;
            }
        }
    }
    "unknown".to_string()
}

// -------------------------------------------------------- observations --

/// What one run saw of one term: how much of the thing happened, and how long
/// it took.
///
/// The *pair* is stored rather than the quotient, so that merging two runs
/// weights them by the work behind them rather than treating a run over eight
/// blocks as equal evidence to a run over eight thousand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observed {
    /// Voxels, bytes, or units of declared compute — whatever the [`Term`] is
    /// *per*.
    pub units: f64,
    pub nanos: f64,
}

impl Observed {
    /// Whether this is a measurement at all.
    ///
    /// Zero time over non-zero work is not a fast run, it is an absent clock —
    /// which is exactly what an `AccountingEnvironment` produces, since it
    /// counts bytes without moving them. Such a record is refused rather than
    /// recorded as a coefficient of zero, which would tell the planner that
    /// everything is free.
    pub fn is_evidence(&self) -> bool {
        self.units.is_finite() && self.nanos.is_finite() && self.units > 0.0 && self.nanos > 0.0
    }
}

/// Everything one run observed, ready to be recorded.
#[derive(Debug, Clone, PartialEq)]
pub struct RunObservations {
    pub machine: MachineKey,
    pub terms: BTreeMap<Term, Observed>,
}

impl RunObservations {
    pub fn get(&self, term: &Term) -> Option<Observed> {
        self.terms.get(term).copied()
    }

    /// Whether this run observed anything worth recording.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

// ------------------------------------------------------------ recorder --

/// The listener that turns a run into coefficients.
///
/// Advisory in the sense `listener.rs` requires: it sees `&Event` and has no
/// channel to the scheduler, the environment or a buffer. A panic in it is
/// caught, counted and the run continues.
///
/// **Attach it to runs that do real work.** An `AccountingEnvironment` reports
/// zero nanoseconds because it moves no bytes, and while [`Observed::is_evidence`]
/// refuses such a record outright, the honest statement is that a simulation is
/// not a measurement and recording is opt-in for that reason.
pub struct Recorder {
    machine: MachineKey,
    /// Per chain slot: its display name and its *declared* cost per voxel.
    ///
    /// Read off the `Chain` once, at construction, rather than carried on the
    /// event: `Event::OpApplied` already names the slot, and a slot index is a
    /// perfectly good key into a table the listener built for itself.
    slots: Vec<(String, f64)>,
    /// Voxels and nanoseconds per slot. One pair of atomics per slot, allocated
    /// before the run, so an `OpApplied` costs two `fetch_add`s and an index.
    per_slot: Vec<(AtomicU64, AtomicU64)>,
    /// The image the workflow output lands in. A write below it is a
    /// materialisation; a write to it is the output.
    output_image: usize,
    read: Traffic,
    written: Traffic,
    materialised: Traffic,
}

#[derive(Default)]
struct Traffic {
    voxels: AtomicU64,
    bytes: AtomicU64,
    nanos: AtomicU64,
}

impl Traffic {
    fn add(&self, voxels: u64, bytes: u64, nanos: u64) {
        self.voxels.fetch_add(voxels, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        self.nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    fn read(&self) -> (f64, f64, f64) {
        (
            self.voxels.load(Ordering::Relaxed) as f64,
            self.bytes.load(Ordering::Relaxed) as f64,
            self.nanos.load(Ordering::Relaxed) as f64,
        )
    }
}

impl Recorder {
    /// A recorder for one run of `chain` under `decomposition`.
    ///
    /// Both are needed and neither is redundant: the chain says what each slot
    /// *declared* it would cost, which is the denominator of every compute
    /// coefficient, and the decomposition says how many phases there are, which
    /// is what distinguishes an intermediate write from the workflow's output.
    pub fn new(chain: &Chain, decomposition: &Decomposition) -> Self {
        let slots = chain
            .slots()
            .iter()
            .map(|slot| (slot.display_name(), slot.cost_per_voxel()))
            .collect::<Vec<_>>();
        let per_slot = slots
            .iter()
            .map(|_| (AtomicU64::new(0), AtomicU64::new(0)))
            .collect();
        Self {
            machine: MachineKey::detect(),
            slots,
            per_slot,
            output_image: decomposition.n_phases(),
            read: Traffic::default(),
            written: Traffic::default(),
            materialised: Traffic::default(),
        }
    }

    /// What the run observed, aggregated.
    ///
    /// **A weighted mean over the run, not a low-order statistic**, and the
    /// difference from `ops::cost::measure` is deliberate. A microbenchmark
    /// takes the minimum because contention is *contamination*: it is measuring
    /// the code, and a sample slowed by another process is a sample of the wrong
    /// thing. A run measures the work as it actually happens, where contention
    /// is part of the cost being predicted — and the planner predicts *totals*,
    /// so `total_nanos / total_units` is precisely the coefficient that
    /// reproduces the total that was observed.
    pub fn observations(&self) -> RunObservations {
        let mut terms: BTreeMap<Term, Observed> = BTreeMap::new();
        let mut push = |term: Term, units: f64, nanos: f64| {
            let observed = Observed { units, nanos };
            if observed.is_evidence() {
                terms.insert(term, observed);
            }
        };

        let (read_voxels, read_bytes, read_nanos) = self.read.read();
        push(Term::Read, read_voxels, read_nanos);
        push(Term::ReadBytes, read_bytes, read_nanos);

        let (write_voxels, write_bytes, write_nanos) = self.written.read();
        let (mat_voxels, mat_bytes, mat_nanos) = self.materialised.read();
        push(Term::Write, write_voxels, write_nanos);
        push(Term::Materialise, mat_voxels, mat_nanos);
        push(
            Term::WriteBytes,
            write_bytes + mat_bytes,
            write_nanos + mat_nanos,
        );

        // Compute, globally and per family. The global term is the one the
        // model can hold; the families are the evidence that it should be able
        // to hold more.
        let mut total_units = 0.0_f64;
        let mut total_nanos = 0.0_f64;
        let mut families: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
        for (index, (name, declared)) in self.slots.iter().enumerate() {
            let voxels = self.per_slot[index].0.load(Ordering::Relaxed) as f64;
            let nanos = self.per_slot[index].1.load(Ordering::Relaxed) as f64;
            let units = voxels * declared;
            total_units += units;
            total_nanos += nanos;
            let entry = families.entry(name.as_str()).or_insert((0.0, 0.0));
            entry.0 += units;
            entry.1 += nanos;
        }
        push(Term::Compute, total_units, total_nanos);
        for (name, (units, nanos)) in families {
            push(Term::ComputeOf(name.to_string()), units, nanos);
        }

        RunObservations {
            machine: self.machine.clone(),
            terms,
        }
    }

    /// The machine these observations are about.
    pub fn machine(&self) -> &MachineKey {
        &self.machine
    }
}

impl EventListener for Recorder {
    fn on_event(&self, event: &Event) {
        match event {
            Event::RegionRead {
                voxels,
                bytes,
                duration_ns,
                ..
            } => self.read.add(*voxels as u64, *bytes, *duration_ns),
            Event::RegionWritten {
                image,
                voxels,
                bytes,
                duration_ns,
                ..
            } => {
                // Which side of the phase boundary this write is on, decided
                // from the image rather than from a neighbouring event: the
                // `BlockWritten` that says `materialised` is emitted from the
                // same thread immediately after, but listeners see concurrent
                // blocks interleaved, so pairing on adjacency would be wrong
                // under any concurrency at all.
                let traffic = if *image < self.output_image {
                    &self.materialised
                } else {
                    &self.written
                };
                traffic.add(*voxels as u64, *bytes, *duration_ns);
            }
            Event::OpApplied {
                slot,
                over,
                duration_ns,
                ..
            } => {
                if let Some((voxels, nanos)) = self.per_slot.get(*slot) {
                    voxels.fetch_add(over.voxels() as u64, Ordering::Relaxed);
                    nanos.fetch_add(*duration_ns, Ordering::Relaxed);
                }
            }
            // A short-circuited block did no work and took no time, so it
            // contributes neither units nor nanoseconds. That the *plan* still
            // priced it is a gap between the model and the run, and it belongs
            // in the model rather than in a fudge here.
            _ => {}
        }
    }
}

// ------------------------------------------------------------- the store --

/// One machine's accumulated evidence: per term, the most recent
/// [`RETAINED_RUNS`] runs, oldest first.
type Machine = BTreeMap<Term, VecDeque<Observed>>;

/// The store: measured coefficients, per machine, across runs.
///
/// **Absent is empty and empty changes nothing.** [`Statistics::load`] of a path
/// that does not exist returns an empty store, whose snapshot returns the
/// caller's `CostModel` unchanged.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Statistics {
    machines: BTreeMap<MachineKey, Machine>,
}

impl Statistics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.machines.is_empty()
    }

    /// Fold one run's observations in.
    ///
    /// **A measurement displaces a seed outright, on the first observation.**
    /// The alternative — blending a measurement towards a constant nobody
    /// believes — would mean the seed is never fully displaced and the store's
    /// accuracy has a floor set by the least trustworthy number in it. The
    /// failure mode that argues for a blend is a single unrepresentative run,
    /// and it is answered by the *run history* instead: a coefficient is the
    /// work-weighted mean over the last [`RETAINED_RUNS`] runs, so one odd run
    /// is decisive only until there is a second, is diluted from then on, and is
    /// gone entirely after eight. Meanwhile `Coefficient::runs` and
    /// `Coefficient::units` are on the snapshot, so a caller that wants to hold
    /// out for more evidence can see exactly how much there is.
    pub fn record(&mut self, run: &RunObservations) {
        let entry = self.machines.entry(run.machine.clone()).or_default();
        for (term, observed) in &run.terms {
            if !observed.is_evidence() {
                continue;
            }
            let history = entry.entry(term.clone()).or_default();
            history.push_back(*observed);
            while history.len() > RETAINED_RUNS {
                history.pop_front();
            }
        }
    }

    /// Freeze what is known about `machine` into an immutable snapshot.
    ///
    /// Freezing is what makes planning reproducible: the planner is handed a
    /// value with a fingerprint, not a live accumulator that another thread may
    /// be adding to.
    pub fn snapshot(&self, machine: &MachineKey) -> Snapshot {
        let coefficients = self
            .machines
            .get(machine)
            .map(|entry| {
                entry
                    .iter()
                    .filter_map(|(term, history)| {
                        let units: f64 = history.iter().map(|o| o.units).sum();
                        let nanos: f64 = history.iter().map(|o| o.nanos).sum();
                        weighted_median(history).map(|nanos_per_unit| {
                            (
                                term.clone(),
                                Coefficient {
                                    nanos_per_unit,
                                    runs: history.len(),
                                    units,
                                    total_nanos: nanos,
                                },
                            )
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Snapshot::new(machine.clone(), coefficients)
    }

    /// [`Statistics::snapshot`] for the machine this process is on.
    pub fn snapshot_here(&self) -> Snapshot {
        self.snapshot(&MachineKey::detect())
    }

    /// The machines this store holds evidence about.
    pub fn machines(&self) -> Vec<MachineKey> {
        self.machines.keys().cloned().collect()
    }

    // ------------------------------------------------------- persistence --

    /// Read a store. **A path that does not exist is an empty store**, not an
    /// error: statistics are advisory, and a first run has nowhere to have read
    /// them from.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Self::new()),
            Err(err) => {
                return Err(Error::InvalidArgument(format!(
                    "statistics: reading {}: {err}",
                    path.display()
                )))
            }
        };
        Self::from_json(&text)
    }

    /// Write a store, merging with whatever is already at `path`.
    ///
    /// **Merge, then replace atomically.** Two processes may finish at the same
    /// time; reading first means the loser of the race loses its own run's
    /// records rather than the winner's whole history, and writing through a
    /// temporary plus `rename` means a reader never sees half a file. There is
    /// no lock: a lost observation is a lost observation, never a wrong answer,
    /// and a lock file that outlives a crashed run would be a worse failure than
    /// the one it prevents.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref();
        let mut merged = Self::load(path)?;
        merged.merge(self);
        let temporary = path.with_extension("statistics.tmp");
        std::fs::write(&temporary, merged.to_json()).map_err(|err| {
            Error::InvalidArgument(format!(
                "statistics: writing {}: {err}",
                temporary.display()
            ))
        })?;
        std::fs::rename(&temporary, path).map_err(|err| {
            Error::InvalidArgument(format!(
                "statistics: renaming into {}: {err}",
                path.display()
            ))
        })
    }

    /// Fold `other`'s records in after this store's own, keeping the bound.
    pub fn merge(&mut self, other: &Statistics) {
        for (key, machine) in &other.machines {
            let entry = self.machines.entry(key.clone()).or_default();
            for (term, history) in machine {
                let mine = entry.entry(term.clone()).or_default();
                for observed in history {
                    mine.push_back(*observed);
                }
                while mine.len() > RETAINED_RUNS {
                    mine.pop_front();
                }
            }
        }
    }

    /// The store as JSON.
    ///
    /// **JSON, by hand, because the crate has `serde_json` and not `serde`** —
    /// the same trade `distributed/wire.rs` makes and for the same reason. The
    /// file is small (a handful of terms per machine, eight records each) and
    /// read once per process, so parsing cost is not a consideration; being
    /// *readable* is, because the first thing anybody does with a suspect
    /// coefficient is look at it, and the second is delete the line.
    ///
    /// `f64` round-trips exactly through `serde_json`, so a store written and
    /// read back holds bit-identical coefficients.
    ///
    /// **Provisional**, and one of only four functions here that knows a file
    /// exists. See the module header for what would trigger the move to
    /// something else and why the swap is contained.
    pub fn to_json(&self) -> String {
        let machines: Vec<Value> = self
            .machines
            .iter()
            .map(|(key, machine)| {
                let terms: Vec<Value> = machine
                    .iter()
                    .map(|(term, history)| {
                        json!({
                            "term": term.key(),
                            "runs": history.iter().map(|o| json!({
                                "units": o.units,
                                "nanos": o.nanos,
                            })).collect::<Vec<_>>(),
                        })
                    })
                    .collect();
                json!({ "machine": key.to_json(), "terms": terms })
            })
            .collect();
        serde_json::to_string_pretty(&json!({
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "retained": RETAINED_RUNS,
            "machines": machines,
        }))
        .unwrap_or_else(|_| String::from("{}"))
    }

    /// Parse a store.
    ///
    /// Refuses an unknown `format` or `version` rather than guessing: a
    /// coefficient read under the wrong interpretation is worse than no
    /// coefficient, because it is silently wrong instead of absently right.
    pub fn from_json(text: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text)
            .map_err(|err| Error::InvalidArgument(format!("statistics: not JSON: {err}")))?;
        let format = value.get("format").and_then(Value::as_str).unwrap_or("");
        if format != FORMAT {
            return Err(Error::InvalidArgument(format!(
                "statistics: format is {format:?}, expected {FORMAT:?}"
            )));
        }
        let version = value.get("version").and_then(Value::as_u64).unwrap_or(0);
        if version != FORMAT_VERSION {
            return Err(Error::InvalidArgument(format!(
                "statistics: version {version}, this build reads {FORMAT_VERSION}"
            )));
        }
        let mut machines = BTreeMap::new();
        for entry in value
            .get("machines")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            let Some(key) = entry.get("machine").and_then(MachineKey::from_json) else {
                return Err(Error::InvalidArgument(
                    "statistics: a machine entry has no usable key".to_string(),
                ));
            };
            let mut terms: Machine = BTreeMap::new();
            for term in entry
                .get("terms")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or(&[])
            {
                // A term this build does not recognise is skipped rather than
                // carried: it cannot be interpreted, so it cannot be evidence,
                // and keeping it as an opaque string would put the file's shape
                // back into the in-memory model to no purpose.
                let Some(name) = term
                    .get("term")
                    .and_then(Value::as_str)
                    .and_then(Term::parse)
                else {
                    continue;
                };
                let mut history = VecDeque::new();
                for run in term
                    .get("runs")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
                {
                    let units = run.get("units").and_then(Value::as_f64).unwrap_or(0.0);
                    let nanos = run.get("nanos").and_then(Value::as_f64).unwrap_or(0.0);
                    let observed = Observed { units, nanos };
                    if observed.is_evidence() {
                        history.push_back(observed);
                    }
                }
                if !history.is_empty() {
                    terms.insert(name, history);
                }
            }
            machines.insert(key, terms);
        }
        Ok(Self { machines })
    }
}

/// The **weighted median** of a term's retained runs: each run contributes its
/// own nanoseconds-per-unit, weighted by the units behind it, and the answer is
/// the quotient at the halfway point of the total weight.
///
/// **Median rather than pooled mean, and that is the whole robustness story.**
/// The pooled quotient — `sum(nanos) / sum(units)` — is the estimator that
/// reproduces the total, and it is also the estimator in which one stalled run
/// contributes every one of its nanoseconds. A stall adds *time* and no *work*,
/// so it inflates the numerator alone; against a term fitted from 0.15 ms of
/// wall time it becomes the whole coefficient. See [`REPRODUCTIONS`] for the
/// measurement.
///
/// **Weighted rather than plain, because the weighting is a documented design
/// decision and this must not quietly undo it.** [`Observed`] stores the pair
/// rather than the quotient precisely so that a run over eight thousand blocks
/// is not equal evidence to a run over eight, and a plain median would throw
/// that away. A stall does not change a run's weight, so weighting costs nothing
/// here and keeps the property the store was built for.
///
/// `None` when nothing usable is recorded, which is what an
/// `AccountingEnvironment`'s zero-nanosecond records produce.
fn weighted_median(history: &VecDeque<Observed>) -> Option<f64> {
    let mut samples: Vec<(f64, f64)> = history
        .iter()
        .filter(|o| o.is_evidence())
        .map(|o| (o.nanos / o.units, o.units))
        .collect();
    if samples.is_empty() {
        return None;
    }
    // `total_cmp`, never `f64::min`/`f64::max` or a partial comparator: a NaN
    // would otherwise sort arbitrarily and pick a neighbour's value silently.
    // `is_evidence` has already excluded them; this is the second lock.
    samples.sort_by(|left, right| left.0.total_cmp(&right.0));
    let total: f64 = samples.iter().map(|(_, weight)| weight).sum();
    let mut seen = 0.0_f64;
    for (quotient, weight) in &samples {
        seen += weight;
        if seen * 2.0 >= total {
            return Some(*quotient);
        }
    }
    samples.last().map(|(quotient, _)| *quotient)
}

// ----------------------------------------------------------- snapshot --

/// One coefficient, and how much evidence is behind it.
///
/// **`runs` and `units` are a weight, not a ledger.** They say roughly how much
/// observation is behind `nanos_per_unit`, which is what a consumer deciding
/// whether to trust it needs, and they must not be reconciled against anything.
/// A distributed run's events reach the recorder through the coordinator, and
/// `distributed::Coordinator` makes them fire-and-forget on purpose — the last
/// few of a run are still in flight when the last task finishes, and the
/// coordinator waits for quiet rather than for an acknowledgement it declines to
/// require. So a coefficient is derived from *most* of a run. For an average
/// over hundreds of thousands of events that is immaterial; for anybody trying
/// to prove `units` equals the voxels a plan predicted, it is not, and the
/// mismatch is this rather than a bug.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coefficient {
    /// Nanoseconds per one unit of whatever the [`Term`] is per.
    ///
    /// The **weighted median** over the retained runs, not the pooled quotient.
    /// See `weighted_median` for why, and [`REPRODUCTIONS`] for the measurement
    /// that forced it.
    pub nanos_per_unit: f64,
    /// Runs contributing, at most [`RETAINED_RUNS`]. Approximate; see above.
    ///
    /// **This is the field that decides whether the measurement is used at
    /// all** — see [`Snapshot::calibrate`] and [`REPRODUCTIONS`].
    pub runs: usize,
    /// Units of work behind it, summed over those runs. Approximate; see above.
    pub units: f64,
    /// Nanoseconds behind it, summed over those runs.
    ///
    /// **Recorded because `units` turned out to be the wrong measure of how much
    /// a coefficient can be trusted, and this is the right one.** Two terms of
    /// one run can differ by 300x in the wall time behind them while differing
    /// hardly at all in event count, and it is the wall time that says whether a
    /// millisecond-scale stall could have been the whole measurement.
    /// [`REPRODUCTIONS`] carries the table. Nothing in this crate *thresholds*
    /// on it — no absolute figure separates a thin term from a fat one on a
    /// machine nobody has measured — but a consumer deciding how much to believe
    /// a number should be reading this rather than `units`.
    pub total_nanos: f64,
}

/// Where a coefficient came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Provenance {
    /// No evidence on this machine. The shipped constant stands, and it is a
    /// **seed**: chosen for having the ordering right, not the scale.
    Seeded,
    /// Measured, with this much behind it — approximately. See [`Coefficient`]
    /// for why these two figures are a weight rather than a count.
    Measured { runs: usize, units: f64 },
    /// Seen, but not enough times to be believed: fewer than [`REPRODUCTIONS`]
    /// runs, so [`Snapshot::calibrate`] treats it exactly as it treats a term it
    /// has never seen.
    ///
    /// **A distinct state rather than folding into [`Provenance::Seeded`]**,
    /// because "no evidence" and "evidence that has not been reproduced" are
    /// different situations for whoever is deciding whether to run the pipeline
    /// again. One of them is fixed by running it twice more.
    Unreproduced { runs: usize },
}

/// A frozen view of the store, and the thing a plan is a function of.
///
/// Immutable by construction: it is taken once, before planning, and the
/// planner cannot see anything a later run records. That is what lets
/// "reproducible given `(metadata, statistics)`" mean something.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    machine: MachineKey,
    coefficients: BTreeMap<Term, Coefficient>,
    fingerprint: u64,
}

impl Default for Snapshot {
    /// The empty snapshot: no evidence, for the machine this process is on.
    /// Calibrating with it returns the caller's model unchanged.
    fn default() -> Self {
        Snapshot::new(MachineKey::detect(), BTreeMap::new())
    }
}

impl Snapshot {
    fn new(machine: MachineKey, coefficients: BTreeMap<Term, Coefficient>) -> Self {
        let fingerprint = fingerprint_of(&machine, &coefficients);
        Self {
            machine,
            coefficients,
            fingerprint,
        }
    }

    /// An empty snapshot for a stated machine. For tests, and for a caller that
    /// wants to plan explicitly without evidence.
    pub fn empty(machine: MachineKey) -> Self {
        Snapshot::new(machine, BTreeMap::new())
    }

    /// This snapshot with one coefficient stated outright.
    ///
    /// **For a caller whose evidence did not come from a run of this crate** —
    /// a test constructing the case it wants to reason about, or a deployment
    /// carrying figures measured elsewhere. Everything else here is built by
    /// `Store::record` from an `ExecutionLog`, which is the route that keeps a
    /// coefficient tied to the work that produced it; this one is deliberately
    /// the other kind, and `Coefficient::runs` is what a reader checks to tell
    /// them apart, exactly as [`Self::believable`] does.
    ///
    /// The fingerprint is recomputed, so a snapshot built this way is as
    /// comparable as any other.
    pub fn with(self, term: Term, coefficient: Coefficient) -> Self {
        let mut coefficients = self.coefficients;
        coefficients.insert(term, coefficient);
        Snapshot::new(self.machine, coefficients)
    }

    pub fn machine(&self) -> &MachineKey {
        &self.machine
    }

    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// The snapshot's identity.
    ///
    /// Over the format version, the machine key and every coefficient's key,
    /// value, run count and evidence. Two snapshots with this fingerprint
    /// calibrate identically, so a plan built against one is the plan built
    /// against the other.
    ///
    /// Deterministic across processes for the same reason
    /// `Decomposition::fingerprint` is: it hashes strings, integers and `f64`
    /// *bit patterns*, never a pointer or a hash-map order. It is **not**
    /// portable across Rust versions — `DefaultHasher`'s algorithm is not
    /// specified — which is the same guarantee the decomposition's fingerprint
    /// gives and is why both are identities to compare within a deployment
    /// rather than values to publish.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub fn coefficient(&self, term: &Term) -> Option<Coefficient> {
        self.coefficients.get(term).copied()
    }

    /// Whether a term is measured here, and how strongly.
    pub fn provenance(&self, term: &Term) -> Provenance {
        match self.coefficient(term) {
            Some(c) if c.runs >= REPRODUCTIONS => Provenance::Measured {
                runs: c.runs,
                units: c.units,
            },
            Some(c) => Provenance::Unreproduced { runs: c.runs },
            None => Provenance::Seeded,
        }
    }

    /// The coefficient for `term` **if it may be believed** — measured, and
    /// reproduced at least [`REPRODUCTIONS`] times.
    ///
    /// [`Snapshot::coefficient`] is the raw reading and stays raw, because a
    /// report should show what was seen. This is what calibration uses.
    pub fn believable(&self, term: &Term) -> Option<Coefficient> {
        self.coefficient(term).filter(|c| c.runs >= REPRODUCTIONS)
    }

    /// The per-op-family compute coefficients, by family name.
    ///
    /// **Recorded, and — since `CostModel::compute_of` existed to hold them —
    /// used.** [`Self::calibrate`] turns each believable one into a correction,
    /// as a ratio to the run-wide `Term::Compute`; this is the raw reading, for
    /// a report. See [`Term::ComputeOf`] and [`Snapshot::family_spread`].
    pub fn families(&self) -> BTreeMap<String, Coefficient> {
        self.coefficients
            .iter()
            .filter_map(|(term, value)| match term {
                Term::ComputeOf(name) => Some((name.clone(), *value)),
                _ => None,
            })
            .collect()
    }

    /// The ratio between the dearest and cheapest op family's coefficient, or
    /// `None` with fewer than two families.
    ///
    /// **This is the number that said what `CostModel` could not express**, and
    /// the measurement that motivated giving it somewhere to put them:
    /// `CostModel::compute_of` now holds one correction per family and
    /// [`Self::calibrate`] fills it, so a spread above one is a plan the model
    /// can now price differently rather than a defect it cannot see. The
    /// paragraph below is why the number is worth reporting either way. Every
    /// family's coefficient is nanoseconds per unit of *declared* cost, so a
    /// spread of 1.0 would mean the shipped constants have every family's
    /// relative cost exactly right and one `compute_scale` serves all of them.
    /// A spread of 3 means the planner is charging one family three times what
    /// it costs relative to another — and until `CostModel::compute_of` existed
    /// there was nowhere in the model to say so, `compute_scale` being a scalar
    /// whose best value is the work-weighted blend that `Term::Compute` already
    /// is. The spread is now what [`Self::calibrate`] turns into corrections,
    /// so it is the size of the *correction* rather than the size of a defect.
    pub fn family_spread(&self) -> Option<f64> {
        let families = self.families();
        if families.len() < 2 {
            return None;
        }
        // `total_cmp` rather than `f64::min`/`f64::max`, which absorb a NaN
        // silently and would report a spread over a coefficient that is not a
        // number. A NaN here would be a bug upstream; the point is that it
        // surfaces as one rather than as a plausible ratio.
        let mut values: Vec<f64> = families.values().map(|c| c.nanos_per_unit).collect();
        values.sort_by(f64::total_cmp);
        let low = values[0];
        let high = values[values.len() - 1];
        (low > 0.0 && high.is_finite()).then_some(high / low)
    }

    /// `model`, with every coefficient this snapshot has evidence for replaced
    /// by the measurement, in nanoseconds.
    ///
    /// **An empty snapshot returns `model` unchanged**, bit for bit. That is the
    /// non-negotiable property: a run with no store plans exactly as it does
    /// today, and every existing test stays green with nothing present.
    ///
    /// **Calibration is all-or-nothing about the unit.** The model's weights are
    /// only meaningful relative to one another — a read weight of 1.0 beside a
    /// compute scale of 1.0 says "a unit of compute costs one read" — so
    /// replacing some of them with nanoseconds and leaving the rest in
    /// map-units would produce a model that is not wrong in any one entry and is
    /// nonsense as a whole. So a term without evidence does not keep its seeded
    /// *number*, it keeps its seeded **ratio**, converted into nanoseconds by an
    /// anchor: the work-weighted mean, over the terms that do have evidence, of
    /// measured-nanoseconds divided by the seed that predicted them. Where every
    /// term is measured the anchor is unused; where none is, there is no anchor
    /// and the model comes back untouched.
    ///
    /// **A term seen fewer than [`REPRODUCTIONS`] times is not used**, and takes
    /// exactly the path a term with no evidence at all takes: its seeded *ratio*,
    /// converted by the anchor. That is the mechanism protecting the model from
    /// its own thinnest coefficient, and [`REPRODUCTIONS`] carries the
    /// measurement that motivates it and the price of being wrong in either
    /// direction. Note what it means for a **cold store**: one run of a pipeline
    /// changes no plan at all, and it takes three before anything is believed —
    /// which is the store's own premise, since a measurement that has never been
    /// reproduced is a fact about one run rather than about the machine.
    ///
    /// `order_conflict_penalty` is scaled by the anchor rather than measured.
    /// It is charged per voxel like the rest, nothing in the event stream can
    /// see a traversal disagreement, and its default is **zero** — so an
    /// unmeasured intuition stays worth exactly nothing, which is the property
    /// `CostModel` documents and this must not quietly undo.
    pub fn calibrate(&self, model: &CostModel) -> CostModel {
        let seeded = [
            (Term::Read, model.read_cost_per_voxel),
            (Term::Write, model.write_cost_per_voxel),
            (Term::Materialise, model.materialise_cost_per_voxel),
            (Term::Compute, model.compute_scale),
        ];
        let mut weighted = 0.0_f64;
        let mut weight = 0.0_f64;
        for (term, seed) in &seeded {
            if let Some(measured) = self.believable(term) {
                if *seed > 0.0 && measured.units > 0.0 {
                    weighted += (measured.nanos_per_unit / seed) * measured.units;
                    weight += measured.units;
                }
            }
        }
        if weight <= 0.0 || !weighted.is_finite() {
            // No evidence at all, or none usable. Today's behaviour, exactly.
            return model.clone();
        }
        let anchor = weighted / weight;
        let resolve = |term: Term, seed: f64| {
            self.believable(&term)
                .map(|c| c.nanos_per_unit)
                .unwrap_or(seed * anchor)
        };
        let compute_scale = resolve(Term::Compute, model.compute_scale);
        CostModel {
            read_cost_per_voxel: resolve(Term::Read, model.read_cost_per_voxel),
            write_cost_per_voxel: resolve(Term::Write, model.write_cost_per_voxel),
            materialise_cost_per_voxel: resolve(
                Term::Materialise,
                model.materialise_cost_per_voxel,
            ),
            compute_scale,
            // **The per-family corrections, which is the half that was measured
            // and never used.** `Term::ComputeOf` is keyed by slot name and has
            // been recorded since it existed under a doc saying it is not
            // consumed, because `CostModel` had one `compute_scale` and nowhere
            // to put a per-family figure. It has one now.
            //
            // A *ratio*: the family's own nanoseconds per unit over the run-wide
            // figure, so the table is dimensionless and its neutral value is
            // `1.0`. A family exactly at the run-wide rate contributes nothing,
            // which is what makes filling this in unable to move a plan it has
            // no evidence about.
            //
            // `believable` only, on the same argument as every other term: a
            // coefficient seen fewer than `REPRODUCTIONS` times is a fact about
            // one run rather than about the machine. And a run-wide figure of
            // zero would make every ratio infinite, so the table stays empty
            // there rather than carrying one.
            compute_of: if compute_scale > 0.0 && compute_scale.is_finite() {
                self.coefficients
                    .keys()
                    .filter_map(|term| match term {
                        Term::ComputeOf(family) => Some(family.clone()),
                        _ => None,
                    })
                    .filter_map(|family| {
                        let measured = self.believable(&Term::ComputeOf(family.clone()))?;
                        let correction = measured.nanos_per_unit / compute_scale;
                        correction.is_finite().then_some((family, correction))
                    })
                    .collect()
            } else {
                model.compute_of.clone()
            },
            order_conflict_penalty: model.order_conflict_penalty * anchor,
        }
    }

    /// The snapshot as a table, for a report or a log line.
    pub fn describe(&self) -> String {
        let mut out = format!(
            "statistics for {} ({} coefficient(s), fingerprint {:016x})\n{:<44} {:>14} {:>6} {:>16}  {:>13}\n",
            self.machine.id(),
            self.coefficients.len(),
            self.fingerprint,
            "term",
            "ns per unit",
            "runs",
            "units",
            "wall time",
        );
        for (term, coefficient) in &self.coefficients {
            out.push_str(&format!(
                "{:<44} {:>14.4} {:>6} {:>16.0}  {:>10.3} ms{}\n",
                term.key(),
                coefficient.nanos_per_unit,
                coefficient.runs,
                coefficient.units,
                // The wall time behind the term, because it is the figure that
                // says how much to believe it and `units` is not — see
                // [`REPRODUCTIONS`]. A table that showed only `units` is what
                // made that mistake easy to keep making.
                coefficient.total_nanos / 1.0e6,
                if coefficient.runs >= REPRODUCTIONS {
                    ""
                } else {
                    "  (not yet reproduced; calibration ignores it)"
                },
            ));
        }
        out
    }
}

fn fingerprint_of(machine: &MachineKey, coefficients: &BTreeMap<Term, Coefficient>) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    FORMAT.hash(&mut hasher);
    FORMAT_VERSION.hash(&mut hasher);
    machine.id().hash(&mut hasher);
    coefficients.len().hash(&mut hasher);
    // A `BTreeMap` iterates in key order, so this is a function of the contents
    // and not of an insertion history. `f64` is hashed by its bits because
    // `f64` is not `Hash` and because the exact value is what calibration uses.
    for (term, coefficient) in coefficients {
        // Hashed by the term's stable key rather than by the enum's shape, so
        // adding a variant does not renumber the identity of a snapshot that
        // does not use it.
        term.key().hash(&mut hasher);
        coefficient.nanos_per_unit.to_bits().hash(&mut hasher);
        coefficient.runs.hash(&mut hasher);
        coefficient.units.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

// ------------------------------------------------------ plan identity --

/// What a plan is a function of: the metadata, and the evidence.
///
/// **The problem this solves.** A plan used to be a function of metadata alone,
/// so the same workflow planned identically twice and `Decomposition::fingerprint`
/// was a complete identity for it. With statistics feeding the cost model that
/// is no longer true — the same workflow may plan differently after a run, which
/// is the adaptive behaviour that is wanted. What must not happen is that it
/// becomes *unanswerable* why two runs planned differently.
///
/// **What is guaranteed.** Given the same `(workflow metadata, constraints,
/// strategy, snapshot)`, planning is deterministic and yields the same
/// `Decomposition` — the search reads `f64` weights but is otherwise integer
/// arithmetic over a fixed enumeration order, and the weights are frozen in the
/// snapshot. So two plans with equal `PlanIdentity` were built from the same
/// evidence, and two plans with equal decomposition fingerprints and *different*
/// statistics fingerprints agree in spite of the evidence having moved.
///
/// **What is not guaranteed.** This is not a content-addressed archive: the
/// identity says *which* snapshot, not what was in it, so answering "why did
/// this plan differ" needs the store to still hold that snapshot's records —
/// and the store is a bounded ring, so after [`RETAINED_RUNS`] further runs it
/// does not. Recovering an old plan therefore means recording the snapshot, not
/// just its fingerprint; `Snapshot::describe` is the intended thing to put in a
/// run record beside this. Nor does it cover anything outside the plan: two
/// identical plans still execute differently under different concurrency,
/// because that is a `Hints` decision and hints are advisory by construction.
///
/// The fingerprints are `DefaultHasher` values, so they are stable within a
/// deployment and are not a portable format across Rust versions — the same
/// caveat `Decomposition::fingerprint` carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanIdentity {
    pub decomposition: u64,
    pub statistics: u64,
}

impl PlanIdentity {
    pub fn of(decomposition: &Decomposition, snapshot: &Snapshot) -> Self {
        Self {
            decomposition: decomposition.fingerprint(),
            statistics: snapshot.fingerprint(),
        }
    }

    /// The pair as one number, for a manifest field that has room for one.
    pub fn digest(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.decomposition.hash(&mut hasher);
        self.statistics.hash(&mut hasher);
        hasher.finish()
    }
}

impl std::fmt::Display for PlanIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}+{:016x}", self.decomposition, self.statistics)
    }
}

// ------------------------------------------------------------ observed --

/// What a run actually spent, in nanoseconds, on the three things the cost
/// model prices.
///
/// The counterpart of `decomposition::predicted_cost`, and stated in the same
/// currency so the two can be subtracted. Reads, writes and op applications
/// only: sidecar and side-output traffic is timed on the event stream but has no
/// term in `CostModel`, so counting it here would make the model look worse than
/// it is at the thing it does model.
pub fn observed_nanos(log: &ExecutionLog) -> f64 {
    let mut total = 0.0_f64;
    for event in log.events() {
        total += match event {
            Event::RegionRead { duration_ns, .. }
            | Event::RegionWritten { duration_ns, .. }
            | Event::OpApplied { duration_ns, .. } => duration_ns as f64,
            _ => 0.0,
        };
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine() -> MachineKey {
        MachineKey {
            host: "a".to_string(),
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            pointer_width: 64,
            cpus: 8,
            profile: "release".to_string(),
        }
    }

    /// A coefficient stated outright, at a run count that decides whether it is
    /// believed.
    fn coefficient(nanos_per_unit: f64, runs: usize) -> Coefficient {
        Coefficient {
            nanos_per_unit,
            runs,
            units: 1000.0,
            total_nanos: nanos_per_unit * 1000.0,
        }
    }

    /// **The per-family corrections are filled as ratios**, which is what makes
    /// them dimensionless and their neutral value one.
    ///
    /// Three families against a run-wide figure: one dearer, one cheaper, one
    /// exactly at it. The third is the one worth asserting — a family that
    /// matches the blend must come out at `1.0` and change no plan, which is the
    /// property that makes filling this table safe on a model nobody has
    /// measured.
    #[test]
    fn calibration_fills_the_per_family_corrections_as_ratios() {
        let snapshot = Snapshot::empty(machine())
            .with(Term::Compute, coefficient(10.0, REPRODUCTIONS))
            .with(
                Term::ComputeOf("skeletonize".to_string()),
                coefficient(200.0, REPRODUCTIONS),
            )
            .with(
                Term::ComputeOf("combine".to_string()),
                coefficient(2.5, REPRODUCTIONS),
            )
            .with(
                Term::ComputeOf("average".to_string()),
                coefficient(10.0, REPRODUCTIONS),
            );
        let model = snapshot.calibrate(&CostModel::default());

        assert_eq!(model.compute_scale, 10.0);
        assert_eq!(model.correction_for("skeletonize"), 20.0);
        assert_eq!(model.correction_for("combine"), 0.25);
        assert_eq!(
            model.correction_for("average"),
            1.0,
            "a family at the run-wide rate is neutral"
        );
        assert_eq!(
            model.correction_for("never measured"),
            1.0,
            "and so is one nobody has measured"
        );
    }

    /// **An unreproduced family is not believed**, on exactly the argument
    /// every other term is held to: a coefficient seen once is a fact about one
    /// run rather than about the machine. `REPRODUCTIONS` is the line and this
    /// is where the new table is put on the right side of it.
    #[test]
    fn a_family_measured_once_moves_no_plan() {
        let snapshot = Snapshot::empty(machine())
            .with(Term::Compute, coefficient(10.0, REPRODUCTIONS))
            .with(
                Term::ComputeOf("smooth".to_string()),
                coefficient(500.0, REPRODUCTIONS - 1),
            );
        let model = snapshot.calibrate(&CostModel::default());
        assert_eq!(
            model.correction_for("smooth"),
            1.0,
            "a family reproduced fewer than {REPRODUCTIONS} times is evidence about one run"
        );
        assert!(
            model.compute_of.is_empty(),
            "and it is absent rather than present at one"
        );
    }

    fn elsewhere() -> MachineKey {
        MachineKey {
            host: "b".to_string(),
            ..machine()
        }
    }

    fn run(machine: MachineKey, terms: &[(Term, f64, f64)]) -> RunObservations {
        RunObservations {
            machine,
            terms: terms
                .iter()
                .map(|(term, units, nanos)| {
                    (
                        term.clone(),
                        Observed {
                            units: *units,
                            nanos: *nanos,
                        },
                    )
                })
                .collect(),
        }
    }

    /// The property the whole module is arranged around: nothing present means
    /// nothing changes.
    #[test]
    fn an_empty_store_returns_the_model_unchanged() {
        let model = CostModel::default();
        let store = Statistics::new();
        let snapshot = store.snapshot(&machine());
        assert!(snapshot.is_empty());
        assert_eq!(snapshot.calibrate(&model), model);

        // and a model that is not the default is equally untouched
        let custom = CostModel {
            read_cost_per_voxel: 3.5,
            write_cost_per_voxel: 0.25,
            compute_scale: 7.0,
            order_conflict_penalty: 0.5,
            materialise_cost_per_voxel: 2.0,
            // A table a caller wrote by hand, which an empty snapshot must also
            // leave alone — the corrections are evidence like every other term,
            // and evidence nobody has is not a reason to discard evidence
            // somebody stated.
            compute_of: [("smooth".to_string(), 4.0)].into_iter().collect(),
        };
        assert_eq!(snapshot.calibrate(&custom), custom);
    }

    #[test]
    fn every_term_round_trips_through_its_key() {
        for term in [
            Term::Read,
            Term::Write,
            Term::Materialise,
            Term::Compute,
            Term::ReadBytes,
            Term::WriteBytes,
            Term::ComputeOf("median".to_string()),
        ] {
            assert_eq!(Term::parse(&term.key()), Some(term.clone()), "{term:?}");
        }
    }

    #[test]
    fn a_measured_term_replaces_its_seed_and_an_unmeasured_one_keeps_its_ratio() {
        let mut store = Statistics::new();
        for _ in 0..REPRODUCTIONS {
            store.record(&run(
                machine(),
                &[
                    (Term::Read, 1000.0, 4000.0),
                    (Term::Compute, 1000.0, 8000.0),
                ],
            ));
        }
        let snapshot = store.snapshot(&machine());
        let model = snapshot.calibrate(&CostModel::default());
        // measured outright
        assert_eq!(model.read_cost_per_voxel, 4.0);
        assert_eq!(model.compute_scale, 8.0);
        // unmeasured: seeded ratio (1.0) times the anchor, which here is the
        // work-weighted mean of 4/1 and 8/1
        assert_eq!(model.write_cost_per_voxel, 6.0);
        assert_eq!(model.materialise_cost_per_voxel, 6.0);
        // and the penalty nobody has measured is still worth nothing
        assert_eq!(model.order_conflict_penalty, 0.0);
    }

    #[test]
    fn a_store_from_another_machine_is_not_evidence_here() {
        let mut store = Statistics::new();
        store.record(&run(elsewhere(), &[(Term::Read, 1000.0, 9999.0)]));
        assert!(!store.is_empty());
        let snapshot = store.snapshot(&machine());
        assert!(snapshot.is_empty(), "{}", snapshot.describe());
        assert_eq!(
            snapshot.calibrate(&CostModel::default()),
            CostModel::default()
        );
        // it is still there, for the machine it is about
        assert!(!store.snapshot(&elsewhere()).is_empty());
    }

    /// Every field of the key partitions, not just the host.
    #[test]
    fn each_field_of_the_key_partitions_the_store() {
        let variants = [
            MachineKey {
                host: "other".to_string(),
                ..machine()
            },
            MachineKey {
                os: "macos".to_string(),
                ..machine()
            },
            MachineKey {
                arch: "aarch64".to_string(),
                ..machine()
            },
            MachineKey {
                pointer_width: 32,
                ..machine()
            },
            MachineKey {
                cpus: 4,
                ..machine()
            },
            MachineKey {
                profile: "debug".to_string(),
                ..machine()
            },
        ];
        for variant in variants {
            let mut store = Statistics::new();
            store.record(&run(variant.clone(), &[(Term::Read, 10.0, 40.0)]));
            assert!(
                store.snapshot(&machine()).is_empty(),
                "{} leaked into {}",
                variant.id(),
                machine().id()
            );
        }
    }

    #[test]
    fn a_store_round_trips_through_json_with_identical_coefficients() {
        let mut store = Statistics::new();
        store.record(&run(
            machine(),
            &[
                (Term::Read, 1234.0, 5678.25),
                (Term::Compute, 1e9, 3.7e10),
                (Term::ComputeOf("median".to_string()), 5.5, 91.125),
            ],
        ));
        store.record(&run(elsewhere(), &[(Term::Write, 7.0, 13.0)]));
        let text = store.to_json();
        let back = Statistics::from_json(&text).expect("a store");
        assert_eq!(back, store);
        for who in [machine(), elsewhere()] {
            let before = store.snapshot(&who);
            let after = back.snapshot(&who);
            assert_eq!(before.fingerprint(), after.fingerprint());
            for term in [
                Term::Read,
                Term::Write,
                Term::Compute,
                Term::ComputeOf("median".to_string()),
            ] {
                assert_eq!(before.coefficient(&term), after.coefficient(&term));
            }
        }
    }

    #[test]
    fn a_file_this_build_cannot_interpret_is_refused_rather_than_guessed() {
        assert!(Statistics::from_json("{\"format\":\"something else\"}").is_err());
        assert!(Statistics::from_json(
            "{\"format\":\"blockflow.statistics\",\"version\":99,\"machines\":[]}"
        )
        .is_err());
        assert!(Statistics::from_json("not json at all").is_err());
    }

    #[test]
    fn an_absent_file_is_an_empty_store() {
        let path = std::env::temp_dir().join("blockflow-statistics-absent-XYZZY.json");
        let _ = std::fs::remove_file(&path);
        let store = Statistics::load(&path).expect("absent is empty");
        assert!(store.is_empty());
    }

    #[test]
    fn saving_merges_rather_than_replaces() {
        let path = std::env::temp_dir().join(format!(
            "blockflow-statistics-merge-{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let mut first = Statistics::new();
        first.record(&run(machine(), &[(Term::Read, 100.0, 200.0)]));
        first.save(&path).expect("a save");

        let mut second = Statistics::new();
        second.record(&run(elsewhere(), &[(Term::Read, 100.0, 900.0)]));
        second.save(&path).expect("a second save");

        let merged = Statistics::load(&path).expect("a load");
        assert_eq!(merged.machines().len(), 2, "{}", merged.to_json());
        assert_eq!(
            merged.snapshot(&machine()).coefficient(&Term::Read),
            Some(Coefficient {
                nanos_per_unit: 2.0,
                runs: 1,
                units: 100.0,
                total_nanos: 200.0
            })
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The bound holds, and it is the *recent* runs that are kept.
    #[test]
    fn a_coefficient_keeps_the_last_few_runs_and_forgets_the_rest() {
        let mut store = Statistics::new();
        for _ in 0..RETAINED_RUNS {
            store.record(&run(machine(), &[(Term::Read, 100.0, 100.0)]));
        }
        assert_eq!(
            store.snapshot(&machine()).coefficient(&Term::Read).unwrap(),
            Coefficient {
                nanos_per_unit: 1.0,
                runs: RETAINED_RUNS,
                units: 100.0 * RETAINED_RUNS as f64,
                total_nanos: 100.0 * RETAINED_RUNS as f64
            }
        );
        // **One outlying run does not move it at all, and this assertion is
        // inverted from the one that stood here.** It used to read "moves it,
        // and is diluted by the evidence beside it" — `> 1.0 && < 9.0` — which
        // was the truth about the pooled quotient this used to compute. The
        // estimator is now the weighted median (see `weighted_median` and
        // `REPRODUCTIONS`), and a median does not dilute an outlier, it outvotes
        // it. That is the whole point of the change: a run that stalled adds
        // nanoseconds and no units, so under pooling it moved the coefficient in
        // proportion to how badly it stalled, without limit.
        store.record(&run(machine(), &[(Term::Read, 100.0, 900.0)]));
        let after = store.snapshot(&machine()).coefficient(&Term::Read).unwrap();
        assert_eq!(after.runs, RETAINED_RUNS);
        assert_eq!(
            after.nanos_per_unit, 1.0,
            "a 9x outlier moved the coefficient; the median is supposed to outvote it"
        );
        // It is *recorded* rather than ignored, though — the totals still carry
        // it, which is what lets a reader see that a run went strange.
        assert!(after.total_nanos > 100.0 * RETAINED_RUNS as f64);
        // and once it is the only thing left, it is the answer
        for _ in 0..RETAINED_RUNS {
            store.record(&run(machine(), &[(Term::Read, 100.0, 900.0)]));
        }
        assert_eq!(
            store
                .snapshot(&machine())
                .coefficient(&Term::Read)
                .unwrap()
                .nanos_per_unit,
            9.0
        );
    }

    #[test]
    fn a_run_that_reports_no_time_is_not_a_measurement() {
        let mut store = Statistics::new();
        // What an AccountingEnvironment produces: work counted, no clock.
        store.record(&run(machine(), &[(Term::Read, 1e9, 0.0)]));
        assert!(store.snapshot(&machine()).is_empty());
        assert_eq!(
            store.snapshot(&machine()).calibrate(&CostModel::default()),
            CostModel::default()
        );
    }

    /// The three states a term can be in, and the one in the middle is the point.
    #[test]
    fn provenance_says_whether_a_coefficient_is_seeded_measured_or_merely_seen() {
        let mut store = Statistics::new();
        for seen in 1..=REPRODUCTIONS {
            store.record(&run(machine(), &[(Term::Read, 64.0, 128.0)]));
            let snapshot = store.snapshot(&machine());
            // Never seen at all is a different state from seen and not yet
            // believed, and both differ from believed.
            assert_eq!(snapshot.provenance(&Term::Write), Provenance::Seeded);
            if seen < REPRODUCTIONS {
                assert_eq!(
                    snapshot.provenance(&Term::Read),
                    Provenance::Unreproduced { runs: seen }
                );
                assert!(
                    snapshot.believable(&Term::Read).is_none(),
                    "a term seen {seen} time(s) was believed"
                );
            } else {
                assert_eq!(
                    snapshot.provenance(&Term::Read),
                    Provenance::Measured {
                        runs: seen,
                        units: 64.0 * seen as f64
                    }
                );
                assert!(snapshot.believable(&Term::Read).is_some());
            }
            // The raw reading is available at every stage regardless — a report
            // shows what was seen, and only calibration cares whether it is
            // enough.
            assert!(snapshot.coefficient(&Term::Read).is_some());
        }
    }

    /// **The rule that protects the model from its thinnest coefficient**, at
    /// the boundary and on both sides of it.
    ///
    /// A term seen fewer than [`REPRODUCTIONS`] times takes the path a term with
    /// no evidence takes: its seeded ratio, converted by the anchor. So a store
    /// that has seen one run of everything changes nothing at all — which is the
    /// cold-start behaviour, and it is a *stronger* statement than the one an
    /// empty store makes.
    #[test]
    fn a_term_is_not_believed_until_it_has_been_reproduced() {
        let seeded = CostModel::default();
        let mut store = Statistics::new();
        let observations = [
            (Term::Read, 1000.0, 4000.0),
            (Term::Write, 1000.0, 4000.0),
            (Term::Materialise, 1000.0, 4000.0),
            (Term::Compute, 1000.0, 4000.0),
        ];
        for seen in 1..REPRODUCTIONS {
            store.record(&run(machine(), &observations));
            let snapshot = store.snapshot(&machine());
            assert!(!snapshot.is_empty(), "the reading itself was discarded");
            assert_eq!(
                snapshot.calibrate(&seeded),
                seeded,
                "a store holding {seen} run(s) of every term calibrated anyway"
            );
        }
        // And the run that makes it a reproduction changes everything at once,
        // because the anchor is all-or-nothing about the unit.
        store.record(&run(machine(), &observations));
        let model = store.snapshot(&machine()).calibrate(&seeded);
        assert_ne!(model, seeded);
        assert_eq!(model.read_cost_per_voxel, 4.0);
        assert_eq!(model.compute_scale, 4.0);
    }

    /// The liveness test beside it: with the rule satisfied, a **stalled** run
    /// still cannot move the answer, because the estimator is a median.
    ///
    /// The two mechanisms are separate and this is what distinguishes them.
    /// [`REPRODUCTIONS`] governs *whether* a term is used; `weighted_median`
    /// governs *what value* it has. A test that only exercised the first would
    /// pass against a store that pooled its runs and was moved by every outlier
    /// the moment the third one arrived.
    #[test]
    fn a_stalled_run_does_not_move_a_reproduced_coefficient() {
        let mut store = Statistics::new();
        for _ in 0..REPRODUCTIONS {
            store.record(&run(machine(), &[(Term::Read, 1000.0, 4000.0)]));
        }
        let before = store.snapshot(&machine()).believable(&Term::Read).unwrap();
        assert_eq!(before.nanos_per_unit, 4.0);

        // A run that did the same work and took 30x as long: a stall adds
        // nanoseconds and no units, which is exactly what made pooling fragile.
        store.record(&run(machine(), &[(Term::Read, 1000.0, 120_000.0)]));
        let after = store.snapshot(&machine()).believable(&Term::Read).unwrap();
        assert_eq!(
            after.nanos_per_unit, 4.0,
            "one stalled run moved a coefficient with {} runs behind it",
            after.runs
        );
        // Pooled, it would have been (3 x 4000 + 120000) / 4000 units = 33.
        assert!(
            after.total_nanos / after.units > 30.0,
            "the totals stopped carrying what actually happened"
        );
        // A majority of stalls is not a stall, it is the machine, and then it
        // *is* the answer — the rule must not be a permanent veto.
        for _ in 0..REPRODUCTIONS {
            store.record(&run(machine(), &[(Term::Read, 1000.0, 120_000.0)]));
        }
        assert_eq!(
            store
                .snapshot(&machine())
                .believable(&Term::Read)
                .unwrap()
                .nanos_per_unit,
            120.0
        );
    }

    #[test]
    fn the_fingerprint_is_a_function_of_the_coefficients() {
        let mut store = Statistics::new();
        store.record(&run(machine(), &[(Term::Read, 100.0, 200.0)]));
        let first = store.snapshot(&machine());
        // taking it twice gives the same identity
        assert_eq!(
            first.fingerprint(),
            store.snapshot(&machine()).fingerprint()
        );
        // recording more evidence moves it
        store.record(&run(machine(), &[(Term::Read, 100.0, 400.0)]));
        let second = store.snapshot(&machine());
        assert_ne!(first.fingerprint(), second.fingerprint());
        // and an empty snapshot is not accidentally equal to a full one
        assert_ne!(
            Snapshot::empty(machine()).fingerprint(),
            first.fingerprint()
        );
    }

    #[test]
    fn the_family_spread_is_the_thing_one_compute_scale_cannot_hold() {
        let mut store = Statistics::new();
        store.record(&run(
            machine(),
            &[
                (Term::ComputeOf("map".to_string()), 100.0, 100.0),
                (Term::ComputeOf("median".to_string()), 100.0, 300.0),
            ],
        ));
        let snapshot = store.snapshot(&machine());
        assert_eq!(snapshot.families().len(), 2);
        assert_eq!(snapshot.family_spread(), Some(3.0));
        // one family alone says nothing about a spread
        let mut single = Statistics::new();
        single.record(&run(
            machine(),
            &[(Term::ComputeOf("map".to_string()), 100.0, 100.0)],
        ));
        assert_eq!(single.snapshot(&machine()).family_spread(), None);
    }
}
