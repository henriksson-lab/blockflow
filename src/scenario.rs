// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! **Cost scenarios: the same plan question asked of machines this one is not.**
//!
//! # What this is for
//!
//! `statistics::Statistics` records what *this* machine did, partitioned by
//! [`MachineKey`](crate::statistics::MachineKey) so that "a coefficient measured
//! elsewhere is not evidence here". That is the right rule for planning a run.
//! It is the wrong rule for **testing a planner**, where the question is the
//! opposite one: does the search still choose well when the disk is ten times
//! slower, when the page cache is a sixteenth of the size, when there are two
//! cores instead of forty?
//!
//! Nothing could ask that before this file. There was one store, keyed by the
//! host it ran on, and a planner change was accepted on the evidence of the
//! machine that happened to be under it — which is the definition of overfitting
//! and is not visible from inside.
//!
//! A [`Scenario`](crate::scenario::Scenario) is therefore a **machine that need not exist**: a snapshot of
//! coefficients, the machine the simulator models, and what the planner is
//! allowed to spend. `costs/` holds one per plausible shape of machine, they are
//! files in the repository rather than figures in a test, and
//! `tests/cost_scenarios.rs` runs the planner against every one of them.
//!
//! # One source of coefficients, two judges
//!
//! A scenario owns a [`Snapshot`](crate::statistics::Snapshot) and **both judges are derived from it**:
//!
//! * the planner, through [`Scenario::model`](crate::scenario::Scenario::model) — `Snapshot::calibrate`, the same
//!   path a real run's evidence takes;
//! * the simulator, through [`Scenario::rates`](crate::scenario::Scenario::rates) — `Rates::from_snapshot`, the
//!   same path again.
//!
//! So a scenario cannot state one cost to the planner and another to the
//! simulator. That is not tidiness: a robustness sweep in which the two halves
//! could disagree would measure the disagreement rather than the planner.
//!
//! What a snapshot has no term for is held beside it and named: the chunk
//! geometry, the two `Rates` fields no coefficient covers, and the whole of
//! [`Machine`](crate::simulate::Machine) — worker count, cache bytes, IO channels, contention — which are
//! properties of a machine that no *cost* coefficient can express.
//!
//! # Deriving one scenario from another
//!
//! The transforms — [`Scenario::with_scaled`](crate::scenario::Scenario::with_scaled), [`Scenario::with_machine`](crate::scenario::Scenario::with_machine) and
//! friends — take a measured baseline and produce a plausible neighbour of it.
//! **Ratios, not new absolutes**: a scenario that multiplies the read
//! coefficient by ten is "a disk ten times slower than the one we measured",
//! which is a claim the measurement supports. Typing an absolute figure for a
//! disk nobody has would be a claim about a machine nobody has measured, and
//! this module has no way to make one.
//!
//! The baseline is kept. `costs/measured.json` is the evidence this crate
//! actually recorded, with each term's provenance in its own note, and every
//! other file in that directory says which transform of it it is.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::decomposition::{Constraints, CostModel};
use crate::error::{Error, Result};
use crate::simulate::{Machine, Rates};
use crate::statistics::{Coefficient, MachineKey, Snapshot, Term, REPRODUCTIONS};

/// The file format tag, and the version of it.
///
/// Refused on anything else, for the reason `statistics`'s own pair is: a
/// coefficient read under the wrong interpretation is silently wrong rather
/// than absently right.
const FORMAT: &str = "blockflow-scenario";
const FORMAT_VERSION: u64 = 1;

/// What a [`Snapshot`] has no coefficient for, and a simulation needs anyway.
///
/// Three of the four are `Rates` fields whose own docs say they are carried
/// from a seed rather than measured; the fourth is the chunk geometry, which is
/// a property of the *storage layout* rather than of the machine. They are here
/// so that a scenario is a complete description of a run's costs, with nothing
/// left to a caller's default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Storage {
    /// The chunk the simulator counts fetches in.
    pub chunk: [usize; 3],
    /// Bytes per voxel of the stored data — what turns the per-voxel
    /// coefficients into the per-byte rates a simulation charges. See
    /// [`Rates::from_snapshot`], whose own doc records that a run whose images
    /// have different widths has no single answer.
    pub bytes_per_voxel: f64,
    /// Fixed cost of a fetch, whatever its size. **The term that makes a small
    /// chunk expensive**, and `0.0` in every figure this crate has recorded —
    /// which is a statement about what was measured, not about storage.
    pub io_latency_ns: f64,
    /// Decode cost per fetched byte, for a compressed store.
    pub decode_ns_per_byte: f64,
}

impl Storage {
    /// Bytes in one chunk, from the geometry and the element width.
    pub fn chunk_bytes(&self) -> u64 {
        (self.chunk.iter().product::<usize>() as f64 * self.bytes_per_voxel) as u64
    }
}

/// A machine to plan for and to be judged on, real or not.
#[derive(Debug, Clone, PartialEq)]
pub struct Scenario {
    /// How this scenario is referred to — in a file name, in a table, and in a
    /// test that says which one regressed.
    pub name: String,
    /// What it is and where its numbers came from, in prose. **Written to the
    /// file**, because a coefficient without a provenance is a number somebody
    /// typed.
    pub note: String,
    /// The coefficients. Both judges are derived from these; see the module
    /// header.
    pub snapshot: Snapshot,
    /// The machine the simulator models.
    pub machine: Machine,
    /// What a snapshot cannot say. See [`Storage`].
    pub storage: Storage,
    /// Bytes the planner may spend on in-flight blocks. `None` is unbounded,
    /// which is `Constraints::default()`'s own answer and is the scenario for a
    /// machine whose memory is not the binding constraint.
    pub budget_bytes: Option<u64>,
}

impl Scenario {
    /// The planner's cost model under this scenario: `seed`, calibrated by the
    /// scenario's own evidence.
    ///
    /// `seed` carries what no coefficient covers — `order_conflict_penalty`,
    /// and any per-family correction a caller stated by hand — and every
    /// measured term replaces its seeded value. This is exactly the path a real
    /// run's evidence takes into the planner, which is the point: a scenario
    /// must not be able to reach the planner by a route a measurement cannot.
    pub fn model(&self, seed: &CostModel) -> CostModel {
        CostModel {
            // The two the planner needs and no measurement can supply: they are
            // properties of the machine the scenario *describes*, so they come
            // from its `Machine` rather than from its snapshot, and calibration
            // carries them through untouched. A scenario that let the planner
            // price for one machine and the simulator run on another would be
            // measuring the mismatch.
            contention: self.machine.contention,
            nodes: self.machine.nodes.max(1),
            ..self.snapshot.calibrate(seed)
        }
    }

    /// The simulator's rates under this scenario.
    ///
    /// The coefficients come from the same snapshot the model does; the chunk
    /// geometry and the two unmeasured terms come from [`Self::storage`].
    pub fn rates(&self, seed: &Rates) -> Rates {
        let mut rates = Rates::from_snapshot(&self.snapshot, seed, self.storage.bytes_per_voxel);
        rates.chunk = self.storage.chunk;
        rates.chunk_bytes = self.storage.chunk_bytes();
        rates.io_latency_ns = self.storage.io_latency_ns;
        rates.decode_ns_per_byte = self.storage.decode_ns_per_byte;
        rates
    }

    /// `base` with this scenario's model, budget and worker count overlaid.
    ///
    /// **The search knobs are the caller's and the machine limits are the
    /// scenario's**, which is the whole division of labour here: a scenario says
    /// what the machine costs and how much of it there is, and the caller says
    /// what the planner is allowed to consider — the block ladder, the axes it
    /// may cut. A scenario that carried a block ladder would be answering the
    /// question the sweep is asking.
    pub fn constraints(&self, base: &Constraints) -> Constraints {
        Constraints {
            model: self.model(&base.model),
            // Per node, both of them. `budget_bytes` is what one computer may
            // spend and `expected_concurrency` is what one computer holds in
            // flight, because `Constraints::affords_working_set` multiplies the
            // two — and a ten-node run does not have to fit eighty blocks into
            // one machine's memory. At one node this is the whole worker count,
            // which is what it always was.
            budget_bytes: self.budget_bytes,
            expected_concurrency: self
                .machine
                .workers
                .max(1)
                .div_ceil(self.machine.nodes.max(1)),
            ..base.clone()
        }
    }

    // ------------------------------------------------------- transforms --

    /// This scenario under another name.
    ///
    /// **The name is not just a label**: the snapshot is filed under a machine
    /// key derived from it — see [`scenario_machine`] — so renaming a scenario
    /// has to re-key it, or a scenario built in memory and the same scenario
    /// read back from its file would be two different values. Every transform
    /// goes through here for that reason, and
    /// `a_scenario_round_trips_through_its_file` is what says they do.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        let mut snapshot = Snapshot::empty(scenario_machine(&name));
        for (term, coefficient) in self.snapshot.coefficients() {
            snapshot = snapshot.with(term.clone(), *coefficient);
        }
        self.snapshot = snapshot;
        self.name = name;
        self
    }

    /// This scenario with every listed term multiplied.
    ///
    /// **A ratio against a measured baseline**, which is the only kind of claim
    /// this module can honestly make about a machine nobody has run on. The
    /// terms not listed are untouched, so "a slower disk" is exactly that and
    /// not a wholesale re-costing.
    ///
    /// A term the baseline does not carry is **added** at `factor` times the
    /// crate's own seed for it, so that scaling a term into existence is
    /// possible and visible rather than silently a no-op.
    pub fn with_scaled(mut self, name: &str, factors: &[(Term, f64)]) -> Self {
        for (term, factor) in factors {
            let existing = self
                .snapshot
                .coefficient(term)
                .map(|c| c.nanos_per_unit)
                .unwrap_or(1.0);
            self.snapshot = self
                .snapshot
                .clone()
                .with(term.clone(), stated(existing * factor));
        }
        self.named(name)
    }

    /// This scenario with the machine replaced.
    pub fn with_machine(mut self, name: &str, machine: Machine) -> Self {
        self.machine = machine;
        self.named(name)
    }

    /// This scenario with a different byte budget for the planner and a page
    /// cache to match.
    ///
    /// The two move together on purpose: "less memory" is not a planner
    /// constraint or a cache size, it is both, and a scenario that shrank one
    /// without the other would be a machine that does not exist.
    pub fn with_memory(mut self, name: &str, bytes: u64) -> Self {
        self.budget_bytes = Some(bytes);
        self.machine.cache_bytes = bytes;
        self.named(name)
    }

    /// This scenario with its note replaced — the provenance of the transform
    /// that produced it.
    pub fn noted(mut self, note: impl Into<String>) -> Self {
        self.note = note.into();
        self
    }

    // ------------------------------------------------------ persistence --

    /// Read one scenario file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|err| {
            Error::InvalidArgument(format!("scenario: reading {}: {err}", path.display()))
        })?;
        Self::from_json(&text)
            .map_err(|err| Error::InvalidArgument(format!("scenario: {}: {err}", path.display())))
    }

    /// Write this scenario, replacing whatever is there.
    ///
    /// **No merge**, where `Statistics::save` merges: a store accumulates
    /// evidence and a scenario *is* a stated thing, so writing one is replacing
    /// a statement rather than adding to a history.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::write(path, self.to_json()).map_err(|err| {
            Error::InvalidArgument(format!("scenario: writing {}: {err}", path.display()))
        })
    }

    /// Every scenario in a directory, by name.
    ///
    /// **The "load multiple databases" entry point.** A `BTreeMap` so the order
    /// is the names' and not the filesystem's: a sweep that reported its rows in
    /// readdir order would be a different report on two machines.
    ///
    /// Files that are not `.json` are skipped; a `.json` that does not parse is
    /// an error naming it, because a scenario file that has rotted is exactly
    /// the thing a silent skip would hide.
    pub fn load_dir(path: impl AsRef<Path>) -> Result<BTreeMap<String, Scenario>> {
        let path = path.as_ref();
        let entries = std::fs::read_dir(path).map_err(|err| {
            Error::InvalidArgument(format!("scenario: reading {}: {err}", path.display()))
        })?;
        let mut found = BTreeMap::new();
        for entry in entries {
            let entry = entry.map_err(|err| {
                Error::InvalidArgument(format!("scenario: reading {}: {err}", path.display()))
            })?;
            let file = entry.path();
            if file.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let scenario = Scenario::load(&file)?;
            if let Some(clash) = found.insert(scenario.name.clone(), scenario) {
                return Err(Error::InvalidArgument(format!(
                    "scenario: two files in {} are both named {:?}. The name is what a report \
                     and a regression bound refer to, so it has to be the file's identity.",
                    path.display(),
                    clash.name
                )));
            }
        }
        Ok(found)
    }

    /// The scenario as JSON, in the hand-rolled style `statistics` uses and for
    /// the reason its own header gives: the format is expected to be replaced,
    /// so it stays out of the types.
    pub fn to_json(&self) -> String {
        // Through a `BTreeMap<String, _>` and not straight from the snapshot's
        // own map: that one is keyed by `Term`, so it iterates in *enum* order —
        // `Read` before `Compute` — and inserting in that order writes a
        // different file under `serde_json/preserve_order`. Keyed by the string
        // the file uses, insertion order is sorted order and the two emissions
        // are the same bytes. See the comment on the object below.
        let terms: Value = self
            .snapshot
            .coefficients()
            .iter()
            .map(|(term, coefficient)| {
                (
                    term.key(),
                    json!({
                        "nanos_per_unit": coefficient.nanos_per_unit,
                        "runs": coefficient.runs,
                        "units": coefficient.units,
                    }),
                )
            })
            .collect::<BTreeMap<String, Value>>()
            .into_iter()
            .collect::<serde_json::Map<String, Value>>()
            .into();
        // **Every key is written in alphabetical order, and that is load-bearing
        // rather than tidy.** `serde_json`'s map is a `BTreeMap` by default and
        // emits sorted; with the `preserve_order` feature — which a *dependency*
        // can turn on without this crate asking, and one in
        // `gui,distributed,zarr,model-segment` does — it is an `IndexMap` and
        // emits insertion order. A file written under one feature set and
        // compared byte for byte under the other differs on every line, which is
        // what `every_committed_scenario_loads_and_round_trips` found. Writing
        // in sorted order makes the two emissions the same bytes.
        serde_json::to_string_pretty(&json!({
            "budget_bytes": self.budget_bytes,
            "format": FORMAT,
            "machine": {
                "cache_bytes": self.machine.cache_bytes,
                "cache_shared": self.machine.cache_shared,
                "contention": self.machine.contention,
                "encoded_fraction": self.machine.encoded_fraction,
                "io_channels": self.machine.io_channels,
                "nodes": self.machine.nodes,
                "prefetch_depth": self.machine.prefetch_depth,
                "wave_synchronous": self.machine.wave_synchronous,
                "workers": self.machine.workers,
            },
            "name": self.name,
            "note": self.note,
            "storage": {
                "bytes_per_voxel": self.storage.bytes_per_voxel,
                "chunk": self.storage.chunk,
                "decode_ns_per_byte": self.storage.decode_ns_per_byte,
                "io_latency_ns": self.storage.io_latency_ns,
            },
            "terms": terms,
            "version": FORMAT_VERSION,
        }))
        .unwrap_or_default()
            + "\n"
    }

    /// The inverse of [`Self::to_json`].
    pub fn from_json(text: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text)
            .map_err(|err| Error::InvalidArgument(format!("not JSON: {err}")))?;
        let format = value.get("format").and_then(Value::as_str);
        let version = value.get("version").and_then(Value::as_u64);
        if format != Some(FORMAT) || version != Some(FORMAT_VERSION) {
            return Err(Error::InvalidArgument(format!(
                "expected {FORMAT} version {FORMAT_VERSION}, found {format:?} version \
                 {version:?}. A scenario read under the wrong interpretation is silently a \
                 different machine."
            )));
        }
        let name = value
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::InvalidArgument("no name".to_string()))?
            .to_string();
        let note = value
            .get("note")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let field =
            |parent: &str, key: &str| -> Option<f64> { value.get(parent)?.get(key)?.as_f64() };
        let machine = Machine {
            // Absent means one, so a scenario file written before nodes existed
            // is the single computer it was.
            nodes: field("machine", "nodes").unwrap_or(1.0) as usize,
            // Absent means the continuous dispatch every recorded figure was
            // taken under; a scenario that wants the executor's own wave
            // discipline says so.
            wave_synchronous: value
                .get("machine")
                .and_then(|m| m.get("wave_synchronous"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            workers: field("machine", "workers").unwrap_or(1.0) as usize,
            cache_bytes: field("machine", "cache_bytes").unwrap_or(0.0) as u64,
            prefetch_depth: field("machine", "prefetch_depth").unwrap_or(0.0) as usize,
            io_channels: field("machine", "io_channels").unwrap_or(1.0) as usize,
            cache_shared: value
                .get("machine")
                .and_then(|m| m.get("cache_shared"))
                .and_then(Value::as_bool)
                .unwrap_or(true),
            encoded_fraction: field("machine", "encoded_fraction").unwrap_or(0.0),
            contention: field("machine", "contention").unwrap_or(0.0),
            // **Not read from the file, and not written to one.** A scenario is
            // a machine and a set of costs; `Machine::candidate_window` is how
            // much of the ready set a *scheduler* is shown, which is a property
            // of the coordinator being simulated rather than of the machine it
            // runs on. Every committed scenario was recorded unbounded, and
            // `to_json` is compared byte for byte against `costs/`, so carrying
            // it would rewrite all of them to state the default.
            candidate_window: 0,
        };
        let chunk = value
            .get("storage")
            .and_then(|s| s.get("chunk"))
            .and_then(Value::as_array)
            .map(|axes| {
                let mut chunk = [1usize; 3];
                for (slot, axis) in chunk.iter_mut().zip(axes) {
                    *slot = axis.as_u64().unwrap_or(1) as usize;
                }
                chunk
            })
            .ok_or_else(|| Error::InvalidArgument("no storage.chunk".to_string()))?;
        let storage = Storage {
            chunk,
            bytes_per_voxel: field("storage", "bytes_per_voxel").unwrap_or(8.0),
            io_latency_ns: field("storage", "io_latency_ns").unwrap_or(0.0),
            decode_ns_per_byte: field("storage", "decode_ns_per_byte").unwrap_or(0.0),
        };
        let budget_bytes = value
            .get("budget_bytes")
            .and_then(Value::as_u64)
            .filter(|_| !value["budget_bytes"].is_null());
        let mut snapshot = Snapshot::empty(scenario_machine(&name));
        if let Some(terms) = value.get("terms").and_then(Value::as_object) {
            for (key, entry) in terms {
                let Some(term) = Term::parse(key) else {
                    continue;
                };
                let nanos = entry
                    .get("nanos_per_unit")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| {
                        Error::InvalidArgument(format!("term {key} has no nanos_per_unit"))
                    })?;
                let runs = entry
                    .get("runs")
                    .and_then(Value::as_u64)
                    .unwrap_or(REPRODUCTIONS as u64) as usize;
                let units = entry.get("units").and_then(Value::as_f64).unwrap_or(1.0);
                snapshot = snapshot.with(
                    term,
                    Coefficient {
                        nanos_per_unit: nanos,
                        runs,
                        units,
                        total_nanos: nanos * units,
                    },
                );
            }
        }
        Ok(Scenario {
            name,
            note,
            snapshot,
            machine,
            storage,
            budget_bytes,
        })
    }
}

/// A coefficient a scenario **states** rather than measures.
///
/// `runs` is [`REPRODUCTIONS`], which is what makes it believable — a scenario
/// whose terms were ignored as unreproduced would be a machine with no costs at
/// all. `units` is one: the field weights `calibrate`'s anchor across terms, and
/// a stated scenario has no work behind it to weight by, so every term carries
/// the same weight rather than a made-up one.
fn stated(nanos_per_unit: f64) -> Coefficient {
    Coefficient {
        nanos_per_unit,
        runs: REPRODUCTIONS,
        units: 1.0,
        total_nanos: nanos_per_unit,
    }
}

/// The machine key a scenario's snapshot is filed under.
///
/// **Not [`MachineKey::detect`]**, which would file a fictional machine's costs
/// under the host that happens to be running the test — and then a later real
/// run on that host would read them back as its own evidence. The host field
/// says what this is.
fn scenario_machine(name: &str) -> MachineKey {
    MachineKey {
        host: format!("scenario:{name}"),
        os: "any".to_string(),
        arch: "any".to_string(),
        pointer_width: 64,
        cpus: 0,
        profile: "scenario".to_string(),
    }
}

/// The baseline: **what this crate has actually measured**, with every term's
/// provenance in the note it carries.
///
/// This is the "keep the benchmarked costs" half. Every figure below appears in
/// the crate already and is cited where it does; nothing here is a new
/// measurement and nothing is invented. Where the crate has no measurement, the
/// seed is carried and *said to be* a seed, because a plausible-looking number
/// with no provenance is the thing this file exists to keep out of a robustness
/// claim.
pub fn measured_baseline() -> Scenario {
    let mut snapshot = Snapshot::empty(scenario_machine("measured"));
    // The tile run's per-phase compute, which is also `Rates::default`'s own
    // figure and the spread `planner-gaps.md` quotes: 3.541 / 98.329 / 201.397
    // nanoseconds per voxel, a factor of 57.
    snapshot = snapshot.with(Term::Compute, stated(98.329));
    for (family, nanos) in [
        ("combine", 3.541),
        ("smooth", 98.329),
        ("skeletonize", 201.397),
    ] {
        snapshot = snapshot.with(Term::ComputeOf(family.to_string()), stated(nanos));
    }
    // Memory bandwidth on this machine measured at 3.1-4.3 GB/s
    // (`docs/design/intra-block.md` §7). At 4 GB/s and eight bytes a voxel, a
    // voxel moved costs 2.0 ns; read and write are charged per voxel here, as
    // `CostModel` charges them.
    snapshot = snapshot.with(Term::Read, stated(2.0));
    snapshot = snapshot.with(Term::Write, stated(2.0));
    // An intermediate compresses better than an output — 19.7x for `bool`
    // against 2.09x for raw `uint16` (`executing-a-run.md`) — so it moves fewer
    // bytes for the same voxels. Half, which is the direction the measurement
    // supports at a magnitude it does not pin.
    snapshot = snapshot.with(Term::Materialise, stated(1.0));
    Scenario {
        name: "measured".to_string(),
        note: "What this crate has measured, cited at `scenario::measured_baseline`: the tile \
               run's per-family compute (3.541 / 98.329 / 201.397 ns per voxel), memory \
               bandwidth of 3.1-4.3 GB/s from `intra-block.md` §7, and the measured contention \
               of 0.40 from 2.41x realised against forty workers. Terms with no measurement \
               carry the crate's seed and are named in the same place."
            .to_string(),
        snapshot,
        machine: Machine {
            // One computer, which is what every figure this crate has recorded
            // was taken on.
            nodes: 1,
            // Continuous dispatch, which is what every figure this crate has
            // recorded was taken under. `costs/wave-synchronous` is the same
            // machine under the executor's own discipline.
            wave_synchronous: false,
            workers: 8,
            // A page cache of 4 GiB: `Machine::cache_bytes`'s own doc says the
            // budget is free RAM, and a figure detected from the host would make
            // every recorded number here unreproducible.
            cache_bytes: 4 << 30,
            prefetch_depth: 1,
            io_channels: 4,
            cache_shared: true,
            encoded_fraction: 0.0,
            // `MEASURED_CONTENTION`, from 2.41x realised against forty
            // requested. Off in `Machine::default` so that older figures stay
            // readable; on here, because this scenario is the machine.
            contention: crate::simulate::MEASURED_CONTENTION,
            // Unbounded, which is what this baseline's figures were measured
            // under. See `Machine::candidate_window`.
            candidate_window: 0,
        },
        storage: Storage {
            chunk: [64, 64, 64],
            bytes_per_voxel: 8.0,
            // Seeds, not measurements: no figure in this crate pins either.
            io_latency_ns: 0.0,
            decode_ns_per_byte: 0.0,
        },
        budget_bytes: Some(4 << 30),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A scenario round-trips through its file exactly**, which is what makes
    /// `costs/` a record rather than a rendering.
    ///
    /// Every field, including the ones a lazy reader would default: a `false`
    /// `cache_shared`, a `None` budget and a non-default chunk are all in here
    /// because each is a value whose default is the *other* answer, so a
    /// serialiser that dropped it would round-trip a different machine.
    #[test]
    fn a_scenario_round_trips_through_its_file() {
        let mut odd = measured_baseline()
            .with_scaled("odd", &[(Term::Read, 3.0)])
            .noted("a note");
        odd.machine.cache_shared = false;
        odd.machine.encoded_fraction = 0.25;
        odd.budget_bytes = None;
        odd.storage.chunk = [32, 16, 8];
        odd.storage.io_latency_ns = 1234.5;

        let back = Scenario::from_json(&odd.to_json()).expect("a scenario");
        assert_eq!(back, odd);
        assert!(!back.machine.cache_shared);
        assert_eq!(
            back.snapshot.machine(),
            odd.snapshot.machine(),
            "the snapshot's key is derived from the name and must survive a rename"
        );
        assert_eq!(back.budget_bytes, None);
        assert_eq!(back.storage.chunk, [32, 16, 8]);
    }

    /// A file this build cannot interpret is refused rather than read.
    #[test]
    fn a_file_of_another_format_or_version_is_refused() {
        let good = measured_baseline().to_json();
        assert!(Scenario::from_json(&good).is_ok());
        assert!(Scenario::from_json(&good.replace("\"version\": 1", "\"version\": 2")).is_err());
        assert!(
            Scenario::from_json(&good.replace("blockflow-scenario", "something-else")).is_err()
        );
        assert!(Scenario::from_json("not json").is_err());
    }

    /// **A transform moves what it names and nothing else.**
    ///
    /// The property that makes a derived scenario a claim the baseline supports:
    /// "a disk ten times slower" has to be the read term times ten, with the
    /// compute and the memory where they were, or the file is a different
    /// machine in more ways than its name says.
    #[test]
    fn a_transform_scales_the_term_it_names_and_leaves_the_rest() {
        let base = measured_baseline();
        let read = base
            .snapshot
            .coefficient(&Term::Read)
            .unwrap()
            .nanos_per_unit;
        let compute = base
            .snapshot
            .coefficient(&Term::Compute)
            .unwrap()
            .nanos_per_unit;

        let slower = base.clone().with_scaled("slower", &[(Term::Read, 10.0)]);
        assert_eq!(
            slower
                .snapshot
                .coefficient(&Term::Read)
                .unwrap()
                .nanos_per_unit,
            read * 10.0
        );
        assert_eq!(
            slower
                .snapshot
                .coefficient(&Term::Compute)
                .unwrap()
                .nanos_per_unit,
            compute,
            "scaling the read term moved the compute term"
        );
        assert_eq!(slower.machine, base.machine, "and it moved the machine");
        assert_eq!(slower.name, "slower");
    }

    /// **Both judges see one set of costs.**
    ///
    /// The model and the rates are derived from the same snapshot, so a
    /// scenario cannot state one cost to the planner and another to the
    /// simulator — and a robustness sweep in which they could would be
    /// measuring the disagreement rather than the planner. Asserted by moving
    /// one term and watching both halves move.
    #[test]
    fn the_model_and_the_rates_come_from_one_snapshot() {
        let base = measured_baseline();
        let slower = base.clone().with_scaled("slower", &[(Term::Read, 10.0)]);
        let seed_model = CostModel::default();
        let seed_rates = Rates::default();

        let before = (
            base.model(&seed_model).read_cost_per_voxel,
            base.rates(&seed_rates).io_ns_per_byte,
        );
        let after = (
            slower.model(&seed_model).read_cost_per_voxel,
            slower.rates(&seed_rates).io_ns_per_byte,
        );
        assert_eq!(after.0, before.0 * 10.0, "the planner did not see it");
        assert_eq!(after.1, before.1 * 10.0, "the simulator did not see it");
    }

    /// The machine limits are the scenario's and the search knobs are the
    /// caller's; see [`Scenario::constraints`].
    #[test]
    fn a_scenario_overlays_the_machine_and_leaves_the_search_space() {
        let scenario = measured_baseline().with_memory("small", 1 << 20);
        let base = Constraints {
            block_candidates: vec![7, 11],
            split_axes: vec![1],
            ..Default::default()
        };
        let constraints = scenario.constraints(&base);
        assert_eq!(constraints.budget_bytes, Some(1 << 20));
        assert_eq!(constraints.expected_concurrency, scenario.machine.workers);
        assert_eq!(
            constraints.block_candidates, base.block_candidates,
            "the ladder is the caller's question, not the machine's answer"
        );
        assert_eq!(constraints.split_axes, base.split_axes);
        assert_eq!(
            constraints.model.compute_scale,
            scenario.model(&base.model).compute_scale
        );
    }
}
