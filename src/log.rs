// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance criterion for the whole framework is *every chunk is affected
// by every op, in the right order*, and that is a **schedule** property, not a
// pixel property. So the executor emits an event log and the criterion is
// asserted directly off it:
//
// * **coverage** — every block appears with every op in the chain;
// * **order** — each block's op sequence equals the chain order;
// * **no duplication** — no `(block, op)` pair appears twice.
//
// Asserting it from a log rather than from encoded pixel values matters for two
// reasons. No arrays are needed, so the check runs on synthetic volumes far
// larger than anything worth allocating; and the assertion reads as the
// property itself rather than as arithmetic standing in for it.
//
// The division of labour, and why both halves are needed
// -----------------------------------------------------
// The log proves the **scheduling** is right — what ran, in what order, how
// often. It cannot prove the **geometry** is right: writing a correct region to
// the wrong offset, an off-by-one valid region, or a mis-stitched block all
// produce a perfect log. Geometry is proven separately, by identity ops over a
// modest real array where the expected output is the input.
//
// So: **log for scheduling at scale, arrays for geometry at small scale.**
//
// The cost accumulation here is what feeds back into validating the cost model.
// A plan predicting N reads against an execution performing 3N is a planner bug
// the log shows and output equality never would.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::error::{Error, Result};
use crate::region::Region;

/// One thing the executor did.
///
/// Three layers, deliberately kept as one enum
/// -------------------------------------------
/// * **scheduling** — `PhaseStarted`, `TaskAdmitted`. What the scheduler chose,
///   before any work happened.
/// * **op** — `BlockRead`, `OpApplied`, `BlockShortCircuited`, `BlockWritten`.
///   The block's lifecycle: what was computed over what extent.
/// * **IO** — `RegionRead`, `RegionWritten`, `Materialised`. Bytes moved, and
///   how long moving them took.
///
/// One enum rather than three streams because the statistics store wants all of
/// it and merging three ingestion paths after the fact is worse than one enum
/// with a `match`. See `EventListener` for what a consumer may and may not do.
///
/// **Granularity of the IO variants.** One event per *caller-level* call — one
/// `RegionRead` per `Environment::read`, not one per chunk. A block spans many
/// chunks, so per-chunk emission would be 10-100x the rate for detail nobody
/// has asked for; `chunks` carries the count instead. Per-chunk detail, if it
/// is ever wanted, belongs in a separate opt-in variant so that the default
/// rate stays bounded by the task count.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    PhaseStarted {
        phase: usize,
    },
    /// The scheduler took this task off the ready set. Emitted before any of
    /// the task's work, so `TaskAdmitted` order **is** the schedule — the one
    /// place an advisory `visit_order` or a priority key becomes observable.
    TaskAdmitted {
        phase: usize,
        index: [usize; 3],
    },
    /// The IO layer moved bytes *in*. `level` is the storage level read from:
    /// level 0 is the workflow input, level `p` is the intermediate phase `p-1`
    /// wrote. Emitted once per `Environment::read`.
    RegionRead {
        /// Human-readable name of what was read — `"level 0"` from the
        /// executor, `RegionSource::describe()` from an observed source.
        source: String,
        level: usize,
        /// The block this read served, when the reader knows. A bare
        /// `RegionSource` does not, and says so rather than inventing one.
        index: Option<[usize; 3]>,
        region: Region,
        voxels: usize,
        bytes: u64,
        chunks: u64,
        duration_ns: u64,
    },
    /// The IO layer moved bytes *out*, once per `Environment::write`.
    RegionWritten {
        sink: String,
        level: usize,
        index: Option<[usize; 3]>,
        region: Region,
        voxels: usize,
        bytes: u64,
        chunks: u64,
        duration_ns: u64,
    },
    /// A phase's output is complete and durable at `level`. `bytes` is what
    /// that phase wrote in total. This is the materialisation the fuse-vs-spill
    /// decision is about, priced.
    Materialised {
        phase: usize,
        level: usize,
        bytes: u64,
        /// `false` for the last phase, whose destination is the workflow
        /// output rather than an intermediate.
        intermediate: bool,
    },
    BlockRead {
        phase: usize,
        index: [usize; 3],
        region: Region,
        voxels: usize,
        chunks: u64,
    },
    OpApplied {
        phase: usize,
        index: [usize; 3],
        slot: usize,
        op: String,
        /// The extent the op was computed over — the block's **read** extent,
        /// not its core. Halo margins are recomputed by every block that reads
        /// them, and that duplication is legitimate; recording the extent makes
        /// it visible and checkable rather than hidden.
        over: Region,
        /// Wall time inside `Environment::apply`. Measured, not modelled — this
        /// is the quantity the cost model's `cost_per_voxel` is checked against.
        duration_ns: u64,
    },
    BlockShortCircuited {
        phase: usize,
        index: [usize; 3],
        /// The uniform input value that licensed the skip.
        from: f64,
        /// The constant every op in the phase declared it maps to.
        to: f64,
        /// The slots that were skipped. They still count as covered: the block
        /// produces exactly what computing them would have.
        slots: Vec<usize>,
        names: Vec<String>,
    },
    BlockWritten {
        phase: usize,
        index: [usize; 3],
        valid: Region,
        /// `true` when the destination is an intermediate rather than the
        /// workflow output.
        materialised: bool,
    },

    // ---------------------------------------------------- sidecar layer --
    //
    // Per-block output that is not a pixel region: a fragment a later global
    // step merges. Keyed by the decomposition's own unit, so these events join
    // the rest of the stream on `(phase, index)` with nothing to translate.
    //
    // Granularity is per fragment, which is per block — the same rate as
    // `BlockWritten`, not the chunk rate the cache events run at.
    /// A block's fragment was written to `stream`.
    SidecarWritten {
        stream: String,
        phase: usize,
        index: [usize; 3],
        bytes: u64,
        duration_ns: u64,
    },
    /// A fragment was asked for. `found` distinguishes "this block wrote
    /// nothing" from "this block wrote nothing *of length zero*", which a byte
    /// count alone cannot.
    SidecarRead {
        stream: String,
        phase: usize,
        index: [usize; 3],
        bytes: u64,
        found: bool,
        duration_ns: u64,
    },
    /// A whole stream was removed, with what went.
    ///
    /// On the event stream and not only in the caller's return value,
    /// deliberately: the cleanup bug this is arranged against was a removal
    /// nobody could see had not happened.
    SidecarDiscarded {
        stream: String,
        fragments: usize,
        bytes: u64,
    },

    /// A block's slice of an array the op writes **beside** its primary result.
    ///
    /// Its own event rather than a `RegionWritten` with a different sink,
    /// because a side output is addressed by name in a space of its own rank —
    /// there is no level for it and its region is not a region of one. The
    /// bytes are stated, since that is the figure a run was silently short of.
    SideOutputWritten {
        output: String,
        phase: usize,
        index: [usize; 3],
        region: Region,
        bytes: u64,
    },

    // ------------------------------------------------------ cache layer --
    //
    // A fifth layer, and the reason it is in the same enum as the other four:
    // a cache hit rate is only interesting *against* the schedule that
    // produced it. Two blocks admitted in a different order give different hit
    // rates over identical work, so a separate cache-event stream would have
    // to be re-joined to this one before it could be read, and joining two
    // logs after the fact is worse than one `match`.
    //
    // Granularity here is per **chunk**, not per caller-level call, which is a
    // deliberate departure from the IO variants above. The whole point of the
    // cache is what happens chunk by chunk — one region read may be four hits
    // and one miss, and reporting it as a single event would erase exactly the
    // thing being measured. The rate is bounded by chunks touched rather than
    // by tasks, so a listener that cannot afford it should not register for a
    // run with the cache on.
    /// A chunk was served without touching the source.
    ///
    /// `tier` says what it cost: `Decoded` is a copy, `Encoded` is a copy plus
    /// a decompression. A cache whose hits are nearly all `Encoded` is saving
    /// IO but not decode, which is a different — and smaller — win than the
    /// prefetcher is trying to buy.
    CacheHit {
        array: String,
        chunk: u64,
        tier: Tier,
        bytes: u64,
        /// Nanoseconds spent turning the entry back into decoded bytes. Zero
        /// for a `Decoded` hit, which is what makes "a hit must be free"
        /// checkable rather than asserted.
        decode_ns: u64,
    },
    /// A chunk was not in the cache and was read from the source.
    ///
    /// `duration_ns` is the source read, so miss cost and hit cost are directly
    /// comparable in one stream.
    CacheMiss {
        array: String,
        chunk: u64,
        bytes: u64,
        duration_ns: u64,
    },
    /// A chunk was dropped to make room for another.
    ///
    /// `for_array` names who took the space. When it differs from `array` the
    /// cache is doing the one thing a per-array cache cannot: moving capacity
    /// between arrays as demand moves.
    CacheEvicted {
        array: String,
        chunk: u64,
        tier: Tier,
        bytes: u64,
        for_array: String,
        /// The entry was prefetched and never read. See `PrefetchWasted`.
        prefetched_unused: bool,
    },
    /// The budget refused the bytes, so the chunk was served but not retained.
    ///
    /// Not an error and not a miss: the read happened and the caller got its
    /// data. It is the cache degrading into a pass-through under memory
    /// pressure, which is the required behaviour and worth being able to see.
    CacheRefused {
        array: String,
        chunk: u64,
        bytes: u64,
    },
    /// A chunk was known empty without reading it, so nothing was read.
    ///
    /// The saving `RegionSource::is_known_empty` exists for. Distinct from a
    /// hit because no capacity was spent to get it.
    CacheKnownEmpty {
        array: String,
        chunk: u64,
        bytes: u64,
    },

    // --------------------------------------------------- prefetch layer --
    /// The prefetcher started fetching a chunk nobody has asked for yet.
    PrefetchIssued {
        array: String,
        chunk: u64,
        rank: u32,
    },
    /// A demand read was served by an entry the prefetcher had put there.
    ///
    /// `waited_ns` is how long the reader had to wait for a fetch still in
    /// flight — zero when the entry was already resident, which is the case the
    /// prefetcher exists to produce.
    PrefetchUsed {
        array: String,
        chunk: u64,
        waited_ns: u64,
    },
    /// A prefetch produced nothing anyone used.
    ///
    /// **This is the cost of depth, and it is the number that says the depth is
    /// wrong.** A prefetcher tuned too shallow shows a low hit rate; one tuned
    /// too deep shows waste. Neither is visible from throughput alone, which is
    /// why this is an event rather than a debug print.
    PrefetchWasted {
        array: String,
        chunk: u64,
        reason: PrefetchWaste,
    },
}

/// Which of the cache's two tiers an entry is in.
///
/// Re-exported from `cache`, where the trade is described. Named in the event
/// stream because "was that hit free?" is the first question anyone asks of a
/// cache hit rate.
pub use crate::cache::Tier;

/// Why a prefetch bought nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrefetchWaste {
    /// It was evicted before anybody read it. Depth is too large for the
    /// capacity, or the rank order does not match the consumption order.
    Evicted,
    /// Its plan was cancelled before it ran. Not a tuning signal — a slab
    /// finishing early is the system working.
    Cancelled,
    /// The budget refused to retain it. Memory pressure, not mis-tuning.
    Refused,
}

/// Everything the executor did, in the order it did it.
///
/// Concurrent tasks interleave, so the *global* order is not deterministic —
/// but a single block's op sequence is, because one task applies a phase's ops
/// back to back. That is exactly the ordering the acceptance criterion is
/// about.
#[derive(Debug, Default)]
pub struct ExecutionLog {
    events: Mutex<Vec<Event>>,
}

impl ExecutionLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, event: Event) {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event);
    }

    pub fn events(&self) -> Vec<Event> {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn len(&self) -> usize {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// For each block, the ops that touched it, in the order they touched it.
    ///
    /// A short-circuited block contributes the ops it skipped: the op was
    /// *applied* in the sense that matters, because the short circuit fires
    /// only where the op has stated the constant it produces, so the block
    /// holds exactly what computing it would have produced.
    pub fn op_sequence_per_block(&self) -> BTreeMap<[usize; 3], Vec<(usize, String)>> {
        let mut out: BTreeMap<[usize; 3], Vec<(usize, String)>> = BTreeMap::new();
        for event in self.events() {
            match event {
                Event::OpApplied {
                    index, slot, op, ..
                } => out.entry(index).or_default().push((slot, op)),
                Event::BlockShortCircuited {
                    index,
                    slots,
                    names,
                    ..
                } => {
                    let entry = out.entry(index).or_default();
                    for (slot, name) in slots.into_iter().zip(names) {
                        entry.push((slot, name));
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// The order blocks were first read, per phase. Advisory `visit_order` is
    /// only observable here, which is why it is recorded.
    pub fn visit_order(&self, phase: usize) -> Vec<[usize; 3]> {
        self.events()
            .into_iter()
            .filter_map(|event| match event {
                Event::BlockRead {
                    phase: at, index, ..
                } if at == phase => Some(index),
                _ => None,
            })
            .collect()
    }

    /// The acceptance criterion, asserted directly: **every block affected by
    /// every op, in the right order, once each.**
    ///
    /// `expected` is the chain's slot order — `(slot index, op name)` pairs.
    /// `expected_blocks` is how many distinct blocks must appear.
    ///
    /// This keys on the block's 3-D index, so it **assumes every phase shares
    /// one block grid** — which every strategy here produces today, because
    /// each picks one block size per phase and the candidates rarely differ.
    /// Per-phase block sizes are explicitly left open by the design; when a
    /// strategy actually varies them, index `[0,0,0]` in phase 0 and in phase 1
    /// will name different regions and this check must grow a per-phase key
    /// before it means anything. Stated here rather than asserted because the
    /// log does not carry the grid, and inventing a signal to detect it would
    /// be a second source of truth for the decomposition.
    pub fn check_coverage_and_order(
        &self,
        expected: &[(usize, String)],
        expected_blocks: usize,
    ) -> Result<()> {
        let per_block = self.op_sequence_per_block();
        if per_block.len() != expected_blocks {
            return Err(Error::InvalidArgument(format!(
                "coverage: {} blocks appear in the log, expected {expected_blocks}",
                per_block.len()
            )));
        }
        for (index, applied) in &per_block {
            if applied.len() != expected.len() {
                return Err(Error::InvalidArgument(format!(
                    "block {index:?}: {} ops applied, expected {}. Applied: {:?}",
                    applied.len(),
                    expected.len(),
                    applied.iter().map(|(_, name)| name).collect::<Vec<_>>()
                )));
            }
            if applied != expected {
                return Err(Error::InvalidArgument(format!(
                    "block {index:?}: op sequence {:?} is not the chain order {:?}",
                    applied, expected
                )));
            }
        }
        Ok(())
    }

    /// `(block, slot)` pairs applied more than once.
    ///
    /// Should always be empty: a phase boundary recomputes a *halo margin*,
    /// which shows up as overlapping `over` extents, never as a second
    /// application to the same block.
    pub fn duplicate_applications(&self) -> Vec<([usize; 3], usize, usize)> {
        let mut counts: BTreeMap<([usize; 3], usize), usize> = BTreeMap::new();
        for (index, applied) in self.op_sequence_per_block() {
            for (slot, _) in applied {
                *counts.entry((index, slot)).or_default() += 1;
            }
        }
        counts
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .map(|((index, slot), count)| (index, slot, count))
            .collect()
    }

    /// Total voxels an op was computed over, minus the voxels actually kept.
    ///
    /// This is the halo redundancy the cost model predicts, measured. A phase
    /// split is supposed to reduce it; if it does not, the partition bought
    /// nothing and the cost model is wrong about something.
    pub fn recomputed_margin_voxels(&self) -> u64 {
        let mut over = 0u64;
        let mut kept = 0u64;
        for event in self.events() {
            match event {
                Event::OpApplied { over: region, .. } => over += region.voxels() as u64,
                Event::BlockWritten { valid, .. } => kept += valid.voxels() as u64,
                _ => {}
            }
        }
        over.saturating_sub(kept)
    }
}

/// What a run cost, whether it moved bytes or only counted them.
#[derive(Debug)]
pub struct Stats {
    pub strategy: &'static str,
    pub decomposition_fingerprint: u64,
    pub phases: usize,
    pub tasks: usize,
    pub tasks_short_circuited: usize,
    pub ops_applied: usize,
    pub blocks_visited: usize,
    /// Writes whose destination is an intermediate, not the workflow output.
    pub materialisations: usize,
    /// Substages each phase ran, and zero for every phase that is not an
    /// iteration.
    ///
    /// **The one figure of a run that is a function of the data**, and therefore
    /// the one that is reported rather than planned: an iterative phase runs to
    /// convergence, so its count appears in no reach, no level allocation and no
    /// phase structure. Two runs of one plan over two datasets may differ here
    /// and agree everywhere else, which is why `same_work_as` does not compare
    /// it — a strategy that honoured a foreign decomposition did the same
    /// *planned* work whatever the data made it iterate.
    pub substages: Vec<usize>,
    /// What each substage of each phase **changed**, in substage order, and an
    /// empty list for every phase that is not an iteration.
    ///
    /// The same reduction [`Stats::substages`] counts, reported as a quantity
    /// instead of only as a threshold. An iteration stops when a substage
    /// changes nothing anywhere, so the executor already compares every block's
    /// core against what it went in as; this is the size of that difference,
    /// summed over the blocks, rather than the bit it is collapsed to.
    ///
    /// It is worth carrying because the *shape* of the sequence is the thing a
    /// caller can act on and the count alone is not: a sequence that decays
    /// geometrically is an iteration converging, one that plateaus is an
    /// iteration that will hit its limit, and the two are indistinguishable from
    /// a substage count taken after the fact. The last entry is zero exactly
    /// when the phase ended by converging, which it always does — the runaway
    /// limit is an error and not a return.
    ///
    /// Zero-length for a phase run under an environment that holds no data, on
    /// the same argument [`crate::env::Environment::same`] returns `None` there:
    /// a simulated run has no values to difference.
    pub substage_changes: Vec<Vec<u64>>,
    /// Calls into the environment's loader, and what they moved. A plan
    /// predicting N reads against a run performing 3N is a planner bug the log
    /// shows and output equality never would.
    pub reads: u64,
    pub writes: u64,
    pub read_voxels: u64,
    pub write_voxels: u64,
    pub chunks_read: u64,
    /// The arrays the ops wrote beside their primary results, and what they
    /// came to. Beside `writes`/`write_voxels` rather than folded into them: a
    /// side output has its own element type and its own rank, so its elements
    /// are not the level's voxels. `side_bytes_written` is the commensurable
    /// figure, and it is the one a run used to be short of — 95.2 MB counted
    /// against 158.6 MB written.
    pub side_writes: u64,
    pub side_bytes_written: u64,
    pub peak_resident_bytes: u64,
    /// Sum over applied ops of `voxels x cost_per_voxel`. A modelled quantity,
    /// useful for ranking strategies against each other and for nothing else —
    /// "trust A beats B, distrust A takes 40 minutes".
    pub estimated_work: f64,
    /// Registered listeners that panicked and were disabled mid-run. Always
    /// zero for a well-behaved run; non-zero means an *observer* broke and the
    /// results are still valid. See `listener::EventListener`.
    pub listener_faults: usize,
    pub log: ExecutionLog,
}

impl Stats {
    /// Whether two runs did the same amount of the things the decomposition
    /// controls. Used to check that a strategy honoured a foreign
    /// decomposition rather than quietly substituting its own.
    pub fn same_work_as(&self, other: &Stats) -> bool {
        self.decomposition_fingerprint == other.decomposition_fingerprint
            && self.phases == other.phases
            && self.tasks == other.tasks
            && self.ops_applied + self.skipped_op_applications()
                == other.ops_applied + other.skipped_op_applications()
            && self.blocks_visited == other.blocks_visited
    }

    fn skipped_op_applications(&self) -> usize {
        self.log
            .events()
            .into_iter()
            .filter_map(|event| match event {
                Event::BlockShortCircuited { slots, .. } => Some(slots.len()),
                _ => None,
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn applied(index: [usize; 3], slot: usize, name: &str) -> Event {
        Event::OpApplied {
            phase: 0,
            index,
            slot,
            op: name.to_string(),
            over: Region::new(&[0, 0, 0], &[1, 1, 1]),
            duration_ns: 0,
        }
    }

    fn expected() -> Vec<(usize, String)> {
        vec![
            (0, "a".to_string()),
            (1, "b".to_string()),
            (2, "c".to_string()),
        ]
    }

    #[test]
    fn a_complete_in_order_log_passes() {
        let log = ExecutionLog::new();
        for block in 0..3 {
            for (slot, name) in [(0, "a"), (1, "b"), (2, "c")] {
                log.push(applied([block, 0, 0], slot, name));
            }
        }
        log.check_coverage_and_order(&expected(), 3).unwrap();
        assert!(log.duplicate_applications().is_empty());
    }

    #[test]
    fn a_missing_op_is_caught() {
        let log = ExecutionLog::new();
        log.push(applied([0, 0, 0], 0, "a"));
        log.push(applied([0, 0, 0], 2, "c"));
        let err = log.check_coverage_and_order(&expected(), 1).unwrap_err();
        assert!(err.to_string().contains("2 ops applied, expected 3"));
    }

    #[test]
    fn ops_out_of_order_are_caught() {
        let log = ExecutionLog::new();
        log.push(applied([0, 0, 0], 0, "a"));
        log.push(applied([0, 0, 0], 2, "c"));
        log.push(applied([0, 0, 0], 1, "b"));
        let err = log.check_coverage_and_order(&expected(), 1).unwrap_err();
        assert!(err.to_string().contains("is not the chain order"));
    }

    #[test]
    fn a_duplicated_op_is_caught_twice_over() {
        let log = ExecutionLog::new();
        for (slot, name) in [(0, "a"), (1, "b"), (1, "b"), (2, "c")] {
            log.push(applied([0, 0, 0], slot, name));
        }
        assert!(log.check_coverage_and_order(&expected(), 1).is_err());
        assert_eq!(log.duplicate_applications(), vec![([0, 0, 0], 1, 2)]);
    }

    #[test]
    fn a_skipped_block_is_caught() {
        let log = ExecutionLog::new();
        for (slot, name) in [(0, "a"), (1, "b"), (2, "c")] {
            log.push(applied([0, 0, 0], slot, name));
        }
        let err = log.check_coverage_and_order(&expected(), 2).unwrap_err();
        assert!(err.to_string().contains("1 blocks appear in the log"));
    }

    #[test]
    fn a_short_circuited_block_still_counts_as_covered() {
        let log = ExecutionLog::new();
        log.push(Event::BlockShortCircuited {
            phase: 0,
            index: [0, 0, 0],
            from: 0.0,
            to: 0.0,
            slots: vec![0, 1, 2],
            names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        });
        log.check_coverage_and_order(&expected(), 1).unwrap();
    }
}
