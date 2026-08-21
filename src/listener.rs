// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Pluggable consumption of the executor's event stream.
//
// `log.rs` already emits one `Event` per `(block, phase, op)` and accumulates
// them into an `ExecutionLog`. That was fine while the log was the *only*
// consumer, but two more want the same stream: a live progress view that a
// visualiser polls **while the run is in flight**, and whatever offline
// analysis wants to look at afterwards. Growing `ExecutionLog` a second and a
// third responsibility would have made the acceptance criterion — which is
// asserted off that type — harder to read, so the stream is made pluggable
// instead and `ExecutionLog` becomes one listener among others.
//
// The rule the whole file is built around
// ---------------------------------------
// > **A listener may observe the run. It may never influence it.**
//
// This is the same principle the design states for hints, cost models and the
// statistics store — everything advisory must be unable to affect correctness —
// applied to observation rather than to planning. Concretely:
//
// * **Listeners see `&Event`, never anything mutable.** There is no channel by
//   which a listener can reach the scheduler, the ready heap, the environment
//   or a block buffer. Dispatch happens *after* the work it describes; an event
//   is a report, not a request.
// * **A panicking listener is isolated, not propagated.** `on_event` runs
//   inside `catch_unwind`. The first panic disables that listener for the rest
//   of the run and increments `Stats::listener_faults`; the run continues and
//   its results are unaffected. The alternative — letting the panic unwind
//   through `run_task` — would turn a broken *observer* into a failed run,
//   which is exactly the influence this file exists to deny. The default panic
//   hook still prints the first panic, which is how the fault gets diagnosed;
//   disabling the listener is what stops it printing 60 000 times.
//   (`catch_unwind` cannot help under `panic = "abort"`. This crate does not
//   set it, and if it ever does, a panicking listener aborts the process —
//   stated here rather than silently assumed.)
// * **A slow listener slows the run and nothing else.** Dispatch is synchronous
//   on the worker thread, so a listener that blocks costs throughput. It cannot
//   change *what* is computed, only when. Listeners that might be slow should
//   buffer and do their work elsewhere; `LatestOpPerChunk` does exactly that by
//   keeping its state to one atomic word per block.
//
// The cost of dispatch
// --------------------
// The event path is hot: one event per `(block, phase, op)`, and a full-scale
// run is ~60 000 tasks x ops. So the dispatch set is **fixed for the duration
// of a run** — listeners are registered before `execute` and the executor holds
// a plain `&[Arc<dyn EventListener>]`. There is no registry lock, no
// `RwLock` around the listener list, nothing to contend on: dispatch with zero
// listeners is a slice-length check, and with N listeners it is N virtual
// calls. The only locking on the path is whatever a listener itself does, which
// is why both listeners here are built to avoid an exclusive lock —
// `LatestOpPerChunk` takes a *shared* read guard plus one atomic CAS, and only
// takes an exclusive guard the first time a given block is seen.
//
// Measured overhead is reported by `dispatch_overhead_report`, which the
// `#[ignore]`d test at the bottom prints.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use super::log::{Event, ExecutionLog};

/// Something that watches the executor.
///
/// Implementations must be cheap and must not assume any ordering between
/// blocks: concurrent tasks interleave, so the only order a listener may rely
/// on is *within one block*, where a task applies a phase's ops back to back.
///
/// An implementation may panic; see the module header. It cannot fail the run.
pub trait EventListener: Send + Sync {
    fn on_event(&self, event: &Event);
}

/// The order log **is** the offline-analysis listener the design asks for, so
/// it is aliased rather than reimplemented. `ExecutionLog` keeps its name where
/// the acceptance criterion is asserted from it; `OrderLog` reads better where
/// it is being used as one listener among others.
pub type OrderLog = ExecutionLog;

impl EventListener for ExecutionLog {
    fn on_event(&self, event: &Event) {
        self.push(event.clone());
    }
}

// ------------------------------------------------------------- dispatch --

/// The executor's fan-out point: the built-in order log, plus whatever the
/// caller registered.
///
/// Holds the built-in log **by value** so it can move straight into `Stats`
/// without an `Arc` dance, and the extra listeners by shared reference so that
/// registering a listener costs one pointer and no allocation per event.
///
/// # What ordering a listener gets, and what it costs
///
/// Dispatch is **not** serialised across workers. Each worker fans one event
/// out to the listeners and then appends it to the built-in log, so:
///
/// * every listener sees **every** event — nothing is dropped or coalesced;
/// * every listener sees **one block's** events in the order they happened,
///   because one task emits them from one thread, back to back;
/// * two listeners may see two **different interleavings** of concurrent
///   blocks, and either may differ from the built-in log's. The global order is
///   a scheduling artefact in the first place.
///
/// Serialising dispatch under one lock would make all three orders identical,
/// and would cost the thing this file exists to protect: every worker would
/// queue behind the slowest listener, so a listener could throttle the run. The
/// built-in log already takes a `Mutex` per event, but only around a `Vec::push`
/// — microseconds of critical section, not an arbitrary callback. Widening that
/// window to include user code is the trade this design refuses.
pub(super) struct Dispatch<'a> {
    log: ExecutionLog,
    listeners: &'a [Arc<dyn EventListener>],
    /// One flag per listener, set when that listener panics. Read on every
    /// event, so `Relaxed` — a one-event delay in seeing a fault is harmless,
    /// and the flag exists to stop repetition, not to synchronise anything.
    faulted: Vec<AtomicBool>,
    faults: AtomicUsize,
}

impl<'a> Dispatch<'a> {
    pub(super) fn new(listeners: &'a [Arc<dyn EventListener>]) -> Self {
        Self {
            log: ExecutionLog::new(),
            faulted: listeners.iter().map(|_| AtomicBool::new(false)).collect(),
            listeners,
            faults: AtomicUsize::new(0),
        }
    }

    /// Report one thing the executor did. Never fails, never propagates a
    /// listener's panic.
    pub(super) fn emit(&self, event: Event) {
        for (slot, listener) in self.listeners.iter().enumerate() {
            if self.faulted[slot].load(Ordering::Relaxed) {
                continue;
            }
            // `AssertUnwindSafe` is the honest annotation: a listener that
            // panics mid-update may leave *its own* state torn, and that is its
            // problem. Nothing the executor owns crosses this boundary, so no
            // invariant of the run can be broken by an unwind here.
            let outcome = catch_unwind(AssertUnwindSafe(|| listener.on_event(&event)));
            if outcome.is_err() {
                self.faulted[slot].store(true, Ordering::Relaxed);
                self.faults.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.log.push(event);
    }

    /// Listeners that panicked and were disabled.
    pub(super) fn faults(&self) -> usize {
        self.faults.load(Ordering::Relaxed)
    }

    pub(super) fn into_log(self) -> ExecutionLog {
        self.log
    }

    pub(super) fn log(&self) -> &ExecutionLog {
        &self.log
    }
}

// ------------------------------------------------------ latest op per chunk --

/// What last happened to a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressKind {
    /// The scheduler picked the block's task; nothing has been read yet.
    Admitted,
    /// The block was read; no op has landed yet.
    Read,
    /// An op was applied.
    OpApplied,
    /// The phase's ops were skipped because the block folded to a constant.
    /// The block holds what computing them would have produced, so for progress
    /// purposes the slot named here *has* been applied.
    ShortCircuited,
    /// The block's valid region was written.
    Written,
}

impl ProgressKind {
    fn code(self) -> u64 {
        match self {
            ProgressKind::Admitted => 0,
            ProgressKind::Read => 1,
            ProgressKind::OpApplied => 2,
            ProgressKind::ShortCircuited => 3,
            ProgressKind::Written => 4,
        }
    }

    fn from_code(code: u64) -> Self {
        match code {
            0 => ProgressKind::Admitted,
            1 => ProgressKind::Read,
            2 => ProgressKind::OpApplied,
            3 => ProgressKind::ShortCircuited,
            _ => ProgressKind::Written,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ProgressKind::Admitted => "admitted",
            ProgressKind::Read => "read",
            ProgressKind::OpApplied => "op_applied",
            ProgressKind::ShortCircuited => "short_circuited",
            ProgressKind::Written => "written",
        }
    }
}

/// One block's most recent state.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockProgress {
    pub index: [usize; 3],
    pub phase: usize,
    pub kind: ProgressKind,
    /// The last op that landed on this block, as a chain slot. **Sticky**: a
    /// read or a write does not clear it, so this stays the answer to "how far
    /// has this block got" between ops and across a phase boundary. `None`
    /// only before the block's first op.
    pub slot: Option<usize>,
    /// The op's display name, resolved from the slot. `None` when no op has
    /// landed yet, or when no event has yet named that slot.
    pub op: Option<String>,
    /// Emission order. Monotonically increasing across the whole run, so
    /// `seq` compares states *between* blocks as well as within one.
    pub seq: u64,
}

/// Per-block "what happened last", readable **while the run is going on**.
///
/// # Consistency
///
/// Two different guarantees, and it matters which is which:
///
/// * **Per block: exact.** A block's state is one `AtomicU64`, written with a
///   single store (guarded by a CAS so it can never move backwards) and read
///   with a single load. Every `BlockProgress` a snapshot returns is a state
///   that block genuinely held at some instant. Nothing is torn, and no
///   combination of `(phase, slot, kind)` is ever observed that the executor
///   did not emit.
/// * **Across blocks: not an instant.** `snapshot` walks the blocks one at a
///   time, so entry A may be read before entry B is updated. The result is a
///   *causally plausible* view — every entry real, no entry from the future,
///   per-block monotone — but not a global snapshot of one moment. Providing
///   one would need the executor to stop, or a full epoch/versioned copy on
///   every event, and a visualiser polling at 30 Hz gains nothing from either.
///   `seq` is included precisely so a consumer can see the spread if it cares.
///
/// A poll never blocks the executor's steady state: readers take the same
/// *shared* guard the write path takes, so they do not exclude each other. The
/// exclusive guard is taken once per block, the first time it is seen.
///
/// # Cost on the write path
///
/// One shared `RwLock` acquire, one hash lookup, one CAS. No allocation, and no
/// growth in memory with the number of events — the state is 8 bytes per block,
/// so a 60 000-block run costs under half a megabyte.
#[derive(Debug, Default)]
pub struct LatestOpPerChunk {
    /// `Box` so the `AtomicU64` has a stable address across rehashes, which is
    /// what lets the shared-guard fast path hand out `&AtomicU64`.
    blocks: RwLock<HashMap<[usize; 3], Box<AtomicU64>, BuildBlockHasher>>,
    /// slot -> display name, learned from the events. Written at most once per
    /// distinct slot, read only by `snapshot`, never on the hot path.
    names: RwLock<HashMap<usize, String>>,
    /// Bit *s* set means slot *s*'s name is already known. Checking this
    /// instead of taking the `names` read guard on every op event is worth
    /// about 30 ns per event, which at 60 000 tasks x ops is the difference
    /// between "free" and "noticeable".
    known_names: AtomicU64,
    seq: AtomicU64,
}

/// A hasher for `[usize; 3]` block indices.
///
/// The default `SipHash` is chosen for HashDoS resistance against adversarial
/// keys. Block indices are generated by our own decomposition and are small,
/// dense and non-adversarial, so the resistance buys nothing and the ~20 ns it
/// costs is on the event path. This is the standard multiply-xor mix.
#[derive(Debug, Default, Clone, Copy)]
struct BlockHasher(u64);

impl std::hash::Hasher for BlockHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.write_usize(*byte as usize);
        }
    }

    fn write_usize(&mut self, value: usize) {
        self.0 = (self.0 ^ value as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 ^= self.0 >> 29;
    }

    fn write_u8(&mut self, value: u8) {
        self.write_usize(value as usize);
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct BuildBlockHasher;

impl std::hash::BuildHasher for BuildBlockHasher {
    type Hasher = BlockHasher;
    fn build_hasher(&self) -> BlockHasher {
        BlockHasher::default()
    }
}

/// Packing. 64 bits, one word, one atomic operation:
///
/// | bits | field |
/// |---|---|
/// | 63..32 | emission sequence (saturating at `u32::MAX`) |
/// | 31..28 | `ProgressKind` code |
/// | 27..16 | phase (saturating at 4095) |
/// | 15..0  | slot + 1, or 0 for "no slot" (saturating at 65534) |
fn pack(seq: u64, kind: ProgressKind, phase: usize, slot: Option<usize>) -> u64 {
    let seq = seq.min(u32::MAX as u64);
    let phase = (phase.min(0xFFF)) as u64;
    let slot = match slot {
        Some(slot) => (slot.min(0xFFFE) + 1) as u64,
        None => 0,
    };
    (seq << 32) | (kind.code() << 28) | (phase << 16) | slot
}

fn unpack(word: u64) -> (u64, ProgressKind, usize, Option<usize>) {
    let seq = word >> 32;
    let kind = ProgressKind::from_code((word >> 28) & 0xF);
    let phase = ((word >> 16) & 0xFFF) as usize;
    let slot = match word & 0xFFFF {
        0 => None,
        raw => Some((raw - 1) as usize),
    };
    (seq, kind, phase, slot)
}

impl LatestOpPerChunk {
    pub fn new() -> Self {
        Self::default()
    }

    fn store(&self, index: [usize; 3], word: u64) {
        // Fast path: the block already has a cell. A *shared* guard, so several
        // workers and a polling visualiser all proceed at once.
        {
            let blocks = self
                .blocks
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(cell) = blocks.get(&index) {
                bump(cell, word);
                return;
            }
        }
        let mut blocks = self
            .blocks
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let cell = blocks
            .entry(index)
            .or_insert_with(|| Box::new(AtomicU64::new(0)));
        bump(cell, word);
    }

    fn learn_name(&self, slot: usize, name: &str) {
        // The common case, and the only one on the hot path: one relaxed load
        // and a bit test, no lock at all.
        let bit = if slot < 64 { 1u64 << slot } else { 0 };
        if bit != 0 && self.known_names.load(Ordering::Relaxed) & bit != 0 {
            return;
        }
        {
            let names = self
                .names
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if names.contains_key(&slot) {
                self.known_names.fetch_or(bit, Ordering::Relaxed);
                return;
            }
        }
        self.names
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(slot)
            .or_insert_with(|| name.to_string());
        self.known_names.fetch_or(bit, Ordering::Relaxed);
    }

    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    /// The most recent state of one block, or `None` if it has not been touched.
    pub fn latest(&self, index: [usize; 3]) -> Option<BlockProgress> {
        let word = {
            let blocks = self
                .blocks
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            blocks.get(&index)?.load(Ordering::Acquire)
        };
        Some(self.decode(index, word))
    }

    /// Every block seen so far, sorted by index. See the type's consistency
    /// note: each entry is exact, the set is not one instant.
    pub fn snapshot(&self) -> Vec<BlockProgress> {
        let raw: Vec<([usize; 3], u64)> = {
            let blocks = self
                .blocks
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            blocks
                .iter()
                .map(|(index, cell)| (*index, cell.load(Ordering::Acquire)))
                .collect()
        };
        let mut out: Vec<BlockProgress> = raw
            .into_iter()
            .map(|(index, word)| self.decode(index, word))
            .collect();
        out.sort_by_key(|progress| progress.index);
        out
    }

    /// How many distinct blocks have been touched.
    pub fn blocks_seen(&self) -> usize {
        self.blocks
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// The last op applied to each block, as `(slot, name)`. This is the same
    /// quantity as the tail of `ExecutionLog::op_sequence_per_block`, arrived at
    /// without keeping any history — the two agreeing is the cross-check that
    /// both listeners saw the same run.
    pub fn last_op_per_block(&self) -> std::collections::BTreeMap<[usize; 3], (usize, String)> {
        self.snapshot()
            .into_iter()
            .filter_map(|progress| {
                let slot = progress.slot?;
                Some((progress.index, (slot, progress.op?)))
            })
            .collect()
    }

    fn decode(&self, index: [usize; 3], word: u64) -> BlockProgress {
        let (seq, kind, phase, slot) = unpack(word);
        let op = slot.and_then(|slot| {
            self.names
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&slot)
                .cloned()
        });
        BlockProgress {
            index,
            phase,
            kind,
            slot,
            op,
            seq,
        }
    }
}

/// Monotone store, with the op slot **sticky**.
///
/// Two things live in the word and they age differently. `kind` is the last
/// transition — read, op, write — and is replaced every time. The op slot is
/// the last op that *landed*, and a subsequent write or a read for the next
/// phase must not erase it: a visualiser asking "how far has this block got"
/// wants "threshold, and written", not "written, op unknown". So an event that
/// names no slot carries the previous one forward.
///
/// Monotone because a block is handled by one task at a time — the DAG's
/// dependencies serialise a block's phases — but asserted here rather than
/// assumed, so that "a poll never sees an impossible state" is a property of
/// the type rather than a consequence of the scheduler, which is advisory.
fn bump(cell: &AtomicU64, word: u64) {
    let mut current = cell.load(Ordering::Relaxed);
    loop {
        if current >= word {
            return;
        }
        let next = if word & SLOT_MASK == 0 {
            word | (current & SLOT_MASK)
        } else {
            word
        };
        match cell.compare_exchange_weak(current, next, Ordering::Release, Ordering::Relaxed) {
            Ok(_) => return,
            Err(seen) => current = seen,
        }
    }
}

const SLOT_MASK: u64 = 0xFFFF;

impl EventListener for LatestOpPerChunk {
    fn on_event(&self, event: &Event) {
        let seq = self.next_seq();
        match event {
            Event::PhaseStarted { .. } => {}
            // The IO-layer and materialisation events carry no per-block state
            // this listener is about — they are for the statistics listener.
            // Ignoring them here keeps a poll's meaning exactly "the last op
            // that landed on this block".
            Event::RegionRead { .. } | Event::RegionWritten { .. } | Event::Materialised { .. } => {
            }
            // The cache and prefetch events are keyed by `(array, chunk)`, not
            // by block. There is no block index to store, and inventing one
            // from the chunk would be wrong: one chunk serves several blocks,
            // which is the reason the cache is worth having.
            Event::CacheHit { .. }
            | Event::CacheMiss { .. }
            | Event::CacheEvicted { .. }
            | Event::CacheRefused { .. }
            | Event::CacheKnownEmpty { .. }
            | Event::PrefetchIssued { .. }
            | Event::PrefetchUsed { .. }
            | Event::PrefetchWasted { .. } => {}
            // Sidecar events *do* carry a block index, and are still ignored
            // here: this listener answers "the last op that landed on this
            // block", and a fragment is not an op. Recording one would move a
            // block's progress state without any op having been applied, which
            // is exactly the kind of plausible-but-wrong reading the view is
            // supposed not to give. `SidecarDiscarded` names no block at all.
            Event::SidecarWritten { .. }
            | Event::SidecarRead { .. }
            | Event::SidecarDiscarded { .. } => {}
            // A side output is written by a block, and is still not an op
            // landing on it: the same argument as the fragment events above,
            // and the same answer. The op that produced it emits `OpApplied`
            // in the same breath, which is what moves the block's state.
            Event::SideOutputWritten { .. } => {}
            Event::TaskAdmitted { phase, index } => {
                self.store(*index, pack(seq, ProgressKind::Admitted, *phase, None));
            }
            Event::BlockRead { phase, index, .. } => {
                self.store(*index, pack(seq, ProgressKind::Read, *phase, None));
            }
            Event::OpApplied {
                phase,
                index,
                slot,
                op,
                ..
            } => {
                self.learn_name(*slot, op);
                self.store(
                    *index,
                    pack(seq, ProgressKind::OpApplied, *phase, Some(*slot)),
                );
            }
            Event::BlockShortCircuited {
                phase,
                index,
                slots,
                names,
                ..
            } => {
                for (slot, name) in slots.iter().zip(names) {
                    self.learn_name(*slot, name);
                }
                // The last skipped slot is the state the block is in: the short
                // circuit is defined to leave exactly what computing the whole
                // run of slots would have left.
                let slot = slots.last().copied();
                self.store(
                    *index,
                    pack(seq, ProgressKind::ShortCircuited, *phase, slot),
                );
            }
            Event::BlockWritten { phase, index, .. } => {
                self.store(*index, pack(seq, ProgressKind::Written, *phase, None));
            }
        }
    }
}

// ------------------------------------------------------------ overhead --

/// What dispatch costs, per event, at 0/1/2 listeners.
///
/// The workload is a **mixed** one — scheduling, IO and op events in the
/// proportions a real task produces (one admit, one region read, one block
/// read, five ops, one region write, one block write) — because an op-only
/// measurement understates the cost: the IO variants carry a `String` and so
/// cost an allocation to construct, and that construction is on the same path.
///
/// Reported by the `#[ignore]`d test below rather than asserted: absolute
/// nanoseconds are machine-dependent and a threshold on them would be a flaky
/// test, but the *shape* of the number is the thing worth knowing before
/// putting a listener on a 60 000-task run.
pub fn dispatch_overhead_report(events: usize) -> String {
    use crate::region::Region;
    use std::time::Instant;

    struct Counting(AtomicU64);
    impl EventListener for Counting {
        fn on_event(&self, _event: &Event) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    // One task's worth of events, in the order a task emits them.
    let region = Region::new(&[0, 0, 0], &[64, 64, 64]);
    let one_task = |index: [usize; 3]| -> Vec<Event> {
        let mut out = vec![
            Event::TaskAdmitted { phase: 0, index },
            Event::RegionRead {
                source: "level 0".to_string(),
                image: 0,
                index: Some(index),
                region: region.clone(),
                voxels: 262_144,
                bytes: 2_097_152,
                chunks: 8,
                duration_ns: 1_000,
            },
            Event::BlockRead {
                phase: 0,
                index,
                region: region.clone(),
                voxels: 262_144,
                chunks: 8,
            },
        ];
        for slot in 0..5 {
            out.push(Event::OpApplied {
                phase: 0,
                index,
                slot,
                op: "median".to_string(),
                over: region.clone(),
                duration_ns: 5_000,
            });
        }
        out.push(Event::RegionWritten {
            sink: "level 1".to_string(),
            image: 1,
            index: Some(index),
            region: region.clone(),
            voxels: 262_144,
            bytes: 2_097_152,
            chunks: 8,
            duration_ns: 1_000,
        });
        out.push(Event::BlockWritten {
            phase: 0,
            index,
            valid: region.clone(),
            materialised: true,
        });
        out
    };
    // Pre-built so the measurement is dispatch, not event construction.
    let batch: Vec<Event> = (0..64).flat_map(|i| one_task([i, 0, 0])).collect();
    let mut cursor = 0usize;
    let event = move || {
        let out = batch[cursor % batch.len()].clone();
        cursor += 1;
        out
    };

    // Best of three. The single-run figure is dominated by allocator noise and
    // by the order log's `Vec` doubling, both of which are the *executor's*
    // cost, not dispatch's; taking the minimum stops that noise from swamping
    // a difference of a few nanoseconds.
    let best_of_three = |mut run: Box<dyn FnMut() -> f64>| {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            best = best.min(run());
        }
        best
    };

    let mut lines = vec![format!(
        "dispatch overhead over {events} mixed events (scheduling + IO + op), best of 3"
    )];

    // What it costs merely to *build* the events, with no dispatch at all. The
    // IO variants carry a `String`, so construction allocates; without this
    // line the 0-listener figure looks like the cost of the event log when it
    // is mostly the cost of the events.
    let construction = best_of_three(Box::new({
        let mut event = event.clone();
        move || {
            let start = Instant::now();
            for _ in 0..events {
                std::hint::black_box(event());
            }
            start.elapsed().as_secs_f64() * 1e9 / events as f64
        }
    }));
    lines.push(format!(
        "  {:<44} {construction:>8.1} ns/event",
        "event construction only (no dispatch)"
    ));
    let counting: Arc<dyn EventListener> = Arc::new(Counting(AtomicU64::new(0)));
    let latest: Arc<dyn EventListener> = Arc::new(LatestOpPerChunk::new());
    let mut baseline = 0.0f64;
    for (label, listeners) in [
        ("0 listeners (built-in order log only)", vec![]),
        ("1 listener  (+ counting)", vec![Arc::clone(&counting)]),
        (
            "2 listeners (+ counting + LatestOpPerChunk)",
            vec![Arc::clone(&counting), Arc::clone(&latest)],
        ),
    ] {
        let per_event = best_of_three(Box::new({
            let mut event = event.clone();
            let listeners = listeners.clone();
            move || {
                let dispatch = Dispatch::new(&listeners);
                let start = Instant::now();
                for _ in 0..events {
                    dispatch.emit(event());
                }
                start.elapsed().as_secs_f64() * 1e9 / events as f64
            }
        }));
        if listeners.is_empty() {
            baseline = per_event;
        }
        lines.push(format!(
            "  {label:<44} {per_event:>8.1} ns/event   ({:+6.1} ns vs 0 listeners)",
            per_event - baseline,
        ));
    }
    lines.push(
        "  the 0-listener figure is event construction + the built-in order log's push;\n  \
         the delta is what registering a listener costs."
            .to_string(),
    );
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::Region;

    fn applied(index: [usize; 3], phase: usize, slot: usize, name: &str) -> Event {
        Event::OpApplied {
            phase,
            index,
            slot,
            op: name.to_string(),
            over: Region::new(&[0, 0, 0], &[1, 1, 1]),
            duration_ns: 0,
        }
    }

    #[test]
    fn packing_round_trips() {
        for (seq, kind, phase, slot) in [
            (0u64, ProgressKind::Read, 0usize, None),
            (7, ProgressKind::OpApplied, 3, Some(0)),
            (u32::MAX as u64, ProgressKind::Written, 4095, Some(65533)),
            (12345, ProgressKind::ShortCircuited, 11, Some(4)),
        ] {
            assert_eq!(
                unpack(pack(seq, kind, phase, slot)),
                (seq, kind, phase, slot)
            );
        }
    }

    #[test]
    fn a_listener_sees_every_event_the_log_sees() {
        #[derive(Default)]
        struct Mirror(std::sync::Mutex<Vec<Event>>);
        impl EventListener for Mirror {
            fn on_event(&self, event: &Event) {
                self.0.lock().unwrap().push(event.clone());
            }
        }
        let mirror = Arc::new(Mirror::default());
        let listeners: Vec<Arc<dyn EventListener>> = vec![mirror.clone()];
        let dispatch = Dispatch::new(&listeners);
        for slot in 0..4 {
            dispatch.emit(applied([0, 0, 0], 0, slot, "op"));
        }
        assert_eq!(dispatch.faults(), 0);
        assert_eq!(
            mirror.0.lock().unwrap().clone(),
            dispatch.into_log().events()
        );
    }

    #[test]
    fn a_panicking_listener_is_disabled_and_the_rest_still_see_everything() {
        struct Exploding;
        impl EventListener for Exploding {
            fn on_event(&self, _event: &Event) {
                panic!("a listener misbehaving");
            }
        }
        let latest = Arc::new(LatestOpPerChunk::new());
        let listeners: Vec<Arc<dyn EventListener>> = vec![
            Arc::new(Exploding),
            latest.clone() as Arc<dyn EventListener>,
        ];

        // Silence the default hook for the duration: the panic is the point of
        // the test, and its backtrace is not.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let dispatch = Dispatch::new(&listeners);
        for slot in 0..3 {
            dispatch.emit(applied([1, 1, 1], 0, slot, "op"));
        }
        std::panic::set_hook(previous);

        // One fault, not three: the listener is disabled after its first panic.
        assert_eq!(dispatch.faults(), 1);
        // and the run's own record is complete
        assert_eq!(dispatch.into_log().events().len(), 3);
        // as is the other listener's
        assert_eq!(
            latest.latest([1, 1, 1]).unwrap().slot,
            Some(2),
            "a well-behaved listener must not be affected by a broken one"
        );
    }

    #[test]
    fn latest_op_per_chunk_tracks_the_most_recent_op() {
        let latest = LatestOpPerChunk::new();
        latest.on_event(&Event::BlockRead {
            phase: 0,
            index: [0, 0, 0],
            region: Region::new(&[0, 0, 0], &[8, 8, 8]),
            voxels: 512,
            chunks: 1,
        });
        assert_eq!(latest.latest([0, 0, 0]).unwrap().kind, ProgressKind::Read);
        latest.on_event(&applied([0, 0, 0], 0, 0, "median"));
        latest.on_event(&applied([0, 0, 0], 0, 1, "deconvolve"));
        let state = latest.latest([0, 0, 0]).unwrap();
        assert_eq!(state.kind, ProgressKind::OpApplied);
        assert_eq!(state.slot, Some(1));
        assert_eq!(state.op.as_deref(), Some("deconvolve"));
        assert_eq!(latest.blocks_seen(), 1);
        assert!(latest.latest([9, 9, 9]).is_none());
    }

    #[test]
    fn a_short_circuited_block_reports_the_last_skipped_slot() {
        let latest = LatestOpPerChunk::new();
        latest.on_event(&Event::BlockShortCircuited {
            phase: 1,
            index: [2, 0, 0],
            from: 0.0,
            to: 0.0,
            slots: vec![3, 4],
            names: vec!["a".to_string(), "b".to_string()],
        });
        let state = latest.latest([2, 0, 0]).unwrap();
        assert_eq!(state.kind, ProgressKind::ShortCircuited);
        assert_eq!(state.slot, Some(4));
        assert_eq!(state.op.as_deref(), Some("b"));
        assert_eq!(state.phase, 1);
    }

    /// The write path is monotone even if events arrive out of order, so a
    /// poll can never see a block go backwards.
    #[test]
    fn state_never_moves_backwards() {
        let latest = LatestOpPerChunk::new();
        latest.on_event(&applied([0, 0, 0], 0, 0, "a"));
        let after_first = latest.latest([0, 0, 0]).unwrap().seq;
        // Replay an older event by hand, as a racing thread would.
        latest.store([0, 0, 0], pack(0, ProgressKind::Read, 0, None));
        let state = latest.latest([0, 0, 0]).unwrap();
        assert_eq!(state.seq, after_first);
        assert_eq!(state.kind, ProgressKind::OpApplied);
    }

    /// A poll concurrent with a flood of events must return promptly and must
    /// only ever see states that were really emitted.
    #[test]
    fn polling_during_a_run_never_blocks_and_never_sees_an_impossible_state() {
        use std::sync::atomic::AtomicBool;
        use std::thread;

        let latest = Arc::new(LatestOpPerChunk::new());
        let running = Arc::new(AtomicBool::new(true));
        let names = ["a", "b", "c", "d"];

        let writers: Vec<_> = (0..4)
            .map(|worker| {
                let latest = Arc::clone(&latest);
                thread::spawn(move || {
                    for round in 0..2_000 {
                        let index = [worker, round % 8, 0];
                        latest.on_event(&Event::BlockRead {
                            phase: 0,
                            index,
                            region: Region::new(&[0, 0, 0], &[4, 4, 4]),
                            voxels: 64,
                            chunks: 1,
                        });
                        for (slot, name) in names.iter().enumerate() {
                            latest.on_event(&applied(index, 0, slot, name));
                        }
                    }
                })
            })
            .collect();

        let mut polls = 0usize;
        let mut worst = std::time::Duration::ZERO;
        while running.load(Ordering::Relaxed) && polls < 500 {
            let start = std::time::Instant::now();
            let snapshot = latest.snapshot();
            worst = worst.max(start.elapsed());
            polls += 1;
            for state in snapshot {
                // Every state observed must be one the writers actually emit.
                match state.kind {
                    // A read may carry the previous round's op forward: the
                    // slot is sticky, the kind is not.
                    ProgressKind::Read | ProgressKind::OpApplied => {
                        if let Some(slot) = state.slot {
                            assert!(slot < names.len());
                            assert_eq!(state.op.as_deref(), Some(names[slot]));
                        }
                    }
                    other => panic!("the writers never emit {other:?}"),
                }
                assert_eq!(state.phase, 0);
                assert!(state.index[0] < 4 && state.index[1] < 8);
            }
            if writers.iter().all(|handle| handle.is_finished()) {
                running.store(false, Ordering::Relaxed);
            }
        }
        for handle in writers {
            handle.join().unwrap();
        }

        assert!(polls > 1, "the poll loop never got going");
        // Not a timing assertion on the machine, an assertion that a poll is
        // not *waiting* on the run: 100 ms would mean it had blocked behind
        // thousands of writes.
        assert!(worst < std::time::Duration::from_millis(100), "{worst:?}");
        assert_eq!(latest.blocks_seen(), 32);
        for state in latest.snapshot() {
            assert_eq!(state.slot, Some(3), "every block ends on the last op");
        }
    }

    #[test]
    #[ignore = "prints a measurement; run with --ignored --nocapture"]
    fn report_dispatch_overhead() {
        println!("{}", dispatch_overhead_report(200_000));
    }
}
