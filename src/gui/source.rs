// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The one shape a live run and a recorded one both answer.
//
// Everything in this file is deliberately small and owned. A source hands back
// plain data, never a borrow into its own state and never a lock guard: the
// HTTP layer serialises on its own thread and must not be able to hold anything
// that a running executor might want. `State` is a `Vec` of at most one entry
// per block, which for a 60 000-block run is a few megabytes serialised — large
// enough to notice, small enough that a poll is a memcpy rather than an event.

use crate::listener::BlockProgress;
use crate::log::Event;

/// Where a view's data is coming from. **Informational only** — the browser
/// shows it in the corner and branches on nothing.
///
/// The thing the client actually branches on is [`Meta::controllable`], which
/// says whether the transport controls do anything. Keeping the two separate is
/// what stops the client from growing a second code path: a live run that
/// someone later teaches to pause would flip `controllable` and need no change
/// in the browser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// An execution that is happening now.
    Live,
    /// A recorded execution being re-emitted.
    Replay,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Live => "live",
            Mode::Replay => "replay",
        }
    }
}

/// What does not change while a run is being watched.
///
/// Fetched once by the client. Everything here is either known before the first
/// task is admitted (the grid, the phases, the chain) or is a property of the
/// recording (`total_events`), so a client that caches it for the lifetime of
/// its connection is correct.
#[derive(Debug, Clone)]
pub struct Meta {
    pub mode: Mode,
    /// Whether [`ProgressSource::control`] does anything. False for a live run,
    /// whose playhead is the execution itself.
    pub controllable: bool,
    /// The strategy's name, as it was given to the executor.
    pub strategy: String,
    /// Voxels per axis, in the volume's own axis order — axis 0 first, no
    /// transposition, the same convention the exported order log uses.
    pub volume: [usize; 3],
    /// Blocks per axis. The grid the view draws.
    pub grid: [usize; 3],
    pub phases: usize,
    /// `(slot, name)` in chain order. The legend, and the colour scale: a
    /// block's progress is its slot's position in this list.
    pub ops: Vec<(usize, String)>,
    /// How many events the recording holds. `None` for a live run, where the
    /// total is not known until it ends — which is exactly the difference
    /// between a scrub bar and a progress spinner, and the client draws it that
    /// way.
    pub total_events: Option<u64>,
}

/// What a block is doing, now or at the playhead.
///
/// [`BlockProgress`] is reused rather than restated: it is already the answer
/// to "how far has this block got", it is already what a live poll produces,
/// and a second type meaning the same thing is how the two modes would start to
/// disagree.
#[derive(Debug, Clone)]
pub struct State {
    /// How many events the playhead has consumed. Monotone within a
    /// connection for a live run; moves either way for a replay.
    pub cursor: u64,
    /// Total events, when known. Mirrors [`Meta::total_events`], repeated here
    /// so a client polling `/api/state` can size its scrub bar without
    /// re-fetching the metadata — a live run's total becomes known when it
    /// finishes.
    pub total: Option<u64>,
    /// The largest emission sequence in `blocks`. A client comparing this with
    /// `cursor` can see how stale the view is; see `LatestOpPerChunk`'s
    /// consistency note for why a snapshot is per-block exact but not a global
    /// instant.
    pub seq: u64,
    /// Whether the playhead is advancing by itself.
    pub running: bool,
    /// One entry per block that has been touched, sorted by index.
    pub blocks: Vec<BlockProgress>,
}

/// One line of the timeline.
///
/// A **projection**, not the wire format: the exported order log
/// (`export::write_order_log_json`) stays the full-fidelity record, and anything
/// analysing a run offline should read that. What the browser needs is a
/// scrollable list of what happened, at a few hundred bytes per entry rather
/// than a few thousand, because it fetches them while a run is going on.
#[derive(Debug, Clone, PartialEq)]
pub struct TimelineEvent {
    pub seq: u64,
    /// The event's type string, identical to the exported order log's — the
    /// two vocabularies must not fork.
    pub kind: &'static str,
    pub phase: Option<usize>,
    pub index: Option<[usize; 3]>,
    pub slot: Option<usize>,
    pub op: Option<String>,
    /// Nanoseconds, for the events that measure something.
    pub duration_ns: Option<u64>,
}

/// Which part of the timeline a client is asking for.
///
/// Two, because there are two honest questions and neither answers the other:
///
/// * `Since` walks the stream forwards, for a consumer accumulating the whole
///   thing — a script, or a page that wants every event once.
/// * `Before` is a window **ending at a point**, which is what a view wants: on
///   a live run the point is now, on a replay it is the playhead, and the same
///   request expresses both. Without it a replay's timeline would show the end
///   of the recording while the grid showed the middle.
///
/// Both are bounded work. That is the property that matters: a client cannot
/// ask a question whose answer costs the executor anything unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    Since(u64),
    Before(u64),
}

/// A page of the timeline.
#[derive(Debug, Clone)]
pub struct TimelinePage {
    /// The lowest `seq` this page could contain — the request's own bound.
    pub since: u64,
    /// What to ask for next. Equal to `since` when there is nothing more yet,
    /// which is how a live client knows to wait rather than to spin.
    pub next: u64,
    /// How many events are available to page through **right now**. This is
    /// not [`Meta::total_events`] — for a live run it grows, and it is the
    /// bound on `since`, not the length of the recording.
    pub available: u64,
    pub events: Vec<TimelineEvent>,
}

/// One event, projected onto the timeline.
///
/// `seq` is the event's position in the stream and is **not** renumbered when
/// an event is filtered out, so `since`/`next` paging stays aligned with the
/// playhead's cursor whatever this function chooses to drop.
///
/// Returns `None` for the cache and prefetch layers. Those are emitted per
/// *chunk* rather than per caller-level call — one block read may be five
/// cache events — so including them would make the timeline a list of chunk
/// traffic with the schedule buried in it, and would multiply the bytes a poll
/// moves by roughly the chunks-per-block ratio. They are in the exported order
/// log, which is where a question about cache behaviour should be asked.
pub fn project(seq: u64, event: &Event) -> Option<TimelineEvent> {
    let base = |kind: &'static str| TimelineEvent {
        seq,
        kind,
        phase: None,
        index: None,
        slot: None,
        op: None,
        duration_ns: None,
    };
    Some(match event {
        Event::PhaseStarted { phase } => TimelineEvent {
            phase: Some(*phase),
            ..base("phase_started")
        },
        Event::TaskAdmitted { phase, index } => TimelineEvent {
            phase: Some(*phase),
            index: Some(*index),
            ..base("task_admitted")
        },
        Event::RegionRead {
            level,
            index,
            duration_ns,
            ..
        } => TimelineEvent {
            phase: Some(*level),
            index: *index,
            duration_ns: Some(*duration_ns),
            ..base("region_read")
        },
        Event::RegionWritten {
            level,
            index,
            duration_ns,
            ..
        } => TimelineEvent {
            phase: Some(*level),
            index: *index,
            duration_ns: Some(*duration_ns),
            ..base("region_written")
        },
        Event::Materialised { phase, .. } => TimelineEvent {
            phase: Some(*phase),
            ..base("materialised")
        },
        Event::BlockRead { phase, index, .. } => TimelineEvent {
            phase: Some(*phase),
            index: Some(*index),
            ..base("block_read")
        },
        Event::OpApplied {
            phase,
            index,
            slot,
            op,
            duration_ns,
            ..
        } => TimelineEvent {
            phase: Some(*phase),
            index: Some(*index),
            slot: Some(*slot),
            op: Some(op.clone()),
            duration_ns: Some(*duration_ns),
            ..base("op_applied")
        },
        Event::BlockShortCircuited {
            phase,
            index,
            slots,
            names,
            ..
        } => TimelineEvent {
            phase: Some(*phase),
            index: Some(*index),
            slot: slots.last().copied(),
            op: names.last().cloned(),
            ..base("block_short_circuited")
        },
        Event::BlockWritten { phase, index, .. } => TimelineEvent {
            phase: Some(*phase),
            index: Some(*index),
            ..base("block_written")
        },
        _ => return None,
    })
}

/// Move a replay's playhead.
///
/// A live source answers `false` to all of these and does nothing — there is no
/// meaningful "pause" for work that is happening, and pretending otherwise
/// would put the browser in a state the run does not share.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Control {
    Play,
    Pause,
    /// Absolute position, in events, clamped to the recording.
    Seek(u64),
    /// Relative move, in events. Negative goes back.
    Step(i64),
    /// Events per second while playing. Clamped to something a browser can
    /// draw.
    Speed(f64),
}

/// Something a view can be pointed at.
///
/// Implemented twice — once over a running executor, once over a recording —
/// and the HTTP layer knows nothing about which it holds. That is the whole
/// design: if a third source ever appears (a log streamed from another node,
/// say) it implements this and the browser needs no change.
///
/// Implementations must be cheap enough to call several times a second and must
/// never block on anything the executor holds.
pub trait ProgressSource: Send + Sync {
    fn meta(&self) -> Meta;

    /// The playhead's view of every block.
    fn state(&self) -> State;

    /// At most `limit` timeline entries from the requested window.
    fn timeline(&self, window: Window, limit: usize) -> TimelinePage;

    /// Move the playhead. Returns whether the command was acted on, so the HTTP
    /// layer can answer 409 rather than pretending.
    fn control(&self, command: Control) -> bool {
        let _ = command;
        false
    }
}
