// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A run that is happening, seen through the listeners it already has.
//
// There is almost no new machinery here and that is the point.
// `LatestOpPerChunk` was built to be polled while a run is in flight — one
// atomic word per block, a shared read guard, per-block exact and causally
// plausible across blocks — and this type does little but put an HTTP shape on
// it. If a live view ever needs something `LatestOpPerChunk` cannot answer, the
// change belongs there, behind the guarantees documented there, and not in a
// second progress tracker that the executor would then have to feed twice.
//
// The two listeners
// -----------------
// * **Progress** (`LatestOpPerChunk`) — required. Bounded memory, no history,
//   answers "where is every block now". This is what the grid draws.
// * **Timeline** ([`TimelineListener`]) — optional. This one *is* new, and the
//   reason it is not `OrderLog` is measured rather than aesthetic: reading an
//   `OrderLog` means `events()`, which clones the entire history **under the
//   same mutex every worker pushes to**. On a 60 000-event run that is tens of
//   milliseconds during which no worker can emit — a viewer throttling the run
//   it is watching, which is exactly the thing this whole module is forbidden
//   to do. Projecting each event as it arrives and keeping only the projection
//   makes a read cost one bounded page instead of one unbounded clone, and
//   costs the write path a `push` it was already paying for.
//
// Two things a client cannot do
// -----------------------------
// Both of the costs a viewer can impose on a run are **bounded at the source**,
// not by asking clients to behave:
//
// * **Polling harder buys nothing.** A snapshot walks every block and takes the
//   progress listener's guard, so an unthrottled client is real load on the
//   executor's data structures. [`LiveSource::with_min_snapshot_interval`] puts a floor
//   under how often one is actually taken; polls inside the floor are served a
//   copy. Any number of clients at any rate therefore cost the run at most
//   twenty snapshots a second, and the page they feed asks for five.
// * **Asking for the whole timeline is not a question that can be posed.** The
//   endpoint takes a window and a limit, both bounded; there is no request that
//   makes the executor's mutex be held for longer than a page.
//
// The measurement is `gui::tests::report_what_watching_costs`, which reports a
// bare run, a run with the server attached, and a run with a client polling as
// hard as it can.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::export::ExportMeta;
use crate::listener::{EventListener, LatestOpPerChunk};
use crate::log::Event;

use super::source::{
    project, Control, Meta, Mode, ProgressSource, State, TimelineEvent, TimelinePage, Window,
};

/// Keeps the event stream as timeline entries, projected on arrival.
///
/// # Ordering
///
/// `seq` is assigned before projection, so it is the event's position in the
/// whole stream and matches what a `LatestOpPerChunk` snapshot reports. The
/// *stored* order is push order, which under concurrency may differ from `seq`
/// order by however many events two workers can emit between one taking the
/// sequence and the other taking the lock — in practice a handful.
///
/// Paging therefore treats the vector as sorted by `seq` and binary-searches
/// it, which is correct to within that handful. A timeline is a list of what
/// happened, not evidence in an argument about order; the global order is a
/// scheduling artefact in the first place, as `listener` says. What the binary
/// search buys is that a page costs a search and a copy of `limit` entries
/// rather than a scan of the history, which is what keeps a poll off the
/// executor's back.
#[derive(Debug, Default)]
pub struct TimelineListener {
    seq: AtomicU64,
    entries: Mutex<Vec<TimelineEvent>>,
}

impl TimelineListener {
    pub fn new() -> Self {
        Self::default()
    }

    /// Events seen, including those the projection dropped.
    pub fn seen(&self) -> u64 {
        self.seq.load(Ordering::Relaxed)
    }

    fn page(&self, window: Window, limit: usize) -> TimelinePage {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let position = |bound: u64| entries.partition_point(|entry| entry.seq < bound);
        let (from, to) = match window {
            Window::Since(since) => {
                let from = position(since);
                (from, (from + limit).min(entries.len()))
            }
            Window::Before(before) => {
                let to = position(before);
                (to.saturating_sub(limit), to)
            }
        };
        let events = entries[from..to].to_vec();
        let next = events
            .last()
            .map(|entry| entry.seq + 1)
            .unwrap_or(match window {
                Window::Since(since) => since,
                Window::Before(before) => before,
            });
        TimelinePage {
            since: events.first().map(|entry| entry.seq).unwrap_or(0),
            next,
            available: self.seen(),
            events,
        }
    }
}

impl EventListener for TimelineListener {
    fn on_event(&self, event: &Event) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        if let Some(entry) = project(seq, event) {
            self.entries
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(entry);
        }
    }
}

/// How often a snapshot may actually be taken, however hard clients poll.
///
/// 20 Hz. Four times the rate the page asks at, so the floor is invisible to
/// it, and a hard ceiling on what any number of clients can cost the run.
pub const DEFAULT_MIN_SNAPSHOT_INTERVAL: Duration = Duration::from_millis(50);

/// A view onto an execution in progress.
///
/// Construct it *before* the run, register [`LiveSource::listeners`] with
/// `execute_observed`, and hand a clone to [`super::serve`]. The whole
/// attachment is five lines:
///
/// ```no_run
/// # use std::sync::Arc;
/// # use blockflow::export::ExportMeta;
/// # use blockflow::gui::{serve, LiveSource, Options};
/// let meta = ExportMeta::new("greedy", [512, 512, 512], 2);
/// let live = Arc::new(LiveSource::new(meta, [8, 8, 8]).with_timeline());
/// let server = serve(live.clone(), Options::default())?;
/// // ... execute_observed(name, &workflow, &plan, &hints, &env, &live.listeners())?;
/// live.finished();
/// # Ok::<(), blockflow::Error>(())
/// ```
pub struct LiveSource {
    meta: ExportMeta,
    grid: [usize; 3],
    progress: Arc<LatestOpPerChunk>,
    timeline: Option<Arc<TimelineListener>>,
    running: AtomicBool,
    /// Set by [`LiveSource::finished`]. Zero means "still unknown", which is
    /// distinguishable from "a run with no events" only in that `running` is
    /// then false as well — and a run with no events has nothing to look at.
    total: AtomicU64,
    min_interval: Duration,
    /// The last snapshot and when it was taken. See the module header: this is
    /// what stops a client's poll rate from being the executor's problem.
    cached: Mutex<Option<(Instant, Arc<State>)>>,
}

impl LiveSource {
    /// `grid` is blocks per axis. It is taken rather than derived from the
    /// events because the view wants to draw the whole lattice from the first
    /// poll, including the blocks nothing has touched yet — a grid inferred
    /// from what has been seen would grow as the run went on, and a picture
    /// that changes size while you watch it is unreadable.
    pub fn new(meta: ExportMeta, grid: [usize; 3]) -> Self {
        Self {
            meta,
            grid,
            progress: Arc::new(LatestOpPerChunk::new()),
            timeline: None,
            running: AtomicBool::new(true),
            total: AtomicU64::new(0),
            min_interval: DEFAULT_MIN_SNAPSHOT_INTERVAL,
            cached: Mutex::new(None),
        }
    }

    /// Also keep a timeline. See the module header for what that costs.
    pub fn with_timeline(mut self) -> Self {
        self.timeline = Some(Arc::new(TimelineListener::new()));
        self
    }

    /// Use a progress listener the caller already has, rather than this type's
    /// own. For a caller that was already polling `LatestOpPerChunk` for its
    /// own reasons and does not want the events fanned out twice.
    pub fn with_progress(mut self, progress: Arc<LatestOpPerChunk>) -> Self {
        self.progress = progress;
        self
    }

    /// Change the floor on how often a snapshot is taken. `Duration::ZERO`
    /// removes it, which is only sensible when nothing is polling from outside
    /// the process.
    pub fn with_min_snapshot_interval(mut self, interval: Duration) -> Self {
        self.min_interval = interval;
        self
    }

    /// What to hand `execute_observed`.
    pub fn listeners(&self) -> Vec<Arc<dyn EventListener>> {
        let mut out: Vec<Arc<dyn EventListener>> = vec![self.progress.clone()];
        if let Some(timeline) = &self.timeline {
            out.push(timeline.clone());
        }
        out
    }

    pub fn progress(&self) -> &Arc<LatestOpPerChunk> {
        &self.progress
    }

    /// The run is over. The view stops saying "running" and, if a timeline was
    /// kept, learns the total so the scrub bar becomes exact.
    ///
    /// Forgetting to call this is a cosmetic fault, not a correctness one: the
    /// view keeps polling a state that has stopped changing.
    pub fn finished(&self) {
        if let Some(timeline) = &self.timeline {
            self.total.store(timeline.seen(), Ordering::Relaxed);
        }
        self.running.store(false, Ordering::Release);
        // Drop the cache so the next poll is the final state rather than one up
        // to `min_interval` old — the last frame is the one somebody reads.
        *self
            .cached
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    fn total_events(&self) -> Option<u64> {
        match self.total.load(Ordering::Relaxed) {
            0 => None,
            total => Some(total),
        }
    }

    fn take_snapshot(&self) -> State {
        let blocks = self.progress.snapshot();
        let seq = blocks.iter().map(|block| block.seq).max().unwrap_or(0);
        // With a timeline attached the cursor is exact — it is how many events
        // have been emitted. Without one the best available answer is the
        // highest emission sequence any block carries, which is a lower bound
        // (the packed word saturates at `u32::MAX`, and events that touch no
        // block do not appear at all). Stated rather than hidden: a client uses
        // `cursor` for staleness and for a window, never for arithmetic on a
        // recording.
        let cursor = match &self.timeline {
            Some(timeline) => timeline.seen(),
            None => seq,
        };
        State {
            cursor,
            total: self.total_events(),
            seq,
            running: self.running.load(Ordering::Acquire),
            blocks,
        }
    }
}

impl ProgressSource for LiveSource {
    fn meta(&self) -> Meta {
        Meta {
            mode: Mode::Live,
            // A live run's playhead is the run. There is nothing to seek.
            controllable: false,
            strategy: self.meta.strategy.clone(),
            volume: self.meta.volume,
            grid: self.grid,
            phases: self.meta.phases,
            ops: self.meta.ops.clone(),
            total_events: self.total_events(),
        }
    }

    fn state(&self) -> State {
        if self.min_interval.is_zero() {
            return self.take_snapshot();
        }
        let mut cached = self
            .cached
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((taken, state)) = cached.as_ref() {
            if taken.elapsed() < self.min_interval {
                return State::clone(state);
            }
        }
        // The lock is held across the snapshot on purpose: it makes concurrent
        // pollers share one snapshot rather than each taking their own, which
        // is the whole point. The snapshot never blocks on the executor — it
        // takes the progress listener's *shared* guard — so the worst a poller
        // waits here is one snapshot, and the run waits for none of it.
        let state = Arc::new(self.take_snapshot());
        *cached = Some((Instant::now(), state.clone()));
        State::clone(&state)
    }

    fn timeline(&self, window: Window, limit: usize) -> TimelinePage {
        let Some(timeline) = &self.timeline else {
            // No timeline was asked for. An empty page with nothing to follow
            // is the "nothing more" answer a polling client already handles, so
            // this needs no special case in the browser.
            let at = match window {
                Window::Since(since) => since,
                Window::Before(before) => before,
            };
            return TimelinePage {
                since: at,
                next: at,
                available: 0,
                events: Vec::new(),
            };
        };
        timeline.page(window, limit)
    }

    fn control(&self, _command: Control) -> bool {
        false
    }
}
