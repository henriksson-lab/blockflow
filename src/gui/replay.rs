// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// An exported order log, played back as though it were happening.
//
// The decision this file rests on
// -------------------------------
// A replay could be served by reading progress straight out of the JSON — walk
// the events, keep a little table of "last op per block", answer polls from it.
// That is less code than what is here, and it is the wrong shape, because the
// little table would be a *second implementation* of `LatestOpPerChunk`. The
// two would agree on the day they were written and drift the first time a new
// event variant appeared, and the drift would show up as a replay disagreeing
// with the live view of the same run — the single worst failure this tool could
// have, because the whole point of a replay is to be believed.
//
// So the events are decoded back into `Event` and fed to a real
// `LatestOpPerChunk`, one instance owned by this source. A replay is then
// literally the live path with a recording where the executor would be, and the
// progress semantics — sticky op slot, monotone state, short-circuit counting
// as applied — are inherited rather than restated. The cost is a decoder, which
// is honest work: the schema is a documented cross-language contract and
// something ought to be able to read it back in the language that writes it.
//
// Seeking, and why rewinding is a rebuild
// ---------------------------------------
// `LatestOpPerChunk` is deliberately monotone: a block's state can never move
// backwards, which is what makes a poll during a run safe. That means a
// backwards seek cannot be served by un-applying events, so it is served by
// throwing the fold away and re-folding from zero. Re-folding 40 000 events
// costs a few milliseconds — measured in `gui::tests` — which is invisible next
// to the round trip that asked for it, and it keeps the monotonicity that the
// live path depends on. Paying a rebuild on the rare operation to keep the
// common one exact is the right way round.
//
// The clock
// ---------
// Nothing ticks. The playhead advances *when it is asked for*, by however much
// wall time has passed since the last poll times the current speed. So there is
// no timer thread, no drift between a timer and a poll rate, and a replay
// nobody is watching consumes nothing at all. A client polling at 5 Hz and one
// polling at 60 Hz see the same replay at the same speed, at different
// smoothness — which is exactly the property a live view has, for free.

use std::sync::Mutex;
use std::time::Instant;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::listener::{EventListener, LatestOpPerChunk};
use crate::log::Event;

use super::source::{project, Control, Meta, Mode, ProgressSource, State, TimelinePage, Window};

/// An exported order log, decoded.
#[derive(Debug, Clone)]
pub struct DecodedLog {
    pub strategy: String,
    pub volume: [usize; 3],
    pub grid: [usize; 3],
    pub phases: usize,
    pub ops: Vec<(usize, String)>,
    pub events: Vec<Event>,
    /// Entries whose `type` this decoder does not know. Not an error: the
    /// schema may grow, and a viewer that refuses to open a log because it
    /// contains one unfamiliar line is worse than one that draws the rest and
    /// says how many it skipped.
    pub unknown: usize,
}

/// The schema identifier a v1 document must carry, and the only version this
/// decoder claims to understand.
const SCHEMA: &str = "clearmap-rs.block_ops.order_log";
const VERSION: u64 = 1;

/// The schema's own reader, borrowed rather than reimplemented.
///
/// This module used to carry a decoder of its own, and the reason it no longer
/// does is the same reason it feeds a real `LatestOpPerChunk` instead of a
/// little table: a second implementation of a documented format agrees with the
/// first on the day it is written and drifts at the first new event variant.
/// `export` writes the schema, so `export` reads it.
use crate::export::event_from_json as decode_event;

/// Where the document-level fields still need reading. Only the header uses
/// these; every per-event field is `export`'s business.
fn triple(value: &Value, name: &str, at: usize) -> Result<[usize; 3]> {
    let array = value
        .as_array()
        .ok_or_else(|| Error::invalid(format!("event {at}: {name:?} is not an array")))?;
    if array.len() != 3 {
        return Err(Error::invalid(format!(
            "event {at}: {name:?} has {} entries, and this crate is three-dimensional",
            array.len()
        )));
    }
    let mut out = [0usize; 3];
    for (axis, entry) in array.iter().enumerate() {
        out[axis] = entry.as_u64().ok_or_else(|| {
            Error::invalid(format!(
                "event {at}: {name:?}[{axis}] is not a whole number"
            ))
        })? as usize;
    }
    Ok(out)
}

/// Read a v1 order-log document back into events.
///
/// Strict about the two things a consumer must not guess at — the schema
/// identifier and the version — and forgiving about everything else. A
/// document from a future version is refused by name rather than
/// mis-interpreted, because a viewer drawing a wrong picture confidently is
/// worse than one that will not open the file.
pub fn decode_order_log(document: &Value) -> Result<DecodedLog> {
    let schema = document.get("schema").and_then(Value::as_str);
    if schema != Some(SCHEMA) {
        return Err(Error::invalid(format!(
            "not an order log: expected schema {SCHEMA:?}, found {schema:?}"
        )));
    }
    let version = document.get("version").and_then(Value::as_u64);
    if version != Some(VERSION) {
        return Err(Error::invalid(format!(
            "order log version {version:?} is not version {VERSION}, which is what \
             this viewer reads. Re-export with a matching build."
        )));
    }

    let volume = triple(
        document
            .get("volume")
            .ok_or_else(|| Error::invalid("order log: no \"volume\""))?,
        "volume",
        0,
    )?;
    let grid = triple(
        document
            .get("grid")
            .ok_or_else(|| Error::invalid("order log: no \"grid\""))?,
        "grid",
        0,
    )?;
    let phases = document
        .get("phases")
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::invalid("order log: no \"phases\""))? as usize;
    let strategy = document
        .get("strategy")
        .and_then(Value::as_str)
        .unwrap_or("unnamed")
        .to_string();

    let ops = document
        .get("ops")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .enumerate()
                .map(|(position, entry)| {
                    let slot = entry
                        .get("slot")
                        .and_then(Value::as_u64)
                        .map(|value| value as usize)
                        .unwrap_or(position);
                    let name = entry
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("op")
                        .to_string();
                    (slot, name)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let raw = document
        .get("events")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::invalid("order log: no \"events\" array"))?;
    let mut events = Vec::with_capacity(raw.len());
    let mut unknown = 0usize;
    for (at, entry) in raw.iter().enumerate() {
        match decode_event(entry, at)? {
            Some(event) => events.push(event),
            None => unknown += 1,
        }
    }

    Ok(DecodedLog {
        strategy,
        volume,
        grid,
        phases,
        ops,
        events,
        unknown,
    })
}

/// Read a v1 order-log document from disk.
pub fn read_order_log(path: impl AsRef<std::path::Path>) -> Result<DecodedLog> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .map_err(|err| Error::invalid(format!("reading {}: {err}", path.display())))?;
    let document: Value = serde_json::from_str(&text)
        .map_err(|err| Error::invalid(format!("parsing {}: {err}", path.display())))?;
    decode_order_log(&document)
}

/// Where the playhead is and how it is moving.
struct Playhead {
    cursor: u64,
    playing: bool,
    /// Events per second.
    speed: f64,
    /// When the cursor was last advanced. Advancing is lazy, so this is the
    /// only clock in the module.
    since: Instant,
    /// Fractional events carried between polls, so a slow poll rate does not
    /// round the speed down to zero.
    carry: f64,
}

/// The fold, and how far into the recording it has been taken.
struct Fold {
    at: u64,
    progress: LatestOpPerChunk,
}

/// A recorded run, served exactly as a live one is.
pub struct ReplaySource {
    log: DecodedLog,
    playhead: Mutex<Playhead>,
    fold: Mutex<Fold>,
}

/// The default playback rate, chosen so a whole recording takes about this long
/// however big it is. A viewer's first question is "what shape did this run
/// have", and the answer arrives in a sitting at any scale; the speed control
/// is there for the second question.
const DEFAULT_PLAYBACK_SECONDS: f64 = 20.0;
/// Ceiling and floor on the rate, in events per second. The floor stops a tiny
/// recording from crawling; the ceiling stops a huge one from folding more per
/// poll than a browser can draw.
const MIN_SPEED: f64 = 10.0;
const MAX_SPEED: f64 = 20_000.0;

impl ReplaySource {
    pub fn new(log: DecodedLog) -> Self {
        let speed =
            (log.events.len() as f64 / DEFAULT_PLAYBACK_SECONDS).clamp(MIN_SPEED, MAX_SPEED);
        Self {
            log,
            playhead: Mutex::new(Playhead {
                cursor: 0,
                // Opening a recording starts it playing. The alternative — a
                // paused first frame with nothing drawn — makes the tool look
                // broken to somebody who has just opened a file.
                playing: true,
                speed,
                since: Instant::now(),
                carry: 0.0,
            }),
            fold: Mutex::new(Fold {
                at: 0,
                progress: LatestOpPerChunk::new(),
            }),
        }
    }

    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self::new(read_order_log(path)?))
    }

    pub fn from_json(document: &Value) -> Result<Self> {
        Ok(Self::new(decode_order_log(document)?))
    }

    pub fn log(&self) -> &DecodedLog {
        &self.log
    }

    fn total(&self) -> u64 {
        self.log.events.len() as u64
    }

    /// Advance the playhead by the wall time since it was last looked at, and
    /// report where it now is.
    fn advance(&self) -> (u64, bool) {
        let mut playhead = self
            .playhead
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let elapsed = playhead.since.elapsed().as_secs_f64();
        playhead.since = Instant::now();
        if playhead.playing {
            let moved = elapsed * playhead.speed + playhead.carry;
            let whole = moved.floor();
            playhead.carry = moved - whole;
            playhead.cursor = (playhead.cursor + whole as u64).min(self.total());
            if playhead.cursor >= self.total() {
                // Stop at the end rather than looping. A loop is for a
                // presentation; this is for looking at what happened, and the
                // final frame is the answer.
                playhead.playing = false;
            }
        }
        (playhead.cursor, playhead.playing)
    }

    /// Bring the fold to `cursor`, rebuilding if the playhead went backwards.
    fn fold_to(&self, cursor: u64) -> Vec<crate::listener::BlockProgress> {
        let mut fold = self
            .fold
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if fold.at > cursor {
            fold.at = 0;
            fold.progress = LatestOpPerChunk::new();
        }
        for event in &self.log.events[fold.at as usize..cursor as usize] {
            fold.progress.on_event(event);
        }
        fold.at = cursor;
        fold.progress.snapshot()
    }
}

impl ProgressSource for ReplaySource {
    fn meta(&self) -> Meta {
        Meta {
            mode: Mode::Replay,
            controllable: true,
            strategy: self.log.strategy.clone(),
            volume: self.log.volume,
            grid: self.log.grid,
            phases: self.log.phases,
            ops: self.log.ops.clone(),
            total_events: Some(self.total()),
        }
    }

    fn state(&self) -> State {
        let (cursor, running) = self.advance();
        let blocks = self.fold_to(cursor);
        let seq = blocks.iter().map(|block| block.seq).max().unwrap_or(0);
        State {
            cursor,
            total: Some(self.total()),
            seq,
            running,
            blocks,
        }
    }

    fn timeline(&self, window: Window, limit: usize) -> TimelinePage {
        // The recording is indexed by `seq`, so a window is a slice and the
        // work is bounded by `limit` whichever direction it is taken in.
        let total = self.total();
        let (from, to) = match window {
            Window::Since(since) => {
                let from = since.min(total);
                (from, total)
            }
            Window::Before(before) => (0, before.min(total)),
        };
        let mut events: Vec<_> = Vec::new();
        match window {
            Window::Since(_) => {
                for (seq, event) in self.log.events[from as usize..to as usize]
                    .iter()
                    .enumerate()
                {
                    if let Some(entry) = project(from + seq as u64, event) {
                        events.push(entry);
                    }
                    if events.len() >= limit {
                        break;
                    }
                }
            }
            Window::Before(_) => {
                // Backwards from the playhead until `limit` survive the
                // projection, then back into stream order.
                for (offset, event) in self.log.events[from as usize..to as usize]
                    .iter()
                    .rev()
                    .enumerate()
                {
                    if let Some(entry) = project(to - 1 - offset as u64, event) {
                        events.push(entry);
                    }
                    if events.len() >= limit {
                        break;
                    }
                }
                events.reverse();
            }
        }
        let next = events
            .last()
            .map(|entry| entry.seq + 1)
            .unwrap_or(match window {
                Window::Since(since) => since,
                Window::Before(before) => before,
            });
        TimelinePage {
            since: events.first().map(|entry| entry.seq).unwrap_or(from),
            next,
            available: total,
            events,
        }
    }

    fn control(&self, command: Control) -> bool {
        let total = self.total();
        let mut playhead = self
            .playhead
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Any command resets the clock: without this, a pause followed by a
        // play would jump forward by however long the pause lasted.
        playhead.since = Instant::now();
        playhead.carry = 0.0;
        match command {
            Control::Play => {
                // Playing from the end restarts, which is what the button
                // plainly means when the playhead is already there.
                if playhead.cursor >= total {
                    playhead.cursor = 0;
                }
                playhead.playing = true;
            }
            Control::Pause => playhead.playing = false,
            Control::Seek(to) => {
                playhead.cursor = to.min(total);
                playhead.playing = false;
            }
            Control::Step(by) => {
                let moved = playhead.cursor as i64 + by;
                playhead.cursor = moved.clamp(0, total as i64) as u64;
                playhead.playing = false;
            }
            Control::Speed(events_per_second) => {
                if !events_per_second.is_finite() {
                    return false;
                }
                playhead.speed = events_per_second.clamp(MIN_SPEED, MAX_SPEED);
            }
        }
        true
    }
}
