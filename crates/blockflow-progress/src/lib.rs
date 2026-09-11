//! The portable JSONL envelope for a live Blockflow progress stream.
//!
//! This crate intentionally knows nothing about Blockflow's Rust `Event` enum.
//! The producer supplies the event value from `blockflow::export::event_json`,
//! and this crate owns only stream framing, schema identity, sequencing, and
//! the durable-write cadence.  That makes it usable by a small jobman reader
//! without making jobman depend on Blockflow's image-processing dependencies.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const SCHEMA: &str = "blockflow.progress";
pub const VERSION: u32 = 1;
pub const DEFAULT_SYNC_CADENCE: Duration = Duration::from_secs(2);
/// The largest amount of detailed event JSON a writer retains by default.
///
/// Heartbeats and the final record are never charged to this budget: detail
/// loss must not make a live run look stale or hide its outcome.
pub const DEFAULT_EVENT_BYTE_CAP: u64 = 256 * 1024 * 1024;
/// A reader waits this many polls before considering a complete malformed
/// record suspect. This protects against a torn append becoming visible on a
/// shared filesystem before its preceding page.
pub const CORRUPT_RETRY_LIMIT: u32 = 3;
/// A malformed record is skipped only when at least this many later bytes make
/// it clear it is not the writer's current, torn append.
pub const CORRUPT_DISTANCE: u64 = 4 * 1024;

/// Detail level requested by a launcher for a progress stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Summary,
    #[default]
    Blocks,
    Ops,
    All,
}

/// A named operation in a phase of the original Blockflow plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Operation {
    pub slot: usize,
    pub name: String,
}

/// Static information needed to render a phase before any tile is computed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseHeader {
    pub phase: usize,
    pub grid: [usize; 3],
    pub blocks: u64,
    #[serde(default)]
    pub barrier: bool,
    #[serde(default)]
    pub ops: Vec<Operation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_cost: Option<f64>,
}

/// The first (`run_started`) record in a progress stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Header {
    pub schema: String,
    pub version: u32,
    pub level: Level,
    pub run: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<String>,
    pub t_unix_ms: u64,
    pub strategy: String,
    pub volume: [usize; 3],
    pub tasks: u64,
    pub phases: Vec<PhaseHeader>,
}

/// A producer event. Unknown properties are retained for forward-compatible
/// consumers rather than discarded by the portable envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventLine {
    pub seq: u64,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub phase: Option<usize>,
    #[serde(default)]
    pub index: Option<[usize; 3]>,
    #[serde(default)]
    pub slot: Option<usize>,
    #[serde(default)]
    pub op: Option<String>,
    #[serde(default)]
    pub duration_ns: Option<u64>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// A summary emitted periodically even when no detailed events arrive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Heartbeat {
    pub t_unix_ms: u64,
    pub elapsed_ms: u64,
    pub seq: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<usize>,
    pub admitted: Vec<u64>,
    pub done: Vec<u64>,
    #[serde(default)]
    pub events_written: u64,
    #[serde(default)]
    pub events_dropped: u64,
    #[serde(default)]
    pub io_faulted: bool,
}

/// The outcome in the durable final (`run_finished`) record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Ok,
    Failed,
    Cancelled,
}

/// The final record emitted after execution reaches an outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trailer {
    pub t_unix_ms: u64,
    pub elapsed_ms: u64,
    pub outcome: Outcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    pub tasks: u64,
    pub done: Vec<u64>,
    #[serde(default)]
    pub events: u64,
    #[serde(default)]
    pub listener_faults: u64,
}

/// One record in the schema-versioned JSONL stream.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Header(Header),
    Event(EventLine),
    Heartbeat(Heartbeat),
    Trailer(Trailer),
}

/// An invalid or unsupported envelope record.
#[derive(Debug)]
pub enum Error {
    Json(serde_json::Error),
    Schema { schema: String, version: u32 },
    MissingType,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(error) => write!(f, "invalid progress record: {error}"),
            Self::Schema { schema, version } => {
                write!(
                    f,
                    "unsupported progress schema `{schema}` version {version}"
                )
            }
            Self::MissingType => f.write_str("progress record has no string `type`"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Schema { .. } | Self::MissingType => None,
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl Record {
    /// Parse one newline-stripped JSONL record, retaining unknown event fields.
    pub fn parse(bytes: &[u8]) -> Result<Self, Error> {
        let value: Value = serde_json::from_slice(bytes)?;
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or(Error::MissingType)?;
        match kind {
            "run_started" => {
                let header: Header = serde_json::from_value(value)?;
                if header.schema != SCHEMA || header.version != VERSION {
                    return Err(Error::Schema {
                        schema: header.schema,
                        version: header.version,
                    });
                }
                Ok(Self::Header(header))
            }
            "heartbeat" => Ok(Self::Heartbeat(serde_json::from_value(value)?)),
            "run_finished" => Ok(Self::Trailer(serde_json::from_value(value)?)),
            _ => Ok(Self::Event(serde_json::from_value(value)?)),
        }
    }

    /// Encode this record without its trailing JSONL newline.
    pub fn json(&self) -> Result<Vec<u8>, Error> {
        let mut value = match self {
            Self::Header(value) => serde_json::to_value(value)?,
            Self::Event(value) => serde_json::to_value(value)?,
            Self::Heartbeat(value) => serde_json::to_value(value)?,
            Self::Trailer(value) => serde_json::to_value(value)?,
        };
        let kind = match self {
            Self::Header(_) => "run_started",
            Self::Heartbeat(_) => "heartbeat",
            Self::Trailer(_) => "run_finished",
            Self::Event(value) => &value.kind,
        };
        if let Some(object) = value.as_object_mut() {
            object.insert("type".into(), Value::String(kind.into()));
        }
        Ok(serde_json::to_vec(&value)?)
    }
}

impl Level {
    pub fn from_jobman_env() -> io::Result<Self> {
        match std::env::var("JOBMAN_PROGRESS_LEVEL").as_deref() {
            Err(_) | Ok("blocks") => Ok(Self::Blocks),
            Ok("summary") => Ok(Self::Summary),
            Ok("ops") => Ok(Self::Ops),
            Ok("all") => Ok(Self::All),
            Ok(_) => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid JOBMAN_PROGRESS_LEVEL",
            )),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::Blocks => "blocks",
            Self::Ops => "ops",
            Self::All => "all",
        }
    }
}

/// One append-only writer. It never uses append mode: a stream has exactly one
/// producer, so a single open file and sequential writes need no filesystem
/// lock. `sync_data` is performed on a cadence rather than on a worker event.
pub struct StreamWriter {
    file: File,
    last_sync: Instant,
    sync_cadence: Duration,
    event_byte_cap: u64,
    event_bytes: u64,
    truncated: bool,
}

impl StreamWriter {
    /// Create a new stream and synchronously publish the initial header.
    pub fn create(path: impl AsRef<Path>, header: Value) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        write_line(&mut file, &header)?;
        file.sync_data()?;
        Ok(Self {
            file,
            last_sync: Instant::now(),
            sync_cadence: DEFAULT_SYNC_CADENCE,
            event_byte_cap: DEFAULT_EVENT_BYTE_CAP,
            event_bytes: 0,
            truncated: false,
        })
    }

    pub fn with_sync_cadence(mut self, cadence: Duration) -> Self {
        self.sync_cadence = cadence;
        self
    }

    /// Configure the maximum amount of detailed event JSON retained.
    pub fn with_event_byte_cap(mut self, cap: u64) -> Self {
        self.event_byte_cap = cap;
        self
    }

    /// Whether the writer has emitted its one `stream_truncated` marker.
    pub fn events_truncated(&self) -> bool {
        self.truncated
    }

    /// Frame a Blockflow event and attach the protocol sequence number.
    pub fn event(&mut self, mut event: Value, seq: u64) -> io::Result<()> {
        let object = event.as_object_mut().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "progress event must be a JSON object",
            )
        })?;
        object.insert("seq".into(), json!(seq));
        self.write_event(&event).map(|_| ())
    }

    /// Write a pre-framed detailed event, enforcing the stream detail budget.
    ///
    /// Returns `true` when the supplied event was retained. After the first
    /// over-budget event, the writer emits exactly one `stream_truncated`
    /// marker and returns `false` for it and every later detailed event.
    /// Summary records written with [`Self::write`] continue normally.
    pub fn write_event(&mut self, event: &Value) -> io::Result<bool> {
        if self.truncated {
            return Ok(false);
        }
        let encoded = serde_json::to_vec(event).map_err(io::Error::other)?;
        let bytes = u64::try_from(encoded.len().saturating_add(1)).unwrap_or(u64::MAX);
        if self.event_bytes.saturating_add(bytes) <= self.event_byte_cap {
            self.event_bytes = self.event_bytes.saturating_add(bytes);
            self.write(event, false)?;
            return Ok(true);
        }

        let seq = event.get("seq").cloned().unwrap_or(Value::Null);
        self.write(
            &json!({
                "type": "stream_truncated",
                "seq": seq,
                "event_byte_cap": self.event_byte_cap,
            }),
            false,
        )?;
        self.truncated = true;
        Ok(false)
    }

    /// Write an envelope record. Terminal records pass `force_sync = true`.
    pub fn write(&mut self, record: &Value, force_sync: bool) -> io::Result<()> {
        write_line(&mut self.file, record)?;
        if force_sync || self.last_sync.elapsed() >= self.sync_cadence {
            self.file.sync_data()?;
            self.last_sync = Instant::now();
        }
        Ok(())
    }
}

fn write_line(file: &mut File, value: &Value) -> io::Result<()> {
    serde_json::to_writer(&mut *file, value).map_err(io::Error::other)?;
    file.write_all(b"\n")?;
    file.flush()
}

/// Incrementally read a live JSONL stream.
///
/// The reader reopens the file on every poll and never advances past a partial
/// final line. It also tolerates a writer restart, NUL holes from a
/// shared-filesystem page race, and a persistently malformed older record.
/// Consequently it is safe for both a local UI and a later shared-filesystem
/// reader without depending on Blockflow's execution crate.
#[derive(Debug, Clone)]
pub struct Tail {
    path: PathBuf,
    offset: u64,
    corrupt_lines: u64,
    pending_corrupt: Option<PendingCorrupt>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingCorrupt {
    start: u64,
    end: u64,
    polls: u32,
}

impl Tail {
    pub fn at(path: impl Into<PathBuf>, offset: u64) -> Self {
        Self {
            path: path.into(),
            offset,
            corrupt_lines: 0,
            pending_corrupt: None,
        }
    }

    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self::at(path, 0)
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn corrupt_lines(&self) -> u64 {
        self.corrupt_lines
    }

    pub fn poll(&mut self) -> io::Result<Vec<Record>> {
        let mut file = match File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let length = file.metadata()?.len();
        // A producer restart truncates the old stream. Reset so the replacement
        // header is visible rather than silently seeking beyond EOF.
        if length < self.offset {
            self.offset = 0;
            self.pending_corrupt = None;
        }
        file.seek(SeekFrom::Start(self.offset))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let mut records = Vec::new();
        let mut consumed = 0_usize;
        while let Some(relative_end) = bytes[consumed..].iter().position(|byte| *byte == b'\n') {
            let end = consumed + relative_end;
            let line = &bytes[consumed..=end];
            match Record::parse(&line[..line.len().saturating_sub(1)]) {
                Ok(record) => {
                    consumed = end + 1;
                    self.pending_corrupt = None;
                    records.push(record);
                }
                Err(_) if line.contains(&0) => {
                    // A NUL hole represents an out-of-order page, not a bad
                    // record. Waiting preserves the chance to see it whole.
                    break;
                }
                Err(_) => {
                    let start = self.offset.saturating_add(consumed as u64);
                    let line_end = self.offset.saturating_add((end + 1) as u64);
                    let polls = match self.pending_corrupt {
                        Some(pending) if pending.start == start && pending.end == line_end => {
                            pending.polls.saturating_add(1)
                        }
                        _ => 1,
                    };
                    self.pending_corrupt = Some(PendingCorrupt {
                        start,
                        end: line_end,
                        polls,
                    });
                    // Skip a durable bad old record only after repeated
                    // observation and sufficient later data. A current torn
                    // append must remain retryable.
                    if polls >= CORRUPT_RETRY_LIMIT
                        && length.saturating_sub(line_end) >= CORRUPT_DISTANCE
                    {
                        consumed = end + 1;
                        self.corrupt_lines = self.corrupt_lines.saturating_add(1);
                        self.pending_corrupt = None;
                        continue;
                    }
                    break;
                }
            }
        }
        self.offset = self.offset.saturating_add(consumed as u64);
        Ok(records)
    }
}

/// The current status of a tile reconstructed from detailed events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TileState {
    Pending,
    Running,
    Computed,
    ShortCircuited,
}

/// A tile and the operations observed for it, in Blockflow slot order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Tile {
    pub index: [usize; 3],
    pub state: TileState,
    pub ops: Vec<String>,
}

/// Progress and detail reconstructed for one phase.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PhaseProgress {
    pub phase: usize,
    pub admitted: u64,
    pub done: u64,
    pub total: u64,
    pub percent: f64,
    pub tiles: Vec<Tile>,
}

/// The renderer-friendly state reconstructed from all observed records.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunProgress {
    pub header: Header,
    pub phases: Vec<PhaseProgress>,
    pub percent: f64,
    pub elapsed_ms: u64,
    pub eta_ms: Option<u64>,
    pub outcome: Option<Outcome>,
    pub events_dropped: u64,
    pub detail_complete: bool,
}

/// Incrementally reconstruct [`RunProgress`] from an append-only stream.
#[derive(Debug, Default)]
pub struct Fold {
    header: Option<Header>,
    admitted: Vec<u64>,
    done: Vec<u64>,
    elapsed_ms: u64,
    outcome: Option<Outcome>,
    events_dropped: u64,
    tiles: Vec<BTreeMap<[usize; 3], Tile>>,
    seen_events: BTreeSet<u64>,
}

impl Fold {
    pub fn feed(&mut self, record: &Record) {
        match record {
            Record::Header(header) => {
                self.admitted = vec![0; header.phases.len()];
                self.done = vec![0; header.phases.len()];
                self.tiles = vec![BTreeMap::new(); header.phases.len()];
                self.header = Some(header.clone());
            }
            Record::Heartbeat(heartbeat) => {
                self.elapsed_ms = heartbeat.elapsed_ms;
                self.admitted.clone_from(&heartbeat.admitted);
                self.done.clone_from(&heartbeat.done);
                self.events_dropped = heartbeat.events_dropped;
            }
            Record::Trailer(trailer) => {
                self.elapsed_ms = trailer.elapsed_ms;
                self.done.clone_from(&trailer.done);
                self.outcome = Some(trailer.outcome.clone());
            }
            Record::Event(event) => self.event(event),
        }
    }

    fn event(&mut self, event: &EventLine) {
        if !self.seen_events.insert(event.seq) {
            return;
        }
        let (Some(phase), Some(index)) = (event.phase, event.index) else {
            return;
        };
        let Some(tiles) = self.tiles.get_mut(phase) else {
            return;
        };
        let tile = tiles.entry(index).or_insert(Tile {
            index,
            state: TileState::Pending,
            ops: Vec::new(),
        });
        match event.kind.as_str() {
            "task_admitted" => {
                tile.state = TileState::Running;
                if let Some(value) = self.admitted.get_mut(phase) {
                    *value += 1;
                }
            }
            "op_applied" => {
                tile.state = TileState::Running;
                if let Some(operation) = &event.op {
                    tile.ops.push(operation.clone());
                }
            }
            "block_short_circuited" => {
                tile.state = TileState::ShortCircuited;
                if let Some(names) = event
                    .extra
                    .get("names")
                    .or_else(|| event.extra.get("ops"))
                    .and_then(Value::as_array)
                {
                    tile.ops
                        .extend(names.iter().filter_map(Value::as_str).map(str::to_owned));
                }
                if let Some(value) = self.done.get_mut(phase) {
                    *value += 1;
                }
            }
            "block_written" => {
                tile.state = TileState::Computed;
                if let Some(value) = self.done.get_mut(phase) {
                    *value += 1;
                }
            }
            _ => {}
        }
    }

    pub fn progress(&self) -> Option<RunProgress> {
        let header = self.header.clone()?;
        let mut completed = 0_u64;
        let mut total = 0_u64;
        let phases = header
            .phases
            .iter()
            .enumerate()
            .map(|(index, phase)| {
                let done = *self.done.get(index).unwrap_or(&0);
                let admitted = *self.admitted.get(index).unwrap_or(&0);
                completed += done.min(phase.blocks);
                total += phase.blocks;
                PhaseProgress {
                    phase: phase.phase,
                    admitted,
                    done,
                    total: phase.blocks,
                    percent: ratio(done, phase.blocks),
                    tiles: self
                        .tiles
                        .get(index)
                        .map(|tiles| tiles.values().cloned().collect())
                        .unwrap_or_default(),
                }
            })
            .collect();
        let percent = ratio(completed, total);
        let eta_ms = (percent > 0.0 && percent < 1.0)
            .then(|| ((self.elapsed_ms as f64) * (1.0 - percent) / percent) as u64);
        Some(RunProgress {
            header,
            phases,
            percent,
            elapsed_ms: self.elapsed_ms,
            eta_ms,
            outcome: self.outcome.clone(),
            events_dropped: self.events_dropped,
            detail_complete: self.events_dropped == 0,
        })
    }
}

fn ratio(done: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        done.min(total) as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_event_and_syncs_terminal_record() {
        let path = std::env::temp_dir().join(format!(
            "blockflow-progress-envelope-{}",
            std::process::id()
        ));
        let mut writer = StreamWriter::create(
            &path,
            json!({
                "type": "run_started", "schema": SCHEMA, "version": VERSION,
            }),
        )
        .unwrap();
        writer.event(json!({"type": "block_written"}), 7).unwrap();
        writer
            .write(&json!({"type": "run_finished"}), true)
            .unwrap();
        let lines: Vec<Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines[0]["schema"], SCHEMA);
        assert_eq!(lines[1]["seq"], 7);
        assert_eq!(lines[2]["type"], "run_finished");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn cap_emits_one_marker_and_preserves_summary_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("progress.jsonl");
        let mut writer = StreamWriter::create(
            &path,
            json!({"type": "run_started", "schema": SCHEMA, "version": VERSION}),
        )
        .unwrap()
        .with_event_byte_cap(0)
        .with_sync_cadence(Duration::ZERO);
        assert!(!writer
            .write_event(&json!({"type": "op_applied", "seq": 7}))
            .unwrap());
        assert!(!writer
            .write_event(&json!({"type": "op_applied", "seq": 8}))
            .unwrap());
        writer
            .write(&json!({"type": "heartbeat", "seq": 9}), false)
            .unwrap();
        assert!(writer.events_truncated());
        let lines: Vec<Value> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[1]["type"], "stream_truncated");
        assert_eq!(lines[1]["seq"], 7);
        assert_eq!(lines[2]["type"], "heartbeat");
    }

    fn header() -> Header {
        Header {
            schema: SCHEMA.into(),
            version: VERSION,
            level: Level::Blocks,
            run: "run".into(),
            worker: None,
            t_unix_ms: 1,
            strategy: "greedy".into(),
            volume: [8, 8, 8],
            tasks: 4,
            phases: vec![PhaseHeader {
                phase: 0,
                grid: [2, 2, 1],
                blocks: 4,
                barrier: false,
                ops: vec![Operation {
                    slot: 0,
                    name: "median".into(),
                }],
                predicted_cost: None,
            }],
        }
    }

    #[test]
    fn reader_reconstructs_tiles_and_retries_a_torn_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("progress.jsonl");
        let header_bytes = Record::Header(header()).json().unwrap();
        std::fs::write(&path, &header_bytes[..header_bytes.len() / 2]).unwrap();
        let mut tail = Tail::new(&path);
        assert!(tail.poll().unwrap().is_empty());

        let event = Record::Event(EventLine {
            seq: 1,
            kind: "op_applied".into(),
            phase: Some(0),
            index: Some([0, 0, 0]),
            slot: Some(0),
            op: Some("median".into()),
            duration_ns: None,
            extra: Default::default(),
        })
        .json()
        .unwrap();
        let mut full = header_bytes;
        full.push(b'\n');
        full.extend_from_slice(&event);
        full.push(b'\n');
        std::fs::write(&path, full).unwrap();
        let mut fold = Fold::default();
        for record in tail.poll().unwrap() {
            fold.feed(&record);
        }
        let progress = fold.progress().unwrap();
        assert_eq!(progress.phases[0].tiles[0].ops, ["median"]);
    }
}
