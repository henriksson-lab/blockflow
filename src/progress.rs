// SPDX-License-Identifier: MIT
//
// Local, append-only progress stream for jobman and other small launchers.
//
// The stream's events are encoded by `export::event_json`; this module owns
// only the envelope, lifecycle counters, and the policy that an observer's IO
// failure must never affect the computation it observes.

use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub use blockflow_progress::Level as ProgressLevel;
use blockflow_progress::{StreamWriter, SCHEMA, VERSION};
use serde_json::{json, Value};

use crate::decomposition::Decomposition;
use crate::env::Environment;
use crate::export::event_json;
use crate::listener::EventListener;
use crate::log::{Event, Stats};
use crate::strategy::{Strategy, Workflow};
use crate::Result;

fn keeps(level: ProgressLevel, event: &Event) -> bool {
    match level {
        ProgressLevel::Summary => false,
        ProgressLevel::Blocks => matches!(
            event,
            Event::PhaseStarted { .. }
                | Event::TaskAdmitted { .. }
                | Event::BlockShortCircuited { .. }
                | Event::BlockWritten { .. }
        ),
        ProgressLevel::Ops => matches!(
            event,
            Event::PhaseStarted { .. }
                | Event::TaskAdmitted { .. }
                | Event::OpApplied { .. }
                | Event::BlockShortCircuited { .. }
                | Event::BlockWritten { .. }
        ),
        ProgressLevel::All => true,
    }
}

/// Plan metadata that starts a progress stream before execution begins.
#[derive(Debug, Clone)]
pub struct ProgressMeta {
    pub run: String,
    pub worker: Option<String>,
    pub strategy: String,
    pub volume: [usize; 3],
    pub tasks: u64,
    phases: Vec<Value>,
}

impl ProgressMeta {
    /// Derive all plan-visible fields from the binding decomposition.
    pub fn from_plan(
        run: impl Into<String>,
        strategy: impl Into<String>,
        workflow: &Workflow,
        decomposition: &Decomposition,
    ) -> Self {
        let phases = decomposition
            .phases
            .iter()
            .enumerate()
            .map(|(phase, item)| {
                json!({
                    "phase": phase,
                    "grid": item.grid.blocks_per_axis(),
                    "blocks": item.blocks.len(),
                    "barrier": false,
                    "ops": item.slots.iter().zip(&item.names).map(|(slot, name)| json!({"slot": slot, "name": name})).collect::<Vec<_>>(),
                    "predicted_cost": Value::Null,
                })
            })
            .collect();
        Self {
            run: run.into(),
            worker: std::env::var("JOBMAN_TASK_INDEX").ok(),
            strategy: strategy.into(),
            volume: workflow.shape,
            tasks: decomposition.n_tasks() as u64,
            phases,
        }
    }
}

/// A best-effort local JSONL writer which implements [`EventListener`].
///
/// Construction may fail, so a launcher can report a bad requested output
/// path.  Once constructed, all failures are latched in `io_faulted` and are
/// deliberately ignored by the executor.
pub struct StreamingLog {
    sender: SyncSender<Message>,
    worker: Mutex<Option<JoinHandle<()>>>,
    level: ProgressLevel,
    started: Instant,
    phase_count: usize,
    admitted: Arc<Vec<AtomicU64>>,
    done: Arc<Vec<AtomicU64>>,
    seq: Arc<AtomicU64>,
    events_written: Arc<AtomicU64>,
    events_dropped: Arc<AtomicU64>,
    io_faulted: Arc<AtomicBool>,
    finished: AtomicBool,
}

/// The listener calls [`SyncSender::try_send`], never performing filesystem
/// IO on an executor thread.  The single flusher owns the file and emits
/// heartbeats even while the workflow has no events to report.
enum Message {
    Event(Value),
    Finish(Value),
}

impl StreamingLog {
    /// Start a stream at `path`, writing its `run_started` header immediately.
    pub fn create(
        path: impl AsRef<Path>,
        meta: ProgressMeta,
        level: ProgressLevel,
    ) -> io::Result<Self> {
        let heartbeat_ms = std::env::var("JOBMAN_PROGRESS_HEARTBEAT_MS")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(1_000);
        Self::create_with_heartbeat(path, meta, level, heartbeat_ms)
    }

    fn create_with_heartbeat(
        path: impl AsRef<Path>,
        meta: ProgressMeta,
        level: ProgressLevel,
        heartbeat_ms: u64,
    ) -> io::Result<Self> {
        let started_ms = unix_ms();
        let phase_count = meta.phases.len();
        let header = json!({
            "type": "run_started", "schema": SCHEMA, "version": VERSION,
            "level": level.name(), "run": meta.run, "worker": meta.worker,
            "t_unix_ms": started_ms, "strategy": meta.strategy, "volume": meta.volume,
            "tasks": meta.tasks, "phases": meta.phases,
        });
        let file = StreamWriter::create(path, header)?;
        // A bounded queue puts an explicit ceiling on observer memory.  A
        // full queue loses detail, never compute time; the heartbeat counters
        // make that loss visible to readers.
        let (sender, receiver) = mpsc::sync_channel(4_096);
        let admitted: Arc<Vec<AtomicU64>> =
            Arc::new((0..phase_count).map(|_| AtomicU64::new(0)).collect());
        let done: Arc<Vec<AtomicU64>> =
            Arc::new((0..phase_count).map(|_| AtomicU64::new(0)).collect());
        let seq = Arc::new(AtomicU64::new(0));
        let events_written = Arc::new(AtomicU64::new(0));
        let events_dropped = Arc::new(AtomicU64::new(0));
        let io_faulted = Arc::new(AtomicBool::new(false));
        let started = Instant::now();
        let worker = spawn_flusher(
            file,
            receiver,
            started,
            heartbeat_ms,
            admitted.clone(),
            done.clone(),
            seq.clone(),
            events_written.clone(),
            events_dropped.clone(),
            io_faulted.clone(),
        );
        Ok(Self {
            sender,
            worker: Mutex::new(Some(worker)),
            level,
            started,
            phase_count,
            admitted,
            done,
            seq,
            events_written,
            events_dropped,
            io_faulted,
            finished: AtomicBool::new(false),
        })
    }

    /// Construct a stream when jobman supplied `JOBMAN_PROGRESS_FILE`.
    pub fn from_jobman_env(meta: ProgressMeta) -> io::Result<Option<Self>> {
        let Some(path) = std::env::var_os("JOBMAN_PROGRESS_FILE") else {
            return Ok(None);
        };
        Self::create(path, meta, ProgressLevel::from_jobman_env()?).map(Some)
    }

    /// Run a strategy with this stream attached, always leaving a terminal record.
    pub fn run<S: Strategy + ?Sized>(
        self: &std::sync::Arc<Self>,
        strategy: &S,
        workflow: &Workflow,
        decomposition: &Decomposition,
        env: &dyn Environment,
    ) -> Result<Stats> {
        let listener: std::sync::Arc<dyn EventListener> = self.clone();
        let outcome = strategy.run_observed(workflow, decomposition, env, &[listener]);
        match &outcome {
            Ok(_) => self.finish("ok", None),
            Err(error) => self.finish("failed", Some(error.to_string())),
        }
        outcome
    }

    /// Write the terminal record. Repeated calls are harmless.
    pub fn finish(&self, outcome: &str, message: Option<String>) {
        if self.finished.swap(true, Ordering::Relaxed) {
            return;
        }
        let terminal = json!({
            "type": "run_finished", "t_unix_ms": unix_ms(), "elapsed_ms": self.elapsed_ms(),
            "outcome": outcome, "message": message, "tasks": self.admitted.iter().map(|x| x.load(Ordering::Relaxed)).sum::<u64>(),
            "done": self.done.iter().map(|x| x.load(Ordering::Relaxed)).collect::<Vec<_>>(),
            "events": self.events_written.load(Ordering::Relaxed), "listener_faults": 0,
        });
        // Finishing happens after strategy execution, so waiting for the
        // flusher is appropriate: it gives the terminal record its promised
        // durable boundary without ever stalling a worker event.
        if self.sender.send(Message::Finish(terminal)).is_err() {
            self.io_faulted.store(true, Ordering::Relaxed);
        }
        if let Ok(mut worker) = self.worker.lock() {
            if let Some(worker) = worker.take() {
                let _ = worker.join();
            }
        }
    }

    fn elapsed_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    fn write_value(&self, value: Value) {
        if self.io_faulted.load(Ordering::Relaxed) {
            self.events_dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        match self.sender.try_send(Message::Event(value)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                self.events_dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {
                self.io_faulted.store(true, Ordering::Relaxed);
                self.events_dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn event(&self, event: &Event) {
        match event {
            Event::TaskAdmitted { phase, .. } if *phase < self.phase_count => {
                self.admitted[*phase].fetch_add(1, Ordering::Relaxed);
            }
            Event::BlockShortCircuited { phase, .. } | Event::BlockWritten { phase, .. }
                if *phase < self.phase_count =>
            {
                self.done[*phase].fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        if keeps(self.level, event) {
            let mut value = event_json(event);
            if let Some(map) = value.as_object_mut() {
                map.insert(
                    "seq".into(),
                    json!(self.seq.fetch_add(1, Ordering::Relaxed)),
                );
            }
            self.write_value(value);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_flusher(
    mut file: StreamWriter,
    receiver: Receiver<Message>,
    started: Instant,
    heartbeat_ms: u64,
    admitted: Arc<Vec<AtomicU64>>,
    done: Arc<Vec<AtomicU64>>,
    seq: Arc<AtomicU64>,
    events_written: Arc<AtomicU64>,
    events_dropped: Arc<AtomicU64>,
    io_faulted: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || loop {
        match receiver.recv_timeout(Duration::from_millis(heartbeat_ms)) {
            Ok(Message::Event(value)) => {
                if matches!(file.write_event(&value), Ok(true)) {
                    events_written.fetch_add(1, Ordering::Relaxed);
                } else if !file.events_truncated() {
                    io_faulted.store(true, Ordering::Relaxed);
                }
            }
            Ok(Message::Finish(value)) => {
                if file.write(&value, true).is_err() {
                    io_faulted.store(true, Ordering::Relaxed);
                }
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let elapsed_ms = started.elapsed().as_millis() as u64;
                let heartbeat = json!({
                    "type": "heartbeat", "t_unix_ms": unix_ms(), "elapsed_ms": elapsed_ms,
                    "seq": seq.load(Ordering::Relaxed), "phase": Value::Null,
                    "admitted": admitted.iter().map(|x| x.load(Ordering::Relaxed)).collect::<Vec<_>>(),
                    "done": done.iter().map(|x| x.load(Ordering::Relaxed)).collect::<Vec<_>>(),
                    "events_written": events_written.load(Ordering::Relaxed), "events_dropped": events_dropped.load(Ordering::Relaxed),
                    "io_faulted": io_faulted.load(Ordering::Relaxed),
                });
                if file.write(&heartbeat, false).is_err() {
                    io_faulted.store(true, Ordering::Relaxed);
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    })
}

impl EventListener for StreamingLog {
    fn on_event(&self, event: &Event) {
        self.event(event);
    }
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dtype, IdentityOp, Trivial};

    #[test]
    fn writes_header_event_and_trailer() {
        let path = std::env::temp_dir().join(format!("blockflow-progress-{}", std::process::id()));
        let workflow = Workflow::new(
            crate::Chain::op(IdentityOp::new("identity", [0, 0, 0])),
            [4, 4, 4],
            Dtype::F32,
        );
        let strategy = Trivial;
        let decomposition = strategy.decompose(&workflow, &Default::default()).unwrap();
        let log = StreamingLog::create(
            &path,
            ProgressMeta::from_plan("test", strategy.name(), &workflow, &decomposition),
            ProgressLevel::Blocks,
        )
        .unwrap();
        log.on_event(&Event::TaskAdmitted {
            phase: 0,
            index: [0, 0, 0],
        });
        log.on_event(&Event::BlockWritten {
            phase: 0,
            index: [0, 0, 0],
            valid: crate::Region::whole(&[4, 4, 4]),
            materialised: false,
        });
        log.finish("ok", None);
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.lines().any(|line| line.contains("run_started")));
        assert!(contents.lines().any(|line| line.contains("task_admitted")));
        assert!(contents.lines().any(|line| line.contains("run_finished")));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn writes_heartbeats_while_execution_is_quiet() {
        let path = std::env::temp_dir().join(format!(
            "blockflow-progress-heartbeat-{}",
            std::process::id()
        ));
        let workflow = Workflow::new(
            crate::Chain::op(IdentityOp::new("identity", [0, 0, 0])),
            [4, 4, 4],
            Dtype::F32,
        );
        let strategy = Trivial;
        let decomposition = strategy.decompose(&workflow, &Default::default()).unwrap();
        let log = StreamingLog::create_with_heartbeat(
            &path,
            ProgressMeta::from_plan("quiet", strategy.name(), &workflow, &decomposition),
            ProgressLevel::Summary,
            1,
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(15));
        log.finish("ok", None);
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.lines().any(|line| line.contains("heartbeat")));
        std::fs::remove_file(path).unwrap();
    }
}
