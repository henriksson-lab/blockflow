// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A coordinator and N workers, on one machine, as **separate processes**.
//
// This is the primary verification vehicle for everything in this module, and
// the "separate processes" is the whole of why. A thread-based fake would share
// one address space, one cache, one memory budget, one allocator and one set of
// file handles, so it would validate the message shapes and nothing else — and
// the things that actually go wrong here are the things a shared address space
// hides: a worker that reads an intermediate before the writer flushed it, two
// processes writing one file, an event stream that only merges correctly by
// accident because both ends were the same object, a work list that only stays
// ahead because the "network" was a function call.
//
// So: real processes, real sockets, real independent caches, real process
// boundaries — and the same two binaries a cluster runs. What differs between a
// laptop and a cluster is which rendezvous is used and how the processes are
// started, which is exactly the amount of difference the design set out to have.
//
// What it verifies, and why each needs this rather than a unit test
// -----------------------------------------------------------------
// * **N workers produce byte-identical output to a single-node run.** The
//   headline. It needs several processes sharing storage, because that is what
//   a wrong seam or a missing flush would show up in.
// * **Every block executed exactly once**, asserted with the crate's existing
//   `check_coverage_and_order` over the *merged* event stream. It needs real
//   workers because the merge is the thing under test.
// * **A worker dies and its task is reissued.** It needs a process, because the
//   failure being modelled is a process disappearing.
// * **The work list stays ahead.** It needs real latency between the ask and the
//   answer, which is the thing a function call does not have.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::decomposition::Decomposition;
use crate::error::{Error, Result};

use super::client::Client;
use super::protocol::{self, JobStatus};
use super::rendezvous::{FileRendezvous, Rendezvous};
use super::spec::JobSpec;

/// Where the two binaries are.
///
/// Defaulted from this process's own location, because the local runner is
/// itself one of the three and they are built into the same directory. A
/// deployment that installs them elsewhere passes the paths.
#[derive(Debug, Clone)]
pub struct Binaries {
    pub coordinator: PathBuf,
    pub worker: PathBuf,
}

impl Binaries {
    pub fn beside_this_one() -> Result<Self> {
        let here = std::env::current_exe()
            .map_err(|err| Error::backend(format!("finding this program: {err}")))?;
        let dir = here
            .parent()
            .ok_or_else(|| Error::backend("this program has no directory".to_string()))?;
        Ok(Self {
            coordinator: dir.join(exe("blockflow-coordinator")),
            worker: dir.join(exe("blockflow-worker")),
        })
    }

    fn check(&self) -> Result<()> {
        for (what, path) in [("coordinator", &self.coordinator), ("worker", &self.worker)] {
            if !path.is_file() {
                return Err(Error::backend(format!(
                    "no {what} binary at {}. Local multi-node mode starts the same \
                     binaries a cluster runs, so they have to be built: `cargo build \
                     --features distributed`.",
                    path.display()
                )));
            }
        }
        Ok(())
    }
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

#[derive(Debug, Clone)]
pub struct LocalOptions {
    pub workers: usize,
    /// Scratch: the job spec, the rendezvous, the level files and the reports.
    pub dir: PathBuf,
    pub binaries: Binaries,
    /// Make worker `index` stop after `tasks` tasks, to demonstrate a death.
    /// The process exits without reporting, so the coordinator finds out the
    /// only way it can — the lease runs out.
    pub stop_after: Vec<(usize, usize)>,
    /// **Kill** worker `index` once the job reports `done` tasks.
    ///
    /// A harder death than `stop_after`, and a more faithful one: `SIGKILL`
    /// takes the process wherever it is, so it can be *inside* a task, with a
    /// block half-written and nothing reported. Triggered on observed progress
    /// rather than on a timer, so it is deterministic — a timer would race the
    /// job and the test would pass by luck.
    pub kill_at_progress: Vec<(usize, usize)>,
    pub timeout: Duration,
    pub inherit_output: bool,
    pub ahead: usize,
}

impl LocalOptions {
    pub fn new(dir: impl Into<PathBuf>, workers: usize) -> Result<Self> {
        Ok(Self {
            workers: workers.max(1),
            dir: dir.into(),
            binaries: Binaries::beside_this_one()?,
            stop_after: Vec::new(),
            kill_at_progress: Vec::new(),
            timeout: Duration::from_secs(120),
            inherit_output: false,
            ahead: 2,
        })
    }
}

/// What a local run produced.
#[derive(Debug, Clone)]
pub struct LocalRun {
    pub status: JobStatus,
    pub dir: PathBuf,
    /// The coordinator's report: status, reissues, and the merged event stream
    /// in the crate's own order-log schema.
    pub report: Value,
    pub workers: Vec<Value>,
    pub elapsed: Duration,
}

impl LocalRun {
    /// Tasks that were handed out more than once — the number that says a
    /// reissue actually happened rather than being merely possible.
    pub fn reissued(&self) -> Vec<(usize, u64)> {
        self.report
            .get("reissued")
            .and_then(Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|entry| {
                        Some((
                            entry.get("task")?.as_u64()? as usize,
                            entry.get("attempts")?.as_u64()?,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Every worker's pipeline counters, summed. `starved` is the one that must
    /// be zero.
    pub fn starved(&self) -> usize {
        self.workers
            .iter()
            .filter_map(|report| report.get("starved").and_then(Value::as_u64))
            .sum::<u64>() as usize
    }

    /// Sidecar fragments written, per worker, in join order. The number that
    /// says the fragments the merge is about to read came from several
    /// *processes* rather than from one.
    pub fn fragments_per_worker(&self) -> Vec<usize> {
        self.workers
            .iter()
            .map(|report| report.get("fragments").and_then(Value::as_u64).unwrap_or(0) as usize)
            .collect()
    }

    pub fn started_ready(&self) -> usize {
        self.workers
            .iter()
            .filter_map(|report| report.get("started_ready").and_then(Value::as_u64))
            .sum::<u64>() as usize
    }
}

/// Children that are killed if this is dropped, however that happens.
///
/// A test that fails part way must not leave a coordinator holding a port and a
/// handful of workers spinning against it. `Drop` rather than a tidy-up call,
/// because the case that matters is the one where the tidy-up call is not
/// reached.
struct Reaped(Vec<Child>);

impl Drop for Reaped {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Start a coordinator and `workers` workers, wait for the job, collect reports.
pub fn run(
    options: &LocalOptions,
    spec: &JobSpec,
    decomposition: &Decomposition,
) -> Result<LocalRun> {
    options.binaries.check()?;
    std::fs::create_dir_all(&options.dir)
        .map_err(|err| Error::backend(format!("creating {}: {err}", options.dir.display())))?;

    let spec_path = options.dir.join("job.json");
    // The spec carries its decomposition, which is therefore binding: the
    // coordinator uses it as given rather than choosing one, so a local run and
    // a cluster run of the same job seam identically.
    let document = spec.to_json(decomposition)?;
    std::fs::write(&spec_path, document.to_string())
        .map_err(|err| Error::backend(format!("writing {}: {err}", spec_path.display())))?;

    let rendezvous_path = options.dir.join("rendezvous.json");
    let _ = std::fs::remove_file(&rendezvous_path);
    let report_path = options.dir.join("coordinator.json");
    let started = Instant::now();

    let mut children = Reaped(Vec::new());
    let mut coordinator = Command::new(&options.binaries.coordinator);
    coordinator
        .arg("--rendezvous")
        .arg(format!("file:{}", rendezvous_path.display()))
        .arg("--job")
        .arg(&spec_path)
        .arg("--exit-when-done")
        .arg("--bind")
        .arg("127.0.0.1:0")
        .arg("--report")
        .arg(&report_path);
    plumb(&mut coordinator, options);
    children.0.push(
        coordinator
            .spawn()
            .map_err(|err| Error::backend(format!("starting the coordinator: {err}")))?,
    );

    // Wait for it to publish, exactly as a worker on another node would.
    let rendezvous = FileRendezvous::new(rendezvous_path.clone());
    let addr = rendezvous.resolve(Duration::from_secs(20))?.addr;

    let mut worker_reports = Vec::new();
    for index in 0..options.workers {
        let path = options.dir.join(format!("worker-{index}.json"));
        worker_reports.push(path.clone());
        let mut worker = Command::new(&options.binaries.worker);
        worker
            .arg("--rendezvous")
            .arg(format!("file:{}", rendezvous_path.display()))
            .arg("--name")
            .arg(format!("worker-{index}"))
            .arg("--ahead")
            .arg(options.ahead.to_string())
            .arg("--report")
            .arg(&path);
        if let Some((_, tasks)) = options.stop_after.iter().find(|(which, _)| *which == index) {
            worker.arg("--stop-after").arg(tasks.to_string());
        }
        plumb(&mut worker, options);
        children.0.push(
            worker
                .spawn()
                .map_err(|err| Error::backend(format!("starting worker {index}: {err}")))?,
        );
    }

    // The coordinator exits when the job is done; that is the signal.
    let deadline = started + options.timeout;
    let mut watcher = (!options.kill_at_progress.is_empty()).then(|| Client::new(addr));
    let mut pending_kills = options.kill_at_progress.clone();
    let coordinator_exit = loop {
        if let Some(client) = watcher.as_mut() {
            if let Ok(value) = client.get(protocol::path::STATUS) {
                let done = value.get("done").and_then(Value::as_u64).unwrap_or(0) as usize;
                pending_kills.retain(|&(index, at)| {
                    if done < at {
                        return true;
                    }
                    if let Some(child) = children.0.get_mut(index + 1) {
                        let _ = child.kill();
                    }
                    false
                });
                if pending_kills.is_empty() {
                    watcher = None;
                }
            }
        }
        match children.0[0].try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(err) => {
                return Err(Error::backend(format!(
                    "waiting for the coordinator: {err}"
                )))
            }
        }
        if Instant::now() > deadline {
            return Err(Error::backend(format!(
                "the job did not finish within {:?}. The coordinator's report so far is at \
                 {}; a job that stalls with tasks pending and no claims usually means every \
                 worker exited.",
                options.timeout,
                report_path.display()
            )));
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    if !coordinator_exit.success() {
        return Err(Error::backend(format!(
            "the coordinator exited with {coordinator_exit}"
        )));
    }
    for child in children.0.iter_mut().skip(1) {
        let _ = wait_briefly(child, Duration::from_secs(10));
    }

    let report = read_json(&report_path)?;
    let status = JobStatus::from_json(report.get("status").unwrap_or(&report))?;
    let workers = worker_reports
        .iter()
        .filter(|path| path.is_file())
        .map(|path| read_json(path))
        .collect::<Result<Vec<_>>>()?;
    Ok(LocalRun {
        status,
        dir: options.dir.clone(),
        report,
        workers,
        elapsed: started.elapsed(),
    })
}

fn plumb(command: &mut Command, options: &LocalOptions) {
    if options.inherit_output {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    } else {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }
}

fn wait_briefly(child: &mut Child, patience: Duration) -> Result<()> {
    let deadline = Instant::now() + patience;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Ok(()),
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(());
            }
            Err(err) => return Err(Error::backend(format!("waiting for a worker: {err}"))),
        }
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| Error::backend(format!("reading {}: {err}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|err| Error::invalid(format!("{} is not JSON: {err}", path.display())))
}
