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
// * **Every block executed exactly once**, asserted with
//   `ExecutionLog::check_coverage_unordered` over the *merged* event stream. It
//   needs real workers because the merge is the thing under test — and the merge
//   is also why the criterion is the unordered one: each worker posts its events
//   from its own reporter thread and the coordinator appends them as they
//   arrive, so arrival order across workers is not the order the work ran in.
//   Asserting the ordered form here failed about one run in five and was asking
//   the log a question it cannot answer; the work's order is the coordinator's
//   state machine, not the log's.
// * **A worker dies and the job aborts, naming what was lost.** The default;
//   node loss is not recovered from. It needs a process, because the failure
//   being modelled is a process disappearing — and because the *detector* is a
//   process exit, seen by this runner, which started them.
// * **A worker dies under an explicit lease and its task is reissued.** The
//   same death with one field of the spec set. Opt-in since 2026-08-17; see the
//   module header of `distributed`.
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
    /// Scratch: the job spec, the rendezvous, the image files and the reports.
    pub dir: PathBuf,
    pub binaries: Binaries,
    /// Make worker `index` stop after `tasks` tasks, to demonstrate a death.
    /// The process exits without reporting, so the coordinator finds out from
    /// the runner, which started it — or, if the job set a lease, from the
    /// lease running out.
    pub stop_after: Vec<(usize, usize)>,
    /// **Kill** worker `index` once the job reports `done` tasks.
    ///
    /// A harder death than `stop_after`: `SIGKILL` takes the process wherever
    /// it is, so it can be *inside* a task, with a block half-written and
    /// nothing reported.
    ///
    /// **It is not deterministic, and it used to say it was.** The trigger is
    /// observed progress rather than a timer, but progress is observed by
    /// polling the coordinator's own HTTP server, and on a loaded machine the
    /// job can pass from the threshold to finished between two samples — after
    /// which this kills a worker that has already gone home. Measured, that
    /// happened often enough to fail both death tests by itself. Prefer
    /// [`Self::abort_worker_after`], which counts on the worker's side and
    /// samples nothing; this stays for the one property that one has not got,
    /// a death *inside* a task.
    ///
    /// What happens next depends on the job, and that is the point: with no
    /// lease — the default — the run **aborts**, because node loss is not
    /// recovered from; with a lease, the claims are reissued.
    pub kill_at_progress: Vec<(usize, usize)>,
    /// Make worker `index` die by unhandled signal having computed `tasks`
    /// tasks, with the last one's completion never sent.
    ///
    /// The deterministic death: nothing is sampled and nothing is polled, so
    /// the worker dies while the job is running whenever it ran at all. See
    /// `WorkerOptions::abort_after` for why that is the property that matters
    /// and what it costs.
    pub abort_worker_after: Vec<(usize, usize)>,
    pub timeout: Duration,
    pub inherit_output: bool,
    pub ahead: usize,
    /// `--threads` for every worker this runner starts: how many threads one of
    /// them may spend **inside one block**.
    ///
    /// **`1` is the default and is what every recorded run of this runner was
    /// taken at.** It is here rather than left to the binary because a runner
    /// that starts `workers` processes on one box is the thing that knows how
    /// the box is being divided — see `WorkerOptions::threads`, which is where
    /// the argument for a per-worker rather than a per-job setting is written.
    ///
    /// Note the interaction with `workers`: `workers x threads` is what this
    /// runner asks of the machine, and nothing checks it against the machine's
    /// cores, exactly as `workers` alone was never checked.
    pub threads: usize,
}

impl LocalOptions {
    pub fn new(dir: impl Into<PathBuf>, workers: usize) -> Result<Self> {
        Ok(Self {
            workers: workers.max(1),
            dir: dir.into(),
            binaries: Binaries::beside_this_one()?,
            stop_after: Vec::new(),
            kill_at_progress: Vec::new(),
            abort_worker_after: Vec::new(),
            timeout: Duration::from_secs(120),
            inherit_output: false,
            ahead: 2,
            // One, so a runner built today starts the workers it always started.
            threads: 1,
        })
    }
}

/// A worker process the runner saw end before the job did.
///
/// The premise of a death test, recorded rather than assumed: a worker that
/// never got a task cannot die holding one, and a run where that happened
/// demonstrates nothing about node loss however green it comes out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Died {
    /// Index into the runner's workers, so `worker` is `worker-{index}`.
    pub index: usize,
    /// How the operating system said it ended.
    pub status: String,
    /// A successful exit. Not a death — a worker asked to stop after N tasks
    /// leaves this way, and the coordinator has nothing to recover from.
    pub cleanly: bool,
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
    /// Workers this runner saw end before the job did. See [`Died`].
    pub died: Vec<Died>,
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

    /// Waits the coordinator answered with "nothing for you now", summed over
    /// the workers. The other outcome of an empty list; see
    /// `WorkerReport::told_to_wait`.
    pub fn told_to_wait(&self) -> usize {
        self.workers
            .iter()
            .filter_map(|report| report.get("told_to_wait").and_then(Value::as_u64))
            .sum::<u64>() as usize
    }

    /// Pulls the **coordinator** answered `Wait` while it had ready work in
    /// hand. See `Job::withheld`: legitimate only for a scarce phase, and the
    /// number that catches a handout answering only idle workers without
    /// depending on any worker's timing to notice.
    pub fn withheld(&self) -> usize {
        self.report
            .get("withheld")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize
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

    /// Tasks executed per worker, in join order. The number that says whether a
    /// run with N workers was a run *by* N workers.
    pub fn tasks_per_worker(&self) -> Vec<usize> {
        self.workers
            .iter()
            .map(|report| report.get("tasks").and_then(Value::as_u64).unwrap_or(0) as usize)
            .collect()
    }

    /// Milliseconds each worker took to be ready to ask for work, in join
    /// order — admission plus rebuilding the plan it was given.
    ///
    /// Beside [`Self::tasks_per_worker`] this is what tells "that worker
    /// arrived after the work was gone" from "that worker was here from the
    /// first millisecond and was given nothing", which look identical in the
    /// task counts and have nothing else in common. See
    /// `WorkerReport::ready`.
    pub fn ready_ms_per_worker(&self) -> Vec<u64> {
        self.workers
            .iter()
            .map(|report| report.get("ready_ms").and_then(Value::as_u64).unwrap_or(0))
            .collect()
    }

    /// The longest any worker waited between asking for work the first time and
    /// being answered, in milliseconds, with the worker it happened to.
    ///
    /// **The handout's liveness, measured rather than assumed.** A pull that is
    /// never answered is indistinguishable, from the worker, from a coordinator
    /// with nothing to give — and indistinguishable, from the coordinator, from
    /// a worker that never asked. It is neither: it is a request sitting unread
    /// between two idle processes, which is what a permanent connection did to
    /// this crate's server until `Client` stopped keeping them. See there.
    pub fn worst_handout_wait(&self) -> (String, u64) {
        self.workers
            .iter()
            .map(|report| {
                let name = report
                    .get("worker")
                    .and_then(Value::as_str)
                    .unwrap_or("a worker")
                    .to_string();
                let first = report
                    .get("first_pull_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let ready = report.get("ready_ms").and_then(Value::as_u64).unwrap_or(0);
                (name, first.saturating_sub(ready))
            })
            .max_by_key(|(_, waited)| *waited)
            .unwrap_or_else(|| ("no workers".to_string(), 0))
    }

    /// Microseconds the coordinator spent on the slowest pull it *served* — the
    /// scale a handout wait is measured against. See
    /// [`Self::worst_handout_wait`].
    pub fn slowest_served_pull_us(&self) -> u64 {
        self.report
            .get("serving")
            .and_then(|serving| serving.get("slowest_pull_us"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
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
            .arg("--threads")
            .arg(options.threads.to_string())
            .arg("--report")
            .arg(&path);
        if let Some((_, tasks)) = options.stop_after.iter().find(|(which, _)| *which == index) {
            worker.arg("--stop-after").arg(tasks.to_string());
        }
        if let Some((_, tasks)) = options
            .abort_worker_after
            .iter()
            .find(|(which, _)| *which == index)
        {
            worker.arg("--abort-after").arg(tasks.to_string());
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
    // Detecting a lost node, and why it is here rather than in the coordinator.
    //
    // **The runner started these processes, so the runner is what the operating
    // system tells when one of them dies.** `try_wait` is that signal: a real
    // event, delivered at the moment it happens, with an exit status attached.
    // The coordinator has nothing equivalent — it did not fork them, its HTTP
    // server hands it requests and not connections, and the only thing it could
    // watch instead is *silence*, which is a timeout, which is the mechanism
    // this design deliberately removed. So the party with the signal relays it,
    // and the coordinator does the naming and the stopping.
    //
    // On a cluster this same role belongs to whatever launched the workers —
    // `srun`, which kills the step when a task dies, or an orchestrator. The
    // shape is the same and so is the decision: node loss ends the job.
    //
    // A job that **set a lease** is opting into reissue, which is a different
    // belief about what a lost node costs, so a death is not fatal there and
    // the runner leaves it to the lease. That is the one branch, and it reads
    // off the spec rather than off an option nobody would keep in step.
    let recovers_by_lease = spec.lease.is_some();
    let mut lost: Option<String> = None;
    let mut died: Vec<Died> = Vec::new();
    let mut seen_exit = vec![false; options.workers];
    let mut teller = Client::new(addr);
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
                    // Whether it was in time is not knowable here and is not
                    // recorded here: `died` below is the record, and it is
                    // taken from the process actually ending before the job
                    // did rather than from this sample.
                    false
                });
                if pending_kills.is_empty() {
                    watcher = None;
                }
            }
        }
        // Watched whatever the job's lease says, because the two questions are
        // different: *reporting* a loss is the no-lease branch, but *recording*
        // that a worker died is what lets a caller tell "the death this test is
        // about happened" from "the run finished with everybody alive". A test
        // whose premise is a death has to be able to check it.
        for index in 0..options.workers {
            if seen_exit[index] {
                continue;
            }
            let Some(child) = children.0.get_mut(index + 1) else {
                continue;
            };
            let Ok(Some(status)) = child.try_wait() else {
                continue;
            };
            seen_exit[index] = true;
            // A worker that exited **after** the coordinator finished the job
            // went home; that is the normal end of a run and is not a loss.
            // Asked of the coordinator rather than assumed, because the two
            // exits race by design.
            let finished = teller
                .get(protocol::path::JOBS)
                .ok()
                .and_then(|value| value.get("all_finished").and_then(Value::as_bool))
                .unwrap_or(false);
            if finished {
                break;
            }
            died.push(Died {
                index,
                status: status.to_string(),
                cleanly: status.success(),
            });
            if !recovers_by_lease && lost.is_none() {
                let worker = format!("worker-{index}");
                let _ = teller.post(
                    protocol::path::LOST,
                    &serde_json::json!({
                        "worker": worker,
                        "why": format!("the process exited with {status} before the job finished"),
                    }),
                );
                lost = Some(worker);
            }
            break;
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
                 worker exited, and one that stalls with none *handed out* means a worker \
                 that was expected never joined — this runner tells the coordinator to hold \
                 its handouts until all {} of them have.",
                options.timeout,
                report_path.display(),
                options.workers
            )));
        }
        // While a kill is still pending the poll interval is the width of the
        // window this runner can miss, and the window is the job's remaining
        // work — which for a probe job is milliseconds. Ten of them was enough
        // to walk past a whole sixteen-task job between two polls. Once the
        // kills have landed there is nothing left to be prompt about.
        std::thread::sleep(Duration::from_millis(if watcher.is_some() {
            1
        } else {
            10
        }));
    };
    if !coordinator_exit.success() {
        // An aborted job exits non-zero on purpose, and its report — written
        // before it left — says which worker went and what it was holding.
        // Preferred over the exit status alone because "the coordinator exited
        // with 1" is exactly the unreadable ending this is meant to replace.
        if let Ok(report) = read_json(&report_path) {
            if let Some(message) = report
                .get("aborted")
                .and_then(|aborted| aborted.get("message"))
                .and_then(Value::as_str)
            {
                return Err(Error::backend(message.to_string()));
            }
        }
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
    // A worker that failed wrote `{"ok": false, ...}` and left. Until this
    // check existed the runner read that report, found no `tasks` in it,
    // counted the worker as having done nothing and returned success — so a
    // three-worker run could be a one-worker run with nothing saying so, and
    // the tests whose premise is several processes failed later and elsewhere
    // for reasons that read like flakes. A worker's failure is the run's.
    let failed: Vec<String> = workers
        .iter()
        .filter(|report| report.get("ok").and_then(Value::as_bool) == Some(false))
        .map(|report| {
            format!(
                "{}: {}",
                report
                    .get("worker")
                    .and_then(Value::as_str)
                    .unwrap_or("a worker"),
                report
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("no reason given")
            )
        })
        .collect();
    if !failed.is_empty() {
        return Err(Error::backend(format!(
            "{} of {} workers failed and the job did not: {}",
            failed.len(),
            options.workers,
            failed.join("; ")
        )));
    }
    let run = LocalRun {
        status,
        dir: options.dir.clone(),
        report,
        workers,
        died,
        elapsed: started.elapsed(),
    };
    // **A pull that was never answered is not a run.**
    //
    // Checked here rather than in one test because it is a property of the
    // runner's promise — these are N worker processes against one coordinator —
    // and because every failure it causes reads as something else: a worker
    // that ran nothing, a spread that did not happen, a death that did not
    // occur. Measured before this check existed, a worker's first pull was
    // answered 263 ms after it asked while the coordinator's slowest *served*
    // pull took 115 microseconds; the wait was in neither process.
    //
    // The bound is taken from the run rather than chosen: a hundred times the
    // longest pull this coordinator actually served, floored so that a job too
    // short to have a slow pull cannot set an impossible one. Both numbers are
    // in the message, so a failure states its own scale.
    let served_us = run.slowest_served_pull_us();
    let bound_ms = (served_us.saturating_mul(100) / 1_000).max(50);
    let (slow_worker, waited_ms) = run.worst_handout_wait();
    if waited_ms > bound_ms {
        return Err(Error::backend(format!(
            "{slow_worker} asked for work and was not answered for {waited_ms} ms, against \
             a coordinator whose slowest *served* pull took {served_us} us and a bound of \
             {bound_ms} ms. The request was sitting unread between two idle processes, so \
             this run says nothing about how work is placed — see \
             `distributed::client::Client`."
        )));
    }
    Ok(run)
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
