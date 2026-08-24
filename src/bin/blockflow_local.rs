// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Local multi-node mode: one command, a coordinator and N workers, **as
// separate processes**.
//
//     blockflow-local --workers 4
//     blockflow-local --workers 8 --blocks 32 --phases 2 --policy nearest-first
//     blockflow-local --workers 3 --kill 0:2        # a worker dies mid-run
//
// Processes rather than threads, and that is the point rather than an
// implementation detail. A thread-based fake shares one address space, one
// cache, one memory budget and one set of file handles, so it would exercise
// the message shapes and none of the things that actually go wrong: a worker
// reading an intermediate the writer had not flushed, two processes writing one
// file, an event stream that merges correctly only because both ends were the
// same object, a work list that stays ahead only because the "network" was a
// function call.
//
// It runs the same two binaries a cluster runs, over the same rendezvous, with
// the same protocol. What differs on a cluster is which rendezvous and who
// starts the processes.

use std::path::PathBuf;
use std::time::Duration;

use blockflow::distributed::local::{self, LocalOptions};
use blockflow::distributed::shared_volume::SharedVolumes;
use blockflow::distributed::spec::{probe_job_over, ChainSpec, StoreSpec};
use blockflow::distributed::HandoutPolicy;
use blockflow::error::{Error, Result};
use ndarray::Array3;

const USAGE: &str = "\
blockflow-local — a coordinator and N workers, as separate processes, here.

    blockflow-local [--workers N] [options]

  --workers N       How many worker processes. Default 2.
  --blocks N        Blocks along the split axis. Default 16.
  --phases N        Phases in the chain. Default 1.
  --policy NAME     naive | nearest-first. Default nearest-first.
                    (cache-modelled is built and refused; --policy cache says why)
  --ahead N         Tasks each worker keeps in hand. Default 2.
  --kill I:N        SIGKILL worker I once the job reports N tasks done. The
                    process is taken wherever it is — possibly inside a task,
                    certainly holding the tasks its list was keeping ahead. With
                    no --lease-ms the run aborts, naming the worker: node loss is
                    not recovered from. With one, the claims are reissued
                    instead.
  --lease-ms N      Opt in to reissuing a claim that goes unreported for N
                    milliseconds. **Off by default** — a claim is held until it
                    completes. Must exceed (--ahead + 1) x task duration or it
                    reissues work nobody lost; see `distributed`'s module
                    header for why this is not a default.
  --dir PATH        Where to put the job, the volumes and the reports.
                    Default: a fresh directory under the system temporary one.
  --keep            Do not delete the directory afterwards.
  --quiet           Do not pass the children's output through.
  --help
";

fn main() {
    if let Err(error) = go() {
        eprintln!("blockflow-local: {error}");
        std::process::exit(1);
    }
}

fn go() -> Result<()> {
    let mut workers = 2usize;
    let mut blocks = 16usize;
    let mut phases = 1usize;
    let mut policy = HandoutPolicy::default();
    let mut ahead = 2usize;
    let mut kill: Vec<(usize, usize)> = Vec::new();
    let mut lease_ms: Option<u64> = None;
    let mut dir: Option<PathBuf> = None;
    let mut keep = false;
    let mut quiet = false;

    let mut rest = std::env::args().skip(1);
    while let Some(flag) = rest.next() {
        let mut value = || {
            rest.next()
                .ok_or_else(|| Error::invalid(format!("{flag} needs a value")))
        };
        let number = |text: String, what: &str| -> Result<usize> {
            text.parse()
                .map_err(|_| Error::invalid(format!("{what} takes a number")))
        };
        match flag.as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--workers" => workers = number(value()?, "--workers")?,
            "--blocks" => blocks = number(value()?, "--blocks")?,
            "--phases" => phases = number(value()?, "--phases")?,
            "--ahead" => ahead = number(value()?, "--ahead")?,
            "--policy" => {
                // `select`, which refuses a policy that is built but not
                // calibrated and says why. The list used to be spelled out here
                // and would have gone stale the moment one was refused.
                policy = HandoutPolicy::select(&value()?)?;
            }
            "--kill" => {
                let text = value()?;
                let (which, after) = text.split_once(':').ok_or_else(|| {
                    Error::invalid("--kill takes WORKER:TASKS, for example 0:2".to_string())
                })?;
                kill.push((
                    number(which.to_string(), "--kill")?,
                    number(after.to_string(), "--kill")?,
                ));
            }
            "--lease-ms" => {
                let ms: u64 = value()?
                    .parse()
                    .map_err(|_| Error::invalid("--lease-ms takes a number".to_string()))?;
                // Zero is refused rather than treated as "no lease": a lease of
                // zero milliseconds expires every claim at the instant it is
                // made, so a reader who meant "off" and typed 0 would get the
                // most destructive setting there is. "Off" is spelled by not
                // passing the flag.
                if ms == 0 {
                    return Err(Error::invalid(
                        "--lease-ms 0 would expire every claim the moment it was handed out.                          Omit --lease-ms for no expiry, which is the default."
                            .to_string(),
                    ));
                }
                lease_ms = Some(ms);
            }
            "--dir" => dir = Some(PathBuf::from(value()?)),
            "--keep" => keep = true,
            "--quiet" => quiet = true,
            other => return Err(Error::invalid(format!("unknown flag {other:?}\n\n{USAGE}"))),
        }
    }

    let dir = dir.unwrap_or_else(|| {
        std::env::temp_dir().join(format!("blockflow-local-{}", std::process::id()))
    });
    std::fs::create_dir_all(&dir)
        .map_err(|err| Error::backend(format!("creating {}: {err}", dir.display())))?;
    let volumes = dir.join("volumes");

    let (mut spec, decomposition) = probe_job_over(
        blocks,
        phases,
        ChainSpec::identity(),
        StoreSpec::Files {
            dir: volumes.clone(),
        },
    );
    spec.policy = policy;
    spec.lease = lease_ms.map(Duration::from_millis);

    // The input, written once, by whoever submits the job. On a cluster this is
    // an array that already exists; here it has to be made.
    let store = SharedVolumes::create(
        &volumes,
        spec.workflow.shape,
        spec.workflow.chunk,
        decomposition.n_phases(),
    )?;
    store.write_image(0, &ramp(spec.workflow.shape))?;
    drop(store);

    let mut options = LocalOptions::new(dir.clone(), workers)?;
    options.kill_at_progress = kill;
    options.inherit_output = !quiet;
    options.ahead = ahead;
    options.timeout = Duration::from_secs(300);

    println!(
        "{} workers, {} tasks over {} phases, {:?} voxels, handout {}",
        workers,
        decomposition.n_tasks(),
        decomposition.n_phases(),
        decomposition.volume,
        spec.policy.as_str()
    );
    let run = local::run(&options, &spec, &decomposition)?;
    println!(
        "\ndone in {:?}: {} of {} tasks, {} reissued, {} events, {} workers",
        run.elapsed,
        run.status.done,
        run.status.tasks,
        run.status.reissued,
        run.status.events,
        run.status.workers
    );
    println!(
        "coverage: {}",
        if run
            .report
            .get("coverage_ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            "every block, every op, in order".to_string()
        } else {
            format!(
                "FAILED — {}",
                run.report
                    .get("coverage")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("(no reason recorded)")
            )
        }
    );
    println!(
        "pipeline: {} tasks started with work already in hand, {} starved",
        run.started_ready(),
        run.starved()
    );
    if !run.reissued().is_empty() {
        println!("reissued: {:?}", run.reissued());
    }
    println!("output and reports in {}", dir.display());
    if !keep {
        std::fs::remove_dir_all(&dir).ok();
    }
    Ok(())
}

/// A volume whose every voxel is different, so a block written to the wrong
/// place is visible rather than plausible.
fn ramp(shape: [usize; 3]) -> Array3<f64> {
    let mut array = Array3::zeros((shape[0], shape[1], shape[2]));
    for (flat, value) in array.iter_mut().enumerate() {
        *value = flat as f64;
    }
    array
}
