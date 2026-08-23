// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The worker, as a program.
//
//     blockflow-worker --rendezvous "file:$SHARED/$JOB_ID.json"
//     blockflow-worker --coordinator 10.0.0.4:8732
//
// It plans nothing, and this file is short because of that: find the
// coordinator, join, run, write down what happened. Every argument here is
// about *finding* the coordinator or about reporting — none of them is about
// what to compute, because that comes from the job.
//
// The one knob that is about computing is `--ahead`, and it is a pipeline
// depth rather than a plan: how many tasks to keep in hand so that the
// prefetcher can fetch block N+1 while block N is being computed. Two is the
// design's "at least one ahead".
//
// This binary resolves workflows with the built-in probe factory. A deployment
// with its own ops ships its own worker binary — five lines of `main` around
// its own `WorkflowFactory` — because a translated kernel cannot live in this
// crate and does not need to.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use blockflow::distributed::rendezvous;
use blockflow::distributed::spec::ProbeWorkflows;
use blockflow::distributed::worker::{self, WorkerOptions};
use blockflow::error::{Error, Result};

const USAGE: &str = "\
blockflow-worker — pulls block tasks from a coordinator and executes them.

    blockflow-worker --rendezvous SPEC [options]
    blockflow-worker --coordinator HOST:PORT [options]

  --rendezvous SPEC     How to find the coordinator. One of
                          file:PATH      a shared filesystem, keyed by job id
                          env:VARIABLE   the scheduler already told everyone
                          object:DIR/KEY an object store, polled
                          direct:HOST:PORT
  --coordinator ADDR    Short for --rendezvous direct:ADDR.
  --job NAME            Which job, where a coordinator is running several.
  --name NAME           This worker's name in the logs.
  --ahead N             Tasks to keep in hand. Default 2: one being computed,
                        one ready, so prefetch can start block N+1 during N.
  --wait SECONDS        How long to wait for the coordinator. Default 60.
  --stop-after N        Exit after N tasks without reporting. For showing that a
                        worker's death is survivable; never for real work.
  --abort-after N       Die by unhandled signal having computed N tasks, with
                        the last one's completion never sent. The same purpose
                        as --stop-after and a harder death: nothing is flushed
                        and every claim is still held, which is what a lost node
                        looks like from the coordinator. Never for real work.
  --threads N           Threads to spend inside one block. Default 1, which is
                        uncut and is what this worker has always done. A worker
                        computes one task at a time, so on a node with cores to
                        spare this is its only parallelism; set it to the node's
                        cores divided by the worker processes sharing them. A
                        chain that has not declared itself sliceable ignores it.
  --report PATH         Write this worker's counters here on exit.
  --verbose
  --help
";

fn main() {
    if let Err(error) = go() {
        eprintln!("blockflow-worker: {error}");
        std::process::exit(1);
    }
}

fn go() -> Result<()> {
    let mut spec: Option<String> = None;
    let mut options = WorkerOptions::new(SocketAddr::from(([127, 0, 0, 1], 0)));
    let mut report: Option<PathBuf> = None;
    let mut wait = Duration::from_secs(60);

    let mut rest = std::env::args().skip(1);
    while let Some(flag) = rest.next() {
        let mut value = || {
            rest.next()
                .ok_or_else(|| Error::invalid(format!("{flag} needs a value")))
        };
        match flag.as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--rendezvous" => spec = Some(value()?),
            "--coordinator" => spec = Some(format!("direct:{}", value()?)),
            "--job" => options.job = Some(value()?),
            "--name" => options.name = Some(value()?),
            "--ahead" => {
                options.ahead = value()?
                    .parse()
                    .map_err(|_| Error::invalid("--ahead takes a number".to_string()))?
            }
            "--wait" => {
                let seconds: u64 = value()?
                    .parse()
                    .map_err(|_| Error::invalid("--wait takes seconds".to_string()))?;
                wait = Duration::from_secs(seconds);
            }
            "--stop-after" => {
                options.stop_after = Some(
                    value()?
                        .parse()
                        .map_err(|_| Error::invalid("--stop-after takes a number".to_string()))?,
                )
            }
            "--abort-after" => {
                options.abort_after = Some(
                    value()?
                        .parse()
                        .map_err(|_| Error::invalid("--abort-after takes a number".to_string()))?,
                )
            }
            "--threads" => {
                options.threads = value()?
                    .parse()
                    .map_err(|_| Error::invalid("--threads takes a number".to_string()))?
            }
            "--report" => report = Some(PathBuf::from(value()?)),
            "--verbose" => options.verbose = true,
            other => return Err(Error::invalid(format!("unknown flag {other:?}\n\n{USAGE}"))),
        }
    }

    let spec = spec.ok_or_else(|| {
        Error::invalid(format!(
            "a worker needs to know where its coordinator is.\n\n{USAGE}"
        ))
    })?;
    let backend = rendezvous::parse(&spec)?;
    let record = backend.resolve(wait)?;
    options.addr = record.addr;
    if options.verbose {
        eprintln!(
            "found the coordinator at {} via {}",
            record.addr,
            backend.describe()
        );
    }

    let outcome = worker::run(options, &ProbeWorkflows);
    // The report is written whether the run succeeded or not: a worker that
    // failed is exactly the one whose counters somebody wants.
    let (result, document) = match &outcome {
        Ok(done) => (
            Ok(()),
            serde_json::json!({
                "worker": done.worker,
                "job": done.job,
                "tasks": done.tasks,
                "short_circuited": done.short_circuited,
                "fragments": done.fragments,
                "sidecar_reads": done.sidecar_reads,
                "fragment_applications": done.fragment_applications,
                "events": done.events,
                "started_ready": done.started_ready,
                "started_after_waiting": done.started_after_waiting,
                "starved": done.starved,
                "told_to_wait": done.told_to_wait,
                "reads": done.reads,
                "chunks_read": done.chunks_read,
                "reductions": done.reductions,
                "reduced_bytes": done.reduced_bytes,
                "listener_faults": done.listener_faults,
                "elapsed_ms": done.elapsed.as_millis() as u64,
                "joined_ms": done.joined.as_millis() as u64,
                "ready_ms": done.ready.as_millis() as u64,
                "refused": done.refused,
                "first_pull_ms": done.first_pull.as_millis() as u64,
                "pull_connect_ms": done.pull_connect.as_millis() as u64,
                "ok": true,
            }),
        ),
        Err(error) => (
            Err(error.to_string()),
            serde_json::json!({"ok": false, "error": error.to_string()}),
        ),
    };
    if let Some(path) = &report {
        std::fs::write(path, document.to_string())
            .map_err(|err| Error::backend(format!("writing {}: {err}", path.display())))?;
    }
    match (result, outcome) {
        (Ok(()), Ok(done)) => {
            println!(
                "{}: {} tasks ({} short-circuited), {} events, {} started ready, \
                 {} after waiting, {} starved, in {:?}",
                done.worker,
                done.tasks,
                done.short_circuited,
                done.events,
                done.started_ready,
                done.started_after_waiting,
                done.starved,
                done.elapsed
            );
            Ok(())
        }
        (_, Err(error)) => Err(error),
        _ => unreachable!("the two arms agree"),
    }
}
