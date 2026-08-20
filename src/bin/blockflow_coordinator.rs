// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The coordinator, as a program. **Always its own program** — never "rank 0
// behaves differently" — so that a worker's life is the same three lines
// wherever it runs.
//
// The two lifetimes are one binary
// --------------------------------
// A **persistent** coordinator accepts jobs over time and outlives them:
//
//     blockflow-coordinator --bind 0.0.0.0:8732 --allow-public
//
// A **per-run** coordinator is the same program started with one job and told
// to leave when it is done:
//
//     blockflow-coordinator --rendezvous "file:$SHARED/$JOB_ID.json" \
//                           --job job.json --exit-when-done
//
// There is no mode flag beyond `--exit-when-done`, and it is read in one place:
// the serving loop's exit condition. Everything else is identical, which is what
// stops the shape somebody tests from drifting away from the shape somebody
// runs.
//
// Under a batch scheduler that starts the script on the first allocated node:
//
//     blockflow-coordinator --rendezvous "$RDV" --job job.json --exit-when-done &
//     <launcher> blockflow-worker --rendezvous "$RDV"
//     wait
//
// That node runs two processes. The coordinator moves no image data and is idle
// between handouts, so it is a fraction of one core out of forty.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use blockflow::distributed::coordinator::Coordinator;
use blockflow::distributed::rendezvous::{self, Record};
use blockflow::distributed::server::{self, Options, DEFAULT_PORT};
use blockflow::distributed::spec::read_job;
use blockflow::error::{Error, Result};
use blockflow::export::{order_log_to_json, ExportMeta};

const USAGE: &str = "\
blockflow-coordinator — hands out block tasks and receives events.

    blockflow-coordinator [--job SPEC] [--rendezvous SPEC] [options]

  --job PATH            A job spec (JSON). Given, this is a per-run coordinator;
                        omitted, it waits for jobs on POST /job.
  --exit-when-done      Leave once every job has finished. The per-run shape.
  --rendezvous SPEC     Where to publish the address. One of
                          file:PATH      a shared filesystem, keyed by job id
                          env:VARIABLE   the scheduler already told everyone
                          object:DIR/KEY an object store, polled
                          direct:HOST:PORT
  --bind ADDR           Default 127.0.0.1:8732. Use 0 for the port to let the
                        system choose; the rendezvous publishes what was bound.
  --allow-public        Permit a non-loopback bind. Needed to serve other nodes,
                        and deliberately not the default: there is no
                        authentication here.
  --advertise HOST[:PORT]
                        The address other nodes should use, when it differs from
                        what was bound. Clusters have management and fabric
                        interfaces with different names for one host.
  --report PATH         On exit, write status, reissues and the merged event
                        stream (the crate's order-log schema) here.
  --linger-ms N         With --exit-when-done, how long the event stream must be
                        quiet before leaving. Default 250. Events are
                        fire-and-forget, so the last few are still in flight when
                        the last task finishes; this is how long the record is
                        given to complete itself. It delays nothing else.
  --help
";

struct Args {
    job: Option<PathBuf>,
    rendezvous: Option<String>,
    bind: SocketAddr,
    allow_public: bool,
    advertise: Option<String>,
    report: Option<PathBuf>,
    exit_when_done: bool,
    linger: Duration,
}

fn parse() -> Result<Option<Args>> {
    let mut args = Args {
        job: None,
        rendezvous: None,
        bind: SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT)),
        allow_public: false,
        advertise: None,
        report: None,
        exit_when_done: false,
        linger: Duration::from_millis(blockflow::distributed::DEFAULT_LINGER_MS),
    };
    let mut rest = std::env::args().skip(1);
    while let Some(flag) = rest.next() {
        let mut value = || {
            rest.next()
                .ok_or_else(|| Error::invalid(format!("{flag} needs a value")))
        };
        match flag.as_str() {
            "--help" | "-h" => return Ok(None),
            "--job" => args.job = Some(PathBuf::from(value()?)),
            "--rendezvous" => args.rendezvous = Some(value()?),
            "--bind" => {
                args.bind = blockflow::net::resolve_one(&value()?, DEFAULT_PORT)?;
            }
            "--allow-public" => args.allow_public = true,
            "--advertise" => args.advertise = Some(value()?),
            "--report" => args.report = Some(PathBuf::from(value()?)),
            "--exit-when-done" => args.exit_when_done = true,
            "--linger-ms" => {
                args.linger = Duration::from_millis(
                    value()?
                        .parse()
                        .map_err(|_| Error::invalid("--linger-ms takes a number".to_string()))?,
                )
            }
            other => return Err(Error::invalid(format!("unknown flag {other:?}\n\n{USAGE}"))),
        }
    }
    Ok(Some(args))
}

fn main() {
    if let Err(error) = go() {
        eprintln!("blockflow-coordinator: {error}");
        std::process::exit(1);
    }
}

fn go() -> Result<()> {
    let Some(args) = parse()? else {
        println!("{USAGE}");
        return Ok(());
    };

    let coordinator = Arc::new(Coordinator::new(args.exit_when_done).with_linger(args.linger));
    let mut job_id = None;
    if let Some(path) = &args.job {
        let (spec, decomposition) = read_job(path)?;
        println!(
            "job {:?}: {} phases, {} tasks over {:?}, handout {}",
            spec.id,
            decomposition.n_phases(),
            decomposition.n_tasks(),
            decomposition.volume,
            spec.policy.as_str()
        );
        job_id = Some(coordinator.submit(spec, decomposition)?);
    } else if args.exit_when_done {
        return Err(Error::invalid(
            "--exit-when-done with no --job would exit immediately. A persistent \
             coordinator takes neither; a per-run one takes both."
                .to_string(),
        ));
    }

    let handle = server::serve(
        coordinator.clone(),
        Options {
            bind: args.bind,
            allow_public: args.allow_public,
            advertise: args.advertise.clone(),
            ..Default::default()
        },
    )?;
    println!(
        "listening on {}, advertising {}",
        handle.bound(),
        handle.advertised()
    );

    // Publish *after* binding, because the point of publishing rather than
    // deriving a port is that the coordinator gets to say what it actually got.
    if let Some(spec) = &args.rendezvous {
        let backend = rendezvous::parse(spec)?;
        let record = Record::new(
            job_id.clone().unwrap_or_else(|| "-".to_string()),
            handle.advertised(),
        );
        backend.publish(&record)?;
        println!("published to {}", backend.describe());
    }

    loop {
        if coordinator.should_exit() {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    if let (Some(path), Some(id)) = (&args.report, &job_id) {
        write_report(&coordinator, id, path)?;
    }
    // An aborted job is a **failed** job, and the exit status has to say so:
    // whatever started this — a batch script, an orchestrator — decides what
    // happens next, and it decides on the status. The report is written first,
    // so the record of how far the job got survives the failure.
    if let Some(aborted) = coordinator.aborted() {
        handle.shutdown();
        return Err(Error::backend(aborted.message()));
    }
    if let Some(id) = &job_id {
        let status = coordinator.status(Some(id))?;
        println!(
            "job {:?} finished: {} tasks, {} reissued, {} events, {} workers",
            status.job, status.done, status.reissued, status.events, status.workers
        );
    }
    handle.shutdown();
    Ok(())
}

/// Everything a run needs to be checked after the fact, in one file.
///
/// The event stream goes out in the crate's **existing** order-log schema
/// rather than a distribution-specific one, which is the point: the merged
/// stream from N workers is the same document a single-node run writes, so
/// every tool that reads one reads the other, and the acceptance criterion is
/// asserted by the same function either way.
fn write_report(coordinator: &Coordinator, id: &str, path: &std::path::Path) -> Result<()> {
    let document = coordinator.inspect(id, |job| {
        let status = job.status();
        let meta = ExportMeta::new(
            "distributed",
            job.decomposition.volume,
            job.decomposition.n_phases(),
        )
        .with_ops(
            job.decomposition
                .op_names_in_order()
                .into_iter()
                .enumerate()
                .collect(),
        );
        let log = order_log_to_json(job.log(), &meta);
        let reissued: Vec<serde_json::Value> = job
            .reissued_tasks()
            .into_iter()
            .map(|(task, attempts)| serde_json::json!({"task": task, "attempts": attempts}))
            .collect();
        let aborted = job.aborted().map(|aborted| {
            serde_json::json!({
                "worker": aborted.worker,
                "why": aborted.why,
                "message": aborted.message(),
                "held": aborted.held.iter().map(|claim| serde_json::json!({
                    "task": claim.task,
                    "phase": claim.phase,
                    "index": claim.index,
                    "held_ms": claim.held.as_millis() as u64,
                })).collect::<Vec<_>>(),
            })
        });
        serde_json::json!({
            "status": status.to_json(),
            "aborted": aborted,
            "reissued": reissued,
            "unknown_events": job.unknown_events(),
            "coverage_ok": job.check_coverage().is_ok(),
            "coverage": job.check_coverage().err().map(|error| error.to_string()),
            "elapsed_ms": job.elapsed().as_millis() as u64,
            "log": log,
        })
    })?;
    std::fs::write(path, document.to_string())
        .map_err(|err| Error::backend(format!("writing {}: {err}", path.display())))
}
