// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The coordinator's HTTP surface: nine paths, one JSON body each.
//
// Blocking, on ordinary threads, for the reason the progress view uses
// `tiny_http`: the executor is synchronous, an async runtime would be a second
// concurrency model inside a crate that has one, and what an async framework
// buys (routing, extractors, middleware) is not what eight endpoints and a
// `match` need.
//
// Where the accept loop went, and why this file no longer has one
// ----------------------------------------------------------------
// It was `tiny_http`'s, then it was this file's, and it is now
// [`crate::http`]'s. The middle step is the one with the measurement behind it
// and it is recorded there in full: `tiny_http` dispatches each accepted
// connection to a thread from a pool, running `for rq in connection { .. }` —
// **a task that does not return while the connection lives** — and the pool
// grows a thread only when it observes none waiting, so connections arriving
// together queue behind tasks that never finish and are read only when some
// *other* connection closes.
//
// Every client here holds a permanent connection — that is the design's
// no-stall property, three per worker so none waits behind another — so "some
// other connection closes" means "the job ends". Measured on this tree: a
// worker ready to ask for work at 7 ms had its first pull answered at 270 ms,
// while the coordinator reported its slowest *served* pull at 115 microseconds
// and its longest wait for the registry lock at 17. Neither process was busy.
// The worker ran nothing, and every test above it reported that as a spread
// that did not happen or a death that did not occur.
//
// The third step is not a measurement, it is the same argument `net::check_bind`
// already stands on. The progress view turned out to have the identical defect —
// its own header has the rates — and repairing it produced a second accept loop,
// a second connection reader and a second `query`. **A transport in two places
// has two chances to drift and no way to notice**, so there is one:
// [`crate::http`], one thread per connection taken at accept.
//
// **What did not move is the JSON**, and that is the seam rather than an
// oversight. `crate::http` hands a handler a request whose body is bytes,
// because the other consumer of it serves a quarter megabyte of WebAssembly to
// a browser; this protocol's bodies are JSON objects or nothing, and
// [`Incoming::of`] is the one line that says so. The coordinator's assumption
// stays in the coordinator and the progress view's generality stays out of it.
//
// The acceptance criterion for all of this is not in this file and not a unit
// test: `local::run` refuses any run in which a worker's first pull went
// unanswered for longer than **a hundred times the slowest pull that
// coordinator actually served**, floored at 50 ms so a job too short to have a
// slow pull cannot set an impossible bound. It is self-calibrating, it runs on
// every multi-node run, and it is what says a transport change did not cost
// anything.
//
// The bind, which is the one decision here with consequences
// ----------------------------------------------------------
// A coordinator **must** be reachable by other nodes, so it is exactly the
// `--allow-public` path the progress view refuses by default. That is not a
// reason to relax it: it is the reason to keep it a conscious choice. The policy
// itself lives in `net::check_bind`, shared with the progress view so the two
// cannot drift, and the address that is *published* is separate from the address
// that is *bound* — see `net::advertised_addr` for why a cluster with a
// management and a fabric network makes that a real distinction rather than a
// tidiness.
//
// Threads
// -------
// One per connection plus one accepting — [`crate::http`]'s, not this file's —
// and the number is not interesting: a request is a claim-table update or a
// `Vec::push` under one lock, and the whole cluster generates under one handout
// per second plus a few events per second. What matters is that no handler ever
// waits on anything except that lock — in particular, no handler asks a worker
// anything — and that no connection waits to be *read*, which is the property a
// pool cost us.

use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;

use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::http::{param, query, Request, Response};
use crate::net::{advertised_addr, check_bind};

use super::coordinator::Coordinator;
use super::protocol::{path, PROTOCOL_VERSION};
use super::spec::{decompose, JobSpec};
use super::wire::{decomposition_from_json, text_or};

/// Not a registered number. One above the progress view's, so a node running
/// both does not need either to be moved.
pub const DEFAULT_PORT: u16 = 8732;

#[derive(Debug, Clone)]
pub struct Options {
    pub bind: SocketAddr,
    /// Permit a non-loopback bind. A coordinator serving other nodes needs it;
    /// see `net::check_bind` before setting it.
    pub allow_public: bool,
    /// What to publish to the rendezvous, when it differs from what was bound.
    pub advertise: Option<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT)),
            allow_public: false,
            advertise: None,
        }
    }
}

/// A running coordinator. Dropping it stops it.
///
/// A thin name over [`crate::http::Server`], kept because a caller of this
/// module holds a *coordinator* and because the address it publishes is not
/// always the address it bound — see [`Self::advertised`].
pub struct ServerHandle {
    server: crate::http::Server,
    advertised: SocketAddr,
}

impl ServerHandle {
    /// What was actually bound. Differs from `Options::bind` when the port was 0.
    pub fn bound(&self) -> SocketAddr {
        self.server.bound()
    }

    /// What to write to a rendezvous — the address another node should use.
    pub fn advertised(&self) -> SocketAddr {
        self.advertised
    }

    /// Stop, and wait for every connection thread.
    ///
    /// The accept loop blocks on `accept` on purpose, so nothing short of a
    /// connection returns it; [`crate::http::Server`] wakes it by dialling the
    /// port it bound, which is the one thing guaranteed to be listening. That
    /// is the same repair this file used to carry itself, and
    /// `a_coordinator_stops_without_waiting_for_a_connection` below still says
    /// it works from this side.
    pub fn shutdown(self) {
        self.server.shutdown();
    }
}

pub fn serve(coordinator: Arc<Coordinator>, options: Options) -> Result<ServerHandle> {
    check_bind(&options.bind, options.allow_public)?;
    let listener = TcpListener::bind(options.bind).map_err(|err| {
        Error::backend(format!(
            "could not listen on {}: {err}. If the port is in use, pass a different one, \
             or 0 to let the system choose — a rendezvous publishes whatever was bound, \
             which is why an operating-system port is safe here.",
            options.bind
        ))
    })?;
    let bound = listener.local_addr().unwrap_or(options.bind);
    let advertised = advertised_addr(bound, options.advertise.as_deref())?;
    if !bound.ip().is_loopback() {
        eprintln!(
            "warning: the coordinator is serving on {bound} and advertising {advertised}, \
             which is reachable from the network. There is no authentication; anyone who \
             can reach this host can read this job's plan and take work from it."
        );
    }
    let server = crate::http::serve(listener, move |request| {
        let incoming = Incoming::of(request);
        let (status, body) = handle(&incoming, coordinator.as_ref());
        // Every reply this protocol makes is a JSON body with a length, which
        // is what `distributed::client` refuses to read anything else in place
        // of. Nothing else is set: there is no cache to defeat here, because a
        // coordinator is asked nothing twice.
        Response::text(status, "application/json; charset=utf-8", &body.to_string())
    })?;
    Ok(ServerHandle { server, advertised })
}

/// One request, as this coordinator needs it: where it was sent and what it
/// carried.
///
/// **The one place this protocol's JSON assumption lives.** [`crate::http`]
/// hands a handler a body of *bytes*, because its other consumer serves a
/// quarter megabyte of WebAssembly to a browser and a shared transport that
/// parsed every body as JSON could not. Here every body is a JSON object or
/// nothing, so it is parsed once at the edge and every route below reads a
/// [`Value`].
struct Incoming {
    url: String,
    body: Value,
}

impl Incoming {
    /// A body that is not JSON becomes `Null` rather than an error, which is
    /// what a `GET` with no body has always produced and what every route below
    /// already handles: they ask the body for named fields and refuse by name
    /// when one is missing, which is a better message than "that was not JSON".
    fn of(request: &Request) -> Self {
        let body = request.body();
        Self {
            url: request.url().to_string(),
            body: if body.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(body).unwrap_or(Value::Null)
            },
        }
    }
}

/// The `job` a request means: from the body, or from the query string, or
/// absent — in which case the coordinator resolves the only job there is, which
/// is the per-run shape.
fn job_of(body: &Value, pairs: &[(String, String)]) -> Option<String> {
    body.get("job")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| param(pairs, "job"))
        .filter(|name| !name.is_empty())
}

/// Route one request and produce its status and body.
fn handle(request: &Incoming, coordinator: &Coordinator) -> (u16, Value) {
    let arrived = std::time::Instant::now();
    let url = &request.url;
    let route = url.split('?').next().unwrap_or("/").to_string();
    let pairs = query(url);
    let body = request.body.clone();
    let job = job_of(&body, &pairs);

    let outcome: Result<Value> = match route.as_str() {
        path::JOIN => {
            let hint = body
                .get("worker")
                .and_then(Value::as_str)
                .map(str::to_string);
            coordinator
                .join(job.as_deref(), hint.as_deref())
                .map(|joined| joined.to_json())
        }
        path::PULL => {
            let worker = text_or(&body, "worker", "");
            match coordinator.status(job.as_deref()) {
                Err(error) => Err(error),
                Ok(status) => coordinator
                    .pull(&status.job, &worker)
                    .map(|handout| handout.to_json()),
            }
        }
        path::REPORT => match (job.as_deref(), body.get("event")) {
            (_, None) => Err(Error::invalid("a report carries an \"event\"".to_string())),
            (named, Some(event)) => {
                let worker = text_or(&body, "worker", "");
                let seq = body.get("seq").and_then(Value::as_u64);
                coordinator
                    .status(named)
                    .and_then(|status| coordinator.report(&status.job, &worker, seq, event))
                    // `accepted` is false for an event this coordinator had
                    // already taken, which is how a sender's count of what it
                    // sent stays equal to the merged stream's length across a
                    // retry. See `Job::reported`.
                    .map(|accepted| json!({"ok": true, "accepted": accepted}))
            }
        },
        path::COMPLETED => {
            let worker = text_or(&body, "worker", "");
            let task = body.get("task").and_then(Value::as_u64).unwrap_or(0) as usize;
            coordinator
                .status(job.as_deref())
                .and_then(|status| coordinator.completed(&status.job, &worker, task))
                .map(|status| status.to_json())
        }
        path::FAILED => {
            let worker = text_or(&body, "worker", "");
            let task = body.get("task").and_then(Value::as_u64).unwrap_or(0) as usize;
            let why = text_or(&body, "why", "unspecified");
            coordinator
                .status(job.as_deref())
                .and_then(|status| coordinator.failed(&status.job, &worker, task, &why))
                .map(|()| json!({"ok": true}))
        }
        // A worker is gone. The coordinator cannot see this — it has no signal
        // for a process it did not start, and the only alternative would be to
        // infer it from silence, which is a timeout, which is the mechanism
        // this design removed. So the party that *did* start the worker says
        // so: `local::run` here, a batch script or an orchestrator on a
        // cluster. See `coordinator::Job::worker_lost`.
        path::LOST => {
            let worker = text_or(&body, "worker", "");
            let why = text_or(&body, "why", "unspecified");
            if worker.is_empty() {
                Err(Error::invalid(
                    "a loss names the worker that was lost; the message it produces is \
                     no use without it"
                        .to_string(),
                ))
            } else {
                coordinator
                    .worker_lost(job.as_deref(), &worker, &why)
                    .map(|aborted| {
                        json!({
                            "aborted": true,
                            "worker": aborted.worker,
                            "why": aborted.why,
                            "message": aborted.message(),
                            "held": aborted.held.iter().map(|claim| json!({
                                "task": claim.task,
                                "phase": claim.phase,
                                "index": claim.index,
                                "held_ms": claim.held.as_millis() as u64,
                            })).collect::<Vec<_>>(),
                        })
                    })
            }
        }
        path::STATUS => coordinator
            .status(job.as_deref())
            .map(|status| status.to_json()),
        path::JOBS => Ok(json!({
            "protocol": PROTOCOL_VERSION,
            "jobs": coordinator.job_ids(),
            "exit_when_done": coordinator.exit_when_done(),
            "all_finished": coordinator.all_finished(),
        })),
        path::SUBMIT => submit(coordinator, &body),
        path::LOG => coordinator
            .status(job.as_deref())
            .and_then(|status| coordinator.inspect(&status.job, |job| job.status().to_json())),
        _ => Err(Error::invalid(format!(
            "{route:?} is not an endpoint. This is a coordinator, not a progress view: \
             {}, {}, {}, {}, {}, {}, {}, {}, {}.",
            path::JOIN,
            path::PULL,
            path::REPORT,
            path::COMPLETED,
            path::FAILED,
            path::LOST,
            path::STATUS,
            path::SUBMIT,
            path::JOBS
        ))),
    };

    // Timed before the response is written, so what is measured is what this
    // process did rather than what the socket did with it. See
    // `coordinator::Serving`.
    coordinator.served(&route, arrived.elapsed());
    match outcome {
        Ok(value) => (200, value),
        Err(error) => (400, json!({"error": error.to_string()})),
    }
}

/// Submit a job. The persistent coordinator's entry point, and the same call the
/// per-run one makes to itself once at startup.
fn submit(coordinator: &Coordinator, body: &Value) -> Result<Value> {
    let spec = JobSpec::from_json(body)?;
    let decomposition = match body.get("decomposition") {
        Some(value) => decomposition_from_json(value)?,
        None => decompose(&spec, 1)?,
    };
    let id = coordinator.submit(spec, decomposition)?;
    let status = coordinator.status(Some(&id))?;
    Ok(status.to_json())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    use super::super::client::Client;

    /// **Every connection is read, however many arrive at once.**
    ///
    /// The property [`crate::http`] exists for, asserted here through *this*
    /// coordinator's own client and handler rather than inferred from a job
    /// finishing — which is a different claim from the transport being correct,
    /// and the one a worker depends on. Sixteen connections are opened together
    /// and *held*, which is what every client here does, and each one then has
    /// to be answered. Under a dispatch that hands connections to a fixed pool
    /// of threads and lets a task occupy its thread for the life of the
    /// connection, the ones past the pool size are never read at all; that is
    /// not a hypothetical, it is what `tiny_http` did here and what the header
    /// describes.
    ///
    /// The premise is asserted too: the connections are all still open when the
    /// last answer arrives, so this measured sixteen concurrent connections and
    /// not sixteen consecutive ones.
    #[test]
    fn every_connection_is_answered_even_when_they_all_arrive_at_once() {
        const CONNECTIONS: usize = 16;
        let coordinator = Arc::new(Coordinator::new(false));
        let handle = serve(
            coordinator,
            Options {
                bind: "127.0.0.1:0".parse().expect("a loopback address"),
                ..Default::default()
            },
        )
        .expect("a coordinator on loopback");
        let addr = handle.bound();

        let (done, answered) = mpsc::channel();
        let (release, go) = mpsc::channel::<()>();
        let go = Arc::new(std::sync::Mutex::new(go));
        let mut threads = Vec::new();
        for which in 0..CONNECTIONS {
            let done = done.clone();
            let go = go.clone();
            threads.push(std::thread::spawn(move || {
                let mut client = Client::new(addr);
                // The first request opens the connection.
                let first = client.get(path::JOBS).is_ok();
                done.send((which, first)).ok();
                // Hold it open until every peer has been answered, then use it
                // again: a connection that was read once must stay readable.
                let _ = go.lock().map(|go| go.recv_timeout(Duration::from_secs(10)));
                client.get(path::JOBS).is_ok()
            }));
        }
        drop(done);

        let mut answers = Vec::new();
        for _ in 0..CONNECTIONS {
            let answer = answered
                .recv_timeout(Duration::from_secs(10))
                .expect("every connection opened at once must be answered, not queued");
            answers.push(answer);
        }
        assert_eq!(answers.len(), CONNECTIONS);
        assert!(
            answers.iter().all(|(_, ok)| *ok),
            "some connection was answered with an error: {answers:?}"
        );
        // Every one of them was still open at this point, which is what makes
        // the count above a count of *concurrent* connections.
        for _ in 0..CONNECTIONS {
            release.send(()).ok();
        }
        for thread in threads {
            assert!(
                thread.join().expect("a client thread"),
                "a connection that had been served once stopped being readable"
            );
        }
        handle.shutdown();
    }

    /// Shutting down does not depend on a connection arriving, and does not
    /// leave the accept thread behind.
    ///
    /// The accept loop blocks on `accept` on purpose — a poll interval there is
    /// the delay before a connection is served, and the first version of this
    /// file proved that by reproducing the very stall it was written to remove.
    /// So `shutdown` has to wake it, and this is what says it still does now
    /// that the waking is [`crate::http::Server`]'s and not this file's.
    #[test]
    fn a_coordinator_stops_without_waiting_for_a_connection() {
        let coordinator = Arc::new(Coordinator::new(false));
        let handle = serve(
            coordinator,
            Options {
                bind: "127.0.0.1:0".parse().expect("a loopback address"),
                ..Default::default()
            },
        )
        .expect("a coordinator on loopback");
        let started = std::time::Instant::now();
        handle.shutdown();
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "shutting down took {:?}, which means it waited for something",
            started.elapsed()
        );
    }
}
