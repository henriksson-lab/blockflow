// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The coordinator's HTTP surface: eight paths, one JSON body each.
//
// `tiny_http` for the same reason the progress view uses it — the executor is
// synchronous, an async runtime would be a second concurrency model inside a
// crate that has one, and what an async framework buys (routing, extractors,
// middleware) is not what eight endpoints and a `match` need. It is a blocking
// server on ordinary threads, and a request loop that reads like a request loop.
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
// A handful, and the number is not interesting: a request is a claim-table
// update or a `Vec::push` under one lock, and the whole cluster generates under
// one handout per second plus a few events per second. What matters is that no
// handler ever waits on anything except that lock — in particular, no handler
// asks a worker anything.

use std::net::SocketAddr;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::{json, Value};
use tiny_http::{Header, Request, Response, Server, StatusCode};

use crate::error::{Error, Result};
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
    pub threads: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT)),
            allow_public: false,
            advertise: None,
            threads: 4,
        }
    }
}

/// A running coordinator. Dropping it stops it.
pub struct ServerHandle {
    bound: SocketAddr,
    advertised: SocketAddr,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl ServerHandle {
    /// What was actually bound. Differs from `Options::bind` when the port was 0.
    pub fn bound(&self) -> SocketAddr {
        self.bound
    }

    /// What to write to a rendezvous — the address another node should use.
    pub fn advertised(&self) -> SocketAddr {
        self.advertised
    }

    pub fn shutdown(mut self) {
        self.stop.store(true, Ordering::Release);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

pub fn serve(coordinator: Arc<Coordinator>, options: Options) -> Result<ServerHandle> {
    check_bind(&options.bind, options.allow_public)?;
    let server = Server::http(options.bind).map_err(|err| {
        Error::backend(format!(
            "could not listen on {}: {err}. If the port is in use, pass a different one, \
             or 0 to let the system choose — a rendezvous publishes whatever was bound, \
             which is why an operating-system port is safe here.",
            options.bind
        ))
    })?;
    let bound = server.server_addr().to_ip().unwrap_or(options.bind);
    let advertised = advertised_addr(bound, options.advertise.as_deref())?;
    if !bound.ip().is_loopback() {
        eprintln!(
            "warning: the coordinator is serving on {bound} and advertising {advertised}, \
             which is reachable from the network. There is no authentication; anyone who \
             can reach this host can read this job's plan and take work from it."
        );
    }
    let server = Arc::new(server);
    let stop = Arc::new(AtomicBool::new(false));
    let mut threads = Vec::new();
    for worker in 0..options.threads.max(1) {
        let server = server.clone();
        let coordinator = coordinator.clone();
        let stop = stop.clone();
        threads.push(
            std::thread::Builder::new()
                .name(format!("blockflow-coordinator-{worker}"))
                .spawn(move || request_loop(&server, coordinator.as_ref(), &stop))
                .map_err(|err| Error::backend(format!("starting a request thread: {err}")))?,
        );
    }
    Ok(ServerHandle {
        bound,
        advertised,
        stop,
        threads,
    })
}

fn request_loop(server: &Server, coordinator: &Coordinator, stop: &AtomicBool) {
    while !stop.load(Ordering::Acquire) {
        let request = match server.recv_timeout(Duration::from_millis(100)) {
            Ok(Some(request)) => request,
            Ok(None) => continue,
            Err(_) => break,
        };
        // A panic in a handler must not take the thread with it: the run is
        // fine, and a coordinator that quietly loses a request thread would
        // degrade into a hang rather than an error.
        let handled = catch_unwind(AssertUnwindSafe(|| handle(request, coordinator)));
        if handled.is_err() {
            eprintln!("coordinator: a request handler panicked; the job is unaffected");
        }
    }
}

fn respond(request: Request, status: u16, body: &Value) {
    let text = body.to_string();
    let response = Response::from_data(text.into_bytes())
        .with_status_code(StatusCode(status))
        .with_header(
            Header::from_bytes(
                &b"Content-Type"[..],
                &b"application/json; charset=utf-8"[..],
            )
            .expect("a constant header parses"),
        );
    let _ = request.respond(response);
}

fn body_of(request: &mut Request) -> Value {
    let mut text = String::new();
    if request.as_reader().read_to_string(&mut text).is_err() || text.is_empty() {
        return Value::Null;
    }
    serde_json::from_str(&text).unwrap_or(Value::Null)
}

fn query(url: &str) -> Vec<(String, String)> {
    let Some((_, tail)) = url.split_once('?') else {
        return Vec::new();
    };
    tail.split('&')
        .filter(|part| !part.is_empty())
        .map(|part| match part.split_once('=') {
            Some((key, value)) => (key.to_string(), value.to_string()),
            None => (part.to_string(), String::new()),
        })
        .collect()
}

fn param(pairs: &[(String, String)], name: &str) -> Option<String> {
    pairs
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.clone())
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

fn handle(mut request: Request, coordinator: &Coordinator) {
    let url = request.url().to_string();
    let route = url.split('?').next().unwrap_or("/").to_string();
    let pairs = query(&url);
    let body = body_of(&mut request);
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
            (named, Some(event)) => coordinator
                .status(named)
                .and_then(|status| coordinator.report(&status.job, event))
                .map(|()| json!({"ok": true})),
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
             {}, {}, {}, {}, {}, {}, {}, {}.",
            path::JOIN,
            path::PULL,
            path::REPORT,
            path::COMPLETED,
            path::FAILED,
            path::STATUS,
            path::SUBMIT,
            path::JOBS
        ))),
    };

    match outcome {
        Ok(value) => respond(request, 200, &value),
        Err(error) => respond(request, 400, &json!({"error": error.to_string()})),
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
