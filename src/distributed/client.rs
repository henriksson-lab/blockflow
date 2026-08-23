// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A JSON-over-HTTP client, in about a page, with no dependency.
//
// Why not a crate
// ---------------
// This crate's `[dependencies]` is four lines and the comment above it says
// every entry is something the *framework* needs. An HTTP client would be the
// largest entry in it, and it would be carrying features for requests this
// protocol never makes: no redirects, no cookies, no compression, no
// multiplexing, no chunked bodies, no TLS. Both ends of this wire are in this
// repository, so the set of requests is closed and small — four paths, a JSON
// body, a JSON reply — and what is left is a request line, three headers and a
// `Content-Length`.
//
// The server half is this crate's own — `distributed::server` routing over
// `crate::http`, which accepts and reads its own connections, one thread per
// connection taken at accept. It was `tiny_http` until a measurement showed
// what that server's connection dispatch does with permanent connections; see
// the note on `Client` below and the header of `crate::http`.
//
// Keep-alive, and why it matters here
// -----------------------------------
// Events are sent **as they happen, unbatched** — the design is explicit that at
// 4-11 events per second a batching window is machinery with nothing to buy. But
// unbatched at a *low* rate is not the same as unbatched at a low cost: a
// connect per event would put a TCP handshake in front of each one. Reusing one
// connection per caller thread keeps the message shape the design asks for while
// making the per-event cost a write and a read. A connection that has gone away
// is reopened once, silently, because "the peer closed an idle keep-alive" is
// not an error worth telling anybody about.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use serde_json::Value;

use crate::error::{Error, Result};

/// A connection to one coordinator.
///
/// Not `Sync`, on purpose: one per calling thread. A worker has three — the
/// loop that pulls, the loop that reports events, and the one that reports
/// completions — so that none of them can ever wait behind another. That is
/// the no-stall property at its smallest scale.
///
/// # The connection is kept, and a measured attempt to stop keeping it
///
/// **A keep-alive connection can go unread for a whole job, and the fault is in
/// the server, not here.** `tiny_http` dispatches each accepted connection to a
/// task-pool thread running `for rq in connection { .. }` — a task that does not
/// return while the connection lives — and its pool grows a thread only when it
/// sees none waiting. Several connections spawned before those threads have
/// picked up their tasks all see the same threads as available, so the surplus
/// is queued with nothing to run it; and because our tasks never return,
/// nothing ever comes back for it. The queued connection is read when some
/// *other* connection closes, which under permanent connections means when the
/// job ends.
///
/// Measured on this tree, and it is not an edge case: three workers open their
/// connections within a few milliseconds of each other, and about one run in
/// six had a worker's pull connection go unread. That worker's report said
/// `ready_ms: 7`, `first_pull_ms: 270`, `pull_connect_ms: 0`, and the
/// coordinator said its slowest *served* pull took **115 microseconds** and its
/// longest wait for the registry lock **17**. Neither process was busy. The
/// request sat unread between them and the worker ran nothing — which every
/// test above it reported as a spread that did not happen or a death that did
/// not occur.
///
/// # Why this does not send `Connection: close`
///
/// It did, briefly, and the measurement is why it does not. One connection per
/// request makes every task finish, so the pool always drains, and it removed
/// the stall completely — 40 runs, no worker unanswered for longer than **5
/// ms**. It also leaves a socket in `TIME_WAIT` per request: **187 000 of them
/// against an ephemeral port range of 28 000**, after which connects queue for
/// a port. The median run was unchanged at 2.5 s and the **slowest went from
/// 2.6 s to 120 s**, with workers exiting non-zero mid-job. A fix whose tail is
/// forty times worse than the fault is not a fix.
///
/// So the connection is kept, and the fault was fixed where it was: a server
/// whose dispatch does not assume tasks finish. That is [`crate::http`], one
/// thread per connection **taken at accept**, shared with the progress view
/// because it turned out to have the identical defect.
///
/// **The detector stayed.** `local::run` still refuses a run in which a
/// worker's first pull went unanswered for longer than a hundred times the
/// slowest pull that coordinator actually served, floored at 50 ms. It is not
/// redundant now that the cause is gone: it is the acceptance criterion for the
/// repair, it is self-calibrating so it needs no number chosen, and it is the
/// only thing that would notice the fault coming back by a different route.
/// Worst observed since: **8 ms**.
pub struct Client {
    addr: SocketAddr,
    stream: Option<BufReader<TcpStream>>,
    timeout: Duration,
    /// How long the last `connect` took.
    ///
    /// Diagnostic, and it exists to split one measurement in two: when a
    /// request takes far longer than the coordinator says it spent on it, the
    /// difference is either the connection being accepted or the request being
    /// read, and only this separates them.
    connected_in: Duration,
}

impl Client {
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            stream: None,
            timeout: Duration::from_secs(30),
            connected_in: Duration::ZERO,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// See [`Client::connected_in`].
    pub fn connected_in(&self) -> Duration {
        self.connected_in
    }

    pub fn get(&mut self, path: &str) -> Result<Value> {
        self.request("GET", path, None)
    }

    pub fn post(&mut self, path: &str, body: &Value) -> Result<Value> {
        self.request("POST", path, Some(body))
    }

    /// One request, with exactly one silent retry on a failed one.
    ///
    /// One, not a loop: a first failure is a connect or a write that did not
    /// take, and reopening is bookkeeping. A *second* failure is the
    /// coordinator being gone, which is a real condition the caller has to see
    /// rather than have retried under it. See the note on [`Client`] for what
    /// the retry can still duplicate and where that is dealt with.
    fn request(&mut self, method: &str, path: &str, body: Option<&Value>) -> Result<Value> {
        match self.attempt(method, path, body) {
            Ok(value) => Ok(value),
            Err(_) => {
                self.stream = None;
                self.attempt(method, path, body)
            }
        }
    }

    fn attempt(&mut self, method: &str, path: &str, body: Option<&Value>) -> Result<Value> {
        if self.stream.is_none() {
            let dialling = std::time::Instant::now();
            let stream = TcpStream::connect_timeout(&self.addr, self.timeout).map_err(|err| {
                Error::backend(format!(
                    "cannot reach the coordinator at {}: {err}. Check that it is running \
                     and that the rendezvous names the address this node can route to.",
                    self.addr
                ))
            })?;
            stream.set_read_timeout(Some(self.timeout)).ok();
            stream.set_write_timeout(Some(self.timeout)).ok();
            stream.set_nodelay(true).ok();
            self.connected_in = dialling.elapsed();
            self.stream = Some(BufReader::new(stream));
        }
        let reader = self.stream.as_mut().expect("connected just above");
        let encoded = body.map(|value| value.to_string()).unwrap_or_default();
        let mut request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: keep-alive\r\n",
            self.addr
        );
        if body.is_some() {
            request.push_str("Content-Type: application/json\r\n");
        }
        request.push_str(&format!("Content-Length: {}\r\n\r\n", encoded.len()));
        request.push_str(&encoded);
        reader
            .get_mut()
            .write_all(request.as_bytes())
            .map_err(|err| Error::backend(format!("sending {method} {path}: {err}")))?;
        reader
            .get_mut()
            .flush()
            .map_err(|err| Error::backend(format!("sending {method} {path}: {err}")))?;
        read_response(reader, path)
    }
}

fn read_response(reader: &mut BufReader<TcpStream>, path: &str) -> Result<Value> {
    let mut status = String::new();
    reader
        .read_line(&mut status)
        .map_err(|err| Error::backend(format!("reading the reply to {path}: {err}")))?;
    if status.is_empty() {
        return Err(Error::backend(format!(
            "the coordinator closed the connection without answering {path}"
        )));
    }
    let code: u16 = status
        .split_whitespace()
        .nth(1)
        .and_then(|field| field.parse().ok())
        .ok_or_else(|| Error::backend(format!("the reply to {path} has no status: {status:?}")))?;

    let mut length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .map_err(|err| Error::backend(format!("reading headers for {path}: {err}")))?;
        if read == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().ok();
            }
        }
    }
    // Every reply this protocol makes is a JSON body with a length. Anything
    // else is a proxy or a mistake, and guessing at it would be worse than
    // saying so.
    let length = length.ok_or_else(|| {
        Error::backend(format!(
            "the reply to {path} has no Content-Length; this client speaks only to a \
             blockflow coordinator"
        ))
    })?;
    let mut body = vec![0u8; length];
    reader
        .read_exact(&mut body)
        .map_err(|err| Error::backend(format!("reading the body of {path}: {err}")))?;
    let text = String::from_utf8_lossy(&body).to_string();
    let value: Value = serde_json::from_str(&text)
        .map_err(|err| Error::backend(format!("the reply to {path} is not JSON: {err}")))?;
    if !(200..300).contains(&code) {
        let message = value
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or(text.as_str());
        return Err(Error::backend(format!(
            "the coordinator refused {path} with {code}: {message}"
        )));
    }
    Ok(value)
}
