// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A blocking HTTP/1.1 server on ordinary threads: accept, read, answer.
//
// Why this crate has one at all
// -----------------------------
// The executor is synchronous and parallel over rayon. An async framework would
// bring a runtime, and with it a second concurrency model living inside a crate
// that has exactly one, plus tens of transitive crates behind a feature flag on
// a library whose `[dependencies]` section is four lines and is defended in a
// comment. What that buys is routing, extractors and middleware, none of which
// a handful of JSON endpoints and a static directory need.
//
// What it does not buy either is correctness, which is the reason this file
// exists rather than a small third-party server. `tiny_http` 0.12 was what both
// of this crate's servers used, and it has a defect that this crate's use is
// exactly the worst case for. Its accept loop hands every connection to a task
// pool as `for rq in connection { .. }` — **a task that does not return while
// the connection lives** — and the pool grows a thread only when it observes
// none waiting. Connections accepted before the woken threads have picked up
// their tasks all read the same threads as free, so the surplus is queued with
// nothing left to run it, and a queued connection is read only when some *other*
// connection closes.
//
// **Both measurements are in this crate's history and both are of a whole
// connection served nothing.** On the distribution path, where every client
// holds a permanent connection, "some other connection closes" means "the job
// ends": a worker ready to ask for work at 7 ms had its first pull answered at
// 270 ms, and with a detector in place that happened in 22 of 40 runs. On the
// progress view, where a browser holds keep-alive connections for a page of
// three files and two poll intervals, the boundary is the pool's own
// `MIN_THREADS`: at four simultaneous connections nothing is lost, and at
// **five, 14 of 20 bursts left a connection that was never read at all** — 16 of
// 20 at eight, 15 of 20 at forty-eight, and no better on a quiet machine than on
// a loaded one. A browser request that is never read does not fail; it hangs,
// and the view of a healthy run freezes.
//
// So this file accepts and reads its own connections: **one thread per
// connection, taken at accept**, which is the simplest dispatch that cannot
// queue a connection behind a task that never finishes. The cost is a thread per
// connection rather than per pool slot, each blocked on a socket read.
//
// Two attempts that failed, recorded so they are not retried
// ----------------------------------------------------------
// * **`Connection: close` on every response.** It makes every task finish and
//   removes the stall completely, and it leaves a socket in `TIME_WAIT` per
//   request — 187 000 of them against an ephemeral port range of 28 000, after
//   which the slowest run went from 2.6 s to 120 s. Keep-alive is honoured here
//   and only declined when the client asks for it to be, which is ordinary
//   HTTP/1.1 rather than a policy.
// * **Polling `accept` on a timer.** A poll interval there is not a poll
//   interval, it is the delay before a connection is accepted at all, and the
//   first version of the distribution server slept 100 ms between attempts and
//   reproduced the very stall it was written to remove, to the millisecond.
//   [`Server::shutdown`] wakes the blocking accept by dialling the port it
//   bound, which is the one thing certain to be listening.
//
// What is deliberately not here
// -----------------------------
// No chunked transfer encoding, no compression, no ranges, no TLS, no
// authentication and no routing. Every response carries a `Content-Length`
// because every body is already in memory; a caller that needed to stream one
// would need a different shape, and neither caller does. Authentication is a
// consequence of the bind policy in [`crate::net::check_bind`] rather than an
// omission — a token on a loopback socket protects against nothing that reaching
// the socket does not already imply.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::error::{Error, Result};

/// How long a connection thread blocks on a read before looking at the stop
/// flag.
///
/// The shutdown latency and nothing else: a connection with a request on it
/// wakes immediately. Long enough that an idle server is not spinning, short
/// enough that a test which stops one does not sit waiting for it.
pub const POLL: Duration = Duration::from_millis(100);

/// The largest request head this server will read before giving up on a
/// connection.
///
/// A ceiling rather than a setting. Nothing either caller serves takes a head
/// anywhere near it; what it stops is a peer that opens a connection and writes
/// bytes without ever completing a head, which would otherwise grow a buffer
/// without bound on a thread that is doing nothing else.
const MAX_HEAD: usize = 64 * 1024;

/// The largest body this server will accept.
///
/// A ceiling, and it is set where nothing legitimate can reach it rather than
/// where a body might plausibly land. **The largest body either caller sends is
/// a job document**, and a job document does not grow with the job: the
/// distribution wire carries a decomposition as *per-phase generators* — slots,
/// names, reach, halo, block edge — that the far end re-derives geometry from,
/// and `wire::decomposition_json` **refuses** a plan whose regions are not
/// expressible that way rather than flattening it. So a submission is
/// `O(phases + ops)`, kilobytes, whether the job has sixty blocks or sixty
/// thousand. The progress view's bodies are empty; its *replies* carry the
/// wasm, and a reply is not bounded here.
///
/// What it stops is a peer that streams without ever completing a message.
/// Nothing is reserved from a declared length — [`Connection::take_request`]
/// waits for the bytes to actually arrive — so the buffer is bounded by
/// `MAX_HEAD + MAX_BODY` and by nothing a peer can claim.
const MAX_BODY: usize = 8 * 1024 * 1024;

// ------------------------------------------------------------- requests --

/// One request, as far as this server understands one.
#[derive(Debug, Clone)]
pub struct Request {
    method: String,
    url: String,
    body: Vec<u8>,
    keep_alive: bool,
}

impl Request {
    pub fn method(&self) -> &str {
        &self.method
    }

    /// The request target, query string and all.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// The path alone.
    pub fn path(&self) -> &str {
        path_of(&self.url)
    }

    /// `?a=1&b=two` as pairs.
    ///
    /// Percent-decoding is deliberately not done: every parameter either caller
    /// takes is a number or a short keyword, so a decoder would be code with no
    /// caller that still had to be got right.
    pub fn query(&self) -> Vec<(String, String)> {
        query(&self.url)
    }

    pub fn body(&self) -> &[u8] {
        &self.body
    }
}

/// The path of a request target: everything before the query string.
pub fn path_of(url: &str) -> &str {
    url.split('?').next().unwrap_or("/")
}

/// `?a=1&b=two` as pairs. See [`Request::query`] for why nothing is decoded.
pub fn query(url: &str) -> Vec<(String, String)> {
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

/// The first value of `name`, or `None`.
pub fn param(pairs: &[(String, String)], name: &str) -> Option<String> {
    pairs
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.clone())
}

// ------------------------------------------------------------ responses --

/// One answer: a status, a content type, any extra headers, and the bytes.
#[derive(Debug, Clone)]
pub struct Response {
    status: u16,
    content_type: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Response {
    pub fn new(status: u16, content_type: impl Into<String>, body: Vec<u8>) -> Self {
        Self {
            status,
            content_type: content_type.into(),
            headers: Vec::new(),
            body,
        }
    }

    pub fn text(status: u16, content_type: impl Into<String>, body: &str) -> Self {
        Self::new(status, content_type, body.as_bytes().to_vec())
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn status(&self) -> u16 {
        self.status
    }

    /// The head and the body as one buffer.
    ///
    /// **One write, not two.** The socket is `nodelay`, so a head and a body
    /// written separately are two packets and two wake-ups on the far side for
    /// every reply.
    fn encode(&self, keep_alive: bool) -> Vec<u8> {
        let mut head = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\n",
            self.status,
            reason(self.status),
            self.content_type,
            self.body.len(),
            if keep_alive { "keep-alive" } else { "close" }
        );
        for (name, value) in &self.headers {
            head.push_str(name);
            head.push_str(": ");
            head.push_str(value);
            head.push_str("\r\n");
        }
        head.push_str("\r\n");
        let mut out = head.into_bytes();
        out.extend_from_slice(&self.body);
        out
    }
}

/// The reason phrases for the statuses this crate's servers return.
///
/// A status not in this table is still sent, with a phrase that says so —
/// clients key on the number, and a wrong word is better than a panic in a
/// server that exists to keep running.
fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

// -------------------------------------------------------------- serving --

/// A running server. Dropping it stops it.
pub struct Server {
    bound: SocketAddr,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl Server {
    /// What was actually bound. Differs from what was asked for when the port
    /// was 0.
    pub fn bound(&self) -> SocketAddr {
        self.bound
    }

    /// Stop accepting and wait for every connection thread to finish.
    pub fn shutdown(mut self) {
        self.halt();
    }

    /// Stop, and wake the accept thread so it notices.
    ///
    /// The dial is the wake-up: `accept` is blocking on purpose — see
    /// [`accept_loop`] — so nothing short of a connection returns it, and the
    /// port this bound is the one thing guaranteed to be listening. The
    /// connection is dropped immediately and the loop, seeing the flag, leaves
    /// without serving it.
    fn halt(&mut self) {
        if self.threads.is_empty() {
            return;
        }
        self.stop.store(true, Ordering::Release);
        let _ = TcpStream::connect_timeout(&self.bound, Duration::from_secs(1));
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.halt();
    }
}

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("bound", &self.bound)
            .finish()
    }
}

/// Serve `listener` with `handler`, on a thread of this server's own.
///
/// Returns once the accept thread is running, so a caller may print the URL and
/// then get on with its work. `handler` is called on the connection's thread,
/// so two requests on two connections run it at once; it must be `Sync` for
/// that reason and must not block on anything a peer controls.
pub fn serve<H>(listener: TcpListener, handler: H) -> Result<Server>
where
    H: Fn(&Request) -> Response + Send + Sync + 'static,
{
    let bound = listener
        .local_addr()
        .map_err(|err| Error::backend(format!("the listener has no address: {err}")))?;
    let stop = Arc::new(AtomicBool::new(false));
    let handler: Arc<dyn Fn(&Request) -> Response + Send + Sync> = Arc::new(handler);
    let accepting = {
        let stop = stop.clone();
        std::thread::Builder::new()
            .name("blockflow-http-accept".to_string())
            .spawn(move || accept_loop(listener, handler, stop))
            .map_err(|err| Error::backend(format!("starting the accept thread: {err}")))?
    };
    Ok(Server {
        bound,
        stop,
        threads: vec![accepting],
    })
}

type Handler = Arc<dyn Fn(&Request) -> Response + Send + Sync>;

/// Accept, and give every connection a thread of its own **at accept**.
///
/// The thread is taken before the connection is read, so a connection can never
/// be waiting for one. That is the whole of the fix described in this file's
/// header, and it is why there is no pool here to size.
///
/// **Blocking, not polled.** See the header for what polling here cost the
/// distribution path. [`Server::shutdown`] wakes this by dialling the port it
/// bound.
fn accept_loop(listener: TcpListener, handler: Handler, stop: Arc<AtomicBool>) {
    let mut connections: Vec<JoinHandle<()>> = Vec::new();
    while !stop.load(Ordering::Acquire) {
        match listener.accept() {
            Ok(_) if stop.load(Ordering::Acquire) => break,
            Ok((stream, _)) => {
                let handler = handler.clone();
                let stop = stop.clone();
                match std::thread::Builder::new()
                    .name("blockflow-http-connection".to_string())
                    .spawn(move || connection_loop(stream, &handler, &stop))
                {
                    Ok(handle) => connections.push(handle),
                    Err(err) => eprintln!("blockflow http: cannot serve a connection: {err}"),
                }
                // Reap what has finished, so a long-lived server does not
                // accumulate handles for connections that closed hours ago.
                connections.retain(|handle| !handle.is_finished());
            }
            // The listener is gone; nothing this thread can do about it, and
            // spinning on the error would burn a core.
            Err(_) => break,
        }
    }
    for handle in connections {
        let _ = handle.join();
    }
}

/// One connection, until it closes or the server stops.
fn connection_loop(stream: TcpStream, handler: &Handler, stop: &AtomicBool) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(POLL));
    let mut connection = Connection::new(stream);
    while !stop.load(Ordering::Acquire) {
        match connection.next_request() {
            Ok(Some(request)) => {
                let keep_alive = request.keep_alive;
                // A panic in a handler must not take the connection with it
                // silently, and must not leave the peer waiting for an answer
                // that is never coming: whatever the run is doing is unaffected,
                // and a browser blocked on a dead request looks exactly like a
                // stalled run.
                let answer = match catch_unwind(AssertUnwindSafe(|| handler(&request))) {
                    Ok(response) => response,
                    Err(_) => {
                        eprintln!(
                            "blockflow http: a request handler panicked on {} {}; \
                             the run is unaffected",
                            request.method(),
                            request.url()
                        );
                        Response::text(
                            500,
                            "text/plain; charset=utf-8",
                            "the request handler panicked; the run is unaffected\n",
                        )
                    }
                };
                if connection.answer(&answer, keep_alive).is_err() || !keep_alive {
                    break;
                }
            }
            // Nothing has arrived yet. The only reason this returns rather than
            // blocking forever is so the stop flag gets looked at.
            Ok(None) => continue,
            Err(_) => break,
        }
    }
}

/// A connection being read.
///
/// Bytes are accumulated rather than parsed as they arrive, because a read can
/// return part of a message — and, here, can time out in the middle of one. A
/// buffer that survives the timeout is what makes "look at the stop flag every
/// [`POLL`]" safe to do half way through a request.
struct Connection {
    stream: TcpStream,
    buffer: Vec<u8>,
}

impl Connection {
    fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            buffer: Vec::with_capacity(1024),
        }
    }

    /// The next complete request, or `None` if none has arrived yet.
    fn next_request(&mut self) -> std::io::Result<Option<Request>> {
        loop {
            if let Some(request) = self.take_request()? {
                return Ok(Some(request));
            }
            let mut chunk = [0u8; 4096];
            match self.stream.read(&mut chunk) {
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "the peer closed the connection",
                    ))
                }
                Ok(read) => self.buffer.extend_from_slice(&chunk[..read]),
                Err(ref err)
                    if matches!(
                        err.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    return Ok(None)
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// Parse one request out of the buffer if a whole one is there.
    fn take_request(&mut self) -> std::io::Result<Option<Request>> {
        let Some(head_end) = find(&self.buffer, b"\r\n\r\n") else {
            if self.buffer.len() > MAX_HEAD {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "a request head that never ended",
                ));
            }
            return Ok(None);
        };
        let head = String::from_utf8_lossy(&self.buffer[..head_end]).into_owned();
        let mut lines = head.split("\r\n");
        let request_line = lines.next().unwrap_or_default();
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or("GET").to_string();
        let url = parts.next().unwrap_or("/").to_string();
        let version = parts.next().unwrap_or("HTTP/1.1").to_string();
        let mut length = 0usize;
        let mut connection_header: Option<String> = None;
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                if name.eq_ignore_ascii_case("content-length") {
                    length = value.trim().parse().unwrap_or(0);
                } else if name.eq_ignore_ascii_case("connection") {
                    connection_header = Some(value.trim().to_ascii_lowercase());
                }
            }
        }
        if length > MAX_BODY {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "a request body larger than this server accepts",
            ));
        }
        let body_at = head_end + 4;
        if self.buffer.len() < body_at + length {
            return Ok(None);
        }
        let body = self.buffer[body_at..body_at + length].to_vec();
        self.buffer.drain(..body_at + length);
        Ok(Some(Request {
            method,
            url,
            body,
            keep_alive: keeps_alive(&version, connection_header.as_deref()),
        }))
    }

    fn answer(&mut self, response: &Response, keep_alive: bool) -> std::io::Result<()> {
        self.stream.write_all(&response.encode(keep_alive))?;
        self.stream.flush()
    }
}

/// Whether the connection survives this request, by the client's rules.
///
/// HTTP/1.1 keeps a connection unless the client says `close`; HTTP/1.0 closes
/// it unless the client says `keep-alive`. Both directions matter here: a
/// browser is the first case and this crate's own one-request-per-connection
/// test client is the second, and a server that ignored the second would leave
/// it reading a socket that is never going to end.
fn keeps_alive(version: &str, connection: Option<&str>) -> bool {
    let asked = connection.map(|value| {
        (
            value.split(',').any(|token| token.trim() == "close"),
            value.split(',').any(|token| token.trim() == "keep-alive"),
        )
    });
    match (version.eq_ignore_ascii_case("HTTP/1.0"), asked) {
        (_, Some((true, _))) => false,
        (true, Some((_, true))) => true,
        (true, _) => false,
        (false, _) => true,
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_query_string_is_read_as_pairs() {
        let pairs = query("/api/events?since=42&limit=10");
        assert_eq!(param(&pairs, "since").as_deref(), Some("42"));
        assert_eq!(param(&pairs, "limit").as_deref(), Some("10"));
        assert_eq!(param(&pairs, "nope"), None);
        assert!(query("/api/state").is_empty());
        // A bare key is a key with an empty value, not a missing one.
        assert_eq!(param(&query("/a?flag"), "flag").as_deref(), Some(""));
        assert_eq!(path_of("/api/events?since=1"), "/api/events");
        assert_eq!(path_of("/"), "/");
    }

    /// Both directions of the rule, because a server that got the HTTP/1.0 half
    /// wrong would leave a one-request client reading a socket forever.
    #[test]
    fn keep_alive_follows_the_version_and_then_the_header() {
        assert!(keeps_alive("HTTP/1.1", None));
        assert!(!keeps_alive("HTTP/1.1", Some("close")));
        assert!(!keeps_alive("HTTP/1.0", None));
        assert!(keeps_alive("HTTP/1.0", Some("keep-alive")));
        assert!(!keeps_alive("HTTP/1.0", Some("close")));
        // A list, which is what a proxy adds to.
        assert!(!keeps_alive("HTTP/1.1", Some("te, close")));
        assert!(keeps_alive("HTTP/1.1", Some("te, keep-alive")));
    }

    #[test]
    fn a_response_carries_its_length_and_its_extra_headers() {
        let response =
            Response::text(200, "application/json", "{}").with_header("Cache-Control", "no-store");
        let bytes = response.encode(true);
        let text = String::from_utf8_lossy(&bytes).into_owned();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
        assert!(text.contains("Content-Length: 2\r\n"), "{text}");
        assert!(text.contains("Cache-Control: no-store\r\n"), "{text}");
        assert!(text.contains("Connection: keep-alive\r\n"), "{text}");
        assert!(text.ends_with("\r\n\r\n{}"), "{text}");

        let closing = String::from_utf8_lossy(&response.encode(false)).into_owned();
        assert!(closing.contains("Connection: close\r\n"), "{closing}");
    }

    /// A body is bytes, not text: the progress view serves a quarter-megabyte
    /// of WebAssembly through this and a lossy conversion anywhere on that path
    /// is a page that does not start.
    #[test]
    fn a_binary_body_survives_encoding() {
        let body: Vec<u8> = (0..=255u8).collect();
        let bytes = Response::new(200, "application/wasm", body.clone()).encode(true);
        let head_end = find(&bytes, b"\r\n\r\n").expect("a head");
        assert_eq!(&bytes[head_end + 4..], &body[..]);
    }
}
