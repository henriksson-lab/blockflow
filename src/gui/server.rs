// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The HTTP surface: four endpoints, some static files, and one security
// decision.
//
// Why this crate's own server and not `axum`, and not `tiny_http` either
// ----------------------------------------------------------------------
// The executor is synchronous and parallel over rayon. `axum` would bring
// tokio, and with it a second concurrency model living inside a crate that has
// exactly one — plus about forty transitive crates behind a feature flag on a
// library whose whole `[dependencies]` section is four lines and is defended in
// a comment. What that buys is routing, extractors and middleware, none of
// which four JSON endpoints and a static directory need. That argument is
// unchanged and it is why this is a blocking server on ordinary threads.
//
// What *has* changed is which blocking server. This module used to be
// `tiny_http`'s, and it is [`crate::http`]'s because of a measurement rather
// than a preference: that crate's pool grows a thread only when it observes
// none waiting, and its connection task never returns while the connection
// lives, so connections arriving together queue behind tasks that never finish
// and are read only when some *other* connection closes.
//
// **The progress view was believed too small for that to matter and it is
// not.** The pool's floor is four threads, and a browser holding a page of
// three files and two poll intervals goes past four without trying. Measured
// here, bursts of simultaneous keep-alive connections held open, 20 bursts per
// row:
//
// | connections | bursts leaving someone never answered |
// |---|---|
// | 2, 3, 4 | **0 / 20** |
// | 5 | **14 / 20** |
// | 8 | 16 / 20 |
// | 48 | 15 / 20 |
//
// A loaded machine was no better (10, 18 and 19 of 20 on the same rows), so it
// is not a scheduling accident that a quiet node would escape. A request that
// is never read does not fail — it hangs, and a healthy run's progress view
// freezes with it. `many_connections_at_once_are_all_answered` in `gui::tests`
// is the assertion, and `a_pool_that_never_frees_a_thread_does_starve_someone`
// beside it is what makes that assertion mean something.
//
// [`crate::http`] takes a thread per connection **at accept**, which is the
// simplest dispatch that cannot queue a connection behind a task that never
// finishes, and its header records the two repairs that failed before it.
//
// There is therefore no thread count to set here any more: there was one, it
// defaulted to two, and two threads answering polls is exactly the shape that
// could not be made safe.
//
// The bind default, which is the one thing here with consequences
// ---------------------------------------------------------------
// **Loopback by default, and a non-local bind must be asked for by name.**
//
// A compute node is a shared machine. Binding `0.0.0.0` there publishes a
// user's run — the shape of their data, their op chain, their file paths — to
// everyone else on that network, silently, with no authentication, and the user
// finds out never. That is a different class of mistake from a bad default
// colour, so it is the one place in this module that refuses rather than warns:
// [`Options::bind`] set to a non-loopback address fails unless
// [`Options::allow_public`] is also set, and the binary spells that flag out in
// full with what it does in its help text.
//
// The correct way to see a run on a compute node is an SSH port forward, which
// costs one flag, needs no new port opened anywhere, and inherits the
// authentication that got you onto the machine in the first place:
//
// ```text
// ssh -N -L 8731:127.0.0.1:8731 user@node
// ```
//
// There is no authentication in this server. That is a deliberate consequence
// of the loopback default rather than an oversight — a token on a loopback
// socket protects against nothing that reaching the socket does not already
// imply — and it is also the reason the public bind is gated rather than merely
// discouraged.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::http::{param, Request, Response, Server};

use super::source::{Control, ProgressSource, Window};
use super::wire::{meta_to_json, state_to_json, timeline_to_json, WIRE_VERSION};

/// The default port. Not a registered number; picked to be memorable and
/// unlikely to collide with a Jupyter or a dashboard already forwarded from the
/// same node.
pub const DEFAULT_PORT: u16 = 8731;

/// How many timeline entries one request may take. A ceiling rather than a
/// setting: the point of paging is that a client cannot ask for a response
/// large enough to matter, and a client that wants more asks again.
const MAX_TIMELINE_PAGE: usize = 2_000;
const DEFAULT_TIMELINE_PAGE: usize = 200;

#[derive(Debug, Clone)]
pub struct Options {
    /// Where to listen. Defaults to `127.0.0.1:8731`.
    pub bind: SocketAddr,
    /// Permit a non-loopback bind. See the module header before setting it.
    pub allow_public: bool,
    /// Where the built browser files are. `None` searches; see [`find_assets`].
    pub assets: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), DEFAULT_PORT),
            allow_public: false,
            assets: None,
        }
    }
}

impl Options {
    pub fn bind(mut self, addr: SocketAddr) -> Self {
        self.bind = addr;
        self
    }

    pub fn assets(mut self, dir: impl Into<PathBuf>) -> Self {
        self.assets = Some(dir.into());
        self
    }

    /// Port 0 asks the operating system for a free one, and
    /// [`ServerHandle::addr`] reports which. Used by the tests, and by anyone
    /// running two of these at once.
    pub fn port(mut self, port: u16) -> Self {
        self.bind.set_port(port);
        self
    }
}

/// A running server. Dropping it stops it.
///
/// A thin name over [`crate::http::Server`], kept because what a caller of this
/// module holds is a *progress view*, and because the address it reports is the
/// one to forward.
#[derive(Debug)]
pub struct ServerHandle {
    server: Server,
}

impl ServerHandle {
    /// What was actually bound, which is what to forward. Differs from
    /// `Options::bind` when the port was 0.
    pub fn addr(&self) -> SocketAddr {
        self.server.bound()
    }

    /// The address to paste into a browser once a forward is up.
    pub fn url(&self) -> String {
        format!("http://{}/", self.addr())
    }

    /// Stop accepting and wait for the connection threads to finish.
    ///
    /// Takes up to one [`crate::http::POLL`], because a connection thread wakes
    /// on a read timeout rather than being interrupted — a request in flight is
    /// always finished, never truncated.
    pub fn shutdown(self) {
        self.server.shutdown();
    }
}

/// The bind policy, which is not this module's any more.
///
/// It moved to `net` when a second server appeared that makes the same
/// decision from the opposite direction — a distribution coordinator *must* be
/// reachable by other nodes, so it is exactly this opt-in path, and it must
/// stay exactly this conscious choice. Re-exported rather than reimplemented,
/// because a security default that exists in two places has two chances to
/// drift and no way to notice.
pub use crate::net::check_bind;

/// Start serving `source`.
///
/// Returns once the socket is bound, so a caller may print the URL and then get
/// on with the run. The request threads outlive this call and are stopped by
/// dropping the handle.
pub fn serve(source: Arc<dyn ProgressSource>, options: Options) -> Result<ServerHandle> {
    check_bind(&options.bind, options.allow_public)?;
    if !options.bind.ip().is_loopback() {
        eprintln!(
            "warning: serving on {}, which is reachable from the network. \
             There is no authentication; anyone who can reach this host can watch \
             this run.",
            options.bind
        );
    }

    let listener = TcpListener::bind(options.bind).map_err(|err| {
        Error::backend(format!(
            "could not listen on {}: {err}. If the port is in use, pass a \
             different one, or 0 to let the system choose.",
            options.bind
        ))
    })?;
    let assets = match &options.assets {
        Some(explicit) => Some(explicit.clone()),
        None => find_assets(),
    };

    let server = crate::http::serve(listener, move |request| {
        handle(request, source.as_ref(), assets.as_deref())
    })?;
    Ok(ServerHandle { server })
}

fn json_response(status: u16, body: &Value) -> Response {
    Response::text(status, "application/json; charset=utf-8", &body.to_string())
        // A progress view answers a different thing every time it is asked, and
        // a cached poll is a view that has silently stopped.
        .with_header("Cache-Control", "no-store")
}

fn html_response(status: u16, body: &str) -> Response {
    Response::text(status, "text/html; charset=utf-8", body)
}

fn handle(request: &Request, source: &dyn ProgressSource, assets: Option<&Path>) -> Response {
    let path = request.path();
    let pairs = request.query();
    match path {
        "/api/meta" => json_response(200, &meta_to_json(&source.meta())),
        "/api/state" => json_response(200, &state_to_json(&source.state())),
        "/api/events" => {
            // `before` is what a view asks for — the window ending at the
            // playhead — and `since` is what a consumer accumulating the stream
            // asks for. Neither can be unbounded; `limit` is clamped either way.
            let number =
                |name: &str| param(&pairs, name).and_then(|value| value.parse::<u64>().ok());
            let window = match (number("before"), number("since")) {
                (Some(before), _) => Window::Before(before),
                (None, Some(since)) => Window::Since(since),
                (None, None) => Window::Since(0),
            };
            let limit = param(&pairs, "limit")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(DEFAULT_TIMELINE_PAGE)
                .clamp(1, MAX_TIMELINE_PAGE);
            json_response(200, &timeline_to_json(&source.timeline(window, limit)))
        }
        "/api/control" => {
            let Some(command) = control_from(&pairs) else {
                return json_response(
                    400,
                    &json!({"error": "action must be one of play, pause, seek, step, speed"}),
                );
            };
            let acted = source.control(command);
            let status = if acted { 200 } else { 409 };
            let state = source.state();
            json_response(
                status,
                &json!({
                    "wire": WIRE_VERSION,
                    "accepted": acted,
                    "cursor": state.cursor,
                    "running": state.running,
                }),
            )
        }
        _ => serve_asset(path, assets),
    }
}

fn control_from(pairs: &[(String, String)]) -> Option<Control> {
    let action = param(pairs, "action")?;
    let value = param(pairs, "value");
    Some(match action.as_str() {
        "play" => Control::Play,
        "pause" => Control::Pause,
        "seek" => Control::Seek(value?.parse().ok()?),
        "step" => Control::Step(value.unwrap_or_else(|| "1".to_string()).parse().ok()?),
        "speed" => Control::Speed(value?.parse().ok()?),
        _ => return None,
    })
}

// ------------------------------------------------------------- assets --

/// Where the built browser files are looked for, in order:
///
/// 1. `Options::assets`;
/// 2. `$BLOCKFLOW_GUI_ASSETS`, for a deployment that puts them somewhere else;
/// 3. `webui/dist` beside this crate's manifest, which is where a `trunk build`
///    from a checkout puts them;
/// 4. `webui/dist` under the working directory.
///
/// The manifest path is baked in at compile time, which is right for a tool run
/// from a checkout and wrong for a shipped binary; a shipped binary sets the
/// environment variable. The same trade `animate::renderer_path` makes, for the
/// same reason.
pub fn find_assets() -> Option<PathBuf> {
    for candidate in asset_candidates() {
        if candidate.join("index.html").is_file() {
            return Some(candidate);
        }
    }
    None
}

fn asset_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(from_env) = std::env::var_os("BLOCKFLOW_GUI_ASSETS") {
        out.push(PathBuf::from(from_env));
    }
    out.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("webui/dist"));
    if let Ok(here) = std::env::current_dir() {
        out.push(here.join("webui/dist"));
    }
    out
}

/// A request path, resolved under `root`, or `None` if it tries to leave.
///
/// Only ordinary components are accepted: no `..`, no absolute paths, no root
/// or prefix components. The server is on loopback by default, but "it is only
/// on loopback" is not a reason to serve `/etc/shadow` to a page in a browser
/// that anything on the machine can point at.
fn resolve(root: &Path, path: &str) -> Option<PathBuf> {
    let trimmed = path.trim_start_matches('/');
    let relative = if trimmed.is_empty() {
        "index.html"
    } else {
        trimmed
    };
    let mut out = root.to_path_buf();
    for component in Path::new(relative).components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(out)
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        // Without this exact type the browser refuses to stream-compile the
        // module and the page does not start. It is the one entry in this table
        // that is load-bearing rather than tidy.
        Some("wasm") => "application/wasm",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    }
}

fn serve_asset(path: &str, assets: Option<&Path>) -> Response {
    let Some(root) = assets else {
        // No frontend built. The API still works, so say so and say how to
        // build the page, rather than 404ing at somebody who has done nothing
        // wrong. A panic here would be the worst of all: the run is fine.
        return html_response(200, &missing_assets_page());
    };
    let Some(file) = resolve(root, path) else {
        return json_response(400, &json!({"error": "path leaves the asset directory"}));
    };
    match std::fs::read(&file) {
        Ok(bytes) => Response::new(200, content_type(&file), bytes),
        Err(_) if path == "/" || path.is_empty() => html_response(200, &missing_assets_page()),
        Err(_) => Response::new(404, "text/plain; charset=utf-8", Vec::new()),
    }
}

/// What a browser sees when the page has not been built.
///
/// Deliberately a real page rather than an error: the endpoints below it work,
/// and somebody who has just started a server on a compute node should be told
/// what to do next, not shown a stack trace. Everything it says is checkable
/// from the machine it is served from.
fn missing_assets_page() -> String {
    let looked: Vec<String> = asset_candidates()
        .iter()
        .map(|path| format!("<li><code>{}</code></li>", path.display()))
        .collect();
    format!(
        "<!doctype html><meta charset=\"utf-8\"><title>blockflow — page not built</title>\
         <style>body{{font:15px/1.5 system-ui,sans-serif;max-width:44rem;margin:3rem auto;\
         padding:0 1rem;color:#20232a}}code,pre{{background:#f2f3f5;border-radius:4px}}\
         code{{padding:.1em .35em}}pre{{padding:.8em;overflow-x:auto}}\
         @media(prefers-color-scheme:dark){{body{{background:#15171c;color:#e6e8eb}}\
         code,pre{{background:#23262d}}}}</style>\
         <h1>The server is running; the page is not built.</h1>\
         <p>The browser half of this tool is a separate WebAssembly crate and is not \
         built by <code>cargo</code>. Build it once:</p>\
         <pre>cargo install trunk\ncd webui\ntrunk build --release</pre>\
         <p>Then reload. Files were looked for in:</p><ul>{}</ul>\
         <p>Point somewhere else with <code>--assets DIR</code> or \
         <code>$BLOCKFLOW_GUI_ASSETS</code>.</p>\
         <h2>The data is available regardless</h2>\
         <p><a href=\"/api/meta\">/api/meta</a> · <a href=\"/api/state\">/api/state</a> · \
         <a href=\"/api/events?since=0&amp;limit=20\">/api/events</a></p>",
        looked.join("")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // The query reader moved to `crate::http` with the server. It is still
    // asserted here, and deliberately: what this test protects is that *these
    // endpoints* read a query string the way their documentation says, which is
    // a different claim from the parser being correct.
    use crate::http::{path_of, query};

    /// The policy moved to `net` when a second server started making the same
    /// decision. It is asserted here as well as there, and deliberately: what
    /// this test protects is that *this* module still refuses, which a
    /// re-export nobody called would not.
    #[test]
    fn a_non_loopback_bind_is_refused_unless_asked_for_by_name() {
        let public: SocketAddr = "0.0.0.0:8731".parse().unwrap();
        let error = check_bind(&public, false).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("not a loopback address"), "{text}");
        assert!(
            text.contains("ssh -N -L"),
            "the message must say the remedy: {text}"
        );
        check_bind(&public, true).unwrap();

        for local in ["127.0.0.1:1", "127.0.0.5:80", "[::1]:8731"] {
            check_bind(&local.parse().unwrap(), false).unwrap();
        }
        // A specific interface address is the shape the mistake usually takes,
        // and it is refused like any other non-local bind.
        assert!(check_bind(&"10.0.0.4:8731".parse().unwrap(), false).is_err());
    }

    #[test]
    fn a_request_path_cannot_leave_the_asset_directory() {
        let root = Path::new("/srv/assets");
        assert_eq!(
            resolve(root, "/index.html"),
            Some(PathBuf::from("/srv/assets/index.html"))
        );
        assert_eq!(
            resolve(root, "/"),
            Some(PathBuf::from("/srv/assets/index.html"))
        );
        assert_eq!(
            resolve(root, "/pkg/app_bg.wasm"),
            Some(PathBuf::from("/srv/assets/pkg/app_bg.wasm"))
        );
        assert_eq!(resolve(root, "/../../etc/passwd"), None);
        assert_eq!(resolve(root, "/pkg/../../etc/passwd"), None);
        assert_eq!(
            resolve(root, "//etc/passwd"),
            Some(PathBuf::from("/srv/assets/etc/passwd"))
        );
    }

    #[test]
    fn the_query_string_is_read_the_way_the_endpoints_document() {
        let pairs = query("/api/events?since=42&limit=10");
        assert_eq!(param(&pairs, "since").as_deref(), Some("42"));
        assert_eq!(param(&pairs, "limit").as_deref(), Some("10"));
        assert_eq!(param(&pairs, "nope"), None);
        assert!(query("/api/state").is_empty());
        assert_eq!(path_of("/api/events?since=1"), "/api/events");
        assert_eq!(path_of("/"), "/");
    }

    #[test]
    fn control_commands_parse_from_the_query_string() {
        let of = |url: &str| control_from(&query(url));
        assert_eq!(of("/api/control?action=play"), Some(Control::Play));
        assert_eq!(of("/api/control?action=pause"), Some(Control::Pause));
        assert_eq!(
            of("/api/control?action=seek&value=120"),
            Some(Control::Seek(120))
        );
        assert_eq!(
            of("/api/control?action=step&value=-5"),
            Some(Control::Step(-5))
        );
        assert_eq!(of("/api/control?action=step"), Some(Control::Step(1)));
        assert_eq!(
            of("/api/control?action=speed&value=250"),
            Some(Control::Speed(250.0))
        );
        assert_eq!(of("/api/control?action=rewind"), None);
        assert_eq!(of("/api/control?action=seek"), None);
    }

    #[test]
    fn the_unbuilt_page_says_what_to_do_and_stays_a_page() {
        let page = missing_assets_page();
        assert!(page.contains("trunk build"));
        assert!(page.contains("/api/state"));
        assert!(page.contains("webui/dist"));
    }

    #[test]
    fn every_extension_the_page_needs_has_a_type() {
        assert_eq!(
            content_type(Path::new("a/index.html")),
            "text/html; charset=utf-8"
        );
        assert_eq!(content_type(Path::new("a/app_bg.wasm")), "application/wasm");
        assert_eq!(
            content_type(Path::new("a/app.js")),
            "text/javascript; charset=utf-8"
        );
        assert_eq!(
            content_type(Path::new("a/blob")),
            "application/octet-stream"
        );
    }
}
