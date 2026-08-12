// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Turning an exported order log into a movie.
//
// Where the language boundary is, and why it is here
// --------------------------------------------------
// The capability has two halves and they are split on purpose:
//
// * **The log is produced in Rust.** `export::write_order_log_json` is the
//   library function — it needs the executor, the decomposition and the event
//   stream, all of which live here, and it must work on a machine with no
//   Python at all. Nothing about producing a log is optional.
// * **The picture is drawn in Python.** The renderer is manim, and manim is a
//   Python library. Reimplementing a scene graph, a rasteriser and a video
//   muxer in Rust to avoid a subprocess would be a much larger and much worse
//   piece of software than the twenty lines below.
//
// So this module is the seam, and it holds exactly one idea: *how to invoke the
// bundled renderer*. The renderer itself ships with this crate, in
// `tools/animate_block_progress.py`, because the crate should own the whole
// capability. It previously lived in the repository that consumes this one,
// which pointed the dependency the wrong way — a crate cannot sensibly ask its
// own consumer for a script.
//
// **manim is not a dependency of this crate.** It is not in `Cargo.toml`, it is
// not checked at build time, and every test in this file passes on a machine
// with no Python. Export works; rendering is opt-in and fails with a message
// that says what to install. That asymmetry is deliberate: a diagnostic
// animation must never be able to break a build.
//
// Usage
// -----
// ```no_run
// use blockflow::animate::{render, RenderRequest, View};
//
// let request = RenderRequest::new("run.json").view(View::Volume3d);
// let movie = render(&request)?;               // -> media/videos/.../*.mp4
// # Ok::<(), blockflow::Error>(())
// ```
//
// `src/bin/block_animate.rs` is a thin shell over exactly this, so the CLI and
// the library reach the same capability by the same path.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

use crate::error::{Error, Result};

/// Which picture to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// One log, drawn as the block lattice in three dimensions, with the camera
    /// rotating around it.
    Volume3d,
    /// The block grid flattened to one square per cell. Accepts two logs and
    /// draws them side by side, which is how a claim that the animation follows
    /// the schedule is checked: two schedules over one decomposition must look
    /// different.
    Grid2d,
}

impl View {
    fn flag(self) -> &'static str {
        match self {
            Self::Volume3d => "--view=3d",
            Self::Grid2d => "--view=2d",
        }
    }
}

/// Frame size and rate, in the renderer's own vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// 854x480. The default: this is a diagnostic, not a trailer.
    Low,
    /// 1280x720.
    Medium,
    /// 1920x1080. Costs roughly four times `Low` per frame in three dimensions.
    High,
}

impl Quality {
    fn flag(self) -> &'static str {
        match self {
            Self::Low => "l",
            Self::Medium => "m",
            Self::High => "h",
        }
    }
}

/// Everything the renderer needs.
///
/// The defaults are chosen so that a first render finishes while you wait: at
/// the measured rate, the CLI's default lattice at `max_steps: 40` is a few
/// minutes. Raise the steps for a smoother picture and pay linearly.
#[derive(Debug, Clone)]
pub struct RenderRequest {
    /// One log for `Volume3d`; one or two for `Grid2d`.
    pub logs: Vec<PathBuf>,
    /// Base name of the movie, without extension.
    pub output: String,
    pub view: View,
    pub quality: Quality,
    /// Ceiling on rendered steps. Events are batched to fit, so a large log
    /// animates in comparable wall time at coarser granularity.
    pub max_steps: usize,
    /// Seconds of video per step.
    pub seconds_per_step: f64,
    /// Frames per second. `None` leaves the renderer's own default.
    pub fps: Option<u32>,
    /// Where manim puts its scratch and its output tree. `None` means `media/`
    /// under the working directory.
    pub media_dir: Option<PathBuf>,
    /// Interpreter to run. `None` means `$BLOCKFLOW_PYTHON`, else `python3`.
    pub python: Option<PathBuf>,
    /// The renderer script. `None` means the bundled one; see [`renderer_path`].
    pub renderer: Option<PathBuf>,
    /// Render a lattice the renderer considers too large to be practical. The
    /// ceiling exists because an accidental overnight render is a real cost;
    /// see the renderer's header for the measurement behind the number.
    pub allow_large: bool,
    /// Print the renderer's own output as it goes.
    pub verbose: bool,
}

impl RenderRequest {
    /// One log, three dimensions, low quality.
    pub fn new(log: impl Into<PathBuf>) -> Self {
        Self {
            logs: vec![log.into()],
            output: "block_flight".to_string(),
            view: View::Volume3d,
            quality: Quality::Low,
            max_steps: 40,
            seconds_per_step: 0.15,
            fps: None,
            media_dir: None,
            python: None,
            renderer: None,
            allow_large: false,
            verbose: false,
        }
    }

    pub fn view(mut self, view: View) -> Self {
        self.view = view;
        self
    }

    pub fn output(mut self, name: impl Into<String>) -> Self {
        self.output = name.into();
        self
    }

    pub fn quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }

    pub fn with_log(mut self, log: impl Into<PathBuf>) -> Self {
        self.logs.push(log.into());
        self
    }

    /// The argument vector, script excluded. Separated out so it can be tested
    /// without running anything.
    pub fn args(&self) -> Vec<OsString> {
        let mut args: Vec<OsString> = Vec::new();
        for log in &self.logs {
            args.push(log.clone().into_os_string());
        }
        args.push(self.view.flag().into());
        args.push("-o".into());
        args.push(self.output.clone().into());
        args.push("-q".into());
        args.push(self.quality.flag().into());
        args.push(format!("--max-steps={}", self.max_steps).into());
        args.push(format!("--seconds-per-step={}", self.seconds_per_step).into());
        if let Some(fps) = self.fps {
            args.push(format!("--fps={fps}").into());
        }
        if let Some(dir) = &self.media_dir {
            args.push("--media-dir".into());
            args.push(dir.clone().into_os_string());
        }
        if self.allow_large {
            args.push("--allow-large".into());
        }
        args
    }
}

/// The bundled renderer, in the order the seam looks for it.
///
/// 1. `$BLOCKFLOW_RENDERER`, for a caller who has their own copy;
/// 2. `tools/animate_block_progress.py` beside this crate's manifest, which is
///    where it ships and where a `cargo run` from a checkout finds it;
/// 3. `tools/animate_block_progress.py` under the working directory.
///
/// The manifest path is baked in at compile time. That is fine for a developer
/// tool run from a checkout and would not be fine for a shipped binary; a
/// shipped binary should set `$BLOCKFLOW_RENDERER` or pass
/// `RenderRequest::renderer`.
pub fn renderer_path() -> Result<PathBuf> {
    let mut tried: Vec<PathBuf> = Vec::new();
    if let Some(from_env) = std::env::var_os("BLOCKFLOW_RENDERER") {
        let path = PathBuf::from(from_env);
        if path.is_file() {
            return Ok(path);
        }
        tried.push(path);
    }
    for base in [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    ] {
        let path = base.join("tools").join("animate_block_progress.py");
        if path.is_file() {
            return Ok(path);
        }
        tried.push(path);
    }
    Err(Error::InvalidArgument(format!(
        "the bundled renderer was not found. Looked at: {}. Set $BLOCKFLOW_RENDERER \
         to its path, or pass RenderRequest::renderer.",
        tried
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

/// The interpreter to run: `RenderRequest::python`, else `$BLOCKFLOW_PYTHON`,
/// else `python3` from `PATH`.
fn interpreter(request: &RenderRequest) -> PathBuf {
    if let Some(explicit) = &request.python {
        return explicit.clone();
    }
    match std::env::var_os("BLOCKFLOW_PYTHON") {
        Some(from_env) => PathBuf::from(from_env),
        None => PathBuf::from("python3"),
    }
}

/// The movie path, as the renderer reported it on its last `wrote …` line.
fn wrote_path(stdout: &str) -> Option<PathBuf> {
    stdout
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix("wrote "))
        .map(|rest| PathBuf::from(rest.trim()))
}

fn check(request: &RenderRequest) -> Result<()> {
    if request.logs.is_empty() {
        return Err(Error::InvalidArgument("render: no log to draw".to_string()));
    }
    if request.view == View::Volume3d && request.logs.len() > 1 {
        return Err(Error::InvalidArgument(
            "render: the three-dimensional view draws one log. Two logs side by \
             side is what View::Grid2d is for."
                .to_string(),
        ));
    }
    if request.logs.len() > 2 {
        return Err(Error::InvalidArgument(
            "render: at most two logs; more than two panels stops being legible".to_string(),
        ));
    }
    for log in &request.logs {
        if !log.is_file() {
            return Err(Error::InvalidArgument(format!(
                "render: no such log: {}",
                log.display()
            )));
        }
    }
    Ok(())
}

/// Run the bundled renderer over an exported order log.
///
/// Returns the movie's path. Every failure mode names its remedy: a missing
/// interpreter, a missing manim and a renderer that refused a lattice as too
/// large are three different messages, because they need three different
/// actions.
pub fn render(request: &RenderRequest) -> Result<PathBuf> {
    check(request)?;
    let script = match &request.renderer {
        Some(explicit) => explicit.clone(),
        None => renderer_path()?,
    };
    let python = interpreter(request);

    let mut command = Command::new(&python);
    command.arg(&script);
    command.args(request.args());
    let output = command.output().map_err(|err| {
        Error::Backend(format!(
            "could not run {} ({err}). Rendering needs a Python interpreter with \
             manim installed; set $BLOCKFLOW_PYTHON to one. Exporting the log \
             needs nothing.",
            python.display()
        ))
    })?;

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if request.verbose {
        print!("{stdout}");
        eprint!("{stderr}");
    }
    if !output.status.success() {
        return Err(Error::Backend(format!(
            "the renderer failed ({}):\n{}",
            output.status,
            if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            }
        )));
    }
    wrote_path(&stdout).ok_or_else(|| {
        Error::Backend(format!(
            "the renderer reported no output file. Its output was:\n{}",
            stdout.trim()
        ))
    })
}

/// Whether rendering would work here, without rendering anything.
///
/// A caller that wants to offer the movie only when it can actually be made —
/// a CLI printing a hint, a test deciding whether to skip — asks this. It runs
/// the renderer's own dependency check, so a `true` means manim imported, not
/// that a file exists.
pub fn renderer_available(request: &RenderRequest) -> bool {
    let script = match request.renderer.clone().or_else(|| renderer_path().ok()) {
        Some(path) => path,
        None => return false,
    };
    Command::new(interpreter(request))
        .arg(script)
        .arg("--probe")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// The renderer's practical ceiling on lattice size, in blocks.
///
/// Not a limit of the format or of the export — a log of any size exports
/// fine — but of what manim's three-dimensional renderer will finish in a
/// sitting. Measured there at roughly **0.12 s per block per step**, so this is
/// already about an hour at the default step count and exists to stop somebody
/// starting an overnight render by accident. A round number, not a threshold
/// anything changes at; the renderer's header carries the measurement and
/// [`the_block_ceiling_matches_the_renderer`] keeps the two equal.
pub const VOLUME3D_BLOCK_CEILING: usize = 512;

/// Whether a lattice of `grid` blocks is within the ceiling.
pub fn within_volume3d_ceiling(grid: [usize; 3]) -> bool {
    grid[0].saturating_mul(grid[1]).saturating_mul(grid[2]) <= VOLUME3D_BLOCK_CEILING
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_request_becomes_the_argument_vector_the_renderer_documents() {
        let request = RenderRequest::new("/tmp/run.json")
            .output("flight")
            .quality(Quality::Medium);
        let args: Vec<String> = request
            .args()
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(args[0], "/tmp/run.json");
        assert!(args.contains(&"--view=3d".to_string()));
        assert!(args.contains(&"-o".to_string()));
        assert!(args.contains(&"flight".to_string()));
        assert!(args.contains(&"m".to_string()));
        assert!(args.iter().any(|arg| arg.starts_with("--max-steps=")));
        assert!(args
            .iter()
            .any(|arg| arg.starts_with("--seconds-per-step=")));
        // Not asked for, so not passed: the renderer's defaults stay the
        // renderer's business.
        assert!(!args.iter().any(|arg| arg.starts_with("--fps")));
        assert!(!args.contains(&"--allow-large".to_string()));
    }

    #[test]
    fn the_two_dimensional_view_takes_two_logs_and_the_three_dimensional_one_does_not() {
        let two = RenderRequest::new("a.json").with_log("b.json");
        assert_eq!(two.logs.len(), 2);
        let error = check(&two.clone().view(View::Volume3d)).unwrap_err();
        assert!(
            format!("{error}").contains("one log"),
            "message should say what the three-dimensional view accepts: {error}"
        );
        // Two dimensions accepts the pair, and then fails on the files not
        // existing rather than on the count.
        let error = check(&two.view(View::Grid2d)).unwrap_err();
        assert!(format!("{error}").contains("no such log"), "{error}");
    }

    #[test]
    fn a_missing_renderer_says_where_it_looked() {
        // Point the environment override at nothing, so the search runs out.
        let request = RenderRequest::new("run.json");
        let missing = PathBuf::from("/nonexistent/renderer.py");
        let error = render(&RenderRequest {
            renderer: Some(missing.clone()),
            ..request
        })
        .unwrap_err();
        // The log check runs first; that is the message we get, and it is the
        // right one — there is no point reporting the renderer for a run that
        // has nothing to draw.
        assert!(format!("{error}").contains("no such log"), "{error}");
    }

    #[test]
    fn the_output_path_is_read_from_the_renderers_last_report() {
        let stdout = "some noise\nwrote media/videos/a/480p15/flight.mp4\n";
        assert_eq!(
            wrote_path(stdout),
            Some(PathBuf::from("media/videos/a/480p15/flight.mp4"))
        );
        assert_eq!(wrote_path("nothing here"), None);
    }

    #[test]
    fn the_bundled_renderer_ships_with_the_crate() {
        let path = renderer_path().expect("the renderer ships in tools/");
        assert!(path.ends_with("animate_block_progress.py"));
    }

    /// The ceiling is quoted in two places — a constant here and a default in
    /// the renderer — and a reader who finds them disagreeing cannot tell which
    /// one the code obeys.
    #[test]
    fn the_block_ceiling_matches_the_renderer() {
        let script = renderer_path().unwrap();
        let text = std::fs::read_to_string(script).unwrap();
        let needle = format!("BLOCK_CEILING = {VOLUME3D_BLOCK_CEILING}");
        assert!(
            text.contains(&needle),
            "the renderer must declare `{needle}`; the two ceilings must agree"
        );
        assert!(within_volume3d_ceiling([6, 6, 6]));
        assert!(!within_volume3d_ceiling([10, 10, 10]));
    }
}
