// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Opt-in per-stage profiling: with `VESSEL_GRAPH_PROFILE` set, each stage
//! boundary prints the wall time and CPU time since the previous boundary, the
//! resident set now and its peak since the previous boundary (Linux: the peak
//! is reset through `/proc/self/clear_refs` at every boundary).
//!
//! [`PhaseProfile`] is the same for the phases of a planned run, as an event
//! listener: a boundary at every `PhaseStarted`. Phases of one plan may
//! overlap, so a phase's figures are those of the stretch from its start to
//! the next phase's start.
//!
//! Nothing here changes a result; without the variable every call is a no-op.

use std::sync::Mutex;
use std::time::Instant;

use blockflow::listener::EventListener;
use blockflow::log::Event;

/// Whether profiling is on.
pub fn enabled() -> bool {
    std::env::var_os("VESSEL_GRAPH_PROFILE").is_some()
}

/// User plus system CPU seconds of this process, from `/proc/self/stat`.
fn cpu_seconds() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return 0.0;
    };
    // The command name is in parentheses and may hold spaces; the fields
    // after the closing parenthesis are fixed: utime and stime are the 12th
    // and 13th of them.
    let Some(tail) = stat.rsplit_once(')').map(|(_, tail)| tail) else {
        return 0.0;
    };
    let fields: Vec<&str> = tail.split_whitespace().collect();
    let ticks = |index: usize| -> f64 {
        fields
            .get(index)
            .and_then(|field| field.parse::<f64>().ok())
            .unwrap_or(0.0)
    };
    (ticks(11) + ticks(12)) / 100.0
}

/// `VmRSS` and `VmHWM` in MB.
fn memory_mb() -> (f64, f64) {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return (0.0, 0.0);
    };
    let field = |name: &str| -> f64 {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|kb| kb.parse::<f64>().ok())
            .map_or(0.0, |kb| kb / 1024.0)
    };
    (field("VmRSS:"), field("VmHWM:"))
}

fn reset_peak() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

struct Mark {
    what: String,
    wall: Instant,
    cpu: f64,
}

/// The `graph` subcommand's stages.
pub static GRAPH: Profile = Profile::new("graph");

/// A sequence of stage boundaries.
pub struct Profile {
    scope: &'static str,
    last: Mutex<Option<Mark>>,
}

impl Profile {
    pub const fn new(scope: &'static str) -> Self {
        Self {
            scope,
            last: Mutex::new(None),
        }
    }

    /// Close the stage that was running (if any) and open `what`.
    pub fn stage(&self, what: impl Into<String>) {
        if !enabled() {
            return;
        }
        let what = what.into();
        let now = Instant::now();
        let cpu = cpu_seconds();
        let mut last = self
            .last
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(mark) = last.take() {
            let (rss, peak) = memory_mb();
            eprintln!(
                "profile {}: {:<28} wall {:>8.2} s  cpu {:>9.2} s  rss {:>8.0} MB  peak {:>8.0} MB",
                self.scope,
                mark.what,
                now.duration_since(mark.wall).as_secs_f64(),
                cpu - mark.cpu,
                rss,
                peak
            );
        }
        reset_peak();
        *last = Some(Mark {
            what,
            wall: now,
            cpu,
        });
    }

    /// Close the running stage.
    pub fn finish(&self) {
        if !enabled() {
            return;
        }
        self.stage("");
        *self
            .last
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = None;
    }
}

/// A [`Profile`] with a boundary at every phase start of a planned run.
pub struct PhaseProfile {
    profile: Profile,
    names: Vec<String>,
    started: Instant,
}

impl PhaseProfile {
    /// `names[p]` labels phase `p` (missing ones are numbered).
    pub fn new(scope: &'static str, names: Vec<String>) -> Self {
        Self {
            profile: Profile::new(scope),
            names,
            started: Instant::now(),
        }
    }

    pub fn finish(&self) {
        self.profile.finish();
    }
}

impl EventListener for PhaseProfile {
    fn on_event(&self, event: &Event) {
        if std::env::var_os("VESSEL_GRAPH_PROFILE_TASKS").is_some() {
            let now = self.started.elapsed().as_secs_f64();
            match event {
                Event::TaskAdmitted { phase, index } => {
                    eprintln!("task {now:.3} admit {phase} {index:?}")
                }
                Event::BlockWritten { phase, index, .. } => {
                    eprintln!("task {now:.3} written {phase} {index:?}")
                }
                _ => {}
            }
        }
        if let Event::PhaseStarted { phase } = event {
            let name = self
                .names
                .get(*phase)
                .cloned()
                .unwrap_or_else(|| format!("phase {phase}"));
            self.profile.stage(format!("{phase:>2} {name}"));
        }
    }
}

/// A label per phase of an assembled plan: the fragment or iterative op's
/// name, or the pixel phase's op names.
pub fn phase_names(assembly: &blockflow::assemble::Assembly) -> Vec<String> {
    use blockflow::fragment::PhaseWork;
    assembly
        .work()
        .iter()
        .zip(&assembly.decomposition.phases)
        .map(|(work, phase)| match work {
            PhaseWork::Fragments(op) => op.name().to_string(),
            PhaseWork::Iterate(op) => op.name().to_string(),
            PhaseWork::IterateReduce(op) => op.name().to_string(),
            PhaseWork::Pixels => {
                let mut names = phase.names.join("+");
                if names.len() > 40 {
                    names.truncate(40);
                }
                names
            }
        })
        .collect()
}

/// The listeners a planned run is given: a [`PhaseProfile`] when profiling is
/// on, none otherwise. Call [`Listening::finish`] after the run.
pub struct Listening {
    profile: Option<std::sync::Arc<PhaseProfile>>,
}

impl Listening {
    pub fn new(scope: &'static str, assembly: &blockflow::assemble::Assembly) -> Self {
        Self {
            profile: enabled()
                .then(|| std::sync::Arc::new(PhaseProfile::new(scope, phase_names(assembly)))),
        }
    }

    pub fn listeners(&self) -> Vec<std::sync::Arc<dyn EventListener>> {
        match &self.profile {
            Some(profile) => vec![profile.clone() as std::sync::Arc<dyn EventListener>],
            None => Vec::new(),
        }
    }

    pub fn finish(&self) {
        if let Some(profile) = &self.profile {
            profile.finish();
        }
    }
}
