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

use std::sync::{Mutex, OnceLock};
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

/// The live heap bytes now and the most held since the previous call, which
/// starts the next interval: what a binary with a counting allocator (the
/// graph's memory test) hands [`set_heap_meter`].
pub type HeapMeter = fn() -> (u64, u64);

static HEAP_METER: OnceLock<HeapMeter> = OnceLock::new();

/// Add the live heap and its peak to every stage line (and start a stage
/// interval at every boundary).
pub fn set_heap_meter(meter: HeapMeter) {
    let _ = HEAP_METER.set(meter);
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
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
        let heap = HEAP_METER.get().map(|meter| meter());
        if let Some(mark) = last.take() {
            let (rss, peak) = memory_mb();
            let heap = heap.map_or(String::new(), |(live, peak)| {
                format!("  heap {:>8.1} MB  peak {:>8.1} MB", mb(live), mb(peak))
            });
            eprintln!(
                "profile {}: {:<28} wall {:>8.2} s  cpu {:>9.2} s  rss {:>8.0} MB  peak {:>8.0} MB{heap}",
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

/// Hands the heap a phase freed back to the system when the next phase
/// starts.
///
/// glibc keeps freed block buffers in its arenas, and a phase whose buffers are
/// a different shape from the last one's cannot reuse all of them, so without
/// this the resident set climbs with the freed-but-kept heap of every phase
/// before it. On the osf-sa3x8 region `segment` peaked at 9.1 GB without it
/// and 8.1 GB with it (7.9 GB with every large buffer handed to `mmap`, which
/// is close to the live set, but at twice the system time). One
/// `malloc_trim` per phase costs milliseconds.
struct HeapRelease;

impl EventListener for HeapRelease {
    fn on_event(&self, event: &Event) {
        if let Event::PhaseStarted { phase } = event {
            if *phase > 0 {
                release_freed_heap();
            }
        }
    }
}

/// Return free heap pages to the system (glibc's `malloc_trim`).
pub fn release_freed_heap() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: `malloc_trim` only returns free heap pages to the system.
    unsafe {
        libc::malloc_trim(0);
    }
}

/// The listeners a planned run is given: a heap release at every phase
/// boundary, and a [`PhaseProfile`] when profiling is on. Call
/// [`Listening::finish`] after the run.
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
        let mut listeners: Vec<std::sync::Arc<dyn EventListener>> =
            vec![std::sync::Arc::new(HeapRelease)];
        if let Some(profile) = &self.profile {
            listeners.push(profile.clone());
        }
        listeners
    }

    pub fn finish(&self) {
        if let Some(profile) = &self.profile {
            profile.finish();
        }
    }
}
