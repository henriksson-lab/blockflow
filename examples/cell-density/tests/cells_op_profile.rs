// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Where the detector spends its time: per-phase wall time and per-op CPU,
//! from the executor's own events. A measurement, so ignored; it runs on the
//! real crop when `CELL_DENSITY_TILE` (or the default `/husky` path) exists
//! and on a synthetic volume otherwise, once per watershed (tiled, whole).
//! `CELL_DENSITY_WINDOW=z,y,x;z,y,x` (start; shape) replaces the README's
//! crop window, and `CELL_DENSITY_CONFIGS` (a comma list of `tiled`,
//! `tiled128`, `whole`; default all three) picks the runs. Each phase reports
//! its wall time, the process's CPU time over it, and the largest resident set
//! seen while it ran (sampled every 50 ms from `/proc/self`). Run with
//! `--release --ignored --nocapture`. `CELL_DENSITY_CACHE_BYTES` sizes the
//! chunk cache (the environment's default otherwise), and `CELL_DENSITY_BLOCK`
//! (`z,y,x`) replaces every run's compute block; `CELL_DENSITY_COMPRESSION=none`
//! stores the intermediates uncompressed.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::log::Event;
use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{Compression, CompressionPolicy};
use blockflow::{EventListener, ZarrEnvironment};
use cell_density::detect::{CellPlan, PlanOptions};
use cell_density::params::{CellParams, Watershed};

/// The process's resident set and CPU time (user + system), from `/proc`.
fn rss_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|rest| {
            rest.trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<u64>()
                .ok()
        })
        .map_or(0, |kb| kb * 1024)
}

fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // Fields after the parenthesised command name; utime and stime are the
    // 14th and 15th fields overall, in clock ticks (100 per second here).
    let rest = stat.rsplit_once(')').map_or("", |(_, rest)| rest);
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let ticks = |i: usize| {
        fields
            .get(i)
            .and_then(|f| f.parse::<f64>().ok())
            .unwrap_or(0.0)
    };
    (ticks(11) + ticks(12)) / 100.0
}

#[derive(Default)]
struct Profile {
    by_op: Mutex<BTreeMap<String, (u64, usize)>>,
    /// Per phase: start, CPU at start, peak RSS while it ran, its op names.
    starts: Mutex<Vec<(usize, Instant, f64)>>,
    peaks: Mutex<BTreeMap<usize, u64>>,
    ops: Mutex<BTreeMap<usize, std::collections::BTreeSet<String>>>,
}

impl EventListener for Profile {
    fn on_event(&self, event: &Event) {
        match event {
            Event::PhaseStarted { phase } => self
                .starts
                .lock()
                .expect("no panic holds this")
                .push((*phase, Instant::now(), cpu_seconds())),
            Event::OpApplied {
                op,
                duration_ns,
                phase,
                ..
            } => {
                self.ops
                    .lock()
                    .expect("no panic holds this")
                    .entry(*phase)
                    .or_default()
                    .insert(op.clone());
                let mut by_op = self.by_op.lock().expect("no panic holds this");
                let slot = by_op.entry(op.clone()).or_insert((0, 0));
                slot.0 += duration_ns;
                slot.1 += 1;
            }
            _ => {}
        }
    }
}

fn run(plan: &CellPlan, env: &dyn Environment, label: &str) {
    let profile = Arc::new(Profile::default());
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sampler = {
        let (profile, done) = (profile.clone(), done.clone());
        std::thread::spawn(move || {
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                let phase = profile.starts.lock().unwrap().last().map(|s| s.0);
                if let Some(phase) = phase {
                    let rss = rss_bytes();
                    let mut peaks = profile.peaks.lock().unwrap();
                    let slot = peaks.entry(phase).or_insert(0);
                    *slot = (*slot).max(rss);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        })
    };
    let started = Instant::now();
    let cpu_started = cpu_seconds();
    plan.execute(
        "profile",
        env,
        &plan.hints(None, &[]),
        &[profile.clone() as Arc<dyn EventListener>],
    )
    .expect("the run");
    let total = started.elapsed();
    let cpu_total = cpu_seconds() - cpu_started;
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    sampler.join().unwrap();
    eprintln!(
        "\n{label}: {:.2} s wall, {cpu_total:.1} s cpu",
        total.as_secs_f64()
    );
    let starts = profile.starts.lock().unwrap().clone();
    let peaks = profile.peaks.lock().unwrap().clone();
    let ops = profile.ops.lock().unwrap().clone();
    let cpu_end = cpu_started + cpu_total;
    for (index, (phase, at, cpu)) in starts.iter().enumerate() {
        let (end, cpu_next) = starts
            .get(index + 1)
            .map(|(_, next, cpu)| (*next, *cpu))
            .unwrap_or((started + total, cpu_end));
        let names: Vec<String> = ops.get(phase).into_iter().flatten().cloned().collect();
        eprintln!(
            "  phase {phase:>2}: {:>8.3} s wall {:>8.1} s cpu {:>6.2} GB peak  {}",
            end.duration_since(*at).as_secs_f64(),
            cpu_next - cpu,
            peaks.get(phase).copied().unwrap_or(0) as f64 / 1e9,
            names.join(", ")
        );
    }
    for (op, (ns, calls)) in profile.by_op.lock().unwrap().iter() {
        eprintln!(
            "  {op:>28}: {:>8.3} s cpu over {calls} blocks",
            *ns as f64 / 1e9
        );
    }
}

#[test]
#[ignore = "a timing; run with --release --ignored --nocapture"]
fn where_the_detector_spends_its_time() {
    let configs = [
        (
            [64, 128, 128],
            CellParams {
                watershed: Watershed::DEFAULT_TILED,
                ..CellParams::default()
            },
        ),
        (
            [128, 128, 128],
            CellParams {
                watershed: Watershed::DEFAULT_TILED,
                ..CellParams::default()
            },
        ),
        (
            [64, 128, 128],
            CellParams {
                watershed: Watershed::Whole,
                ..CellParams::default()
            },
        ),
    ];
    let wanted = std::env::var("CELL_DENSITY_CONFIGS")
        .unwrap_or_else(|_| "tiled,tiled128,whole".to_string());
    let names = ["tiled", "tiled128", "whole"];
    let configs: Vec<_> = configs
        .into_iter()
        .zip(names)
        .filter(|(_, name)| wanted.split(',').any(|w| w == *name))
        .map(|(config, _)| config)
        .map(
            |(block, params)| match std::env::var("CELL_DENSITY_BLOCK") {
                Ok(text) => {
                    let v: Vec<usize> = text.split(',').map(|t| t.parse().unwrap()).collect();
                    ([v[0], v[1], v[2]], params)
                }
                Err(_) => (block, params),
            },
        )
        .collect();
    let tile = PathBuf::from(std::env::var("CELL_DENSITY_TILE").unwrap_or_else(|_| {
        "/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles/tile_05x06.zarr".to_string()
    }));
    let (start, shape) = match std::env::var("CELL_DENSITY_WINDOW") {
        Ok(text) => {
            let triple = |part: &str| -> [usize; 3] {
                let v: Vec<usize> = part.split(',').map(|t| t.parse().unwrap()).collect();
                [v[0], v[1], v[2]]
            };
            let (start, shape) = text.split_once(';').expect("start;shape");
            (triple(start), triple(shape))
        }
        Err(_) => ([1792, 640, 16], [256, 384, 384]),
    };
    if tile.exists() {
        let source = OmeVolumePyramid::open(&tile, 0, 0)
            .unwrap()
            .attached_level(0)
            .unwrap()
            .window(start, shape);
        let (dtype, volume) = source.metadata().unwrap();
        for (block, params) in &configs {
            let block = *block;
            let plan =
                CellPlan::assemble(params, dtype, volume, block, PlanOptions::default()).unwrap();
            let scratch =
                std::env::temp_dir().join(format!("cell-density-profile-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&scratch);
            let policy = match std::env::var("CELL_DENSITY_COMPRESSION").as_deref() {
                Ok("none") => CompressionPolicy::uniform(Compression::None),
                _ => CompressionPolicy::derived(),
            };
            let mut env = ZarrEnvironment::attach_with_compression(
                &scratch,
                std::slice::from_ref(&source),
                policy,
            )
            .unwrap();
            if let Ok(bytes) = std::env::var("CELL_DENSITY_CACHE_BYTES") {
                env = env.with_cache(bytes.parse().expect("a byte count"));
            }
            run(
                &plan,
                &env,
                &format!("block {block:?}, {:?}", params.watershed),
            );
            drop(env);
            let _ = std::fs::remove_dir_all(&scratch);
        }
    } else {
        eprintln!("{} is absent; profiling a synthetic volume", tile.display());
        let scene = Scene::new(
            SceneSpec::new([128, 256, 256], 3)
                .with_objects(2000)
                .with_radius(2.5, 4.0)
                .with_noise(0.05),
        )
        .unwrap();
        let input = Voxels::U16(scene.render().intensity.mapv(|v| (v * 2048.0) as u16));
        for (block, params) in &configs {
            let block = *block;
            let plan = CellPlan::assemble(
                params,
                input.dtype(),
                input.shape(),
                block,
                PlanOptions::default(),
            )
            .unwrap();
            let env = ArrayEnvironment::for_decomposition(
                input.clone(),
                &plan.assembly.decomposition,
                [8, 8, 8],
            )
            .unwrap();
            run(
                &plan,
                &env,
                &format!("block {block:?}, {:?}", params.watershed),
            );
        }
    }
}
