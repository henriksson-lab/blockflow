// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The graph's heap, stage by stage: a counting allocator behind the
//! profile's stage lines (`VESSEL_GRAPH_PROFILE`), so each line also says
//! the live heap at the boundary and the most the stage held.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use std::sync::{Arc, Mutex};

use blockflow::voxels::Voxels;
use blockflow_vessel_graph::graph::cli::{run_graph, GraphArgs};
use blockflow_vessel_graph::graph::planned::GRAPH_BYTES_PER_VERTEX;
use blockflow_vessel_graph::graph::{
    build, carried_radii, finish, BuildParams, Carried, Column, GraphParams, RadiusParams,
    RadiusSearch,
};
use clap::Parser;
use ndarray::Array3;

/// One test at a time: the counters are the whole process's.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

struct Counting;

static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static MOST: AtomicU64 = AtomicU64::new(0);

fn grew(bytes: usize) {
    let now = LIVE.fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
    PEAK.fetch_max(now, Ordering::Relaxed);
    MOST.fetch_max(now, Ordering::Relaxed);
}

fn shrank(bytes: usize) {
    LIVE.fetch_sub(bytes as u64, Ordering::Relaxed);
}

// SAFETY: every call is forwarded to `System` unchanged; only counters move.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        shrank(layout.size());
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, layout, size) };
        if !moved.is_null() {
            if size >= layout.size() {
                grew(size - layout.size());
            } else {
                shrank(layout.size() - size);
            }
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The live heap and the most since the previous call.
fn meter() -> (u64, u64) {
    let live = LIVE.load(Ordering::Relaxed);
    let peak = PEAK.swap(live, Ordering::Relaxed);
    (live, peak.max(live))
}

#[derive(Debug, Parser)]
struct Command {
    #[command(flatten)]
    graph: GraphArgs,
}

/// The `graph` subcommand on real data with the heap per stage. Needs
/// `GRAPH_MEMORY_ARGS`, the subcommand's arguments (an image `segment`
/// published into, which gets its tables rewritten); `GRAPH_MEMORY_MB`
/// states a memory cap. Run with `--nocapture`.
/// A lattice of crossing centrelines every `step` voxels along all three
/// axes, with a few diagonal steps that make cliques at some crossings, and
/// tubes of radius 2 around them.
fn lattice(edge: usize, step: usize) -> (Array3<bool>, Array3<bool>) {
    let mut skeleton = Array3::from_elem((edge, edge, edge), false);
    for a in (step / 2..edge).step_by(step) {
        for b in (step / 2..edge).step_by(step) {
            for t in 1..edge - 1 {
                skeleton[[t, a, b]] = true;
                skeleton[[a, t, b]] = true;
                skeleton[[a, b, t]] = true;
            }
            // A diagonal neighbour beside the crossing: a clique of branch
            // points there.
            skeleton[[a + 1, b + 1, a.min(edge - 2) + 1]] = true;
        }
    }
    let mut vessels = skeleton.clone();
    for ((z, y, x), set) in skeleton.indexed_iter() {
        if !*set {
            continue;
        }
        for dz in -2isize..=2 {
            for dy in -2isize..=2 {
                for dx in -2isize..=2 {
                    if dz * dz + dy * dy + dx * dx > 4 {
                        continue;
                    }
                    let at = [z as isize + dz, y as isize + dy, x as isize + dx];
                    if at.iter().all(|value| (0..edge as isize).contains(value)) {
                        vessels[[at[0] as usize, at[1] as usize, at[2] as usize]] = true;
                    }
                }
            }
        }
    }
    (skeleton, vessels)
}

/// The resident stages (clean, reduce, assemble, largest component,
/// lengths) hold at most what the graph's memory lease charges a raw vertex
/// ([`GRAPH_BYTES_PER_VERTEX`]), the raw graph and its columns included.
#[test]
fn the_resident_graph_stays_within_its_bytes_per_raw_vertex() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let (skeleton, vessels) = lattice(96, 8);
    let skeleton = Voxels::from(skeleton);
    let vessels = Voxels::from(vessels);
    let raw = build(&skeleton, &BuildParams::default()).expect("a raw graph");
    let n_raw = raw.n_vertices();
    let search = Arc::new(
        RadiusSearch::new(RadiusParams {
            max_radius: [6, 6, 6],
            ..RadiusParams::default()
        })
        .expect("a search"),
    );
    let stops = search.stops(&vessels, raw.coordinates()).expect("stops");
    let spacing = [2.0, 1.6, 1.6];
    let mut carried = carried_radii(Arc::clone(&search), &stops, spacing).expect("radii");
    drop(stops);
    // An intensity of a `uint16` channel and a mask, as the planned stages
    // carry them.
    let intensity: Vec<f64> = (0..n_raw).map(|at| (at * 7919 % 4093) as f64).collect();
    let flags: Vec<bool> = (0..n_raw).map(|at| at % 5 == 0).collect();
    carried.push(Carried::max("artery_raw", Column::scalar(intensity)));
    carried.push(Carried::binary("artery_binary", Column::flags(&flags)));
    let held = raw.structural_bytes()
        + n_raw * 4
        + carried[4..]
            .iter()
            .map(|carry| carry.column.bytes())
            .sum::<usize>();
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let graph = finish(raw, carried, &GraphParams::default(), spacing).expect("a graph");
    let most = PEAK.load(Ordering::Relaxed) - before + held as u64;
    let per_vertex = most as f64 / n_raw as f64;
    eprintln!(
        "{n_raw} raw vertices to {} vertices, {} edges and {} points: the resident stages \
         held {:.1} MB, {per_vertex:.1} bytes a raw vertex; the graph holds {:.1}",
        graph.n_vertices(),
        graph.n_edges(),
        graph.n_points(),
        most as f64 / 1e6,
        graph.bytes() as f64 / n_raw as f64,
    );
    assert!(n_raw > 30_000 && graph.n_edges() > 1_000);
    assert!(
        per_vertex <= GRAPH_BYTES_PER_VERTEX as f64,
        "{per_vertex:.1} bytes a raw vertex"
    );
}

#[test]
#[ignore = "needs GRAPH_MEMORY_ARGS: a segmented OME-Zarr image and graph arguments"]
fn the_heap_of_each_graph_stage_on_real_data() {
    let _one = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let args = std::env::var("GRAPH_MEMORY_ARGS").expect("GRAPH_MEMORY_ARGS");
    let command = Command::parse_from(std::iter::once("graph").chain(args.split_whitespace()));
    if let Some(mib) = std::env::var("GRAPH_MEMORY_MB")
        .ok()
        .and_then(|mib| mib.parse::<u64>().ok())
    {
        blockflow::budget::set_process_cap(blockflow::budget::MemoryCap::resolve(Some(mib << 20)));
    }
    std::env::set_var("VESSEL_GRAPH_PROFILE", "1");
    blockflow_vessel_graph::profile::set_heap_meter(meter);
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    MOST.store(before, Ordering::Relaxed);
    let started = std::time::Instant::now();
    let summary = run_graph(&command.graph).expect("graph");
    eprintln!(
        "graph memory: {} vertices, {} edges in {:.1} s; heap before {:.1} MB, live after {:.1} MB, peak {:.1} MB",
        summary.vertices,
        summary.edges,
        started.elapsed().as_secs_f64(),
        before as f64 / 1048576.0,
        LIVE.load(Ordering::Relaxed) as f64 / 1048576.0,
        MOST.load(Ordering::Relaxed) as f64 / 1048576.0
    );
}
