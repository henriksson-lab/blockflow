// SPDX-License-Identifier: MIT
// Original work for this crate.

use blockflow_vessel_graph::graph::cli::{run_graph, GraphArgs};
use blockflow_vessel_graph::segment::cli::{run_segment, run_tract, SegmentArgs, TractArgs};
use clap::{Parser, Subcommand};

/// Vessel segmentation, skeletons and tract maps over OME-Zarr.
#[derive(Debug, Parser)]
#[command(name = "vessel-graph")]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// Memory cap in MiB: the most the run may hold resident (default:
    /// $BLOCKFLOW_MEMORY_CAP, else the cgroup's limit, else a share of the
    /// machine). The planner fits the blocks in flight under it and the chunk
    /// caches fill what they leave; the results do not depend on it.
    #[arg(long, global = true)]
    memory_mb: Option<u64>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Vessel channel(s) to a binary mask and its skeleton, published as labels.
    Segment(SegmentArgs),
    /// A tract channel to a mask and a point density, published as labels.
    Tract(TractArgs),
    /// A published skeleton and masks to a vessel graph, written as object tables
    /// (and optionally JSON), annotated and with a branch density on request.
    Graph(GraphArgs),
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("vessel-graph: use a release build (cargo run --release)");
        std::process::exit(2);
    }
    limit_malloc_arenas();
    let cli = Cli::parse();
    let cap = blockflow::budget::MemoryCap::resolve(cli.memory_mb.map(|mib| mib << 20));
    blockflow::budget::set_process_cap(cap);
    eprintln!(
        "memory cap: {cap}, {} MiB for blocks in flight and caches",
        cap.usable_bytes() >> 20
    );
    let outcome = match cli.command {
        Command::Segment(args) => run_segment(&args).map(|summary| {
            println!(
                "mask={} skeleton={} channel_masks={} phases={}",
                summary.mask_label,
                summary.skeleton_label,
                summary.channel_mask_labels.join(","),
                summary.phases
            )
        }),
        Command::Tract(args) => run_tract(&args).map(|summary| {
            println!(
                "clip_range={},{} points={} mask={} density={}",
                summary.range.0,
                summary.range.1,
                summary.points,
                summary.mask_label,
                summary.density_label
            )
        }),
        Command::Graph(args) => run_graph(&args).map(|summary| {
            println!(
                "vertices={} edges={} classified={} annotated={}",
                summary.vertices, summary.edges, summary.classified, summary.annotated
            )
        }),
    };
    if let Err(error) = outcome {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

/// Two malloc arenas instead of glibc's eight per core, unless
/// `MALLOC_ARENA_MAX` says otherwise: forty worker threads each freeing block
/// buffers into an arena of their own keep freed memory resident (the same
/// setting as `cell-density`'s). Set before any thread starts.
fn limit_malloc_arenas() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    if std::env::var_os("MALLOC_ARENA_MAX").is_none() {
        // SAFETY: `mallopt` only sets an allocator parameter; it is called
        // before any other thread exists.
        unsafe {
            libc::mallopt(libc::M_ARENA_MAX, 2);
        }
    }
}
