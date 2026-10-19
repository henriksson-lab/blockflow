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
    let outcome = match Cli::parse().command {
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
