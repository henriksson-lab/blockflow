// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use blockflow_bead_layer_distance::{run, Config};
use clap::Parser;

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("bead-layer-distance: use a release build (cargo run --release)");
        std::process::exit(2);
    }
    match run(&Config::parse()) {
        Ok(summary) => println!(
            "candidates={} beads={} in_tissue={} large_objects={}",
            summary.candidates, summary.beads, summary.in_tissue, summary.large_objects
        ),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
