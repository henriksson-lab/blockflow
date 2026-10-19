// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Where the binarisation's time goes, arm by arm: the full chain against the
//! chain with one arm removed, on a slab of a tile's cross-section.
//!
//! Timing only, so ignored; run with
//! `cargo test --release -p blockflow-vessel-graph --test segment_attribution -- --ignored --nocapture`.

use std::time::Instant;

use blockflow_vessel_graph::segment::synthetic::vessel_volume;
use blockflow_vessel_graph::segment::{binarize_volume, BinarizeParams};

/// An arm's name and how to switch it off.
type Removal = (&'static str, fn(&mut BinarizeParams));

#[test]
#[ignore = "timing only: runs the binarisation seven times over 12 x 1304 x 404 voxels"]
fn the_binarisations_wall_clock_is_attributable_to_its_arms() {
    let shape = [12, 1304, 404];
    let block = [12, 256, 256];
    let volume = vessel_volume(shape, 1);
    let full = BinarizeParams::default();
    let time = |params: &BinarizeParams| {
        let started = Instant::now();
        binarize_volume(volume.clone(), params, block).expect("a run");
        started.elapsed().as_secs_f64()
    };
    let total = time(&full);
    println!("full chain: {total:.2}s over {shape:?}");
    let removals: [Removal; 6] = [
        ("lightsheet", |p| p.lightsheet = None),
        ("median", |p| p.median = None),
        ("deconvolve", |p| p.deconvolve = None),
        ("adaptive", |p| p.adaptive = None),
        ("equalize", |p| p.equalize = None),
        ("vesselize", |p| p.vesselize = None),
    ];
    for (arm, remove) in removals {
        let mut params = full.clone();
        remove(&mut params);
        let without = time(&params);
        println!(
            "{arm:>10}: {without:.2}s without it, so about {:.2}s ({:.0}%) is the arm",
            total - without,
            100.0 * (total - without) / total
        );
    }
}
