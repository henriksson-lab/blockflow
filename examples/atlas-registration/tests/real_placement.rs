// SPDX-License-Identifier: MIT
// Original work for this crate.

// The placement search on a real partial mosaic: nine fused tiles of a
// light-sheet sample in its autofluorescence, which carries no stage
// position. A measurement on real data, so it is ignored by default and needs
// both images and the translation measured independently (by a normalised
// cross-correlation on 50 and 30 µm grids, the lightsheet-pipeline README):
//
// ```text
// PLACE_FIXED_ZARR=/path/to/stitch/fused.ome.zarr \
// PLACE_MOVING_ZARR=/path/to/autofluorescence.zarr \
// PLACE_EXPECTED=0,3420,4535 \
//     cargo test --release -p blockflow-atlas-registration --test real_placement \
//     -- --ignored --nocapture
// ```

use std::path::PathBuf;

use blockflow_atlas_registration::pipeline::{place, Input};
use blockflow_atlas_registration::place::SearchParams;

fn input(variable: &str) -> Input {
    Input {
        path: PathBuf::from(
            std::env::var(variable).unwrap_or_else(|_| panic!("{variable} is not set")),
        ),
        channel: 0,
        level: None,
    }
}

#[test]
#[ignore = "needs a partial mosaic and its autofluorescence: set PLACE_FIXED_ZARR, PLACE_MOVING_ZARR and PLACE_EXPECTED (z,y,x)"]
fn a_partial_mosaic_is_found_in_its_autofluorescence() {
    let expected: Vec<f64> = std::env::var("PLACE_EXPECTED")
        .expect("PLACE_EXPECTED is not set")
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect();
    let placement = place(
        &input("PLACE_FIXED_ZARR"),
        &input("PLACE_MOVING_ZARR"),
        Some([25.0; 3]),
        &SearchParams::default(),
    )
    .expect("placed");
    let found = placement.translation();
    println!(
        "found {found:?}, expected {expected:?}, NCC {:.3}, margin {:.3}, {:.2} s",
        placement.best.ncc,
        placement.margin(),
        placement.seconds
    );
    // within one registration lattice voxel on every axis
    for a in 0..3 {
        assert!(
            (found[a] - expected[a]).abs() < 25.0,
            "{found:?} vs {expected:?}"
        );
    }
}
