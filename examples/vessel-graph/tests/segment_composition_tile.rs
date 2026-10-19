// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The segmentation planned at the size of one light-sheet tile — 3369 x 1304
//! x 404 voxels, 1.775 G — without running it: the plan assembles, the
//! decomposition checks, and what it reads relative to the volume falls as the
//! blocks grow.

use blockflow::Dtype;
use blockflow_vessel_graph::segment::{segment_plan, SegmentParams};

const TILE: [usize; 3] = [3369, 1304, 404];

#[test]
fn the_segmentation_plans_at_tile_scale() {
    let voxels: f64 = TILE.iter().product::<usize>() as f64;
    let params = SegmentParams::with_channels(1);
    let mut previous = f64::INFINITY;
    for block in [[64, 128, 128], [128, 256, 256], [256, 512, 512]] {
        let (assembly, _) =
            segment_plan(TILE, &[Dtype::U16], &params, block).expect("the plan assembles");
        assembly.decomposition.check().expect("the plan checks");
        let reads: usize = assembly.decomposition.exact_read_voxels().iter().sum();
        let amplification = reads as f64 / voxels;
        println!(
            "{block:?}: {} phases, {amplification:.2}x the volume read",
            assembly.n_phases()
        );
        assert!(amplification.is_finite());
        assert!(
            amplification < previous * 1.05,
            "larger blocks read more: {amplification:.2}x after {previous:.2}x"
        );
        previous = amplification;
    }
}

#[test]
fn two_channels_plan_at_tile_scale() {
    let params = SegmentParams::with_channels(2);
    let (assembly, segmented) =
        segment_plan(TILE, &[Dtype::U16, Dtype::U16], &params, [128, 256, 256])
            .expect("the plan assembles");
    assembly.decomposition.check().expect("the plan checks");
    assert_eq!(segmented.channels.len(), 2);
    assert!(segmented.mask.id.index() < segmented.skeleton.id.index());
}
