// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The segmentation does not depend on how the volume is cut.
//!
//! Every op here declares its reach, the planner grants exactly that as the
//! halo, and the sampled statistics sit on a lattice anchored to the volume
//! rather than to the block, so a blocked run has to agree with a one-block
//! run voxel for voxel. These run the whole chain — every arm, including the
//! two the planner may cut into several phases — at several block grids,
//! among them grids whose edges do not divide the volume.

use blockflow::voxels::Voxels;
use blockflow_vessel_graph::segment::synthetic::vessel_volume;
use blockflow_vessel_graph::segment::{binarize_volume, segment, BinarizeParams, SegmentParams};

const SHAPE: [usize; 3] = [20, 48, 56];

fn same(label: &str, a: &Voxels, b: &Voxels) {
    let a = a.widened();
    let b = b.widened();
    let differing = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(differing, 0, "{label}: {differing} voxels differ");
}

#[test]
fn the_binarisation_is_the_same_at_every_block_grid() {
    let volume = vessel_volume(SHAPE, 3);
    let params = BinarizeParams::default();
    let (filtered, binary) = binarize_volume(volume.clone(), &params, SHAPE).expect("one block");
    assert!(
        binary.widened().iter().any(|v| *v != 0.0),
        "the fixture binarises to nothing"
    );
    for block in [[10, 24, 28], [7, 16, 19], [20, 13, 56], [4, 48, 9]] {
        let (f, b) = binarize_volume(volume.clone(), &params, block).expect("blocked");
        same(&format!("filtered at {block:?}"), &filtered, &f);
        same(&format!("binary at {block:?}"), &binary, &b);
    }
}

#[test]
fn the_whole_segmentation_is_the_same_at_every_block_grid() {
    let volume = vessel_volume(SHAPE, 11);
    let params = SegmentParams::with_channels(1);
    let whole = segment(vec![volume.clone()], &params, SHAPE).expect("one block");
    for block in [[10, 24, 28], [6, 17, 20]] {
        let cut = segment(vec![volume.clone()], &params, block).expect("blocked");
        same(&format!("mask at {block:?}"), &whole.mask, &cut.mask);
        same(
            &format!("skeleton at {block:?}"),
            &whole.skeleton,
            &cut.skeleton,
        );
    }
}
