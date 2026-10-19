// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The chain composed: each step changes what it is given, in the direction
//! it should, and the steps agree with each other.

use blockflow::ops::Connectivity;
use blockflow::voxels::Voxels;
use blockflow_vessel_graph::segment::synthetic::{tube_mask, vessel_volume};
use blockflow_vessel_graph::segment::{
    mask_steps, segment, tract_map, MaskSteps, PostProcess, SegmentParams, SkeletonParams,
    SmoothParams, TractParams,
};
use ndarray::Array3;

fn count(voxels: &Voxels) -> usize {
    voxels
        .widened()
        .iter()
        .filter(|value| **value != 0.0)
        .count()
}

fn bools(voxels: &Voxels) -> Array3<bool> {
    voxels.widened().mapv(|value| value != 0.0)
}

/// A solid bar with a sealed cavity: the smoothing keeps it, the fill closes
/// the cavity, and the thinning leaves a curve a small fraction of the bar.
#[test]
fn the_mask_steps_compose_on_a_bar_with_a_cavity() {
    let mut bar = Array3::from_elem((32usize, 32, 32), false);
    for z in 4..28 {
        for y in 12..20 {
            for x in 12..20 {
                bar[[z, y, x]] = true;
            }
        }
    }
    for y in 15..17 {
        for x in 15..17 {
            bar[[16, y, x]] = false;
        }
    }
    let bar = Voxels::Bool(bar);
    let before = count(&bar);
    let smoothed = mask_steps(
        bar.clone(),
        &MaskSteps {
            post: Some(PostProcess {
                smooth: Some(SmoothParams::default()),
                fill: None,
            }),
            skeleton: None,
        },
        [16, 16, 16],
    )
    .unwrap()
    .1;
    let (_, filled, skeleton) = mask_steps(
        bar.clone(),
        &MaskSteps {
            post: Some(PostProcess::default()),
            skeleton: Some(SkeletonParams::default()),
        },
        [16, 16, 16],
    )
    .unwrap();
    let skeleton = skeleton.expect("asked for");
    assert!(
        count(&filled) >= count(&smoothed),
        "the fill removed voxels"
    );
    assert!(
        bools(&filled)[[16, 15, 15]],
        "the cavity is enclosed and is filled"
    );
    let thinned = count(&skeleton);
    assert!(
        thinned > 0 && thinned * 20 < before,
        "the thinning left {thinned} of {before}"
    );
    let filled = bools(&filled);
    assert!(
        bools(&skeleton).iter().zip(&filled).all(|(s, m)| !*s || *m),
        "the skeleton leaves the mask it thinned"
    );
}

/// The whole segmentation on synthetic vessels: the mask covers the tubes'
/// axes, the skeleton lies inside the mask and is thin, and the channel's own
/// post-processed mask is the answer when there is one channel.
#[test]
fn the_segmentation_finds_the_tubes() {
    let shape = [24, 64, 64];
    let volume = vessel_volume(shape, 5);
    let run = segment(vec![volume], &SegmentParams::with_channels(1), [12, 32, 32]).unwrap();
    let mask = bools(&run.mask);
    let axes = tube_mask(shape, 1.0);
    let covered = axes.iter().zip(&mask).filter(|(a, m)| **a && **m).count();
    let on_axis = axes.iter().filter(|a| **a).count();
    println!(
        "mask {} voxels, skeleton {}, {covered} of {on_axis} axis voxels covered",
        count(&run.mask),
        count(&run.skeleton)
    );
    assert!(
        covered * 20 > on_axis * 17,
        "{covered} of {on_axis} axis voxels"
    );
    assert_eq!(
        run.channels[0].processed.widened(),
        run.mask.widened(),
        "one channel: the channel's own mask is the answer"
    );
    assert!(bools(&run.skeleton)
        .iter()
        .zip(&mask)
        .all(|(s, m)| !*s || *m));
    assert!(count(&run.skeleton) * 5 < count(&run.mask));
    assert!(
        count(&run.channels[0].processed) >= count(&run.channels[0].binary) / 2,
        "the post-processing removed most of the binarisation"
    );
}

/// Two channels: the union contains each channel's mask before its second
/// post-processing, and a channel that sees the same data adds nothing new.
#[test]
fn two_channels_unite() {
    let shape = [24, 48, 48];
    let a = vessel_volume(shape, 1);
    let b = vessel_volume(shape, 2);
    let mut params = SegmentParams::with_channels(2);
    params.combined = PostProcess {
        smooth: None,
        fill: Some(Connectivity::Faces),
    };
    let run = segment(vec![a, b], &params, [12, 24, 24]).unwrap();
    let mask = bools(&run.mask);
    for channel in &run.channels {
        assert!(bools(&channel.processed)
            .iter()
            .zip(&mask)
            .all(|(c, m)| !*c || *m));
    }
    let largest = run
        .channels
        .iter()
        .map(|channel| count(&channel.processed))
        .max()
        .unwrap();
    assert!(count(&run.mask) >= largest);
}

/// The tract spine: a clip range from the volume's own percentiles, a mask, a
/// point per mask voxel, and a density that is the points spread over a ball.
#[test]
fn the_tract_spine_composes() {
    let shape = [24, 48, 48];
    let volume = vessel_volume(shape, 9);
    let params = TractParams {
        percentiles: (70.0, 99.9),
        sample_every: 7,
        density_radius: [2, 2, 2],
        ..TractParams::default()
    };
    let map = tract_map(volume.clone(), &params, [12, 24, 24]).unwrap();
    let raw = volume.widened();
    let max = raw.iter().copied().fold(f64::MIN, f64::max);
    let min = raw.iter().copied().fold(f64::MAX, f64::min);
    assert!(
        map.range.0 >= min && map.range.0 < map.range.1 && map.range.1 <= max,
        "{:?}",
        map.range
    );
    assert_eq!(map.points.len(), count(&map.mask), "a point per mask voxel");
    assert!(!map.points.is_empty());
    for at in &map.points {
        assert!(map.mask.widened()[*at] == 1.0);
    }
    // Every voxel of the density is the number of points within the ball, so
    // it is positive at every point and the whole sums to the points times the
    // ball, less what falls outside the volume.
    let density = map.density.widened();
    assert!(map.points.iter().all(|at| density[*at] >= 1.0));
    let ball = blockflow::ops::voxelize::ball(params.density_radius).len();
    let total: f64 = density.iter().sum();
    assert!(total <= (map.points.len() * ball) as f64);
    assert!(total > 0.5 * (map.points.len() * ball) as f64);

    // The same answer on another block grid.
    let again = tract_map(volume, &params, [24, 16, 48]).unwrap();
    assert_eq!(again.range, map.range);
    assert_eq!(again.points, map.points);
    assert_eq!(again.density.widened(), map.density.widened());
}
