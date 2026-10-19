// SPDX-License-Identifier: MIT
// Original work for this crate.

//! `wobbly::measure_layout`: a rigid layout and its tiles in, the
//! `WobblyLayout` the placement takes out.

mod common;

use blockflow::voxels::Voxels;
use tile_stitching::layout::{TileLayout, TileSource};
use tile_stitching::wobbly::{measure_layout, AlignmentStatus, MeasureParams};

fn views(tiles: &[Voxels]) -> Vec<ndarray::ArrayView3<'_, u16>> {
    tiles
        .iter()
        .map(|tile| tile.view::<u16>().expect("u16 tiles"))
        .collect()
}

/// A pair whose second tile drifts in x along z: every measured slice of the
/// overlap reports the true displacement, rigid offset included.
#[test]
fn measure_layout_recovers_the_planted_displacements() {
    let drift = |k: usize| [0, (k / 8) as isize];
    let (tiles, _) = common::pair([2, 3, -4], drift);
    let nominal_x = common::TILE[2] as isize - common::OVERLAP_X;
    // The rigid step would put the second tile here: z and y corrected, x
    // nominal.
    let rigid = [[0, 0, 0], [2, 3, nominal_x]];
    let wobbly = measure_layout(
        &rigid,
        &[(0, 1)],
        &views(&tiles),
        0,
        &MeasureParams::default(),
    )
    .expect("a measurement");
    assert_eq!(wobbly.axis, 0);
    assert_eq!(wobbly.sources.len(), 2);
    assert_eq!(wobbly.sources[1].position, vec![2, 3, nominal_x]);
    let track = &wobbly.alignments[0].track;
    // The overlap along z is 2..24.
    assert_eq!(track.len(), 22);
    for slice in 0..track.len() {
        if slice % 3 != 0 {
            assert_eq!(track.status[slice], AlignmentStatus::UNALIGNED);
            continue;
        }
        assert_eq!(
            track.status[slice],
            AlignmentStatus::ALIGNED,
            "slice {slice}"
        );
        assert_eq!(
            track.displacements[slice],
            vec![3, nominal_x - 4 + drift(slice)[1]],
            "slice {slice}"
        );
    }
}

/// A pair that does not overlap along the wobble axis is refused by name.
#[test]
fn a_pair_without_a_common_slice_is_refused() {
    let (tiles, _) = common::pair([0, 0, 0], |_| [0, 0]);
    let rigid = [[0, 0, 0], [30, 0, 40]];
    let error = measure_layout(
        &rigid,
        &[(0, 1)],
        &views(&tiles),
        0,
        &MeasureParams::default(),
    )
    .expect_err("a refusal");
    assert!(error.to_string().contains("do not overlap"), "{error}");
}

/// Two neighbouring real tiles: the measurement runs end to end on real
/// signal and aligns most measured slices.
#[test]
#[ignore = "reads two light-sheet tiles (about 500 MB) from TILE_STITCHING_TILES or /husky; run with --release --ignored"]
fn measure_layout_runs_on_a_real_pair() {
    let root = common::real_tiles_root();
    let paths = [root.join("tile_04x05.zarr"), root.join("tile_04x06.zarr")];
    if !paths.iter().all(|path| path.exists()) {
        eprintln!("no tiles under {}; skipped", root.display());
        return;
    }
    let tiles = paths
        .iter()
        .map(|path| {
            tile_stitching::io::read_tile(path, 0, 0, Some((1200, 1440)))
                .expect("a tile")
                .voxels
        })
        .collect::<Vec<_>>();
    let shape = tiles[0].shape();
    let mut layout = TileLayout::on_grid(
        (0..2)
            .map(|x| TileSource {
                shape,
                tile_position: vec![x, 0],
            })
            .collect(),
        &[66, 212],
        &[2, 1],
    )
    .expect("a pair");
    let params = tile_stitching::StitchParams {
        rigid: tile_stitching::rigid::RigidParams::from_overlaps(&[66, 212], &[2, 1]),
        ..tile_stitching::StitchParams::default()
    };
    tile_stitching::align_rigid(&tiles, &mut layout, &params).expect("the rigid step");
    let started = std::time::Instant::now();
    let wobbly = measure_layout(
        layout.positions(),
        &[(0, 1)],
        &views(&tiles),
        0,
        &params.measure,
    )
    .expect("a measurement");
    let track = &wobbly.alignments[0].track;
    eprintln!(
        "{} slices in {:.1?}: {:?}",
        track.len(),
        started.elapsed(),
        track.census()
    );
    let aligned = track
        .status
        .iter()
        .filter(|&&s| s == AlignmentStatus::ALIGNED)
        .count();
    let measured = track.len().div_ceil(params.measure.slice_step);
    assert!(
        aligned * 2 >= measured,
        "{aligned} of {measured} measured slices aligned"
    );
}
