// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The whole chain: tiles cropped from one scene at known offsets, stitched
//! from the nominal grid, and checked against the offsets and the scene.

mod common;

use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{PyramidSpec, ZarrEnvironment};
use tile_stitching::io::{positions_json, read_tile, write_volume, TileSetFile};
use tile_stitching::rigid::RigidParams;
use tile_stitching::wobbly::SourceStatus;
use tile_stitching::{stitch, StitchParams, Stitched};

fn params(overlaps: &[usize], grid_axes: &[usize]) -> StitchParams {
    StitchParams {
        rigid: RigidParams::from_overlaps(overlaps, grid_axes),
        ..StitchParams::default()
    }
}

/// Every fused voxel a usable tile plane covers equals that tile's voxel.
/// Overlapping tiles agree wherever their relative placement is right, so this
/// fails at any voxel where it is not.
fn assert_fused_matches_tiles(stitched: &Stitched, tiles: &[Voxels]) -> usize {
    let placement = &stitched.placement;
    let origin = placement.origin();
    let axis = placement.axis();
    let fused = stitched.fused.view::<u16>().expect("u16");
    let mut checked = 0;
    for (index, (source, tile)) in placement.sources().iter().zip(tiles).enumerate() {
        let tile = tile.view::<u16>().expect("u16");
        for plane in 0..source.shape()[axis] {
            if !source.is_valid(source.coordinate() + plane as i64) {
                continue;
            }
            let at = source.position(plane);
            for ((k, j, i), &value) in tile
                .index_axis(ndarray::Axis(axis), plane)
                .insert_axis(ndarray::Axis(axis))
                .indexed_iter()
            {
                let global = [
                    (at[0] + k as i64 - origin[0]) as usize,
                    (at[1] + j as i64 - origin[1]) as usize,
                    (at[2] + i as i64 - origin[2]) as usize,
                ];
                assert_eq!(
                    fused[global], value,
                    "tile {index} plane {plane} at {global:?}"
                );
                checked += 1;
            }
        }
    }
    checked
}

/// A pair offset by `(2, 3, -4)` from the nominal grid: the rigid step finds
/// z and y, the per-slice step finds x, and the fused volume is the scene.
#[test]
fn the_chain_recovers_the_offsets_and_reproduces_the_scene() {
    let (tiles, layout) = common::pair([2, 3, -4], |_| [0, 0]);
    let stitched = stitch(&tiles, layout, &params(&[30], &[2])).expect("a stitch");
    let nominal_x = common::TILE[2] as isize - common::OVERLAP_X;
    assert_eq!(stitched.layout.positions(), [[0, 0, 0], [2, 3, nominal_x]]);

    let placed = stitched.placed.as_ref().expect("the per-slice placement");
    for (k, position) in placed.positions[1].iter().enumerate() {
        assert_eq!(
            *position,
            vec![2 + k as isize, 3, nominal_x - 4],
            "plane {k}"
        );
    }
    assert!(placed
        .status
        .iter()
        .flatten()
        .all(|&s| s == SourceStatus::VALID));

    let fused = stitched.fused.view::<u16>().expect("u16");
    assert_eq!(fused.shape(), [26, 83, 106]);
    for ((z, y, x), &value) in fused.indexed_iter() {
        let inside = |origin: [usize; 3]| {
            (0..3).all(|a| (origin[a]..origin[a] + common::TILE[a]).contains(&[z, y, x][a]))
        };
        if inside([0, 0, 0]) || inside([2, 3, 36]) {
            assert_eq!(value, common::scene(z as isize, y as isize, x as isize));
        }
    }
}

/// Without the per-slice step the x offset is never measured, so the tiles
/// disagree across the seam; with it they agree everywhere.
#[test]
fn the_rigid_step_alone_leaves_the_tiling_axis_uncorrected() {
    let (tiles, layout) = common::pair([2, 3, -4], |_| [0, 0]);
    let rigid_only = stitch(
        &tiles,
        layout.clone(),
        &StitchParams {
            wobbly: false,
            ..params(&[30], &[2])
        },
    )
    .expect("a rigid stitch");
    assert!(rigid_only.placed.is_none());
    assert_eq!(rigid_only.fused.shape(), [26, 83, 110]);
    let scene_at = |z: usize, y: usize, x: usize| common::scene(z as isize, y as isize, x as isize);
    let view = rigid_only.fused.view::<u16>().expect("u16");
    // In the second tile's own region beyond the overlap the rigid volume holds
    // the scene shifted by the uncorrected 4 voxels.
    assert_ne!(view[[10, 40, 100]], scene_at(10, 40, 100));
    assert_eq!(view[[10, 40, 100]], scene_at(10, 40, 96));

    let wobbly = stitch(&tiles, layout, &params(&[30], &[2])).expect("a stitch");
    assert!(assert_fused_matches_tiles(&wobbly, &tiles) > 0);
}

/// A second tile whose planes drift in x along z: the chain follows the drift
/// plane by plane, so every fused voxel agrees with every tile covering it.
#[test]
fn the_chain_follows_a_drift_along_the_stack() {
    let drift = |k: usize| [0, (k / 6) as isize];
    let (tiles, layout) = common::pair([2, 3, -4], drift);
    let stitched = stitch(
        &tiles,
        layout,
        &StitchParams {
            measure: tile_stitching::wobbly::MeasureParams {
                slice_step: 1,
                ..Default::default()
            },
            place: tile_stitching::place::PlaceParams {
                smooth: tile_stitching::wobbly::SmoothParams {
                    window: None,
                    window_length: 1,
                    binary: None,
                },
                smooth_optimized: None,
                ..Default::default()
            },
            ..params(&[30], &[2])
        },
    )
    .expect("a stitch");
    let placed = stitched.placed.as_ref().expect("the per-slice placement");
    for k in 0..common::TILE[0] {
        // Relative to the first tile's plane at the same z.
        let z = 2 + k;
        if z >= common::TILE[0] {
            break;
        }
        let (a, b) = (&placed.positions[0][z], &placed.positions[1][k]);
        assert_eq!(b[1] - a[1], 3, "plane {k}");
        assert_eq!(b[2] - a[2], 36 + drift(k)[1], "plane {k}");
    }
    assert!(assert_fused_matches_tiles(&stitched, &tiles) > 0);
}

/// A 2 x 2 grid with a different offset per tile.
#[test]
fn a_two_by_two_grid_is_stitched() {
    let (ty, tx) = (
        common::TILE[1] as isize - common::OVERLAP_Y,
        common::TILE[2] as isize - common::OVERLAP_X,
    );
    let offsets = [[0, 0, 0], [1, 2, -3], [-1, 3, 2], [2, -2, 1]];
    let grid = [[0, 0], [1, 0], [0, 1], [1, 1]];
    let tiles = grid
        .iter()
        .zip(offsets)
        .map(|(g, offset)| {
            common::crop(
                [
                    2 + offset[0],
                    5 + g[1] * ty + offset[1],
                    5 + g[0] * tx + offset[2],
                ],
                common::TILE,
                |_| [0, 0],
            )
        })
        .collect::<Vec<_>>();
    let layout = common::grid_layout(&grid);
    let stitched = stitch(
        &tiles,
        layout,
        &params(
            &[common::OVERLAP_X as usize, common::OVERLAP_Y as usize],
            &[2, 1],
        ),
    )
    .expect("a stitch");
    let placed = stitched.placed.as_ref().expect("the per-slice placement");
    // Every tile's true origin relative to tile 0, in every plane both hold.
    for (index, offset) in offsets.iter().enumerate() {
        let truth = [
            offset[0],
            grid[index][1] * ty + offset[1],
            grid[index][0] * tx + offset[2],
        ];
        let start = stitched.layout.positions()[index][0] - stitched.layout.positions()[0][0];
        assert_eq!(start, truth[0], "tile {index} z");
        for (k, position) in placed.positions[index].iter().enumerate() {
            let z = (start + k as isize) as usize;
            let Some(reference) = placed.positions[0].get(z) else {
                continue;
            };
            assert_eq!(
                position[1] - reference[1],
                truth[1],
                "tile {index} plane {k} y"
            );
            assert_eq!(
                position[2] - reference[2],
                truth[2],
                "tile {index} plane {k} x"
            );
        }
    }
    assert!(assert_fused_matches_tiles(&stitched, &tiles) > 0);
}

/// The CLI's path: tiles written as OME-Zarr images, a tile-set file, the
/// stitch, the fused image written and read back, and the positions JSON.
#[test]
fn the_ome_zarr_round_trip_stitches_the_scene() {
    let dir = common::scratch("ome_zarr_round_trip");
    let (tiles, _) = common::pair([2, 3, -4], |_| [0, 0]);
    for (index, tile) in tiles.iter().enumerate() {
        ZarrEnvironment::build_multiscale(
            dir.join(format!("tile_{index}.zarr")),
            tile,
            &PyramidSpec::new(vec![[1, 2, 2]]).expect("a pyramid"),
        )
        .expect("a tile image");
    }
    std::fs::write(
        dir.join("tiles.json"),
        r#"{
          "grid_axes": ["x"],
          "overlap": [30],
          "tiles": [
            {"path": "tile_0.zarr", "grid": [0]},
            {"path": "tile_1.zarr", "grid": [1]}
          ]
        }"#,
    )
    .expect("a tile-set file");

    let tile_set = TileSetFile::read(&dir.join("tiles.json")).expect("the tile set");
    let read = tile_set
        .tiles
        .iter()
        .map(|entry| {
            read_tile(&entry.path, 0, 0, Some((0, 24)))
                .expect("a tile")
                .voxels
        })
        .collect::<Vec<_>>();
    assert_eq!(read, tiles);
    let shapes = read.iter().map(Voxels::shape).collect::<Vec<_>>();
    let layout = tile_set.layout(&shapes).expect("a layout");
    let grid_axes = tile_set.grid_axes().expect("grid axes");
    let stitched = stitch(
        &read,
        layout.clone(),
        &params(&tile_set.overlaps(&layout), &grid_axes),
    )
    .expect("a stitch");

    let out = dir.join("fused.zarr");
    write_volume(
        &out,
        &stitched.fused,
        [2.0, 0.5, 0.5],
        Some("micrometer"),
        2,
    )
    .expect("the fused image");
    let pyramid = OmeVolumePyramid::open(&out, 0, 0).expect("an OME-Zarr image");
    assert_eq!(pyramid.levels().len(), 2);
    assert_eq!(pyramid.voxel_size(), [2.0, 0.5, 0.5]);
    let back = pyramid
        .attached_level(0)
        .expect("level 0")
        .read()
        .expect("the fused volume");
    assert_eq!(back, stitched.fused);

    let paths = tile_set
        .tiles
        .iter()
        .map(|t| t.path.clone())
        .collect::<Vec<_>>();
    let json = positions_json(&stitched, &layout, &paths, 0);
    assert_eq!(json["shape"], serde_json::json!([26, 83, 106]));
    assert_eq!(json["tiles"][1]["rigid"], serde_json::json!([2, 3, 40]));
    assert_eq!(json["tiles"][1]["slices"][0], serde_json::json!([2, 3, 36]));
    assert_eq!(json["alignments"][0]["shift"], serde_json::json!([2, 3, 0]));
    assert!(
        write_volume(&out, &stitched.fused, [1.0; 3], None, 1).is_err(),
        "an existing output"
    );
}

/// Two neighbouring real tiles, 240 planes of each, stitched end to end.
#[test]
#[ignore = "reads two light-sheet tiles (about 500 MB) from TILE_STITCHING_TILES or /husky and fuses about 460 MB; run with --release --ignored"]
fn a_real_pair_is_stitched_end_to_end() {
    let root = common::real_tiles_root();
    let paths = [root.join("tile_04x05.zarr"), root.join("tile_04x06.zarr")];
    if !paths.iter().all(|path| path.exists()) {
        eprintln!("no tiles under {}; skipped", root.display());
        return;
    }
    let started = std::time::Instant::now();
    let tiles = paths
        .iter()
        .map(|path| {
            read_tile(path, 0, 0, Some((1200, 1440)))
                .expect("a tile")
                .voxels
        })
        .collect::<Vec<_>>();
    let shape = tiles[0].shape();
    let layout = tile_stitching::layout::TileLayout::on_grid(
        (0..2)
            .map(|x| tile_stitching::layout::TileSource {
                shape,
                tile_position: vec![x, 0],
            })
            .collect(),
        &[66, 212],
        &[2, 1],
    )
    .expect("a pair");
    let stitched = stitch(&tiles, layout, &params(&[66, 212], &[2, 1])).expect("a stitch");
    let fused = stitched.fused.shape();
    eprintln!(
        "fused {fused:?} in {:.1?}; rigid {:?}",
        started.elapsed(),
        stitched.layout.positions()
    );
    let placed = stitched.placed.as_ref().expect("the per-slice placement");
    eprintln!("{:?}", placed.report);
    assert!(
        (fused[2] as isize - (2 * shape[2] as isize - 66)).abs() <= 60,
        "{fused:?}"
    );
    assert!(
        (fused[1] as isize - shape[1] as isize).abs() <= 60,
        "{fused:?}"
    );
}
