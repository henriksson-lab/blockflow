// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Tile sets built from a directory of converted tiles, and fitted layouts
//! reused: fused again from the positions file, over a box, and at another
//! pyramid level, in the same physical frame.

mod common;

use std::path::Path;

use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::region::Region;
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{PhysicalFrame, PyramidSpec, ZarrEnvironment};
use serde_json::json;
use tile_stitching::fuse::FuseParams;
use tile_stitching::io::{positions_json, read_tile, Neighbours};
use tile_stitching::layout::{TileLayout, TileSource};
use tile_stitching::relayout::{anchor, fused_translation, StoredLayout};
use tile_stitching::rigid::RigidParams;
use tile_stitching::tile_dir::{from_directory, grid_from_name, TileDirOptions};
use tile_stitching::{stitch, StitchParams, Stitched};

fn write_tile(path: &Path, tile: &Voxels, translation: [f64; 3]) {
    ZarrEnvironment::build_multiscale(
        path,
        tile,
        &PyramidSpec::new(vec![[1, 2, 2]])
            .expect("a pyramid")
            .with_physical(
                PhysicalFrame::new([1.0, 1.0, 1.0], Some("micrometer")).with_origin(translation),
            ),
    )
    .expect("a tile image");
}

fn params(overlaps: &[usize]) -> StitchParams {
    StitchParams {
        rigid: RigidParams::from_overlaps(overlaps, &[2, 1]),
        ..StitchParams::default()
    }
}

fn read_all(tile_set: &tile_stitching::io::TileSetFile, level: usize) -> Vec<Voxels> {
    tile_set
        .tiles
        .iter()
        .map(|entry| {
            read_tile(&entry.path, 0, level, None)
                .expect("a tile")
                .voxels
        })
        .collect()
}

fn stitch_set(tile_set: &tile_stitching::io::TileSetFile, level: usize) -> (Stitched, Vec<Voxels>) {
    let tiles = read_all(tile_set, level);
    let shapes = tiles.iter().map(Voxels::shape).collect::<Vec<_>>();
    let layout = tile_set.layout(&shapes).expect("a layout");
    let stitched =
        stitch(&tiles, layout.clone(), &params(&tile_set.overlaps(&layout))).expect("a stitch");
    (stitched, tiles)
}

#[test]
fn grid_indices_are_read_from_tile_names() {
    assert_eq!(grid_from_name("tile_03x04.zarr"), Some([3, 4]));
    assert_eq!(grid_from_name("sample[03 x 02]_c0.ome.zarr"), Some([3, 2]));
    assert_eq!(grid_from_name("a_1x2_b_07X11.zarr"), Some([7, 11]));
    assert_eq!(grid_from_name("autofluorescence.zarr"), None);
    assert_eq!(grid_from_name("tile_x12.zarr"), None);
    assert_eq!(grid_from_name("tile_12x.zarr"), None);
}

/// Two tiles named by row and column with no translations, and an unrelated
/// image beside them: without an overlap the set is refused; with one, the
/// derived set is the hand-written one, the stitch recovers the planted
/// offset, and a level or a row/column selection change only what they
/// should.
#[test]
fn a_tile_directory_gives_the_grid_and_the_overlap() {
    let dir = common::scratch("tile_dir_overlap");
    let (tiles, _) = common::pair([2, 3, -4], |_| [0, 0]);
    write_tile(&dir.join("tile_05x07.zarr"), &tiles[0], [0.0; 3]);
    write_tile(&dir.join("tile_05x08.zarr"), &tiles[1], [0.0; 3]);
    write_tile(&dir.join("autofluorescence.zarr"), &tiles[0], [0.0; 3]);

    let refused =
        from_directory(&dir, &TileDirOptions::default()).expect_err("no positions and no overlap");
    assert!(refused.to_string().contains("--overlap"), "{refused}");

    let options = TileDirOptions {
        overlap: Some([24.0, 30.0]),
        ..TileDirOptions::default()
    };
    let derived = from_directory(&dir, &options).expect("a tile set");
    assert_eq!(derived.positions_from, "overlap option");
    let set = &derived.tile_set;
    assert_eq!(set.grid_axes, vec!["x", "y"]);
    assert_eq!(set.overlap, Some(vec![30, 24]));
    assert_eq!(set.tiles.len(), 2);
    assert_eq!(set.tiles[0].grid, vec![0, 0]);
    assert_eq!(set.tiles[1].grid, vec![1, 0]);
    assert!(set.tiles[1].path.ends_with("tile_05x08.zarr"));
    assert_eq!(
        derived.provenance["skipped"],
        json!(["autofluorescence.zarr"])
    );

    let (stitched, read) = stitch_set(set, 0);
    assert_eq!(read, tiles);
    // Tile 1 relative to tile 0: nominal 40 in x, planted (2, 3, -4).
    let placed = stitched.placed.as_ref().expect("a per-slice placement");
    let start = stitched.layout.positions()[1][0] - stitched.layout.positions()[0][0];
    assert_eq!(start, 2);
    let (a, b) = (&placed.positions[0][2], &placed.positions[1][0]);
    assert_eq!([b[1] - a[1], b[2] - a[2]], [3, 36]);

    // The option is in level-0 voxels, and is halved at level 1.
    let options = TileDirOptions {
        overlap: Some([20.0, 28.0]),
        level: 1,
        ..TileDirOptions::default()
    };
    let derived = from_directory(&dir, &options).expect("a tile set");
    assert_eq!(derived.positions_from, "overlap option");
    assert_eq!(derived.tile_set.overlap, Some(vec![14, 10]));

    // A column selection keeps one tile, rebased to grid 0.
    let options = TileDirOptions {
        columns: Some((8, 9)),
        overlap: Some([24.0, 30.0]),
        ..TileDirOptions::default()
    };
    let derived = from_directory(&dir, &options).expect("a tile set");
    assert_eq!(derived.tile_set.tiles.len(), 1);
    assert_eq!(derived.tile_set.tiles[0].grid, vec![0, 0]);
    let options = TileDirOptions {
        rows: Some((0, 5)),
        overlap: Some([24.0, 30.0]),
        ..TileDirOptions::default()
    };
    assert!(from_directory(&dir, &options).is_err(), "no rows selected");
}

/// Tiles whose translations place them: the nominal positions are the
/// translations in voxels, and the stitched frame keeps the first tile's.
#[test]
fn distinct_translations_give_the_nominal_positions() {
    let dir = common::scratch("tile_dir_translations");
    let (tiles, _) = common::pair([2, 3, -4], |_| [0, 0]);
    let nominal_x = (common::TILE[2] as isize - common::OVERLAP_X) as f64;
    write_tile(&dir.join("tile_00x00.zarr"), &tiles[0], [100.0, -50.0, 7.0]);
    write_tile(
        &dir.join("tile_00x01.zarr"),
        &tiles[1],
        [100.0, -50.0, 7.0 + nominal_x],
    );
    // No overlap anywhere: the translations are enough.
    let derived = from_directory(&dir, &TileDirOptions::default()).expect("a tile set");
    assert_eq!(derived.positions_from, "translation");
    assert_eq!(derived.tile_set.overlap, None);
    assert_eq!(derived.tile_set.tiles[0].position, Some([0, 0, 0]));
    assert_eq!(derived.tile_set.tiles[1].position, Some([0, 0, 40]));

    let (stitched, _) = stitch_set(&derived.tile_set, 0);
    let placed = stitched.placed.as_ref().expect("a per-slice placement");
    let positions = placed.positions[0]
        .iter()
        .map(|p| [p[0] as i64, p[1] as i64, p[2] as i64])
        .collect::<Vec<_>>();
    let origin = stitched.placement.origin();
    let translation = fused_translation(
        [100.0, -50.0, 7.0],
        [1.0; 3],
        anchor(&positions),
        origin,
        0,
        [0; 3],
    );
    // Tile 0's voxel v sits at fused voxel (position - origin) + v, so the
    // fused frame puts it where tile 0's own frame does.
    let at = positions[0];
    for a in 0..3 {
        let fused_index = (at[a] - origin[a]) as f64;
        let physical = translation[a] + fused_index;
        let own = [100.0, -50.0, 7.0][a];
        assert!(
            (physical - own).abs() < 1e-9,
            "axis {a}: {physical} vs {own}"
        );
    }
}

/// Neighbours from boxes: on a regular grid (also jittered by a few voxels,
/// or with a tile missing) they are the grid neighbours, in the same order and
/// along the same axes, and no diagonal; a staggered row is aligned with both
/// tiles it straddles.
#[test]
fn box_neighbours_are_the_grid_neighbours_and_no_diagonals() {
    let grid: Vec<[isize; 2]> = (0..3).flat_map(|y| (0..3).map(move |x| [x, y])).collect();
    let on_grid = common::grid_layout(&grid);
    let unnamed = |n: usize| {
        (0..n)
            .map(|_| TileSource {
                shape: common::TILE,
                tile_position: Vec::new(),
            })
            .collect::<Vec<_>>()
    };
    let pairs = |layout: &TileLayout| {
        (0..layout.alignments().len())
            .map(|i| {
                let a = &layout.alignments()[i];
                (a.pre, a.post, a.displacement, layout.tiling_axis(i))
            })
            .collect::<Vec<_>>()
    };
    let boxes = TileLayout::from_boxes(unnamed(9), on_grid.positions().to_vec(), &[2, 1])
        .expect("a layout");
    assert_eq!(pairs(&boxes), pairs(&on_grid));
    assert_eq!(boxes.alignments().len(), 12);

    // a few voxels of stage error change no pair
    let jittered: Vec<[isize; 3]> = on_grid
        .positions()
        .iter()
        .enumerate()
        .map(|(i, p)| {
            [
                p[0] + (i % 2) as isize,
                p[1] + (i % 3) as isize - 1,
                p[2] - (i % 2) as isize * 3,
            ]
        })
        .collect();
    let boxes = TileLayout::from_boxes(unnamed(9), jittered, &[2, 1]).expect("a layout");
    let ends = |layout: &TileLayout| {
        layout
            .alignments()
            .iter()
            .map(|a| (a.pre, a.post))
            .collect::<Vec<_>>()
    };
    assert_eq!(ends(&boxes), ends(&on_grid));

    // the centre tile missing: the eight seams around it
    let ring: Vec<[isize; 3]> = on_grid
        .positions()
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 4)
        .map(|(_, p)| *p)
        .collect();
    let boxes = TileLayout::from_boxes(unnamed(8), ring, &[2, 1]).expect("a layout");
    assert_eq!(boxes.alignments().len(), 8);

    // a second row shifted by half a pitch in x
    let pitch_x = common::TILE[2] as isize - common::OVERLAP_X;
    let pitch_y = common::TILE[1] as isize - common::OVERLAP_Y;
    let staggered = vec![[0, 0, 0], [0, 0, pitch_x], [0, pitch_y, pitch_x / 2]];
    let boxes = TileLayout::from_boxes(unnamed(3), staggered, &[2, 1]).expect("a layout");
    assert_eq!(ends(&boxes), vec![(0, 1), (0, 2), (1, 2)]);
    assert_eq!(boxes.tiling_axis(0), Some(2));
    assert_eq!(boxes.tiling_axis(1), Some(1));
    assert_eq!(
        boxes.alignments()[2].displacement,
        [0, pitch_y, -pitch_x / 2]
    );
}

/// Tiles whose names carry no grid index, placed by their translations alone:
/// the set aligns the pairs whose boxes overlap and the stitch recovers the
/// planted offsets. Duplicated translations, and a directory with neither
/// translations nor names, are refused.
#[test]
fn translations_place_and_pair_tiles_without_grid_names() {
    let dir = common::scratch("tile_dir_boxes");
    let pitch = [
        (common::TILE[1] as isize - common::OVERLAP_Y) as f64,
        (common::TILE[2] as isize - common::OVERLAP_X) as f64,
    ];
    // a 2 x 2 mosaic; the true origins are the nominal ones moved by a few
    // voxels, so the rigid step has something to find
    let planted = [[0, 0, 0], [2, 3, -4], [1, -2, 3], [-1, 2, 2]];
    let cells = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let names = ["a.zarr", "b.zarr", "c.zarr", "d.zarr"];
    let grid_named = common::scratch("tile_dir_boxes_named");
    for (i, ((name, cell), offset)) in names.iter().zip(cells).zip(planted).enumerate() {
        let nominal = [0.0, cell[0] * pitch[0], cell[1] * pitch[1]];
        let origin: [isize; 3] = std::array::from_fn(|a| nominal[a] as isize + offset[a]);
        let tile = common::crop(origin, common::TILE, |_| [0, 0]);
        write_tile(
            &dir.join(name),
            &tile,
            [5.0 + nominal[0], -9.0 + nominal[1], 4.0 + nominal[2]],
        );
        // the same tiles named by grid index, at the origin
        write_tile(
            &grid_named.join(format!("tile_{:02}x{:02}.zarr", i / 2, i % 2)),
            &tile,
            [0.0; 3],
        );
    }
    let derived = from_directory(
        &dir,
        &TileDirOptions {
            overlap: Some([1.0, 1.0]),
            ..TileDirOptions::default()
        },
    )
    .expect("a tile set");
    assert_eq!(derived.positions_from, "translation");
    assert_eq!(derived.provenance["neighbours"], "boxes");
    assert!(
        derived.provenance["notes"][0]
            .as_str()
            .unwrap()
            .contains("--overlap is ignored"),
        "{}",
        derived.provenance
    );
    let set = &derived.tile_set;
    assert!(set.tiles.iter().all(|tile| tile.grid.is_empty()));
    assert_eq!(set.tiles[3].position, Some([0, 56, 40]));
    let json = set.to_json();
    assert_eq!(json["neighbours"], "boxes");
    assert!(json["tiles"][0].get("grid").is_none());

    // The stitch is the one the grid names and the overlap give.
    let by_grid = from_directory(
        &grid_named,
        &TileDirOptions {
            overlap: Some([common::OVERLAP_Y as f64, common::OVERLAP_X as f64]),
            ..TileDirOptions::default()
        },
    )
    .expect("a gridded set");
    assert_eq!(by_grid.provenance["neighbours"], "grid");
    let (stitched, _) = stitch_set(set, 0);
    let (gridded, _) = stitch_set(&by_grid.tile_set, 0);
    let ends = |s: &Stitched| {
        s.layout
            .alignments()
            .iter()
            .map(|a| (a.pre, a.post, a.displacement))
            .collect::<Vec<_>>()
    };
    assert_eq!(stitched.layout.alignments().len(), 4);
    assert_eq!(ends(&stitched), ends(&gridded));
    let relative = |s: &Stitched| {
        let placed = s.placed.as_ref().expect("a per-slice placement");
        placed
            .positions
            .iter()
            .map(|slices| {
                slices
                    .iter()
                    .map(|p| {
                        std::array::from_fn::<isize, 3, _>(|a| p[a] - placed.positions[0][0][a])
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(relative(&stitched), relative(&gridded));
    // and it found the planted offsets: tile b's first plane against a's
    let r = relative(&stitched);
    assert_eq!(r[1][0][1..], [planted[1][1], 40 + planted[1][2]]);

    // the same as a hand-written set with positions and no grid
    let file = dir.join("set.json");
    std::fs::write(&file, serde_json::to_string(&json).unwrap()).unwrap();
    let read = tile_stitching::io::TileSetFile::read(&file).expect("read back");
    assert_eq!(read.neighbours(), Neighbours::Boxes);
    let shapes = vec![common::TILE; 4];
    assert_eq!(
        read.layout(&shapes).unwrap().alignments(),
        set.layout(&shapes).unwrap().alignments()
    );

    // two tiles at one translation: refused
    let tile = common::crop([0, 0, 0], common::TILE, |_| [0, 0]);
    write_tile(&dir.join("e.zarr"), &tile, [5.0, -9.0, 4.0]);
    let refused = from_directory(&dir, &TileDirOptions::default()).expect_err("a duplicate");
    assert!(
        refused.to_string().contains("same translation"),
        "{refused}"
    );

    // no translations and no names: nothing places them
    let plain = common::scratch("tile_dir_plain");
    write_tile(&plain.join("a.zarr"), &tile, [0.0; 3]);
    write_tile(&plain.join("b.zarr"), &tile, [0.0; 3]);
    let refused = from_directory(
        &plain,
        &TileDirOptions {
            overlap: Some([1.0, 1.0]),
            ..TileDirOptions::default()
        },
    )
    .expect_err("nothing places them");
    assert!(refused.to_string().contains("grid index"), "{refused}");
    let refused = from_directory(
        &plain,
        &TileDirOptions {
            rows: Some((0, 1)),
            ..TileDirOptions::default()
        },
    )
    .expect_err("no names to select by");
    assert!(refused.to_string().contains("--rows"), "{refused}");
}

/// A stitch written as a positions file and read back fuses to the same
/// volume, any box of it equals that crop of the volume, and the same
/// positions carried from level 1 to level 0 fuse the tiles' level 0 in the
/// frame the level-1 volume describes.
#[test]
fn a_stored_layout_fuses_again_over_a_box_and_at_another_level() {
    let dir = common::scratch("tile_dir_relayout");
    // Even in-plane offsets survive the halving to level 1, and the overlap
    // cut stays large enough there for the per-slice measurement (1500
    // voxels) to see signal.
    const SHAPE: [usize; 3] = [24, 200, 80];
    const OVERLAP: isize = 40;
    let tiles = [
        common::crop([0, 0, 0], SHAPE, |_| [0, 0]),
        common::crop([2, 4, SHAPE[2] as isize - OVERLAP - 4], SHAPE, |_| [0, 0]),
    ];
    write_tile(&dir.join("tile_00x00.zarr"), &tiles[0], [0.0; 3]);
    write_tile(&dir.join("tile_00x01.zarr"), &tiles[1], [0.0; 3]);
    let fuse = FuseParams {
        weighting: tile_stitching::fuse::Weighting::Pyramid,
        ..FuseParams::default()
    };

    for level in [0, 1] {
        let options = TileDirOptions {
            overlap: Some([OVERLAP as f64, OVERLAP as f64]),
            level,
            ..TileDirOptions::default()
        };
        let derived = from_directory(&dir, &options).expect("a tile set");
        let tiles_l = read_all(&derived.tile_set, level);
        let shapes = tiles_l.iter().map(Voxels::shape).collect::<Vec<_>>();
        let layout = derived.tile_set.layout(&shapes).expect("a layout");
        let mut params = params(&derived.tile_set.overlaps(&layout));
        params.fuse = fuse;
        let stitched = stitch(&tiles_l, layout.clone(), &params).expect("a stitch");
        // Tile 1 is where it was cropped, at this level.
        let placed = stitched.placed.as_ref().expect("a per-slice placement");
        let (a, b) = (&placed.positions[0][2], &placed.positions[1][0]);
        let step = 1 << level;
        assert_eq!(
            [b[1] - a[1], b[2] - a[2]],
            [4 / step, (SHAPE[2] as isize - OVERLAP - 4) / step],
            "level {level}"
        );
        let paths = derived
            .tile_set
            .tiles
            .iter()
            .map(|t| t.path.clone())
            .collect::<Vec<_>>();
        let mut json = positions_json(&stitched, &layout, &paths, 0);
        let voxel = if level == 0 {
            [1.0, 1.0, 1.0]
        } else {
            [1.0, 2.0, 2.0]
        };
        json["voxel_size"] = json!(voxel);

        let stored = StoredLayout::from_json(&json, None).expect("a stored layout");
        assert_eq!(stored.shape().expect("a shape"), stitched.fused.shape());
        let whole = stored
            .fuse(&fuse, None, 0, level)
            .expect("the whole volume");
        assert_eq!(whole, stitched.fused, "level {level}: fused again");
        let shape = whole.shape();
        let region = Region::new(&[3, 5, shape[2] / 2 - 9], &[shape[0] - 6, shape[1] - 9, 18]);
        let boxed = stored.fuse(&fuse, Some(&region), 0, level).expect("a box");
        assert_eq!(
            boxed,
            whole.slice_region(&region).expect("a crop"),
            "level {level}: a box"
        );
        if level == 0 {
            continue;
        }

        // Level 1 -> level 0: positions doubled in-plane.
        let at_zero = stored
            .rescaled(
                [1.0; 3],
                &tiles.iter().map(Voxels::shape).collect::<Vec<_>>(),
            )
            .expect("rescaled");
        assert_eq!(at_zero.origin[1], 2 * stored.origin[1]);
        let fused0 = at_zero.fuse(&fuse, None, 0, 0).expect("level 0");
        // Every fused voxel a tile covers is that tile's voxel: the doubled
        // positions are the true ones.
        let view = fused0.view::<u16>().expect("u16");
        for (index, tile) in at_zero.tiles.iter().enumerate() {
            let data = tiles[index].view::<u16>().expect("u16");
            for (k, at) in tile.positions.iter().enumerate() {
                for j in (0..tile.shape[1]).step_by(7) {
                    for i in (0..tile.shape[2]).step_by(5) {
                        let g = [
                            (at[0] - at_zero.origin[0]) as usize,
                            (at[1] - at_zero.origin[1]) as usize + j,
                            (at[2] - at_zero.origin[2]) as usize + i,
                        ];
                        assert_eq!(view[g], data[[k, j, i]], "tile {index} at {g:?}");
                    }
                }
            }
        }
        // The two frames agree: level-1 voxel u is the mean of level-0 voxels
        // 2u and 2u + 1, whose centres average to the level-1 centre.
        let level1 = read_tile(&paths[0], 0, 1, Some((0, 1))).expect("a tile");
        let level0 = read_tile(&paths[0], 0, 0, Some((0, 1))).expect("a tile");
        let t1 = fused_translation(
            level1.translation,
            level1.voxel_size,
            stored.anchor(),
            stored.origin,
            0,
            [0; 3],
        );
        let t0 = fused_translation(
            level0.translation,
            level0.voxel_size,
            at_zero.anchor(),
            at_zero.origin,
            0,
            [0; 3],
        );
        for a in 1..3 {
            let centre1 = t1[a];
            let centre0 = t0[a] + 0.5 * level0.voxel_size[a];
            assert!(
                (centre1 - centre0).abs() < 1e-9,
                "axis {a}: {t1:?} vs {t0:?}"
            );
        }
        assert!((t1[0] - t0[0]).abs() < 1e-9);
    }
}

/// A layout fitted on a level that halves z too (osf-sa3x8's pyramid) carried
/// to level 0: each plane stands for the two it covers, at twice its z and
/// its in-plane position doubled (a level-0 plane beyond twice the coarse
/// count, or a coarse plane's second beyond the level-0 count, is left out),
/// and the fusion puts every tile voxel where the scene has it.
/// Ratios that are not whole are refused.
#[test]
fn a_layout_on_a_level_that_halves_z_is_carried_to_level_0() {
    let dir = common::scratch("tile_dir_relayout_z");
    // an odd plane count, so the halving does not divide it
    const SHAPE: [usize; 3] = [25, 200, 80];
    const OVERLAP: isize = 40;
    let tiles = [
        common::crop([0, 0, 0], SHAPE, |_| [0, 0]),
        common::crop([2, 4, SHAPE[2] as isize - OVERLAP - 4], SHAPE, |_| [0, 0]),
    ];
    let halving = |path: &Path, tile: &Voxels| {
        ZarrEnvironment::build_multiscale(
            path,
            tile,
            &PyramidSpec::new(vec![[2, 2, 2]])
                .expect("a pyramid")
                .with_physical(PhysicalFrame::new([1.0, 1.0, 1.0], Some("micrometer"))),
        )
        .expect("a tile image");
    };
    halving(&dir.join("tile_00x00.zarr"), &tiles[0]);
    halving(&dir.join("tile_00x01.zarr"), &tiles[1]);
    let derived = from_directory(
        &dir,
        &TileDirOptions {
            overlap: Some([OVERLAP as f64, OVERLAP as f64]),
            level: 1,
            ..TileDirOptions::default()
        },
    )
    .expect("a tile set");
    let (stitched, _) = stitch_set(&derived.tile_set, 1);
    let paths = derived
        .tile_set
        .tiles
        .iter()
        .map(|t| t.path.clone())
        .collect::<Vec<_>>();
    let mut json = positions_json(&stitched, &stitched.layout, &paths, 0);
    json["voxel_size"] = json!([2.0, 2.0, 2.0]);
    let stored = StoredLayout::from_json(&json, None).expect("a stored layout");
    let coarse_planes = stored.tiles[0].shape[0];
    assert!(
        coarse_planes == 12 || coarse_planes == 13,
        "{coarse_planes}"
    );
    let planes = (2 * coarse_planes).min(25);
    let shapes = tiles.iter().map(Voxels::shape).collect::<Vec<_>>();
    let at_zero = stored.rescaled([1.0; 3], &shapes).expect("rescaled");
    for (tile, coarse) in at_zero.tiles.iter().zip(&stored.tiles) {
        assert_eq!(tile.shape[0], planes);
        for k in 0..planes {
            let p = coarse.positions[k / 2];
            assert_eq!(
                tile.positions[k],
                [2 * p[0] + (k % 2) as i64, 2 * p[1], 2 * p[2]]
            );
        }
    }
    assert_eq!(at_zero.origin, stored.origin.map(|v| 2 * v));
    let fuse = FuseParams {
        weighting: tile_stitching::fuse::Weighting::Pyramid,
        ..FuseParams::default()
    };
    let fused0 = at_zero.fuse(&fuse, None, 0, 0).expect("level 0");
    let view = fused0.view::<u16>().expect("u16");
    for (index, tile) in at_zero.tiles.iter().enumerate() {
        let data = tiles[index].view::<u16>().expect("u16");
        for (k, at) in tile.positions.iter().enumerate() {
            for j in (0..tile.shape[1]).step_by(7) {
                for i in (0..tile.shape[2]).step_by(5) {
                    let g = [
                        (at[0] - at_zero.origin[0]) as usize,
                        (at[1] - at_zero.origin[1]) as usize + j,
                        (at[2] - at_zero.origin[2]) as usize + i,
                    ];
                    assert_eq!(view[g], data[[k, j, i]], "tile {index} at {g:?}");
                }
            }
        }
    }
    // 2 -> 1.5 is not a whole factor
    let refused = stored
        .rescaled([1.5, 1.0, 1.0], &shapes)
        .expect_err("not a whole factor");
    assert!(refused.to_string().contains("whole factor"), "{refused}");
}

/// The CLI end to end: `--tile-dir` writes the derived tile set and a positions
/// file with the frame, and `--layout-from --region` writes a box whose OME-Zarr
/// frame is the box's place in the fused frame.
#[test]
fn the_cli_stitches_a_directory_and_fuses_a_box_in_the_same_frame() {
    let dir = common::scratch("tile_dir_cli");
    let (tiles, _) = common::pair([2, 4, -4], |_| [0, 0]);
    write_tile(&dir.join("tile_00x00.zarr"), &tiles[0], [0.0; 3]);
    write_tile(&dir.join("tile_00x01.zarr"), &tiles[1], [0.0; 3]);
    let binary = env!("CARGO_BIN_EXE_tile-stitching");
    let fused = dir.join("fused.zarr");
    let status = std::process::Command::new(binary)
        .args(["--tile-dir", dir.to_str().expect("utf-8")])
        .args(["--overlap", "24,30", "--level", "1", "--out-levels", "1"])
        .args([
            "--max-shift",
            "8,8",
            "--rigid-max-shift",
            "8,8",
            "--max-shift-z",
            "4",
        ])
        .arg("--out")
        .arg(&fused)
        .status()
        .expect("the CLI");
    assert!(status.success());
    let tile_set: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("fused.zarr.tiles.json")).expect("tiles"))
            .expect("JSON");
    assert_eq!(tile_set["overlap"], json!([15, 12]));
    assert_eq!(
        tile_set["derived"]["positions_from"],
        json!("overlap option")
    );
    let positions: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("fused.zarr.positions.json")).expect("positions"),
    )
    .expect("JSON");
    assert_eq!(positions["voxel_size"], json!([1.0, 2.0, 2.0]));
    assert_eq!(positions["level"], json!(1));

    let boxed = dir.join("box.zarr");
    let status = std::process::Command::new(binary)
        .arg("--layout-from")
        .arg(dir.join("fused.zarr.positions.json"))
        .args([
            "--level",
            "0",
            "--region",
            "2:20,10:60,20:90",
            "--out-levels",
            "1",
        ])
        .arg("--out")
        .arg(&boxed)
        .status()
        .expect("the CLI");
    assert!(status.success());
    let level1 = OmeVolumePyramid::open(&fused, 0, 0).expect("the fused image");
    let level0 = OmeVolumePyramid::open(&boxed, 0, 0).expect("the box");
    let translation = |pyramid: &OmeVolumePyramid| -> [f64; 3] {
        let row = pyramid.levels()[0]
            .coordinate_transformations
            .as_array()
            .expect("transformations")
            .iter()
            .find(|row| row["type"] == "translation")
            .expect("a translation")
            .clone();
        std::array::from_fn(|a| row["translation"][a].as_f64().expect("a number"))
    };
    let (t1, t0) = (translation(&level1), translation(&level0));
    // Box voxel 0 is fused level-0 voxel (2, 10, 20); level-1 voxel u covers
    // level-0 voxels 2u and 2u + 1, centred half a level-0 voxel above 2u.
    assert!((t0[0] - (t1[0] + 2.0)).abs() < 1e-9, "{t0:?} {t1:?}");
    for a in 1..3 {
        let start = [2.0, 10.0, 20.0][a];
        assert!(
            (t0[a] - (t1[a] - 0.5 + start)).abs() < 1e-9,
            "{t0:?} {t1:?}"
        );
    }
    let back = level0
        .attached_level(0)
        .expect("level 0")
        .read()
        .expect("the box");
    assert_eq!(back.shape(), [18, 50, 70]);
}

/// The converted tiles at hand: their names give the grids, and their
/// metadata carries no positions, so an overlap is needed (see the README).
#[test]
#[ignore = "reads the metadata of the OSF conversions under /husky/otherdataset/teresa/clearmap_em; run with --release --ignored"]
fn the_osf_conversions_give_their_grids_from_their_names() {
    let root = Path::new("/husky/otherdataset/teresa/clearmap_em");
    let cfos = root.join("osf-cztq2/cFos");
    if !cfos.exists() {
        eprintln!("no {}; skipped", cfos.display());
        return;
    }
    assert!(
        from_directory(&cfos, &TileDirOptions::default()).is_err(),
        "the conversions carry no translations"
    );
    let at_level1 = from_directory(
        &cfos,
        &TileDirOptions {
            level: 1,
            overlap: Some([120.0, 120.0]),
            ..TileDirOptions::default()
        },
    )
    .expect("a tile set");
    let set = &at_level1.tile_set;
    assert_eq!(set.tiles.len(), 40);
    assert_eq!(set.overlap, Some(vec![60, 60]));
    assert_eq!(
        set.tiles[7].grid,
        vec![2, 1],
        "tile_01x02 is column 2, row 1"
    );

    let vessels = root.join("osf-sa3x8/tiles");
    let derived = from_directory(
        &vessels,
        &TileDirOptions {
            overlap: Some([212.0, 66.0]),
            ..TileDirOptions::default()
        },
    )
    .expect("a tile set");
    assert_eq!(derived.tile_set.tiles.len(), 9);
    assert_eq!(derived.tile_set.overlap, Some(vec![66, 212]));
}

/// osf-sa3x8's nine tiles with the LaVision stage positions img2omezarr
/// wrote (`set-positions --from-originals`, on a metadata-only mirror): they
/// are placed and paired from their translations alone, with no overlap and
/// without reading the names' grid.
#[test]
#[ignore = "reads the metadata-only mirror /husky/henriksson/lightsheet-reference/positions/osf-sa3x8/tiles; run with --release --ignored"]
fn the_osf_sa3x8_mirror_is_placed_by_its_stage_positions() {
    let tiles = Path::new("/husky/henriksson/lightsheet-reference/positions/osf-sa3x8/tiles");
    if !tiles.exists() {
        eprintln!("no {}; skipped", tiles.display());
        return;
    }
    let derived = from_directory(tiles, &TileDirOptions::default()).expect("a tile set");
    assert_eq!(derived.positions_from, "translation");
    let set = &derived.tile_set;
    assert_eq!(set.tiles.len(), 9);
    let shapes = vec![[3369, 1304, 404]; 9];
    let layout = set.layout(&shapes).expect("a layout");
    // a 3 x 3 grid: 6 column seams and 6 row seams
    assert_eq!(layout.alignments().len(), 12);
    // stage pitch 590.85 um in x (363.6 voxels of 1.625) and 1907.1 um in y
    // (1173.6 voxels)
    let a = set.tiles[0].position.expect("a position");
    let b = set.tiles[1].position.expect("a position");
    let c = set.tiles[3].position.expect("a position");
    assert_eq!([b[1] - a[1], b[2] - a[2]], [0, 364]);
    assert_eq!([c[1] - a[1], c[2] - a[2]], [1174, 0]);
}
