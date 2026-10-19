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
use tile_stitching::io::{positions_json, read_tile};
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
