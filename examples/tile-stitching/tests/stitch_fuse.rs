// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The fusion: placed tiles written into one volume, checked against the
//! blend's definition and across block lattices.

mod common;

use blockflow::geometry::BlockGrid;
use blockflow::region::Region;
use blockflow::voxels::Voxels;
use ndarray::Array3;
use tile_stitching::fuse::{
    blocked, fuse_region, resident, FuseParams, Method, PlacedSource, Placement, Weighting,
};

/// Three 9 x 7 tiles in one plane, at `(0, 0)`, `(2, 0)` and `(4, 3)`: regions
/// covered by one, two and three tiles.
fn three_source_plane() -> (Vec<Voxels>, Placement, [[i64; 2]; 3]) {
    let shape = [9usize, 7, 1];
    let positions = [[0i64, 0], [2, 0], [4, 3]];
    let mut tiles = Vec::new();
    let mut sources = Vec::new();
    for (index, position) in positions.iter().enumerate() {
        tiles.push(Voxels::U16(Array3::from_shape_fn(
            (shape[0], shape[1], shape[2]),
            |(i, j, _)| (((i * 7919 + j * 104_729 + index * 15_485_863) % 60_013) + 11) as u16,
        )));
        sources.push(
            PlacedSource::all_valid(shape, 2, vec![[position[0], position[1], 0]])
                .expect("a placed source"),
        );
    }
    (
        tiles,
        Placement::new(2, sources).expect("a placement"),
        positions,
    )
}

/// The weighted mean written out from its definition: every tile covering a
/// voxel, weighted by one plus its distance to that tile's nearest face.
fn by_definition(
    tiles: &[Voxels],
    positions: &[[i64; 2]],
    shape: [usize; 2],
    weighting: Weighting,
) -> Vec<Vec<u16>> {
    let mut out = vec![vec![0u16; shape[1]]; shape[0]];
    for (u, row) in out.iter_mut().enumerate() {
        for (v, slot) in row.iter_mut().enumerate() {
            let (mut numerator, mut denominator) = (0.0, 0.0);
            for (tile, position) in tiles.iter().zip(positions) {
                let view = tile.view::<u16>().expect("u16");
                let (i, j) = (u as i64 - position[0], v as i64 - position[1]);
                let (rows, cols) = (view.shape()[0] as i64, view.shape()[1] as i64);
                if i < 0 || j < 0 || i >= rows || j >= cols {
                    continue;
                }
                let w = match weighting {
                    Weighting::Uniform => 1.0,
                    Weighting::Pyramid => (i.min(rows - 1 - i).min(j).min(cols - 1 - j) + 1) as f64,
                };
                numerator += w * view[[i as usize, j as usize, 0]] as f64;
                denominator += w;
            }
            if denominator > 0.0 {
                *slot = (numerator / denominator).round() as u16;
            }
        }
    }
    out
}

fn plane_of(volume: &Voxels) -> Vec<Vec<u16>> {
    let view = volume.view::<u16>().expect("u16");
    (0..view.shape()[0])
        .map(|i| (0..view.shape()[1]).map(|j| view[[i, j, 0]]).collect())
        .collect()
}

fn differing(left: &Voxels, right: &Voxels) -> usize {
    let (a, b) = (
        left.view::<u16>().expect("left"),
        right.view::<u16>().expect("right"),
    );
    assert_eq!(a.shape(), b.shape());
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

#[test]
fn three_sources_are_blended_by_the_definition() {
    let (tiles, placement, positions) = three_source_plane();
    assert_eq!(placement.shape().expect("a shape"), [13, 10, 1]);
    for weighting in [Weighting::Pyramid, Weighting::Uniform] {
        let params = FuseParams {
            method: Method::Interpolation,
            weighting,
        };
        let fused = resident(&params, &tiles, &placement).expect("the fusion");
        assert_eq!(
            plane_of(&fused),
            by_definition(&tiles, &positions, [13, 10], weighting),
            "{weighting:?}"
        );
    }
    let pyramid = resident(&FuseParams::default(), &tiles, &placement).expect("the fusion");
    let uniform = resident(
        &FuseParams {
            weighting: Weighting::Uniform,
            ..FuseParams::default()
        },
        &tiles,
        &placement,
    )
    .expect("the fusion");
    assert!(differing(&pyramid, &uniform) > 0, "the weighting is read");
}

/// The unweighted methods on a two-tile overlap of known values. The mean
/// rounds to the nearest integer.
#[test]
fn the_unweighted_methods_are_their_definitions() {
    let shape = [2usize, 1, 1];
    let tiles = vec![
        Voxels::U16(Array3::from_shape_fn((2, 1, 1), |(i, _, _)| [10u16, 20][i])),
        Voxels::U16(Array3::from_shape_fn((2, 1, 1), |(i, _, _)| {
            [201u16, 25][i]
        })),
    ];
    let placement = Placement::new(
        2,
        vec![
            PlacedSource::all_valid(shape, 2, vec![[0, 0, 0]]).expect("a source"),
            PlacedSource::all_valid(shape, 2, vec![[1, 0, 0]]).expect("a source"),
        ],
    )
    .expect("a placement");
    let at = |method| {
        let out = resident(
            &FuseParams {
                method,
                weighting: Weighting::Pyramid,
            },
            &tiles,
            &placement,
        )
        .expect("a fusion");
        let view = out.view::<u16>().expect("u16");
        assert_eq!(view.shape(), [3, 1, 1]);
        [view[[0, 0, 0]], view[[1, 0, 0]], view[[2, 0, 0]]]
    };
    assert_eq!(at(Method::Max), [10, 201, 25]);
    assert_eq!(at(Method::Min), [10, 20, 25]);
    assert_eq!(at(Method::Mean), [10, 111, 25], "(20 + 201) / 2 rounds up");
    // Both tiles are one voxel deep in y, so every pyramid weight is one.
    assert_eq!(at(Method::Interpolation), [10, 111, 25]);
}

/// Floating-point tiles are blended without rounding.
#[test]
fn floating_point_tiles_are_not_rounded() {
    let shape = [2usize, 1, 1];
    let tiles = vec![
        Voxels::F32(Array3::from_elem((2, 1, 1), 1.0f32)),
        Voxels::F32(Array3::from_elem((2, 1, 1), 2.0f32)),
    ];
    let placement = Placement::new(
        2,
        vec![
            PlacedSource::all_valid(shape, 2, vec![[0, 0, 0]]).expect("a source"),
            PlacedSource::all_valid(shape, 2, vec![[1, 0, 0]]).expect("a source"),
        ],
    )
    .expect("a placement");
    let out = resident(
        &FuseParams {
            method: Method::Mean,
            weighting: Weighting::Uniform,
        },
        &tiles,
        &placement,
    )
    .expect("a fusion");
    assert_eq!(out.view::<f32>().expect("f32")[[1, 0, 0]], 1.5);
}

/// Tiles cropped from one scene at their true offsets fuse back into the
/// scene: where they overlap they agree, so any weighted mean of them is the
/// value they share.
#[test]
fn tiles_cropped_from_one_scene_fuse_back_into_it() {
    let origins = [[0isize, 0, 0], [2, 3, 36]];
    let tiles = origins
        .iter()
        .map(|&origin| common::crop(origin, common::TILE, |_| [0, 0]))
        .collect::<Vec<_>>();
    let placement =
        Placement::from_rigid(&origins, &[common::TILE; 2], 0).expect("a rigid placement");
    let fused = resident(&FuseParams::default(), &tiles, &placement).expect("a fusion");
    assert_eq!(fused.shape(), [26, 83, 106]);
    let view = fused.view::<u16>().expect("u16");
    let covered = |origin: [isize; 3], at: [usize; 3]| {
        (0..3).all(|a| {
            let local = at[a] as isize - origin[a];
            (0..common::TILE[a] as isize).contains(&local)
        })
    };
    for ((z, y, x), &value) in view.indexed_iter() {
        let at = [z, y, x];
        if origins.iter().any(|&origin| covered(origin, at)) {
            assert_eq!(
                value,
                common::scene(z as isize, y as isize, x as isize),
                "{at:?}"
            );
        } else {
            assert_eq!(value, 0, "{at:?} is covered by no tile");
        }
    }
}

/// A tile's unusable plane is left out: where no other tile covers it, the
/// volume is zero.
#[test]
fn an_unusable_plane_is_left_out() {
    let shape = [3usize, 2, 2];
    let tiles = vec![Voxels::U16(Array3::from_elem((3, 2, 2), 50))];
    let positions = (0..3).map(|k| [k as i64, 0, 0]).collect();
    let placement = Placement::new(
        0,
        vec![PlacedSource::new(shape, 0, positions, vec![0, -2, 2]).expect("a source")],
    )
    .expect("a placement");
    let fused = resident(&FuseParams::default(), &tiles, &placement).expect("a fusion");
    let view = fused.view::<u16>().expect("u16");
    assert!(view
        .index_axis(ndarray::Axis(0), 0)
        .iter()
        .all(|&v| v == 50));
    assert!(view.index_axis(ndarray::Axis(0), 1).iter().all(|&v| v == 0));
    assert!(view
        .index_axis(ndarray::Axis(0), 2)
        .iter()
        .all(|&v| v == 50));
}

/// The volume is the same on every lattice, including ones whose boundaries
/// fall inside the overlap and one a voxel thick across the seam.
#[test]
fn the_lattices_cut_the_seam() {
    let origins = [[0isize, 0, 0], [2, 3, 36]];
    let tiles = origins
        .iter()
        .map(|&origin| common::crop(origin, common::TILE, |k| [0, (k % 3) as isize]))
        .collect::<Vec<_>>();
    // Each plane of the second tile placed where its drift put it.
    let drifting = PlacedSource::all_valid(
        common::TILE,
        0,
        (0..common::TILE[0])
            .map(|k| [2 + k as i64, 3, 36 + (k % 3) as i64])
            .collect(),
    )
    .expect("a drifting source");
    let placement = Placement::new(
        0,
        vec![
            PlacedSource::rigid(common::TILE, 0, [0, 0, 0]).expect("a source"),
            drifting,
        ],
    )
    .expect("a placement");
    let shape = placement.shape().expect("a shape");
    let params = FuseParams::default();
    let whole = resident(&params, &tiles, &placement).expect("the whole volume");
    for block in [
        shape,
        [5, 17, 36],
        [26, 83, 53],
        [26, 83, 70],
        [7, 9, 11],
        [26, 83, 1],
    ] {
        let grid = BlockGrid::new(shape, block).expect("a grid");
        let cut = blocked(&params, &tiles, &placement, &grid).expect("a blocked fusion");
        assert_eq!(differing(&whole, &cut), 0, "block {block:?}");
    }
    let region = Region::new(&[3, 10, 30], &[5, 20, 40]);
    let piece = fuse_region(&params, &tiles, &placement, &region).expect("a region");
    assert_eq!(
        piece,
        whole
            .slice_region(&region)
            .expect("the same region of the whole"),
    );
}
