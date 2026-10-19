// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The rigid half: tiles and a nominal overlap in, one integer position per
//! tile out. Every test here is on constructed data; the last runs on a real
//! tile pair and is ignored by default.

mod common;

use ndarray::{Array2, Array3};
use tile_stitching::layout::{
    self, Alignment, FixedSource, PlacementMethod, RigidPlacementParams, TileLayout, TileSource,
};
use tile_stitching::rigid::{self, Background, Face, MinimumCount, RigidParams};

/// A deterministic pattern with no symmetry the correlation could latch onto.
fn pattern(row: isize, column: isize, plane: isize) -> f64 {
    (common::hash(row, column, plane) % 4096) as f64
}

/// Two 80 x 100 x 60 tiles adjacent on grid axis 0 (volume axis 0), whose
/// facing faces differ by a planted shift on the other two axes. Plane `p` of
/// the second tile's face is plane `14 + p` of the first's, so the match at the
/// planted shift is exact.
fn planted_volumes(shift: [isize; 2]) -> (Array3<f64>, Array3<f64>) {
    let pre = Array3::from_shape_fn((80, 100, 60), |(i, j, k)| {
        pattern(i as isize, j as isize, k as isize)
    });
    let post = Array3::from_shape_fn((80, 100, 60), |(i, j, k)| {
        pattern(
            i as isize + 14,
            j as isize + shift[0],
            k as isize + shift[1],
        )
    });
    (pre, post)
}

fn planted_layout() -> TileLayout {
    TileLayout::on_grid(
        vec![
            TileSource {
                shape: [80, 100, 60],
                tile_position: vec![0, 0],
            },
            TileSource {
                shape: [80, 100, 60],
                tile_position: vec![1, 0],
            },
        ],
        &[66, 212],
        &[0, 1],
    )
    .expect("a two-tile grid")
}

fn planted_params() -> RigidParams {
    RigidParams::from_overlaps(&[66, 212], &[0, 1])
}

/// Three tiles round a cycle whose pairwise offsets do not close: `0 -> 1` and
/// `1 -> 2` put the third tile at `(10, 10, 0)` and `0 -> 2` at `(10, 10, -3)`.
fn cycle_alignments() -> Vec<Alignment> {
    let alignment = |pre, post, displacement| Alignment {
        pre,
        post,
        displacement,
        quality: -1.0,
    };
    vec![
        alignment(0, 1, [10, 0, 0]),
        alignment(1, 2, [0, 10, 0]),
        alignment(0, 2, [10, 10, -3]),
    ]
}

#[test]
fn the_window_is_the_declared_depth_at_the_facing_ends() {
    assert_eq!(rigid::face_range(404, 66, Face::Trailing), 338..404);
    assert_eq!(rigid::face_range(404, 66, Face::Leading), 0..66);
    // A depth wider than the tile takes the whole tile.
    assert_eq!(rigid::face_range(40, 66, Face::Trailing), 0..40);
    assert_eq!(rigid::face_range(40, 66, Face::Leading), 0..40);
}

#[test]
fn the_grid_positions_are_the_shape_less_the_overlap() {
    let layout = TileLayout::on_grid(
        vec![
            TileSource {
                shape: [404, 1304, 240],
                tile_position: vec![0, 0],
            },
            TileSource {
                shape: [404, 1304, 240],
                tile_position: vec![1, 0],
            },
        ],
        &[66, 212],
        &[0, 1],
    )
    .expect("a pair");
    assert_eq!(layout.positions(), [[0, 0, 0], [338, 0, 0]]);
    assert_eq!(layout.alignments().len(), 1);
    assert_eq!(layout.alignments()[0].displacement, [338, 0, 0]);
    assert_eq!(layout.shift(0), Some([0, 0, 0]));

    // A 2 x 2 grid uses the second overlap on the second grid axis.
    let square = TileLayout::on_grid(
        (0..4)
            .map(|index| TileSource {
                shape: [100, 80, 20],
                tile_position: vec![index % 2, index / 2],
            })
            .collect(),
        &[10, 20],
        &[0, 1],
    )
    .expect("a 2 x 2 grid");
    assert_eq!(
        square.positions(),
        [[0, 0, 0], [90, 0, 0], [0, 60, 0], [90, 60, 0]]
    );
    assert_eq!(square.alignments().len(), 4);

    // The same grid tiled along x then y of a (z, y, x) volume.
    let zyx = TileLayout::on_grid(
        (0..4)
            .map(|index| TileSource {
                shape: [20, 80, 100],
                tile_position: vec![index % 2, index / 2],
            })
            .collect(),
        &[10, 20],
        &[2, 1],
    )
    .expect("a 2 x 2 grid in x and y");
    assert_eq!(
        zyx.positions(),
        [[0, 0, 0], [0, 0, 90], [0, 60, 0], [0, 60, 90]]
    );
}

/// The shift found is the shift planted, and the tiling axis is never
/// corrected.
#[test]
fn the_correlation_finds_a_planted_shift() {
    let (pre, post) = planted_volumes([3, -2]);
    let mut layout = planted_layout();
    let pair = rigid::align_pair(&layout, 0, pre.view(), post.view(), &planted_params())
        .expect("the correlation");
    assert_eq!(pair.shift, [0, 3, -2]);
    assert!(
        pair.quality.abs() < 1e-6,
        "an exact match scores zero: {}",
        pair.quality
    );

    layout
        .set_shift(0, pair.shift, pair.quality)
        .expect("write");
    let placed = layout
        .place(&RigidPlacementParams::default())
        .expect("the placement");
    // The grid put the second tile at 80 - 66 = 14; the correlation moves it
    // 3 and -2, and the lowering takes the -2 back out of both tiles.
    assert_eq!(placed, vec![[0, 0, 2], [14, 3, 0]]);
}

/// `align_layout` and `place_in_place` give the same answer as the pieces.
#[test]
fn the_layout_driver_gives_the_same_answer_as_the_pieces() {
    let (pre, post) = planted_volumes([3, -2]);
    let mut layout = planted_layout();
    rigid::align_layout(&mut layout, &[pre.view(), post.view()], &planted_params())
        .expect("every pair aligned");
    assert_eq!(layout.shift(0), Some([0, 3, -2]));
    layout
        .place_in_place(&RigidPlacementParams::default())
        .expect("the placement");
    assert_eq!(layout.positions(), [[0, 0, 2], [14, 3, 0]]);
}

/// A pair the background check refuses is dropped, not placed at zero: both
/// tiles keep the grid's positions.
#[test]
fn a_refused_pair_is_dropped_rather_than_placed() {
    let (pre, post) = planted_volumes([3, -2]);
    let mut layout = planted_layout();
    let mut params = planted_params();
    params.background = Background::Counted {
        level: f64::INFINITY,
        minimum: MinimumCount::Absolute(1),
    };
    let pair = rigid::align_pair(&layout, 0, pre.view(), post.view(), &params)
        .expect("the correlation still returns");
    assert_eq!(pair.shift, [0, 0, 0]);
    assert_eq!(pair.quality, f64::NEG_INFINITY);
    layout
        .set_shift(0, pair.shift, pair.quality)
        .expect("write");
    let placed = layout
        .place(&RigidPlacementParams::default())
        .expect("the placement");
    assert_eq!(placed, layout.positions());
}

/// A search range wider than the projections is clamped to the shifts that
/// overlap, and still reaches a planted shift near its edge.
#[test]
fn a_narrow_overlap_still_reaches_the_planted_shift() {
    let pre = Array2::from_shape_fn((20, 18), |(i, j)| pattern(i as isize, j as isize, 0));
    let post = Array2::from_shape_fn((20, 18), |(i, j)| pattern(i as isize + 15, j as isize, 0));
    let found = rigid::align_projections(
        pre.view(),
        [0, 0],
        post.view(),
        [0, 0],
        [(-30, 30), (-20, 20)],
        &RigidParams::default(),
    )
    .expect("the correlation");
    assert_eq!(found.shift, [15, 0]);
}

/// Clipping matters where a saturated structure would otherwise decide: half
/// the columns are bright and match at a row shift of 3, half are dim and
/// match at -4. Unclipped the bright half decides; clipped to 200 it is a
/// constant in both windows and the dim half decides.
#[test]
fn the_clip_decides_where_saturation_would() {
    let bright = |row: isize, column: isize| 1000.0 + pattern(row, column, 1) % 3000.0;
    let dim = |row: isize, column: isize| pattern(row, column, 2) % 50.0;
    let pre = Array2::from_shape_fn((40, 20), |(i, j)| {
        let (i, j) = (i as isize, j as isize);
        if j < 10 {
            bright(i, j)
        } else {
            dim(i, j)
        }
    });
    let post = Array2::from_shape_fn((40, 20), |(i, j)| {
        let (i, j) = (i as isize, j as isize);
        if j < 10 {
            bright(i + 3, j)
        } else {
            dim(i - 4, j)
        }
    });
    let shifts = [(-8, 8), (0, 0)];
    let shipped = RigidParams::default();
    let clipped = RigidParams {
        clip: Some((None, Some(200.0))),
        ..shipped
    };
    let run = |params: &RigidParams| {
        rigid::align_projections(pre.view(), [0, 0], post.view(), [0, 0], shifts, params)
            .expect("the correlation")
            .shift
    };
    assert_eq!(run(&shipped), [3, 0]);
    assert_eq!(run(&clipped), [-4, 0]);
}

/// Normalising matters where the two tiles were acquired at different gains:
/// with one tile at a hundredth of the other over a strong ramp, the plain
/// squared difference is least where the overlap sits in the brighter tile's
/// darkest rows, and the normalised one finds the planted shift.
#[test]
fn normalising_aligns_tiles_of_different_gain() {
    let scene = |row: isize, column: isize| 300.0 * row as f64 + pattern(row, column, 0);
    let pre = Array2::from_shape_fn((40, 30), |(i, j)| scene(i as isize, j as isize));
    let post = Array2::from_shape_fn((40, 30), |(i, j)| 0.01 * scene(i as isize + 3, j as isize));
    let shifts = [(-8, 8), (0, 0)];
    let plain = RigidParams::default();
    let normalised = RigidParams {
        normalize: true,
        ..plain
    };
    let run = |params: &RigidParams| {
        rigid::align_projections(pre.view(), [0, 0], post.view(), [0, 0], shifts, params)
            .expect("the correlation")
            .shift
    };
    assert_eq!(run(&plain), [-8, 0]);
    assert_eq!(run(&normalised), [3, 0]);
}

/// Least squares and the tree walk disagree round a cycle whose offsets do
/// not close: the tree takes two edges as read, the fit spreads the third
/// edge's 3 over all three tiles.
#[test]
fn the_two_reconciliations_disagree_on_a_cycle() {
    let alignments = cycle_alignments();
    let positions = vec![[0, 0, 0]; 3];
    let optimization = layout::place(&positions, &alignments, 3, &RigidPlacementParams::default())
        .expect("the least-squares placement");
    let tree_params = RigidPlacementParams {
        method: PlacementMethod::Tree,
        ..RigidPlacementParams::default()
    };
    let tree = layout::place(&positions, &alignments, 3, &tree_params).expect("the tree placement");
    assert_eq!(tree, vec![[0, 0, 0], [10, 0, 0], [10, 10, 0]]);
    assert_eq!(optimization, vec![[0, 0, 2], [10, 0, 1], [10, 10, 0]]);
}

/// The least-squares fit rounds a half to even.
#[test]
fn the_least_squares_solution_rounds_half_to_even() {
    let alignment = |displacement| Alignment {
        pre: 0,
        post: 1,
        displacement,
        quality: -1.0,
    };
    let alignments = vec![alignment([0, 0, 0]), alignment([1, 0, 0])];
    let placed = layout::place(
        &[[0, 0, 0]; 2],
        &alignments,
        2,
        &RigidPlacementParams::default(),
    )
    .expect("the placement");
    assert_eq!(placed, vec![[0, 0, 0], [0, 0, 0]], "0.5 rounds to 0");
}

/// With the lowering on, the pinned source cannot be seen: both pins are
/// translations and the lowering removes translations.
#[test]
fn the_pinned_source_is_seen_only_without_the_lowering() {
    let alignments = cycle_alignments();
    let positions = vec![[0, 0, 0]; 3];
    let run = |fixed_source, lower_to_origin| {
        layout::place(
            &positions,
            &alignments,
            3,
            &RigidPlacementParams {
                fixed_source,
                lower_to_origin,
                ..RigidPlacementParams::default()
            },
        )
        .expect("the placement")
    };
    assert_eq!(
        run(FixedSource::Middle, true),
        run(FixedSource::First, true)
    );
    assert_ne!(
        run(FixedSource::Middle, false),
        run(FixedSource::First, false)
    );
}

/// A quality floor above every alignment drops them all; every tile keeps its
/// position and the lowering still runs.
#[test]
fn a_quality_floor_above_every_alignment_places_nothing() {
    let positions = vec![[1, 2, 3], [11, 2, 3], [11, 12, 3]];
    let params = RigidPlacementParams {
        min_quality: 0.0,
        ..RigidPlacementParams::default()
    };
    let placed = layout::place(&positions, &cycle_alignments(), 3, &params).expect("the placement");
    assert_eq!(placed, vec![[0, 0, 0], [10, 0, 0], [10, 10, 0]]);
}

/// A per-column maximum is associative, so any cut of the slab folds to the
/// same projection: along the projected axis, across it, and both.
#[test]
fn the_projection_folds_the_same_under_every_cut() {
    let slab = Array3::from_shape_fn((9, 7, 5), |(i, j, k)| {
        f64::from((i * 31 + j * 17 + k * 7) as u32 % 23) - 11.0
    });
    let resident = rigid::project_max(slab.view(), 0).expect("the resident projection");
    let project = |planes: std::ops::Range<usize>, rows: std::ops::Range<usize>| {
        rigid::project_max(slab.slice(ndarray::s![planes, rows, ..]), 0).expect("a partial")
    };

    let mut folded = Array2::from_elem(resident.raw_dim(), f64::NEG_INFINITY);
    for start in (0..9).step_by(4) {
        let partial = project(start..(start + 4).min(9), 0..7);
        rigid::fold_projection(&mut folded, partial.view()).expect("the fold");
    }
    assert_eq!(folded, resident, "cut along the projected axis");

    let mut assembled = Array2::from_elem(resident.raw_dim(), f64::NAN);
    for row in (0..7).step_by(3) {
        let end = (row + 3).min(7);
        assembled
            .slice_mut(ndarray::s![row..end, ..])
            .assign(&project(0..9, row..end));
    }
    assert_eq!(assembled, resident, "cut across the projected axis");

    let mut both = Array2::from_elem(resident.raw_dim(), f64::NEG_INFINITY);
    for start in (0..9).step_by(2) {
        let mut plane = Array2::from_elem(resident.raw_dim(), f64::NEG_INFINITY);
        for row in (0..7).step_by(5) {
            let end = (row + 5).min(7);
            plane
                .slice_mut(ndarray::s![row..end, ..])
                .assign(&project(start..(start + 2).min(9), row..end));
        }
        rigid::fold_projection(&mut both, plane.view()).expect("the fold");
    }
    assert_eq!(both, resident, "cut both ways");
}

/// `+0.0` beats `-0.0` whichever comes first.
#[test]
fn the_projection_orders_the_two_zeroes() {
    let slab = Array3::from_shape_vec((2, 1, 1), vec![-0.0f64, 0.0]).expect("two zeroes");
    let projection = rigid::project_max(slab.view(), 0).expect("the projection");
    assert!(projection[[0, 0]].is_sign_positive());
    let reversed = Array3::from_shape_vec((2, 1, 1), vec![0.0f64, -0.0]).expect("two zeroes");
    assert_eq!(
        rigid::project_max(reversed.view(), 0).expect("the projection"),
        projection
    );
}

/// The synthetic pair the end-to-end tests use: the rigid step recovers the
/// offsets transverse to the tiling axis and leaves the tiling axis nominal.
#[test]
fn the_rigid_step_recovers_the_transverse_offset_of_cropped_tiles() {
    let (tiles, mut layout) = common::pair([2, 3, -4], |_| [0, 0]);
    let params = tile_stitching::StitchParams {
        rigid: RigidParams::from_overlaps(&[common::OVERLAP_X as usize], &[2]),
        ..tile_stitching::StitchParams::default()
    };
    tile_stitching::align_rigid(&tiles, &mut layout, &params).expect("the rigid step");
    let nominal_x = common::TILE[2] as isize - common::OVERLAP_X;
    assert_eq!(layout.positions(), [[0, 0, 0], [2, 3, nominal_x]]);
    assert!(layout.alignments()[0].quality.is_finite());
}

/// Two neighbouring real tiles, cropped to 240 planes: the rigid step keeps the
/// nominal x spacing and finds a transverse correction inside the search range.
#[test]
#[ignore = "reads two light-sheet tiles (about 500 MB) from TILE_STITCHING_TILES or /husky; run with --release --ignored"]
fn a_real_pair_is_aligned_within_the_search_range() {
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
        rigid: RigidParams::from_overlaps(&[66, 212], &[2, 1]),
        ..tile_stitching::StitchParams::default()
    };
    tile_stitching::align_rigid(&tiles, &mut layout, &params).expect("the rigid step");
    let shift = layout.alignments()[0].displacement;
    eprintln!(
        "displacement {shift:?} quality {}",
        layout.alignments()[0].quality
    );
    assert_eq!(
        shift[2],
        shape[2] as isize - 66,
        "the tiling axis is not corrected"
    );
    assert!(shift[0].abs() <= 20 && shift[1].abs() <= 30, "{shift:?}");
    assert!(layout.alignments()[0].quality.is_finite());
}
