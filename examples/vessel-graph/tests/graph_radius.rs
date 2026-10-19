// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The radius walk: one stopping offset, written four ways.

use std::path::PathBuf;

use blockflow::npy;
use blockflow::ops::rows::Limit;
use blockflow::ops::walk::walk_from;
use blockflow::voxels::Voxels;
use blockflow_vessel_graph::graph::columns::norm;
use blockflow_vessel_graph::graph::radius::NOT_FOUND;
use blockflow_vessel_graph::graph::{build, BuildParams, RadiusParams, RadiusSearch};
use ndarray::{Array3, Ix3};

fn ball(volume: [usize; 3], centre: [usize; 3], radius: f64) -> Voxels {
    let mut mask = Array3::from_elem((volume[0], volume[1], volume[2]), false);
    for ((i, j, k), slot) in mask.indexed_iter_mut() {
        let at = [i, j, k];
        let distance = (0..3)
            .map(|axis| (at[axis] as f64 - centre[axis] as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        *slot = distance <= radius;
    }
    Voxels::from(mask)
}

fn search(maximum: usize) -> RadiusSearch {
    RadiusSearch::new(RadiusParams {
        max_radius: [maximum; 3],
        ..RadiusParams::default()
    })
    .expect("a search")
}

#[test]
fn the_scalar_is_the_norm_of_the_axial() {
    let source = ball([21, 21, 21], [10, 10, 10], 4.0);
    let points: Vec<[u32; 3]> = (6..15).map(|i| [i, 10, 10]).collect();
    let measured = search(10)
        .measure(&source, &points, [1.6, 1.6, 2.5])
        .expect("a measurement");
    for row in 0..measured.len() {
        assert_eq!(measured.radii[row], norm(&measured.radii_axial[row]));
        assert_eq!(
            measured.radius_units[row],
            norm(&measured.radius_units_axial[row])
        );
    }
    // The norm and not the mean.
    assert!(measured
        .radii
        .iter()
        .zip(&measured.radii_axial)
        .any(|(scalar, axial)| *scalar != (axial[0] + axial[1] + axial[2]) / 3.0));
}

#[test]
fn the_units_split_is_live_under_an_anisotropic_spacing() {
    let source = ball([31, 31, 31], [15, 15, 15], 6.0);
    let points: Vec<[u32; 3]> = (10..21).map(|i| [i, 15, 15]).collect();
    let stage = search(12);
    let isotropic = stage
        .measure(&source, &points, [1.0; 3])
        .expect("a measurement");
    assert_eq!(isotropic.radii, isotropic.radius_units);
    assert_eq!(isotropic.radii_axial, isotropic.radius_units_axial);

    let spacing = [1.625, 1.625, 3.0];
    let anisotropic = stage
        .measure(&source, &points, spacing)
        .expect("a measurement");
    assert_ne!(anisotropic.radii, anisotropic.radius_units);
    assert_eq!(anisotropic.radii_axial, isotropic.radii_axial);
    for row in 0..points.len() {
        for axis in 0..3 {
            assert_eq!(
                anisotropic.radius_units_axial[row][axis],
                isotropic.radii_axial[row][axis] * spacing[axis]
            );
        }
    }
}

#[test]
fn a_vertex_on_background_gets_radius_zero() {
    let source = ball([21, 21, 21], [10, 10, 10], 4.0);
    let measured = search(8)
        .measure(&source, &[[0, 0, 0], [10, 10, 10]], [1.0; 3])
        .expect("a measurement");
    assert_eq!(measured.radii[0], 0.0);
    assert_eq!(measured.radii_axial[0], [0.0; 3]);
    assert!(measured.found[0]);
    assert!(measured.radii[1] > 0.0);
}

/// Offsets outside the volume are skipped, not background: a vertex in a
/// solid volume is measured to the one background voxel there is.
#[test]
fn a_vertex_whose_search_runs_into_the_volume_face() {
    let mut mask = Array3::from_elem((9, 9, 9), true);
    mask[[8, 8, 8]] = false;
    let source = Voxels::from(mask);
    let at = [[0u32, 0, 0]];
    let reaching = search(16)
        .measure(&source, &at, [1.0; 3])
        .expect("a measurement");
    assert!(reaching.found[0]);
    assert_eq!(reaching.radii_axial[0], [8.0; 3]);
    assert_eq!(reaching.radii[0], 192.0f64.sqrt());

    let short = search(4)
        .measure(&source, &at, [1.0; 3])
        .expect("a measurement");
    assert!(!short.found[0]);
    assert_eq!(short.not_found(), 1);
}

#[test]
fn the_not_found_answer_is_one_sentinel_in_every_array() {
    let source = Voxels::from(Array3::from_elem((9, 9, 9), true));
    let measured = search(4)
        .measure(&source, &[[4, 4, 4]], [1.0, 2.0, 4.0])
        .expect("a measurement");
    assert!(!measured.found[0]);
    assert_eq!(measured.radii[0], NOT_FOUND);
    assert_eq!(measured.radius_units[0], NOT_FOUND);
    assert_eq!(measured.radii_axial[0], [NOT_FOUND; 3]);
    assert_eq!(measured.radius_units_axial[0], [NOT_FOUND; 3]);
}

/// Reversing every tie group of the walk moves the axial answers and not the
/// scalar one, which is a function of the tie level.
#[test]
fn the_scalar_radius_does_not_depend_on_the_tie_order() {
    let source = ball([25, 25, 25], [12, 12, 12], 7.0);
    let points: Vec<[u32; 3]> = (5..20)
        .flat_map(|i| (5..20).map(move |j| [i, j, 12u32]))
        .collect();
    let stage = search(10);
    let measured = stage
        .measure(&source, &points, [1.0; 3])
        .expect("a measurement");
    let reversed = stage.sequence().with_ties_reversed();
    let mut axial_moved = 0;
    for (row, at) in points.iter().enumerate() {
        let at = at.map(|value| value as usize);
        let stopped = walk_from(&reversed, at, [25; 3], &source, [0; 3], Limit::AtMost(0.0))
            .expect("a walk")
            .expect("a stop");
        assert_eq!(reversed.distances()[stopped], measured.radii[row]);
        let offset = reversed.offsets()[stopped].map(|v| v.unsigned_abs() as f64);
        if offset != measured.radii_axial[row] {
            axial_moved += 1;
            assert!(!measured.determined[row]);
        }
    }
    assert!(axial_moved > 0, "the fixture cannot see the tie order");
}

#[test]
fn an_anisotropic_maximum_orders_by_euclidean_length() {
    let stage = RadiusSearch::new(RadiusParams {
        max_radius: [6, 3, 2],
        ..RadiusParams::default()
    })
    .expect("a search");
    let sequence = stage.sequence();
    assert_eq!(sequence.maximum(), [6, 3, 2]);
    assert!(sequence
        .distances()
        .windows(2)
        .all(|pair| pair[0] <= pair[1]));
    assert!(sequence
        .offsets()
        .iter()
        .all(|o| o[0].abs() <= 6 && o[1].abs() <= 3 && o[2].abs() <= 2));
}

#[test]
fn a_bad_spacing_is_refused() {
    let source = ball([9, 9, 9], [4, 4, 4], 2.0);
    assert!(search(3)
        .measure(&source, &[[4, 4, 4]], [1.0, 0.0, 1.0])
        .is_err());
}

fn fixture_root() -> PathBuf {
    PathBuf::from(
        std::env::var("VESSEL_GRAPH_FIXTURES")
            .unwrap_or_else(|_| "/husky/henriksson/clearmap-img/outputs".to_string()),
    )
}

fn read_bool(path: PathBuf) -> Voxels {
    let elements = npy::read_elements_file(&path, npy::OrderPolicy::Only(npy::Order::C))
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let array = elements
        .get::<bool>(&path.display().to_string())
        .expect("a bool volume")
        .clone()
        .into_dimensionality::<Ix3>()
        .expect("3-D");
    Voxels::from(array)
}

#[test]
#[ignore = "reads a recorded vessel crop under VESSEL_GRAPH_FIXTURES (default /husky/...)/vasc-binarize/{binary,skeleton}.npy"]
fn the_four_arrays_are_one_search_on_a_real_crop() {
    let root = fixture_root().join("vasc-binarize");
    let binary = read_bool(root.join("binary.npy"));
    let skeleton = read_bool(root.join("skeleton.npy"));
    let raw = build(&skeleton, &BuildParams::default()).expect("a graph");
    let measured = search(50)
        .measure(&binary, raw.coordinates(), [1.0; 3])
        .expect("a measurement");
    assert_eq!(measured.not_found(), 0);
    for row in 0..measured.len() {
        assert_eq!(measured.radii_axial[row], measured.radius_units_axial[row]);
        assert_eq!(measured.radii[row], norm(&measured.radii_axial[row]));
        assert!(measured.radii[row] > 0.0, "a skeleton vertex on background");
    }
}
