// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Branch density: which vertices reach `ops::voxelize`, with what weight.

use blockflow::points::Point;
use blockflow::voxels::Voxels;
use blockflow_vessel_graph::graph::branch::{branch_points, density};
use blockflow_vessel_graph::graph::Selection;
use blockflow_vessel_graph::graph::{branch_density, BranchParams, Column};
use blockflow_vessel_graph::graph::{
    build, clean, reduce, BuildParams, CleanParams, ReduceParams, VesselGraph,
};
use ndarray::Array3;

/// Six vertices on a line, degrees 3, 1, 3, 1, 3, 1, radii doubling.
fn fixture() -> (Vec<[f64; 3]>, Vec<usize>, Vec<f64>) {
    let coordinates: Vec<[f64; 3]> = (0..6).map(|i| [i as f64, 1.0, 1.0]).collect();
    let degrees = vec![3, 1, 3, 1, 3, 1];
    let radii = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
    (coordinates, degrees, radii)
}

#[test]
fn every_vertex_is_a_branch_by_default_and_a_degree_range_selects() {
    let (coordinates, degrees, _) = fixture();
    let every =
        branch_points(&coordinates, &degrees, None, Selection::Every, [8, 8, 8]).expect("runs");
    assert_eq!(every.points.len(), 6, "degree-1 endpoints included");
    assert!(every.points.iter().all(|point| point.weight == 1.0));
    let three = branch_points(
        &coordinates,
        &degrees,
        None,
        Selection::degree(3),
        [8, 8, 8],
    )
    .expect("runs");
    assert_eq!(
        three.points.iter().map(|p| p.at).collect::<Vec<_>>(),
        vec![[0, 1, 1], [2, 1, 1], [4, 1, 1]]
    );
    assert!(Selection::degree(3).admits(3));
    assert!(
        !Selection::degree(3).admits(5),
        "the range is inclusive and exact"
    );
}

#[test]
fn a_weight_is_read_from_the_selected_vertex() {
    let (coordinates, degrees, radii) = fixture();
    let placed = branch_points(
        &coordinates,
        &degrees,
        Some(&radii),
        Selection::degree(3),
        [8, 8, 8],
    )
    .expect("runs");
    assert_eq!(
        placed.points.iter().map(|p| p.weight).collect::<Vec<_>>(),
        vec![1.0, 4.0, 16.0]
    );
}

#[test]
fn coordinates_round_and_outside_points_are_dropped_and_counted() {
    let coordinates = vec![
        [1.4, 1.6, 0.5],
        [-0.6, 1.0, 1.0],
        [7.6, 1.0, 1.0],
        [7.4, 1.0, 1.0],
    ];
    let placed =
        branch_points(&coordinates, &[1; 4], None, Selection::Every, [8, 8, 8]).expect("runs");
    assert_eq!(placed.outside, 2);
    assert_eq!(
        placed.points.iter().map(|p| p.at).collect::<Vec<_>>(),
        vec![[1, 2, 1], [7, 1, 1]]
    );
    assert!(branch_points(
        &[[f64::NAN, 0.0, 0.0]],
        &[1],
        None,
        Selection::Every,
        [8, 8, 8]
    )
    .is_err());
}

#[test]
fn the_density_is_one_ball_of_weight_per_point() {
    let points = [Point::weighted([8, 8, 8], 2.0), Point::unit([3, 3, 3])];
    let rendered = density(&points, [17, 17, 17], [2, 2, 2]).expect("a density");
    let values = rendered.view::<f64>().expect("f64");
    let ball = blockflow::ops::voxelize::ball([2, 2, 2]).len() as f64;
    assert_eq!(values.sum(), 3.0 * ball);
    assert_eq!(values[[8, 8, 8]], 2.0);
    assert_eq!(values[[3, 3, 3]], 1.0);
    assert_eq!(values[[0, 0, 0]], 0.0);
}

#[test]
fn the_branch_density_of_a_graph() {
    let mut mask = Array3::from_elem((24, 24, 5), false);
    for x in 2..22 {
        mask[[x, 12, 2]] = true;
    }
    for y in 13..22 {
        mask[[12, y, 2]] = true;
    }
    let raw = build(&Voxels::from(mask), &BuildParams::default()).expect("raw");
    let cleaned = clean(&raw, &CleanParams::default()).expect("cleaned");
    let reduced = reduce(&cleaned, &ReduceParams::default()).expect("reduced");
    let mut graph = VesselGraph::assemble(&raw, &cleaned, &reduced, &[]).expect("a graph");
    graph.rescale_to_atlas([12, 12, 5]).expect("rescaled");
    assert_eq!(graph.n_vertices(), 4, "three ends and a junction");

    let params = BranchParams {
        radius: [1, 1, 1],
        ..BranchParams::default()
    };
    let (every, outside) = branch_density(&graph, [12, 12, 5], &params).expect("a density");
    assert_eq!(outside, 0);
    let ball = blockflow::ops::voxelize::ball([1, 1, 1]).len() as f64;
    let total = every.view::<f64>().expect("f64").sum();
    assert!(total <= 4.0 * ball && total > 3.0 * ball, "{total}");

    let junction = BranchParams {
        selection: Selection::degree(3),
        radius: [0, 0, 0],
        ..BranchParams::default()
    };
    let (one, _) = branch_density(&graph, [12, 12, 5], &junction).expect("a density");
    assert_eq!(one.view::<f64>().expect("f64").sum(), 1.0);

    graph
        .set_vertex("radii", Column::scalar(vec![2.0; 4]))
        .expect("a column");
    let weighted = BranchParams {
        weight: Some("radii".to_string()),
        ..junction
    };
    let (two, _) = branch_density(&graph, [12, 12, 5], &weighted).expect("a density");
    assert_eq!(two.view::<f64>().expect("f64").sum(), 2.0);
}
