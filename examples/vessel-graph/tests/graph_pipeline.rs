// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Skeleton to annotated, classified vessel graph on a synthetic T of two
//! tubes: a thick "artery" along x and a thin vessel branching off along y.

use std::sync::Arc;

use atlas_regions::{AnnotateParams, Annotator, AtlasVolumes, Ontology};
use blockflow::voxels::Voxels;
use blockflow_vessel_graph::graph::columns::{onto_edges, through_clean};
use blockflow_vessel_graph::graph::{
    annotate, build, classify, clean, largest_component, reduce, vessel_graph, Aggregate,
    BuildParams, ClassifyParams, CleanParams, Column, EdgeFold, GraphParams, Inputs, ReduceParams,
    VesselGraph,
};
use ndarray::Array3;

const SHAPE: [usize; 3] = [40, 40, 20];
const THICK: f64 = 3.0;
const THIN: f64 = 1.5;

/// Tube A along x at (y 10, z 10), tube B along y at (x 20, z 10) from the
/// junction outward, and their centrelines.
fn scene() -> (Array3<bool>, Array3<bool>, Array3<bool>) {
    let mut vessels = Array3::from_elem((SHAPE[0], SHAPE[1], SHAPE[2]), false);
    let mut artery = vessels.clone();
    let mut skeleton = vessels.clone();
    for ((x, y, z), slot) in vessels.indexed_iter_mut() {
        let (x, y, z) = (x as f64, y as f64, z as f64);
        let in_a =
            (2.0..=37.0).contains(&x) && ((y - 10.0).powi(2) + (z - 10.0).powi(2)).sqrt() <= THICK;
        let in_b =
            (10.0..=37.0).contains(&y) && ((x - 20.0).powi(2) + (z - 10.0).powi(2)).sqrt() <= THIN;
        *slot = in_a || in_b;
        if in_a {
            artery[[x as usize, y as usize, z as usize]] = true;
        }
    }
    for x in 2..38 {
        skeleton[[x, 10, 10]] = true;
    }
    for y in 11..38 {
        skeleton[[20, y, 10]] = true;
    }
    (skeleton, vessels, artery)
}

fn built() -> VesselGraph {
    let (skeleton, vessels, artery) = scene();
    let skeleton = Voxels::from(skeleton);
    let vessels = Voxels::from(vessels);
    let artery_mask = Voxels::from(artery.clone());
    let intensity = Voxels::from(artery.mapv(|set| if set { 500.0f64 } else { 0.0 }));
    let inputs = Inputs {
        vessels: Some(&vessels),
        artery_intensity: Some(&intensity),
        artery_mask: Some(&artery_mask),
        ..Inputs::new(&skeleton, [2.0, 2.0, 2.0])
    };
    let params = GraphParams {
        radius: blockflow_vessel_graph::graph::RadiusParams {
            max_radius: [8, 8, 8],
            ..Default::default()
        },
        ..GraphParams::default()
    };
    vessel_graph(&inputs, &params).expect("a vessel graph")
}

#[test]
fn the_t_reduces_to_three_edges_with_every_column_at_every_level() {
    let graph = built();
    assert_eq!((graph.n_vertices(), graph.n_edges()), (4, 3));
    assert_eq!(graph.degrees().iter().filter(|d| **d == 3).count(), 1);
    for name in [
        "coordinates",
        "radii",
        "radii_axial",
        "radius_units",
        "radius_units_axial",
        "artery_raw",
        "artery_binary",
    ] {
        assert!(graph.vertex(name).is_some(), "vertex {name}");
        assert!(graph.along(name).is_some(), "along {name}");
    }
    for name in ["radii", "radius_units", "artery_raw", "artery_binary"] {
        assert!(graph.edge(name).is_some(), "edge {name}");
    }
    assert!(
        graph.edge("radii_axial").is_none(),
        "wide columns do not fold onto edges"
    );
    assert_eq!(
        graph.along("coordinates").expect("along").len(),
        graph.n_points()
    );
    let radius_units = graph.vertex("radius_units").expect("µm");
    let radii = graph.vertex("radii").expect("voxels");
    for row in 0..graph.n_vertices() {
        assert_eq!(radius_units.row(row)[0], 2.0 * radii.row(row)[0]);
    }
}

#[test]
fn the_thick_edges_are_arterial_and_the_thin_one_is_not() {
    let graph = built();
    let radii = graph.edge("radii").expect("radii").values().to_vec();
    let binary = graph
        .edge("artery_binary")
        .expect("binary")
        .as_flags()
        .expect("flags");
    // Tube B is the edge whose polyline reaches furthest along y.
    let along = graph.along_points("coordinates").expect("along");
    let reach = |edge: usize| {
        let [start, end] = graph.ranges()[edge];
        along[start..end].iter().map(|p| p[1]).fold(0.0, f64::max)
    };
    let thin = (0..graph.n_edges())
        .max_by(|a, b| reach(*a).total_cmp(&reach(*b)))
        .expect("an edge");
    for edge in 0..graph.n_edges() {
        assert_eq!(binary[edge], edge != thin, "edge {edge}");
        assert!(radii[edge] >= 2.0, "edge {edge}: {}", radii[edge]);
    }
    let raw = graph.edge("artery_raw").expect("raw").values();
    assert!(
        raw.iter().all(|value| *value == 500.0),
        "the junction ball reaches tube A"
    );
}

#[test]
fn classification_traces_from_the_masked_edges() {
    let params = ClassifyParams {
        radius: "radii".to_string(),
        restrictive_vein_radius: 100.0,
        permissive_vein_radius: 100.0,
        final_vein_radius: 100.0,
        arteries_min_noise_edges: 1,
        artery_intensity_min: 0.0,
        artery_trace_radius: 0.0,
        max_artery_trace_iter: 0,
        min_artery_component_edges: 1,
        min_vein_component_edges: 1,
        ..ClassifyParams::default()
    };
    let mut graph = built();
    let trace = classify(&mut graph, &params).expect("classified");
    assert_eq!((trace.arteries(), trace.veins()), (2, 0));
    assert_eq!(
        graph
            .edge("artery")
            .expect("artery")
            .as_flags()
            .expect("flags"),
        graph
            .edge("artery_binary")
            .expect("binary")
            .as_flags()
            .expect("flags")
    );

    let mut graph = built();
    let traced = ClassifyParams {
        max_artery_trace_iter: 1,
        ..params.clone()
    };
    assert_eq!(
        classify(&mut graph, &traced)
            .expect("classified")
            .arteries(),
        3
    );

    let mut graph = built();
    let too_few = ClassifyParams {
        min_artery_component_edges: 3,
        ..params
    };
    assert_eq!(
        classify(&mut graph, &too_few)
            .expect("classified")
            .arteries(),
        0
    );
}

#[test]
fn annotation_reads_the_atlas_at_every_level_and_the_edge_takes_the_polyline_minimum() {
    let mut graph = built();
    graph.rescale_to_atlas(SHAPE).expect("identity resample");
    let mut regions = Array3::<u16>::from_elem((SHAPE[0], SHAPE[1], SHAPE[2]), 1);
    for ((x, _, _), value) in regions.indexed_iter_mut() {
        if x >= 20 {
            *value = 2;
        }
    }
    let distance = Voxels::from(Array3::from_shape_fn(
        (SHAPE[0], SHAPE[1], SHAPE[2]),
        |(x, y, _)| (x + y) as f64,
    ));
    let annotation = Voxels::from(regions);
    let annotator = Annotator::new(AnnotateParams::default()).with_ontology(Arc::new(
        Ontology::new([(1, Some(1), None), (2, Some(1), None)]).expect("a tree"),
    ));
    let volumes = AtlasVolumes::new(&annotation).with_distance_to_surface(&distance);
    annotate(&mut graph, &annotator, volumes).expect("annotated");

    let atlas = graph.points("coordinates_atlas").expect("atlas");
    let labels = graph.vertex("annotation").expect("labels").values();
    for (row, at) in atlas.iter().enumerate() {
        assert_eq!(labels[row], if at[0] >= 20.0 { 2.0 } else { 1.0 });
    }
    let along = graph.along("distance_to_surface").expect("along").values();
    let edge = graph.edge("distance_to_surface").expect("edge").values();
    for (index, [start, end]) in graph.ranges().into_iter().enumerate() {
        let minimum = along[start..end]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        assert_eq!(edge[index], minimum);
    }

    graph.scale_radii([2.0; 3], [20, 20, 10]).expect("scaled");
    let scaled = graph.vertex("radii_atlas").expect("twin").values();
    let radii = graph.vertex("radii").expect("radii").values();
    for row in 0..graph.n_vertices() {
        assert_eq!(scaled[row], radii[row] * 0.5);
    }
    assert_eq!(
        graph.vertex("radius_units_atlas").expect("twin").values(),
        graph.vertex("radii_atlas").expect("twin").values(),
        "µm over 4 µm atlas voxels equals voxels at half the resolution"
    );
    assert!(graph.edge("radii_atlas").is_some());
    assert!(graph.along("radii_axial_atlas").is_some());
}

#[test]
fn the_json_round_trip_is_exact() {
    let mut graph = built();
    graph.rescale_to_atlas([20, 20, 10]).expect("rescaled");
    let back = VesselGraph::from_json(&graph.to_json()).expect("parsed");
    assert_eq!(back, graph);
    let text = serde_json::to_string(&graph.to_json()).expect("text");
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("json");
    assert_eq!(VesselGraph::from_json(&parsed).expect("parsed"), graph);
}

#[test]
fn a_clique_centre_folds_each_column_on_its_own() {
    // A 2 x 2 blob with a tail: the blob is one clique.
    let mut mask = Array3::from_elem((8, 4, 3), false);
    for x in 3..5 {
        for y in 1..3 {
            mask[[x, y, 1]] = true;
        }
    }
    for x in 0..3 {
        mask[[x, 0, 1]] = true;
    }
    let raw = build(&Voxels::from(mask), &BuildParams::default()).expect("raw");
    let cleaned = clean(&raw, &CleanParams::default()).expect("cleaned");
    assert_eq!(cleaned.report().cliques_total, 1);
    let rows: Vec<[f64; 3]> = (0..raw.n_vertices())
        .map(|v| [v as f64, -(v as f64), 1.0])
        .collect();
    let folded =
        through_clean(&raw, &cleaned, &Column::rows3(&rows), Aggregate::Max).expect("folded");
    let members = cleaned.cliques()[0].members();
    let centre = folded.row(cleaned.n_vertices() - 1);
    assert_eq!(centre[0], *members.last().unwrap() as f64);
    assert_eq!(centre[1], -(members[0] as f64));
    assert_eq!(centre[2], 1.0);

    let reduced = reduce(&cleaned, &ReduceParams::default()).expect("reduced");
    let flags = Column::flags(
        &(0..cleaned.n_vertices())
            .map(|v| v % 3 != 0)
            .collect::<Vec<_>>(),
    );
    assert!(onto_edges(&reduced, &flags, EdgeFold::BINARY).is_ok());
    let not_flags = Column::scalar(vec![0.5; cleaned.n_vertices()]);
    assert!(onto_edges(&reduced, &not_flags, EdgeFold::BINARY).is_err());
}

#[test]
fn the_largest_component_wins_and_ties_go_to_the_smallest_vertex() {
    let edges = [[0u32, 1], [2, 3], [3, 4], [5, 6], [6, 7]];
    let component = largest_component(8, &edges).expect("a component");
    assert_eq!(component.vertices(), &[2, 3, 4]);
    assert_eq!(component.edges(), &[1, 2]);
    assert_eq!(component.connectivity(), &[[0, 1], [1, 2]]);
    assert_eq!(component.components(), 3);
    assert!(largest_component(0, &[]).is_err());
}

/// The T's two straight tubes: tube A runs 2..=37 along the first axis and is
/// cut at the junction (20), tube B runs from the junction to 37 along the
/// second. Along a straight line the polyline length is the end-to-end
/// distance, and at 2 µm voxels the physical length is twice it.
#[test]
fn an_edge_is_as_long_as_its_polyline() {
    let graph = built();
    let lengths = graph.edge("length").expect("lengths").values();
    let physical = graph.edge("length_units").expect("µm").values();
    let along = graph.along_points("coordinates").expect("polylines");
    let mut total = 0.0;
    for (edge, [start, end]) in graph.ranges().into_iter().enumerate() {
        let (first, last) = (along[start], along[end - 1]);
        let straight = (0..3)
            .map(|axis| (last[axis] - first[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((lengths[edge] - straight).abs() < 1e-9, "edge {edge}");
        assert!((physical[edge] - 2.0 * lengths[edge]).abs() < 1e-9);
        total += lengths[edge];
    }
    // 35 voxels of tube A and 18 of tube B from the junction (y 10 to 37,
    // less the clique and the stub to it).
    assert!((50.0..=63.0).contains(&total), "total {total}");

    // An anisotropic spacing weighs each axis by its own pitch.
    let (voxels, scaled) = graph.edge_lengths([1.0, 3.0, 1.0]).expect("lengths");
    assert_eq!(voxels, lengths);
    for (edge, [start, end]) in graph.ranges().into_iter().enumerate() {
        let (first, last) = (along[start], along[end - 1]);
        let expected = ((last[0] - first[0]).powi(2)
            + (3.0 * (last[1] - first[1])).powi(2)
            + (last[2] - first[2]).powi(2))
        .sqrt();
        assert!((scaled[edge] - expected).abs() < 1e-9);
    }
}

/// A registration plugs in as a closure over a batch of points.
#[test]
fn a_closure_maps_the_coordinates() {
    let mut graph = built();
    let shift = |points: &[[f64; 3]]| -> blockflow::error::Result<Vec<[f64; 3]>> {
        Ok(points.iter().map(|p| [p[0] + 1.0, p[1], -p[2]]).collect())
    };
    graph
        .transform_coordinates("coordinates_atlas", &shift)
        .expect("mapped");
    let before = graph.along_points("coordinates").unwrap();
    let after = graph.along_points("coordinates_atlas").unwrap();
    for (p, q) in before.iter().zip(&after) {
        assert_eq!(*q, [p[0] + 1.0, p[1], -p[2]]);
    }
    let short = |points: &[[f64; 3]]| -> blockflow::error::Result<Vec<[f64; 3]>> {
        Ok(points[1..].to_vec())
    };
    assert!(graph.transform_coordinates("bad", &short).is_err());
}
