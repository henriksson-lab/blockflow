// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The chain contraction, on a hand-built graph with one of every chain kind.

use blockflow::voxels::Voxels;
use blockflow_vessel_graph::graph::{
    build, clean, reduce, BuildParams, CleanGraph, CleanParams, RawGraph, ReduceParams,
    ReducedGraph,
};
use ndarray::Array3;

/// Two chains X (0-1-2-3-4) and Y (0-5-6-4) between the same pair of branch
/// points, an ear 4-8-9-4 (a self-loop), a leaf 0-7, a pure degree-2 loop
/// 10-11-12, a lone pair 13-14 and an isolated vertex 15.
const N_VERTICES: usize = 16;
const EDGES: &[(usize, usize)] = &[
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 4),
    (0, 7),
    (4, 8),
    (8, 9),
    (9, 4),
    (10, 11),
    (11, 12),
    (12, 10),
    (13, 14),
];

/// A graph whose vertex `i` sits at `[i, 0, 0]`, so ids survive the clean.
fn graph_over(vertices: &[usize], edges: &[(usize, usize)]) -> RawGraph {
    let volume = [vertices.iter().copied().max().unwrap_or(0) + 1, 1, 1];
    let coordinates: Vec<[usize; 3]> = vertices.iter().map(|index| [*index, 0, 0]).collect();
    let pairs: Vec<([usize; 3], [usize; 3])> = edges
        .iter()
        .map(|(a, b)| ([*a, 0, 0], [*b, 0, 0]))
        .collect();
    RawGraph::assemble(volume, &coordinates, &pairs).expect("a raw graph")
}

fn graph_from(n_vertices: usize, edges: &[(usize, usize)]) -> RawGraph {
    graph_over(&(0..n_vertices).collect::<Vec<_>>(), edges)
}

fn cleaned_over(vertices: &[usize], edges: &[(usize, usize)]) -> CleanGraph {
    let clean = clean(&graph_over(vertices, edges), &CleanParams::default()).expect("cleaned");
    assert!(clean.report().untouched, "the fixture must have no clique");
    clean
}

fn cleaned() -> CleanGraph {
    cleaned_over(&(0..N_VERTICES).collect::<Vec<_>>(), EDGES)
}

fn reduced_from(clean: &CleanGraph) -> ReducedGraph {
    reduce(clean, &ReduceParams::default()).expect("a reduced graph")
}

fn reduced() -> (CleanGraph, ReducedGraph) {
    let clean = cleaned();
    let reduced = reduced_from(&clean);
    (clean, reduced)
}

#[test]
fn the_fixture_reduces_to_exactly_this() {
    let (_, reduced) = reduced();
    assert_eq!(reduced.sources(), &[0, 4, 7, 13, 14, 15]);
    assert_eq!(
        reduced.coordinates(),
        &[
            [0, 0, 0],
            [4, 0, 0],
            [7, 0, 0],
            [13, 0, 0],
            [14, 0, 0],
            [15, 0, 0]
        ]
    );
    assert_eq!(
        reduced.edges(),
        &[[0, 1], [0, 1], [1, 1], [0, 2], [3, 4]],
        "X, Y, the ear, the leaf, the pair"
    );
    assert_eq!(reduced.self_loops(), 1);
    assert_eq!(reduced.duplicate_edges(), 1);
    let geometry = reduced.geometry().expect("the geometry");
    assert_eq!(geometry.lengths(), vec![5, 4, 4, 2, 2]);
    assert_eq!(geometry.n_points(), 17);
    assert_eq!(geometry.path(0), &[0, 1, 2, 3, 4]);
    assert_eq!(geometry.path(1), &[0, 5, 6, 4]);
    assert_eq!(geometry.path(2), &[4, 8, 9, 4]);
    assert_eq!(geometry.path(3), &[0, 7]);
    assert_eq!(geometry.path(4), &[13, 14]);
    let report = reduced.report();
    assert_eq!(report.source_vertices, 16);
    assert_eq!(report.source_edges, 15);
    assert_eq!(report.degree_2_vertices, 10);
    assert_eq!(report.isolated_vertices, 1);
    assert_eq!(report.traced_chains, 3);
    assert_eq!(report.direct_chains, 2);
    assert_eq!(report.loop_chains, 1);
    assert_eq!(report.dropped_chains, 1);
    assert_eq!(report.vertices_dropped_with_chains, 3);
    assert_eq!(report.chains(), 6);
}

#[test]
fn the_vertices_are_the_non_degree_2_source_vertices_ascending() {
    let (clean, reduced) = reduced();
    let csr = clean.adjacency();
    let expected: Vec<u32> = (0..clean.n_vertices() as u32)
        .filter(|vertex| csr.degree(*vertex) != 2)
        .collect();
    assert_eq!(reduced.sources(), expected.as_slice());
    for (index, source) in reduced.sources().iter().enumerate() {
        assert_eq!(
            reduced.coordinates()[index],
            clean.coordinates()[*source as usize]
        );
    }
    // Numbering by chain discovery would differ and would lose the isolated
    // vertex.
    let geometry = reduced.geometry().expect("the geometry");
    let mut by_discovery = Vec::new();
    for edge in 0..geometry.n_edges() {
        let path = geometry.path(edge);
        for end in [path[0], path[path.len() - 1]] {
            if !by_discovery.contains(&end) {
                by_discovery.push(end);
            }
        }
    }
    assert_ne!(by_discovery, reduced.sources().to_vec());
    assert!(!by_discovery.contains(&15));
}

#[test]
fn each_path_runs_from_the_edges_source_to_its_target() {
    let (clean, reduced) = reduced();
    let csr = clean.adjacency();
    let geometry = reduced.geometry().expect("the geometry");
    assert_eq!(geometry.n_edges(), reduced.n_edges());
    for (edge, [source, target]) in reduced.edges().iter().enumerate() {
        let path = geometry.path(edge);
        assert!(path.len() >= 2);
        assert_eq!(reduced.sources()[*source as usize], path[0]);
        assert_eq!(reduced.sources()[*target as usize], path[path.len() - 1]);
        for interior in &path[1..path.len() - 1] {
            assert_eq!(
                csr.degree(*interior),
                2,
                "edge {edge} runs through a branch"
            );
        }
        for step in path.windows(2) {
            assert!(csr.neighbours(step[0]).contains(&step[1]));
        }
    }
}

#[test]
fn a_traced_path_starts_at_its_lowest_end_edge() {
    let (clean, reduced) = reduced();
    let geometry = reduced.geometry().expect("the geometry");
    let mut checked = 0;
    for edge in 0..geometry.n_edges() {
        let path = geometry.path(edge);
        if path.len() < 3 {
            continue;
        }
        let mut seeds = Vec::new();
        for (branch, interior, is_start) in [
            (path[0], path[1], true),
            (path[path.len() - 1], path[path.len() - 2], false),
        ] {
            let found = clean
                .edges()
                .iter()
                .position(|[a, b]| {
                    (*a == branch && *b == interior) || (*a == interior && *b == branch)
                })
                .expect("the end edge");
            seeds.push((found, is_start));
        }
        seeds.sort_unstable();
        assert!(
            seeds[0].1,
            "edge {edge} starts at the higher of its end edges"
        );
        checked += 1;
    }
    assert_eq!(checked, 3);
}

#[test]
fn a_pure_degree_2_loop_vanishes_whole() {
    let (clean, reduced) = reduced();
    let geometry = reduced.geometry().expect("the geometry");
    for vertex in [10u32, 11, 12] {
        assert!(!reduced.sources().contains(&vertex));
        assert!(!geometry.points().contains(&vertex));
    }
    let kept: Vec<usize> = (0..N_VERTICES)
        .filter(|v| ![10, 11, 12].contains(v))
        .collect();
    let edges: Vec<(usize, usize)> = EDGES
        .iter()
        .copied()
        .filter(|(a, b)| ![10, 11, 12].contains(a) && ![10, 11, 12].contains(b))
        .collect();
    let without = cleaned_over(&kept, &edges);
    let reduced_without = reduced_from(&without);
    assert_eq!(reduced_without.coordinates(), reduced.coordinates());
    assert_eq!(
        reduced_without.edge_coordinates(),
        reduced.edge_coordinates()
    );
    for edge in 0..reduced.n_edges() {
        assert_eq!(
            reduced_without
                .path_coordinates(&without, edge)
                .expect("coordinates"),
            reduced.path_coordinates(&clean, edge).expect("coordinates")
        );
    }
    assert_eq!(reduced_without.report().loop_chains, 0);
    assert_eq!(reduced_without.report().source_vertices, 13);
}

#[test]
fn two_adjacent_branch_points_reach_this_stage_as_one_centre() {
    let raw = graph_from(6, &[(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)]);
    let clean = clean(&raw, &CleanParams::default()).expect("cleaned");
    assert_eq!(clean.cliques()[0].members(), &[0, 3]);
    let reduced = reduced_from(&clean);
    let geometry = reduced.geometry().expect("the geometry");
    assert_eq!((reduced.n_vertices(), reduced.n_edges()), (5, 4));
    assert_eq!(reduced.report().direct_chains, 4);
    assert_eq!(reduced.report().traced_chains, 0);
    assert!((0..geometry.n_edges()).all(|edge| geometry.path(edge).len() == 2));
}

#[test]
fn a_chain_that_returns_to_its_branch_point_is_a_self_loop() {
    let (_, reduced) = reduced();
    let geometry = reduced.geometry().expect("the geometry");
    let loops: Vec<usize> = (0..reduced.n_edges())
        .filter(|edge| reduced.edges()[*edge][0] == reduced.edges()[*edge][1])
        .collect();
    assert_eq!(loops, vec![2]);
    let path = geometry.path(2);
    assert!(path.len() >= 3);
    assert_eq!(path[0], path[path.len() - 1]);
}

#[test]
fn without_geometry_only_the_polylines_go() {
    let clean = cleaned();
    let with = reduced_from(&clean);
    let without = reduce(
        &clean,
        &ReduceParams {
            edge_geometry: false,
        },
    )
    .expect("reduced");
    assert!(without.geometry().is_none());
    assert!(without.path_coordinates(&clean, 0).is_err());
    assert_eq!(with.coordinates(), without.coordinates());
    assert_eq!(with.edges(), without.edges());
    assert_eq!(with.report(), without.report());
}

#[test]
fn the_geometry_is_one_flat_buffer_cut_by_offsets() {
    let (clean, reduced) = reduced();
    let geometry = reduced.geometry().expect("the geometry");
    let lengths = geometry.lengths();
    assert_eq!(
        lengths.iter().map(|l| *l as usize).sum::<usize>(),
        geometry.n_points()
    );
    let mut at = 0usize;
    for (edge, length) in lengths.iter().enumerate() {
        let length = *length as usize;
        assert_eq!(&geometry.points()[at..at + length], geometry.path(edge));
        assert_eq!(
            reduced.path_coordinates(&clean, edge).expect("coordinates"),
            geometry
                .path(edge)
                .iter()
                .map(|point| clean.coordinates()[*point as usize])
                .collect::<Vec<_>>()
        );
        at += length;
    }
    assert_eq!(reduced.geometry_coordinates(&clean).expect("all").len(), at);
}

#[test]
fn a_gather_against_the_wrong_source_graph_is_refused() {
    let (clean, reduced) = reduced();
    let other = cleaned_over(&[0, 1, 2, 3], &[(0, 1), (1, 2), (2, 3)]);
    assert!(reduced.path_coordinates(&clean, 0).is_ok());
    assert!(reduced.path_coordinates(&other, 0).is_err());
    assert!(reduced.path_coordinates(&clean, 99).is_err());
}

#[test]
fn every_source_vertex_is_accounted_for_once() {
    let (clean, reduced) = reduced();
    let geometry = reduced.geometry().expect("the geometry");
    let mut seen = vec![0usize; clean.n_vertices()];
    for source in reduced.sources() {
        seen[*source as usize] += 1;
    }
    for edge in 0..geometry.n_edges() {
        let path = geometry.path(edge);
        for interior in &path[1..path.len() - 1] {
            seen[*interior as usize] += 1;
        }
    }
    assert!(seen.iter().all(|count| *count <= 1));
    assert_eq!(
        seen.iter().filter(|count| **count == 0).count(),
        reduced.report().vertices_dropped_with_chains
    );
}

#[test]
fn a_chain_with_no_direction_is_refused() {
    let raw = graph_from(2, &[(0, 1), (0, 1)]);
    let clean = clean(&raw, &CleanParams::default()).expect("cleaned");
    assert_eq!(clean.n_edges(), 2);
    assert!(reduce(&clean, &ReduceParams::default()).is_err());
}

#[test]
fn a_real_skeleton_line_reduces_to_one_edge() {
    let mut mask = Array3::from_elem((3, 3, 13), false);
    for step in 0..11 {
        mask[[1, 1, step + 1]] = true;
    }
    let raw = build(&Voxels::from(mask), &BuildParams::default()).expect("a raw graph");
    assert_eq!((raw.n_vertices(), raw.n_edges()), (11, 10));
    let clean = clean(&raw, &CleanParams::default()).expect("cleaned");
    let reduced = reduced_from(&clean);
    assert_eq!((reduced.n_vertices(), reduced.n_edges()), (2, 1));
    assert_eq!(reduced.geometry().expect("geometry").lengths(), vec![11]);
    assert_eq!(
        reduced.path_coordinates(&clean, 0).expect("coordinates"),
        (0..11)
            .map(|step| [1, 1, step + 1])
            .collect::<Vec<[u32; 3]>>()
    );
}
