// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The clique collapse, on fixtures built so every branch of it is exercised.

use std::collections::{BTreeMap, BTreeSet};

use blockflow::ops::adjacency::Pair;
use blockflow::ops::coordinates::set_voxels;
use blockflow::voxels::Voxels;
use blockflow_vessel_graph::graph::clean as collapse;
use blockflow_vessel_graph::graph::clean::{CleanVertex, Clique};
use blockflow_vessel_graph::graph::{build, BuildParams, CleanGraph, CleanParams, Csr, RawGraph};
use ndarray::Array3;

/// 17 x 12 x 5, asymmetric on every axis: two rings in two planes (so the
/// cleaned graph keeps two cycles), six cliques of sizes 2, 2, 4, 4, 9 and 16
/// (five even, so the upper median is exercised), one clique with no outside
/// neighbour (dropped), clique-to-outside edges that deduplicate (26 to 18),
/// two isolated voxels, and a diagonal staircase.
fn scene() -> Array3<bool> {
    let mut mask = Array3::from_elem((17, 12, 5), false);
    for x in 2..9 {
        mask[[x, 2, 1]] = true;
        mask[[x, 7, 1]] = true;
    }
    for y in 2..8 {
        mask[[2, y, 1]] = true;
        mask[[8, y, 1]] = true;
    }
    for x in 11..13 {
        for y in 4..6 {
            for z in 1..3 {
                mask[[x, y, z]] = true;
            }
        }
    }
    for x in 9..11 {
        mask[[x, 4, 1]] = true;
    }
    for x in 2..7 {
        mask[[x, 2, 3]] = true;
        mask[[x, 5, 3]] = true;
    }
    for y in 2..6 {
        mask[[2, y, 3]] = true;
        mask[[6, y, 3]] = true;
    }
    for x in 13..16 {
        for y in 9..12 {
            mask[[x, y, 3]] = true;
        }
    }
    mask[[0, 0, 4]] = true;
    mask[[16, 0, 0]] = true;
    for i in 0..5 {
        mask[[i, i + 6, 4]] = true;
    }
    mask
}

fn raw_of(mask: Array3<bool>) -> RawGraph {
    build(&Voxels::from(mask), &BuildParams::default()).expect("a raw graph")
}

fn raw() -> RawGraph {
    raw_of(scene())
}

fn cleaned() -> CleanGraph {
    collapse(&raw(), &CleanParams::default()).expect("a cleaned graph")
}

/// The scene's raw graph with one self-loop planted on an isolated vertex.
fn raw_with_a_self_loop() -> RawGraph {
    let mask = scene();
    let coordinates = set_voxels(mask.view());
    let graph = raw();
    let loop_at = [0usize, 0, 4];
    assert_eq!(
        graph
            .adjacency()
            .degree(graph.id_of([0, 0, 4]).expect("a vertex")),
        0
    );
    let mut pairs: Vec<Pair> = Vec::new();
    let mut planted = false;
    for edge in 0..graph.n_edges() {
        let (lower, higher) = graph.edge_endpoints(edge).expect("an edge");
        let lower = lower.map(|v| v as usize);
        let higher = higher.map(|v| v as usize);
        if !planted && (lower, higher) > (loop_at, loop_at) {
            pairs.push((loop_at, loop_at));
            planted = true;
        }
        pairs.push((lower, higher));
    }
    if !planted {
        pairs.push((loop_at, loop_at));
    }
    RawGraph::assemble([17, 12, 5], &coordinates, &pairs).expect("a graph with a self-loop")
}

#[test]
fn the_collapse_counts_are_what_the_fixture_was_built_for() {
    let raw = raw();
    let clean = cleaned();
    let report = clean.report();
    assert_eq!((raw.n_vertices(), raw.n_edges()), (62, 104));
    assert_eq!((clean.n_vertices(), clean.n_edges()), (28, 27));
    assert_eq!(report.cliques_total, 6);
    assert_eq!(report.cliques_collapsed, 5);
    assert_eq!(report.vertices_in_cliques, 37);
    assert_eq!(report.intra_clique_edges, 69);
    assert_eq!(report.clique_outside_edges, 26);
    assert_eq!(report.clique_outside_neighbours, 18);
    assert_eq!(report.self_loops_removed, 0);
    assert_eq!(report.isolated_removed, 2);
    assert!(!report.untouched);
    let sizes: Vec<usize> = clean.cliques().iter().map(|c| c.members().len()).collect();
    assert_eq!(sizes, vec![2, 4, 2, 4, 16, 9]);
    assert_eq!(clean.cliques().iter().filter(|c| c.is_dropped()).count(), 1);
}

#[test]
fn a_cliques_neighbours_are_never_branch_points() {
    let raw = raw();
    let csr = raw.adjacency();
    for clique in cleaned().cliques() {
        for neighbour in clique.outside() {
            assert!(csr.degree(*neighbour) <= 2);
        }
    }
}

#[test]
fn survivors_ascend_then_centres_ascend_by_anchor() {
    let raw = raw();
    let clean = cleaned();
    let mut seen_a_centre = false;
    let mut previous_voxel: Option<[u32; 3]> = None;
    let mut previous_anchor: Option<u32> = None;
    for (index, origin) in clean.origins().iter().enumerate() {
        match origin {
            CleanVertex::Voxel(at) => {
                assert!(!seen_a_centre, "vertex {index} is a voxel after a centre");
                assert!(previous_voxel.is_none_or(|previous| previous < *at));
                previous_voxel = Some(*at);
                assert_eq!(clean.coordinates()[index], *at);
            }
            CleanVertex::Centre { anchor } => {
                seen_a_centre = true;
                assert!(previous_anchor.is_none_or(|previous| previous < *anchor));
                previous_anchor = Some(*anchor);
            }
        }
    }
    assert!(seen_a_centre);

    // Retained raw edges first, in raw order; then clique edges by clique
    // and ascending neighbour.
    let mut retained = Vec::new();
    let mut clique_edges = Vec::new();
    for (index, [a, b]) in clean.edges().iter().enumerate() {
        match (clean.origins()[*a as usize], clean.origins()[*b as usize]) {
            (CleanVertex::Voxel(lower), CleanVertex::Voxel(higher)) => {
                assert!(clique_edges.is_empty(), "edge {index} after a clique edge");
                retained.push([lower, higher]);
            }
            (CleanVertex::Voxel(neighbour), CleanVertex::Centre { anchor }) => {
                clique_edges.push((anchor, neighbour));
            }
            other => panic!("edge {index} has an unexpected shape: {other:?}"),
        }
    }
    let raw_order: Vec<[[u32; 3]; 2]> = raw
        .edges()
        .iter()
        .map(|[a, b]| {
            [
                raw.coordinates()[*a as usize],
                raw.coordinates()[*b as usize],
            ]
        })
        .collect();
    let mut cursor = 0;
    for edge in &retained {
        cursor += raw_order[cursor..]
            .iter()
            .position(|candidate| candidate == edge)
            .expect("a retained edge in raw order")
            + 1;
    }
    let mut sorted = clique_edges.clone();
    sorted.sort_unstable();
    assert_eq!(clique_edges, sorted);
}

#[test]
fn a_clique_edge_is_oriented_by_id_not_by_coordinate() {
    let clean = cleaned();
    let higher_first = clean
        .edge_coordinates()
        .into_iter()
        .filter(|row| [row[0], row[1], row[2]] > [row[3], row[4], row[5]])
        .count();
    assert!(higher_first > 0);
    assert!(clean.edges().iter().all(|[a, b]| a < b));
}

fn labels(csr: &Csr) -> Vec<usize> {
    let mut label = vec![usize::MAX; csr.n_vertices()];
    let mut stack = Vec::new();
    let mut next = 0;
    for root in 0..csr.n_vertices() {
        if label[root] != usize::MAX {
            continue;
        }
        label[root] = next;
        stack.push(root as u32);
        while let Some(vertex) = stack.pop() {
            for neighbour in csr.neighbours(vertex) {
                if label[*neighbour as usize] == usize::MAX {
                    label[*neighbour as usize] = next;
                    stack.push(*neighbour);
                }
            }
        }
        next += 1;
    }
    label
}

#[test]
fn the_collapse_destroys_clique_cycles_and_splits_no_component() {
    let raw = raw();
    let clean = cleaned();
    let raw_csr = raw.adjacency();
    let clean_csr = clean.adjacency();
    let before = raw_csr.betti_1(raw.n_edges());
    let after = clean_csr.betti_1(clean.n_edges());
    assert_eq!((raw_csr.components(), before), (6, 48));
    assert_eq!((clean_csr.components(), after), (3, 2));
    assert_eq!(before - after, clean.report().predicted_cycles_removed());

    let raw_label = labels(&raw_csr);
    let clean_label = labels(&clean_csr);
    let mut mapping = BTreeMap::<usize, BTreeSet<usize>>::new();
    for (index, origin) in clean.origins().iter().enumerate() {
        let raw_id = match origin {
            CleanVertex::Voxel(at) => raw.id_of(*at).expect("a raw vertex"),
            CleanVertex::Centre { anchor } => *anchor,
        };
        mapping
            .entry(raw_label[raw_id as usize])
            .or_default()
            .insert(clean_label[index]);
    }
    assert!(mapping.values().all(|components| components.len() == 1));
    assert_eq!(mapping.len(), clean_csr.components());
    assert_eq!(
        raw_csr.components() - mapping.len(),
        clean.report().isolated_removed + clean.cliques().iter().filter(|c| c.is_dropped()).count()
    );
}

#[test]
fn a_centre_can_land_on_a_surviving_voxel() {
    let mut mask = Array3::from_elem((10, 12, 2), false);
    for y in 0..9 {
        for x in [1, 2, 7, 8] {
            mask[[x, y, 0]] = true;
        }
    }
    for x in 1..9 {
        for y in 0..2 {
            mask[[x, y, 0]] = true;
        }
    }
    mask[[0, 9, 0]] = true;
    mask[[0, 10, 0]] = true;
    for y in 3..6 {
        mask[[5, y, 0]] = true;
    }
    let clean = collapse(&raw_of(mask), &CleanParams::default()).expect("a cleaned graph");
    assert_eq!(clean.report().cliques_total, 1);
    assert_eq!(clean.cliques()[0].members().len(), 44);
    assert_eq!(clean.cliques()[0].centre(), [5, 3, 0]);
    assert_eq!((clean.n_vertices(), clean.n_edges()), (6, 4));
    assert_eq!(clean.duplicate_coordinates(), 1);
    let colliding: Vec<usize> = (0..clean.n_vertices())
        .filter(|index| clean.coordinates()[*index] == [5, 3, 0])
        .collect();
    assert_eq!(colliding.len(), 2);
    assert_eq!(clean.origins()[colliding[0]], CleanVertex::Voxel([5, 3, 0]));
    assert!(matches!(
        clean.origins()[colliding[1]],
        CleanVertex::Centre { .. }
    ));
    for (index, origin) in clean.origins().iter().enumerate() {
        assert_eq!(clean.id_of(*origin), Some(index as u32));
    }
    assert_eq!(clean.adjacency().components(), 2);
}

#[test]
fn the_fixture_graphs_have_no_accidental_collision() {
    assert_eq!(cleaned().duplicate_coordinates(), 0);
}

#[test]
fn both_removals_change_the_answer() {
    let raw = raw_with_a_self_loop();
    let run = |params: CleanParams| {
        let clean = collapse(&raw, &params).expect("clean");
        (clean.n_vertices(), clean.n_edges())
    };
    assert_eq!(run(CleanParams::default()), (28, 27));
    assert_eq!(
        run(CleanParams {
            remove_self_loops: false,
            ..CleanParams::default()
        }),
        (29, 28)
    );
    assert_eq!(
        run(CleanParams {
            remove_isolated_vertices: false,
            ..CleanParams::default()
        }),
        (30, 27)
    );
}

#[test]
fn the_collapse_cannot_create_a_self_loop() {
    let keep_loops = CleanParams {
        remove_self_loops: false,
        ..CleanParams::default()
    };
    let raw = raw();
    assert_eq!(raw.self_loops(), 0);
    let without = collapse(&raw, &keep_loops).expect("clean");
    assert_eq!(without.self_loops(), 0);
    assert_eq!(without, cleaned());
    let planted = raw_with_a_self_loop();
    assert_eq!(planted.self_loops(), 1);
    assert_eq!(
        collapse(&planted, &keep_loops).expect("clean").self_loops(),
        1
    );
    assert_eq!(
        collapse(&planted, &CleanParams::default())
            .expect("clean")
            .report()
            .self_loops_removed,
        1
    );
}

#[test]
fn the_lower_median_would_move_a_centre() {
    let raw = raw();
    let clean = cleaned();
    let lower: Vec<[u32; 3]> = clean
        .cliques()
        .iter()
        .map(|clique| {
            [0, 1, 2].map(|axis| {
                let mut column: Vec<u32> = clique
                    .members()
                    .iter()
                    .map(|member| raw.coordinates()[*member as usize][axis])
                    .collect();
                column.sort_unstable();
                column[(column.len() - 1) / 2]
            })
        })
        .collect();
    let upper: Vec<[u32; 3]> = clean.cliques().iter().map(Clique::centre).collect();
    assert_ne!(upper, lower, "the fixture does not exercise the tie-break");
}

#[test]
fn a_graph_with_no_clique_is_returned_untouched() {
    let mut mask = Array3::from_elem((9, 3, 2), false);
    for x in 0..6 {
        mask[[x, 0, 0]] = true;
    }
    mask[[8, 2, 1]] = true;
    let raw = raw_of(mask);
    let clean = collapse(&raw, &CleanParams::default()).expect("a cleaned graph");
    assert!(clean.report().untouched);
    assert_eq!(clean.report().branch_points, 0);
    assert_eq!(clean.coordinates(), raw.coordinates());
    assert_eq!(clean.edges(), raw.edges());
    // The untouched path keeps the isolated vertex.
    assert_eq!(
        clean
            .adjacency()
            .degree(clean.id_of(CleanVertex::Voxel([8, 2, 1])).expect("it")),
        0
    );

    let mut junction = Array3::from_elem((8, 8, 8), false);
    for at in [
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
        [3, 3, 3],
        [2, 2, 2],
        [5, 3, 4],
        [6, 2, 4],
    ] {
        junction[at] = true;
    }
    let raw = raw_of(junction);
    assert_eq!(raw.adjacency().degree(raw.id_of([4, 4, 4]).expect("it")), 3);
    let clean = collapse(&raw, &CleanParams::default()).expect("a cleaned graph");
    assert_eq!(clean.report().branch_points, 1);
    assert!(
        clean.report().untouched,
        "a lone branch point is not a clique"
    );
}

/// The voxel touching a 2 x 2 x 2 blob diagonally has degree 3 and joins the
/// clique, so the clique is nine voxels and no axis of its median ties.
#[test]
fn a_solid_blob_and_the_voxel_that_touches_it_collapse_to_one_vertex() {
    let mut mask = Array3::from_elem((12, 6, 4), false);
    for x in 6..8 {
        for y in 1..3 {
            for z in 1..3 {
                mask[[x, y, z]] = true;
            }
        }
    }
    mask[[5, 0, 1]] = true;
    mask[[4, 0, 1]] = true;
    mask[[3, 0, 1]] = true;
    let raw = raw_of(mask);
    let clean = collapse(&raw, &CleanParams::default()).expect("a cleaned graph");
    assert_eq!((raw.n_vertices(), raw.n_edges()), (11, 32));
    let clique = &clean.cliques()[0];
    assert_eq!(clean.report().cliques_total, 1);
    assert_eq!(clique.members().len(), 9);
    assert_eq!(clique.centre(), [6, 1, 1]);
    assert_eq!(Some(clique.anchor()), raw.id_of([5, 0, 1]));
    assert_eq!(
        clique.outside(),
        &[raw.id_of([4, 0, 1]).expect("the way out")]
    );
    assert_eq!(clean.report().intra_clique_edges, 30);
    assert_eq!(clean.report().clique_outside_edges, 1);
    assert_eq!(clean.coordinates(), &[[3, 0, 1], [4, 0, 1], [6, 1, 1]]);
    assert_eq!(clean.adjacency().betti_1(clean.n_edges()), 0);
}

#[test]
fn the_collapse_shrinks_the_structural_graph() {
    let raw = raw();
    let clean = cleaned();
    assert!(clean.structural_bytes() < raw.structural_bytes());
}
