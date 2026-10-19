// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The raw graph: vertex ids are the lexicographic rank of the coordinate,
//! edges are lexicographic on the coordinate pair, and the planned build gives
//! the same graph on every block lattice.

use std::collections::BTreeSet;
use std::path::PathBuf;

use blockflow::geometry::BlockGrid;
use blockflow::npy;
use blockflow::ops::adjacency::adjacent_pairs;
use blockflow::ops::coordinates::{blocks_concatenate_in_order, set_voxels};
use blockflow::ops::Connectivity;
use blockflow::voxels::Voxels;
use blockflow_vessel_graph::graph::{build, build_blocked, BuildParams, RawGraph};
use ndarray::{Array3, Ix3};

const VOLUME: [usize; 3] = [26, 18, 10];

/// A solid block (dense cliques), two rings in different planes (cycles), a
/// diagonal staircase (26-adjacency is load-bearing) and a line along the
/// fastest axis, packed unevenly so blocks of every cut differ in load.
fn scene() -> Array3<bool> {
    let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    for i in 1..7 {
        for j in 1..5 {
            for k in 1..4 {
                mask[[i, j, k]] = true;
            }
        }
    }
    for i in 9..15 {
        mask[[i, 2, 6]] = true;
        mask[[i, 7, 6]] = true;
    }
    for j in 2..8 {
        mask[[9, j, 6]] = true;
        mask[[14, j, 6]] = true;
    }
    for j in 7..12 {
        mask[[14, j, 6]] = true;
        mask[[14, j, 9]] = true;
    }
    for k in 6..10 {
        mask[[14, 7, k]] = true;
        mask[[14, 11, k]] = true;
    }
    for i in 0..10 {
        mask[[i, i, i % 10]] = true;
    }
    for k in 0..VOLUME[2] {
        mask[[25, 1, k]] = true;
    }
    mask
}

fn cuts() -> Vec<([usize; 3], &'static str)> {
    vec![
        (VOLUME, "one block"),
        ([4, 4, 4], "cubes of 4"),
        ([8, 8, 8], "cubes of 8"),
        ([5, 7, 3], "ragged, coprime with the volume"),
        ([1, 18, 10], "slabs of one on the slowest axis"),
        ([7, 18, 10], "slabs of seven on the slowest axis"),
        ([26, 18, 1], "planes on the fastest axis"),
        ([26, 4, 10], "slabs on the middle axis"),
    ]
}

fn serial() -> RawGraph {
    build(&Voxels::from(scene()), &BuildParams::default()).expect("a graph")
}

#[test]
fn the_fixture_has_cycles_and_components() {
    let graph = serial();
    let csr = graph.adjacency();
    assert!(csr.betti_1(graph.n_edges()) >= 2);
    assert!(csr.components() > 1);
}

#[test]
fn a_vertex_id_is_the_lexicographic_rank_of_its_coordinate() {
    let graph = serial();
    let mut sorted = graph.coordinates().to_vec();
    sorted.sort_unstable();
    assert_eq!(graph.coordinates(), sorted.as_slice());
    for (index, at) in graph.coordinates().iter().enumerate() {
        assert_eq!(graph.id_of(*at), Some(index as u32));
    }
    let mask = scene();
    for ((i, j, k), &set) in mask.indexed_iter() {
        let at = [i as u32, j as u32, k as u32];
        assert_eq!(graph.id_of(at).is_some(), set, "{at:?}");
    }
}

#[test]
fn the_edge_list_is_lexicographic_on_the_coordinate_pair() {
    let graph = serial();
    let pairs = graph.edge_coordinates();
    let mut sorted = pairs.clone();
    sorted.sort_unstable();
    assert_eq!(pairs, sorted);
    assert!(graph.edges().iter().all(|[a, b]| a < b));
    assert_eq!(graph.duplicate_edges(), 0);
    assert_eq!(graph.self_loops(), 0);
}

#[test]
fn the_fixture_discriminates_every_cut() {
    let graph = serial();
    let mut cuts_with_empty_blocks = 0usize;
    let mut worst_imbalance = 0.0f64;
    for (block, name) in cuts() {
        if block == VOLUME {
            continue;
        }
        let per_axis = [0, 1, 2].map(|axis| VOLUME[axis].div_ceil(block[axis]));
        let index = |at: [u32; 3]| {
            (at[0] as usize / block[0]) * per_axis[1] * per_axis[2]
                + (at[1] as usize / block[1]) * per_axis[2]
                + (at[2] as usize / block[2])
        };
        let crossing = (0..graph.n_edges())
            .filter(|edge| {
                let (a, b) = graph.edge_endpoints(*edge).expect("an edge");
                index(a) != index(b)
            })
            .count();
        let mut per_block = vec![0usize; per_axis.iter().product()];
        for at in graph.coordinates() {
            per_block[index(*at)] += 1;
        }
        let empty = per_block.iter().filter(|count| **count == 0).count();
        let heaviest = *per_block.iter().max().expect("a block");
        let mean = graph.n_vertices() as f64 / per_block.len() as f64;
        assert!(crossing > 0, "{name}: no edge crosses a seam");
        if empty > 0 {
            cuts_with_empty_blocks += 1;
        }
        worst_imbalance = worst_imbalance.max(heaviest as f64 / mean);
    }
    assert!(cuts_with_empty_blocks >= 4);
    assert!(worst_imbalance > 5.0, "{worst_imbalance}");
}

#[test]
fn the_graph_is_the_same_under_every_cut() {
    let mask = Voxels::from(scene());
    let expected = serial();
    for (block, name) in cuts() {
        let got = build_blocked(&mask, &BuildParams::default(), block)
            .unwrap_or_else(|error| panic!("{name} ({block:?}): {error}"));
        assert_eq!(got, expected, "{name} ({block:?})");
    }
}

#[test]
fn the_sweep_contains_cuts_no_base_index_scheme_could_number() {
    let not = cuts()
        .into_iter()
        .filter(|(block, _)| {
            !blocks_concatenate_in_order(&BlockGrid::new(VOLUME, *block).expect("a grid"))
        })
        .count();
    assert!(not >= 5, "only {not} cuts interleave the blocks");
    assert!(not < cuts().len());
}

/// Concatenating per-block vertex lists is a different numbering on a lattice
/// that interleaves blocks, and `assemble` refuses it rather than searching it.
#[test]
fn an_id_scheme_that_is_not_a_sort_is_refused() {
    let block = [4usize, 4, 4];
    let mask = scene();
    let graph = serial();
    let per_axis = [0, 1, 2].map(|axis| VOLUME[axis].div_ceil(block[axis]));
    let mut by_block: Vec<Vec<[usize; 3]>> = vec![Vec::new(); per_axis.iter().product()];
    for at in set_voxels(mask.view()) {
        let index = (at[0] / block[0]) * per_axis[1] * per_axis[2]
            + (at[1] / block[1]) * per_axis[2]
            + (at[2] / block[2]);
        by_block[index].push(at);
    }
    let concatenated: Vec<[usize; 3]> = by_block.into_iter().flatten().collect();
    let as_u32: Vec<[u32; 3]> = concatenated.iter().map(|at| at.map(|v| v as u32)).collect();
    assert_eq!(
        as_u32.iter().copied().collect::<BTreeSet<_>>(),
        graph.coordinates().iter().copied().collect::<BTreeSet<_>>()
    );
    assert_ne!(as_u32.as_slice(), graph.coordinates());
    let pairs = adjacent_pairs(mask.view(), Connectivity::FacesEdgesAndCorners).expect("pairs");
    assert!(RawGraph::assemble(VOLUME, &concatenated, &pairs).is_err());
}

#[test]
fn an_edge_naming_a_missing_vertex_is_refused() {
    let coordinates = vec![[0usize, 0, 0], [0, 0, 1]];
    let pairs = vec![([0usize, 0, 1], [0usize, 0, 2])];
    assert!(RawGraph::assemble([4, 4, 4], &coordinates, &pairs).is_err());
}

#[test]
fn the_three_connectivities_are_three_graphs() {
    let mask = Voxels::from(scene());
    let counts: Vec<(usize, usize)> = [
        Connectivity::Faces,
        Connectivity::FacesAndEdges,
        Connectivity::FacesEdgesAndCorners,
    ]
    .into_iter()
    .map(|connectivity| {
        let graph = build(&mask, &BuildParams { connectivity }).expect("a graph");
        (graph.n_vertices(), graph.n_edges())
    })
    .collect();
    assert_eq!(counts[0].0, counts[1].0);
    assert_eq!(counts[1].0, counts[2].0);
    assert!(counts[0].1 < counts[1].1 && counts[1].1 < counts[2].1);
}

#[test]
fn a_non_bool_skeleton_is_read_as_non_zero() {
    let as_u8 = scene().mapv(u8::from);
    let graph = build(&Voxels::from(as_u8), &BuildParams::default()).expect("a graph");
    assert_eq!(graph, serial());
}

#[test]
fn the_csr_is_the_edge_list_indexed() {
    let graph = serial();
    let csr = graph.adjacency();
    let total: usize = (0..csr.n_vertices()).map(|v| csr.degree(v as u32)).sum();
    assert_eq!(total, 2 * graph.n_edges());
    for [a, b] in graph.edges() {
        assert!(csr.neighbours(*a).contains(b));
        assert!(csr.neighbours(*b).contains(a));
    }
    assert_eq!(
        graph.structural_bytes(),
        graph.n_vertices() * 12 + graph.n_edges() * 8
    );
    assert_eq!(
        csr.bytes(),
        (graph.n_vertices() + 1) * 4 + graph.n_edges() * 8
    );
}

/// Real data: the skeleton of a 96 x 96 x 32 vessel crop.
fn fixture_root() -> PathBuf {
    PathBuf::from(
        std::env::var("VESSEL_GRAPH_FIXTURES")
            .unwrap_or_else(|_| "/husky/henriksson/clearmap-img/outputs".to_string()),
    )
}

#[test]
#[ignore = "reads a recorded skeleton crop under VESSEL_GRAPH_FIXTURES (default /husky/...)/vasc-binarize/skeleton.npy"]
fn a_real_skeleton_is_the_same_graph_under_every_cut() {
    let path = fixture_root().join("vasc-binarize/skeleton.npy");
    let elements = npy::read_elements_file(&path, npy::OrderPolicy::Only(npy::Order::C))
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let mask: Array3<bool> = elements
        .get::<bool>("skeleton")
        .expect("a bool skeleton")
        .clone()
        .into_dimensionality::<Ix3>()
        .expect("3-D");
    let mask = Voxels::from(mask);
    let serial = build(&mask, &BuildParams::default()).expect("a graph");
    assert!(serial.n_vertices() > 1000);
    for block in [[32, 32, 32], [17, 23, 9], [96, 96, 1]] {
        let got = build_blocked(&mask, &BuildParams::default(), block).expect("a graph");
        assert_eq!(got, serial, "{block:?}");
    }
}
