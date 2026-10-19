// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The largest connected component, with the index maps that carry every
//! column onto it.
//!
//! Components come from `blockflow::ops::components::Union`. The winner is
//! the component with the most vertices; a tie goes to the component holding
//! the smallest vertex id, so the answer does not depend on a traversal order.

use blockflow::error::{Error, Result};
use blockflow::ops::components::Union;

/// The kept part of a graph and how it maps back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    source_vertices: usize,
    source_edges: usize,
    /// Old vertex id of each kept vertex, ascending.
    vertices: Vec<u32>,
    /// Old edge id of each kept edge, ascending.
    edges: Vec<u32>,
    /// The kept edges, renumbered.
    connectivity: Vec<[u32; 2]>,
    /// The number of components in the source graph.
    components: usize,
}

impl Component {
    pub fn source_vertices(&self) -> usize {
        self.source_vertices
    }

    pub fn source_edges(&self) -> usize {
        self.source_edges
    }

    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn vertices(&self) -> &[u32] {
        &self.vertices
    }

    pub fn edges(&self) -> &[u32] {
        &self.edges
    }

    pub fn connectivity(&self) -> &[[u32; 2]] {
        &self.connectivity
    }

    pub fn components(&self) -> usize {
        self.components
    }

    /// Per-edge `[start, end)` ranges of a flat per-point buffer, carried onto
    /// the kept edges: the new ranges, and the old row of every new point.
    pub fn gather_ranges(&self, ranges: &[[u32; 2]]) -> Result<(Vec<[u32; 2]>, Vec<u32>)> {
        if ranges.len() != self.source_edges {
            return Err(Error::invalid(format!(
                "{} polyline ranges against a graph of {} edges.",
                ranges.len(),
                self.source_edges
            )));
        }
        let mut new_ranges = Vec::with_capacity(self.edges.len());
        let mut rows = Vec::new();
        for old in &self.edges {
            let [start, end] = ranges[*old as usize];
            let first = rows.len() as u32;
            rows.extend(start..end);
            new_ranges.push([first, rows.len() as u32]);
        }
        Ok((new_ranges, rows))
    }
}

/// The largest connected component of a graph given as an edge list.
pub fn largest_component(n_vertices: usize, edges: &[[u32; 2]]) -> Result<Component> {
    if n_vertices == 0 {
        return Err(Error::invalid(
            "the largest component of a graph with no vertices: the reduction produced nothing.",
        ));
    }
    let mut union = Union::new(n_vertices);
    for (id, [a, b]) in edges.iter().enumerate() {
        if *a as usize >= n_vertices || *b as usize >= n_vertices {
            return Err(Error::invalid(format!(
                "edge {id} names a vertex of {:?} in a graph of {n_vertices}.",
                [a, b]
            )));
        }
        union.union(*a as usize, *b as usize);
    }
    let roots: Vec<usize> = (0..n_vertices).map(|vertex| union.find(vertex)).collect();
    let mut size = vec![0usize; n_vertices];
    for root in &roots {
        size[*root] += 1;
    }
    // Scanning vertices in id order meets each component first at its
    // smallest vertex, so a strict `>` keeps the earliest of equal sizes.
    let mut winner = roots[0];
    let mut components = 0;
    let mut met = vec![false; n_vertices];
    for root in &roots {
        if !met[*root] {
            met[*root] = true;
            components += 1;
            if size[*root] > size[winner] {
                winner = *root;
            }
        }
    }
    let mut new_id = vec![u32::MAX; n_vertices];
    let mut vertices = Vec::new();
    for (old, root) in roots.iter().enumerate() {
        if *root == winner {
            new_id[old] = vertices.len() as u32;
            vertices.push(old as u32);
        }
    }
    let mut kept = Vec::new();
    let mut connectivity = Vec::new();
    for (id, [a, b]) in edges.iter().enumerate() {
        if roots[*a as usize] == winner {
            kept.push(id as u32);
            connectivity.push([new_id[*a as usize], new_id[*b as usize]]);
        }
    }
    Ok(Component {
        source_vertices: n_vertices,
        source_edges: edges.len(),
        vertices,
        edges: kept,
        connectivity,
        components,
    })
}
