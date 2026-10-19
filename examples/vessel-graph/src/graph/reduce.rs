// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Degree-2 chain contraction: a vessel between two branch points becomes one
//! edge carrying its polyline.
//!
//! The reduced vertices are the cleaned vertices whose degree is not 2
//! (branch points, endpoints and isolated vertices), ascending by cleaned id.
//! Every cleaned edge lies on exactly one *chain*:
//!
//! * a **traced** chain starts at an edge with exactly one degree-2 endpoint,
//!   in edge order, and walks through degree-2 vertices until it reaches a
//!   vertex that is not; it can return to where it started (a self-loop);
//! * a **direct** chain is an edge whose two endpoints are both not degree 2;
//! * a **pure loop** is a cycle of degree-2 vertices with no branch point on
//!   it. It has no endpoint to become an edge between and is dropped whole.
//!
//! Each surviving chain becomes one reduced edge between its two ends. Its
//! *edge geometry* is the chain's vertex list, endpoints included, as cleaned
//! vertex ids: the polyline that later stages sample properties along. Two
//! chains can join the same pair of branch points, so the polylines cannot be
//! recovered from the edge list afterwards.

use blockflow::error::{Error, Result};

use super::clean::CleanGraph;
use super::raw::{duplicate_edges, edge_coordinates, self_loops, Csr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceParams {
    /// Keep the polylines. Off saves their memory when only the topology is
    /// wanted; every property carried along an edge needs them.
    pub edge_geometry: bool,
}

impl Default for ReduceParams {
    fn default() -> Self {
        Self {
            edge_geometry: true,
        }
    }
}

/// Every vertex's `(edge, neighbour)` entries, ascending by edge id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Incidence {
    offsets: Vec<u32>,
    entries: Vec<(u32, u32)>,
}

impl Incidence {
    pub(crate) fn of(n_vertices: usize, edges: &[[u32; 2]]) -> Self {
        let mut offsets = vec![0u32; n_vertices + 1];
        for [a, b] in edges {
            offsets[*a as usize + 1] += 1;
            offsets[*b as usize + 1] += 1;
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }
        let mut cursor = offsets.clone();
        let mut entries = vec![(0u32, 0u32); edges.len() * 2];
        for (edge, [a, b]) in edges.iter().enumerate() {
            let edge = edge as u32;
            entries[cursor[*a as usize] as usize] = (edge, *b);
            cursor[*a as usize] += 1;
            entries[cursor[*b as usize] as usize] = (edge, *a);
            cursor[*b as usize] += 1;
        }
        Self { offsets, entries }
    }

    pub(crate) fn entries(&self, vertex: u32) -> &[(u32, u32)] {
        let at = vertex as usize;
        &self.entries[self.offsets[at] as usize..self.offsets[at + 1] as usize]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChainKind {
    Traced,
    Direct,
    PureLoop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Chain {
    vertices: Vec<u32>,
    kind: ChainKind,
}

/// The polylines: one `[offsets[e], offsets[e + 1])` range of cleaned vertex
/// ids per reduced edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeGeometry {
    offsets: Vec<u32>,
    points: Vec<u32>,
}

impl EdgeGeometry {
    pub fn n_edges(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    pub fn path(&self, edge: usize) -> &[u32] {
        &self.points[self.offsets[edge] as usize..self.offsets[edge + 1] as usize]
    }

    pub fn points(&self) -> &[u32] {
        &self.points
    }

    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// `[start, end)` per edge.
    pub fn ranges(&self) -> Vec<[u32; 2]> {
        self.offsets
            .windows(2)
            .map(|window| [window[0], window[1]])
            .collect()
    }

    pub fn lengths(&self) -> Vec<u32> {
        self.offsets
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect()
    }

    pub fn bytes(&self) -> usize {
        self.offsets.len() * 4 + self.points.len() * 4
    }
}

/// What the contraction did, as counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReduceReport {
    pub source_vertices: usize,
    pub source_edges: usize,
    pub degree_2_vertices: usize,
    pub isolated_vertices: usize,
    pub traced_chains: usize,
    pub direct_chains: usize,
    pub loop_chains: usize,
    /// Chains without two ends: the pure loops.
    pub dropped_chains: usize,
    pub vertices_dropped_with_chains: usize,
}

impl ReduceReport {
    pub fn chains(&self) -> usize {
        self.traced_chains + self.direct_chains + self.loop_chains
    }
}

/// The reduced graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReducedGraph {
    volume: [usize; 3],
    source_vertices: usize,
    coordinates: Vec<[u32; 3]>,
    sources: Vec<u32>,
    edges: Vec<[u32; 2]>,
    geometry: Option<EdgeGeometry>,
    report: ReduceReport,
}

impl ReducedGraph {
    pub fn volume(&self) -> [usize; 3] {
        self.volume
    }

    pub fn n_vertices(&self) -> usize {
        self.coordinates.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn coordinates(&self) -> &[[u32; 3]] {
        &self.coordinates
    }

    /// The cleaned vertex id each reduced vertex came from, strictly
    /// ascending.
    pub fn sources(&self) -> &[u32] {
        &self.sources
    }

    /// Edges as `[first end, last end]` of their chain.
    pub fn edges(&self) -> &[[u32; 2]] {
        &self.edges
    }

    pub fn geometry(&self) -> Option<&EdgeGeometry> {
        self.geometry.as_ref()
    }

    pub fn report(&self) -> &ReduceReport {
        &self.report
    }

    /// The number of vertices of the cleaned graph this was reduced from.
    pub fn source_vertices(&self) -> usize {
        self.source_vertices
    }

    /// The polyline of one edge, as coordinates of `source`.
    pub fn path_coordinates(&self, source: &CleanGraph, edge: usize) -> Result<Vec<[u32; 3]>> {
        let geometry = self.require_geometry()?;
        self.check_source(source)?;
        if edge >= geometry.n_edges() {
            return Err(Error::invalid(format!(
                "edge {edge} of a reduced graph with {} of them.",
                geometry.n_edges()
            )));
        }
        Ok(geometry
            .path(edge)
            .iter()
            .map(|point| source.coordinates()[*point as usize])
            .collect())
    }

    /// Every polyline point, as coordinates of `source`.
    pub fn geometry_coordinates(&self, source: &CleanGraph) -> Result<Vec<[u32; 3]>> {
        let geometry = self.require_geometry()?;
        self.check_source(source)?;
        Ok(geometry
            .points()
            .iter()
            .map(|point| source.coordinates()[*point as usize])
            .collect())
    }

    pub fn self_loops(&self) -> usize {
        self_loops(&self.edges)
    }

    pub fn duplicate_edges(&self) -> usize {
        duplicate_edges(&self.edges)
    }

    pub fn edge_coordinates(&self) -> Vec<[u32; 6]> {
        edge_coordinates(&self.coordinates, &self.edges)
    }

    pub fn adjacency(&self) -> Csr {
        Csr::of(self.n_vertices(), &self.edges)
    }

    pub fn structural_bytes(&self) -> usize {
        self.coordinates.len() * 12
            + self.edges.len() * 8
            + self.geometry.as_ref().map_or(0, EdgeGeometry::bytes)
    }

    pub(crate) fn require_geometry(&self) -> Result<&EdgeGeometry> {
        self.geometry.as_ref().ok_or_else(|| {
            Error::invalid(
                "this reduced graph carries no edge geometry: it was reduced with \
                 `edge_geometry: false`. The polylines are not recoverable from the edge list, \
                 because two chains can join the same pair of branch points.",
            )
        })
    }

    fn check_source(&self, source: &CleanGraph) -> Result<()> {
        if source.n_vertices() != self.source_vertices {
            return Err(Error::invalid(format!(
                "the edge geometry indexes a graph of {} vertices and was handed one of {}.",
                self.source_vertices,
                source.n_vertices()
            )));
        }
        Ok(())
    }
}

/// The contraction, resident over the cleaned graph.
pub fn reduce(graph: &CleanGraph, params: &ReduceParams) -> Result<ReducedGraph> {
    let n_vertices = graph.n_vertices();
    let edges = graph.edges();
    let mut degree = vec![0u32; n_vertices];
    for [a, b] in edges {
        degree[*a as usize] += 1;
        degree[*b as usize] += 1;
    }
    let incidence = Incidence::of(n_vertices, edges);
    let chains = find_chains(n_vertices, edges, &degree, &incidence)?;

    let mut old_to_new = vec![u32::MAX; n_vertices];
    let mut sources = Vec::new();
    let mut coordinates = Vec::new();
    for (vertex, degree) in degree.iter().enumerate() {
        if *degree != 2 {
            old_to_new[vertex] = sources.len() as u32;
            sources.push(vertex as u32);
            coordinates.push(graph.coordinates()[vertex]);
        }
    }

    let mut reduced = Vec::<[u32; 2]>::new();
    let mut points = Vec::<u32>::new();
    let mut offsets = vec![0u32];
    let mut report = ReduceReport {
        source_vertices: n_vertices,
        source_edges: edges.len(),
        degree_2_vertices: degree.iter().filter(|degree| **degree == 2).count(),
        isolated_vertices: degree.iter().filter(|degree| **degree == 0).count(),
        ..ReduceReport::default()
    };
    for chain in &chains {
        match chain.kind {
            ChainKind::Traced => report.traced_chains += 1,
            ChainKind::Direct => report.direct_chains += 1,
            ChainKind::PureLoop => report.loop_chains += 1,
        }
        let mut ends = chain
            .vertices
            .iter()
            .filter(|vertex| degree[**vertex as usize] != 2);
        let (Some(first), Some(last), None) = (ends.next(), ends.next(), ends.next()) else {
            report.dropped_chains += 1;
            report.vertices_dropped_with_chains += chain.vertices.len();
            continue;
        };
        if Some(first) != chain.vertices.first() || Some(last) != chain.vertices.last() {
            return Err(Error::invalid(format!(
                "a chain's endpoints are {first} and {last} but its path runs {:?}..{:?}.",
                chain.vertices.first(),
                chain.vertices.last()
            )));
        }
        reduced.push([old_to_new[*first as usize], old_to_new[*last as usize]]);
        if params.edge_geometry {
            points.extend_from_slice(&chain.vertices);
            let end = u32::try_from(points.len()).map_err(|_| {
                Error::invalid(format!(
                    "the edge geometry passed {} points, which does not fit a u32 offset.",
                    points.len()
                ))
            })?;
            offsets.push(end);
        }
    }
    Ok(ReducedGraph {
        volume: graph.volume(),
        source_vertices: n_vertices,
        coordinates,
        sources,
        edges: reduced,
        geometry: params
            .edge_geometry
            .then_some(EdgeGeometry { offsets, points }),
        report,
    })
}

fn find_chains(
    n_vertices: usize,
    edges: &[[u32; 2]],
    degree: &[u32],
    incidence: &Incidence,
) -> Result<Vec<Chain>> {
    let mut chains = Vec::new();
    let mut visited_edge = vec![false; edges.len()];
    for (edge, [a, b]) in edges.iter().enumerate() {
        let (left, right) = (degree[*a as usize] == 2, degree[*b as usize] == 2);
        if left == right || visited_edge[edge] {
            continue;
        }
        let (start, mut current) = if left { (*b, *a) } else { (*a, *b) };
        visited_edge[edge] = true;
        let mut vertices = vec![start, current];
        let mut previous = start;
        while degree[current as usize] == 2 {
            let Some((edge, neighbour)) = incidence
                .entries(current)
                .iter()
                .copied()
                .find(|(edge, neighbour)| *neighbour != previous && !visited_edge[*edge as usize])
            else {
                break;
            };
            visited_edge[edge as usize] = true;
            vertices.push(neighbour);
            previous = current;
            current = neighbour;
        }
        chains.push(Chain {
            vertices,
            kind: ChainKind::Traced,
        });
    }
    for [a, b] in edges {
        if degree[*a as usize] != 2 && degree[*b as usize] != 2 {
            chains.push(Chain {
                vertices: vec![*a, *b],
                kind: ChainKind::Direct,
            });
        }
    }
    let mut visited_vertex = vec![false; n_vertices];
    for chain in &chains {
        for vertex in &chain.vertices {
            visited_vertex[*vertex as usize] = true;
        }
    }
    for vertex in 0..n_vertices {
        if visited_vertex[vertex] || degree[vertex] != 2 {
            continue;
        }
        let vertices = loop_from(vertex as u32, degree, incidence)?;
        for vertex in &vertices {
            visited_vertex[*vertex as usize] = true;
        }
        chains.push(Chain {
            vertices,
            kind: ChainKind::PureLoop,
        });
    }
    if let Some(vertex) = (0..n_vertices).find(|v| !visited_vertex[*v] && degree[*v] > 0) {
        return Err(Error::invalid(format!(
            "vertex {vertex}, of degree {}, lies on no chain; the walk and the incidence \
             structure disagree.",
            degree[vertex]
        )));
    }
    Ok(chains)
}

/// The vertices of the degree-2 run through `v0`, in walk order.
fn loop_from(v0: u32, degree: &[u32], incidence: &Incidence) -> Result<Vec<u32>> {
    let entries = incidence.entries(v0);
    let (mut forward, endpoint) = follow(entries[0], v0, degree, incidence)?;
    if endpoint == v0 {
        return Ok(forward);
    }
    let (backward, _) = follow(entries[1], v0, degree, incidence)?;
    forward.reverse();
    forward.push(v0);
    forward.extend(backward);
    Ok(forward)
}

fn follow(
    entry: (u32, u32),
    v0: u32,
    degree: &[u32],
    incidence: &Incidence,
) -> Result<(Vec<u32>, u32)> {
    let (_, mut vertex) = entry;
    let mut previous = v0;
    let mut vertices = vec![vertex];
    while degree[vertex as usize] == 2 && vertex != v0 {
        let Some((_, next)) = incidence
            .entries(vertex)
            .iter()
            .copied()
            .find(|(_, neighbour)| *neighbour != previous)
        else {
            return Err(Error::invalid(format!(
                "the walk through vertex {vertex} has nowhere to go but back: two parallel \
                 edges between two degree-2 vertices are a chain with no direction."
            )));
        };
        previous = vertex;
        vertex = next;
        vertices.push(vertex);
    }
    Ok((vertices, vertex))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::clean::{clean, CleanParams};
    use crate::graph::raw::RawGraph;

    #[test]
    fn the_incidence_entries_are_in_ascending_edge_id() {
        let coordinates: Vec<[usize; 3]> = (0..6).map(|x| [x, 0, 0]).collect();
        let pairs: Vec<_> = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (0, 5)]
            .iter()
            .map(|(a, b)| ([*a, 0, 0], [*b, 0, 0]))
            .collect();
        let raw = RawGraph::assemble([6, 1, 1], &coordinates, &pairs).expect("a graph");
        let cleaned = clean(&raw, &CleanParams::default()).expect("cleaned");
        let incidence = Incidence::of(cleaned.n_vertices(), cleaned.edges());
        let csr = cleaned.adjacency();
        for vertex in 0..cleaned.n_vertices() as u32 {
            let entries = incidence.entries(vertex);
            assert_eq!(entries.len(), csr.degree(vertex));
            assert!(entries.windows(2).all(|pair| pair[0].0 < pair[1].0));
            for (edge, neighbour) in entries {
                let [a, b] = cleaned.edges()[*edge as usize];
                assert!((a == vertex && b == *neighbour) || (b == vertex && a == *neighbour));
            }
        }
    }
}
