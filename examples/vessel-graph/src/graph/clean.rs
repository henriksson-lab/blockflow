// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Clique collapse: every cluster of adjacent branch points becomes one vertex.
//!
//! A thinned skeleton seldom meets itself at a single voxel. Where three or
//! more vessels join, several voxels of degree >= 3 touch each other and form a
//! small dense *clique* of spurious cycles. The collapse:
//!
//! 1. marks the branch points (raw degree >= 3);
//! 2. takes the connected components of the branch-point subgraph with more
//!    than one member: the cliques, numbered by their smallest member (the
//!    *anchor*);
//! 3. replaces each clique by a centre at the per-axis median of its members
//!    (upper median on even counts), joined once to every distinct raw vertex
//!    outside the clique that one of its members touched;
//! 4. drops a clique that touches nothing outside it (a component made only of
//!    branch points), and then self-loops and isolated vertices.
//!
//! The vertex order is the surviving raw vertices ascending, then the centres
//! ascending by anchor. **A coordinate is not a key**: a centre can land on a
//! surviving voxel (`tests/graph_clean.rs` builds one), so every vertex also
//! carries its [`CleanVertex`] origin and properties are carried by origin.

use std::collections::BTreeSet;
use std::sync::Arc;

use blockflow::error::{Error, Result};

use super::raw::{duplicate_edges, self_loops, Csr, RawGraph};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CleanParams {
    pub remove_self_loops: bool,
    pub remove_isolated_vertices: bool,
}

impl Default for CleanParams {
    fn default() -> Self {
        Self {
            remove_self_loops: true,
            remove_isolated_vertices: true,
        }
    }
}

/// What a cleaned vertex is: a raw voxel that survived, or a collapsed
/// clique's centre, named by its anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CleanVertex {
    Voxel([u32; 3]),
    Centre { anchor: u32 },
}

/// One clique, as found and as collapsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clique {
    members: Vec<u32>,
    centre: [u32; 3],
    outside: Vec<u32>,
}

impl Clique {
    /// The raw vertex ids in the clique, ascending.
    pub fn members(&self) -> &[u32] {
        &self.members
    }

    /// The smallest member id, which names the clique.
    pub fn anchor(&self) -> u32 {
        self.members[0]
    }

    /// The per-axis median of the members' coordinates, upper on ties.
    pub fn centre(&self) -> [u32; 3] {
        self.centre
    }

    /// The distinct raw vertices outside the clique that an edge reached,
    /// ascending.
    pub fn outside(&self) -> &[u32] {
        &self.outside
    }

    /// True when the clique touches nothing outside it and so has no vertex in
    /// the cleaned graph.
    pub fn is_dropped(&self) -> bool {
        self.outside.is_empty()
    }
}

/// What the collapse did, as counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CleanReport {
    /// Components of branch points holding more than one vertex.
    pub cliques_total: usize,
    /// Those with at least one outside neighbour.
    pub cliques_collapsed: usize,
    pub vertices_in_cliques: usize,
    /// Raw vertices of degree >= 3, cliqued or not.
    pub branch_points: usize,
    /// Raw edges inside one clique (dropped).
    pub intra_clique_edges: usize,
    /// Raw edges with exactly one endpoint in a clique.
    pub clique_outside_edges: usize,
    /// Distinct outside neighbours over all cliques.
    pub clique_outside_neighbours: usize,
    pub self_loops_removed: usize,
    pub isolated_removed: usize,
    /// True when there was no clique and the graph was returned unchanged.
    pub untouched: bool,
}

impl CleanReport {
    /// Cycles the collapse is expected to remove, from the clique geometry
    /// alone: internal edges, less the vertices absorbed, plus the edges the
    /// deduplication of outside neighbours removed.
    pub fn predicted_cycles_removed(&self) -> isize {
        let absorbed = self.vertices_in_cliques as isize - self.cliques_total as isize;
        let deduplicated =
            self.clique_outside_edges as isize - self.clique_outside_neighbours as isize;
        self.intra_clique_edges as isize - absorbed + deduplicated
    }
}

/// The cleaned structural graph: the raw vertices that survived, by raw id,
/// then the kept clique centres, and the edges between them. A surviving
/// vertex's coordinate is read off the raw graph's (shared), so a cleaned
/// graph adds four bytes a vertex to it, and its edges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CleanGraph {
    volume: [usize; 3],
    raw_coordinates: Arc<Vec<[u32; 3]>>,
    /// Raw ids of the surviving raw vertices, ascending: cleaned vertex `i`
    /// below its length is raw vertex `survivors[i]`.
    survivors: Arc<Vec<u32>>,
    /// Per kept centre, in vertex order (after the survivors): its clique's
    /// index in `cliques`.
    centres: Vec<u32>,
    edges: Vec<[u32; 2]>,
    cliques: Vec<Clique>,
    report: CleanReport,
}

impl CleanGraph {
    pub fn volume(&self) -> [usize; 3] {
        self.volume
    }

    pub fn n_vertices(&self) -> usize {
        self.survivors.len() + self.centres.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Vertex `vertex`'s coordinate.
    pub fn coordinate(&self, vertex: usize) -> [u32; 3] {
        match self.survivors.get(vertex) {
            Some(raw) => self.raw_coordinates[*raw as usize],
            None => self.cliques[self.centres[vertex - self.survivors.len()] as usize].centre,
        }
    }

    /// Coordinates in vertex order. Neither sorted nor unique.
    pub fn coordinates(&self) -> Vec<[u32; 3]> {
        (0..self.n_vertices())
            .map(|vertex| self.coordinate(vertex))
            .collect()
    }

    /// Vertex `vertex`'s origin.
    pub fn origin(&self, vertex: usize) -> CleanVertex {
        match self.survivors.get(vertex) {
            Some(raw) => CleanVertex::Voxel(self.raw_coordinates[*raw as usize]),
            None => CleanVertex::Centre {
                anchor: self.cliques[self.centres[vertex - self.survivors.len()] as usize].anchor(),
            },
        }
    }

    /// The origin of every vertex, in vertex order.
    pub fn origins(&self) -> Vec<CleanVertex> {
        (0..self.n_vertices())
            .map(|vertex| self.origin(vertex))
            .collect()
    }

    /// Retained raw edges in raw order, then the clique edges by clique and
    /// ascending neighbour, each as `[neighbour, centre]`.
    pub fn edges(&self) -> &[[u32; 2]] {
        &self.edges
    }

    /// Every clique found, ascending by anchor, dropped ones included.
    pub fn cliques(&self) -> &[Clique] {
        &self.cliques
    }

    pub fn report(&self) -> &CleanReport {
        &self.report
    }

    /// Raw ids of the surviving raw vertices, which are the first cleaned
    /// vertices, in order.
    pub fn survivors(&self) -> &[u32] {
        &self.survivors
    }

    pub(crate) fn shared_survivors(&self) -> Arc<Vec<u32>> {
        Arc::clone(&self.survivors)
    }

    /// Per kept centre, after the survivors in vertex order, its clique's
    /// index in [`Self::cliques`].
    pub fn centre_cliques(&self) -> &[u32] {
        &self.centres
    }

    /// Let the edges go once the graph has been reduced.
    pub(crate) fn drop_edges(&mut self) {
        self.edges = Vec::new();
    }

    /// The id of a vertex by its origin. There is no lookup by coordinate,
    /// because coordinates are not unique here.
    pub fn id_of(&self, key: CleanVertex) -> Option<u32> {
        match key {
            CleanVertex::Voxel(at) => {
                let raw = self.raw_coordinates.binary_search(&at).ok()? as u32;
                self.survivors
                    .binary_search(&raw)
                    .ok()
                    .map(|index| index as u32)
            }
            CleanVertex::Centre { anchor } => self
                .centres
                .iter()
                .position(|clique| self.cliques[*clique as usize].anchor() == anchor)
                .map(|index| (self.survivors.len() + index) as u32),
        }
    }

    /// How many vertices share a coordinate with an earlier one.
    pub fn duplicate_coordinates(&self) -> usize {
        let mut seen = BTreeSet::new();
        (0..self.n_vertices())
            .filter(|vertex| !seen.insert(self.coordinate(*vertex)))
            .count()
    }

    pub fn self_loops(&self) -> usize {
        self_loops(&self.edges)
    }

    pub fn duplicate_edges(&self) -> usize {
        duplicate_edges(&self.edges)
    }

    /// Every edge as its coordinate pair, in this graph's own orientation.
    pub fn edge_coordinates(&self) -> Vec<[u32; 6]> {
        self.edges
            .iter()
            .map(|[a, b]| {
                let p = self.coordinate(*a as usize);
                let q = self.coordinate(*b as usize);
                [p[0], p[1], p[2], q[0], q[1], q[2]]
            })
            .collect()
    }

    pub fn adjacency(&self) -> Csr {
        Csr::of(self.n_vertices(), &self.edges)
    }

    /// What the cleaned graph adds to the raw one: a raw id per survivor, a
    /// clique per centre, and the edges.
    pub fn structural_bytes(&self) -> usize {
        self.survivors.len() * 4 + self.centres.len() * 4 + self.edges.len() * 8
    }
}

/// The collapse, resident over the graph. No volume is read.
pub fn clean(graph: &RawGraph, params: &CleanParams) -> Result<CleanGraph> {
    let n_vertices = graph.n_vertices();
    // A raw graph is simple and 26-connected at most, so a byte holds a
    // degree; only `>= 3` is asked of it.
    let mut degree = vec![0u8; n_vertices];
    for [a, b] in graph.edges() {
        degree[*a as usize] = degree[*a as usize].saturating_add(1);
        degree[*b as usize] = degree[*b as usize].saturating_add(1);
    }
    // The branch points, ascending, and the cliques among them.
    let branch: Vec<u32> = (0..n_vertices as u32)
        .filter(|vertex| degree[*vertex as usize] >= 3)
        .collect();
    let branch_points = branch.len();
    let branch_index = |vertex: u32| -> Option<usize> {
        match degree[vertex as usize] >= 3 {
            true => branch.binary_search(&vertex).ok(),
            false => None,
        }
    };
    let members = branch_components(graph.edges(), &branch, &branch_index);

    if members.is_empty() {
        return Ok(CleanGraph {
            volume: graph.volume(),
            raw_coordinates: graph.shared_coordinates(),
            survivors: Arc::new((0..n_vertices as u32).collect()),
            centres: Vec::new(),
            edges: graph.edges().to_vec(),
            cliques: Vec::new(),
            report: CleanReport {
                branch_points,
                untouched: true,
                ..CleanReport::default()
            },
        });
    }

    let expanded = n_vertices + members.len();
    if u32::try_from(expanded).is_err() {
        return Err(Error::invalid(format!(
            "{n_vertices} vertices plus {} clique centres do not fit a u32 vertex id.",
            members.len()
        )));
    }
    // The clique of each branch point, by its index among them.
    let mut clique_of_branch = vec![u32::MAX; branch.len()];
    for (clique, vertices) in members.iter().enumerate() {
        for vertex in vertices {
            let index = branch_index(*vertex).expect("a clique member is a branch point");
            clique_of_branch[index] = clique as u32;
        }
    }
    let clique_of = |vertex: u32| -> u32 {
        branch_index(vertex).map_or(u32::MAX, |index| clique_of_branch[index])
    };

    // The edges in expanded ids: a raw vertex is its raw id, a centre is
    // `n_vertices + clique`.
    let mut edges = Vec::<[u32; 2]>::new();
    let mut outside = vec![BTreeSet::<u32>::new(); members.len()];
    let mut intra_clique_edges = 0usize;
    let mut clique_outside_edges = 0usize;
    for [a, b] in graph.edges() {
        let (left, right) = (clique_of(*a), clique_of(*b));
        match (left != u32::MAX, right != u32::MAX) {
            // Two cliques never touch: they would be one component.
            (true, true) => intra_clique_edges += 1,
            (true, false) => {
                outside[left as usize].insert(*b);
                clique_outside_edges += 1;
            }
            (false, true) => {
                outside[right as usize].insert(*a);
                clique_outside_edges += 1;
            }
            (false, false) => edges.push([*a, *b]),
        }
    }
    let cliques: Vec<Clique> = members
        .into_iter()
        .zip(outside)
        .map(|(vertices, outside)| Clique {
            centre: clique_centre(graph.coordinates(), &vertices),
            members: vertices,
            outside: outside.into_iter().collect(),
        })
        .collect();
    for (index, clique) in cliques.iter().enumerate() {
        let centre = (n_vertices + index) as u32;
        for neighbour in &clique.outside {
            edges.push([*neighbour, centre]);
        }
    }

    // Which vertices stay: the raw vertices outside every clique and the
    // centres of the cliques that touch something, then (if asked) only
    // those an edge still reaches.
    let mut keep = vec![true; expanded];
    for clique in &cliques {
        for vertex in &clique.members {
            keep[*vertex as usize] = false;
        }
    }
    for (index, clique) in cliques.iter().enumerate() {
        if clique.is_dropped() {
            keep[n_vertices + index] = false;
        }
    }
    let mut self_loops_removed = 0;
    if params.remove_self_loops {
        let before = edges.len();
        edges.retain(|[a, b]| a != b);
        self_loops_removed = before - edges.len();
    }
    let mut isolated_removed = 0;
    if params.remove_isolated_vertices {
        let mut touched = vec![false; expanded];
        for [a, b] in &edges {
            touched[*a as usize] = true;
            touched[*b as usize] = true;
        }
        for (keep, touched) in keep.iter_mut().zip(&touched) {
            if *keep && !touched {
                *keep = false;
                isolated_removed += 1;
            }
        }
    }

    // Renumber: the kept raw vertices ascending, then the kept centres by
    // anchor; every edge's endpoints are kept by construction.
    let mut renumbered = vec![u32::MAX; expanded];
    let mut survivors = Vec::new();
    let mut centres = Vec::new();
    for (old, keep) in keep.iter().enumerate() {
        if !*keep {
            continue;
        }
        renumbered[old] = (survivors.len() + centres.len()) as u32;
        match old < n_vertices {
            true => survivors.push(old as u32),
            false => centres.push((old - n_vertices) as u32),
        }
    }
    drop(keep);
    for edge in &mut edges {
        *edge = edge.map(|end| renumbered[end as usize]);
    }
    drop(renumbered);

    let report = CleanReport {
        cliques_total: cliques.len(),
        cliques_collapsed: cliques.iter().filter(|clique| !clique.is_dropped()).count(),
        vertices_in_cliques: cliques.iter().map(|clique| clique.members.len()).sum(),
        branch_points,
        intra_clique_edges,
        clique_outside_edges,
        clique_outside_neighbours: cliques.iter().map(|clique| clique.outside.len()).sum(),
        self_loops_removed,
        isolated_removed,
        untouched: false,
    };
    Ok(CleanGraph {
        volume: graph.volume(),
        raw_coordinates: graph.shared_coordinates(),
        survivors: Arc::new(survivors),
        centres,
        edges,
        cliques,
        report,
    })
}

/// Components of the branch-point subgraph with more than one vertex,
/// ascending by smallest member, each ascending. `branch` lists the branch
/// points ascending and `index` finds one in it.
fn branch_components(
    edges: &[[u32; 2]],
    branch: &[u32],
    index: &dyn Fn(u32) -> Option<usize>,
) -> Vec<Vec<u32>> {
    // The branch-to-branch edges as an adjacency over branch indices.
    let mut offsets = vec![0u32; branch.len() + 1];
    let mut between = Vec::new();
    for [a, b] in edges {
        if let (Some(left), Some(right)) = (index(*a), index(*b)) {
            offsets[left + 1] += 1;
            offsets[right + 1] += 1;
            between.push([left as u32, right as u32]);
        }
    }
    for at in 1..offsets.len() {
        offsets[at] += offsets[at - 1];
    }
    let mut cursor = offsets.clone();
    let mut targets = vec![0u32; between.len() * 2];
    for [a, b] in &between {
        targets[cursor[*a as usize] as usize] = *b;
        cursor[*a as usize] += 1;
        targets[cursor[*b as usize] as usize] = *a;
        cursor[*b as usize] += 1;
    }
    drop(between);
    let mut seen = vec![false; branch.len()];
    let mut stack = Vec::new();
    let mut out = Vec::new();
    for start in 0..branch.len() {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start as u32);
        let mut component = Vec::new();
        while let Some(vertex) = stack.pop() {
            component.push(branch[vertex as usize]);
            let at = vertex as usize;
            for next in &targets[offsets[at] as usize..offsets[at + 1] as usize] {
                if !seen[*next as usize] {
                    seen[*next as usize] = true;
                    stack.push(*next);
                }
            }
        }
        if component.len() > 1 {
            component.sort_unstable();
            out.push(component);
        }
    }
    out
}

/// The per-axis median of the members' coordinates, upper median when the
/// count is even, so that the centre is always a coordinate some member has on
/// that axis.
fn clique_centre(coordinates: &[[u32; 3]], members: &[u32]) -> [u32; 3] {
    let mut centre = [0u32; 3];
    let mut column = Vec::with_capacity(members.len());
    for (axis, value) in centre.iter_mut().enumerate() {
        column.clear();
        column.extend(
            members
                .iter()
                .map(|member| coordinates[*member as usize][axis]),
        );
        column.sort_unstable();
        *value = column[column.len() / 2];
    }
    centre
}
