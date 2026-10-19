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

use blockflow::error::{Error, Result};

use super::raw::{duplicate_edges, edge_coordinates, self_loops, Csr, RawGraph};

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

/// The cleaned structural graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CleanGraph {
    volume: [usize; 3],
    coordinates: Vec<[u32; 3]>,
    origins: Vec<CleanVertex>,
    edges: Vec<[u32; 2]>,
    cliques: Vec<Clique>,
    report: CleanReport,
}

impl CleanGraph {
    pub fn volume(&self) -> [usize; 3] {
        self.volume
    }

    pub fn n_vertices(&self) -> usize {
        self.coordinates.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Coordinates in vertex order. Neither sorted nor unique.
    pub fn coordinates(&self) -> &[[u32; 3]] {
        &self.coordinates
    }

    /// The origin of every vertex, in vertex order.
    pub fn origins(&self) -> &[CleanVertex] {
        &self.origins
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

    /// The id of a vertex by its origin. There is no lookup by coordinate,
    /// because coordinates are not unique here.
    pub fn id_of(&self, key: CleanVertex) -> Option<u32> {
        self.origins
            .iter()
            .position(|origin| *origin == key)
            .map(|index| index as u32)
    }

    /// How many vertices share a coordinate with an earlier one.
    pub fn duplicate_coordinates(&self) -> usize {
        let mut seen = BTreeSet::new();
        self.coordinates.len()
            - self
                .coordinates
                .iter()
                .filter(|at| seen.insert(**at))
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
        edge_coordinates(&self.coordinates, &self.edges)
    }

    pub fn adjacency(&self) -> Csr {
        Csr::of(self.n_vertices(), &self.edges)
    }

    pub fn structural_bytes(&self) -> usize {
        self.coordinates.len() * 12 + self.edges.len() * 8
    }
}

/// The collapse, resident over the graph. No volume is read.
pub fn clean(graph: &RawGraph, params: &CleanParams) -> Result<CleanGraph> {
    let n_vertices = graph.n_vertices();
    let csr = graph.adjacency();
    let branch: Vec<bool> = (0..n_vertices)
        .map(|vertex| csr.degree(vertex as u32) >= 3)
        .collect();
    let branch_points = branch.iter().filter(|is| **is).count();
    let members = branch_components(&csr, &branch);

    if members.is_empty() {
        return Ok(CleanGraph {
            volume: graph.volume(),
            coordinates: graph.coordinates().to_vec(),
            origins: graph
                .coordinates()
                .iter()
                .map(|at| CleanVertex::Voxel(*at))
                .collect(),
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
    let mut clique_of = vec![u32::MAX; n_vertices];
    for (clique, vertices) in members.iter().enumerate() {
        for vertex in vertices {
            clique_of[*vertex as usize] = clique as u32;
        }
    }

    let mut retained = Vec::<[u32; 2]>::new();
    let mut outside = vec![BTreeSet::<u32>::new(); members.len()];
    let mut intra_clique_edges = 0usize;
    let mut clique_outside_edges = 0usize;
    for [a, b] in graph.edges() {
        let (left, right) = (clique_of[*a as usize], clique_of[*b as usize]);
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
            (false, false) => retained.push([*a, *b]),
        }
    }

    let cliques: Vec<Clique> = members
        .iter()
        .enumerate()
        .map(|(clique, vertices)| Clique {
            centre: clique_centre(graph.coordinates(), vertices),
            members: vertices.clone(),
            outside: outside[clique].iter().copied().collect(),
        })
        .collect();

    // Centres are numbered after the raw vertices; `subset` renumbers.
    let mut keep = vec![true; expanded];
    for clique in &cliques {
        for vertex in &clique.members {
            keep[*vertex as usize] = false;
        }
    }
    let mut edges = retained;
    for (index, clique) in cliques.iter().enumerate() {
        let centre = (n_vertices + index) as u32;
        if clique.is_dropped() {
            keep[centre as usize] = false;
            continue;
        }
        for neighbour in &clique.outside {
            edges.push([*neighbour, centre]);
        }
    }
    let mut coordinates = graph.coordinates().to_vec();
    let mut origins: Vec<CleanVertex> = graph
        .coordinates()
        .iter()
        .map(|at| CleanVertex::Voxel(*at))
        .collect();
    for clique in &cliques {
        coordinates.push(clique.centre);
        origins.push(CleanVertex::Centre {
            anchor: clique.anchor(),
        });
    }
    let (mut coordinates, mut origins, mut edges) = subset(&coordinates, &origins, &edges, &keep);

    let mut self_loops_removed = 0;
    if params.remove_self_loops {
        let before = edges.len();
        edges.retain(|[a, b]| a != b);
        self_loops_removed = before - edges.len();
    }
    let mut isolated_removed = 0;
    if params.remove_isolated_vertices {
        let mut degree = vec![0usize; coordinates.len()];
        for [a, b] in &edges {
            degree[*a as usize] += 1;
            degree[*b as usize] += 1;
        }
        let keep: Vec<bool> = degree.iter().map(|degree| *degree > 0).collect();
        isolated_removed = keep.iter().filter(|keep| !**keep).count();
        (coordinates, origins, edges) = subset(&coordinates, &origins, &edges, &keep);
    }

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
        coordinates,
        origins,
        edges,
        cliques,
        report,
    })
}

/// Components of the branch-point subgraph with more than one vertex,
/// ascending by smallest member, each ascending.
fn branch_components(csr: &Csr, branch: &[bool]) -> Vec<Vec<u32>> {
    let mut seen = vec![false; branch.len()];
    let mut stack = Vec::new();
    let mut out = Vec::new();
    for start in 0..branch.len() {
        if !branch[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start as u32);
        let mut component = Vec::new();
        while let Some(vertex) = stack.pop() {
            component.push(vertex);
            for next in csr.neighbours(vertex) {
                if branch[*next as usize] && !seen[*next as usize] {
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

type Subset = (Vec<[u32; 3]>, Vec<CleanVertex>, Vec<[u32; 2]>);

/// Keep the marked vertices in order, renumber, and keep every edge both of
/// whose endpoints survived, in edge order.
fn subset(
    coordinates: &[[u32; 3]],
    origins: &[CleanVertex],
    edges: &[[u32; 2]],
    keep: &[bool],
) -> Subset {
    let mut renumbered = vec![u32::MAX; keep.len()];
    let mut kept_coordinates = Vec::new();
    let mut kept_origins = Vec::new();
    for (old, keep) in keep.iter().enumerate() {
        if *keep {
            renumbered[old] = kept_coordinates.len() as u32;
            kept_coordinates.push(coordinates[old]);
            kept_origins.push(origins[old]);
        }
    }
    let kept_edges = edges
        .iter()
        .filter(|[a, b]| keep[*a as usize] && keep[*b as usize])
        .map(|[a, b]| [renumbered[*a as usize], renumbered[*b as usize]])
        .collect();
    (kept_coordinates, kept_origins, kept_edges)
}
