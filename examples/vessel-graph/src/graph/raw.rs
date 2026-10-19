// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The raw structural graph: a skeleton mask in, a vertex list and an edge
//! list out.
//!
//! **The raw graph is a sort, not a construction.** A vertex's id is the
//! lexicographic rank of its coordinate, and an edge is a pair of adjacent set
//! voxels, lower endpoint first. `blockflow::ops::coordinates` emits one row
//! per set voxel and `blockflow::ops::adjacency` one row per adjacent pair, and
//! both are merged in lexicographic order by `Table`, on every block lattice.
//! So this module only *assembles* a graph from two lists: the one computation
//! left is the coordinate-to-id mapping, a binary search over the vertex list.
//!
//! The binary search is only correct over a strictly ascending list, so
//! [`RawGraph::assemble`] refuses one that is not. An unsorted list would
//! otherwise give a well-formed graph with permuted ids.

use std::collections::BTreeSet;

use blockflow::assemble::{Assembly, PlanBuilder};
use blockflow::env::ArrayEnvironment;
use blockflow::error::{Error, Result};
use blockflow::geometry::BlockGrid;
use blockflow::ops::adjacency::{self, adjacent_pairs, collect_pairs, AdjacentPairsOp, Pair};
use blockflow::ops::coordinates::{self, collect_coordinates, set_voxels};
use blockflow::ops::Connectivity;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

/// The fragment stream the planned vertex phase writes.
pub const VERTEX_STREAM: &str = "graph.vertices";
/// The fragment stream the planned edge phase writes.
pub const EDGE_STREAM: &str = "graph.edges";

/// Vertices are set voxels; edges join set voxels adjacent under
/// `connectivity`. Full 26-connectivity is the default because a thinned
/// skeleton is 26-connected: a diagonal step is the usual way a one-voxel-wide
/// line turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildParams {
    pub connectivity: Connectivity,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            connectivity: Connectivity::FacesEdgesAndCorners,
        }
    }
}

/// A skeleton as a graph: one vertex per set voxel, in lexicographic order of
/// the coordinate, and one edge per adjacent pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawGraph {
    volume: [usize; 3],
    coordinates: Vec<[u32; 3]>,
    edges: Vec<[u32; 2]>,
}

impl RawGraph {
    /// Assemble from a strictly ascending coordinate list and the adjacent
    /// pairs over the same mask.
    pub fn assemble(
        volume: [usize; 3],
        coordinates: &[[usize; 3]],
        pairs: &[Pair],
    ) -> Result<Self> {
        for (axis, extent) in volume.iter().enumerate() {
            if u32::try_from(*extent).is_err() {
                return Err(Error::invalid(format!(
                    "a volume of {extent} on axis {axis} does not fit a u32 coordinate."
                )));
            }
        }
        let narrow = coordinates
            .iter()
            .map(|at| narrowed(*at, volume))
            .collect::<Result<Vec<_>>>()?;
        if let Some(window) = narrow.windows(2).find(|window| window[0] >= window[1]) {
            return Err(Error::invalid(format!(
                "the vertex list is not strictly ascending: {:?} is followed by {:?}. A vertex id \
                 is the lexicographic rank of its coordinate, and the mapping back is a binary \
                 search over this list, so an unsorted list would permute the ids silently.",
                window[0], window[1]
            )));
        }
        let mut edges = Vec::with_capacity(pairs.len());
        for (lower, higher) in pairs {
            let a = search(&narrow, narrowed(*lower, volume)?)?;
            let b = search(&narrow, narrowed(*higher, volume)?)?;
            edges.push([a, b]);
        }
        Ok(Self {
            volume,
            coordinates: narrow,
            edges,
        })
    }

    pub fn volume(&self) -> [usize; 3] {
        self.volume
    }

    pub fn n_vertices(&self) -> usize {
        self.coordinates.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Vertex coordinates, indexed by id and strictly ascending.
    pub fn coordinates(&self) -> &[[u32; 3]] {
        &self.coordinates
    }

    /// Edges as id pairs, lower id first, ascending on the coordinate pair.
    pub fn edges(&self) -> &[[u32; 2]] {
        &self.edges
    }

    /// The id of the vertex at `at`, if that voxel is set.
    pub fn id_of(&self, at: [u32; 3]) -> Option<u32> {
        self.coordinates
            .binary_search(&at)
            .ok()
            .map(|index| index as u32)
    }

    pub fn edge_endpoints(&self, edge: usize) -> Result<([u32; 3], [u32; 3])> {
        let Some([a, b]) = self.edges.get(edge).copied() else {
            return Err(Error::invalid(format!(
                "edge {edge} of a graph with {} of them.",
                self.edges.len()
            )));
        };
        Ok((self.coordinates[a as usize], self.coordinates[b as usize]))
    }

    /// Every edge as its coordinate pair, `[a0, a1, a2, b0, b1, b2]`.
    pub fn edge_coordinates(&self) -> Vec<[u32; 6]> {
        edge_coordinates(&self.coordinates, &self.edges)
    }

    pub fn duplicate_edges(&self) -> usize {
        duplicate_edges(&self.edges)
    }

    pub fn self_loops(&self) -> usize {
        self_loops(&self.edges)
    }

    pub fn adjacency(&self) -> Csr {
        Csr::of(self.n_vertices(), &self.edges)
    }

    /// `12 V + 8 E`: the two flat arrays and nothing else.
    pub fn structural_bytes(&self) -> usize {
        self.coordinates.len() * 12 + self.edges.len() * 8
    }
}

fn narrowed(at: [usize; 3], volume: [usize; 3]) -> Result<[u32; 3]> {
    let mut out = [0u32; 3];
    for axis in 0..3 {
        if at[axis] >= volume[axis] {
            return Err(Error::invalid(format!(
                "a coordinate at {at:?} is outside the volume {volume:?} on axis {axis}."
            )));
        }
        out[axis] = at[axis] as u32;
    }
    Ok(out)
}

fn search(coordinates: &[[u32; 3]], at: [u32; 3]) -> Result<u32> {
    coordinates
        .binary_search(&at)
        .map(|index| index as u32)
        .map_err(|_| {
            Error::invalid(format!(
                "an edge names {at:?}, which is not in the vertex list. The two lists must come \
                 from the same mask."
            ))
        })
}

pub(crate) fn edge_coordinates(coordinates: &[[u32; 3]], edges: &[[u32; 2]]) -> Vec<[u32; 6]> {
    edges
        .iter()
        .map(|[a, b]| {
            let p = coordinates[*a as usize];
            let q = coordinates[*b as usize];
            [p[0], p[1], p[2], q[0], q[1], q[2]]
        })
        .collect()
}

pub(crate) fn duplicate_edges(edges: &[[u32; 2]]) -> usize {
    let mut seen = BTreeSet::new();
    edges.len() - edges.iter().filter(|edge| seen.insert(**edge)).count()
}

pub(crate) fn self_loops(edges: &[[u32; 2]]) -> usize {
    edges.iter().filter(|[a, b]| a == b).count()
}

/// Compressed adjacency: every vertex's neighbours, in edge order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Csr {
    offsets: Vec<u32>,
    targets: Vec<u32>,
}

impl Csr {
    pub fn of(n_vertices: usize, edges: &[[u32; 2]]) -> Self {
        let mut offsets = vec![0u32; n_vertices + 1];
        for [a, b] in edges {
            offsets[*a as usize + 1] += 1;
            offsets[*b as usize + 1] += 1;
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }
        let mut cursor = offsets.clone();
        let mut targets = vec![0u32; edges.len() * 2];
        for [a, b] in edges {
            targets[cursor[*a as usize] as usize] = *b;
            cursor[*a as usize] += 1;
            targets[cursor[*b as usize] as usize] = *a;
            cursor[*b as usize] += 1;
        }
        Self { offsets, targets }
    }

    pub fn n_vertices(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn degree(&self, vertex: u32) -> usize {
        let at = vertex as usize;
        (self.offsets[at + 1] - self.offsets[at]) as usize
    }

    pub fn neighbours(&self, vertex: u32) -> &[u32] {
        let at = vertex as usize;
        &self.targets[self.offsets[at] as usize..self.offsets[at + 1] as usize]
    }

    /// The number of connected components, isolated vertices included.
    pub fn components(&self) -> usize {
        let mut seen = vec![false; self.n_vertices()];
        let mut stack = Vec::new();
        let mut found = 0;
        for root in 0..self.n_vertices() {
            if seen[root] {
                continue;
            }
            found += 1;
            seen[root] = true;
            stack.push(root as u32);
            while let Some(vertex) = stack.pop() {
                for &next in self.neighbours(vertex) {
                    if !seen[next as usize] {
                        seen[next as usize] = true;
                        stack.push(next);
                    }
                }
            }
        }
        found
    }

    /// The number of independent cycles, `E - V + C`.
    pub fn betti_1(&self, n_edges: usize) -> isize {
        n_edges as isize - self.n_vertices() as isize + self.components() as isize
    }

    pub fn bytes(&self) -> usize {
        self.offsets.len() * 4 + self.targets.len() * 4
    }
}

/// The skeleton as a set mask: `bool` as it is, any other type as `!= 0`.
fn mask_of(skeleton: &Voxels) -> Result<Voxels> {
    Ok(match skeleton.dtype() {
        Dtype::Bool => skeleton.clone(),
        _ => Voxels::from(skeleton.nonzero()),
    })
}

/// The raw graph of a skeleton held in memory, by a single walk.
pub fn build(skeleton: &Voxels, params: &BuildParams) -> Result<RawGraph> {
    let mask = mask_of(skeleton)?;
    let view = mask.view::<bool>()?;
    let coordinates = set_voxels(view);
    let pairs = adjacent_pairs(view, params.connectivity)?;
    RawGraph::assemble(mask.shape(), &coordinates, &pairs)
}

/// The same graph, by the two planned row phases over a block lattice.
///
/// The vertex and edge phases each read the mask, and blockflow refuses two
/// pixel-reading fragment phases in one plan, so they run as two plans. The
/// answer is identical to [`build`] on every lattice, which is what
/// `tests/graph_raw.rs` checks.
pub fn build_blocked(
    skeleton: &Voxels,
    params: &BuildParams,
    block: [usize; 3],
) -> Result<RawGraph> {
    let mask = mask_of(skeleton)?;
    let volume = mask.shape();

    let grid = BlockGrid::new(volume, block)?;
    let mut plan = PlanBuilder::new(volume, Dtype::Bool, grid.clone());
    coordinates::append_to(&mut plan, VERTEX_STREAM, Lifecycle::DeleteOnExit)?;
    let env = run(&mask, &plan.finish()?, "graph vertices")?;
    let vertices = collect_coordinates(&env, VERTEX_STREAM, 0, volume)?;

    let mut plan = PlanBuilder::new(volume, Dtype::Bool, grid);
    let op = AdjacentPairsOp::new("adjacent pairs", EDGE_STREAM, Lifecycle::DeleteOnExit)
        .connecting(params.connectivity);
    adjacency::append_to(&mut plan, op)?;
    let env = run(&mask, &plan.finish()?, "graph edges")?;
    let pairs = collect_pairs(&env, EDGE_STREAM, 0, volume)?;

    RawGraph::assemble(volume, &vertices, &pairs)
}

fn run(mask: &Voxels, plan: &Assembly, name: &'static str) -> Result<ArrayEnvironment> {
    let env = ArrayEnvironment::for_decomposition(mask.clone(), &plan.decomposition, [8, 8, 8])?;
    execute_phases(
        name,
        &plan.workflow,
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &plan.work(),
    )?;
    Ok(env)
}
