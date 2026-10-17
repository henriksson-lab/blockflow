// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Splitting one segment of a supervoxel grouping along its weakest membrane.**
//
// Proofreading edits a fragment -> segment map (`crates/ngff-label-map`): a
// false merge is a segment whose fragments belong to two cells. The user marks
// a few fragments of each cell ("seeds"); this module finds the partition of the
// segment's fragments that separates the seed groups and cuts the least
// attraction, as an exact minimum s-t cut ([`crate::graph_cut`]) over the
// segment's subgraph of a [`Rag`].
//
// Capacity rule
// -------------
// The capacity of the edge between fragments `u` and `v` is their *attraction*,
// what it costs to put them on different sides:
//
// ```text
// capacity = max(1, round(scale * (area / mean_area) * -ln(clamp(p, min_probability, 1))))
// ```
//
// where `p` is [`EdgeStats::mean`] (the mean boundary probability over the
// shared face), `area` its physical face area and `mean_area` the mean face
// area over the segment's internal edges (so the integers do not depend on the
// physical unit). `-ln p` is the cost of a cut under the model "a face is a real
// boundary with probability `p`": a strong membrane (`p` near 1) is nearly free
// to cut, a face through cytoplasm (`p` near 0) is expensive, and it rises much
// faster towards small `p` than `1 - p` does, which is what keeps a cut from
// shortcutting through a weakly stained interior. Area weighting makes a large
// shared face cost more than a small one of the same probability. The floor of
// 1 keeps every RAG edge a real connection, which makes the treatment of
// connectivity below exact.
//
// Seeds and connectivity
// ----------------------
// Seed fragments are tied to their terminal with [`INFINITE_CAPACITY`], so they
// are always on their own side. The segment's subgraph can be disconnected (a
// grouping made at a coarse level, or fragments without a RAG entry): a
// connected component that holds seeds of only one group goes wholly to that
// group without a cut, and a component with **no seed at all** goes to the
// side that is **larger by voxel count** after the cut ([`NodeStats::voxels`];
// fragments the RAG has no node for count 1), ties to side A. Every component
// with seeds of both groups is cut; because all capacities are positive, each
// side of such a cut is connected to its own seeds within that side.
//
// k-way
// -----
// [`split_segment_k`] separates three or more seed groups by repeated two-way
// cuts: group 0 against the union of the rest, then group 1 against the rest of
// what remains, and so on. Each cut is exact; the sequence is a greedy
// approximation of the multiway cut (which is NP-hard for k >= 3), and its
// answer depends on the group order. Seedless components go to the largest
// final part.
//
// Scale
// -----
// The [`RagIndex`] (CSR adjacency of the whole RAG, 8 bytes per edge end) is
// built once per RAG; a split then touches only the segment's fragments and
// their incident edges.

use std::collections::HashMap;

use super::agglomerate::FastHash;
use super::rag::{EdgeStats, Rag};
use crate::error::{Error, Result};
use crate::graph_cut::{CutGraph, INFINITE_CAPACITY};

/// Node -> incident edges of a [`Rag`], built once and reused for every split.
#[derive(Debug, Clone)]
pub struct RagIndex {
    slot: HashMap<u32, u32, FastHash>,
    offsets: Vec<u64>,
    /// `(neighbour label, edge index)` per edge end.
    adjacency: Vec<(u32, u32)>,
    max_label: u32,
}

impl RagIndex {
    pub fn new(rag: &Rag) -> Result<Self> {
        if rag.edge_count() >= u32::MAX as usize {
            return Err(Error::invalid(format!(
                "segment_split: {} edges do not fit the index's u32 edge numbers",
                rag.edge_count()
            )));
        }
        let labels = rag.labels();
        let max_label = labels.last().copied().unwrap_or(0);
        let slot: HashMap<u32, u32, FastHash> = labels
            .iter()
            .enumerate()
            .map(|(i, &label)| (label, i as u32))
            .collect();
        let mut degree = vec![0u64; labels.len() + 1];
        for i in 0..rag.edge_count() {
            let ((a, b), _) = rag.edge(i);
            degree[slot[&a] as usize + 1] += 1;
            degree[slot[&b] as usize + 1] += 1;
        }
        for i in 1..degree.len() {
            degree[i] += degree[i - 1];
        }
        let offsets = degree;
        let mut fill = offsets.clone();
        let mut adjacency = vec![(0u32, 0u32); rag.edge_count() * 2];
        for i in 0..rag.edge_count() {
            let ((a, b), _) = rag.edge(i);
            for (from, to) in [(a, b), (b, a)] {
                let at = &mut fill[slot[&from] as usize];
                adjacency[*at as usize] = (to, i as u32);
                *at += 1;
            }
        }
        Ok(Self {
            slot,
            offsets,
            adjacency,
            max_label,
        })
    }

    /// `(neighbour, edge index)` of every edge at `label` (empty for an unknown label).
    pub fn neighbours(&self, label: u32) -> &[(u32, u32)] {
        match self.slot.get(&label) {
            Some(&s) => {
                let s = s as usize;
                &self.adjacency[self.offsets[s] as usize..self.offsets[s + 1] as usize]
            }
            None => &[],
        }
    }

    /// The largest label in the RAG (0 when it is empty).
    pub fn max_label(&self) -> u32 {
        self.max_label
    }

    /// Heap bytes held, for a cache's accounting.
    pub fn heap_bytes(&self) -> usize {
        self.adjacency.len() * 8 + self.offsets.len() * 8 + self.slot.len() * 16
    }
}

/// Parameters of the capacity rule (see the module header).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitOptions {
    /// Integer units per mean-area face of probability `1/e`.
    pub scale: f64,
    /// Probabilities below this are treated as this, bounding `-ln p`.
    pub min_probability: f64,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self {
            scale: 1000.0,
            min_probability: 1e-4,
        }
    }
}

/// The capacity of an edge with statistics `stats` (see the module header).
pub fn edge_capacity(stats: &EdgeStats, mean_area: f64, options: &SplitOptions) -> i64 {
    let p = stats.mean();
    let p = if p.is_finite() { p } else { 0.5 };
    let p = p.clamp(options.min_probability.max(f64::MIN_POSITIVE), 1.0);
    let area = if mean_area > 0.0 && stats.area.is_finite() {
        stats.area / mean_area
    } else {
        1.0
    };
    let value = options.scale * area * -p.ln();
    if value.is_finite() {
        (value.round() as i64).clamp(1, INFINITE_CAPACITY / (1 << 16))
    } else {
        1
    }
}

/// A partition of one segment's fragments.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentSplit {
    /// One ascending fragment list per seed group, in the order the groups were given.
    pub parts: Vec<Vec<u64>>,
    /// Sum of the capacities of the edges that separate different parts.
    pub cut_cost: i64,
    /// Number of RAG edges that separate different parts.
    pub cut_edges: usize,
    /// Fragments in components with no seed, assigned by the larger-side rule.
    pub unseeded: Vec<u64>,
}

/// Split the segment made of `fragments` into the part holding `seeds_a` and the part
/// holding `seeds_b`, by an exact minimum cut over the segment's subgraph of `rag`.
/// `index` must have been built from `rag`.
pub fn split_segment(
    rag: &Rag,
    index: &RagIndex,
    fragments: &[u64],
    seeds_a: &[u64],
    seeds_b: &[u64],
    options: &SplitOptions,
) -> Result<SegmentSplit> {
    split_segment_k(
        rag,
        index,
        fragments,
        &[seeds_a.to_vec(), seeds_b.to_vec()],
        options,
    )
}

/// [`split_segment`] for two or more seed groups (repeated two-way cuts; see the header).
pub fn split_segment_k(
    rag: &Rag,
    index: &RagIndex,
    fragments: &[u64],
    seeds: &[Vec<u64>],
    options: &SplitOptions,
) -> Result<SegmentSplit> {
    if seeds.len() < 2 {
        return Err(Error::invalid(
            "segment_split: give at least two seed groups",
        ));
    }
    let mut nodes: Vec<u64> = fragments.to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    let local: HashMap<u64, usize, FastHash> =
        nodes.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    // group of each node's seed, if any
    let mut seed_group = vec![usize::MAX; nodes.len()];
    for (group, list) in seeds.iter().enumerate() {
        if list.is_empty() {
            return Err(Error::invalid(format!(
                "segment_split: seed group {group} is empty"
            )));
        }
        for &f in list {
            let &i = local.get(&f).ok_or_else(|| {
                Error::invalid(format!(
                    "segment_split: seed fragment {f} is not in the segment"
                ))
            })?;
            if seed_group[i] != usize::MAX && seed_group[i] != group {
                return Err(Error::invalid(format!(
                    "segment_split: fragment {f} is a seed of groups {} and {group}",
                    seed_group[i]
                )));
            }
            seed_group[i] = group;
        }
    }
    // internal edges: (u, v, rag edge index), u < v local indices
    let mut edges = Vec::new();
    for (u, &f) in nodes.iter().enumerate() {
        let Ok(label) = u32::try_from(f) else {
            continue;
        };
        for &(neighbour, edge) in index.neighbours(label) {
            if let Some(&v) = local.get(&(neighbour as u64)) {
                if u < v {
                    edges.push((u, v, edge as usize));
                }
            }
        }
    }
    let mean_area = if edges.is_empty() {
        1.0
    } else {
        edges
            .iter()
            .map(|&(_, _, e)| rag.edge(e).1.area)
            .sum::<f64>()
            / edges.len() as f64
    };
    let capacity: Vec<i64> = edges
        .iter()
        .map(|&(_, _, e)| edge_capacity(rag.edge(e).1, mean_area, options))
        .collect();
    let mut adjacency = vec![Vec::new(); nodes.len()];
    for (k, &(u, v, _)) in edges.iter().enumerate() {
        adjacency[u].push((v, k));
        adjacency[v].push((u, k));
    }
    let voxels: Vec<u64> = nodes
        .iter()
        .map(|&f| {
            u32::try_from(f)
                .ok()
                .and_then(|l| rag.node(l))
                .map_or(1, |n| n.voxels.max(1))
        })
        .collect();

    // connected components
    let mut component = vec![usize::MAX; nodes.len()];
    let mut components: Vec<Vec<usize>> = Vec::new();
    for start in 0..nodes.len() {
        if component[start] != usize::MAX {
            continue;
        }
        let id = components.len();
        let mut members = vec![start];
        component[start] = id;
        let mut head = 0;
        while head < members.len() {
            let u = members[head];
            head += 1;
            for &(v, _) in &adjacency[u] {
                if component[v] == usize::MAX {
                    component[v] = id;
                    members.push(v);
                }
            }
        }
        components.push(members);
    }

    let mut part = vec![usize::MAX; nodes.len()];
    let mut unseeded = Vec::new();
    for members in &components {
        let mut groups: Vec<usize> = members
            .iter()
            .map(|&u| seed_group[u])
            .filter(|&g| g != usize::MAX)
            .collect();
        groups.sort_unstable();
        groups.dedup();
        match groups.len() {
            0 => unseeded.extend(members.iter().copied()),
            1 => members.iter().for_each(|&u| part[u] = groups[0]),
            _ => {
                // repeated two-way cuts over the groups present, in the given order
                let mut remaining: Vec<usize> = members.clone();
                for (step, &group) in groups.iter().enumerate() {
                    if step + 1 == groups.len() {
                        remaining.iter().for_each(|&u| part[u] = group);
                        break;
                    }
                    let rest = &groups[step + 1..];
                    let source = two_way_cut(
                        &remaining,
                        &adjacency,
                        &capacity,
                        |u| seed_group[u] == group,
                        |u| seed_group[u] != usize::MAX && rest.contains(&seed_group[u]),
                    )?;
                    let mut next = Vec::new();
                    for (&u, &in_source) in remaining.iter().zip(&source) {
                        if in_source {
                            part[u] = group;
                        } else {
                            next.push(u);
                        }
                    }
                    remaining = next;
                }
            }
        }
    }
    if !unseeded.is_empty() {
        let mut size = vec![0u64; seeds.len()];
        for (u, &p) in part.iter().enumerate() {
            if p != usize::MAX {
                size[p] += voxels[u];
            }
        }
        // the largest part; ties to the earliest group (side A)
        let target = (0..seeds.len())
            .max_by(|&a, &b| size[a].cmp(&size[b]).then(b.cmp(&a)))
            .expect("at least two groups");
        unseeded.iter().for_each(|&u| part[u] = target);
    }

    let mut cut_cost = 0i64;
    let mut cut_edges = 0usize;
    for (k, &(u, v, _)) in edges.iter().enumerate() {
        if part[u] != part[v] {
            cut_cost += capacity[k];
            cut_edges += 1;
        }
    }
    let mut parts = vec![Vec::new(); seeds.len()];
    for (u, &p) in part.iter().enumerate() {
        parts[p].push(nodes[u]);
    }
    let mut unseeded: Vec<u64> = unseeded.into_iter().map(|u| nodes[u]).collect();
    unseeded.sort_unstable();
    Ok(SegmentSplit {
        parts,
        cut_cost,
        cut_edges,
        unseeded,
    })
}

/// Minimum cut over the subgraph induced by `members`; returns, per member, whether it is on
/// the source side.
fn two_way_cut(
    members: &[usize],
    adjacency: &[Vec<(usize, usize)>],
    capacity: &[i64],
    is_source: impl Fn(usize) -> bool,
    is_sink: impl Fn(usize) -> bool,
) -> Result<Vec<bool>> {
    let mut position: HashMap<usize, usize, FastHash> = HashMap::default();
    for (i, &u) in members.iter().enumerate() {
        position.insert(u, i);
    }
    let arcs: usize = members.iter().map(|&u| adjacency[u].len()).sum();
    let mut graph = CutGraph::new(members.len(), arcs)?;
    for (i, &u) in members.iter().enumerate() {
        if is_source(u) {
            graph.add_terminal_weights(i, INFINITE_CAPACITY, 0)?;
        } else if is_sink(u) {
            graph.add_terminal_weights(i, 0, INFINITE_CAPACITY)?;
        }
        for &(v, k) in &adjacency[u] {
            if let Some(&j) = position.get(&v) {
                if i < j {
                    graph.add_edge(i, j, capacity[k], capacity[k])?;
                }
            }
        }
    }
    graph.max_flow();
    Ok((0..members.len()).map(|i| graph.in_source_set(i)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::rag::{rag, RagOptions};
    use ndarray::Array3;

    /// A 1 x 4 x (4 * n) strip of `n` fragments, each a 4 x 4 block, left to right
    /// (labels 1..=n); the membrane probability on the face between fragment `i` and `i + 1`
    /// is `boundary[i - 1]`, and 0.02 everywhere else.
    fn strip(boundary: &[f64]) -> Rag {
        let n = boundary.len() + 1;
        let mut labels = Array3::<u32>::zeros((1, 4, 4 * n));
        let mut evidence = Array3::<f64>::from_elem((1, 4, 4 * n), 0.02);
        for x in 0..4 * n {
            for y in 0..4 {
                labels[[0, y, x]] = (x / 4) as u32 + 1;
                if x % 4 == 3 && x / 4 < boundary.len() {
                    // a direct face samples the mean of its two voxels
                    evidence[[0, y, x]] = boundary[x / 4];
                    evidence[[0, y, x + 1]] = boundary[x / 4];
                }
            }
        }
        rag(
            labels.view(),
            evidence.view(),
            None,
            &RagOptions::new([1.0, 1.0, 1.0], 0, (0.0, 1.0)),
        )
        .unwrap()
    }

    fn solve(rag: &Rag, fragments: &[u64], a: &[u64], b: &[u64]) -> SegmentSplit {
        let index = RagIndex::new(rag).unwrap();
        split_segment(rag, &index, fragments, a, b, &SplitOptions::default()).unwrap()
    }

    #[test]
    fn the_cut_follows_the_strong_membrane() {
        let rag = strip(&[0.1, 0.05, 0.9, 0.1, 0.05]);
        let split = solve(&rag, &[1, 2, 3, 4, 5, 6], &[1], &[6]);
        assert_eq!(split.parts, vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(split.cut_edges, 1);
        assert!(split.unseeded.is_empty());
        // moving the seeds does not move the cut
        let split = solve(&rag, &[1, 2, 3, 4, 5, 6], &[2, 3], &[5]);
        assert_eq!(split.parts, vec![vec![1, 2, 3], vec![4, 5, 6]]);
    }

    #[test]
    fn seeds_are_respected_even_across_a_weak_face() {
        let rag = strip(&[0.9, 0.05, 0.05]);
        // the strong membrane is between 1 and 2, but 2 and 3 are seeded apart
        let split = solve(&rag, &[1, 2, 3, 4], &[1, 2], &[3]);
        assert_eq!(split.parts, vec![vec![1, 2], vec![3, 4]]);
    }

    #[test]
    fn capacity_rises_as_probability_falls_and_with_area() {
        let options = SplitOptions::default();
        let face = |p: f64, area: f64| EdgeStats {
            faces: 4,
            axis_faces: [0, 0, 4],
            bridged: 0,
            area,
            sum: 4.0 * p,
            sum_sq: 4.0 * p * p,
            min: p,
            max: p,
        };
        let c = |p, area| edge_capacity(&face(p, area), 4.0, &options);
        assert!(c(0.01, 4.0) > c(0.5, 4.0));
        assert!(c(0.5, 4.0) > c(0.95, 4.0));
        assert_eq!(c(1.0, 4.0), 1, "a certain membrane costs the floor");
        assert_eq!(c(0.0, 4.0), c(options.min_probability, 4.0));
        assert!(c(0.5, 8.0) > c(0.5, 4.0));
        assert_eq!(c(1.0 / std::f64::consts::E, 4.0), options.scale as i64);
    }

    #[test]
    fn a_seedless_component_goes_to_the_larger_side() {
        let rag = strip(&[0.05, 0.9, 0.05, 0.05]);
        // 1-2 | 3-4-5: side B is larger; fragment 77 has no RAG edge to the rest
        let split = solve(&rag, &[1, 2, 3, 4, 5, 77], &[1], &[5]);
        assert_eq!(split.parts, vec![vec![1, 2], vec![3, 4, 5, 77]]);
        assert_eq!(split.unseeded, vec![77]);
        // a fragment id beyond u32 is also isolated, and the result is deterministic
        let again = solve(
            &rag,
            &[77, 5, 4, 3, 2, 1, u64::from(u32::MAX) + 5],
            &[1],
            &[5],
        );
        assert_eq!(again.parts[1], vec![3, 4, 5, 77, u64::from(u32::MAX) + 5]);
        // a component holding only B seeds goes to B without a cut
        let split = solve(&rag, &[1, 2, 77], &[1], &[77]);
        assert_eq!(split.parts, vec![vec![1, 2], vec![77]]);
        assert_eq!(split.cut_edges, 0);
    }

    #[test]
    fn edges_leaving_the_segment_are_ignored() {
        let rag = strip(&[0.05, 0.05, 0.9, 0.05]);
        // fragment 5 belongs to another segment; 1..=4 is cut at the 3 | 4 membrane
        let split = solve(&rag, &[1, 2, 3, 4], &[1], &[4]);
        assert_eq!(split.parts, vec![vec![1, 2, 3], vec![4]]);
    }

    #[test]
    fn bad_seeds_are_refused() {
        let rag = strip(&[0.5]);
        let index = RagIndex::new(&rag).unwrap();
        let o = SplitOptions::default();
        assert!(split_segment(&rag, &index, &[1, 2], &[], &[2], &o).is_err());
        assert!(split_segment(&rag, &index, &[1, 2], &[1], &[1], &o).is_err());
        assert!(split_segment(&rag, &index, &[1, 2], &[1], &[3], &o).is_err());
    }

    #[test]
    fn three_groups_cut_at_both_membranes() {
        let rag = strip(&[0.05, 0.9, 0.05, 0.92, 0.05]);
        let index = RagIndex::new(&rag).unwrap();
        let split = split_segment_k(
            &rag,
            &index,
            &[1, 2, 3, 4, 5, 6],
            &[vec![1], vec![3], vec![6]],
            &SplitOptions::default(),
        )
        .unwrap();
        assert_eq!(split.parts, vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        assert_eq!(split.cut_edges, 2);
    }

    #[test]
    fn the_index_survives_the_rag_byte_format() {
        let rag = strip(&[0.2, 0.9]);
        let decoded = Rag::from_bytes(&rag.to_bytes()).unwrap();
        let index = RagIndex::new(&decoded).unwrap();
        assert_eq!(index.max_label(), 3);
        let mut around: Vec<u32> = index.neighbours(2).iter().map(|&(n, _)| n).collect();
        around.sort_unstable();
        assert_eq!(around, vec![1, 3]);
        assert!(index.neighbours(99).is_empty());
        let split = split_segment(
            &decoded,
            &index,
            &[1, 2, 3],
            &[1],
            &[3],
            &Default::default(),
        )
        .unwrap();
        assert_eq!(split.parts, vec![vec![1, 2], vec![3]]);
    }
}
