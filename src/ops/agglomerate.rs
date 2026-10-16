// SPDX-License-Identifier: MIT
//
// Original work for this crate, written from the papers cited on each function.
//
// **Graph partitioning for agglomeration: the correlation-clustering family.**
//
// Everything here works on a plain weighted graph — `n_nodes` and a list of
// `(u, v, weight)` edges over node indices `0..n_nodes` — and knows nothing about
// volumes. [`super::rag`] builds such a graph from a supervoxel volume
// ([`super::rag::Rag::graph`]); [`relabel_volume`] maps a partition back.
//
// Sign convention
// ---------------
// Weights are **signed affinities**: positive = attractive (evidence the two
// nodes belong together), negative = repulsive. The multicut energy of a
// partition is the sum of the weights of the edges it **cuts**
// ([`multicut_objective`]); lower is better, and the all-singletons partition
// has energy `sum of all weights`. [`boundary_cost`] turns a boundary
// probability into such a weight.
//
// What is here
// ------------
// | function | algorithm |
// |---|---|
// | [`mutex_watershed`] | Wolf et al. 2018/2020: Kruskal over attractive and repulsive edges with mutex constraints |
// | [`gasp`] | Bailoni et al. 2022, generalised agglomerative clustering with sum / average / max linkage; sum at threshold 0 is GAEC ([`gaec`]) |
// | [`kernighan_lin`] | greedy local search for multicut: single-node moves, two-cluster Kernighan–Lin exchanges, cluster joins |
// | [`multicut`] | [`gaec`] then [`kernighan_lin`] |
// | [`lifted_gasp`] | lifted multicut heuristic: lifted (non-adjacent) edges add to the merge priority, contraction only along local edges |
//
// None of these is exact; multicut is NP-hard. The heuristics are the ones the
// connectomics literature uses on supervoxel graphs, and the tests check them
// against brute force on small graphs.
//
// Output is always a node -> cluster id map with ids consecutive from 1 in order
// of first appearance (0 is left free to mean "background" after
// [`relabel_volume`]).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};

use ndarray::{Array3, ArrayView3, ArrayViewMut3, Axis};
use rayon::prelude::*;

use super::components::Union;
use crate::error::{Error, Result};

// ------------------------------------------------------------- hashing --

/// Multiplicative hasher for integer keys: the adjacency maps here are keyed by
/// node ids, where SipHash's DoS resistance buys nothing and costs ~3x.
#[derive(Default, Clone, Copy)]
pub(crate) struct FastHasher(u64);

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(byte as u64);
        }
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.write_u64(value as u64);
    }
}

pub(crate) type FastHash = BuildHasherDefault<FastHasher>;

// --------------------------------------------------------------- costs --

/// Probabilities are clamped to `[COST_EPS, 1 - COST_EPS]` before the logit, so
/// a cost is bounded by about `±ln(1/COST_EPS)` plus the prior term.
pub const COST_EPS: f64 = 1e-6;

/// Signed multicut weight from a boundary probability `p`:
/// `ln((1-p)/p) + ln((1-beta)/beta)`. `beta` is the boundary prior: 0.5 is
/// neutral, larger values cut more. Positive = attractive.
pub fn boundary_cost(p: f64, beta: f64) -> f64 {
    let p = p.clamp(COST_EPS, 1.0 - COST_EPS);
    let beta = beta.clamp(COST_EPS, 1.0 - COST_EPS);
    ((1.0 - p) / p).ln() + ((1.0 - beta) / beta).ln()
}

/// [`boundary_cost`] per edge, each optionally multiplied by `sizes[i]` (for
/// example the face area from [`super::rag::EdgeStats::area`]).
pub fn boundary_costs(probabilities: &[f64], sizes: Option<&[f64]>, beta: f64) -> Vec<f64> {
    if let Some(sizes) = sizes {
        assert_eq!(
            sizes.len(),
            probabilities.len(),
            "boundary_costs: one size per edge"
        );
    }
    probabilities
        .iter()
        .enumerate()
        .map(|(i, &p)| boundary_cost(p, beta) * sizes.map_or(1.0, |s| s[i]))
        .collect()
}

// ----------------------------------------------------------- utilities --

/// Compact arbitrary cluster ids to `1..=k`, in order of first appearance.
pub fn relabel_consecutive<T: Copy + Eq + std::hash::Hash>(clusters: &[T]) -> Vec<u32> {
    let mut map: HashMap<T, u32, FastHash> = HashMap::default();
    clusters
        .iter()
        .map(|c| {
            let next = map.len() as u32 + 1;
            *map.entry(*c).or_insert(next)
        })
        .collect()
}

fn roots_to_labels(sets: &mut Union, n: usize) -> Vec<u32> {
    let roots: Vec<usize> = (0..n).map(|v| sets.find(v)).collect();
    relabel_consecutive(&roots)
}

fn check_edges(n_nodes: usize, edges: &[(usize, usize, f64)], what: &str) {
    for &(u, v, _) in edges {
        assert!(
            u < n_nodes && v < n_nodes,
            "{what}: edge ({u}, {v}) out of range for {n_nodes} nodes"
        );
    }
}

/// Multicut energy: the sum of the weights of edges whose endpoints are in
/// different clusters. Lower is better.
pub fn multicut_objective(edges: &[(usize, usize, f64)], clusters: &[u32]) -> f64 {
    edges
        .iter()
        .filter(|&&(u, v, _)| clusters[u] != clusters[v])
        .map(|&(_, _, w)| w)
        .sum()
}

/// Lifted multicut energy: cut local weights plus cut lifted weights.
pub fn lifted_multicut_objective(
    local: &[(usize, usize, f64)],
    lifted: &[(usize, usize, f64)],
    clusters: &[u32],
) -> f64 {
    multicut_objective(local, clusters) + multicut_objective(lifted, clusters)
}

// ------------------------------------------------------ mutex watershed --

/// Mutex watershed (Wolf et al., ECCV 2018 / TPAMI 2020).
///
/// `attractive` and `repulsive` carry non-negative **strengths**. All edges are
/// visited in descending strength: an attractive edge joins its two clusters
/// unless a mutex (an earlier repulsive edge) separates them; a repulsive edge
/// between distinct clusters adds a mutex. No threshold, no parameters.
pub fn mutex_watershed(
    n_nodes: usize,
    attractive: &[(usize, usize, f64)],
    repulsive: &[(usize, usize, f64)],
) -> Vec<u32> {
    check_edges(n_nodes, attractive, "mutex_watershed");
    check_edges(n_nodes, repulsive, "mutex_watershed");
    // (strength, is_attractive, index); ties: attractive first, then index.
    let mut order: Vec<(f64, bool, usize)> = attractive
        .iter()
        .enumerate()
        .map(|(i, e)| (e.2, true, i))
        .chain(repulsive.iter().enumerate().map(|(i, e)| (e.2, false, i)))
        .collect();
    order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    let mut sets = Union::new(n_nodes);
    // Per root: sorted ids of repulsive edges with one end in the cluster. Two
    // clusters are mutually exclusive iff their lists share an id.
    let mut mutex: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
    let shares = |a: &[u32], b: &[u32]| {
        let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        small.iter().any(|x| large.binary_search(x).is_ok())
    };
    for (_, is_attractive, i) in order {
        let (u, v) = if is_attractive {
            (attractive[i].0, attractive[i].1)
        } else {
            (repulsive[i].0, repulsive[i].1)
        };
        let (ru, rv) = (sets.find(u), sets.find(v));
        if ru == rv {
            continue;
        }
        if is_attractive {
            if shares(&mutex[ru], &mutex[rv]) {
                continue;
            }
            sets.union(ru, rv);
            let root = sets.find(ru);
            let a = std::mem::take(&mut mutex[ru]);
            let b = std::mem::take(&mut mutex[rv]);
            mutex[root] = merge_sorted(a, b);
        } else {
            let id = i as u32;
            insert_sorted(&mut mutex[ru], id);
            insert_sorted(&mut mutex[rv], id);
        }
    }
    roots_to_labels(&mut sets, n_nodes)
}

/// [`mutex_watershed`] over one signed edge list: positive weights are
/// attractive, negative repulsive with strength `|w|`.
pub fn mutex_watershed_signed(n_nodes: usize, edges: &[(usize, usize, f64)]) -> Vec<u32> {
    let (attractive, repulsive): (Vec<_>, Vec<_>) = edges.iter().partition(|e| e.2 >= 0.0);
    let repulsive: Vec<_> = repulsive.into_iter().map(|(u, v, w)| (u, v, -w)).collect();
    mutex_watershed(n_nodes, &attractive, &repulsive)
}

fn insert_sorted(list: &mut Vec<u32>, id: u32) {
    if let Err(at) = list.binary_search(&id) {
        list.insert(at, id);
    }
}

fn merge_sorted(a: Vec<u32>, b: Vec<u32>) -> Vec<u32> {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

// ---------------------------------------------------------------- GASP --

/// How [`gasp`] scores the edge between two clusters from the original edges
/// joining them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Sum of weights: greedy additive edge contraction (GAEC).
    Sum,
    /// Size-weighted mean of weights.
    Average,
    /// Largest single weight.
    Max,
}

#[derive(Clone, Copy)]
struct Link {
    sum: f64,
    weighted: f64,
    size: f64,
    max: f64,
    local: bool,
    stamp: u64,
}

impl Link {
    fn new(weight: f64, size: f64, local: bool) -> Self {
        Self {
            sum: weight,
            weighted: weight * size,
            size,
            max: weight,
            local,
            stamp: 0,
        }
    }

    fn fold(&mut self, other: &Link) {
        self.sum += other.sum;
        self.weighted += other.weighted;
        self.size += other.size;
        self.max = self.max.max(other.max);
        self.local |= other.local;
    }

    fn priority(&self, linkage: Linkage) -> f64 {
        match linkage {
            Linkage::Sum => self.sum,
            Linkage::Average => {
                if self.size > 0.0 {
                    self.weighted / self.size
                } else {
                    self.sum
                }
            }
            Linkage::Max => self.max,
        }
    }
}

struct Candidate {
    priority: f64,
    a: usize,
    b: usize,
    stamp: u64,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| other.a.cmp(&self.a))
            .then_with(|| other.b.cmp(&self.b))
    }
}

/// The contraction engine behind [`gasp`] and [`lifted_gasp`]: a cluster graph
/// of per-node adjacency hash maps (each link stored on both ends) and a
/// max-heap of candidate merges with lazy deletion by stamp.
fn contract(
    n_nodes: usize,
    edges: impl Iterator<Item = (usize, usize, f64, f64, bool)>,
    linkage: Linkage,
    threshold: f64,
) -> Vec<u32> {
    let edges: Vec<_> = edges.collect();
    let mut degree = vec![0usize; n_nodes];
    for &(u, v, ..) in &edges {
        if u < n_nodes && v < n_nodes {
            degree[u] += 1;
            degree[v] += 1;
        }
    }
    let mut adj: Vec<HashMap<usize, Link, FastHash>> = degree
        .iter()
        .map(|&d| HashMap::with_capacity_and_hasher(d, FastHash::default()))
        .collect();
    for (u, v, w, size, local) in edges {
        assert!(
            u < n_nodes && v < n_nodes,
            "gasp: edge ({u}, {v}) out of range for {n_nodes} nodes"
        );
        if u == v {
            continue;
        }
        let link = Link::new(w, size, local);
        adj[u]
            .entry(v)
            .and_modify(|l| l.fold(&link))
            .or_insert(link);
        adj[v]
            .entry(u)
            .and_modify(|l| l.fold(&link))
            .or_insert(link);
    }
    let mut heap = BinaryHeap::new();
    for (u, links) in adj.iter().enumerate() {
        for (&v, link) in links {
            // A link at or below the threshold is never a candidate unless a
            // later fold changes it, and every fold re-pushes; so skip it.
            if u < v && link.local && link.priority(linkage) > threshold {
                heap.push(Candidate {
                    priority: link.priority(linkage),
                    a: u,
                    b: v,
                    stamp: 0,
                });
            }
        }
    }
    let mut alive = vec![true; n_nodes];
    let mut sets = Union::new(n_nodes);
    let mut stamp = 0u64;
    while let Some(top) = heap.pop() {
        let (a, b) = (top.a, top.b);
        if !alive[a] || !alive[b] {
            continue;
        }
        match adj[a].get(&b) {
            Some(link) if link.stamp == top.stamp => {}
            _ => continue,
        }
        if top.priority.is_nan() || top.priority <= threshold {
            break;
        }
        let (s, t) = if adj[a].len() >= adj[b].len() {
            (a, b)
        } else {
            (b, a)
        };
        adj[s].remove(&t);
        let absorbed = std::mem::take(&mut adj[t]);
        for (c, link) in absorbed {
            if c == s {
                continue;
            }
            adj[c].remove(&t);
            stamp += 1;
            let entry = adj[s].entry(c).or_insert(Link {
                sum: 0.0,
                weighted: 0.0,
                size: 0.0,
                max: f64::NEG_INFINITY,
                local: false,
                stamp: 0,
            });
            entry.fold(&link);
            entry.stamp = stamp;
            let merged = *entry;
            adj[c].insert(s, merged);
            if merged.local && merged.priority(linkage) > threshold {
                heap.push(Candidate {
                    priority: merged.priority(linkage),
                    a: s.min(c),
                    b: s.max(c),
                    stamp,
                });
            }
        }
        alive[t] = false;
        sets.union(s, t);
    }
    roots_to_labels(&mut sets, n_nodes)
}

/// GASP (Bailoni et al., "GASP, a generalized framework for agglomerative
/// clustering of signed graphs", ECCV 2020 / TPAMI 2022) without cannot-link
/// constraints: repeatedly contract the cluster pair of highest linkage
/// priority while it exceeds `threshold`. `sizes` (one per edge, default 1)
/// weight [`Linkage::Average`]; they do not affect the other linkages.
/// Duplicate edges are folded together; self-loops are ignored.
pub fn gasp(
    n_nodes: usize,
    edges: &[(usize, usize, f64)],
    sizes: Option<&[f64]>,
    linkage: Linkage,
    threshold: f64,
) -> Vec<u32> {
    if let Some(sizes) = sizes {
        assert_eq!(sizes.len(), edges.len(), "gasp: one size per edge");
    }
    contract(
        n_nodes,
        edges
            .iter()
            .enumerate()
            .map(|(i, &(u, v, w))| (u, v, w, sizes.map_or(1.0, |s| s[i]), true)),
        linkage,
        threshold,
    )
}

/// Greedy additive edge contraction (Keuper et al. 2015): [`gasp`] with sum
/// linkage, stopping when no positive inter-cluster sum remains.
pub fn gaec(n_nodes: usize, edges: &[(usize, usize, f64)]) -> Vec<u32> {
    gasp(n_nodes, edges, None, Linkage::Sum, 0.0)
}

/// Lifted-multicut agglomeration (the lifted GAEC of Keuper et al. 2015,
/// generalised to the [`Linkage`]s): `lifted` edges contribute to the linkage
/// between clusters exactly as local ones do, but a pair of clusters is only a
/// merge candidate while at least one **local** edge joins them. With
/// [`Linkage::Sum`] and threshold 0 this greedily decreases
/// [`lifted_multicut_objective`]. Lifted edges get size 1 for
/// [`Linkage::Average`].
pub fn lifted_gasp(
    n_nodes: usize,
    local: &[(usize, usize, f64)],
    lifted: &[(usize, usize, f64)],
    linkage: Linkage,
    threshold: f64,
) -> Vec<u32> {
    contract(
        n_nodes,
        local
            .iter()
            .map(|&(u, v, w)| (u, v, w, 1.0, true))
            .chain(lifted.iter().map(|&(u, v, w)| (u, v, w, 1.0, false))),
        linkage,
        threshold,
    )
}

// ----------------------------------------------------- multicut local search --

/// Limits for [`kernighan_lin`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KlOptions {
    /// Outer rounds (moves + exchanges + joins); stops early when a round does
    /// not improve.
    pub max_rounds: usize,
    /// Minimum energy decrease that counts as an improvement.
    pub epsilon: f64,
    /// A two-cluster pass stops after this many consecutive moves without a
    /// new best prefix. Bounds the pass to the neighbourhood of the boundary.
    pub patience: usize,
}

impl KlOptions {
    pub fn new(max_rounds: usize, epsilon: f64, patience: usize) -> Self {
        Self {
            max_rounds,
            epsilon,
            patience,
        }
    }
}

/// Local search for the multicut energy, starting from `initial` (any cluster
/// ids, one per node). Each round:
///
/// 1. **single-node moves**: every node may move to a neighbouring cluster or
///    to a new singleton if that lowers the energy;
/// 2. **two-cluster Kernighan–Lin**: for each pair of adjacent clusters, move
///    nodes between them one at a time in order of best gain (each at most
///    once, accepting temporarily worse states) and keep the best prefix. The
///    pass is seeded with the nodes on the shared boundary and grows only to
///    neighbours of moved nodes, so its cost scales with the boundary rather
///    than with the clusters;
/// 3. **joins**: GAEC on the cluster graph merges clusters whose connecting
///    weight is positive.
///
/// Every accepted step strictly lowers [`multicut_objective`], so the result is
/// never worse than `initial`. Clusters are finally split into connected
/// components (which does not change the energy) and numbered from 1.
pub fn kernighan_lin(
    n_nodes: usize,
    edges: &[(usize, usize, f64)],
    initial: &[u32],
    options: KlOptions,
) -> Vec<u32> {
    assert_eq!(
        initial.len(),
        n_nodes,
        "kernighan_lin: one cluster per node"
    );
    check_edges(n_nodes, edges, "kernighan_lin");
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_nodes];
    for &(u, v, w) in edges {
        if u != v {
            adj[u].push((v, w));
            adj[v].push((u, w));
        }
    }
    let mut lab: Vec<usize> = relabel_consecutive(initial)
        .into_iter()
        .map(|c| c as usize - 1)
        .collect();
    let mut next_id = lab.iter().copied().max().map_or(0, |m| m + 1);
    let eps = options.epsilon;
    let mut scratch: HashMap<usize, f64, FastHash> = HashMap::default();
    let mut pass = KlPass::new(n_nodes);

    for _ in 0..options.max_rounds {
        let mut improved = false;

        // 1. Single-node moves.
        for v in 0..n_nodes {
            scratch.clear();
            for &(u, w) in &adj[v] {
                *scratch.entry(lab[u]).or_insert(0.0) += w;
            }
            let own = scratch.get(&lab[v]).copied().unwrap_or(0.0);
            // Moving v from A to C changes the energy by w(v,A) - w(v,C).
            let mut best: (f64, Option<usize>) = (own, None);
            for (&c, &w) in &scratch {
                if c != lab[v] {
                    let delta = own - w;
                    if delta < best.0 || (delta == best.0 && best.1.is_some_and(|b| c < b)) {
                        best = (delta, Some(c));
                    }
                }
            }
            if best.0 < -eps {
                lab[v] = match best.1 {
                    Some(c) => c,
                    None => {
                        next_id += 1;
                        next_id - 1
                    }
                };
                improved = true;
            }
        }

        // 2. Two-cluster Kernighan–Lin exchanges, seeded per pair with the
        //    endpoints of the cut edges between them.
        let mut boundary: Vec<(usize, usize, usize)> = Vec::new();
        for &(u, v, _) in edges {
            let (a, b) = (lab[u], lab[v]);
            if a != b {
                let pair = (a.min(b), a.max(b));
                boundary.push((pair.0, pair.1, u));
                boundary.push((pair.0, pair.1, v));
            }
        }
        boundary.sort_unstable();
        boundary.dedup();
        let mut start = 0;
        while start < boundary.len() {
            let (a, b) = (boundary[start].0, boundary[start].1);
            let mut end = start;
            while end < boundary.len() && (boundary[end].0, boundary[end].1) == (a, b) {
                end += 1;
            }
            let seeds = boundary[start..end].iter().map(|s| s.2);
            if pass.run(&adj, &mut lab, a, b, seeds, eps, options.patience) {
                improved = true;
            }
            start = end;
        }

        // 3. Joins.
        let cluster_edges: Vec<(usize, usize, f64)> = edges
            .iter()
            .filter(|e| lab[e.0] != lab[e.1])
            .map(|e| (lab[e.0], lab[e.1], e.2))
            .collect();
        let joined = gaec(next_id, &cluster_edges);
        let mut present = vec![false; next_id];
        for &c in &lab {
            present[c] = true;
        }
        let mut first: HashMap<u32, usize, FastHash> = HashMap::default();
        for c in (0..next_id).filter(|&c| present[c]) {
            if *first.entry(joined[c]).or_insert(c) != c {
                improved = true;
            }
        }
        for c in lab.iter_mut() {
            *c = first[&joined[*c]];
        }

        if !improved {
            break;
        }
    }

    // Split clusters into connected components.
    let mut sets = Union::new(n_nodes);
    for &(u, v, _) in edges {
        if lab[u] == lab[v] {
            sets.union(u, v);
        }
    }
    roots_to_labels(&mut sets, n_nodes)
}

/// Scratch state for one two-cluster Kernighan–Lin pass, reused across passes
/// so a pass costs only what it touches.
struct KlPass {
    stamp: Vec<u32>,
    locked: Vec<bool>,
    touched: Vec<usize>,
}

impl KlPass {
    fn new(n: usize) -> Self {
        Self {
            stamp: vec![0; n],
            locked: vec![false; n],
            touched: Vec::new(),
        }
    }

    /// Moves nodes between clusters `a` and `b` in place in `lab`, keeps the
    /// best prefix, and returns whether it improved the energy.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        adj: &[Vec<(usize, f64)>],
        lab: &mut [usize],
        a: usize,
        b: usize,
        seeds: impl Iterator<Item = usize>,
        eps: f64,
        patience: usize,
    ) -> bool {
        // Gain of moving v to the other cluster = w(v, other) - w(v, own),
        // over neighbours in a ∪ b only (edges to other clusters stay cut).
        let gain = |v: usize, lab: &[usize]| -> f64 {
            let mine = lab[v];
            adj[v]
                .iter()
                .filter(|&&(u, _)| lab[u] == a || lab[u] == b)
                .map(|&(u, w)| if lab[u] == mine { -w } else { w })
                .sum()
        };
        let mut heap = BinaryHeap::new();
        for v in seeds {
            if (lab[v] == a || lab[v] == b) && self.stamp[v] == 0 {
                self.stamp[v] = 1;
                self.touched.push(v);
                heap.push(Candidate {
                    priority: gain(v, lab),
                    a: v,
                    b: 0,
                    stamp: 1,
                });
            }
        }
        let mut moved: Vec<usize> = Vec::new();
        let (mut total, mut best, mut best_len) = (0.0f64, 0.0f64, 0usize);
        while let Some(top) = heap.pop() {
            let v = top.a;
            if self.locked[v] || self.stamp[v] as u64 != top.stamp {
                continue;
            }
            self.locked[v] = true;
            lab[v] = if lab[v] == a { b } else { a };
            moved.push(v);
            total += top.priority;
            if total > best + eps {
                best = total;
                best_len = moved.len();
            } else if moved.len() - best_len >= patience {
                break;
            }
            for &(u, _) in &adj[v] {
                if (lab[u] == a || lab[u] == b) && !self.locked[u] {
                    if self.stamp[u] == 0 {
                        self.touched.push(u);
                    }
                    self.stamp[u] += 1;
                    heap.push(Candidate {
                        priority: gain(u, lab),
                        a: u,
                        b: 0,
                        stamp: self.stamp[u] as u64,
                    });
                }
            }
        }
        // Revert everything past the best prefix.
        for &v in &moved[best_len..] {
            lab[v] = if lab[v] == a { b } else { a };
        }
        for &v in &self.touched {
            self.stamp[v] = 0;
            self.locked[v] = false;
        }
        self.touched.clear();
        best_len > 0
    }
}

/// Multicut by [`gaec`] followed by [`kernighan_lin`].
pub fn multicut(n_nodes: usize, edges: &[(usize, usize, f64)], options: KlOptions) -> Vec<u32> {
    let initial = gaec(n_nodes, edges);
    kernighan_lin(n_nodes, edges, &initial, options)
}

// ------------------------------------------------------------- relabel --

/// A supervoxel -> cluster lookup, built once and applied to many blocks.
///
/// Dense (a `Vec<u32>` indexed by label) when the largest label is not much
/// larger than the number of entries, otherwise a sorted pair list searched by
/// bisection. Label 0, and any label not in the table, maps to 0.
#[derive(Debug, Clone, PartialEq)]
pub enum LabelLookup {
    Dense(Vec<u32>),
    Sorted(Vec<(u32, u32)>),
}

impl LabelLookup {
    /// `labels[n] -> clusters[n]`, e.g. [`super::rag::RagGraph::labels`] and a
    /// partition. Refuses unequal lengths and a label mapped twice to
    /// different clusters.
    pub fn new(labels: &[u32], clusters: &[u32]) -> Result<Self> {
        if labels.len() != clusters.len() {
            return Err(Error::invalid(format!(
                "relabel_volume: {} labels but {} clusters",
                labels.len(),
                clusters.len()
            )));
        }
        Self::from_pairs(
            labels
                .iter()
                .copied()
                .zip(clusters.iter().copied())
                .collect(),
        )
    }

    /// From `(label, cluster)` pairs in any order.
    pub fn from_pairs(mut pairs: Vec<(u32, u32)>) -> Result<Self> {
        pairs.sort_unstable();
        pairs.dedup();
        if let Some(w) = pairs.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(Error::invalid(format!(
                "relabel_volume: label {} is mapped to both {} and {}",
                w[0].0, w[0].1, w[1].1
            )));
        }
        let max = pairs.last().map_or(0, |p| p.0) as usize;
        if max <= pairs.len() * 8 + (1 << 20) {
            let mut table = vec![0u32; max + 1];
            for &(l, c) in &pairs {
                table[l as usize] = c;
            }
            table[0] = 0;
            Ok(LabelLookup::Dense(table))
        } else {
            Ok(LabelLookup::Sorted(pairs))
        }
    }

    /// The cluster of `label` (0 for 0 and for a label not in the table).
    #[inline]
    pub fn get(&self, label: u32) -> u32 {
        match self {
            LabelLookup::Dense(table) => table.get(label as usize).copied().unwrap_or(0),
            LabelLookup::Sorted(pairs) => {
                if label == 0 {
                    return 0;
                }
                pairs
                    .binary_search_by_key(&label, |p| p.0)
                    .map_or(0, |i| pairs[i].1)
            }
        }
    }

    /// Map `volume` into `out` (same shape), in parallel over z-planes.
    pub fn apply_into(
        &self,
        volume: ArrayView3<'_, u32>,
        mut out: ArrayViewMut3<'_, u32>,
    ) -> Result<()> {
        if volume.shape() != out.shape() {
            return Err(Error::invalid(format!(
                "relabel_volume: output shape {:?} differs from input {:?}",
                out.shape(),
                volume.shape()
            )));
        }
        let mut planes: Vec<_> = out.axis_iter_mut(Axis(0)).collect();
        planes.par_iter_mut().enumerate().for_each(|(z, plane)| {
            let source = volume.index_axis(Axis(0), z);
            plane.zip_mut_with(&source, |o, &l| *o = self.get(l));
        });
        Ok(())
    }
}

/// Map each voxel's supervoxel label through `labels[n] -> clusters[n]` (for
/// example [`super::rag::RagGraph::labels`] and a partition), in parallel over
/// z-planes. Label 0, and any label not in `labels`, maps to 0. See
/// [`LabelLookup`] to build the table once for many blocks.
pub fn relabel_volume_into(
    volume: ArrayView3<'_, u32>,
    labels: &[u32],
    clusters: &[u32],
    out: ArrayViewMut3<'_, u32>,
) -> Result<()> {
    LabelLookup::new(labels, clusters)?.apply_into(volume, out)
}

/// [`relabel_volume_into`] into a new array.
pub fn relabel_volume(
    volume: ArrayView3<'_, u32>,
    labels: &[u32],
    clusters: &[u32],
) -> Result<Array3<u32>> {
    let shape = volume.raw_dim();
    let mut out = Array3::zeros(shape);
    relabel_volume_into(volume, labels, clusters, out.view_mut())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn random_graph(rng: &mut Rng, n: usize, density: f64) -> Vec<(usize, usize, f64)> {
        let mut edges = Vec::new();
        for u in 0..n {
            for v in u + 1..n {
                if rng.unit() < density {
                    edges.push((u, v, rng.unit() * 2.0 - 1.0));
                }
            }
        }
        edges
    }

    /// Minimum multicut energy over all set partitions (restricted growth
    /// strings).
    fn brute_force(n: usize, edges: &[(usize, usize, f64)]) -> f64 {
        let mut best = f64::INFINITY;
        let mut rgs = vec![0u32; n];
        fn rec(
            i: usize,
            max: u32,
            rgs: &mut Vec<u32>,
            edges: &[(usize, usize, f64)],
            best: &mut f64,
        ) {
            if i == rgs.len() {
                *best = best.min(multicut_objective(edges, rgs));
                return;
            }
            for c in 0..=max + 1 {
                rgs[i] = c;
                rec(i + 1, max.max(c), rgs, edges, best);
            }
        }
        if n == 0 {
            return 0.0;
        }
        rgs[0] = 0;
        rec(1, 0, &mut rgs, edges, &mut best);
        best
    }

    fn same_partition(a: &[u32], b: &[u32]) -> bool {
        relabel_consecutive(a) == relabel_consecutive(b)
    }

    #[test]
    fn costs() {
        assert!(boundary_cost(0.5, 0.5).abs() < 1e-12);
        assert!(boundary_cost(0.1, 0.5) > 0.0);
        assert!(boundary_cost(0.9, 0.5) < 0.0);
        assert!(boundary_cost(0.5, 0.7) < 0.0);
        assert!(boundary_cost(0.0, 0.5).is_finite());
        let c = boundary_costs(&[0.1, 0.9], Some(&[2.0, 3.0]), 0.5);
        assert!((c[0] - 2.0 * (9.0f64).ln()).abs() < 1e-12);
        assert!((c[1] + 3.0 * (9.0f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn mutex_watershed_hand_example() {
        let attractive = [(0, 1, 0.9), (1, 2, 0.5), (2, 3, 0.8)];
        // Strong repulsion 0-3 is seen before the weak 1-2 attraction.
        let labels = mutex_watershed(4, &attractive, &[(0, 3, 0.7)]);
        assert_eq!(labels, vec![1, 1, 2, 2]);
        // Weak repulsion comes too late to separate anything.
        let labels = mutex_watershed(4, &attractive, &[(0, 3, 0.4)]);
        assert_eq!(labels, vec![1, 1, 1, 1]);
        // Signed form; node 4 isolated.
        let labels =
            mutex_watershed_signed(5, &[(0, 1, 0.9), (1, 2, 0.5), (2, 3, 0.8), (0, 3, -0.7)]);
        assert_eq!(labels, vec![1, 1, 2, 2, 3]);
    }

    #[test]
    fn gaec_and_linkages() {
        // Triangle 0-1-2 with a strong repulsion that sum linkage outweighs.
        let edges = [(0, 1, 2.0), (1, 2, 2.0), (0, 2, -1.0), (2, 3, -0.5)];
        assert_eq!(gaec(4, &edges), vec![1, 1, 1, 2]);
        // Max linkage ignores the repulsion inside the triangle too.
        assert_eq!(gasp(4, &edges, None, Linkage::Max, 0.0), vec![1, 1, 1, 2]);
        // Average: after 0-1 merges, link to 2 is (2-1)/2 = 0.5 > 0.
        assert_eq!(
            gasp(4, &edges, None, Linkage::Average, 0.0),
            vec![1, 1, 1, 2]
        );
        // A threshold of 0.6 stops average before the third merge.
        let a = gasp(4, &edges, None, Linkage::Average, 0.6);
        assert_eq!(a.iter().collect::<std::collections::HashSet<_>>().len(), 3);
        // Sizes weight the average: a big repulsion wins.
        let a = gasp(
            4,
            &edges,
            Some(&[1.0, 1.0, 10.0, 1.0]),
            Linkage::Average,
            0.0,
        );
        assert_ne!(a[0], a[2]);
    }

    #[test]
    fn lifted_edges() {
        let local = [(0, 1, 1.0), (1, 2, 1.0)];
        assert_eq!(
            lifted_gasp(3, &local, &[], Linkage::Sum, 0.0),
            vec![1, 1, 1]
        );
        let l = lifted_gasp(3, &local, &[(0, 2, -5.0)], Linkage::Sum, 0.0);
        assert_ne!(l[0], l[2]);
        // Attractive lifted evidence carries a merge across a weak local edge.
        let local = [(0, 1, 1.0), (1, 2, -0.5)];
        assert_eq!(
            lifted_gasp(3, &local, &[], Linkage::Sum, 0.0),
            vec![1, 1, 2]
        );
        assert_eq!(
            lifted_gasp(3, &local, &[(0, 2, 2.0)], Linkage::Sum, 0.0),
            vec![1, 1, 1]
        );
        // A lifted edge alone never contracts.
        assert_eq!(
            lifted_gasp(2, &[], &[(0, 1, 9.0)], Linkage::Sum, 0.0),
            vec![1, 2]
        );
        let obj = lifted_multicut_objective(&local, &[(0, 2, 2.0)], &[1, 1, 2]);
        assert!((obj - 1.5).abs() < 1e-12);
    }

    #[test]
    fn multicut_against_brute_force() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let opts = KlOptions::new(100, 1e-12, 50);
        let (mut cases, mut optimal) = (0, 0);
        for trial in 0..200 {
            let n = 5 + trial % 4;
            let edges = random_graph(&mut rng, n, 0.6);
            let g = gaec(n, &edges);
            let m = kernighan_lin(n, &edges, &g, opts);
            let (eg, em, eo) = (
                multicut_objective(&edges, &g),
                multicut_objective(&edges, &m),
                brute_force(n, &edges),
            );
            assert!(em <= eg + 1e-12, "KL worse than GAEC: {em} > {eg}");
            assert!(em >= eo - 1e-12);
            cases += 1;
            if em <= eo + 1e-9 {
                optimal += 1;
            }
        }
        eprintln!("GAEC+KL optimal on {optimal}/{cases}");
        assert!(
            optimal * 10 >= cases * 9,
            "optimal on only {optimal}/{cases}"
        );
        // KL from singletons also never gets worse.
        let edges = random_graph(&mut rng, 8, 0.7);
        let start: Vec<u32> = (0..8).collect();
        let m = kernighan_lin(8, &edges, &start, opts);
        assert!(multicut_objective(&edges, &m) <= multicut_objective(&edges, &start) + 1e-12);
    }

    #[test]
    fn kl_fixes_a_bad_start() {
        // Two attractive triangles joined by a repulsive edge; start with
        // nodes 2 and 3 swapped.
        let edges = [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 2, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (3, 5, 1.0),
            (2, 3, -1.0),
        ];
        let m = kernighan_lin(
            6,
            &edges,
            &[1, 1, 2, 1, 2, 2],
            KlOptions::new(10, 1e-12, 50),
        );
        assert!(same_partition(&m, &[1, 1, 1, 2, 2, 2]));
        assert!(same_partition(
            &multicut(6, &edges, KlOptions::new(10, 1e-12, 50)),
            &[1, 1, 1, 2, 2, 2]
        ));
    }

    #[test]
    fn relabel() {
        let volume = Array3::from_shape_vec((1, 2, 3), vec![0, 5, 7, 9, 5, 42]).unwrap();
        let out = relabel_volume(volume.view(), &[5, 7, 9], &[1, 2, 1]).unwrap();
        assert_eq!(out.into_raw_vec_and_offset().0, vec![0, 1, 2, 1, 1, 0]);
        // Sparse labels take the binary-search path.
        let volume =
            Array3::from_shape_vec((2, 1, 2), vec![4_000_000_000, 3, 0, 4_000_000_000]).unwrap();
        let out = relabel_volume(volume.view(), &[3, 4_000_000_000], &[7, 8]).unwrap();
        assert_eq!(out.into_raw_vec_and_offset().0, vec![8, 7, 0, 8]);
        assert!(relabel_volume(volume.view(), &[3], &[]).is_err());
        assert_eq!(relabel_consecutive(&[9, 3, 9, 4]), vec![1, 2, 1, 3]);
    }

    /// `cargo test --release --lib ops::agglomerate::tests::timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn timing() {
        // ~1e6 edges: a 3D grid graph of 70^3 nodes.
        let side = 70usize;
        let n = side * side * side;
        let mut rng = Rng(99);
        let mut edges = Vec::new();
        let id = |z: usize, y: usize, x: usize| (z * side + y) * side + x;
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    if x + 1 < side {
                        edges.push((id(z, y, x), id(z, y, x + 1), rng.unit() * 2.0 - 0.9));
                    }
                    if y + 1 < side {
                        edges.push((id(z, y, x), id(z, y + 1, x), rng.unit() * 2.0 - 0.9));
                    }
                    if z + 1 < side {
                        edges.push((id(z, y, x), id(z + 1, y, x), rng.unit() * 2.0 - 0.9));
                    }
                }
            }
        }
        let t = std::time::Instant::now();
        let g = gaec(n, &edges);
        eprintln!(
            "gaec: {} edges in {:.2?}, energy {:.1}",
            edges.len(),
            t.elapsed(),
            multicut_objective(&edges, &g)
        );
        let t = std::time::Instant::now();
        let m = kernighan_lin(n, &edges, &g, KlOptions::new(5, 1e-9, 50));
        eprintln!(
            "kl: {:.2?}, energy {:.1}",
            t.elapsed(),
            multicut_objective(&edges, &m)
        );
        let t = std::time::Instant::now();
        let _ = mutex_watershed_signed(n, &edges);
        eprintln!("mws: {:.2?}", t.elapsed());
    }
}
