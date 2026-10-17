// SPDX-License-Identifier: MIT
//
// Written for this crate from the published algorithm. Its structure and
// field names (`tr_cap`, `r_cap`, orphans, timestamps) follow the paper and the
// authors' reference implementation, `maxflow-v3`, whose licence (GPL) is not
// MIT; no source was copied from it, but anyone auditing provenance should
// know the resemblance is deliberate.
//
// **Exact minimum s-t cut on a graph held in memory.**
//
// The Boykov–Kolmogorov augmenting-path algorithm ("An experimental comparison
// of min-cut/max-flow algorithms for energy minimization in vision", IEEE PAMI
// 26(9), 2004). Two search trees grow from the source and the sink; when they
// touch, flow is pushed along the path, the saturated arcs orphan parts of the
// trees, and the orphans are re-adopted or freed instead of rebuilding the
// trees. Distances and timestamps pick short parents during adoption, as in the
// paper. On the grid-like graphs of vision problems it is much faster in
// practice than generic augmenting-path or push-relabel solvers, despite a
// worse worst case.
//
// What it is for here: energies of the form "a cost per node per label plus
// pairwise or hard constraints between nodes" that are representable as a cut
// — in particular optimal surfaces (one height per column, with smoothness and
// ordering limits; Li, Wu, Chen, Sonka, PAMI 2006), which is what first needed
// it. This module is the resident solver only; it knows nothing of blocks.
// Callers that run under the planner build the (small) graph inside a barrier's
// `reduce` or from a pooled reduction, never from a whole-resolution volume.
//
// Representation: integer capacities (`i64`, so costs must be scaled to
// integers by the caller), arcs stored in pairs so the reverse of arc `a` is
// `a ^ 1`, adjacency as singly linked lists through `first` / `next`. Terminal
// capacities are folded into one signed residual per node (`tr_cap`: positive
// = residual from the source, negative = residual to the sink); the part both
// terminal arcs share is cut whatever the partition, so it is added to the flow
// up front.
//
// Hard constraints use [`INFINITE_CAPACITY`]. It is large but far from
// `i64::MAX`, so sums of a few thousand of them cannot overflow; a cut that
// would cross one is never minimal while any finite cut exists.
//
// Tests compare the flow value and the reported partition with an exhaustive
// search over every partition of small random graphs.

use std::collections::VecDeque;

use crate::error::{Error, Result};

const NONE: u32 = u32::MAX;
const TERMINAL: u32 = u32::MAX - 1;
const ORPHAN: u32 = u32::MAX - 2;
const INFINITE_DISTANCE: u32 = u32::MAX;

/// Capacity for arcs that must never be cut (hard constraints).
pub const INFINITE_CAPACITY: i64 = 1 << 50;

/// A directed graph with source and sink terminals, solved for its minimum cut.
///
/// Build it with [`Self::add_terminal_weights`] and [`Self::add_edge`], call
/// [`Self::max_flow`] once, then read the partition with
/// [`Self::in_source_set`].
///
/// ```
/// use blockflow::graph_cut::CutGraph;
/// let mut g = CutGraph::new(2, 2).unwrap();
/// g.add_terminal_weights(0, 5, 1).unwrap(); // cutting 0 from the source costs 5
/// g.add_terminal_weights(1, 1, 5).unwrap(); // cutting 1 from the sink costs 5
/// g.add_edge(0, 1, 3, 0).unwrap();          // 0 -> 1 costs 3 to cut
/// assert_eq!(g.max_flow(), 1 + 1 + 3);
/// assert!(g.in_source_set(0) && !g.in_source_set(1));
/// ```
pub struct CutGraph {
    // per node
    first: Vec<u32>,
    parent: Vec<u32>,
    tr_cap: Vec<i64>,
    is_sink: Vec<bool>,
    ts: Vec<u32>,
    dist: Vec<u32>,
    active: Vec<bool>,
    // per arc
    head: Vec<u32>,
    next: Vec<u32>,
    r_cap: Vec<i64>,
    flow: i64,
    queue: VecDeque<u32>,
    orphans: VecDeque<u32>,
    time: u32,
    solved: bool,
}

impl CutGraph {
    /// A graph of `nodes` nodes; `arcs_hint` reserves room for that many arcs
    /// (two per [`Self::add_edge`]).
    pub fn new(nodes: usize, arcs_hint: usize) -> Result<Self> {
        if nodes >= ORPHAN as usize {
            return Err(Error::invalid(format!(
                "a cut graph of {nodes} nodes does not fit the u32 node indices; split the \
                 problem or solve it at a coarser resolution"
            )));
        }
        Ok(Self {
            first: vec![NONE; nodes],
            parent: vec![NONE; nodes],
            tr_cap: vec![0; nodes],
            is_sink: vec![false; nodes],
            ts: vec![0; nodes],
            dist: vec![0; nodes],
            active: vec![false; nodes],
            head: Vec::with_capacity(arcs_hint),
            next: Vec::with_capacity(arcs_hint),
            r_cap: Vec::with_capacity(arcs_hint),
            flow: 0,
            queue: VecDeque::new(),
            orphans: VecDeque::new(),
            time: 0,
            solved: false,
        })
    }

    pub fn nodes(&self) -> usize {
        self.first.len()
    }

    /// Number of arcs added so far (two per edge).
    pub fn arcs(&self) -> usize {
        self.head.len()
    }

    fn check_node(&self, i: usize, what: &str) -> Result<()> {
        if i >= self.nodes() {
            return Err(Error::invalid(format!(
                "{what}: node {i} is outside a cut graph of {} nodes",
                self.nodes()
            )));
        }
        if self.solved {
            return Err(Error::invalid(format!(
                "{what}: the graph was already solved; build a new one"
            )));
        }
        Ok(())
    }

    /// Add `source -> i` with capacity `from_source` (paid when `i` ends on
    /// the sink side) and `i -> sink` with `to_sink` (paid when `i` ends on the
    /// source side). Repeated calls accumulate.
    pub fn add_terminal_weights(&mut self, i: usize, from_source: i64, to_sink: i64) -> Result<()> {
        self.check_node(i, "add_terminal_weights")?;
        if from_source < 0 || to_sink < 0 {
            return Err(Error::invalid(format!(
                "terminal capacities must be non-negative, got {from_source} and {to_sink}"
            )));
        }
        let delta = self.tr_cap[i];
        let (s, t) = if delta > 0 {
            (from_source + delta, to_sink)
        } else {
            (from_source, to_sink - delta)
        };
        // only the difference decides the cut; the shared part is always paid
        self.flow += s.min(t);
        self.tr_cap[i] = s - t;
        Ok(())
    }

    /// Add `i -> j` with capacity `capacity` (paid when `i` is on the source
    /// side and `j` on the sink side) and `j -> i` with `reverse_capacity`.
    pub fn add_edge(
        &mut self,
        i: usize,
        j: usize,
        capacity: i64,
        reverse_capacity: i64,
    ) -> Result<()> {
        self.check_node(i, "add_edge")?;
        self.check_node(j, "add_edge")?;
        if i == j {
            return Err(Error::invalid(format!(
                "add_edge: a self-loop on node {i} never crosses a cut"
            )));
        }
        if capacity < 0 || reverse_capacity < 0 {
            return Err(Error::invalid(format!(
                "edge capacities must be non-negative, got {capacity} and {reverse_capacity}"
            )));
        }
        let a = u32::try_from(self.head.len())
            .ok()
            .filter(|&a| a < ORPHAN - 1)
            .ok_or_else(|| {
                Error::invalid("too many arcs for the u32 arc indices of a cut graph")
            })?;
        self.head.push(j as u32);
        self.next.push(self.first[i]);
        self.r_cap.push(capacity);
        self.first[i] = a;
        self.head.push(i as u32);
        self.next.push(self.first[j]);
        self.r_cap.push(reverse_capacity);
        self.first[j] = a + 1;
        Ok(())
    }

    #[inline]
    fn set_active(&mut self, i: u32) {
        if !self.active[i as usize] {
            self.active[i as usize] = true;
            self.queue.push_back(i);
        }
    }

    fn next_active(&mut self) -> Option<u32> {
        while let Some(i) = self.queue.pop_front() {
            self.active[i as usize] = false;
            if self.parent[i as usize] != NONE {
                return Some(i);
            }
        }
        None
    }

    #[inline]
    fn set_orphan_front(&mut self, i: u32) {
        self.parent[i as usize] = ORPHAN;
        self.orphans.push_front(i);
    }

    #[inline]
    fn set_orphan_rear(&mut self, i: u32) {
        self.parent[i as usize] = ORPHAN;
        self.orphans.push_back(i);
    }

    /// Solve. Returns the value of the maximum flow, which equals the cost of
    /// the minimum cut. Call once; the graph cannot be extended afterwards.
    pub fn max_flow(&mut self) -> i64 {
        self.solved = true;
        let n = self.nodes();
        for i in 0..n {
            let c = self.tr_cap[i];
            if c != 0 {
                self.is_sink[i] = c < 0;
                self.parent[i] = TERMINAL;
                self.ts[i] = 0;
                self.dist[i] = 1;
                self.set_active(i as u32);
            } else {
                self.parent[i] = NONE;
            }
        }
        let mut current: Option<u32> = None;
        loop {
            let i = match current {
                Some(i) if self.parent[i as usize] != NONE => i,
                _ => match self.next_active() {
                    Some(i) => i,
                    None => break,
                },
            };
            current = None;
            let iu = i as usize;
            // grow
            let mut found = NONE;
            if !self.is_sink[iu] {
                let mut a = self.first[iu];
                while a != NONE {
                    let au = a as usize;
                    if self.r_cap[au] > 0 {
                        let j = self.head[au] as usize;
                        if self.parent[j] == NONE {
                            self.is_sink[j] = false;
                            self.parent[j] = a ^ 1;
                            self.ts[j] = self.ts[iu];
                            self.dist[j] = self.dist[iu] + 1;
                            self.set_active(j as u32);
                        } else if self.is_sink[j] {
                            found = a;
                            break;
                        } else if self.ts[j] <= self.ts[iu] && self.dist[j] > self.dist[iu] {
                            self.parent[j] = a ^ 1;
                            self.ts[j] = self.ts[iu];
                            self.dist[j] = self.dist[iu] + 1;
                        }
                    }
                    a = self.next[au];
                }
            } else {
                let mut a = self.first[iu];
                while a != NONE {
                    let au = a as usize;
                    if self.r_cap[(a ^ 1) as usize] > 0 {
                        let j = self.head[au] as usize;
                        if self.parent[j] == NONE {
                            self.is_sink[j] = true;
                            self.parent[j] = a ^ 1;
                            self.ts[j] = self.ts[iu];
                            self.dist[j] = self.dist[iu] + 1;
                            self.set_active(j as u32);
                        } else if !self.is_sink[j] {
                            found = a ^ 1;
                            break;
                        } else if self.ts[j] <= self.ts[iu] && self.dist[j] > self.dist[iu] {
                            self.parent[j] = a ^ 1;
                            self.ts[j] = self.ts[iu];
                            self.dist[j] = self.dist[iu] + 1;
                        }
                    }
                    a = self.next[au];
                }
            }
            self.time = self.time.wrapping_add(1);
            if found != NONE {
                current = Some(i);
                self.augment(found);
                while let Some(o) = self.orphans.pop_front() {
                    if self.is_sink[o as usize] {
                        self.process_sink_orphan(o);
                    } else {
                        self.process_source_orphan(o);
                    }
                }
            }
        }
        self.flow
    }

    /// `middle` goes from a source-tree node to a sink-tree node.
    fn augment(&mut self, middle: u32) {
        let mut bottleneck = self.r_cap[middle as usize];
        // source side
        let mut i = self.head[(middle ^ 1) as usize] as usize;
        loop {
            let a = self.parent[i];
            if a == TERMINAL {
                break;
            }
            bottleneck = bottleneck.min(self.r_cap[(a ^ 1) as usize]);
            i = self.head[a as usize] as usize;
        }
        bottleneck = bottleneck.min(self.tr_cap[i]);
        // sink side
        let mut i = self.head[middle as usize] as usize;
        loop {
            let a = self.parent[i];
            if a == TERMINAL {
                break;
            }
            bottleneck = bottleneck.min(self.r_cap[a as usize]);
            i = self.head[a as usize] as usize;
        }
        bottleneck = bottleneck.min(-self.tr_cap[i]);

        self.r_cap[(middle ^ 1) as usize] += bottleneck;
        self.r_cap[middle as usize] -= bottleneck;
        let mut i = self.head[(middle ^ 1) as usize] as usize;
        loop {
            let a = self.parent[i];
            if a == TERMINAL {
                break;
            }
            self.r_cap[a as usize] += bottleneck;
            self.r_cap[(a ^ 1) as usize] -= bottleneck;
            if self.r_cap[(a ^ 1) as usize] == 0 {
                self.set_orphan_front(i as u32);
            }
            i = self.head[a as usize] as usize;
        }
        self.tr_cap[i] -= bottleneck;
        if self.tr_cap[i] == 0 {
            self.set_orphan_front(i as u32);
        }
        let mut i = self.head[middle as usize] as usize;
        loop {
            let a = self.parent[i];
            if a == TERMINAL {
                break;
            }
            self.r_cap[(a ^ 1) as usize] += bottleneck;
            self.r_cap[a as usize] -= bottleneck;
            if self.r_cap[a as usize] == 0 {
                self.set_orphan_front(i as u32);
            }
            i = self.head[a as usize] as usize;
        }
        self.tr_cap[i] += bottleneck;
        if self.tr_cap[i] == 0 {
            self.set_orphan_front(i as u32);
        }
        self.flow += bottleneck;
    }

    /// Distance from `j` to its terminal, or `INFINITE_DISTANCE` if the path hits an orphan.
    fn origin_distance(&mut self, j: usize) -> u32 {
        let mut d: u32 = 0;
        let mut jj = j;
        loop {
            if self.ts[jj] == self.time {
                return d.saturating_add(self.dist[jj]);
            }
            let a = self.parent[jj];
            d += 1;
            if a == TERMINAL {
                self.ts[jj] = self.time;
                self.dist[jj] = 1;
                return d;
            }
            if a == ORPHAN {
                return INFINITE_DISTANCE;
            }
            jj = self.head[a as usize] as usize;
        }
    }

    fn mark_path(&mut self, j: usize, mut d: u32) {
        let mut jj = j;
        while self.ts[jj] != self.time {
            self.ts[jj] = self.time;
            self.dist[jj] = d;
            d -= 1;
            jj = self.head[self.parent[jj] as usize] as usize;
        }
    }

    fn process_orphan(&mut self, i: u32, sink: bool) {
        let iu = i as usize;
        let mut best = NONE;
        let mut d_min = INFINITE_DISTANCE;
        let mut a0 = self.first[iu];
        while a0 != NONE {
            let residual = if sink {
                self.r_cap[a0 as usize]
            } else {
                self.r_cap[(a0 ^ 1) as usize]
            };
            if residual > 0 {
                let j = self.head[a0 as usize] as usize;
                if self.is_sink[j] == sink && self.parent[j] != NONE {
                    let d = self.origin_distance(j);
                    if d < INFINITE_DISTANCE {
                        if d < d_min {
                            best = a0;
                            d_min = d;
                        }
                        self.mark_path(j, d);
                    }
                }
            }
            a0 = self.next[a0 as usize];
        }
        self.parent[iu] = best;
        if best != NONE {
            self.ts[iu] = self.time;
            self.dist[iu] = d_min + 1;
            return;
        }
        // i becomes free: wake neighbours that could adopt it, orphan its children
        let mut a0 = self.first[iu];
        while a0 != NONE {
            let j = self.head[a0 as usize] as usize;
            let a = self.parent[j];
            if self.is_sink[j] == sink && a != NONE {
                let residual = if sink {
                    self.r_cap[a0 as usize]
                } else {
                    self.r_cap[(a0 ^ 1) as usize]
                };
                if residual > 0 {
                    self.set_active(j as u32);
                }
                if a != TERMINAL && a != ORPHAN && self.head[a as usize] as usize == iu {
                    self.set_orphan_rear(j as u32);
                }
            }
            a0 = self.next[a0 as usize];
        }
    }

    fn process_source_orphan(&mut self, i: u32) {
        self.process_orphan(i, false)
    }

    fn process_sink_orphan(&mut self, i: u32) {
        self.process_orphan(i, true)
    }

    /// After [`Self::max_flow`]: is node `i` on the source side of the
    /// minimum cut, i.e. reachable from the source in the residual graph?
    /// Nodes on neither tree are reported on the sink side.
    pub fn in_source_set(&self, i: usize) -> bool {
        self.parent[i] != NONE && !self.is_sink[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Terminal = (i64, i64);
    type Edge = (usize, usize, i64, i64);

    /// Cost of the cut that puts the nodes in `source` on the source side.
    fn cut_cost(terminals: &[Terminal], edges: &[Edge], source: impl Fn(usize) -> bool) -> i64 {
        let mut cost = 0;
        for (i, &(from_source, to_sink)) in terminals.iter().enumerate() {
            cost += if source(i) { to_sink } else { from_source };
        }
        for &(i, j, forward, reverse) in edges {
            if source(i) && !source(j) {
                cost += forward;
            }
            if source(j) && !source(i) {
                cost += reverse;
            }
        }
        cost
    }

    fn brute_force(terminals: &[Terminal], edges: &[Edge]) -> i64 {
        (0..1u32 << terminals.len())
            .map(|mask| cut_cost(terminals, edges, |i| mask & (1 << i) != 0))
            .min()
            .unwrap()
    }

    fn solve(terminals: &[Terminal], edges: &[Edge]) -> (i64, CutGraph) {
        let mut g = CutGraph::new(terminals.len(), edges.len() * 2).unwrap();
        for (i, &(a, b)) in terminals.iter().enumerate() {
            g.add_terminal_weights(i, a, b).unwrap();
        }
        for &(i, j, a, b) in edges {
            g.add_edge(i, j, a, b).unwrap();
        }
        (g.max_flow(), g)
    }

    #[test]
    fn matches_exhaustive_search_on_random_graphs() {
        let mut seed = 12345u64;
        let mut random = |m: i64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as i64) % m
        };
        for _ in 0..400 {
            let n = 2 + random(9) as usize;
            let terminals: Vec<Terminal> = (0..n).map(|_| (random(10), random(10))).collect();
            let mut edges = Vec::new();
            for _ in 0..random(20) {
                let i = random(n as i64) as usize;
                let j = random(n as i64) as usize;
                if i != j {
                    edges.push((i, j, random(8), random(8)));
                }
            }
            let (flow, g) = solve(&terminals, &edges);
            assert_eq!(flow, brute_force(&terminals, &edges));
            // the reported partition achieves the flow value
            assert_eq!(cut_cost(&terminals, &edges, |i| g.in_source_set(i)), flow);
        }
    }

    #[test]
    fn hard_constraints_are_never_cut() {
        // 0 prefers the sink, 1 prefers the source, but "1 on the source side
        // implies 0 on the source side" is a hard constraint (arc 1 -> 0).
        let terminals = [(0, 10), (10, 0)];
        let edges = [(1, 0, INFINITE_CAPACITY, 0)];
        let (flow, g) = solve(&terminals, &edges);
        assert_eq!(flow, 10);
        assert_eq!(g.in_source_set(0), g.in_source_set(1));
    }

    #[test]
    fn refuses_malformed_graphs_by_name() {
        let mut g = CutGraph::new(2, 0).unwrap();
        assert!(g.add_edge(0, 0, 1, 1).is_err());
        assert!(g.add_edge(0, 2, 1, 1).is_err());
        assert!(g.add_edge(0, 1, -1, 0).is_err());
        assert!(g.add_terminal_weights(0, -1, 0).is_err());
        g.max_flow();
        assert!(g.add_edge(0, 1, 1, 1).is_err());
    }
}
