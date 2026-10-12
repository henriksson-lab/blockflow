// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! The version-1 linker: **frame-to-frame linear assignment**.
//!
//! This is the first step of Jaqaman et al., *Robust single-particle tracking
//! in live-cell time-lapse sequences* (Nature Methods, 2008) — the step that
//! TrackMate's LAP tracker and `laptrack` also run first. Gap closing,
//! division and merging are that paper's second step and are not done here;
//! [`FrameLapLinker::capabilities`] says so and the graph validator holds the
//! output to it.
//!
//! # The problem, per adjacent frame pair
//!
//! Every source detection at `t` either links to one target at `t + 1` (cost
//! `c_ij`) or ends (its death cost); every target either is linked or starts
//! (its birth cost). Minimise the total. The standard augmented matrix
//!
//! ```text
//!            targets      death dummies
//! sources  [   C       |   diag(d)    ]
//! births   [ diag(b)   |     0        ]
//! ```
//!
//! encodes exactly that, with forbidden entries (no candidate, off-diagonal
//! alternatives) infinite. The zero block lets each birth dummy that is not
//! needed for a target absorb a death dummy left free by a link.
//!
//! # Sparse, not dense
//!
//! The matrix is never built. A frame pair is solved in an equivalent sparse
//! form — see `solve_pair` — whose size is the number of candidate edges.
//! An earlier version split the gated graph into connected components and
//! solved each densely; on a real 2052 x 2456 field with a noisy segmentation
//! one component held about 4,400 detections and a frame pair took minutes.
//! Dense clusters of touching cells produce the same shape, so the solver
//! must not be cubic in a component's size.
//!
//! # Determinism
//!
//! Rows (sources) and columns (targets) are ordered by id, and the solver
//! breaks ties by node index, so it is deterministic. Identical input, parameters and solver version give identical
//! output. **This is not a promise of a particular assignment among equal
//! optima**, and a different LAP solver — `laptrack`'s, say — may pick a
//! different one when costs tie exactly.

use std::collections::BTreeMap;

use serde_json::json;

use super::graph::{LinkKind, TrackGraph, TrackLink};
use super::{CandidateSet, DetectionSet, Linker, LinkerCapabilities};
use crate::error::{Error, Result};

/// The cost of ending or starting a track instead of linking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlternativeCost {
    /// The same cost for every birth and death, in link-cost units.
    Fixed(f64),
    /// `factor × the largest candidate link cost` of the frame pair, the
    /// Jaqaman default with `factor = 1.05`. A pair with no candidate uses
    /// `fallback`.
    RelativeToMaxLink { factor: f64, fallback: f64 },
}

impl Default for AlternativeCost {
    fn default() -> Self {
        Self::RelativeToMaxLink {
            factor: 1.05,
            fallback: 1.0,
        }
    }
}

/// Link cost terms. The cost of a candidate is
///
/// `distance² + iou_weight · (1 − IoU) + size_weight · relative_size_change`,
///
/// with distance in registered physical units. Every weight must be finite and
/// non-negative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LapParams {
    pub iou_weight: f64,
    pub size_weight: f64,
    pub alternative: AlternativeCost,
}

impl Default for LapParams {
    fn default() -> Self {
        Self {
            iou_weight: 0.0,
            size_weight: 0.0,
            alternative: AlternativeCost::default(),
        }
    }
}

impl LapParams {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("iou_weight", self.iou_weight),
            ("size_weight", self.size_weight),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "{name} must be finite and non-negative, got {value}"
                )));
            }
        }
        match self.alternative {
            AlternativeCost::Fixed(cost) if !cost.is_finite() || cost < 0.0 => Err(
                Error::invalid(format!("alternative cost must be finite and non-negative, got {cost}")),
            ),
            AlternativeCost::RelativeToMaxLink { factor, fallback }
                if !factor.is_finite() || factor <= 0.0 || !fallback.is_finite() || fallback < 0.0 =>
            {
                Err(Error::invalid(format!(
                    "relative alternative cost needs a positive factor and non-negative fallback, got {factor} and {fallback}"
                )))
            }
            _ => Ok(()),
        }
    }
}

/// Frame-to-frame LAP. See the module header.
#[derive(Debug, Clone, Default)]
pub struct FrameLapLinker {
    params: LapParams,
}

impl FrameLapLinker {
    pub fn new(params: LapParams) -> Result<Self> {
        params.validate()?;
        Ok(Self { params })
    }

    fn cost(&self, edge: &super::Candidate) -> Result<f64> {
        let mut cost = edge.distance * edge.distance;
        if self.params.iou_weight > 0.0 {
            let overlap = edge.overlap.ok_or_else(|| {
                Error::invalid(format!(
                    "iou_weight is set but candidate {} -> {} has no overlap features",
                    edge.from, edge.to
                ))
            })?;
            cost += self.params.iou_weight * (1.0 - overlap.iou);
        }
        cost += self.params.size_weight * edge.relative_size_change;
        if !cost.is_finite() || cost < 0.0 {
            return Err(Error::invalid(format!(
                "candidate {} -> {} has invalid cost {cost}",
                edge.from, edge.to
            )));
        }
        Ok(cost)
    }
}

impl Linker for FrameLapLinker {
    fn name(&self) -> &'static str {
        "frame-lap"
    }

    fn capabilities(&self) -> LinkerCapabilities {
        LinkerCapabilities::default()
    }

    fn parameters(&self) -> serde_json::Value {
        let alternative = match self.params.alternative {
            AlternativeCost::Fixed(cost) => json!({"kind": "fixed", "cost": cost}),
            AlternativeCost::RelativeToMaxLink { factor, fallback } => {
                json!({"kind": "relative_to_max_link", "factor": factor, "fallback": fallback})
            }
        };
        json!({
            "cost": "distance^2 + iou_weight*(1-iou) + size_weight*relative_size_change",
            "iou_weight": self.params.iou_weight,
            "size_weight": self.params.size_weight,
            "alternative": alternative,
        })
    }

    fn link(&self, detections: &DetectionSet, candidates: &CandidateSet) -> Result<TrackGraph> {
        TrackGraph::new(
            detections,
            self.frame_links(detections, candidates)?,
            self.capabilities(),
        )
    }
}

impl FrameLapLinker {
    /// The frame-to-frame links alone, before any graph is built — the first
    /// step a segment linker builds on.
    pub fn frame_links(
        &self,
        detections: &DetectionSet,
        candidates: &CandidateSet,
    ) -> Result<Vec<TrackLink>> {
        // Group candidates by the frame they leave.
        let mut by_pair: BTreeMap<u64, Vec<(usize, f64)>> = BTreeMap::new();
        for (index, edge) in candidates.edges().iter().enumerate() {
            let from = detections.get(edge.from).ok_or_else(|| {
                Error::invalid(format!("candidate source {} is not a detection", edge.from))
            })?;
            let to = detections.get(edge.to).ok_or_else(|| {
                Error::invalid(format!("candidate target {} is not a detection", edge.to))
            })?;
            if from.t_index != edge.from_t || to.t_index != edge.from_t + edge.dt {
                return Err(Error::invalid(format!(
                    "candidate {} -> {} disagrees with the detections' frames",
                    edge.from, edge.to
                )));
            }
            if edge.dt != 1 {
                return Err(Error::invalid(format!(
                    "frame-lap links consecutive frames only; candidate {} -> {} spans {} frames",
                    edge.from, edge.to, edge.dt
                )));
            }
            by_pair
                .entry(edge.from_t)
                .or_default()
                .push((index, self.cost(edge)?));
        }

        let mut links = Vec::new();
        for (_, pair) in by_pair {
            let alternative = match self.params.alternative {
                AlternativeCost::Fixed(cost) => cost,
                AlternativeCost::RelativeToMaxLink { factor, fallback } => {
                    let max = pair
                        .iter()
                        .map(|&(_, c)| c)
                        .fold(f64::NEG_INFINITY, f64::max);
                    if max.is_finite() && max > 0.0 {
                        factor * max
                    } else {
                        fallback
                    }
                }
            };
            for (index, cost) in solve_pair(candidates, &pair, alternative)? {
                let edge = &candidates.edges()[index];
                links.push(TrackLink {
                    from: edge.from,
                    to: edge.to,
                    dt: edge.dt,
                    elapsed_seconds: edge.elapsed_seconds,
                    cost,
                    kind: LinkKind::Continue,
                });
            }
        }
        Ok(links)
    }
}

/// Solve one frame pair. Returns `(candidate index, cost)` for every chosen
/// link, in candidate order.
///
/// The augmented matrix in the module header is solved in an equivalent
/// sparse form. Matching target `j` saves its birth cost, so a link costs
/// `c_ij - b_j` and the births of every target are added back as a constant;
/// each source then either takes one target or its own death column at `d_i`.
/// That is a row-complete assignment of the sources with no dummy rows at all
/// and only the candidate edges plus one death edge per source — see
/// [`assign_sparse`]. The optimum is the augmented matrix's optimum.
fn solve_pair(
    candidates: &CandidateSet,
    pair: &[(usize, f64)],
    alternative: f64,
) -> Result<Vec<(usize, f64)>> {
    let mut sources: Vec<u64> = pair
        .iter()
        .map(|&(i, _)| candidates.edges()[i].from)
        .collect();
    let mut targets: Vec<u64> = pair
        .iter()
        .map(|&(i, _)| candidates.edges()[i].to)
        .collect();
    sources.sort_unstable();
    sources.dedup();
    targets.sort_unstable();
    targets.dedup();
    let n = sources.len();
    let m = targets.len();
    // Columns: targets 0..m, then source i's death column m + i.
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut edge_of: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); n];
    for &(index, cost) in pair {
        let edge = &candidates.edges()[index];
        let row = sources.binary_search(&edge.from).expect("source listed");
        let col = targets.binary_search(&edge.to).expect("target listed");
        rows[row].push((col, cost - alternative));
        edge_of[row].push((col, index, cost));
    }
    for (row, edges) in rows.iter_mut().enumerate() {
        edges.push((m + row, alternative));
    }
    let assignment = assign_sparse(&rows, m + n)?;
    let mut chosen = Vec::new();
    for (row, &col) in assignment.iter().enumerate() {
        if col < m {
            let &(_, index, cost) = edge_of[row]
                .iter()
                .find(|&&(c, _, _)| c == col)
                .expect("assigned along a candidate edge");
            chosen.push((index, cost));
        }
    }
    chosen.sort_unstable_by_key(|&(index, _)| index);
    Ok(chosen)
}

/// Minimum-cost assignment of every row to a distinct column, over sparse
/// edges `rows[i] = [(column, cost), ...]`. Columns may stay unassigned.
/// Costs may be negative. Returns each row's column.
///
/// Successive shortest augmenting paths with Johnson potentials: rows are
/// added in order, and each is placed by a Dijkstra search over reduced costs
/// on the residual graph that **stops at the first free column it settles**.
/// Optimality is kept after every row, so the final assignment is optimal.
/// Work per row is the part of the graph the search touches before it finds a
/// free column — in a tracking frame pair, usually a handful of nodes — so a
/// pair costs about `O(E log V)` instead of the `O(V³)` of a dense solve, and
/// a large connected cluster of candidates is no longer quadratic in memory
/// or cubic in time.
///
/// Deterministic: ties in the queue break by node index.
pub(crate) fn assign_sparse(rows: &[Vec<(usize, f64)>], columns: usize) -> Result<Vec<usize>> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(PartialEq)]
    struct Item(f64, usize);
    impl Eq for Item {}
    impl PartialOrd for Item {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Item {
        // Min-heap on distance, then on node.
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .0
                .total_cmp(&self.0)
                .then_with(|| other.1.cmp(&self.1))
        }
    }

    let n = rows.len();
    for (row, edges) in rows.iter().enumerate() {
        if edges.is_empty() {
            return Err(Error::invalid(format!("assignment row {row} has no edge")));
        }
        for &(col, cost) in edges {
            if col >= columns || !cost.is_finite() {
                return Err(Error::invalid(format!(
                    "assignment row {row} has an invalid edge to column {col} costing {cost}"
                )));
            }
        }
    }
    // Node ids: rows 0..n, columns n..n+columns.
    let mut potential = vec![0.0f64; n + columns];
    let mut row_of = vec![usize::MAX; columns]; // column -> row
    let mut col_of = vec![usize::MAX; n]; // row -> column
    let mut dist = vec![f64::INFINITY; n + columns];
    let mut done = vec![false; n + columns];
    let mut parent = vec![usize::MAX; n + columns]; // column -> row it was reached from
    let mut touched: Vec<usize> = Vec::new();

    for root in 0..n {
        // A new row has no incoming residual edge, so its potential is free:
        // choose it to make every outgoing reduced cost non-negative.
        potential[root] = rows[root]
            .iter()
            .map(|&(col, cost)| potential[n + col] - cost)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut heap = BinaryHeap::new();
        dist[root] = 0.0;
        touched.push(root);
        heap.push(Item(0.0, root));
        let mut free_end = None;
        while let Some(Item(d, node)) = heap.pop() {
            if done[node] || d > dist[node] {
                continue;
            }
            done[node] = true;
            if node >= n {
                let col = node - n;
                match row_of[col] {
                    usize::MAX => {
                        free_end = Some((col, d));
                        break;
                    }
                    row => {
                        // The matched edge, reversed, is tight: reduced cost 0.
                        if !done[row] && d < dist[row] {
                            if dist[row].is_infinite() {
                                touched.push(row);
                            }
                            dist[row] = d;
                            heap.push(Item(d, row));
                        }
                    }
                }
            } else {
                for &(col, cost) in &rows[node] {
                    if col_of[node] == col {
                        continue;
                    }
                    let target = n + col;
                    if done[target] {
                        continue;
                    }
                    let reduced = (cost + potential[node] - potential[target]).max(0.0);
                    let next = d + reduced;
                    if next < dist[target] {
                        if dist[target].is_infinite() {
                            touched.push(target);
                        }
                        dist[target] = next;
                        parent[target] = node;
                        heap.push(Item(next, target));
                    }
                }
            }
        }
        let (end, total) = free_end.ok_or_else(|| {
            Error::invalid("assignment has no free column for a row; every row needs its own")
        })?;
        // Potentials: p(v) += min(dist(v), total) - total keeps every reduced
        // cost non-negative and makes the path tight. The `- total` makes the
        // shift zero for every node the search never reached (dist = inf), so
        // only touched nodes need updating — without it, untouched nodes would
        // each need `+ total` and leaving them out breaks optimality.
        for &node in &touched {
            potential[node] += dist[node].min(total) - total;
        }
        // Augment along the path back to the root.
        let mut col = end;
        loop {
            let row = parent[n + col];
            let previous = col_of[row];
            row_of[col] = row;
            col_of[row] = col;
            if row == root {
                break;
            }
            col = previous;
        }
        for &node in &touched {
            dist[node] = f64::INFINITY;
            done[node] = false;
            parent[node] = usize::MAX;
        }
        touched.clear();
    }
    Ok(col_of)
}

/// Minimum-cost perfect assignment on a dense square matrix whose forbidden
/// entries are `+∞`. Kept as the reference the sparse solver is tested
/// against. Returns, for each row, its column.
///
/// The shortest-augmenting-path Hungarian algorithm with potentials,
/// `O(n³)`: rows are added one at a time and each is placed by a Dijkstra-like
/// search over reduced costs. Ties are resolved by the lowest column index
/// reached first, which makes the result a deterministic function of the
/// matrix.
#[cfg(test)]
pub(crate) fn hungarian(matrix: &[f64], n: usize) -> Result<Vec<usize>> {
    if matrix.len() != n * n {
        return Err(Error::invalid("assignment matrix is not square"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    // 1-based arrays, column 0 is the virtual start.
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = row matched to column j
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = usize::MAX;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let a = matrix[(i0 - 1) * n + (j - 1)];
                let cur = if a.is_finite() {
                    a - u[i0] - v[j]
                } else {
                    f64::INFINITY
                };
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            if j1 == usize::MAX || !delta.is_finite() {
                return Err(Error::invalid(
                    "assignment has no finite solution; every row needs a finite alternative",
                ));
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else if minv[j].is_finite() {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut assignment = vec![0usize; n];
    for j in 1..=n {
        assignment[p[j] - 1] = j - 1;
    }
    Ok(assignment)
}
