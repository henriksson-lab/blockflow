// SPDX-License-Identifier: MIT

//! Optional continuation/gap assignment after any fixed-mask tracker.
//!
//! Existing edges and divisions are immutable. Callers generate candidates
//! from current ends to current starts and score the *incremental* cost of a
//! join relative to leaving both unmatched. This avoids mixing a first-stage
//! model's energy scale with an unrelated gap model or implicit alternatives.

use std::collections::{BTreeMap, BTreeSet};

use super::{assign_sparse, DetectionSet, LinkKind, ScoredCandidates, TrackGraph, TrackLink};
use crate::error::{Error, Result};

/// Join existing ends to existing starts by minimum-cost sparse assignment.
///
/// Scores must be finite incremental energies in one common scale. Leaving
/// endpoints unmatched costs zero; only strictly negative joins are eligible.
/// No candidates, scores or motion priors are inferred here. Existing links
/// keep their original costs; costs across stages need not form one posterior.
/// Gating, maximum temporal separation and division plausibility belong to
/// upstream candidate/scoring policies, which should be retained as provenance.
///
/// An isolated detection can both receive and send a join. Forward time and
/// one-to-one assignment preserve acyclicity and cannot create a new division
/// or merge. Adding endpoints does not repair an existing wrong division.
pub fn join_endpoints(
    detections: &DetectionSet,
    graph: &TrackGraph,
    scores: &ScoredCandidates,
) -> Result<TrackGraph> {
    if !graph.detection_frames().eq(detections
        .detections()
        .iter()
        .map(|d| (d.id, d.t_index))
        .collect::<BTreeMap<_, _>>())
    {
        return Err(Error::invalid(
            "endpoint refinement needs the same detection IDs and frames",
        ));
    }
    scores.validate(detections)?;
    let mut from_ids = BTreeSet::new();
    let mut to_ids = BTreeSet::new();
    for edge in scores.candidates().edges() {
        if !graph.successors(edge.from).is_empty() || !graph.predecessors(edge.to).is_empty() {
            return Err(Error::invalid(
                "endpoint candidates must join current ends to current starts",
            ));
        }
        from_ids.insert(edge.from);
        to_ids.insert(edge.to);
    }
    let from_ids = from_ids.into_iter().collect::<Vec<_>>();
    let to_ids = to_ids.into_iter().collect::<Vec<_>>();
    let from_row = from_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect::<BTreeMap<_, _>>();
    let to_col = to_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect::<BTreeMap<_, _>>();
    let mut rows = (0..from_ids.len())
        .map(|i| vec![(to_ids.len() + i, 0.0)])
        .collect::<Vec<_>>();
    let mut selected_edge = BTreeMap::new();
    for (edge, &cost) in scores.candidates().edges().iter().zip(scores.costs()) {
        if cost < 0.0 {
            let row = from_row[&edge.from];
            let col = to_col[&edge.to];
            rows[row].push((col, cost));
            selected_edge.insert((row, col), (edge, cost));
        }
    }
    let assignment = assign_sparse(&rows, to_ids.len() + from_ids.len())?;
    let mut links = graph.links().to_vec();
    let mut capabilities = graph.capabilities();
    for (row, col) in assignment.into_iter().enumerate() {
        if col < to_ids.len() {
            let (edge, cost) = selected_edge[&(row, col)];
            capabilities.gaps |= edge.dt > 1;
            links.push(TrackLink {
                from: edge.from,
                to: edge.to,
                dt: edge.dt,
                elapsed_seconds: edge.elapsed_seconds,
                cost,
                kind: if edge.dt > 1 {
                    LinkKind::Gap
                } else {
                    LinkKind::Continue
                },
            });
        }
    }
    TrackGraph::new(detections, links, capabilities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track::{
        Candidate, CandidateSet, Detection, Frame, LinkerCapabilities, SeriesGeometry,
    };
    use serde_json::json;

    fn input() -> (DetectionSet, TrackGraph) {
        let detections = DetectionSet::new(
            SeriesGeometry::new([1, 20, 20], [1.; 3]).unwrap(),
            (0..4)
                .map(|t| Frame {
                    t_index: t,
                    elapsed_seconds: Some(t as f64 * 10.),
                })
                .collect(),
            (1..=5)
                .map(|id| Detection {
                    id: (1 << 60) + id,
                    t_index: match id {
                        1 => 0,
                        2 | 3 => 1,
                        _ => 3,
                    },
                    native: [0., 5., 5.],
                    registered: [0., 5., 5.],
                    size: 10.,
                })
                .collect(),
        )
        .unwrap();
        let graph = TrackGraph::new(
            &detections,
            [2, 3]
                .into_iter()
                .map(|id| TrackLink {
                    from: (1 << 60) + 1,
                    to: (1 << 60) + id,
                    dt: 1,
                    elapsed_seconds: Some(10.),
                    cost: 7.,
                    kind: LinkKind::Divide,
                })
                .collect(),
            LinkerCapabilities {
                divisions: true,
                ..Default::default()
            },
        )
        .unwrap();
        (detections, graph)
    }

    fn scores(detections: &DetectionSet, pairs: &[(u64, u64, f64)]) -> ScoredCandidates {
        let edges = pairs
            .iter()
            .map(|&(a, b, _)| {
                let from = detections.get((1 << 60) + a).unwrap();
                let to = detections.get((1 << 60) + b).unwrap();
                Candidate {
                    from: from.id,
                    to: to.id,
                    from_t: from.t_index,
                    dt: to.t_index - from.t_index,
                    elapsed_seconds: Some((to.t_index - from.t_index) as f64 * 10.),
                    distance: 0.,
                    relative_size_change: 0.,
                    overlap: None,
                }
            })
            .collect();
        let by_pair = pairs
            .iter()
            .map(|&(a, b, c)| (((1 << 60) + a, (1 << 60) + b), c))
            .collect::<BTreeMap<_, _>>();
        let candidates = CandidateSet::from_edges(edges).unwrap();
        let costs = candidates
            .edges()
            .iter()
            .map(|e| by_pair[&(e.from, e.to)])
            .collect();
        ScoredCandidates::from_costs(candidates, costs, json!({"role":"test incremental costs"}))
            .unwrap()
    }

    #[test]
    fn global_endpoint_assignment_preserves_division_and_uses_real_gap_time() {
        let (detections, graph) = input();
        // Greedy would take2->4 (-5), losing the better joint choice (-8).
        let scored = scores(&detections, &[(2, 4, -5.), (2, 5, -4.), (3, 4, -4.)]);
        let refined = join_endpoints(&detections, &graph, &scored).unwrap();
        assert_eq!(refined.links().len(), 4);
        assert!(graph.links().iter().all(|e| refined.links().contains(e)));
        assert_eq!(refined.successors((1 << 60) + 2), &[(1 << 60) + 5]);
        assert_eq!(refined.successors((1 << 60) + 3), &[(1 << 60) + 4]);
        let gaps = refined
            .links()
            .iter()
            .filter(|e| e.kind == LinkKind::Gap)
            .collect::<Vec<_>>();
        assert_eq!(gaps.len(), 2);
        assert!(gaps
            .iter()
            .all(|e| e.dt == 2 && e.elapsed_seconds == Some(20.)));
        assert!(refined.capabilities().gaps);
        assert_eq!(
            refined
                .links()
                .iter()
                .filter(|e| e.kind == LinkKind::Divide)
                .count(),
            2
        );
    }

    #[test]
    fn nonnegative_and_empty_candidates_leave_graph_unchanged() {
        let (detections, graph) = input();
        for pairs in [vec![], vec![(2, 4, 0.), (3, 5, 1.)]] {
            let refined =
                join_endpoints(&detections, &graph, &scores(&detections, &pairs)).unwrap();
            assert_eq!(refined.links(), graph.links());
            assert_eq!(refined.capabilities(), graph.capabilities());
        }
    }

    #[test]
    fn rejects_candidates_from_a_dividing_mother_or_into_linked_daughter() {
        let (detections, graph) = input();
        assert!(join_endpoints(&detections, &graph, &scores(&detections, &[(1, 4, -1.)])).is_err());
        assert!(join_endpoints(&detections, &graph, &scores(&detections, &[(1, 2, -1.)])).is_err());
    }
}
