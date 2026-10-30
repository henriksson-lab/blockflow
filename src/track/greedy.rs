// SPDX-License-Identifier: MIT

//! A deliberately simple association baseline. Sort candidates by cost and
//! accept a pair if both ends remain free and linking is cheaper than a birth
//! plus a death. This is deterministic, but does not minimize the global LAP
//! objective. No divisions, merges, or gap closing are inferred.

use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

use super::{
    AlternativeCost, CandidateSet, DetectionSet, FrameLapLinker, GeometricCost, LapParams,
    LinkKind, Linker, LinkerCapabilities, ScoredCandidates, ScoredLinker, TrackGraph, TrackLink,
};
use crate::error::{Error, Result};

#[derive(Debug, Clone, Default)]
pub struct GreedyFrameLinker {
    alternative: AlternativeCost,
}

impl GreedyFrameLinker {
    pub fn new(alternative: AlternativeCost) -> Result<Self> {
        FrameLapLinker::new(LapParams {
            alternative,
            ..LapParams::default()
        })?;
        Ok(Self { alternative })
    }
}

impl Linker for GreedyFrameLinker {
    fn name(&self) -> &'static str {
        "frame-greedy"
    }
    fn capabilities(&self) -> LinkerCapabilities {
        LinkerCapabilities::default()
    }
    fn parameters(&self) -> serde_json::Value {
        let alternative = match self.alternative {
            AlternativeCost::Fixed(cost) => json!({"kind": "fixed", "cost": cost}),
            AlternativeCost::RelativeToMaxLink { factor, fallback } => {
                json!({"kind": "relative_to_max_link", "factor": factor, "fallback": fallback})
            }
        };
        json!({"default_cost": "distance^2", "alternative": alternative,
            "selection": "ascending cost, then from/to ID; require cost < birth + death"})
    }
    fn link(&self, detections: &DetectionSet, candidates: &CandidateSet) -> Result<TrackGraph> {
        self.link_scored(
            detections,
            &ScoredCandidates::score(detections, candidates, &GeometricCost::default())?,
        )
    }
}

impl ScoredLinker for GreedyFrameLinker {
    fn link_scored(
        &self,
        detections: &DetectionSet,
        scores: &ScoredCandidates,
    ) -> Result<TrackGraph> {
        scores.validate(detections)?;
        let candidates = scores.candidates().edges();
        let mut by_frame: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (i, edge) in candidates.iter().enumerate() {
            if edge.dt != 1 {
                return Err(Error::invalid("frame-greedy links consecutive frames only"));
            }
            by_frame.entry(edge.from_t).or_default().push(i);
        }
        let mut links = Vec::new();
        for mut indices in by_frame.into_values() {
            let alternative = match self.alternative {
                AlternativeCost::Fixed(cost) => cost,
                AlternativeCost::RelativeToMaxLink { factor, fallback } => {
                    let max = indices
                        .iter()
                        .map(|&i| scores.costs()[i])
                        .fold(f64::NEG_INFINITY, f64::max);
                    if max > 0.0 {
                        factor * max
                    } else {
                        fallback
                    }
                }
            };
            if !alternative.is_finite() {
                return Err(Error::invalid("resolved alternative cost is non-finite"));
            }
            indices.sort_by(|&a, &b| {
                scores.costs()[a].total_cmp(&scores.costs()[b]).then(
                    (candidates[a].from, candidates[a].to)
                        .cmp(&(candidates[b].from, candidates[b].to)),
                )
            });
            let (mut sources, mut targets) = (BTreeSet::new(), BTreeSet::new());
            for index in indices {
                let edge = &candidates[index];
                let cost = scores.costs()[index];
                if cost - alternative >= alternative
                    || sources.contains(&edge.from)
                    || targets.contains(&edge.to)
                {
                    continue;
                }
                sources.insert(edge.from);
                targets.insert(edge.to);
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
        TrackGraph::new(detections, links, self.capabilities())
    }
}
