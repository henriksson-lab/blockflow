// SPDX-License-Identifier: MIT

//! Candidate scoring, independent of assignment.
//!
//! A scorer can capture a table keyed by detection ID for shape descriptors or
//! learned features. It receives only the gated pairs, so expensive image
//! features should be measured once upstream, not recomputed here.
//! Costs are energies, not automatically calibrated probabilities. They may
//! be negative but must be finite; omit a candidate to forbid an association.

use serde_json::{json, Value};

use super::{Candidate, CandidateSet, DetectionSet, Linker, TrackGraph};
use crate::error::{Error, Result};

/// A pluggable, thread-safe pairwise cost. Lower means a preferred link.
/// This does not describe a joint mother/two-daughter event or cell age.
pub trait LinkCost: Send + Sync {
    fn name(&self) -> &'static str;
    fn parameters(&self) -> Value;
    fn cost(&self, detections: &DetectionSet, candidate: &Candidate) -> Result<f64>;
}

/// Candidate features and their costs stay together in canonical edge order.
/// Reuse this object to compare solvers without recomputing the scores.
#[derive(Debug, Clone)]
pub struct ScoredCandidates {
    candidates: CandidateSet,
    costs: Vec<f64>,
    provenance: Value,
}

impl ScoredCandidates {
    pub fn score(
        detections: &DetectionSet,
        candidates: &CandidateSet,
        scorer: &dyn LinkCost,
    ) -> Result<Self> {
        Self::validate_candidates(detections, candidates)?;
        let costs = candidates
            .edges()
            .iter()
            .map(|edge| scorer.cost(detections, edge))
            .collect::<Result<Vec<_>>>()?;
        Self::from_costs(
            candidates.clone(),
            costs,
            json!({
                "name": scorer.name(), "parameters": scorer.parameters(),
            }),
        )
    }

    /// Import scores in precisely `candidates.edges()` order. Imported model
    /// scores should include checkpoint identity and configuration in provenance.
    pub fn from_costs(
        candidates: CandidateSet,
        costs: Vec<f64>,
        provenance: Value,
    ) -> Result<Self> {
        if candidates.len() != costs.len() {
            return Err(Error::invalid(
                "one score is required per candidate in canonical edge order",
            ));
        }
        for (edge, &cost) in candidates.edges().iter().zip(&costs) {
            if !cost.is_finite() {
                return Err(Error::invalid(format!(
                    "candidate {} -> {} has non-finite cost {cost}",
                    edge.from, edge.to
                )));
            }
        }
        Ok(Self {
            candidates,
            costs,
            provenance,
        })
    }

    pub fn candidates(&self) -> &CandidateSet {
        &self.candidates
    }
    pub fn costs(&self) -> &[f64] {
        &self.costs
    }
    pub fn provenance(&self) -> &Value {
        &self.provenance
    }

    /// Check against the actual detection set supplied to a solver. Frame
    /// metadata and timestamps must agree; imported candidates cannot spoof dt.
    pub fn validate(&self, detections: &DetectionSet) -> Result<()> {
        Self::validate_candidates(detections, &self.candidates)
    }

    fn validate_candidates(detections: &DetectionSet, candidates: &CandidateSet) -> Result<()> {
        for edge in candidates.edges() {
            let from = detections.get(edge.from).ok_or_else(|| {
                Error::invalid(format!("candidate source {} is not a detection", edge.from))
            })?;
            let to = detections.get(edge.to).ok_or_else(|| {
                Error::invalid(format!("candidate target {} is not a detection", edge.to))
            })?;
            if edge.dt == 0
                || from.t_index != edge.from_t
                || from.t_index.checked_add(edge.dt) != Some(to.t_index)
            {
                return Err(Error::invalid(format!(
                    "candidate {} -> {} disagrees with the detections' frames",
                    edge.from, edge.to
                )));
            }
            let expected_elapsed = match (
                detections
                    .frame(from.t_index)
                    .and_then(|f| f.elapsed_seconds),
                detections.frame(to.t_index).and_then(|f| f.elapsed_seconds),
            ) {
                (Some(a), Some(b)) => Some(b - a),
                _ => None,
            };
            if edge.elapsed_seconds != expected_elapsed {
                return Err(Error::invalid(format!(
                    "candidate {} -> {} disagrees with the acquired timestamps",
                    edge.from, edge.to
                )));
            }
        }
        Ok(())
    }
}

/// Assignment of precomputed costs. Existing [`Linker`] implementations keep
/// their original default scoring; this additional API permits controlled
/// score/solver combinations. Capabilities still constrain graph validation.
pub trait ScoredLinker: Linker {
    fn link_scored(
        &self,
        detections: &DetectionSet,
        scores: &ScoredCandidates,
    ) -> Result<TrackGraph>;
}

/// Use a scorer/solver combination wherever an ordinary [`Linker`] is accepted.
/// Both configurations are included in output provenance.
pub struct CostedLinker<S, C> {
    pub solver: S,
    pub scorer: C,
}

impl<S: ScoredLinker, C: LinkCost> Linker for CostedLinker<S, C> {
    fn name(&self) -> &'static str {
        "costed-linker"
    }
    fn capabilities(&self) -> super::LinkerCapabilities {
        self.solver.capabilities()
    }
    fn parameters(&self) -> Value {
        json!({
            "solver": { "name": self.solver.name(), "parameters": self.solver.parameters() },
            "scorer": { "name": self.scorer.name(), "parameters": self.scorer.parameters() },
        })
    }
    fn link(&self, detections: &DetectionSet, candidates: &CandidateSet) -> Result<TrackGraph> {
        self.solver.link_scored(
            detections,
            &ScoredCandidates::score(detections, candidates, &self.scorer)?,
        )
    }
}

/// The legacy geometric cost, available independently from the LAP solver.
/// Units are squared physical distance. Optional weights use compatible units.
#[derive(Debug, Clone, Copy, Default)]
pub struct GeometricCost {
    iou_weight: f64,
    size_weight: f64,
}

impl GeometricCost {
    pub fn new(iou_weight: f64, size_weight: f64) -> Result<Self> {
        for (name, value) in [("iou_weight", iou_weight), ("size_weight", size_weight)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "{name} must be finite and non-negative, got {value}"
                )));
            }
        }
        Ok(Self {
            iou_weight,
            size_weight,
        })
    }
}

impl LinkCost for GeometricCost {
    fn name(&self) -> &'static str {
        "geometric"
    }
    fn parameters(&self) -> Value {
        json!({"formula": "distance^2 + iou_weight*(1-iou) + size_weight*relative_size_change",
            "iou_weight": self.iou_weight, "size_weight": self.size_weight})
    }
    fn cost(&self, _: &DetectionSet, edge: &Candidate) -> Result<f64> {
        let mut cost = edge.distance * edge.distance;
        if self.iou_weight > 0.0 {
            let overlap = edge.overlap.ok_or_else(|| {
                Error::invalid(format!(
                    "iou_weight is set but candidate {} -> {} has no overlap features",
                    edge.from, edge.to
                ))
            })?;
            if !overlap.iou.is_finite() || !(0.0..=1.0).contains(&overlap.iou) {
                return Err(Error::invalid("candidate IoU must be finite and in [0, 1]"));
            }
            cost += self.iou_weight * (1.0 - overlap.iou);
        }
        if !edge.relative_size_change.is_finite()
            || !(0.0..=1.0).contains(&edge.relative_size_change)
        {
            return Err(Error::invalid(
                "relative size change must be finite and in [0, 1]",
            ));
        }
        cost += self.size_weight * edge.relative_size_change;
        if !cost.is_finite() || cost < 0.0 {
            return Err(Error::invalid(format!(
                "candidate {} -> {} has invalid cost {cost}",
                edge.from, edge.to
            )));
        }
        Ok(cost)
    }
}
