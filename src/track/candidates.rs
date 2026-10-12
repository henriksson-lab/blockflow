// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Gated candidate edges between consecutive acquired frames.
//!
//! A candidate is a link a linker *may* choose. Gating is the only decision
//! made here: an edge longer than the gate does not exist, so no linker can
//! choose it. Everything else — costs, alternatives, assignment — is the
//! linker's.
//!
//! The gate is a physical distance in registered space: either a fixed
//! `max_distance`, or `max_speed × elapsed` when the frames carry timestamps
//! (uneven intervals then gate correctly). When both are given the smaller
//! wins.
//!
//! Candidates are found with a uniform grid whose cell is the gate, so each
//! detection looks at 27 cells; edges are then sorted by `(from, to)`, which
//! is the only order anything downstream sees.

use std::collections::{BTreeMap, HashMap};

use ndarray::ArrayView3;

use super::{DetectionSet, Frame};
use crate::error::{Error, Result};

/// One possible link from a detection in frame `t` to one in frame `t + dt`.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub from: u64,
    pub to: u64,
    pub from_t: u64,
    /// Frame-index difference. Always 1 in version 1.
    pub dt: u64,
    /// Elapsed time between the two frames, when the timeline has timestamps.
    pub elapsed_seconds: Option<f64>,
    /// Physical distance in registered space.
    pub distance: f64,
    /// Size of `to` relative to `from`: `|s_to - s_from| / max(s_to, s_from)`.
    pub relative_size_change: f64,
    /// Overlap features, when [`CandidateSet::attach_overlap`] supplied them.
    pub overlap: Option<OverlapFeatures>,
}

/// Pixel overlap between the two detections of a candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverlapFeatures {
    pub shared: u64,
    pub iou: f64,
}

/// The gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateParams {
    /// Largest physical displacement between consecutive frames.
    pub max_distance: Option<f64>,
    /// Largest physical speed; the gate is `max_speed × elapsed`. Needs
    /// timestamps on the timeline.
    pub max_speed: Option<f64>,
}

impl CandidateParams {
    pub fn distance(max_distance: f64) -> Self {
        Self {
            max_distance: Some(max_distance),
            max_speed: None,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.max_distance.is_none() && self.max_speed.is_none() {
            return Err(Error::invalid(
                "candidate gating needs max_distance, max_speed or both",
            ));
        }
        for (name, value) in [
            ("max_distance", self.max_distance),
            ("max_speed", self.max_speed),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || value <= 0.0 {
                    return Err(Error::invalid(format!(
                        "{name} must be finite and positive, got {value}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// The gate between two frames.
    fn gate(&self, from: &Frame, to: &Frame) -> Result<f64> {
        let mut gate = self.max_distance.unwrap_or(f64::INFINITY);
        if let Some(speed) = self.max_speed {
            let (Some(a), Some(b)) = (from.elapsed_seconds, to.elapsed_seconds) else {
                return Err(Error::invalid(
                    "max_speed gating needs elapsed_seconds on every frame",
                ));
            };
            gate = gate.min(speed * (b - a));
        }
        Ok(gate)
    }
}

/// Every candidate edge of one series, sorted by `(from, to)`.
#[derive(Debug, Clone, Default)]
pub struct CandidateSet {
    edges: Vec<Candidate>,
}

impl CandidateSet {
    /// Build from edges in any order. Refuses duplicates and non-finite values.
    pub fn from_edges(mut edges: Vec<Candidate>) -> Result<Self> {
        edges.sort_by_key(|e| (e.from, e.to));
        for pair in edges.windows(2) {
            if (pair[0].from, pair[0].to) == (pair[1].from, pair[1].to) {
                return Err(Error::invalid(format!(
                    "candidate {} -> {} occurs more than once",
                    pair[0].from, pair[0].to
                )));
            }
        }
        for edge in &edges {
            if !edge.distance.is_finite() || edge.distance < 0.0 {
                return Err(Error::invalid(format!(
                    "candidate {} -> {} has invalid distance {}",
                    edge.from, edge.to, edge.distance
                )));
            }
        }
        Ok(Self { edges })
    }

    pub fn edges(&self) -> &[Candidate] {
        &self.edges
    }

    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Attach pixel overlap to every candidate leaving frame `from_t`.
    ///
    /// `counts` must cover the whole of both frames: a candidate whose pair is
    /// absent from it gets `shared = 0` and `iou = 0`, which is only true if
    /// nothing was left out.
    pub fn attach_overlap(&mut self, from_t: u64, counts: &OverlapCounts) -> Result<()> {
        for edge in self.edges.iter_mut().filter(|e| e.from_t == from_t) {
            let size_a = *counts.sizes_a.get(&edge.from).ok_or_else(|| {
                Error::invalid(format!(
                    "overlap counts for frame {from_t} have no pixels for detection {}",
                    edge.from
                ))
            })?;
            let size_b = *counts.sizes_b.get(&edge.to).ok_or_else(|| {
                Error::invalid(format!(
                    "overlap counts for frame {} have no pixels for detection {}",
                    from_t + edge.dt,
                    edge.to
                ))
            })?;
            let shared = counts
                .shared
                .get(&(edge.from, edge.to))
                .copied()
                .unwrap_or(0);
            let union = size_a + size_b - shared;
            let iou = if union == 0 {
                0.0
            } else {
                shared as f64 / union as f64
            };
            edge.overlap = Some(OverlapFeatures { shared, iou });
        }
        Ok(())
    }
}

/// Gate every adjacent acquired frame pair of `detections`.
///
/// Pairs separated by a missing frame index produce no edges (version 1 never
/// links across a gap).
pub fn generate_candidates(
    detections: &DetectionSet,
    params: &CandidateParams,
) -> Result<CandidateSet> {
    params.validate()?;
    let mut edges = Vec::new();
    for (from_frame, to_frame) in detections.adjacent_pairs() {
        let gate = params.gate(&from_frame, &to_frame)?;
        if !(gate > 0.0) {
            continue;
        }
        let elapsed = match (from_frame.elapsed_seconds, to_frame.elapsed_seconds) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        };
        let sources = detections.in_frame(from_frame.t_index);
        let targets = detections.in_frame(to_frame.t_index);
        if sources.is_empty() || targets.is_empty() {
            continue;
        }
        let cell_of = |p: [f64; 3]| -> [i64; 3] {
            [
                (p[0] / gate).floor() as i64,
                (p[1] / gate).floor() as i64,
                (p[2] / gate).floor() as i64,
            ]
        };
        let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for (index, target) in targets.iter().enumerate() {
            grid.entry(cell_of(target.registered))
                .or_default()
                .push(index);
        }
        for source in sources {
            let cell = cell_of(source.registered);
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let key = [cell[0] + dz, cell[1] + dy, cell[2] + dx];
                        let Some(bucket) = grid.get(&key) else {
                            continue;
                        };
                        for &index in bucket {
                            let target = &targets[index];
                            let distance = distance(source.registered, target.registered);
                            if distance > gate {
                                continue;
                            }
                            let larger = source.size.max(target.size);
                            let relative_size_change = if larger > 0.0 {
                                (target.size - source.size).abs() / larger
                            } else {
                                0.0
                            };
                            edges.push(Candidate {
                                from: source.id,
                                to: target.id,
                                from_t: from_frame.t_index,
                                dt: to_frame.t_index - from_frame.t_index,
                                elapsed_seconds: elapsed,
                                distance,
                                relative_size_change,
                                overlap: None,
                            });
                        }
                    }
                }
            }
        }
    }
    CandidateSet::from_edges(edges)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Pixel counts of two label frames and of every label pair that shares a
/// pixel. Zero is background.
///
/// Mergeable, so a large frame can be counted block by block — each block over
/// a disjoint region — and folded with [`OverlapCounts::merge`]; the sum is
/// independent of the cut.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OverlapCounts {
    pub sizes_a: BTreeMap<u64, u64>,
    pub sizes_b: BTreeMap<u64, u64>,
    pub shared: BTreeMap<(u64, u64), u64>,
}

impl OverlapCounts {
    pub fn merge(&mut self, other: &OverlapCounts) {
        for (k, v) in &other.sizes_a {
            *self.sizes_a.entry(*k).or_default() += v;
        }
        for (k, v) in &other.sizes_b {
            *self.sizes_b.entry(*k).or_default() += v;
        }
        for (k, v) in &other.shared {
            *self.shared.entry(*k).or_default() += v;
        }
    }
}

/// Count `a`, `b` and their overlaps over two same-shaped label regions —
/// frame `t` and frame `t + 1`, in the same coordinates. Overlap compares
/// native pixels, so it is only meaningful when the two frames are not shifted
/// relative to each other; a caller with drift correction must not use it.
pub fn overlap_counts(a: ArrayView3<'_, u64>, b: ArrayView3<'_, u64>) -> Result<OverlapCounts> {
    if a.shape() != b.shape() {
        return Err(Error::invalid(format!(
            "overlap needs equal shapes, got {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let mut counts = OverlapCounts::default();
    for (&la, &lb) in a.iter().zip(b.iter()) {
        if la != 0 {
            *counts.sizes_a.entry(la).or_default() += 1;
        }
        if lb != 0 {
            *counts.sizes_b.entry(lb).or_default() += 1;
        }
        if la != 0 && lb != 0 {
            *counts.shared.entry((la, lb)).or_default() += 1;
        }
    }
    Ok(counts)
}
