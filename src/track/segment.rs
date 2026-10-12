// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! The second linker: **segment linking** — gap closing, splitting and
//! merging — after frame-to-frame linking.
//!
//! This is the second step of Jaqaman et al. (Nature Methods, 2008), as in
//! TrackMate and `laptrack`. Frame-to-frame linking ([`FrameLapLinker`])
//! produces unbranched segments. One more global assignment then decides,
//! for every segment end and start, whether to
//!
//! * **close a gap** — an end at `t` joins a start at `t + 1 ..= t + max_gap`,
//! * **merge** — an end at `t` joins any detection at `t + 1` (that detection
//!   then has two predecessors),
//! * **split** — a start at `t` is joined from any detection at `t − 1` (that
//!   detection then has two successors),
//! * or stay a true start or end.
//!
//! A division is a split. So is a cell that segmentation cut into two pieces
//! for a frame — and when the pieces rejoin, a merge follows: the lineage
//! shows a small split-then-merge *bubble*. That is deliberate. The graph
//! records what linkage observed, before any interpretation; collapsing
//! bubbles into one cell is a separate, later correction.
//!
//! # The assignment
//!
//! Rows are segment ends and split candidates, columns are segment starts and
//! merge candidates; every row and column has an alternative (track end, no
//! split, track start, no merge). It is solved in the same sparse form as the
//! frame step — see [`assign_sparse`] — with the convention `laptrack` and
//! TrackMate use for the augmented matrix's lower-right block: every chosen
//! link also pays `min(alternative costs)`. With that convention the results
//! are comparable link for link with `laptrack`.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde_json::json;

use super::graph::{LinkKind, TrackGraph, TrackLink};
use super::lap::{assign_sparse, FrameLapLinker, LapParams};
use super::{CandidateSet, Detection, DetectionSet, Linker, LinkerCapabilities};
use crate::error::{Error, Result};

/// Gap closing: an end may join a start up to `max_frame_gap` frame indices
/// later, within `max_distance`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GapClosing {
    pub max_distance: f64,
    pub max_frame_gap: u64,
}

/// Alternative costs of the segment step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentAlternative {
    Fixed {
        track_start: f64,
        track_end: f64,
        no_splitting: f64,
        no_merging: f64,
    },
    /// `factor ×` the `percentile`-th candidate cost (numpy's "lower"
    /// method), the same for all four — `laptrack`'s default with 90 and 1.05.
    Percentile { percentile: f64, factor: f64 },
}

impl Default for SegmentAlternative {
    fn default() -> Self {
        Self::Percentile {
            percentile: 90.0,
            factor: 1.05,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SegmentParams {
    /// The frame-to-frame step.
    pub frame: LapParams,
    pub gap_closing: Option<GapClosing>,
    /// Largest distance from a new segment's first detection to the detection
    /// it splits from, one frame earlier.
    pub splitting: Option<f64>,
    /// Largest distance from a segment's last detection to the detection it
    /// merges into, one frame later.
    pub merging: Option<f64>,
    pub alternative: SegmentAlternative,
}

/// Frame-to-frame linking, then gap closing, splitting and merging.
#[derive(Debug, Clone)]
pub struct SegmentLapLinker {
    params: SegmentParams,
    frame: FrameLapLinker,
}

impl SegmentLapLinker {
    pub fn new(params: SegmentParams) -> Result<Self> {
        let positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(Error::invalid(format!(
                    "{name} must be finite and positive, got {value}"
                )))
            }
        };
        if let Some(gap) = params.gap_closing {
            positive("gap-closing distance", gap.max_distance)?;
            if gap.max_frame_gap == 0 {
                return Err(Error::invalid("gap closing needs max_frame_gap >= 1"));
            }
        }
        if let Some(d) = params.splitting {
            positive("splitting distance", d)?;
        }
        if let Some(d) = params.merging {
            positive("merging distance", d)?;
        }
        match params.alternative {
            SegmentAlternative::Fixed {
                track_start,
                track_end,
                no_splitting,
                no_merging,
            } => {
                for (name, v) in [
                    ("track_start", track_start),
                    ("track_end", track_end),
                    ("no_splitting", no_splitting),
                    ("no_merging", no_merging),
                ] {
                    if !v.is_finite() || v < 0.0 {
                        return Err(Error::invalid(format!(
                            "{name} cost must be finite and non-negative"
                        )));
                    }
                }
            }
            SegmentAlternative::Percentile { percentile, factor } => {
                if !(0.0..=100.0).contains(&percentile) || !(factor > 0.0) {
                    return Err(Error::invalid(
                        "percentile must be in [0, 100] and factor positive",
                    ));
                }
            }
        }
        Ok(Self {
            frame: FrameLapLinker::new(params.frame)?,
            params,
        })
    }
}

impl Linker for SegmentLapLinker {
    fn name(&self) -> &'static str {
        "segment-lap"
    }

    fn capabilities(&self) -> LinkerCapabilities {
        LinkerCapabilities {
            gaps: self.params.gap_closing.is_some_and(|g| g.max_frame_gap > 1),
            divisions: self.params.splitting.is_some(),
            merges: self.params.merging.is_some(),
        }
    }

    fn parameters(&self) -> serde_json::Value {
        let alternative = match self.params.alternative {
            SegmentAlternative::Fixed {
                track_start,
                track_end,
                no_splitting,
                no_merging,
            } => json!({"kind": "fixed", "track_start": track_start, "track_end": track_end,
                         "no_splitting": no_splitting, "no_merging": no_merging}),
            SegmentAlternative::Percentile { percentile, factor } => {
                json!({"kind": "percentile", "percentile": percentile, "factor": factor})
            }
        };
        json!({
            "frame": self.frame.parameters(),
            "gap_closing": self.params.gap_closing.map(|g| json!({"max_distance": g.max_distance, "max_frame_gap": g.max_frame_gap})),
            "splitting_max_distance": self.params.splitting,
            "merging_max_distance": self.params.merging,
            "segment_cost": "distance^2 + min(alternatives) per chosen link (laptrack/TrackMate convention)",
            "alternative": alternative,
        })
    }

    fn link(&self, detections: &DetectionSet, candidates: &CandidateSet) -> Result<TrackGraph> {
        let mut links = self.frame.frame_links(detections, candidates)?;
        let capabilities = self.capabilities();
        let segments = TrackGraph::new(detections, links.clone(), LinkerCapabilities::default())?
            .tracks(detections, 0.0)?;
        let starts: Vec<u64> = segments.iter().map(|s| s.first_id()).collect();
        let ends: Vec<u64> = segments.iter().map(|s| s.last_id()).collect();
        let start_index: HashMap<u64, usize> =
            starts.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        let reach = [
            self.params.gap_closing.map(|g| g.max_distance),
            self.params.splitting,
            self.params.merging,
        ]
        .into_iter()
        .flatten()
        .fold(0.0f64, f64::max);
        if reach == 0.0 {
            TrackGraph::classify(&mut links);
            return TrackGraph::new(detections, links, capabilities);
        }
        let index = FrameIndex::new(detections, reach);
        let get = |id: u64| detections.get(id).expect("segment ids are detections");

        // Candidate edges: (row, column, squared distance).
        enum Row {
            End(usize),
            Split(u64),
        }
        enum Column {
            Start(usize),
            Merge(u64),
        }
        let mut gap_edges: Vec<(usize, usize, f64)> = Vec::new();
        let mut merge_edges: Vec<(usize, u64, f64)> = Vec::new();
        let mut split_edges: Vec<(u64, usize, f64)> = Vec::new();
        for (segment, &end) in ends.iter().enumerate() {
            let e = get(end);
            if let Some(gap) = self.params.gap_closing {
                for dt in 1..=gap.max_frame_gap {
                    for other in index.within(e.t_index + dt, e.registered, gap.max_distance) {
                        if let Some(&j) = start_index.get(&other.id) {
                            gap_edges.push((segment, j, squared(e.registered, other.registered)));
                        }
                    }
                }
            }
            if let Some(d) = self.params.merging {
                for other in index.within(e.t_index + 1, e.registered, d) {
                    merge_edges.push((segment, other.id, squared(e.registered, other.registered)));
                }
            }
        }
        if let Some(d) = self.params.splitting {
            for (segment, &start) in starts.iter().enumerate() {
                let s = get(start);
                if s.t_index == 0 {
                    continue;
                }
                for other in index.within(s.t_index - 1, s.registered, d) {
                    split_edges.push((other.id, segment, squared(s.registered, other.registered)));
                }
            }
        }
        let all_costs: Vec<f64> = gap_edges
            .iter()
            .map(|e| e.2)
            .chain(merge_edges.iter().map(|e| e.2))
            .chain(split_edges.iter().map(|e| e.2))
            .collect();
        if all_costs.is_empty() {
            TrackGraph::classify(&mut links);
            return TrackGraph::new(detections, links, capabilities);
        }
        let (track_start, track_end, no_splitting, no_merging) = match self.params.alternative {
            SegmentAlternative::Fixed {
                track_start,
                track_end,
                no_splitting,
                no_merging,
            } => (track_start, track_end, no_splitting, no_merging),
            SegmentAlternative::Percentile { percentile, factor } => {
                let mut sorted = all_costs.clone();
                sorted.sort_by(f64::total_cmp);
                let rank = ((percentile / 100.0) * (sorted.len() - 1) as f64).floor() as usize;
                let mut cost = sorted[rank] * factor;
                if cost == 0.0 {
                    cost = 1e-6;
                }
                (cost, cost, cost, cost)
            }
        };
        let penalty = track_start.min(track_end).min(no_splitting).min(no_merging);

        // Rows: ends 0..m, then split points; columns: starts 0..m, then merge
        // points; then one private alternative column per row.
        let m = segments.len();
        let split_points: Vec<u64> = split_edges
            .iter()
            .map(|e| e.0)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let merge_points: Vec<u64> = merge_edges
            .iter()
            .map(|e| e.1)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let split_row = |id: u64| m + split_points.binary_search(&id).expect("listed");
        let merge_col = |id: u64| m + merge_points.binary_search(&id).expect("listed");
        let rows_n = m + split_points.len();
        let cols_n = m + merge_points.len();
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); rows_n];
        let mut chosen_of: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut add = |rows: &mut Vec<Vec<(usize, f64)>>,
                       row: usize,
                       col: usize,
                       cost: f64,
                       column_alt: f64| {
            rows[row].push((col, cost + penalty - column_alt));
            chosen_of.insert((row, col), cost);
        };
        for &(row, col, cost) in &gap_edges {
            add(&mut rows, row, col, cost, track_start);
        }
        for &(row, id, cost) in &merge_edges {
            add(&mut rows, row, merge_col(id), cost, no_merging);
        }
        for &(id, col, cost) in &split_edges {
            add(&mut rows, split_row(id), col, cost, track_start);
        }
        for (row, edges) in rows.iter_mut().enumerate() {
            let alternative = if row < m { track_end } else { no_splitting };
            edges.push((cols_n + row, alternative));
        }
        let assignment = assign_sparse(&rows, cols_n + rows_n)?;

        let elapsed = |a: &Detection, b: &Detection| -> Option<f64> {
            match (detections.frame(a.t_index), detections.frame(b.t_index)) {
                (Some(fa), Some(fb)) => match (fa.elapsed_seconds, fb.elapsed_seconds) {
                    (Some(x), Some(y)) => Some(y - x),
                    _ => None,
                },
                _ => None,
            }
        };
        for (row, &col) in assignment.iter().enumerate() {
            if col >= cols_n {
                continue;
            }
            let from = if row < m {
                Row::End(row)
            } else {
                Row::Split(split_points[row - m])
            };
            let to = if col < m {
                Column::Start(col)
            } else {
                Column::Merge(merge_points[col - m])
            };
            let from_id = match from {
                Row::End(segment) => ends[segment],
                Row::Split(id) => id,
            };
            let to_id = match to {
                Column::Start(segment) => starts[segment],
                Column::Merge(id) => id,
            };
            let (a, b) = (get(from_id), get(to_id));
            links.push(TrackLink {
                from: from_id,
                to: to_id,
                dt: b.t_index - a.t_index,
                elapsed_seconds: elapsed(a, b),
                cost: chosen_of[&(row, col)],
                kind: LinkKind::Continue,
            });
        }
        // One edge can be chosen twice: a segment's last detection that is
        // also a split candidate may both merge into the next frame's
        // detection (as an end) and split to it (as a split point), when that
        // detection is a segment start. It is one link. `laptrack` holds its
        // result in a graph, which collapses the duplicate the same way.
        links.sort_by(|a, b| {
            (a.from, a.to)
                .cmp(&(b.from, b.to))
                .then(a.cost.total_cmp(&b.cost))
        });
        links.dedup_by_key(|l| (l.from, l.to));
        TrackGraph::classify(&mut links);
        TrackGraph::new(detections, links, capabilities)
    }
}

fn squared(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Detections per frame on a uniform grid of cell `reach`, for radius queries.
struct FrameIndex<'a> {
    reach: f64,
    frames: HashMap<u64, HashMap<[i64; 3], Vec<&'a Detection>>>,
}

impl<'a> FrameIndex<'a> {
    fn new(detections: &'a DetectionSet, reach: f64) -> Self {
        let mut frames: HashMap<u64, HashMap<[i64; 3], Vec<&'a Detection>>> = HashMap::new();
        for d in detections.detections() {
            let cell = Self::cell(d.registered, reach);
            frames
                .entry(d.t_index)
                .or_default()
                .entry(cell)
                .or_default()
                .push(d);
        }
        Self { reach, frames }
    }

    fn cell(p: [f64; 3], reach: f64) -> [i64; 3] {
        p.map(|v| (v / reach).floor() as i64)
    }

    /// Detections of frame `t` strictly closer than `distance` to `at`, in
    /// `(id)` order.
    fn within(&self, t: u64, at: [f64; 3], distance: f64) -> Vec<&'a Detection> {
        let Some(grid) = self.frames.get(&t) else {
            return Vec::new();
        };
        let c = Self::cell(at, self.reach);
        let mut found = Vec::new();
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    for &d in grid
                        .get(&[c[0] + dz, c[1] + dy, c[2] + dx])
                        .into_iter()
                        .flatten()
                    {
                        if squared(at, d.registered) < distance * distance {
                            found.push(d);
                        }
                    }
                }
            }
        }
        found.sort_by_key(|d| d.id);
        found
    }
}
