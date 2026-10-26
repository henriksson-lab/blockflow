// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! **Linking per-frame detections into tracks.**
//!
//! Rows in, graph out. Nothing here reads a pixel or knows about blocks: a
//! time-lapse is segmented one frame at a time by ordinary spatial plans, and
//! what crosses from one frame to the next is a table of detections. Time is
//! therefore never a grid axis — see `TRACKING_PLAN.md` for why putting it in
//! slot 0 of the rank-3 block was rejected.
//!
//! | piece | what it owns |
//! |---|---|
//! | [`DetectionSet`] | the acquired timeline of one series and its detections, validated |
//! | [`candidates`] | gated candidate edges between consecutive acquired frames, optional overlap features |
//! | [`Linker`] | the extension point: candidates in, [`TrackGraph`] out |
//! | [`lap::FrameLapLinker`] | the version-1 linker: frame-to-frame LAP (Jaqaman et al. 2008, step one) |
//! | [`segment::SegmentLapLinker`] | frame LAP, then gap closing, splitting and merging (step two) |
//! | [`graph`] | the graph, its validation, tracks as maximal unbranched chains, observational events |
//! | [`export`] | detection/track/link rows, CTC `res_track.txt`, napari-style CSV |
//! | [`drift`] | stage drift between frames by phase correlation |
//!
//! # Coordinates
//!
//! A detection has two positions. **Native** is where it sits in the frame's
//! own voxel grid `[z, y, x]` and is what labels and tables address.
//! **Registered** is its position in one physical coordinate space shared by
//! the whole series — native times spacing, plus any drift correction — and is
//! what distances and gating are measured in. Without drift correction the two
//! differ only by spacing. 2-D frames have `z = 0`; nothing in this module
//! assumes 2-D.
//!
//! # Timeline
//!
//! A series is an ordered list of acquired frame indices. A frame with no
//! detections is still a frame; an index that was never acquired is a *gap*.
//! Version 1 never links across a gap: tracks end before it and start after it
//! with [`graph::StartReason::FrameGap`] / [`graph::EndReason::FrameGap`], so a
//! later gap-closing linker can find and rejoin exactly those.

use std::collections::BTreeMap;

use crate::error::{Error, Result};

pub mod candidates;
pub mod drift;
pub mod export;
pub mod graph;
pub mod lap;
pub mod segment;

pub use candidates::{
    generate_candidates, overlap_counts, Candidate, CandidateParams, CandidateSet, OverlapCounts,
};
pub use graph::{
    BorderState, EndReason, LinkKind, StartReason, Track, TrackEvents, TrackGraph, TrackLink,
};
pub use lap::{AlternativeCost, FrameLapLinker, LapParams};
pub use segment::{GapClosing, SegmentAlternative, SegmentLapLinker, SegmentParams};

/// One acquired frame of a series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// Index on the acquisition's time axis. Strictly increasing within a
    /// series; a missing integer is a gap.
    pub t_index: u64,
    /// Time since the series started, when the acquisition records it.
    pub elapsed_seconds: Option<f64>,
}

/// One segmented object in one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Unique within the series. The plate-wide identity is
    /// `(series key, id)`; the number alone is not plate-global.
    pub id: u64,
    pub t_index: u64,
    /// Position in the frame's voxel grid, `[z, y, x]`.
    pub native: [f64; 3],
    /// Position in the series' physical space, `[z, y, x]`.
    pub registered: [f64; 3],
    /// Area (2-D) or volume (3-D), in voxels.
    pub size: f64,
}

/// The geometry a series' detections live in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeriesGeometry {
    /// Frame extent `[Z, Y, X]` in voxels; `Z = 1` for 2-D.
    pub extent: [usize; 3],
    /// Physical size of one voxel `[z, y, x]`.
    pub spacing: [f64; 3],
}

impl SeriesGeometry {
    pub fn new(extent: [usize; 3], spacing: [f64; 3]) -> Result<Self> {
        if extent.contains(&0) {
            return Err(Error::invalid(format!(
                "series extent {extent:?} has an empty axis"
            )));
        }
        if spacing.iter().any(|&s| !s.is_finite() || s <= 0.0) {
            return Err(Error::invalid(format!(
                "series spacing {spacing:?} must be finite and positive"
            )));
        }
        Ok(Self { extent, spacing })
    }

    /// Native voxel position to physical position, without drift correction.
    pub fn physical(&self, native: [f64; 3]) -> [f64; 3] {
        [
            native[0] * self.spacing[0],
            native[1] * self.spacing[1],
            native[2] * self.spacing[2],
        ]
    }

    /// Physical distance from `native` to the nearest frame edge, over the
    /// axes that are not degenerate. A voxel centre at index 0 is half a voxel
    /// from the edge.
    pub fn border_distance(&self, native: [f64; 3]) -> f64 {
        let mut best = f64::INFINITY;
        for axis in 0..3 {
            if self.extent[axis] <= 1 {
                continue;
            }
            let low = native[axis] + 0.5;
            let high = self.extent[axis] as f64 - 0.5 - native[axis];
            best = best.min(low.min(high) * self.spacing[axis]);
        }
        best.max(0.0)
    }
}

/// The detections of one series over its acquired timeline, validated and in
/// canonical order: by `(t_index, id)`.
#[derive(Debug, Clone)]
pub struct DetectionSet {
    geometry: SeriesGeometry,
    frames: Vec<Frame>,
    detections: Vec<Detection>,
    /// `t_index -> range into detections`.
    by_frame: BTreeMap<u64, (usize, usize)>,
    /// `id -> index into detections`.
    by_id: BTreeMap<u64, usize>,
}

impl DetectionSet {
    /// Validate and order. Refuses: no frames, frame indices not strictly
    /// increasing, timestamps present on some frames but not others or not
    /// strictly increasing, a detection in a frame that is not on the timeline,
    /// duplicate ids, and any non-finite or negative value.
    pub fn new(
        geometry: SeriesGeometry,
        frames: Vec<Frame>,
        mut detections: Vec<Detection>,
    ) -> Result<Self> {
        if frames.is_empty() {
            return Err(Error::invalid("a series needs at least one frame"));
        }
        for pair in frames.windows(2) {
            if pair[1].t_index <= pair[0].t_index {
                return Err(Error::invalid(format!(
                    "frame indices must be strictly increasing; {} follows {}",
                    pair[1].t_index, pair[0].t_index
                )));
            }
        }
        let stamped = frames
            .iter()
            .filter(|f| f.elapsed_seconds.is_some())
            .count();
        if stamped != 0 && stamped != frames.len() {
            return Err(Error::invalid(
                "either every frame or no frame may carry elapsed_seconds",
            ));
        }
        if stamped != 0 {
            for frame in &frames {
                let s = frame.elapsed_seconds.expect("checked above");
                if !s.is_finite() || s < 0.0 {
                    return Err(Error::invalid(format!(
                        "frame {} has invalid elapsed_seconds {s}",
                        frame.t_index
                    )));
                }
            }
            for pair in frames.windows(2) {
                if pair[1].elapsed_seconds <= pair[0].elapsed_seconds {
                    return Err(Error::invalid(format!(
                        "elapsed_seconds must increase; frame {} is not after frame {}",
                        pair[1].t_index, pair[0].t_index
                    )));
                }
            }
        }
        detections.sort_by_key(|d| (d.t_index, d.id));
        let mut by_id = BTreeMap::new();
        for (index, detection) in detections.iter().enumerate() {
            let finite = detection
                .native
                .iter()
                .chain(&detection.registered)
                .all(|v| v.is_finite());
            if !finite || !detection.size.is_finite() || detection.size < 0.0 {
                return Err(Error::invalid(format!(
                    "detection {} has a non-finite position or invalid size",
                    detection.id
                )));
            }
            if by_id.insert(detection.id, index).is_some() {
                return Err(Error::invalid(format!(
                    "detection id {} occurs more than once",
                    detection.id
                )));
            }
        }
        let mut by_frame = BTreeMap::new();
        let mut start = 0;
        while start < detections.len() {
            let t = detections[start].t_index;
            let mut end = start;
            while end < detections.len() && detections[end].t_index == t {
                end += 1;
            }
            by_frame.insert(t, (start, end));
            start = end;
        }
        for &t in by_frame.keys() {
            if frames.binary_search_by_key(&t, |f| f.t_index).is_err() {
                return Err(Error::invalid(format!(
                    "detections reference frame {t}, which is not on the series timeline"
                )));
            }
        }
        Ok(Self {
            geometry,
            frames,
            detections,
            by_frame,
            by_id,
        })
    }

    pub fn geometry(&self) -> SeriesGeometry {
        self.geometry
    }

    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }

    /// Every detection, ordered by `(t_index, id)`.
    pub fn detections(&self) -> &[Detection] {
        &self.detections
    }

    /// The detections of one frame, ordered by id. Empty for a frame with no
    /// detections or not on the timeline.
    pub fn in_frame(&self, t_index: u64) -> &[Detection] {
        self.by_frame
            .get(&t_index)
            .map_or(&[][..], |&(start, end)| &self.detections[start..end])
    }

    pub fn get(&self, id: u64) -> Option<&Detection> {
        self.by_id.get(&id).map(|&index| &self.detections[index])
    }

    /// The frame record for `t_index`.
    pub fn frame(&self, t_index: u64) -> Option<&Frame> {
        self.frames
            .binary_search_by_key(&t_index, |f| f.t_index)
            .ok()
            .map(|i| &self.frames[i])
    }

    /// Position of `t_index` on the timeline.
    pub fn frame_position(&self, t_index: u64) -> Option<usize> {
        self.frames
            .binary_search_by_key(&t_index, |f| f.t_index)
            .ok()
    }

    /// Consecutive acquired frame pairs whose indices differ by exactly one —
    /// the pairs version 1 links. A pair separated by a missing index is a gap
    /// and is not returned.
    pub fn adjacent_pairs(&self) -> Vec<(Frame, Frame)> {
        self.frames
            .windows(2)
            .filter(|pair| pair[1].t_index == pair[0].t_index + 1)
            .map(|pair| (pair[0], pair[1]))
            .collect()
    }

    /// Is the frame before `t_index` on the timeline missing — an acquired
    /// frame whose predecessor index was never acquired? `false` for the first
    /// frame.
    pub fn gap_before(&self, t_index: u64) -> bool {
        match self.frame_position(t_index) {
            Some(0) | None => false,
            Some(position) => self.frames[position - 1].t_index + 1 != t_index,
        }
    }

    /// Is the index after `t_index` missing before the next acquired frame?
    /// `false` for the last frame.
    pub fn gap_after(&self, t_index: u64) -> bool {
        match self.frame_position(t_index) {
            None => false,
            Some(position) if position + 1 == self.frames.len() => false,
            Some(position) => self.frames[position + 1].t_index != t_index + 1,
        }
    }
}

/// What graph shapes a linker can produce. The graph validator holds a linker
/// to what it declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LinkerCapabilities {
    /// Links that skip one or more frames (`kind = gap`).
    pub gaps: bool,
    /// One detection linked to more than one successor.
    pub divisions: bool,
    /// More than one detection linked to one successor.
    pub merges: bool,
}

/// The extension point: candidate edges in, a validated track graph out.
///
/// One call per series. Series are independent; a linker never sees two.
pub trait Linker: Send + Sync {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> LinkerCapabilities;
    /// Parameters as JSON, recorded in the output provenance.
    fn parameters(&self) -> serde_json::Value;
    fn link(&self, detections: &DetectionSet, candidates: &CandidateSet) -> Result<TrackGraph>;
}

#[cfg(test)]
mod tests;
