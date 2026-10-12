// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! The track graph: detections are nodes, links are edges. Tracks and events
//! are derived from it, never stored beside it, so they cannot disagree.
//!
//! # Tracks
//!
//! A track is a **maximal unbranched chain**: it continues while its last node
//! has exactly one successor and that successor has exactly one predecessor.
//! A division ends the parent track and starts one track per child; a merge
//! ends every incoming track and starts one. `track_id` is the id of the
//! track's first detection — stable, unique within the series, and needing no
//! counter.
//!
//! # Events are observations
//!
//! A track that ends without a successor has *ended*; whether the cell died,
//! left the field, or was missed by segmentation is not something linkage can
//! tell. So the reasons recorded here are what was observed — the timeline
//! ended, nothing was linked, a frame was missing, the graph branched — plus
//! whether the endpoint was near the frame border, by a recorded physical
//! threshold. Interpretations such as death belong to a classifier with its
//! own evidence.

use std::collections::BTreeMap;

use super::{DetectionSet, LinkerCapabilities};
use crate::error::{Error, Result};

/// What an edge means. The kind is a function of the graph — see
/// [`TrackGraph::classify`] — so it cannot disagree with the node degrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LinkKind {
    /// One-to-one between consecutive frame indices.
    Continue,
    /// Leaves a node with more than one successor: one branch of a division
    /// (or of a segmentation split).
    Divide,
    /// Enters a node with more than one predecessor, from a node that does not
    /// branch: one branch of a merge (or of fragments rejoining).
    Merge,
    /// One-to-one, skipping one or more frame indices.
    Gap,
}

impl LinkKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Continue => "continue",
            Self::Divide => "divide",
            Self::Merge => "merge",
            Self::Gap => "gap",
        }
    }

    pub fn code(self) -> u8 {
        match self {
            Self::Continue => 0,
            Self::Divide => 1,
            Self::Merge => 2,
            Self::Gap => 3,
        }
    }
}

/// One edge.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackLink {
    pub from: u64,
    pub to: u64,
    pub dt: u64,
    pub elapsed_seconds: Option<f64>,
    pub cost: f64,
    pub kind: LinkKind,
}

/// Why a track starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartReason {
    /// The first frame of the series.
    FirstFrame,
    /// Nothing was linked into it.
    Unmatched,
    /// The frame index before it was never acquired.
    FrameGap,
    /// It is one child of a division.
    Division,
    /// It continues from a merge.
    Merge,
}

/// Why a track ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    /// The last frame of the series.
    LastFrame,
    /// Nothing was linked out of it.
    Unmatched,
    /// The frame index after it was never acquired.
    FrameGap,
    /// It divides.
    Division,
    /// It merges into another.
    Merge,
}

impl StartReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FirstFrame => "first_frame",
            Self::Unmatched => "unmatched",
            Self::FrameGap => "frame_gap",
            Self::Division => "division",
            Self::Merge => "merge",
        }
    }
    pub fn code(self) -> u8 {
        match self {
            Self::FirstFrame => 0,
            Self::Unmatched => 1,
            Self::FrameGap => 2,
            Self::Division => 3,
            Self::Merge => 4,
        }
    }
}

impl EndReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LastFrame => "last_frame",
            Self::Unmatched => "unmatched",
            Self::FrameGap => "frame_gap",
            Self::Division => "division",
            Self::Merge => "merge",
        }
    }
    pub fn code(self) -> u8 {
        match self {
            Self::LastFrame => 0,
            Self::Unmatched => 1,
            Self::FrameGap => 2,
            Self::Division => 3,
            Self::Merge => 4,
        }
    }
}

/// Whether an endpoint lies within the border threshold of the frame edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderState {
    NearBorder,
    Interior,
}

impl BorderState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NearBorder => "near_border",
            Self::Interior => "interior",
        }
    }
    pub fn code(self) -> u8 {
        match self {
            Self::NearBorder => 1,
            Self::Interior => 0,
        }
    }
}

/// Observations at a track's two ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackEvents {
    pub start_reason: StartReason,
    pub end_reason: EndReason,
    pub start_border: BorderState,
    pub end_border: BorderState,
}

impl TrackEvents {
    /// A derived annotation: started unmatched near the border.
    pub fn likely_entered_fov(&self) -> bool {
        self.start_reason == StartReason::Unmatched && self.start_border == BorderState::NearBorder
    }

    /// A derived annotation: ended unmatched near the border.
    pub fn likely_left_fov(&self) -> bool {
        self.end_reason == EndReason::Unmatched && self.end_border == BorderState::NearBorder
    }
}

/// One maximal unbranched chain.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub track_id: u64,
    /// The detection ids, in time order.
    pub detections: Vec<u64>,
    pub t_start: u64,
    pub t_end: u64,
    pub parents: Vec<u64>,
    pub children: Vec<u64>,
    pub events: TrackEvents,
}

impl Track {
    pub fn first_id(&self) -> u64 {
        self.detections[0]
    }
    pub fn last_id(&self) -> u64 {
        *self.detections.last().expect("a track has a detection")
    }
}

/// The validated graph of one series.
#[derive(Debug, Clone)]
pub struct TrackGraph {
    capabilities: LinkerCapabilities,
    /// Sorted by `(from, to)`.
    links: Vec<TrackLink>,
    /// `id -> t_index` for every detection, linked or not.
    nodes: BTreeMap<u64, u64>,
    successors: BTreeMap<u64, Vec<u64>>,
    predecessors: BTreeMap<u64, Vec<u64>>,
}

impl TrackGraph {
    /// Validate `links` against `detections` and `capabilities`: both ends
    /// exist, time strictly increases, each kind agrees with its `dt` and with
    /// the node degrees, and degrees do not exceed what the linker declared.
    /// Increasing time makes the graph acyclic.
    pub fn new(
        detections: &DetectionSet,
        mut links: Vec<TrackLink>,
        capabilities: LinkerCapabilities,
    ) -> Result<Self> {
        links.sort_by_key(|l| (l.from, l.to));
        let nodes: BTreeMap<u64, u64> = detections
            .detections()
            .iter()
            .map(|d| (d.id, d.t_index))
            .collect();
        let mut successors: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        let mut predecessors: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for (index, link) in links.iter().enumerate() {
            if index > 0 && (links[index - 1].from, links[index - 1].to) == (link.from, link.to) {
                return Err(Error::invalid(format!(
                    "link {} -> {} occurs more than once",
                    link.from, link.to
                )));
            }
            let (Some(&tf), Some(&tt)) = (nodes.get(&link.from), nodes.get(&link.to)) else {
                return Err(Error::invalid(format!(
                    "link {} -> {} names a detection that does not exist",
                    link.from, link.to
                )));
            };
            if tt <= tf || tt - tf != link.dt {
                return Err(Error::invalid(format!(
                    "link {} -> {} must go forward in time by its dt {} (frames {tf} -> {tt})",
                    link.from, link.to, link.dt
                )));
            }
            if !link.cost.is_finite() {
                return Err(Error::invalid(format!(
                    "link {} -> {} has a non-finite cost",
                    link.from, link.to
                )));
            }
            if link.dt > 1 && !capabilities.gaps {
                return Err(Error::invalid(format!(
                    "link {} -> {} skips frames, from a linker without gap capability",
                    link.from, link.to
                )));
            }
            match link.kind {
                LinkKind::Gap if link.dt < 2 => {
                    return Err(Error::invalid("a gap link must skip at least one frame"))
                }
                LinkKind::Continue if link.dt != 1 => {
                    return Err(Error::invalid(format!(
                        "continue link {} -> {} must join consecutive frames",
                        link.from, link.to
                    )))
                }
                _ => {}
            }
            successors.entry(link.from).or_default().push(link.to);
            predecessors.entry(link.to).or_default().push(link.from);
        }
        for (id, out) in &successors {
            if out.len() > 1 && !capabilities.divisions {
                return Err(Error::invalid(format!(
                    "detection {id} has {} successors from a linker without division capability",
                    out.len()
                )));
            }
        }
        for (id, incoming) in &predecessors {
            if incoming.len() > 1 && !capabilities.merges {
                return Err(Error::invalid(format!(
                    "detection {id} has {} predecessors from a linker without merge capability",
                    incoming.len()
                )));
            }
        }
        for link in &links {
            let branched_out = successors[&link.from].len() > 1;
            let branched_in = predecessors[&link.to].len() > 1;
            let consistent = match link.kind {
                LinkKind::Divide => branched_out,
                LinkKind::Merge => branched_in && !branched_out,
                LinkKind::Continue | LinkKind::Gap => !branched_out && !branched_in,
            };
            if !consistent {
                return Err(Error::invalid(format!(
                    "link {} -> {} is marked {} but the node degrees disagree",
                    link.from,
                    link.to,
                    link.kind.as_str()
                )));
            }
        }
        Ok(Self {
            capabilities,
            links,
            nodes,
            successors,
            predecessors,
        })
    }

    pub fn capabilities(&self) -> LinkerCapabilities {
        self.capabilities
    }

    /// Set every link's kind from the node degrees: `Divide` if its source has
    /// more than one successor, else `Merge` if its target has more than one
    /// predecessor, else `Gap` if it skips frames, else `Continue`.
    pub fn classify(links: &mut [TrackLink]) {
        let mut out: BTreeMap<u64, usize> = BTreeMap::new();
        let mut into: BTreeMap<u64, usize> = BTreeMap::new();
        for link in links.iter() {
            *out.entry(link.from).or_default() += 1;
            *into.entry(link.to).or_default() += 1;
        }
        for link in links.iter_mut() {
            link.kind = if out[&link.from] > 1 {
                LinkKind::Divide
            } else if into[&link.to] > 1 {
                LinkKind::Merge
            } else if link.dt > 1 {
                LinkKind::Gap
            } else {
                LinkKind::Continue
            };
        }
    }

    pub fn links(&self) -> &[TrackLink] {
        &self.links
    }

    pub fn successors(&self, id: u64) -> &[u64] {
        self.successors.get(&id).map_or(&[][..], Vec::as_slice)
    }

    pub fn predecessors(&self, id: u64) -> &[u64] {
        self.predecessors.get(&id).map_or(&[][..], Vec::as_slice)
    }

    /// Every track, ordered by `track_id`, with events. `border_threshold` is
    /// a physical distance; an endpoint closer than it to the frame edge is
    /// [`BorderState::NearBorder`].
    pub fn tracks(&self, detections: &DetectionSet, border_threshold: f64) -> Result<Vec<Track>> {
        if !border_threshold.is_finite() || border_threshold < 0.0 {
            return Err(Error::invalid(format!(
                "border threshold must be finite and non-negative, got {border_threshold}"
            )));
        }
        let continues = |id: u64| -> Option<u64> {
            let out = self.successors(id);
            if out.len() == 1 && self.predecessors(out[0]).len() == 1 {
                Some(out[0])
            } else {
                None
            }
        };
        let starts_track = |id: u64| -> bool {
            let incoming = self.predecessors(id);
            incoming.len() != 1 || self.successors(incoming[0]).len() != 1
        };
        let mut track_of: BTreeMap<u64, u64> = BTreeMap::new();
        let mut chains = Vec::new();
        // Nodes in (t, id) order, so a chain is always met at its start.
        for detection in detections.detections() {
            if !starts_track(detection.id) {
                continue;
            }
            let mut chain = vec![detection.id];
            let mut at = detection.id;
            while let Some(next) = continues(at) {
                chain.push(next);
                at = next;
            }
            for &id in &chain {
                track_of.insert(id, detection.id);
            }
            chains.push(chain);
        }
        if track_of.len() != self.nodes.len() {
            return Err(Error::invalid(
                "track decomposition did not cover every detection",
            ));
        }
        let geometry = detections.geometry();
        let first_t = detections.frames()[0].t_index;
        let last_t = detections
            .frames()
            .last()
            .expect("non-empty timeline")
            .t_index;
        let border = |id: u64| -> BorderState {
            let d = detections.get(id).expect("validated node");
            if geometry.border_distance(d.native) < border_threshold {
                BorderState::NearBorder
            } else {
                BorderState::Interior
            }
        };
        let mut tracks = Vec::with_capacity(chains.len());
        for chain in chains {
            let first = chain[0];
            let last = *chain.last().expect("non-empty chain");
            let t_start = self.nodes[&first];
            let t_end = self.nodes[&last];
            let mut parents: Vec<u64> = self
                .predecessors(first)
                .iter()
                .map(|p| track_of[p])
                .collect();
            let mut children: Vec<u64> =
                self.successors(last).iter().map(|c| track_of[c]).collect();
            parents.sort_unstable();
            parents.dedup();
            children.sort_unstable();
            children.dedup();
            let incoming = self.predecessors(first);
            let start_reason = if incoming.len() > 1 {
                StartReason::Merge
            } else if incoming.len() == 1 {
                // Starts a track with one predecessor only if that predecessor
                // branches.
                StartReason::Division
            } else if t_start == first_t {
                StartReason::FirstFrame
            } else if detections.gap_before(t_start) {
                StartReason::FrameGap
            } else {
                StartReason::Unmatched
            };
            let outgoing = self.successors(last);
            let end_reason = if outgoing.len() > 1 {
                EndReason::Division
            } else if outgoing.len() == 1 {
                EndReason::Merge
            } else if t_end == last_t {
                EndReason::LastFrame
            } else if detections.gap_after(t_end) {
                EndReason::FrameGap
            } else {
                EndReason::Unmatched
            };
            tracks.push(Track {
                track_id: first,
                t_start,
                t_end,
                parents,
                children,
                events: TrackEvents {
                    start_reason,
                    end_reason,
                    start_border: border(first),
                    end_border: border(last),
                },
                detections: chain,
            });
        }
        tracks.sort_by_key(|t| t.track_id);
        Ok(tracks)
    }

    /// `detection id -> track_id` for every detection.
    pub fn track_ids(tracks: &[Track]) -> BTreeMap<u64, u64> {
        let mut map = BTreeMap::new();
        for track in tracks {
            for &id in &track.detections {
                map.insert(id, track.track_id);
            }
        }
        map
    }
}
