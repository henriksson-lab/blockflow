// SPDX-License-Identifier: MIT

//! Run parameters, and the hashes that decide whether a stored frame artifact
//! is still valid.

use std::path::PathBuf;

use blockflow::track::{AlternativeCost, CandidateParams, LapParams};
use blockflow::{Error, Result};
use serde_json::{json, Value};

/// Which network (or stand-in) segments each frame.
#[derive(Clone, Debug, PartialEq)]
pub enum Segmenter {
    /// Connected components above a threshold, on normalised values when a
    /// normalisation is set, raw values otherwise. No model; deterministic.
    Threshold { threshold: f32 },
    /// StarDist 2-D (`--features stardist`).
    Stardist {
        model: PathBuf,
        weights: Option<PathBuf>,
        prob_threshold: Option<f32>,
        nms_threshold: Option<f32>,
    },
    /// Cellpose (`--features cellpose`).
    Cellpose {
        model: PathBuf,
        cuda_device: Option<usize>,
        diameter: Option<f32>,
        flow_threshold: f32,
        cellprob_threshold: f32,
        min_size: i32,
    },
}

/// How intensities are mapped to `[0, 1]` before segmentation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Normalization {
    /// Raw values. Only sensible for the threshold segmenter.
    None,
    /// The same `low`/`high` for every frame of every series.
    Fixed { low: f32, high: f32 },
    /// Percentiles over the whole movie of each series, computed once in a
    /// deterministic prepass and shared by every frame job.
    MoviePercentile { low: f64, high: f64 },
    /// Percentiles of each frame separately; robust to bleaching, not
    /// comparable across frames.
    FramePercentile { low: f64, high: f64 },
}

impl Normalization {
    pub fn to_json(self) -> Value {
        match self {
            Self::None => json!({"kind": "none"}),
            Self::Fixed { low, high } => json!({"kind": "fixed", "low": low, "high": high}),
            Self::MoviePercentile { low, high } => {
                json!({"kind": "movie_percentile", "low_percentile": low, "high_percentile": high})
            }
            Self::FramePercentile { low, high } => {
                json!({"kind": "frame_percentile", "low_percentile": low, "high_percentile": high})
            }
        }
    }
}

/// Everything a run is configured with.
#[derive(Clone, Debug)]
pub struct Params {
    pub segmenter: Segmenter,
    pub normalization: Normalization,
    /// Default channels; a collection manifest may override them per series.
    pub segment_channel: usize,
    pub measure_channel: usize,
    /// Label/table name.
    pub layer: String,
    /// Segmentation halo in pixels; at least the largest cell diameter.
    pub halo: usize,
    /// Candidate square block edges for segmentation and measurement plans.
    pub blocks: Vec<usize>,
    /// `[y, x]` chunk of the label arrays.
    pub label_chunk: usize,
    /// Executor threads inside one frame plan.
    pub block_workers: usize,
    /// Frames processed concurrently, across all series.
    pub frame_jobs: usize,
    /// Series finalised concurrently.
    pub series_jobs: usize,
    pub max_distance: Option<f64>,
    pub max_speed: Option<f64>,
    pub iou_weight: f64,
    pub size_weight: f64,
    /// `None` = 1.05 × the largest candidate link cost of each frame pair.
    pub alternative_cost: Option<f64>,
    /// Segment linking (Jaqaman step two). Any of these enables it.
    /// Gap closing: largest distance, physical units.
    pub gap_closing: Option<f64>,
    /// Gap closing: largest frame-index difference.
    pub max_gap: u64,
    /// Splitting (divisions, segmentation splits): largest distance.
    pub splitting: Option<f64>,
    /// Merging (fragments rejoining, fused cells): largest distance.
    pub merging: Option<f64>,
    /// Fixed alternative cost for the segment step; `None` = 1.05 × the 90th
    /// percentile of its candidate costs.
    pub segment_alternative_cost: Option<f64>,
    /// Physical distance under which a track endpoint is `near_border`.
    pub border_distance: f64,
    /// Estimate stage drift between consecutive frames by phase correlation of
    /// the segmentation channel, and gate links on drift-corrected positions.
    pub drift_correction: bool,
    /// Pyramid level to estimate drift at; `None` picks the finest level
    /// whose larger side is at most 1536 pixels.
    pub drift_level: Option<usize>,
    /// Also write `labels/<layer>-tracks`, pixels relabelled by track id.
    pub track_labels: bool,
    /// Replace existing outputs with the same layer name.
    pub overwrite: bool,
    /// Re-hash every completed frame's label plane on resume.
    pub verify_planes: bool,
    /// Write results under this directory instead of inside each field image.
    pub output: Option<PathBuf>,
    /// Stop after a stage, keeping the work directory (for resumability tests).
    pub stop_after: Option<Stage>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Frames,
}

impl Params {
    pub fn validate(&self) -> Result<()> {
        if self.layer.is_empty() || self.layer.starts_with('.') || self.layer.contains(['/', '\\'])
        {
            return Err(Error::invalid("--layer must be one plain path component"));
        }
        if self.blocks.is_empty() || self.blocks.contains(&0) || self.label_chunk == 0 {
            return Err(Error::invalid(
                "--blocks and --label-chunk must be positive",
            ));
        }
        if let Some(&bad) = self.blocks.iter().find(|&&b| b % self.label_chunk != 0) {
            return Err(Error::invalid(format!(
                "block edge {bad} is not a multiple of --label-chunk {}; frames write straight \
                 into label chunks, and two blocks may not share one",
                self.label_chunk
            )));
        }
        if self.iou_weight > 0.0 && self.max_distance.is_none() && self.max_speed.is_none() {
            return Err(Error::invalid("set --max-distance or --max-speed"));
        }
        if self.drift_correction && self.iou_weight > 0.0 {
            return Err(Error::invalid(
                "--iou-weight compares native pixels and cannot be combined with \
                 --drift-correction until shifted-label overlap exists",
            ));
        }
        if !(self.border_distance.is_finite() && self.border_distance >= 0.0) {
            return Err(Error::invalid(
                "--border-distance must be finite and non-negative",
            ));
        }
        match (&self.segmenter, self.normalization) {
            (Segmenter::Threshold { .. }, _) => {}
            (_, Normalization::None) => {
                return Err(Error::invalid(
                    "a network segmenter needs a normalisation (--normalize fixed|movie|frame)",
                ))
            }
            _ => {}
        }
        if let Normalization::Fixed { low, high } = self.normalization {
            if !(high > low) {
                return Err(Error::invalid("--high must exceed --low"));
            }
        }
        if let Normalization::MoviePercentile { low, high }
        | Normalization::FramePercentile { low, high } = self.normalization
        {
            if !(0.0..=100.0).contains(&low) || !(0.0..=100.0).contains(&high) || high <= low {
                return Err(Error::invalid(
                    "percentiles must satisfy 0 <= low < high <= 100",
                ));
            }
        }
        self.candidate_params()?;
        self.lap_params()?;
        Ok(())
    }

    pub fn candidate_params(&self) -> Result<CandidateParams> {
        if self.max_distance.is_none() && self.max_speed.is_none() {
            return Err(Error::invalid("set --max-distance or --max-speed"));
        }
        Ok(CandidateParams {
            max_distance: self.max_distance,
            max_speed: self.max_speed,
        })
    }

    pub fn lap_params(&self) -> Result<LapParams> {
        Ok(LapParams {
            iou_weight: self.iou_weight,
            size_weight: self.size_weight,
            alternative: match self.alternative_cost {
                Some(cost) => AlternativeCost::Fixed(cost),
                None => AlternativeCost::default(),
            },
        })
    }

    /// The parameters a frame artifact depends on. A change to any of them
    /// invalidates every completed frame; tracking parameters do not.
    pub fn frame_json(&self, segment: usize, measure: usize) -> Value {
        let segmenter = match &self.segmenter {
            Segmenter::Threshold { threshold } => {
                json!({"kind": "threshold", "threshold": threshold})
            }
            Segmenter::Stardist {
                model,
                weights,
                prob_threshold,
                nms_threshold,
            } => json!({
                "kind": "stardist", "model": model, "weights": weights,
                "prob_threshold": prob_threshold, "nms_threshold": nms_threshold
            }),
            Segmenter::Cellpose {
                model,
                diameter,
                flow_threshold,
                cellprob_threshold,
                min_size,
                ..
            } => json!({
                "kind": "cellpose", "model": model, "diameter": diameter,
                "flow_threshold": flow_threshold, "cellprob_threshold": cellprob_threshold,
                "min_size": min_size
            }),
        };
        json!({
            "artifact_version": 1,
            "segmenter": segmenter,
            "normalization": self.normalization.to_json(),
            "segment_channel": segment,
            "measure_channel": measure,
            "halo": self.halo,
            "label_chunk": self.label_chunk,
            "measurements": "shape:standard,intensity:standard",
        })
    }

    /// The linker these parameters select.
    pub fn linker(&self) -> Result<Box<dyn blockflow::track::Linker>> {
        use blockflow::track::{
            FrameLapLinker, GapClosing, SegmentAlternative, SegmentLapLinker, SegmentParams,
        };
        if self.gap_closing.is_none() && self.splitting.is_none() && self.merging.is_none() {
            return Ok(Box::new(FrameLapLinker::new(self.lap_params()?)?));
        }
        Ok(Box::new(SegmentLapLinker::new(SegmentParams {
            frame: self.lap_params()?,
            gap_closing: self.gap_closing.map(|max_distance| GapClosing {
                max_distance,
                max_frame_gap: self.max_gap,
            }),
            splitting: self.splitting,
            merging: self.merging,
            alternative: match self.segment_alternative_cost {
                Some(cost) => SegmentAlternative::Fixed {
                    track_start: cost,
                    track_end: cost,
                    no_splitting: cost,
                    no_merging: cost,
                },
                None => SegmentAlternative::default(),
            },
        })?))
    }

    pub fn tracking_json(&self) -> Value {
        json!({
            "max_distance": self.max_distance,
            "max_speed": self.max_speed,
            "border_distance": self.border_distance,
            "drift_correction": self.drift_correction,
            "drift_level": self.drift_level,
        })
    }
}

/// FNV-1a over bytes: a stable, dependency-free content hash for manifests.
pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

pub fn hash_json(value: &Value) -> String {
    format!("{:016x}", fnv64(value.to_string().as_bytes()))
}
