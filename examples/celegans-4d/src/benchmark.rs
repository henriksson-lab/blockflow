// SPDX-License-Identifier: MIT

//! **Held-out-keyframe benchmark** for nucleus position refinement.
//!
//! A hand-placed Endrov keyframe is the best position truth these recordings
//! have. Remove keyframes, interpolate across the hole with Endrov's own rule
//! ([`crate::endrov::Nucleus::interpolate`], the code the importer uses), ask a
//! [`Refiner`] for the position and radius at the removed frame, and compare
//! with what the annotator placed there.
//!
//! | scenario | removed | evaluated |
//! |---|---|---|
//! | [`Scenario::Single`] | one interior keyframe at a time; everything else stays | every interior keyframe of every cell |
//! | [`Scenario::Stride`]`(s)` | in every cell at once, all but every `s`-th keyframe (the first and last always stay) | every removed keyframe |
//!
//! *Interior* means neither the first nor the last keyframe a cell shows (its
//! *visible* keyframes: inside `[start=, end=]` and before its first child, so
//! shadowed parent keyframes never count). The first keyframe of a cell is its
//! division frame and the last precedes its children's, so division frames
//! are never held out; nothing is extrapolated.
//!
//! The baseline [`Linear`] returns the interpolation unchanged. Every refiner
//! is scored against the same rows, and the linear error is kept beside the
//! refined one in each row, so a refiner's gain is a paired comparison.
//!
//! Outputs (`write_outputs`): `keyframes.csv` (one row per scenario and held-out
//! keyframe), `summary.json` (statistics overall and by gap, annotated cell
//! count, z plane and time) and `report.md`. Everything is computed in a fixed
//! order, so a rerun writes identical files.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use blockflow::time_series::TimeSeriesImage;
use blockflow::{Error, Result};
use ndarray::Array3;
use serde::Serialize;
use serde_json::{json, Value};

use crate::endrov::{Keyframe, LeafEnd, Lineage, Nucleus, Source};
use crate::import::{frame_ids, grid, PixelOrigin, WorldMapping};

/// A nucleus as a sphere in Endrov world µm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Sphere {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub r: f64,
}

impl From<&Keyframe> for Sphere {
    fn from(k: &Keyframe) -> Self {
        Self {
            x: k.x,
            y: k.y,
            z: k.z,
            r: k.r,
        }
    }
}

/// One recording opened for refiners: the image the lineage annotates, its
/// timeline (Endrov frame ids) and the world ↔ voxel mapping of the importer.
pub struct Recording {
    pub image_path: PathBuf,
    pub image: TimeSeriesImage,
    pub mapping: WorldMapping,
    /// Level-0 `[z, y, x]` extent.
    pub extent: [usize; 3],
    /// Endrov frame id (seconds) of each stored time index.
    pub frames: Vec<f64>,
    /// The channel [`Recording::read_box`] reads.
    pub channel: usize,
}

impl Recording {
    pub fn open(image: &Path, origin: PixelOrigin, channel: usize) -> Result<Self> {
        let source = TimeSeriesImage::open(image)?;
        let extent = source.frame_shape(0)?;
        let level0 = grid(image, &source, 0)?;
        let frames = frame_ids(image, source.n_frames(), level0.t.0, level0.t.1)?;
        if channel >= source.n_channels() {
            return Err(Error::invalid(format!(
                "channel {channel} outside the {} channels of {}",
                source.n_channels(),
                image.display()
            )));
        }
        Ok(Self {
            image_path: image.to_path_buf(),
            mapping: WorldMapping::new(&level0, origin),
            image: source,
            extent,
            frames,
            channel,
        })
    }

    /// The stored time index whose Endrov frame id is `frame`.
    pub fn t_index(&self, frame: f64) -> Option<usize> {
        let i = self.frames.partition_point(|&f| f < frame - 1e-6);
        (i < self.frames.len() && (self.frames[i] - frame).abs() <= 1e-6).then_some(i)
    }

    /// Level-0 voxel `[z, y, x]` of a sphere's centre.
    pub fn voxel(&self, s: &Sphere) -> [f64; 3] {
        self.mapping.voxel(s.x, s.y, s.z)
    }

    /// Level-0 voxels of stored frame `t` in the box of half-size `half`
    /// (µm, `[z, y, x]`) around world centre `centre`, clipped to the stack.
    /// Returns the voxels (as `f64`) and the box's first voxel.
    pub fn read_box(
        &self,
        t: usize,
        centre: &Sphere,
        half: [f64; 3],
    ) -> Result<(Array3<f64>, [usize; 3])> {
        let c = self.voxel(centre);
        let mut start = [0usize; 3];
        let mut shape = [0usize; 3];
        for a in 0..3 {
            let reach = (half[a] / self.mapping.scale[a]).ceil();
            let lo = (c[a] - reach).floor().max(0.0) as usize;
            let hi = ((c[a] + reach).ceil().max(0.0) as usize).min(self.extent[a] - 1);
            start[a] = lo.min(self.extent[a] - 1);
            shape[a] = hi.saturating_sub(start[a]) + 1;
        }
        let voxels = self
            .image
            .frame(0, self.channel, t)?
            .read_region(start, shape)?;
        Ok((voxels.widened(), start))
    }
}

/// Another cell at the query's frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Neighbour {
    pub name: String,
    pub sphere: Sphere,
    /// Whether that cell has a keyframe here or is interpolated / held.
    pub source: Source,
}

/// What a refiner is asked.
pub struct Query<'a> {
    pub cell: &'a str,
    /// Endrov frame id of the held-out keyframe.
    pub frame: f64,
    /// Its stored time index, when the frame is on the image timeline.
    pub t_index: Option<usize>,
    /// Endrov linear interpolation from the remaining keyframes.
    pub interpolated: Sphere,
    /// The cell's remaining keyframes (the held-out ones removed).
    pub keyframes: &'a [Keyframe],
    /// The lineage as the refiner may see it: held-out keyframes removed, in
    /// this cell and (for a stride scenario) in every cell.
    pub lineage: &'a Lineage,
    /// Every other cell present at `frame`, positioned from `lineage`.
    pub neighbours: &'a [Neighbour],
}

/// A refiner's answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Refined {
    pub sphere: Sphere,
    pub confidence: Option<f64>,
}

/// A position refinement method: classic DIC refinement (M4), a learned
/// detector (M5), or the [`Linear`] baseline.
pub trait Refiner {
    /// Short, file-name safe.
    fn name(&self) -> &str;
    fn refine(&self, recording: &Recording, query: &Query<'_>) -> Result<Refined>;
}

/// The baseline: Endrov's linear interpolation, unchanged.
pub struct Linear;

impl Refiner for Linear {
    fn name(&self) -> &str {
        "linear"
    }
    fn refine(&self, _: &Recording, query: &Query<'_>) -> Result<Refined> {
        Ok(Refined {
            sphere: query.interpolated,
            confidence: None,
        })
    }
}

/// The baseline with z rounded to the nearest stack plane. Annotators place
/// keyframes on planes only, so this measures how much of the linear z error
/// is the truth's quantisation rather than motion.
pub struct LinearPlaneZ;

impl Refiner for LinearPlaneZ {
    fn name(&self) -> &str {
        "linear-plane-z"
    }
    fn refine(&self, recording: &Recording, query: &Query<'_>) -> Result<Refined> {
        let (scale, translation) = (recording.mapping.scale[0], recording.mapping.translation[0]);
        let mut sphere = query.interpolated;
        sphere.z = ((sphere.z - translation) / scale).round() * scale + translation;
        Ok(Refined {
            sphere,
            confidence: None,
        })
    }
}

/// A reference point below the baseline: the nearer remaining keyframe in
/// time, held (ties go to the earlier one).
pub struct Nearest;

impl Refiner for Nearest {
    fn name(&self) -> &str {
        "nearest"
    }
    fn refine(&self, _: &Recording, query: &Query<'_>) -> Result<Refined> {
        let after = query.keyframes.partition_point(|k| k.frame < query.frame);
        let pick = match (
            after.checked_sub(1).map(|i| &query.keyframes[i]),
            query.keyframes.get(after),
        ) {
            (Some(a), Some(b)) => {
                if query.frame - a.frame <= b.frame - query.frame {
                    a
                } else {
                    b
                }
            }
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => {
                return Err(Error::invalid(format!(
                    "{} has no keyframe left",
                    query.cell
                )))
            }
        };
        Ok(Refined {
            sphere: pick.into(),
            confidence: None,
        })
    }
}

/// Which keyframes are removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Scenario {
    Single,
    Stride(usize),
}

impl Scenario {
    pub fn label(self) -> String {
        match self {
            Self::Single => "single".into(),
            Self::Stride(s) => format!("stride{s}"),
        }
    }
}

/// One held-out keyframe, scored.
#[derive(Debug, Clone, Serialize)]
pub struct Row {
    pub scenario: String,
    pub cell: String,
    pub named: bool,
    pub frame: f64,
    pub t_index: Option<usize>,
    pub key_before: f64,
    pub key_after: f64,
    /// `key_after − key_before` in stored frames (frame-id difference over
    /// the recording's frame interval).
    pub gap_frames: f64,
    /// Where in the gap, 0 at `key_before`, 1 at `key_after`.
    pub gap_fraction: f64,
    /// Annotated cells present at this frame (the full lineage).
    pub cells_at_frame: usize,
    /// z plane of the truth (rounded level-0 voxel z).
    pub z_plane: i64,
    pub truth: Sphere,
    pub refined: Sphere,
    pub confidence: Option<f64>,
    pub interpolated: Sphere,
    pub error: Errors,
    /// The same errors for the linear interpolation.
    pub linear_error: Errors,
}

/// Position and radius errors, µm.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Errors {
    pub d3: f64,
    pub xy: f64,
    /// Signed, refined − truth.
    pub dz: f64,
    /// Signed, refined − truth.
    pub dr: f64,
}

impl Errors {
    fn of(p: &Sphere, truth: &Sphere) -> Self {
        let (dx, dy, dz) = (p.x - truth.x, p.y - truth.y, p.z - truth.z);
        Self {
            d3: (dx * dx + dy * dy + dz * dz).sqrt(),
            xy: (dx * dx + dy * dy).sqrt(),
            dz,
            dr: p.r - truth.r,
        }
    }
}

/// Indices into `nucleus.keyframes` of the keyframes the cell shows.
fn visible(lineage: &Lineage, name: &str) -> Vec<usize> {
    let nucleus = &lineage.nuclei[name];
    let child_first = lineage.child_first_frame(name);
    (0..nucleus.keyframes.len())
        .filter(|&i| {
            nucleus
                .interpolate(nucleus.keyframes[i].frame, child_first, LeafEnd::Hold)
                .is_some_and(|s| s.source == Source::Keyframe)
        })
        .collect()
}

/// Every cell present at `frame` except `except`, from `lineage`.
fn neighbours(lineage: &Lineage, frame: f64, except: &str) -> Vec<Neighbour> {
    lineage
        .nuclei
        .iter()
        .filter(|(name, n)| name.as_str() != except && !n.is_marker())
        .filter_map(|(name, _)| {
            let s = lineage.sample(name, frame, LeafEnd::LastKeyframe)?;
            Some(Neighbour {
                name: name.clone(),
                sphere: Sphere {
                    x: s.x,
                    y: s.y,
                    z: s.z,
                    r: s.r,
                },
                source: s.source,
            })
        })
        .collect()
}

/// Annotation statistics that bear on what the benchmark can resolve.
#[derive(Debug, Clone, Serialize)]
pub struct AnnotationStats {
    pub cells: usize,
    pub keyframes_visible: usize,
    pub keyframes_interior: usize,
    pub frame_interval_seconds: f64,
    /// Visible keyframes per cell: quantiles 0, 0.1, 0.5, 0.9, 1.
    pub keyframes_per_cell: Vec<f64>,
    /// Spacing of consecutive visible keyframes, in stored frames: quantiles
    /// 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1, and a histogram.
    pub spacing_frames_quantiles: Vec<f64>,
    pub spacing_frames_histogram: Vec<(String, usize)>,
    /// Fraction of keyframes whose z lies on a z plane (`|z/resZ − round| < 1e-6`).
    pub z_on_plane: f64,
    pub radius_quantiles: Vec<f64>,
    /// Fraction of consecutive keyframe pairs of one cell with a different radius.
    pub radius_changes: f64,
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let p = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (p.floor() as usize, p.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo as f64)
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(f64::total_cmp);
    v
}

/// Gap bins, in stored frames.
const GAP_BINS: [(f64, f64, &str); 7] = [
    (0.0, 2.5, "2"),
    (2.5, 4.5, "3-4"),
    (4.5, 8.5, "5-8"),
    (8.5, 16.5, "9-16"),
    (16.5, 32.5, "17-32"),
    (32.5, 64.5, "33-64"),
    (64.5, f64::INFINITY, ">64"),
];
const CELL_BINS: [(usize, usize, &str); 8] = [
    (1, 4, "1-4"),
    (5, 8, "5-8"),
    (9, 16, "9-16"),
    (17, 32, "17-32"),
    (33, 64, "33-64"),
    (65, 100, "65-100"),
    (101, 150, "101-150"),
    (151, usize::MAX, ">150"),
];

fn gap_bin(gap: f64) -> &'static str {
    GAP_BINS
        .iter()
        .find(|(lo, hi, _)| gap > *lo && gap <= *hi)
        .map_or(">64", |b| b.2)
}

fn cell_bin(n: usize) -> &'static str {
    CELL_BINS
        .iter()
        .find(|(lo, hi, _)| n >= *lo && n <= *hi)
        .map_or(">150", |b| b.2)
}

pub fn annotation_stats(lineage: &Lineage, recording: &Recording) -> AnnotationStats {
    let dt = frame_interval(&recording.frames);
    let mut per_cell = Vec::new();
    let mut spacing = Vec::new();
    let (mut on_plane, mut total) = (0usize, 0usize);
    let mut radii = Vec::new();
    let (mut r_changed, mut pairs) = (0usize, 0usize);
    let mut interior = 0usize;
    let mut cells = 0usize;
    for (name, nucleus) in &lineage.nuclei {
        if nucleus.is_marker() {
            continue;
        }
        cells += 1;
        let keys: Vec<&Keyframe> = visible(lineage, name)
            .into_iter()
            .map(|i| &nucleus.keyframes[i])
            .collect();
        per_cell.push(keys.len() as f64);
        interior += keys.len().saturating_sub(2);
        for k in &keys {
            total += 1;
            let q = (k.z - recording.mapping.translation[0]) / recording.mapping.scale[0];
            on_plane += usize::from((q - q.round()).abs() < 1e-6);
            radii.push(k.r);
        }
        for w in keys.windows(2) {
            spacing.push((w[1].frame - w[0].frame) / dt);
            pairs += 1;
            r_changed += usize::from(w[1].r != w[0].r);
        }
    }
    let spacing = sorted(spacing);
    let histogram: Vec<(String, usize)> = [(0.0, 1.5, "1")]
        .into_iter()
        .chain([(1.5, 2.5, "2")])
        .chain(GAP_BINS[1..].iter().copied())
        .map(|(lo, hi, label)| {
            (
                label.to_owned(),
                spacing.iter().filter(|&&g| g > lo && g <= hi).count(),
            )
        })
        .collect();
    let per_cell = sorted(per_cell);
    let radii = sorted(radii);
    AnnotationStats {
        cells,
        keyframes_visible: total,
        keyframes_interior: interior,
        frame_interval_seconds: dt,
        keyframes_per_cell: [0.0, 0.1, 0.5, 0.9, 1.0]
            .iter()
            .map(|&q| quantile(&per_cell, q))
            .collect(),
        spacing_frames_quantiles: [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            .iter()
            .map(|&q| quantile(&spacing, q))
            .collect(),
        spacing_frames_histogram: histogram,
        z_on_plane: on_plane as f64 / total.max(1) as f64,
        radius_quantiles: [0.0, 0.1, 0.5, 0.9, 1.0]
            .iter()
            .map(|&q| quantile(&radii, q))
            .collect(),
        radius_changes: r_changed as f64 / pairs.max(1) as f64,
    }
}

/// The recording's frame interval: the median difference of frame ids.
fn frame_interval(frames: &[f64]) -> f64 {
    let d = sorted(frames.windows(2).map(|w| w[1] - w[0]).collect());
    let m = quantile(&d, 0.5);
    if m.is_finite() && m > 0.0 {
        m
    } else {
        1.0
    }
}

/// Run `refiner` over `scenarios`. Rows come in scenario order, then cell
/// name order, then frame order.
pub fn evaluate(
    lineage: &Lineage,
    recording: &Recording,
    refiner: &dyn Refiner,
    scenarios: &[Scenario],
) -> Result<Vec<Row>> {
    let dt = frame_interval(&recording.frames);
    let mut count_cache: BTreeMap<u64, usize> = BTreeMap::new();
    let mut cells_at = |frame: f64| -> usize {
        *count_cache.entry(frame.to_bits()).or_insert_with(|| {
            lineage
                .nuclei
                .iter()
                .filter(|(name, n)| {
                    !n.is_marker() && lineage.sample(name, frame, LeafEnd::LastKeyframe).is_some()
                })
                .count()
        })
    };
    let mut rows = Vec::new();
    for &scenario in scenarios {
        // (cell, held-out keyframe indices) and the lineage with them removed.
        let mut held: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (name, nucleus) in &lineage.nuclei {
            if nucleus.is_marker() {
                continue;
            }
            let vis = visible(lineage, name);
            if vis.len() < 3 {
                continue;
            }
            let interior = &vis[1..vis.len() - 1];
            let removed: Vec<usize> = match scenario {
                Scenario::Single => interior.to_vec(),
                Scenario::Stride(s) => (1..vis.len() - 1)
                    .filter(|j| j % s != 0)
                    .map(|j| vis[j])
                    .collect(),
            };
            if !removed.is_empty() {
                held.insert(name.clone(), removed);
            }
        }
        let thinned = match scenario {
            Scenario::Single => None,
            Scenario::Stride(_) => {
                let mut l = lineage.clone();
                for (name, removed) in &held {
                    let n = l.nuclei.get_mut(name).expect("cell exists");
                    n.keyframes = drop_indices(&lineage.nuclei[name], removed);
                }
                Some(l)
            }
        };
        for (name, removed) in &held {
            let nucleus = &lineage.nuclei[name];
            for &i in removed {
                let truth_key = nucleus.keyframes[i];
                let single;
                let view: &Lineage = match &thinned {
                    Some(l) => l,
                    None => {
                        let mut l = lineage.clone();
                        l.nuclei.get_mut(name).expect("cell exists").keyframes =
                            drop_indices(nucleus, &[i]);
                        single = l;
                        &single
                    }
                };
                let frame = truth_key.frame;
                let sample = view
                    .sample(name, frame, LeafEnd::LastKeyframe)
                    .filter(|s| s.source == Source::Interpolated)
                    .ok_or_else(|| {
                        Error::invalid(format!(
                            "{name} at {frame}: an interior keyframe did not interpolate"
                        ))
                    })?;
                let interpolated = Sphere {
                    x: sample.x,
                    y: sample.y,
                    z: sample.z,
                    r: sample.r,
                };
                let others = neighbours(view, frame, name);
                let query = Query {
                    cell: name,
                    frame,
                    t_index: recording.t_index(frame),
                    interpolated,
                    keyframes: &view.nuclei[name].keyframes,
                    lineage: view,
                    neighbours: &others,
                };
                let refined = refiner.refine(recording, &query)?;
                let truth = Sphere::from(&truth_key);
                let key_after = sample.key_after.expect("interpolated has a key after");
                rows.push(Row {
                    scenario: scenario.label(),
                    cell: name.clone(),
                    named: nucleus.is_named(),
                    frame,
                    t_index: query.t_index,
                    key_before: sample.key_before,
                    key_after,
                    gap_frames: (key_after - sample.key_before) / dt,
                    gap_fraction: (frame - sample.key_before) / (key_after - sample.key_before),
                    cells_at_frame: cells_at(frame),
                    z_plane: recording.voxel(&truth)[0].round() as i64,
                    truth,
                    refined: refined.sphere,
                    confidence: refined.confidence,
                    interpolated,
                    error: Errors::of(&refined.sphere, &truth),
                    linear_error: Errors::of(&interpolated, &truth),
                });
            }
        }
    }
    Ok(rows)
}

fn drop_indices(nucleus: &Nucleus, removed: &[usize]) -> Vec<Keyframe> {
    nucleus
        .keyframes
        .iter()
        .enumerate()
        .filter(|(i, _)| !removed.contains(i))
        .map(|(_, k)| *k)
        .collect()
}

/// Summary statistics of one error.
#[derive(Debug, Clone, Serialize)]
pub struct Stats {
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    pub p90: f64,
    pub max: f64,
}

impl Stats {
    fn of(values: impl Iterator<Item = f64>) -> Self {
        let v = sorted(values.collect());
        Self {
            n: v.len(),
            mean: if v.is_empty() {
                f64::NAN
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            },
            median: quantile(&v, 0.5),
            p90: quantile(&v, 0.9),
            max: v.last().copied().unwrap_or(f64::NAN),
        }
    }
}

/// The statistics of one group of rows.
#[derive(Debug, Clone, Serialize)]
pub struct Group {
    pub n: usize,
    pub d3: Stats,
    pub xy: Stats,
    /// `|dz|`
    pub z: Stats,
    /// `|dr|`
    pub r: Stats,
    /// `d3 / truth radius`
    pub d3_over_radius: Stats,
    /// Mean signed dz and dr (bias).
    pub mean_dz: f64,
    pub mean_dr: f64,
    /// The same rows' linear-interpolation 3D error, and the ratio of the
    /// medians (refined / linear): the M4/M5 target is ≤ 0.5.
    pub linear_d3: Stats,
    pub median_ratio_to_linear: f64,
}

impl Group {
    fn of(rows: &[&Row]) -> Self {
        let mean = |f: &dyn Fn(&Row) -> f64| {
            if rows.is_empty() {
                f64::NAN
            } else {
                rows.iter().map(|r| f(r)).sum::<f64>() / rows.len() as f64
            }
        };
        let d3 = Stats::of(rows.iter().map(|r| r.error.d3));
        let linear_d3 = Stats::of(rows.iter().map(|r| r.linear_error.d3));
        Self {
            n: rows.len(),
            xy: Stats::of(rows.iter().map(|r| r.error.xy)),
            z: Stats::of(rows.iter().map(|r| r.error.dz.abs())),
            r: Stats::of(rows.iter().map(|r| r.error.dr.abs())),
            d3_over_radius: Stats::of(rows.iter().map(|r| r.error.d3 / r.truth.r)),
            mean_dz: mean(&|r| r.error.dz),
            mean_dr: mean(&|r| r.error.dr),
            median_ratio_to_linear: d3.median / linear_d3.median,
            d3,
            linear_d3,
        }
    }
}

fn grouped(rows: &[&Row], key: &dyn Fn(&Row) -> (i64, String)) -> Value {
    let mut groups: BTreeMap<i64, (String, Vec<&Row>)> = BTreeMap::new();
    for &r in rows {
        let (order, label) = key(r);
        groups
            .entry(order)
            .or_insert_with(|| (label, Vec::new()))
            .1
            .push(r);
    }
    Value::Array(
        groups
            .into_values()
            .map(|(label, rows)| json!({"bin": label, "stats": Group::of(&rows)}))
            .collect(),
    )
}

fn gap_order(g: f64) -> i64 {
    GAP_BINS
        .iter()
        .position(|b| g > b.0 && g <= b.1)
        .unwrap_or(GAP_BINS.len()) as i64
}

/// Every scenario's rows pooled and grouped by gap length: the error as a
/// function of the time between the bracketing keyframes. A keyframe held
/// out in several scenarios counts once per scenario.
pub fn pooled_by_gap(rows: &[Row]) -> Value {
    grouped(&rows.iter().collect::<Vec<_>>(), &|r| {
        (gap_order(r.gap_frames), gap_bin(r.gap_frames).to_owned())
    })
}

/// Statistics per scenario: overall and by gap, annotated-cell count, z plane
/// and time (100-frame bins).
pub fn summarize(rows: &[Row]) -> Value {
    let mut by_scenario: BTreeMap<(u8, usize), Vec<&Row>> = BTreeMap::new();
    for r in rows {
        let key = match r.scenario.strip_prefix("stride") {
            Some(s) => (1, s.parse().unwrap_or(0)),
            None => (0, 0),
        };
        by_scenario.entry(key).or_default().push(r);
    }
    let mut out = serde_json::Map::new();
    for rows in by_scenario.values() {
        let label = rows[0].scenario.clone();
        let cell_order = |n: usize| {
            CELL_BINS
                .iter()
                .position(|b| n >= b.0 && n <= b.1)
                .unwrap_or(CELL_BINS.len()) as i64
        };
        out.insert(
            label,
            json!({
                "overall": Group::of(rows),
                "named_cells": Group::of(&rows.iter().copied().filter(|r| r.named).collect::<Vec<_>>()),
                "by_gap_frames": grouped(rows, &|r| (gap_order(r.gap_frames), gap_bin(r.gap_frames).to_owned())),
                "by_cells_at_frame": grouped(rows, &|r| (cell_order(r.cells_at_frame), cell_bin(r.cells_at_frame).to_owned())),
                "by_z_plane": grouped(rows, &|r| (r.z_plane, r.z_plane.to_string())),
                "by_t_index_100": grouped(rows, &|r| {
                    let b = r.t_index.map_or(-1, |t| (t / 100) as i64);
                    (b, if b < 0 { "off-timeline".into() } else { format!("{}-{}", b * 100, b * 100 + 99) })
                }),
            }),
        );
    }
    Value::Object(out)
}

fn fmt(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.4}")
    } else {
        String::new()
    }
}

fn csv(rows: &[Row]) -> String {
    let mut s = String::from(
        "scenario,cell,named,frame,t_index,key_before,key_after,gap_frames,gap_fraction,cells_at_frame,z_plane,\
         truth_x,truth_y,truth_z,truth_r,refined_x,refined_y,refined_z,refined_r,confidence,\
         interp_x,interp_y,interp_z,interp_r,err_3d,err_xy,err_dz,err_dr,linear_err_3d,linear_err_xy,linear_err_dz,linear_err_dr\n",
    );
    for r in rows {
        let cell = if r.cell.contains([',', '"']) {
            format!("\"{}\"", r.cell.replace('"', "\"\""))
        } else {
            r.cell.clone()
        };
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.scenario,
            cell,
            u8::from(r.named),
            r.frame,
            r.t_index.map(|t| t.to_string()).unwrap_or_default(),
            r.key_before,
            r.key_after,
            fmt(r.gap_frames),
            fmt(r.gap_fraction),
            r.cells_at_frame,
            r.z_plane,
            fmt(r.truth.x),
            fmt(r.truth.y),
            fmt(r.truth.z),
            fmt(r.truth.r),
            fmt(r.refined.x),
            fmt(r.refined.y),
            fmt(r.refined.z),
            fmt(r.refined.r),
            r.confidence.map(fmt).unwrap_or_default(),
            fmt(r.interpolated.x),
            fmt(r.interpolated.y),
            fmt(r.interpolated.z),
            fmt(r.interpolated.r),
            fmt(r.error.d3),
            fmt(r.error.xy),
            fmt(r.error.dz),
            fmt(r.error.dr),
            fmt(r.linear_error.d3),
            fmt(r.linear_error.xy),
            fmt(r.linear_error.dz),
            fmt(r.linear_error.dr),
        );
    }
    s
}

/// Round every float in a JSON value to 6 significant decimals, so reports
/// diff cleanly; non-finite numbers become null.
fn rounded(v: Value) -> Value {
    match v {
        Value::Number(n) if n.is_f64() => {
            let f = n.as_f64().unwrap_or(f64::NAN);
            serde_json::Number::from_f64((f * 1e6).round() / 1e6).map_or(Value::Null, Value::Number)
        }
        Value::Array(a) => Value::Array(a.into_iter().map(rounded).collect()),
        Value::Object(o) => Value::Object(o.into_iter().map(|(k, v)| (k, rounded(v))).collect()),
        other => other,
    }
}

fn report(header: &Value, summary: &Value, pooled: &Value) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "# Held-out-keyframe benchmark: {}\n",
        header["recording"].as_str().unwrap_or("")
    );
    let _ = writeln!(s, "Refiner `{}`, lineage `{}`. Errors in µm; `z` is |dz|, `r` is |dr|; `ratio` is the median 3D error over the linear baseline's on the same rows.\n",
        header["refiner"].as_str().unwrap_or(""), header["lineage_id"].as_str().unwrap_or(""));
    let a = &header["annotation"];
    let _ = writeln!(s, "## Annotation\n");
    let _ = writeln!(
        s,
        "- {} cells, {} visible keyframes, {} interior; frame interval {} s.",
        a["cells"], a["keyframes_visible"], a["keyframes_interior"], a["frame_interval_seconds"]
    );
    let _ = writeln!(
        s,
        "- Keyframe spacing (stored frames), quantiles 0/10/25/50/75/90/100 %: {}.",
        a["spacing_frames_quantiles"]
    );
    let _ = writeln!(s, "- Spacing histogram: {}.", a["spacing_frames_histogram"]);
    let _ = writeln!(
        s,
        "- Keyframes per cell, quantiles 0/10/50/90/100 %: {}.",
        a["keyframes_per_cell"]
    );
    let _ = writeln!(s, "- z on a stack plane: {}; radius quantiles 0/10/50/90/100 %: {} µm; radius differs between consecutive keyframes in {} of pairs.\n",
        a["z_on_plane"], a["radius_quantiles"], a["radius_changes"]);
    let row = |label: &str, g: &Value| -> String {
        let st = |k: &str, f: &str| fmt(g[k][f].as_f64().unwrap_or(f64::NAN));
        format!(
            "| {label} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            g["n"],
            st("d3", "median"),
            st("d3", "p90"),
            st("d3", "mean"),
            st("xy", "median"),
            st("xy", "p90"),
            st("z", "median"),
            st("z", "p90"),
            st("r", "median"),
            fmt(g["median_ratio_to_linear"].as_f64().unwrap_or(f64::NAN)),
        )
    };
    let head = "| | n | 3D med | 3D p90 | 3D mean | XY med | XY p90 | z med | z p90 | r med | ratio |\n|---|---|---|---|---|---|---|---|---|---|---|\n";
    let _ = writeln!(
        s,
        "## Overall\n\n{head}{}",
        summary
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, v)| row(k, &v["overall"]))
            .collect::<String>()
    );
    let _ = writeln!(
        s,
        "## All scenarios pooled, by gap (stored frames)\n\n{head}{}",
        pooled
            .as_array()
            .into_iter()
            .flatten()
            .map(|b| row(b["bin"].as_str().unwrap_or(""), &b["stats"]))
            .collect::<String>()
    );
    for (scenario, v) in summary.as_object().into_iter().flatten() {
        for (key, title) in [
            ("by_gap_frames", "gap (stored frames)"),
            ("by_cells_at_frame", "annotated cells present"),
            ("by_z_plane", "z plane"),
            ("by_t_index_100", "time index"),
        ] {
            let _ = writeln!(
                s,
                "## {scenario}: by {title}\n\n{head}{}",
                v[key]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|b| row(b["bin"].as_str().unwrap_or(""), &b["stats"]))
                    .collect::<String>()
            );
        }
    }
    s
}

/// Write `keyframes.csv`, `summary.json` and `report.md` into `out`.
pub fn write_outputs(out: &Path, header: Value, rows: &[Row]) -> Result<Value> {
    fs::create_dir_all(out)
        .map_err(|e| Error::backend(format!("create {}: {e}", out.display())))?;
    let summary = rounded(summarize(rows));
    let header = rounded(header);
    let mut document = header.clone();
    document["scenarios"] = summary.clone();
    let pooled = rounded(pooled_by_gap(rows));
    document["pooled_by_gap_frames"] = pooled.clone();
    let write = |name: &str, bytes: &[u8]| {
        fs::write(out.join(name), bytes)
            .map_err(|e| Error::backend(format!("write {}: {e}", out.join(name).display())))
    };
    write("keyframes.csv", csv(rows).as_bytes())?;
    write(
        "summary.json",
        &serde_json::to_vec_pretty(&document).map_err(Error::backend)?,
    )?;
    write("report.md", report(&header, &summary, &pooled).as_bytes())?;
    Ok(document)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endrov::parse_lineages;

    const XML: &str = r#"<ost><nuclineage id="1"><data>
      <nuc name="P0">
        <pos f="0" x="0" y="0" z="0" r="1" />
        <pos f="10" x="1" y="0" z="0" r="1" />
        <pos f="20" x="4" y="0" z="0" r="1" />
        <pos f="30" x="6" y="0" z="0" r="1" />
        <pos f="40" x="8" y="0" z="3" r="2" />
        <pos f="60" x="9" y="0" z="3" r="2" />
        <child name="AB" />
        <child name="P1" />
      </nuc>
      <nuc name="AB"><pos f="50" x="0" y="0" z="0" r="1" /><pos f="70" x="0" y="0" z="0" r="1" /></nuc>
      <nuc name="P1"><pos f="50" x="0" y="0" z="0" r="1" /></nuc>
    </data></nuclineage></ost>"#;

    fn recording_free_rows(scenarios: &[Scenario]) -> Vec<(String, f64, f64, f64)> {
        // `evaluate` without an image: exercise the selection and interpolation directly.
        let lineage = parse_lineages(XML).unwrap().remove(0);
        let mut out = Vec::new();
        for &scenario in scenarios {
            let vis = visible(&lineage, "P0");
            // f=60 is shadowed by the children at 50: visible = 0, 10, 20, 30, 40.
            assert_eq!(vis, vec![0, 1, 2, 3, 4]);
            let removed: Vec<usize> = match scenario {
                Scenario::Single => vis[1..vis.len() - 1].to_vec(),
                Scenario::Stride(s) => (1..vis.len() - 1)
                    .filter(|j| j % s != 0)
                    .map(|j| vis[j])
                    .collect(),
            };
            for &i in &removed {
                let mut l = lineage.clone();
                let drop: Vec<usize> = if scenario == Scenario::Single {
                    vec![i]
                } else {
                    removed.clone()
                };
                l.nuclei.get_mut("P0").unwrap().keyframes =
                    drop_indices(&lineage.nuclei["P0"], &drop);
                let k = lineage.nuclei["P0"].keyframes[i];
                let s = l.sample("P0", k.frame, LeafEnd::LastKeyframe).unwrap();
                assert_eq!(s.source, Source::Interpolated);
                out.push((
                    scenario.label(),
                    k.frame,
                    s.x,
                    Errors::of(
                        &Sphere {
                            x: s.x,
                            y: s.y,
                            z: s.z,
                            r: s.r,
                        },
                        &(&k).into(),
                    )
                    .d3,
                ));
            }
        }
        out
    }

    #[test]
    fn holdout_selection_and_interpolation() {
        let rows = recording_free_rows(&[Scenario::Single, Scenario::Stride(2)]);
        // Single: 10 → (0+4)/2 = 2 (err 1); 20 → (1+6)/2 = 3.5 (err 0.5); 30 → x (4+8)/2 = 6, z (0+3)/2 = 1.5 (err 1.5).
        let single: Vec<_> = rows
            .iter()
            .filter(|r| r.0 == "single")
            .map(|r| (r.1, r.2, r.3))
            .collect();
        assert_eq!(
            single,
            vec![(10.0, 2.0, 1.0), (20.0, 3.5, 0.5), (30.0, 6.0, 1.5)]
        );
        // Stride 2 keeps 0, 20, 40 and removes 10 and 30.
        let stride: Vec<_> = rows
            .iter()
            .filter(|r| r.0 == "stride2")
            .map(|r| (r.1, r.2))
            .collect();
        assert_eq!(stride, vec![(10.0, 2.0), (30.0, 6.0)]);
    }

    #[test]
    fn stats_and_bins() {
        let s = Stats::of([1.0, 2.0, 3.0, 4.0].into_iter());
        assert_eq!((s.median, s.max), (2.5, 4.0));
        assert!((s.p90 - 3.7).abs() < 1e-12);
        assert_eq!(
            (gap_bin(2.0), gap_bin(3.0), gap_bin(70.0)),
            ("2", "3-4", ">64")
        );
        assert_eq!(
            (cell_bin(4), cell_bin(100), cell_bin(151)),
            ("1-4", "65-100", ">150")
        );
    }
}
