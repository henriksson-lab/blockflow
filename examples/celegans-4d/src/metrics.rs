// SPDX-License-Identifier: MIT

//! **Tracking metrics** against a hand-annotated lineage.
//!
//! Both sides are tracking tables in the `plate-timelapse-tracking` format —
//! the importer's ground truth and a tracker's output — read into a
//! [`Tracking`]: detections with a physical position (µm, from the level-0
//! voxel columns and the image transform), links, and optional track names
//! (`tracking.names` on the tracks table).
//!
//! | metric | definition |
//! |---|---|
//! | detection P / R / F1 | per frame, a minimum-total-distance one-to-one matching of truth to predicted detections closer than `gate` µm (3-D) — Blockflow's own LAP solver, [`blockflow::track::assign_sparse`], each truth detection also offered "unmatched" at cost `gate` |
//! | division P / R / F1 | division = a detection with ≥ 2 outgoing links; time = its children's first frame, position = the mother detection. Matched one-to-one if `|Δt| ≤ division_tolerance` frames and the mothers are within `gate` |
//! | link recall / precision | a truth link whose two ends both matched is recovered when the predicted graph links the two matched detections directly; a predicted link whose ends both matched is correct when truth links the two |
//! | naming accuracy | over truth detections of named cells (not Endrov's `:N`): matched to a prediction whose track carries the same name. `of_matched` divides by the matched ones, `end_to_end` by all |
//!
//! # Partial ground truth
//!
//! An annotator may follow only some cells (ceh37 names 66 of 147 nuclei and
//! the embryo has more cells than are annotated). A prediction on an
//! unannotated cell is then not a false positive. [`Coverage`] says how much
//! of the embryo the truth covers:
//!
//! - `Full`: every unmatched prediction is a false positive;
//! - `Partial { radius }`: an unmatched prediction is a false positive only
//!   when it lies within `radius` µm of a truth detection of the same frame
//!   (a duplicate or a displaced detection of an annotated nucleus); others
//!   are counted as `unscored`. The same rule applies to unmatched predicted
//!   divisions (radius around the truth detections of the division frame or
//!   the one before). Precision then means "precision around annotated cells";
//! - `RecallOnly`: no false positives; precision and F1 are not reported.
//!
//! In every mode only frames where the truth has detections are scored
//! (optionally narrowed by `t_range`); predictions on other frames are
//! ignored. Truth detections the importer clamped into the stack
//! (`outside_stack = 1`) are dropped, with their links.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use blockflow::object_table::TableReader;
use blockflow::time_series::TimeSeriesImage;
use blockflow::track::assign_sparse;
use blockflow::{Error, Result};
use serde::Serialize;
use serde_json::Value;

use crate::import::grid;

/// One detection with its physical position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub id: u64,
    pub track: u64,
    pub t: u64,
    /// `[z, y, x]` µm: level-0 voxel · scale + translation.
    pub pos: [f64; 3],
}

/// A tracking result or a ground truth.
#[derive(Debug, Clone, Default)]
pub struct Tracking {
    pub points: Vec<Point>,
    /// `(from, to)` detection ids.
    pub links: Vec<(u64, u64)>,
    /// Track id → name, when the producer names cells.
    pub names: BTreeMap<u64, String>,
}

/// A division event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Division {
    pub mother: u64,
    /// The earliest first frame of the children.
    pub t: u64,
    pub pos: [f64; 3],
}

fn context<E: std::fmt::Display>(path: &Path) -> impl Fn(E) -> Error + '_ {
    move |e| Error::backend(format!("{}: {e}", path.display()))
}

fn read_table(path: &Path) -> Result<TableReader> {
    TableReader::open(path).map_err(context(path))
}

fn resolve(base: &Path, relative: &str) -> PathBuf {
    let mut out = base.to_path_buf();
    for part in relative.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            p => out.push(p),
        }
    }
    out
}

impl Tracking {
    /// Read a detections table plus the tracks and links tables its
    /// `tracking` extension names. `image` defaults to the table's
    /// coordinate space (`../../`, the image it lives in) and supplies the
    /// level-0 scale and translation.
    pub fn read(detections: &Path, image: Option<&Path>) -> Result<Self> {
        let table = read_table(detections)?;
        let spec = table.spec().clone();
        let image = match image {
            Some(i) => i.to_path_buf(),
            None => resolve(detections, &spec.coordinate_space),
        };
        let level0 = grid(&image, &TimeSeriesImage::open(&image)?, 0)?;
        let n = spec.row_count;
        let has = |name: &str| spec.columns.iter().any(|c| c.name == name);
        let ids = table
            .read_u64(&spec.identity, 0..n)
            .map_err(context(detections))?;
        let tracks = table
            .read_u64("track_id", 0..n)
            .map_err(context(detections))?;
        let ts = table
            .read_u64("t_index", 0..n)
            .map_err(context(detections))?;
        let f64s = |name: &str| table.read_f64(name, 0..n).map_err(context(detections));
        let (y, x) = (f64s("y")?, f64s("x")?);
        let z = if has("z") {
            f64s("z")?
        } else {
            vec![0.0; n as usize]
        };
        let outside = if has("outside_stack") {
            table
                .read_u8("outside_stack", 0..n)
                .map_err(context(detections))?
        } else {
            vec![0; n as usize]
        };
        let mut points = Vec::with_capacity(n as usize);
        let mut dropped = BTreeSet::new();
        for i in 0..n as usize {
            if outside[i] != 0 {
                dropped.insert(ids[i]);
                continue;
            }
            let voxel = [z[i], y[i], x[i]];
            points.push(Point {
                id: ids[i],
                track: tracks[i],
                t: ts[i],
                pos: std::array::from_fn(|a| voxel[a] * level0.scale[a] + level0.translation[a]),
            });
        }
        let tracking = spec.extensions.get("tracking");
        let related = |key: &str| {
            tracking
                .and_then(|t| t.get(key))
                .and_then(Value::as_str)
                .map(|r| resolve(detections, r))
        };
        let mut links = Vec::new();
        if let Some(path) = related("links").filter(|p| p.join("zarr.json").is_file()) {
            let t = read_table(&path)?;
            let m = t.spec().row_count;
            let from = t.read_u64("from_id", 0..m).map_err(context(&path))?;
            let to = t.read_u64("to_id", 0..m).map_err(context(&path))?;
            links = from
                .into_iter()
                .zip(to)
                .filter(|(a, b)| !dropped.contains(a) && !dropped.contains(b))
                .collect();
        }
        let mut names = BTreeMap::new();
        if let Some(path) = related("tracks").filter(|p| p.join("zarr.json").is_file()) {
            let t = read_table(&path)?;
            if let Some(map) = t
                .spec()
                .extensions
                .get("tracking")
                .and_then(|x| x.get("names"))
                .and_then(Value::as_object)
            {
                for (k, v) in map {
                    if let (Ok(id), Some(name)) = (k.parse::<u64>(), v.as_str()) {
                        names.insert(id, name.to_owned());
                    }
                }
            }
        }
        Ok(Self {
            points,
            links,
            names,
        })
    }

    /// Detections with ≥ 2 outgoing links, in mother-id order.
    pub fn divisions(&self) -> Vec<Division> {
        let by_id: BTreeMap<u64, &Point> = self.points.iter().map(|p| (p.id, p)).collect();
        let mut children: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for &(a, b) in &self.links {
            children.entry(a).or_default().push(b);
        }
        children
            .into_iter()
            .filter(|(_, c)| c.len() >= 2)
            .filter_map(|(m, c)| {
                let mother = by_id.get(&m)?;
                let t = c.iter().filter_map(|id| by_id.get(id).map(|p| p.t)).min()?;
                Some(Division {
                    mother: m,
                    t,
                    pos: mother.pos,
                })
            })
            .collect()
    }

    /// Every position moved by isotropic Gaussian noise of `sigma` µm per
    /// axis (deterministic in `seed`).
    pub fn jittered(&self, sigma: f64, seed: u64) -> Self {
        let mut rng = SplitMix(seed);
        let mut out = self.clone();
        for p in &mut out.points {
            for a in 0..3 {
                p.pos[a] += sigma * rng.normal();
            }
        }
        out
    }

    /// Every `every`-th division (1 = all) undone by removing the mother's
    /// outgoing links: the children become new roots.
    pub fn without_divisions(&self, every: usize) -> Self {
        let mothers: BTreeSet<u64> = self
            .divisions()
            .iter()
            .enumerate()
            .filter(|(i, _)| i % every.max(1) == 0)
            .map(|(_, d)| d.mother)
            .collect();
        let mut out = self.clone();
        out.links.retain(|(a, _)| !mothers.contains(a));
        out
    }

    /// Only the detections (and links between them) of tracks `keep` accepts.
    pub fn restricted(&self, keep: impl Fn(u64) -> bool) -> Self {
        let points: Vec<Point> = self
            .points
            .iter()
            .copied()
            .filter(|p| keep(p.track))
            .collect();
        let ids: BTreeSet<u64> = points.iter().map(|p| p.id).collect();
        Self {
            links: self
                .links
                .iter()
                .copied()
                .filter(|(a, b)| ids.contains(a) && ids.contains(b))
                .collect(),
            names: self
                .names
                .iter()
                .filter(|(t, _)| keep(**t))
                .map(|(t, n)| (*t, n.clone()))
                .collect(),
            points,
        }
    }

    /// Track id → the root track of its lineage tree (following links
    /// between tracks upward).
    pub fn lineage_roots(&self) -> BTreeMap<u64, u64> {
        let track_of: BTreeMap<u64, u64> = self.points.iter().map(|p| (p.id, p.track)).collect();
        let mut parent: BTreeMap<u64, u64> = BTreeMap::new();
        for (a, b) in &self.links {
            if let (Some(&ta), Some(&tb)) = (track_of.get(a), track_of.get(b)) {
                if ta != tb {
                    parent.insert(tb, ta);
                }
            }
        }
        track_of
            .values()
            .map(|&t| {
                let mut root = t;
                let mut steps = 0;
                while let Some(&p) = parent.get(&root) {
                    root = p;
                    steps += 1;
                    if steps > parent.len() {
                        break;
                    }
                }
                (t, root)
            })
            .collect()
    }

    /// Every `every`-th detection dropped (and its links).
    pub fn thinned(&self, every: usize) -> Self {
        let points: Vec<Point> = self
            .points
            .iter()
            .enumerate()
            .filter(|(i, _)| i % every.max(1) != 0)
            .map(|(_, p)| *p)
            .collect();
        let ids: BTreeSet<u64> = points.iter().map(|p| p.id).collect();
        Self {
            links: self
                .links
                .iter()
                .copied()
                .filter(|(a, b)| ids.contains(a) && ids.contains(b))
                .collect(),
            names: self.names.clone(),
            points,
        }
    }

    /// Names of sibling tracks swapped (children of one division exchange
    /// names), as a naming error that keeps the topology.
    pub fn siblings_swapped(&self) -> Self {
        let track_of: BTreeMap<u64, u64> = self.points.iter().map(|p| (p.id, p.track)).collect();
        let mut out = self.clone();
        let mut children: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for &(a, b) in &self.links {
            children.entry(a).or_default().push(b);
        }
        for c in children.values().filter(|c| c.len() == 2) {
            let (Some(&ta), Some(&tb)) = (track_of.get(&c[0]), track_of.get(&c[1])) else {
                continue;
            };
            if let (Some(na), Some(nb)) = (self.names.get(&ta), self.names.get(&tb)) {
                out.names.insert(ta, nb.clone());
                out.names.insert(tb, na.clone());
            }
        }
        out
    }
}

/// Deterministic generator for the sanity perturbations.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// How much of the embryo the truth annotates (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Coverage {
    Full,
    Partial { radius: f64 },
    RecallOnly,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ScoreParams {
    /// Matching gate, µm (3-D).
    pub gate: f64,
    /// Division time tolerance, frames.
    pub division_tolerance: u64,
    pub coverage: Coverage,
    /// Inclusive time-index range to score.
    pub t_range: Option<(u64, u64)>,
}

impl Default for ScoreParams {
    fn default() -> Self {
        Self {
            gate: 3.0,
            division_tolerance: 1,
            coverage: Coverage::Full,
            t_range: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Counts {
    pub tp: usize,
    pub fp: usize,
    #[serde(rename = "fn")]
    pub fn_: usize,
    /// Unmatched predictions not counted as false positives (`Partial`, `RecallOnly`).
    pub unscored: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

impl Counts {
    fn new(tp: usize, fp: usize, fn_: usize, unscored: usize, precision_defined: bool) -> Self {
        let ratio = |a: usize, b: usize| (b > 0).then(|| a as f64 / b as f64);
        let precision = if precision_defined {
            ratio(tp, tp + fp)
        } else {
            None
        };
        let recall = ratio(tp, tp + fn_);
        let f1 = match (precision, recall) {
            (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
            (Some(_), Some(_)) => Some(0.0),
            _ => None,
        };
        Self {
            tp,
            fp,
            fn_,
            unscored,
            precision,
            recall,
            f1,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Naming {
    /// Truth detections of named cells, in scored frames.
    pub named_truth: usize,
    pub matched: usize,
    pub correct: usize,
    pub of_matched: Option<f64>,
    pub end_to_end: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Links {
    pub truth_links: usize,
    pub truth_links_matched_ends: usize,
    pub recovered: usize,
    pub recall: Option<f64>,
    pub predicted_links_matched_ends: usize,
    pub correct: usize,
    pub precision: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Scores {
    pub params: ScoreParams,
    pub frames_scored: usize,
    pub truth_detections: usize,
    pub predicted_detections: usize,
    pub detection: Counts,
    /// Mean 3-D distance of matched pairs, µm.
    pub matched_distance_mean: Option<f64>,
    pub division: Counts,
    pub division_dt_mean: Option<f64>,
    pub links: Links,
    /// `None` when the prediction carries no names.
    pub naming: Option<Naming>,
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Minimum-cost matching of `rows` to `cols` over `edges(i) -> [(j, cost)]`,
/// each row also offered "unmatched" at `alternative`. Returns
/// `row -> Some(col)`.
fn matching(
    n_rows: usize,
    n_cols: usize,
    alternative: f64,
    edges: impl Fn(usize) -> Vec<(usize, f64)>,
) -> Result<Vec<Option<usize>>> {
    let rows: Vec<Vec<(usize, f64)>> = (0..n_rows)
        .map(|i| {
            let mut e = edges(i);
            e.push((n_cols + i, alternative));
            e
        })
        .collect();
    let assignment = assign_sparse(&rows, n_cols + n_rows)?;
    Ok(assignment
        .into_iter()
        .map(|c| (c < n_cols).then_some(c))
        .collect())
}

fn is_named(name: &str) -> bool {
    !name.starts_with(':')
}

/// Score `predicted` against `truth`.
pub fn score(truth: &Tracking, predicted: &Tracking, params: &ScoreParams) -> Result<Scores> {
    let in_range = |t: u64| params.t_range.is_none_or(|(a, b)| t >= a && t <= b);
    let mut truth_by_t: BTreeMap<u64, Vec<&Point>> = BTreeMap::new();
    for p in truth.points.iter().filter(|p| in_range(p.t)) {
        truth_by_t.entry(p.t).or_default().push(p);
    }
    let mut pred_by_t: BTreeMap<u64, Vec<&Point>> = BTreeMap::new();
    for p in predicted
        .points
        .iter()
        .filter(|p| truth_by_t.contains_key(&p.t))
    {
        pred_by_t.entry(p.t).or_default().push(p);
    }
    let near_truth = |t: u64, pos: &[f64; 3], radius: f64| {
        truth_by_t
            .get(&t)
            .is_some_and(|v| v.iter().any(|q| distance(&q.pos, pos) <= radius))
    };
    let precision_defined = params.coverage != Coverage::RecallOnly;
    let fp_rule = |t: u64, pos: &[f64; 3]| match params.coverage {
        Coverage::Full => true,
        Coverage::Partial { radius } => near_truth(t, pos, radius),
        Coverage::RecallOnly => false,
    };

    // Detections.
    let mut matched: BTreeMap<u64, u64> = BTreeMap::new(); // truth id -> predicted id
    let (mut tp, mut fp, mut fn_, mut unscored) = (0, 0, 0, 0);
    let mut distances = Vec::new();
    let empty = Vec::new();
    for (&t, gt) in &truth_by_t {
        let pr = pred_by_t.get(&t).unwrap_or(&empty);
        let assignment = matching(gt.len(), pr.len(), params.gate, |i| {
            pr.iter()
                .enumerate()
                .filter_map(|(j, q)| {
                    let d = distance(&gt[i].pos, &q.pos);
                    (d < params.gate).then_some((j, d))
                })
                .collect()
        })?;
        let mut used = vec![false; pr.len()];
        for (i, a) in assignment.iter().enumerate() {
            match a {
                Some(j) => {
                    used[*j] = true;
                    tp += 1;
                    distances.push(distance(&gt[i].pos, &pr[*j].pos));
                    matched.insert(gt[i].id, pr[*j].id);
                }
                None => fn_ += 1,
            }
        }
        for (_, q) in pr.iter().enumerate().filter(|(j, _)| !used[*j]) {
            if fp_rule(t, &q.pos) {
                fp += 1;
            } else {
                unscored += 1;
            }
        }
    }
    let mean = |v: &[f64]| (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64);

    // Divisions.
    let gt_div: Vec<Division> = truth
        .divisions()
        .into_iter()
        .filter(|d| truth_by_t.contains_key(&d.t))
        .collect();
    let pr_div: Vec<Division> = predicted
        .divisions()
        .into_iter()
        .filter(|d| {
            (d.t.saturating_sub(params.division_tolerance)..=d.t + params.division_tolerance)
                .any(|t| truth_by_t.contains_key(&t))
        })
        .collect();
    let assignment = matching(
        gt_div.len(),
        pr_div.len(),
        params.gate + params.division_tolerance as f64 + 1.0,
        |i| {
            pr_div
                .iter()
                .enumerate()
                .filter_map(|(j, q)| {
                    let dt = gt_div[i].t.abs_diff(q.t);
                    let d = distance(&gt_div[i].pos, &q.pos);
                    (dt <= params.division_tolerance && d < params.gate)
                        .then_some((j, d + dt as f64))
                })
                .collect()
        },
    )?;
    let mut used = vec![false; pr_div.len()];
    let (mut dtp, mut dfn) = (0, 0);
    let mut dts = Vec::new();
    for (i, a) in assignment.iter().enumerate() {
        match a {
            Some(j) => {
                used[*j] = true;
                dtp += 1;
                dts.push(gt_div[i].t.abs_diff(pr_div[*j].t) as f64);
            }
            None => dfn += 1,
        }
    }
    let (mut dfp, mut dun) = (0, 0);
    for (j, q) in pr_div.iter().enumerate() {
        if used[j] {
            continue;
        }
        let scored = match params.coverage {
            Coverage::Full => true,
            Coverage::Partial { radius } => {
                near_truth(q.t, &q.pos, radius) || near_truth(q.t.saturating_sub(1), &q.pos, radius)
            }
            Coverage::RecallOnly => false,
        };
        if scored {
            dfp += 1;
        } else {
            dun += 1;
        }
    }

    // Links.
    let reverse: BTreeMap<u64, u64> = matched.iter().map(|(g, p)| (*p, *g)).collect();
    let pred_links: BTreeSet<(u64, u64)> = predicted.links.iter().copied().collect();
    let truth_links: BTreeSet<(u64, u64)> = truth.links.iter().copied().collect();
    let truth_t: BTreeMap<u64, u64> = truth.points.iter().map(|p| (p.id, p.t)).collect();
    let scored_truth_links: Vec<(u64, u64)> = truth_links
        .iter()
        .copied()
        .filter(|(a, b)| {
            [a, b]
                .iter()
                .all(|id| truth_t.get(id).is_some_and(|t| truth_by_t.contains_key(t)))
        })
        .collect();
    let with_ends: Vec<(u64, u64)> = scored_truth_links
        .iter()
        .filter_map(|(a, b)| Some((*matched.get(a)?, *matched.get(b)?)))
        .collect();
    let recovered = with_ends.iter().filter(|l| pred_links.contains(l)).count();
    let pred_with_ends: Vec<(u64, u64)> = pred_links
        .iter()
        .filter_map(|(a, b)| Some((*reverse.get(a)?, *reverse.get(b)?)))
        .collect();
    let correct = pred_with_ends
        .iter()
        .filter(|l| truth_links.contains(l))
        .count();
    let ratio = |a: usize, b: usize| (b > 0).then(|| a as f64 / b as f64);

    // Naming.
    let naming = (!predicted.names.is_empty()).then(|| {
        let pred_track: BTreeMap<u64, u64> =
            predicted.points.iter().map(|p| (p.id, p.track)).collect();
        let (mut named, mut m, mut c) = (0, 0, 0);
        for gt in truth_by_t.values().flatten() {
            let Some(name) = truth.names.get(&gt.track).filter(|n| is_named(n)) else {
                continue;
            };
            named += 1;
            if let Some(p) = matched.get(&gt.id) {
                m += 1;
                let pname = pred_track.get(p).and_then(|t| predicted.names.get(t));
                c += usize::from(pname == Some(name));
            }
        }
        Naming {
            named_truth: named,
            matched: m,
            correct: c,
            of_matched: ratio(c, m),
            end_to_end: ratio(c, named),
        }
    });

    Ok(Scores {
        params: *params,
        frames_scored: truth_by_t.len(),
        truth_detections: truth_by_t.values().map(Vec::len).sum(),
        predicted_detections: pred_by_t.values().map(Vec::len).sum(),
        detection: Counts::new(tp, fp, fn_, unscored, precision_defined),
        matched_distance_mean: mean(&distances),
        division: Counts::new(dtp, dfp, dfn, dun, precision_defined),
        division_dt_mean: mean(&dts),
        links: Links {
            truth_links: scored_truth_links.len(),
            truth_links_matched_ends: with_ends.len(),
            recovered,
            recall: ratio(recovered, scored_truth_links.len()),
            predicted_links_matched_ends: pred_with_ends.len(),
            correct,
            precision: ratio(correct, pred_with_ends.len()),
        },
        naming,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two frames of a mother, then two daughters for two frames.
    fn toy() -> Tracking {
        let p = |id, track, t, x: f64| Point {
            id,
            track,
            t,
            pos: [0.0, 0.0, x],
        };
        Tracking {
            points: vec![
                p(1, 1, 0, 0.0),
                p(2, 1, 1, 0.0),
                p(3, 3, 2, -3.0),
                p(4, 4, 2, 3.0),
                p(5, 3, 3, -4.0),
                p(6, 4, 3, 4.0),
                p(7, 7, 0, 20.0),
            ],
            links: vec![(1, 2), (2, 3), (2, 4), (3, 5), (4, 6)],
            names: [
                (1, "AB".into()),
                (3, "ABa".into()),
                (4, "ABp".into()),
                (7, ":1".into()),
            ]
            .into(),
        }
    }

    #[test]
    fn perfect_and_perturbed() {
        let gt = toy();
        let params = ScoreParams {
            gate: 2.0,
            ..ScoreParams::default()
        };
        let s = score(&gt, &gt, &params).unwrap();
        assert_eq!(
            (
                s.detection.f1,
                s.division.f1,
                s.links.recall,
                s.links.precision
            ),
            (Some(1.0), Some(1.0), Some(1.0), Some(1.0))
        );
        assert_eq!(s.naming.as_ref().unwrap().named_truth, 6);
        assert_eq!(s.naming.unwrap().of_matched, Some(1.0));
        let s = score(&gt, &gt.without_divisions(1), &params).unwrap();
        assert_eq!((s.division.recall, s.division.precision), (Some(0.0), None));
        assert_eq!(s.detection.f1, Some(1.0));
        let s = score(&gt, &gt.siblings_swapped(), &params).unwrap();
        assert_eq!(s.naming.unwrap().correct, 2); // only AB's two frames
    }

    #[test]
    fn division_tolerance_and_partial_coverage() {
        let gt = toy();
        // The prediction divides one frame late: daughters appear at t = 3.
        let mut late = gt.clone();
        late.links = vec![(1, 2), (2, 3), (3, 5), (3, 6)];
        let params = ScoreParams {
            gate: 2.0,
            ..ScoreParams::default()
        };
        let s = score(&gt, &late, &params).unwrap();
        // Mother at t=2 is x=-3, 3 µm from the true mother: outside the 2 µm gate.
        assert_eq!(s.division.tp, 0);
        let s = score(
            &gt,
            &late,
            &ScoreParams {
                gate: 3.5,
                ..params
            },
        )
        .unwrap();
        assert_eq!((s.division.tp, s.division_dt_mean), (1, Some(1.0)));
        // Partial truth: only the AB lineage annotated; the far cell 7 is unscored.
        let partial = gt.restricted(|t| t != 7);
        let full = score(&partial, &gt, &params).unwrap();
        assert_eq!((full.detection.fp, full.detection.unscored), (1, 0));
        let s = score(
            &partial,
            &gt,
            &ScoreParams {
                coverage: Coverage::Partial { radius: 2.0 },
                ..params
            },
        )
        .unwrap();
        assert_eq!(
            (s.detection.fp, s.detection.unscored, s.detection.precision),
            (0, 1, Some(1.0))
        );
        let s = score(
            &partial,
            &gt,
            &ScoreParams {
                coverage: Coverage::RecallOnly,
                ..params
            },
        )
        .unwrap();
        assert_eq!(
            (s.detection.precision, s.detection.recall),
            (None, Some(1.0))
        );
    }
}
