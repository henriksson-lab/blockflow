// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Star-convex shells: fitted from the image along rays, and painted as labels.**
//
// The objects are round, hollow-walled shells: a thin bright wall with a darker
// band just inside it (cysts, oocysts, vesicles in micro-CT or EM). A candidate
// centre comes from elsewhere — a template scan, a blob detector — and the fit
// here finds the object's surface as one radius per ray:
//
// 1. **Centre search.** On a coarse-to-fine grid of centres around the
//    candidate, the sphere collecting the most *wall evidence* (summed over a
//    sparse ray set) wins; it also gives the radius to search around.
// 2. **Rays.** Along each of `n` rays the boundary candidates are local maxima
//    of the dark-to-bright step into the wall. A global radius starts every ray;
//    then each ray repeatedly picks the candidate that balances its step height
//    against the median radius of its neighbours, within a bounded jump. Rays
//    with no acceptable candidate take their neighbours' radius.
// 3. **Surface.** The boundary is moved out to the wall's intensity peak plus a
//    pad, lightly smoothed over neighbours, and the centre is re-fitted as the
//    least-squares sphere through the supported points. Repeat until the
//    centre settles.
//
// Features for accepting a fit: the fraction of rays with a strong wall step
// (relative to a reference step), the *ring depth* — interior brightness minus
// the dark band under the wall, which separates a shell from a dark lumen
// bounded by bright tissue — coverage, roughness and size.
//
// **Planned shape.** [`ShellFitOp`] reads an image with a halo that bounds
// every sample the fit can take, so the answer does not depend on the block
// grid: samples are clamped at the volume's faces, never at a block's. With a
// prefilter it reads the raw image and smooths only the tiles some candidate's
// fit can sample — a block without candidates smooths nothing, and no smoothed
// copy of the volume is ever written — each tile with its full Gaussian reach,
// so a smoothed voxel is the same bits a whole-volume `SmoothOp` gives. Each block fits the candidate rows it owns. A fitted row stays
// keyed at its candidate voxel and carries the refined centre as columns,
// because the centre may move into another block. Per-ray radii leave on a
// separate stream, so the row table stays small and the rays can be dropped
// once painted. [`StarConvexLabelsOp`] paints every accepted object into a
// `u64` image from its rays, reading rows and rays of the neighbouring blocks,
// and gives a voxel claimed twice to the object it is relatively deepest in.

use std::collections::BTreeMap;
use std::sync::Arc;

use ndarray::{s, Array3, ArrayView3};
use rayon::prelude::*;

use crate::assemble::ImageId;
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, SeamFold,
    SidecarSize, SourceBlocks,
};
use crate::op::SourceInput;
use crate::reach::Reach;
use crate::ops::edges::trilinear_at;
use crate::ops::ridge::gaussian_smooth_into_with;
use crate::ops::smooth::Gaussian;
use crate::voxels::Voxels;
use crate::ops::rows::sealed_rows;
use crate::sidecar::Lifecycle;
use crate::sphere::{fit_sphere, RaySet};
use crate::table::{Column, RowBuilder, Schema, Value};

/// Radial sample spacing along a ray, in voxels.
const STEP: f64 = 0.5;
const RAYS_MAGIC: u64 = 0x5354_5241_5953_3031;

/// Parameters of [`fit_shell`]. Lengths in voxels of an isotropic grid.
#[derive(Debug, Clone, PartialEq)]
pub struct ShellFitParams {
    /// Expected outer radius; the searches scale with it.
    pub radius: f64,
    /// Radii searched by the centre search, as fractions of `radius`.
    pub search: [f64; 2],
    /// Radii the rays search around the radius the centre search found.
    pub refine: [f64; 2],
    /// Added to the wall peak to reach the outer wall surface.
    pub wall_pad: f64,
    /// Largest radius change from the neighbours' median that still counts as
    /// the same surface.
    pub max_jump: f64,
    /// Neighbour-averaging passes over the final radii.
    pub smoothing: usize,
    /// The fitted centre stays within this distance of the candidate.
    pub max_shift: f64,
}

impl ShellFitParams {
    /// The defaults the cyst detector was tuned with, for a given radius.
    pub fn for_radius(radius: f64) -> Self {
        Self {
            radius,
            search: [0.55, 1.6],
            refine: [0.8, 1.25],
            wall_pad: 1.0,
            max_jump: 3.0,
            smoothing: 2,
            max_shift: 0.5 * radius,
        }
    }

    fn check(&self) -> Result<()> {
        let ok = self.radius.is_finite()
            && self.radius > 0.0
            && self.search[0] > 0.0
            && self.search[1] > self.search[0]
            && self.refine[0] > 0.0
            && self.refine[1] > self.refine[0]
            && self.wall_pad.is_finite()
            && self.max_jump > 0.0
            && self.max_shift >= 0.0;
        if ok {
            Ok(())
        } else {
            Err(Error::invalid(format!("invalid shell fit parameters {self:?}")))
        }
    }

    /// How far from the candidate any sample of the fit can lie, plus the
    /// trilinear neighbour: the halo a block needs for a decomposition-invariant
    /// answer.
    pub fn reach(&self) -> usize {
        let centre = self.max_shift.max(CENTRE_SEARCH);
        let evidence = self.search[1] * self.radius + 6.0;
        let rays = self.refine[1] * self.search[1] * self.radius + 4.0;
        (centre + evidence.max(rays)).ceil() as usize + 2
    }
}

/// Largest offset the centre search grid visits: 2 x 4 + 2 + 1.
const CENTRE_SEARCH: f64 = 11.0;

/// One fitted shell.
#[derive(Debug, Clone, PartialEq)]
pub struct ShellFit {
    /// `[z, y, x]` in volume voxels; may lie outside the volume for an object
    /// cut by a face.
    pub center: [f64; 3],
    /// Outer radius per ray, in [`RaySet`] order.
    pub radii: Vec<f32>,
    /// Wall step height per ray, zero where no wall was found.
    pub steps: Vec<f32>,
    /// Rays whose boundary lies inside the volume; the rest cannot be judged.
    pub valid: Vec<bool>,
    pub coverage: f64,
    pub mean_r: f64,
    pub std_r: f64,
    pub min_r: f64,
    pub max_r: f64,
    /// Median wall step over supported rays, in image units.
    pub step_median: f64,
    /// `step_median` over the 5-95 percentile spread of the samples taken.
    pub contrast: f64,
    /// Median over supported rays of interior brightness minus the darkest
    /// value just inside the wall, in image units.
    pub ring_depth: f64,
    /// Distance from the candidate to the fitted centre.
    pub shift: f64,
}

impl ShellFit {
    /// Fraction of judgeable rays whose wall step is at least
    /// `fraction * reference`. Zero when fewer than 30 % of rays are judgeable.
    pub fn strong(&self, reference: f64, fraction: f64) -> f64 {
        let valid = self.valid.iter().filter(|&&v| v).count();
        if valid * 10 < self.valid.len() * 3 {
            return 0.0;
        }
        let strong = self
            .steps
            .iter()
            .zip(&self.valid)
            .filter(|(&s, &v)| v && s as f64 >= fraction * reference)
            .count();
        strong as f64 / valid as f64
    }
}

/// An image window placed in its volume, sampled with clamping at the
/// **volume's** faces.
struct Window<'a> {
    values: ArrayView3<'a, f64>,
    origin: [usize; 3],
    volume: [usize; 3],
}

impl Window<'_> {
    fn sample(&self, p: [f64; 3]) -> f32 {
        let local: [f64; 3] = std::array::from_fn(|a| {
            let q = p[a].clamp(0.0, (self.volume[a] - 1) as f64) - self.origin[a] as f64;
            // Inside the window by the declared reach; clamped again only as a
            // guard against a caller passing a short window.
            q.clamp(0.0, (self.values.shape()[a] - 1) as f64)
        });
        trilinear_at(self.values, local).unwrap_or(0.0) as f32
    }

    fn profile(&self, centre: [f64; 3], u: [f64; 3], start: f64, count: usize, out: &mut Vec<f32>) {
        out.clear();
        out.extend((0..count).map(|k| {
            let r = start + k as f64 * STEP;
            self.sample([centre[0] + r * u[0], centre[1] + r * u[1], centre[2] + r * u[2]])
        }));
    }
}

/// Mean over `[r, r + 2]` minus mean over `[r - 4, r - 1]` at sample `k`.
fn step_response(profile: &[f32], k: usize) -> f32 {
    let up: f32 = (0..=4).map(|o| profile[k + o]).sum::<f32>() / 5.0;
    let down: f32 = (2..=8).map(|o| profile[k - o]).sum::<f32>() / 7.0;
    up - down
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// `(evidence, radius)` of the best sphere centred at `centre`: per radius, the
/// sum over rays of the strongest step within 1.5 voxels of it.
fn sphere_evidence(window: &Window, centre: [f64; 3], rays: &RaySet, lo: f64, hi: f64) -> (f32, f64) {
    let start = (lo - 6.0).max(0.0);
    let count = ((hi + 6.0 - start) / STEP) as usize + 1;
    let radii = (hi - lo) as usize + 1;
    let mut total = vec![0f32; radii];
    let mut profile = Vec::with_capacity(count);
    for &u in rays.directions() {
        window.profile(centre, u, start, count, &mut profile);
        for (j, t) in total.iter_mut().enumerate() {
            let r = lo + j as f64;
            let k0 = (((r - 1.5 - start) / STEP).round() as usize).max(8);
            let k1 = (((r + 1.5 - start) / STEP).round() as usize).min(count - 5);
            *t += (k0..=k1).map(|k| step_response(&profile, k)).fold(0.0, f32::max);
        }
    }
    let (j, e) = total
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .expect("at least one radius");
    (*e, lo + j as f64)
}

/// Fit the shell around `candidate` in a window of a smoothed, isotropic image.
///
/// `window` holds the voxels `origin .. origin + shape` of a volume of extent
/// `volume`; it must cover `candidate` plus [`ShellFitParams::reach`] on every
/// side (clipped at the volume) for the answer not to depend on the window.
/// `rays` gives the surface resolution, `search_rays` the cheaper set the
/// centre search uses.
pub fn fit_shell(
    window: ArrayView3<'_, f64>,
    origin: [usize; 3],
    volume: [usize; 3],
    candidate: [f64; 3],
    rays: &RaySet,
    search_rays: &RaySet,
    params: &ShellFitParams,
) -> Result<ShellFit> {
    params.check()?;
    if rays.is_empty() || search_rays.is_empty() {
        return Err(Error::invalid("a shell fit needs rays"));
    }
    let window = Window {
        values: window,
        origin,
        volume,
    };
    let (lo, hi) = (params.search[0] * params.radius, params.search[1] * params.radius);
    let mut best = (sphere_evidence(&window, candidate, search_rays, lo, hi), candidate);
    for step in [4.0, 2.0, 1.0] {
        let around = best.1;
        let span: i32 = if step == 4.0 { 2 } else { 1 };
        for dz in -span..=span {
            for dy in -span..=span {
                for dx in -span..=span {
                    let c = [
                        around[0] + dz as f64 * step,
                        around[1] + dy as f64 * step,
                        around[2] + dx as f64 * step,
                    ];
                    let e = sphere_evidence(&window, c, search_rays, lo, hi);
                    if e.0 > best.0 .0 {
                        best = (e, c);
                    }
                }
            }
        }
    }
    let found = best.0 .1;
    let mut centre = best.1;
    let mut fit = None;
    for _ in 0..8 {
        let next = fit_once(&window, rays, centre, found, params);
        let settled = distance(next.center, centre) < 0.5;
        let too_far = distance(next.center, candidate) > params.max_shift;
        if too_far {
            // Keep the surface found at the last admissible centre.
            let mut kept = next;
            kept.center = centre;
            fit = Some(kept);
            break;
        }
        centre = next.center;
        fit = Some(next);
        if settled {
            break;
        }
    }
    let mut fit = fit.expect("at least one round");
    fit.shift = distance(candidate, fit.center);
    Ok(fit)
}

fn fit_once(window: &Window, rays: &RaySet, centre: [f64; 3], guess: f64, p: &ShellFitParams) -> ShellFit {
    let n = rays.len();
    let (lo, hi) = (p.refine[0] * guess, p.refine[1] * guess);
    let start = (lo - 4.0).max(0.0);
    let count = ((hi + 4.0 - start) / STEP) as usize + 1;
    let radius_at = |k: usize| start + k as f64 * STEP;
    let mut profiles = Vec::with_capacity(n);
    for &u in rays.directions() {
        let mut profile = Vec::with_capacity(count);
        window.profile(centre, u, start, count, &mut profile);
        profiles.push(profile);
    }
    // per ray: local maxima of the step, strongest first, at most five
    let candidates: Vec<Vec<(f64, f32, usize)>> = profiles
        .iter()
        .map(|profile| {
            let step = |k: usize| {
                if k >= 8 && k + 4 < count && radius_at(k) >= lo && radius_at(k) <= hi {
                    step_response(profile, k)
                } else {
                    f32::NEG_INFINITY
                }
            };
            let mut found: Vec<(f64, f32, usize)> = (1..count - 1)
                .filter_map(|k| {
                    let s = step(k);
                    (s > 0.0 && s >= step(k - 1) && s > step(k + 1)).then(|| (radius_at(k), s, k))
                })
                .collect();
            found.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.2.cmp(&b.2)));
            found.truncate(5);
            found
        })
        .collect();
    let mut strongest: Vec<f32> = candidates.iter().filter_map(|c| c.first().map(|x| x.1)).collect();
    let typical = if strongest.is_empty() { 1.0 } else { median(&mut strongest).max(1e-3) };
    // the sphere collecting the most step evidence starts every ray
    let mut global = guess;
    let mut best = f32::NEG_INFINITY;
    let mut r = lo;
    while r <= hi {
        let s: f32 = candidates
            .iter()
            .map(|c| c.iter().filter(|x| (x.0 - r).abs() <= 2.0).map(|x| x.1).fold(0.0, f32::max))
            .sum();
        if s > best {
            best = s;
            global = r;
        }
        r += STEP;
    }
    let lambda = 0.15 * typical;
    let pick = |c: &[(f64, f32, usize)], target: f64| {
        c.iter()
            .filter(|x| (x.0 - target).abs() <= p.max_jump)
            .max_by(|a, b| {
                let score = |x: &(f64, f32, usize)| x.1 - lambda * (x.0 - target).abs() as f32;
                score(a).total_cmp(&score(b)).then(b.2.cmp(&a.2))
            })
            .copied()
    };
    let mut chosen: Vec<Option<(f64, f32, usize)>> = candidates.iter().map(|c| pick(c, global)).collect();
    let mut current: Vec<f64> = chosen.iter().map(|c| c.map_or(global, |x| x.0)).collect();
    for _ in 0..12 {
        let target: Vec<f64> = (0..n)
            .map(|i| {
                let mut v: Vec<f32> = rays.neighbours(i).iter().map(|&j| current[j] as f32).collect();
                v.push(current[i] as f32);
                median(&mut v) as f64
            })
            .collect();
        for i in 0..n {
            chosen[i] = pick(&candidates[i], target[i]);
            current[i] = chosen[i].map_or(target[i], |x| x.0);
        }
    }
    // outer surface: the wall's peak just outside the step, plus the pad
    let mut radii: Vec<f32> = (0..n)
        .map(|i| match chosen[i] {
            Some((_, _, k)) => {
                let end = (k + 8).min(count - 1);
                let peak = (k..=end)
                    .max_by(|&a, &b| profiles[i][a].total_cmp(&profiles[i][b]).then(b.cmp(&a)))
                    .expect("non-empty range");
                (radius_at(peak) + p.wall_pad) as f32
            }
            None => f32::NAN,
        })
        .collect();
    let supported: Vec<bool> = chosen.iter().map(Option::is_some).collect();
    for _ in 0..20 {
        let previous = radii.clone();
        let mut complete = true;
        for i in 0..n {
            if previous[i].is_nan() {
                let known: Vec<f32> =
                    rays.neighbours(i).iter().map(|&j| previous[j]).filter(|v| !v.is_nan()).collect();
                if known.is_empty() {
                    complete = false;
                } else {
                    radii[i] = known.iter().sum::<f32>() / known.len() as f32;
                }
            }
        }
        if complete {
            break;
        }
    }
    let fallback = (global + p.wall_pad) as f32;
    radii.iter_mut().filter(|r| r.is_nan()).for_each(|r| *r = fallback);
    for _ in 0..p.smoothing {
        let previous = radii.clone();
        for i in 0..n {
            let around = rays.neighbours(i);
            let sum: f32 = around.iter().map(|&j| previous[j]).sum::<f32>() + 2.0 * previous[i];
            radii[i] = sum / (around.len() as f32 + 2.0);
        }
    }
    let points: Vec<[f64; 3]> = (0..n)
        .filter(|&i| supported[i])
        .map(|i| std::array::from_fn(|a| centre[a] + radii[i] as f64 * rays.directions()[i][a]))
        .collect();
    let refitted = fit_sphere(&points)
        .map(|(c, _)| c)
        .filter(|c| distance(*c, centre) < 0.3 * guess)
        .unwrap_or(centre);
    let mean_r = radii.iter().map(|&r| r as f64).sum::<f64>() / n as f64;
    let std_r = (radii.iter().map(|&r| (r as f64 - mean_r).powi(2)).sum::<f64>() / n as f64).sqrt();
    let mut supported_steps: Vec<f32> = chosen.iter().filter_map(|c| c.map(|x| x.1)).collect();
    let step_median = if supported_steps.is_empty() { 0.0 } else { median(&mut supported_steps) as f64 };
    let mut all: Vec<f32> = profiles.iter().flatten().copied().collect();
    all.sort_by(|a, b| a.total_cmp(b));
    let spread = (all[all.len() * 95 / 100] - all[all.len() * 5 / 100]).max(1e-3) as f64;
    let mut depths: Vec<f32> = (0..n)
        .filter_map(|i| {
            let (r, _, k) = chosen[i]?;
            let u = rays.directions()[i];
            let at = |rr: f64| window.sample([centre[0] + rr * u[0], centre[1] + rr * u[1], centre[2] + rr * u[2]]);
            let inner = (0..5).map(|q| at(r * (0.15 + 0.1 * q as f64))).sum::<f32>() / 5.0;
            let ring = (k.saturating_sub(8)..k).map(|q| profiles[i][q]).fold(f32::INFINITY, f32::min);
            Some(inner - ring)
        })
        .collect();
    let ring_depth = if depths.is_empty() { 0.0 } else { median(&mut depths) as f64 };
    let valid = (0..n)
        .map(|i| {
            (0..3).all(|a| {
                let q = centre[a] + radii[i] as f64 * rays.directions()[i][a];
                q >= 1.0 && q <= window.volume[a] as f64 - 2.0
            })
        })
        .collect();
    ShellFit {
        center: refitted,
        steps: chosen.iter().map(|c| c.map_or(0.0, |x| x.1)).collect(),
        valid,
        coverage: supported.iter().filter(|&&s| s).count() as f64 / n as f64,
        min_r: radii.iter().copied().fold(f32::INFINITY, f32::min) as f64,
        max_r: radii.iter().copied().fold(0.0, f32::max) as f64,
        radii,
        mean_r,
        std_r,
        step_median,
        contrast: step_median / spread,
        ring_depth,
        shift: distance(refitted, centre),
    }
}

// ------------------------------------------------------------ acceptance --

/// What a fit is judged against, folded into columns a plain row filter can
/// test: the two-tier `strong` rule becomes one margin that is non-negative
/// when the rule passes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellJudge {
    /// The wall step of a known object (e.g. an exemplar's `step_median`).
    pub step_reference: f64,
    /// A ray is strong at `>= strong_step * step_reference`.
    pub strong_step: f64,
    /// Fraction of strong rays a small object needs.
    pub min_strong: f64,
    /// Fraction a large object needs; large is `mean_r >= large_from * radius`.
    pub min_strong_large: f64,
    pub large_from: f64,
}

impl ShellJudge {
    pub fn margin(&self, fit: &ShellFit, radius: f64) -> (f64, f64) {
        let strong = fit.strong(self.step_reference, self.strong_step);
        let need = if fit.mean_r >= self.large_from * radius {
            self.min_strong_large
        } else {
            self.min_strong
        };
        (strong, strong - need)
    }
}

/// Columns [`ShellFitOp`] appends to its input rows, in order.
pub const SHELL_COLUMNS: [&str; 13] = [
    "center_z",
    "center_y",
    "center_x",
    "mean_r",
    "std_r",
    "min_r",
    "max_r",
    "roughness",
    "coverage",
    "strong",
    "strong_margin",
    "ring_depth",
    "contrast",
];

/// The input schema with [`SHELL_COLUMNS`] appended as `f64`, plus `shift`.
pub fn shell_schema(input: &Schema) -> Result<Schema> {
    let mut columns: Vec<Column> = input.columns().to_vec();
    for name in SHELL_COLUMNS.iter().chain(["shift"].iter()) {
        if input.index_of(name).is_some() {
            return Err(Error::invalid(format!(
                "the candidate rows already have a column {name:?}"
            )));
        }
        columns.push(Column::f64(*name));
    }
    Schema::new(columns)
}

// ------------------------------------------------------------- rays wire --

/// One object's surface as carried between phases.
#[derive(Debug, Clone, PartialEq)]
pub struct StarShape {
    /// The candidate voxel its row is keyed at.
    pub at: [usize; 3],
    pub center: [f64; 3],
    pub radii: Vec<f32>,
}

pub fn encode_shapes(rays: usize, shapes: &[StarShape]) -> Vec<u8> {
    let mut out = Vec::with_capacity(24 + shapes.len() * (48 + 4 * rays));
    out.extend_from_slice(&RAYS_MAGIC.to_le_bytes());
    out.extend_from_slice(&(rays as u64).to_le_bytes());
    out.extend_from_slice(&(shapes.len() as u64).to_le_bytes());
    for s in shapes {
        for v in s.at {
            out.extend_from_slice(&(v as u64).to_le_bytes());
        }
        for v in s.center {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for r in &s.radii {
            out.extend_from_slice(&r.to_le_bytes());
        }
    }
    out
}

pub fn decode_shapes(bytes: &[u8]) -> Result<(usize, Vec<StarShape>)> {
    let bad = || Error::invalid("not a star-shape fragment");
    let word = |i: usize| -> Result<[u8; 8]> {
        bytes.get(i..i + 8).map(|b| b.try_into().expect("eight")).ok_or_else(bad)
    };
    if u64::from_le_bytes(word(0)?) != RAYS_MAGIC {
        return Err(bad());
    }
    let rays = u64::from_le_bytes(word(8)?) as usize;
    let count = u64::from_le_bytes(word(16)?) as usize;
    let each = 48 + 4 * rays;
    if bytes.len() != 24 + count * each {
        return Err(bad());
    }
    let mut shapes = Vec::with_capacity(count);
    for k in 0..count {
        let o = 24 + k * each;
        let at = std::array::from_fn(|a| u64::from_le_bytes(word(o + 8 * a).unwrap()) as usize);
        let center = std::array::from_fn(|a| f64::from_le_bytes(word(o + 24 + 8 * a).unwrap()));
        let radii = (0..rays)
            .map(|r| {
                let p = o + 48 + 4 * r;
                f32::from_le_bytes(bytes[p..p + 4].try_into().expect("four"))
            })
            .collect();
        shapes.push(StarShape { at, center, radii });
    }
    Ok((rays, shapes))
}

// ------------------------------------------------------------ the fit op --

/// **Candidate rows in, fitted rows and their rays out.**
///
/// Reads `image` — isotropic, smoothed by an earlier phase or raw with
/// [`Self::with_prefilter`] — with [`ShellFitParams::reach`] (plus the
/// prefilter's reach) as its halo, and fits every candidate row its block owns. A source input rather than the phase input, so the candidate
/// rows can come from any earlier phase. Each output row is its input row with [`shell_schema`]'s columns; the
/// rays go to a separate stream of [`StarShape`]s.
pub struct ShellFitOp {
    name: &'static str,
    image: ImageId,
    input: (String, usize),
    input_schema: Schema,
    output: String,
    lifecycle: Lifecycle,
    rays_stream: String,
    rays_lifecycle: Lifecycle,
    schema: Schema,
    params: ShellFitParams,
    judge: ShellJudge,
    rays: Arc<RaySet>,
    search_rays: Arc<RaySet>,
    prefilter: Option<Gaussian>,
}

/// Edge of the tiles a prefilter is computed on: small enough that a block's
/// candidates select little more than what their fits sample, large enough that
/// the Gaussian's halo around each tile stays a modest overhead.
const PREFILTER_TILE: usize = 32;

impl ShellFitOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &'static str,
        image: ImageId,
        input: (impl Into<String>, usize),
        input_schema: Schema,
        output: (impl Into<String>, Lifecycle),
        rays_stream: (impl Into<String>, Lifecycle),
        params: ShellFitParams,
        judge: ShellJudge,
        rays: usize,
    ) -> Result<Self> {
        params.check()?;
        if rays < 20 {
            return Err(Error::invalid("a shell fit needs at least 20 rays"));
        }
        let schema = shell_schema(&input_schema)?;
        Ok(Self {
            name,
            image,
            input: (input.0.into(), input.1),
            input_schema,
            output: output.0.into(),
            lifecycle: output.1,
            rays_stream: rays_stream.0.into(),
            rays_lifecycle: rays_stream.1,
            schema,
            params,
            judge,
            rays: Arc::new(RaySet::fibonacci(rays, 8)),
            search_rays: Arc::new(RaySet::fibonacci(120, 8)),
            prefilter: None,
        })
    }

    /// Smooth `image` with `gaussian` before fitting, inside the op and only
    /// where a fit samples. Equivalent to a `SmoothOp` phase in front, without
    /// computing or storing the smoothed volume away from the candidates.
    pub fn with_prefilter(mut self, gaussian: Gaussian) -> Self {
        self.prefilter = Some(gaussian);
        self
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn rays(&self) -> &Arc<RaySet> {
        &self.rays
    }
}

impl FragmentOp for ShellFitOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        self.params.reach()
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        let reach = self.params.reach();
        let halo: [usize; 3] =
            std::array::from_fn(|a| reach + self.prefilter.as_ref().map_or(0, |g| g.reach(a)));
        vec![SourceInput::new(self.image, Reach::symmetric(halo))]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        // Each row is fitted by its owning block alone; nothing is folded.
        Some(SeamFold::Unordered)
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.input.0, self.input.1)]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![
            FragmentOutput::new(&self.output, self.lifecycle, Coverage::EveryBlock)
                .sized(SidecarSize::row_table(&self.schema, 1)),
            FragmentOutput::new(&self.rays_stream, self.rays_lifecycle, Coverage::EveryBlock),
        ]
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid("the shell fit reads its image through apply_with"))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let volume = at.volume();
        let rows = sealed_rows(volume, &self.input_schema, at.own(&self.input.0).unwrap_or(&[]))?;
        let mut builder = RowBuilder::new(Arc::new(self.schema.clone()));
        let mut shapes = Vec::with_capacity(rows.len());
        if !rows.is_empty() {
            let raw = sources.get(self.image.index())?.as_array()?;
            let origin = [at.read.start[0], at.read.start[1], at.read.start[2]];
            let pixels = match &self.prefilter {
                None => raw.widened(),
                Some(gaussian) => {
                    let shape = raw.shape();
                    let reach = self.params.reach() as isize;
                    let boxes: Vec<[[usize; 3]; 2]> = rows
                        .iter()
                        .map(|row| {
                            let local: [isize; 3] = std::array::from_fn(|a| row.at[a] as isize - origin[a] as isize);
                            [
                                std::array::from_fn(|a| (local[a] - reach).clamp(0, shape[a] as isize) as usize),
                                std::array::from_fn(|a| (local[a] + reach + 1).clamp(0, shape[a] as isize) as usize),
                            ]
                        })
                        .collect();
                    smooth_where_sampled(raw, gaussian, &boxes, PREFILTER_TILE)?
                }
            };
            let fits: Vec<ShellFit> = rows
                .par_iter()
                .map(|row| {
                    fit_shell(
                        pixels.view(),
                        origin,
                        volume,
                        row.at.map(|v| v as f64),
                        &self.rays,
                        &self.search_rays,
                        &self.params,
                    )
                })
                .collect::<Result<_>>()?;
            for (row, fit) in rows.iter().zip(fits) {
                let (strong, margin) = self.judge.margin(&fit, self.params.radius);
                let mut values = row.values.clone();
                values.extend(
                    [
                        fit.center[0],
                        fit.center[1],
                        fit.center[2],
                        fit.mean_r,
                        fit.std_r,
                        fit.min_r,
                        fit.max_r,
                        fit.std_r / fit.mean_r.max(1e-9),
                        fit.coverage,
                        strong,
                        margin,
                        fit.ring_depth / self.judge.step_reference.max(1e-12),
                        fit.contrast,
                        fit.shift,
                    ]
                    .map(Value::F64),
                );
                builder.push(row.at, &values)?;
                shapes.push(StarShape {
                    at: row.at,
                    center: fit.center,
                    radii: fit.radii,
                });
            }
        }
        Ok(BlockOutput::fragment(&self.output, builder.encode())
            .with_fragment(&self.rays_stream, encode_shapes(self.rays.len(), &shapes)))
    }
}

/// `raw` smoothed by `gaussian` on the tiles of edge `tile` that meet one of
/// `boxes` (half-open, buffer coordinates), and zero elsewhere.
///
/// Each tile is smoothed over itself grown by the Gaussian's reach, clamped to
/// the buffer, so every computed voxel sums exactly the taps — in the same
/// order, from the same values — that smoothing the whole buffer would. Tiles
/// are done in parallel, a bounded number at a time so the pieces in flight
/// stay small next to the result.
fn smooth_where_sampled(raw: &Voxels, gaussian: &Gaussian, boxes: &[[[usize; 3]; 2]], tile: usize) -> Result<Array3<f64>> {
    let shape = raw.shape();
    let mut needed = std::collections::BTreeSet::new();
    for [lo, hi] in boxes {
        if (0..3).any(|a| lo[a] >= hi[a]) {
            continue;
        }
        for t0 in lo[0] / tile..=(hi[0] - 1) / tile {
            for t1 in lo[1] / tile..=(hi[1] - 1) / tile {
                for t2 in lo[2] / tile..=(hi[2] - 1) / tile {
                    needed.insert([t0, t1, t2]);
                }
            }
        }
    }
    let needed: Vec<[usize; 3]> = needed.into_iter().collect();
    let mut out = Array3::<f64>::zeros(shape);
    for batch in needed.chunks(4 * rayon::current_num_threads().max(1)) {
        let pieces = batch
            .par_iter()
            .map(|t| {
                let core: [[usize; 2]; 3] =
                    std::array::from_fn(|a| [t[a] * tile, ((t[a] + 1) * tile).min(shape[a])]);
                let grown: [[usize; 2]; 3] = std::array::from_fn(|a| {
                    let r = gaussian.reach(a);
                    [core[a][0].saturating_sub(r), (core[a][1] + r).min(shape[a])]
                });
                let mut smoothed =
                    Array3::<f64>::zeros(std::array::from_fn::<usize, 3, _>(|a| grown[a][1] - grown[a][0]));
                let window = s![grown[0][0]..grown[0][1], grown[1][0]..grown[1][1], grown[2][0]..grown[2][1]];
                dispatch_f64_input!(
                    raw,
                    Error::invalid("a shell fit prefilter cannot read half-precision"),
                    |view| { gaussian_smooth_into_with(view.slice(window), gaussian.kernels(), gaussian.boundary(), smoothed.view_mut()) }
                )?;
                let inner = smoothed
                    .slice(s![
                        core[0][0] - grown[0][0]..core[0][1] - grown[0][0],
                        core[1][0] - grown[1][0]..core[1][1] - grown[1][0],
                        core[2][0] - grown[2][0]..core[2][1] - grown[2][0]
                    ])
                    .to_owned();
                Ok((core, inner))
            })
            .collect::<Result<Vec<_>>>()?;
        for (core, piece) in pieces {
            out.slice_mut(s![core[0][0]..core[0][1], core[1][0]..core[1][1], core[2][0]..core[2][1]])
                .assign(&piece);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------- the painter --

/// Stable label of the object whose row is keyed at `at`: its linear index + 1.
pub fn shape_label(at: [usize; 3], volume: [usize; 3]) -> u64 {
    ((at[0] * volume[1] + at[1]) * volume[2] + at[2]) as u64 + 1
}

/// **Accepted rows and their rays in, a `u64` label image out.**
///
/// Paints, for its own core, every object whose row survived into `rows` from
/// this block and its 26 neighbours, using the shape from `rays` keyed at the
/// same voxel. A voxel inside several objects goes to the one it is relatively
/// deepest in (`|p - c| / r(direction)` smallest; ties to the smaller label), so
/// the answer does not depend on visiting order. Objects must fit within one
/// block of their row's block: their extent is checked against the grid.
pub struct StarConvexLabelsOp {
    name: &'static str,
    rows: (String, usize),
    rows_schema: Schema,
    shapes: (String, usize),
    rays: Arc<RaySet>,
}

impl StarConvexLabelsOp {
    pub fn new(
        name: &'static str,
        rows: (impl Into<String>, usize),
        rows_schema: Schema,
        shapes: (impl Into<String>, usize),
        rays: Arc<RaySet>,
    ) -> Self {
        Self {
            name,
            rows: (rows.0.into(), rows.1),
            rows_schema,
            shapes: (shapes.0.into(), shapes.1),
            rays,
        }
    }
}

impl FragmentOp for StarConvexLabelsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U64
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![
            FragmentInput::own(&self.rows.0, self.rows.1).with_reach([1, 1, 1]),
            FragmentInput::own(&self.shapes.0, self.shapes.1).with_reach([1, 1, 1]),
        ]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let volume = at.volume();
        let mut accepted = std::collections::BTreeSet::new();
        for (_key, bytes) in at.fragments(&self.rows.0) {
            for row in sealed_rows(volume, &self.rows_schema, bytes)? {
                accepted.insert(row.at);
            }
        }
        let mut shapes: BTreeMap<[usize; 3], StarShape> = BTreeMap::new();
        for (_key, bytes) in at.fragments(&self.shapes.0) {
            let (rays, found) = decode_shapes(bytes)?;
            if rays != self.rays.len() {
                return Err(Error::invalid(format!(
                    "shapes carry {rays} rays, the painter was built for {}",
                    self.rays.len()
                )));
            }
            for s in found {
                if accepted.contains(&s.at) {
                    shapes.insert(s.at, s);
                }
            }
        }
        let block = at.grid.block();
        let core = at.core;
        let mut buffer = at.output_buffer(0.0)?;
        if let Some(out) = buffer.as_array_mut() {
            let mut view = out.view_mut::<u64>()?;
            let origin = [at.read.start[0], at.read.start[1], at.read.start[2]];
            // depth over the core only: the read extent is a block wider on every
            // side (the fragment reach), and only the core is painted
            let mut depth = ndarray::Array3::<f32>::from_elem(core.shape3(), f32::INFINITY);
            for shape in shapes.values() {
                let reach = shape.radii.iter().copied().fold(0.0, f32::max) as f64 + 1.0;
                let drift = distance(shape.center, shape.at.map(|v| v as f64));
                for a in 0..3 {
                    if reach + drift > block[a] as f64 {
                        return Err(Error::invalid(format!(
                            "an object of radius {reach:.1} at {:?} is larger than the block edge \
                             {} on axis {a}; use larger blocks",
                            shape.at, block[a]
                        )));
                    }
                }
                let label = shape_label(shape.at, volume);
                let lo: [usize; 3] = std::array::from_fn(|a| {
                    (shape.center[a] - reach).floor().max(core.start[a] as f64) as usize
                });
                let hi: [usize; 3] = std::array::from_fn(|a| {
                    let end = core.start[a] + core.shape[a];
                    ((shape.center[a] + reach).ceil() + 1.0).max(0.0).min(end as f64) as usize
                });
                for z in lo[0]..hi[0] {
                    for y in lo[1]..hi[1] {
                        for x in lo[2]..hi[2] {
                            let d = [
                                z as f64 - shape.center[0],
                                y as f64 - shape.center[1],
                                x as f64 - shape.center[2],
                            ];
                            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() as f32;
                            let relative = len / self.rays.interpolate(&shape.radii, d).max(1e-6);
                            let i = [z - origin[0], y - origin[1], x - origin[2]];
                            let c = [z - core.start[0], y - core.start[1], x - core.start[2]];
                            let better = relative < depth[c]
                                || (relative == depth[c] && label < view[i]);
                            if relative <= 1.0 && better {
                                depth[c] = relative;
                                view[i] = label;
                            }
                        }
                    }
                }
            }
        }
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble::PlanBuilder;
    use crate::env::ArrayEnvironment;
    use crate::geometry::BlockGrid;
    use crate::ops::rows::{collect_rows, RowSourceOp, RowValues};
    use crate::strategy::{execute_phases, Hints};
    use crate::voxels::Voxels;
    use ndarray::Array3;

    const VOLUME: [usize; 3] = [56, 96, 96];
    /// Outer wall surface radius of the synthetic shells.
    const WALL: f64 = 16.0;

    /// Background 100; per shell a bright wall `[WALL - 2, WALL]` at 220, a dark
    /// band 4 voxels under it at 50, and a textured interior around 130.
    fn phantom(centres: &[[f64; 3]]) -> Array3<f64> {
        Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
            let mut v = 100.0 + ((z * 7 + y * 13 + x * 3) % 11) as f64;
            for c in centres {
                let r = ((z as f64 - c[0]).powi(2) + (y as f64 - c[1]).powi(2) + (x as f64 - c[2]).powi(2)).sqrt();
                if r <= WALL {
                    v = if r > WALL - 2.0 {
                        220.0
                    } else if r > WALL - 6.0 {
                        50.0
                    } else {
                        130.0 + 30.0 * ((x / 3 + y / 3 + z / 3) % 2) as f64
                    };
                }
            }
            v
        })
    }

    fn params() -> ShellFitParams {
        ShellFitParams::for_radius(WALL)
    }

    fn judge() -> ShellJudge {
        ShellJudge {
            step_reference: 100.0,
            strong_step: 0.4,
            min_strong: 0.45,
            min_strong_large: 0.3,
            large_from: 0.75,
        }
    }

    #[test]
    fn fit_finds_a_displaced_shell() {
        let centre = [28.3, 47.6, 50.2];
        let image = phantom(&[centre]);
        let rays = RaySet::fibonacci(300, 8);
        let search = RaySet::fibonacci(120, 8);
        let fit = fit_shell(image.view(), [0; 3], VOLUME, [31.0, 44.0, 53.0], &rays, &search, &params()).unwrap();
        assert!(distance(fit.center, centre) < 1.0, "{:?}", fit.center);
        // the wall peak sits inside [WALL - 2, WALL]; the pad adds one
        assert!((fit.mean_r - WALL).abs() < 1.5, "{}", fit.mean_r);
        assert!(fit.std_r < 1.0 && fit.coverage > 0.9, "{fit:?}");
        assert!(fit.ring_depth > 50.0, "{}", fit.ring_depth);
        assert!(fit.strong(100.0, 0.4) > 0.9);
    }

    /// Planned over two lattices — one whose seams cut both objects — the fit
    /// rows and the label image are identical, and the overlap between the two
    /// shells is split by relative depth.
    #[test]
    fn planned_fit_and_paint_do_not_depend_on_the_grid() {
        let centres = [[27.5, 46.0, 30.0], [28.0, 50.0, 57.0]];
        let image = phantom(&centres);
        let schema = Schema::new(vec![Column::f64("score")]).unwrap();
        let candidates = vec![
            RowValues::new([25, 45, 31], vec![Value::F64(2.0)]),
            RowValues::new([29, 52, 55], vec![Value::F64(1.0)]),
        ];
        let mut answers = Vec::new();
        for block in [[56, 96, 96], [28, 48, 44]] {
            let grid = BlockGrid::new(VOLUME, block).unwrap();
            let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
            let smoothed = builder
                .pixels(crate::op::Chain::op(crate::ops::SmoothOp::isotropic("smooth", 1.0, 4.0).unwrap()))
                .unwrap();
            let source = builder
                .fragments(RowSourceOp::new("c", "candidates", Lifecycle::DeleteOnExit, schema.clone(), candidates.clone()))
                .unwrap();
            let fit = ShellFitOp::new(
                "fit",
                smoothed.image().unwrap(),
                ("candidates", source.index()),
                schema.clone(),
                ("fitted", Lifecycle::Persistent),
                ("shapes", Lifecycle::DeleteOnExit),
                params(),
                judge(),
                300,
            )
            .unwrap();
            let fitted_schema = fit.schema().clone();
            let rays = fit.rays().clone();
            let fitted = builder.fragments(fit).unwrap();
            builder
                .fragments(StarConvexLabelsOp::new(
                    "paint",
                    ("fitted", fitted.index()),
                    fitted_schema.clone(),
                    ("shapes", fitted.index()),
                    rays,
                ))
                .unwrap();
            let assembly = builder.finish().unwrap();
            let env = ArrayEnvironment::for_decomposition(Voxels::from(image.clone()), &assembly.decomposition, block).unwrap();
            execute_phases("shells", &assembly.workflow, &assembly.decomposition, &Hints::default(), &env, &[], &assembly.work())
                .unwrap();
            let rows = collect_rows(&env, "fitted", fitted.index(), VOLUME, fitted_schema).unwrap();
            let labels = env.output().view::<u64>().unwrap().to_owned();
            answers.push((rows, labels));
        }
        let (rows, labels) = &answers[0];
        assert_eq!(rows.len(), 2);
        for (row, centre) in rows.iter().zip(&centres) {
            let c = [row.values[1], row.values[2], row.values[3]].map(|v| match v {
                Value::F64(f) => f,
                Value::U64(_) => unreachable!(),
            });
            assert!(distance(c, *centre) < 1.0, "{c:?} vs {centre:?}");
        }
        let a = shape_label([25, 45, 31], VOLUME);
        let b = shape_label([29, 52, 55], VOLUME);
        let count = |l: u64| labels.iter().filter(|&&v| v == l).count() as f64;
        let ball = 4.0 / 3.0 * std::f64::consts::PI * WALL.powi(3);
        assert!((count(a) / ball - 1.0).abs() < 0.15 && (count(b) / ball - 1.0).abs() < 0.2, "{} {} {}", count(a), count(b), ball);
        // the overlap is split, nearer to each centre goes to it
        assert_eq!(labels[[28, 48, 36]], a);
        assert_eq!(labels[[28, 49, 50]], b);
        assert_eq!(&answers[1].0, rows);
        assert_eq!(&answers[1].1, labels);
    }
}
