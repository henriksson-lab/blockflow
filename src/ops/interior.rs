// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **What is inside a fitted shell: bright sub-objects, segmented per object.**
//
// A star-convex shell ([`crate::ops::star_convex`]) often holds structure of
// its own — cells packed in a cyst, lobes, one fused mass, a crescent against
// the wall. Shell-averaged features describe that texture but cannot count or
// size it, so this segments it, one object at a time, on a window around the
// object:
//
// 1. **Normalise.** The window is smoothed at `smooth`, and the median per bin
//    of the shape-adapted radius `s = |p - c| / R(direction)` (bins of 0.05)
//    gives the darkest band for `s` in 0.5–1.0 (`0`) and the wall peak for `s`
//    in 0.8–1.2 (`1`). The outside is not used: it is often bright tissue.
// 2. **Bright phase.** Interior voxels (`s < inner`, which keeps the wall out)
//    whose normalised level reaches `level`.
// 3. **Blobs.** Seeds are the regional maxima of the window smoothed at
//    `seed_smooth`, inside the bright phase; a seeded watershed floods the
//    bright phase from them; then neighbouring basins merge, highest saddle
//    first, while the lower of their two peaks rises less than `prominence`
//    above the saddle (merging by dynamics). Blobs under `min_voxels` are
//    dropped. Levels and prominence are in the normalised units of step 1, so
//    they do not depend on staining or gain.
// 4. **Measures.** Per blob: size, centroid, mean level, radial position,
//    aspect, and how many blobs share its connected bright region. Per object:
//    a summary ([`SUMMARY_COLUMNS`]).
//
// **Planned shape.** [`InteriorOp`] reads the image with a halo that covers the
// window of every object it can be handed (from a bound on the objects' extent,
// checked per object), and segments the objects whose rows its block owns —
// each object once. The window is clipped at the volume's faces only, so the
// answer does not depend on the block grid. Blobs leave as rows keyed at their
// peak voxel, the summary as rows keyed at the object's, and the voxels as
// x-runs ([`LabelRun`]) that [`RunLabelsOp`] paints into a `u64` image, reading
// the runs of the neighbouring blocks. Where objects overlap, a voxel is
// segmented by the object it is relatively deepest in — the painter's rule —
// from the neighbouring blocks' shapes that [`NearbyShapesOp`] hands each
// block, so no voxel is segmented twice.

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
use crate::ops::components::{label_members_into_with, Connectivity};
use crate::ops::regional::regional_maxima_with;
use crate::ops::ridge::{gaussian_smooth_into_with, symmetric_eigenvalues};
use crate::ops::rows::sealed_rows;
use crate::ops::smooth::Gaussian;
use crate::ops::star_convex::{decode_shapes, encode_shapes, shape_label, StarShape};
use crate::ops::watershed::{seeded_watershed_into, Separation};
use crate::reach::Reach;
use crate::sidecar::Lifecycle;
use crate::sphere::RaySet;
use crate::table::{Column, RowBuilder, Schema, Value};
use crate::voxels::Voxels;

const RUNS_MAGIC: u64 = 0x4C52_554E_5330_3031;
/// Truncation of both smoothings, in sigmas.
const TRUNCATE: f64 = 4.0;
/// The window reaches this far out in `s`, for the wall peak of step 1.
const OUTER: f64 = 1.2;
/// Bin width of the radial profile, in `s`.
const BIN: f64 = 0.05;

/// Parameters of [`segment_interior`]. Lengths in voxels of an isotropic grid;
/// levels in the normalised units (dark band 0, wall 1).
#[derive(Debug, Clone, PartialEq)]
pub struct InteriorParams {
    /// The interior is `s < inner`; below 1 so the wall stays out.
    pub inner: f64,
    /// Smoothing (sigma) for the normalisation and the bright phase.
    pub smooth: f64,
    /// Smoothing (sigma) for the seeds; about a blob's radius over two.
    pub seed_smooth: f64,
    /// A voxel is bright from this normalised level.
    pub level: f64,
    /// Two basins stay apart when the lower peak rises at least this far above
    /// their saddle.
    pub prominence: f64,
    /// Smaller blobs are dropped. This is what separates sub-objects from the
    /// grain of a grey interior: on the nedal cysts, neither the blobs' mean
    /// nor their peak level does (both overlap), and size does.
    pub min_voxels: usize,
}

impl Default for InteriorParams {
    fn default() -> Self {
        Self {
            inner: 0.85,
            smooth: 1.0,
            seed_smooth: 2.0,
            level: 0.65,
            prominence: 0.1,
            min_voxels: 250,
        }
    }
}

impl InteriorParams {
    fn check(&self) -> Result<()> {
        let ok = self.inner > 0.0
            && self.inner < OUTER
            && self.smooth > 0.0
            && self.seed_smooth > 0.0
            && self.level.is_finite()
            && self.prominence >= 0.0
            && self.min_voxels > 0;
        if ok {
            Ok(())
        } else {
            Err(Error::invalid(format!(
                "invalid interior parameters {self:?}"
            )))
        }
    }

    fn margin(&self) -> usize {
        (self.smooth.max(self.seed_smooth) * TRUNCATE).ceil() as usize
    }

    /// The halo a block needs to segment any object whose voxels lie within
    /// `max_extent` of its row's voxel (centre drift plus largest radius).
    pub fn halo(&self, max_extent: f64) -> usize {
        (OUTER * max_extent).ceil() as usize + self.margin() + 2
    }
}

/// The window [`segment_interior`] reads for `shape`: `(start, extent)` in
/// volume coordinates, clipped at the volume's faces only.
pub fn interior_window(
    shape: &StarShape,
    params: &InteriorParams,
    volume: [usize; 3],
) -> ([usize; 3], [usize; 3]) {
    let r = shape.radii.iter().copied().fold(0.0, f32::max) as f64;
    let half = (OUTER * r).ceil() + 1.0 + params.margin() as f64;
    let lo: [usize; 3] =
        std::array::from_fn(|a| (shape.center[a] - half).floor().max(0.0) as usize);
    let hi: [usize; 3] = std::array::from_fn(|a| {
        ((shape.center[a] + half).ceil() + 1.0)
            .max(0.0)
            .min(volume[a] as f64) as usize
    });
    let lo: [usize; 3] = std::array::from_fn(|a| lo[a].min(hi[a]));
    (lo, std::array::from_fn(|a| hi[a] - lo[a]))
}

/// The per-object columns, in [`Interior::summary`] order.
pub const SUMMARY_COLUMNS: [&str; 11] = [
    "inside_voxels",
    "inside_bright_fraction",
    "inside_blobs",
    "inside_components",
    "inside_blobs_per_component",
    "inside_largest_share",
    "inside_blob_diameter_median",
    "inside_blob_diameter_cv",
    "inside_blob_aspect_median",
    "inside_bright_s",
    "inside_dark_std",
];

/// The per-blob columns of [`blob_schema`] after `parent`.
pub const BLOB_COLUMNS: [&str; 10] = [
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "voxels",
    "diameter",
    "level",
    "s",
    "aspect",
    "siblings",
    "peak",
];

/// One sub-object.
#[derive(Debug, Clone, PartialEq)]
pub struct InteriorBlob {
    /// Its brightest voxel (seed smoothing; ties to the lowest index): the row
    /// key, and [`shape_label`] of it is the blob's label.
    pub peak: [usize; 3],
    pub voxels: usize,
    pub centroid: [f64; 3],
    /// Mean normalised level.
    pub level: f64,
    /// Shape-adapted radius of the centroid.
    pub s: f64,
    /// sqrt(smallest / largest eigenvalue of the voxel covariance); 1 is round.
    pub aspect: f64,
    /// Blobs in its connected bright region, itself included; above 1 means
    /// touching neighbours (fused or packed).
    pub siblings: usize,
    /// Level at its peak, at the seed smoothing.
    pub peak_level: f64,
}

/// A run of `len` voxels along x from `at`, all labelled `label`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelRun {
    pub label: u64,
    pub at: [usize; 3],
    pub len: usize,
}

/// One object's interior.
#[derive(Debug, Clone, PartialEq)]
pub struct Interior {
    pub summary: [f64; SUMMARY_COLUMNS.len()],
    pub blobs: Vec<InteriorBlob>,
    pub runs: Vec<LabelRun>,
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Segment the interior of `shape` from `window`, the image over
/// [`interior_window`] (`start` is its origin in the volume).
///
/// `rivals` are other objects that may overlap it: a voxel inside several
/// belongs to the one it is relatively deepest in (`|p - c| / r(direction)`
/// smallest, ties to the smaller label), the rule [`StarConvexLabelsOp`]
/// paints by, so every sub-object lies in its object's painted region and no
/// two objects segment the same voxel.
///
/// [`StarConvexLabelsOp`]: crate::ops::star_convex::StarConvexLabelsOp
pub fn segment_interior(
    window: ArrayView3<'_, f64>,
    start: [usize; 3],
    volume: [usize; 3],
    shape: &StarShape,
    rivals: &[&StarShape],
    rays: &RaySet,
    params: &InteriorParams,
) -> Result<Interior> {
    params.check()?;
    if shape.radii.len() != rays.len() {
        return Err(Error::invalid(format!(
            "a shape with {} rays and a ray set of {}",
            shape.radii.len(),
            rays.len()
        )));
    }
    let dim = window.raw_dim();
    let n = [dim[0], dim[1], dim[2]];
    let smooth = |sigma: f64| -> Result<Array3<f64>> {
        let g = Gaussian::isotropic(sigma, TRUNCATE)?;
        let mut out = Array3::<f64>::zeros(dim);
        gaussian_smooth_into_with(window, g.kernels(), g.boundary(), out.view_mut())?;
        Ok(out)
    };
    let fine = smooth(params.smooth)?;
    let coarse = smooth(params.seed_smooth)?;
    let c = shape.center;
    let s_of = |d: [f64; 3]| -> f64 {
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len < 1e-9 {
            0.0
        } else {
            len / (rays.interpolate(&shape.radii, d) as f64).max(1e-6)
        }
    };
    // relative depth, in f32 exactly as the painter computes it
    let depth = |o: &StarShape, i: [usize; 3]| -> f32 {
        let d: [f64; 3] = std::array::from_fn(|a| (start[a] + i[a]) as f64 - o.center[a]);
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() as f32;
        len / rays.interpolate(&o.radii, d).max(1e-6)
    };
    let s = Array3::from_shape_fn(dim, |(z, y, x)| depth(shape, [z, y, x]));
    // the rivals that can reach the window, with their labels
    let label = shape_label(shape.at, volume);
    let rivals: Vec<(&StarShape, u64, f64)> = rivals
        .iter()
        .filter(|o| o.at != shape.at)
        .map(|o| {
            let r = o.radii.iter().copied().fold(0.0, f32::max) as f64 + 1.0;
            (*o, shape_label(o.at, volume), r)
        })
        .filter(|(o, _, r)| {
            let gap: f64 = (0..3)
                .map(|a| {
                    let (lo, hi) = (start[a] as f64, (start[a] + n[a]) as f64 - 1.0);
                    let c = o.center[a];
                    let d = if c < lo {
                        lo - c
                    } else if c > hi {
                        c - hi
                    } else {
                        0.0
                    };
                    d * d
                })
                .sum();
            gap <= r * r
        })
        .collect();

    // 1. normalisation from the radial profile
    let bins = (OUTER / BIN).round() as usize;
    let mut profile = vec![Vec::new(); bins];
    for (i, &v) in s.indexed_iter() {
        if let Some(b) = profile.get_mut((v as f64 / BIN) as usize) {
            b.push(fine[i]);
        }
    }
    let profile: Vec<Option<f64>> = profile
        .iter_mut()
        .map(|b| (!b.is_empty()).then(|| median(b)))
        .collect();
    let extreme = |lo: f64, hi: f64, max: bool| {
        let range = (lo / BIN).round() as usize..((hi / BIN).round() as usize).min(bins);
        profile[range]
            .iter()
            .flatten()
            .copied()
            .reduce(if max { f64::max } else { f64::min })
    };
    let wall = extreme(0.8, OUTER, true).unwrap_or(0.0);
    let dark = extreme(0.5, 1.0, false).unwrap_or(wall).min(wall);
    let scale = (wall - dark).max(1.0);
    let norm = |v: f64| (v - dark) / scale;

    // 2. interior and bright phase
    let inside = Array3::from_shape_fn(dim, |(z, y, x)| {
        let own = s[[z, y, x]];
        (own as f64) < params.inner
            && rivals.iter().all(|&(o, l, _)| {
                let theirs = depth(o, [z, y, x]);
                !(theirs < own || (theirs == own && l < label))
            })
    });
    let bright = Array3::from_shape_fn(dim, |(z, y, x)| {
        inside[[z, y, x]] && norm(fine[[z, y, x]]) >= params.level
    });
    let interior_voxels = inside.iter().filter(|&&b| b).count();
    let bright_voxels = bright.iter().filter(|&&b| b).count();

    // 3. seeds, flood, merge by dynamics
    let field = Array3::from_shape_fn(dim, |(z, y, x)| {
        if bright[[z, y, x]] {
            norm(coarse[[z, y, x]])
        } else {
            f64::NEG_INFINITY
        }
    });
    let maxima = regional_maxima_with(field.view(), Connectivity::FacesEdgesAndCorners)?;
    let mut seeds = Array3::<u32>::zeros(dim);
    let count = label_members_into_with(
        n,
        Connectivity::FacesEdgesAndCorners,
        |p| maxima[p] && bright[p],
        seeds.view_mut(),
    )? as usize;
    let cost = field.mapv(|v| if v.is_finite() { -v } else { 0.0 });
    let mut basins = Array3::<u32>::zeros(dim);
    seeded_watershed_into(
        cost.view(),
        seeds.view(),
        Some(bright.view()),
        Separation::Adjacent,
        basins.view_mut(),
    )?;
    let mut peak = vec![f64::NEG_INFINITY; count + 1];
    for (i, &l) in seeds.indexed_iter() {
        if l != 0 {
            peak[l as usize] = peak[l as usize].max(field[i]);
        }
    }
    let mut saddles: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    for ((z, y, x), &a) in basins.indexed_iter() {
        if a == 0 {
            continue;
        }
        for step in [[1, 0, 0], [0, 1, 0], [0, 0, 1]] {
            let j = [z + step[0], y + step[1], x + step[2]];
            if (0..3).any(|k| j[k] >= n[k]) {
                continue;
            }
            let b = basins[j];
            if b == 0 || b == a {
                continue;
            }
            let v = field[[z, y, x]].min(field[j]);
            let e = saddles.entry((a.min(b), a.max(b))).or_insert(v);
            *e = e.max(v);
        }
    }
    let mut pairs: Vec<((u32, u32), f64)> = saddles.into_iter().collect();
    pairs.sort_by(|p, q| q.1.total_cmp(&p.1).then(p.0.cmp(&q.0)));
    let mut parent: Vec<usize> = (0..=count).collect();
    for ((a, b), saddle) in pairs {
        let (ra, rb) = (find(&mut parent, a as usize), find(&mut parent, b as usize));
        if ra == rb || peak[ra].min(peak[rb]) - saddle >= params.prominence {
            continue;
        }
        let (keep, gone) = (ra.min(rb), ra.max(rb));
        parent[gone] = keep;
        peak[keep] = peak[keep].max(peak[gone]);
    }

    // 4. blobs
    struct Acc {
        voxels: usize,
        sum: [f64; 3],
        cross: [f64; 6],
        level: f64,
        peak: ([usize; 3], f64),
    }
    let mut acc: BTreeMap<usize, Acc> = BTreeMap::new();
    let mut root = Array3::<u32>::zeros(dim);
    for ((z, y, x), &l) in basins.indexed_iter() {
        if l == 0 {
            continue;
        }
        let r = find(&mut parent, l as usize);
        root[[z, y, x]] = r as u32;
        let p = [z, y, x].map(|v| v as f64);
        let a = acc.entry(r).or_insert(Acc {
            voxels: 0,
            sum: [0.0; 3],
            cross: [0.0; 6],
            level: 0.0,
            peak: ([z, y, x], f64::NEG_INFINITY),
        });
        a.voxels += 1;
        for k in 0..3 {
            a.sum[k] += p[k];
        }
        let cross = [
            p[0] * p[0],
            p[1] * p[1],
            p[2] * p[2],
            p[0] * p[1],
            p[0] * p[2],
            p[1] * p[2],
        ];
        for k in 0..6 {
            a.cross[k] += cross[k];
        }
        a.level += norm(fine[[z, y, x]]);
        // row-major order: a later equal value does not replace the first
        if field[[z, y, x]] > a.peak.1 {
            a.peak = ([z, y, x], field[[z, y, x]]);
        }
    }
    acc.retain(|_, a| a.voxels >= params.min_voxels);
    let mut components = Array3::<u32>::zeros(dim);
    label_members_into_with(n, Connectivity::Faces, |p| bright[p], components.view_mut())?;
    let mut per_component: BTreeMap<u32, usize> = BTreeMap::new();
    for a in acc.values() {
        *per_component.entry(components[a.peak.0]).or_insert(0) += 1;
    }
    let mut blobs = Vec::with_capacity(acc.len());
    let mut label_of: BTreeMap<usize, u64> = BTreeMap::new();
    for (&r, a) in &acc {
        let m = a.voxels as f64;
        let mean: [f64; 3] = std::array::from_fn(|k| a.sum[k] / m);
        let pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
        let cov: [f64; 6] =
            std::array::from_fn(|k| a.cross[k] / m - mean[pairs[k].0] * mean[pairs[k].1]);
        let [high, _, low] = symmetric_eigenvalues(cov);
        let peak = std::array::from_fn(|k| start[k] + a.peak.0[k]);
        let centroid: [f64; 3] = std::array::from_fn(|k| start[k] as f64 + mean[k]);
        label_of.insert(r, shape_label(peak, volume));
        blobs.push(InteriorBlob {
            peak,
            voxels: a.voxels,
            centroid,
            level: a.level / m,
            s: s_of(std::array::from_fn(|k| centroid[k] - c[k])),
            aspect: if high > 0.0 {
                (low.max(0.0) / high).sqrt()
            } else {
                1.0
            },
            siblings: per_component[&components[a.peak.0]],
            peak_level: a.peak.1,
        });
    }
    blobs.sort_by_key(|a| a.peak);

    let mut runs = Vec::new();
    for z in 0..n[0] {
        for y in 0..n[1] {
            let mut x = 0;
            while x < n[2] {
                let Some(&label) = label_of.get(&(root[[z, y, x]] as usize)) else {
                    x += 1;
                    continue;
                };
                let r = root[[z, y, x]];
                let from = x;
                while x < n[2] && root[[z, y, x]] == r {
                    x += 1;
                }
                runs.push(LabelRun {
                    label,
                    at: [start[0] + z, start[1] + y, start[2] + from],
                    len: x - from,
                });
            }
        }
    }

    // summary
    let labelled: usize = blobs.iter().map(|b| b.voxels).sum();
    let mut diameters: Vec<f64> = blobs
        .iter()
        .map(|b| (6.0 * b.voxels as f64 / std::f64::consts::PI).cbrt())
        .collect();
    let d_mean = diameters.iter().sum::<f64>() / diameters.len().max(1) as f64;
    let d_sd = (diameters.iter().map(|d| (d - d_mean).powi(2)).sum::<f64>()
        / diameters.len().max(1) as f64)
        .sqrt();
    let mut aspects: Vec<f64> = blobs.iter().map(|b| b.aspect).collect();
    let regions = per_component.len();
    let (mut bright_s, mut dark) = (0.0, Vec::new());
    for (i, &v) in inside.indexed_iter() {
        if bright[i] {
            bright_s += s[i] as f64;
        } else if v {
            dark.push(norm(fine[i]));
        }
    }
    let dark_mean = dark.iter().sum::<f64>() / dark.len().max(1) as f64;
    let dark_std = (dark.iter().map(|v| (v - dark_mean).powi(2)).sum::<f64>()
        / dark.len().max(1) as f64)
        .sqrt();
    let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
    let summary = [
        interior_voxels as f64,
        ratio(bright_voxels as f64, interior_voxels as f64),
        blobs.len() as f64,
        regions as f64,
        ratio(blobs.len() as f64, regions as f64),
        ratio(
            blobs.iter().map(|b| b.voxels).max().unwrap_or(0) as f64,
            labelled as f64,
        ),
        median(&mut diameters),
        ratio(d_sd, d_mean),
        median(&mut aspects),
        ratio(bright_s, bright_voxels as f64),
        dark_std,
    ];
    Ok(Interior {
        summary,
        blobs,
        runs,
    })
}

// ------------------------------------------------------------- runs wire --

pub fn encode_runs(runs: &[LabelRun]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + runs.len() * 40);
    out.extend_from_slice(&RUNS_MAGIC.to_le_bytes());
    out.extend_from_slice(&(runs.len() as u64).to_le_bytes());
    for r in runs {
        out.extend_from_slice(&r.label.to_le_bytes());
        for v in r.at {
            out.extend_from_slice(&(v as u64).to_le_bytes());
        }
        out.extend_from_slice(&(r.len as u64).to_le_bytes());
    }
    out
}

pub fn decode_runs(bytes: &[u8]) -> Result<Vec<LabelRun>> {
    let bad = || Error::invalid("not a label-run fragment");
    let word = |i: usize| -> Result<u64> {
        bytes
            .get(8 * i..8 * i + 8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("eight")))
            .ok_or_else(bad)
    };
    if word(0)? != RUNS_MAGIC {
        return Err(bad());
    }
    let count = word(1)? as usize;
    if bytes.len() != 16 + 40 * count {
        return Err(bad());
    }
    (0..count)
        .map(|k| {
            let o = 2 + 5 * k;
            Ok(LabelRun {
                label: word(o)?,
                at: [
                    word(o + 1)? as usize,
                    word(o + 2)? as usize,
                    word(o + 3)? as usize,
                ],
                len: word(o + 4)? as usize,
            })
        })
        .collect()
}

/// The blob rows: `parent` (the object's [`shape_label`]) then [`BLOB_COLUMNS`].
pub fn blob_schema() -> Result<Schema> {
    let mut columns = vec![Column::u64("parent")];
    columns.extend(BLOB_COLUMNS.iter().map(|c| Column::f64(*c)));
    Schema::new(columns)
}

/// The summary rows, keyed at the object's row voxel: [`SUMMARY_COLUMNS`].
pub fn interior_summary_schema() -> Result<Schema> {
    Schema::new(SUMMARY_COLUMNS.iter().map(|c| Column::f64(*c)).collect())
}

// ------------------------------------------------------------ the op --

/// **Object rows and their shapes in; per-object summaries, blob rows and
/// blob voxels out.**
///
/// Segments, with [`segment_interior`], every object whose row (in `rows`) its
/// block owns, using the shape keyed at the same voxel in `shapes` (both
/// streams of this block only). Reads `image` with [`InteriorParams::halo`] of
/// `max_extent`; an object reaching further from its row voxel is refused.
pub struct InteriorOp {
    name: &'static str,
    image: ImageId,
    rows: (String, usize),
    rows_schema: Schema,
    shapes: (String, usize),
    nearby: (String, usize),
    rays: Arc<RaySet>,
    params: InteriorParams,
    max_extent: f64,
    summary: (String, Lifecycle),
    blobs: (String, Lifecycle),
    runs: (String, Lifecycle),
    summary_schema: Schema,
    blob_schema: Schema,
}

/// The input and output streams of [`InteriorOp`] beside its rows and shapes.
pub struct InteriorStreams {
    /// Other blocks' objects near this block's, from [`NearbyShapesOp`].
    pub nearby: (String, usize),
    pub summary: (String, Lifecycle),
    pub blobs: (String, Lifecycle),
    pub runs: (String, Lifecycle),
}

impl InteriorOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &'static str,
        image: ImageId,
        rows: (impl Into<String>, usize),
        rows_schema: Schema,
        shapes: (impl Into<String>, usize),
        rays: Arc<RaySet>,
        params: InteriorParams,
        max_extent: f64,
        streams: InteriorStreams,
    ) -> Result<Self> {
        params.check()?;
        if !(max_extent > 0.0 && max_extent.is_finite()) {
            return Err(Error::invalid(format!(
                "an interior op needs a positive extent bound, got {max_extent}"
            )));
        }
        Ok(Self {
            name,
            image,
            rows: (rows.0.into(), rows.1),
            rows_schema,
            shapes: (shapes.0.into(), shapes.1),
            nearby: streams.nearby,
            rays,
            params,
            max_extent,
            summary: streams.summary,
            blobs: streams.blobs,
            runs: streams.runs,
            summary_schema: interior_summary_schema()?,
            blob_schema: blob_schema()?,
        })
    }

    pub fn summary_schema(&self) -> &Schema {
        &self.summary_schema
    }

    pub fn blob_schema(&self) -> &Schema {
        &self.blob_schema
    }
}

impl FragmentOp for InteriorOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        self.params.halo(self.max_extent)
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(
            self.image,
            Reach::symmetric([self.params.halo(self.max_extent); 3]),
        )]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        // each object is segmented by the block owning its row, from its own
        // window; nothing is folded
        Some(SeamFold::PerBlock)
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![
            FragmentInput::own(&self.rows.0, self.rows.1),
            FragmentInput::own(&self.shapes.0, self.shapes.1),
            FragmentInput::own(&self.nearby.0, self.nearby.1),
        ]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![
            FragmentOutput::new(&self.summary.0, self.summary.1, Coverage::EveryBlock)
                .sized(SidecarSize::row_table(&self.summary_schema, 1)),
            FragmentOutput::new(&self.blobs.0, self.blobs.1, Coverage::EveryBlock),
            FragmentOutput::new(&self.runs.0, self.runs.1, Coverage::EveryBlock),
        ]
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "the interior op reads its image through apply_with",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let volume = at.volume();
        let rows = sealed_rows(
            volume,
            &self.rows_schema,
            at.own(&self.rows.0).unwrap_or(&[]),
        )?;
        let mut shapes = BTreeMap::new();
        if let Some(bytes) = at.own(&self.shapes.0) {
            let (rays, found) = decode_shapes(bytes)?;
            if rays != self.rays.len() {
                return Err(Error::invalid(format!(
                    "shapes carry {rays} rays, the interior op was built for {}",
                    self.rays.len()
                )));
            }
            shapes.extend(found.into_iter().map(|s| (s.at, s)));
        }
        let mut nearby = Vec::new();
        if let Some(bytes) = at.own(&self.nearby.0) {
            nearby = decode_shapes(bytes)?.1;
        }
        let objects: Vec<&StarShape> = rows
            .iter()
            .map(|row| {
                shapes
                    .get(&row.at)
                    .ok_or_else(|| Error::invalid(format!("no shape for the row at {:?}", row.at)))
            })
            .collect::<Result<_>>()?;
        let mut summary = RowBuilder::new(Arc::new(self.summary_schema.clone()));
        let mut blob_rows = RowBuilder::new(Arc::new(self.blob_schema.clone()));
        let mut runs = Vec::new();
        if !objects.is_empty() {
            let raw = sources.get(self.image.index())?.as_array()?;
            let read = [at.read.start[0], at.read.start[1], at.read.start[2]];
            let read_end: [usize; 3] = std::array::from_fn(|a| read[a] + at.read.shape[a]);
            let results: Vec<Interior> = objects
                .par_iter()
                .map(|shape| {
                    let (start, extent) = interior_window(shape, &self.params, volume);
                    for a in 0..3 {
                        if start[a] < read[a] || start[a] + extent[a] > read_end[a] {
                            return Err(Error::invalid(format!(
                                "the object at {:?} reaches beyond the interior halo (extent \
                                 bound {:.1}); raise the bound",
                                shape.at, self.max_extent
                            )));
                        }
                    }
                    let window = window_of(raw, start, extent, read)?;
                    let rivals: Vec<&StarShape> =
                        objects.iter().copied().chain(nearby.iter()).collect();
                    segment_interior(
                        window.view(),
                        start,
                        volume,
                        shape,
                        &rivals,
                        &self.rays,
                        &self.params,
                    )
                })
                .collect::<Result<_>>()?;
            for (shape, interior) in objects.iter().zip(results) {
                summary.push(shape.at, &interior.summary.map(Value::F64))?;
                let parent = shape_label(shape.at, volume);
                for b in &interior.blobs {
                    let mut values = vec![Value::U64(parent)];
                    values.extend(
                        [
                            b.centroid[0],
                            b.centroid[1],
                            b.centroid[2],
                            b.voxels as f64,
                            (6.0 * b.voxels as f64 / std::f64::consts::PI).cbrt(),
                            b.level,
                            b.s,
                            b.aspect,
                            b.siblings as f64,
                            b.peak_level,
                        ]
                        .map(Value::F64),
                    );
                    blob_rows.push(b.peak, &values)?;
                }
                runs.extend(interior.runs);
            }
        }
        Ok(BlockOutput::fragment(&self.summary.0, summary.encode())
            .with_fragment(&self.blobs.0, blob_rows.encode())
            .with_fragment(&self.runs.0, encode_runs(&runs)))
    }
}

/// A view of any element type `dispatch_f64_input!` hands over, as `f64`.
///
/// A function rather than `mapv(|v| v.into())` in the macro's body, because
/// the body is expanded once per element type and for `f64` that conversion is
/// the identity.
fn widened<T: Copy + Into<f64>>(view: ArrayView3<'_, T>) -> Array3<f64> {
    view.mapv(Into::into)
}

/// `extent` voxels from `start` (volume coordinates) of a buffer read from
/// `read`, widened to `f64`.
fn window_of(
    raw: &Voxels,
    start: [usize; 3],
    extent: [usize; 3],
    read: [usize; 3],
) -> Result<Array3<f64>> {
    let o: [usize; 3] = std::array::from_fn(|a| start[a] - read[a]);
    let cut = s![
        o[0]..o[0] + extent[0],
        o[1]..o[1] + extent[1],
        o[2]..o[2] + extent[2]
    ];
    dispatch_f64_input!(
        raw,
        Error::invalid("the interior op cannot read half-precision"),
        |view| { Ok(widened(view.slice(cut))) }
    )
}

// ------------------------------------------------------- nearby shapes --

/// **Rows and shapes of this block and its neighbours in; the other blocks'
/// objects that come near this block's out**, as shapes (sorted by row
/// voxel), for [`InteriorOp`] to split overlaps by.
///
/// Near: the two objects' bounding spheres (centre, largest radius + 1)
/// intersect. Such an object's row lies within `2 * max_extent` of one of
/// this block's, so that must not exceed the block edge (checked).
pub struct NearbyShapesOp {
    name: &'static str,
    rows: (String, usize),
    rows_schema: Schema,
    shapes: (String, usize),
    rays: usize,
    output: (String, Lifecycle),
    max_extent: f64,
}

impl NearbyShapesOp {
    pub fn new(
        name: &'static str,
        rows: (impl Into<String>, usize),
        rows_schema: Schema,
        shapes: (impl Into<String>, usize),
        rays: usize,
        output: (impl Into<String>, Lifecycle),
        max_extent: f64,
    ) -> Self {
        Self {
            name,
            rows: (rows.0.into(), rows.1),
            rows_schema,
            shapes: (shapes.0.into(), shapes.1),
            rays,
            output: (output.0.into(), output.1),
            max_extent,
        }
    }
}

impl FragmentOp for NearbyShapesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![
            FragmentInput::own(&self.rows.0, self.rows.1).with_reach([1, 1, 1]),
            FragmentInput::own(&self.shapes.0, self.shapes.1).with_reach([1, 1, 1]),
        ]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            &self.output.0,
            self.output.1,
            Coverage::EveryBlock,
        )]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        // a set of shapes, emitted sorted
        Some(SeamFold::Unordered)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let block = at.grid.block();
        for (a, &edge) in block.iter().enumerate() {
            if 2.0 * self.max_extent > edge as f64 {
                return Err(Error::invalid(format!(
                    "overlapping objects can have rows {:.1} voxels apart, more than the block \
                     edge {edge} on axis {a}; use larger blocks",
                    2.0 * self.max_extent
                )));
            }
        }
        let volume = at.volume();
        let mut kept = std::collections::BTreeSet::new();
        for (_key, bytes) in at.fragments(&self.rows.0) {
            for row in sealed_rows(volume, &self.rows_schema, bytes)? {
                kept.insert(row.at);
            }
        }
        let mut shapes: BTreeMap<[usize; 3], StarShape> = BTreeMap::new();
        for (_key, bytes) in at.fragments(&self.shapes.0) {
            let (rays, found) = decode_shapes(bytes)?;
            if rays != self.rays {
                return Err(Error::invalid(format!(
                    "shapes carry {rays} rays, expected {}",
                    self.rays
                )));
            }
            shapes.extend(
                found
                    .into_iter()
                    .filter(|s| kept.contains(&s.at))
                    .map(|s| (s.at, s)),
            );
        }
        let core = at.core;
        let owned = |p: [usize; 3]| {
            (0..3).all(|a| core.start[a] <= p[a] && p[a] < core.start[a] + core.shape[a])
        };
        let reach = |s: &StarShape| s.radii.iter().copied().fold(0.0, f32::max) as f64 + 1.0;
        let (own, other): (Vec<&StarShape>, Vec<&StarShape>) =
            shapes.values().partition(|s| owned(s.at));
        let near: Vec<StarShape> = other
            .into_iter()
            .filter(|x| {
                own.iter().any(|o| {
                    let d: f64 = (0..3)
                        .map(|a| (o.center[a] - x.center[a]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    d <= reach(o) + reach(x)
                })
            })
            .cloned()
            .collect();
        Ok(BlockOutput::fragment(
            &self.output.0,
            encode_shapes(self.rays, &near),
        ))
    }
}

// ---------------------------------------------------------- the painter --

/// **Label runs in, a `u64` label image out.**
///
/// Paints, for its own core, the runs of this block and its 26 neighbours. A
/// voxel in runs of two labels takes the smaller, so the answer does not depend
/// on the gather order. Runs must lie within one block of the block that
/// emitted them; `max_extent` (the emitter's bound on how far its runs reach
/// from its own core) is checked against the grid when the plan is built.
pub struct RunLabelsOp {
    name: &'static str,
    runs: (String, usize),
    max_extent: f64,
}

impl RunLabelsOp {
    pub fn new(name: &'static str, runs: (impl Into<String>, usize), max_extent: f64) -> Self {
        Self {
            name,
            runs: (runs.0.into(), runs.1),
            max_extent,
        }
    }
}

/// Metadata selector shared by painters of run-encoded image outputs.
pub fn prepare_run_paint(
    at: &crate::fragment::PreparationView<'_>,
    stream: &str,
    max_extent: f64,
) -> Result<crate::fragment::BlockPreparation> {
    use crate::fragment::BlockPreparation;
    if !at.complete(stream) {
        return Ok(BlockPreparation::Run);
    }
    // Check neighbours as well as the owner: a run can enter this core from
    // a different block. Keep the dense path's extent validation in force.
    if max_extent > at.core.shape.iter().copied().min().unwrap_or(0) as f64 {
        return Ok(BlockPreparation::Run);
    }
    for (_, bytes) in at.fragments(stream) {
        for r in decode_runs(bytes)? {
            let [z, y, x] = r.at;
            if z >= at.core.start[0]
                && z < at.core.start[0] + at.core.shape[0]
                && y >= at.core.start[1]
                && y < at.core.start[1] + at.core.shape[1]
                && x < at.core.start[2] + at.core.shape[2]
                && x + r.len > at.core.start[2]
            {
                return Ok(BlockPreparation::Run);
            }
        }
    }
    Ok(BlockPreparation::Complete {
        fragments: Vec::new(),
        constant: Some(0.0),
    })
}

impl FragmentOp for RunLabelsOp {
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

    /// Unlabelled voxels are `0`, and most voxels are unlabelled.
    fn background(&self) -> Option<f64> {
        Some(0.0)
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.runs.0, self.runs.1).with_reach([1, 1, 1])]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn prepares(&self) -> bool {
        true
    }

    fn prepare(
        &self,
        at: &crate::fragment::PreparationView<'_>,
    ) -> Result<crate::fragment::BlockPreparation> {
        prepare_run_paint(at, &self.runs.0, self.max_extent)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let block = at.grid.block();
        for (a, &edge) in block.iter().enumerate() {
            if self.max_extent > edge as f64 {
                return Err(Error::invalid(format!(
                    "label runs reach {:.1} voxels beyond their block, more than the block edge \
                     {edge} on axis {a}; use larger blocks",
                    self.max_extent
                )));
            }
        }
        let core = at.core;
        let lo = [core.start[0], core.start[1], core.start[2]];
        let hi: [usize; 3] = std::array::from_fn(|a| lo[a] + core.shape[a]);
        let mut buffer = at.output_buffer(0.0)?;
        if let Some(out) = buffer.as_array_mut() {
            let mut view = out.view_mut::<u64>()?;
            let origin = [at.read.start[0], at.read.start[1], at.read.start[2]];
            for (_key, bytes) in at.fragments(&self.runs.0) {
                for r in decode_runs(bytes)? {
                    let [z, y, x] = r.at;
                    if z < lo[0] || z >= hi[0] || y < lo[1] || y >= hi[1] {
                        continue;
                    }
                    for x in x.max(lo[2])..(x + r.len).min(hi[2]) {
                        let slot = &mut view[[z - origin[0], y - origin[1], x - origin[2]]];
                        if *slot == 0 || r.label < *slot {
                            *slot = r.label;
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

    const VOLUME: [usize; 3] = [64, 96, 128];
    const WALL: f64 = 20.0;

    /// Background 100; per shell a bright wall `[WALL - 2, WALL]` at 220, a dark
    /// band 4 voxels under it at 50, an interior at 70, and bright balls
    /// `(offset, radius)` at 200 inside.
    fn phantom(shells: &[([f64; 3], Vec<([f64; 3], f64)>)]) -> Array3<f64> {
        Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
            let p = [z as f64, y as f64, x as f64];
            let dist = |a: [f64; 3], b: [f64; 3]| {
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
            };
            let mut v = 100.0 + ((z * 7 + y * 13 + x * 3) % 11) as f64;
            for (c, balls) in shells {
                let r = dist(p, *c);
                if r <= WALL {
                    v = if r > WALL - 2.0 {
                        220.0
                    } else if r > WALL - 6.0 {
                        50.0
                    } else {
                        70.0 + ((z + y + x) % 5) as f64
                    };
                    for (o, br) in balls {
                        let b = [c[0] + o[0], c[1] + o[1], c[2] + o[2]];
                        if dist(p, b) <= *br {
                            v = 200.0;
                        }
                    }
                }
            }
            v
        })
    }

    /// The defaults, with blobs from 30 voxels: the phantom's balls are small.
    fn small() -> InteriorParams {
        InteriorParams {
            min_voxels: 30,
            ..InteriorParams::default()
        }
    }

    fn sphere(at: [usize; 3], centre: [f64; 3], rays: &RaySet) -> StarShape {
        StarShape {
            at,
            center: centre,
            radii: vec![WALL as f32; rays.len()],
        }
    }

    fn segment(image: &Array3<f64>, shape: &StarShape, rays: &RaySet) -> Interior {
        let params = small();
        let (start, extent) = interior_window(shape, &params, VOLUME);
        let window = image
            .slice(s![
                start[0]..start[0] + extent[0],
                start[1]..start[1] + extent[1],
                start[2]..start[2] + extent[2]
            ])
            .to_owned();
        segment_interior(window.view(), start, VOLUME, shape, &[], rays, &params).unwrap()
    }

    fn column(name: &str) -> usize {
        SUMMARY_COLUMNS.iter().position(|&c| c == name).unwrap()
    }

    #[test]
    fn separate_balls_are_counted_and_a_dark_interior_has_none() {
        let rays = RaySet::fibonacci(300, 8);
        let c = [32.0, 48.0, 64.0];
        let balls = vec![
            ([0.0, 0.0, 7.0], 3.5),
            ([0.0, 0.0, -7.0], 3.5),
            ([0.0, 7.0, 0.0], 3.5),
            ([0.0, -7.0, 0.0], 3.5),
            ([7.0, 0.0, 0.0], 3.5),
        ];
        let image = phantom(&[(c, balls)]);
        let got = segment(&image, &sphere([32, 48, 64], c, &rays), &rays);
        assert_eq!(got.blobs.len(), 5, "{:?}", got.summary);
        assert_eq!(got.summary[column("inside_components")], 5.0);
        assert!(got.blobs.iter().all(|b| b.siblings == 1 && b.aspect > 0.7));
        let d = got.summary[column("inside_blob_diameter_median")];
        assert!((d - 7.0).abs() < 1.5, "{d}");
        let painted: usize = got.runs.iter().map(|r| r.len).sum();
        assert_eq!(painted, got.blobs.iter().map(|b| b.voxels).sum::<usize>());

        let empty = phantom(&[(c, Vec::new())]);
        let got = segment(&empty, &sphere([32, 48, 64], c, &rays), &rays);
        assert_eq!(got.blobs.len(), 0, "{:?}", got.summary);
        assert_eq!(got.summary[column("inside_bright_fraction")], 0.0);
    }

    #[test]
    fn touching_lobes_stay_apart_and_a_merged_mass_is_one() {
        let rays = RaySet::fibonacci(300, 8);
        let c = [32.0, 48.0, 64.0];
        // two balls overlapping a little: one region, two lobes
        let lobes = phantom(&[(c, vec![([0.0, 0.0, -5.0], 5.0), ([0.0, 0.0, 5.0], 5.0)])]);
        let got = segment(&lobes, &sphere([32, 48, 64], c, &rays), &rays);
        assert_eq!(got.blobs.len(), 2, "{:?}", got.summary);
        assert_eq!(got.summary[column("inside_components")], 1.0);
        assert!(got.blobs.iter().all(|b| b.siblings == 2));
        // the same balls almost fully overlapping: one blob
        let mass = phantom(&[(c, vec![([0.0, 0.0, -1.5], 6.0), ([0.0, 0.0, 1.5], 6.0)])]);
        let got = segment(&mass, &sphere([32, 48, 64], c, &rays), &rays);
        assert_eq!(got.blobs.len(), 1, "{:?}", got.summary);
    }

    /// Planned over two lattices — one whose seams cut the objects — the
    /// summaries, blob rows and painted labels are identical; a ball across
    /// the plane of equal depth of two overlapping objects is split there, each
    /// half segmented by the object it is deeper in.
    #[test]
    fn planned_interiors_do_not_depend_on_the_grid() {
        let rays = Arc::new(RaySet::fibonacci(300, 8));
        let a = [30.0, 40.0, 38.0];
        let b = [33.0, 52.0, 88.0];
        let c = [33.0, 52.0, 118.0];
        let image = phantom(&[
            (a, vec![([0.0, 0.0, 6.0], 4.0), ([0.0, 0.0, -6.0], 4.0)]),
            (
                b,
                vec![
                    ([0.0, 5.0, 0.0], 4.0),
                    ([0.0, -5.0, 0.0], 4.0),
                    ([0.0, 0.0, 7.0], 3.5),
                ],
            ),
            (c, vec![([0.0, -6.0, -15.0], 3.5)]),
        ]);
        let schema = Schema::new(vec![Column::f64("score")]).unwrap();
        let shapes = vec![
            sphere([30, 41, 37], a, &rays),
            sphere([34, 52, 88], b, &rays),
            sphere([33, 52, 118], c, &rays),
        ];
        let rows: Vec<RowValues> = shapes
            .iter()
            .map(|s| RowValues::new(s.at, vec![Value::F64(1.0)]))
            .collect();
        let extent = 3.0 + WALL;
        let params = small();
        let mut answers = Vec::new();
        for block in [[64, 96, 128], [64, 48, 64]] {
            let grid = BlockGrid::new(VOLUME, block).unwrap();
            let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
            let source = builder
                .fragments(RowSourceOp::new(
                    "objects",
                    "objects",
                    Lifecycle::DeleteOnExit,
                    schema.clone(),
                    rows.clone(),
                ))
                .unwrap();
            let shaped = builder
                .fragments(ShapeSourceOp {
                    shapes: shapes.clone(),
                    rays: rays.len(),
                })
                .unwrap();
            let near = builder
                .fragments(NearbyShapesOp::new(
                    "nearby",
                    ("objects", source.index()),
                    schema.clone(),
                    ("shapes", shaped.index()),
                    rays.len(),
                    ("nearby", Lifecycle::DeleteOnExit),
                    extent,
                ))
                .unwrap();
            let op = InteriorOp::new(
                "interior",
                ImageId::from(0),
                ("objects", source.index()),
                schema.clone(),
                ("shapes", shaped.index()),
                rays.clone(),
                params.clone(),
                extent,
                InteriorStreams {
                    nearby: ("nearby".into(), near.index()),
                    summary: ("summary".into(), Lifecycle::Persistent),
                    blobs: ("blobs".into(), Lifecycle::Persistent),
                    runs: ("runs".into(), Lifecycle::DeleteOnExit),
                },
            )
            .unwrap();
            let (summary_schema, blob_schema) =
                (op.summary_schema().clone(), op.blob_schema().clone());
            let inside = builder.fragments(op).unwrap();
            builder
                .fragments(RunLabelsOp::new(
                    "paint",
                    ("runs", inside.index()),
                    OUTER * extent + 2.0,
                ))
                .unwrap();
            let assembly = builder.finish().unwrap();
            let env = ArrayEnvironment::for_decomposition(
                Voxels::from(image.clone()),
                &assembly.decomposition,
                block,
            )
            .unwrap();
            execute_phases(
                "interiors",
                &assembly.workflow,
                &assembly.decomposition,
                &Hints::default(),
                &env,
                &[],
                &assembly.work(),
            )
            .unwrap();
            let summary =
                collect_rows(&env, "summary", inside.index(), VOLUME, summary_schema).unwrap();
            let blobs = collect_rows(&env, "blobs", inside.index(), VOLUME, blob_schema).unwrap();
            let labels = env.output().view::<u64>().unwrap().to_owned();
            answers.push((summary, blobs, labels));
        }
        let (summary, blobs, labels) = &answers[0];
        assert_eq!(summary.len(), 3);
        assert_eq!(blobs.len(), 7);
        let distinct: std::collections::BTreeSet<u64> =
            labels.iter().copied().filter(|&l| l != 0).collect();
        assert_eq!(distinct.len(), 7);
        let parent_at = |x: usize| {
            let l = labels[[33, 46, x]];
            blobs
                .iter()
                .find(|r| shape_label(r.at, VOLUME) == l)
                .map(|r| r.values[0])
        };
        let label = |at| Some(Value::U64(shape_label(at, VOLUME)));
        assert_eq!(parent_at(101), label([34, 52, 88]));
        assert_eq!(parent_at(105), label([33, 52, 118]));
        assert_eq!(&answers[1].0, summary);
        assert_eq!(&answers[1].1, blobs);
        assert_eq!(&answers[1].2, labels);
    }

    /// Emits each shape from the block owning its row voxel.
    struct ShapeSourceOp {
        shapes: Vec<StarShape>,
        rays: usize,
    }

    impl FragmentOp for ShapeSourceOp {
        fn name(&self) -> &'static str {
            "shapes"
        }

        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }

        fn reads_pixels(&self) -> bool {
            false
        }

        fn seam_fold(&self) -> Option<SeamFold> {
            Some(SeamFold::PerBlock)
        }

        fn outputs(&self) -> Vec<FragmentOutput> {
            vec![FragmentOutput::new(
                "shapes",
                Lifecycle::DeleteOnExit,
                Coverage::EveryBlock,
            )]
        }

        fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
            let mine: Vec<StarShape> = self
                .shapes
                .iter()
                .filter(|s| {
                    (0..3).all(|a| {
                        at.core.start[a] <= s.at[a] && s.at[a] < at.core.start[a] + at.core.shape[a]
                    })
                })
                .cloned()
                .collect();
            Ok(BlockOutput::fragment(
                "shapes",
                crate::ops::star_convex::encode_shapes(self.rays, &mine),
            ))
        }
    }
}
