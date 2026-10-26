// SPDX-License-Identifier: MIT
//! Filled shell-interior territories from dark-sheet evidence and a multicut.
//!
//! This is independent of `interior`'s thresholded bright blobs. Every voxel
//! in the geometric interior belongs to exactly one connected territory.
//! Fine seeded watershed fragments are only computational units; signed RAG
//! costs and GAEC + Kernighan-Lin infer the final number of territories.
//! Boundary scores are heuristic evidence, not calibrated probabilities.

use super::agglomerate::{boundary_cost, multicut, KlOptions};
use super::distance::{distance_transform, DistanceParams};
use super::interior::{decode_runs, encode_runs, interior_window, InteriorParams, LabelRun};
use super::rag::{rag, RagOptions};
use super::ridge::{
    gaussian_smooth_into_with, ridge_response_into, symmetric_eigenvalues, PlateResponse, Polarity,
    ScaleSpace,
};
use super::rows::sealed_rows;
use super::smooth::Gaussian;
use super::star_convex::{decode_shapes, shape_label, StarShape};
use super::watershed::{seeded_watershed_into, Separation};
use crate::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, SeamFold,
    SidecarSize, SourceBlocks,
};
use crate::op::SourceInput;
use crate::reach::Reach;
use crate::sidecar::Lifecycle;
use crate::sphere::RaySet;
use crate::table::{Column, RowBuilder, Schema, Value};
use crate::{Dtype, Error, ImageId, Result, Voxels};
use ndarray::{s, Array3, ArrayView3};
use rayon::prelude::*;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

/// Lengths are in isotropic voxels, intensity is normalised dark-band=0, wall=1.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionParams {
    /// Explicit geometric wall exclusion; no intensity threshold defines foreground.
    pub inner: f64,
    /// Cap the foreground at the half-depth inner edge of the fitted radial
    /// dark band. If no supported dip exists, retain `inner`.
    pub adaptive_inner: bool,
    pub scales: Vec<f64>,
    /// Hessian sheet-strength sensitivity in normalised intensity units.
    pub strength: f64,
    /// Initial fragments are finer than the expected territories.
    pub seed_spacing: usize,
    /// Minimum seed clearance from boundary evidence and the geometric mask edge.
    pub core_distance: f64,
    /// Scores at or above this level are barriers for seed clearance only.
    pub core_boundary_level: f64,
    /// Neutral at 0.5; larger values favour splitting.
    pub boundary_prior: f64,
    /// Distance in voxels excluded from the graph at the mask edge.
    /// Solved core labels are extended through this margin afterwards.
    /// Zero retains the full-mask graph. Tiny components keep one core voxel.
    pub graph_margin: f64,
    /// Optional smoothed, wall-normalised minimum intensity for normal seeds.
    /// This never removes foreground; components without eligible seeds keep
    /// one fallback at their brightest voxel. None preserves v1 seeding.
    pub seed_level: Option<f64>,
    /// Quantile mixed with the mean shared-face evidence; a robust score statistic.
    pub face_quantile: f64,
    /// Merge smaller territories across their weakest supported interface;
    /// never remove foreground or force equal territory volumes. 0 disables.
    pub min_voxels: usize,
}
impl Default for PartitionParams {
    fn default() -> Self {
        Self {
            inner: 0.85,
            adaptive_inner: false,
            scales: vec![1.0, 1.5, 2.0],
            strength: 0.035,
            seed_spacing: 5,
            core_distance: 3.0,
            core_boundary_level: 0.4,
            boundary_prior: 0.5,
            graph_margin: 0.0,
            seed_level: None,
            face_quantile: 0.75,
            min_voxels: 32,
        }
    }
}
impl PartitionParams {
    pub fn check(&self) -> Result<()> {
        if !(self.inner > 0.0
            && self.inner < 1.0
            && self.strength > 0.0
            && self.strength.is_finite()
            && self.seed_spacing > 0
            && self.seed_spacing <= 1024
            && self.core_distance.is_finite()
            && self.core_distance >= 0.0
            && self.core_boundary_level > 0.0
            && self.core_boundary_level < 1.0
            && self.seed_level.is_none_or(|v| v.is_finite())
            && self.graph_margin.is_finite()
            && self.graph_margin >= 0.0
            && self.boundary_prior > 0.0
            && self.boundary_prior < 1.0
            && self.face_quantile >= 0.0
            && self.face_quantile <= 1.0)
        {
            return Err(Error::invalid("invalid interior partition parameters"));
        }
        ScaleSpace::isotropic(&self.scales, 4.0, 1.0)?;
        Ok(())
    }
    fn window_params(&self) -> InteriorParams {
        InteriorParams {
            inner: self.inner,
            seed_smooth: self.scales.iter().copied().fold(1.0, f64::max),
            ..InteriorParams::default()
        }
    }
    pub fn halo(&self, extent: f64) -> usize {
        self.window_params().halo(extent) + 2
    }
    pub fn window(&self, shape: &StarShape, volume: [usize; 3]) -> ([usize; 3], [usize; 3]) {
        let (start, size) = interior_window(shape, &self.window_params(), volume);
        let lo = start.map(|v| v.saturating_sub(2));
        let hi: [usize; 3] = std::array::from_fn(|a| (start[a] + size[a] + 2).min(volume[a]));
        (lo, std::array::from_fn(|a| hi[a] - lo[a]))
    }
}
pub const SUMMARY_COLUMNS: [&str; 7] = [
    "partition_voxels",
    "partition_territories",
    "partition_fragments",
    "partition_largest_share",
    "partition_boundary_mean",
    "partition_cut_mean",
    "partition_inner",
];
pub const TERRITORY_COLUMNS: [&str; 12] = [
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "voxels",
    "diameter",
    "level",
    "s",
    "aspect",
    "neighbours",
    "boundary_mean",
    "contact_faces",
    "surface_faces",
];
#[derive(Debug, Clone, PartialEq)]
pub struct Territory {
    /// Smallest global voxel of this connected territory; deterministic identity.
    pub anchor: [usize; 3],
    pub values: [f64; TERRITORY_COLUMNS.len()],
}
#[derive(Debug, Clone, PartialEq)]
pub struct Contact {
    pub a: u64,
    pub b: u64,
    pub faces: usize,
    pub evidence_mean: f64,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Partition {
    pub summary: [f64; SUMMARY_COLUMNS.len()],
    pub territories: Vec<Territory>,
    pub contacts: Vec<Contact>,
    pub runs: Vec<LabelRun>,
    /// Quantised evidence runs: label / 255 is the evidence; zero omitted.
    pub evidence: Vec<LabelRun>,
}
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    (v[(v.len() - 1) / 2] + v[v.len() / 2]) * 0.5
}
fn neighbours(p: [usize; 3], n: [usize; 3]) -> impl Iterator<Item = [usize; 3]> {
    (0..6).filter_map(move |k| {
        let a = k / 2;
        let mut q = p;
        if k % 2 == 0 {
            if q[a] == 0 {
                return None;
            }
            q[a] -= 1;
        } else {
            q[a] += 1;
            if q[a] >= n[a] {
                return None;
            }
        }
        Some(q)
    })
}

/// Segment one object's full geometric interior. Rivals must include all shells
/// whose painted ownership regions intersect the window. No global image is loaded.
pub fn segment_partition(
    window: ArrayView3<'_, f64>,
    start: [usize; 3],
    volume: [usize; 3],
    shape: &StarShape,
    rivals: &[&StarShape],
    rays: &RaySet,
    params: &PartitionParams,
) -> Result<Partition> {
    segment_partition_owned(window, start, volume, shape, rivals, rays, params, None)
}

#[allow(clippy::too_many_arguments)]
fn segment_partition_owned(
    window: ArrayView3<'_, f64>,
    start: [usize; 3],
    volume: [usize; 3],
    shape: &StarShape,
    rivals: &[&StarShape],
    rays: &RaySet,
    params: &PartitionParams,
    ownership: Option<ArrayView3<'_, u64>>,
) -> Result<Partition> {
    params.check()?;
    if ownership
        .as_ref()
        .is_some_and(|o| o.shape() != window.shape())
    {
        return Err(Error::invalid("parent ownership shape mismatch"));
    }
    if shape.radii.len() != rays.len()
        || shape.radii.iter().any(|r| !r.is_finite() || *r <= 0.0)
        || shape.center.iter().any(|x| !x.is_finite())
        || window.iter().any(|x| !x.is_finite())
    {
        return Err(Error::invalid(
            "partition needs finite pixels and positive fitted radii",
        ));
    }
    for rival in rivals {
        if rival.radii.len() != rays.len() {
            return Err(Error::invalid("partition rival ray count mismatch"));
        }
    }
    let n = [window.shape()[0], window.shape()[1], window.shape()[2]];
    if (0..3).any(|a| start[a].saturating_add(n[a]) > volume[a]) {
        return Err(Error::invalid("partition window lies outside the volume"));
    }
    let dim = window.raw_dim();
    let depth = |o: &StarShape, p: [usize; 3]| -> f32 {
        let d: [f64; 3] = std::array::from_fn(|a| (start[a] + p[a]) as f64 - o.center[a]);
        let length = d.iter().map(|x| x * x).sum::<f64>().sqrt() as f32;
        length / rays.interpolate(&o.radii, d).max(1e-6)
    };
    let parent = shape_label(shape.at, volume);
    let radial = Array3::from_shape_fn(dim, |(z, y, x)| depth(shape, [z, y, x]));
    let mut inside = Array3::from_shape_fn(dim, |(z, y, x)| {
        let own = radial[[z, y, x]];
        (own as f64) < params.inner
            && ownership.as_ref().map_or_else(
                || {
                    rivals.iter().filter(|o| o.at != shape.at).all(|o| {
                        let theirs = depth(o, [z, y, x]);
                        !(theirs < own || (theirs == own && shape_label(o.at, volume) < parent))
                    })
                },
                |o| o[[z, y, x]] == parent,
            )
    });
    let mut count = inside.iter().filter(|&&b| b).count();
    if count == 0 {
        return Ok(Partition {
            summary: [0.0; SUMMARY_COLUMNS.len()],
            territories: vec![],
            contacts: vec![],
            runs: vec![],
            evidence: vec![],
        });
    }
    let gaussian = Gaussian::isotropic(1.0, 4.0)?;
    let mut fine = Array3::<f64>::zeros(dim);
    gaussian_smooth_into_with(
        window,
        gaussian.kernels(),
        gaussian.boundary(),
        fine.view_mut(),
    )?;
    let mut bins = vec![Vec::new(); 24];
    for (i, &r) in radial.indexed_iter() {
        if let Some(b) = bins.get_mut((r as f64 / 0.05) as usize) {
            b.push(fine[i]);
        }
    }
    let profile: Vec<Option<f64>> = bins
        .iter_mut()
        .map(|v| (!v.is_empty()).then(|| median(v)))
        .collect();
    let wall = profile[16..24]
        .iter()
        .flatten()
        .copied()
        .reduce(f64::max)
        .unwrap_or(0.0);
    let dark = profile[10..20]
        .iter()
        .flatten()
        .copied()
        .reduce(f64::min)
        .unwrap_or(wall);
    let interior_limit = if params.adaptive_inner {
        radial_interior_edge(&profile, params.inner).unwrap_or(params.inner)
    } else {
        params.inner
    };
    if params.adaptive_inner {
        for (p, yes) in inside.indexed_iter_mut() {
            *yes &= (radial[p] as f64) < interior_limit;
        }
        count = inside.iter().filter(|&&b| b).count();
        if count == 0 {
            let mut summary = [0.0; SUMMARY_COLUMNS.len()];
            summary[6] = interior_limit;
            return Ok(Partition {
                summary,
                territories: vec![],
                contacts: vec![],
                runs: vec![],
                evidence: vec![],
            });
        }
    }
    // No intensity floor in the normaliser: invariance under positive affine changes.
    let lo = window.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // An almost-flat radial profile cannot calibrate a useful wall contrast.
    // Revised seeding falls back to the window range rather than amplifying
    // interpolation noise. Keep the v1 normalisation exactly when both new
    // controls are disabled, so existing runs remain reproducible.
    let revised = params.adaptive_inner || params.graph_margin > 0.0 || params.seed_level.is_some();
    let (dark, scale) = if revised {
        // Use the typical smoothed range inside this fitted parent, not extrema
        // elsewhere in its read window. One distant hot pixel must not switch
        // a well-calibrated cyst into the low-contrast fallback.
        let (robust_lo, robust_hi) = parent_intensity_range(&bins).unwrap_or((lo, hi));
        if wall - dark > (robust_hi - robust_lo) * 0.05 {
            (dark, wall - dark)
        } else {
            (robust_lo, (robust_hi - robust_lo).max(1e-9))
        }
    } else {
        (
            dark,
            if wall - dark > (hi - lo) * 1e-6 {
                wall - dark
            } else {
                (hi - lo).max(1e-9)
            },
        )
    };
    let normalised = window.mapv(|v| (v - dark) / scale);
    let mut boundary = Array3::<f64>::zeros(dim);
    ridge_response_into(
        normalised.view(),
        &ScaleSpace::isotropic(&params.scales, 4.0, 1.0)?,
        &PlateResponse::new(0.5, 0.5, params.strength, Polarity::Valley)?,
        boundary.view_mut(),
        None,
    )?;

    let solve_inside = graph_core(&inside, params.graph_margin)?;

    // Fine, deterministic *core* fragments. Seeding dark sheets creates thin
    // sheet territories and lets true interiors join around them. Place seeds
    // away from sheets and the mask edge, without removing a foreground voxel.
    let free = Array3::from_shape_fn(dim, |p| {
        inside[p] && boundary[p] < params.core_boundary_level
    });
    let clearance = distance_transform(free.view(), &DistanceParams::default())?;
    let eligible = |p| {
        params
            .seed_level
            .is_none_or(|level| (fine[p] - dark) / scale >= level)
    };
    let step = params.seed_spacing;
    let mut best: BTreeMap<[usize; 3], ([usize; 3], f64)> = BTreeMap::new();
    for ((z, y, x), &yes) in inside.indexed_iter() {
        let p = [z, y, x];
        if !yes || !solve_inside[p] || clearance[p] < params.core_distance || !eligible(p) {
            continue;
        }
        let key = p.map(|v| v / step);
        let score = -clearance[p];
        let slot = best.entry(key).or_insert((p, score));
        if score < slot.1 {
            *slot = (p, score);
        }
    }
    let mut seeds = Array3::<u32>::zeros(dim);
    let mut next = 0u32;
    for &(p, _) in best.values() {
        next += 1;
        seeds[p] = next;
    }
    let mut visited = Array3::from_elem(dim, false);
    for ((z, y, x), &yes) in inside.indexed_iter() {
        let p = [z, y, x];
        if !yes || visited[p] {
            continue;
        }
        let mut q = VecDeque::from([p]);
        visited[p] = true;
        let mut seeded = false;
        let mut fallback = None;
        while let Some(p) = q.pop_front() {
            if solve_inside[p] {
                let better = fallback.is_none_or(|old| {
                    if params.seed_level.is_some() {
                        fine[p] > fine[old]
                    } else {
                        clearance[p] > clearance[old]
                    }
                });
                if better {
                    fallback = Some(p);
                }
            }
            seeded |= seeds[p] != 0;
            for nb in neighbours(p, n) {
                if inside[nb] && !visited[nb] {
                    visited[nb] = true;
                    q.push_back(nb);
                }
            }
        }
        if !seeded {
            next += 1;
            seeds[fallback.expect("each component has a graph core")] = next;
        }
    }
    let mut fragments = Array3::<u32>::zeros(dim);
    seeded_watershed_into(
        boundary.view(),
        seeds.view(),
        Some(inside.view()),
        Separation::Adjacent,
        fragments.view_mut(),
    )?;
    if inside
        .indexed_iter()
        .any(|(p, &yes)| yes && fragments[p] == 0)
    {
        return Err(Error::invalid("watershed left unassigned interior voxels"));
    }
    // Fragment the full mask, but score contacts only in the core. Peripheral
    // faces must not act as attractive shortcuts around real internal seams.
    let graph_fragments =
        Array3::from_shape_fn(dim, |p| if solve_inside[p] { fragments[p] } else { 0 });
    let graph = rag(
        graph_fragments.view(),
        boundary.view(),
        None,
        &RagOptions::new([1.0; 3], 64, (0.0, 1.0)),
    )?;
    let g = graph.graph(|r, i| {
        let (_, e) = r.edge(i);
        // A robust face statistic, rather than one unusually dark edge voxel.
        let p = 0.5 * (e.mean() + r.quantile(i, params.face_quantile));
        e.area * boundary_cost(p, params.boundary_prior)
    });
    let clusters = multicut(g.labels.len(), &g.edges, KlOptions::new(10, 1e-9, 30));
    let lookup: BTreeMap<u32, u32> = g.labels.iter().copied().zip(clusters).collect();
    let mut merged = graph_fragments.mapv(|f| if f == 0 { 0 } else { lookup[&f] });
    if params.graph_margin > 0.0 {
        // All solved core voxels are fixed markers. The margin cannot add graph
        // nodes or provide shortcuts around an internal separating sheet.
        let mut extended = Array3::<u32>::zeros(dim);
        seeded_watershed_into(
            boundary.view(),
            merged.view(),
            Some(inside.view()),
            Separation::Adjacent,
            extended.view_mut(),
        )?;
        merged = extended;
    }
    // Make each identity face-connected even if a graph heuristic returns the
    // same cluster ID for two disconnected pieces.
    let mut labels = Array3::<u32>::zeros(dim);
    let mut k = 0u32;
    for ((z, y, x), &c) in merged.indexed_iter() {
        let p = [z, y, x];
        if c == 0 || labels[p] != 0 {
            continue;
        }
        k += 1;
        let mut q = VecDeque::from([p]);
        labels[p] = k;
        while let Some(p) = q.pop_front() {
            for nb in neighbours(p, n) {
                if merged[nb] == c && labels[nb] == 0 {
                    labels[nb] = k;
                    q.push_back(nb);
                }
            }
        }
    }
    // Agglomerate tiny parts into a face neighbour. Never create holes, never
    // join disconnected parts, and never enforce a target number or equal size.
    if params.min_voxels > 0 {
        loop {
            let mut sizes = BTreeMap::<u32, usize>::new();
            for &l in &labels {
                if l != 0 {
                    *sizes.entry(l).or_default() += 1;
                }
            }
            let mut faces = BTreeMap::<(u32, u32), (f64, usize)>::new();
            for ((z, y, x), &l) in labels.indexed_iter() {
                if l == 0 {
                    continue;
                }
                let p = [z, y, x];
                for nb in neighbours(p, n) {
                    let b = labels[nb];
                    if b != 0 && b != l {
                        let e = faces.entry((l, b)).or_default();
                        e.0 += (boundary[p] + boundary[nb]) * 0.5;
                        e.1 += 1;
                    }
                }
            }
            let candidate = sizes
                .iter()
                .filter(|(_, v)| **v < params.min_voxels)
                .filter_map(|(&a, _)| {
                    faces
                        .iter()
                        .filter(|((x, _), _)| *x == a)
                        .min_by(|(ka, va), (kb, vb)| {
                            (va.0 / va.1 as f64)
                                .total_cmp(&(vb.0 / vb.1 as f64))
                                .then(ka.cmp(kb))
                        })
                        .map(|(&(a, b), _)| (a, b))
                })
                .next();
            let Some((a, b)) = candidate else {
                break;
            };
            labels.mapv_inplace(|l| if l == a { b } else { l });
        }
    }
    // Use the first voxel as the anchor, independent of graph and tiling IDs.
    let mut anchors = BTreeMap::<u32, [usize; 3]>::new();
    for ((z, y, x), &l) in labels.indexed_iter() {
        if l != 0 {
            anchors
                .entry(l)
                .or_insert([start[0] + z, start[1] + y, start[2] + x]);
        }
    }
    let mut accum = BTreeMap::<u32, Vec<[usize; 3]>>::new();
    let mut faces = BTreeMap::<(u32, u32), (usize, f64)>::new();
    let mut surfaces = BTreeMap::<u32, usize>::new();
    let mut boundary_sum = 0.0;
    for ((z, y, x), &l) in labels.indexed_iter() {
        if l == 0 {
            continue;
        }
        let p = [z, y, x];
        accum.entry(l).or_default().push(p);
        boundary_sum += boundary[p];
        let mut count_neighbours = 0;
        for nb in neighbours(p, n) {
            count_neighbours += 1;
            let b = labels[nb];
            if b != l {
                *surfaces.entry(l).or_default() += 1;
            }
            if b != 0 && l < b {
                let e = faces.entry((l, b)).or_default();
                e.0 += 1;
                e.1 += (boundary[p] + boundary[nb]) * 0.5;
            }
        }
        *surfaces.entry(l).or_default() += 6 - count_neighbours;
    }
    let mut territories = Vec::new();
    for (&l, points) in &accum {
        let m = points.len() as f64;
        let mean: [f64; 3] =
            std::array::from_fn(|a| points.iter().map(|p| p[a] as f64).sum::<f64>() / m);
        let pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
        let cov = std::array::from_fn(|a| {
            points
                .iter()
                .map(|p| {
                    (p[pairs[a].0] as f64 - mean[pairs[a].0])
                        * (p[pairs[a].1] as f64 - mean[pairs[a].1])
                })
                .sum::<f64>()
                / m
        });
        let [high, _, low] = symmetric_eigenvalues(cov);
        let centroid: [f64; 3] = std::array::from_fn(|a| mean[a] + start[a] as f64);
        let d: [f64; 3] = std::array::from_fn(|a| centroid[a] - shape.center[a]);
        let radius = d.iter().map(|v| v * v).sum::<f64>().sqrt()
            / (rays.interpolate(&shape.radii, d) as f64).max(1e-6);
        let adjacent: Vec<_> = faces
            .iter()
            .filter(|((a, b), _)| *a == l || *b == l)
            .collect();
        territories.push(Territory {
            anchor: anchors[&l],
            values: [
                centroid[0],
                centroid[1],
                centroid[2],
                m,
                (6.0 * m / std::f64::consts::PI).cbrt(),
                points.iter().map(|p| normalised[*p]).sum::<f64>() / m,
                radius,
                if high > 0.0 {
                    (low.max(0.0) / high).sqrt()
                } else {
                    1.0
                },
                adjacent.len() as f64,
                points.iter().map(|p| boundary[*p]).sum::<f64>() / m,
                adjacent.iter().map(|(_, e)| e.0).sum::<usize>() as f64,
                surfaces[&l] as f64,
            ],
        });
    }
    territories.sort_by_key(|t| t.anchor);
    let mut contacts = Vec::new();
    let mut cut_sum = 0.0;
    let mut cut_faces = 0usize;
    for (&(a, b), &(n, total)) in &faces {
        let (a, b) = (
            shape_label(anchors[&a], volume),
            shape_label(anchors[&b], volume),
        );
        contacts.push(Contact {
            a: a.min(b),
            b: a.max(b),
            faces: n,
            evidence_mean: total / n as f64,
        });
        cut_sum += total;
        cut_faces += n;
    }
    contacts.sort_by_key(|c| (c.a, c.b));
    let mut runs = Vec::new();
    let mut evidence = Vec::new();
    for z in 0..n[0] {
        for y in 0..n[1] {
            let mut x = 0;
            while x < n[2] {
                let l = labels[[z, y, x]];
                if l == 0 {
                    x += 1;
                    continue;
                }
                let from = x;
                while x < n[2] && labels[[z, y, x]] == l {
                    x += 1;
                }
                runs.push(LabelRun {
                    label: shape_label(anchors[&l], volume),
                    at: [start[0] + z, start[1] + y, start[2] + from],
                    len: x - from,
                });
            }
            let quant = |x: usize| {
                if inside[[z, y, x]] {
                    (boundary[[z, y, x]].clamp(0.0, 1.0) * 255.0).round() as u64
                } else {
                    0
                }
            };
            let mut x = 0;
            while x < n[2] {
                let q = quant(x);
                if q == 0 {
                    x += 1;
                    continue;
                }
                let from = x;
                while x < n[2] && quant(x) == q {
                    x += 1;
                }
                evidence.push(LabelRun {
                    label: q,
                    at: [start[0] + z, start[1] + y, start[2] + from],
                    len: x - from,
                });
            }
        }
    }
    merged.fill(0); // release scratch before returning the encoded products
    Ok(Partition {
        summary: [
            count as f64,
            territories.len() as f64,
            next as f64,
            accum.values().map(Vec::len).max().unwrap_or(0) as f64 / count as f64,
            boundary_sum / count as f64,
            if cut_faces > 0 {
                cut_sum / cut_faces as f64
            } else {
                0.0
            },
            interior_limit,
        ],
        territories,
        contacts,
        runs,
        evidence,
    })
}

pub fn summary_schema() -> Result<Schema> {
    Schema::new(SUMMARY_COLUMNS.iter().map(|c| Column::f64(*c)).collect())
}
pub fn territory_schema() -> Result<Schema> {
    let mut columns = vec![Column::u64("parent")];
    columns.extend(TERRITORY_COLUMNS.iter().map(|c| Column::f64(*c)));
    Schema::new(columns)
}
pub fn contact_schema() -> Result<Schema> {
    Schema::new(vec![
        Column::u64("parent"),
        Column::u64("a"),
        Column::u64("b"),
        Column::f64("faces"),
        Column::f64("evidence_mean"),
    ])
}
pub struct PartitionOp {
    ownership: Option<ImageId>,
    name: &'static str,
    image: ImageId,
    rows: (String, usize),
    rows_schema: Schema,
    shapes: (String, usize),
    nearby: (String, usize),
    rays: Arc<RaySet>,
    params: PartitionParams,
    max_extent: f64,
    summary: (String, Lifecycle),
    territories: (String, Lifecycle),
    runs: (String, Lifecycle),
    evidence: (String, Lifecycle),
    contacts: (String, Lifecycle),
    summary_schema: Schema,
    territory_schema: Schema,
}

/// The input and output streams of [`PartitionOp`] beside its rows and shapes.
pub struct PartitionStreams {
    /// Other blocks' objects near this block's, from [`NearbyShapesOp`].
    pub nearby: (String, usize),
    pub summary: (String, Lifecycle),
    pub territories: (String, Lifecycle),
    pub runs: (String, Lifecycle),
    pub evidence: (String, Lifecycle),
    pub contacts: (String, Lifecycle),
}

impl PartitionOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &'static str,
        image: ImageId,
        rows: (impl Into<String>, usize),
        rows_schema: Schema,
        shapes: (impl Into<String>, usize),
        rays: Arc<RaySet>,
        params: PartitionParams,
        max_extent: f64,
        streams: PartitionStreams,
    ) -> Result<Self> {
        params.check()?;
        if !(max_extent > 0.0 && max_extent.is_finite()) {
            return Err(Error::invalid(format!(
                "a partition op needs a positive extent bound, got {max_extent}"
            )));
        }
        Ok(Self {
            ownership: None,
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
            territories: streams.territories,
            runs: streams.runs,
            evidence: streams.evidence,
            contacts: streams.contacts,
            summary_schema: summary_schema()?,
            territory_schema: territory_schema()?,
        })
    }

    /// Use the published U64 parent labels for exact voxel ownership, rather
    /// than reconstructing overlaps from rounded saved fits.
    pub fn with_parent_labels(mut self, image: ImageId) -> Self {
        self.ownership = Some(image);
        self
    }

    pub fn summary_schema(&self) -> &Schema {
        &self.summary_schema
    }

    pub fn territory_schema(&self) -> &Schema {
        &self.territory_schema
    }
}

impl FragmentOp for PartitionOp {
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
        let mut images = vec![SourceInput::new(
            self.image,
            Reach::symmetric([self.params.halo(self.max_extent); 3]),
        )];
        if let Some(image) = self.ownership {
            images.push(
                SourceInput::new(
                    image,
                    Reach::symmetric([self.params.halo(self.max_extent); 3]),
                )
                .holding(Dtype::U64),
            );
        }
        images
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
            FragmentOutput::new(
                &self.territories.0,
                self.territories.1,
                Coverage::EveryBlock,
            ),
            FragmentOutput::new(&self.runs.0, self.runs.1, Coverage::EveryBlock),
            FragmentOutput::new(&self.evidence.0, self.evidence.1, Coverage::EveryBlock),
            FragmentOutput::new(&self.contacts.0, self.contacts.1, Coverage::EveryBlock),
        ]
    }

    fn prepares(&self) -> bool {
        true
    }

    fn prepare(
        &self,
        at: &crate::fragment::PreparationView<'_>,
    ) -> Result<crate::fragment::BlockPreparation> {
        use crate::fragment::{coalesce_read_windows, BlockPreparation, ReadWindow};
        match at.own_rows_empty(&self.rows.0, &self.rows_schema)? {
            None => return Ok(BlockPreparation::Run),
            Some(true) => {
                // Preserve metadata validation even when no image work is needed.
                if let Some(bytes) = at.own(&self.shapes.0) {
                    if decode_shapes(bytes)?.0 != self.rays.len() {
                        return Err(Error::invalid(
                            "shape ray count differs from partition rays",
                        ));
                    }
                }
                if let Some(bytes) = at.own(&self.nearby.0) {
                    decode_shapes(bytes)?;
                }
                return Ok(BlockPreparation::Complete {
                    fragments: vec![
                        (
                            self.summary.0.clone(),
                            RowBuilder::new(Arc::new(self.summary_schema.clone())).encode(),
                        ),
                        (
                            self.territories.0.clone(),
                            RowBuilder::new(Arc::new(self.territory_schema.clone())).encode(),
                        ),
                        (
                            self.contacts.0.clone(),
                            RowBuilder::new(Arc::new(contact_schema()?)).encode(),
                        ),
                        (self.runs.0.clone(), encode_runs(&[])),
                        (self.evidence.0.clone(), encode_runs(&[])),
                    ],
                    constant: None,
                });
            }
            Some(false) => (),
        }
        let Some(bytes) = at.own(&self.shapes.0) else {
            return Ok(BlockPreparation::Run);
        };
        let (_, shapes) = decode_shapes(bytes)?;
        let rows = sealed_rows(
            at.volume(),
            &self.rows_schema,
            at.own(&self.rows.0).unwrap(),
        )?;
        let mut windows = Vec::new();
        for row in rows.iter() {
            let shape = shapes
                .iter()
                .find(|s| s.at == row.at)
                .ok_or_else(|| Error::invalid("missing owned shape during source preparation"))?;
            let (start, extent) = self.params.window(shape, at.volume());
            for image in std::iter::once(self.image).chain(self.ownership) {
                windows.push(ReadWindow {
                    image: image.index(),
                    region: crate::Region::new(&start, &extent),
                });
            }
        }
        Ok(BlockPreparation::ReadWindows(coalesce_read_windows(
            windows,
        )))
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "the partition op reads its image through apply_with",
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
                    "shapes carry {rays} rays, the partition op was built for {}",
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
        let mut territory_rows = RowBuilder::new(Arc::new(self.territory_schema.clone()));
        let mut runs = Vec::new();
        let mut evidence = Vec::new();
        let mut contacts = RowBuilder::new(Arc::new(contact_schema()?));
        if !objects.is_empty() {
            let read_end: [usize; 3] = std::array::from_fn(|a| at.read.start[a] + at.read.shape[a]);
            let results: Vec<Partition> = objects
                .par_iter()
                .map(|shape| {
                    let (start, extent) = self.params.window(shape, volume);
                    for a in 0..3 {
                        if start[a] < at.read.start[a] || start[a] + extent[a] > read_end[a] {
                            return Err(Error::invalid(format!(
                                "the object at {:?} reaches beyond the interior halo (extent \
                                 bound {:.1}); raise the bound",
                                shape.at, self.max_extent
                            )));
                        }
                    }
                    let region = crate::Region::new(&start, &extent);
                    let (raw, source_region) =
                        sources.covering(self.image.index(), &region, at.read)?;
                    let read = [
                        source_region.start[0],
                        source_region.start[1],
                        source_region.start[2],
                    ];
                    let window = window_of(raw.as_array()?, start, extent, read)?;
                    let rivals: Vec<&StarShape> = shapes.values().chain(nearby.iter()).collect();
                    let owned = if let Some(image) = self.ownership {
                        let (p, parent_region) =
                            sources.covering(image.index(), &region, at.read)?;
                        let p = p.as_array()?.view::<u64>()?;
                        let o: [usize; 3] =
                            std::array::from_fn(|a| start[a] - parent_region.start[a]);
                        Some(
                            p.slice(s![
                                o[0]..o[0] + extent[0],
                                o[1]..o[1] + extent[1],
                                o[2]..o[2] + extent[2]
                            ])
                            .to_owned(),
                        )
                    } else {
                        None
                    };
                    segment_partition_owned(
                        window.view(),
                        start,
                        volume,
                        shape,
                        &rivals,
                        &self.rays,
                        &self.params,
                        owned.as_ref().map(|p| p.view()),
                    )
                })
                .collect::<Result<_>>()?;
            for (shape, interior) in objects.iter().zip(results) {
                summary.push(shape.at, &interior.summary.map(Value::F64))?;
                let parent = shape_label(shape.at, volume);
                for b in &interior.territories {
                    let mut values = vec![Value::U64(parent)];
                    values.extend(b.values.map(Value::F64));
                    territory_rows.push(b.anchor, &values)?;
                }
                for c in &interior.contacts {
                    contacts.push(
                        shape.at,
                        &[
                            Value::U64(parent),
                            Value::U64(c.a),
                            Value::U64(c.b),
                            Value::F64(c.faces as f64),
                            Value::F64(c.evidence_mean),
                        ],
                    )?;
                }
                evidence.extend(interior.evidence);
                runs.extend(interior.runs);
            }
        }
        Ok(BlockOutput::fragment(&self.summary.0, summary.encode())
            .with_fragment(&self.territories.0, territory_rows.encode())
            .with_fragment(&self.runs.0, encode_runs(&runs))
            .with_fragment(&self.evidence.0, encode_runs(&evidence))
            .with_fragment(&self.contacts.0, contacts.encode()))
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
        Error::invalid("the partition op cannot read half-precision"),
        |view| { Ok(widened(view.slice(cut))) }
    )
}

/// Paint quantised boundary evidence (F32, score/255) from per-object runs.
pub struct PartitionEvidenceOp {
    name: &'static str,
    runs: (String, usize),
    max_extent: f64,
}

impl PartitionEvidenceOp {
    pub fn new(name: &'static str, runs: (impl Into<String>, usize), max_extent: f64) -> Self {
        Self {
            name,
            runs: (runs.0.into(), runs.1),
            max_extent,
        }
    }
}

impl FragmentOp for PartitionEvidenceOp {
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
        Dtype::F32
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
        super::interior::prepare_run_paint(at, &self.runs.0, self.max_extent)
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
            let mut view = out.view_mut::<f32>()?;
            let origin = [at.read.start[0], at.read.start[1], at.read.start[2]];
            for (_key, bytes) in at.fragments(&self.runs.0) {
                for r in decode_runs(bytes)? {
                    let [z, y, x] = r.at;
                    if z < lo[0] || z >= hi[0] || y < lo[1] || y >= hi[1] {
                        continue;
                    }
                    for x in x.max(lo[2])..(x + r.len).min(hi[2]) {
                        let slot = &mut view[[z - origin[0], y - origin[1], x - origin[2]]];
                        *slot = slot.max(r.label as f32 / 255.0);
                    }
                }
            }
        }
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

/// Typical intensity range within radial depth 1.2. The bins already contain
/// smoothed values used for the radial profile. Select percentiles in linear
/// time; no full sort or influence from the rest of the processing window.
fn parent_intensity_range(bins: &[Vec<f64>]) -> Option<(f64, f64)> {
    let mut values: Vec<f64> = bins.iter().flatten().copied().collect();
    if values.is_empty() {
        return None;
    }
    let lower = ((values.len() - 1) as f64 * 0.005).floor() as usize;
    let upper = ((values.len() - 1) as f64 * 0.995).ceil() as usize;
    let low = *values.select_nth_unstable_by(lower, f64::total_cmp).1;
    let high = *values.select_nth_unstable_by(upper, f64::total_cmp).1;
    Some((low, high))
}

/// Find the inner half-depth crossing of a supported outer radial dark band.
/// The bright reference comes from just inside that band, avoiding dark centres.
fn radial_interior_edge(profile: &[Option<f64>], cap: f64) -> Option<f64> {
    let (b, dark) = (10..20)
        .filter_map(|b| profile.get(b).copied().flatten().map(|v| (b, v)))
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    let wall = profile
        .get(16..24)?
        .iter()
        .flatten()
        .copied()
        .reduce(f64::max)?;
    let mut reference: Vec<_> = profile[b.saturating_sub(6)..b.saturating_sub(2)]
        .iter()
        .flatten()
        .copied()
        .collect();
    if reference.len() < 2 || wall <= dark {
        return None;
    }
    let bright = median(&mut reference);
    if bright - dark <= 0.1 * (wall - dark) {
        return None;
    }
    let level = dark + 0.5 * (bright - dark);
    for i in (0..b).rev() {
        if let (Some(left), Some(right)) = (profile[i], profile[i + 1]) {
            if left >= level && right < level {
                let fraction = (left - level) / (left - right);
                let radius = (i as f64 + 0.5 + fraction) * 0.05;
                return (radius >= 0.35 && radius < cap).then_some(radius);
            }
        }
    }
    None
}

/// Erode only the graph domain, keeping at least one marker in every connected
/// foreground component. The original foreground remains the output domain.
fn graph_core(inside: &Array3<bool>, margin: f64) -> Result<Array3<bool>> {
    if margin == 0.0 {
        return Ok(inside.clone());
    }
    let distance = distance_transform(inside.view(), &DistanceParams::default())?;
    let n = [inside.shape()[0], inside.shape()[1], inside.shape()[2]];
    let mut core = Array3::from_shape_fn(inside.raw_dim(), |p| inside[p] && distance[p] > margin);
    let mut visited = Array3::from_elem(inside.raw_dim(), false);
    for ((z, y, x), &yes) in inside.indexed_iter() {
        let first = [z, y, x];
        if !yes || visited[first] {
            continue;
        }
        let mut queue = VecDeque::from([first]);
        visited[first] = true;
        let mut best = first;
        let mut has_core = false;
        while let Some(p) = queue.pop_front() {
            has_core |= core[p];
            if distance[p] > distance[best] {
                best = p;
            }
            for nb in neighbours(p, n) {
                if inside[nb] && !visited[nb] {
                    visited[nb] = true;
                    queue.push_back(nb);
                }
            }
        }
        if !has_core {
            core[best] = true;
        }
    }
    Ok(core)
}

#[cfg(test)]
mod tests {
    use super::super::interior::NearbyShapesOp;
    use super::*;
    use crate::env::ArrayEnvironment;
    use crate::ops::rows::{collect_rows, RowSourceOp, RowValues};
    use crate::strategy::{execute_phases, Hints};
    use crate::{BlockGrid, PlanBuilder};
    use std::collections::BTreeSet;
    fn shape(c: [f64; 3], r: f32, rays: &RaySet) -> StarShape {
        StarShape {
            at: c.map(|v| v as usize),
            center: c,
            radii: vec![r; rays.len()],
        }
    }
    fn params() -> PartitionParams {
        PartitionParams {
            seed_spacing: 5,
            core_distance: 3.0,
            core_boundary_level: 0.4,
            min_voxels: 8,
            graph_margin: 3.0,
            boundary_prior: 0.25,
            ..PartitionParams::default()
        }
    }
    fn labels(p: &Partition, n: [usize; 3]) -> Array3<u64> {
        let mut out = Array3::zeros(n);
        for r in &p.runs {
            for x in r.at[2]..r.at[2] + r.len {
                assert_eq!(out[[r.at[0], r.at[1], x]], 0);
                out[[r.at[0], r.at[1], x]] = r.label;
            }
        }
        out
    }
    #[test]
    fn flat_interior_is_one_filled_connected_territory() {
        let n = [32; 3];
        let rays = RaySet::fibonacci(100, 8);
        let s = shape([16.0; 3], 12.0, &rays);
        let raw = Array3::from_elem(n, 200.0);
        let got = segment_partition(raw.view(), [0; 3], n, &s, &[], &rays, &params()).unwrap();
        assert_eq!(got.territories.len(), 1);
        assert!(got.contacts.is_empty());
        let painted = labels(&got, n);
        assert_eq!(
            painted.iter().filter(|&&v| v > 0).count(),
            got.summary[0] as usize
        );
        assert_eq!(
            painted[[16, 16, 16]],
            shape_label(got.territories[0].anchor, n)
        );
        assert_eq!(painted[[0, 0, 0]], 0);
    }
    #[test]
    fn dark_sheet_separates_bright_interiors_without_leaving_dark_holes() {
        let n = [40; 3];
        let rays = RaySet::fibonacci(100, 8);
        let s = shape([20.0; 3], 16.0, &rays);
        let raw =
            Array3::from_shape_fn(n, |(_, _, x)| if x == 19 || x == 20 { 20.0 } else { 200.0 });
        let got = segment_partition(raw.view(), [0; 3], n, &s, &[], &rays, &params()).unwrap();
        let painted = labels(&got, n);
        assert_eq!(
            got.territories.len(),
            2,
            "{:?} territories={:?}",
            got.summary,
            got.territories
        );
        assert_ne!(
            painted[[20, 20, 13]],
            painted[[20, 20, 27]],
            "{:?}",
            got.summary
        );
        assert_ne!(painted[[20, 20, 19]], 0);
        assert_eq!(
            painted.iter().filter(|&&v| v > 0).count(),
            got.summary[0] as usize
        );
        assert!(!got.contacts.is_empty());
        let changed = raw.mapv(|v| v * 3.0 + 73.0);
        let same = segment_partition(changed.view(), [0; 3], n, &s, &[], &rays, &params()).unwrap();
        assert_eq!(got.runs, same.runs);
        assert_eq!(got.evidence, same.evidence);
    }
    #[test]
    fn overlap_has_exclusive_parent_ownership_and_no_duplicate_ids() {
        let n = [40; 3];
        let rays = RaySet::fibonacci(100, 8);
        let a = shape([20.0, 20.0, 15.0], 12.0, &rays);
        let b = shape([20.0, 20.0, 25.0], 12.0, &rays);
        let raw = Array3::from_elem(n, 200.0);
        let pa = segment_partition(raw.view(), [0; 3], n, &a, &[&b], &rays, &params()).unwrap();
        let pb = segment_partition(raw.view(), [0; 3], n, &b, &[&a], &rays, &params()).unwrap();
        let la = labels(&pa, n);
        let lb = labels(&pb, n);
        assert!(la.iter().zip(lb.iter()).all(|(&a, &b)| a == 0 || b == 0));
        let ids: BTreeSet<_> = pa
            .territories
            .iter()
            .chain(pb.territories.iter())
            .map(|t| shape_label(t.anchor, n))
            .collect();
        assert_eq!(ids.len(), pa.territories.len() + pb.territories.len());
    }
    struct Shapes(Vec<StarShape>, usize);
    impl FragmentOp for Shapes {
        fn name(&self) -> &'static str {
            "partition-test-shapes"
        }
        fn reach(&self, _: usize, _: usize) -> usize {
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
                Lifecycle::Persistent,
                Coverage::EveryBlock,
            )]
        }
        fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
            let mine: Vec<_> = self
                .0
                .iter()
                .filter(|s| {
                    (0..3).all(|a| {
                        s.at[a] >= at.core.start[a] && s.at[a] < at.core.start[a] + at.core.shape[a]
                    })
                })
                .cloned()
                .collect();
            Ok(BlockOutput::fragment(
                "shapes",
                super::super::star_convex::encode_shapes(self.1, &mine),
            ))
        }
    }
    #[test]
    fn planned_partition_and_evidence_are_grid_invariant() {
        let n = [48, 48, 96];
        let rays = Arc::new(RaySet::fibonacci(100, 8));
        let shapes = vec![
            shape([24.0, 24.0, 42.0], 12.0, &rays),
            shape([24.0, 24.0, 60.0], 12.0, &rays),
        ];
        let raw =
            Array3::from_shape_fn(n, |(_, y, _)| if y == 23 || y == 24 { 20.0 } else { 200.0 });
        let schema = Schema::new(vec![Column::f64("score")]).unwrap();
        let mut answers = Vec::new();
        for block in [[48, 48, 96], [48, 48, 48]] {
            let mut b = PlanBuilder::new(n, Dtype::F64, BlockGrid::new(n, block).unwrap());
            let rows = b
                .fragments(RowSourceOp::new(
                    "rows",
                    "rows",
                    Lifecycle::Persistent,
                    schema.clone(),
                    shapes
                        .iter()
                        .map(|s| RowValues::new(s.at, vec![Value::F64(1.0)]))
                        .collect(),
                ))
                .unwrap();
            let shaped = b.fragments(Shapes(shapes.clone(), rays.len())).unwrap();
            let near = b
                .fragments(NearbyShapesOp::new(
                    "near",
                    ("rows", rows.index()),
                    schema.clone(),
                    ("shapes", shaped.index()),
                    rays.len(),
                    ("near", Lifecycle::Persistent),
                    13.0,
                ))
                .unwrap();
            let part = b
                .fragments(
                    PartitionOp::new(
                        "partition",
                        ImageId::from(0),
                        ("rows", rows.index()),
                        schema.clone(),
                        ("shapes", shaped.index()),
                        rays.clone(),
                        params(),
                        13.0,
                        PartitionStreams {
                            nearby: ("near".into(), near.index()),
                            summary: ("summary".into(), Lifecycle::Persistent),
                            territories: ("territories".into(), Lifecycle::Persistent),
                            runs: ("runs".into(), Lifecycle::Persistent),
                            evidence: ("evidence".into(), Lifecycle::Persistent),
                            contacts: ("contacts".into(), Lifecycle::Persistent),
                        },
                    )
                    .unwrap(),
                )
                .unwrap();
            let painted = b
                .fragments(super::super::interior::RunLabelsOp::new(
                    "labels",
                    ("runs", part.index()),
                    18.0,
                ))
                .unwrap();
            let _evidence = b
                .fragments(PartitionEvidenceOp::new(
                    "evidence",
                    ("evidence", part.index()),
                    18.0,
                ))
                .unwrap();
            let assembly = b.finish().unwrap();
            let env = ArrayEnvironment::for_decomposition(
                Voxels::from(raw.clone()),
                &assembly.decomposition,
                block,
            )
            .unwrap();
            execute_phases(
                "partition-test",
                &assembly.workflow,
                &assembly.decomposition,
                &Hints {
                    keep_images: [painted.image().unwrap()].into_iter().collect(),
                    ..Hints::default()
                },
                &env,
                &[],
                &assembly.work(),
            )
            .unwrap();
            let rows = collect_rows(
                &env,
                "territories",
                part.index(),
                n,
                territory_schema().unwrap(),
            )
            .unwrap();
            let summaries =
                collect_rows(&env, "summary", part.index(), n, summary_schema().unwrap()).unwrap();
            let pixels = env.output().view::<f32>().unwrap().to_owned();
            let labels = env
                .image(painted.image().unwrap().index())
                .view::<u64>()
                .unwrap()
                .to_owned();
            answers.push((rows, summaries, labels, pixels));
        }
        assert_eq!(answers[0], answers[1]);
    }
    #[test]
    fn published_parent_labels_override_rounded_rival_geometry() {
        let n = [32; 3];
        let rays = RaySet::fibonacci(100, 8);
        let s = shape([16.0; 3], 12.0, &rays);
        let raw = Array3::from_elem(n, 200.0);
        let ownership = Array3::from_shape_fn(
            n,
            |(_, _, x)| if x > 16 { shape_label(s.at, n) } else { 999 },
        );
        let got = segment_partition_owned(
            raw.view(),
            [0; 3],
            n,
            &s,
            &[],
            &rays,
            &params(),
            Some(ownership.view()),
        )
        .unwrap();
        let labels = labels(&got, n);
        assert_eq!(labels[[16, 16, 16]], 0);
        assert_ne!(labels[[16, 16, 17]], 0);
        assert!(labels
            .indexed_iter()
            .all(|(p, &l)| l == 0 || ownership[p] == shape_label(s.at, n)));
    }
    #[test]
    fn dark_nuclear_sized_centre_is_filled_without_an_extra_territory() {
        let n = [32; 3];
        let rays = RaySet::fibonacci(100, 8);
        let s = shape([16.0; 3], 12.0, &rays);
        let raw = Array3::from_shape_fn(n, |(z, y, x)| {
            let d =
                (z as f64 - 16.0).powi(2) + (y as f64 - 16.0).powi(2) + (x as f64 - 16.0).powi(2);
            if d <= 4.0 {
                20.0
            } else {
                200.0
            }
        });
        let got = segment_partition(raw.view(), [0; 3], n, &s, &[], &rays, &params()).unwrap();
        assert_eq!(got.territories.len(), 1, "{:?}", got.summary);
        assert_ne!(labels(&got, n)[[16, 16, 16]], 0);
    }
    #[test]
    fn graph_margin_does_not_turn_a_dark_enclosing_band_into_territories() {
        let n = [40; 3];
        let rays = RaySet::fibonacci(100, 8);
        let s = shape([20.0; 3], 16.0, &rays);
        let raw = Array3::from_shape_fn(n, |(z, y, x)| {
            let r =
                ((z as f64 - 20.0).powi(2) + (y as f64 - 20.0).powi(2) + (x as f64 - 20.0).powi(2))
                    .sqrt();
            if (10.0..14.0).contains(&r) {
                20.0
            } else {
                200.0
            }
        });
        let p = PartitionParams {
            graph_margin: 5.0,
            ..params()
        };
        let got = segment_partition(raw.view(), [0; 3], n, &s, &[], &rays, &p).unwrap();
        assert_eq!(got.territories.len(), 1, "{:?}", got.summary);
        let painted = labels(&got, n);
        let expected = (0..40)
            .flat_map(|z| (0..40).flat_map(move |y| (0..40).map(move |x| (z, y, x))))
            .filter(|&(z, y, x)| {
                ((z as f64 - 20.0).powi(2) + (y as f64 - 20.0).powi(2) + (x as f64 - 20.0).powi(2))
                    .sqrt()
                    < 16.0 * p.inner
            })
            .count();
        assert_eq!(painted.iter().filter(|&&l| l != 0).count(), expected);
    }
    #[test]
    fn bright_seeds_do_not_give_a_large_dark_cavity_its_own_territory() {
        let n = [40; 3];
        let rays = RaySet::fibonacci(100, 8);
        let s = shape([20.0; 3], 16.0, &rays);
        let raw = Array3::from_shape_fn(n, |(z, y, x)| {
            let r =
                ((z as f64 - 20.0).powi(2) + (y as f64 - 20.0).powi(2) + (x as f64 - 20.0).powi(2))
                    .sqrt();
            if r < 5.0 || (11.0..14.0).contains(&r) {
                20.0
            } else {
                200.0
            }
        });
        let p = PartitionParams {
            graph_margin: 3.0,
            seed_level: Some(0.35),
            core_distance: 2.0,
            ..params()
        };
        let got = segment_partition(raw.view(), [0; 3], n, &s, &[], &rays, &p).unwrap();
        assert_eq!(got.territories.len(), 1, "{:?}", got.territories);
        assert_ne!(labels(&got, n)[[20, 20, 20]], 0);
    }
    #[test]
    fn unrelated_extreme_pixels_do_not_change_revised_partition_calibration() {
        let n = [64; 3];
        let rays = RaySet::fibonacci(100, 8);
        let s = shape([32.0; 3], 16.0, &rays);
        let raw = Array3::from_shape_fn(n, |(z, y, x)| {
            let r =
                ((z as f64 - 32.0).powi(2) + (y as f64 - 32.0).powi(2) + (x as f64 - 32.0).powi(2))
                    .sqrt();
            if (11.0..14.0).contains(&r) || x == 31 || x == 32 {
                20.0
            } else {
                200.0
            }
        });
        let p = PartitionParams {
            adaptive_inner: true,
            graph_margin: 1.5,
            seed_level: Some(0.35),
            core_distance: 2.0,
            ..params()
        };
        let got = segment_partition(raw.view(), [0; 3], n, &s, &[], &rays, &p).unwrap();
        let mut changed = raw.clone();
        changed[[1, 1, 1]] = 1e8;
        changed[[2, 1, 1]] = -1e8;
        let other = segment_partition(changed.view(), [0; 3], n, &s, &[], &rays, &p).unwrap();
        assert_eq!(got.runs, other.runs);
        assert_eq!(got.evidence, other.evidence);
        assert_eq!(got.territories, other.territories);
        assert_eq!(got.summary, other.summary);
    }
    #[test]
    fn parent_range_is_robust_to_sparse_outliers_and_affine_changes() {
        let mut values: Vec<_> = (0..10000).map(|i| 10.0 + (i % 100) as f64).collect();
        values[0] = -1e9;
        values[1] = 1e9;
        let bins = vec![values];
        let (lo, hi) = parent_intensity_range(&bins).unwrap();
        assert!(lo >= 10.0 && hi <= 109.0);
        let changed = vec![bins[0].iter().map(|x| 3.0 * x + 71.0).collect()];
        let (a, b) = parent_intensity_range(&changed).unwrap();
        assert_eq!((a, b), (3.0 * lo + 71.0, 3.0 * hi + 71.0));
        assert!(parent_intensity_range(&[]).is_none());
    }
    #[test]
    fn radial_edge_uses_the_inner_band_face_and_falls_back_without_a_dip() {
        let mut profile = vec![Some(200.0); 24];
        profile[12] = Some(150.0);
        profile[13] = Some(60.0);
        profile[14] = Some(20.0);
        profile[15] = Some(50.0);
        profile[16] = Some(150.0);
        let edge = radial_interior_edge(&profile, 0.85).unwrap();
        assert!(edge > 0.625 && edge < 0.675, "{edge}");
        let changed: Vec<_> = profile.iter().map(|v| v.map(|x| 3.0 * x + 71.0)).collect();
        assert!((radial_interior_edge(&changed, 0.85).unwrap() - edge).abs() < 1e-12);
        assert!(radial_interior_edge(&vec![Some(200.0); 24], 0.85).is_none());
        assert!(radial_interior_edge(&vec![None; 24], 0.85).is_none());
    }
    #[test]
    fn adaptive_mask_excludes_the_enclosing_band_and_preserves_parent_ownership() {
        let n = [48; 3];
        let rays = RaySet::fibonacci(100, 8);
        let s = shape([24.0; 3], 20.0, &rays);
        let raw = Array3::from_shape_fn(n, |(z, y, x)| {
            let r =
                ((z as f64 - 24.0).powi(2) + (y as f64 - 24.0).powi(2) + (x as f64 - 24.0).powi(2))
                    .sqrt();
            if (12.5..16.5).contains(&r) {
                20.0
            } else {
                200.0
            }
        });
        let p = PartitionParams {
            adaptive_inner: true,
            graph_margin: 1.0,
            seed_level: Some(0.35),
            ..params()
        };
        let got = segment_partition(raw.view(), [0; 3], n, &s, &[], &rays, &p).unwrap();
        assert!(
            got.summary[6] < 0.7 && got.summary[6] > 0.5,
            "{:?}",
            got.summary
        );
        let image = labels(&got, n);
        assert_ne!(image[[24, 24, 24]], 0);
        assert_eq!(image[[24, 24, 40]], 0);
        for ((z, y, x), &label) in image.indexed_iter() {
            let r =
                ((z as f64 - 24.0).powi(2) + (y as f64 - 24.0).powi(2) + (x as f64 - 24.0).powi(2))
                    .sqrt();
            assert_eq!(label > 0, r < 20.0 * got.summary[6]);
        }
    }
    #[test]
    fn eroded_away_components_keep_markers_and_zero_margin_is_identity() {
        let mut mask = Array3::from_elem([12; 3], false);
        mask[[1, 1, 1]] = true;
        for z in 5..9 {
            for y in 5..9 {
                for x in 5..9 {
                    mask[[z, y, x]] = true;
                }
            }
        }
        assert_eq!(graph_core(&mask, 0.0).unwrap(), mask);
        let core = graph_core(&mask, 100.0).unwrap();
        assert!(core[[1, 1, 1]]);
        assert_eq!(core.iter().filter(|&&b| b).count(), 2);
        assert!(core.indexed_iter().all(|(p, &b)| !b || mask[p]));
    }
    #[test]
    fn rejects_invalid_parameters() {
        for p in [
            PartitionParams {
                seed_level: Some(f64::NAN),
                ..params()
            },
            PartitionParams {
                graph_margin: f64::NAN,
                ..params()
            },
            PartitionParams {
                graph_margin: -1.0,
                ..params()
            },
            PartitionParams {
                inner: 1.0,
                ..params()
            },
            PartitionParams {
                boundary_prior: 0.0,
                ..params()
            },
            PartitionParams {
                scales: vec![],
                ..params()
            },
            PartitionParams {
                strength: f64::NAN,
                ..params()
            },
        ] {
            assert!(p.check().is_err());
        }
    }
}
