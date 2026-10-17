// SPDX-License-Identifier: MIT
//
// Original work for this crate, written from Li, Wu, Chen & Sonka, "Optimal
// surface segmentation in volumetric images — a graph-theoretic approach",
// IEEE PAMI 28(1), 2006.
//
// **Layered height-field surfaces, found as one exact minimum cut.**
//
// A *surface* here is a height field `z = h(y, x)`: one crossing per column.
// `K` surfaces are stacked top to bottom and found jointly, as the global
// optimum of
//
//     sum over surfaces k, columns c of  cost_k(c, h_k(c))
//
// under hard shape limits:
//
// * **step:** `|h_k(p) - h_k(q)| <= max_step_k` for 4-neighbouring columns;
// * **separation:** `min <= h_{k+1} - h_k <= max` for consecutive surfaces.
//
// Compact bright clutter cannot pull a surface towards it without breaking the
// step limit over many columns, which is the shape prior doing its work. The
// construction is the standard one: one node per (surface, z, column), vertical
// arcs that make each column's source set a prefix `0..=h`, and infinite arcs
// for the limits, solved with [`crate::graph_cut::CutGraph`].
//
// Under the planner
// -----------------
// The cut is global, but it does not need the volume: it needs each surface's
// cost **pooled into columns** of `stride x stride` voxels, which is
// `stride²` times smaller. So the work is three phases
// ([`append_layered_surface_phases`]):
//
// 1. [`SurfaceScaleSamplesOp`] — every block emits the magnitudes of the cue
//    voxels whose global linear index is a multiple of 7, for the cues that are
//    normalised to unit robust magnitude.
// 2. [`SurfacePoolOp`] (barrier) — `reduce` turns the samples into one scale
//    per cue (the 99.5th percentile of the magnitudes); every block then forms
//    each surface's cost voxel by voxel from its cues and emits its columns'
//    means.
// 3. [`SurfaceSolveOp`] (barrier) — `reduce` assembles the pooled costs,
//    scales each to unit robust magnitude, builds the graph and solves it; every
//    block then paints region labels from the bilinearly upsampled heights:
//    the label is the number of surfaces at or above the voxel, so `0` above
//    the first surface and `K` at and below the last.
//
// Blocks must be multiples of `stride` in y and x so that no column straddles a
// seam; with that, the result is the same, bit for bit, for every block grid,
// and the same as [`solve_layered_surfaces`] on the whole arrays.
//
// Memory: phase 1 keeps at most `max_scale_samples` magnitudes per normalised
// cue (4 bytes each) for the reduce, and phase 3 holds the pooled graph —
// `K x nz x (ny/stride) x (nx/stride)` nodes, about 200 bytes each — in memory.
// The stride is therefore the knob that bounds memory. Refining at a finer
// stride without the whole graph is future work.

use ndarray::{Array2, Array3, ArrayView3, ArrayViewMut3, Axis};

use crate::assemble::{ImageId, Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, PhaseView,
    SeamFold, SourceBlocks,
};
use crate::graph_cut::{CutGraph, INFINITE_CAPACITY};
use crate::op::SourceInput;
use crate::sidecar::Lifecycle;

/// One image contributing to a surface's cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceCue {
    pub image: ImageId,
    pub weight: f64,
    /// Divide the cue by its robust magnitude (99.5th percentile of `|value|`)
    /// before weighting, so cues on different scales are comparable.
    pub normalise: bool,
    /// Element type of a *supplied* image (one no phase writes), which the
    /// plan cannot infer; `None` for images the plan produces.
    pub holding: Option<Dtype>,
}

impl SurfaceCue {
    pub fn raw(image: impl Into<ImageId>, weight: f64) -> Self {
        Self {
            image: image.into(),
            weight,
            normalise: false,
            holding: None,
        }
    }

    pub fn normalised(image: impl Into<ImageId>, weight: f64) -> Self {
        Self {
            image: image.into(),
            weight,
            normalise: true,
            holding: None,
        }
    }

    /// Declare the element type of a supplied image.
    pub fn holding(mut self, dtype: Dtype) -> Self {
        self.holding = Some(dtype);
        self
    }
}

/// One surface: its cost (a weighted sum of cues; low where the surface
/// should be), its weight against the other surfaces, and its step limit.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceLayer {
    pub cues: Vec<SurfaceCue>,
    /// Applied after the pooled cost is scaled to unit robust magnitude.
    pub weight: f64,
    /// Largest z step, in slices, between 4-neighbouring columns.
    pub max_step: usize,
}

/// The shape limits of a stack of surfaces, without the images.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceShape {
    /// xy pooling between voxels and graph columns.
    pub stride: usize,
    /// Per surface: weight after unit scaling.
    pub weights: Vec<f64>,
    /// Per surface: largest step between neighbouring columns.
    pub max_steps: Vec<usize>,
    /// Per consecutive pair `(k, k + 1)`: `[min, max]` of `h_{k+1} - h_k`.
    pub separations: Vec<[usize; 2]>,
}

impl SurfaceShape {
    fn check(&self) -> Result<()> {
        let k = self.weights.len();
        if k == 0 || self.max_steps.len() != k || self.separations.len() + 1 != k {
            return Err(Error::invalid(format!(
                "a stack of {k} surfaces needs {k} weights, {k} step limits and {} separations; \
                 got {}, {} and {}",
                k.saturating_sub(1),
                self.weights.len(),
                self.max_steps.len(),
                self.separations.len()
            )));
        }
        if self.stride == 0 {
            return Err(Error::invalid("surface stride must be at least 1"));
        }
        if let Some([lo, hi]) = self.separations.iter().find(|[lo, hi]| lo > hi) {
            return Err(Error::invalid(format!(
                "surface separation [{lo}, {hi}] is empty"
            )));
        }
        Ok(())
    }
}

/// A stack of surfaces to find, with the images their costs come from.
#[derive(Debug, Clone, PartialEq)]
pub struct LayeredSurfaces {
    pub layers: Vec<SurfaceLayer>,
    /// Per consecutive pair: `[min, max]` of `h_{k+1} - h_k`, in slices.
    pub separations: Vec<[usize; 2]>,
    pub stride: usize,
    /// At most this many magnitudes are sampled per normalised cue for its
    /// robust scale: every 7th voxel, or a wider multiple of 7 when the volume
    /// is larger than `7 * max_scale_samples`. Bounds phase 2's reduce.
    pub max_scale_samples: usize,
    pub method: SurfaceMethod,
}

impl LayeredSurfaces {
    /// Nodes of the cut graph for a `[nz, ny, nx]` volume at `stride`; the
    /// solve holds about 200 bytes per node.
    pub fn graph_nodes(&self, volume: [usize; 3], stride: usize) -> usize {
        self.layers.len() * volume[0] * volume[1].div_ceil(stride) * volume[2].div_ceil(stride)
    }

    /// Voxels between scale samples for a volume of `voxels`: 7, or the
    /// smallest multiple of 7 that keeps the count within the cap.
    pub fn scale_sample_spacing(&self, voxels: usize) -> usize {
        7 * voxels.div_ceil(7 * self.max_scale_samples.max(1)).max(1)
    }
    pub fn shape(&self) -> SurfaceShape {
        SurfaceShape {
            stride: self.stride,
            weights: self.layers.iter().map(|l| l.weight).collect(),
            max_steps: self.layers.iter().map(|l| l.max_step).collect(),
            separations: self.separations.clone(),
        }
    }

    /// Every cue image, deduplicated, in first-use order.
    fn images(&self) -> Vec<ImageId> {
        let mut out = Vec::new();
        for cue in self.layers.iter().flat_map(|l| &l.cues) {
            if !out.contains(&cue.image) {
                out.push(cue.image);
            }
        }
        out
    }

    /// How a phase reads `image`: voxelwise, with any declared element type.
    fn source(&self, image: ImageId) -> SourceInput {
        let input = SourceInput::voxelwise(image);
        match self
            .layers
            .iter()
            .flat_map(|l| &l.cues)
            .find_map(|c| (c.image == image).then_some(c.holding).flatten())
        {
            Some(dtype) => input.holding(dtype),
            None => input,
        }
    }

    /// The normalised cues, deduplicated by image, in first-use order. Scales
    /// are per image, so one image normalised in two layers has one scale.
    fn normalised_images(&self) -> Vec<ImageId> {
        let mut out = Vec::new();
        for cue in self
            .layers
            .iter()
            .flat_map(|l| &l.cues)
            .filter(|c| c.normalise)
        {
            if !out.contains(&cue.image) {
                out.push(cue.image);
            }
        }
        out
    }
}

// ------------------------------------------------------------ resident --

/// The 99.5th percentile of the finite magnitudes, or 1 when there are none
/// or it is zero.
pub fn robust_magnitude(mut magnitudes: Vec<f64>) -> f64 {
    magnitudes.retain(|v| v.is_finite());
    if magnitudes.is_empty() {
        return 1.0;
    }
    let k = (((magnitudes.len() as f64) * 0.995) as usize).min(magnitudes.len() - 1);
    let (_, kth, _) = magnitudes.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    if *kth > 0.0 {
        *kth
    } else {
        1.0
    }
}

/// [`robust_magnitude`] of every 7th element in memory order.
fn sampled_magnitude<'a>(values: impl Iterator<Item = &'a f64>) -> f64 {
    robust_magnitude(values.step_by(7).map(|v| v.abs()).collect())
}

/// Mean over `stride x stride` columns: `[z, y, x]` to `[z, ceil(y/s), ceil(x/s)]`.
pub fn pool_columns(cost: ArrayView3<f64>, stride: usize) -> Array3<f64> {
    let (nz, ny, nx) = cost.dim();
    let s = stride.max(1);
    let (cy, cx) = (ny.div_ceil(s), nx.div_ceil(s));
    let mut out = Array3::<f64>::zeros((nz, cy, cx));
    let mut n = Array2::<f64>::zeros((cy, cx));
    for y in 0..ny {
        for x in 0..nx {
            n[[y / s, x / s]] += 1.0;
        }
    }
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                out[[z, y / s, x / s]] += cost[[z, y, x]];
            }
        }
        for j in 0..cy {
            for i in 0..cx {
                out[[z, j, i]] /= n[[j, i]];
            }
        }
    }
    out
}

/// Fixed-point scale of the graph's costs.
const COST_SCALE: f64 = 1000.0;

/// Solve on pooled costs (`[z, column_y, column_x]`, one per surface). Each is
/// scaled to unit robust magnitude and weighted, then the cut is solved.
/// Returns integer column heights, one map per surface.
pub fn solve_pooled(pooled: Vec<Array3<f64>>, shape: &SurfaceShape) -> Result<Vec<Array2<f32>>> {
    shape.check()?;
    if pooled.len() != shape.weights.len() {
        return Err(Error::invalid(format!(
            "{} pooled costs for {} surfaces",
            pooled.len(),
            shape.weights.len()
        )));
    }
    let dim = pooled[0].dim();
    if pooled.iter().any(|c| c.dim() != dim) {
        return Err(Error::invalid("pooled surface costs differ in shape"));
    }
    let (nz, cy, cx) = dim;
    if nz == 0 || cy == 0 || cx == 0 {
        return Err(Error::invalid("surface costs are empty"));
    }
    let costs = scale_pooled(pooled, shape);
    solve_scaled(&costs, shape)
}

/// Each pooled cost scaled to unit robust magnitude and weighted: the
/// energy [`solve_pooled`] minimises.
pub fn scale_pooled(pooled: Vec<Array3<f64>>, shape: &SurfaceShape) -> Vec<Array3<f64>> {
    pooled
        .into_iter()
        .zip(&shape.weights)
        .map(|(mut c, &w)| {
            let scale = sampled_magnitude(c.iter());
            c.mapv_inplace(|v| v / scale * w);
            c
        })
        .collect()
}

/// The exact cut on already scaled costs.
pub fn solve_scaled(costs: &[Array3<f64>], shape: &SurfaceShape) -> Result<Vec<Array2<f32>>> {
    let (nz, cy, cx) = costs[0].dim();
    let k_count = costs.len();
    let ncol = cy * cx;
    let per_surface = nz * ncol;
    let id = |k: usize, z: usize, j: usize, i: usize| k * per_surface + z * ncol + j * cx + i;
    let nodes = k_count * per_surface;
    let mut g = CutGraph::new(nodes, nodes * 13)?;
    // Forces every column's z = 0 node into the source set. It must exceed any
    // column's total positive terminal weight; the first term is the historical
    // constant (kept so results do not move), the second the guarantee.
    let largest_column: f64 = costs
        .iter()
        .flat_map(|c| (0..cy).flat_map(move |j| (0..cx).map(move |i| (c, j, i))))
        .map(|(c, j, i)| {
            (1..nz)
                .map(|z| ((c[[z, j, i]] - c[[z - 1, j, i]]) * COST_SCALE).abs())
                .sum::<f64>()
        })
        .fold(0.0, f64::max);
    let base =
        -((COST_SCALE * 4.0 * (nz as f64) * 2.0 + 1.0).max(2.0 * largest_column + 1.0)) as i64;

    for (k, c) in costs.iter().enumerate() {
        for z in 0..nz {
            for j in 0..cy {
                for i in 0..cx {
                    let w = if z == 0 {
                        (c[[0, j, i]] * COST_SCALE).round() as i64 + base
                    } else {
                        ((c[[z, j, i]] - c[[z - 1, j, i]]) * COST_SCALE).round() as i64
                    };
                    let n = id(k, z, j, i);
                    if w < 0 {
                        g.add_terminal_weights(n, -w, 0)?;
                    } else {
                        g.add_terminal_weights(n, 0, w)?;
                    }
                    if z > 0 {
                        g.add_edge(n, id(k, z - 1, j, i), INFINITE_CAPACITY, 0)?;
                    }
                }
            }
        }
        let step = shape.max_steps[k];
        for z in 0..nz {
            let zl = z.saturating_sub(step);
            for j in 0..cy {
                for i in 0..cx {
                    let n = id(k, z, j, i);
                    if i + 1 < cx {
                        g.add_edge(n, id(k, zl, j, i + 1), INFINITE_CAPACITY, 0)?;
                        g.add_edge(id(k, z, j, i + 1), id(k, zl, j, i), INFINITE_CAPACITY, 0)?;
                    }
                    if j + 1 < cy {
                        g.add_edge(n, id(k, zl, j + 1, i), INFINITE_CAPACITY, 0)?;
                        g.add_edge(id(k, z, j + 1, i), id(k, zl, j, i), INFINITE_CAPACITY, 0)?;
                    }
                }
            }
        }
    }
    for (k, &[sep_min, sep_max]) in shape.separations.iter().enumerate() {
        for z in 0..nz {
            for j in 0..cy {
                for i in 0..cx {
                    let lower = (z + sep_min).min(nz - 1);
                    g.add_edge(id(k, z, j, i), id(k + 1, lower, j, i), INFINITE_CAPACITY, 0)?;
                    if z >= sep_max {
                        g.add_edge(
                            id(k + 1, z, j, i),
                            id(k, z - sep_max, j, i),
                            INFINITE_CAPACITY,
                            0,
                        )?;
                    }
                }
            }
        }
    }
    g.max_flow();
    Ok((0..k_count)
        .map(|k| {
            Array2::from_shape_fn((cy, cx), |(j, i)| {
                let mut top = 0;
                for z in 0..nz {
                    if g.in_source_set(id(k, z, j, i)) {
                        top = z;
                    } else {
                        break;
                    }
                }
                top as f32
            })
        })
        .collect())
}

// ----------------------------------------------------------- multigrid --

/// How the solve phase finds the surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceMethod {
    /// One exact cut on the pooled columns.
    #[default]
    Exact,
    /// Exact cut on columns pooled `factor` times further (step limits scaled
    /// by `factor`, so the allowed slope is the same), then refinement at the
    /// stride by exact cuts on `tile x tile`-column tiles with shifting edges,
    /// within `band` slices of it ([`refine_tiled`]). A local optimum near the
    /// global one, from graphs bounded by the coarse level and the tile.
    Multigrid {
        factor: usize,
        band: usize,
        tile: usize,
    },
    /// Both; logs how far the multigrid answer is from the exact one, and uses
    /// the exact one. For deciding whether `Multigrid` is good enough.
    Compare {
        factor: usize,
        band: usize,
        tile: usize,
    },
}

/// Energy of `heights` under scaled `costs`: the sum over surfaces and
/// columns of the cost at the column's height. What the cut minimises.
pub fn surface_energy(costs: &[Array3<f64>], heights: &[Array2<f32>]) -> f64 {
    costs
        .iter()
        .zip(heights)
        .map(|(c, h)| {
            h.indexed_iter()
                .map(|((j, i), &z)| c[[z as usize, j, i]])
                .sum::<f64>()
        })
        .sum()
}

/// Largest step between 4-neighbouring columns beyond `max_steps[k]`, and
/// separation violations, summed over columns: zero for a feasible answer.
pub fn surface_violations(heights: &[Array2<f32>], shape: &SurfaceShape) -> usize {
    let mut bad = 0;
    for (k, h) in heights.iter().enumerate() {
        let (cy, cx) = h.dim();
        for j in 0..cy {
            for i in 0..cx {
                for (jj, ii) in [(j + 1, i), (j, i + 1)] {
                    if jj < cy
                        && ii < cx
                        && (h[[j, i]] - h[[jj, ii]]).abs() > shape.max_steps[k] as f32
                    {
                        bad += 1;
                    }
                }
                if let Some(&[lo, hi]) = shape.separations.get(k) {
                    let d = heights[k + 1][[j, i]] - h[[j, i]];
                    if d < lo as f32 || d > hi as f32 {
                        bad += 1;
                    }
                }
            }
        }
    }
    bad
}

/// Refine `start` heights on scaled `costs` by exact per-column moves: each
/// sweep visits the columns in two checkerboard colours, and every column
/// takes the heights (all surfaces jointly) that minimise its cost given its
/// neighbours' current heights, the step and separation limits and a band of
/// `band` slices around `start`. Neighbours never move together, so the energy
/// never rises; stops when a sweep changes nothing or after `max_sweeps`.
/// Returns the heights and the number of sweeps.
pub fn refine_heights(
    costs: &[Array3<f64>],
    shape: &SurfaceShape,
    start: &[Array2<f32>],
    band: usize,
    max_sweeps: usize,
) -> (Vec<Array2<f32>>, usize) {
    let (nz, cy, cx) = costs[0].dim();
    let k_count = costs.len();
    let mut h: Vec<Array2<i64>> = start.iter().map(|s| s.mapv(|v| v as i64)).collect();
    let anchor = h.clone();
    let mut sweeps = 0;
    for _ in 0..max_sweeps {
        sweeps += 1;
        let mut changed = 0usize;
        for colour in 0..2 {
            for j in 0..cy {
                for i in 0..cx {
                    if (j + i) % 2 != colour {
                        continue;
                    }
                    // feasible range per surface from neighbours and band
                    let mut range = Vec::with_capacity(k_count);
                    for k in 0..k_count {
                        let step = shape.max_steps[k] as i64;
                        let a = anchor[k][[j, i]];
                        let (mut lo, mut hi) = (
                            (a - band as i64).max(0),
                            (a + band as i64).min(nz as i64 - 1),
                        );
                        for (dj, di) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                            let (jj, ii) = (j as i64 + dj, i as i64 + di);
                            if jj >= 0 && ii >= 0 && (jj as usize) < cy && (ii as usize) < cx {
                                let n = h[k][[jj as usize, ii as usize]];
                                lo = lo.max(n - step);
                                hi = hi.min(n + step);
                            }
                        }
                        range.push((lo, hi));
                    }
                    if range.iter().any(|(lo, hi)| lo > hi) {
                        continue; // neighbours disagree beyond the limits: leave it
                    }
                    // chain DP over the surfaces with the separation limits
                    let mut best: Vec<Vec<(f64, i64)>> = Vec::with_capacity(k_count);
                    for k in 0..k_count {
                        let (lo, hi) = range[k];
                        let row: Vec<(f64, i64)> = (lo..=hi)
                            .map(|z| {
                                let own = costs[k][[z as usize, j, i]];
                                if k == 0 {
                                    return (own, -1);
                                }
                                let [smin, smax] = shape.separations[k - 1];
                                let (plo, _) = range[k - 1];
                                let mut pick = (f64::INFINITY, -1);
                                for (p, &(f, _)) in best[k - 1].iter().enumerate() {
                                    let zp = plo + p as i64;
                                    let d = z - zp;
                                    if d >= smin as i64 && d <= smax as i64 && f < pick.0 {
                                        pick = (f, zp);
                                    }
                                }
                                (own + pick.0, pick.1)
                            })
                            .collect();
                        best.push(row);
                    }
                    let (lo_last, _) = range[k_count - 1];
                    let Some((p, &(total, _))) = best[k_count - 1]
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1 .0.total_cmp(&b.1 .0))
                    else {
                        continue;
                    };
                    if !total.is_finite() {
                        continue;
                    }
                    let mut chosen = vec![0i64; k_count];
                    chosen[k_count - 1] = lo_last + p as i64;
                    for k in (1..k_count).rev() {
                        let (lo, _) = range[k];
                        chosen[k - 1] = best[k][(chosen[k] - lo) as usize].1;
                    }
                    // keep the current heights on a tie
                    let current: Vec<i64> = (0..k_count).map(|k| h[k][[j, i]]).collect();
                    let feasible_now = current
                        .iter()
                        .zip(&range)
                        .all(|(z, (lo, hi))| z >= lo && z <= hi)
                        && (1..k_count).all(|k| {
                            let d = current[k] - current[k - 1];
                            let [smin, smax] = shape.separations[k - 1];
                            d >= smin as i64 && d <= smax as i64
                        });
                    let now: f64 = (0..k_count)
                        .map(|k| costs[k][[current[k] as usize, j, i]])
                        .sum();
                    if chosen != current && (!feasible_now || total < now) {
                        for k in 0..k_count {
                            h[k][[j, i]] = chosen[k];
                        }
                        changed += 1;
                    }
                }
            }
        }
        if changed == 0 {
            break;
        }
    }
    (
        h.into_iter().map(|a| a.mapv(|v| v as f32)).collect(),
        sweeps,
    )
}

/// Refine `start` heights by exact cuts on tiles of `tile x tile` columns:
/// each tile is re-solved with every column outside it held fixed (its
/// border columns may only take heights within the step limit of their
/// outside neighbours) and every column within `band` slices of `start`.
/// Tile edges shift by half a tile on alternate passes, so a region held at
/// one pass's edge lies inside a tile on the next. Every move is the exact
/// optimum of its tile given the rest, so the energy never rises; stops after
/// two consecutive passes (both offsets) that change nothing, or `max_passes`.
/// Each tile's graph spans only the heights its band allows, so memory is set
/// by the tile, not the image.
pub fn refine_tiled(
    costs: &[Array3<f64>],
    shape: &SurfaceShape,
    start: &[Array2<f32>],
    band: usize,
    tile: usize,
    max_passes: usize,
) -> Result<(Vec<Array2<f32>>, usize)> {
    let (nz, cy, cx) = costs[0].dim();
    let k_count = costs.len();
    let tile = tile.max(2);
    let mut h: Vec<Array2<i64>> = start.iter().map(|s| s.mapv(|v| v as i64)).collect();
    let anchor = h.clone();
    let (mut passes, mut quiet) = (0, 0);
    while passes < max_passes && quiet < 2 {
        let offset = if passes % 2 == 0 { 0 } else { tile / 2 };
        passes += 1;
        let mut changed = 0usize;
        let starts = |n: usize| -> Vec<(usize, usize)> {
            let mut out = Vec::new();
            let mut a = 0usize;
            let first = if offset == 0 { tile } else { offset };
            let mut b = first.min(n);
            while a < n {
                out.push((a, b));
                a = b;
                b = (b + tile).min(n);
            }
            out
        };
        for &(j0, j1) in &starts(cy) {
            for &(i0, i1) in &starts(cx) {
                let (th, tw) = (j1 - j0, i1 - i0);
                // allowed heights per surface and column
                let mut lo = vec![Array2::<i64>::zeros((th, tw)); k_count];
                let mut hi = vec![Array2::<i64>::zeros((th, tw)); k_count];
                for k in 0..k_count {
                    let step = shape.max_steps[k] as i64;
                    for j in j0..j1 {
                        for i in i0..i1 {
                            let a = anchor[k][[j, i]];
                            let (mut l, mut u) = (
                                (a - band as i64).max(0),
                                (a + band as i64).min(nz as i64 - 1),
                            );
                            for (dj, di) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                                let (jj, ii) = (j as i64 + dj, i as i64 + di);
                                let outside = jj < j0 as i64
                                    || jj >= j1 as i64
                                    || ii < i0 as i64
                                    || ii >= i1 as i64;
                                if outside
                                    && jj >= 0
                                    && ii >= 0
                                    && (jj as usize) < cy
                                    && (ii as usize) < cx
                                {
                                    let n = h[k][[jj as usize, ii as usize]];
                                    l = l.max(n - step);
                                    u = u.min(n + step);
                                }
                            }
                            lo[k][[j - j0, i - i0]] = l;
                            hi[k][[j - j0, i - i0]] = u.max(l);
                        }
                    }
                }
                let zlo = lo
                    .iter()
                    .flat_map(|a| a.iter())
                    .copied()
                    .min()
                    .unwrap_or(0)
                    .max(0) as usize;
                let zhi = hi
                    .iter()
                    .flat_map(|a| a.iter())
                    .copied()
                    .max()
                    .unwrap_or(0)
                    .min(nz as i64 - 1) as usize;
                let sub: Vec<Array3<f64>> = (0..k_count)
                    .map(|k| {
                        Array3::from_shape_fn((zhi - zlo + 1, th, tw), |(z, j, i)| {
                            let zz = (z + zlo) as i64;
                            let penalty = if zz < lo[k][[j, i]] || zz > hi[k][[j, i]] {
                                1.0e3
                            } else {
                                0.0
                            };
                            costs[k][[z + zlo, j + j0, i + i0]] + penalty
                        })
                    })
                    .collect();
                let local = solve_scaled(&sub, shape)?;
                for k in 0..k_count {
                    for ((j, i), &z) in local[k].indexed_iter() {
                        let z = z as i64 + zlo as i64;
                        if h[k][[j + j0, i + i0]] != z {
                            h[k][[j + j0, i + i0]] = z;
                            changed += 1;
                        }
                    }
                }
            }
        }
        quiet = if changed == 0 { quiet + 1 } else { 0 };
    }
    Ok((
        h.into_iter().map(|a| a.mapv(|v| v as f32)).collect(),
        passes,
    ))
}

/// [`SurfaceMethod::Multigrid`] on scaled stride-level `costs`.
pub fn solve_multigrid(
    costs: &[Array3<f64>],
    shape: &SurfaceShape,
    factor: usize,
    band: usize,
    tile: usize,
) -> Result<(Vec<Array2<f32>>, usize)> {
    let factor = factor.max(1);
    let (_, cy, cx) = costs[0].dim();
    let coarse: Vec<Array3<f64>> = costs
        .iter()
        .map(|c| pool_columns(c.view(), factor))
        .collect();
    let coarse_shape = SurfaceShape {
        stride: shape.stride * factor,
        weights: shape.weights.clone(),
        max_steps: shape.max_steps.iter().map(|s| s * factor).collect(),
        separations: shape.separations.clone(),
    };
    let rough = solve_scaled(&coarse, &coarse_shape)?;
    let start: Vec<Array2<f32>> = rough
        .iter()
        .map(|h| {
            Array2::from_shape_fn((cy, cx), |(j, i)| upsampled_height(h, factor, j, i).round())
        })
        .collect();
    refine_tiled(costs, shape, &start, band, tile, 1_000)
}

/// Bilinear height at voxel `(y, x)` from column heights `h` (column centres).
pub fn upsampled_height(h: &Array2<f32>, stride: usize, y: usize, x: usize) -> f32 {
    let (cy, cx) = h.dim();
    let s = stride.max(1);
    let coord = |v: usize, n: usize| -> (usize, usize, f32) {
        let f = ((v as f32 + 0.5) / s as f32 - 0.5).clamp(0.0, (n - 1) as f32);
        let a = f.floor() as usize;
        let b = (a + 1).min(n - 1);
        (a, b, f - a as f32)
    };
    let (y0, y1, fy) = coord(y, cy);
    let (x0, x1, fx) = coord(x, cx);
    let top = h[[y0, x0]] * (1.0 - fx) + h[[y0, x1]] * fx;
    let bottom = h[[y1, x0]] * (1.0 - fx) + h[[y1, x1]] * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Column heights upsampled to a `(ny, nx)` voxel grid.
pub fn upsample_heights(h: &Array2<f32>, stride: usize, ny: usize, nx: usize) -> Array2<f32> {
    Array2::from_shape_fn((ny, nx), |(y, x)| upsampled_height(h, stride, y, x))
}

/// Region label at slice `z` given the voxel's surface heights: the number of
/// surfaces at or above it.
#[inline]
pub fn region_label(z: usize, heights: impl Iterator<Item = f32>) -> u32 {
    let zf = z as f32;
    heights.filter(|h| zf >= h.round()).count() as u32
}

/// The whole computation on resident arrays: `costs[k]` is surface `k`'s cost
/// at full resolution. Returns heights upsampled to the voxel grid.
pub fn solve_layered_surfaces(
    costs: &[ArrayView3<f64>],
    shape: &SurfaceShape,
) -> Result<Vec<Array2<f32>>> {
    shape.check()?;
    let (_, ny, nx) = costs
        .first()
        .ok_or_else(|| Error::invalid("no surface costs"))?
        .dim();
    let pooled = costs
        .iter()
        .map(|c| pool_columns(c.view(), shape.stride))
        .collect();
    Ok(solve_pooled(pooled, shape)?
        .iter()
        .map(|h| upsample_heights(h, shape.stride, ny, nx))
        .collect())
}

/// Paint region labels for a `[z, y, x]` buffer at `offset` in the volume.
pub fn paint_regions(
    columns: &[Array2<f32>],
    stride: usize,
    offset: [usize; 3],
    mut out: ArrayViewMut3<u32>,
) {
    let (_, by, bx) = out.dim();
    let heights: Vec<Array2<f32>> = columns
        .iter()
        .map(|h| {
            Array2::from_shape_fn((by, bx), |(y, x)| {
                upsampled_height(h, stride, offset[1] + y, offset[2] + x)
            })
        })
        .collect();
    for (z, mut plane) in out.axis_iter_mut(Axis(0)).enumerate() {
        for ((y, x), o) in plane.indexed_iter_mut() {
            *o = region_label(offset[0] + z, heights.iter().map(|h| h[[y, x]]));
        }
    }
}

// ------------------------------------------------------------ encoding --

const SAMPLES_MAGIC: &[u8; 4] = b"SSa1";
const POOLED_MAGIC: &[u8; 4] = b"SPo1";
const SCALES_MAGIC: &[u8; 4] = b"SSc1";
const HEIGHTS_MAGIC: &[u8; 4] = b"SHe2";

struct Reader<'a> {
    bytes: &'a [u8],
    what: &'static str,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], magic: &[u8; 4], what: &'static str) -> Result<Self> {
        if bytes.len() < 4 || &bytes[..4] != magic {
            return Err(Error::invalid(format!(
                "{what}: not a {what} blob (bad magic or version)"
            )));
        }
        Ok(Self {
            bytes: &bytes[4..],
            what,
        })
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.bytes.len() < n {
            return Err(Error::invalid(format!("{}: truncated", self.what)));
        }
        let (head, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(head)
    }

    fn u64(&mut self) -> Result<usize> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()) as usize)
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(self
            .take(n * 4)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }

    fn f64s(&mut self, n: usize) -> Result<Vec<f64>> {
        Ok(self
            .take(n * 8)?
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}

fn put_u64(out: &mut Vec<u8>, v: usize) {
    out.extend((v as u64).to_le_bytes());
}

// ------------------------------------------------------------ phase 1 --

/// Phase 1: magnitudes of the normalised cues at every 7th global voxel.
#[derive(Debug, Clone)]
pub struct SurfaceScaleSamplesOp {
    stream: String,
    images: Vec<ImageId>,
    sources: Vec<SourceInput>,
    every: usize,
}

impl FragmentOp for SurfaceScaleSamplesOp {
    fn name(&self) -> &'static str {
        "surface cue scale samples"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            &self.stream,
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        self.sources.clone()
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        // only reached when no cue is normalised: nothing to sample
        self.apply_with(at, SourceBlocks::none())
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let [_, ny, nx] = at.volume();
        let start = &at.core.start;
        let mut out = SAMPLES_MAGIC.to_vec();
        put_u64(&mut out, self.images.len());
        for &image in &self.images {
            let cue = sources.get(image.index())?.as_array()?.widened();
            let mut kept = Vec::new();
            for ((z, y, x), &v) in cue.indexed_iter() {
                let global = ((start[0] + z) * ny + start[1] + y) * nx + start[2] + x;
                if global % self.every == 0 {
                    kept.push(v.abs() as f32);
                }
            }
            put_u64(&mut out, kept.len());
            out.extend(kept.iter().flat_map(|v| v.to_le_bytes()));
        }
        Ok(BlockOutput::fragment(&self.stream, out))
    }
}

// ------------------------------------------------------------ phase 2 --

/// Phase 2 (barrier): cue scales once, then each block's pooled surface costs.
#[derive(Debug, Clone)]
pub struct SurfacePoolOp {
    samples_stream: String,
    samples_phase: usize,
    stream: String,
    surfaces: LayeredSurfaces,
}

impl SurfacePoolOp {
    fn cost_of(
        &self,
        layer: &SurfaceLayer,
        cues: &[(ImageId, Array3<f64>)],
        scales: &[(ImageId, f64)],
        shape: (usize, usize, usize),
    ) -> Result<Array3<f64>> {
        let mut acc: Option<Array3<f64>> = None;
        for cue in &layer.cues {
            let values = &cues
                .iter()
                .find(|(i, _)| *i == cue.image)
                .expect("every cue image is read")
                .1;
            let w = cue.weight;
            if cue.normalise {
                let scale = scales
                    .iter()
                    .find(|(i, _)| *i == cue.image)
                    .ok_or_else(|| Error::invalid("missing cue scale"))?
                    .1;
                let a = acc.get_or_insert_with(|| Array3::zeros(shape));
                a.zip_mut_with(values, |b, &c| *b += w * c / scale);
            } else if acc.is_none() && w == 1.0 {
                acc = Some(values.clone());
            } else {
                let a = acc.get_or_insert_with(|| Array3::zeros(shape));
                a.zip_mut_with(values, |b, &c| *b += w * c);
            }
        }
        Ok(acc.unwrap_or_else(|| Array3::zeros(shape)))
    }
}

impl FragmentOp for SurfacePoolOp {
    fn name(&self) -> &'static str {
        "surface cost pooling"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.samples_stream, self.samples_phase)]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            &self.stream,
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        self.surfaces
            .images()
            .into_iter()
            .map(|i| self.surfaces.source(i))
            .collect()
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let images = self.surfaces.normalised_images();
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); images.len()];
        at.stream_fragments(&self.samples_stream, &mut |_key, bytes| {
            let mut r = Reader::new(bytes, SAMPLES_MAGIC, "surface scale samples")?;
            if r.u64()? != images.len() {
                return Err(Error::invalid("surface scale samples: cue count changed"));
            }
            for s in samples.iter_mut() {
                let n = r.u64()?;
                s.extend(r.f32s(n)?.into_iter().map(f64::from));
            }
            Ok(())
        })?;
        let mut out = SCALES_MAGIC.to_vec();
        put_u64(&mut out, images.len());
        for s in samples {
            out.extend(robust_magnitude(s).to_le_bytes());
        }
        Ok(out)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "surface cost pooling reads its cues through apply_with",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let s = self.surfaces.stride;
        let [_, ny, nx] = at.volume();
        let start = [at.core.start[0], at.core.start[1], at.core.start[2]];
        let shape = at.core.shape3();
        if start[1] % s != 0
            || start[2] % s != 0
            || (shape[1] % s != 0 && start[1] + shape[1] != ny)
            || (shape[2] % s != 0 && start[2] + shape[2] != nx)
        {
            return Err(Error::invalid(format!(
                "surface pooling needs blocks that are multiples of the stride {s} in y and x, \
                 so no column straddles a seam; block at {start:?} is {shape:?}"
            )));
        }
        let mut r = Reader::new(at.reduced, SCALES_MAGIC, "surface cue scales")?;
        let n = r.u64()?;
        let values = r.f64s(n)?;
        let scales: Vec<(ImageId, f64)> = self
            .surfaces
            .normalised_images()
            .into_iter()
            .zip(values)
            .collect();
        let cues: Vec<(ImageId, Array3<f64>)> = self
            .surfaces
            .images()
            .into_iter()
            .map(|i| Ok((i, sources.get(i.index())?.as_array()?.widened())))
            .collect::<Result<_>>()?;
        let dims = (shape[0], shape[1], shape[2]);
        let mut out = POOLED_MAGIC.to_vec();
        let (cy0, cx0) = (start[1] / s, start[2] / s);
        let (ncy, ncx) = (shape[1].div_ceil(s), shape[2].div_ceil(s));
        for v in [
            start[0],
            shape[0],
            cy0,
            ncy,
            cx0,
            ncx,
            self.surfaces.layers.len(),
        ] {
            put_u64(&mut out, v);
        }
        for layer in &self.surfaces.layers {
            let cost = self.cost_of(layer, &cues, &scales, dims)?;
            let pooled = pool_columns(cost.view(), s);
            out.extend(pooled.iter().flat_map(|v| v.to_le_bytes()));
        }
        Ok(BlockOutput::fragment(&self.stream, out))
    }
}

// ------------------------------------------------------------ phase 3 --

/// Phase 3 (barrier): solve on the pooled costs once, then paint region labels.
#[derive(Debug, Clone)]
pub struct SurfaceSolveOp {
    pooled_stream: String,
    pooled_phase: usize,
    heights_stream: String,
    shape: SurfaceShape,
    method: SurfaceMethod,
}

/// The solved surfaces at column resolution, as the solve phase publishes them
/// on `<stream>.heights` for later phases.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceColumns {
    pub stride: usize,
    /// Per surface, integer height per column.
    pub columns: Vec<Array2<f32>>,
}

impl SurfaceColumns {
    pub fn encode(&self) -> Vec<u8> {
        let (cy, cx) = self.columns.first().map_or((0, 0), |c| c.dim());
        let mut out = HEIGHTS_MAGIC.to_vec();
        for v in [self.stride, self.columns.len(), cy, cx] {
            put_u64(&mut out, v);
        }
        for h in &self.columns {
            out.extend(h.iter().flat_map(|v| v.to_le_bytes()));
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes, HEIGHTS_MAGIC, "surface heights")?;
        let (stride, k, cy, cx) = (r.u64()?, r.u64()?, r.u64()?, r.u64()?);
        let columns = (0..k)
            .map(|_| Array2::from_shape_vec((cy, cx), r.f32s(cy * cx)?).map_err(Error::backend))
            .collect::<Result<_>>()?;
        Ok(Self { stride, columns })
    }

    /// In a later barrier op's `reduce`: the columns published on `stream`
    /// (declare `FragmentInput::own(stream, solve_phase)` as an input).
    pub fn from_stream(at: &PhaseView<'_>, stream: &str) -> Result<Self> {
        let mut found = None;
        at.stream_fragments(stream, &mut |_key, bytes| {
            found = Some(Self::decode(bytes)?);
            Ok(())
        })?;
        found.ok_or_else(|| Error::invalid(format!("no surface heights on stream {stream:?}")))
    }

    /// Surface `k` at voxel `(y, x)`: the upsampled height rounded to a slice —
    /// the first z labelled below it in the region image.
    pub fn slice_at(&self, k: usize, y: usize, x: usize) -> f32 {
        upsampled_height(&self.columns[k], self.stride, y, x).round()
    }
}

impl FragmentOp for SurfaceSolveOp {
    fn name(&self) -> &'static str {
        "layered surface solve"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.pooled_stream, self.pooled_phase)]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        // persistent: a caller reads the solved surfaces back after the run
        vec![FragmentOutput::new(
            &self.heights_stream,
            Lifecycle::Persistent,
            Coverage::Sparse,
        )]
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let s = self.shape.stride;
        let [nz, ny, nx] = at.volume();
        let (cy, cx) = (ny.div_ceil(s), nx.div_ceil(s));
        let k_count = self.shape.weights.len();
        let mut pooled = vec![Array3::<f64>::zeros((nz, cy, cx)); k_count];
        at.stream_fragments(&self.pooled_stream, &mut |_key, bytes| {
            let mut r = Reader::new(bytes, POOLED_MAGIC, "pooled surface costs")?;
            let (z0, bz, cy0, ncy, cx0, ncx, k) = (
                r.u64()?,
                r.u64()?,
                r.u64()?,
                r.u64()?,
                r.u64()?,
                r.u64()?,
                r.u64()?,
            );
            if k != k_count {
                return Err(Error::invalid(
                    "pooled surface costs: surface count changed",
                ));
            }
            for p in pooled.iter_mut() {
                let values = r.f64s(bz * ncy * ncx)?;
                let block =
                    Array3::from_shape_vec((bz, ncy, ncx), values).map_err(Error::backend)?;
                p.slice_mut(ndarray::s![z0..z0 + bz, cy0..cy0 + ncy, cx0..cx0 + ncx])
                    .assign(&block);
            }
            Ok(())
        })?;
        self.shape.check()?;
        let costs = scale_pooled(pooled, &self.shape);
        let columns = match self.method {
            SurfaceMethod::Exact => solve_scaled(&costs, &self.shape)?,
            SurfaceMethod::Multigrid { factor, band, tile } => {
                solve_multigrid(&costs, &self.shape, factor, band, tile)?.0
            }
            SurfaceMethod::Compare { factor, band, tile } => {
                let t0 = std::time::Instant::now();
                let exact = solve_scaled(&costs, &self.shape)?;
                let t1 = std::time::Instant::now();
                let (multi, sweeps) = solve_multigrid(&costs, &self.shape, factor, band, tile)?;
                let t2 = std::time::Instant::now();
                let (e_exact, e_multi) = (
                    surface_energy(&costs, &exact),
                    surface_energy(&costs, &multi),
                );
                for (k, (a, b)) in exact.iter().zip(&multi).enumerate() {
                    let diff: Vec<f32> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
                    let n = diff.len() as f64;
                    eprintln!(
                        "  surface {k}: multigrid x{factor} band {band} tile {tile}: columns differing {:.2}%, >1 slice {:.2}%, \
                         mean |diff| {:.3}, max {}",
                        100.0 * diff.iter().filter(|d| **d > 0.0).count() as f64 / n,
                        100.0 * diff.iter().filter(|d| **d > 1.0).count() as f64 / n,
                        diff.iter().map(|d| *d as f64).sum::<f64>() / n,
                        diff.iter().cloned().fold(0.0f32, f32::max)
                    );
                }
                eprintln!(
                    "  energy exact {e_exact:.3} multigrid {e_multi:.3} (gap {:.4}%), violations {}, \
                     passes {sweeps}, exact {:.1}s, multigrid {:.1}s",
                    100.0 * (e_multi - e_exact) / e_exact.abs().max(1e-12),
                    surface_violations(&multi, &self.shape),
                    (t1 - t0).as_secs_f64(),
                    (t2 - t1).as_secs_f64()
                );
                exact
            }
        };
        Ok(SurfaceColumns { stride: s, columns }.encode())
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let surfaces = SurfaceColumns::decode(at.reduced)?;
        let mut buffer = at.output_buffer(0.0)?;
        if let Some(out) = buffer.as_array_mut() {
            let offset = [at.read.start[0], at.read.start[1], at.read.start[2]];
            paint_regions(
                &surfaces.columns,
                surfaces.stride,
                offset,
                out.view_mut::<u32>()?,
            );
        }
        // one block publishes the columns for later phases
        let output = if at.index == [0, 0, 0] {
            BlockOutput::fragment(&self.heights_stream, at.reduced.to_vec())
        } else {
            BlockOutput::nothing()
        };
        Ok(output.with_pixels(buffer))
    }
}

/// Append the three phases that find `surfaces` and write a `u32` region-label
/// image (0 above the first surface, `K` at and below the last). Streams are
/// named `<stream>.samples` and `<stream>.pooled`; the solved columns are
/// published on `<stream>.heights` by the returned phase (read them with
/// [`SurfaceColumns::from_stream`]). Block edges in y and x must be multiples
/// of `surfaces.stride`.
pub fn append_layered_surface_phases(
    plan: &mut PlanBuilder,
    stream: &str,
    surfaces: LayeredSurfaces,
) -> Result<Phase> {
    surfaces.shape().check()?;
    if surfaces.layers.iter().any(|l| l.cues.is_empty()) {
        return Err(Error::invalid("every surface needs at least one cue image"));
    }
    let samples_stream = format!("{stream}.samples");
    let pooled_stream = format!("{stream}.pooled");
    let images = surfaces.normalised_images();
    let sources = images.iter().map(|&i| surfaces.source(i)).collect();
    let every = surfaces.scale_sample_spacing(plan.grid().volume().iter().product());
    let samples = plan.fragments(SurfaceScaleSamplesOp {
        stream: samples_stream.clone(),
        images,
        sources,
        every,
    })?;
    let shape = surfaces.shape();
    let method = surfaces.method;
    let pool = plan.fragments(SurfacePoolOp {
        samples_stream,
        samples_phase: samples.index(),
        stream: pooled_stream.clone(),
        surfaces,
    })?;
    plan.fragments(SurfaceSolveOp {
        pooled_stream,
        pooled_phase: pool.index(),
        heights_stream: format!("{stream}.heights"),
        shape,
        method,
    })
}

// ---------------------------------------------------------------- band --

/// A `u8` mask of the voxels between two solved surfaces, each shifted:
/// `z >= slice(upper) - upper_shift` and `z < slice(lower) - lower_shift`
/// (slices; `slice` as in [`SurfaceColumns::slice_at`]). A barrier that reads
/// the columns a solve phase published, so it can follow it anywhere in the
/// plan.
#[derive(Debug, Clone)]
pub struct SurfaceBandOp {
    heights_stream: String,
    heights_phase: usize,
    upper: (usize, f32),
    lower: (usize, f32),
}

impl SurfaceBandOp {
    /// `heights` is the phase [`append_layered_surface_phases`] returned and
    /// `stream` the name it was given.
    pub fn new(stream: &str, heights: Phase, upper: (usize, f32), lower: (usize, f32)) -> Self {
        Self {
            heights_stream: format!("{stream}.heights"),
            heights_phase: heights.index(),
            upper,
            lower,
        }
    }
}

impl FragmentOp for SurfaceBandOp {
    fn name(&self) -> &'static str {
        "band between surfaces"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U8
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.heights_stream, self.heights_phase)]
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let surfaces = SurfaceColumns::from_stream(at, &self.heights_stream)?;
        let k = surfaces.columns.len();
        if self.upper.0 >= k || self.lower.0 >= k {
            return Err(Error::invalid(format!(
                "a band between surfaces {} and {} of a stack of {k}",
                self.upper.0, self.lower.0
            )));
        }
        Ok(surfaces.encode())
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let surfaces = SurfaceColumns::decode(at.reduced)?;
        let mut buffer = at.output_buffer(0.0)?;
        if let Some(out) = buffer.as_array_mut() {
            let start = [at.read.start[0], at.read.start[1], at.read.start[2]];
            let mut view = out.view_mut::<u8>()?;
            let (_, by, bx) = view.dim();
            for y in 0..by {
                for x in 0..bx {
                    let (gy, gx) = (start[1] + y, start[2] + x);
                    let top = surfaces.slice_at(self.upper.0, gy, gx) - self.upper.1;
                    let bottom = surfaces.slice_at(self.lower.0, gy, gx) - self.lower.1;
                    for (z, v) in view.slice_mut(ndarray::s![.., y, x]).iter_mut().enumerate() {
                        let zf = (start[0] + z) as f32;
                        *v = u8::from(zf >= top && zf < bottom);
                    }
                }
            }
        }
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

// --------------------------------------------------------------- lines --

/// One-voxel surfaces for display: surface `k` is painted `k + 1` from its
/// slice down to one above its lowest 4-neighbour, so the line stays connected
/// across height steps in XZ and YZ views. Later surfaces win where they meet.
/// `out` is a block at `offset` of a volume `[nz, ny, nx]`.
pub fn paint_surface_lines(
    surfaces: &SurfaceColumns,
    offset: [usize; 3],
    volume: [usize; 3],
    mut out: ArrayViewMut3<u64>,
) {
    let (bz, by, bx) = out.dim();
    let [nz, ny, nx] = volume;
    for k in 0..surfaces.columns.len() {
        for y in 0..by {
            for x in 0..bx {
                let (gy, gx) = (offset[1] + y, offset[2] + x);
                let z0 = surfaces.slice_at(k, gy, gx) as usize;
                if z0 >= nz {
                    continue;
                }
                let mut z1 = z0;
                for (dy, dx) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (j, i) = (gy as i64 + dy, gx as i64 + dx);
                    if j >= 0 && i >= 0 && (j as usize) < ny && (i as usize) < nx {
                        let zn = surfaces.slice_at(k, j as usize, i as usize) as usize;
                        if zn < nz {
                            z1 = z1.max(zn.saturating_sub(1));
                        }
                    }
                }
                for z in z0.max(offset[0])..=z1.min(offset[0] + bz - 1) {
                    out[[z - offset[0], y, x]] = k as u64 + 1;
                }
            }
        }
    }
}

/// Writes [`paint_surface_lines`] as a `u64` label image, from the columns a
/// solve phase published.
#[derive(Debug, Clone)]
pub struct SurfaceLinesOp {
    heights_stream: String,
    heights_phase: usize,
}

impl SurfaceLinesOp {
    pub fn new(stream: &str, heights: Phase) -> Self {
        Self {
            heights_stream: format!("{stream}.heights"),
            heights_phase: heights.index(),
        }
    }
}

impl FragmentOp for SurfaceLinesOp {
    fn name(&self) -> &'static str {
        "surface lines"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U64
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.heights_stream, self.heights_phase)]
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        Ok(SurfaceColumns::from_stream(at, &self.heights_stream)?.encode())
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let surfaces = SurfaceColumns::decode(at.reduced)?;
        let mut buffer = at.output_buffer(0.0)?;
        if let Some(out) = buffer.as_array_mut() {
            let offset = [at.read.start[0], at.read.start[1], at.read.start[2]];
            paint_surface_lines(&surfaces, offset, at.volume(), out.view_mut::<u64>()?);
        }
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two noisy tilted planes plus a strong decoy above the lower one that a
    /// column-wise minimum would latch onto; the coupled solve ignores it.
    fn planes() -> (
        Array3<f64>,
        Array3<f64>,
        impl Fn(usize, usize) -> usize,
        impl Fn(usize, usize) -> usize,
    ) {
        let (nz, ny, nx) = (40, 24, 24);
        let truth_top = |y: usize, _x: usize| 8 + y / 6;
        let truth_bot = |_y: usize, x: usize| 25 + x / 8;
        let mut seed = 7u64;
        let mut noise = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.4
        };
        let mut top = Array3::<f64>::zeros((nz, ny, nx));
        let mut bot = Array3::<f64>::zeros((nz, ny, nx));
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    top[[z, y, x]] = noise() - if z == truth_top(y, x) { 1.0 } else { 0.0 };
                    bot[[z, y, x]] = noise() - if z == truth_bot(y, x) { 1.0 } else { 0.0 };
                }
            }
        }
        for y in 10..13 {
            for x in 10..13 {
                bot[[15, y, x]] = -3.0;
            }
        }
        (top, bot, truth_top, truth_bot)
    }

    fn shape(stride: usize) -> SurfaceShape {
        SurfaceShape {
            stride,
            weights: vec![1.0, 1.0],
            max_steps: vec![1, 1],
            separations: vec![[2, 35]],
        }
    }

    #[test]
    fn recovers_planes_and_ignores_a_decoy() {
        let (top, bot, truth_top, truth_bot) = planes();
        let h = solve_layered_surfaces(&[top.view(), bot.view()], &shape(1)).unwrap();
        let mut wrong = 0;
        for y in 0..24 {
            for x in 0..24 {
                wrong += usize::from((h[0][[y, x]] - truth_top(y, x) as f32).abs() > 0.5);
                wrong += usize::from((h[1][[y, x]] - truth_bot(y, x) as f32).abs() > 0.5);
            }
        }
        assert!(wrong < 10, "{wrong} wrong columns");
    }

    #[test]
    fn separation_and_step_limits_hold() {
        let (top, bot, _, _) = planes();
        let mut sh = shape(2);
        sh.separations = vec![[5, 30]];
        let pooled = vec![pool_columns(top.view(), 2), pool_columns(bot.view(), 2)];
        let h = solve_pooled(pooled, &sh).unwrap();
        for ((y, x), &t) in h[0].indexed_iter() {
            let b = h[1][[y, x]];
            assert!(b - t >= 5.0 && b - t <= 30.0);
            for (j, i) in [(y + 1, x), (y, x + 1)] {
                if let (Some(&n0), Some(&n1)) = (h[0].get((j, i)), h[1].get((j, i))) {
                    assert!((n0 - t).abs() <= 1.0 && (n1 - b).abs() <= 1.0);
                }
            }
        }
    }

    /// Multigrid lands on a feasible answer whose energy is at least the exact
    /// one's, and on these planes very close to it.
    #[test]
    fn multigrid_is_feasible_and_close_to_exact() {
        let (top, bot, _, _) = planes();
        let sh = shape(1);
        let costs = scale_pooled(
            vec![pool_columns(top.view(), 1), pool_columns(bot.view(), 1)],
            &sh,
        );
        let exact = solve_scaled(&costs, &sh).unwrap();
        let (multi, sweeps) = solve_multigrid(&costs, &sh, 4, 3, 8).unwrap();
        assert_eq!(surface_violations(&exact, &sh), 0);
        assert_eq!(surface_violations(&multi, &sh), 0);
        let (e, m) = (
            surface_energy(&costs, &exact),
            surface_energy(&costs, &multi),
        );
        assert!(m >= e - 1e-9, "multigrid {m} below the exact optimum {e}");
        let differing = exact
            .iter()
            .zip(&multi)
            .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
            .sum::<usize>();
        assert!(
            differing <= 24 * 24 * 2 / 20,
            "{differing} columns differ after {sweeps} sweeps"
        );
    }

    #[test]
    fn region_labels_count_surfaces_above() {
        let columns = vec![
            Array2::from_elem((2, 2), 3.0f32),
            Array2::from_elem((2, 2), 6.0f32),
        ];
        let mut out = Array3::<u32>::zeros((8, 2, 2));
        paint_regions(&columns, 1, [0, 0, 0], out.view_mut());
        let labels: Vec<u32> = (0..8).map(|z| out[[z, 0, 0]]).collect();
        assert_eq!(labels, vec![0, 0, 0, 1, 1, 1, 2, 2]);
    }

    /// The three planned phases give the resident answer, bit for bit, on
    /// several block grids — including one whose blocks split z.
    #[test]
    fn planned_phases_match_resident_on_any_stride_aligned_grid() {
        use crate::env::ArrayEnvironment;
        use crate::geometry::BlockGrid;
        use crate::strategy::{execute_phases, Hints};
        use crate::voxels::Voxels;

        let (top, bot, _, _) = planes();
        let volume = [40, 24, 24];
        let cue2 = Array3::from_shape_fn((40, 24, 24), |(z, y, x)| {
            ((z * 31 + y * 7 + x) % 13) as f64 - 6.0
        });
        let surfaces = LayeredSurfaces {
            layers: vec![
                SurfaceLayer {
                    cues: vec![SurfaceCue::raw(0, 1.0)],
                    weight: 1.0,
                    max_step: 1,
                },
                SurfaceLayer {
                    cues: vec![
                        SurfaceCue::normalised(ImageId::supplied(0), 1.0).holding(Dtype::F64),
                        SurfaceCue::normalised(ImageId::supplied(1), 0.05).holding(Dtype::F64),
                    ],
                    weight: 1.0,
                    max_step: 1,
                },
            ],
            separations: vec![[2, 35]],
            stride: 2,
            max_scale_samples: 1 << 24,
            method: SurfaceMethod::Exact,
        };
        let scale = |a: &Array3<f64>| sampled_magnitude(a.iter());
        let (s0, s1) = (scale(&bot), scale(&cue2));
        let mut bottom = Array3::<f64>::zeros(bot.raw_dim());
        bottom.zip_mut_with(&bot, |b, &c| *b += 1.0 * c / s0);
        bottom.zip_mut_with(&cue2, |b, &c| *b += 0.05 * c / s1);
        let shape = surfaces.shape();
        let pooled = vec![pool_columns(top.view(), 2), pool_columns(bottom.view(), 2)];
        let columns = solve_pooled(pooled, &shape).unwrap();
        let mut expected = Array3::<u32>::zeros((40, 24, 24));
        paint_regions(&columns, 2, [0, 0, 0], expected.view_mut());
        let solved = SurfaceColumns {
            stride: 2,
            columns: columns.clone(),
        };
        let mut expected_lines = Array3::<u64>::zeros((40, 24, 24));
        paint_surface_lines(&solved, [0, 0, 0], volume, expected_lines.view_mut());
        assert!(expected_lines.iter().filter(|&&v| v == 1).count() >= 24 * 24);
        let expected_band = Array3::from_shape_fn((40, 24, 24), |(z, y, x)| {
            let zf = z as f32;
            u8::from(zf >= solved.slice_at(0, y, x) - 2.0 && zf < solved.slice_at(1, y, x) - 1.5)
        });
        assert!(expected_band.iter().any(|&v| v == 1) && expected_band.iter().any(|&v| v == 0));

        for block in [[40, 24, 24], [8, 8, 8], [16, 4, 12]] {
            let mut plan =
                PlanBuilder::new(volume, Dtype::F64, BlockGrid::new(volume, block).unwrap());
            let phase = append_layered_surface_phases(&mut plan, "surf", surfaces.clone()).unwrap();
            let lines = plan.fragments(SurfaceLinesOp::new("surf", phase)).unwrap();
            let band = plan
                .fragments(SurfaceBandOp::new("surf", phase, (0, 2.0), (1, 1.5)))
                .unwrap();
            let assembly = plan.finish().unwrap();
            let env = ArrayEnvironment::with_inputs(
                Voxels::from(top.clone()),
                vec![Voxels::from(bot.clone()), Voxels::from(cue2.clone())],
                &assembly.decomposition,
                [8, 8, 8],
            )
            .unwrap();
            execute_phases(
                "surfaces",
                &assembly.workflow,
                &assembly.decomposition,
                &Hints {
                    keep_images: [phase.image().unwrap(), lines.image().unwrap()]
                        .into_iter()
                        .collect(),
                    ..Hints::default()
                },
                &env,
                &[],
                &assembly.work(),
            )
            .unwrap();
            let got = env.image(phase.image().unwrap().index());
            assert_eq!(
                got.view::<u32>().unwrap(),
                expected.view(),
                "block {block:?}"
            );
            let got_lines = env.image(lines.image().unwrap().index());
            assert_eq!(
                got_lines.view::<u64>().unwrap(),
                expected_lines.view(),
                "lines, block {block:?}"
            );
            let got_band = env.image(band.image().unwrap().index());
            assert_eq!(
                got_band.view::<u8>().unwrap(),
                expected_band.view(),
                "band, block {block:?}"
            );
        }
    }

    #[test]
    fn refuses_inconsistent_shapes() {
        let mut sh = shape(1);
        sh.separations.clear();
        assert!(sh.check().is_err());
        let mut sh = shape(1);
        sh.separations = vec![[4, 2]];
        assert!(sh.check().is_err());
    }
}
