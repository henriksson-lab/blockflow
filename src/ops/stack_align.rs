// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Slice-to-slice stack alignment: estimating and removing the in-plane
// translation between consecutive z-planes of a serially acquired stack.
//
// Estimation
// ----------
// For each consecutive pair of planes, a central window of both (optionally a
// caller-sized one) is centred and scaled to unit variance — so a brightness
// drift between the planes does not move the answer — and their mean squared
// difference is evaluated at every integer lag within a bound, through
// [`fft::SquaredDifference`](super::fft::SquaredDifference). The lowest lag is
// refined to sub-pixel precision by a quadratic fit over its 3 x 3
// neighbourhood (parabolic peak interpolation in two dimensions). That fit is
// exact for a quadratic bowl; on the test's smooth texture the accumulated
// error over nine pairs is below 0.01 px (an axis-by-axis parabola, which
// ignores the bowl's tilt, was 0.17 px). The bias accumulates along the
// path like a random walk, so long stacks drift by more than one pair errs.
//
// The pairwise displacements are summed into a path: `path[z]` is how far plane
// `z`'s content sits from plane `0`'s. A path also contains genuine structure
// — the sample cut at a slant moves features steadily across the planes — and
// aligning that away would shear the volume. So the path is **detrended** by a
// caller's choice ([`Detrend`]): subtracting a running mean keeps slow drift and
// removes jitter; subtracting a least-squares line keeps a constant tilt;
// nothing subtracts the whole path and registers every plane to plane `0`.
//
// The estimate is a whole-stack pass over pairs of planes — it is not local in
// z, and it is not a `BlockOp`.
//
// Application
// -----------
// [`apply_slice_shifts_into`] translates each plane by its shift with bilinear
// interpolation, `out_z(x) = in_z(x + shift[z])`, writing a fill value where
// the source falls outside the plane. [`SliceShiftOp`] is the same as a
// [`BlockOp`]: reach nothing on z and `floor(max |shift|) + 1` in-plane, a
// stencil, reading global coordinates from the [`Anchor`].

use ndarray::{s, Array2, ArrayView2, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Slicing};
use crate::voxels::Voxels;

use super::fft::{Landscape, Padding, ShiftWindow, SquaredDifference};
use super::shapes_agree;

/// What is subtracted from the cumulative path before it is applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Detrend {
    /// Nothing: every plane is registered to plane `0`.
    None,
    /// A centred running mean of `width` planes (odd), clipped at the ends.
    /// Keeps drift slower than the window, removes jitter.
    RunningMean { width: usize },
    /// A least-squares line in `z`, per axis. Keeps a constant tilt.
    Linear,
}

/// How [`estimate_slice_shifts`] searches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliceShiftParams {
    max_shift: [usize; 2],
    window: Option<[usize; 2]>,
    detrend: Detrend,
}

impl SliceShiftParams {
    /// * `max_shift`: the largest displacement, in pixels per axis, searched
    ///   between **consecutive** planes.
    /// * `window`: the size of the central window compared, or `None` for the
    ///   whole plane. Clipped to the plane.
    /// * `detrend`: see [`Detrend`].
    pub fn new(
        max_shift: [usize; 2],
        window: Option<[usize; 2]>,
        detrend: Detrend,
    ) -> Result<Self> {
        if let Some(window) = window {
            if window.iter().zip(max_shift).any(|(&w, m)| w <= m) {
                return Err(Error::InvalidArgument(format!(
                    "a comparison window of {window:?} cannot be searched over shifts up to \
                     {max_shift:?}: every lag must leave an overlap"
                )));
            }
        }
        if let Detrend::RunningMean { width } = detrend {
            if width == 0 || width % 2 == 0 {
                return Err(Error::InvalidArgument(format!(
                    "a running-mean detrend needs an odd, positive width; got {width}"
                )));
            }
        }
        Ok(Self {
            max_shift,
            window,
            detrend,
        })
    }
}

/// The central `window` of a plane, centred and scaled to unit variance.
fn standardised<T: Copy + Into<f64>>(plane: ArrayView2<'_, T>, window: [usize; 2]) -> Array2<f64> {
    let (rows, cols) = plane.dim();
    let start = [(rows - window[0]) / 2, (cols - window[1]) / 2];
    let crop = plane.slice(s![
        start[0]..start[0] + window[0],
        start[1]..start[1] + window[1]
    ]);
    let mut out = crop.mapv(|v| v.into());
    let n = out.len() as f64;
    let mean = out.sum() / n;
    let variance = out.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let scale = if variance > 0.0 {
        1.0 / variance.sqrt()
    } else {
        1.0
    };
    out.mapv_inplace(|v| (v - mean) * scale);
    out
}

/// The vertex of the parabola through `(-1, minus)`, `(0, centre)`, `(1, plus)`,
/// or `0` when the three do not bound a minimum.
fn parabolic_offset(minus: f64, centre: f64, plus: f64) -> f64 {
    let curvature = minus - 2.0 * centre + plus;
    if !(curvature > 0.0) || !minus.is_finite() || !plus.is_finite() {
        return 0.0;
    }
    (0.5 * (minus - plus) / curvature).clamp(-0.5, 0.5)
}

/// The sub-pixel lag of a landscape's minimum.
///
/// A quadratic in both lags fitted to the 3 x 3 neighbourhood of the integer
/// minimum, cross term included — a texture that is not isotropic makes a
/// tilted bowl, and two one-axis parabolas through a tilted bowl miss its
/// vertex. Where the neighbourhood is incomplete (the minimum is on the edge of
/// the search window) or the fit is not a bowl, the two one-axis parabolas are
/// used instead, and failing those the integer lag.
fn refined_minimum(landscape: &Landscape) -> Result<[f64; 2]> {
    let (lag, centre) = landscape
        .argmin()
        .ok_or_else(|| Error::InvalidArgument("no lag in the search window overlaps".into()))?;
    let at = |d0: isize, d1: isize| {
        landscape
            .at([lag[0] + d0, lag[1] + d1])
            .filter(|value| value.is_finite())
    };
    let mut refined = [lag[0] as f64, lag[1] as f64];
    let neighbourhood: Option<Vec<f64>> = (-1..=1)
        .flat_map(|d0| (-1..=1).map(move |d1| (d0, d1)))
        .map(|(d0, d1)| at(d0, d1))
        .collect();
    if let Some(v) = neighbourhood {
        let f = |d0: usize, d1: usize| v[d0 * 3 + d1];
        let g0 = 0.5 * (f(2, 1) - f(0, 1));
        let g1 = 0.5 * (f(1, 2) - f(1, 0));
        let h00 = f(2, 1) - 2.0 * centre + f(0, 1);
        let h11 = f(1, 2) - 2.0 * centre + f(1, 0);
        let h01 = 0.25 * (f(2, 2) - f(2, 0) - f(0, 2) + f(0, 0));
        let det = h00 * h11 - h01 * h01;
        if h00 > 0.0 && det > 0.0 {
            let d0 = -(h11 * g0 - h01 * g1) / det;
            let d1 = -(h00 * g1 - h01 * g0) / det;
            if d0.abs() <= 1.0 && d1.abs() <= 1.0 {
                return Ok([refined[0] + d0, refined[1] + d1]);
            }
        }
    }
    for axis in 0..2 {
        let step = |sign: isize| {
            if axis == 0 {
                at(sign, 0)
            } else {
                at(0, sign)
            }
        };
        if let (Some(m), Some(p)) = (step(-1), step(1)) {
            refined[axis] += parabolic_offset(m, centre, p);
        }
    }
    Ok(refined)
}

/// Displacement of plane `z + 1`'s content relative to plane `z`, for each
/// consecutive pair: `nz - 1` entries, `[axis 1, axis 2]` in pixels.
///
/// A displacement `d` means `plane[z + 1](x) ~ plane[z](x - d)`.
pub fn estimate_pairwise_shifts<T>(
    volume: ArrayView3<'_, T>,
    params: &SliceShiftParams,
) -> Result<Vec<[f64; 2]>>
where
    T: Copy + Into<f64>,
{
    let (planes, rows, cols) = volume.dim();
    if planes < 2 {
        return Ok(Vec::new());
    }
    let window = params
        .window
        .map(|w| [w[0].min(rows), w[1].min(cols)])
        .unwrap_or([rows, cols]);
    if window[0] <= params.max_shift[0] || window[1] <= params.max_shift[1] {
        return Err(Error::InvalidArgument(format!(
            "a comparison window of {window:?} cannot be searched over shifts up to {:?}",
            params.max_shift
        )));
    }
    let lags = ShiftWindow::symmetric(params.max_shift);
    let mut plan = SquaredDifference::new(window, window, lags, Padding::Smooth)?;
    let mut landscape = plan.empty_landscape();
    let mut previous = standardised(volume.index_axis(ndarray::Axis(0), 0), window);
    let mut out = Vec::with_capacity(planes - 1);
    for z in 1..planes {
        let current = standardised(volume.index_axis(ndarray::Axis(0), z), window);
        plan.landscape_into(previous.view(), current.view(), &mut landscape)?;
        // The landscape compares `a[x]` with `b[x - k]`; with `b(x) = a(x - d)`
        // the match is at `k = -d`.
        let lag = refined_minimum(&landscape)?;
        out.push([-lag[0], -lag[1]]);
        previous = current;
    }
    Ok(out)
}

/// The running sum of pairwise displacements: `nz` entries, the first zero.
pub fn accumulate_shifts(pairwise: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut path = Vec::with_capacity(pairwise.len() + 1);
    path.push([0.0, 0.0]);
    for step in pairwise {
        let last = *path.last().unwrap();
        path.push([last[0] + step[0], last[1] + step[1]]);
    }
    path
}

/// `path` with `detrend`'s trend subtracted.
pub fn detrend_path(path: &[[f64; 2]], detrend: Detrend) -> Result<Vec<[f64; 2]>> {
    let n = path.len();
    match detrend {
        Detrend::None => Ok(path.to_vec()),
        Detrend::RunningMean { width } => {
            if width == 0 || width % 2 == 0 {
                return Err(Error::InvalidArgument(format!(
                    "a running-mean detrend needs an odd, positive width; got {width}"
                )));
            }
            let half = width / 2;
            Ok((0..n)
                .map(|z| {
                    let window = &path[z.saturating_sub(half)..(z + half + 1).min(n)];
                    let count = window.len() as f64;
                    let mut mean = [0.0; 2];
                    for point in window {
                        mean[0] += point[0] / count;
                        mean[1] += point[1] / count;
                    }
                    [path[z][0] - mean[0], path[z][1] - mean[1]]
                })
                .collect())
        }
        Detrend::Linear => {
            if n < 2 {
                return Ok(vec![[0.0, 0.0]; n]);
            }
            let zs: Vec<f64> = (0..n).map(|z| z as f64).collect();
            let z_mean = zs.iter().sum::<f64>() / n as f64;
            let z_var: f64 = zs.iter().map(|z| (z - z_mean).powi(2)).sum();
            let mut fit = [(0.0, 0.0); 2];
            for (axis, slot) in fit.iter_mut().enumerate() {
                let mean = path.iter().map(|p| p[axis]).sum::<f64>() / n as f64;
                let covariance: f64 = path
                    .iter()
                    .zip(&zs)
                    .map(|(p, z)| (p[axis] - mean) * (z - z_mean))
                    .sum();
                let slope = covariance / z_var;
                *slot = (slope, mean - slope * z_mean);
            }
            Ok(path
                .iter()
                .zip(&zs)
                .map(|(p, z)| {
                    [
                        p[0] - (fit[0].0 * z + fit[0].1),
                        p[1] - (fit[1].0 * z + fit[1].1),
                    ]
                })
                .collect())
        }
    }
}

/// The per-plane shifts to apply: the pairwise estimates, accumulated and
/// detrended. `nz` entries; hand them to [`apply_slice_shifts_into`] or
/// [`SliceShiftOp`].
pub fn estimate_slice_shifts<T>(
    volume: ArrayView3<'_, T>,
    params: &SliceShiftParams,
) -> Result<Vec<[f64; 2]>>
where
    T: Copy + Into<f64>,
{
    if volume.dim().0 == 0 {
        return Ok(Vec::new());
    }
    let pairwise = estimate_pairwise_shifts(volume, params)?;
    detrend_path(&accumulate_shifts(&pairwise), params.detrend)
}

fn validate_shifts(shifts: &[[f64; 2]]) -> Result<()> {
    if shifts.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::InvalidArgument("slice shifts must be finite".into()));
    }
    Ok(())
}

/// Translate each plane of `input` by its shift: `out_z(x) = in_z(x +
/// shifts[z])`, bilinear, `fill` where the source leaves the volume.
///
/// `at` places `input` in the volume (use [`Anchor::whole`] for a whole
/// volume): `shifts` has one entry per plane of `at.volume`, and the source
/// position is tested against the volume's extent, not the buffer's. A buffer
/// that is a block must therefore carry a halo of [`SliceShiftOp`]'s reach.
pub fn apply_slice_shifts_into<T>(
    input: ArrayView3<'_, T>,
    shifts: &[[f64; 2]],
    fill: f64,
    at: &Anchor,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    shapes_agree(input.shape(), out.shape(), "apply_slice_shifts_into")?;
    validate_shifts(shifts)?;
    if shifts.len() != at.volume[0] {
        return Err(Error::InvalidArgument(format!(
            "{} slice shifts for a volume of {} planes",
            shifts.len(),
            at.volume[0]
        )));
    }
    let (planes, rows, cols) = input.dim();
    let extent = [at.volume[1] as f64, at.volume[2] as f64];
    for z in 0..planes {
        let shift = shifts[at.offset[0] + z];
        let source = input.index_axis(ndarray::Axis(0), z);
        let mut target = out.index_axis_mut(ndarray::Axis(0), z);
        for i in 0..rows {
            let y = (at.offset[1] + i) as f64 + shift[0];
            for j in 0..cols {
                let x = (at.offset[2] + j) as f64 + shift[1];
                target[[i, j]] = if y < 0.0 || x < 0.0 || y > extent[0] - 1.0 || x > extent[1] - 1.0
                {
                    fill
                } else {
                    // A halo voxel whose source is beyond the buffer is not
                    // trusted by the caller; it gets the fill rather than a panic.
                    bilinear(source, [y, x], [at.offset[1], at.offset[2]]).unwrap_or(fill)
                };
            }
        }
    }
    Ok(())
}

/// Bilinear sample at a global in-plane position of a buffer whose origin is
/// at `origin`, or `None` if a tap of non-zero weight is outside the buffer
/// (which happens only in a block's halo, never in its core). A tap of weight
/// zero is not read.
fn bilinear<T: Copy + Into<f64>>(
    plane: ArrayView2<'_, T>,
    at: [f64; 2],
    origin: [usize; 2],
) -> Option<f64> {
    let y0 = at[0].floor();
    let x0 = at[1].floor();
    let (ty, tx) = (at[0] - y0, at[1] - x0);
    let (rows, cols) = plane.dim();
    let i = (y0 as usize).checked_sub(origin[0])?;
    let j = (x0 as usize).checked_sub(origin[1])?;
    let last_i = i + usize::from(ty != 0.0);
    let last_j = j + usize::from(tx != 0.0);
    if last_i >= rows || last_j >= cols {
        return None;
    }
    let sample = |di: usize, dj: usize| -> f64 { plane[[i + di, j + dj]].into() };
    let row = |di: usize| -> f64 {
        if tx == 0.0 {
            sample(di, 0)
        } else {
            (1.0 - tx) * sample(di, 0) + tx * sample(di, 1)
        }
    };
    Some(if ty == 0.0 {
        row(0)
    } else {
        (1.0 - ty) * row(0) + ty * row(1)
    })
}

/// [`apply_slice_shifts_into`] as a [`BlockOp`]. The shift table is a parameter
/// with one entry per plane of the volume the op runs on.
pub struct SliceShiftOp {
    name: &'static str,
    shifts: Vec<[f64; 2]>,
    fill: f64,
    reach: [usize; 2],
    cost: f64,
}

impl SliceShiftOp {
    pub fn new(name: &'static str, shifts: Vec<[f64; 2]>, fill: f64) -> Result<Self> {
        validate_shifts(&shifts)?;
        let mut reach = [0usize; 2];
        for shift in &shifts {
            for axis in 0..2 {
                let magnitude = shift[axis].abs();
                // `floor` is the furthest whole tap; a fractional part reaches one
                // more. An integral shift reads no second tap.
                let taps = magnitude.floor() as usize + usize::from(magnitude.fract() > 0.0);
                reach[axis] = reach[axis].max(taps);
            }
        }
        Ok(Self {
            name,
            shifts,
            fill,
            reach,
            cost: SLICE_SHIFT_COST,
        })
    }

    pub fn shifts(&self) -> &[[f64; 2]] {
        &self.shifts
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for SliceShiftOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Nothing on z; in-plane, the furthest bilinear tap over every plane's
    /// shift: `ceil(max |shift|)` on that axis.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        match axis {
            0 => 0,
            axis => self.reach[axis - 1],
        }
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let out = out.view_mut::<f64>()?;
        dispatch_f64_input!(
            input,
            Error::InvalidArgument(format!("{}: no buffer holds half-precision", self.name)),
            |view| { apply_slice_shifts_into(view, &self.shifts, self.fill, at, out) }
        )
    }

    /// Not declared: near the volume's faces a constant becomes the fill.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        None
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Estimated, not measured: four taps and a few comparisons per voxel.
const SLICE_SHIFT_COST: f64 = 4.0;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A band-limited texture that can be evaluated at any real position, so a
    /// sub-pixel shift is exact rather than interpolated.
    fn texture(seed: u64) -> impl Fn(f64, f64) -> f64 {
        let mut rng = Rng(seed);
        let waves: Vec<[f64; 4]> = (0..30)
            .map(|_| {
                let angle = rng.next() * std::f64::consts::PI;
                let frequency = 0.02 + 0.08 * rng.next();
                [
                    frequency * angle.cos(),
                    frequency * angle.sin(),
                    rng.next() * std::f64::consts::TAU,
                    0.5 + rng.next(),
                ]
            })
            .collect();
        move |y, x| {
            waves
                .iter()
                .map(|w| w[3] * (std::f64::consts::TAU * (w[0] * y + w[1] * x) + w[2]).sin())
                .sum()
        }
    }

    /// Planes of one texture, plane `z` displaced by `truth[z]`, with a
    /// per-plane brightness change the estimate must ignore.
    fn shifted_stack(dim: (usize, usize, usize), truth: &[[f64; 2]]) -> Array3<f64> {
        let field = texture(17);
        Array3::from_shape_fn(dim, |(z, i, j)| {
            let gain = 1.0 + 0.1 * z as f64;
            100.0 + gain * field(i as f64 - truth[z][0], j as f64 - truth[z][1])
        })
    }

    fn random_truth(planes: usize) -> Vec<[f64; 2]> {
        let mut rng = Rng(5);
        (0..planes)
            .map(|_| [6.0 * (rng.next() - 0.5), 6.0 * (rng.next() - 0.5)])
            .collect()
    }

    #[test]
    fn the_parabola_finds_the_vertex_of_a_quadratic() {
        let offset = 0.3;
        let q = |x: f64| (x - offset).powi(2) + 2.0;
        assert!((parabolic_offset(q(-1.0), q(0.0), q(1.0)) - offset).abs() < 1e-12);
        assert_eq!(parabolic_offset(1.0, 1.0, 1.0), 0.0);
    }

    /// Random sub-pixel offsets per plane are recovered, relative to plane 0, to
    /// better than 0.2 px on every plane.
    #[test]
    fn random_subpixel_shifts_are_recovered() {
        let dim = (10usize, 96usize, 112usize);
        let truth = random_truth(dim.0);
        let stack = shifted_stack(dim, &truth);
        let params = SliceShiftParams::new([6, 6], Some([72, 80]), Detrend::None).unwrap();
        let path = estimate_slice_shifts(stack.view(), &params).unwrap();
        assert_eq!(path.len(), dim.0);
        let mut worst = 0.0f64;
        for z in 0..dim.0 {
            for axis in 0..2 {
                let want = truth[z][axis] - truth[0][axis];
                worst = worst.max((path[z][axis] - want).abs());
            }
        }
        assert!(worst < 0.2, "worst error {worst} px");

        // and applying them registers every plane onto plane 0
        let mut out = Array3::zeros(dim);
        apply_slice_shifts_into(
            stack.view(),
            &path,
            f64::NAN,
            &Anchor::whole([10, 96, 112]),
            out.view_mut(),
        )
        .unwrap();
        let field = texture(17);
        let mut residual = 0.0f64;
        for z in 0..dim.0 {
            let gain = 1.0 + 0.1 * z as f64;
            for i in 10..86 {
                for j in 10..102 {
                    let want = 100.0 + gain * field(i as f64 - truth[0][0], j as f64 - truth[0][1]);
                    residual = residual.max((out[[z, i, j]] - want).abs() / gain);
                }
            }
        }
        assert!(residual < 1.0, "worst registered residual {residual}");
    }

    #[test]
    fn detrending_keeps_tilt_and_removes_jitter() {
        let jitter = [0.4, -0.3, 0.2, -0.5, 0.1, 0.3, -0.2, 0.0, 0.25];
        let path: Vec<[f64; 2]> = jitter
            .iter()
            .enumerate()
            .map(|(z, &j)| [0.7 * z as f64 + j, -0.2 * z as f64])
            .collect();
        let linear = detrend_path(&path, Detrend::Linear).unwrap();
        for (point, _) in linear.iter().zip(&jitter) {
            assert!(point[1].abs() < 1e-12, "a pure tilt is entirely trend");
            assert!(point[0].abs() < 1.0);
        }
        // the residual is the jitter minus its own best line, so its spread is
        // the jitter's and not the tilt's
        let spread = linear.iter().map(|p| p[0] * p[0]).sum::<f64>().sqrt();
        assert!(spread < 1.0, "{spread}");
        let running = detrend_path(&path, Detrend::RunningMean { width: 3 }).unwrap();
        assert!(
            running[4][1].abs() < 1e-12,
            "an interior linear trend is its own mean"
        );
        assert_eq!(detrend_path(&path, Detrend::None).unwrap(), path);
        assert!(SliceShiftParams::new([4, 4], None, Detrend::RunningMean { width: 4 }).is_err());
        assert!(SliceShiftParams::new([8, 8], Some([8, 20]), Detrend::None).is_err());
    }

    /// The op on a block with a halo reproduces the whole-volume answer in its
    /// core; out-of-volume sources are the fill.
    #[test]
    fn the_op_is_cut_invariant_and_fills_outside_the_volume() {
        let dim = (4usize, 30usize, 34usize);
        let input = Array3::from_shape_fn(dim, |(z, i, j)| (z * 1000 + i * 34 + j) as f64);
        let shifts = vec![[0.0, 0.0], [1.5, -2.25], [-3.0, 0.5], [0.75, 2.0]];
        let op = SliceShiftOp::new("shift", shifts, -1.0).unwrap();
        assert_eq!(
            [op.reach(0, 4), op.reach(1, 30), op.reach(2, 34)],
            [0, 3, 3]
        );
        let volume = [4, 30, 34];
        let whole_in: Voxels = input.clone().into();
        let mut whole = Voxels::zeros(Dtype::F64, volume).unwrap();
        op.apply(&whole_in, &mut whole, &Anchor::whole(volume))
            .unwrap();
        let whole = whole.view::<f64>().unwrap().to_owned();
        // linear ramp: a shift of (1.5, -2.25) adds 1.5*34 - 2.25 inside
        assert_eq!(whole[[1, 10, 10]], input[[1, 10, 10]] + 1.5 * 34.0 - 2.25);
        assert_eq!(whole[[2, 0, 5]], -1.0, "source row -3 is outside");

        // core [1..4, 8..20, 10..25] with a halo of the reach, clipped
        let (core0, core1) = ([1, 8, 10], [4, 20, 25]);
        let read0 = [core0[0], core0[1] - 3, core0[2] - 3];
        let read1 = [core1[0], core1[1] + 3, core1[2] + 3];
        let block: Voxels = input
            .slice(s![
                read0[0]..read1[0],
                read0[1]..read1[1],
                read0[2]..read1[2]
            ])
            .to_owned()
            .into();
        let shape = [
            read1[0] - read0[0],
            read1[1] - read0[1],
            read1[2] - read0[2],
        ];
        let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
        op.apply(&block, &mut out, &Anchor::new(read0, volume))
            .unwrap();
        let out = out.view::<f64>().unwrap();
        assert_eq!(
            out.slice(s![.., 3..15, 3..18]),
            whole.slice(s![1..4, 8..20, 10..25])
        );
    }
}
