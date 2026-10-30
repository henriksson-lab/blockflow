// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the image-formation model of
// differential interference contrast and the cited papers, not adapted from any
// implementation.

//! **Differential interference contrast (DIC): normalisation, phase
//! reconstruction, a smoothness map, and shear-direction estimation.**
//!
//! # The model, and the one angle every function here takes
//!
//! A DIC image is, to first order, a constant bias plus a multiple of the
//! derivative of the specimen's optical path (phase) `φ` along the shear
//! direction `s`:
//!
//! ```text
//!     I(x) ≈ I0 + κ · ∂φ/∂s (x),      κ > 0
//! ```
//!
//! The direction is given as **one angle `theta` in radians, in image
//! coordinates**: `x` is the column (to the right), `y` the row (downwards), and
//! angles turn from `+x` towards `+y`. `theta` is the **bright-edge angle**: an
//! object that is optically denser than its surroundings appears bright on the
//! edges whose outward normal points along `theta`, and dark on the opposite
//! side. It is the `θ` of the eggshell fit `I_edge(φ) = a + b·cos(φ − θ)`
//! (`super::eggshell`) and the signed estimate of [`GradientMoments`], so a
//! calibration result can be handed straight to [`DicIntegration`]. The phase
//! derivative direction of the model above is `s = theta + π` (the phase of a
//! dense object rises where one enters it, so the entering edge — facing `-s` —
//! is the bright one).
//!
//! # What is here
//!
//! | piece | what |
//! |---|---|
//! | [`dic_normalise_plane`], [`DicNormaliseOp`] | `(I − B)/scale`: a background `B` (Gaussian of a large sigma, or the plane median) removed and a per-movie relief scale divided out |
//! | [`dic_relief_scale`] | that scale: 1.4826 × MAD of `I − B`, optionally inside a mask (the embryo) |
//! | [`dic_integrate_plane`], [`DicIntegrateOp`] | phase-like reconstruction: Tikhonov-regularised inverse of the directional derivative, in the Fourier domain |
//! | [`dic_texture_volume`], [`DicTextureOp`] | smoothness `−ln E`, with `E` the local energy of a band-passed image — plain, or the *isotropic* part (twice the smaller structure-tensor eigenvalue), which ignores straight edges such as the shear relief along a nucleus border |
//! | [`centre_surround_volume`], [`CentreSurroundOp`] | nucleus-likeness: a disc-minus-surround of the smoothness map at one or several radii, maximum over the radii |
//! | [`GradientMoments`] | a shell-free estimate of the shear: orientation from the second moments of the image gradient, sign from the third |
//! | [`fit_cosine`] | robust fit of `a + b·cos(φ − θ)` (and optionally a `cos 2φ` term) |
//!
//! # Phase reconstruction: which method
//!
//! The reconstruction minimises `‖∂ψ/∂s − d‖² + λ‖ψ‖²` with `d = (I − I0)/κ`,
//! whose solution in the Fourier domain is
//!
//! ```text
//!     Ψ(k) = −i k_s D(k) / (k_s² + λ),     k_s = k · s,  λ = 1/ℓ²
//! ```
//!
//! i.e. the Fourier-space division by `i k_s` of Arnison et al. (2004), *Linear
//! phase imaging using differential interference contrast microscopy*, J.
//! Microsc. 214:7–12, with a Tikhonov term in place of their Wiener constant. In
//! the spatial domain it is the line integration along the shear of Kam (1998),
//! *Microscopic differential interference contrast image processing by line
//! integration (LID) and deconvolution*, Bioimaging 6:166–176, with an
//! exponential leak: the kernel is `½·sign(t)·exp(−|t|/ℓ)` along `s`. `ℓ`
//! (`leak_length`) is the longest structure that is integrated faithfully;
//! longer ones (and the streaks a pure integral leaves perpendicular to the
//! shear) are suppressed. The plane is zero-padded by `3ℓ` on every side, which
//! is the physically right extension once the background is removed (the
//! derivative of a uniform medium is zero).
//!
//! # Smoothness: why the isotropic part
//!
//! In DIC a nucleus is a smooth, low-texture region among granular cytoplasm,
//! but its border carries the shear relief — a bright and a dark crescent —
//! which has as much band-passed energy as the granules around it. That relief
//! is a straight-ish edge: at the integration scale its gradients all point one
//! way, so the structure tensor has one large and one small eigenvalue, while
//! granules give two comparable ones. `2·λ_min` keeps the granules and drops the
//! edge. ([`TextureMeasure::Energy`] keeps both; it is there to compare.)
//!
//! All of the per-plane work is 2-D: DIC is a 2-D measurement of each focal
//! plane. The texture energy can additionally be smoothed across planes
//! (`z_sigma`), which is the cheap 3-D-aware step: a nucleus is a smooth disc
//! over several planes.

use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut3, Zip};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Slicing};
use crate::reach::{AxisReach, Reach};
use crate::voxels::Voxels;

use super::fft::{next_smooth_length, Complex, RealTransform2};
use super::ridge::{gaussian_radius, gaussian_weights};
use super::threshold::percentile_of;

const TRUNCATE: f64 = 3.0;

// ---------------------------------------------------------------------------
// separable plane filters (clamped edges)
// ---------------------------------------------------------------------------

/// Correlate every row (axis 1) of `input` with `kernel` (odd length, centred),
/// clamping at the plane's edge.
fn filter_rows(input: ArrayView2<'_, f64>, kernel: &[f64]) -> Array2<f64> {
    let (rows, cols) = input.dim();
    let radius = (kernel.len() / 2) as isize;
    let mut out = Array2::<f64>::zeros((rows, cols));
    if kernel.len() == 1 {
        out.assign(&input);
        out.mapv_inplace(|v| v * kernel[0]);
        return out;
    }
    let mut line = vec![0.0; cols + 2 * radius as usize];
    for r in 0..rows {
        let row = input.row(r);
        for (i, slot) in line.iter_mut().enumerate() {
            let c = (i as isize - radius).clamp(0, cols as isize - 1) as usize;
            *slot = row[c];
        }
        let mut target = out.row_mut(r);
        for c in 0..cols {
            let mut acc = 0.0;
            for (k, &w) in kernel.iter().enumerate() {
                acc += w * line[c + k];
            }
            target[c] = acc;
        }
    }
    out
}

/// The same along columns (axis 0).
fn filter_cols(input: ArrayView2<'_, f64>, kernel: &[f64]) -> Array2<f64> {
    filter_rows(input.t(), kernel).reversed_axes()
}

fn separable(input: ArrayView2<'_, f64>, ky: &[f64], kx: &[f64]) -> Array2<f64> {
    let rows = filter_rows(input, kx);
    filter_cols(rows.view(), ky)
}

/// A Gaussian of `sigma` pixels in-plane (zero: identity).
pub fn gaussian_plane(input: ArrayView2<'_, f64>, sigma: f64) -> Result<Array2<f64>> {
    let k = gaussian_weights(sigma, TRUNCATE)?;
    Ok(separable(input, &k, &k))
}

/// The derivative-of-Gaussian taps at `sigma` (`sigma > 0`), correlation order:
/// `out[x] = Σ k[j] · in[x + j − r]` estimates `d/dx` of the smoothed input.
fn gaussian_derivative_weights(sigma: f64) -> Result<Vec<f64>> {
    if !(sigma > 0.0) || !sigma.is_finite() {
        return Err(Error::invalid(format!(
            "a derivative scale must be positive and finite; got {sigma}"
        )));
    }
    let g = gaussian_weights(sigma, TRUNCATE)?;
    let r = (g.len() / 2) as isize;
    Ok(g.iter()
        .enumerate()
        .map(|(j, &w)| {
            let x = (j as isize - r) as f64;
            w * x / (sigma * sigma)
        })
        .collect())
}

/// Gaussian gradient `(d/dy, d/dx)` of a plane at `sigma` pixels.
pub fn gaussian_gradient_plane(
    input: ArrayView2<'_, f64>,
    sigma: f64,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let g = gaussian_weights(sigma, TRUNCATE)?;
    let d = gaussian_derivative_weights(sigma)?;
    Ok((separable(input, &d, &g), separable(input, &g, &d)))
}

fn widen_plane<T: Copy + Into<f64>>(plane: ArrayView2<'_, T>) -> Array2<f64> {
    plane.mapv(Into::into)
}

// ---------------------------------------------------------------------------
// normalisation
// ---------------------------------------------------------------------------

/// What [`dic_normalise_plane`] removes and divides by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DicNormalise {
    /// In-plane sigma of the background, in pixels. `0` uses the plane's median
    /// (one constant per plane), which keeps every scale of relief — what the
    /// phase reconstruction wants — at the price of a whole-plane reach.
    pub background_sigma: f64,
    /// The relief scale divided out, typically [`dic_relief_scale`] of the
    /// movie (or the shell relief strength `b`). Must be positive.
    pub scale: f64,
}

impl DicNormalise {
    pub fn new(background_sigma: f64, scale: f64) -> Result<Self> {
        if !(background_sigma >= 0.0) || !background_sigma.is_finite() {
            return Err(Error::invalid(format!(
                "a DIC background sigma must be non-negative and finite; got {background_sigma}"
            )));
        }
        if !(scale > 0.0) || !scale.is_finite() {
            return Err(Error::invalid(format!(
                "a DIC relief scale must be positive and finite; got {scale}"
            )));
        }
        Ok(Self {
            background_sigma,
            scale,
        })
    }
}

fn background_plane(plane: ArrayView2<'_, f64>, sigma: f64) -> Result<Array2<f64>> {
    if sigma == 0.0 {
        let mut values: Vec<f64> = plane.iter().copied().filter(|v| v.is_finite()).collect();
        let median = if values.is_empty() {
            0.0
        } else {
            percentile_of(&mut values, 50.0)
        };
        Ok(Array2::from_elem(plane.dim(), median))
    } else {
        gaussian_plane(plane, sigma)
    }
}

/// `(I − B) / scale` of one plane.
pub fn dic_normalise_plane<T: Copy + Into<f64>>(
    plane: ArrayView2<'_, T>,
    params: &DicNormalise,
) -> Result<Array2<f64>> {
    let plane = widen_plane(plane);
    let background = background_plane(plane.view(), params.background_sigma)?;
    let inv = 1.0 / params.scale;
    Ok(Zip::from(&plane)
        .and(&background)
        .map_collect(|&v, &b| (v - b) * inv))
}

/// [`dic_normalise_plane`] over every plane of `volume`, into `out`.
pub fn dic_normalise_into<T: Copy + Into<f64>>(
    volume: ArrayView3<'_, T>,
    params: &DicNormalise,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    super::shapes_agree(volume.shape(), out.shape(), "dic_normalise_into")?;
    for (plane, mut target) in volume.outer_iter().zip(out.outer_iter_mut()) {
        target.assign(&dic_normalise_plane(plane, params)?);
    }
    Ok(())
}

/// The robust relief scale of a movie: `1.4826 · MAD(I − B)` over the voxels
/// where `mask` is true (all, without a mask), sampled every `stride` pixels.
pub fn dic_relief_scale<T: Copy + Into<f64>>(
    volume: ArrayView3<'_, T>,
    mask: Option<ArrayView3<'_, bool>>,
    background_sigma: f64,
    stride: usize,
) -> Result<f64> {
    if let Some(mask) = mask.as_ref() {
        super::shapes_agree(volume.shape(), mask.shape(), "dic_relief_scale (mask)")?;
    }
    let stride = stride.max(1);
    let mut values = Vec::new();
    for (z, plane) in volume.outer_iter().enumerate() {
        let plane = widen_plane(plane);
        let background = background_plane(plane.view(), background_sigma)?;
        for i in (0..plane.nrows()).step_by(stride) {
            for j in (0..plane.ncols()).step_by(stride) {
                if mask.as_ref().is_some_and(|m| !m[[z, i, j]]) {
                    continue;
                }
                let v = plane[[i, j]] - background[[i, j]];
                if v.is_finite() {
                    values.push(v);
                }
            }
        }
    }
    if values.is_empty() {
        return Err(Error::invalid(
            "no voxel was counted for the DIC relief scale (empty mask?)",
        ));
    }
    let median = percentile_of(&mut values, 50.0);
    for v in &mut values {
        *v = (*v - median).abs();
    }
    let mad = percentile_of(&mut values, 50.0);
    if !(mad > 0.0) {
        return Err(Error::invalid(
            "the DIC relief has no spread (constant image?)",
        ));
    }
    Ok(1.4826 * mad)
}

/// [`dic_normalise_plane`] as a [`BlockOp`]. Reach: the background Gaussian's
/// radius in-plane and nothing in `z`; with a plane-median background the whole
/// plane (a planning barrier in `y` and `x`).
pub struct DicNormaliseOp {
    name: &'static str,
    params: DicNormalise,
}

impl DicNormaliseOp {
    pub fn new(name: &'static str, params: DicNormalise) -> Self {
        Self { name, params }
    }
}

impl BlockOp for DicNormaliseOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        match axis {
            0 => 0,
            _ if self.params.background_sigma == 0.0 => volume_len,
            _ => gaussian_radius(self.params.background_sigma, TRUNCATE),
        }
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        if self.params.background_sigma == 0.0 {
            Reach::per_axis([AxisReach::none(), AxisReach::All, AxisReach::All])
        } else {
            let r = gaussian_radius(self.params.background_sigma, TRUNCATE);
            Reach::symmetric([0, r, r])
        }
    }

    fn slicing(&self) -> Slicing {
        Slicing::UNDECLARED
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let widened = input.widened();
        let out = out.view_mut::<f64>()?;
        dic_normalise_into(widened.view(), &self.params, out)
    }

    fn cost_per_voxel(&self) -> f64 {
        2.0 + 1.4 * 2.0 * (2 * gaussian_radius(self.params.background_sigma, TRUNCATE) + 1) as f64
    }
}

// ---------------------------------------------------------------------------
// phase reconstruction
// ---------------------------------------------------------------------------

/// Parameters of [`dic_integrate_plane`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DicIntegration {
    /// The bright-edge angle (see the module header), radians.
    pub theta: f64,
    /// `ℓ`, pixels: structures longer than this along the shear are suppressed.
    pub leak_length: f64,
}

impl DicIntegration {
    pub fn new(theta: f64, leak_length: f64) -> Result<Self> {
        if !theta.is_finite() {
            return Err(Error::invalid("a shear angle must be finite"));
        }
        if !(leak_length > 0.0) || !leak_length.is_finite() || leak_length > 1e5 {
            return Err(Error::invalid(format!(
                "a leak length must be positive, finite and below 1e5 pixels; got {leak_length}"
            )));
        }
        Ok(Self { theta, leak_length })
    }

    /// The zero padding on each side, pixels: `3ℓ`, where the kernel has fallen
    /// to `e^-3` of its peak.
    pub fn padding(&self) -> usize {
        (3.0 * self.leak_length).ceil() as usize
    }
}

/// Phase-like reconstruction of one **background-free** plane (e.g. the output
/// of [`dic_normalise_plane`]): `Ψ = −i k_s D / (k_s² + 1/ℓ²)`, `s = theta + π`.
/// The result is in units of the input times pixels.
pub fn dic_integrate_plane(
    plane: ArrayView2<'_, f64>,
    params: &DicIntegration,
) -> Result<Array2<f64>> {
    let (rows, cols) = plane.dim();
    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }
    let pad = params.padding();
    let shape = [
        next_smooth_length(rows + 2 * pad),
        next_smooth_length(cols + 2 * pad),
    ];
    // Place the plane at (pad, pad) of a zero plane; the transform's own
    // zero-padding puts its input at the origin, so build the padded plane here.
    let mut padded = Array2::<f64>::zeros((shape[0], shape[1]));
    Zip::from(padded.slice_mut(s![pad..pad + rows, pad..pad + cols]))
        .and(&plane)
        .for_each(|slot, &v| *slot = if v.is_finite() { v } else { 0.0 });
    let mut transform = RealTransform2::new(shape)?;
    let mut spectrum = transform.spectrum();
    transform.forward(padded.view(), &mut spectrum)?;
    let shear = params.theta + std::f64::consts::PI;
    let (cs, sn) = (shear.cos(), shear.sin());
    let lambda = 1.0 / (params.leak_length * params.leak_length);
    let two_pi = 2.0 * std::f64::consts::PI;
    let (srows, scols) = spectrum.dim();
    for r in 0..srows {
        let fy = if r <= shape[0] / 2 {
            r as f64
        } else {
            r as f64 - shape[0] as f64
        } / shape[0] as f64;
        for c in 0..scols {
            let fx = c as f64 / shape[1] as f64;
            let ks = two_pi * (fx * cs + fy * sn);
            // −i ks / (ks² + λ)
            let g = Complex::new(0.0, -ks / (ks * ks + lambda));
            spectrum[[r, c]] *= g;
        }
    }
    let mut full = Array2::<f64>::zeros((shape[0], shape[1]));
    transform.inverse(&mut spectrum, &mut full)?;
    Ok(full.slice(s![pad..pad + rows, pad..pad + cols]).to_owned())
}

/// [`dic_integrate_plane`] over every plane of `volume`.
pub fn dic_integrate_into(
    volume: ArrayView3<'_, f64>,
    params: &DicIntegration,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    super::shapes_agree(volume.shape(), out.shape(), "dic_integrate_into")?;
    for (plane, mut target) in volume.outer_iter().zip(out.outer_iter_mut()) {
        target.assign(&dic_integrate_plane(plane, params)?);
    }
    Ok(())
}

/// [`dic_integrate_plane`] as a [`BlockOp`]: a Fourier transform of each whole
/// plane, so the reach is every `y` and `x` (a planning barrier in-plane) and
/// none in `z`. Run it on a grid whose blocks are whole planes; a DIC plane is
/// at most a few megapixels.
pub struct DicIntegrateOp {
    name: &'static str,
    params: DicIntegration,
}

impl DicIntegrateOp {
    pub fn new(name: &'static str, params: DicIntegration) -> Self {
        Self { name, params }
    }
}

impl BlockOp for DicIntegrateOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        if axis == 0 {
            0
        } else {
            volume_len
        }
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::per_axis([AxisReach::none(), AxisReach::All, AxisReach::All])
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(dtype, Dtype::F64 | Dtype::F32)
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let widened = input.widened();
        let out = out.view_mut::<f64>()?;
        dic_integrate_into(widened.view(), &self.params, out)
    }

    /// Linear: zero maps to zero.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (value == 0.0).then_some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        // Two transforms of a plane padded by 3ℓ per side, ~5 log2(N) flops each.
        60.0
    }
}

// ---------------------------------------------------------------------------
// texture / smoothness
// ---------------------------------------------------------------------------

/// Which local texture energy [`dic_texture_volume`] measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureMeasure {
    /// `G_w * b²`, `b` the band-passed plane: every structure in the band.
    Energy,
    /// `2·λ_min` of the structure tensor of `b` (derivative scale `band.0`,
    /// integration scale `window`): texture without the straight edges.
    Isotropic,
}

/// Parameters of [`dic_texture_volume`]. All lengths in pixels (`z_sigma` in
/// planes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DicTexture {
    /// Difference of Gaussians `(inner, outer)`: the granule band. `inner` may
    /// be 0 (no smoothing).
    pub band: (f64, f64),
    /// Integration (energy averaging) sigma.
    pub window: f64,
    pub measure: TextureMeasure,
    /// Smoothing of the energy across planes; 0 keeps planes independent.
    pub z_sigma: f64,
    /// Added to the energy before the logarithm, in the energy's units.
    pub floor: f64,
}

impl DicTexture {
    pub fn new(
        band: (f64, f64),
        window: f64,
        measure: TextureMeasure,
        z_sigma: f64,
        floor: f64,
    ) -> Result<Self> {
        let ok = |v: f64| v >= 0.0 && v.is_finite();
        if !ok(band.0) || !(band.1 > band.0) || !band.1.is_finite() {
            return Err(Error::invalid(format!(
                "a texture band needs 0 <= inner < outer, both finite; got {band:?}"
            )));
        }
        if !(window > 0.0) || !window.is_finite() || !ok(z_sigma) {
            return Err(Error::invalid(format!(
                "a texture window must be positive and the z sigma non-negative; got {window}, {z_sigma}"
            )));
        }
        if !(floor > 0.0) || !floor.is_finite() {
            return Err(Error::invalid("the energy floor must be positive"));
        }
        Ok(Self {
            band,
            window,
            measure,
            z_sigma,
            floor,
        })
    }

    /// In-plane reach of the whole computation, pixels.
    pub fn reach_in_plane(&self) -> usize {
        let derivative = match self.measure {
            TextureMeasure::Energy => 0,
            TextureMeasure::Isotropic => gaussian_radius(self.band.0.max(0.7), TRUNCATE),
        };
        gaussian_radius(self.band.1, TRUNCATE) + derivative + gaussian_radius(self.window, TRUNCATE)
    }

    pub fn reach_z(&self) -> usize {
        gaussian_radius(self.z_sigma, TRUNCATE)
    }
}

/// The band-passed plane `G_inner * I − G_outer * I`.
pub fn band_pass_plane(plane: ArrayView2<'_, f64>, band: (f64, f64)) -> Result<Array2<f64>> {
    let inner = gaussian_plane(plane, band.0)?;
    let outer = gaussian_plane(plane, band.1)?;
    Ok(inner - outer)
}

/// The local texture energy of one plane (before the logarithm).
pub fn texture_energy_plane(
    plane: ArrayView2<'_, f64>,
    params: &DicTexture,
) -> Result<Array2<f64>> {
    let band = band_pass_plane(plane, params.band)?;
    match params.measure {
        TextureMeasure::Energy => {
            let squared = band.mapv(|v| v * v);
            gaussian_plane(squared.view(), params.window)
        }
        TextureMeasure::Isotropic => {
            let (gy, gx) = gaussian_gradient_plane(band.view(), params.band.0.max(0.7))?;
            let jxx = gaussian_plane((&gx * &gx).view(), params.window)?;
            let jyy = gaussian_plane((&gy * &gy).view(), params.window)?;
            let jxy = gaussian_plane((&gx * &gy).view(), params.window)?;
            Ok(Zip::from(&jxx)
                .and(&jyy)
                .and(&jxy)
                .map_collect(|&a, &b, &c| {
                    let half = 0.5 * (a + b);
                    let disc = (0.25 * (a - b) * (a - b) + c * c).sqrt();
                    (2.0 * (half - disc)).max(0.0)
                }))
        }
    }
}

/// Smoothness `−ln(E + floor)` of every plane of `volume`, with `E` optionally
/// smoothed across planes. Larger is smoother.
pub fn dic_texture_volume<T: Copy + Into<f64>>(
    volume: ArrayView3<'_, T>,
    params: &DicTexture,
) -> Result<Array3<f64>> {
    let (nz, ny, nx) = volume.dim();
    let mut energy = Array3::<f64>::zeros((nz, ny, nx));
    for (plane, mut target) in volume.outer_iter().zip(energy.outer_iter_mut()) {
        let plane = widen_plane(plane);
        target.assign(&texture_energy_plane(plane.view(), params)?);
    }
    if params.z_sigma > 0.0 && nz > 1 {
        let k = gaussian_weights(params.z_sigma, TRUNCATE)?;
        let r = (k.len() / 2) as isize;
        let source = energy.clone();
        for z in 0..nz as isize {
            let mut target = energy.index_axis_mut(ndarray::Axis(0), z as usize);
            target.fill(0.0);
            for (j, &w) in k.iter().enumerate() {
                let zz = (z + j as isize - r).clamp(0, nz as isize - 1) as usize;
                target.scaled_add(w, &source.index_axis(ndarray::Axis(0), zz));
            }
        }
    }
    let floor = params.floor;
    energy.mapv_inplace(|e| -(e + floor).ln());
    Ok(energy)
}

/// [`dic_texture_volume`] as a [`BlockOp`].
pub struct DicTextureOp {
    name: &'static str,
    params: DicTexture,
}

impl DicTextureOp {
    pub fn new(name: &'static str, params: DicTexture) -> Self {
        Self { name, params }
    }
}

impl BlockOp for DicTextureOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        if axis == 0 {
            self.params.reach_z()
        } else {
            self.params.reach_in_plane()
        }
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let widened = input.widened();
        let result = dic_texture_volume(widened.view(), &self.params)?;
        out.view_mut::<f64>()?.assign(&result);
        Ok(())
    }

    fn cost_per_voxel(&self) -> f64 {
        1.4 * 2.0 * (4 * self.params.reach_in_plane() + 4) as f64
    }
}

/// Parameters of [`centre_surround_volume`]: per radius `r` (pixels), the
/// Gaussian disc mean at `r/√2` minus the surround mean at `ratio·r/√2`.
#[derive(Debug, Clone, PartialEq)]
pub struct CentreSurround {
    pub radii: Vec<f64>,
    pub ratio: f64,
}

impl CentreSurround {
    pub fn new(radii: Vec<f64>, ratio: f64) -> Result<Self> {
        if radii.is_empty() || radii.iter().any(|&r| !(r > 0.0) || !r.is_finite()) {
            return Err(Error::invalid(
                "centre-surround needs at least one positive, finite radius",
            ));
        }
        if !(ratio > 1.0) || !ratio.is_finite() {
            return Err(Error::invalid("the surround ratio must exceed one"));
        }
        Ok(Self { radii, ratio })
    }

    pub fn reach_in_plane(&self) -> usize {
        self.radii
            .iter()
            .map(|&r| gaussian_radius(self.ratio * r / std::f64::consts::SQRT_2, TRUNCATE))
            .max()
            .unwrap_or(0)
    }
}

/// Nucleus-likeness of one plane of a smoothness map: `max_r (G_{r/√2} − G_{ratio·r/√2}) * S`.
pub fn centre_surround_plane(
    plane: ArrayView2<'_, f64>,
    params: &CentreSurround,
) -> Result<Array2<f64>> {
    let mut best: Option<Array2<f64>> = None;
    for &r in &params.radii {
        let inner = gaussian_plane(plane, r / std::f64::consts::SQRT_2)?;
        let outer = gaussian_plane(plane, params.ratio * r / std::f64::consts::SQRT_2)?;
        let response = inner - outer;
        best = Some(match best {
            None => response,
            Some(mut b) => {
                Zip::from(&mut b)
                    .and(&response)
                    .for_each(|b, &v| *b = b.max(v));
                b
            }
        });
    }
    Ok(best.expect("at least one radius"))
}

/// [`centre_surround_plane`] over every plane.
pub fn centre_surround_volume(
    volume: ArrayView3<'_, f64>,
    params: &CentreSurround,
) -> Result<Array3<f64>> {
    let mut out = Array3::<f64>::zeros(volume.raw_dim());
    for (plane, mut target) in volume.outer_iter().zip(out.outer_iter_mut()) {
        target.assign(&centre_surround_plane(plane, params)?);
    }
    Ok(out)
}

/// [`centre_surround_volume`] as a [`BlockOp`].
pub struct CentreSurroundOp {
    name: &'static str,
    params: CentreSurround,
}

impl CentreSurroundOp {
    pub fn new(name: &'static str, params: CentreSurround) -> Self {
        Self { name, params }
    }
}

impl BlockOp for CentreSurroundOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        if axis == 0 {
            0
        } else {
            self.params.reach_in_plane()
        }
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(dtype, Dtype::F64 | Dtype::F32)
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let widened = input.widened();
        let result = centre_surround_volume(widened.view(), &self.params)?;
        out.view_mut::<f64>()?.assign(&result);
        Ok(())
    }

    /// A difference of two normalised means: a constant maps to zero exactly
    /// only when it is zero.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (value == 0.0).then_some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        1.4 * 4.0 * (2 * self.params.reach_in_plane() + 1) as f64 * self.params.radii.len() as f64
    }
}

// ---------------------------------------------------------------------------
// shear from gradient statistics
// ---------------------------------------------------------------------------

/// Second and third moments of the image gradient `(g_x, g_y)`, accumulated
/// over planes and masks, and the shear they imply.
///
/// For `I = I0 + κ ∂φ/∂s` the gradient energy along a direction `v` is largest
/// for `v ∥ s` when the specimen is statistically isotropic (`E[k_v² k_s²]` is
/// maximal at `v = s`), which gives the orientation modulo π without any
/// object. The third moment `E[(∂_v I)³]` is odd in `v`; for specimens made of
/// dense (phase-positive) bodies it is positive along the bright-edge
/// direction `−s`, which gives the sign. Both assume many objects of no
/// preferred orientation — granular cytoplasm is ideal; a stack of aligned
/// fibres is not.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GradientMoments {
    pub n: f64,
    pub xx: f64,
    pub xy: f64,
    pub yy: f64,
    pub xxx: f64,
    pub xxy: f64,
    pub xyy: f64,
    pub yyy: f64,
}

/// What [`GradientMoments::estimate`] concludes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientShear {
    /// Axis of the largest gradient energy, radians in `[0, π)`.
    pub orientation: f64,
    /// Ratio of the larger to the smaller second-moment eigenvalue (1: no
    /// preferred direction, so no orientation).
    pub anisotropy: f64,
    /// The bright-edge angle: `orientation` or `orientation + π`, whichever has
    /// the positive third moment, radians in `(−π, π]`.
    pub theta: f64,
    /// The unconstrained maximiser of the normalised third moment over the
    /// circle; agrees with `theta` when the data obey the model.
    pub theta_free: f64,
    /// The normalised third moment along `theta`: `E[(∂_θ I)³] / E[(∂_θ I)²]^{3/2}`.
    pub skewness: f64,
}

impl GradientMoments {
    pub fn merge(&mut self, other: &GradientMoments) {
        self.n += other.n;
        self.xx += other.xx;
        self.xy += other.xy;
        self.yy += other.yy;
        self.xxx += other.xxx;
        self.xxy += other.xxy;
        self.xyy += other.xyy;
        self.yyy += other.yyy;
    }

    /// Accumulate one plane's Gaussian gradient at `sigma` pixels where `mask`
    /// is true.
    pub fn add_plane<T: Copy + Into<f64>>(
        &mut self,
        plane: ArrayView2<'_, T>,
        mask: Option<ArrayView2<'_, bool>>,
        sigma: f64,
    ) -> Result<()> {
        let plane = widen_plane(plane);
        if let Some(mask) = mask.as_ref() {
            super::shapes_agree(plane.shape(), mask.shape(), "GradientMoments (mask)")?;
        }
        let (gy, gx) = gaussian_gradient_plane(plane.view(), sigma)?;
        for ((idx, &x), &y) in gx.indexed_iter().zip(gy.iter()) {
            if mask.as_ref().is_some_and(|m| !m[idx]) || !x.is_finite() || !y.is_finite() {
                continue;
            }
            self.n += 1.0;
            self.xx += x * x;
            self.xy += x * y;
            self.yy += y * y;
            self.xxx += x * x * x;
            self.xxy += x * x * y;
            self.xyy += x * y * y;
            self.yyy += y * y * y;
        }
        Ok(())
    }

    fn second(&self, angle: f64) -> f64 {
        let (c, s) = (angle.cos(), angle.sin());
        (c * c * self.xx + 2.0 * c * s * self.xy + s * s * self.yy) / self.n
    }

    fn third(&self, angle: f64) -> f64 {
        let (c, s) = (angle.cos(), angle.sin());
        (c * c * c * self.xxx
            + 3.0 * c * c * s * self.xxy
            + 3.0 * c * s * s * self.xyy
            + s * s * s * self.yyy)
            / self.n
    }

    fn normalised_third(&self, angle: f64) -> f64 {
        let second = self.second(angle);
        if second > 0.0 {
            self.third(angle) / second.powf(1.5)
        } else {
            0.0
        }
    }

    pub fn estimate(&self) -> Result<GradientShear> {
        if !(self.n > 0.0) {
            return Err(Error::invalid("no gradient sample was accumulated"));
        }
        let (a, b, c) = (self.xx / self.n, self.yy / self.n, self.xy / self.n);
        let orientation = (0.5 * (2.0 * c).atan2(a - b)).rem_euclid(std::f64::consts::PI);
        let half = 0.5 * (a + b);
        let disc = (0.25 * (a - b) * (a - b) + c * c).sqrt();
        let anisotropy = if half - disc > 0.0 {
            (half + disc) / (half - disc)
        } else {
            f64::INFINITY
        };
        let theta = if self.third(orientation) >= 0.0 {
            orientation
        } else {
            orientation - std::f64::consts::PI
        };
        let steps = 3600;
        let mut theta_free = 0.0;
        let mut best = f64::NEG_INFINITY;
        for i in 0..steps {
            let angle =
                -std::f64::consts::PI + (i as f64 + 0.5) * std::f64::consts::TAU / steps as f64;
            let v = self.normalised_third(angle);
            if v > best {
                best = v;
                theta_free = angle;
            }
        }
        Ok(GradientShear {
            orientation,
            anisotropy,
            theta: wrap_angle(theta),
            theta_free,
            skewness: self.normalised_third(theta),
        })
    }
}

/// An angle wrapped into `(−π, π]`.
pub fn wrap_angle(angle: f64) -> f64 {
    let wrapped =
        (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
    if wrapped <= -std::f64::consts::PI {
        wrapped + std::f64::consts::TAU
    } else {
        wrapped
    }
}

/// The signed difference `a − b` of two angles, wrapped into `(−π, π]`.
pub fn angle_difference(a: f64, b: f64) -> f64 {
    wrap_angle(a - b)
}

// ---------------------------------------------------------------------------
// the cosine fit
// ---------------------------------------------------------------------------

/// A robust fit of `v(φ) = a + b·cos(φ − θ) [+ b2·cos(2(φ − θ2))]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosineFit {
    /// The symmetric (angle-independent) term.
    pub a: f64,
    /// The relief strength, `≥ 0`.
    pub b: f64,
    /// The angle of the maximum, radians in `(−π, π]`.
    pub theta: f64,
    /// Second-harmonic amplitude and phase, when fitted.
    pub b2: Option<f64>,
    pub theta2: Option<f64>,
    /// Robust (Tukey-weighted) residual scale, 1.4826 × MAD of the residuals.
    pub residual: f64,
    /// Samples, and samples with non-zero final weight.
    pub n: usize,
    pub inliers: usize,
    /// Approximate standard error of `theta`, radians: `residual / (b·√(n_in/2))`.
    pub theta_se: f64,
}

fn solve_normal(mut a: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        rhs.swap(col, pivot);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for k in row + 1..n {
            acc -= a[row][k] * x[k];
        }
        x[row] = acc / a[row][row];
    }
    Some(x)
}

/// Weighted linear least squares of `y ≈ X β`.
pub(crate) fn weighted_least_squares(rows: &[Vec<f64>], y: &[f64], w: &[f64]) -> Option<Vec<f64>> {
    let p = rows.first()?.len();
    let mut ata = vec![vec![0.0; p]; p];
    let mut aty = vec![0.0; p];
    for ((row, &yi), &wi) in rows.iter().zip(y).zip(w) {
        if wi <= 0.0 {
            continue;
        }
        for i in 0..p {
            aty[i] += wi * row[i] * yi;
            for j in 0..p {
                ata[i][j] += wi * row[i] * row[j];
            }
        }
    }
    solve_normal(ata, aty)
}

/// Fit [`CosineFit`] to samples `(phi[i], value[i])` by iteratively reweighted
/// least squares with Tukey's biweight (c = 4.685), 20 iterations.
pub fn fit_cosine(phi: &[f64], value: &[f64], second_harmonic: bool) -> Result<CosineFit> {
    let pairs: Vec<(f64, f64)> = phi
        .iter()
        .zip(value)
        .filter(|(p, v)| p.is_finite() && v.is_finite())
        .map(|(&p, &v)| (p, v))
        .collect();
    let p = if second_harmonic { 5 } else { 3 };
    if pairs.len() < p + 2 {
        return Err(Error::invalid(format!(
            "a cosine fit needs at least {} finite samples; got {}",
            p + 2,
            pairs.len()
        )));
    }
    let rows: Vec<Vec<f64>> = pairs
        .iter()
        .map(|&(f, _)| {
            let mut row = vec![1.0, f.cos(), f.sin()];
            if second_harmonic {
                row.extend([(2.0 * f).cos(), (2.0 * f).sin()]);
            }
            row
        })
        .collect();
    let y: Vec<f64> = pairs.iter().map(|&(_, v)| v).collect();
    let mut w = vec![1.0; y.len()];
    let mut beta = vec![0.0; p];
    let mut scale = 0.0;
    for _ in 0..20 {
        beta = weighted_least_squares(&rows, &y, &w)
            .ok_or_else(|| Error::invalid("the cosine fit is singular (angles not spread?)"))?;
        let residuals: Vec<f64> = rows
            .iter()
            .zip(&y)
            .map(|(row, &yi)| yi - row.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        let mut abs: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
        scale = 1.4826 * percentile_of(&mut abs, 50.0);
        if !(scale > 0.0) {
            break;
        }
        for (wi, r) in w.iter_mut().zip(&residuals) {
            let u = r / (4.685 * scale);
            *wi = if u.abs() < 1.0 {
                (1.0 - u * u).powi(2)
            } else {
                0.0
            };
        }
    }
    let b = beta[1].hypot(beta[2]);
    let inliers = w.iter().filter(|&&wi| wi > 0.0).count();
    let theta = beta[2].atan2(beta[1]);
    let (b2, theta2) = if second_harmonic {
        (
            Some(beta[3].hypot(beta[4])),
            Some(0.5 * beta[4].atan2(beta[3])),
        )
    } else {
        (None, None)
    };
    Ok(CosineFit {
        a: beta[0],
        b,
        theta,
        b2,
        theta2,
        residual: scale,
        n: y.len(),
        inliers,
        theta_se: if b > 0.0 {
            scale / (b * (inliers as f64 / 2.0).sqrt())
        } else {
            f64::INFINITY
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A field of Gaussian phase bumps (dense bodies), its DIC image for a
    /// given bright-edge angle, and the bumps' centres.
    fn bumps(
        shape: (usize, usize),
        sigma: f64,
        count: usize,
        seed: u64,
    ) -> (Array2<f64>, Vec<(f64, f64)>) {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let centres: Vec<(f64, f64)> = (0..count)
            .map(|_| (next() * shape.0 as f64, next() * shape.1 as f64))
            .collect();
        let phase = Array2::from_shape_fn(shape, |(i, j)| {
            centres
                .iter()
                .map(|&(cy, cx)| {
                    let d2 = (i as f64 - cy).powi(2) + (j as f64 - cx).powi(2);
                    (-0.5 * d2 / (sigma * sigma)).exp()
                })
                .sum()
        });
        (phase, centres)
    }

    /// `κ ∂φ/∂s` with `s = theta + π`, by central differences.
    fn dic_of(phase: &Array2<f64>, theta: f64) -> Array2<f64> {
        let (ny, nx) = phase.dim();
        let s = theta + PI;
        let (c, sn) = (s.cos(), s.sin());
        Array2::from_shape_fn((ny, nx), |(i, j)| {
            let at = |ii: isize, jj: isize| {
                phase[[
                    ii.clamp(0, ny as isize - 1) as usize,
                    jj.clamp(0, nx as isize - 1) as usize,
                ]]
            };
            let (i, j) = (i as isize, j as isize);
            let dx = 0.5 * (at(i, j + 1) - at(i, j - 1));
            let dy = 0.5 * (at(i + 1, j) - at(i - 1, j));
            c * dx + sn * dy
        })
    }

    #[test]
    fn the_cosine_fit_recovers_a_b_theta_and_rejects_outliers() {
        let theta = 2.2;
        let phi: Vec<f64> = (0..360).map(|k| -PI + k as f64 * PI / 180.0).collect();
        let mut value: Vec<f64> = phi.iter().map(|&f| 0.3 + 1.5 * (f - theta).cos()).collect();
        for k in (0..360).step_by(17) {
            value[k] += 25.0;
        }
        let fit = fit_cosine(&phi, &value, false).unwrap();
        assert!((fit.a - 0.3).abs() < 1e-6, "{fit:?}");
        assert!((fit.b - 1.5).abs() < 1e-6);
        assert!(angle_difference(fit.theta, theta).abs() < 1e-6);
        assert_eq!(fit.inliers, 360 - 22);
        let two = fit_cosine(&phi, &value, true).unwrap();
        assert!(two.b2.unwrap() < 1e-6);
    }

    #[test]
    fn integration_inverts_the_shear_derivative_with_the_right_sign() {
        for &theta in &[0.0, 0.8, -2.3, PI] {
            let phase = Array2::from_shape_fn((96, 128), |(i, j)| {
                let d2 = (i as f64 - 48.0).powi(2) + (j as f64 - 64.0).powi(2);
                (-0.5 * d2 / 16.0).exp()
            });
            let image = dic_of(&phase, theta);
            let params = DicIntegration::new(theta, 300.0).unwrap();
            let rec = dic_integrate_plane(image.view(), &params).unwrap();
            // dense body reconstructs positive, peaked at its centre
            let peak = rec[[48, 64]];
            assert!(peak > 0.9 && peak < 1.05, "theta {theta}: peak {peak}");
            let err: f64 =
                (&rec - &phase).mapv(|v| v * v).sum().sqrt() / phase.mapv(|v| v * v).sum().sqrt();
            assert!(err < 0.15, "theta {theta}: relative error {err}");
            // the wrong direction reconstructs a negative body
            let wrong = DicIntegration::new(theta + PI, 300.0).unwrap();
            assert!(dic_integrate_plane(image.view(), &wrong).unwrap()[[48, 64]] < -0.8);
        }
    }

    #[test]
    fn gradient_moments_find_the_bright_edge_direction() {
        let (phase, _) = bumps((160, 160), 2.5, 120, 3);
        for &theta in &[0.4, 2.0, -1.0, -2.8] {
            let image = dic_of(&phase, theta);
            let mut m = GradientMoments::default();
            m.add_plane(image.view(), None, 1.0).unwrap();
            let est = m.estimate().unwrap();
            assert!(
                angle_difference(est.theta, theta).abs() < 0.12,
                "theta {theta}: {est:?}"
            );
            assert!(
                angle_difference(est.theta_free, theta).abs() < 0.2,
                "theta {theta}: {est:?}"
            );
            assert!(est.anisotropy > 1.5 && est.skewness > 0.0);
        }
    }

    #[test]
    fn the_isotropic_texture_ignores_a_straight_edge_that_the_energy_sees() {
        // left half: a smooth step (an "edge"), right half: granules
        let (granules, _) = bumps((80, 80), 1.2, 160, 11);
        let image = Array2::from_shape_fn((80, 160), |(i, j)| {
            if j < 80 {
                if j < 40 {
                    0.0
                } else {
                    1.0
                }
            } else {
                granules[[i, j - 80]]
            }
        });
        let volume = image.clone().insert_axis(ndarray::Axis(0));
        let energy = DicTexture::new((1.0, 3.0), 3.0, TextureMeasure::Energy, 0.0, 1e-9).unwrap();
        let iso = DicTexture::new((1.0, 3.0), 3.0, TextureMeasure::Isotropic, 0.0, 1e-9).unwrap();
        let se = dic_texture_volume(volume.view(), &energy).unwrap();
        let si = dic_texture_volume(volume.view(), &iso).unwrap();
        // smoothness at the edge relative to the granules: much higher with the isotropic measure
        let edge_e = se[[0, 40, 40]] - se[[0, 40, 120]];
        let edge_i = si[[0, 40, 40]] - si[[0, 40, 120]];
        assert!(edge_i > edge_e + 2.0, "energy {edge_e}, isotropic {edge_i}");
    }

    #[test]
    fn centre_surround_peaks_on_a_smooth_disc() {
        let s = Array2::from_shape_fn((81, 81), |(i, j)| {
            let d = ((i as f64 - 40.0).powi(2) + (j as f64 - 40.0).powi(2)).sqrt();
            if d < 12.0 {
                1.0
            } else {
                0.0
            }
        })
        .insert_axis(ndarray::Axis(0));
        let cs = CentreSurround::new(vec![8.0, 12.0, 16.0], 1.6).unwrap();
        let n = centre_surround_volume(s.view(), &cs).unwrap();
        let (mut best, mut at) = (f64::NEG_INFINITY, (0, 0));
        for ((_, i, j), &v) in n.indexed_iter() {
            if v > best {
                best = v;
                at = (i, j);
            }
        }
        assert_eq!(at, (40, 40));
    }

    #[test]
    fn the_ops_match_the_free_functions() {
        let (phase, _) = bumps((40, 48), 2.0, 30, 5);
        let image = dic_of(&phase, 0.7);
        let volume = Array3::from_shape_fn((3, 40, 48), |(z, i, j)| {
            image[[i, j]] * (1.0 + z as f64) + 5.0
        });
        let input: Voxels = volume.clone().into();
        let shape = [3, 40, 48];

        let norm = DicNormalise::new(0.0, 2.0).unwrap();
        let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
        DicNormaliseOp::new("n", norm)
            .apply(&input, &mut out, &Anchor::whole(shape))
            .unwrap();
        let mut expect = Array3::zeros((3, 40, 48));
        dic_normalise_into(volume.view(), &norm, expect.view_mut()).unwrap();
        assert_eq!(out.view::<f64>().unwrap(), expect.view());

        let integ = DicIntegration::new(0.7, 20.0).unwrap();
        let normalised: Voxels = expect.clone().into();
        let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
        DicIntegrateOp::new("i", integ)
            .apply(&normalised, &mut out, &Anchor::whole(shape))
            .unwrap();
        let mut rec = Array3::zeros((3, 40, 48));
        dic_integrate_into(expect.view(), &integ, rec.view_mut()).unwrap();
        assert_eq!(out.view::<f64>().unwrap(), rec.view());

        let tex = DicTexture::new((1.0, 3.0), 2.0, TextureMeasure::Isotropic, 1.0, 1e-6).unwrap();
        let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
        let op = DicTextureOp::new("t", tex);
        assert_eq!(op.reach(0, 3), 3);
        op.apply(&input, &mut out, &Anchor::whole(shape)).unwrap();
        assert_eq!(
            out.view::<f64>().unwrap(),
            dic_texture_volume(volume.view(), &tex).unwrap().view()
        );
    }

    #[test]
    fn the_relief_scale_is_the_mad_of_the_relief() {
        let volume = Array3::from_shape_fn((2, 30, 30), |(_, i, j)| {
            100.0 + if (i + j) % 2 == 0 { 3.0 } else { -3.0 }
        });
        let scale = dic_relief_scale(volume.view(), None, 0.0, 1).unwrap();
        assert!((scale - 1.4826 * 3.0).abs() < 1e-9);
    }
}
