// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! **Nonlinear diffusion and total-variation denoising.**
//!
//! Structure-preserving smoothing: noise is flattened while edges (and, for
//! coherence-enhancing diffusion, thin sheets such as cell membranes in volume
//! EM) are kept or even strengthened. Every model here evolves
//!
//! ```text
//!     du/dt = div( D(u) grad u )
//! ```
//!
//! and differs only in the diffusion tensor `D`, which is built from the image
//! *pre-smoothed* at a derivative scale `sigma` (the Catté regularisation that
//! makes the problem well posed):
//!
//! * [`DiffusionModel::PeronaMalik`] — scalar `D = g(|grad u_sigma|^2) I`;
//! * [`DiffusionModel::EdgeEnhancing`] — Weickert EED: `D` has eigenvector
//!   `grad u_sigma` with eigenvalue `g(|grad u_sigma|^2)` and eigenvalue `1`
//!   on the plane orthogonal to it, so smoothing runs freely *along* an edge
//!   and is throttled *across* it;
//! * [`DiffusionModel::CoherenceEnhancing`] — Weickert CED, adapted to sheets.
//!   `D` is built from the structure tensor `J = G_rho * (grad u_sigma grad
//!   u_sigma^T)` with eigenvalues `mu1 >= mu2 >= mu3` and eigenvectors `w1, w2,
//!   w3`. For a membrane (a sheet) `w1` is the sheet normal and `w2, w3` span
//!   the sheet. The mapping chosen is
//!
//!   ```text
//!       across  (w1):      lambda1 = alpha
//!       in-sheet (w2, w3): lambda2 = lambda3 = alpha + (1 - alpha) exp(-C / kappa),
//!       kappa = (mu1 - mu2)^2
//!   ```
//!
//!   `kappa` is the *sheet* coherence — one dominant gradient direction —
//!   rather than Weickert's 2-D `(mu1 - mu2)^2` read as line coherence or the
//!   tube measure `(mu2 - mu3)^2`. Both in-sheet eigenvalues are equal, so `D =
//!   lambda2 I + (alpha - lambda2) w1 w1^T` needs only the principal
//!   eigenvector: `w2` and `w3` are ill-conditioned exactly on a flat sheet
//!   (where `mu2 ~ mu3`) and never enter. Where gradients disagree (sheet
//!   junctions, tubes, noise, `mu1 ~ mu2`) `kappa -> 0` and the tensor
//!   relaxes to the small isotropic `alpha I`, which is what keeps CED from
//!   inventing orientation in texture.
//!
//! # The scheme, and its step-size bound
//!
//! One explicit Euler step `u' = u + dt div(D grad u)` on the voxel grid with
//! per-axis spacing `h`. Diagonal terms use half-point averaged
//! diffusivities, off-diagonal terms the standard central-difference stencil
//! (Weickert 1998, §3.4). With `S = sum_a 1 / h_a^2`:
//!
//! * scalar (Perona–Malik): `dt <= 1 / (2 S)` — the classic bound, under which
//!   the step is a convex combination of neighbours (maximum principle);
//! * tensor (EED, CED): the step matrix is symmetric and, since `D` has
//!   eigenvalues in `[0, 1]` (so `|D_ab| <= 1/2`), Gershgorin bounds its
//!   spectrum by `4 S + sum_{a<b} 1/(h_a h_b) <= 5 S`, so `dt <= 2 / (5 S)`
//!   keeps the explicit step from amplifying.
//!
//! [`max_stable_dt`] returns these, and [`DiffusionParams::new`] refuses a
//! larger step rather than substepping quietly.
//!
//! # Out-of-core
//!
//! Modelled on `ops::level_set`: **one iteration is one [`BlockOp`]**,
//! [`DiffusionStepOp`], whose reach is the Gaussian radius of `sigma`, one for
//! the gradient's central difference, the radius of `rho` for CED's
//! integration, and one for the divergence stencil. A chain of `K` steps
//! ([`diffusion_chain`], [`append_diffusion_phases`]) therefore runs blockwise
//! with a halo of at most `K` times that. [`diffuse_into`] is the in-memory
//! convenience. Every derivative clamps at the buffer edge (the crate-wide
//! convention: wrong at a block seam, made right by the halo, and a mirror
//! boundary at the real volume edge).
//!
//! # TV denoising
//!
//! [`tv_denoise_into`] solves the ROF model `min_u TV(u) + |u - f|^2 / (2
//! lambda)` by Chambolle's (2004) dual projection. It is resident only: the
//! dual variable is a three-component vector field carried between iterations,
//! and the crate has no multi-channel phase state for it to live in, so a
//! blockwise step op is not offered.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::assemble::{Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Chain, Slicing};
use crate::reach::Reach;
use crate::voxels::Voxels;

use super::ridge::{
    gaussian_radius, gaussian_smooth_into, gaussian_weights, symmetric_eigenvalues,
    SMOOTH_COST_PER_TAP,
};
use super::shapes_agree;
use super::structure_tensor::gaussian_gradient_into;

/// Weickert's constant for the `m = 4` diffusivity, chosen so that the flux
/// `g(s^2) s` is increasing for `s < lambda` and decreasing above it.
pub const WEICKERT_C4: f64 = 3.31488;

/// The edge-stopping function `g(|grad u|^2)`, with contrast parameter
/// `lambda`. Every choice is `1` at zero gradient and decreasing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Diffusivity {
    /// Perona–Malik's rational `1 / (1 + s^2 / lambda^2)`.
    PeronaMalik,
    /// Perona–Malik's exponential `exp(-s^2 / lambda^2)`.
    Exponential,
    /// Charbonnier's `1 / sqrt(1 + s^2 / lambda^2)`; convex energy, so no
    /// edge sharpening but no staircasing either.
    Charbonnier,
    /// Weickert's `1 - exp(-C4 / (s / lambda)^8)`; nearly `1` below the
    /// contrast and falling sharply above it.
    Weickert,
}

impl Diffusivity {
    /// `g` at a squared gradient magnitude, for contrast `lambda > 0`.
    pub fn at(self, gradient_sq: f64, contrast: f64) -> f64 {
        let s = gradient_sq / (contrast * contrast);
        match self {
            Diffusivity::PeronaMalik => 1.0 / (1.0 + s),
            Diffusivity::Exponential => (-s).exp(),
            Diffusivity::Charbonnier => 1.0 / (1.0 + s).sqrt(),
            Diffusivity::Weickert => {
                if s <= 0.0 {
                    1.0
                } else {
                    1.0 - (-WEICKERT_C4 / (s * s * s * s)).exp()
                }
            }
        }
    }
}

/// Which diffusion tensor a step builds. See the module header.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiffusionModel {
    /// Scalar diffusivity `g(|grad u_sigma|^2)`.
    PeronaMalik {
        diffusivity: Diffusivity,
        contrast: f64,
    },
    /// Weickert's edge-enhancing tensor from `grad u_sigma`.
    EdgeEnhancing {
        diffusivity: Diffusivity,
        contrast: f64,
    },
    /// Weickert's coherence-enhancing tensor, sheet variant.
    ///
    /// `rho` is the integration scale in voxels per axis, `alpha` in `(0, 1]`
    /// the diffusion across the sheet, `coherence` the threshold `C` (in units
    /// of squared structure-tensor eigenvalues) above which in-sheet diffusion
    /// switches on.
    CoherenceEnhancing {
        rho: [f64; 3],
        alpha: f64,
        coherence: f64,
    },
}

impl DiffusionModel {
    fn is_scalar(&self) -> bool {
        matches!(self, DiffusionModel::PeronaMalik { .. })
    }
}

/// The largest explicit step [`DiffusionParams::new`] accepts for `model` at
/// `spacing`: `1 / (2 S)` for the scalar model and `2 / (5 S)` for the tensor
/// models, `S = sum 1 / h^2`. Derivation in the module header.
pub fn max_stable_dt(model: &DiffusionModel, spacing: [f64; 3]) -> f64 {
    let s: f64 = spacing.iter().map(|h| 1.0 / (h * h)).sum();
    if model.is_scalar() {
        1.0 / (2.0 * s)
    } else {
        2.0 / (5.0 * s)
    }
}

/// One explicit diffusion step, fully specified.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionParams {
    model: DiffusionModel,
    sigma: [f64; 3],
    spacing: [f64; 3],
    dt: f64,
    truncate: f64,
}

impl DiffusionParams {
    /// `sigma` is the pre-smoothing (derivative) scale in **voxels** per axis,
    /// as everywhere in the crate — for an isotropic physical scale pass
    /// `sigma_phys / spacing`. `spacing` is the physical voxel size, used for
    /// every derivative and for the step bound. `truncate` is where the
    /// Gaussians are cut, in standard deviations.
    pub fn new(
        model: DiffusionModel,
        sigma: [f64; 3],
        spacing: [f64; 3],
        dt: f64,
        truncate: f64,
    ) -> Result<Self> {
        let params = Self {
            model,
            sigma,
            spacing,
            dt,
            truncate,
        };
        params.validate()?;
        Ok(params)
    }

    pub fn model(&self) -> DiffusionModel {
        self.model
    }

    pub fn sigma(&self) -> [f64; 3] {
        self.sigma
    }

    pub fn spacing(&self) -> [f64; 3] {
        self.spacing
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    pub fn truncate(&self) -> f64 {
        self.truncate
    }

    /// Voxels read beyond the voxel written, along `axis`, by one step: the
    /// pre-smoothing radius, the gradient's central difference, CED's
    /// integration radius, and the divergence stencil. Stages applied in turn,
    /// so they add.
    pub fn reach(&self, axis: usize) -> usize {
        let integration = match self.model {
            DiffusionModel::CoherenceEnhancing { rho, .. } => {
                gaussian_radius(rho[axis], self.truncate)
            }
            _ => 0,
        };
        gaussian_radius(self.sigma[axis], self.truncate) + 1 + integration + 1
    }

    fn validate(&self) -> Result<()> {
        for axis in 0..3 {
            let h = self.spacing[axis];
            if !(h.is_finite() && h > 0.0) {
                return Err(Error::InvalidArgument(format!(
                    "diffusion: spacing[{axis}] is {h}; a voxel size is positive and finite"
                )));
            }
            let s = self.sigma[axis];
            if !(s.is_finite() && s >= 0.0) {
                return Err(Error::InvalidArgument(format!(
                    "diffusion: sigma[{axis}] is {s}; a scale is a non-negative finite number \
                     of voxels"
                )));
            }
        }
        if !(self.truncate.is_finite() && self.truncate > 0.0) {
            return Err(Error::InvalidArgument(format!(
                "diffusion: truncate is {}; it must be positive",
                self.truncate
            )));
        }
        match self.model {
            DiffusionModel::PeronaMalik { contrast, .. }
            | DiffusionModel::EdgeEnhancing { contrast, .. } => {
                if !(contrast.is_finite() && contrast > 0.0) {
                    return Err(Error::InvalidArgument(format!(
                        "diffusion: contrast is {contrast}; it must be positive and finite"
                    )));
                }
            }
            DiffusionModel::CoherenceEnhancing {
                rho,
                alpha,
                coherence,
            } => {
                if rho.iter().any(|r| !(r.is_finite() && *r >= 0.0))
                    || rho.iter().all(|&r| r == 0.0)
                {
                    return Err(Error::InvalidArgument(format!(
                        "coherence-enhancing diffusion: rho is {rho:?}; the integration scale \
                         must be non-negative and positive on at least one axis, or the \
                         structure tensor has rank one and no coherence to measure"
                    )));
                }
                if !(alpha > 0.0 && alpha <= 1.0) {
                    return Err(Error::InvalidArgument(format!(
                        "coherence-enhancing diffusion: alpha is {alpha}; it must lie in (0, 1]"
                    )));
                }
                if !(coherence.is_finite() && coherence > 0.0) {
                    return Err(Error::InvalidArgument(format!(
                        "coherence-enhancing diffusion: the coherence threshold is {coherence}; \
                         it must be positive and finite"
                    )));
                }
            }
        }
        let bound = max_stable_dt(&self.model, self.spacing);
        if !(self.dt.is_finite() && self.dt > 0.0) {
            return Err(Error::InvalidArgument(format!(
                "diffusion: dt is {}; it must be positive and finite",
                self.dt
            )));
        }
        if self.dt > bound {
            return Err(Error::InvalidArgument(format!(
                "diffusion: dt {} is above the explicit stability bound {bound} for this model \
                 and spacing; take more, smaller steps",
                self.dt
            )));
        }
        Ok(())
    }

    fn cost(&self) -> f64 {
        let taps = |scale: [f64; 3]| -> usize {
            (0..3)
                .map(|axis| 2 * gaussian_radius(scale[axis], self.truncate) + 1)
                .sum()
        };
        let integration = match self.model {
            DiffusionModel::CoherenceEnhancing { rho, .. } => {
                6.0 * taps(rho) as f64 * SMOOTH_COST_PER_TAP + DIFFUSION_DECOMPOSITION_COST
            }
            _ => 0.0,
        };
        SMOOTH_COST_PER_TAP * taps(self.sigma) as f64 + integration + DIFFUSION_STENCIL_COST
    }
}

/// Placeholder per-voxel cost of the divergence stencil and tensor assembly,
/// in voxelwise-map units. Not measured; retake through the cost harness.
pub const DIFFUSION_STENCIL_COST: f64 = 40.0;

/// Placeholder per-voxel cost of CED's eigen-decomposition (eigenvalues plus
/// the principal eigenvector). Not measured.
pub const DIFFUSION_DECOMPOSITION_COST: f64 = 70.0;

/// The diffusion tensor field: scalar, or six components in the crate's
/// `[xx, yy, zz, xy, xz, yz]` order (axis 0 is `x`).
enum TensorField {
    Scalar(Array3<f64>),
    Full(Box<[Array3<f64>; 6]>),
}

/// Pairs `(a, b)` for each off-diagonal component index `3..6`.
const OFF_DIAGONAL: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// One explicit diffusion step of `input` into `out`.
pub fn diffusion_step_into<T>(
    input: ArrayView3<'_, T>,
    params: &DiffusionParams,
    out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    params.validate()?;
    shapes_agree(input.shape(), out.shape(), "diffusion_step_into")?;
    let u = input.mapv(|value| value.into());
    validate_finite(u.view(), "diffusion input")?;
    step_f64(u.view(), params, out)
}

/// `iterations` explicit steps, resident. Equal, step for step, to a chain of
/// `iterations` [`DiffusionStepOp`]s run over the whole volume.
pub fn diffuse_into<T>(
    input: ArrayView3<'_, T>,
    params: &DiffusionParams,
    iterations: usize,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    params.validate()?;
    shapes_agree(input.shape(), out.shape(), "diffuse_into")?;
    let mut current = input.mapv(|value| value.into());
    validate_finite(current.view(), "diffusion input")?;
    let mut next = Array3::<f64>::zeros(current.raw_dim());
    for _ in 0..iterations {
        step_f64(current.view(), params, next.view_mut())?;
        std::mem::swap(&mut current, &mut next);
    }
    out.assign(&current);
    Ok(())
}

fn step_f64(
    u: ArrayView3<'_, f64>,
    params: &DiffusionParams,
    out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    if u.is_empty() {
        return Ok(());
    }
    let tensor = diffusion_tensor(u, params)?;
    explicit_update(u, &tensor, params, out);
    Ok(())
}

/// The physical gradient of `u` pre-smoothed at `sigma`.
fn smoothed_gradient(u: ArrayView3<'_, f64>, params: &DiffusionParams) -> Result<[Array3<f64>; 3]> {
    let dim = u.raw_dim();
    let mut gradient = [
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
    ];
    {
        let [g0, g1, g2] = &mut gradient;
        gaussian_gradient_into(
            u,
            params.sigma,
            params.truncate,
            [g0.view_mut(), g1.view_mut(), g2.view_mut()],
        )?;
    }
    for (axis, component) in gradient.iter_mut().enumerate() {
        let h = params.spacing[axis];
        component.mapv_inplace(|value| value / h);
    }
    Ok(gradient)
}

fn diffusion_tensor(u: ArrayView3<'_, f64>, params: &DiffusionParams) -> Result<TensorField> {
    let gradient = smoothed_gradient(u, params)?;
    let dim = u.raw_dim();
    match params.model {
        DiffusionModel::PeronaMalik {
            diffusivity,
            contrast,
        } => {
            let mut g = Array3::<f64>::zeros(dim);
            for ((index, slot), &g0) in g.indexed_iter_mut().zip(gradient[0].iter()) {
                let g1 = gradient[1][index];
                let g2 = gradient[2][index];
                *slot = diffusivity.at(g0 * g0 + g1 * g1 + g2 * g2, contrast);
            }
            Ok(TensorField::Scalar(g))
        }
        DiffusionModel::EdgeEnhancing {
            diffusivity,
            contrast,
        } => {
            let mut d = zeros6(dim);
            for (index, &g0) in gradient[0].indexed_iter() {
                let v = [g0, gradient[1][index], gradient[2][index]];
                let norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                // D = I + (g - 1) n n^T with n = v / |v|; at |v| = 0, g = 1 and D = I.
                let c = if norm_sq > 0.0 {
                    (diffusivity.at(norm_sq, contrast) - 1.0) / norm_sq
                } else {
                    0.0
                };
                write_tensor(&mut d, index, 1.0, c, v);
            }
            Ok(TensorField::Full(Box::new(d)))
        }
        DiffusionModel::CoherenceEnhancing {
            rho,
            alpha,
            coherence,
        } => {
            // The structure tensor: products of the physical gradient,
            // integrated at rho.
            let mut product = zeros6(dim);
            for (index, &g0) in gradient[0].indexed_iter() {
                let v = [g0, gradient[1][index], gradient[2][index]];
                for axis in 0..3 {
                    product[axis][index] = v[axis] * v[axis];
                }
                for (slot, &(a, b)) in OFF_DIAGONAL.iter().enumerate() {
                    product[3 + slot][index] = v[a] * v[b];
                }
            }
            let kernels = [
                gaussian_weights(rho[0], params.truncate)?,
                gaussian_weights(rho[1], params.truncate)?,
                gaussian_weights(rho[2], params.truncate)?,
            ];
            let mut j = zeros6(dim);
            for component in 0..6 {
                gaussian_smooth_into(product[component].view(), &kernels, j[component].view_mut())?;
            }
            let mut d = zeros6(dim);
            for (index, &jxx) in j[0].indexed_iter() {
                let tensor = [
                    jxx,
                    j[1][index],
                    j[2][index],
                    j[3][index],
                    j[4][index],
                    j[5][index],
                ];
                let mu = symmetric_eigenvalues(tensor);
                let kappa = (mu[0] - mu[1]) * (mu[0] - mu[1]);
                let in_sheet = if kappa > 0.0 {
                    alpha + (1.0 - alpha) * (-coherence / kappa).exp()
                } else {
                    alpha
                };
                // D = in_sheet I + (alpha - in_sheet) w1 w1^T; when there is
                // no principal direction, in_sheet is alpha and D = alpha I.
                match principal_eigenvector(tensor, mu[0]) {
                    Some(w) => write_tensor(&mut d, index, in_sheet, alpha - in_sheet, w),
                    None => write_tensor(&mut d, index, in_sheet, 0.0, [0.0; 3]),
                }
            }
            Ok(TensorField::Full(Box::new(d)))
        }
    }
}

fn zeros6(dim: ndarray::Ix3) -> [Array3<f64>; 6] {
    [
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
    ]
}

/// Store `diag I + c v v^T` at `index`.
fn write_tensor(
    d: &mut [Array3<f64>; 6],
    index: (usize, usize, usize),
    diag: f64,
    c: f64,
    v: [f64; 3],
) {
    for axis in 0..3 {
        d[axis][index] = diag + c * v[axis] * v[axis];
    }
    for (slot, &(a, b)) in OFF_DIAGONAL.iter().enumerate() {
        d[3 + slot][index] = c * v[a] * v[b];
    }
}

/// Unit eigenvector of the symmetric `[xx, yy, zz, xy, xz, yz]` matrix for its
/// eigenvalue `mu`, as the largest cross product of two rows of `M - mu I`.
/// `None` when `mu` is (numerically) repeated, where no direction is defined.
fn principal_eigenvector(m: [f64; 6], mu: f64) -> Option<[f64; 3]> {
    let rows = [
        [m[0] - mu, m[3], m[4]],
        [m[3], m[1] - mu, m[5]],
        [m[4], m[5], m[2] - mu],
    ];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let mut best = [0.0; 3];
    let mut best_norm = 0.0;
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        let c = cross(rows[a], rows[b]);
        let n = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        if n > best_norm {
            best_norm = n;
            best = c;
        }
    }
    let scale = m.iter().fold(0.0f64, |w, v| w.max(v.abs()));
    if !(best_norm > 0.0) || best_norm <= (1e-24 * scale * scale) * scale * scale {
        return None;
    }
    let n = best_norm.sqrt();
    Some([best[0] / n, best[1] / n, best[2] / n])
}

/// `out = u + dt div(D grad u)`, clamped at the buffer edge. At the edge the
/// clamped sample is the mirror ghost for `u` and the diagonal of `D`; the
/// off-diagonal of `D` changes sign under that mirror, which is applied
/// explicitly so the scheme stays conservative.
fn explicit_update(
    u: ArrayView3<'_, f64>,
    tensor: &TensorField,
    params: &DiffusionParams,
    mut out: ArrayViewMut3<'_, f64>,
) {
    let shape = [u.shape()[0], u.shape()[1], u.shape()[2]];
    let h = params.spacing;
    let inv_h2 = [
        1.0 / (h[0] * h[0]),
        1.0 / (h[1] * h[1]),
        1.0 / (h[2] * h[2]),
    ];
    let step = |at: [usize; 3], axis: usize, delta: isize| -> ([usize; 3], bool) {
        let mut next = at;
        let moved = at[axis] as isize + delta;
        if moved < 0 || moved >= shape[axis] as isize {
            (at, true)
        } else {
            next[axis] = moved as usize;
            (next, false)
        }
    };
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                let here = u[at];
                let mut div = 0.0;
                for axis in 0..3 {
                    let (hi, _) = step(at, axis, 1);
                    let (lo, _) = step(at, axis, -1);
                    let (d_here, d_hi, d_lo) = match tensor {
                        TensorField::Scalar(g) => (g[at], g[hi], g[lo]),
                        TensorField::Full(d) => (d[axis][at], d[axis][hi], d[axis][lo]),
                    };
                    let flux_hi = 0.5 * (d_here + d_hi) * (u[hi] - here);
                    let flux_lo = 0.5 * (d_here + d_lo) * (here - u[lo]);
                    div += (flux_hi - flux_lo) * inv_h2[axis];
                }
                if let TensorField::Full(d) = tensor {
                    for (slot, &(p, q)) in OFF_DIAGONAL.iter().enumerate() {
                        let dpq = &d[3 + slot];
                        // d/da (D_ab d/db u) for (a, b) = (p, q) and (q, p).
                        for (a, b) in [(p, q), (q, p)] {
                            let flux = |centre: [usize; 3]| -> f64 {
                                let (bh, _) = step(centre, b, 1);
                                let (bl, _) = step(centre, b, -1);
                                (u[bh] - u[bl]) / (2.0 * h[b])
                            };
                            let here_flux = dpq[at] * flux(at);
                            let (hi, hi_ghost) = step(at, a, 1);
                            let (lo, lo_ghost) = step(at, a, -1);
                            let f_hi = if hi_ghost {
                                -here_flux
                            } else {
                                dpq[hi] * flux(hi)
                            };
                            let f_lo = if lo_ghost {
                                -here_flux
                            } else {
                                dpq[lo] * flux(lo)
                            };
                            div += (f_hi - f_lo) / (2.0 * h[a]);
                        }
                    }
                }
                out[at] = here + params.dt * div;
            }
        }
    }
}

fn validate_finite(field: ArrayView3<'_, f64>, what: &str) -> Result<()> {
    if let Some(((i, j, k), value)) = field.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(Error::InvalidArgument(format!(
            "{what} contains non-finite value {value} at [{i}, {j}, {k}]"
        )));
    }
    Ok(())
}

// ------------------------------------------------------------- the op --

/// One explicit diffusion step as a stencil [`BlockOp`]. Reads any non-`f16`
/// element type, writes `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionStepOp {
    name: &'static str,
    params: DiffusionParams,
    cost: f64,
}

impl DiffusionStepOp {
    pub fn new(name: &'static str, params: DiffusionParams) -> Self {
        let cost = params.cost();
        Self { name, params, cost }
    }

    pub fn params(&self) -> &DiffusionParams {
        &self.params
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for DiffusionStepOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.params.reach(axis)
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::symmetric([
            self.params.reach(0),
            self.params.reach(1),
            self.params.reach(2),
        ])
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

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let out = out.view_mut::<f64>()?;
        dispatch_f64_input!(
            input,
            Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; `accepts` refuses it before a run starts",
                self.name
            )),
            |view| { diffusion_step_into(view, &self.params, out) }
        )
    }

    /// A constant has no gradient, so every flux is zero and the step is the
    /// identity — exactly, not to rounding.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// `iterations` steps as one sequence; its reach is the sum of the steps'.
pub fn diffusion_chain(name: &'static str, params: &DiffusionParams, iterations: usize) -> Chain {
    Chain::sequence(
        (0..iterations)
            .map(|_| Chain::op(DiffusionStepOp::new(name, params.clone())))
            .collect(),
    )
}

/// Append one pixel phase per step, so a planner can materialise between
/// steps instead of carrying a `K`-fold halo.
pub fn append_diffusion_phases(
    builder: &mut PlanBuilder,
    name: &'static str,
    params: &DiffusionParams,
    iterations: usize,
) -> Result<Vec<Phase>> {
    params.validate()?;
    (0..iterations)
        .map(|_| builder.pixels(Chain::op(DiffusionStepOp::new(name, params.clone()))))
        .collect()
}

// ---------------------------------------------------------- TV (ROF) --

/// Parameters for ROF total-variation denoising by Chambolle's projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TvParams {
    lambda: f64,
    spacing: [f64; 3],
    tau: f64,
}

impl TvParams {
    /// `lambda` weighs the fidelity: the model is `TV(u) + |u - f|^2 / (2
    /// lambda)`, so a larger `lambda` smooths more. The dual step is set to
    /// Chambolle's convergence bound `1 / (4 S)`, `S = sum 1 / h^2`.
    pub fn new(lambda: f64, spacing: [f64; 3]) -> Result<Self> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(Error::InvalidArgument(format!(
                "tv denoise: lambda is {lambda}; it must be positive and finite"
            )));
        }
        for (axis, h) in spacing.iter().enumerate() {
            if !(h.is_finite() && *h > 0.0) {
                return Err(Error::InvalidArgument(format!(
                    "tv denoise: spacing[{axis}] is {h}; a voxel size is positive and finite"
                )));
            }
        }
        Ok(Self {
            lambda,
            spacing,
            tau: Self::tau_bound(spacing),
        })
    }

    /// `1 / ||div||^2 = 1 / (4 S)`.
    pub fn tau_bound(spacing: [f64; 3]) -> f64 {
        let s: f64 = spacing.iter().map(|h| 1.0 / (h * h)).sum();
        1.0 / (4.0 * s)
    }

    /// A smaller dual step; refused above the bound.
    pub fn with_tau(mut self, tau: f64) -> Result<Self> {
        let bound = Self::tau_bound(self.spacing);
        if !(tau > 0.0 && tau <= bound) {
            return Err(Error::InvalidArgument(format!(
                "tv denoise: tau {tau} must lie in (0, {bound}]"
            )));
        }
        self.tau = tau;
        Ok(self)
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn spacing(&self) -> [f64; 3] {
        self.spacing
    }

    pub fn tau(&self) -> f64 {
        self.tau
    }
}

/// Forward-difference gradient (zero across the last face, Neumann).
fn forward_gradient(u: &Array3<f64>, spacing: [f64; 3]) -> [Array3<f64>; 3] {
    let dim = u.raw_dim();
    let shape = [dim[0], dim[1], dim[2]];
    let mut out = [
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
    ];
    for ((i, j, k), &here) in u.indexed_iter() {
        let at = [i, j, k];
        for axis in 0..3 {
            if at[axis] + 1 < shape[axis] {
                let mut next = at;
                next[axis] += 1;
                out[axis][at] = (u[next] - here) / spacing[axis];
            }
        }
    }
    out
}

/// The negative adjoint of [`forward_gradient`].
fn backward_divergence(p: &[Array3<f64>; 3], spacing: [f64; 3]) -> Array3<f64> {
    let dim = p[0].raw_dim();
    let shape = [dim[0], dim[1], dim[2]];
    let mut out = Array3::<f64>::zeros(dim);
    for ((i, j, k), slot) in out.indexed_iter_mut() {
        let at = [i, j, k];
        let mut div = 0.0;
        for axis in 0..3 {
            let last = at[axis] + 1 == shape[axis];
            let here = if last { 0.0 } else { p[axis][at] };
            let before = if at[axis] == 0 {
                0.0
            } else {
                let mut prev = at;
                prev[axis] -= 1;
                p[axis][prev]
            };
            div += (here - before) / spacing[axis];
        }
        *slot = div;
    }
    out
}

/// The ROF energy `sum |grad u| + sum (u - f)^2 / (2 lambda)`, with the same
/// forward-difference gradient the solver uses.
pub fn tv_energy<T>(u: ArrayView3<'_, f64>, f: ArrayView3<'_, T>, params: &TvParams) -> Result<f64>
where
    T: Copy + Into<f64>,
{
    shapes_agree(u.shape(), f.shape(), "tv_energy")?;
    let u = u.to_owned();
    let gradient = forward_gradient(&u, params.spacing);
    let mut energy = 0.0;
    for (index, &value) in u.indexed_iter() {
        let g = [gradient[0][index], gradient[1][index], gradient[2][index]];
        let residual = value - f[index].into();
        energy += (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt()
            + residual * residual / (2.0 * params.lambda);
    }
    Ok(energy)
}

/// ROF denoising by `iterations` of Chambolle's fixed-point projection:
/// `u = f - lambda div p`, with
/// `p <- (p + tau w) / (1 + tau |w|)`, `w = grad(div p - f / lambda)`.
pub fn tv_denoise_into<T>(
    input: ArrayView3<'_, T>,
    params: &TvParams,
    iterations: usize,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    shapes_agree(input.shape(), out.shape(), "tv_denoise_into")?;
    let f = input.mapv(|value| value.into());
    validate_finite(f.view(), "tv denoise input")?;
    let dim = f.raw_dim();
    let mut p = [
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
    ];
    let inv_lambda = 1.0 / params.lambda;
    for _ in 0..iterations {
        let mut field = backward_divergence(&p, params.spacing);
        ndarray::Zip::from(&mut field)
            .and(&f)
            .for_each(|d, &f| *d -= f * inv_lambda);
        let w = forward_gradient(&field, params.spacing);
        for (index, &w0) in w[0].indexed_iter() {
            let wv = [w0, w[1][index], w[2][index]];
            let norm = (wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2]).sqrt();
            let denominator = 1.0 + params.tau * norm;
            for axis in 0..3 {
                p[axis][index] = (p[axis][index] + params.tau * wv[axis]) / denominator;
            }
        }
    }
    let div = backward_divergence(&p, params.spacing);
    ndarray::Zip::from(&mut out)
        .and(&f)
        .and(&div)
        .for_each(|o, &f, &d| *o = f - params.lambda * d);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::decomposition::{Decomposition, PhaseDecomposition};
    use crate::env::ArrayEnvironment;
    use crate::geometry::BlockGrid;
    use crate::strategy::{execute, Hints, Workflow};

    fn noise(dim: (usize, usize, usize), seed: u64) -> Array3<f64> {
        let mut state = seed | 1;
        Array3::from_shape_fn(dim, |_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        })
    }

    fn models() -> Vec<(&'static str, DiffusionModel)> {
        vec![
            (
                "perona-malik",
                DiffusionModel::PeronaMalik {
                    diffusivity: Diffusivity::PeronaMalik,
                    contrast: 0.5,
                },
            ),
            (
                "edge-enhancing",
                DiffusionModel::EdgeEnhancing {
                    diffusivity: Diffusivity::Weickert,
                    contrast: 0.5,
                },
            ),
            (
                "coherence-enhancing",
                DiffusionModel::CoherenceEnhancing {
                    rho: [2.0, 2.0, 2.0],
                    alpha: 0.01,
                    coherence: 1e-4,
                },
            ),
        ]
    }

    fn params(model: DiffusionModel, spacing: [f64; 3]) -> DiffusionParams {
        let dt = 0.9 * max_stable_dt(&model, spacing);
        DiffusionParams::new(model, [1.0, 1.0, 1.0], spacing, dt, 3.0).unwrap()
    }

    fn run(input: &Array3<f64>, params: &DiffusionParams, iterations: usize) -> Array3<f64> {
        let mut out = Array3::<f64>::zeros(input.raw_dim());
        diffuse_into(input.view(), params, iterations, out.view_mut()).unwrap();
        out
    }

    fn variance(a: &Array3<f64>) -> f64 {
        let mean = a.mean().unwrap();
        a.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / a.len() as f64
    }

    #[test]
    fn diffusivities_are_one_at_zero_and_decreasing() {
        for d in [
            Diffusivity::PeronaMalik,
            Diffusivity::Exponential,
            Diffusivity::Charbonnier,
            Diffusivity::Weickert,
        ] {
            assert_eq!(d.at(0.0, 1.0), 1.0, "{d:?}");
            let mut last = 1.0;
            for step in 1..50 {
                let g = d.at(step as f64 * 0.1, 1.0);
                assert!(g <= last && g >= 0.0, "{d:?} at {step}");
                last = g;
            }
            assert!(last < 0.5, "{d:?} never stops diffusing");
        }
    }

    #[test]
    fn a_constant_image_is_exactly_preserved() {
        let input = Array3::<f64>::from_elem((6, 7, 8), 3.25);
        for (name, model) in models() {
            let p = params(model, [2.0, 1.0, 1.0]);
            assert_eq!(run(&input, &p, 3), input, "{name}");
        }
    }

    #[test]
    fn the_step_bound_is_enforced_and_depends_on_spacing() {
        let model = models()[0].1;
        let bound = max_stable_dt(&model, [1.0, 1.0, 1.0]);
        assert!((bound - 1.0 / 6.0).abs() < 1e-15);
        assert!(DiffusionParams::new(model, [0.0; 3], [1.0; 3], bound * 1.01, 3.0).is_err());
        assert!(max_stable_dt(&model, [2.0, 1.0, 1.0]) > bound);
        let tensor = models()[1].1;
        assert!((max_stable_dt(&tensor, [1.0; 3]) - 2.0 / 15.0).abs() < 1e-15);
    }

    #[test]
    fn with_unit_diffusivity_the_step_is_the_anisotropic_laplacian() {
        // u = x0^2 + x1^2 + x2^2 in physical units has Laplacian 6 everywhere.
        let spacing = [2.0, 1.0, 0.5];
        let model = DiffusionModel::PeronaMalik {
            diffusivity: Diffusivity::Exponential,
            contrast: 1e12,
        };
        let dt = 0.5 * max_stable_dt(&model, spacing);
        let p = DiffusionParams::new(model, [0.0; 3], spacing, dt, 3.0).unwrap();
        let input = Array3::from_shape_fn((7, 7, 7), |(i, j, k)| {
            let x = [
                i as f64 * spacing[0],
                j as f64 * spacing[1],
                k as f64 * spacing[2],
            ];
            x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
        });
        let out = run(&input, &p, 1);
        for i in 1..6 {
            for j in 1..6 {
                for k in 1..6 {
                    let change = (out[[i, j, k]] - input[[i, j, k]]) / dt;
                    assert!((change - 6.0).abs() < 1e-6, "{change} at {i} {j} {k}");
                }
            }
        }
    }

    #[test]
    fn diffusion_conserves_the_mean_and_lowers_the_variance_every_step() {
        let input = noise((8, 9, 10), 7);
        for (name, model) in models() {
            let p = params(model, [1.5, 1.0, 1.0]);
            let mut current = input.clone();
            let mut last = variance(&current);
            for step in 0..5 {
                current = run(&current, &p, 1);
                let now = variance(&current);
                assert!(now < last, "{name}: variance rose at step {step}");
                last = now;
            }
            let drift = (current.mean().unwrap() - input.mean().unwrap()).abs();
            assert!(drift < 1e-12, "{name}: mean drifted by {drift}");
        }
    }

    /// A noisy step along axis 2: the plateaus flatten, the edge stays put.
    #[test]
    fn a_noisy_step_edge_is_denoised_and_does_not_move() {
        let dim = (6, 10, 20);
        let jitter = noise(dim, 11);
        let input = Array3::from_shape_fn(dim, |(i, j, k)| {
            (if k >= 10 { 1.0 } else { 0.0 }) + 0.1 * jitter[[i, j, k]]
        });
        for model in [
            DiffusionModel::PeronaMalik {
                diffusivity: Diffusivity::PeronaMalik,
                contrast: 0.05,
            },
            DiffusionModel::EdgeEnhancing {
                diffusivity: Diffusivity::Weickert,
                contrast: 0.05,
            },
        ] {
            let spacing = [1.0; 3];
            let dt = 0.9 * max_stable_dt(&model, spacing);
            let p = DiffusionParams::new(model, [1.0; 3], spacing, dt, 3.0).unwrap();
            let out = run(&input, &p, 40);
            let plateau_noise = |a: &Array3<f64>| {
                let left = a.slice(ndarray::s![.., .., 1..7]).to_owned();
                variance(&left)
            };
            assert!(
                plateau_noise(&out) < 0.25 * plateau_noise(&input),
                "{model:?}: {} vs {}",
                plateau_noise(&out),
                plateau_noise(&input)
            );
            // The edge: mean profile still crosses one half between k = 9 and 10,
            // and keeps most of its contrast.
            let profile: Vec<f64> = (0..20)
                .map(|k| out.slice(ndarray::s![.., .., k]).mean().unwrap())
                .collect();
            assert!(
                profile[9] < 0.5 && profile[10] > 0.5,
                "{model:?}: {profile:?}"
            );
            assert!(profile[12] - profile[7] > 0.8, "{model:?}: {profile:?}");
        }
    }

    /// A bright sheet (axis 1 = const) with a hole: CED fills the hole along
    /// the sheet much more than it spreads the sheet across.
    #[test]
    fn coherence_enhancing_diffusion_closes_a_gap_in_a_noisy_sheet() {
        let dim = (10, 21, 24);
        let jitter = noise(dim, 5);
        let in_gap = |i: usize, k: usize| (10..14).contains(&k) && (3..7).contains(&i);
        let input = Array3::from_shape_fn(dim, |(i, j, k)| {
            let sheet = j == 10 && !in_gap(i, k);
            (if sheet { 1.0 } else { 0.0 }) + 0.05 * jitter[[i, j, k]]
        });
        let model = DiffusionModel::CoherenceEnhancing {
            rho: [3.0, 3.0, 3.0],
            alpha: 0.001,
            coherence: 1e-6,
        };
        let spacing = [1.0; 3];
        let dt = 0.9 * max_stable_dt(&model, spacing);
        let p = DiffusionParams::new(model, [0.7; 3], spacing, dt, 3.0).unwrap();
        let out = run(&input, &p, 60);
        let gap = |a: &Array3<f64>, j: usize| {
            let mut sum = 0.0;
            let mut n = 0.0;
            for i in 3..7 {
                for k in 10..14 {
                    sum += a[[i, j, k]];
                    n += 1.0;
                }
            }
            sum / n
        };
        let before = gap(&input, 10);
        let after = gap(&out, 10);
        let beside = gap(&out, 12);
        assert!(after > before + 0.1, "gap response {before} -> {after}");
        assert!(after > 2.0 * beside, "on-sheet {after}, beside {beside}");
        // Far from the gap, the sheet stays a sheet.
        assert!(out[[5, 10, 2]] > 3.0 * out[[5, 12, 2]].abs());
    }

    #[test]
    fn the_reach_adds_the_stages() {
        let pm = params(models()[0].1, [1.0; 3]);
        // radius(1, 3) = 3, + gradient + stencil
        assert_eq!(pm.reach(0), 5);
        let ced = params(models()[2].1, [1.0; 3]);
        assert_eq!(ced.reach(1), 3 + 1 + 6 + 1);
    }

    fn plan(
        workflow: &Workflow,
        volume: [usize; 3],
        block: usize,
        axes: &[usize],
    ) -> Decomposition {
        let slots = workflow.chain.slots();
        let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
        let reach = workflow.chain.reach3(&volume);
        let grid = BlockGrid::along(volume, axes, block).unwrap();
        let mut plan = Decomposition {
            volume,
            dtype: workflow.dtype,
            phases: vec![PhaseDecomposition::derive(
                (0..slots.len()).collect(),
                names,
                reach,
                reach,
                grid,
            )],
            chain_reach: reach,
        };
        plan.declare_dtypes(&workflow.chain).unwrap();
        plan
    }

    #[test]
    fn blockwise_steps_equal_the_in_memory_steps() {
        let volume = [12, 26, 28];
        let input = noise((volume[0], volume[1], volume[2]), 3);
        for (name, model) in models() {
            let p = DiffusionParams::new(
                model,
                [0.8, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                0.9 * max_stable_dt(&model, [2.0, 1.0, 1.0]),
                2.5,
            )
            .unwrap();
            for iterations in [1usize, 2] {
                let want = run(&input, &p, iterations);
                let workflow =
                    Workflow::new(diffusion_chain(name, &p, iterations), volume, Dtype::F64);
                for (block, axes) in [(5usize, vec![0]), (9, vec![1, 2]), (7, vec![0, 1, 2])] {
                    let decomposition = plan(&workflow, volume, block, &axes);
                    decomposition.check().unwrap();
                    let env = ArrayEnvironment::new(
                        input.clone().into(),
                        decomposition.n_phases(),
                        [4, 4, 4],
                    )
                    .unwrap();
                    execute(
                        "diffuse",
                        &workflow,
                        &decomposition,
                        &Hints::default(),
                        &env,
                    )
                    .unwrap();
                    let got = env.output().view::<f64>().unwrap().to_owned();
                    let worst = got
                        .iter()
                        .zip(want.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        worst < 1e-12,
                        "{name} x{iterations}, block {block} axes {axes:?}: off by {worst}"
                    );
                }
            }
        }
    }

    #[test]
    fn tv_of_a_step_matches_the_rof_closed_form() {
        // 1-D along axis 2: plateaus of length 4 at 0 and 1. With lambda = 1
        // the ROF solution is 1/4 and 3/4 (each plateau moves by lambda / n).
        let input = Array3::from_shape_fn((2, 3, 8), |(_, _, k)| if k >= 4 { 1.0 } else { 0.0 });
        let p = TvParams::new(1.0, [1.0; 3]).unwrap();
        let mut out = Array3::<f64>::zeros(input.raw_dim());
        tv_denoise_into(input.view(), &p, 3000, out.view_mut()).unwrap();
        for ((_, _, k), &v) in out.indexed_iter() {
            let want = if k >= 4 { 0.75 } else { 0.25 };
            assert!((v - want).abs() < 1e-3, "k {k}: {v}");
        }
    }

    #[test]
    fn tv_lowers_the_rof_energy_and_keeps_constants() {
        let input = noise((6, 8, 9), 13);
        let p = TvParams::new(0.3, [2.0, 1.0, 1.0]).unwrap();
        let mut out = Array3::<f64>::zeros(input.raw_dim());
        tv_denoise_into(input.view(), &p, 200, out.view_mut()).unwrap();
        let before = tv_energy(input.view(), input.view(), &p).unwrap();
        let after = tv_energy(out.view(), input.view(), &p).unwrap();
        assert!(after < 0.9 * before, "{before} -> {after}");
        assert!(variance(&out) < variance(&input));
        let flat = Array3::<f64>::from_elem((3, 4, 5), 2.0);
        let mut out = Array3::<f64>::zeros(flat.raw_dim());
        tv_denoise_into(flat.view(), &p, 20, out.view_mut()).unwrap();
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-12));
    }
}
