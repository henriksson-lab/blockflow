// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! **The structure tensor, and its eigenvalues.**
//!
//! The one feature in Labkit's pixel-classification stack this crate did not
//! have — see `docs/design/pixel-classification.md` for the stack and why it is
//! being built. Everything else in that list is `ops::smooth`, `ops::convolve`,
//! `ops::ridge`, `ops::rank` and `ops::local`.
//!
//! # What it is, and how it differs from the Hessian
//!
//! `ops::ridge` decomposes the **Hessian** — second derivatives of the
//! intensity, which answer "how is the surface curving here". This decomposes
//! the **structure tensor**, the smoothed outer product of the *first*
//! derivatives:
//!
//! ```text
//!     J = G_rho * ( grad I_sigma  x  grad I_sigma ^T )
//! ```
//!
//! which answers a different question: "over a neighbourhood, how consistently
//! do the gradients here point the same way". A single edge and a dense texture
//! have similar gradient *magnitudes* and quite different structure tensors —
//! the edge's eigenvalues are one large and two small, the texture's are three
//! comparable — which is why a classifier wants both and why the two are
//! separate features rather than one.
//!
//! # Two scales, and both are load-bearing
//!
//! * **sigma**, the *derivative* scale: the gradient is taken of the image
//!   smoothed at sigma, so sigma decides what counts as an edge rather than
//!   noise;
//! * **rho**, the *integration* scale: the outer product is averaged over a
//!   neighbourhood of rho, so rho decides how large a region the gradients have
//!   to agree over.
//!
//! They are genuinely independent — that is the whole point of the construction
//! — and a tensor with `rho = 0` has rank one at every voxel, so its second and
//! third eigenvalues are identically zero and two thirds of the feature is a
//! constant. [`StructureTensor::new`] refuses that rather than emitting it.
//!
//! Labkit takes `rho = gamma * sigma` at `gamma = 1, 3`, which is
//! [`StructureTensor::at_gamma`], and is why its stack has **six** structure
//! tensor outputs per scale in 3-D: two integration scales times three
//! eigenvalues.
//!
//! # One eigenvalue per op, and what that costs
//!
//! A `BlockOp` writes one image, so [`StructureTensorOp`] selects which
//! eigenvalue it emits and a caller wanting all three builds three ops. Each
//! recomputes the tensor, so the three cost **three times** what one shared pass
//! would.
//!
//! That is stated rather than optimised, and deliberately: the crate has no
//! multi-image output for a pixel phase, adding one is a change to `Chain` and
//! the image accounting rather than to this file, and
//! `docs/design/pixel-classification.md` says the predictor is expected to
//! dominate this workload by a wide margin. Optimising a term before measuring
//! which term binds is the mistake that document exists to avoid. If a
//! measurement later says the feature stack binds, this is where the saving is.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Slicing};
use crate::reach::Reach;
use crate::voxels::Voxels;

use super::ridge::{
    gaussian_radius, gaussian_smooth_into, gaussian_weights, symmetric_eigenvalues,
};
use super::shapes_agree;

/// Which eigenvalue an op emits, in the **descending** order
/// [`symmetric_eigenvalues`] returns.
///
/// Named rather than indexed because the index is only meaningful next to that
/// ordering, and a caller who writes `2` and means "the largest" gets a plausible
/// wrong feature rather than an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Eigenvalue {
    /// The largest. Along a strong edge this is the one that carries it.
    Largest,
    Middle,
    /// The smallest. Large only where the gradients disagree in every
    /// direction, which is what distinguishes texture from structure.
    Smallest,
}

impl Eigenvalue {
    /// Every one, in the order they are indexed. For a caller building the
    /// three-op family, and for a test that must not silently cover two.
    pub const ALL: [Eigenvalue; 3] = [
        Eigenvalue::Largest,
        Eigenvalue::Middle,
        Eigenvalue::Smallest,
    ];

    /// Its position in [`symmetric_eigenvalues`]'s descending order.
    ///
    /// `pub(super)` rather than private because `ops::ridge`'s
    /// [`HessianEigenvalueOp`](super::ridge::HessianEigenvalueOp) selects with
    /// the same enum over the same decomposition — the two ops differ in which
    /// matrix they build, not in how its eigenvalues are ordered, and a second
    /// mapping would be a second chance to disagree.
    pub(super) fn index(self) -> usize {
        match self {
            Eigenvalue::Largest => 0,
            Eigenvalue::Middle => 1,
            Eigenvalue::Smallest => 2,
        }
    }

    /// The name this eigenvalue goes by in a feature stack's channel list.
    pub fn as_str(self) -> &'static str {
        match self {
            Eigenvalue::Largest => "largest",
            Eigenvalue::Middle => "middle",
            Eigenvalue::Smallest => "smallest",
        }
    }
}

/// The two scales and the truncation, validated once.
#[derive(Debug, Clone, PartialEq)]
pub struct StructureTensor {
    sigma: [f64; 3],
    rho: [f64; 3],
    truncate: f64,
}

impl StructureTensor {
    /// The derivative scale, the integration scale, and where the Gaussians are
    /// cut off.
    ///
    /// **`rho` must be positive on at least one axis.** At `rho = 0` the tensor
    /// is an outer product of a vector with itself at every voxel, which has
    /// rank one: the second and third eigenvalues are identically zero, and two
    /// of the three features this op can emit are a constant image. That is not
    /// a degenerate case worth supporting quietly — it is a caller who meant
    /// [`GradientMagnitudeOp`], which computes the rank-one tensor's only
    /// non-trivial invariant directly and for a third of the work.
    pub fn new(sigma: [f64; 3], rho: [f64; 3], truncate: f64) -> Result<Self> {
        for (name, scale) in [("sigma", sigma), ("rho", rho)] {
            for axis in 0..3 {
                if !scale[axis].is_finite() || scale[axis] < 0.0 {
                    return Err(Error::InvalidArgument(format!(
                        "structure tensor: {name}[{axis}] is {}; a scale is a non-negative \
                         finite number of voxels",
                        scale[axis]
                    )));
                }
            }
        }
        if rho.iter().all(|&r| r == 0.0) {
            return Err(Error::InvalidArgument(
                "structure tensor: the integration scale is zero on every axis, so the tensor \
                 is an outer product of one gradient with itself — rank one, with its second \
                 and third eigenvalues identically zero. The gradient magnitude is that \
                 tensor's only non-trivial invariant, and `GradientMagnitudeOp` computes it \
                 directly."
                    .to_string(),
            ));
        }
        if !truncate.is_finite() || truncate <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "structure tensor: truncate is {truncate}; it is how many standard deviations \
                 the Gaussian is cut off at and must be positive"
            )));
        }
        Ok(Self {
            sigma,
            rho,
            truncate,
        })
    }

    /// Labkit's parameterisation: the integration scale is a multiple of the
    /// derivative scale.
    ///
    /// `gamma = 1` and `gamma = 3` are the two that stack uses, which is where
    /// its six outputs per scale in 3-D come from.
    pub fn at_gamma(sigma: [f64; 3], gamma: f64, truncate: f64) -> Result<Self> {
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "structure tensor: gamma is {gamma}; the integration scale is gamma times the \
                 derivative scale and must be positive"
            )));
        }
        Self::new(
            sigma,
            [sigma[0] * gamma, sigma[1] * gamma, sigma[2] * gamma],
            truncate,
        )
    }

    pub fn sigma(&self) -> [f64; 3] {
        self.sigma
    }

    pub fn rho(&self) -> [f64; 3] {
        self.rho
    }

    pub fn truncate(&self) -> f64 {
        self.truncate
    }

    /// **What one voxel of output reads, per axis — and it is a sum, not a
    /// maximum.**
    ///
    /// Three stages compose here and each consumes the one before it further
    /// out: the derivative smoothing reaches `radius(sigma)`, the central
    /// difference on top of it reaches one more, and the integration smoothing
    /// reaches `radius(rho)` beyond that. So they add.
    ///
    /// This is the opposite of `ridge::Scales::reach`, which takes a *maximum*
    /// over its scales — and the difference is not arithmetic taste. There the
    /// scales are alternatives folded by a maximum, the same distinction `Chain`
    /// draws between `Alternative` and `Sequence`; here they are stages applied
    /// in turn. Getting it wrong by taking the maximum would under-declare the
    /// reach and produce a plausible wrong volume at every block seam, which is
    /// the failure `docs/design/BLOCK_OPS.md` exists to remove.
    pub fn reach(&self, axis: usize) -> usize {
        gaussian_radius(self.sigma[axis], self.truncate)
            + 1
            + gaussian_radius(self.rho[axis], self.truncate)
    }

    /// The eigenvalues of the tensor at every voxel, all three, into three
    /// buffers.
    ///
    /// The form that computes the tensor **once**. [`StructureTensorOp`] emits
    /// one of them and therefore pays three times over for the three; this is
    /// here so that a caller who wants all three outside the op machinery — a
    /// test, or a future multi-output phase — does not have to.
    pub fn eigenvalues_into<T>(
        &self,
        input: ArrayView3<'_, T>,
        out: [ArrayViewMut3<'_, f64>; 3],
    ) -> Result<()>
    where
        T: Copy + Into<f64>,
    {
        let what = "structure_tensor";
        let mut out = out;
        for slot in out.iter() {
            shapes_agree(input.shape(), slot.shape(), what)?;
        }
        let dim = (input.shape()[0], input.shape()[1], input.shape()[2]);

        // 1. The image at the derivative scale.
        let mut smoothed = Array3::<f64>::zeros(dim);
        let derivative = [
            gaussian_weights(self.sigma[0], self.truncate)?,
            gaussian_weights(self.sigma[1], self.truncate)?,
            gaussian_weights(self.sigma[2], self.truncate)?,
        ];
        gaussian_smooth_into(input, &derivative, smoothed.view_mut())?;

        // 2. The six independent products of its gradient, in the order
        //    `symmetric_eigenvalues` reads: `[xx, yy, zz, xy, xz, yz]`. The same
        //    order `ridge::hessian_at` returns, so the two cannot drift.
        //
        //    The gradient is a central difference of the smoothed field, which
        //    is the same construction `ridge` takes for its second differences
        //    and clamps at the buffer edge the same way — the clamp being
        //    deliberately wrong at a block seam and made right by the halo.
        let mut product = [
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
        ];
        for i in 0..dim.0 {
            for j in 0..dim.1 {
                for k in 0..dim.2 {
                    let gradient = gradient_at(smoothed.view(), [i, j, k]);
                    product[0][[i, j, k]] = gradient[0] * gradient[0];
                    product[1][[i, j, k]] = gradient[1] * gradient[1];
                    product[2][[i, j, k]] = gradient[2] * gradient[2];
                    product[3][[i, j, k]] = gradient[0] * gradient[1];
                    product[4][[i, j, k]] = gradient[0] * gradient[2];
                    product[5][[i, j, k]] = gradient[1] * gradient[2];
                }
            }
        }

        // 3. Each component averaged over the integration scale. **This is the
        //    step that makes it a structure tensor** rather than an outer
        //    product: without it the matrix has rank one everywhere.
        let integration = [
            gaussian_weights(self.rho[0], self.truncate)?,
            gaussian_weights(self.rho[1], self.truncate)?,
            gaussian_weights(self.rho[2], self.truncate)?,
        ];
        let mut integrated = [
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
        ];
        for component in 0..6 {
            gaussian_smooth_into(
                product[component].view(),
                &integration,
                integrated[component].view_mut(),
            )?;
        }

        // 4. The decomposition, which is `ridge`'s. One closed form in the
        //    crate, and its documented care about repeated roots and scaling is
        //    care this op inherits rather than repeats.
        for i in 0..dim.0 {
            for j in 0..dim.1 {
                for k in 0..dim.2 {
                    let tensor = [
                        integrated[0][[i, j, k]],
                        integrated[1][[i, j, k]],
                        integrated[2][[i, j, k]],
                        integrated[3][[i, j, k]],
                        integrated[4][[i, j, k]],
                        integrated[5][[i, j, k]],
                    ];
                    let values = symmetric_eigenvalues(tensor);
                    for which in 0..3 {
                        out[which][[i, j, k]] = values[which];
                    }
                }
            }
        }
        Ok(())
    }
}

/// **The first central difference of a field, at one voxel, clamped at the
/// buffer edge.**
///
/// The exact counterpart of `ridge::hessian_at`, one derivative down, and
/// clamped for the same reason: the clamp is *deliberately wrong* at a block
/// seam and is made right by the halo, so a block computed with a correct halo
/// never reaches a clamped sample. A boundary rule that were correct in
/// isolation — reflection, say — would produce a different answer at the volume
/// edge depending on where the block boundary fell, which is exactly the failure
/// this crate is built to exclude.
///
/// The factor of a half is the central difference's, not a normalisation: over a
/// ramp of slope `a` this returns `a`.
pub fn gradient_at(field: ArrayView3<'_, f64>, at: [usize; 3]) -> [f64; 3] {
    let extent = [
        field.shape()[0] as isize,
        field.shape()[1] as isize,
        field.shape()[2] as isize,
    ];
    let here = [at[0] as isize, at[1] as isize, at[2] as isize];
    let mut gradient = [0.0f64; 3];
    for axis in 0..3 {
        let sample = |step: isize| -> f64 {
            let mut place = [at[0], at[1], at[2]];
            place[axis] = (here[axis] + step).clamp(0, extent[axis] - 1) as usize;
            field[place]
        };
        gradient[axis] = (sample(1) - sample(-1)) * 0.5;
    }
    gradient
}

/// One eigenvalue of the structure tensor, as an op.
pub struct StructureTensorOp {
    name: &'static str,
    tensor: StructureTensor,
    which: Eigenvalue,
    cost: f64,
}

impl StructureTensorOp {
    pub fn new(name: &'static str, tensor: StructureTensor, which: Eigenvalue) -> Self {
        let cost = cost_for(&tensor);
        Self {
            name,
            tensor,
            which,
            cost,
        }
    }

    pub fn tensor(&self) -> &StructureTensor {
        &self.tensor
    }

    pub fn which(&self) -> Eigenvalue {
        self.which
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for StructureTensorOp {
    /// **A stencil.** Two separable convolutions and a difference, all at fixed
    /// offsets, and nothing carried along the scan.
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.tensor.reach(axis)
    }

    /// Symmetric, because a Gaussian is and a central difference is. Stated
    /// through `reach_spec` anyway rather than left to the default, so that the
    /// per-side form is the one the planner folds.
    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::symmetric([
            self.tensor.reach(0),
            self.tensor.reach(1),
            self.tensor.reach(2),
        ])
    }

    /// Every element type but `F16`, as `ops::ridge` accepts: the op reads and
    /// accumulates, and what it accumulates in is its own business.
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    /// `f64` whatever came in. An eigenvalue of a smoothed outer product of
    /// differences is fractional and unbounded above even for a `u8` volume;
    /// narrowing it would be inventing a quantisation nobody asked for.
    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let dim = (out.shape()[0], out.shape()[1], out.shape()[2]);
        let mut values = [
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
        ];
        {
            let [first, second, third] = &mut values;
            let slots = [first.view_mut(), second.view_mut(), third.view_mut()];
            // The same dispatch `ops::ridge` takes, and for the same reason: the
            // types that are already narrower than `f64` go straight through,
            // and the three that are not — `bool`, and the 64-bit integers,
            // whose `Into<f64>` does not exist because it would be lossy — go
            // through the widening `Voxels` already knows how to do.
            macro_rules! direct {
                ($type:ty) => {
                    self.tensor.eigenvalues_into(input.view::<$type>()?, slots)
                };
            }
            match input.dtype() {
                Dtype::U8 => direct!(u8),
                Dtype::U16 => direct!(u16),
                Dtype::U32 => direct!(u32),
                Dtype::I8 => direct!(i8),
                Dtype::I16 => direct!(i16),
                Dtype::I32 => direct!(i32),
                Dtype::F32 => direct!(f32),
                Dtype::F64 => direct!(f64),
                Dtype::Bool | Dtype::U64 | Dtype::I64 => {
                    let widened = input.widened();
                    self.tensor.eigenvalues_into(widened.view(), slots)
                }
                Dtype::F16 => Err(Error::InvalidArgument(format!(
                    "{}: no buffer holds half-precision; `accepts` refuses it before a run starts",
                    self.name
                ))),
            }?;
        }
        let chosen = &values[self.which.index()];
        let mut out = out.view_mut::<f64>()?;
        ndarray::Zip::from(&mut out)
            .and(chosen)
            .for_each(|slot, &value| *slot = value);
        Ok(())
    }

    /// **Zero, for every constant.** A constant field has no gradient, so the
    /// tensor is zero and so is every eigenvalue of it — and the clamp does not
    /// disturb that, because a clamped neighbour of a constant field is the same
    /// constant. This is the declaration that lets a uniform block be skipped.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        Some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

// ------------------------------------------------- the gradient magnitude --

/// **`|grad G_sigma * I|`** — Labkit's "Gaussian gradient magnitude", and this
/// module's rather than `ops::convolve`'s for a reason the mathematics gives:
/// it is the one non-trivial invariant of the structure tensor at `rho = 0`,
/// where [`StructureTensor::new`] refuses to build. A caller who asks for that
/// degenerate tensor is sent here, so here is where it has to live.
///
/// Not expressible as a [`ConvolveOp`](super::convolve::ConvolveOp) either,
/// which is the other place it might have gone: a magnitude is a square root of
/// a sum of squares of three linear filters, and a `Kernel` is one linear
/// filter. Three convolutions joined by a combine would have been the other
/// shape, at three block-sized intermediates and a fan-in, against one pass
/// here.
pub struct GradientMagnitudeOp {
    name: &'static str,
    sigma: [f64; 3],
    truncate: f64,
    cost: f64,
}

impl GradientMagnitudeOp {
    pub fn new(name: &'static str, sigma: [f64; 3], truncate: f64) -> Result<Self> {
        // Routed through `StructureTensor` so the two ops cannot disagree about
        // what a valid scale is — with `rho` set to `sigma` purely to get past
        // the rank-one refusal, since nothing here integrates.
        let validated = StructureTensor::new(sigma, sigma, truncate)?;
        Ok(Self {
            name,
            sigma: validated.sigma,
            truncate: validated.truncate,
            cost: gradient_magnitude_cost_for(&validated),
        })
    }

    pub fn sigma(&self) -> [f64; 3] {
        self.sigma
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// `radius(sigma) + 1` per axis: the smoothing, then the stencil on top of
    /// it. No integration scale, which is exactly what makes this cheaper than
    /// [`StructureTensorOp`] rather than merely a different fold of it.
    pub fn reach_on(&self, axis: usize) -> usize {
        gaussian_radius(self.sigma[axis], self.truncate) + 1
    }

    fn magnitude_into<T>(
        &self,
        input: ArrayView3<'_, T>,
        mut out: ArrayViewMut3<'_, f64>,
    ) -> Result<()>
    where
        T: Copy + Into<f64>,
    {
        shapes_agree(input.shape(), out.shape(), "gradient_magnitude")?;
        let dim = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut smoothed = Array3::<f64>::zeros(dim);
        let kernels = [
            gaussian_weights(self.sigma[0], self.truncate)?,
            gaussian_weights(self.sigma[1], self.truncate)?,
            gaussian_weights(self.sigma[2], self.truncate)?,
        ];
        gaussian_smooth_into(input, &kernels, smoothed.view_mut())?;
        for i in 0..dim.0 {
            for j in 0..dim.1 {
                for k in 0..dim.2 {
                    let gradient = gradient_at(smoothed.view(), [i, j, k]);
                    out[[i, j, k]] = (gradient[0] * gradient[0]
                        + gradient[1] * gradient[1]
                        + gradient[2] * gradient[2])
                        .sqrt();
                }
            }
        }
        Ok(())
    }
}

impl BlockOp for GradientMagnitudeOp {
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.reach_on(axis)
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::symmetric([self.reach_on(0), self.reach_on(1), self.reach_on(2)])
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let out = out.view_mut::<f64>()?;
        macro_rules! direct {
            ($type:ty) => {
                self.magnitude_into(input.view::<$type>()?, out)
            };
        }
        match input.dtype() {
            Dtype::U8 => direct!(u8),
            Dtype::U16 => direct!(u16),
            Dtype::U32 => direct!(u32),
            Dtype::I8 => direct!(i8),
            Dtype::I16 => direct!(i16),
            Dtype::I32 => direct!(i32),
            Dtype::F32 => direct!(f32),
            Dtype::F64 => direct!(f64),
            Dtype::Bool | Dtype::U64 | Dtype::I64 => {
                let widened = input.widened();
                self.magnitude_into(widened.view(), out)
            }
            Dtype::F16 => Err(Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; `accepts` refuses it before a run starts",
                self.name
            ))),
        }
    }

    /// Zero for every constant, and exactly — a magnitude of a gradient that is
    /// exactly zero is `sqrt(0)`, which is `0` and not merely near it.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        Some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// One smoothing pass rather than seven, and the same fixed slab minus the
/// eigen-decomposition — which is most of it.
///
/// The slab is charged at a **quarter** of the tensor's. That fraction is
/// derived rather than measured: `ops::cost`'s ridge row puts the decomposition
/// and its response at roughly three quarters of the per-voxel work of an op of
/// this shape, and what is left here is the three differences, three multiplies
/// and a `sqrt`. It is the one number in this module that is an estimate, and it
/// is marked so that a measurement can replace it; the error it can cause is
/// bounded by the term being a fifth of a feature stack's total cost.
fn gradient_magnitude_cost_for(tensor: &StructureTensor) -> f64 {
    let taps: usize = (0..3)
        .map(|axis| 2 * gaussian_radius(tensor.sigma[axis], tensor.truncate) + 1)
        .sum();
    STRUCTURE_TENSOR_COST_PER_TAP * taps as f64 + STRUCTURE_TENSOR_VOXEL_COST * 0.25
}

/// Measured; see [`COST_MEASUREMENT`] below for the run and the fit.
///
/// **The two scales are not symmetric, and a model that made them so would be
/// wrong by a factor of six.** The derivative smoothing runs once, on the
/// intensity; the integration smoothing runs *six times*, once per independent
/// component of the tensor. So the tap count charged is
/// `taps(sigma) + 6 * taps(rho)`, and widening the integration scale is much the
/// more expensive of the two things a caller can do. Labkit's `gamma = 3` widens
/// exactly that one, which is why this is worth getting right rather than
/// folding into a single number.
///
/// On top of the taps sits a fixed per-voxel slab that no kernel width changes:
/// the gradient stencil, the six products, and the eigen-decomposition. The last
/// dominates it — it is a cubic root-finder — and at every scale a feature stack
/// actually uses, the slab is the larger of the two terms.
pub(super) fn cost_for(tensor: &StructureTensor) -> f64 {
    let taps = |scale: [f64; 3]| -> usize {
        (0..3)
            .map(|axis| 2 * gaussian_radius(scale[axis], tensor.truncate) + 1)
            .sum()
    };
    STRUCTURE_TENSOR_COST_PER_TAP * (taps(tensor.sigma) + 6 * taps(tensor.rho)) as f64
        + STRUCTURE_TENSOR_VOXEL_COST
}

/// Measured; see [`COST_MEASUREMENT`]. One tap of one separable pass, relative
/// to a voxelwise map.
///
/// **Not `ridge::SMOOTH_COST_PER_TAP`, although the pass being counted is
/// literally the same function.** That figure is 0.79 and this one is 0.0567,
/// and the gap is not a disagreement about the code — it is drift, quantified
/// in [`COST_MEASUREMENT`]. Sharing the constant would have been tidier and
/// would have mispriced this op by an order of magnitude.
pub(super) const STRUCTURE_TENSOR_COST_PER_TAP: f64 = 0.0567;

/// Measured; see [`COST_MEASUREMENT`]. The gradient stencil, the six products
/// and the eigenvalues at one voxel, relative to a voxelwise map.
///
/// Larger than `ridge`'s stored [`super::ridge::DECOMPOSITION_COST`] of 41.2, and that
/// comparison is worth stating carefully because it is not the one it looks
/// like. Against ridge's *stored* slab this is 1.38x; against ridge's slab as
/// **measured in the same run**, 56.60 in these units, it is 1.004. The two ops
/// spend nearly the same time per voxel below the smoothing, which is what one
/// would expect of the same closed form over the same six numbers — this op
/// forms six products where ridge takes six second differences, and ridge
/// evaluates three exponentials for its response where this one copies out a
/// number. The 1.38x is ridge's stored split between its two constants being
/// stale, not a difference between the ops.
pub(super) const STRUCTURE_TENSOR_VOXEL_COST: f64 = 56.9;

/// The measurement the two constants above came from, kept as text so a re-run
/// elsewhere can be compared against it rather than merely replacing it. Taken
/// through `ops::cost::report`, which is where the three structure tensor rows
/// and the ridge row beside them live.
///
/// ```text
/// op cost, 96x64x64, best of 5
/// op                                             ns/voxel   relative   per element
/// voxelwise map (the unit)                          0.991       1.00         1.000
/// gaussian smooth, sigma 1 truncate 3 (21 taps)     6.494       6.55         0.312
/// gaussian smooth, sigma 2 truncate 3 (39 taps)    11.686      11.79         0.302
/// ridge, sigmas [1.0], truncate 3 (21 taps)       244.445     246.61        11.744
/// structure tensor, sigma 1 rho 1 (147 charged)   276.090     278.54         1.895
/// structure tensor, sigma 3 rho 1 (183 charged)   283.900     286.42         1.565
/// structure tensor, sigma 1 rho 3 (363 charged)   327.558     330.47         0.910
/// ```
///
/// **The shape first, because a constant fitted to the wrong shape is worse
/// than no constant.** A least squares line through the three structure tensor
/// rows against the charged tap count `taps(sigma) + 6 taps(rho)` fits with
/// residuals of `+0.37, -0.45, +0.07` ns on values near 300 — 0.15%, which for
/// a timing measurement is as close to exact as the instrument goes. The
/// six-fold multiplier is what makes that fit: taken as raw slopes, widening the
/// derivative scale by 36 taps costs 0.217 ns/tap and widening the integration
/// scale by the same 36 costs 1.430, a ratio of **6.59**. Three runs of this
/// table gave 4.95, 5.00 and 6.59 for that ratio; the model's 6 is the derived
/// value — six independent components — and the spread is the measurement's,
/// not the model's.
///
/// The fitted line is `0.2397 ns/tap` and a slab of `240.49 ns`. The slope can
/// be checked against something independent: the two smoothing rows above give
/// `(11.686 - 6.494) / 18 = 0.288 ns/tap` for the same function called on its
/// own. 0.240 against 0.288 is 17%, and in the direction one would expect — the
/// tensor's seven passes amortise the per-call scratch allocation over more
/// work than a single `SmoothOp` does.
///
/// **The anchoring, and why it is on the total rather than on the per-tap
/// figure.** These constants are consumed by `CostModel`, which applies one
/// `compute_scale` across every op family, so what has to be right is this op's
/// cost *relative to its siblings* — `ops::mod`'s `COST_MEASUREMENT` is explicit
/// that a systematic factor is absorbed by `statistics::calibrate` and a
/// relative one is not. The stored constants elsewhere in this module were taken
/// when the voxelwise map cost 6.05 ns; it now costs 0.991, and the families
/// have not all drifted by the same factor. So the ridge row exists: ridge at
/// sigma 1 is stored as `0.79 * 21 + 41.2 = 57.79` and measured at 244.445 ns,
/// which fixes the conversion at `0.2364` stored units per nanosecond. Under it
/// the fitted line becomes `0.0567` per tap and `56.85` for the slab, and
/// `cost_for` at sigma 1 rho 1 comes to 65.2 against ridge's 57.79 — a ratio of
/// 1.128, where the two rows were measured 1.130 apart. That agreement is the
/// point of the exercise.
///
/// **What this records but does not fix.** Applying the same conversion to
/// ridge's own measured row puts its slab at 56.60 stored units against the 41.2
/// it stores, so ridge's split between its per-tap and per-voxel constants no
/// longer matches its own timings — its total is right and its shape has drifted.
/// Correcting it would move every plan the committed `costs/` scenarios were
/// recorded against, which is a change to make deliberately with those
/// regressions in view rather than as a side effect of adding an op.
pub const COST_MEASUREMENT: &str = "ops::cost::report";

#[cfg(test)]
mod tests {
    use super::*;

    use crate::op::Anchor;
    use crate::ops::ridge::DECOMPOSITION_COST;

    /// Deterministic, and not `rand`: a test that only passes for one seed is
    /// not testing the property it names.
    fn noise(dim: (usize, usize, usize), seed: u64) -> Array3<f64> {
        let mut state = seed | 1;
        Array3::from_shape_fn(dim, |_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        })
    }

    fn eigenvalues(tensor: &StructureTensor, input: ArrayView3<'_, f64>) -> [Array3<f64>; 3] {
        let dim = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut out = [
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
            Array3::<f64>::zeros(dim),
        ];
        {
            let [first, second, third] = &mut out;
            tensor
                .eigenvalues_into(
                    input,
                    [first.view_mut(), second.view_mut(), third.view_mut()],
                )
                .expect("the fixtures agree in shape");
        }
        out
    }

    /// **The closed form.** Over a linear ramp `I = a x` the gradient is exactly
    /// `(a, 0, 0)` at every voxel — a Gaussian preserves a ramp, because its
    /// weights sum to one and its first moment is zero, and a central difference
    /// of a ramp is its slope — so the tensor is `diag(a^2, 0, 0)` and stays
    /// that under any integration scale. Eigenvalues `[a^2, 0, 0]`.
    ///
    /// The claim is made only on the interior, `reach` voxels in from each end
    /// of the ramp axis, because the clamp at the buffer edge is deliberately
    /// not the ramp. Along the other two axes the field is constant, and a
    /// clamped neighbour of a constant is the same constant, so no margin is
    /// needed there — which is why the fixture can be thin.
    ///
    /// **The two zeros are held to `sqrt(eps)` and not to `eps`, and that is the
    /// documented behaviour of the solver rather than slack.**
    /// [`symmetric_eigenvalues`] says it keeps about half the mantissa at a
    /// repeated root, because `acos` is evaluated at an endpoint of its domain
    /// where its derivative is unbounded. A rank-one tensor has the double root
    /// zero, so it is exactly that case — and it is not an exotic one *here*:
    /// every locally one-dimensional structure, which is what an edge is and
    /// what this feature is largely for, produces a near-repeated pair. The
    /// simple root is unaffected and is checked to `1e-9`. Measured on this
    /// fixture: the two zeros come back at `2.3e-9` against a largest of
    /// `0.5625`, which is `4e-9` relative — in line with the `1.5e-8` that
    /// function records.
    #[test]
    fn a_ramp_has_one_eigenvalue_and_it_is_the_slope_squared() {
        let slope = 0.75;
        let dim = (25, 5, 5);
        let input = Array3::from_shape_fn(dim, |(i, _, _)| slope * i as f64);
        let tensor = StructureTensor::new([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], 3.0).unwrap();
        assert_eq!(tensor.reach(0), 10);

        let values = eigenvalues(&tensor, input.view());
        let repeated_root = f64::EPSILON.sqrt() * slope * slope;
        let mut checked = 0;
        for i in tensor.reach(0)..dim.0 - tensor.reach(0) {
            for j in 0..dim.1 {
                for k in 0..dim.2 {
                    assert!(
                        (values[0][[i, j, k]] - slope * slope).abs() < 1e-9,
                        "largest at {i},{j},{k} is {}",
                        values[0][[i, j, k]]
                    );
                    for which in [1, 2] {
                        assert!(
                            values[which][[i, j, k]].abs() < repeated_root,
                            "the {} eigenvalue of a rank-one tensor is {:e} at {i},{j},{k}, \
                             beyond what the closed form loses at a repeated root",
                            Eigenvalue::ALL[which].as_str(),
                            values[which][[i, j, k]]
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 0,
            "the interior was empty and nothing was checked"
        );
    }

    /// **The feature answers**, and answers the question it is for rather than
    /// the gradient magnitude's.
    ///
    /// A step edge and a noise field are told apart by the *coherence* of their
    /// gradients — `smallest / largest` — where a gradient magnitude feature,
    /// which sees only the trace, could not tell them apart at all.
    ///
    /// The two fixtures have quite different gradient energy, 26x apart, and
    /// that cannot explain the result: coherence is a ratio of two eigenvalues
    /// of a form quadratic in the intensity, so scaling the input scales both
    /// and leaves it fixed. Rather than trust that argument, the test scales one
    /// fixture by 5 and shows the coherence does not move — which is the same
    /// claim, checked.
    #[test]
    fn a_coherent_edge_and_an_incoherent_texture_are_told_apart() {
        let dim = (21, 21, 21);
        let tensor = StructureTensor::new([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], 2.0).unwrap();
        let at = [10, 10, 10];
        let coherence = |values: &[Array3<f64>; 3]| values[2][at] / values[0][at];

        let edge = Array3::from_shape_fn(dim, |(i, _, _)| if i < 10 { 0.0 } else { 1.0 });
        let texture = noise(dim, 20260831);

        let edge_coherence = coherence(&eigenvalues(&tensor, edge.view()));
        let texture_coherence = coherence(&eigenvalues(&tensor, texture.view()));

        assert!(
            edge_coherence < 1e-6,
            "a step edge's gradients all point one way, so its smallest eigenvalue must \
             vanish beside its largest; got {edge_coherence:e}"
        );
        assert!(
            texture_coherence > 0.05,
            "noise has no preferred direction, so all three eigenvalues must be \
             comparable; got {texture_coherence:e}"
        );

        // The contrast between the fixtures is not what produced that gap.
        let brighter = texture.mapv(|value| value * 5.0);
        let brighter = eigenvalues(&tensor, brighter.view());
        assert!(
            (coherence(&brighter) - texture_coherence).abs() < 1e-9,
            "coherence moved with contrast, so it is not the invariant this test claims"
        );
        // And the scaling did change the tensor, so the line above is not
        // comparing a fixture with itself.
        assert!(brighter[0][at] > 20.0 * eigenvalues(&tensor, texture.view())[0][at]);
    }

    /// **Relabelling the axes cannot change an eigenvalue**, because permuting
    /// them is a similarity transform of the tensor. Both the volume and the two
    /// scales are permuted together — permuting only one would be a different
    /// filter, and the test would be asserting nothing.
    ///
    /// This is the check that the six products are paired with the six slots
    /// `symmetric_eigenvalues` reads in the order it reads them. Swapping `xy`
    /// with `xz` passes the ramp oracle above, because a ramp's cross terms are
    /// all zero; it does not pass this.
    #[test]
    fn an_axis_relabelling_does_not_change_the_eigenvalues() {
        let dim = (13, 11, 9);
        let input = noise(dim, 4711);
        let sigma = [1.0, 0.7, 1.3];
        let rho = [2.0, 1.5, 1.1];

        let straight = eigenvalues(
            &StructureTensor::new(sigma, rho, 2.0).unwrap(),
            input.view(),
        );
        // `permuted_axes([2, 0, 1])` makes `turned[i, j, k] == input[j, k, i]`,
        // so the scale that was on the input's axis 2 is now on axis 0.
        let turned = input.view().permuted_axes([2, 0, 1]).to_owned();
        let rolled = eigenvalues(
            &StructureTensor::new(
                [sigma[2], sigma[0], sigma[1]],
                [rho[2], rho[0], rho[1]],
                2.0,
            )
            .unwrap(),
            turned.view(),
        );

        let mut largest_gap = 0.0f64;
        for i in 0..dim.2 {
            for j in 0..dim.0 {
                for k in 0..dim.1 {
                    for which in 0..3 {
                        largest_gap = largest_gap
                            .max((rolled[which][[i, j, k]] - straight[which][[j, k, i]]).abs());
                    }
                }
            }
        }
        assert!(
            largest_gap < 1e-12,
            "relabelling the axes moved an eigenvalue by {largest_gap:e}"
        );
        // And the fixture has something to move: a volume of zeros would pass
        // the line above.
        assert!(straight[0].iter().cloned().fold(0.0, f64::max) > 1e-6);
    }

    /// **The reach is the sum of the three stages, not the maximum over them.**
    ///
    /// Held here against the radii it is built from, so that a change to either
    /// Gaussian's truncation shows up as this test rather than as wrong values
    /// at a seam. `tests/structure_tensor.rs` holds the behavioural half — that
    /// one voxel less is visibly wrong.
    #[test]
    fn the_reach_adds_the_derivative_scale_the_stencil_and_the_integration_scale() {
        let tensor = StructureTensor::new([1.0, 2.0, 4.0], [3.0, 1.0, 0.5], 3.0).unwrap();
        for axis in 0..3 {
            let derivative = gaussian_radius(tensor.sigma()[axis], 3.0);
            let integration = gaussian_radius(tensor.rho()[axis], 3.0);
            assert_eq!(
                tensor.reach(axis),
                derivative + 1 + integration,
                "axis {axis}"
            );
            // The maximum an alternatives-fold would have given, which is what
            // this must not be. On every axis of this fixture the two differ.
            assert!(
                tensor.reach(axis) > derivative.max(integration) + 1,
                "axis {axis}"
            );
        }
    }

    /// A constant field has no gradient, so every eigenvalue is zero — including
    /// at the clamped edge, where the clamped neighbour of a constant is the
    /// same constant. This is what `constant_maps_to` promises the executor.
    #[test]
    fn a_constant_is_zero_everywhere_including_the_edges() {
        let tensor = StructureTensor::new([1.5, 1.5, 1.5], [2.0, 2.0, 2.0], 3.0).unwrap();
        for constant in [0.0, 0.25, -3.5, 1e6] {
            let input = Array3::from_elem((9, 8, 7), constant);
            for values in eigenvalues(&tensor, input.view()) {
                assert!(
                    values.iter().all(|&value| value == 0.0),
                    "a constant {constant} did not give exactly zero"
                );
            }
        }
    }

    /// **The gradient magnitude's closed form.** Over `I = a x + b y` the
    /// smoothed gradient is exactly `(a, b, 0)`, so the magnitude is
    /// `sqrt(a^2 + b^2)` — a number that depends on both components, unlike the
    /// single-axis ramp, so a magnitude that had dropped a term would show.
    ///
    /// Exact to `1e-12` and not to `sqrt(eps)`: there is no eigen-decomposition
    /// here and therefore no repeated root to lose digits at, which is the whole
    /// reason this op is cheaper than the tensor it is an invariant of.
    #[test]
    fn the_gradient_magnitude_of_a_tilted_plane_is_the_slope_of_the_tilt() {
        let (a, b) = (0.75, -0.4);
        let dim = (21, 21, 5);
        let input = Array3::from_shape_fn(dim, |(i, j, _)| a * i as f64 + b * j as f64);
        let op = GradientMagnitudeOp::new("grad", [1.0, 1.0, 1.0], 3.0).unwrap();
        assert_eq!(op.reach_on(0), 4);

        let source: Voxels = input.into();
        let mut out = Voxels::zeros(Dtype::F64, [dim.0, dim.1, dim.2]).unwrap();
        op.apply(&source, &mut out, &Anchor::whole([dim.0, dim.1, dim.2]))
            .unwrap();
        let got = out.view::<f64>().unwrap();

        let want = (a * a + b * b).sqrt();
        let mut checked = 0;
        for i in op.reach_on(0)..dim.0 - op.reach_on(0) {
            for j in op.reach_on(1)..dim.1 - op.reach_on(1) {
                for k in 0..dim.2 {
                    assert!(
                        (got[[i, j, k]] - want).abs() < 1e-12,
                        "{} at {i},{j},{k}, want {want}",
                        got[[i, j, k]]
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0);
    }

    /// **The magnitude is the tensor's rank-one invariant**, which is the claim
    /// [`StructureTensor::new`]'s refusal makes when it sends a caller here. At
    /// a small integration scale the tensor's largest eigenvalue approaches the
    /// squared magnitude and the other two approach zero, so the two ops have to
    /// agree in the limit — and if they did not, the refusal would be sending
    /// callers somewhere that computes something else.
    ///
    /// Asserted as a trend rather than an equality, because `rho` cannot be zero
    /// and the crate has no limits: the agreement must get *better* as the
    /// integration scale shrinks. A constant offset between the two would hold
    /// the error flat and fail this.
    #[test]
    fn the_magnitude_is_what_the_tensor_tends_to_as_the_integration_scale_shrinks() {
        let dim = (17, 17, 17);
        let input = noise(dim, 90210);
        let at = [8, 8, 8];
        let op = GradientMagnitudeOp::new("grad", [1.5; 3], 3.0).unwrap();
        let source: Voxels = input.clone().into();
        let mut out = Voxels::zeros(Dtype::F64, [dim.0, dim.1, dim.2]).unwrap();
        op.apply(&source, &mut out, &Anchor::whole([dim.0, dim.1, dim.2]))
            .unwrap();
        let magnitude = out.view::<f64>().unwrap()[at];
        assert!(magnitude > 1e-6, "the fixture is flat and proves nothing");

        let mut worse_than_the_last = 0;
        let mut previous = f64::INFINITY;
        for rho in [1.0, 0.6, 0.3, 0.15] {
            let tensor = StructureTensor::new([1.5; 3], [rho; 3], 3.0).unwrap();
            let largest = eigenvalues(&tensor, input.view())[0][at];
            let error = (largest.sqrt() - magnitude).abs() / magnitude;
            if error >= previous {
                worse_than_the_last += 1;
            }
            previous = error;
        }
        assert_eq!(
            worse_than_the_last, 0,
            "shrinking the integration scale did not bring the tensor closer to the \
             magnitude, so they are not the same quantity"
        );
        assert!(
            previous < 0.05,
            "at the smallest integration scale the two still differ by {:.1}%",
            previous * 100.0
        );
    }

    /// The rank-one refusal, which is the one piece of validation here that is
    /// about the mathematics rather than about a number being finite.
    #[test]
    fn a_zero_integration_scale_is_refused_with_its_reason() {
        let err = StructureTensor::new([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 3.0)
            .expect_err("rank one is not a structure tensor")
            .to_string();
        assert!(err.contains("rank one"), "{err}");
        // Positive on one axis only is still a real tensor, and is allowed: a
        // 2-D stack integrated in-plane is exactly that.
        StructureTensor::new([1.0, 1.0, 0.0], [2.0, 2.0, 0.0], 3.0).unwrap();
    }

    #[test]
    fn gamma_is_the_ratio_of_the_two_scales() {
        let tensor = StructureTensor::at_gamma([1.0, 2.0, 0.5], 3.0, 3.0).unwrap();
        assert_eq!(tensor.rho(), [3.0, 6.0, 1.5]);
        assert!(StructureTensor::at_gamma([1.0; 3], 0.0, 3.0).is_err());
        // Labkit's two integration scales over one derivative scale.
        assert_ne!(
            StructureTensor::at_gamma([1.0; 3], 1.0, 3.0).unwrap(),
            StructureTensor::at_gamma([1.0; 3], 3.0, 3.0).unwrap()
        );
    }

    /// **The integration scale is the expensive one**, by close to the factor of
    /// six that `cost_for` claims — six components smoothed against one.
    /// Asserted because it is the whole content of that model: a symmetric cost
    /// would misprice Labkit's `gamma = 3` stack badly, and symmetric is what
    /// the obvious model would have been.
    #[test]
    fn widening_the_integration_scale_costs_about_six_times_widening_the_derivative_scale() {
        let base = cost_for(&StructureTensor::new([1.0; 3], [1.0; 3], 3.0).unwrap());
        let wider_derivative = cost_for(&StructureTensor::new([4.0; 3], [1.0; 3], 3.0).unwrap());
        let wider_integration = cost_for(&StructureTensor::new([1.0; 3], [4.0; 3], 3.0).unwrap());
        assert!(wider_derivative > base);
        assert!(wider_integration > base);
        let ratio = (wider_integration - base) / (wider_derivative - base);
        assert!(
            (ratio - 6.0).abs() < 1e-9,
            "the two scales are priced {ratio:.2}x apart, not 6x"
        );
    }

    /// The fixed slab is the dominant term at the scales this op is actually
    /// used at, which is why it is measured separately rather than absorbed into
    /// the per-tap figure.
    #[test]
    fn the_per_voxel_slab_is_most_of_the_cost_at_a_small_scale() {
        let tensor = StructureTensor::at_gamma([1.0; 3], 1.0, 2.0).unwrap();
        assert!(cost_for(&tensor) < 2.0 * STRUCTURE_TENSOR_VOXEL_COST);
        // And it exceeds `ridge`'s, which does the same decomposition and fewer
        // products — a smaller figure here would mean one of the two is wrong.
        assert!(STRUCTURE_TENSOR_VOXEL_COST > DECOMPOSITION_COST);
    }
}
