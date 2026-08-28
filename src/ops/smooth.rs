// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation,
// not adapted from any implementation of it.
//
// A separable Gaussian smoothing, on its own.
//
// Why this is a file and not a parameter
// --------------------------------------
// The convolution itself already existed. `ridge.rs` needs a smoothed field
// before it can differentiate one, so [`gaussian_weights`], [`gaussian_radius`]
// and [`gaussian_smooth_into`] have been there since that filter was written,
// and they are what this file calls — there is no second convolution here and
// no second kernel builder. What was missing was the ability for a *caller* to
// ask for a smoothing, because the only shell over that kernel computed a ridge
// response with it and threw the smoothed field away.
//
// So this file is almost entirely a shell, and that is the point rather than an
// apology for it. `mod.rs` describes the shape every op here has — a free
// function generic over the element type, and a thin `BlockOp` that adapts a
// tagged buffer to it — and the claim that shape makes is that a second caller
// of the same kernel costs a shell. This is the first time that claim has been
// tested by an op that wanted a kernel someone else had already written, and it
// came to about eighty lines with no change to the kernel at all.
//
// The reach, and the one voxel that is *not* here
// -----------------------------------------------
// `reach = ceil(truncate * sigma)` per axis, which is [`gaussian_radius`] of the
// same two numbers the kernel is built from. Nothing configures it.
//
// It is worth stating what is absent. `RidgeFilterOp` declares
// `gaussian_radius + 1`, and the `+ 1` is its second-difference stencil: the
// answer at `v` needs the smoothed field at `v ± 1`, and the smoothed field at
// `u` needs the input within a radius of `u`, so the two compose by addition.
// A smoothing has no stencil on top of the convolution, so it has no `+ 1`. The
// temptation to copy the neighbour's formula is exactly the kind of thing that
// makes a reach a number someone wrote down rather than a number that follows,
// and the difference between the two ops is one voxel — the smallest possible
// overstatement, and one that would cost real fetches at every block.
//
// It is *tight*. The outermost tap of the truncated Gaussian has a non-zero
// weight, so the answer at `v` genuinely depends on the input at `v ± radius`,
// and understating it by one voxel changes the output. There is a test.
//
// The one thing outside the array
// -------------------------------
// A kernel of radius `r` asks for the sample at `v - r` when `v` is the first
// voxel of an axis, and nothing computes that — it is a convention, and
// `ridge::Boundary` is where this crate names it. `Gaussian::with_boundary`
// states it and defaults to the clamp that was the only behaviour before the
// choice existed, so a caller who does not care is untouched down to the last
// bit.
//
// Two things about it are worth having in the preamble rather than only at the
// method. It **changes no reach**: whichever convention resolves a tap, the
// position it lands on is inside `[v - r, v + r]`, because a reflection folds an
// offset that left the array back towards the edge it left by. And it is a rule
// about the **array's own edge**, which under decomposition is the volume's face
// where the volume has one and is otherwise an interior position the halo keeps
// every core voxel's taps away from — so a decomposed run reflects about the
// volume and not about the block, which is the property the acceptance suite
// asserts rather than assumes.
//
// What it produces, and why not the type it was given
// ---------------------------------------------------
// **`f64`, whatever it read.** A weighted mean of integers is not an integer,
// and there is no rounding of it that is right for every caller — round, floor,
// saturate, or keep the fraction and let the next op decide. Handing back the
// input's type would mean choosing one of those silently, inside an op whose
// name says nothing about rounding.
//
// The cost of that honesty is real and is stated here rather than discovered:
// smoothing a `u16` image writes an image four times its size. A caller who wants
// the narrow type back says so with a [`VoxelwiseMapOp`](super::voxelwise) —
// which is one op, is visible in the chain, and puts the rounding rule where a
// reader can see which one was chosen.
//
// The constant it declares, and the three it does not
// ---------------------------------------------------
// Only zero. The weights are normalised to sum to one, but that normalisation is
// a division in binary floating point and their sum is not exactly `1.0`; even
// where it were, `sum(w_i * c)` accumulated in some order is not `c`. So a block
// of constant `c` that was *skipped* would differ in the last bit from one that
// was *computed*, and `mod.rs`'s rule is that `constant_maps_to` is declared only
// where it is exactly true.
//
// Zero is exactly true, including for a negative zero, and the reason is worth
// keeping because it is not obvious: each tap contributes `w * (-0.0)`, which is
// `-0.0`, but the accumulator starts at `+0.0` and `(+0.0) + (-0.0)` is `+0.0`
// under round-to-nearest. So both zeros smooth to `+0.0`, all three passes of
// the separable convolution preserve it, and declaring `+0.0` is exact for both.
// The test checks the bits rather than the equality, because `-0.0 == 0.0`.
//
// Cost
// ----
// **Separable, and the cost model has to know it**, or every planner decision
// about where to put this op is wrong by a factor of the kernel's area. A
// three-dimensional Gaussian of radius `r` is not `(2r+1)^3` taps per voxel; it
// is three passes of `2r+1`, so the cost is proportional to the **sum** of the
// kernel lengths and not their product. At radius 4 that is 27 taps against
// 729 — a factor of 27, which is not a rounding error in a plan.
//
// The stored constant is therefore per *tap*, in the same spirit as
// `morphology.rs`'s per-element-voxel figure: a number that survives a change of
// sigma. Measured; see `super::COST_MEASUREMENT`.

use ndarray::ArrayViewMut3;

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Slicing};
use crate::voxels::Voxels;

use super::ridge::{gaussian_radius, gaussian_smooth_into_with, gaussian_weights, Boundary};

/// One Gaussian: a per-axis sigma, a truncation, and the three kernels they
/// determine.
///
/// **Per-axis rather than scalar**, for [`ScaleSpace`](super::ridge::ScaleSpace)'s
/// reason: a volume's voxels are not required to be cubes, and a scalar sigma
/// would quietly assume they are. [`Self::isotropic`] is the convenience for
/// when they are.
///
/// The kernels are built once, at construction, and held. That is not a cache —
/// it is what makes the reach and the arithmetic two views of one object rather
/// than two computations that could drift: [`Self::reach`] and
/// [`Self::kernels`] are answered from the same `sigma` and `truncate`, and
/// there is no setter for either.
#[derive(Debug, Clone, PartialEq)]
pub struct Gaussian {
    sigma: [f64; 3],
    truncate: f64,
    boundary: Boundary,
    kernels: [Vec<f64>; 3],
}

impl Gaussian {
    /// `truncate` is how many standard deviations the kernel is cut at. It is a
    /// parameter and has no default here: it trades reach against faithfulness,
    /// and which side of that a caller wants is the caller's knowledge.
    ///
    /// **A sigma of zero on an axis means that axis is not blurred**, and is the
    /// way to state a plane-by-plane smoothing of a volume: `[s, s, 0.0]` blurs
    /// within each plane and mixes none of them. The kernel there is the single
    /// tap `[1.0]`, the reach is zero, and the cost falls accordingly, so
    /// nothing downstream needs to know it is a special case.
    ///
    /// This is deliberately the same meaning `StructuringElement::from_radius`
    /// has always given a zero radius. The two disagreed until a caller wanted a
    /// blur flat on one axis and found it could not be constructed, which is the
    /// kind of inconsistency that costs an entire feature rather than a line.
    pub fn new(sigma: [f64; 3], truncate: f64) -> Result<Self> {
        for axis in 0..3 {
            if sigma[axis] < 0.0 || !sigma[axis].is_finite() {
                return Err(Error::InvalidArgument(format!(
                    "every axis of a smoothing scale must be a non-negative, finite number of \
                     voxels; got {sigma:?}. Zero is legitimate and means the axis is not \
                     blurred; a negative or non-finite scale has no kernel."
                )));
            }
        }
        if !(truncate > 0.0) || !truncate.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "a truncation must be a positive, finite multiple of the scale; got {truncate}"
            )));
        }
        // The same bound `ScaleSpace::new` states, for the same reason: the
        // radius is a `ceil` cast to `usize`, a Rust float-to-integer cast
        // saturates rather than wrapping, and a sigma of `1e30` would therefore
        // give a reach of `usize::MAX` that overflows everything derived from
        // it. Generous rather than tuned — nothing that reads a million voxels
        // either side of the one it writes is a local op.
        for axis in 0..3 {
            if truncate * sigma[axis] > 1e6 {
                return Err(Error::InvalidArgument(format!(
                    "a smoothing scale of {sigma:?} truncated at {truncate} reaches more than \
                     a million voxels on axis {axis}. That is not a local operation, and a \
                     reach that large overflows every number derived from it."
                )));
            }
        }
        let kernels = [
            gaussian_weights(sigma[0], truncate)?,
            gaussian_weights(sigma[1], truncate)?,
            gaussian_weights(sigma[2], truncate)?,
        ];
        Ok(Self {
            sigma,
            truncate,
            boundary: Boundary::default(),
            kernels,
        })
    }

    /// The same sigma on every axis.
    pub fn isotropic(sigma: f64, truncate: f64) -> Result<Self> {
        Self::new([sigma; 3], truncate)
    }

    /// State how the convolution resolves the array's own edge.
    ///
    /// **A builder rather than a third argument to [`Self::new`]**, and that is
    /// the whole of why this is additive: the convention has a default that was
    /// the only behaviour before it was a choice, so every caller that does not
    /// care keeps its call, its answer and its last bit, and the one that does
    /// care says one thing more.
    ///
    /// It changes no kernel, no radius and no [`Self::reach`] — see
    /// [`Boundary`], and the test that checks the claim rather than repeating
    /// it. The taps are the same taps; what changes is which sample a tap that
    /// left the array reads.
    pub fn with_boundary(mut self, boundary: Boundary) -> Self {
        self.boundary = boundary;
        self
    }

    pub fn sigma(&self) -> [f64; 3] {
        self.sigma
    }

    pub fn truncate(&self) -> f64 {
        self.truncate
    }

    pub fn boundary(&self) -> Boundary {
        self.boundary
    }

    pub fn kernels(&self) -> &[Vec<f64>; 3] {
        &self.kernels
    }

    /// **The reach, derived and not configured.** `ceil(truncate * sigma)`, and
    /// no stencil term — see this file's preamble for the one voxel that is
    /// deliberately absent.
    pub fn reach(&self, axis: usize) -> usize {
        gaussian_radius(self.sigma[axis], self.truncate)
    }

    /// How many multiply-adds one voxel costs: the **sum** of the three kernel
    /// lengths, because the convolution is separable. Not their product, which
    /// is what a cost model that had not noticed the separability would charge —
    /// at radius 4 the two differ by a factor of 27.
    pub fn taps(&self) -> usize {
        self.kernels[0].len() + self.kernels[1].len() + self.kernels[2].len()
    }
}

/// Smooth a block with a separable Gaussian.
pub struct SmoothOp {
    name: &'static str,
    gaussian: Gaussian,
    cost: f64,
}

impl SmoothOp {
    pub fn new(name: &'static str, gaussian: Gaussian) -> Self {
        let cost = cost_for(&gaussian);
        Self {
            name,
            gaussian,
            cost,
        }
    }

    /// The same sigma on every axis, truncated at `truncate`.
    pub fn isotropic(name: &'static str, sigma: f64, truncate: f64) -> Result<Self> {
        Ok(Self::new(name, Gaussian::isotropic(sigma, truncate)?))
    }

    pub fn gaussian(&self) -> &Gaussian {
        &self.gaussian
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// The dispatch from the tagged buffer to the generic kernel.
    ///
    /// The same three groups `RidgeFilterOp::evaluate` has, and for the same
    /// reasons: a type with a lossless `Into<f64>` reaches the kernel with **no
    /// copy**, which is what the kernel's standard-library bound is for; `bool`,
    /// `u64` and `i64` have no such conversion, so the shell widens them through
    /// `Voxels::widened`, which is a copy the *shell* chooses rather than a
    /// conversion the kernel would have to invent; and `f16` is refused before a
    /// run starts, by `accepts`, because no buffer holds one.
    fn evaluate(&self, input: &Voxels, out: ArrayViewMut3<'_, f64>) -> Result<()> {
        let boundary = self.gaussian.boundary();
        macro_rules! direct {
            ($type:ty) => {
                gaussian_smooth_into_with(
                    input.view::<$type>()?,
                    self.gaussian.kernels(),
                    boundary,
                    out,
                )
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
                gaussian_smooth_into_with(widened.view(), self.gaussian.kernels(), boundary, out)
            }
            Dtype::F16 => Err(Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; `accepts` refuses it before a run starts",
                self.name
            ))),
        }
    }
}

impl BlockOp for SmoothOp {
    /// **A stencil, and this one was verified rather than reasoned.** A
    /// separable Gaussian is three passes over the buffer, and a multi-pass
    /// implementation is exactly where "bounded reach" stops implying
    /// "sliceable": pass two reads pass one's output, so a slab has to hold
    /// enough of pass one to be right, which is a property of the
    /// implementation rather than of the mathematics.
    ///
    /// `tests/intra_block_slicing.rs` is what settles it, and it is the bar.
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn name(&self) -> &'static str {
        self.name
    }

    /// The Gaussian radius, per axis, and nothing on top of it.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.gaussian.reach(axis)
    }

    /// Every element type a buffer holds. What it accumulates into is its own
    /// business and is always `f64`.
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    /// `f64`, whatever it read. See the preamble: rounding a weighted mean back
    /// to the input's type is a decision, and it is not this op's to make.
    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let out = out.view_mut::<f64>()?;
        self.evaluate(input, out)
    }

    /// Zero, exactly, and nothing else. See the preamble.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (value == 0.0).then_some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Measured; see `super::COST_MEASUREMENT`.
///
/// Per **tap**, and a tap is one multiply-add of the separable convolution. The
/// op charges `taps()`, which is the sum of the three kernel lengths.
pub(super) fn cost_for(gaussian: &Gaussian) -> f64 {
    SMOOTH_COST_PER_TAP * gaussian.taps() as f64
}

/// Measured; see `super::COST_MEASUREMENT`.
///
/// Two rows are measured rather than one, because a single sample cannot tell
/// whether the cost scales with the sum of the kernel lengths or their product.
/// They came out at **1.390 per tap at 21 taps and 1.342 at 39** — near enough
/// the same number over nearly double the kernel that the linear-in-taps model
/// is the right shape, which is the fact worth having.
///
/// The small drift between them is a fixed per-block overhead: the separable
/// convolution allocates two scratch volumes whatever the sigma, so a straight
/// line through the two points has an intercept of about 2.2 voxelwise maps.
/// The model deliberately does not carry it. An intercept is a cost per *block*
/// masquerading as a cost per voxel, so its value would depend on the block size
/// the measurement happened to use — and `cost_per_voxel` has nowhere to put it
/// honestly. One number across the range is the better lie.
///
/// For scale: at 21 taps this op measured 29.2 voxelwise maps, against 81.1 for
/// a dense 27-voxel rank filter. Separability is most of why smoothing is cheap.
pub(super) const SMOOTH_COST_PER_TAP: f64 = 1.37;

#[cfg(test)]
mod tests {
    use ndarray::Array3;

    use super::*;

    fn ramp(shape: (usize, usize, usize)) -> Array3<f64> {
        Array3::from_shape_fn(shape, |(i, j, k)| {
            ((i * 7919 + j * 104729 + k * 1013) % 251) as f64
        })
    }

    fn smoothed(op: &SmoothOp, input: &Voxels) -> Array3<f64> {
        let shape = input.shape();
        let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
        op.apply(input, &mut out, &Anchor::whole(shape)).unwrap();
        out.view::<f64>().unwrap().to_owned()
    }

    /// The reach is the Gaussian's radius and carries **no** stencil term — the
    /// one voxel `RidgeFilterOp` adds is that filter's second difference, not
    /// the smoothing's. This is the test that fails if someone copies the
    /// neighbour's formula.
    #[test]
    fn the_reach_is_the_gaussian_radius_and_not_the_ridge_filters() {
        use super::super::ridge::{Polarity, RidgeFilterOp, RidgeResponse, ScaleSpace};

        let sigma = [2.0, 2.0, 2.0];
        let truncate = 3.0;
        let smooth = SmoothOp::new("smooth", Gaussian::new(sigma, truncate).unwrap());
        assert_eq!(smooth.reach(0, 1000), 6, "ceil(3 * 2)");

        let ridge = RidgeFilterOp::new(
            "ridge",
            ScaleSpace::new(vec![sigma], truncate, 1.0).unwrap(),
            RidgeResponse::new(0.5, 0.5, 1.0, Polarity::Ridge).unwrap(),
        );
        assert_eq!(
            ridge.reach(0, 1000),
            smooth.reach(0, 1000) + 1,
            "the ridge filter differentiates the smoothed field and reaches one further"
        );
    }

    /// The reach is *tight*: the outermost tap has a non-zero weight, so the
    /// answer at a voxel genuinely depends on the input exactly `radius` away.
    /// A declaration short by one would be a silently wrong volume at every
    /// block seam.
    #[test]
    fn the_answer_depends_on_the_input_exactly_at_the_declared_reach() {
        let op = SmoothOp::new("smooth", Gaussian::isotropic(1.5, 3.0).unwrap());
        let radius = op.reach(0, 1000);
        assert_eq!(radius, 5);

        let centre = radius + 3;
        let extent = 2 * centre + 1;
        let shape = (extent, 3, 3);

        let base = Array3::<f64>::zeros(shape);
        let quiet = smoothed(&op, &base.clone().into());

        // one voxel exactly `radius` away on axis 0
        let mut poked = base.clone();
        poked[[centre - radius, 1, 1]] = 1.0;
        let loud = smoothed(&op, &poked.into());
        assert_ne!(
            quiet[[centre, 1, 1]],
            loud[[centre, 1, 1]],
            "the tap at the declared radius carries weight, so the reach is not an overstatement"
        );

        // and one voxel beyond it does not reach
        let mut far = base;
        far[[centre - radius - 1, 1, 1]] = 1.0;
        let unheard = smoothed(&op, &far.into());
        assert_eq!(
            quiet[[centre, 1, 1]],
            unheard[[centre, 1, 1]],
            "nothing beyond the truncation contributes, so the reach is not an understatement"
        );
    }

    /// A separable Gaussian is a weighted mean, so it must preserve a constant
    /// field to within the normalisation — and the interior, away from the
    /// clamp, is where that is checkable. This is the property that would break
    /// if a kernel were built unnormalised.
    #[test]
    fn a_constant_field_survives_smoothing_to_within_the_normalisation() {
        let op = SmoothOp::new("smooth", Gaussian::isotropic(1.0, 4.0).unwrap());
        let input: Voxels = Array3::from_elem((15, 15, 15), 7.0).into();
        let out = smoothed(&op, &input);
        for value in out.iter() {
            assert!(
                (value - 7.0).abs() < 1e-12,
                "a normalised weighted mean of a constant is that constant; got {value}"
            );
        }
    }

    /// ...and *not* exactly, which is why `constant_maps_to` refuses to declare
    /// anything but zero. If this ever became an equality, the declaration could
    /// widen; until then a skipped block and a computed block would differ in
    /// the last bit, and that is a parity difference nobody would look for.
    #[test]
    fn a_non_zero_constant_is_not_reproduced_bit_for_bit_which_is_why_it_is_not_declared() {
        let op = SmoothOp::new("smooth", Gaussian::isotropic(1.7, 3.0).unwrap());
        assert_eq!(op.constant_maps_to(7.0), None);
        assert_eq!(op.constant_maps_to(1.0), None);

        let input: Voxels = Array3::from_elem((15, 15, 15), 7.0).into();
        let out = smoothed(&op, &input);
        let exact = out.iter().filter(|value| **value == 7.0).count();
        assert!(
            exact < out.len(),
            "if every voxel came back bit-exact the declaration could widen; it does not"
        );
    }

    /// Zero is exact, and so is negative zero — because the accumulator starts
    /// at `+0.0` and `(+0.0) + (-0.0)` is `+0.0`. Checked on the **bits**, since
    /// `-0.0 == 0.0` would pass the obvious assertion either way.
    #[test]
    fn both_zeros_smooth_to_positive_zero_which_is_what_is_declared() {
        let op = SmoothOp::new("smooth", Gaussian::isotropic(1.3, 3.0).unwrap());
        assert_eq!(op.constant_maps_to(0.0), Some(0.0));
        assert_eq!(op.constant_maps_to(-0.0), Some(0.0));

        for constant in [0.0f64, -0.0f64] {
            let input: Voxels = Array3::from_elem((9, 9, 9), constant).into();
            let out = smoothed(&op, &input);
            for value in out.iter() {
                assert_eq!(
                    value.to_bits(),
                    0.0f64.to_bits(),
                    "smoothing {constant} must give exactly +0.0, not merely something equal to it"
                );
            }
        }
    }

    /// Every element type reaches the same kernel and gives the same answer. The
    /// integer arms take the no-copy path and the three widened ones take the
    /// copy; a bug in either dispatch shows up as a difference here.
    #[test]
    fn every_accepted_element_type_gives_the_same_answer_as_f64() {
        let op = SmoothOp::new("smooth", Gaussian::isotropic(1.0, 3.0).unwrap());
        let pattern =
            Array3::from_shape_fn((7, 6, 5), |(i, j, k)| ((i * 13 + j * 7 + k) % 5) as u8);
        let reference = smoothed(&op, &pattern.mapv(f64::from).into());

        let cases: Vec<Voxels> = vec![
            pattern.clone().into(),
            pattern.mapv(u16::from).into(),
            pattern.mapv(u32::from).into(),
            pattern.mapv(u64::from).into(),
            pattern.mapv(|value| value as i8).into(),
            pattern.mapv(i16::from).into(),
            pattern.mapv(i32::from).into(),
            pattern.mapv(i64::from).into(),
            pattern.mapv(f32::from).into(),
            pattern.mapv(|value| value != 0).into(),
        ];
        for input in &cases {
            let dtype = input.dtype();
            assert!(op.accepts(dtype), "{dtype:?}");
            assert_eq!(op.produces(dtype), Dtype::F64, "{dtype:?}");
            let out = smoothed(&op, input);
            if dtype == Dtype::Bool {
                // a different field — the mask, not the values — so only the
                // dispatch is asserted here, by it running at all
                assert_eq!(out.shape(), reference.shape());
                continue;
            }
            assert_eq!(out, reference, "{dtype:?} disagreed with f64");
        }

        assert!(!op.accepts(Dtype::F16));
    }

    /// **The reflected answer against arithmetic done by hand**, not against a
    /// second call to the same code.
    ///
    /// One axis, one kernel, five voxels, and the expected value written as the
    /// literal sum of taps times samples with the reflected indices spelled out.
    /// That is what pins the convention: an implementation that mirrored about
    /// the edge sample instead — `-1` reading `1` — passes every symmetry and
    /// constancy check in this file and fails this one.
    #[test]
    fn the_reflected_answer_is_the_hand_written_sum_of_taps() {
        // ceil(3 * 0.4) = 2, so the kernel is five taps and both ends of it
        // leave a five-voxel axis at the faces
        let gaussian = Gaussian::new([0.4, 0.0, 0.0], 3.0)
            .unwrap()
            .with_boundary(Boundary::Reflect);
        assert_eq!(gaussian.reach(0), 2, "ceil(3 * 0.4)");
        let taps = gaussian.kernels()[0].clone();
        assert_eq!(taps.len(), 5);

        let samples = [2.0f64, 3.0, 5.0, 7.0, 11.0];
        let source = || -> Voxels {
            let mut input = Array3::<f64>::zeros((5, 1, 1));
            for (index, value) in samples.iter().enumerate() {
                input[[index, 0, 0]] = *value;
            }
            input.into()
        };
        let out = smoothed(&SmoothOp::new("reflect", gaussian), &source());

        // At voxel 0 the taps land on -2, -1, 0, 1, 2. Reflected, that reads
        // samples 1, 0, 0, 1, 2 — the edge sample repeated once and the run
        // then reversed. Summed lowest offset first, as the kernel is.
        let want = (((taps[0] * samples[1] + taps[1] * samples[0]) + taps[2] * samples[0])
            + taps[3] * samples[1])
            + taps[4] * samples[2];
        assert_eq!(
            out[[0, 0, 0]].to_bits(),
            want.to_bits(),
            "the reflected sum must agree in every bit, summation order included"
        );

        // and at the far face: taps on 2, 3, 4, 5, 6 read samples 2, 3, 4, 4, 3
        let want = (((taps[0] * samples[2] + taps[1] * samples[3]) + taps[2] * samples[4])
            + taps[3] * samples[4])
            + taps[4] * samples[3];
        assert_eq!(out[[4, 0, 0]].to_bits(), want.to_bits());

        // the clamped answer at the same voxel is a different number, which is
        // what makes the choice worth having
        let clamped = smoothed(
            &SmoothOp::new("clamp", Gaussian::new([0.4, 0.0, 0.0], 3.0).unwrap()),
            &source(),
        );
        assert_ne!(clamped[[0, 0, 0]], out[[0, 0, 0]]);
        // and the middle voxel, whose taps never leave the array, is identical
        assert_eq!(clamped[[2, 0, 0]].to_bits(), out[[2, 0, 0]].to_bits());
    }

    /// **A kernel wider than the axis it runs along.** A sigma of ten truncated
    /// at four reaches forty voxels, and an axis of three is thirteen folds
    /// short of containing it — which is an ordinary thing to ask of a wide blur
    /// on a thin volume, and the case a reflection that mirrors only once gets
    /// wrong by indexing outside the array altogether.
    ///
    /// Checked against the sum written out over the whole kernel with the
    /// indices folded by hand, so nothing here rests on the implementation
    /// being right about the fold.
    #[test]
    fn a_kernel_far_wider_than_the_axis_still_reflects_into_it() {
        let gaussian = Gaussian::new([10.0, 0.0, 0.0], 4.0)
            .unwrap()
            .with_boundary(Boundary::Reflect);
        assert_eq!(gaussian.reach(0), 40, "ceil(4 * 10)");
        let taps = gaussian.kernels()[0].clone();
        assert_eq!(taps.len(), 81);

        let samples = [2.0f64, 3.0, 5.0];
        let mut input = Array3::<f64>::zeros((3, 1, 1));
        for (index, value) in samples.iter().enumerate() {
            input[[index, 0, 0]] = *value;
        }
        let out = smoothed(&SmoothOp::new("wide", gaussian), &input.into());

        // fold by mirroring until the position is inside, which is the
        // definition the closed form has to match
        fn fold(mut position: isize, extent: isize) -> usize {
            loop {
                if position < 0 {
                    position = -position - 1;
                } else if position >= extent {
                    position = 2 * extent - 1 - position;
                } else {
                    return position as usize;
                }
            }
        }

        for centre in 0..3isize {
            let mut want = 0.0f64;
            for (step, weight) in taps.iter().enumerate() {
                want += weight * samples[fold(centre + step as isize - 40, 3)];
            }
            assert_eq!(
                out[[centre as usize, 0, 0]].to_bits(),
                want.to_bits(),
                "voxel {centre} of a three-voxel axis under a radius-forty kernel"
            );
        }

        // every answer is finite and inside the range of the samples, which a
        // fold that escaped the array could not have managed
        for value in out.iter() {
            assert!(
                value.is_finite() && *value >= 2.0 && *value <= 5.0,
                "{value}"
            );
        }
    }

    /// The convention changes **no reach**, which is the claim the module makes
    /// and this is the check of it rather than a restatement.
    ///
    /// Both halves matter. The declared number is the same — that is arithmetic
    /// on the sigma and could hardly be otherwise. The number is also still
    /// *sufficient and tight* under the reflected convention: the answer moves
    /// when the input at exactly the radius moves, and does not when the input
    /// beyond it does. The second is the one that could have gone wrong, because
    /// a reflection reads positions the caller never named.
    #[test]
    fn the_reflected_convention_declares_and_needs_the_same_reach() {
        let clamped = Gaussian::new([1.0, 1.5, 0.6], 3.0).unwrap();
        let reflected = clamped.clone().with_boundary(Boundary::Reflect);
        assert_eq!(
            clamped.boundary(),
            Boundary::Clamp,
            "the default is additive"
        );
        for axis in 0..3 {
            assert_eq!(clamped.reach(axis), reflected.reach(axis));
        }
        assert_eq!(clamped.kernels(), reflected.kernels());
        assert_eq!(clamped.taps(), reflected.taps());
        assert_eq!(
            SmoothOp::new("a", clamped).cost_per_voxel(),
            SmoothOp::new("b", reflected).cost_per_voxel()
        );

        let op = SmoothOp::new(
            "reflect",
            Gaussian::isotropic(1.5, 3.0)
                .unwrap()
                .with_boundary(Boundary::Reflect),
        );
        let radius = op.reach(0, 1000);
        assert_eq!(radius, 5);

        let centre = radius + 3;
        let extent = 2 * centre + 1;
        let shape = (extent, 3, 3);

        let base = Array3::<f64>::zeros(shape);
        let quiet = smoothed(&op, &base.clone().into());

        let mut poked = base.clone();
        poked[[centre - radius, 1, 1]] = 1.0;
        let loud = smoothed(&op, &poked.into());
        assert_ne!(
            quiet[[centre, 1, 1]],
            loud[[centre, 1, 1]],
            "the tap at the declared radius carries weight under either convention"
        );

        let mut far = base;
        far[[centre - radius - 1, 1, 1]] = 1.0;
        let unheard = smoothed(&op, &far.into());
        assert_eq!(
            quiet[[centre, 1, 1]],
            unheard[[centre, 1, 1]],
            "a reflection folds an offset back towards the edge it left by, never \
             further out, so nothing beyond the truncation contributes"
        );
    }

    /// A reflection is still a weighted mean of samples that were really there,
    /// so it keeps a constant constant — the property that separates it from a
    /// convention that invents zeros outside the array.
    #[test]
    fn the_reflected_convention_also_keeps_a_constant_constant_at_the_faces() {
        let op = SmoothOp::new(
            "reflect",
            Gaussian::isotropic(2.0, 3.0)
                .unwrap()
                .with_boundary(Boundary::Reflect),
        );
        let input: Voxels = Array3::from_elem((5, 5, 5), 3.0).into();
        let out = smoothed(&op, &input);
        for value in out.iter() {
            assert!((value - 3.0).abs() < 1e-12, "{value}");
        }
        assert_eq!(op.constant_maps_to(0.0), Some(0.0));
    }

    /// The clamp is at the array's own edge, which is right at a volume boundary
    /// and deliberately wrong at a block seam. A clamped edge keeps a constant
    /// constant, which is the observable half of that convention.
    #[test]
    fn the_boundary_clamp_keeps_a_constant_constant_at_the_faces() {
        let op = SmoothOp::new("smooth", Gaussian::isotropic(2.0, 3.0).unwrap());
        let input: Voxels = Array3::from_elem((5, 5, 5), 3.0).into();
        let out = smoothed(&op, &input);
        // every voxel of a 5-cube is within the radius of a face
        for value in out.iter() {
            assert!((value - 3.0).abs() < 1e-12);
        }
    }

    /// Smoothing does not move mass around asymmetrically: an impulse at the
    /// centre of a volume large enough to contain the whole kernel smooths to a
    /// field symmetric about it. This is what fails if the kernel is built with
    /// an off-by-one in its offsets, which a monotone test would not notice.
    #[test]
    fn an_impulse_smooths_to_a_field_symmetric_about_it() {
        let op = SmoothOp::new("smooth", Gaussian::isotropic(1.0, 3.0).unwrap());
        let extent = 2 * op.reach(0, 1000) + 5;
        let centre = extent / 2;
        let mut impulse = Array3::<f64>::zeros((extent, extent, extent));
        impulse[[centre, centre, centre]] = 1.0;
        let out = smoothed(&op, &impulse.into());

        for step in 1..=op.reach(0, 1000) {
            for axis in 0..3 {
                let mut low = [centre; 3];
                let mut high = [centre; 3];
                low[axis] -= step;
                high[axis] += step;
                assert_eq!(
                    out[low], out[high],
                    "axis {axis} at ±{step} must be bit-identical; the weights are computed \
                     from x*x and the offsets run symmetrically"
                );
            }
        }
        // and it is a maximum at the impulse
        let peak = out[[centre, centre, centre]];
        assert!(out.iter().all(|value| *value <= peak));
    }

    /// The cost follows the **separability**: three passes of `2r+1` taps, not
    /// one pass of `(2r+1)^3`. A model that charged the product would price this
    /// op 27x too high at radius 4 and put it in the wrong place in every plan.
    #[test]
    fn the_cost_is_linear_in_the_taps_and_not_in_their_product() {
        let narrow = Gaussian::isotropic(1.0, 3.0).unwrap();
        let wide = Gaussian::isotropic(2.0, 3.0).unwrap();
        let (r_narrow, r_wide) = (narrow.reach(0), wide.reach(0));
        assert_eq!((r_narrow, r_wide), (3, 6));

        assert_eq!(narrow.taps(), 3 * (2 * r_narrow + 1));
        assert_eq!(wide.taps(), 3 * (2 * r_wide + 1));

        let cheap = SmoothOp::new("narrow", narrow).cost_per_voxel();
        let dear = SmoothOp::new("wide", wide).cost_per_voxel();
        assert!(dear > cheap);

        let separable = dear / cheap;
        assert!(
            (separable - 13.0 / 7.0).abs() < 1e-12,
            "the ratio must be the ratio of the tap counts, 39/21; got {separable}"
        );

        // and the size of the mistake, stated absolutely rather than as a ratio
        // of ratios: what a model that had not noticed the separability would
        // charge for the wide kernel, against what the wide kernel costs.
        let charged = SmoothOp::new("wide", Gaussian::isotropic(2.0, 3.0).unwrap())
            .gaussian()
            .taps();
        let dense = (2 * r_wide + 1).pow(3);
        assert_eq!((charged, dense), (39, 2197));
        assert!(
            dense > 50 * charged,
            "a dense model would price this op {}x too high",
            dense / charged
        );
    }

    /// Every parameter that cannot produce a kernel is refused at construction,
    /// not at the first block.
    #[test]
    fn an_unusable_scale_is_refused_when_it_is_stated() {
        assert!(Gaussian::new([-1.0, 1.0, 1.0], 3.0).is_err());
        assert!(Gaussian::new([f64::NAN, 1.0, 1.0], 3.0).is_err());
        assert!(Gaussian::new([f64::INFINITY, 1.0, 1.0], 3.0).is_err());
        assert!(Gaussian::isotropic(1.0, 0.0).is_err());
        assert!(Gaussian::isotropic(1.0, f64::NAN).is_err());
        // and the saturating-cast bound
        assert!(Gaussian::isotropic(1e9, 3.0).is_err());
        assert!(Gaussian::isotropic(1.0, 3.0).is_ok());
    }

    /// A sigma of zero on an axis is the identity **on that axis**: the kernel is
    /// one tap, the reach is zero, and the volume is blurred plane by plane.
    ///
    /// This is the case that could not be constructed at all until a caller
    /// needed it, while a zero *radius* had always been accepted for a
    /// structuring element. The test asserts the three things that make it a
    /// real feature rather than a relaxed check: the reach really is zero on
    /// that axis, planes really do not mix, and within a plane the answer is
    /// exactly the two-dimensional blur.
    #[test]
    fn a_zero_sigma_is_the_identity_on_that_axis_and_blurs_the_others() {
        let flat = Gaussian::new([1.0, 1.0, 0.0], 3.0).unwrap();
        assert_eq!(flat.reach(0), 3);
        assert_eq!(flat.reach(1), 3);
        assert_eq!(flat.reach(2), 0, "a zero sigma reaches nothing");
        assert_eq!(flat.kernels()[2], vec![1.0], "one tap, weight exactly one");

        let op = SmoothOp::new("plane", flat);
        assert_eq!(op.reach(2, 1000), 0);

        // two planes that differ everywhere; neither may leak into the other
        let mut input = Array3::<f64>::zeros((9, 9, 2));
        input[[4, 4, 0]] = 1.0;
        let out = smoothed(&op, &input.clone().into());
        assert!(
            out.slice(ndarray::s![.., .., 1])
                .iter()
                .all(|value| *value == 0.0),
            "an impulse in plane 0 reached plane 1"
        );
        assert!(
            out[[4, 4, 0]] > 0.0 && out[[3, 4, 0]] > 0.0,
            "and it did blur in-plane"
        );

        // within a plane it is exactly the 2-D blur of that plane alone
        let mut plane = Array3::<f64>::zeros((9, 9, 1));
        plane[[4, 4, 0]] = 1.0;
        let alone = smoothed(&op, &plane.into());
        for i in 0..9 {
            for j in 0..9 {
                assert_eq!(out[[i, j, 0]], alone[[i, j, 0]]);
            }
        }
    }

    /// A smoothing is not a rank filter: it does not select a value that was
    /// read. This is the sanity check that the two families stay distinguishable
    /// — a "smoothing" whose output is always one of its inputs is a bug.
    #[test]
    fn the_output_is_a_mean_and_not_a_selection() {
        let op = SmoothOp::new("smooth", Gaussian::isotropic(1.0, 3.0).unwrap());
        let input = ramp((11, 11, 11));
        let out = smoothed(&op, &input.clone().into());
        let seen: std::collections::HashSet<u64> = input.iter().map(|v| v.to_bits()).collect();
        let novel = out.iter().filter(|v| !seen.contains(&v.to_bits())).count();
        assert!(
            novel > out.len() / 2,
            "most smoothed values must be values that were not in the input"
        );
    }
}
