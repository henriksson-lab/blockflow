// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definitions of the operations,
// not adapted from any implementation of them.
//
// Removing a locally estimated level, and dividing by a locally estimated
// spread. Two operations, one arrangement: **estimate on a sample lattice,
// interpolate back, combine with the voxel it was estimated for.**
//
// What is new here and what is not
// --------------------------------
// Not new: the estimate. `super::local` already evaluates a statistic over a
// parameterised window at the points of a globally anchored [`SampleLattice`]
// and interpolates it back to every voxel, and that machinery is used here
// unchanged — [`LocalStatistic::evaluate_into`] is called, not reimplemented,
// and there is no second copy of the gather, the reducer, the lattice or the
// interpolation in this file. Both operations below are that machinery with a
// different **combination step**, which is the same relationship
// [`super::local::AdaptiveThresholdOp`] has to it: that one compares the voxel
// against the estimate, these two subtract it or divide by it.
//
// New: the combination itself, and the fact that the second operation runs the
// estimator **twice** — a centre and a spread, on two independent lattices —
// which no existing op does and which makes the reach a maximum over two
// two-term reaches rather than one.
//
// | | estimate | combination |
// |---|---|---|
// | [`AdaptiveThresholdOp`](super::local::AdaptiveThresholdOp) | one statistic | `value > scale * s + offset` |
// | [`LevelCorrectionOp`] | one statistic | `value - s`, or `value / max(s, floor)` |
// | [`LocalContrastOp`] | **two** statistics | `(value - c) / max(s, floor)` |
//
// Why they are two ops and not one, and where they touch
// ------------------------------------------------------
// [`LevelCorrectionOp`] removes an estimate of the *level*: the slowly varying
// component a voxel sits on top of, taken out either additively or
// multiplicatively because which of the two a signal carries is a property of
// how it was acquired and not something this crate can know.
// [`LocalContrastOp`] divides by an estimate of the *spread*, having first
// removed a centre, so that a fixed contrast means the same thing everywhere in
// the volume regardless of the local level.
//
// They touch at exactly one point and it is worth naming rather than
// discovering: `LocalContrastOp` with its centre removed would be
// `LevelCorrectionOp`'s `Divide`, so **it is not offered**. The centre is
// mandatory. A caller who wants the division alone asks for the division alone,
// from the op whose parameterisation is about a level, and there is one
// implementation of that arithmetic rather than two that can drift.
//
// The anchoring, which is the whole hazard
// ----------------------------------------
// `docs/design/XY_BLOCK_SPLITTING.md` records a sampled-estimate implementation
// that lays its lattice out relative to **the array it is handed** — count from
// the block's shape, first sample from the block's shape, upsampling factor from
// the block's shape — and therefore gives the same voxel a different estimate
// depending on which block it landed in, *throughout* the block rather than at
// its seams. A halo does not fix it: widening the read does not change what the
// lattice was laid out relative to. In the pipeline that document is about, this
// is what blocks two of the three axes from being cut at all.
//
// This file cannot commit that defect, and the reason is structural rather than
// careful: it never constructs a lattice. It calls
// [`LocalStatistic::evaluate_into`], which builds one from `Anchor::volume`, and
// [`SampleLattice::centred`](super::local::SampleLattice::centred) is the
// only constructor there is. There is no parameter on either op below that
// could make an estimate a function of the buffer, because neither op is ever
// told the buffer's shape by anything but the buffer.
//
// The reach, and where each term comes from
// -----------------------------------------
// [`LocalStatistic::reach`] is already the two-term reach — how far the
// interpolation reaches for a lattice point, plus how far that point's window
// reaches beyond it — and it is already tight to one voxel: `bracket` withholds
// the bracketing sample whose weight would be zero, so a coordinate sitting on a
// sample never reads the one a full spacing away and the interpolation term is
// `spacing - 1` rather than `spacing`. That tightening is **inherited here, not
// repeated**: the ops below derive their reach by asking the statistics they
// hold, so there is no second expression to keep in step, and the one-voxel
// question was answered where the lattice lives.
//
// What each op adds to it:
//
// * [`LevelCorrectionOp`]: nothing. The combination is voxelwise — it reads the
//   voxel it writes and the estimate at that voxel — so the reach is the
//   statistic's, unchanged;
// * [`LocalContrastOp`]: the **maximum** over its two statistics, per axis,
//   because a voxel's answer depends on everything either estimate read. Not the
//   sum: the two estimates are computed from the same buffer in parallel rather
//   than composed, so the dependency cone is the union of two cones about the
//   same voxel and its half-width is the larger of the two.
//
// The constant algebra, and why subtraction and division may declare more than
// the mean underneath them can
// ---------------------------------------------------------------------------
// `super::local` withholds `constant_maps_to` for a mean at any value but zero,
// because `(v + v + ... + v) / m` is not `v` in binary floating point and a
// short-circuited block would differ from a computed one in the last bit. The
// arithmetic in *this* file adds no such gap, and the argument is short:
//
// 1. where the statistic declares a value, that value is **bit-identical** to
//    what the kernel computed — that is what the declaration means;
// 2. [`normalise_value`] is the only expression either path evaluates. The
//    kernel calls it per voxel and `constant_maps_to` calls it once, with the
//    same operands;
// 3. IEEE 754 subtraction, division and `max` are correctly rounded and
//    deterministic: the same two `f64` operands give the same `f64` result at
//    every evaluation. Rust performs no floating-point contraction and no
//    excess-precision evaluation of `f64`, so "the same expression on the same
//    operands" is a statement about the emitted code and not only about the
//    source.
//
// So these ops declare **exactly where their statistics declare**, and nowhere
// else — which is not a widening of what `local` says, it is the same statement
// carried through an operation that cannot lose it. `a_declared_constant_is_the_
// computed_one_bit_for_bit` computes both and compares, so the argument above is
// checked rather than believed.

use ndarray::{Array3, ArrayView3, ArrayViewMut3, Zip};

use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp};
use crate::voxels::Voxels;

use super::local::{cost_for, LocalStatistic};
use super::shapes_agree;
use super::voxelwise::{combine_into, COMBINE_COST};

// -------------------------------------------------------------- kernels --

/// The arithmetic, as a scalar, in one place.
///
/// `(value - centre) / max(spread, floor)`, with either estimate absent.
///
/// **Why this exists as a function rather than as three expressions.** The
/// kernel evaluates it per voxel and [`BlockOp::constant_maps_to`] evaluates it
/// once, and the declaration is only exactly true if the two evaluate the *same*
/// expression. Written twice they would be the same expression until somebody
/// changed one of them; written once they cannot be. The `Option`s are constants
/// at every call site — each arm of [`normalise_against_into`] passes literals —
/// so nothing branches per voxel once this is inlined.
///
/// `floor` bounds the divisor away from zero. It is not a convenience: an
/// estimate that reaches zero would otherwise divide a finite value into an
/// infinity, or zero into a `NaN` that no equality test can ever compare equal
/// to itself, and a single such voxel makes every decomposition test in this
/// crate meaningless rather than failing. Both ops therefore refuse a floor that
/// is not finite and strictly positive, at construction, where it is one
/// argument rather than a volume of `NaN`.
#[inline]
pub fn normalise_value(value: f64, centre: Option<f64>, spread: Option<f64>, floor: f64) -> f64 {
    let centred = match centre {
        Some(centre) => value - centre,
        None => value,
    };
    match spread {
        Some(spread) => centred / spread.max(floor),
        None => centred,
    }
}

/// Combine every voxel with the estimates co-located with it.
///
/// Generic over the input element type, which is as far as the algorithm allows:
/// the estimates are `f64` because [`LocalStatistic`] produces `f64` — a mean or
/// a deviation must be accumulated in something wider than the values it is
/// taken over — and the quotient of a value and an `f64` is an `f64` whatever
/// the value was. So the one type that can genuinely vary here does, and the
/// three that cannot are stated.
///
/// The three useful combinations are the three arms below, and **each is a
/// separate loop**: the choice is made once for the buffer rather than once per
/// voxel. Neither estimate present is not a fourth arm, it is a caller who has
/// asked for a copy under another name, and it is refused.
pub fn normalise_against_into<T>(
    input: ArrayView3<'_, T>,
    centre: Option<ArrayView3<'_, f64>>,
    spread: Option<ArrayView3<'_, f64>>,
    floor: f64,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    shapes_agree(input.shape(), out.shape(), "normalise_against_into")?;
    match (centre, spread) {
        (None, None) => Err(Error::InvalidArgument(
            "normalise_against_into: neither a centre nor a spread was supplied, so there \
             is nothing to normalise against and the operation is a copy"
                .to_string(),
        )),
        (Some(centre), None) => combine_into(input, centre, out, |&value, &centre| {
            normalise_value(value.into(), Some(centre), None, floor)
        }),
        (None, Some(spread)) => combine_into(input, spread, out, |&value, &spread| {
            normalise_value(value.into(), None, Some(spread), floor)
        }),
        (Some(centre), Some(spread)) => {
            shapes_agree(
                input.shape(),
                centre.shape(),
                "normalise_against_into centre",
            )?;
            shapes_agree(
                input.shape(),
                spread.shape(),
                "normalise_against_into spread",
            )?;
            Zip::from(&mut out)
                .and(input)
                .and(centre)
                .and(spread)
                .for_each(|slot, &value, &centre, &spread| {
                    *slot = normalise_value(value.into(), Some(centre), Some(spread), floor);
                });
            Ok(())
        }
    }
}

// ------------------------------------------------------------ parameters --

/// How an estimated level is taken out of the value it was estimated for.
///
/// The two cases are the additive and the multiplicative one, and which is right
/// depends on how the signal was formed — an offset that was added to it, or a
/// gain that was applied to it. This crate cannot know which, so it is a
/// parameter, and it is this enum rather than a `bool` so that the divisor's
/// floor travels with the case that needs it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Removal {
    /// `value - level`.
    Subtract,
    /// `value / max(level, floor)`.
    ///
    /// `floor` must be finite and strictly positive; see [`normalise_value`].
    Divide { floor: f64 },
}

impl Removal {
    /// Which slot of [`normalise_value`] the single estimate occupies.
    ///
    /// Generic in the estimate so that [`LevelCorrectionOp::apply`] and
    /// [`LevelCorrectionOp::constant_maps_to`] place an array and a scalar by
    /// **the same code**. Two `match`es on this enum, one holding views and one
    /// holding numbers, would be two things that agree until one is edited.
    fn slots<A>(self, level: A) -> (Option<A>, Option<A>) {
        match self {
            Removal::Subtract => (Some(level), None),
            Removal::Divide { .. } => (None, Some(level)),
        }
    }

    /// The divisor's floor. Zero for [`Removal::Subtract`], which has no
    /// divisor — the value is passed to [`normalise_value`] and never consulted,
    /// because the `spread` slot it guards is `None`.
    fn floor(self) -> f64 {
        match self {
            Removal::Subtract => 0.0,
            Removal::Divide { floor } => floor,
        }
    }
}

/// A floor must be usable as a divisor, and the check is here rather than at the
/// two call sites so that both ops refuse the same set of arguments.
fn check_floor(floor: f64, what: &str) -> Result<()> {
    if !floor.is_finite() || floor <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "{what}: a divisor floor must be finite and strictly positive, got {floor}. It \
             bounds an estimate away from zero; without it a flat region divides to an \
             infinity or a NaN, and a NaN is not merely a wrong value — it compares unequal \
             to itself, so every byte-identity check downstream stops meaning anything."
        )));
    }
    Ok(())
}

// -------------------------------------------------------------- adapters --

/// Estimate a slowly varying level on a sample lattice and remove it.
///
/// The estimate is a [`LocalStatistic`], which is to say: a statistic of a
/// parameterised window, evaluated at the points of a globally anchored lattice
/// and interpolated back to full resolution. Which statistic, which window and
/// which spacing are the caller's — a low order statistic over a window several
/// times the size of whatever is to be kept is the usual choice for a level, but
/// "several times" is a fact about the caller's data and not about this crate.
///
/// **The estimate is coarse by construction and that is the point.** A spacing
/// of `s` costs `s^3` times less window evaluation than sampling every voxel,
/// and a level that varies slowly is exactly the thing a coarse lattice can
/// represent. What it costs instead is a reach: the interpolation reaches for
/// lattice points up to a spacing away, and each of those reaches its window
/// further still.
#[derive(Debug)]
pub struct LevelCorrectionOp {
    name: &'static str,
    level: LocalStatistic,
    removal: Removal,
    cost: f64,
}

impl LevelCorrectionOp {
    pub fn new(name: &'static str, level: LocalStatistic, removal: Removal) -> Result<Self> {
        if let Removal::Divide { floor } = removal {
            check_floor(floor, name)?;
        }
        let cost = cost_for(&level) + COMBINE_COST;
        Ok(Self {
            name,
            level,
            removal,
            cost,
        })
    }

    pub fn level(&self) -> &LocalStatistic {
        &self.level
    }

    pub fn removal(&self) -> Removal {
        self.removal
    }

    /// Override the composed cost. See [`cost_per_voxel`](BlockOp::cost_per_voxel).
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for LevelCorrectionOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The statistic's own two-term reach, and nothing added: the removal is
    /// voxelwise. Both terms come from the parameters; neither is settable.
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        self.level.reach(axis, volume_len)
    }

    /// `f64`, for [`LocalStatisticOp`](super::local::LocalStatisticOp)'s reason:
    /// [`LocalStatistic::evaluate_into`] is stated in `f64` on both sides, and
    /// widening it is kernel work in `local` rather than shell work here. The
    /// kernel this file owns is generic over the input element type, so when
    /// that day comes this shell is what changes and `normalise_against_into` is
    /// not.
    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let input = input.view::<f64>()?;
        let mut level = Array3::<f64>::zeros(input.dim());
        self.level.evaluate_into(input, at, level.view_mut())?;
        let (centre, spread) = self.removal.slots(level.view());
        normalise_against_into(
            input,
            centre,
            spread,
            self.removal.floor(),
            out.view_mut::<f64>()?,
        )
    }

    /// Exactly where the statistic declares, and by evaluating the operation.
    ///
    /// Both cases are worth reading concretely. Subtracting an order statistic
    /// from a constant volume gives **exactly** zero, because the order
    /// statistic selected a value that was read and `v - v` is zero for every
    /// finite `v`. Dividing by one gives exactly one wherever the constant
    /// exceeds the floor. Neither is asserted here — both are computed by
    /// [`normalise_value`], which is what the kernel computes.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        let level = self.level.statistic().constant_maps_to(value)?;
        let (centre, spread) = self.removal.slots(level);
        Some(normalise_value(value, centre, spread, self.removal.floor()))
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Normalise every voxel against a local centre and a local spread, so that a
/// given contrast means the same thing everywhere in the volume.
///
/// `(value - centre) / max(spread, floor)`, with both estimates sampled and
/// interpolated exactly as [`LevelCorrectionOp`]'s single one is. Removing the
/// centre is what makes the quotient a statement about *contrast* rather than
/// about level: without it, a region twice as bright normalises to twice the
/// value at equal contrast.
///
/// **Two statistics, two lattices, independently parameterised.** The centre and
/// the spread are separate [`LocalStatistic`]s, each with its own window,
/// spacing and reducer, because the sensible choices are not the same shape — a
/// centre wants an estimator that ignores the structure (an order statistic, or
/// a mean over a wide window), a spread wants one that measures it. Nothing here
/// requires them to share a lattice, and both are anchored to the volume, so a
/// block evaluates both at the same global sample points the whole volume would.
///
/// **The floor is what makes this bounded.** Where the local spread is near
/// zero — a flat region, an empty one — an unbounded quotient turns whatever
/// noise is there into full-scale output. `max(spread, floor)` caps the gain at
/// `1 / floor`, and the cap is a parameter because how much amplification is
/// tolerable is a fact about the caller's data.
///
/// **What this deliberately does not do.** No output scale and no ceiling: both
/// are voxelwise maps of the result, both are already expressible as a
/// [`VoxelwiseMapOp`](super::voxelwise::VoxelwiseMapOp) composed after this one
/// at reach zero, and an op that absorbs its neighbours' work is an op with more
/// parameters to get wrong. The floor cannot be composed that way — it applies
/// to the estimate, inside the division — which is why it is here and they are
/// not.
#[derive(Debug)]
pub struct LocalContrastOp {
    name: &'static str,
    centre: LocalStatistic,
    spread: LocalStatistic,
    floor: f64,
    cost: f64,
}

impl LocalContrastOp {
    pub fn new(
        name: &'static str,
        centre: LocalStatistic,
        spread: LocalStatistic,
        floor: f64,
    ) -> Result<Self> {
        check_floor(floor, name)?;
        let cost = cost_for(&centre) + cost_for(&spread) + CONTRAST_COMBINE_COST;
        Ok(Self {
            name,
            centre,
            spread,
            floor,
            cost,
        })
    }

    pub fn centre(&self) -> &LocalStatistic {
        &self.centre
    }

    pub fn spread(&self) -> &LocalStatistic {
        &self.spread
    }

    pub fn floor(&self) -> f64 {
        self.floor
    }

    /// Override the composed cost. See [`cost_per_voxel`](BlockOp::cost_per_voxel).
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for LocalContrastOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The **larger** of the two statistics' reaches, per axis.
    ///
    /// Not the sum. The two estimates are taken from the same buffer rather than
    /// one from the other's output, so the set of voxels this op's answer
    /// depends on is the union of two neighbourhoods about the same voxel, and
    /// the half-width of a union of two centred intervals is the larger of the
    /// two half-widths. Summing would be safe and would cost halo for nothing;
    /// `a_short_reach_is_visible_in_the_output` is what keeps this honest in the
    /// other direction.
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        self.centre
            .reach(axis, volume_len)
            .max(self.spread.reach(axis, volume_len))
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let input = input.view::<f64>()?;
        let mut centre = Array3::<f64>::zeros(input.dim());
        let mut spread = Array3::<f64>::zeros(input.dim());
        self.centre.evaluate_into(input, at, centre.view_mut())?;
        self.spread.evaluate_into(input, at, spread.view_mut())?;
        normalise_against_into(
            input,
            Some(centre.view()),
            Some(spread.view()),
            self.floor,
            out.view_mut::<f64>()?,
        )
    }

    /// Only where **both** statistics declare, because the answer depends on
    /// both. The usual case is an empty block: a mean and a deviation are each
    /// exactly zero over a window of zeros, so the quotient is `0.0 / floor`,
    /// which is zero and is what the kernel computes for such a block.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        let centre = self.centre.statistic().constant_maps_to(value)?;
        let spread = self.spread.statistic().constant_maps_to(value)?;
        Some(normalise_value(
            value,
            Some(centre),
            Some(spread),
            self.floor,
        ))
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Measured; see [`COST_MEASUREMENT`](super::COST_MEASUREMENT) for the method
/// and `print_the_cost_of_these_ops` below for the run that takes it again.
///
/// **What was measured.** `96 x 64 x 64`, `--release`, best of five per op, four
/// runs, on a machine that was busy — so the minimum across runs is the figure,
/// for the reason `ops::cost` gives: the contamination is one-sided. Relative to
/// the voxelwise map, which is this module's unit:
///
/// | op | relative |
/// |---|--:|
/// | voxelwise map | 1.00 |
/// | local mean, 729-voxel window, spacing 8 | 4.91 |
/// | local deviation, same | 5.44 |
/// | level correction (subtract) over that mean | 5.25 |
/// | level correction (divide) over that mean | 5.20 |
/// | local contrast over that mean and that deviation | 11.56 |
///
/// **What the deltas say.** Removing an estimate costs `5.25 - 4.91 = 0.34`
/// beyond producing it, which is a two-operand voxelwise combination and is what
/// [`COMBINE_COST`]'s measured `0.49` already prices — the ops above therefore
/// reuse that constant rather than introduce a second one, and it is
/// conservative here by about a sixth of a voxelwise map, on a five-unit op.
///
/// The three-operand combination has no such constant to reuse: `11.56 - 4.91 -
/// 5.44 = 1.21`. It is more than twice the two-operand figure because there are
/// two whole `f64` estimate buffers to allocate, write and stream back rather
/// than one, and at this arithmetic intensity that traffic is the cost.
pub(super) const CONTRAST_COMBINE_COST: f64 = 1.21;

#[cfg(test)]
mod tests {
    use super::super::element::{ElementShape, Rank, StructuringElement};
    use super::super::local::{axis_max_distance, SampleLattice, Statistic};
    use super::*;

    fn box_element(radius: [usize; 3]) -> StructuringElement {
        StructuringElement::from_radius(ElementShape::Box, radius)
    }

    fn ramp(shape: [usize; 3]) -> Array3<f64> {
        Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
            ((i * 7919 + j * 104729 + k * 1299709) % 1013) as f64
        })
    }

    fn statistic(spacing: [usize; 3], statistic: Statistic) -> LocalStatistic {
        LocalStatistic::new(box_element([1, 1, 1]), spacing, statistic).unwrap()
    }

    fn apply(op: &dyn BlockOp, input: &Array3<f64>, at: &Anchor) -> Array3<f64> {
        let source: Voxels = input.clone().into();
        let mut out = Voxels::zeros(
            crate::dtype::Dtype::F64,
            [input.dim().0, input.dim().1, input.dim().2],
        )
        .unwrap();
        op.apply(&source, &mut out, at).unwrap();
        out.view::<f64>().unwrap().to_owned()
    }

    // ------------------------------------------------------- the kernel --

    #[test]
    fn the_kernel_is_the_arithmetic_it_says_it_is() {
        let input = Array3::from_shape_vec((2, 1, 1), vec![10.0_f64, 4.0]).unwrap();
        let centre = Array3::from_shape_vec((2, 1, 1), vec![3.0_f64, 4.0]).unwrap();
        let spread = Array3::from_shape_vec((2, 1, 1), vec![2.0_f64, 0.0]).unwrap();

        let mut out = Array3::zeros((2, 1, 1));
        normalise_against_into(input.view(), Some(centre.view()), None, 0.0, out.view_mut())
            .unwrap();
        assert_eq!(out[[0, 0, 0]], 7.0);
        assert_eq!(out[[1, 0, 0]], 0.0);

        normalise_against_into(input.view(), None, Some(spread.view()), 0.5, out.view_mut())
            .unwrap();
        assert_eq!(out[[0, 0, 0]], 5.0);
        // the floor is what the second voxel divides by, its spread being zero
        assert_eq!(out[[1, 0, 0]], 8.0);

        normalise_against_into(
            input.view(),
            Some(centre.view()),
            Some(spread.view()),
            0.5,
            out.view_mut(),
        )
        .unwrap();
        assert_eq!(out[[0, 0, 0]], 3.5);
        assert_eq!(out[[1, 0, 0]], 0.0);
    }

    /// The kernel takes a narrower input than the estimates it is combined with,
    /// which is the generality the algorithm allows and the shells cannot yet
    /// use.
    #[test]
    fn the_kernel_is_generic_over_the_input_element_type() {
        let input = Array3::from_shape_vec((2, 1, 1), vec![200_u8, 100]).unwrap();
        let spread = Array3::from_shape_vec((2, 1, 1), vec![4.0_f64, 0.0]).unwrap();
        let mut out = Array3::zeros((2, 1, 1));
        normalise_against_into(input.view(), None, Some(spread.view()), 2.0, out.view_mut())
            .unwrap();
        assert_eq!(out[[0, 0, 0]], 50.0);
        assert_eq!(out[[1, 0, 0]], 50.0);
    }

    #[test]
    fn the_kernel_refuses_a_combination_that_is_a_copy() {
        let input = Array3::<f64>::zeros((2, 1, 1));
        let mut out = Array3::zeros((2, 1, 1));
        let err = normalise_against_into(input.view(), None, None, 1.0, out.view_mut())
            .unwrap_err()
            .to_string();
        assert!(err.contains("nothing to normalise against"), "got: {err}");
    }

    #[test]
    fn a_floor_that_could_not_bound_a_divisor_is_refused_at_construction() {
        for floor in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = LevelCorrectionOp::new(
                "level",
                statistic([4, 4, 4], Statistic::Mean),
                Removal::Divide { floor },
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("finite and strictly positive"), "got: {err}");

            let err = LocalContrastOp::new(
                "contrast",
                statistic([4, 4, 4], Statistic::Mean),
                statistic([4, 4, 4], Statistic::Deviation),
                floor,
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("finite and strictly positive"), "got: {err}");
        }
        // and a subtraction has no divisor to floor, so it needs none
        assert!(LevelCorrectionOp::new(
            "level",
            statistic([4, 4, 4], Statistic::Mean),
            Removal::Subtract
        )
        .is_ok());
    }

    // --------------------------------------------------------- the reach --

    /// Both terms of the statistic's reach, carried through unchanged, and the
    /// maximum rather than the sum when there are two statistics.
    #[test]
    fn the_reach_is_the_statistics_and_the_combination_adds_nothing() {
        // volume 64, spacing 8 -> interpolation term 7; box radius 1 -> window
        // term 1
        assert_eq!(axis_max_distance(64, 8), 7);
        let level = LevelCorrectionOp::new(
            "level",
            statistic([8, 8, 8], Statistic::Mean),
            Removal::Subtract,
        )
        .unwrap();
        assert_eq!(level.reach(0, 64), 7 + 1);

        // a spacing of one is the no-lattice case and contributes nothing
        let dense = LevelCorrectionOp::new(
            "level",
            statistic([1, 1, 1], Statistic::Mean),
            Removal::Divide { floor: 1.0 },
        )
        .unwrap();
        assert_eq!(dense.reach(0, 64), 1);

        // two statistics: the larger per axis, not the sum
        let contrast = LocalContrastOp::new(
            "contrast",
            statistic([8, 8, 8], Statistic::Mean),
            statistic([3, 3, 3], Statistic::Deviation),
            0.01,
        )
        .unwrap();
        assert_eq!(axis_max_distance(64, 3), 2);
        // the centre reaches 7 + 1 and the spread 2 + 1
        assert_eq!(contrast.reach(0, 64), 8, "the larger of the two, per axis");
        assert_ne!(contrast.reach(0, 64), 11, "and never their sum");
    }

    // ---------------------------------------------------- the anchoring --

    /// **The property this file is exposed to**, with the defect it avoids
    /// demonstrated first so that the assertion below is not vacuous.
    ///
    /// One fixed voxel; two extents that both contain it with room for the
    /// derived reach. A lattice laid out over *the array handed in* — which is
    /// the construction `XY_BLOCK_SPLITTING.md` records — puts that voxel's
    /// nearest sample at a **different global coordinate** in each extent, so
    /// the estimate removed from it, and therefore the corrected value,
    /// depends on which block it landed in. That is shown here rather than
    /// described. Then the ops are run over both extents and over the whole
    /// volume, and the corrected voxel is the same in all three.
    #[test]
    fn an_unanchored_lattice_moves_with_the_block_and_these_ops_do_not() {
        let volume = [37usize, 12, 11];
        let spacing = [5usize, 3, 4];
        let voxel = [20usize, 5, 6];
        let input = ramp(volume);
        let extents = [(14usize, 14usize), (8, 20)];

        let level = LevelCorrectionOp::new(
            "level",
            statistic(spacing, Statistic::Rank(Rank::lowest())),
            Removal::Subtract,
        )
        .unwrap();
        let contrast = LocalContrastOp::new(
            "contrast",
            statistic(spacing, Statistic::Mean),
            statistic([4, 3, 5], Statistic::Deviation),
            0.5,
        )
        .unwrap();

        for op in [&level as &dyn BlockOp, &contrast as &dyn BlockOp] {
            for &(offset, len) in &extents {
                assert!(
                    voxel[0] - offset >= op.reach(0, volume[0])
                        && offset + len - voxel[0] > op.reach(0, volume[0]),
                    "the voxel must be trustworthy in both extents or this test proves nothing"
                );
            }
        }

        // The defect: where the nearest sample sits when the lattice is laid out
        // over the array that was handed in.
        let unanchored_sample = |offset: usize, len: usize| {
            let block = SampleLattice::centred([len, volume[1], volume[2]], spacing).unwrap();
            offset + block.centre(0, block.index_below(0, voxel[0] - offset))
        };
        assert_eq!(unanchored_sample(extents[0].0, extents[0].1), 18);
        assert_eq!(unanchored_sample(extents[1].0, extents[1].1), 20);
        assert_ne!(
            unanchored_sample(extents[0].0, extents[0].1),
            unanchored_sample(extents[1].0, extents[1].1),
            "if these agreed there would be no defect and the rest of this test would \
             be measuring nothing"
        );
        // The anchored lattice, which is the one that runs, puts it at 18 for
        // every extent — which one of the two unanchored layouts happens to
        // agree with. That coincidence is not a weakness of this test, it is the
        // reason a single decomposition agreeing proves nothing: the extent that
        // agrees here is wrong about other voxels and about the other axes, and
        // only the sweep below sees it.
        let anchored = SampleLattice::centred(volume, spacing).unwrap();
        assert_eq!(anchored.centre(0, anchored.index_below(0, voxel[0])), 18);

        // The ops: the whole-volume answer from either extent, bit for bit.
        for op in [&level as &dyn BlockOp, &contrast as &dyn BlockOp] {
            let whole = apply(op, &input, &Anchor::whole(volume));
            for &(offset, len) in &extents {
                let piece = input
                    .slice(ndarray::s![offset..offset + len, .., ..])
                    .to_owned();
                let got = apply(op, &piece, &Anchor::new([offset, 0, 0], volume));
                assert_eq!(
                    got[[voxel[0] - offset, voxel[1], voxel[2]]],
                    whole[voxel],
                    "{}: extent {offset}..{}",
                    op.name(),
                    offset + len
                );
            }
        }
    }

    /// Decomposition invariance asserted at the kernel, over every voxel of a
    /// trustworthy core rather than one: a sub-buffer with a halo of the derived
    /// reach reproduces the whole-volume answer exactly.
    #[test]
    fn a_sub_buffer_reproduces_the_whole_volume_answer_inside_its_trustworthy_extent() {
        let volume = [29usize, 13, 11];
        let input = ramp(volume);
        let ops: Vec<Box<dyn BlockOp>> = vec![
            Box::new(
                LevelCorrectionOp::new(
                    "subtract",
                    statistic([3, 2, 4], Statistic::Rank(Rank::lowest())),
                    Removal::Subtract,
                )
                .unwrap(),
            ),
            Box::new(
                LevelCorrectionOp::new(
                    "divide",
                    statistic([7, 5, 3], Statistic::Mean),
                    Removal::Divide { floor: 1.0 },
                )
                .unwrap(),
            ),
            Box::new(
                LocalContrastOp::new(
                    "contrast",
                    statistic([5, 3, 3], Statistic::Mean),
                    statistic([2, 4, 6], Statistic::Deviation),
                    0.25,
                )
                .unwrap(),
            ),
        ];
        for op in &ops {
            let whole = apply(op.as_ref(), &input, &Anchor::whole(volume));
            let reach = op.reach(0, volume[0]);
            let core = 10usize..20;
            let start = core.start.saturating_sub(reach);
            let end = (core.end + reach).min(volume[0]);
            let piece = input.slice(ndarray::s![start..end, .., ..]).to_owned();
            let got = apply(op.as_ref(), &piece, &Anchor::new([start, 0, 0], volume));
            for i in core.clone() {
                for j in 0..volume[1] {
                    for k in 0..volume[2] {
                        assert_eq!(
                            got[[i - start, j, k]],
                            whole[[i, j, k]],
                            "{} at {i},{j},{k}",
                            op.name()
                        );
                    }
                }
            }
        }
    }

    // --------------------------------------------- the constant algebra --

    /// What is declared is what is computed, **bit for bit**, which is the whole
    /// claim of `constant_maps_to` and the one thing about it that can be
    /// checked rather than argued.
    #[test]
    fn a_declared_constant_is_the_computed_one_bit_for_bit() {
        let volume = [9usize, 9, 9];
        let element = box_element([1, 1, 1]);
        let rank = LocalStatistic::new(
            element.clone(),
            [3, 3, 3],
            Statistic::Rank(Rank::median(&element)),
        )
        .unwrap();
        let ops: Vec<(f64, Box<dyn BlockOp>)> = vec![
            (
                0.1,
                Box::new(
                    LevelCorrectionOp::new("subtract", rank.clone(), Removal::Subtract).unwrap(),
                ),
            ),
            (
                0.1,
                Box::new(
                    LevelCorrectionOp::new("divide", rank.clone(), Removal::Divide { floor: 0.01 })
                        .unwrap(),
                ),
            ),
            (
                0.0,
                Box::new(
                    LocalContrastOp::new(
                        "contrast",
                        statistic([3, 3, 3], Statistic::Mean),
                        statistic([3, 3, 3], Statistic::Deviation),
                        0.25,
                    )
                    .unwrap(),
                ),
            ),
            (
                0.1,
                Box::new(LocalContrastOp::new("contrast", rank.clone(), rank, 0.25).unwrap()),
            ),
        ];
        for (value, op) in &ops {
            let declared = op
                .constant_maps_to(*value)
                .unwrap_or_else(|| panic!("{} declares nothing at {value}", op.name()));
            let input = Array3::from_elem((volume[0], volume[1], volume[2]), *value);
            let computed = apply(op.as_ref(), &input, &Anchor::whole(volume));
            assert!(
                computed.iter().all(|&got| got == declared),
                "{}: declared {declared}, computed {:?}",
                op.name(),
                computed.iter().find(|&&got| got != declared)
            );
        }
        // the concrete algebra, stated so a change to it is visible
        assert_eq!(
            LevelCorrectionOp::new(
                "subtract",
                statistic([3, 3, 3], Statistic::Rank(Rank::lowest())),
                Removal::Subtract
            )
            .unwrap()
            .constant_maps_to(0.1),
            Some(0.0),
            "an order statistic of a constant is that constant, so subtracting it is zero"
        );
        assert_eq!(
            LevelCorrectionOp::new(
                "divide",
                statistic([3, 3, 3], Statistic::Rank(Rank::lowest())),
                Removal::Divide { floor: 0.01 }
            )
            .unwrap()
            .constant_maps_to(0.1),
            Some(1.0),
            "and dividing by it is one, exactly, wherever the constant clears the floor"
        );
    }

    /// The other half: what the statistic underneath withholds, these ops
    /// withhold, and they do not invent a mapping the arithmetic cannot make
    /// exact.
    #[test]
    fn what_the_statistic_withholds_these_ops_withhold() {
        let mean = LevelCorrectionOp::new(
            "subtract",
            statistic([4, 4, 4], Statistic::Mean),
            Removal::Subtract,
        )
        .unwrap();
        assert_eq!(mean.constant_maps_to(0.0), Some(0.0));
        assert_eq!(
            mean.constant_maps_to(0.1),
            None,
            "the mean of m copies of 0.1 is not 0.1, so nothing downstream of it may claim \
             to know the answer"
        );

        // one statistic withholding is enough to withhold the pair
        let mixed = LocalContrastOp::new(
            "contrast",
            statistic([4, 4, 4], Statistic::Rank(Rank::lowest())),
            statistic([4, 4, 4], Statistic::Deviation),
            0.25,
        )
        .unwrap();
        assert_eq!(mixed.constant_maps_to(0.0), Some(0.0));
        assert_eq!(mixed.constant_maps_to(0.1), None);
    }

    /// The gap the declaration is protecting against, made concrete: a mean over
    /// a constant volume is *not* that constant, so a `Some(0.0)` for a
    /// subtraction at every value would be wrong in the last bit rather than in
    /// principle.
    #[test]
    fn the_mean_leaves_a_residue_a_subtraction_would_have_to_lie_about() {
        let volume = [5usize, 1, 1];
        let op = LevelCorrectionOp::new(
            "subtract",
            LocalStatistic::new(box_element([1, 0, 0]), [1, 1, 1], Statistic::Mean).unwrap(),
            Removal::Subtract,
        )
        .unwrap();
        let input = Array3::from_elem((5, 1, 1), 0.1_f64);
        let got = apply(&op, &input, &Anchor::whole(volume));
        assert_ne!(
            got[[2, 0, 0]],
            0.0,
            "if this ever becomes exact the declaration can be widened, but it must be \
             widened deliberately"
        );
    }

    // ---------------------------------------------------------- the cost --

    /// Retaking the measurement these ops' constants rest on. Ignored for the
    /// reason `ops::cost` gives — timing in a test suite measures the machine's
    /// mood — but it runs, and it is one command:
    ///
    /// ```text
    /// cargo test --release -- --ignored --nocapture ops::normalise::tests::print_the_cost
    /// ```
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn print_the_cost_of_these_ops() {
        use std::time::Instant;

        let shape = [96usize, 64, 64];
        let voxels = (shape[0] * shape[1] * shape[2]) as f64;
        let input: Voxels = ramp(shape).into();
        let anchor = Anchor::whole(shape);
        let window = StructuringElement::from_size(ElementShape::Box, [9, 9, 9]).unwrap();
        let wide = |statistic| LocalStatistic::new(window.clone(), [8, 8, 8], statistic).unwrap();

        let cases: Vec<(String, Box<dyn BlockOp>)> = vec![
            (
                "voxelwise map (the unit)".to_string(),
                Box::new(super::super::voxelwise::VoxelwiseMapOp::threshold(
                    "map", 500.0, 1.0, 0.0,
                )),
            ),
            (
                "local mean, 729-voxel window, spacing 8".to_string(),
                Box::new(super::super::local::LocalStatisticOp::new(
                    "mean",
                    wide(Statistic::Mean),
                )),
            ),
            (
                "local deviation, 729-voxel window, spacing 8".to_string(),
                Box::new(super::super::local::LocalStatisticOp::new(
                    "deviation",
                    wide(Statistic::Deviation),
                )),
            ),
            (
                "level correction (subtract), same estimate".to_string(),
                Box::new(
                    LevelCorrectionOp::new("level", wide(Statistic::Mean), Removal::Subtract)
                        .unwrap(),
                ),
            ),
            (
                "level correction (divide), same estimate".to_string(),
                Box::new(
                    LevelCorrectionOp::new(
                        "level",
                        wide(Statistic::Mean),
                        Removal::Divide { floor: 1.0 },
                    )
                    .unwrap(),
                ),
            ),
            (
                "local contrast, two such estimates".to_string(),
                Box::new(
                    LocalContrastOp::new(
                        "contrast",
                        wide(Statistic::Mean),
                        wide(Statistic::Deviation),
                        1.0,
                    )
                    .unwrap(),
                ),
            ),
        ];

        let mut unit = 1.0;
        println!(
            "{:<48} {:>10} {:>10} {:>10}",
            "op", "ns/voxel", "relative", "declared"
        );
        for (index, (name, op)) in cases.iter().enumerate() {
            let mut out =
                Voxels::zeros(op.produces(input.dtype()), op.output_shape(shape)).unwrap();
            op.apply(&input, &mut out, &anchor).unwrap();
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let started = Instant::now();
                op.apply(&input, &mut out, &anchor).unwrap();
                let elapsed = started.elapsed().as_secs_f64() * 1e9;
                std::hint::black_box(out.view::<f64>().unwrap().iter().take(1).sum::<f64>());
                best = best.min(elapsed / voxels);
            }
            if index == 0 {
                unit = best;
            }
            println!(
                "{name:<48} {best:>10.3} {:>10.2} {:>10.2}",
                best / unit,
                op.cost_per_voxel()
            );
        }
    }

    /// What can be asserted about a measured cost without measuring: the
    /// orderings the constants encode are the orderings the ops have. Removing
    /// an estimate costs more than the estimate alone; two estimates cost more
    /// than one.
    #[test]
    fn the_stored_costs_keep_the_order_the_measurement_found() {
        let estimate = statistic([8, 8, 8], Statistic::Mean);
        let bare = super::super::local::LocalStatisticOp::new("mean", estimate.clone());
        let level = LevelCorrectionOp::new("level", estimate.clone(), Removal::Subtract).unwrap();
        let contrast = LocalContrastOp::new(
            "contrast",
            estimate.clone(),
            statistic([8, 8, 8], Statistic::Deviation),
            0.25,
        )
        .unwrap();
        assert!(level.cost_per_voxel() > bare.cost_per_voxel());
        assert!(contrast.cost_per_voxel() > level.cost_per_voxel());

        // and the lattice is what divides the work, here as everywhere
        let dense = LevelCorrectionOp::new(
            "level",
            statistic([1, 1, 1], Statistic::Mean),
            Removal::Subtract,
        )
        .unwrap();
        assert!(dense.cost_per_voxel() > level.cost_per_voxel());
    }
}
