// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The rank filter: at each voxel, select an order statistic of a parameterised
// neighbourhood. The **median is the `k = n / 2` case** and shares this one
// implementation; there is no second median anywhere in the crate, which is the
// same discipline `docs/design/BLOCK_OPS.md` asks of a fused op — "call the ops'
// helpers; only the plumbing is yours".
//
// Edge behaviour, stated because it is the op's own business
// ---------------------------------------------------------
// The element is clamped to the array handed in. At a **real volume boundary**
// that is right: there is nothing beyond to read, and the whole-volume reference
// clamps identically. At a **block seam** it is wrong — there *is* something
// beyond — and that is precisely what makes a short halo diverge instead of
// passing quietly. The clamp is not a fallback; it is the detector.
//
// What clamping does *not* do is decide which statistic is taken; the `Rank`
// does, and it carries **two conventions** for a truncated window because there
// are two defensible ones. `Rank::Nth` rescales the rank to the surviving
// population and `Rank::CeilingPercentile` states the statistic against that
// population directly; they agree at a half over an odd population and differ
// elsewhere, including on the untruncated window. See `Rank::resolve` for the
// arithmetic and `tests/rank_truncation_rule.rs` for both halves of that,
// pinned. What neither does is *clip* the rank, which would turn a median into
// a maximum wherever the element is truncated.
//
// The window is the element's offsets, so an element that is **stepped** —
// `StructuringElement::from_size_stepped` — simply gathers fewer of them, and
// every number this file derives from `element.len()` follows without a second
// path: the cost below, and the rank a `median` constructor names.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp};
use crate::reach::Reach;
use crate::voxels::{VoxelElement, Voxels};

use super::element::{select_nth, Rank, StructuringElement, Total};
use super::shapes_agree;

/// Select `rank` of `element` around every voxel of `input`.
///
/// Generic over `Ord`, which is exactly the requirement the algorithm has: it
/// compares values and hands one of them back, never combining two. The value
/// written is a value that was read, bit for bit.
///
/// `out` may not alias `input` — the filter is not in-place, because a rank read
/// from a partially overwritten array is a different filter.
pub fn rank_filter_into<T: Ord + Copy>(
    input: ArrayView3<'_, T>,
    element: &StructuringElement,
    rank: Rank,
    out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    selecting(input, None, element, rank, out, "rank_filter_into")
}

/// [`rank_filter_into`], with `mask` deciding which voxels are in the window's
/// **population**.
///
/// The mask is consulted at *every offset in the element*, not at the centre —
/// that is the whole of the difference, and it is why this cannot be done as a
/// voxelwise pre-step. **The obvious workaround does not work**: replacing
/// excluded voxels with a sentinel and running the ordinary filter fails,
/// because a rank is resolved against the count of surviving voxels and a
/// sentinel keeps them in the count. It is worth stating because it is the
/// first thing anyone tries.
///
/// `mask` covers the same buffer as `input`, so the caller reads it at the same
/// region — see `BlockOp::source_inputs`, which is what makes the plan fetch it
/// at the element's reach rather than at the voxel.
///
/// **When nothing survives**, the centre's own value is written. That is the
/// only answer that keeps the filter's defining property — every value written
/// is a value that was read — and it is stated here rather than left to the
/// selection, which has no value to hand back at all. The caller should know
/// that where the centre is *itself* excluded this returns a value the mask
/// asked to leave out; a window with no population has no better answer, and
/// pretending otherwise would be inventing one.
pub fn masked_rank_filter_into<T: Ord + Copy>(
    input: ArrayView3<'_, T>,
    mask: ArrayView3<'_, bool>,
    element: &StructuringElement,
    rank: Rank,
    out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    shapes_agree(
        input.shape(),
        mask.shape(),
        "masked_rank_filter_into (mask)",
    )?;
    selecting(
        input,
        Some(mask),
        element,
        rank,
        out,
        "masked_rank_filter_into",
    )
}

/// The one selection kernel, masked or not.
///
/// One function rather than two so that the masked filter cannot drift from the
/// plain one: the gather, the clamp at the buffer edge and
/// [`Rank::resolve`]'s truncation rule are the same code, and masking is one
/// more reason an offset does not join the window. The unmasked path is
/// unchanged down to the error it raises.
fn selecting<T: Ord + Copy>(
    input: ArrayView3<'_, T>,
    mask: Option<ArrayView3<'_, bool>>,
    element: &StructuringElement,
    rank: Rank,
    mut out: ArrayViewMut3<'_, T>,
    what: &str,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), what)?;
    if element.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "{what}: an empty element selects nothing"
        )));
    }
    let extent = [
        input.shape()[0] as isize,
        input.shape()[1] as isize,
        input.shape()[2] as isize,
    ];
    let full = element.len();
    let mut window: Vec<T> = Vec::with_capacity(full);
    for i in 0..input.shape()[0] {
        for j in 0..input.shape()[1] {
            for k in 0..input.shape()[2] {
                window.clear();
                let centre = [i as isize, j as isize, k as isize];
                for offset in element.offsets() {
                    let a = centre[0] + offset[0];
                    let b = centre[1] + offset[1];
                    let c = centre[2] + offset[2];
                    if a < 0 || b < 0 || c < 0 || a >= extent[0] || b >= extent[1] || c >= extent[2]
                    {
                        continue;
                    }
                    let at = [a as usize, b as usize, c as usize];
                    if let Some(mask) = mask.as_ref() {
                        if !mask[at] {
                            continue;
                        }
                    }
                    window.push(input[at]);
                }
                let index = rank.resolve(full, window.len());
                match select_nth(&mut window, index) {
                    Some(value) => out[[i, j, k]] = value,
                    // Unmasked, an empty window means the element does not
                    // contain its own centre, which is a malformed element and
                    // an error. Masked, it means the mask excluded every
                    // neighbour, which is ordinary data — see the header of
                    // `masked_rank_filter_into` for why the centre is the
                    // answer.
                    None if mask.is_some() => out[[i, j, k]] = input[[i, j, k]],
                    None => {
                        return Err(Error::InvalidArgument(format!(
                            "{what}: an element that misses its own centre"
                        )))
                    }
                }
            }
        }
    }
    Ok(())
}

/// `rank_filter_into` over a `f64` volume, through the total order.
///
/// The copy into `Total` is what **floating point** costs: `f64` is not `Ord`,
/// so the ordered kernel cannot see it directly. Now that the element type is a
/// tag, an integer or `bool` volume reaches the kernel with no copy at all —
/// see `RankFilterOp::apply` — and this wrapper is the floating-point case it
/// was always for.
pub fn rank_filter_f64_into(
    input: ArrayView3<'_, f64>,
    element: &StructuringElement,
    rank: Rank,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), "rank_filter_f64_into")?;
    let ordered = input.mapv(Total);
    let mut selected = Array3::from_elem(ordered.raw_dim(), Total(0.0));
    rank_filter_into(ordered.view(), element, rank, selected.view_mut())?;
    ndarray::Zip::from(&mut out)
        .and(&selected)
        .for_each(|slot, value| *slot = value.0);
    Ok(())
}

/// Select an order statistic of a parameterised neighbourhood at every voxel.
pub struct RankFilterOp {
    name: &'static str,
    element: StructuringElement,
    rank: Rank,
    cost: f64,
}

impl RankFilterOp {
    pub fn new(name: &'static str, element: StructuringElement, rank: Rank) -> Self {
        let cost = cost_for(&element);
        Self {
            name,
            element,
            rank,
            cost,
        }
    }

    /// The `k = n / 2` case of the same op. Not a separate implementation, and
    /// not a separate type — a constructor, so that a reader can see that the
    /// median has no code of its own.
    pub fn median(name: &'static str, element: StructuringElement) -> Self {
        let rank = Rank::median(&element);
        Self::new(name, element, rank)
    }

    pub fn element(&self) -> &StructuringElement {
        &self.element
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for RankFilterOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The element's wider side. **Derived, with nothing to configure it** — an
    /// element of size 7 reaches 3 and there is no field that could say
    /// otherwise.
    ///
    /// The bound; [`Self::reach_spec`] is the exact statement.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.element.reach(axis)
    }

    /// The element's two sides, per axis, which is what the window actually
    /// reads. An element of size 10 reads five below the anchor and four above,
    /// and declaring five on both would fetch a plane per block that no voxel
    /// of the answer depends on.
    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        self.element.reach_spec()
    }

    /// Every ordered element type, and the two floats through the total order.
    ///
    /// The kernel asks for `Ord` and nothing else, so this is the widest set the
    /// shell can honestly bridge. `f16` is not in it because no buffer holds
    /// one.
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        /// The integer and `bool` case: straight into the kernel, no copy.
        fn ordered<T: VoxelElement + Ord>(
            input: &Voxels,
            out: &mut Voxels,
            element: &StructuringElement,
            rank: Rank,
        ) -> Result<()> {
            let source = input.view::<T>()?;
            rank_filter_into(source, element, rank, out.view_mut::<T>()?)
        }

        match input.dtype() {
            Dtype::Bool => ordered::<bool>(input, out, &self.element, self.rank),
            Dtype::U8 => ordered::<u8>(input, out, &self.element, self.rank),
            Dtype::U16 => ordered::<u16>(input, out, &self.element, self.rank),
            Dtype::U32 => ordered::<u32>(input, out, &self.element, self.rank),
            Dtype::U64 => ordered::<u64>(input, out, &self.element, self.rank),
            Dtype::I8 => ordered::<i8>(input, out, &self.element, self.rank),
            Dtype::I16 => ordered::<i16>(input, out, &self.element, self.rank),
            Dtype::I32 => ordered::<i32>(input, out, &self.element, self.rank),
            Dtype::I64 => ordered::<i64>(input, out, &self.element, self.rank),
            // The floats have no total order of their own, so they take the
            // `Total` detour. `f32` widens to `f64` on the way in and back on
            // the way out, which is exact both ways because the filter selects a
            // value it read rather than combining two.
            Dtype::F64 => rank_filter_f64_into(
                input.view::<f64>()?,
                &self.element,
                self.rank,
                out.view_mut::<f64>()?,
            ),
            Dtype::F32 => {
                let widened = input.view::<f32>()?.mapv(f64::from);
                let mut selected = Array3::zeros(widened.raw_dim());
                rank_filter_f64_into(
                    widened.view(),
                    &self.element,
                    self.rank,
                    selected.view_mut(),
                )?;
                let mut out = out.view_mut::<f32>()?;
                ndarray::Zip::from(&mut out)
                    .and(&selected)
                    .for_each(|slot, &value| *slot = value as f32);
                Ok(())
            }
            Dtype::F16 => Err(Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; `accepts` refuses it before a run starts",
                self.name
            ))),
        }
    }

    /// Exactly the constant, for **every** rank and at every truncation.
    ///
    /// The filter selects a value that was read; if every value read is `value`,
    /// the value selected is `value`. Nothing is summed, averaged or rounded, so
    /// this holds bit for bit rather than approximately — which is the standard
    /// this declaration has to meet, since a short-circuited block must produce
    /// exactly what a computed one would have.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// [`RankFilterOp`], with a stored level deciding which voxels count.
///
/// **The case it exists for** is a percentile taken over a window with
/// background voxels removed from the population, so that the statistic
/// describes the structure present rather than the proportion of the window it
/// occupies. It is a `BlockOp` like any other; what is new is that it reads a
/// *second* level over the same window it reads its input over, which is what
/// [`BlockOp::source_inputs`] exists to declare.
///
/// **Why a stored mask rather than a predicate baked into the kernel.** Both
/// are legitimate and the choice is a cost question, not a correctness one. The
/// mask here is read `|element|` times — once per window covering it — which is
/// the condition that favours computing it once and storing it. A predicate
/// cheap enough to beat a byte of memory traffic (a bare comparison, say) is
/// better fused into the gather, and that is a fusion of this op rather than a
/// different op: it produces the same answer from the same declaration, which
/// is exactly the property that lets a planner choose between them later.
pub struct MaskedRankFilterOp {
    name: &'static str,
    element: StructuringElement,
    rank: Rank,
    mask: usize,
    cost: f64,
}

impl MaskedRankFilterOp {
    /// `mask` is the level holding the population, which must be a `Bool`
    /// level — checked when the plan is made, by the same `check_dtypes` fold
    /// that checks every other level.
    pub fn new(
        name: &'static str,
        element: StructuringElement,
        rank: Rank,
        mask: impl Into<crate::assemble::Level>,
    ) -> Self {
        let cost = cost_for(&element) * MASK_COST_FACTOR;
        Self {
            name,
            element,
            rank,
            mask: mask.into().index(),
            cost,
        }
    }

    /// The percentile form, which is the one the reference's arm needs.
    pub fn percentile(
        name: &'static str,
        element: StructuringElement,
        fraction: f64,
        mask: impl Into<crate::assemble::Level>,
    ) -> Result<Self> {
        let rank = Rank::ceiling_percentile(fraction)?;
        Ok(Self::new(name, element, rank, mask))
    }

    pub fn element(&self) -> &StructuringElement {
        &self.element
    }

    /// The level this op reads its population from.
    pub fn mask_level(&self) -> usize {
        self.mask
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for MaskedRankFilterOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.element.reach(axis)
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        self.element.reach_spec()
    }

    /// The mask, over **the same window** the input is read over.
    ///
    /// Equal to this op's own reach by construction rather than by arrangement:
    /// the population is consulted at the element's offsets, so it is the
    /// element that states both. That equality is also what keeps this op
    /// inside what a plan can currently fetch — see `check_source_levels`.
    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<crate::op::SourceInput> {
        vec![crate::op::SourceInput::new(
            self.mask,
            self.element.reach_spec(),
        )]
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    /// Refuses, and the refusal is the point: an op whose population comes from
    /// a second array cannot be computed from one. See [`BlockOp::apply_with`].
    fn apply(&self, _input: &Voxels, _out: &mut Voxels, _at: &Anchor) -> Result<()> {
        Err(Error::InvalidArgument(format!(
            "{}: the population comes from level {}, so this op has no answer from its input \
             alone. It is applied through `apply_with`.",
            self.name, self.mask
        )))
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: crate::op::SourceInputs<'_>,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<()> {
        let mask = sources.get(self.mask)?;
        if mask.dtype() != Dtype::Bool {
            return Err(Error::InvalidArgument(format!(
                "{}: the population is read from level {}, which holds {}. A population is a \
                 yes-or-no per voxel and is stored as one; a wider type would leave 'which \
                 non-zero values count' to be decided somewhere this op cannot see.",
                self.name,
                self.mask,
                mask.dtype().numpy_name()
            )));
        }
        let mask = mask.view::<bool>()?;

        fn ordered<T: VoxelElement + Ord>(
            input: &Voxels,
            mask: ArrayView3<'_, bool>,
            out: &mut Voxels,
            element: &StructuringElement,
            rank: Rank,
        ) -> Result<()> {
            let source = input.view::<T>()?;
            masked_rank_filter_into(source, mask, element, rank, out.view_mut::<T>()?)
        }

        /// The floating-point detour, shared by `f64` and `f32`: neither has a
        /// total order, and the filter hands back a value it read, so the trip
        /// through `Total` is exact in both directions.
        fn through_total(
            widened: ndarray::Array3<f64>,
            mask: ArrayView3<'_, bool>,
            element: &StructuringElement,
            rank: Rank,
        ) -> Result<Array3<f64>> {
            let ordered = widened.mapv(Total);
            let mut selected = Array3::from_elem(ordered.raw_dim(), Total(0.0));
            masked_rank_filter_into(ordered.view(), mask, element, rank, selected.view_mut())?;
            Ok(selected.mapv(|value| value.0))
        }

        match input.dtype() {
            Dtype::Bool => ordered::<bool>(input, mask, out, &self.element, self.rank),
            Dtype::U8 => ordered::<u8>(input, mask, out, &self.element, self.rank),
            Dtype::U16 => ordered::<u16>(input, mask, out, &self.element, self.rank),
            Dtype::U32 => ordered::<u32>(input, mask, out, &self.element, self.rank),
            Dtype::U64 => ordered::<u64>(input, mask, out, &self.element, self.rank),
            Dtype::I8 => ordered::<i8>(input, mask, out, &self.element, self.rank),
            Dtype::I16 => ordered::<i16>(input, mask, out, &self.element, self.rank),
            Dtype::I32 => ordered::<i32>(input, mask, out, &self.element, self.rank),
            Dtype::I64 => ordered::<i64>(input, mask, out, &self.element, self.rank),
            Dtype::F64 => {
                let selected = through_total(
                    input.view::<f64>()?.to_owned(),
                    mask,
                    &self.element,
                    self.rank,
                )?;
                out.view_mut::<f64>()?.assign(&selected);
                Ok(())
            }
            Dtype::F32 => {
                let selected = through_total(
                    input.view::<f32>()?.mapv(f64::from),
                    mask,
                    &self.element,
                    self.rank,
                )?;
                let mut out = out.view_mut::<f32>()?;
                ndarray::Zip::from(&mut out)
                    .and(&selected)
                    .for_each(|slot, &value| *slot = value as f32);
                Ok(())
            }
            Dtype::F16 => Err(Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; `accepts` refuses it before a run starts",
                self.name
            ))),
        }
    }

    /// Exactly the constant, and **whatever the mask says**.
    ///
    /// Every value written is a value that was read: where the population is
    /// non-empty the selection returns one of them, and where it is empty the
    /// centre is written. On a constant block both are the constant, so this
    /// holds bit for bit without the short circuit needing to know the mask —
    /// which matters, because the mask is a level the short circuit has not
    /// read.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// What consulting the population adds, as a multiple of the unmasked filter.
///
/// **A seed, in the sense `CostModel` means it** — it needs the ordering right
/// and nothing more, because a run that records statistics replaces it with a
/// measured coefficient for this op's family. One byte of a second array per
/// offset visited, against the compare-and-select already there.
const MASK_COST_FACTOR: f64 = 1.35;

/// Measured; see `super::COST_MEASUREMENT`. The filter's work is proportional to
/// the element it is given, so the cost is a function of the element rather than
/// a constant — a 27-voxel median and a 343-voxel median are not the same op as
/// far as a planner is concerned.
pub(super) fn cost_for(element: &StructuringElement) -> f64 {
    RANK_COST_PER_ELEMENT_VOXEL * element.len() as f64
}

/// Measured; see `super::COST_MEASUREMENT`.
pub(super) const RANK_COST_PER_ELEMENT_VOXEL: f64 = 3.87;

#[cfg(test)]
mod tests {
    use super::super::element::ElementShape;
    use super::*;
    use ndarray::Array3;

    fn ramp(shape: (usize, usize, usize)) -> Array3<f64> {
        let mut array = Array3::zeros(shape);
        for (flat, value) in array.iter_mut().enumerate() {
            *value = ((flat * 7919) % 1013) as f64;
        }
        array
    }

    /// The definition, written out, against the implementation. Not a second
    /// implementation for production — a statement of what the op means.
    fn by_definition(input: &Array3<f64>, element: &StructuringElement, rank: Rank) -> Array3<f64> {
        let shape = input.dim();
        let mut out = Array3::zeros(shape);
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    let mut window = Vec::new();
                    for offset in element.offsets() {
                        let a = i as isize + offset[0];
                        let b = j as isize + offset[1];
                        let c = k as isize + offset[2];
                        if a < 0 || b < 0 || c < 0 {
                            continue;
                        }
                        let (a, b, c) = (a as usize, b as usize, c as usize);
                        if a >= shape.0 || b >= shape.1 || c >= shape.2 {
                            continue;
                        }
                        window.push(input[[a, b, c]]);
                    }
                    window.sort_by(|left, right| left.total_cmp(right));
                    let index = rank.resolve(element.len(), window.len());
                    out[[i, j, k]] = window[index];
                }
            }
        }
        out
    }

    #[test]
    fn the_filter_agrees_with_the_definition_for_every_rank_and_shape() {
        let input = ramp((9, 7, 6));
        for shape in [ElementShape::Box, ElementShape::Ellipsoid] {
            for radius in [[1, 1, 1], [2, 1, 0], [0, 0, 3]] {
                let element = StructuringElement::from_radius(shape, radius);
                for rank in [
                    Rank::lowest(),
                    Rank::median(&element),
                    Rank::highest(&element),
                    Rank::Nth(element.len() / 4),
                ] {
                    let mut got = Array3::zeros(input.dim());
                    rank_filter_f64_into(input.view(), &element, rank, got.view_mut()).unwrap();
                    assert_eq!(
                        got,
                        by_definition(&input, &element, rank),
                        "{shape:?} {radius:?} {rank:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_median_is_the_half_rank_of_the_same_op() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let input = ramp((6, 5, 4));
        let mut through_median = Array3::zeros(input.dim());
        rank_filter_f64_into(
            input.view(),
            &element,
            Rank::median(&element),
            through_median.view_mut(),
        )
        .unwrap();
        let mut through_nth = Array3::zeros(input.dim());
        rank_filter_f64_into(
            input.view(),
            &element,
            Rank::Nth(element.len() / 2),
            through_nth.view_mut(),
        )
        .unwrap();
        assert_eq!(through_median, through_nth);
    }

    #[test]
    fn the_reach_is_the_radius_and_nothing_configures_it() {
        let op = RankFilterOp::median(
            "median",
            StructuringElement::from_size(ElementShape::Box, [7, 5, 1]).unwrap(),
        );
        assert_eq!(op.reach(0, 1000), 3);
        assert_eq!(op.reach(1, 1000), 2);
        assert_eq!(op.reach(2, 1000), 0);
    }

    #[test]
    fn a_constant_selects_that_constant() {
        let element = StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 2, 2]);
        let op = RankFilterOp::median("median", element);
        assert_eq!(op.constant_maps_to(0.0), Some(0.0));
        assert_eq!(op.constant_maps_to(-3.5), Some(-3.5));

        // and the declaration matches the computation, bit for bit
        let input = Array3::from_elem((5, 5, 5), -3.5);
        let mut out = Array3::zeros(input.dim());
        rank_filter_f64_into(
            input.view(),
            op.element(),
            Rank::median(op.element()),
            out.view_mut(),
        )
        .unwrap();
        assert!(out.iter().all(|&value| value == -3.5));
    }

    /// A rank filter on integers needs no wrapper at all, which is the point of
    /// the kernel being generic rather than `f64`-shaped.
    #[test]
    fn the_kernel_runs_on_an_ordered_element_type_directly() {
        let input = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| (i * 16 + j * 4 + k) as u16);
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 0, 0]);
        let mut out = Array3::<u16>::zeros(input.dim());
        rank_filter_into(input.view(), &element, Rank::lowest(), out.view_mut()).unwrap();
        assert_eq!(out[[0, 0, 0]], 0);
        assert_eq!(out[[2, 0, 0]], 16);
    }
}
