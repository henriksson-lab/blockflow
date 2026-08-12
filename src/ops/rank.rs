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
// What clamping does *not* do is change which statistic is taken. A rank is a
// relative position in the sorted window, so a truncated window rescales it
// rather than clipping it — see `Rank::resolve`, and the test there for what
// clipping would have done to a median at every face of the volume.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp};
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
    mut out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), "rank_filter_into")?;
    if element.is_empty() {
        return Err(Error::InvalidArgument(
            "rank_filter_into: an empty element selects nothing".to_string(),
        ));
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
                    window.push(input[[a as usize, b as usize, c as usize]]);
                }
                let index = rank.resolve(full, window.len());
                match select_nth(&mut window, index) {
                    Some(value) => out[[i, j, k]] = value,
                    None => {
                        return Err(Error::InvalidArgument(
                            "rank_filter_into: an element that misses its own centre".to_string(),
                        ))
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

    /// The element's radius. **Derived, with nothing to configure it** — an
    /// element of size 7 reaches 3 and there is no field that could say
    /// otherwise.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.element.reach(axis)
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
