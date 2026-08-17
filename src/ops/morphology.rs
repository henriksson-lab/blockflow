// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Binary morphology over a parameterised structuring element: erosion,
// dilation, and the two compositions of them.
//
// Opening and closing are written **once**, as compositions
// -------------------------------------------------------
// `open = dilate(erode(x))`, `close = erode(dilate(x))`, and neither has a loop
// of its own. Two consequences follow and both are worth having:
//
// * there is no second erosion to drift from the first;
// * the reach falls out of the composition rather than being asserted. An
//   opening reads **twice** the element's radius, because its dilation consumes
//   erosion values up to a radius away and each of those consumed input up to a
//   radius beyond that. A hand-written opening is exactly the kind of op whose
//   author writes `radius` in the halo formula and is wrong by a factor of two
//   with no symptom except at block seams — which is the failure
//   `docs/design/BLOCK_OPS.md` exists to remove.
//
// Edge behaviour
// --------------
// The element is clamped to the array handed in, and the clamp is the identity
// of the operation: erosion takes the conjunction over the voxels that exist, so
// what lies outside behaves as set; dilation takes the disjunction, so it
// behaves as clear. That is the standard "the boundary does not erode the image
// away" convention, it is right at a real volume boundary, and — as for the rank
// filter — it is deliberately *wrong* at a block seam, which is what makes a
// short halo visible instead of silent.
//
// Which offsets, and where they are asked for
// -------------------------------------------
// `sweep` asks the element what it reads **at each voxel's position in the
// volume**, through `StructuringElement::offsets_at`, rather than gathering one
// offset set everywhere. That matters for one element and one only: a step
// counted from `StepOrigin::ClippedStart` re-phases where the window is clipped
// at a low face of the volume, so a filter that read `offsets` there would
// compute the anchored window under a name that says otherwise. For every other
// element `offsets_at` hands back the element's own slice, so the loop is the
// loop it was and the answer is byte-identical.
//
// This is not a courtesy to the element type. `ops::rank`'s extreme ranks *are*
// this file's two primitives over the same element — the test below pins that
// equality, and `ops::background` builds a grey opening out of the rank filter
// on the strength of it — so an origin honoured on one side of that equality and
// not the other would be two filters wearing one name.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp};
use crate::reach::Reach;
use crate::voxels::Voxels;

use super::element::{StepOrigin, StructuringElement};
use super::shapes_agree;
use super::voxelwise::{from_set, is_set};

/// The conjunction of the element around every voxel.
///
/// **`input` is read as the whole volume**, which is what a caller handing over
/// a bare array is saying; [`erode_into_at`] is the form that says where the
/// array sits in a larger one. The two differ for exactly one element — see the
/// module header — and are the same call for every other.
pub fn erode_into(
    input: ArrayView3<'_, bool>,
    element: &StructuringElement,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    let at = whole(input.shape());
    erode_into_at(input, &at, element, out)
}

/// [`erode_into`] with the buffer's place in its volume stated.
///
/// `at` decides where the element's low faces are, and therefore what a
/// re-phasing element reads; see the module header. It changes nothing for an
/// element whose offsets are one set, which is every element without a step.
pub fn erode_into_at(
    input: ArrayView3<'_, bool>,
    at: &Anchor,
    element: &StructuringElement,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    sweep(input, at, element, out, false, "erode_into")
}

/// The disjunction of the element around every voxel.
///
/// Reads `input` as the whole volume; [`dilate_into_at`] is the anchored form.
pub fn dilate_into(
    input: ArrayView3<'_, bool>,
    element: &StructuringElement,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    let at = whole(input.shape());
    dilate_into_at(input, &at, element, out)
}

/// [`dilate_into`] with the buffer's place in its volume stated; see
/// [`erode_into_at`].
pub fn dilate_into_at(
    input: ArrayView3<'_, bool>,
    at: &Anchor,
    element: &StructuringElement,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    sweep(input, at, element, out, true, "dilate_into")
}

/// An erosion followed by a dilation. **Reaches twice the element's radius.**
pub fn open_into(
    input: ArrayView3<'_, bool>,
    element: &StructuringElement,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    let at = whole(input.shape());
    open_into_at(input, &at, element, out)
}

/// [`open_into`] with the buffer's place in its volume stated.
///
/// **The same anchor for both passes**, and it has to be: the intermediate
/// covers the same buffer at the same place, so the second pass's low faces are
/// the first pass's and the composition is the composition of the two filters
/// the volume names.
pub fn open_into_at(
    input: ArrayView3<'_, bool>,
    at: &Anchor,
    element: &StructuringElement,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    let mut between = Array3::from_elem(input.raw_dim(), false);
    erode_into_at(input, at, element, between.view_mut())?;
    dilate_into_at(between.view(), at, element, out)
}

/// A dilation followed by an erosion. **Reaches twice the element's radius.**
pub fn close_into(
    input: ArrayView3<'_, bool>,
    element: &StructuringElement,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    let at = whole(input.shape());
    close_into_at(input, &at, element, out)
}

/// [`close_into`] with the buffer's place in its volume stated; see
/// [`open_into_at`].
pub fn close_into_at(
    input: ArrayView3<'_, bool>,
    at: &Anchor,
    element: &StructuringElement,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    let mut between = Array3::from_elem(input.raw_dim(), false);
    dilate_into_at(input, at, element, between.view_mut())?;
    erode_into_at(between.view(), at, element, out)
}

/// The anchor a caller who handed over a bare array is stating: this array is
/// the volume. One function rather than four call sites, so that the reading
/// every anchor-free entry point here takes is one statement.
fn whole(shape: &[usize]) -> Anchor {
    Anchor::whole([shape[0], shape[1], shape[2]])
}

/// The one loop both primitives use.
///
/// `hit` is the value that decides the answer as soon as it is seen: `true` for
/// a dilation (any set voxel sets the output), `false` for an erosion (any clear
/// voxel clears it). Everything else about the two is identical, including the
/// clamp, so writing them as one loop is not a saving of lines — it is a
/// guarantee that they clamp the same way.
fn sweep(
    input: ArrayView3<'_, bool>,
    at: &Anchor,
    element: &StructuringElement,
    mut out: ArrayViewMut3<'_, bool>,
    hit: bool,
    what: &str,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), what)?;
    if element.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "{what}: an empty element has nothing to reduce over"
        )));
    }
    let extent = [
        input.shape()[0] as isize,
        input.shape()[1] as isize,
        input.shape()[2] as isize,
    ];
    // Checked only where the anchor decides anything, which is where the element
    // re-phases; see the same argument, at greater length, in `ops::rank`. An
    // element that reads one offset set everywhere cannot tell a wrong anchor
    // from a right one, so demanding a right one would refuse calls that were
    // always correct.
    if element.origin() == StepOrigin::ClippedStart {
        for axis in 0..3 {
            if at.offset[axis] + extent[axis] as usize > at.volume[axis] {
                return Err(Error::InvalidArgument(format!(
                    "{what}: a buffer of {:?} at {:?} does not fit a volume of {:?}, and this \
                     element's step counts from the clipped start of the window, so where the \
                     buffer sits in the volume is part of the operation",
                    input.shape(),
                    at.offset,
                    at.volume
                )));
            }
        }
    }
    // The element's offsets at one voxel, for the one element that has more than
    // one set of them. Owned out here so that a voxel pays no allocation, and
    // untouched by every other element.
    let mut offsets: Vec<[isize; 3]> = Vec::new();
    // The one offset set, where there is one, lifted out of the loop — the same
    // slice `offsets_at` would hand back, and `ops::rank` gives the argument for
    // lifting it at greater length.
    let fixed = (element.origin() == StepOrigin::Anchor).then(|| element.offsets());
    for i in 0..input.shape()[0] {
        for j in 0..input.shape()[1] {
            for k in 0..input.shape()[2] {
                let centre = [i as isize, j as isize, k as isize];
                let gathered = match fixed {
                    Some(offsets) => offsets,
                    // The same voxel in the volume's coordinates: the element is
                    // asked there, the array is read here.
                    None => {
                        let placed = [
                            centre[0] + at.offset[0] as isize,
                            centre[1] + at.offset[1] as isize,
                            centre[2] + at.offset[2] as isize,
                        ];
                        element.offsets_at(placed, at.volume, &mut offsets)
                    }
                };
                let mut answer = !hit;
                for offset in gathered {
                    let a = centre[0] + offset[0];
                    let b = centre[1] + offset[1];
                    let c = centre[2] + offset[2];
                    if a < 0 || b < 0 || c < 0 || a >= extent[0] || b >= extent[1] || c >= extent[2]
                    {
                        continue;
                    }
                    if input[[a as usize, b as usize, c as usize]] == hit {
                        answer = hit;
                        break;
                    }
                }
                out[[i, j, k]] = answer;
            }
        }
    }
    Ok(())
}

/// Which of the four.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Morphology {
    Erode,
    Dilate,
    Open,
    Close,
}

impl Morphology {
    /// How many times the element's radius this operation reads.
    ///
    /// **Derived from the composition, not declared beside it.** A composition
    /// of two passes reads two radii; that is the whole of the derivation, and
    /// it is here rather than in the op so that a caller reasoning about the
    /// kernels gets the same number as the planner does.
    pub fn reach_factor(self) -> usize {
        match self {
            Morphology::Erode | Morphology::Dilate => 1,
            Morphology::Open | Morphology::Close => 2,
        }
    }

    pub fn apply_into(
        self,
        input: ArrayView3<'_, bool>,
        element: &StructuringElement,
        out: ArrayViewMut3<'_, bool>,
    ) -> Result<()> {
        let at = whole(input.shape());
        self.apply_into_at(input, &at, element, out)
    }

    /// [`Self::apply_into`] with the buffer's place in its volume stated; see
    /// [`erode_into_at`].
    pub fn apply_into_at(
        self,
        input: ArrayView3<'_, bool>,
        at: &Anchor,
        element: &StructuringElement,
        out: ArrayViewMut3<'_, bool>,
    ) -> Result<()> {
        match self {
            Morphology::Erode => erode_into_at(input, at, element, out),
            Morphology::Dilate => dilate_into_at(input, at, element, out),
            Morphology::Open => open_into_at(input, at, element, out),
            Morphology::Close => close_into_at(input, at, element, out),
        }
    }
}

/// Erode, dilate, open or close a mask over a parameterised element.
pub struct MorphologyOp {
    name: &'static str,
    kind: Morphology,
    element: StructuringElement,
    cost: f64,
}

impl MorphologyOp {
    pub fn new(name: &'static str, kind: Morphology, element: StructuringElement) -> Self {
        let cost = cost_for(kind, &element);
        Self {
            name,
            kind,
            element,
            cost,
        }
    }

    pub fn element(&self) -> &StructuringElement {
        &self.element
    }

    pub fn kind(&self) -> Morphology {
        self.kind
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for MorphologyOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The element's wider side times the number of passes. Nothing configures
    /// it, and an opening reports twice what its element does.
    ///
    /// The **bound**; [`Self::reach_spec`] is the exact statement, and for an
    /// element with a centre voxel the two are the same number.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.element.reach(axis) * self.kind.reach_factor()
    }

    /// What the composition actually reads, per side.
    ///
    /// **This is the case the doubled reach gets subtly wrong when it is
    /// assumed symmetric.** `sweep` applies the element as written for both
    /// erosion and dilation — it does not reflect it between the two passes —
    /// so composing them gives offsets `o + o'` with both drawn from the same
    /// element, whose extremes are twice each side: an element reading five
    /// below and four above makes an opening that reads ten below and eight
    /// above. A symmetric assumption would either fetch ten on both sides,
    /// which wastes two planes per block on the narrow side, or — the failure
    /// worth naming — take one side and apply it to the other, which
    /// under-fetches by two on the wide side and produces a plausible, wrong
    /// volume at every seam.
    ///
    /// If `sweep`'s dilation is ever reflected, this becomes `lo + hi` on both
    /// sides and [`StructuringElement::reach_spec_after`] stops being the right
    /// method to call; that method says so at its own definition, because the
    /// two are one derivation and it must not be possible to change one of them
    /// alone.
    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        self.element.reach_spec_after(self.kind.reach_factor())
    }

    /// A mask, held as a mask or held as `f64`.
    ///
    /// **`Bool` is the point of this pass.** A binary volume is what this op is
    /// for and what the storage it comes from holds; carrying it as `f64` cost
    /// eight bytes a voxel to represent one bit, plus a widen on the way in and
    /// a narrow on the way out at every block. Both are now gone for the
    /// `Bool` arm. `F64` stays because a chain may hold a mask as `f64` under
    /// this module's `is_set`/`from_set` convention, and dropping it would break
    /// every such chain for no gain.
    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(dtype, Dtype::Bool | Dtype::F64)
    }

    /// **`at` is read rather than ignored**, so that an element whose step counts
    /// from the clipped start re-phases at the volume's faces and not at a block
    /// seam. Every other element reads the same offsets everywhere and cannot
    /// tell the difference.
    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        match input.dtype() {
            // No conversion and no intermediate: the kernel is a `bool` kernel
            // and the buffer is a `bool` buffer.
            Dtype::Bool => self.kind.apply_into_at(
                input.view::<bool>()?,
                at,
                &self.element,
                out.view_mut::<bool>()?,
            ),
            _ => {
                let mask = input.view::<f64>()?.mapv(is_set);
                let mut result = Array3::from_elem(mask.raw_dim(), false);
                self.kind
                    .apply_into_at(mask.view(), at, &self.element, result.view_mut())?;
                let mut out = out.view_mut::<f64>()?;
                ndarray::Zip::from(&mut out)
                    .and(&result)
                    .for_each(|slot, &value| *slot = from_set(value));
                Ok(())
            }
        }
    }

    /// Exactly the constant, as a mask, for all four operations.
    ///
    /// An erosion of an all-set neighbourhood is set and of an all-clear one is
    /// clear; a dilation likewise; and a composition of two such passes is the
    /// same constant again. The output is `0.0` or `1.0`, so a non-zero constant
    /// maps to `1.0` — that is the mask convention applied consistently, not a
    /// loss of information, since the op writes a mask whatever it was given.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(from_set(is_set(value)))
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Measured; see `super::COST_MEASUREMENT`.
pub(super) fn cost_for(kind: Morphology, element: &StructuringElement) -> f64 {
    MORPHOLOGY_COST_PER_ELEMENT_VOXEL * element.len() as f64 * kind.reach_factor() as f64
}

/// Measured; see `super::COST_MEASUREMENT`.
pub(super) const MORPHOLOGY_COST_PER_ELEMENT_VOXEL: f64 = 1.38;

#[cfg(test)]
mod tests {
    use super::super::element::{ElementShape, Rank};
    use super::super::rank::rank_filter_into;
    use super::*;

    fn speckle(shape: (usize, usize, usize)) -> Array3<bool> {
        Array3::from_shape_fn(shape, |(i, j, k)| (i * 31 + j * 17 + k * 7) % 5 < 2)
    }

    /// Erosion and dilation *are* the extreme ranks over the same element. The
    /// morphology loop short-circuits and the rank filter selects, so they are
    /// different code; they must not be different answers.
    #[test]
    fn erosion_and_dilation_are_the_extreme_ranks_of_the_same_element() {
        let input = speckle((7, 6, 5));
        for shape in [ElementShape::Box, ElementShape::Ellipsoid] {
            for radius in [[1, 1, 1], [2, 0, 1]] {
                let element = StructuringElement::from_radius(shape, radius);
                let mut eroded = Array3::from_elem(input.raw_dim(), false);
                erode_into(input.view(), &element, eroded.view_mut()).unwrap();
                let mut lowest = Array3::from_elem(input.raw_dim(), false);
                rank_filter_into(input.view(), &element, Rank::lowest(), lowest.view_mut())
                    .unwrap();
                assert_eq!(eroded, lowest, "erode {shape:?} {radius:?}");

                let mut dilated = Array3::from_elem(input.raw_dim(), false);
                dilate_into(input.view(), &element, dilated.view_mut()).unwrap();
                let mut highest = Array3::from_elem(input.raw_dim(), false);
                rank_filter_into(
                    input.view(),
                    &element,
                    Rank::highest(&element),
                    highest.view_mut(),
                )
                .unwrap();
                assert_eq!(dilated, highest, "dilate {shape:?} {radius:?}");
            }
        }
    }

    /// The same equality, over the one element whose window depends on where it
    /// is evaluated — and at a buffer that is not the whole volume, which is the
    /// only place the two could disagree.
    ///
    /// **This is the assertion that fails if either side stops honouring the
    /// step's origin.** `ops::rank` gathers `offsets_at` in its own loop and this
    /// file gathers it in another; they must gather the same set at the same
    /// voxel or the crate has two filters under one name, and `ops::background`
    /// builds a grey opening out of the rank filter on the strength of exactly
    /// this equality.
    #[test]
    fn the_extreme_ranks_agree_over_a_re_phasing_element_too() {
        use super::super::element::StepOrigin;
        use super::super::rank::rank_filter_into_at;

        // A comb along axis 0: set on the odd coordinates and clear on the even
        // ones. A decimation by two either lands on the teeth or between them, so
        // the two origins are as far apart as a `bool` volume can put them —
        // which is what makes the last assertion below say something.
        let input = Array3::from_shape_fn((7, 6, 1), |(i, j, _)| i % 2 == 1 || j == 4);
        let size = [9, 3, 1];
        let step = [2, 1, 1];
        let clipped = StructuringElement::from_size_stepped_at(
            ElementShape::Box,
            size,
            step,
            StepOrigin::ClippedStart,
        )
        .unwrap();
        let anchored = StructuringElement::from_size_stepped_at(
            ElementShape::Box,
            size,
            step,
            StepOrigin::Anchor,
        )
        .unwrap();

        let mut differences = 0usize;
        // the whole volume, and a buffer holding the far end of a longer one —
        // the second is where a rule keyed on the buffer's edge would show
        for at in [Anchor::whole([7, 6, 1]), Anchor::new([3, 0, 0], [10, 6, 1])] {
            let mut eroded = Array3::from_elem(input.raw_dim(), false);
            erode_into_at(input.view(), &at, &clipped, eroded.view_mut()).unwrap();
            let mut lowest = Array3::from_elem(input.raw_dim(), false);
            rank_filter_into_at(
                input.view(),
                &at,
                &clipped,
                Rank::lowest(),
                lowest.view_mut(),
            )
            .unwrap();
            assert_eq!(eroded, lowest, "erode at {at:?}");

            let mut dilated = Array3::from_elem(input.raw_dim(), false);
            dilate_into_at(input.view(), &at, &clipped, dilated.view_mut()).unwrap();
            let mut highest = Array3::from_elem(input.raw_dim(), false);
            rank_filter_into_at(
                input.view(),
                &at,
                &clipped,
                Rank::highest(&clipped),
                highest.view_mut(),
            )
            .unwrap();
            assert_eq!(dilated, highest, "dilate at {at:?}");

            // and the origin is really reaching the sweep: the other origin over
            // the same box is a different erosion here
            let mut other = Array3::from_elem(input.raw_dim(), false);
            erode_into_at(input.view(), &at, &anchored, other.view_mut()).unwrap();
            differences += eroded
                .iter()
                .zip(other.iter())
                .filter(|(left, right)| left != right)
                .count();
        }
        assert!(
            differences > 0,
            "the two origins must be two erosions here, or the equalities above are an \
             equality of the anchored gather with itself"
        );
    }

    /// **The anchored sweep did not move**, which every element in this crate's
    /// history is and which an unstepped element normalises to: an element that
    /// reads one offset set everywhere cannot tell where its buffer sits, so
    /// stating an anchor changes nothing at all.
    #[test]
    fn the_anchored_sweep_is_byte_unchanged() {
        let input = speckle((7, 6, 5));
        for element in [
            StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]),
            StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 1, 0]),
            StructuringElement::from_size(ElementShape::Box, [4, 3, 2]).unwrap(),
        ] {
            for kind in [
                Morphology::Erode,
                Morphology::Dilate,
                Morphology::Open,
                Morphology::Close,
            ] {
                let mut plain = Array3::from_elem(input.raw_dim(), false);
                kind.apply_into(input.view(), &element, plain.view_mut())
                    .unwrap();
                let mut placed = Array3::from_elem(input.raw_dim(), false);
                kind.apply_into_at(
                    input.view(),
                    &Anchor::new([40, 30, 20], [100, 90, 80]),
                    &element,
                    placed.view_mut(),
                )
                .unwrap();
                assert_eq!(plain, placed, "{kind:?}");
            }
        }
    }

    /// The clamp behaves as the identity of the operation, which is what keeps a
    /// volume from eroding away at its own faces.
    #[test]
    fn the_boundary_clamp_is_the_identity_of_the_operation() {
        let input = Array3::from_elem((3, 3, 3), true);
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let mut eroded = Array3::from_elem(input.raw_dim(), false);
        erode_into(input.view(), &element, eroded.view_mut()).unwrap();
        assert!(
            eroded.iter().all(|&value| value),
            "an all-set volume must survive its own boundary"
        );

        let input = Array3::from_elem((3, 3, 3), false);
        let mut dilated = Array3::from_elem(input.raw_dim(), true);
        dilate_into(input.view(), &element, dilated.view_mut()).unwrap();
        assert!(dilated.iter().all(|&value| !value));
    }

    #[test]
    fn opening_removes_an_isolated_voxel_and_closing_fills_an_isolated_hole() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);

        let mut speck = Array3::from_elem((7, 7, 7), false);
        speck[[3, 3, 3]] = true;
        let mut opened = Array3::from_elem(speck.raw_dim(), true);
        open_into(speck.view(), &element, opened.view_mut()).unwrap();
        assert!(opened.iter().all(|&value| !value));

        let mut hole = Array3::from_elem((7, 7, 7), true);
        hole[[3, 3, 3]] = false;
        let mut closed = Array3::from_elem(hole.raw_dim(), false);
        close_into(hole.view(), &element, closed.view_mut()).unwrap();
        assert!(closed.iter().all(|&value| value));
    }

    /// The composition, and the reach that follows from it.
    /// The `bool` arm and the `f64` arm are the same operation, and the `bool`
    /// arm holds an eighth of the bytes. Both halves asserted, because the
    /// second is the reason for the pass and the first is what makes it safe.
    #[test]
    fn a_mask_held_as_bool_and_as_f64_gives_the_same_answer_in_an_eighth_of_the_space() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let op = MorphologyOp::new("open", Morphology::Open, element);
        let mask = speckle((7, 6, 5));

        let flags: Voxels = mask.clone().into();
        let mut narrow = Voxels::zeros(Dtype::Bool, [7, 6, 5]).unwrap();
        op.apply(&flags, &mut narrow, &Anchor::whole([7, 6, 5]))
            .unwrap();

        let wide_in: Voxels = mask.mapv(from_set).into();
        let mut wide = Voxels::zeros(Dtype::F64, [7, 6, 5]).unwrap();
        op.apply(&wide_in, &mut wide, &Anchor::whole([7, 6, 5]))
            .unwrap();

        let narrow_view = narrow.view::<bool>().unwrap();
        let wide_view = wide.view::<f64>().unwrap();
        for (flag, value) in narrow_view.iter().zip(wide_view.iter()) {
            assert_eq!(from_set(*flag), *value);
        }
        assert_eq!(wide.bytes(), narrow.bytes() * 8);
    }

    #[test]
    fn an_opening_reaches_twice_what_its_element_does() {
        let element = StructuringElement::from_size(ElementShape::Box, [7, 7, 7]).unwrap();
        let erode = MorphologyOp::new("erode", Morphology::Erode, element.clone());
        let open = MorphologyOp::new("open", Morphology::Open, element);
        assert_eq!(erode.reach(0, 1000), 3);
        assert_eq!(open.reach(0, 1000), 6);
    }

    #[test]
    fn opening_is_the_composition_and_not_a_second_implementation() {
        let input = speckle((9, 8, 7));
        let element = StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 2, 2]);
        let mut composed = Array3::from_elem(input.raw_dim(), false);
        {
            let mut between = Array3::from_elem(input.raw_dim(), false);
            erode_into(input.view(), &element, between.view_mut()).unwrap();
            dilate_into(between.view(), &element, composed.view_mut()).unwrap();
        }
        let mut opened = Array3::from_elem(input.raw_dim(), false);
        open_into(input.view(), &element, opened.view_mut()).unwrap();
        assert_eq!(opened, composed);
    }

    #[test]
    fn every_operation_declares_the_constant_it_produces() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        for kind in [
            Morphology::Erode,
            Morphology::Dilate,
            Morphology::Open,
            Morphology::Close,
        ] {
            let op = MorphologyOp::new("m", kind, element.clone());
            assert_eq!(op.constant_maps_to(0.0), Some(0.0), "{kind:?}");
            assert_eq!(op.constant_maps_to(1.0), Some(1.0), "{kind:?}");
            assert_eq!(op.constant_maps_to(9.0), Some(1.0), "{kind:?}");

            // and computing it agrees with declaring it
            for (constant, want) in [(0.0, 0.0), (9.0, 1.0)] {
                let input: Voxels = Array3::from_elem((5, 5, 5), constant).into();
                let mut out = Voxels::zeros(Dtype::F64, [5, 5, 5]).unwrap();
                op.apply(&input, &mut out, &Anchor::whole([5, 5, 5]))
                    .unwrap();
                assert!(
                    out.view::<f64>()
                        .unwrap()
                        .iter()
                        .all(|&value| value == want),
                    "{kind:?} on {constant}"
                );
            }
        }
    }
}
