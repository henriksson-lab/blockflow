// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Binary morphology over a parameterised structuring element: erosion,
// dilation, and the two compositions of them.
//
// Two dilations, and which one an opening is built from
// -----------------------------------------------------
// There are **two** dilations here and the difference between them is the
// difference between an opening and a translation of the image.
//
// * `dilate_into` gathers: `out[c] = OR over o of in[c + o]`. That is the
//   disjunction over the neighbourhood the element names, it is the extreme
//   rank of `ops::rank` over the same element — an equality this file's own
//   test pins and `ops::background` builds a grey opening on — and it is a
//   dilation by the **reflected** element, `X + B̌`;
// * `dilate_placed_into` scatters: the element is placed at every set voxel and
//   the union taken. That is `X + B`, and it is the only dilation that is the
//   erosion's **adjoint**.
//
// `open = dilate_placed(erode(x))`, `close = erode(dilate_placed(x))`, and
// neither has a loop of its own. Three consequences, and all three are the
// point:
//
// * there is no second erosion to drift from the first;
// * an adjunction makes the compositions an opening and a closing — anti-
//   extensive and idempotent, extensive and idempotent — for **every** element,
//   with nothing assumed about its symmetry. Composing the gather with the
//   gather instead gives `(X ⊖ B) ⊕ B̌`, which for an element without a centre
//   voxel obeys none of those laws and translates the image once per
//   application. That is what this file did until `dilate_placed_into` existed;
//   `tests/morphology_laws.rs` measures the laws now rather than assuming them;
// * the reach falls out of the composition rather than being asserted, and
//   reflecting changes it: `lo + hi` on **both** sides, not twice each side. The
//   two agree for a centred element, which is why nothing here noticed until an
//   even extent was built. A hand-written opening is exactly the kind of op
//   whose author writes `radius` in the halo formula and is wrong with no
//   symptom except at block seams — which is the failure
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
// offset set everywhere; `dilate_placed_into` asks it at each **source**
// voxel's position, which is the same rule read at the other end of the scatter
// and is what keeps the adjunction for a re-phasing element. That matters for one element and one only: a step
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
use crate::op::{Anchor, BlockOp, Slicing};
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

/// The dilation **by** the element — `X ⊕ B` — written as the element placed
/// at every set voxel.
///
/// **The second dilation in this file, and the difference between the two is
/// what makes an opening an opening.** [`dilate_into`] reads
/// `input[centre + offset]` and takes the disjunction: that is the extreme rank
/// of the same neighbourhood — the equality `ops::rank` is held to below — and
/// it is a dilation by the *reflected* element, `X ⊕ B̌`. This one takes the
/// union of the element placed at each set voxel, which is the textbook `X ⊕ B`
/// and the only dilation that is [`erode_into`]'s adjoint. The two are the same
/// filter exactly when `B = B̌`, which every centred element is and no even
/// extent is.
///
/// **The adjunction is the reason this exists**, not the textbook. `δ(Y) ⊆ Z`
/// iff `Y ⊆ ε(Z)`: both sides say "for every set `p` and every `o` the element
/// holds at `p`, `p + o` is in `Z`", offset for offset. An adjunction makes
/// `δ∘ε` an opening and `ε∘δ` a closing — anti-extensive, increasing and
/// idempotent, and extensive, increasing and idempotent — with no assumption
/// about the element whatever. Composing [`erode_into`] with [`dilate_into`]
/// instead gives `(X ⊖ B) ⊕ B̌`, which is none of those things for an
/// asymmetric element: it **translates the image** by the element's own
/// asymmetry, once per application. That was this module's behaviour until this
/// function existed, and `tests/morphology_laws.rs` is where the three laws are
/// now measured rather than assumed.
///
/// **The element is read at the source voxel**, which is what keeps the
/// adjunction for an element whose window depends on where it is evaluated: a
/// step counted from [`StepOrigin::ClippedStart`] re-phases near a low face, so
/// `B` is a set per position and its reflection is not a window at all — see
/// [`StructuringElement::reflected`], which refuses exactly that element. A
/// scatter has no such difficulty because it never needs the reflection: it
/// asks the element at `p` and writes at `p + o`, which is the same rule
/// `ops::label` and `ops::voxelize` stamp under and the same argument
/// `tests/clipped_start_through_the_gathering_ops.rs` makes for them.
///
/// The clamp is the module's clamp: what falls outside the buffer is dropped,
/// so the boundary behaves as clear for a dilation. At a real volume face that
/// is the convention; at a block seam it is deliberately wrong, and the halo is
/// what makes it right.
pub fn dilate_placed_into(
    input: ArrayView3<'_, bool>,
    element: &StructuringElement,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    let at = whole(input.shape());
    dilate_placed_into_at(input, &at, element, out)
}

/// [`dilate_placed_into`] with the buffer's place in its volume stated; see
/// [`erode_into_at`].
pub fn dilate_placed_into_at(
    input: ArrayView3<'_, bool>,
    at: &Anchor,
    element: &StructuringElement,
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    let what = "dilate_placed_into";
    let extent = preflight(input.shape(), at, element, out.shape(), what)?;
    // A scatter accumulates, so the destination starts empty. A gather writes
    // every voxel it visits and needs no such statement — the one asymmetry
    // between the two loops, and it is here rather than left to the caller.
    out.fill(false);
    let mut offsets: Vec<[isize; 3]> = Vec::new();
    let fixed = (element.origin() == StepOrigin::Anchor).then(|| element.offsets());
    for i in 0..input.shape()[0] {
        for j in 0..input.shape()[1] {
            for k in 0..input.shape()[2] {
                if !input[[i, j, k]] {
                    continue;
                }
                let source = [i as isize, j as isize, k as isize];
                let placed = match fixed {
                    Some(offsets) => offsets,
                    // The element at **the source's** position in the volume,
                    // which is what makes this the erosion's adjoint; see the
                    // header of this function.
                    None => {
                        let at_source = [
                            source[0] + at.offset[0] as isize,
                            source[1] + at.offset[1] as isize,
                            source[2] + at.offset[2] as isize,
                        ];
                        element.offsets_at(at_source, at.volume, &mut offsets)
                    }
                };
                for offset in placed {
                    let a = source[0] + offset[0];
                    let b = source[1] + offset[1];
                    let c = source[2] + offset[2];
                    if a < 0 || b < 0 || c < 0 || a >= extent[0] || b >= extent[1] || c >= extent[2]
                    {
                        continue;
                    }
                    out[[a as usize, b as usize, c as usize]] = true;
                }
            }
        }
    }
    Ok(())
}

/// An erosion followed by the **dilation by the same element**, which is an
/// opening. **Reaches `lo + hi` on both sides of every axis.**
///
/// Not `2 * radius` per side, which is what a composition that did not reflect
/// would read and what this file reported before it reflected; the two are the
/// same number for a centred element. See
/// [`StructuringElement::reach_spec_reflected_pair`].
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
    dilate_placed_into_at(between.view(), at, element, out)
}

/// The **dilation by the element** followed by an erosion, which is a closing.
/// **Reaches `lo + hi` on both sides of every axis**, as an opening does.
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
    dilate_placed_into_at(input, at, element, between.view_mut())?;
    erode_into_at(between.view(), at, element, out)
}

/// The anchor a caller who handed over a bare array is stating: this array is
/// the volume. One function rather than four call sites, so that the reading
/// every anchor-free entry point here takes is one statement.
fn whole(shape: &[usize]) -> Anchor {
    Anchor::whole([shape[0], shape[1], shape[2]])
}

/// What both loops check before either of them runs, and the extent both of
/// them clamp against.
///
/// One function rather than two copies, for the reason the two loops are one
/// loop where they can be: a gather and a scatter that disagreed about which
/// calls are legal would be two operations wearing one module's name.
fn preflight(
    shape: &[usize],
    at: &Anchor,
    element: &StructuringElement,
    out_shape: &[usize],
    what: &str,
) -> Result<[isize; 3]> {
    shapes_agree(shape, out_shape, what)?;
    if element.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "{what}: an empty element has nothing to reduce over"
        )));
    }
    let extent = [shape[0] as isize, shape[1] as isize, shape[2] as isize];
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
                    shape, at.offset, at.volume
                )));
            }
        }
    }
    Ok(extent)
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
    let extent = preflight(input.shape(), at, element, out.shape(), what)?;
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
    /// How many passes over the volume this operation makes.
    ///
    /// **A pass count, and no longer a reach factor**, which is what it was
    /// called and used as while the composition did not reflect. It still
    /// prices the work — two passes cost twice one — but the reach does not
    /// follow from it: an opening reflects between its passes, so it reads
    /// `lo + hi` per side rather than twice each side, and the two numbers
    /// differ for every element without a centre voxel. What derives a reach
    /// asks [`StructuringElement::reach_spec_reflected_pair`]; what prices work
    /// asks this.
    pub fn passes(self) -> usize {
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
    /// **A stencil.** This is a *binary* morphology — an `f64` buffer goes
    /// through `is_set` on the way in — so an erosion is a conjunction and a
    /// dilation a disjunction over a fixed neighbourhood read at fixed offsets.
    /// Both are associative and commutative, so no order a cut could change is
    /// visible in the answer, and nothing is carried along the scan.
    ///
    /// **Not "the extreme of a neighbourhood", which is what this comment said
    /// first and would have been the grey morphology's reason.** The distinction
    /// is not pedantic: it is what decides how a fixture can perturb this op at
    /// all, and `tests/intra_block_slicing.rs` had to learn it — adding a large
    /// value to a voxel moves a minimum and moves nothing here, because a value
    /// made larger or smaller is set either way.
    ///
    /// Held to it by `tests/intra_block_slicing.rs`.
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn name(&self) -> &'static str {
        self.name
    }

    /// The element's wider side for a primitive; `lo + hi` for a composition.
    /// Nothing configures it.
    ///
    /// The **bound**; [`Self::reach_spec`] is the exact statement. For a
    /// primitive the two differ where the element has an even extent; for a
    /// composition they never do, because reflecting makes the pair symmetric.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        match self.kind {
            Morphology::Erode | Morphology::Dilate => self.element.reach(axis),
            Morphology::Open | Morphology::Close => self.element.reach_reflected_pair(axis),
        }
    }

    /// What the composition actually reads, per side.
    ///
    /// **The one place the reflection shows up in the geometry.** `sweep`
    /// applies the element as written, so a primitive reads exactly the
    /// element's own two sides — an element reading five below and four above
    /// makes an erosion that reads `(5, 4)`, and a halo that took `max(lo, hi)`
    /// and applied it symmetrically would over-fetch on the narrow side or, the
    /// failure worth naming, under-fetch on the wide one.
    ///
    /// A **composition reflects**: [`open_into`] dilates by placing the element
    /// rather than by gathering it, so the composed offset is `o - o'` with both
    /// drawn from the element and the pair is `(lo + hi, lo + hi)` — symmetric,
    /// and *narrower* than the `(2 * lo, 2 * hi)` this reported while its
    /// dilation was unreflected. That is not a saving with the same answer: the
    /// old number belonged to an operation that translated the image, and this
    /// one belongs to an opening.
    ///
    /// [`StructuringElement::reach_spec_after`] is what a repetition **without**
    /// a reflection needs and is deliberately not called here; that method says
    /// so at its own definition, because the two are one derivation and it must
    /// not be possible to change one of them alone.
    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        match self.kind {
            Morphology::Erode | Morphology::Dilate => self.element.reach_spec(),
            Morphology::Open | Morphology::Close => self.element.reach_spec_reflected_pair(),
        }
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
    MORPHOLOGY_COST_PER_ELEMENT_VOXEL * element.len() as f64 * kind.passes() as f64
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

    /// An opening reads `lo + hi` per side, which is twice the radius of a
    /// centred element and **is not** twice each side of one that is not.
    ///
    /// Both halves matter. The first is the number every symmetric element in
    /// this crate's history produced, so a change that broke it would be
    /// breaking the common case; the second is the number that changed when the
    /// dilation started reflecting, and asserting it here is what stops
    /// `reach_spec_after` from being quietly reinstated.
    #[test]
    fn an_opening_reaches_the_sum_of_the_two_sides_and_not_twice_each() {
        let element = StructuringElement::from_size(ElementShape::Box, [7, 7, 7]).unwrap();
        let erode = MorphologyOp::new("erode", Morphology::Erode, element.clone());
        let open = MorphologyOp::new("open", Morphology::Open, element);
        assert_eq!(erode.reach(0, 1000), 3);
        assert_eq!(open.reach(0, 1000), 6);
        assert_eq!(open.reach_spec([1000; 3]).at(0, 0, 1000), (6, 6));

        // and an element with no centre voxel, where the two rules part
        let element = StructuringElement::from_size(ElementShape::Box, [10, 5, 4]).unwrap();
        assert_eq!(element.sides(0), (5, 4));
        let erode = MorphologyOp::new("erode", Morphology::Erode, element.clone());
        let open = MorphologyOp::new("open", Morphology::Open, element);
        assert_eq!(erode.reach_spec([1000; 3]).at(0, 0, 1000), (5, 4));
        assert_eq!(
            open.reach_spec([1000; 3]).at(0, 0, 1000),
            (9, 9),
            "the reflected composition is symmetric at lo + hi, not (10, 8)"
        );
        assert_eq!(open.reach(0, 1000), 9);
    }

    /// The opening is the composition of the two published primitives — the
    /// erosion and **the placed dilation** — and not a third loop.
    ///
    /// Over an element with no centre voxel, which is the case that tells the
    /// two dilations apart: composing with the *gather* instead gives a
    /// different volume, and the test says so rather than leaving the reader to
    /// trust that the right one was called.
    #[test]
    fn opening_is_the_composition_and_not_a_second_implementation() {
        let input = speckle((9, 8, 7));
        for element in [
            StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 2, 2]),
            StructuringElement::from_size(ElementShape::Box, [4, 3, 2]).unwrap(),
        ] {
            let mut between = Array3::from_elem(input.raw_dim(), false);
            erode_into(input.view(), &element, between.view_mut()).unwrap();

            let mut composed = Array3::from_elem(input.raw_dim(), false);
            dilate_placed_into(between.view(), &element, composed.view_mut()).unwrap();
            let mut opened = Array3::from_elem(input.raw_dim(), false);
            open_into(input.view(), &element, opened.view_mut()).unwrap();
            assert_eq!(opened, composed);

            // and the two dilations are one filter exactly when the element is
            // its own reflection. Over a **single interior voxel**, because that
            // is the volume on which the two are furthest apart — each is the
            // element itself, one reflected — where the speckle above is dense
            // enough that either dilation fills it and the comparison would be
            // one of two full volumes.
            let mut speck = Array3::from_elem(input.raw_dim(), false);
            speck[[4, 4, 3]] = true;
            let mut placed = Array3::from_elem(input.raw_dim(), false);
            dilate_placed_into(speck.view(), &element, placed.view_mut()).unwrap();
            let mut gathered = Array3::from_elem(input.raw_dim(), false);
            dilate_into(speck.view(), &element, gathered.view_mut()).unwrap();
            assert_eq!(
                gathered == placed,
                element.is_symmetric(),
                "the two dilations agree exactly when the element is its own reflection"
            );
        }
    }

    /// **The placed dilation is the gather over the reflected element**, which
    /// is what makes the scatter a dilation rather than a second convention.
    ///
    /// The scatter is written as a scatter because that is the form that
    /// survives an element whose window depends on where it is evaluated — see
    /// the test below. For every element that *has* a reflection the two forms
    /// must be one filter, and this is where that is pinned; without it,
    /// `StructuringElement::reflected` and `dilate_placed_into` could drift into
    /// two different dilations with nothing to notice.
    #[test]
    fn the_placed_dilation_is_the_gather_over_the_reflected_element() {
        let input = speckle((9, 8, 7));
        for element in [
            StructuringElement::from_radius(ElementShape::Box, [1, 2, 1]),
            StructuringElement::from_size(ElementShape::Box, [4, 3, 2]).unwrap(),
            StructuringElement::from_sides(ElementShape::Box, [3, 0, 1], [1, 2, 1]),
            StructuringElement::from_offsets([[0, 0, 0], [2, 1, 0], [-1, 0, 1]]).unwrap(),
        ] {
            let reflected = element.reflected().expect("an anchored element reflects");
            assert_eq!(reflected.sides(0), (element.sides(0).1, element.sides(0).0));
            assert_eq!(
                reflected.reflected().unwrap().offsets(),
                element.offsets(),
                "reflecting twice is the element again"
            );

            let mut placed = Array3::from_elem(input.raw_dim(), false);
            dilate_placed_into(input.view(), &element, placed.view_mut()).unwrap();
            let mut gathered = Array3::from_elem(input.raw_dim(), false);
            dilate_into(input.view(), &reflected, gathered.view_mut()).unwrap();
            assert_eq!(placed, gathered);
        }
    }

    /// **The adjunction**, over the one element that has no reflection at all.
    ///
    /// `dilate(Y) ⊆ Z` if and only if `Y ⊆ erode(Z)`, checked over every pair of
    /// masks a small volume can be given here rather than over the two the
    /// author would have picked. This is the property the opening laws rest on —
    /// an adjunction gives an opening and a closing with nothing assumed about
    /// the element — so it is the property to hold a re-phasing element to,
    /// where `StructuringElement::reflected` refuses and the equality above has
    /// nothing to say.
    #[test]
    fn the_placed_dilation_is_the_erosions_adjoint_over_a_re_phasing_element() {
        let element = StructuringElement::from_size_stepped_at(
            ElementShape::Box,
            [5, 3, 1],
            [2, 2, 1],
            StepOrigin::ClippedStart,
        )
        .unwrap();
        assert!(element.reflected().is_err(), "and it has no reflection");
        let shape = (6, 4, 3);
        let at = Anchor::whole([shape.0, shape.1, shape.2]);

        // Two families at graded densities, so that the pairs straddle the
        // equivalence instead of sitting on one side of it: a dense `y` dilates
        // to cover everything and would make every pair fail, which would be a
        // test of nothing. The sparse end of one family and the dense end of the
        // other are what put pairs on both sides.
        let sparse = |density: usize| -> Array3<bool> {
            Array3::from_shape_fn(shape, |(i, j, k)| (i * 7 + j * 13 + k * 29) % 16 < density)
        };
        let dense = |density: usize| -> Array3<bool> {
            Array3::from_shape_fn(shape, |(i, j, k)| (i * 5 + j * 3 + k * 7) % 16 < density)
        };
        let mut both_ways = 0;
        for lower in 0..8 {
            for upper in 0..8 {
                let y = sparse(lower);
                let z = dense(2 * upper + 2);
                let mut dilated = Array3::from_elem(y.raw_dim(), false);
                dilate_placed_into_at(y.view(), &at, &element, dilated.view_mut()).unwrap();
                let mut eroded = Array3::from_elem(z.raw_dim(), false);
                erode_into_at(z.view(), &at, &element, eroded.view_mut()).unwrap();

                let contained = |a: &Array3<bool>, b: &Array3<bool>| {
                    a.iter().zip(b.iter()).all(|(&x, &y)| !x || y)
                };
                assert_eq!(
                    contained(&dilated, &z),
                    contained(&y, &eroded),
                    "the adjunction failed at ({lower}, {upper})"
                );
                both_ways += contained(&dilated, &z) as usize;
            }
        }
        assert!(
            both_ways > 0 && both_ways < 64,
            "the pairs must fall on both sides of the equivalence, or it is vacuous; \
             {both_ways} of 64 held"
        );
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
