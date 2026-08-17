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
// The centre, where a population is given, is a third thing again
// -----------------------------------------------------------------
// The masked filter reads the population at every offset of the element, and one
// of those offsets — usually — is the centre itself. So there are two ways a
// voxel can end up with no window to select from, and they are not the same
// question:
//
// * **the gathered window came out empty**, because every offset that was in
//   bounds was out of the population;
// * **the centre's own bit is clear**, whatever the neighbours did.
//
// For an element that contains its own centre these coincide in one direction —
// a centre in the population is a member of its own window — and for one that
// does not they come apart entirely, which this crate's elements are allowed to
// be. `ExcludedCentre` is the parameter for the second, and it defaults to the
// behaviour that was there before it existed. The first stays what it was: the
// centre's own value, because it is the only value there is to write.
//
// The window is the element's offsets, so an element that is **stepped** —
// `StructuringElement::from_size_stepped` — simply gathers fewer of them, and
// every number this file derives from `element.len()` follows without a second
// path: the cost below, and the rank a `median` constructor names.
//
// **And it is `offsets_at`, not `offsets`**, which is this file's answer to the
// one element whose members are not a single set. A step counted from
// `StepOrigin::ClippedStart` re-phases where the window is clipped at a low face
// of the **volume**, so what it reads is a function of the anchor and of the
// volume's extent — both decomposition invariants, neither of them a property of
// the buffer in hand. `selecting` therefore asks the element what it reads at
// each voxel instead of assuming one answer for all of them, and asks it at the
// voxel's position *in the volume*: that is what the `Anchor` every `BlockOp`
// is handed is carried this far down for, and it is why a block seam — which is
// not a face — cannot re-phase anything. `rank_filter_into` and its siblings
// keep their signatures and read the array they are given as the whole volume,
// which is what a caller who hands over a bare array is saying.
//
// **What honouring it costs, and what it does not.** `offsets_at` hands back the
// element's own slice, with no copy and no allocation, for every element whose
// origin is `StepOrigin::Anchor` — which is every unstepped element, since a
// stride of one makes the two rules the same rule. `selecting` lifts that slice
// out of the voxel loop where there is one, so the anchored path is the field
// load and the slice walk it always was, and it is byte-identical to what it was:
// `the_anchored_window_is_byte_unchanged` is that measurement rather than that
// claim, and `the_cost_of_asking_the_element_per_voxel` is the other half of it.
//
// **Two counts stop being one count**, and both are safe. `element.len()` is the
// interior population, and a re-phased window can hold a different number —
// fewer where a face truncates it, and for a shaped element occasionally more,
// since a phase that lands nearer the centre of a ball keeps a wider
// cross-section of it. `Rank::resolve(full, available)` is already written for
// `available != full`, that being the truncation rule, and it never returns an
// index past `available - 1` at either sign; the window buffer is sized from
// `full` as a hint and grows once if a phase exceeds it.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp};
use crate::reach::Reach;
use crate::voxels::{VoxelElement, Voxels};

use super::element::{select_nth, Rank, StepOrigin, StructuringElement, Total};
use super::shapes_agree;

/// Select `rank` of `element` around every voxel of `input`.
///
/// Generic over `Ord`, which is exactly the requirement the algorithm has: it
/// compares values and hands one of them back, never combining two. The value
/// written is a value that was read, bit for bit.
///
/// `out` may not alias `input` — the filter is not in-place, because a rank read
/// from a partially overwritten array is a different filter.
///
/// **`input` is read as the whole volume**, which is what a caller handing over a
/// bare array is saying. That matters for exactly one element — one whose step
/// counts from [`StepOrigin::ClippedStart`](super::StepOrigin::ClippedStart),
/// whose window re-phases at a low face — and [`rank_filter_into_at`] is the form
/// that says where the array sits in a larger volume. For every other element the
/// two are the same call.
pub fn rank_filter_into<T: Ord + Copy>(
    input: ArrayView3<'_, T>,
    element: &StructuringElement,
    rank: Rank,
    out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    let at = whole(input.shape());
    rank_filter_into_at(input, &at, element, rank, out)
}

/// [`rank_filter_into`] with the buffer's place in its volume stated.
///
/// `at` is not decoration, and it is not decoration for the same reason it is
/// not in `ops::local`: an element whose decimation counts from
/// [`StepOrigin::ClippedStart`](super::StepOrigin::ClippedStart) reads a
/// different set of offsets where the window is clipped at a **low face of the
/// volume**, and a block holding the middle of a volume has no such face. Asking
/// the element at `at.offset + voxel` inside `at.volume` is what makes a block's
/// answer the answer the whole-volume run would have written there.
///
/// The window is still clamped to the buffer, which at a real face is the global
/// clamp and short of a sufficient halo is the truncation that makes a short halo
/// visible. That is unchanged and is a separate question from which offsets are
/// gathered.
pub fn rank_filter_into_at<T: Ord + Copy>(
    input: ArrayView3<'_, T>,
    at: &Anchor,
    element: &StructuringElement,
    rank: Rank,
    out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    selecting(
        input,
        at,
        None,
        element,
        rank,
        ExcludedCentre::Select,
        out,
        "rank_filter_into",
    )
}

/// The anchor a caller who handed over a bare array is stating: this array is
/// the volume.
///
/// One function rather than four call sites, so that the reading every
/// anchor-free entry point in this file takes is one statement. `Anchor::whole`
/// puts the offset at the origin, so an element that re-phases at a low face
/// re-phases at *this array's* low face — which is the whole volume's, because
/// the caller said so.
fn whole(shape: &[usize]) -> Anchor {
    Anchor::whole([shape[0], shape[1], shape[2]])
}

/// What the masked filter writes at a voxel the population excludes **at its
/// centre**.
///
/// A window whose centre is outside the population is a different situation from
/// a window that merely lost some of its neighbours, and the two answers a
/// caller can reasonably want are far enough apart that guessing between them is
/// not on: either the filter is computed there anyway, from whatever neighbours
/// did survive, or the voxel is one the caller has already decided about and
/// wants a stated value at.
///
/// **The condition is the centre's own bit, and not "the window came out
/// empty".** Those two coincide for an element that contains its own centre —
/// where they are the same voxel — and come apart for one that does not, which
/// this crate's elements are allowed to be. Keeping them distinct is what lets
/// [`Self::Fill`] mean what its name says at every element rather than only at
/// the usual ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExcludedCentre<T> {
    /// Filter anyway. The centre's own exclusion costs it nothing beyond the one
    /// value it does not contribute, and the answer comes from the neighbours
    /// that survived; where **none** did, the centre's own value is written, for
    /// the reason [`masked_rank_filter_into`] gives.
    ///
    /// The default, and the only behaviour there was before there was a choice.
    #[default]
    Select,
    /// Write this value and read no window at all.
    ///
    /// The filter's usual guarantee — every value written is a value that was
    /// read — is given up here **on purpose and only here**: the caller has
    /// named a value for voxels it has already excluded, and a stated constant
    /// is a decision rather than an invention. Everywhere the centre is inside
    /// the population this behaves exactly as [`Self::Select`] does, the
    /// empty-window carry included, which is the case an element that misses its
    /// own centre can still reach.
    Fill(T),
}

impl<T> ExcludedCentre<T> {
    /// The same policy over another element type.
    ///
    /// The shells state the fill in `f64`, as every other constant in this
    /// module is stated, and the kernel needs it in the type it selects; this is
    /// the one conversion between them, so a widened path and a direct path
    /// cannot drift.
    pub fn map<U>(self, convert: impl FnOnce(T) -> U) -> ExcludedCentre<U> {
        match self {
            ExcludedCentre::Select => ExcludedCentre::Select,
            ExcludedCentre::Fill(value) => ExcludedCentre::Fill(convert(value)),
        }
    }
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
///
/// That last sentence is a *default* rather than a law — see
/// [`masked_rank_filter_into_with`] for the caller who wants an excluded centre
/// left at a value of their own instead.
pub fn masked_rank_filter_into<T: Ord + Copy>(
    input: ArrayView3<'_, T>,
    mask: ArrayView3<'_, bool>,
    element: &StructuringElement,
    rank: Rank,
    out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    masked_rank_filter_into_with(input, mask, element, rank, ExcludedCentre::Select, out)
}

/// The same filter with the policy at an excluded centre stated.
///
/// [`masked_rank_filter_into`] is this at [`ExcludedCentre::Select`], which is
/// what it did before there was a choice, so the shorter name keeps every
/// existing caller's answer to the bit.
///
/// The policy is a **per-voxel function of the mask over the same window the
/// filter already reads**, so it costs no reach and survives decomposition for
/// the reason everything else here does: whether a centre is excluded is a fact
/// about the volume's mask at that voxel, and which block the voxel landed in
/// cannot change it.
pub fn masked_rank_filter_into_with<T: Ord + Copy>(
    input: ArrayView3<'_, T>,
    mask: ArrayView3<'_, bool>,
    element: &StructuringElement,
    rank: Rank,
    centre: ExcludedCentre<T>,
    out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    let at = whole(input.shape());
    masked_rank_filter_into_at(input, &at, mask, element, rank, centre, out)
}

/// [`masked_rank_filter_into_with`] with the buffer's place in its volume
/// stated; see [`rank_filter_into_at`] for what `at` decides and for the one
/// element it decides anything for.
#[allow(clippy::too_many_arguments)]
pub fn masked_rank_filter_into_at<T: Ord + Copy>(
    input: ArrayView3<'_, T>,
    at: &Anchor,
    mask: ArrayView3<'_, bool>,
    element: &StructuringElement,
    rank: Rank,
    centre: ExcludedCentre<T>,
    out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    shapes_agree(
        input.shape(),
        mask.shape(),
        "masked_rank_filter_into (mask)",
    )?;
    selecting(
        input,
        at,
        Some(mask),
        element,
        rank,
        centre,
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
///
/// `centre` is meaningful only where there is a mask to be excluded by; the
/// unmasked callers pass [`ExcludedCentre::Select`], which is also what makes
/// the arm below `if let`-shaped rather than a second branch of the loop.
#[allow(clippy::too_many_arguments)]
fn selecting<T: Ord + Copy>(
    input: ArrayView3<'_, T>,
    at: &Anchor,
    mask: Option<ArrayView3<'_, bool>>,
    element: &StructuringElement,
    rank: Rank,
    centre: ExcludedCentre<T>,
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
    // **Where the anchor becomes load-bearing, and only there.** An element whose
    // origin is `StepOrigin::Anchor` reads the same offsets wherever it is
    // evaluated, so a wrong `at` could not change its answer and checking one
    // would refuse calls that were always correct. A re-phasing element reads the
    // volume's low faces, so a buffer that claims to sit outside its own volume
    // would ask `offsets_at` a question with no answer — and get an empty window,
    // which this kernel would report as an element that misses its own centre.
    // Said here instead, once per call.
    if element.origin() == StepOrigin::ClippedStart {
        for axis in 0..3 {
            if at.offset[axis] + extent[axis] as usize > at.volume[axis] {
                return Err(Error::InvalidArgument(format!(
                    "{what}: a buffer of {:?} at {:?} does not fit a volume of {:?}, and this \
                     element's step counts from the clipped start of the window, so where the \
                     buffer sits in the volume is part of the filter",
                    input.shape(),
                    at.offset,
                    at.volume
                )));
            }
        }
    }
    let full = element.len();
    let mut window: Vec<T> = Vec::with_capacity(full);
    // The element's offsets **at one voxel**, for the one element that has more
    // than one set of them. Owned out here so that a voxel pays no allocation for
    // it, and untouched by every other element.
    let mut offsets: Vec<[isize; 3]> = Vec::new();
    // **The one offset set, where there is one**, lifted out of the loop rather
    // than asked for at every voxel. `offsets_at` hands back exactly this slice
    // for an anchored element, so the two are the same answer; the lift is what
    // keeps the path that has nothing to ask about — every element without a
    // step, and therefore almost every call this kernel gets — a field load and a
    // slice walk, which is what it was before this file honoured anything.
    // `the_cost_of_asking_the_element_per_voxel` prices all three arrangements
    // and could not separate them above the noise of the machine it ran on, so
    // this is the shape of the thing rather than a measured saving.
    let fixed = (element.origin() == StepOrigin::Anchor).then(|| element.offsets());
    for i in 0..input.shape()[0] {
        for j in 0..input.shape()[1] {
            for k in 0..input.shape()[2] {
                // The one place the *centre's* own membership is consulted. It
                // is asked before the window is gathered, because under a fill
                // there is nothing to gather it for — and it is asked of the
                // mask rather than of the window's emptiness, which is the same
                // question only for an element that holds its own centre.
                if let (Some(mask), ExcludedCentre::Fill(value)) = (mask.as_ref(), centre) {
                    if !mask[[i, j, k]] {
                        out[[i, j, k]] = value;
                        continue;
                    }
                }
                window.clear();
                let anchor = [i as isize, j as isize, k as isize];
                let gathered = match fixed {
                    Some(offsets) => offsets,
                    // The same voxel, in the volume's coordinates. The element is
                    // asked there and the array is read here: which offsets are
                    // gathered is the volume's business — its low faces are the
                    // same faces from inside every block — and the clamp below is
                    // the buffer's, which is the global clamp intersected with
                    // what is held.
                    None => {
                        let placed = [
                            anchor[0] + at.offset[0] as isize,
                            anchor[1] + at.offset[1] as isize,
                            anchor[2] + at.offset[2] as isize,
                        ];
                        element.offsets_at(placed, at.volume, &mut offsets)
                    }
                };
                for offset in gathered {
                    let a = anchor[0] + offset[0];
                    let b = anchor[1] + offset[1];
                    let c = anchor[2] + offset[2];
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
                    // answer. Under a fill this arm is still reachable, and only
                    // for an element that misses its own centre: a centre that
                    // is in the mask and in the element is in the window.
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
    out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    let at = whole(input.shape());
    rank_filter_f64_into_at(input, &at, element, rank, out)
}

/// [`rank_filter_f64_into`] with the buffer's place in its volume stated; see
/// [`rank_filter_into_at`].
pub fn rank_filter_f64_into_at(
    input: ArrayView3<'_, f64>,
    at: &Anchor,
    element: &StructuringElement,
    rank: Rank,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), "rank_filter_f64_into")?;
    let ordered = input.mapv(Total);
    let mut selected = Array3::from_elem(ordered.raw_dim(), Total(0.0));
    rank_filter_into_at(ordered.view(), at, element, rank, selected.view_mut())?;
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

    /// **`at` is read rather than ignored**, which is what lets an element whose
    /// step counts from the clipped start be the same filter under every
    /// decomposition: the window re-phases at the volume's low faces, and only
    /// the anchor says where those are from inside a block. Every other element
    /// reads the same offsets everywhere and cannot tell the difference.
    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        /// The integer and `bool` case: straight into the kernel, no copy.
        fn ordered<T: VoxelElement + Ord>(
            input: &Voxels,
            at: &Anchor,
            out: &mut Voxels,
            element: &StructuringElement,
            rank: Rank,
        ) -> Result<()> {
            let source = input.view::<T>()?;
            rank_filter_into_at(source, at, element, rank, out.view_mut::<T>()?)
        }

        match input.dtype() {
            Dtype::Bool => ordered::<bool>(input, at, out, &self.element, self.rank),
            Dtype::U8 => ordered::<u8>(input, at, out, &self.element, self.rank),
            Dtype::U16 => ordered::<u16>(input, at, out, &self.element, self.rank),
            Dtype::U32 => ordered::<u32>(input, at, out, &self.element, self.rank),
            Dtype::U64 => ordered::<u64>(input, at, out, &self.element, self.rank),
            Dtype::I8 => ordered::<i8>(input, at, out, &self.element, self.rank),
            Dtype::I16 => ordered::<i16>(input, at, out, &self.element, self.rank),
            Dtype::I32 => ordered::<i32>(input, at, out, &self.element, self.rank),
            Dtype::I64 => ordered::<i64>(input, at, out, &self.element, self.rank),
            // The floats have no total order of their own, so they take the
            // `Total` detour. `f32` widens to `f64` on the way in and back on
            // the way out, which is exact both ways because the filter selects a
            // value it read rather than combining two.
            Dtype::F64 => rank_filter_f64_into_at(
                input.view::<f64>()?,
                at,
                &self.element,
                self.rank,
                out.view_mut::<f64>()?,
            ),
            Dtype::F32 => {
                let widened = input.view::<f32>()?.mapv(f64::from);
                let mut selected = Array3::zeros(widened.raw_dim());
                rank_filter_f64_into_at(
                    widened.view(),
                    at,
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
    centre: ExcludedCentre<f64>,
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
            centre: ExcludedCentre::Select,
            cost,
        }
    }

    /// State what happens at a voxel whose **own** membership the population
    /// denies. See [`ExcludedCentre`]; the default is to filter there anyway.
    ///
    /// A builder rather than an argument to [`Self::new`], so that the choice is
    /// additive: a caller who never had it keeps its call and its answer.
    ///
    /// The value is stated in `f64` and converted to the level's element type by
    /// the saturating `VoxelElement::from_f64`, which is how every other
    /// constant in this module crosses that boundary.
    pub fn with_excluded_centre(mut self, centre: ExcludedCentre<f64>) -> Self {
        self.centre = centre;
        self
    }

    /// [`Self::with_excluded_centre`] at [`ExcludedCentre::Fill`], which is the
    /// spelling most callers of it want.
    pub fn filling_excluded_centres(self, value: f64) -> Self {
        self.with_excluded_centre(ExcludedCentre::Fill(value))
    }

    pub fn excluded_centre(&self) -> ExcludedCentre<f64> {
        self.centre
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

    /// `at` is read, for the reason [`RankFilterOp::apply`] gives.
    fn apply_with(
        &self,
        input: &Voxels,
        sources: crate::op::SourceInputs<'_>,
        out: &mut Voxels,
        at: &Anchor,
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

        #[allow(clippy::too_many_arguments)]
        fn ordered<T: VoxelElement + Ord>(
            input: &Voxels,
            at: &Anchor,
            mask: ArrayView3<'_, bool>,
            out: &mut Voxels,
            element: &StructuringElement,
            rank: Rank,
            centre: ExcludedCentre<f64>,
        ) -> Result<()> {
            let source = input.view::<T>()?;
            masked_rank_filter_into_at(
                source,
                at,
                mask,
                element,
                rank,
                centre.map(T::from_f64),
                out.view_mut::<T>()?,
            )
        }

        /// The floating-point detour, shared by `f64` and `f32`: neither has a
        /// total order, and the filter hands back a value it read, so the trip
        /// through `Total` is exact in both directions.
        fn through_total(
            widened: ndarray::Array3<f64>,
            at: &Anchor,
            mask: ArrayView3<'_, bool>,
            element: &StructuringElement,
            rank: Rank,
            centre: ExcludedCentre<f64>,
        ) -> Result<Array3<f64>> {
            let ordered = widened.mapv(Total);
            let mut selected = Array3::from_elem(ordered.raw_dim(), Total(0.0));
            masked_rank_filter_into_at(
                ordered.view(),
                at,
                mask,
                element,
                rank,
                centre.map(Total),
                selected.view_mut(),
            )?;
            Ok(selected.mapv(|value| value.0))
        }

        let centre = self.centre;
        match input.dtype() {
            Dtype::Bool => ordered::<bool>(input, at, mask, out, &self.element, self.rank, centre),
            Dtype::U8 => ordered::<u8>(input, at, mask, out, &self.element, self.rank, centre),
            Dtype::U16 => ordered::<u16>(input, at, mask, out, &self.element, self.rank, centre),
            Dtype::U32 => ordered::<u32>(input, at, mask, out, &self.element, self.rank, centre),
            Dtype::U64 => ordered::<u64>(input, at, mask, out, &self.element, self.rank, centre),
            Dtype::I8 => ordered::<i8>(input, at, mask, out, &self.element, self.rank, centre),
            Dtype::I16 => ordered::<i16>(input, at, mask, out, &self.element, self.rank, centre),
            Dtype::I32 => ordered::<i32>(input, at, mask, out, &self.element, self.rank, centre),
            Dtype::I64 => ordered::<i64>(input, at, mask, out, &self.element, self.rank, centre),
            Dtype::F64 => {
                let selected = through_total(
                    input.view::<f64>()?.to_owned(),
                    at,
                    mask,
                    &self.element,
                    self.rank,
                    centre,
                )?;
                out.view_mut::<f64>()?.assign(&selected);
                Ok(())
            }
            Dtype::F32 => {
                let selected = through_total(
                    input.view::<f32>()?.mapv(f64::from),
                    at,
                    mask,
                    &self.element,
                    self.rank,
                    centre,
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

    /// Exactly the constant, and **whatever the mask says** — but only while the
    /// filter writes nothing the input did not hold.
    ///
    /// Under [`ExcludedCentre::Select`] every value written is a value that was
    /// read: where the population is non-empty the selection returns one of
    /// them, and where it is empty the centre is written. On a constant block
    /// both are the constant, so this holds bit for bit without the short
    /// circuit needing to know the mask — which matters, because the mask is a
    /// level the short circuit has not read.
    ///
    /// A [`ExcludedCentre::Fill`] breaks exactly that. The output of a constant
    /// block is then the constant where the mask keeps the centre and the fill
    /// where it does not, and **which of the two a voxel gets is a fact about
    /// the mask** — the one thing the short circuit cannot see. So the
    /// declaration is withdrawn, and survives only in the case where the answer
    /// does not depend on the mask at all: a fill that *is* the constant. A
    /// wrong declaration here would be a block skipped into the wrong values, so
    /// the narrow rule is the only safe one.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        match self.centre {
            ExcludedCentre::Select => Some(value),
            ExcludedCentre::Fill(fill) => (fill == value).then_some(value),
        }
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
///
/// `pub(super)` because the lattice statistic charges the same thing for the
/// same reason — one byte of a second array per offset visited — and a second
/// copy of the number is a second thing to re-measure.
pub(super) const MASK_COST_FACTOR: f64 = 1.35;

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

    /// The same definition asked **at a stated anchor inside a stated volume**,
    /// which is the whole of what honouring the origin means: the element is
    /// asked what it reads there rather than assumed to read one set everywhere.
    ///
    /// Written out with `offsets_at` for the same reason `by_definition` is
    /// written out with `offsets` — it is a statement of the operation, not a
    /// second implementation of it.
    fn by_definition_at(
        input: &Array3<f64>,
        at: &Anchor,
        element: &StructuringElement,
        rank: Rank,
    ) -> Array3<f64> {
        let shape = input.dim();
        let mut out = Array3::zeros(shape);
        let mut scratch = Vec::new();
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    let placed = [
                        (i + at.offset[0]) as isize,
                        (j + at.offset[1]) as isize,
                        (k + at.offset[2]) as isize,
                    ];
                    let mut window = Vec::new();
                    for offset in element.offsets_at(placed, at.volume, &mut scratch) {
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

    /// A step counted **from the anchor**, which is every element without a step
    /// and every element that names that origin.
    fn anchored(size: [usize; 3], step: [usize; 3]) -> StructuringElement {
        StructuringElement::from_size_stepped_at(ElementShape::Box, size, step, StepOrigin::Anchor)
            .unwrap()
    }

    fn clipped(shape: ElementShape, size: [usize; 3], step: [usize; 3]) -> StructuringElement {
        StructuringElement::from_size_stepped_at(shape, size, step, StepOrigin::ClippedStart)
            .unwrap()
    }

    /// **The anchored window did not move**, which every existing caller depends
    /// on and which an unstepped element normalises to, so it is most of the
    /// crate's elements and all of its history.
    ///
    /// Two halves. The filter still is the `offsets` definition, bit for bit;
    /// and an anchored element **cannot see the anchor** — the same array filtered
    /// as the whole volume and as a buffer claiming to sit deep inside a larger
    /// one gives the identical answer, because there is nothing about its window
    /// for a face to change.
    #[test]
    fn the_anchored_window_is_byte_unchanged() {
        let input = ramp((9, 7, 6));
        let elements = [
            StructuringElement::from_radius(ElementShape::Box, [2, 1, 1]),
            StructuringElement::from_size(ElementShape::Ellipsoid, [5, 5, 3]).unwrap(),
            anchored([8, 8, 1], [2, 2, 1]),
            anchored([9, 7, 3], [2, 2, 1]),
        ];
        for element in elements {
            assert_eq!(
                element.origin(),
                StepOrigin::Anchor,
                "the subject of this test"
            );
            for rank in [
                Rank::lowest(),
                Rank::median(&element),
                Rank::highest(&element),
                Rank::ceiling_percentile(0.25).unwrap(),
            ] {
                let mut whole_volume = Array3::zeros(input.dim());
                rank_filter_f64_into(input.view(), &element, rank, whole_volume.view_mut())
                    .unwrap();
                assert_eq!(
                    whole_volume,
                    by_definition(&input, &element, rank),
                    "{element:?} {rank:?}"
                );

                let inside = Anchor::new([40, 30, 20], [100, 90, 80]);
                let mut placed = Array3::zeros(input.dim());
                rank_filter_f64_into_at(input.view(), &inside, &element, rank, placed.view_mut())
                    .unwrap();
                assert_eq!(
                    placed, whole_volume,
                    "an anchored element reads the same offsets everywhere"
                );
            }
        }
    }

    /// **The re-phasing element gathers the window it names**, which is the gap
    /// this file used to have and used to say so.
    ///
    /// Against `offsets_at` written out, and — the half that keeps it from being
    /// a tautology — asserted to differ from the anchored gather, so that the
    /// first comparison is over a case where the two rules disagree.
    #[test]
    fn a_clipped_start_element_gathers_the_window_it_names() {
        // A window wider than the volume on axis 0, so that *every* anchor there
        // is inside `lo` of the low face and the phase moves at every voxel.
        let input = ramp((7, 5, 1));
        let at = Anchor::whole([7, 5, 1]);
        let mut differed = 0usize;
        for element in [
            clipped(ElementShape::Box, [9, 3, 1], [2, 1, 1]),
            clipped(ElementShape::Box, [8, 4, 1], [2, 2, 1]),
            clipped(ElementShape::Ellipsoid, [9, 5, 1], [2, 2, 1]),
        ] {
            assert_eq!(element.origin(), StepOrigin::ClippedStart);
            for rank in [
                Rank::median(&element),
                Rank::ceiling_percentile(0.4).unwrap(),
            ] {
                let mut got = Array3::zeros(input.dim());
                rank_filter_f64_into(input.view(), &element, rank, got.view_mut()).unwrap();
                assert_eq!(
                    got,
                    by_definition_at(&input, &at, &element, rank),
                    "{element:?} {rank:?}"
                );
                differed += got
                    .iter()
                    .zip(by_definition(&input, &element, rank).iter())
                    .filter(|(left, right)| left.to_bits() != right.to_bits())
                    .count();
            }
        }
        assert!(
            differed > 0,
            "the two rules must disagree somewhere on this volume, or the comparison above is \
             a comparison of one rule with itself"
        );
    }

    /// **The phase is keyed on the volume's face, not the buffer's.** A buffer
    /// that holds the middle of a volume has no low face, and every voxel of it
    /// far enough from its own edges must get the whole-volume answer.
    ///
    /// The margin is the element's own reach: the window is still clamped to the
    /// buffer, so a voxel within a side of the buffer's edge is reading a
    /// truncated window and is the halo the caller is expected to discard. That
    /// is the existing clamp convention and is a separate question from which
    /// offsets are gathered.
    #[test]
    fn the_phase_is_keyed_on_the_volumes_face_and_not_the_buffers() {
        let volume = [12usize, 9, 1];
        let input = ramp((volume[0], volume[1], volume[2]));
        let element = clipped(ElementShape::Box, [9, 5, 1], [2, 2, 1]);
        let rank = Rank::median(&element);
        let mut whole_volume = Array3::zeros(input.dim());
        rank_filter_f64_into(input.view(), &element, rank, whole_volume.view_mut()).unwrap();

        // a buffer holding the upper part of axis 0, halo included
        let offset = [4usize, 0, 0];
        let extent = [volume[0] - offset[0], volume[1], volume[2]];
        let buffer = input
            .slice(ndarray::s![
                offset[0]..offset[0] + extent[0],
                0..extent[1],
                0..extent[2]
            ])
            .to_owned();
        let at = Anchor::new(offset, volume);
        let mut blocked = Array3::zeros(buffer.dim());
        rank_filter_f64_into_at(buffer.view(), &at, &element, rank, blocked.view_mut()).unwrap();

        let (lo, _) = element.sides(0);
        let mut compared = 0usize;
        for i in lo..extent[0] {
            for j in 0..extent[1] {
                for k in 0..extent[2] {
                    assert_eq!(
                        blocked[[i, j, k]].to_bits(),
                        whole_volume[[i + offset[0], j, k]].to_bits(),
                        "at {:?}",
                        [i + offset[0], j, k]
                    );
                    compared += 1;
                }
            }
        }
        assert!(compared > 0);

        // and the negative control: the same buffer read as a volume of its own
        // re-phases at its own low face and is a different answer there, so the
        // comparison above is not one an indifferent implementation would pass
        let mut as_its_own_volume = Array3::zeros(buffer.dim());
        rank_filter_f64_into(buffer.view(), &element, rank, as_its_own_volume.view_mut()).unwrap();
        assert_ne!(
            as_its_own_volume, blocked,
            "a buffer's own edge must not be a face, and here the two readings differ"
        );
    }

    /// A buffer that claims to sit outside its own volume is refused, and only
    /// for the element the claim could change. See `selecting`.
    #[test]
    fn a_buffer_outside_its_volume_is_refused_where_the_element_re_phases() {
        let input = ramp((6, 4, 1));
        let outside = Anchor::new([4, 0, 0], [6, 4, 1]);
        let element = clipped(ElementShape::Box, [7, 3, 1], [2, 1, 1]);
        let mut out = Array3::zeros(input.dim());
        let failed = rank_filter_f64_into_at(
            input.view(),
            &outside,
            &element,
            Rank::median(&element),
            out.view_mut(),
        );
        let message = failed
            .expect_err("a buffer that does not fit is refused")
            .to_string();
        assert!(message.contains("does not fit"), "{message}");
        assert!(message.contains("clipped start"), "{message}");

        // the same anchor with an element that reads one offset set everywhere is
        // accepted, because nothing about that element's window depends on it
        let anchored = StructuringElement::from_radius(ElementShape::Box, [1, 1, 0]);
        rank_filter_f64_into_at(
            input.view(),
            &outside,
            &anchored,
            Rank::median(&anchored),
            out.view_mut(),
        )
        .expect("an anchored element cannot be affected by the anchor, so it is not asked about");
    }

    /// **What asking the element per voxel costs**, on the path that does not
    /// need to be asked — every anchored element, which is every element without
    /// a step.
    ///
    /// Three loops, written out here rather than called, and identical in every
    /// line but the one being priced — a wrapper around any of them would price
    /// the wrapper, and the difference is a handful of instructions.
    ///
    /// * `fixed`, the gather this kernel had: `element.offsets()`, one set;
    /// * `lifted`, the gather it has: the one set taken out of the loop into an
    ///   `Option` and matched per voxel, which is what `selecting`'s `let fixed`
    ///   is;
    /// * `asked`, the gather it would have if it asked `offsets_at` at every
    ///   voxel — the same answer for an anchored element, and what the lift
    ///   exists so that the path with nothing to ask about does not do.
    ///
    /// **What it measured**, on the machine this was written on and over four
    /// runs: the three sit within a few per cent of each other and of the run to
    /// run spread, which was itself large enough to invert the ordering once. So
    /// the honest reading is that honouring the origin costs the anchored path
    /// **nothing separable from noise**, and that the lift is worth keeping for
    /// the narrower reason that it makes that path a field load and a slice walk,
    /// exactly as it was, rather than for a number. A quieter machine could
    /// separate them; this one could not, and saying so is better than quoting
    /// the one figure that came out of it.
    ///
    /// Ignored, because it is a measurement and not an assertion — `super`'s
    /// convention for every cost figure in this module. Run it with
    /// `cargo test --release -- --ignored the_cost_of_asking`.
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn the_cost_of_asking_the_element_per_voxel() {
        use std::time::Instant;

        let values = ramp((60, 60, 12)).mapv(Total);
        let shape = values.dim();
        let voxels = values.len() as f64;
        let at = Anchor::whole([shape.0, shape.1, shape.2]);
        for size in [[5, 5, 3], [15, 15, 1]] {
            let element = StructuringElement::from_size(ElementShape::Box, size).unwrap();
            let rank = Rank::median(&element);
            let full = element.len();
            let lift = (element.origin() == StepOrigin::Anchor).then(|| element.offsets());
            let mut out = Array3::from_elem(values.raw_dim(), Total(0.0));
            let mut window: Vec<Total> = Vec::with_capacity(full);
            let mut scratch: Vec<[isize; 3]> = Vec::new();

            // The body all three share, as a macro rather than a closure so that
            // the gather is inlined into it exactly as it is in the kernel and
            // nothing is being priced through an indirect call.
            macro_rules! sweep {
                ($anchor:ident, $gather:expr) => {{
                    let started = Instant::now();
                    for i in 0..shape.0 {
                        for j in 0..shape.1 {
                            for k in 0..shape.2 {
                                window.clear();
                                let $anchor = [i as isize, j as isize, k as isize];
                                for offset in $gather {
                                    let a = $anchor[0] + offset[0];
                                    let b = $anchor[1] + offset[1];
                                    let c = $anchor[2] + offset[2];
                                    if a < 0
                                        || b < 0
                                        || c < 0
                                        || a as usize >= shape.0
                                        || b as usize >= shape.1
                                        || c as usize >= shape.2
                                    {
                                        continue;
                                    }
                                    window.push(values[[a as usize, b as usize, c as usize]]);
                                }
                                let index = rank.resolve(full, window.len());
                                out[[i, j, k]] = select_nth(&mut window, index).unwrap();
                            }
                        }
                    }
                    started.elapsed().as_secs_f64() * 1e9 / voxels
                }};
            }

            let mut best = [f64::INFINITY; 3];
            for repeat in 0..10 {
                let fixed = sweep!(anchor, element.offsets());
                let lifted = sweep!(
                    anchor,
                    match lift {
                        Some(offsets) => offsets,
                        None => {
                            let centre = [
                                anchor[0] + at.offset[0] as isize,
                                anchor[1] + at.offset[1] as isize,
                                anchor[2] + at.offset[2] as isize,
                            ];
                            element.offsets_at(centre, at.volume, &mut scratch)
                        }
                    }
                );
                let asked = sweep!(anchor, {
                    let centre = [
                        anchor[0] + at.offset[0] as isize,
                        anchor[1] + at.offset[1] as isize,
                        anchor[2] + at.offset[2] as isize,
                    ];
                    element.offsets_at(centre, at.volume, &mut scratch)
                });
                // the first pass pays for a cold output array
                if repeat > 0 {
                    for (slot, measured) in best.iter_mut().zip([fixed, lifted, asked]) {
                        *slot = slot.min(measured);
                    }
                }
            }
            println!(
                "{size:?} ({full} offsets): fixed {:.3}, lifted {:.3} ({:+.1}%), \
                 asked {:.3} ({:+.1}%) ns/voxel",
                best[0],
                best[1],
                (best[1] / best[0] - 1.0) * 100.0,
                best[2],
                (best[2] / best[0] - 1.0) * 100.0
            );
        }
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
