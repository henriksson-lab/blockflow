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

use std::collections::BTreeMap;

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Slicing};
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
    // **Asked once for the whole volume**, by the function that already owns
    // this question: it is a property of the rank and the element, not of any
    // voxel's truncated window, and `extreme_of`'s own header carries the
    // arithmetic that makes it agree with `Rank::resolve` at every truncation.
    let ends = extreme_of(rank, full);
    // **The flat path**, for the case that is almost every call this kernel gets
    // from a morphology: one fixed offset list, reduced to an end, no mask, and
    // an input whose memory is one contiguous run in C order.
    //
    // What it removes is not the compare — that stays — but everything around
    // it: the per-tap clamp against the volume, and `ndarray`'s own index
    // arithmetic and bounds check on `[[a, b, c]]`. A tap becomes one add and
    // one slice read.
    //
    // It applies only to the **interior**, where the whole element fits inside
    // the volume, because that is exactly where the clamp is known to be a
    // no-op. Every boundary voxel falls through to the general path, which is
    // the one that has always been right about truncation — see
    // `Rank::resolve`, whose rescaling rule is the thing a fast path must not
    // reimplement.
    let flat = match (ends, fixed, mask.is_some(), input.as_slice()) {
        (Some(_), Some(offsets), false, Some(values)) if !offsets.is_empty() => {
            let strides = input.strides();
            let mut lo = [isize::MAX; 3];
            let mut hi = [isize::MIN; 3];
            for offset in offsets {
                for axis in 0..3 {
                    lo[axis] = lo[axis].min(offset[axis]);
                    hi[axis] = hi[axis].max(offset[axis]);
                }
            }
            let steps: Vec<isize> = offsets
                .iter()
                .map(|offset| {
                    offset[0] * strides[0] + offset[1] * strides[1] + offset[2] * strides[2]
                })
                .collect();
            Some((values, strides.to_vec(), steps, lo, hi))
        }
        _ => None,
    };
    // **The interior, by running extrema**, where the element is planar and the
    // rank is an end. This writes the interior outright and hands back what it
    // wrote, so the loop below owns only the boundary shell — the part whose
    // truncation rule is `Rank::resolve`'s and must not be reimplemented.
    //
    // It is tried before the flat path rather than instead of it: an element
    // with depth, or a row the runs cannot span, still gets the flat interior.
    let running = match (ends, flat.as_ref()) {
        (Some(take), Some((values, _, _, lo, hi))) => {
            let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
            let offsets = fixed.expect("the flat path is built from one fixed offset list");
            by_separable_box(values, shape, offsets, take, *lo, *hi, &mut out)
                .or_else(|| by_running_extremes(values, shape, offsets, take, *lo, *hi, &mut out))
        }
        _ => None,
    };
    for i in 0..input.shape()[0] {
        for j in 0..input.shape()[1] {
            for k in 0..input.shape()[2] {
                // Already written, by the running kernel above.
                if let Some([[i0, i1], [j0, j1], [k0, k1]]) = running {
                    if i >= i0 && i < i1 && j >= j0 && j < j1 && k >= k0 && k < k1 {
                        continue;
                    }
                }
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
                // **The neighbour, or nothing.** One place where a voxel is
                // clamped to the buffer and tested against the mask, so that
                // the two paths below cannot drift on either rule.
                let value_at = |offset: &[isize; 3]| -> Option<T> {
                    let a = anchor[0] + offset[0];
                    let b = anchor[1] + offset[1];
                    let c = anchor[2] + offset[2];
                    if a < 0 || b < 0 || c < 0 || a >= extent[0] || b >= extent[1] || c >= extent[2]
                    {
                        return None;
                    }
                    let at = [a as usize, b as usize, c as usize];
                    if let Some(mask) = mask.as_ref() {
                        if !mask[at] {
                            return None;
                        }
                    }
                    Some(input[at])
                };
                // **A min or a max is folded as the neighbours are read.** The
                // window exists to be *selected over*; an end needs no selection,
                // so gathering into it would be building a buffer to throw away.
                // `Rank::extreme` can answer this before the gather — that is
                // what it is for — where `Rank::resolve` cannot, since it needs
                // the count the gather produces.
                //
                // An erosion is `Rank::lowest` and a dilation is `Rank::highest`,
                // so a grey opening takes this path twice, and the cell chain's
                // background arm is exactly one opening.
                // The interior, flat. `gathered` is the same offset list the
                // general path walks, so this is the same set of taps read a
                // cheaper way.
                let interior = flat.as_ref().and_then(|(values, strides, steps, lo, hi)| {
                    let inside = (0..3).all(|axis| {
                        anchor[axis] + lo[axis] >= 0 && anchor[axis] + hi[axis] < extent[axis]
                    });
                    if !inside {
                        return None;
                    }
                    let base =
                        anchor[0] * strides[0] + anchor[1] * strides[1] + anchor[2] * strides[2];
                    let mut best = values[(base + steps[0]) as usize];
                    match ends {
                        Some(Extreme::Lowest) => {
                            for step in &steps[1..] {
                                let value = values[(base + step) as usize];
                                if value < best {
                                    best = value;
                                }
                            }
                        }
                        _ => {
                            for step in &steps[1..] {
                                let value = values[(base + step) as usize];
                                if value > best {
                                    best = value;
                                }
                            }
                        }
                    }
                    Some(best)
                });
                let selected = match ends {
                    _ if interior.is_some() => interior,
                    Some(Extreme::Lowest) => gathered.iter().filter_map(&value_at).min(),
                    Some(Extreme::Highest) => gathered.iter().filter_map(&value_at).max(),
                    None => {
                        window.clear();
                        window.extend(gathered.iter().filter_map(&value_at));
                        let index = rank.resolve(full, window.len());
                        select_nth(&mut window, index)
                    }
                };
                match selected {
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

// ------------------------------------------- the run-decomposed candidate --

/// Which of the four arms of the rank-kernel experiment to run.
///
/// **An experiment, not a setting.** The two changes are separable and the whole
/// point of building them beside each other is to price them apart: a fast path
/// for the extremes that never selects, and a run decomposition of the element
/// so that the per-voxel cost stops scaling with its population. Attributing one
/// to the other is exactly the mistake the arms exist to prevent.
///
/// **Both arms live here, in the library**, and that is a design decision about
/// the *measurement* rather than about the code. A previous attempt at this
/// comparison put the candidate in the test and left the incumbent in the crate,
/// and the test-local arm won in both arrangements — the signature of the
/// compiler being free to inline one and not the other. Two library functions in
/// one module, called the same way, cannot differ in that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankPath {
    /// What ships: gather every offset into a scratch buffer, then select.
    Gather,
    /// Gather as before, but answer a minimum or a maximum without selecting.
    GatherExtreme,
    /// Gather along the element's contiguous runs, then select.
    Runs,
    /// Both.
    RunsExtreme,
}

impl RankPath {
    fn runs(self) -> bool {
        matches!(self, RankPath::Runs | RankPath::RunsExtreme)
    }

    fn extreme(self) -> bool {
        matches!(self, RankPath::GatherExtreme | RankPath::RunsExtreme)
    }
}

/// Which end of the window a rank asks for, when it asks for an end at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Extreme {
    Lowest,
    Highest,
}

/// Is this rank an extreme **for every window size**, so the question can be
/// settled once rather than per voxel?
///
/// `Rank::resolve` is the authority and this must agree with it exactly, so the
/// reasoning is written out rather than assumed. For `Nth(k)` with `k = 0` the
/// resolved index is `(0 * (m - 1) + d / 2) / d`, which is `0` for every `m`.
/// For `k = full - 1` it is `((full - 1)(m - 1) + d/2) / d` with `d = full - 1`,
/// which is `m - 1` for every `m`. Those are exactly `Rank::lowest()` and
/// `Rank::highest(element)`. A percentile of `0` resolves to `0` and one of `1`
/// to `m - 1` by the ceiling rule. Nothing else is an extreme at every size, and
/// `the_extreme_fast_path_agrees_with_selection_everywhere` checks the whole
/// claim against `resolve` rather than trusting this comment.
fn extreme_of(rank: Rank, full: usize) -> Option<Extreme> {
    match rank {
        Rank::Nth(0) => Some(Extreme::Lowest),
        Rank::Nth(k) if full > 0 && k >= full - 1 => Some(Extreme::Highest),
        Rank::CeilingPercentile(percentile) if percentile.fraction() <= 0.0 => {
            Some(Extreme::Lowest)
        }
        Rank::CeilingPercentile(percentile) if percentile.fraction() >= 1.0 => {
            Some(Extreme::Highest)
        }
        _ => None,
    }
}

/// The element's offsets grouped into **runs contiguous along the fastest
/// axis**: `(o0, o1, first_o2, length)`.
///
/// `StructuringElement::offsets` is ascending lexicographic by contract, so
/// offsets sharing `(o0, o1)` are adjacent and their `o2` ascends; a run is a
/// maximal span where `o2` increments by one. A **stepped** element decimates
/// that axis, so its runs are all of length one and this decomposition costs a
/// little and buys nothing — which is the honest behaviour rather than a special
/// case, and is why the report below includes a stepped row.
fn runs_of(offsets: &[[isize; 3]]) -> Vec<(isize, isize, isize, usize)> {
    let mut runs: Vec<(isize, isize, isize, usize)> = Vec::new();
    for offset in offsets {
        match runs.last_mut() {
            Some(last)
                if last.0 == offset[0]
                    && last.1 == offset[1]
                    && last.2 + last.3 as isize == offset[2] =>
            {
                last.3 += 1;
            }
            _ => runs.push((offset[0], offset[1], offset[2], 1)),
        }
    }
    runs
}

/// One end of a pair, chosen by which end the rank asked for.
///
/// Written once so that the running kernel below cannot fold a min where the
/// gather folds a max: every comparison in this file's fast paths goes through
/// either this or [`fold_into`].
fn pick<T: Ord>(take: Extreme, a: T, b: T) -> T {
    match take {
        Extreme::Lowest => a.min(b),
        Extreme::Highest => a.max(b),
    }
}

/// Fold `from` into `into`, elementwise, at the end the rank asked for.
///
/// **The shape of this loop is most of the point.** The same comparisons
/// written per voxel — walk the element, gather what each offset contributes —
/// measured `119 ns/voxel` for work that is a few dozen operations. Written over
/// two aligned slices it is one instruction a lane, and the branch on `take` is
/// hoisted out of the loop rather than taken inside it.
fn fold_into<T: Ord + Copy>(into: &mut [T], from: &[T], take: Extreme) {
    match take {
        Extreme::Lowest => {
            for (slot, &value) in into.iter_mut().zip(from) {
                if value < *slot {
                    *slot = value;
                }
            }
        }
        Extreme::Highest => {
            for (slot, &value) in into.iter_mut().zip(from) {
                if value > *slot {
                    *slot = value;
                }
            }
        }
    }
}

/// One run of an element, taken along **axis 1**: every offset it holds shares
/// an axis-0 step of `d0` and an axis-2 step of `d2`, and its axis-1 steps are
/// the contiguous span `start ..= start + len - 1`.
#[derive(Clone, Copy, Debug)]
struct RowRun {
    d0: isize,
    d2: isize,
    start: isize,
    len: usize,
}

/// The element's offsets as runs along **axis 1**.
///
/// # Why axis 1 and not the fastest axis
///
/// The obvious axis to run along is the fastest one, where a run is a
/// contiguous span of memory. That was the first thing written here, and it was
/// **inert on the element that motivated it**: a background disk stated as
/// `10 x 10 x 1` puts its flat axis last, so every offset shares `d2 == 0` and
/// every run along axis 2 has length one. A decomposition that turns an
/// eighty-tap element into eighty runs of one is the gather with extra
/// arithmetic, and it measured as exactly that — a `1.7x` faster kernel that
/// moved the chain it was written for by nothing.
///
/// Running along axis 1 costs nothing for it: the running pass folds whole
/// **rows** of the fastest axis into each other, so every inner loop is still
/// contiguous and still one instruction a lane. It is the *outer* index that
/// steps, not the inner one.
///
/// # It does not need a planar element
///
/// Grouping by `(d0, d2)` says nothing about depth, so a three-dimensional
/// element decomposes too — it simply has more groups. Runs are emitted maximal
/// rather than required contiguous, so an element with gaps decomposes as well;
/// it just gets more of them, and [`by_running_extremes`] declines when there
/// are so many that the gather would be cheaper.
fn row_runs_of(offsets: &[[isize; 3]]) -> Vec<RowRun> {
    let mut by_key: BTreeMap<(isize, isize), Vec<isize>> = BTreeMap::new();
    for offset in offsets {
        by_key
            .entry((offset[0], offset[2]))
            .or_default()
            .push(offset[1]);
    }
    let mut runs = Vec::new();
    for ((d0, d2), mut steps) in by_key {
        steps.sort_unstable();
        steps.dedup();
        let mut start = steps[0];
        let mut len = 1;
        for pair in steps.windows(2) {
            if pair[1] == pair[0] + 1 {
                len += 1;
            } else {
                runs.push(RowRun { d0, d2, start, len });
                start = pair[1];
                len = 1;
            }
        }
        runs.push(RowRun { d0, d2, start, len });
    }
    runs
}

/// The running extremum over **every window of `len` consecutive chunks**.
///
/// The data is `n1` chunks of `n2` elements each; `out`'s chunk `j` is the
/// elementwise extremum of chunks `j ..= j + len - 1`, for every `j` where that
/// window fits. Chunks past `n1 - len` are written but hold no promise, and no
/// caller reads them.
///
/// A chunk is a **row** when this walks a plane and a **plane** when it walks a
/// volume, which is what lets one function serve both of the slow axes: the
/// inner loop is contiguous either way, because it is the *outer* index that
/// steps.
///
/// This is van Herk's algorithm, and the reason it is worth having is that its
/// cost **does not depend on `len`**. Cut the rows into blocks of `len` and
/// take, for each row, the extremum from its block's start down to it
/// (`forward`) and from it to its block's end (`out`, folded in place). A window
/// of exactly `len` rows straddles exactly one block boundary, so it is covered
/// by one suffix and one prefix, and the answer is those two folded together —
/// three row-folds a row, whatever `len` is, against `len` for the gather.
///
/// The two halves meet without a gap: for a window starting at block offset
/// `r > 0` the suffix covers `[j, j + len - r - 1]` and the prefix covers
/// `[j + len - r, j + len - 1]`; at `r == 0` both cover the whole block, which
/// is the same answer twice rather than a hole.
fn running_extreme_strided<T: Ord + Copy>(
    plane: &[T],
    n1: usize,
    n2: usize,
    len: usize,
    take: Extreme,
    forward: &mut [T],
    out: &mut [T],
) {
    if len == 0 || len > n1 {
        return;
    }
    // The suffix within each block, folded in place. Walked block by block
    // rather than tested per row: the obvious spelling of "is this the last row
    // of its block" is a modulo, and a division in the innermost loop of the
    // innermost kernel is most of what this algorithm was supposed to save.
    out.copy_from_slice(plane);
    let mut start = 0;
    while start < n1 {
        let end = (start + len).min(n1);
        for j in (start..end.saturating_sub(1)).rev() {
            let (head, tail) = out.split_at_mut((j + 1) * n2);
            fold_into(&mut head[j * n2..], &tail[..n2], take);
        }
        start = end;
    }
    // The prefix within each block.
    forward.copy_from_slice(plane);
    let mut start = 0;
    while start < n1 {
        let end = (start + len).min(n1);
        for j in start + 1..end {
            let (head, tail) = forward.split_at_mut(j * n2);
            fold_into(&mut tail[..n2], &head[(j - 1) * n2..j * n2], take);
        }
        start = end;
    }
    for j in 0..=(n1 - len) {
        let source = (j + len - 1) * n2;
        let (head, tail) = if source > j * n2 {
            let (a, b) = out.split_at_mut(source);
            (&mut a[j * n2..j * n2 + n2], &b[..n2])
        } else {
            // `len == 1`: the window is the row itself and the prefix is a copy
            // of it, so there is nothing to fold.
            continue;
        };
        fold_into(head, &forward[source..source + n2], take);
        let _ = tail;
    }
}

/// The running extremum of one **contiguous line**, over every window of `len`.
///
/// [`running_extreme_strided`] with a chunk of one element would answer this,
/// and would spend the whole of its inner loop on slices of length one. The
/// fastest axis gets its own spelling for that reason and no other; the
/// algorithm is the same.
fn running_extreme_line<T: Ord + Copy>(
    row: &[T],
    len: usize,
    take: Extreme,
    forward: &mut [T],
    out: &mut [T],
) {
    let n = row.len();
    if len == 0 || len > n {
        return;
    }
    out.copy_from_slice(row);
    let mut start = 0;
    while start < n {
        let end = (start + len).min(n);
        for k in (start..end.saturating_sub(1)).rev() {
            out[k] = pick(take, out[k], out[k + 1]);
        }
        start = end;
    }
    forward.copy_from_slice(row);
    let mut start = 0;
    while start < n {
        let end = (start + len).min(n);
        for k in start + 1..end {
            forward[k] = pick(take, forward[k - 1], forward[k]);
        }
        start = end;
    }
    for k in 0..=(n - len) {
        out[k] = pick(take, out[k], forward[k + len - 1]);
    }
}

/// The element as a **full rectangular box**, or nothing.
///
/// Returns the per-axis window widths when the offsets are exactly the Cartesian
/// product of three contiguous ranges. Offsets are distinct, so counting them
/// against the bounding box's volume settles it — no search, and no way for a
/// box with a hole in it to pass.
fn box_widths(offsets: &[[isize; 3]], lo: [isize; 3], hi: [isize; 3]) -> Option<[usize; 3]> {
    let widths = [
        (hi[0] - lo[0] + 1) as usize,
        (hi[1] - lo[1] + 1) as usize,
        (hi[2] - lo[2] + 1) as usize,
    ];
    (offsets.len() == widths[0] * widths[1] * widths[2]).then_some(widths)
}

/// Fill the **interior** of a **box** element by three separable passes.
///
/// # Why a box is worth its own path
///
/// An extremum over a box factorises. `max` over `w0 x w1 x w2` offsets is `max`
/// along axis 2, then along axis 1, then along axis 0 — in any order, because
/// each is an extremum over a set and the box is a product of sets. Each pass is
/// van Herk's, which costs the same whatever its window is, so the whole filter
/// is **about a dozen operations a voxel no matter how large the box is**,
/// against one per tap for the gather.
///
/// The general run decomposition below already handles a box — it just handles
/// it as `w0 x w2` runs of `w1`, which is `w0 x w2` folds a voxel where this is
/// three passes. For the `5 x 5 x 5` window a local-maximum test uses, that is
/// twenty-five folds against three: the difference between a path worth taking
/// and the one worth taking instead.
///
/// # What it costs
///
/// Two scratch volumes, where the gather needs none — and the gather's `125`
/// reads all land in cache, being a `5 x 5 x 5` neighbourhood, where these
/// stream. The trade is real and it is why the guard below is a *tap count*
/// rather than "is it a box": a `3 x 3 x 3` box is twenty-seven cache-resident
/// reads, and three streaming passes over two volumes would be the slower way
/// to get the same answer.
#[allow(clippy::too_many_arguments)]
fn by_separable_box<T: Ord + Copy>(
    values: &[T],
    shape: [usize; 3],
    offsets: &[[isize; 3]],
    take: Extreme,
    lo: [isize; 3],
    hi: [isize; 3],
    out: &mut ArrayViewMut3<'_, T>,
) -> Option<[[usize; 2]; 3]> {
    let widths = box_widths(offsets, lo, hi)?;
    // Below this the gather's taps are cache-resident and cheaper than three
    // passes over memory. Measured on the two boxes this crate's consumers
    // use — `3 x 3 x 3` keeps the gather, `5 x 5 x 5` does not.
    if offsets.len() < 64 {
        return None;
    }
    let [n0, n1, n2] = shape;
    if widths[0] > n0 || widths[1] > n1 || widths[2] > n2 {
        return None;
    }
    let bound = |axis: usize, extent: usize| {
        let low = (-lo[axis]).max(0) as usize;
        let high = (extent as isize - hi[axis]).max(0) as usize;
        (low < high).then_some([low, high])
    };
    let inside = [bound(0, n0)?, bound(1, n1)?, bound(2, n2)?];
    let out = out.as_slice_mut()?;
    let plane = n1 * n2;
    let volume = n0 * plane;
    let mut first: Vec<T> = vec![values[0]; volume];
    let mut second: Vec<T> = vec![values[0]; volume];
    let mut forward: Vec<T> = vec![values[0]; volume];
    // Axis 2, along each contiguous line.
    for line in 0..n0 * n1 {
        let base = line * n2;
        running_extreme_line(
            &values[base..base + n2],
            widths[2],
            take,
            &mut forward[base..base + n2],
            &mut first[base..base + n2],
        );
    }
    // Axis 1, across the rows of each plane.
    for i in 0..n0 {
        let base = i * plane;
        running_extreme_strided(
            &first[base..base + plane],
            n1,
            n2,
            widths[1],
            take,
            &mut forward[base..base + plane],
            &mut second[base..base + plane],
        );
    }
    // Axis 0, across the planes of the volume.
    running_extreme_strided(
        &second,
        n0,
        plane,
        widths[0],
        take,
        &mut forward,
        &mut first,
    );
    // The composed volume answers at the window's **start**; the element answers
    // at its anchor, which sits `lo` inside it.
    let [[i0, i1], [j0, j1], [k0, k1]] = inside;
    let width = k1 - k0;
    for i in i0..i1 {
        for j in j0..j1 {
            let source = (((i as isize + lo[0]) as usize) * n1 + (j as isize + lo[1]) as usize)
                * n2
                + (k0 as isize + lo[2]) as usize;
            let target = (i * n1 + j) * n2 + k0;
            out[target..target + width].copy_from_slice(&first[source..source + width]);
        }
    }
    Some(inside)
}

/// Fill the **interior** by running extrema, or decline.
///
/// Returns the half-open ranges on each axis that it wrote, so the caller can
/// skip them; `None` means the caller's own loop owns every voxel, unchanged.
///
/// It applies to the interior only, where the whole element fits inside the
/// buffer and the gather's per-tap clamp is therefore known to be a no-op —
/// exactly the predicate the flat path applies per voxel, hoisted into a range.
/// Every boundary voxel falls through to the general path, which is the one
/// that has always been right about truncation.
///
/// **It declines an element it cannot beat.** One running pass per run is three
/// row-folds; one run of length `len` costs the gather `len`. So the
/// decomposition pays whenever the runs are longer than about three on average,
/// and an element that decomposes into short runs — a stepped one, whose runs
/// are all of length one — is left to the gather rather than given a slower
/// path with a nicer name.
#[allow(clippy::too_many_arguments)]
fn by_running_extremes<T: Ord + Copy>(
    values: &[T],
    shape: [usize; 3],
    offsets: &[[isize; 3]],
    take: Extreme,
    lo: [isize; 3],
    hi: [isize; 3],
    out: &mut ArrayViewMut3<'_, T>,
) -> Option<[[usize; 2]; 3]> {
    let runs = row_runs_of(offsets);
    if runs.is_empty() {
        return None;
    }
    let [n0, n1, n2] = shape;
    if runs.iter().any(|run| run.len > n1) {
        return None;
    }
    // Three row-folds a run against `len` taps a run: below this the gather is
    // the cheaper answer, and saying so here is what keeps a stepped element off
    // a path that would only slow it down.
    if offsets.len() < 4 * runs.len() {
        return None;
    }
    let bound = |axis: usize, extent: usize| {
        let low = (-lo[axis]).max(0) as usize;
        let high = (extent as isize - hi[axis]).max(0) as usize;
        (low < high).then_some([low, high])
    };
    let inside = [bound(0, n0)?, bound(1, n1)?, bound(2, n2)?];
    // The destination as one run of memory, or nothing. Writing through
    // `out[[i, j, k]]` would put `ndarray`'s index arithmetic back into the loop
    // this exists to take it out of, and a caller whose output is not one run is
    // rare enough to keep the gather.
    let out = out.as_slice_mut()?;
    let [[i0, i1], [j0, j1], [k0, k1]] = inside;
    let width = k1 - k0;
    let plane = n1 * n2;
    // **A cache of running planes, keyed by the plane and the window.**
    //
    // The straightforward loop recomputes one running pass per run per output
    // plane — ten passes over a plane to produce one, and the scratch traffic
    // that generates dominates everything else the kernel does. It does not have
    // to: run `g` reads source plane `i + d0_g` with window `L_g`, so advancing
    // the output plane by one asks for `(i + 1 + d0_g, L_g)`, and the previous
    // output plane computed `(i + d0_{g'}, L_{g'})` for every `g'`. Those meet
    // whenever two rows one apart have the **same length** — which, for a disk,
    // most of the middle rows do.
    //
    // So the passes that are repeated are skipped and the rest are not, which is
    // the whole of it. One slot per run is always enough: a round asks for at
    // most that many keys, so a slot not yet used this round is always free.
    let slots = runs.len();
    let mut keys: Vec<Option<(usize, usize)>> = vec![None; slots];
    let mut used: Vec<bool> = vec![false; slots];
    let mut cache: Vec<Vec<T>> = vec![vec![values[0]; plane]; slots];
    let mut forward: Vec<T> = vec![values[0]; plane];
    for i in i0..i1 {
        used.iter_mut().for_each(|flag| *flag = false);
        for (ordinal, run) in runs.iter().enumerate() {
            let source_plane = (i as isize + run.d0) as usize;
            let key = (source_plane, run.len);
            let slot = match keys.iter().position(|held| *held == Some(key)) {
                Some(slot) => slot,
                None => {
                    let slot = used
                        .iter()
                        .position(|flag| !flag)
                        .expect("one slot per run leaves one unused every round");
                    let base = source_plane * plane;
                    running_extreme_strided(
                        &values[base..base + plane],
                        n1,
                        n2,
                        run.len,
                        take,
                        &mut forward,
                        &mut cache[slot],
                    );
                    keys[slot] = Some(key);
                    slot
                }
            };
            used[slot] = true;
            for j in j0..j1 {
                let source =
                    ((j as isize + run.start) as usize) * n2 + (k0 as isize + run.d2) as usize;
                let target = (i * n1 + j) * n2 + k0;
                if ordinal == 0 {
                    out[target..target + width]
                        .copy_from_slice(&cache[slot][source..source + width]);
                } else {
                    fold_into(
                        &mut out[target..target + width],
                        &cache[slot][source..source + width],
                        take,
                    );
                }
            }
        }
    }
    Some(inside)
}

/// The experiment's kernel: [`selecting`] with either change, both, or neither.
///
/// Byte for byte the same answer as [`selecting`] on every arm — asserted, not
/// intended, because a rank filter selects an existing value and so agreement is
/// exact rather than approximate.
///
/// The run path applies only where the offsets are one fixed list — an element
/// whose step counts from the clipped start re-phases per voxel, so its runs
/// cannot be precomputed and it falls through to the gather. Masked windows take
/// the runs too, but pay a mask test per voxel inside them, so the run's saving
/// there is the index arithmetic and not the branch.
#[allow(clippy::too_many_arguments)]
pub fn selecting_by<T: Ord + Copy>(
    input: ArrayView3<'_, T>,
    at: &Anchor,
    mask: Option<ArrayView3<'_, bool>>,
    element: &StructuringElement,
    rank: Rank,
    centre: ExcludedCentre<T>,
    mut out: ArrayViewMut3<'_, T>,
    path: RankPath,
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
    let fixed = (element.origin() == StepOrigin::Anchor).then(|| element.offsets());
    let runs = match (path.runs(), fixed) {
        (true, Some(offsets)) => Some(runs_of(offsets)),
        _ => None,
    };
    let ends = path.extreme().then(|| extreme_of(rank, full)).flatten();
    let mut window: Vec<T> = Vec::with_capacity(full);
    let mut offsets: Vec<[isize; 3]> = Vec::new();

    for i in 0..input.shape()[0] {
        for j in 0..input.shape()[1] {
            for k in 0..input.shape()[2] {
                if let (Some(mask), ExcludedCentre::Fill(value)) = (mask.as_ref(), centre) {
                    if !mask[[i, j, k]] {
                        out[[i, j, k]] = value;
                        continue;
                    }
                }
                let anchor = [i as isize, j as isize, k as isize];
                // The two reductions are written as one walk each so that a
                // voxel pays for exactly one of them.
                let mut best: Option<T> = None;
                window.clear();
                let mut take = |value: T, window: &mut Vec<T>| match ends {
                    Some(Extreme::Lowest) => {
                        best = Some(match best {
                            Some(seen) if seen <= value => seen,
                            _ => value,
                        })
                    }
                    Some(Extreme::Highest) => {
                        best = Some(match best {
                            Some(seen) if seen >= value => seen,
                            _ => value,
                        })
                    }
                    None => window.push(value),
                };
                match &runs {
                    Some(runs) => {
                        for &(o0, o1, o2, length) in runs {
                            let a = anchor[0] + o0;
                            let b = anchor[1] + o1;
                            if a < 0 || b < 0 || a >= extent[0] || b >= extent[1] {
                                continue;
                            }
                            let low = (anchor[2] + o2).max(0);
                            let high = (anchor[2] + o2 + length as isize).min(extent[2]);
                            if low >= high {
                                continue;
                            }
                            let (a, b) = (a as usize, b as usize);
                            for c in low as usize..high as usize {
                                if let Some(mask) = mask.as_ref() {
                                    if !mask[[a, b, c]] {
                                        continue;
                                    }
                                }
                                take(input[[a, b, c]], &mut window);
                            }
                        }
                    }
                    None => {
                        let gathered = match fixed {
                            Some(offsets) => offsets,
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
                            if a < 0
                                || b < 0
                                || c < 0
                                || a >= extent[0]
                                || b >= extent[1]
                                || c >= extent[2]
                            {
                                continue;
                            }
                            let at = [a as usize, b as usize, c as usize];
                            if let Some(mask) = mask.as_ref() {
                                if !mask[at] {
                                    continue;
                                }
                            }
                            take(input[at], &mut window);
                        }
                    }
                }
                let chosen = match ends {
                    Some(_) => best,
                    None => {
                        let index = rank.resolve(full, window.len());
                        select_nth(&mut window, index)
                    }
                };
                match chosen {
                    Some(value) => out[[i, j, k]] = value,
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

/// Time the incumbent kernel against the four experimental arms, interleaved.
///
/// **The first row of every block is an A/A control.** `RankPath::Gather` is the
/// candidate function reproducing the incumbent's own path, so it should cost
/// what `selecting` costs. If it does not, the two are not comparable and no
/// ratio below means anything — which is the failure the previous attempt at
/// this comparison had and could not see, because its two arms were in
/// different crates.
///
/// Arms are interleaved **in a randomised order**: every arm is timed once
/// before any arm is timed twice, and the order within a repeat is shuffled. The
/// reported figure is the best of `repeats`, and nothing here asserts on an
/// absolute time.
///
/// The `[noise floor]` row is the harness's own error bar, measured rather than
/// assumed: where the rank is not an extreme, `extreme_of` declines and the two
/// `*Extreme` arms execute exactly the code of their twins, so the gap between
/// them is two identical programs disagreeing.
///
/// What it measured, and what it does **not** support
/// -------------------------------------------------
/// ```text
/// 64 x 96 x 96 u16, best of 9, randomised order, machine at ~2x its core count in load
/// element             taps  rank    incumbent  A/A gather  extreme   runs  runs+extreme  noise
/// disk 11x11 (2-D)      81  lowest      1.000       1.025    1.073  0.690         0.462      -
/// disk 11x11 (2-D)      81  median      1.000       1.107    1.111  0.925         0.820  0.114
/// ball 5x5x5            33  lowest      1.000       1.498    0.997  1.015         0.760      -
/// ball 5x5x5            33  median      1.000       1.219    1.299  1.518         1.459  0.065
/// stepped box           25  lowest      1.000       1.041    1.023  1.046         1.037      -
/// stepped box           25  median      1.000       1.057    1.141  1.248         1.219  0.079
/// ```
///
/// **The A/A control fails, and that bounds everything else.** `A/A gather` is
/// the candidate reproducing the incumbent's own path and should read `1.000`;
/// it reads `1.025` to **`1.498`**. So the candidate carries a structural
/// penalty — the runtime dispatch between arms and the closure that feeds
/// either reduction — that a shipped, specialised version would not have, and
/// no ratio here is a clean measurement of either change.
///
/// **The noise floor is `6.5%`-`11.4%`**, and against it most of the table is
/// nothing. Worse, a first run of this report at `repeats = 5` and a *fixed* arm
/// order **reversed the sign** of the `ball / median / runs+extreme` cell —
/// `0.669` then `1.459` — which is two runs of the same code disagreeing about
/// its direction. Randomising the order fixed the mechanism that was suspected
/// (the last arm always read the warmest input) and did not make the cell
/// reproducible.
///
/// **One cell reproduces and it is the one the consumer's chain uses.** The
/// `11x11` disk at `Rank::lowest` — the erosion inside a Disk opening — came out
/// at `0.515` and `0.462` of the incumbent across two independent runs, both far
/// outside the noise floor, and `runs` alone at `0.730` and `0.690`. So **run
/// decomposition is worth about `1.4x` on a large element and the two changes
/// together about `2.1x`**, and the extreme path alone is worth nothing at all
/// (`1.073`, `0.997`, `1.023`).
///
/// **That is not the `10x` the reading predicted, and nothing is shipped on it.**
/// The diagnosis that motivated this experiment reasoned from a `12.4x` chain
/// gap to a per-voxel cost scaling with the element's population; the experiment
/// says the kernel has about `2.1x` in it on the best case and nothing on the
/// others. A `2.1x` on one arm of one chain is worth having, but claiming it
/// needs three things this measurement does not have: a specialised
/// implementation with no dispatch, its own A/A control passing, and a machine
/// that is not at twice its core count. **Both arms stay here as the fixture for
/// that experiment** — which is what a hypothesis filed for measurement is for —
/// and `RankFilterOp` still calls [`selecting`].
///
/// **The stepped row is the honest negative and it behaved as predicted.** An
/// element whose step decimates the fastest axis has runs of length one, so the
/// decomposition costs a little and buys nothing: `1.046` and `1.248`.
pub fn cost_report(shape: [usize; 3], repeats: usize) -> String {
    use std::time::Instant;

    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let input = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 48) as u16
    });
    let at = crate::op::Anchor::whole(shape);
    let mut out = Array3::<u16>::zeros((shape[0], shape[1], shape[2]));

    let disk = StructuringElement::from_radius(super::element::ElementShape::Ellipsoid, [0, 5, 5]);
    let ball = StructuringElement::from_radius(super::element::ElementShape::Ellipsoid, [2, 2, 2]);
    let stepped = StructuringElement::from_sides_stepped(
        super::element::ElementShape::Box,
        [0, 4, 4],
        [0, 4, 4],
        [1, 2, 2],
    )
    .expect("a stepped element");

    let mut report = String::from(
        "element                 taps  rank      arm            ns/voxel   x incumbent\n",
    );
    for (name, element) in [
        ("disk 11x11 (2-D)", &disk),
        ("ball 5x5x5", &ball),
        ("stepped box", &stepped),
    ] {
        for (rank_name, rank) in [
            ("lowest", Rank::lowest()),
            ("median", Rank::median(element)),
        ] {
            let arms: [(&str, Option<RankPath>); 5] = [
                ("incumbent", None),
                ("A/A gather", Some(RankPath::Gather)),
                ("extreme", Some(RankPath::GatherExtreme)),
                ("runs", Some(RankPath::Runs)),
                ("runs+extreme", Some(RankPath::RunsExtreme)),
            ];
            let mut best = [f64::INFINITY; 5];
            // **Arm order is randomised per repeat, not fixed.** A fixed order
            // gives every arm a different cache history — the last arm always
            // reads an input four passes warm — and the first run of this report
            // showed two arms that execute *identical* code differing by 39%,
            // which is that confound and not a result. A cheap linear
            // congruential shuffle costs nothing and removes it.
            let mut seed = 0x2545_F491_4F6C_DD1Du64;
            for _ in 0..repeats.max(1) {
                let mut order = [0usize, 1, 2, 3, 4];
                for slot in (1..order.len()).rev() {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    order.swap(slot, (seed >> 33) as usize % (slot + 1));
                }
                for index in order {
                    let (_, path) = &arms[index];
                    let started = Instant::now();
                    match path {
                        None => selecting(
                            input.view(),
                            &at,
                            None,
                            element,
                            rank,
                            ExcludedCentre::Select,
                            out.view_mut(),
                            "report",
                        ),
                        Some(path) => selecting_by(
                            input.view(),
                            &at,
                            None,
                            element,
                            rank,
                            ExcludedCentre::Select,
                            out.view_mut(),
                            *path,
                            "report",
                        ),
                    }
                    .expect("a pass");
                    let elapsed = started.elapsed().as_secs_f64() * 1e9 / voxels;
                    if elapsed.total_cmp(&best[index]).is_lt() {
                        best[index] = elapsed;
                    }
                }
            }
            // **The noise floor, measured rather than assumed.** Where the rank
            // is not an extreme, `extreme_of` declines and the two `*Extreme`
            // arms execute exactly the code of their non-extreme twins. The gap
            // between two identical programs is this harness's own error bar,
            // and no ratio below it means anything.
            let twins = extreme_of(rank, element.len()).is_none();
            if twins {
                report.push_str(&format!(
                    "{name:22}  {:4}  {rank_name:8}  {:13}  {:8.2}   {:9.3}\n",
                    element.len(),
                    "[noise floor]",
                    0.0,
                    ((best[2] / best[1]) - 1.0)
                        .abs()
                        .max(((best[4] / best[3]) - 1.0).abs()),
                ));
            }
            for (index, (arm, _)) in arms.iter().enumerate() {
                report.push_str(&format!(
                    "{name:22}  {:4}  {rank_name:8}  {arm:13}  {:8.2}   {:9.3}\n",
                    element.len(),
                    best[index],
                    best[index] / best[0],
                ));
            }
        }
    }
    report
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
    /// **A stencil, and the family it belongs to is not** — which is why this
    /// declaration is on the op and not on the family.
    ///
    /// `ops::rank`'s kernel gathers a window per voxel and selects from it; it
    /// carries nothing from one voxel to the next, so where the scan starts
    /// cannot be seen in the answer. `ops::sliding` computes the same statistic
    /// by carrying a histogram along the scan with joining and leaving sets, and
    /// that one is **not** a stencil however similar its reach looks. A reach
    /// says what an op reads; it does not say the answer is a function only of
    /// what was read.
    ///
    /// Held to it by `tests/intra_block_slicing.rs`.
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

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

/// [`RankFilterOp`], with a stored image deciding which voxels count.
///
/// **The case it exists for** is a percentile taken over a window with
/// background voxels removed from the population, so that the statistic
/// describes the structure present rather than the proportion of the window it
/// occupies. It is a `BlockOp` like any other; what is new is that it reads a
/// *second* image over the same window it reads its input over, which is what
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
    /// `mask` is the image holding the population, which must be a `Bool`
    /// image — checked when the plan is made, by the same `check_dtypes` fold
    /// that checks every other image.
    pub fn new(
        name: &'static str,
        element: StructuringElement,
        rank: Rank,
        mask: impl Into<crate::assemble::ImageId>,
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
    /// The value is stated in `f64` and converted to the image's element type by
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
        mask: impl Into<crate::assemble::ImageId>,
    ) -> Result<Self> {
        let rank = Rank::ceiling_percentile(fraction)?;
        Ok(Self::new(name, element, rank, mask))
    }

    pub fn element(&self) -> &StructuringElement {
        &self.element
    }

    /// The image this op reads its population from.
    pub fn mask_image(&self) -> usize {
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
    /// inside what a plan can currently fetch — see `check_source_images`.
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
            "{}: the population comes from image {}, so this op has no answer from its input \
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
                "{}: the population is read from image {}, which holds {}. A population is a \
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
    /// image the short circuit has not read.
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

    /// The running kernel against the definition, on shapes that reach it.
    ///
    /// `the_filter_agrees_with_the_definition_for_every_rank_and_shape` cannot:
    /// its input is `9 x 7 x 6` and a disk of any size fills it, so there is no
    /// interior and the running path declines every time. These inputs are
    /// large enough to have one, and the radii are chosen so that the runs have
    /// **several distinct lengths** — an ellipsoid's rows are `5, 7, 9, 9, 9, 7,
    /// 5` and a box's are all equal, which are the two ways the length table can
    /// go wrong.
    ///
    /// Both ends, because the kernel folds a min and a max through one `pick`,
    /// and the odd/even window sizes because van Herk's blocks are `len` long
    /// and a window that divides the row evenly is the case where its prefix and
    /// suffix coincide.
    #[test]
    fn the_running_extremum_agrees_with_the_definition() {
        let input = ramp((4, 21, 23));
        for shape in [ElementShape::Ellipsoid, ElementShape::Box] {
            for radius in [
                [0, 4, 4],
                [0, 3, 5],
                [0, 1, 6],
                [0, 5, 1],
                [0, 0, 4],
                [0, 2, 2],
            ] {
                let element = StructuringElement::from_radius(shape, radius);
                for rank in [Rank::lowest(), Rank::highest(&element)] {
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

    /// The separable box path, against the definition.
    ///
    /// Only boxes of at least sixty-four taps reach it, so the radii here are
    /// chosen to clear that: `[2, 2, 2]` is the `5 x 5 x 5` window a local
    /// maximum test uses and the reason the path exists. The uneven ones are
    /// where a per-axis width could be applied to the wrong axis and a symmetric
    /// cube would never show it.
    ///
    /// `ExtentEllipsoid` at the same sizes is included deliberately: it is *not*
    /// a box, so it must take the run decomposition instead, and an element that
    /// slipped through `box_widths` would answer a rectangle where a disk was
    /// asked for.
    #[test]
    fn the_separable_box_agrees_with_the_definition() {
        let input = ramp((17, 19, 21));
        for radius in [
            [2, 2, 2],
            [2, 2, 1],
            [1, 3, 2],
            [3, 1, 2],
            [2, 1, 3],
            [4, 4, 4],
        ] {
            for shape in [ElementShape::Box, ElementShape::Ellipsoid] {
                let element = StructuringElement::from_radius(shape, radius);
                for rank in [Rank::lowest(), Rank::highest(&element)] {
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
        // An even extent, where the element is not symmetric about its anchor and
        // the shift the composed volume needs is not `w / 2`.
        for size in [[6, 6, 6], [4, 8, 5], [8, 4, 5]] {
            let element =
                StructuringElement::from_size(ElementShape::Box, size).expect("a stated size");
            for rank in [Rank::lowest(), Rank::highest(&element)] {
                let mut got = Array3::zeros(input.dim());
                rank_filter_f64_into(input.view(), &element, rank, got.view_mut()).unwrap();
                assert_eq!(
                    got,
                    by_definition(&input, &element, rank),
                    "{size:?} {rank:?}"
                );
            }
        }
    }

    /// A **deeper** element still agrees, by declining the running path.
    ///
    /// The guard is `run.0 != 0`, and a filter that silently took the planar
    /// path for a depth-3 element would answer a different question, so this
    /// asserts the decline rather than trusting it.
    /// The shipped background element's own **orientation**, against the
    /// definition.
    ///
    /// `size = [10, 10, 1]` states a disk whose flat axis is the *fastest* one,
    /// which is the orientation the first version of the running path could not
    /// decompose and silently declined. An assertion that only ever saw disks in
    /// the last two axes would not have noticed, so this states the shape the
    /// caller actually uses — even extents included, which is where the element
    /// is not symmetric about its own centre.
    #[test]
    fn the_shipped_orientation_agrees_with_the_definition() {
        let input = ramp((23, 21, 19));
        for size in [
            [10, 10, 1],
            [9, 9, 1],
            [6, 10, 1],
            [10, 1, 6],
            [1, 10, 10],
            [5, 5, 5],
        ] {
            let element = StructuringElement::from_size(ElementShape::ExtentEllipsoid, size)
                .expect("a stated size");
            for rank in [Rank::lowest(), Rank::highest(&element)] {
                let mut got = Array3::zeros(input.dim());
                rank_filter_f64_into(input.view(), &element, rank, got.view_mut()).unwrap();
                assert_eq!(
                    got,
                    by_definition(&input, &element, rank),
                    "{size:?} {rank:?}"
                );
            }
        }
    }

    #[test]
    fn an_element_with_depth_agrees_too() {
        let input = ramp((11, 13, 17));
        for radius in [[1, 2, 2], [2, 0, 3], [3, 3, 3]] {
            let element = StructuringElement::from_radius(ElementShape::Ellipsoid, radius);
            for rank in [Rank::lowest(), Rank::highest(&element)] {
                let mut got = Array3::zeros(input.dim());
                rank_filter_f64_into(input.view(), &element, rank, got.view_mut()).unwrap();
                assert_eq!(
                    got,
                    by_definition(&input, &element, rank),
                    "{radius:?} {rank:?}"
                );
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

    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn the_cost_of_the_four_arms() {
        println!("{}", cost_report([64, 96, 96], 9));
    }

    /// **Every arm of the experiment is the incumbent, byte for byte.**
    ///
    /// A rank filter selects a value it read, so agreement here is exact rather
    /// than to a tolerance, and anything less is a different filter. The sweep
    /// covers the two things the run decomposition and the extreme path can each
    /// get wrong: an element whose runs are long (a disk), one whose runs are all
    /// of length one (a stepped box, where the decomposition buys nothing and
    /// must still be correct), a masked window, and a rank that is *not* an
    /// extreme so the fast path must decline to take itself.
    #[test]
    fn every_arm_of_the_experiment_answers_what_the_incumbent_answers() {
        use crate::ops::element::ElementShape;

        let shape = [7usize, 11, 13];
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let input = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 55) as u16
        });
        let mask = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(a, b, c)| {
            (a * 7 + b * 3 + c) % 5 != 0
        });
        let at = crate::op::Anchor::whole(shape);
        let elements = [
            StructuringElement::from_radius(ElementShape::Ellipsoid, [1, 3, 3]),
            StructuringElement::from_radius(ElementShape::Box, [1, 2, 2]),
            StructuringElement::from_sides_stepped(
                ElementShape::Box,
                [0, 2, 2],
                [0, 2, 2],
                [1, 2, 2],
            )
            .expect("a stepped element"),
        ];
        let mut compared = 0usize;
        let mut moved = 0usize;
        for element in &elements {
            let ranks = [
                Rank::lowest(),
                Rank::highest(element),
                Rank::median(element),
                Rank::Nth(1),
            ];
            for rank in ranks {
                for masked in [false, true] {
                    let view = masked.then(|| mask.view());
                    let mut expected = Array3::<u16>::zeros((shape[0], shape[1], shape[2]));
                    selecting(
                        input.view(),
                        &at,
                        view,
                        element,
                        rank,
                        ExcludedCentre::Select,
                        expected.view_mut(),
                        "expected",
                    )
                    .expect("the incumbent runs");
                    for path in [
                        RankPath::Gather,
                        RankPath::GatherExtreme,
                        RankPath::Runs,
                        RankPath::RunsExtreme,
                    ] {
                        let mut got = Array3::<u16>::zeros((shape[0], shape[1], shape[2]));
                        selecting_by(
                            input.view(),
                            &at,
                            view,
                            element,
                            rank,
                            ExcludedCentre::Select,
                            got.view_mut(),
                            path,
                            "candidate",
                        )
                        .expect("the candidate runs");
                        assert_eq!(got, expected, "{path:?} disagrees at rank {rank:?}");
                        compared += 1;
                    }
                    // **Liveness.** Equality proves nothing if every arm writes
                    // the same trivial volume, so the fixture must actually be
                    // filtered: the answer has to differ from the input.
                    moved += input
                        .iter()
                        .zip(expected.iter())
                        .filter(|(a, b)| a != b)
                        .count();
                }
            }
        }
        assert!(
            compared >= 90,
            "only {compared} comparisons, which is not the sweep this claims"
        );
        assert!(
            moved > shape.iter().product::<usize>(),
            "the filters moved only {moved} voxels in total, so the fixtures are not being filtered"
        );
    }

    /// **The extreme fast path takes itself exactly where `Rank::resolve` says
    /// it may, and nowhere else.**
    ///
    /// `extreme_of` decides once per call what `resolve` decides per voxel, so
    /// the two are one quantity stated twice — checked here against `resolve`
    /// itself over every window size, rather than against the comment that
    /// derives it.
    #[test]
    fn the_extreme_fast_path_agrees_with_selection_everywhere() {
        use crate::ops::element::Percentile;

        for full in [1usize, 2, 7, 40, 121] {
            for (rank, expected) in [
                (Rank::Nth(0), Some(Extreme::Lowest)),
                (
                    Rank::Nth(full.saturating_sub(1)),
                    if full <= 1 {
                        Some(Extreme::Lowest)
                    } else {
                        Some(Extreme::Highest)
                    },
                ),
                (
                    Rank::CeilingPercentile(Percentile::new(0.0).unwrap()),
                    Some(Extreme::Lowest),
                ),
                (
                    Rank::CeilingPercentile(Percentile::new(1.0).unwrap()),
                    Some(Extreme::Highest),
                ),
            ] {
                let taken = extreme_of(rank, full);
                assert_eq!(taken, expected, "full {full}, rank {rank:?}");
                // And it is right: for every window size the resolved index is
                // the end it claims.
                for available in 1..=full.max(1) {
                    let index = rank.resolve(full, available);
                    match expected {
                        Some(Extreme::Lowest) => {
                            assert_eq!(index, 0, "full {full} available {available}")
                        }
                        Some(Extreme::Highest) => {
                            assert_eq!(index, available - 1, "full {full} available {available}")
                        }
                        None => {}
                    }
                }
            }
            // **The declining half, which is the one a fast path gets wrong.** A
            // middle rank is not an extreme at any size above two, and must not
            // be taken.
            if full > 3 {
                let middle = Rank::Nth(full / 2);
                assert_eq!(extreme_of(middle, full), None, "a median is not an extreme");
                let index = middle.resolve(full, full);
                assert!(
                    index > 0 && index < full - 1,
                    "and its index is interior: {index}"
                );
            }
        }
    }
}
