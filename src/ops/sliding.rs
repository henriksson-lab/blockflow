// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the traversal,
// not adapted from any implementation of it.
//
// A windowed statistic taken from a histogram that is **carried along a scan
// line** rather than rebuilt at every voxel.
//
// What the cost actually is
// -------------------------
// `ops::rank` gathers the whole element at every voxel: `|element|` reads and a
// selection over `|element|` values, per voxel, and nothing about the voxel
// before it is reused. That is the honest way to state the filter and it is what
// every other property of that module is checked against — but the window at
// `v + 1` and the window at `v` differ by a handful of voxels, and re-reading
// the other tens of thousands is the whole of the gap. For a `200 x 200 x 1`
// element the two windows share 39 800 of their 40 000 members.
//
// So the state that slides is a **histogram over the values under the window**,
// with a running count of how many voxels are actually in it. Moving the centre
// one step along a scan axis retires the offsets the step drops and admits the
// ones it gains; the histogram is otherwise untouched, and the statistic is
// whatever a query makes of the counts.
//
// The step rule, and why it needs no edge cases
// ---------------------------------------------
// Moving the centre from `v` to `v + 1` along axis `s` takes the window from
// `v + E` to `v + 1 + E`. As position sets those differ by exactly
//
// * **the leavers** `L = { o in E : o - e_s not in E }`, read at the *old*
//   centre, and
// * **the joiners** `J = { o in E : o + e_s not in E }`, read at the *new* one.
//
// `|L| = |J|` always, both being `|E|` minus the overlap, and both are computed
// once for the element rather than once per voxel.
//
// **Every candidate is bounds-checked and mask-checked at its own absolute
// position**, on the way in and on the way out alike. That is the whole reason
// there is no special case for the volume's faces: an offset that fell outside
// the array was never admitted, so the same test declines to retire it, and the
// running population is by construction the number of voxels that are really
// there. Truncation at a face and exclusion by a mask are the same mechanism,
// and the population a query is handed is the truncated, masked one — which is
// exactly what `Rank::resolve` is written against.
//
// Each scan line is independent: the histogram is primed by one full gather at
// the line's first voxel and emptied again at its last, so a line cannot inherit
// a neighbour's state. Emptying is done by whichever of two routes is shorter —
// running the step rule backwards over the last window, at `|element|`, or
// zeroing the counts, at `|domain|` — and a debug build always takes the first
// and asserts that it came back to zero exactly, which is the standing proof
// that admission and retirement agree about what is in the window.
//
// The bounded-domain constraint is stated, not worked around
// ----------------------------------------------------------
// A histogram needs somewhere to put a value, so this traversal applies to
// element types whose values index bins directly — see [`BinnedElement`] — or to
// a [`Domain`] a caller states. **Floating point is refused by name.** Binning
// an `f64` would answer a different question from the one `ops::rank` answers,
// silently and at every voxel, and a filter that is "the same but for the bins"
// is not the same filter. `SlidingHistogramOp::accepts` says no to both floats
// and to the signed integers, whose values do not index anything from zero.

use std::collections::HashSet;

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, SourceInput, SourceInputs};
use crate::reach::Reach;
use crate::voxels::{VoxelElement, Voxels};

use super::element::{Rank, StructuringElement};
use super::rank::ExcludedCentre;
use super::shapes_agree;

/// An element type whose values index histogram bins directly.
///
/// The bound is the honest one: a histogram is an array indexed by the value, so
/// the value has to be a non-negative integer with a stated ceiling. `bool`,
/// `u8` and `u16` carry their own; `u32` does not — 2^32 bins is not an array —
/// so it is implemented here with no natural domain and is usable only where a
/// caller states one.
///
/// **There is deliberately no implementation for `f32` or `f64`.** See the
/// module header: binning a float changes the answer, so the refusal is a
/// missing implementation rather than a runtime conversion.
pub trait BinnedElement: VoxelElement + Ord {
    /// How many bins cover every value this type can hold, where that is a
    /// number an array can be allocated at.
    const NATURAL_DOMAIN: Option<usize>;

    /// Which bin `self` counts in.
    fn bin(self) -> usize;
}

impl BinnedElement for bool {
    const NATURAL_DOMAIN: Option<usize> = Some(2);

    fn bin(self) -> usize {
        usize::from(self)
    }
}

impl BinnedElement for u8 {
    const NATURAL_DOMAIN: Option<usize> = Some(1 << 8);

    fn bin(self) -> usize {
        self as usize
    }
}

impl BinnedElement for u16 {
    const NATURAL_DOMAIN: Option<usize> = Some(1 << 16);

    fn bin(self) -> usize {
        self as usize
    }
}

impl BinnedElement for u32 {
    /// None: a `u32` spans 2^32 values and no histogram is allocated that wide.
    /// A caller with `u32` data inside a known range says so with
    /// [`Domain::of_size`], and a value outside it is an error naming the value.
    const NATURAL_DOMAIN: Option<usize> = None;

    fn bin(self) -> usize {
        self as usize
    }
}

/// How many bins a histogram is taken over.
///
/// A separate type rather than a bare `usize` because it is the one parameter
/// that decides whether the traversal is even applicable, and because the two
/// ways of arriving at it — the element type's own range, or a range the caller
/// knows about their data — deserve to be two named constructors rather than a
/// number whose provenance is lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Domain {
    bins: usize,
}

impl Domain {
    /// The widest histogram this module will allocate, per scan.
    ///
    /// A ceiling rather than a tuning knob: `u16`'s own domain is 65 536 and has
    /// to fit, and something has to refuse a caller who states `usize::MAX`
    /// before the allocation does it less politely.
    pub const LARGEST: usize = 1 << 20;

    /// A stated number of bins. Values `0 .. bins` are countable and anything at
    /// or above `bins` is an error naming the value when the scan meets it.
    pub fn of_size(bins: usize) -> Result<Self> {
        if bins == 0 {
            return Err(Error::InvalidArgument(
                "a histogram over zero bins counts nothing".to_string(),
            ));
        }
        if bins > Self::LARGEST {
            return Err(Error::InvalidArgument(format!(
                "a histogram of {bins} bins is past the {} this module allocates. A domain that \
                 wide is a sign the values want re-scaling before they are counted, which is a \
                 decision for the caller rather than a silent one here.",
                Self::LARGEST
            )));
        }
        Ok(Self { bins })
    }

    /// The element type's own range, where it has one an array can be allocated
    /// at.
    ///
    /// Errors, naming the type, where it does not — which is the refusal the
    /// module header promises rather than a quiet fallback to some default width.
    pub fn of<T: BinnedElement>() -> Result<Self> {
        match T::NATURAL_DOMAIN {
            Some(bins) => Self::of_size(bins),
            None => Err(Error::InvalidArgument(format!(
                "{} has no bounded histogram domain of its own, so one has to be stated: \
                 `Domain::of_size(bins)`, with every value below `bins`.",
                T::DTYPE.numpy_name()
            ))),
        }
    }

    pub fn bins(&self) -> usize {
        self.bins
    }
}

/// What a sliding traversal asks of the counts under one window.
///
/// Object-safe on purpose, and stated over the counts rather than over the
/// values: everything the traversal maintains is in the two arguments, so a
/// query has no state of its own and cannot be told apart from a query that
/// re-gathered the window itself. That is what makes byte-identity with the
/// dense path a property of the *traversal* rather than of each query.
///
/// `evaluate` is **never called with `population == 0`**. An empty window has no
/// value to hand back — see [`sliding_histogram_into`] for what is written
/// there — and folding that case into every query would put the same three lines
/// in each of them.
pub trait HistogramQuery: Send + Sync {
    /// What this query is, for the error a scan raises and for an op that takes
    /// its name from the question rather than the caller.
    fn name(&self) -> &'static str;

    /// The statistic of a window in which `counts[value]` voxels hold `value`
    /// and `population` voxels are present at all.
    ///
    /// `population` is `counts.iter().sum()`, carried rather than recomputed;
    /// it is the count that *survived* truncation and masking, which is the
    /// count an order statistic is resolved against.
    fn evaluate(&self, counts: &[u32], population: usize) -> f64;

    /// What one [`Self::evaluate`] costs over a `domain`-wide histogram, in the
    /// units `BlockOp::cost_per_voxel` is stated in.
    fn cost_per_evaluation(&self, domain: usize) -> f64;

    /// What this query answers on a window every voxel of which holds `value` —
    /// `None` where that is not exactly known.
    ///
    /// The same contract as `BlockOp::constant_maps_to`, and it is here because
    /// it is the query rather than the traversal that decides: the traversal
    /// carries counts, and whether a pile of counts all in one bin has an exact
    /// answer is the question's business.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        None
    }
}

/// The order statistic of the window, as [`super::rank`] defines it.
///
/// The first [`HistogramQuery`], and the one the whole traversal exists for. It
/// is deliberately *not* a second definition of a rank filter: it holds a
/// [`Rank`] and the element's full population and resolves them with
/// [`Rank::resolve`], the same call the dense kernel makes, so the two cannot
/// drift apart on the truncation rule.
#[derive(Debug, Clone, Copy)]
pub struct RankQuery {
    rank: Rank,
    full: usize,
}

impl RankQuery {
    /// `rank` of `element`. The element is taken whole rather than as a count so
    /// that a caller cannot pair a rank with the wrong population.
    pub fn new(rank: Rank, element: &StructuringElement) -> Self {
        Self {
            rank,
            full: element.len(),
        }
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }
}

impl HistogramQuery for RankQuery {
    fn name(&self) -> &'static str {
        "rank"
    }

    fn evaluate(&self, counts: &[u32], population: usize) -> f64 {
        let index = self.rank.resolve(self.full, population);
        // `resolve` never returns an index past `population - 1`, and the counts
        // sum to `population`, so the walk always lands.
        first_bin_past(counts, index) as f64
    }

    fn cost_per_evaluation(&self, domain: usize) -> f64 {
        RANK_SCAN_COST_PER_BIN * domain as f64
    }

    /// Exactly the constant, for every rank and at every truncation — the same
    /// declaration `RankFilterOp` makes and for the same reason. Every count sits
    /// in the constant's own bin, so whichever index the rank resolves to, the
    /// bin the cumulative sum first passes it in is that one.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }
}

/// Bins summed as one block by [`first_bin_past`].
///
/// The scan is a chain of dependent loads if it is written a bin at a time, and
/// the answer for a percentile over a wide domain is typically a long way down
/// it. A block sum is independent adds the compiler can vectorise, and skipping
/// a block that cannot contain the answer is **exact rather than approximate**:
/// counts are non-negative, so the cumulative sum never decreases, and a block
/// whose running total has not passed the target holds no bin that has. The
/// comparison that ends the scan is the same one a bin-at-a-time walk would
/// have made, so this is a faster route to the identical bin and not a heuristic.
///
/// **Measured**, over 32 / 64 / 128 with a 4096-bin domain and both a 150-voxel
/// line and a 709-voxel disc: 32 and 64 are within noise of each other and 128
/// is worse, because a wider block does more work before it can stop. 32 is four
/// cache lines and is where this sits.
const SCAN_BLOCK: usize = 32;

/// The smallest bin whose cumulative count is strictly greater than `index` —
/// that is, the `index`-th smallest value under the window, counting from zero.
///
/// Ties need no thought: every voxel holding the winning value is counted in the
/// same bin, so which of them "is" the answer is not a question the histogram
/// can be asked.
fn first_bin_past(counts: &[u32], index: usize) -> usize {
    // `u32` throughout, and not for tidiness: the counts sum to the window's
    // population, which is at most the element's size, so nothing here can
    // overflow — and a 32-bit accumulator packs twice as many lanes into a
    // vector register as a 64-bit one, which is the whole reason the block sum
    // is worth writing out.
    let target = index as u32;
    let mut cumulative = 0u32;
    let mut base = 0usize;
    let mut blocks = counts.chunks_exact(SCAN_BLOCK);
    for block in &mut blocks {
        // The `try_into` is what makes this worth writing: over a `&[u32]` of
        // unknown length the sum is a dynamic loop, and over a `&[u32; 32]` it
        // is a fixed-length integer reduction the compiler will vectorise
        // outright. Integer addition associates, so no manual accumulator
        // splitting is needed once the length is a constant — which is the
        // opposite of the floating-point case, where reassociation would change
        // the answer and the splitting has to be written by hand.
        let block: &[u32; SCAN_BLOCK] = block
            .try_into()
            .expect("chunks_exact yields blocks of exactly its own width");
        let total: u32 = block.iter().sum();
        if cumulative + total > target {
            return base + refine(block, cumulative, target);
        }
        cumulative += total;
        base += SCAN_BLOCK;
    }
    let tail = blocks.remainder();
    if !tail.is_empty() && cumulative + tail.iter().sum::<u32>() > target {
        return base + refine(tail, cumulative, target);
    }
    // Unreachable while `index < population`, which the caller guarantees. The
    // last bin is the least wrong answer if it ever is reached.
    counts.len().saturating_sub(1)
}

/// The bin-at-a-time walk, inside the one block that contains the answer.
///
/// **Outlined, and that is worth more than everything else in this scan put
/// together.** Inlined, this walk and the block sum above are two loops over the
/// same slice, and the compiler fuses them into one pass that keeps every
/// running prefix alive — which turns a vectorisable reduction into a spill.
/// Forcing the call halved the cost of a 4096-bin scan at both element sizes
/// measured (a 150-voxel line went from 528 to 254 ns/voxel), where widening the
/// accumulators and pinning the block length to a constant had changed nothing.
#[inline(never)]
fn refine(block: &[u32], mut cumulative: u32, target: u32) -> usize {
    for (offset, &count) in block.iter().enumerate() {
        cumulative += count;
        if cumulative > target {
            return offset;
        }
    }
    block.len().saturating_sub(1)
}

/// Which axis a window slides along, and what a step of one costs.
///
/// Computed from the element and nothing else. The axis is the one that leaves
/// the most of the window in place — equivalently the one with the fewest
/// leavers — because that count *is* the per-voxel cost of the traversal. For a
/// `200 x 200 x 1` element that is one of the wide axes at 200 leavers; sliding
/// along the flat one would retire and re-admit all 40 000 every step and be
/// slower than gathering.
///
/// Ties go to the highest axis index, which is the contiguous one: when two axes
/// cost the same number of updates, the one whose neighbours are adjacent in
/// memory is the cheaper of the two.
#[derive(Debug, Clone)]
pub struct ScanPlan {
    axis: usize,
    fixed: [usize; 2],
    /// Offsets from the **old** centre that a step of one retires.
    leaving: Vec<[isize; 3]>,
    /// Offsets from the **new** centre that a step of one admits.
    joining: Vec<[isize; 3]>,
}

impl ScanPlan {
    pub fn new(element: &StructuringElement) -> Self {
        let members: HashSet<[isize; 3]> = element.offsets().iter().copied().collect();
        let mut best = 0usize;
        let mut best_cost = usize::MAX;
        for axis in 0..3 {
            let cost = element
                .offsets()
                .iter()
                .filter(|offset| {
                    let mut probe = **offset;
                    probe[axis] -= 1;
                    !members.contains(&probe)
                })
                .count();
            if cost <= best_cost {
                best_cost = cost;
                best = axis;
            }
        }
        Self::along(element, best)
    }

    /// The same decomposition along an axis the caller names.
    ///
    /// [`Self::new`] picks the cheapest axis and is what an op wants. This is
    /// here because "the answer does not depend on which axis the window slid
    /// along" is a claim worth being able to *test*, and because a measurement
    /// that wants to price the axes against each other has to be able to ask for
    /// the slow one.
    pub fn along(element: &StructuringElement, axis: usize) -> Self {
        assert!(
            axis < 3,
            "an axis of a three-dimensional volume is 0, 1 or 2"
        );
        let members: HashSet<[isize; 3]> = element.offsets().iter().copied().collect();
        let shifted_by = |offset: &[isize; 3], axis: usize, delta: isize| {
            let mut probe = *offset;
            probe[axis] += delta;
            members.contains(&probe)
        };
        let leaving: Vec<[isize; 3]> = element
            .offsets()
            .iter()
            .filter(|offset| !shifted_by(offset, axis, -1))
            .copied()
            .collect();
        let joining: Vec<[isize; 3]> = element
            .offsets()
            .iter()
            .filter(|offset| !shifted_by(offset, axis, 1))
            .copied()
            .collect();
        debug_assert_eq!(
            leaving.len(),
            joining.len(),
            "a step admits exactly as many offsets as it retires"
        );
        let fixed = match axis {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        Self {
            axis,
            fixed,
            leaving,
            joining,
        }
    }

    /// The axis the window slides along.
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// How many voxels one step retires, which is also how many it admits.
    ///
    /// The per-voxel cost of the traversal, and the number to compare against
    /// the element's own size when asking whether sliding is worth it at all.
    pub fn step_size(&self) -> usize {
        self.leaving.len()
    }
}

/// Select `query` of `element` around every voxel of `input`, carrying the
/// histogram along the scan axis.
///
/// Byte-identical to the dense gather in [`super::rank`] where `query` is a
/// [`RankQuery`] over the same rank: same window, same truncation at every face,
/// same treatment of a mask and of an [`ExcludedCentre`]. What differs is only
/// how the window's contents are reached.
///
/// `mask`, where given, decides membership **at every offset**, exactly as
/// `masked_rank_filter_into` states it; `centre` decides what is written at a
/// voxel the mask excludes at its own position and is meaningful only alongside
/// a mask, which is also how the dense path reads it.
///
/// **Where the window comes out empty** the centre's own value is written if
/// there is a mask, and the call is an error if there is not — the same two
/// answers, for the same two reasons, as the dense kernel gives.
pub fn sliding_histogram_into<T: BinnedElement>(
    input: ArrayView3<'_, T>,
    mask: Option<ArrayView3<'_, bool>>,
    element: &StructuringElement,
    domain: Domain,
    query: &dyn HistogramQuery,
    centre: ExcludedCentre<T>,
    out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    let plan = ScanPlan::new(element);
    sliding_histogram_with_plan(
        input,
        mask,
        element,
        &plan,
        domain,
        query,
        centre,
        out,
        "sliding_histogram_into",
    )
}

/// [`sliding_histogram_into`] with the element's decomposition already computed.
///
/// The plan is a function of the element alone, so an op that holds one element
/// for its whole life computes it once rather than once per block — which for a
/// 40 000-offset element is the difference between a set-up cost comparable to
/// the block and one that is not there at all.
#[allow(clippy::too_many_arguments)]
pub fn sliding_histogram_with_plan<T: BinnedElement>(
    input: ArrayView3<'_, T>,
    mask: Option<ArrayView3<'_, bool>>,
    element: &StructuringElement,
    plan: &ScanPlan,
    domain: Domain,
    query: &dyn HistogramQuery,
    centre: ExcludedCentre<T>,
    mut out: ArrayViewMut3<'_, T>,
    what: &str,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), what)?;
    if let Some(mask) = mask.as_ref() {
        shapes_agree(input.shape(), mask.shape(), what)?;
    }
    if element.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "{what}: an empty element selects nothing"
        )));
    }
    let extent = [input.shape()[0], input.shape()[1], input.shape()[2]];
    if extent.contains(&0) {
        return Ok(());
    }

    // A contiguous read is what makes the flat arithmetic below legal at all;
    // a view that is not standard-layout is copied once rather than indexed
    // three-dimensionally at every offset of every step.
    let copied;
    let values: &[T] = match input.as_slice() {
        Some(slice) => slice,
        None => {
            copied = input.to_owned();
            copied
                .as_slice()
                .expect("an owned array is in standard layout")
        }
    };
    let mask_copied;
    let mask_values: Option<&[bool]> = match mask.as_ref() {
        None => None,
        Some(mask) => Some(match mask.as_slice() {
            Some(slice) => slice,
            None => {
                mask_copied = mask.to_owned();
                mask_copied
                    .as_slice()
                    .expect("an owned array is in standard layout")
            }
        }),
    };

    let bins = domain.bins();
    // The one place a value's range is checked. Doing it up front rather than at
    // every histogram update keeps the check out of the loop and makes the
    // refusal a property of the input rather than of which offsets happened to
    // be in bounds. A type whose whole range fits needs no pass at all.
    if T::NATURAL_DOMAIN.is_none_or(|natural| natural > bins) {
        if let Some(value) = values.iter().find(|value| value.bin() >= bins) {
            return Err(Error::InvalidArgument(format!(
                "{what}: the value {} is at or past the {bins} bins of the stated domain. A \
                 histogram cannot count it, and widening the domain silently would answer a \
                 different question from the one asked.",
                value.bin()
            )));
        }
    }

    let strides = [extent[1] * extent[2], extent[2], 1usize];
    let scan = plan.axis;
    let fixed = plan.fixed;
    let scan_stride = strides[scan] as isize;
    let scan_len = extent[scan];

    let mut counts = vec![0u32; bins];
    let mut population = 0usize;
    let mut leaving: Vec<(isize, isize)> = Vec::with_capacity(plan.leaving.len());
    let mut joining: Vec<(isize, isize)> = Vec::with_capacity(plan.joining.len());
    let mut unmasked_empty = false;
    // See the end of the line loop: the two ways of emptying the histogram cost
    // `|element|` and `|domain|`, and this is which of them is shorter.
    let unwind_at_line_end = element.len() <= bins;

    for first in 0..extent[fixed[0]] {
        for second in 0..extent[fixed[1]] {
            // Offsets whose *fixed* coordinates fall outside the array cannot
            // contribute anywhere on this line, so they are dropped once here
            // rather than tested at every step. What survives is folded into a
            // flat anchor that already carries the offset along the scan axis,
            // so a step's address is `anchor + position * scan_stride` and the
            // multiply is hoisted out of the inner loop.
            resolve_line(
                &plan.leaving,
                first,
                second,
                fixed,
                scan,
                &extent,
                &strides,
                scan_stride,
                &mut leaving,
            );
            resolve_line(
                &plan.joining,
                first,
                second,
                fixed,
                scan,
                &extent,
                &strides,
                scan_stride,
                &mut joining,
            );

            let mut centre_coord = [0usize; 3];
            centre_coord[fixed[0]] = first;
            centre_coord[fixed[1]] = second;

            // Prime by one full gather at the line's first voxel.
            centre_coord[scan] = 0;
            for offset in element.offsets() {
                if let Some(at) = resolve_offset(&centre_coord, offset, &extent, &strides) {
                    if mask_values.is_none_or(|mask| mask[at]) {
                        counts[values[at].bin()] += 1;
                        population += 1;
                    }
                }
            }

            let line_base = (first * strides[fixed[0]] + second * strides[fixed[1]]) as isize;
            for position in 0..scan_len {
                if position > 0 {
                    let previous = (position as isize - 1) * scan_stride;
                    for &(anchor, along) in &leaving {
                        let coordinate = position as isize - 1 + along;
                        if (coordinate as usize) < scan_len {
                            let at = (anchor + previous) as usize;
                            if mask_values.is_none_or(|mask| mask[at]) {
                                counts[values[at].bin()] -= 1;
                                population -= 1;
                            }
                        }
                    }
                    let current = position as isize * scan_stride;
                    for &(anchor, along) in &joining {
                        let coordinate = position as isize + along;
                        if (coordinate as usize) < scan_len {
                            let at = (anchor + current) as usize;
                            if mask_values.is_none_or(|mask| mask[at]) {
                                counts[values[at].bin()] += 1;
                                population += 1;
                            }
                        }
                    }
                }

                let here = (line_base + position as isize * scan_stride) as usize;
                centre_coord[scan] = position;
                let slot = [centre_coord[0], centre_coord[1], centre_coord[2]];

                // The centre's own membership, asked of the mask rather than of
                // the window's emptiness — the same distinction `ExcludedCentre`
                // is documented under, and the same order the dense kernel asks
                // it in. The window slides either way; only the write changes.
                if let (Some(mask), ExcludedCentre::Fill(value)) = (mask_values, centre) {
                    if !mask[here] {
                        out[slot] = value;
                        continue;
                    }
                }

                if population == 0 {
                    if mask_values.is_some() {
                        out[slot] = values[here];
                    } else {
                        // Raised after the line unwinds, so the histogram is
                        // never left dirty by the early return.
                        unmasked_empty = true;
                    }
                    continue;
                }
                out[slot] = T::from_f64(query.evaluate(&counts, population));
            }

            // Empty the histogram for the next line, by whichever of the two
            // routes is shorter. **Unwinding** retires the last window by the
            // same test that admitted it and costs `|element|`; **zeroing**
            // costs `|domain|`. Neither is always the cheaper — a 27-voxel
            // element over 65 536 bins wants the first and a 40 000-voxel
            // element over 4096 bins wants the second — and the choice is a
            // comparison rather than a preference.
            //
            // The unwind is the one that can be *checked*, since it is the step
            // rule run backwards, so a debug build always runs it and asserts
            // that it emptied the histogram exactly. That assertion is the
            // standing proof that admission and retirement agree about what is
            // in the window; the zeroing path would hide a disagreement.
            if unwind_at_line_end || cfg!(debug_assertions) {
                centre_coord[scan] = scan_len - 1;
                for offset in element.offsets() {
                    if let Some(at) = resolve_offset(&centre_coord, offset, &extent, &strides) {
                        if mask_values.is_none_or(|mask| mask[at]) {
                            counts[values[at].bin()] -= 1;
                            population -= 1;
                        }
                    }
                }
                debug_assert_eq!(population, 0, "a line left voxels in the window");
                debug_assert!(
                    counts.iter().all(|&count| count == 0),
                    "a line left counts in the histogram"
                );
            } else {
                counts.fill(0);
                population = 0;
            }

            if unmasked_empty {
                return Err(Error::InvalidArgument(format!(
                    "{what}: an element that misses its own centre"
                )));
            }
        }
    }
    Ok(())
}

/// The offsets of `set` that can contribute on this line, as
/// `(flat anchor, offset along the scan axis)`.
#[allow(clippy::too_many_arguments)]
fn resolve_line(
    set: &[[isize; 3]],
    first: usize,
    second: usize,
    fixed: [usize; 2],
    scan: usize,
    extent: &[usize; 3],
    strides: &[usize; 3],
    scan_stride: isize,
    out: &mut Vec<(isize, isize)>,
) {
    out.clear();
    for offset in set {
        let a = first as isize + offset[fixed[0]];
        let b = second as isize + offset[fixed[1]];
        if a < 0 || b < 0 || a as usize >= extent[fixed[0]] || b as usize >= extent[fixed[1]] {
            continue;
        }
        let base = a as usize * strides[fixed[0]] + b as usize * strides[fixed[1]];
        out.push((base as isize + offset[scan] * scan_stride, offset[scan]));
    }
}

/// The flat index of `centre + offset`, or `None` where that is outside the
/// array. The one bounds test, used by the priming gather and the unwind alike
/// so that the two cannot disagree about what was in the window.
#[inline(always)]
fn resolve_offset(
    centre: &[usize; 3],
    offset: &[isize; 3],
    extent: &[usize; 3],
    strides: &[usize; 3],
) -> Option<usize> {
    let mut at = 0usize;
    for axis in 0..3 {
        let coordinate = centre[axis] as isize + offset[axis];
        if coordinate < 0 || coordinate as usize >= extent[axis] {
            return None;
        }
        at += coordinate as usize * strides[axis];
    }
    Some(at)
}

/// A windowed statistic over a histogram carried along a scan line.
///
/// The same declarations as `RankFilterOp` — the reach is the element's and
/// nothing configures it — with two of its own: a [`Domain`], because the
/// traversal is only defined over one, and a [`HistogramQuery`], because the
/// traversal is the part that is shared and the question is the part that is not.
///
/// Optionally masked, in which case the population is read from a stored level
/// over the same window, exactly as `MaskedRankFilterOp` reads it. One op rather
/// than two, because the mask is one more reason an offset does not join the
/// window and not a second traversal.
pub struct SlidingHistogramOp {
    name: &'static str,
    element: StructuringElement,
    plan: ScanPlan,
    domain: Domain,
    query: Box<dyn HistogramQuery>,
    mask: Option<usize>,
    centre: ExcludedCentre<f64>,
    cost: f64,
}

impl SlidingHistogramOp {
    pub fn new(
        name: &'static str,
        element: StructuringElement,
        domain: Domain,
        query: Box<dyn HistogramQuery>,
    ) -> Self {
        let plan = ScanPlan::new(&element);
        let cost = cost_for(&plan, domain, query.as_ref());
        Self {
            name,
            element,
            plan,
            domain,
            query,
            mask: None,
            centre: ExcludedCentre::Select,
            cost,
        }
    }

    /// The order statistic, which is the case this op was built for.
    pub fn rank(
        name: &'static str,
        element: StructuringElement,
        rank: Rank,
        domain: Domain,
    ) -> Self {
        let query = RankQuery::new(rank, &element);
        Self::new(name, element, domain, Box::new(query))
    }

    /// Read the window's population from `mask`, which must be a `Bool` level.
    pub fn masked_by(mut self, mask: impl Into<crate::assemble::Level>) -> Self {
        self.mask = Some(mask.into().index());
        self.cost *= MASK_COST_FACTOR;
        self
    }

    /// State what happens at a voxel whose own membership the population denies;
    /// see [`ExcludedCentre`]. Meaningful only alongside [`Self::masked_by`].
    pub fn with_excluded_centre(mut self, centre: ExcludedCentre<f64>) -> Self {
        self.centre = centre;
        self
    }

    pub fn element(&self) -> &StructuringElement {
        &self.element
    }

    pub fn plan(&self) -> &ScanPlan {
        &self.plan
    }

    pub fn domain(&self) -> Domain {
        self.domain
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    fn run<T: BinnedElement>(
        &self,
        input: &Voxels,
        mask: Option<ArrayView3<'_, bool>>,
        out: &mut Voxels,
    ) -> Result<()> {
        let source = input.view::<T>()?;
        sliding_histogram_with_plan(
            source,
            mask,
            &self.element,
            &self.plan,
            self.domain,
            self.query.as_ref(),
            self.centre.map(T::from_f64),
            out.view_mut::<T>()?,
            self.name,
        )
    }

    fn dispatch(
        &self,
        input: &Voxels,
        mask: Option<ArrayView3<'_, bool>>,
        out: &mut Voxels,
    ) -> Result<()> {
        match input.dtype() {
            Dtype::Bool => self.run::<bool>(input, mask, out),
            Dtype::U8 => self.run::<u8>(input, mask, out),
            Dtype::U16 => self.run::<u16>(input, mask, out),
            Dtype::U32 => self.run::<u32>(input, mask, out),
            other => Err(Error::InvalidArgument(format!(
                "{}: a histogram is an array indexed by the value, so this op counts unsigned \
                 integers and nothing else; {} has no bins. Binning it would answer a different \
                 question from the one the dense filter answers, at every voxel and silently, so \
                 it is refused rather than converted. `accepts` refuses it before a run starts.",
                self.name,
                other.numpy_name()
            ))),
        }
    }
}

impl BlockOp for SlidingHistogramOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.element.reach(axis)
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        self.element.reach_spec()
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        match self.mask {
            Some(mask) => vec![SourceInput::new(mask, self.element.reach_spec())],
            None => Vec::new(),
        }
    }

    /// The unsigned integers, and nothing else. See [`BinnedElement`]: the
    /// refusal of the floats is the constraint stated rather than worked around,
    /// and the signed integers are refused with them because a negative value
    /// does not index a bin from zero.
    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(dtype, Dtype::Bool | Dtype::U8 | Dtype::U16 | Dtype::U32)
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        if let Some(level) = self.mask {
            return Err(Error::InvalidArgument(format!(
                "{}: the population comes from level {level}, so this op has no answer from its \
                 input alone. It is applied through `apply_with`.",
                self.name
            )));
        }
        self.dispatch(input, None, out)
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        at: &Anchor,
    ) -> Result<()> {
        let Some(level) = self.mask else {
            return self.apply(input, out, at);
        };
        let mask = sources.get(level)?;
        if mask.dtype() != Dtype::Bool {
            return Err(Error::InvalidArgument(format!(
                "{}: the population is read from level {level}, which holds {}. A population is a \
                 yes-or-no per voxel and is stored as one.",
                self.name,
                mask.dtype().numpy_name()
            )));
        }
        let mask = mask.view::<bool>()?;
        self.dispatch(input, Some(mask), out)
    }

    /// The query's declaration, withdrawn where a fill could make the answer a
    /// fact about the mask instead — the same rule, for the same reason, as
    /// `MaskedRankFilterOp::constant_maps_to` states.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        let answer = self.query.constant_maps_to(value)?;
        match self.centre {
            ExcludedCentre::Select => Some(answer),
            ExcludedCentre::Fill(fill) => (self.mask.is_none() || fill == answer).then_some(answer),
        }
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// A seed, in the sense `super::COST_MEASUREMENT` means it.
///
/// Two terms, because the traversal has two: one histogram update per voxel per
/// offset the step moves — *not* per offset of the element, which is the whole
/// point — and one evaluation of the query over the domain. What is **not**
/// modelled is the per-line priming gather, which costs `|element|` per line and
/// therefore `|element| / line length` per voxel: `cost_per_voxel` is handed no
/// volume, so a term that depends on the line length cannot be stated here. It
/// is the reason a short scan line is worse than this number claims.
fn cost_for(plan: &ScanPlan, domain: Domain, query: &dyn HistogramQuery) -> f64 {
    SLIDING_UPDATE_COST * 2.0 * plan.step_size() as f64 + query.cost_per_evaluation(domain.bins())
}

/// What admitting or retiring one voxel costs: a bounds test, a mask test, a
/// load and a counter.
const SLIDING_UPDATE_COST: f64 = 1.6;

/// What one bin of the blocked cumulative scan costs. Well below one because
/// `SCAN_BLOCK` bins are summed as independent adds.
const RANK_SCAN_COST_PER_BIN: f64 = 0.06;

/// What consulting the population adds, as a multiple of the unmasked traversal.
/// The same seed `super::rank` uses, and for the same reason.
const MASK_COST_FACTOR: f64 = 1.35;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::element::ElementShape;
    use ndarray::Array3;

    fn ramp(shape: (usize, usize, usize), modulus: u16) -> Array3<u16> {
        Array3::from_shape_fn(shape, |(i, j, k)| {
            (((i * 7919 + j * 104729 + k * 1299709) % modulus as usize) as u16).min(modulus - 1)
        })
    }

    #[test]
    fn the_step_admits_exactly_what_it_retires() {
        for element in [
            StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]),
            StructuringElement::from_radius(ElementShape::Ellipsoid, [3, 2, 0]),
            StructuringElement::from_size(ElementShape::Box, [4, 6, 2]).unwrap(),
        ] {
            let plan = ScanPlan::new(&element);
            assert_eq!(plan.leaving.len(), plan.joining.len());
            assert!(plan.step_size() <= element.len());
        }
    }

    /// The axis chosen is the one with the fewest leavers, which for a flat wide
    /// element is one of the wide axes and never the flat one.
    #[test]
    fn the_scan_axis_is_the_one_that_moves_the_least() {
        let element = StructuringElement::from_size(ElementShape::Box, [21, 21, 1]).unwrap();
        let plan = ScanPlan::new(&element);
        assert!(plan.axis() < 2, "axis {} is the flat one", plan.axis());
        assert_eq!(plan.step_size(), 21);
    }

    #[test]
    fn a_float_domain_is_refused_by_name() {
        let op = SlidingHistogramOp::rank(
            "sliding",
            StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]),
            Rank::lowest(),
            Domain::of::<u8>().unwrap(),
        );
        assert!(!op.accepts(Dtype::F64));
        assert!(!op.accepts(Dtype::F32));
        assert!(!op.accepts(Dtype::I16));
        assert!(op.accepts(Dtype::U16));
    }

    #[test]
    fn a_type_with_no_bounded_domain_has_to_be_told_one() {
        assert!(Domain::of::<u32>().is_err());
        assert!(Domain::of_size(4096).is_ok());
        assert!(Domain::of_size(0).is_err());
        assert!(Domain::of_size(Domain::LARGEST + 1).is_err());
    }

    #[test]
    fn a_value_past_the_domain_is_named_rather_than_folded_away() {
        let input = ramp((5, 5, 5), 4096);
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let mut out = Array3::<u16>::zeros(input.dim());
        let query = RankQuery::new(Rank::median(&element), &element);
        let error = sliding_histogram_into(
            input.view(),
            None,
            &element,
            Domain::of_size(64).unwrap(),
            &query,
            ExcludedCentre::Select,
            out.view_mut(),
        )
        .unwrap_err();
        assert!(format!("{error}").contains("bins"), "{error}");
    }

    #[test]
    fn the_cumulative_walk_finds_the_order_statistic() {
        let counts = {
            let mut counts = vec![0u32; 100];
            counts[3] = 2;
            counts[40] = 1;
            counts[99] = 4;
            counts
        };
        assert_eq!(first_bin_past(&counts, 0), 3);
        assert_eq!(first_bin_past(&counts, 1), 3);
        assert_eq!(first_bin_past(&counts, 2), 40);
        assert_eq!(first_bin_past(&counts, 3), 99);
        assert_eq!(first_bin_past(&counts, 6), 99);
    }

    #[test]
    fn a_constant_selects_that_constant() {
        let element = StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 2, 2]);
        let op = SlidingHistogramOp::rank(
            "sliding",
            element.clone(),
            Rank::median(&element),
            Domain::of::<u8>().unwrap(),
        );
        assert_eq!(op.constant_maps_to(7.0), Some(7.0));

        let input = Array3::from_elem((6, 6, 6), 7u8);
        let mut out = Array3::<u8>::zeros(input.dim());
        let query = RankQuery::new(Rank::median(&element), &element);
        sliding_histogram_into(
            input.view(),
            None,
            &element,
            Domain::of::<u8>().unwrap(),
            &query,
            ExcludedCentre::Select,
            out.view_mut(),
        )
        .unwrap();
        assert!(out.iter().all(|&value| value == 7));
    }
}
