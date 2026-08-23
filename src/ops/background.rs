// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation,
// not adapted from any implementation of it.
//
// **Large-scale background estimation, and its removal.**
//
// The construction is the classical one. Estimate the background by a
// *morphological opening* over an element large enough that no object of
// interest survives it — the opening is anti-extensive, so what remains is the
// surface the objects sit on — and then subtract that estimate from the original.
// The difference is the top-hat.
//
// This file computes nothing over a neighbourhood
// -----------------------------------------------
// There is no sweep, no window and no accumulator anywhere below — the only loop
// over voxels in the whole file fills a ramp for the cost measurement, and the
// only loops in the tests are a written-out statement of the definition being
// checked against. That is the finding rather than an omission, and it is worth
// stating before anything else, because "implement background subtraction" reads
// like a request for a kernel and is not one here. Every piece of the arithmetic
// already existed; what did not exist was the **shape**.
//
// | piece | what it is | where it comes from |
// |---|---|---|
// | the estimate | a grey opening: the minimum over the element, then the maximum | [`RankFilterOp`] twice, at [`Rank::lowest`] and [`Rank::highest`] — **no new kernel** |
// | the original | an identity | [`VoxelwiseMapOp`] over [`Identity`](super::voxelwise::Identity) — **no new kernel** |
// | the shape | one input, two arms, one sink: a **diamond** | [`Chain::Parallel`] |
// | the sink | a voxelwise subtraction | [`combine_into`] existed; a [`Combine`] over it did not. **This is the new code, and it is one `-`.** |
//
// Why a rank filter and not `super::morphology`
// ---------------------------------------------
// `morphology::open_into` is the same operation over `bool`, and a background is
// not a mask. A *grey* erosion is the minimum over the element and a grey
// dilation is the maximum, which are the extreme ranks of the same element —
// `morphology`'s own
// `erosion_and_dilation_are_the_extreme_ranks_of_the_same_element` already pins
// that equality, over the same [`StructuringElement`], in the crate rather than
// in a comment here. So the grey opening is `rank(lowest)` then `rank(highest)`,
// and adding a second morphology kernel that happened to be generic over `Ord`
// would have been a second implementation of a filter this module already has.
//
// The clamp carries across with it. [`Rank::resolve`] sends the lowest rank to
// `0` and the highest to `available - 1` at every truncation, so at a real volume
// boundary the minimum is taken over the voxels that exist and the maximum
// likewise — which is exactly `morphology`'s convention ("what lies outside
// behaves as set for an erosion, clear for a dilation") stated for a grey image.
// At a block seam it is deliberately wrong, and that is what makes a short halo
// diverge instead of passing quietly.
//
// The diamond, and why it is a `Parallel` rather than anything else
// ----------------------------------------------------------------
// Both arms consume **the same input**: the original is one of the two operands
// of the subtraction, not something fetched from elsewhere. That is the
// definition of a fan-in, and `docs/design/BLOCK_OPS.md` records what happens
// when such a thing is modelled as a `Chain::Alternative` instead — 903
// comparisons passed, because reach folds as the max either way and no reach
// test can tell the two readings apart. What tells them apart is that *both arms
// run*, and `tests/background_removal.rs` counts the applications of each arm and
// compares them against the task count.
//
// It could also have been written as a [`super::voxelwise::CombineOp`] holding
// the whole original volume and slicing it at the anchor. That is a different
// shape and a worse one here: the operand would be an array the chain did not
// produce, so the original would have to be materialised in full alongside the
// input it already is, and the chain would no longer say that the two arms come
// from one place.
//
// **Branch order is the operation, not a convention.** `Chain::Parallel` hands
// results to the combine in branch order; `LogicCombine` folds a commutative
// connective and does not care, and a difference does. Branch 0 is the minuend
// — the original — and branch 1 the subtrahend. Swapping them negates the
// answer, which is asserted below rather than trusted.
//
// **Neither arm declares a side output**, so the union `Chain::Parallel` folds
// over them is empty and there is nothing here that could leave a hole. That is
// a decision rather than an oversight: keeping the estimate is a thing a caller
// may well want, and the way to have it is [`background_estimate`] as a chain of
// its own — it is the same two ops, and running it separately costs a second
// pass over the volume rather than an array the executor has to be told about.
// An op that declared the estimate as a side output would have to be a single op
// rather than a `Sequence`, which is exactly the cut point the estimate is a
// `Sequence` in order to keep. If that trade ever wants making, it is one op in a
// new file and nothing here changes.
//
// What this file does about the element's step origin: nothing, and that is
// the answer rather than an omission
// ------------------------------------------------------------------------
// An element whose step counts from `StepOrigin::ClippedStart` reads a different
// set of offsets where the window is clipped at a low face of the volume, so an
// op that gathered one offset set would compute a filter other than the one it
// names. There is no gather here to get that wrong: both arms of the estimate
// **are** [`RankFilterOp`], which asks the element what it reads at each voxel's
// position in the volume, and the sink is voxelwise. So the origin is honoured
// by construction, at whatever the rank filter honours it at, and a second
// statement of the rule in this file would be a second thing to keep in step.
// `the_estimate_honours_the_step_origin_through_the_filter_it_is_made_of` is
// that composition asserted rather than assumed.
//
// The reach
// ---------
// `2 * radius` per axis, and **nothing here writes a 2**. The estimate is a
// [`Chain::Sequence`] of two rank filters, each of which derives its own reach
// from the element it holds, and `Chain::reach` adds along a sequence: an
// erosion consumes input a radius away and the dilation consumes erosion values
// a radius away from that. The combine reaches zero — it reads the voxel it
// writes, in each operand — so the fan-in folds to `max(0, 2r) + 0 = 2r`.
//
// [`background_reach`] states the same number the other way, from
// [`Morphology::reach_factor`] — the composition factor `morphology` already
// derives for an opening — so that the two statements can be checked against each
// other rather than trusted. `the_reach_is_twice_the_element_and_the_two_
// statements_agree` is that check. Neither of them is a constant a caller can
// set: there is no field to set.
//
// It is **tight**, in the sense the crate means: understating it by one voxel
// produces visibly wrong seams, which `tests/background_removal.rs` demonstrates
// by doing it.
//
// The constant algebra, proved rather than argued
// -----------------------------------------------
// A top-hat of a constant field is exactly `+0.0`, and the proof is three lines
// because both operands are exact:
//
// 1. the identity arm reproduces the constant, bit for bit — it is `|v| v`;
// 2. the opening arm reproduces it too, and this is the part that would fail for
//    a *smoothed* estimate. A rank filter selects a value that was read and never
//    combines two, so both passes hand back the constant unchanged at every
//    truncation of the element. A windowed mean would not: `(v + ... + v) / m` is
//    not `v` in binary floating point, which is why `super::local` withholds its
//    declaration for a mean at any value but zero;
// 3. IEEE 754 gives `x - x = +0` exactly, for every finite `x`, in every
//    rounding mode Rust uses (`roundTiesToEven`) and in every binary format. Not
//    "approximately zero" and not "-0.0": the sign of an exact zero result from a
//    subtraction of equal operands is positive.
//
// So [`DifferenceCombine::constant_maps_to`] declares `+0.0` for equal finite
// operands and **nothing else at all**, which is narrower than the arithmetic
// alone would allow and is deliberate:
//
// * two *unequal* constants have a difference that is exact in `f64` and is not
//   necessarily the `f32` kernel's answer, because the executor would narrow a
//   `f64` declaration to fill a `f32` block and a double rounding is not a
//   rounding. Equal operands are immune — `+0.0` is `+0.0` in every format;
// * two infinities differ to `NaN`, and a `NaN` is not equal to itself, so a
//   declaration of one could not be checked against the computed block by the
//   standard this crate holds declarations to.
//
// `a_constant_field_maps_to_positive_zero_in_bits` checks the claim as bits, in
// both element types, against the computed answer.
//
// Element types
// -------------
// The **combine** bridges `f64` and `f32`: subtraction is generic over
// `Sub<Output = Self>` and both floats are closed under it. It does **not**
// accept the integers, and that is a statement about a general two-operand
// combine rather than about a top-hat: `u8` subtraction wraps where the minuend
// does not dominate, and this combine cannot know that its branches arrange for
// it to. (In a top-hat it does — see the anti-extensivity note below — but that
// is a property of one arrangement of two branches, not of the sink.)
//
// The **assembled chain** is `f64` only, because the identity arm is a
// [`VoxelwiseMapOp`], which holds an `f64 -> f64` map and says so. A caller
// wanting the `f32` path today builds the diamond with an identity of their own;
// nothing in this file would change.
//
// Anti-extensivity, which makes the difference non-negative
// ---------------------------------------------------------
// `open(f) <= f` pointwise, so the top-hat never goes below zero, and it is a
// theorem rather than a clamp. A [`StructuringElement`] is generated over
// `-r..=r` with a predicate that depends on each offset only through its square,
// so it contains its own centre and is symmetric under negation. Then for any
// voxel `v` in the array, `open(f)(v) = max_o min_p f(v + o + p)` over offsets
// that stay inside the array, and every inner minimum has `p = -o` available —
// which lands on `v` — so every term is at most `f(v)`. The clamp does not break
// it: it only removes terms from maxima and minima that were already bounded by
// `f(v)`.
//
// Costs
// -----
// Measured, by [`cost_report`], which is runnable and prints the table it was
// taken from. The estimate's cost is not this file's to state — it is two
// [`RankFilterOp`]s, and `rank`'s constant is stored **per element voxel** so
// that a 123-voxel element and a 1331-voxel one are not one number. That is what
// makes a large-radius background estimate priced as the expensive thing it is:
// the planner sees `2 * 3.87 * |element|` per voxel and not a flat figure.

use std::ops::Sub;
use std::time::Instant;

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Chain, Combine, Slicing};
use crate::reach::Reach;
use crate::voxels::Voxels;

use super::element::{Rank, StructuringElement};
use super::morphology::Morphology;
use super::rank::RankFilterOp;
use super::voxelwise::{combine_into, VoxelwiseMapOp};

// --------------------------------------------------------------- kernel --

/// `out = minuend - subtrahend`, voxelwise.
///
/// Generic over `Sub`, which is the whole of what the operation requires: it
/// reads one voxel from each operand and writes one, and imposes no order, no
/// accumulator and no conversion. The shell below is the only part that has to
/// know which element types a buffer can hold.
///
/// `out` may alias neither operand; nothing here checks that, on the same
/// footing as the other kernels in `super`, because the buffers come from the
/// executor and it does not alias them.
pub fn difference_into<T>(
    minuend: ArrayView3<'_, T>,
    subtrahend: ArrayView3<'_, T>,
    out: ArrayViewMut3<'_, T>,
) -> Result<()>
where
    T: Copy + Sub<Output = T>,
{
    combine_into(minuend, subtrahend, out, |&left, &right| left - right)
}

// -------------------------------------------------------------- the sink --

/// The sink of the diamond: **branch 0 minus branch 1**, voxelwise.
///
/// The counterpart of [`super::voxelwise::LogicCombine`] for arithmetic, and it
/// differs from that one in exactly two ways, both of which follow from
/// subtraction not being a connective:
///
/// * **exactly two branches**, where `LogicCombine` takes any number. `And`,
///   `Or` and `Xor` are associative, so folding them left over `n` results is
///   well defined and is what a three-arm diamond means. A left fold of
///   subtraction is *defined* but it is not a difference — `a - b - c` is a
///   convention about where the parentheses went, and a combine that silently
///   picked one would be a worse thing than a combine that refuses. The arity is
///   checked in [`Combine::accepts`], so the refusal happens when the plan is
///   made;
/// * **the order is load-bearing.** Branch order is the argument order.
///
/// It reaches zero on every axis, so a fan-in whose sink is this one has the
/// reach of its widest arm and no more.
pub struct DifferenceCombine {
    name: &'static str,
    cost: f64,
}

impl DifferenceCombine {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            cost: DIFFERENCE_COST,
        }
    }

    /// Override the measured cost.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl Combine for DifferenceCombine {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Zero, on every axis and at every volume size: one voxel is read from each
    /// operand and one is written. Stated explicitly because [`Combine::reach`]
    /// has no default, and a fan-in's halo is its widest branch's plus this.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    /// **A stencil**, and this is the declaration a fan-in cannot get from its
    /// branches. A `Parallel` node is only as sliceable as its narrowest part,
    /// so a diamond whose arms are declared stencils is still refused while its
    /// sink says nothing — which is the position every fan-in in this crate was
    /// in until this line existed.
    ///
    /// The claim itself is [`difference_into`]'s: it writes each output voxel from the
    /// co-located voxel of each operand, through one
    /// `Zip` that reads no neighbour and carries no accumulator between voxels.
    /// So the output at `v` is a function of the inputs at `v`, the reach is
    /// zero on every axis, and the output lattice is the input lattice — the
    /// three conditions [`Slicing::Stencil`] states.
    ///
    /// Held to it rather than believed: `tests/intra_block_slicing.rs` runs a
    /// fan-in whose sink is this one uncut and then cut at every thread count
    /// and requires the same bits.
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    /// Exactly two branches, of the same floating-point element type.
    ///
    /// The integers are refused rather than wrapped; see the module header.
    fn accepts(&self, inputs: &[Dtype]) -> bool {
        inputs.len() == 2 && matches!(inputs[0], Dtype::F64 | Dtype::F32) && inputs[1] == inputs[0]
    }

    fn produces(&self, inputs: &[Dtype]) -> Dtype {
        inputs[0]
    }

    /// Both branches must have produced the same extent, and the two that did
    /// not are named — a voxelwise difference of buffers with no correspondence
    /// between their voxels is not a difference.
    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        match inputs {
            [minuend, subtrahend] if minuend == subtrahend => Ok(*minuend),
            [minuend, subtrahend] => Err(Error::InvalidArgument(format!(
                "{}: branch 0 produced {minuend:?} and branch 1 produced {subtrahend:?}. A \
                 voxelwise difference subtracts co-located voxels, and buffers of different \
                 extents have no such pairing.",
                self.name
            ))),
            other => Err(Error::InvalidArgument(format!(
                "{}: a difference has a minuend and a subtrahend and was handed {} branch \
                 results. Subtraction is not associative, so there is no fold over a longer \
                 list that is not a convention about parentheses.",
                self.name,
                other.len()
            ))),
        }
    }

    fn apply(&self, inputs: &[Voxels], out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let (minuend, subtrahend) = match inputs {
            [minuend, subtrahend] => (minuend, subtrahend),
            other => {
                return Err(Error::InvalidArgument(format!(
                    "{}: a difference joins exactly two results and was handed {}",
                    self.name,
                    other.len()
                )));
            }
        };
        match minuend.dtype() {
            Dtype::F64 => difference_into(
                minuend.view::<f64>()?,
                subtrahend.view::<f64>()?,
                out.view_mut::<f64>()?,
            ),
            Dtype::F32 => difference_into(
                minuend.view::<f32>()?,
                subtrahend.view::<f32>()?,
                out.view_mut::<f32>()?,
            ),
            other => Err(Error::InvalidArgument(format!(
                "{}: a difference is stated over the floating-point element types and was \
                 handed {}. `accepts` refuses it before a run starts.",
                self.name,
                other.numpy_name()
            ))),
        }
    }

    /// `+0.0` where the two operands are the same finite value, and nothing
    /// anywhere else. The three-line proof, and the two exclusions, are in the
    /// module header.
    fn constant_maps_to(&self, values: &[f64]) -> Option<f64> {
        match values {
            [minuend, subtrahend]
                if minuend.is_finite() && minuend.to_bits() == subtrahend.to_bits() =>
            {
                Some(0.0)
            }
            _ => None,
        }
    }

    /// One pair's worth of work. The arity is an argument to this method because
    /// a pairwise combine folded over `n` branches does `n - 1` pairs; this one
    /// only ever sees two, and says the same thing in the same form so that a
    /// reader comparing it against [`super::voxelwise::LogicCombine`] sees one
    /// rule rather than two.
    fn cost_per_voxel(&self, branches: usize) -> f64 {
        self.cost * branches.saturating_sub(1) as f64
    }
}

// ---------------------------------------------------------- the assembly --

/// The name each piece of the diamond carries into a plan, a log and a
/// progress display. Fixed literals because [`BlockOp::name`] is `&'static str`
/// — there is nothing to format a caller's prefix into — and distinct so that a
/// phase table naming two rank filters says which is which.
const LOWEST: &str = "background.lowest";
const HIGHEST: &str = "background.highest";
const ORIGINAL: &str = "background.original";
const DIFFERENCE: &str = "background.difference";

/// The background estimate: a **grey opening** over `element`.
///
/// Two rank filters in sequence — the lowest rank of the element, then the
/// highest — which is a grey erosion followed by a grey dilation. Objects
/// smaller than the element do not survive the erosion and are not restored by
/// the dilation; everything larger is, to within the element's own scale.
/// Choosing `element` is therefore the whole of the parameterisation, and it is
/// the caller's: there is no default size in this crate.
///
/// **A [`Chain::Sequence`] rather than one op, deliberately.** `Chain::slots`
/// flattens sequences, so a planner may cut *between* the two passes and give
/// them separate phases if that is what the cost model prefers. An op that did
/// both passes internally would be one indivisible slot and would have to state
/// its own `2 * radius`; this way the reach is the sum the fold computes.
///
/// A radius of zero on every axis makes the estimate an identity and the
/// removal identically zero. That is the honest answer for that parameter
/// rather than an error: the element is a parameter, and an element of one voxel
/// opens nothing.
pub fn background_estimate(element: &StructuringElement) -> Chain {
    Chain::sequence(vec![
        Chain::op(RankFilterOp::new(LOWEST, element.clone(), Rank::lowest())),
        Chain::op(RankFilterOp::new(
            HIGHEST,
            element.clone(),
            Rank::highest(element),
        )),
    ])
}

/// The removal: **the original minus [`background_estimate`]**, as a diamond.
///
/// A `Chain::Parallel` over two arms that read the same buffer at the same
/// anchor — an identity, and the estimate — joined by [`DifferenceCombine`].
/// Branch 0 is the minuend.
///
/// Fallible only because [`Chain::parallel`] is: it refuses a fan-in of fewer
/// than two branches, and this one always hands it two, so the `Err` arm is
/// unreachable and is propagated rather than unwrapped so that a change which
/// made it reachable would not panic in a library.
pub fn remove_background(element: &StructuringElement) -> Result<Chain> {
    Chain::parallel(
        vec![
            Chain::op(VoxelwiseMapOp::identity(ORIGINAL)),
            background_estimate(element),
        ],
        Box::new(DifferenceCombine::new(DIFFERENCE)),
    )
}

/// What [`background_estimate`] and [`remove_background`] read beyond the voxel
/// they write, per axis: the element's radius times the number of passes an
/// opening makes.
///
/// **The second statement of one quantity, and that is what it is for.** The
/// authority is `Chain::reach`, which folds the sequence and the fan-in and is
/// what every plan is built from. This one derives the same number from
/// [`Morphology::reach_factor`] — the composition factor `super::morphology`
/// already owns, so that "an opening is two passes" is written down once in the
/// crate — and the test below asserts the two agree for a range of elements. A
/// caller sizing a halo before building a chain can use it; a caller holding a
/// chain should ask the chain.
pub fn background_reach(element: &StructuringElement) -> [usize; 3] {
    let passes = Morphology::Open.reach_factor();
    [
        element.reach(0) * passes,
        element.reach(1) * passes,
        element.reach(2) * passes,
    ]
}

/// The same, per side.
///
/// [`background_reach`] is the symmetric **bound** — the element's wider side
/// times the passes — and for an element with a centre voxel the two say the
/// same thing. For an element with an even extent they do not: the two rank
/// filters both read further below the anchor than above it, and the sequence
/// adds side by side, so this is `(2 * lo, 2 * hi)` where the triple is
/// `(2 * max, 2 * max)`. This is the one a plan is built from — `Chain::
/// reach_spec` folds exactly it out of the two `RankFilterOp`s — and the test
/// below asserts that, rather than the equality being assumed.
pub fn background_reach_spec(element: &StructuringElement) -> Reach {
    element.reach_spec_after(Morphology::Open.reach_factor())
}

// ---------------------------------------------------------------- costs --

/// Measured; see [`cost_report`], and `super::COST_MEASUREMENT` for the method.
///
/// Relative to the voxelwise map, which is this module's unit of work — the same
/// unit `super::cost` uses, and the two reports agree on it to within half a
/// percent on the machine this was taken on (2.44 against 2.45 ns per voxel over
/// 96 x 64 x 64), which is the cross-check that makes the two tables
/// comparable at all.
///
/// **The spread is stated because it is wide.** Four runs gave 0.28, 0.37, 0.43
/// and 0.44; `0.40` is stored. A per-voxel figure below a nanosecond is a
/// measurement of memory bandwidth and of what else the machine was doing, not
/// of the subtraction, and the ratio inherits the noise of *both* minima it is
/// formed from. What the planner needs from this number is that a voxelwise sink
/// is a fraction of a neighbourhood pass rather than a rival to it, and every one
/// of those four runs says that by two and a half orders of magnitude.
///
/// Stored separately from `super::voxelwise::COMBINE_COST` even though the two
/// are close: a subtraction and a masked connective are the same *shape* of work
/// and not the same work, and a constant shared between two ops stops being a
/// measurement of either as soon as one of them changes.
const DIFFERENCE_COST: f64 = 0.40;

/// Retake the measurement, here, through the same entry points the executor
/// uses. Runnable; `print_the_cost_table` below is the one command.
///
/// The unit is the voxelwise map, as in `super::cost`. For the estimate the
/// useful column is the cost **per element voxel**, because its work scales with
/// the element and one number would be right for one filter size only — which
/// is the whole reason `rank`'s constant is stored that way and the reason a
/// large-radius background estimate is priced as an expensive op rather than as
/// a filter.
pub fn cost_report(shape: [usize; 3], repetitions: usize) -> String {
    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let input = ramp(shape);
    let repetitions = repetitions.max(1);

    let best_of = |mut run: Box<dyn FnMut()>| -> f64 {
        run();
        let mut best = f64::INFINITY;
        for _ in 0..repetitions {
            let started = Instant::now();
            run();
            best = best.min(started.elapsed().as_secs_f64() * 1e9 / voxels);
        }
        best
    };

    let mut rows: Vec<(String, f64, f64)> = Vec::new();

    // the unit
    {
        let op = VoxelwiseMapOp::threshold("map", 500.0, 1.0, 0.0);
        let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
        let input = input.clone();
        let anchor = Anchor::whole(shape);
        rows.push((
            "voxelwise map".to_string(),
            best_of(Box::new(move || {
                op.apply(&input, &mut out, &anchor).unwrap();
            })),
            1.0,
        ));
    }

    // the sink
    {
        let combine = DifferenceCombine::new("difference");
        let operands = [input.clone(), ramp(shape)];
        let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
        let anchor = Anchor::whole(shape);
        rows.push((
            "difference combine (two branch results)".to_string(),
            best_of(Box::new(move || {
                combine.apply(&operands, &mut out, &anchor).unwrap();
            })),
            1.0,
        ));
    }

    // the estimate and the whole diamond, at two element sizes
    for radius in [1usize, 3] {
        let element =
            StructuringElement::from_radius(super::element::ElementShape::Ellipsoid, [radius; 3]);
        let passes = Morphology::Open.reach_factor() as f64;
        for (label, chain) in [
            ("estimate", background_estimate(&element)),
            ("remove", remove_background(&element).unwrap()),
        ] {
            let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
            let input = input.clone();
            let anchor = Anchor::whole(shape);
            rows.push((
                format!("{label}, {}-voxel element", element.len()),
                best_of(Box::new(move || {
                    chain.apply(&input, &mut out, &anchor).unwrap();
                })),
                passes * element.len() as f64,
            ));
        }
    }

    let unit = rows.first().map(|(_, nanos, _)| *nanos).unwrap_or(1.0);
    let mut out = format!(
        "background op cost, {}x{}x{}, best of {repetitions}\n{:<44} {:>10} {:>10} {:>14}\n",
        shape[0], shape[1], shape[2], "op", "ns/voxel", "relative", "per element"
    );
    for (name, nanos, divisor) in rows {
        out.push_str(&format!(
            "{name:<44} {nanos:>10.3} {:>10.2} {:>14.3}\n",
            nanos / unit,
            nanos / unit / divisor.max(1e-12)
        ));
    }
    out
}

fn ramp(shape: [usize; 3]) -> Voxels {
    let mut array = ndarray::Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
    for (flat, value) in array.iter_mut().enumerate() {
        *value = ((flat * 7919) % 1013) as f64;
    }
    array.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    use super::super::element::ElementShape;

    const SHAPE: [usize; 3] = [9, 8, 7];

    fn speckle() -> Array3<f64> {
        Array3::from_shape_fn((SHAPE[0], SHAPE[1], SHAPE[2]), |(i, j, k)| {
            (((i * 31 + j * 17 + k * 7) % 23) as f64) / 23.0 + (i as f64) * 0.1
        })
    }

    fn applied(chain: &Chain, input: &Array3<f64>) -> Array3<f64> {
        let source: Voxels = input.clone().into();
        let mut out = Voxels::zeros(Dtype::F64, SHAPE).unwrap();
        chain
            .apply(&source, &mut out, &Anchor::whole(SHAPE))
            .unwrap();
        out.view::<f64>().unwrap().to_owned()
    }

    /// The definition, written out with its own loops, against the composition
    /// this module assembles. Not a second implementation for production — a
    /// statement of what a top-hat *is*, so that "it is composition" is checked
    /// against the operation rather than against itself.
    fn by_definition(input: &Array3<f64>, element: &StructuringElement) -> Array3<f64> {
        let shape = input.dim();
        let sweep = |source: &Array3<f64>, take_max: bool| {
            Array3::from_shape_fn(shape, |(i, j, k)| {
                let mut chosen: Option<f64> = None;
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
                    let value = source[[a, b, c]];
                    chosen = Some(match chosen {
                        None => value,
                        Some(best) if take_max => best.max(value),
                        Some(best) => best.min(value),
                    });
                }
                chosen.expect("an element contains its own centre")
            })
        };
        let opened = sweep(&sweep(input, false), true);
        Array3::from_shape_fn(shape, |(i, j, k)| input[[i, j, k]] - opened[[i, j, k]])
    }

    #[test]
    fn the_assembled_diamond_is_the_top_hat_by_definition() {
        let input = speckle();
        for shape in [ElementShape::Box, ElementShape::Ellipsoid] {
            for radius in [[1, 1, 1], [2, 1, 0], [0, 0, 3], [2, 2, 2]] {
                let element = StructuringElement::from_radius(shape, radius);
                let got = applied(&remove_background(&element).unwrap(), &input);
                assert_eq!(got, by_definition(&input, &element), "{shape:?} {radius:?}");
                assert!(
                    got.iter().any(|&value| value > 0.0),
                    "{shape:?} {radius:?} removed nothing, so the comparison is vacuous"
                );
            }
        }
    }

    /// **The origin of the element's step reaches the estimate, and nothing here
    /// carries it.**
    ///
    /// An element whose step counts from `StepOrigin::ClippedStart` reads a
    /// different set of offsets where the window is clipped at a low face of the
    /// volume. This file has no gather, so there is nothing here that could get
    /// that right or wrong — both arms are `RankFilterOp`, which asks the element
    /// what it reads at each voxel's position in the volume. The definition below
    /// is written out with `offsets_at` for that reason: it states what the
    /// composed operation means and lets the composition be checked against it.
    ///
    /// The second half is what keeps the first from being vacuous. Under the
    /// other origin the same box is a different opening on this volume, so the
    /// comparison above is over a case where the two rules disagree.
    #[test]
    fn the_estimate_honours_the_step_origin_through_the_filter_it_is_made_of() {
        use super::super::element::StepOrigin;

        let input = speckle();
        let at = Anchor::whole(SHAPE);
        let size = [9, 5, 1];
        let step = [2, 2, 1];
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

        let got = applied(&remove_background(&clipped).unwrap(), &input);
        assert_eq!(got, by_definition_at(&input, &at, &clipped));
        assert_ne!(
            got,
            by_definition(&input, &clipped),
            "the anchored gather must be a different top-hat here, or the comparison above \
             is a comparison of one rule with itself"
        );
        assert_ne!(
            got,
            applied(&remove_background(&anchored).unwrap(), &input),
            "and the two origins must be two operations through the composition too"
        );
    }

    /// [`by_definition`] with the window asked of the element **at each voxel's
    /// position in the volume**, which is what the operation means for an element
    /// whose offsets are not one set. Identical to it for every other element.
    fn by_definition_at(
        input: &Array3<f64>,
        at: &Anchor,
        element: &StructuringElement,
    ) -> Array3<f64> {
        let shape = input.dim();
        let sweep = |source: &Array3<f64>, take_max: bool| {
            let mut scratch = Vec::new();
            let mut out = Array3::zeros(shape);
            for i in 0..shape.0 {
                for j in 0..shape.1 {
                    for k in 0..shape.2 {
                        let placed = [
                            (i + at.offset[0]) as isize,
                            (j + at.offset[1]) as isize,
                            (k + at.offset[2]) as isize,
                        ];
                        let mut chosen: Option<f64> = None;
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
                            let value = source[[a, b, c]];
                            chosen = Some(match chosen {
                                None => value,
                                Some(best) if take_max => best.max(value),
                                Some(best) => best.min(value),
                            });
                        }
                        out[[i, j, k]] = chosen.expect("a window that met the volume");
                    }
                }
            }
            out
        };
        let opened = sweep(&sweep(input, false), true);
        Array3::from_shape_fn(shape, |(i, j, k)| input[[i, j, k]] - opened[[i, j, k]])
    }

    /// The property that makes the difference a *residual* rather than a signed
    /// quantity, and it is a theorem about the element rather than a clamp: the
    /// element contains its centre and is symmetric, so the opening cannot
    /// exceed the original anywhere, including where the boundary truncates it.
    #[test]
    fn the_estimate_never_exceeds_the_original_so_the_difference_is_non_negative() {
        let input = speckle();
        for shape in [ElementShape::Box, ElementShape::Ellipsoid] {
            for radius in [[1, 1, 1], [3, 2, 1], [2, 2, 2]] {
                let element = StructuringElement::from_radius(shape, radius);
                // the two facts the argument rests on, checked rather than
                // assumed of the constructor
                assert!(element.offsets().contains(&[0, 0, 0]));
                for offset in element.offsets() {
                    assert!(
                        element
                            .offsets()
                            .contains(&[-offset[0], -offset[1], -offset[2]]),
                        "{shape:?} {radius:?} is not symmetric at {offset:?}"
                    );
                }

                let estimate = applied(&background_estimate(&element), &input);
                let difference = applied(&remove_background(&element).unwrap(), &input);
                for (position, value) in difference.indexed_iter() {
                    assert!(
                        *value >= 0.0,
                        "{shape:?} {radius:?}: {value} at {position:?}"
                    );
                }
                for (want, got) in input.iter().zip(estimate.iter()) {
                    assert!(got <= want, "{shape:?} {radius:?}: {got} > {want}");
                }
            }
        }
    }

    /// An opening is idempotent. Nothing in this file arranges for that — it
    /// follows from the composition being a real opening — so it is the evidence
    /// that `rank(lowest)` then `rank(highest)` is the operation this module
    /// claims and not merely two filters.
    #[test]
    fn the_estimate_is_idempotent_which_is_what_makes_it_an_opening() {
        let input = speckle();
        let element = StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 2, 2]);
        let once = applied(&background_estimate(&element), &input);
        let twice = applied(&background_estimate(&element), &once);
        assert_eq!(once, twice);
    }

    /// Two statements of one quantity, checked against each other.
    #[test]
    fn the_reach_is_twice_the_element_and_the_two_statements_agree() {
        let volume = [64usize, 64, 64];
        for shape in [ElementShape::Box, ElementShape::Ellipsoid] {
            for radius in [[1, 1, 1], [5, 0, 2], [3, 4, 7]] {
                let element = StructuringElement::from_radius(shape, radius);
                let declared = background_reach(&element);
                assert_eq!(
                    declared,
                    [radius[0] * 2, radius[1] * 2, radius[2] * 2],
                    "{shape:?} {radius:?}"
                );
                assert_eq!(background_estimate(&element).reach3(&volume), declared);
                assert_eq!(
                    remove_background(&element).unwrap().reach3(&volume),
                    declared,
                    "the sink reaches zero, so the fan-in reaches its widest arm"
                );
            }
        }
    }

    /// The same pair of statements for an element with no centre voxel, where
    /// the symmetric triple is a bound and the pair is the truth.
    ///
    /// The point of asserting the fold rather than the formula: the chain gets
    /// its answer by adding two `RankFilterOp` specs side by side, and this file
    /// gets its answer from the composition factor. If either drifted — a
    /// sequence that took a max instead of a sum, or a factor that stopped being
    /// two — the two would stop agreeing here rather than at a seam.
    #[test]
    fn the_per_side_reach_of_an_off_centre_element_is_twice_each_side() {
        let volume = [64usize, 64, 64];
        for shape in [
            ElementShape::Box,
            ElementShape::Ellipsoid,
            ElementShape::ExtentEllipsoid,
        ] {
            for size in [[10, 5, 4], [6, 6, 6], [2, 9, 3]] {
                let element = StructuringElement::from_size(shape, size).unwrap();
                let declared = background_reach_spec(&element);
                for axis in 0..3 {
                    let (lo, hi) = element.sides(axis);
                    assert_eq!(declared.at(axis, 0, volume[axis]), (2 * lo, 2 * hi));
                }
                assert_eq!(
                    background_estimate(&element).reach_spec(volume).unwrap(),
                    declared,
                    "{shape:?} {size:?}"
                );
                assert_eq!(
                    remove_background(&element)
                        .unwrap()
                        .reach_spec(volume)
                        .unwrap(),
                    declared,
                    "the sink reaches zero, so the fan-in reaches its widest arm"
                );
                // and the triple stays a bound on it, which is what
                // `Chain::reach_spec` checks rather than assumes
                let bound = background_reach(&element);
                for axis in 0..3 {
                    let (lo, hi) = declared.at(axis, 0, volume[axis]);
                    assert!(lo.max(hi) <= bound[axis], "{shape:?} {size:?} axis {axis}");
                }
            }
        }
    }

    /// Branch order is the operation. If a future change reordered the branches
    /// — or if `Chain::Parallel` stopped preserving the order — the answer would
    /// be the negation of this one and nothing about the types would notice.
    #[test]
    fn the_arms_are_ordered_and_swapping_them_negates_the_answer() {
        let input = speckle();
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let forwards = applied(&remove_background(&element).unwrap(), &input);
        let backwards = applied(
            &Chain::parallel(
                vec![
                    background_estimate(&element),
                    Chain::op(VoxelwiseMapOp::identity(ORIGINAL)),
                ],
                Box::new(DifferenceCombine::new(DIFFERENCE)),
            )
            .unwrap(),
            &input,
        );
        assert!(forwards.iter().any(|&value| value > 0.0));
        for (plus, minus) in forwards.iter().zip(backwards.iter()) {
            assert_eq!(*plus, -*minus);
        }
    }

    /// The declaration, in bits, against the computed answer — in both element
    /// types the combine accepts, because the `f32` path is the one a `f64`
    /// declaration could be narrowed into.
    #[test]
    fn a_constant_field_maps_to_positive_zero_in_bits() {
        let element = StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 2, 2]);
        let chain = remove_background(&element).unwrap();
        for constant in [0.0_f64, -0.0, 1.0, -3.5, 1e300, f64::MIN_POSITIVE] {
            let declared = chain
                .constant_maps_to(constant)
                .unwrap_or_else(|| panic!("nothing declared for {constant}"));
            assert_eq!(
                declared.to_bits(),
                0.0_f64.to_bits(),
                "{constant} declared {declared}, whose bits are not those of +0.0"
            );

            let input: Voxels = Array3::from_elem((5, 5, 5), constant).into();
            let mut out = Voxels::zeros(Dtype::F64, [5, 5, 5]).unwrap();
            chain
                .apply(&input, &mut out, &Anchor::whole([5, 5, 5]))
                .unwrap();
            for value in out.view::<f64>().unwrap().iter() {
                assert_eq!(
                    value.to_bits(),
                    0.0_f64.to_bits(),
                    "computing {constant} gave {value}"
                );
            }
        }

        // and the same through the `f32` kernel, which is where a narrowed
        // declaration would have had to survive a double rounding
        let combine = DifferenceCombine::new("difference");
        for constant in [1.0_f32, -3.5, 1e30] {
            let operand: Voxels = Array3::from_elem((3, 3, 3), constant).into();
            let mut out = Voxels::zeros(Dtype::F32, [3, 3, 3]).unwrap();
            combine
                .apply(
                    &[operand.clone(), operand],
                    &mut out,
                    &Anchor::whole([3, 3, 3]),
                )
                .unwrap();
            for value in out.view::<f32>().unwrap().iter() {
                assert_eq!(
                    value.to_bits(),
                    0.0_f32.to_bits(),
                    "{constant} gave {value}"
                );
            }
        }
    }

    /// What is not exactly true is not declared, and the two exclusions are the
    /// ones the header argues for.
    #[test]
    fn nothing_is_declared_for_unequal_or_non_finite_operands() {
        let combine = DifferenceCombine::new("difference");
        assert_eq!(combine.constant_maps_to(&[2.5, 2.5]), Some(0.0));
        assert_eq!(combine.constant_maps_to(&[2.5, 1.0]), None);
        assert_eq!(
            combine.constant_maps_to(&[f64::INFINITY, f64::INFINITY]),
            None
        );
        assert_eq!(combine.constant_maps_to(&[f64::NAN, f64::NAN]), None);
        // +0.0 and -0.0 are equal as numbers and differ as bits, and the
        // difference of the two is -0.0 rather than +0.0
        assert_eq!(combine.constant_maps_to(&[-0.0, 0.0]), None);
        assert_eq!(combine.constant_maps_to(&[-0.0, -0.0]), Some(0.0));
        assert_eq!(combine.constant_maps_to(&[1.0]), None);
        assert_eq!(combine.constant_maps_to(&[1.0, 1.0, 1.0]), None);
    }

    #[test]
    fn the_combine_refuses_what_it_cannot_join() {
        let combine = DifferenceCombine::new("difference");
        assert!(combine.accepts(&[Dtype::F64, Dtype::F64]));
        assert!(combine.accepts(&[Dtype::F32, Dtype::F32]));
        assert!(!combine.accepts(&[Dtype::F64, Dtype::F32]));
        assert!(!combine.accepts(&[Dtype::U16, Dtype::U16]));
        assert!(!combine.accepts(&[Dtype::Bool, Dtype::Bool]));
        assert!(!combine.accepts(&[Dtype::F64]));
        assert!(!combine.accepts(&[Dtype::F64, Dtype::F64, Dtype::F64]));

        let err = combine
            .output_shape(&[[4, 4, 4], [4, 4, 3]])
            .unwrap_err()
            .to_string();
        assert!(err.contains("no such pairing"), "got: {err}");
        let err = combine
            .output_shape(&[[4, 4, 4], [4, 4, 4], [4, 4, 4]])
            .unwrap_err()
            .to_string();
        assert!(err.contains("minuend"), "got: {err}");

        // and the refusal reaches the plan, which is where it is useful
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let err = remove_background(&element)
            .unwrap()
            .produces(Dtype::U16)
            .unwrap_err()
            .to_string();
        assert!(!err.is_empty());
    }

    /// The element is what a background estimate costs, so the price must move
    /// with it — that is why `rank`'s constant is per element voxel and why a
    /// flat figure here would misprice the case this op exists for.
    #[test]
    fn a_larger_element_is_priced_as_a_larger_element() {
        let small = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let large = StructuringElement::from_radius(ElementShape::Box, [4, 4, 4]);
        let cheap = remove_background(&small).unwrap().cost_per_voxel();
        let dear = remove_background(&large).unwrap().cost_per_voxel();
        assert!(dear > cheap * 10.0, "{dear} against {cheap}");

        // the fan-in is priced as the sum of its arms plus the sink, so the
        // difference between the two is the estimate's alone
        let ratio = (dear - cheap) / (large.len() - small.len()) as f64;
        assert!(ratio > 0.0 && ratio < 100.0, "per element voxel: {ratio}");

        // and the sink is a voxelwise cost, not a neighbourhood one
        let sink = DifferenceCombine::new("difference");
        assert!(sink.cost_per_voxel(2) < cheap);
        assert_eq!(sink.cost_per_voxel(2), DIFFERENCE_COST);
    }

    /// Retaking the measurement. Ignored because timing inside a test suite
    /// measures the machine's mood rather than the code, but it is here and it
    /// is one command:
    ///
    /// ```text
    /// cargo test --release -- --ignored --nocapture ops::background
    /// ```
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn print_the_cost_table() {
        println!("{}", cost_report([96, 64, 64], 5));
    }
}
