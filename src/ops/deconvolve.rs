// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the iteration —
// forward-project, compare, back-project, multiply — not adapted from any
// implementation of it.
//
// Iterative deconvolution: recover an estimate whose blur by a **known** kernel
// agrees with what was observed.
//
// Which of the two families this is, and why
// ------------------------------------------
// Deconvolution comes in two families whose reach could not be more different,
// and the choice between them is the whole design of this file.
//
// | family | how it inverts | what it reads to write one voxel |
// |---|---|---|
// | **frequency domain** (Wiener, or an FFT-accelerated iterative form) | one global transform, a per-frequency division, one inverse transform | **the whole volume** — a Fourier coefficient is a sum over every voxel |
// | **spatial domain, iterative** (the one implemented here) | repeated correlation against the kernel itself | a compact neighbourhood per iteration, `2r` per axis, `2rn` after `n` of them |
//
// This module implements the **spatial** form, and the reason is what a global
// dependency costs *this* framework rather than a general preference. A reach
// that spans an axis is [`crate::reach::AxisReach::All`], which is a planning
// barrier **by type**: `decomposition` resolves it to a single block, so the
// volume must fit in memory at once — which is the one thing an out-of-core
// framework exists not to require — and a distributed run of it hands one worker
// everything and the rest nothing. Its cost is under-priced on top of that: a
// `cost_per_voxel` multiplied by the voxel count is a linear model, and a
// transform is `n log n` with a constant nothing here would have measured.
//
// **There used to be a third argument here and it has expired.** It read: there
// is no FFT in the dependency list, and `Cargo.toml` says every entry is
// something the *framework* needs, so an op that requires one cannot be added
// without changing what the crate is. That was true when this file was written
// and it is not true now — [`crate::ops::fft`] is a two-dimensional real
// transform over `rustfft` and `realfft`, and `Cargo.toml` now says in as many
// words that an FFT is what one *op* needs rather than what the framework needs.
// So the dependency argument is gone, and what remains is the two paragraphs
// above it: the planning barrier and the cost model. Those were always the real
// reasons and they are untouched.
//
// The spatial form gives all of that up for a stated price: it is iterative
// rather than closed-form, so it approaches the answer instead of computing it,
// and its reach grows **linearly with the iteration count**. At `n` iterations of
// a kernel of radius `r` it reaches `2rn`, which is a real number that a caller
// can make too large — and when they do, the guard says so at planning time
// instead of a transform silently serialising the run.
//
// **What the frequency-domain form would need, and which of it now exists.**
// Still unbuilt, and still deliberately — but the list is shorter than it was
// and the honest answer to "is it buildable now?" is **yes**, so the list is
// written as a specification rather than as an obstacle.
//
// | what it needs | state |
// |---|---|
// | a transform, with the zero padding and the real-input symmetry a volume of arbitrary extent needs | **done**, in [`crate::ops::fft`], for **two** axes. Its normalisation convention is stated there: forward unnormalised, inverse carries `1/N` |
// | the **third** axis | ~~**missing.**~~ **built**, and by exactly the route this row predicted: [`crate::ops::fft::RealTransform3`] is a real transform along the last axis and `transform_lanes` along the other two, because a `[d0, d1, w]` row-major buffer *is* a `[d0, d1 * w]` one to a first-axis pass. The row's estimate — "the same twenty lines, over `Array3`" — held |
// | a place to put the spectrum | **missing, and this is the awkward one.** [`crate::voxels::Voxels`] has eleven element variants and none of them is complex, so a `BlockOp` cannot hand a spectrum to the next op. The frequency-domain form would therefore have to be *one* op that transforms, divides and inverse-transforms internally — which it is anyway, since there is nothing useful to do with a spectrum in this pipeline — and never expose one → **this clause was the useful one and it has since been acted on.** The ops survey's G3 row was a request for the complex element type; it has been **refused** on this clause's own argument, and [`crate::ops::convolve::TransformConvolveOp`] is the demonstration — a frequency-domain `BlockOp` with a bounded reach, decomposition-invariant to the bit, whose spectrum never leaves one `apply`. So this row is not an obstacle to a frequency-domain deconvolution and never was |
// | `reach_spec` returning [`crate::reach::Reach::all`] | **unchanged.** The barrier declared rather than detected, exactly as [`crate::ops::watershed`] declares its own |
// | a `cost_per_voxel` honest about `log n` | **unchanged, and still the binding problem.** The trait's per-voxel shape cannot express it, so the cost model has to grow a term before the planner can compare a transform against anything. Until it does, an FFT deconvolution would be priced as though it were linear, and `docs/design/BLOCK_OPS.md` is blunt about what a wrong model costs: the search returns the optimum for whatever model it is given |
// | the padding rule for a *convolution* rather than a correlation | **available.** [`crate::ops::fft::minimal_wrap_free_length`] is stated for a lag window; a linear convolution with a kernel of radius `r` is the same rule with the window `[-r, r]`, which is where a spatial-domain kernel's own reach already lives |
//
// So what stops it is no longer the dependency list; it is the cost model and the
// planning barrier, which is where the argument at the top of this file put it in
// the first place. Nothing of it is started here, and this table is what a later
// worker should read before starting.
//
// The iteration
// -------------
// One step, per voxel, with `d` the observation, `u` the current estimate, `h`
// the kernel and `h*` its reflection:
//
// ```text
// u <- u . ( (d / (u (*) h)) (*) h* )
// ```
//
// Read left to right that is: blur the estimate the way the instrument would
// have; divide the observation by it to get a per-voxel correction ratio;
// spread that ratio back through the same kernel reflected; scale the estimate
// by it. The estimate starts as the observation itself — a fixed point of the
// step wherever the observation already agrees with its own blur, which is
// everywhere the volume is flat — and that initialisation is what the reach
// below is derived for; a flat starting estimate would reach `r(2n-1)` instead.
//
// Every step multiplies a non-negative estimate by a non-negative correction, so
// an estimate that starts non-negative stays non-negative for a non-negative
// observation. That is why this family is used at all, and it is why the kernel
// here **requires non-negative weights** rather than accepting any: with a
// negative tap the blurred estimate can vanish where the estimate does not, and
// the guarded division would be carrying the model rather than protecting it.
//
// The reach, and where its three factors come from
// ------------------------------------------------
// `reach = 2 * radius * iterations`, per axis. Derived, in
// [`Deconvolution::reach`], from the two things it follows from and from nothing
// else — there is no field that sets it and no argument that could feed it the
// halo.
//
// The derivation is one recursion. To write `u'(v)` the step above reads:
//
// * `u(v)` itself, for the leading factor;
// * the ratio within `r` of `v`, because the back-projection is a correlation;
// * and each of those ratios reads the blurred estimate at its own voxel, which
//   reads `u` within another `r`.
//
// So one step reads `u` within `2r` and `d` within `r`. With `u` initialised to
// `d`, `n` steps read `d` within `2rn` — the other term, `2r(n-1) + r`, is
// strictly smaller and never binds.
//
// It is **tight**, and both factors are visible in the test that checks it. The
// outermost tap of the kernel is required to be non-zero (`PointSpread::new`
// refuses a kernel whose end tap is zero, because such a kernel *is* a narrower
// kernel and would make the reach a voxel wider than what the answer depends
// on), so the chain above genuinely reaches `2rn`: understating it by one, on
// any axis, at any iteration count, changes the answer at the seams.
//
// Why this is one op rather than `n` of them in a `Chain::sequence`
// ----------------------------------------------------------------
// It looks like a loop of identical steps and it is not one, in a way that is a
// fact about the algorithm rather than about this crate. The step has **two**
// operands — the running estimate and the fixed observation — and it reads the
// observation afresh every time. `Chain::Sequence` hands each child exactly one
// buffer, the previous child's output, so a chain of `n` one-step ops would feed
// step `k` its own predecessor's estimate *as the observation*, which is a
// different and much worse algorithm that converges to something else.
// `Chain::Parallel` does not help either: its branches share one input and are
// joined once, so it expresses a diamond and not a recurrence.
//
// So the iteration count is a parameter of one op, the reach is derived from it,
// and the loop lives inside `apply` where the two operands can both be in scope.
// A framework that wanted the other form would need a chain node that carries a
// second, invariant input past its children — which is a change to `Chain`, not
// to an op.
//
// Edge behaviour
// --------------
// Both correlations clamp at the array they are handed, which is the convention
// every neighbourhood op in this module already uses and which the shared
// separable pass in `ridge` implements. At a real volume boundary that is the
// whole story and the whole-volume reference clamps identically; at a block seam
// it is deliberately wrong, and the halo guard is what converts that into a loud
// failure instead of a quiet one.
//
// Costs
// -----
// Measured; see [`cost_report`], which is runnable and prints the table the
// constants at the bottom of this file were read off. The figure
// `cost_per_voxel` returns is for the **whole op**, all iterations included,
// because that is what one `apply` costs; the per-iteration figure is the same
// number divided by the iteration count, and [`cost_for`] is written that way
// round so it is visible which is which.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp};
use crate::voxels::Voxels;

use super::ridge::{gaussian_smooth_into, gaussian_weights};
use super::shapes_agree;

// --------------------------------------------------------------- kernel --

/// The separable point-spread kernel the observation is assumed to have been
/// blurred by: one set of weights per axis, each odd-length and centred.
///
/// **Separable, and that is a restriction rather than an approximation.** A
/// general 3-D kernel of radius `r` costs `(2r+1)^3` taps per voxel per pass
/// where a separable one costs `3(2r+1)`, and at `r = 4` that is 729 against 27.
/// What it cannot express is a kernel that is not an outer product — a diagonal
/// streak, say. Such a kernel is a different type with the same reach
/// arithmetic, and nothing here would have to change to admit one.
///
/// **The weights are the caller's.** [`Self::gaussian`] is offered because it is
/// the shape most instruments are described by and because this crate already
/// has an exactly-symmetric, normalised generator for it — but it is a
/// constructor, not a default, and [`Self::new`] takes whatever the caller
/// measured.
///
/// Three things are checked rather than assumed, and each rules out a way the
/// arithmetic below stops meaning anything:
///
/// * **Odd length.** The kernel has to have a centre for "the estimate at `v`"
///   and "the blurred estimate at `v`" to be about the same voxel. An even
///   kernel shifts the estimate by half a voxel every iteration.
/// * **Non-negative, and summing to something positive.** The iteration is a
///   product of ratios of these numbers; a negative tap makes the blurred
///   estimate able to be zero where the estimate is not, and the division's
///   guard would then be doing the work the model should be doing.
/// * **A non-zero outermost tap.** A kernel whose end tap is zero *is* a
///   narrower kernel written with padding, and the reach derived from its stated
///   radius would be wider than what the answer depends on — which would make
///   the tightness test below fail for a reason that is nobody's bug. Refused
///   where it is written instead.
///
/// The weights are normalised to sum to one on each axis, which is what makes
/// the blur of a constant approximately that constant and what the
/// non-divergence argument on [`deconvolve_into`] rests on. "Approximately" is
/// exact about itself: see [`DeconvolveOp::constant_maps_to`].
#[derive(Debug, Clone, PartialEq)]
pub struct PointSpread {
    weights: [Vec<f64>; 3],
    reflected: [Vec<f64>; 3],
}

impl PointSpread {
    /// A kernel from explicit per-axis weights.
    pub fn new(weights: [Vec<f64>; 3]) -> Result<Self> {
        let mut normalised: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (axis, taps) in weights.into_iter().enumerate() {
            if taps.len() % 2 != 1 {
                return Err(Error::InvalidArgument(format!(
                    "a point spread on axis {axis} has {} taps. It has to have a centre — an \
                     even kernel displaces the estimate by half a voxel every iteration — so \
                     the count is odd.",
                    taps.len()
                )));
            }
            if !taps
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0)
            {
                return Err(Error::InvalidArgument(format!(
                    "a point spread on axis {axis} has a negative or non-finite tap. The \
                     iteration is a product of ratios of these numbers, and a negative tap makes \
                     the blurred estimate able to vanish where the estimate does not."
                )));
            }
            let total: f64 = taps.iter().sum();
            if !(total > 0.0) || !total.is_finite() {
                return Err(Error::InvalidArgument(format!(
                    "a point spread on axis {axis} sums to {total}, so there is nothing to \
                     normalise it by."
                )));
            }
            let ends = taps.first().copied().unwrap_or(0.0) + taps.last().copied().unwrap_or(0.0);
            if taps.len() > 1 && !(ends > 0.0) {
                return Err(Error::InvalidArgument(format!(
                    "a point spread on axis {axis} has zero weight at both ends. That is a \
                     narrower kernel written with padding, and the reach derived from the stated \
                     radius would be wider than what the answer depends on — so it is refused \
                     here rather than silently loosening the halo."
                )));
            }
            normalised[axis] = taps.iter().map(|weight| weight / total).collect();
        }
        let reflected = [
            reflect(&normalised[0]),
            reflect(&normalised[1]),
            reflect(&normalised[2]),
        ];
        Ok(Self {
            weights: normalised,
            reflected,
        })
    }

    /// A Gaussian kernel of a per-axis standard deviation, truncated at a stated
    /// multiple of it.
    ///
    /// Per-axis rather than scalar for the reason `ridge` states: a volume's
    /// voxels are not required to be cubes, and which sigma belongs on which
    /// axis is the caller's knowledge.
    ///
    /// The weights come from the generator `ridge` already has, which builds
    /// them from `x * x` so that the tap at `-k` and the tap at `+k` are the
    /// same bits. That matters here for a different reason than it does there: a
    /// bit-symmetric kernel is its own reflection, so the forward pass and the
    /// back-projection below are the same arithmetic and an asymmetry in the
    /// result cannot have come from the kernel.
    pub fn gaussian(sigma: [f64; 3], truncate: f64) -> Result<Self> {
        Self::new([
            gaussian_weights(sigma[0], truncate)?,
            gaussian_weights(sigma[1], truncate)?,
            gaussian_weights(sigma[2], truncate)?,
        ])
    }

    /// How far the kernel extends from its centre, on `axis`.
    pub fn radius(&self, axis: usize) -> usize {
        self.weights[axis].len() / 2
    }

    /// The normalised weights on `axis`, lowest offset first.
    pub fn weights(&self, axis: usize) -> &[f64] {
        &self.weights[axis]
    }

    /// The weight of the centre tap of the whole 3-D kernel: the product of the
    /// three centre taps.
    ///
    /// Not decoration — it is the constant in the bound that keeps the iteration
    /// from running away. See [`deconvolve_into`].
    pub fn centre_weight(&self) -> f64 {
        (0..3)
            .map(|axis| self.weights[axis][self.radius(axis)])
            .product()
    }

    /// How many taps one separable forward pass evaluates per voxel, summed over
    /// the three axes. The unit the measured cost is stated in.
    pub fn taps(&self) -> usize {
        (0..3).map(|axis| self.weights[axis].len()).sum()
    }

    fn forward(&self) -> &[Vec<f64>; 3] {
        &self.weights
    }

    fn backward(&self) -> &[Vec<f64>; 3] {
        &self.reflected
    }
}

/// The reflection of a kernel: what turns a correlation into its adjoint.
fn reflect(weights: &[f64]) -> Vec<f64> {
    weights.iter().rev().copied().collect()
}

/// Blur `input` with `spread`, writing into `out`.
///
/// The **forward model**, offered publicly because a caller who wants to check a
/// deconvolution has to be able to state what it is inverting, and because the
/// only honest way to test a restoration is to blur something whose original is
/// known.
///
/// Generic over the element type as far as the algorithm allows: a correlation
/// accumulates, so it needs a way into `f64` and nothing else.
///
/// One separable pass per axis, in axis order 0, 1, 2, clamped at the array's
/// edge. It is the pass `ridge` already implements and is called rather than
/// rewritten: two implementations of one summation would agree everywhere except
/// in the last bit, which is exactly the disagreement a byte-identity suite
/// exists to catch and would then be reporting about the test rather than about
/// the code.
pub fn blur_into<T>(
    input: ArrayView3<'_, T>,
    spread: &PointSpread,
    out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    shapes_agree(input.shape(), out.shape(), "blur_into")?;
    gaussian_smooth_into(input, spread.forward(), out)
}

// ------------------------------------------------------------ parameters --

/// Everything the iteration is determined by: the kernel, and how many times to
/// apply the step.
///
/// One type holding both, because the reach is a function of both and a reach
/// assembled from two separately-held parameters is a reach that can be
/// assembled from the wrong two.
#[derive(Debug, Clone, PartialEq)]
pub struct Deconvolution {
    spread: PointSpread,
    iterations: usize,
}

impl Deconvolution {
    /// At least one iteration, and few enough that the reach is a number.
    ///
    /// **Zero is refused.** An op that returns its input is an identity, and an
    /// identity that is spelled as a deconvolution reaches zero, costs what it
    /// costs, and quietly does nothing — while every guarantee this file makes
    /// about the output (that it is finite, whatever came in) is a guarantee
    /// about a value that went through the step at least once.
    ///
    /// The upper limit is not a policy: `2 * radius * iterations` is computed in
    /// `usize`, and a count that overflows it is refused where it is written
    /// rather than wrapping somewhere the halo is derived.
    pub fn new(spread: PointSpread, iterations: usize) -> Result<Self> {
        if iterations == 0 {
            return Err(Error::InvalidArgument(
                "a deconvolution runs at least one iteration; zero of them is an identity \
                 wearing this op's name, its reach, and its cost."
                    .to_string(),
            ));
        }
        for axis in 0..3 {
            spread
                .radius(axis)
                .checked_mul(2)
                .and_then(|both| both.checked_mul(iterations))
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "a radius of {} on axis {axis} over {iterations} iterations reaches \
                         further than a `usize` counts. The reach is derived from these two \
                         numbers, so they are refused here rather than overflowing where a halo \
                         is derived from them.",
                        spread.radius(axis)
                    ))
                })?;
        }
        Ok(Self { spread, iterations })
    }

    pub fn spread(&self) -> &PointSpread {
        &self.spread
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// **The derived reach**: `2 * radius * iterations`, on `axis`.
    ///
    /// The three factors and no fourth. Nothing configures it, nothing may set
    /// it, and it does not consult the volume — which is what makes this op a
    /// local one rather than the planning barrier the frequency-domain family
    /// would be.
    ///
    /// The `2` is the round trip: one step blurs the estimate and correlates the
    /// ratio back, so it reads the estimate `radius` away through the first and
    /// another `radius` away through the second. The `iterations` is that step
    /// composed with itself, and reaches compose by addition — the same
    /// arithmetic `Chain::Sequence` does, done here because the recurrence
    /// cannot be written as a sequence (see this module's header).
    pub fn reach(&self, axis: usize) -> usize {
        2 * self.spread.radius(axis) * self.iterations
    }
}

// --------------------------------------------------------------- kernel --

/// Deconvolve `observed` under `parameters`, writing the estimate into `out`.
///
/// Generic over the element type as far as the algorithm allows — the
/// observation is read and immediately accumulated, so `Copy + Into<f64>` is
/// the whole requirement — while everything the iteration holds is `f64` and has
/// to be: a ratio of intensities is fractional whatever the intensities were,
/// which is the reason `local.rs` gives for its own accumulator and the same one
/// here.
///
/// **The estimate starts as the observation.** That is a decision with a
/// consequence in the reach — a flat initial estimate would reach `r(2n-1)`
/// instead of `2rn` — and it is the initialisation the derivation above and the
/// declared reach are both for.
///
/// Division, and the two ways it could produce a NaN
/// -------------------------------------------------
/// The step divides, and a NaN would be worse here than any wrong number: it
/// compares unequal to itself, so a volume containing one makes every
/// byte-identity check in this crate pass vacuously rather than fail. Both
/// routes to one are closed, in the arithmetic rather than in a comment:
///
/// * **`0 / 0`.** A blurred estimate of zero is a real case, not a pathological
///   one — it is every voxel of an empty region — and the observation there is
///   zero too. The ratio is taken only where the denominator is strictly
///   positive and is zero elsewhere, which is also the right answer on its own
///   terms: where the model predicts nothing was received, there is no
///   correction to apply.
/// * **`0 * inf`.** Reachable only through an overflow, and the multiplicative
///   update is what would have to overflow. It is bounded: every tap is
///   non-negative and the weights sum to one, so the blurred estimate at `v` is
///   at least `w_c * u(v)` where `w_c` is [`PointSpread::centre_weight`], hence
///   the ratio is at most `1 / w_c` and one step multiplies the estimate by at
///   most that. Overflowing therefore needs `max(d) / w_c^n` past `1e308`, which
///   is an input this file will not pretend to have an opinion about — so the
///   update is checked and a non-finite one is recorded as zero. A wrong number
///   that compares equal to itself, deliberately, over a value that would make
///   every equality in the suite meaningless.
///
/// Together those make a promise worth stating: **the output is finite for any
/// input**, including one carrying NaNs or infinities of its own. There is a
/// test.
pub fn deconvolve_into<T>(
    observed: ArrayView3<'_, T>,
    parameters: &Deconvolution,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    shapes_agree(observed.shape(), out.shape(), "deconvolve_into")?;
    let dim = (
        observed.shape()[0],
        observed.shape()[1],
        observed.shape()[2],
    );
    if dim.0 == 0 || dim.1 == 0 || dim.2 == 0 {
        return Ok(());
    }

    let spread = parameters.spread();
    // The estimate, initialised to the observation. The observation itself is
    // *not* widened into a second array: it is read through the view one voxel
    // at a time where the ratio is formed, which is an image's worth of `f64`
    // this op does not allocate.
    let mut estimate = observed.map(|value| (*value).into());
    let mut blurred = Array3::<f64>::zeros(dim);
    let mut ratio = Array3::<f64>::zeros(dim);
    let mut correction = Array3::<f64>::zeros(dim);

    for _ in 0..parameters.iterations() {
        // Forward: what the instrument would have seen, given this estimate.
        gaussian_smooth_into(estimate.view(), spread.forward(), blurred.view_mut())?;
        for i in 0..dim.0 {
            for j in 0..dim.1 {
                for k in 0..dim.2 {
                    let denominator = blurred[[i, j, k]];
                    ratio[[i, j, k]] = if denominator > 0.0 {
                        observed[[i, j, k]].into() / denominator
                    } else {
                        0.0
                    };
                }
            }
        }
        // Backward: spread each voxel's correction over the neighbourhood it
        // came from, through the reflected kernel. For a symmetric kernel this
        // is the same arithmetic as the forward pass; for any other it is not,
        // and getting it wrong would bias the estimate in the direction of the
        // kernel's own asymmetry.
        gaussian_smooth_into(ratio.view(), spread.backward(), correction.view_mut())?;
        for i in 0..dim.0 {
            for j in 0..dim.1 {
                for k in 0..dim.2 {
                    let updated = estimate[[i, j, k]] * correction[[i, j, k]];
                    estimate[[i, j, k]] = if updated.is_finite() { updated } else { 0.0 };
                }
            }
        }
    }

    out.assign(&estimate);
    Ok(())
}

// ---------------------------------------------------------------- shell --

/// Iterative spatial-domain deconvolution as a [`BlockOp`].
pub struct DeconvolveOp {
    name: &'static str,
    parameters: Deconvolution,
    cost: f64,
}

impl DeconvolveOp {
    pub fn new(name: &'static str, parameters: Deconvolution) -> Self {
        let cost = cost_for(&parameters);
        Self {
            name,
            parameters,
            cost,
        }
    }

    pub fn parameters(&self) -> &Deconvolution {
        &self.parameters
    }

    /// Override the measured cost, for a machine where it was retaken.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// The dispatch from the tagged buffer to the generic kernel.
    ///
    /// The types with a lossless `Into<f64>` reach the kernel with **no copy**.
    /// `bool`, `u64` and `i64` have no such conversion, so the shell widens them
    /// through `Voxels::widened` — a copy it is choosing rather than a
    /// conversion the kernel would have to invent. `f16` is refused before a run
    /// starts, by `accepts`, because no buffer holds one.
    fn evaluate(&self, input: &Voxels, out: ArrayViewMut3<'_, f64>) -> Result<()> {
        macro_rules! direct {
            ($type:ty) => {
                deconvolve_into(input.view::<$type>()?, &self.parameters, out)
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
                deconvolve_into(widened.view(), &self.parameters, out)
            }
            Dtype::F16 => Err(Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; `accepts` refuses it before a run starts",
                self.name
            ))),
        }
    }
}

impl BlockOp for DeconvolveOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Twice the kernel's radius, times the iteration count. Both terms come
    /// from the parameters and there is no field that could set either; see
    /// [`Deconvolution::reach`] for the recursion they come out of.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.parameters.reach(axis)
    }

    /// Every element type a buffer holds. The op reads and accumulates; what it
    /// accumulates into is its own business and is always `f64`.
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    /// `f64` whatever came in. An estimate is the observation scaled by a
    /// product of ratios, which is fractional even for a `u8` volume; narrowing
    /// it back would be the shell inventing a quantisation the caller did not
    /// ask for, and doing so *between* iterations would stop the iteration
    /// converging at all.
    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        self.evaluate(input, out.view_mut::<f64>()?)
    }

    /// **Zero, and only zero** — and the "only" is the part that is earned.
    ///
    /// At `+0.0` the whole iteration is exact and stays exact, whatever the
    /// weights are. Every tap of the blurred estimate is `w * 0.0`, which is
    /// `+0.0` for a non-negative `w`; the sum of those, accumulated from an
    /// initial `+0.0`, is `+0.0`; a denominator of zero takes the guarded branch,
    /// so the ratio is `+0.0` rather than a NaN — which is the case that would
    /// otherwise make this declaration catastrophic instead of merely wrong; the
    /// back-projection of zeros is `+0.0`; and `0.0 * 0.0` is `+0.0`, at every
    /// iteration. Nothing in that chain depends on a weight, a sigma or a count.
    ///
    /// At any other constant it is **not** exact, and the reason is one
    /// subtraction-free line: the weights are normalised by dividing by their
    /// sum, and the sum of the quotients is not exactly `1.0` in binary floating
    /// point. So the blur of a constant `c` is a value near `c` and not `c`, the
    /// ratio is near `1` and not `1`, and the estimate drifts in the last bits —
    /// once per iteration. That is the same reason `local.rs` gives for the local
    /// mean withholding its mapping, arrived at from the other direction, and it
    /// is why this op declares nothing there: a skipped block must produce
    /// exactly what a computed one would have, and here it would not.
    ///
    /// **`-0.0` is excluded on purpose.** It compares equal to `+0.0`, so a
    /// bare `value == 0.0` would accept it — and the op would then answer `-0.0`
    /// where the short circuit wrote `+0.0`. The two are equal under `==` and
    /// differ in their bits, which is precisely the disagreement a byte-identity
    /// suite is built to catch and would be unable to see.
    ///
    /// The test checks the bits over a range of constants rather than trusting
    /// this paragraph.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (value == 0.0 && value.is_sign_positive()).then_some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

// ---------------------------------------------------------------- cost --

/// Measured; see [`cost_report`].
///
/// **The figure is for the whole op, all iterations included**, because that is
/// what one `apply` costs and what the planner is pricing. It is written as
/// `iterations * (per-iteration)` so that the two are separable by eye: an
/// iteration is two separable passes over the same taps — the forward blur and
/// the back-projection — plus a fixed slab of per-voxel work for the divide, the
/// guard and the multiply that no kernel width changes.
pub fn cost_for(parameters: &Deconvolution) -> f64 {
    let taps = parameters.spread().taps() as f64;
    parameters.iterations() as f64 * (2.0 * CONVOLUTION_COST_PER_TAP * taps + UPDATE_COST)
}

/// Measured; see [`cost_report`]. One tap of one separable pass, relative to a
/// voxelwise map.
///
/// It is not the same number `ridge` stores for what is literally the same
/// summation (0.79), and the difference is not a contradiction: that figure was
/// fitted over a scale list whose passes are followed by a per-voxel root-finder
/// and this one over a kernel width that ranges to 51 taps, so the two fits put
/// the shared overhead in different places. Both are ratios and both reproduce
/// their own tables; `docs/design/BLOCK_OPS.md` §"Simulating strategies" is the
/// standard being met — trust "A beats B", distrust "A takes 40 minutes".
pub const CONVOLUTION_COST_PER_TAP: f64 = 0.60;

/// Measured; see [`cost_report`]. The divide, its guard, the multiplicative
/// update and the three block-sized buffers one iteration touches, at one voxel,
/// relative to a voxelwise map. Not negligible — the iteration is memory-bound
/// between its correlations — but small against the correlations themselves,
/// which is what makes the kernel width and the iteration count the two things
/// this op's cost is about.
pub const UPDATE_COST: f64 = 4.8;

/// The measurement the two constants above came from, kept as text so a re-run
/// elsewhere can be compared against it rather than merely replacing it.
/// `--release`, one thread, 48 x 32 x 32, best of 3, on the machine this crate
/// was developed on; the unit is the voxelwise map at 6.03 ns/voxel.
///
/// ```text
/// case                                     ns/voxel  taps  iter  relative  per iteration
/// voxelwise map (the unit)                     6.03     0     0      1.00              -
/// deconvolve, sigma 1, truncate 2, 1 iter    143.23    15     1     23.74          23.74
/// deconvolve, sigma 1, truncate 2, 2 iter    268.06    15     2     44.43          22.22
/// deconvolve, sigma 1, truncate 2, 4 iter    551.18    15     4     91.36          22.84
/// deconvolve, sigma 2, truncate 2, 1 iter    229.76    27     1     38.08          38.08
/// deconvolve, sigma 2, truncate 2, 4 iter    876.70    27     4    145.31          36.33
/// deconvolve, sigma 4, truncate 2, 2 iter    799.75    51     2    132.55          66.28
/// ```
///
/// The last column is the one the constants are fitted over — least squares of
/// `per iteration = 2 * tap * taps + update` across the six rows, two predictors
/// and no intercept beyond `update`, because the model *is* the shape of the
/// work. It reproduces them to within 4% at worst.
///
/// **Four repetitions of the whole table spread by about 10% between them**, and
/// the fit above is over the middle one. That spread is larger than the fit's own
/// error, which is the honest reason these are ratios: the ordering between rows
/// was identical in all four runs and the absolute nanoseconds were not.
///
/// **The cost is linear in the iteration count and the table shows it**, which
/// is the property the frequency-domain family would not have: the six rows are
/// three kernel widths at two or three counts each, and the per-iteration column
/// is flat within each width.
pub const COST_MEASUREMENT: &str = "ops::deconvolve::cost_report";

/// Retake the measurement the constants above were read off.
///
/// Same method as `super::cost` and `super::ridge`: through `BlockOp::apply`,
/// which is the entry point the executor uses; the **best** of several
/// repetitions, because contention on a shared machine is one-sided and a mean
/// over such samples is worthless; expressed relative to the voxelwise map,
/// which is this module's unit of work.
///
/// The table prints the tap count and the iteration count, so the two constants
/// can be read off as a plane: holding iterations fixed and varying the radius
/// gives the per-tap slope, and holding the radius fixed and varying the
/// iterations gives the rest.
pub fn cost_report(shape: [usize; 3], repetitions: usize) -> String {
    use std::time::Instant;

    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let anchor = Anchor::whole(shape);
    let mut input = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
    for (flat, value) in input.iter_mut().enumerate() {
        *value = ((flat * 7919) % 1013) as f64;
    }
    let input: Voxels = input.into();

    let mut cases: Vec<(String, Box<dyn BlockOp>, f64, f64)> = vec![(
        "voxelwise map (the unit)".to_string(),
        Box::new(super::voxelwise::VoxelwiseMapOp::threshold(
            "map", 500.0, 1.0, 0.0,
        )),
        0.0,
        0.0,
    )];
    for (sigma, truncate, iterations) in [
        (1.0f64, 2.0f64, 1usize),
        (1.0, 2.0, 2),
        (1.0, 2.0, 4),
        (2.0, 2.0, 1),
        (2.0, 2.0, 4),
        (4.0, 2.0, 2),
    ] {
        let spread = PointSpread::gaussian([sigma; 3], truncate).unwrap();
        let taps = spread.taps() as f64;
        let parameters = Deconvolution::new(spread, iterations).unwrap();
        cases.push((
            format!("deconvolve, sigma {sigma}, truncate {truncate}, {iterations} iterations"),
            Box::new(DeconvolveOp::new("deconvolve", parameters)),
            taps,
            iterations as f64,
        ));
    }

    let mut rows = Vec::new();
    for (name, op, taps, iterations) in cases {
        let mut out = Voxels::zeros(op.produces(input.dtype()), op.output_shape(shape)).unwrap();
        // One untimed pass: a freshly allocated output pays a page fault per
        // page on first touch, and that fault is the measurement for the
        // cheapest op here.
        op.apply(&input, &mut out, &anchor).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..repetitions.max(1) {
            let started = Instant::now();
            op.apply(&input, &mut out, &anchor).unwrap();
            let elapsed = started.elapsed().as_secs_f64() * 1e9;
            std::hint::black_box(out.view::<f64>().unwrap().iter().take(1).sum::<f64>());
            best = best.min(elapsed / voxels);
        }
        rows.push((name, best, taps, iterations, op.cost_per_voxel()));
    }

    let unit = rows.first().map(|(_, nanos, ..)| *nanos).unwrap_or(1.0);
    let mut out = format!(
        "deconvolution cost, {}x{}x{}, best of {repetitions}\n{:<58} {:>9} {:>6} {:>6} {:>9} \
         {:>8}\n",
        shape[0], shape[1], shape[2], "case", "ns/voxel", "taps", "iter", "measured", "stored"
    );
    for (name, nanos, taps, iterations, stored) in &rows {
        out.push_str(&format!(
            "{name:<58} {nanos:>9.2} {taps:>6.0} {iterations:>6.0} {:>9.2} {stored:>8.2}\n",
            nanos / unit
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(dim: (usize, usize, usize)) -> Array3<f64> {
        let mut array = Array3::zeros(dim);
        for (flat, value) in array.iter_mut().enumerate() {
            *value = (((flat * 7919) % 1013) as f64) / 8.0;
        }
        array
    }

    fn spread() -> PointSpread {
        PointSpread::gaussian([1.0, 1.0, 1.0], 2.0).unwrap()
    }

    // ---------------------------------------------------------- kernel --

    #[test]
    fn a_kernel_is_odd_non_negative_and_open_at_its_ends() {
        assert!(PointSpread::new([vec![1.0, 2.0, 1.0], vec![1.0], vec![1.0]]).is_ok());
        // even, so it has no centre
        assert!(PointSpread::new([vec![1.0, 1.0], vec![1.0], vec![1.0]]).is_err());
        // negative
        assert!(PointSpread::new([vec![1.0, -2.0, 1.0], vec![1.0], vec![1.0]]).is_err());
        // nothing to normalise by
        assert!(PointSpread::new([vec![0.0, 0.0, 0.0], vec![1.0], vec![1.0]]).is_err());
        // padded: the stated radius is wider than the kernel, which would make
        // the derived reach loose
        assert!(PointSpread::new([vec![0.0, 1.0, 0.0], vec![1.0], vec![1.0]]).is_err());
        // non-finite
        assert!(PointSpread::new([vec![1.0, f64::NAN, 1.0], vec![1.0], vec![1.0]]).is_err());
    }

    #[test]
    fn the_weights_are_normalised_and_the_reflection_is_the_reverse() {
        let manual = PointSpread::new([vec![1.0, 2.0, 5.0], vec![1.0], vec![1.0]]).unwrap();
        assert_eq!(manual.weights(0).to_vec(), vec![0.125, 0.25, 0.625]);
        assert_eq!(manual.backward()[0], vec![0.625, 0.25, 0.125]);
        assert_eq!(manual.radius(0), 1);
        assert_eq!(manual.radius(1), 0);
        assert_eq!(manual.taps(), 5);
        assert_eq!(manual.centre_weight(), 0.25);
        // a Gaussian is its own reflection, bit for bit
        let gauss = spread();
        assert_eq!(gauss.forward(), gauss.backward());
    }

    // ----------------------------------------------------------- reach --

    /// The three factors, separately visible.
    #[test]
    fn the_reach_is_twice_the_radius_times_the_iteration_count() {
        let spread = PointSpread::gaussian([1.0, 2.0, 0.5], 2.0).unwrap();
        // radii: ceil(2), ceil(4), ceil(1)
        assert_eq!(
            [spread.radius(0), spread.radius(1), spread.radius(2)],
            [2, 4, 1]
        );
        for iterations in [1usize, 2, 3, 7] {
            let parameters = Deconvolution::new(spread.clone(), iterations).unwrap();
            for axis in 0..3 {
                assert_eq!(parameters.reach(axis), 2 * spread.radius(axis) * iterations);
            }
            // and it does not consult the volume, which is what makes this a
            // local op rather than the barrier the other family would be
            let op = DeconvolveOp::new("deconvolve", parameters);
            assert_eq!(op.reach(0, 16), op.reach(0, 1_000_000));
        }
    }

    #[test]
    fn a_deconvolution_runs_at_least_once_and_its_reach_stays_a_number() {
        assert!(Deconvolution::new(spread(), 0).is_err());
        assert!(Deconvolution::new(spread(), 1).is_ok());
        assert!(Deconvolution::new(spread(), usize::MAX).is_err());
    }

    // ---------------------------------------------------------- kernel --

    /// The property the whole family rests on, on a field whose original is
    /// known: blur it, invert the blur, and the estimate is closer to the
    /// original than the blurred field was.
    ///
    /// A smooth bump on a positive background, which is what the multiplicative
    /// iteration is defined for. The volume-scale version of this claim, on
    /// `synthetic::Scene`, is in `tests/deconvolution.rs`; this one is here so
    /// that a change to the kernel fails at the kernel.
    #[test]
    fn deconvolving_a_blurred_field_moves_it_back_towards_the_original() {
        let dim = (15usize, 13, 11);
        let truth = Array3::from_shape_fn(dim, |(i, j, k)| {
            let (x, y, z) = (i as f64 - 7.0, j as f64 - 6.0, k as f64 - 5.0);
            10.0 + 90.0 * (-(x * x + y * y + z * z) / 6.0).exp()
        });
        let mut observed = Array3::<f64>::zeros(dim);
        blur_into(truth.view(), &spread(), observed.view_mut()).unwrap();

        let parameters = Deconvolution::new(spread(), 12).unwrap();
        let mut estimate = Array3::<f64>::zeros(dim);
        deconvolve_into(observed.view(), &parameters, estimate.view_mut()).unwrap();

        let distance = |field: &Array3<f64>| -> f64 {
            field
                .iter()
                .zip(truth.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        };
        assert!(
            distance(&estimate) < 0.5 * distance(&observed),
            "the iteration did not restore anything: {} from the original against the \
             blurred field's {}",
            distance(&estimate),
            distance(&observed)
        );
    }

    /// **The shell reaches the kernel in the element type it was handed.** A
    /// `u16` volume is deconvolved to the same bits as the widened copy of it,
    /// which is what `accepts` promises and what the no-copy dispatch would
    /// break silently if it converted somewhere on the way.
    #[test]
    fn a_narrower_element_type_gives_the_same_answer_as_widening_it_by_hand() {
        let dim = (9usize, 7, 6);
        let narrow: Array3<u16> =
            Array3::from_shape_fn(dim, |(i, j, k)| ((i * 37 + j * 11 + k * 3) % 400) as u16);
        let op = DeconvolveOp::new("deconvolve", Deconvolution::new(spread(), 3).unwrap());
        assert!(op.accepts(Dtype::U16));
        assert_eq!(op.produces(Dtype::U16), Dtype::F64);
        assert!(!op.accepts(Dtype::F16));

        let input: Voxels = narrow.clone().into();
        let mut through_shell = Voxels::zeros(Dtype::F64, [dim.0, dim.1, dim.2]).unwrap();
        op.apply(
            &input,
            &mut through_shell,
            &Anchor::whole([dim.0, dim.1, dim.2]),
        )
        .unwrap();

        let widened = narrow.mapv(f64::from);
        let mut by_hand = Array3::<f64>::zeros(dim);
        deconvolve_into(
            widened.view(),
            &Deconvolution::new(spread(), 3).unwrap(),
            by_hand.view_mut(),
        )
        .unwrap();
        assert_eq!(through_shell.view::<f64>().unwrap(), by_hand.view());
    }

    #[test]
    fn an_empty_block_is_not_an_error() {
        let parameters = Deconvolution::new(spread(), 2).unwrap();
        let empty = Array3::<f64>::zeros((0, 4, 4));
        let mut out = Array3::<f64>::zeros((0, 4, 4));
        deconvolve_into(empty.view(), &parameters, out.view_mut()).unwrap();
    }

    #[test]
    fn a_shape_disagreement_is_refused_rather_than_truncated() {
        let parameters = Deconvolution::new(spread(), 1).unwrap();
        let input = Array3::<f64>::zeros((4, 4, 4));
        let mut out = Array3::<f64>::zeros((4, 4, 3));
        assert!(deconvolve_into(input.view(), &parameters, out.view_mut()).is_err());
    }

    // ------------------------------------------------------- the zeros --

    /// The declaration `constant_maps_to` makes, checked in the bits rather than
    /// argued — including that a zero input does not become a NaN through the
    /// division, which is the failure the guard exists for.
    #[test]
    fn a_zero_volume_stays_exactly_positive_zero() {
        for iterations in [1usize, 2, 5] {
            let parameters = Deconvolution::new(spread(), iterations).unwrap();
            let input = Array3::<f64>::zeros((7, 6, 5));
            let mut out = Array3::<f64>::from_elem((7, 6, 5), f64::NAN);
            deconvolve_into(input.view(), &parameters, out.view_mut()).unwrap();
            for value in out.iter() {
                assert_eq!(value.to_bits(), 0.0f64.to_bits(), "{iterations} iterations");
            }
        }
    }

    /// The other half of the declaration: at any constant but zero the answer is
    /// **not** the constant, so declaring one would license a skip that changes
    /// the volume. Shown rather than asserted from theory — the drift is in the
    /// last bits and is exactly what a byte-identity suite would catch.
    #[test]
    fn a_non_zero_constant_is_not_reproduced_exactly_and_is_therefore_not_declared() {
        let op = DeconvolveOp::new("deconvolve", Deconvolution::new(spread(), 4).unwrap());
        assert_eq!(op.constant_maps_to(0.0), Some(0.0));
        // negative zero is excluded: it compares equal to +0.0 and has other bits
        assert_eq!(op.constant_maps_to(-0.0), None);
        for constant in [0.25f64, 1.0, -3.5, 1e6] {
            assert_eq!(op.constant_maps_to(constant), None, "{constant}");
        }

        // and the reason, demonstrated: the normalised weights do not sum to
        // exactly one, so a constant is not a fixed point of the iteration.
        let drifted = [0.25f64, 1.0, 7.0].iter().any(|&constant| {
            let parameters = Deconvolution::new(spread(), 4).unwrap();
            let input = Array3::<f64>::from_elem((5, 5, 5), constant);
            let mut out = Array3::<f64>::zeros((5, 5, 5));
            deconvolve_into(input.view(), &parameters, out.view_mut()).unwrap();
            out.iter()
                .any(|value| value.to_bits() != constant.to_bits())
        });
        let sums_to_one = spread().weights(0).iter().sum::<f64>() == 1.0;
        assert!(
            drifted || sums_to_one,
            "the weights sum to something other than one and yet every constant came \
             back bit-identical; the argument for withholding the declaration needs \
             rechecking"
        );
    }

    /// **The output is finite whatever came in.** A NaN in a volume compares
    /// unequal to itself and would make every byte-identity check in this crate
    /// pass vacuously, so the kernel is required not to produce one even from an
    /// input that already has.
    #[test]
    fn no_input_produces_a_non_finite_output() {
        let parameters = Deconvolution::new(spread(), 3).unwrap();
        for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX, -1.0] {
            let mut input = ramp((6, 6, 6));
            input[[3, 3, 3]] = poison;
            input[[0, 0, 0]] = poison;
            let mut out = Array3::<f64>::zeros((6, 6, 6));
            deconvolve_into(input.view(), &parameters, out.view_mut()).unwrap();
            assert!(
                out.iter().all(|value| value.is_finite()),
                "a {poison} in the input produced a non-finite estimate"
            );
        }
    }

    // ------------------------------------------------------------ cost --

    /// What can be asserted about a measured cost without measuring: the
    /// ordering the model encodes is the ordering the op actually has.
    #[test]
    fn the_stored_cost_grows_with_both_of_the_things_the_work_grows_with() {
        let narrow = PointSpread::gaussian([1.0; 3], 2.0).unwrap();
        let wide = PointSpread::gaussian([3.0; 3], 2.0).unwrap();
        let cost = |spread: PointSpread, iterations: usize| {
            DeconvolveOp::new("d", Deconvolution::new(spread, iterations).unwrap()).cost_per_voxel()
        };
        assert!(cost(wide.clone(), 2) > cost(narrow.clone(), 2));
        assert!(cost(narrow.clone(), 4) > cost(narrow.clone(), 2));
        // exactly proportional to the iteration count, which is the claim the
        // doc on `cost_for` makes about the figure being per-op and not
        // per-iteration
        assert!((cost(narrow.clone(), 4) - 2.0 * cost(narrow, 2)).abs() < 1e-9);
        // and more than a voxelwise map, which is the unit
        assert!(cost(wide, 1) > 1.0);
    }

    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn print_the_cost_table() {
        println!("{}", cost_report([48, 32, 32], 3));
    }
}
