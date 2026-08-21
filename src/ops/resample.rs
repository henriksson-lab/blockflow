// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation,
// not adapted from any implementation of it.
//
// Resizing a volume by an arbitrary factor per axis: the first op here whose
// **output image is a different shape from the input image**, which is the
// capability `docs/design/BLOCK_OPS.md` §"Step 3" opened and left unused.
//
// What is resampled, and by what map
// ----------------------------------
// The factor is an **exact rational per axis**, `up / down`, reduced. The output
// extent of an axis is `floor(in * up / down)` by default; a factor that does not
// divide the extent evenly drops the remainder rather than producing a partial
// voxel, and the dropped voxels are named in [`Resample::unsampled_tail`] rather
// than left for a reader to derive.
//
// **By default, and not always** — see "An output extent the caller states"
// below. `floor` is the honest reading of *"resize by this ratio"* and stays what
// [`Resample::new`] means; it is not the only extent a caller may want from the
// same operation, and the alternative is a statement rather than a second
// rounding rule.
//
// The map from an output index to a source coordinate is the **centred**
// ("half-voxel") one:
//
// ```text
//     x(o) = (o + 0.5) * down/up - 0.5 = ((2o + 1) * down - up) / (2 * up)
// ```
//
// stated as that exact fraction and evaluated in integer arithmetic, so nothing
// here depends on how an `f64` rounded a ratio. Two properties are why this
// convention and not the corner-aligned `x(o) = o * (in - 1) / (out - 1)`:
//
// * it is **shift-invariant** — `x(o + k) = x(o) + k * down/up` — which is what
//   makes a block and the whole volume compute the same voxel from the same
//   samples, and it is the reason this op is decomposable at all;
// * it is defined for an output extent of 1, where the corner-aligned form
//   divides by zero.
//
// What it costs is that `x(o)` is not an integer at the array's ends, so the two
// end voxels sample outside the array and are **clamped** to it. That is this
// op's edge rule, and it is the same rule every other op in this module states:
// at a real volume boundary the clamp is the whole story, and at a block seam it
// is wrong on purpose, because a silent wrong answer is what the halo guard
// exists to turn into a loud one.
//
// The interpolations, and the one that is not here
// ------------------------------------------------
// | kind | samples per axis | element types | reach |
// |---|---|---|---|
// | [`Interpolation::Nearest`] | 1 | **every** type a buffer holds | 0, at every factor |
// | [`Interpolation::Linear`] | 2 (1 where the coordinate is exact) | every numeric type; not `bool` | derived below |
//
// **Nearest is the case a general library gets asked for and the one that keeps
// label data meaningful**: it selects a value that was read, bit for bit, so a
// volume of integer labels resampled by it still holds labels. Its kernel is
// therefore generic over `T: Copy` and nothing more — it never combines two
// values, so it never needs to know what they mean.
//
// Linear combines two values, which needs a numeric currency; its kernel is
// generic over [`VoxelElement`], the crate's own bridge, and accumulates in
// `f64`. `bool` is refused by the shell rather than blended: `from_f64` maps
// every non-zero blend to `true`, so a linear pass over a mask would be a silent
// dilation of it. Integers are accepted and the result is **rounded** to the
// nearest, not truncated — a blend of 7 and 8 is 8 at `t > 0.5`, and truncation
// would bias every interpolated volume downward by half a step.
//
// **Higher order is deliberately absent, and there are two different reasons.**
// A cubic convolution (Catmull-Rom, or Keys with a parameter) is four samples
// per axis and would fit here exactly — the reach derivation below is written
// against a tap count and would extend to it — but *which* cubic is a parameter
// nobody has asked for, and the choice changes the answer, so shipping one would
// be inventing a default in a crate whose first rule is that everything is a
// parameter. Spline interpolation of order 3 and above is a different matter: it
// is not a wider window at all but a **global IIR prefilter** followed by a local
// evaluation, and the prefilter's dependency is the whole axis
// ([`AxisReach::All`]) — a planning barrier, i.e. a separate op with a separate
// reach, not a variant of this one. Recording which of the two a missing feature
// is worth more than the feature.
//
// Aliasing: whose job is the prefilter
// ------------------------------------
// **Downsampling here point-samples, and therefore aliases. That is the
// caller's to fix, and this op will not do it silently.** The reasons are in
// this order:
//
// * The correct anti-aliasing filter is a low-pass whose radius follows the
//   factor, i.e. **a neighbourhood op with a reach of its own**. This module
//   already has ops that are exactly that. Composed —
//   `Chain::sequence([smoothing, resampling])` — the plan prices both terms,
//   halos both terms and can cut between them. Hidden inside this op, the same
//   two terms become one reach nobody can inspect and a cost model with two
//   ops' work under one name.
// * It would be **wrong for the data nearest exists for**. A mean of labels is
//   not a label. An op that low-passed before a nearest resampling would corrupt
//   exactly the case it is most often reached for, and one that low-passed only
//   for linear would have a rule a caller has to remember.
// * It would weaken what this op can *declare*. `constant_maps_to` is exact here
//   (see below) and a mean's is not: `(v + v + ... + v) / m` is not `v` in binary
//   floating point, which `local.rs` records and pins with a test.
//
// So the aliasing is stated, demonstrated by a test that shows a pattern at the
// sampling frequency surviving into the output as a constant, and left where the
// remedy is.
//
// An output extent the caller states
// ----------------------------------
// [`Resample::to_extent`] takes the two extents — the input volume and the
// output volume — and derives the factor from them, instead of taking the factor
// and deriving the output volume from it. The sample map does not change and is
// not a second convention: `Ratio::new(out, in)` is `in/out` as an exact
// rational, and `x(o) = (o + 0.5) * in/out - 0.5` is the same centred map with
// the same clamp. What changes is which of the two numbers is the input to the
// arithmetic and which is derived from it.
//
// **Why the facility is needed at all, rather than being spelled `Ratio`.** It
// can be spelled `Ratio` — for the *whole volume*. `Ratio::new(out, in)` reduces,
// and `outputs(in)` of the reduced pair is exactly `out`, so a resident call
// needs nothing here. What cannot be spelled that way is the **decomposition**.
// The fetch mapping below rests on `up` being the period of the sampling lattice
// in output space, and an extent-derived ratio is essentially always in lowest
// terms — `348/2137` and `249/3823` both have `gcd = 1` — so the period is the
// whole axis and the only legal cut is no cut. That is a fact about the ratio and
// not a defect of the mapping, and the way past it is not a smaller period but a
// different kind of declaration.
//
// So under a stated extent this op takes the shape it writes from the **plan**
// ([`BlockOp::placed_output_shape`]) and states its fetch as a region per block,
// exactly as `lattice.rs` does and for the identical reason: two blocks fetching
// the same number of source voxels write different numbers of output voxels, so
// no shape-to-shape function exists to invert. The waiver is declared
// ([`BlockOp::takes_extent_from_placement`]) and the check it owes is in
// [`ResampleOp::apply_placed`] — every output voxel written must bracket between
// source voxels the buffer actually holds, refused by name rather than clamped to
// the buffer's edge. Nothing else about the op moves: the alignment, the halo and
// the reach all become *nothing* under a stated extent, because there is no
// lattice left to snap a fetch onto and the dependency is carried by the fetch
// region rather than by a distance in this phase's own voxels.
//
// **The default is byte-unchanged**, and that is asserted rather than assumed:
// `the_factor_exact_default_is_unmoved_by_the_stated_extent` below runs every
// factor-path quantity — extent, reach, alignment, halo, fetch and the voxels
// themselves — through both spellings of a factor whose extent divides evenly, so
// a change to the shared arithmetic that moved the default would fail there.
//
// What this op reads beyond what it writes
// ----------------------------------------
// A resampling phase's blocks live in **two coordinate spaces**: its grid is cut
// over the *output* volume, and what it fetches is a region of the *input*
// image. `BlockGeometry::source` is where that fetch is stated and
// [`resample_phase`] is what states it — the mapping is this op's own geometry,
// so shipping the plan-builder beside the op is what keeps the two from
// drifting.
//
// The fetch for a read extent `[s, e)` is the **image of that extent under the
// same rational factor**, `[s*down/up, e*down/up)`, which is why `s` and `e` are
// required to be multiples of `up`: otherwise the image is not a voxel boundary
// and [`BlockOp::output_shape`] — a function of the fetched *shape* alone —
// could not name the extent it produces. The alignment is not asked of the
// caller's block grid: it is bought by the **halo**, which is snapped up to the
// next multiple of `up` per block and per side ([`Resample::halo`]). Any block
// edge therefore plans, and the price is stated per block instead of being a
// constraint on the caller.
//
// Against that fetch, the reach is derived per side from the factor and the
// interpolation, and from nothing else:
//
// * **Nearest reaches nothing, at any factor.** `round(x(o))` never leaves the
//   image of the block's own read extent — shown in `reach_is_derived_and_tight`
//   rather than asserted.
// * **Linear reaches nothing when downsampling** (`up <= down`): the coordinate
//   of an output voxel lies inside its own cell, and so does the sample above it.
// * **Linear upsampling reaches `ceil((up - down) / (2 * down))` voxels**, and
//   the two sides are computed *separately* — the dependency of one output voxel
//   is genuinely one-sided, `floor(x)` and `floor(x) + 1` and never below — and
//   then come out **equal**, because the centred map places every block's outputs
//   symmetrically inside their own cells. That equality is a result and not an
//   assumption: the two sides are derived by different arguments (see
//   [`Resample::reach`]) and each is shown tight to one voxel by a test that
//   finds, for every factor, an output one voxel closer to the seam whose value
//   differs from the whole-volume answer.
//
// Where the two-sided, per-block form is **not** optional is the halo. The
// alignment snapping is per block and per side by construction — a core starting
// two voxels into a cell needs two more below it and none above, and the block
// after it needs the opposite — so a halo that could only say one number, or one
// number per block, could not put every read boundary on the fetch lattice at
// all. `tests/resample_ops.rs` shows both refusals, so this is a case where
// `AxisReach::PerBlock` with unequal `(lo, hi)` is what makes an arbitrary block
// edge **planable** rather than what makes it cheaper. What the alignment itself
// costs is measured there too, against the tight dependency: 1.00x for every
// downsampling and every integer growth, 1.52x to 2.01x for a rational growth at
// a small block edge, and shrinking with the edge.

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::decomposition::PhaseDecomposition;
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::geometry::BlockGrid;
use crate::op::{Anchor, BlockOp, Geometry, InputMap, Placement as OpPlacement};
use crate::reach::{AxisReach, Reach, Space};
use crate::region::Region;
use crate::voxels::{VoxelElement, Voxels};

// --------------------------------------------------------------- factor --

/// A resizing factor for one axis: the exact rational `up / down`, reduced.
///
/// **Exact rather than an `f64`**, because every question this op answers about
/// a block — which source voxels it needs, whether a fetch is a voxel boundary,
/// what extent a fetched block produces — is an integer question, and a factor
/// that arrived as `0.333333` would answer some of them differently from one
/// that arrived as `1/3`. A `Decomposition` is binding and reproducible; a
/// rounding that depends on how a caller spelled a third is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ratio {
    up: usize,
    down: usize,
}

impl Ratio {
    /// `up / down`, reduced. Neither may be zero: a factor of zero empties the
    /// volume and a division by zero has no meaning to fall back on.
    pub fn new(up: usize, down: usize) -> Result<Self> {
        if up == 0 || down == 0 {
            return Err(Error::InvalidArgument(format!(
                "a resampling factor is a ratio of two positive counts; got {up}/{down}"
            )));
        }
        let divisor = greatest_common_divisor(up, down);
        Ok(Self {
            up: up / divisor,
            down: down / divisor,
        })
    }

    /// `n` output voxels per input voxel.
    pub fn larger(n: usize) -> Result<Self> {
        Self::new(n, 1)
    }

    /// One output voxel per `n` input voxels.
    pub fn smaller(n: usize) -> Result<Self> {
        Self::new(1, n)
    }

    /// The axis is left alone.
    pub fn identity() -> Self {
        Self { up: 1, down: 1 }
    }

    pub fn up(&self) -> usize {
        self.up
    }

    pub fn down(&self) -> usize {
        self.down
    }

    pub fn is_identity(&self) -> bool {
        self.up == 1 && self.down == 1
    }

    /// Whether the output is longer than the input.
    pub fn grows(&self) -> bool {
        self.up > self.down
    }

    /// Output voxels from `n` input voxels: `floor(n * up / down)`.
    ///
    /// Floored, and that is the definition rather than a rounding choice: the
    /// last partial output voxel would have to be interpolated from source voxels
    /// that do not exist, so it is not produced at all and
    /// [`Resample::unsampled_tail`] says how many input voxels went unread
    /// because of it.
    pub fn outputs(&self, inputs: usize) -> usize {
        ((inputs as u128 * self.up as u128) / self.down as u128) as usize
    }

    /// Input voxels an aligned output extent of `outputs` is the image of.
    ///
    /// Defined only where `up` divides `outputs`, which is what the halo
    /// snapping arranges; the caller that needs it for an arbitrary extent wants
    /// [`Resample::source_region`], which handles the volume's last block.
    fn aligned_inputs(&self, outputs: usize) -> Option<usize> {
        (outputs % self.up == 0).then(|| outputs / self.up * self.down)
    }

    /// The source index [`Interpolation::Nearest`] reads for output `index`:
    /// `floor(((2i + 1) * down) / (2 * up))`, which is `round(x(i))` with ties
    /// going up.
    ///
    /// The tie rule is a decision and it is written here rather than implied by
    /// a cast: at a factor of exactly one half every output coordinate lands on a
    /// half, so *every* voxel is a tie and the rule alone decides whether the
    /// even or the odd source voxels survive.
    fn nearest(&self, index: usize) -> i128 {
        div_floor(
            (2 * index as i128 + 1) * self.down as i128,
            2 * self.up as i128,
        )
    }

    /// The two source indices [`Interpolation::Linear`] reads for output
    /// `index`, and how far between them it lands.
    ///
    /// **A sample whose weight would be zero is not returned**, exactly as
    /// `local.rs`'s bracket does not return one, and for the same reason: an op
    /// that read a voxel its answer does not depend on would have to declare a
    /// reach covering that read, and a reach that is wider than the dependency is
    /// a reach nobody can check. Where `x(index)` is an integer — every voxel of
    /// an identity, and every third voxel of a factor of three — the second
    /// sample is dropped and the weight is zero.
    fn bracket(&self, index: usize) -> (i128, i128, f64) {
        let denominator = 2 * self.up as i128;
        let numerator = (2 * index as i128 + 1) * self.down as i128 - self.up as i128;
        let base = div_floor(numerator, denominator);
        let fraction = numerator - base * denominator;
        if fraction == 0 {
            return (base, base, 0.0);
        }
        (base, base + 1, fraction as f64 / denominator as f64)
    }

    /// `(lo, hi)`: how far, in **output** voxels, an output voxel's dependency
    /// reaches outside the image of its own position, for `interpolation`.
    ///
    /// The two sides are separate derivations of two different conditions, and
    /// they are kept separate even though they agree — see the module header.
    ///
    /// * `lo` is the smallest `d` for which the **first** sample of the output
    ///   `d` voxels above a fetch boundary is at or above that boundary:
    ///   `(2d + 1) * down >= up`, i.e. `ceil((up - down) / (2 * down))`. It is
    ///   unconditional, because the first sample is read at every weight.
    /// * `hi` is one more than the largest `d` for which the output `d` voxels
    ///   below the far boundary still reads a **second** sample past it. That one
    ///   is conditional: where the coordinate is exact there is no second sample,
    ///   so the scan below skips those `d` instead of assuming the worst — which
    ///   is what makes it tight rather than merely safe. The loop runs at most
    ///   `up / (2 * down)` times.
    fn reach(&self, interpolation: Interpolation) -> (usize, usize) {
        if interpolation == Interpolation::Nearest {
            return (0, 0);
        }
        let (up, down) = (self.up, self.down);
        let lo = (up + down - 1) / (2 * down);
        let mut hi = 0;
        let mut distance = 0usize;
        while (2 * distance + 1) * down <= up {
            if ((2 * distance + 1) * down + up) % (2 * up) != 0 {
                hi = distance + 1;
            }
            distance += 1;
        }
        (lo, hi)
    }
}

fn greatest_common_divisor(a: usize, b: usize) -> usize {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a.max(1)
}

/// Floor division for a signed numerator, which `/` is not: `-1 / 4` is `0` in
/// Rust and the sample below a volume's lower edge is at `-1`.
fn div_floor(numerator: i128, denominator: i128) -> i128 {
    let quotient = numerator / denominator;
    if numerator % denominator != 0 && (numerator < 0) != (denominator < 0) {
        quotient - 1
    } else {
        quotient
    }
}

/// How a value is taken between the samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Interpolation {
    /// The single closest source voxel, ties going to the higher index.
    ///
    /// The value written is a value that was read, so this is the only choice
    /// that is meaningful for labels, masks and any other data whose values are
    /// names rather than quantities.
    Nearest,
    /// The weighted mean of the bracketing samples on each axis — trilinear over
    /// three axes, evaluated as `a + t * (b - a)` so that a constant field is
    /// reproduced exactly.
    Linear,
}

impl Interpolation {
    /// Whether a buffer of `dtype` can be resampled this way.
    ///
    /// The one refusal is `bool` under `Linear`; the module header says why. It
    /// is refused at plan time through [`BlockOp::accepts`] rather than at the
    /// block, which is what the method exists for.
    pub fn accepts(&self, dtype: Dtype) -> bool {
        match self {
            Self::Nearest => dtype != Dtype::F16,
            Self::Linear => !matches!(dtype, Dtype::F16 | Dtype::Bool),
        }
    }
}

/// What decides an axis's output extent.
///
/// **Two values rather than a `bool` or an `Option<[usize; 3]>`**, because the
/// second arm carries the input volume it was stated against and the first arm
/// carries nothing at all. A resampling that knows the extent it writes also
/// knows the extent it was told to write it *from* — the pair is what makes the
/// factor exact — and an arm holding only the output would be an extent whose
/// scale a reader has to go and find.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputExtent {
    /// `floor(n * up / down)` per axis, a function of the factor and the extent
    /// handed in. What [`Resample::new`] means, and what every caller that
    /// existed before this enum did meant.
    Factor,
    /// The extent stated by the caller, against the one input volume it was
    /// stated for.
    ///
    /// The factor is `Ratio::new(output[axis], input[axis])` and is therefore
    /// *derived* from this rather than checked against it, which is what keeps
    /// the two from disagreeing. See [`Resample::to_extent`].
    Stated {
        input: [usize; 3],
        output: [usize; 3],
    },
}

/// A factor per axis, one interpolation, and what decides the output extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Resample {
    ratios: [Ratio; 3],
    interpolation: Interpolation,
    extent: OutputExtent,
}

impl Resample {
    pub fn new(ratios: [Ratio; 3], interpolation: Interpolation) -> Self {
        Self {
            ratios,
            interpolation,
            extent: OutputExtent::Factor,
        }
    }

    /// The same factor on every axis.
    pub fn uniform(ratio: Ratio, interpolation: Interpolation) -> Self {
        Self::new([ratio; 3], interpolation)
    }

    /// **The two extents, with the factor derived from them** — the other way
    /// round from [`Resample::new`].
    ///
    /// The sample map is unchanged and is not a second convention: the factor is
    /// `Ratio::new(output[axis], input[axis])`, so the scale is `input/output`
    /// exactly and `x(o) = (o + 0.5) * input/output - 0.5` is the same centred
    /// map with the same clamp. Reduction cannot move it — `bracket` evaluates
    /// the fraction `((2o + 1) * down - up) / (2 * up)`, and dividing both terms
    /// by a common factor is the same rational and the same correctly-rounded
    /// `f64`.
    ///
    /// What it buys is the *decomposition*, and the module header is where the
    /// argument is: an extent-derived ratio is essentially always in lowest
    /// terms, so [`Resample::alignment`]'s period is the whole output axis and
    /// the factor path can only plan one block. Under a stated extent the write
    /// extent comes from the plan and the fetch is a region per block, so any cut
    /// plans.
    ///
    /// Refuses an extent of zero on either side, by [`Ratio::new`]'s refusal: a
    /// scale with a zero in it has no meaning to fall back on.
    pub fn to_extent(
        input: [usize; 3],
        output: [usize; 3],
        interpolation: Interpolation,
    ) -> Result<Self> {
        let mut ratios = [Ratio::identity(); 3];
        for axis in 0..3 {
            ratios[axis] = Ratio::new(output[axis], input[axis]).map_err(|_| {
                Error::InvalidArgument(format!(
                    "resampling axis {axis} from {} voxels to {}: an extent of zero on either side \
                     has no scale, so there is no map from an output index to a source coordinate \
                     to state.",
                    input[axis], output[axis]
                ))
            })?;
        }
        Ok(Self {
            ratios,
            interpolation,
            extent: OutputExtent::Stated { input, output },
        })
    }

    pub fn ratio(&self, axis: usize) -> Ratio {
        self.ratios[axis]
    }

    pub fn interpolation(&self) -> Interpolation {
        self.interpolation
    }

    /// What decides this resampling's output extent.
    pub fn extent(&self) -> OutputExtent {
        self.extent
    }

    /// The input and output volumes a stated extent was stated against, or
    /// `None` for the factor-exact default.
    fn stated(&self) -> Option<([usize; 3], [usize; 3])> {
        match self.extent {
            OutputExtent::Factor => None,
            OutputExtent::Stated { input, output } => Some((input, output)),
        }
    }

    /// The extent this produces from `inputs` on `axis`, by the factor.
    ///
    /// **The same function under both arms of [`OutputExtent`], and that is a
    /// result rather than an oversight.** For a stated extent the factor is
    /// `reduce(out, n)`, and `floor(n * (out/g) / (n/g))` is `out` exactly — so
    /// asked about the volume it was stated for, this returns the stated extent
    /// and no branch is needed. Asked about anything else under a stated extent
    /// the answer is the factor's, which is right for an interior block and
    /// wrong for the two at the ends; [`BlockOp::output_shape`] is where that is
    /// refused rather than papered over, and it is exactly why the stated arm
    /// takes its write extent from the plan.
    pub fn output_extent(&self, axis: usize, inputs: usize) -> usize {
        self.ratios[axis].outputs(inputs)
    }

    /// The shape of the whole output volume.
    ///
    /// Fails where an axis would be emptied — an extent of 5 shrunk by a factor
    /// of 8 is zero output voxels, and an image of zero extent is not an image.
    /// That is a fact about the request, so it is refused by name rather than
    /// rounded up to one.
    /// A stated extent is bound to one input volume, and asking it about another
    /// is refused rather than answered by the factor. The factor is only exact
    /// *at* the volume it was derived from — `348/2137` applied to 2000 voxels is
    /// `floor(2000 * 348/2137) = 325`, which is a number and not the answer to
    /// any question the caller asked.
    pub fn output_volume(&self, input: [usize; 3]) -> Result<[usize; 3]> {
        if let Some((stated_input, stated_output)) = self.stated() {
            if input != stated_input {
                return Err(Error::InvalidArgument(format!(
                    "this resampling states an output extent of {stated_output:?} for an input \
                     volume of {stated_input:?}, and was asked about {input:?}. A stated extent is \
                     bound to the volume it was stated against; the factor derived from it is \
                     exact only there."
                )));
            }
            return Ok(stated_output);
        }
        let mut volume = [0usize; 3];
        for axis in 0..3 {
            volume[axis] = self.output_extent(axis, input[axis]);
            if volume[axis] == 0 {
                return Err(Error::InvalidArgument(format!(
                    "resampling axis {axis} of {} by {}/{} produces no voxel at all. An extent \
                     shorter than one output voxel is a fact about the request, not something to \
                     round up.",
                    input[axis],
                    self.ratios[axis].up(),
                    self.ratios[axis].down()
                )));
            }
        }
        Ok(volume)
    }

    /// Input voxels at the top of `axis` that no output voxel samples, because
    /// the factor does not divide the extent evenly.
    ///
    /// Reported rather than hidden: it is the whole content of "the factor does
    /// not divide the volume evenly", and a caller comparing an output against
    /// an expectation wants to know that the last `n` input voxels were dropped
    /// rather than blended in.
    pub fn unsampled_tail(&self, axis: usize, inputs: usize) -> usize {
        let outputs = self.output_extent(axis, inputs);
        let ratio = self.ratios[axis];
        let sampled =
            (outputs as u128 * ratio.down() as u128).div_ceil(ratio.up() as u128) as usize;
        inputs.saturating_sub(sampled)
    }

    /// What an output voxel reads outside the image of its own read extent, per
    /// side, in this phase's own (output) voxels. See [`Ratio::reach`].
    ///
    /// **`(0, 0)` under a stated extent, and it is not the reach-0 evasion.**
    /// The quantity this method names is a distance measured against *the image
    /// of the read extent under the factor*, and under a stated extent there is
    /// no such image: what a block fetches is its own tight source span, stated
    /// per block by [`Resample::source_region`], and every output voxel of the
    /// read extent reads inside it by construction. The dependency has not gone
    /// anywhere — it has moved from a distance to a region, which is what
    /// [`Resample::reach_spec`]'s `Units::SourceIndex` tag says and what
    /// `ResampleOp::apply_placed`'s own check is against.
    pub fn reach(&self, axis: usize) -> (usize, usize) {
        if self.stated().is_some() {
            return (0, 0);
        }
        self.ratios[axis].reach(self.interpolation)
    }

    /// The full statement, asymmetric per side and in the phase's own voxels.
    ///
    /// **`Frame::Phase`, deliberately, and it is not the reach-0 evasion the
    /// design warns about.** The frame decides one thing: whether a read clamped
    /// at this phase's own volume edge may be trusted. Here it may, and for the
    /// reason the exception exists — the output volume's edge *is* the image of
    /// the input volume's edge, the fetch for the last block runs to the end of
    /// the image below ([`Resample::source_region`]), and the samples an edge
    /// output voxel wants beyond it do not exist for the whole-volume reference
    /// either. The claim is checked rather than asserted: the acceptance suite
    /// compares every voxel, edges included, against that reference.
    /// **Under a stated extent this is `Units::SourceIndex` instead**, which is
    /// `lattice.rs`'s statement and is made for its reason: the dependency is a
    /// region of the image below, named per block, and there is no factor
    /// converting it into a voxel of this phase — `Space::converts_to_voxels` is
    /// `false` for exactly that, so a plan cannot mistake the zero for a distance
    /// it may grow a halo by.
    pub fn reach_spec(&self) -> Reach {
        if self.stated().is_some() {
            return Reach::none().in_space(Space::source_index());
        }
        let mut axes = [AxisReach::none(), AxisReach::none(), AxisReach::none()];
        for axis in 0..3 {
            let (lo, hi) = self.reach(axis);
            axes[axis] = AxisReach::Bounded { lo, hi };
        }
        Reach::per_axis(axes)
    }

    /// The output positions a fetch boundary may sit at: `up` per axis.
    ///
    /// A fetch region is the image of a read extent under `down/up`, so a read
    /// boundary that is not a multiple of `up` has no image on the source
    /// lattice. This is the number the halo snaps to.
    /// **`[1, 1, 1]` under a stated extent**: every output boundary is a legal
    /// fetch boundary there, because a fetch is no longer required to be the
    /// image of a read extent under the factor. That requirement existed to let
    /// [`BlockOp::output_shape`] invert the fetch, and a stated extent does not
    /// invert it at all — it takes the extent from the plan. So the constraint
    /// that made the period matter is the one that has gone, not the period.
    pub fn alignment(&self) -> [usize; 3] {
        if self.stated().is_some() {
            return [1, 1, 1];
        }
        [
            self.ratios[0].up(),
            self.ratios[1].up(),
            self.ratios[2].up(),
        ]
    }

    /// The halo to grant over `grid`: the reach, raised per block and per side to
    /// put the read boundary on a multiple of [`Resample::alignment`].
    ///
    /// **Per block and per side, because the snapping is, and not as an
    /// optimisation.** A core starting two voxels into a cell needs two more
    /// below it and none above; the block after it needs the opposite. One
    /// number granted to every block — or one number granted to both sides of a
    /// block — puts most of the read boundaries *off* the lattice, and a read
    /// boundary off the lattice has no fetch region at all rather than a larger
    /// one. `tests/resample_ops.rs` provokes both refusals. Where the snapping is
    /// trivial — every downsampling factor, whose `up` is 1 — every entry agrees
    /// and this returns the uniform form, so a plan that does not need the table
    /// does not carry one.
    ///
    /// This is a **granted** halo and is deliberately not what
    /// [`Resample::reach_spec`] returns. The two are the quantities the tiling
    /// guard compares; deriving one from the other would make the guard compare a
    /// number against itself.
    ///
    /// **Nothing under a stated extent**, and for the same reason `lattice.rs`
    /// grants nothing: every output voxel is written by exactly one block, and
    /// what a block needs beyond its own core is in the image below and is stated
    /// as a fetch region rather than as a distance in this phase's voxels. There
    /// is no lattice to snap to and no reach to cover.
    pub fn halo(&self, grid: &BlockGrid) -> Reach {
        if self.stated().is_some() {
            return Reach::none();
        }
        let counts = grid.blocks_per_axis();
        let block = grid.block();
        let volume = grid.volume();
        let mut axes = [AxisReach::none(), AxisReach::none(), AxisReach::none()];
        for axis in 0..3 {
            let (reach_lo, reach_hi) = self.reach(axis);
            let step = self.ratios[axis].up() as i128;
            let mut table = Vec::with_capacity(counts[axis]);
            for index in 0..counts[axis] {
                let core_lo = index * block[axis];
                let core_hi = (core_lo + block[axis]).min(volume[axis]);
                // Down to the cell boundary at or below `core_lo - reach_lo`,
                // and up to the one at or above `core_hi + reach_hi`. A read
                // clamped at 0 or at the volume's end stays aligned: 0 is a
                // multiple of anything, and the far end is the one boundary
                // `source_region` resolves against the input volume instead.
                let below = (core_lo as i128 - reach_lo as i128).rem_euclid(step) as usize;
                let above = (-((core_hi + reach_hi) as i128)).rem_euclid(step) as usize;
                table.push((reach_lo + below, reach_hi + above));
            }
            let first = table[0];
            axes[axis] = if table.iter().all(|entry| *entry == first) {
                AxisReach::Bounded {
                    lo: first.0,
                    hi: first.1,
                }
            } else {
                AxisReach::PerBlock(table)
            };
        }
        Reach::per_axis(axes)
    }

    /// The source voxels on `axis` that the outputs `[start, start + len)`
    /// depend on: `(low, high)`, half-open, clamped to `input_len`.
    ///
    /// The **tight** interval, before any lattice alignment — which is exactly
    /// what makes it worth having in public. [`Resample::source_region`] states
    /// what a block *fetches*, and that is an aligned superset of this; the
    /// difference between the two is what the alignment costs, and a caller (or
    /// a test) that wants to price it needs both numbers rather than one number
    /// and an argument.
    pub fn source_span(
        &self,
        axis: usize,
        start: usize,
        len: usize,
        input_len: usize,
    ) -> (usize, usize) {
        if len == 0 || input_len == 0 {
            return (0, 0);
        }
        let ratio = self.ratios[axis];
        let last = input_len as i128 - 1;
        let (low, high) = match self.interpolation {
            Interpolation::Nearest => (ratio.nearest(start), ratio.nearest(start + len - 1)),
            Interpolation::Linear => (ratio.bracket(start).0, ratio.bracket(start + len - 1).1),
        };
        (
            low.clamp(0, last) as usize,
            high.clamp(0, last) as usize + 1,
        )
    }

    /// The region of the image below that a block reading `read` must fetch.
    ///
    /// The image of `read` under the factor, with one exception at the top of
    /// each axis: a read extent that ends at the output volume's edge fetches to
    /// the **input** volume's edge. That is what puts the voxels the factor
    /// dropped ([`Resample::unsampled_tail`]) on the same side of the arithmetic
    /// as everything else — `floor(n * up / down)` of the resulting extent is
    /// exactly the read extent, at that block as at every other, which is the
    /// identity [`BlockOp::output_shape`] rests on.
    ///
    /// It fails, naming the axis, on a read boundary that is not a multiple of
    /// [`Resample::alignment`]. A plan built by [`resample_phase`] cannot produce
    /// one; a plan built by hand can, and the refusal is how it finds out.
    pub fn source_region(&self, read: &Region, input_volume: [usize; 3]) -> Result<Region> {
        if read.ndim() != 3 {
            return Err(Error::InvalidArgument(format!(
                "resampling is 3-D, got a read region of rank {}",
                read.ndim()
            )));
        }
        let output_volume = self.output_volume(input_volume)?;
        // **The tight span, under a stated extent**, rather than an aligned
        // superset of it. There is nothing to align to: `alignment` is `[1, 1, 1]`
        // there, the write extent comes from the plan rather than from an
        // inversion of the fetch, and so the honest fetch is exactly what the
        // block's own output voxels bracket. `source_span` is that interval and
        // is already public for the callers that wanted to price the alignment
        // against it; here it *is* the answer.
        if self.stated().is_some() {
            let mut start = [0usize; 3];
            let mut shape = [0usize; 3];
            for axis in 0..3 {
                let low = read.start[axis];
                let len = read.shape[axis];
                if low + len > output_volume[axis] {
                    return Err(Error::InvalidArgument(format!(
                        "resampling: a block reading {low}..{} on axis {axis} runs past this \
                         phase's output volume of {}",
                        low + len,
                        output_volume[axis]
                    )));
                }
                let (source_low, source_high) =
                    self.source_span(axis, low, len, input_volume[axis]);
                start[axis] = source_low;
                shape[axis] = source_high - source_low;
            }
            return Ok(Region::new(&start, &shape));
        }
        let mut start = [0usize; 3];
        let mut shape = [0usize; 3];
        for axis in 0..3 {
            let ratio = self.ratios[axis];
            let low = read.start[axis];
            let high = low + read.shape[axis];
            let source_low = ratio.aligned_inputs(low).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "resampling: a block reading from {low} on axis {axis} has no fetch region — \
                     a fetch is the image of the read extent under {}/{}, so its boundaries have \
                     to be multiples of {}. `Resample::halo` is what puts them there.",
                    ratio.up(),
                    ratio.down(),
                    ratio.up()
                ))
            })?;
            let source_high = if high == output_volume[axis] {
                input_volume[axis]
            } else {
                ratio.aligned_inputs(high).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "resampling: a block reading up to {high} on axis {axis} has no fetch \
                         region — a fetch is the image of the read extent under {}/{}, so its \
                         boundaries have to be multiples of {} unless they are the volume's own \
                         edge.",
                        ratio.up(),
                        ratio.down(),
                        ratio.up()
                    ))
                })?
            };
            if source_high > input_volume[axis] || source_low > source_high {
                return Err(Error::InvalidArgument(format!(
                    "resampling: a block reading {low}..{high} on axis {axis} would fetch \
                     {source_low}..{source_high} of an image of {}",
                    input_volume[axis]
                )));
            }
            start[axis] = source_low;
            shape[axis] = source_high - source_low;
        }
        Ok(Region::new(&start, &shape))
    }
}

// -------------------------------------------------------------- kernels --

/// Where one axis of a buffer sits, and which of its voxels this call writes.
#[derive(Debug, Clone, Copy)]
struct Placement {
    source_start: usize,
    source_len: usize,
    output_start: usize,
    outputs: usize,
}

/// One output voxel's two samples on one axis, as **buffer** indices, and the
/// weight between them.
#[derive(Debug, Clone, Copy)]
struct Tap {
    low: usize,
    high: usize,
    weight: f64,
}

/// Resolve the anchor into what each axis is doing, or say why it cannot be.
///
/// The one derivation worth pointing at: the output position of the block is
/// recovered from where the *input* buffer sits, because that — and not the
/// output offset — is what an `Anchor` carries for a phase that reads across
/// grids. `Anchor::of_region(fetch, volume_at(phase))` is what the executor
/// builds, so a fetch starting at source voxel `f` is the image of output voxel
/// `f / down * up`, and a fetch that is not on the lattice is refused here
/// rather than silently resampled half a voxel out of phase.
fn placements(
    input: [usize; 3],
    at: &Anchor,
    resample: &Resample,
    out: [usize; 3],
) -> Result<[Placement; 3]> {
    let mut placements = [Placement {
        source_start: 0,
        source_len: 0,
        output_start: 0,
        outputs: 0,
    }; 3];
    for axis in 0..3 {
        if at.offset[axis] + input[axis] > at.volume[axis] {
            return Err(Error::InvalidArgument(format!(
                "resampling: a buffer of {input:?} at {:?} does not fit a volume of {:?}",
                at.offset, at.volume
            )));
        }
        let ratio = resample.ratio(axis);
        if at.offset[axis] % ratio.down() != 0 {
            return Err(Error::InvalidArgument(format!(
                "resampling: this buffer starts at {} on axis {axis}, which is not a multiple of \
                 {} and therefore is not the image of any output voxel under {}/{}. The fetch \
                 region a block is handed has to be the one `Resample::source_region` states.",
                at.offset[axis],
                ratio.down(),
                ratio.up(),
                ratio.down()
            )));
        }
        let outputs = ratio.outputs(input[axis]);
        if outputs != out[axis] {
            return Err(Error::InvalidArgument(format!(
                "resampling: {} voxels on axis {axis} produce {outputs} at {}/{}, and the output \
                 buffer holds {}",
                input[axis],
                ratio.up(),
                ratio.down(),
                out[axis]
            )));
        }
        placements[axis] = Placement {
            source_start: at.offset[axis],
            source_len: input[axis],
            output_start: at.offset[axis] / ratio.down() * ratio.up(),
            outputs,
        };
    }
    Ok(placements)
}

/// The same three placements from the two positions **stated** rather than one
/// derived from the other.
///
/// This is the arrangement `lattice_interpolate_into_with` has and it exists for
/// the same reason: under a stated output extent the block's first output voxel
/// is not a function of where its buffer starts, so the two ends of the map are
/// two facts and the plan holds both. The factor path keeps deriving one from the
/// other — `placements` above — so nothing it computes moves.
fn placements_at(
    input: [usize; 3],
    source_start: [usize; 3],
    output_start: [usize; 3],
    out: [usize; 3],
) -> [Placement; 3] {
    let mut placements = [Placement {
        source_start: 0,
        source_len: 0,
        output_start: 0,
        outputs: 0,
    }; 3];
    for axis in 0..3 {
        placements[axis] = Placement {
            source_start: source_start[axis],
            source_len: input[axis],
            output_start: output_start[axis],
            outputs: out[axis],
        };
    }
    placements
}

/// Every output voxel's samples on one axis, in buffer indices.
///
/// **The clamp is to the buffer**, which at a real volume boundary is the global
/// clamp — there is nothing beyond the end to have read — and at a block seam is
/// a truncation the whole volume would not have made. That is deliberate: it is
/// what makes a short halo produce a different number instead of a plausible one.
fn axis_taps(placement: &Placement, ratio: Ratio, interpolation: Interpolation) -> Vec<Tap> {
    let last = placement.source_len.saturating_sub(1) as i128;
    let origin = placement.source_start as i128;
    let clamp = |index: i128| (index - origin).clamp(0, last) as usize;
    (0..placement.outputs)
        .map(|step| {
            let output = placement.output_start + step;
            match interpolation {
                Interpolation::Nearest => {
                    let index = clamp(ratio.nearest(output));
                    Tap {
                        low: index,
                        high: index,
                        weight: 0.0,
                    }
                }
                Interpolation::Linear => {
                    let (low, high, weight) = ratio.bracket(output);
                    Tap {
                        low: clamp(low),
                        high: clamp(high),
                        weight,
                    }
                }
            }
        })
        .collect()
}

/// Resize `input` into `out` by taking the closest source voxel to each output
/// voxel's centre.
///
/// Generic over `T: Copy` and nothing else, which is exactly the requirement the
/// algorithm has: it copies a value it read and never combines two. That is what
/// makes it the resampling for **label data** — an integer volume of object ids,
/// a `bool` mask — where a blend would invent a value that names nothing.
///
/// `at` says where `input` sits in the image being read, and it is not
/// decoration: the sample lattice is anchored to the whole volume, so a block
/// computes the same output voxels from the same source voxels the whole volume
/// would. There is no way to ask for a lattice laid out relative to the buffer,
/// which is the defect `local.rs` records at length and this op would have in the
/// same form.
pub fn resample_nearest_into<T: Copy>(
    input: ArrayView3<'_, T>,
    at: &Anchor,
    ratios: &[Ratio; 3],
    out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    let shape = shape_of(&input);
    let target = [out.shape()[0], out.shape()[1], out.shape()[2]];
    let resample = Resample::new(*ratios, Interpolation::Nearest);
    let placements = placements(shape, at, &resample, target)?;
    nearest_core(input, &placements, ratios, out)
}

/// [`resample_nearest_into`] with **both** positions stated: where the buffer
/// sits in the image below, and which output voxel the block's first output is.
///
/// The pair is what an [`Anchor`] cannot carry and what a stated output extent
/// needs — see [`Resample::to_extent`]. Nothing else differs: the same taps, the
/// same clamp to the buffer, the same values.
pub fn resample_nearest_into_with<T: Copy>(
    input: ArrayView3<'_, T>,
    source_start: [usize; 3],
    output_start: [usize; 3],
    ratios: &[Ratio; 3],
    out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    let shape = shape_of(&input);
    let target = [out.shape()[0], out.shape()[1], out.shape()[2]];
    let placements = placements_at(shape, source_start, output_start, target);
    nearest_core(input, &placements, ratios, out)
}

fn nearest_core<T: Copy>(
    input: ArrayView3<'_, T>,
    placements: &[Placement; 3],
    ratios: &[Ratio; 3],
    mut out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    let shape = shape_of(&input);
    let target = [out.shape()[0], out.shape()[1], out.shape()[2]];
    if shape.contains(&0) || target.contains(&0) {
        return Ok(());
    }
    let taps: Vec<Vec<Tap>> = (0..3)
        .map(|axis| axis_taps(&placements[axis], ratios[axis], Interpolation::Nearest))
        .collect();
    for i in 0..target[0] {
        let a = taps[0][i].low;
        for j in 0..target[1] {
            let b = taps[1][j].low;
            for k in 0..target[2] {
                out[[i, j, k]] = input[[a, b, taps[2][k].low]];
            }
        }
    }
    Ok(())
}

/// Resize `input` into `out` by interpolating linearly between the bracketing
/// source voxels on each axis.
///
/// Generic over [`VoxelElement`], which is where this kernel's bound has to
/// stop: interpolating **combines** two values, so it needs a numeric currency,
/// and `VoxelElement` is the crate's own bridge to one. The accumulation is in
/// `f64` whatever the input holds — a blend of two `u8`s is not a `u8` — and the
/// result is converted back at the end, **rounded to nearest** for an integer or
/// `bool`-shaped type and passed through for a float. Truncating instead would
/// bias every interpolated volume down by half a step.
///
/// The weight is applied as `a + t * (b - a)` rather than `(1 - t) * a + t * b`,
/// for `local.rs`'s reason and with its consequence: where `a == b` the first
/// form returns `a` **exactly**, whatever `t` is, and that is what makes
/// [`BlockOp::constant_maps_to`] a declaration rather than an approximation.
pub fn resample_linear_into<T: VoxelElement>(
    input: ArrayView3<'_, T>,
    at: &Anchor,
    ratios: &[Ratio; 3],
    out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    let shape = shape_of(&input);
    let target = [out.shape()[0], out.shape()[1], out.shape()[2]];
    let resample = Resample::new(*ratios, Interpolation::Linear);
    let placements = placements(shape, at, &resample, target)?;
    linear_core(input, &placements, ratios, out)
}

/// [`resample_linear_into`] with **both** positions stated. See
/// [`resample_nearest_into_with`].
pub fn resample_linear_into_with<T: VoxelElement>(
    input: ArrayView3<'_, T>,
    source_start: [usize; 3],
    output_start: [usize; 3],
    ratios: &[Ratio; 3],
    out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    let shape = shape_of(&input);
    let target = [out.shape()[0], out.shape()[1], out.shape()[2]];
    let placements = placements_at(shape, source_start, output_start, target);
    linear_core(input, &placements, ratios, out)
}

fn linear_core<T: VoxelElement>(
    input: ArrayView3<'_, T>,
    placements: &[Placement; 3],
    ratios: &[Ratio; 3],
    mut out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    let shape = shape_of(&input);
    let target = [out.shape()[0], out.shape()[1], out.shape()[2]];
    if shape.contains(&0) || target.contains(&0) {
        return Ok(());
    }
    let taps: Vec<Vec<Tap>> = (0..3)
        .map(|axis| axis_taps(&placements[axis], ratios[axis], Interpolation::Linear))
        .collect();
    let round = !matches!(T::DTYPE, Dtype::F32 | Dtype::F64);
    for i in 0..target[0] {
        let ti = taps[0][i];
        for j in 0..target[1] {
            let tj = taps[1][j];
            for k in 0..target[2] {
                let tk = taps[2][k];
                let along_k = |a: usize, b: usize| {
                    lerp(
                        input[[a, b, tk.low]].into_f64(),
                        input[[a, b, tk.high]].into_f64(),
                        tk.weight,
                    )
                };
                let along_j = |a: usize| lerp(along_k(a, tj.low), along_k(a, tj.high), tj.weight);
                let value = lerp(along_j(ti.low), along_j(ti.high), ti.weight);
                out[[i, j, k]] = T::from_f64(if round { value.round() } else { value });
            }
        }
    }
    Ok(())
}

/// `a + t * (b - a)`, which returns `a` exactly where `a == b`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

fn shape_of<T>(input: &ArrayView3<'_, T>) -> [usize; 3] {
    [input.shape()[0], input.shape()[1], input.shape()[2]]
}

// ---------------------------------------------------------------- shell --

/// Resize the volume by a rational factor per axis.
pub struct ResampleOp {
    name: &'static str,
    resample: Resample,
    cost: f64,
}

impl ResampleOp {
    pub fn new(name: &'static str, resample: Resample) -> Self {
        let cost = cost_for(&resample);
        Self {
            name,
            resample,
            cost,
        }
    }

    /// The parameters, for a caller building the plan this op needs —
    /// [`resample_phase`] takes exactly this.
    pub fn resample(&self) -> &Resample {
        &self.resample
    }

    /// Override the measured cost — for a caller whose machine is not the one
    /// `super::COST_MEASUREMENT` was taken on.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// The element-type dispatch, for both entry points.
    ///
    /// One match rather than two, because the two entry points differ in exactly
    /// one thing — where the block's first output voxel is — and duplicating
    /// twenty-two arms to say that would be twenty-two places for the two paths
    /// to drift apart in.
    fn dispatch(&self, input: &Voxels, out: &mut Voxels, at: &Placed<'_>) -> Result<()> {
        let ratios = &[
            self.resample.ratio(0),
            self.resample.ratio(1),
            self.resample.ratio(2),
        ];

        /// The nearest case: straight into the kernel, no copy and no widening,
        /// for every type the enum holds.
        fn nearest<T: VoxelElement>(
            input: &Voxels,
            out: &mut Voxels,
            r: &[Ratio; 3],
            at: &Placed<'_>,
        ) -> Result<()> {
            let view = input.view::<T>()?;
            let target = out.view_mut::<T>()?;
            match at {
                Placed::Anchored(anchor) => resample_nearest_into(view, anchor, r, target),
                Placed::At { source, output } => {
                    resample_nearest_into_with(view, *source, *output, r, target)
                }
            }
        }

        fn linear<T: VoxelElement>(
            input: &Voxels,
            out: &mut Voxels,
            r: &[Ratio; 3],
            at: &Placed<'_>,
        ) -> Result<()> {
            let view = input.view::<T>()?;
            let target = out.view_mut::<T>()?;
            match at {
                Placed::Anchored(anchor) => resample_linear_into(view, anchor, r, target),
                Placed::At { source, output } => {
                    resample_linear_into_with(view, *source, *output, r, target)
                }
            }
        }

        match (self.resample.interpolation(), input.dtype()) {
            (_, Dtype::F16) => Err(Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; `accepts` refuses it before a run starts",
                self.name
            ))),
            (Interpolation::Linear, Dtype::Bool) => Err(Error::InvalidArgument(format!(
                "{}: a linear blend of a mask is not a mask — every non-zero weight would round \
                 to `true`, which is a dilation nobody asked for. `accepts` refuses this before a \
                 run starts; nearest is the resampling a mask has.",
                self.name
            ))),
            (Interpolation::Nearest, Dtype::Bool) => nearest::<bool>(input, out, ratios, at),
            (Interpolation::Nearest, Dtype::U8) => nearest::<u8>(input, out, ratios, at),
            (Interpolation::Nearest, Dtype::U16) => nearest::<u16>(input, out, ratios, at),
            (Interpolation::Nearest, Dtype::U32) => nearest::<u32>(input, out, ratios, at),
            (Interpolation::Nearest, Dtype::U64) => nearest::<u64>(input, out, ratios, at),
            (Interpolation::Nearest, Dtype::I8) => nearest::<i8>(input, out, ratios, at),
            (Interpolation::Nearest, Dtype::I16) => nearest::<i16>(input, out, ratios, at),
            (Interpolation::Nearest, Dtype::I32) => nearest::<i32>(input, out, ratios, at),
            (Interpolation::Nearest, Dtype::I64) => nearest::<i64>(input, out, ratios, at),
            (Interpolation::Nearest, Dtype::F32) => nearest::<f32>(input, out, ratios, at),
            (Interpolation::Nearest, Dtype::F64) => nearest::<f64>(input, out, ratios, at),
            (Interpolation::Linear, Dtype::U8) => linear::<u8>(input, out, ratios, at),
            (Interpolation::Linear, Dtype::U16) => linear::<u16>(input, out, ratios, at),
            (Interpolation::Linear, Dtype::U32) => linear::<u32>(input, out, ratios, at),
            (Interpolation::Linear, Dtype::U64) => linear::<u64>(input, out, ratios, at),
            (Interpolation::Linear, Dtype::I8) => linear::<i8>(input, out, ratios, at),
            (Interpolation::Linear, Dtype::I16) => linear::<i16>(input, out, ratios, at),
            (Interpolation::Linear, Dtype::I32) => linear::<i32>(input, out, ratios, at),
            (Interpolation::Linear, Dtype::I64) => linear::<i64>(input, out, ratios, at),
            (Interpolation::Linear, Dtype::F32) => linear::<f32>(input, out, ratios, at),
            (Interpolation::Linear, Dtype::F64) => linear::<f64>(input, out, ratios, at),
        }
    }
}

/// Where a block sits, in whichever of the two forms its caller has.
///
/// The factor path derives the output position from the buffer's own — the
/// fetch is the image of the read extent, so one number states both — and the
/// stated-extent path cannot, so it carries the pair. Private, because it is the
/// seam between the two entry points and not a third contract.
enum Placed<'a> {
    Anchored(&'a Anchor),
    At {
        source: [usize; 3],
        output: [usize; 3],
    },
}

impl BlockOp for ResampleOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The symmetric bound on [`Self::reach_spec`], which is what a caller
    /// holding one integer per axis must use. Derived from the factor and the
    /// interpolation; there is no field that could set it.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        let (lo, hi) = self.resample.reach(axis);
        lo.max(hi)
    }

    /// The per-side statement. See [`Resample::reach_spec`] for why it is in the
    /// phase's own frame.
    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        self.resample.reach_spec()
    }

    /// Every element type a buffer can hold, less what the interpolation cannot
    /// mean: `f16` is held by nothing, and `bool` has no midpoint.
    fn accepts(&self, dtype: Dtype) -> bool {
        self.resample.interpolation().accepts(dtype)
    }

    /// **The first op to state a map rather than a reach.**
    ///
    /// The two halves it separates are already separate in `Resample` and were
    /// merely reported through two methods: the factor decides the output
    /// volume, and the interpolation's taps decide what a halo has to cover.
    /// `Geometry` says both in one place, and `Chain` derives the reach from it
    /// rather than asking a second time.
    ///
    /// Per-axis rather than through `output_volume`, which is fallible: an
    /// extent is a number this op can always produce, and a resampling of a
    /// volume that cannot be represented is caught where the plan is built.
    ///
    /// **Under a stated extent the map is a stencil in `Units::SourceIndex`**,
    /// which is `lattice.rs`'s declaration and says the honest thing: an
    /// `InputMap::Affine` would name a factor a halo could be grown by, and the
    /// dependency here is a region stated per block. The output volume is the
    /// stated one, whatever `input_volume` is — the op is bound to the volume it
    /// was stated for, and `output_volume` is where being asked about another is
    /// refused.
    fn geometry(&self, input_volume: [usize; 3]) -> Geometry {
        if let Some((_, output)) = self.resample.stated() {
            return Geometry::new(output, vec![InputMap::Stencil(self.resample.reach_spec())]);
        }
        let mut up = [0usize; 3];
        let mut down = [0usize; 3];
        let mut output = [0usize; 3];
        for axis in 0..3 {
            let ratio = self.resample.ratio(axis);
            up[axis] = ratio.up();
            down[axis] = ratio.down();
            output[axis] = self.resample.output_extent(axis, input_volume[axis]);
        }
        Geometry::new(
            output,
            vec![InputMap::Affine {
                up,
                down,
                window: self.resample.reach_spec(),
            }],
        )
    }

    /// The declaration a resizing phase exists on: `floor(n * up / down)` per
    /// axis, a function of the shape and nothing else.
    ///
    /// **Under a stated extent it is a function of the shape only for the whole
    /// volume**, and the honest answer for anything else is that the question has
    /// no answer — [`Self::placed_output_shape`] is where it is asked of
    /// something that can answer it. The input's own shape is returned instead,
    /// which is deliberately a shape the executor will reject if it ever reaches
    /// it, rather than an arithmetic that would be right for the interior blocks
    /// and quietly wrong for the two at the ends. `lattice.rs` makes the same
    /// move for the same reason.
    fn output_shape(&self, input: [usize; 3]) -> [usize; 3] {
        if let Some((stated_input, stated_output)) = self.resample.stated() {
            if input == stated_input {
                return stated_output;
            }
            return input;
        }
        [
            self.resample.output_extent(0, input[0]),
            self.resample.output_extent(1, input[1]),
            self.resample.output_extent(2, input[2]),
        ]
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        self.dispatch(input, out, &Placed::Anchored(at))
    }

    /// **The waiver, stated.** Under a stated output extent this op's write
    /// extent is not a function of its read extent — two blocks fetching the
    /// same number of source voxels write different numbers of output voxels,
    /// because the sampling scale is `n/out` and `out` divides nothing — so
    /// [`Self::placed_output_shape`] answers out of the plan, and
    /// [`crate::decomposition::check_output_shapes`] would refuse the plan
    /// without this.
    ///
    /// `false` under the factor-exact default, where `output_shape` inverts the
    /// fetch at every block and the executor's own comparison is a real check.
    /// The check this buys instead is in [`Self::apply_placed`].
    fn takes_extent_from_placement(&self) -> bool {
        matches!(self.resample.extent(), OutputExtent::Stated { .. })
    }

    /// The extent the **plan** states for this block, under a stated output
    /// extent, and [`Self::output_shape`] otherwise.
    ///
    /// A [`OpPlacement`] whose output anchor is in this op's own output space is
    /// a placement from an executor running this phase, and its `writes` is the
    /// block's read region — cut from the output volume, which is what this
    /// phase's grid is cut from. Anything else is a caller applying the chain by
    /// hand, and falls through.
    fn placed_output_shape(&self, input: [usize; 3], at: &OpPlacement) -> [usize; 3] {
        if let OutputExtent::Stated { output, .. } = self.resample.extent() {
            if at.output.volume == output {
                if let Some(extent) = at.writes() {
                    return extent;
                }
            }
        }
        self.output_shape(input)
    }

    /// Resample the block's own output voxels from the source voxels it was
    /// handed, with **both** positions taken from the placement.
    ///
    /// The factor path is untouched: it has no second position to use, so it
    /// falls through to [`Self::apply`] exactly as the default would.
    ///
    /// **The check [`Self::placed_output_shape`] traded away** is the block of
    /// code below, and it is the stronger of the two because it is against the
    /// buffer rather than against a declaration: every output voxel this block
    /// writes has to bracket between source voxels the buffer actually holds.
    /// `bracket` is monotone in the coordinate, so the first and last output
    /// voxel bound the rest and [`Resample::source_span`] is exactly that
    /// interval. A short fetch is refused here by name; clamping to the buffer's
    /// edge instead — which is what [`axis_taps`] would do, deliberately, one
    /// line later — would produce a plausible volume that the whole-volume answer
    /// does not agree with.
    fn apply_placed(
        &self,
        input: &Voxels,
        sources: crate::op::SourceInputs<'_>,
        out: &mut Voxels,
        at: &OpPlacement,
    ) -> Result<()> {
        let Some((stated_input, stated_output)) = self.resample.stated() else {
            return self.apply_with(input, sources, out, &at.input);
        };
        if at.output.volume != stated_output {
            // Not a placement in this op's own output space, so there is no
            // second position to use: the anchor path states what it can and
            // names what is missing.
            return self.apply_with(input, sources, out, &at.input);
        }
        if at.input.volume != stated_input {
            return Err(Error::InvalidArgument(format!(
                "op {:?}: this resampling states {stated_output:?} from an input volume of \
                 {stated_input:?}, and the placement says the image being read is {:?}",
                self.name, at.input.volume
            )));
        }
        let written = out.shape();
        let held = input.shape();
        let offset = at.input.offset;
        let origin = at.output.offset;
        for axis in 0..3 {
            if written[axis] == 0 || held[axis] == 0 {
                continue;
            }
            if origin[axis] + written[axis] > stated_output[axis] {
                return Err(Error::InvalidArgument(format!(
                    "op {:?}: a block writing output voxels {}..{} of axis {axis} runs past this \
                     phase's output volume of {}",
                    self.name,
                    origin[axis],
                    origin[axis] + written[axis],
                    stated_output[axis]
                )));
            }
            let (low, high) =
                self.resample
                    .source_span(axis, origin[axis], written[axis], stated_input[axis]);
            if low < offset[axis] || high > offset[axis] + held[axis] {
                return Err(Error::InvalidArgument(format!(
                    "op {:?}: a block writing output voxels {}..{} of axis {axis} reads source \
                     {low}..{high} and was handed source {}..{}. Every voxel this block writes has \
                     to bracket between source voxels the buffer holds; clamping to the buffer's \
                     edge instead would produce a plausible volume that the whole-volume answer \
                     does not agree with.",
                    self.name,
                    origin[axis],
                    origin[axis] + written[axis],
                    offset[axis],
                    offset[axis] + held[axis]
                )));
            }
        }
        self.dispatch(
            input,
            out,
            &Placed::At {
                source: offset,
                output: origin,
            },
        )
    }

    /// Exactly the constant, for both interpolations — and the two halves are
    /// exact for two different reasons, which is why this is not one sentence.
    ///
    /// * **Nearest** writes a value it read. If every value read is `value`, the
    ///   value written is `value`, bit for bit, at any factor and at any
    ///   truncation of the window by the volume's edge.
    /// * **Linear** evaluates `a + t * (b - a)`. Where `a == b` that is `a + t *
    ///   0`, which is `a` **exactly** in binary floating point for every finite
    ///   `a` and `t` — unlike `(1 - t) * a + t * b`, which is not. The three axes
    ///   compose the same way, each one's input being the previous one's exact
    ///   `a`. `local.rs` chose that form for this reason and pins it with a test;
    ///   `a_declared_constant_is_the_computed_one` below pins it again here,
    ///   through `apply`, in every element type this op accepts.
    ///
    /// The value flows through `f64` in both directions, which is where the
    /// short circuit meets it: `Environment::uniform` widens the buffer's value
    /// the same way `Voxels::filled` narrows this one back, so the block a skip
    /// produces is the block the work would have produced.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Measured; see `super::COST_MEASUREMENT` for the method, and
/// `resample::tests::print_the_cost_table` for the command that retakes it.
///
/// **Per output voxel**, which is the unit the planner charges in: `price_phase`
/// multiplies by the phase's own core voxels, and a resampling phase's cores are
/// cut from its output volume. A nearest pass is one gather per output voxel; a
/// linear pass is eight gathers and seven blends.
///
/// **What the measurement showed that the arithmetic does not.** Over 96x64x64,
/// best of five, relative to the voxelwise map: nearest cost **0.31** per output
/// voxel at 3/2 and **0.76** at 1/2, and linear **1.50** and **1.57** at the
/// same two factors. Nearest is therefore not one number — halving reads its
/// gathers a stride apart and pays for the locality, where growing reads each
/// source voxel several times in a row and does not. The stored figure is the
/// **higher** of the two, so a plan is never told that a resampling phase is
/// cheaper than it turned out to be; the shape of the dependence (on the factor,
/// through locality) is recorded here rather than modelled, because two points
/// are not a model.
pub(super) fn cost_for(resample: &Resample) -> f64 {
    match resample.interpolation() {
        Interpolation::Nearest => NEAREST_COST,
        Interpolation::Linear => LINEAR_COST,
    }
}

/// Measured; see `super::COST_MEASUREMENT` and [`cost_for`].
pub(super) const NEAREST_COST: f64 = 0.76;
/// Measured; see `super::COST_MEASUREMENT` and [`cost_for`].
pub(super) const LINEAR_COST: f64 = 1.58;

// ----------------------------------------------------------------- plan --

/// The phase a resampling op needs: the grid over its **output** volume, the
/// halo that puts every fetch on the source lattice, and each block's fetch
/// region.
///
/// **Shipped with the op rather than written at the call site**, because the
/// fetch mapping is the op's own geometry and a plan that states it differently
/// from the way the op reads it is a plan whose blocks are silently half a voxel
/// out. The executor would catch the gross case — it compares
/// `output_shape(fetch)` against the read extent and names both — but not a
/// mapping that has the right shape in the wrong place. So there is one
/// derivation and both halves use it.
///
/// `grid` is over the output volume and any block edge is allowed: the halo,
/// not the caller, buys the alignment the fetch needs.
///
/// **Under a stated output extent it buys nothing, because there is nothing to
/// buy.** The halo is `Reach::none()`, the reach is `Units::SourceIndex` zero,
/// and each block's fetch is its own tight source span — so `valid` is `core` at
/// every block, any edge plans, and the cost of an edge is zero rather than the
/// 1.00x-2.01x the factor path's alignment measures. What pays for that is the
/// waiver: the write extent comes from the plan and
/// [`ResampleOp::apply_placed`] owes the check against the buffer.
pub fn resample_phase(
    slots: Vec<usize>,
    names: Vec<String>,
    resample: &Resample,
    input_volume: [usize; 3],
    grid: BlockGrid,
) -> Result<PhaseDecomposition> {
    let output_volume = resample.output_volume(input_volume)?;
    if grid.volume() != output_volume {
        return Err(Error::InvalidArgument(format!(
            "a resampling phase's grid is cut from what it writes: {input_volume:?} resampled is \
             {output_volume:?}, and this grid is over {:?}.",
            grid.volume()
        )));
    }
    let halo = resample.halo(&grid);
    let phase = PhaseDecomposition::derive(slots, names, resample.reach_spec(), halo, grid);
    // Resolved before the plan is built rather than inside the mapping closure,
    // which cannot fail: a misaligned read is a fact about the plan and has to
    // be reportable as one.
    let sources = phase
        .blocks
        .iter()
        .map(|block| resample.source_region(&block.read, input_volume))
        .collect::<Result<Vec<Region>>>()?;
    check_fetch_covers_valid(resample, &phase.blocks, &sources, input_volume)?;
    Ok(phase.with_sources(|block| sources[block.flat].clone()))
}

/// The one thing the framework cannot check for a resampling phase: that each
/// block's fetch contains what its **trustworthy** output voxels read.
///
/// The tiling guard compares a granted halo against a required reach, both in
/// the phase's own output space, and never looks at `source` at all — so it
/// passes a fetch mapping that is the right size in the wrong place, and it
/// passes a reach that is too small for the fetch the plan states. Those are the
/// two ways this phase can be wrong without any existing guard noticing, and
/// this is where they are caught, by asking the sample geometry directly rather
/// than by trusting the derivation that produced both numbers.
fn check_fetch_covers_valid(
    resample: &Resample,
    blocks: &[crate::geometry::BlockGeometry],
    sources: &[Region],
    input_volume: [usize; 3],
) -> Result<()> {
    for (block, source) in blocks.iter().zip(sources) {
        for axis in 0..3 {
            if block.valid.shape[axis] == 0 {
                continue;
            }
            let (low, high) = resample.source_span(
                axis,
                block.valid.start[axis],
                block.valid.shape[axis],
                input_volume[axis],
            );
            let fetched = source.start[axis]..source.start[axis] + source.shape[axis];
            if low < fetched.start || high > fetched.end {
                return Err(Error::InvalidArgument(format!(
                    "resampling block {:?}: its trustworthy output voxels on axis {axis} read \
                     source {low}..{high}, and the fetch this plan states is {fetched:?}. Nothing \
                     else checks that pair — the tiling guard compares distances in the output \
                     space and never looks at where a block reads.",
                    block.index
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn ramp(shape: [usize; 3]) -> Array3<f64> {
        Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
            ((i * 7919 + j * 104729 + k * 1299709) % 1013) as f64 + 1.0
        })
    }

    fn ratios(specs: [(usize, usize); 3]) -> [Ratio; 3] {
        [
            Ratio::new(specs[0].0, specs[0].1).unwrap(),
            Ratio::new(specs[1].0, specs[1].1).unwrap(),
            Ratio::new(specs[2].0, specs[2].1).unwrap(),
        ]
    }

    /// Resample a whole volume in one call, which is the reference every
    /// decomposition is compared against.
    fn whole(input: &Array3<f64>, resample: &Resample) -> Array3<f64> {
        let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
        let volume = resample.output_volume(shape).unwrap();
        let mut out = Array3::zeros((volume[0], volume[1], volume[2]));
        let all = [resample.ratio(0), resample.ratio(1), resample.ratio(2)];
        match resample.interpolation() {
            Interpolation::Nearest => {
                resample_nearest_into(input.view(), &Anchor::whole(shape), &all, out.view_mut())
                    .unwrap()
            }
            Interpolation::Linear => {
                resample_linear_into(input.view(), &Anchor::whole(shape), &all, out.view_mut())
                    .unwrap()
            }
        }
        out
    }

    #[test]
    fn a_factor_is_exact_and_reduced() {
        let ratio = Ratio::new(6, 4).unwrap();
        assert_eq!((ratio.up(), ratio.down()), (3, 2));
        assert_eq!(ratio.outputs(24), 36);
        // floored, and the tail it drops is named
        assert_eq!(Ratio::smaller(5).unwrap().outputs(24), 4);
        assert!(Ratio::new(0, 3).is_err());
        assert!(Ratio::new(3, 0).is_err());
        assert!(Ratio::identity().is_identity());
    }

    #[test]
    fn the_dropped_tail_is_reported_rather_than_hidden() {
        let resample = Resample::uniform(Ratio::smaller(5).unwrap(), Interpolation::Nearest);
        // 24 / 5 is 4 output voxels over 20 input voxels; 4 are never sampled
        assert_eq!(resample.output_extent(0, 24), 4);
        assert_eq!(resample.unsampled_tail(0, 24), 4);
        // and a factor that divides evenly drops nothing
        let even = Resample::uniform(Ratio::smaller(4).unwrap(), Interpolation::Nearest);
        assert_eq!(even.unsampled_tail(0, 24), 0);
        // an axis that would be emptied is refused by name
        let message = Resample::uniform(Ratio::smaller(8).unwrap(), Interpolation::Nearest)
            .output_volume([5, 40, 40])
            .unwrap_err()
            .to_string();
        assert!(message.contains("produces no voxel at all"), "{message}");
    }

    /// The map is the centred one, stated as the fraction it is.
    #[test]
    fn the_sample_map_is_centred_and_computed_exactly() {
        // halving: output o takes source 2o + 1 (the tie goes up)
        let half = Ratio::smaller(2).unwrap();
        assert_eq!(half.nearest(0), 1);
        assert_eq!(half.nearest(3), 7);
        // doubling: x(o) = o/2 - 1/4, so the weights alternate 3/4 and 1/4 and
        // the sample below the volume is asked for at o = 0
        let twice = Ratio::larger(2).unwrap();
        assert_eq!(twice.bracket(0), (-1, 0, 0.75));
        assert_eq!(twice.bracket(1), (0, 1, 0.25));
        assert_eq!(twice.bracket(2), (0, 1, 0.75));
        // nearest at the same factor is the pair-wise repeat, ties going up
        assert_eq!(
            (0..4).map(|o| twice.nearest(o)).collect::<Vec<i128>>(),
            vec![0, 0, 1, 1]
        );
        // the identity samples itself, with no second sample at all
        assert_eq!(Ratio::identity().bracket(7), (7, 7, 0.0));
        assert_eq!(Ratio::identity().nearest(7), 7);
    }

    /// The reach is derived from the factor and the interpolation, and is
    /// **tight to one voxel on each side**: the check re-derives it by finding,
    /// for every factor, the exact set of output voxels whose samples fall
    /// outside the image of a fetch, and compares that with what is declared.
    #[test]
    fn the_reach_is_derived_and_tight_on_each_side() {
        for up in 1..=8usize {
            for down in 1..=8usize {
                let ratio = Ratio::new(up, down).unwrap();
                // nearest never leaves the image of its own extent
                assert_eq!(ratio.reach(Interpolation::Nearest), (0, 0), "{up}/{down}");

                let (lo, hi) = ratio.reach(Interpolation::Linear);
                // A fetch for the aligned read extent [s, e) covers source
                // [s*down/up, e*down/up). The two boundaries are independent
                // questions, so they are asked separately — and one period of
                // outputs is enough to settle each, because `x(o + up)` is
                // `x(o) + down` and the pattern of exact coordinates repeats.
                let period = ratio.up();
                let start = 8 * period;
                let end = start + 8 * period;
                let source_low = (start / period * ratio.down()) as i128;
                let source_high = (end / period * ratio.down()) as i128;
                let below_is_inside =
                    |distance: usize| ratio.bracket(start + distance).0 >= source_low;
                let above_is_inside =
                    |distance: usize| ratio.bracket(end - 1 - distance).1 < source_high;
                let wanted_lo = (0..)
                    .find(|&d| (d..d + period).all(below_is_inside))
                    .unwrap();
                let wanted_hi = (0..)
                    .find(|&d| (d..d + period).all(above_is_inside))
                    .unwrap();
                assert_eq!((lo, hi), (wanted_lo, wanted_hi), "{up}/{down} linear");
            }
        }
        // and the shape of the answer, spelled out for three of them
        let down3 = Resample::uniform(Ratio::smaller(3).unwrap(), Interpolation::Linear);
        assert_eq!(down3.reach(0), (0, 0));
        let up2 = Resample::uniform(Ratio::larger(2).unwrap(), Interpolation::Linear);
        assert_eq!(up2.reach(0), (1, 1));
        let up4 = Resample::uniform(Ratio::larger(4).unwrap(), Interpolation::Linear);
        assert_eq!(up4.reach(0), (2, 2));
    }

    /// The identity of the whole arrangement: what a block of `n` fetched voxels
    /// produces is what the plan derived for it, at every block of every grid.
    ///
    /// This is the equality `strategy.rs` compares at run time, checked here over
    /// far more grids than a run would visit — including the last block of an
    /// axis, where the fetch runs to the input volume's edge and the arithmetic
    /// is not the interior case.
    #[test]
    fn the_declared_output_shape_inverts_the_fetch_at_every_block() {
        for up in 1..=5usize {
            for down in 1..=5usize {
                let resample =
                    Resample::uniform(Ratio::new(up, down).unwrap(), Interpolation::Linear);
                for extent in [7usize, 11, 16, 17, 24, 31] {
                    let input = [extent, 4, 4];
                    let Ok(volume) = resample.output_volume(input) else {
                        continue;
                    };
                    for edge in 1..=volume[0] {
                        let grid = BlockGrid::along(volume, &[0], edge).unwrap();
                        let phase = resample_phase(
                            vec![0],
                            vec!["resample".to_string()],
                            &resample,
                            input,
                            grid,
                        )
                        .unwrap();
                        for block in &phase.blocks {
                            let fetched = [
                                block.source.shape[0],
                                block.source.shape[1],
                                block.source.shape[2],
                            ];
                            let produced = [
                                resample.output_extent(0, fetched[0]),
                                resample.output_extent(1, fetched[1]),
                                resample.output_extent(2, fetched[2]),
                            ];
                            assert_eq!(
                                produced.to_vec(),
                                block.read.shape,
                                "{up}/{down} extent {extent} edge {edge} block {:?}",
                                block.index
                            );
                            assert!(block.valid == block.core, "{:?}", block.index);
                        }
                    }
                }
            }
        }
    }

    /// A sub-buffer reproduces the whole-volume answer inside its trustworthy
    /// extent, and **not** outside it — which is the pair of assertions that
    /// makes the reach a measurement rather than a decoration.
    #[test]
    fn a_block_reproduces_the_whole_volume_answer_exactly_where_it_declares_it_does() {
        let input = ramp([24, 5, 5]);
        for interpolation in [Interpolation::Nearest, Interpolation::Linear] {
            for factor in [(1usize, 2usize), (1, 3), (2, 3), (3, 1), (3, 2), (5, 2)] {
                let resample = Resample::new(
                    [
                        Ratio::new(factor.0, factor.1).unwrap(),
                        Ratio::identity(),
                        Ratio::identity(),
                    ],
                    interpolation,
                );
                let reference = whole(&input, &resample);
                let volume = resample.output_volume([24, 5, 5]).unwrap();
                let (lo, hi) = resample.reach(0);
                let up = resample.ratio(0).up();
                // an interior read extent, on the lattice at both ends
                let read = Region::new(&[3 * up, 0, 0], &[4 * up, 5, 5]);
                if read.start[0] + read.shape[0] >= volume[0] {
                    continue;
                }
                let source = resample.source_region(&read, [24, 5, 5]).unwrap();
                let buffer = input
                    .slice(ndarray::s![
                        source.start[0]..source.start[0] + source.shape[0],
                        ..,
                        ..
                    ])
                    .to_owned();
                let mut got = Array3::zeros((read.shape[0], read.shape[1], read.shape[2]));
                let all = [resample.ratio(0), resample.ratio(1), resample.ratio(2)];
                let at = Anchor::new([source.start[0], 0, 0], [24, 5, 5]);
                match interpolation {
                    Interpolation::Nearest => {
                        resample_nearest_into(buffer.view(), &at, &all, got.view_mut()).unwrap()
                    }
                    Interpolation::Linear => {
                        resample_linear_into(buffer.view(), &at, &all, got.view_mut()).unwrap()
                    }
                }
                for step in lo..read.shape[0] - hi {
                    let output = read.start[0] + step;
                    assert_eq!(
                        got[[step, 2, 2]],
                        reference[[output, 2, 2]],
                        "{factor:?} {interpolation:?} at {output}"
                    );
                }
                // and one voxel closer to the seam than the reach allows, the
                // clamp at the buffer's edge shows: the declared distance is not
                // one voxel larger than it needs to be
                if lo > 0 {
                    assert_ne!(
                        got[[lo - 1, 2, 2]],
                        reference[[read.start[0] + lo - 1, 2, 2]],
                        "{factor:?}: the low reach could be smaller than declared"
                    );
                }
                if hi > 0 {
                    let step = read.shape[0] - hi;
                    assert_ne!(
                        got[[step, 2, 2]],
                        reference[[read.start[0] + step, 2, 2]],
                        "{factor:?}: the high reach could be smaller than declared"
                    );
                }
            }
        }
    }

    /// Nearest resampling of labels: the values written are values that were
    /// read, so a label volume stays a label volume.
    #[test]
    fn nearest_carries_labels_through_untouched() {
        let labels = Array3::from_shape_fn((8, 4, 4), |(i, j, k)| (i * 16 + j * 4 + k) as u16 + 1);
        let ratios = ratios([(1, 2), (1, 1), (1, 1)]);
        let mut out = Array3::<u16>::zeros((4, 4, 4));
        resample_nearest_into(
            labels.view(),
            &Anchor::whole([8, 4, 4]),
            &ratios,
            out.view_mut(),
        )
        .unwrap();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    // the tie goes up: output i takes source 2i + 1
                    assert_eq!(out[[i, j, k]], labels[[2 * i + 1, j, k]]);
                }
            }
        }
        // every value in the output is a value that was in the input
        assert!(out
            .iter()
            .all(|value| labels.iter().any(|had| had == value)));
    }

    /// A mask is resampled by nearest and refused by linear, in the shell and at
    /// plan time rather than at the block.
    #[test]
    fn a_mask_is_resampled_by_nearest_and_refused_by_linear() {
        let nearest = ResampleOp::new(
            "shrink",
            Resample::uniform(Ratio::smaller(2).unwrap(), Interpolation::Nearest),
        );
        assert!(nearest.accepts(Dtype::Bool));
        let linear = ResampleOp::new(
            "shrink",
            Resample::uniform(Ratio::smaller(2).unwrap(), Interpolation::Linear),
        );
        assert!(!linear.accepts(Dtype::Bool));
        assert!(linear.accepts(Dtype::U8));
        assert!(!linear.accepts(Dtype::F16) && !nearest.accepts(Dtype::F16));

        let nearest = ResampleOp::new(
            "shrink",
            Resample::new(
                [
                    Ratio::smaller(2).unwrap(),
                    Ratio::identity(),
                    Ratio::identity(),
                ],
                Interpolation::Nearest,
            ),
        );
        let linear = ResampleOp::new(
            "shrink",
            Resample::new(
                [
                    Ratio::smaller(2).unwrap(),
                    Ratio::identity(),
                    Ratio::identity(),
                ],
                Interpolation::Linear,
            ),
        );
        let mask: Voxels = Array3::from_shape_fn((8, 2, 2), |(i, _, _)| i % 3 == 0).into();
        let mut out = Voxels::zeros(Dtype::Bool, [4, 2, 2]).unwrap();
        nearest
            .apply(&mask, &mut out, &Anchor::whole([8, 2, 2]))
            .unwrap();
        let source = mask.view::<bool>().unwrap();
        let got = out.view::<bool>().unwrap();
        for i in 0..4 {
            assert_eq!(got[[i, 0, 0]], source[[2 * i + 1, 0, 0]]);
        }
        // and the refusal, when a plan slipped past `accepts`, names the reason
        let message = linear
            .apply(&mask, &mut out, &Anchor::whole([8, 2, 2]))
            .unwrap_err()
            .to_string();
        assert!(message.contains("not a mask"), "{message}");
    }

    /// The linear blend, against the definition written out.
    #[test]
    fn linear_interpolation_is_the_weighted_mean_of_the_bracketing_voxels() {
        let input = Array3::from_shape_fn((4, 1, 1), |(i, _, _)| i as f64);
        let ratios = ratios([(2, 1), (1, 1), (1, 1)]);
        let mut out = Array3::<f64>::zeros((8, 1, 1));
        resample_linear_into(
            input.view(),
            &Anchor::whole([4, 1, 1]),
            &ratios,
            out.view_mut(),
        )
        .unwrap();
        // x(o) = o/2 - 1/4: clamped at the ends, quarter steps between
        let want = [0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0];
        for (index, value) in want.iter().enumerate() {
            assert_eq!(out[[index, 0, 0]], *value, "output {index}");
        }
    }

    /// An integer blend rounds rather than truncating, which is a decision and
    /// therefore has a test.
    #[test]
    fn an_integer_blend_rounds_to_nearest() {
        let input = Array3::from_shape_fn((2, 1, 1), |(i, _, _)| (i * 10) as u8);
        let ratios = ratios([(4, 1), (1, 1), (1, 1)]);
        let mut out = Array3::<u8>::zeros((8, 1, 1));
        resample_linear_into(
            input.view(),
            &Anchor::whole([2, 1, 1]),
            &ratios,
            out.view_mut(),
        )
        .unwrap();
        // x(o) = o/4 - 3/8, clamped: 0, 0, 0.125, 0.375, 0.625, 0.875, 1, 1
        // times ten and rounded to nearest: 0, 0, 1, 4, 6, 9, 10, 10.
        // Truncating would write 0, 0, 1, 3, 6, 8, 10, 10 — biased low by half
        // a step at every voxel whose coordinate is not exact.
        assert_eq!(
            out.iter().copied().collect::<Vec<u8>>(),
            vec![0, 0, 1, 4, 6, 9, 10, 10]
        );
    }

    /// The declared constant is the computed one, bit for bit, in every element
    /// type and for both interpolations. `constant_maps_to` licenses skipping a
    /// block, so a near miss here would be a volume that differs where a block
    /// happened to be uniform.
    #[test]
    fn a_declared_constant_is_the_computed_one() {
        for value in [0.0f64, 1.0, 7.0, 255.0] {
            for interpolation in [Interpolation::Nearest, Interpolation::Linear] {
                let op = ResampleOp::new(
                    "resample",
                    Resample::uniform(Ratio::new(3, 2).unwrap(), interpolation),
                );
                for dtype in [Dtype::U8, Dtype::U16, Dtype::I32, Dtype::F32, Dtype::F64] {
                    let input = Voxels::filled(dtype, [4, 4, 4], value).unwrap();
                    let mut out = Voxels::zeros(dtype, [6, 6, 6]).unwrap();
                    op.apply(&input, &mut out, &Anchor::whole([4, 4, 4]))
                        .unwrap();
                    let declared = op.constant_maps_to(value).unwrap();
                    let expected = Voxels::filled(dtype, [6, 6, 6], declared).unwrap();
                    assert_eq!(out, expected, "{dtype:?} {interpolation:?} at {value}");
                }
            }
        }
        // the non-representable case is the one that would have made this
        // approximate: an interpolation weight that cannot be exact
        let op = ResampleOp::new(
            "resample",
            Resample::uniform(Ratio::new(3, 1).unwrap(), Interpolation::Linear),
        );
        let input = Voxels::filled(Dtype::F64, [3, 3, 3], 0.1).unwrap();
        let mut out = Voxels::zeros(Dtype::F64, [9, 9, 9]).unwrap();
        op.apply(&input, &mut out, &Anchor::whole([3, 3, 3]))
            .unwrap();
        assert!(
            out.view::<f64>().unwrap().iter().all(|&v| v == 0.1),
            "a + t * (b - a) with a == b must be a exactly, or the declaration is a near miss"
        );
    }

    /// Downsampling point-samples, so it aliases — stated in the module header
    /// as the caller's to fix, and shown here so that "the caller's" is a
    /// decision a reader can see the consequence of.
    #[test]
    fn downsampling_point_samples_and_therefore_aliases() {
        // a pattern at the sampling frequency: alternating 0 and 100
        let stripes = Array3::from_shape_fn((16, 1, 1), |(i, _, _)| (i % 2) as f64 * 100.0);
        let ratios = ratios([(1, 2), (1, 1), (1, 1)]);
        let mut out = Array3::<f64>::zeros((8, 1, 1));
        resample_nearest_into(
            stripes.view(),
            &Anchor::whole([16, 1, 1]),
            &ratios,
            out.view_mut(),
        )
        .unwrap();
        assert!(
            out.iter().all(|&value| value == 100.0),
            "the structure is gone and what is left is one of its phases; a low-pass before this \
             op is what a caller composes to keep it, and that is the whole aliasing decision"
        );
    }

    /// A misaligned fetch is refused rather than resampled half a voxel out of
    /// phase — the guard on a plan somebody built by hand.
    #[test]
    fn a_fetch_that_is_not_on_the_lattice_is_refused_by_name() {
        let resample = Resample::uniform(Ratio::larger(3).unwrap(), Interpolation::Linear);
        let message = resample
            .source_region(&Region::new(&[4, 0, 0], &[6, 4, 4]), [8, 4, 4])
            .unwrap_err()
            .to_string();
        assert!(message.contains("multiples of 3"), "{message}");

        // The same question asked of the kernel, which sees the fetch rather
        // than the read: a buffer whose start is not a multiple of `down` is not
        // the image of any output voxel, so there is no output position to
        // resample it at. (An integer factor has `down == 1` and every source
        // voxel is on the lattice, which is why this is asked at 3/2.)
        let op = ResampleOp::new(
            "grow",
            Resample::uniform(Ratio::new(3, 2).unwrap(), Interpolation::Linear),
        );
        let input = Voxels::zeros(Dtype::F64, [2, 4, 4]).unwrap();
        let mut out = Voxels::zeros(Dtype::F64, [3, 6, 6]).unwrap();
        let message = op
            .apply(&input, &mut out, &Anchor::new([1, 0, 0], [8, 4, 4]))
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("not the image of any output voxel"),
            "{message}"
        );
    }

    /// The halo is per block where the snapping makes it so, and uniform where it
    /// does not — and it is never below the reach, which is what keeps the valid
    /// regions covering their cores.
    #[test]
    fn the_halo_snaps_the_fetch_onto_the_lattice_per_block() {
        let resample = Resample::uniform(Ratio::larger(3).unwrap(), Interpolation::Linear);
        let grid = BlockGrid::along([24, 4, 4], &[0], 4).unwrap();
        let halo = resample.halo(&grid);
        let (lo, hi) = resample.reach(0);
        match halo.axis(0) {
            AxisReach::PerBlock(table) => {
                assert_eq!(table.len(), 6);
                // core 0..4 with reach 1: down to -1 -> the cell boundary below
                // is -3, and up to 5 -> the boundary above is 6
                assert_eq!(table[0], (1 + 2, 1 + 1));
                // and the next block is already on the lattice on both sides,
                // so it pays the reach and nothing more — one number valid for
                // every block would have to be the widest of these
                assert_eq!(table[1], (1 + 0, 1 + 0));
                for entry in table {
                    assert!(entry.0 >= lo && entry.1 >= hi);
                }
            }
            other => panic!("the snapping differs per block here, got {other:?}"),
        }
        // a downsampling factor has nothing to snap to, so the table collapses
        let shrink = Resample::uniform(Ratio::smaller(2).unwrap(), Interpolation::Linear);
        let grid = BlockGrid::along([12, 4, 4], &[0], 5).unwrap();
        assert_eq!(
            *shrink.halo(&grid).axis(0),
            AxisReach::Bounded { lo: 0, hi: 0 }
        );
    }

    /// What the coordinate frame decides, shown rather than argued.
    ///
    /// [`Resample::reach_spec`] states `Frame::Phase`, and the design records
    /// that as the dangerous choice for a phase that reads across grids — a
    /// cropping phase's edges are interior positions of the image below, so a
    /// read clamped there is not an edge and trusting it is a false premise.
    /// Here it is not false, because this phase's output volume edge *is* the
    /// image of the input volume's edge and the last block's fetch runs to it.
    ///
    /// The difference is visible: the same reach in `Frame::Source` denies every
    /// volume-edge block its clamp exception, and those blocks lose part of their
    /// core, and the plan stops tiling. That is the guard change 6 added, doing
    /// exactly what it was built to do — on a phase where the answer happens to
    /// be that the exception is deserved.
    #[test]
    fn the_source_frame_denies_the_clamp_exception_and_this_phase_earns_it() {
        use crate::decomposition::Decomposition;
        use crate::reach::Space;

        let resample = Resample::uniform(Ratio::larger(3).unwrap(), Interpolation::Linear);
        let input_volume = [8usize, 4, 4];
        let output = resample.output_volume(input_volume).unwrap();
        let grid = BlockGrid::along(output, &[0], 6).unwrap();

        let honest = resample_phase(
            vec![0],
            vec!["resample".to_string()],
            &resample,
            input_volume,
            grid.clone(),
        )
        .unwrap();
        assert!(honest.blocks_missing_valid_core().is_empty());

        let source_framed = PhaseDecomposition::derive(
            vec![0],
            vec!["resample".to_string()],
            resample.reach_spec().in_space(Space::source_voxels()),
            resample.halo(&grid),
            grid,
        );
        let short = source_framed.blocks_missing_valid_core();
        assert!(
            !short.is_empty(),
            "the frame has to make a difference here or the choice of frame is not a decision"
        );
        // the same fetch mapping, so that the only difference under test is the
        // frame the reach is stated in
        let sources = source_framed
            .blocks
            .iter()
            .map(|block| resample.source_region(&block.read, input_volume))
            .collect::<Result<Vec<Region>>>()
            .unwrap();
        let source_framed = source_framed.with_sources(|block| sources[block.flat].clone());
        let plan = Decomposition {
            volume: input_volume,
            dtype: Dtype::F64,
            phases: vec![source_framed],
            chain_reach: [1, 1, 1],
        };
        let message = plan.check().unwrap_err().to_string();
        assert!(
            message.contains("do not tile the volume exactly"),
            "{message}"
        );
    }

    /// The fetch-covers-valid check, **seen to fire**. It is the only guard in
    /// this file that no framework check duplicates, so a version of it that
    /// could not fail would be worse than none.
    ///
    /// The provocation is the failure the whole crate is arranged against: a
    /// reach that is too small. The halo, the grid and the fetch mapping are the
    /// real ones; only the declared reach is understated, so every block's valid
    /// region covers its whole core, the regions tile, `Decomposition::check`
    /// is satisfied — and the voxels at the block boundary read source voxels
    /// nobody fetched.
    #[test]
    fn an_understated_reach_is_caught_against_the_fetch_it_would_read_outside() {
        let resample = Resample::uniform(Ratio::larger(3).unwrap(), Interpolation::Linear);
        let input_volume = [8usize, 4, 4];
        let output = resample.output_volume(input_volume).unwrap();
        let grid = BlockGrid::along(output, &[0], 6).unwrap();

        // the honest plan, for comparison: it checks out
        resample_phase(
            vec![0],
            vec!["resample".to_string()],
            &resample,
            input_volume,
            grid.clone(),
        )
        .unwrap();

        // The halo that snaps the fetch onto the lattice and grants nothing
        // else — which is what `Resample::halo` reduces to if the reach it is
        // built from is the zero somebody writes to get past a guard.
        let aligned = Reach::per_axis([
            AxisReach::PerBlock(
                (0..grid.blocks_per_axis()[0])
                    .map(|index| {
                        let core_lo = index * grid.block()[0];
                        let core_hi = (core_lo + grid.block()[0]).min(output[0]);
                        (core_lo % 3, (3 - core_hi % 3) % 3)
                    })
                    .collect(),
            ),
            AxisReach::none(),
            AxisReach::none(),
        ]);
        let dishonest = PhaseDecomposition::derive(
            vec![0],
            vec!["resample".to_string()],
            [0, 0, 0],
            aligned,
            grid,
        );
        // the plan itself is clean: every valid region is its whole core
        assert!(dishonest
            .blocks
            .iter()
            .all(|block| block.valid == block.core));
        let sources = dishonest
            .blocks
            .iter()
            .map(|block| resample.source_region(&block.read, input_volume))
            .collect::<Result<Vec<Region>>>()
            .unwrap();
        let message =
            check_fetch_covers_valid(&resample, &dishonest.blocks, &sources, input_volume)
                .unwrap_err()
                .to_string();
        assert!(
            message.contains("read source") && message.contains("fetch this plan states"),
            "{message}"
        );
    }

    /// Retaking the cost measurement, by the method `super::COST_MEASUREMENT`
    /// describes: best of several repetitions through `BlockOp::apply`, per
    /// **output** voxel, relative to the voxelwise map that is this module's unit
    /// of work.
    ///
    /// ```text
    /// cargo test --release -- --ignored --nocapture ops::resample
    /// ```
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn print_the_cost_table() {
        use super::super::voxelwise::VoxelwiseMapOp;
        use std::time::Instant;

        let shape = [96usize, 64, 64];
        let input: Voxels = ramp(shape).into();
        let anchor = Anchor::whole(shape);
        let cases: Vec<(String, Box<dyn BlockOp>)> = vec![
            (
                "voxelwise map (the unit)".to_string(),
                Box::new(VoxelwiseMapOp::threshold("map", 500.0, 1.0, 0.0)),
            ),
            (
                "resample, nearest, 1/2 on every axis".to_string(),
                Box::new(ResampleOp::new(
                    "nearest",
                    Resample::uniform(Ratio::smaller(2).unwrap(), Interpolation::Nearest),
                )),
            ),
            (
                "resample, nearest, 3/2 on every axis".to_string(),
                Box::new(ResampleOp::new(
                    "nearest",
                    Resample::uniform(Ratio::new(3, 2).unwrap(), Interpolation::Nearest),
                )),
            ),
            (
                "resample, linear, 1/2 on every axis".to_string(),
                Box::new(ResampleOp::new(
                    "linear",
                    Resample::uniform(Ratio::smaller(2).unwrap(), Interpolation::Linear),
                )),
            ),
            (
                "resample, linear, 3/2 on every axis".to_string(),
                Box::new(ResampleOp::new(
                    "linear",
                    Resample::uniform(Ratio::new(3, 2).unwrap(), Interpolation::Linear),
                )),
            ),
        ];
        let mut unit = 1.0;
        println!("op cost, {shape:?}, best of 5, per output voxel");
        for (index, (name, op)) in cases.iter().enumerate() {
            let produced = op.output_shape(shape);
            let outputs = (produced[0] * produced[1] * produced[2]) as f64;
            let mut out = Voxels::zeros(op.produces(input.dtype()), produced).unwrap();
            op.apply(&input, &mut out, &anchor).unwrap();
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let started = Instant::now();
                op.apply(&input, &mut out, &anchor).unwrap();
                let elapsed = started.elapsed().as_secs_f64() * 1e9;
                std::hint::black_box(out.view::<f64>().unwrap().iter().take(1).sum::<f64>());
                best = best.min(elapsed / outputs);
            }
            if index == 0 {
                unit = best;
            }
            println!("{name:<44} {best:>10.3} ns  {:>8.2} relative", best / unit);
        }
    }

    /// What can be asserted about a measured cost without measuring: the
    /// ordering the constants encode is the ordering the ops have. Eight gathers
    /// and seven blends cannot be cheaper than one gather.
    #[test]
    fn the_stored_costs_keep_the_order_the_measurement_found() {
        let nearest = ResampleOp::new(
            "nearest",
            Resample::uniform(Ratio::larger(2).unwrap(), Interpolation::Nearest),
        );
        let linear = ResampleOp::new(
            "linear",
            Resample::uniform(Ratio::larger(2).unwrap(), Interpolation::Linear),
        );
        assert!(linear.cost_per_voxel() > nearest.cost_per_voxel());
        // and both are inside the band the measurement found, so a constant
        // edited without re-running `print_the_cost_table` fails here
        assert!((0.3..1.0).contains(&nearest.cost_per_voxel()));
        assert!((1.4..2.0).contains(&linear.cost_per_voxel()));
    }
}
