// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation,
// not adapted from any implementation of it.
//
// **One row per region of a label volume, reducing a second array over that
// region's voxels.**
//
// What it is
// ----------
// Two arrays of the same extent. The first is a *label volume*: every voxel
// holds a whole number, `0` meaning "no region" and every other value naming
// one. The second is a *value array*, whatever quantity the caller wants
// reduced. The output is a `crate::table` row per label actually present,
// carrying — over exactly the voxels that label occupies —
//
// | column | what it is | how two partials combine |
// |---|---|---|
// | `label` | the label the row is about | it is the key |
// | `count` | voxels carrying it | `+` |
// | `nonfinite` | of those, how many held a non-finite value | `+` |
// | `sum_q{n}` | fixed-point sum of the finite values | `+` |
// | `min` | the smallest finite value, **as it was read** | `min` |
// | `max` | the largest finite value, **as it was read** | `max` |
// | `sum_0..2` | per-axis sum of the voxels' coordinates | `+` |
// | `moment_0..2_q{n}` | fixed-point sum of value times coordinate | `+` |
//
// and positioned at the region's rounded centroid, which is what `sum_0..2` and
// `count` say exactly and the position says to a voxel. Every combine in that
// right-hand column is associative and commutative **in the type it is
// performed in**, and that is the whole design; the rest of this header is why
// it had to be.
//
// The first moment, which is the one column that is not a reading of the others
// ----------------------------------------------------------------------------
// `moment_0..2_q{n}` is `sum_i (q(v_i) * x_i[a])` — the **cross moment** of value
// against position, per axis. It is here because it is the one per-region
// quantity a consumer cannot derive from the columns beside it: `sum_q{n}` and
// `sum_0..2` do not determine it, since two regions with the same voxel count,
// the same total value and the same coordinate totals can hold their value
// differently over their voxels and have different first moments. A caller
// wanting it from the other columns would have to re-walk the volume; here it is
// three more words folded by `+` in an accumulator that already holds both arrays
// and already visits every voxel once.
//
// The **weighted centroid** is its quotient, `moment_a / sum`, and the scale is
// not in it: both are integers at the same `2^n`, so the `2^n` cancels exactly
// and the ratio is a pure number. That is why the moment is quantised on the
// *value* alone and multiplied by the coordinate as the exact integer it already
// is — the coordinate needs no scale, and giving it one would only narrow the
// range. `RegionValues::weighted_centroid` is the quotient, taken once, at the
// end, on two integers that are each already decomposition-invariant.
//
// **It is taken over the finite voxels**, exactly as `sum` is, because it is a
// quotient of two of them and a numerator over one voxel set and a denominator
// over another would not be a centroid of anything. A region's `nonfinite` count
// is therefore as much a caveat on its weighted centroid as on its sum.
//
// **It is not where the row sits.** The row's position stays the *geometric*
// centroid, and that is not an omission: block ownership is decided from the
// position — see [`MergeTabulationOp`] — and a weighted centroid over signed
// values need not be inside the region, inside its bounding box, or inside the
// volume at all. A position that could leave the lattice is not a position.
//
// Why it is not `ops::detect`
// ---------------------------
// `ops::detect` measures *connected components of a mask* and its
// `Emission::Measured` already emits one row per component. A label volume is
// not a component labelling: a region of it may be disconnected, and two regions
// may be face-adjacent. A partition produced by a seeded flooding is the case
// that forces the distinction — its regions are not the connected components of
// the same voxel set and are not in general of the same number — so a count
// taken by connectivity is not a count of them. This op takes the regions as
// given rather than deriving them, and that is the only reason it exists beside
// `detect`.
//
// The seam, which is the whole difficulty
// ---------------------------------------
// A region straddles block seams. Its voxels are visited in pieces, one piece
// per block, and the pieces are combined in a merge. If the sum is accumulated
// in `f64` then that combine does not associate: `(a + b) + c` and `a + (b + c)`
// differ in the last bits, so the total depends on the order the blocks merged
// in and therefore on **how the volume was cut**. The same data planned two ways
// gives two answers and neither looks wrong. `ops/detect.rs` deferred a weighted
// centroid on exactly this and named the two honest ways out: a fixed-point
// accumulator, or a stated tolerance.
//
// `crate::fragment::SeamFold` makes the choice impossible to leave unstated —
// `Unordered` is *checked*, by applying each block a second time with its
// neighbourhood reversed and requiring the same bytes — but it does not make it.
//
// **The sum takes the first way out: a fixed-point accumulator, and therefore
// `SeamFold::Unordered`.** Every value is quantised to an integer *once, at the
// voxel it is read at*, and from there to the end of the run the addition is
// integer addition. Integer addition associates exactly, so the merged total is
// a function of the label's voxel set alone and the answer is byte-identical
// across decompositions. The alternative — an `f64` total and
// `SeamFold::OrderDependent` — was rejected because a quantity that depends on
// the block size is not a measurement, and every consumer of this table filters
// on one.
//
// **The selection needs no way out, and so it does not take one.** `min` and
// `max` are not accumulations: they *choose one of the values they were handed*
// and never compute a new one. In `f64` they are associative, commutative and
// idempotent already, so a region cut across blocks selects the same bits
// whatever order the pieces merged in, and `SeamFold::Unordered` holds for them
// on their own terms rather than on the accumulator's. See "Why the selection
// carries no scale" below, which is where that argument is finished.
//
// What the fixed point costs, which is a real limit and not a footnote
// --------------------------------------------------------------------
// [`FixedPoint`] is a **parameter**, because the trade it makes is the caller's
// and there is no value that is right for every array. With `n` fraction bits:
//
// * **resolution** is `2^-n`. Each value is rounded to the nearest multiple of
//   it *before* being added, so a region of `N` voxels carries a quantisation
//   error of at most `N * 2^-(n+1)`. That error is deterministic and is a
//   function of the data alone — it does not move when the plan does, which is
//   the property being bought;
// * **range** is `+/- 2^(63-n)`, on the *sum* and on every individual value.
//   Beyond it the answer is not representable and the op **refuses, naming the
//   value and the limit**, rather than saturating or wrapping.
//
// At the default of 20 bits that is a resolution of about `9.5e-7` and a range
// of about `+/- 8.8e12`. A caller summing large arrays of large values trades
// bits down; a caller measuring small quantities trades them up. Both numbers
// are `FixedPoint::resolution` and `FixedPoint::limit`, so they can be asserted
// rather than believed.
//
// **The first moment is the column the range binds on first, and by a known
// factor.** `moment_a` is a sum of `q(v) * x[a]`, so it is the sum's own bound
// multiplied by the largest coordinate on that axis — `extent[a] - 1`. Its
// column is the same signed 64-bit word, so the same `+/- 2^(63-n)` applies to
// it, and a run whose *sum* clears the range by less than that factor will be
// refused on a moment rather than on the sum. That is stated rather than
// discovered: the refusal names the moment and says which axis it was.
//
// A caller with no scale in mind can **derive** one rather than pick one: a
// region holds at most every voxel of the volume, each finite value is at most
// the array's own peak magnitude, quantising moves each of them at most half a
// step further from zero, and each contributes to a moment at most its own
// magnitude times the largest coordinate on that axis. So no total can exceed
//
//     voxels * (magnitude + 0.5) * max(1, extent[0] - 1, extent[1] - 1, extent[2] - 1)
//
// — the trailing factor being what the moment adds and `1` being what it is when
// the sum is the only thing bounded. The largest `n` whose `FixedPoint::limit`
// exceeds that bound is the finest scale the range admits, and it is an
// arithmetic bound on the answer rather than a measurement of it — which is what
// makes it a derivation and not a value chosen because it happened to pass.
//
// At the default of twenty bits the moment's `+/- 8.8e12` covers, for instance, a
// region of a million voxels whose values are of magnitude up to 4096 on an axis
// a thousand voxels long only just — `4.1e12` — so a caller with regions and
// extents of that size is one of the callers the parameter exists for, and trades
// bits down. A caller with values of order one on an extent of a few hundred has
// four orders of magnitude of headroom and need not think about it.
//
// The scale is not carried in the blob as data — it is carried in the **column
// names**, `sum_q20` and `moment_0..2_q20`. That is deliberate: two tabulations
// at different scales are not the same schema, and `Table::write` checks the
// schema in the blob against its own, so mixing them is refused rather than
// silently averaged. The moment carries the suffix for the same reason the sum
// does and not by analogy with it: a moment at four bits and a moment at twenty
// are different integers standing for the same quantity.
//
// Why the selection carries no scale
// ----------------------------------
// `min` and `max` are `F64` columns named `min` and `max`, with no suffix,
// because **there is no scale in them to name**. The whole of the paragraph
// above is an argument about an *addition*: `f64` `+` does not associate, so a
// total taken across a seam depends on the cut, and the fixed point is what buys
// the association back. None of it carries to a selection.
//
// A selection returns one of the values it was given. Under a total order it is
// associative, commutative and idempotent in `f64` itself, so folding the same
// set of partials in any order picks out the same voxel and therefore the same
// bits — which is exactly what `SeamFold::Unordered` claims and exactly what the
// executor's reversal check compares. The order used is [`f64::total_cmp`]
// rather than [`f64::min`]: the two differ only on values that compare equal
// without being the same bits — `-0.0` against `0.0` — and there `f64::min` may
// return either operand, which is an order dependence in the one place this op
// cannot have one.
//
// What quantising a selection cost, before this: the column came back as
// `round(v * 2^n) / 2^n` where the question asked for *the voxel's own value*.
// A caller wanting byte identity with a value it can see in the array therefore
// needed an `n` at which every extremal value happened to be exactly
// representable, while still needing an `n` small enough that no total left the
// range — two requirements pulling opposite ways, leaving a window a couple of
// bits wide that closes as soon as a region is wider or a volume brighter. That
// is a scale chosen because it passed, which is not a scale. Now there is
// nothing to choose: the sum has a scale because an addition needs one, and the
// selection has none because a selection does not.
//
// Negatives, `NaN` and infinities
// -------------------------------
// * **Negative values are exact.** The accumulator is signed throughout — `i128`
//   in the fold, an `i64` in **offset binary** in the column, so that the
//   column's unsigned bits still order the way its signed values do — and a
//   value array that straddles zero therefore sums to the same number whatever
//   order the pieces arrive in. Nothing here takes an absolute value or assumes
//   a sign. The moments are the same accumulator with the same guarantee.
// * **A signed value array makes the weighted centroid a ratio and not a point,
//   and this op admits that rather than constraining it.** With every value of
//   one sign the quotient is a convex combination of the region's coordinates
//   and therefore lies inside its bounding box. Let the values straddle zero and
//   it need not: the denominator can be small while the numerator is not, and the
//   answer then sits outside the region, outside the volume, or at any distance
//   at all. That is the *correct* value of `sum(v*x)/sum(v)` and not a defect in
//   computing it, so it is neither clamped nor refused. Refusing would be worse
//   than useless here: the sign of an array is not knowable at construction, a
//   refusal at the voxel would throw away every other region for one negative
//   value — the argument the non-finite rule already makes — and a refusal at the
//   end would throw away the run for one region. A caller who needs the quotient
//   to be a point in the region is asking for a weight and should give the op a
//   non-negative one; the op reports what it was handed.
// * **`NaN` and `+/-inf` are excluded from `sum`, `min` and `max`, and counted
//   in `nonfinite`.** Neither refusing the run nor folding them in is right.
//   Folding them in destroys the row: a single `NaN` anywhere in a region makes
//   its whole sum a `NaN`, and an infinity makes it an infinity, so one bad
//   voxel takes every other voxel's measurement with it — and neither has a
//   fixed-point image to be quantised to in the first place. Refusing the run
//   throws away every other region for the same one voxel. So they are set aside
//   and *reported*: a row with `nonfinite == count` has no finite value at all —
//   its `sum` is the fixed-point zero, its `min` and `max` are `0.0`, and the
//   count is what says why — and a row with `0 < nonfinite < count` is a partial
//   measurement and says so. A `max` that came back `inf` because one voxel was
//   broken is the failure this rule exists to prevent, and it is the reason the
//   selection is filtered rather than merely ordered: `total_cmp` would happily
//   rank an infinity above every real value.
// * A **finite** value too large for the fixed point is a different thing from a
//   non-finite one and is treated differently: it is a real overflow, so it is
//   refused by name. The selection would have carried it — an `f64` column has
//   the whole range — but the sum is taken over the same voxels, so the refusal
//   is the run's and not the column's.
//
// Degenerate cases, stated rather than discovered
// -----------------------------------------------
// * **A label that appears in no voxel** — of a numbering with gaps, or above
//   everything present — gets **no row**. There is no way for this op to know it
//   was expected: it reads voxels, not a caller's label list. A consumer that
//   needs a dense table joins against its own numbering.
// * **A label in exactly one voxel** gets a row with `count == 1`, a position
//   that is that voxel, `min` and `max` that are that voxel's value **bit for
//   bit**, and a `sum` that is its quantisation — which is the same number only
//   when the value was on the fixed point's lattice to begin with.
// * **A region spanning every block** is the ordinary case, not a special one:
//   the merge reaches the whole lattice, so a region touching every block is
//   folded from every block's partial exactly as one touching two is.
// * **Label `0`** is background and is never a row. A negative, fractional or
//   non-finite label is refused by name — a label volume's convention has no
//   negative half, and rounding a fraction would invent a region.
// * **A region whose finite values total exactly zero** has no weighted
//   centroid: `moment_a / sum` is `0/0` when the region held nothing but zeros,
//   and `k/0` when its values cancelled. The *moment columns are still exact and
//   still written* — `sum(v*x)` is defined whatever `sum(v)` is — and it is only
//   the quotient that does not exist, so the absence is reported where the
//   quotient is: [`RegionValues::weighted_centroid`] is an `Option` and is `None`
//   exactly when `sum_fixed == 0`.
//
//   Three things it deliberately is not. Not a `NaN`: `Table::write` refuses a
//   non-finite `F64` column, and a `NaN` a caller has to test for is a value that
//   propagates silently through everything that does not. Not the unweighted
//   centroid: that is a different measurement, and substituting it would make a
//   region with no weight indistinguishable from one whose weight happened to be
//   uniform. And not [`super::local::EmptyPopulation`], which was the obvious
//   vocabulary to reach for and does not fit — its two answers are "ask the
//   statistic", which here is the `0/0` that has no answer, and "take the sample
//   centre's own value", which needs a centre voxel that a region does not have.
//   `Option` is the vocabulary that fits, and it is the one [`Tally::centroid`]
//   and [`Tally::min`] already use for a quantity that is absent rather than
//   zero.
//
// The shape, and why it is two phases
// -----------------------------------
// [`TabulateValuesOp`] is `(volume, volume) -> fragments`: it reads **both**
// arrays as declared source levels and emits one partial per block, over that
// block's core only, so every voxel is counted exactly once whatever halo the
// plan granted. It declares `SeamFold::PerBlock`, which is true of it: a partial
// is a function of its own block.
//
// [`MergeTabulationOp`] is `fragments -> fragments` over the whole lattice. Every
// block folds every partial, so every block holds the whole answer; each then
// emits only the rows **it owns** — the block whose core holds the region's
// centroid — so the union over the lattice is the table exactly once, with
// nothing lost and nothing duplicated. That is `ops::detect`'s ownership rule and
// it is reused rather than restated: the centroid is a function of the merged
// region, the cores tile the volume, so exactly one block owns each row.
//
// Neither phase reads the level it is handed. `reads_pixels()` is `false` on
// both, so a run of this pair moves exactly two arrays and the read counters name
// which two.
//
// What it costs
// -------------
// Phase one is one pass over two arrays and is halo-free. Phase two declares a
// whole-lattice fragment reach, so on `N` blocks it moves `N` partials to each of
// `N` blocks and runs the same fold `N` times — the price of ending a run in a
// fragment phase, the same one `ops::fill` and `ops::detect` pay. A partial is
// sixteen words per label *present in that block*, not per label in the volume.
//
// `SeamFold::Unordered` adds one more application of the merge per block, and —
// because the merge streams rather than gathers — one more pass of fragment
// reads. That is the cost of the claim being checked instead of believed.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::decomposition::Decomposition;
use crate::dtype::Dtype;
use crate::env::BlockBuf;
use crate::error::{Error, Result};
use crate::fragment::{
    fragment_phase, pack_u64, unpack_u64, BlockOutput, BlockView, Coverage, FragmentInput,
    FragmentOp, FragmentOutput, SourceBlocks,
};
use crate::geometry::BlockGrid;
use crate::op::SourceInput;
use crate::region::Region;
use crate::sidecar::{FragmentKey, Lifecycle};
use crate::table::{Column, ColumnType, Row, Schema, Table, Value, POSITION_WORDS};

use super::detect::owner_of;
use super::label::MAX_EXACT_LABEL;

// ------------------------------------------------------------ fixed point --

/// The integer domain a value array is accumulated in, and the one decision
/// that makes this op decomposition-invariant.
///
/// A value `v` is carried as `round(v * 2^n)`, an integer. Integers add
/// associatively, so a region cut across blocks totals to the same number
/// whatever order the pieces are combined in; `f64` does not, and that is the
/// whole reason this type exists rather than an `f64` accumulator.
///
/// **A parameter with no universally right value.** More fraction bits is finer
/// resolution and a narrower range, fewer is the reverse, and which side to be
/// on is a fact about the caller's array. Both ends are readable —
/// [`Self::resolution`] and [`Self::limit`] — so a caller can assert the trade
/// rather than assume it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedPoint {
    fraction_bits: u32,
}

/// Fraction bits [`FixedPoint::default`] uses: a resolution of about `9.5e-7`
/// and a range of about `+/- 8.8e12`.
///
/// A default at all, rather than none, because the *type* is the decision this
/// op needed the caller to make and the exact bit count usually is not — and
/// twenty is the value at which neither end of the trade is obviously wrong for
/// an array of ordinary measurements. A caller for whom it is wrong will find
/// out loudly: the range is a refusal, not a saturation.
pub const DEFAULT_FRACTION_BITS: u32 = 20;

/// The most fraction bits a [`FixedPoint`] admits.
///
/// The column holds a signed 64-bit integer, so `2^(63 - n)` is the range; at
/// 62 bits that is `+/- 2`, and beyond it there is no range left to have.
pub const MAX_FRACTION_BITS: u32 = 62;

impl Default for FixedPoint {
    fn default() -> Self {
        Self {
            fraction_bits: DEFAULT_FRACTION_BITS,
        }
    }
}

impl FixedPoint {
    /// `n` fraction bits, refused above [`MAX_FRACTION_BITS`].
    pub fn bits(fraction_bits: u32) -> Result<Self> {
        if fraction_bits > MAX_FRACTION_BITS {
            return Err(Error::invalid(format!(
                "a fixed-point accumulator of {fraction_bits} fraction bit(s) leaves \
                 2^(63 - {fraction_bits}) of range in a 64-bit column, which is none. At most \
                 {MAX_FRACTION_BITS} are admitted, and that already leaves a range of +/- 2."
            )));
        }
        Ok(Self { fraction_bits })
    }

    pub fn fraction_bits(self) -> u32 {
        self.fraction_bits
    }

    /// `2^n`: how many integer steps one unit of value is worth.
    pub fn scale(self) -> f64 {
        (1u128 << self.fraction_bits) as f64
    }

    /// `2^-n`: the spacing of the values this accumulator can represent, and
    /// therefore the most one voxel's rounding can be out by, doubled.
    pub fn resolution(self) -> f64 {
        1.0 / self.scale()
    }

    /// `2^(63-n)`: the magnitude at which a value, or a sum, stops being
    /// representable and the op refuses.
    pub fn limit(self) -> f64 {
        (1u128 << (63 - self.fraction_bits)) as f64
    }

    /// The suffix the value columns carry, so that the scale travels with the
    /// blob rather than beside it. See the module header.
    pub fn suffix(self) -> String {
        format!("_q{}", self.fraction_bits)
    }

    /// `round(value * 2^n)`, or `Ok(None)` for a value that is not finite.
    ///
    /// Non-finite is `Ok(None)` rather than an error because it is a *stated
    /// outcome* of this op — the voxel is counted in `nonfinite` and left out of
    /// the reductions. A finite value that does not fit is an error, because
    /// that is an overflow rather than a category.
    pub fn quantise(self, value: f64) -> Result<Option<i128>> {
        if !value.is_finite() {
            return Ok(None);
        }
        let scaled = (value * self.scale()).round();
        // Checked against the column's range rather than the accumulator's: a
        // single value the answer could not hold cannot be part of an answer,
        // and refusing at the voxel names the voxel. `as i128` saturates in
        // Rust, so the check has to happen before the cast rather than after.
        // `scaled` is finite here, so the comparison is total.
        if scaled.abs() >= 9_223_372_036_854_775_808.0 {
            return Err(Error::invalid(format!(
                "tabulate: the value {value:e} scales to {scaled:e} at {} fraction bit(s), which \
                 a signed 64-bit fixed-point column cannot hold. This accumulator's range is \
                 +/- {:e}; each fraction bit given up doubles it and halves the resolution, \
                 which is {:e} here.",
                self.fraction_bits,
                self.limit(),
                self.resolution()
            )));
        }
        Ok(Some(scaled as i128))
    }

    /// The value a fixed-point integer stands for. Exact for every integer this
    /// accumulator admits, since both `2^-n` and any `i64` under `2^53 * 2^n`
    /// are exactly representable; above that the division is the only rounding
    /// on the whole path, and it happens once, at the end, on a number that is
    /// already decomposition-invariant.
    pub fn value_of(self, fixed: i64) -> f64 {
        fixed as f64 / self.scale()
    }

    /// The column word for a fixed-point integer: **offset binary**, `fixed +
    /// 2^63`.
    ///
    /// Not two's complement, and the difference matters. `crate::table` orders
    /// rows by their own words and a `U64` column's bits are compared as they
    /// stand; under two's complement every negative sum would sort above every
    /// positive one. Offset binary makes the unsigned bit order the signed value
    /// order, so the canonical row order stays the order a reader would expect.
    pub fn to_column(self, fixed: i128) -> Result<u64> {
        let low = -(1i128 << 63);
        let high = 1i128 << 63;
        if fixed < low || fixed >= high {
            return Err(Error::invalid(format!(
                "tabulate: a fixed-point total of {fixed} at {} fraction bit(s) is {} the range \
                 of a signed 64-bit column, which is +/- {}. The fold itself is exact — it is \
                 `i128` — so this is the answer being too large to report rather than too large \
                 to compute; use fewer fraction bits.",
                self.fraction_bits,
                if fixed < low { "below" } else { "above" },
                self.limit()
            )));
        }
        Ok((fixed + high) as u64)
    }

    /// The other half of [`Self::to_column`].
    pub fn from_column(self, word: u64) -> i64 {
        (word as i128 - (1i128 << 63)) as i64
    }
}

// ----------------------------------------------------------------- schema --

/// Column names that do not depend on the scale. `pub` so a consumer can name a
/// column without spelling the string.
pub const LABEL: &str = "label";
/// Voxels carrying the label.
pub const COUNT: &str = "count";
/// Of those, how many held a value that was not finite.
pub const NONFINITE: &str = "nonfinite";
/// Per-axis sum of the voxels' coordinates: `sum_0`, `sum_1`, `sum_2`.
pub const POSITION_SUM: [&str; 3] = ["sum_0", "sum_1", "sum_2"];
/// Stem of the one column whose name carries the fixed-point scale: the whole
/// name is `sum` followed by [`FixedPoint::suffix`].
pub const SUM: &str = "sum";
/// The smallest finite value in the region, as an `F64` column. A whole name
/// rather than a stem — a selection has no scale to put in it.
pub const MIN: &str = "min";
/// The largest finite value in the region. See [`MIN`].
pub const MAX: &str = "max";
/// Stems of the three first-moment columns: `sum_i (v_i * x_i[a])` per axis, in
/// the same fixed point the sum is in, so the whole names are these followed by
/// [`FixedPoint::suffix`]. A stem rather than a whole name, and for exactly
/// [`SUM`]'s reason: a moment is an accumulation and an accumulation has a scale.
pub const MOMENT: [&str; 3] = ["moment_0", "moment_1", "moment_2"];

/// Payload columns a tabulated row has.
pub const COLUMNS: usize = 12;

/// Words a tabulated row occupies: the three positions and the nine columns.
pub const ROW_WORDS: usize = POSITION_WORDS + COLUMNS;

/// The schema this op writes, at `fixed`.
///
/// **Ten `U64` columns and two `F64` ones, and the split is the whole of what
/// this op decided.** The entry condition `ops::detect::measurement_schema`
/// states — that a column here is a merged accumulator, and an `F64` column
/// merged across a seam is not the same number as the whole fold — is an
/// argument about an *accumulation*, and it is right about every accumulation
/// here: the counts, the coordinate sums, `sum_q{n}` and `moment_0..2_q{n}` are
/// `U64` for exactly that reason, and the four signed ones are offset-binary
/// fixed point on top of it, see [`FixedPoint::to_column`].
///
/// `min` and `max` accumulate nothing. They select one of the values they were
/// handed, which is associative, commutative and idempotent in `f64` under a
/// total order, so a partial merged across a seam **is** the whole fold, bit for
/// bit. Making them `U64` would have bought nothing and cost the answer: the
/// column would report `round(v * 2^n) / 2^n` where the question was which value
/// a voxel held. The module header carries the argument in full.
///
/// The scale is in the four accumulated-value names — `sum_q{n}` and
/// `moment_0..2_q{n}` — so a blob written at one scale cannot be written into a
/// table built at another: `Table::write` compares the blob's schema against its
/// own and refuses. The two selection columns and the three coordinate sums are
/// the same at every scale, because they have none: a selection is a value that
/// was never scaled and a coordinate is an integer that never needed to be.
///
/// **The moments are appended rather than placed beside the sum**, so that every
/// column that existed before this one keeps the index it had. A consumer reading
/// by index is reading the same column, and a consumer reading by name was never
/// affected either way.
pub fn tabulation_schema(fixed: FixedPoint) -> Schema {
    let suffix = fixed.suffix();
    let columns = vec![
        Column::u64(LABEL),
        Column::u64(COUNT),
        Column::u64(NONFINITE),
        Column::u64(format!("{SUM}{suffix}")),
        Column::f64(MIN),
        Column::f64(MAX),
        Column::u64(POSITION_SUM[0]),
        Column::u64(POSITION_SUM[1]),
        Column::u64(POSITION_SUM[2]),
        Column::u64(format!("{}{suffix}", MOMENT[0])),
        Column::u64(format!("{}{suffix}", MOMENT[1])),
        Column::u64(format!("{}{suffix}", MOMENT[2])),
    ];
    // Twelve distinct, non-empty names, so this cannot fail; expressed as a
    // `Result` internally and unwrapped here rather than making every caller
    // handle an impossibility.
    Schema::new(columns).expect("the tabulation schema names twelve distinct columns")
}

// ------------------------------------------------------------------ tally --

/// What one label accumulates, in the types the combines are exact in.
///
/// `i128` for the sum rather than the `i64` the column holds: the fold has no
/// range limit of its own, so a run whose *total* fits reports it even if a
/// partial ordering of the additions would not have. The one stated limit is at
/// the boundary where the answer becomes a column, which is the honest place for
/// it.
#[derive(Debug, Clone, Copy)]
pub struct Tally {
    pub label: u64,
    /// Voxels carrying the label.
    pub count: u64,
    /// Of those, how many held a value that was not finite.
    pub nonfinite: u64,
    /// Fixed-point sum over the finite values.
    pub sum: i128,
    /// The extremes over the finite values, **as they were read** — no fixed
    /// point, because a selection has nothing to quantise — or `None` when there
    /// were no finite values at all.
    ///
    /// Always finite when it is `Some`: the non-finite values never reach here,
    /// which is what stops one broken voxel becoming the whole region's answer.
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// Per-axis sum of the voxels' coordinates, over every voxel of the label.
    pub position: [u64; 3],
    /// Per-axis fixed-point **first moment** of value against position:
    /// `sum_i (q(v_i) * x_i[a])`, over the **finite** voxels only, at the same
    /// scale as [`Self::sum`].
    ///
    /// Only the value is quantised; the coordinate enters as the exact integer it
    /// already is. So the product is exact, the fold is integer `+`, and the
    /// weighted centroid — [`Self::weighted_centroid`] — is `moment[a] / sum`
    /// with the `2^n` cancelling rather than being divided out.
    ///
    /// Over the finite voxels only, because it is the numerator of a quotient
    /// whose denominator is [`Self::sum`]: a numerator taken over one voxel set
    /// and a denominator over another would not be a centroid of either.
    pub moment: [i128; 3],
}

/// A selection's bits, which is what two tallies are compared on.
fn selection_bits(value: Option<f64>) -> Option<u64> {
    value.map(f64::to_bits)
}

/// **Equality on the bits, not on the values**, and the difference is `-0.0`
/// against `0.0`: `f64`'s own `==` calls those two equal and this op promises
/// byte identity, so a comparison that could not tell them apart would be unable
/// to observe the property it is used to assert.
impl PartialEq for Tally {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label
            && self.count == other.count
            && self.nonfinite == other.nonfinite
            && self.sum == other.sum
            && self.position == other.position
            && self.moment == other.moment
            && selection_bits(self.min) == selection_bits(other.min)
            && selection_bits(self.max) == selection_bits(other.max)
    }
}

/// Sound because [`PartialEq`] above is comparison of `u64`s and `i128`s
/// throughout: no `f64` comparison survives into it, so there is no value that
/// is unequal to itself.
impl Eq for Tally {}

/// The smaller of two finite values under [`f64::total_cmp`].
///
/// `total_cmp` rather than [`f64::min`], and the reason is the whole of this
/// op's claim. The two agree except on operands that compare equal without being
/// the same bits — `-0.0` and `0.0` — where `f64::min` may return either one.
/// That is an order dependence, and an order dependence in the seam combine is
/// the thing [`crate::fragment::SeamFold::Unordered`] exists to forbid.
/// `total_cmp` is a total order on the bit patterns, so a tie is only ever
/// between identical bits and the answer is the same whichever way round the two
/// arrive — which makes this a genuine semilattice: associative, commutative and
/// idempotent, in `f64`, with no accumulator underneath it.
fn least(a: f64, b: f64) -> f64 {
    if b.total_cmp(&a).is_lt() {
        b
    } else {
        a
    }
}

/// The larger of two finite values. See [`least`].
fn greatest(a: f64, b: f64) -> f64 {
    if b.total_cmp(&a).is_gt() {
        b
    } else {
        a
    }
}

impl Tally {
    pub fn new(label: u64) -> Self {
        Self {
            label,
            count: 0,
            nonfinite: 0,
            sum: 0,
            min: None,
            max: None,
            position: [0; 3],
            moment: [0; 3],
        }
    }

    /// One voxel at global coordinate `at`, holding `value`, accumulated at
    /// `fixed`.
    ///
    /// **The value and its quantisation arrive together rather than the
    /// quantisation alone**, because the sum and the selection have to be taken
    /// over the *same* voxels: the sum needs `round(value * 2^n)` and the
    /// selection needs `value` itself, and a caller handed only one of them
    /// could supply a pair that disagreed about which voxels were finite. So
    /// this method quantises, which makes "counted in `nonfinite`", "left out of
    /// the sum" and "left out of the selection" one decision taken in one place.
    ///
    /// Refuses, naming the voxel's value, when it is finite and outside the
    /// fixed point's range: that is a real overflow of the sum. See
    /// [`FixedPoint::quantise`].
    pub fn add(&mut self, at: [usize; 3], value: f64, fixed: FixedPoint) -> Result<()> {
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| overflowed("count"))?;
        for (axis, sum) in self.position.iter_mut().enumerate() {
            *sum = sum
                .checked_add(at[axis] as u64)
                .ok_or_else(|| overflowed("a coordinate sum"))?;
        }
        match fixed.quantise(value)? {
            None => {
                self.nonfinite = self
                    .nonfinite
                    .checked_add(1)
                    .ok_or_else(|| overflowed("the non-finite count"))?
            }
            Some(quantised) => {
                self.sum = self
                    .sum
                    .checked_add(quantised)
                    .ok_or_else(|| overflowed("the fixed-point sum"))?;
                // The first moment, on the same voxel and the same quantisation
                // as the sum — one `quantise` call, so the two cannot come to
                // disagree about which voxels were finite. The coordinate is not
                // quantised: it is already an integer, and scaling it would only
                // spend range.
                for (axis, moment) in self.moment.iter_mut().enumerate() {
                    let term = quantised
                        .checked_mul(at[axis] as i128)
                        .ok_or_else(|| overflowed("a fixed-point first moment"))?;
                    *moment = moment
                        .checked_add(term)
                        .ok_or_else(|| overflowed("a fixed-point first moment"))?;
                }
                self.min = Some(self.min.map_or(value, |seen| least(seen, value)));
                self.max = Some(self.max.map_or(value, |seen| greatest(seen, value)));
            }
        }
        Ok(())
    }

    /// Fold another partial for the same label in.
    ///
    /// **This is the seam combine**, and every operation in it is associative
    /// and commutative in its own type: integer `+` for the counts and the
    /// fixed-point sum, and a total-order [`least`]/[`greatest`] in `f64` for
    /// the selection. That is what [`crate::fragment::SeamFold::Unordered`]
    /// claims and what the executor checks by applying the block again with its
    /// neighbourhood reversed.
    pub fn merge(&mut self, other: &Tally) -> Result<()> {
        if self.label != other.label {
            return Err(Error::invalid(format!(
                "tabulate: partials for labels {} and {} were merged. A row is keyed by its \
                 label, so folding two of them would report one region's voxels under another's \
                 name.",
                self.label, other.label
            )));
        }
        self.count = self
            .count
            .checked_add(other.count)
            .ok_or_else(|| overflowed("count"))?;
        self.nonfinite = self
            .nonfinite
            .checked_add(other.nonfinite)
            .ok_or_else(|| overflowed("the non-finite count"))?;
        self.sum = self
            .sum
            .checked_add(other.sum)
            .ok_or_else(|| overflowed("the fixed-point sum"))?;
        self.min = match (self.min, other.min) {
            (Some(mine), Some(theirs)) => Some(least(mine, theirs)),
            (mine, theirs) => mine.or(theirs),
        };
        self.max = match (self.max, other.max) {
            (Some(mine), Some(theirs)) => Some(greatest(mine, theirs)),
            (mine, theirs) => mine.or(theirs),
        };
        for axis in 0..3 {
            self.position[axis] = self.position[axis]
                .checked_add(other.position[axis])
                .ok_or_else(|| overflowed("a coordinate sum"))?;
            self.moment[axis] = self.moment[axis]
                .checked_add(other.moment[axis])
                .ok_or_else(|| overflowed("a fixed-point first moment"))?;
        }
        Ok(())
    }

    /// The rounded centroid, which is where the row sits, or `None` for a tally
    /// with no voxels.
    ///
    /// The same half-up rounding `ops::detect::Moments::centroid` uses, so a row
    /// this op writes for a region lands on the same voxel a row `detect` writes
    /// for the same voxel set does.
    pub fn centroid(&self) -> Option<[usize; 3]> {
        if self.count == 0 {
            return None;
        }
        let count = self.count as u128;
        let mut at = [0usize; 3];
        for (axis, coordinate) in at.iter_mut().enumerate() {
            // Bounded by the largest coordinate that was added, which came from
            // a `usize`, so this cannot truncate.
            *coordinate = ((2 * self.position[axis] as u128 + count) / (2 * count)) as usize;
        }
        Some(at)
    }

    /// `moment[a] / sum` per axis — the **weighted centroid** — or `None` when
    /// the finite values totalled exactly zero and the quotient does not exist.
    ///
    /// `None` rather than a `NaN` and rather than [`Self::centroid`]. A `NaN`
    /// propagates silently through everything that does not test for it, and
    /// `Table::write` will not carry one in an `F64` column anyway; the
    /// unweighted centroid is a *different measurement*, and substituting it
    /// would make a region with no weight indistinguishable from one whose
    /// weight was uniform. `None` is the same vocabulary [`Self::min`] uses for
    /// the region that held no finite value at all, and for the same reason: the
    /// quantity is absent, not zero.
    ///
    /// **The scale is not in this.** Numerator and denominator are integers at
    /// the same `2^n`, so it cancels exactly and the quotient is a pure ratio of
    /// two decomposition-invariant integers. The `as f64` on each is the only
    /// rounding on the whole path and it happens once, at the end, on numbers
    /// that are already the same under every cut — so the quotient is too.
    ///
    /// Over the **finite** voxels only, both halves. And not necessarily inside
    /// the region: see the module header on signed values.
    pub fn weighted_centroid(&self) -> Option<[f64; 3]> {
        if self.sum == 0 {
            return None;
        }
        let mass = self.sum as f64;
        let mut at = [0.0f64; 3];
        for (axis, coordinate) in at.iter_mut().enumerate() {
            *coordinate = self.moment[axis] as f64 / mass;
        }
        Some(at)
    }

    /// The row's fifteen words: the position, then the payload in schema order.
    ///
    /// The words rather than a struct, because **this array is the canonical
    /// sort key** — `crate::table` orders rows by their own words, so sorting
    /// these is the canonical order by construction rather than by a comparator
    /// that has to be kept in step with the schema.
    ///
    /// A tally whose values were all non-finite has no extremes; both report
    /// `0.0`, and `nonfinite == count` is what says so. `0.0` rather than a
    /// `NaN` because `Table::write` refuses a non-finite `F64` column — the
    /// canonical order tiebreaks on those bits — so the absence has to be
    /// reported by the count that already reports it.
    pub fn row_words(&self, fixed: FixedPoint) -> Result<Option<[u64; ROW_WORDS]>> {
        let Some(at) = self.centroid() else {
            return Ok(None);
        };
        let mut words = [0u64; ROW_WORDS];
        for axis in 0..3 {
            words[axis] = at[axis] as u64;
        }
        words[POSITION_WORDS] = self.label;
        words[POSITION_WORDS + 1] = self.count;
        words[POSITION_WORDS + 2] = self.nonfinite;
        words[POSITION_WORDS + 3] = fixed.to_column(self.sum)?;
        words[POSITION_WORDS + 4] = self.min.unwrap_or(0.0).to_bits();
        words[POSITION_WORDS + 5] = self.max.unwrap_or(0.0).to_bits();
        for axis in 0..3 {
            words[POSITION_WORDS + 6 + axis] = self.position[axis];
            // The moment narrows to a column here and nowhere earlier, exactly as
            // the sum does: the fold has no range limit of its own, so a run
            // whose merged moment fits reports it even where a partial ordering
            // of the products would not have. The refusal names the axis.
            words[POSITION_WORDS + 9 + axis] =
                fixed.to_column(self.moment[axis]).map_err(|failed| {
                    Error::invalid(format!(
                        "{failed} This is the first moment on axis {axis} — `sum(value * \
                         coordinate)` — whose range is the sum's divided by the largest \
                         coordinate on that axis, so it is the column that binds first."
                    ))
                })?;
        }
        Ok(Some(words))
    }
}

fn overflowed(what: &str) -> Error {
    Error::invalid(format!(
        "tabulate: {what} overflowed its accumulator. Every accumulator here is checked rather \
         than wrapping, because a wrapped total is a plausible number and therefore the \
         expensive kind of wrong."
    ))
}

// -------------------------------------------------------- the wire format --

/// Words one label occupies in a **partial**: the label, the two counts, the
/// `i128` sum as a word pair, the two selections as one word each, the three
/// coordinate sums, and the three `i128` first moments as a word pair each.
///
/// The sum and the moments are wider here than in a row on purpose. A partial is
/// folded, not read, so it carries the accumulator's own type — `i128`, which has
/// no range limit worth stating — and the narrowing to a column happens once, at
/// the end, where the limit is stated. A partial that narrowed early would refuse
/// a run whose answer fits, and that matters most for the moments: a moment can
/// be large in one block and cancel in the merge. The selections are one word
/// because they are already the type they end in: nothing widens an `f64` that is
/// only ever compared.
///
/// The moments are appended rather than placed beside the sum, so the entry's
/// leading words are the ones they always were.
const PARTIAL_WORDS: usize = 16;

fn put_i128(words: &mut Vec<u64>, value: i128) {
    let bits = value as u128;
    words.push(bits as u64);
    words.push((bits >> 64) as u64);
}

fn get_i128(low: u64, high: u64) -> i128 {
    (((high as u128) << 64) | low as u128) as i128
}

/// The wire word for "no finite value was seen", which is a `NaN`.
///
/// Unambiguous rather than merely improbable: a selection only ever holds a
/// value that was finite when it was read, so no `NaN` can be a real one and the
/// absence needs no flag word beside it. Written as its bits rather than as
/// `f64::NAN.to_bits()` so that the wire form is a stated constant instead of
/// whichever quiet `NaN` the platform happens to spell.
const ABSENT: u64 = 0x7ff8_0000_0000_0000;

fn put_selection(words: &mut Vec<u64>, value: Option<f64>) {
    words.push(value.map_or(ABSENT, f64::to_bits));
}

/// The other half of [`put_selection`]. Reads `is_finite` rather than comparing
/// against [`ABSENT`], so that any `NaN` and any infinity decodes as absent —
/// neither is a value this op can have selected, so neither can be let through
/// as one.
fn get_selection(word: u64) -> Option<f64> {
    let value = f64::from_bits(word);
    value.is_finite().then_some(value)
}

/// Tallies as a fragment, ascending by label.
pub fn encode_partial(tallies: &BTreeMap<u64, Tally>) -> Vec<u8> {
    let mut words = Vec::with_capacity(tallies.len() * PARTIAL_WORDS);
    for tally in tallies.values() {
        words.push(tally.label);
        words.push(tally.count);
        words.push(tally.nonfinite);
        put_i128(&mut words, tally.sum);
        put_selection(&mut words, tally.min);
        put_selection(&mut words, tally.max);
        for axis in 0..3 {
            words.push(tally.position[axis]);
        }
        for axis in 0..3 {
            put_i128(&mut words, tally.moment[axis]);
        }
    }
    pack_u64(&words)
}

/// The other half of [`encode_partial`]. A length that is not a whole number of
/// entries is a truncated fragment and says so.
pub fn decode_partial(bytes: &[u8]) -> Result<Vec<Tally>> {
    let words = unpack_u64(bytes)?;
    if words.len() % PARTIAL_WORDS != 0 {
        return Err(Error::invalid(format!(
            "tabulate: a partial is a whole number of {PARTIAL_WORDS}-word entries; this one is \
             {} word(s)",
            words.len()
        )));
    }
    let mut found = Vec::with_capacity(words.len() / PARTIAL_WORDS);
    for entry in words.chunks_exact(PARTIAL_WORDS) {
        found.push(Tally {
            label: entry[0],
            count: entry[1],
            nonfinite: entry[2],
            sum: get_i128(entry[3], entry[4]),
            min: get_selection(entry[5]),
            max: get_selection(entry[6]),
            position: [entry[7], entry[8], entry[9]],
            moment: [
                get_i128(entry[10], entry[11]),
                get_i128(entry[12], entry[13]),
                get_i128(entry[14], entry[15]),
            ],
        });
    }
    Ok(found)
}

// ------------------------------------------------------------- phase one --

/// `(label volume, value array) -> fragments`. One partial per block.
///
/// Reads **both** arrays as declared source levels and declares
/// `reads_pixels() == false`, so the level the phase is handed is not read at
/// all: a run of this phase moves exactly two arrays and the read counters name
/// which two. Which array is which is the caller's statement rather than a
/// position in a chain, because the two are not interchangeable and a plan that
/// swapped them would produce a complete, well-formed, entirely wrong table.
///
/// Only the block's **core** is visited, so every voxel is counted exactly once
/// however wide a halo the plan granted. The reach is zero on both operands, so
/// `fragment_phase` grants no halo at all and the core is the whole read extent
/// in the ordinary case; the filter is there for the plan that arrived from
/// somewhere else.
///
/// [`crate::fragment::SeamFold::PerBlock`], which is true of it and is checked
/// against its fragment reach: a partial is a function of its own block, and the
/// fold across the seam belongs to [`MergeTabulationOp`].
pub struct TabulateValuesOp {
    name: &'static str,
    labels: usize,
    values: usize,
    fixed: FixedPoint,
    stream: String,
    lifecycle: Lifecycle,
}

impl TabulateValuesOp {
    /// The two levels, which must be different ones.
    ///
    /// Reducing an array over its own regions is refused rather than computed:
    /// the answer is `label * count` and the caller meant something else. It is
    /// also unplannable — `fragment_phase` records one source level per
    /// declaration and `check_phase_work` compares that record against the
    /// deduplicated list — so the refusal here is the readable version of an
    /// error that would otherwise arrive from the plan checker.
    pub fn new(
        name: &'static str,
        labels: usize,
        values: usize,
        fixed: FixedPoint,
        stream: impl Into<String>,
        lifecycle: Lifecycle,
    ) -> Result<Self> {
        if labels == values {
            return Err(Error::invalid(format!(
                "tabulate: the label volume and the value array are both level {labels}. \
                 Reducing an array over the regions of itself gives `label * count` and nothing \
                 else; the second array is the point of this op."
            )));
        }
        Ok(Self {
            name,
            labels,
            values,
            fixed,
            stream: stream.into(),
            lifecycle,
        })
    }

    pub fn fixed(&self) -> FixedPoint {
        self.fixed
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// The tallies of one block, from the two arrays over `read` and the core
    /// they are restricted to.
    ///
    /// A free function in all but name, and separated from the `FragmentOp`
    /// shell for this module's stated reason: the arithmetic is written over the
    /// narrowest thing it needs — two views and two regions — so that it can be
    /// driven from a test without a plan, an environment or a lattice.
    pub fn tally_block(
        &self,
        labels: &BlockBuf,
        values: &BlockBuf,
        read: &Region,
        core: &Region,
    ) -> Result<BTreeMap<u64, Tally>> {
        let mut tallies: BTreeMap<u64, Tally> = BTreeMap::new();
        // A simulated run holds no arrays, so there is nothing to say about
        // which labels are present — not even how many. An empty fragment is
        // *present and empty*, which is a different fact from absent and is the
        // one the coverage guard checks; a count invented here would be a
        // measurement of the lattice rather than of the data.
        let (BlockBuf::Array(labels), BlockBuf::Array(values)) = (labels, values) else {
            return Ok(tallies);
        };
        let shape = [read.shape[0], read.shape[1], read.shape[2]];
        for (what, array) in [("label volume", labels), ("value array", values)] {
            if array.shape() != shape {
                return Err(Error::invalid(format!(
                    "tabulate: the {what} arrived as {:?} for a block read extent of {shape:?}. \
                     Both operands are fetched at the block's own fetch region, so a disagreement \
                     here is the plan handing over two different geometries.",
                    array.shape()
                )));
            }
        }
        let labels = labels.widened();
        let values = values.widened();
        let offset = [read.start[0], read.start[1], read.start[2]];
        for (index, raw) in labels.indexed_iter() {
            let at = [
                offset[0] + index.0,
                offset[1] + index.1,
                offset[2] + index.2,
            ];
            if !holds(core, at) {
                continue;
            }
            let label = label_at(*raw, at)?;
            if label == 0 {
                continue;
            }
            let value = values[[index.0, index.1, index.2]];
            tallies
                .entry(label)
                .or_insert_with(|| Tally::new(label))
                .add(at, value, self.fixed)?;
        }
        Ok(tallies)
    }
}

/// The label a voxel carries, refusing everything that is not one.
///
/// Four refusals, and each is a value that would otherwise become a region the
/// caller did not name: non-finite, negative — a label volume's convention has
/// no negative half — fractional, since rounding would merge two labels a
/// fraction apart, and above [`MAX_EXACT_LABEL`], beyond which an `f64` no
/// longer names every integer and two distinct labels can arrive as one.
fn label_at(raw: f64, at: [usize; 3]) -> Result<u64> {
    if !raw.is_finite() || raw < 0.0 || raw.fract() != 0.0 || raw > MAX_EXACT_LABEL as f64 {
        return Err(Error::invalid(format!(
            "tabulate: the label volume holds {raw} at {at:?}, which is not a label. A label is \
             a whole number from 0 to {MAX_EXACT_LABEL}; 0 means no region and there is no \
             negative half. A fractional or non-finite entry is a level that was written by \
             something other than a labelling, and rounding it would invent a region."
        )));
    }
    Ok(raw as u64)
}

/// Whether `at` is inside `region`.
fn holds(region: &Region, at: [usize; 3]) -> bool {
    (0..3).all(|axis| {
        at[axis] >= region.start[axis] && at[axis] < region.start[axis] + region.shape[axis]
    })
}

impl FragmentOp for TabulateValuesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![
            SourceInput::voxelwise(self.labels),
            SourceInput::voxelwise(self.values),
        ]
    }

    fn seam_fold(&self) -> Option<crate::fragment::SeamFold> {
        Some(crate::fragment::SeamFold::PerBlock)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "tabulate: a per-region reduction reads a label volume and a value array and cannot \
             be computed without them. It is applied through `apply_with`.",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let tallies = self.tally_block(
            sources.get(self.labels)?,
            sources.get(self.values)?,
            at.read,
            at.core,
        )?;
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            encode_partial(&tallies),
        ))
    }
}

// ------------------------------------------------------------- phase two --

/// `fragments -> fragments`. Every block's partials, folded into the whole
/// answer, and emitted as the rows this block owns.
///
/// Declares a whole-lattice reach and `gathers() == false`, so the partials are
/// streamed one at a time rather than all made resident: the reads are the same
/// and the residency is one fragment plus the accumulator.
///
/// [`crate::fragment::SeamFold::Unordered`], and it is the honest claim rather
/// than the convenient one — every combine in [`Tally::merge`] is integer `+`,
/// or a selection under a total order in `f64`, which is a semilattice and
/// therefore order-independent on its own account rather than on the integers'.
/// The executor checks it by applying each block a second time with the
/// neighbourhood reversed and requiring byte-identical output; an `f64`
/// *accumulator* would fail that on the first block with three partials, which
/// is the hazard the variant exists to catch and which is why the sum is the one
/// column that is quantised.
///
/// **Each block emits only the rows it owns** — the block whose core holds the
/// region's centroid — which is `ops::detect`'s ownership rule and is exact for
/// the same reason: the centroid is a function of the merged region, the cores
/// tile the volume with no overlap and no gap, so exactly one block owns each
/// row. Without it every block would write the whole table and a `Table` that
/// took them all would hold every row once per block; `Table::write` has no
/// ownership rule of its own and would be right not to.
pub struct MergeTabulationOp {
    name: &'static str,
    input: String,
    input_phase: usize,
    lattice: [usize; 3],
    fixed: FixedPoint,
    stream: String,
    lifecycle: Lifecycle,
}

impl MergeTabulationOp {
    /// `lattice` is the blocks-per-axis of the phase this runs on, which is what
    /// makes the reach the whole lattice; see `BlockGrid::blocks_per_axis`.
    pub fn new(
        name: &'static str,
        input: impl Into<String>,
        input_phase: usize,
        lattice: [usize; 3],
        fixed: FixedPoint,
        stream: impl Into<String>,
        lifecycle: Lifecycle,
    ) -> Self {
        Self {
            name,
            input: input.into(),
            input_phase,
            lattice,
            fixed,
            stream: stream.into(),
            lifecycle,
        }
    }

    pub fn fixed(&self) -> FixedPoint {
        self.fixed
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// The schema this op's blobs carry.
    pub fn schema(&self) -> Schema {
        tabulation_schema(self.fixed)
    }

    /// Every partial folded into one map, keyed by label.
    ///
    /// The fold, as a free function over the partials themselves, so that
    /// "combining these in any order gives this map" is assertable without a
    /// run. Order-independent by construction: a `BTreeMap` keyed by label and
    /// [`Tally::merge`] underneath it.
    pub fn fold<'a>(&self, partials: impl IntoIterator<Item = &'a [u8]>) -> Result<Vec<Tally>> {
        let mut totals: BTreeMap<u64, Tally> = BTreeMap::new();
        for bytes in partials {
            absorb(&mut totals, bytes)?;
        }
        Ok(totals.into_values().collect())
    }

    /// The tallies `block` owns, as the blob it writes.
    ///
    /// Ownership first, encoding second: which block writes a row is a property
    /// of the region, so it must not be able to depend on the form it is written
    /// in.
    pub fn encode_owned(
        &self,
        totals: &[Tally],
        grid: &BlockGrid,
        block: [usize; 3],
    ) -> Result<Vec<u8>> {
        let mut rows: Vec<[u64; ROW_WORDS]> = Vec::new();
        for tally in totals {
            let Some(words) = tally.row_words(self.fixed)? else {
                continue;
            };
            let at = [words[0] as usize, words[1] as usize, words[2] as usize];
            if owner_of(grid, at) != block {
                continue;
            }
            rows.push(words);
        }
        rows.sort_unstable();

        let schema = Arc::new(self.schema());
        let mut builder = crate::table::RowBuilder::new(schema.clone());
        for row in &rows {
            let at = [row[0] as usize, row[1] as usize, row[2] as usize];
            // Tagged from the schema rather than from a fixed list, so the two
            // `F64` columns are float-typed here because the schema says they
            // are — a column whose type moved and whose word did not is then a
            // refusal at the push rather than a `u64` that decodes as a
            // plausible float.
            let values: Vec<Value> = schema
                .columns()
                .iter()
                .zip(&row[POSITION_WORDS..])
                .map(|(column, word)| match column.kind() {
                    ColumnType::U64 => Value::U64(*word),
                    ColumnType::F64 => Value::F64(f64::from_bits(*word)),
                })
                .collect();
            // Round-tripped through the typed push rather than written straight
            // into the buffer, so a schema that grew a column without
            // `row_words` growing a word is refused here instead of producing
            // rows that decode as something plausible.
            builder.push(at, &values)?;
        }
        Ok(builder.encode())
    }
}

impl FragmentOp for MergeTabulationOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.input.clone(), self.input_phase).with_reach(self.lattice)]
    }

    /// One fragment resident at a time; see the type's documentation.
    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<crate::fragment::SeamFold> {
        Some(crate::fragment::SeamFold::Unordered)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mut totals: BTreeMap<u64, Tally> = BTreeMap::new();
        at.stream_fragments(&self.input, &mut |_: &FragmentKey, bytes: &[u8]| {
            absorb(&mut totals, bytes)
        })?;
        let totals: Vec<Tally> = totals.into_values().collect();
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            self.encode_owned(&totals, at.grid, at.index)?,
        ))
    }
}

/// One partial's tallies folded into `totals`.
///
/// The whole of the seam combine's plumbing, in one place, so that the streamed
/// path in [`MergeTabulationOp::apply`] and the free-function [`MergeTabulationOp::fold`]
/// cannot drift into folding differently — which would be two answers to the
/// question this op exists to have one answer to.
fn absorb(totals: &mut BTreeMap<u64, Tally>, bytes: &[u8]) -> Result<()> {
    for tally in decode_partial(bytes)? {
        match totals.get_mut(&tally.label) {
            Some(seen) => seen.merge(&tally)?,
            None => {
                totals.insert(tally.label, tally);
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------- the phases --

/// The two phases, on one lattice, appended to a plan that already has some.
///
/// Both are built with `fragment_phase`, so both halos come from the ops'
/// declarations rather than from this function: zero for the tabulation, whose
/// operands are voxelwise, and the whole lattice for the merge, whose reach is
/// the dependency edge that makes every block's partial available to every other
/// one.
///
/// Neither phase declares an element type — neither writes a level — so
/// `check_dtypes` skips both.
///
/// Returns the plan and the phase index the **rows** are keyed under, which is
/// where a reader has to look for them: a stream written by two phases holds two
/// generations, so the phase is half the address.
pub fn append_tabulate_phases(
    mut plan: Decomposition,
    tabulate: &TabulateValuesOp,
    merge: &MergeTabulationOp,
) -> Result<(Decomposition, usize)> {
    let grid = plan
        .phases
        .last()
        .ok_or_else(|| {
            Error::invalid(
                "tabulate: the phases are appended to a plan that already has one, because the \
                 lattice is inherited — fragments are keyed by block index, so a phase reading \
                 another's fragments on a different lattice would address blocks that \
                 correspond to nothing.",
            )
        })?
        .grid
        .clone();
    plan.phases.push(fragment_phase(tabulate, grid.clone())?);
    plan.phases.push(fragment_phase(merge, grid)?);
    let rows_phase = plan.phases.len() - 1;
    plan.check()?;
    Ok((plan, rows_phase))
}

/// A plan that is these two phases and nothing else, over `volume`.
///
/// For a caller whose label volume and value array are already levels: level 0
/// is whichever of them the environment was built from, and the levels the ops
/// name have to exist. `dtype` is level 0's.
pub fn tabulate_phases(
    grid: BlockGrid,
    dtype: Dtype,
    tabulate: &TabulateValuesOp,
    merge: &MergeTabulationOp,
) -> Result<Decomposition> {
    let volume = grid.volume();
    let plan = Decomposition {
        volume,
        dtype,
        phases: vec![
            fragment_phase(tabulate, grid.clone())?,
            fragment_phase(merge, grid)?,
        ],
        chain_reach: [0, 0, 0],
    };
    plan.check()?;
    Ok(plan)
}

// -------------------------------------------------------------- the merge --

/// One region's row, decoded.
///
/// **Both forms of the sum are carried, and only one form of the selection.**
/// The sum's `f64` is what a caller filters on and its fixed-point integer is
/// the thing the invariance claim is about, so offering only the float would
/// hand back a number whose exactness a reader would have to take on trust.
/// `min` and `max` have no second form to offer: they are the voxel's own value,
/// which is already the number and already invariant, and an integer beside them
/// would be a quantisation of an answer that did not need one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionValues {
    pub label: u64,
    /// The rounded centroid, which is where the row sits in the table.
    pub at: [usize; 3],
    pub count: u64,
    /// Voxels of this region whose value was `NaN` or an infinity. They are in
    /// `count` and in `centroid`, and in none of `sum`, `min` or `max`.
    pub nonfinite: u64,
    pub sum_fixed: i64,
    pub sum: f64,
    /// The smallest finite value the region held, **bit for bit as the array
    /// holds it**, or `0.0` when it held none — which [`Self::all_nonfinite`]
    /// is how to tell apart from a region whose smallest value really is zero.
    pub min: f64,
    /// The largest, on the same terms as [`Self::min`].
    pub max: f64,
    /// The sub-voxel centre, which `at` rounds. Exact from the coordinate sums.
    pub centroid: [f64; 3],
    /// The per-axis **first moment** `sum_i (v_i * x_i[a])` over the finite
    /// voxels, in the fixed point's own integers — which is the form the
    /// decomposition-invariance claim is about, so it is offered beside the
    /// `f64` rather than behind it. See [`Self::sum_fixed`].
    pub moment_fixed: [i64; 3],
    /// The same, in the value array's units.
    pub moment: [f64; 3],
    /// **The weighted centroid**, `moment[a] / sum` — the one quantity in this
    /// row that no arrangement of the others determines.
    ///
    /// `None` exactly when `sum_fixed == 0`, which is the region whose finite
    /// values totalled zero: the quotient does not exist and this says so rather
    /// than reporting a `NaN` or quietly substituting [`Self::centroid`]. The
    /// moments themselves are exact and present in either case.
    ///
    /// Over the finite voxels, so a region with a non-zero `nonfinite` has a
    /// weighted centroid of the voxels that had a value. And not necessarily
    /// inside the region: with values of both signs the denominator can be small
    /// where the numerator is not, and the ratio is then outside the bounding box
    /// or outside the volume. That is what `sum(v*x)/sum(v)` is, and this op
    /// reports it rather than constraining it.
    pub weighted_centroid: Option<[f64; 3]>,
}

impl RegionValues {
    /// Whether every voxel of this region held a non-finite value, in which case
    /// `sum`, `min`, `max` and the moments are all zero because there was
    /// nothing to reduce — and [`Self::weighted_centroid`] is `None`, since a
    /// zero sum is a zero denominator.
    pub fn all_nonfinite(&self) -> bool {
        self.nonfinite == self.count
    }
}

/// Decode one row of [`tabulation_schema`].
pub fn region_values(row: &Row<'_>, fixed: FixedPoint) -> Result<RegionValues> {
    let at = row.at();
    let count = row.u64(1)?;
    let sum_fixed = fixed.from_column(row.u64(3)?);
    let mut centroid = [0.0f64; 3];
    for (axis, coordinate) in centroid.iter_mut().enumerate() {
        *coordinate = if count == 0 {
            0.0
        } else {
            row.u64(6 + axis)? as f64 / count as f64
        };
    }
    let mut moment_fixed = [0i64; 3];
    let mut moment = [0.0f64; 3];
    for axis in 0..3 {
        moment_fixed[axis] = fixed.from_column(row.u64(9 + axis)?);
        moment[axis] = fixed.value_of(moment_fixed[axis]);
    }
    // The quotient of the two integers, not of the two `f64`s: the scale is the
    // same in both, so it cancels rather than being divided out twice, and the
    // rounding is the one `as f64` on each side. `Tally::weighted_centroid` is
    // the same arithmetic on the same numbers before they narrowed, and the
    // narrowing is lossless — `to_column` refuses anything it would not be — so
    // the two agree bit for bit.
    let weighted_centroid = (sum_fixed != 0).then(|| {
        let mass = sum_fixed as f64;
        [
            moment_fixed[0] as f64 / mass,
            moment_fixed[1] as f64 / mass,
            moment_fixed[2] as f64 / mass,
        ]
    });
    Ok(RegionValues {
        label: row.u64(0)?,
        at,
        count,
        nonfinite: row.u64(2)?,
        sum_fixed,
        sum: fixed.value_of(sum_fixed),
        moment_fixed,
        moment,
        weighted_centroid,
        // Read as floats, so a schema whose selection columns were integers
        // again is a refusal here naming the column rather than a `u64` reported
        // as an enormous float.
        min: row.f64(4)?,
        max: row.f64(5)?,
        centroid,
    })
}

/// Every block's row blob, in the canonical order, as one list.
///
/// **This is where the order is restored, and it is restored from the rows
/// rather than from anything about the run.** The blobs go into one
/// [`Table`] over `volume`, which holds its rows in the canonical order —
/// lexicographic on the position and then the payload — so the result is a
/// function of the row set alone: not of the lattice, not of which block
/// finished first, not of the order the blobs are handed over.
///
/// The block index is carried only so that a refusal can name the blob it came
/// from; `Table::write` keeps no trace of it.
pub fn merge_tabulation<'a>(
    volume: [usize; 3],
    fixed: FixedPoint,
    blobs: impl IntoIterator<Item = ([usize; 3], &'a [u8])>,
) -> Result<Vec<RegionValues>> {
    let mut table = Table::new(volume, tabulation_schema(fixed))?;
    for (block, bytes) in blobs {
        table.write(block, bytes)?;
    }
    ordered_rows(&mut table, volume, fixed)
}

/// [`merge_tabulation`] over a stream in a store.
///
/// Streams the fragments one at a time — `Environment::sidecar_fragments` would
/// make every blob resident on top of the table that is about to hold their
/// rows, which doubles the one residency this operation has. `phase` is half the
/// address: a stream written by two phases holds two generations, and a blob
/// from the wrong one would decode perfectly and answer differently.
pub fn collect_tabulation(
    env: &dyn crate::env::Environment,
    stream: &str,
    phase: usize,
    volume: [usize; 3],
    fixed: FixedPoint,
) -> Result<Vec<RegionValues>> {
    let mut table = Table::new(volume, tabulation_schema(fixed))?;
    crate::fragment::fold_fragments(env, stream, &mut |key, bytes| {
        if key.phase != phase {
            return Ok(());
        }
        table.write(key.block, bytes)
    })?;
    ordered_rows(&mut table, volume, fixed)
}

fn ordered_rows(
    table: &mut Table,
    volume: [usize; 3],
    fixed: FixedPoint,
) -> Result<Vec<RegionValues>> {
    table.seal()?;
    let mut found = Vec::with_capacity(table.len());
    // A loop rather than a `collect` on the tail expression: the scan borrows
    // the table, and a borrow in a function's final expression outlives the
    // local it is taken from.
    for row in table.scan(&Region::whole(&volume))? {
        found.push(region_values(&row, fixed)?);
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxels::Voxels;
    use ndarray::Array3;

    fn array(shape: [usize; 3], values: &[f64]) -> BlockBuf {
        BlockBuf::Array(Voxels::from(
            Array3::from_shape_vec((shape[0], shape[1], shape[2]), values.to_vec())
                .expect("a shaped array"),
        ))
    }

    fn tabulator(fixed: FixedPoint) -> TabulateValuesOp {
        TabulateValuesOp::new("tabulate", 0, 1, fixed, "partials", Lifecycle::DeleteOnExit)
            .expect("two different levels")
    }

    // ------------------------------------------------------ the fixed point --

    #[test]
    fn the_fixed_point_states_its_range_and_its_resolution() {
        let fixed = FixedPoint::default();
        assert_eq!(fixed.fraction_bits(), 20);
        assert_eq!(fixed.resolution(), 2f64.powi(-20));
        assert_eq!(fixed.limit(), 2f64.powi(43));
        // and the two really do trade against each other
        let coarse = FixedPoint::bits(4).expect("four bits");
        assert!(coarse.resolution() > fixed.resolution());
        assert!(coarse.limit() > fixed.limit());
        assert!(FixedPoint::bits(MAX_FRACTION_BITS + 1).is_err());
    }

    #[test]
    fn a_quantised_value_round_trips_within_the_resolution_and_negatives_are_exact() {
        let fixed = FixedPoint::default();
        for value in [0.0, 1.0, -1.0, 0.5, -0.5, 1234.5, -1234.5, 1e-6, -1e-6] {
            let quantised = fixed.quantise(value).expect("finite").expect("finite");
            let back = fixed.value_of(quantised as i64);
            assert!(
                (back - value).abs() <= fixed.resolution() / 2.0,
                "{value} came back as {back}"
            );
        }
        // exact, not merely close, for a value on the lattice
        assert_eq!(fixed.quantise(-3.0).unwrap(), Some(-3 * (1 << 20)));
    }

    /// The column order is the *value* order, which two's complement would not
    /// give and which the canonical row order depends on.
    #[test]
    fn the_signed_column_is_offset_binary_so_its_bits_order_as_its_values_do() {
        let fixed = FixedPoint::default();
        let words: Vec<u64> = [-1000i128, -1, 0, 1, 1000]
            .into_iter()
            .map(|value| fixed.to_column(value).expect("in range"))
            .collect();
        let mut sorted = words.clone();
        sorted.sort_unstable();
        assert_eq!(words, sorted, "the bits do not order as the values do");
        for (word, value) in words.iter().zip([-1000i64, -1, 0, 1, 1000]) {
            assert_eq!(fixed.from_column(*word), value);
        }
        assert!(fixed.to_column(1i128 << 63).is_err());
        assert!(fixed.to_column(-(1i128 << 63) - 1).is_err());
    }

    #[test]
    fn a_value_too_large_for_the_fixed_point_is_refused_by_name() {
        let fixed = FixedPoint::default();
        let failed = fixed
            .quantise(1e300)
            .expect_err("1e300 does not fit 43 bits of range")
            .to_string();
        assert!(failed.contains("1e300"), "{failed}");
        assert!(failed.contains("fraction bit"), "{failed}");
        assert!(failed.contains("8.796093022208e12"), "{failed}");
        // and the limit itself is where it says it is
        assert!(fixed.quantise(fixed.limit() - 1.0).is_ok());
        assert!(fixed.quantise(fixed.limit() * 2.0).is_err());
    }

    // ------------------------------------------------------------ the tally --

    /// The property `SeamFold::Unordered` claims, asserted directly on the
    /// combine rather than through a run: every permutation of the same partials
    /// gives the same tally, bit for bit.
    #[test]
    fn merging_partials_is_independent_of_their_order() {
        let fixed = FixedPoint::default();
        let mut pieces = Vec::new();
        for (voxel, value) in [(0usize, 1.0f64), (1, -2.5), (2, 1e9), (3, -1e9), (4, 0.25)] {
            let mut tally = Tally::new(7);
            tally.add([voxel, 0, 0], value, fixed).unwrap();
            pieces.push(tally);
        }
        let reference = {
            let mut total = pieces[0];
            for piece in &pieces[1..] {
                total.merge(piece).unwrap();
            }
            total
        };
        // Every rotation, and the reverse of each, which is what the executor's
        // reversal check exercises one block at a time.
        for rotate in 0..pieces.len() {
            for reversed in [false, true] {
                let mut order: Vec<Tally> = pieces[rotate..]
                    .iter()
                    .chain(pieces[..rotate].iter())
                    .copied()
                    .collect();
                if reversed {
                    order.reverse();
                }
                let mut total = order[0];
                for piece in &order[1..] {
                    total.merge(piece).unwrap();
                }
                assert_eq!(
                    total, reference,
                    "rotation {rotate}, reversed {reversed} gave a different tally"
                );
            }
        }
        // and it is the whole-array answer, not merely a stable one
        assert_eq!(reference.count, 5);
        assert_eq!(reference.nonfinite, 0);
        assert_eq!(
            reference.sum,
            fixed
                .quantise(1.0 - 2.5 + 1e9 - 1e9 + 0.25)
                .unwrap()
                .unwrap()
        );
        // and the selection is the voxels' own values, not a quantisation of
        // them: `-2.5` and `1e9` are both on the fixed point's lattice here, so
        // the discriminating fixture is the one in `region_tabulation.rs`; what
        // this asserts is that the *bits* survived the permutations above.
        assert_eq!(reference.min.map(f64::to_bits), Some((-1e9f64).to_bits()));
        assert_eq!(reference.max.map(f64::to_bits), Some(1e9f64.to_bits()));
        // The first moment folded the same way, and it is the *whole* moment:
        // `sum(v * z)` over the five voxels at z = 0..5.
        let expected_moment: i128 = [(0usize, 1.0f64), (1, -2.5), (2, 1e9), (3, -1e9), (4, 0.25)]
            .into_iter()
            .map(|(voxel, value)| {
                fixed.quantise(value).unwrap().unwrap() * i128::try_from(voxel).unwrap()
            })
            .sum();
        assert_eq!(reference.moment[0], expected_moment);
        // and the two axes that hold no coordinate hold no moment either
        assert_eq!(reference.moment[1], 0);
        assert_eq!(reference.moment[2], 0);
    }

    /// **The weighted centroid is not a reading of the columns beside it**, and
    /// this is the assertion rather than the claim: two tallies with the *same*
    /// `count`, the same `sum` and the same `position` — so the same unweighted
    /// centroid, exactly — whose weighted centroids differ.
    ///
    /// Two voxels at `z = 0` and `z = 4`. One region puts `3` at the near voxel
    /// and `1` at the far one, the other the reverse. Both total `4` over two
    /// voxels whose coordinates total `4`, so every existing column agrees; the
    /// moments are `0*3 + 4*1 = 4` and `0*1 + 4*3 = 12`, and the centres are
    /// `4/4 = 1` and `12/4 = 3`.
    #[test]
    // Both moments are written as `value * coordinate` for **both** voxels, so
    // the two lines read as each other's mirror and a reader can check the cross
    // moment term by term. Collapsing the `* 0` and the `1 *` — which is what
    // clippy asks for — would delete exactly the halves that make the pair
    // comparable, so the lint is refused here rather than obeyed.
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn the_weighted_centroid_is_not_determined_by_the_count_the_sum_and_the_positions() {
        let fixed = FixedPoint::default();
        let mut near_heavy = Tally::new(1);
        near_heavy.add([0, 0, 0], 3.0, fixed).unwrap();
        near_heavy.add([4, 0, 0], 1.0, fixed).unwrap();
        let mut far_heavy = Tally::new(1);
        far_heavy.add([0, 0, 0], 1.0, fixed).unwrap();
        far_heavy.add([4, 0, 0], 3.0, fixed).unwrap();

        // Everything the op reported before this column is identical.
        assert_eq!(near_heavy.count, far_heavy.count);
        assert_eq!(near_heavy.sum, far_heavy.sum);
        assert_eq!(near_heavy.position, far_heavy.position);
        assert_eq!(near_heavy.centroid(), far_heavy.centroid());
        assert_eq!(near_heavy.min, far_heavy.min);
        assert_eq!(near_heavy.max, far_heavy.max);

        // And the new column is not. The arithmetic is written out rather than
        // taken from the code: one unit of value is `2^20` fixed-point steps.
        let one = 1i128 << 20;
        assert_eq!(near_heavy.moment[0], 3 * one * 0 + 1 * one * 4);
        assert_eq!(far_heavy.moment[0], 1 * one * 0 + 3 * one * 4);
        assert_eq!(near_heavy.weighted_centroid(), Some([1.0, 0.0, 0.0]));
        assert_eq!(far_heavy.weighted_centroid(), Some([3.0, 0.0, 0.0]));
        // neither of which is the unweighted centre of the same two voxels
        assert_eq!(near_heavy.centroid(), Some([2, 0, 0]));
    }

    /// The degenerate denominator, both ways it arises, pinned to `None`.
    ///
    /// A region of zeros and a region whose values cancelled are the same fact —
    /// `sum == 0`, so `moment / sum` does not exist — and they are reported the
    /// same way. The moments themselves stay exact and stay written: `sum(v*x)`
    /// is defined whatever `sum(v)` is, and the cancelling region's is not even
    /// zero.
    #[test]
    fn a_region_whose_values_total_zero_has_no_weighted_centroid_and_says_so() {
        let fixed = FixedPoint::default();
        let one = 1i128 << 20;

        // Nothing but zeros: `0/0`.
        let mut zeros = Tally::new(1);
        zeros.add([1, 0, 0], 0.0, fixed).unwrap();
        zeros.add([2, 0, 0], 0.0, fixed).unwrap();
        assert_eq!(zeros.sum, 0);
        assert_eq!(zeros.moment, [0, 0, 0]);
        assert_eq!(zeros.weighted_centroid(), None);
        // and the unweighted one is still there, which is what makes the absence
        // an absence rather than a failure of the row
        assert_eq!(zeros.centroid(), Some([2, 0, 0]));

        // Values that cancelled: `k/0`, with `k` non-zero. The numerator is a
        // real number and the quotient still does not exist.
        let mut cancelling = Tally::new(2);
        cancelling.add([1, 0, 0], 5.0, fixed).unwrap();
        cancelling.add([3, 0, 0], -5.0, fixed).unwrap();
        assert_eq!(cancelling.sum, 0);
        assert_eq!(cancelling.moment[0], 5 * one - 15 * one);
        assert_ne!(cancelling.moment[0], 0, "the numerator is not zero");
        assert_eq!(cancelling.weighted_centroid(), None);

        // Every value non-finite is the same denominator by a different road.
        let mut nothing = Tally::new(3);
        nothing.add([1, 0, 0], f64::NAN, fixed).unwrap();
        assert_eq!(nothing.sum, 0);
        assert_eq!(nothing.moment, [0, 0, 0]);
        assert_eq!(nothing.weighted_centroid(), None);
    }

    /// **Signed values put the quotient outside the region, and that is the
    /// answer rather than a defect.** Two voxels at `z = 10` and `z = 11` whose
    /// values are `1` and `-0.5`: the sum is `0.5`, the moment is `10 - 5.5 =
    /// 4.5`, and the centre is `9`, which is not between the two voxels.
    ///
    /// Pinned, because the alternatives — clamping it into the bounding box,
    /// refusing a value array that straddles zero — would each report a different
    /// quantity from the one the column names.
    #[test]
    fn a_weighted_centroid_over_signed_values_may_fall_outside_the_region() {
        let fixed = FixedPoint::default();
        let mut tally = Tally::new(1);
        tally.add([10, 0, 0], 1.0, fixed).unwrap();
        tally.add([11, 0, 0], -0.5, fixed).unwrap();
        let one = 1i128 << 20;
        assert_eq!(tally.sum, one / 2);
        assert_eq!(tally.moment[0], 10 * one - 11 * one / 2);
        let centre = tally.weighted_centroid().expect("a non-zero denominator");
        assert_eq!(centre[0], 9.0);
        assert!(
            centre[0] < 10.0,
            "the fixture was supposed to leave the region"
        );
        // and the geometric centre did not leave it
        assert_eq!(tally.centroid(), Some([11, 0, 0]));
    }

    /// The selection is a **semilattice in `f64`**, and the one pair that could
    /// have made it not one is `-0.0` against `0.0`: they compare equal and are
    /// different bits, so `f64::min` is free to return either operand and the
    /// merge would depend on the order after all. `total_cmp` is what removes
    /// that, and this is the assertion rather than the comment.
    #[test]
    fn a_signed_zero_selects_the_same_bits_whichever_way_round_it_arrives() {
        let fixed = FixedPoint::default();
        let mut forwards = Tally::new(1);
        forwards.add([0, 0, 0], -0.0, fixed).unwrap();
        forwards.add([1, 0, 0], 0.0, fixed).unwrap();
        let mut backwards = Tally::new(1);
        backwards.add([1, 0, 0], 0.0, fixed).unwrap();
        backwards.add([0, 0, 0], -0.0, fixed).unwrap();

        assert_eq!(
            forwards.min.map(f64::to_bits),
            backwards.min.map(f64::to_bits)
        );
        assert_eq!(
            forwards.max.map(f64::to_bits),
            backwards.max.map(f64::to_bits)
        );
        assert_eq!(forwards.min.map(f64::to_bits), Some((-0.0f64).to_bits()));
        assert_eq!(forwards.max.map(f64::to_bits), Some(0.0f64.to_bits()));
        // and the same through the merge, which is where the seam is
        let mut one = Tally::new(1);
        one.add([0, 0, 0], -0.0, fixed).unwrap();
        let mut other = Tally::new(1);
        other.add([1, 0, 0], 0.0, fixed).unwrap();
        let mut left = one;
        left.merge(&other).unwrap();
        let mut right = other;
        right.merge(&one).unwrap();
        assert_eq!(left, right, "a merged signed zero moved with the order");
        assert_eq!(left.min.map(f64::to_bits), Some((-0.0f64).to_bits()));
    }

    #[test]
    fn partials_round_trip_and_a_truncated_one_is_refused() {
        let fixed = FixedPoint::default();
        let mut tallies = BTreeMap::new();
        let mut one = Tally::new(3);
        one.add([1, 2, 3], -4.5, fixed).unwrap();
        one.add([1, 2, 4], f64::NAN, fixed).unwrap();
        tallies.insert(3, one);
        tallies.insert(9, Tally::new(9));
        let bytes = encode_partial(&tallies);
        let back = decode_partial(&bytes).expect("a partial");
        assert_eq!(back.len(), 2);
        assert_eq!(back[0], one);
        // the label with nothing in it keeps "no extreme" as a distinct fact
        assert_eq!(back[1].min, None);
        assert_eq!(back[1].max, None);
        assert!(decode_partial(&bytes[..bytes.len() - 8]).is_err());
    }

    // -------------------------------------------------------- the tabulation --

    #[test]
    fn a_block_tallies_only_its_core_and_skips_the_background() {
        let fixed = FixedPoint::default();
        let op = tabulator(fixed);
        let read = Region::new(&[0, 0, 0], &[4, 1, 1]);
        let core = Region::new(&[1, 0, 0], &[2, 1, 1]);
        let labels = array([4, 1, 1], &[1.0, 1.0, 0.0, 2.0]);
        let values = array([4, 1, 1], &[10.0, 20.0, 30.0, 40.0]);
        let tallies = op.tally_block(&labels, &values, &read, &core).unwrap();
        // voxel 0 is outside the core, voxel 2 is background, voxel 3 is outside
        assert_eq!(tallies.len(), 1);
        assert_eq!(tallies[&1].count, 1);
        assert_eq!(tallies[&1].sum, fixed.quantise(20.0).unwrap().unwrap());
    }

    /// Both stated outcomes at once: a `NaN` and an infinity leave the
    /// reductions alone and appear in `nonfinite`, and a negative sums exactly.
    #[test]
    fn non_finite_values_are_set_aside_and_counted_rather_than_folded_in() {
        let fixed = FixedPoint::default();
        let op = tabulator(fixed);
        let read = Region::new(&[0, 0, 0], &[4, 1, 1]);
        let labels = array([4, 1, 1], &[5.0, 5.0, 5.0, 5.0]);
        let values = array([4, 1, 1], &[-2.0, f64::NAN, f64::INFINITY, 0.5]);
        let tallies = op.tally_block(&labels, &values, &read, &read).unwrap();
        let tally = tallies[&5];
        assert_eq!(tally.count, 4);
        assert_eq!(tally.nonfinite, 2);
        assert_eq!(tally.sum, fixed.quantise(-1.5).unwrap().unwrap());
        // The values themselves, not their quantisations, and not the infinity
        // that `total_cmp` would otherwise have ranked above every one of them.
        assert_eq!(tally.min.map(f64::to_bits), Some((-2.0f64).to_bits()));
        assert_eq!(tally.max.map(f64::to_bits), Some(0.5f64.to_bits()));
        // and the centroid still counts every voxel, including the two set aside
        assert_eq!(tally.position, [6, 0, 0], "0 + 1 + 2 + 3");
    }

    #[test]
    fn a_region_whose_every_value_is_non_finite_reports_zero_and_says_why() {
        let fixed = FixedPoint::default();
        let op = tabulator(fixed);
        let read = Region::new(&[0, 0, 0], &[2, 1, 1]);
        let labels = array([2, 1, 1], &[8.0, 8.0]);
        let values = array([2, 1, 1], &[f64::NAN, f64::NEG_INFINITY]);
        let tally = op.tally_block(&labels, &values, &read, &read).unwrap()[&8];
        assert_eq!(tally.nonfinite, tally.count);
        assert_eq!(tally.min, None);
        assert_eq!(tally.max, None);
        let words = tally.row_words(fixed).unwrap().expect("a row");
        assert_eq!(words[POSITION_WORDS + 3], fixed.to_column(0).unwrap());
        // `0.0`, not a `NaN`: `Table::write` refuses a non-finite `F64` column,
        // so the absence is reported by `nonfinite == count` and by nothing else.
        assert_eq!(words[POSITION_WORDS + 4], 0.0f64.to_bits());
        assert_eq!(words[POSITION_WORDS + 5], 0.0f64.to_bits());
        // The moments are the fixed-point zero on the same terms — nothing
        // finite reached them — and the weighted centroid is `None` rather than
        // a point, because a zero sum is a zero denominator.
        for axis in 0..3 {
            assert_eq!(
                words[POSITION_WORDS + 9 + axis],
                fixed.to_column(0).unwrap()
            );
        }
        assert_eq!(tally.weighted_centroid(), None);
    }

    #[test]
    fn a_label_that_is_not_a_whole_positive_number_is_refused_by_name() {
        let fixed = FixedPoint::default();
        let op = tabulator(fixed);
        let read = Region::new(&[0, 0, 0], &[1, 1, 1]);
        let values = array([1, 1, 1], &[1.0]);
        for bad in [-1.0, 0.5, f64::NAN] {
            let labels = array([1, 1, 1], &[bad]);
            let failed = op
                .tally_block(&labels, &values, &read, &read)
                .expect_err("not a label")
                .to_string();
            assert!(failed.contains("not a label"), "{failed}");
        }
    }

    #[test]
    fn a_simulated_block_writes_a_fragment_that_is_present_and_empty() {
        let fixed = FixedPoint::default();
        let op = tabulator(fixed);
        let read = Region::new(&[0, 0, 0], &[2, 1, 1]);
        let accounted = BlockBuf::Accounted {
            region: read.clone(),
            dtype: Dtype::F64,
            uniform: None,
        };
        let tallies = op
            .tally_block(&accounted, &accounted, &read, &read)
            .unwrap();
        assert!(tallies.is_empty());
        assert!(encode_partial(&tallies).is_empty());
    }

    #[test]
    fn reducing_an_array_over_its_own_regions_is_refused() {
        let failed = TabulateValuesOp::new(
            "tabulate",
            2,
            2,
            FixedPoint::default(),
            "partials",
            Lifecycle::DeleteOnExit,
        )
        .err()
        .expect("one level cannot be both operands")
        .to_string();
        assert!(failed.contains("level 2"), "{failed}");
    }

    // ------------------------------------------------------------ the schema --

    #[test]
    fn the_schema_is_twelve_columns_and_the_accumulated_values_carry_the_scale() {
        let schema = tabulation_schema(FixedPoint::default());
        assert_eq!(schema.len(), COLUMNS);
        assert_eq!(schema.width(), ROW_WORDS);
        let expected = [
            ("label", ColumnType::U64),
            ("count", ColumnType::U64),
            ("nonfinite", ColumnType::U64),
            ("sum_q20", ColumnType::U64),
            ("min", ColumnType::F64),
            ("max", ColumnType::F64),
            ("sum_0", ColumnType::U64),
            ("sum_1", ColumnType::U64),
            ("sum_2", ColumnType::U64),
            ("moment_0_q20", ColumnType::U64),
            ("moment_1_q20", ColumnType::U64),
            ("moment_2_q20", ColumnType::U64),
        ];
        for (index, column) in schema.columns().iter().enumerate() {
            assert_eq!(column.name(), expected[index].0);
            assert_eq!(column.kind(), expected[index].1);
        }
        // Every accumulated column is `U64` — that entry condition has not
        // moved. What moved is which columns accumulate.
        for name in [
            "label",
            "count",
            "nonfinite",
            "sum_q20",
            "sum_0",
            "moment_0_q20",
        ] {
            let index = schema.index_of(name).expect("a named column");
            assert_eq!(schema.columns()[index].kind(), ColumnType::U64);
        }
        // The nine columns that existed before the moments kept their indices,
        // which is what "appended" means and is the only reason it matters where
        // they went.
        for (index, name) in ["label", "count", "nonfinite", "sum_q20", "min", "max"]
            .into_iter()
            .enumerate()
        {
            assert_eq!(schema.index_of(name), Some(index));
        }
        for (axis, name) in POSITION_SUM.into_iter().enumerate() {
            assert_eq!(schema.index_of(name), Some(6 + axis));
        }

        // a different scale is a different schema, which is what stops two
        // tabulations at two scales being merged into one table
        let other = tabulation_schema(FixedPoint::bits(8).unwrap());
        assert_ne!(other, schema);
        // and the selection columns are the same in both, because they have no
        // scale to differ in — as are the coordinate sums, which are integers
        // that never needed one
        for name in [MIN, MAX, POSITION_SUM[0], POSITION_SUM[1], POSITION_SUM[2]] {
            assert!(
                other.index_of(name).is_some(),
                "{name} moved with the scale"
            );
        }
        // The moments did move with it, because a moment is an accumulation.
        for stem in MOMENT {
            assert!(schema.index_of(&format!("{stem}_q20")).is_some());
            assert!(other.index_of(&format!("{stem}_q8")).is_some());
            assert!(
                other.index_of(&format!("{stem}_q20")).is_none(),
                "{stem} did not move with the scale"
            );
        }
    }

    /// The moment narrows to its column at the same boundary the sum does, and
    /// the refusal **names the axis** — because the moment's range is the sum's
    /// divided by the largest coordinate, so it is the column a run hits first
    /// and a message that did not say which one would send a caller to the wrong
    /// number.
    #[test]
    fn a_first_moment_too_large_for_its_column_is_refused_naming_the_axis() {
        // 62 fraction bits leaves +/- 2 of range. A value of 1 at coordinate 3
        // on axis 1 fits the sum — one unit — and gives a moment of three, which
        // does not.
        let fixed = FixedPoint::bits(62).expect("62 bits");
        let mut tally = Tally::new(1);
        tally.add([0, 3, 0], 1.0, fixed).unwrap();
        assert_eq!(tally.sum, 1i128 << 62);
        assert_eq!(tally.moment[1], 3 * (1i128 << 62));
        // the sum on its own is representable
        assert!(fixed.to_column(tally.sum).is_ok());
        let failed = tally
            .row_words(fixed)
            .expect_err("three units at 62 fraction bits is 3 * 2^62")
            .to_string();
        assert!(failed.contains("first moment on axis 1"), "{failed}");
        assert!(failed.contains("binds first"), "{failed}");
        assert!(failed.contains("fewer fraction bits"), "{failed}");
    }
}
