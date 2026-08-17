// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation —
// a fixed sequence of offsets, walked in order until a test on a stored array
// holds — not adapted from any implementation of it.
//
// **A table of rows in, the same rows with one more column out, where the
// column is a distance read out of a volume.**
//
// One row is one *starting coordinate*. The op holds an [`OffsetSequence`]: a
// fixed list of relative offsets, in a fixed order, with a distance attached to
// each. For every row it walks that list from the front, tests a [`Limit`] on
// the array at `row + offset`, and stops at the first offset where the test
// holds. What it writes is **the distance attached to the offset it stopped
// at** — not the offset. The difference between those two answers is the
// subject of this file and is stated in "One answer is exact and the other is
// not" below.
//
// Why this is a `FragmentOp` that reads no pixels
// -----------------------------------------------
// The rows arrive as a fragment from an earlier phase and the array is a
// **second** array, declared with [`FragmentOp::source_inputs`] and handed over
// by `apply_with`. That is the arrangement `ops::rows` describes for a gather
// and for the same reason: a fragment phase `p` reads level `p`, the phase
// before a row op is the row *producer*, and a producer writes fragments and no
// level — so there is no arrangement of phases in which this op reads its rows
// from an earlier phase and reaches the array through `reads_pixels`. The
// operand declaration is the one that fits, and it is the one that costs one
// array rather than two.
//
// The reach is a **stated maximum**, and it is the offset list itself
// -------------------------------------------------------------------
// A walk of this shape has a data-dependent stopping point: a row in a thick
// region walks further than one in a thin region, and nothing at plan time
// knows which is which. That is the hard question for a framework whose whole
// premise is a declared reach, and this file answers it in the only way that
// does not lie: **the walk cannot go further than the offset list, because the
// offset list is finite and fixed before any row is read.** So the reach is
// `max |offset|` per axis, exactly, derived from the sequence rather than
// configured beside it, and a row whose walk reaches the end of the list has an
// answer that says so rather than a distance.
//
// There is no barrier here and none is needed. What would have needed one is
// the *unbounded* version of this question — walk outward until the test holds,
// however far that is — and this op does not offer it, because a reach that is
// "however far the data says" is a reach that cannot be declared and therefore
// cannot be checked. A caller wanting a larger answer states a larger maximum
// and pays for a larger halo, which is a decision written into the plan.
//
// **The one thing that must not happen is a walk truncated at a block seam and
// reported as a distance.** That is a wrong answer wearing the shape of a
// measurement, and it is refused rather than avoided by argument: for every
// offset, [`walk_from`] distinguishes three cases and only two of them are
// answers.
//
// * the target is **outside the volume** — skipped, and the walk continues.
//   That is the behaviour being reproduced, and it is not the same as stopping:
//   a row on the volume's face has offsets that name nothing, and treating them
//   as a stop would report a distance the array never justified;
// * the target is **inside the volume and inside the block's fetched window** —
//   read and tested;
// * the target is **inside the volume and outside the fetched window** —
//   **refused**, naming the offset and the window. This cannot happen with the
//   reach this op declares, which is exactly why it is checked: the check costs
//   two comparisons per offset and it is the only thing standing between a
//   short halo and a plausible number.
//
// One answer is exact and the other is not
// ----------------------------------------
// The order of the offsets is the whole of this operation, and the order that
// is being reproduced comes from an **unstable sort**. Offsets at equal
// ordering key are permuted by it in a way that depends on the machine's vector
// width, so it is not a rule that can be written down and re-implemented. What
// follows is not one limitation but two, and they differ:
//
// * **the distance is exact.** It does not depend on *which* of several
//   equidistant offsets came first, only on which *level* the walk stopped in —
//   and the level is determined, because a sort by key is total across distinct
//   keys whatever it does inside a tie. So a distance produced here is
//   byte-identical to the one the unstable order produces;
// * **the identity of the stopping offset is not.** It names a particular
//   member of a tie group, and the member is chosen by an order this crate
//   cannot reproduce.
//
// So the op writes the distance and does not write the offset. The offset is
// still available — [`walk_from`] returns the index, because a caller inside
// the crate has uses for it — and it comes with the one thing that makes it
// honest: [`OffsetSequence::stop_is_determined`] answers, **for that particular
// index**, whether the answer was forced or was a pick among equals. An index
// in a tie group of one is a fact; an index in a tie group of thirty is this
// crate's arbitrary choice and disagrees with the order being reproduced.
//
// **No tie-break is invented to make the second case look settled.** The order
// within a level here is ascending lexicographic, which is a stated choice and
// not an approximation of anything: the order being reproduced is neither
// lexicographic nor stable nor any other rule, and a different order that
// disagrees is not closer to it than an honest statement that it cannot agree.
// [`OffsetSequence::with_ties_reversed`] exists so that the dependence can be
// *measured* rather than described — the same idiom `SeamFold::Unordered` uses,
// where the claim of order-independence is checked by applying the thing again
// in the opposite order and comparing bytes.
//
// The exactness of the distance is a **precondition, and it is checked**
// ----------------------------------------------------------------------
// "The distance does not depend on the tie order" is true only while every
// offset at one ordering key reports one distance. That is not automatic. It
// holds for an isotropic maximum under an isotropic spacing, and it **fails**
// as soon as the two disagree: with a maximum of `[3, 3, 1]` and unit spacing,
// the outermost level holds both the offset `[0, 0, 1]` at distance `1` and the
// offset `[3, 0, 0]` at distance `3`, because the key normalises each axis by
// its own maximum and the distance does not. In that configuration the
// *distance* is a function of the tie order too, and nothing can make it exact.
//
// So [`OffsetSequence`] checks it at construction and **refuses** a sequence
// that fails it, naming the key and the two distances. A refusal at
// construction is the whole value of the check: the alternative is a number
// that looks like every other number this op produces and is reproducible on
// one machine only.
//
// What a row is allowed to be, and where it must be
// -------------------------------------------------
// A block is handed the rows of its own fragment and the array over its own
// window, and those two agree only while every row lies inside the block's
// core. That is the precondition `ops::rows` states for a gather, it holds for
// the rows this crate's producers write, and it stops holding after a scale —
// so it is checked here too, per row, naming the row and the region.
//
// What it costs
// -------------
// Per row, at worst one pass over the offset list, which is `len()` bounds
// tests and up to `len()` reads of single voxels. Per block, one fetch of the
// operand over the core dilated by the stated maximum. Nothing is accumulated
// across rows and nothing is accumulated across blocks, which is why the seam
// declaration is [`SeamFold::PerBlock`] and why it is true rather than hoped.

use std::sync::Arc;

use crate::env::BlockBuf;
use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, SeamFold,
    SourceBlocks,
};
use crate::op::SourceInput;
use crate::ops::rows::{value_at, Limit, RowStreams};
use crate::reach::Reach;
use crate::region::Region;
use crate::table::{Column, Row, RowBuilder, Schema, Table, Value};
use crate::voxels::Voxels;

// ------------------------------------------------------ the offset sequence --

/// A fixed list of relative offsets, in a fixed order, each with the distance
/// it reports.
///
/// Three lists of the same length and one invariant tying them together:
///
/// * `keys` is **non-decreasing**, which is what makes the list an order;
/// * `distances` is **constant wherever `keys` is constant**, which is what
///   makes the reported answer independent of how the equal keys were
///   permuted. See the module header: this is the precondition the exactness
///   claim rests on, and it is checked here rather than assumed.
///
/// Neither the offsets nor the order are defaulted. `ops::element`'s
/// [`StructuringElement`](crate::ops::element::StructuringElement) builds offset
/// *sets* including this one, but its offsets are documented to come back in
/// ascending lexicographic order and that order is part of its contract — "a
/// windowed sum walks these in sequence, and a different sequence is a
/// different floating-point answer". A distance ordering is a different order
/// over the same set, so it is built here and that type is left alone.
#[derive(Debug, Clone, PartialEq)]
pub struct OffsetSequence {
    offsets: Vec<[isize; 3]>,
    keys: Vec<f64>,
    distances: Vec<f64>,
    spans: [usize; 3],
}

impl OffsetSequence {
    /// The offsets inside an ellipsoid of half-extent `maximum`, ordered
    /// outward, with distances measured under `spacing`.
    ///
    /// The membership rule and the ordering key are the same quantity, and it is
    /// the one an axis-normalised ellipsoid is defined by: with
    /// `d_a = max(1, maximum_a)`,
    ///
    /// ```text
    /// key(o) = ((o_0/d_0)^2 + (o_1/d_1)^2) + (o_2/d_2)^2,   member iff key <= 1
    /// ```
    ///
    /// summed left to right in `f64`, because a sum of three floats does not
    /// associate and the membership of an offset on the surface is decided by
    /// the last bit of it. The distance is a **different** quantity —
    /// `|o| * spacing`, Euclidean, summed left to right likewise — and the two
    /// coincide in *order* only when `spacing_a * maximum_a` is the same on
    /// every axis. Where they do not, the constructor refuses; see the module
    /// header.
    ///
    /// A `maximum` of `[0, 0, 0]` gives the single offset `[0, 0, 0]`, which is
    /// a legitimate degenerate: the walk tests the row's own voxel and nothing
    /// else.
    pub fn ellipsoid(maximum: [usize; 3], spacing: [f64; 3]) -> Result<Self> {
        for (axis, value) in spacing.iter().enumerate() {
            if !value.is_finite() || *value <= 0.0 {
                return Err(Error::invalid(format!(
                    "the spacing on axis {axis} is {value}, and a distance measured under it \
                     would be negative, zero or not a number. Every axis of a spacing is a \
                     positive finite length."
                )));
            }
        }
        let divisor = [
            maximum[0].max(1) as f64,
            maximum[1].max(1) as f64,
            maximum[2].max(1) as f64,
        ];
        let mut members = Vec::new();
        for a in -(maximum[0] as isize)..=(maximum[0] as isize) {
            for b in -(maximum[1] as isize)..=(maximum[1] as isize) {
                for c in -(maximum[2] as isize)..=(maximum[2] as isize) {
                    let offset = [a, b, c];
                    let mut key = 0.0f64;
                    for axis in 0..3 {
                        let scaled = offset[axis] as f64 / divisor[axis];
                        key += scaled * scaled;
                    }
                    if key <= 1.0 {
                        members.push((offset, key));
                    }
                }
            }
        }
        Self::assemble(members, spacing)
    }

    /// An arbitrary offset set, ordered by the distance it reports.
    ///
    /// The ordering key **is** the distance here, so a tie in the order is a tie
    /// in the answer by construction and the exactness precondition cannot fail.
    /// What it buys a caller is an offset set that is not an ellipsoid; what it
    /// does not buy is the ordering of the sequence above, which normalises each
    /// axis by its own maximum and is therefore a different order over the same
    /// set whenever the maxima differ.
    pub fn from_offsets(
        offsets: impl IntoIterator<Item = [isize; 3]>,
        spacing: [f64; 3],
    ) -> Result<Self> {
        for (axis, value) in spacing.iter().enumerate() {
            if !value.is_finite() || *value <= 0.0 {
                return Err(Error::invalid(format!(
                    "the spacing on axis {axis} is {value}, and a distance measured under it \
                     would be negative, zero or not a number. Every axis of a spacing is a \
                     positive finite length."
                )));
            }
        }
        let members: Vec<([isize; 3], f64)> = offsets
            .into_iter()
            .map(|offset| (offset, distance_of(offset, spacing)))
            .collect();
        Self::assemble(members, spacing)
    }

    /// Sort, validate, and record. The one place a sequence comes into
    /// existence, so the invariants have one place to hold.
    fn assemble(mut members: Vec<([isize; 3], f64)>, spacing: [f64; 3]) -> Result<Self> {
        if members.is_empty() {
            return Err(Error::invalid(
                "an offset sequence needs at least one offset. A walk over an empty sequence \
                 stops at nothing for every row, so every row would be given the same \
                 not-found answer — a complete, well-formed column that measured nothing. It \
                 is refused here rather than produced."
                    .to_string(),
            ));
        }
        // The tie rule, stated: ascending lexicographic within a level. It is a
        // choice and not a reconstruction — see the module header on why no
        // tie-break can be the right one — and it is here so that this crate's
        // own answers are at least a function of the offset set.
        members.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        for pair in members.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(Error::invalid(format!(
                    "the offset {:?} appears more than once in the sequence. A repeated offset \
                     is tested twice at the same distance, which changes nothing about the \
                     answer and everything about the index that names it, so it is refused \
                     rather than folded away.",
                    pair[0].0
                )));
            }
        }
        let offsets: Vec<[isize; 3]> = members.iter().map(|(offset, _)| *offset).collect();
        let keys: Vec<f64> = members.iter().map(|(_, key)| *key).collect();
        let distances: Vec<f64> = offsets
            .iter()
            .map(|offset| distance_of(*offset, spacing))
            .collect();
        // **The exactness precondition.** Every offset at one key must report
        // one distance, or the answer this op writes is a function of an order
        // nothing here can reproduce. Refused at construction, naming both
        // distances, because the alternative is a number that looks like every
        // other number.
        for index in 1..keys.len() {
            if keys[index] == keys[index - 1] && distances[index] != distances[index - 1] {
                return Err(Error::invalid(format!(
                    "the offsets {:?} and {:?} share the ordering key {} and report the \
                     different distances {} and {}. The order among offsets of equal key is \
                     not reproducible, so a walk stopping in that group would report whichever \
                     of the two the order happened to put first — the distance would be a \
                     property of the sort rather than of the data. The usual cause is a \
                     maximum and a spacing that disagree: the key normalises each axis by its \
                     own maximum and the distance does not, so they order the same set the \
                     same way only when `spacing[axis] * maximum[axis]` is equal on every \
                     axis.",
                    offsets[index - 1],
                    offsets[index],
                    keys[index],
                    distances[index - 1],
                    distances[index]
                )));
            }
        }
        let mut spans = [0usize; 3];
        for offset in &offsets {
            for axis in 0..3 {
                spans[axis] = spans[axis].max(offset[axis].unsigned_abs());
            }
        }
        Ok(Self {
            offsets,
            keys,
            distances,
            spans,
        })
    }

    /// The same set, the same levels, each level's members in the opposite
    /// order.
    ///
    /// **A measuring instrument, not a variant a caller picks between.** The
    /// order among equal keys is not reproducible here, so the honest thing to
    /// do with any answer that depends on it is to *measure* the dependence:
    /// walk twice, once with each, and compare. The distances must agree, and
    /// [`Self::stop_is_determined`] says exactly where the indices may not. It
    /// is the same idiom `SeamFold::Unordered` is checked by.
    pub fn with_ties_reversed(&self) -> Self {
        let mut offsets = self.offsets.clone();
        let mut distances = self.distances.clone();
        let mut start = 0;
        while start < self.keys.len() {
            let mut end = start + 1;
            while end < self.keys.len() && self.keys[end] == self.keys[start] {
                end += 1;
            }
            offsets[start..end].reverse();
            distances[start..end].reverse();
            start = end;
        }
        Self {
            offsets,
            keys: self.keys.clone(),
            distances,
            spans: self.spans,
        }
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Never true — an empty sequence is refused at construction. Present
    /// because clippy asks for it beside `len`, and because a caller reading it
    /// should see the answer rather than write the check.
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// The offsets, in walk order.
    pub fn offsets(&self) -> &[[isize; 3]] {
        &self.offsets
    }

    /// The ordering key of each offset: non-decreasing, and the thing ties are
    /// ties *in*.
    pub fn keys(&self) -> &[f64] {
        &self.keys
    }

    /// The distance each offset reports. Constant wherever [`Self::keys`] is.
    pub fn distances(&self) -> &[f64] {
        &self.distances
    }

    /// The largest `|offset|` on each axis: the stated maximum, and the reach
    /// this op declares.
    pub fn maximum(&self) -> [usize; 3] {
        self.spans
    }

    /// [`Self::maximum`] as the operand reach a phase is planned with. Derived
    /// from the offsets rather than stated beside them, so there is nothing that
    /// can drift from the walk.
    pub fn reach(&self) -> Reach {
        Reach::symmetric(self.spans)
    }

    /// The half-open range of indices sharing `index`'s ordering key.
    ///
    /// Panics on an index past the end, which is a caller bug rather than a
    /// data condition: every index this crate hands out came from a walk over
    /// this sequence.
    pub fn tie_group(&self, index: usize) -> (usize, usize) {
        assert!(
            index < self.keys.len(),
            "index {index} is past the end of a sequence of {} offsets",
            self.keys.len()
        );
        let key = self.keys[index];
        let mut start = index;
        while start > 0 && self.keys[start - 1] == key {
            start -= 1;
        }
        let mut end = index + 1;
        while end < self.keys.len() && self.keys[end] == key {
            end += 1;
        }
        (start, end)
    }

    /// Was *this* stop forced, or was it a pick among equals?
    ///
    /// **The one thing that makes an index from [`walk_from`] honest.** `true`
    /// means the offset at `index` is the only one at its ordering key, so it is
    /// the offset the order being reproduced would have stopped at too. `false`
    /// means it is one of several and the choice among them is this crate's tie
    /// rule rather than a fact about the data — the *distance* is still exact,
    /// which is why that is what the op writes.
    pub fn stop_is_determined(&self, index: usize) -> bool {
        let (start, end) = self.tie_group(index);
        end - start == 1
    }

    /// Does any offset report exactly `distance`?
    ///
    /// What [`OffsetWalkOp`] validates its not-found value against, so that a
    /// value standing for "the walk did not stop" can never be mistaken for one
    /// that stands for a measurement.
    pub fn reports(&self, distance: f64) -> bool {
        self.distances.contains(&distance)
    }
}

/// `sqrt((|o_0| s_0)^2 + (|o_1| s_1)^2 + (|o_2| s_2)^2)`, summed left to right.
///
/// The association is written out because a sum of three `f64`s does not
/// associate and this number is compared byte for byte.
fn distance_of(offset: [isize; 3], spacing: [f64; 3]) -> f64 {
    let mut total = 0.0f64;
    for axis in 0..3 {
        let side = offset[axis].unsigned_abs() as f64 * spacing[axis];
        total += side * side;
    }
    total.sqrt()
}

// ---------------------------------------------------------------- the walk --

/// Walk `sequence` from `at`, and give the index of the first offset where
/// `stop` holds on `pixels`.
///
/// `pixels` holds the array over a window whose lowest voxel sits at `origin` in
/// a volume of shape `volume`. The three cases an offset can be in are the
/// module header's, and only two of them are answers:
///
/// * outside `volume` — skipped, and the walk continues to the next offset;
/// * inside `volume` and inside the window — read and tested;
/// * inside `volume` and outside the window — **refused**, naming the offset.
///   A walk that quietly ended there would report the distance it had reached,
///   which is a shorter answer than the data justifies and is indistinguishable
///   from a real one.
///
/// `None` means the sequence ran out: the walk reached the stated maximum
/// without the test ever holding. That is a different fact from a distance and
/// the caller is expected to keep it one.
///
/// **The index is not reproducible where it lands in a tie group.** Ask
/// [`OffsetSequence::stop_is_determined`] about it. The *distance* at that index
/// is exact whatever the group.
pub fn walk_from(
    sequence: &OffsetSequence,
    at: [usize; 3],
    volume: [usize; 3],
    pixels: &Voxels,
    origin: [usize; 3],
    stop: Limit,
) -> Result<Option<usize>> {
    let window = pixels.shape();
    'offsets: for (index, offset) in sequence.offsets().iter().enumerate() {
        let mut local = [0usize; 3];
        for axis in 0..3 {
            let target = at[axis] as isize + offset[axis];
            if target < 0 || target >= volume[axis] as isize {
                // Outside the volume: this offset names nothing, so it is
                // skipped and the walk goes on. Not a stop — see the module
                // header, and see `a_row_on_the_volumes_face_skips_what_is_not_there`.
                continue 'offsets;
            }
            let target = target as usize;
            if target < origin[axis] || target - origin[axis] >= window[axis] {
                return Err(Error::invalid(format!(
                    "a walk from {at:?} reached {target} on axis {axis} — offset {offset:?} — \
                     which is inside the volume {volume:?} and outside the window this block \
                     was handed, which starts at {origin:?} and is shaped {window:?}. The \
                     walk cannot be continued and it must not be ended, because ending it \
                     would report the distance reached so far as though the array had \
                     justified it. The operand reach must be at least the sequence's stated \
                     maximum {:?} on every axis.",
                    sequence.maximum()
                )));
            }
            local[axis] = target - origin[axis];
        }
        if stop.holds(value_at(pixels, local)?) {
            return Ok(Some(index));
        }
    }
    Ok(None)
}

/// The distance a walk's outcome reports: the stopping offset's, or
/// `not_found`.
///
/// A function rather than an expression at each call site so that the two
/// outcomes are joined in exactly one place, and so that the not-found value is
/// visibly a parameter rather than a constant somebody chose.
pub fn walked_distance(sequence: &OffsetSequence, stopped: Option<usize>, not_found: f64) -> f64 {
    match stopped {
        Some(index) => sequence.distances()[index],
        None => not_found,
    }
}

// ----------------------------------------------------------------- the rows --

/// The schema a walk produces: the input's columns, then one more.
///
/// Appended rather than inserted, so a consumer holding a column index into the
/// input keeps it. Refuses a name the input already uses, naming the operation
/// that caused it.
pub fn walk_schema(input: &Schema, column: &str) -> Result<Schema> {
    if input.index_of(column).is_some() {
        return Err(Error::invalid(format!(
            "a walk was asked to write a column named {column:?} and the rows already have \
             one. Two columns of that name would make every message about \"the column named \
             {column:?}\" ambiguous, and the measured one would be indistinguishable from the \
             one that was already there."
        )));
    }
    let mut columns = input.columns().to_vec();
    columns.push(Column::f64(column));
    Schema::new(columns)
}

/// Every row of `table`, with the distance its walk stopped at appended.
///
/// `within` is the region the rows are **required** to lie in and `origin` is
/// where `pixels[0, 0, 0]` sits in the volume. A row outside `within` is refused
/// naming the row and the region, for the reason `ops::rows` gives for the same
/// check on a gather: a block reads its own window, so a row belonging somewhere
/// else would be measured against the wrong array.
///
/// The coordinate travels untouched. A walk is about *what is around* a row, not
/// about where the row is.
#[allow(clippy::too_many_arguments)]
pub fn walk_into(
    table: &Table,
    volume: [usize; 3],
    within: &Region,
    sequence: &OffsetSequence,
    stop: Limit,
    not_found: f64,
    pixels: &Voxels,
    origin: [usize; 3],
    out: &mut RowBuilder,
) -> Result<()> {
    crate::ops::rows::walk_rows(table, volume, out, |row| {
        let at = row.at();
        for axis in 0..3 {
            if at[axis] < within.start[axis] || at[axis] >= within.start[axis] + within.shape[axis]
            {
                return Err(Error::invalid(format!(
                    "a walk was handed a row at {at:?} and a region starting {:?} of shape \
                     {:?}, which does not hold it: it is outside on axis {axis}. A block walks \
                     the array its own window holds, so a row somewhere else would be measured \
                     against the wrong voxels. The usual cause is a scale between the rows and \
                     this phase, which moves rows out of the block that carries them; merge \
                     and re-scatter over the new volume rather than widening a reach, because \
                     a scaled row can move arbitrarily far.",
                    within.start, within.shape
                )));
            }
        }
        let stopped = walk_from(sequence, at, volume, pixels, origin, stop)?;
        let mut values = values_of(row)?;
        values.push(Value::F64(walked_distance(sequence, stopped, not_found)));
        Ok(Some((at, values)))
    })
}

fn values_of(row: &Row<'_>) -> Result<Vec<Value>> {
    let mut values = Vec::with_capacity(row.schema().len());
    for column in 0..row.schema().len() {
        values.push(row.value(column)?);
    }
    Ok(values)
}

/// One blob in, one blob out: [`walk_into`] over the bytes a block holds.
#[allow(clippy::too_many_arguments)]
pub fn walk_blob(
    volume: [usize; 3],
    schema: &Schema,
    blob: &[u8],
    column: &str,
    within: &Region,
    sequence: &OffsetSequence,
    stop: Limit,
    not_found: f64,
    pixels: &Voxels,
    origin: [usize; 3],
) -> Result<Vec<u8>> {
    let mut table = Table::new(volume, schema.clone())?;
    table.write([0, 0, 0], blob)?;
    table.seal()?;
    let mut out = RowBuilder::new(Arc::new(walk_schema(schema, column)?));
    walk_into(
        &table, volume, within, sequence, stop, not_found, pixels, origin, &mut out,
    )?;
    Ok(out.encode())
}

// ------------------------------------------------------------------ the op --

/// **Rows in, the same rows with a measured distance out.**
///
/// Reads no pixels of its own level, reads one stored level as an operand,
/// writes no level, and has reach zero: the answer for a row is written for the
/// row's own coordinate, whatever window the operand was read over. The operand
/// reach is the sequence's stated maximum and is derived from it.
///
/// The column it appends is the **distance**. The identity of the offset the
/// walk stopped at is not written, and the module header says why.
pub struct OffsetWalkOp {
    name: &'static str,
    rows: RowStreams,
    level: usize,
    column: String,
    sequence: OffsetSequence,
    stop: Limit,
    not_found: f64,
}

impl OffsetWalkOp {
    /// `level` is the stored array walked, in the numbering
    /// `PhaseDecomposition::source_levels` uses. `not_found` is what a row whose
    /// walk reached the stated maximum is given.
    ///
    /// **`not_found` is a parameter, it must be finite, and it may not be a
    /// distance the sequence reports.** Finite because a table column refuses
    /// anything else — the canonical row order tiebreaks on the column's bits —
    /// which means the infinity a caller might reach for is not available here
    /// and the substitution is stated rather than made quietly. Distinguishable
    /// because a not-found marker that collides with a real answer is precisely
    /// a wrong answer wearing the shape of a measurement; the check is exact,
    /// since the set of distances the sequence can report is known before any
    /// row is read.
    pub fn new(
        name: &'static str,
        rows: RowStreams,
        level: usize,
        column: impl Into<String>,
        sequence: OffsetSequence,
        stop: Limit,
        not_found: f64,
    ) -> Result<Self> {
        let column = column.into();
        walk_schema(&rows.schema, &column)?;
        if !not_found.is_finite() {
            return Err(Error::invalid(format!(
                "the not-found value is {not_found}, and a table's f64 column refuses anything \
                 that is not finite: the canonical row order tiebreaks on the column's bits, so \
                 a non-finite one would make the order of the answer undefined. State a finite \
                 value that the sequence cannot report — a negative one is the obvious choice, \
                 since every distance is non-negative."
            )));
        }
        if sequence.reports(not_found) {
            return Err(Error::invalid(format!(
                "the not-found value is {not_found} and the sequence reports that same distance \
                 for at least one offset, so a row whose walk reached the stated maximum would \
                 be indistinguishable from a row that stopped. State a value no offset can \
                 report; every distance here is non-negative, so a negative one always works."
            )));
        }
        Ok(Self {
            name,
            rows,
            level,
            column,
            sequence,
            stop,
            not_found,
        })
    }

    /// The schema of the rows this op emits: its input's, with the distance
    /// column appended.
    pub fn schema(&self) -> Result<Schema> {
        walk_schema(&self.rows.schema, &self.column)
    }

    pub fn sequence(&self) -> &OffsetSequence {
        &self.sequence
    }

    pub fn column(&self) -> &str {
        &self.column
    }

    pub fn not_found(&self) -> f64 {
        self.not_found
    }
}

impl FragmentOp for OffsetWalkOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        // Zero, and it is not the same number as the operand reach. A row's
        // answer is written for the row's own coordinate; the window the
        // operand was read over widens the halo and not the region this block
        // is authoritative for. `fragment.rs` states the distinction.
        0
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        // Reach `[0, 0, 0]`: this block's rows and no neighbour's. A row is read
        // by exactly one block, and an overlap here would duplicate rows rather
        // than cost recomputation.
        vec![FragmentInput::own(self.rows.input.clone(), self.rows.phase)]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.rows.output.clone(),
            self.rows.lifecycle,
            // Every block, always. This phase writes no level, so the tiling
            // check has nothing to bite on and this declaration is the only
            // guard there is.
            Coverage::EveryBlock,
        )]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(self.level, self.sequence.reach())]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        // Nothing crosses a seam. Each row's answer is a function of that row
        // and of the array around it, and no two blocks contribute to one
        // answer — there is no accumulation here at all, so there is no order
        // for one to depend on.
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(format!(
            "the op {:?} walks a stored array, which it cannot do without the operand it \
             declared. It is applied through `apply_with`.",
            self.name
        )))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let schema = self.schema()?;
        let blob = at.own(&self.rows.input).unwrap_or(&[]);
        let BlockBuf::Array(pixels) = sources.get(self.level)? else {
            // An accounting run holds no data. It still writes a fragment,
            // because what such a run measures is the IO and a phase that
            // silently produced nothing would be a measurement of a different
            // program. What it must not do is invent distances, so the fragment
            // is present and empty rather than present and fabricated.
            return Ok(BlockOutput::fragment(
                self.rows.output.clone(),
                RowBuilder::new(Arc::new(schema)).encode(),
            ));
        };
        // The operand covers the block's fetch region, whose lowest voxel is
        // where this block's anchor sits. Taken from the anchor rather than from
        // `read` because that is the region the executor read the operand over.
        let origin = at.at.offset;
        Ok(BlockOutput::fragment(
            self.rows.output.clone(),
            walk_blob(
                at.volume(),
                &self.rows.schema,
                blob,
                &self.column,
                at.core,
                &self.sequence,
                self.stop,
                self.not_found,
                pixels,
                origin,
            )?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array3;

    /// The sizes the ellipsoid rule produces, as a fingerprint of the
    /// arithmetic.
    ///
    /// These are not round numbers and they are not monotone in any simple way
    /// — the count of distinct ordering *levels* falls from 56 at maximum 7 to
    /// 55 at maximum 8 — because membership on the surface is decided by the
    /// last bit of a sum of three `f64`s. Pinning them is what makes a later
    /// "simplification" of that arithmetic fail here.
    #[test]
    fn the_ellipsoid_rule_has_a_fingerprint() {
        let cases = [
            (0usize, 1usize, 1usize),
            (1, 7, 2),
            (2, 33, 5),
            (3, 123, 10),
            (4, 257, 15),
            (5, 515, 27),
            (7, 1419, 56),
            (8, 2109, 55),
            (10, 4169, 132),
        ];
        for (maximum, offsets, levels) in cases {
            let sequence = OffsetSequence::ellipsoid([maximum; 3], [1.0; 3])
                .expect("an isotropic sequence is exact");
            assert_eq!(sequence.len(), offsets, "maximum {maximum}");
            let distinct = sequence
                .keys()
                .windows(2)
                .filter(|pair| pair[0] != pair[1])
                .count()
                + 1;
            assert_eq!(distinct, levels, "maximum {maximum}");
            assert_eq!(sequence.maximum(), [maximum; 3]);
        }
    }

    /// **The distance a level reports, bit for bit.**
    ///
    /// One line per ordering level of the maximum-3 sequence, as the exact
    /// `f64` bits, so that a change to the arithmetic that moves a last bit
    /// fails here rather than in a comparison somewhere downstream. Compared as
    /// bits and not as numbers, because a distance that is one unit in the last
    /// place away prints identically and is not the same answer.
    ///
    /// **There are ten levels and only nine distinct distances**, and the
    /// duplicate is the whole reason this is pinned at all: the offsets with
    /// squared length 6 fall into *two* ordering levels, because the key is a
    /// sum of three `f64`s and `(1/3)^2 + (1/3)^2 + (2/3)^2` and
    /// `(2/3)^2 + (1/3)^2 + (1/3)^2` do not associate to the same number. Both
    /// levels report the same distance, which is why that split costs nothing —
    /// and it is exactly the kind of detail a "simplification" of the key would
    /// quietly remove.
    #[test]
    fn each_level_reports_one_distance_and_here_are_the_bits() {
        let sequence =
            OffsetSequence::ellipsoid([3, 3, 3], [1.0; 3]).expect("an isotropic sequence");
        let expected: [u64; 10] = [
            0x0000000000000000, // 0
            0x3ff0000000000000, // 1
            0x3ff6a09e667f3bcd, // sqrt 2
            0x3ffbb67ae8584caa, // sqrt 3
            0x4000000000000000, // 2
            0x4001e3779b97f4a8, // sqrt 5
            0x4003988e1409212e, // sqrt 6
            0x4003988e1409212e, // sqrt 6 again: see above
            0x4006a09e667f3bcd, // 2 sqrt 2
            0x4008000000000000, // 3
        ];
        let mut levels = Vec::new();
        for index in 0..sequence.len() {
            if index == 0 || sequence.keys()[index] != sequence.keys()[index - 1] {
                levels.push(sequence.distances()[index].to_bits());
            }
            // The invariant the exactness claim rests on, asserted over every
            // offset rather than only at the level boundaries.
            let (start, _) = sequence.tie_group(index);
            assert_eq!(
                sequence.distances()[index].to_bits(),
                sequence.distances()[start].to_bits(),
                "offset {:?} reports a different distance from the first of its level",
                sequence.offsets()[index]
            );
        }
        assert_eq!(levels, expected);
        assert_eq!(levels[6], levels[7], "the split level is really a split");
    }

    /// **The set is `ops::element`'s; only the order is this module's.**
    ///
    /// `StructuringElement`'s ellipsoid is the same membership test — each
    /// offset divided by the side it is on, squared, summed left to right,
    /// compared `<= 1` — so the two agree offset for offset, and this pins that
    /// they do. What that type cannot supply is the *ordering*: its `offsets()`
    /// is documented to come back in ascending lexicographic order and that
    /// order is part of its contract, because a windowed sum walks it in
    /// sequence and a different sequence is a different floating-point answer.
    ///
    /// So the ordering is built here and that type is left alone. The keys are
    /// not recomputed from its output either: the key *is* the membership test,
    /// and computing it twice in two files that must agree in the last bit is
    /// the drift this assertion exists to catch instead.
    #[test]
    fn the_offset_set_is_the_one_ops_element_builds() {
        use crate::ops::element::{ElementShape, StructuringElement};
        for maximum in [1usize, 2, 3, 5, 8] {
            let sequence = OffsetSequence::ellipsoid([maximum; 3], [1.0; 3])
                .expect("an isotropic sequence is exact");
            let element = StructuringElement::from_radius(ElementShape::Ellipsoid, [maximum; 3]);
            let mut mine = sequence.offsets().to_vec();
            mine.sort_unstable();
            assert_eq!(
                mine,
                element.offsets(),
                "the two membership rules disagree at maximum {maximum}"
            );
            // And the orders really are different, or there would have been
            // nothing to build here.
            assert_ne!(sequence.offsets(), element.offsets());
        }
    }

    /// The first offset is the anchor and it is alone at its key, so a walk that
    /// stops immediately is the one case whose *index* is reproducible too.
    #[test]
    fn the_sequence_starts_at_the_anchor_and_is_ordered() {
        let sequence =
            OffsetSequence::ellipsoid([3, 3, 3], [1.0; 3]).expect("an isotropic sequence");
        assert_eq!(sequence.offsets()[0], [0, 0, 0]);
        assert_eq!(sequence.distances()[0], 0.0);
        assert!(sequence.stop_is_determined(0));
        for pair in sequence.keys().windows(2) {
            assert!(pair[0] <= pair[1], "the keys must not decrease");
        }
        for pair in sequence.distances().windows(2) {
            assert!(pair[0] <= pair[1], "the distances must not decrease either");
        }
        // The last level is the stated maximum and it is a large tie group, so
        // the fixture below really does exercise the case it claims to.
        let last = sequence.len() - 1;
        assert_eq!(sequence.distances()[last], 3.0);
        assert!(!sequence.stop_is_determined(last));
    }

    /// **The precondition, in the direction that fails.** A maximum of
    /// `[3, 3, 1]` normalises the flat axis by 1 and the others by 3, so the
    /// outermost key holds offsets a whole unit apart in reported distance.
    /// Nothing can make that answer exact, and it is refused rather than
    /// produced.
    #[test]
    fn a_sequence_whose_ties_are_not_distance_ties_is_refused() {
        let failed = OffsetSequence::ellipsoid([3, 3, 1], [1.0; 3])
            .expect_err("an anisotropic maximum under a uniform spacing cannot be exact")
            .to_string();
        assert!(failed.contains("ordering key"), "{failed}");
        assert!(failed.contains("not reproducible"), "{failed}");
        assert!(failed.contains("spacing[axis] * maximum[axis]"), "{failed}");

        // The same hazard from the other side: an isotropic maximum under an
        // anisotropic spacing.
        let failed = OffsetSequence::ellipsoid([5, 5, 5], [1.0, 1.0, 2.0])
            .expect_err("an anisotropic spacing under a uniform maximum cannot be exact either")
            .to_string();
        assert!(failed.contains("ordering key"), "{failed}");

        // And the configuration where the two disagreements cancel is accepted,
        // which is what makes the check a rule rather than a ban on anisotropy:
        // `spacing * maximum` is 4 on every axis.
        let sequence = OffsetSequence::ellipsoid([4, 2, 2], [1.0, 2.0, 2.0])
            .expect("the key and the distance order this set the same way");
        assert_eq!(sequence.len(), 61);
    }

    /// An empty sequence would answer "not found" for every row: a complete,
    /// well-formed column that measured nothing.
    #[test]
    fn an_empty_offset_set_is_refused() {
        let failed = OffsetSequence::from_offsets(Vec::new(), [1.0; 3])
            .expect_err("a walk over no offsets is not a walk")
            .to_string();
        assert!(failed.contains("at least one offset"), "{failed}");
        assert!(failed.contains("measured nothing"), "{failed}");
    }

    #[test]
    fn a_repeated_offset_is_refused() {
        let failed = OffsetSequence::from_offsets([[0, 0, 0], [1, 0, 0], [1, 0, 0]], [1.0; 3])
            .expect_err("a repeated offset is refused")
            .to_string();
        assert!(failed.contains("more than once"), "{failed}");
    }

    /// Reversing the ties permutes the sequence and moves nothing between
    /// levels: same keys, same distances at every index, a different offset at
    /// some of them.
    #[test]
    fn reversing_the_ties_keeps_every_level_where_it_was() {
        let sequence =
            OffsetSequence::ellipsoid([3, 3, 3], [1.0; 3]).expect("an isotropic sequence");
        let reversed = sequence.with_ties_reversed();
        assert_eq!(reversed.keys(), sequence.keys());
        assert_eq!(reversed.distances(), sequence.distances());
        assert_eq!(reversed.maximum(), sequence.maximum());
        assert_ne!(
            reversed.offsets(),
            sequence.offsets(),
            "a sequence with ties must actually be permuted, or the instrument measures nothing"
        );
        // Reversal is an involution, which is what makes it a permutation of
        // each level rather than a re-sort.
        assert_eq!(reversed.with_ties_reversed().offsets(), sequence.offsets());
    }

    // ------------------------------------------------------- the fixture --

    /// A volume with a slab against the `x = 0` face, a larger body away from
    /// it, and three isolated holes inside the body.
    ///
    /// Every point in `POINTS` is placed against this and each one exercises a
    /// different outcome; see `EXPECTED`.
    fn fixture() -> Array3<f64> {
        let mut array = Array3::<f64>::zeros((16, 16, 16));
        for j in 0..16 {
            for k in 0..16 {
                for i in 0..3 {
                    array[[i, j, k]] = 1.0;
                }
            }
        }
        for i in 5..15 {
            for j in 3..13 {
                for k in 3..13 {
                    array[[i, j, k]] = 1.0;
                }
            }
        }
        for hole in [[6, 6, 6], [10, 10, 10], [12, 10, 10]] {
            array[hole] = 0.0;
        }
        array
    }

    const POINTS: [[usize; 3]; 8] = [
        [6, 6, 6],
        [7, 6, 6],
        [7, 7, 6],
        [7, 7, 7],
        [8, 6, 6],
        [9, 7, 7],
        [11, 10, 10],
        [0, 8, 8],
    ];

    /// What the walk reports for each of `POINTS`, written out rather than
    /// recomputed.
    ///
    /// The list is a statement of the rule and covers every outcome the
    /// acceptance asks for: a stop at zero (the row's own voxel already
    /// satisfies the test), stops at four different distances including two
    /// irrational ones, a stop at the stated maximum, and a row whose walk runs
    /// off the end of the sequence.
    const EXPECTED: [Option<f64>; 8] = [
        Some(0.0),
        Some(1.0),
        Some(std::f64::consts::SQRT_2),
        Some(1.7320508075688772),
        Some(2.0),
        None,
        Some(1.0),
        Some(3.0),
    ];

    fn sequence() -> OffsetSequence {
        OffsetSequence::ellipsoid([3, 3, 3], [1.0; 3]).expect("an isotropic sequence")
    }

    /// The walk over the whole volume in one piece: the oracle every decomposed
    /// answer is compared against.
    fn whole_volume(sequence: &OffsetSequence) -> Vec<Option<usize>> {
        let array: Voxels = fixture().into();
        POINTS
            .iter()
            .map(|at| {
                walk_from(
                    sequence,
                    *at,
                    [16, 16, 16],
                    &array,
                    [0, 0, 0],
                    Limit::AtMost(0.0),
                )
                .expect("the whole volume is its own window")
            })
            .collect()
    }

    #[test]
    fn the_distance_is_the_one_the_rule_states() {
        let sequence = sequence();
        for (point, expected) in POINTS.iter().zip(EXPECTED) {
            let array: Voxels = fixture().into();
            let stopped = walk_from(
                &sequence,
                *point,
                [16, 16, 16],
                &array,
                [0, 0, 0],
                Limit::AtMost(0.0),
            )
            .expect("the whole volume is its own window");
            match expected {
                Some(distance) => {
                    let index = stopped.expect("this row's walk stops");
                    assert_eq!(
                        sequence.distances()[index],
                        distance,
                        "the walk from {point:?} reports a different distance"
                    );
                }
                None => assert!(
                    stopped.is_none(),
                    "the walk from {point:?} was supposed to run off the end"
                ),
            }
        }
        // The fixture must contain all four kinds or it certifies less than it
        // claims: an immediate stop, a stop at the stated maximum, and a walk
        // that never stops.
        assert_eq!(EXPECTED[0], Some(0.0));
        assert_eq!(EXPECTED[7], Some(sequence.distances()[sequence.len() - 1]));
        assert!(EXPECTED[5].is_none());
    }

    /// **The inexact case, measured.**
    ///
    /// Walked twice over the same data with the two orders, and the two answers
    /// are compared in both directions: every *distance* is byte-identical, and
    /// at least one *offset* is different. The row at `[11, 10, 10]` has a hole
    /// on either side of it at exactly one unit, so which of the two stopped the
    /// walk is decided by nothing but the order among equals — and
    /// `stop_is_determined` says so about that index and not about the others.
    #[test]
    fn the_distance_is_exact_where_the_stopping_offset_is_not() {
        let sequence = sequence();
        let reversed = sequence.with_ties_reversed();
        let forwards = whole_volume(&sequence);
        let backwards = whole_volume(&reversed);

        let mut moved = 0;
        let mut tied = 0;
        for (index, (left, right)) in forwards.iter().zip(&backwards).enumerate() {
            let point = POINTS[index];
            assert_eq!(
                left.map(|i| sequence.distances()[i]),
                right.map(|i| reversed.distances()[i]),
                "the distance from {point:?} moved with the order among equal offsets, which \
                 the construction check was supposed to make impossible"
            );
            if let (Some(left), Some(right)) = (left, right) {
                if sequence.offsets()[*left] != reversed.offsets()[*right] {
                    moved += 1;
                    assert!(
                        !sequence.stop_is_determined(*left),
                        "the offset from {point:?} moved and the sequence claimed it was forced"
                    );
                }
                if !sequence.stop_is_determined(*left) {
                    tied += 1;
                }
            }
        }
        // A fixture with no equidistant candidates cannot see a tie-order
        // problem. This one has them, and at least one of them really does pick
        // a different offset under a different order.
        assert!(
            moved >= 1,
            "the fixture was supposed to contain a row whose stopping offset is a pick among \
             equals, and no row's offset moved"
        );
        assert!(tied >= 1);
        // And the row that stops at the anchor is forced, so "not determined" is
        // not simply true everywhere.
        let anchor = forwards[0].expect("the first row stops immediately");
        assert!(sequence.stop_is_determined(anchor));
    }

    /// **The anti-truncation guard.** A window too small for the sequence is
    /// refused at the offset that leaves it, rather than answering with the
    /// distance reached so far.
    #[test]
    fn a_window_shorter_than_the_stated_maximum_is_refused() {
        let sequence = sequence();
        let array: Voxels = fixture()
            .slice(ndarray::s![6..12, 6..12, 6..12])
            .to_owned()
            .into();
        let failed = walk_from(
            &sequence,
            [9, 7, 7],
            [16, 16, 16],
            &array,
            [6, 6, 6],
            Limit::AtMost(0.0),
        )
        .expect_err("a walk that leaves its window must not answer")
        .to_string();
        assert!(failed.contains("inside the volume"), "{failed}");
        assert!(failed.contains("outside the window"), "{failed}");
        assert!(failed.contains("[3, 3, 3]"), "{failed}");
    }

    /// **A row on the volume's face.** Offsets that leave the volume name
    /// nothing and are skipped; the walk continues past them. Had they counted
    /// as a stop this row would report `1`, and it reports `3`.
    #[test]
    fn a_row_on_the_volumes_face_skips_what_is_not_there() {
        let sequence = sequence();
        let array: Voxels = fixture().into();
        let stopped = walk_from(
            &sequence,
            [0, 8, 8],
            [16, 16, 16],
            &array,
            [0, 0, 0],
            Limit::AtMost(0.0),
        )
        .expect("the whole volume is its own window")
        .expect("this row's walk stops");
        assert_eq!(sequence.distances()[stopped], 3.0);
        assert_eq!(sequence.offsets()[stopped], [3, 0, 0]);
        // The discriminating half: an offset that left the volume came first and
        // would have stopped the walk at 1 if leaving had counted.
        let outside = sequence
            .offsets()
            .iter()
            .position(|offset| *offset == [-1, 0, 0])
            .expect("the sequence holds the offset that leaves the volume");
        assert!(outside < stopped);
        assert_eq!(sequence.distances()[outside], 1.0);
    }

    /// `Result::expect_err` wants the `Ok` type to be `Debug` and an op is not,
    /// so the refusal is unwrapped here instead of a derive being added to a
    /// type that has no other reason for one.
    fn refusal<T>(built: Result<T>, why: &str) -> String {
        match built {
            Ok(_) => panic!("{why}"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn a_not_found_value_that_collides_with_a_distance_is_refused() {
        let streams = RowStreams::new(
            "rows.in",
            0,
            "rows.out",
            crate::sidecar::Lifecycle::DeleteOnExit,
            crate::ops::coordinates::coordinate_schema(),
        )
        .expect("two different streams");
        let failed = refusal(
            OffsetWalkOp::new(
                "walk",
                streams.clone(),
                0,
                "distance",
                sequence(),
                Limit::AtMost(0.0),
                2.0,
            ),
            "2.0 is a distance this sequence reports",
        );
        assert!(failed.contains("indistinguishable"), "{failed}");

        let failed = refusal(
            OffsetWalkOp::new(
                "walk",
                streams.clone(),
                0,
                "distance",
                sequence(),
                Limit::AtMost(0.0),
                f64::INFINITY,
            ),
            "a table column refuses a non-finite value",
        );
        assert!(failed.contains("not finite"), "{failed}");

        OffsetWalkOp::new(
            "walk",
            streams,
            0,
            "distance",
            sequence(),
            Limit::AtMost(0.0),
            -1.0,
        )
        .expect("a negative value is one no distance can be");
    }
}
