// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definitions of the operations
// — a row in, a row out — not adapted from any implementation of them.
//
// **A table of rows in, a table of rows out.** The three operations here are
// the ones that transform a `crate::table::Table` rather than a volume:
//
// | op | what one row becomes | reads a level | its `FragmentOp` shell |
// |---|---|---|---|
// | scale | the same row at a scaled coordinate | no | [`ScaleRowsOp`] |
// | gather | the same row with one more column, read at the row's own coordinate | **yes — a declared second array** | [`GatherRowsOp`] |
// | filter | itself, or nothing | no | [`FilterRowsOp`] |
//
// One kernel, three ops, and why it is not one op
// -----------------------------------------------
// Mechanically all three are the same loop: walk the rows, emit zero or one row
// per row read. A single op parameterised by a `Fn(Row) -> Option<Row>` would
// express every one of them, and it was the first thing tried. It is rejected
// here, because what the three do *not* share is the part a caller has to
// reason about:
//
// * **the volume.** A scale moves every coordinate, so its output rows live in
//   a *different* volume from its input rows — see [`scaled_bound`], and see the
//   warning below about what that costs the block a row is in. A filter and a
//   gather leave every coordinate exactly where it was, so their output volume
//   is their input volume and cannot be got wrong;
// * **the order.** A filter's output is a **subsequence** of its input: it
//   removes rows and moves none, so its ordered output is its ordered input
//   restricted. A scale's is not — it re-sorts, because it moves rows past each
//   other — and two rows may land on one coordinate that were distinct before;
// * **the IO.** A gather reads a second array and does a block's worth of pixel
//   IO per phase. The other two read no pixels at all, and their phases cost one
//   fragment in and one fragment out.
//
// A single op cannot make any of those three declarations, because it would
// have to make the *weakest* of each: it would claim it might move a row, so
// nothing could rely on the subsequence property; and it would either read
// pixels always, taxing a filter with a volume-sized read it does not use, or
// never, which is a gather that cannot be written. So: **one kernel, spelled
// once in [`walk_rows`], and three ops that each declare something the other two
// cannot.**
//
// They decompose by **row range**, and there is no overlap
// -------------------------------------------------------
// This is the opposite of every volume op in this crate and it is worth stating
// here rather than leaving to a reader. A neighbourhood op reads outside the
// voxel it writes, so a block is given a halo and the halo is the whole subject
// of `decomposition.rs`. **A row op reads exactly one row to write one row.**
// Its reach is zero and it can never be anything else — there is no "the row
// next to this one", because a table is a set and its ordering is derived,
// not structural.
//
// So the unit of decomposition is a **range of rows**, `axes = [0]` on the row
// index, and a halo on that axis would be a **defect rather than a cost**: a row
// duplicated into two ranges is emitted twice, and the answer has an extra row
// in it that no check downstream can distinguish from a real one. An overlap in
// a volume op costs recomputation; an overlap here costs correctness. That is
// why nothing in this module takes a reach parameter, why the shells here return
// `0` from `reach` without consulting anything, and why the gather's kernel
// takes the region its rows must lie in rather than a halo width.
//
// The order is in the payload, not in the assignment — again
// ----------------------------------------------------------
// `ops::coordinates` records the finding this module inherits: a base index per
// block can only express a **block-major** order, and the canonical order of a
// table interleaves the blocks, so no assignment of index ranges to blocks can
// produce it. Nothing here assigns an output index to anything. Every op writes
// rows carrying their own coordinates and payload, and the order is restored at
// the merge by [`Table`], whose canonical order is the lexicographic order of
// the coordinate triple then the payload words.
//
// Two consequences, and both are properties rather than cases handled:
//
// * the answer is a function of the **row set** alone — not of how the rows were
//   split, not of which range finished first, not of the order the blobs are
//   handed to the merge. `merging_is_insensitive_to_how_the_rows_were_split`
//   asserts it directly, splitting the same rows every way;
// * a split that falls **inside a run of rows sharing a coordinate** is not a
//   case at all. Those rows tiebreak on their payload words, which travel with
//   them, so the run reassembles in the same order whichever side of it the cut
//   fell. A mechanism that numbered rows would have had to get this right; this
//   one has nothing to get wrong.
//
// The rounding rule, which is the one thing here that is not obvious
// ------------------------------------------------------------------
// [`ScaleRowsOp`] multiplies a coordinate by an `f64` and has to land on a
// `usize`. **The rule is round-half-to-**even**, [`f64::round_ties_even`], and
// `f64::round` is wrong.**
//
// This is not a hypothetical. It is a fixed defect in the port this crate was
// extracted alongside, recorded there as *"`np.round` is round-half-to-even;
// `f64::round` is not"*: `f64::round` breaks ties **away from zero**, NumPy and
// Python's builtin `round` break them **to even**, and the obvious-looking
// translation of one to the other is wrong at every tie. The measured cost of
// getting it wrong on one recorded path was 160 of 378 rows reading the wrong
// voxel — because a scale factor of one half puts an *exact* `.5` on about half
// of every axis, so on that path ties were not a corner case but the common
// case.
//
// The two rules differ **only** where the value is exactly `k + 0.5` with `k`
// odd: `0.5` is `0` under ties-to-even and `1` under `f64::round`, `2.5` is `2`
// against `3`, `4.5` is `4` against `5`. Where `k` is even they agree — `1.5`
// and `3.5` are both `2` and `4`. A fixture must therefore contain ties of
// *both* parities to be worth anything: one with only even-floored ties
// certifies nothing, and one with no ties at all certifies nothing whatever the
// factor, which is why [`the_rounding_rule_is_ties_to_even`] states its
// expectations against a written-out table of both rules rather than against a
// recomputation.
//
// How the gather gets its level, and why not through `reads_pixels`
// ------------------------------------------------------------------
// A gather reads a **level** at a scattered coordinate, which is a shape no
// other op in this file has. The shell to reach for looks like the one
// `ops::fill`'s second phase uses: `reads_pixels() == true` alongside a
// `FragmentInput::own` at reach `[0, 0, 0]`, so the executor reads this block's
// pixels, hands over this block's rows, and the op indexes one against the
// other. That combination is legal in the trait and **cannot be arranged in a
// plan**, for a reason about the *phase index* rather than about gathering, and
// both halves of it were measured against the executor rather than argued:
//
// * **a fragment phase `p` reads level `p`.** So a gather at phase `p` needs
//   phase `p - 1` to have written a level, and the phase before a row op is the
//   row *producer*, which writes fragments and no level. The executor says so:
//   *"phase 1 reads level 1, which phase 0 did not write: it runs a fragment op
//   that declares `writes_pixels() == false`"*;
// * **a fragment input must come from a strictly earlier phase.** So the gather
//   cannot be phase 0 — where it would read level 0, the array the run was
//   handed — because its rows would have to come from phase 0 too. Splitting it
//   into a second `execute_phases` over the same store does not help; the phase
//   index is a plan-local number and the check fires again: *"reads stream
//   \"rows.set\" from phase 0, which is this phase or a later one"*.
//
// **Both refusals still stand, and [`GatherRowsOp`] does not go round either.**
// What resolves it is that the level a gather wants was never level `p`: it is a
// *second* array, and [`FragmentOp::source_inputs`] is the declaration for one.
// So the shell declares `reads_pixels() == false` — it never touches the level
// its phase was handed, and `fragment.rs` is explicit that the two declarations
// are independent, so it *"pays for one array rather than two"* — names the level
// it samples with [`SourceInput::voxelwise`], and is applied through
// `apply_with`, whose default refuses rather than dropping the operand.
//
// The level it names may be any level at or below the phase's own, level 0
// included, and it is fetched over the block's own fetch region and recorded on
// the phase, so the DAG depends on its producer, the level lifetimes keep it
// alive and the read counters see it. **Rows from one array and values from a
// second** — the case the consumers have, and the one folding the gather into
// the row producer cannot serve — is therefore the ordinary arrangement rather
// than the awkward one; a gather naming the array its own rows came from is the
// degenerate case of the same declaration.
//
// The seam, which is a claim and not a formality
// -----------------------------------------------
// [`crate::fragment::SeamFold::PerBlock`]: **nothing here crosses a block
// boundary.** A row is read by exactly one block — the fragment reach is
// `[0, 0, 0]`, so a block is handed its own rows and no neighbour's — and its
// value comes from exactly one voxel, because the coordinate is used as it
// stands and cores tile with no overlap. Nothing is accumulated, so there is no
// order for an answer to depend on and no accumulator whose type could fail to
// associate.
//
// `Unordered` would also be a true statement here and would be a **worse** one:
// the executor's reversal check is skipped when the neighbourhood holds at most
// one fragment, which is exactly this op's case, so `Unordered` would be a claim
// nothing checks. `PerBlock` is checked — the framework refuses it beside a
// non-zero fragment reach — and it is the reach that carries the property, so
// the declaration and the thing it claims cannot drift apart.
//
// **Reach 0 is honest only because of a precondition, and the precondition is
// checked rather than assumed.** A block is handed the rows of *its own*
// fragment, and it reads *its own* block of the level; those two agree only while
// every row in a block's fragment lies inside that block's core. That holds for
// the rows `ops::coordinates` and `ops::detect` write, and it stops holding the
// moment a [`ScaleRowsOp`] runs: after a scale, block `B`'s fragment holds rows
// at scaled coordinates that are somewhere else entirely. A gather that trusted
// its reach would then read a real value at the wrong place, and every row would
// still be well-formed.
//
// So [`gather_into`] takes the region the rows must lie in and **refuses a row
// outside it, naming the row and the region**, and [`GatherRowsOp`] passes its
// block's *core* as that region — not its read extent, which a halo could widen
// past what the block owns. A scale followed by a gather over the same lattice
// fails loudly on the first row rather than answering. What a caller wanting that
// composition must do instead is merge the scaled rows and re-scatter them over
// the new volume's lattice, which is a phase boundary and not a halo; there is no
// reach that fixes it, because the rows can move arbitrarily far.
//
// Two other things a gather has to say, and both are said here rather than
// discovered:
//
// * **on a block boundary.** Cores are half-open and tile with no overlap, so a
//   coordinate on the boundary between two blocks belongs to exactly one of them
//   — the one whose core starts there. No halo, no tiebreak, no duplicate. The
//   value read is the value the whole-volume gather reads, because it is the
//   same voxel;
// * **outside the volume.** It cannot happen, and that is a property of the
//   store rather than a case: [`Table::write`] refuses a coordinate outside the
//   table's volume, so a row that could be gathered out of bounds cannot be in
//   the table to be gathered. What *can* happen is a row outside the **block**,
//   which is the precondition above and is refused. The application-level
//   conventions for a genuinely out-of-bounds row — clamp to the edge, fill an
//   invalid marker, drop the row — are three different answers that three
//   different consumers want, and none of them belongs in a library op: a caller
//   that gathers from a volume other than its table's own resolves it before the
//   rows reach here.
//
// A gathered value is a **finite `f64`**, and a level whose element type cannot
// be carried in one exactly — `u64` and `i64`, whose ranges exceed `f64`'s exact
// integers — is refused rather than rounded. Everything narrower converts with
// no loss. That keeps the schema a function of the column name alone, which is
// what a plan needs before it has seen any data.
//
// Filtering renumbers, and says so
// --------------------------------
// [`FilterRowsOp`] removes rows and moves none, so the surviving rows come back
// in the order they went in. It does **not** preserve a row's *index*, and it
// cannot: a table row's only name is its position in the ordered list, so
// deleting a row renames every row after it. That is stated behaviour rather
// than an accident, and `filtering_renumbers_the_survivors` asserts it in the
// discriminating direction — a survivor whose index moved — so that a future
// change to a numbering scheme fails here instead of downstream.
//
// A consumer that addresses rows by position must therefore not filter between
// taking an index and using it. One that needs a name that survives a filter
// carries it as a column, which is what the schema is for.
//
// The predicate is a **conjunction of bounds on named columns**, and each bound
// says whether it is strict. Both spellings exist because both are in the
// consumers: a half-open `[min, max)` range test, and a strictly-greater
// threshold. They are one comparison apart and nothing is gained by making a
// caller express one as the other on floats, where `>= min` and
// `> next_after(min)` are not the same predicate.
//
// What it costs
// -------------
// One pass over the rows per op, and a sort of `N` rows at the merge — the same
// bound `ops::coordinates` states and for the same reason. The rows are resident
// at the merge and only there; a block holds its own range.
//
// A per-block op decodes its blob through a [`Table`] rather than through a
// second decoder written here. That costs a sort of the block's own rows, which
// is not needed — the merge sorts everything again anyway. It is paid
// deliberately: the wire format has exactly one reader, in `table.rs`, and a
// second one in this file would be the drift `table.rs`'s header exists to
// prevent. If the sort ever shows up in a measurement, the fix is a public row
// iterator on the blob in `table.rs`, not a decoder here.

use std::sync::Arc;

use crate::env::{BlockBuf, Environment};
use crate::error::{Error, Result};
use crate::fragment::{
    fold_fragments, BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput,
    SeamFold, SourceBlocks,
};
use crate::op::SourceInput;
use crate::region::Region;
use crate::sidecar::Lifecycle;
use crate::table::{Column, Row, RowBuilder, Schema, Table, Value};
use crate::voxels::Voxels;

// ------------------------------------------------------- the ordered list --

/// One row, owned: the answer a merge returns.
///
/// [`Row`] is a cursor into a live table and cannot outlive it, which is right
/// for a streaming consumer and useless for an assertion. This is the value
/// form, and it carries the coordinate and every column so that two runs can be
/// compared **as a whole ordered list** rather than as a set — which is the only
/// comparison that can see a permutation.
#[derive(Debug, Clone, PartialEq)]
pub struct RowValues {
    pub at: [usize; 3],
    pub values: Vec<Value>,
}

impl RowValues {
    pub fn new(at: [usize; 3], values: Vec<Value>) -> Self {
        Self { at, values }
    }

    fn of(row: &Row<'_>) -> Result<Self> {
        let mut values = Vec::with_capacity(row.schema().len());
        for column in 0..row.schema().len() {
            values.push(row.value(column)?);
        }
        Ok(Self::new(row.at(), values))
    }
}

/// Every blob's rows, as one list in the canonical order.
///
/// **This is where the order is restored, and it is restored from the rows
/// rather than from anything about the run.** The result is a function of the
/// row set alone: not of how the rows were split into blobs, not of which range
/// finished first, not of the order the blobs arrive in.
///
/// The block index travels only so that a refusal can name the blob it came
/// from; [`Table::write`] keeps no trace of it, which is what makes an order
/// that tiebreaks on the source unrepresentable rather than discouraged.
pub fn merge_rows<'a>(
    volume: [usize; 3],
    schema: Schema,
    blobs: impl IntoIterator<Item = ([usize; 3], &'a [u8])>,
) -> Result<Vec<RowValues>> {
    let mut table = Table::new(volume, schema)?;
    for (block, bytes) in blobs {
        table.write(block, bytes)?;
    }
    ordered_rows(&mut table, volume)
}

/// [`merge_rows`] over a stream in a store.
///
/// Streams the fragments one at a time rather than gathering them, so the only
/// residency is the table that is about to hold the rows. `phase` is half the
/// address: a stream written by two phases holds two generations, and a blob
/// from the wrong one decodes perfectly and answers differently.
pub fn collect_rows(
    env: &dyn Environment,
    stream: &str,
    phase: usize,
    volume: [usize; 3],
    schema: Schema,
) -> Result<Vec<RowValues>> {
    let mut table = Table::new(volume, schema)?;
    fold_fragments(env, stream, &mut |key, bytes| {
        if key.phase != phase {
            return Ok(());
        }
        table.write(key.block, bytes)
    })?;
    ordered_rows(&mut table, volume)
}

fn ordered_rows(table: &mut Table, volume: [usize; 3]) -> Result<Vec<RowValues>> {
    table.seal()?;
    let mut found = Vec::with_capacity(table.len());
    for row in table.scan(&Region::whole(&volume))? {
        found.push(RowValues::of(&row)?);
    }
    Ok(found)
}

/// One blob's rows, in the canonical order, as a sealed table.
///
/// The one decode path in this module; see the module header on why it goes
/// through [`Table`] rather than through a second reader of the wire format.
fn sealed(volume: [usize; 3], schema: Schema, blob: &[u8]) -> Result<Table> {
    let mut table = Table::new(volume, schema)?;
    table.write([0, 0, 0], blob)?;
    table.seal()?;
    Ok(table)
}

/// The one loop all three operations are: walk the rows of `table`, emit zero
/// or one row per row read.
///
/// Written once so that a disagreement between the three is a difference in
/// their rules rather than a difference in their loops, and so that the
/// **no-overlap** property has one place to hold: `rule` is called exactly once
/// per row and cannot see a row twice.
pub fn walk_rows(
    table: &Table,
    volume: [usize; 3],
    out: &mut RowBuilder,
    mut rule: impl FnMut(&Row<'_>) -> Result<Option<([usize; 3], Vec<Value>)>>,
) -> Result<()> {
    for row in table.scan(&Region::whole(&volume))? {
        if let Some((at, values)) = rule(&row)? {
            out.push(at, &values)?;
        }
    }
    Ok(())
}

fn values_of(row: &Row<'_>) -> Result<Vec<Value>> {
    let mut values = Vec::with_capacity(row.schema().len());
    for column in 0..row.schema().len() {
        values.push(row.value(column)?);
    }
    Ok(values)
}

// --------------------------------------------------------- the map: scale --

/// One coordinate scaled and rounded, **ties to even**.
///
/// The whole of the module header's rounding section, as three lines. `f64::
/// round` here instead of [`f64::round_ties_even`] is the recorded defect and
/// differs on exactly the ties whose floor is odd.
///
/// Refuses a factor that is not finite and non-negative, and a product that does
/// not land in a `usize`. A negative factor would send rows to coordinates a
/// table cannot hold, and refusing it here names the factor rather than making
/// every row fail at the merge.
pub fn scaled_index(coordinate: usize, factor: f64) -> Result<usize> {
    if !factor.is_finite() || factor < 0.0 {
        return Err(Error::invalid(format!(
            "a scale factor of {factor} is not a finite, non-negative number. A table holds \
             `usize` coordinates, so a negative or non-finite factor produces rows no table can \
             accept, and it is refused where the factor is given rather than once per row."
        )));
    }
    // Rounded before the cast, and rounded to **even**: see the module header.
    // `np.round`, Python's builtin `round` and this agree; `f64::round` does not.
    let scaled = (coordinate as f64 * factor).round_ties_even();
    if !scaled.is_finite() || scaled < 0.0 || scaled > usize::MAX as f64 {
        return Err(Error::invalid(format!(
            "coordinate {coordinate} scaled by {factor} is {scaled}, which is not a coordinate \
             this platform can hold"
        )));
    }
    Ok(scaled as usize)
}

/// A whole coordinate triple, scaled.
pub fn scaled_at(at: [usize; 3], factor: [f64; 3]) -> Result<[usize; 3]> {
    let mut out = [0usize; 3];
    for axis in 0..3 {
        out[axis] = scaled_index(at[axis], factor[axis])?;
    }
    Ok(out)
}

/// The smallest volume that holds every coordinate of `volume` scaled by
/// `factor`.
///
/// **Derived, not guessed.** The largest coordinate a volume of `v` holds is
/// `v - 1`, the scale is monotone in the coordinate, so the largest coordinate
/// out is `scaled_index(v - 1)` and the volume that holds it is one more than
/// that. A caller is free to supply a larger output volume — a resampling has
/// its own target shape and it is usually not this — but supplying this one can
/// never refuse a row, and that is the point of offering it.
///
/// An empty axis stays empty: there is no coordinate to scale.
pub fn scaled_bound(volume: [usize; 3], factor: [f64; 3]) -> Result<[usize; 3]> {
    let mut out = [0usize; 3];
    for axis in 0..3 {
        out[axis] = if volume[axis] == 0 {
            0
        } else {
            scaled_index(volume[axis] - 1, factor[axis])?
                .checked_add(1)
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "the scaled extent of axis {axis} overflows a usize, which needs more \
                         voxels than the address space holds"
                    ))
                })?
        };
    }
    Ok(out)
}

/// Every row of `table`, at a scaled coordinate, with its columns unchanged.
///
/// The payload travels untouched: a scale is about **where** a row is. A column
/// that is itself a length and should scale with the coordinate is the caller's
/// to rescale, and it is not done here because whether a column is a length is
/// not something a schema says.
pub fn scale_into(
    table: &Table,
    volume: [usize; 3],
    factor: [f64; 3],
    out: &mut RowBuilder,
) -> Result<()> {
    walk_rows(table, volume, out, |row| {
        Ok(Some((scaled_at(row.at(), factor)?, values_of(row)?)))
    })
}

/// One blob in, one blob out: [`scale_into`] over the bytes a block holds.
///
/// `volume` is the **input** volume, which is what the rows arriving are stated
/// in. The rows leaving are in the scaled volume and this function does not know
/// it — that is the merge's business, and it is [`scaled_bound`] or the caller's
/// own target shape.
pub fn scale_blob(
    volume: [usize; 3],
    schema: &Schema,
    blob: &[u8],
    factor: [f64; 3],
) -> Result<Vec<u8>> {
    let table = sealed(volume, schema.clone(), blob)?;
    let mut out = RowBuilder::new(Arc::new(schema.clone()));
    scale_into(&table, volume, factor, &mut out)?;
    Ok(out.encode())
}

// ------------------------------------------------------------- the gather --

/// The schema a gather produces: the input's columns, then one more.
///
/// Appended rather than inserted, so a consumer holding a column index into the
/// input keeps it. Refuses a name the input already uses — [`Schema::new`] would
/// too, and refusing here names the operation that caused it.
pub fn gathered_schema(input: &Schema, column: &str) -> Result<Schema> {
    if input.index_of(column).is_some() {
        return Err(Error::invalid(format!(
            "a gather was asked to write a column named {column:?} and the rows already have \
             one. Two columns of that name would make every message about \"the column named \
             {column:?}\" ambiguous, and the gathered one would be indistinguishable from the \
             one that was already there."
        )));
    }
    let mut columns = input.columns().to_vec();
    columns.push(Column::f64(column));
    Schema::new(columns)
}

/// One voxel, as the finite `f64` a table column holds.
///
/// `u64` and `i64` are refused rather than rounded: their ranges exceed the
/// integers an `f64` holds exactly, so a value beyond `2^53` would arrive in the
/// table as a nearby number that is indistinguishable from a measurement. Every
/// narrower type converts exactly.
///
/// A non-finite value is refused naming the coordinate. An unwritten level in
/// this crate is filled with NaN precisely so that an absence cannot pass for a
/// value, and gathering one is that absence reaching a consumer.
pub fn value_at(pixels: &Voxels, at: [usize; 3]) -> Result<f64> {
    let shape = pixels.shape();
    for axis in 0..3 {
        if at[axis] >= shape[axis] {
            return Err(Error::invalid(format!(
                "a gather was asked for {at:?} of a buffer shaped {shape:?}, which is outside it \
                 on axis {axis}"
            )));
        }
    }
    let index = [at[0], at[1], at[2]];
    let value = match pixels {
        Voxels::Bool(array) => {
            if array[index] {
                1.0
            } else {
                0.0
            }
        }
        Voxels::U8(array) => array[index] as f64,
        Voxels::U16(array) => array[index] as f64,
        Voxels::U32(array) => array[index] as f64,
        Voxels::I8(array) => array[index] as f64,
        Voxels::I16(array) => array[index] as f64,
        Voxels::I32(array) => array[index] as f64,
        Voxels::F32(array) => array[index] as f64,
        Voxels::F64(array) => array[index],
        Voxels::U64(_) | Voxels::I64(_) => {
            return Err(Error::invalid(format!(
                "a gather was asked to read a {} level into an f64 column. That type holds \
                 integers beyond the ones an f64 represents exactly, so a large value would \
                 arrive as a nearby number nothing downstream could tell from a measurement. \
                 Narrow the level, or carry the value some other way.",
                pixels.dtype().numpy_name()
            )))
        }
    };
    if !value.is_finite() {
        return Err(Error::invalid(format!(
            "the level holds {value} at {at:?}, which is not finite. An unwritten level in this \
             crate is NaN so that an absence cannot pass for a value, and gathering one is that \
             absence reaching a consumer; it is refused here rather than carried."
        )));
    }
    Ok(value)
}

/// Every row of `table`, with the level's value at its own coordinate appended.
///
/// `within` is the region the rows are **required** to lie in and `origin` is
/// where `pixels[0, 0, 0]` sits in the volume. A row outside `within` is refused
/// naming the row and the region: see the module header — after a scale a
/// block's rows are no longer in that block, and a gather that read anyway would
/// return a real value from the wrong place.
///
/// The coordinate is used as it stands. There is no rounding here and there must
/// not be: a table coordinate is already an integer, and a rounding rule applied
/// twice is a rounding rule applied to the wrong thing.
pub fn gather_into(
    table: &Table,
    volume: [usize; 3],
    within: &Region,
    pixels: &Voxels,
    origin: [usize; 3],
    out: &mut RowBuilder,
) -> Result<()> {
    walk_rows(table, volume, out, |row| {
        let at = row.at();
        for axis in 0..3 {
            if at[axis] < within.start[axis] || at[axis] >= within.start[axis] + within.shape[axis]
            {
                return Err(Error::invalid(format!(
                    "a gather was handed a row at {at:?} and a region starting {:?} of shape \
                     {:?}, which does not hold it: it is outside on axis {axis}. A gather reads \
                     the level its own block holds, so a row somewhere else would be given a \
                     real value read at the wrong place. The usual cause is a scale between the \
                     rows and this phase, which moves rows out of the block that carries them; \
                     merge and re-scatter over the new volume rather than widening a reach, \
                     because a scaled row can move arbitrarily far.",
                    within.start, within.shape
                )));
            }
        }
        let mut values = values_of(row)?;
        let local = [at[0] - origin[0], at[1] - origin[1], at[2] - origin[2]];
        values.push(Value::F64(value_at(pixels, local)?));
        Ok(Some((at, values)))
    })
}

/// One blob in, one blob out: [`gather_into`] over the bytes a block holds.
pub fn gather_blob(
    volume: [usize; 3],
    schema: &Schema,
    blob: &[u8],
    column: &str,
    within: &Region,
    pixels: &Voxels,
    origin: [usize; 3],
) -> Result<Vec<u8>> {
    let table = sealed(volume, schema.clone(), blob)?;
    let mut out = RowBuilder::new(Arc::new(gathered_schema(schema, column)?));
    gather_into(&table, volume, within, pixels, origin, &mut out)?;
    Ok(out.encode())
}

// ------------------------------------------------------------- the filter --

/// One side of a test on a column, and whether it is strict.
///
/// Both spellings exist because both are in the consumers: a half-open
/// `[min, max)` range, and a strict threshold. On floats they are not one
/// substitution apart — `> t` is `>= next_after(t)` and nobody should have to
/// write that — so the strictness is a parameter rather than a convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Limit {
    /// `value >= limit`.
    AtLeast(f64),
    /// `value > limit`.
    Above(f64),
    /// `value <= limit`.
    AtMost(f64),
    /// `value < limit`.
    Below(f64),
}

impl Limit {
    /// Does `value` satisfy this bound?
    pub fn holds(self, value: f64) -> bool {
        match self {
            Limit::AtLeast(limit) => value >= limit,
            Limit::Above(limit) => value > limit,
            Limit::AtMost(limit) => value <= limit,
            Limit::Below(limit) => value < limit,
        }
    }

    fn limit(self) -> f64 {
        match self {
            Limit::AtLeast(limit)
            | Limit::Above(limit)
            | Limit::AtMost(limit)
            | Limit::Below(limit) => limit,
        }
    }

    fn describe(self) -> String {
        let symbol = match self {
            Limit::AtLeast(_) => ">=",
            Limit::Above(_) => ">",
            Limit::AtMost(_) => "<=",
            Limit::Below(_) => "<",
        };
        format!("{symbol} {}", self.limit())
    }
}

/// A conjunction of bounds on named columns.
///
/// Every bound must hold for a row to survive; an empty predicate keeps
/// everything, and [`RowFilter::new`] refuses it — see there for why.
///
/// A column is read as an `f64` whatever it is stored as, because a bound is a
/// number. A `u64` column compares exactly against a bound below `2^53` and is
/// the only place in this module where a width is quietly crossed; it is done
/// here rather than making a caller write two kinds of bound, and the exactness
/// limit is the same one [`value_at`] states.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnTest {
    column: String,
    lower: Option<Limit>,
    upper: Option<Limit>,
}

impl ColumnTest {
    /// Refuses a test with no bound at all: it keeps every row and its presence
    /// in a predicate reads as a constraint that is not there.
    pub fn new(
        column: impl Into<String>,
        lower: Option<Limit>,
        upper: Option<Limit>,
    ) -> Result<Self> {
        let column = column.into();
        if lower.is_none() && upper.is_none() {
            return Err(Error::invalid(format!(
                "the test on column {column:?} has neither a lower nor an upper bound, so it \
                 keeps every row. A predicate that cannot reject anything is indistinguishable \
                 from a missing one, and a caller who meant no test says so by not writing it."
            )));
        }
        Ok(Self {
            column,
            lower,
            upper,
        })
    }

    /// `[min, max)`: at least `min`, below `max`. The range convention the
    /// consumers use.
    pub fn range(column: impl Into<String>, min: f64, max: f64) -> Result<Self> {
        Self::new(column, Some(Limit::AtLeast(min)), Some(Limit::Below(max)))
    }

    /// `value > limit`, and nothing else.
    pub fn above(column: impl Into<String>, limit: f64) -> Result<Self> {
        Self::new(column, Some(Limit::Above(limit)), None)
    }

    /// `value >= limit`, and nothing else.
    pub fn at_least(column: impl Into<String>, limit: f64) -> Result<Self> {
        Self::new(column, Some(Limit::AtLeast(limit)), None)
    }

    pub fn column(&self) -> &str {
        &self.column
    }

    fn holds(&self, row: &Row<'_>, index: usize) -> Result<bool> {
        let value = match row.value(index)? {
            Value::U64(value) => value as f64,
            Value::F64(value) => value,
        };
        Ok(self.lower.map(|bound| bound.holds(value)).unwrap_or(true)
            && self.upper.map(|bound| bound.holds(value)).unwrap_or(true))
    }

    fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(bound) = self.lower {
            parts.push(bound.describe());
        }
        if let Some(bound) = self.upper {
            parts.push(bound.describe());
        }
        format!("{:?} {}", self.column, parts.join(" and "))
    }
}

/// Every test, all of which must hold.
#[derive(Debug, Clone, PartialEq)]
pub struct RowFilter {
    tests: Vec<ColumnTest>,
}

impl RowFilter {
    /// Refuses an empty predicate.
    ///
    /// A filter with no test keeps every row, which is an identity wearing the
    /// name of a filter — and, in a fixture, a filter that tests nothing while
    /// looking like it tests something. A caller who wants every row does not
    /// run this phase.
    pub fn new(tests: Vec<ColumnTest>) -> Result<Self> {
        if tests.is_empty() {
            return Err(Error::invalid(
                "a filter with no test keeps every row, which is an identity under another \
                 name. A caller who wants every row omits the phase."
                    .to_string(),
            ));
        }
        Ok(Self { tests })
    }

    pub fn tests(&self) -> &[ColumnTest] {
        &self.tests
    }

    /// The column index of each test against `schema`, or a refusal naming the
    /// column and what the rows do have.
    ///
    /// Resolved once per blob rather than once per row: the schema is the same
    /// for every row, and a name lookup per row per test is the whole cost of
    /// the filter for a wide predicate.
    pub fn resolve(&self, schema: &Schema) -> Result<Vec<usize>> {
        self.tests
            .iter()
            .map(|test| {
                schema.index_of(&test.column).ok_or_else(|| {
                    Error::invalid(format!(
                        "the filter tests a column named {:?} and these rows have no such \
                         column; they have {}. A missing column is refused rather than treated \
                         as a row that fails the test, because the two answers differ in every \
                         row and only one of them is a filter.",
                        test.column,
                        schema
                            .columns()
                            .iter()
                            .map(|column| format!("{:?}", column.name()))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })
            })
            .collect()
    }

    /// Does this row survive? `indexes` is [`Self::resolve`]'s answer.
    pub fn keeps(&self, row: &Row<'_>, indexes: &[usize]) -> Result<bool> {
        for (test, &index) in self.tests.iter().zip(indexes) {
            if !test.holds(row, index)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// What this filter tests, for a message.
    pub fn describe(&self) -> String {
        self.tests
            .iter()
            .map(ColumnTest::describe)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// The rows of `table` that satisfy `filter`, unchanged and in the same order.
///
/// The coordinates and the columns travel untouched — a filter selects, it does
/// not transform — so the output is a **subsequence** of the input. What it does
/// not preserve is a row's index; see the module header.
pub fn filter_into(
    table: &Table,
    volume: [usize; 3],
    filter: &RowFilter,
    out: &mut RowBuilder,
) -> Result<()> {
    let indexes = filter.resolve(table.schema())?;
    walk_rows(table, volume, out, |row| {
        if filter.keeps(row, &indexes)? {
            Ok(Some((row.at(), values_of(row)?)))
        } else {
            Ok(None)
        }
    })
}

/// One blob in, one blob out: [`filter_into`] over the bytes a block holds.
pub fn filter_blob(
    volume: [usize; 3],
    schema: &Schema,
    blob: &[u8],
    filter: &RowFilter,
) -> Result<Vec<u8>> {
    let table = sealed(volume, schema.clone(), blob)?;
    let mut out = RowBuilder::new(Arc::new(schema.clone()));
    filter_into(&table, volume, filter, &mut out)?;
    Ok(out.encode())
}

// ---------------------------------------------------------------- the ops --

/// What every op here shares: where the rows come from and where they go.
///
/// A struct rather than a copy of four fields per op, because the ops differ in
/// their *rule* and not in their plumbing, and a copy each is a place each for
/// the phase index to be wrong.
#[derive(Debug, Clone)]
pub struct RowStreams {
    /// The stream the rows arrive on.
    pub input: String,
    /// The phase that wrote them. Half the address: a stream written by two
    /// phases holds two generations, and a blob from the wrong one decodes
    /// perfectly and answers differently.
    pub phase: usize,
    /// The stream the rows leave on. Different from `input`, and refused
    /// otherwise: a phase that read and wrote one stream would have two
    /// generations under one key and no way to say which a reader wanted.
    pub output: String,
    pub lifecycle: Lifecycle,
    /// The schema the rows arriving have. Carried rather than read from the
    /// blob so that the *output* schema is known before any block runs, which
    /// is what a consumer planning against this phase needs.
    pub schema: Schema,
}

impl RowStreams {
    pub fn new(
        input: impl Into<String>,
        phase: usize,
        output: impl Into<String>,
        lifecycle: Lifecycle,
        schema: Schema,
    ) -> Result<Self> {
        let input = input.into();
        let output = output.into();
        if input == output {
            return Err(Error::invalid(format!(
                "a row op was asked to read and write the stream {input:?}. One stream would \
                 then hold this phase's rows and the previous phase's under keys that differ \
                 only in a phase index nobody downstream carries, so a reader could not say \
                 which generation it had."
            )));
        }
        Ok(Self {
            input,
            phase,
            output,
            lifecycle,
            schema,
        })
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        // Reach `[0, 0, 0]`: this block's fragment and no neighbour's. See the
        // module header — a row op reads one row to write one row, and an
        // overlap here duplicates rows rather than costing recomputation.
        vec![FragmentInput::own(self.input.clone(), self.phase)]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.output.clone(),
            self.lifecycle,
            // Every block, always. These phases write no level, so the tiling
            // check has nothing to bite on and this declaration is the only
            // guard there is; a range whose every row was filtered out writes a
            // header and no rows, which is present and therefore checkable.
            Coverage::EveryBlock,
        )]
    }

    /// This block's blob, or the empty one.
    ///
    /// Absent is treated as empty rather than refused. The upstream phases in
    /// this crate declare [`Coverage::EveryBlock`], so absent should not happen
    /// and the coverage guard is what says so — checking it a second time here
    /// would report it as this op's fault, in a message about a stream this op
    /// does not write.
    fn own<'a>(&self, at: &'a BlockView<'a>) -> &'a [u8] {
        at.own(&self.input).unwrap_or(&[])
    }
}

/// **Rows in, the same rows at scaled coordinates out.**
///
/// Reads no pixels, writes no level, reach 0. The rows it emits are in the
/// **scaled** volume, which is not the volume this phase is anchored in — see
/// [`scaled_bound`], and see the module header for why a gather must not follow
/// this in the same lattice.
pub struct ScaleRowsOp {
    name: &'static str,
    rows: RowStreams,
    factor: [f64; 3],
}

impl ScaleRowsOp {
    /// Refuses a factor that is not finite and non-negative, at construction
    /// rather than once per row.
    pub fn new(name: &'static str, rows: RowStreams, factor: [f64; 3]) -> Result<Self> {
        for axis in 0..3 {
            // Validated by asking the one function that defines the rule, so
            // there is no second statement of what a usable factor is.
            scaled_index(0, factor[axis])?;
        }
        Ok(Self { name, rows, factor })
    }

    pub fn factor(&self) -> [f64; 3] {
        self.factor
    }

    /// The volume the rows this op emits live in, given the one it reads.
    pub fn volume(&self, input: [usize; 3]) -> Result<[usize; 3]> {
        scaled_bound(input, self.factor)
    }

    /// The schema of the rows this op emits: its input's, unchanged.
    pub fn schema(&self) -> &Schema {
        &self.rows.schema
    }
}

impl FragmentOp for ScaleRowsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        self.rows.inputs()
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        self.rows.outputs()
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        Ok(BlockOutput::fragment(
            self.rows.output.clone(),
            scale_blob(
                at.volume(),
                &self.rows.schema,
                self.rows.own(at),
                self.factor,
            )?,
        ))
    }
}

/// **Rows in, the same rows with one more column out** — the value a stored
/// level holds at each row's own coordinate.
///
/// Reads no pixels, writes no level, reach 0. `reads_pixels` is about the level
/// the phase is *handed* and a gather never wants that one; the level it samples
/// is declared as a [`SourceInput`], fetched at the block's own fetch region and
/// recorded on the phase, so this op pays for one array rather than two and the
/// plan says which. See the module header for why that declaration is what makes
/// the shell possible at all, and for the [`SeamFold::PerBlock`] claim.
///
/// The level may be any at or below the phase's own — level 0 included — so
/// *rows from one array, values from a second* is the ordinary arrangement.
pub struct GatherRowsOp {
    name: &'static str,
    rows: RowStreams,
    level: usize,
    column: String,
    /// The input's schema plus the gathered column.
    ///
    /// Computed once at construction rather than per block: it is what a
    /// consumer planning against this phase needs *before* any block has run,
    /// and it cannot change between blocks — a schema rebuilt per fragment
    /// would be one more place for two blocks to disagree.
    schema: Schema,
}

impl GatherRowsOp {
    /// Refuses a column name the input rows already carry, at construction
    /// rather than on the first block: the answer is a function of the schema
    /// and the name alone, so a run that would fail every block fails before it
    /// starts.
    pub fn new(
        name: &'static str,
        rows: RowStreams,
        level: usize,
        column: impl Into<String>,
    ) -> Result<Self> {
        let column = column.into();
        let schema = gathered_schema(&rows.schema, &column)?;
        Ok(Self {
            name,
            rows,
            level,
            column,
            schema,
        })
    }

    /// The level this op samples.
    pub fn level(&self) -> usize {
        self.level
    }

    /// The column it appends.
    pub fn column(&self) -> &str {
        &self.column
    }

    /// The schema of the rows this op emits: its input's, plus one `f64` column.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// One block's rows, gathered against the block of `level` covering `read`,
    /// with the rows required to lie in `core`.
    ///
    /// A free function in all but name, and separated from the [`FragmentOp`]
    /// shell for `ops::tabulate`'s reason: it is written over the narrowest
    /// things it needs — a blob, a buffer and two regions — so it can be driven
    /// from a test without a plan, an environment or a lattice, and so the shell
    /// below is plumbing with nothing to get wrong.
    pub fn gather_block(
        &self,
        volume: [usize; 3],
        blob: &[u8],
        level: &BlockBuf,
        read: &Region,
        core: &Region,
    ) -> Result<Vec<u8>> {
        let BlockBuf::Array(pixels) = level else {
            // A simulated run holds no array, so there is no value to read.
            // The rows are **not** emitted with an invented one: a gathered
            // column is a measurement, and a fabricated measurement is the
            // plausible wrong answer this module refuses everywhere else. A
            // header and no rows is *present and empty*, which is the fact the
            // coverage guard checks and is a different fact from absent.
            return Ok(RowBuilder::new(Arc::new(self.schema.clone())).encode());
        };
        let shape = [read.shape[0], read.shape[1], read.shape[2]];
        if pixels.shape() != shape {
            return Err(Error::invalid(format!(
                "a gather was handed level {} as {:?} for a block read extent of {shape:?}. A \
                 source level is fetched at the block's own fetch region, so a disagreement here \
                 is the plan handing over two geometries rather than a row in the wrong place.",
                self.level,
                pixels.shape()
            )));
        }
        gather_blob(
            volume,
            &self.rows.schema,
            blob,
            &self.column,
            core,
            pixels,
            [read.start[0], read.start[1], read.start[2]],
        )
    }
}

impl FragmentOp for GatherRowsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        self.rows.inputs()
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        self.rows.outputs()
    }

    /// The level sampled, at exactly the extent the block owns. A gather reads
    /// one voxel per row and every row is inside this block's core, so there is
    /// nothing outside the block's own fetch to read; a reach here would widen
    /// the phase halo and buy nothing.
    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.level)]
    }

    /// See the module header: a row is read by one block and its value comes
    /// from one voxel, so nothing crosses a seam and there is no order to depend
    /// on. The framework checks this against the fragment reach, which is
    /// `[0, 0, 0]`.
    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "a gather reads a level at each row's coordinate and cannot be computed from the \
             rows alone. It is applied through `apply_with`.",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        Ok(BlockOutput::fragment(
            self.rows.output.clone(),
            self.gather_block(
                at.volume(),
                self.rows.own(at),
                sources.get(self.level)?,
                at.read,
                at.core,
            )?,
        ))
    }
}

/// **Rows in, the rows that satisfy the predicate out.**
///
/// Reads no pixels, writes no level, reach 0. Coordinates and columns travel
/// untouched, so the output is a subsequence of the input — and the surviving
/// rows are renumbered, because a row's only name is its position. The module
/// header says what follows from that.
pub struct FilterRowsOp {
    name: &'static str,
    rows: RowStreams,
    filter: RowFilter,
}

impl FilterRowsOp {
    /// Refuses a predicate naming a column the input rows do not have, at
    /// construction rather than on the first block.
    pub fn new(name: &'static str, rows: RowStreams, filter: RowFilter) -> Result<Self> {
        filter.resolve(&rows.schema)?;
        Ok(Self { name, rows, filter })
    }

    /// The schema of the rows this op emits: its input's, unchanged.
    pub fn schema(&self) -> &Schema {
        &self.rows.schema
    }

    pub fn filter(&self) -> &RowFilter {
        &self.filter
    }
}

impl FragmentOp for FilterRowsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        self.rows.inputs()
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        self.rows.outputs()
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        Ok(BlockOutput::fragment(
            self.rows.output.clone(),
            filter_blob(
                at.volume(),
                &self.rows.schema,
                self.rows.own(at),
                &self.filter,
            )?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array3;

    const VOLUME: [usize; 3] = [8, 8, 8];

    fn schema() -> Schema {
        Schema::new(vec![Column::u64("key"), Column::f64("score")])
            .expect("two differently named columns")
    }

    fn row(at: [usize; 3], key: u64, score: f64) -> ([usize; 3], Vec<Value>) {
        (at, vec![Value::U64(key), Value::F64(score)])
    }

    /// One blob holding `rows`, in the order given — which is deliberately not
    /// the canonical order in most of these fixtures.
    fn blob(rows: &[([usize; 3], Vec<Value>)]) -> Vec<u8> {
        let mut builder = RowBuilder::new(Arc::new(schema()));
        for (at, values) in rows {
            builder
                .push(*at, values)
                .expect("the fixture matches the schema");
        }
        builder.encode()
    }

    fn merged(volume: [usize; 3], schema: Schema, blobs: &[Vec<u8>]) -> Vec<RowValues> {
        merge_rows(
            volume,
            schema,
            blobs
                .iter()
                .enumerate()
                .map(|(index, bytes)| ([index, 0, 0], bytes.as_slice())),
        )
        .expect("the fixture merges")
    }

    /// Every way of cutting `rows` into contiguous ranges, including the
    /// one-range and one-row-per-range extremes.
    ///
    /// Enumerated rather than sampled: there are `2^(n-1)` of them and `n` is
    /// small, so "every way" is affordable and is a stronger claim than any
    /// number of chosen cuts.
    fn every_split(rows: &[([usize; 3], Vec<Value>)]) -> Vec<Vec<Vec<u8>>> {
        let cuts = rows.len().saturating_sub(1);
        (0..(1usize << cuts))
            .map(|mask| {
                let mut ranges = Vec::new();
                let mut start = 0;
                for cut in 0..cuts {
                    if mask & (1 << cut) != 0 {
                        ranges.push(blob(&rows[start..=cut]));
                        start = cut + 1;
                    }
                }
                ranges.push(blob(&rows[start..]));
                ranges
            })
            .collect()
    }

    // ------------------------------------------------ decomposition ------

    /// **The acceptance property.** The same rows out however the rows were
    /// split into ranges, compared as a whole ordered list rather than as a set
    /// — a set comparison cannot see a permutation, which is the failure this is
    /// arranged against.
    ///
    /// The fixture contains a **run of three rows sharing one coordinate**, with
    /// distinct payloads, and `every_split` puts a cut at every position
    /// including inside that run. Those three tiebreak on their payload words,
    /// which travel with them, so the run reassembles the same way whichever
    /// side of it the cut fell. A mechanism that gave a range a base index would
    /// have to get this right; this one has nothing to get wrong.
    #[test]
    fn merging_is_insensitive_to_how_the_rows_were_split() {
        let rows = vec![
            row([5, 1, 1], 10, 1.0),
            row([2, 2, 2], 20, 2.0),
            row([2, 2, 2], 21, 3.0),
            row([2, 2, 2], 22, 4.0),
            row([0, 7, 3], 30, 5.0),
            row([1, 0, 0], 40, 6.0),
        ];
        let whole = merged(VOLUME, schema(), &[blob(&rows)]);
        assert_eq!(whole.len(), rows.len());
        // The run really is a run: three rows at one coordinate, adjacent in
        // the answer. Asserted so that a fixture edit that broke the case the
        // test exists for fails here rather than passing quietly.
        assert_eq!(whole.iter().filter(|row| row.at == [2, 2, 2]).count(), 3);

        for (index, split) in every_split(&rows).into_iter().enumerate() {
            assert_eq!(
                merged(VOLUME, schema(), &split),
                whole,
                "split {index} into {} range(s) gave a different list",
                split.len()
            );
        }
    }

    /// The same list whichever order the blobs arrive in — the property a
    /// completion order would break.
    #[test]
    fn merging_is_insensitive_to_the_order_blobs_arrive() {
        let rows = vec![
            row([5, 1, 1], 10, 1.0),
            row([2, 2, 2], 20, 2.0),
            row([0, 7, 3], 30, 5.0),
        ];
        let mut blobs: Vec<Vec<u8>> = rows
            .iter()
            .map(|row| blob(std::slice::from_ref(row)))
            .collect();
        let forwards = merged(VOLUME, schema(), &blobs);
        blobs.reverse();
        assert_eq!(merged(VOLUME, schema(), &blobs), forwards);
    }

    // ------------------------------------------------------ rounding ------

    /// **The rounding rule, stated against a written-out table of both rules.**
    ///
    /// What this catches: a scale written with `f64::round`. The two rules agree
    /// everywhere except at an exact `.5` whose floor is **odd**, so the fixture
    /// carries ties of both parities — `0.5`, `2.5` and `4.5`, where they
    /// differ, and `1.5` and `3.5`, where they agree. A fixture of random floats
    /// never produces a tie at all and would certify either rule; a fixture with
    /// only even-floored ties would certify either rule too.
    ///
    /// The expected column is written out rather than computed, so that this
    /// test is a statement of the rule and not a copy of the implementation.
    #[test]
    fn the_rounding_rule_is_ties_to_even() {
        // coordinate, halved: exact tie, ties-to-even, f64::round
        let cases = [
            (0usize, 0.0f64, 0usize, 0usize),
            (1, 0.5, 0, 1),
            (2, 1.0, 1, 1),
            (3, 1.5, 2, 2),
            (4, 2.0, 2, 2),
            (5, 2.5, 2, 3),
            (6, 3.0, 3, 3),
            (7, 3.5, 4, 4),
            (9, 4.5, 4, 5),
        ];
        let mut differed = 0;
        let mut agreed = 0;
        for (coordinate, exact, even, away) in cases {
            assert_eq!(
                coordinate as f64 * 0.5,
                exact,
                "the fixture's own arithmetic must be exact, or it is testing nothing"
            );
            assert_eq!(
                scaled_index(coordinate, 0.5).expect("a finite factor"),
                even,
                "{coordinate} * 0.5 is {exact}, which is {even} under ties-to-even"
            );
            if even == away {
                agreed += 1;
            } else {
                differed += 1;
                assert_eq!(
                    exact.round() as usize,
                    away,
                    "the fixture's claim about f64::round must be right for the case to \
                     discriminate"
                );
            }
        }
        // The fixture must contain both kinds or it proves nothing: cases where
        // the two rules differ are what makes it a test of the rule, cases where
        // they agree are what stops it over-claiming.
        assert_eq!(differed, 3, "ties at 0.5, 2.5 and 4.5 discriminate");
        assert!(agreed >= 2, "ties at 1.5 and 3.5 do not");
    }

    /// The rule reaches the op, not just the helper: a whole table scaled by a
    /// half lands where ties-to-even says and not where `f64::round` does.
    #[test]
    fn a_scaled_table_rounds_ties_to_even() {
        let rows = vec![
            row([1, 1, 1], 1, 1.0),
            row([5, 5, 5], 2, 2.0),
            row([3, 3, 3], 3, 3.0),
        ];
        let scaled =
            scale_blob(VOLUME, &schema(), &blob(&rows), [0.5, 0.5, 0.5]).expect("a finite factor");
        let out = merged(
            scaled_bound(VOLUME, [0.5; 3]).expect("a finite factor"),
            schema(),
            &[scaled],
        );
        assert_eq!(
            out.iter().map(|row| row.at).collect::<Vec<_>>(),
            // 1 -> 0.5 -> 0, 3 -> 1.5 -> 2, 5 -> 2.5 -> 2. Under `f64::round`
            // they would be 1, 2 and 3, and the first and last rows would be
            // somewhere else.
            vec![[0, 0, 0], [2, 2, 2], [2, 2, 2]]
        );
        // Two distinct rows landed on one coordinate, which a scale may do and a
        // filter may not. They are still two rows.
        assert_eq!(out.len(), 3);
    }

    /// The scaled volume is derived from the largest coordinate, so it can never
    /// refuse a row the scale produced.
    #[test]
    fn the_scaled_bound_holds_every_scaled_row() {
        for factor in [0.5f64, 1.0, 2.0, 0.3333333333333333] {
            let bound = scaled_bound(VOLUME, [factor; 3]).expect("a finite factor");
            for coordinate in 0..VOLUME[0] {
                assert!(
                    scaled_index(coordinate, factor).expect("a finite factor") < bound[0],
                    "{coordinate} scaled by {factor} is outside the bound {}",
                    bound[0]
                );
            }
        }
    }

    #[test]
    fn a_negative_or_infinite_factor_is_refused() {
        assert!(scaled_index(1, -1.0).is_err());
        assert!(scaled_index(1, f64::NAN).is_err());
        assert!(scaled_index(1, f64::INFINITY).is_err());
    }

    // -------------------------------------------------------- filter ------

    fn scores() -> Vec<([usize; 3], Vec<Value>)> {
        vec![
            row([0, 0, 0], 0, 1.0),
            row([1, 1, 1], 1, 5.0),
            row([2, 2, 2], 2, 2.0),
            row([3, 3, 3], 3, 9.0),
            row([4, 4, 4], 4, 5.0),
        ]
    }

    /// The survivors come back in the order they went in, and the filter is a
    /// **subsequence** of the input — asserted against the input restricted,
    /// which is a claim about order rather than about membership.
    ///
    /// The predicate rejects rows at both ends of the list and in the middle, so
    /// a filter that reversed, sorted differently or dropped the wrong rows
    /// fails. A predicate that kept everything would test nothing at all.
    #[test]
    fn filtering_preserves_order_and_keeps_a_subsequence() {
        let rows = scores();
        let whole = merged(VOLUME, schema(), &[blob(&rows)]);
        let filter = RowFilter::new(vec![
            ColumnTest::range("score", 2.0, 9.0).expect("two bounds")
        ])
        .expect("one test");
        let kept = merged(
            VOLUME,
            schema(),
            &[filter_blob(VOLUME, &schema(), &blob(&rows), &filter).expect("the column exists")],
        );

        // `[2.0, 9.0)`: 1.0 is below, 9.0 is not below 9.0. Three survive, so
        // the predicate rejects at both ends and the test is not vacuous.
        assert_eq!(kept.len(), 3);
        let expected: Vec<RowValues> = whole
            .iter()
            .filter(|row| {
                let Value::F64(score) = row.values[1] else {
                    unreachable!("the schema says column 1 is an f64")
                };
                (2.0..9.0).contains(&score)
            })
            .cloned()
            .collect();
        assert_eq!(kept, expected);
    }

    /// **Filtering renumbers, and that is stated behaviour.** A row's only name is
    /// its position, so removing rows before it renames it.
    ///
    /// Asserted in the discriminating direction: the surviving row's index
    /// *moved*. A test that only checked the row was still there would pass
    /// under a numbering scheme that kept the input index, which is the other
    /// possible behaviour and is not this one.
    #[test]
    fn filtering_renumbers_the_survivors() {
        let rows = scores();
        let whole = merged(VOLUME, schema(), &[blob(&rows)]);
        let filter = RowFilter::new(vec![ColumnTest::above("score", 4.0).expect("one bound")])
            .expect("one test");
        let kept = merged(
            VOLUME,
            schema(),
            &[filter_blob(VOLUME, &schema(), &blob(&rows), &filter).expect("the column exists")],
        );

        let survivor = &kept[0];
        let was = whole
            .iter()
            .position(|row| row == survivor)
            .expect("it survived");
        assert_eq!(
            was, 1,
            "it was second in, because the row scoring 1.0 preceded it"
        );
        assert_eq!(kept.iter().position(|row| row == survivor), Some(0));
        assert_ne!(
            was, 0,
            "so its index moved, which is what renumbering means"
        );
    }

    /// A strict bound and a closed one are different predicates, and the
    /// difference shows on a row sitting exactly on the limit.
    #[test]
    fn a_strict_bound_and_a_closed_one_differ_on_the_limit() {
        let rows = scores();
        let on_the_limit = |bound: Limit| {
            let filter = RowFilter::new(vec![
                ColumnTest::new("score", Some(bound), None).expect("a bound")
            ])
            .expect("one test");
            merged(
                VOLUME,
                schema(),
                &[filter_blob(VOLUME, &schema(), &blob(&rows), &filter).expect("the column")],
            )
            .len()
        };
        // Two rows score exactly 5.0.
        assert_eq!(on_the_limit(Limit::AtLeast(5.0)), 3);
        assert_eq!(on_the_limit(Limit::Above(5.0)), 1);
    }

    #[test]
    fn an_empty_predicate_and_an_empty_test_are_refused() {
        assert!(RowFilter::new(Vec::new()).is_err());
        assert!(ColumnTest::new("score", None, None).is_err());
    }

    #[test]
    fn a_filter_naming_a_column_the_rows_do_not_have_is_refused() {
        let filter = RowFilter::new(vec![ColumnTest::above("missing", 0.0).expect("a bound")])
            .expect("one test");
        assert!(filter.resolve(&schema()).is_err());
    }

    // -------------------------------------------------------- gather ------

    fn ramp() -> Voxels {
        Voxels::F64(Array3::from_shape_fn(
            (VOLUME[0], VOLUME[1], VOLUME[2]),
            |(i, j, k)| (i * 100 + j * 10 + k) as f64,
        ))
    }

    /// The gathered column is the level's value at the row's own coordinate, and
    /// the schema is the input's plus one.
    #[test]
    fn a_gather_reads_the_level_at_the_rows_coordinate() {
        let rows = vec![row([1, 2, 3], 0, 0.0), row([7, 0, 0], 1, 0.0)];
        let out = gather_blob(
            VOLUME,
            &schema(),
            &blob(&rows),
            "level",
            &Region::whole(&VOLUME),
            &ramp(),
            [0, 0, 0],
        )
        .expect("the rows are inside");
        let gathered = gathered_schema(&schema(), "level").expect("a fresh name");
        assert_eq!(gathered.len(), 3);
        let merged = merged(VOLUME, gathered, &[out]);
        assert_eq!(
            merged
                .iter()
                .map(|row| (row.at, row.values[2]))
                .collect::<Vec<_>>(),
            vec![
                ([1, 2, 3], Value::F64(123.0)),
                ([7, 0, 0], Value::F64(700.0))
            ]
        );
    }

    /// **The block boundary.** Cores are half-open and tile with no overlap, so
    /// a coordinate on a boundary is in exactly one block — the one starting
    /// there — and the value it reads is the value the whole-volume gather
    /// reads, because it is the same voxel.
    ///
    /// Driven directly against two adjacent regions rather than through a
    /// lattice, so that what is asserted is the rule and not the executor.
    #[test]
    fn a_row_on_a_block_boundary_belongs_to_exactly_one_side() {
        let low = Region::new(&[0, 0, 0], &[4, VOLUME[1], VOLUME[2]]);
        let high = Region::new(&[4, 0, 0], &[4, VOLUME[1], VOLUME[2]]);
        // 3 is the last coordinate of the low core, 4 is the first of the high.
        let boundary = vec![row([3, 1, 1], 0, 0.0), row([4, 1, 1], 1, 0.0)];
        let pixels = ramp();

        for (region, mine, theirs) in [(&low, 0, 1), (&high, 1, 0)] {
            let ours = blob(std::slice::from_ref(&boundary[mine]));
            let out = gather_blob(
                VOLUME,
                &schema(),
                &ours,
                "level",
                region,
                &pixels,
                [0, 0, 0],
            )
            .expect("its own row is inside its own region");
            let gathered = gathered_schema(&schema(), "level").expect("a fresh name");
            assert_eq!(merged(VOLUME, gathered, &[out]).len(), 1);

            // And the other side's row is refused rather than answered, which is
            // what "exactly one" means: there is no reading of it that produces
            // two values for one row.
            let not_ours = blob(std::slice::from_ref(&boundary[theirs]));
            assert!(gather_blob(
                VOLUME,
                &schema(),
                &not_ours,
                "level",
                region,
                &pixels,
                [0, 0, 0]
            )
            .is_err());
        }
    }

    /// A row outside the region the block covers is refused, naming it — the
    /// precondition reach 0 rests on, and the composition it forbids.
    #[test]
    fn a_row_outside_the_block_is_refused_rather_than_read() {
        let core = Region::new(&[0, 0, 0], &[4, VOLUME[1], VOLUME[2]]);
        let rows = vec![row([6, 0, 0], 0, 0.0)];
        let err = gather_blob(
            VOLUME,
            &schema(),
            &blob(&rows),
            "level",
            &core,
            &ramp(),
            [0, 0, 0],
        )
        .expect_err("the row is outside the core");
        let text = format!("{err}");
        assert!(
            text.contains("[6, 0, 0]"),
            "the message names the row: {text}"
        );
    }

    /// A coordinate outside the *volume* cannot reach a gather at all: the store
    /// refuses it, so the case is unrepresentable rather than handled.
    #[test]
    fn a_row_outside_the_volume_cannot_be_in_the_table() {
        let mut table = Table::new(VOLUME, schema()).expect("a volume and a schema");
        let outside = blob(&[row([VOLUME[0], 0, 0], 0, 0.0)]);
        assert!(table.write([0, 0, 0], &outside).is_err());
    }

    /// Two levels an `f64` column cannot carry exactly are refused rather than
    /// rounded; every narrower one converts with no loss.
    #[test]
    fn a_level_too_wide_for_an_f64_column_is_refused() {
        let one = [1usize, 1, 1];
        assert!(value_at(&Voxels::U64(Array3::zeros((2, 2, 2))), one).is_err());
        assert!(value_at(&Voxels::I64(Array3::zeros((2, 2, 2))), one).is_err());
        assert_eq!(
            value_at(&Voxels::I32(Array3::from_elem((2, 2, 2), -7)), one).expect("exact"),
            -7.0
        );
        assert_eq!(
            value_at(&Voxels::Bool(Array3::from_elem((2, 2, 2), true)), one).expect("exact"),
            1.0
        );
    }

    /// An unwritten level is NaN so that an absence cannot pass for a value;
    /// gathering one is refused where the coordinate can still be named.
    #[test]
    fn gathering_a_non_finite_value_is_refused() {
        let unwritten = Voxels::F64(Array3::from_elem((2, 2, 2), f64::NAN));
        assert!(value_at(&unwritten, [0, 0, 0]).is_err());
    }

    /// A gather may not overwrite a column that is already there.
    #[test]
    fn a_gather_onto_an_existing_column_is_refused() {
        assert!(gathered_schema(&schema(), "score").is_err());
        assert!(gathered_schema(&schema(), "level").is_ok());
    }

    // ------------------------------------------------- the op shells ------

    #[test]
    fn a_row_op_reading_and_writing_one_stream_is_refused() {
        assert!(RowStreams::new("rows", 0, "rows", Lifecycle::DeleteOnExit, schema()).is_err());
        assert!(RowStreams::new("in", 0, "out", Lifecycle::DeleteOnExit, schema()).is_ok());
    }

    /// Every op here declares reach 0 on every axis, and that is the module's
    /// no-overlap claim in the form the geometry reads.
    #[test]
    fn every_row_op_reaches_nothing() {
        let rows = || {
            RowStreams::new("in", 0, "out", Lifecycle::DeleteOnExit, schema()).expect("two names")
        };
        let scale = ScaleRowsOp::new("scale", rows(), [0.5; 3]).expect("a finite factor");
        let filter = FilterRowsOp::new(
            "filter",
            rows(),
            RowFilter::new(vec![ColumnTest::above("score", 0.0).expect("a bound")])
                .expect("one test"),
        )
        .expect("the column exists");
        let gather = GatherRowsOp::new("gather", rows(), 0, "level").expect("a fresh name");
        let ops: [&dyn FragmentOp; 3] = [&scale, &filter, &gather];
        for op in ops {
            for axis in 0..3 {
                assert_eq!(op.reach(axis, VOLUME[axis]), 0, "{}", op.name());
            }
            assert_eq!(op.inputs()[0].reach, [0, 0, 0], "{}", op.name());
            assert!(!op.writes_pixels(), "{}", op.name());
            // **None of the three reads the level its phase is handed**, the
            // gather included: the array a gather samples is a *second* one, and
            // it says so with `source_inputs` rather than by taking level `p`.
            // See the module header — that is the whole reason the shell can be
            // written, and it costs one array rather than two.
            assert!(!op.reads_pixels(), "{}", op.name());
        }
        assert_eq!(scale.source_inputs(VOLUME), Vec::<SourceInput>::new());
        assert_eq!(filter.source_inputs(VOLUME), Vec::<SourceInput>::new());
        assert_eq!(
            gather.source_inputs(VOLUME),
            vec![SourceInput::voxelwise(0)],
            "the gather names its level, at exactly the extent the block owns"
        );
    }

    /// **The seam claim, and that it is the checked one.**
    ///
    /// `PerBlock` beside a non-zero fragment reach is refused by the framework,
    /// so the declaration is tied to the reach rather than merely written down.
    /// `Unordered` would also be true here and would be *unchecked*, because the
    /// executor skips its reversal when the neighbourhood holds one fragment —
    /// which is exactly this op's neighbourhood.
    #[test]
    fn a_gather_claims_per_block_and_reaches_one_fragment() {
        let rows =
            RowStreams::new("in", 0, "out", Lifecycle::DeleteOnExit, schema()).expect("two names");
        let gather = GatherRowsOp::new("gather", rows, 0, "level").expect("a fresh name");
        assert_eq!(gather.seam_fold(), Some(SeamFold::PerBlock));
        assert_eq!(gather.inputs().len(), 1);
        assert_eq!(gather.inputs()[0].reach, [0, 0, 0]);
        assert_eq!(gather.outputs()[0].coverage, Coverage::EveryBlock);
        // The output schema is known before any block runs, which is what a
        // consumer planning against this phase needs.
        assert_eq!(gather.schema().len(), schema().len() + 1);
        assert_eq!(gather.schema().index_of("level"), Some(2));
        assert_eq!(gather.level(), 0);
        assert_eq!(gather.column(), "level");
    }

    #[test]
    fn a_gather_onto_an_existing_column_is_refused_at_construction() {
        let rows = || {
            RowStreams::new("in", 0, "out", Lifecycle::DeleteOnExit, schema()).expect("two names")
        };
        assert!(GatherRowsOp::new("gather", rows(), 0, "score").is_err());
        assert!(GatherRowsOp::new("gather", rows(), 0, "level").is_ok());
    }

    /// The shell's own arithmetic — the offset between a block's buffer and the
    /// volume — driven with no plan, no environment and no lattice.
    ///
    /// The block is the **second** half of the volume, so a shell that ignored
    /// `read.start` would read `[1, 2, 3]` of the buffer for the row at
    /// `[5, 2, 3]` and return a plausible number from the wrong voxel. The ramp
    /// is injective, so it returns the wrong *number*.
    #[test]
    fn a_gather_block_offsets_the_buffer_against_the_volume() {
        let core = Region::new(&[4, 0, 0], &[4, VOLUME[1], VOLUME[2]]);
        let block = Voxels::F64(Array3::from_shape_fn(
            (core.shape[0], core.shape[1], core.shape[2]),
            |(i, j, k)| ((core.start[0] + i) * 100 + j * 10 + k) as f64,
        ));
        let gather = GatherRowsOp::new(
            "gather",
            RowStreams::new("in", 0, "out", Lifecycle::DeleteOnExit, schema()).expect("two names"),
            0,
            "level",
        )
        .expect("a fresh name");

        let rows = vec![row([5, 2, 3], 0, 0.0), row([7, 7, 7], 1, 0.0)];
        let out = gather
            .gather_block(
                VOLUME,
                &blob(&rows),
                &BlockBuf::Array(block.clone()),
                &core,
                &core,
            )
            .expect("both rows are in the core");
        assert_eq!(
            merged(VOLUME, gather.schema().clone(), &[out])
                .iter()
                .map(|row| (row.at, row.values[2]))
                .collect::<Vec<_>>(),
            vec![
                ([5, 2, 3], Value::F64(523.0)),
                ([7, 7, 7], Value::F64(777.0))
            ]
        );

        // A row belonging to the other block is refused rather than read at an
        // offset that happens to be inside the buffer.
        assert!(gather
            .gather_block(
                VOLUME,
                &blob(&[row([1, 2, 3], 0, 0.0)]),
                &BlockBuf::Array(block),
                &core,
                &core,
            )
            .is_err());
    }

    /// A buffer that is not the block's read extent is the plan handing over two
    /// geometries, and it is named as that rather than discovered as a row that
    /// fell outside a buffer.
    #[test]
    fn a_gather_block_refuses_a_buffer_of_the_wrong_extent() {
        let core = Region::new(&[0, 0, 0], &[4, VOLUME[1], VOLUME[2]]);
        let gather = GatherRowsOp::new(
            "gather",
            RowStreams::new("in", 0, "out", Lifecycle::DeleteOnExit, schema()).expect("two names"),
            0,
            "level",
        )
        .expect("a fresh name");
        let error = gather
            .gather_block(
                VOLUME,
                &blob(&[row([1, 1, 1], 0, 0.0)]),
                &BlockBuf::Array(ramp()),
                &core,
                &core,
            )
            .expect_err("the whole volume is not this block's read extent");
        assert!(format!("{error}").contains("fetch region"), "{error}");
    }

    /// **A simulated run gathers nothing rather than inventing a value.** It
    /// holds no array, and a fabricated measurement is exactly the plausible
    /// wrong answer this module refuses everywhere else. The fragment is still
    /// written: present and empty is a different fact from absent, and it is the
    /// one the coverage guard checks.
    #[test]
    fn a_simulated_block_gathers_no_rows_rather_than_inventing_values() {
        let core = Region::new(&[0, 0, 0], &[4, VOLUME[1], VOLUME[2]]);
        let gather = GatherRowsOp::new(
            "gather",
            RowStreams::new("in", 0, "out", Lifecycle::DeleteOnExit, schema()).expect("two names"),
            0,
            "level",
        )
        .expect("a fresh name");
        let out = gather
            .gather_block(
                VOLUME,
                &blob(&[row([1, 1, 1], 0, 0.0)]),
                &BlockBuf::Accounted {
                    region: core.clone(),
                    dtype: crate::dtype::Dtype::F64,
                    uniform: None,
                },
                &core,
                &core,
            )
            .expect("a simulated block still writes a fragment");
        assert!(!out.is_empty(), "the fragment is present");
        assert!(merged(VOLUME, gather.schema().clone(), &[out]).is_empty());
    }

    #[test]
    fn an_op_refuses_a_bad_parameter_at_construction() {
        let rows = || {
            RowStreams::new("in", 0, "out", Lifecycle::DeleteOnExit, schema()).expect("two names")
        };
        assert!(ScaleRowsOp::new("scale", rows(), [-1.0, 1.0, 1.0]).is_err());
        assert!(FilterRowsOp::new(
            "filter",
            rows(),
            RowFilter::new(vec![ColumnTest::above("missing", 0.0).expect("a bound")])
                .expect("one test"),
        )
        .is_err());
    }
}
