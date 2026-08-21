// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definitions of the operations
// — a row in, a row out — not adapted from any implementation of them.
//
// **A table of rows in, a table of rows out.** The operations here are the ones
// that transform a `crate::table::Table` rather than a volume:
//
// | op | what one row becomes | reads an image | its `FragmentOp` shell |
// |---|---|---|---|
// | scale | the same row at a scaled coordinate | no | [`ScaleRowsOp`] |
// | gather | the same row with one more column, read at the row's own coordinate | **yes — a declared second array** | [`GatherRowsOp`] |
// | filter | itself, or nothing | no | [`FilterRowsOp`] |
// | **group** | **part of one row of a smaller table** | no | [`GroupRowsOp`] + [`MergeGroupsOp`] |
//
// The last one is the odd member and the rest of this header is written about
// the first three; see *"The reduction, which is the one that is not a map"*
// below for what it does not share with them and why it is here rather than in
// a module of its own.
//
// One kernel, three ops, and why it is not one op
// -----------------------------------------------
// Mechanically all three of the *maps* are the same loop: walk the rows, emit
// zero or one row per row read. A single op parameterised by a
// `Fn(Row) -> Option<Row>` would express every one of them, and it was the first
// thing tried. It is rejected here, because what the three do *not* share is the
// part a caller has to reason about:
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
// The reduction, which is the one that is not a map
// -------------------------------------------------
// [`GroupRowsOp`] and [`MergeGroupsOp`] are a **grouped reduction over rows**:
// many rows in, one row per distinct value of a key column tuple out. It breaks
// all three of the properties the maps above are built on — it does not preserve
// the row count, its output is not a subsequence of its input, and its answer is
// not a function of one row — so it is a second kernel in this module and not a
// fourth case of the first.
//
// It is here rather than in `ops::tabulate` because of what it groups **by**.
// `tabulate` is the same reduction keyed on a **label read out of a volume at
// the row's voxel**; this one is keyed on a **value the row already carries**.
// The two are not a generalisation of one another and the difference is not
// cosmetic: routing a key the rows already hold through a label volume would
// mean rasterising a column into an image the size of the volume in order to
// read it back one voxel at a time, which is an intermediate the size of the
// data invented to express a `GROUP BY` over a table.
//
// What it *does* borrow from `tabulate` is everything that was right there, and
// borrows it rather than restating it: [`FixedPoint`] as the accumulator, for
// exactly the reason that module gives — `f64` addition does not associate, so a
// seam fold in `f64` would give a different last bit for a different cut, and the
// only honest way to declare [`SeamFold::Unordered`] is to quantise once at the
// row and total in `i128` to the end of the run. The one stated limit is where
// the total becomes a column, and it is **refused by name** there rather than
// wrapped through an `as i64`.
//
// The ownership rule is the one thing it does *not* borrow. `tabulate` and
// `ops::detect` own a row by the **centroid** of the region it describes, which
// is right when the row is about a blob of voxels. A group of rows has no
// centroid worth the name — its rows are scattered wherever the key took them,
// and a centroid of them is a position no row occupies. So a group is owned by
// **the block whose core holds the group's least coordinate**, and that
// coordinate is also where the output row sits. It is exact for the same reason
// the centroid rule is: the least coordinate is a function of the merged group,
// cores tile the volume with no overlap and no gap, so exactly one block owns
// each output row.
//
// And it has a theorem the maps do not need, which is what makes the merge's
// tiebreak trivial rather than delicate: **two partials cannot report the same
// least coordinate for one group**. A partial is one block's, a block's rows lie
// in its core, and cores are disjoint — so if two partials do agree on a least
// coordinate, one row position was written into two blocks, which is the
// duplication this module's *"an overlap here costs correctness"* paragraph is
// about. [`GroupFold::merge`] **refuses** it by name rather than picking a side,
// which turns the one silent failure a row reduction can have into a diagnostic.
//
// Two aggregates for one question, because two references disagree
// ----------------------------------------------------------------
// [`Aggregate::FirstPresent`] and [`Aggregate::FirstRow`] are both "the first
// value of this column in this group" and they are different answers. The first
// is *per column*: the value at the least coordinate **among the rows where this
// column has a value**, so two columns of one output row can come from two
// different input rows. The second is *per row*: whatever the row at the group's
// least coordinate holds in that column, absent or not.
//
// They are both here because the pair is a **measured disagreement** rather than
// a preference. Two independent implementations of one published grouped
// statistic were compared row by row against a recorded run of it: one takes the
// first non-null entry per column and the other takes the first row's entry
// including its nulls, and on a table with any absence in it the two answer
// differently. An op that offered only one of them would make that
// unrepresentable — and the consumer that found it holds the choice as a
// parameter and needs both readings out of one plan, which is what this pair is
// for.
//
// What counts as a value, and why it is a **mask column** and not a `NaN`
// ----------------------------------------------------------------------
// Every table format these callers read from spells a missing number `NaN`, and
// that was the first design here: present would mean finite. It is
// unrepresentable. `Table` **refuses a non-finite `F64`** at the push, by name,
// because its canonical order tiebreaks on a column's bits — so a `NaN` there
// would make the order of a table, and every sum over it, a function of which
// quiet `NaN` the platform happened to spell. A row carrying `NaN`-as-null
// cannot exist in this crate at all, and an op whose presence rule was "finite"
// would have been a rule about values no table can hold.
//
// **That refusal is right and it is not this op's to relax**, which is worth
// saying because the alternative was available and was rejected rather than
// overlooked. Widening it would put the crate's one total row order at the mercy
// of a bit pattern, in every op, to spare one reduction a column. And the thing
// the absence was needed *for* survives the change intact: an absence is still a
// first-class case here, still distinguishes [`Aggregate::FirstPresent`] from
// [`Aggregate::FirstRow`], and a consumer holding the choice between those two
// as a parameter still gets both readings out of one plan. What moved is where
// the absence is written down, not whether it can be.
//
// Presence is therefore **a `U64` column the caller nominates**:
// [`Reduction::present`], non-zero for a row that has a value here. It is
// better than the bit pattern would have been on its own terms — it is explicit,
// it is checkable at construction, one mask can serve several reductions, and it
// works for a column of names as well as one of measurements. The value column
// still holds *something* in an absent row; nothing reads it.
//
// A reduction with no mask treats every row as present, which is the ordinary
// case and is what a column with no missing entries wants. The one combination
// refused is [`Aggregate::Count`] **with** no mask, because that is
// [`GROUP_ROWS`] under another name.
//
// [`Aggregate::Sum`], [`Aggregate::Min`] and [`Aggregate::Max`] are refused on a
// `U64` column, and that refusal survives the change: a `U64` column in this
// crate is a *name* — a label, a key, a fixed-point word in offset binary — and
// the sum of two names is not a name, nor is the least of them a fact about
// anything. [`Aggregate::Count`] and the two `First`s are defined over both,
// because counting names and taking the first of one are.
//
// **An empty selection reports zero**, which is `ops::tabulate`'s own convention
// and is forced by the same refusal: a group in which no row was present has no
// least value, no greatest and no first, and the table cannot hold the absence.
// So those columns hold `0` and a [`Aggregate::Count`] over the same column is
// how a reader tells the two apart — exactly as `RegionValues::min` and
// `RegionValues::all_nonfinite` do it one module over.
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
// How the gather gets its image, and why not through `reads_pixels`
// ------------------------------------------------------------------
// A gather reads an **image** at a scattered coordinate, which is a shape no
// other op in this file has. The shell to reach for looks like the one
// `ops::fill`'s second phase uses: `reads_pixels() == true` alongside a
// `FragmentInput::own` at reach `[0, 0, 0]`, so the executor reads this block's
// pixels, hands over this block's rows, and the op indexes one against the
// other. That combination is legal in the trait and **cannot be arranged in a
// plan**, for a reason about the *phase index* rather than about gathering, and
// both halves of it were measured against the executor rather than argued:
//
// * **a fragment phase `p` reads image `p`.** So a gather at phase `p` needs
//   phase `p - 1` to have written an image, and the phase before a row op is the
//   row *producer*, which writes fragments and no image. The executor says so:
//   *"phase 1 reads image 1, which phase 0 did not write: it runs a fragment op
//   that declares `writes_pixels() == false`"*;
// * **a fragment input must come from a strictly earlier phase.** So the gather
//   cannot be phase 0 — where it would read image 0, the array the run was
//   handed — because its rows would have to come from phase 0 too. Splitting it
//   into a second `execute_phases` over the same store does not help; the phase
//   index is a plan-local number and the check fires again: *"reads stream
//   \"rows.set\" from phase 0, which is this phase or a later one"*.
//
// **Both refusals still stand, and [`GatherRowsOp`] does not go round either.**
// What resolves it is that the image a gather wants was never image `p`: it is a
// *second* array, and [`FragmentOp::source_inputs`] is the declaration for one.
// So the shell declares `reads_pixels() == false` — it never touches the image
// its phase was handed, and `fragment.rs` is explicit that the two declarations
// are independent, so it *"pays for one array rather than two"* — names the image
// it samples with [`SourceInput::voxelwise`], and is applied through
// `apply_with`, whose default refuses rather than dropping the operand.
//
// The image it names may be any image at or below the phase's own, image 0
// included, and it is fetched over the block's own fetch region and recorded on
// the phase, so the DAG depends on its producer, the image lifetimes keep it
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
// fragment, and it reads *its own* block of the image; those two agree only while
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
// A gathered value is a **finite `f64`**, and an image whose element type cannot
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

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::decomposition::Decomposition;
use crate::env::{BlockBuf, Environment};
use crate::error::{Error, Result};
use crate::fragment::{
    fold_fragments, fragment_phase, pack_u64, unpack_u64, BlockOutput, BlockView, Coverage,
    FragmentInput, FragmentOp, FragmentOutput, SeamFold, SourceBlocks,
};
use crate::geometry::BlockGrid;
use crate::op::SourceInput;
use crate::region::Region;
use crate::sidecar::{FragmentKey, Lifecycle};
use crate::table::{Column, ColumnType, Row, RowBuilder, Schema, Table, Value, POSITION_WORDS};
use crate::voxels::Voxels;

use super::detect::owner_of;
use super::tabulate::FixedPoint;

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
/// A non-finite value is refused naming the coordinate. An unwritten image in
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
                "a gather was asked to read a {} image into an f64 column. That type holds \
                 integers beyond the ones an f64 represents exactly, so a large value would \
                 arrive as a nearby number nothing downstream could tell from a measurement. \
                 Narrow the image, or carry the value some other way.",
                pixels.dtype().numpy_name()
            )))
        }
    };
    if !value.is_finite() {
        return Err(Error::invalid(format!(
            "the image holds {value} at {at:?}, which is not finite. An unwritten image in this \
             crate is NaN so that an absence cannot pass for a value, and gathering one is that \
             absence reaching a consumer; it is refused here rather than carried."
        )));
    }
    Ok(value)
}

/// Every row of `table`, with the image's value at its own coordinate appended.
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
                     the image its own block holds, so a row somewhere else would be given a \
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
            // Every block, always. These phases write no image, so the tiling
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
/// Reads no pixels, writes no image, reach 0. The rows it emits are in the
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
/// image holds at each row's own coordinate.
///
/// Reads no pixels, writes no image, reach 0. `reads_pixels` is about the image
/// the phase is *handed* and a gather never wants that one; the image it samples
/// is declared as a [`SourceInput`], fetched at the block's own fetch region and
/// recorded on the phase, so this op pays for one array rather than two and the
/// plan says which. See the module header for why that declaration is what makes
/// the shell possible at all, and for the [`SeamFold::PerBlock`] claim.
///
/// The image may be any at or below the phase's own — image 0 included — so
/// *rows from one array, values from a second* is the ordinary arrangement.
pub struct GatherRowsOp {
    name: &'static str,
    rows: RowStreams,
    image: usize,
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
        image: impl Into<crate::assemble::ImageId>,
        column: impl Into<String>,
    ) -> Result<Self> {
        let image = image.into().index();
        let column = column.into();
        let schema = gathered_schema(&rows.schema, &column)?;
        Ok(Self {
            name,
            rows,
            image,
            column,
            schema,
        })
    }

    /// The image this op samples.
    pub fn image(&self) -> usize {
        self.image
    }

    /// The column it appends.
    pub fn column(&self) -> &str {
        &self.column
    }

    /// The schema of the rows this op emits: its input's, plus one `f64` column.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// One block's rows, gathered against the block of `image` covering `read`,
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
        image: &BlockBuf,
        read: &Region,
        core: &Region,
    ) -> Result<Vec<u8>> {
        let BlockBuf::Array(pixels) = image else {
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
                "a gather was handed image {} as {:?} for a block read extent of {shape:?}. A \
                 source image is fetched at the block's own fetch region, so a disagreement here \
                 is the plan handing over two geometries rather than a row in the wrong place.",
                self.image,
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

    /// The image sampled, at exactly the extent the block owns. A gather reads
    /// one voxel per row and every row is inside this block's core, so there is
    /// nothing outside the block's own fetch to read; a reach here would widen
    /// the phase halo and buy nothing.
    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.image)]
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
            "a gather reads an image at each row's coordinate and cannot be computed from the \
             rows alone. It is applied through `apply_with`.",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        Ok(BlockOutput::fragment(
            self.rows.output.clone(),
            self.gather_block(
                at.volume(),
                self.rows.own(at),
                sources.get(self.image)?,
                at.read,
                at.core,
            )?,
        ))
    }
}

/// **Rows in, the rows that satisfy the predicate out.**
///
/// Reads no pixels, writes no image, reach 0. Coordinates and columns travel
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

// ----------------------------------------------------------- the reduction --

/// The column of the output table holding how many input rows a group had.
///
/// Every row of the group, whatever any column of it held — which is the one
/// number a caller always wants and the one number a [`Aggregate::Count`] over a
/// column is not. A key column of this name is refused, because two columns of
/// one schema cannot share it.
pub const GROUP_ROWS: &str = "rows";

/// What one column of a group is reduced to.
///
/// See the module header for the presence rule, for why four of the six are
/// refused on a `U64` column, and for why there are two `First`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregate {
    /// How many rows of the group were **present** in this column, by its
    /// [`Reduction::present`] mask. Refused without one, because the answer is
    /// then [`GROUP_ROWS`].
    Count,
    /// The fixed-point total of those values, as a `U64` column in offset
    /// binary — the same form and the same scale [`FixedPoint`] gives
    /// `ops::tabulate`'s. `F64` only.
    ///
    /// The integer is emitted rather than the float deliberately: it is the form
    /// the invariance claim is about, and handing back only the `f64` would give
    /// a reader a number whose exactness they would have to take on trust.
    /// [`GroupValues::sum`] is the float, computed once, from it.
    Sum,
    /// The least value, by [`f64::total_cmp`], over the present rows. `F64`
    /// only, and `0.0` for a group where no row was present — see the module
    /// header on the empty selection.
    Min,
    /// The greatest, on the same terms. `F64` only.
    Max,
    /// The value at the least coordinate **among the present rows** — so two
    /// columns of one output row can come from two different input rows.
    /// pandas' `GroupBy.first()`. Either column type, and `0` for a group where
    /// no row was present.
    FirstPresent,
    /// The value this column holds in the row at the group's **least
    /// coordinate**, present or not — so every column so reduced comes from one
    /// row. Either column type.
    FirstRow,
}

impl Aggregate {
    /// The prefix this aggregate's output column carries.
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Sum => "sum",
            Self::Min => "min",
            Self::Max => "max",
            Self::FirstPresent => "first_present",
            Self::FirstRow => "first_row",
        }
    }

    /// Whether this aggregate is defined over a column of **names**, which is
    /// what a `U64` column is here. Counting names and taking the first of one
    /// are; totalling them and ordering them are not.
    fn accepts_names(self) -> bool {
        matches!(self, Self::Count | Self::FirstPresent | Self::FirstRow)
    }
}

/// One column reduced one way, over the rows a mask says have a value there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reduction {
    /// An index into the **input** schema.
    pub column: usize,
    pub aggregate: Aggregate,
    /// The `U64` column whose non-zero entries mark the rows this column has a
    /// value in, or `None` for a column every row has a value in.
    ///
    /// See the module header for why an absence is a column here rather than a
    /// `NaN`: `Table` refuses a non-finite `F64` at the push, so the bit pattern
    /// every dataframe uses cannot reach a row of this crate at all.
    pub present: Option<usize>,
}

impl Reduction {
    /// A reduction over a column every row has a value in.
    pub fn new(column: usize, aggregate: Aggregate) -> Self {
        Self {
            column,
            aggregate,
            present: None,
        }
    }

    /// The same, over the rows `mask` marks.
    pub fn masked(column: usize, aggregate: Aggregate, mask: usize) -> Self {
        Self {
            column,
            aggregate,
            present: Some(mask),
        }
    }
}

/// **What the grouping is**: which columns are the key, which are reduced and
/// how, and at what scale the sums are taken.
///
/// Built once and validated once, so that a key naming a column the rows do not
/// have, or a `Sum` over a column of names, is a refusal at construction rather
/// than on the first block of a run.
#[derive(Debug, Clone)]
pub struct Grouping {
    input: Schema,
    key: Vec<usize>,
    reductions: Vec<Reduction>,
    fixed: FixedPoint,
    output: Schema,
}

impl Grouping {
    /// Refuses everything that would make the output table meaningless, by name.
    ///
    /// * an empty key. Grouping by nothing is a reduction of the whole table to
    ///   one row, which is a different operation with a different ownership rule
    ///   — there is no key to own it by — and pretending it is this one would
    ///   give a table whose single row's coordinate is an accident;
    /// * a key column that is **not `U64`**. A group key is a *name* and a float
    ///   is not one: two `NaN`s with different payloads would be two groups, and
    ///   `0.0` and `-0.0` two more, which is a partition of the rows nobody
    ///   asked for;
    /// * a repeated key column, and a key column named [`GROUP_ROWS`];
    /// * a reduction over a column that is part of the key — the answer is the
    ///   key back, which is `ops::tabulate`'s own refusal of a label volume
    ///   reduced over itself;
    /// * an aggregate that needs an absence, over a column that has none.
    pub fn new(
        input: Schema,
        key: Vec<usize>,
        reductions: Vec<Reduction>,
        fixed: FixedPoint,
    ) -> Result<Self> {
        if key.is_empty() {
            return Err(Error::invalid(
                "a grouping with no key columns reduces the whole table to one row. That is a \
                 different operation: this one owns each output row by the block holding its \
                 group's least coordinate, and a single row over the whole table would be owned \
                 by whichever block happened to hold the least row of the lot."
                    .to_string(),
            ));
        }
        for (position, column) in key.iter().enumerate() {
            let held = input.columns().get(*column).ok_or_else(|| {
                Error::invalid(format!(
                    "the key names column {column} and the rows have {}",
                    input.len()
                ))
            })?;
            if held.kind() != ColumnType::U64 {
                return Err(Error::invalid(format!(
                    "column {column} ({:?}) is a key of this grouping and holds floats. A group \
                     key is a name: under a float key two `NaN`s with different payloads are two \
                     groups and `0.0` and `-0.0` are two more, so the partition would depend on \
                     bits no caller controls. Carry the key as a `U64` column.",
                    held.name()
                )));
            }
            if held.name() == GROUP_ROWS {
                return Err(Error::invalid(format!(
                    "column {column} is a key of this grouping and is named {GROUP_ROWS:?}, \
                     which is the name the output's group size carries. One schema cannot hold \
                     two columns of one name."
                )));
            }
            if key[..position].contains(column) {
                return Err(Error::invalid(format!(
                    "column {column} appears twice in the key. Repeating it partitions the rows \
                     exactly as naming it once does and puts two columns of one name in the \
                     output."
                )));
            }
        }
        let mut columns = Vec::with_capacity(1 + key.len() + reductions.len());
        for column in &key {
            columns.push(input.columns()[*column].clone());
        }
        columns.push(Column::u64(GROUP_ROWS));
        for reduction in &reductions {
            let held = input.columns().get(reduction.column).ok_or_else(|| {
                Error::invalid(format!(
                    "a reduction names column {} and the rows have {}",
                    reduction.column,
                    input.len()
                ))
            })?;
            if key.contains(&reduction.column) {
                return Err(Error::invalid(format!(
                    "column {} ({:?}) is both a key of this grouping and reduced by it. Every \
                     row of a group holds the same value there, so the answer is the key back \
                     under a second name.",
                    reduction.column,
                    held.name()
                )));
            }
            if held.kind() == ColumnType::U64 && !reduction.aggregate.accepts_names() {
                return Err(Error::invalid(format!(
                    "column {} ({:?}) holds `U64` and this grouping takes its {}. A `U64` column \
                     here is a name — a label, a key, a fixed-point word in offset binary — and \
                     the sum of two names is not a name, nor is the least of them a fact about \
                     anything. That is a category error rather than a limitation; `Count`, \
                     `FirstPresent` and `FirstRow` are defined over it.",
                    reduction.column,
                    held.name(),
                    reduction.aggregate.prefix()
                )));
            }
            if reduction.aggregate == Aggregate::Count && reduction.present.is_none() {
                return Err(Error::invalid(format!(
                    "column {} ({:?}) is counted with no presence mask, and every row of a group \
                     is then present in it — so the answer is {GROUP_ROWS:?} under another name. \
                     Name the `U64` column that says which rows have a value here, or read \
                     {GROUP_ROWS:?}.",
                    reduction.column,
                    held.name()
                )));
            }
            if let Some(mask) = reduction.present {
                let flag = input.columns().get(mask).ok_or_else(|| {
                    Error::invalid(format!(
                        "a reduction's presence mask names column {mask} and the rows have {}",
                        input.len()
                    ))
                })?;
                if flag.kind() != ColumnType::U64 {
                    return Err(Error::invalid(format!(
                        "column {mask} ({:?}) is a presence mask and holds floats. A mask is read \
                         as non-zero-is-present, and over floats that would make `-0.0` present \
                         and `0.0` absent on some paths and neither on others. Carry it as a \
                         `U64` column.",
                        flag.name()
                    )));
                }
            }
            columns.push(output_column(held, reduction.aggregate, fixed));
        }
        let output = Schema::new(columns)?;
        Ok(Self {
            input,
            key,
            reductions,
            fixed,
            output,
        })
    }

    /// The schema of the rows the reduction reads.
    pub fn input(&self) -> &Schema {
        &self.input
    }

    /// The schema of the rows the reduction writes: the key columns under their
    /// own names, then [`GROUP_ROWS`], then one per reduction in order.
    pub fn output(&self) -> &Schema {
        &self.output
    }

    pub fn key(&self) -> &[usize] {
        &self.key
    }

    pub fn reductions(&self) -> &[Reduction] {
        &self.reductions
    }

    pub fn fixed(&self) -> FixedPoint {
        self.fixed
    }

    /// How many `u64` one group occupies on the wire.
    fn entry_words(&self) -> usize {
        self.key.len() + GROUP_HEAD_WORDS + self.reductions.len() * COLUMN_FOLD_WORDS
    }

    /// The rows of one blob, folded. The block-local half of the reduction.
    ///
    /// A free function in all but name — it takes a schema and a blob and no
    /// lattice — for `ops::tabulate`'s stated reason: the arithmetic should be
    /// drivable from a test without a plan, an environment or a grid.
    pub fn fold_blob(
        &self,
        volume: [usize; 3],
        blob: &[u8],
    ) -> Result<BTreeMap<Vec<u64>, GroupFold>> {
        let table = sealed(volume, self.input.clone(), blob)?;
        let mut groups: BTreeMap<Vec<u64>, GroupFold> = BTreeMap::new();
        // Canonical order, which is what makes "the first row of this group" a
        // property of the row set rather than of the order the producer wrote
        // them in. The merge then never has to break a tie; see the module
        // header's theorem about disjoint cores.
        for row in table.scan(&Region::whole(&volume))? {
            let key: Vec<u64> = self
                .key
                .iter()
                .map(|column| row.u64(*column))
                .collect::<Result<_>>()?;
            let fold = groups
                .entry(key)
                .or_insert_with(|| GroupFold::new(row.at(), self.reductions.len()));
            fold.absorb(&row, &self.reductions, self.input.columns(), self.fixed)?;
        }
        Ok(groups)
    }

    /// One group as the row it becomes.
    ///
    /// The output row's coordinate is the group's **least**, which is also what
    /// owns it — one statement, used twice, so the two cannot drift.
    pub fn finish(&self, key: &[u64], fold: &GroupFold) -> Result<([usize; 3], Vec<Value>)> {
        let mut values: Vec<Value> = Vec::with_capacity(self.output.len());
        for word in key {
            values.push(Value::U64(*word));
        }
        values.push(Value::U64(fold.rows));
        for (reduction, column) in self.reductions.iter().zip(&fold.columns) {
            let kind = self.input.columns()[reduction.column].kind();
            values.push(match reduction.aggregate {
                Aggregate::Count => Value::U64(column.present),
                // **The one stated limit, and it is refused rather than
                // wrapped.** The fold is `i128` and has no range of its own; a
                // total that a signed 64-bit column cannot hold is the answer
                // being too large to report rather than too large to compute,
                // and `as i64` there would turn a large positive total into a
                // small negative mean with nothing failing.
                Aggregate::Sum => Value::U64(self.fixed.to_column(column.total)?),
                // **Zero for an empty selection**, which is `ops::tabulate`'s
                // convention and is forced by the same rule: a `Table` refuses
                // a non-finite `F64`, so the absence has no spelling and a
                // `Count` over the same column is how a reader tells a group
                // that selected nothing from one whose least value is zero.
                Aggregate::Min => Value::F64(column.min.unwrap_or(0.0)),
                Aggregate::Max => Value::F64(column.max.unwrap_or(0.0)),
                Aggregate::FirstPresent => {
                    typed(kind, column.first_present.map_or(0, |(_, word)| word))
                }
                Aggregate::FirstRow => typed(kind, column.first_row),
            });
        }
        Ok((fold.least, values))
    }
}

/// The output column one reduction contributes.
fn output_column(held: &Column, aggregate: Aggregate, fixed: FixedPoint) -> Column {
    let name = format!("{}.{}", aggregate.prefix(), held.name());
    match aggregate {
        Aggregate::Count => Column::u64(name),
        // The scale lives in the column name, so two groupings at two scales are
        // *different schemas* and `Table::write` refuses to mix them — which is
        // `ops::tabulate`'s rule, inherited rather than restated.
        Aggregate::Sum => Column::u64(format!("{name}{}", fixed.suffix())),
        Aggregate::Min | Aggregate::Max => Column::f64(name),
        // The only output column whose type follows its input's: the first of a
        // name is a name and the first of a measurement is a measurement.
        Aggregate::FirstPresent | Aggregate::FirstRow => match held.kind() {
            ColumnType::U64 => Column::u64(name),
            ColumnType::F64 => Column::f64(name),
        },
    }
}

fn typed(kind: ColumnType, word: u64) -> Value {
    match kind {
        ColumnType::U64 => Value::U64(word),
        ColumnType::F64 => Value::F64(f64::from_bits(word)),
    }
}

/// The wire word for "this column had no value in this group", which is a
/// `NaN`.
///
/// The same constant and the same argument `ops::tabulate` gives its own: a
/// selection only ever holds a value that was finite when it was read, so no
/// `NaN` can be a real one and the absence needs no flag beside it. It can only
/// ever be written into an `F64` column, because the aggregates that can be
/// absent are refused on the other kind.
const ABSENT_WORD: u64 = 0x7ff8_0000_0000_0000;

/// Words per group before the per-column records: the least coordinate and the
/// row count.
const GROUP_HEAD_WORDS: usize = 4;
/// Words per reduced column: the present count, the `i128` total as two, the
/// two selections, the present-first word and its packed position, and the
/// least row's word. See [`ColumnFold`].
const COLUMN_FOLD_WORDS: usize = 8;

/// What one column of one group accumulates.
///
/// Every field combines associatively and commutatively — `+` on the counts and
/// on the `i128` total, a selection under a total order for the rest — which is
/// what lets [`MergeGroupsOp`] declare [`SeamFold::Unordered`] honestly. Nothing
/// here is an `f64` addition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnFold {
    /// Rows of the group in which this column had a value.
    pub present: u64,
    /// The fixed-point total over those values.
    pub total: i128,
    /// The least and greatest of them, by [`f64::total_cmp`].
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// The least coordinate at which this column had a value, and the word
    /// there.
    pub first_present: Option<([usize; 3], u64)>,
    /// The word this column holds in the row at the group's least coordinate,
    /// whatever it is.
    pub first_row: u64,
}

impl ColumnFold {
    fn new() -> Self {
        Self {
            present: 0,
            total: 0,
            min: None,
            max: None,
            first_present: None,
            // Overwritten by the group's first row, which every group has by
            // construction — a group exists because a row created it.
            first_row: 0,
        }
    }

    /// `take_other` is the group-level decision about which side holds the
    /// smaller least coordinate, passed down rather than recomputed per column
    /// so that every column of a group answers [`Aggregate::FirstRow`] from
    /// **one** row.
    fn merge(&mut self, other: &ColumnFold, take_other: bool) {
        self.present += other.present;
        self.total += other.total;
        self.min = select(self.min, other.min, std::cmp::Ordering::Less);
        self.max = select(self.max, other.max, std::cmp::Ordering::Greater);
        if match (self.first_present, other.first_present) {
            (_, None) => false,
            (None, Some(_)) => true,
            (Some((mine, _)), Some((theirs, _))) => theirs < mine,
        } {
            self.first_present = other.first_present;
        }
        if take_other {
            self.first_row = other.first_row;
        }
    }
}

/// The one of two values that is `wanted` under [`f64::total_cmp`], carrying
/// absence through.
///
/// `total_cmp` rather than `f64::min`/`max` because those are not a total order:
/// they return the *other* operand for a `NaN`, so a fold over them depends on
/// the order the values arrived in — which is the property this whole module is
/// about.
fn select(mine: Option<f64>, theirs: Option<f64>, wanted: std::cmp::Ordering) -> Option<f64> {
    match (mine, theirs) {
        (Some(mine), Some(theirs)) => Some(if theirs.total_cmp(&mine) == wanted {
            theirs
        } else {
            mine
        }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

/// What one group accumulates.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupFold {
    /// The least coordinate any row of the group sits at. **The output row's
    /// position and the block that owns it**, in one number.
    pub least: [usize; 3],
    /// Rows of the group, whatever any column of them held.
    pub rows: u64,
    /// One per reduction, in the grouping's order.
    pub columns: Vec<ColumnFold>,
}

impl GroupFold {
    fn new(least: [usize; 3], reductions: usize) -> Self {
        Self {
            least,
            rows: 0,
            columns: vec![ColumnFold::new(); reductions],
        }
    }

    /// One row into this group's accumulators.
    fn absorb(
        &mut self,
        row: &Row<'_>,
        reductions: &[Reduction],
        columns: &[Column],
        fixed: FixedPoint,
    ) -> Result<()> {
        let at = row.at();
        // Rows arrive in canonical order, so this is the first row seen; the
        // comparison is kept anyway because `fold_blob` is not the only caller a
        // future reader will write, and a `least` that depended on the walk
        // order would be a silent cut-dependence.
        let leading = at < self.least || self.rows == 0;
        if leading {
            self.least = at;
        }
        self.rows += 1;
        for (reduction, fold) in reductions.iter().zip(&mut self.columns) {
            let word = row.words()[POSITION_WORDS + reduction.column];
            if leading {
                fold.first_row = word;
            }
            // The mask, and nothing else, decides presence. A reduction with no
            // mask treats every row as present; see the module header on why an
            // absence is a column here and not a `NaN`.
            let present = match reduction.present {
                None => true,
                Some(mask) => row.words()[POSITION_WORDS + mask] != 0,
            };
            if !present {
                continue;
            }
            fold.present += 1;
            if fold.first_present.is_none_or(|(seen, _)| at < seen) {
                fold.first_present = Some((at, word));
            }
            if columns[reduction.column].kind() == ColumnType::U64 {
                // A name is counted and can be first, and the aggregates that
                // would total or order it are refused at construction — so
                // there is nothing further to accumulate.
                continue;
            }
            let value = f64::from_bits(word);
            // `Table` refuses a non-finite `F64` at the push, so this can only
            // be reached by a blob that did not come through `RowBuilder`.
            // Refused by name rather than skipped: skipping would make the sum
            // depend on which rows a hand-built blob happened to poison, and do
            // it silently.
            if !value.is_finite() {
                return Err(Error::invalid(format!(
                    "the row at {at:?} holds {value} in column {} ({:?}), which a `Table` refuses \
                     at the push. This blob did not come through `RowBuilder`, and a reduction \
                     that skipped the value would answer differently for the same rows depending \
                     on how they were written.",
                    reduction.column,
                    columns[reduction.column].name()
                )));
            }
            fold.total += fixed.quantise(value)?.unwrap_or(0);
            fold.min = select(fold.min, Some(value), std::cmp::Ordering::Less);
            fold.max = select(fold.max, Some(value), std::cmp::Ordering::Greater);
        }
        Ok(())
    }

    /// Fold another block's partial for the same group into this one.
    ///
    /// **Refuses two partials that report the same least coordinate**, which is
    /// the module header's theorem being checked rather than assumed: a block's
    /// rows lie in its core and cores are disjoint, so an agreement here means
    /// one row position reached two blocks. That is the duplication a row op can
    /// never recover from — the group's `rows` would be doubled and its total
    /// with it — and it is the one silent failure this reduction can have.
    pub fn merge(&mut self, other: &GroupFold) -> Result<()> {
        if self.columns.len() != other.columns.len() {
            return Err(Error::invalid(format!(
                "two partials of one group carry {} and {} reduced column(s). They were folded \
                 under two different groupings, so their columns do not mean the same things.",
                self.columns.len(),
                other.columns.len()
            )));
        }
        if self.least == other.least {
            return Err(Error::invalid(format!(
                "two partials of one group both report {:?} as the group's least row. A partial \
                 is one block's and a block's rows lie in its core, so cores being disjoint makes \
                 this impossible — unless one row position was written into two blocks, in which \
                 case the group's row count and its totals are already doubled. Refused here \
                 rather than folded, because a duplicated row is indistinguishable from a real \
                 one once it is in the sum.",
                self.least
            )));
        }
        let take_other = other.least < self.least;
        for (mine, theirs) in self.columns.iter_mut().zip(&other.columns) {
            mine.merge(theirs, take_other);
        }
        if take_other {
            self.least = other.least;
        }
        self.rows += other.rows;
        Ok(())
    }
}

/// A block's groups as a fragment, ascending by key.
pub fn encode_groups(
    grouping: &Grouping,
    groups: &BTreeMap<Vec<u64>, GroupFold>,
) -> Result<Vec<u8>> {
    let mut words = Vec::with_capacity(groups.len() * grouping.entry_words());
    for (key, fold) in groups {
        if key.len() != grouping.key.len() {
            return Err(Error::invalid(format!(
                "a group carries a {}-word key and this grouping has {} key column(s)",
                key.len(),
                grouping.key.len()
            )));
        }
        words.extend_from_slice(key);
        for axis in 0..3 {
            words.push(fold.least[axis] as u64);
        }
        words.push(fold.rows);
        for column in &fold.columns {
            words.push(column.present);
            let bits = column.total as u128;
            words.push(bits as u64);
            words.push((bits >> 64) as u64);
            words.push(column.min.map_or(ABSENT_WORD, f64::to_bits));
            words.push(column.max.map_or(ABSENT_WORD, f64::to_bits));
            match column.first_present {
                // The position travels beside the word because the merge selects
                // on it; `ABSENT_WORD` in the value with a zero position would
                // have been one flag fewer and one ambiguity more, since a real
                // first can sit at the origin.
                Some((at, word)) => {
                    words.push(word);
                    words.push(encoded_position(at)?);
                }
                None => {
                    words.push(ABSENT_WORD);
                    words.push(NO_POSITION);
                }
            }
            words.push(column.first_row);
        }
    }
    Ok(pack_u64(&words))
}

/// The wire word for "no position", which no real one can be: a position is
/// three `usize` packed into one word only when each fits 21 bits, and this is
/// every bit set.
const NO_POSITION: u64 = u64::MAX;

/// Three coordinates in one word, 21 bits each.
///
/// A partial's per-column first sits at a coordinate, and carrying three words
/// for it would make the record half position. 21 bits is 2 097 151 per axis,
/// which is past any volume this crate can hold in memory or on one node; a
/// coordinate above it is refused rather than truncated, because a truncated
/// position selects the wrong row and does it silently.
fn encoded_position(at: [usize; 3]) -> Result<u64> {
    for axis in 0..3 {
        if at[axis] > MAX_PACKED_COORDINATE {
            return Err(Error::invalid(format!(
                "a grouped partial carries the coordinate {at:?}, whose axis {axis} is past \
                 {MAX_PACKED_COORDINATE} — the largest this record's packed position can hold. \
                 Truncating it would select a different row as the group's first and do it \
                 silently, so it is refused; the fix is a wider record rather than a wider \
                 volume."
            )));
        }
    }
    Ok((at[0] as u64) | ((at[1] as u64) << 21) | ((at[2] as u64) << 42))
}

fn decoded_position(word: u64) -> [usize; 3] {
    [
        (word & 0x1f_ffff) as usize,
        ((word >> 21) & 0x1f_ffff) as usize,
        ((word >> 42) & 0x1f_ffff) as usize,
    ]
}

/// The largest coordinate [`encoded_position`] can carry on any axis.
pub const MAX_PACKED_COORDINATE: usize = 0x1f_ffff;

/// The other half of [`encode_groups`]. A length that is not a whole number of
/// entries is a truncated fragment and says so.
pub fn decode_groups(grouping: &Grouping, bytes: &[u8]) -> Result<Vec<(Vec<u64>, GroupFold)>> {
    let words = unpack_u64(bytes)?;
    let entry = grouping.entry_words();
    if words.len() % entry != 0 {
        return Err(Error::invalid(format!(
            "a grouped partial is a whole number of {entry}-word entries — {} key word(s), \
             {GROUP_HEAD_WORDS} of head and {COLUMN_FOLD_WORDS} per reduced column — and this \
             one is {} word(s)",
            grouping.key.len(),
            words.len()
        )));
    }
    let mut found = Vec::with_capacity(words.len() / entry);
    for chunk in words.chunks_exact(entry) {
        let key = chunk[..grouping.key.len()].to_vec();
        let head = &chunk[grouping.key.len()..];
        let least = [head[0] as usize, head[1] as usize, head[2] as usize];
        let mut columns = Vec::with_capacity(grouping.reductions.len());
        for record in head[GROUP_HEAD_WORDS..].chunks_exact(COLUMN_FOLD_WORDS) {
            let position = record[6];
            columns.push(ColumnFold {
                present: record[0],
                total: (((record[2] as u128) << 64) | record[1] as u128) as i128,
                min: finite(record[3]),
                max: finite(record[4]),
                first_present: (position != NO_POSITION)
                    .then(|| (decoded_position(position), record[5])),
                first_row: record[7],
            });
        }
        if columns.len() != grouping.reductions.len() {
            return Err(Error::invalid(format!(
                "a grouped partial carries {} reduced column(s) and this grouping has {}",
                columns.len(),
                grouping.reductions.len()
            )));
        }
        found.push((
            key,
            GroupFold {
                least,
                rows: head[3],
                columns,
            },
        ));
    }
    Ok(found)
}

/// Reads `is_finite` rather than comparing against [`ABSENT_WORD`], so that any
/// `NaN` and any infinity decodes as absent — neither is a value a selection can
/// hold, so neither can be let through as one.
fn finite(word: u64) -> Option<f64> {
    let value = f64::from_bits(word);
    value.is_finite().then_some(value)
}

/// One partial's groups folded into `totals`.
///
/// The whole of the seam combine's plumbing in one place, so that the streamed
/// path in [`MergeGroupsOp::apply`] and the free-function [`MergeGroupsOp::fold`]
/// cannot drift into folding differently.
fn absorb_groups(
    grouping: &Grouping,
    totals: &mut BTreeMap<Vec<u64>, GroupFold>,
    bytes: &[u8],
) -> Result<()> {
    for (key, fold) in decode_groups(grouping, bytes)? {
        match totals.get_mut(&key) {
            Some(seen) => seen.merge(&fold)?,
            None => {
                totals.insert(key, fold);
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------ the grouping phase --

/// **Rows in, one partial per block out.** Phase one of the reduction.
///
/// Reads no pixels, writes no image, and declares reach `[0, 0, 0]` on its row
/// stream — a block folds its own rows and nobody else's, which is why an
/// overlap upstream is a correctness bug rather than a cost and why
/// [`GroupFold::merge`] refuses the evidence of one.
///
/// [`SeamFold::PerBlock`], which is true of it: a partial is a function of the
/// rows in one block, and the fold across the seam belongs to
/// [`MergeGroupsOp`].
pub struct GroupRowsOp {
    name: &'static str,
    rows: RowStreams,
    grouping: Grouping,
}

impl GroupRowsOp {
    /// Refuses a grouping whose input schema is not the one the rows arrive
    /// with, at construction rather than on the first block.
    pub fn new(name: &'static str, rows: RowStreams, grouping: Grouping) -> Result<Self> {
        if grouping.input() != &rows.schema {
            return Err(Error::invalid(format!(
                "this grouping reduces rows of {} and the stream {:?} carries {}. A blob of the \
                 wrong schema decodes perfectly and answers differently, so the two are compared \
                 here rather than at the first block.",
                schema_names(grouping.input()),
                rows.input,
                schema_names(&rows.schema)
            )));
        }
        Ok(Self {
            name,
            rows,
            grouping,
        })
    }

    /// The schema of the rows this op's *partials* eventually become — the
    /// grouping's output, which is what a consumer of [`MergeGroupsOp`] reads.
    pub fn schema(&self) -> &Schema {
        self.grouping.output()
    }

    pub fn grouping(&self) -> &Grouping {
        &self.grouping
    }
}

fn schema_names(schema: &Schema) -> String {
    format!(
        "[{}]",
        schema
            .columns()
            .iter()
            .map(|column| column.name().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

impl FragmentOp for GroupRowsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        self.rows.inputs()
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        self.rows.outputs()
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let groups = self.grouping.fold_blob(at.volume(), self.rows.own(at))?;
        Ok(BlockOutput::fragment(
            self.rows.output.clone(),
            encode_groups(&self.grouping, &groups)?,
        ))
    }
}

// --------------------------------------------------------- the merge phase --

/// **Partials in, the grouped table out.** Phase two of the reduction.
///
/// Declares a whole-lattice reach and `gathers() == false`, so the partials are
/// streamed one at a time rather than all made resident: the reads are the same
/// and the residency is one fragment plus the accumulator.
///
/// [`SeamFold::Unordered`], and it is the honest claim rather than the
/// convenient one — see the module header on the accumulator. The executor
/// checks it by applying each block a second time with the neighbourhood
/// reversed and requiring byte-identical output, which an `f64` running sum
/// would fail on the first group split three ways.
///
/// **Each block emits only the groups it owns** — the block whose core holds the
/// group's least coordinate — so the union over the lattice is the table exactly
/// once. Without it every block would write the whole table and a [`Table`] that
/// took them all would hold every row once per block.
pub struct MergeGroupsOp {
    name: &'static str,
    input: String,
    input_phase: usize,
    lattice: [usize; 3],
    grouping: Grouping,
    output: String,
    lifecycle: Lifecycle,
}

impl MergeGroupsOp {
    /// `lattice` is the blocks-per-axis of the phase this runs on, which is what
    /// makes the reach the whole lattice; see `BlockGrid::blocks_per_axis`.
    pub fn new(
        name: &'static str,
        input: impl Into<String>,
        input_phase: usize,
        lattice: [usize; 3],
        grouping: Grouping,
        output: impl Into<String>,
        lifecycle: Lifecycle,
    ) -> Result<Self> {
        let input = input.into();
        let output = output.into();
        if input == output {
            return Err(Error::invalid(format!(
                "the grouped merge was asked to read and write the stream {input:?}. One stream \
                 would then hold this phase's rows and the grouping phase's partials under keys \
                 that differ only in a phase index, and the two are not even the same wire \
                 format."
            )));
        }
        Ok(Self {
            name,
            input,
            input_phase,
            lattice,
            grouping,
            output,
            lifecycle,
        })
    }

    /// The schema of the rows this op writes.
    pub fn schema(&self) -> &Schema {
        self.grouping.output()
    }

    pub fn grouping(&self) -> &Grouping {
        &self.grouping
    }

    pub fn stream(&self) -> &str {
        &self.output
    }

    /// Every partial folded into one map, keyed by the group key.
    ///
    /// The fold as a free function over the partials themselves, so that
    /// "combining these in any order gives this map" is assertable without a
    /// run.
    pub fn fold<'a>(
        &self,
        partials: impl IntoIterator<Item = &'a [u8]>,
    ) -> Result<BTreeMap<Vec<u64>, GroupFold>> {
        let mut totals: BTreeMap<Vec<u64>, GroupFold> = BTreeMap::new();
        for bytes in partials {
            absorb_groups(&self.grouping, &mut totals, bytes)?;
        }
        Ok(totals)
    }

    /// The groups `block` owns, as the blob it writes.
    ///
    /// Ownership first, encoding second: which block writes a row is a property
    /// of the group, so it must not be able to depend on the form it is written
    /// in.
    pub fn encode_owned(
        &self,
        totals: &BTreeMap<Vec<u64>, GroupFold>,
        grid: &BlockGrid,
        block: [usize; 3],
    ) -> Result<Vec<u8>> {
        let mut builder = RowBuilder::new(Arc::new(self.grouping.output().clone()));
        for (key, fold) in totals {
            if owner_of(grid, fold.least) != block {
                continue;
            }
            let (at, values) = self.grouping.finish(key, fold)?;
            // Round-tripped through the typed push rather than written straight
            // into the buffer, so a schema that grew a column without `finish`
            // growing a value is refused here rather than producing rows that
            // decode as something plausible.
            builder.push(at, &values)?;
        }
        Ok(builder.encode())
    }
}

impl FragmentOp for MergeGroupsOp {
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

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.output.clone(),
            self.lifecycle,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mut totals: BTreeMap<Vec<u64>, GroupFold> = BTreeMap::new();
        at.stream_fragments(&self.input, &mut |_: &FragmentKey, bytes: &[u8]| {
            absorb_groups(&self.grouping, &mut totals, bytes)
        })?;
        Ok(BlockOutput::fragment(
            self.output.clone(),
            self.encode_owned(&totals, at.grid, at.index)?,
        ))
    }
}

/// The two phases, on one lattice, appended to a plan that already has some.
///
/// Both are built with `fragment_phase`, so both halos come from the ops'
/// declarations rather than from this function: zero for the grouping, whose
/// input is one block's rows, and the whole lattice for the merge, whose reach
/// is the dependency edge that makes every block's partial available to every
/// other one.
///
/// Neither phase writes an image, so `check_dtypes` skips both.
///
/// Returns the plan and the phase index the **grouped rows** are keyed under,
/// which is where a reader has to look for them: a stream written by two phases
/// holds two generations, so the phase is half the address.
pub fn append_group_phases(
    mut plan: Decomposition,
    group: &GroupRowsOp,
    merge: &MergeGroupsOp,
) -> Result<(Decomposition, usize)> {
    let grid = plan
        .phases
        .last()
        .ok_or_else(|| {
            Error::invalid(
                "the grouped reduction is appended to a plan that already has a phase, because \
                 the lattice is inherited and because its input is a row stream some earlier \
                 phase must have written — fragments are keyed by block index, so a phase \
                 reading another's fragments on a different lattice would address blocks that \
                 correspond to nothing.",
            )
        })?
        .grid
        .clone();
    plan.phases.push(fragment_phase(group, grid.clone())?);
    plan.phases.push(fragment_phase(merge, grid)?);
    let rows_phase = plan.phases.len() - 1;
    plan.check()?;
    Ok((plan, rows_phase))
}

/// One group's row, decoded — the reader's half of [`Grouping::output`].
///
/// **Both forms of a sum are carried**, for `ops::tabulate`'s reason: the
/// fixed-point integer is what the invariance claim is about and the `f64` is
/// what a caller filters on, so offering only the float would hand back a number
/// whose exactness a reader would have to take on trust.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupValues {
    /// The key tuple, in the grouping's own key order.
    pub key: Vec<u64>,
    /// The group's least coordinate, which is where its row sits.
    pub at: [usize; 3],
    /// Rows of the group, whatever any column of them held.
    pub rows: u64,
    /// One per reduction, in the grouping's order.
    ///
    /// **An empty selection reads as zero**, and that is a convention rather
    /// than an answer: a group in which no row was present has no least value,
    /// no greatest and no first, and `Table` has no spelling for the absence
    /// because it refuses a non-finite `F64`. A [`Aggregate::Count`] over the
    /// same column is how to tell a group that selected nothing from one whose
    /// least value really is zero — the same pairing `ops::tabulate`'s
    /// `RegionValues::min` and `RegionValues::all_nonfinite` make.
    pub values: Vec<Value>,
}

impl GroupValues {
    /// The fixed-point sum of reduction `index`, in the value column's own
    /// units.
    ///
    /// Fails naming the reduction if it is not a [`Aggregate::Sum`], because a
    /// caller reading a count as a sum would get a plausible number.
    pub fn sum(&self, grouping: &Grouping, index: usize) -> Result<f64> {
        let reduction = grouping.reductions().get(index).ok_or_else(|| {
            Error::invalid(format!(
                "reduction {index} was asked for and this grouping has {}",
                grouping.reductions().len()
            ))
        })?;
        if reduction.aggregate != Aggregate::Sum {
            return Err(Error::invalid(format!(
                "reduction {index} is a {} and not a sum, so there is no fixed-point word to \
                 rescale.",
                reduction.aggregate.prefix()
            )));
        }
        match self.values[index] {
            Value::U64(word) => Ok(grouping
                .fixed()
                .value_of(grouping.fixed().from_column(word))),
            other => Err(Error::invalid(format!(
                "reduction {index} is a sum and its column holds {other:?}"
            ))),
        }
    }
}

/// One row of a grouped table, decoded.
pub fn group_values(grouping: &Grouping, row: &Row<'_>) -> Result<GroupValues> {
    let arity = grouping.key().len();
    let mut key = Vec::with_capacity(arity);
    for column in 0..arity {
        key.push(row.u64(column)?);
    }
    let rows = row.u64(arity)?;
    let mut values = Vec::with_capacity(grouping.reductions().len());
    for index in 0..grouping.reductions().len() {
        values.push(row.value(arity + 1 + index)?);
    }
    Ok(GroupValues {
        key,
        at: row.at(),
        rows,
        values,
    })
}

/// Every grouped row of a sealed stream, in the canonical order.
///
/// [`collect_rows`] with the decode attached, so a caller does not have to know
/// which column index a key sits at.
pub fn collect_groups(
    env: &dyn Environment,
    stream: &str,
    phase: usize,
    volume: [usize; 3],
    grouping: &Grouping,
) -> Result<Vec<GroupValues>> {
    let mut table = Table::new(volume, grouping.output().clone())?;
    fold_fragments(env, stream, &mut |key, bytes| {
        if key.phase != phase {
            return Ok(());
        }
        table.write(key.block, bytes)
    })?;
    table.seal()?;
    let mut found = Vec::with_capacity(table.len());
    for row in table.scan(&Region::whole(&volume))? {
        found.push(group_values(grouping, &row)?);
    }
    Ok(found)
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

    /// The gathered column is the image's value at the row's own coordinate, and
    /// the schema is the input's plus one.
    #[test]
    fn a_gather_reads_the_image_at_the_rows_coordinate() {
        let rows = vec![row([1, 2, 3], 0, 0.0), row([7, 0, 0], 1, 0.0)];
        let out = gather_blob(
            VOLUME,
            &schema(),
            &blob(&rows),
            "image",
            &Region::whole(&VOLUME),
            &ramp(),
            [0, 0, 0],
        )
        .expect("the rows are inside");
        let gathered = gathered_schema(&schema(), "image").expect("a fresh name");
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
                "image",
                region,
                &pixels,
                [0, 0, 0],
            )
            .expect("its own row is inside its own region");
            let gathered = gathered_schema(&schema(), "image").expect("a fresh name");
            assert_eq!(merged(VOLUME, gathered, &[out]).len(), 1);

            // And the other side's row is refused rather than answered, which is
            // what "exactly one" means: there is no reading of it that produces
            // two values for one row.
            let not_ours = blob(std::slice::from_ref(&boundary[theirs]));
            assert!(gather_blob(
                VOLUME,
                &schema(),
                &not_ours,
                "image",
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
            "image",
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

    /// Two images an `f64` column cannot carry exactly are refused rather than
    /// rounded; every narrower one converts with no loss.
    #[test]
    fn an_image_too_wide_for_an_f64_column_is_refused() {
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

    /// An unwritten image is NaN so that an absence cannot pass for a value;
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
        assert!(gathered_schema(&schema(), "image").is_ok());
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
        let gather = GatherRowsOp::new("gather", rows(), 0, "image").expect("a fresh name");
        let ops: [&dyn FragmentOp; 3] = [&scale, &filter, &gather];
        for op in ops {
            for axis in 0..3 {
                assert_eq!(op.reach(axis, VOLUME[axis]), 0, "{}", op.name());
            }
            assert_eq!(op.inputs()[0].reach, [0, 0, 0], "{}", op.name());
            assert!(!op.writes_pixels(), "{}", op.name());
            // **None of the three reads the image its phase is handed**, the
            // gather included: the array a gather samples is a *second* one, and
            // it says so with `source_inputs` rather than by taking image `p`.
            // See the module header — that is the whole reason the shell can be
            // written, and it costs one array rather than two.
            assert!(!op.reads_pixels(), "{}", op.name());
        }
        assert_eq!(scale.source_inputs(VOLUME), Vec::<SourceInput>::new());
        assert_eq!(filter.source_inputs(VOLUME), Vec::<SourceInput>::new());
        assert_eq!(
            gather.source_inputs(VOLUME),
            vec![SourceInput::voxelwise(0)],
            "the gather names its image, at exactly the extent the block owns"
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
        let gather = GatherRowsOp::new("gather", rows, 0, "image").expect("a fresh name");
        assert_eq!(gather.seam_fold(), Some(SeamFold::PerBlock));
        assert_eq!(gather.inputs().len(), 1);
        assert_eq!(gather.inputs()[0].reach, [0, 0, 0]);
        assert_eq!(gather.outputs()[0].coverage, Coverage::EveryBlock);
        // The output schema is known before any block runs, which is what a
        // consumer planning against this phase needs.
        assert_eq!(gather.schema().len(), schema().len() + 1);
        assert_eq!(gather.schema().index_of("image"), Some(2));
        assert_eq!(gather.image(), 0);
        assert_eq!(gather.column(), "image");
    }

    #[test]
    fn a_gather_onto_an_existing_column_is_refused_at_construction() {
        let rows = || {
            RowStreams::new("in", 0, "out", Lifecycle::DeleteOnExit, schema()).expect("two names")
        };
        assert!(GatherRowsOp::new("gather", rows(), 0, "score").is_err());
        assert!(GatherRowsOp::new("gather", rows(), 0, "image").is_ok());
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
            "image",
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
            "image",
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
            "image",
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
