// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation —
// one row per set voxel, in the volume's own walk order — not adapted from any
// implementation of it.
//
// **A mask in, the coordinate of every set voxel out.**
//
// The operation is the smallest one in `ops/` and its specification is almost
// entirely about something other than which voxels it returns. What voxels it
// returns is trivial: the set ones. What it has to get right is the **order**,
// because the consumers of this list address its entries **by position** — an
// entry's index in the list is the only name it has. Two runs that produce the
// same coordinates in two orders have produced two different answers, and both
// of them are well-formed lists that pass any check phrased as a set.
//
// So the whole of this module is one claim:
//
// > the list a run produces is **the same list, in the same order**, whatever
// > lattice the volume was cut into — and it is the list a single-block walk
// > would produce.
//
// The order, stated
// -----------------
// The single-block walk visits `(i, j, k)` with axis 0 slowest and axis 2
// fastest, so the list it produces is the set voxels sorted **lexicographically
// on the coordinate triple**. That is the definition used here, and it is worth
// writing as a sort rather than as a loop nest, because writing it as a sort is
// what makes the next section's argument visible.
//
// Why a base index per block cannot express it
// ---------------------------------------------
// The obvious mechanism, and the one to reach for first, is a **prefix sum over
// per-block counts**: every block counts its set voxels, the counts are summed
// in block order to give each block a base index, and a block writes its
// coordinates starting there. It is cheap, it needs one scalar per block, and
// it is **wrong** — not awkward, not approximate, wrong — for any lattice that
// is cut on more than the slowest axis.
//
// A base index per block can only ever express a **block-major** order: every
// entry of block `B` before every entry of block `C`, for `B` before `C`.
// Lexicographic order on coordinates is not block-major, because the blocks
// *interleave* under it. The smallest case is a `[2, 4, 1]` volume cut
// `[2, 2, 1]`, which is two blocks side by side on axis 1:
//
// | voxel | block | lexicographic rank | block-major rank |
// |---|---|---|---|
// | `[0, 2, 0]` | `[0, 1, 0]` | 0 | 1 |
// | `[1, 0, 0]` | `[0, 0, 0]` | 1 | 0 |
//
// `[0, 2, 0]` is in the *second* block and comes *first*, because axis 0 leads
// the comparison and both blocks span the whole of it. No assignment of a base
// index to each block puts those two entries in that order, so there is no
// prefix sum to get right here — the mechanism itself does not reach.
//
// [`blocks_concatenate_in_order`] is exactly when it does reach, computed from
// the lattice rather than asserted: a lattice cut only on the slowest axis
// makes the block ranges of axis 0 disjoint and ordered, so block-major *is*
// lexicographic. That is a real and useful case — it is the shape a caller
// picks when it wants the cheap merge — and the point of naming it is that a
// caller who takes it takes it **deliberately**. A lattice that misses it by a
// single block silently permutes the entire list, and every downstream answer
// stays well-formed while every entry changes its name. That is the failure
// this module is arranged against.
//
// What is used instead: the order is in the payload
// --------------------------------------------------
// An entry's index is its rank in the lexicographic order of the whole set, and
// the coordinates **determine** that rank. Nothing else is needed to recover
// it: not a base index, not a count from a neighbour, not the order the blocks
// finished in. So the writing phase communicates nothing at all — reach 0, no
// fragment inputs, no barrier before any block writes — and the order is
// restored at the merge by sorting on the coordinate.
//
// `crate::table` is where that sort already lives, and it is why the rows go
// there rather than into `crate::points`. Its canonical order is *the
// lexicographic order of the coordinate triple, then the payload words*, held
// by the store rather than promised by a query. With no payload columns, the
// canonical order **is** the single-block walk, so [`merge_coordinates`] is
// three lines and the ordering claim is `table.rs`'s claim rather than a second
// one made here. It follows that the merge is insensitive to the order the
// blobs arrive in, which is the property a block-completion order would break
// and the one `tests/coordinate_list.rs` drives directly.
//
// The rows carry **no payload column at all**, and that is a decision.
// `crate::points` would have been the other home, and a `Point` has a weight —
// a number this operation does not have and would have to invent. A weight of
// `1.0` in the output that is not in the input is a value a consumer can read
// and be misled by. A three-word row is a coordinate and nothing else, which is
// what was asked for.
//
// What it costs
// -------------
// The merge sorts `N` rows, where `N` is the number of set voxels, and holds
// them while it does. That is worth comparing against the prefix sum honestly
// rather than dismissing, because the prefix sum is cheaper *per block* — one
// scalar against one coordinate per set voxel.
//
// * **The prefix sum costs a barrier.** The merge has to know every block's
//   count before any block writes its coordinates. In this crate's machinery
//   that is not a cheaper version of this phase, it is a different plan: a
//   `volume -> fragments` counting phase, a whole-lattice `fragments ->
//   fragments` phase to sum the counts, and a *second* pass over the volume to
//   write the coordinates against the bases. The volume is read twice and there
//   is a full-reach phase in the middle, which `fragment.rs` describes as a
//   reboot — nothing fuses across it and no cache state survives it.
// * **The sort costs `O(N log N)` once, at the reader.** The volume is read
//   once, the phase has no halo, no reach and no barrier, and every block is
//   independent. `N` is the set voxels rather than the volume.
//
// So the sort is not the price of being right here; it is cheaper on the axis
// that dominates — passes over the volume — as well as being the only one of
// the two that is right on an arbitrary lattice. What it does cost is
// **residency at the merge**: the whole list is in memory at once. That is
// inherent to the request rather than to the mechanism — the answer *is* the
// whole list, and a consumer that addresses entries by position needs it all —
// but it is a real bound and it is stated here rather than discovered later.
//
// Blocks that find nothing
// ------------------------
// A block with no set voxel in it writes a header and no rows, and the output
// stream declares [`Coverage::EveryBlock`] so that it does. Present-and-empty
// is a different fact from absent, and it is the one the coverage guard can
// check — this phase writes no pixel image, so the tiling check has nothing to
// bite on and the coverage declaration is the only guard there is.
//
// Such a block contributes nothing to the list **and shifts nobody's index**,
// and that is a property of the mechanism rather than a case handled: there is
// no base index for an empty block to have got wrong. Under the prefix sum it
// would be a case, and a case that a fixture with no empty block never reaches.
//
// Duplicates and coverage
// -----------------------
// The cores of a lattice tile the volume with no overlap and no gap, and each
// block walks its own core, so every set voxel is written exactly once by
// exactly one block. The merge does not deduplicate and does not need to.
// [`crate::table::Table::write`] refuses a coordinate outside the volume, so a
// block that somehow wrote outside its lattice is caught at the merge rather
// than sorted quietly into the middle of the answer.

use std::collections::BTreeMap;
use std::sync::Arc;

use ndarray::{s, ArrayView3};

use crate::assemble::{Phase, PlanBuilder};
use crate::decomposition::Decomposition;
use crate::dtype::Dtype;
use crate::env::{BlockBuf, Environment};
use crate::error::{Error, Result};
use crate::fragment::{
    fold_fragments, fragment_phase, BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput,
};
use crate::geometry::BlockGrid;
use crate::region::Region;
use crate::sidecar::Lifecycle;
use crate::table::{Column, RowBuilder, Schema, Table, Value};

use super::fill::as_mask;

// -------------------------------------------------------------- the schema --

/// The schema a coordinate blob carries: **no payload columns**.
///
/// A row is its three position words and nothing else, which is what a
/// coordinate is. The module header says why this is a `crate::table` row
/// rather than a `crate::points` point: a point has a weight, and a weight this
/// operation does not have would be a number in the output that is not in the
/// input.
///
/// The blob is still self-describing in the way that matters — it carries the
/// table magic and a column count of zero — so a consumer handed a stream from
/// a different producer is refused rather than reading its words as positions.
/// What it cannot distinguish is a *different* zero-column table, and there is
/// only one of those.
pub fn coordinate_schema() -> Schema {
    // No columns, so none can be unnamed and none can repeat.
    Schema::new(Vec::new()).expect("a schema with no columns names nothing twice")
}

/// The schema of a **valued** coordinate table: one `f64` column, named by the
/// caller.
///
/// [`coordinate_schema`] answers "which voxels are set" and has no columns at
/// all, deliberately — the module header's argument that the merged order is a
/// function of the coordinate set alone rests on there being no payload to
/// tiebreak on. This answers a different question, *"which voxels are set and
/// what does the image hold there"*, and pays for it in exactly one place: two
/// rows on one coordinate would now tiebreak on the value's bits, and no two
/// rows here share a coordinate because a voxel is in exactly one core.
///
/// **Why `f64` and not the image's own type.** The value is read through
/// [`Voxels::widened`], which is what every other consumer of an arbitrary
/// element type in this crate does, so one column type serves every input dtype
/// and a consumer's schema does not change when the input's does. A column that
/// followed the input would make a downstream `Grouping` depend on a decision
/// taken upstream of it.
///
/// [`Voxels::widened`]: crate::voxels::Voxels::widened
pub fn valued_coordinate_schema(column: &str) -> Result<Schema> {
    Schema::new(vec![Column::f64(column)])
}

/// Push one row per set voxel of `values`, carrying that voxel's value.
///
/// The **same walk** as [`set_voxels_into`] — axis 0 slowest, axis 2 fastest —
/// and set means the same thing it means there: not zero. Written beside it
/// rather than sharing a body, because sharing would mean a closure per row on
/// the path that has no payload at all, and that path is the one every
/// `ops::coordinates` consumer is on.
pub fn set_voxel_values_into(
    values: ArrayView3<'_, f64>,
    offset: [usize; 3],
    rows: &mut RowBuilder,
) -> Result<()> {
    let shape = [values.shape()[0], values.shape()[1], values.shape()[2]];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let value = values[[i, j, k]];
                if value != 0.0 {
                    rows.push(
                        [offset[0] + i, offset[1] + j, offset[2] + k],
                        &[Value::F64(value)],
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// One array's set voxels and their values as the bytes a block writes.
pub fn encode_set_voxel_values(
    values: ArrayView3<'_, f64>,
    offset: [usize; 3],
    column: &str,
) -> Result<Vec<u8>> {
    let mut rows = RowBuilder::new(Arc::new(valued_coordinate_schema(column)?));
    set_voxel_values_into(values, offset, &mut rows)?;
    Ok(rows.encode())
}

/// A blob with no rows in it: what a block that found no set voxel writes.
///
/// Present and empty rather than absent — see the module header, and
/// [`Coverage::EveryBlock`], which is what makes the difference checkable.
pub fn empty_coordinates() -> Vec<u8> {
    RowBuilder::new(Arc::new(coordinate_schema())).encode()
}

// -------------------------------------------------------------- the walk --

/// Push one row per set voxel of `mask` into `rows`, in the walk order.
///
/// Axis 0 slowest, axis 2 fastest — the order the module header defines as the
/// specification. `offset` is where `mask`'s lowest voxel sits in the volume, so
/// the coordinates written are **volume** coordinates; block-local ones would
/// make the answer a function of where the blocks were, which is the whole thing
/// being avoided.
///
/// The one kernel, called by the whole-volume reference and by a block alike.
/// Written once so that a disagreement between them is a decomposition bug and
/// not a modelling difference.
pub fn set_voxels_into(
    mask: ArrayView3<'_, bool>,
    offset: [usize; 3],
    rows: &mut RowBuilder,
) -> Result<()> {
    let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if mask[[i, j, k]] {
                    rows.push([offset[0] + i, offset[1] + j, offset[2] + k], &[])?;
                }
            }
        }
    }
    Ok(())
}

/// The whole-volume answer: every set voxel's coordinate, in the walk order.
///
/// **Not a second implementation.** It is what the blocked path is measured
/// against, so if it were written a second way a disagreement would be a
/// modelling difference rather than a decomposition bug. `pub` because the
/// acceptance suite in `tests/` is a separate crate and needs exactly this.
pub fn set_voxels(mask: ArrayView3<'_, bool>) -> Vec<[usize; 3]> {
    let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    let mut found = Vec::new();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if mask[[i, j, k]] {
                    found.push([i, j, k]);
                }
            }
        }
    }
    found
}

/// One array's set voxels as the bytes a block writes.
///
/// `offset` is the array's lowest voxel in volume coordinates; the whole-volume
/// reference passes `[0, 0, 0]` and a block passes its core's corner.
pub fn encode_set_voxels(mask: ArrayView3<'_, bool>, offset: [usize; 3]) -> Result<Vec<u8>> {
    let mut rows = RowBuilder::new(Arc::new(coordinate_schema()));
    set_voxels_into(mask, offset, &mut rows)?;
    Ok(rows.encode())
}

/// The whole-volume answer as the **blob** a single-block run would write.
///
/// The byte-level reference, for a suite that wants to compare what landed
/// rather than what was decoded from it.
pub fn set_voxel_rows(mask: ArrayView3<'_, bool>) -> Result<Vec<u8>> {
    encode_set_voxels(mask, [0, 0, 0])
}

// -------------------------------------------------------------- the merge --

/// Every block's blob, in the walk order, as one list.
///
/// **This is where the order is restored, and it is restored from the
/// coordinates rather than from anything about the run.** The blobs go into one
/// [`Table`] over `volume`, which holds its rows in the canonical order —
/// lexicographic on the coordinate triple, and with no payload column that is
/// the whole of it. So the result is a function of the coordinate set alone: not
/// of the lattice, not of which block finished first, not of the order the
/// blobs are handed over. `merging_is_insensitive_to_the_order_blobs_arrive`
/// asserts that directly by feeding the same blobs backwards.
///
/// The block index is carried only so that a refusal can name the blob it came
/// from; [`Table::write`] keeps no trace of it, which is what makes an order
/// that tiebreaks on the source block unrepresentable rather than discouraged.
///
/// A coordinate outside `volume` is refused rather than sorted into the answer.
pub fn merge_coordinates<'a>(
    volume: [usize; 3],
    blobs: impl IntoIterator<Item = ([usize; 3], &'a [u8])>,
) -> Result<Vec<[usize; 3]>> {
    let mut table = Table::new(volume, coordinate_schema())?;
    for (block, bytes) in blobs {
        table.write(block, bytes)?;
    }
    ordered_rows(&mut table, volume)
}

/// [`merge_coordinates`] over a stream in a store.
///
/// Streams the fragments one at a time — `Environment::sidecar_fragments` would
/// make every blob resident on top of the table that is about to hold their
/// rows, which doubles the one residency this operation has. `phase` is half the
/// address: a stream written by two phases holds two generations, and a blob
/// from the wrong one would decode perfectly and answer differently.
pub fn collect_coordinates(
    env: &dyn Environment,
    stream: &str,
    phase: usize,
    volume: [usize; 3],
) -> Result<Vec<[usize; 3]>> {
    let mut table = Table::new(volume, coordinate_schema())?;
    fold_fragments(env, stream, &mut |key, bytes| {
        if key.phase != phase {
            return Ok(());
        }
        table.write(key.block, bytes)
    })?;
    ordered_rows(&mut table, volume)
}

fn ordered_rows(table: &mut Table, volume: [usize; 3]) -> Result<Vec<[usize; 3]>> {
    table.seal()?;
    let mut found = Vec::with_capacity(table.len());
    // A loop rather than a `collect` on the tail expression: the scan borrows
    // the table, and a borrow in a function's final expression outlives the
    // local it is taken from.
    for row in table.scan(&Region::whole(&volume))? {
        found.push(row.at());
    }
    Ok(found)
}

// ------------------------------------------------------- the cheap merge --

/// Whether concatenating the blocks **in block-index order** gives the walk
/// order, for every mask over this lattice.
///
/// This is the precondition of the prefix-sum mechanism the module header
/// rejects as a general answer, stated as a computation on the lattice so that a
/// caller can take it deliberately instead of discovering it. When this is true,
/// a base index per block is enough and no sort is needed; when it is false, no
/// assignment of base indices can produce the walk order at all.
///
/// The derivation. Block-index order is lexicographic on `(a, b, c)`, and the
/// walk order is lexicographic on `(i, j, k)`. Two blocks that differ first on
/// axis `a` occupy disjoint, correctly ordered ranges of that axis — so entries
/// of the earlier block come first **provided** no axis before `a` can decide
/// the comparison the other way. It cannot when every block spans exactly one
/// voxel of every earlier axis, and it can otherwise: take the largest earlier
/// coordinate in the first block against the smallest in the second. So the
/// condition is: for every axis that is cut into more than one block, every axis
/// before it is one voxel wide within a block.
///
/// The common case that satisfies it is a lattice cut **only on the slowest
/// axis** — a stack of full cross-sections. The degenerate ones are volumes
/// whose leading axes are a single voxel, and they are admitted rather than
/// excluded because the derivation admits them and a conservative answer here
/// would refuse a lattice that works.
pub fn blocks_concatenate_in_order(grid: &BlockGrid) -> bool {
    let counts = grid.blocks_per_axis();
    let edge = grid.block();
    let volume = grid.volume();
    for axis in 1..3 {
        if counts[axis] <= 1 {
            continue;
        }
        for earlier in 0..axis {
            // The widest any block is on `earlier`; the last one may be shorter,
            // and a lattice where even the widest is one voxel has them all so.
            if edge[earlier].min(volume[earlier]) > 1 {
                return false;
            }
        }
    }
    true
}

/// The base index of every block, from the per-block counts — **refused on a
/// lattice where a base index cannot express the walk order**.
///
/// The prefix sum, offered with its precondition made mechanical. It is here
/// because it is genuinely the cheaper merge where it applies — one scalar per
/// block against one coordinate per set voxel — and because offering it without
/// the guard is the failure mode the module header is about: on a lattice that
/// interleaves, this returns bases that produce a complete, well-formed,
/// entirely permuted list, and nothing downstream can tell.
///
/// `counts` must hold every block of `grid`; a missing block is refused rather
/// than assumed empty, because "absent" and "present with nothing to say" are
/// different facts and only one of them is a block that ran.
///
/// The bases are summed in block-index order, which is the order
/// [`blocks_concatenate_in_order`] has just certified is the walk order.
pub fn block_base_indices(
    grid: &BlockGrid,
    counts: &BTreeMap<[usize; 3], usize>,
) -> Result<BTreeMap<[usize; 3], usize>> {
    if !blocks_concatenate_in_order(grid) {
        return Err(Error::invalid(format!(
            "a base index per block cannot express the walk order on a lattice of {:?} block(s) \
             of {:?} over {:?}. A base index can only ever put every entry of one block before \
             every entry of the next, and the walk order interleaves them: axis 0 leads the \
             comparison, so a voxel low on axis 0 in a later block precedes a voxel high on \
             axis 0 in an earlier one. The bases would be a complete, well-formed, entirely \
             permuted numbering that nothing downstream could tell from the right one. Cut only \
             the slowest axis, or merge by coordinate with `merge_coordinates`, which is \
             correct on every lattice.",
            grid.blocks_per_axis(),
            grid.block(),
            grid.volume()
        )));
    }
    let mut bases = BTreeMap::new();
    let mut running = 0usize;
    for core in grid.cores() {
        let index = core.index;
        let count = counts.get(&index).ok_or_else(|| {
            Error::invalid(format!(
                "no count was given for block {index:?} of a lattice of {:?}. A block that found \
                 nothing has a count of zero, which is a different fact from a block that did \
                 not report, and only the first can be summed.",
                grid.blocks_per_axis()
            ))
        })?;
        bases.insert(index, running);
        running = running.checked_add(*count).ok_or_else(|| {
            Error::invalid(
                "the running total of set voxels overflowed a usize, which needs more voxels \
                 than the address space holds"
                    .to_string(),
            )
        })?;
    }
    Ok(bases)
}

// -------------------------------------------------------------- the phase --

/// **A mask in, one row per set voxel out**, at reach 0.
///
/// Reads its block's pixels, writes one fragment and no image. Every coordinate
/// it writes is a **volume** coordinate and lies in this block's core, so the
/// blobs of a run partition the answer; the order they are put back into is
/// [`merge_coordinates`]', and is a function of the coordinates rather than of
/// this op.
///
/// **Reach 0, and it is the honest number rather than a convenient one.** A set
/// voxel is a fact about one voxel, so nothing outside the block is read, no
/// halo is declared and no block depends on another. That is what lets the
/// phase carry no barrier at all, which is the half of the module header's
/// argument that is about cost rather than correctness.
pub struct SetVoxelsOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    /// The value column's name, when one was asked for. See
    /// [`Self::with_values`].
    value_column: Option<String>,
}

impl SetVoxelsOp {
    /// `lifecycle` is the output stream's and is not defaulted:
    /// [`Lifecycle::Persistent`] is the usual answer because this stream **is**
    /// the run's output, but what a run leaves behind is the caller's decision.
    pub fn new(name: &'static str, stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
            value_column: None,
        }
    }

    /// Also carry each voxel's **own value**, in a column of this name.
    ///
    /// **A column, not a mechanism.** Everything this op already decides is
    /// unchanged: reach 0, [`Coverage::EveryBlock`], and the ownership rule that
    /// a voxel belongs to exactly one core and its coordinate is a volume
    /// coordinate. What changes is the schema, from [`coordinate_schema`] to
    /// [`valued_coordinate_schema`], and what a row carries beside its position.
    ///
    /// It exists because a plan that wants to *filter or group rows by something
    /// the image holds* had no producer at all: this op's rows had no payload,
    /// and `ops::rows::RowSourceOp` takes a table the caller already has. So a
    /// suite testing a row filter had nothing to filter **on**, and wrote its own
    /// producer to get one — see `docs/ops-survey/README.md`, G19, and note that
    /// the test reporting it said so in its own words.
    ///
    /// **Not a substitute for `GatherRowsOp`, and the distinction is the useful
    /// half.** This reads the value out of *the image the rows came from*, at the
    /// voxel the row is about, so where those are the same array no gather is
    /// needed at all. A gather samples a **second** array at rows that already
    /// exist, which is the other case and the one a consumer usually has.
    pub fn with_values(mut self, column: impl Into<String>) -> Self {
        self.value_column = Some(column.into());
        self
    }

    /// The stream the coordinate rows are written to.
    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// The schema of the rows this writes — which a consumer needs before any
    /// block has run, to plan against.
    pub fn schema(&self) -> Result<Schema> {
        match &self.value_column {
            Some(column) => valued_coordinate_schema(column),
            None => Ok(coordinate_schema()),
        }
    }
}

impl FragmentOp for SetVoxelsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Nothing crosses this block's boundary. The op reports the set voxels of
    /// its own core, and a voxel is in exactly one core — no answer here is a
    /// function of a neighbour, so there is nothing to shrink off the read
    /// extent.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            // Every block, always — and here it is the only guard there is,
            // since this phase writes no image for the tiling check to bite on.
            // A block that found no set voxel writes a header and no rows, which
            // is present and therefore checkable.
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = at.pixels()? else {
            // An accounting run has no data. It still writes a fragment, because
            // what it is measuring is the IO, and a phase that silently produced
            // nothing would measure a different program.
            return Ok(BlockOutput::fragment(
                self.stream.clone(),
                match &self.value_column {
                    Some(column) => {
                        RowBuilder::new(Arc::new(valued_coordinate_schema(column)?)).encode()
                    }
                    None => empty_coordinates(),
                },
            ));
        };
        let mask = as_mask(pixels)?;

        // The pixels cover the **read** extent, which for this op is the core —
        // reach 0 and no fragment input, so `fragment_phase` derives no halo.
        // Sliced anyway rather than trusted: a coordinate written from a halo
        // would be a duplicate in the merge, and the two lines that make that
        // impossible are cheaper than the argument that it cannot happen.
        let mut origin = [0usize; 3];
        let mut extent = [0usize; 3];
        for axis in 0..3 {
            origin[axis] = at.core.start[axis] - at.read.start[axis];
            extent[axis] = at.core.shape[axis];
        }
        let core = mask.slice(s![
            origin[0]..origin[0] + extent[0],
            origin[1]..origin[1] + extent[1],
            origin[2]..origin[2] + extent[2],
        ]);
        let offset = [at.core.start[0], at.core.start[1], at.core.start[2]];
        if let Some(column) = &self.value_column {
            // The same core, sliced the same way, read as values rather than as
            // a mask. `widened` is what makes one column type serve every input
            // dtype; see `valued_coordinate_schema`.
            let widened = pixels.widened();
            let values = widened.slice(s![
                origin[0]..origin[0] + extent[0],
                origin[1]..origin[1] + extent[1],
                origin[2]..origin[2] + extent[2],
            ]);
            return Ok(BlockOutput::fragment(
                self.stream.clone(),
                encode_set_voxel_values(values, offset, column)?,
            ));
        }
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            encode_set_voxels(core, offset)?,
        ))
    }
}

/// The one phase, as a whole `Decomposition`.
///
/// The op declares no reach and no fragment input, so `fragment_phase` derives
/// no halo and every block's valid region is its core. The phase declares no
/// element type because it writes no image, and `check_dtypes` skips it for that
/// reason.
pub fn set_voxels_phase(
    grid: BlockGrid,
    mask_dtype: Dtype,
    op: &SetVoxelsOp,
) -> Result<Decomposition> {
    let volume = grid.volume();
    let plan = Decomposition {
        volume,
        dtype: mask_dtype,
        phases: vec![fragment_phase(op, grid)?],
        chain_reach: [0, 0, 0],
    };
    plan.check()?;
    Ok(plan)
}

/// The same phase, **appended to a plan that already has some**.
///
/// [`set_voxels_phase`] builds a whole `Decomposition`, so it is unusable as
/// soon as this phase sits inside something larger; this is its body against a
/// builder. Returns the phase the blobs are keyed under, which is where a reader
/// has to look for them — see [`collect_coordinates`], whose `phase` argument
/// this is.
pub fn append_to(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
) -> Result<Phase> {
    plan.fragments(SetVoxelsOp::new("set voxels", stream, lifecycle))
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array3;

    fn mask_of(shape: [usize; 3], set: &[[usize; 3]]) -> Array3<bool> {
        let mut mask = Array3::from_elem((shape[0], shape[1], shape[2]), false);
        for &at in set {
            mask[at] = true;
        }
        mask
    }

    /// The walk order is lexicographic, which is the statement the whole module
    /// rests on. Asserted against a sort rather than against a second loop nest,
    /// so that it is a claim about the order and not a copy of the code.
    #[test]
    fn the_walk_is_the_lexicographic_order_of_the_set() {
        let mask = mask_of(
            [4, 4, 4],
            &[[3, 0, 0], [0, 3, 0], [0, 0, 3], [1, 2, 1], [0, 0, 0]],
        );
        let found = set_voxels(mask.view());
        let mut sorted = found.clone();
        sorted.sort_unstable();
        assert_eq!(found, sorted);
        assert_eq!(
            found,
            vec![[0, 0, 0], [0, 0, 3], [0, 3, 0], [1, 2, 1], [3, 0, 0]]
        );
    }

    /// A round trip through the blob and the merge, on one block, changes
    /// nothing — the floor the decomposition tests build on.
    #[test]
    fn one_blob_merges_back_to_the_walk() {
        let mask = mask_of([4, 5, 6], &[[0, 4, 5], [2, 0, 0], [3, 3, 3]]);
        let blob = set_voxel_rows(mask.view()).unwrap();
        let merged = merge_coordinates([4, 5, 6], [([0, 0, 0], blob.as_slice())]).unwrap();
        assert_eq!(merged, set_voxels(mask.view()));
    }

    /// A blob with no rows is a blob, and it contributes nothing.
    #[test]
    fn an_empty_blob_is_well_formed_and_adds_nothing() {
        let empty = empty_coordinates();
        assert_eq!(
            crate::table::encoded_schema(&empty).unwrap(),
            coordinate_schema()
        );
        let mask = mask_of([2, 2, 2], &[[1, 1, 1]]);
        let full = set_voxel_rows(mask.view()).unwrap();
        let merged = merge_coordinates(
            [2, 2, 2],
            [
                ([0, 0, 0], empty.as_slice()),
                ([0, 0, 1], full.as_slice()),
                ([0, 1, 0], empty.as_slice()),
            ],
        )
        .unwrap();
        assert_eq!(merged, vec![[1, 1, 1]]);
    }

    /// The smallest lattice on which a base index cannot express the walk order,
    /// worked through in full. This is the module header's table, executable.
    #[test]
    fn the_smallest_interleaving_lattice_defeats_a_base_index() {
        let grid = BlockGrid::new([2, 4, 1], [2, 2, 1]).unwrap();
        assert_eq!(grid.blocks_per_axis(), [1, 2, 1]);
        assert!(!blocks_concatenate_in_order(&grid));

        let mask = mask_of([2, 4, 1], &[[1, 0, 0], [0, 2, 0]]);
        // The walk puts the voxel of the *second* block first.
        assert_eq!(set_voxels(mask.view()), vec![[0, 2, 0], [1, 0, 0]]);

        // And the prefix sum refuses rather than producing the other order.
        let counts = BTreeMap::from([([0, 0, 0], 1usize), ([0, 1, 0], 1usize)]);
        let error = block_base_indices(&grid, &counts).unwrap_err().to_string();
        assert!(error.contains("cannot express the walk order"), "{error}");
    }

    /// A lattice cut only on the slowest axis is the case the cheap merge is
    /// for, and the bases it gives are the running total in block order.
    #[test]
    fn a_slab_lattice_concatenates_and_gets_bases() {
        let grid = BlockGrid::new([8, 4, 4], [2, 4, 4]).unwrap();
        assert_eq!(grid.blocks_per_axis(), [4, 1, 1]);
        assert!(blocks_concatenate_in_order(&grid));

        let counts = BTreeMap::from([
            ([0, 0, 0], 3usize),
            ([1, 0, 0], 0usize),
            ([2, 0, 0], 5usize),
            ([3, 0, 0], 1usize),
        ]);
        let bases = block_base_indices(&grid, &counts).unwrap();
        assert_eq!(bases[&[0, 0, 0]], 0);
        assert_eq!(bases[&[1, 0, 0]], 3);
        // The empty block shifts nobody: the block after it starts where it did.
        assert_eq!(bases[&[2, 0, 0]], 3);
        assert_eq!(bases[&[3, 0, 0]], 8);
    }

    /// A missing block is refused rather than read as an empty one.
    #[test]
    fn a_block_that_did_not_report_is_not_an_empty_block() {
        let grid = BlockGrid::new([4, 2, 2], [2, 2, 2]).unwrap();
        let counts = BTreeMap::from([([0, 0, 0], 1usize)]);
        let error = block_base_indices(&grid, &counts).unwrap_err().to_string();
        assert!(error.contains("no count was given for block"), "{error}");
    }

    /// The predicate's derivation, on the cases that separate it from "cut only
    /// on axis 0".
    #[test]
    fn the_concatenation_predicate_follows_its_derivation() {
        // Cut on axis 1, and axis 0 is a single voxel per block: the earlier
        // axis cannot decide the comparison, so concatenation is the walk.
        let thin = BlockGrid::new([1, 8, 4], [1, 2, 4]).unwrap();
        assert!(blocks_concatenate_in_order(&thin));
        // Give the earlier axis two voxels and it can.
        let thick = BlockGrid::new([2, 8, 4], [2, 2, 4]).unwrap();
        assert!(!blocks_concatenate_in_order(&thick));
        // Cut on axis 2 needs *both* earlier axes to be one voxel wide.
        let deep = BlockGrid::new([1, 2, 8], [1, 2, 2]).unwrap();
        assert!(!blocks_concatenate_in_order(&deep));
        let flat = BlockGrid::new([1, 1, 8], [1, 1, 2]).unwrap();
        assert!(blocks_concatenate_in_order(&flat));
    }

    /// The phase declares no halo and no reach, so every valid region is a core.
    #[test]
    fn the_phase_has_no_halo_and_its_valid_regions_are_the_cores() {
        let op = SetVoxelsOp::new("set voxels", "coordinates", Lifecycle::DeleteOnExit);
        let grid = BlockGrid::new([8, 4, 4], [4, 4, 4]).unwrap();
        let phase = fragment_phase(&op, grid).unwrap();
        assert_eq!(phase.reach, [0, 0, 0]);
        assert_eq!(phase.halo, [0, 0, 0]);
        for block in &phase.blocks {
            assert_eq!(block.read, block.core);
            assert_eq!(block.valid, block.core);
        }
    }
}
