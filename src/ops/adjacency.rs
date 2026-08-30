// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation —
// one row per adjacent pair of set voxels, in the volume's own walk order — not
// adapted from any implementation of it.
//
// **A mask in, every adjacent pair of set voxels out.**
//
// This is the pair half of what [`super::coordinates`] is the single half of.
// That module emits one row per set voxel; this one emits one row per *pair* of
// set voxels that touch. The two compose: run both over the same mask and the
// second's rows name positions the first's rows hold, with no third thing
// needed to relate them.
//
// A pair does not need a name for its endpoints
// ---------------------------------------------
// The thing that makes this operation small is worth stating before the shape,
// because everything else follows from it.
//
// The obvious way to emit a pair is to number the set voxels — position 0,
// position 1, and so on in the walk order — and emit a pair as two of those
// numbers. It is compact, it is what a consumer that indexes a list wants in the
// end, and it makes this operation **hard**: a number in that scheme is a fact
// about the whole volume, so a block cannot compute one without knowing how many
// set voxels lie before its own. A pair that straddles a seam then names a
// position the block on the far side has not counted yet, and the phase needs a
// barrier, a peer read, or both.
//
// So do not emit the numbers. **Emit the two coordinates.** A coordinate is a
// fact about one voxel and a block already knows it; the numbering, if a
// consumer wants one, is the rank of the coordinate in the merged list and can
// be assigned once, at the end, by whoever wants it. Put as one sentence, since
// it is the whole design:
//
// > a pair crossing a seam is only hard when it names its endpoints by an index
// > into some global list, and such an index is a global fact that need not
// > exist until the merge.
//
// The shape that follows
// ----------------------
// A pair of adjacent voxels has a **lexicographically lower** endpoint and a
// higher one — they are distinct positions, so one of the two orders is the
// order. Emit the pair as a row at the lower endpoint carrying the higher one in
// three payload columns, and give it to **the block whose core holds the lower
// endpoint**. Then:
//
// * every pair is emitted **exactly once**, because exactly one block's core
//   holds any given position and the cores tile the volume with no overlap;
// * no block reads another block's fragments, so the phase has no fragment input
//   and no barrier;
// * the only thing a block needs beyond its core is the **one voxel** of context
//   around it, which is a read halo of 1 and nothing more.
//
// And the order is free. [`crate::table`]'s canonical order is the lexicographic
// order of the row's words — the position first, then the payload in schema
// order — which for these rows is exactly *(lower endpoint, then higher
// endpoint)*. The walk that produces them visits the lower endpoints in
// lexicographic order and, within one of them, the higher endpoints in
// lexicographic order too, because [`forward_offsets`] is sorted. So **the blob
// a block writes is already in the canonical order**, the merge is a sort and
// nothing else, and a single-block run's bytes are the merged bytes. That is
// `coordinates`' finding again and for the same reason: the output is a sort of
// itself, so there is no sort to get wrong.
//
// Why the coordinate pair is a complete name **here**
// ---------------------------------------------------
// A consumer of this operation is likely to go on to contract runs of these
// pairs into pairs of the positions at their ends, and at *that* stage a
// coordinate pair stops identifying anything: two distinct runs can join the
// same two positions, and a run can return to the position it started from, so a
// table keyed on the pair silently merges the first and drops the second. That
// is a real hazard and it has been paid for before.
//
// It does not arise here, and the reason is worth writing down so that nobody
// re-derives it under pressure. **Adjacency is a relation between two positions
// and nothing else.** Two given positions are adjacent or they are not — there
// is no second, different way for them to be adjacent — and no position is
// adjacent to itself, because a zero offset is not a step ([`Connectivity::
// joins`] says so). So on this operation's output the coordinate pair is a key:
// a duplicate row and a row whose two endpoints are equal are both *impossible*
// rather than merely unexpected, and the sweep asserts both rather than trusting
// this paragraph.
//
// The consequence is a decision not to act. There is no discriminating column
// here — no per-pair serial number, no counter — because at this stage there is
// nothing for it to discriminate, and a column that is always the same thing is
// a number in the output that is not in the input. The stage that contracts runs
// is where the discriminator belongs, since that is where two rows can genuinely
// be different and look identical, and it is the stage that can say what makes
// them different. Inventing one here would put the cost everywhere and the
// benefit nowhere.
//
// Connectivity, and a default that departs from the rest of the crate
// -------------------------------------------------------------------
// Which of the twenty-six surrounding voxels count as adjacent is a parameter,
// and it is [`Connectivity`] — the one this crate already has — rather than a
// neighbour list written out again here. The offsets are taken from
// [`Connectivity::offsets`] and filtered to the lexicographically positive half,
// which is precisely the half that steps from a lower endpoint to a higher one.
//
// **The default here is [`Connectivity::FacesEdgesAndCorners`], and every other
// op in this crate defaults to [`Connectivity::Faces`].** That is deliberate and
// it is the one place this module is opinionated. The ops that default to faces
// are labelling ops, where the narrow choice is right because the complementary
// pair convention pairs a narrow foreground with a wide background; this op
// answers "which voxels touch", where the widest reading is the one a caller
// asking that question almost always means, and a caller who wanted only faces
// says so. Both are reachable through [`AdjacentPairsOp::connecting`] and
// through every free function's `connectivity` argument, and the sweep pins that
// the choice **changes the answer** rather than being decoration.
//
// The halo is symmetric and the reach is not
// ------------------------------------------
// A pair reaches one voxel, so the halo is one voxel on every axis. The *reach*
// is genuinely asymmetric: a lexicographically positive offset never steps down
// on axis 0, so nothing below the core on that axis is ever read, and under
// [`Connectivity::Faces`] nothing below the core is read on any axis. That
// cannot be said in [`FragmentOp::reach`], which holds one number per axis, so
// the wider side is declared and the over-fetch is written down here: at most one
// plane per block face more than is read. It is the arrangement `ops/mod.rs`
// describes for every other signature that can hold only one integer per axis.
//
// What it costs
// -------------
// At most thirteen tests per set voxel and one row per pair that passes, so the
// output is bounded by thirteen rows per set voxel and is usually far smaller —
// a voxel with no set neighbour above it emits nothing. The merge holds the
// whole list, which is inherent to the request in the same way it is for
// `coordinates`: the answer is the list.
//
// The volume is read once, with a one-voxel halo, and no block waits for any
// other. There is no counting pass and no prefix sum, because there is no number
// to prefix-sum: see the top of this header for why the numbering does not
// exist yet.
//
// Blocks that find nothing
// ------------------------
// A block whose core holds no lower endpoint writes a header and no rows, and
// the stream declares [`Coverage::EveryBlock`] so that present-and-empty is
// distinguishable from absent. Empty blocks are common rather than exceptional
// on sparse masks, and they cost nothing and shift nothing: there is no base
// index for an empty block to have got wrong.

use std::sync::Arc;

use ndarray::ArrayView3;

use crate::assemble::{Phase, PlanBuilder};
use crate::decomposition::Decomposition;
use crate::dtype::Dtype;
use crate::env::{BlockBuf, Environment};
use crate::error::{Error, Result};
use crate::fragment::{
    fold_fragments, fragment_phase, BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput,
    SidecarSize,
};
use crate::geometry::BlockGrid;
use crate::region::Region;
use crate::sidecar::Lifecycle;
use crate::table::{Column, RowBuilder, Schema, Table, Value};

use super::components::Connectivity;
use super::fill::as_mask;

/// A pair, as this module speaks of one: the lexicographically lower endpoint
/// first.
///
/// A named alias rather than a struct, because it is what the merge returns and
/// a tuple of two coordinates is already the whole of it.
pub type Pair = ([usize; 3], [usize; 3]);

// -------------------------------------------------------------- the schema --

/// The three payload columns, in schema order: the **higher** endpoint.
///
/// Named for the role rather than for an axis letter, because which of the two
/// endpoints a column belongs to is the thing a reader has to get right and the
/// axis is obvious from the index. The row's position is the lower endpoint and
/// has no column, for the reason `crate::table` gives: a position is not a
/// column.
pub const HIGHER_COLUMNS: [&str; 3] = ["higher 0", "higher 1", "higher 2"];

/// The schema a pair blob carries: three `U64` columns holding the higher
/// endpoint.
///
/// `U64` rather than `F64` because a coordinate is an exact integer and the
/// canonical order tiebreaks on the column's bits — as integers those bits sort
/// in the same order as the coordinates, which is what makes the canonical order
/// of a row *(lower endpoint, higher endpoint)* lexicographically and therefore
/// what makes the merge a sort of the walk.
pub fn pair_schema() -> Schema {
    Schema::new(
        HIGHER_COLUMNS
            .iter()
            .map(|name| Column::u64(*name))
            .collect(),
    )
    .expect("three columns, each named once")
}

/// A blob with no rows in it: what a block that found no pair writes.
///
/// Present and empty rather than absent, which is what [`Coverage::EveryBlock`]
/// makes checkable.
pub fn empty_pairs() -> Vec<u8> {
    RowBuilder::new(Arc::new(pair_schema())).encode()
}

// ------------------------------------------------------------ the offsets --

/// The steps from a voxel to a **lexicographically later** neighbour, in
/// lexicographic order.
///
/// Half of [`Connectivity::offsets`] and derived from it rather than written
/// out: the twenty-six offsets come in negatives, and for each pair exactly one
/// member is lexicographically positive — which is exactly the step from the
/// lower endpoint of a pair to the higher one. Three of them for
/// [`Connectivity::Faces`], nine for [`Connectivity::FacesAndEdges`], thirteen
/// for [`Connectivity::FacesEdgesAndCorners`].
///
/// **Sorted, and the sort is part of the answer.** Walking the offsets in this
/// order makes a block's rows come out already in the canonical order, since for
/// a fixed lower endpoint the higher endpoints sort exactly as their offsets do.
/// `offsets()` is grouped by how many axes a step moves along, which is the
/// right grouping for its own callers and is not this one, so the order is
/// imposed here instead of assumed there.
///
/// Allocates: thirteen elements, once per block rather than once per voxel.
pub fn forward_offsets(connectivity: Connectivity) -> Vec<[isize; 3]> {
    let mut forward: Vec<[isize; 3]> = connectivity
        .offsets()
        .iter()
        .copied()
        // `[isize; 3]` compares lexicographically, which is the comparison the
        // whole module is stated in.
        .filter(|by| *by > [0, 0, 0])
        .collect();
    forward.sort_unstable();
    forward
}

// --------------------------------------------------------------- the walk --

/// Every adjacent pair whose **lower** endpoint lies in `owned`, in the walk
/// order, handed to `each`.
///
/// The one kernel. The whole-volume reference, the blob a block writes and the
/// list a test builds all go through this, so a disagreement between them is a
/// decomposition bug and not a modelling difference.
///
/// * `mask` is what was read and `origin` is where its lowest voxel sits in the
///   volume, so every coordinate handed to `each` is a **volume** coordinate.
/// * `owned` is the region of the volume whose voxels this call may emit a pair
///   *from*. It is the block's core in a run and the whole volume in the
///   reference, and it is a parameter rather than "all of `mask`" because that
///   difference **is** the ownership rule: a call that owned everything it could
///   read would emit every seam-crossing pair twice.
/// * `volume` bounds the higher endpoint. A step off the end of the volume is
///   not a pair, and the mask alone cannot say where the end is when the mask is
///   one block of it.
///
/// **The halo is checked rather than assumed.** `mask` must cover `owned` grown
/// by one voxel and clipped to `volume`; a short read is refused naming the two
/// regions instead of quietly dropping the pairs that reach into the missing
/// plane. That failure is invisible in the answer — a smaller list of well-formed
/// pairs — and it is exactly what a halo that stopped being fetched would
/// produce, so it is the check worth paying for.
pub fn walk_adjacent_pairs(
    mask: ArrayView3<'_, bool>,
    origin: [usize; 3],
    owned: &Region,
    volume: [usize; 3],
    connectivity: Connectivity,
    each: &mut dyn FnMut([usize; 3], [usize; 3]) -> Result<()>,
) -> Result<()> {
    let extent = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    for axis in 0..3 {
        let read_lo = origin[axis];
        let read_hi = origin[axis] + extent[axis];
        if read_hi > volume[axis] {
            return Err(Error::invalid(format!(
                "a pair walk was handed an array covering {read_lo}..{read_hi} on axis {axis} of \
                 a volume {} long, so part of what it would read is outside the volume entirely.",
                volume[axis]
            )));
        }
        let owned_lo = owned.start[axis];
        let owned_hi = owned_lo + owned.shape[axis];
        let wanted_lo = owned_lo.saturating_sub(1);
        let wanted_hi = (owned_hi + 1).min(volume[axis]);
        if owned_hi > volume[axis] || read_lo > wanted_lo || read_hi < wanted_hi {
            return Err(Error::invalid(format!(
                "a pair walk owns {owned_lo}..{owned_hi} on axis {axis} and was handed \
                 {read_lo}..{read_hi} of a volume {} long, but a pair reaches one voxel, so it \
                 needs {wanted_lo}..{wanted_hi}. The pairs reaching into the missing plane would \
                 be dropped, and a shorter list of well-formed pairs is a failure nothing \
                 downstream can see.",
                volume[axis]
            )));
        }
    }

    let steps = forward_offsets(connectivity);
    for i in owned.start[0]..owned.start[0] + owned.shape[0] {
        for j in owned.start[1]..owned.start[1] + owned.shape[1] {
            for k in owned.start[2]..owned.start[2] + owned.shape[2] {
                let lower = [i, j, k];
                if !mask[[i - origin[0], j - origin[1], k - origin[2]]] {
                    continue;
                }
                for by in &steps {
                    let mut higher = [0usize; 3];
                    let mut inside = true;
                    for axis in 0..3 {
                        let moved = lower[axis] as isize + by[axis];
                        if moved < 0 || moved >= volume[axis] as isize {
                            inside = false;
                            break;
                        }
                        higher[axis] = moved as usize;
                    }
                    if !inside {
                        continue;
                    }
                    // In range by the halo check above: `higher` is one voxel
                    // from `lower`, which is in `owned`, and inside the volume.
                    if mask[[
                        higher[0] - origin[0],
                        higher[1] - origin[1],
                        higher[2] - origin[2],
                    ]] {
                        each(lower, higher)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// [`walk_adjacent_pairs`] into a [`RowBuilder`] of [`pair_schema`].
pub fn adjacent_pairs_into(
    mask: ArrayView3<'_, bool>,
    origin: [usize; 3],
    owned: &Region,
    volume: [usize; 3],
    connectivity: Connectivity,
    rows: &mut RowBuilder,
) -> Result<()> {
    walk_adjacent_pairs(
        mask,
        origin,
        owned,
        volume,
        connectivity,
        &mut |lower, higher| {
            rows.push(
                lower,
                &[
                    Value::U64(higher[0] as u64),
                    Value::U64(higher[1] as u64),
                    Value::U64(higher[2] as u64),
                ],
            )
        },
    )
}

/// The whole-volume answer: every adjacent pair, in the walk order.
///
/// **Not a second implementation** — the same kernel over one block, which is
/// what makes a disagreement with a blocked run a decomposition bug. `pub`
/// because the acceptance suite is a separate crate and needs exactly this.
pub fn adjacent_pairs(mask: ArrayView3<'_, bool>, connectivity: Connectivity) -> Result<Vec<Pair>> {
    let volume = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    let mut found = Vec::new();
    walk_adjacent_pairs(
        mask,
        [0, 0, 0],
        &Region::whole(&volume),
        volume,
        connectivity,
        &mut |lower, higher| {
            found.push((lower, higher));
            Ok(())
        },
    )?;
    Ok(found)
}

/// One array's pairs as the bytes a block writes.
pub fn encode_adjacent_pairs(
    mask: ArrayView3<'_, bool>,
    origin: [usize; 3],
    owned: &Region,
    volume: [usize; 3],
    connectivity: Connectivity,
) -> Result<Vec<u8>> {
    let mut rows = RowBuilder::new(Arc::new(pair_schema()));
    adjacent_pairs_into(mask, origin, owned, volume, connectivity, &mut rows)?;
    Ok(rows.encode())
}

/// The whole-volume answer as the **blob** a single-block run would write.
///
/// The byte-level reference. It is in the canonical order already — see the
/// module header — so it is also what the merge of any decomposition re-encodes
/// to, and the suite compares against it directly.
pub fn adjacent_pair_rows(
    mask: ArrayView3<'_, bool>,
    connectivity: Connectivity,
) -> Result<Vec<u8>> {
    let volume = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    encode_adjacent_pairs(
        mask,
        [0, 0, 0],
        &Region::whole(&volume),
        volume,
        connectivity,
    )
}

// -------------------------------------------------------------- the merge --

/// Every block's blob, in the walk order, as one list.
///
/// The order is restored from the rows themselves: the blobs go into one
/// [`Table`], whose canonical order is lexicographic on the row's words, and the
/// words of these rows are the lower endpoint followed by the higher one. So the
/// result is a function of the pair set alone — not of the lattice, not of which
/// block finished first, not of the order the blobs arrive in.
///
/// **It does not deduplicate**, and that is a decision rather than an omission.
/// Every pair is emitted exactly once by construction (the cores tile the volume
/// and the lower endpoint picks the owner), so there is nothing to remove; and a
/// merge that removed duplicates would make the one failure the ownership rule
/// could produce — a seam counted twice — invisible in the answer. It stays
/// visible, and the sweep looks straight at it.
///
/// A row is refused if its higher endpoint is outside `volume`, is not
/// lexicographically after the lower one, or is more than one voxel away on any
/// axis. Those three are true of every pair under every connectivity, so
/// checking them costs no knowledge of which one was used and catches a blob
/// that came from somewhere else or from an ownership rule that had been
/// inverted.
pub fn merge_pairs<'a>(
    volume: [usize; 3],
    blobs: impl IntoIterator<Item = ([usize; 3], &'a [u8])>,
) -> Result<Vec<Pair>> {
    let mut table = Table::new(volume, pair_schema())?;
    for (block, bytes) in blobs {
        table.write(block, bytes)?;
    }
    ordered_pairs(&mut table, volume)
}

/// [`merge_pairs`] over a stream in a store.
///
/// Streams the fragments one at a time rather than gathering them, for
/// `coordinates`' reason: holding every blob on top of the table that is about
/// to hold their rows doubles the one residency this operation has. `phase` is
/// half the address — a stream written by two phases holds two generations, and
/// a blob from the wrong one would decode perfectly and answer differently.
pub fn collect_pairs(
    env: &dyn Environment,
    stream: &str,
    phase: usize,
    volume: [usize; 3],
) -> Result<Vec<Pair>> {
    let mut table = Table::new(volume, pair_schema())?;
    fold_fragments(env, stream, &mut |key, bytes| {
        if key.phase != phase {
            return Ok(());
        }
        table.write(key.block, bytes)
    })?;
    ordered_pairs(&mut table, volume)
}

fn ordered_pairs(table: &mut Table, volume: [usize; 3]) -> Result<Vec<Pair>> {
    table.seal()?;
    let mut found = Vec::with_capacity(table.len());
    // A loop rather than a `collect` on the tail expression: the scan borrows
    // the table, and a borrow in a function's final expression outlives the
    // local it is taken from.
    for row in table.scan(&Region::whole(&volume))? {
        let lower = row.at();
        let mut higher = [0usize; 3];
        for axis in 0..3 {
            let word = row.u64(axis)?;
            let at = usize::try_from(word).map_err(|_| {
                Error::invalid(format!(
                    "a pair row at {lower:?} names {word} on axis {axis} of its higher endpoint, \
                     which is not a position this machine can address."
                ))
            })?;
            if at >= volume[axis] {
                return Err(Error::invalid(format!(
                    "a pair row at {lower:?} names {at} on axis {axis} of its higher endpoint, \
                     and the volume is only {} long on that axis. A pair is refused here rather \
                     than sorted quietly into the middle of the answer.",
                    volume[axis]
                )));
            }
            higher[axis] = at;
        }
        for axis in 0..3 {
            let step = higher[axis] as isize - lower[axis] as isize;
            if step.abs() > 1 {
                return Err(Error::invalid(format!(
                    "a pair row joins {lower:?} to {higher:?}, which are {} apart on axis {axis}. \
                     Adjacency reaches one voxel under every connectivity, so this blob was \
                     written by something else.",
                    step.abs()
                )));
            }
        }
        if higher <= lower {
            return Err(Error::invalid(format!(
                "a pair row at {lower:?} names {higher:?} as its higher endpoint, which is not \
                 lexicographically after it. Every pair is emitted from its lower endpoint by the \
                 block that owns that position, so a row like this is an ownership rule that has \
                 been inverted and a pair that some other block is also emitting."
            )));
        }
        found.push((lower, higher));
    }
    Ok(found)
}

// -------------------------------------------------------------- the phase --

/// **A mask in, one row per adjacent pair of set voxels out**, at reach 1.
///
/// Reads its block's pixels with a one-voxel halo, writes one fragment and no
/// image, and depends on no other block. Every row it writes has its lower
/// endpoint in this block's core, so the blobs of a run partition the answer;
/// the order they are put back into is [`merge_pairs`]', and is a function of
/// the rows rather than of this op.
pub struct AdjacentPairsOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    connectivity: Connectivity,
}

impl AdjacentPairsOp {
    /// `lifecycle` is the output stream's and is not defaulted, for
    /// `coordinates`' reason: this stream **is** the run's output, but what a run
    /// leaves behind is the caller's decision.
    ///
    /// The connectivity **is** defaulted, to
    /// [`Connectivity::FacesEdgesAndCorners`] — the module header says why this
    /// one op defaults wide where the labelling ops default narrow, and
    /// [`Self::connecting`] is how a caller says otherwise.
    pub fn new(name: &'static str, stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
            connectivity: Connectivity::FacesEdgesAndCorners,
        }
    }

    /// State which voxels count as adjacent.
    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// The stream the pair rows are written to.
    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// Which voxels this op counts as adjacent.
    pub fn connectivity(&self) -> Connectivity {
        self.connectivity
    }
}

impl FragmentOp for AdjacentPairsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// One voxel on every axis.
    ///
    /// The honest number is asymmetric — a lexicographically positive offset
    /// never steps down on axis 0, and under [`Connectivity::Faces`] never steps
    /// down at all — and this signature holds one number per axis, so the wider
    /// side is declared. The over-fetch is at most one plane per block face and
    /// is written down in the module header rather than left to be discovered.
    ///
    /// Independent of the configured halo, as the trait requires: it is a
    /// property of what a pair reaches and would be `1` whatever the lattice.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        1
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
            // Every block, always. This phase writes no image, so the tiling
            // check has nothing to bite on and the coverage declaration is the
            // only guard there is — and a block that owns no pair is the common
            // case on a sparse mask rather than an oddity.
            Coverage::EveryBlock,
            // A row table of ordered label pairs. A voxel can contribute at most one pair
            // per lexicographically-later neighbour, and `Connectivity::offsets`
            // is exactly that list — the same one `encode_adjacent_pairs` steps over.
        )
        .sized(SidecarSize::row_table(
            &pair_schema(),
            forward_offsets(self.connectivity).len() as u64,
        ))]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = at.pixels()? else {
            // An accounting run has no data. It still writes a fragment,
            // because what it is measuring is the IO, and a phase that silently
            // produced nothing would measure a different program.
            return Ok(BlockOutput::fragment(self.stream.clone(), empty_pairs()));
        };
        let mask = as_mask(pixels)?;
        let origin = [at.read.start[0], at.read.start[1], at.read.start[2]];
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            encode_adjacent_pairs(
                mask.view(),
                origin,
                at.core,
                at.grid.volume(),
                self.connectivity,
            )?,
        ))
    }
}

/// The one phase, as a whole `Decomposition`.
///
/// The op declares a reach of one and no fragment input, so `fragment_phase`
/// derives a halo of one and every block's valid region is its core — the read
/// extent is the core grown by one and shrunk by one again. The phase declares
/// no element type because it writes no image.
pub fn adjacent_pairs_phase(
    grid: BlockGrid,
    mask_dtype: Dtype,
    op: &AdjacentPairsOp,
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
/// [`adjacent_pairs_phase`] builds a whole `Decomposition`, so it is unusable as
/// soon as this phase sits inside something larger. Returns the phase the blobs
/// are keyed under, which is [`collect_pairs`]' `phase` argument.
pub fn append_to(plan: &mut PlanBuilder, op: AdjacentPairsOp) -> Result<Phase> {
    plan.fragments(op)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array3;

    use crate::table::encoded_schema;

    fn mask_of(shape: [usize; 3], set: &[[usize; 3]]) -> Array3<bool> {
        let mut mask = Array3::from_elem((shape[0], shape[1], shape[2]), false);
        for &at in set {
            mask[at] = true;
        }
        mask
    }

    const WIDE: Connectivity = Connectivity::FacesEdgesAndCorners;

    /// The forward offsets are half of the neighbourhood, and the half is the
    /// lexicographically positive one.
    ///
    /// Asserted against `components`' own tables rather than against a list
    /// written here, which is what makes this reuse rather than a copy: if
    /// `Connectivity::offsets` ever grows an offset, this says whether the
    /// filter still splits it in two.
    #[test]
    fn the_forward_offsets_are_half_the_neighbourhood() {
        for connectivity in [
            Connectivity::Faces,
            Connectivity::FacesAndEdges,
            Connectivity::FacesEdgesAndCorners,
        ] {
            let forward = forward_offsets(connectivity);
            let all = connectivity.offsets();
            assert_eq!(forward.len() * 2, all.len(), "{connectivity:?}");
            for by in &forward {
                let back = [-by[0], -by[1], -by[2]];
                assert!(all.contains(by), "{by:?} is not a neighbour offset");
                assert!(all.contains(&back), "{back:?} is not a neighbour offset");
                assert!(connectivity.joins(*by), "{by:?} should join");
            }
        }
        assert_eq!(forward_offsets(Connectivity::Faces).len(), 3);
        assert_eq!(forward_offsets(Connectivity::FacesAndEdges).len(), 9);
        assert_eq!(forward_offsets(WIDE).len(), 13);
    }

    /// And they are the same set `components` calls the forward directions of a
    /// lattice — the same "each pair once, from the lower one" rule, one image
    /// up. A cross-check between two tables that must agree and are written
    /// apart.
    #[test]
    fn the_forward_offsets_are_the_forward_directions() {
        for connectivity in [
            Connectivity::Faces,
            Connectivity::FacesAndEdges,
            Connectivity::FacesEdgesAndCorners,
        ] {
            let mut directions = connectivity.directions().to_vec();
            directions.sort_unstable();
            assert_eq!(
                forward_offsets(connectivity),
                directions,
                "{connectivity:?}"
            );
        }
    }

    /// The offsets are sorted, and sorted is what makes the walk canonical.
    #[test]
    fn the_offsets_are_in_lexicographic_order() {
        let forward = forward_offsets(WIDE);
        let mut sorted = forward.clone();
        sorted.sort_unstable();
        assert_eq!(forward, sorted);
        assert_eq!(forward.first(), Some(&[0, 0, 1]));
        assert_eq!(forward.last(), Some(&[1, 1, 1]));
    }

    /// **The output is already a sort of itself**, which is the module header's
    /// claim and the reason the merge is three lines.
    #[test]
    fn the_walk_is_the_canonical_order_of_the_pairs() {
        // Dense, and dense on purpose: a sparse mask can have every one of its
        // voxels emit at most one pair, and then the walk is sorted whatever
        // order the offsets are tried in. The claim is about the *inner* order,
        // so the fixture has to make a voxel emit several.
        for mask in [
            Array3::from_elem((3, 4, 5), true),
            mask_of(
                [4, 4, 4],
                &[
                    [1, 1, 1],
                    [0, 0, 0],
                    [2, 2, 2],
                    [1, 2, 1],
                    [2, 1, 2],
                    [1, 1, 2],
                ],
            ),
        ] {
            for connectivity in [
                Connectivity::Faces,
                Connectivity::FacesAndEdges,
                Connectivity::FacesEdgesAndCorners,
            ] {
                let found = adjacent_pairs(mask.view(), connectivity).unwrap();
                let mut sorted = found.clone();
                sorted.sort_unstable();
                assert_eq!(found, sorted, "the walk should need no sort");
                assert!(!found.is_empty());
                // and some voxel really did emit more than one row, or the
                // inner order was never exercised
                assert!(
                    found.windows(2).any(|two| two[0].0 == two[1].0),
                    "no voxel emitted two pairs, so this fixture cannot see the inner order"
                );
            }
        }
    }

    /// Every row has its endpoints the right way round and one voxel apart.
    #[test]
    fn every_pair_runs_from_the_lower_endpoint_to_a_neighbour() {
        let mask = Array3::from_elem((4, 3, 5), true);
        for (lower, higher) in adjacent_pairs(mask.view(), WIDE).unwrap() {
            assert!(higher > lower, "{lower:?} -> {higher:?}");
            for axis in 0..3 {
                let step = higher[axis] as isize - lower[axis] as isize;
                assert!(step.abs() <= 1, "{lower:?} -> {higher:?} on axis {axis}");
            }
        }
    }

    /// A voxel is never adjacent to itself, and no pair is emitted twice — the
    /// two properties that make the coordinate pair a complete name at this
    /// stage. The module header argues it; this executes it.
    #[test]
    fn the_pair_is_a_key_at_this_stage() {
        let mask = Array3::from_elem((4, 4, 4), true);
        let found = adjacent_pairs(mask.view(), WIDE).unwrap();
        let mut distinct = found.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), found.len(), "a pair was emitted twice");
        assert!(
            found.iter().all(|(lower, higher)| lower != higher),
            "a voxel was paired with itself"
        );
    }

    /// The count over a full box, against a closed form rather than against the
    /// code: for each forward step, the number of positions from which it stays
    /// inside is the product of `extent - |step|` over the axes.
    #[test]
    fn a_full_box_has_the_number_of_pairs_the_arithmetic_says() {
        let shape = [4usize, 3, 5];
        let mask = Array3::from_elem((shape[0], shape[1], shape[2]), true);
        for connectivity in [
            Connectivity::Faces,
            Connectivity::FacesAndEdges,
            Connectivity::FacesEdgesAndCorners,
        ] {
            let want: usize = forward_offsets(connectivity)
                .iter()
                .map(|by| {
                    (0..3)
                        .map(|axis| shape[axis] - by[axis].unsigned_abs())
                        .product::<usize>()
                })
                .sum();
            assert_eq!(
                adjacent_pairs(mask.view(), connectivity).unwrap().len(),
                want,
                "{connectivity:?}"
            );
        }
    }

    /// Two voxels touching at a corner: a pair under the widest connectivity and
    /// nothing at all under the narrowest.
    #[test]
    fn a_corner_touch_is_a_pair_only_under_the_widest_connectivity() {
        let mask = mask_of([2, 2, 2], &[[0, 0, 0], [1, 1, 1]]);
        assert_eq!(
            adjacent_pairs(mask.view(), WIDE).unwrap(),
            vec![([0, 0, 0], [1, 1, 1])]
        );
        assert!(adjacent_pairs(mask.view(), Connectivity::FacesAndEdges)
            .unwrap()
            .is_empty());
        assert!(adjacent_pairs(mask.view(), Connectivity::Faces)
            .unwrap()
            .is_empty());
    }

    /// One voxel is no pair, and neither is none.
    #[test]
    fn a_lone_voxel_and_an_empty_volume_have_no_pairs() {
        assert!(
            adjacent_pairs(mask_of([3, 3, 3], &[[1, 1, 1]]).view(), WIDE)
                .unwrap()
                .is_empty()
        );
        assert!(adjacent_pairs(mask_of([3, 3, 3], &[]).view(), WIDE)
            .unwrap()
            .is_empty());
    }

    /// A round trip through the blob and the merge, on one block, changes
    /// nothing — the floor the decomposition tests build on.
    #[test]
    fn one_blob_merges_back_to_the_walk() {
        let mask = mask_of(
            [4, 5, 6],
            &[[0, 4, 5], [1, 3, 4], [2, 0, 0], [3, 0, 1], [3, 3, 3]],
        );
        let blob = adjacent_pair_rows(mask.view(), WIDE).unwrap();
        let merged = merge_pairs([4, 5, 6], [([0, 0, 0], blob.as_slice())]).unwrap();
        assert_eq!(merged, adjacent_pairs(mask.view(), WIDE).unwrap());
    }

    /// A blob with no rows is a blob, and it contributes nothing.
    #[test]
    fn an_empty_blob_is_well_formed_and_adds_nothing() {
        let empty = empty_pairs();
        assert_eq!(encoded_schema(&empty).unwrap(), pair_schema());
        let mask = mask_of([2, 2, 2], &[[1, 1, 0], [1, 1, 1]]);
        let full = adjacent_pair_rows(mask.view(), WIDE).unwrap();
        let merged = merge_pairs(
            [2, 2, 2],
            [
                ([0, 0, 0], empty.as_slice()),
                ([0, 0, 1], full.as_slice()),
                ([0, 1, 0], empty.as_slice()),
            ],
        )
        .unwrap();
        assert_eq!(merged, vec![([1, 1, 0], [1, 1, 1])]);
    }

    /// A short halo is refused rather than answering with fewer pairs.
    ///
    /// The one voxel below the owned region on axis 1 is missing, which is
    /// exactly what a halo that stopped being fetched would look like, and the
    /// answer it would produce is a well-formed list nothing downstream could
    /// tell from the right one.
    #[test]
    fn a_read_that_is_short_of_the_halo_is_refused() {
        let mask = Array3::from_elem((4, 4, 4), true);
        let owned = Region::new(&[1, 1, 1], &[2, 2, 2]);
        // Handed only the owned region itself, with no context at all.
        let short = mask.slice(ndarray::s![1..3, 1..3, 1..3]);
        let error =
            walk_adjacent_pairs(
                short,
                [1, 1, 1],
                &owned,
                [4, 4, 4],
                WIDE,
                &mut |_, _| Ok(()),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("needs"), "{error}");
    }

    /// The merge refuses a row whose endpoints are the wrong way round, which is
    /// what an inverted ownership rule writes.
    #[test]
    fn the_merge_refuses_an_inverted_pair() {
        let mut rows = RowBuilder::new(Arc::new(pair_schema()));
        rows.push([1, 1, 1], &[Value::U64(1), Value::U64(1), Value::U64(0)])
            .unwrap();
        let error = merge_pairs([2, 2, 2], [([0, 0, 0], rows.encode().as_slice())])
            .unwrap_err()
            .to_string();
        assert!(error.contains("lexicographically after"), "{error}");
    }

    /// And one whose endpoints are too far apart to touch.
    #[test]
    fn the_merge_refuses_a_pair_that_does_not_touch() {
        let mut rows = RowBuilder::new(Arc::new(pair_schema()));
        rows.push([0, 0, 0], &[Value::U64(2), Value::U64(0), Value::U64(0)])
            .unwrap();
        let error = merge_pairs([4, 4, 4], [([0, 0, 0], rows.encode().as_slice())])
            .unwrap_err()
            .to_string();
        assert!(error.contains("apart on axis"), "{error}");
    }

    /// The phase declares a halo of one and its valid regions are still the
    /// cores, which is what lets them tile the volume with no overlap.
    #[test]
    fn the_phase_has_a_halo_of_one_and_its_valid_regions_are_the_cores() {
        let op = AdjacentPairsOp::new("adjacent pairs", "pairs", Lifecycle::DeleteOnExit);
        let grid = BlockGrid::new([8, 4, 4], [4, 4, 4]).unwrap();
        let phase = fragment_phase(&op, grid).unwrap();
        assert_eq!(phase.reach, [1, 1, 1]);
        assert_eq!(phase.halo, [1, 1, 1]);
        for block in &phase.blocks {
            assert_eq!(block.valid, block.core);
            // and the read extent really is wider where there is room for it
            assert!(block.read.shape[0] >= block.core.shape[0]);
        }
    }

    /// The op's connectivity is the one it was given, and the default is the
    /// wide one — the departure the module header argues for, pinned so that a
    /// change to it is a change to this test.
    #[test]
    fn the_default_connectivity_is_the_widest() {
        let op = AdjacentPairsOp::new("adjacent pairs", "pairs", Lifecycle::DeleteOnExit);
        assert_eq!(op.connectivity(), Connectivity::FacesEdgesAndCorners);
        assert_eq!(
            op.connecting(Connectivity::Faces).connectivity(),
            Connectivity::Faces
        );
    }
}
