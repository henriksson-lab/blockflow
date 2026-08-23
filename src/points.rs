// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the shape of the problem — a set
// of positions produced piecewise and read by extent — not adapted from any
// implementation.
//
// **A point set, stored by position rather than by producer.**
//
// This module is now a **special case of [`crate::table`]**, and a thin one: a
// `PointStore` is a `Table` over the schema `[weight: f64]`, and a `Point` is a
// row of it read through that fixed schema. What is left here is the point's own
// type, its encoding, and the store's surface. Everything structural — the two
// states, the canonical order, the streaming `scan`, the two interchangeable
// indexes and the k-way merge — lives in `table`, is argued for in its header,
// and is shared rather than repeated. `table`'s header is also where the
// decision to unify rather than to keep two stores is recorded, with what it
// bought and what it risked.
//
// What did *not* change when that happened is worth saying plainly, because it
// is the reason the move was cheap: a point blob is three coordinates and
// `weight.to_bits()`, which is exactly a row of the point schema, so the
// canonical order over rows is byte-for-byte the order over points. No answer
// moved, and this module's suite — including its white-box tests of the index,
// which now drive `table`'s — passes as it was written.
//
// The problem this removes
// ------------------------
// Every other node this crate materialises has a partitioning that is a
// *function of its domain*: an image is voxels partitioned by chunk, a fragment
// stream is blocks partitioned by block. A point set is the odd one out — its
// domain is position, and its partitioning is **whichever block happened to emit
// it**. That is not a property of the data; it is a property of the run that
// produced the data, and it leaks in two directions:
//
// * *outwards*, as an obligation on the producer. `ops::voxelize` requires that
//   a point in block B's fragment lie in B's core, and refuses anything else.
//   That rule is what makes its block reach derivable, and it is correct there.
//   What it costs is every producer whose points may land outside its own core —
//   a centroid of an object that straddles a seam, a set written by an earlier
//   run on a different lattice, a set written by another tool — none of which
//   can feed such an op at all;
// * *inwards*, as a hazard in the consumer. A reduction that visits points in
//   the order they arrive gets an answer that is a function of the cut, not of
//   the data. `voxelize` pays for that by re-deriving its own order over
//   everything it gathered, which works and is per block.
//
// A store keyed by *position* has neither. A block writes what it produced,
// wherever it landed; sorting is the store's job, once; and what comes back is
// in an order that is a function of the point set alone. **Ownership stops being
// a producer obligation and becomes the sorter's problem**, which is the only
// place it can be answered without knowing the run.
//
// What a point carries, and why it is one number
// ----------------------------------------------
// A point is a position and **one** `f64`. That is the shape `ops::voxelize`
// consumes — a weight to deposit — and it is the shape `ops::detect` produces.
// A consumer that wants a row per object, with a size and a total and a measure
// of shape, wants a [`crate::table::Table`] with those columns and not a second
// point set per column: point sets joined on a coordinate triple are correct
// only while every one of them holds exactly the same positions, which is a
// precondition nothing states and nothing can check.
//
// The weight is `f64` rather than something narrower because a consumer
// accumulating in anything narrower would put its own disagreement at around
// `1e-7` of the total, and it is per point rather than per set so that "count
// the points" and "sum a quantity over the points" are the same operation with a
// different column. It is a *float* column, and `table`'s header is explicit
// about what that costs: a fold over it is reproducible under every
// decomposition, because the store fixes the order, and it is not exact, because
// nothing can make `+` associate. That is why `ops::detect` accumulates in
// integers and converts once at the end, and why an op that needs an exact merge
// across a seam wants a `u64` column in a table rather than a point weight.
//
// The canonical order, and why the tiebreak is the payload
// --------------------------------------------------------
// Points come back sorted by **the coordinate triple, lexicographically, then by
// the weight's bits** — which is `table`'s canonical order applied to this
// schema, since a row's words are its position followed by its payload. The
// tiebreak is intrinsic to the point: not the source block, and not the order of
// insertion, both of which are facts about the run rather than about the data.
// The store does not merely decline to record where a point came from; there is
// nowhere in a row for it to go.
//
// Sealing is a barrier
// --------------------
// Every writer must have finished before [`PointStore::seal`] is called, and
// nothing here enforces that beyond refusing the writes that arrive afterwards.
// See `table`'s header for what a distributed sort over a set too large for one
// node would take; it is the next problem and not this one.

use crate::error::{Error, Result};
use crate::fragment::{
    pack_u64, unpack_u64, BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput,
};
use crate::geometry::BlockGrid;
use crate::region::Region;
use crate::sidecar::Lifecycle;
use crate::table::{RowBuffer, Schema, Table, Value};

// The store's surface keeps these names, because they are what a caller of a
// point store has always said. They are `table`'s types; a point store and a
// table are in the same two states and choose between the same two indexes,
// because they are the same store.
pub use crate::table::{Layout, State, TableIndex as PointIndex};

// The index's structural claims — the merge's cross-section residency, the
// bucket occupancy the derivation aims at — are asserted in this module's suite
// rather than in `table`'s, and deliberately: they are claims about the shared
// index, and asserting them through the point payload is what demonstrates that
// it *is* shared.
#[cfg(test)]
use crate::table::{bucket_counts, bucket_edge, Gridded, FLAT_LIMIT, TARGET_PER_BUCKET};
#[cfg(test)]
use std::cmp::Ordering;

// ----------------------------------------------------------------- points --

/// One point of the set: a voxel of the volume, and what it carries there.
///
/// The weight is `f64` because a consumer accumulating in anything narrower
/// would put its own disagreement at around `1e-7` of the total, and it is per
/// point rather than per set so that "count the points" and "sum a quantity over
/// the points" are the same operation with a different column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// The voxel, in **volume** coordinates. Not block-local: a point travels,
    /// and a coordinate that meant something only next to the block index that
    /// wrote it would be one more thing to get wrong — and would be exactly the
    /// producer-shaped partitioning this module exists to drop.
    pub at: [usize; 3],
    /// What this point carries. Its bits are also the order's tiebreak; see the
    /// module header.
    pub weight: f64,
}

impl Point {
    /// A point that carries `1` — the counting case.
    pub fn unit(at: [usize; 3]) -> Self {
        Self { at, weight: 1.0 }
    }

    pub fn weighted(at: [usize; 3], weight: f64) -> Self {
        Self { at, weight }
    }
}

/// `u64` words per encoded point: three coordinates and the weight's bits.
///
/// The same number as `Schema::points().width()`, and necessarily so — a point
/// blob's words *are* a table row's words, which is what makes the point store
/// the four-word case of `table` rather than something beside it.
pub const WORDS_PER_POINT: usize = 4;

/// Points as a blob.
///
/// Four little-endian `u64`s per point, through [`pack_u64`], because that is
/// the shape every other fragment in this crate has and a second encoding would
/// be a second thing to get subtly different. The weight travels as
/// `f64::to_bits` rather than as a decimal or a fixed-point integer, so it round
/// trips exactly — a store whose weights were rounded on the way through would
/// have an order, and answers, that were reproducible only by accident.
///
/// **Headerless, unlike a table blob.** A table carries its schema in front so
/// that a stream from the wrong producer is refused rather than read; a point
/// blob cannot be from the wrong producer, because there is only one point
/// schema and any four-word blob is a valid point set. The header would be a
/// constant the reader already knows. That is also why this encoding is kept
/// rather than replaced: it is the format `ops::detect` writes and
/// `ops::voxelize` reads today, and changing it would be a parity-visible change
/// to two ops for no property gained.
pub fn encode_points(points: &[Point]) -> Vec<u8> {
    let mut words = Vec::with_capacity(points.len() * WORDS_PER_POINT);
    for point in points {
        words.push(point.at[0] as u64);
        words.push(point.at[1] as u64);
        words.push(point.at[2] as u64);
        words.push(point.weight.to_bits());
    }
    pack_u64(&words)
}

/// The other half of [`encode_points`].
///
/// Three things are refused rather than accepted quietly, and each is a blob
/// that would otherwise read as something plausible: a length that is not a
/// whole number of points, a coordinate that does not fit this platform's
/// `usize`, and a weight that is not finite. The last is the one worth naming —
/// a NaN weight is not ordered by its own bits in the way a reader would expect,
/// and any sum over a set containing one stops meaning anything at all.
pub fn decode_points(bytes: &[u8]) -> Result<Vec<Point>> {
    let words = unpack_u64(bytes)?;
    if words.len() % WORDS_PER_POINT != 0 {
        return Err(Error::invalid(format!(
            "a point blob is {WORDS_PER_POINT} words per point: three coordinates and the \
             weight's bits. This one is {} word(s), which is {} point(s) and a remainder.",
            words.len(),
            words.len() / WORDS_PER_POINT
        )));
    }
    let mut points = Vec::with_capacity(words.len() / WORDS_PER_POINT);
    for (index, record) in words.chunks_exact(WORDS_PER_POINT).enumerate() {
        let mut at = [0usize; 3];
        for axis in 0..3 {
            at[axis] = usize::try_from(record[axis]).map_err(|_| {
                Error::invalid(format!(
                    "point {index} of this blob has coordinate {} on axis {axis}, which does \
                     not fit this platform's usize",
                    record[axis]
                ))
            })?;
        }
        let weight = f64::from_bits(record[3]);
        if !weight.is_finite() {
            return Err(Error::invalid(format!(
                "point {index} of this blob has weight {weight}, which is not finite. The \
                 canonical order tiebreaks on the weight's bits and any sum over the set would \
                 stop being a check on anything, so it is refused here rather than carried."
            )));
        }
        points.push(Point { at, weight });
    }
    Ok(points)
}

/// Points as rows, which is what they already were.
///
/// The words are identical to what [`encode_points`] writes, so this is a
/// relabelling rather than a conversion. It exists so that an index can be built
/// straight from a `Vec<Point>` — which is how this module's suite drives
/// `table`'s index directly.
impl From<Vec<Point>> for RowBuffer {
    fn from(points: Vec<Point>) -> Self {
        let mut words = Vec::with_capacity(points.len() * WORDS_PER_POINT);
        for point in &points {
            words.push(point.at[0] as u64);
            words.push(point.at[1] as u64);
            words.push(point.at[2] as u64);
            words.push(point.weight.to_bits());
        }
        RowBuffer::from_words(std::sync::Arc::new(Schema::points()), words)
    }
}

/// The canonical order, entire: the coordinate triple, then the weight's bits.
///
/// Test-only, and written from the definition in the module header rather than
/// by calling `table`'s comparison — an oracle that called the code it is
/// checking would assert nothing.
#[cfg(test)]
fn canonical(left: &Point, right: &Point) -> Ordering {
    left.at
        .cmp(&right.at)
        .then_with(|| left.weight.to_bits().cmp(&right.weight.to_bits()))
}

// --------------------------------------------------------------- producer --

/// A point set already in memory, written into the block whose **core**
/// contains each point.
///
/// The kernel of [`PointSourceOp`], and a free function so the keying rule —
/// which is the whole of what this op does — can be checked without building a
/// block view.
///
/// **The division is not clamped, and that is the rule rather than an
/// oversight.** A point whose coordinate lies outside the volume divides to a
/// block index the lattice does not have, so it is written into **no**
/// fragment at all. `ops::voxelize` requires every point in block `B`'s
/// fragment to lie in `B`'s core and refuses anything else; keying such a point
/// into the last block instead would hand that op a fragment it must reject, so
/// the honest answer is that this producer has nowhere to put it. That is the
/// difference from `ops::rows::RowSourceOp`, whose `ops::detect::owner_of`
/// **is** clamped because a table refuses an out-of-volume row later and by
/// name. Two producers, two rules, and neither is a case of the other — see
/// [`PointSourceOp`] for the other half of why.
fn block_points(grid: &BlockGrid, index: [usize; 3], points: &[Point]) -> Vec<u8> {
    let edge = grid.block();
    let mine: Vec<Point> = points
        .iter()
        .copied()
        .filter(|point| (0..3).all(|axis| point.at[axis] / edge[axis] == index[axis]))
        .collect();
    encode_points(&mine)
}

/// **A point set in, one fragment per block out** — the producer the point
/// world had, three times, in three different files.
///
/// `ops::detect` produces points *from an image*, which is a detector; this
/// produces them from a list somebody already has — a coordinate file, a set
/// from an earlier run, a fixture. Every consumer of points in this crate
/// follows a phase that emitted some, so a plan whose input *is* a point set
/// had nothing to start it and each caller wrote the same fifteen lines:
/// `tests/voxelize.rs` and `tests/point_labels.rs` here, character for
/// character, and once more in a consumer of this crate.
///
/// **Why this is not `ops::rows::RowSourceOp` over the point schema.** The
/// words are the same — a point blob's four words *are* a row of
/// [`Schema::points`], which is what makes this module the four-word case of
/// [`crate::table`] — but a table blob carries its schema **in front** and a
/// point blob is headerless, for the reason [`encode_points`] gives: any
/// four-word blob is a valid point set, so the header would be a constant the
/// reader already knows, and it is the format `ops::detect` writes and
/// `ops::voxelize` reads today. A general row producer here would write the
/// right words behind a header those two ops do not read. That is a
/// parity-visible change to two ops in exchange for deleting a struct, and this
/// module already declined it once.
///
/// So there are two producers of the same *shape* over two encodings, and the
/// encoding is the half with consumers. The keying differs too, and
/// deliberately: see [`block_points`].
///
/// Reads no image and declares no fragment input, so it is a **phase-0 op** —
/// the thing there is nothing before. [`Coverage::EveryBlock`], because a block
/// with no points writes an **empty** fragment rather than none: present and
/// empty is a different fact from absent, and only the first is checkable.
/// `ops::voxelize` tolerates either and declaring the stronger one is free.
///
/// Every block filters the whole set, which is `blocks x points` of work — the
/// honest cost of a plan whose input is a list rather than an image. A producer
/// that already knew which points were whose would be a detector.
pub struct PointSourceOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    points: Vec<Point>,
}

impl PointSourceOp {
    /// `points` are taken as given. A coordinate this op cannot place is not
    /// refused here — it is written nowhere, which is what [`block_points`]
    /// documents and what `ops::voxelize`'s own coverage check is positioned to
    /// notice.
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        lifecycle: Lifecycle,
        points: Vec<Point>,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
            points,
        }
    }

    /// The stream this writes, which is the one a consumer names as its input.
    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// How many points it was handed — the number a consumer's total must come
    /// to, and the only thing about the set this op will say.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl FragmentOp for PointSourceOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// **Zero, and it is a different quantity from a block reach.** This one is
    /// voxels of the block's own image, and this op reads no image at all — the
    /// points are held in memory and filtered to this block's index. What
    /// crosses a block boundary is a *point*, and none does: each is written
    /// into the one block whose core contains it, once, which is what makes
    /// `ops::voxelize`'s reach in blocks derivable at all.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            block_points(at.grid, at.index, &self.points),
        ))
    }
}

// ------------------------------------------------------------------ store --

/// A point set, written per block and read by region.
///
/// A [`Table`] over the schema `[weight: f64]`, and nothing else: every method
/// here is the table's, with the point's four words read back as a [`Point`].
/// See `crate::table`'s header for the two states, the canonical order and why
/// the index is not observable.
pub struct PointStore {
    table: Table,
}

impl PointStore {
    /// An empty, accumulating store over `volume`.
    ///
    /// A volume with a zero-length axis holds no voxels, so no point could ever
    /// be inside it and no query could ever be answered; that is refused here
    /// rather than turned into a store that is silently always empty.
    pub fn new(volume: [usize; 3]) -> Result<Self> {
        Ok(Self {
            // The noun is what the table calls itself in its own diagnostics.
            // A caller who asked for a point store and never mentioned a table
            // should not be told about one.
            table: Table::named(volume, Schema::points(), "point store")?,
        })
    }

    pub fn volume(&self) -> [usize; 3] {
        self.table.volume()
    }

    pub fn state(&self) -> State {
        self.table.state()
    }

    /// Which index was built, or `None` while accumulating.
    pub fn layout(&self) -> Option<Layout> {
        self.table.layout()
    }

    /// How many points have been written.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Take one block's blob.
    ///
    /// **There is no ownership rule.** A block writes whatever points it
    /// produced, wherever they landed, and nothing here asks whether they are
    /// near the block that wrote them — which is the only thing a block *can*
    /// do, since a `BlockOutput::fragment` is keyed by its own phase and its own
    /// block and there is no way to write into a neighbour's key. Which region
    /// eventually holds a point is decided by [`Self::seal`], from the
    /// coordinate.
    ///
    /// `block` is used for the error message and for nothing else. The store
    /// keeps no record of where a point came from, which is what makes an order
    /// that tiebreaks on the source block unrepresentable rather than merely
    /// discouraged.
    ///
    /// What *is* checked is the volume: a point outside it is at a coordinate no
    /// region of this store can name, so it would be held and never returned.
    /// That is the silent-loss shape, and it is refused.
    ///
    /// The blob is decoded here rather than handed to the table as bytes,
    /// because a point blob is headerless where a table blob is not — see
    /// [`encode_points`] — and because the diagnostics a point producer needs
    /// name points and weights rather than rows and columns.
    pub fn write(&mut self, block: [usize; 3], bytes: &[u8]) -> Result<()> {
        if self.state() == State::Sealed {
            return Err(Error::invalid(format!(
                "this point store is {}, so it takes no more points; block {block:?} tried to \
                 write {} byte(s). Sealing is a barrier: the index was built from everything \
                 that had been written, and a point arriving afterwards would be in the store's \
                 data and not in its answers.",
                State::Sealed.as_str(),
                bytes.len()
            )));
        }
        let points = decode_points(bytes).map_err(|err| {
            Error::invalid(format!(
                "the blob written by block {block:?} is unusable: {err}"
            ))
        })?;
        // Every point is checked before any is kept, so a refused write leaves
        // nothing behind: "how many points does this store hold" must not become
        // a fact about where in the blob the failure was.
        let volume = self.volume();
        for (index, point) in points.iter().enumerate() {
            for axis in 0..3 {
                if point.at[axis] >= volume[axis] {
                    return Err(Error::invalid(format!(
                        "point {index} of block {block:?} is at {:?}, which is outside this \
                         store's volume {:?} on axis {axis}. A point store does not care which \
                         block a point came from — that is the whole point of it — but a \
                         coordinate outside the volume is one no query can ever name, so it \
                         would be kept and never returned.",
                        point.at, volume
                    )));
                }
            }
        }
        for point in &points {
            self.table.push(point.at, &[Value::F64(point.weight)])?;
        }
        Ok(())
    }

    /// Sort what has been written and build an index for it, choosing from the
    /// point count.
    ///
    /// **This is a barrier.** Every writer must have finished; nothing here can
    /// check that, and the most it does is refuse the writes that arrive
    /// afterwards.
    pub fn seal(&mut self) -> Result<()> {
        self.table.seal()
    }

    /// Seal with a named index rather than the derived one.
    ///
    /// Here so that the two can be built over the same input and compared, which
    /// is how "the choice is not observable" is asserted rather than asserted
    /// about. It is **not** a tuning knob: the two answer identically, so the
    /// only thing choosing differently from [`Self::seal`] can do is make a run
    /// slower.
    pub fn seal_as(&mut self, layout: Layout) -> Result<()> {
        self.table.seal_as(layout)
    }

    /// A stream of every point in `region`, in the canonical order.
    ///
    /// **This is the read interface.** It borrows the store and yields as it
    /// goes; at no point is the answer held, which is what makes a whole-volume
    /// read possible over a set that a second copy of would not fit.
    /// [`Self::query`] collects it, and is a convenience rather than the
    /// primitive.
    ///
    /// Half-open: a point at `region.start` is in the answer, one at
    /// `region.start + region.shape` is not.
    ///
    /// A region reaching outside the volume is refused rather than clipped. The
    /// store's domain is the volume; a question about somewhere outside it has
    /// no answer, and "nothing there" is the answer a genuinely empty region
    /// gets, so returning it would make the two indistinguishable.
    pub fn scan(&self, region: &Region) -> Result<Box<dyn Iterator<Item = Point> + '_>> {
        Ok(Box::new(self.table.scan(region)?.map(|row| {
            Point {
                at: row.at(),
                // Cannot fire: the schema is fixed by `new` and its only column is
                // the weight. The alternative is to read the word directly, which
                // would be this module reaching past the accessor that keeps the
                // storage layout private.
                weight: row
                    .f64(0)
                    .expect("a point store's schema has one column and it is the f64 weight"),
            }
        })))
    }

    /// [`Self::scan`], collected.
    ///
    /// Here because collecting is often what a caller wants and writing
    /// `.collect()` at every call site is noise. It is deliberately *not* the
    /// primitive: the allocation is now a decision visible where it is made,
    /// rather than something the interface imposes on a caller who only wanted
    /// to fold.
    pub fn query(&self, region: &Region) -> Result<Vec<Point>> {
        Ok(self.scan(region)?.collect())
    }
}

impl std::fmt::Debug for PointStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PointStore")
            .field("volume", &self.volume())
            .field("state", &self.state().as_str())
            .field("points", &self.len())
            .field("layout", &self.layout().map(Layout::as_str))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------- the producer ------

    /// The producer's fixture: points on both sides of every seam, one that
    /// *is* the first voxel of a core, and one in the short last block.
    fn scattered() -> Vec<Point> {
        vec![
            Point::unit([0, 0, 0]),
            Point::weighted([2, 0, 0], 2.5),
            Point::unit([3, 0, 0]),
            Point::weighted([3, 3, 3], -1.0),
            Point::unit([7, 7, 7]),
            Point::weighted([6, 1, 0], 0.25),
            Point::unit([0, 7, 2]),
        ]
    }

    /// **Every point inside the volume is written once, into the block whose
    /// core contains it.**
    ///
    /// The producer's half of the rule `ops::voxelize` checks from the reading
    /// side, and the precondition that makes that op's reach in *blocks*
    /// derivable at all. Asserted as a partition rather than as a membership
    /// test: a point written into two blocks is counted twice and a point
    /// written into none disappears, and neither is visible to "each point is
    /// in the right block".
    ///
    /// Membership is checked against `BlockGrid::cores`, not against the
    /// division the producer itself uses — an oracle that recomputed the code
    /// under test would assert nothing.
    #[test]
    fn a_point_set_is_written_into_the_block_whose_core_contains_each_point() {
        for block in [[4, 4, 4], [3, 3, 3]] {
            let grid = BlockGrid::new([8, 8, 8], block).expect("a cut of the volume");
            let points = scattered();
            let cores = grid.cores();

            // The discriminator, first. A fixture whose points fall in one
            // block, or none of which straddles a seam, cannot tell a correct
            // keying rule from no rule at all.
            let reached: std::collections::BTreeSet<[usize; 3]> = cores
                .iter()
                .filter(|core| {
                    let (lo, hi) = (core.core.ranges(), core.core.end());
                    points.iter().any(|point| {
                        (0..3).all(|axis| point.at[axis] >= lo[axis].0 && point.at[axis] < hi[axis])
                    })
                })
                .map(|core| core.index)
                .collect();
            assert!(
                reached.len() >= 4,
                "at {block:?} the fixture reaches only {} core(s), so a producer that \
                 ignored the coordinate would pass",
                reached.len()
            );

            let mut gathered: Vec<Point> = Vec::new();
            for core in &cores {
                let blob = block_points(&grid, core.index, &points);
                let mine = decode_points(&blob).expect("a blob this op wrote decodes");
                let (lo, hi) = (core.core.ranges(), core.core.end());
                for point in &mine {
                    assert!(
                        (0..3)
                            .all(|axis| point.at[axis] >= lo[axis].0 && point.at[axis] < hi[axis]),
                        "at {block:?}, block {:?} was given the point at {:?}, which its core \
                         {:?}..{:?} does not contain",
                        core.index,
                        point.at,
                        core.core
                            .ranges()
                            .iter()
                            .map(|&(lo, _)| lo)
                            .collect::<Vec<_>>(),
                        hi
                    );
                }
                gathered.extend(mine);
            }

            let mut got: Vec<[usize; 3]> = gathered.iter().map(|point| point.at).collect();
            got.sort_unstable();
            let mut want: Vec<[usize; 3]> = points.iter().map(|point| point.at).collect();
            want.sort_unstable();
            assert_eq!(
                got, want,
                "at {block:?} the blocks together are not the point set exactly once"
            );
        }
    }

    /// **A point the lattice cannot place is written nowhere — not into the
    /// last block.**
    ///
    /// This is the whole difference between this producer and
    /// `ops::rows::RowSourceOp`, and it is the reason the two are not one op
    /// with a parameter. That one keys by `ops::detect::owner_of`, which
    /// **clamps**, because a table refuses an out-of-volume row later and by
    /// name. Clamping here would hand `ops::voxelize` a fragment holding a
    /// point outside the block's core, which is precisely what that op refuses
    /// — so the honest answer is that this producer has nowhere to put it.
    ///
    /// The cut tiles exactly, so "past the volume" and "past the lattice" are
    /// the same coordinate. Under a cut that does *not* tile exactly they are
    /// not, and the producer's guarantee is the weaker one it states: into the
    /// block the division names. The consumer's core check is what catches the
    /// rest, which is why it is there.
    #[test]
    fn a_point_the_lattice_cannot_place_is_written_nowhere() {
        let grid = BlockGrid::new([8, 8, 8], [4, 4, 4]).expect("a cut that tiles exactly");
        let inside = Point::unit([7, 7, 7]);
        let outside = Point::unit([8, 0, 0]);
        let points = vec![inside, outside];

        let mut written = 0;
        for core in grid.cores() {
            let blob = block_points(&grid, core.index, &points);
            let mine = decode_points(&blob).expect("a blob this op wrote decodes");
            assert!(
                !mine.iter().any(|point| point.at == outside.at),
                "block {:?} was given the point at {:?}, which no core contains; a clamped \
                 keying would put it in the last block and `ops::voxelize` would refuse \
                 that fragment",
                core.index,
                outside.at
            );
            written += mine.len();
        }
        // The premise: the *other* point was placed, so this measured a keying
        // rule and not a producer that writes nothing.
        assert_eq!(
            written, 1,
            "the point inside the volume must still be written exactly once"
        );
    }

    /// **A block with no points writes an empty blob, not nothing.**
    ///
    /// `Coverage::EveryBlock` is the only guard a phase that writes no image
    /// has, and it can only check a fragment that is there.
    #[test]
    fn a_block_with_no_points_still_writes_a_readable_fragment() {
        let grid = BlockGrid::new([8, 8, 8], [4, 4, 4]).expect("a cut of the volume");
        let only = vec![Point::unit([0, 0, 0])];
        let empty = block_points(&grid, [1, 1, 1], &only);
        assert!(
            empty.is_empty(),
            "a point blob is headerless; empty is zero bytes"
        );
        assert!(decode_points(&empty)
            .expect("an empty blob decodes")
            .is_empty());
    }

    /// A SplitMix64 stream. The tests want *many* regions and *many* points and
    /// want the same ones every run; a named constant seed is what makes a
    /// failure reproducible rather than a thing that happened once.
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn below(&mut self, limit: usize) -> usize {
            (self.next() % limit as u64) as usize
        }
    }

    /// The blobs a set of blocks would have written, with the points scattered
    /// between them **without regard to where they are** — which is the write
    /// path this module exists to allow and which `ops::voxelize` would refuse.
    fn blobs(points: &[Point], blocks: usize, seed: u64) -> Vec<([usize; 3], Vec<u8>)> {
        let mut stream = Stream(seed);
        let mut buckets = vec![Vec::new(); blocks];
        for point in points {
            buckets[stream.below(blocks)].push(*point);
        }
        buckets
            .into_iter()
            .enumerate()
            .map(|(index, mine)| ([index, 0, 0], encode_points(&mine)))
            .collect()
    }

    fn sealed(volume: [usize; 3], blobs: &[([usize; 3], Vec<u8>)], layout: Layout) -> PointStore {
        let mut store = PointStore::new(volume).unwrap();
        for (block, bytes) in blobs {
            store.write(*block, bytes).unwrap();
        }
        store.seal_as(layout).unwrap();
        store
    }

    fn scatter(volume: [usize; 3], count: usize, seed: u64) -> Vec<Point> {
        let mut stream = Stream(seed);
        (0..count)
            .map(|_| {
                let at = [
                    stream.below(volume[0]),
                    stream.below(volume[1]),
                    stream.below(volume[2]),
                ];
                // Weights that repeat, so that coincident points with equal
                // payloads and coincident points with different payloads both
                // occur in a set this size.
                let weight = (stream.below(5) as f64) - 2.0;
                Point::weighted(at, weight)
            })
            .collect()
    }

    /// Regions of every shape the interface has to survive: empty, one voxel,
    /// slabs, the whole volume, and boxes at the far corner.
    fn regions(volume: [usize; 3], seed: u64) -> Vec<Region> {
        let mut stream = Stream(seed);
        let half = [volume[0] / 2, volume[1] / 2, volume[2] / 2];
        let mut out = vec![
            Region::whole(&volume),
            // empty in three ways: no extent at all, and no extent on one axis
            Region::new(&[0, 0, 0], &[0, 0, 0]),
            Region::new(&[1, 1, 0], &[0, half[1], volume[2]]),
            Region::new(&[0, 0, 0], &[half[0], half[1], 0]),
            // one voxel, at the near corner and at the far one
            Region::new(&[0, 0, 0], &[1, 1, 1]),
            Region::new(&[volume[0] - 1, volume[1] - 1, volume[2] - 1], &[1, 1, 1]),
            // slabs along each axis
            Region::new(&[0, 0, 0], &[volume[0], 1, 1]),
            Region::new(&[0, 0, 0], &[1, volume[1], volume[2]]),
            // and the far octant, which is the one an off-by-one in the bucket
            // span reaches past
            Region::new(
                &half,
                &[
                    volume[0] - half[0],
                    volume[1] - half[1],
                    volume[2] - half[2],
                ],
            ),
        ];
        for _ in 0..200 {
            let mut start = [0usize; 3];
            let mut shape = [0usize; 3];
            for axis in 0..3 {
                start[axis] = stream.below(volume[axis]);
                shape[axis] = stream.below(volume[axis] - start[axis] + 1);
            }
            out.push(Region::new(&start, &shape));
        }
        out
    }

    // ------------------------------------------------------- the main one --

    /// A point, as the bits that distinguish it. Comparing `Point` directly
    /// would use `f64`'s own equality, under which `-0.0 == 0.0`; two answers
    /// differing there would be a real difference in the bytes a caller writes
    /// out, so the comparison is on the bits.
    fn key(point: Point) -> ([usize; 3], u64) {
        (point.at, point.weight.to_bits())
    }

    /// **The most important test in the file.** If the two indexes ever answer
    /// differently, they are not two implementations of one thing and every
    /// claim about the choice being invisible is false.
    ///
    /// Compared **element by element off the two streams**, rather than by
    /// collecting them: a difference in position fails at that position, a
    /// difference in length fails as `Some` against `None`, and neither side is
    /// materialised to make the comparison.
    #[test]
    fn both_indexes_answer_every_query_identically() {
        for (volume, count, seed) in [
            ([40usize, 40, 40], 5_000usize, 11u64),
            // Also the degenerate end: few enough points that the derivation
            // gives the gridded index a single bucket, which is the case where
            // an off-by-one in the span would otherwise hide.
            ([16, 12, 9], 12, 23),
            // And a flat volume, where an axis is clamped to its own length.
            ([64, 64, 1], 3_000, 37),
            // And a set with no points at all.
            ([8, 8, 8], 0, 41),
        ] {
            let points = scatter(volume, count, seed);
            let written = blobs(&points, 7, seed ^ 0xabc);
            let flat = sealed(volume, &written, Layout::Flat);
            let gridded = sealed(volume, &written, Layout::Gridded);
            assert_eq!(flat.len(), count);
            assert_eq!(gridded.len(), count);

            let asked = regions(volume, seed ^ 0xdef);
            let mut answered = 0usize;
            for region in &asked {
                let mut left = flat.scan(region).unwrap().map(key);
                let mut right = gridded.scan(region).unwrap().map(key);
                let mut at = 0usize;
                loop {
                    let (from_flat, from_gridded) = (left.next(), right.next());
                    assert_eq!(
                        from_flat, from_gridded,
                        "volume {volume:?}, {count} point(s), region {region:?}: the flat and \
                         gridded streams differ at position {at}"
                    );
                    if from_flat.is_none() {
                        break;
                    }
                    at += 1;
                    answered += 1;
                }
            }
            // Not vacuous: the sweep has to actually be finding points, or two
            // empty answers would agree for the wrong reason.
            if count > 0 {
                assert!(
                    answered > count,
                    "the region sweep returned {answered} point(s) over {} region(s); it is \
                     not exercising the indexes",
                    asked.len()
                );
            }
        }
    }

    // ---------------------------------------------------------- the states --

    #[test]
    fn a_query_before_sealing_fails_and_names_the_state() {
        let mut store = PointStore::new([8, 8, 8]).unwrap();
        assert_eq!(store.state(), State::Accumulating);
        assert_eq!(store.layout(), None);
        store
            .write([0, 0, 0], &encode_points(&[Point::unit([1, 1, 1])]))
            .unwrap();

        let whole = Region::whole(&[8, 8, 8]);
        // Both the primitive and the convenience, because a stream that opened
        // and yielded nothing would be the quiet answer this refuses to give.
        for error in [
            store.scan(&whole).err().map(|err| err.to_string()).unwrap(),
            store.query(&whole).unwrap_err().to_string(),
        ] {
            assert!(error.contains("accumulating"), "{error}");
            assert!(error.contains("sealed"), "{error}");
            assert!(error.contains("seal()"), "{error}");
        }

        store.seal().unwrap();
        assert_eq!(store.state(), State::Sealed);
        assert_eq!(store.layout(), Some(Layout::Flat));
        assert_eq!(store.query(&whole).unwrap().len(), 1);
        assert_eq!(store.scan(&whole).unwrap().count(), 1);
    }

    #[test]
    fn a_sealed_store_takes_no_more_points_and_cannot_be_sealed_twice() {
        let mut store = PointStore::new([8, 8, 8]).unwrap();
        store.seal().unwrap();
        let error = store
            .write([2, 0, 0], &encode_points(&[Point::unit([0, 0, 0])]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("sealed"), "{error}");
        assert!(error.contains("[2, 0, 0]"), "{error}");
        assert_eq!(store.len(), 0, "a refused write must not land");

        let error = store.seal().unwrap_err().to_string();
        assert!(error.contains("already"), "{error}");
        assert!(error.contains("sealed"), "{error}");
    }

    /// The sealer picks from the count, and both sides of the threshold are
    /// reachable. The *answers* are compared elsewhere; this only pins that the
    /// derivation is live rather than always giving the same index.
    #[test]
    fn the_sealer_chooses_from_the_point_count() {
        let volume = [64usize, 64, 64];
        let small = scatter(volume, FLAT_LIMIT, 5);
        let mut store = PointStore::new(volume).unwrap();
        store.write([0, 0, 0], &encode_points(&small)).unwrap();
        store.seal().unwrap();
        assert_eq!(store.layout(), Some(Layout::Flat));

        let large = scatter(volume, FLAT_LIMIT + 1, 6);
        let mut store = PointStore::new(volume).unwrap();
        store.write([0, 0, 0], &encode_points(&large)).unwrap();
        store.seal().unwrap();
        assert_eq!(store.layout(), Some(Layout::Gridded));
    }

    // ------------------------------------------------------- what is in it --

    /// Exactly the points inside, and the half-open boundary stated in the
    /// header. Built by hand so the expected answer comes from the definition
    /// rather than from a second run of this code.
    #[test]
    fn a_query_returns_the_points_inside_and_the_boundary_is_half_open() {
        let volume = [10usize, 10, 10];
        let points = vec![
            Point::unit([2, 2, 2]), // the low corner of the region: in
            Point::unit([5, 5, 5]), // the high corner of the region: out
            Point::unit([4, 4, 4]), // strictly inside
            Point::unit([1, 2, 2]), // one short on the first axis: out
            Point::unit([2, 1, 2]), // out on the second
            Point::unit([2, 2, 1]), // out on the third
            Point::unit([5, 4, 4]), // on the far face of the first axis: out
            Point::unit([4, 5, 4]), // the second: out
            Point::unit([4, 4, 5]), // the third: out
        ];
        for layout in [Layout::Flat, Layout::Gridded] {
            let store = sealed(volume, &blobs(&points, 3, 99), layout);
            let found = store.query(&Region::new(&[2, 2, 2], &[3, 3, 3])).unwrap();
            assert_eq!(
                found,
                vec![Point::unit([2, 2, 2]), Point::unit([4, 4, 4])],
                "{}",
                layout.as_str()
            );
            // A zero-shape region holds nothing, including the point at its
            // own start.
            assert!(store
                .query(&Region::new(&[2, 2, 2], &[0, 3, 3]))
                .unwrap()
                .is_empty());
            // And the whole volume holds everything.
            assert_eq!(store.query(&Region::whole(&volume)).unwrap().len(), 9);
        }
    }

    #[test]
    fn a_region_outside_the_volume_or_of_the_wrong_rank_is_refused() {
        let store = sealed([8, 8, 8], &[], Layout::Flat);
        let error = store
            .query(&Region::new(&[6, 0, 0], &[4, 1, 1]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("point store query region"), "{error}");
        let error = store
            .query(&Region::new(&[0, 0], &[1, 1]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("rank 2"), "{error}");
    }

    #[test]
    fn a_point_outside_the_volume_is_refused_and_says_it_is_not_about_ownership() {
        let mut store = PointStore::new([8, 4, 4]).unwrap();
        // Deliberately keyed to a block nowhere near the point: that part is
        // fine here, and the message has to say so.
        let error = store
            .write([0, 0, 0], &encode_points(&[Point::unit([8, 0, 0])]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside this store's volume"), "{error}");
        assert!(error.contains("does not care which block"), "{error}");
        assert_eq!(store.len(), 0);
        assert!(PointStore::new([8, 0, 4]).is_err());
    }

    /// The write path with the ownership rule dropped: every point written by
    /// the block furthest from it, which `ops::voxelize` refuses by design.
    #[test]
    fn a_block_may_write_a_point_that_landed_anywhere() {
        let volume = [16usize, 16, 16];
        let points = scatter(volume, 40, 77);
        let mut store = PointStore::new(volume).unwrap();
        // One blob, from one block, holding points spread over the whole volume.
        store.write([3, 3, 3], &encode_points(&points)).unwrap();
        store.seal().unwrap();
        assert_eq!(store.query(&Region::whole(&volume)).unwrap().len(), 40);
    }

    // -------------------------------------------------------- the ordering --

    /// Shuffle the blobs, seal again, and the bytes must not move. The set
    /// deliberately contains coincident points with *different* payloads, so the
    /// tiebreak is exercised rather than vacuous — and the assertion below
    /// checks that it is, by pinning the order those two come back in against
    /// the order they were written in.
    #[test]
    fn the_canonical_order_survives_a_shuffle_of_the_blobs() {
        let volume = [24usize, 24, 24];
        let mut points = scatter(volume, 400, 101);
        // Three points at one coordinate, with three different payloads, and
        // written in an order that is not the order of their bits.
        points.push(Point::weighted([7, 7, 7], 4.0));
        points.push(Point::weighted([7, 7, 7], 1.0));
        points.push(Point::weighted([7, 7, 7], 2.0));

        let mut written = blobs(&points, 9, 202);
        let reference = sealed(volume, &written, Layout::Flat);
        let expected = encode_points(&reference.query(&Region::whole(&volume)).unwrap());

        let mut stream = Stream(303);
        for round in 0..8 {
            // Fisher-Yates over the blobs, which is the only freedom a caller
            // has: which block wrote a point, and in which order the blobs
            // arrive.
            for index in (1..written.len()).rev() {
                written.swap(index, stream.below(index + 1));
            }
            for layout in [Layout::Flat, Layout::Gridded] {
                let store = sealed(volume, &written, layout);
                let got = encode_points(&store.query(&Region::whole(&volume)).unwrap());
                assert_eq!(
                    got,
                    expected,
                    "round {round}, {}: the order moved when the blobs did",
                    layout.as_str()
                );
            }
        }

        // The tiebreak, pinned: the three coincident points come back ordered
        // by their bits — 1.0, 2.0, 4.0 — and not in the order they were
        // written, which was 4.0, 1.0, 2.0.
        let coincident = reference
            .query(&Region::new(&[7, 7, 7], &[1, 1, 1]))
            .unwrap();
        let weights: Vec<f64> = coincident
            .iter()
            .filter(|point| point.at == [7, 7, 7])
            .map(|point| point.weight)
            .collect();
        assert_eq!(
            weights[weights.len() - 3..],
            [1.0, 2.0, 4.0],
            "the tiebreak is not on the payload bits"
        );
    }

    /// The store's own promise: what comes back is in the canonical order, not
    /// merely the same order twice.
    ///
    /// This became a real test when the sort left [`PointStore::query`]. While
    /// the store sorted centrally it could only have failed if the sort itself
    /// were wrong; now each index has to *store* its points in that order and a
    /// scan has to yield them in it, so this is what stands under the flat
    /// index's range scan and the gridded index's merge.
    #[test]
    fn a_query_answers_in_the_canonical_order() {
        let volume = [30usize, 30, 30];
        let points = scatter(volume, 6_000, 13);
        for layout in [Layout::Flat, Layout::Gridded] {
            let store = sealed(volume, &blobs(&points, 11, 14), layout);
            for region in regions(volume, 15).iter().take(30) {
                let found = store.query(region).unwrap();
                for pair in found.windows(2) {
                    assert_ne!(
                        canonical(&pair[0], &pair[1]),
                        Ordering::Greater,
                        "{}: {:?} came back before {:?}",
                        layout.as_str(),
                        pair[0],
                        pair[1]
                    );
                }
            }
        }
    }

    // ------------------------------------------------------- the residency --

    /// **What makes the stream worth having.** A whole-volume read must not hold
    /// a structure that grows with the point set.
    ///
    /// Peak allocation is not something a Rust test can measure without a
    /// custom allocator, so what is asserted is the structural bound instead:
    /// the merge's heap holds one cursor per bucket of a single grid
    /// cross-section, never more, and that cross-section does not grow when the
    /// set grows along the first axis. Two sets of the same density, one ten
    /// times as long, get ten times the points and ten times the buckets and the
    /// *same* live width.
    ///
    /// The rest of the residency claim rests on the types rather than on a
    /// measurement, and deliberately so: `GriddedScan` borrows the index and
    /// holds a heap of slice cursors, and `Flat::scan` is a slice iterator with
    /// two adapters. Neither has anywhere to put a copy of the answer. That is
    /// checkable by reading the two structs, which is a stronger guarantee than
    /// a high-water mark that happened not to be reached on one run.
    #[test]
    fn a_stream_holds_a_cross_section_and_not_the_set() {
        let short = [64usize, 64, 64];
        let long = [640usize, 64, 64];
        let small = Gridded::build(short, scatter(short, 20_000, 31));
        // Ten times the volume on the first axis, at the same density, so the
        // derivation gives the same bucket edge and the same cross-section.
        let large = Gridded::build(long, scatter(long, 200_000, 32));

        let whole_short = Region::whole(&short);
        let whole_long = Region::whole(&long);
        let narrow = small.merge(&whole_short).unwrap();
        let wide = large.merge(&whole_long).unwrap();
        assert_eq!(
            narrow.width(),
            wide.width(),
            "the live width followed the set instead of the cross-section"
        );
        assert!(
            large.buckets() >= 9 * small.buckets(),
            "the two indexes are not the ten-to-one pair this test needs: {} and {} bucket(s)",
            small.buckets(),
            large.buckets()
        );
        assert!(
            wide.width() * 100 < 200_000,
            "{} cursor(s) for 200000 point(s) is not a cross-section",
            wide.width()
        );

        // And the bound really holds while the stream runs: drain the larger of
        // the two and watch the heap at every step.
        let mut scan = large.merge(&whole_long).unwrap();
        let bound = scan.width();
        let mut yielded = 0usize;
        let mut peak = 0usize;
        while scan.next().is_some() {
            peak = peak.max(scan.heap.len());
            assert!(
                scan.heap.len() <= bound,
                "the merge held {} cursor(s), past the cross-section bound of {bound}",
                scan.heap.len()
            );
            yielded += 1;
        }
        assert_eq!(yielded, 200_000, "the stream did not yield the whole set");
        assert!(
            peak > 1,
            "the merge never held more than one cursor, so the bound is vacuous"
        );

        // An empty region opens no merge at all.
        assert!(small.merge(&Region::new(&[0, 0, 0], &[0, 0, 0])).is_none());
    }

    // --------------------------------------------------------- the buckets --

    /// A small region must read a small part of the index. Asserted as a count
    /// of buckets, which is a property of the structure; a duration would be a
    /// property of the machine.
    #[test]
    fn a_small_region_reads_few_buckets_and_the_whole_volume_reads_all_of_them() {
        let volume = [64usize, 64, 64];
        let points = scatter(volume, 20_000, 21);
        let index = Gridded::build(volume, points.clone());
        assert!(
            index.buckets() >= 64,
            "the derivation gave {} bucket(s) for 20000 points; this test cannot say \
             anything about a grid that coarse",
            index.buckets()
        );

        let one_voxel = Region::new(&[31, 31, 31], &[1, 1, 1]);
        assert_eq!(index.buckets_touched(&one_voxel), 1);
        assert_eq!(
            index.buckets_touched(&Region::new(&[0, 0, 0], &[0, 0, 0])),
            0
        );
        assert_eq!(
            index.buckets_touched(&Region::whole(&volume)),
            index.buckets()
        );

        // A region an eighth of the volume on a side reads a small fraction,
        // not merely fewer than all.
        let eighth = Region::new(&[8, 8, 8], &[8, 8, 8]);
        assert!(
            index.buckets_touched(&eighth) * 4 < index.buckets(),
            "{} of {} bucket(s) for an eighth-edge region",
            index.buckets_touched(&eighth),
            index.buckets()
        );
    }

    /// What the derivation aims at, checked as arithmetic rather than as prose:
    /// about [`TARGET_PER_BUCKET`] points per bucket, never more buckets than
    /// points, and an axis shorter than the derived edge clamped to itself.
    #[test]
    fn the_bucket_edge_lands_near_the_target_occupancy() {
        for (volume, count) in [
            ([64usize, 64, 64], 20_000usize),
            ([256, 256, 256], 1_000_000),
            ([100, 100, 1], 10_000),
        ] {
            let edge = bucket_edge(volume, count);
            let counts = bucket_counts(volume, edge).unwrap();
            let buckets = counts[0] * counts[1] * counts[2];
            assert!(buckets <= count.max(1), "{volume:?}: {buckets} > {count}");
            let occupancy = count as f64 / buckets as f64;
            assert!(
                occupancy >= TARGET_PER_BUCKET as f64 / 8.0
                    && occupancy <= TARGET_PER_BUCKET as f64 * 8.0,
                "{volume:?}/{count}: edge {edge:?} gives {buckets} bucket(s), {occupancy} \
                 point(s) each, which is not near the target of {TARGET_PER_BUCKET}"
            );
        }
        // A flat volume gets a flat bucket rather than a grid one bucket deep.
        assert_eq!(bucket_edge([100, 100, 1], 10_000)[2], 1);
        // And a sparse set never gets more slots than it has entries.
        let edge = bucket_edge([1_000, 1_000, 1_000], 3);
        let counts = bucket_counts([1_000, 1_000, 1_000], edge).unwrap();
        assert!(counts[0] * counts[1] * counts[2] <= 3);
    }

    // -------------------------------------------------------- the encoding --

    /// Moved from `ops::voxelize`, which is where these lived when the point
    /// type did.
    #[test]
    fn encoding_round_trips_and_a_broken_blob_is_refused() {
        let points = vec![
            Point::unit([0, 0, 0]),
            Point::weighted([3, 1, 4], -2.5),
            Point::weighted([7, 8, 9], f64::MIN_POSITIVE),
        ];
        let bytes = encode_points(&points);
        assert_eq!(bytes.len(), points.len() * WORDS_PER_POINT * 8);
        assert_eq!(decode_points(&bytes).unwrap(), points);
        assert_eq!(decode_points(&[]).unwrap(), Vec::new());

        // a whole number of words, but not a whole number of points
        let error = decode_points(&bytes[..8]).unwrap_err().to_string();
        assert!(error.contains("words per point"), "{error}");
        // not a whole number of words at all: `unpack_u64`'s own guard
        assert!(decode_points(&bytes[..9]).is_err());

        let poisoned = encode_points(&[Point::weighted([0, 0, 0], f64::NAN)]);
        let error = decode_points(&poisoned).unwrap_err().to_string();
        assert!(error.contains("not finite"), "{error}");

        // and a store refuses the same two, naming the block that wrote them
        let mut store = PointStore::new([16, 16, 16]).unwrap();
        let error = store.write([1, 2, 3], &bytes[..8]).unwrap_err().to_string();
        assert!(error.contains("[1, 2, 3]"), "{error}");
        assert!(error.contains("words per point"), "{error}");
        let error = store.write([1, 2, 3], &poisoned).unwrap_err().to_string();
        assert!(error.contains("not finite"), "{error}");
    }
}
