// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the shape of the problem — a set
// of positions produced piecewise and read by extent — not adapted from any
// implementation.
//
// **A point set, stored by position rather than by producer.**
//
// The problem this removes
// ------------------------
// Every other node this crate materialises has a partitioning that is a
// *function of its domain*: a level is voxels partitioned by chunk, a fragment
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
// Two states, and why reading early is an error rather than an emptiness
// ---------------------------------------------------------------------
// A store is [`State::Accumulating`] — blocks write, nobody reads — or
// [`State::Sealed`] — nobody writes, anybody queries. There is no third.
//
// A query on an accumulating store **fails, naming the state**. It does not
// return what has arrived so far, and it does not return nothing, because both
// of those are answers: the first is a fact about which writers happened to have
// finished, which is the decomposition showing through in the worst possible
// place, and the second is indistinguishable from a region that is genuinely
// empty. This crate fills an unwritten level with NaN rather than zeros for the
// same reason — an absence must not be able to pass for a value.
//
// Making the states a *type* rather than a convention is what turns "every
// writer must have finished" from something a caller has to remember into
// something the store enforces. It is checked at run time rather than by a
// typestate — `seal` transitions one value instead of consuming it and returning
// another — for one reason, and it is a compromise rather than a preference:
// whether every writer has finished is a *run-time* fact, known to whoever drove
// the phase, and a store is reached through a shared handle by the tasks that
// write it. A typestate would make the illegal call unwriteable, which is
// strictly better where the caller owns the value, and would put the check
// nowhere at all where it does not.
//
// The read interface is one method, and it streams
// ------------------------------------------------
// [`PointStore::scan`] takes a region and returns an **iterator** over the
// points in it. That is the whole of it, and the granularity of whatever index
// was built is invisible from the outside: nothing downstream can tell a flat
// store from a gridded one except by timing it. Keeping the interface at one
// method is not economy for its own sake — it is what makes that claim
// checkable, because there is exactly one thing to check.
//
// It is an iterator rather than a `Vec` because the case this store exists for
// is the one where the answer does not fit. A whole-volume query returning a
// `Vec` allocates a second copy of the entire set, which is precisely what an
// out-of-core structure must not do; a stream borrows the index and yields.
// [`PointStore::query`] collects, and is kept because collecting is often what a
// caller wants — but it is now a **choice visible at the call site** rather than
// the only thing on offer.
//
// Half-open, like every other region in this crate: a point at `region.start` is
// in the answer and a point at `region.start + region.shape` is not.
//
// The canonical order, and why the tiebreak is the payload
// -------------------------------------------------------
// Points come back sorted by **the coordinate triple, lexicographically, then by
// the weight's bits**. Two properties depend on it and both are load-bearing:
//
// * **the implementations are interchangeable.** If they returned the same set
//   in different orders, "invisible downstream" would be false the first time
//   somebody wrote the answer to a file or hashed it;
// * **a consumer that folds in store order gets a function of the data.**
//   Floating-point `+` does not associate, so a sum takes its value from the
//   order it is taken in. In store order that value depends on the point set and
//   on nothing else — not on the block grid, not on which worker finished first,
//   not on whether the index was rebuilt. `ops::voxelize` re-derives exactly
//   this order for exactly this reason; a consumer reading from here does not
//   have to.
//
// Lexicographic on the coordinate triple, rather than Morton: it is the order
// `voxelize` already sums in, so the two agree without a conversion; it makes a
// query a contiguous run on the first axis, which is what the flat index binary
// searches for; and it is readable in an error message, where a Morton code is
// a number nobody can place. Morton would give a query better locality in three
// dimensions and is the right answer if that ever dominates — it needs a fixed
// bit budget per axis, which is a limit this has no need to take on yet.
//
// **The tiebreak is `weight.to_bits()`, which is intrinsic to the point.** Not
// the source block, and not the order of insertion — both of those are facts
// about the run rather than about the data, and either would make the answer
// move when the cut moved. The store does not merely decline to use the source
// block: it never records it, so a tiebreak on it is unrepresentable rather than
// forbidden. Comparing bits is not comparing magnitudes (negative floats come
// back reversed) and that is fine, because this is a tiebreak and not a sort by
// weight: what is needed is a deterministic total order on the payload, and the
// payload *is* those bits. Two points that are coincident **and** carry the same
// bits are indistinguishable by any observation, so their relative order cannot
// matter and is not fixed.
//
// Why the order is a property of the *storage* and not of the query
// -----------------------------------------------------------------
// The obvious way to guarantee a canonical order is to sort the answer on the
// way out, in one place, so no implementation can get it wrong. **Streaming
// forbids it**: sorting requires having the whole answer, which is the copy the
// stream exists not to make. So one of three things has to give, and only one of
// them is right:
//
// * *each index sorts its own output* — back to one obligation in as many places
//   as there are indexes, and per call;
// * *the stream is unordered and the caller sorts* — which hands the
//   float-accumulation-order problem back to every consumer, and that problem is
//   the reason this module exists;
// * **each index stores its points already in the canonical order, so yielding
//   them in it is free.** This is the one taken.
//
// The obligation does move back into the implementations, and that is worth
// being clear about — but it changes shape on the way. It is no longer "return
// sorted", which is a promise made afresh on every call and can be broken by any
// one of them; it is "store sorted", which a constructor establishes once and a
// query cannot undo. For [`Flat`] it is what the structure already was: a sorted
// array, and a region query is a range scan that is in order because the array
// is. For [`Gridded`] it means each bucket holds its points in order and a
// multi-bucket region is answered by a **k-way merge** — a heap over one cursor
// per live bucket, `O(log k)` per point, materialising nothing.
//
// The merge's working set is a *cross-section*, not the whole span
// ---------------------------------------------------------------
// A naive merge holds a cursor for every bucket the region touches, and for a
// whole-volume query that is every bucket — which is a fraction of the point
// count and therefore still proportional to the set. The lexicographic order
// removes it: a point in a bucket at grid position `bx` on the first axis has
// its first coordinate inside that bucket's range, so **every point of grid
// plane `bx` precedes every point of plane `bx + 1`**, and planes cannot
// interleave. The merge therefore runs one plane at a time and its heap holds at
// most the buckets of one cross-section — `|second span| * |third span|`. For a
// whole-volume query that is `O(buckets^(2/3))`, and it does not grow at all
// when the set grows along the first axis. That is the property
// `a_stream_holds_a_cross_section_and_not_the_set` asserts.
//
// Two indexes, and why the choice may not show
// --------------------------------------------
// | index | shape | right when |
// |---|---|---|
// | flat | one sorted array; binary search, then an in-order scan | there are few points |
// | gridded | sorted buckets on a coarse grid; a k-way merge over the overlapping ones | there are many |
//
// [`PointStore::seal`] picks from the point count. The choice is *reportable* —
// [`PointStore::layout`] says which was built, so a slow run can be explained —
// and it is not *observable*: the same queries return the same bytes either way,
// which is what `both_indexes_answer_every_query_identically` asserts directly.
// That property is the whole design. If it ever fails, the abstraction is gone
// and the two are not implementations of one thing but two things with a shared
// name.
//
// Sealing is a barrier, and the distributed sort is a later problem
// ----------------------------------------------------------------
// `seal` sorts in one place, driven by the caller: **every writer must have
// finished before it is called**, and nothing here enforces that beyond refusing
// the writes that arrive afterwards. It is not wired into the executor's phase
// machinery and it is not a distributed sort. A store larger than one node can
// hold needs a sample-and-partition sort with the buckets as the partition, and
// that is real work with its own decomposition-invariance argument — it is the
// next problem and not this one. Naming it here so that the barrier is a stated
// limit rather than a discovered one.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{Error, Result};
use crate::fragment::{pack_u64, unpack_u64};
use crate::region::Region;

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
pub const WORDS_PER_POINT: usize = 4;

/// Points as a blob.
///
/// Four little-endian `u64`s per point, through [`pack_u64`], because that is
/// the shape every other fragment in this crate has and a second encoding would
/// be a second thing to get subtly different. The weight travels as
/// `f64::to_bits` rather than as a decimal or a fixed-point integer, so it round
/// trips exactly — a store whose weights were rounded on the way through would
/// have an order, and answers, that were reproducible only by accident.
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

/// The canonical order, entire: the coordinate triple, then the weight's bits.
///
/// Total on every pair of points that can be told apart; see the module header
/// for why that is the right strength and why the tiebreak is the payload.
fn canonical(left: &Point, right: &Point) -> Ordering {
    left.at
        .cmp(&right.at)
        .then_with(|| left.weight.to_bits().cmp(&right.weight.to_bits()))
}

/// Half-open containment, on all three axes.
fn holds(region: &Region, at: [usize; 3]) -> bool {
    (0..3).all(|axis| {
        at[axis] >= region.start[axis] && at[axis] - region.start[axis] < region.shape[axis]
    })
}

// ------------------------------------------------------------------ index --

/// What an index has to be able to do, and nothing else.
///
/// One method, on purpose. Everything a caller could otherwise ask — how many
/// buckets, how big they are, which one a point is in — is granularity, and
/// granularity showing through is the failure this whole module is arranged
/// against. The trait is small to *forbid* variety rather than to enable it.
///
/// **The stream is in the canonical order, and an implementation meets that by
/// storing its points in it** rather than by sorting on the way out. The module
/// header says why streaming forces this and why "store sorted" is a weaker
/// thing to ask than "return sorted": a constructor establishes it once, where
/// sorting on the way out would be a promise made afresh on every call.
///
/// **There is deliberately no way to hand a store a foreign index.** The trait
/// is public because it is the contract every guarantee in this module is
/// stated over, and a reader has to be able to see it; the two implementations
/// are private and [`PointStore::seal`] chooses between them. Admitting a third
/// from outside would turn "the choice is not observable" from something this
/// module tests into a claim about somebody else's code — and a third index is
/// a change to this file, where the agreement test is, rather than a plugin.
pub trait PointIndex: Send + Sync {
    /// Every point of the set that lies in `region`, in the canonical order,
    /// **without materialising them**.
    ///
    /// The returned iterator borrows the index and the region; an implementation
    /// that collected into a `Vec` and returned its `into_iter` would satisfy
    /// the signature and defeat the point of it.
    ///
    /// `region` is rank 3 and inside the volume; [`PointStore::scan`] has
    /// already checked both, so an implementation may assume them.
    fn scan<'a>(&'a self, region: &'a Region) -> Box<dyn Iterator<Item = Point> + 'a>;
}

/// Which index a store was sealed with.
///
/// Reportable so that a run's speed can be explained; never observable in an
/// answer. See the module header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Everything in one array, sorted once. A scan is a binary search and a
    /// walk, in order because the array is.
    Flat,
    /// Sorted buckets on a coarse grid. A scan reads only the overlapping ones,
    /// merged.
    Gridded,
}

impl Layout {
    pub fn as_str(self) -> &'static str {
        match self {
            Layout::Flat => "flat",
            Layout::Gridded => "gridded",
        }
    }
}

/// At or below this many points, [`PointStore::seal`] builds a [`Layout::Flat`]
/// index.
///
/// A few thousand points is a few tens of kilobytes in one contiguous run,
/// which a scan crosses about as fast as memory can supply it, and the bucket
/// grid's own arrays plus the per-bucket bookkeeping cost more to build and walk
/// than the scan they would save. The number is a threshold between two
/// implementations that give the same answer, so choosing it badly costs time
/// and can never cost correctness — which is exactly why it is a constant here
/// rather than something a caller sets, and why no test asserts a query result
/// against it.
const FLAT_LIMIT: usize = 4_096;

/// What the bucket-edge derivation aims for: points per bucket, on average over
/// the volume.
const TARGET_PER_BUCKET: usize = 64;

// ------------------------------------------------------------------- flat --

/// Everything in one array, in the canonical order.
///
/// The order is established by the constructor and nothing afterwards disturbs
/// it, which is what makes a scan in that order cost nothing.
struct Flat {
    points: Vec<Point>,
}

impl Flat {
    fn build(mut points: Vec<Point>) -> Self {
        points.sort_by(canonical);
        Self { points }
    }
}

impl PointIndex for Flat {
    fn scan<'a>(&'a self, region: &'a Region) -> Box<dyn Iterator<Item = Point> + 'a> {
        // The canonical order is lexicographic, so every point of the answer
        // lies in one contiguous run of the first axis. Find its start by
        // binary search, stop when the axis leaves the region, and filter the
        // other two as they go. Nothing here allocates and nothing here is
        // reordered: the run is already in the order it has to be yielded in.
        let lo = region.start[0];
        let hi = lo + region.shape[0];
        let from = self.points.partition_point(|point| point.at[0] < lo);
        Box::new(
            self.points[from..]
                .iter()
                .copied()
                .take_while(move |point| point.at[0] < hi)
                .filter(move |point| holds(region, point.at)),
        )
    }
}

// ---------------------------------------------------------------- gridded --

/// Points bucketed on a coarse grid, stored as one array plus the offsets that
/// cut it into buckets.
///
/// One allocation for the points and one for the offsets, rather than a map of
/// vectors: the bucket count is derived to be a fraction of the point count, so
/// the offsets are small, and a bucket's points being contiguous is what makes
/// reading one a scan rather than a chase.
///
/// **Each bucket holds its points in the canonical order**, which is what lets a
/// scan merge them rather than gather and sort them. The whole array is sorted
/// bucket-major with the canonical order inside each bucket, so that property
/// costs one sort at construction and nothing per query.
struct Gridded {
    edge: [usize; 3],
    counts: [usize; 3],
    /// `counts.product() + 1` offsets into `points`; bucket `b` is
    /// `points[starts[b]..starts[b + 1]]`.
    starts: Vec<usize>,
    points: Vec<Point>,
}

/// The bucket edge, in voxels, for `points` points spread over `volume`.
///
/// **What it aims at.** A bucket should hold about [`TARGET_PER_BUCKET`] points.
/// Fewer, and a query pays the per-bucket cost — a bounds computation and a
/// fresh cache line — more often than it reads anything; more, and a query for a
/// small region drags in a large multiple of what it returns, which is the whole
/// reason not to use the flat index. So the edge is the cube root of "the volume
/// one bucket's worth of points occupies at the average density":
/// `cbrt(TARGET_PER_BUCKET * |volume| / |points|)`.
///
/// **Cubic, in voxels.** The shape of the regions that will be asked for is not
/// known when the index is built, so no axis has earned a finer grid than
/// another. An axis shorter than the derived edge is clamped to its own length,
/// which is what makes a flat volume get a flat bucket rather than a grid one
/// bucket wide.
///
/// **Then a guard: never more buckets than points.** The clamp above, and a
/// volume that is mostly empty, can both leave the grid with more slots than
/// entries — an index spending memory and a walk on emptiness. Doubling the edge
/// until the count comes down is crude and terminates quickly (a big enough edge
/// is one bucket), and it is a correction rather than a policy: for any point set
/// dense enough to have been given this index in the first place, it does not
/// fire.
///
/// Derived, with nothing a caller can set. A configurable bucket size would be a
/// number that changes performance and not answers, which is precisely the kind
/// of knob that gets set once, wrongly, and never revisited.
fn bucket_edge(volume: [usize; 3], points: usize) -> [usize; 3] {
    let extent = volume[0] as f64 * volume[1] as f64 * volume[2] as f64;
    let wanted = (points / TARGET_PER_BUCKET).max(1);
    let mut edge = (extent / wanted as f64).cbrt().max(1.0);
    let cap = points.max(1);
    // Bounded by construction: each pass doubles, and an edge past the longest
    // axis leaves one bucket, which is under any cap.
    for _ in 0..64 {
        let candidate = clamp_edge(volume, edge);
        match bucket_counts(volume, candidate) {
            Some(counts)
                if counts[0]
                    .saturating_mul(counts[1])
                    .saturating_mul(counts[2])
                    <= cap =>
            {
                return candidate
            }
            _ => edge *= 2.0,
        }
    }
    volume
}

fn clamp_edge(volume: [usize; 3], edge: f64) -> [usize; 3] {
    let mut out = [1usize; 3];
    for axis in 0..3 {
        let rounded = if edge >= volume[axis] as f64 {
            volume[axis]
        } else {
            edge.ceil() as usize
        };
        out[axis] = rounded.clamp(1, volume[axis].max(1));
    }
    out
}

/// Buckets per axis, or `None` if the product would not fit a `usize`.
fn bucket_counts(volume: [usize; 3], edge: [usize; 3]) -> Option<[usize; 3]> {
    let mut counts = [0usize; 3];
    for axis in 0..3 {
        counts[axis] = volume[axis].div_ceil(edge[axis].max(1)).max(1);
    }
    counts[0].checked_mul(counts[1])?.checked_mul(counts[2])?;
    Some(counts)
}

impl Gridded {
    fn build(volume: [usize; 3], mut points: Vec<Point>) -> Self {
        let edge = bucket_edge(volume, points.len());
        let counts = bucket_counts(volume, edge).unwrap_or([1, 1, 1]);
        let total = counts[0] * counts[1] * counts[2];

        // Bucket-major, canonical inside a bucket. The bucket triple compares
        // lexicographically in the same order as its linear index, so one sort
        // gives both.
        points.sort_by(|left, right| {
            bucket_of(left.at, edge)
                .cmp(&bucket_of(right.at, edge))
                .then_with(|| canonical(left, right))
        });

        let mut starts = vec![0usize; total + 1];
        for point in &points {
            let linear = linear_bucket(bucket_of(point.at, edge), counts);
            starts[linear + 1] += 1;
        }
        for index in 1..starts.len() {
            starts[index] += starts[index - 1];
        }
        Self {
            edge,
            counts,
            starts,
            points,
        }
    }

    /// The inclusive bucket range `region` overlaps, or `None` for a region with
    /// no voxels in it.
    fn span(&self, region: &Region) -> Option<([usize; 3], [usize; 3])> {
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for axis in 0..3 {
            if region.shape[axis] == 0 {
                return None;
            }
            lo[axis] = region.start[axis] / self.edge[axis];
            hi[axis] = ((region.start[axis] + region.shape[axis] - 1) / self.edge[axis])
                .min(self.counts[axis] - 1);
            if lo[axis] > hi[axis] {
                return None;
            }
        }
        Some((lo, hi))
    }

    /// How many buckets a query for `region` would read.
    ///
    /// A measurement hook, and deliberately **not** on [`PointIndex`]: a caller
    /// that could ask this could tell a gridded store from a flat one without
    /// timing it, which is the property the module is arranged to keep. It is
    /// visible to this module's tests, which is where the claim "a small region
    /// reads a small part of the index" is asserted as a count rather than as a
    /// duration. Compiled only for them, so that the escape hatch cannot be
    /// reached from a build that is not the one measuring it.
    #[cfg(test)]
    fn buckets_touched(&self, region: &Region) -> usize {
        match self.span(region) {
            None => 0,
            Some((lo, hi)) => (0..3).map(|axis| hi[axis] - lo[axis] + 1).product(),
        }
    }

    /// How many buckets the derivation produced. Test-only, for the same reason
    /// as [`Self::buckets_touched`].
    #[cfg(test)]
    fn buckets(&self) -> usize {
        self.counts[0] * self.counts[1] * self.counts[2]
    }
}

fn bucket_of(at: [usize; 3], edge: [usize; 3]) -> [usize; 3] {
    [at[0] / edge[0], at[1] / edge[1], at[2] / edge[2]]
}

fn linear_bucket(bucket: [usize; 3], counts: [usize; 3]) -> usize {
    (bucket[0] * counts[1] + bucket[1]) * counts[2] + bucket[2]
}

impl Gridded {
    /// The merge itself, as its own type rather than behind the `dyn`.
    ///
    /// Split out so this module's tests can hold the concrete cursor and watch
    /// the heap, which is where the residency claim is checked. `None` is a
    /// region with no voxels in it.
    fn merge<'a>(&'a self, region: &'a Region) -> Option<GriddedScan<'a>> {
        let (lo, hi) = self.span(region)?;
        Some(GriddedScan {
            index: self,
            region,
            lo,
            hi,
            next_plane: lo[0],
            heap: BinaryHeap::new(),
        })
    }
}

impl PointIndex for Gridded {
    fn scan<'a>(&'a self, region: &'a Region) -> Box<dyn Iterator<Item = Point> + 'a> {
        match self.merge(region) {
            None => Box::new(std::iter::empty()),
            Some(merge) => Box::new(merge),
        }
    }
}

/// One bucket's remaining points, with the next one that qualifies already
/// found.
///
/// Ordered **backwards** on purpose: [`BinaryHeap`] is a max-heap and the merge
/// wants the canonically smallest head, so the comparison is reversed here
/// rather than every use being wrapped in `Reverse`.
struct Head<'a> {
    point: Point,
    rest: &'a [Point],
}

impl<'a> Head<'a> {
    /// The first point of `slice` that lies in `region`, and what follows it.
    ///
    /// Points that fail the region test are skipped here rather than filtered
    /// out in advance: filtering in advance is a copy, and a copy of a bucket is
    /// the materialisation this whole shape exists to avoid.
    fn take(mut slice: &'a [Point], region: &Region) -> Option<Self> {
        while let Some((first, rest)) = slice.split_first() {
            if holds(region, first.at) {
                return Some(Self {
                    point: *first,
                    rest,
                });
            }
            slice = rest;
        }
        None
    }
}

impl Ord for Head<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        canonical(&other.point, &self.point)
    }
}

impl PartialOrd for Head<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Head<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Head<'_> {}

/// A k-way merge over the buckets a region touches, one grid plane at a time.
///
/// The plane at a time is the part that matters, and the module header derives
/// it: the canonical order is lexicographic, so no point of grid plane `bx` can
/// follow a point of plane `bx + 1`, and the two therefore never need to be live
/// together. The heap holds one cursor per bucket of a single cross-section —
/// `|second span| * |third span|` — rather than one per bucket of the span, so a
/// whole-volume stream does not hold a structure proportional to the point set.
struct GriddedScan<'a> {
    index: &'a Gridded,
    region: &'a Region,
    lo: [usize; 3],
    hi: [usize; 3],
    /// The next grid plane on the first axis to load, once the heap runs dry.
    next_plane: usize,
    heap: BinaryHeap<Head<'a>>,
}

impl GriddedScan<'_> {
    /// How many cursors a full cross-section needs, which is the bound the heap
    /// never exceeds.
    ///
    /// Test-only, like the other measurement hooks on [`Gridded`]: a caller who
    /// could ask this could tell a gridded store from a flat one without timing
    /// it.
    #[cfg(test)]
    fn width(&self) -> usize {
        (self.hi[1] - self.lo[1] + 1) * (self.hi[2] - self.lo[2] + 1)
    }
}

impl Iterator for GriddedScan<'_> {
    type Item = Point;

    fn next(&mut self) -> Option<Point> {
        loop {
            if let Some(head) = self.heap.pop() {
                if let Some(next) = Head::take(head.rest, self.region) {
                    self.heap.push(next);
                }
                return Some(head.point);
            }
            // The plane is exhausted, so every point it held has been yielded
            // and none of the next plane's could have come before them.
            if self.next_plane > self.hi[0] {
                return None;
            }
            let first = self.next_plane;
            self.next_plane += 1;
            for second in self.lo[1]..=self.hi[1] {
                for third in self.lo[2]..=self.hi[2] {
                    let linear = linear_bucket([first, second, third], self.index.counts);
                    let bucket = &self.index.points
                        [self.index.starts[linear]..self.index.starts[linear + 1]];
                    if let Some(head) = Head::take(bucket, self.region) {
                        self.heap.push(head);
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ state --

/// Which half of its life a store is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Blocks write; nobody may read. There is no partial answer, because a
    /// partial answer is a fact about which writers happened to have finished.
    Accumulating,
    /// Nobody writes; anybody may query.
    Sealed,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            State::Accumulating => "accumulating",
            State::Sealed => "sealed",
        }
    }
}

// ------------------------------------------------------------------ store --

/// A point set, written per block and read by region.
///
/// See the module header for the two states, the canonical order and why the
/// index is not observable.
pub struct PointStore {
    volume: [usize; 3],
    /// Held while accumulating; taken by [`Self::seal`]. Decoded on the way in
    /// rather than at seal so that a malformed blob is refused while the block
    /// that wrote it is still known.
    pending: Vec<Point>,
    index: Option<Box<dyn PointIndex>>,
    layout: Option<Layout>,
    /// Kept separately from `pending`, because `pending` is emptied by sealing
    /// and "how many points does this store hold" must keep its answer.
    count: usize,
}

impl PointStore {
    /// An empty, accumulating store over `volume`.
    ///
    /// A volume with a zero-length axis holds no voxels, so no point could ever
    /// be inside it and no query could ever be answered; that is refused here
    /// rather than turned into a store that is silently always empty.
    pub fn new(volume: [usize; 3]) -> Result<Self> {
        if volume.iter().any(|&length| length == 0) {
            return Err(Error::invalid(format!(
                "a point store over {volume:?} has no voxels for a point to be at, so every \
                 write would be refused and every query would answer nothing. A volume with a \
                 zero-length axis is a mistake upstream rather than an empty store."
            )));
        }
        Ok(Self {
            volume,
            pending: Vec::new(),
            index: None,
            layout: None,
            count: 0,
        })
    }

    pub fn volume(&self) -> [usize; 3] {
        self.volume
    }

    pub fn state(&self) -> State {
        if self.index.is_some() {
            State::Sealed
        } else {
            State::Accumulating
        }
    }

    /// Which index was built, or `None` while accumulating.
    pub fn layout(&self) -> Option<Layout> {
        self.layout
    }

    /// How many points have been written.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
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
    pub fn write(&mut self, block: [usize; 3], bytes: &[u8]) -> Result<()> {
        if self.index.is_some() {
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
        for (index, point) in points.iter().enumerate() {
            for axis in 0..3 {
                if point.at[axis] >= self.volume[axis] {
                    return Err(Error::invalid(format!(
                        "point {index} of block {block:?} is at {:?}, which is outside this \
                         store's volume {:?} on axis {axis}. A point store does not care which \
                         block a point came from — that is the whole point of it — but a \
                         coordinate outside the volume is one no query can ever name, so it \
                         would be kept and never returned.",
                        point.at, self.volume
                    )));
                }
            }
        }
        self.count += points.len();
        self.pending.extend(points);
        Ok(())
    }

    /// Sort what has been written and build an index for it, choosing from the
    /// point count.
    ///
    /// **This is a barrier.** Every writer must have finished; nothing here can
    /// check that, and the most it does is refuse the writes that arrive
    /// afterwards. The sort happens in one place, in this process — a
    /// distributed sort over a set too large for one node is the later problem
    /// and the module header says what it would take.
    pub fn seal(&mut self) -> Result<()> {
        let layout = if self.count <= FLAT_LIMIT {
            Layout::Flat
        } else {
            Layout::Gridded
        };
        self.seal_as(layout)
    }

    /// Seal with a named index rather than the derived one.
    ///
    /// Here so that the two can be built over the same input and compared, which
    /// is how "the choice is not observable" is asserted rather than asserted
    /// about. It is **not** a tuning knob: the two answer identically, so the
    /// only thing choosing differently from [`Self::seal`] can do is make a run
    /// slower.
    pub fn seal_as(&mut self, layout: Layout) -> Result<()> {
        if self.index.is_some() {
            return Err(Error::invalid(format!(
                "this point store is already {}. Sealing twice would either discard the index \
                 and rebuild it — work with no effect, since nothing can have been written \
                 since — or quietly mean something different from what it says.",
                State::Sealed.as_str()
            )));
        }
        let points = std::mem::take(&mut self.pending);
        let index: Box<dyn PointIndex> = match layout {
            Layout::Flat => Box::new(Flat::build(points)),
            Layout::Gridded => Box::new(Gridded::build(self.volume, points)),
        };
        self.index = Some(index);
        self.layout = Some(layout);
        Ok(())
    }

    /// A stream of every point in `region`, in the canonical order.
    ///
    /// **This is the read interface.** It borrows the store and the region and
    /// yields as it goes; at no point is the answer held, which is what makes a
    /// whole-volume read possible over a set that a second copy of would not
    /// fit. [`Self::query`] collects it, and is a convenience rather than the
    /// primitive.
    ///
    /// Half-open: a point at `region.start` is in the answer, one at
    /// `region.start + region.shape` is not.
    ///
    /// A region reaching outside the volume is refused rather than clipped. The
    /// store's domain is the volume; a question about somewhere outside it has
    /// no answer, and "nothing there" is the answer a genuinely empty region
    /// gets, so returning it would make the two indistinguishable.
    pub fn scan<'a>(&'a self, region: &'a Region) -> Result<Box<dyn Iterator<Item = Point> + 'a>> {
        let Some(index) = self.index.as_ref() else {
            return Err(Error::invalid(format!(
                "this point store is {}, not {}, so it has no answer to give: {} point(s) have \
                 been written and no index has been built. An answer now would be a fact about \
                 which writers happened to have finished rather than about the point set, and \
                 an empty one would be indistinguishable from a region that really is empty. \
                 Call seal() once every writer has finished.",
                State::Accumulating.as_str(),
                State::Sealed.as_str(),
                self.count
            )));
        };
        if region.ndim() != 3 {
            return Err(Error::invalid(format!(
                "a point store is queried with a 3-D region, got rank {}",
                region.ndim()
            )));
        }
        region.check_within(&self.volume, "point store query region")?;
        Ok(index.scan(region))
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
            .field("volume", &self.volume)
            .field("state", &self.state().as_str())
            .field("points", &self.count)
            .field("layout", &self.layout.map(Layout::as_str))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
