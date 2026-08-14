// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation —
// one point per connected region, at the region's centroid — not adapted from
// any implementation of it.
//
// **A mask of connected regions in, one point per region out.**
//
// This is the join between the two worlds this crate holds. `crate::points` is
// where a set of positions is stored, sorted and queried; `ops::voxelize` is
// where one is rendered back into pixels. Until this op there was no *producer*
// for either: a point set could be written by hand or by a test and by nothing
// that had looked at a volume. This is the op that looks at a volume.
//
// The shape, and why it is two phases
// -----------------------------------
// A connected region is a global object — it can be arbitrarily long and
// arbitrarily crooked, so no halo of any size can decide where one ends. That is
// `ops::fill`'s argument and its header is where it is made. So this is the third
// **fragment-and-join** op in the crate, on the same program `ops::components`
// holds:
//
// | phase | shape | what it does |
// |---|---|---|
// | 0 | `volume -> fragments` | label the block's regions locally at reach 0; emit the block's six face planes plus, per label, a voxel count and a per-axis position sum |
// | 1 | `fragments -> fragments` | read every block's faces, close the local labels into global components with a union-find, add the counts and the sums over each component, and emit the points for the components **this** block owns |
//
// It differs from the two ops before it in one structural way, and the way is
// worth naming: **neither phase writes a level.** `fill` and `regional` both end
// by rewriting a label volume into a mask, so phase 0 has to write the labels
// down for phase 1 to read back. Here the answer is not a volume at all — it is
// a handful of points — and everything phase 1 needs is already in the fragment.
// The labels are a within-block temporary, so nothing allocates a `u32` level the
// size of the volume to hold a numbering nobody reads. That makes this the first
// op in `ops/` of `fragment.rs`'s second shape, `fragments -> fragments`, and
// the guard on it is the output stream's `Coverage::EveryBlock`: a phase that
// writes no level is not constrained by the tiling check at all, so the coverage
// declaration is the only thing that can fail, and it is declared.
//
// The accumulators, which are why this is tractable at all
// --------------------------------------------------------
// A centroid looks like it needs the whole region in one place. It does not. Per
// component two quantities are enough:
//
// * `count` — how many voxels it has;
// * `sums[axis]` — the sum of that axis' coordinate over its voxels, in
//   **volume** coordinates rather than block-local ones.
//
// Both are sums, so both are **associative and commutative**, and the merge
// across a seam is therefore *exact* rather than approximate: a region cut into
// four blocks contributes four partial sums that add to the same total the whole
// region would have given, in any order they happen to arrive. There is no
// tolerance anywhere in this op and no place where one could hide. That is the
// property `moments_merge_in_any_order_and_the_seam_is_exact` asserts and it is
// what the whole design rests on.
//
// They are **integers**, and that is a decision rather than an accident. A
// running mean in `f64` would be the obvious alternative and it would be
// order-dependent in the last bits — which is to say the answer would be a
// function of how the volume was cut, which is the one defect this crate exists
// to prevent. Integer addition has no such freedom. The bound is worth stating
// because it is the price: the first moment of a component covering an entire
// cubic volume of edge `L` is about `L^4 / 2`, so a `u64` accumulator carries any
// volume up to about `L = 65536` on a side — 2.8e14 voxels — exactly. Past that
// the sum is **refused** rather than wrapped; see [`Moments::add`]. For
// comparison, an `f64` accumulator would stop being exact at `L^4 / 2 > 2^53`,
// which is `L` of about 11500 — 1.5e12 voxels, which is a volume this crate is
// meant to be pointed at. So the integers are not pedantry: they are a factor of
// nearly two hundred in the volume that can be answered exactly.
//
// The centroid is fractional and a point's position is not
// --------------------------------------------------------
// `sums[axis] / count` is a rational, and `Point::at` is a voxel. The rule here
// is: **accumulate exactly, divide once, round once, and round half up.**
//
// * *accumulate exactly* — integers, as above, so the ratio is the same rational
//   whatever the decomposition;
// * *divide once* — never per block, so there is no partial rounding to
//   accumulate;
// * *round once, half up* — `floor(sums / count + 1/2)`, computed as
//   `(2 * sums + count) / (2 * count)` in integer arithmetic. No floating point
//   is involved at any step, so "does a centroid at exactly `x.5` round the same
//   way under every cut" is not a question about a division's last bit: the two
//   cuts compute the same two integers and take the same quotient. Half goes up,
//   towards the far end of the axis, which is `f64::round`'s convention for the
//   non-negative numbers a coordinate is.
//
// The rounded centroid is always **inside the volume**: it lies between the
// component's smallest and largest coordinate on each axis, and rounding a value
// no greater than an integer cannot exceed it. It is not necessarily inside the
// *component* — the centroid of a ring is its hole, and of an L is off the L —
// and that is the definition rather than a defect. A caller who needs a point on
// the object wants a different operation.
//
// What a point carries, and why it is the count
// ---------------------------------------------
// `Point::weight` is the component's **voxel count**, as an `f64` — exact, since
// a count that exceeded `2^53` would need a volume of nine petavoxels.
//
// It is the count rather than `1.0` because it is the choice that loses nothing:
// a caller who wants the pure counting point set divides by it or ignores it,
// and one who wants a density weighted by region size gets it by rendering
// straight through `ops::voxelize` with no second pass over anything. It also
// makes the point set self-describing in the one respect that matters when it is
// all that is left of the volume — how big the thing was — and `points.rs` says
// the weight is per point exactly so that "count the points" and "sum a quantity
// over the points" are one operation with a different column.
//
// Which block owns a point, and why exactly one does
// --------------------------------------------------
// Phase 1 runs the same merge in every block — it has to, because a fragment
// phase cannot hand its answer to a later phase without going through a level —
// so every block computes every component's centroid. What stops the same point
// being written N times is the ownership rule:
//
// > **the block whose core holds the component's centroid emits the point.**
//
// That is a **total function of the component**: the centroid is a function of
// the component alone, `BlockGrid` starts every core at `index * edge` so the
// cores tile the volume with no overlap and no gap, and the centroid is inside
// the volume — so exactly one core holds it, exactly one block emits the point,
// and none is lost or duplicated. [`owner_of`] is the whole of the arithmetic.
//
// The rule was chosen to agree with `ops::voxelize`, which **requires** that a
// point in block B's fragment lie in B's core and refuses anything else — its
// header names "the centroid of an object that straddles a seam" as the producer
// it cannot take. This op is that producer, and it re-keys its own points, so the
// two compose directly. Feeding a `PointStore` needs no such rule at all, since a
// store is keyed by position; the rule costs nothing there and buys the pipe to
// `voxelize`.
//
// The weighted centroid, which is not here
// ----------------------------------------
// A centroid weighted by a per-voxel quantity — an intensity-weighted centre
// rather than a geometric one — is the obvious extension and it is **not
// implemented**. Two reasons, and the second is the one that would still hold if
// the first were fixed:
//
// * it needs a **second input array** beside the mask, and a `FragmentOp` reads
//   one level. That capability is being built; this op does not get to invent a
//   second mechanism for it in the meantime;
// * the tempting shortcut — take the weight from the mask level itself, since a
//   mask may arrive as any width and `is_set` only asks whether a voxel is
//   non-zero — would give up the exactness above. The weights would be arbitrary
//   reals, `f64` addition does not associate, and the seam merge would stop being
//   exact and start being *nearly* exact, which is precisely the
//   decomposition-dependent answer the integer accumulators exist to rule out. A
//   weighted variant has to say what it does about that, and the honest answers
//   are a fixed-point accumulator or a stated tolerance. Neither is a hook; both
//   are a design.
//
// So the hook is left unbuilt rather than half-built, and this is where it goes.
//
// Connectivity is six
// -------------------
// Regions are face-connected components of the set voxels, which is what
// `ops::components` is built for and why: under 6-connectivity a component
// crosses a seam only through the shared plane, so a block's whole contribution
// to the merge is six planes. Two voxels touching only at an edge or a corner are
// two regions and get two points. That is a limit and it is stated as one; it is
// not a limit of the framework.
//
// What this costs
// ---------------
// `ops::fill`'s costs, minus the pixels. Phase 0 is halo-free and embarrassingly
// parallel. Phase 1 declares a whole-lattice fragment reach, so on `N` blocks it
// moves `N` fragments to each of `N` blocks and runs the same union-find `N`
// times; what it does *not* do is read a level, because it does not need one, so
// the read amplification `fill`'s header measures is not paid here at all. The
// fragment is six planes of labels plus eight words per label, against a block of
// pixels — the same shape, for the same reason.

use std::collections::BTreeMap;

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::assemble::{Phase, PlanBuilder};
use crate::decomposition::Decomposition;
use crate::dtype::Dtype;
use crate::env::BlockBuf;
use crate::error::{Error, Result};
use crate::fragment::{
    fragment_phase, BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput,
};
use crate::geometry::BlockGrid;
use crate::points::{encode_points, Point};
use crate::sidecar::Lifecycle;

use super::components::{
    bytes_to_words, empty_planes, expect_end, label_members_into, planes_of, push_planes,
    read_header, take_planes, walk_seams, words_to_bytes, FacePlanes, LabelIndex, Union,
    UNLABELLED,
};
use super::fill::as_mask;
use super::shapes_agree;

// ------------------------------------------------------------ the moments --

/// What a component contributes to its own centroid: how many voxels it has and
/// where they are.
///
/// The zeroth and first moments, and nothing else — a centroid needs no more,
/// and anything else in here would be a field that has to be merged correctly
/// for no reason. Both fields are exact integers and both merge by addition; the
/// module header is where that is argued for and where the bound is stated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Moments {
    /// Voxels in the component.
    pub count: u64,
    /// Per axis, the sum of that coordinate over the component's voxels, in
    /// **volume** coordinates. Block-local ones would make the merge a function
    /// of where the blocks were, which is the whole thing being avoided.
    pub sums: [u64; 3],
}

impl Moments {
    /// The identity of the merge: no voxels, and therefore no centroid.
    pub const EMPTY: Self = Self {
        count: 0,
        sums: [0, 0, 0],
    };

    /// Add one voxel, at a **volume** coordinate.
    ///
    /// Checked rather than wrapping. An overflowing sum is not a slightly wrong
    /// centroid, it is an arbitrary one, and a silently arbitrary answer is the
    /// failure mode this crate is arranged against. The header says how far the
    /// accumulator carries before this can fire: a cubic volume of about 65536 on
    /// a side, which is 2.8e14 voxels.
    pub fn add(&mut self, at: [usize; 3]) -> Result<()> {
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| overflowed("count"))?;
        for axis in 0..3 {
            self.sums[axis] = self.sums[axis]
                .checked_add(at[axis] as u64)
                .ok_or_else(|| overflowed("position sum"))?;
        }
        Ok(())
    }

    /// Fold another partial accumulation into this one.
    ///
    /// **This is the whole of the seam.** Addition is associative and
    /// commutative, so a component cut into any number of pieces gives the same
    /// totals as the same component seen whole, whatever order the pieces arrive
    /// in — which is why a centroid across a seam is *exact* rather than close.
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        self.count = self
            .count
            .checked_add(other.count)
            .ok_or_else(|| overflowed("count"))?;
        for axis in 0..3 {
            self.sums[axis] = self.sums[axis]
                .checked_add(other.sums[axis])
                .ok_or_else(|| overflowed("position sum"))?;
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The centroid, rounded to a voxel, or `None` for a component with no
    /// voxels in it.
    ///
    /// `floor(sums / count + 1/2)` per axis, taken as
    /// `(2 * sums + count) / (2 * count)` in integer arithmetic so that nothing
    /// rounds twice and nothing rounds in binary floating point. Half goes up.
    /// See the module header for why that is decomposition-independent by
    /// construction rather than by testing, and for why the answer is always
    /// inside the volume.
    ///
    /// In `u128` for the doubling alone: the sums are `u64` by the time they get
    /// here, and `2 * sums + count` is the one expression that could leave the
    /// range they were checked into.
    pub fn centroid(&self) -> Option<[usize; 3]> {
        if self.count == 0 {
            return None;
        }
        let count = self.count as u128;
        let mut at = [0usize; 3];
        for axis in 0..3 {
            let rounded = (2 * self.sums[axis] as u128 + count) / (2 * count);
            // Bounded by the largest coordinate that was added, which came from
            // a `usize`, so this cannot truncate.
            at[axis] = rounded as usize;
        }
        Some(at)
    }

    /// The point this component becomes: its centroid, carrying its voxel count.
    ///
    /// The count converts to `f64` exactly — see the module header — so the
    /// weight is the count rather than an approximation of it.
    pub fn point(&self) -> Option<Point> {
        self.centroid()
            .map(|at| Point::weighted(at, self.count as f64))
    }
}

fn overflowed(what: &str) -> Error {
    Error::invalid(format!(
        "a region's {what} overflowed a 64-bit accumulator. The first moment of a component \
         covering a cubic volume of edge L is about L^4 / 2, so this cannot happen below about \
         L = 65536; a wrapped sum would be an arbitrary centroid rather than an imprecise one, \
         so it is refused here."
    ))
}

// ------------------------------------------------------------ the kernels --

/// Label the **set** voxels of `mask` into `out`, six-connected, and return how
/// many regions were found.
///
/// The traversal is `components::label_members_into`, shared with `ops::fill`,
/// and the whole of what is said here is the membership test: a voxel belongs to
/// a region exactly when the mask sets it. A clear voxel is left [`UNLABELLED`]
/// and is in no region.
pub fn label_regions_into(mask: ArrayView3<'_, bool>, out: ArrayViewMut3<'_, u32>) -> Result<u32> {
    shapes_agree(mask.shape(), out.shape(), "label_regions_into")?;
    let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    label_members_into(shape, |at| mask[at], out)
}

/// The moments of every label of `labels`, in label order, over an array whose
/// lowest voxel sits at `offset` of the volume.
///
/// `offset` is what makes the sums volume coordinates rather than block-local
/// ones, and it is an argument rather than a field because the whole-volume
/// reference passes `[0, 0, 0]` and a block passes its own corner — the same
/// kernel, called twice, which is what makes the reference a reference.
pub fn moments_of_labels(
    labels: ArrayView3<'_, u32>,
    count: u32,
    offset: [usize; 3],
) -> Result<Vec<Moments>> {
    let shape = [labels.shape()[0], labels.shape()[1], labels.shape()[2]];
    let mut moments = vec![Moments::EMPTY; count as usize];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let label = labels[[i, j, k]];
                if label == UNLABELLED {
                    continue;
                }
                let slot = moments.get_mut(label as usize - 1).ok_or_else(|| {
                    Error::invalid(format!(
                        "label {label} is outside the {count} label(s) this array was said to \
                         hold, so the labels and the count came from two different runs."
                    ))
                })?;
                slot.add([offset[0] + i, offset[1] + j, offset[2] + k])?;
            }
        }
    }
    Ok(moments)
}

/// The points of a set of components, in the canonical order of
/// [`crate::points`].
///
/// Components with no voxels are dropped rather than turned into a point at the
/// origin: an empty accumulator is a `(block, label)` slot that no voxel was
/// found for, which is a thing that exists in the flat numbering and not in the
/// volume.
///
/// Sorted here so that a block's blob is a function of the component set and not
/// of the order the union-find happened to produce its roots in. It is the same
/// order `PointStore` keeps, so a store built from these blobs re-sorts nothing
/// that was not already in place.
pub fn centroid_points(components: &[Moments]) -> Vec<Point> {
    let mut points: Vec<Point> = components.iter().filter_map(Moments::point).collect();
    points.sort_by(|left, right| {
        left.at
            .cmp(&right.at)
            .then_with(|| left.weight.to_bits().cmp(&right.weight.to_bits()))
    });
    points
}

/// The whole-volume answer: the same kernels, called once, over everything.
///
/// **Not a second implementation.** It is what the blocked path is measured
/// against, so if it were written a second way a disagreement would be a
/// modelling difference rather than a decomposition bug. `pub` because the
/// acceptance suite in `tests/` is a separate crate and needs exactly this.
pub fn detect_regions(mask: ArrayView3<'_, bool>) -> Result<Vec<Point>> {
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    let count = label_regions_into(mask, labels.view_mut())?;
    let moments = moments_of_labels(labels.view(), count, [0, 0, 0])?;
    Ok(centroid_points(&moments))
}

// -------------------------------------------------------------- ownership --

/// Which block of `grid` owns `at`.
///
/// `BlockGrid` starts every core at `index * edge`, so this division *is* the
/// core that holds the coordinate — cores tile the volume, so the answer exists
/// for every voxel of it and is unique. Clamped to the lattice for the same
/// reason the last core is short: a coordinate in the final, partial block still
/// divides to the last index.
///
/// `pub` because it is half of the ownership rule and a test that recomputed it
/// would be asserting this code against itself; the other half is that the
/// centroid it is asked about is a function of the component alone.
pub fn owner_of(grid: &BlockGrid, at: [usize; 3]) -> [usize; 3] {
    let edge = grid.block();
    let counts = grid.blocks_per_axis();
    let mut index = [0usize; 3];
    for axis in 0..3 {
        index[axis] = (at[axis] / edge[axis]).min(counts[axis] - 1);
    }
    index
}

/// The points of `components` that `block` owns, in the canonical order.
///
/// Exactly one block of the lattice returns any given point, because
/// [`owner_of`] is a function and the centroid is a function of the component.
pub fn points_owned_by(components: &[Moments], grid: &BlockGrid, block: [usize; 3]) -> Vec<Point> {
    let mine: Vec<Moments> = components
        .iter()
        .filter(|moments| match moments.centroid() {
            None => false,
            Some(at) => owner_of(grid, at) == block,
        })
        .copied()
        .collect();
    centroid_points(&mine)
}

// --------------------------------------------------------------- fragment --

/// What one block tells the merge about itself.
///
/// Six planes of labels and one accumulator per label, and nothing else.
/// Deliberately not the block's label volume: what the merge needs is which
/// regions meet across a seam and what each contributes, and only the faces can
/// meet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionMoments {
    /// How many regions this block found.
    pub labels: u32,
    /// Per label, the voxel count and position sums of that region's part of
    /// this block. Partial by construction — the merge is what makes them whole.
    pub moments: Vec<Moments>,
    /// The six faces, ordered `axis * 2 + side` with side 0 low and 1 high, each
    /// as `(shape, labels)` in row-major order over the two axes that are not
    /// this face's.
    pub faces: FacePlanes,
}

impl RegionMoments {
    /// Read a block's faces off its label volume and pair them with the
    /// accumulators.
    pub fn of(labels: ArrayView3<'_, u32>, count: u32, moments: Vec<Moments>) -> Result<Self> {
        if moments.len() != count as usize {
            return Err(Error::invalid(format!(
                "{count} label(s) but {} accumulator(s)",
                moments.len()
            )));
        }
        Ok(Self {
            labels: count,
            moments,
            faces: planes_of(labels),
        })
    }

    /// The empty report: a block with nothing to say, which is what an
    /// accounting run produces and is a different fact from no fragment at all.
    pub fn empty() -> Self {
        Self {
            labels: 0,
            moments: Vec::new(),
            faces: empty_planes(),
        }
    }

    /// A self-describing byte form: little-endian `u32` throughout, with a magic
    /// and a version in front, and each accumulator as four `u64`s — the count
    /// and the three sums — low word first.
    ///
    /// The words rather than a float or a decimal, because the merge adds these
    /// and the whole claim of the op is that the addition is exact. A number that
    /// had been through a lossy encoding on the way to the merge would make the
    /// claim false in transit, where nothing would catch it.
    pub fn encode(&self) -> Vec<u8> {
        let mut words: Vec<u32> = vec![MAGIC, VERSION, self.labels];
        for moments in &self.moments {
            push_u64(&mut words, moments.count);
            for axis in 0..3 {
                push_u64(&mut words, moments.sums[axis]);
            }
        }
        push_planes(&self.faces, &mut words);
        words_to_bytes(&words)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        const NOUN: &str = "a region-moments fragment";
        let words = bytes_to_words(bytes, NOUN)?;
        let labels = read_header(&words, MAGIC, VERSION, NOUN)?;
        let count = labels as usize;

        let mut at = 3usize;
        let payload = count.checked_mul(WORDS_PER_LABEL).ok_or_else(|| {
            Error::invalid(format!(
                "{NOUN} declares more labels than the address space holds"
            ))
        })?;
        if words.len() < at + payload {
            return Err(Error::invalid(format!(
                "{NOUN} ends inside its accumulators"
            )));
        }
        let mut moments = Vec::with_capacity(count);
        for record in words[at..at + payload].chunks_exact(WORDS_PER_LABEL) {
            moments.push(Moments {
                count: take_u64(record, 0),
                sums: [
                    take_u64(record, 2),
                    take_u64(record, 4),
                    take_u64(record, 6),
                ],
            });
        }
        at += payload;

        let faces = take_planes(&words, &mut at, NOUN)?;
        expect_end(&words, at, NOUN)?;
        Ok(Self {
            labels,
            moments,
            faces,
        })
    }
}

/// `"DETC"` little-endian. Distinct from `fill`'s and `regional`'s, which is the
/// point: three fragments in this crate are six planes and a per-label payload,
/// so a stream name reused by two of them would otherwise decode one as another.
const MAGIC: u32 = 0x4354_4544;
const VERSION: u32 = 1;

/// Four `u64`s — the count and the three sums — as eight `u32` words.
const WORDS_PER_LABEL: usize = 8;

fn push_u64(words: &mut Vec<u32>, value: u64) {
    words.push(value as u32);
    words.push((value >> 32) as u32);
}

fn take_u64(words: &[u32], at: usize) -> u64 {
    (words[at + 1] as u64) << 32 | words[at] as u64
}

// -------------------------------------------------------------- the merge --

/// Close every block's local regions into global components and total the
/// accumulators over each.
///
/// `reports` is keyed by block index and must hold every block of `counts`; a
/// missing block is refused rather than assumed empty, because "absent" and
/// "present with nothing to say" are different facts and only one of them is a
/// block that ran.
///
/// **A seam meeting always unions**, as in `ops::fill` and unlike
/// `ops::regional`: two set voxels that touch across a seam are one region and
/// there is nothing to compare. The labelling already ran on the mask, so both
/// sides being labelled at all is the whole of the condition.
///
/// The totals are returned in root order of the flat `(block, label)` numbering,
/// which is deterministic; the *set* of components is a function of the mask
/// alone, and the order they come back in is not observable in the answer
/// because [`centroid_points`] sorts.
pub fn merge_moments(
    reports: &BTreeMap<[usize; 3], RegionMoments>,
    counts: [usize; 3],
) -> Result<Vec<Moments>> {
    let index = LabelIndex::build(reports, counts, |report| report.labels)?;
    let parts = index.gather(reports, |report| &report.moments[..], Moments::EMPTY);
    let mut sets = Union::new(index.total());

    walk_seams(
        reports,
        counts,
        &index,
        |report| &report.faces,
        |a, b| sets.union(a, b),
    )?;

    // Folded onto the roots after every union rather than as they happen, for
    // `Union::fold_or`'s reason: the answer must not depend on the order the
    // unions arrived in. Here it could not anyway — addition commutes — but the
    // shape is the same and the cheaper claim is the one that stays true if the
    // accumulator ever grows a field that does not commute.
    let mut totals = vec![Moments::EMPTY; index.total()];
    for (node, part) in parts.iter().enumerate() {
        let root = sets.find(node);
        totals[root].merge(part)?;
    }
    Ok(totals)
}

// ---------------------------------------------------------------- phases --

/// Phase 0: label each block's regions and say what crosses its faces.
///
/// **Reach zero.** A block-local labelling reads nothing outside its own core;
/// everything that would need a neighbour is in the fragment instead. That is the
/// point of the split — the expensive, per-voxel half is fully parallel and
/// halo-free, and only the cheap, global half is a reduction.
///
/// **Writes no level.** The labels are consumed inside this call — the face
/// planes and the accumulators are everything phase 1 reads — so there is
/// nothing to hand on, and the module header says what that saves.
pub struct LabelRegionsOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
}

impl LabelRegionsOp {
    pub fn new(name: &'static str, stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
        }
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }
}

impl FragmentOp for LabelRegionsOp {
    fn name(&self) -> &'static str {
        self.name
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
            // Every block, always. A block whose mask is empty still has six
            // faces of zeros and a label count of nought, and the merge needs to
            // see that rather than infer it from an absence.
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
                RegionMoments::empty().encode(),
            ));
        };
        let mask = as_mask(pixels)?;

        let mut labels = Array3::<u32>::zeros(mask.raw_dim());
        let count = label_regions_into(mask.view(), labels.view_mut())?;
        // The read extent's corner, which is the core's here because this phase
        // has no halo — and the sums have to be in volume coordinates or the
        // merge would be adding numbers from three different origins.
        let moments = moments_of_labels(labels.view(), count, at.at.offset)?;
        let report = RegionMoments::of(labels.view(), count, moments)?;
        Ok(BlockOutput::fragment(self.stream.clone(), report.encode()))
    }
}

/// Phase 1: close the components and emit one point per region.
///
/// Declares a **whole-lattice** fragment reach, which is what makes it a planning
/// barrier: nothing can be fused across it, because its answer depends on every
/// block.
///
/// Reads no pixels and writes none. What it writes is one blob of encoded points
/// per block — the points of the components that block owns — which is what
/// `PointStore::write` takes.
pub struct RegionPointsOp {
    name: &'static str,
    moments_stream: String,
    moments_phase: usize,
    points_stream: String,
    lifecycle: Lifecycle,
    lattice: [usize; 3],
}

impl RegionPointsOp {
    /// `moments_phase` is the phase whose blocks wrote the accumulators — part of
    /// the address rather than a default, for `FragmentInput`'s reason: a stream
    /// written by two phases holds two generations and "the fragments of stream
    /// s" is not a well-formed request.
    ///
    /// `lifecycle` is the points stream's, and `Lifecycle::Persistent` is the
    /// usual answer because this stream **is** the run's output.
    pub fn new(
        name: &'static str,
        moments_stream: impl Into<String>,
        moments_phase: usize,
        points_stream: impl Into<String>,
        lifecycle: Lifecycle,
        grid: &BlockGrid,
    ) -> Self {
        Self {
            name,
            moments_stream: moments_stream.into(),
            moments_phase,
            points_stream: points_stream.into(),
            lifecycle,
            lattice: grid.blocks_per_axis(),
        }
    }

    /// The same op, addressed by a [`Phase`] handle instead of a number.
    ///
    /// See [`RegionPointsOp::new`] for what the phase half of the address is
    /// for, and `crate::assemble::Phase` for why a handle is not a `usize`: a
    /// literal that is off by one reads a different generation of the stream and
    /// answers differently, with nothing to refuse it.
    pub fn reading(
        name: &'static str,
        moments_stream: impl Into<String>,
        moments: Phase,
        points_stream: impl Into<String>,
        lifecycle: Lifecycle,
        grid: &BlockGrid,
    ) -> Self {
        Self::new(
            name,
            moments_stream,
            moments.index(),
            points_stream,
            lifecycle,
            grid,
        )
    }

    /// The stream the points are written to.
    pub fn points_stream(&self) -> &str {
        &self.points_stream
    }

    /// The lattice this op was built for, which is also the reach it declares.
    pub fn lattice(&self) -> [usize; 3] {
        self.lattice
    }
}

impl FragmentOp for RegionPointsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    /// The whole lattice, stated as the lattice rather than as a large number.
    ///
    /// This is why the constructor takes a grid. "Everything" is a different
    /// integer on every lattice, and a saturating sentinel is not a way out: the
    /// reach is multiplied by the block edge to get a halo, and a sentinel
    /// overflows the geometry rather than clamping.
    fn inputs(&self) -> Vec<FragmentInput> {
        vec![
            FragmentInput::own(self.moments_stream.clone(), self.moments_phase)
                .with_reach(self.lattice),
        ]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.points_stream.clone(),
            self.lifecycle,
            // Every block, always — and here it is the only guard there is, since
            // this phase writes no level for the tiling check to bite on. A block
            // that owns no component writes a zero-length blob, which is present
            // and therefore checkable.
            Coverage::EveryBlock,
        )]
    }

    /// Gathered rather than streamed, unlike the other whole-lattice reader in
    /// this crate.
    ///
    /// `FragmentReduceOp` streams because it folds one number and never needs two
    /// fragments at once. The seam walk does: it compares block `b`'s high face
    /// against block `b + 1`'s low face, so the merge holds every report anyway
    /// and streaming would move the residency from the executor's gather into
    /// this op's own map without removing it. Saying `true` keeps the fetch count
    /// measurable from outside, which is what `fragment.rs` says the flag is for.
    fn gathers(&self) -> bool {
        true
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mut reports = BTreeMap::new();
        for (key, bytes) in at.fragments(&self.moments_stream) {
            reports.insert(key.block, RegionMoments::decode(bytes)?);
        }
        let components = merge_moments(&reports, at.grid.blocks_per_axis())?;
        let mine = points_owned_by(&components, at.grid, at.index);
        Ok(BlockOutput::fragment(
            self.points_stream.clone(),
            encode_points(&mine),
        ))
    }
}

/// The two phases, on one lattice.
///
/// Both are built with `fragment_phase`, so both halos come from the ops'
/// declarations rather than from this function: zero for the labelling, the whole
/// lattice for the points. `ops::fill`'s header explains why the second one has
/// to be that — the halo is the dependency edge between pipelined phases, not
/// merely a fetch extent — and here it is *only* that, because this phase reads
/// no pixels at all.
///
/// Neither phase declares a `dtype`: neither writes a level, so there is no level
/// whose width could be wrong, and `check_dtypes` skips both for that reason.
pub fn detect_phases(
    grid: BlockGrid,
    mask_dtype: Dtype,
    label: &LabelRegionsOp,
    points: &RegionPointsOp,
) -> Result<Decomposition> {
    let volume = grid.volume();
    let labelling = fragment_phase(label, grid.clone())?;
    let detecting = fragment_phase(points, grid)?;
    let plan = Decomposition {
        volume,
        dtype: mask_dtype,
        phases: vec![labelling, detecting],
        chain_reach: [0, 0, 0],
    };
    plan.check()?;
    Ok(plan)
}

/// The same two phases, **appended to a plan that already has some**.
///
/// [`detect_phases`] builds a whole `Decomposition`, so it is unusable as soon
/// as these two phases sit inside something larger; this is its body against a
/// builder. Neither phase declares an element type — the labelling writes
/// fragments and no pixels, and so does the emission — which is why this one is
/// shorter than `regional::append_to` and not because it is doing less.
///
/// `moments_stream` carries the per-block accumulators from the labelling to the
/// emission; the phase half of that address is wired here. `points_stream` is
/// the run's output, which is why its lifecycle is nearly always
/// [`Lifecycle::Persistent`].
///
/// **Both lifecycles are the caller's** and neither is defaulted, even though
/// one of them has an obvious answer. What a run leaves behind is a decision,
/// and this function is here to remove bookkeeping rather than to make
/// decisions on a caller's behalf where the caller cannot see them.
///
/// Returns the emission's phase, which is where the point blobs are keyed and
/// therefore where a reader has to look for them.
pub fn append_to(
    plan: &mut PlanBuilder,
    moments_stream: impl Into<String>,
    moments_lifecycle: Lifecycle,
    points_stream: impl Into<String>,
    points_lifecycle: Lifecycle,
) -> Result<Phase> {
    let moments_stream = moments_stream.into();
    let grid = plan.grid().clone();
    let moments = plan.fragments(LabelRegionsOp::new(
        "region labelling",
        moments_stream.clone(),
        moments_lifecycle,
    ))?;
    plan.fragments(RegionPointsOp::reading(
        "region points",
        moments_stream,
        moments,
        points_stream,
        points_lifecycle,
        &grid,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mask with the named voxels set.
    fn mask_of(shape: [usize; 3], set: &[[usize; 3]]) -> Array3<bool> {
        let mut mask = Array3::from_elem((shape[0], shape[1], shape[2]), false);
        for &at in set {
            mask[at] = true;
        }
        mask
    }

    fn points_of(mask: &Array3<bool>) -> Vec<Point> {
        detect_regions(mask.view()).unwrap()
    }

    // ------------------------------------------------------- the arithmetic --

    /// A symmetric region's centroid is its centre, and an L-shaped one's is
    /// **not** its bounding box's centre — which is the assertion that catches an
    /// implementation that tracked extents instead of moments.
    #[test]
    fn a_symmetric_region_centres_and_an_l_shape_does_not_centre_on_its_box() {
        // A 3 x 3 x 3 cube at 2..=4 on every axis: the centre is [3, 3, 3].
        let mut cube = Array3::from_elem((7, 7, 7), false);
        for i in 2..=4 {
            for j in 2..=4 {
                for k in 2..=4 {
                    cube[[i, j, k]] = true;
                }
            }
        }
        let found = points_of(&cube);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].at, [3, 3, 3]);
        assert_eq!(found[0].weight, 27.0);

        // An L in one plane, with arms of five and three so that the two
        // answers cannot coincide: five voxels along x from 0 to 4 and three up
        // from its far end, bounding box [0..=4] x [0..=2].
        let l = mask_of(
            [8, 8, 1],
            &[
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [4, 0, 0],
                [4, 1, 0],
                [4, 2, 0],
            ],
        );
        let found = points_of(&l);
        assert_eq!(found.len(), 1);
        // sums: axis 0 = 0+1+2+3+4+4+4 = 18, axis 1 = 0+0+0+0+0+1+2 = 3, over 7
        // voxels: (2.571..., 0.428...) -> [3, 0].
        assert_eq!(found[0].at, [3, 0, 0]);
        assert_eq!(found[0].weight, 7.0);
        // and the bounding box's centre, which a shortcut would give, is [2, 1]
        // — a different voxel on **both** axes, so this discriminates.
        assert_ne!(found[0].at, [2, 1, 0]);
    }

    /// The rounding, stated as arithmetic on the accumulators rather than as a
    /// property of a run: half goes up, and the ratio is taken once.
    #[test]
    fn a_centroid_on_a_half_rounds_up_and_is_taken_from_the_exact_ratio() {
        // Two voxels at 3 and 4: the mean is 3.5 exactly.
        let mut moments = Moments::EMPTY;
        moments.add([3, 0, 0]).unwrap();
        moments.add([4, 1, 1]).unwrap();
        assert_eq!(moments.count, 2);
        assert_eq!(moments.sums, [7, 1, 1]);
        assert_eq!(
            moments.centroid(),
            Some([4, 1, 1]),
            "3.5 must round up, and 0.5 with it"
        );

        // Just below a half stays down, and just above goes up: three voxels
        // whose sums are 4 and 5 over 3 (1.33 and 1.66).
        let mut low = Moments::EMPTY;
        for at in [[1, 0, 0], [1, 0, 0], [2, 0, 0]] {
            low.add(at).unwrap();
        }
        assert_eq!(low.centroid().unwrap()[0], 1);
        let mut high = Moments::EMPTY;
        for at in [[1, 0, 0], [2, 0, 0], [2, 0, 0]] {
            high.add(at).unwrap();
        }
        assert_eq!(high.centroid().unwrap()[0], 2);

        // An empty accumulator has no centroid at all, rather than one at the
        // origin.
        assert_eq!(Moments::EMPTY.centroid(), None);
        assert_eq!(Moments::EMPTY.point(), None);
    }

    /// **The property the whole op rests on**: the accumulators merge exactly, in
    /// any order, so a component cut into pieces gives the same centroid as the
    /// same component seen whole.
    #[test]
    fn moments_merge_in_any_order_and_the_seam_is_exact() {
        // A line of nine voxels, and every way of cutting it into three pieces.
        let voxels: Vec<[usize; 3]> = (0..9).map(|x| [x, 2, 5]).collect();
        let mut whole = Moments::EMPTY;
        for &at in &voxels {
            whole.add(at).unwrap();
        }

        for first in 1..8 {
            for second in first + 1..9 {
                let mut pieces = Vec::new();
                for slice in [&voxels[..first], &voxels[first..second], &voxels[second..]] {
                    let mut part = Moments::EMPTY;
                    for &at in slice {
                        part.add(at).unwrap();
                    }
                    pieces.push(part);
                }
                // Forwards, backwards, and middle-out: all three total to the
                // same thing, bit for bit, because integer addition associates.
                let orders = [[0, 1, 2], [2, 1, 0], [1, 0, 2]];
                for order in orders {
                    let mut merged = Moments::EMPTY;
                    for slot in order {
                        merged.merge(&pieces[slot]).unwrap();
                    }
                    assert_eq!(merged, whole, "cut at {first}/{second}, order {order:?}");
                    assert_eq!(merged.centroid(), whole.centroid());
                }
            }
        }
        assert_eq!(whole.centroid(), Some([4, 2, 5]));
    }

    /// An overflowing accumulator is refused rather than wrapped, because a
    /// wrapped sum is an arbitrary centroid.
    #[test]
    fn an_overflowing_accumulator_is_refused_rather_than_wrapped() {
        let mut moments = Moments {
            count: 1,
            sums: [u64::MAX, 0, 0],
        };
        let error = moments.add([1, 0, 0]).unwrap_err().to_string();
        assert!(error.contains("position sum"), "{error}");
        assert!(error.contains("65536"), "{error}");

        let mut counting = Moments {
            count: u64::MAX,
            sums: [0, 0, 0],
        };
        assert!(counting
            .merge(&Moments {
                count: 1,
                sums: [0; 3]
            })
            .is_err());
    }

    // ------------------------------------------------------- the labelling --

    #[test]
    fn an_empty_mask_has_no_points_and_a_full_one_has_exactly_one() {
        let empty = Array3::from_elem((6, 5, 4), false);
        assert!(points_of(&empty).is_empty());

        let full = Array3::from_elem((6, 5, 4), true);
        let found = points_of(&full);
        assert_eq!(found.len(), 1, "a full mask is one region");
        // sums over 6 x 5 x 4: means 2.5, 2, 1.5 -> [3, 2, 2] under round-half-up
        assert_eq!(found[0].at, [3, 2, 2]);
        assert_eq!(found[0].weight, 120.0);
    }

    /// Six-connectivity, asserted rather than assumed: two voxels touching only
    /// at a corner are two regions and get two points. A test that did not
    /// distinguish them would pass under either connectivity.
    #[test]
    fn regions_are_six_connected_so_a_corner_touch_is_two_of_them() {
        let corner = mask_of([5, 5, 5], &[[1, 1, 1], [2, 2, 2]]);
        let found = points_of(&corner);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].at, [1, 1, 1]);
        assert_eq!(found[1].at, [2, 2, 2]);

        // and a face touch is one, whose centroid is between the two
        let face = mask_of([5, 5, 5], &[[1, 1, 1], [2, 1, 1]]);
        let found = points_of(&face);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].at, [2, 1, 1], "1.5 rounds up");
        assert_eq!(found[0].weight, 2.0);
    }

    /// Regions on the faces and in the corners of the volume are regions like any
    /// other: nothing here treats the volume boundary as special.
    #[test]
    fn regions_on_the_faces_and_corners_are_counted_like_any_other() {
        let shape = [6usize, 6, 6];
        let corners: Vec<[usize; 3]> = (0..8)
            .map(|bits| {
                [
                    if bits & 1 == 0 { 0 } else { 5 },
                    if bits & 2 == 0 { 0 } else { 5 },
                    if bits & 4 == 0 { 0 } else { 5 },
                ]
            })
            .collect();
        let found = points_of(&mask_of(shape, &corners));
        assert_eq!(found.len(), 8);
        for &at in &corners {
            assert!(
                found.iter().any(|point| point.at == at),
                "the corner {at:?} lost its point"
            );
        }
    }

    // -------------------------------------------------------- the fragment --

    #[test]
    fn a_moments_fragment_survives_a_round_trip_and_a_wrong_one_is_refused() {
        let mask = mask_of([5, 4, 3], &[[0, 0, 0], [1, 0, 0], [3, 2, 1], [4, 3, 2]]);
        let mut labels = Array3::<u32>::zeros(mask.raw_dim());
        let count = label_regions_into(mask.view(), labels.view_mut()).unwrap();
        let moments = moments_of_labels(labels.view(), count, [8, 16, 32]).unwrap();
        let report = RegionMoments::of(labels.view(), count, moments).unwrap();

        let bytes = report.encode();
        assert_eq!(RegionMoments::decode(&bytes).unwrap(), report);
        // the offset really travelled, or the sums are block-local
        assert!(report.moments.iter().all(|part| part.sums[0] >= 8));

        // a stream written by something else — including, specifically, the two
        // other six-plane ops in this crate
        let mut foreign = bytes.clone();
        foreign[0] ^= 0xff;
        assert!(RegionMoments::decode(&foreign).is_err());
        assert!(super::super::fill::BlockFaces::decode(&bytes).is_err());
        assert!(super::super::regional::PlateauFaces::decode(&bytes).is_err());
        assert!(RegionMoments::decode(&super::super::fill::BlockFaces::empty().encode()).is_err());
        // a truncated one
        assert!(RegionMoments::decode(&bytes[..bytes.len() - 4]).is_err());
        // one with something appended
        let mut extra = bytes;
        extra.extend_from_slice(&[0, 0, 0, 0]);
        assert!(RegionMoments::decode(&extra).is_err());
        // and something that is not words at all
        assert!(RegionMoments::decode(&[1, 2, 3]).is_err());
        // a count that disagrees with the payload
        assert!(RegionMoments::of(labels.view(), count + 1, Vec::new()).is_err());
    }

    /// A 64-bit accumulator survives the fragment exactly, which is the reason it
    /// travels as two words rather than as anything narrower or lossier.
    #[test]
    fn a_fragment_carries_a_large_accumulator_through_unchanged() {
        let report = RegionMoments {
            labels: 1,
            moments: vec![Moments {
                count: u64::MAX,
                sums: [1, u64::MAX - 1, 1 << 53],
            }],
            faces: empty_planes(),
        };
        assert_eq!(RegionMoments::decode(&report.encode()).unwrap(), report);
    }

    // ----------------------------------------------------------- the merge --

    /// The merge refuses a lattice it was not given every block of. "Absent" and
    /// "present and empty" are different facts, and only one of them is a block
    /// that ran.
    #[test]
    fn the_merge_refuses_a_lattice_with_a_block_missing() {
        let mut reports = BTreeMap::new();
        reports.insert([0, 0, 0], RegionMoments::empty());
        assert!(merge_moments(&reports, [2, 1, 1]).is_err());
        reports.insert([1, 0, 0], RegionMoments::empty());
        assert!(merge_moments(&reports, [2, 1, 1]).is_ok());
    }

    /// Two blocks whose faces are different shapes came from two different
    /// lattices, and the merge says so rather than silently zipping the shorter.
    #[test]
    fn the_merge_refuses_faces_from_two_different_lattices() {
        use super::super::components::face_index;
        let mut small = RegionMoments::empty();
        small.faces[face_index(0, 1)] = ([2, 2], vec![0; 4]);
        let mut large = RegionMoments::empty();
        large.faces[face_index(0, 0)] = ([3, 3], vec![0; 9]);
        let mut reports = BTreeMap::new();
        reports.insert([0, 0, 0], small);
        reports.insert([1, 0, 0], large);
        let message = merge_moments(&reports, [2, 1, 1]).unwrap_err().to_string();
        assert!(message.contains("two different lattices"), "{message}");
    }

    /// The seam, driven directly: two labels that meet across it are one
    /// component whose accumulators are the sum of the two.
    #[test]
    fn a_seam_meeting_joins_the_two_accumulators_into_one_component() {
        use super::super::components::face_index;

        let build = |touching: bool| {
            let mut left = RegionMoments {
                labels: 1,
                moments: vec![Moments {
                    count: 2,
                    sums: [1, 0, 0],
                }],
                faces: empty_planes(),
            };
            left.faces[face_index(0, 1)] = ([1, 1], vec![1]);
            let mut right = RegionMoments {
                labels: 1,
                moments: vec![Moments {
                    count: 2,
                    sums: [9, 0, 0],
                }],
                faces: empty_planes(),
            };
            right.faces[face_index(0, 0)] = ([1, 1], vec![if touching { 1 } else { UNLABELLED }]);
            let mut reports = BTreeMap::new();
            reports.insert([0, 0, 0], left);
            reports.insert([1, 0, 0], right);
            merge_moments(&reports, [2, 1, 1]).unwrap()
        };

        let joined: Vec<Moments> = build(true).into_iter().filter(|m| !m.is_empty()).collect();
        assert_eq!(joined.len(), 1, "the two labels are one component");
        assert_eq!(joined[0].count, 4);
        assert_eq!(joined[0].sums, [10, 0, 0]);

        // and with nothing labelled on the far face they stay two
        let apart: Vec<Moments> = build(false).into_iter().filter(|m| !m.is_empty()).collect();
        assert_eq!(apart.len(), 2);
    }

    // ------------------------------------------------------- the ownership --

    /// The ownership rule is a function of the coordinate, so every voxel of the
    /// volume is owned by exactly one block of every lattice.
    #[test]
    fn every_voxel_is_owned_by_the_one_block_whose_core_holds_it() {
        for volume in [[20usize, 9, 5], [16, 16, 16]] {
            for edge in [[8usize, 4, 2], [5, 3, 1], [16, 16, 16]] {
                let grid = BlockGrid::new(volume, edge).unwrap();
                let cores = grid.cores();
                for i in 0..volume[0] {
                    for j in 0..volume[1] {
                        for k in 0..volume[2] {
                            let at = [i, j, k];
                            let owner = owner_of(&grid, at);
                            let holders: Vec<[usize; 3]> = cores
                                .iter()
                                .filter(|core| {
                                    (0..3).all(|axis| {
                                        at[axis] >= core.core.start[axis]
                                            && at[axis]
                                                < core.core.start[axis] + core.core.shape[axis]
                                    })
                                })
                                .map(|core| core.index)
                                .collect();
                            assert_eq!(
                                holders,
                                vec![owner],
                                "{volume:?}/{edge:?}: {at:?} is owned by {owner:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// And therefore every component is emitted by exactly one block, whatever
    /// the lattice — asserted over the blocks rather than over the answer.
    #[test]
    fn each_component_is_emitted_by_exactly_one_block() {
        let components = vec![
            Moments {
                count: 3,
                sums: [3, 6, 9],
            },
            Moments {
                count: 1,
                sums: [15, 0, 0],
            },
            Moments::EMPTY,
        ];
        let grid = BlockGrid::new([16, 16, 16], [8, 8, 8]).unwrap();
        let mut seen: Vec<Point> = Vec::new();
        for core in grid.cores() {
            let mine = points_owned_by(&components, &grid, core.index);
            for point in &mine {
                assert_eq!(
                    owner_of(&grid, point.at),
                    core.index,
                    "a block emitted a point it does not own"
                );
            }
            seen.extend(mine);
        }
        assert_eq!(seen.len(), 2, "the empty accumulator is not a point");
        seen.sort_by_key(|point| point.at);
        assert_eq!(seen[0].at, [1, 2, 3]);
        assert_eq!(seen[1].at, [15, 0, 0]);
    }
}
