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
// worth naming: **neither phase writes an image.** `fill` and `regional` both end
// by rewriting a label volume into a mask, so phase 0 has to write the labels
// down for phase 1 to read back. Here the answer is not a volume at all — it is
// a handful of points — and everything phase 1 needs is already in the fragment.
// The labels are a within-block temporary, so nothing allocates a `u32` image the
// size of the volume to hold a numbering nobody reads. That makes this the first
// op in `ops/` of `fragment.rs`'s second shape, `fragments -> fragments`, and
// the guard on it is the output stream's `Coverage::EveryBlock`: a phase that
// writes no image is not constrained by the tiling check at all, so the coverage
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
// One point per region, or one row
// --------------------------------
// A point is a position and one number. That is enough for "there is something
// here and it is this big" and it is not enough for anything that wants to
// *filter* on what was found — which is what a consumer of this op does next.
// So the op can emit a `crate::table` row per component instead, under
// [`measurement_schema`], and the choice is a parameter: [`Emission`].
//
// **The default is the point form and it is byte-identical to what this op has
// always written.** Same stream, same headerless four-word encoding, same order,
// same bytes. That was chosen over the alternative — always emit the row, and
// project back to points for `ops::voxelize` — for two reasons:
//
// * `ops::voxelize` reads a point blob and holds **no schema**. Feeding it a
//   table would mean either teaching it to find a weight column by name, turning
//   a fixed four-word layout into a run-time lookup that can fail, or writing two
//   streams from this phase — which doubles the output and makes the lifecycle
//   and coverage decisions for a stream half the callers do not want;
// * a row is about a hundred and four bytes per component against a point's
//   thirty-two. A caller who wants a density render should not pay three times
//   over in the run's persistent output for columns it will not read.
//
// There is a second, smaller cost with a sharper edge, and it is worth naming
// because it is the one that scales the wrong way. A table blob carries its
// schema, and this schema is ten named columns — about **two hundred and
// seventy bytes of header, per blob**. The output stream declares
// `Coverage::EveryBlock`, so every block writes one whether or not it owns a
// component, and a block that owns none writes a header and no rows where the
// point form writes nothing at all. On a large lattice over a sparse mask that
// is a fixed charge per block rather than per object. It buys the thing the
// self-description is for — a consumer handed the wrong stream is told which
// column disagrees — and it is a fair trade for a run that wanted the columns;
// it is not a fair trade for a run that did not, which is the third reason the
// default is the point form.
//
// What it costs, and it is a real cost: **the two forms are not distinguishable
// from the bytes alone in the direction that matters.** A table blob announces
// itself — it has a magic word and its schema in front — so a reader handed one
// where it expected points refuses it. A point blob does not: any four-word blob
// is a valid point set, so a reader expecting rows and handed points refuses it
// by the magic, but a reader expecting *points* and handed a truncated anything
// is on its own. That asymmetry was already there and this does not add to it;
// the mitigation is that the op knows which it wrote and the plan is where it is
// wired.
//
// [`points_of_measurements`] is the projection back, for a caller who asked for
// rows and still wants to render them.
//
// What the columns are, and why not more
// --------------------------------------
// Ten, all `U64`, all merged by an associative and commutative operation:
// `count` and `sum_0..2` by addition, `min_0..2` and `max_0..2` by minimum and
// maximum. [`measurement_schema`] is the table and the argument for each.
//
// **The constraint that decided the set is the merge.** A column here is a
// merged accumulator, and a quantity that cannot be folded exactly across a seam
// makes the answer a function of where the volume was cut — which is the one
// defect this crate exists to prevent. That rules out, concretely:
//
// * **anything accumulated in `f64`.** A running mean, a running variance, a
//   normalised moment. Floating-point addition does not associate, so the seam
//   merge would stop being exact and start being nearly exact, and "nearly" is
//   the decomposition-dependent answer the integer accumulators exist to rule
//   out. The derived quantity is the caller's to compute from exact integers,
//   once, after the merge — which is what the sums are for;
// * **second moments**, which are a closer call and are left out on a different
//   ground. `sum(x^2)` merges by addition and is exactly as associative as
//   `sum(x)`, so a radius of gyration or an anisotropy could be carried here
//   honestly. What stops it is the range: the first moment of a cubic volume of
//   edge `L` is about `L^4 / 2` and the second is about `L^5 / 3`, so a `u64`
//   carries the first to `L` of about 65536 and the second only to about 1800 —
//   a volume this crate is routinely pointed past, where every row would be a
//   refusal. Six more columns that stop working at a tenth of the volume the
//   other ten survive is not a trade worth making before a consumer asks for it,
//   and none does;
// * **a surface area, or any count of exposed faces.** It looks associative and
//   is not: phase 0 is halo-free, so a voxel on a block boundary cannot tell a
//   neighbour that is outside the volume from one that is outside the *block*,
//   and would count a face that is not there. Making it right needs a halo of
//   one and a rule about which side of a seam owns the face — that is a design,
//   not a field, and it is not this op's;
// * **anything needing a second input array** — an intensity total, a weighted
//   centroid, a background-corrected maximum. A `FragmentOp` reads one image and
//   this op reads the mask. See "The weighted centroid, which is not here"
//   below, which is where that hook goes.
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
// phase cannot hand its answer to a later phase without going through an image —
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
// implemented**. It used to be blocked on two things and is now blocked on one,
// and the one is the one that always mattered:
//
// * it needs a **second input array** beside the mask. **That capability is
//   built.** A [`FragmentOp`] declares the other images it reads with
//   [`FragmentOp::source_inputs`], is handed them through
//   [`FragmentOp::apply_with`], and states what its fold does across a seam with
//   [`SeamFold`]; the phase records the images, so the executor fetches them, the
//   DAG orders them and `exact_read_voxels` counts them. `ops::tabulate` uses all
//   three to reduce a second array over the regions of a label volume. Nothing is
//   waiting on a mechanism any more;
// * the tempting shortcut — take the weight from the mask image itself, since a
//   mask may arrive as any width and `is_set` only asks whether a voxel is
//   non-zero — would give up the exactness above. The weights would be arbitrary
//   reals, `f64` addition does not associate, and the seam merge would stop being
//   exact and start being *nearly* exact, which is precisely the
//   decomposition-dependent answer the integer accumulators exist to rule out. A
//   weighted variant has to say what it does about that, and the honest answers
//   are a fixed-point accumulator or a stated tolerance. Neither is a hook; both
//   are a design. `ops::tabulate` took the first answer and its header argues it;
//   this op has not made the choice, and until it does the accumulators here stay
//   integer counts and integer coordinate sums.
//
// **What `ops::tabulate` covers, and what it does not.** Over the regions of a
// label volume it emits `count`, `nonfinite`, `sum`, `min` and `max` of a second
// array, plus the per-axis coordinate sums `sum_0..2` — so the total of the
// quantity and the *geometric* centroid both come out of it, exactly and
// independently of the cut. A weighted centroid is neither of those and is not a
// ratio of them: it is `sum(v_i * x_i) / sum(v_i)`, a **cross moment** of value
// against position, and a sum of values and a sum of coordinates do not
// determine it — the same two totals are produced by arrangements with different
// weighted centres. So it is a different accumulator, not a different reading of
// the columns that exist.
//
// It is a small different accumulator — three more fixed-point words, folded by
// `+` like the ones beside them — and it lives where the label volume and the
// value array are already in one place, which is `ops::tabulate` and not here:
// its `moment_0..2_q{n}` columns are that accumulator and
// `RegionValues::weighted_centroid` is their quotient. This op labels a mask; a
// caller who wants a weighted centre labels with `ops::label` and tabulates.
// Nothing is added here for it, and this paragraph is the whole of the note.
//
// Connectivity: one choice, and it is the **foreground's**
// ---------------------------------------------------------
// Regions are connected components of the **set** voxels, and which neighbours
// count as connected is a [`Connectivity`] the caller states —
// [`Connectivity::Faces`] unless one is. There is exactly one here, because the
// set voxels are the only thing this op labels: two voxels touching only at a
// corner are two regions and get two points under `Faces`, and one region and one
// point under `FacesEdgesAndCorners`, and both answers are right about different
// questions.
//
// **This is the foreground's connectivity, and `ops::fill`'s is the
// background's.** They are separate parameters on separate ops rather than one
// shared choice, and that is deliberate: the *complementary pair* convention in
// the literature analyses a 6-connected foreground against a 26-connected
// background and vice versa, so a plan that fills holes at `Faces` and then
// detects regions at `FacesEdgesAndCorners` is the topologically consistent
// combination. Neither op can make that choice on the other's behalf — each sees
// one of the two sets — so each takes its own.
//
// **The fragment did not have to change.** A voxel with any neighbour outside its
// block lies on a *face* of that block, so the six planes already are the whole
// boundary shell, and the twelve edge lines and eight corner voxels a wider
// connectivity meets across are rows and single entries of them. `components`'s
// header argues it. What widened is the seam walk's inputs; the accumulators, the
// ownership rule and the encoding are untouched, because a component is still the
// transitive closure of an adjacency and only the pairs generating it grew.
//
// **Both phases carry the choice and [`detect_phases`] refuses a pair that
// disagrees**, for `ops::fill`'s reason: a plan that labelled at twenty-six and
// merged at six would join within a block what it kept apart across a seam, and
// the centroids would depend on where the volume was cut.
//
// What this costs
// ---------------
// `ops::fill`'s costs, minus the pixels. Phase 0 is halo-free and embarrassingly
// parallel. Phase 1 declares a whole-lattice fragment reach, so on `N` blocks it
// moves every block's fragment to each of `N` blocks and runs the same union-find
// once per block; what it does *not* do is read an image, because it does not
// need one — `reads_pixels` is `false`, so the executor performs no pixel IO for
// the phase at all and the read amplification `fill`'s header measures is not
// paid here.
//
// **That escape is smaller than this paragraph used to imply, and the correction
// matters more here than in `fill`.** What it says is true: the pixel half is not
// paid. What it left the reader with is that this op therefore gets the cheap
// version of the shape — and it does not. Of the three costs `fill`'s header now
// separates, this op escapes **one** and pays the other two in full:
//
// | | `ops::fill` | here |
// |---|---|---|
// | pixel re-reads, `N x` the label image | paid | **not paid** |
// | fragment transfers, `(1 + N) x` the whole fragment set | paid | paid |
// | the union-find, once per block | paid | paid |
//
// And the third of those is the one that was measured last and turned out
// largest. `fill`'s header has the figures and the sweep; the short form is that
// one merge is small at every lattice, there is one per block, and at a fine cut
// their sum exceeds the whole rest of the pipeline. This op runs the *larger*
// merge of the two — face labels plus a count and three position sums per label,
// folded over every component — so nothing here is cheaper than what was measured
// there.
//
// The old closing sentence said the fragment is six planes of labels plus eight
// words per label, "against a block of pixels — the same shape, for the same
// reason". That comparison is per block and it is the same false reassurance:
// against a *block* of pixels a fragment is small, and the phase moves the whole
// fragment set once per block rather than one fragment once. Past a fine enough
// cut the fragment set exceeds the whole volume, measured, and this op has no
// pixels to be small beside anyway.
//
// `docs/design/barriers.md` specifies the way out and prices it, and the way it
// lands here is the opposite of what "minus the pixels" suggests. That note
// separates two changes: a **barrier**, which relieves the halo, and a barrier
// that additionally lets the phase run its **reduction once**. A barrier alone
// recovers the pixel re-reads — so it recovers **nothing at all for this op**,
// which does not pay them. Everything this op pays is in the second change.
// Being the cheapest of the three ops today makes it the one with the least to
// gain from half the fix and the most to gain, proportionally, from all of it.

use std::collections::BTreeMap;
use std::sync::Arc;

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
use crate::region::Region;
use crate::sidecar::Lifecycle;
use crate::table::{Column, RowBuilder, Schema, Table, Value, POSITION_WORDS};

use super::components::{
    bytes_to_words, empty_planes, expect_end, label_members_into, label_members_into_with,
    planes_of, push_planes, read_header, take_planes, walk_seams_with, words_to_bytes,
    Connectivity, FacePlanes, LabelIndex, Union, UNLABELLED,
};
use super::fill::{agree_on_connectivity, as_mask};
use super::shapes_agree;

// ------------------------------------------------------------ the moments --

/// What a component contributes to its own row: how many voxels it has, where
/// they are, and how far they reach.
///
/// **Every field is an exact integer under an associative, commutative merge**
/// — `+` for the count and the sums, `min` and `max` for the corners — and that
/// is the entry condition rather than a description. A quantity that cannot be
/// folded that way makes a component's answer a function of where the volume was
/// cut, and there is no field here that is not. The module header is where the
/// argument is made and where the bounds are stated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Moments {
    /// Voxels in the component.
    pub count: u64,
    /// Per axis, the sum of that coordinate over the component's voxels, in
    /// **volume** coordinates. Block-local ones would make the merge a function
    /// of where the blocks were, which is the whole thing being avoided.
    pub sums: [u64; 3],
    /// Per axis, the smallest coordinate the component occupies, in volume
    /// coordinates. `u64::MAX` when there are no voxels, which is `min`'s
    /// identity and not a coordinate — read it through [`Moments::bounds`],
    /// which returns `None` instead.
    pub min: [u64; 3],
    /// Per axis, the **largest** coordinate the component occupies — inclusive,
    /// so the extent along an axis is `max - min + 1`.
    ///
    /// Inclusive rather than half-open, against the rest of the crate's
    /// convention, and the reason is that this is not a region anybody asked
    /// for: it is a voxel the component actually contains, and a column holding
    /// one-past-it would report a coordinate that is not in the object. The
    /// convention exists for query extents, and this is a measurement.
    pub max: [u64; 3],
}

/// The identity, which is what a default accumulator has to be.
///
/// Spelled out rather than derived: `min`'s identity is `u64::MAX` and a derived
/// `Default` would give `0`, which is a *valid coordinate* and would therefore
/// pin every component's low corner to the origin. That is the failure mode an
/// all-zero identity hides, so the derive is refused here.
impl Default for Moments {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Moments {
    /// The identity of the merge: no voxels, and therefore no centroid and no
    /// bounds. The corners are each monoid's identity — `u64::MAX` for the
    /// minimum, `0` for the maximum — so merging an empty accumulator in is a
    /// no-op on every field.
    pub const EMPTY: Self = Self {
        count: 0,
        sums: [0, 0, 0],
        min: [u64::MAX; 3],
        max: [0, 0, 0],
    };

    /// Add one voxel, at a **volume** coordinate.
    ///
    /// Checked rather than wrapping, for the sums. An overflowing sum is not a
    /// slightly wrong centroid, it is an arbitrary one, and a silently arbitrary
    /// answer is the failure mode this crate is arranged against. The header says
    /// how far the accumulator carries before this can fire: a cubic volume of
    /// about 65536 on a side, which is 2.8e14 voxels. The corners need no such
    /// check — `min` and `max` of coordinates are coordinates.
    pub fn add(&mut self, at: [usize; 3]) -> Result<()> {
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| overflowed("count"))?;
        for axis in 0..3 {
            let coordinate = at[axis] as u64;
            self.sums[axis] = self.sums[axis]
                .checked_add(coordinate)
                .ok_or_else(|| overflowed("position sum"))?;
            self.min[axis] = self.min[axis].min(coordinate);
            self.max[axis] = self.max[axis].max(coordinate);
        }
        Ok(())
    }

    /// Fold another partial accumulation into this one.
    ///
    /// **This is the whole of the seam.** Every field's operation is associative
    /// and commutative — addition for the count and the sums, `min` and `max`
    /// for the corners — so a component cut into any number of pieces gives the
    /// same totals as the same component seen whole, whatever order the pieces
    /// arrive in. That is why a row across a seam is *exact* rather than close,
    /// and it is the reason nothing else is allowed in this struct.
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        self.count = self
            .count
            .checked_add(other.count)
            .ok_or_else(|| overflowed("count"))?;
        for axis in 0..3 {
            self.sums[axis] = self.sums[axis]
                .checked_add(other.sums[axis])
                .ok_or_else(|| overflowed("position sum"))?;
            self.min[axis] = self.min[axis].min(other.min[axis]);
            self.max[axis] = self.max[axis].max(other.max[axis]);
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The component's bounding box, as an **inclusive** pair of corners, or
    /// `None` for a component with no voxels in it.
    ///
    /// `None` rather than the identity pair, for [`Moments::centroid`]'s reason:
    /// `(u64::MAX, 0)` is a well-formed identity and an absurd box, and handing
    /// it out would put the burden of noticing on every caller.
    pub fn bounds(&self) -> Option<([usize; 3], [usize; 3])> {
        if self.count == 0 {
            return None;
        }
        let mut low = [0usize; 3];
        let mut high = [0usize; 3];
        for axis in 0..3 {
            // Both came from a `usize` in `add`, so neither can truncate.
            low[axis] = self.min[axis] as usize;
            high[axis] = self.max[axis] as usize;
        }
        Some((low, high))
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
///
/// Six-connected because that is [`Connectivity`]'s default;
/// [`label_regions_into_with`] is the form that says which.
pub fn label_regions_into(mask: ArrayView3<'_, bool>, out: ArrayViewMut3<'_, u32>) -> Result<u32> {
    shapes_agree(mask.shape(), out.shape(), "label_regions_into")?;
    let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    label_members_into(shape, |at| mask[at], out)
}

/// [`label_regions_into`] under a stated [`Connectivity`], which is the
/// **foreground's** — see the module header for why there is only one here and
/// why it is not `ops::fill`'s.
///
/// Everything that function promises holds: the membership test, the scan-order
/// numbering, the iterative traversal. A wider choice leaves fewer, larger
/// regions and therefore fewer points.
pub fn label_regions_into_with(
    mask: ArrayView3<'_, bool>,
    connectivity: Connectivity,
    out: ArrayViewMut3<'_, u32>,
) -> Result<u32> {
    shapes_agree(mask.shape(), out.shape(), "label_regions_into_with")?;
    let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    label_members_into_with(shape, connectivity, |at| mask[at], out)
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
    Ok(centroid_points(&region_moments(mask)?))
}

/// [`detect_regions`] under a stated [`Connectivity`]: the whole-volume reference
/// a blocked run at that choice is measured against.
pub fn detect_regions_with(
    mask: ArrayView3<'_, bool>,
    connectivity: Connectivity,
) -> Result<Vec<Point>> {
    Ok(centroid_points(&region_moments_with(mask, connectivity)?))
}

/// The whole-volume answer as **rows**: [`detect_regions`]'s blob, in the richer
/// form.
///
/// Same kernels again, and for the same reason — the reference and the blocked
/// path must differ only in how the volume was cut, or a disagreement stops
/// being evidence of anything.
pub fn detect_region_rows(mask: ArrayView3<'_, bool>) -> Result<Vec<u8>> {
    encode_measurements(&region_moments(mask)?)
}

/// The accumulators of every component of a whole mask, unmerged because there
/// is nothing to merge: one array, one labelling, one pass.
pub fn region_moments(mask: ArrayView3<'_, bool>) -> Result<Vec<Moments>> {
    region_moments_with(mask, Connectivity::Faces)
}

/// [`region_moments`] under a stated [`Connectivity`].
pub fn region_moments_with(
    mask: ArrayView3<'_, bool>,
    connectivity: Connectivity,
) -> Result<Vec<Moments>> {
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    let count = label_regions_into_with(mask, connectivity, labels.view_mut())?;
    moments_of_labels(labels.view(), count, [0, 0, 0])
}

// -------------------------------------------------------------- the table --

/// What one component becomes when it is emitted.
///
/// **The default is [`Emission::Point`] and it is byte-identical to what this op
/// has always written** — same stream, same headerless four-word encoding, same
/// order. The richer form is opt-in. That choice is argued for in the module
/// header under "One point per region, or one row"; the short version is that
/// `ops::voxelize` reads point blobs and holds no schema, so making the rich form
/// the default would have meant either teaching `voxelize` to find a weight
/// column by name — a run-time lookup where it now has a fixed layout — or
/// writing two streams from one op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Emission {
    /// A [`Point`]: the rounded centroid, carrying the component's voxel count
    /// as its weight. Thirty-two bytes per component, and what `ops::voxelize`
    /// takes directly.
    #[default]
    Point,
    /// A `crate::table` row: every measurement the mask alone determines, under
    /// [`measurement_schema`]. About a hundred and four bytes per component
    /// against a point's thirty-two, which is why it is not the default.
    Measured,
}

impl Emission {
    /// The schema a blob of this form carries, or `None` for the point form —
    /// which is headerless and therefore has no schema *in the blob*, even
    /// though its words are a row of `Schema::points()`.
    pub fn schema(self) -> Option<Schema> {
        match self {
            Emission::Point => None,
            Emission::Measured => Some(measurement_schema()),
        }
    }

    /// The components of one block, in the canonical order, as the bytes that
    /// block writes.
    pub fn encode(self, components: &[Moments]) -> Result<Vec<u8>> {
        match self {
            Emission::Point => Ok(encode_points(&centroid_points(components))),
            Emission::Measured => encode_measurements(components),
        }
    }
}

/// Column names, in schema order. `pub` so that a consumer can name a column
/// without spelling the string, and so that a rename is one edit.
pub const COUNT: &str = "count";
/// Per-axis first-moment columns: `sum_0`, `sum_1`, `sum_2`.
pub const SUM: [&str; 3] = ["sum_0", "sum_1", "sum_2"];
/// Per-axis bounding-box low corner: `min_0`, `min_1`, `min_2`.
pub const MIN: [&str; 3] = ["min_0", "min_1", "min_2"];
/// Per-axis bounding-box high corner, **inclusive**: `max_0`, `max_1`, `max_2`.
pub const MAX: [&str; 3] = ["max_0", "max_1", "max_2"];

/// The schema [`Emission::Measured`] writes: ten `U64` columns, and nothing else.
///
/// **Every column is `U64`, and that is the entry condition rather than a
/// preference.** A column here is a merged accumulator, and `crate::table`'s
/// header says what an `F64` column can and cannot promise: the bits round trip,
/// and a partial fold merged across a seam is not the same number as the whole
/// fold. This op's whole claim is that it *is* the same number, so a float
/// column would be a place for the claim to stop being true.
///
/// | column | what it is | how it merges |
/// |---|---|---|
/// | `count` | voxels in the component | `+` |
/// | `sum_0..2` | per-axis sum of coordinates, volume-relative | `+` |
/// | `min_0..2` | per-axis smallest coordinate occupied | `min` |
/// | `max_0..2` | per-axis largest coordinate occupied, inclusive | `max` |
///
/// **Why the sums are here and not just the centroid.** The row's *position* is
/// the rounded centroid, which is a voxel; `sums / count` is the rational the
/// rounding threw away. A consumer that wants the sub-voxel centre — which is
/// what a reference implementation of this operation reports, in floating point
/// — recovers it exactly from these four numbers and cannot recover it from the
/// position. They also make the row **closed under merging**: two rows for the
/// same component, from two runs or two halves of a lattice, can be added, and
/// a row holding only a derived centroid cannot be.
///
/// **Why the corners.** They are the one further measurement a mask alone
/// determines that merges exactly, and they are what separates a compact
/// component from a long thin one of the same `count` — the extent along an axis
/// is `max - min + 1`. A consumer filtering on size alone cannot tell those
/// apart.
///
/// **What is deliberately absent** is anything needing a second input array — an
/// intensity, a weighted centroid, a background-corrected total. A `FragmentOp`
/// reads one image and this op reads the mask; the module header says what that
/// would take and why it is not a hook.
pub fn measurement_schema() -> Schema {
    let mut columns = vec![Column::u64(COUNT)];
    for group in [&SUM, &MIN, &MAX] {
        for name in group {
            columns.push(Column::u64(*name));
        }
    }
    // Ten distinct, non-empty names, so this cannot fail; expressed as a
    // `Result` internally and unwrapped here rather than making every caller
    // handle an impossibility.
    Schema::new(columns).expect("the measurement schema names ten distinct columns")
}

/// Words a measured row occupies: the three positions and the ten columns.
const MEASURED_WORDS: usize = POSITION_WORDS + 10;

/// One component's row, as the words it is stored and ordered by, or `None` for
/// a component with no voxels.
///
/// The words rather than a struct, because **this array is the canonical sort
/// key**: `crate::table` orders rows by their own words, positions first and
/// then the payload in schema order, so sorting these arrays is the canonical
/// order by construction rather than by a comparator that has to be kept in step
/// with the schema. Every column is `U64`, whose bits order the way its values
/// do, so there is no place for the two to disagree.
fn measured_row(moments: &Moments) -> Option<[u64; MEASURED_WORDS]> {
    let at = moments.centroid()?;
    let (low, high) = moments.bounds()?;
    let mut words = [0u64; MEASURED_WORDS];
    for axis in 0..3 {
        words[axis] = at[axis] as u64;
    }
    words[POSITION_WORDS] = moments.count;
    for axis in 0..3 {
        words[POSITION_WORDS + 1 + axis] = moments.sums[axis];
        words[POSITION_WORDS + 4 + axis] = low[axis] as u64;
        words[POSITION_WORDS + 7 + axis] = high[axis] as u64;
    }
    Some(words)
}

/// A set of components as a table blob, in the canonical order.
///
/// Components with no voxels are dropped rather than turned into a row at the
/// origin, for [`centroid_points`]'s reason: an empty accumulator is a
/// `(block, label)` slot that no voxel was found for, which is a thing that
/// exists in the flat numbering and not in the volume.
///
/// Sorted here so that a block's blob is a function of the component set and not
/// of the order the union-find happened to produce its roots in — the same
/// property `centroid_points` establishes for the point form, and it has to hold
/// for both or the two forms would be reproducible to different degrees.
pub fn encode_measurements(components: &[Moments]) -> Result<Vec<u8>> {
    let mut rows: Vec<[u64; MEASURED_WORDS]> = components.iter().filter_map(measured_row).collect();
    rows.sort_unstable();

    let mut builder = RowBuilder::new(Arc::new(measurement_schema()));
    for row in &rows {
        let at = [row[0] as usize, row[1] as usize, row[2] as usize];
        let values: Vec<Value> = row[POSITION_WORDS..]
            .iter()
            .copied()
            .map(Value::U64)
            .collect();
        // Round-trips the words through the typed push rather than writing them
        // straight into the buffer, so that a schema that grew a column without
        // `measured_row` growing a word is refused here instead of producing
        // rows that decode as something plausible.
        builder.push(at, &values)?;
    }
    Ok(builder.encode())
}

/// The points of a measured blob: the projection back to what
/// [`Emission::Point`] would have written.
///
/// Here so that a caller who asked for rows can still feed `ops::voxelize`
/// without re-running anything, and so that "the two forms agree" is a statement
/// with one implementation behind it —
/// `the_two_emissions_describe_the_same_components` drives this against the
/// point blob and compares bytes.
///
/// Refuses a blob that is not this schema, naming the column, because that is
/// what carrying the schema in the blob is for. `volume` is the volume the rows
/// were measured over — a row outside it is refused rather than projected, which
/// is `Table::write`'s check and not a second one here.
///
/// The result is in the canonical point order and is **byte-identical** through
/// `encode_points` to what the point form would have written for the same
/// components: both sequences are the same multiset sorted on the position and
/// then the count, and two entries that tie on both are the same four words.
pub fn points_of_measurements(volume: [usize; 3], bytes: &[u8]) -> Result<Vec<Point>> {
    let schema = measurement_schema();
    let count = schema
        .index_of(COUNT)
        .expect("the measurement schema has a count column");
    let mut table = Table::new(volume, schema)?;
    table.write([0, 0, 0], bytes)?;
    // Sealed so the scan is in the canonical order, which is what makes the
    // byte-identity above a property rather than a coincidence of the blob's
    // own order.
    table.seal()?;
    // A loop rather than a `collect` on the tail expression: the scan borrows
    // the table, and a borrow in a function's final expression outlives the
    // local it is taken from.
    let mut points = Vec::with_capacity(table.len());
    for row in table.scan(&Region::new(&[0, 0, 0], &volume))? {
        // Exact: a count past 2^53 would need a volume of nine petavoxels,
        // which is the same bound the point form's weight has always had.
        points.push(Point::weighted(row.at(), row.u64(count)? as f64));
    }
    Ok(points)
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

/// The components of `components` that `block` owns.
///
/// Exactly one block of the lattice returns any given component, because
/// [`owner_of`] is a function and the centroid is a function of the component —
/// so the ownership rule holds whatever the component is eventually emitted *as*,
/// which is why the filter is here and not in either encoder.
pub fn moments_owned_by(
    components: &[Moments],
    grid: &BlockGrid,
    block: [usize; 3],
) -> Vec<Moments> {
    components
        .iter()
        .filter(|moments| match moments.centroid() {
            None => false,
            Some(at) => owner_of(grid, at) == block,
        })
        .copied()
        .collect()
}

/// The points of `components` that `block` owns, in the canonical order.
pub fn points_owned_by(components: &[Moments], grid: &BlockGrid, block: [usize; 3]) -> Vec<Point> {
    centroid_points(&moments_owned_by(components, grid, block))
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
    /// and a version in front, and each accumulator as ten `u64`s — the count,
    /// the three sums and the two corners — low word first.
    ///
    /// The words rather than a float or a decimal, because the merge adds these
    /// and the whole claim of the op is that the addition is exact. A number that
    /// had been through a lossy encoding on the way to the merge would make the
    /// claim false in transit, where nothing would catch it.
    ///
    /// **The corners travel whether or not the run will emit them.** A fragment
    /// whose payload depended on which [`Emission`] phase 1 was configured for
    /// would be a second format, decoded by the same reader, distinguishable only
    /// by a field the reader does not have — and the saving is six words per
    /// *label* against six planes of labels, which is nothing. One format, always.
    pub fn encode(&self) -> Vec<u8> {
        let mut words: Vec<u32> = vec![MAGIC, VERSION, self.labels];
        for moments in &self.moments {
            push_u64(&mut words, moments.count);
            for axis in 0..3 {
                push_u64(&mut words, moments.sums[axis]);
            }
            for axis in 0..3 {
                push_u64(&mut words, moments.min[axis]);
            }
            for axis in 0..3 {
                push_u64(&mut words, moments.max[axis]);
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
                min: [
                    take_u64(record, 8),
                    take_u64(record, 10),
                    take_u64(record, 12),
                ],
                max: [
                    take_u64(record, 14),
                    take_u64(record, 16),
                    take_u64(record, 18),
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

/// Bumped from 1 when the accumulator grew its bounding box.
///
/// A version 1 blob is one word-count short per label, so a new reader would
/// have refused it anyway — somewhere inside the face planes, with a message
/// about the wrong thing. The bump makes the refusal say what actually happened,
/// which is the only reason a version number is worth carrying.
const VERSION: u32 = 2;

/// Ten `u64`s — the count, the three sums and the two corners — as twenty `u32`
/// words.
const WORDS_PER_LABEL: usize = 20;

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
    merge_moments_with(reports, counts, Connectivity::Faces)
}

/// [`merge_moments`] under a stated [`Connectivity`], which must be the one the
/// regions were labelled under.
///
/// The accumulators are untouched by the choice — they are per `(block, label)`
/// and the labels did not move. What the choice changes is which of them end up
/// on one root, and addition does not care how many pieces it is handed.
pub fn merge_moments_with(
    reports: &BTreeMap<[usize; 3], RegionMoments>,
    counts: [usize; 3],
    connectivity: Connectivity,
) -> Result<Vec<Moments>> {
    let index = LabelIndex::build(reports, counts, |report| report.labels)?;
    let parts = index.gather(reports, |report| &report.moments[..], Moments::EMPTY);
    let mut sets = Union::new(index.total());

    walk_seams_with(
        reports,
        counts,
        &index,
        connectivity,
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
/// **Writes no image.** The labels are consumed inside this call — the face
/// planes and the accumulators are everything phase 1 reads — so there is
/// nothing to hand on, and the module header says what that saves.
pub struct LabelRegionsOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    /// Which set voxels count as one region. [`Connectivity::Faces`] unless a
    /// caller said otherwise, so every existing caller gets the points it always
    /// got.
    connectivity: Connectivity,
}

impl LabelRegionsOp {
    pub fn new(name: &'static str, stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
            connectivity: Connectivity::Faces,
        }
    }

    /// The same op, labelling the foreground under a stated [`Connectivity`].
    ///
    /// A consuming builder rather than a fourth argument to [`Self::new`], for
    /// [`RegionPointsOp::emitting`]'s reason: every call site that does not say
    /// this word keeps its signature and its answer.
    ///
    /// The merge has to be told the same thing; [`detect_phases`] refuses a pair
    /// that disagrees.
    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Which set voxels this op counts as one region.
    pub fn connectivity(&self) -> Connectivity {
        self.connectivity
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
        let count = label_regions_into_with(mask.view(), self.connectivity, labels.view_mut())?;
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
    /// What one component becomes. [`Emission::Point`] unless a caller said
    /// otherwise, and that default is the whole of the compatibility story:
    /// every existing constructor leaves it alone, so every existing caller gets
    /// the bytes it always got.
    emission: Emission,
    /// Which set voxels count as one region **across a seam**, which has to be
    /// what the labelling used within a block. [`Connectivity::Faces`] unless a
    /// caller said otherwise.
    connectivity: Connectivity,
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
            emission: Emission::Point,
            connectivity: Connectivity::Faces,
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

    /// The same op, emitting rows instead of points.
    ///
    /// A consuming builder rather than a parameter on [`RegionPointsOp::new`],
    /// so that the two existing constructors keep their signatures and every
    /// call site that does not say this word is unchanged — which is the point
    /// of defaulting rather than projecting. What a caller takes on by saying it
    /// is that the output stream is now a `crate::table` blob and not a point
    /// blob: `ops::voxelize` cannot read it directly, and
    /// [`points_of_measurements`] is the projection back.
    pub fn emitting(mut self, emission: Emission) -> Self {
        self.emission = emission;
        self
    }

    /// What one component becomes in this op's output.
    pub fn emission(&self) -> Emission {
        self.emission
    }

    /// The same op, closing the components under a stated [`Connectivity`].
    ///
    /// **It must be the labelling's**, for `ops::fill::agree_on_connectivity`'s
    /// reason: the flood inside a block and the walk across a seam are two halves
    /// of one relation. [`detect_phases`] and [`append_connected`] check the
    /// pair; this builder alone cannot, because it sees one of the two ops.
    ///
    /// Independent of [`Self::emitting`] and composable with it in either order:
    /// which components exist is decided before what they are written as, which
    /// is the same ordering `apply` uses.
    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Which set voxels this op counts as one region across a seam.
    pub fn connectivity(&self) -> Connectivity {
        self.connectivity
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
            // this phase writes no image for the tiling check to bite on. A block
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
        let components =
            merge_moments_with(&reports, at.grid.blocks_per_axis(), self.connectivity)?;
        // Ownership first, emission second: which block writes a component is a
        // property of the component, so it must not be able to depend on the
        // form it is written in.
        let mine = moments_owned_by(&components, at.grid, at.index);
        Ok(BlockOutput::fragment(
            self.points_stream.clone(),
            self.emission.encode(&mine)?,
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
/// Neither phase declares a `dtype`: neither writes an image, so there is no image
/// whose width could be wrong, and `check_dtypes` skips both for that reason.
pub fn detect_phases(
    grid: BlockGrid,
    mask_dtype: Dtype,
    label: &LabelRegionsOp,
    points: &RegionPointsOp,
) -> Result<Decomposition> {
    agree_on_connectivity(label.connectivity(), points.connectivity())?;
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
    append_emitting(
        plan,
        moments_stream,
        moments_lifecycle,
        points_stream,
        points_lifecycle,
        Emission::Point,
    )
}

/// [`append_to`], with the output form said out loud.
///
/// The general one; `append_to` is this with [`Emission::Point`], which is the
/// form the two phases have always written. Kept as two functions rather than
/// one with a sixth argument so that no existing call site has to be edited to
/// say what it was already doing — a mechanical edit across every caller is
/// exactly the kind of change that gets one call site wrong.
pub fn append_emitting(
    plan: &mut PlanBuilder,
    moments_stream: impl Into<String>,
    moments_lifecycle: Lifecycle,
    points_stream: impl Into<String>,
    points_lifecycle: Lifecycle,
    emission: Emission,
) -> Result<Phase> {
    append_connected(
        plan,
        moments_stream,
        moments_lifecycle,
        points_stream,
        points_lifecycle,
        emission,
        Connectivity::Faces,
    )
}

/// [`append_emitting`], with the foreground's [`Connectivity`] said out loud.
///
/// The most general of the three, and the other two are this at their defaults —
/// which is why they are separate functions rather than one with two more
/// arguments: no existing call site has to be edited to say what it was already
/// doing.
///
/// **The choice goes to both phases from here**, which is what makes this the
/// safe way to ask for a wider one: a caller building the two ops by hand has to
/// remember to say it twice, and [`detect_phases`] is where that is caught.
pub fn append_connected(
    plan: &mut PlanBuilder,
    moments_stream: impl Into<String>,
    moments_lifecycle: Lifecycle,
    points_stream: impl Into<String>,
    points_lifecycle: Lifecycle,
    emission: Emission,
    connectivity: Connectivity,
) -> Result<Phase> {
    let moments_stream = moments_stream.into();
    let grid = plan.grid().clone();
    let moments = plan.fragments(
        LabelRegionsOp::new(
            "region labelling",
            moments_stream.clone(),
            moments_lifecycle,
        )
        .connecting(connectivity),
    )?;
    plan.fragments(
        RegionPointsOp::reading(
            "region points",
            moments_stream,
            moments,
            points_stream,
            points_lifecycle,
            &grid,
        )
        .emitting(emission)
        .connecting(connectivity),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::points::decode_points;
    use crate::table::ColumnType;

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

    /// The identity is each field's own identity, and that is not all zeros.
    ///
    /// The one that matters is `min`: a derived `Default` would put `0` there,
    /// which is a valid coordinate, so every component's low corner would be
    /// pinned to the origin and no test of a *merged* answer would notice —
    /// because merging in an identity with `min = 0` is what would break it.
    #[test]
    fn the_identity_is_a_no_op_on_every_field_including_the_corners() {
        assert_eq!(Moments::default(), Moments::EMPTY);
        assert_eq!(Moments::EMPTY.min, [u64::MAX; 3]);
        assert_eq!(Moments::EMPTY.max, [0; 3]);
        // and it has no box, rather than the absurd one its identities spell
        assert_eq!(Moments::EMPTY.bounds(), None);

        let mut real = Moments::EMPTY;
        for at in [[4, 9, 2], [6, 9, 3]] {
            real.add(at).unwrap();
        }
        assert_eq!(real.bounds(), Some(([4, 9, 2], [6, 9, 3])));

        // Merging the identity in, on either side, changes nothing at all.
        let mut left = real;
        left.merge(&Moments::EMPTY).unwrap();
        assert_eq!(left, real);
        let mut right = Moments::EMPTY;
        right.merge(&real).unwrap();
        assert_eq!(right, real);
    }

    /// A single voxel is its own box, so a component of one has extent one on
    /// every axis rather than zero. The corners are **inclusive** and this is
    /// where that is pinned.
    #[test]
    fn one_voxel_is_its_own_box_and_the_corners_are_inclusive() {
        let mut one = Moments::EMPTY;
        one.add([7, 3, 1]).unwrap();
        assert_eq!(one.bounds(), Some(([7, 3, 1], [7, 3, 1])));
        let row = measured_row(&one).expect("a row");
        for axis in 0..3 {
            let low = row[POSITION_WORDS + 4 + axis];
            let high = row[POSITION_WORDS + 7 + axis];
            assert_eq!(high - low + 1, 1, "axis {axis} spans one voxel");
        }
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
        assert_eq!(whole.bounds(), Some(([0, 2, 5], [8, 2, 5])));
    }

    /// **The exactness bar, at the row.** A component cut into four pieces gives
    /// the same row as the same component seen whole, *for every column, bit for
    /// bit*, in every one of the twenty-four orders the pieces can be merged in.
    ///
    /// Four rather than three because that is the number in the brief and, more
    /// usefully, because four is the smallest count that puts a piece in the
    /// interior on two axes at once — a component cut by a seam on x and a seam
    /// on y, which is the case where a low corner and a high corner come from
    /// *different* pieces on the same axis. A three-way cut of a line cannot
    /// reach that.
    #[test]
    fn a_component_split_four_ways_gives_the_same_row_in_every_merge_order() {
        // A 4 x 4 slab, cut by a seam at x = 2 and one at y = 2 into four
        // quadrants of four voxels each. Deliberately not symmetric in what each
        // quadrant holds: one voxel is dropped from one quadrant so that no
        // column is the same in two pieces by accident.
        let voxels: Vec<[usize; 3]> = (0..4)
            .flat_map(|x| (0..4).map(move |y| [x + 5, y + 9, 3]))
            .filter(|at| *at != [5, 9, 3])
            .collect();
        let mut whole = Moments::EMPTY;
        for &at in &voxels {
            whole.add(at).unwrap();
        }

        let mut quadrants = [Moments::EMPTY; 4];
        for &at in &voxels {
            let quadrant = usize::from(at[0] >= 7) * 2 + usize::from(at[1] >= 11);
            quadrants[quadrant].add(at).unwrap();
        }
        for (index, part) in quadrants.iter().enumerate() {
            assert!(!part.is_empty(), "quadrant {index} got no voxels");
        }

        let expected = measured_row(&whole).expect("a row");
        for order in permutations() {
            let mut merged = Moments::EMPTY;
            for slot in order {
                merged.merge(&quadrants[slot]).unwrap();
            }
            assert_eq!(merged, whole, "order {order:?}");
            assert_eq!(
                measured_row(&merged),
                Some(expected),
                "order {order:?} gave a different row"
            );
            // and through the blob, which is where a column could be written in
            // the wrong slot without either of the above noticing
            assert_eq!(
                encode_measurements(&[merged]).unwrap(),
                encode_measurements(&[whole]).unwrap(),
                "order {order:?} encoded differently"
            );
        }
    }

    /// Every ordering of four things, by Heap's algorithm written out — a
    /// dependency-free twenty-four, so the test above really does mean "every
    /// order" rather than "three of them".
    fn permutations() -> Vec<[usize; 4]> {
        let mut all = Vec::new();
        let mut current = [0usize, 1, 2, 3];
        let mut counters = [0usize; 4];
        all.push(current);
        let mut at = 0;
        while at < 4 {
            if counters[at] < at {
                let swap = if at % 2 == 0 { 0 } else { counters[at] };
                current.swap(swap, at);
                all.push(current);
                counters[at] += 1;
                at = 0;
            } else {
                counters[at] = 0;
                at += 1;
            }
        }
        assert_eq!(all.len(), 24);
        all
    }

    /// An overflowing accumulator is refused rather than wrapped, because a
    /// wrapped sum is an arbitrary centroid.
    #[test]
    fn an_overflowing_accumulator_is_refused_rather_than_wrapped() {
        let mut moments = Moments {
            count: 1,
            sums: [u64::MAX, 0, 0],
            ..Moments::EMPTY
        };
        let error = moments.add([1, 0, 0]).unwrap_err().to_string();
        assert!(error.contains("position sum"), "{error}");
        assert!(error.contains("65536"), "{error}");

        let mut counting = Moments {
            count: u64::MAX,
            sums: [0, 0, 0],
            ..Moments::EMPTY
        };
        assert!(counting
            .merge(&Moments {
                count: 1,
                sums: [0; 3],
                ..Moments::EMPTY
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

    /// The parameter reaches the answer, asked in the way that can tell: the
    /// same corner-touching pair as above, under each connectivity.
    ///
    /// Two points at six and at eighteen, one at twenty-six — and the one is not
    /// merely a count, it is a point at the pair's midpoint carrying both voxels'
    /// weight, which is what says the *accumulators* were merged rather than one
    /// of the two dropped.
    #[test]
    fn the_foreground_connectivity_reaches_the_points_and_the_bare_form_is_faces() {
        let corner = mask_of([5, 5, 5], &[[1, 1, 1], [2, 2, 2]]);

        let found = detect_regions_with(corner.view(), Connectivity::Faces).unwrap();
        assert_eq!(found.len(), 2);
        let edges = detect_regions_with(corner.view(), Connectivity::FacesAndEdges).unwrap();
        assert_eq!(edges.len(), 2, "a corner step is two edges' worth");

        let joined =
            detect_regions_with(corner.view(), Connectivity::FacesEdgesAndCorners).unwrap();
        assert_eq!(joined.len(), 1);
        assert_eq!(joined[0].at, [2, 2, 2], "1.5 rounds up on every axis");
        assert_eq!(joined[0].weight, 2.0, "both voxels are in it");

        // an edge-touching pair, which is what separates eighteen from six
        let edge = mask_of([5, 5, 5], &[[1, 1, 1], [2, 2, 1]]);
        assert_eq!(
            detect_regions_with(edge.view(), Connectivity::Faces)
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            detect_regions_with(edge.view(), Connectivity::FacesAndEdges)
                .unwrap()
                .len(),
            1
        );

        // and the bare form is the face-connected one
        assert_eq!(detect_regions(corner.view()).unwrap(), found);
    }

    /// The two phases are two halves of one relation, and a plan whose halves
    /// disagree is refused before it is scheduled.
    ///
    /// Also that the two builders compose: an op can be told what to emit and
    /// what counts as adjacent, in either order, and neither forgets.
    #[test]
    fn a_plan_whose_two_phases_disagree_about_connectivity_is_refused() {
        let grid = BlockGrid::new([8, 8, 8], [4, 4, 4]).unwrap();
        let label = LabelRegionsOp::new("label", "s", Lifecycle::DeleteOnExit)
            .connecting(Connectivity::FacesEdgesAndCorners);
        let points = RegionPointsOp::new("points", "s", 0, "p", Lifecycle::Persistent, &grid);
        let message = detect_phases(grid.clone(), Dtype::Bool, &label, &points)
            .unwrap_err()
            .to_string();
        assert!(message.contains("same connectivity"), "{message}");

        for connectivity in [
            Connectivity::Faces,
            Connectivity::FacesAndEdges,
            Connectivity::FacesEdgesAndCorners,
        ] {
            let label =
                LabelRegionsOp::new("label", "s", Lifecycle::DeleteOnExit).connecting(connectivity);
            let points = RegionPointsOp::new("points", "s", 0, "p", Lifecycle::Persistent, &grid)
                .connecting(connectivity)
                .emitting(Emission::Measured);
            assert_eq!(points.connectivity(), connectivity);
            assert_eq!(points.emission(), Emission::Measured);
            // and the other order forgets neither
            let swapped = RegionPointsOp::new("points", "s", 0, "p", Lifecycle::Persistent, &grid)
                .emitting(Emission::Measured)
                .connecting(connectivity);
            assert_eq!(swapped.connectivity(), connectivity);
            assert_eq!(swapped.emission(), Emission::Measured);
            assert!(detect_phases(grid.clone(), Dtype::Bool, &label, &points).is_ok());
        }
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
                min: [0, 1, u64::MAX - 3],
                max: [u64::MAX, 1 << 53, u64::MAX - 2],
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
                    min: [0, 0, 0],
                    max: [1, 0, 0],
                }],
                faces: empty_planes(),
            };
            left.faces[face_index(0, 1)] = ([1, 1], vec![1]);
            let mut right = RegionMoments {
                labels: 1,
                moments: vec![Moments {
                    count: 2,
                    sums: [9, 0, 0],
                    min: [4, 0, 0],
                    max: [5, 0, 0],
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
        // and the box is the union of the two, which is min and max rather than
        // either side's alone
        assert_eq!(joined[0].bounds(), Some(([0, 0, 0], [5, 0, 0])));

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
                min: [0, 1, 2],
                max: [2, 3, 4],
            },
            Moments {
                count: 1,
                sums: [15, 0, 0],
                min: [15, 0, 0],
                max: [15, 0, 0],
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

    /// And the ownership rule does not know which form it is emitting: the same
    /// components, under both, are written by the same blocks and counted once.
    #[test]
    fn the_emission_does_not_change_which_block_owns_a_component() {
        let mask = mask_of(
            [16, 16, 16],
            &[[1, 1, 1], [1, 1, 2], [9, 3, 4], [2, 12, 5], [14, 14, 14]],
        );
        let components = region_moments(mask.view()).unwrap();
        let grid = BlockGrid::new([16, 16, 16], [8, 8, 8]).unwrap();

        let mut points = 0usize;
        let mut rows = 0usize;
        for core in grid.cores() {
            let mine = moments_owned_by(&components, &grid, core.index);
            points += decode_points(&Emission::Point.encode(&mine).unwrap())
                .unwrap()
                .len();
            let measured = Emission::Measured.encode(&mine).unwrap();
            let projected = points_of_measurements([16, 16, 16], &measured).unwrap();
            rows += projected.len();
            for point in &projected {
                assert_eq!(
                    owner_of(&grid, point.at),
                    core.index,
                    "a block emitted a row it does not own"
                );
            }
        }
        // Four regions: the pair at [1, 1, 1..=2] is one.
        assert_eq!(points, 4);
        assert_eq!(rows, points, "the two forms lose or duplicate differently");
    }

    // -------------------------------------------------------- the emission --

    /// Every column is exact, and there are ten of them. Asserted against the
    /// names because a consumer filters by name, so a rename is a break.
    #[test]
    fn the_measurement_schema_is_ten_exact_columns() {
        let schema = measurement_schema();
        assert_eq!(schema.len(), 10);
        assert_eq!(schema.width(), MEASURED_WORDS);
        let expected = [
            "count", "sum_0", "sum_1", "sum_2", "min_0", "min_1", "min_2", "max_0", "max_1",
            "max_2",
        ];
        for (index, column) in schema.columns().iter().enumerate() {
            assert_eq!(column.name(), expected[index]);
            assert_eq!(
                column.kind(),
                ColumnType::U64,
                "column {:?} is a float; a float column does not merge exactly across a seam, \
                 which is the one thing this op promises",
                column.name()
            );
        }
        assert_eq!(schema.index_of(COUNT), Some(0));
    }

    /// One component, every column, read back out of the blob by name.
    ///
    /// The arithmetic is worked out in the comment rather than recomputed from
    /// the accumulator, so this is a check on the encoding and not on itself.
    #[test]
    fn a_measured_row_carries_every_column_through_the_blob() {
        // An L in one plane, arms of five and three: voxels (0..=4, 0) and
        // (4, 1..=2) at z = 6.
        let mut moments = Moments::EMPTY;
        for at in [
            [0, 0, 6],
            [1, 0, 6],
            [2, 0, 6],
            [3, 0, 6],
            [4, 0, 6],
            [4, 1, 6],
            [4, 2, 6],
        ] {
            moments.add(at).unwrap();
        }

        let volume = [8, 8, 8];
        let mut table = Table::new(volume, measurement_schema()).unwrap();
        table
            .write([0, 0, 0], &encode_measurements(&[moments]).unwrap())
            .unwrap();
        table.seal().unwrap();
        let rows = table.query(&Region::new(&[0, 0, 0], &volume)).unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];

        // sums: x = 0+1+2+3+4+4+4 = 18, y = 0+0+0+0+0+1+2 = 3, z = 7 * 6 = 42,
        // over seven voxels, so the centroid rounds to [3, 0, 6].
        assert_eq!(row.at(), [3, 0, 6]);
        assert_eq!(row.by_name(COUNT).unwrap(), Value::U64(7));
        for (axis, expected) in [18u64, 3, 42].into_iter().enumerate() {
            assert_eq!(
                row.by_name(SUM[axis]).unwrap(),
                Value::U64(expected),
                "sum on axis {axis}"
            );
        }
        // The box is [0..=4] x [0..=2] x [6..=6] — which the *position* cannot
        // tell you, and which separates this L from a compact blob of seven.
        for (axis, expected) in [0u64, 0, 6].into_iter().enumerate() {
            assert_eq!(row.by_name(MIN[axis]).unwrap(), Value::U64(expected));
        }
        for (axis, expected) in [4u64, 2, 6].into_iter().enumerate() {
            assert_eq!(row.by_name(MAX[axis]).unwrap(), Value::U64(expected));
        }
    }

    /// The two forms describe the same components: projecting a measured blob
    /// back gives, byte for byte, the point blob the same components would have
    /// produced.
    ///
    /// This is what lets a caller opt into rows without losing the pipe to
    /// `ops::voxelize`, and it is the check that the richer form did not quietly
    /// reorder or drop anything.
    #[test]
    fn the_two_emissions_describe_the_same_components() {
        let volume = [20usize, 12, 9];
        // Includes two components that share a rounded centroid on two axes, so
        // the order's tiebreak is actually exercised rather than assumed away.
        let mask = mask_of(
            volume,
            &[
                [3, 3, 3],
                [4, 3, 3],
                [10, 5, 1],
                [10, 5, 2],
                [10, 5, 3],
                [17, 9, 8],
                [1, 11, 0],
                [19, 0, 0],
            ],
        );
        let components = region_moments(mask.view()).unwrap();

        let points = Emission::Point.encode(&components).unwrap();
        let measured = Emission::Measured.encode(&components).unwrap();
        assert_ne!(points, measured, "the two forms are not the same bytes");

        let projected = points_of_measurements(volume, &measured).unwrap();
        assert_eq!(
            encode_points(&projected),
            points,
            "the projection back is not what the point form writes"
        );
        assert_eq!(projected, decode_points(&points).unwrap());
        // The richer form really is richer, and by about the factor claimed.
        assert!(measured.len() > points.len() * 2);
    }

    /// A block's blob is a function of the component set and not of the order the
    /// union-find produced its roots in — the same property `centroid_points`
    /// establishes for points, which has to hold for both forms or the two would
    /// be reproducible to different degrees.
    #[test]
    fn a_measured_blob_is_a_function_of_the_set_and_not_its_arrival_order() {
        let volume = [24usize, 24, 24];
        let mask = mask_of(
            volume,
            &[
                [1, 1, 1],
                [2, 1, 1],
                [8, 8, 8],
                [8, 8, 9],
                [8, 9, 8],
                [20, 3, 17],
                [5, 22, 2],
            ],
        );
        let components = region_moments(mask.view()).unwrap();
        assert!(components.len() >= 4);
        let expected = encode_measurements(&components).unwrap();

        // Reversed, rotated, and back to front in pairs. Three orders rather
        // than every one, because the sort is over the whole row and a sort that
        // was order-dependent would fail on the first of them.
        let mut reversed = components.clone();
        reversed.reverse();
        let mut rotated = components.clone();
        rotated.rotate_left(1);
        let mut swapped = components.clone();
        for pair in swapped.chunks_exact_mut(2) {
            pair.swap(0, 1);
        }
        for shuffled in [reversed, rotated, swapped] {
            assert_eq!(encode_measurements(&shuffled).unwrap(), expected);
        }
    }

    /// A measured blob announces itself, so a reader expecting something else
    /// refuses it naming the column rather than reading the words as its own.
    ///
    /// The asymmetry the module header names is asserted in the other direction
    /// too: a *point* blob is headerless, so a table asked to read one refuses it
    /// on the magic word.
    #[test]
    fn a_measured_blob_is_refused_by_a_reader_that_wants_another_schema() {
        let volume = [8usize, 8, 8];
        let mut moments = Moments::EMPTY;
        moments.add([2, 2, 2]).unwrap();
        let measured = encode_measurements(&[moments]).unwrap();

        let mut points = Table::new(volume, Schema::points()).unwrap();
        let message = points.write([0, 0, 0], &measured).unwrap_err().to_string();
        assert!(message.contains("column"), "{message}");

        let mut measured_table = Table::new(volume, measurement_schema()).unwrap();
        assert!(measured_table
            .write([0, 0, 0], &encode_points(&[Point::unit([2, 2, 2])]))
            .is_err());
        // and the projection refuses the same thing, since it is that check
        assert!(points_of_measurements(volume, &encode_points(&[Point::unit([2, 2, 2])])).is_err());
    }

    /// The whole-volume references agree with each other: `detect_region_rows`
    /// is `detect_regions` with more columns and not a second labelling.
    #[test]
    fn the_two_whole_volume_references_see_the_same_regions() {
        let volume = [9usize, 7, 5];
        let mask = mask_of(
            volume,
            &[[0, 0, 0], [8, 6, 4], [4, 3, 2], [4, 3, 3], [4, 4, 3]],
        );
        let points = detect_regions(mask.view()).unwrap();
        let rows = detect_region_rows(mask.view()).unwrap();
        assert_eq!(points_of_measurements(volume, &rows).unwrap(), points);
        assert_eq!(points.len(), 3);
    }
}
