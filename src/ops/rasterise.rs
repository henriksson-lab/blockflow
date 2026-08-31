// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation —
// rings in, a label volume out — not adapted from any implementation of it.
//
// **Stored outlines in, dense labels out.** A stream of *vertex* rows arrives as
// table fragments, one blob per block; this renders the paths and regions they
// describe into an integer volume, one class index per covered voxel.
//
// Why neither of the two ops that nearly do this
// ----------------------------------------------
// `ops::voxelize` is the structural template and it is the wrong op twice over.
// An open stroke of width `w` genuinely *is* a union of balls of radius `w / 2`
// along its path, so a caller whose vertices are dense enough can already get a
// stroke out of it — but the vertices are not dense, they are what a hand drew,
// and the segments between them are the whole shape. A ball at every vertex of a
// square is four dots, not a square.
//
// `ops::fill` covers the other half and stops one case short. Rasterise a closed
// outline into a mask, fill every enclosed background component, and a *simple*
// region comes out correct. A region with a **hole** does not: a hole is an
// enclosed background component, so the fill closes it, and the only way to keep
// it would be to tell the fill which components are holes — which is the
// question the rings already answer and the fill has no channel for. Rings with
// holes are first-class in the source data, so that case is what forces a real
// rasteriser rather than a composition of the two ops that exist.
//
// So: **rings in, a label volume out, even-odd so a hole stays a hole.**
//
// What arrives, and why it is one row per vertex
// ---------------------------------------------
// The stream is `crate::table` blobs under [`vertex_schema`] — **one row per
// vertex**, because a table's columns are scalars and a path is variable-length.
// A shape is reassembled from `shape`, `ring` and `vertex`.
//
// | column | what it says |
// |---|---|
// | `shape` | which shape this vertex belongs to. Also the collision key; see below |
// | `ring` | 0 is a region's exterior, every higher one is a hole |
// | `vertex` | the position of this vertex along its ring |
// | `class` | the value written into the volume. An index; the names live outside a table, whose columns are numbers |
// | `closed` | 1 if the ring closes back on itself and encloses area, 0 for an open path |
// | `dense` | read and **not used**; see *What this deliberately does not do* |
// | `z_extent` | how many *further* planes beyond the row's own this shape covers. Zero is one plane |
// | `x`, `y` | the exact position, in the volume's own units |
// | `half_width` | how far from the path a stroke covers. **Already halved** by the producer, so it is a distance and not a width, and 0 means the shape asserts nothing about area along its outline |
//
// **The position key is an address and not the value.** A table is keyed by whole
// voxels and that key is what routes a row to a block; the exact coordinate is
// fractional and travels in `x` and `y`, with the key rounded from it. Every
// piece of geometry below is read from the columns. The key is used for exactly
// two things — deciding which block owns a row, and which plane a shape sits in
// — and a consumer that took its geometry from the key would be off by up to
// half a voxel everywhere, silently.
//
// The axis order is the key's: **axis 0 is the plane axis**, axis 1 is `y` and
// axis 2 is `x`. That is the producer's convention and this op adopts it rather
// than inventing a second one; nothing here assumes anything else about what the
// axes mean.
//
// The rule, which is declared rather than chosen here
// ---------------------------------------------------
// A stored outline is an *intention* until somebody says what set of voxels it
// means, and two rasterisers that answer differently make two curators' work
// incomparable at the last step. So the producer declares the rule beside the
// geometry and this op implements exactly it:
//
// * **stroke** — the points within `half_width` of the path, with **round** caps
//   and joins;
// * **region** — **even-odd** over the shape's closed rings, ring 0 the exterior;
// * **sampling** — 4x4 subsamples per pixel, the pixel on at **7 of 16 or more**.
//
// Three notes on implementing it, each of which is a decision the rule does not
// make and which had to be made the same way twice or not at all:
//
// * **round caps and joins are not code.** The set within `half_width` of a
//   *polyline* is the union over its segments of the set within `half_width` of
//   each — a stadium per segment — and a union of stadia has round caps at the
//   ends and round joins at the corners by construction. So [`Shape::covers`] is
//   a minimum of point-to-segment distances and there is no cap or join case in
//   it. A one-vertex ring is the degenerate segment from the point to itself,
//   which is the disc, which is what a round cap on nothing is.
// * **the pixel's centre is at its integer coordinate.** Not at `index + 0.5`.
//   The producer rounds a fractional coordinate to get the routing key, and
//   rounding is the map to the *nearest* sample; a convention that put the centre
//   at `index + 0.5` would make the containing pixel a floor, and the key would
//   name the wrong block for half of every axis. The subsample offsets below —
//   `-0.375, -0.125, +0.125, +0.375` — are the 4x4 grid over `[-0.5, +0.5)`
//   around that centre.
// * **the sampling rule governs the stroke too**, and this is the one place the
//   declared rule is ambiguous. It is written as a peer of `stroke` and `region`
//   with nothing restricting it to regions, and the stroke clause names a
//   *centre*, which is the 1x1 case of the same thing. Supersampling both is the
//   reading that leaves one code path deciding every pixel, so a thin region and
//   a thin stroke of the same outline agree. It also costs little where it
//   matters, and the arithmetic is worth writing down rather than asserting:
//   along a straight run of a stroke, the centre rule selects `2*floor(h)+1`
//   rows and the sixteen-subsample rule selects `2*floor(h + 1/8)+1`, which are
//   the same number for every `h` whose fractional part is below `7/8` — so for
//   every whole stroke width, which is every width a pointer produces, the
//   choice is invisible on the inputs it was ambiguous about.
//   `a_stroke_covers_the_pixels_its_width_names` asserts the first expression
//   against the second.
//
// A shape is **filled and stroked**, in that order and both, which is what the
// two clauses say when read together rather than as alternatives: a closed ring
// with a `half_width` is a region whose outline is also fattened, and it
// degrades to a plain fill at `half_width == 0` and to a plain stroke when
// nothing is closed. An **open path of zero width is refused** rather than
// rendered, because it is a curve of no area: it can only produce an empty
// volume, and an empty answer is indistinguishable from a shape that never
// arrived.
//
// The hard part: a segment has no bounded reach, and what is done about it
// ------------------------------------------------------------------------
// `ops::voxelize` is plannable because a point's influence is its kernel radius,
// so the blocks that can affect a given block are computable before the run, and
// its header is explicit that a support the *data* decides makes that unknowable
// — leaving only a full-reach phase, which `fragment.rs` calls a reboot.
//
// **A segment is exactly that case, and the region rule is worse than the stroke
// rule.** A stroke's support is its path dilated by `half_width`, which is
// bounded by the path, and a path may span the volume. A region's is not even
// local in that sense: even-odd parity at a voxel counts the ring's crossings
// along a ray, so an edge arbitrarily far away flips the answer. The honest
// statement of the dependency is therefore not "within `r` of the vertex" but
// **"every vertex of every shape whose support meets this block"**, and how far
// that is, is a property of the shape.
//
// So it is made a **declared maximum extent**: `max_extent`, a per-axis count of
// voxels, given to [`RasteriseOutlinesOp::new`]. Both reaches follow from it, by
// exactly `ops::voxelize`'s arithmetic:
//
// * the **voxel** reach is `max_extent - 1` on that axis, clamped to the volume;
// * the **block** reach is `ceil((max_extent - 1) / edge)`, clamped to the
//   lattice — [`super::voxelize::block_reach`], called rather than rewritten, so
//   the two ops cannot drift about what a block step is worth.
//
// The derivation, since it is the whole justification for the parameter. Let a
// shape's support span at most `E` voxel indices on an axis, say `[a, a+E-1]`,
// and let it meet a core `[c, c+w-1]`; then `a <= c+w-1` and `a+E-1 >= c`. Every
// vertex *key* of the shape lies inside `[a, a+E-1]`, because the key is the
// coordinate rounded and the support is the coordinate's own interval widened by
// `half_width` and then by the half-voxel a rounding can move it. So a key sits
// in `[c - (E-1), c + w - 1 + (E-1)]`, which is a reach of `E-1` voxels — and
// `ops::voxelize`'s block-reach argument converts it exactly, on the same
// premise about where `BlockGrid` puts a short core.
//
// **A shape that exceeds it is refused by name, never clipped**, because a
// clipped ring is not a smaller ring: drop one edge of a closed outline and the
// even-odd parity beyond the gap inverts, so the failure is a plausible volume
// with its inside and outside exchanged over half its area. See
// [`Shape::check_extent`] for what a block can and cannot see of that, which is
// the honest limit of the check and is stated there rather than here.
//
// The parameter **degrades continuously to the case it is protecting against**,
// and that is what makes it a declaration rather than a restriction: a caller
// whose shapes really do span the volume passes the volume, the block reach
// becomes the whole lattice, and the phase becomes precisely the full-reach one
// this exists to avoid — priced as one, planned as one, named as one. Nothing is
// forbidden; what changes is that the cost is in the plan instead of in a
// surprise.
//
// What was considered instead, and why it is not this
// ---------------------------------------------------
// **A barrier with a whole-lattice fragment reach** — `FragmentOp::barrier` and
// [`FragmentInput::whole`], which is what `ops::fill`'s second phase does — would
// be simpler and would need no parameter at all: every block streams every
// fragment, so every ring is complete everywhere and there is no premise to
// state. It is rejected for two reasons and the first is the one that decides
// it:
//
// * a barrier is a **planning barrier**, so nothing fuses across this phase and
//   no cache survives it. That is a real cost paid on every run, to buy a
//   guarantee only the shapes that violate the declaration need;
// * each block would read the whole vertex set, which is `blocks x rows` of
//   sidecar traffic where the declared reach reads a bounded neighbourhood.
//
// The parameter is bounded, declared, in the plan's fingerprint, and checkable,
// which is what `ops::voxelize` gets from its kernel and this op has no kernel to
// get from. That is the trade, and a caller who prefers the other side of it can
// still take it by declaring the volume.
//
// A block sees whole shapes where it matters, and parts of shapes elsewhere
// --------------------------------------------------------------------------
// The declaration buys exactly one thing and it is worth stating as the property
// it is: **if a shape's support meets this block, every row of that shape is in
// this block's gathered neighbourhood.** The converse does not hold and must not
// be assumed — a block gathers rows from as far as the reach allows, so it
// routinely holds *some* of the vertices of shapes that are near it and do not
// touch it, and one of those is a ring with a gap in it.
//
// That is harmless, and it is harmless for a reason rather than by luck. A
// partial shape's bounding box is contained in the whole shape's, so a shape
// whose support misses this block has a *gathered* support that misses it too,
// and [`rasterise_into`] skips it before a single pixel is decided. Nothing that
// is drawn is ever drawn from a subset. What this does constrain is the checks:
// anything asked about a shape here must be a question a subset answers the same
// way. `class`, the per-shape agreement of the columns, a repeated
// `(ring, vertex)` and the extent are all such questions, because a subset that
// fails one of them is a shape that fails it. "Does this ring have three
// vertices" is not, which is why the degenerate-shape refusal in [`assemble`] is
// written against the `closed` flag and not against a count.
//
// Where the vertices must be
// --------------------------
// The rule `ops::voxelize` and `ops::label` both state, enforced by the same
// check: **a row in block B's fragment must be keyed inside B's core.** Cores
// tile the volume, so ownership is a total function of the coordinate, a vertex
// on a seam lands in exactly one fragment, and the reach derivation above has
// something to rest on. A fragment that breaks it is refused, naming the row and
// the core — not dropped and not drawn somewhere else.
//
// The collision rule: **the higher `shape` wins**
// -----------------------------------------------
// Two shapes may cover one voxel — overlapping regions, a stroke crossing a fill
// — and one class has to be written. The rule is that the larger `shape` value
// does: a voxel's class is the class of the **maximum** `shape` covering it,
// which is `ops::label`'s rule with the comparison turned round. `max` is
// associative, commutative and idempotent, so the answer is a function of the set
// of shapes covering the voxel and of nothing else — not of which fragments this
// block was given, not of the order they arrived in, not of the cut.
//
// It is `max` rather than `min` because `shape` is the producer's own ordering of
// its shapes, and the thing drawn last covering the thing beneath it is what
// somebody drawing them saw. The tie — one shape covering a voxel twice, through
// two of its own rings — is not a tie at all: it is one shape and one class.
//
// **It is spelled as an overwrite and that is not a shortcut.** [`shapes_of`]
// yields the shapes in ascending `shape`, keyed on a column that travels with the
// rows, so a later shape is by construction the larger one and assigning over the
// earlier is `max` evaluated in a convenient order. The order is derived from the
// data and not from the gather, which is the property that matters; what it saves
// is a second block-sized buffer holding, per voxel, which shape put the class
// there.
//
// **The class is written, not a running count**, so a voxel's value is a fact
// about the shape and not about how many shapes preceded it. `0` is reserved for
// "no label" exactly as `ops::label` reserves it, and a row carrying class `0` is
// **refused**: it would render to the same volume as a shape that never arrived.
// A producer numbering its classes from zero adds one, which is a decision about
// what its classes are called and is not this op's to make silently.
//
// Coverage
// --------
// No output stream is declared. `Coverage` is a decision about a *fragment*
// output and this phase's output is pixels, where the tiling guard over the valid
// regions is a real check rather than the vacuous one `fragment.rs` warns about.
// On the input side either coverage is tolerated: to a rasteriser, "this block
// wrote an empty list" and "this block wrote nothing" are the same fact.
//
// What this deliberately does not do
// ----------------------------------
// **Anti-aliasing.** The output is a label volume, and a label is an identity:
// there is no value between class 3 and class 4. The 4-of-16 threshold is where
// that gets resolved and it is resolved once, at the sampling rule, rather than
// per consumer.
//
// **Parts of one shape that restart their ring numbering.** A producer emitting
// several disjoint parts under one `shape`, each numbering its rings from 0,
// makes `(shape, ring, vertex)` non-unique — and once the rows have been through
// a table's canonical order, which sorts by position, there is nothing left to
// say which part a vertex came from. That is refused by name rather than
// reassembled from a guess, because the guess would silently join two outlines
// into one polyline across the gap between them. A producer with parts gives each
// part's rings distinct indices; the even-odd rule then covers the rest, since
// parity over the union of every part's rings is the same answer for disjoint
// parts as parity per part.
//
// **`dense`.** It is a statement about how to read the voxels a shape does *not*
// cover — whether an uncovered voxel is known background or merely unexamined —
// and that is a fact about supervision rather than about geometry. A label volume
// has one channel and it is already spent on the class; expressing this would
// take a second output, and inventing one here would be inventing a convention no
// consumer agreed to. It is required in the schema, so it survives to whoever
// does want it, and it is read by nothing below.
//
// **Sub-plane geometry.** A shape sits in the plane its key names and extends
// through `z_extent` further planes; every vertex of one shape must name the same
// plane. `x` and `y` are fractional and the plane axis is not, because the rule
// being implemented is two-dimensional per plane and an interpolation between
// planes is a different operation with a different answer.

use ndarray::{Array3, ArrayViewMut3, Axis, Slice};
use std::collections::BTreeMap;

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::fragment::{BlockOutput, BlockView, FragmentInput, FragmentOp};
use crate::geometry::BlockGrid;
use crate::region::Region;
use crate::sidecar::check_stream_name;
use crate::table::{Column, Schema, Table};
use crate::voxels::Voxels;

use super::label::label_ceiling;
use super::voxelize::block_reach;

// ----------------------------------------------------------------- schema --

/// The columns a vertex row carries, in the order they are encoded.
///
/// The order is part of the schema — `crate::table` refuses two schemas whose
/// columns are the same set permuted — so this is the whole compatibility
/// surface between this op and whatever wrote the stream, and a blob written
/// against a different one is refused naming the column that disagrees rather
/// than decoded as somebody else's numbers.
pub fn vertex_schema() -> Schema {
    Schema::new(vec![
        Column::u64("shape"),
        Column::u64("ring"),
        Column::u64("vertex"),
        Column::u64("class"),
        Column::u64("closed"),
        Column::u64("dense"),
        Column::u64("z_extent"),
        Column::f64("x"),
        Column::f64("y"),
        Column::f64("half_width"),
    ])
    .expect("the vertex schema names ten distinct columns")
}

const SHAPE: usize = 0;
const RING: usize = 1;
const VERTEX: usize = 2;
const CLASS: usize = 3;
const CLOSED: usize = 4;
const Z_EXTENT: usize = 6;
const X: usize = 7;
const Y: usize = 8;
const HALF_WIDTH: usize = 9;

/// The 4x4 subsample offsets from a pixel's centre, on one axis.
///
/// The centres of the four equal parts of `[-0.5, +0.5)`. Stated once because
/// the count and the threshold below are one rule: sixteen subsamples, on at
/// seven.
const SUBSAMPLES: [f64; 4] = [-0.375, -0.125, 0.125, 0.375];

/// How many of the sixteen must be inside for the pixel to be on.
const ON_AT: usize = 7;

// ------------------------------------------------------------------ rows --

/// One vertex, as this op reads it.
#[derive(Debug, Clone, Copy)]
struct VertexRow {
    at: [usize; 3],
    shape: u64,
    ring: u64,
    vertex: u64,
    class: u64,
    closed: bool,
    z_extent: u64,
    x: f64,
    y: f64,
    half_width: f64,
}

// ---------------------------------------------------------------- shapes --

/// One ring of a shape: a path, and whether it closes.
#[derive(Debug, Clone, PartialEq)]
pub struct Ring {
    /// The ring's own index. 0 is a region's exterior; every higher one is a
    /// hole, which is the one thing the vertices cannot say for themselves — a
    /// hole and an exterior are both closed rings.
    pub ring: u64,
    /// Whether the last vertex joins back to the first, and therefore whether
    /// this ring encloses area at all.
    pub closed: bool,
    /// `[y, x]` per vertex, in the volume's own units, in vertex order.
    pub points: Vec<[f64; 2]>,
}

/// One shape, reassembled from its vertex rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub shape: u64,
    /// The value written into the volume where this shape covers.
    pub class: u64,
    /// The first plane this shape covers, and the last, inclusive.
    pub planes: (usize, usize),
    /// How far from its paths this shape covers. Zero for an outline that
    /// asserts nothing about area.
    pub half_width: f64,
    pub rings: Vec<Ring>,
}

impl Shape {
    /// Is `[y, x]` inside this shape, before any sampling?
    ///
    /// The two clauses of the declared rule, unioned: even-odd over the closed
    /// rings, and within `half_width` of any path. The distance is a minimum over
    /// segments, which is what makes the caps and the joins round without a case
    /// for either; see the module header.
    pub fn covers(&self, y: f64, x: f64) -> bool {
        if self.half_width > 0.0 && self.within(y, x) {
            return true;
        }
        self.encloses(y, x)
    }

    /// Even-odd over every closed ring of this shape, taken together.
    ///
    /// Together and not per ring: the parity of the crossings along one ray is
    /// the whole rule, and it is what makes ring 0 an exterior and ring 1 a hole
    /// without either being labelled as one. Nested rings alternate for the same
    /// reason and by the same arithmetic.
    ///
    /// The ray goes towards `+x`. A vertex exactly on it is resolved by the
    /// half-open comparison `(y0 > y) != (y1 > y)`, which counts a vertex as
    /// above the ray and never as both endpoints of a crossing — the standard
    /// resolution, and the reason a shape's boundary vertex does not toggle
    /// twice.
    fn encloses(&self, y: f64, x: f64) -> bool {
        let mut inside = false;
        for ring in &self.rings {
            if !ring.closed || ring.points.len() < 3 {
                continue;
            }
            let points = &ring.points;
            let mut previous = points[points.len() - 1];
            for point in points {
                let (y0, x0) = (previous[0], previous[1]);
                let (y1, x1) = (point[0], point[1]);
                if (y0 > y) != (y1 > y) {
                    let crossing = x0 + (y - y0) / (y1 - y0) * (x1 - x0);
                    if x < crossing {
                        inside = !inside;
                    }
                }
                previous = *point;
            }
        }
        inside
    }

    /// Is `[y, x]` within `half_width` of any of this shape's paths?
    fn within(&self, y: f64, x: f64) -> bool {
        let limit = self.half_width;
        for ring in &self.rings {
            let points = &ring.points;
            if points.len() == 1 {
                // A one-vertex ring is a point, and the set within `limit` of a
                // point is a disc: the degenerate segment from it to itself.
                if distance_to_segment(y, x, points[0], points[0]) <= limit {
                    return true;
                }
                continue;
            }
            for pair in points.windows(2) {
                if distance_to_segment(y, x, pair[0], pair[1]) <= limit {
                    return true;
                }
            }
            if ring.closed && points.len() > 2 {
                let last = points[points.len() - 1];
                if distance_to_segment(y, x, last, points[0]) <= limit {
                    return true;
                }
            }
        }
        false
    }

    /// Is the pixel at `[plane, row, column]` on, under the sampling rule?
    pub fn covers_pixel(&self, row: usize, column: usize) -> bool {
        let mut on = 0usize;
        for dy in SUBSAMPLES {
            for dx in SUBSAMPLES {
                if self.covers(row as f64 + dy, column as f64 + dx) {
                    on += 1;
                    if on >= ON_AT {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// The voxel indices this shape can possibly touch, per axis, as
    /// `(lo, hi)` inclusive and **unclamped** — a shape may sit partly outside
    /// the volume, and where it does the part outside simply has no voxels.
    ///
    /// Widened by `half_width` for the stroke and then by half a voxel, which is
    /// the furthest a subsample sits from its pixel's centre rounded outwards. A
    /// pixel outside this cannot have a subsample inside the shape, so nothing
    /// is lost by not visiting it.
    pub fn bounds(&self) -> [(i64, i64); 3] {
        let (mut lo_y, mut hi_y) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut lo_x, mut hi_x) = (f64::INFINITY, f64::NEG_INFINITY);
        for ring in &self.rings {
            for point in &ring.points {
                lo_y = lo_y.min(point[0]);
                hi_y = hi_y.max(point[0]);
                lo_x = lo_x.min(point[1]);
                hi_x = hi_x.max(point[1]);
            }
        }
        let pad = self.half_width + 0.5;
        [
            (self.planes.0 as i64, self.planes.1 as i64),
            ((lo_y - pad).floor() as i64, (hi_y + pad).floor() as i64),
            ((lo_x - pad).floor() as i64, (hi_x + pad).floor() as i64),
        ]
    }

    /// How many voxel indices this shape's support spans, per axis.
    pub fn extent(&self) -> [usize; 3] {
        let bounds = self.bounds();
        let mut extent = [0usize; 3];
        for (axis, value) in extent.iter_mut().enumerate() {
            *value = (bounds[axis].1 - bounds[axis].0 + 1).max(0) as usize;
        }
        extent
    }

    /// Refuse a shape whose support is wider than the reach was derived for.
    ///
    /// **What this can and cannot see, stated rather than left to be
    /// discovered.** A block holds the vertices of its own gathered
    /// neighbourhood, so the extent measured here is over a *subset* of the
    /// shape's vertices — never more than the shape really spans. That makes the
    /// check sound in the direction that matters: an extent seen to exceed the
    /// declaration is a real violation, and every block that can see one refuses.
    /// It is not complete, and cannot be: a shape long enough to reach past the
    /// neighbourhood on both sides is exactly the shape whose far vertices this
    /// block never receives, and no local measurement recovers a number taken
    /// over rows that are not here.
    ///
    /// So this is the same shape of guarantee `ops::voxelize` makes about a point
    /// lying in its writer's core — a premise stated in the header, enforced
    /// wherever it is visible, and belonging to the producer where it is not. A
    /// caller who wants the complete check runs it over the whole set before the
    /// run, where the whole set exists; [`check_extent`] is that function.
    pub fn check_extent(&self, max_extent: [usize; 3]) -> Result<()> {
        let extent = self.extent();
        for axis in 0..3 {
            if extent[axis] > max_extent[axis] {
                return Err(Error::invalid(format!(
                    "rasterise: shape {} spans {} voxel(s) on axis {axis} and the declared \
                     maximum extent is {}. The reach this op declares is derived from that \
                     maximum, so a wider shape needs vertices from blocks the plan never asked \
                     for — and a ring rendered from some of its vertices is not a smaller ring, \
                     it is one whose even-odd parity inverts beyond the missing edge. Declare a \
                     larger `max_extent`, or split the shape.",
                    self.shape, extent[axis], max_extent[axis]
                )));
            }
        }
        Ok(())
    }
}

/// The distance from `[y, x]` to the segment `from`-`to`, both `[y, x]`.
///
/// A degenerate segment is the point itself, which is what a one-vertex ring
/// needs and what a repeated vertex — the closing vertex a closed ring usually
/// carries — costs nothing to allow.
fn distance_to_segment(y: f64, x: f64, from: [f64; 2], to: [f64; 2]) -> f64 {
    let (dy, dx) = (to[0] - from[0], to[1] - from[1]);
    let length = dy * dy + dx * dx;
    let along = if length <= 0.0 {
        0.0
    } else {
        (((y - from[0]) * dy + (x - from[1]) * dx) / length).clamp(0.0, 1.0)
    };
    let (ny, nx) = (from[0] + along * dy, from[1] + along * dx);
    ((y - ny) * (y - ny) + (x - nx) * (x - nx)).sqrt()
}

/// Every shape of `fragments`, reassembled and checked, in ascending `shape`
/// order.
///
/// Both halves of the check are here and nowhere else, so that a run which
/// allocates no buffer refuses exactly what a run which allocates one refuses:
/// every row is keyed inside the core of the block whose fragment carries it,
/// and every shape is one this destination and this declaration can hold.
///
/// The order of the result is a function of the row set alone — the shapes are
/// keyed by `shape` and the vertices by `(ring, vertex)`, both of which travel
/// with the rows — so it does not move when the fragments do. That the *drawing*
/// does not depend on it either is the collision rule's job, not this one's.
pub fn shapes_of(
    fragments: &[([usize; 3], Vec<u8>)],
    grid: &BlockGrid,
    max_extent: [usize; 3],
    ceiling: u64,
) -> Result<Vec<Shape>> {
    let volume = grid.volume();
    let mut rows = Vec::new();
    for (block, bytes) in fragments {
        let core = core_of(grid, *block).ok_or_else(|| {
            Error::invalid(format!(
                "rasterise: a fragment is keyed to block {block:?}, which is outside this \
                 phase's lattice of {:?} blocks",
                grid.blocks_per_axis()
            ))
        })?;
        for (index, row) in decode(volume, *block, bytes)?.into_iter().enumerate() {
            if !contains(&core, row.at) {
                return Err(Error::invalid(format!(
                    "rasterise: row {index} of block {block:?} is keyed at {:?}, which is \
                     outside that block's core {:?}..{:?}. A row must be keyed inside the core \
                     of the block whose fragment carries it: cores tile the volume, so that is \
                     what makes every vertex owned by exactly one block, and it is the premise \
                     the block reach is derived from. A producer whose rows may land outside \
                     their own core must re-key them to the block that owns them.",
                    row.at,
                    core.start,
                    core.end()
                )));
            }
            rows.push(row);
        }
    }

    let mut grouped: BTreeMap<u64, Vec<VertexRow>> = BTreeMap::new();
    for row in rows {
        grouped.entry(row.shape).or_default().push(row);
    }

    let mut shapes = Vec::with_capacity(grouped.len());
    for (id, mut rows) in grouped {
        rows.sort_by_key(|row| (row.ring, row.vertex));
        for pair in rows.windows(2) {
            if pair[0].ring == pair[1].ring && pair[0].vertex == pair[1].vertex {
                return Err(Error::invalid(format!(
                    "rasterise: shape {id} has two rows at ring {}, vertex {}. A shape made of \
                     several parts, each numbering its rings from zero, cannot be reassembled \
                     from these rows: they have been through a table's canonical order, which \
                     sorts by position, so nothing is left to say which part a vertex came \
                     from — and joining the two would draw a segment across the gap between \
                     them. Give each part's rings distinct indices; the even-odd rule then \
                     covers the rest.",
                    pair[0].ring, pair[0].vertex
                )));
            }
        }
        let shape = assemble(id, &rows, ceiling)?;
        shape.check_extent(max_extent)?;
        shapes.push(shape);
    }
    Ok(shapes)
}

/// One shape's rows, sorted by `(ring, vertex)`, as a [`Shape`].
fn assemble(id: u64, rows: &[VertexRow], ceiling: u64) -> Result<Shape> {
    let first = rows.first().ok_or_else(|| {
        Error::invalid(format!(
            "rasterise: shape {id} was grouped from no rows at all"
        ))
    })?;
    // Everything a shape carries once must be carried identically by every one
    // of its rows. Which row would otherwise win is an accident of the sort, so
    // the disagreement is refused rather than resolved.
    for (index, row) in rows.iter().enumerate() {
        let disagreement = if row.class != first.class {
            Some(("class", row.class.to_string(), first.class.to_string()))
        } else if row.z_extent != first.z_extent {
            Some((
                "z_extent",
                row.z_extent.to_string(),
                first.z_extent.to_string(),
            ))
        } else if row.half_width.to_bits() != first.half_width.to_bits() {
            Some((
                "half_width",
                row.half_width.to_string(),
                first.half_width.to_string(),
            ))
        } else if row.at[0] != first.at[0] {
            Some(("plane", row.at[0].to_string(), first.at[0].to_string()))
        } else {
            None
        };
        if let Some((column, mine, theirs)) = disagreement {
            return Err(Error::invalid(format!(
                "rasterise: row {index} of shape {id} says {column} is {mine} and the shape's \
                 first row says {theirs}. That column describes the whole shape, so two answers \
                 leave which one is drawn to the order the rows happened to sort in."
            )));
        }
    }
    if first.class == 0 {
        return Err(Error::invalid(format!(
            "rasterise: shape {id} carries class 0, and zero is reserved for a voxel no shape \
             covers. A shape rendering it would produce exactly the volume a shape that never \
             arrived produces. A producer numbering its classes from zero adds one to them; \
             which number names which class is not this op's to decide silently."
        )));
    }
    if first.class > ceiling {
        return Err(Error::invalid(format!(
            "rasterise: shape {id} carries class {}, and the largest this destination holds is \
             {ceiling}. A class written past the end of its type saturates onto another class, \
             which is two kinds of thing becoming one with nothing to see afterwards.",
            first.class
        )));
    }
    if first.half_width < 0.0 {
        return Err(Error::invalid(format!(
            "rasterise: shape {id} carries a half width of {}, and a distance from a path is \
             not negative. The column is already halved by the producer, so it is a distance \
             and not a width.",
            first.half_width
        )));
    }

    let mut rings: Vec<Ring> = Vec::new();
    for row in rows {
        match rings.last_mut() {
            Some(ring) if ring.ring == row.ring => {
                if ring.closed != row.closed {
                    return Err(Error::invalid(format!(
                        "rasterise: ring {} of shape {id} is marked both closed and open. A ring \
                         either encloses area or it does not, and the two rules that follow — \
                         even-odd and a stroke of the declared half width — are different \
                         answers.",
                        row.ring
                    )));
                }
                ring.points.push([row.y, row.x]);
            }
            _ => rings.push(Ring {
                ring: row.ring,
                closed: row.closed,
                points: vec![[row.y, row.x]],
            }),
        }
    }

    // The `closed` flag and not the vertex count, because a block holds only the
    // rows of its gathered neighbourhood: a closed ring can arrive here with two
    // of its vertices and it is still a closed ring, and refusing it would refuse
    // a shape for being far away. The count is consulted where it belongs, in
    // `Shape::encloses`, which is only ever asked about a window the whole shape
    // reaches — see `rasterise_into`.
    let closes = rings.iter().any(|ring| ring.closed);
    if !closes && first.half_width <= 0.0 {
        return Err(Error::invalid(format!(
            "rasterise: shape {id} encloses no area and carries no half width, so it covers a \
             curve and a curve has no voxels. Rendering it would produce the volume an absent \
             shape produces, which is the one outcome a rasteriser must not report as success. \
             Give it a width, or close it."
        )));
    }

    let plane = first.at[0];
    let last = plane.saturating_add(first.z_extent as usize);
    Ok(Shape {
        shape: id,
        class: first.class,
        planes: (plane, last),
        half_width: first.half_width,
        rings,
    })
}

/// One blob's rows, decoded through [`Table`].
///
/// Through `Table` and not through a second reader of the wire format, for
/// `ops::rows`' reason: the encoding has one decoder, so a blob this op accepts
/// is a blob every other consumer accepts. What it buys here beyond that is the
/// schema check — a stream written under different columns is refused naming the
/// column that disagrees — and the refusal of a row keyed outside the volume.
fn decode(volume: [usize; 3], block: [usize; 3], bytes: &[u8]) -> Result<Vec<VertexRow>> {
    let mut table = Table::new(volume, vertex_schema())?;
    table.write(block, bytes)?;
    table.seal()?;
    let mut rows = Vec::with_capacity(table.len());
    for row in table.scan(&Region::whole(&volume))? {
        let integer = |column: usize| row.u64(column);
        let float = |column: usize| row.f64(column);
        rows.push(VertexRow {
            at: row.at(),
            shape: integer(SHAPE)?,
            ring: integer(RING)?,
            vertex: integer(VERTEX)?,
            class: integer(CLASS)?,
            closed: integer(CLOSED)? != 0,
            z_extent: integer(Z_EXTENT)?,
            x: float(X)?,
            y: float(Y)?,
            half_width: float(HALF_WIDTH)?,
        });
    }
    Ok(rows)
}

/// The complete extent check, over a whole shape set.
///
/// Here because [`Shape::check_extent`] runs per block and a block holds a
/// subset — see there for exactly what that can and cannot see. This is the same
/// rule at the one place the whole set exists, which is before the run, in the
/// caller that has the rows. It is not called by the op, because the op never has
/// the whole set; calling it is how a producer converts a premise into a check.
pub fn check_extent(shapes: &[Shape], max_extent: [usize; 3]) -> Result<()> {
    for shape in shapes {
        shape.check_extent(max_extent)?;
    }
    Ok(())
}

// ------------------------------------------------------------------ paint --

/// Render `fragments` into `window` as class indices.
///
/// `out` covers `window` of the volume `grid` is cut from and **must arrive
/// zero-filled**, zero being the value of a voxel no shape covers; everything
/// outside `window` is dropped, which is how a block draws its own part of the
/// answer without knowing anything about the others. Rows are **not** required to
/// be keyed inside `window` — a shape outside it whose body reaches in is exactly
/// what the block reach exists for — but every row must be keyed inside the core
/// of the block whose fragment carries it.
///
/// Where two shapes meet, the larger `shape` wins; see the module header for why
/// that needs no order and therefore no sort.
///
/// Free function first, `FragmentOp` shell on top, for this module's usual
/// reason: a test can permute this function's argument and cannot permute what
/// the executor gathers.
pub fn rasterise_into(
    fragments: &[([usize; 3], Vec<u8>)],
    grid: &BlockGrid,
    max_extent: [usize; 3],
    ceiling: u64,
    window: &Region,
    mut out: ArrayViewMut3<'_, u64>,
) -> Result<()> {
    let volume = grid.volume();
    if window.ndim() != 3 {
        return Err(Error::invalid(format!(
            "rasterise: a window is 3-D, got rank {}",
            window.ndim()
        )));
    }
    window.check_within(&volume, "rasterise window")?;
    let shape = [out.shape()[0], out.shape()[1], out.shape()[2]];
    if shape.to_vec() != window.shape {
        return Err(Error::ShapeMismatch {
            expected: window.shape.clone(),
            got: shape.to_vec(),
        });
    }

    for drawn in shapes_of(fragments, grid, max_extent, ceiling)? {
        let bounds = drawn.bounds();
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        let mut empty = false;
        for axis in 0..3 {
            // Clamped to the window on both sides: the part of a shape that
            // falls outside this block belongs to another block, and the part
            // that falls outside the volume has no voxels at all.
            //
            // **This is also where a partial shape is dropped**, and it is why
            // one is never drawn. A shape's gathered bounding box is contained in
            // its whole one, so a shape whose support misses this window has a
            // gathered support that misses it too and leaves here; and a shape
            // whose support meets this window arrived complete, because that is
            // what the declared extent buys. See the module header.
            let start = window.start[axis] as i64;
            let end = start + shape[axis] as i64;
            let first = bounds[axis].0.max(start);
            let last = (bounds[axis].1 + 1).min(end);
            if first >= last {
                empty = true;
                break;
            }
            lo[axis] = first as usize;
            hi[axis] = last as usize;
        }
        if empty {
            continue;
        }
        for plane in lo[0]..hi[0] {
            for row in lo[1]..hi[1] {
                for column in lo[2]..hi[2] {
                    if !drawn.covers_pixel(row, column) {
                        continue;
                    }
                    let local = [
                        plane - window.start[0],
                        row - window.start[1],
                        column - window.start[2],
                    ];
                    // The collision rule: the larger `shape` wins. `shapes_of`
                    // yields them in ascending `shape`, so the assignment *is*
                    // the maximum — see the module header on why that order is a
                    // fact about the rows rather than about the gather.
                    out[local] = drawn.class;
                }
            }
        }
    }
    Ok(())
}

/// The core of `index` on `grid`, or `None` for an index off the lattice.
///
/// The same arithmetic `ops::voxelize` and `ops::label` do, recomputed rather
/// than searched for in `BlockGrid::cores` for their reason: that allocates every
/// core of the lattice to answer one question about one of them.
fn core_of(grid: &BlockGrid, index: [usize; 3]) -> Option<Region> {
    let volume = grid.volume();
    let edge = grid.block();
    let mut start = [0usize; 3];
    let mut shape = [0usize; 3];
    for axis in 0..3 {
        start[axis] = index[axis].checked_mul(edge[axis])?;
        if start[axis] >= volume[axis] {
            return None;
        }
        shape[axis] = (start[axis] + edge[axis]).min(volume[axis]) - start[axis];
    }
    Some(Region::new(&start, &shape))
}

fn contains(region: &Region, at: [usize; 3]) -> bool {
    (0..3).all(|axis| {
        at[axis] >= region.start[axis] && at[axis] < region.start[axis] + region.shape[axis]
    })
}

// ------------------------------------------------------------------ shell --

/// Render a stream of vertex rows into a label volume.
///
/// Reads one fragment stream, writes pixels, and declares no fragment stream of
/// its own. Both reaches are derived from the declared maximum extent and the
/// lattice; see the module header for why that parameter exists and what it is
/// standing in for.
pub struct RasteriseOutlinesOp {
    name: &'static str,
    stream: String,
    stream_phase: usize,
    max_extent: [usize; 3],
    grid: BlockGrid,
    block_reach: [usize; 3],
}

impl RasteriseOutlinesOp {
    /// `max_extent` is a per-axis count of **voxel indices** a shape's support
    /// may span — its vertices' bounding box widened by its half width, and on
    /// the plane axis the planes it covers, which is `z_extent + 1`.
    ///
    /// The lattice is an argument because the block reach is a statement in block
    /// indices and cannot be derived without one, exactly as `ops::voxelize`
    /// takes one. It is kept, not just measured: the phase this op runs on must
    /// be cut the same way, and [`Self::check_grid`] refuses a `BlockView` whose
    /// grid disagrees rather than reaching with a number derived for a different
    /// lattice.
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        stream_phase: usize,
        max_extent: [usize; 3],
        grid: &BlockGrid,
    ) -> Result<Self> {
        let stream = stream.into();
        check_stream_name(&stream)?;
        for axis in 0..3 {
            if max_extent[axis] == 0 {
                return Err(Error::invalid(format!(
                    "rasterise op {name:?} was given a maximum shape extent of {max_extent:?}, \
                     which is zero on axis {axis}. An extent is a count of voxel indices and the \
                     smallest shape there is spans one of them, so zero admits no shape at all \
                     and every stream would be refused."
                )));
            }
        }
        let mut reach = [0usize; 3];
        for (axis, value) in reach.iter_mut().enumerate() {
            *value = block_reach(grid, max_extent[axis] - 1, axis);
        }
        Ok(Self {
            name,
            stream,
            stream_phase,
            max_extent,
            grid: grid.clone(),
            block_reach: reach,
        })
    }

    /// The declared maximum support of one shape, in voxel indices per axis.
    pub fn max_extent(&self) -> [usize; 3] {
        self.max_extent
    }

    /// The declared reach in **blocks**, which is what the executor builds the
    /// gather neighbourhood from. Public so a test can check it against the
    /// geometry rather than against itself.
    pub fn block_reach(&self) -> [usize; 3] {
        self.block_reach
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// The lattice this op's reach was derived for is the lattice it may run on.
    ///
    /// A phase cut differently would hand this op a `BlockView` whose block
    /// indices mean something else, and the gather neighbourhood would be a
    /// number computed for a grid that is not the one being walked.
    pub fn check_grid(&self, grid: &BlockGrid) -> Result<()> {
        if grid.volume() != self.grid.volume() || grid.block() != self.grid.block() {
            return Err(Error::invalid(format!(
                "rasterise op {:?} derived its block reach {:?} from a lattice of {:?}/{:?} and \
                 is running on {:?}/{:?}. The reach is a count of block indices, so it means \
                 something else on another lattice; build the op with the grid the phase runs \
                 on.",
                self.name,
                self.block_reach,
                self.grid.volume(),
                self.grid.block(),
                grid.volume(),
                grid.block()
            )));
        }
        Ok(())
    }
}

impl FragmentOp for RasteriseOutlinesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// `max_extent - 1`, clamped to the volume.
    ///
    /// The derivation is in the module header and the subtraction is the whole of
    /// it: an extent is a count of indices and a reach is a distance between two
    /// of them, so a shape confined to one voxel index on an axis reaches nothing
    /// on it.
    ///
    /// **Symmetric, and on the plane axis that over-declares on purpose.** A
    /// shape covers the planes *after* the one it names and never those before,
    /// so the honest statement there is one-sided; this signature holds one
    /// integer per axis, and the wider side is the safe one to declare. What it
    /// costs is charged in whole blocks by [`Self::inputs`], where a one-sided
    /// extent usually rounds to the same neighbour count.
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        (self.max_extent[axis] - 1).min(volume_len)
    }

    /// No pixel IO on the way in: the input is the fragment stream, and a phase
    /// running this op reads not one voxel of the image below it.
    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    /// Gathered rather than streamed, for `ops::voxelize`'s reason: the reach is
    /// bounded by the declared extent, so it is a neighbourhood and not the
    /// lattice, and gathering is what makes the fetch count an observable
    /// property of the declaration. A caller who declares the whole volume has
    /// asked for the lattice and gets it — that is the case this op is honest
    /// about rather than the case it is arranged for.
    fn gathers(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.stream_phase).with_reach(self.block_reach)]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        self.check_grid(at.grid)?;

        let mut pixels = at.output_buffer(0.0)?;
        // `BlockBuf::dtype` answers for a simulated block too, so the element
        // type is checked before anything is allocated and a run over a volume
        // too large to hold still refuses a destination that cannot carry
        // classes.
        let dtype = pixels.dtype();
        let ceiling = label_ceiling(dtype).ok_or_else(|| {
            Error::invalid(format!(
                "rasterise op {:?} writes class indices — zero is uncovered and every other \
                 value names a class — and an image of {} holds no such value. A class is an \
                 identity and is compared for equality, which a two-valued type cannot express \
                 and a float type expresses only until a value has been through a rounding. \
                 Write an integer image.",
                self.name,
                dtype.numpy_name()
            ))
        })?;

        let mut fragments = Vec::new();
        for (key, bytes) in at.fragments(&self.stream) {
            fragments.push((key.block, bytes.clone()));
        }

        if pixels.as_array_mut().is_none() {
            // A simulated run allocates no buffer — those runs exist for volumes
            // that could not be held — but it still **checks**, because every
            // refusal in `shapes_of` is a fact about the row set rather than
            // about the buffer, and a simulated run that accepted a stream the
            // real one refuses would be worth less than one that did not run.
            shapes_of(&fragments, at.grid, self.max_extent, ceiling)?;
            return Ok(BlockOutput::nothing().with_pixels(pixels));
        }

        // Only the valid region is filled; the halo margin around it may have
        // contributors beyond the gathered neighbourhood, so it is left
        // uncovered rather than given a class nobody may trust.
        let valid = at.valid;
        let mut classes = Array3::<u64>::zeros((valid.shape[0], valid.shape[1], valid.shape[2]));
        rasterise_into(
            &fragments,
            at.grid,
            self.max_extent,
            ceiling,
            valid,
            classes.view_mut(),
        )?;

        let mut offset = [0usize; 3];
        for axis in 0..3 {
            offset[axis] = valid.start[axis] - at.read.start[axis];
        }
        if let Some(array) = pixels.as_array_mut() {
            store(&classes, offset, array)?;
        }
        Ok(BlockOutput::nothing().with_pixels(pixels))
    }
}

/// Write a `u64` class buffer into `at` of a buffer of the image's own type.
///
/// No rounding and no saturation, for `ops::label::store`'s reason: every class
/// has already been checked against [`label_ceiling`] for this element type, so
/// the cast is exact and a value that would have saturated was refused naming it.
/// A count clipped to a type's maximum is a bad number; a class clipped to it is
/// a different kind of thing.
fn store(classes: &Array3<u64>, at: [usize; 3], into: &mut Voxels) -> Result<()> {
    let shape = [classes.shape()[0], classes.shape()[1], classes.shape()[2]];
    let held = into.shape();
    for axis in 0..3 {
        if at[axis] + shape[axis] > held[axis] {
            return Err(Error::invalid(format!(
                "rasterise: writing {shape:?} at {at:?} of a block of {held:?} would run off \
                 axis {axis}"
            )));
        }
    }
    macro_rules! arm {
        ($type:ty) => {{
            let mut window = into.view_mut::<$type>()?;
            for axis in 0..3 {
                window
                    .slice_axis_inplace(Axis(axis), Slice::from(at[axis]..at[axis] + shape[axis]));
            }
            for (target, &value) in window.iter_mut().zip(classes.iter()) {
                *target = value as $type;
            }
        }};
    }
    match into.dtype() {
        Dtype::U8 => arm!(u8),
        Dtype::U16 => arm!(u16),
        Dtype::U32 => arm!(u32),
        Dtype::U64 => arm!(u64),
        Dtype::I8 => arm!(i8),
        Dtype::I16 => arm!(i16),
        Dtype::I32 => arm!(i32),
        Dtype::I64 => arm!(i64),
        // `label_ceiling` answers `None` for each of these and `apply` refuses
        // before reaching here, so this arm is unreachable through the op.
        other => {
            return Err(Error::invalid(format!(
                "rasterise: an image of {} holds no class indices",
                other.numpy_name()
            )))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::{fragment_phase, neighbourhood};
    use crate::table::{RowBuilder, Value};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// One shape, as a test writes it: the fields every one of its rows carries,
    /// and the rings.
    #[derive(Clone)]
    struct Drawn {
        shape: u64,
        class: u64,
        plane: usize,
        z_extent: u64,
        half_width: f64,
        /// `(ring index, closed, points as [y, x])`.
        rings: Vec<(u64, bool, Vec<[f64; 2]>)>,
    }

    impl Drawn {
        fn region(shape: u64, class: u64, rings: Vec<Vec<[f64; 2]>>) -> Self {
            Self {
                shape,
                class,
                plane: 0,
                z_extent: 0,
                half_width: 0.0,
                rings: rings
                    .into_iter()
                    .enumerate()
                    .map(|(at, points)| (at as u64, true, points))
                    .collect(),
            }
        }

        fn stroke(shape: u64, class: u64, half_width: f64, points: Vec<[f64; 2]>) -> Self {
            Self {
                shape,
                class,
                plane: 0,
                z_extent: 0,
                half_width,
                rings: vec![(0, false, points)],
            }
        }

        fn on_planes(mut self, plane: usize, z_extent: u64) -> Self {
            self.plane = plane;
            self.z_extent = z_extent;
            self
        }
    }

    /// The rows a set of shapes becomes, in the producer's own order.
    fn rows_of(drawn: &[Drawn]) -> Vec<([usize; 3], Vec<Value>)> {
        let mut rows = Vec::new();
        for shape in drawn {
            for (ring, closed, points) in &shape.rings {
                for (vertex, point) in points.iter().enumerate() {
                    rows.push((
                        [
                            shape.plane,
                            point[0].max(0.0).round() as usize,
                            point[1].max(0.0).round() as usize,
                        ],
                        vec![
                            Value::U64(shape.shape),
                            Value::U64(*ring),
                            Value::U64(vertex as u64),
                            Value::U64(shape.class),
                            Value::U64(u64::from(*closed)),
                            Value::U64(0),
                            Value::U64(shape.z_extent),
                            Value::F64(point[1]),
                            Value::F64(point[0]),
                            Value::F64(shape.half_width),
                        ],
                    ));
                }
            }
        }
        rows
    }

    /// Which block of `grid` owns `at` — the total function the "a row is keyed
    /// inside its writer's core" rule makes possible, and what a producer keys by.
    fn owner(grid: &BlockGrid, at: [usize; 3]) -> [usize; 3] {
        let edge = grid.block();
        [at[0] / edge[0], at[1] / edge[1], at[2] / edge[2]]
    }

    /// A shape set split into per-block blobs the way a producer must.
    fn split(grid: &BlockGrid, drawn: &[Drawn]) -> Vec<([usize; 3], Vec<u8>)> {
        let schema = Arc::new(vertex_schema());
        let mut builders: BTreeMap<[usize; 3], RowBuilder> = BTreeMap::new();
        for (at, values) in rows_of(drawn) {
            builders
                .entry(owner(grid, at))
                .or_insert_with(|| RowBuilder::new(schema.clone()))
                .push(at, &values)
                .unwrap();
        }
        builders
            .into_iter()
            .map(|(block, builder)| (block, builder.encode()))
            .collect()
    }

    /// Render a shape set over one block, which is the reference every
    /// decomposition is compared against.
    fn whole(volume: [usize; 3], drawn: &[Drawn]) -> Result<Array3<u64>> {
        let grid = BlockGrid::whole(volume)?;
        let mut out = Array3::<u64>::zeros((volume[0], volume[1], volume[2]));
        rasterise_into(
            &split(&grid, drawn),
            &grid,
            volume,
            u64::MAX,
            &Region::whole(&volume),
            out.view_mut(),
        )?;
        Ok(out)
    }

    /// Render a shape set the way the executor would: every block draws its own
    /// core from the fragments its declared reach gathers, and the cores are
    /// stitched.
    ///
    /// This is the harness the invariance test needs and the whole-volume one is
    /// not: `whole` hands every fragment to one window, which tests that the
    /// answer does not depend on the *order* they arrive in. This tests that it
    /// does not depend on the **cut** — which blocks exist, which fragments each
    /// one is given, and where the seams fall.
    fn stitched(volume: [usize; 3], block: [usize; 3], drawn: &[Drawn]) -> Result<Array3<u64>> {
        let grid = BlockGrid::new(volume, block)?;
        let fragments = split(&grid, drawn);
        let op = RasteriseOutlinesOp::new("rasterise", "outlines", 0, volume, &grid)?;
        let counts = grid.blocks_per_axis();
        let mut out = Array3::<u64>::zeros((volume[0], volume[1], volume[2]));
        for core in grid.cores() {
            let wanted = neighbourhood(core.index, op.block_reach(), counts);
            let gathered: Vec<_> = fragments
                .iter()
                .filter(|(key, _)| wanted.contains(key))
                .cloned()
                .collect();
            let shape = [core.core.shape[0], core.core.shape[1], core.core.shape[2]];
            let mut piece = Array3::<u64>::zeros((shape[0], shape[1], shape[2]));
            rasterise_into(
                &gathered,
                &grid,
                volume,
                u64::MAX,
                &core.core,
                piece.view_mut(),
            )?;
            for plane in 0..shape[0] {
                for row in 0..shape[1] {
                    for column in 0..shape[2] {
                        out[[
                            core.core.start[0] + plane,
                            core.core.start[1] + row,
                            core.core.start[2] + column,
                        ]] = piece[[plane, row, column]];
                    }
                }
            }
        }
        Ok(out)
    }

    /// A closed square ring from `lo` to `hi`, as `[y, x]`.
    ///
    /// Called with **half-integer** corners throughout, deliberately: a pixel's
    /// centre is at its integer coordinate, so a ring on the halves runs between
    /// pixel centres and every pixel it covers is covered by all sixteen of its
    /// subsamples. That takes the sampling rule out of every test that is not
    /// about the sampling rule, where it would otherwise show up as a boundary
    /// row that is half in. `the_sampling_rule_is_sixteen_subsamples_on_at_seven`
    /// puts a ring *through* the centres and is the one that pins that.
    fn square(lo: f64, hi: f64) -> Vec<[f64; 2]> {
        vec![[lo, lo], [lo, hi], [hi, hi], [hi, lo], [lo, lo]]
    }

    // ------------------------------------------------------------ strokes --

    /// The width a stroke declares is the width it covers, counted against the
    /// definition rather than against a run of this code: a straight stroke of
    /// half width `h` covers the rows whose centres are within `h` of it.
    #[test]
    fn a_stroke_covers_the_pixels_its_width_names() {
        // The count is the definition and not a reading of this code: the pixels
        // whose centre is within `h` of a line through a row of centres are the
        // `2 * floor(h) + 1` rows around it.
        for (half_width, expected) in [(0.5f64, 1usize), (1.0, 3), (2.5, 5), (3.0, 7)] {
            assert_eq!(expected, 2 * half_width.floor() as usize + 1);
            let drawn = Drawn::stroke(1, 9, half_width, vec![[8.0, 6.0], [8.0, 14.0]]);
            let rendered = whole([1, 21, 21], &[drawn]).unwrap();
            let column: Vec<u64> = (0..21).map(|row| rendered[[0, row, 10]]).collect();
            let covered = column.iter().filter(|&&value| value == 9).count();
            assert_eq!(
                covered, expected,
                "half width {half_width} covered {column:?}"
            );
            // and it is centred on the path, not offset by one
            assert_eq!(rendered[[0, 8, 10]], 9, "half width {half_width}");
        }
    }

    /// A stroke's end is a disc and not a square: the pixel diagonally beyond the
    /// last vertex is outside a round cap and inside a square one.
    #[test]
    fn a_strokes_cap_is_round() {
        let drawn = Drawn::stroke(1, 3, 2.0, vec![[8.0, 4.0], [8.0, 8.0]]);
        let rendered = whole([1, 17, 17], &[drawn]).unwrap();
        // straight past the end, within the half width: covered
        assert_eq!(rendered[[0, 8, 10]], 3);
        // diagonally past it, at a distance of sqrt(8) > 2: not
        assert_eq!(rendered[[0, 6, 10]], 0);
    }

    /// A single vertex with a width is a disc, which is what a round cap on
    /// nothing is, and it is the case that has no segment to take a distance to.
    #[test]
    fn a_single_vertex_with_a_width_is_a_disc() {
        let drawn = Drawn::stroke(1, 4, 2.0, vec![[8.0, 8.0]]);
        let rendered = whole([1, 17, 17], &[drawn]).unwrap();
        assert_eq!(rendered[[0, 8, 8]], 4);
        assert_eq!(rendered[[0, 8, 10]], 4);
        assert_eq!(rendered[[0, 6, 6]], 0, "a corner of the square is outside");
        let covered = rendered.iter().filter(|&&value| value == 4).count();
        assert_eq!(covered, 13, "the discrete disc of radius 2");
    }

    // ------------------------------------------------------------ regions --

    #[test]
    fn a_closed_region_is_filled_and_not_merely_outlined() {
        // between the centres of pixels 3 and 13, so it covers 4..=12
        let drawn = Drawn::region(1, 5, vec![square(3.5, 12.5)]);
        let rendered = whole([1, 17, 17], &[drawn]).unwrap();
        assert_eq!(rendered[[0, 8, 8]], 5, "the middle is filled");
        assert_eq!(rendered[[0, 4, 4]], 5, "and so is the corner");
        assert_eq!(rendered[[0, 3, 8]], 0, "and outside is not");
        assert_eq!(rendered[[0, 8, 13]], 0);
        let covered = rendered.iter().filter(|&&value| value == 5).count();
        assert_eq!(covered, 9 * 9, "the 9x9 block the ring encloses");
    }

    /// **The test this op exists for.** `ops::fill` fills every enclosed
    /// background component and a hole is one, so it closes it; the even-odd rule
    /// does not, and this is the difference.
    #[test]
    fn a_polygon_with_a_hole_keeps_the_hole() {
        // pixels 2..=14 enclosed, of which 6..=10 are the hole
        let drawn = Drawn::region(1, 7, vec![square(1.5, 14.5), square(5.5, 10.5)]);
        let rendered = whole([1, 17, 17], &[drawn]).unwrap();
        assert_eq!(rendered[[0, 8, 8]], 0, "the middle of the hole is empty");
        assert_eq!(rendered[[0, 6, 6]], 0, "and so is its corner");
        assert_eq!(rendered[[0, 4, 8]], 7, "between the rings is filled");
        assert_eq!(rendered[[0, 2, 2]], 7, "and so is the outer corner");
        assert_eq!(rendered[[0, 1, 8]], 0, "outside the exterior is not");
        let covered = rendered.iter().filter(|&&value| value == 7).count();
        assert_eq!(covered, 13 * 13 - 5 * 5, "the ring between the two squares");
    }

    /// Even-odd is a parity and not a two-level rule: a third ring inside the
    /// hole is solid again. Nothing in the data says which ring is which — the
    /// arithmetic says it.
    #[test]
    fn nested_rings_alternate_by_the_even_odd_rule() {
        let drawn = Drawn::region(
            1,
            2,
            vec![square(0.5, 15.5), square(3.5, 12.5), square(5.5, 10.5)],
        );
        let rendered = whole([1, 17, 17], &[drawn]).unwrap();
        assert_eq!(rendered[[0, 8, 2]], 2, "inside the first ring");
        assert_eq!(rendered[[0, 8, 5]], 0, "inside the second: a hole");
        assert_eq!(rendered[[0, 8, 8]], 2, "inside the third: solid again");
        let covered = rendered.iter().filter(|&&value| value == 2).count();
        assert_eq!(covered, 15 * 15 - 9 * 9 + 5 * 5);
    }

    /// A filled region carrying a width is both, which is what the two clauses of
    /// the rule say when read together: the fill, and the outline fattened.
    #[test]
    fn a_closed_region_with_a_width_is_filled_and_stroked() {
        let mut drawn = Drawn::region(1, 6, vec![square(5.5, 10.5)]);
        drawn.half_width = 2.0;
        let rendered = whole([1, 17, 17], &[drawn]).unwrap();
        assert_eq!(rendered[[0, 8, 8]], 6, "the fill");
        assert_eq!(rendered[[0, 8, 12]], 6, "within the width of the edge");
        assert_eq!(rendered[[0, 8, 13]], 0, "beyond it: neither");
    }

    // ------------------------------------------------------------- planes --

    #[test]
    fn a_shape_covers_the_planes_its_z_extent_names_and_no_others() {
        let drawn = Drawn::region(1, 3, vec![square(3.5, 8.5)]).on_planes(2, 2);
        let rendered = whole([6, 13, 13], &[drawn]).unwrap();
        for plane in 0..6 {
            let expected = if (2..=4).contains(&plane) { 3 } else { 0 };
            assert_eq!(rendered[[plane, 6, 6]], expected, "plane {plane}");
        }
    }

    // --------------------------------------------------------- collisions --

    /// Two shapes over one voxel: the larger `shape` wins, and it wins whichever
    /// order the fragments arrive in, because `max` has no order to depend on.
    #[test]
    fn overlapping_shapes_resolve_to_the_larger_shape_whatever_the_order() {
        let lower = Drawn::region(3, 11, vec![square(1.5, 10.5)]);
        let upper = Drawn::region(9, 12, vec![square(5.5, 14.5)]);
        let forwards = whole([1, 17, 17], &[lower.clone(), upper.clone()]).unwrap();
        let backwards = whole([1, 17, 17], &[upper, lower]).unwrap();
        assert_eq!(forwards[[0, 8, 8]], 12, "the overlap takes shape 9's class");
        assert_eq!(forwards[[0, 3, 3]], 11, "and shape 3 keeps its own");
        assert_eq!(forwards[[0, 13, 13]], 12);
        assert_eq!(forwards, backwards);
    }

    // ------------------------------------------------- decomposition --

    /// **The property that matters most here**: the answer is a function of the
    /// shapes and not of how the volume was cut. Every block draws its own core
    /// from the fragments its declared reach gathers, and the stitched result
    /// must equal the one-block reference exactly, at every cut — including cuts
    /// whose seams fall inside a hole, inside a stroke and along a ring's edge.
    #[test]
    fn every_cut_of_the_same_shapes_renders_the_same_volume() {
        let volume = [3usize, 24, 24];
        let drawn = vec![
            Drawn::region(1, 4, vec![square(1.5, 20.5), square(7.5, 14.5)]),
            Drawn::stroke(5, 7, 1.5, vec![[3.0, 3.0], [20.0, 11.0], [11.0, 21.0]]).on_planes(1, 1),
            Drawn::region(2, 9, vec![square(15.5, 22.5)]).on_planes(2, 0),
        ];
        let reference = whole(volume, &drawn).unwrap();
        assert!(
            reference.iter().any(|&value| value != 0),
            "the fixture renders nothing, so this test would pass on a broken op"
        );
        for block in [
            [3usize, 24, 24],
            [1, 24, 24],
            [3, 12, 12],
            [1, 8, 8],
            [2, 7, 5],
            [1, 3, 3],
        ] {
            let cut = stitched(volume, block, &drawn).unwrap();
            assert_eq!(cut, reference, "block {block:?}");
        }
    }

    /// The same property against a *narrow* declaration, which is the one the
    /// reach derivation is actually about: with `max_extent` no wider than the
    /// shapes, the gathered neighbourhood is a few blocks rather than the
    /// lattice, and the answer must still not move.
    #[test]
    fn a_narrow_declaration_gathers_fewer_blocks_and_answers_the_same() {
        let volume = [1usize, 32, 32];
        let drawn = vec![
            Drawn::region(1, 3, vec![square(1.5, 12.5), square(4.5, 9.5)]),
            Drawn::region(2, 5, vec![square(17.5, 28.5)]),
        ];
        let reference = whole(volume, &drawn).unwrap();
        let max_extent = [1usize, 13, 13];
        let mut ever_partial = false;
        for block in [[1usize, 8, 8], [1, 4, 4], [1, 16, 16]] {
            let grid = BlockGrid::new(volume, block).unwrap();
            let fragments = split(&grid, &drawn);
            let op =
                RasteriseOutlinesOp::new("rasterise", "outlines", 0, max_extent, &grid).unwrap();
            let counts = grid.blocks_per_axis();
            let mut out = Array3::<u64>::zeros((volume[0], volume[1], volume[2]));
            for core in grid.cores() {
                let wanted = neighbourhood(core.index, op.block_reach(), counts);
                let gathered: Vec<_> = fragments
                    .iter()
                    .filter(|(key, _)| wanted.contains(key))
                    .cloned()
                    .collect();
                ever_partial |= gathered.len() < fragments.len();
                let mut piece = Array3::<u64>::zeros((
                    core.core.shape[0],
                    core.core.shape[1],
                    core.core.shape[2],
                ));
                rasterise_into(
                    &gathered,
                    &grid,
                    max_extent,
                    u64::MAX,
                    &core.core,
                    piece.view_mut(),
                )
                .unwrap();
                for row in 0..core.core.shape[1] {
                    for column in 0..core.core.shape[2] {
                        out[[0, core.core.start[1] + row, core.core.start[2] + column]] =
                            piece[[0, row, column]];
                    }
                }
            }
            assert_eq!(out, reference, "block {block:?}");
        }
        assert!(
            ever_partial,
            "every block was handed every fragment, so the narrow reach was never exercised"
        );
    }

    // ------------------------------------------------------------ refusals --

    #[test]
    fn a_shape_wider_than_the_declared_extent_is_refused_by_name() {
        let volume = [1usize, 32, 32];
        let grid = BlockGrid::whole(volume).unwrap();
        let drawn = [Drawn::region(4, 1, vec![square(2.0, 20.0)])];
        let mut out = Array3::<u64>::zeros((volume[0], volume[1], volume[2]));
        let error = rasterise_into(
            &split(&grid, &drawn),
            &grid,
            [1, 8, 8],
            u64::MAX,
            &Region::whole(&volume),
            out.view_mut(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("shape 4"), "{error}");
        assert!(error.contains("declared maximum extent is 8"), "{error}");
        // and the same shape passes once the declaration covers it
        let shapes = shapes_of(&split(&grid, &drawn), &grid, [1, 20, 20], u64::MAX).unwrap();
        check_extent(&shapes, [1, 20, 20]).unwrap();
        assert_eq!(shapes[0].extent(), [1, 20, 20]);
    }

    #[test]
    fn a_row_keyed_outside_its_writers_core_is_refused() {
        let volume = [1usize, 16, 16];
        let grid = BlockGrid::new(volume, [1, 8, 8]).unwrap();
        let drawn = [Drawn::region(1, 1, vec![square(1.0, 5.0)])];
        // every row of this shape belongs to block [0, 0, 0]; hand it to [0, 1, 1]
        let mut fragments = split(&grid, &drawn);
        assert_eq!(fragments.len(), 1);
        fragments[0].0 = [0, 1, 1];
        let error = shapes_of(&fragments, &grid, volume, u64::MAX)
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside that block's core"), "{error}");
    }

    #[test]
    fn a_shape_whose_parts_share_a_ring_index_is_refused() {
        let volume = [1usize, 16, 16];
        let grid = BlockGrid::whole(volume).unwrap();
        let mut drawn = Drawn::region(6, 1, vec![square(1.0, 5.0)]);
        // a second part, numbering its own rings from zero, as a producer with
        // no running offset would write it
        drawn.rings.push((0, true, square(9.0, 13.0)));
        let error = shapes_of(&split(&grid, &[drawn]), &grid, volume, u64::MAX)
            .unwrap_err()
            .to_string();
        assert!(error.contains("shape 6 has two rows"), "{error}");
    }

    #[test]
    fn class_zero_is_refused_because_it_is_the_uncovered_value() {
        let volume = [1usize, 16, 16];
        let grid = BlockGrid::whole(volume).unwrap();
        let drawn = [Drawn::region(1, 0, vec![square(1.0, 5.0)])];
        let error = shapes_of(&split(&grid, &drawn), &grid, volume, u64::MAX)
            .unwrap_err()
            .to_string();
        assert!(error.contains("carries class 0"), "{error}");
    }

    #[test]
    fn a_class_the_destination_cannot_hold_is_refused() {
        let volume = [1usize, 16, 16];
        let grid = BlockGrid::whole(volume).unwrap();
        let drawn = [Drawn::region(1, 300, vec![square(1.0, 5.0)])];
        let ceiling = label_ceiling(Dtype::U8).unwrap();
        let error = shapes_of(&split(&grid, &drawn), &grid, volume, ceiling)
            .unwrap_err()
            .to_string();
        assert!(error.contains("carries class 300"), "{error}");
        assert!(error.contains("255"), "{error}");
        assert_eq!(label_ceiling(Dtype::F32), None, "a float holds no class");
        assert_eq!(label_ceiling(Dtype::Bool), None);
    }

    #[test]
    fn an_open_path_of_no_width_is_refused() {
        let volume = [1usize, 16, 16];
        let grid = BlockGrid::whole(volume).unwrap();
        let drawn = [Drawn::stroke(8, 1, 0.0, vec![[2.0, 2.0], [9.0, 9.0]])];
        let error = shapes_of(&split(&grid, &drawn), &grid, volume, u64::MAX)
            .unwrap_err()
            .to_string();
        assert!(error.contains("encloses no area"), "{error}");
    }

    #[test]
    fn a_stream_written_under_other_columns_is_refused_naming_one() {
        let volume = [1usize, 8, 8];
        let schema = Arc::new(Schema::new(vec![Column::u64("shape"), Column::f64("x")]).unwrap());
        let mut builder = RowBuilder::new(schema);
        builder
            .push([0, 1, 1], &[Value::U64(1), Value::F64(1.0)])
            .unwrap();
        let grid = BlockGrid::whole(volume).unwrap();
        let error = shapes_of(&[([0, 0, 0], builder.encode())], &grid, volume, u64::MAX)
            .unwrap_err()
            .to_string();
        assert!(error.contains("column"), "{error}");
    }

    // -------------------------------------------------------------- reach --

    /// The two reaches are the same parameter said twice, and the phase built
    /// from them keeps every block's core trustworthy.
    #[test]
    fn the_phase_a_rasterise_op_builds_keeps_every_core_valid() {
        let grid = BlockGrid::new([20, 32, 32], [8, 16, 16]).unwrap();
        let op = RasteriseOutlinesOp::new("rasterise", "outlines", 0, [3, 17, 9], &grid).unwrap();
        assert_eq!(op.reach(0, 20), 2, "an extent of 3 planes reaches 2");
        assert_eq!(op.reach(1, 32), 16);
        assert_eq!(op.reach(2, 32), 8);
        assert_eq!(
            op.block_reach(),
            [1, 1, 1],
            "ceil(2/8), ceil(16/16), ceil(8/16)"
        );
        let phase = fragment_phase(&op, grid).unwrap();
        assert_eq!(phase.reach, [2, 16, 8]);
        assert_eq!(phase.halo, [8, 16, 16], "one block on every axis");
        for block in &phase.blocks {
            assert_eq!(block.valid, block.core, "block {:?}", block.index);
        }
    }

    /// The declaration degrades to the full-reach phase rather than forbidding
    /// it: a caller whose shapes span the volume says so, and gets the lattice.
    #[test]
    fn declaring_the_volume_gathers_the_whole_lattice() {
        let grid = BlockGrid::new([1, 32, 32], [1, 8, 8]).unwrap();
        let op = RasteriseOutlinesOp::new("rasterise", "outlines", 0, [1, 32, 32], &grid).unwrap();
        assert_eq!(op.block_reach(), [0, 3, 3], "the lattice is 4 blocks wide");
        let counts = grid.blocks_per_axis();
        assert_eq!(
            neighbourhood([0, 0, 0], op.block_reach(), counts).len(),
            grid.n_blocks(),
            "every block of the lattice"
        );
    }

    /// The declared neighbourhood holds every block that can carry a vertex of a
    /// shape reaching this one — checked against the geometry rather than against
    /// the derivation that produced it.
    #[test]
    fn the_declared_neighbourhood_holds_every_block_that_can_reach_this_one() {
        let mut short = Vec::new();
        for volume in [[1usize, 20, 20], [4, 9, 17], [3, 32, 5]] {
            for block in [[1usize, 4, 4], [1, 8, 8], [2, 5, 3], [4, 32, 5]] {
                if (0..3).any(|axis| block[axis] > volume[axis]) {
                    continue;
                }
                let grid = BlockGrid::new(volume, block).unwrap();
                for extent in [1usize, 2, 3, 7, 12] {
                    let max_extent = [extent.min(volume[0]).max(1), extent, extent];
                    let Ok(op) =
                        RasteriseOutlinesOp::new("rasterise", "outlines", 0, max_extent, &grid)
                    else {
                        continue;
                    };
                    let counts = grid.blocks_per_axis();
                    for core in grid.cores() {
                        let wanted = neighbourhood(core.index, op.block_reach(), counts);
                        for other in grid.cores() {
                            // a shape keyed in `other` whose support just reaches
                            // `core`: every axis within `max_extent - 1` voxels
                            let reaches = (0..3).all(|axis| {
                                let (lo, hi) = (
                                    core.core.start[axis] as i64,
                                    (core.core.start[axis] + core.core.shape[axis]) as i64 - 1,
                                );
                                let (olo, ohi) = (
                                    other.core.start[axis] as i64,
                                    (other.core.start[axis] + other.core.shape[axis]) as i64 - 1,
                                );
                                let gap = (olo - hi).max(lo - ohi).max(0);
                                gap <= (max_extent[axis] as i64 - 1)
                            });
                            if reaches && !wanted.contains(&other.index) {
                                short.push(format!(
                                    "volume {volume:?} block {block:?} extent {max_extent:?}: \
                                     {:?} can reach {:?} and is not gathered",
                                    other.index, core.index
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(short.is_empty(), "{} short: {}", short.len(), short[0]);
    }

    #[test]
    fn an_op_refuses_a_lattice_its_reach_was_not_derived_for() {
        let grid = BlockGrid::new([1, 32, 32], [1, 8, 8]).unwrap();
        let op = RasteriseOutlinesOp::new("rasterise", "outlines", 0, [1, 9, 9], &grid).unwrap();
        op.check_grid(&grid).unwrap();
        let error = op
            .check_grid(&BlockGrid::new([1, 32, 32], [1, 4, 4]).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("derived its block reach"), "{error}");
    }

    #[test]
    fn an_unusable_declaration_is_refused_at_construction() {
        let grid = BlockGrid::new([1, 32, 32], [1, 8, 8]).unwrap();
        assert!(RasteriseOutlinesOp::new("r", "", 0, [1, 4, 4], &grid).is_err());
        assert!(RasteriseOutlinesOp::new("r", "a/b", 0, [1, 4, 4], &grid).is_err());
        let error = match RasteriseOutlinesOp::new("r", "outlines", 0, [1, 0, 4], &grid) {
            Ok(_) => panic!("an extent of zero was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("zero on axis 1"), "{error}");
    }

    // ----------------------------------------------------------- sampling --

    /// The sampling rule, stated against its own arithmetic rather than against
    /// a rendering: sixteen subsamples on a 4x4 grid over the pixel, on at seven.
    #[test]
    fn the_sampling_rule_is_sixteen_subsamples_on_at_seven() {
        assert_eq!(SUBSAMPLES.len() * SUBSAMPLES.len(), 16);
        assert_eq!(ON_AT, 7);
        // the offsets are the centres of the four equal parts of [-0.5, +0.5)
        for (index, offset) in SUBSAMPLES.iter().enumerate() {
            assert_eq!(*offset, -0.5 + (2.0 * index as f64 + 1.0) / 8.0);
        }
        // A ring running *through* pixel centres is the case the threshold
        // decides, and the three answers it gives are arithmetic: a pixel the
        // edge halves keeps eight subsamples, which is on; a pixel the corner
        // quarters keeps four, which is off; and the pixel beyond keeps none.
        let drawn = Drawn::region(1, 1, vec![square(4.0, 8.0)]);
        let rendered = whole([1, 13, 13], &[drawn]).unwrap();
        assert_eq!(rendered[[0, 6, 6]], 1, "the interior");
        assert_eq!(rendered[[0, 8, 6]], 1, "half covered: 8 of 16, on");
        assert_eq!(rendered[[0, 8, 8]], 0, "a quarter covered: 4 of 16, off");
        assert_eq!(rendered[[0, 9, 6]], 0, "beyond the ring: none");
    }
}
