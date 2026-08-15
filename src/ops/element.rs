// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The neighbourhood, and the order statistic taken over it. Both are shared by
// every neighbourhood op in this module — the rank filter, morphology, and the
// windowed statistics — because a second description of "which voxels are in the
// window" is a second thing that can disagree with the reach derived from it.
//
// Two decisions worth naming
// --------------------------
// **The element is stored as its offsets, in one canonical order.** Generating
// them once means every consumer walks the same neighbourhood in the same
// sequence, which is what makes a floating-point reduction over the window
// bit-identical between a block and the whole volume. An op that regenerated the
// offsets per voxel would still be *correct*; it would not necessarily be
// *reproducible*, and reproducibility across decompositions is the property this
// crate exists to defend.
//
// **The reach is derived from the element, and nothing configures it.** There is
// no field to set it to something else. A caller who wants a wider halo sets the
// halo, which is a hint. That asymmetry is the whole design:
// `docs/design/BLOCK_OPS.md` is explicit that a reach fed by the configured halo
// makes the guard compare a number against itself.
//
// **The reach has two sides, because an element can have two.** An element of
// even extent has no centre voxel, so its anchor is off centre and it reads one
// voxel further on one side than the other. `sides` is the exact statement and
// is what every op's `reach_spec` is built from; `reach` is the wider of the two
// and is the symmetric bound `BlockOp::reach` has to keep reporting. Rounding
// the short side up to the long one would be safe for the halo and *wrong for
// the element* — a different set of voxels is a different filter — so the
// asymmetry is carried rather than smoothed away, and `src/reach.rs` has had
// `AxisReach::Bounded { lo, hi }` to carry it in for longer than this module has
// been able to produce one.
//
// **The two sides are read back off the offsets, not off the box asked for.**
// They used to be the constructor's arguments, which was the same number for
// every shape that shipped — a box and both ball rules contain the poles of
// their own bounding box, so the widest member offset *was* the argument. A
// [stepped](StructuringElement::from_size_stepped) element breaks that: a
// 200-wide axis stepped by two has its last surviving offset at 98, not at 99,
// and an element that declared 99 would fetch a plane no voxel of the answer
// depends on and, worse, would state a dependency it does not have. Deriving
// them from the generated offsets keeps `reach` a *consequence* of the element
// under every parameterisation, which is the invariant the whole module is
// arranged around; there is still nothing that could set it independently.

use std::hash::{Hash, Hasher};

use crate::error::{Error, Result};
use crate::reach::Reach;

/// Which voxels of the bounding box belong to the neighbourhood.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementShape {
    /// Every voxel of the bounding box: a separable rectangular window.
    Box,
    /// The voxels inside the axis-aligned ellipsoid inscribed in the bounding
    /// box, i.e. `sum (d_a / r_a)^2 <= 1`. An axis with radius zero admits only
    /// offset zero on that axis, so a flat element is a lower-dimensional
    /// ellipse rather than an empty set.
    ///
    /// Where the two sides of an axis differ — see [`StructuringElement::sides`]
    /// — each offset is divided by **the side it is on**, so the surface still
    /// passes through both poles and `reach` stays exact on both sides. That is
    /// the only generalisation that keeps this shape's defining property (it is
    /// inscribed in its bounding box and touches every face of it); dividing
    /// both sides by the wider one would pull the short side inside its own
    /// face, and by the narrower one would push the long side outside the box.
    /// For a symmetric element the two sides are equal and this is the rule it
    /// always was.
    Ellipsoid,
    /// The ellipsoid a **different, common convention** describes: semi-axes of
    /// `ceil(extent / 2)` rather than of the radius, measured from the box's
    /// *continuous* centre rather than from a centre voxel, and **open** rather
    /// than closed.
    ///
    /// Concretely, with `n_a` the extent of the bounding box on axis `a` and
    /// `o_a` the offset from the anchor:
    ///
    /// ```text
    /// sum_a ( (o_a + (lo_a - hi_a) / 2) / ceil(n_a / 2) )^2  <  1
    /// ```
    ///
    /// The `(lo - hi) / 2` term is the anchor's displacement from the box's
    /// continuous centre — zero for a centred element, a half voxel for one
    /// built from an even extent, and whatever a hand-placed anchor makes it —
    /// so the coordinate being tested is always the voxel's own centre measured
    /// from the centre of the box, and the shape is symmetric under reflection
    /// of the box whether or not a centre voxel exists.
    ///
    /// **Why this exists at all, given [`ElementShape::Ellipsoid`].** It is a
    /// strictly larger set for an odd extent: `n = 2r + 1` normalises by `r + 1`
    /// rather than by `r`, so a size-11 element admits every offset with
    /// `d^2 < 36` where the inscribed ellipsoid keeps `d^2 <= 25`. Neither rule
    /// is more correct — the inscribed one is the more defensible and stays the
    /// default of this crate — but a caller reproducing another implementation's
    /// numbers needs *that* implementation's set, voxel for voxel, and a
    /// parameter space that cannot express it forces the caller to substitute a
    /// different filter and call it the same one.
    ///
    /// `tests/element_reference_rule.rs` states the rule a second time, in the
    /// arithmetic the implementation it came from states it in, and checks this
    /// shape against it over a thousand sizes. That is where the derivation is
    /// recorded and where a reader should go to check it; this crate takes no
    /// dependency on the implementation itself, whose licence it could not carry.
    ///
    /// A zero-extent axis cannot occur — `lo + hi + 1 >= 1` — so the divisor is
    /// never zero, and a flat axis (`lo = hi = 0`) contributes exactly zero to
    /// the sum, which is the same degenerate behaviour the inscribed ellipsoid
    /// has.
    ExtentEllipsoid,
}

/// A neighbourhood, anchored on the voxel it is evaluated at.
///
/// Parameterised by shape and by a per-axis pair of one-sided extents, and by
/// nothing else — there is no default size anywhere in this crate, because a
/// filter size is a property of the images being processed and therefore of the
/// caller.
///
/// **The two sides of an axis may differ**, which is what makes an even extent
/// expressible: an even window has no centre voxel, so the anchor is off centre
/// by construction and the element reads one voxel further on one side than on
/// the other. See [`Self::from_size`] for which side, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuringElement {
    /// Which named shape generated the offsets, or `None` for an element built
    /// from an explicit list.
    ///
    /// Descriptive only — nothing derives geometry, cost or membership from it
    /// after construction, which is why an element with no name is a perfectly
    /// ordinary element rather than a second kind of thing.
    shape: Option<ElementShape>,
    /// Voxels the element reads **below** the anchor, per axis. Derived from
    /// `offsets` — the widest surviving member on that side — and not from the
    /// box the constructor was handed. See the module header.
    lo: [usize; 3],
    /// Voxels the element reads **above** the anchor, per axis. Derived, as
    /// `lo` is.
    hi: [usize; 3],
    /// The decimation applied to the bounding box while generating `offsets`;
    /// `[1, 1, 1]` for every element that is not stepped. Carried so that the
    /// element can describe itself, never to compute a reach from — the reach
    /// comes from the offsets that survived it.
    step: [usize; 3],
    offsets: Vec<[isize; 3]>,
}

impl StructuringElement {
    /// From a per-axis radius. The element spans `2 * radius + 1` on each axis
    /// and is symmetric, which is the case every op in this crate shipped with.
    pub fn from_radius(shape: ElementShape, radius: [usize; 3]) -> Self {
        Self::from_sides(shape, radius, radius)
    }

    /// From the two sides of each axis stated separately: `lo` voxels below the
    /// anchor and `hi` above, so the bounding box spans `lo + hi + 1`.
    ///
    /// The general constructor. [`Self::from_radius`] is `lo == hi` and
    /// [`Self::from_size`] derives the pair from an extent; this is here because
    /// an anchor that is neither centred nor derived from an extent is a
    /// legitimate thing to want, and because it is the form the rest of the type
    /// is written against.
    pub fn from_sides(shape: ElementShape, lo: [usize; 3], hi: [usize; 3]) -> Self {
        Self::build(shape, lo, hi, [1, 1, 1])
    }

    /// From an explicit list of offsets — the constructor for a shape this
    /// crate does not name.
    ///
    /// **This is the extension point, and it is a constructor rather than a
    /// trait for a reason.** [`ElementShape`]'s membership test runs once per
    /// candidate offset *at construction* and never again, so opening it as a
    /// trait would buy a caller nothing at run time and would put a generic
    /// parameter on a type that is stored, cloned and compared everywhere. What
    /// a caller actually cannot do today is hand over a set of offsets, and that
    /// is what this fixes.
    ///
    /// Three properties are established here rather than trusted, because every
    /// one of them is something the rest of the crate assumes:
    ///
    /// * **the offsets are sorted**, ascending on axis 0, then 1, then 2 — the
    ///   order [`from_sides`](Self::from_sides) generates. A gathered window is
    ///   summed in offset order and floating-point addition does not associate,
    ///   so the answer depends on it: sorting makes the element a function of
    ///   the *set* a caller meant rather than of the order they happened to
    ///   write it in, and makes two callers who meant the same element get the
    ///   same numbers;
    /// * **duplicates are refused**, naming the repeat. A duplicate offset is
    ///   gathered twice, so it doubles that voxel's weight in a mean and shifts
    ///   every rank — a silent change to the filter rather than to the window;
    /// * **an empty list is refused**, for the reason every constructor here
    ///   refuses one: an element with no members selects nothing, and every op
    ///   that reduces over it has no value to write.
    ///
    /// The reach is derived from the offsets given, exactly as it is for a
    /// generated element — see [`Self::sides`]. There is no way to state it
    /// separately and therefore nothing that can drift from the arithmetic.
    pub fn from_offsets(offsets: impl IntoIterator<Item = [isize; 3]>) -> Result<Self> {
        let mut offsets: Vec<[isize; 3]> = offsets.into_iter().collect();
        if offsets.is_empty() {
            return Err(Error::InvalidArgument(
                "a structuring element needs at least one offset; an empty element selects \
                 nothing and every op that reduces over one has no value to write"
                    .to_string(),
            ));
        }
        offsets.sort_unstable();
        if let Some(pair) = offsets.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidArgument(format!(
                "the offset {:?} appears more than once. A repeated offset is gathered twice, \
                 so it doubles that voxel's weight in a mean and shifts every rank — which is \
                 a change to the filter rather than to the window, and is refused rather than \
                 quietly folded away.",
                pair[0]
            )));
        }
        let (lo, hi) = spanned_sides(&offsets);
        Ok(Self {
            shape: None,
            lo,
            hi,
            step: [1, 1, 1],
            offsets,
        })
    }

    /// The general constructor with a **step**: only every `step[axis]`-th
    /// offset along each axis is a member.
    ///
    /// **What this is, and what it is not.** It decimates the *element* — the
    /// window still spans the box `lo + hi + 1` asked for, but holds a fraction
    /// `1 / (step[0] * step[1] * step[2])` of the offsets inside it, and
    /// [`Self::len`] is the count that survived. It is **not** a coarser grid of
    /// sampling positions: the filter is still evaluated at every voxel. The
    /// crate has that separately, as `ops::local`'s `SampleLattice`, and the two
    /// are composable and not interchangeable — a lattice makes the *answer*
    /// coarse and interpolates it back, while a step makes the *window* sparse
    /// and leaves the answer at full resolution.
    ///
    /// **Why a caller wants it.** The cost of a rank filter is its element, and
    /// the elements that dominate a chain are the large flat ones — a
    /// `200 x 200 x 1` window is 40 000 values to gather and select from at
    /// every voxel. Decimating it by two on each of the wide axes leaves 10 000,
    /// a quarter of the cost, and for a statistic that describes a smooth
    /// low-frequency field the two answers are close. `cost_per_voxel` follows
    /// automatically, because it is a function of `len`.
    ///
    /// **The phase, and what it costs at a low boundary.** The surviving offsets
    /// are counted from the **low** end of the box: `-lo`, `-lo + step`, … up to
    /// `hi`. The anchor itself is a member only when `lo` is a multiple of
    /// `step`, which is a property of the element and is stated rather than
    /// patched. Because the offsets are fixed, a window truncated at a boundary
    /// keeps exactly the members that still land inside — the phase does not
    /// shift. An implementation that instead slices the *clipped* range with a
    /// stride (`a[max(0, c - lo) : min(c + hi, n) : step]`, which is what the
    /// obvious array expression does) re-phases the decimation wherever
    /// `(lo - c) % step != 0`, so at the low face it reads a different set of
    /// voxels for every centre. This crate does not do that: a fixed offset set
    /// is what makes the element the same filter everywhere, which is what makes
    /// a block and a whole volume agree.
    ///
    /// The shape's membership test is applied to the surviving offset's **true**
    /// position, so a stepped ellipsoid is the ellipsoid's offsets decimated and
    /// not a decimated index space reinterpreted as an ellipsoid.
    ///
    /// Every `step[axis]` must be non-zero; `1` is the ordinary element and is
    /// exactly [`Self::from_sides`].
    pub fn from_sides_stepped(
        shape: ElementShape,
        lo: [usize; 3],
        hi: [usize; 3],
        step: [usize; 3],
    ) -> Result<Self> {
        for axis in 0..3 {
            if step[axis] == 0 {
                return Err(Error::InvalidArgument(format!(
                    "an element's step must be non-zero on every axis; got {step:?}. A step of \
                     zero names no offsets at all, which is an empty window rather than a \
                     cheaper one."
                )));
            }
        }
        let element = Self::build(shape, lo, hi, step);
        // A step can strand a shape entirely: a radius-3 ball stepped by nine
        // keeps only the offset `[-3, -3, -3]`, which is outside its own
        // surface, and the element comes out empty. Unstepped elements cannot
        // do this — a box is never empty and both ball rules contain the anchor
        // — so this is the one constructor that has to say so. Refused rather
        // than returned, because an empty element is not a cheaper filter: it is
        // a window with nothing in it, and every consumer downstream would have
        // to invent an answer for a rank over no values.
        if element.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "a step of {step:?} leaves no offset inside a {:?} element of sides {lo:?}/{hi:?}; \
                 the decimation is coarser than the shape it decimates",
                shape
            )));
        }
        Ok(element)
    }

    fn build(shape: ElementShape, lo: [usize; 3], hi: [usize; 3], step: [usize; 3]) -> Self {
        let offsets = generate(shape, lo, hi, step);
        let (lo, hi) = spanned_sides(&offsets);
        Self {
            shape: Some(shape),
            lo,
            hi,
            step,
            offsets,
        }
    }

    /// From a per-axis size, which must be non-zero on every axis.
    ///
    /// **Where the anchor goes when the size is even, and why.** An even size
    /// has no centre voxel, so the anchor is a decision rather than a
    /// consequence. This crate puts it at index `size / 2`, which leaves
    /// `size / 2` voxels below it and `size / 2 - 1` above: the element reads
    /// **one voxel further below the anchor than above it**.
    ///
    /// Three reasons for that side rather than the other:
    ///
    /// * it is what the reference implementations a caller is likely to have to
    ///   match do. The convention that an even footprint's origin sits at
    ///   `floor(n / 2)` is the one the established array-filtering libraries
    ///   use, and the element definition [`ElementShape::ExtentEllipsoid`] was
    ///   derived from indexes its own mesh from `floor(n / 2)` for exactly the
    ///   same reason — so an anchor on the other side would make that shape
    ///   match the set it reproduces and not the *position* it reproduces it at,
    ///   which is a parity that fails at every voxel by a shift of one;
    /// * it makes the bounding box grow one voxel at a time in a fixed pattern —
    ///   `size` to `size + 1` adds a plane below when the new size is even and
    ///   above when it is odd — so an extent swept upward never jumps by two and
    ///   never shifts the anchor backwards;
    /// * whichever side is chosen, the answer is a shifted volume, and a
    ///   *stated* shift can be undone by a caller who wanted the other one while
    ///   a silent one cannot.
    ///
    /// The resulting reach is genuinely asymmetric — `lo != hi` on that axis —
    /// and every op that derives geometry from an element says so per side. It
    /// is deliberately **not** rounded up to a symmetric window: that would
    /// fetch a voxel nobody reads on the narrow side, and, far worse, would make
    /// the element a different *set* from the one asked for.
    pub fn from_size(shape: ElementShape, size: [usize; 3]) -> Result<Self> {
        Self::from_size_stepped(shape, size, [1, 1, 1])
    }

    /// [`Self::from_size`] with the step of [`Self::from_sides_stepped`], which
    /// is where both rules are stated.
    ///
    /// The anchor is placed from the **full** extent, exactly as it is without a
    /// step — a `200 x 200 x 1` box anchors at `(100, 99)` on each wide axis
    /// whether or not it is then decimated — so a caller who steps an element
    /// gets the same window position and a sparser sample of it, rather than a
    /// window that also moved.
    pub fn from_size_stepped(
        shape: ElementShape,
        size: [usize; 3],
        step: [usize; 3],
    ) -> Result<Self> {
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for axis in 0..3 {
            if size[axis] == 0 {
                return Err(Error::InvalidArgument(format!(
                    "an element needs a non-zero size on every axis; got {size:?}"
                )));
            }
            lo[axis] = size[axis] / 2;
            hi[axis] = size[axis] - 1 - lo[axis];
        }
        Self::from_sides_stepped(shape, lo, hi, step)
    }

    /// Which named shape generated this element, or `None` if it was built from
    /// an explicit offset list. Descriptive only.
    pub fn shape(&self) -> Option<ElementShape> {
        self.shape
    }

    /// The decimation this element was built with; `[1, 1, 1]` unless it came
    /// from one of the stepped constructors.
    ///
    /// Descriptive only. Nothing in this crate derives geometry or cost from it
    /// — those come from the offsets and from [`Self::len`], which already
    /// account for it.
    pub fn step(&self) -> [usize; 3] {
        self.step
    }

    /// The per-axis radius, when there is one.
    ///
    /// Equal to [`Self::reach`] and, for an element with a centre voxel, to both
    /// of its [`sides`](Self::sides). An element with an even extent has no
    /// radius, and this reports the wider of its two sides; `sides` is the exact
    /// statement and is what anything deriving geometry must use.
    pub fn radius(&self) -> [usize; 3] {
        [self.reach(0), self.reach(1), self.reach(2)]
    }

    /// The bounding box of what the element actually reads, per axis:
    /// `lo + hi + 1`, with `lo` and `hi` the widest surviving offsets.
    ///
    /// For an unstepped box or ball this is the size it was built from, because
    /// those contain the poles of their own box. For a stepped element it is the
    /// span the survivors reach: a 200-wide axis stepped by two spans
    /// `-100 ..= 98`, so 199. Deriving it rather than storing the requested
    /// extent is the same decision `sides` makes and for the same reason —
    /// there is one description of the window and it is the offsets.
    pub fn size(&self) -> [usize; 3] {
        [
            self.lo[0] + self.hi[0] + 1,
            self.lo[1] + self.hi[1] + 1,
            self.lo[2] + self.hi[2] + 1,
        ]
    }

    /// Voxels read **below** and **above** the anchor along `axis`.
    ///
    /// The exact statement of what this element reads, and the one every op that
    /// derives a `Reach` from an element must use. [`Self::reach`] is the
    /// symmetric bound on this pair, kept because `BlockOp::reach` is a single
    /// integer per axis; the pair is what `BlockOp::reach_spec` carries.
    pub fn sides(&self, axis: usize) -> (usize, usize) {
        (self.lo[axis], self.hi[axis])
    }

    /// Whether every axis reads as far below the anchor as above it.
    ///
    /// The question to ask where a signature can hold only one integer per axis
    /// and the choice is between over-declaring and refusing. Both places in
    /// this crate that face it — `HExtremaOp::operands` and `VoxelizeOp::reach`
    /// — over-declare and say so at the declaration; a caller who would rather
    /// refuse than pay for the wider side asks this first.
    pub fn is_symmetric(&self) -> bool {
        self.lo == self.hi
    }

    /// What one pass of this element reads, per side and per axis.
    ///
    /// This is the honest statement; a composition of `passes` such passes reads
    /// `passes` times it on each side, which is [`Self::reach_spec_after`].
    pub fn reach_spec(&self) -> Reach {
        self.reach_spec_after(1)
    }

    /// The same, for an operation that applies this element `passes` times in
    /// sequence **without reflecting it** between passes.
    ///
    /// The reflection matters and is why this is stated here rather than
    /// multiplied at each call site: composing `x -> x + o` with itself for
    /// `o` in the element gives offsets `o + o'`, whose extremes are `passes`
    /// times each side. Had the second pass used the *reflected* element — as a
    /// textbook opening does — the extremes would be `lo + hi` on **both**
    /// sides, which is a different number for an off-centre window and the same
    /// one for a centred window. `src/ops/morphology.rs` composes without
    /// reflecting, so this is the rule it needs; anything that starts reflecting
    /// must stop using this method rather than quietly inherit it.
    pub fn reach_spec_after(&self, passes: usize) -> Reach {
        Reach::asymmetric([
            (self.lo[0] * passes, self.hi[0] * passes),
            (self.lo[1] * passes, self.hi[1] * passes),
            (self.lo[2] * passes, self.hi[2] * passes),
        ])
    }

    /// Voxels read beyond the anchor, along `axis`, **on the wider side**.
    ///
    /// The symmetric bound: what a caller that can only hold one integer per
    /// axis must use, and what `BlockOp::reach` reports. It is a bound and not
    /// the whole truth for an element with an even extent, which is why
    /// [`Self::sides`] exists and why every op in `src/ops/` states its
    /// `reach_spec` from that instead — over-declaring here costs reads, and
    /// `Chain::reach_spec` checks that the exact form stays inside this bound.
    ///
    /// It is the radius even for an ellipsoid, because the ellipsoid still
    /// contains the two poles on each axis. A shape that excluded them would
    /// report less, which is why this asks the shape rather than assuming.
    pub fn reach(&self, axis: usize) -> usize {
        self.lo[axis].max(self.hi[axis])
    }

    /// How many voxels the element contains when nothing clamps it.
    ///
    /// This is the `n` an order statistic is expressed against; see [`Rank`].
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// The member offsets, in ascending lexicographic order of `[a, b, c]`.
    ///
    /// The order is part of the contract, not an accident of construction: a
    /// windowed sum walks these in sequence, and a different sequence is a
    /// different floating-point answer.
    pub fn offsets(&self) -> &[[isize; 3]] {
        &self.offsets
    }
}

fn generate(
    shape: ElementShape,
    lo: [usize; 3],
    hi: [usize; 3],
    step: [usize; 3],
) -> Vec<[isize; 3]> {
    let along: Vec<Vec<isize>> = (0..3)
        .map(|axis| axis_offsets(lo[axis], hi[axis], step[axis]))
        .collect();
    let mut offsets = Vec::new();
    for &a in &along[0] {
        for &b in &along[1] {
            for &c in &along[2] {
                // The membership test sees the *true* offset, so a decimated
                // ball is the ball's members thinned out rather than a smaller
                // ball drawn in the decimated index space. The two differ: the
                // second would round the surface inwards on every axis it
                // stepped.
                if member(shape, lo, hi, [a, b, c]) {
                    offsets.push([a, b, c]);
                }
            }
        }
    }
    offsets
}

/// The surviving offsets along one axis: `-lo`, `-lo + step`, … up to `hi`.
///
/// Counted from the low end, which is where [`StructuringElement::from_sides_stepped`]
/// says the phase is anchored. `step == 1` gives the whole range back and is the
/// unstepped case, unchanged.
fn axis_offsets(lo: usize, hi: usize, step: usize) -> Vec<isize> {
    let mut offsets = Vec::new();
    let mut offset = -(lo as isize);
    while offset <= hi as isize {
        offsets.push(offset);
        offset += step as isize;
    }
    offsets
}

/// The two sides each axis actually reaches, read off the offsets.
///
/// The zero floor is not a clamp of something that could be negative — it is the
/// statement that a side reaching *no further than the anchor* reaches zero.
/// `Reach` counts voxels beyond the anchor and cannot express "not even the
/// anchor plane", so zero is the tightest truth available for an element whose
/// offsets all fall on one side; over-declaring by that one plane costs a plane
/// of halo and never costs correctness.
fn spanned_sides(offsets: &[[isize; 3]]) -> ([usize; 3], [usize; 3]) {
    let mut lo = [0isize; 3];
    let mut hi = [0isize; 3];
    for offset in offsets {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(offset[axis]);
            hi[axis] = hi[axis].max(offset[axis]);
        }
    }
    (
        [
            lo[0].unsigned_abs(),
            lo[1].unsigned_abs(),
            lo[2].unsigned_abs(),
        ],
        [hi[0] as usize, hi[1] as usize, hi[2] as usize],
    )
}

fn member(shape: ElementShape, lo: [usize; 3], hi: [usize; 3], offset: [isize; 3]) -> bool {
    match shape {
        ElementShape::Box => true,
        ElementShape::Ellipsoid => {
            let mut total = 0.0_f64;
            for axis in 0..3 {
                // The side the offset is on is the one it is measured against,
                // so the surface passes through both poles of an off-centre
                // axis. See `ElementShape::Ellipsoid` for why not the other two
                // candidates.
                let side = if offset[axis] < 0 { lo[axis] } else { hi[axis] };
                if side == 0 {
                    if offset[axis] != 0 {
                        return false;
                    }
                    continue;
                }
                let scaled = offset[axis] as f64 / side as f64;
                total += scaled * scaled;
            }
            total <= 1.0
        }
        // Transcribed from the rule stated on `ElementShape::ExtentEllipsoid`,
        // and arranged so that the arithmetic is the arithmetic that rule
        // describes: the shift is a half voxel or nothing, the divisor is the
        // half-extent rounded up, and the comparison is strict. The sum runs
        // over the axes in order because a sum of three `f64`s is not
        // associative and this crate's elements are compared voxel for voxel.
        ElementShape::ExtentEllipsoid => {
            let mut total = 0.0_f64;
            for axis in 0..3 {
                // Half the difference between the sides: what moves the origin
                // from the anchor to the box's continuous centre. Exactly zero
                // for a centred element and exactly a half for one built from
                // an even extent, and exact for any hand-placed anchor too —
                // the difference of two integers halved is representable.
                let shift = (lo[axis] as f64 - hi[axis] as f64) / 2.0;
                // `ceil(n / 2)` with `n = lo + hi + 1`, so at least one.
                let divisor = ((lo[axis] + hi[axis] + 2) / 2) as f64;
                let scaled = (offset[axis] as f64 + shift) / divisor;
                total += scaled * scaled;
            }
            total < 1.0
        }
    }
}

/// A fraction of the way up a sorted window, validated once.
///
/// Exists so that [`Rank`] can carry a percentile and still be `Eq + Hash` —
/// `Statistic` in `ops::local` derives both and holds a `Rank`. The validation
/// is what makes the derives sound rather than merely available: no `NaN` (a
/// window that compared unequal to itself would turn every byte-identity check
/// downstream into a pass that means nothing), and `-0.0` normalised to `+0.0`
/// so that the hash agrees with the equality. `Isodata` in `ops::local` carries
/// its `f64` under exactly the same rules, for the same reason.
///
/// Out-of-range is refused rather than clamped, which is the opposite of
/// [`Rank::percentile`]'s convention and is deliberate: that constructor takes a
/// fraction as a convenience for naming a rank and has a defensible reading for
/// anything outside `[0, 1]`, while this one *is* the parameter of a filter, and
/// a caller who asked for the 130th percentile has made an arithmetic mistake
/// that a silent clamp would carry into the output.
#[derive(Debug, Clone, Copy)]
pub struct Percentile(f64);

impl Percentile {
    /// A fraction in `[0, 1]`: `0.0` the lowest value, `1.0` the highest.
    pub fn new(fraction: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&fraction) {
            return Err(Error::InvalidArgument(format!(
                "a percentile is a fraction in [0, 1], got {fraction}"
            )));
        }
        // `-0.0` and `+0.0` are the same fraction and different bit patterns;
        // normalising here is cheaper than a `Hash` that has to remember the
        // exception, and keeps `Hash`'s contract with `Eq`.
        let fraction = if fraction == 0.0 { 0.0 } else { fraction };
        Ok(Self(fraction))
    }

    pub fn fraction(&self) -> f64 {
        self.0
    }
}

impl PartialEq for Percentile {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for Percentile {}

impl Hash for Percentile {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Which order statistic of a neighbourhood to select.
///
/// **The median is the `k = n / 2` case, not a separate implementation** — it is
/// a constructor, and every rank goes through the same selection. There is one
/// selection in this crate and both variants below reach it.
///
/// The two variants, and why the second one exists
/// -----------------------------------------------
/// They are not two statistics. They are two **truncation conventions**: two
/// answers to the question a window clipped at a volume boundary asks, where the
/// element holds `n` offsets but only `m < n` of them landed inside. Both are
/// defensible general policies, both are in wide use, and they disagree, so the
/// choice is a parameter rather than a decision this crate makes on a caller's
/// behalf.
///
/// * [`Rank::Nth`] — **proportional rescaling of the rank.** The statistic is
///   named as a position `k` in the full element's `n`, and a truncated window
///   moves it to the same relative position among the `m` values that exist:
///   `round(k * (m - 1) / (n - 1))`. The median of a half-window stays its
///   median.
/// * [`Rank::CeilingPercentile`] — **a ceiling percentile over the surviving
///   population.** The statistic is named as a fraction `p`, and the answer is
///   the `ceil(p * m)`-th smallest of the `m` values actually read, one-based.
///   The full element never enters the arithmetic at all.
///
/// **Where they agree, and where they do not.** At `p = 0.5` the proportional
/// rule reduces to `floor(m / 2)` for every `m` and every `n` — the upper of the
/// two middles — and the ceiling rule gives `ceil(m / 2) - 1`. Those are the
/// same index for **odd** `m` and differ by one for even `m`. Away from a half
/// they differ generally, and they differ on the *untruncated* window too:
/// `Rank::percentile(e, 0.25)` on a 27-voxel element is `Nth(7)` where the
/// ceiling rule is `6`, because `round(p * (n - 1))` and `ceil(p * n) - 1` are
/// not the same map. `tests/rank_truncation_rule.rs` sweeps both and pins both
/// halves of that.
///
/// [`Rank::Nth`] is the default in the sense that matters — it is what every
/// constructor in this type produces, so nothing changes for a caller who does
/// not ask for the other one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rank {
    /// The `k`-th smallest value, counting from zero, of the **full** element's
    /// `n` values, **rescaled proportionally** where the window is truncated.
    ///
    /// Where the element is truncated — at a real volume boundary, where there
    /// is nothing beyond to read — only `m < n` values exist, and `k` is
    /// rescaled to the values that do: `round(k * (m - 1) / (n - 1))`. See
    /// [`Rank::resolve`] for why that is the right rule rather than clamping.
    Nth(usize),
    /// The `ceil(p * m)`-th smallest, one-based, of the `m` values actually
    /// read — **a ceiling percentile over the surviving population**.
    ///
    /// Truncation needs no rule here, because the population is already the
    /// truncated one: the statistic is defined against what was read rather than
    /// against what the element would have held. That is the whole difference,
    /// and it is why this variant carries a fraction where the other carries a
    /// position — the fraction cannot be recovered from a position, so a policy
    /// that could only re-interpret a `k` would be a different rule again.
    ///
    /// `p = 0` selects the window minimum. An implementation that walks a
    /// histogram from its first bin answers the *first bin index* instead —
    /// zero, occupied or not — and this crate does not reproduce that: a rank
    /// filter here writes a value it read, which is what makes
    /// `constant_maps_to` exact at every truncation, and a literal zero from an
    /// empty bin is not one of the values read.
    CeilingPercentile(Percentile),
}

impl Rank {
    /// The smallest value: an erosion, on a totally ordered element type.
    pub fn lowest() -> Self {
        Rank::Nth(0)
    }

    /// The median of `element`. `k = n / 2`, and for an even `n` that is the
    /// upper of the two central values — stated so that it is a decision on
    /// record rather than a consequence of integer division.
    pub fn median(element: &StructuringElement) -> Self {
        Rank::Nth(element.len() / 2)
    }

    /// The largest value: a dilation, on a totally ordered element type.
    pub fn highest(element: &StructuringElement) -> Self {
        Rank::Nth(element.len().saturating_sub(1))
    }

    /// The rank `fraction` of the way up the element's sorted values — `0.0` the
    /// lowest, `1.0` the highest, `0.5` the median.
    ///
    /// A percentile is a way of *naming* a rank, not a second kind of statistic,
    /// so it is a constructor over the same variant and goes through the same
    /// selection. `percentile(e, 0.5)` and `median(e)` agree by arithmetic and
    /// not by coincidence; the test below pins that.
    pub fn percentile(element: &StructuringElement, fraction: f64) -> Self {
        let full = element.len();
        if full <= 1 {
            return Rank::Nth(0);
        }
        let position = fraction.clamp(0.0, 1.0) * (full - 1) as f64;
        Rank::Nth(position.round() as usize)
    }

    /// The other truncation convention: the `ceil(fraction * m)`-th smallest,
    /// one-based, of the `m` values a window actually read.
    ///
    /// Takes no element, which is the point — the rule is stated against the
    /// surviving population and never consults the full one. See
    /// [`Rank::CeilingPercentile`] for how it differs from the proportional
    /// rule, and where the two coincide.
    pub fn ceiling_percentile(fraction: f64) -> Result<Self> {
        Ok(Rank::CeilingPercentile(Percentile::new(fraction)?))
    }

    /// Where in the `available` values actually read this rank lands, given that
    /// the element holds `full` when nothing clamps it.
    ///
    /// **Why [`Rank::Nth`] rescales rather than clamps.** Clamping `k` to
    /// `available - 1` would turn a median into a maximum wherever the element
    /// is truncated, so every face of the volume would get a systematically
    /// different filter from the interior. Rescaling keeps the *relative
    /// position* in the sorted window, which is what a rank means: the median of
    /// a half-window is still its median. The arithmetic is integer and rounds
    /// half up, so it is exactly reproducible.
    ///
    /// [`Rank::CeilingPercentile`] does not rescale, because there is nothing to
    /// rescale from: its rule is already written against `available`, and `full`
    /// is unused for it. That is the substance of the two conventions and not an
    /// implementation detail of this method.
    pub fn resolve(&self, full: usize, available: usize) -> usize {
        if available <= 1 {
            return 0;
        }
        match *self {
            Rank::Nth(k) => {
                if full <= 1 {
                    return 0;
                }
                let k = k.min(full - 1);
                let denominator = full - 1;
                (k * (available - 1) + denominator / 2) / denominator
            }
            Rank::CeilingPercentile(percentile) => {
                // One-based `ceil(p * m)`, taken to a zero-based index. The
                // `ceil` is exact: `p * m` is a single rounded product and
                // `f64::ceil` is exact on it, so two runs on the same inputs
                // cannot land in different bins.
                //
                // `p * m` is at most `m`, so the index is at most `m - 1`; the
                // `min` is belt and braces against a product that rounded up
                // past `m` and costs one comparison per voxel. The floor at zero
                // is `p = 0`, where the ceiling is zero and the one-based rank
                // would be the value before the first.
                let position = (percentile.fraction() * available as f64).ceil();
                if position < 1.0 {
                    0
                } else {
                    (position as usize - 1).min(available - 1)
                }
            }
        }
    }
}

/// The `index`-th smallest of `values`, by selection rather than by sorting.
///
/// Takes `&mut` because selection permutes in place; the caller owns a scratch
/// buffer and reuses it across voxels, which is what keeps a rank filter from
/// allocating once per voxel.
pub fn select_nth<T: Ord + Copy>(values: &mut [T], index: usize) -> Option<T> {
    if values.is_empty() {
        return None;
    }
    let index = index.min(values.len() - 1);
    let (_, nth, _) = values.select_nth_unstable(index);
    Some(*nth)
}

/// A total order over `f64`, so that a rank filter generic over `Ord` can be
/// applied to floating-point images.
///
/// `f64::total_cmp` is a total order over every bit pattern, which is what `Ord`
/// requires and what `PartialOrd` on `f64` does not give. The filter selects an
/// *existing* value and hands it back, so the wrapper never perturbs a number —
/// it only decides which one wins. NaNs sort above every finite value and
/// negative zero below positive zero; both are consequences of `total_cmp`, are
/// deterministic, and are the reason this is a named type rather than an inline
/// closure that a caller might write differently somewhere else.
#[derive(Debug, Clone, Copy)]
pub struct Total(pub f64);

impl PartialEq for Total {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == std::cmp::Ordering::Equal
    }
}

impl Eq for Total {}

impl PartialOrd for Total {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Total {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {

    /// The extension point: an element this crate has no name for, built from a
    /// hand-written offset list.
    ///
    /// Asserts the three properties `from_offsets` establishes rather than
    /// trusts — sorted, no duplicates, non-empty — plus the one that matters
    /// most: the reach is **derived from the offsets given**, so a caller cannot
    /// state it and cannot get it wrong.
    #[test]
    fn an_element_from_an_explicit_offset_list_derives_everything_it_needs() {
        // A shape no `ElementShape` produces: a diagonal cross, asymmetric on
        // axis 0, offered out of order to prove the sort.
        let raw = vec![[2, 0, 0], [0, 0, 0], [-1, 1, 0], [0, -1, 1], [-1, -1, 0]];
        let element = StructuringElement::from_offsets(raw).unwrap();

        assert_eq!(element.len(), 5);
        assert_eq!(element.shape(), None, "it is not one of the named shapes");
        assert_eq!(
            element.offsets(),
            &[[-1, -1, 0], [-1, 1, 0], [0, -1, 1], [0, 0, 0], [2, 0, 0]],
            "sorted ascending on axis 0, then 1, then 2 — the order `from_sides` generates"
        );

        // Derived, per side, from the widest surviving offset. Axis 0 reaches 1
        // below and 2 above; a symmetric reading would report 2 on both and
        // fetch a plane nothing depends on.
        assert_eq!(element.sides(0), (1, 2));
        assert_eq!(element.sides(1), (1, 1));
        assert_eq!(element.sides(2), (0, 1));
        assert_eq!(element.reach(0), 2, "the symmetric bound is the wider side");

        // brute force against the offsets themselves, which is the statement
        // the derivation is supposed to make
        for axis in 0..3 {
            let widest_below = element
                .offsets()
                .iter()
                .map(|offset| (-offset[axis]).max(0) as usize)
                .max()
                .unwrap();
            let widest_above = element
                .offsets()
                .iter()
                .map(|offset| offset[axis].max(0) as usize)
                .max()
                .unwrap();
            assert_eq!(element.sides(axis), (widest_below, widest_above));
        }
    }

    #[test]
    fn an_offset_list_that_cannot_be_an_element_is_refused_where_it_is_stated() {
        let empty: Vec<[isize; 3]> = Vec::new();
        assert!(StructuringElement::from_offsets(empty).is_err());

        let repeated = StructuringElement::from_offsets(vec![[0, 0, 0], [1, 0, 0], [0, 0, 0]]);
        let message = repeated.unwrap_err().to_string();
        assert!(message.contains("[0, 0, 0]"), "{message}");
        assert!(message.contains("more than once"), "{message}");

        // one offset is enough, and it need not be the origin
        assert!(StructuringElement::from_offsets(vec![[3, 0, 0]]).is_ok());
    }

    /// A hand-built element that reproduces a named one must *be* it — same
    /// offsets, same sides, same length. If it were not, the two constructors
    /// would be two different notions of an element.
    #[test]
    fn an_explicit_list_of_a_named_shape_s_offsets_rebuilds_that_element() {
        for named in [
            StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]),
            StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 1, 0]),
            StructuringElement::from_size(ElementShape::ExtentEllipsoid, [4, 4, 1]).unwrap(),
        ] {
            let rebuilt = StructuringElement::from_offsets(named.offsets().to_vec()).unwrap();
            assert_eq!(rebuilt.offsets(), named.offsets());
            assert_eq!(rebuilt.len(), named.len());
            for axis in 0..3 {
                assert_eq!(rebuilt.sides(axis), named.sides(axis));
                assert_eq!(rebuilt.reach(axis), named.reach(axis));
            }
        }
    }
    use super::*;

    #[test]
    fn a_size_of_seven_reaches_three() {
        let element = StructuringElement::from_size(ElementShape::Box, [7, 7, 7]).unwrap();
        assert_eq!(element.radius(), [3, 3, 3]);
        assert_eq!(element.reach(0), 3);
        assert_eq!(element.len(), 7 * 7 * 7);
    }

    /// Replaces `an_even_size_is_refused_rather_than_rounded`, which pinned the
    /// refusal this module used to make. The refusal was not a fact about
    /// geometry — an even window is a perfectly definite set — it was a refusal
    /// to *choose an anchor*, and the choice is now made and stated. What has to
    /// stay pinned is that the choice is a shift and never a rounding: a size of
    /// four must span four, not five.
    #[test]
    fn an_even_size_is_anchored_below_centre_rather_than_rounded_up() {
        let element = StructuringElement::from_size(ElementShape::Box, [4, 3, 3]).unwrap();
        assert_eq!(element.sides(0), (2, 1), "one further below than above");
        assert_eq!(element.sides(1), (1, 1));
        assert_eq!(element.size(), [4, 3, 3]);
        assert_eq!(element.len(), 4 * 3 * 3);
        assert!(!element.is_symmetric());
        // the offsets span exactly the four positions, not five
        assert!(element.offsets().iter().all(|o| (-2..=1).contains(&o[0])));
        assert!(element.offsets().contains(&[-2, 0, 0]));
        assert!(!element.offsets().contains(&[2, 0, 0]));

        // and a size of zero is still nothing this can build
        let err = StructuringElement::from_size(ElementShape::Box, [0, 3, 3]).unwrap_err();
        assert!(err.to_string().contains("non-zero size"), "got: {err}");
    }

    /// The whole point of the asymmetry: an even element's reach is not a
    /// symmetric integer, and everything that derives geometry has to see the
    /// pair. The symmetric bound is kept as a bound and marked as one.
    #[test]
    fn an_even_elements_reach_is_asymmetric_and_the_integer_is_only_a_bound() {
        let element = StructuringElement::from_size(ElementShape::Box, [10, 7, 4]).unwrap();
        assert_eq!(element.sides(0), (5, 4));
        assert_eq!(element.sides(1), (3, 3));
        assert_eq!(element.sides(2), (2, 1));
        assert_eq!(element.reach(0), 5, "the wider side");
        assert_eq!(element.radius(), [5, 3, 2]);

        let spec = element.reach_spec();
        assert_eq!(
            spec.as_symmetric(),
            None,
            "it is not a triple and never was"
        );
        assert_eq!(spec.at(0, 0, 100), (5, 4));
        assert_eq!(spec.at(2, 0, 100), (2, 1));
        // two passes double each side; they do not become `lo + hi` on both,
        // which is what a reflected composition would give
        assert_eq!(element.reach_spec_after(2).at(0, 0, 100), (10, 8));
    }

    /// An odd size is exactly the element it was before even sizes existed —
    /// same sides, same offsets, same length, for every shape that shipped.
    #[test]
    fn an_odd_size_is_unchanged_by_the_anchor_becoming_a_decision() {
        for size in [[1, 1, 1], [3, 5, 7], [9, 3, 1], [11, 11, 11]] {
            for shape in [ElementShape::Box, ElementShape::Ellipsoid] {
                let element = StructuringElement::from_size(shape, size).unwrap();
                let radius = [size[0] / 2, size[1] / 2, size[2] / 2];
                assert_eq!(element, StructuringElement::from_radius(shape, radius));
                assert!(element.is_symmetric());
                assert_eq!(element.reach_spec(), Reach::from(radius));
            }
        }
    }

    #[test]
    fn an_ellipsoid_is_inscribed_in_the_box_and_keeps_its_poles() {
        let element = StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 2, 2]);
        assert!(element.len() < 125, "an ellipsoid is smaller than its box");
        // the six poles are members, so the reach is still the radius
        for pole in [
            [2, 0, 0],
            [-2, 0, 0],
            [0, 2, 0],
            [0, -2, 0],
            [0, 0, 2],
            [0, 0, -2],
        ] {
            assert!(element.offsets().contains(&pole), "missing pole {pole:?}");
        }
        assert!(!element.offsets().contains(&[2, 2, 2]));
        assert_eq!(element.reach(2), 2);
    }

    /// The second ball rule, against the three things that make it a different
    /// set from the inscribed one. Its agreement with the implementation it was
    /// derived from, size by size and voxel by voxel, is
    /// `tests/element_reference_rule.rs`; this pins the properties a reader
    /// needs in order to choose between the two.
    #[test]
    fn the_extent_ellipsoid_normalises_by_the_half_extent_rounded_up() {
        // odd: normalised by r + 1 = 6, so `d^2 < 36` rather than `d^2 <= 25`
        let wide =
            StructuringElement::from_size(ElementShape::ExtentEllipsoid, [11, 11, 11]).unwrap();
        let inscribed =
            StructuringElement::from_size(ElementShape::Ellipsoid, [11, 11, 11]).unwrap();
        assert!(wide.len() > inscribed.len());
        for offset in wide.offsets() {
            let squared: isize = offset.iter().map(|d| d * d).sum();
            assert!(squared < 36, "{offset:?} has d^2 = {squared}");
        }
        for offset in [[3, 3, 3], [5, 3, 0], [0, 4, 4]] {
            assert!(wide.offsets().contains(&offset), "{offset:?} is inside 36");
            assert!(!inscribed.offsets().contains(&offset));
        }
        // strictly open: the corner at exactly 36 is out
        assert!(!wide.offsets().contains(&[4, 4, 2]));

        // even: normalised by n / 2 = 5, measured from the box's continuous
        // centre, so the whole 10-box is inside on one axis
        let even =
            StructuringElement::from_size(ElementShape::ExtentEllipsoid, [10, 1, 1]).unwrap();
        assert_eq!(even.len(), 10, "|(o + 0.5) / 5| <= 0.9 for every o");
        assert_eq!(even.sides(0), (5, 4));

        // and the set is symmetric under reflection of the *box*, which is what
        // the half-voxel shift is for: offset o and offset -1 - o agree.
        let box_even =
            StructuringElement::from_size(ElementShape::ExtentEllipsoid, [10, 10, 1]).unwrap();
        for offset in box_even.offsets() {
            let mirrored = [-1 - offset[0], -1 - offset[1], offset[2]];
            assert!(
                box_even.offsets().contains(&mirrored),
                "{offset:?} is in but its mirror {mirrored:?} is not"
            );
        }
    }

    /// The inscribed ellipsoid over an off-centre axis still touches both faces
    /// of its box, which is what keeps `reach` exact rather than optimistic.
    #[test]
    fn an_off_centre_ellipsoid_keeps_the_poles_on_both_sides() {
        let element = StructuringElement::from_size(ElementShape::Ellipsoid, [10, 6, 1]).unwrap();
        assert_eq!(element.sides(0), (5, 4));
        assert_eq!(element.sides(1), (3, 2));
        for pole in [[-5, 0, 0], [4, 0, 0], [0, -3, 0], [0, 2, 0]] {
            assert!(element.offsets().contains(&pole), "missing pole {pole:?}");
        }
        assert!(!element.offsets().contains(&[-5, -3, 0]));
        // so the reach on the wider side is real, and the pair is the truth
        assert_eq!(element.reach_spec().at(0, 0, 64), (5, 4));
    }

    #[test]
    fn a_flat_axis_admits_only_offset_zero_on_it() {
        let element = StructuringElement::from_radius(ElementShape::Ellipsoid, [0, 2, 2]);
        assert!(element.offsets().iter().all(|offset| offset[0] == 0));
        assert_eq!(element.reach(0), 0);
    }

    #[test]
    fn offsets_are_in_ascending_lexicographic_order() {
        let element = StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 3, 1]);
        let mut sorted = element.offsets().to_vec();
        sorted.sort();
        assert_eq!(element.offsets(), sorted.as_slice());
    }

    /// A truncated window must keep the *relative* position of the rank, or
    /// every face of a volume gets a different filter from its interior.
    #[test]
    fn a_truncated_window_rescales_the_rank_rather_than_clamping_it() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let median = Rank::median(&element);
        assert_eq!(median, Rank::Nth(13));
        // full window: the middle of 27
        assert_eq!(median.resolve(27, 27), 13);
        // a face: 18 values, and the median is still the middle of them
        assert_eq!(median.resolve(27, 18), 9);
        // an edge: 12 values
        assert_eq!(median.resolve(27, 12), 6);
        // a corner: 8 values
        assert_eq!(median.resolve(27, 8), 4);
        // the extremes stay extreme wherever they are evaluated
        assert_eq!(Rank::lowest().resolve(27, 8), 0);
        assert_eq!(Rank::highest(&element).resolve(27, 8), 7);
    }

    /// A percentile names a rank; it does not add one.
    #[test]
    fn a_percentile_is_a_name_for_a_rank_and_agrees_with_the_median_at_a_half() {
        for radius in [[1, 1, 1], [2, 0, 3], [0, 0, 0]] {
            for shape in [ElementShape::Box, ElementShape::Ellipsoid] {
                let element = StructuringElement::from_radius(shape, radius);
                assert_eq!(Rank::percentile(&element, 0.5), Rank::median(&element));
                assert_eq!(Rank::percentile(&element, 0.0), Rank::lowest());
                assert_eq!(Rank::percentile(&element, 1.0), Rank::highest(&element));
                // out of range is clamped rather than wrapped or refused
                assert_eq!(Rank::percentile(&element, -3.0), Rank::lowest());
                assert_eq!(Rank::percentile(&element, 9.0), Rank::highest(&element));
            }
        }
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        assert_eq!(Rank::percentile(&element, 0.25), Rank::Nth(7));
    }

    /// A step subsamples the **offsets**, and everything derived from the
    /// element follows the survivors rather than the box that was asked for.
    #[test]
    fn a_step_decimates_the_offsets_and_the_reach_follows_the_survivors() {
        let plain = StructuringElement::from_size(ElementShape::Box, [200, 200, 1]).unwrap();
        assert_eq!(plain.len(), 200 * 200);
        assert_eq!(plain.sides(0), (100, 99));
        assert_eq!(plain.step(), [1, 1, 1]);

        let stepped =
            StructuringElement::from_size_stepped(ElementShape::Box, [200, 200, 1], [2, 2, 1])
                .unwrap();
        assert_eq!(stepped.step(), [2, 2, 1]);
        // a quarter of the members, which is what every cost and every rank
        // built from `len` will now see
        assert_eq!(stepped.len(), 100 * 100);
        assert_eq!(Rank::median(&stepped), Rank::Nth(5000));

        // the surviving offsets are the low-anchored lattice, and the last one
        // above the anchor is 98 rather than the 99 the box reaches
        let along: Vec<isize> = stepped
            .offsets()
            .iter()
            .filter(|offset| offset[1] == -100 && offset[2] == 0)
            .map(|offset| offset[0])
            .collect();
        assert_eq!(along.first(), Some(&-100));
        assert_eq!(along.last(), Some(&98));
        assert!(along.iter().all(|offset| offset % 2 == 0));

        // and the reach is that widest surviving offset, on both sides, with
        // the bounding box saying the same thing
        assert_eq!(stepped.sides(0), (100, 98));
        assert_eq!(stepped.sides(1), (100, 98));
        assert_eq!(stepped.sides(2), (0, 0));
        assert_eq!(stepped.reach(0), 100);
        assert_eq!(stepped.size(), [199, 199, 1]);
        assert_eq!(stepped.reach_spec().at(0, 0, 1000), (100, 98));

        // a step of one is exactly the unstepped element, not a near copy
        assert_eq!(
            StructuringElement::from_size_stepped(ElementShape::Box, [200, 200, 1], [1, 1, 1])
                .unwrap(),
            plain
        );

        // a step of zero names no offsets and is refused rather than emptied
        let err = StructuringElement::from_size_stepped(ElementShape::Box, [7, 7, 7], [2, 0, 1])
            .unwrap_err();
        assert!(err.to_string().contains("non-zero"), "got: {err}");

        // and a step coarser than the shape it decimates is refused too: a
        // radius-3 ball stepped by nine keeps only `[-3, -3, -3]`, which is
        // outside its own surface. A box of the same size survives, so this is
        // a fact about the shape and not about the step alone.
        let err =
            StructuringElement::from_size_stepped(ElementShape::Ellipsoid, [7, 7, 7], [9, 9, 9])
                .unwrap_err();
        assert!(err.to_string().contains("leaves no offset"), "got: {err}");
        assert_eq!(
            StructuringElement::from_size_stepped(ElementShape::Box, [7, 7, 7], [9, 9, 9])
                .unwrap()
                .len(),
            1
        );
    }

    /// The reach is read off the offsets, so a step that strands the far pole
    /// shortens it — checked against a brute-force scan of the offsets so that
    /// the assertion is not the implementation restated.
    #[test]
    fn the_reach_is_the_widest_surviving_offset_whatever_the_step() {
        for shape in [
            ElementShape::Box,
            ElementShape::Ellipsoid,
            ElementShape::ExtentEllipsoid,
        ] {
            for size in [[7usize, 7, 7], [9, 4, 1], [10, 6, 3], [13, 13, 5]] {
                for step in [[1usize, 1, 1], [2, 2, 1], [3, 1, 2], [4, 5, 3], [9, 9, 9]] {
                    // A step coarser than the shape empties it, which is
                    // refused; the sweep includes such a case on purpose and
                    // the refusal is the assertion for it.
                    let element = match StructuringElement::from_size_stepped(shape, size, step) {
                        Ok(element) => element,
                        Err(err) => {
                            assert!(
                                err.to_string().contains("leaves no offset"),
                                "{shape:?} {size:?} {step:?}: {err}"
                            );
                            continue;
                        }
                    };
                    for axis in 0..3 {
                        let widest_below = element
                            .offsets()
                            .iter()
                            .map(|offset| (-offset[axis]).max(0) as usize)
                            .max()
                            .unwrap();
                        let widest_above = element
                            .offsets()
                            .iter()
                            .map(|offset| offset[axis].max(0) as usize)
                            .max()
                            .unwrap();
                        assert_eq!(
                            element.sides(axis),
                            (widest_below, widest_above),
                            "{shape:?} size {size:?} step {step:?} axis {axis}"
                        );
                        assert_eq!(element.reach(axis), widest_below.max(widest_above));
                    }
                    // and the count is the count, which is what a cost and a
                    // median are both built from
                    assert_eq!(element.len(), element.offsets().len());
                }
            }
        }
    }

    /// The shape's surface is tested against the true offset, so a stepped
    /// element is a subset of the unstepped one and never a redrawn shape.
    #[test]
    fn a_stepped_element_is_a_subset_of_the_element_it_decimates() {
        for shape in [
            ElementShape::Box,
            ElementShape::Ellipsoid,
            ElementShape::ExtentEllipsoid,
        ] {
            for size in [[11usize, 11, 11], [10, 6, 4], [15, 15, 1]] {
                let whole = StructuringElement::from_size(shape, size).unwrap();
                for step in [[2usize, 2, 1], [3, 2, 1], [2, 1, 2]] {
                    let stepped = StructuringElement::from_size_stepped(shape, size, step).unwrap();
                    for offset in stepped.offsets() {
                        assert!(
                            whole.offsets().contains(offset),
                            "{shape:?} {size:?} step {step:?}: {offset:?} is not in the \
                             element it decimates"
                        );
                    }
                    assert!(stepped.len() <= whole.len());
                    // every surviving offset is on the low-anchored lattice
                    for offset in stepped.offsets() {
                        for axis in 0..3 {
                            let from_low = offset[axis] + whole.sides(axis).0 as isize;
                            assert_eq!(from_low % step[axis] as isize, 0, "{offset:?}");
                        }
                    }
                }
            }
        }
    }

    /// The two truncation conventions, side by side: **equal at a half over an
    /// odd surviving population, and unequal everywhere else**.
    ///
    /// Both halves are asserted, because either one alone is a test that passes
    /// for the wrong reason — the equivalence alone would be satisfied by two
    /// conventions that were secretly the same rule, and the divergence alone
    /// would be satisfied by one that was simply wrong at a half too.
    #[test]
    fn the_two_truncation_conventions_agree_at_a_half_over_an_odd_population() {
        let half = Rank::ceiling_percentile(0.5).unwrap();
        for full in [8usize, 27, 26, 125, 100] {
            let proportional = Rank::Nth(full / 2);
            for available in 1..=full {
                let ours = proportional.resolve(full, available);
                let theirs = half.resolve(full, available);
                // the proportional rule at a half is `floor(m / 2)`, for every
                // `m` and every `n` — this is the identity that made a median
                // filter agree with a population-based one at every face
                assert_eq!(
                    ours,
                    available / 2,
                    "proportional half: full {full}, available {available}"
                );
                // the ceiling rule is `ceil(m / 2) - 1`
                assert_eq!(theirs, available.div_ceil(2) - 1);
                if available % 2 == 1 {
                    assert_eq!(ours, theirs, "odd {available} must agree");
                } else {
                    assert_eq!(
                        ours,
                        theirs + 1,
                        "even {available} must differ by exactly one"
                    );
                }
            }
        }
    }

    /// Away from a half the two diverge on the untruncated window as well,
    /// which is the part that makes them different *filters* and not merely
    /// different boundary treatments.
    #[test]
    fn the_two_conventions_differ_away_from_a_half_and_at_full_population() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        assert_eq!(element.len(), 27);

        let proportional = Rank::percentile(&element, 0.25);
        let ceiling = Rank::ceiling_percentile(0.25).unwrap();
        assert_eq!(proportional, Rank::Nth(7), "round(0.25 * 26) = 7");
        // the full window, unclamped: 7 against 6
        assert_eq!(proportional.resolve(27, 27), 7);
        assert_eq!(ceiling.resolve(27, 27), 6, "ceil(0.25 * 27) - 1 = 6");

        // a face, an edge and a corner of a 3x3x3 window
        for (available, ours, theirs) in [(18usize, 5usize, 4usize), (12, 3, 2), (8, 2, 1)] {
            assert_eq!(proportional.resolve(27, available), ours, "m = {available}");
            assert_eq!(ceiling.resolve(27, available), theirs, "m = {available}");
        }

        // the extremes are the extremes under both, which is what keeps an
        // erosion an erosion however it is named
        assert_eq!(Rank::ceiling_percentile(1.0).unwrap().resolve(27, 8), 7);
        assert_eq!(Rank::highest(&element).resolve(27, 8), 7);
        // `p = 0` is the minimum here, not a histogram's first bin
        assert_eq!(Rank::ceiling_percentile(0.0).unwrap().resolve(27, 8), 0);
        assert_eq!(Rank::lowest().resolve(27, 8), 0);
    }

    /// The fraction is validated once, so that `Rank` can stay `Eq + Hash`.
    #[test]
    fn a_percentile_refuses_what_it_could_not_hash() {
        assert!(Percentile::new(f64::NAN).is_err());
        assert!(Percentile::new(1.5).is_err());
        assert!(Percentile::new(-0.5).is_err());
        assert!(Rank::ceiling_percentile(2.0).is_err());
        // `-0.0` is accepted and normalised, so equality and hashing agree
        assert_eq!(
            Percentile::new(-0.0).unwrap(),
            Percentile::new(0.0).unwrap()
        );
        let mut set = std::collections::HashSet::new();
        set.insert(Rank::ceiling_percentile(-0.0).unwrap());
        assert!(set.contains(&Rank::ceiling_percentile(0.0).unwrap()));
        assert!(!set.contains(&Rank::ceiling_percentile(0.5).unwrap()));
    }

    #[test]
    fn selection_returns_the_value_and_not_a_recomputation_of_it() {
        let mut values = vec![5, 1, 9, 3, 7];
        assert_eq!(select_nth(&mut values, 0), Some(1));
        let mut values = vec![5, 1, 9, 3, 7];
        assert_eq!(select_nth(&mut values, 2), Some(5));
        let mut values = vec![5, 1, 9, 3, 7];
        assert_eq!(select_nth(&mut values, 4), Some(9));
        let mut empty: Vec<i32> = Vec::new();
        assert_eq!(select_nth(&mut empty, 0), None);
    }

    #[test]
    fn the_total_order_is_total_where_the_partial_one_is_not() {
        let mut values = [
            Total(3.0),
            Total(f64::NAN),
            Total(-1.0),
            Total(f64::INFINITY),
        ];
        values.sort();
        assert_eq!(values[0].0, -1.0);
        assert_eq!(values[1].0, 3.0);
        assert_eq!(values[2].0, f64::INFINITY);
        assert!(values[3].0.is_nan());
    }
}
