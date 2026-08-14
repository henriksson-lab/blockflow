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
    shape: ElementShape,
    /// Voxels the element reads **below** the anchor, per axis.
    lo: [usize; 3],
    /// Voxels the element reads **above** the anchor, per axis.
    hi: [usize; 3],
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
        let offsets = generate(shape, lo, hi);
        Self {
            shape,
            lo,
            hi,
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
        Ok(Self::from_sides(shape, lo, hi))
    }

    pub fn shape(&self) -> ElementShape {
        self.shape
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

    /// The bounding box, per axis: `lo + hi + 1`.
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

fn generate(shape: ElementShape, lo: [usize; 3], hi: [usize; 3]) -> Vec<[isize; 3]> {
    let mut offsets = Vec::new();
    for a in -(lo[0] as isize)..=hi[0] as isize {
        for b in -(lo[1] as isize)..=hi[1] as isize {
            for c in -(lo[2] as isize)..=hi[2] as isize {
                if member(shape, lo, hi, [a, b, c]) {
                    offsets.push([a, b, c]);
                }
            }
        }
    }
    offsets
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

/// Which order statistic of a neighbourhood to select.
///
/// One variant, deliberately. **The median is the `k = n / 2` case, not a
/// separate implementation** — it is a constructor, and every rank goes through
/// the same selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rank {
    /// The `k`-th smallest value, counting from zero, of the **full** element's
    /// `n` values.
    ///
    /// Where the element is truncated — at a real volume boundary, where there
    /// is nothing beyond to read — only `m < n` values exist, and `k` is
    /// rescaled to the values that do: `round(k * (m - 1) / (n - 1))`. See
    /// [`Rank::resolve`] for why that is the right rule rather than clamping.
    Nth(usize),
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

    /// Where in the `available` values actually read this rank lands, given that
    /// the element holds `full` when nothing clamps it.
    ///
    /// **Why rescale rather than clamp.** Clamping `k` to `available - 1` would
    /// turn a median into a maximum wherever the element is truncated, so every
    /// face of the volume would get a systematically different filter from the
    /// interior. Rescaling keeps the *relative position* in the sorted window,
    /// which is what a rank means: the median of a half-window is still its
    /// median. The arithmetic is integer and rounds half up, so it is exactly
    /// reproducible.
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
