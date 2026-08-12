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
// **The reach is the radius, and nothing configures it.** `reach` is derived
// from the element; there is no field to set it to something else. A caller who
// wants a wider halo sets the halo, which is a hint. That asymmetry is the whole
// design: `docs/design/BLOCK_OPS.md` is explicit that a reach fed by the
// configured halo makes the guard compare a number against itself.

use crate::error::{Error, Result};

/// Which voxels of the bounding box belong to the neighbourhood.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementShape {
    /// Every voxel of the bounding box: a separable rectangular window.
    Box,
    /// The voxels inside the axis-aligned ellipsoid inscribed in the bounding
    /// box, i.e. `sum (d_a / r_a)^2 <= 1`. An axis with radius zero admits only
    /// offset zero on that axis, so a flat element is a lower-dimensional
    /// ellipse rather than an empty set.
    Ellipsoid,
}

/// A neighbourhood, centred on the voxel it is evaluated at.
///
/// Parameterised by shape and per-axis radius, and by nothing else — there is no
/// default size anywhere in this crate, because a filter size is a property of
/// the images being processed and therefore of the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuringElement {
    shape: ElementShape,
    radius: [usize; 3],
    offsets: Vec<[isize; 3]>,
}

impl StructuringElement {
    /// From a per-axis radius. The element spans `2 * radius + 1` on each axis.
    pub fn from_radius(shape: ElementShape, radius: [usize; 3]) -> Self {
        let offsets = generate(shape, radius);
        Self {
            shape,
            radius,
            offsets,
        }
    }

    /// From a per-axis size, which must be **odd** on every axis.
    ///
    /// An even size has no centre voxel, so "the neighbourhood of this voxel"
    /// would have to pick a side, and the two choices differ by a shift of the
    /// whole output. Rather than choose silently, refuse: a caller who wants the
    /// shifted window can say which radius they mean.
    pub fn from_size(shape: ElementShape, size: [usize; 3]) -> Result<Self> {
        let mut radius = [0usize; 3];
        for axis in 0..3 {
            if size[axis] == 0 || size[axis].is_multiple_of(2) {
                return Err(Error::InvalidArgument(format!(
                    "a centred element needs an odd size on every axis; got {size:?}"
                )));
            }
            radius[axis] = size[axis] / 2;
        }
        Ok(Self::from_radius(shape, radius))
    }

    pub fn shape(&self) -> ElementShape {
        self.shape
    }

    pub fn radius(&self) -> [usize; 3] {
        self.radius
    }

    /// Voxels read beyond the centre, along `axis`. **This is what every op in
    /// this module derives its `reach` from.**
    ///
    /// It is the radius even for an ellipsoid, because the ellipsoid still
    /// contains the two poles on each axis. A shape that excluded them would
    /// report less, which is why this asks the shape rather than assuming.
    pub fn reach(&self, axis: usize) -> usize {
        self.radius[axis]
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

fn generate(shape: ElementShape, radius: [usize; 3]) -> Vec<[isize; 3]> {
    let bound = [radius[0] as isize, radius[1] as isize, radius[2] as isize];
    let mut offsets = Vec::new();
    for a in -bound[0]..=bound[0] {
        for b in -bound[1]..=bound[1] {
            for c in -bound[2]..=bound[2] {
                if member(shape, radius, [a, b, c]) {
                    offsets.push([a, b, c]);
                }
            }
        }
    }
    offsets
}

fn member(shape: ElementShape, radius: [usize; 3], offset: [isize; 3]) -> bool {
    match shape {
        ElementShape::Box => true,
        ElementShape::Ellipsoid => {
            let mut total = 0.0_f64;
            for axis in 0..3 {
                if radius[axis] == 0 {
                    if offset[axis] != 0 {
                        return false;
                    }
                    continue;
                }
                let scaled = offset[axis] as f64 / radius[axis] as f64;
                total += scaled * scaled;
            }
            total <= 1.0
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

    #[test]
    fn an_even_size_is_refused_rather_than_rounded() {
        let err = StructuringElement::from_size(ElementShape::Box, [4, 3, 3]).unwrap_err();
        assert!(err.to_string().contains("odd size"), "got: {err}");
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
