// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The lowest or highest value under a structuring element, by running
//! extremes along rows, over the whole buffer.
//!
//! The same filter as `ops::rank::RankFilterOp` at `Rank::lowest()` or
//! `Rank::highest(element)`, voxel for voxel: a selection under the element's
//! offsets, the window clipped to the buffer, which for every voxel the plan
//! trusts is the window clipped to the volume. `RankFilterOp` takes running
//! extremes only where the whole element fits in the buffer and gathers the
//! element tap by tap everywhere else; in a chain that "everywhere else" is the
//! halo of every block, and on the detector's blocks it was about 40% of a
//! whole run's CPU (the background opening and the window maxima). Here every
//! voxel takes the running path.
//!
//! A run is a maximal set of offsets contiguous along `x` at one `(z, y)`. Each
//! distinct `(first x, length)` is slid along every row of the buffer once
//! (van Herk / Gil-Werman, two passes and a combine, whatever the length), and
//! a voxel takes the extreme over its runs. A box is done axis by axis, three
//! slides in all. `f64` is ordered by `total_cmp`, the total order the rank
//! filter's float path uses, so the same bits are selected.

use std::cmp::Ordering;

use ndarray::{ArrayView3, ArrayViewMut3};

use blockflow::error::{Error, Result};
use blockflow::op::{Anchor, BlockOp, Slicing};
use blockflow::ops::element::{Rank, StructuringElement};
use blockflow::ops::rank::RankFilterOp;
use blockflow::reach::Reach;
use blockflow::voxels::Voxels;
use blockflow::{Chain, Dtype};

/// Which end of the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum End {
    Lowest,
    Highest,
}

/// The filter, or `RankFilterOp` where this one does not apply (a stepped
/// element, or one without its own centre, whose clipped window can be empty).
pub fn extreme_filter(name: &'static str, element: &StructuringElement, end: End) -> Chain {
    match ExtremeFilterOp::new(name, element, end) {
        Some(op) => Chain::op(op),
        None => {
            let rank = match end {
                End::Lowest => Rank::lowest(),
                End::Highest => Rank::highest(element),
            };
            Chain::op(RankFilterOp::new(name, element.clone(), rank))
        }
    }
}

/// See the module header.
#[derive(Debug, Clone)]
pub struct ExtremeFilterOp {
    name: &'static str,
    element: StructuringElement,
    end: End,
    /// `(dz, dy, x0, length)`.
    runs: Vec<(isize, isize, isize, usize)>,
    /// Per axis `(lo, hi)` offsets, when the element is the whole box.
    whole_box: Option<[(isize, isize); 3]>,
}

impl ExtremeFilterOp {
    pub fn new(name: &'static str, element: &StructuringElement, end: End) -> Option<Self> {
        let offsets = element.offsets();
        if element.step() != [1, 1, 1] || !offsets.contains(&[0, 0, 0]) {
            return None;
        }
        let mut runs: Vec<(isize, isize, isize, usize)> = Vec::new();
        for offset in offsets {
            match runs.last_mut() {
                Some((dz, dy, x0, length))
                    if *dz == offset[0]
                        && *dy == offset[1]
                        && *x0 + *length as isize == offset[2] =>
                {
                    *length += 1
                }
                _ => runs.push((offset[0], offset[1], offset[2], 1)),
            }
        }
        let bounds: [(isize, isize); 3] = std::array::from_fn(|axis| {
            let values = offsets.iter().map(|offset| offset[axis]);
            (values.clone().min().unwrap_or(0), values.max().unwrap_or(0))
        });
        let box_len: usize = bounds
            .iter()
            .map(|(lo, hi)| (hi - lo + 1) as usize)
            .product();
        let whole_box = (box_len == offsets.len()).then_some(bounds);
        Some(Self {
            name,
            element: element.clone(),
            end,
            runs,
            whole_box,
        })
    }
}

/// The comparisons the filter needs, in the order the rank filter selects by.
trait Ordered: Copy + Send + Sync + 'static {
    fn order(&self, other: &Self) -> Ordering;
}

macro_rules! ordered_int {
    ($($t:ty),*) => {$(
        impl Ordered for $t {
            fn order(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
        }
    )*};
}
ordered_int!(u8, u16, u32, u64, i8, i16, i32, i64);

impl Ordered for f64 {
    fn order(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

impl Ordered for f32 {
    fn order(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

impl Ordered for bool {
    fn order(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

/// The better of two values for `end`; ties keep `a`, and a tie is the same
/// bits, so which one is kept cannot be seen.
#[inline]
fn pick<T: Ordered>(end: End, a: T, b: T) -> T {
    match (end, b.order(&a)) {
        (End::Lowest, Ordering::Less) | (End::Highest, Ordering::Greater) => b,
        _ => a,
    }
}

/// Along an axis of `count` steps, each step `width` contiguous elements of
/// `data`: step `i` of `out` is the elementwise extreme over the steps
/// `[i + x0, i + x0 + length)` clipped to `[0, count)`. Steps whose window is
/// empty are left alone; the range of the others is returned. `prefix` and
/// `suffix` are scratch.
///
/// van Herk / Gil-Werman: blocks of `length` steps from 0, the running extreme
/// from each block's start (`prefix`) and to its end (`suffix`), and a window,
/// which spans at most two blocks, is `suffix[a]` against `prefix[b]`. A window
/// clipped by the axis's ends lies in one block and starts at its start or ends
/// at the axis's end, so one of the two is exact there.
#[allow(clippy::too_many_arguments)]
fn slide_lines<T: Ordered>(
    end: End,
    data: &[T],
    count: usize,
    width: usize,
    x0: isize,
    length: usize,
    prefix: &mut Vec<T>,
    suffix: &mut Vec<T>,
    out: &mut [T],
) -> std::ops::Range<usize> {
    debug_assert_eq!(data.len(), count * width);
    if width == 1 {
        return slide_row(end, data, x0, length, prefix, suffix, out);
    }
    prefix.clear();
    prefix.extend_from_slice(data);
    suffix.clear();
    suffix.extend_from_slice(data);
    let mut start = 0;
    while start < count {
        let stop = (start + length).min(count);
        for i in start + 1..stop {
            let (done, rest) = prefix.split_at_mut(i * width);
            let previous = &done[(i - 1) * width..];
            for (slot, &before) in rest[..width].iter_mut().zip(previous) {
                *slot = pick(end, before, *slot);
            }
        }
        for i in (start..stop - 1).rev() {
            let (here, after) = suffix.split_at_mut((i + 1) * width);
            let here = &mut here[i * width..];
            for (slot, &later) in here.iter_mut().zip(&after[..width]) {
                *slot = pick(end, later, *slot);
            }
        }
        start = stop;
    }
    let length_i = length as isize;
    let count_i = count as isize;
    // Steps whose window meets the axis.
    let first = (-(x0 + length_i - 1)).clamp(0, count_i) as usize;
    let last = (count_i - x0).clamp(0, count_i) as usize;
    for i in first..last {
        let a = (i as isize + x0).max(0) as usize;
        let b = (i as isize + x0 + length_i - 1).min(count_i - 1) as usize;
        let target = &mut out[i * width..(i + 1) * width];
        let full = b - a + 1 == length;
        if full || a / length != b / length {
            let (from_a, from_b) = (&suffix[a * width..], &prefix[b * width..]);
            for ((slot, &x), &y) in target.iter_mut().zip(from_a).zip(&from_b[..width]) {
                *slot = pick(end, x, y);
            }
        } else if a.is_multiple_of(length) {
            target.copy_from_slice(&prefix[b * width..(b + 1) * width]);
        } else {
            target.copy_from_slice(&suffix[a * width..(a + 1) * width]);
        }
    }
    first..last
}

/// [`slide_lines`] along one row (`width == 1`), as plain scalar loops.
fn slide_row<T: Ordered>(
    end: End,
    row: &[T],
    x0: isize,
    length: usize,
    prefix: &mut Vec<T>,
    suffix: &mut Vec<T>,
    out: &mut [T],
) -> std::ops::Range<usize> {
    let n = row.len();
    prefix.clear();
    prefix.extend_from_slice(row);
    suffix.clear();
    suffix.extend_from_slice(row);
    for block in prefix.chunks_mut(length) {
        for i in 1..block.len() {
            block[i] = pick(end, block[i - 1], block[i]);
        }
    }
    for block in suffix.chunks_mut(length) {
        for i in (0..block.len() - 1).rev() {
            block[i] = pick(end, block[i + 1], block[i]);
        }
    }
    let (length_i, n_i) = (length as isize, n as isize);
    let first = (-(x0 + length_i - 1)).clamp(0, n_i) as usize;
    let last = (n_i - x0).clamp(0, n_i) as usize;
    // Where the whole window is inside the row: no clipping, no division.
    let inner_first = ((-x0).max(0) as usize).clamp(first, last);
    let inner_last = ((n_i - x0 - length_i + 1).max(0) as usize).clamp(inner_first, last);
    let clipped = |i: usize| {
        let a = (i as isize + x0).max(0) as usize;
        let b = (i as isize + x0 + length_i - 1).min(n_i - 1) as usize;
        if a / length != b / length {
            pick(end, suffix[a], prefix[b])
        } else if a.is_multiple_of(length) {
            prefix[b]
        } else {
            suffix[a]
        }
    };
    for i in first..inner_first {
        out[i] = clipped(i);
    }
    let shift = x0 + length_i - 1;
    for i in inner_first..inner_last {
        let a = (i as isize + x0) as usize;
        let b = (i as isize + shift) as usize;
        out[i] = pick(end, suffix[a], prefix[b]);
    }
    for i in inner_last..last {
        out[i] = clipped(i);
    }
    first..last
}

/// The filter of `input` into `out` (the same shape).
fn filter_into<T: Ordered>(
    op: &ExtremeFilterOp,
    input: ArrayView3<'_, T>,
    mut out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    if out.shape() != input.shape() {
        return Err(Error::invalid(format!(
            "{}: the output is {:?} and the input {shape:?}",
            op.name,
            out.shape()
        )));
    }
    let end = op.end;
    let contiguous = input.as_standard_layout();
    let data = contiguous
        .as_slice()
        .expect("a standard-layout array is one slice");
    let (n, plane) = (shape[2], shape[1] * shape[2]);
    let (mut prefix, mut suffix) = (Vec::new(), Vec::new());

    if let Some(bounds) = op.whole_box {
        // Axis by axis: a clipped box is the product of clipped intervals, and
        // the centre is in every one of them, so no window is empty.
        let mut current = data.to_vec();
        let mut next = data.to_vec();
        for axis in [2, 1, 0] {
            let (lo, hi) = bounds[axis];
            let length = (hi - lo + 1) as usize;
            if length == 1 {
                continue;
            }
            match axis {
                2 => {
                    for (row, target) in current.chunks_exact(n).zip(next.chunks_exact_mut(n)) {
                        slide_lines(end, row, n, 1, lo, length, &mut prefix, &mut suffix, target);
                    }
                }
                1 => {
                    for (slab, target) in current
                        .chunks_exact(plane)
                        .zip(next.chunks_exact_mut(plane))
                    {
                        let rows = shape[1];
                        slide_lines(
                            end,
                            slab,
                            rows,
                            n,
                            lo,
                            length,
                            &mut prefix,
                            &mut suffix,
                            target,
                        );
                    }
                }
                _ => {
                    let planes = shape[0];
                    slide_lines(
                        end,
                        &current,
                        planes,
                        plane,
                        lo,
                        length,
                        &mut prefix,
                        &mut suffix,
                        &mut next,
                    );
                }
            }
            std::mem::swap(&mut current, &mut next);
        }
        for (slot, value) in out.iter_mut().zip(current) {
            *slot = value;
        }
        return Ok(());
    }

    // Every distinct `(x0, length)` slid along every row once.
    let mut kinds: Vec<(isize, usize)> = op.runs.iter().map(|r| (r.2, r.3)).collect();
    kinds.sort_unstable();
    kinds.dedup();
    let mut slid: Vec<(Vec<T>, std::ops::Range<usize>)> = Vec::with_capacity(kinds.len());
    for &(x0, length) in &kinds {
        let mut all = data.to_vec();
        let mut range = 0..0;
        for (row, target) in data.chunks_exact(n).zip(all.chunks_exact_mut(n)) {
            range = slide_lines(end, row, n, 1, x0, length, &mut prefix, &mut suffix, target);
        }
        slid.push((all, range));
    }
    // The centre's run first: its window holds the voxel itself, so it
    // covers every voxel and the other runs only improve on it.
    let mut runs: Vec<(isize, isize, usize)> = op
        .runs
        .iter()
        .map(|&(dz, dy, x0, length)| {
            let kind = kinds
                .binary_search(&(x0, length))
                .expect("every run's kind was collected");
            (dz, dy, kind)
        })
        .collect();
    let centre = op
        .runs
        .iter()
        .position(|&(dz, dy, x0, length)| dz == 0 && dy == 0 && x0 <= 0 && x0 + length as isize > 0)
        .expect("the element holds its centre");
    runs.swap(0, centre);
    let mut result = vec![data[0]; data.len()];
    for z in 0..shape[0] {
        for y in 0..shape[1] {
            let at = (z * shape[1] + y) * n;
            let target = &mut result[at..at + n];
            for (index, &(dz, dy, kind)) in runs.iter().enumerate() {
                let (zz, yy) = (z as isize + dz, y as isize + dy);
                if zz < 0 || yy < 0 || zz >= shape[0] as isize || yy >= shape[1] as isize {
                    continue;
                }
                let from = (zz as usize * shape[1] + yy as usize) * n;
                let (values, range) = &slid[kind];
                let source = &values[from..from + n];
                if index == 0 {
                    target.copy_from_slice(source);
                } else {
                    let range = range.clone();
                    for (slot, &value) in target[range.clone()].iter_mut().zip(&source[range]) {
                        *slot = pick(end, *slot, value);
                    }
                }
            }
        }
    }
    for (slot, value) in out.iter_mut().zip(result) {
        *slot = value;
    }
    Ok(())
}

impl BlockOp for ExtremeFilterOp {
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.element.reach(axis)
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        self.element.reach_spec()
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        macro_rules! typed {
            ($t:ty) => {
                filter_into::<$t>(self, input.view::<$t>()?, out.view_mut::<$t>()?)
            };
        }
        match input.dtype() {
            Dtype::Bool => typed!(bool),
            Dtype::U8 => typed!(u8),
            Dtype::U16 => typed!(u16),
            Dtype::U32 => typed!(u32),
            Dtype::U64 => typed!(u64),
            Dtype::I8 => typed!(i8),
            Dtype::I16 => typed!(i16),
            Dtype::I32 => typed!(i32),
            Dtype::I64 => typed!(i64),
            Dtype::F32 => typed!(f32),
            Dtype::F64 => typed!(f64),
            Dtype::F16 => Err(Error::invalid(format!(
                "{}: no buffer holds half-precision",
                self.name
            ))),
        }
    }

    /// A selection of values that were read: a uniform buffer maps to itself.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blockflow::ops::element::ElementShape;
    use ndarray::Array3;

    fn hash(i: usize, j: usize, k: usize, salt: u64) -> u64 {
        let mut x = (i as u64).wrapping_mul(73_856_093)
            ^ (j as u64).wrapping_mul(19_349_663)
            ^ (k as u64).wrapping_mul(83_492_791)
            ^ salt.wrapping_mul(0x9e37_79b9);
        x ^= x >> 17;
        x = x.wrapping_mul(0xed5a_d4bb);
        x ^ (x >> 11)
    }

    fn elements() -> Vec<StructuringElement> {
        let mut all = Vec::new();
        for size in [[1, 10, 10], [1, 9, 9], [1, 4, 7], [3, 4, 4], [1, 1, 1]] {
            for shape in [ElementShape::Ellipsoid, ElementShape::Box] {
                all.push(StructuringElement::from_size(shape, size).unwrap());
            }
        }
        all.push(StructuringElement::from_radius(
            ElementShape::Box,
            [2, 2, 2],
        ));
        all.push(StructuringElement::from_radius(
            ElementShape::Ellipsoid,
            [1, 3, 2],
        ));
        all
    }

    /// Voxel for voxel the rank filter's answer, at every element, end, type
    /// and anchor tried, including off-centre (even) elements and buffers
    /// narrower than the element.
    #[test]
    fn the_running_filter_is_the_rank_filter() {
        let mut checked = 0;
        for (e, element) in elements().into_iter().enumerate() {
            for end in [End::Lowest, End::Highest] {
                let op = ExtremeFilterOp::new("extreme", &element, end).expect("a plain element");
                let rank = match end {
                    End::Lowest => Rank::lowest(),
                    End::Highest => Rank::highest(&element),
                };
                let reference = RankFilterOp::new("rank", element.clone(), rank);
                for shape in [[5, 23, 31], [2, 3, 4], [1, 17, 6], [7, 40, 41]] {
                    let at = Anchor::new([1, 2, 3], [shape[0] + 4, shape[1] + 5, shape[2] + 6]);
                    let narrow = Array3::from_shape_fn(shape, |(i, j, k)| {
                        (hash(i, j, k, e as u64) % 40) as u16
                    });
                    let wide = Array3::from_shape_fn(shape, |(i, j, k)| {
                        (hash(i, j, k, e as u64 + 7) % 1000) as f64 / 7.0 - 50.0
                    });
                    for input in [Voxels::from(narrow), Voxels::from(wide)] {
                        let mut ours = Voxels::zeros(input.dtype(), shape).unwrap();
                        let mut theirs = Voxels::zeros(input.dtype(), shape).unwrap();
                        op.apply(&input, &mut ours, &at).unwrap();
                        reference.apply(&input, &mut theirs, &at).unwrap();
                        assert_eq!(
                            ours,
                            theirs,
                            "element {e} ({:?}), {end:?}, {shape:?}, {:?}",
                            element.size(),
                            input.dtype()
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 50);
    }

    #[test]
    fn a_stepped_element_is_left_to_the_rank_filter() {
        let stepped =
            StructuringElement::from_size_stepped(ElementShape::Box, [1, 5, 5], [1, 2, 2]).unwrap();
        assert!(ExtremeFilterOp::new("extreme", &stepped, End::Lowest).is_none());
    }
}
