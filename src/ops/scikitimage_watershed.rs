// SPDX-License-Identifier: BSD-3-Clause
//
// DERIVED FROM SCIKIT-IMAGE — BSD-3-Clause.
// Copyright (C) 2019, the scikit-image team. All rights reserved.
//
// This file is a translation of the sources listed below and carries **their**
// licence, not the rest of this crate's. The crate is MIT and BSD-3-Clause is
// compatible with it, but the notice below has to travel with the code, so this
// material lives in a file of its own: do not paste it into an MIT-headed
// module, and do not paste MIT-headed code in here.
//
// Original sources:
// - skimage/segmentation/_watershed.py      (padding, marker masking, neighbour order)
// - skimage/segmentation/_watershed_cy.pyx  (`watershed_raveled`, `_diff_neighbors`)
// - skimage/segmentation/heap_general.pxi   (the binary heap)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//  3. Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived from this
//     software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ---------------------------------------------------------------------------
//
// What this is, and why it is a translation rather than a reimplementation
// ------------------------------------------------------------------------
// A seeded watershed: flood a cost volume outward from labelled seeds, lowest
// cost first, and partition it into one basin per seed.
//
// Three details drive exact parity with `skimage.segmentation.watershed` and
// every one of them is reproduced deliberately, because each is a place where a
// plausible-looking priority flood gives a *different partition* rather than a
// slightly different boundary:
//
// * The priority queue is skimage's own binary heap, ordered by `(value, age)`
//   with a *strict* `smaller` comparison. Seeds all carry `age == 0`, so their
//   relative pop order is decided by the heap array's internal layout rather
//   than by a stable rule. Reimplementing the same sift-up and sift-down
//   therefore matters: the pop order of equal-valued seeds sets the `age` of
//   everything they push, and `age` breaks every later tie. **There is no way
//   to state this behaviour except as the algorithm that produces it**, which
//   is the whole reason this file is a port.
//
// * `_diff_neighbors` mutates `mask`, clearing voxels that sit between two
//   different labels. That is what carves the separating line, and later pops
//   observe the mutation — so the mask a voxel is tested against is not the
//   mask the flood started with.
//
// * A neighbour's priority is raised to its source's (`if value < elem_value`),
//   so a seed cannot win a basin merely by spilling into it one step early.
//
// skimage pads the volume by one and relies on the zero border of the padded
// mask to stop neighbour walks. This keeps the volume unpadded and bounds-checks
// instead, which is equivalent (an out-of-range neighbour is exactly a
// `mask == 0` neighbour) and avoids copying the whole block.

use crate::error::{Error, Result};

const MAX_NDIM: usize = 8;

#[derive(Clone, Copy)]
struct HeapItem {
    value: f64,
    /// The push counter, **`i32` because skimage's `Heapitem` is** — see
    /// [`AGE_LIMIT`] for what happens at the end of it and why this file
    /// refuses rather than widening the field.
    age: i32,
    index: usize,
    source: usize,
}

/// The largest push counter this port can represent, and therefore the largest
/// flood it will run.
///
/// **The field is `i32` because skimage's is.** `heap_general.pxi` declares
/// `DTYPE_INT32_t age` and `_watershed_cy.pyx` increments a `Py_ssize_t` into
/// it, so the original truncates here too — and, being C, truncates into a
/// *signed* type. Past this point `age` goes negative, `smaller` inverts on
/// every tie, and the flood stops being a watershed by anybody's definition:
/// a later push sorts *before* an earlier one, so a seed reaches a basin it
/// never should.
///
/// **So this refuses instead of widening, and that is the parity argument
/// rather than a limitation of the port.** Widening to `i64` would produce a
/// *better* algorithm than skimage's above the limit and a different one, which
/// is exactly what a file whose whole reason for existing is exact parity must
/// not do quietly — see this file's header. Above the limit there is no
/// behaviour to be in parity *with*: the original's is C signed overflow.
/// (It would even be free in memory: the struct pads to 32 bytes either way.
/// It is declined on the parity ground, not a cost one.)
///
/// **When this is reachable.** `age` counts pushes, not voxels, and the two
/// differ by the mode:
///
/// * `watershed_line = false` labels a voxel *at push time*, so no voxel is
///   ever pushed twice and the count is bounded by the voxels — 2.1 G of them,
///   about `1290^3`;
/// * `watershed_line = true` settles labels at pop, so a voxel can be pushed
///   once per neighbour: `2 * ndim` times, which for a 3-D volume is six. The
///   bound is then about `710^3`.
///
/// `super::watershed`'s own header contemplates volumes above both. The refusal
/// is what makes that a stated limit rather than a silently different
/// partition.
const AGE_LIMIT: i64 = i32::MAX as i64;

/// The bytes one queued item occupies, which is what the working-set arithmetic
/// in `super::watershed` is stated in. Asserted below rather than asserted in
/// prose, because the number is used to size a machine.
pub const HEAP_ITEM_BYTES: usize = std::mem::size_of::<HeapItem>();

#[inline(always)]
fn smaller(a: &HeapItem, b: &HeapItem) -> bool {
    if a.value != b.value {
        a.value < b.value
    } else {
        a.age < b.age
    }
}

/// skimage's binary heap: push writes at the end and sifts up, pop takes slot 0,
/// moves the last item to the root and sifts down.
struct Heap {
    data: Vec<HeapItem>,
    /// The largest length `data` ever reached. Not part of the algorithm — it is
    /// read by the test that turns the memory claim into a measurement.
    peak: usize,
}

impl Heap {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity.max(1000)),
            peak: 0,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    fn push(&mut self, item: HeapItem) {
        self.data.push(item);
        if self.data.len() > self.peak {
            self.peak = self.data.len();
        }
        let mut child = self.data.len() - 1;
        while child > 0 {
            let parent = child.div_ceil(2) - 1;
            if smaller(&self.data[child], &self.data[parent]) {
                self.data.swap(parent, child);
                child = parent;
            } else {
                break;
            }
        }
    }

    fn pop(&mut self) -> HeapItem {
        let dest = self.data[0];
        let items = self.data.len() - 1;
        if items == 0 {
            self.data.clear();
            return dest;
        }
        // Mirror the Cython: move the last element to the root, shrink, sift down.
        self.data.swap(0, items);
        self.data.truncate(items);

        let mut i = 0usize;
        loop {
            let l = i * 2 + 1;
            let r = i * 2 + 2;
            let mut smallest = i;
            if l < items {
                if smaller(&self.data[l], &self.data[i]) {
                    smallest = l;
                }
                if r < items && smaller(&self.data[r], &self.data[smallest]) {
                    smallest = r;
                }
            } else {
                break;
            }
            if smallest == i {
                break;
            }
            self.data.swap(i, smallest);
            i = smallest;
        }
        dest
    }
}

/// Neighbour offsets for `connectivity=1`, in skimage's order.
///
/// `_offsets_to_raveled_neighbors` sorts by Euclidean distance from the centre;
/// for connectivity 1 every offset is at distance 1 and the group comes out in
/// ascending raveled order, i.e. `-stride[0] .. -stride[n-1], +stride[n-1] .. +stride[0]`.
struct Neighbourhood {
    ndim: usize,
    shape: [usize; MAX_NDIM],
    strides: [usize; MAX_NDIM],
}

impl Neighbourhood {
    fn new(shape: &[usize]) -> Self {
        assert!(
            shape.len() <= MAX_NDIM,
            "watershed supports up to 8 dimensions"
        );
        let ndim = shape.len();
        let mut dims = [0usize; MAX_NDIM];
        let mut strides = [0usize; MAX_NDIM];
        let mut stride = 1usize;
        for axis in (0..ndim).rev() {
            dims[axis] = shape[axis];
            strides[axis] = stride;
            stride *= shape[axis];
        }
        Self {
            ndim,
            shape: dims,
            strides,
        }
    }

    #[inline(always)]
    fn coords(&self, index: usize) -> [usize; MAX_NDIM] {
        let mut coords = [0usize; MAX_NDIM];
        let mut rem = index;
        for axis in (0..self.ndim).rev() {
            coords[axis] = rem % self.shape[axis];
            rem /= self.shape[axis];
        }
        coords
    }

    /// Visits in-range neighbours of `index` in ascending raveled-offset order.
    #[inline(always)]
    fn for_each<F: FnMut(usize)>(&self, index: usize, mut visit: F) {
        let coords = self.coords(index);
        for axis in 0..self.ndim {
            if coords[axis] > 0 {
                visit(index - self.strides[axis]);
            }
        }
        for axis in (0..self.ndim).rev() {
            if coords[axis] + 1 < self.shape[axis] {
                visit(index + self.strides[axis]);
            }
        }
    }
}

/// `_diff_neighbors`: report (and record in `mask`) whether `index` borders more
/// than one distinct label, which makes it a separating-line voxel.
#[inline]
fn diff_neighbors(
    output: &[u32],
    mask: &mut [bool],
    nbrs: &Neighbourhood,
    index: usize,
    label: u32,
) -> bool {
    if !mask[index] {
        return true;
    }
    let mut differs = false;
    nbrs.for_each(index, |neighbour| {
        if differs {
            return;
        }
        if mask[neighbour] {
            let neighbour_label = output[neighbour];
            if neighbour_label != 0 && neighbour_label != label {
                differs = true;
            }
        }
    });
    if differs {
        mask[index] = false;
    }
    differs
}

/// Flood `output` from its nonzero entries, in place.
///
/// `image` is the cost volume; lower floods first. `mask` selects the floodable
/// region and is mutated to carve separating lines when `watershed_line` is set.
/// `output` must already hold the seed labels and is filled with basin labels.
pub fn watershed_raveled(
    image: &[f64],
    shape: &[usize],
    mask: &mut [bool],
    output: &mut [u32],
    watershed_line: bool,
) -> Result<()> {
    watershed_raveled_reporting_peak(image, shape, mask, output, watershed_line)?;
    Ok(())
}

/// [`watershed_raveled`], returning the largest number of items the queue ever
/// held. The figure is the whole of this op's non-dense memory cost and it is
/// only knowable by running, so it is returned rather than estimated.
pub fn watershed_raveled_reporting_peak(
    image: &[f64],
    shape: &[usize],
    mask: &mut [bool],
    output: &mut [u32],
    watershed_line: bool,
) -> Result<usize> {
    flood(image, shape, mask, output, watershed_line, AGE_LIMIT)
}

/// [`watershed_raveled_reporting_peak`] with the push limit stated.
///
/// The limit is a parameter for one reason: **so that the refusal can be
/// reached by a test.** Reaching it for real takes 2.1 billion pushes, which is
/// minutes of flooding and gigabytes of volume, so a suite that could only
/// trigger it honestly would not trigger it at all — and an untriggered guard
/// is a guard nobody knows is wired. Every caller outside this file passes
/// [`AGE_LIMIT`]; the test passes a small one and checks the same code path.
fn flood(
    image: &[f64],
    shape: &[usize],
    mask: &mut [bool],
    output: &mut [u32],
    watershed_line: bool,
    age_limit: i64,
) -> Result<usize> {
    debug_assert_eq!(image.len(), output.len());
    debug_assert_eq!(image.len(), mask.len());

    let nbrs = Neighbourhood::new(shape);
    let marker_locations = output
        .iter()
        .enumerate()
        .filter_map(|(index, &label)| (label != 0).then_some(index))
        .collect::<Vec<_>>();

    let mut heap = Heap::with_capacity(marker_locations.len() * 4);
    for &index in &marker_locations {
        heap.push(HeapItem {
            value: image[index],
            age: 0,
            index,
            source: index,
        });
    }

    let mut age: i64 = 1;
    while heap.len() > 0 {
        let elem = heap.pop();

        if watershed_line {
            // Labels are only settled at pop time: we can just observe that all
            // neighbours have been labelled once the voxel comes off the heap.
            if output[elem.index] != 0 && elem.index != elem.source {
                continue;
            }
            let source_label = output[elem.source];
            if !diff_neighbors(output, mask, &nbrs, elem.index, source_label) {
                output[elem.index] = source_label;
            }
        }

        let elem_label = output[elem.index];
        let elem_value = elem.value;
        let elem_index = elem.index;
        let elem_source = elem.source;
        let mut pending: [usize; 2 * MAX_NDIM] = [0; 2 * MAX_NDIM];
        let mut pending_len = 0usize;
        nbrs.for_each(elem_index, |neighbour| {
            pending[pending_len] = neighbour;
            pending_len += 1;
        });

        for &neighbour in &pending[..pending_len] {
            if !mask[neighbour] {
                // Includes basin boundaries, i.e. separating lines.
                continue;
            }
            if output[neighbour] != 0 {
                // Pre-labelled neighbour is not added to the queue.
                continue;
            }
            age += 1;
            if age > age_limit {
                // See `AGE_LIMIT`. One predictable comparison per push, against
                // a tie-break that inverts silently and a partition that is
                // then wrong everywhere two floods meet.
                return Err(Error::InvalidArgument(format!(
                    "watershed: the flood pushed more than {age_limit} voxels onto the queue. \
                     The push counter is the `i32` skimage's `Heapitem` declares, so beyond \
                     this it wraps negative, the `(value, age)` tie-break inverts, and the \
                     partition stops being a watershed. Flood a smaller region: this is \
                     {} voxels{}.",
                    image.len(),
                    if watershed_line {
                        ", and a separating line lets a voxel be queued once per neighbour \
                         rather than once"
                    } else {
                        ""
                    }
                )));
            }
            let mut value = image[neighbour];
            if !watershed_line {
                // Without separating lines a voxel can be labelled the moment it
                // is pushed: it cannot be reached at lower cost later.
                output[neighbour] = elem_label;
            }
            // A neighbour never costs less than the voxel it came from, so a
            // seed cannot win a basin merely by spilling into it one step early.
            if value < elem_value {
                value = elem_value;
            }
            heap.push(HeapItem {
                value,
                age: age as i32,
                index: neighbour,
                source: elem_source,
            });
        }
    }
    Ok(heap.peak)
}

/// The one checked-in case that separates this algorithm from a plausible
/// substitute, kept beside the code it pins rather than in a test file, because
/// the numbers are third-party reference output and their provenance is this
/// file's licence header.
///
/// A `5 x 5 x 3` random integer volume, three seeds, `mask = source > 2`,
/// `cost = -source`, `watershed_line = True`. Reference produced with
/// scikit-image 0.26.0:
///
/// ```python
/// skimage.segmentation.watershed(-source.astype(np.float64), seeds * mask,
///                                mask=source > 2, watershed_line=True)
/// ```
///
/// skimage assigns **32 / 9 / 10** voxels to the three seeds. A priority flood
/// that labels a voxel when it is *pushed*, keeps the neighbour's own cost as
/// its priority, and carves the line by zeroing an already-labelled neighbour —
/// which is what an independent reimplementation tends to produce, and which
/// agrees with this one on every smooth synthetic blob — assigns **25 / 9 / 13**.
pub mod reference_case {
    pub const SHAPE: [usize; 3] = [5, 5, 3];

    /// The threshold the mask is built at: `mask = source > THRESHOLD`.
    pub const THRESHOLD: f64 = 2.0;

    /// `(label, [z, y, x])` for each seed.
    pub const SEEDS: [(u32, [usize; 3]); 3] = [(1, [0, 0, 0]), (2, [4, 4, 2]), (3, [0, 4, 1])];

    /// Basin sizes skimage produces, in label order.
    pub const SIZES: [usize; 3] = [32, 9, 10];

    /// Basin sizes the plausible substitute produces. Held so a test can assert
    /// the two really are different, and therefore that the case is decisive.
    pub const NAIVE_SIZES: [usize; 3] = [25, 9, 13];

    #[rustfmt::skip]
    pub const SOURCE: [f64; 75] = [
        10.0, 7.0, 6.0, 3.0, 3.0, 0.0, 0.0, 0.0, 2.0, 9.0, 7.0, 10.0, 6.0, 7.0, 11.0,
        8.0, 7.0, 6.0, 6.0, 11.0, 3.0, 9.0, 8.0, 0.0, 4.0, 10.0, 6.0, 0.0, 9.0, 8.0,
        10.0, 2.0, 1.0, 10.0, 0.0, 6.0, 0.0, 3.0, 5.0, 5.0, 4.0, 0.0, 0.0, 1.0, 0.0,
        8.0, 6.0, 7.0, 3.0, 7.0, 9.0, 4.0, 5.0, 11.0, 9.0, 11.0, 4.0, 8.0, 11.0, 7.0,
        10.0, 8.0, 8.0, 4.0, 10.0, 1.0, 6.0, 8.0, 10.0, 6.0, 4.0, 3.0, 5.0, 5.0, 8.0,
    ];

    #[rustfmt::skip]
    pub const EXPECTED: [u32; 75] = [
        1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3,
        1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 0, 3, 3,
        1, 0, 0, 1, 0, 1, 0, 1, 1, 2, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 2, 0, 2, 2, 2,
        1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decisive case, straight against the kernel. Written before anything
    /// else in this op, because "a priority flood that looks right" is the
    /// failure mode and it passes every other test here.
    #[test]
    fn matches_skimage_where_a_naive_priority_flood_would_not() {
        use reference_case::*;

        let shape = SHAPE;
        let image = SOURCE.iter().map(|value| -value).collect::<Vec<_>>();
        let mut mask = SOURCE
            .iter()
            .map(|value| *value > THRESHOLD)
            .collect::<Vec<_>>();
        let mut output = vec![0u32; SOURCE.len()];
        for (label, coords) in SEEDS {
            let index = (coords[0] * shape[1] + coords[1]) * shape[2] + coords[2];
            assert!(mask[index], "seed {label} must survive the threshold");
            output[index] = label;
        }

        watershed_raveled(&image, &shape, &mut mask, &mut output, true).expect("a flood");

        assert_eq!(output, EXPECTED);
        let sizes = (1..=3)
            .map(|label| output.iter().filter(|value| **value == label).count())
            .collect::<Vec<_>>();
        assert_eq!(sizes, SIZES.to_vec());
        assert_ne!(
            SIZES, NAIVE_SIZES,
            "if these agreed the case would pin nothing"
        );
    }

    /// **The push counter runs out, and the flood says so instead of wrapping.**
    ///
    /// `age` is the `i32` skimage declares, so past [`AGE_LIMIT`] it goes
    /// negative and `smaller` inverts on every tie — a later push sorting before
    /// an earlier one, which is a different partition and not a slightly
    /// different boundary. Reaching that honestly takes 2.1 billion pushes, so
    /// `flood` takes the limit as a parameter and this test hands it a small
    /// one: the refusal is the same branch on the same counter, reached in
    /// microseconds.
    ///
    /// Both halves are asserted. The refusal, and that the *same* volume floods
    /// to the *same* labels under the real limit — otherwise a guard that
    /// refused everything would pass the first half.
    #[test]
    fn a_flood_that_outruns_the_push_counter_is_refused_rather_than_wrapped() {
        use reference_case::*;

        let shape = SHAPE;
        let image = SOURCE.iter().map(|value| -value).collect::<Vec<_>>();
        let seed = |mask: &mut Vec<bool>, output: &mut Vec<u32>| {
            for (label, coords) in SEEDS {
                let index = (coords[0] * shape[1] + coords[1]) * shape[2] + coords[2];
                assert!(mask[index]);
                output[index] = label;
            }
        };
        let fresh = || {
            let mask = SOURCE
                .iter()
                .map(|value| *value > THRESHOLD)
                .collect::<Vec<bool>>();
            let output = vec![0u32; SOURCE.len()];
            (mask, output)
        };

        let (mut mask, mut output) = fresh();
        seed(&mut mask, &mut output);
        let err = flood(&image, &shape, &mut mask, &mut output, true, 8)
            .expect_err("a limit of eight pushes must be reached by this fixture")
            .to_string();
        assert!(
            err.contains("pushed more than 8") && err.contains("wraps negative"),
            "the refusal must say what ran out and what it would have cost: {err}"
        );

        // and the same fixture under the real limit is the reference partition,
        // so the guard is a ceiling and not a change to the flood.
        let (mut mask, mut output) = fresh();
        seed(&mut mask, &mut output);
        watershed_raveled(&image, &shape, &mut mask, &mut output, true).expect("a flood");
        assert_eq!(output, EXPECTED);
    }

    /// **Which volumes reach the limit**, computed rather than guessed, because
    /// the answer decides whether the refusal above is a formality or a
    /// ceiling this crate's own stated target crosses.
    ///
    /// `age` counts pushes and the two modes push differently: with no
    /// separating line a voxel is labelled *at push time* and is never queued
    /// twice, so the bound is the voxel count; with one, labels settle at pop
    /// and a voxel can be queued once per neighbour — `2 * ndim`, six in 3-D.
    ///
    /// `super::super::watershed`'s header contemplates volumes above both
    /// figures, which is what makes this worth a guard rather than a comment.
    #[test]
    fn the_limit_is_reachable_at_the_volumes_this_crate_contemplates() {
        let limit = AGE_LIMIT as f64;
        let voxels = |edge: f64| edge * edge * edge;

        // No line: one push per voxel.
        assert!(voxels(1024.0) < limit, "a 1024^3 flood without a line fits");
        assert!(
            voxels(1400.0) > limit,
            "and one at 1400^3 does not: {} pushes against {limit}",
            voxels(1400.0)
        );

        // With a line: up to six pushes per voxel in three dimensions.
        assert!(
            6.0 * voxels(1024.0) > limit,
            "a 1024^3 flood with a separating line runs out: {} pushes against {limit}",
            6.0 * voxels(1024.0)
        );
        assert!(
            6.0 * voxels(512.0) < limit,
            "and a 512^3 one does not, so the ceiling is between them"
        );
        println!(
            "the push limit is {AGE_LIMIT}; a 1024^3 flood pushes up to {} without a \
             separating line and {} with one",
            voxels(1024.0),
            6.0 * voxels(1024.0)
        );
    }

    /// The tie-breaking rule, isolated from the flood: equal values are ordered
    /// by `age`, and `age` is what the pop order of the seeds sets.
    #[test]
    fn heap_pops_in_value_then_age_order() {
        let mut heap = Heap::with_capacity(8);
        for (value, age) in [(2.0, 1), (1.0, 5), (1.0, 2), (3.0, 0)] {
            heap.push(HeapItem {
                value,
                age,
                index: 0,
                source: 0,
            });
        }
        let mut popped = Vec::new();
        while heap.len() > 0 {
            let item = heap.pop();
            popped.push((item.value, item.age));
        }
        assert_eq!(popped, vec![(1.0, 2), (1.0, 5), (2.0, 1), (3.0, 0)]);
    }

    /// The half of the tie-breaking that has **no stable rule to state**: items
    /// that are equal on both keys come out in the order the array's layout
    /// dictates, which is neither insertion order nor its reverse. A heap that
    /// happened to be stable would order the seeds differently, and every `age`
    /// downstream of them with it.
    #[test]
    fn equal_keys_pop_in_layout_order_which_is_not_insertion_order() {
        let mut heap = Heap::with_capacity(8);
        for index in 0..7usize {
            heap.push(HeapItem {
                value: 0.0,
                age: 0,
                index,
                source: index,
            });
        }
        let mut popped = Vec::new();
        while heap.len() > 0 {
            popped.push(heap.pop().index);
        }
        assert_eq!(popped, vec![0, 6, 5, 4, 3, 2, 1]);
        assert_ne!(popped, (0..7).collect::<Vec<_>>(), "not insertion order");
        assert_ne!(popped, (0..7).rev().collect::<Vec<_>>(), "nor its reverse");
    }

    #[test]
    fn splits_a_one_dimensional_valley_between_two_seeds() {
        // image (already negated): a valley at both ends, a ridge in the middle.
        let image = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        let mut mask = vec![true; image.len()];
        let mut output = vec![0u32; image.len()];
        output[0] = 1;
        output[6] = 2;
        watershed_raveled(&image, &[7], &mut mask, &mut output, true).expect("a flood");
        assert_eq!(output[0], 1);
        assert_eq!(output[1], 1);
        assert_eq!(output[5], 2);
        assert_eq!(output[6], 2);
        // The ridge is a separating line.
        assert_eq!(output[3], 0);
    }

    #[test]
    fn without_a_separating_line_every_masked_voxel_is_labelled() {
        let image = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        let mut mask = vec![true; image.len()];
        let mut output = vec![0u32; image.len()];
        output[0] = 1;
        output[6] = 2;
        watershed_raveled(&image, &[7], &mut mask, &mut output, false).expect("a flood");
        assert!(output.iter().all(|&label| label != 0));
    }

    #[test]
    fn respects_the_mask() {
        let image = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        let mut mask = vec![true, true, false, false, false, true, true];
        let mut output = vec![0u32; image.len()];
        output[0] = 1;
        output[6] = 2;
        watershed_raveled(&image, &[7], &mut mask, &mut output, true).expect("a flood");
        assert_eq!(output, vec![1, 1, 0, 0, 0, 2, 2]);
    }

    /// The number the memory arithmetic in `super::watershed` is stated in, kept
    /// honest against the type it describes.
    #[test]
    fn a_queued_item_is_thirty_two_bytes() {
        assert_eq!(HEAP_ITEM_BYTES, 32);
    }
}
