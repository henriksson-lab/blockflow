// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The sample lattice read as a grid of its own.**
//
// A windowed statistic evaluated on a coarse lattice and interpolated back is
// one op today, and its reach is `lattice distance + element radius` against the
// fine volume — two terms belonging to two different dependencies, summed into
// one number. Split into a statistic phase writing the lattice and an
// interpolation phase reading it, the statistic reaches the element and nothing
// more, and the interpolation reaches one *coarse* voxel.
//
// The blocks of such a phase are cut on the **lattice**, not on the fine volume.
// That is not a convenience: a phase's valid regions must tile its own output
// and no chunk of that output may be cut by a region boundary, so cutting the
// fine grid and deriving lattice counts would put boundaries wherever the
// arithmetic landed. Cutting the lattice cannot. What varies instead is how much
// of the fine image each block reads, which is stated per block.
//
// This file pins the geometry that makes that work, before any op uses it:
//
// 1. The coarse volume is the sample counts.
// 2. **A span covers every voxel its samples read** — the property everything
//    else rests on, checked against the window directly rather than against the
//    formula that produced it.
// 3. **Adjacent blocks leave no gap**: the union over a partition of the lattice
//    covers every voxel the whole-volume evaluation reads.
// 4. Spans clamp at the volume's edge and never run past it.
// 5. **Interior blocks of equal sample count get equal-width spans**, which is
//    what lets a block's output shape stay a function of its input shape — the
//    reason for cutting the output grid rather than the input one.
// 6. A region handed in as voxels rather than as lattice indices is refused.

use blockflow::ops::{ElementShape, SampleLattice, Sampling, StructuringElement};
use blockflow::region::Region;

const VOLUME: [usize; 3] = [64, 48, 7];
const SPACING: [usize; 3] = [9, 7, 3];

fn lattice() -> SampleLattice {
    SampleLattice::of(&Sampling::every(SPACING), VOLUME).unwrap()
}

/// Deliberately **even on one axis**, so the window is off-centre there and a
/// span derived from a symmetric radius would be wrong on one side.
fn element() -> StructuringElement {
    StructuringElement::from_size(ElementShape::Box, [5, 4, 3]).unwrap()
}

fn window() -> [(usize, usize); 3] {
    let element = element();
    [element.sides(0), element.sides(1), element.sides(2)]
}

/// The fine voxels the sample at `index` on `axis` actually reads, clamped the
/// way the kernel clamps.
fn read_by(lattice: &SampleLattice, axis: usize, index: usize) -> (usize, usize) {
    let (lo, hi) = window()[axis];
    let centre = lattice.centre(axis, index);
    (
        centre.saturating_sub(lo),
        (centre + hi + 1).min(VOLUME[axis]),
    )
}

// ------------------------------------------------------------- 1. shape --

#[test]
fn the_coarse_volume_is_the_sample_count() {
    let lattice = lattice();
    assert_eq!(
        lattice.lattice_volume(),
        [lattice.count(0), lattice.count(1), lattice.count(2)]
    );
    // And it is genuinely coarser, or nothing below is being tested.
    for axis in 0..3 {
        assert!(lattice.count(axis) < VOLUME[axis], "axis {axis}");
    }
}

// ----------------------------------------------------------- 2. coverage --

#[test]
fn a_span_covers_every_voxel_its_samples_read() {
    let lattice = lattice();
    let (lo, hi) = (window()[0].0, window()[0].1);
    for axis in 0..3 {
        let (lo, hi) = if axis == 0 { (lo, hi) } else { window()[axis] };
        let count = lattice.count(axis);
        for start in 0..count {
            for len in 1..=(count - start) {
                let (low, high) = lattice.source_span(axis, start, len, lo, hi);
                for index in start..start + len {
                    let (want_low, want_high) = read_by(&lattice, axis, index);
                    assert!(
                        low <= want_low && high >= want_high,
                        "axis {axis} block {start}+{len} spans {low}..{high} and sample {index} \
                         reads {want_low}..{want_high}"
                    );
                }
            }
        }
    }
}

// --------------------------------------------------------------- 3. gaps --

#[test]
fn a_partition_of_the_lattice_leaves_no_voxel_unread() {
    let lattice = lattice();
    for axis in 0..3 {
        let (lo, hi) = window()[axis];
        let count = lattice.count(axis);
        for width in 1..=count {
            // Every sample must be inside exactly one part, and every voxel any
            // sample reads must be inside that part's span.
            let mut covered = vec![false; VOLUME[axis]];
            let mut start = 0;
            while start < count {
                let len = width.min(count - start);
                let (low, high) = lattice.source_span(axis, start, len, lo, hi);
                for voxel in covered.iter_mut().take(high).skip(low) {
                    *voxel = true;
                }
                start += len;
            }
            for index in 0..count {
                let (want_low, want_high) = read_by(&lattice, axis, index);
                for voxel in want_low..want_high {
                    assert!(
                        covered[voxel],
                        "axis {axis} width {width}: voxel {voxel}, read by sample {index}, is in \
                         no block's span"
                    );
                }
            }
        }
    }
}

// -------------------------------------------------------------- 4. edges --

#[test]
fn a_span_never_runs_past_the_volume() {
    let lattice = lattice();
    for axis in 0..3 {
        let (lo, hi) = window()[axis];
        let count = lattice.count(axis);
        for start in 0..count {
            for len in 0..=(count - start) {
                let (low, high) = lattice.source_span(axis, start, len, lo, hi);
                assert!(low <= high, "axis {axis}: {low}..{high} is inverted");
                assert!(
                    high <= VOLUME[axis],
                    "axis {axis} block {start}+{len}: {high} is past {}",
                    VOLUME[axis]
                );
            }
        }
    }
}

// ------------------------------------------------- 5. equal-width interiors --

/// The property the whole arrangement rests on: away from the volume's ends,
/// two blocks holding the same number of samples read the same number of
/// voxels. That is what keeps a block's output shape a function of its input
/// shape, which is what `BlockOp::output_shape` can express.
#[test]
fn interior_blocks_of_equal_count_read_equal_widths() {
    let lattice = lattice();
    for axis in 0..3 {
        let (lo, hi) = window()[axis];
        let count = lattice.count(axis);
        for len in 1..=count {
            let mut widths: Vec<(usize, usize)> = Vec::new();
            for start in 0..=(count - len) {
                let (low, high) = lattice.source_span(axis, start, len, lo, hi);
                // Interior means neither side clamped: the span is the one the
                // arithmetic gives rather than the one the array's end allows.
                let clamped_low = lattice.centre(axis, start) < lo;
                let clamped_high = lattice.centre(axis, start + len - 1) + hi + 1 > VOLUME[axis];
                if !clamped_low && !clamped_high {
                    widths.push((start, high - low));
                }
            }
            if let Some(&(first_start, first)) = widths.first() {
                for &(start, width) in &widths {
                    assert_eq!(
                        width, first,
                        "axis {axis} len {len}: block at {start} reads {width} voxels and block \
                         at {first_start} reads {first}"
                    );
                }
            }
        }
    }
}

// ------------------------------------------------------- 6. the two spaces --

#[test]
fn a_region_in_voxels_rather_than_samples_is_refused() {
    let lattice = lattice();
    // A region the size of the fine volume: legal as voxels, past the end as
    // lattice indices, which is exactly the confusion worth refusing.
    let voxels = Region::new(&[0, 0, 0], &VOLUME);
    let failed = lattice.source_region(&voxels, window()).unwrap_err();
    let message = failed.to_string();
    assert!(
        message.contains("lattice indices"),
        "the refusal must say which space the region is in: {message}"
    );
}

/// The whole lattice does **not** read the whole volume, and that is a property
/// of `Sampling::Centred` rather than a defect here.
///
/// The lattice leaves an unsampled margin at each end — `first = (volume −
/// (count−1)·spacing) / 2` — so where the margin is wider than the window's
/// radius, the outermost voxels are read by no sample at all. They still get a
/// value: the interpolation gives them the nearest sample's, which is the
/// degenerate bracket. Asserting the exact span rather than "the whole volume"
/// is what keeps that visible instead of being discovered as an off-by-five.
#[test]
fn a_region_of_the_lattice_maps_to_the_span_its_samples_read() {
    let lattice = lattice();
    let whole = Region::new(&[0, 0, 0], &lattice.lattice_volume());
    let source = lattice.source_region(&whole, window()).unwrap();
    for axis in 0..3 {
        let (lo, hi) = window()[axis];
        let last = lattice.count(axis) - 1;
        let expected_low = lattice.centre(axis, 0).saturating_sub(lo);
        let expected_high = (lattice.centre(axis, last) + hi + 1).min(VOLUME[axis]);
        assert_eq!(source.start[axis], expected_low, "axis {axis} start");
        assert_eq!(
            source.shape[axis],
            expected_high - expected_low,
            "axis {axis} extent"
        );
    }
    // And on axis 0 the margin really does exceed the window, so this test is
    // distinguishing the two answers rather than agreeing with both.
    assert!(
        source.start[0] > 0,
        "axis 0 was expected to leave an unread margin; it starts at {}",
        source.start[0]
    );
}

/// An irregular lattice is the escape hatch `Sampling::At` exists for, and the
/// geometry must not assume a constant gap anywhere.
#[test]
fn an_irregular_lattice_has_a_correct_span_too() {
    let positions = [vec![0, 1, 20, 63], vec![4, 30], vec![0, 6]];
    let lattice = SampleLattice::at(VOLUME, positions.clone()).unwrap();
    for axis in 0..3 {
        let (lo, hi) = window()[axis];
        let count = positions[axis].len();
        for start in 0..count {
            for len in 1..=(count - start) {
                let (low, high) = lattice.source_span(axis, start, len, lo, hi);
                for index in start..start + len {
                    let centre = positions[axis][index];
                    let want_low = centre.saturating_sub(lo);
                    let want_high = (centre + hi + 1).min(VOLUME[axis]);
                    assert!(
                        low <= want_low && high >= want_high,
                        "axis {axis} block {start}+{len} spans {low}..{high}, sample {index} at \
                         {centre} reads {want_low}..{want_high}"
                    );
                }
            }
        }
    }
}

// ------------------------------------------- 7. the handed window, inverted --

/// The round trip `output_shape` rests on: a block handed the window for `n`
/// samples must be derivable back to exactly `n`.
///
/// If this ever fails, a statistic phase would allocate an output of the wrong
/// count and the mismatch would surface as a shape error deep inside a run
/// rather than as a fact about the lattice.
#[test]
fn the_handed_window_inverts_to_the_sample_count() {
    let lattice = lattice();
    for axis in 0..3 {
        let (lo, hi) = window()[axis];
        let count = lattice.count(axis);
        for start in 0..count {
            for len in 1..=(count - start) {
                let Some((low, high)) = lattice.source_window(axis, start, len, lo, hi) else {
                    continue;
                };
                assert_eq!(
                    lattice.samples_for_window(axis, high - low, lo, hi),
                    Some(len),
                    "axis {axis} block {start}+{len} was handed {} voxels",
                    high - low
                );
            }
        }
    }
}

/// Sliding inward must not push a sample out of the buffer it is meant to be
/// computed in — that would be a window reading past its own block.
#[test]
fn the_handed_window_still_contains_every_samples_read() {
    let lattice = lattice();
    for axis in 0..3 {
        let (lo, hi) = window()[axis];
        let count = lattice.count(axis);
        for start in 0..count {
            for len in 1..=(count - start) {
                let Some((low, high)) = lattice.source_window(axis, start, len, lo, hi) else {
                    continue;
                };
                for index in start..start + len {
                    let (want_low, want_high) = read_by(&lattice, axis, index);
                    assert!(
                        low <= want_low && high >= want_high,
                        "axis {axis} block {start}+{len} handed {low}..{high}, sample {index} \
                         reads {want_low}..{want_high}"
                    );
                }
            }
        }
    }
}

/// Every block of the same sample count is handed the same width, wherever it
/// sits — including at the ends, which is the difference from `source_span`.
#[test]
fn the_handed_window_is_the_same_width_everywhere() {
    let lattice = lattice();
    for axis in 0..3 {
        let (lo, hi) = window()[axis];
        let count = lattice.count(axis);
        for len in 1..=count {
            let widths: Vec<usize> = (0..=(count - len))
                .filter_map(|start| lattice.source_window(axis, start, len, lo, hi))
                .map(|(low, high)| high - low)
                .collect();
            if let Some(&first) = widths.first() {
                assert!(
                    widths.iter().all(|&width| width == first),
                    "axis {axis} len {len} gave widths {widths:?}"
                );
            }
        }
    }
}

/// An irregular lattice has no constant width to slide, and says so rather than
/// inventing one. It keeps the fused op, which needs no inversion.
#[test]
fn an_irregular_lattice_has_no_handed_window() {
    let lattice = SampleLattice::at(VOLUME, [vec![0, 1, 20, 63], vec![4, 30], vec![0, 6]]).unwrap();
    assert_eq!(lattice.uniform_gap(0), None);
    assert_eq!(lattice.source_window(0, 0, 2, 2, 2), None);
    assert_eq!(lattice.samples_for_window(0, 10, 2, 2), None);
}

/// A regular lattice does have one, so the test above is about irregularity and
/// not about the method never answering.
#[test]
fn a_regular_lattice_reports_its_gap() {
    let lattice = lattice();
    for axis in 0..3 {
        assert_eq!(
            lattice.uniform_gap(axis),
            Some(SPACING[axis]),
            "axis {axis}"
        );
    }
}

/// A window wider than the volume cannot be slid anywhere and is refused rather
/// than clipped into a different width.
#[test]
fn a_window_wider_than_the_volume_is_refused() {
    let lattice = lattice();
    assert_eq!(lattice.source_window(2, 0, lattice.count(2), 50, 50), None);
}
