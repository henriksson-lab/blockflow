// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The per-slice half: the measurement of one pair's per-slice displacements,
//! the rule that fills in unmeasured slices, and the placement that turns
//! tracks into one position per tile per slice. All on constructed data.

mod common;

use blockflow::ops::fft::squared_difference_direct;
use ndarray::Array2;
use tile_stitching::place::{place_layout, AlignmentSlot, PlaceParams, SourceSlab, WobblyLayout};
use tile_stitching::plane::PlaneGeometry;
use tile_stitching::wobbly::{
    measure_wobbly_axis, prepare_normalization, select_trajectories, AlignmentStatus,
    MeasureParams, ShiftMethod, SourceStatus, Track,
};

/// `measured.len() * 3` slices, every third measured with the given status.
fn constructed_track(measured: &[i32], displacement: impl Fn(usize) -> [isize; 2]) -> Track {
    let n = measured.len() * 3;
    let mut displacements = vec![vec![0isize, 0]; n];
    let mut qualities = vec![f64::NEG_INFINITY; n];
    let mut status = vec![AlignmentStatus::UNALIGNED; n];
    for (index, &value) in measured.iter().enumerate() {
        let slot = index * 3;
        status[slot] = value;
        if value >= AlignmentStatus::VALID {
            displacements[slot] = displacement(index).to_vec();
            qualities[slot] = -1.0e-9 * (index as f64 + 1.0);
        } else {
            displacements[slot] = vec![-30, -30];
        }
    }
    Track::new(displacements, qualities, status).expect("a track")
}

fn two_tile_layout(track: Track, positions: [Vec<isize>; 2], axis: usize) -> WobblyLayout {
    let height = track.len();
    WobblyLayout {
        sources: positions
            .into_iter()
            .map(|position| SourceSlab { position, height })
            .collect(),
        alignments: vec![AlignmentSlot {
            pre: 0,
            post: 1,
            track,
        }],
        axis,
    }
}

fn unsmoothed() -> PlaceParams {
    PlaceParams {
        smooth: tile_stitching::wobbly::SmoothParams {
            window: None,
            window_length: 1,
            binary: None,
        },
        smooth_optimized: None,
        ..PlaceParams::default()
    }
}

// ---------------------------------------------------- the resolution rule --

/// A run with nothing usable below it holds the slice above; the failed slice
/// itself is never repaired.
#[test]
fn a_leading_run_holds_the_slice_above() {
    let mut track = constructed_track(
        &[AlignmentStatus::NO_SIGNAL, AlignmentStatus::ALIGNED],
        |_| [7, -2],
    );
    let report = track.resolve_rejected();
    assert_eq!(report.held_right, 2, "slices 1 and 2 are held at slice 3");
    assert_eq!(report.held_left, 2, "slices 4 and 5 are held at slice 3");
    assert_eq!(report.interpolated, 0);
    assert_eq!(track.displacements[1], vec![7, -2]);
    assert_eq!(track.status[1], AlignmentStatus::FIXED);
    assert_eq!(track.status[0], AlignmentStatus::NO_SIGNAL);
    assert_eq!(track.displacements[0], vec![-30, -30]);
}

/// Between two measured slices the run is interpolated and rounded.
#[test]
fn a_run_between_two_measurements_is_interpolated() {
    let mut track = constructed_track(&[AlignmentStatus::ALIGNED; 2], |index| {
        [3 * index as isize, 0]
    });
    let report = track.resolve_rejected();
    assert_eq!(report.interpolated, 2);
    assert_eq!(track.displacements[1], vec![1, 0]);
    assert_eq!(track.displacements[2], vec![2, 0]);
    assert!(track.qualities[1].is_finite());
}

/// A track never measured at all is invalidated whole and its displacements
/// are not touched.
#[test]
fn a_track_that_was_never_measured_is_invalidated() {
    let track = Track::new(
        vec![vec![0, 0]; 12],
        vec![f64::NEG_INFINITY; 12],
        vec![AlignmentStatus::UNALIGNED; 12],
    )
    .expect("a track");
    let mut resolved = track.clone();
    let report = resolved.resolve_rejected();
    assert!(report.whole_stack_invalidated);
    assert_eq!(report.invalidated, 12);
    assert!(resolved
        .status
        .iter()
        .all(|&status| status == AlignmentStatus::INVALID));
    assert_eq!(resolved.displacements, track.displacements);
}

/// `NO_MINIMA` and `UNTRACED` are failures like any other: a run between them
/// has no usable neighbour and is invalidated.
#[test]
fn a_run_between_two_failures_is_invalidated() {
    let mut track = constructed_track(
        &[AlignmentStatus::NO_MINIMA, AlignmentStatus::UNTRACED],
        |_| [0, 0],
    );
    let report = track.resolve_rejected();
    assert_eq!(report.unaligned, 4);
    assert_eq!(report.invalidated, 4);
    assert_eq!(track.status[0], AlignmentStatus::NO_MINIMA);
    assert_eq!(track.status[3], AlignmentStatus::UNTRACED);
    assert!(track.status[1..3]
        .iter()
        .all(|&status| status == AlignmentStatus::INVALID));
    assert!(track.valids(f64::NEG_INFINITY).iter().all(|usable| !usable));
}

/// The quality filter is a plain strict threshold.
#[test]
fn the_quality_filter_is_a_strict_threshold() {
    let track = Track::new(
        vec![vec![0, 0]; 3],
        vec![f64::NEG_INFINITY, -1.0, 1.0],
        vec![AlignmentStatus::ALIGNED; 3],
    )
    .expect("a track");
    assert_eq!(track.valids(f64::NEG_INFINITY), vec![false, true, true]);
    assert_eq!(track.valids(-1.0), vec![false, false, true]);
    assert_eq!(track.valids(0.0), vec![false, false, true]);
}

// -------------------------------------------------------------- placement --

/// The relative placement of two tiles is the displacement, slice by slice,
/// and the lowering puts the lower corner on the origin.
#[test]
fn the_placement_follows_the_displacement_and_is_lowered() {
    let track = constructed_track(&[AlignmentStatus::ALIGNED; 4], |index| {
        [10 + 4 * index as isize, 0]
    });
    let layout = two_tile_layout(track, [vec![0, 0, 0], vec![10, 0, 0]], 2);
    let lowered = place_layout(&layout, &unsmoothed()).expect("a placement");
    let raw = place_layout(
        &layout,
        &PlaceParams {
            lower_to_origin: false,
            ..unsmoothed()
        },
    )
    .expect("a placement");
    for slice in 0..12 {
        let (a, b) = (&lowered.positions[0][slice], &lowered.positions[1][slice]);
        // Measured every third slice, interpolated between, held after the last.
        let expected = if slice >= 9 {
            22
        } else {
            10 + 4 * (slice / 3) as isize + [0, 1, 3][slice % 3]
        };
        assert_eq!(b[0] - a[0], expected, "slice {slice}");
        assert_eq!(a[2], slice as isize, "the wobble-axis coordinate");
    }
    let lowest = lowered
        .positions
        .iter()
        .flatten()
        .map(|position| position[0])
        .min();
    assert_eq!(lowest, Some(0));
    assert!(raw
        .positions
        .iter()
        .flatten()
        .any(|position| position[0] < 0));
}

/// A layout whose wobble-axis coordinates do not start at zero is placed the
/// same as one that does, offset along the axis.
#[test]
fn a_layout_not_starting_at_zero_is_placed_the_same() {
    let track = constructed_track(&[AlignmentStatus::ALIGNED; 3], |index| {
        [10 + index as isize, 0]
    });
    let at_zero = place_layout(
        &two_tile_layout(track.clone(), [vec![0, 0, 0], vec![10, 0, 0]], 2),
        &unsmoothed(),
    )
    .expect("a placement");
    let raised = place_layout(
        &two_tile_layout(track, [vec![0, 0, 5], vec![10, 0, 5]], 2),
        &unsmoothed(),
    )
    .expect("a placement");
    assert_eq!(at_zero.status, raised.status);
    for (zero, five) in at_zero
        .positions
        .iter()
        .flatten()
        .zip(raised.positions.iter().flatten())
    {
        assert_eq!(vec![zero[0], zero[1], zero[2] + 5], *five);
    }
}

/// The wobble axis is a parameter: the same numbers placed along axis 0 give
/// the same answer with the coordinates permuted.
#[test]
fn the_wobble_axis_is_not_hard_coded() {
    let track = constructed_track(&[AlignmentStatus::ALIGNED; 4], |index| {
        [10 + 4 * index as isize, 1]
    });
    let along_two = place_layout(
        &two_tile_layout(track.clone(), [vec![0, 0, 0], vec![10, 0, 0]], 2),
        &unsmoothed(),
    )
    .expect("a placement along axis 2");
    let along_zero = place_layout(
        &two_tile_layout(track, [vec![0, 0, 0], vec![0, 10, 0]], 0),
        &unsmoothed(),
    )
    .expect("a placement along axis 0");
    assert_eq!(along_two.status, along_zero.status);
    for (two, zero) in along_two
        .positions
        .iter()
        .flatten()
        .zip(along_zero.positions.iter().flatten())
    {
        assert_eq!(vec![two[2], two[0], two[1]], *zero);
    }
}

/// A slice where the alignment failed leaves both tiles isolated there.
/// `fix_isolated` interpolates them and lets them into the fusion; without it
/// they stay out.
#[test]
fn isolated_slices_are_fixed_or_left_out() {
    let mut track = constructed_track(
        &[
            AlignmentStatus::ALIGNED,
            AlignmentStatus::NO_SIGNAL,
            AlignmentStatus::ALIGNED,
        ],
        |_| [10, 0],
    );
    track.resolve_rejected();
    let layout = two_tile_layout(track, [vec![0, 0, 0], vec![10, 0, 0]], 2);
    let fixed = place_layout(&layout, &unsmoothed()).expect("a placement");
    let left_out = place_layout(
        &layout,
        &PlaceParams {
            fix_isolated: false,
            ..unsmoothed()
        },
    )
    .expect("a placement");
    // Slice 3 failed; slices 1-2 and 4-5 were filled by the resolution rule
    // from their one usable side.
    assert_eq!(left_out.report.isolated, 2, "slice 3 of both tiles");
    assert_eq!(left_out.status[1][3], SourceStatus::ISOLATED);
    assert_eq!(fixed.status[1][3], SourceStatus::FIXED);
    assert_eq!(fixed.positions[1][3][0] - fixed.positions[0][3][0], 10);
}

/// A tile alone in a slice because no other tile reaches it there is neither
/// isolated nor left out of the fusion.
#[test]
fn a_tile_alone_by_geometry_is_kept() {
    let track = constructed_track(&[AlignmentStatus::ALIGNED; 2], |_| [10, 0]);
    let height = track.len();
    let layout = WobblyLayout {
        sources: vec![
            SourceSlab {
                position: vec![0, 0, 0],
                height: height + 3,
            },
            SourceSlab {
                position: vec![10, 0, 3],
                height,
            },
        ],
        alignments: vec![AlignmentSlot {
            pre: 0,
            post: 1,
            track,
        }],
        axis: 2,
    };
    let placed = place_layout(&layout, &PlaceParams::default()).expect("a placement");
    assert_eq!(placed.report.isolated, 0);
    assert!(placed
        .status
        .iter()
        .flatten()
        .all(|&status| status == SourceStatus::VALID));
    assert_eq!(placed.positions[0].len(), height + 3);
}

// ------------------------------------------------------------ measurement --

/// Two planes per slice, the second cut from the same scene at a nominal
/// offset of 40 columns plus a planted per-slice shift.
fn planted_cuts(
    slices: usize,
    shift: impl Fn(usize) -> [isize; 2],
) -> (Vec<Array2<f64>>, Vec<Array2<f64>>, PlaneGeometry) {
    let shape = [80usize, 60];
    let geometry =
        PlaneGeometry::new([0, 0], shape, [0, 40], shape, [[-5, 5], [-5, 5]]).expect("a geometry");
    let plane = |z: usize, offset: [isize; 2], cut: [(usize, usize); 2]| {
        Array2::from_shape_fn((cut[0].1 - cut[0].0, cut[1].1 - cut[1].0), |(r, c)| {
            common::scene(
                z as isize,
                (cut[0].0 + r) as isize + offset[0],
                (cut[1].0 + c) as isize + offset[1],
            ) as f64
        })
    };
    let a = (0..slices)
        .map(|z| plane(z, [0, 0], geometry.cut_a))
        .collect();
    let b = (0..slices)
        .map(|z| {
            let s = shift(z);
            plane(z, [s[0], 40 + s[1]], geometry.cut_b)
        })
        .collect();
    (a, b, geometry)
}

fn drift(z: usize) -> [isize; 2] {
    [z as isize / 4 - 1, 1 - z as isize / 6]
}

/// Both shift methods recover a planted per-slice shift on every measured
/// slice, and the unmeasured ones come back `UNALIGNED`.
#[test]
fn the_measurement_recovers_a_planted_drift() {
    for method in [ShiftMethod::Tracing, ShiftMethod::Minimization] {
        let (a, b, geometry) = planted_cuts(12, drift);
        let params = MeasureParams {
            max_shifts: [[-5, 5], [-5, 5]],
            slice_step: 2,
            find_shifts: method,
            ..MeasureParams::default()
        };
        let measured =
            measure_wobbly_axis(a, b, &geometry, [0, 40], &params).expect("a measurement");
        assert_eq!(measured.measured, 6, "{method:?}");
        for slice in 0..12 {
            if slice % 2 == 1 {
                assert_eq!(measured.track.status[slice], AlignmentStatus::UNALIGNED);
                continue;
            }
            assert_eq!(
                measured.track.status[slice],
                AlignmentStatus::ALIGNED,
                "{method:?} slice {slice}"
            );
            let s = drift(slice);
            assert_eq!(
                measured.shifts[slice],
                s.to_vec(),
                "{method:?} slice {slice}"
            );
            assert_eq!(
                measured.track.displacements[slice],
                vec![s[0], 40 + s[1]],
                "the displacement carries the nominal offset"
            );
            let lowest = measured.landscapes[slice]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            assert_eq!(
                measured.track.qualities[slice], -lowest,
                "the quality is the minimum"
            );
        }
    }
}

/// The landscape agrees with a padding-free direct walk over the prepared
/// planes, which pins the cut and the offset between a shift and a lag.
#[test]
fn the_landscape_agrees_with_a_direct_walk() {
    let (a, b, geometry) = planted_cuts(4, drift);
    let (mut prepared_a, mut prepared_b) = (a.clone(), b.clone());
    prepare_normalization(&mut prepared_a);
    prepare_normalization(&mut prepared_b);
    let params = MeasureParams {
        max_shifts: [[-5, 5], [-5, 5]],
        slice_step: 1,
        ..MeasureParams::default()
    };
    let measured = measure_wobbly_axis(a, b, &geometry, [0, 40], &params).expect("a measurement");
    let window = geometry.window().expect("a window");
    for slice in 0..4 {
        let direct =
            squared_difference_direct(prepared_a[slice].view(), prepared_b[slice].view(), window);
        let scale = direct.mean_squared().iter().copied().fold(0.0, f64::max);
        for (x, y) in measured.landscapes[slice]
            .iter()
            .zip(direct.mean_squared().iter())
        {
            assert!((x - y).abs() <= 1e-9 * scale, "{x} against {y}");
        }
    }
}

/// A stack without foreground is not measured: every slice is `NO_SIGNAL` and
/// keeps the nominal offset. The same stack above the floor is measured, and
/// a constant landscape has no minima.
#[test]
fn a_dark_stack_is_not_measured_and_a_flat_one_has_no_minima() {
    let geometry =
        PlaneGeometry::new([0, 0], [8, 8], [4, 0], [8, 8], [[-2, 2], [-2, 2]]).expect("a geometry");
    let planes = |value: f64| {
        let shape = geometry.cut_shape_a();
        vec![Array2::from_elem((shape[0], shape[1]), value); 6]
    };
    let params = MeasureParams {
        max_shifts: [[-2, 2], [-2, 2]],
        slice_step: 1,
        slice_validation: None,
        ..MeasureParams::default()
    };
    let dark = measure_wobbly_axis(planes(10.0), planes(10.0), &geometry, [4, 0], &params)
        .expect("a measurement");
    assert!(dark
        .track
        .status
        .iter()
        .all(|&status| status == AlignmentStatus::NO_SIGNAL));
    assert!(dark
        .track
        .displacements
        .iter()
        .all(|displacement| displacement == &vec![4, 0]));
    assert_eq!(dark.measured, 0);

    let flat = measure_wobbly_axis(planes(1000.0), planes(1000.0), &geometry, [4, 0], &params)
        .expect("a measurement");
    assert_eq!(flat.measured, 6);
    assert!(flat
        .track
        .status
        .iter()
        .all(|&status| status == AlignmentStatus::NO_MINIMA));
}

/// A slab projection along the wobble axis is accepted at the stack's ends.
#[test]
fn an_axis_projection_is_clamped_to_the_stack() {
    let (a, b, geometry) = planted_cuts(6, |_| [1, -1]);
    let params = MeasureParams {
        max_shifts: [[-5, 5], [-5, 5]],
        slice_step: 1,
        axis_mip: Some((1, 0)),
        ..MeasureParams::default()
    };
    let measured = measure_wobbly_axis(a, b, &geometry, [0, 40], &params).expect("a measurement");
    assert_eq!(measured.measured, 6);
}

/// Between equally long trajectories the one with the lower summed error
/// wins, each point read from the landscape of the slice it is on.
#[test]
fn the_greedy_pick_reads_each_points_own_slice() {
    // Measured slices 0 and 3 of a stretch of six; slices 1 and 2 hold
    // landscapes that would reverse the choice if they were read instead.
    let mut landscapes = vec![Array2::from_elem((3, 3), 5.0); 6];
    landscapes[0][[0, 0]] = 1.0;
    landscapes[0][[2, 2]] = 2.0;
    landscapes[3][[0, 0]] = 1.0;
    landscapes[3][[2, 2]] = 2.0;
    landscapes[1][[0, 0]] = 9.0;
    landscapes[1][[2, 2]] = 0.0;
    let positions = vec![vec![[0.0, 0.0], [2.0, 2.0]], vec![[0.0, 0.0], [2.0, 2.0]]];
    let trajectories = vec![vec![(0, 1), (1, 1)], vec![(0, 0), (1, 0)]];
    let chosen = select_trajectories(trajectories, &positions, &landscapes, &[0, 3]);
    assert_eq!(chosen, vec![vec![(0, 0), (1, 0)]]);
}
