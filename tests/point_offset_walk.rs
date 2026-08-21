// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A walk over a fixed offset sequence, run as a phase, and the two halves of
// its answer told apart.**
//
// `ops::walk` is a row op that reads a *second array*: for each row it walks a
// fixed list of relative offsets in a fixed order, tests a bound on the array at
// each, and writes the distance attached to the offset it stopped at. This file
// runs it through the executor and asserts the four things the operation is
// accepted on.
//
// 1. **The distance is the distance the rule states**, at every outcome the
//    operation has: a stop at zero, stops at four different distances including
//    two irrational ones, a stop at the stated maximum, and a walk that reaches
//    the end of the sequence without stopping.
// 2. **The answer does not depend on the decomposition.** The same rows, the
//    same numbers, over four lattices — including ones whose seams fall inside
//    the window of nearly every row, so a walk truncated at a seam would show as
//    a shorter distance rather than as nothing at all.
// 3. **The half of the answer that is not reproducible is measured, not
//    hidden.** The order among offsets at equal distance is decided, in the
//    implementation being reproduced, by an unstable sort whose tie order is a
//    property of the machine. The consequence is not uniform: the *distance* is
//    exact, because it is a function of the level the walk stopped in and the
//    level is determined; the *identity of the stopping offset* is not, because
//    it names one member of a level. So the op writes the distance, and the
//    identity comes with `OffsetSequence::stop_is_determined`, which answers for
//    a particular index whether it was forced. Here that is exercised by walking
//    the same data twice with the two orders and comparing both halves.
// 4. **The reach is a stated maximum and a short one is refused.** The walk
//    cannot go further than the offset list, so the operand reach is derived
//    from the list and a target inside the volume but outside the window this
//    block was handed is an error rather than an early stop. That distinction is
//    the whole guard: an early stop would report the distance reached so far,
//    which is a wrong answer with the shape of a measurement.
//
// What is deliberately *not* asserted anywhere here: that a particular offset
// stopped a particular walk in a tie group. That claim cannot be made and no
// fixture in this file makes it.

use std::sync::Arc;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::error::Result;
use blockflow::fragment::{
    append_fragment_phase, BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput, PhaseWork,
};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::coordinates::coordinate_schema;
use blockflow::ops::rows::{collect_rows, Limit, RowStreams, RowValues};
use blockflow::ops::walk::{walk_from, OffsetSequence, OffsetWalkOp};
use blockflow::probes::IdentityOp;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::table::{Schema, Value};
use blockflow::Voxels;
use ndarray::Array3;

const VOLUME: [usize; 3] = [16, 16, 16];
const MAXIMUM: [usize; 3] = [3, 3, 3];

// ------------------------------------------------------------- the fixture --

/// A slab against the `x = 0` face, a body away from it, and three isolated
/// holes inside the body.
///
/// Everything in the fixture is there to make one of the points below land on a
/// different outcome. The slab exists so that a point *on the volume's face* is
/// inside the set being walked; the gap between the slab and the body is what
/// that point's walk eventually finds; the holes are what the interior points
/// find, at four different distances.
fn fixture() -> Voxels {
    let mut array = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for i in 0..3 {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                array[[i, j, k]] = 1.0;
            }
        }
    }
    for i in 5..15 {
        for j in 3..13 {
            for k in 3..13 {
                array[[i, j, k]] = 1.0;
            }
        }
    }
    for hole in [[6, 6, 6], [10, 10, 10], [12, 10, 10]] {
        array[hole] = 0.0;
    }
    array.into()
}

/// The rows the walk is run for.
///
/// Chosen for two properties at once. Each lands on a different outcome — see
/// `EXPECTED` — and several of them sit **on or beside a block seam** for the
/// lattices used below: with four-voxel blocks the seams are at 4, 8 and 12, so
/// `[8, 6, 6]` is the first voxel of its block and `[7, 7, 7]` is the last of
/// its own, and both of their windows straddle a seam on every axis. A walk
/// truncated at a seam would report a *shorter* distance for them, which is
/// exactly what the decomposition assertion is looking for.
const POINTS: [[usize; 3]; 8] = [
    [0, 8, 8],
    [6, 6, 6],
    [7, 6, 6],
    [7, 7, 6],
    [7, 7, 7],
    [8, 6, 6],
    [9, 7, 7],
    [11, 10, 10],
];

/// A value no distance in the sequence can be, so a row whose walk ran off the
/// end can never be confused with one that stopped. Negative because every
/// distance is non-negative; the op refuses anything that collides.
const NOT_FOUND: f64 = -1.0;

/// The distance each of `POINTS` gets, in the canonical row order — which is
/// lexicographic on the coordinate, and `POINTS` is written in it.
///
/// Written out rather than recomputed, so this is a statement of the rule and
/// not a copy of the implementation. Reading down: a walk that runs to the
/// stated maximum before finding anything, a walk that stops on the row's own
/// voxel, and then stops at 1, sqrt 2, sqrt 3 and 2, a walk that never stops,
/// and a stop at 1 where **two** offsets one unit away both satisfy the bound.
const EXPECTED: [f64; 8] = [
    3.0,
    0.0,
    1.0,
    std::f64::consts::SQRT_2,
    1.7320508075688772,
    2.0,
    NOT_FOUND,
    1.0,
];

fn sequence() -> OffsetSequence {
    OffsetSequence::ellipsoid(MAXIMUM, [1.0, 1.0, 1.0])
        .expect("an isotropic maximum under a uniform spacing orders exactly")
}

// -------------------------------------------------------- the row producer --

/// Writes the rows of `POINTS` that lie in this block's core, and nothing else.
///
/// Reads no pixels and no fragments: it exists so that the walk has a row source
/// whose contents are a constant of the test rather than a function of the
/// array, which is what lets `EXPECTED` be written out. Every block writes a
/// fragment, empty or not, so the coverage guard has something to check.
struct PointsOp {
    stream: String,
}

impl FragmentOp for PointsOp {
    fn name(&self) -> &'static str {
        "points"
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.stream.clone(),
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mut rows = blockflow::table::RowBuilder::new(Arc::new(coordinate_schema()));
        for point in POINTS {
            let mine = (0..3).all(|axis| {
                point[axis] >= at.core.start[axis]
                    && point[axis] < at.core.start[axis] + at.core.shape[axis]
            });
            if mine {
                rows.push(point, &[])?;
            }
        }
        Ok(BlockOutput::fragment(self.stream.clone(), rows.encode()))
    }
}

// -------------------------------------------------------------- the runner --

fn identity_workflow() -> Workflow {
    Workflow::new(
        Chain::op(IdentityOp::new("identity", [0, 0, 0])),
        VOLUME,
        Dtype::F64,
    )
}

fn one_pixel_phase(block: [usize; 3]) -> Decomposition {
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![PhaseDecomposition::derive(
            vec![0],
            vec!["identity".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            BlockGrid::new(VOLUME, block).expect("a lattice"),
        )],
        chain_reach: [0, 0, 0],
    }
}

/// The three phases: a pixel phase to have an image, the row producer, and the
/// walk. The walk reads the rows from phase 1 and the **array from image 0**,
/// which is the arrangement that makes a row op able to read a volume at all —
/// a fragment phase `p` reads image `p`, and the phase before a row op writes no
/// image.
fn run(block: [usize; 3], out_of_band: Option<OffsetSequence>) -> Result<Vec<RowValues>> {
    let points = PointsOp {
        stream: "rows.points".to_string(),
    };
    let streams = RowStreams::new(
        "rows.points",
        1,
        "rows.walked",
        Lifecycle::DeleteOnExit,
        coordinate_schema(),
    )?;
    let walk = OffsetWalkOp::new(
        "walk",
        streams,
        0,
        "distance",
        out_of_band.unwrap_or_else(sequence),
        Limit::AtMost(0.0),
        NOT_FOUND,
    )?;
    let plan = append_fragment_phase(one_pixel_phase(block), &points)?;
    let plan = append_fragment_phase(plan, &walk)?;
    let env = ArrayEnvironment::new(fixture(), plan.n_phases(), [4, 4, 4])?;
    execute_phases(
        "walk",
        &identity_workflow(),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&points),
            PhaseWork::Fragments(&walk),
        ],
    )?;
    collect_rows(&env, "rows.walked", 2, VOLUME, walk.schema()?)
}

fn distances(rows: &[RowValues]) -> Vec<([usize; 3], f64)> {
    rows.iter()
        .map(|row| {
            let Value::F64(distance) = row.values[0] else {
                panic!("the appended column is an f64")
            };
            (row.at, distance)
        })
        .collect()
}

// ------------------------------------------------- 1. the stated answers --

#[test]
fn every_row_gets_the_distance_the_rule_states() {
    let measured = distances(&run([16, 16, 16], None).expect("one block runs"));
    assert_eq!(measured.len(), POINTS.len());
    for (index, (at, distance)) in measured.iter().enumerate() {
        assert_eq!(*at, POINTS[index], "the canonical order is by coordinate");
        assert_eq!(
            *distance, EXPECTED[index],
            "the walk from {at:?} reports a different distance"
        );
    }
    // The fixture must contain every outcome or it certifies less than it
    // claims. A set of rows that all stop at distance 1 tests no ordering at
    // all, and one that never reaches the end of the sequence tests nothing
    // about the stated maximum.
    let sequence = sequence();
    let furthest = sequence.distances()[sequence.len() - 1];
    assert!(EXPECTED.contains(&0.0), "a walk that stops immediately");
    assert!(EXPECTED.contains(&furthest), "a walk that reaches the end");
    assert!(EXPECTED.contains(&NOT_FOUND), "a walk that never stops");
    assert!(
        EXPECTED.iter().filter(|value| **value >= 0.0).count() >= 6,
        "several different stopping distances"
    );
    let mut stopping: Vec<f64> = EXPECTED.iter().copied().filter(|d| *d >= 0.0).collect();
    stopping.sort_by(f64::total_cmp);
    stopping.dedup();
    assert!(
        stopping.len() >= 5,
        "the distances must actually differ, or the fixture measures one case five times"
    );
}

// ----------------------------------------- 2. decomposition invariance --

/// **The acceptance property.** Four lattices, one answer.
///
/// The lattices are chosen so that the seams move: a single block, an eight-way
/// split, a sixty-four-way split whose seams at 4, 8 and 12 fall inside almost
/// every row's window, and an uneven one whose blocks do not divide the volume.
/// The rows were placed against those seams on purpose — a walk that stopped at
/// a block boundary and reported the distance reached would give a *shorter*
/// number here, and a shorter number is what this compares against.
#[test]
fn the_answer_does_not_depend_on_the_lattice() {
    let whole = distances(&run([16, 16, 16], None).expect("one block runs"));
    for block in [[8, 8, 8], [4, 4, 4], [6, 5, 7], [16, 4, 16]] {
        let split = distances(&run(block, None).expect("every lattice runs"));
        assert_eq!(
            split, whole,
            "blocks of {block:?} gave a different answer from one block"
        );
    }
}

/// The rows really do sit against the seams the previous test relies on.
///
/// Asserted rather than assumed, because a fixture whose rows all sat in the
/// middle of a block would pass `the_answer_does_not_depend_on_the_lattice`
/// while testing nothing: the walk would never cross a boundary and a truncation
/// bug would be invisible.
#[test]
fn the_rows_are_placed_where_a_truncated_walk_would_show() {
    let block = 4usize;
    let mut straddling = 0;
    let mut on_a_seam = 0;
    for point in POINTS {
        let mut crosses = false;
        for axis in 0..3 {
            let core = point[axis] / block;
            let low = point[axis].saturating_sub(MAXIMUM[axis]);
            let high = (point[axis] + MAXIMUM[axis]).min(VOLUME[axis] - 1);
            if low / block != core || high / block != core {
                crosses = true;
            }
            if point[axis] % block == 0 && point[axis] != 0 {
                on_a_seam += 1;
            }
        }
        if crosses {
            straddling += 1;
        }
    }
    assert_eq!(
        straddling,
        POINTS.len(),
        "every row's window must cross a seam of the four-voxel lattice"
    );
    assert!(
        on_a_seam >= 1,
        "at least one row must be the first voxel of its block"
    );
}

// ------------------------------- 3. the inexact half, measured not hidden --

/// **What the tie order decides, and what it does not.**
///
/// The same data walked twice: once with the sequence, once with every level's
/// members in the opposite order. Neither is the order being reproduced — that
/// one is an unstable sort's, and it is not a rule that can be re-implemented —
/// which is the point. What the two runs bracket is the *dependence*.
///
/// The distances must agree byte for byte, because a level's members all report
/// one distance and the level the walk stops in is determined. The stopping
/// offsets must **not** all agree, or the fixture contains no equidistant
/// candidates and cannot see the problem at all.
#[test]
fn the_distance_agrees_where_the_stopping_offset_cannot() {
    let forwards = sequence();
    let backwards = forwards.with_ties_reversed();

    let one = distances(&run([4, 4, 4], Some(forwards.clone())).expect("a run"));
    let other = distances(&run([4, 4, 4], Some(backwards.clone())).expect("a run"));
    assert_eq!(
        one, other,
        "a distance moved with the order among equidistant offsets, which the sequence's \
         construction check was supposed to make impossible"
    );

    // And now the half that does not agree. The op does not write it — this is
    // the kernel, which returns the index and is the only way to see it.
    let array = fixture();
    let mut moved = Vec::new();
    let mut forced = 0;
    for point in POINTS {
        let left = walk_from(
            &forwards,
            point,
            VOLUME,
            &array,
            [0, 0, 0],
            Limit::AtMost(0.0),
        )
        .expect("the whole volume is its own window");
        let right = walk_from(
            &backwards,
            point,
            VOLUME,
            &array,
            [0, 0, 0],
            Limit::AtMost(0.0),
        )
        .expect("the whole volume is its own window");
        let (Some(left), Some(right)) = (left, right) else {
            assert_eq!(
                left, right,
                "a walk that stops under one order stops under both"
            );
            continue;
        };
        assert_eq!(
            forwards.distances()[left],
            backwards.distances()[right],
            "the distance from {point:?} depends on the tie order"
        );
        if forwards.offsets()[left] != backwards.offsets()[right] {
            moved.push(point);
            assert!(
                !forwards.stop_is_determined(left),
                "the offset from {point:?} moved, and the sequence claimed it was forced"
            );
        }
        if forwards.stop_is_determined(left) {
            forced += 1;
        }
    }
    assert!(
        !moved.is_empty(),
        "the fixture was supposed to hold a row with equidistant candidates on both sides of \
         it, and no row's stopping offset moved. A fixture with no equidistant candidates \
         cannot see a tie-order problem."
    );
    assert!(
        forced >= 1,
        "and at least one row's offset must be forced, or `stop_is_determined` is simply false \
         everywhere and asserts nothing"
    );
}

/// The reason no tie-break was invented: the set of offsets is a fact and the
/// order within a level is not. Two sequences over the same set differ only by a
/// permutation of each level, which this pins directly.
#[test]
fn reversing_the_ties_is_a_permutation_of_the_levels_and_nothing_else() {
    let forwards = sequence();
    let backwards = forwards.with_ties_reversed();
    assert_eq!(forwards.keys(), backwards.keys());
    assert_eq!(forwards.distances(), backwards.distances());
    assert_ne!(forwards.offsets(), backwards.offsets());
    let mut left = forwards.offsets().to_vec();
    let mut right = backwards.offsets().to_vec();
    left.sort_unstable();
    right.sort_unstable();
    assert_eq!(left, right, "the same set, in a different order");
}

// ---------------------------------- 4. the reach, and the degenerate ones --

/// The operand reach is the sequence's own stated maximum, derived rather than
/// configured, and it is the phase's halo. The phase's *reach* stays zero: a
/// row's answer is written at the row's own coordinate whatever window the
/// operand was read over.
#[test]
fn the_operand_reach_is_the_stated_maximum() {
    let points = PointsOp {
        stream: "rows.points".to_string(),
    };
    let streams = RowStreams::new(
        "rows.points",
        1,
        "rows.walked",
        Lifecycle::DeleteOnExit,
        coordinate_schema(),
    )
    .expect("two different streams");
    let walk = OffsetWalkOp::new(
        "walk",
        streams,
        0,
        "distance",
        sequence(),
        Limit::AtMost(0.0),
        NOT_FOUND,
    )
    .expect("a distinguishable not-found value");
    assert_eq!(walk.sequence().maximum(), MAXIMUM);
    let plan = append_fragment_phase(one_pixel_phase([4, 4, 4]), &points).expect("a row phase");
    let plan = append_fragment_phase(plan, &walk).expect("a walk phase");
    assert_eq!(plan.phases[2].halo, MAXIMUM);
    assert_eq!(
        plan.phases[2].reach,
        [0, 0, 0],
        "a widened operand window is not a widened trust region"
    );
    assert_eq!(plan.phases[2].source_images, vec![0]);
    plan.check().expect("the valid regions still tile");
}

/// **A window shorter than the stated maximum is refused, at the offset that
/// leaves it.**
///
/// This is the failure the whole reach discussion exists for. The walk is handed
/// a window six voxels on a side for a sequence that reaches three in every
/// direction, so an offset lands inside the volume and outside the window. It
/// must not be treated as the end of the walk: the distance reached so far is
/// shorter than the data justifies and looks exactly like a measurement.
#[test]
fn a_window_shorter_than_the_maximum_is_refused_rather_than_truncated() {
    let sequence = sequence();
    let Voxels::F64(array) = fixture() else {
        panic!("the fixture is f64")
    };
    let window: Voxels = array
        .slice(ndarray::s![6..12, 6..12, 6..12])
        .to_owned()
        .into();
    let failed = walk_from(
        &sequence,
        [9, 7, 7],
        VOLUME,
        &window,
        [6, 6, 6],
        Limit::AtMost(0.0),
    )
    .expect_err("a walk that leaves its window must not answer")
    .to_string();
    assert!(failed.contains("inside the volume"), "{failed}");
    assert!(failed.contains("outside the window"), "{failed}");
    assert!(failed.contains("stated maximum"), "{failed}");
}

/// The same walk with the operand reach **understated**, to show that the guard
/// fires through the executor and not only in a hand-built call.
///
/// Everything about it is the real op — the same sequence, the same rows, the
/// same kernel — except that it tells the plan it reads the operand at the voxel
/// it writes. The planner believes it, the halo comes out zero, and the first
/// row whose walk leaves its block's core is refused. **A run that answered here
/// would have produced a complete table of plausible, short distances**, which
/// is the failure this whole arrangement exists to make impossible.
struct UnderdeclaredWalkOp(OffsetWalkOp);

impl FragmentOp for UnderdeclaredWalkOp {
    fn name(&self) -> &'static str {
        "underdeclared-walk"
    }

    fn inputs(&self) -> Vec<blockflow::fragment::FragmentInput> {
        self.0.inputs()
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        self.0.outputs()
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<blockflow::op::SourceInput> {
        vec![blockflow::op::SourceInput::voxelwise(0)]
    }

    fn seam_fold(&self) -> Option<blockflow::fragment::SeamFold> {
        self.0.seam_fold()
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        self.0.apply(at)
    }

    fn apply_with(
        &self,
        at: &BlockView<'_>,
        sources: blockflow::fragment::SourceBlocks<'_>,
    ) -> Result<BlockOutput> {
        self.0.apply_with(at, sources)
    }
}

#[test]
fn a_short_operand_reach_is_refused_by_the_run_rather_than_answered() {
    let points = PointsOp {
        stream: "rows.points".to_string(),
    };
    let streams = RowStreams::new(
        "rows.points",
        1,
        "rows.walked",
        Lifecycle::DeleteOnExit,
        coordinate_schema(),
    )
    .expect("two different streams");
    let walk = UnderdeclaredWalkOp(
        OffsetWalkOp::new(
            "walk",
            streams,
            0,
            "distance",
            sequence(),
            Limit::AtMost(0.0),
            NOT_FOUND,
        )
        .expect("a distinguishable not-found value"),
    );
    let plan = append_fragment_phase(one_pixel_phase([4, 4, 4]), &points).expect("a row phase");
    let plan = append_fragment_phase(plan, &walk).expect("a walk phase");
    // The plan really is the short one, or the test would be asserting nothing.
    assert_eq!(plan.phases[2].halo, [0, 0, 0]);
    let env = ArrayEnvironment::new(fixture(), plan.n_phases(), [4, 4, 4]).expect("an environment");
    let failed = execute_phases(
        "short",
        &identity_workflow(),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&points),
            PhaseWork::Fragments(&walk),
        ],
    )
    .expect_err("a walk that leaves its window must not answer")
    .to_string();
    assert!(failed.contains("outside the window"), "{failed}");
    assert!(failed.contains("stated maximum"), "{failed}");
}

/// A row on the volume's face. Offsets that leave the volume name nothing and
/// are skipped; the walk goes on past them. That is not the same as stopping,
/// and the row is placed so the two answers differ: it reports 3, and would
/// report 1 if leaving the volume had counted as a stop.
#[test]
fn a_row_on_the_volumes_face_walks_past_what_is_not_there() {
    let measured = distances(&run([16, 16, 16], None).expect("one block runs"));
    let (at, distance) = measured[0];
    assert_eq!(at, [0, 8, 8]);
    assert_eq!(distance, 3.0);
    let sequence = sequence();
    let leaving = sequence
        .offsets()
        .iter()
        .position(|offset| *offset == [-1, 0, 0])
        .expect("the sequence holds an offset that leaves the volume from x = 0");
    assert_eq!(sequence.distances()[leaving], 1.0);
    assert!(
        sequence.distances()[leaving] < distance,
        "the fixture must have an out-of-volume offset earlier than the stop, or it cannot \
         tell skipping from stopping"
    );
}

/// A walk that runs off the end of the sequence gets the stated not-found value,
/// and the op refuses a value that could be mistaken for a distance.
#[test]
fn a_walk_that_never_stops_says_so_in_a_value_no_distance_can_be() {
    let measured = distances(&run([16, 16, 16], None).expect("one block runs"));
    let (at, distance) = measured[6];
    assert_eq!(at, [9, 7, 7]);
    assert_eq!(distance, NOT_FOUND);
    assert!(
        !sequence().reports(NOT_FOUND),
        "the not-found value must be one no offset reports"
    );
    let streams = RowStreams::new(
        "rows.points",
        1,
        "rows.walked",
        Lifecycle::DeleteOnExit,
        coordinate_schema(),
    )
    .expect("two different streams");
    let collides = OffsetWalkOp::new(
        "walk",
        streams,
        0,
        "distance",
        sequence(),
        Limit::AtMost(0.0),
        1.0,
    );
    let message = match collides {
        Ok(_) => panic!("1.0 is a distance the sequence reports"),
        Err(error) => error.to_string(),
    };
    assert!(message.contains("indistinguishable"), "{message}");
}

/// An empty offset set is refused rather than run. Every row's walk would stop
/// at nothing, so the column would be the not-found value everywhere: complete,
/// well-formed, and a measurement of nothing.
#[test]
fn an_empty_offset_set_is_refused() {
    let message = match OffsetSequence::from_offsets(Vec::new(), [1.0, 1.0, 1.0]) {
        Ok(_) => panic!("a walk over no offsets is not a walk"),
        Err(error) => error.to_string(),
    };
    assert!(message.contains("at least one offset"), "{message}");
}

/// **The exactness precondition, in the direction that fails.** A maximum that
/// differs between axes normalises each axis by its own maximum while the
/// distance does not, so one ordering key holds offsets a whole unit apart. In
/// that configuration even the distance is a function of the tie order, and the
/// sequence is refused at construction rather than producing a number that is
/// reproducible on one machine only.
#[test]
fn a_sequence_whose_equal_keys_report_different_distances_is_refused() {
    let message = match OffsetSequence::ellipsoid([3, 3, 1], [1.0, 1.0, 1.0]) {
        Ok(_) => panic!("this configuration cannot give an exact distance"),
        Err(error) => error.to_string(),
    };
    assert!(message.contains("ordering key"), "{message}");
    assert!(message.contains("not reproducible"), "{message}");
    // The check is a rule about the two orders agreeing, not a ban on
    // anisotropy: where `spacing * maximum` is equal on every axis they agree
    // and the sequence is accepted.
    let accepted = OffsetSequence::ellipsoid([4, 2, 2], [1.0, 2.0, 2.0])
        .expect("the key and the distance order this set the same way");
    assert_eq!(accepted.maximum(), [4, 2, 2]);
}

/// The schema is the input's with one column appended, and a name the rows
/// already carry is refused.
#[test]
fn the_walk_appends_one_column_and_refuses_a_name_already_used() {
    let schema =
        Schema::new(vec![blockflow::table::Column::f64("distance")]).expect("one named column");
    let streams = RowStreams::new(
        "rows.points",
        1,
        "rows.walked",
        Lifecycle::DeleteOnExit,
        schema,
    )
    .expect("two different streams");
    let message = match OffsetWalkOp::new(
        "walk",
        streams,
        0,
        "distance",
        sequence(),
        Limit::AtMost(0.0),
        NOT_FOUND,
    ) {
        Ok(_) => panic!("two columns of one name are ambiguous"),
        Err(error) => error.to_string(),
    };
    assert!(message.contains("already have"), "{message}");
}
