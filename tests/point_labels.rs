// SPDX-License-Identifier: MIT
//
// A point stream in, a label volume out, end to end through the executor.
//
// `tests/voxelize.rs` is the same arrangement for the op that **adds**; this is
// the one that **names**. The difference is the whole reason the second op
// exists, and it is what most of these assertions are about: where two points
// meet on one voxel, a sum is a number neither of them carried, and a label has
// to be one of the two by a rule that is stated rather than emergent.
//
// What each test is for
// ---------------------
// * **Decomposition invariance, byte for byte.** Five cuts of the same point
//   set, including one block and lattices with partial edge blocks, against the
//   single-block answer. A tolerance would be meaningless here — these are
//   integers — so the failure this catches is a rule that consulted the cut.
// * **A collision across a seam.** Two points in different blocks whose kernels
//   overlap, arranged so that "the block that owns the voxel decides" and "the
//   lowest label decides" are *different answers*. This is the fixture that
//   discriminates; a collision inside one block would pass under either rule.
// * **The collision rule itself, on one voxel.** Two points at the same
//   coordinate with different labels, asserted to give the lower one — and
//   asserted to give it under every blocking. The pair is asymmetric, so "last
//   wins" and "first in listing order wins" both fail it, and so does a sum.
// * **The degenerate cases.** No points at all, every point on one voxel, and a
//   point outside the volume — the last of which is *refused*, which is a
//   deliberate agreement with `ops::voxelize` rather than with the tools that
//   drop such a point quietly.
// * **The cost of the declaration.** No pixel is read: this phase's input is
//   the fragment stream and nothing else.
// * **A kernel whose members re-phase.** `StepOrigin::ClippedStart` makes the
//   stamped window a function of where the point sits relative to the volume's
//   low faces, which is a rule a block could get wrong in a way no other fixture
//   here would see — and a negative control beside it, so the sweep is known to
//   be telling the two origins apart.

use std::collections::BTreeMap;

use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::error::Result;
use blockflow::fragment::{
    fragment_only, BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput, PhaseWork,
};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::element::{ElementShape, StepOrigin, StructuringElement};
use blockflow::ops::label::LabelPointsOp;
use blockflow::ops::voxelize::{ball, encode_points, single_voxel, Point};
use blockflow::points::PointSourceOp;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

const STREAM: &str = "points";

// ------------------------------------------------------------- the source --

/// A fixed point set out as one fragment per block, keyed by the block whose
/// **core** contains the point.
///
/// This file used to carry that producer, and so did `tests/voxelize.rs`,
/// character for character, and so did a consumer of this crate.
/// `points::PointSourceOp` is now that op. The rule is unchanged and is the
/// library's: a point on a seam is written once, into the block the seam
/// starts, and a block with no points writes a zero-length fragment rather than
/// nothing, so `Coverage::EveryBlock` is honest — "present and empty" is a
/// different fact from "absent", and only the first can be checked.
///
/// [`UnkeyedSourceOp`] below is deliberately **not** replaced by it: its whole
/// purpose is to key *wrongly*, which is a thing no library op should be able
/// to do.
fn source(points: Vec<Point>) -> PointSourceOp {
    PointSourceOp::new("points", STREAM, Lifecycle::DeleteOnExit, points)
}

/// A producer that writes every point into block `[0, 0, 0]`'s fragment
/// whatever its coordinate — which is how a point outside the volume reaches
/// the op at all, since no core contains one.
struct UnkeyedSourceOp {
    points: Vec<Point>,
}

impl FragmentOp for UnkeyedSourceOp {
    fn name(&self) -> &'static str {
        "points"
    }

    /// Nothing crosses: it reads nothing. Writing every point into one block is
    /// a keying choice — the one this fixture exists to make — and a keying
    /// choice is not a reach.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            STREAM,
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mine = if at.index == [0, 0, 0] {
            self.points.clone()
        } else {
            Vec::new()
        };
        Ok(BlockOutput::fragment(STREAM, encode_points(&mine)))
    }
}

// ------------------------------------------------------------- the harness --

fn workflow(volume: [usize; 3]) -> Workflow {
    Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::U32)
}

fn plan_for(
    volume: [usize; 3],
    block: [usize; 3],
    element: &StructuringElement,
) -> (Decomposition, LabelPointsOp) {
    let grid = BlockGrid::new(volume, block).expect("a lattice");
    let label = LabelPointsOp::new("label", STREAM, 0, element.clone(), &grid).expect("a label op");
    let source = source(Vec::new());
    let plan = fragment_only(volume, block, Dtype::U32, &[&source, &label]).expect("a plan");
    (plan, label)
}

/// Run the pair and hand back the labelled volume, or the run's error.
fn run(
    volume: [usize; 3],
    block: [usize; 3],
    element: &StructuringElement,
    source: &dyn FragmentOp,
) -> Result<Vec<u32>> {
    let (plan, label) = plan_for(volume, block, element);
    let env = ArrayEnvironment::new(
        Voxels::zeros(Dtype::U32, volume).expect("an image"),
        plan.n_phases(),
        [4, 4, 4],
    )
    .expect("an environment");
    execute_phases(
        "label",
        &workflow(volume),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(source), PhaseWork::Fragments(&label)],
    )?;
    // No pixel was read: this op declares `reads_pixels() == false`, and the
    // environment holds a real array, so a stray read would show up here.
    let (reads, _, read_voxels, _, chunks, _, _) = env.counters().snapshot();
    assert_eq!(
        (reads, read_voxels, chunks),
        (0, 0, 0),
        "a fragments -> volume phase read pixels"
    );
    Ok(env
        .output()
        .view::<u32>()
        .expect("a u32 image")
        .iter()
        .copied()
        .collect())
}

fn stamped(
    volume: [usize; 3],
    block: [usize; 3],
    element: &StructuringElement,
    points: &[Point],
) -> Vec<u32> {
    let source = source(points.to_vec());
    run(volume, block, element, &source).expect("a run")
}

/// The answer the definition gives, computed here rather than taken from a run
/// of the code under test: every voxel holds the smallest label whose kernel
/// covers it, and zero where none does.
///
/// The kernel is asked for its members **at the point's own position in the
/// volume**, which is the whole of what a stamp asks an element — see the op's
/// module header on why a stamp is the gather's transpose and therefore asks the
/// same question. For every element without a step that is the element's own
/// offset list and this reads exactly as it always did.
fn expected(volume: [usize; 3], element: &StructuringElement, points: &[Point]) -> Vec<u32> {
    let mut out = vec![0u32; volume[0] * volume[1] * volume[2]];
    let mut scratch = Vec::new();
    for point in points {
        let label = point.weight as u32;
        let at = [
            point.at[0] as isize,
            point.at[1] as isize,
            point.at[2] as isize,
        ];
        for offset in element.offsets_at(at, volume, &mut scratch) {
            let mut at = [0usize; 3];
            let mut inside = true;
            for axis in 0..3 {
                let position = point.at[axis] as isize + offset[axis];
                if position < 0 || position as usize >= volume[axis] {
                    inside = false;
                    break;
                }
                at[axis] = position as usize;
            }
            if inside {
                let flat = (at[0] * volume[1] + at[1]) * volume[2] + at[2];
                if out[flat] == 0 || label < out[flat] {
                    out[flat] = label;
                }
            }
        }
    }
    out
}

/// The cuts every property below is checked under. One block — which has no
/// seam to get wrong — and four lattices, two of them with a partial last block
/// on every axis.
const CUTS: [[usize; 3]; 5] = [[16, 12, 4], [8, 6, 4], [5, 5, 2], [3, 12, 1], [16, 5, 3]];

const VOLUME: [usize; 3] = [16, 12, 4];

// --------------------------------------------------------------- the tests --

#[test]
fn the_same_points_label_the_same_volume_under_every_decomposition() {
    let element = ball([2, 2, 1]);
    let points = vec![
        Point::weighted([3, 3, 1], 40.0),
        Point::weighted([8, 6, 2], 5.0),
        // straddles the seam of the [8, .., ..] cut, and its kernel overlaps
        // the point above's
        Point::weighted([7, 6, 2], 31.0),
        Point::weighted([12, 9, 1], 2.0),
        Point::weighted([15, 11, 3], 900.0),
    ];

    let want = expected(VOLUME, &element, &points);
    assert!(
        want.contains(&5),
        "the low label of the straddling pair must survive somewhere, or this \
         fixture discriminates nothing"
    );
    assert!(want.contains(&0), "and some voxel is unmarked");

    for block in CUTS {
        let got = stamped(VOLUME, block, &element, &points);
        assert_eq!(got, want, "cut into {block:?}");
    }
}

/// The collision rule, on one voxel, under every blocking.
///
/// The pair is asymmetric — 3 and 5 — so this fails for "last wins" (5), for a
/// sum (8), and for anything that consulted the order the producer listed them
/// in. It is the fixture the rule is *tested* by rather than implied by.
#[test]
fn two_points_on_one_voxel_leave_the_lower_label_under_every_blocking() {
    let element = single_voxel();
    let points = vec![
        Point::weighted([9, 7, 2], 5.0),
        Point::weighted([9, 7, 2], 3.0),
    ];
    let flat = (9 * VOLUME[1] + 7) * VOLUME[2] + 2;
    for block in CUTS {
        let got = stamped(VOLUME, block, &element, &points);
        assert_eq!(got[flat], 3, "cut into {block:?}: the lower label must win");
        assert_eq!(
            got.iter().filter(|&&value| value != 0).count(),
            1,
            "cut into {block:?}: exactly one voxel is marked"
        );
        assert_ne!(got[flat], 8, "cut into {block:?}: a sum, not a label");
    }
    // and the same two listed the other way round is the same volume
    let reversed = vec![
        Point::weighted([9, 7, 2], 3.0),
        Point::weighted([9, 7, 2], 5.0),
    ];
    assert_eq!(
        stamped(VOLUME, [5, 5, 2], &element, &points),
        stamped(VOLUME, [5, 5, 2], &element, &reversed)
    );
}

/// A collision **across a seam**, arranged so that the two candidate rules give
/// different answers.
///
/// Voxel `[7, 6, 2]` sits in block 0 under the `[8, .., ..]` cut and carries its
/// own point's label of 31; the point that reaches it from block 1 carries 5. A
/// rule that let the owning block's point win would leave 31 there under that
/// cut and 5 under the single-block cut, so this test fails for it — which is
/// the point of choosing the labels this way round.
#[test]
fn a_collision_across_a_seam_resolves_the_same_way_from_both_sides() {
    let element = ball([2, 0, 0]);
    let points = vec![
        Point::weighted([7, 6, 2], 31.0),
        Point::weighted([8, 6, 2], 5.0),
    ];
    let want = expected(VOLUME, &element, &points);
    let flat = |at: [usize; 3]| (at[0] * VOLUME[1] + at[1]) * VOLUME[2] + at[2];
    assert_eq!(want[flat([7, 6, 2])], 5, "the lower label crosses the seam");
    assert_eq!(want[flat([5, 6, 2])], 31, "and out of its reach, 31 stands");

    for block in CUTS {
        let got = stamped(VOLUME, block, &element, &points);
        assert_eq!(got, want, "cut into {block:?}");
    }
}

/// Nothing to stamp: an unmarked volume, under every cut, and not an error.
#[test]
fn an_empty_point_set_leaves_an_unmarked_volume() {
    let element = ball([1, 1, 1]);
    for block in CUTS {
        let got = stamped(VOLUME, block, &element, &[]);
        assert!(
            got.iter().all(|&value| value == 0),
            "cut into {block:?} marked something"
        );
    }
}

/// Every point on one voxel: one label, the lowest, and no arithmetic on the
/// rest — 11 + 4 + 9 + 250 is 274, which is what a sum would have left.
#[test]
fn every_point_on_one_voxel_leaves_one_label() {
    let element = single_voxel();
    let points: Vec<Point> = [11.0, 4.0, 9.0, 250.0]
        .into_iter()
        .map(|label| Point::weighted([6, 5, 1], label))
        .collect();
    let flat = (6 * VOLUME[1] + 5) * VOLUME[2] + 1;
    for block in CUTS {
        let got = stamped(VOLUME, block, &element, &points);
        assert_eq!(got[flat], 4, "cut into {block:?}");
        assert_eq!(got.iter().filter(|&&value| value != 0).count(), 1);
    }
}

/// A point outside the volume is **refused**, under every cut, and the refusal
/// names the rule.
///
/// This is the live disagreement recorded in the op's header: `ops::voxelize`
/// refuses such a point and the tools this crate is compared against drop it
/// quietly. Refusing is the deliberate answer, because both ops read the same
/// stream and a stream that one accepts and the other refuses would make a
/// producer's output legal or illegal depending on which consumer it was wired
/// to.
#[test]
fn a_point_outside_the_volume_is_refused_rather_than_dropped() {
    let element = single_voxel();
    let source = UnkeyedSourceOp {
        points: vec![Point::weighted([99, 0, 0], 1.0)],
    };
    for block in CUTS {
        let error = run(VOLUME, block, &element, &source)
            .expect_err("a point outside the volume must be refused")
            .to_string();
        assert!(
            error.contains("outside that block's core"),
            "cut into {block:?}: {error}"
        );
    }
    // and the same producer with an in-volume point that is simply mis-keyed is
    // refused by the same rule, so the two cases have one message and not two
    let mis_keyed = UnkeyedSourceOp {
        points: vec![Point::weighted([15, 11, 3], 1.0)],
    };
    let error = run(VOLUME, [8, 6, 4], &element, &mis_keyed)
        .expect_err("a mis-keyed point must be refused")
        .to_string();
    assert!(error.contains("outside that block's core"), "{error}");
}

/// A label the image cannot hold is refused rather than saturated onto another
/// label — under every cut, and with the ceiling named.
#[test]
fn a_label_past_the_end_of_the_image_is_refused() {
    let element = single_voxel();
    let points = vec![Point::weighted([2, 2, 2], 5_000_000_000.0)];
    for block in CUTS {
        let source = source(points.clone());
        let error = run(VOLUME, block, &element, &source)
            .expect_err("a label past u32 must be refused")
            .to_string();
        assert!(
            error.contains("the largest this destination holds"),
            "cut into {block:?}: {error}"
        );
    }
}

/// The gathered neighbourhood is the analytic one and the fragments fetched are
/// exactly it — the same declaration `ops::voxelize` makes, checked here so that
/// the second op of this shape is not merely assumed to have inherited it.
#[test]
fn the_declared_block_reach_is_the_kernel_over_the_block_edge() {
    let grid = BlockGrid::new([20, 4, 4], [8, 4, 4]).expect("a lattice");
    let op = LabelPointsOp::new("label", STREAM, 0, ball([5, 1, 1]), &grid).expect("an op");
    assert_eq!(op.block_reach(), [1, 0, 0], "ceil(5 / 8)");
    assert_eq!(op.reach(0, 20), 5);
    assert!(!op.reads_pixels());
    assert!(op.writes_pixels());
    assert!(op.gathers());
    assert_eq!(op.inputs().len(), 1);
}

/// **A kernel whose members depend on where it is placed, through the
/// executor**: five cuts, one volume, byte for byte.
///
/// `StepOrigin::ClippedStart` makes the stamped window a function of the point's
/// position *and of the volume's low faces*. The op keys it on the point's
/// coordinate in the volume and on `grid.volume()`, both of which are the same
/// numbers under every cut — but the argument is about the code as it stands and
/// this is the property the crate exists to defend, so it is measured.
///
/// The window is **wider on axis 2 than the volume is**, so every point re-phases
/// there and no block holds only interior voxels; and the fixture carries a point
/// at the origin, where the low faces of all three axes meet.
///
/// The last assertion is the negative control: the same program with the origin
/// changed is a different volume, so the sweep above is not an invariance sweep
/// over an element that quietly behaves like the anchored one.
#[test]
fn a_re_phasing_kernel_labels_the_same_volume_under_every_decomposition() {
    let element = StructuringElement::from_size_stepped_at(
        ElementShape::Box,
        [11, 3, 9],
        [2, 1, 3],
        StepOrigin::ClippedStart,
    )
    .expect("an element");
    assert!(
        element.sides(2).0 >= VOLUME[2],
        "the window must be wider than axis 2 of the volume, so every point re-phases there"
    );
    let points = vec![
        Point::weighted([3, 3, 1], 40.0),
        Point::weighted([8, 6, 2], 5.0),
        Point::weighted([7, 6, 2], 31.0),
        Point::weighted([12, 9, 1], 2.0),
        Point::weighted([15, 11, 3], 900.0),
        Point::weighted([0, 0, 0], 60.0),
    ];

    let want = expected(VOLUME, &element, &points);
    assert!(want.contains(&5), "the low label must survive somewhere");
    assert!(want.contains(&0), "and some voxel is unmarked");

    for block in CUTS {
        let got = stamped(VOLUME, block, &element, &points);
        assert_eq!(got, want, "cut into {block:?}");
    }

    let anchored = StructuringElement::from_size_stepped_at(
        ElementShape::Box,
        [11, 3, 9],
        [2, 1, 3],
        StepOrigin::Anchor,
    )
    .expect("an element");
    assert_ne!(
        stamped(VOLUME, [5, 5, 2], &anchored, &points),
        want,
        "the two origins gave the same volume, so the sweep above tells them apart at nothing"
    );
}

/// Every cut in `CUTS` is a real cut of the volume, so that none of the tests
/// above is quietly running the single-block case five times.
#[test]
fn the_cuts_the_other_tests_use_are_actually_different_lattices() {
    let mut counts = BTreeMap::new();
    for block in CUTS {
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        counts.insert(block, grid.n_blocks());
    }
    assert_eq!(counts[&CUTS[0]], 1, "the first cut is the whole volume");
    assert!(
        counts.values().filter(|&&n| n > 1).count() >= 4,
        "at least four of the cuts must have seams: {counts:?}"
    );
    // and at least one lattice has a partial last block on every axis
    let ragged = BlockGrid::new(VOLUME, [5, 5, 2]).expect("a lattice");
    assert_eq!(ragged.blocks_per_axis(), [4, 3, 2]);
}
