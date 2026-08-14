// SPDX-License-Identifier: MIT
//
// Voxelization end to end: a phase that writes points, a phase that renders
// them, and the properties that separate a correct rendering from one that
// merely looks plausible.
//
// Every assertion here is stated against a number that comes from the
// *definition* — the point count, the element's member count, the analytic
// neighbourhood — or against the single-block answer, which is the one
// configuration with no seams in it at all. Nothing is compared against a
// recorded output of this code.
//
// What each test is for
// ---------------------
// * **Decomposition invariance, byte for byte.** Four cuts of the same point
//   set, including one block and a lattice with a partial edge block, and the
//   volumes must agree bit for bit — not to a tolerance. A tolerance would pass
//   for an implementation whose answer depended on the gather order, which is
//   the failure this op's accumulation order exists to prevent.
// * **A seam counts a point once.** Points sitting exactly on block boundaries,
//   with a one-voxel kernel, so the total mass is the point count and any
//   double-counting or dropped point shows up as an integer.
// * **A kernel that straddles a seam.** The contribution has to appear on both
//   sides, which is what the block reach buys, and the two sides together have
//   to equal the whole-volume answer.
// * **A short reach, both kinds.** The pixel-side one fires: a forced halo
//   under this op's voxel reach loses the interior cores and the tiling check
//   says so. The fragment-side one does *not* fire, and that is recorded here
//   as a test rather than as a comment — see
//   `under_declaring_the_block_reach_is_wrong_and_no_framework_guard_sees_it`.
// * **The cost of the declaration, measured.** No pixel is read, and the number
//   of fragments fetched is the analytic neighbourhood size and not one more.

use std::collections::BTreeMap;
use std::sync::Arc;

use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::error::Result;
use blockflow::fragment::{
    check_phase_work, fragment_only, neighbourhood_size, BlockOutput, BlockView, Coverage,
    FragmentOp, FragmentOutput, PhaseWork,
};
use blockflow::geometry::BlockGrid;
use blockflow::log::{Event, ExecutionLog};
use blockflow::op::Chain;
use blockflow::ops::element::StructuringElement;
use blockflow::ops::voxelize::{
    ball, encode_points, single_voxel, Point, VoxelizeOp, WORDS_PER_POINT,
};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

const STREAM: &str = "points";

// ------------------------------------------------------------- the source --

/// Writes a fixed point set out as one fragment per block: the producer half of
/// the pair, and the thing a real detector would be.
///
/// It keys each point by the block whose **core** contains it, which is the rule
/// `ops::voxelize` states and the reason a point on a seam is written once. A
/// block with no points writes a zero-length fragment rather than nothing, which
/// is what `Coverage::EveryBlock` means and why it is declared here: "present
/// and empty" is a different fact from "absent", and the coverage guard can only
/// check the first.
struct PointSourceOp {
    points: Vec<Point>,
}

impl PointSourceOp {
    fn new(points: Vec<Point>) -> Self {
        Self { points }
    }
}

impl FragmentOp for PointSourceOp {
    fn name(&self) -> &'static str {
        "points"
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            STREAM,
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let edge = at.grid.block();
        let mine: Vec<Point> = self
            .points
            .iter()
            .copied()
            .filter(|point| (0..3).all(|axis| point.at[axis] / edge[axis] == at.index[axis]))
            .collect();
        Ok(BlockOutput::fragment(STREAM, encode_points(&mine)))
    }
}

// ------------------------------------------------------------- the harness --

fn workflow(volume: [usize; 3]) -> Workflow {
    Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::F64)
}

fn plan_for(
    volume: [usize; 3],
    block: [usize; 3],
    element: &StructuringElement,
) -> (Decomposition, VoxelizeOp) {
    let grid = BlockGrid::new(volume, block).expect("a lattice");
    let voxelize =
        VoxelizeOp::new("voxelize", STREAM, 0, element.clone(), &grid).expect("a voxelize op");
    let source = PointSourceOp::new(Vec::new());
    let plan = fragment_only(volume, block, Dtype::F64, &[&source, &voxelize]).expect("a plan");
    (plan, voxelize)
}

/// Run the pair and hand back the rendered volume, with the store's event log.
fn render_with_log(
    volume: [usize; 3],
    block: [usize; 3],
    element: &StructuringElement,
    points: &[Point],
) -> (Vec<f64>, Arc<ExecutionLog>) {
    let (plan, voxelize) = plan_for(volume, block, element);
    let source = PointSourceOp::new(points.to_vec());
    let env = ArrayEnvironment::new(
        Voxels::zeros(Dtype::F64, volume).expect("a level"),
        plan.n_phases(),
        [4, 4, 4],
    )
    .expect("an environment");
    let log = Arc::new(ExecutionLog::new());
    env.sidecars().expect("a store").attach(log.clone());
    execute_phases(
        "voxelize",
        &workflow(volume),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Fragments(&source),
            PhaseWork::Fragments(&voxelize),
        ],
    )
    .expect("a run");
    // No pixel was read: this op declares `reads_pixels() == false`, and the
    // environment holds a real array, so a stray read would show up here.
    let (reads, _, read_voxels, _, chunks, _, _) = env.counters().snapshot();
    assert_eq!(
        (reads, read_voxels, chunks),
        (0, 0, 0),
        "a fragments -> volume phase read pixels"
    );
    let output = env.output();
    let rendered = output
        .view::<f64>()
        .expect("an f64 level")
        .iter()
        .copied()
        .collect();
    (rendered, log)
}

fn render(
    volume: [usize; 3],
    block: [usize; 3],
    element: &StructuringElement,
    points: &[Point],
) -> Vec<f64> {
    render_with_log(volume, block, element, points).0
}

fn identical(left: &[f64], right: &[f64], what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: different sizes");
    for (index, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: voxel {index} is {a} against {b}"
        );
    }
}

/// The mass a point set deposits, from the definition: one weight per kernel
/// member that lands inside the volume.
fn expected_mass(volume: [usize; 3], element: &StructuringElement, points: &[Point]) -> f64 {
    let mut total = 0.0;
    for point in points {
        for offset in element.offsets() {
            let inside = (0..3).all(|axis| {
                let position = point.at[axis] as isize + offset[axis];
                position >= 0 && (position as usize) < volume[axis]
            });
            if inside {
                total += point.weight;
            }
        }
    }
    total
}

// --------------------------------------------------------------- the tests --

/// Four cuts, one answer, bit for bit — including the single block, which has
/// no seam to get wrong, and a lattice whose last block on every axis is
/// partial.
#[test]
fn the_same_points_render_the_same_volume_under_every_decomposition() {
    let volume = [20usize, 12, 6];
    let element = ball([2, 2, 1]);
    let points = vec![
        Point::unit([4, 4, 2]),
        Point::unit([8, 3, 3]),
        Point::weighted([9, 8, 2], 2.0),
        Point::weighted([15, 6, 3], 0.5),
        // a pair whose kernels overlap, so some voxel takes two contributions
        Point::unit([11, 6, 2]),
        Point::unit([12, 6, 2]),
    ];

    let whole = render(volume, volume, &element, &points);
    assert_eq!(
        whole.iter().sum::<f64>(),
        expected_mass(volume, &element, &points),
        "the single-block answer does not hold the mass the definition says"
    );

    for block in [[8usize, 4, 3], [20, 12, 3], [7, 5, 4], [4, 12, 6]] {
        let grid = BlockGrid::new(volume, block).expect("a lattice");
        let rendered = render(volume, block, &element, &points);
        identical(&whole, &rendered, &format!("cut into {block:?}"));
        assert!(
            grid.n_blocks() > 1,
            "cut {block:?} is not a cut at all: {} block(s)",
            grid.n_blocks()
        );
    }

    // and one of those cuts really does leave a partial block on every axis
    let ragged = BlockGrid::new(volume, [7, 5, 4]).expect("a lattice");
    assert_eq!(ragged.blocks_per_axis(), [3, 3, 2]);
    for axis in 0..3 {
        let counts = ragged.blocks_per_axis()[axis];
        let edge = ragged.block()[axis];
        assert!(
            volume[axis] - (counts - 1) * edge < edge,
            "axis {axis} has no partial block"
        );
    }
}

/// A point exactly on a seam belongs to exactly one core, so the total is the
/// point count and not one more. With a one-voxel kernel any double-count or
/// dropped point is a whole unit.
#[test]
fn a_point_on_a_block_boundary_is_counted_exactly_once() {
    let volume = [24usize, 8, 4];
    let element = single_voxel();
    // every point sits on a seam of at least one of the cuts below
    let points = vec![
        Point::unit([0, 0, 0]),
        Point::unit([8, 0, 0]),
        Point::unit([16, 4, 2]),
        Point::unit([6, 4, 0]),
        Point::unit([12, 2, 2]),
        Point::unit([23, 7, 3]),
    ];
    let expected = points.len() as f64;

    for block in [[24usize, 8, 4], [8, 8, 4], [6, 4, 2], [5, 3, 4]] {
        let rendered = render(volume, block, &element, &points);
        assert_eq!(
            rendered.iter().sum::<f64>(),
            expected,
            "cut into {block:?} counted {} of {expected} point(s)",
            rendered.iter().sum::<f64>()
        );
        assert_eq!(
            rendered.iter().filter(|&&value| value != 0.0).count(),
            points.len(),
            "cut into {block:?} put the mass in the wrong number of voxels"
        );
    }
}

/// The test a short block reach fails: a point whose kernel crosses a seam must
/// deposit on both sides of it, and the two sides together must be the
/// whole-volume answer.
#[test]
fn a_kernel_that_straddles_a_seam_contributes_to_both_sides() {
    let volume = [24usize, 8, 8];
    let element = ball([3, 3, 3]);
    // one voxel short of the seam at 8, so the kernel reaches into block 1
    let points = vec![Point::unit([7, 4, 4])];

    let whole = render(volume, volume, &element, &points);
    let cut = render(volume, [8, 8, 8], &element, &points);
    identical(&whole, &cut, "a kernel across a seam");

    // and the straddle is real: both sides hold part of the mass
    let plane = |rendered: &[f64], predicate: &dyn Fn(usize) -> bool| -> f64 {
        let mut total = 0.0;
        for (flat, value) in rendered.iter().enumerate() {
            let x = flat / (volume[1] * volume[2]);
            if predicate(x) {
                total += value;
            }
        }
        total
    };
    let below = plane(&cut, &|x| x < 8);
    let above = plane(&cut, &|x| x >= 8);
    assert!(below > 0.0 && above > 0.0, "{below} below, {above} above");
    assert_eq!(below + above, element.len() as f64);
}

/// The pixel-side guard, provoked the way this crate provokes guards that have
/// never been seen to fire: force a halo shorter than the op's voxel reach and
/// the valid regions stop tiling.
#[test]
fn a_halo_shorter_than_the_kernel_radius_fails_the_tiling_check() {
    let volume = [24usize, 8, 8];
    let element = ball([3, 3, 3]);
    let (plan, _) = plan_for(volume, [8, 8, 8], &element);
    plan.check().expect("the derived plan tiles");
    assert_eq!(plan.phases[1].reach, [3, 3, 3]);

    // one voxel short of the reach on the split axis
    let short = plan.with_forced_halo([2, 8, 8]);
    let error = short
        .check()
        .expect_err("a halo under the kernel radius must fail the tiling check")
        .to_string();
    assert!(error.contains("lost part of their core"), "{error}");
}

/// The fragment-side guard, which is a different check on a different number:
/// `check_phase_work` refuses a phase whose halo is too small to make the
/// neighbours whose fragments it reads into dependencies of its own tasks.
#[test]
fn a_halo_too_short_for_the_declared_block_reach_is_refused_before_the_run() {
    let volume = [40usize, 8, 8];
    let element = ball([9, 0, 0]);
    let (plan, voxelize) = plan_for(volume, [8, 8, 8], &element);
    assert_eq!(voxelize.block_reach(), [2, 0, 0], "ceil(9 / 8)");
    let source = PointSourceOp::new(Vec::new());
    let work = [
        PhaseWork::Fragments(&source),
        PhaseWork::Fragments(&voxelize),
    ];
    check_phase_work(&plan, &work).expect("the derived plan is consistent");

    // a halo of one block, against a reach of two
    let short = plan.with_forced_halo([8, 8, 8]);
    let error = check_phase_work(&short, &work)
        .expect_err("a halo under the fragment reach must be refused")
        .to_string();
    assert!(error.contains("reaches 2 block(s) on axis 0"), "{error}");
    assert!(error.contains("short halo"), "{error}");
}

/// **A finding, pinned as a test.** An op that under-declares its *block* reach
/// while declaring its voxel reach honestly produces a wrong answer that no
/// guard in the framework can see: the halo still covers the voxel reach, so the
/// valid regions tile and `Decomposition::check` passes; the declared block
/// reach and the halo derived from it agree with each other, so
/// `check_phase_work` passes; and the fragments that were never fetched leave no
/// trace anywhere except in the pixels.
///
/// That is why `ops::voxelize` derives the block reach from the kernel with no
/// field to set it. The check is that the derivation is right, not that the
/// framework will catch it being wrong.
#[test]
fn under_declaring_the_block_reach_is_wrong_and_no_framework_guard_sees_it() {
    /// A voxelize op with its block reach shortened by one, and everything else
    /// identical.
    struct ShortReach {
        inner: VoxelizeOp,
        reach: [usize; 3],
    }

    impl FragmentOp for ShortReach {
        fn name(&self) -> &'static str {
            "short-reach"
        }

        fn reach(&self, axis: usize, volume_len: usize) -> usize {
            self.inner.reach(axis, volume_len)
        }

        fn writes_pixels(&self) -> bool {
            true
        }

        fn inputs(&self) -> Vec<blockflow::fragment::FragmentInput> {
            vec![blockflow::fragment::FragmentInput::own(STREAM, 0).with_reach(self.reach)]
        }

        fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
            self.inner.apply(at)
        }
    }

    let volume = [40usize, 8, 8];
    let block = [8usize, 8, 8];
    let element = ball([9, 0, 0]);
    // a point at the lower corner of block 2, whose kernel of radius 9 reaches
    // back to voxel 7 — a voxel of block 0, two block indices away
    let points = vec![Point::unit([16, 4, 4])];
    let correct = render(volume, block, &element, &points);
    assert_eq!(
        correct.iter().sum::<f64>(),
        expected_mass(volume, &element, &points)
    );

    let grid = BlockGrid::new(volume, block).expect("a lattice");
    let honest = VoxelizeOp::new("voxelize", STREAM, 0, element.clone(), &grid).expect("an op");
    assert_eq!(honest.block_reach(), [2, 0, 0]);
    let short = ShortReach {
        inner: honest,
        reach: [1, 0, 0],
    };
    let source = PointSourceOp::new(points.clone());
    let plan = fragment_only(volume, block, Dtype::F64, &[&source, &short]).expect("a plan");

    // Both guards pass. That is the finding.
    plan.check().expect("the valid regions still tile");
    let work = [PhaseWork::Fragments(&source), PhaseWork::Fragments(&short)];
    check_phase_work(&plan, &work).expect("the fragment reach still agrees with the halo");

    let env = ArrayEnvironment::new(
        Voxels::zeros(Dtype::F64, volume).expect("a level"),
        plan.n_phases(),
        [4, 4, 4],
    )
    .expect("an environment");
    execute_phases(
        "short",
        &workflow(volume),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .expect("a run that no guard refuses");
    let output = env.output();
    let rendered: Vec<f64> = output
        .view::<f64>()
        .expect("an f64 level")
        .iter()
        .copied()
        .collect();

    // ...and the answer is wrong, by exactly the voxels of block 0 that the
    // kernel covers: 7 is the only one, and it is the one block 0 would have
    // needed block 2's fragment to know about.
    assert_eq!(
        rendered.iter().sum::<f64>(),
        correct.iter().sum::<f64>() - 1.0,
        "the short reach lost nothing, so this case does not exercise it"
    );
    let stride = volume[1] * volume[2];
    assert_eq!(correct[7 * stride + 4 * volume[2] + 4], 1.0);
    assert_eq!(rendered[7 * stride + 4 * volume[2] + 4], 0.0);
}

/// The declaration's cost, measured against the analytic answer rather than
/// against itself: a reach of `r` blocks fetches `neighbourhood_size` fragments
/// per block, and not one more.
#[test]
fn the_fetch_count_is_the_declared_neighbourhood_and_no_larger() {
    let volume = [40usize, 8, 8];
    let block = [8usize, 8, 8];
    let points = vec![Point::unit([20, 4, 4])];

    for (element, reach) in [
        (single_voxel(), [0usize, 0, 0]),
        (ball([8, 0, 0]), [1, 0, 0]),
        (ball([9, 0, 0]), [2, 0, 0]),
    ] {
        let grid = BlockGrid::new(volume, block).expect("a lattice");
        let op = VoxelizeOp::new("voxelize", STREAM, 0, element.clone(), &grid).expect("an op");
        assert_eq!(op.block_reach(), reach, "the derivation moved");

        let (_, log) = render_with_log(volume, block, &element, &points);
        let mut per_block: BTreeMap<[usize; 3], usize> = BTreeMap::new();
        for event in log.events() {
            if let Event::SidecarRead {
                stream,
                phase,
                index,
                ..
            } = event
            {
                if stream == STREAM && phase == 0 {
                    *per_block.entry(index).or_insert(0) += 1;
                }
            }
        }
        let measured: usize = per_block.values().sum();
        let counts = grid.blocks_per_axis();
        let analytic: usize = grid
            .cores()
            .iter()
            .map(|core| neighbourhood_size(core.index, reach, counts))
            .sum();
        assert_eq!(
            measured, analytic,
            "a reach of {reach:?} fetched {measured} fragment(s), wanted {analytic}"
        );
    }
}

/// The payload is the caller's, and its size is a stated fact rather than a
/// surprise: four `u64`s per point, so a block that found `n` of them writes
/// `32 n` bytes and a block that found none writes a fragment of zero length —
/// which is present, and therefore checkable.
#[test]
fn a_block_with_no_points_still_writes_a_fragment() {
    let volume = [24usize, 8, 8];
    let block = [8usize, 8, 8];
    let points = vec![Point::unit([1, 1, 1])];
    let (plan, voxelize) = plan_for(volume, block, &single_voxel());
    let source = PointSourceOp::new(points.clone());
    let env = ArrayEnvironment::new(
        Voxels::zeros(Dtype::F64, volume).expect("a level"),
        plan.n_phases(),
        [4, 4, 4],
    )
    .expect("an environment");
    execute_phases(
        "voxelize",
        &workflow(volume),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Fragments(&source),
            PhaseWork::Fragments(&voxelize),
        ],
    )
    .expect("a run");

    let keys = env.sidecar_keys(STREAM).expect("the keys");
    assert_eq!(keys.len(), 3, "every block of the lattice writes one");
    let sizes: Vec<usize> = keys
        .iter()
        .map(|key| {
            env.read_sidecar(&key.stream, key.phase, key.block)
                .expect("a read")
                .expect("the fragment is there")
                .len()
        })
        .collect();
    assert_eq!(sizes, vec![WORDS_PER_POINT * 8, 0, 0]);
}
