// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `ops::detect`: a mask of connected regions in, one
// point per region out, at the region's centroid.
//
// This is the op that joins the two halves of the crate — `crate::points` stores
// point sets and `ops::voxelize` renders them, and until this op neither had a
// producer that had ever looked at a volume. So the suite ends where the op does:
// the fragments a run leaves behind are fed to a `PointStore`, sealed, and
// queried, because a producer whose output cannot be read by the consumer it was
// written for has not been tested at all.
//
// The bar is `image_ops.rs`': byte-identical output under every decomposition,
// against a whole-volume reference computed by the same kernels called once. A
// global op needs one more, because it has a failure mode a local one does not —
// **it can be right on easy data and wrong on hard data at the same block size**.
// A mask whose regions all sit inside one block gives the right answer even if
// the merge does nothing.
//
// So the scene puts one region across a seam and one across four blocks, and
// that is then *asserted rather than assumed*: `the_merge_changes_the_answer_and_
// the_suite_says_by_how_much` pins the block-local answer's point count against
// the global one's, so if a change ever makes the two agree, this suite says it
// has stopped testing the merge instead of passing quietly.

use std::collections::{BTreeMap, BTreeSet};

use ndarray::Array3;

use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::PhaseWork;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::detect::{
    centroid_points, detect_phases, detect_regions, label_regions_into, moments_of_labels,
    owner_of, LabelRegionsOp, RegionMoments, RegionPointsOp,
};
use blockflow::ops::voxelize::{single_voxel, voxelize_into};
use blockflow::points::{decode_points, encode_points, Layout, Point, PointStore};
use blockflow::region::Region;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [24, 16, 12];
const MOMENTS: &str = "detect.moments";
const POINTS: &str = "detect.points";

/// The blocking the disagreement test cuts with, and the one the scene's two
/// seam-crossing regions are placed against: seams at x = 8 and 16, y = 8,
/// z = 6.
const CUT: [usize; 3] = [8, 8, 6];

// ------------------------------------------------------------- the scene --

/// A mask with nine regions in it, each there to catch something different.
///
/// 1. **A 2-cube wholly inside one block**, at 2..=3 on every axis. The easy
///    case, and the one that would still pass if the merge did nothing.
/// 2. **A bar across one seam**: x from 6 to 9 at y = 3, z = 1, cut by the seam
///    at x = 8. Four voxels, so its centroid is at x = 7.5 — *on the seam* —
///    which makes it the rounding case and the merge case at once.
/// 3. **A slab across two seams**: x and y both from 6 to 9 at z = 2, so it is
///    cut into four blocks by the seams at x = 8 and y = 8. This is the one a
///    block-local answer turns into four points.
/// 4. **Five single voxels at the volume's faces and corners**, including the
///    two opposite corners. Nothing here treats the volume boundary as special
///    and this is what says so.
/// 5. **A bar lying along a volume face**, at x = 23, y from 1 to 3, z = 5.
///
/// None of them touches another under face connectivity, and
/// `the_scene_holds_the_regions_it_claims` checks that by counting rather than
/// by inspection.
fn scene() -> Array3<bool> {
    let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);

    // 1: a 2 x 2 x 2 cube; mean 2.5 on every axis, so the centroid rounds up.
    for i in 2..=3 {
        for j in 2..=3 {
            for k in 2..=3 {
                mask[[i, j, k]] = true;
            }
        }
    }

    // 2: the bar across the seam at x = 8.
    for x in 6..=9 {
        mask[[x, 3, 1]] = true;
    }

    // 3: the slab across the seams at x = 8 and y = 8.
    for x in 6..=9 {
        for y in 6..=9 {
            mask[[x, y, 2]] = true;
        }
    }

    // 4: faces and corners.
    for at in corners() {
        mask[at] = true;
    }

    // 5: a bar on the far face of axis 0.
    for y in 1..=3 {
        mask[[23, y, 5]] = true;
    }

    mask
}

fn corners() -> Vec<[usize; 3]> {
    vec![[0, 0, 0], [23, 0, 0], [0, 15, 0], [0, 0, 11], [23, 15, 11]]
}

/// What the scene's regions must come out as, written from their definitions
/// rather than from a run of the code under test.
fn expected() -> Vec<Point> {
    let mut points = vec![
        // 1: eight voxels, mean 2.5 on each axis, half rounds up
        Point::weighted([3, 3, 3], 8.0),
        // 2: (6+7+8+9)/4 = 7.5, half rounds up
        Point::weighted([8, 3, 1], 4.0),
        // 3: 7.5 on both cut axes, sixteen voxels
        Point::weighted([8, 8, 2], 16.0),
        // 5: three voxels along y
        Point::weighted([23, 2, 5], 3.0),
    ];
    points.extend(corners().into_iter().map(Point::unit));
    canonical(points)
}

/// The canonical order of `crate::points`, which is what every comparison here
/// is made in.
fn canonical(mut points: Vec<Point>) -> Vec<Point> {
    points.sort_by(|left, right| {
        left.at
            .cmp(&right.at)
            .then_with(|| left.weight.to_bits().cmp(&right.weight.to_bits()))
    });
    points
}

// ------------------------------------------------------------ the oracle --

/// The whole-volume reference: the same kernels, called once, over everything.
///
/// Not a second implementation — that would make a disagreement a modelling
/// difference rather than a decomposition bug. `detect_regions` is the op's own
/// composition of its three kernels and is `pub` for exactly this.
fn reference(mask: &Array3<bool>) -> Vec<Point> {
    detect_regions(mask.view()).expect("the reference runs")
}

// --------------------------------------------------------------- the run --

fn plan_for(
    volume: [usize; 3],
    block: [usize; 3],
) -> (Decomposition, LabelRegionsOp, RegionPointsOp) {
    let grid = BlockGrid::new(volume, block).expect("a lattice");
    let label = LabelRegionsOp::new("label", MOMENTS, Lifecycle::DeleteOnExit);
    let points = RegionPointsOp::new(
        "points",
        MOMENTS,
        0,
        POINTS,
        // The points stream is the output of the run, so it outlives it.
        Lifecycle::Persistent,
        &grid,
    );
    let plan = detect_phases(grid, Dtype::Bool, &label, &points).expect("a plan");
    (plan, label, points)
}

/// Run the two phases and hand back the environment, which is where the answer
/// is: this op writes no level at all, so there is nothing to read out of
/// `env.output()` and everything to read out of the store.
fn run(mask: &Array3<bool>, block: [usize; 3]) -> (ArrayEnvironment, Decomposition) {
    let volume = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    let (plan, label, points) = plan_for(volume, block);
    let input: Voxels = mask.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, [4, 4, 4]).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::Bool);
    execute_phases(
        "detect",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&points)],
    )
    .expect("a run");
    (env, plan)
}

/// Every block's point blob, in block order.
fn blobs(env: &ArrayEnvironment, plan: &Decomposition) -> Vec<([usize; 3], Vec<u8>)> {
    every_block(&plan.phases[1].grid)
        .into_iter()
        .map(|index| {
            let bytes = env
                .read_sidecar(POINTS, 1, index)
                .expect("the store answers")
                .unwrap_or_else(|| panic!("block {index:?} wrote no point blob"));
            (index, bytes)
        })
        .collect()
}

/// Every point a run produced, in the canonical order.
fn points_of(env: &ArrayEnvironment, plan: &Decomposition) -> Vec<Point> {
    let mut found = Vec::new();
    for (index, bytes) in blobs(env, plan) {
        found.extend(
            decode_points(&bytes).unwrap_or_else(|error| panic!("block {index:?}: {error}")),
        );
    }
    canonical(found)
}

fn detected(mask: &Array3<bool>, block: [usize; 3]) -> Vec<Point> {
    let (env, plan) = run(mask, block);
    points_of(&env, &plan)
}

/// The decompositions the suite sweeps. One block — no seams at all — several
/// shapes, and three that leave a partial block on an axis.
fn blockings() -> Vec<[usize; 3]> {
    vec![
        VOLUME,
        [8, 16, 12],
        [8, 8, 12],
        CUT,
        [12, 8, 6],
        [7, 5, 5],
        [5, 16, 12],
    ]
}

fn every_block(grid: &BlockGrid) -> Vec<[usize; 3]> {
    grid.cores().into_iter().map(|core| core.index).collect()
}

// --------------------------------------------------------- the properties --

/// The scene must contain what it claims to. Asserted first, because every other
/// test in this file is worthless if it does not.
#[test]
fn the_scene_holds_the_regions_it_claims() {
    let found = reference(&scene());
    assert_eq!(
        found,
        expected(),
        "the whole-volume answer is not the one the scene was written for"
    );
    assert_eq!(found.len(), 9);
    // The weights are the voxel counts, and they add up to the mask.
    let voxels: f64 = found.iter().map(|point| point.weight).sum();
    assert_eq!(
        voxels,
        scene().iter().filter(|&&set| set).count() as f64,
        "the weights must account for every set voxel exactly once"
    );
}

/// The property this whole file exists for. Every decomposition, byte-identical
/// to the whole-volume reference.
///
/// Compared as **encoded bytes** rather than as a `Vec<Point>`, because that is
/// what a caller would write out and because `f64`'s own equality would call
/// `-0.0` and `0.0` the same weight.
#[test]
fn every_decomposition_reproduces_the_whole_volume_reference() {
    let mask = scene();
    let want = encode_points(&reference(&mask));
    for block in blockings() {
        assert_eq!(
            encode_points(&detected(&mask, block)),
            want,
            "decomposition {block:?} disagreed"
        );
    }
}

/// A block-local answer is **wrong** on this scene, and by exactly the two
/// regions that cross a seam. Without this, the sweep above could be passing
/// because the merge is a no-op.
///
/// The local answer is what each block would produce if it labelled only its own
/// core and took the centroids there — which is precisely what an op without the
/// merge would do, at any halo shorter than a region.
#[test]
fn the_merge_changes_the_answer_and_the_suite_says_by_how_much() {
    let mask = scene();
    let global = reference(&mask);
    let grid = BlockGrid::new(VOLUME, CUT).expect("a lattice");

    let mut local = Vec::new();
    for core in grid.cores() {
        let (low, extent) = (
            [core.core.start[0], core.core.start[1], core.core.start[2]],
            [core.core.shape[0], core.core.shape[1], core.core.shape[2]],
        );
        let cut = Array3::from_shape_fn((extent[0], extent[1], extent[2]), |(i, j, k)| {
            mask[[low[0] + i, low[1] + j, low[2] + k]]
        });
        let mut labels = Array3::<u32>::zeros(cut.raw_dim());
        let count = label_regions_into(cut.view(), labels.view_mut()).unwrap();
        // The same kernel, given the block's own corner, so the local points are
        // in volume coordinates and can be compared with the global ones.
        let moments = moments_of_labels(labels.view(), count, low).unwrap();
        local.extend(centroid_points(&moments));
    }
    let local = canonical(local);

    // The bar becomes two points and the slab becomes four, so thirteen where
    // there are nine. Pinned as numbers rather than as "more": if a change makes
    // the local and global answers agree, this scene has stopped testing the
    // merge and the counts say which way it moved.
    assert_eq!(global.len(), 9);
    assert_eq!(
        local.len(),
        13,
        "the block-local answer must split the bar in two and the slab in four"
    );
    let shared = local
        .iter()
        .filter(|point| global.iter().any(|other| other.at == point.at))
        .count();
    assert_eq!(
        shared, 7,
        "exactly the seven regions inside one block must be answered the same way by \
         both, or this count is measuring something other than the seams"
    );
    // And the two that differ are the ones named: the merged answers appear in
    // the global set and in no block's local one.
    for at in [[8usize, 3, 1], [8, 8, 2]] {
        assert!(global.iter().any(|point| point.at == at));
        assert!(
            !local.iter().any(|point| point.at == at),
            "{at:?} is reachable without the merge, so it proves nothing"
        );
    }
}

/// A region straddling two blocks, and then four, is **one** point at the
/// whole-volume centroid — asserted on its own, away from the scene, so the
/// failure is unambiguous.
#[test]
fn a_region_across_a_seam_is_one_point_at_the_whole_volume_centroid() {
    // Two blocks: a bar of four voxels centred on the seam at x = 8.
    let mut bar = Array3::from_elem((16, 8, 8), false);
    for x in 6..=9 {
        bar[[x, 4, 4]] = true;
    }
    let want = reference(&bar);
    assert_eq!(want, vec![Point::weighted([8, 4, 4], 4.0)]);
    for block in [[16usize, 8, 8], [8, 8, 8], [4, 8, 8], [3, 8, 8], [5, 5, 5]] {
        let found = detected(&bar, block);
        assert_eq!(found.len(), 1, "decomposition {block:?} split the bar");
        assert_eq!(found, want, "decomposition {block:?} disagreed");
    }

    // Four blocks: a square straddling the seams at x = 8 and y = 8.
    let mut square = Array3::from_elem((16, 16, 4), false);
    for x in 6..=9 {
        for y in 6..=9 {
            square[[x, y, 1]] = true;
        }
    }
    let want = reference(&square);
    assert_eq!(want, vec![Point::weighted([8, 8, 1], 16.0)]);
    for block in [[16usize, 16, 4], [8, 8, 4], [4, 4, 4], [5, 5, 2], [8, 8, 1]] {
        let found = detected(&square, block);
        assert_eq!(
            found.len(),
            1,
            "decomposition {block:?} split the square into {} point(s)",
            found.len()
        );
        assert_eq!(found, want, "decomposition {block:?} disagreed");
    }
}

/// Exactly one point per component, none lost and none duplicated — asserted
/// against the *blocks* rather than against the answer, which is where a
/// duplicate would come from: every block runs the same merge and sees every
/// component, and only the ownership rule stops all of them emitting it.
#[test]
fn exactly_one_block_emits_each_point_and_it_lands_in_that_block_s_core() {
    let mask = scene();
    for block in blockings() {
        let (env, plan) = run(&mask, block);
        let grid = &plan.phases[1].grid;
        let blobs = blobs(&env, &plan);
        assert_eq!(
            blobs.len(),
            grid.n_blocks(),
            "decomposition {block:?}: a block wrote no blob at all"
        );

        let mut total = 0usize;
        let mut seen: BTreeMap<[usize; 3], usize> = BTreeMap::new();
        for (index, bytes) in &blobs {
            for point in decode_points(bytes).expect("a point blob") {
                total += 1;
                *seen.entry(point.at).or_default() += 1;
                // The rule, checked where it is applied: a block emits only the
                // points whose centroid its own core holds. That is also
                // `ops::voxelize`'s precondition, so this output can feed it
                // directly.
                assert_eq!(
                    owner_of(grid, point.at),
                    *index,
                    "decomposition {block:?}: block {index:?} emitted a point at {:?}",
                    point.at
                );
            }
        }
        assert_eq!(
            total, 9,
            "decomposition {block:?} emitted {total} point(s) for nine regions"
        );
        assert!(
            seen.values().all(|&count| count == 1),
            "decomposition {block:?} emitted a coordinate twice: {seen:?}"
        );
    }
}

/// The other consumer, and the reason the ownership rule is the one it is:
/// `ops::voxelize` **refuses** a point that does not lie in the core of the block
/// whose fragment carries it, so a producer that got the rule wrong cannot feed
/// it at all.
///
/// Driven through `voxelize_into` rather than through a third phase, because that
/// free function is where the rule is enforced and a phase would only add a plan
/// between the assertion and the thing asserted. What comes out is the density
/// the weight choice was made for: each region's voxel count, deposited at its
/// centroid.
#[test]
fn the_points_a_run_writes_are_keyed_the_way_voxelize_requires() {
    let mask = scene();
    let (env, plan) = run(&mask, CUT);
    let grid = &plan.phases[1].grid;

    let fragments: Vec<([usize; 3], Vec<Point>)> = blobs(&env, &plan)
        .into_iter()
        .map(|(index, bytes)| (index, decode_points(&bytes).expect("a point blob")))
        .collect();

    let mut rendered = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    voxelize_into(
        &fragments,
        grid,
        &single_voxel(),
        &Region::whole(&VOLUME),
        rendered.view_mut(),
    )
    .expect("every point lies in the core of the block that wrote it");

    for point in expected() {
        assert_eq!(
            rendered[point.at], point.weight,
            "the point at {:?} did not land with its count",
            point.at
        );
    }
    assert_eq!(
        rendered.iter().sum::<f64>(),
        mask.iter().filter(|&&set| set).count() as f64,
        "the rendered density must weigh exactly what the mask holds"
    );
    assert_eq!(rendered.iter().filter(|&&value| value != 0.0).count(), 9);
}

/// The rounding is decomposition-independent, and the case that could not be is
/// a centroid landing exactly on a half **at a seam**.
///
/// The bar's centroid is 7.5 on axis 0, so the cut at eight puts the seam
/// through it: half the voxels are in each block and the answer has to come from
/// the totals rather than from either side. Every cut must round the same way,
/// and the way is up.
#[test]
fn a_centroid_on_a_half_rounds_the_same_way_under_every_cut() {
    let mut bar = Array3::from_elem((16, 16, 2), false);
    for x in 6..=9 {
        for y in 6..=9 {
            bar[[x, y, 0]] = true;
        }
    }
    // 7.5 on both axes; the alternative rounding would give [7, 7, 0], so the
    // assertion discriminates rather than merely being stable.
    let want = vec![Point::weighted([8, 8, 0], 16.0)];
    assert_eq!(reference(&bar), want);

    for block in [
        [16usize, 16, 2],
        // the seam exactly on the centroid, on one axis and then on both
        [8, 16, 2],
        [8, 8, 2],
        // seams either side of it
        [4, 4, 2],
        [3, 3, 1],
        [5, 5, 1],
        [7, 7, 2],
        [9, 9, 2],
    ] {
        assert_eq!(
            detected(&bar, block),
            want,
            "decomposition {block:?} rounded the other way"
        );
    }
}

/// The empty mask and the full one, through the blocked path: no points and one
/// point.
///
/// The empty one is also the coverage case — every block writes a zero-length
/// blob, which is present and therefore checkable, and a run that wrote nothing
/// at all would fail the every-block guard rather than pass silently.
#[test]
fn an_empty_mask_produces_no_points_and_a_full_one_produces_exactly_one() {
    let empty = Array3::from_elem((13, 9, 7), false);
    for block in [[13usize, 9, 7], [4, 4, 4], [5, 9, 7]] {
        let (env, plan) = run(&empty, block);
        assert!(points_of(&env, &plan).is_empty(), "{block:?}");
        // present, and empty
        for (index, bytes) in blobs(&env, &plan) {
            assert!(
                bytes.is_empty(),
                "block {index:?} wrote {} bytes",
                bytes.len()
            );
        }
    }

    let full = Array3::from_elem((13, 9, 7), true);
    let want = reference(&full);
    assert_eq!(want.len(), 1);
    assert_eq!(want[0].weight, (13 * 9 * 7) as f64);
    for block in [[13usize, 9, 7], [4, 4, 4], [5, 9, 7], [13, 3, 2]] {
        assert_eq!(detected(&full, block), want, "decomposition {block:?}");
    }
}

/// **The end the op exists for.** The fragments a run leaves are what
/// `PointStore::write` takes; sealing makes them queryable; and what comes back
/// is the point set the volume held.
#[test]
fn the_points_a_run_writes_go_into_a_store_and_come_back_by_region() {
    let mask = scene();
    let (env, plan) = run(&mask, CUT);

    let mut store = PointStore::new(VOLUME).expect("a store");
    for (index, bytes) in blobs(&env, &plan) {
        store
            .write(index, &bytes)
            .unwrap_or_else(|error| panic!("block {index:?}: {error}"));
    }
    assert_eq!(store.len(), 9);
    // A query before sealing is an error rather than a partial answer.
    assert!(store.query(&Region::whole(&VOLUME)).is_err());
    store.seal().expect("sealing");
    assert_eq!(store.layout(), Some(Layout::Flat));

    // The whole volume is the whole answer, in the canonical order.
    assert_eq!(store.query(&Region::whole(&VOLUME)).unwrap(), expected());

    // A region holds exactly the points inside it. The box below covers the
    // 2-cube's centroid at [3, 3, 3] and the bar's at [8, 3, 1] and nothing
    // else — worked out from the scene rather than from the store.
    let found = store
        .query(&Region::new(&[1, 1, 0], &[9, 6, 6]))
        .expect("a query");
    assert_eq!(
        found,
        vec![
            Point::weighted([3, 3, 3], 8.0),
            Point::weighted([8, 3, 1], 4.0),
        ]
    );

    // And a region with nothing in it answers nothing, which is a different
    // thing from an unsealed store's refusal above.
    assert!(store
        .query(&Region::new(&[12, 0, 0], &[4, 4, 4]))
        .unwrap()
        .is_empty());

    // The other index answers identically, which is `points.rs`' promise and is
    // worth exercising through a producer's output rather than only through
    // scattered test data.
    let mut gridded = PointStore::new(VOLUME).expect("a store");
    for (index, bytes) in blobs(&env, &plan) {
        gridded.write(index, &bytes).expect("a write");
    }
    gridded.seal_as(Layout::Gridded).expect("sealing");
    assert_eq!(
        gridded.query(&Region::whole(&VOLUME)).unwrap(),
        store.query(&Region::whole(&VOLUME)).unwrap()
    );
}

/// What this op costs in *pixels*, measured rather than described, and it is the
/// number that distinguishes it from the two ops it shares its program with.
///
/// `fill` and `regional` both end by rewriting a label level, so their phase 1
/// reads the whole volume once per block — a read amplification equal to the
/// block count. This op's phase 1 reads no pixels at all, because everything it
/// needs is in the fragments, so the whole run reads each voxel exactly once.
#[test]
fn the_labelling_is_halo_free_and_the_merge_reads_no_pixels_at_all() {
    let mask = scene();
    let (plan, _, _) = plan_for(VOLUME, CUT);
    for block in &plan.phases[0].blocks {
        assert_eq!(
            block.read.ranges(),
            block.core.ranges(),
            "the labelling must read exactly its core"
        );
    }
    // Phase 1's halo is the whole volume — that is the dependency edge, and
    // `fill`'s header is where it is argued for — but the read extent costs
    // nothing here because the op declares it reads no pixels.
    for block in &plan.phases[1].blocks {
        assert_eq!(block.read.voxels(), VOLUME.iter().product::<usize>());
        assert_eq!(block.valid.ranges(), block.core.ranges());
    }

    let (env, _) = run(&mask, CUT);
    let (reads, writes, read_voxels, write_voxels, ..) = env.counters().snapshot();
    assert_eq!(
        read_voxels,
        VOLUME.iter().product::<usize>() as u64,
        "the run read {read_voxels} voxel(s) for a volume of {}",
        VOLUME.iter().product::<usize>()
    );
    assert_eq!(
        write_voxels, 0,
        "neither phase writes a level, so nothing may be written"
    );
    assert!(reads > 0 && writes == 0);
}

/// Every block writes a moments fragment, the merge reads every one of them, and
/// what they carry is a summary rather than a volume.
#[test]
fn the_merge_reads_every_block_and_the_fragments_are_summaries() {
    let mask = scene();
    let (env, plan) = run(&mask, CUT);
    let grid = &plan.phases[0].grid;

    let mut stored = BTreeMap::new();
    for index in every_block(grid) {
        let bytes = env
            .read_sidecar(MOMENTS, 0, index)
            .expect("the store answers")
            .unwrap_or_else(|| panic!("block {index:?} wrote no moments fragment"));
        stored.insert(index, RegionMoments::decode(&bytes).expect("a fragment"));
    }
    assert_eq!(stored.len(), grid.n_blocks());

    // The accumulators are partial and they add up: every set voxel is counted
    // once, across the whole lattice.
    let counted: u64 = stored
        .values()
        .flat_map(|report| report.moments.iter().map(|part| part.count))
        .sum();
    assert_eq!(counted, mask.iter().filter(|&&set| set).count() as u64);

    // The fragment is six planes and a little, not a volume: far smaller than
    // the block it summarises, which is the whole point of the shape.
    let block_bytes = CUT[0] * CUT[1] * CUT[2] * 4;
    for (index, report) in &stored {
        assert!(
            report.encode().len() < block_bytes,
            "block {index:?}'s fragment is {} bytes against a {block_bytes}-byte block",
            report.encode().len()
        );
    }

    // Some block found more than one region, or the merge had nothing to join.
    assert!(
        stored.values().any(|report| report.labels > 1),
        "no block found two regions; the union-find is an identity on this scene"
    );
    // And the blocks the two seam-crossing regions were cut into really did each
    // see a piece: four blocks hold part of the slab at z = 2.
    let holders: BTreeSet<[usize; 3]> = stored
        .iter()
        .filter(|(_, report)| report.labels > 0)
        .map(|(index, _)| *index)
        .collect();
    assert!(
        holders.len() >= 4,
        "only {} block(s) hold anything",
        holders.len()
    );
}
