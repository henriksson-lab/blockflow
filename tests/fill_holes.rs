// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `ops::fill`, which is the first genuinely **global**
// operation in the crate: whether a background voxel is inside a hole depends on
// whether a path exists from it to the outside of the volume, and no halo of any
// size can answer that.
//
// The bar is the same one `image_ops.rs` sets — byte-identical output under
// every decomposition, measured against a whole-volume reference computed by the
// same kernels called once. What is different here is that passing it is not
// nearly enough, because a global op has a failure mode a local one does not:
// **it can be right on easy data and wrong on hard data at the same block size.**
// A mask whose holes all sit inside one block gives the right answer even if the
// merge does nothing at all.
//
// So the scene is built to make the merge load-bearing and that is then
// *asserted rather than assumed*:
//
// * a cavity that spans several blocks, so closing it requires unioning labels
//   across two seams;
// * a cavity that drains through a **winding channel** which leaves and re-enters
//   blocks, so a block-local answer would fill it and the global answer must not;
// * an assertion that the local answers genuinely disagree with the global one —
//   if they ever stop disagreeing, this suite has stopped testing the merge and
//   says so instead of passing quietly.

use std::collections::BTreeMap;

use ndarray::Array3;

use blockflow::assemble::ImageId;
use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::{neighbourhood_size, FragmentOp, PhaseWork};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::components::Merge;
use blockflow::ops::fill::{
    fill_from_labels_into, fill_phases, label_background_into, outside_flags, BlockFaces,
    FillHolesOp, LabelBackgroundOp,
};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [24, 16, 12];
const STREAM: &str = "fill.faces";

// ------------------------------------------------------------- the scene --

/// A mask with three things in it, each there to catch something different, and
/// the second and third are the ones that matter.
///
/// 1. **A box with a cavity wholly inside one block.** The easy case, and the
///    one that would pass with no merge at all.
/// 2. **A long box whose cavity spans the lattice**, x from 2 to 10 over blocks
///    of 8, so the cavity is two or three block-local components that only the
///    merge joins. It is **closed**, so it must fill — and a block deciding on
///    its own would drain it, because the cavity touches that block's own faces.
///    This is the case where local and global answers *differ*, and it is the
///    reason the suite can claim the merge does anything.
/// 3. **A second long box with a winding channel out to the volume's face.** The
///    channel leaves along x, turns along y, and reaches the outside — so this
///    cavity must **not** fill, however far from the drain a voxel is. A
///    block-local implementation reaches the same answer here for the wrong
///    reason, which is why case 2 exists as well.
fn scene() -> Array3<bool> {
    let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);

    // 1: a small closed box, wholly inside the first block on x.
    fill_box(&mut mask, [1, 1, 1], [5, 5, 5], true);
    fill_box(&mut mask, [2, 2, 2], [4, 4, 4], false);

    // 2: a long *closed* box, spanning the x seam at 8.
    fill_box(&mut mask, [1, 1, 7], [11, 5, 10], true);
    fill_box(&mut mask, [2, 2, 8], [10, 4, 9], false);

    // 3: a long box whose cavity drains, spanning several seams.
    fill_box(&mut mask, [1, 8, 6], [22, 14, 10], true);
    fill_box(&mut mask, [2, 9, 7], [21, 13, 9], false);
    // out through the wall at x=22, then along y to the volume's own face.
    mask[[22, 10, 8]] = false;
    for y in 10..=14 {
        mask[[22, y, 8]] = false;
    }
    mask
}

fn fill_box(mask: &mut Array3<bool>, low: [usize; 3], high: [usize; 3], value: bool) {
    for i in low[0]..=high[0] {
        for j in low[1]..=high[1] {
            for k in low[2]..=high[2] {
                mask[[i, j, k]] = value;
            }
        }
    }
}

// ---------------------------------------------------------- the oracle --

/// The whole-volume reference: the same kernels, called once, over everything.
///
/// Not a second implementation — that would make a disagreement a modelling
/// difference rather than a decomposition bug.
fn reference(mask: &Array3<bool>) -> Array3<bool> {
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    let count = label_background_into(mask.view(), labels.view_mut()).unwrap();
    let flags = outside_flags(labels.view(), count, [0, 0, 0], VOLUME, VOLUME);
    let mut out = Array3::from_elem(mask.raw_dim(), false);
    fill_from_labels_into(labels.view(), &flags, out.view_mut()).unwrap();
    out
}

// ------------------------------------------------------------- the run --

fn plan_for(block: [usize; 3]) -> (Decomposition, LabelBackgroundOp, FillHolesOp) {
    plan_for_merging(block, Merge::default())
}

/// The same plan with the merge placed by hand. See [`Merge`]: the default is
/// what ships and the other two are here so that what it buys can be measured
/// and so that the barrier has a control.
fn plan_for_merging(
    block: [usize; 3],
    merge: Merge,
) -> (Decomposition, LabelBackgroundOp, FillHolesOp) {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let label = LabelBackgroundOp::new("label", STREAM, Lifecycle::DeleteOnExit);
    let fill = FillHolesOp::new("fill", STREAM, 0, Dtype::Bool, &grid).merging(merge);
    let plan = fill_phases(grid, Dtype::Bool, &label, &fill).expect("a plan");
    (plan, label, fill)
}

fn run(mask: &Array3<bool>, block: [usize; 3]) -> (Array3<bool>, ArrayEnvironment, Decomposition) {
    run_keeping(mask, block, &[])
}

/// The same, pinning some internal images so a test can look at them.
///
/// Without this the label image is freed the moment the phase that reads it
/// finishes — which is the point of `Visibility::Internal`, and which makes
/// `keep_images` the way a *debugging* test says it wants the intermediate.
fn run_keeping(
    mask: &Array3<bool>,
    block: [usize; 3],
    keep: &[usize],
) -> (Array3<bool>, ArrayEnvironment, Decomposition) {
    let (plan, label, fill) = plan_for(block);
    let input: Voxels = mask.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, [4, 4, 4]).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::Bool);
    let hints = Hints {
        keep_images: keep.iter().copied().map(ImageId::from).collect(),
        ..Hints::default()
    };
    execute_phases(
        "fill",
        &workflow,
        &plan,
        &hints,
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&fill)],
    )
    .expect("a run");
    let out = env.output().view::<bool>().unwrap().to_owned();
    (out, env, plan)
}

/// The decompositions the suite sweeps. One block, several shapes, and two that
/// leave a partial block on an axis.
fn blockings() -> Vec<[usize; 3]> {
    vec![
        VOLUME,
        [8, 16, 12],
        [8, 8, 12],
        [8, 8, 6],
        [12, 8, 6],
        [7, 5, 5],
        [5, 16, 12],
    ]
}

// --------------------------------------------------------- the properties --

/// The scene must make the merge matter. Asserted first, because every other
/// test in this file is worthless if it does not.
#[test]
fn the_scene_has_a_hole_and_a_drain_and_the_drain_is_not_a_hole() {
    let mask = scene();
    let filled = reference(&mask);

    // 1: the small closed cavity fills
    assert!(!mask[[3, 3, 3]]);
    assert!(filled[[3, 3, 3]], "an enclosed cavity must fill");

    // 2: the closed cavity that spans the x seam fills, on both sides of it
    assert!(!mask[[3, 3, 8]] && !mask[[10, 3, 8]]);
    assert!(filled[[3, 3, 8]], "left of the seam");
    assert!(filled[[10, 3, 8]], "right of the seam");

    // 3: the draining cavity does *not* fill, however far from the drain
    assert!(!mask[[11, 11, 8]]);
    assert!(
        !filled[[11, 11, 8]],
        "a cavity that drains to the volume face is not a hole, however long the drain"
    );
    assert!(!filled[[3, 11, 8]], "and not at the far end of it either");

    // and the walls are untouched
    assert!(filled[[1, 8, 6]]);
    assert!(!filled[[0, 0, 0]], "the outside must not fill");
}

/// The property this whole file exists for. Every decomposition, byte-identical
/// to the whole-volume reference.
#[test]
fn every_decomposition_reproduces_the_whole_volume_reference() {
    let mask = scene();
    let want = reference(&mask);
    for block in blockings() {
        let (got, _, _) = run(&mask, block);
        assert_eq!(got, want, "decomposition {block:?} disagreed");
    }
}

/// A block-local answer is **wrong** on this scene, and by a lot. Without this,
/// the sweep above could be passing because the merge is a no-op.
///
/// The local answer is what each block would produce if it treated its own faces
/// as the outside world: label its core, drain whatever touches any of its own
/// six faces, fill the rest.
#[test]
fn the_merge_changes_the_answer_and_the_suite_says_by_how_much() {
    let mask = scene();
    let global = reference(&mask);
    let block = [8usize, 8, 6];
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");

    let mut disagreements = 0usize;
    for index in every_block(&grid) {
        let (low, extent) = core_of(&grid, index);
        let mut cut = Array3::from_elem((extent[0], extent[1], extent[2]), false);
        for i in 0..extent[0] {
            for j in 0..extent[1] {
                for k in 0..extent[2] {
                    cut[[i, j, k]] = mask[[low[0] + i, low[1] + j, low[2] + k]];
                }
            }
        }
        // the block deciding on its own, with its own faces as the outside
        let mut labels = Array3::<u32>::zeros(cut.raw_dim());
        let count = label_background_into(cut.view(), labels.view_mut()).unwrap();
        let flags = outside_flags(labels.view(), count, [0, 0, 0], extent, extent);
        let mut local = Array3::from_elem(cut.raw_dim(), false);
        fill_from_labels_into(labels.view(), &flags, local.view_mut()).unwrap();

        for i in 0..extent[0] {
            for j in 0..extent[1] {
                for k in 0..extent[2] {
                    if local[[i, j, k]] != global[[low[0] + i, low[1] + j, low[2] + k]] {
                        disagreements += 1;
                    }
                }
            }
        }
    }
    // Exactly the interior of cavity 2 — 9 x 3 x 2 voxels, from [2,2,8] to
    // [10,4,9]. Every one of them fills globally and drains locally, because
    // the cavity touches the faces of every block it is cut into. Asserted as a
    // number rather than as "more than none": if a change makes the local and
    // global answers agree, this scene has stopped testing the merge, and the
    // count says which way it moved.
    assert_eq!(
        disagreements,
        9 * 3 * 2,
        "the block-local answer must differ from the global one over exactly the closed \
         cavity that spans the seam"
    );
}

/// **The inverted assertion.** This test used to require that every block of the
/// merge read the entire label image, and said in as many words that if a future
/// change let a phase state "after all of phase 0" without a halo, it should
/// fail and the reason it failed would be the improvement. That change landed —
/// `docs/design/barriers.md` — so the requirement is turned round rather than
/// deleted, and the shape it replaced is kept beside it as the control.
///
/// Phase 0 is halo-free: a block-local labelling reads exactly its own core.
/// Phase 1 now is too, because the dependency it needs — *after all of phase 0*
/// — is stated as a barrier instead of bought with a halo.
#[test]
fn the_merge_declares_a_barrier_instead_of_reading_the_whole_image() {
    let volume_voxels = VOLUME[0] * VOLUME[1] * VOLUME[2];
    let (plan, _, _) = plan_for([8, 8, 6]);

    for block in &plan.phases[0].blocks {
        assert_eq!(
            block.read.ranges(),
            block.core.ranges(),
            "the labelling must read exactly its core"
        );
        assert_eq!(block.valid.ranges(), block.core.ranges());
    }

    assert!(
        !plan.phases[0].barrier,
        "the labelling depends on nothing global and must not serialise the run"
    );
    assert!(
        plan.phases[1].barrier,
        "the merge's answer depends on every block, which is what a barrier says"
    );

    let blocks = plan.phases[1].blocks.len();
    let mut read = 0usize;
    for block in &plan.phases[1].blocks {
        assert_eq!(
            block.read.ranges(),
            block.core.ranges(),
            "with the dependency stated, the merge fetches its own core"
        );
        assert_eq!(block.valid.ranges(), block.core.ranges());
        read += block.read.voxels();
    }
    assert_eq!(
        read, volume_voxels,
        "the whole phase reads the label image once, not {blocks} times"
    );

    // **The control**, and it is the assertion this test used to make. The same
    // op with the merge left in every block is the shape the framework used to
    // admit, and it still costs exactly what it cost — so the relief above is
    // attributable to the declaration and not to some other change.
    let (in_plan, _, _) = plan_for_merging([8, 8, 6], Merge::PerBlock);
    assert!(!in_plan.phases[1].barrier);
    let mut amplified = 0usize;
    for block in &in_plan.phases[1].blocks {
        assert_eq!(
            block.read.voxels(),
            volume_voxels,
            "without the barrier the halo is the fragment reach and covers the volume"
        );
        amplified += block.read.voxels();
    }
    assert_eq!(
        amplified,
        blocks * volume_voxels,
        "the read amplification a barrier removes is the block count: {blocks}x"
    );
}

/// The label image is decomposition-*dependent* while the output is not. Stated
/// in `ops/fill.rs` and asserted here, because it looks like the defect the
/// crate exists to prevent and a reader deserves the evidence that it is not.
#[test]
fn the_intermediate_labels_differ_between_decompositions_and_the_output_does_not() {
    let mask = scene();
    let (coarse_out, coarse_env, _) = run_keeping(&mask, VOLUME, &[1]);
    let (fine_out, fine_env, _) = run_keeping(&mask, [8, 8, 6], &[1]);
    assert_eq!(coarse_out, fine_out);

    let coarse_labels = coarse_env.image(1).view::<u32>().unwrap().to_owned();
    let fine_labels = fine_env.image(1).view::<u32>().unwrap().to_owned();
    assert_ne!(
        coarse_labels, fine_labels,
        "if the two label volumes agree, this test is asserting nothing"
    );
    // and the same decomposition twice gives the same labels
    let (_, again, _) = run_keeping(&mask, [8, 8, 6], &[1]);
    assert_eq!(
        fine_labels,
        again.image(1).view::<u32>().unwrap().to_owned()
    );
}

/// Every block writes a faces fragment, and the **phase** reads every one of
/// them — once, rather than once per block.
///
/// The fetch count is asserted against the analytic figure the declared reach
/// produces rather than against itself: a reach is a declaration, and this is
/// what it costs. The whole-lattice figure is kept as the control, because it is
/// what the same op still asks for with the merge left in every block.
#[test]
fn the_merge_reads_every_block_of_the_lattice_once_for_the_phase() {
    let mask = scene();
    let block = [8usize, 8, 6];
    let (_, env, plan) = run(&mask, block);
    let counts = plan.phases[0].grid.blocks_per_axis();
    let n_blocks = counts[0] * counts[1] * counts[2];

    let mut stored = BTreeMap::new();
    for index in every_block(&plan.phases[0].grid) {
        let bytes = env
            .read_sidecar(STREAM, 0, index)
            .expect("the store answers")
            .unwrap_or_else(|| panic!("block {index:?} wrote no faces fragment"));
        stored.insert(index, BlockFaces::decode(&bytes).expect("a faces fragment"));
    }
    assert_eq!(stored.len(), n_blocks);

    // what the hoisted shape asks each *block* for, from the op's own
    // declaration rather than from a number written here
    let (_, _, hoisted) = plan_for_merging(block, Merge::OnceForThePhase);
    let reach = hoisted.inputs()[0].reach;
    assert_eq!(
        reach,
        [0, 0, 0],
        "a hoisted merge needs no fragment per block"
    );
    assert_eq!(
        neighbourhood_size([0, 0, 0], reach, counts),
        1,
        "a reach of nothing is still the block itself, and `gathers() == false` \
         is what keeps even that off the wire"
    );
    assert!(!hoisted.gathers());

    // **The control.** With the merge in every block the reach is the lattice
    // and the fetch is the whole set, per block — which is the `(1 + blocks) x F`
    // multiplier the hoisting removes.
    let (_, _, in_plan) = plan_for_merging(block, Merge::PerBlock);
    let in_plan_reach = in_plan.inputs()[0].reach;
    assert_eq!(in_plan_reach, counts);
    assert_eq!(
        neighbourhood_size([0, 0, 0], in_plan_reach, counts),
        n_blocks,
        "a whole-lattice reach must ask for the whole lattice"
    );
    assert!(in_plan.gathers());

    // and at least one block found more than one background component, which is
    // what makes the union-find do work rather than be an identity
    assert!(
        stored.values().any(|faces| faces.labels > 1),
        "no block found two components, so the merge had nothing to join"
    );
}

/// A run that carries the mask as `f64` under the module's set/clear convention
/// gives the same answer as one that carries it as `bool`.
#[test]
fn a_mask_carried_as_f64_gives_the_same_answer_as_one_carried_as_bool() {
    let mask = scene();
    let (narrow, _, _) = run(&mask, [8, 8, 6]);

    let grid = BlockGrid::new(VOLUME, [8, 8, 6]).expect("a lattice");
    let label = LabelBackgroundOp::new("label", STREAM, Lifecycle::DeleteOnExit);
    let fill = FillHolesOp::new("fill", STREAM, 0, Dtype::F64, &grid);
    let plan = fill_phases(grid, Dtype::F64, &label, &fill).expect("a plan");
    let wide: Voxels = mask.mapv(|flag| if flag { 1.0 } else { 0.0 }).into();
    let env = ArrayEnvironment::for_decomposition(wide, &plan, [4, 4, 4]).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::F64);
    execute_phases(
        "fill",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&fill)],
    )
    .expect("a run");

    let out = env.output().view::<f64>().unwrap().to_owned();
    for (flag, value) in narrow.iter().zip(out.iter()) {
        assert_eq!(if *flag { 1.0 } else { 0.0 }, *value);
    }
}

// ------------------------------------------------------------- helpers --

fn every_block(grid: &BlockGrid) -> Vec<[usize; 3]> {
    let counts = grid.blocks_per_axis();
    let mut out = Vec::new();
    for i in 0..counts[0] {
        for j in 0..counts[1] {
            for k in 0..counts[2] {
                out.push([i, j, k]);
            }
        }
    }
    out
}

fn core_of(grid: &BlockGrid, index: [usize; 3]) -> ([usize; 3], [usize; 3]) {
    let volume = grid.volume();
    let edge = grid.block();
    let mut low = [0usize; 3];
    let mut extent = [0usize; 3];
    for axis in 0..3 {
        low[axis] = index[axis] * edge[axis];
        extent[axis] = edge[axis].min(volume[axis] - low[axis]);
    }
    (low, extent)
}
