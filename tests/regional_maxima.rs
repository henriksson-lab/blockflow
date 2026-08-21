// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `ops::regional`: the connected plateaus of a
// greyscale volume from which one cannot ascend.
//
// This is the second genuinely **global** op in the crate, and it is global for
// a sharper reason than `fill` is. A plateau is a connected set of equal values,
// so it can run the length of the volume; whether it is a maximum depends on
// every voxel touching it, and the voxel that settles the question may be at the
// far end of a ridge that crosses every block on an axis. No halo of any size
// answers that.
//
// The bar is the one `image_ops.rs` sets — byte-identical output under every
// decomposition, measured against a whole-volume reference computed by the same
// kernels called once — plus the extra bar a global op needs, because it has a
// failure mode a local one does not: **it can be right on easy data and wrong on
// hard data at the same block size.** A volume whose plateaus all sit inside one
// block gives the right answer even if the merge does nothing at all.
//
// So the scene is built to make the merge load-bearing and that is then
// *asserted rather than assumed*: a flat ridge crossing two seams with the only
// higher voxel touching its far end, and an assertion that the block-local
// answers disagree with the global one over exactly that ridge. If they ever
// stop disagreeing, this suite has stopped testing the merge and says so instead
// of passing quietly.

use std::collections::BTreeMap;

use ndarray::Array3;

use blockflow::assemble::ImageId;
use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::{neighbourhood_size, PhaseWork};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::regional::{
    ascending_neighbours, label_plateaux_into, maxima_from_labels_into, regional_maxima,
    regional_phases, LabelPlateauxOp, PlateauFaces, RegionalMaximaOp,
};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [24, 16, 12];
const STREAM: &str = "regional.faces";

/// The blocking the disagreement test cuts with, and the one the spikes below
/// are placed against.
const CUT: [usize; 3] = [8, 8, 6];

// ------------------------------------------------------------- the scene --

/// A volume with four things in it, each there to catch something different,
/// and the third is the one that matters.
///
/// 1. **A flat surround** at 1.0. One enormous plateau spanning every block,
///    with higher things standing on it, so it is not a maximum anywhere — and
///    a block that decided on its own would agree only because of item 2.
/// 2. **One spike per block**, at 9.0. Single-voxel maxima, and they are placed
///    one to a block of `CUT` on purpose: without them the surround inside a
///    block with nothing higher in it would be a *local* maximum, and the
///    disagreement count in `the_merge_changes_the_answer_and_the_suite_says_by_
///    how_much` would be a property of the surround rather than of the ridge.
/// 3. **A flat ridge at 5.0 crossing two block seams**, with a single 7.0 voxel
///    touching only its far end — and touching it **across the second seam**.
///    The whole ridge is one plateau and **none** of it is a maximum, and
///    reaching that answer needs both halves of the merge and neither would do
///    on its own: the seam at x = 16 is where a *comparison* discovers that
///    something higher stands over the ridge, and the seam at x = 8 is where the
///    *union* carries that discovery back to the piece of ridge two blocks away
///    that can see neither the 7.0 nor the block that met it. Cut into three,
///    every piece that does not hold the 7.0 sees nothing higher and would call
///    itself a maximum. This is the case where the local and global answers
///    differ, and it is the reason this suite can claim the merge does anything.
/// 4. **A 3x3x3 plateau at 8.0**, itself cut by a seam, with nothing higher
///    anywhere near it. The whole of it is a maximum, and it is here so that the
///    suite distinguishes "the merge joins plateaus" from "the merge clears
///    them": a change that flagged every seam-crossing plateau as non-maximal
///    would pass every other assertion in this file.
fn scene() -> Array3<f64> {
    let mut values = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 1.0);

    // 2: one spike in each of the 3 x 2 x 2 blocks of `CUT`.
    for &x in &[4usize, 12, 20] {
        for &y in &[4usize, 12] {
            for &z in &[3usize, 9] {
                values[[x, y, z]] = 9.0;
            }
        }
    }

    // 3: the ridge, x from 0 to 15 over blocks of 8, so it fills the first two
    // blocks exactly; the voxel that settles it is the first voxel of the third,
    // which puts it on the far side of a seam from every voxel of the ridge.
    for x in 0..=15 {
        values[[x, 1, 1]] = 5.0;
    }
    values[[16, 1, 1]] = 7.0;

    // 4: a plateau that *is* a maximum, cut by the seam at x = 16.
    for x in 15..=17 {
        for y in 0..=2 {
            for z in 8..=10 {
                values[[x, y, z]] = 8.0;
            }
        }
    }
    values
}

/// Where the ridge lives, as a predicate, so the assertions below name it once.
fn on_the_ridge(x: usize, y: usize, z: usize) -> bool {
    y == 1 && z == 1 && x <= 15
}

// ------------------------------------------------------------- the oracle --

/// The whole-volume reference: the same kernels, called once, over everything.
///
/// Not a second implementation — that would make a disagreement a modelling
/// difference rather than a decomposition bug. `regional_maxima` is the op's own
/// composition of its three kernels and is `pub` for exactly this.
fn reference(values: &Array3<f64>) -> Array3<bool> {
    regional_maxima(values.view()).expect("the reference runs")
}

// --------------------------------------------------------------- the run --

fn plan_for(
    volume: [usize; 3],
    block: [usize; 3],
) -> (Decomposition, LabelPlateauxOp, RegionalMaximaOp) {
    let grid = BlockGrid::new(volume, block).expect("a lattice");
    let label = LabelPlateauxOp::new("label", STREAM, Lifecycle::DeleteOnExit);
    let maxima = RegionalMaximaOp::new("maxima", STREAM, 0, Dtype::Bool, &grid);
    let plan = regional_phases(grid, Dtype::F64, &label, &maxima).expect("a plan");
    (plan, label, maxima)
}

fn run(values: &Array3<f64>, block: [usize; 3]) -> Array3<bool> {
    run_keeping(values, block, &[]).0
}

/// The same, pinning some internal images so a test can look at them.
///
/// Without this the label image is freed the moment the phase that reads it
/// finishes — which is the point of `Visibility::Internal`, and which makes
/// `keep_images` the way a *debugging* test says it wants the intermediate.
fn run_keeping(
    values: &Array3<f64>,
    block: [usize; 3],
    keep: &[usize],
) -> (Array3<bool>, ArrayEnvironment, Decomposition) {
    let volume = [values.shape()[0], values.shape()[1], values.shape()[2]];
    let (plan, label, maxima) = plan_for(volume, block);
    let input: Voxels = values.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, [4, 4, 4]).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::F64);
    let hints = Hints {
        keep_images: keep.iter().copied().map(ImageId::from).collect(),
        ..Hints::default()
    };
    execute_phases(
        "regional",
        &workflow,
        &plan,
        &hints,
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&maxima)],
    )
    .expect("a run");
    let out = env.output().view::<bool>().unwrap().to_owned();
    (out, env, plan)
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

// --------------------------------------------------------- the properties --

/// The scene must contain what it claims to. Asserted first, because every other
/// test in this file is worthless if it does not.
#[test]
fn the_scene_has_a_spike_a_maximal_plateau_and_a_ridge_that_is_not_one() {
    let values = scene();
    let found = reference(&values);

    // 2: every spike is a single-voxel maximum
    for &x in &[4usize, 12, 20] {
        for &y in &[4usize, 12] {
            for &z in &[3usize, 9] {
                assert!(found[[x, y, z]], "the spike at {x},{y},{z}");
            }
        }
    }

    // 3: no part of the ridge is a maximum, however far from the 7.0
    for x in 0..=15 {
        assert!(
            !found[[x, 1, 1]],
            "the ridge is one plateau with something higher at its end, so no voxel \
             of it is a maximum — this one is at x = {x}"
        );
    }
    assert!(
        found[[16, 1, 1]],
        "and the voxel that settles it is a maximum"
    );

    // 4: the whole cut plateau is a maximum
    for x in 15..=17 {
        for y in 0..=2 {
            for z in 8..=10 {
                assert!(found[[x, y, z]], "the flat plateau at {x},{y},{z}");
            }
        }
    }

    // 1: the surround is not
    assert!(!found[[0, 0, 0]]);
    assert!(!found[[10, 10, 5]]);

    // and exactly the things named above are maxima: 12 spikes, one 7.0, and a
    // 3-cube. Counted rather than sampled, so a change that set some extra voxel
    // somewhere else in the volume fails here rather than passing unnoticed.
    let set = found.iter().filter(|&&flag| flag).count();
    assert_eq!(set, 12 + 1 + 27);
}

/// The property this whole file exists for. Every decomposition, byte-identical
/// to the whole-volume reference.
#[test]
fn every_decomposition_reproduces_the_whole_volume_reference() {
    let values = scene();
    let want = reference(&values);
    for block in blockings() {
        assert_eq!(
            run(&values, block),
            want,
            "decomposition {block:?} disagreed"
        );
    }
}

/// A block-local answer is **wrong** on this scene, and by exactly the ridge.
/// Without this, the sweep above could be passing because the merge is a no-op.
///
/// The local answer is what each block would produce if it labelled only its own
/// core and looked for higher neighbours only inside it — which is precisely
/// what a `BlockOp` at any halo shorter than the ridge would do.
#[test]
fn the_merge_changes_the_answer_and_the_suite_says_by_how_much() {
    let values = scene();
    let global = reference(&values);
    let grid = BlockGrid::new(VOLUME, CUT).expect("a lattice");

    let mut disagreements = 0usize;
    let mut disagreed_off_the_ridge = 0usize;
    for index in every_block(&grid) {
        let (low, extent) = core_of(&grid, index);
        let cut = Array3::from_shape_fn((extent[0], extent[1], extent[2]), |(i, j, k)| {
            values[[low[0] + i, low[1] + j, low[2] + k]]
        });

        let mut labels = Array3::<u32>::zeros(cut.raw_dim());
        let (count, _) = label_plateaux_into(cut.view(), labels.view_mut()).unwrap();
        let ascends = ascending_neighbours(cut.view(), labels.view(), count).unwrap();
        let mut local = Array3::from_elem(cut.raw_dim(), false);
        maxima_from_labels_into(labels.view(), &ascends, local.view_mut()).unwrap();

        for i in 0..extent[0] {
            for j in 0..extent[1] {
                for k in 0..extent[2] {
                    let (x, y, z) = (low[0] + i, low[1] + j, low[2] + k);
                    if local[[i, j, k]] != global[[x, y, z]] {
                        disagreements += 1;
                        if !on_the_ridge(x, y, z) {
                            disagreed_off_the_ridge += 1;
                        }
                    }
                }
            }
        }
    }

    // Exactly the ridge, which is x from 0 to 7 and x from 8 to 15: sixteen
    // voxels, none of them in the block holding the 7.0. Every one of them is a
    // maximum locally and is not one globally, because the plateau they belong
    // to reaches past a seam to a voxel that stands over it. Asserted as a
    // number rather than as "more than none": if a change makes the local and
    // global answers agree, this scene has stopped testing the merge, and the
    // count says which way it moved.
    assert_eq!(
        disagreements, 16,
        "the block-local answer must differ from the global one over exactly the ridge, \
         no block of which can see the higher voxel"
    );
    assert_eq!(
        disagreed_off_the_ridge, 0,
        "the surround and the two plateaus that *are* maxima must be answered the same \
         way by both, or this count is measuring something other than the seam"
    );
}

/// The seam case stated on its own, and the one the union-find exists for.
///
/// A plateau extended by an equal-valued voxel **inside** a block and the same
/// plateau extended by an equal-valued voxel **across a seam** are the same
/// shape, so they must get the same answer. A merge that unioned nothing would
/// answer the second one differently, and would do so in the direction of
/// reporting a maximum that is not one.
#[test]
fn a_tie_across_a_seam_is_the_same_plateau_as_a_tie_inside_a_block() {
    let block = [8usize, 4, 4];
    // A line of four equal voxels with a higher one just past its right end.
    // `start` is where the line begins; at 2 it is wholly inside the first
    // block, at 6 it straddles the seam at 8.
    let build = |start: usize| {
        let mut values = Array3::from_elem((16, 4, 4), 0.0);
        for x in start..start + 4 {
            values[[x, 1, 1]] = 5.0;
        }
        values[[start + 4, 1, 1]] = 7.0;
        values
    };

    let inside = build(2);
    let across = build(6);
    assert_eq!(run(&inside, block), reference(&inside));
    assert_eq!(run(&across, block), reference(&across));

    for offset in 0..4 {
        assert!(
            !run(&inside, block)[[2 + offset, 1, 1]],
            "the intra-block tie: the whole line is one plateau under a higher voxel"
        );
        assert!(
            !run(&across, block)[[6 + offset, 1, 1]],
            "the tie across the seam must be answered the same way, and at offset \
             {offset} the block-local answer would say maximum"
        );
    }
    assert!(
        run(&across, block)[[10, 1, 1]],
        "the higher voxel is the maximum"
    );

    // and the shapes really do differ in where the seam falls, or this test is
    // comparing two identical situations
    assert!(2 + 3 < block[0] && 6 + 3 >= block[0]);
}

/// The degenerate case: nothing is higher than anything, so the whole volume is
/// one plateau and the whole volume is a regional maximum. This is what an
/// off-by-one in the adjacency test — reading past an edge, or treating a block
/// seam as an edge — gets wrong.
#[test]
fn a_constant_volume_is_one_regional_maximum_covering_everything() {
    let flat = Array3::from_elem((13, 9, 7), 4.0);
    for block in [[13usize, 9, 7], [4, 4, 4], [5, 9, 7], [13, 3, 2]] {
        let found = run(&flat, block);
        assert!(
            found.iter().all(|&set| set),
            "decomposition {block:?} left a voxel of a constant volume out of the one \
             regional maximum"
        );
    }
}

/// And the one voxel that changes it. A constant volume with a single higher
/// corner has **one** maximum, and every other voxel of it belongs to a plateau
/// that spans the entire lattice — so this fails at every block but the corner's
/// unless the merge closes the plateau across every seam of the volume.
#[test]
fn one_higher_corner_disqualifies_the_whole_rest_of_the_volume() {
    let mut values = Array3::from_elem((13, 9, 7), 4.0);
    values[[0, 0, 0]] = 5.0;
    let want = reference(&values);
    assert_eq!(want.iter().filter(|&&flag| flag).count(), 1);
    assert!(want[[0, 0, 0]]);

    for block in [[13usize, 9, 7], [4, 4, 4], [5, 9, 7], [13, 3, 2]] {
        assert_eq!(
            run(&values, block),
            want,
            "decomposition {block:?} disagreed"
        );
    }
}

/// A monotone ramp has no maximum except its top slice, at every decomposition.
/// A block-local answer would report the top slice of *every* block.
#[test]
fn a_monotone_ramp_has_no_maximum_except_its_top() {
    let values = Array3::from_shape_fn((12, 5, 3), |(i, _, _)| i as f64);
    let want = reference(&values);
    for i in 0..12 {
        assert_eq!(want[[i, 0, 0]], i == 11);
    }
    for block in [[12usize, 5, 3], [4, 5, 3], [5, 2, 2], [3, 5, 3]] {
        assert_eq!(
            run(&values, block),
            want,
            "decomposition {block:?} disagreed"
        );
    }
}

/// NaN through the blocked path, on the rule `ops::regional` states: a NaN voxel
/// is in no plateau and therefore in no maximum, and it disqualifies nothing,
/// because it is neither equal to nor greater than anything.
///
/// The NaN is put **on a seam** so that the merge has to skip it rather than
/// join across it: the two 3.0 runs either side of it are two plateaus, not one.
#[test]
fn a_nan_on_a_seam_joins_nothing_and_disqualifies_nothing() {
    let mut values = Array3::from_elem((16, 4, 4), 0.0);
    for x in 4..8 {
        values[[x, 1, 1]] = 3.0;
    }
    values[[8, 1, 1]] = f64::NAN;
    for x in 9..13 {
        values[[x, 1, 1]] = 3.0;
    }

    let want = reference(&values);
    for x in 4..8 {
        assert!(want[[x, 1, 1]], "the run left of the NaN is a maximum");
    }
    for x in 9..13 {
        assert!(want[[x, 1, 1]], "and the run right of it is a separate one");
    }
    assert!(!want[[8, 1, 1]], "the NaN voxel itself is in no maximum");

    for block in [[16usize, 4, 4], [8, 4, 4], [4, 4, 4], [5, 4, 4]] {
        assert_eq!(
            run(&values, block),
            want,
            "decomposition {block:?} disagreed"
        );
    }
}

/// What a global reduction costs in *pixels*, measured rather than described.
///
/// Phase 0 is halo-free: a block-local labelling reads exactly its own core.
/// Phase 1 is not, and cannot be — its whole-lattice fragment reach is also the
/// dependency edge that makes it wait for phase 0's blocks, so its halo covers
/// the volume and every block of it reads the entire label image.
///
/// The number is asserted because it is the argument for a barrier. If a future
/// change lets a phase state "after all of phase 0" without a halo, this test
/// fails and the reason it fails is the improvement.
#[test]
fn the_labelling_is_halo_free_and_the_merge_reads_the_whole_image() {
    let (plan, _, _) = plan_for(VOLUME, CUT);
    let volume_voxels = VOLUME[0] * VOLUME[1] * VOLUME[2];

    for block in &plan.phases[0].blocks {
        assert_eq!(
            block.read.ranges(),
            block.core.ranges(),
            "the labelling must read exactly its core"
        );
        assert_eq!(block.valid.ranges(), block.core.ranges());
    }

    let blocks = plan.phases[1].blocks.len();
    let mut read = 0usize;
    for block in &plan.phases[1].blocks {
        assert_eq!(
            block.read.voxels(),
            volume_voxels,
            "the merge reads the whole image, which is what its halo says"
        );
        assert_eq!(block.valid.ranges(), block.core.ranges());
        read += block.read.voxels();
    }
    assert_eq!(
        read,
        blocks * volume_voxels,
        "the read amplification of the merge is the block count: {blocks}x"
    );
}

/// Every block writes a faces fragment, the merge reads every one of them, and
/// what they carry is what the merge needs.
#[test]
fn the_merge_reads_every_block_and_the_fragments_carry_the_plateau_values() {
    let values = scene();
    let (_, env, plan) = run_keeping(&values, CUT, &[]);
    let counts = plan.phases[0].grid.blocks_per_axis();

    let mut stored = BTreeMap::new();
    for index in every_block(&plan.phases[0].grid) {
        let bytes = env
            .read_sidecar(STREAM, 0, index)
            .expect("the store answers")
            .unwrap_or_else(|| panic!("block {index:?} wrote no faces fragment"));
        stored.insert(
            index,
            PlateauFaces::decode(&bytes).expect("a faces fragment"),
        );
    }
    assert_eq!(stored.len(), counts[0] * counts[1] * counts[2]);

    // what a whole-lattice reach asks each block for
    let analytic = neighbourhood_size([0, 0, 0], counts, counts);
    assert_eq!(
        analytic,
        counts[0] * counts[1] * counts[2],
        "a whole-lattice reach must ask for the whole lattice"
    );

    // every block found several plateaus, which is what makes the union-find do
    // work rather than be an identity
    assert!(
        stored.values().all(|faces| faces.labels > 1),
        "a block that found one plateau gave the merge nothing to join"
    );
    // and the values travel, or a seam could not be resolved at all
    assert!(
        stored.values().any(|faces| faces.values.contains(&9.0)),
        "no fragment carries the value of the spike in its block"
    );
    // the fragment is six planes and a little, not a volume: it must be far
    // smaller than the block it summarises, which is the whole point of the shape
    let block_bytes = CUT[0] * CUT[1] * CUT[2] * 4;
    for (index, faces) in &stored {
        assert!(
            faces.encode().len() < block_bytes,
            "block {index:?}'s fragment is {} bytes against a {block_bytes}-byte block",
            faces.encode().len()
        );
    }
}

/// The label image is decomposition-*dependent* while the output is not. Stated
/// in `ops/fill.rs` for the op this one is modelled on, and asserted here too,
/// because it looks like the defect the crate exists to prevent and a reader
/// deserves the evidence that it is not.
#[test]
fn the_intermediate_labels_differ_between_decompositions_and_the_output_does_not() {
    let values = scene();
    let (coarse_out, coarse_env, _) = run_keeping(&values, VOLUME, &[1]);
    let (fine_out, fine_env, _) = run_keeping(&values, CUT, &[1]);
    assert_eq!(coarse_out, fine_out);

    let coarse_labels = coarse_env.image(1).view::<u32>().unwrap().to_owned();
    let fine_labels = fine_env.image(1).view::<u32>().unwrap().to_owned();
    assert_ne!(
        coarse_labels, fine_labels,
        "if the two label volumes agree, this test is asserting nothing"
    );
    // and the same decomposition twice gives the same labels
    let (_, again, _) = run_keeping(&values, CUT, &[1]);
    assert_eq!(
        fine_labels,
        again.image(1).view::<u32>().unwrap().to_owned()
    );
}

/// A run that carries the answer as `f64` under the module's set/clear
/// convention gives the same answer as one that carries it as `bool`.
#[test]
fn a_mask_carried_as_f64_gives_the_same_answer_as_one_carried_as_bool() {
    let values = scene();
    let narrow = run(&values, CUT);

    let grid = BlockGrid::new(VOLUME, CUT).expect("a lattice");
    let label = LabelPlateauxOp::new("label", STREAM, Lifecycle::DeleteOnExit);
    let maxima = RegionalMaximaOp::new("maxima", STREAM, 0, Dtype::F64, &grid);
    let plan = regional_phases(grid, Dtype::F64, &label, &maxima).expect("a plan");
    let env = ArrayEnvironment::for_decomposition(values.clone().into(), &plan, [4, 4, 4])
        .expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::F64);
    execute_phases(
        "regional",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&maxima)],
    )
    .expect("a run");

    let out = env.output().view::<f64>().unwrap().to_owned();
    for (flag, value) in narrow.iter().zip(out.iter()) {
        assert_eq!(if *flag { 1.0 } else { 0.0 }, *value);
    }
}

/// A plan over an element type the fragment cannot carry exactly is refused
/// before anything is scheduled, rather than at the first block or, worse, not
/// at all.
#[test]
fn a_plan_over_an_element_type_the_fragment_cannot_carry_is_refused() {
    let grid = BlockGrid::new(VOLUME, CUT).expect("a lattice");
    let label = LabelPlateauxOp::new("label", STREAM, Lifecycle::DeleteOnExit);
    let maxima = RegionalMaximaOp::new("maxima", STREAM, 0, Dtype::Bool, &grid);
    let message = regional_phases(grid, Dtype::U64, &label, &maxima)
        .unwrap_err()
        .to_string();
    assert!(message.contains("where the seam fell"), "{message}");

    // and the widths that do fit are planned without complaint
    for dtype in [Dtype::U8, Dtype::U16, Dtype::U32, Dtype::I32, Dtype::F32] {
        let grid = BlockGrid::new(VOLUME, CUT).expect("a lattice");
        let maxima = RegionalMaximaOp::new("maxima", STREAM, 0, Dtype::Bool, &grid);
        regional_phases(grid, dtype, &label, &maxima)
            .unwrap_or_else(|error| panic!("{dtype:?} was refused: {error}"));
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
