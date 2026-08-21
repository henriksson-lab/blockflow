// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The plan's own statement of what it holds.** `Decomposition::
// peak_image_bytes` is the image-lifetime walk that three consumer suites had
// each open-coded, promoted to the type that owns every one of its inputs.
//
// What this file has to establish, in order:
//
// 1. **It is the same walk**, so promoting it changes no recorded figure. The
//    reference is the copy those suites carry, transcribed here verbatim and
//    run against the method over a set of plans.
// 2. **Where it deliberately differs**, and by how much. The copies assume phase
//    `p` writes image `p + 1` and that phase `p` reads image `p`; a fragment op
//    may do neither, and the plan knows.
// 3. **It counts supplied inputs**, which the copies do not, because
//    `n_images()` excludes them by design and a `0..n_images()` loop therefore
//    walks past an array that is as resident as image 0.
// 4. **It is refused rather than guessed** for a phase nobody described.
// 5. **It is not a function of the block size**, which is the single most
//    important thing about it: it is a statement about arrays, and a run's
//    resident bytes are not. That is the property that makes it wrong as a bound
//    and right as a comparison between plans.

use blockflow::decomposition::{Decomposition, PhaseDecomposition, Visibility};
use blockflow::fragment::{
    fragment_only, BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput, PhaseWork,
};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::probes::IdentityOp;
use blockflow::sidecar::Lifecycle;
use blockflow::Result;
use blockflow::{Dtype, ImageId};

const VOLUME: [usize; 3] = [32, 16, 8];

/// The walk as three consumer residency suites each carry it.
///
/// Transcribed rather than referenced, and deliberately: the copies are what the
/// recorded figures were taken with, they live in a crate this one must not
/// name or depend on, and a test that reached across to them could not be run by
/// anyone who does not have that crate. So the reference travels here as code.
fn open_coded_reference(plan: &Decomposition) -> u64 {
    let bytes_of = |image: usize| -> u64 {
        let volume = plan.volume_at(image);
        volume.iter().product::<usize>() as u64 * plan.dtype_at(image).size_of() as u64
    };
    let mut live = vec![false; plan.n_images()];
    live[0] = true;
    let mut peak = bytes_of(0);
    for phase in 0..plan.n_phases() {
        if phase + 1 < plan.n_images() {
            live[phase + 1] = true;
        }
        let now = (0..plan.n_images())
            .filter(|image| live[*image])
            .map(bytes_of)
            .sum();
        if now > peak {
            peak = now;
        }
        for image in plan.images_dead_after(phase) {
            if plan.image_visibility(image) == Visibility::Internal {
                live[image] = false;
            }
        }
    }
    peak
}

/// An all-pixel plan of `n` phases at `edge`, each phase one identity slot.
fn pixel_plan(n: usize, edge: usize) -> (Chain, Decomposition) {
    let grid = BlockGrid::along(VOLUME, &[0, 1, 2], edge).expect("a grid");
    let chain = Chain::sequence(
        (0..n)
            .map(|_| Chain::op(IdentityOp::new("identity", [0, 0, 0])))
            .collect(),
    );
    let phases = (0..n)
        .map(|slot| {
            PhaseDecomposition::derive(
                vec![slot],
                vec!["identity".to_string()],
                [0usize, 0, 0],
                [0usize, 0, 0],
                grid.clone(),
            )
        })
        .collect();
    (
        chain,
        Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases,
            chain_reach: [0, 0, 0],
        },
    )
}

/// A `fragments -> fragments` op: reads no pixel, writes no image.
struct Merge;

impl FragmentOp for Merge {
    fn name(&self) -> &'static str {
        "merge"
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            "merged",
            Lifecycle::DeleteOnExit,
            Coverage::Sparse,
        )]
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Ok(BlockOutput::nothing())
    }
}

/// Promoting the walk moved nothing: for every plan the copies can express, the
/// method is the copies.
#[test]
fn the_method_is_the_walk_the_consumer_suites_open_coded() {
    for phases in 1..=6 {
        for edge in [4usize, 8, 32] {
            let (_, plan) = pixel_plan(phases, edge);
            assert_eq!(
                plan.peak_image_bytes(&[]).expect("an all-pixel plan"),
                open_coded_reference(&plan),
                "{phases} phase(s) at edge {edge}"
            );
        }
    }
}

/// **Three images at a time, however long the chain**, and which three is the
/// whole content of the walk: the run's input, which is `Published` and is never
/// discarded, plus the two an identity step is between. A six-phase chain names
/// seven images and holds three.
///
/// Without this the equivalence above would be satisfied by two functions that
/// are wrong in the same way — summing everything the plan names would pass it.
#[test]
fn a_chain_of_identities_holds_three_images_however_long_it_gets() {
    let voxels: u64 = VOLUME.iter().product::<usize>() as u64;
    let one = voxels * Dtype::F64.size_of() as u64;
    for phases in 2..=6 {
        let (_, plan) = pixel_plan(phases, 8);
        assert_eq!(
            plan.peak_image_bytes(&[]).expect("an all-pixel plan"),
            3 * one,
            "a chain of {phases} identities should hold the published input and the two images \
             one step is between"
        );
        assert_eq!(
            plan.n_images(),
            phases + 1,
            "and the plan really does name more images than that"
        );
        assert_eq!(
            plan.image_visibility(0),
            Visibility::Published,
            "the third is image 0, and it is held because nothing may discard it"
        );
    }
}

/// **The deliberate difference.** A `fragments -> fragments` phase writes no
/// image, and the open-coded copies charge for one anyway because they assume
/// phase `p` fills image `p + 1`. Measured, so the size of the correction is on
/// the record rather than the direction alone.
#[test]
fn a_phase_that_writes_no_image_is_not_charged_for_one() {
    let merge = Merge;
    let plan = fragment_only(VOLUME, [8, 8, 8], Dtype::F64, &[&merge])
        .expect("a fragments-only decomposition");
    let told = plan
        .peak_image_bytes(&[PhaseWork::Fragments(&merge)])
        .expect("a described plan");
    let assumed = open_coded_reference(&plan);
    let one = VOLUME.iter().product::<usize>() as u64 * Dtype::F64.size_of() as u64;
    assert_eq!(
        told, one,
        "a fragments-only run holds the array it was cut from and nothing else"
    );
    assert_eq!(
        assumed,
        2 * one,
        "and the open-coded walk charges for an image the run never writes"
    );
}

/// **Supplied inputs are resident and the copies walk past them.** `n_images()`
/// counts what the plan fills in, and a supplied array is addressed in a
/// disjoint high range — so a `0..n_images()` loop can never see one.
#[test]
fn a_supplied_input_is_counted_and_the_open_coded_walk_cannot_see_it() {
    let grid = BlockGrid::along(VOLUME, &[0, 1, 2], 8).expect("a grid");
    let supplied = ImageId::supplied(0);
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![PhaseDecomposition::derive(
            vec![0],
            vec!["identity".to_string()],
            [0usize, 0, 0],
            [0usize, 0, 0],
            grid,
        )
        .with_source_images([supplied.index()])
        .with_supplied_dtypes([(supplied.index(), Dtype::F64)])],
        chain_reach: [0, 0, 0],
    };
    let one = VOLUME.iter().product::<usize>() as u64 * Dtype::F64.size_of() as u64;
    assert_eq!(
        plan.peak_image_bytes(&[]).expect("an all-pixel plan"),
        3 * one,
        "image 0, the supplied array and the output are all held at once"
    );
    assert_eq!(
        open_coded_reference(&plan),
        2 * one,
        "and the open-coded walk misses the supplied array entirely"
    );
}

/// A phase that owns no chain slot and is not described is refused, not assumed
/// to write. The same rule and the same message `predicted_cost` uses, because
/// it is the same question.
#[test]
fn an_undescribed_slotless_phase_is_refused_rather_than_assumed_to_write() {
    let merge = Merge;
    let plan = fragment_only(VOLUME, [8, 8, 8], Dtype::F64, &[&merge])
        .expect("a fragments-only decomposition");
    let message = plan
        .peak_image_bytes(&[])
        .expect_err("a slotless phase with no work entry must not be guessed at")
        .to_string();
    assert!(
        message.contains("owns no chain slot"),
        "the refusal should say what is missing: {message}"
    );
}

/// **The property that makes this a comparison and not a bound.** The figure is
/// a statement about arrays, so it does not move with the block size — while a
/// run's resident bytes very much do, in both directions and by amounts nothing
/// in a `Decomposition` can see: at one block a fan-in's per-branch buffers
/// shrink with a narrowed reach and no image entry moves, and at a fine grid
/// `ImageStore::pending` allocates lazily and frees after the last reader, so
/// the moment this walk finds is a moment the run does not have.
///
/// Asserted rather than written in a comment, because a future version that
/// started varying with the edge would be claiming to be something it is not.
#[test]
fn the_figure_does_not_move_with_the_block_size_and_that_is_the_point() {
    let edges = [2usize, 4, 8, 16, 32];
    let figures: Vec<u64> = edges
        .iter()
        .map(|&edge| {
            let (_, plan) = pixel_plan(4, edge);
            plan.peak_image_bytes(&[]).expect("an all-pixel plan")
        })
        .collect();
    assert!(
        figures.windows(2).all(|pair| pair[0] == pair[1]),
        "the image figure moved with the block edge: {edges:?} gave {figures:?}. It is a \
         statement about whole arrays; if it now depends on the grid it is answering a \
         different question and its callers are reading it wrong"
    );
    // And the grids really are different, so the invariance is a property of the
    // figure rather than of a fixture in which nothing varied.
    let blocks: Vec<usize> = edges
        .iter()
        .map(|&edge| pixel_plan(4, edge).1.phases[0].grid.n_blocks())
        .collect();
    assert!(
        blocks.windows(2).any(|pair| pair[0] != pair[1]),
        "every edge produced the same grid: {blocks:?}"
    );
}
