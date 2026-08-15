// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A rank filter whose window has a population.**
//
// A percentile over a neighbourhood with background voxels removed — not
// replaced, *removed*, so they do not count toward the rank. The statistic then
// describes the structure present in the window rather than how much of the
// window it fills, which is the reason a caller wants it at all.
//
// What this file asserts, in the order the claims depend on each other:
//
// 1. **The sentinel workaround really does not work.** Excluding voxels by
//    setting them to ±inf and running the ordinary filter gives a different
//    answer, because the rank is resolved against the count of survivors and a
//    sentinel stays in the count. This is claim one because if it were false the
//    whole feature would be unnecessary.
// 2. **An all-true mask is the unmasked filter**, bit for bit — so the masked
//    kernel is the same filter and not a second one that agrees approximately.
// 3. **The mask changes the answer**, so nothing below is vacuous.
// 4. **An empty population writes the centre**, the one answer that keeps every
//    written value a value that was read.
// 5. **Decomposition invariance**: the same volume from every block size,
//    against the whole-volume reference. The acceptance bar for the whole crate,
//    and a second input read over a window is the obvious way to break it.
// 6. **The declaration is checked**: a mask level that does not hold `Bool` is
//    refused by name, and the op refuses to run without its operand at all.
//
// No assertion here is on wall-clock time.

use ndarray::{Array3, ArrayView3};

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::error::{Error, Result};
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain, SourceInputs};
use blockflow::ops::{
    masked_rank_filter_into, rank_filter_into, ElementShape, MaskedRankFilterOp, Rank,
    StructuringElement, Total,
};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [16, 12, 10];
/// Written by phase 0, read by phase 1 as the window's population.
const MASK: usize = 1;
/// The threshold that makes the mask, chosen so it keeps roughly half the
/// volume — a mask that is almost all true or almost all false would make the
/// masked and unmasked filters agree for an uninteresting reason.
const CUT: f64 = 8.0;

// ------------------------------------------------------------- fixtures --

/// A value per voxel with no plateaus: every window holds distinct values, so
/// selecting a different rank selects a different number and a wrong rank
/// cannot hide behind a tie.
fn image() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        ((i * 7 + j * 3 + k * 11) % 17) as f64
    })
}

fn mask_of(image: &Array3<f64>) -> Array3<bool> {
    image.mapv(|value| value > CUT)
}

fn element() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, [1, 1, 1])
}

/// The percentile the reference's arm takes, through the convention that states
/// the rank against the surviving population — which is the convention a masked
/// window makes visible in the first place.
fn rank() -> Rank {
    Rank::ceiling_percentile(0.25).unwrap()
}

/// The whole-volume answer, computed in one call by the same kernel the blocked
/// run uses — so a disagreement is a decomposition bug rather than two
/// implementations drifting.
fn reference() -> Array3<f64> {
    let image = image();
    let mask = mask_of(&image);
    let ordered = image.mapv(Total);
    let mut out = Array3::from_elem(ordered.raw_dim(), Total(0.0));
    masked_rank_filter_into(
        ordered.view(),
        mask.view(),
        &element(),
        rank(),
        out.view_mut(),
    )
    .unwrap();
    out.mapv(|value| value.0)
}

// ------------------------------------------------------------- the op --

/// `image > CUT`, into a `Bool` level. The mask producer, kept here rather than
/// taken from `src/ops` because what is under test is the *consumer*.
struct Binarize;

impl BlockOp for Binarize {
    fn name(&self) -> &'static str {
        "binarize"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let source = input.view::<f64>()?;
        let mut out = out.view_mut::<bool>()?;
        ndarray::Zip::from(&mut out)
            .and(&source)
            .for_each(|slot, &value| *slot = value > CUT);
        Ok(())
    }
}

// ------------------------------------- 1. the sentinel workaround fails --

/// Replacing excluded voxels with a sentinel and running the ordinary filter is
/// the first thing anyone tries. It is a different filter.
///
/// The reason is arithmetic rather than incidental: `Rank::resolve` places the
/// rank within the values the window actually read, and a sentinel is a value it
/// read. Removing a voxel shrinks the population; replacing it does not.
#[test]
fn a_sentinel_does_not_reproduce_a_removed_voxel() {
    let image = image();
    let mask = mask_of(&image);

    // Excluded voxels pushed above every real value, which is the friendliest
    // version of the workaround for a low percentile: they sort to the top and
    // "should" be out of the way.
    let sentinelled = Array3::from_shape_fn(image.raw_dim(), |index| {
        if mask[index] {
            Total(image[index])
        } else {
            Total(f64::INFINITY)
        }
    });
    let mut workaround = Array3::from_elem(sentinelled.raw_dim(), Total(0.0));
    rank_filter_into(
        sentinelled.view(),
        &element(),
        rank(),
        workaround.view_mut(),
    )
    .unwrap();

    let honest = reference();
    let workaround = workaround.mapv(|value| value.0);
    assert_ne!(
        honest, workaround,
        "if a sentinel reproduced a removed voxel, this op would not need to exist"
    );
}

// ------------------------------------------- 2. an all-true mask is plain --

#[test]
fn a_mask_that_keeps_everything_is_the_unmasked_filter() {
    let image = image();
    let ordered = image.mapv(Total);
    let all = Array3::from_elem(image.raw_dim(), true);

    let mut masked = Array3::from_elem(ordered.raw_dim(), Total(0.0));
    masked_rank_filter_into(
        ordered.view(),
        all.view(),
        &element(),
        rank(),
        masked.view_mut(),
    )
    .unwrap();

    let mut plain = Array3::from_elem(ordered.raw_dim(), Total(0.0));
    rank_filter_into(ordered.view(), &element(), rank(), plain.view_mut()).unwrap();

    assert_eq!(masked, plain, "one kernel, and masking is one more skip");
}

// ------------------------------------------------------ 3. non-vacuity --

#[test]
fn the_mask_changes_the_answer() {
    let image = image();
    let ordered = image.mapv(Total);
    let mut plain = Array3::from_elem(ordered.raw_dim(), Total(0.0));
    rank_filter_into(ordered.view(), &element(), rank(), plain.view_mut()).unwrap();
    assert_ne!(
        reference(),
        plain.mapv(|value| value.0),
        "the mask excluded nothing that mattered, so every test here is vacuous"
    );
}

// ------------------------------------------------ 4. an empty population --

#[test]
fn a_window_with_no_population_writes_the_centre() {
    let image = image();
    let ordered = image.mapv(Total);
    let none = Array3::from_elem(image.raw_dim(), false);
    let mut out = Array3::from_elem(ordered.raw_dim(), Total(0.0));
    masked_rank_filter_into(
        ordered.view(),
        none.view(),
        &element(),
        rank(),
        out.view_mut(),
    )
    .unwrap();
    assert_eq!(
        out.mapv(|value| value.0),
        image,
        "with nothing to select from, the value written is the value at the centre"
    );
}

// ---------------------------------------- 5. decomposition invariance --

/// Phase 0 writes the mask; phase 1 reads the image back through a source leaf
/// and filters it against that mask.
///
/// **Two different mechanisms in one phase, deliberately.** The leaf is the
/// reach-zero operand this crate already had, and the mask is the windowed
/// operand it did not — so this plan exercises both paths through
/// `SourceInputs` at once, which is where they could disagree.
fn chain() -> Chain {
    Chain::sequence(vec![
        Chain::op(Binarize),
        Chain::source(0usize, Dtype::F64),
        Chain::op(MaskedRankFilterOp::new(
            "masked-percentile",
            element(),
            rank(),
            MASK,
        )),
    ])
}

fn plan(chain: &Chain, grid: &BlockGrid) -> Decomposition {
    let slots = chain.slots();
    let phases = vec![
        PhaseDecomposition::derive(
            vec![0],
            vec![slots[0].display_name()],
            [0usize, 0, 0],
            [0usize, 0, 0],
            grid.clone(),
        ),
        PhaseDecomposition::derive(
            vec![1, 2],
            vec![slots[1].display_name(), slots[2].display_name()],
            [1usize, 1, 1],
            [1usize, 1, 1],
            grid.clone(),
        ),
    ];
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [1, 1, 1],
    };
    plan.declare_dtypes(chain).unwrap();
    plan.declare_source_levels(chain).unwrap();
    plan
}

fn grids() -> Vec<BlockGrid> {
    vec![
        BlockGrid::new(VOLUME, VOLUME).unwrap(),
        BlockGrid::along(VOLUME, &[0], 4).unwrap(),
        BlockGrid::along(VOLUME, &[0], 8).unwrap(),
        BlockGrid::along(VOLUME, &[1], 4).unwrap(),
        BlockGrid::along(VOLUME, &[2], 5).unwrap(),
        BlockGrid::along(VOLUME, &[0, 1], 4).unwrap(),
        BlockGrid::along(VOLUME, &[0, 1, 2], 4).unwrap(),
    ]
}

fn run(grid: &BlockGrid) -> Array3<f64> {
    let chain = chain();
    let decomposition = plan(&chain, grid);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env =
        ArrayEnvironment::for_decomposition(image().into(), &decomposition, [4, 4, 4]).unwrap();
    execute("masked", &workflow, &decomposition, &Hints::default(), &env).expect("a run");
    env.output().view::<f64>().unwrap().to_owned()
}

#[test]
fn every_decomposition_gives_the_whole_volume_answer() {
    let expected = reference();
    for grid in grids() {
        assert_eq!(run(&grid), expected, "block {:?}", grid.block());
    }
}

// --------------------------------------------- 6. the declaration holds --

#[test]
fn a_mask_level_that_is_not_bool_is_refused_by_name() {
    let op = MaskedRankFilterOp::new("masked", element(), rank(), MASK);
    let input: Voxels = image().into();
    let wrong: Voxels = image().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    let entries = [(MASK, &wrong)];
    let failed = op
        .apply_with(
            &input,
            SourceInputs::new(&entries),
            &mut out,
            &Anchor::whole(VOLUME),
        )
        .unwrap_err();
    let message = failed.to_string();
    assert!(
        message.contains("float64") && message.contains(&MASK.to_string()),
        "the refusal must name the level and what it holds: {message}"
    );
}

#[test]
fn the_op_refuses_to_run_without_its_population() {
    let op = MaskedRankFilterOp::new("masked", element(), rank(), MASK);
    let input: Voxels = image().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    let failed = op
        .apply(&input, &mut out, &Anchor::whole(VOLUME))
        .unwrap_err();
    assert!(
        matches!(failed, Error::InvalidArgument(ref message) if message.contains("apply_with")),
        "an op with an operand must not have an answer without one: {failed}"
    );
}

#[test]
fn the_op_declares_the_mask_at_the_elements_reach() {
    let op = MaskedRankFilterOp::new("masked", element(), rank(), MASK);
    let declared = op.source_inputs(VOLUME);
    assert_eq!(declared.len(), 1);
    assert_eq!(declared[0].level, MASK);
    assert_eq!(
        declared[0].reach,
        element().reach_spec(),
        "the population is consulted over the element, so the element states both"
    );
}

/// The masked filter still short circuits a constant block, and the declaration
/// is honest for the empty-population case too — which matters because the
/// short circuit has not read the mask and cannot know it.
#[test]
fn a_constant_block_maps_to_the_constant() {
    let op = MaskedRankFilterOp::new("masked", element(), rank(), MASK);
    assert_eq!(op.constant_maps_to(3.5), Some(3.5));

    let constant = Array3::from_elem((6, 6, 6), Total(3.5));
    let none = Array3::from_elem((6, 6, 6), false);
    let mut out = Array3::from_elem(constant.raw_dim(), Total(0.0));
    masked_rank_filter_into(
        constant.view(),
        none.view(),
        &element(),
        rank(),
        out.view_mut(),
    )
    .unwrap();
    assert!(
        out.iter().all(|value| value.0 == 3.5),
        "an empty population on a constant block is still the constant"
    );
}

/// A mask view of the wrong shape is refused rather than read out of bounds.
#[test]
fn a_mask_of_the_wrong_shape_is_refused() {
    let image = image();
    let ordered = image.mapv(Total);
    let small = Array3::from_elem((4, 4, 4), true);
    let mut out = Array3::from_elem(ordered.raw_dim(), Total(0.0));
    let failed = masked_rank_filter_into(
        ordered.view(),
        ArrayView3::from(small.view()),
        &element(),
        rank(),
        out.view_mut(),
    )
    .unwrap_err();
    assert!(
        failed.to_string().contains("mask"),
        "the refusal must say which array disagreed: {failed}"
    );
}
