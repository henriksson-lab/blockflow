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
// 7. **The policy at an excluded centre**, which is a parameter rather than a
//    rule: filter there anyway, or write a stated value. Asserted with masks
//    that are *not* all-true — a mask that keeps everything makes the parameter
//    invisible, which is exactly why the question went unasked for so long — and
//    with an element that does **not** contain its own centre, where "the centre
//    is excluded" and "the window came out empty" stop implying one another.
// 8. **Decomposition invariance for that policy too**: whether a centre is
//    excluded is a fact about the volume's population, and a run that decided it
//    from the block's own view would be wrong in a way no single block size
//    reveals.
//
// No assertion here is on wall-clock time.

use ndarray::{Array3, ArrayView3};

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::error::{Error, Result};
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain, SourceInputs};
use blockflow::ops::{
    masked_rank_filter_into, masked_rank_filter_into_with, rank_filter_into, ElementShape,
    ExcludedCentre, MaskedRankFilterOp, Rank, StructuringElement, Total,
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

// --------------------------------------- 7. the policy at the centre --

/// The element the two conditions come apart on: **it does not contain its own
/// centre.** Six faces and nothing in the middle.
///
/// Every test above uses a box that holds its centre, where "the centre is out
/// of the population" and "the window came out empty" imply one another in the
/// direction that matters. With this element a centre can be *in* the population
/// and still have an empty window, and a centre can be *out* of it while the
/// window is full. That is why it is here.
fn hollow() -> StructuringElement {
    StructuringElement::from_offsets([
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
    ])
    .unwrap()
}

fn filtered(
    image: &Array3<f64>,
    mask: &Array3<bool>,
    element: &StructuringElement,
    centre: ExcludedCentre<f64>,
) -> Array3<f64> {
    let ordered = image.mapv(Total);
    let mut out = Array3::from_elem(ordered.raw_dim(), Total(0.0));
    masked_rank_filter_into_with(
        ordered.view(),
        mask.view(),
        element,
        rank(),
        centre.map(Total),
        out.view_mut(),
    )
    .unwrap();
    out.mapv(|value| value.0)
}

/// **A centre the population excludes, with neighbours it does not.**
///
/// The two policies must give different answers here, and each must give the
/// answer it names: `Select` filters from the surviving neighbours, `Fill`
/// writes the stated value without reading them. A single small volume with one
/// hole punched in the mask is enough to separate them, and the hole is placed
/// in the interior so nothing about the array's edge is involved.
#[test]
fn an_excluded_centre_is_filtered_anyway_or_filled_as_the_caller_says() {
    let image = image();
    let mut mask = Array3::from_elem(image.raw_dim(), true);
    let hole = [5usize, 5, 5];
    mask[hole] = false;

    let selected = filtered(&image, &mask, &element(), ExcludedCentre::Select);
    let filled = filtered(&image, &mask, &element(), ExcludedCentre::Fill(-1.0));

    // the excluded centre, and only it, moved
    assert_eq!(filled[hole], -1.0, "the caller's value, written verbatim");
    assert_ne!(
        selected[hole], -1.0,
        "under Select the neighbours still have a statistic to give"
    );
    // the centre's own value is in the window under Select and not the answer
    // the fill gives, so the two policies are genuinely different filters
    let mut differing = 0;
    for (a, b) in selected.iter().zip(filled.iter()) {
        if a != b {
            differing += 1;
        }
    }
    assert_eq!(
        differing, 1,
        "exactly the one excluded centre may differ; a policy that leaked into \
         its neighbours' windows would move more"
    );

    // and the fill is the caller's, not a hardcoded zero
    let zeroed = filtered(&image, &mask, &element(), ExcludedCentre::Fill(0.0));
    assert_eq!(zeroed[hole], 0.0);
    assert_ne!(zeroed[hole], filled[hole]);
}

/// **The mirror case: the centre is in the population and its neighbours are
/// not.** No policy may touch this voxel — it is not an excluded centre — and
/// the answer is the centre's own value under both, because the centre is the
/// whole of the surviving window.
#[test]
fn a_centre_in_the_population_is_untouched_by_either_policy() {
    let image = image();
    let mut mask = Array3::from_elem(image.raw_dim(), false);
    let lone = [5usize, 5, 5];
    mask[lone] = true;

    let selected = filtered(&image, &mask, &element(), ExcludedCentre::Select);
    let filled = filtered(&image, &mask, &element(), ExcludedCentre::Fill(-1.0));
    assert_eq!(
        selected[lone], image[lone],
        "the only survivor of the window is the answer whatever the rank"
    );
    assert_eq!(
        filled[lone], image[lone],
        "a policy for excluded centres must not fire at a centre that is included"
    );
    // everywhere else the centre is excluded, so the fill is everywhere else
    let elsewhere = filled.iter().filter(|value| **value == -1.0).count();
    assert_eq!(elsewhere, image.len() - 1);
}

/// **An all-false population.** Every centre is excluded, so `Fill` is the whole
/// volume and `Select` is the input carried through unchanged — which is the
/// existing behaviour, restated here against the policy that could have
/// disturbed it.
#[test]
fn an_empty_population_is_the_input_or_the_fill_throughout() {
    let image = image();
    let none = Array3::from_elem(image.raw_dim(), false);

    assert_eq!(
        filtered(&image, &none, &element(), ExcludedCentre::Select),
        image,
        "with nothing to select from, the value written is the value at the centre"
    );
    assert_eq!(
        filtered(&image, &none, &element(), ExcludedCentre::Fill(7.5)),
        Array3::from_elem(image.raw_dim(), 7.5)
    );
}

/// **Where the two conditions come apart**, which is the reason the policy is
/// keyed on the centre's own bit rather than on the window turning out empty.
///
/// With an element that misses its own centre:
///
/// * a centre **in** the population whose six neighbours are all out has an
///   empty window and is *not* an excluded centre — so `Fill` must leave it to
///   the empty-window rule, which carries the centre's value;
/// * a centre **out** of the population whose neighbours are all in has a full
///   window and *is* an excluded centre — so `Fill` must fire there even though
///   there was a perfectly good statistic to take.
///
/// A policy that had been keyed on the empty window would get both of these
/// backwards, and no test using a box element could tell.
#[test]
fn the_policy_follows_the_centres_own_bit_and_not_an_empty_window() {
    let image = image();

    // a centre in the population, every neighbour out of it
    let mut lone = Array3::from_elem(image.raw_dim(), false);
    let at = [5usize, 5, 5];
    lone[at] = true;
    let filled = filtered(&image, &lone, &hollow(), ExcludedCentre::Fill(-1.0));
    assert_eq!(
        filled[at], image[at],
        "an included centre with an empty window is the empty-window rule's case, \
         not the excluded centre's, and the two must not have been collapsed"
    );
    assert_eq!(
        filled[[5, 5, 6]],
        -1.0,
        "its neighbours are excluded centres and do get the fill"
    );

    // a centre out of the population, every neighbour in it
    let mut hole = Array3::from_elem(image.raw_dim(), true);
    hole[at] = false;
    let selected = filtered(&image, &hole, &hollow(), ExcludedCentre::Select);
    let filled = filtered(&image, &hole, &hollow(), ExcludedCentre::Fill(-1.0));
    assert_eq!(
        filled[at], -1.0,
        "an excluded centre gets the fill even where the window was full"
    );
    assert_ne!(
        selected[at], -1.0,
        "and Select takes the statistic that was there"
    );
    // the hollow element never reads its own centre, so under Select this voxel
    // is decided entirely by six neighbours the mask kept
    assert!(selected[at] >= 0.0);
}

/// The two policies agree everywhere the population keeps the centre, so a mask
/// that keeps everything makes the parameter invisible. That is the statement
/// that the change is **additive** rather than a new filter.
#[test]
fn a_population_that_keeps_every_centre_makes_the_policy_invisible() {
    let image = image();
    let all = Array3::from_elem(image.raw_dim(), true);
    for element in [element(), hollow()] {
        assert_eq!(
            filtered(&image, &all, &element, ExcludedCentre::Select),
            filtered(&image, &all, &element, ExcludedCentre::Fill(-1.0)),
        );
    }
    // and the default really is the old behaviour, through the op as well as
    // through the kernel
    let op = MaskedRankFilterOp::new("masked", element(), rank(), MASK);
    assert_eq!(op.excluded_centre(), ExcludedCentre::Select);
    assert_eq!(
        MaskedRankFilterOp::new("masked", element(), rank(), MASK)
            .filling_excluded_centres(0.0)
            .excluded_centre(),
        ExcludedCentre::Fill(0.0)
    );
}

/// **The short circuit's declaration follows the policy**, because under a fill
/// the output of a constant block is no longer constant — it is the constant
/// where the mask keeps the centre and the fill where it does not, and the
/// short circuit has not read the mask.
#[test]
fn a_fill_that_is_not_the_constant_withdraws_the_short_circuit() {
    let plain = MaskedRankFilterOp::new("masked", element(), rank(), MASK);
    assert_eq!(plain.constant_maps_to(3.5), Some(3.5));

    let filling =
        MaskedRankFilterOp::new("masked", element(), rank(), MASK).filling_excluded_centres(0.0);
    assert_eq!(
        filling.constant_maps_to(3.5),
        None,
        "a block of 3.5 comes out part 3.5 and part 0.0, and which is which is a \
         fact about a level the short circuit never read"
    );
    assert_eq!(
        filling.constant_maps_to(0.0),
        Some(0.0),
        "where the fill is the constant the answer does not depend on the mask, \
         and the declaration is exactly true again"
    );

    // the values behind the declaration, so it is checked rather than reasoned
    let constant = Array3::from_elem((6, 6, 6), 3.5);
    let mut mask = Array3::from_elem(constant.raw_dim(), true);
    mask[[3, 3, 3]] = false;
    let out = filtered(&constant, &mask, &element(), ExcludedCentre::Fill(0.0));
    assert_eq!(out[[3, 3, 3]], 0.0);
    assert_eq!(out[[2, 2, 2]], 3.5);
}

// ------------------- 8. decomposition invariance, under the new policy --

/// The population is read over the element's window and the *centre's* bit is
/// read at the voxel, so the policy adds no reach — but a fill that fired on the
/// block's own idea of the mask rather than the volume's would still be a
/// silently wrong volume, and the only thing that says otherwise is the sweep.
fn chain_filling(fill: f64) -> Chain {
    Chain::sequence(vec![
        Chain::op(Binarize),
        Chain::source(0usize, Dtype::F64),
        Chain::op(
            MaskedRankFilterOp::new("masked-percentile", element(), rank(), MASK)
                .filling_excluded_centres(fill),
        ),
    ])
}

#[test]
fn a_filled_centre_is_decided_by_the_volumes_population_at_every_block_size() {
    const FILL: f64 = -1.0;
    let image = image();
    let mask = mask_of(&image);
    let expected = filtered(&image, &mask, &element(), ExcludedCentre::Fill(FILL));

    // non-vacuity: the fill really does land somewhere, and not everywhere
    let filled = expected.iter().filter(|value| **value == FILL).count();
    assert!(
        filled > 0 && filled < expected.len(),
        "{filled} of {} voxels were filled; a sweep over a volume that is all \
         fill or no fill asserts nothing",
        expected.len()
    );
    assert_ne!(
        expected,
        reference(),
        "and the policy changed the answer, or this is the old test again"
    );

    for grid in grids() {
        let chain = chain_filling(FILL);
        let decomposition = plan(&chain, &grid);
        let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
        let env =
            ArrayEnvironment::for_decomposition(image.clone().into(), &decomposition, [4, 4, 4])
                .unwrap();
        execute("filled", &workflow, &decomposition, &Hints::default(), &env).expect("a run");
        assert_eq!(
            env.output().view::<f64>().unwrap().to_owned(),
            expected,
            "block {:?}: a centre's exclusion is a fact about the volume's \
             population, not about which block the voxel landed in",
            grid.block()
        );
    }
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
