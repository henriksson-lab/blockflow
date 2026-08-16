// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A window that is carried rather than rebuilt, held to the answer of the one
// that is rebuilt.**
//
// `ops::sliding` computes the same statistic as `ops::rank` by a completely
// different route: the dense filter gathers the element at every voxel, the
// sliding one keeps a histogram and moves it one step at a time. Two routes to
// one answer is a rare and very strong test position — there is an oracle, it is
// already trusted, and it is exact rather than approximate — so most of this
// file is that comparison, run over every case where the two could plausibly
// come apart:
//
// 1. **Elements of every shape**, including ones that are elongated on each of
//    the three axes in turn (so each is chosen as the scan axis), flat ones,
//    even-sided ones whose anchor is off centre, stepped ones whose offsets are
//    not contiguous along any axis at all, an element that does **not** contain
//    its own centre, and one that does not even straddle it.
// 2. **Truncation at every face.** The step rule tests each candidate at its own
//    absolute position rather than special-casing the boundary, which is either
//    exactly right or wrong at every edge; volumes thinner than the element on
//    each axis in turn are where that shows.
// 3. **A population**, at masks that keep everything, keep nothing and keep
//    half — and both `ExcludedCentre` policies over them.
// 4. **Every axis the window could slide along**, forced, because "the answer
//    does not depend on the traversal order" is the claim the whole design rests
//    on and the chosen axis is only ever one of the three.
// 5. **Decomposition invariance**, against a whole-volume reference at seven
//    block grids. A traversal with state between voxels is exactly the shape of
//    op that can leak a scan line across a block seam, and no single block size
//    reveals it.
//
// Assertions are on the **bit pattern** of every voxel, not on a tolerance and
// not on a summary: [`assert_identical`] compares `to_le_bytes` and names the
// first coordinate that differs. For an unsigned integer that is the same
// relation `==` gives, which is the point — there is no rounding anywhere in
// either path, so anything short of exact equality is a bug rather than a
// difference of method.
//
// The last test is a **measurement** and is `#[ignore]`d, because a wall-clock
// number is not a pass or a fail. It reports ns/voxel for both paths at three
// element populations and states the ratio it actually got.

use std::time::Instant;

use ndarray::{Array3, ArrayView3};

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::error::Result;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::{
    masked_rank_filter_into_with, rank_filter_into, sliding_histogram_with_plan, Domain,
    ElementShape, ExcludedCentre, Rank, RankQuery, ScanPlan, SlidingHistogramOp,
    StructuringElement,
};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

/// Wide enough that a median is not a tie and narrow enough that a window of a
/// few hundred voxels holds repeats: both halves of a histogram's behaviour.
const DOMAIN: usize = 4096;

// ------------------------------------------------------------- fixtures --

/// A value per voxel with no structure a filter could accidentally reproduce.
///
/// Multiplicative hashing rather than a ramp: a ramp is monotone along every
/// axis, so a window's order statistic is a smooth function of position and a
/// traversal that dropped one voxel of the window would still land on a
/// plausible value. This does not forgive that.
fn image(shape: [usize; 3], domain: usize) -> Array3<u16> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        let mixed = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
            .wrapping_add((k as u64).wrapping_mul(0x1656_67B1_9E37_79F9));
        ((mixed >> 29) % domain as u64) as u16
    })
}

/// Roughly half the volume, in a pattern uncorrelated with the image, so a mask
/// that happened to keep exactly the values the filter selects cannot make the
/// two paths agree for the wrong reason.
fn half_mask(shape: [usize; 3]) -> Array3<bool> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        (i * 5 + j * 3 + k * 7) % 3 != 0
    })
}

/// Every element this file sweeps, named so a failure says which one.
///
/// The list is chosen for the *traversal's* sake rather than the filter's: one
/// element elongated on each axis in turn, so `ScanPlan` picks each of the three
/// at least once; a flat one, where the cheap axis and the contiguous axis are
/// not the same; even sides, where the anchor is not the centre; a step, where
/// the offsets are not contiguous along any axis and the leaver and joiner sets
/// are the whole element; and two that do not hold their own centre.
fn elements() -> Vec<(&'static str, StructuringElement)> {
    let hollow: Vec<[isize; 3]> = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1])
        .offsets()
        .iter()
        .copied()
        .filter(|offset| *offset != [0, 0, 0])
        .collect();
    vec![
        (
            "box 3x3x3",
            StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]),
        ),
        (
            "ellipsoid r2",
            StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 2, 2]),
        ),
        (
            "long on axis 0",
            StructuringElement::from_radius(ElementShape::Box, [4, 1, 0]),
        ),
        (
            "long on axis 1",
            StructuringElement::from_radius(ElementShape::Box, [0, 4, 1]),
        ),
        (
            "long on axis 2",
            StructuringElement::from_radius(ElementShape::Box, [1, 0, 4]),
        ),
        (
            "flat 5x5x1",
            StructuringElement::from_size(ElementShape::Box, [5, 5, 1]).unwrap(),
        ),
        (
            "even 4x6x2",
            StructuringElement::from_size(ElementShape::Box, [4, 6, 2]).unwrap(),
        ),
        (
            "stepped 7x7x3 by 2",
            StructuringElement::from_size_stepped(ElementShape::Box, [7, 7, 3], [2, 2, 2]).unwrap(),
        ),
        (
            "disc r3",
            StructuringElement::from_radius(ElementShape::Ellipsoid, [3, 3, 0]),
        ),
        (
            "hollow box",
            StructuringElement::from_offsets(hollow).unwrap(),
        ),
        (
            "off centre entirely",
            StructuringElement::from_offsets([[1, 0, 0], [2, 0, 0], [3, 0, 0], [2, 1, -1]])
                .unwrap(),
        ),
        (
            "one voxel, displaced",
            StructuringElement::from_offsets([[2, -1, 1]]).unwrap(),
        ),
    ]
}

/// Ranks that exercise both truncation conventions and both ends of the window.
fn ranks(element: &StructuringElement) -> Vec<(&'static str, Rank)> {
    vec![
        ("lowest", Rank::lowest()),
        ("median", Rank::median(element)),
        ("highest", Rank::highest(element)),
        ("a third", Rank::Nth(element.len() / 3)),
        ("ceiling 0", Rank::ceiling_percentile(0.0).unwrap()),
        ("ceiling 0.25", Rank::ceiling_percentile(0.25).unwrap()),
        ("ceiling 0.5", Rank::ceiling_percentile(0.5).unwrap()),
        ("ceiling 1", Rank::ceiling_percentile(1.0).unwrap()),
    ]
}

// ------------------------------------------------------ the two kernels --

/// The oracle: the dense gather, unchanged.
fn dense(
    input: ArrayView3<'_, u16>,
    mask: Option<ArrayView3<'_, bool>>,
    element: &StructuringElement,
    rank: Rank,
    centre: ExcludedCentre<u16>,
) -> Result<Array3<u16>> {
    let mut out = Array3::<u16>::zeros(input.raw_dim());
    match mask {
        Some(mask) => {
            masked_rank_filter_into_with(input, mask, element, rank, centre, out.view_mut())?
        }
        None => rank_filter_into(input, element, rank, out.view_mut())?,
    }
    Ok(out)
}

/// The traversal under test, along an axis the caller names.
fn sliding(
    input: ArrayView3<'_, u16>,
    mask: Option<ArrayView3<'_, bool>>,
    element: &StructuringElement,
    rank: Rank,
    centre: ExcludedCentre<u16>,
    axis: Option<usize>,
) -> Result<Array3<u16>> {
    let plan = match axis {
        Some(axis) => ScanPlan::along(element, axis),
        None => ScanPlan::new(element),
    };
    let query = RankQuery::new(rank, element);
    let mut out = Array3::<u16>::zeros(input.raw_dim());
    sliding_histogram_with_plan(
        input,
        mask,
        element,
        &plan,
        Domain::of_size(DOMAIN).unwrap(),
        &query,
        centre,
        out.view_mut(),
        "sliding under test",
    )?;
    Ok(out)
}

/// Bit-for-bit, naming the first voxel that differs.
///
/// `to_le_bytes` rather than `==` so that what is asserted is the *stored bit
/// pattern* and not some notion of numeric equality that could differ from it.
/// For `u16` the two coincide; writing it this way is what keeps the claim true
/// if this file is ever pointed at a type where they do not.
#[track_caller]
fn assert_identical(got: &Array3<u16>, want: &Array3<u16>, what: &str) {
    assert_eq!(got.dim(), want.dim(), "{what}: shape");
    for ((i, j, k), value) in got.indexed_iter() {
        let expected = want[[i, j, k]];
        assert_eq!(
            value.to_le_bytes(),
            expected.to_le_bytes(),
            "{what}: at [{i}, {j}, {k}] the sliding window gave {value} where the dense gather \
             gave {expected}"
        );
    }
}

/// A sweep that agreed because every answer was the same number asserts nothing.
#[track_caller]
fn assert_not_flat(volume: &Array3<u16>, what: &str) {
    let first = volume.iter().next().copied().unwrap_or(0);
    assert!(
        volume.iter().any(|&value| value != first),
        "{what}: the reference is the constant {first} everywhere, so agreeing with it is free"
    );
}

// ------------------------------------- 1. the unmasked filter, exactly --

#[test]
fn the_carried_window_is_the_gathered_window_at_every_element_and_rank() {
    for shape in [[11, 9, 7], [6, 5, 4]] {
        let input = image(shape, DOMAIN);
        for (element_name, element) in elements() {
            let holds_centre = element.offsets().contains(&[0, 0, 0]);
            for (rank_name, rank) in ranks(&element) {
                let want = match dense(input.view(), None, &element, rank, ExcludedCentre::Select) {
                    Ok(want) => want,
                    // The dense path refuses an element that misses its own
                    // centre without a mask; so must the sliding one, and that
                    // is asserted on its own below.
                    Err(_) => {
                        assert!(!holds_centre);
                        continue;
                    }
                };
                let got = sliding(
                    input.view(),
                    None,
                    &element,
                    rank,
                    ExcludedCentre::Select,
                    None,
                )
                .unwrap();
                assert_identical(
                    &got,
                    &want,
                    &format!("{shape:?} / {element_name} / {rank_name}"),
                );
            }
        }
    }
}

/// Non-vacuity for the sweep above: the reference really does vary, and the
/// ranks really do disagree with each other.
#[test]
fn the_reference_the_sweep_agrees_with_is_not_a_constant() {
    let shape = [11, 9, 7];
    let input = image(shape, DOMAIN);
    let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
    let lowest = dense(
        input.view(),
        None,
        &element,
        Rank::lowest(),
        ExcludedCentre::Select,
    )
    .unwrap();
    let highest = dense(
        input.view(),
        None,
        &element,
        Rank::highest(&element),
        ExcludedCentre::Select,
    )
    .unwrap();
    assert_not_flat(&lowest, "the minimum filter");
    assert_not_flat(&highest, "the maximum filter");
    assert_ne!(lowest, highest, "every rank gave the same volume");
}

// ------------------------------------------ 2. truncation at every face --

/// Volumes thinner than the element on one axis at a time, so that every voxel
/// is a boundary voxel on that axis and the surviving population is smaller than
/// the element everywhere.
#[test]
fn a_window_truncated_at_every_face_agrees_with_the_dense_clamp() {
    let element = StructuringElement::from_radius(ElementShape::Box, [2, 2, 2]);
    for shape in [
        [1, 9, 8],
        [9, 1, 8],
        [9, 8, 1],
        [2, 2, 2],
        [1, 1, 1],
        [3, 12, 1],
        [1, 1, 15],
    ] {
        let input = image(shape, DOMAIN);
        for (rank_name, rank) in ranks(&element) {
            let want = dense(input.view(), None, &element, rank, ExcludedCentre::Select).unwrap();
            let got = sliding(
                input.view(),
                None,
                &element,
                rank,
                ExcludedCentre::Select,
                None,
            )
            .unwrap();
            assert_identical(&got, &want, &format!("{shape:?} / {rank_name}"));
        }
    }
}

/// An element wider than the volume on every axis: every window is truncated on
/// both sides of every axis at once, which is the case the per-candidate bounds
/// test either handles by construction or does not handle at all.
#[test]
fn an_element_wider_than_the_volume_agrees() {
    let shape = [3, 4, 2];
    let input = image(shape, DOMAIN);
    let element = StructuringElement::from_radius(ElementShape::Box, [5, 5, 5]);
    for (rank_name, rank) in ranks(&element) {
        let want = dense(input.view(), None, &element, rank, ExcludedCentre::Select).unwrap();
        let got = sliding(
            input.view(),
            None,
            &element,
            rank,
            ExcludedCentre::Select,
            None,
        )
        .unwrap();
        assert_identical(&got, &want, rank_name);
    }
}

// --------------------------------------------------- 3. with a mask --

#[test]
fn a_masked_window_agrees_at_every_element_rank_and_policy() {
    let shape = [11, 9, 7];
    let input = image(shape, DOMAIN);
    let masks: Vec<(&str, Array3<bool>)> = vec![
        (
            "all true",
            Array3::from_elem((shape[0], shape[1], shape[2]), true),
        ),
        (
            "all false",
            Array3::from_elem((shape[0], shape[1], shape[2]), false),
        ),
        ("half", half_mask(shape)),
    ];
    let policies = [
        ("select", ExcludedCentre::Select),
        ("fill 0", ExcludedCentre::Fill(0u16)),
        ("fill 4095", ExcludedCentre::Fill(4095u16)),
    ];
    for (mask_name, mask) in &masks {
        for (element_name, element) in elements() {
            for (rank_name, rank) in ranks(&element) {
                for (policy_name, policy) in policies {
                    let what =
                        format!("{mask_name} / {element_name} / {rank_name} / {policy_name}");
                    let want =
                        dense(input.view(), Some(mask.view()), &element, rank, policy).unwrap();
                    let got = sliding(
                        input.view(),
                        Some(mask.view()),
                        &element,
                        rank,
                        policy,
                        None,
                    )
                    .unwrap();
                    assert_identical(&got, &want, &what);
                }
            }
        }
    }
}

/// The mask must be doing something, or the test above is the unmasked one
/// three times.
#[test]
fn the_mask_changes_the_answer_it_is_asserted_against() {
    let shape = [11, 9, 7];
    let input = image(shape, DOMAIN);
    let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
    let rank = Rank::median(&element);
    let plain = sliding(
        input.view(),
        None,
        &element,
        rank,
        ExcludedCentre::Select,
        None,
    )
    .unwrap();
    let mask = half_mask(shape);
    let masked = sliding(
        input.view(),
        Some(mask.view()),
        &element,
        rank,
        ExcludedCentre::Select,
        None,
    )
    .unwrap();
    assert_ne!(plain, masked, "the population made no difference");

    let filled = sliding(
        input.view(),
        Some(mask.view()),
        &element,
        rank,
        ExcludedCentre::Fill(7),
        None,
    )
    .unwrap();
    let count = filled.iter().filter(|&&value| value == 7).count();
    assert!(
        count > 0 && count < filled.len(),
        "{count} of {} voxels were filled",
        filled.len()
    );
}

/// An element that does not contain its own centre is where "the centre is
/// excluded" and "the window came out empty" stop implying one another, and it
/// is the only shape at which a `Fill` reaches the empty-window arm at all.
#[test]
fn an_element_that_misses_its_own_centre_agrees_on_both_arms() {
    let shape = [9, 8, 7];
    let input = image(shape, DOMAIN);
    let hollow: Vec<[isize; 3]> = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1])
        .offsets()
        .iter()
        .copied()
        .filter(|offset| *offset != [0, 0, 0])
        .collect();
    let element = StructuringElement::from_offsets(hollow).unwrap();
    let mask = half_mask(shape);
    for (rank_name, rank) in ranks(&element) {
        for policy in [
            ExcludedCentre::Select,
            ExcludedCentre::Fill(0u16),
            ExcludedCentre::Fill(1234u16),
        ] {
            let want = dense(input.view(), Some(mask.view()), &element, rank, policy).unwrap();
            let got = sliding(
                input.view(),
                Some(mask.view()),
                &element,
                rank,
                policy,
                None,
            )
            .unwrap();
            assert_identical(&got, &want, &format!("{rank_name} / {policy:?}"));
        }
    }
}

/// Unmasked, an element that misses its own centre is a malformed element, and
/// both paths say so rather than one of them inventing a value.
#[test]
fn an_empty_window_without_a_population_is_refused_by_both_paths() {
    let input = image([1, 1, 1], DOMAIN);
    let element = StructuringElement::from_offsets([[3, 0, 0]]).unwrap();
    let rank = Rank::lowest();
    assert!(dense(input.view(), None, &element, rank, ExcludedCentre::Select).is_err());
    assert!(sliding(
        input.view(),
        None,
        &element,
        rank,
        ExcludedCentre::Select,
        None
    )
    .is_err());
}

// ------------------------------------------ 4. the axis does not matter --

/// The claim the whole design rests on: the multiset under the window is a
/// function of the window, so the order the traversal reaches its voxels in
/// cannot change the answer. Forced along all three axes rather than trusting
/// the one `ScanPlan::new` picks.
#[test]
fn every_scan_axis_gives_the_same_volume() {
    let shape = [9, 8, 7];
    let input = image(shape, DOMAIN);
    let mask = half_mask(shape);
    for (element_name, element) in elements() {
        for (rank_name, rank) in ranks(&element) {
            let want = dense(
                input.view(),
                Some(mask.view()),
                &element,
                rank,
                ExcludedCentre::Select,
            )
            .unwrap();
            for axis in 0..3 {
                let got = sliding(
                    input.view(),
                    Some(mask.view()),
                    &element,
                    rank,
                    ExcludedCentre::Select,
                    Some(axis),
                )
                .unwrap();
                assert_identical(
                    &got,
                    &want,
                    &format!("{element_name} / {rank_name} / axis {axis}"),
                );
            }
        }
    }
}

// -------------------------------------------- 5. decomposition holds --

const VOLUME: [usize; 3] = [16, 12, 10];
/// Written by phase 0, read by phase 1 as the window's population.
const MASK: usize = 1;
/// Keeps roughly half the volume.
const CUT: u16 = 2048;

fn decomposed_element() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, [1, 1, 1])
}

fn decomposed_rank() -> Rank {
    Rank::ceiling_percentile(0.25).unwrap()
}

/// `image > CUT`, into a `Bool` level — the mask producer. Kept here rather than
/// taken from `src/ops` because what is under test is the consumer.
struct Binarize;

impl BlockOp for Binarize {
    fn name(&self) -> &'static str {
        "binarize"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::U16
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let source = input.view::<u16>()?;
        let mut out = out.view_mut::<bool>()?;
        ndarray::Zip::from(&mut out)
            .and(&source)
            .for_each(|slot, &value| *slot = value > CUT);
        Ok(())
    }
}

fn chain(centre: ExcludedCentre<f64>) -> Chain {
    Chain::sequence(vec![
        Chain::op(Binarize),
        Chain::source(0usize, Dtype::U16),
        Chain::op(
            SlidingHistogramOp::rank(
                "sliding-percentile",
                decomposed_element(),
                decomposed_rank(),
                Domain::of_size(DOMAIN).unwrap(),
            )
            .masked_by(MASK)
            .with_excluded_centre(centre),
        ),
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
        dtype: Dtype::U16,
        phases,
        chain_reach: [1, 1, 1],
    };
    plan.declare_dtypes(chain).unwrap();
    plan.declare_source_levels(chain).unwrap();
    plan
}

/// Block grids that split the volume on each axis alone, on two of them and on
/// all three — so a scan line is cut on the axis the window slides along
/// whichever axis that turns out to be.
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

fn run(grid: &BlockGrid, centre: ExcludedCentre<f64>) -> Array3<u16> {
    let chain = chain(centre);
    let decomposition = plan(&chain, grid);
    let workflow = Workflow::new(chain, VOLUME, Dtype::U16);
    let env = ArrayEnvironment::for_decomposition(
        image(VOLUME, DOMAIN).into(),
        &decomposition,
        [4, 4, 4],
    )
    .unwrap();
    execute(
        "sliding",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .expect("a run");
    env.output().view::<u16>().unwrap().to_owned()
}

fn whole_volume(centre: ExcludedCentre<u16>) -> Array3<u16> {
    let input = image(VOLUME, DOMAIN);
    let mask = input.mapv(|value| value > CUT);
    dense(
        input.view(),
        Some(mask.view()),
        &decomposed_element(),
        decomposed_rank(),
        centre,
    )
    .unwrap()
}

#[test]
fn every_decomposition_gives_the_whole_volume_answer() {
    let expected = whole_volume(ExcludedCentre::Select);
    assert_not_flat(&expected, "the whole-volume reference");
    for grid in grids() {
        assert_identical(
            &run(&grid, ExcludedCentre::Select),
            &expected,
            &format!("block {:?}", grid.block()),
        );
    }
}

/// The same, with a fill: whether a centre is excluded is a fact about the
/// volume's population, and a scan line that started inside a block would decide
/// it from the block's own view.
#[test]
fn a_filled_centre_survives_every_decomposition() {
    const FILL: f64 = 11.0;
    let expected = whole_volume(ExcludedCentre::Fill(FILL as u16));
    let filled = expected
        .iter()
        .filter(|&&value| value == FILL as u16)
        .count();
    assert!(
        filled > 0 && filled < expected.len(),
        "{filled} of {} voxels were filled",
        expected.len()
    );
    for grid in grids() {
        assert_identical(
            &run(&grid, ExcludedCentre::Fill(FILL)),
            &expected,
            &format!("block {:?}", grid.block()),
        );
    }
}

/// The op declares the mask over the element's own window, which is what makes
/// the plan fetch it at the reach the kernel reads it at.
#[test]
fn the_op_declares_its_population_at_the_elements_reach() {
    let op = SlidingHistogramOp::rank(
        "sliding",
        decomposed_element(),
        decomposed_rank(),
        Domain::of_size(DOMAIN).unwrap(),
    )
    .masked_by(MASK);
    let declared = op.source_inputs(VOLUME);
    assert_eq!(declared.len(), 1);
    assert_eq!(declared[0].level, MASK);
    assert_eq!(op.reach(0, 100), 1);
    assert_eq!(op.reach(1, 100), 1);
    assert_eq!(op.reach(2, 100), 1);
}

/// An op that declares a population and is applied without one refuses, rather
/// than filtering the input alone and returning a plausible volume.
#[test]
fn the_op_refuses_to_run_without_its_population() {
    let op = SlidingHistogramOp::rank(
        "sliding",
        decomposed_element(),
        decomposed_rank(),
        Domain::of_size(DOMAIN).unwrap(),
    )
    .masked_by(MASK);
    let input: Voxels = image([4, 4, 4], DOMAIN).into();
    let mut out: Voxels = Array3::<u16>::zeros((4, 4, 4)).into();
    let at = Anchor::new([0, 0, 0], VOLUME);
    assert!(op.apply(&input, &mut out, &at).is_err());
}

// ------------------------------------------------- 6. the measurement --

/// What the sliding traversal costs against the dense gather, per voxel, at
/// three element populations and a 4096-bin domain.
///
/// **Ignored, because a wall-clock number is neither a pass nor a fail.** Run it
/// with
///
/// ```text
/// cargo test --release --test sliding_histogram -- --ignored --nocapture
/// ```
///
/// and read the table. A debug build measures the wrong program entirely — the
/// dense path's `select_nth_unstable` and this file's histogram walk are both
/// dominated by bounds checks and unoptimised iterators there.
///
/// Both paths are single-threaded, and the volumes are sized so that the dense
/// path finishes in a reasonable time rather than to flatter either one. The
/// element populations are the ones that matter: a line, a disc, and a plate
/// whose window is 40 000 voxels.
///
/// **The best of several repetitions**, and the *same* number of them for both
/// paths. Contamination on a shared machine is one-sided — something else can
/// take the core away but nothing can give this one more of it — so the minimum
/// is the robust statistic and a mean is a measurement of the machine's other
/// tenants. A single shot measures page faults and a cold cache: it made the
/// sliding path look three to four times worse than it is, and it would have
/// done the same to the dense one.
#[test]
#[ignore = "a measurement, not an assertion; run with --release --ignored --nocapture"]
fn the_sliding_traversal_against_the_dense_gather() {
    struct Case {
        what: &'static str,
        shape: [usize; 3],
        element: StructuringElement,
        /// Fewer where one repetition already costs seconds; the same count is
        /// used for both paths of the same case, which is the part that matters.
        repetitions: usize,
    }

    let cases = vec![
        Case {
            what: "line 150x1x1",
            shape: [192, 24, 6],
            element: StructuringElement::from_size(ElementShape::Box, [150, 1, 1]).unwrap(),
            repetitions: 7,
        },
        Case {
            what: "disc r15",
            shape: [96, 96, 6],
            element: StructuringElement::from_radius(ElementShape::Ellipsoid, [15, 15, 0]),
            repetitions: 7,
        },
        Case {
            what: "plate 200x200x1",
            shape: [160, 160, 1],
            element: StructuringElement::from_size(ElementShape::Box, [200, 200, 1]).unwrap(),
            repetitions: 3,
        },
    ];

    println!();
    println!("domain {DOMAIN} bins, rank = ceiling percentile 0.25, single-threaded");
    println!(
        "{:<20} {:>8} {:>6} {:>5} {:>13} {:>13} {:>9}",
        "element", "|E|", "step", "axis", "dense ns/vx", "slide ns/vx", "speed-up"
    );
    for case in cases {
        let input = image(case.shape, DOMAIN);
        let voxels = case.shape.iter().product::<usize>() as f64;
        let rank = Rank::ceiling_percentile(0.25).unwrap();
        let plan = ScanPlan::new(&case.element);

        let mut dense_ns = f64::MAX;
        let mut want = None;
        for _ in 0..case.repetitions {
            let started = Instant::now();
            let answer = dense(
                input.view(),
                None,
                &case.element,
                rank,
                ExcludedCentre::Select,
            )
            .unwrap();
            dense_ns = dense_ns.min(started.elapsed().as_secs_f64() * 1e9 / voxels);
            want = Some(answer);
        }

        let mut sliding_ns = f64::MAX;
        let mut got = None;
        for _ in 0..case.repetitions {
            let started = Instant::now();
            let answer = sliding(
                input.view(),
                None,
                &case.element,
                rank,
                ExcludedCentre::Select,
                None,
            )
            .unwrap();
            sliding_ns = sliding_ns.min(started.elapsed().as_secs_f64() * 1e9 / voxels);
            got = Some(answer);
        }

        // The measurement is worthless if the two are not computing the same
        // thing, so it is checked here rather than trusted from the sweep above.
        assert_identical(&got.unwrap(), &want.unwrap(), case.what);

        println!(
            "{:<20} {:>8} {:>6} {:>5} {:>13.1} {:>13.1} {:>8.1}x",
            case.what,
            case.element.len(),
            plan.step_size(),
            plan.axis(),
            dense_ns,
            sliding_ns,
            dense_ns / sliding_ns
        );
    }
    println!();
}
