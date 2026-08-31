// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `blockflow::ops::features`, and — the reason it was
// written before anything consumes the stack — the **measurement** step 2 of
// `docs/design/pixel-classification.md` asks for.
//
// That document names the 91-arm fan-in as the plan's largest unknown:
//
//   > `Chain::Parallel` was designed for a diamond with two arms. Whether the
//   > partition search, the reach fold and the residency accounting survive two
//   > orders of magnitude more is not known.
//
// So this file answers it, and answers it before a classifier exists to
// confound the numbers. `the_planner_survives_the_whole_stack` is the assertion
// half — it runs and states what must not regress — and
// `print_the_fan_in_measurement` is the table, `#[ignore]`d for the reason every
// timing in this crate is.

use blockflow::decomposition::{Constraints, CostModel};
use blockflow::env::ArrayEnvironment;
use blockflow::op::{Anchor, Chain, Combine};
use blockflow::ops::element::{ElementShape, Rank, StructuringElement};
use blockflow::ops::local::{LocalStatistic, LocalStatisticOp, Statistic};
use blockflow::ops::rank::RankFilterOp;
use blockflow::ops::{Arithmetic, ArithmeticCombine};
use blockflow::ops::{Family, FeatureStack, Geometry};
use blockflow::strategy::{
    execute, Enumerating, Hints, PartitionSearch, SchedulePriority, Strategy, Workflow,
};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

use ndarray::Array3;

/// Labkit's own default scale list, five sigmas, which is what makes the
/// channel count 91 rather than an arbitrary number.
const SIGMAS: [f64; 5] = [1.0, 2.0, 4.0, 8.0, 16.0];

/// **Large, and it has to be**: the stack's reach at the widest sigma is 193
/// voxels, so a volume of 128 cannot be tiled by it at all. See
/// `the_planner_survives_the_whole_stack`, which states where 193 comes from.
/// Nothing is allocated at this extent — these tests plan, they do not run.
const VOLUME: [usize; 3] = [512, 512, 512];

fn constraints() -> Constraints {
    Constraints {
        budget_bytes: None,
        expected_concurrency: 1,
        model: CostModel::default(),
        ..Default::default()
    }
}

/// A join that exists only so the stack can be closed into one chain. The real
/// one is the forest predictor; this one folds the arms with a maximum, which
/// costs nothing to price and — being an `ArithmeticCombine` — is machinery the
/// crate already proves in `tests/fan_in.rs`.
///
/// **What it does not stand in for**, stated here because every measurement in
/// this file inherits the limit: an `ArithmeticCombine` declares a
/// [`Combine::fold_carrier`], so a fan-in joined by it holds **three** block
/// buffers whatever its arity — the partial, the branch just finished, and their
/// join. A forest predictor cannot declare one. It needs all 91 channels at a
/// voxel simultaneously to walk a tree, which is the definition of not being a
/// left fold over pairs, so its fan-in holds one buffer **per arm**. Every
/// residency number below is therefore a floor for the real chain and not an
/// estimate of it; `the_placeholder_folds_and_the_predictor_will_not` pins the
/// distinction so that a later reader cannot mistake one for the other, and the
/// allocator measurement belongs with the predictor in step 3.
fn placeholder_combine() -> Box<dyn Combine> {
    Box::new(ArithmeticCombine::new("stack", Arithmetic::Maximum))
}

/// **The scope of every residency claim in this file.**
///
/// See [`placeholder_combine`]. Asserted rather than left in prose because the
/// difference is a factor of thirty at 91 arms and the two chains are otherwise
/// indistinguishable — same ops, same reach, same plan, same priced cost.
#[test]
fn the_placeholder_folds_and_the_predictor_will_not() {
    let combine = ArithmeticCombine::new("stack", Arithmetic::Maximum);
    let arms = vec![Dtype::F64; 91];
    assert!(
        combine.fold_carrier(&arms).is_some(),
        "the placeholder stopped folding, so the measurements here now describe a \
         different residency than they say they do"
    );
    // And the plan cannot tell the two apart: `working_set_bytes_per_block` is
    // `resident_voxels x bytes_per_voxel x 2.0` whatever the arity, so no
    // decomposition of either chain would differ. That is the gap `budget.rs`
    // records under `FrameworkFigure::Assumed`, at an arity two orders of
    // magnitude past anything measured for it.
    let stack = FeatureStack::labkit(&SIGMAS).unwrap();
    assert_eq!(stack.branches().unwrap().len(), 91);
}

// ------------------------------------------------------ 1. the arithmetic --

/// **91, and where each one comes from.**
///
/// The design document's memory arithmetic — 0.4 TiB for a `1024^3` stack — rests
/// on this count, so it is asserted term by term rather than as a total. A total
/// alone would pass for a stack that had lost the differences of Gaussians and
/// gained ten Gaussians.
#[test]
fn the_labkit_stack_at_five_sigmas_is_ninety_one_channels() {
    let stack = FeatureStack::labkit(&SIGMAS).unwrap();
    let n = SIGMAS.len();
    let expected = [
        (Family::Original, 1),
        (Family::Gaussian, n),
        (Family::DifferenceOfGaussians, n * (n - 1) / 2),
        (Family::GradientMagnitude, n),
        (Family::LaplacianOfGaussian, n),
        (Family::Hessian, 3 * n),
        (Family::StructureTensor, 6 * n),
        (Family::Morphological, 4 * n),
    ];
    for (family, want) in expected {
        let only = FeatureStack::labkit(&SIGMAS)
            .unwrap()
            .with_families(&[family])
            .unwrap();
        assert_eq!(only.len(), want, "{family:?}");
        assert_eq!(only.channels().unwrap().len(), want, "{family:?}");
    }
    assert_eq!(expected.iter().map(|(_, count)| count).sum::<usize>(), 91);
    assert_eq!(stack.len(), 91);
    assert_eq!(stack.channels().unwrap().len(), 91);
}

/// Plane-wise the two eigenvalue families shrink and **nothing else does**,
/// which is the claim `Geometry`'s documentation makes: the mode is a scale of
/// zero, not a different stack.
#[test]
fn the_plane_wise_stack_differs_only_in_the_eigenvalue_families() {
    let volumetric = FeatureStack::labkit(&SIGMAS).unwrap();
    let plane_wise = FeatureStack::labkit(&SIGMAS)
        .unwrap()
        .with_geometry(Geometry::PlaneWise { normal: 0 })
        .unwrap();
    for family in Family::ALL {
        let one = |stack: &FeatureStack| stack.clone().with_families(&[family]).unwrap().len();
        let (three_d, two_d) = (one(&volumetric), one(&plane_wise));
        match family {
            Family::Hessian => assert_eq!((three_d, two_d), (15, 10)),
            Family::StructureTensor => assert_eq!((three_d, two_d), (30, 20)),
            _ => assert_eq!(three_d, two_d, "{family:?} changed with the geometry"),
        }
    }
    assert_eq!(plane_wise.len(), 91 - 5 - 10);
}

/// A forest's split names a column, so two columns may not share a name.
#[test]
fn every_channel_has_a_distinct_name() {
    for geometry in [Geometry::Volumetric, Geometry::PlaneWise { normal: 0 }] {
        let stack = FeatureStack::labkit(&SIGMAS)
            .unwrap()
            .with_geometry(geometry)
            .unwrap();
        let mut names = stack.channel_names().unwrap();
        let total = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), total, "{geometry:?} repeats a channel name");
    }
}

/// The refusals, each of something that would otherwise be a column carrying
/// nothing.
#[test]
fn a_stack_refuses_what_would_give_it_an_empty_column() {
    assert!(FeatureStack::labkit(&[]).is_err());
    assert!(FeatureStack::labkit(&[1.0, 1.0]).is_err());
    assert!(FeatureStack::labkit(&[0.0]).is_err());
    assert!(FeatureStack::labkit(&[f64::NAN]).is_err());
    assert!(FeatureStack::labkit(&SIGMAS)
        .unwrap()
        .with_families(&[])
        .is_err());
    assert!(FeatureStack::labkit(&SIGMAS)
        .unwrap()
        .with_geometry(Geometry::PlaneWise { normal: 3 })
        .is_err());
    // A single-channel stack is not a fan-in, and says so rather than building
    // a degenerate `Parallel`.
    assert!(FeatureStack::labkit(&SIGMAS)
        .unwrap()
        .with_families(&[Family::Original])
        .unwrap()
        .into_chain(placeholder_combine())
        .is_err());
}

// ------------------------------------------------ 2. the planner survives --

/// **The unknown the design document names, answered.**
///
/// The whole 91-arm stack is built, folded and planned. What is asserted is what
/// a regression would break, not a particular plan:
///
/// * the chain builds and its reach folds to the widest arm's rather than to
///   their sum — a fan-in takes the **maximum**, and at 91 arms an accidental
///   sum would be enormous and would still tile, so nothing else would catch it;
/// * the search prices the `n(n+1)/2` contiguous runs it is supposed to and
///   refuses none of them;
/// * the plan checks out.
#[test]
fn the_planner_survives_the_whole_stack() {
    let stack = FeatureStack::labkit(&SIGMAS).unwrap();
    let chain = stack.into_chain(placeholder_combine()).unwrap();

    // **The widest arm decides the fan-in's reach, and the widest arm is not the
    // one it looks like.** At sigma 16 and truncate 3:
    //
    //   morphological box    floor(1 + 2*16) = 33
    //   Gaussian             ceil(3*16)      = 48
    //   gradient, Hessian, LoG                 49   (48, and the stencil's 1)
    //   structure tensor g=1                   97   (48 + 1 + 48)
    //   structure tensor g=3                  193   (48 + 1 + ceil(3*48))
    //
    // So the structure tensor at gamma 3 sets it, at four times the next arm and
    // six times the box a reader would have guessed. That is a fact about the
    // stack rather than about this crate, and it is the single most consequential
    // number the step-2 measurement turned up: a halo of 193 per side is 386
    // voxels of overlap around every block, so at the default 32-voxel candidate
    // a block reads 418 voxels per axis to write 32. See
    // `print_the_fan_in_measurement` for what that does to the plan.
    let reach = chain.reach3(&VOLUME);
    assert_eq!(reach, [193, 193, 193], "the fan-in folded its arms wrongly");
    for axis in 0..3 {
        assert!(
            reach[axis] < VOLUME[axis],
            "the stack reaches the whole volume, so no plan can tile it"
        );
    }
    // A fan-in takes the maximum over its arms, never the sum. Both halves are
    // asserted: that the fold *is* the maximum, and that the arms are spread
    // enough — their sum is fifteen times their maximum — for the two to be
    // distinguishable at all. The second is what stops this becoming vacuous if
    // the scale list is ever narrowed.
    let arm_reaches: Vec<usize> = FeatureStack::labkit(&SIGMAS)
        .unwrap()
        .branches()
        .unwrap()
        .iter()
        .map(|arm| arm.reach3(&VOLUME)[0])
        .collect();
    assert_eq!(reach[0], arm_reaches.iter().copied().max().unwrap());
    assert!(
        arm_reaches.iter().sum::<usize>() > 10 * reach[0],
        "the arms are too alike for this to distinguish a maximum from a sum"
    );

    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let slots = workflow.chain.slots().len();
    let strategy = Enumerating {
        concurrency: 4,
        priority: SchedulePriority::PhaseMajor,
        search: PartitionSearch::Dp,
    };
    let (decomposition, account) = strategy
        .decompose_accounted(&workflow, &constraints())
        .expect("the 91-arm stack must be plannable");
    decomposition.check().expect("and the plan must check out");

    assert_eq!(account.slots, slots);
    assert_eq!(
        account.runs_priced + account.runs_forbidden_by_barrier,
        slots * (slots + 1) / 2,
        "the search did not cover the space it claims to"
    );
    assert_eq!(account.runs_refused, 0);
    assert!(!account.chosen.is_empty());
}

/// **The two searches still agree at 91 arms.** `PartitionSearch`'s whole
/// contract is that the dynamic program and the exhaustive search return the
/// *same* partition, and that contract has been swept over small random chains.
/// This is the one place it is checked at the size the workload actually has.
#[test]
fn the_two_partition_searches_agree_on_the_whole_stack() {
    // Three sigmas rather than five: the exhaustive search is exponential in the
    // slot count and this is about agreement, not about scale. The DP half of
    // the pair is exercised at 91 arms by the test above.
    let stack = FeatureStack::labkit(&[1.0, 2.0, 4.0]).unwrap();
    let build = || {
        Workflow::new(
            FeatureStack::labkit(&[1.0, 2.0, 4.0])
                .unwrap()
                .into_chain(placeholder_combine())
                .unwrap(),
            VOLUME,
            Dtype::F64,
        )
    };
    assert!(stack.len() > 40, "the fixture shrank and proves less");

    let plan = |search| {
        Enumerating {
            concurrency: 4,
            priority: SchedulePriority::PhaseMajor,
            search,
        }
        .decompose(&build(), &constraints())
        .expect("a plan")
    };
    let dp = plan(PartitionSearch::Dp);
    let exhaustive = plan(PartitionSearch::Exhaustive);
    assert_eq!(dp.n_phases(), exhaustive.n_phases());
    for (left, right) in dp.phases.iter().zip(exhaustive.phases.iter()) {
        assert_eq!(left.slots, right.slots);
        assert_eq!(left.grid.block(), right.grid.block());
    }
}

/// **The defect step 2 was written to find, pinned.**
///
/// A halo of `r` puts a floor under the working set of a block that **no choice
/// of block size can get below**: a block of edge `b` reads `(b + 2r)^3` voxels,
/// which tends to `(2r)^3` as `b` shrinks. Cutting smaller does not help — it
/// makes the amplification worse while leaving the absolute footprint almost
/// unchanged — so a budget below that floor cannot be met at all, and the
/// planner refuses rather than returning a plan that would exhaust memory.
///
/// At Labkit's five default sigmas the stack's reach is 193, so the floor is
/// `386^3 * 8 = 460 MB` per block in `f64`, before any concurrency. That is the
/// finding: **the 91-channel stack cannot be a single fused phase at those
/// sigmas under any ordinary budget**, and the reason is the halo rather than
/// the arm count the design document worried about.
///
/// Asserted in three parts, because each could regress on its own: that a
/// generous budget plans, that a budget below the floor is refused, and that the
/// refusal says why.
#[test]
fn the_halo_puts_a_floor_under_the_working_set_that_no_block_size_escapes() {
    let stack = FeatureStack::labkit(&SIGMAS).unwrap();
    let workflow = Workflow::new(
        stack.into_chain(placeholder_combine()).unwrap(),
        VOLUME,
        Dtype::F64,
    );
    let reach = workflow.chain.reach3(&VOLUME)[0];
    assert_eq!(reach, 193);

    // The floor, in bytes, for the smallest candidate the planner is offered.
    // It is within a factor of two of `(2r)^3` and would not move much if the
    // candidate were 1 rather than 32 — which is the whole point.
    let smallest = 32usize;
    let floor = ((smallest + 2 * reach) as u64).pow(3) * 8;
    let bare = ((2 * reach) as u64).pow(3) * 8;
    assert!(
        floor < 2 * bare,
        "the smallest candidate is not small enough for this to be a statement about \
         the halo rather than about the block"
    );

    let plan = |budget: Option<u64>| {
        Enumerating {
            concurrency: 1,
            priority: SchedulePriority::PhaseMajor,
            search: PartitionSearch::Dp,
        }
        .decompose(
            &Workflow::new(
                FeatureStack::labkit(&SIGMAS)
                    .unwrap()
                    .into_chain(placeholder_combine())
                    .unwrap(),
                VOLUME,
                Dtype::F64,
            ),
            &Constraints {
                split_axes: vec![0, 1, 2],
                budget_bytes: budget,
                expected_concurrency: 1,
                ..constraints()
            },
        )
    };

    // Unbudgeted, the planner declines to cut at all: one block, the whole
    // volume, no halo to pay. That is the right answer and it is only available
    // when the volume fits.
    let whole = plan(None).expect("a plan with no budget");
    assert_eq!(whole.phases[0].grid.block(), VOLUME);

    // Below the floor there is no plan, at any block size.
    let err = plan(Some(floor / 4))
        .expect_err("a budget below the halo's floor cannot be met")
        .to_string();
    assert!(
        err.contains("no partition") && err.contains("budget"),
        "the refusal does not explain itself: {err}"
    );
}

/// **The separable box is the same filter as the direct box**, which is the
/// claim `separable_extreme` rests on and the reason the morphological family
/// costs a thousandth of what it did.
///
/// Checked against the direct `RankFilterOp` over the whole box, at several
/// radii and on a fixture with a real volume boundary — the truncation is the
/// half worth checking, because a box clipped by the volume is still a product
/// of clipped intervals and that is what makes the composition exact rather than
/// approximately right in the interior.
///
/// **Byte-identical for the extremes**, which is the strong claim available for
/// a minimum and a maximum: they select a value that was read, so no arithmetic
/// happens and no ordering matters. The mean is checked separately and to a
/// tolerance, for the reason `separable_mean` documents.
#[test]
fn the_separable_box_is_the_same_filter_as_the_direct_box() {
    let volume = [17usize, 15, 13];
    let mut state: u64 = 20260831;
    let input: Voxels = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    })
    .into();

    let run = |chain: &Chain| {
        let mut out = Voxels::zeros(Dtype::F64, volume).unwrap();
        chain
            .apply(&input, &mut out, &Anchor::whole(volume))
            .unwrap();
        out.view::<f64>().unwrap().to_owned()
    };

    let mut checked = 0;
    // Sigmas chosen so that `floor(1 + 2 sigma)` lands on 1, 2, 4 and 7 — the
    // radius is the builder's business and this drives it from the outside.
    for (sigma, radius) in [(0.25f64, 1usize), (0.75, 2), (1.75, 4), (3.25, 7)] {
        let element = StructuringElement::from_size(
            ElementShape::Box,
            [2 * radius + 1, 2 * radius + 1, 2 * radius + 1],
        )
        .unwrap();
        for (label, rank) in [("min", Rank::lowest()), ("max", Rank::highest(&element))] {
            let direct = run(&Chain::op(RankFilterOp::new(
                "direct",
                element.clone(),
                rank,
            )));
            // The stack's own arm, reached through the builder so that this
            // tests what a caller gets rather than a reconstruction of it.
            let stack = FeatureStack::labkit(&[sigma])
                .unwrap()
                .with_families(&[Family::Morphological])
                .unwrap();
            let channel = stack
                .channels()
                .unwrap()
                .into_iter()
                .find(|channel| channel.name.ends_with(label))
                .unwrap();
            let separable = run(&channel.chain);
            assert_eq!(
                separable, direct,
                "{label} at radius {radius} differs between the separable and direct forms"
            );
            // **And it reads `(2r+1)^2 / 3` times less**, which is the arithmetic
            // the whole change rests on: a cube of `(2r+1)^3` against three
            // lines of `(2r+1)`. Asserted as the ratio rather than as "cheaper",
            // because "cheaper" would pass for a form that saved a constant
            // factor and the point is that the saving grows with the radius —
            // 3x at radius 1, and 1,496x at the radius Labkit's widest sigma
            // asks for.
            let edge = (2 * radius + 1) as f64;
            let direct_cost =
                Chain::op(RankFilterOp::new("direct", element.clone(), rank)).cost_per_voxel();
            let ratio = direct_cost / channel.chain.cost_per_voxel();
            assert!(
                (ratio - edge * edge / 3.0).abs() < 0.01 * edge * edge,
                "{label} at radius {radius}: the separable form is {ratio:.1}x cheaper \
                 where the arithmetic says {:.1}x",
                edge * edge / 3.0
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 8);
}

/// The mean, to a tolerance rather than to bits — see `separable_mean`.
#[test]
fn the_separable_box_mean_agrees_with_the_direct_one_to_rounding() {
    let volume = [15usize, 13, 11];
    let mut state: u64 = 4242;
    let input: Voxels = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as f64 / (1u64 << 31) as f64
    })
    .into();
    let run = |chain: &Chain| {
        let mut out = Voxels::zeros(Dtype::F64, volume).unwrap();
        chain
            .apply(&input, &mut out, &Anchor::whole(volume))
            .unwrap();
        out.view::<f64>().unwrap().to_owned()
    };

    for (sigma, radius) in [(0.25f64, 1usize), (1.25, 3)] {
        let element = StructuringElement::from_size(
            ElementShape::Box,
            [2 * radius + 1, 2 * radius + 1, 2 * radius + 1],
        )
        .unwrap();
        let direct = run(&Chain::op(LocalStatisticOp::new(
            "direct",
            LocalStatistic::new(element, [1, 1, 1], Statistic::Mean).unwrap(),
        )));
        let channel = FeatureStack::labkit(&[sigma])
            .unwrap()
            .with_families(&[Family::Morphological])
            .unwrap()
            .channels()
            .unwrap()
            .into_iter()
            .find(|channel| channel.name.ends_with("mean"))
            .unwrap();
        let separable = run(&channel.chain);
        let worst = separable
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-12,
            "the separable mean differs from the direct one by {worst:e} at radius \
             {radius}, which is more than a difference in summation order"
        );
        assert!(
            worst > 0.0 || radius == 1,
            "identical bits would be a surprise worth knowing"
        );
    }
}

/// **The two-moment deviation agrees with the two-pass one**, on data where
/// cancellation is not in play — and visibly does not, on data where it is.
///
/// Both halves are asserted. The first is the claim the default rests on; the
/// second is the claim `with_exact_deviation` exists for, and asserting it is
/// what keeps that option from being decoration. The fixture for it is a large
/// constant offset with a small modulation, which is what a detector pedestal
/// looks like.
#[test]
fn the_two_moment_deviation_agrees_except_where_it_says_it_will_not() {
    let volume = [15usize, 13, 11];
    let sigma = 1.25;

    let run = |input: &Voxels, exact: bool| {
        let channel = FeatureStack::labkit(&[sigma])
            .unwrap()
            .with_families(&[Family::Morphological])
            .unwrap()
            .with_exact_deviation(exact)
            .channels()
            .unwrap()
            .into_iter()
            .find(|channel| channel.name.ends_with("deviation"))
            .unwrap();
        let mut out = Voxels::zeros(Dtype::F64, volume).unwrap();
        channel
            .chain
            .apply(input, &mut out, &Anchor::whole(volume))
            .unwrap();
        out.view::<f64>().unwrap().to_owned()
    };

    let mut state: u64 = 31337;
    let mut draw = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };

    // Well conditioned: values of order one, spread of order one.
    let ordinary: Voxels =
        Array3::from_shape_fn((volume[0], volume[1], volume[2]), |_| draw()).into();
    let (fast, exact) = (run(&ordinary, false), run(&ordinary, true));
    let worst = fast
        .iter()
        .zip(exact.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst < 1e-12,
        "on well-conditioned data the two forms differ by {worst:e}, which is more than \
         rounding"
    );
    assert!(
        exact.iter().any(|&value| value > 0.01),
        "the fixture is flat"
    );

    // Badly conditioned: a pedestal of 3e4 with a modulation of 1e-3, so
    // `(mean / sd)^2` is about 1e15 and the identity has no digits left.
    let pedestal: Voxels = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |_| {
        3.0e4 + 1.0e-3 * draw()
    })
    .into();
    let (fast, exact) = (run(&pedestal, false), run(&pedestal, true));
    let relative = fast
        .iter()
        .zip(exact.iter())
        .filter(|(_, &want)| want > 0.0)
        .map(|(got, want)| (got - want).abs() / want)
        .fold(0.0f64, f64::max);
    assert!(
        relative > 0.1,
        "the two-moment form was expected to lose most of its digits on a pedestal and \
         lost only {relative:.1e}; either the fixture is not badly conditioned or the \
         documentation on `separable_deviation` overstates the risk"
    );
    // And it never produces a NaN, however bad the cancellation — which is what
    // the clamp before the square root is for.
    assert!(fast.iter().all(|value| value.is_finite() && *value >= 0.0));
}

// ------------------------------------------------ 3. it computes something --

/// **The stack runs, block-decomposed, and agrees with itself run whole.**
///
/// A small stack on a small volume — the point is the fan-in and the halo, and
/// both are size-independent. Every arm's own decomposition invariance is
/// covered by that op's suite; what is new here is 21 of them under one halo.
#[test]
fn a_small_stack_reproduces_its_whole_volume_reference_under_decomposition() {
    let volume = [24usize, 20, 18];
    let stack = FeatureStack::labkit(&[1.0, 2.0])
        .unwrap()
        .with_truncate(2.0)
        .unwrap();
    let chain = stack.into_chain(placeholder_combine()).unwrap();
    let reach = chain.reach3(&volume);
    assert!(reach.iter().enumerate().all(|(axis, &r)| r < volume[axis]));

    let mut state: u64 = 20260831;
    let input = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    });

    let source: blockflow::voxels::Voxels = input.clone().into();
    let mut out = blockflow::voxels::Voxels::zeros(Dtype::F64, volume).unwrap();
    chain
        .apply(&source, &mut out, &Anchor::whole(volume))
        .expect("the whole-volume reference must run");
    let want = out.view::<f64>().unwrap().to_owned();

    let workflow = Workflow::new(
        FeatureStack::labkit(&[1.0, 2.0])
            .unwrap()
            .with_truncate(2.0)
            .unwrap()
            .into_chain(placeholder_combine())
            .unwrap(),
        volume,
        Dtype::F64,
    );
    let strategy = Enumerating {
        concurrency: 2,
        priority: SchedulePriority::PhaseMajor,
        search: PartitionSearch::Dp,
    };
    let decomposition = strategy
        .decompose(
            &workflow,
            &Constraints {
                block_candidates: vec![8, 12, 16],
                ..constraints()
            },
        )
        .expect("a plan");
    decomposition.check().unwrap();
    let env = ArrayEnvironment::new(input.into(), decomposition.n_phases(), [4, 4, 4]).unwrap();
    execute("stack", &workflow, &decomposition, &Hints::default(), &env).unwrap();
    assert_eq!(env.output().view::<f64>().unwrap().to_owned(), want);
}

// --------------------------------------------------------- 4. the table --

/// The measurement step 2 of the design document asks for: what the fan-in costs
/// the planner as the arm count grows.
///
/// ```text
/// cargo test --release --test feature_stack -- --ignored --nocapture
/// ```
///
/// The recorded run, on the machine this crate was developed on, and **the three
/// things it found**:
///
/// ```text
/// unbudgeted, split axis 2, candidates [32, 64, 128]
///  sigmas arms slots reach phases     block   read  amplification  search ms
///       1   17     1    13      1  512x512x128  512x512x154    1.2        0.3
///       2   34     1    25      1  512x512x128  512x512x178    1.4        0.3
///       3   52     1    49      1  512x512x128  512x512x226    1.8        0.5
///       4   71     1    97      1  512x512x128  512x512x322    2.5        1.0
///       5   91     1   193      1  512x512x128  512x512x512    4.0        0.3
/// ```
///
/// **1. The 91-arm fan-in is one slot, so the partition search never sees it.**
/// The `slots` column is 1 at every arm count and `runs_priced` is 1. The design
/// document's stated risk — "whether the partition search survives two orders of
/// magnitude more" — does not arise, because `Chain::Parallel` is a single slot
/// however many branches it has. That is a relief and a constraint in the same
/// fact: the planner cannot put a phase boundary *inside* the stack. All 91
/// channels are computed, joined and discarded within one block, which is
/// exactly the fusion the memory arithmetic needs, and it is not a choice the
/// planner makes — it is the only thing it can do.
///
/// **2. The search is fast and does not grow.** Sub-millisecond at every size,
/// with no trend, for the same reason: it is searching over one slot.
///
/// **3. The reach is the real cost, and it grows fourfold with the last
/// sigma.** 13, 25, 49, 97, 193 — each sigma doubles, and the structure tensor
/// at `gamma = 3` multiplies by four on top. By five sigmas the halo is 193 per
/// side, the default 128-voxel block reads the whole 512-voxel axis, and the
/// read amplification is 4. **This is the number the workload lives or dies on**,
/// and it is a property of Labkit's scale list rather than of this crate: the
/// same stack in Labkit reads the same neighbourhood. What this crate can do
/// about it is choose the block, which is what the second table measures.
#[test]
#[ignore = "a measurement, not an assertion"]
fn print_the_fan_in_measurement() {
    let header = || {
        println!(
            "{:>7} {:>5} {:>6} {:>6} {:>7} {:>14} {:>14} {:>7} {:>10}",
            "sigmas", "arms", "slots", "reach", "phases", "block", "read", "amplif", "search ms"
        );
    };
    let row = |sigmas: &[f64], constraints: &Constraints| {
        let stack = FeatureStack::labkit(sigmas).unwrap();
        let arms = stack.len();
        let workflow = Workflow::new(
            stack.into_chain(placeholder_combine()).unwrap(),
            VOLUME,
            Dtype::F64,
        );
        let slots = workflow.chain.slots().len();
        let reach = workflow.chain.reach3(&VOLUME)[0];
        let started = std::time::Instant::now();
        let planned = Enumerating {
            concurrency: 4,
            priority: SchedulePriority::PhaseMajor,
            search: PartitionSearch::Dp,
        }
        .decompose_accounted(&workflow, constraints);
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        let (decomposition, _account) = match planned {
            Ok(planned) => planned,
            Err(err) => {
                println!(
                    "{:>7} {arms:>5} {slots:>6} {reach:>6}   refused: {err}",
                    sigmas.len()
                );
                return;
            }
        };
        let phase = &decomposition.phases[0];
        let block = phase.grid.block();
        // Voxels read per voxel written, in three dimensions — the number this
        // table exists for. The read extent is clamped to the volume, because a
        // halo that runs off the end reads nothing: a block covering an axis
        // whole has no overlap on it however wide the reach is.
        let read: Vec<usize> = (0..3)
            .map(|axis| (block[axis] + 2 * reach).min(VOLUME[axis]))
            .collect();
        let amplification: f64 = (0..3)
            .map(|axis| read[axis] as f64 / block[axis] as f64)
            .product();
        println!(
            "{:>7} {arms:>5} {slots:>6} {reach:>6} {:>7} {:>14} {:>14} {amplification:>7.1} \
             {elapsed:>10.1}",
            sigmas.len(),
            decomposition.n_phases(),
            format!("{}x{}x{}", block[0], block[1], block[2]),
            format!("{}x{}x{}", read[0], read[1], read[2]),
        );
    };

    println!("unbudgeted, split axis 2, candidates [32, 64, 128]");
    header();
    for count in 1..=SIGMAS.len() {
        row(&SIGMAS[..count], &constraints());
    }

    // **The same stack forced to cut on every axis.** The default splits axis 2
    // alone, which lets a block keep two axes whole and hides most of the halo:
    // a `512x512x128` block pays overlap on one axis out of three. A run large
    // enough to need cubes pays it on all three, and that is the case the memory
    // arithmetic in the design document describes.
    println!("\ncut on every axis, candidates [32, 64, 128]");
    header();
    for count in 1..=SIGMAS.len() {
        row(
            &SIGMAS[..count],
            &Constraints {
                split_axes: vec![0, 1, 2],
                ..constraints()
            },
        );
    }

    // **And with the whole-volume escape closed.** Both tables above end with
    // the planner declining to block at all at five sigmas — one block, the
    // whole volume, amplification 1 — which is the right answer when the volume
    // fits in memory and is not available when it does not. A budget of a
    // quarter of one image forces the cut, and what it costs is the number a
    // real run pays.
    println!("\ncut on every axis, budget 256 MiB");
    header();
    for count in 1..=SIGMAS.len() {
        row(
            &SIGMAS[..count],
            &Constraints {
                split_axes: vec![0, 1, 2],
                budget_bytes: Some(256 << 20),
                expected_concurrency: 4,
                ..constraints()
            },
        );
    }
}

/// **Is the declared cost of the widest arms honest?**
///
/// The declared cost of the whole stack is dominated three orders of magnitude
/// over by the morphological family at sigma 16 — a rank filter over a 67-voxel
/// box, which is 300,763 element voxels per output voxel. That is either the
/// true shape of this workload or a mispricing large enough to make every plan
/// over this chain wrong, and the difference is a measurement.
///
/// ```text
/// cargo test --release --test feature_stack -- --ignored --nocapture arms_measured
/// ```
#[test]
#[ignore = "a measurement, not an assertion"]
fn print_what_each_family_declares_against_what_it_takes() {
    let shape = [48usize, 32, 32];
    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let mut state = 7u64;
    let input: blockflow::voxels::Voxels =
        Array3::from_shape_fn((shape[0], shape[1], shape[2]), |_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1u64 << 31) as f64
        })
        .into();

    println!(
        "{:>34} {:>12} {:>12} {:>10}",
        "arm", "declared", "ns/voxel", "ns/unit"
    );
    for sigma in [1.0f64, 16.0] {
        for family in Family::ALL {
            // `Original` has no scale and a difference of Gaussians needs two,
            // so at one sigma neither has a channel to time.
            if matches!(family, Family::Original | Family::DifferenceOfGaussians) {
                continue;
            }
            let stack = FeatureStack::labkit(&[sigma])
                .unwrap()
                .with_families(&[family])
                .unwrap();
            let channel = stack.channels().unwrap().into_iter().next().unwrap();
            let declared = channel.chain.cost_per_voxel();
            let mut out = blockflow::voxels::Voxels::zeros(
                channel.chain.produces(Dtype::F64).unwrap(),
                shape,
            )
            .unwrap();
            channel
                .chain
                .apply(&input, &mut out, &Anchor::whole(shape))
                .unwrap();
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let started = std::time::Instant::now();
                channel
                    .chain
                    .apply(&input, &mut out, &Anchor::whole(shape))
                    .unwrap();
                best = best.min(started.elapsed().as_secs_f64() * 1e9 / voxels);
            }
            println!(
                "{:>34} {declared:>12.0} {best:>12.1} {:>10.4}",
                channel.name,
                best / declared
            );
        }
    }
}

/// Which arms actually cost anything, ranked. The diagnostic that found the
/// morphological family, and the one to run again after any change to a cost.
#[test]
#[ignore = "a measurement, not an assertion"]
fn print_the_most_expensive_arms() {
    let stack = FeatureStack::labkit(&SIGMAS).unwrap();
    let mut rows: Vec<(f64, String)> = stack
        .channels()
        .unwrap()
        .into_iter()
        .map(|channel| (channel.chain.cost_per_voxel(), channel.name))
        .collect();
    let total: f64 = rows.iter().map(|(cost, _)| cost).sum();
    rows.sort_by(|left, right| right.0.total_cmp(&left.0));
    println!("91 arms, {total:.0} total declared");
    for (cost, name) in rows.iter().take(12) {
        println!("{name:>28} {cost:>12.0} {:>6.1}%", 100.0 * cost / total);
    }
}
