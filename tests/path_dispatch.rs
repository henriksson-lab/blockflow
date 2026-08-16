// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The planner choosing which of two ways to compute one result runs.**
//
// `Chain::Alternative` has always meant "mutually exclusive branches, of which
// `taken` is live", and every fold a plan is built from treats it as the max
// over branches — so a plan built for one branch is a plan for all of them.
// What was missing is anything that *chose*: `taken` was whatever the chain's
// author wrote, so a chain carrying a fast path and a general one ran whichever
// was named. `strategy::choose_branches` is that choice, and this file is what
// pins it.
//
// Four things are asserted, in the order they matter:
//
// 1. **the answer does not move.** The two branches are byte-identical over the
//    whole volume, at every element and rank, which is the chain author's claim
//    and the precondition for anything choosing between them. Asserted first
//    because every other assertion here is worthless without it.
// 2. **the choice is by declared cost.** The same two branches, at a block
//    where the fast path wins and at a block where it does not, and the branch
//    the planner takes flips. No threshold is named anywhere; the numbers come
//    from `BlockOp::cost_per_voxel` and `BlockOp::cost_per_voxel_in`.
// 3. **the plan does not move with the choice.** The decomposition built from
//    the resolved chain is the decomposition built from the unresolved one,
//    fingerprint for fingerprint. Choosing a branch is not allowed to change
//    the halo, the grid or the valid regions, and the reason it cannot is that
//    every fold takes the max over branches.
// 4. **the run agrees with the oracle.** Executed under the resolved chain, the
//    volume is the one a single whole-volume block of the general path
//    produces.

use ndarray::{Array3, ArrayView3};

use blockflow::decomposition::{predicted_cost, Constraints, CostModel};
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::error::Result;
use blockflow::op::Chain;
use blockflow::ops::{
    rank_filter_into, Domain, ElementShape, Rank, RankFilterOp, ScanPlan, SlidingHistogramOp,
    StructuringElement,
};
use blockflow::strategy::{choose_branches, choose_paths, execute, Enumerating, Hints, Strategy};
use blockflow::strategy::{Trivial, Workflow};
use blockflow::voxels::Voxels;

const DOMAIN: usize = 4096;
const VOLUME: [usize; 3] = [256, 8, 4];

/// The two element sizes the measurement in `forme.md` was taken at, on the two
/// sides of what the declarations say: a `150 x 1 x 1` line, where the fast
/// path's steady state is a fifth of the gather it replaces, and a `9 x 1 x 1`
/// one, where the population is too small to pay for the walk over 4096 bins and
/// the general path is cheaper. Neither is a threshold — both are
/// `BlockOp::cost_per_voxel` read back.
const LONG: [usize; 3] = [150, 1, 1];
const SHORT: [usize; 3] = [9, 1, 1];

fn ramp(shape: [usize; 3]) -> Array3<u16> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        (((i * 7919 + j * 104729 + k * 1299709) % DOMAIN) as u16).min(DOMAIN as u16 - 1)
    })
}

/// The general path and the fast one, over the same element at the same rank.
///
/// Both are handed the same name, because the name is what a `Decomposition`
/// records and a plan that changed its names with the branch would make
/// assertion 3 vacuous.
fn branches(element: &StructuringElement, rank: Rank) -> Vec<Chain> {
    vec![
        Chain::op(RankFilterOp::new("rank", element.clone(), rank)),
        Chain::op(SlidingHistogramOp::rank(
            "rank",
            element.clone(),
            rank,
            Domain::of_size(DOMAIN).unwrap(),
        )),
    ]
}

fn dense(input: ArrayView3<'_, u16>, element: &StructuringElement, rank: Rank) -> Array3<u16> {
    let mut out = Array3::<u16>::zeros(input.raw_dim());
    rank_filter_into(input, element, rank, out.view_mut()).unwrap();
    out
}

/// 1. **The precondition.** Nothing here is worth anything if the two branches
///    are not the same function.
///
/// This duplicates no coverage from `sliding_histogram.rs`, which asserts the
/// same identity over a much wider sweep; it is here so that this file's own
/// claim — *choosing between these two changes nothing* — rests on an assertion
/// in this file rather than on a cross-reference.
#[test]
fn the_two_branches_are_the_same_function() -> Result<()> {
    let input = ramp(VOLUME);
    for element in [
        StructuringElement::from_size(ElementShape::Box, [9, 1, 1])?,
        StructuringElement::from_size(ElementShape::Box, [5, 5, 3])?,
    ] {
        for rank in [
            Rank::lowest(),
            Rank::highest(&element),
            Rank::ceiling_percentile(0.25).unwrap(),
        ] {
            let want = dense(input.view(), &element, rank);
            let general = Chain::op(RankFilterOp::new("rank", element.clone(), rank));
            let fast = Chain::op(SlidingHistogramOp::rank(
                "rank",
                element.clone(),
                rank,
                Domain::of_size(DOMAIN).unwrap(),
            ));
            for (what, chain) in [("general", general), ("fast", fast)] {
                let held: Voxels = input.clone().into();
                let mut out = Voxels::zeros(Dtype::U16, VOLUME)?;
                chain.apply(&held, &mut out, &blockflow::op::Anchor::whole(VOLUME))?;
                assert_eq!(out, Voxels::from(want.clone()), "{what}, rank {rank:?}");
            }
        }
    }
    Ok(())
}

/// 2. **The choice, and that it is a choice.**
///
/// Two directions, both out of the declarations and neither out of a size named
/// here:
///
/// * **the population against the domain.** A `150`-voxel line costs the general
///   path `3.87 x 150`; the fast one costs one histogram update per step plus one
///   walk of 4096 bins, and wins. A `9`-voxel line costs the general path almost
///   nothing and the fast one the same 4096-bin walk, and loses. That is the
///   limit `ops::sliding` documents as *small populations are dominated by the
///   bin walk, not by the gather*, and the planner sees it because the op
///   declared it.
/// * **the population against the block.** The fast path primes once per scan
///   line at `|element|`, which `cost_per_voxel` cannot state because it is
///   handed no block. `cost_per_voxel_in` can, and at a line one voxel long the
///   priming happens at every voxel: the traversal *is* the gather it replaced,
///   and the choice goes back to the general path.
#[test]
fn the_branch_taken_flips_with_the_population_and_with_the_block() -> Result<()> {
    let taken = |size: [usize; 3], block: [usize; 3]| -> usize {
        let element = StructuringElement::from_size(ElementShape::Box, size).unwrap();
        match choose_branches(
            Chain::alternative(
                branches(&element, Rank::ceiling_percentile(0.25).unwrap()),
                0,
            )
            .unwrap(),
            block,
        ) {
            Chain::Alternative { taken, .. } => taken,
            other => panic!("{} is not an alternative", other.display_name()),
        }
    };

    assert_eq!(
        taken(LONG, VOLUME),
        1,
        "150 voxels, whole volume: the fast path"
    );
    assert_eq!(
        taken(SHORT, VOLUME),
        0,
        "9 voxels: the bin walk is not paid for"
    );
    assert_eq!(
        taken(LONG, [1, 8, 4]),
        0,
        "a one-voxel scan line: every window is primed, so the fast path is the gather"
    );
    assert_eq!(
        taken(LONG, [2, 8, 4]),
        1,
        "and two voxels is already enough to amortise it — the term is |E| / line"
    );
    Ok(())
}

/// The same choice through the entry point a caller uses, which derives the
/// block from the constraints rather than being handed one.
#[test]
fn the_constraints_alone_decide_which_path_the_planner_takes() -> Result<()> {
    let long_block = Constraints {
        block_candidates: vec![64],
        split_axes: vec![0],
        ..Constraints::default()
    };
    let one_voxel_block = Constraints {
        block_candidates: vec![1],
        split_axes: vec![0],
        ..Constraints::default()
    };
    for (what, size, constraints, expected) in [
        ("a long line, a long block", LONG, &long_block, 1usize),
        ("a short line", SHORT, &long_block, 0),
        ("a long line, a one-voxel block", LONG, &one_voxel_block, 0),
    ] {
        let element = StructuringElement::from_size(ElementShape::Box, size)?;
        let chain = Chain::alternative(branches(&element, Rank::lowest()), 0)?;
        let resolved = choose_paths(chain, VOLUME, Dtype::U16, constraints)?;
        match resolved {
            Chain::Alternative { taken, .. } => assert_eq!(taken, expected, "{what}"),
            other => panic!("{} is not an alternative", other.display_name()),
        }
    }
    Ok(())
}

/// 3. **The plan does not move with the choice.**
///
/// The decomposition is built from the max over branches — reach, and therefore
/// halo, grid and valid regions — so resolving the alternative cannot change it.
/// Asserted on the fingerprint, which is the whole binding half in one integer.
#[test]
fn choosing_a_branch_does_not_move_the_plan() -> Result<()> {
    let element = StructuringElement::from_size(ElementShape::Box, [5, 5, 3])?;
    let constraints = Constraints {
        block_candidates: vec![8, 16, 32],
        split_axes: vec![0],
        ..Constraints::default()
    };
    let plan_for = |chain: Chain| -> Result<u64> {
        let workflow = Workflow::new(chain, VOLUME, Dtype::U16);
        Ok(Enumerating::default()
            .decompose(&workflow, &constraints)?
            .fingerprint())
    };
    let unresolved = plan_for(Chain::alternative(branches(&element, Rank::lowest()), 0)?)?;
    let resolved = plan_for(choose_paths(
        Chain::alternative(branches(&element, Rank::lowest()), 0)?,
        VOLUME,
        Dtype::U16,
        &constraints,
    )?)?;
    assert_eq!(
        resolved, unresolved,
        "resolving the alternative moved the binding half of the plan"
    );
    Ok(())
}

/// 4. **And the run agrees with the oracle**, which is what all of it is for.
#[test]
fn the_resolved_chain_runs_and_gives_the_whole_volume_answer() -> Result<()> {
    let element = StructuringElement::from_size(ElementShape::Box, LONG)?;
    let rank = Rank::ceiling_percentile(0.25).unwrap();
    let input = ramp(VOLUME);
    let want: Voxels = dense(input.view(), &element, rank).into();

    let constraints = Constraints {
        block_candidates: vec![64],
        split_axes: vec![0],
        ..Constraints::default()
    };
    let chain = choose_paths(
        Chain::alternative(branches(&element, rank), 0)?,
        VOLUME,
        Dtype::U16,
        &constraints,
    )?;
    match &chain {
        Chain::Alternative { taken, .. } => {
            assert_eq!(*taken, 1, "150 voxels: the fast path")
        }
        other => panic!("{} is not an alternative", other.display_name()),
    }

    let workflow = Workflow::new(chain, VOLUME, Dtype::U16);
    let plan = Enumerating::default().decompose(&workflow, &constraints)?;
    assert!(
        plan.phases[0].grid.n_blocks() > 1,
        "a single block would prove nothing about the seams"
    );
    let env = ArrayEnvironment::for_decomposition(input.clone().into(), &plan, [8, 8, 8])?;
    execute("dispatch", &workflow, &plan, &Hints::default(), &env)?;
    assert_eq!(env.level(plan.n_phases()), want, "the fast path, blocked");

    // The oracle: the general path, one block, no seams.
    let general = Workflow::new(
        Chain::op(RankFilterOp::new("rank", element.clone(), rank)),
        VOLUME,
        Dtype::U16,
    );
    let oracle = Trivial.decompose(&general, &Constraints::default())?;
    let env = ArrayEnvironment::for_decomposition(input.into(), &oracle, [8, 8, 8])?;
    execute("oracle", &general, &oracle, &Hints::default(), &env)?;
    assert_eq!(env.level(oracle.n_phases()), want, "the oracle");
    Ok(())
}

/// The block-dependent term is **declared**, not inferred: the fast path says
/// its cost is worse on a short line, and `predicted_cost` reads it back.
///
/// Without this the planner would price the steady state at every block size and
/// the flip above could not happen. It is asserted through `predicted_cost`
/// rather than by calling the op, because what matters is that the figure
/// reaches the plan.
#[test]
fn the_priming_term_reaches_the_plans_predicted_cost() -> Result<()> {
    let element = StructuringElement::from_size(ElementShape::Box, LONG)?;
    let fast = || {
        Chain::op(SlidingHistogramOp::rank(
            "rank",
            element.clone(),
            Rank::lowest(),
            Domain::of_size(DOMAIN).unwrap(),
        ))
    };
    let cost = |edge: usize| -> Result<f64> {
        let constraints = Constraints {
            block_candidates: vec![edge],
            split_axes: vec![0],
            ..Constraints::default()
        };
        let workflow = Workflow::new(fast(), VOLUME, Dtype::U16);
        let plan = Enumerating::default().decompose(&workflow, &constraints)?;
        predicted_cost(&workflow.chain, &plan, &CostModel::default())
    };
    let per_voxel = |edge: usize| -> Result<f64> {
        Ok(cost(edge)? / (VOLUME.iter().product::<usize>() as f64))
    };
    assert!(
        per_voxel(1)? > per_voxel(64)?,
        "the priming gather is not in the plan's cost: {} vs {}",
        per_voxel(1)?,
        per_voxel(64)?
    );

    // And the scan axis is the one the declaration is measured along: the same
    // element scanned along an axis the block does not cut prices the same at
    // every candidate, because the line length does not move.
    let flat = Constraints {
        block_candidates: vec![1, 64],
        split_axes: vec![2],
        ..Constraints::default()
    };
    let workflow = Workflow::new(fast(), VOLUME, Dtype::U16);
    let plan = Enumerating::default().decompose(&workflow, &flat)?;
    assert_eq!(
        ScanPlan::new(&element).axis(),
        0,
        "the element is a line along axis 0, so that is where the priming lives"
    );
    assert_eq!(plan.phases[0].grid.block()[0], VOLUME[0]);
    Ok(())
}
