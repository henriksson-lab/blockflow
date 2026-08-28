// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **What the refined ladder costs the planner, timed.**
//
// `decomposition::refined_ladder` was made opt-in rather than the default for
// two stated reasons. This file measures the first of them; the second is a
// blast-radius question that no timing answers.
//
// The reason being tested
// -----------------------
// The candidate search was described as `partitions x candidates^phases`, and
// refining a three-rung ladder to five squares-ish the per-phase factor: at four
// phases, `81` combinations become `625`. That was a fair thing to weigh, **and
// the ground under it moved**: `boxes_tile_exactly` was `O(n^2)` and is now a
// linear separating pass, which took `PlanBuilder::finish` from 445 ms to 4.2 ms
// at 8192 blocks. A candidate multiplier is only expensive relative to what one
// candidate costs, and one candidate got much cheaper.
//
// So the number to have is not the combination count — that is arithmetic and
// was never in doubt — but **the wall-clock of `Strategy::decompose` under both
// ladders**, on a chain long enough to reach the phase counts where the
// exponent was supposed to bite.
//
// The chain, described by its shape
// ---------------------------------
// A chain of slots that are pointwise, read a narrow neighbourhood, or read a
// wide one, with **whole-volume steps** between groups of them. A whole-volume
// step is a planning barrier — `decomposition::is_planning_barrier` — so it must
// be alone in its phase, and `n` of them force at least `n` phases whatever the
// cost model prefers. That is how a four-phase plan is arranged here rather than
// hoped for, and it is the shape any pipeline has that reduces over everything
// more than once.
//
// A second chain with no whole-volume step in it is used for the plan-change
// sweep, because a barrier phase's block is the whole volume and a ladder has
// nothing to choose there. Which chain answers which question is stated at each
// test.
//
// How it is measured, and what that is worth
// ------------------------------------------
// * **The minimum of several repetitions**, not the mean. This box is shared, so
//   a mean measures the neighbours; a minimum measures the least contaminated
//   run, which is the closest thing to the quantity wanted.
// * **The same chain, planned twice**, so the only thing that differs is the
//   candidate list.
// * **The phase count is asserted equal**, because a timing comparison between a
//   four-phase plan and a two-phase plan would be a comparison of two different
//   questions.
// * **Printed as well as asserted.** The assertion is a ceiling loose enough to
//   survive a loaded machine; the printed table is the measurement. A tight
//   assertion on wall clock is a test that fails when somebody else compiles.

use std::time::{Duration, Instant};

use blockflow::decomposition::{refined_ladder, Constraints, CostModel, Decomposition};
use blockflow::dtype::Dtype;
use blockflow::error::Result;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::strategy::{Enumerating, PartitionSearch, Strategy, Workflow};
use blockflow::voxels::Voxels;

/// A slot with a stated reach and cost. The planner reads nothing else.
///
/// `full` makes it a **planning barrier**: a full-reach slot must be alone in
/// its phase, so a chain with `n` of them plans into at least `n` phases
/// whatever the cost model prefers. That is how a four-phase plan is arranged
/// here rather than hoped for — the pipeline's binarize chain gets its phases
/// the same way, from arms that reduce over everything.
struct Slot {
    name: &'static str,
    reach: usize,
    cost: f64,
    full: bool,
}

impl BlockOp for Slot {
    fn name(&self) -> &'static str {
        self.name
    }
    fn reach(&self, _axis: usize, volume_len: usize) -> usize {
        if self.full {
            volume_len
        } else {
            self.reach
        }
    }
    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }
    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }
    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Names that describe a slot's **shape** and nothing else: whether it is
/// pointwise, reads a neighbourhood, or reduces over everything. What a slot
/// would be *for* is the caller's business and this crate does not know it.
const NAMES: [&str; 6] = [
    "pointwise",
    "narrow-window",
    "wide-window",
    "pointwise-again",
    "narrow-again",
    "wide-again",
];

/// A chain that plans into exactly `phases` phases.
///
/// `phases - 1` full-reach slots force the cuts, with two ordinary slots either
/// side of each so the partition search has runs to price rather than a chain of
/// nothing but barriers. Reaches and costs vary along it, so a cut costs
/// something and the search is not trivial.
fn chain(phases: usize) -> Chain {
    let mut slots = Vec::new();
    for group in 0..phases {
        if group > 0 {
            slots.push(Chain::op(Slot {
                name: "whole-volume",
                reach: 0,
                cost: 1.0,
                full: true,
            }));
        }
        for step in 0..2 {
            let index = group * 2 + step;
            slots.push(Chain::op(Slot {
                name: NAMES[index % NAMES.len()],
                reach: match index % 3 {
                    0 => 0,
                    1 => 1,
                    _ => 4,
                },
                cost: 1.0 + (index % 4) as f64,
                full: false,
            }));
        }
    }
    Chain::sequence(slots)
}

/// The same slots with no barrier in them, so every phase's block is the
/// ladder's to choose.
fn flat_chain(slots: usize) -> Chain {
    Chain::sequence(
        (0..slots)
            .map(|index| {
                Chain::op(Slot {
                    name: NAMES[index % NAMES.len()],
                    reach: match index % 3 {
                        0 => 0,
                        1 => 1,
                        _ => 4,
                    },
                    cost: 1.0 + (index % 4) as f64,
                    full: false,
                })
            })
            .collect(),
    )
}

const VOLUME: [usize; 3] = [512, 512, 512];

/// Budgets that bracket the rungs, so the ladder has something to decide.
///
/// A `128`-cube of `f64` is 16.8 MiB and a `96`-cube is 7.1 MiB, so the
/// interesting budgets are the ones between those times the concurrency — where
/// the coarse ladder must fall to `64` and the refined one can stop at `96`.
/// Above the top of this range both ladders take the largest rung and the plans
/// are identical, which is a true and uninteresting answer.
const BUDGETS: [u64; 8] = [
    48 << 20,
    64 << 20,
    96 << 20,
    128 << 20,
    192 << 20,
    256 << 20,
    512 << 20,
    1024 << 20,
];

fn constraints(candidates: Vec<usize>, budget: Option<u64>) -> Constraints {
    Constraints {
        budget_bytes: budget,
        expected_concurrency: 8,
        model: CostModel::default(),
        block_candidates: candidates,
        split_axes: vec![0, 1, 2],
        ..Default::default()
    }
}

/// The least contaminated of `reps` plans, and the plan itself.
fn time_plan(
    workflow: &Workflow,
    constraints: &Constraints,
    reps: u32,
) -> (Duration, Decomposition) {
    let strategy = Enumerating {
        concurrency: 8,
        search: PartitionSearch::Exhaustive,
        ..Enumerating::default()
    };
    let mut best = Duration::MAX;
    let mut plan = None;
    for _ in 0..reps {
        let started = Instant::now();
        let decomposition = strategy
            .decompose(workflow, constraints)
            .expect("the chain plans");
        let took = started.elapsed();
        if took < best {
            best = took;
        }
        plan = Some(decomposition);
    }
    (best, plan.expect("at least one repetition"))
}

/// **The measurement.** Both ladders, over chains that plan into two, three,
/// four and five phases, with the four-phase row being the shape the argument
/// is about.
#[test]
fn the_refined_ladder_is_priced_against_the_coarse_one_end_to_end() {
    let coarse = vec![32usize, 64, 128];
    let fine = refined_ladder(&coarse);
    assert_eq!(fine.len(), 5, "the ladder under test");
    // **Unbounded**, and that is deliberate: the question is what the *search*
    // costs, and a budget that rejects most candidates early makes the search
    // cheaper than the one a caller would run. It also keeps the barrier phases
    // plannable — a full-reach slot needs the whole volume in one block, which
    // is 1 GiB of `f64` here and over every budget worth sweeping.
    let budget = None;

    eprintln!(
        "\n{:>7} | {:>11} {:>11} {:>7} | {:>9} {:>9}",
        "phases", "coarse", "refined", "ratio", "combos c", "combos f"
    );
    let mut worst_ratio = 0.0f64;
    let mut four_phase_seen = false;

    for want in [2usize, 3, 4, 5] {
        let workflow = Workflow::new(chain(want), VOLUME, Dtype::F64);
        let (coarse_time, coarse_plan) =
            time_plan(&workflow, &constraints(coarse.clone(), budget), 9);
        let (fine_time, fine_plan) = time_plan(&workflow, &constraints(fine.clone(), budget), 9);

        // A timing comparison between two different questions would be
        // meaningless, so the two plans must agree on how many phases they are.
        assert_eq!(
            coarse_plan.n_phases(),
            fine_plan.n_phases(),
            "{want} groups: the two ladders planned different phase counts, so the times \
             below are not comparable"
        );
        let phases = coarse_plan.n_phases();
        if phases >= 4 {
            four_phase_seen = true;
        }
        let ratio = fine_time.as_secs_f64() / coarse_time.as_secs_f64().max(1e-9);
        worst_ratio = worst_ratio.max(ratio);
        eprintln!(
            "{phases:>7} | {:>10.3?} {:>10.3?} {ratio:>6.2}x | {:>9} {:>9}",
            coarse_time,
            fine_time,
            coarse.len().pow(phases as u32),
            fine.len().pow(phases as u32),
        );
    }

    assert!(
        four_phase_seen,
        "no chain here planned into four phases, so this file does not price the case the \
         argument is about"
    );
    // A ceiling, not a measurement: loose enough to survive a loaded box, tight
    // enough that a planner that became quadratic in candidates would fail it.
    // The combination count grows by at most `(5/3)^phases`, which is 7.7x at
    // four phases, so anything far past that is not candidate arithmetic.
    assert!(
        worst_ratio < 20.0,
        "the refined ladder cost {worst_ratio:.1}x the planning time of the coarse one. The \
         candidate count grows by at most (5/3)^phases, so a ratio far past that is the \
         planner scaling in something other than candidates."
    );
    eprintln!("worst refined/coarse planning ratio: {worst_ratio:.2}x");
}

/// **What the extra planning time buys.** A timing that did not also say whether
/// the plan changed would be pricing a search whose answer might be identical —
/// and at a loose budget it *is* identical, because both ladders take the
/// largest rung and the largest rung is the same. The budget is what makes the
/// ladder matter, so it is the axis this sweeps.
#[test]
fn the_refined_ladder_changes_the_plan_it_costs_more_to_find() {
    let coarse = vec![32usize, 64, 128];
    let fine = refined_ladder(&coarse);
    let mut differed = 0usize;
    let mut cells = 0usize;
    let mut best_gain = 1.0f64;

    eprintln!(
        "\n{:>8} {:>7} | {:>18} {:>18}",
        "budget", "phases", "coarse blocks", "refined blocks"
    );
    // **No planning barriers here**, unlike the timing chain: a full-reach slot
    // needs the whole volume in one block, so its phase's budget is fixed and
    // the ladder has nothing to choose. The budget is the axis that makes a
    // ladder matter, so the chain has to be one the budget can bite.
    for slots in [2usize, 6] {
        let workflow = Workflow::new(flat_chain(slots), VOLUME, Dtype::F64);
        for budget in BUDGETS.map(Some) {
            let coarse_plan = match Enumerating::default()
                .decompose(&workflow, &constraints(coarse.clone(), budget))
            {
                Ok(plan) => plan,
                // A budget too tight for the coarse ladder to plan at all is
                // itself a result, and the refined one may still manage.
                Err(_) => {
                    let fine_plan = Enumerating::default()
                        .decompose(&workflow, &constraints(fine.clone(), budget));
                    eprintln!(
                        "{:>6} MiB {slots:>7} | {:>18} {:>18}",
                        budget.expect("a budget") >> 20,
                        "no plan",
                        if fine_plan.is_ok() {
                            "a plan"
                        } else {
                            "no plan"
                        }
                    );
                    if fine_plan.is_ok() {
                        differed += 1;
                    }
                    cells += 1;
                    continue;
                }
            };
            let fine_plan = Enumerating::default()
                .decompose(&workflow, &constraints(fine.clone(), budget))
                .expect("the refined ladder plans wherever the coarse one does");
            let edges = |plan: &Decomposition| -> Vec<usize> {
                plan.phases.iter().map(|p| p.grid.block()[2]).collect()
            };
            let (a, b) = (edges(&coarse_plan), edges(&fine_plan));
            eprintln!(
                "{:>6} MiB {:>7} | {:>18?} {:>18?}",
                budget.expect("a budget") >> 20,
                coarse_plan.n_phases(),
                a,
                b
            );
            cells += 1;
            if a != b {
                differed += 1;
            }
            // Whatever it chose, the refined plan's blocks are never smaller:
            // the ladder only adds rungs, and admission takes the largest that
            // fits.
            for (coarse_edge, fine_edge) in a.iter().zip(&b) {
                assert!(
                    fine_edge >= coarse_edge,
                    "at {} MiB the refined ladder chose a smaller block ({fine_edge}) than the \
                     coarse one ({coarse_edge})",
                    budget.expect("a budget") >> 20
                );
                best_gain = best_gain.max((*fine_edge as f64 / *coarse_edge as f64).powi(3));
            }
        }
    }
    assert!(
        differed > 0,
        "the refined ladder produced an identical plan in all {cells} cells, so the extra \
         planning time bought nothing here and this file is not measuring the trade"
    );
    eprintln!(
        "plans differed in {differed} of {cells} (chain, budget) cells; best gain \
         {best_gain:.2}x in block volume"
    );
}

/// **Can a refined ladder plan where a coarse one cannot?** No, and the reason
/// is structural rather than measured — but it is measured too, because the
/// question arose from a fixture that looked like the finding.
///
/// # Where the question came from
///
/// An earlier version of this file swept a *bounded* budget over the chain with
/// whole-volume steps in it, and every row of both columns read "no plan". That
/// looks exactly like the strongest possible argument for the refinement — a
/// coarse ladder straddling a budget with nothing admissible between its rungs —
/// and it is not that at all. **A whole-volume step's phase cannot be cut on any
/// axis**, so its block is the volume whatever the candidates are: 1 GiB of
/// `f64` at `512^3`, which no budget worth sweeping admits at concurrency 8.
/// Both ladders failed for a reason neither of them could have fixed. It was the
/// fixture.
///
/// # And the finding it was mistaken for cannot happen
///
/// Feasibility is decided by the **floor** of the ladder and by nothing else: a
/// plan exists if some partition has every phase fitting its cheapest candidate,
/// and the cheapest candidate is the smallest rung. `refined_ladder` adds a rung
/// at three quarters of each entry **except the smallest**, so it adds nothing
/// below the floor and cannot make an infeasible plan feasible.
///
/// So the refinement changes *which* block is chosen and never *whether* a plan
/// exists. That is worth asserting rather than reasoning about once, because it
/// is the difference between the refinement being a quality improvement — which
/// it is — and a feasibility one, which it is not and should not be sold as.
#[test]
fn the_refined_ladder_never_plans_where_the_coarse_one_cannot() {
    let coarse = vec![32usize, 64, 128];
    let fine = refined_ladder(&coarse);
    assert_eq!(
        fine.iter().min(),
        coarse.iter().min(),
        "the refinement moved the floor, and feasibility is decided by the floor"
    );

    let mut infeasible = 0usize;
    let mut feasible = 0usize;
    // Budgets far below anything admissible through to comfortably above, so the
    // sweep spans the boundary rather than sitting on one side of it.
    for shift in 20..34u32 {
        let budget = Some(1u64 << shift);
        for slots in [2usize, 6] {
            let workflow = Workflow::new(flat_chain(slots), VOLUME, Dtype::F64);
            let coarse_plan =
                Enumerating::default().decompose(&workflow, &constraints(coarse.clone(), budget));
            let fine_plan =
                Enumerating::default().decompose(&workflow, &constraints(fine.clone(), budget));
            match (coarse_plan.is_ok(), fine_plan.is_ok()) {
                (false, true) => panic!(
                    "at {} MiB with {slots} slots the refined ladder planned where the coarse \
                     one could not. Feasibility is supposed to be decided by the floor, which \
                     the refinement does not move — so either the floor moved or feasibility \
                     depends on something else.",
                    (1u64 << shift) >> 20
                ),
                (false, false) => infeasible += 1,
                (true, _) => feasible += 1,
            }
        }
    }
    // Both halves have to be non-empty or the sweep sat on one side of the
    // boundary and asserted nothing.
    assert!(
        infeasible > 0 && feasible > 0,
        "the sweep found {infeasible} infeasible and {feasible} feasible cells; it must span \
         the boundary to be saying anything"
    );
    eprintln!(
        "\nfeasibility: {infeasible} cell(s) where neither ladder plans, {feasible} where both \
         do, 0 where only the refined one does"
    );

    // And the fixture that raised the question, stated as what it is: a
    // whole-volume step's phase is uncuttable, so no ladder helps it.
    let barriered = Workflow::new(chain(4), VOLUME, Dtype::F64);
    let tight = Some(4u64 << 30);
    assert!(
        Enumerating::default()
            .decompose(&barriered, &constraints(coarse.clone(), tight))
            .is_err(),
        "the barriered chain is supposed to be the infeasible fixture"
    );
    assert!(
        Enumerating::default()
            .decompose(&barriered, &constraints(fine.clone(), tight))
            .is_err(),
        "and the refined ladder must fail it too — if it did not, the earlier reading of \
         those rows as a ladder finding would have been right after all"
    );
}
