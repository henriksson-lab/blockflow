// SPDX-License-Identifier: MIT
//
// `PlanBuilder::iterate` hands `iterative_phase` the grid of the phase before
// it. That is not a cheap choice, it is the absence of one: an iterative phase's
// reach is its substage element's, which has nothing to do with the reach the
// previous phase was cut for, and a lattice chosen for a local map is a bad
// lattice for a wide-element thinning. `PlanBuilder::iterate_priced` is the
// missing half — the same "choose for me" that `partition` is to `pixels` — and
// this file is what says the choice is real, is the plan's own number, and is
// worth making.
//
// **What it buys, measured.** Predicted makespan of the inherited lattice over
// the priced one, `256³` f64, candidates `16..256`, split on all three axes:
//
// | substage reach | inherited 16 | 32 | 64 | 128 | 256 |
// |---|---|---|---|---|---|
// | `0`  | 1.00 | 1.00 | 1.00 | 1.00 | 1.50 |
// | `1`  | 1.18 | 1.07 | 1.02 | 1.00 | 1.47 |
// | `4`  | 1.99 | 1.34 | 1.10 | 1.00 | 1.36 |
// | `16` | 9.48 | 3.05 | 1.48 | 1.00 | 1.02 |
// | `32` | 42.0 | 9.33 | 3.00 | 1.46 | 1.00 |
//
// at `workers = 40`; at `workers = 1` the same table runs to `83.7x`. The
// diagonal of ones is the case where the previous phase happened to be cut where
// this one wants to be, and the answer to "is the sweep worth switching on" is
// the rest of the table.
//
// **What decides the answer** is not what an author reaches for first, and the
// tests below pin all three of the following because two of them were carried
// into this file as claims and did not survive being swept:
//
// * `workers == 1` collapses the objective to serial work, which is monotone in
//   the edge, so the sweep answers "the largest candidate" always. Nothing else
//   moves it.
// * Above `workers == 1` the **declared compute** moves the argmin the whole
//   width of the candidate list. The claim carried in was that it could not,
//   because compute is charged over the same extent as the read; that holds
//   only where the pool bound binds for every candidate, which is `workers ==
//   1`.
// * **Which weight the write is charged at** moves the argmin too, so
//   `Materialisation` is asked of the caller. The claim carried in — this
//   file's own, and wrong — was that it could not, because the write enters the
//   channel bound as the volume exactly at every candidate. True of the channel
//   and false of the pool, where the write enters as
//   `mean_core * ceil(n / workers)` and `ceil(n / workers) / n` is a function of
//   the block count.
//
// Under `CostModel::default()` the two write weights are both `1.0`, so the last
// of those is invisible: a test suite that only ever used the default model
// would have passed with the wrong answer hard-coded. That is asserted here too,
// because "the bug was invisible" is a property of the test suite and belongs in
// it.

// **The substage count, hunted rather than argued.** `iterate_priced` prices one
// substage of a phase that runs an unknown number of them, and the argument for
// that being harmless had two parts. Both are tested below and only one of them
// survives:
//
// * The count *is* independent of the lattice. The executor is run to a fixed
//   point over thirteen grids — including one block per voxel, where every step
//   of the propagation crosses a seam, and grids that divide no axis evenly —
//   across four substage reaches and two data shapes, one of them a one-voxel
//   serpentine that forces a long geodesic (`35` substages against the open
//   field's `13`). The count is the whole-volume count every time, and so is the
//   volume written. That is the halo doing its job: it is one substage's reach
//   wide and holds the neighbours' cores from the substage before, so a seam is
//   crossed at the rate the inside of a block is.
// * A constant multiplier is *not* neutral here, because it does not multiply
//   the whole price. A phase runs `S` substages of read and compute and writes
//   its image once, at the fixed point, so ranking at `S == 1` weighs the write
//   `S` times too heavily and the residual `(S - 1) * (read + compute)` is a
//   function of the block edge. The choice departs at counts as ordinary as `2`,
//   and the one-substage choice costs up to `1.125x` the right one.
//
// So `Substages` is asked for where a caller has the number — `Stats::substages`
// reports it — and `Substages::Unknown` is byte-identical to not asking. The
// discriminating pair is the load-bearing test: a multiplier on the whole price
// moves nothing, one on its repeated half moves the choice. A correction that
// scaled everything by the count would pass every other test here and do
// nothing.
//
use ndarray::Array3;

use blockflow::assemble::{Materialisation, PlanBuilder, Substages};
use blockflow::decomposition::{Constraints, CostModel};
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::iterate::{IterativeOp, Substage, SubstageLimit, SubstageOperand};
use blockflow::strategy::{execute_phases, predicted_makespan, Hints};
use blockflow::voxels::Voxels;
use blockflow::Result;

const VOLUME: [usize; 3] = [256, 256, 256];
const CANDIDATES: [usize; 5] = [16, 32, 64, 128, 256];

/// A one-substage spread whose reach and per-voxel cost the test sets.
///
/// Both are parameters because both are levers under test and a probe that
/// fixed either would be a probe that could not see it move.
struct Spread {
    reach: usize,
    cost: f64,
}

impl IterativeOp for Spread {
    fn name(&self) -> &'static str {
        "spread"
    }

    fn operands(&self) -> Vec<SubstageOperand> {
        vec![
            SubstageOperand::running([self.reach, self.reach, self.reach]),
            SubstageOperand::fixed([0, 0, 0]),
        ]
    }

    fn limit(&self) -> SubstageLimit {
        SubstageLimit::of(64).expect("a positive limit")
    }

    fn substage(&self, at: &Substage<'_>, out: &mut Voxels) -> Result<()> {
        out.assign(at.operand(0)?)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

fn constraints(model: CostModel, axes: Vec<usize>) -> Constraints {
    Constraints {
        budget_bytes: None,
        expected_concurrency: 1,
        model,
        block_candidates: CANDIDATES.to_vec(),
        split_axes: axes,
        ..Default::default()
    }
}

/// The edge the sweep picks, and what it priced it at.
fn choose(
    reach: usize,
    cost: f64,
    workers: usize,
    model: CostModel,
    axes: Vec<usize>,
    materialisation: Materialisation,
    substages: Substages,
) -> (usize, f64) {
    // The lattice the builder is holding is deliberately *not* one the sweep
    // would pick, so a sweep that quietly returned `self.grid()` would show.
    let grid = BlockGrid::new(VOLUME, [64, 64, 64]).expect("a lattice");
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let priced = plan
        .iterate_priced(
            Spread { reach, cost },
            &constraints(model, axes),
            workers,
            materialisation,
            substages,
        )
        .expect("a priced iterative phase");
    (priced.grid().block()[2], priced.ranked_makespan())
}

// ------------------------------------------------------- the plan's number --

/// **The criterion.** What the sweep minimised is what the finished plan
/// reports, for a phase that is the plan's output.
///
/// A search minimising a quantity nobody can read back off the plan is a claim
/// nobody can check, which is `predicted_makespan`'s own reason for existing.
/// Equality is exact rather than approximate: the two are the same calls on the
/// same arguments, so anything but the same bits would mean they are not.
#[test]
fn the_price_the_sweep_chose_on_is_the_price_the_plan_reports() {
    let model = CostModel::default();
    let grid = BlockGrid::new(VOLUME, [64, 64, 64]).expect("a lattice");
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let priced = plan
        .iterate_priced(
            Spread {
                reach: 4,
                cost: 1.0,
            },
            &constraints(model, vec![0, 1, 2]),
            40,
            Materialisation::Output,
            Substages::Unknown,
        )
        .expect("a priced phase");
    let chosen = priced.makespan();
    let assembly = plan.finish().expect("a plan");
    let work = assembly.work();
    let whole = predicted_makespan(
        &assembly.workflow.chain,
        &assembly.decomposition,
        &work,
        &model,
        40,
    )
    .expect("a priced plan");
    assert_eq!(chosen, whole);
}

/// The same, for a phase another phase reads — which is the case the caller has
/// to declare and the case a builder cannot infer.
///
/// Two iterative phases, the first declared `Intermediate` because it is, the
/// second `Output` because it is. The plan's own price must be their sum: it is
/// additive over phases by construction, so any disagreement is a disagreement
/// about what one of them was charged.
///
/// The model is calibrated — the two write weights differ — because under
/// `CostModel::default()` they do not and the assertion would hold whatever
/// `Materialisation` said.
#[test]
fn a_phase_declared_intermediate_is_priced_as_the_plan_prices_it() {
    let model = CostModel {
        write_cost_per_voxel: 1.0,
        materialise_cost_per_voxel: 7.0,
        ..CostModel::default()
    };
    let grid = BlockGrid::new(VOLUME, [64, 64, 64]).expect("a lattice");
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let first = plan
        .iterate_priced(
            Spread {
                reach: 4,
                cost: 1.0,
            },
            &constraints(model, vec![0, 1, 2]),
            40,
            Materialisation::Intermediate,
            Substages::Unknown,
        )
        .expect("a priced phase");
    plan.regrid(first.grid().clone());
    let second = plan
        .iterate_priced(
            Spread {
                reach: 1,
                cost: 1.0,
            },
            &constraints(model, vec![0, 1, 2]),
            40,
            Materialisation::Output,
            Substages::Unknown,
        )
        .expect("a priced phase");
    let sum = first.makespan() + second.makespan();
    let assembly = plan.finish().expect("a plan");
    let work = assembly.work();
    let whole = predicted_makespan(
        &assembly.workflow.chain,
        &assembly.decomposition,
        &work,
        &model,
        40,
    )
    .expect("a priced plan");
    assert_eq!(sum, whole);
}

/// The liveness test for the one above: declaring the wrong thing is visible.
///
/// The negative control is the same program with one thing changed — the first
/// phase says `Output` when the plan will make it an intermediate — and the
/// sweep's number then stops being the plan's number. If this passed, the
/// argument that `Materialisation` has to be asked for would have no evidence
/// behind it and the parameter could be deleted.
#[test]
fn declaring_the_wrong_materialisation_makes_the_sweep_disagree_with_the_plan() {
    let model = CostModel {
        write_cost_per_voxel: 1.0,
        materialise_cost_per_voxel: 7.0,
        ..CostModel::default()
    };
    let grid = BlockGrid::new(VOLUME, [64, 64, 64]).expect("a lattice");
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let first = plan
        .iterate_priced(
            Spread {
                reach: 4,
                cost: 1.0,
            },
            &constraints(model, vec![0, 1, 2]),
            40,
            // The lie.
            Materialisation::Output,
            Substages::Unknown,
        )
        .expect("a priced phase");
    plan.regrid(first.grid().clone());
    let second = plan
        .iterate_priced(
            Spread {
                reach: 1,
                cost: 1.0,
            },
            &constraints(model, vec![0, 1, 2]),
            40,
            Materialisation::Output,
            Substages::Unknown,
        )
        .expect("a priced phase");
    let assembly = plan.finish().expect("a plan");
    let work = assembly.work();
    let whole = predicted_makespan(
        &assembly.workflow.chain,
        &assembly.decomposition,
        &work,
        &model,
        40,
    )
    .expect("a priced plan");
    assert_ne!(
        first.makespan() + second.makespan(),
        whole,
        "a phase priced as the plan's output and then buried mid-plan was priced identically \
         either way, so `Materialisation` is not reaching the price and the argument for asking \
         the caller for it has no evidence"
    );
}

// ------------------------------------------------------------ the argmin --

/// The sweep really is a minimum over the candidates it was offered, not a
/// preference dressed up as one.
///
/// Checked against every candidate priced through the plan itself, so the
/// comparison is against the number the plan would report rather than against a
/// second copy of the sweep's arithmetic.
#[test]
fn the_chosen_edge_is_the_cheapest_candidate_the_plan_would_price() {
    let model = CostModel::default();
    for workers in [1usize, 8, 40] {
        for reach in [0usize, 1, 4, 16, 32] {
            let (chosen_edge, chosen) = choose(
                reach,
                1.0,
                workers,
                model,
                vec![0, 1, 2],
                Materialisation::Output,
                Substages::Unknown,
            );
            for &edge in &CANDIDATES {
                // Price this candidate as a whole one-phase plan, which is the
                // only way to get the plan's own opinion of a lattice.
                let grid = match BlockGrid::along(VOLUME, &[0, 1, 2], edge) {
                    Ok(grid) => grid,
                    Err(_) => continue,
                };
                let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
                plan.iterate(Spread { reach, cost: 1.0 }).expect("a phase");
                let assembly = plan.finish().expect("a plan");
                let work = assembly.work();
                let priced = predicted_makespan(
                    &assembly.workflow.chain,
                    &assembly.decomposition,
                    &work,
                    &model,
                    workers,
                )
                .expect("a priced plan");
                assert!(
                    chosen <= priced,
                    "workers {workers}, reach {reach}: the sweep chose edge {chosen_edge} at \
                     {chosen} and candidate {edge} prices at {priced}"
                );
            }
        }
    }
}

/// At one worker the objective is the phase's serial work, which falls
/// monotonically as the block grows, so the sweep answers the largest candidate
/// and nothing moves it.
///
/// Asserted rather than left implicit because it is the negative control for
/// every other test in this file: a caller who leaves `workers` at one has a
/// sweep that cannot choose, and should know that rather than discover it.
#[test]
fn at_one_worker_the_sweep_always_answers_the_largest_candidate() {
    let largest = *CANDIDATES.iter().max().expect("candidates");
    for reach in [0usize, 1, 4, 16, 32] {
        for cost in [1e-3f64, 1.0, 1e3] {
            for materialisation in [Materialisation::Intermediate, Materialisation::Output] {
                let model = CostModel {
                    write_cost_per_voxel: 1.0,
                    materialise_cost_per_voxel: 7.0,
                    ..CostModel::default()
                };
                let (edge, _) = choose(
                    reach,
                    cost,
                    1,
                    model,
                    vec![0, 1, 2],
                    materialisation,
                    Substages::Unknown,
                );
                assert_eq!(
                    edge, largest,
                    "reach {reach}, cost {cost}, {materialisation:?}"
                );
            }
        }
    }
}

/// The liveness test for the one above: above one worker the sweep does choose
/// something other than the largest, so "always the largest" is a fact about
/// `workers == 1` and not about the sweep.
#[test]
fn above_one_worker_the_sweep_declines_the_largest_candidate() {
    let largest = *CANDIDATES.iter().max().expect("candidates");
    let (edge, _) = choose(
        0,
        1.0,
        40,
        CostModel::default(),
        vec![0, 1, 2],
        Materialisation::Output,
        Substages::Unknown,
    );
    assert_ne!(
        edge, largest,
        "at 40 workers a zero-reach iteration still took the coarsest lattice, so the pool bound \
         is not reaching the objective and every choice in this file is the serial one"
    );
}

// ------------------------------------------- the three levers, as measured --

/// **The corrected claim.** The declared compute moves the chosen edge above one
/// worker, and does not move it at one.
///
/// Both halves are asserted, because the claim carried into this work was the
/// first half's negation and it was right about the second. A test that only
/// checked `workers == 1` would have confirmed the wrong claim.
#[test]
fn the_declared_compute_moves_the_edge_above_one_worker_and_not_at_one() {
    let model = CostModel::default();
    let costs = [1e-3f64, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3];

    for reach in [0usize, 1, 4, 16] {
        let edges: Vec<usize> = costs
            .iter()
            .map(|&cost| {
                choose(
                    reach,
                    cost,
                    1,
                    model,
                    vec![0, 1, 2],
                    Materialisation::Output,
                    Substages::Unknown,
                )
                .0
            })
            .collect();
        assert!(
            edges.iter().all(|&edge| edge == edges[0]),
            "at one worker the compute figure moved the edge for reach {reach}: {edges:?}"
        );
    }

    let mut moved = 0;
    for reach in [0usize, 1, 4, 16] {
        let edges: Vec<usize> = costs
            .iter()
            .map(|&cost| {
                choose(
                    reach,
                    cost,
                    40,
                    model,
                    vec![0, 1, 2],
                    Materialisation::Output,
                    Substages::Unknown,
                )
                .0
            })
            .collect();
        // Compute only ever pushes towards finer blocks: it is in the pool bound
        // and not the channel, so raising it can only make the pool bind sooner.
        // A non-monotone answer would mean something else moved with it.
        assert!(
            edges.windows(2).all(|pair| pair[1] <= pair[0]),
            "reach {reach}: raising the compute figure did not move monotonically towards finer \
             blocks: {edges:?}"
        );
        if edges.iter().any(|&edge| edge != edges[0]) {
            moved += 1;
        }
    }
    assert!(
        moved > 0,
        "at 40 workers a thousandfold sweep of the declared compute moved no edge at all, so the \
         compute figure is not reaching the pool bound"
    );
}

/// **The measurement that put `Materialisation` in the signature.** Which weight
/// the write is charged at moves the chosen edge.
///
/// If this ever stops holding, the parameter can go — but it holds, and the
/// argument that it could not (the write enters the channel as the volume
/// exactly, at every candidate) is true of one term of a roofline and false of
/// the other.
#[test]
fn which_weight_the_write_is_charged_at_moves_the_chosen_edge() {
    // A model whose two write weights are far apart, which is what a calibrated
    // one looks like: the default's note puts the real spread at 2.09x for raw
    // uint16 against 19.7x for the bool volumes after a binarisation.
    let model = CostModel {
        write_cost_per_voxel: 1.0,
        materialise_cost_per_voxel: 1000.0,
        ..CostModel::default()
    };
    let mut differed = 0;
    for workers in [3usize, 7, 40] {
        for reach in [0usize, 1, 4, 16] {
            for cost in [1.0f64, 10.0, 1e3] {
                let intermediate = choose(
                    reach,
                    cost,
                    workers,
                    model,
                    vec![2],
                    Materialisation::Intermediate,
                    Substages::Unknown,
                )
                .0;
                let output = choose(
                    reach,
                    cost,
                    workers,
                    model,
                    vec![2],
                    Materialisation::Output,
                    Substages::Unknown,
                )
                .0;
                if intermediate != output {
                    differed += 1;
                }
            }
        }
    }
    assert!(
        differed > 0,
        "no configuration priced an intermediate onto a different lattice than an output, so \
         `Materialisation` cannot change a plan and asking the caller for it is ceremony"
    );
}

/// And why the wrong answer was invisible: under the default model the two
/// weights are the same number, so the parameter changes nothing at all.
///
/// This is the test that explains the shape of the mistake rather than catching
/// it — a suite that only ever priced against `CostModel::default()` would have
/// passed with `Materialisation` hard-coded either way.
#[test]
fn under_the_default_model_the_two_materialisations_are_indistinguishable() {
    let model = CostModel::default();
    assert_eq!(model.write_cost_per_voxel, model.materialise_cost_per_voxel);
    for workers in [1usize, 3, 7, 40] {
        for reach in [0usize, 1, 4, 16] {
            let intermediate = choose(
                reach,
                1.0,
                workers,
                model,
                vec![2],
                Materialisation::Intermediate,
                Substages::Unknown,
            );
            let output = choose(
                reach,
                1.0,
                workers,
                model,
                vec![2],
                Materialisation::Output,
                Substages::Unknown,
            );
            assert_eq!(intermediate, output);
        }
    }
}

// ------------------------------------------------------- what it is worth --

/// **The reason to switch it on.** The priced lattice against the one
/// `PlanBuilder::iterate` would have inherited.
///
/// The assertion is one-sided on purpose: the sweep minimises over the
/// candidates, so where the inherited edge is itself a candidate the priced
/// price can never be the worse of the two, and that is a theorem worth pinning.
/// The size of the win is not asserted — it is a property of the cost model, and
/// a number here would be a number to re-baseline whenever the model moves —
/// but the file header carries the table it was read off.
#[test]
fn pricing_never_loses_to_inheriting_and_usually_beats_it() {
    let model = CostModel::default();
    let mut strictly_better = 0;
    for workers in [1usize, 8, 40] {
        for reach in [0usize, 1, 4, 16, 32] {
            let (_, priced) = choose(
                reach,
                1.0,
                workers,
                model,
                vec![0, 1, 2],
                Materialisation::Output,
                Substages::Unknown,
            );
            for &inherited in &CANDIDATES {
                let grid = BlockGrid::new(VOLUME, [inherited; 3]).expect("a lattice");
                let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
                plan.iterate(Spread { reach, cost: 1.0 }).expect("a phase");
                let assembly = plan.finish().expect("a plan");
                let work = assembly.work();
                let inherited_price = predicted_makespan(
                    &assembly.workflow.chain,
                    &assembly.decomposition,
                    &work,
                    &model,
                    workers,
                )
                .expect("a priced plan");
                assert!(
                    priced <= inherited_price,
                    "workers {workers}, reach {reach}: the sweep priced at {priced} and \
                     inheriting a {inherited} lattice prices at {inherited_price}"
                );
                if priced < inherited_price {
                    strictly_better += 1;
                }
            }
        }
    }
    assert!(
        strictly_better > 0,
        "the sweep never beat an inherited lattice anywhere in the sweep, so it is choosing the \
         grid it was handed and there is nothing to switch on"
    );
}

// ----------------------------------------------------------- what refuses --

/// An empty candidate list is a caller asking to be chosen for with nothing to
/// choose between, and is refused by name rather than answered with the grid the
/// builder happened to be holding.
#[test]
fn an_empty_candidate_list_is_refused_and_names_the_alternative() {
    let grid = BlockGrid::new(VOLUME, [64, 64, 64]).expect("a lattice");
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let message = plan
        .iterate_priced(
            Spread {
                reach: 1,
                cost: 1.0,
            },
            &Constraints {
                block_candidates: Vec::new(),
                ..Constraints::default()
            },
            40,
            Materialisation::Output,
            Substages::Unknown,
        )
        .expect_err("an empty candidate list")
        .to_string();
    assert!(message.contains("spread"), "{message}");
    assert!(message.contains("iterate"), "{message}");
}

/// A budget no candidate fits is refused with the tally in the message: how many
/// were offered, how many produced no grid, how many were too big, and what
/// would change it.
///
/// The alternative — the cheapest candidate, silently over budget — is the
/// failure this crate is arranged against.
#[test]
fn a_budget_no_candidate_fits_is_refused_with_the_tally() {
    let grid = BlockGrid::new(VOLUME, [64, 64, 64]).expect("a lattice");
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let message = plan
        .iterate_priced(
            Spread {
                reach: 1,
                cost: 1.0,
            },
            &Constraints {
                budget_bytes: Some(16),
                ..constraints(CostModel::default(), vec![0, 1, 2])
            },
            40,
            Materialisation::Output,
            Substages::Unknown,
        )
        .expect_err("a budget nothing fits")
        .to_string();
    assert!(message.contains("spread"), "{message}");
    assert!(message.contains("budget"), "{message}");
    assert!(message.contains("16"), "{message}");
}

/// The tally is what the sweep actually did, not a restatement of the candidate
/// list: everything offered is accounted for.
#[test]
fn the_tally_accounts_for_every_candidate() {
    let grid = BlockGrid::new(VOLUME, [64, 64, 64]).expect("a lattice");
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let priced = plan
        .iterate_priced(
            Spread {
                reach: 1,
                cost: 1.0,
            },
            &constraints(CostModel::default(), vec![0, 1, 2]),
            40,
            Materialisation::Output,
            Substages::Unknown,
        )
        .expect("a priced phase");
    let tally = priced.tally();
    assert_eq!(tally.offered, CANDIDATES.len());
    assert_eq!(
        tally.priced + tally.no_grid + tally.over_budget,
        tally.offered
    );
    assert!(tally.priced > 0);
}

// ------------------------------------------------ the unpriced door stays --

/// `PlanBuilder::iterate` is untouched: it still puts the phase on the lattice
/// the builder is holding, whatever a cost model would have said.
///
/// Asserted because the whole of this change is additive, and "not one existing
/// plan may change" is a claim that needs a test rather than a diff.
#[test]
fn iterate_still_inherits_the_builders_lattice() {
    for edge in CANDIDATES {
        let grid = BlockGrid::new(VOLUME, [edge; 3]).expect("a lattice");
        let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid.clone());
        plan.iterate(Spread {
            reach: 32,
            cost: 1.0,
        })
        .expect("a phase");
        let assembly = plan.finish().expect("a plan");
        assert_eq!(assembly.decomposition.phases[0].grid.block(), grid.block());
    }
}

/// And the priced door does not move the builder's own lattice either: the grid
/// it settled on is handed back for the caller to apply or not, on
/// `Partition::grid`'s argument.
#[test]
fn iterate_priced_offers_its_lattice_rather_than_imposing_it() {
    let held = BlockGrid::new(VOLUME, [64, 64, 64]).expect("a lattice");
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, held.clone());
    let priced = plan
        .iterate_priced(
            Spread {
                reach: 32,
                cost: 1.0,
            },
            &constraints(CostModel::default(), vec![0, 1, 2]),
            40,
            Materialisation::Output,
            Substages::Unknown,
        )
        .expect("a priced phase");
    assert_ne!(
        priced.grid().block(),
        held.block(),
        "the probe is meant to price onto a different lattice than the one held, or this test \
         cannot tell imposing from offering"
    );
    assert_eq!(plan.grid().block(), held.block());
}

// ------------------------------------------- the count against the lattice --
//
// The exclusion that had to be hunted rather than argued. `iterate_priced`
// prices one substage; the phase runs an unknown number of them. The argument
// for that being harmless has two parts and only one of them survives, so both
// are tested: the count really is independent of the lattice, and the exclusion
// is a bias anyway.

/// Larger of two, by `total_cmp`. `f64::max` is not used anywhere in this crate:
/// selecting between two floats through a partial order is how a `NaN` becomes
/// an answer instead of a fault.
fn larger(left: f64, right: f64) -> f64 {
    if left.total_cmp(&right).is_lt() {
        right
    } else {
        left
    }
}

fn smaller(left: f64, right: f64) -> f64 {
    if left.total_cmp(&right).is_gt() {
        right
    } else {
        left
    }
}

/// `g <- min(dilate(g, reach), f)`: reconstruction by dilation, which is the
/// shape the hypothesis is about. Information spreads one `reach` per substage
/// and is capped by the phase's input, so the count is the geodesic length of
/// the path from the seed divided by the reach.
///
/// Written here rather than borrowed from `probes` because the probe there
/// spreads along one axis at a fixed radius of one, and the question is whether
/// a *seam* costs a substage — which needs a reach wider than one, a reach that
/// differs per axis, and a spread that turns corners.
struct CappedSpread {
    reach: [usize; 3],
}

impl IterativeOp for CappedSpread {
    fn name(&self) -> &'static str {
        "capped-spread"
    }

    fn operands(&self) -> Vec<SubstageOperand> {
        vec![
            SubstageOperand::running(self.reach),
            SubstageOperand::fixed([0, 0, 0]),
        ]
    }

    fn limit(&self) -> SubstageLimit {
        // Generous on purpose: this is the runaway guard, and a tight one here
        // would turn "the count moved" into "the limit fired", which is a
        // different failure and would hide the one being looked for.
        SubstageLimit::of(4096).expect("a positive limit")
    }

    fn substage(&self, at: &Substage<'_>, out: &mut Voxels) -> Result<()> {
        let running = at.operand(0)?.view::<f64>()?;
        let cap = at.operand(1)?.view::<f64>()?;
        let shape = out.shape();
        let mut target = out.view_mut::<f64>()?;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    if at.index() == 0 {
                        let value = running[[i, j, k]];
                        target[[i, j, k]] = if value >= SEED { value } else { 0.0 };
                        continue;
                    }
                    let mut best = running[[i, j, k]];
                    for di in i.saturating_sub(self.reach[0])..(i + self.reach[0] + 1).min(shape[0])
                    {
                        for dj in
                            j.saturating_sub(self.reach[1])..(j + self.reach[1] + 1).min(shape[1])
                        {
                            for dk in k.saturating_sub(self.reach[2])
                                ..(k + self.reach[2] + 1).min(shape[2])
                            {
                                best = larger(best, running[[di, dj, dk]]);
                            }
                        }
                    }
                    target[[i, j, k]] = smaller(best, cap[[i, j, k]]);
                }
            }
        }
        Ok(())
    }
}

const SEED: f64 = 100.0;
const SPREAD_VOLUME: [usize; 3] = [12, 6, 6];

/// Open field: nothing blocks the spread, so the front is a growing box and the
/// count is the Chebyshev radius of the volume over the reach.
fn open_field() -> Voxels {
    let mut field = Array3::<f64>::from_elem(SPREAD_VOLUME, 1.0);
    field[[0, 0, 0]] = 255.0;
    field.into()
}

/// A one-voxel-wide serpentine with walls of zero cap between the rows, joined
/// end to end, so the spread has to walk the whole snake rather than cross it.
///
/// The row spacing is `2 * reach`, which is what makes the walls walls: a box
/// element of that radius may reach *into* a wall cell, but the cap zeroes it,
/// so the only way between rows is through a connector.
fn serpentine(reach: usize) -> Voxels {
    let spacing = 2 * reach.max(1);
    let mut field = Array3::<f64>::zeros(SPREAD_VOLUME);
    let mut path: Vec<[usize; 3]> = Vec::new();
    let mut row = 0usize;
    let mut forward = true;
    while row < SPREAD_VOLUME[1] {
        for step in 0..SPREAD_VOLUME[0] {
            let along = if forward {
                step
            } else {
                SPREAD_VOLUME[0] - 1 - step
            };
            path.push([along, row, 0]);
        }
        let turn = if forward { SPREAD_VOLUME[0] - 1 } else { 0 };
        for gap in 1..spacing {
            if row + gap < SPREAD_VOLUME[1] {
                path.push([turn, row + gap, 0]);
            }
        }
        forward = !forward;
        row += spacing;
    }
    for cell in &path {
        field[[cell[0], cell[1], cell[2]]] = 1.0;
    }
    field[[0, 0, 0]] = 255.0;
    field.into()
}

/// Run one iterative phase to its fixed point on `block`, and report the
/// substage count the executor discovered and the volume it wrote.
fn spread_on(reach: [usize; 3], source: Voxels, block: [usize; 3]) -> (usize, usize, Array3<f64>) {
    let grid = BlockGrid::new(SPREAD_VOLUME, block).expect("a lattice");
    let blocks = grid.n_blocks();
    let mut plan = PlanBuilder::new(SPREAD_VOLUME, Dtype::F64, grid);
    plan.iterate(CappedSpread { reach }).expect("a phase");
    let assembly = plan.finish().expect("a plan");
    let work = assembly.work();
    let env = ArrayEnvironment::for_decomposition(source, &assembly.decomposition, [4, 4, 4])
        .expect("an environment");
    let stats = execute_phases(
        "iterate",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .expect("a run");
    (
        stats.substages[0],
        blocks,
        env.output().view::<f64>().expect("f64").to_owned(),
    )
}

/// The lattices the count is checked against. Deliberately awkward: one block
/// per voxel, so that *every* step of the propagation is a halo exchange; slabs
/// on each axis in turn; and edges that divide no axis evenly, because a ragged
/// last block is where a seam argument would break first if it were going to.
///
/// The volume is small and the list is short for one reason, stated so nobody
/// widens either by accident: this test runs the real executor to a fixed point
/// once per lattice per reach per data shape, and the cost is the product of the
/// block count and the substage count. At one block per voxel that product is
/// already the largest thing in this file, and the crate is built on a shared
/// machine.
const LATTICES: [[usize; 3]; 13] = [
    [12, 6, 6],
    [6, 6, 6],
    [3, 6, 6],
    [1, 6, 6],
    [12, 3, 6],
    [12, 1, 6],
    [12, 6, 2],
    [12, 6, 1],
    [4, 3, 3],
    [2, 2, 2],
    [1, 1, 1],
    [5, 4, 4],
    [7, 5, 3],
];

const REACHES: [[usize; 3]; 4] = [[1, 0, 0], [1, 1, 1], [2, 1, 0], [3, 2, 2]];

/// **The measurement the exclusion rests on.** An iteration takes the same
/// number of substages whatever the volume is cut into.
///
/// The hypothesis this is aimed at is that a spreading iteration needs *more*
/// substages when the blocks are small, because a substage moves information one
/// reach inside a block and the halo exchange is what crosses the seam. It does
/// not, and the reason is the halo: it is one substage's reach wide and holds
/// the neighbours' cores from the substage before, so a seam is crossed at
/// exactly the rate the inside of a block is. `[1, 1, 1]` in the table is the
/// extreme case of that — one block per voxel, every step a seam — and it agrees
/// with the whole volume.
///
/// The written volume is compared too. A count that agreed while the answer
/// differed would be a worse defect than either, and checking only one of them
/// would find only one of the two.
#[test]
fn the_substage_count_does_not_depend_on_the_block_edge() {
    for reach in REACHES {
        let widest = reach.iter().copied().max().expect("three axes");
        for (name, source) in [
            ("open field", open_field()),
            ("serpentine", serpentine(widest)),
        ] {
            let (whole, one, reference) = spread_on(reach, source.clone(), SPREAD_VOLUME);
            assert_eq!(
                one, 1,
                "the reference has to be the whole volume in one block"
            );
            assert!(
                whole > 1,
                "reach {reach:?} on the {name} converged in {whole} substage(s), which is too \
                 few for a count to be able to move"
            );
            let mut cut_into = Vec::new();
            for block in LATTICES {
                let (count, blocks, written) = spread_on(reach, source.clone(), block);
                cut_into.push(blocks);
                assert_eq!(
                    count, whole,
                    "reach {reach:?} on the {name}: block {block:?} took {count} substage(s) \
                     against the whole volume's {whole}"
                );
                assert_eq!(
                    written, reference,
                    "reach {reach:?} on the {name}: block {block:?} wrote a different volume"
                );
            }
            // The plans really were cut, which is the hypothesis the equality
            // above is a statement about. Without this the whole table would
            // pass unchanged against a `spread_on` that ignored its `block`
            // argument — nineteen runs of the same plan agreeing with itself.
            assert!(
                cut_into.iter().copied().max().expect("lattices") > 1,
                "no lattice cut the volume into more than one block"
            );
            assert!(
                cut_into.windows(2).any(|pair| pair[0] != pair[1]),
                "every lattice produced the same block count: {cut_into:?}"
            );
        }
    }
}

/// The liveness test for the one above: the serpentine really does make the
/// iteration long, so the invariance is being checked over a propagation that
/// has somewhere to go.
///
/// Without this the table above could be nineteen agreeing counts of three, and
/// three substages is not a distance.
#[test]
fn the_serpentine_makes_the_iteration_longer_than_the_open_field() {
    let mut longer = 0;
    for reach in REACHES {
        let widest = reach.iter().copied().max().expect("three axes");
        let (open, _, _) = spread_on(reach, open_field(), SPREAD_VOLUME);
        let (snake, _, _) = spread_on(reach, serpentine(widest), SPREAD_VOLUME);
        if snake > open {
            longer += 1;
        }
    }
    assert!(
        longer > 0,
        "the serpentine never took longer than the open field, so the geodesic case the \
         invariance is claimed over is not being exercised at all"
    );
}

// ------------------------------------------- the count against the argmin --

/// **The second half, and the half the argument got wrong.** A count that is
/// constant across candidates still moves the choice, because it multiplies only
/// the repeated half of the price.
///
/// A phase runs `S` substages of read and compute and writes its image once, at
/// the fixed point. Ranking at `S == 1` therefore weighs the write `S` times too
/// heavily against the rest, and the residual `(S - 1) * (read + compute)` is a
/// function of the block edge through the read amplification.
#[test]
fn a_measured_substage_count_can_move_the_chosen_edge() {
    let model = CostModel::default();
    let mut moved = 0;
    let mut earliest = usize::MAX;
    let mut worst = 1.0f64;
    for axes in [vec![2usize], vec![0, 1, 2]] {
        for workers in [3usize, 7, 40, 100] {
            for reach in [0usize, 1, 2, 4, 8, 16, 32] {
                for cost in [0.1f64, 1.0, 10.0] {
                    let at_one = choose(
                        reach,
                        cost,
                        workers,
                        model,
                        axes.clone(),
                        Materialisation::Output,
                        Substages::Unknown,
                    )
                    .0;
                    for count in [2usize, 4, 8, 32, 256] {
                        let (with, best) = choose(
                            reach,
                            cost,
                            workers,
                            model,
                            axes.clone(),
                            Materialisation::Output,
                            Substages::Measured(count),
                        );
                        if with != at_one {
                            moved += 1;
                            earliest = earliest.min(count);
                            // What ranking at one substage cost: the price of the
                            // edge it chose, under the objective the phase really
                            // has. This is the number the exclusion was worth.
                            let regret = price_of(
                                at_one,
                                reach,
                                cost,
                                workers,
                                model,
                                axes.clone(),
                                Materialisation::Output,
                                Substages::Measured(count),
                            );
                            worst = larger(worst, regret / best);
                            break;
                        }
                    }
                }
            }
        }
    }
    assert!(
        moved > 0,
        "no measured substage count anywhere in the sweep chose a different lattice than pricing \
         one substage, so `Substages` cannot change a plan and asking for it is ceremony"
    );
    // And it is not an artefact of an absurd count: the departure happens at
    // counts an ordinary iteration reaches.
    assert!(
        earliest <= 8,
        "the choice only departed at a count of {earliest}, which is high enough that the \
         exclusion would have been harmless in practice"
    );
    // The size of it, which is what decides whether the parameter earns its
    // place. Asserted only to be greater than one — the magnitude is a property
    // of the cost model and belongs in the header where it can be re-baselined
    // rather than in an assertion that would have to move with the model.
    assert!(
        worst > 1.0,
        "every departure priced identically under the measured objective, so the one-substage \
         choice was never actually worse and there was nothing to correct"
    );
    println!("worst regret of ranking at one substage: {worst:.3}x");
}

/// The price of one stated lattice, by offering the sweep nothing else.
fn price_of(
    edge: usize,
    reach: usize,
    cost: f64,
    workers: usize,
    model: CostModel,
    axes: Vec<usize>,
    materialisation: Materialisation,
    substages: Substages,
) -> f64 {
    let grid = BlockGrid::new(VOLUME, [64, 64, 64]).expect("a lattice");
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let mut only = constraints(model, axes);
    only.block_candidates = vec![edge];
    plan.iterate_priced(
        Spread { reach, cost },
        &only,
        workers,
        materialisation,
        substages,
    )
    .expect("a priced phase")
    .ranked_makespan()
}

/// An unknown count is byte-identical to not asking, which is what makes the
/// parameter additive rather than a change to every existing plan.
#[test]
fn an_unknown_count_prices_exactly_as_one_substage() {
    let model = CostModel {
        write_cost_per_voxel: 1.0,
        materialise_cost_per_voxel: 7.0,
        ..CostModel::default()
    };
    for workers in [1usize, 3, 7, 40] {
        for reach in [0usize, 1, 4, 16] {
            for materialisation in [Materialisation::Intermediate, Materialisation::Output] {
                let unknown = choose(
                    reach,
                    1.0,
                    workers,
                    model,
                    vec![0, 1, 2],
                    materialisation,
                    Substages::Unknown,
                );
                let one = choose(
                    reach,
                    1.0,
                    workers,
                    model,
                    vec![0, 1, 2],
                    materialisation,
                    Substages::Measured(1),
                );
                assert_eq!(unknown, one, "workers {workers}, reach {reach}");
            }
        }
    }
}

/// The plan goes on reporting a one-substage price whatever the sweep ranked on,
/// because that is the only price the plan can report: nothing downstream of an
/// iterative phase knows the count.
///
/// So the two numbers are separate and both readable, and the one that has to
/// agree with `predicted_makespan` still does.
#[test]
fn the_reported_price_is_the_plans_whatever_the_sweep_ranked_on() {
    let model = CostModel::default();
    for count in [
        Substages::Unknown,
        Substages::Measured(1),
        Substages::Measured(37),
    ] {
        let grid = BlockGrid::new(VOLUME, [64, 64, 64]).expect("a lattice");
        let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
        let priced = plan
            .iterate_priced(
                Spread {
                    reach: 4,
                    cost: 1.0,
                },
                &constraints(model, vec![0, 1, 2]),
                40,
                Materialisation::Output,
                count,
            )
            .expect("a priced phase");
        let reported = priced.makespan();
        let ranked = priced.ranked_makespan();
        if matches!(count, Substages::Measured(37)) {
            assert_ne!(
                reported, ranked,
                "a thirty-seven substage phase ranked on the same number it reports, so the \
                 count is not reaching the ranking"
            );
        } else {
            assert_eq!(reported, ranked);
        }
        let assembly = plan.finish().expect("a plan");
        let work = assembly.work();
        let whole = predicted_makespan(
            &assembly.workflow.chain,
            &assembly.decomposition,
            &work,
            &model,
            40,
        )
        .expect("a priced plan");
        assert_eq!(reported, whole, "{count:?}");
    }
}

/// A measured count of zero is a mis-transcribed measurement, not a measurement,
/// and is refused by name.
#[test]
fn a_measured_count_of_zero_is_refused_and_names_the_alternative() {
    let grid = BlockGrid::new(VOLUME, [64, 64, 64]).expect("a lattice");
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let message = plan
        .iterate_priced(
            Spread {
                reach: 1,
                cost: 1.0,
            },
            &constraints(CostModel::default(), vec![0, 1, 2]),
            40,
            Materialisation::Output,
            Substages::Measured(0),
        )
        .expect_err("a count of zero")
        .to_string();
    assert!(message.contains("spread"), "{message}");
    assert!(message.contains("Unknown"), "{message}");
}

/// **The discriminating pair**, and the reason `Substages` scales what it scales.
///
/// A multiplier on the *whole* price is a uniform positive scaling of a
/// roofline, so it scales the objective and cannot move an argmin — which is
/// exactly the argument the exclusion rested on, and it is asserted here because
/// it is true. What breaks the argument is that the substage count is not such a
/// multiplier: it multiplies the read and the compute and leaves the single
/// image write alone, and *that* moves the choice.
///
/// Without this pair, a correction that scaled everything by the count would
/// pass every other test in this file while doing nothing the exclusion did not
/// already do.
#[test]
fn a_multiplier_on_the_whole_price_moves_nothing_and_one_on_its_repeated_half_moves_the_choice() {
    let base = CostModel::default();
    let mut differed = 0;
    for workers in [3usize, 7, 40, 100] {
        for reach in [0usize, 1, 2, 4, 8, 16, 32] {
            // The declared compute is swept with the rest: it is the term that
            // decides which side of the roofline binds, and a pair like this one
            // held at a single compute figure would be a pair held on one side
            // of it.
            for cost in [0.1f64, 1.0, 10.0] {
                let plain = choose(
                    reach,
                    cost,
                    workers,
                    base,
                    vec![2],
                    Materialisation::Output,
                    Substages::Unknown,
                )
                .0;
                for scale in [2.0f64, 4.0, 8.0, 256.0] {
                    // Every coefficient, and the op's own compute with them.
                    let uniform = CostModel {
                        read_cost_per_voxel: base.read_cost_per_voxel * scale,
                        write_cost_per_voxel: base.write_cost_per_voxel * scale,
                        materialise_cost_per_voxel: base.materialise_cost_per_voxel * scale,
                        compute_scale: base.compute_scale * scale,
                        order_conflict_penalty: base.order_conflict_penalty * scale,
                    };
                    let scaled = choose(
                        reach,
                        cost,
                        workers,
                        uniform,
                        vec![2],
                        Materialisation::Output,
                        Substages::Unknown,
                    )
                    .0;
                    assert_eq!(
                        scaled, plain,
                        "workers {workers}, reach {reach}, cost {cost}: multiplying every term of \
                     the price by {scale} moved the choice from {plain} to {scaled}, which a \
                     uniform positive scaling of a roofline cannot do"
                    );

                    let repeated = choose(
                        reach,
                        cost,
                        workers,
                        base,
                        vec![2],
                        Materialisation::Output,
                        Substages::Measured(scale as usize),
                    )
                    .0;
                    if repeated != plain {
                        differed += 1;
                    }
                }
            }
        }
    }
    assert!(
        differed > 0,
        "scaling only the repeated half of the price never moved the choice either, so the \
         substage count is being applied as a uniform multiplier and `Substages` is doing \
         nothing the exclusion did not already do"
    );
}
