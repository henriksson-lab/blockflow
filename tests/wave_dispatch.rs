// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The two models of a run, compared** — item C of
// `docs/design/planner-gaps.md`, settled far enough to say what it is worth.
//
// `strategy::execute` pops a wave of ready tasks, runs it, and **joins the whole
// wave** before the next. `simulate` dispatches continuously: a task starts the
// moment its own dependencies are met, so phases overlap. Neither module said so
// and nothing could tell them apart, which is what that item records.
//
// `Machine::wave_synchronous` is the field that makes both simulable, and it
// costs nothing to model: the ready set already knew how to hold a task back
// until an earlier phase finished, because a barrier phase does exactly that.
// Turning it on for *every* phase is the executor's discipline.
//
// What this file establishes, in order:
//
// | claim | why it is here |
// |---|---|
// | continuous dispatch overlaps phases and wave-synchronous does not | the field must do the thing it is named for, measured on `Outcome::phase_overlap` |
// | with nothing contending, the two models agree to within a percent | which is why the divergence went unnoticed: at `contention: 0.0`, the default of every recorded figure, there is almost nothing to notice |
// | with contention on, they disagree by **47%** on a mixed-grid plan | and that is the number item C never had |
// | the disagreement is *about the plan*: the two models rank two plans oppositely | a divergence that changed no decision would be a curiosity. This one changes which plan a planner should choose |

use std::collections::BTreeSet;

use blockflow::assemble::{Assembly, PlanBuilder};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::probes::IdentityOp;
use blockflow::simulate::{
    simulate, ExecutorOrder, Machine, Outcome, PerPhase, Rates, MEASURED_CONTENTION,
};
use blockflow::Dtype;

const VOLUME: [usize; 3] = [96, 96, 96];

/// A chain whose phases are wildly unequal in cost, which is what makes the
/// overlap worth anything: two cheap phases either side of an expensive one, so
/// that letting the cheap one start early puts workers beside the expensive one.
///
/// The costs are the tile run's own spread — `combine` at 3.541 against
/// `skeletonize` at 201.397 nanoseconds per voxel — expressed as declared costs
/// against one rate.
fn plan(edges: [usize; 3]) -> Assembly {
    let grid = BlockGrid::new(VOLUME, [edges[0]; 3]).expect("a grid");
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    builder
        .pixels(Chain::op(
            IdentityOp::new("smooth", [4, 4, 4]).with_cost(2.0),
        ))
        .expect("a phase");
    builder.regrid(BlockGrid::new(VOLUME, [edges[1]; 3]).expect("a grid"));
    builder
        .pixels(Chain::op(
            IdentityOp::new("combine", [1, 1, 1]).with_cost(0.05),
        ))
        .expect("a phase");
    builder.regrid(BlockGrid::new(VOLUME, [edges[2]; 3]).expect("a grid"));
    builder
        .pixels(Chain::op(
            IdentityOp::new("skeletonize", [2, 2, 2]).with_cost(8.0),
        ))
        .expect("a phase");
    builder.finish().expect("an assembly")
}

fn machine(contention: f64, wave_synchronous: bool) -> Machine {
    Machine {
        nodes: 2,
        workers: 16,
        cache_bytes: 1 << 30,
        prefetch_depth: 1,
        io_channels: 4,
        cache_shared: true,
        encoded_fraction: 0.0,
        contention,
        wave_synchronous,
    }
}

fn run(assembly: &Assembly, machine: Machine) -> Outcome {
    simulate(
        &assembly.decomposition,
        &assembly.work(),
        &machine,
        &Rates {
            chunk: [64, 64, 64],
            chunk_bytes: 64 * 64 * 64 * 8,
            ..Rates::default()
        },
        &BTreeSet::new(),
        &BTreeSet::new(),
        PerPhase::default(),
        &mut ExecutorOrder::phase_major(),
    )
    .expect("a simulable plan")
}

/// **The field does what it is named for**: phases stop overlapping.
///
/// `Outcome::phase_overlap` is the sum of the phases' own spans over the
/// makespan — above one is a run that pipelined. Under the executor's discipline
/// it must be exactly `1.000`, because a phase cannot begin before the one
/// before it has ended and the spans tile the run.
#[test]
fn wave_synchronous_dispatch_stops_the_phases_overlapping() {
    let assembly = plan([48, 24, 48]);
    let continuous = run(&assembly, machine(MEASURED_CONTENTION, false));
    let waves = run(&assembly, machine(MEASURED_CONTENTION, true));

    assert!(
        continuous.phase_overlap().expect("a run") > 1.0,
        "continuous dispatch did not overlap the phases, so this fixture cannot tell the two \
         models apart"
    );
    assert_eq!(
        waves.phase_overlap(),
        Some(1.0),
        "a phase started before the one before it finished"
    );
    // The plan is the plan either way: a dispatch discipline chooses when, never
    // what.
    assert_eq!(continuous.tasks_run, waves.tasks_run);
    assert_eq!(continuous.written_bytes, waves.written_bytes);
    assert_eq!(continuous.materialised_bytes, waves.materialised_bytes);
    println!(
        "overlap {:.3} continuous, {:.3} in waves",
        continuous.phase_overlap().unwrap(),
        waves.phase_overlap().unwrap()
    );
}

/// **What the divergence is worth, at both ends of the contention axis** — on
/// the plan and the machine it was found on rather than a fixture built to
/// show it.
///
/// The measurement item C never had. `costs/two-nodes` is the machine, the chain
/// is `tests/cost_scenarios.rs`'s, and the two plans are the ones that file
/// records: the mixed grid the planner chooses there, and the uniform grid the
/// simulator prefers. Both judged through the arena, so the per-phase compute
/// rates come from the scenario's own snapshot — which is what makes the phases
/// unequal enough for the overlap to cost anything.
///
/// Recorded:
///
/// ```text
///     contention   mixed / uniform, continuous   mixed / uniform, in waves
///          0.00              0.991                        0.991
///          0.40              1.467                        0.986
/// ```
///
/// With nothing contending the two models of a run agree, which is exactly why
/// the divergence went unnoticed: `contention` defaults to zero and every figure
/// this crate has recorded was taken there. Turn it on — to the value fitted to
/// a real run — and **the two models rank the two plans oppositely**. The
/// continuous model makes the mixed grid **47% worse**; the executor's own
/// discipline makes it **0.3% better** — the same plan, the same machine and
/// the same tasks, and the only difference is whether a phase may start before
/// the one before it has ended.
///
/// **The simulator is the pessimistic one, and that is the direction that
/// matters**: a planner tuned against it avoids a plan the executor would run
/// well. Which of the two to believe is now a stated question with a number on
/// it rather than an unnoticed difference between two modules.
#[test]
fn the_two_models_rank_a_mixed_grid_oppositely_once_workers_contend() {
    use blockflow::arena::Arena;
    use blockflow::decomposition::{Constraints, CostModel};
    use blockflow::probes::AffineOp;
    use blockflow::scenario::Scenario;
    use blockflow::strategy::{Enumerating, Strategy, Workflow};

    // The chain and the ladder of `tests/cost_scenarios.rs`, which is where the
    // mixed grid and its 1.467 were recorded.
    let chain = Chain::sequence(vec![
        Chain::op(IdentityOp::new("smooth", [4, 4, 4]).with_cost(2.0)),
        Chain::op(AffineOp::new("combine", 1.5, 0.5, [1, 1, 1]).with_cost(1.0)),
        Chain::op(IdentityOp::new("skeletonize", [2, 2, 2]).with_cost(8.0)),
    ]);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let scenario = Scenario::load("costs/two-nodes.json").expect("the committed scenario");
    let base = Constraints {
        block_candidates: vec![16, 24, 32, 48],
        split_axes: vec![0, 1, 2],
        model: CostModel::default(),
        ..Default::default()
    };
    let constraints = scenario.constraints(&base);
    let workers = scenario.machine.workers.max(1);
    let plan_with = |constraints: &Constraints| {
        Enumerating {
            concurrency: workers,
            ..Enumerating::default()
        }
        .plan(&workflow, constraints)
        .expect("a plan")
    };
    let mixed = plan_with(&constraints);
    let uniform = plan_with(&Constraints {
        block_candidates: vec![48],
        ..constraints.clone()
    });
    let shape = |plan: &blockflow::strategy::Plan| {
        plan.decomposition
            .phases
            .iter()
            .map(|phase| phase.grid.block()[0])
            .collect::<Vec<_>>()
    };
    assert_ne!(
        shape(&mixed),
        shape(&uniform),
        "the planner no longer chooses a mixed grid here, so there are not two plans to rank"
    );
    println!(
        "the planner chooses {:?}; the uniform rung is {:?}",
        shape(&mixed),
        shape(&uniform)
    );

    println!("{:>12} {:>14} {:>14}", "contention", "continuous", "waves");
    let mut ratios = Vec::new();
    for contention in [0.0, MEASURED_CONTENTION] {
        let mut row = Vec::new();
        for wave_synchronous in [false, true] {
            let machine = Machine {
                contention,
                wave_synchronous,
                ..scenario.machine
            };
            let mut arena = Arena::new(machine, scenario.rates(&Rates::default()))
                .with_snapshot(scenario.snapshot.clone());
            for (name, plan) in [("mixed", &mixed), ("uniform", &uniform)] {
                arena
                    .enter_plan(name.to_string(), plan.clone(), constraints.clone())
                    .expect("a plan the arena can hold");
            }
            let judged = arena.judge(&workflow).expect("both plans simulate");
            row.push(judged.verdicts[0].simulated_ns() / judged.verdicts[1].simulated_ns());
        }
        println!("{contention:>12.2} {:>14.3} {:>14.3}", row[0], row[1]);
        ratios.push(row);
    }

    let (free, contended) = (&ratios[0], &ratios[1]);
    assert!(
        (free[0] - free[1]).abs() < 0.02,
        "without contention the two dispatch models rank the plans differently ({:.3} against \
         {:.3}), where they should agree: the wave discipline changes when work happens, not \
         how much",
        free[0],
        free[1]
    );
    assert!(
        contended[0] > 1.2,
        "the continuous model no longer penalises the mixed grid ({:.3}), which is the \
         finding this test was written around and the 1.467 `tests/cost_scenarios.rs` records",
        contended[0]
    );
    assert!(
        contended[1] < 1.05,
        "under the executor's own discipline the mixed grid costs {:.3}x the uniform one. The \
         two models used to rank these oppositely; if they now agree, item C of \
         `docs/design/planner-gaps.md` is settled and this is where the new numbers go.",
        contended[1]
    );
}
