// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A simulator-backed report for planner work.
//
// This is the scoreboard TODO4 asks for: every committed scenario, the current
// planner choice, a richer field of rejected candidates, and the simulator's
// winner under the scheduling policies that can change the answer.

use blockflow::arena::{CandidateFieldBuilder, Judgement, SimulationObjective, Verdict};
use blockflow::decomposition::Constraints;
use blockflow::distributed::handout::HandoutPolicy;
use blockflow::scenario::Scenario;
use blockflow::simulate::{ExecutorOrder, Handout, Machine, Rates, Scheduler};
use blockflow::strategy::{Enumerating, PartitionSearch, Strategy, Workflow};

mod support;

use support::planner_perf::{
    base_constraints, enumerating_for, scenarios, simulator_backed_for,
    uniform_simulator_backed_for, workflow, LADDER,
};

#[derive(Clone, Copy)]
enum SchedulerCase {
    Continuous,
    Waves,
    NearestFirst,
    Coalescing,
}

impl SchedulerCase {
    fn name(self) -> &'static str {
        match self {
            Self::Continuous => "continuous",
            Self::Waves => "waves",
            Self::NearestFirst => "nearest-first",
            Self::Coalescing => "coalescing",
        }
    }

    fn machine(self, mut machine: Machine) -> Machine {
        machine.wave_synchronous = matches!(self, Self::Waves);
        machine
    }

    fn scheduler(self) -> Box<dyn Scheduler> {
        match self {
            Self::Continuous | Self::Waves => Box::new(ExecutorOrder::phase_major()),
            Self::NearestFirst => Box::new(Handout::new(HandoutPolicy::NearestFirst)),
            Self::Coalescing => Box::new(Handout::new(HandoutPolicy::Coalescing)),
        }
    }
}

fn edge_signature(verdict: &Verdict) -> String {
    let edges: Vec<usize> = verdict.edges.iter().map(|edge| edge[0]).collect();
    format!("{}p{:?}", verdict.phases, edges)
}

fn judge(
    scenario: &Scenario,
    workflow: &Workflow,
    base: &Constraints,
    case: SchedulerCase,
) -> Judgement {
    let constraints = scenario.constraints(base);
    let workers = scenario.machine.workers.max(1);
    let enumerating = enumerating_for(scenario.machine);
    let strategy = |constraints: &Constraints| enumerating.plan(workflow, constraints);
    let chosen = strategy(&constraints)
        .unwrap_or_else(|err| panic!("{}: the planner must plan: {err}", scenario.name));

    let mut field = CandidateFieldBuilder::new(
        case.machine(scenario.machine),
        scenario.rates(&Rates::default()),
    )
    .with_snapshot(scenario.snapshot.clone());
    field
        .enter_plan("planner", chosen, constraints.clone())
        .expect("a plan the arena can hold");

    for (label, search) in [
        ("search-dp", PartitionSearch::Dp),
        ("search-exhaustive", PartitionSearch::Exhaustive),
        ("search-single", PartitionSearch::SingleGroup),
    ] {
        let variant = Enumerating {
            concurrency: workers,
            search,
            ..Enumerating::default()
        };
        if let Ok(plan) = variant.plan(workflow, &constraints) {
            field
                .enter_plan(label.to_string(), plan, constraints.clone())
                .expect("a plan the arena can hold");
        }
    }

    field
        .enter_pinned_edges(
            |edge| format!("edge-{edge}"),
            &enumerating_for(scenario.machine),
            workflow,
            &constraints,
        )
        .expect("pinned-edge candidates can be entered");
    field
        .enter_mixed_edges(
            |_, chosen| format!("mixed-{chosen:?}"),
            &LADDER,
            &constraints,
        )
        .expect("mixed-edge candidates can be entered");

    field
        .finish()
        .judge_with(workflow, &mut || case.scheduler())
        .unwrap_or_else(|err| panic!("{} under {}: {err}", scenario.name, case.name()))
}

/// **Planner oracle report.**
///
/// This is a report first and a bound second. Run with `--nocapture` to get the
/// table to paste into design docs:
///
/// ```text
/// cargo test -p blockflow --test planner_oracle -- --nocapture
/// ```
///
/// The candidate field is the current planner pick, distinct partition-search
/// answers, uniform block-edge rungs, and mixed per-phase edge variants derived
/// from those partitions.
#[test]
fn report_planner_choices_against_the_simulator_oracle() {
    let workflow = workflow();
    let base = base_constraints();
    let scenarios = scenarios();
    let cases = [
        SchedulerCase::Continuous,
        SchedulerCase::Waves,
        SchedulerCase::NearestFirst,
        SchedulerCase::Coalescing,
    ];

    println!(
        "{:<24} {:<14} {:<16} {:<16} {:>7} {:>7} {:>5} {:>9} {:>8} {:>8} {:>8}",
        "scenario",
        "scheduler",
        "model-pick",
        "sim-pick",
        "regret",
        "tau",
        "inad",
        "fetchMiB",
        "prefMiB",
        "dup",
        "ioWait"
    );

    let mut exact_continuous = 0usize;
    let mut worst_continuous = 1.0f64;
    let mut worst_any = 1.0f64;
    for (name, scenario) in &scenarios {
        for case in cases {
            let judgement = judge(scenario, &workflow, &base, case);
            let model = judgement
                .model_pick()
                .unwrap_or_else(|| panic!("{name} {}: no model pick", case.name()));
            let simulated = judgement
                .simulated_pick()
                .unwrap_or_else(|| panic!("{name} {}: no simulator pick", case.name()));
            let regret = judgement
                .regret()
                .unwrap_or_else(|| panic!("{name} {}: no regret", case.name()));
            let tau = judgement
                .kendall_tau()
                .unwrap_or_else(|| panic!("{name} {}: no rank correlation", case.name()));
            let inadmissible = judgement
                .verdicts
                .iter()
                .filter(|verdict| !verdict.admissible)
                .count();
            let pref_mib = simulated.outcome.prefetched_bytes as f64 / (1024.0 * 1024.0);
            println!(
                "{name:<24} {:<14} {:<16} {:<16} {:>7.3} {:>7.3} {:>5} {:>9.1} {:>8.1} {:>8} {:>8.1}",
                case.name(),
                format!("{} {}", model.name, edge_signature(model)),
                format!("{} {}", simulated.name, edge_signature(simulated)),
                regret,
                tau,
                inadmissible,
                simulated.outcome.fetched_bytes as f64 / (1024.0 * 1024.0),
                pref_mib,
                simulated.outcome.duplicated_fetches,
                simulated.outcome.io_wait_ns as f64 / 1_000_000.0,
            );

            assert!(
                regret >= 1.0,
                "{name} {}: regret below one is impossible",
                case.name()
            );
            assert!(
                regret <= 2.5,
                "{name} {}: planner regret {regret:.3} exceeded the recorded oracle-report \
                 ceiling. If this is deliberate, update TODO4's baseline and this table.",
                case.name()
            );
            worst_any = worst_any.max(regret);
            if matches!(case, SchedulerCase::Continuous) {
                worst_continuous = worst_continuous.max(regret);
                exact_continuous += usize::from(regret <= 1.01);
            }
        }
    }

    println!(
        "summary: exact continuous {exact_continuous}/{}; worst continuous {worst_continuous:.3}; worst any {worst_any:.3}",
        scenarios.len()
    );
    assert!(
        exact_continuous >= 4,
        "the continuous baseline no longer has the four exact scenarios TODO4 records"
    );
    assert!(
        worst_continuous <= 2.25,
        "continuous worst regret {worst_continuous:.3} exceeded the recorded cost-scenario bound"
    );
}

#[test]
fn simulator_backed_ranking_is_an_opt_in_strategy_that_closes_the_two_node_oracle_gap() {
    let workflow = workflow();
    let base = base_constraints();
    let scenario = Scenario::load("costs/two-nodes.json").expect("the committed scenario");
    let constraints = scenario.constraints(&base);
    let current = enumerating_for(scenario.machine);
    let current_plan = current
        .plan(&workflow, &constraints)
        .expect("the current planner must plan");
    let current_edges: Vec<usize> = current_plan
        .decomposition
        .phases
        .iter()
        .map(|phase| phase.grid.block()[0])
        .collect();

    let uniform_oracle = uniform_simulator_backed_for(&scenario);
    let uniform_oracle_plan = uniform_oracle
        .plan(&workflow, &constraints)
        .expect("the simulator-backed wrapper must return the uniform-ladder oracle winner");
    let uniform_oracle_edges: Vec<usize> = uniform_oracle_plan
        .decomposition
        .phases
        .iter()
        .map(|phase| phase.grid.block()[0])
        .collect();

    let mixed_oracle = simulator_backed_for(&scenario);
    let mixed_oracle_plan = mixed_oracle
        .plan(&workflow, &constraints)
        .expect("the simulator-backed wrapper must return the mixed-edge oracle winner");
    let mixed_oracle_edges: Vec<usize> = mixed_oracle_plan
        .decomposition
        .phases
        .iter()
        .map(|phase| phase.grid.block()[0])
        .collect();

    assert_eq!(
        current_edges,
        vec![48, 24, 48],
        "the fixture no longer exercises the recorded current planner gap"
    );
    assert_eq!(
        uniform_oracle_edges,
        vec![48, 48, 48],
        "the conservative simulator-backed strategy should pick the uniform-ladder simulator winner"
    );
    assert_eq!(
        mixed_oracle_edges,
        vec![48, 32, 48],
        "the mixed-edge simulator-backed strategy should pick the richer-field simulator winner"
    );
}

#[test]
fn simulator_backed_ranking_can_choose_the_wave_synchronous_machine_contract() {
    let workflow = workflow();
    let base = base_constraints();
    let scenario = Scenario::load("costs/two-nodes.json").expect("the committed scenario");
    let constraints = scenario.constraints(&base);
    let mut waves = scenario.machine;
    waves.wave_synchronous = true;

    let oracle = simulator_backed_for(&scenario).with_machine_variant("waves", waves);

    let fastest = oracle
        .plan_with_machine(&workflow, &constraints)
        .expect("the simulator-backed wrapper must choose the fastest machine contract");
    assert_eq!(
        fastest.machine_name, "default",
        "continuous dispatch still has the fastest sampled simulator oracle for this fixture"
    );

    let chosen = oracle
        .plan_with_machine_by(
            &workflow,
            &constraints,
            SimulationObjective::LowestPlannerRegret,
        )
        .expect("the simulator-backed wrapper must choose a machine contract");

    assert_eq!(
        chosen.machine_name, "waves",
        "the explicit wave-synchronous contract should win when the policy objective is \
         planner-vs-simulator regret"
    );
    assert!(
        chosen.machine.wave_synchronous,
        "the chosen machine contract must state the dispatch model"
    );
    assert!(
        chosen.regret <= 1.05,
        "under the chosen dispatch contract the planner field should have little regret, got {:.3}",
        chosen.regret
    );
}
