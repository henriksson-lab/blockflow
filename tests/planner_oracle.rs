// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A simulator-backed report for planner work.
//
// This is the scoreboard TODO4 asks for: every committed scenario, the current
// planner choice, a richer field of rejected candidates, and the simulator's
// winner under the scheduling policies that can change the answer.

use std::collections::{BTreeMap, BTreeSet};

use blockflow::arena::{Arena, Judgement, SimulationObjective, SimulatorBacked, Verdict};
use blockflow::decomposition::{Constraints, CostModel, Decomposition, PhaseDecomposition};
use blockflow::distributed::handout::HandoutPolicy;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::probes::{AffineOp, IdentityOp};
use blockflow::scenario::Scenario;
use blockflow::simulate::{ExecutorOrder, Handout, Machine, Rates, Scheduler};
use blockflow::strategy::{Enumerating, PartitionSearch, Plan, Strategy, Workflow};
use blockflow::Dtype;

const COSTS: &str = "costs";
const VOLUME: [usize; 3] = [96, 96, 96];
const LADDER: [usize; 4] = [16, 24, 32, 48];

fn chain() -> Chain {
    Chain::sequence(vec![
        Chain::op(IdentityOp::new("smooth", [4, 4, 4]).with_cost(2.0)),
        Chain::op(AffineOp::new("combine", 1.5, 0.5, [1, 1, 1]).with_cost(1.0)),
        Chain::op(IdentityOp::new("skeletonize", [2, 2, 2]).with_cost(8.0)),
    ])
}

fn workflow() -> Workflow {
    Workflow::new(chain(), VOLUME, Dtype::F64)
}

fn base_constraints() -> Constraints {
    Constraints {
        block_candidates: LADDER.to_vec(),
        split_axes: vec![0, 1, 2],
        model: CostModel::default(),
        ..Default::default()
    }
}

fn scenarios() -> BTreeMap<String, Scenario> {
    Scenario::load_dir(COSTS).unwrap_or_else(|err| {
        panic!("the committed scenarios must load: {err}");
    })
}

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

fn plan_signature(plan: &Plan) -> Vec<(Vec<usize>, usize)> {
    plan.decomposition
        .phases
        .iter()
        .map(|phase| (phase.slots.clone(), phase.grid.block()[0]))
        .collect()
}

fn mixed_edge_plans(seed: &Plan, edges: &[usize]) -> Vec<(String, Plan)> {
    assert!(
        seed.decomposition
            .phases
            .iter()
            .all(|phase| !phase.reads_across_grids()),
        "the oracle report's mixed-edge expander only preserves default source regions"
    );
    let mut out = Vec::new();
    let mut chosen = vec![0usize; seed.decomposition.n_phases()];
    fn visit(
        at: usize,
        seed: &Plan,
        edges: &[usize],
        chosen: &mut [usize],
        out: &mut Vec<(String, Plan)>,
    ) {
        if at == chosen.len() {
            let mut phases = Vec::with_capacity(seed.decomposition.phases.len());
            for (phase, edge) in seed.decomposition.phases.iter().zip(chosen.iter().copied()) {
                let grid = BlockGrid::new(phase.volume(), [edge; 3])
                    .expect("a candidate edge produces a grid");
                let mut rebuilt = PhaseDecomposition::derive(
                    phase.slots.clone(),
                    phase.names.clone(),
                    phase.reach.clone(),
                    phase.halo.clone(),
                    grid,
                )
                .with_source_images(phase.source_images.clone())
                .with_supplied_dtypes(phase.supplied_dtypes.clone())
                .reading_input_image(phase.reads_input_image)
                .with_barrier(phase.barrier);
                if let Some(dtype) = phase.dtype {
                    rebuilt = rebuilt.with_dtype(dtype);
                }
                phases.push(rebuilt);
            }
            out.push((
                format!("mixed-{:?}", chosen),
                Plan {
                    decomposition: Decomposition {
                        volume: seed.decomposition.volume,
                        dtype: seed.decomposition.dtype,
                        phases,
                        chain_reach: seed.decomposition.chain_reach,
                    },
                    hints: seed.hints.clone(),
                },
            ));
            return;
        }
        for &edge in edges {
            chosen[at] = edge;
            visit(at + 1, seed, edges, chosen, out);
        }
    }
    visit(0, seed, edges, &mut chosen, &mut out);
    out
}

fn judge(
    scenario: &Scenario,
    workflow: &Workflow,
    base: &Constraints,
    case: SchedulerCase,
) -> Judgement {
    let constraints = scenario.constraints(base);
    let workers = scenario.machine.workers.max(1);
    let strategy = |constraints: &Constraints| {
        Enumerating {
            concurrency: workers,
            ..Enumerating::default()
        }
        .plan(workflow, constraints)
    };
    let chosen = strategy(&constraints)
        .unwrap_or_else(|err| panic!("{}: the planner must plan: {err}", scenario.name));

    let mut arena = Arena::new(
        case.machine(scenario.machine),
        scenario.rates(&Rates::default()),
    )
    .with_snapshot(scenario.snapshot.clone());
    let mut seen = BTreeSet::new();
    seen.insert(plan_signature(&chosen));
    arena
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
            if seen.insert(plan_signature(&plan)) {
                arena
                    .enter_plan(label.to_string(), plan, constraints.clone())
                    .expect("a plan the arena can hold");
            }
        }
    }

    for edge in LADDER {
        let pinned = Constraints {
            block_candidates: vec![edge],
            ..constraints.clone()
        };
        if let Ok(plan) = strategy(&pinned) {
            if seen.insert(plan_signature(&plan)) {
                arena
                    .enter_plan(format!("edge-{edge}"), plan, pinned)
                    .expect("a plan the arena can hold");
            }
        }
    }

    let seeds: Vec<Plan> = arena
        .entrants()
        .iter()
        .map(|entrant| entrant.plan.clone())
        .collect();
    for seed in seeds {
        for (name, plan) in mixed_edge_plans(&seed, &LADDER) {
            if seen.insert(plan_signature(&plan)) {
                arena
                    .enter_plan(name, plan, constraints.clone())
                    .expect("a mixed-edge candidate the arena can hold");
            }
        }
    }

    arena
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
    let workers = scenario.machine.workers.max(1);
    let current = Enumerating {
        concurrency: workers,
        ..Enumerating::default()
    };
    let current_plan = current
        .plan(&workflow, &constraints)
        .expect("the current planner must plan");
    let current_edges: Vec<usize> = current_plan
        .decomposition
        .phases
        .iter()
        .map(|phase| phase.grid.block()[0])
        .collect();

    let uniform_oracle = SimulatorBacked::new(
        Enumerating {
            concurrency: workers,
            ..Enumerating::default()
        },
        scenario.machine,
        scenario.rates(&Rates::default()),
        || Box::new(ExecutorOrder::phase_major()),
    )
    .with_snapshot(scenario.snapshot.clone());
    let uniform_oracle_plan = uniform_oracle
        .plan(&workflow, &constraints)
        .expect("the simulator-backed wrapper must return the uniform-ladder oracle winner");
    let uniform_oracle_edges: Vec<usize> = uniform_oracle_plan
        .decomposition
        .phases
        .iter()
        .map(|phase| phase.grid.block()[0])
        .collect();

    let mixed_oracle = SimulatorBacked::new(
        current,
        scenario.machine,
        scenario.rates(&Rates::default()),
        || Box::new(ExecutorOrder::phase_major()),
    )
    .with_snapshot(scenario.snapshot.clone())
    .with_mixed_edges();
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
    let workers = scenario.machine.workers.max(1);
    let mut waves = scenario.machine;
    waves.wave_synchronous = true;

    let oracle = SimulatorBacked::new(
        Enumerating {
            concurrency: workers,
            ..Enumerating::default()
        },
        scenario.machine,
        scenario.rates(&Rates::default()),
        || Box::new(ExecutorOrder::phase_major()),
    )
    .with_snapshot(scenario.snapshot.clone())
    .with_mixed_edges()
    .with_machine_variant("waves", waves);

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
