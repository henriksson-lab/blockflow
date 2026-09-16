use std::collections::BTreeMap;

use blockflow::arena::{
    plan_fit, Arena, Judgement, PlanAdmission, PlanShape, RobustCase, SimulatedPlan,
    SimulatorBacked,
};
use blockflow::decomposition::{Constraints, CostModel};
use blockflow::op::Chain;
use blockflow::probes::{AffineOp, IdentityOp};
use blockflow::scenario::Scenario;
use blockflow::simulate::{ExecutorOrder, Machine, Rates, Scheduler};
use blockflow::strategy::{Enumerating, Plan, Strategy, Workflow};
use blockflow::Dtype;

pub const COSTS: &str = "costs";
pub const VOLUME: [usize; 3] = [96, 96, 96];
pub const LADDER: [usize; 4] = [16, 24, 32, 48];

pub type PlannerSimulator = SimulatorBacked<Enumerating, fn() -> Box<dyn Scheduler>>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegretBudget {
    ratio: f64,
    label: &'static str,
}

impl RegretBudget {
    pub const EXACT_CONTINUOUS: Self = Self {
        ratio: 1.01,
        label: "1%",
    };
    pub const LOW_REGRET_MACHINE: Self = Self {
        ratio: 1.05,
        label: "5%",
    };
    pub const NEAR_ORACLE: Self = Self {
        ratio: 1.10,
        label: "10%",
    };
    pub const LOCAL_ORACLE_CEILING: Self = Self {
        ratio: 1.30,
        label: "30%",
    };
    pub const ORACLE_REPORT_CEILING: Self = Self {
        ratio: 2.25,
        label: "125%",
    };

    pub fn accepts(self, regret: f64) -> bool {
        regret <= self.ratio
    }

    pub fn crossed_by(self, regret: f64) -> bool {
        regret >= self.ratio
    }

    pub fn assess(self, comparison: OracleComparison) -> RegretAssessment {
        RegretAssessment {
            budget: self,
            comparison,
        }
    }

    pub fn ratio(self) -> f64 {
        self.ratio
    }

    pub fn label(self) -> &'static str {
        self.label
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegretAssessment {
    budget: RegretBudget,
    comparison: OracleComparison,
}

impl RegretAssessment {
    pub fn accepts(self) -> bool {
        self.budget.accepts(self.comparison.regret())
    }

    pub fn crossed_by(self) -> bool {
        self.budget.crossed_by(self.comparison.regret())
    }

    pub fn regret(self) -> f64 {
        self.comparison.regret()
    }

    pub fn budget(self) -> RegretBudget {
        self.budget
    }

    pub fn comparison(self) -> OracleComparison {
        self.comparison
    }
}

pub fn chain() -> Chain {
    Chain::sequence(vec![
        Chain::op(IdentityOp::new("smooth", [4, 4, 4]).with_cost(2.0)),
        Chain::op(AffineOp::new("combine", 1.5, 0.5, [1, 1, 1]).with_cost(1.0)),
        Chain::op(IdentityOp::new("skeletonize", [2, 2, 2]).with_cost(8.0)),
    ])
}

pub fn workflow() -> Workflow {
    Workflow::new(chain(), VOLUME, Dtype::F64)
}

pub fn base_constraints() -> Constraints {
    Constraints {
        block_candidates: LADDER.to_vec(),
        split_axes: vec![0, 1, 2],
        model: CostModel::default(),
        ..Default::default()
    }
}

pub fn scenarios() -> BTreeMap<String, Scenario> {
    Scenario::load_dir(COSTS).unwrap_or_else(|err| {
        panic!("the committed scenarios must load: {err}");
    })
}

pub fn enumerating_for(machine: Machine) -> Enumerating {
    Enumerating {
        concurrency: machine.workers.max(1),
        ..Enumerating::default()
    }
}

pub fn simulator_backed_for(scenario: &Scenario) -> PlannerSimulator {
    uniform_simulator_backed_for(scenario).with_mixed_edges()
}

pub fn uniform_simulator_backed_for(scenario: &Scenario) -> PlannerSimulator {
    SimulatorBacked::new(
        enumerating_for(scenario.machine),
        scenario.machine,
        scenario.rates(&Rates::default()),
        phase_major_scheduler as fn() -> Box<dyn Scheduler>,
    )
    .with_snapshot(scenario.snapshot.clone())
}

pub fn plan_shape(plan: &Plan) -> String {
    PlanShape::from_plan(plan).to_string()
}

pub fn simulator_backed_plan_for(
    scenario: &Scenario,
    workflow: &Workflow,
    base: &Constraints,
) -> (Plan, Constraints) {
    let (chosen, constraints) = simulator_backed_choice_for(scenario, workflow, base);
    (chosen.plan, constraints)
}

pub fn simulator_backed_choice_for(
    scenario: &Scenario,
    workflow: &Workflow,
    base: &Constraints,
) -> (SimulatedPlan, Constraints) {
    let constraints = scenario.constraints(base);
    let planner = simulator_backed_for(scenario);
    let chosen = planner
        .plan_with_machine(workflow, &constraints)
        .unwrap_or_else(|err| panic!("{}: simulator-backed planning failed: {err}", scenario.name));
    (chosen, constraints)
}

#[derive(Clone)]
pub struct PlanChoice {
    pub name: String,
    pub plan: Plan,
    pub constraints: Constraints,
}

pub fn planner_choice_for(
    scenario: &Scenario,
    workflow: &Workflow,
    base: &Constraints,
) -> PlanChoice {
    let constraints = scenario.constraints(base);
    let plan = enumerating_for(scenario.machine)
        .plan(workflow, &constraints)
        .unwrap_or_else(|err| panic!("{}: the planner must plan: {err}", scenario.name));
    PlanChoice {
        name: scenario.name.clone(),
        plan,
        constraints,
    }
}

pub struct RobustPlanChoice {
    pub choice: PlanChoice,
    pub oracle_worst: f64,
    pub local_regret: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleTargetKind {
    SimulatorCandidateField,
    ExhaustiveSimulatorCandidateField,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OracleTarget {
    kind: OracleTargetKind,
    ns: f64,
    winner_name: Option<String>,
    winner_shape: Option<PlanShape>,
    admissible_candidates: Option<usize>,
}

impl OracleTarget {
    pub fn simulator_candidate_field(context: &str, ns: f64) -> Self {
        assert!(
            ns.is_finite() && ns > 0.0,
            "{context}: simulator candidate-field oracle must be finite and positive, got {ns}"
        );
        Self {
            kind: OracleTargetKind::SimulatorCandidateField,
            ns,
            winner_name: None,
            winner_shape: None,
            admissible_candidates: None,
        }
    }

    pub fn best_simulated_candidate(context: &str, judgement: &Judgement) -> Option<Self> {
        judgement.simulated_pick().map(|verdict| {
            let mut target = Self::simulator_candidate_field(context, verdict.simulated_ns());
            target.winner_name = Some(verdict.name.clone());
            target.winner_shape = Some(verdict.shape.clone());
            target
        })
    }

    pub fn exhaustive_simulator_candidate_field(
        context: &str,
        judgement: &Judgement,
    ) -> Option<Self> {
        let admissible_candidates = judgement
            .verdicts
            .iter()
            .filter(|verdict| verdict.admissible)
            .count();
        judgement.simulated_pick().map(|verdict| {
            let mut target = Self::simulator_candidate_field(context, verdict.simulated_ns());
            target.kind = OracleTargetKind::ExhaustiveSimulatorCandidateField;
            target.winner_name = Some(verdict.name.clone());
            target.winner_shape = Some(verdict.shape.clone());
            target.admissible_candidates = Some(admissible_candidates);
            target
        })
    }

    pub fn kind(&self) -> OracleTargetKind {
        self.kind
    }

    pub fn ns(&self) -> f64 {
        self.ns
    }

    pub fn winner_name(&self) -> Option<&str> {
        self.winner_name.as_deref()
    }

    pub fn winner_shape(&self) -> Option<&PlanShape> {
        self.winner_shape.as_ref()
    }

    pub fn admissible_candidates(&self) -> Option<usize> {
        self.admissible_candidates
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OracleComparison {
    candidate_ns: f64,
    oracle_ns: f64,
    regret: f64,
    target_kind: OracleTargetKind,
}

impl OracleComparison {
    pub fn against(context: &str, candidate_ns: f64, target: OracleTarget) -> Self {
        let oracle_ns = target.ns();
        assert!(
            candidate_ns.is_finite() && candidate_ns > 0.0,
            "{context}: candidate time must be finite and positive, got {candidate_ns}"
        );
        assert!(
            oracle_ns.is_finite() && oracle_ns > 0.0,
            "{context}: oracle time must be finite and positive, got {oracle_ns}"
        );
        let regret = candidate_ns / oracle_ns;
        assert!(
            regret >= 1.0,
            "{context}: regret below one is impossible ({candidate_ns} / {oracle_ns} = {regret})"
        );
        Self {
            candidate_ns,
            oracle_ns,
            regret,
            target_kind: target.kind(),
        }
    }

    pub fn regret(self) -> f64 {
        self.regret
    }

    pub fn candidate_ns(self) -> f64 {
        self.candidate_ns
    }

    pub fn oracle_ns(self) -> f64 {
        self.oracle_ns
    }

    pub fn target_kind(self) -> OracleTargetKind {
        self.target_kind
    }
}

pub fn simulated_ns_on(
    plan: &Plan,
    scenario: &Scenario,
    workflow: &Workflow,
    base: &Constraints,
) -> Option<f64> {
    match judge_plan_on_scenario(plan, scenario, workflow, base) {
        PlanAdmission::Fits { value, .. } => Some(value),
        PlanAdmission::OverBudget { .. } => None,
    }
}

pub fn judge_plan_on_scenario(
    plan: &Plan,
    scenario: &Scenario,
    workflow: &Workflow,
    base: &Constraints,
) -> PlanAdmission<f64> {
    admitted_plan_on_scenario(plan, scenario, workflow, base, |constraints| {
        let mut arena = arena_for(scenario);
        arena
            .enter_plan("candidate".to_string(), plan.clone(), constraints.clone())
            .expect("a plan the arena can hold");
        let judgement = arena
            .judge(workflow)
            .unwrap_or_else(|err| panic!("{}: candidate did not simulate: {err}", scenario.name));
        judgement.verdicts[0].simulated_ns()
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum MatrixCell {
    Ratio(f64),
    OverBudget,
}

impl MatrixCell {
    pub fn render(&self) -> String {
        match self {
            Self::Ratio(ratio) => format!("{ratio:.3}"),
            Self::OverBudget => "over".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorstCell {
    pub row: String,
    pub column: String,
    pub ratio: f64,
}

pub struct PlanMatrix {
    pub names: Vec<String>,
    pub rows: BTreeMap<String, Vec<MatrixCell>>,
    pub worst: WorstCell,
    pub column_worst: BTreeMap<String, f64>,
}

impl PlanMatrix {
    pub fn from_choices(
        choices: &[PlanChoice],
        scenarios: &BTreeMap<String, Scenario>,
        workflow: &Workflow,
        base: &Constraints,
    ) -> Self {
        let names: Vec<String> = choices.iter().map(|choice| choice.name.clone()).collect();
        let mut rows = BTreeMap::new();
        let mut worst = WorstCell {
            row: String::new(),
            column: String::new(),
            ratio: 1.0,
        };
        let mut column_worst = BTreeMap::new();

        for row in choices {
            let mut cells = Vec::with_capacity(choices.len());
            for column in choices {
                let scenario = &scenarios[&column.name];
                match transfer_ratio(&row.plan, &column.plan, scenario, workflow, base) {
                    PlanAdmission::OverBudget { .. } => cells.push(MatrixCell::OverBudget),
                    PlanAdmission::Fits { value: ratio, .. } => {
                        let entry = column_worst.entry(column.name.clone()).or_insert(1.0);
                        if ratio > *entry {
                            *entry = ratio;
                        }
                        if ratio > worst.ratio {
                            worst = WorstCell {
                                row: row.name.clone(),
                                column: column.name.clone(),
                                ratio,
                            };
                        }
                        cells.push(MatrixCell::Ratio(ratio));
                    }
                }
            }
            rows.insert(row.name.clone(), cells);
        }

        Self {
            names,
            rows,
            worst,
            column_worst,
        }
    }

    pub fn print(&self, title: &str) {
        println!("{title}");
        print!("{:<24}", "");
        for name in &self.names {
            print!("{:>10.10}", name);
        }
        println!();
        for (name, cells) in &self.rows {
            print!("{name:<24}");
            for cell in cells {
                print!("{:>10}", cell.render());
            }
            println!();
        }
    }

    pub fn contains_over_budget(&self) -> bool {
        self.rows
            .values()
            .flatten()
            .any(|cell| matches!(cell, MatrixCell::OverBudget))
    }
}

pub fn transfer_ratio(
    foreign: &Plan,
    native: &Plan,
    scenario: &Scenario,
    workflow: &Workflow,
    base: &Constraints,
) -> PlanAdmission<f64> {
    admitted_plan_on_scenario(foreign, scenario, workflow, base, |constraints| {
        let mut arena = arena_for(scenario);
        for (name, plan) in [("foreign", foreign), ("native", native)] {
            arena
                .enter_plan(name.to_string(), plan.clone(), constraints.clone())
                .expect("a plan the arena can hold");
        }
        let judgement = arena
            .judge(workflow)
            .unwrap_or_else(|err| panic!("{}: transfer simulation failed: {err}", scenario.name));
        judgement.verdicts[0].simulated_ns() / judgement.verdicts[1].simulated_ns()
    })
}

pub fn robust_simulator_backed_plan_for(
    scenario: &Scenario,
    scenarios: &BTreeMap<String, Scenario>,
    workflow: &Workflow,
    base: &Constraints,
) -> RobustPlanChoice {
    let constraints = scenario.constraints(base);
    let planner = simulator_backed_for(scenario);
    let candidate_arena = planner
        .candidate_arena(workflow, &constraints)
        .unwrap_or_else(|err| panic!("{}: candidate arena failed: {err}", scenario.name));
    let baselines: BTreeMap<&str, f64> = scenarios
        .iter()
        .map(|(name, column)| {
            let (choice, _) = simulator_backed_choice_for(column, workflow, base);
            let best = choice
                .judgement
                .simulated_pick()
                .expect("a simulator-backed candidate field has a winner")
                .simulated_ns();
            (name.as_str(), best)
        })
        .collect();
    let cases: Vec<RobustCase> = scenarios
        .iter()
        .map(|(name, column)| {
            let case = RobustCase::new(
                name.clone(),
                column.machine,
                column.rates(&Rates::default()),
                column.constraints(base),
                baselines[name.as_str()],
            )
            .with_snapshot(column.snapshot.clone());
            if name == &scenario.name {
                case.local()
            } else {
                case
            }
        })
        .collect();
    let chosen = candidate_arena
        .robust_pick_with(workflow, &cases, &mut || {
            Box::new(ExecutorOrder::phase_major())
        })
        .unwrap_or_else(|err| {
            panic!(
                "{}: robust candidate selection failed: {err}",
                scenario.name
            )
        });
    RobustPlanChoice {
        choice: PlanChoice {
            name: scenario.name.clone(),
            plan: chosen.plan,
            constraints,
        },
        oracle_worst: chosen.worst_regret,
        local_regret: chosen.local_regret,
    }
}

fn admitted_plan_on_scenario<T>(
    plan: &Plan,
    scenario: &Scenario,
    workflow: &Workflow,
    base: &Constraints,
    value: impl FnOnce(&Constraints) -> T,
) -> PlanAdmission<T> {
    let constraints = scenario.constraints(base);
    let workers = scenario.machine.workers.max(1);
    let fit = plan_fit(workflow, &plan.decomposition, &constraints, workers)
        .unwrap_or_else(|err| panic!("{}: fit failed: {err}", scenario.name));
    if let Some(refusal) = fit.refused() {
        return refusal;
    }
    fit.with_admitted_value(|| value(&constraints))
}

fn arena_for(scenario: &Scenario) -> Arena {
    Arena::new(scenario.machine, scenario.rates(&Rates::default()))
        .with_snapshot(scenario.snapshot.clone())
}

fn phase_major_scheduler() -> Box<dyn Scheduler> {
    Box::new(ExecutorOrder::phase_major())
}
