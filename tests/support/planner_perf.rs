use std::collections::BTreeMap;

use blockflow::arena::SimulatorBacked;
use blockflow::decomposition::{Constraints, CostModel};
use blockflow::op::Chain;
use blockflow::probes::{AffineOp, IdentityOp};
use blockflow::scenario::Scenario;
use blockflow::simulate::{ExecutorOrder, Machine, Rates, Scheduler};
use blockflow::strategy::{Enumerating, Plan, Workflow};
use blockflow::Dtype;

pub const COSTS: &str = "costs";
pub const VOLUME: [usize; 3] = [96, 96, 96];
pub const LADDER: [usize; 4] = [16, 24, 32, 48];

pub type PlannerSimulator = SimulatorBacked<Enumerating, fn() -> Box<dyn Scheduler>>;

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
    let edges: Vec<usize> = plan
        .decomposition
        .phases
        .iter()
        .map(|phase| phase.grid.block()[0])
        .collect();
    format!("{} phase(s) at {:?}", plan.decomposition.n_phases(), edges)
}

fn phase_major_scheduler() -> Box<dyn Scheduler> {
    Box::new(ExecutorOrder::phase_major())
}
