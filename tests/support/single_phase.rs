use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, Chain};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

pub struct RunF64 {
    pub output: Array3<f64>,
    pub tasks_short_circuited: usize,
}

pub fn workflow_f64(chain: Chain, volume: [usize; 3]) -> Workflow {
    Workflow::new(chain, volume, Dtype::F64)
}

pub fn plan(
    workflow: &Workflow,
    volume: [usize; 3],
    block: usize,
    split_axes: &[usize],
) -> Decomposition {
    plan_with_reach(
        workflow,
        volume,
        block,
        split_axes,
        workflow.chain.reach3(&volume),
    )
}

pub fn plan_with_reach(
    workflow: &Workflow,
    volume: [usize; 3],
    block: usize,
    split_axes: &[usize],
    reach: [usize; 3],
) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(volume, split_axes, block).unwrap();
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid);
    Decomposition {
        volume,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: reach,
    }
}

pub fn reference_f64(chain: &Chain, input: &Array3<f64>, volume: [usize; 3]) -> Array3<f64> {
    let source: Voxels = input.clone().into();
    let mut out = Voxels::zeros(Dtype::F64, volume).unwrap();
    chain
        .apply(&source, &mut out, &Anchor::whole(volume))
        .expect("the whole-volume reference must run");
    out.view::<f64>().unwrap().to_owned()
}

pub fn run_f64(
    name: &'static str,
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Array3<f64>,
    chunk: [usize; 3],
) -> RunF64 {
    let env = ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), chunk).unwrap();
    let stats = execute(name, workflow, decomposition, &Hints::default(), &env).unwrap();
    RunF64 {
        output: env.output().view::<f64>().unwrap().to_owned(),
        tasks_short_circuited: stats.tasks_short_circuited,
    }
}
