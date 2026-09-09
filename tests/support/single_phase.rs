use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, Chain};
use blockflow::reach::Reach;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

pub struct RunF64 {
    pub output: Array3<f64>,
    pub tasks_short_circuited: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Suite {
    name: &'static str,
    volume: [usize; 3],
    chunk: [usize; 3],
}

impl Suite {
    pub const fn new(name: &'static str, volume: [usize; 3], chunk: [usize; 3]) -> Self {
        Self {
            name,
            volume,
            chunk,
        }
    }

    pub fn workflow(&self, chain: Chain) -> Workflow {
        workflow_f64(chain, self.volume)
    }

    /// One phase holding the whole chain, using the chain's own reach.
    pub fn plan(&self, workflow: &Workflow, block: usize, split_axes: &[usize]) -> Decomposition {
        plan(workflow, self.volume, block, split_axes)
    }

    /// One phase with an explicitly supplied reach, for failure fixtures.
    pub fn plan_with_reach(
        &self,
        workflow: &Workflow,
        block: usize,
        split_axes: &[usize],
        reach: [usize; 3],
    ) -> Decomposition {
        plan_with_reach(workflow, self.volume, block, split_axes, reach)
    }

    /// One phase with the chain's own per-side reach.
    pub fn plan_with_reach_spec(
        &self,
        workflow: &Workflow,
        block: usize,
        split_axes: &[usize],
    ) -> Decomposition {
        let reach = workflow
            .chain
            .reach_spec(self.volume)
            .expect("a foldable reach");
        self.plan_with_halo_spec(workflow, block, split_axes, reach.clone(), reach)
    }

    /// One phase with explicit per-side reach and halo, for failure fixtures.
    pub fn plan_with_halo_spec(
        &self,
        workflow: &Workflow,
        block: usize,
        split_axes: &[usize],
        reach: Reach,
        halo: Reach,
    ) -> Decomposition {
        plan_with_halo_spec(workflow, self.volume, block, split_axes, reach, halo)
    }

    pub fn reference_f64(&self, chain: &Chain, input: &Array3<f64>) -> Array3<f64> {
        reference_f64(chain, input, self.volume)
    }

    pub fn run_f64(
        &self,
        workflow: &Workflow,
        decomposition: &Decomposition,
        input: &Array3<f64>,
    ) -> RunF64 {
        run_f64(self.name, workflow, decomposition, input, self.chunk)
    }
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

pub fn plan_with_reach_and_output_dtype(
    workflow: &Workflow,
    volume: [usize; 3],
    block: usize,
    split_axes: &[usize],
    reach: [usize; 3],
    output_dtype: Dtype,
) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(volume, split_axes, block).unwrap();
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid)
        .with_dtype(output_dtype);
    Decomposition {
        volume,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: reach,
    }
}

pub fn plan_on_grid(workflow: &Workflow, volume: [usize; 3], grid: BlockGrid) -> Decomposition {
    plan_on_grid_for_chain(&workflow.chain, volume, workflow.dtype, grid)
}

pub fn plan_on_grid_for_chain(
    chain: &Chain,
    volume: [usize; 3],
    dtype: Dtype,
    grid: BlockGrid,
) -> Decomposition {
    let reach = chain.reach3(&volume);
    let slots = chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid);
    Decomposition {
        volume,
        dtype,
        phases: vec![phase],
        chain_reach: reach,
    }
}

pub fn typed_plan_on_grid(
    workflow: &Workflow,
    volume: [usize; 3],
    grid: BlockGrid,
) -> Decomposition {
    let mut plan = plan_on_grid(workflow, volume, grid);
    plan.declare_dtypes(&workflow.chain).expect("element types");
    plan
}

pub fn declared_plan_on_grid(
    workflow: &Workflow,
    volume: [usize; 3],
    grid: BlockGrid,
) -> Decomposition {
    let mut plan = typed_plan_on_grid(workflow, volume, grid);
    plan.declare_source_images(&workflow.chain)
        .expect("source images");
    plan
}

pub fn declared_plan_on_grid_for_chain(
    chain: &Chain,
    volume: [usize; 3],
    dtype: Dtype,
    grid: BlockGrid,
) -> Decomposition {
    let mut plan = plan_on_grid_for_chain(chain, volume, dtype, grid);
    plan.declare_dtypes(chain).expect("element types");
    plan.declare_source_images(chain).expect("source images");
    plan
}

pub fn one_phase_per_slot(
    chain: &Chain,
    volume: [usize; 3],
    dtype: Dtype,
    grid: &BlockGrid,
    reach: [usize; 3],
) -> Decomposition {
    let slots = chain.slots();
    let phases = slots
        .iter()
        .enumerate()
        .map(|(index, slot)| {
            PhaseDecomposition::derive(
                vec![index],
                vec![slot.display_name()],
                reach,
                reach,
                grid.clone(),
            )
        })
        .collect();
    Decomposition {
        volume,
        dtype,
        phases,
        chain_reach: reach,
    }
}

pub fn typed_one_phase_per_slot(
    chain: &Chain,
    volume: [usize; 3],
    dtype: Dtype,
    grid: &BlockGrid,
    reach: [usize; 3],
) -> Decomposition {
    let mut plan = one_phase_per_slot(chain, volume, dtype, grid, reach);
    plan.declare_dtypes(chain).expect("element types");
    plan
}

pub fn declared_one_phase_per_slot(
    chain: &Chain,
    volume: [usize; 3],
    dtype: Dtype,
    grid: &BlockGrid,
    reach: [usize; 3],
) -> Decomposition {
    let mut plan = typed_one_phase_per_slot(chain, volume, dtype, grid, reach);
    plan.declare_source_images(chain).expect("source images");
    plan
}

pub fn plan_with_halo_spec(
    workflow: &Workflow,
    volume: [usize; 3],
    block: usize,
    split_axes: &[usize],
    reach: Reach,
    halo: Reach,
) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(volume, split_axes, block).unwrap();
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, halo, grid);
    Decomposition {
        volume,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: workflow.chain.reach3(&volume),
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
