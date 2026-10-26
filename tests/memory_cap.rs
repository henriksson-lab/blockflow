// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A memory cap as a boundary condition: the grid choice sizes every phase's
// active set (its blocks in flight) under the budget and prices what that
// costs; a cap too small for any candidate is refused with the cap that would
// do; and the executor admits a wave only as far as its blocks fit, with the
// same voxels as an unadmitted run.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::Array3;

use blockflow::assemble::PlanBuilder;
use blockflow::budget::MemoryBudget;
use blockflow::decomposition::CostModel;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::listener::EventListener;
use blockflow::log::Event;
use blockflow::op::Chain;
use blockflow::ops::smooth::{Gaussian, SmoothOp};
use blockflow::strategy::{
    choose_grid, execute_phases, expected_bytes_per_block, ChosenGrid, Hints, Host,
};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [64, 256, 256];

/// A blur in the image plane only: a reach of 32 in `y` and `x`, none in `z`.
fn plane_blur(grid: BlockGrid) -> blockflow::Result<(blockflow::assemble::Assembly, ())> {
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    plan.pixels(Chain::op(SmoothOp::new(
        "plane blur",
        Gaussian::new([0.0, 8.0, 8.0], 4.0)?,
    )))?;
    Ok((plan.finish()?, ()))
}

fn host(workers: usize, budget_bytes: u64) -> Host {
    Host {
        workers,
        budget_bytes,
    }
}

fn choose(workers: usize, budget_bytes: u64) -> blockflow::Result<ChosenGrid<()>> {
    choose_grid(
        VOLUME,
        &host(workers, budget_bytes),
        &CostModel::default(),
        plane_blur,
    )
}

#[test]
fn with_room_to_spare_every_phase_runs_a_full_wave_and_the_choice_does_not_move() {
    let roomy = choose(8, 64 << 30).unwrap();
    let vast = choose(8, u64::MAX / 4).unwrap();
    assert_eq!(roomy.grid.block(), vast.grid.block());
    assert_eq!(roomy.makespan.to_bits(), vast.makespan.to_bits());
    for (set, phase) in roomy
        .active
        .iter()
        .zip(&roomy.assembly.decomposition.phases)
    {
        assert_eq!(set.in_flight, 8.min(phase.grid.n_blocks()));
    }
}

#[test]
fn under_a_cap_every_phase_fits_and_the_plan_is_priced_slower() {
    let roomy = choose(8, 64 << 30).unwrap();
    let widest = roomy.active.iter().map(|set| set.bytes()).max().unwrap();
    // Room for about a third of the roomy plan's widest wave.
    let budget = widest / 3;
    let tight = choose(8, budget).unwrap();
    for set in &tight.active {
        assert!(set.bytes() <= budget, "{set:?} over {budget}");
        assert!(set.in_flight >= 1);
    }
    // It fits by running fewer blocks at once or by cutting smaller, and
    // either way it is not predicted to be faster.
    let fewer = tight
        .active
        .iter()
        .zip(&tight.assembly.decomposition.phases)
        .any(|(set, phase)| set.in_flight < 8.min(phase.grid.n_blocks()));
    assert!(fewer || tight.grid.n_blocks() > roomy.grid.n_blocks());
    assert!(tight.makespan >= roomy.makespan);
}

#[test]
fn a_cap_below_every_candidate_is_refused_with_the_cap_that_would_do() {
    // The finest candidate's block is the least any plan can hold.
    let finest = blockflow::strategy::grid_candidates(VOLUME)
        .into_iter()
        .map(|block| {
            let (assembly, ()) = plane_blur(BlockGrid::new(VOLUME, block).unwrap()).unwrap();
            expected_bytes_per_block(&assembly.workflow.chain, &assembly.decomposition)
                .into_iter()
                .max()
                .unwrap()
        })
        .min()
        .unwrap();
    let refused = match choose(8, finest / 2) {
        Ok(chosen) => panic!("{:?} was chosen under half its block", chosen.grid.block()),
        Err(error) => error.to_string(),
    };
    assert!(refused.contains("least memory cap"), "{refused}");
    let needed = blockflow::budget::MemoryCap::needed_for(finest).div_ceil(1 << 20);
    assert!(refused.contains(&format!("{needed} MiB")), "{refused}");
    // And at that block's own size it runs, one block at a time if need be.
    assert!(choose(8, finest).is_ok());
}

/// The most blocks the executor had admitted and not yet written at once.
#[derive(Default)]
struct InFlight {
    now: AtomicUsize,
    most: AtomicUsize,
}

impl EventListener for InFlight {
    fn on_event(&self, event: &Event) {
        match event {
            Event::TaskAdmitted { .. } => {
                let now = self.now.fetch_add(1, Ordering::SeqCst) + 1;
                self.most.fetch_max(now, Ordering::SeqCst);
            }
            Event::BlockWritten { .. } => {
                self.now.fetch_sub(1, Ordering::SeqCst);
            }
            _ => {}
        }
    }
}

fn input() -> Voxels {
    Voxels::from(Array3::from_shape_fn(
        (VOLUME[0], VOLUME[1], VOLUME[2]),
        |(z, y, x)| ((z * 31 + y * 7 + x * 13) % 101) as f64,
    ))
}

fn run(memory: Option<MemoryBudget>) -> (Voxels, usize) {
    let (assembly, ()) = plane_blur(BlockGrid::new(VOLUME, [8, 256, 256]).unwrap()).unwrap();
    let env =
        ArrayEnvironment::for_decomposition(input(), &assembly.decomposition, [8, 64, 64]).unwrap();
    let counting = Arc::new(InFlight::default());
    let listeners: Vec<Arc<dyn EventListener>> = vec![counting.clone()];
    execute_phases(
        "memory cap",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints {
            concurrency: 8,
            memory,
            ..Hints::default()
        },
        &env,
        &listeners,
        &assembly.work(),
    )
    .unwrap();
    (env.output(), counting.most.load(Ordering::SeqCst))
}

#[test]
fn the_executor_admits_only_the_blocks_that_fit_and_writes_the_same_voxels() {
    let (assembly, ()) = plane_blur(BlockGrid::new(VOLUME, [8, 256, 256]).unwrap()).unwrap();
    let charge = blockflow::strategy::active_bytes_per_block(
        &assembly.workflow.chain,
        &assembly.decomposition,
    )[0];
    let (free, free_most) = run(None);
    assert_eq!(free_most, 8, "eight blocks, eight workers, no admission");
    let (roomy, roomy_most) = run(Some(MemoryBudget::new(charge * 64)));
    assert_eq!(roomy_most, 8, "room for every block changes nothing");
    // Room for two blocks' charge: at most two at once (fewer if the resident
    // set the executor measures near the cap says so), and never none.
    let (tight, tight_most) = run(Some(MemoryBudget::new(charge * 2 + charge / 2)));
    assert!((1..=2).contains(&tight_most), "{tight_most} in flight");
    assert!(free == roomy, "admission with room to spare moved a voxel");
    assert!(
        free == tight,
        "admission under a tight budget moved a voxel"
    );
}

/// An op whose blocks are serialised elsewhere (one device): it says so, and
/// the planner prices one in flight and the executor admits one at a time.
struct OneAtATime;

impl blockflow::op::BlockOp for OneAtATime {
    fn name(&self) -> &'static str {
        "one at a time"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }

    fn apply(
        &self,
        input: &Voxels,
        out: &mut Voxels,
        _at: &blockflow::op::Anchor,
    ) -> blockflow::Result<()> {
        out.assign(input)
    }

    fn max_in_flight(&self) -> Option<usize> {
        Some(1)
    }
}

#[test]
fn an_op_that_runs_one_block_at_a_time_is_planned_and_admitted_so() {
    let grid = BlockGrid::new(VOLUME, [8, 256, 256]).unwrap();
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    plan.pixels(Chain::op(OneAtATime)).unwrap();
    let assembly = plan.finish().unwrap();
    let (_, active) = blockflow::strategy::price_under_budget(
        &assembly.workflow.chain,
        &assembly.decomposition,
        &assembly.work(),
        &CostModel::default(),
        8,
        64 << 30,
    )
    .unwrap();
    assert!(active.iter().all(|set| set.in_flight == 1), "{active:?}");
    let env =
        ArrayEnvironment::for_decomposition(input(), &assembly.decomposition, [8, 64, 64]).unwrap();
    let counting = Arc::new(InFlight::default());
    let listeners: Vec<Arc<dyn EventListener>> = vec![counting.clone()];
    execute_phases(
        "one at a time",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints {
            concurrency: 8,
            ..Hints::default()
        },
        &env,
        &listeners,
        &assembly.work(),
    )
    .unwrap();
    assert_eq!(counting.most.load(Ordering::SeqCst), 1);
    assert!(env.output() == input());
}
