// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// `strategy::choose_grid` and the candidates it prices. What is held here is
// the shape of the choice rather than any figure: the candidates are per-axis
// and fewest blocks first; one worker gets the coarsest grid that fits; more
// workers get more blocks; an op that reaches far in the image plane and not
// across it is cut across it; a budget too small for a coarse grid moves the
// choice to a finer one; and the choice is the same every time.

use blockflow::assemble::PlanBuilder;
use blockflow::decomposition::CostModel;
use blockflow::dtype::Dtype;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::smooth::{Gaussian, SmoothOp};
use blockflow::strategy::{choose_grid, grid_candidates, Host, MIN_GRID_EDGE};

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

fn host(workers: usize) -> Host {
    Host {
        workers,
        budget_bytes: 64 << 30,
    }
}

fn blocks(volume: [usize; 3], block: [usize; 3]) -> usize {
    (0..3)
        .map(|axis| volume[axis].div_ceil(block[axis]))
        .product()
}

#[test]
fn the_candidates_are_per_axis_and_fewest_blocks_first() {
    let candidates = grid_candidates(VOLUME);
    assert_eq!(candidates[0], VOLUME, "the whole volume comes first");
    let counts: Vec<usize> = candidates.iter().map(|&b| blocks(VOLUME, b)).collect();
    assert!(
        counts.windows(2).all(|pair| pair[0] <= pair[1]),
        "{counts:?}"
    );
    for block in &candidates {
        for axis in 0..3 {
            assert!(
                block[axis] == VOLUME[axis] || block[axis] >= MIN_GRID_EDGE,
                "{block:?} cuts axis {axis} below the floor"
            );
        }
    }
    // A slab across the narrow axis and a column along it are both offered,
    // which no scalar edge could do.
    assert!(candidates.contains(&[8, 256, 256]));
    assert!(candidates.contains(&[64, 32, 32]));
    let unique: std::collections::BTreeSet<_> = candidates.iter().collect();
    assert_eq!(unique.len(), candidates.len(), "a candidate is repeated");
}

#[test]
fn one_worker_gets_the_coarsest_grid_and_more_workers_get_more_blocks() {
    let model = CostModel::default();
    let one = choose_grid(VOLUME, &host(1), &model, plane_blur).unwrap();
    assert_eq!(one.grid.n_blocks(), 1, "{:?}", one.grid.block());
    let eight = choose_grid(VOLUME, &host(8), &model, plane_blur).unwrap();
    assert!(eight.grid.n_blocks() >= 8, "{:?}", eight.grid.block());
    assert!(eight.makespan < one.makespan * 0.5);
}

#[test]
fn a_plane_wide_reach_is_cut_across_the_plane() {
    let chosen = choose_grid(VOLUME, &host(8), &CostModel::default(), plane_blur).unwrap();
    let block = chosen.grid.block();
    // Slabs: the planes are whole, so the blur has no halo to read twice.
    assert_eq!([block[1], block[2]], [VOLUME[1], VOLUME[2]], "{block:?}");
}

#[test]
fn a_budget_too_small_for_a_coarse_grid_moves_the_choice_finer() {
    let model = CostModel::default();
    let roomy = choose_grid(VOLUME, &host(1), &model, plane_blur).unwrap();
    let tight = Host {
        workers: 1,
        budget_bytes: 96 << 20,
    };
    let chosen = choose_grid(VOLUME, &tight, &model, plane_blur).unwrap();
    assert!(chosen.grid.n_blocks() > roomy.grid.n_blocks());
    assert!(chosen
        .considered
        .iter()
        .any(|(_, priced)| priced.as_ref().is_err_and(|why| why.contains("budget"))));
}

#[test]
fn the_choice_is_the_same_every_time() {
    let model = CostModel::default();
    let first = choose_grid(VOLUME, &host(6), &model, plane_blur).unwrap();
    for _ in 0..3 {
        let again = choose_grid(VOLUME, &host(6), &model, plane_blur).unwrap();
        assert_eq!(again.grid.block(), first.grid.block());
        assert_eq!(again.makespan.to_bits(), first.makespan.to_bits());
    }
}

#[test]
fn the_detected_host_has_a_worker_and_a_budget() {
    let host = Host::detect();
    assert!(host.workers >= 1);
    assert!(host.budget_bytes >= 1 << 30);
    assert_eq!(host.with_workers(0).workers, 1);
    assert_eq!(host.constraints().expected_concurrency, host.workers);
}
