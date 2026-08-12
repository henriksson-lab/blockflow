// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Run a block-ops workflow and export its order log as JSON, for
// `tools/animate_block_progress.py`.
//
// The point is comparison: the *same* workflow and the *same* decomposition
// run under two schedules — block-major (depth-first fusion: push one block as
// far through the phases as its dependencies allow) and phase-major
// (breadth-first: finish every block of a phase before starting the next).
// They compute identical results and produce completely different pictures,
// which is the thing the animation is for.
//
//     cargo run --release --bin block_progress_export -- [outdir] [blocks-per-side]
//
// Writes `<outdir>/block_major.json` and `<outdir>/phase_major.json`, defaults
// to the current directory and a 6 x 6 grid. Runs against the accounting
// environment, so no array is allocated and the grid can be as large as you
// like.

use blockflow::decomposition::{groups_for, summarise_slots, Decomposition, PhaseDecomposition};
use blockflow::env::AccountingEnvironment;
use blockflow::export::{write_order_log_json, ExportMeta};
use blockflow::geometry::BlockGrid;
use blockflow::listener::{EventListener, LatestOpPerChunk};
use blockflow::probes::IdentityOp;
use blockflow::strategy::{execute_observed, Hints, SchedulePriority, Workflow};
use blockflow::{Chain, Dtype, Result};

use std::sync::Arc;

fn main() -> Result<()> {
    let outdir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let outdir = std::path::PathBuf::from(outdir);
    std::fs::create_dir_all(&outdir).ok();

    // A `side` x `side` grid of blocks in the two axes the animation projects
    // onto, one block deep in the third: legible as a grid, and 36 blocks is
    // enough for the two schedules to look nothing alike.
    let side: usize = std::env::args()
        .nth(2)
        .map(|arg| arg.parse().expect("blocks-per-side must be a number"))
        .unwrap_or(6);
    let shape = [64, side * 64, side * 64];
    let chain = Chain::sequence(vec![
        Chain::op(IdentityOp::new("median", [0, 2, 2]).with_cost(1.0)),
        Chain::op(IdentityOp::new("background", [0, 6, 6]).with_cost(2.0)),
        Chain::op(IdentityOp::new("threshold", [0, 0, 0]).with_cost(0.5)),
    ]);
    let ops: Vec<(usize, String)> = chain
        .slots()
        .iter()
        .enumerate()
        .map(|(slot, sub)| (slot, sub.display_name()))
        .collect();
    let workflow = Workflow::new(chain, shape, Dtype::U16);

    // Three phases, stated rather than planned: the comparison is about the
    // *schedule*, so the decomposition must be identical in both runs.
    let slots = workflow.chain.slots();
    let phases: Vec<PhaseDecomposition> = groups_for(0b11, slots.len())
        .iter()
        .map(|group| {
            let (reach, _, names, _) = summarise_slots(&slots, group, shape);
            let grid = BlockGrid::along(shape, &[1, 2], 64).unwrap();
            PhaseDecomposition::derive(group.clone(), names, reach, reach, grid)
        })
        .collect();
    let decomposition = Decomposition {
        volume: shape,
        dtype: workflow.dtype,
        phases,
        chain_reach: workflow.chain.reach3(&shape),
    };
    decomposition.check()?;

    for (name, priority) in [
        ("block_major", SchedulePriority::BlockMajor),
        ("phase_major", SchedulePriority::PhaseMajor),
    ] {
        // A live progress view attached to the same run, to show that the two
        // listeners see one run and to print a final summary from 8 bytes per
        // block instead of from the whole history.
        let latest = Arc::new(LatestOpPerChunk::new());
        let listeners: Vec<Arc<dyn EventListener>> = vec![latest.clone()];

        let hints = Hints {
            priority,
            concurrency: 4,
            ..Hints::default()
        };
        let env = AccountingEnvironment::new(shape, [32, 64, 64], 2);
        let stats = execute_observed(name, &workflow, &decomposition, &hints, &env, &listeners)?;

        let path = outdir.join(format!("{name}.json"));
        let meta = ExportMeta::new(name, shape, decomposition.n_phases()).with_ops(ops.clone());
        write_order_log_json(&stats.log, &meta, &path)?;

        let last = latest.last_op_per_block();
        println!(
            "{name:<12} {:>5} tasks  {:>6} events  {:>4} blocks  faults {}  -> {}",
            stats.tasks,
            stats.log.len(),
            last.len(),
            stats.listener_faults,
            path.display()
        );
    }
    Ok(())
}
