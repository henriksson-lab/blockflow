// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A phase whose blocks hold more than the plan charges them must still run
// under the memory cap. Its own test binary, because the assertion is on the
// process's resident set and another test running beside it would move it.

use std::time::Duration;

use ndarray::Array3;

use blockflow::assemble::PlanBuilder;
use blockflow::budget::{release_freed_heap, resident_bytes, MemoryBudget, ResidentSampler};
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::strategy::{
    active_bytes_per_block, execute_phases, expected_bytes_per_block, Hints,
};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [32, 512, 512];
const BLOCK: [usize; 3] = [1, 512, 512];

/// A copy that allocates `hidden` bytes per block that nothing declares,
/// touches every page of them and holds them long enough to be sampled.
struct Undeclared {
    hidden: usize,
}

impl BlockOp for Undeclared {
    fn name(&self) -> &'static str {
        "undeclared scratch"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> blockflow::Result<()> {
        let mut scratch = vec![0u8; self.hidden];
        for page in (0..self.hidden).step_by(4096) {
            scratch[page] = 1;
        }
        std::hint::black_box(&scratch);
        std::thread::sleep(Duration::from_millis(60));
        drop(scratch);
        out.assign(input)
    }
}

#[test]
fn a_phase_holding_more_than_it_declares_still_runs_under_the_cap() {
    let grid = BlockGrid::new(VOLUME, BLOCK).unwrap();
    let plan_with = |hidden: usize| {
        let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid.clone());
        plan.pixels(Chain::op(Undeclared { hidden })).unwrap();
        plan.finish().unwrap()
    };
    let probe = plan_with(0);
    let expected = expected_bytes_per_block(&probe.workflow.chain, &probe.decomposition)[0];
    let charged = active_bytes_per_block(&probe.workflow.chain, &probe.decomposition)[0];
    // A block holds three and a half times what the plan expects: past the
    // margin the plan's charge carries for op scratch.
    let hidden = (expected * 5 / 2) as usize;
    let held = expected + hidden as u64;
    assert!(
        held > charged * 3 / 2,
        "{held} against a charge of {charged}"
    );
    let assembly = plan_with(hidden);

    let input = Voxels::from(Array3::from_shape_fn(
        (VOLUME[0], VOLUME[1], VOLUME[2]),
        |(z, y, x)| ((z * 31 + y * 7 + x * 13) % 101) as f64,
    ));
    let env =
        ArrayEnvironment::for_decomposition(input.clone(), &assembly.decomposition, BLOCK).unwrap();
    let run = |memory: Option<MemoryBudget>| {
        execute_phases(
            "undeclared scratch",
            &assembly.workflow,
            &assembly.decomposition,
            &Hints {
                concurrency: 32,
                memory,
                ..Hints::default()
            },
            &env,
            &[],
            &assembly.work(),
        )
        .unwrap();
    };
    // Once without a cap, so that the output's pages are resident before the
    // measured run rather than counted against its cap.
    run(None);
    assert!(env.output() == input);
    release_freed_heap();
    // A cap of what the process holds now and room for eight blocks at the
    // plan's charge: at the plan's charge, thirty-two workers would run eight,
    // and eight blocks holding what these hold is far over it.
    let before = resident_bytes().expect("the resident set is readable here");
    let cap = before + 8 * charged;
    assert!(8 * held > (cap - before) * 3 / 2);
    let sampler = ResidentSampler::start().unwrap();
    sampler.reset(resident_bytes().unwrap());
    run(Some(MemoryBudget::new(cap)));
    let peak = sampler.peak();
    assert!(
        peak <= cap,
        "the resident set peaked at {} MiB against a cap of {} MiB ({} MiB before the run, a \
         block holding {} MiB and charged {} MiB)",
        peak >> 20,
        cap >> 20,
        before >> 20,
        held >> 20,
        charged >> 20
    );
    assert!(env.output() == input);
}
