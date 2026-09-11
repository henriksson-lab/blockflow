// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::{append_expand_labels_phase, expand_labels_into};
use blockflow::strategy::{execute_phases, Hints};
use ndarray::Array3;

const VOLUME: [usize; 3] = [6, 3, 3];

fn run(input: &Array3<u16>, block: [usize; 3]) -> Array3<u16> {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::U16, grid);
    append_expand_labels_phase(&mut builder, 2.0).unwrap();
    let assembly = builder.finish().expect("expand-labels phase");
    let env = ArrayEnvironment::for_decomposition(
        input.clone().into(),
        &assembly.decomposition,
        [4, 4, 4],
    )
    .unwrap();
    execute_phases(
        "expand labels",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .expect("an expand-labels run");
    env.output().view::<u16>().unwrap().to_owned()
}

#[test]
fn expand_labels_phase_uses_halo_sources_across_block_seams() {
    let mut labels = Array3::<u16>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    labels[[1, 1, 1]] = 9;
    labels[[4, 1, 1]] = 3;

    let mut expected = Array3::<u16>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    expand_labels_into(labels.view(), 2.0, expected.view_mut()).unwrap();
    assert_eq!(expected[[2, 1, 1]], 9);
    assert_eq!(expected[[3, 1, 1]], 3);
    assert_eq!(run(&labels, VOLUME), expected);
    assert_eq!(run(&labels, [3, 3, 3]), expected);
}
