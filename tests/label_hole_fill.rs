// SPDX-License-Identifier: MIT

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::{
    append_fill_label_holes_2d_by_label_phase, fill_label_holes_2d_by_label_into,
};
use blockflow::strategy::{execute_phases, Hints};
use ndarray::Array3;

const VOLUME: [usize; 3] = [1, 5, 7];

fn run(input: &Array3<u32>, block: [usize; 3]) -> Array3<u32> {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::U32, grid);
    append_fill_label_holes_2d_by_label_phase(&mut builder).unwrap();
    let assembly = builder.finish().expect("label-hole fill phase");
    let env =
        ArrayEnvironment::for_decomposition(input.clone().into(), &assembly.decomposition, block)
            .unwrap();
    execute_phases(
        "label-hole fill",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .expect("a fill run");
    env.output().view::<u32>().unwrap().to_owned()
}

#[test]
fn label_hole_fill_phase_matches_whole_image_rule_across_blocks() {
    let mut labels = Array3::<u32>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for y in 0..5 {
        labels[[0, y, 0]] = 7;
        labels[[0, y, 4]] = 7;
    }
    for x in 0..=4 {
        labels[[0, 0, x]] = 7;
        labels[[0, 4, x]] = 7;
    }
    labels[[0, 1, 1]] = 3;
    labels[[0, 1, 2]] = 3;
    labels[[0, 2, 1]] = 3;
    labels[[0, 2, 2]] = 3;

    let mut expected = Array3::<u32>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    fill_label_holes_2d_by_label_into(labels.view(), expected.view_mut()).unwrap();
    assert_eq!(expected[[0, 2, 3]], 7);
    assert_eq!(expected[[0, 1, 5]], 0);

    assert_eq!(run(&labels, VOLUME), expected);
    assert_eq!(run(&labels, [1, 2, 3]), expected);
}
