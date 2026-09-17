// SPDX-License-Identifier: MIT

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::append_filter_labels_by_size_phases;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use ndarray::Array3;

const VOLUME: [usize; 3] = [1, 4, 6];

fn run(input: &Array3<u32>, block: [usize; 3]) -> Array3<u32> {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::U32, grid);
    append_filter_labels_by_size_phases(
        &mut builder,
        "label-size-filter",
        Lifecycle::DeleteOnExit,
        2,
        Some(4),
    )
    .unwrap();
    let assembly = builder.finish().expect("label-size filter phases");
    let env =
        ArrayEnvironment::for_decomposition(input.clone().into(), &assembly.decomposition, block)
            .unwrap();
    execute_phases(
        "label-size filter",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .expect("a filter run");
    env.output().view::<u32>().unwrap().to_owned()
}

#[test]
fn label_size_filter_phases_merge_counts_before_rewriting_labels() {
    let mut labels = Array3::<u32>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    labels[[0, 0, 0]] = 1;
    labels[[0, 3, 5]] = 1;

    labels[[0, 0, 1]] = 2;

    for at in [[0, 0, 2], [0, 0, 3], [0, 1, 2], [0, 2, 2], [0, 3, 2]] {
        labels[at] = 3;
    }

    for at in [[0, 1, 4], [0, 1, 5], [0, 2, 4], [0, 2, 5]] {
        labels[at] = 4;
    }

    let mut expected = labels.clone();
    for value in &mut expected {
        if *value == 2 || *value == 3 {
            *value = 0;
        }
    }

    assert_eq!(run(&labels, VOLUME), expected);
    assert_eq!(run(&labels, [1, 2, 3]), expected);
}
