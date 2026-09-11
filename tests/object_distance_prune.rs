// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::{append_object_distance_prune_phases, prune_by_object_distance_into};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use ndarray::Array3;

const VOLUME: [usize; 3] = [6, 2, 2];
const STREAM: &str = "object-distance";

fn run(input: &Array3<u16>, block: [usize; 3]) -> Array3<u16> {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::U16, grid);
    append_object_distance_prune_phases(&mut builder, STREAM, Lifecycle::DeleteOnExit, 2.0)
        .unwrap();
    let assembly = builder.finish().expect("object-distance phases");
    let env = ArrayEnvironment::for_decomposition(
        input.clone().into(),
        &assembly.decomposition,
        [4, 4, 4],
    )
    .unwrap();
    execute_phases(
        "object distance prune",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .expect("an object-distance run");
    env.output().view::<u16>().unwrap().to_owned()
}

#[test]
fn object_distance_prune_phases_match_scalar_rule_across_seams() {
    let mut labels = Array3::<u16>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    labels[[1, 0, 0]] = 4;
    labels[[3, 0, 0]] = 7;
    labels[[5, 1, 1]] = 9;

    let mut expected = Array3::<u16>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    prune_by_object_distance_into(labels.view(), 2.0, expected.view_mut()).unwrap();
    assert_eq!(expected[[1, 0, 0]], 4);
    assert_eq!(expected[[3, 0, 0]], 0);
    assert_eq!(expected[[5, 1, 1]], 9);
    assert_eq!(run(&labels, VOLUME), expected);
    assert_eq!(run(&labels, [3, 2, 2]), expected);
}
