// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::{append_hysteresis_threshold_phases, hysteresis_threshold_into, Connectivity};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use ndarray::Array3;

const VOLUME: [usize; 3] = [4, 4, 4];
const STREAM: &str = "hysteresis";

fn run(input: &Array3<f64>, block: [usize; 3], connectivity: Connectivity) -> Array3<bool> {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    append_hysteresis_threshold_phases(
        &mut builder,
        STREAM,
        Lifecycle::DeleteOnExit,
        0.3,
        0.8,
        connectivity,
    )
    .unwrap();
    let assembly = builder.finish().expect("hysteresis phases");
    let env = ArrayEnvironment::for_decomposition(
        input.clone().into(),
        &assembly.decomposition,
        [4, 4, 4],
    )
    .unwrap();
    execute_phases(
        "hysteresis",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .expect("a hysteresis run");
    env.output().view::<bool>().unwrap().to_owned()
}

#[test]
fn hysteresis_threshold_phases_link_weak_edges_across_block_seams() {
    let mut response = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    response[[0, 1, 1]] = 0.9;
    response[[1, 1, 1]] = 0.4;
    response[[2, 1, 1]] = 0.35;
    response[[3, 1, 1]] = 0.31;
    response[[0, 3, 3]] = 0.5;

    let mut expected = Array3::<bool>::default((VOLUME[0], VOLUME[1], VOLUME[2]));
    hysteresis_threshold_into(
        response.view(),
        0.3,
        0.8,
        Connectivity::Faces,
        expected.view_mut(),
    )
    .unwrap();

    assert!(expected[[0, 1, 1]]);
    assert!(expected[[3, 1, 1]]);
    assert!(!expected[[0, 3, 3]]);
    assert_eq!(run(&response, VOLUME, Connectivity::Faces), expected);
    assert_eq!(run(&response, [2, 4, 4], Connectivity::Faces), expected);
}

#[test]
fn hysteresis_threshold_phases_honour_wider_connectivity() {
    let mut response = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    response[[1, 1, 1]] = 0.9;
    response[[2, 2, 2]] = 0.4;

    let faces = run(&response, [2, 2, 2], Connectivity::Faces);
    let full = run(&response, [2, 2, 2], Connectivity::FacesEdgesAndCorners);
    assert!(faces[[1, 1, 1]]);
    assert!(!faces[[2, 2, 2]]);
    assert!(full[[1, 1, 1]]);
    assert!(full[[2, 2, 2]]);
}
