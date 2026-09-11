// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::{append_canny_edges_phases, canny_edges_into, Connectivity};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use ndarray::Array3;

const VOLUME: [usize; 3] = [7, 7, 5];
const STREAM: &str = "canny";

fn run(input: &Array3<f64>, block: [usize; 3]) -> Array3<bool> {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    append_canny_edges_phases(
        &mut builder,
        STREAM,
        Lifecycle::DeleteOnExit,
        [0.0, 0.0, 0.0],
        3.0,
        1.0,
        4.0,
        Connectivity::Faces,
    )
    .unwrap();
    let assembly = builder.finish().expect("canny phases");
    let env = ArrayEnvironment::for_decomposition(
        input.clone().into(),
        &assembly.decomposition,
        [4, 4, 4],
    )
    .unwrap();
    execute_phases(
        "canny",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .expect("a canny run");
    env.output().view::<bool>().unwrap().to_owned()
}

#[test]
fn canny_edges_phases_match_the_named_scalar_composition() {
    let input = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        if i < 3 {
            0.0
        } else {
            10.0 + j as f64 * 0.25 + k as f64 * 0.125
        }
    });
    let mut expected = Array3::<bool>::default((VOLUME[0], VOLUME[1], VOLUME[2]));
    canny_edges_into(
        input.view(),
        [0.0, 0.0, 0.0],
        3.0,
        1.0,
        4.0,
        Connectivity::Faces,
        expected.view_mut(),
    )
    .unwrap();
    assert_eq!(run(&input, VOLUME), expected);
    assert_eq!(run(&input, [3, 7, 5]), expected);
}
