// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::{append_equalize_adapthist_phases, equalize_adapthist_into};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use ndarray::{Array3, Zip};

const VOLUME: [usize; 3] = [5, 4, 3];
const STREAM: &str = "clahe";

fn run(input: &Array3<f64>, block: [usize; 3]) -> Array3<f64> {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    append_equalize_adapthist_phases(
        &mut builder,
        STREAM,
        Lifecycle::DeleteOnExit,
        [2, 2, 2],
        4,
        3.0,
    )
    .unwrap();
    let assembly = builder.finish().expect("adaptive equalization phases");
    let env = ArrayEnvironment::for_decomposition(
        input.clone().into(),
        &assembly.decomposition,
        [4, 4, 4],
    )
    .unwrap();
    execute_phases(
        "adaptive histogram equalization",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .expect("a clahe run");
    env.output().view::<f64>().unwrap().to_owned()
}

#[test]
fn equalize_adapthist_phases_match_scalar_rule_across_decompositions() {
    let input = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        (i * 7 + j * 3 + k) as f64
    });
    let mut expected = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    equalize_adapthist_into(input.view(), [2, 2, 2], 4, 3.0, expected.view_mut()).unwrap();

    for got in [run(&input, VOLUME), run(&input, [3, 2, 3])] {
        Zip::from(&got)
            .and(&expected)
            .for_each(|&actual, &want| assert!((actual - want).abs() <= f64::EPSILON));
    }
}
