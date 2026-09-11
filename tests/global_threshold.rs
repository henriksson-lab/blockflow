// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::{
    append_global_threshold_phases, GlobalThreshold, GlobalThresholdOutput,
    GlobalThresholdSelection, ThresholdTest,
};
use blockflow::strategy::{execute_phases, Hints};
use blockflow::Voxels;
use ndarray::Array3;

const VOLUME: [usize; 3] = [4, 4, 4];
const STREAM: &str = "threshold-values";

fn ramp() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        (z * 16 + y * 4 + x) as f64
    })
}

fn run(
    input: &Array3<f64>,
    block: [usize; 3],
    selection: GlobalThresholdSelection,
    output: GlobalThresholdOutput,
) -> Voxels {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    append_global_threshold_phases(&mut builder, STREAM, selection, output).unwrap();
    let assembly = builder.finish().expect("threshold phases");
    let env = ArrayEnvironment::for_decomposition(
        input.clone().into(),
        &assembly.decomposition,
        [4, 4, 4],
    )
    .unwrap();
    execute_phases(
        "global threshold",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .expect("a threshold run");
    env.output()
}

#[test]
fn global_mean_threshold_mask_is_invariant_to_the_blocking() {
    let input = ramp();
    let selection = GlobalThresholdSelection::single(GlobalThreshold::Mean);
    let output = GlobalThresholdOutput::Mask {
        test: ThresholdTest::Above,
    };
    let whole = run(&input, VOLUME, selection, output);
    let split = run(&input, [2, 4, 4], selection, output);
    let expected = input.mapv(|value| value > 31.5);
    assert_eq!(whole.view::<bool>().unwrap(), expected.view());
    assert_eq!(split.view::<bool>().unwrap(), expected.view());
}

#[test]
fn global_multi_otsu_classes_are_invariant_to_the_blocking() {
    let input = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, _, _)| match z {
        0 | 1 => 0.0,
        2 => 10.0,
        _ => 20.0,
    });
    let selection = GlobalThresholdSelection::multi_otsu(3, 3).unwrap();
    let whole = run(&input, VOLUME, selection, GlobalThresholdOutput::Classes);
    let split = run(&input, [1, 4, 4], selection, GlobalThresholdOutput::Classes);
    let expected = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, _, _)| match z {
        0 | 1 => 0,
        2 => 1,
        _ => 2,
    });
    assert_eq!(whole.view::<u32>().unwrap(), expected.view());
    assert_eq!(split.view::<u32>().unwrap(), expected.view());
}
