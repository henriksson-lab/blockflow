// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::{append_equalize_histogram_phases, equalize_histogram};
use blockflow::strategy::{execute_phases, Hints};
use ndarray::Array3;

const VOLUME: [usize; 3] = [4, 4, 4];
const STREAM: &str = "histogram-values";

fn run(input: &Array3<f64>, block: [usize; 3], bins: usize) -> Array3<f64> {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    append_equalize_histogram_phases(&mut builder, STREAM, bins).unwrap();
    let assembly = builder.finish().expect("equalization phases");
    let env = ArrayEnvironment::for_decomposition(
        input.clone().into(),
        &assembly.decomposition,
        [4, 4, 4],
    )
    .unwrap();
    execute_phases(
        "global equalize",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .expect("an equalization run");
    env.output().view::<f64>().unwrap().to_owned()
}

#[test]
fn global_histogram_equalization_is_invariant_to_the_blocking() {
    let input = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        ((z * 3 + y * 5 + x * 7) % 11) as f64
    });
    let expected = Array3::from_shape_vec(
        (VOLUME[0], VOLUME[1], VOLUME[2]),
        equalize_histogram(input.as_slice().unwrap(), 8).unwrap(),
    )
    .unwrap();
    let whole = run(&input, VOLUME, 8);
    let split = run(&input, [2, 4, 4], 8);
    assert_eq!(whole, expected);
    assert_eq!(split, expected);
}

#[test]
fn global_histogram_equalization_preserves_non_finite_values() {
    let mut input = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 7.0);
    input[[1, 2, 3]] = f64::NAN;
    input[[3, 2, 1]] = f64::INFINITY;
    let whole = run(&input, VOLUME, 4);
    let split = run(&input, [1, 4, 4], 4);
    assert_eq!(whole[[0, 0, 0]], 0.0);
    assert_eq!(split[[0, 0, 0]], 0.0);
    assert!(whole[[1, 2, 3]].is_nan());
    assert!(split[[1, 2, 3]].is_nan());
    assert!(whole[[3, 2, 1]].is_infinite());
    assert!(split[[3, 2, 1]].is_infinite());
}
