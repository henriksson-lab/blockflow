// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::rows::collect_rows;
use blockflow::ops::{
    append_response_peak_table_phases, response_peak_points, Connectivity, RESPONSE_COLUMN,
};
use blockflow::points::Point;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::table::{Schema, Value};
use ndarray::Array3;

const VOLUME: [usize; 3] = [4, 4, 4];
const STREAM: &str = "response-peaks";

fn run(input: &Array3<f64>, block: [usize; 3], minimum_response: f64) -> Vec<Point> {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let (phase, schema) = append_response_peak_table_phases(
        &mut builder,
        STREAM,
        blockflow::sidecar::Lifecycle::Persistent,
        Connectivity::Faces,
        minimum_response,
    )
    .unwrap();
    assert_eq!(
        schema,
        Schema::new(vec![blockflow::table::Column::f64(RESPONSE_COLUMN)]).unwrap()
    );
    let assembly = builder.finish().expect("response peak phases");
    let env = ArrayEnvironment::for_decomposition(
        input.clone().into(),
        &assembly.decomposition,
        [4, 4, 4],
    )
    .unwrap();
    execute_phases(
        "response peaks",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .expect("a response peak run");
    collect_rows(&env, STREAM, phase.index(), VOLUME, schema)
        .unwrap()
        .into_iter()
        .map(|row| {
            let [Value::F64(response)] = row.values.as_slice() else {
                panic!("response peak rows carry one f64 response value")
            };
            Point::weighted(row.at, *response)
        })
        .collect()
}

#[test]
fn response_peak_rows_match_scalar_peak_points_across_block_seams() {
    let mut input = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        (z + y + x) as f64
    });
    input[[1, 1, 1]] = 20.0;
    input[[2, 1, 1]] = 20.0;
    input[[0, 3, 3]] = 12.0;
    input[[3, 3, 3]] = 6.0;

    let expected = response_peak_points(input.view(), Connectivity::Faces, 10.0).unwrap();
    assert_eq!(
        expected,
        vec![
            Point::weighted([0, 3, 3], 12.0),
            Point::weighted([1, 1, 1], 20.0),
        ]
    );
    assert_eq!(run(&input, VOLUME, 10.0), expected);
    assert_eq!(run(&input, [2, 4, 4], 10.0), expected);
}

#[test]
fn response_peak_rows_refuse_non_finite_thresholds() {
    let grid = BlockGrid::new(VOLUME, VOLUME).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let error = append_response_peak_table_phases(
        &mut builder,
        STREAM,
        blockflow::sidecar::Lifecycle::Persistent,
        Connectivity::Faces,
        f64::NAN,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("response peak threshold"), "{error}");
}
