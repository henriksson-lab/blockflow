// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Orientation against the reference's own `orientation.py`, on a 4 x 3 x 2
//! array and eight points, for twelve orientations in both directions. The
//! fixture is written by `scripts/orientation_fixtures.py`, already converted
//! to `[z, y, x]` order.

use atlas_regions::Orientation;
use blockflow::voxels::Voxels;
use ndarray::Array3;
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/orientation.json")).expect("the fixture")
}

fn triple<T: Copy>(value: &Value, f: impl Fn(&Value) -> T) -> [T; 3] {
    let entries = value.as_array().expect("an array");
    [f(&entries[0]), f(&entries[1]), f(&entries[2])]
}

fn shape(value: &Value) -> [usize; 3] {
    triple(value, |v| v.as_u64().unwrap() as usize)
}

fn volume(shape_value: &Value, data: &Value) -> Voxels {
    let shape = shape(shape_value);
    let values = data
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    Voxels::I64(Array3::from_shape_vec(shape, values).unwrap())
}

fn points(value: &Value) -> Vec<[f64; 3]> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|p| triple(p, |v| v.as_f64().unwrap()))
        .collect()
}

#[test]
fn every_operation_agrees_with_the_reference() {
    let fixture = fixture();
    let data_shape = shape(&fixture["data_shape_zyx"]);
    let input_points = points(&fixture["points_zyx"]);
    let resolution = triple(&fixture["resolution_zyx"], |v| v.as_f64().unwrap());
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 12);
    for case in cases {
        let tuple = triple(&case["orientation_xyz"], |v| v.as_i64().unwrap() as i32);
        let o = Orientation::from_xyz(tuple).unwrap();
        assert_eq!(o.to_xyz(), tuple);

        let forward = &case["forward"];
        let input = volume(&forward["input_shape_zyx"], &forward["input_zyx"]);
        let expected = volume(&forward["output_shape_zyx"], &forward["output_zyx"]);
        assert_eq!(o.orient_voxels(&input), expected, "volume {tuple:?}");
        assert_eq!(o.orient_shape(data_shape), shape(&forward["shape_zyx"]));
        let expected_resolution = triple(&forward["resolution_zyx"], |v| v.as_f64().unwrap());
        assert_eq!(o.orient_resolution(resolution), expected_resolution);
        let expected_points = points(&forward["points_zyx"]);
        let got = o.orient_points(&input_points, data_shape);
        for (g, e) in got.iter().zip(&expected_points) {
            for a in 0..3 {
                assert!(
                    (g[a] - e[a]).abs() < 1e-12,
                    "points {tuple:?}: {g:?} vs {e:?}"
                );
            }
        }

        let inverse = &case["inverse"];
        let input = volume(&inverse["input_shape_zyx"], &inverse["input_zyx"]);
        let expected = volume(&inverse["output_shape_zyx"], &inverse["output_zyx"]);
        assert_eq!(
            o.unorient_voxels(&input),
            expected,
            "inverse volume {tuple:?}"
        );
        assert_eq!(
            o.inverse().orient_shape(data_shape),
            shape(&inverse["shape_zyx"])
        );
        let expected_resolution = triple(&inverse["resolution_zyx"], |v| v.as_f64().unwrap());
        assert_eq!(o.unorient_resolution(resolution), expected_resolution);
        // The reference's inverse takes the shape of the unoriented data.
        let expected_points = points(&inverse["points_zyx"]);
        let got = o.unorient_points(&input_points, data_shape);
        for (g, e) in got.iter().zip(&expected_points) {
            for a in 0..3 {
                assert!(
                    (g[a] - e[a]).abs() < 1e-12,
                    "inverse points {tuple:?}: {g:?} vs {e:?}"
                );
            }
        }
    }
}
