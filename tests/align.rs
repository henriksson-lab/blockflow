// SPDX-License-Identifier: MIT

use blockflow::ops::align::{run, TransformModel, VolumeFitParams};
use blockflow::voxels::Voxels;
use ndarray::Array3;

const SHAPE: [usize; 3] = [9, 7, 5];

fn gaussian(center: [f64; 3]) -> Voxels {
    Array3::from_shape_fn((SHAPE[0], SHAPE[1], SHAPE[2]), |(x, y, z)| {
        let dx = x as f64 - center[0];
        let dy = y as f64 - center[1];
        let dz = z as f64 - center[2];
        (-(dx * dx + dy * dy + dz * dz) / 3.0).exp()
    })
    .into()
}

#[test]
fn shifted_smooth_volume_fits_translation() {
    let fixed = gaussian([4.0, 3.0, 2.0]);
    let moving = gaussian([5.0, 2.0, 2.0]);
    let fitted = run(&VolumeFitParams::new(), &fixed, &moving, 4).expect("fit computes");

    assert_eq!(fitted.model, TransformModel::Translation);
    assert!(
        (fitted.params[0] - 1.0).abs() <= 0.25,
        "{:?}",
        fitted.params
    );
    assert!(
        (fitted.params[1] + 1.0).abs() <= 0.25,
        "{:?}",
        fitted.params
    );
    assert!(fitted.params[2].abs() <= 0.25, "{:?}", fitted.params);
}
