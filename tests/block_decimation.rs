#![cfg(feature = "zarr")]

use blockflow::{voxels::Voxels, zarr_env::downsample_block};
use ndarray::Array3;

#[test]
fn block_means_preserve_axis_order_anisotropy_and_drop_partial_windows() {
    let input = Array3::from_shape_fn((7, 10, 11), |(z, y, x)| (100 * z + 10 * y + x) as f32);
    let voxels: Voxels = input.into();
    for factor in [[2, 3, 4], [1, 2, 2], [3, 1, 2]] {
        let result = downsample_block::<f32>(&voxels, factor).unwrap();
        let shape = [7 / factor[0], 10 / factor[1], 11 / factor[2]];
        assert_eq!(result.shape(), &shape);
        for ((z, y, x), &value) in result.indexed_iter() {
            let expected = 100. * (z * factor[0]) as f64
                + 10. * (y * factor[1]) as f64
                + (x * factor[2]) as f64
                + 100. * (factor[0] - 1) as f64 / 2.
                + 10. * (factor[1] - 1) as f64 / 2.
                + (factor[2] - 1) as f64 / 2.;
            assert_eq!(value, expected as f32);
        }
    }
}

#[test]
fn block_decimation_keeps_strict_nonfinite_and_bool_rules_and_rejects_invalid_requests() {
    let mut image = Array3::from_elem((4, 4, 4), 3f32);
    image[[1, 1, 1]] = f32::NAN;
    let voxels: Voxels = image.into();
    let result = downsample_block::<f32>(&voxels, [2; 3]).unwrap();
    assert!(result[[0, 0, 0]].is_nan());
    assert_eq!(result[[1, 1, 1]], 3.);
    assert!(downsample_block::<f32>(&voxels, [0, 1, 1]).is_err());
    assert!(downsample_block::<f32>(&voxels, [5, 1, 1]).is_err());
    assert!(downsample_block::<f64>(&voxels, [2; 3]).is_err());
    let binary = Array3::from_shape_fn((4, 4, 4), |(z, y, x)| {
        z % 2 == 0 && y % 2 == 0 && x % 2 == 0
    });
    let result = downsample_block::<bool>(&binary.into(), [2; 3]).unwrap();
    assert!(result.iter().all(|v| *v));
}
