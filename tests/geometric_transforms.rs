// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::{
    affine_transform_into, append_warp_phase, log_polar_transform_into, polar_transform_into,
    projective_transform_into, remap_into, rotate_into, TransformBoundary, TransformInterpolation,
    WarpOp,
};
use blockflow::strategy::{execute, Hints};
use blockflow::Voxels;
use ndarray::Array3;

const IDENTITY_3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const IDENTITY_4: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn image(shape: [usize; 3]) -> Array3<u16> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        (100 * i + 10 * j + k) as u16
    })
}

#[test]
fn affine_transform_samples_inverse_coordinates() {
    let input = image([3, 3, 2]);
    let mut out = Array3::<u16>::zeros((3, 3, 2));
    affine_transform_into(
        input.view(),
        IDENTITY_3,
        [1.0, 0.0, 0.0],
        TransformInterpolation::Nearest,
        TransformBoundary::Constant(999.0),
        out.view_mut(),
    )
    .unwrap();

    assert_eq!(out[[0, 1, 1]], input[[1, 1, 1]]);
    assert_eq!(out[[1, 2, 0]], input[[2, 2, 0]]);
    assert_eq!(out[[2, 0, 0]], 999);
}

#[test]
fn projective_identity_matches_input() {
    let input = image([3, 4, 2]);
    let mut out = Array3::<u16>::zeros((3, 4, 2));
    projective_transform_into(
        input.view(),
        IDENTITY_4,
        TransformInterpolation::Nearest,
        TransformBoundary::Constant(0.0),
        out.view_mut(),
    )
    .unwrap();

    assert_eq!(out, input);
}

#[test]
fn remap_uses_explicit_coordinate_fields() {
    let input = image([3, 3, 2]);
    let shape = (2, 2, 1);
    let x = Array3::from_shape_vec(shape, vec![0.0, 1.0, 2.0, 7.0]).unwrap();
    let y = Array3::from_shape_vec(shape, vec![0.0, 1.0, 2.0, 0.0]).unwrap();
    let z = Array3::from_elem(shape, 0.0);
    let mut out = Array3::<u16>::zeros(shape);

    remap_into(
        input.view(),
        [x.view(), y.view(), z.view()],
        TransformInterpolation::Nearest,
        TransformBoundary::Constant(5.0),
        out.view_mut(),
    )
    .unwrap();

    assert_eq!(out[[0, 0, 0]], input[[0, 0, 0]]);
    assert_eq!(out[[0, 1, 0]], input[[1, 1, 0]]);
    assert_eq!(out[[1, 0, 0]], input[[2, 2, 0]]);
    assert_eq!(out[[1, 1, 0]], 5);
}

#[test]
fn rotate_uses_centered_inverse_rotation() {
    let input = image([3, 3, 1]);
    let mut out = Array3::<u16>::zeros((3, 3, 1));
    rotate_into(
        input.view(),
        [1.0, 1.0],
        std::f64::consts::FRAC_PI_2,
        TransformInterpolation::Nearest,
        TransformBoundary::Constant(0.0),
        out.view_mut(),
    )
    .unwrap();

    assert_eq!(out[[0, 1, 0]], input[[1, 2, 0]]);
    assert_eq!(out[[1, 1, 0]], input[[1, 1, 0]]);
    assert_eq!(out[[2, 1, 0]], input[[1, 0, 0]]);
}

#[test]
fn polar_and_log_polar_share_radial_convention() {
    let input = image([4, 4, 1]);
    let mut polar = Array3::<u16>::zeros((3, 2, 1));
    polar_transform_into(
        input.view(),
        [1.0, 1.0],
        1.0,
        1.0,
        TransformInterpolation::Nearest,
        TransformBoundary::Constant(0.0),
        polar.view_mut(),
    )
    .unwrap();
    assert_eq!(polar[[0, 0, 0]], input[[1, 1, 0]]);
    assert_eq!(polar[[1, 0, 0]], input[[2, 1, 0]]);

    let mut log_polar = Array3::<u16>::zeros((2, 1, 1));
    log_polar_transform_into(
        input.view(),
        [1.0, 1.0],
        1.0,
        1.0,
        TransformInterpolation::Nearest,
        TransformBoundary::Constant(0.0),
        log_polar.view_mut(),
    )
    .unwrap();
    assert_eq!(log_polar[[0, 0, 0]], input[[1, 1, 0]]);
    assert_eq!(log_polar[[1, 0, 0]], input[[3, 1, 0]]);
}

#[test]
fn linear_interpolation_is_rejected_for_bool_masks() {
    let input = Array3::<bool>::from_elem((2, 2, 1), true);
    let mut out = Array3::<bool>::from_elem((2, 2, 1), false);
    let error = affine_transform_into(
        input.view(),
        IDENTITY_3,
        [0.0, 0.0, 0.0],
        TransformInterpolation::Linear,
        TransformBoundary::Constant(0.0),
        out.view_mut(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("does not support Bool"));
}

#[test]
fn warp_op_matches_direct_transform_when_decomposed() {
    let volume = [4, 4, 2];
    let output = [3, 4, 2];
    let input = image(volume);
    let mut expected = Array3::<u16>::zeros((output[0], output[1], output[2]));
    affine_transform_into(
        input.view(),
        IDENTITY_3,
        [1.0, 0.0, 0.0],
        TransformInterpolation::Nearest,
        TransformBoundary::Constant(0.0),
        expected.view_mut(),
    )
    .unwrap();

    let mut builder = PlanBuilder::new(volume, Dtype::U16, BlockGrid::new(volume, volume).unwrap());
    let output_grid = BlockGrid::new(output, [2, 2, 1]).unwrap();
    append_warp_phase(
        &mut builder,
        WarpOp::affine(
            "warp",
            IDENTITY_3,
            [1.0, 0.0, 0.0],
            output,
            TransformInterpolation::Nearest,
            TransformBoundary::Constant(0.0),
        )
        .unwrap(),
        volume,
        output_grid,
    )
    .unwrap();
    let assembly = builder.finish().unwrap();
    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(input),
        &assembly.decomposition,
        [2, 2, 1],
    )
    .unwrap();
    execute(
        "warp",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();

    assert_eq!(env.output().view::<u16>().unwrap(), &expected);
}
