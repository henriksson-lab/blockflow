// SPDX-License-Identifier: MIT
// Original work for this crate.

use blockflow::pyramid::PyramidRecipe;
use blockflow::voxels::Voxels;
use blockflow_atlas_registration::compute::{
    elastix_setting_support, params_from_elastix_text, plan, pyramid_levels_from_recipe,
    require_supported_elastix_setting, resident, resident_pyramid, resident_pyramid_levels,
    resident_pyramid_with_recipe, run, run_with_diagnostics, run_with_transform_text,
    run_with_workers, ControlGrid, ElastixSettingStatus, FitConvergence, Interpolator, Metric,
    PyramidSchedule, PyramidSmoothing, Sampling, SpatialFrame, TransformModel, VolumeFitParams,
    VolumePyramidLevel,
};
use blockflow_atlas_registration::{ElastixTransform, ImageGeometry, RegisterParams};
use ndarray::Array3;

const SHAPE: [usize; 3] = [9, 7, 5];
const MEDIUM: [usize; 3] = [19, 17, 13];

fn point_image(at: [usize; 3]) -> Voxels {
    let mut image = Array3::from_elem((SHAPE[0], SHAPE[1], SHAPE[2]), 0.0);
    image[[at[0], at[1], at[2]]] = 10.0;
    image.into()
}

fn gaussian(center: [f64; 3]) -> Voxels {
    Array3::from_shape_fn((SHAPE[0], SHAPE[1], SHAPE[2]), |(x, y, z)| {
        let dx = x as f64 - center[0];
        let dy = y as f64 - center[1];
        let dz = z as f64 - center[2];
        (-(dx * dx + dy * dy + dz * dz) / 3.0).exp()
    })
    .into()
}

fn scaled_gaussian(center: [f64; 3], scale: f64) -> Voxels {
    let mut image = gaussian(center).view::<f64>().unwrap().to_owned();
    image.mapv_inplace(|value| value * scale);
    image.into()
}

fn asymmetric_smooth(center: [f64; 3]) -> Voxels {
    Array3::from_shape_fn((SHAPE[0], SHAPE[1], SHAPE[2]), |(x, y, z)| {
        let dx = x as f64 - center[0];
        let dy = y as f64 - center[1];
        let dz = z as f64 - center[2];
        (-(dx * dx / 2.0 + dy * dy / 4.0 + dz * dz / 6.0)).exp()
            + 0.35
                * (-((x as f64 - 6.4).powi(2) / 3.0
                    + (y as f64 - 1.8).powi(2) / 2.5
                    + (z as f64 - 3.2).powi(2) / 2.0))
                    .exp()
    })
    .into()
}

fn gaussian_in(shape: [usize; 3], center: [f64; 3]) -> Voxels {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(x, y, z)| {
        let dx = x as f64 - center[0];
        let dy = y as f64 - center[1];
        let dz = z as f64 - center[2];
        (-(dx * dx / 4.0 + dy * dy / 5.0 + dz * dz / 6.0)).exp()
            + 0.42
                * (-((x as f64 - 13.2).powi(2) / 7.0
                    + (y as f64 - 4.6).powi(2) / 4.0
                    + (z as f64 - 8.1).powi(2) / 5.0))
                    .exp()
            + 0.18
                * (-((x as f64 - 5.0).powi(2) / 3.0
                    + (y as f64 - 12.0).powi(2) / 6.0
                    + (z as f64 - 3.0).powi(2) / 2.0))
                    .exp()
    })
    .into()
}

fn point_image_in(shape: [usize; 3], at: [usize; 3]) -> Voxels {
    let mut image = Array3::from_elem((shape[0], shape[1], shape[2]), 0.0);
    image[[at[0], at[1], at[2]]] = 10.0;
    image.into()
}

fn affine_medium_pair(
    matrix: [[f64; 3]; 3],
    centre: [f64; 3],
    shift: [f64; 3],
) -> (Voxels, Voxels, [f64; 3]) {
    let fixed = gaussian_in(MEDIUM, centre);
    let fixed_array = fixed.view::<f64>().unwrap().to_owned();
    let inverse = invert3(matrix);
    let translated_centre = mat_vec3(matrix, centre);
    let translation = [
        centre[0] + shift[0] - translated_centre[0],
        centre[1] + shift[1] - translated_centre[1],
        centre[2] + shift[2] - translated_centre[2],
    ];
    let moving = Array3::from_shape_fn((MEDIUM[0], MEDIUM[1], MEDIUM[2]), |(x, y, z)| {
        let shifted = [
            x as f64 - translation[0],
            y as f64 - translation[1],
            z as f64 - translation[2],
        ];
        trilinear_sample(&fixed_array, mat_vec3(inverse, shifted)).unwrap_or(0.0)
    });
    (fixed, moving.into(), translation)
}

fn apply_affine(matrix: [[f64; 3]; 3], translation: [f64; 3], point: [f64; 3]) -> [f64; 3] {
    let mapped = mat_vec3(matrix, point);
    [
        mapped[0] + translation[0],
        mapped[1] + translation[1],
        mapped[2] + translation[2],
    ]
}

fn mat_vec3(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn invert3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let m = matrix;
    let determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    assert!(determinant.abs() > f64::EPSILON);
    let mut inverse = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let (r0, r1) = ((column + 1) % 3, (column + 2) % 3);
            let (c0, c1) = ((row + 1) % 3, (row + 2) % 3);
            inverse[row][column] = (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / determinant;
        }
    }
    inverse
}

fn trilinear_sample(image: &Array3<f64>, point: [f64; 3]) -> Option<f64> {
    let shape = image.shape();
    let x0 = point[0].floor();
    let y0 = point[1].floor();
    let z0 = point[2].floor();
    if x0 < 0.0
        || y0 < 0.0
        || z0 < 0.0
        || x0 as usize + 1 >= shape[0]
        || y0 as usize + 1 >= shape[1]
        || z0 as usize + 1 >= shape[2]
    {
        return None;
    }
    let x = x0 as usize;
    let y = y0 as usize;
    let z = z0 as usize;
    let dx = point[0] - x0;
    let dy = point[1] - y0;
    let dz = point[2] - z0;
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let c00 = lerp(image[[x, y, z]], image[[x + 1, y, z]], dx);
    let c10 = lerp(image[[x, y + 1, z]], image[[x + 1, y + 1, z]], dx);
    let c01 = lerp(image[[x, y, z + 1]], image[[x + 1, y, z + 1]], dx);
    let c11 = lerp(image[[x, y + 1, z + 1]], image[[x + 1, y + 1, z + 1]], dx);
    Some(lerp(lerp(c00, c10, dy), lerp(c01, c11, dy), dz))
}

fn error_text(result: blockflow::error::Result<impl std::fmt::Debug>) -> String {
    result.unwrap_err().to_string()
}

fn assert_point_close(got: [f64; 3], want: [f64; 3], tolerance: f64) {
    for (got, want) in got.into_iter().zip(want) {
        assert!(
            (got - want).abs() <= tolerance,
            "{got} vs {want} with tolerance {tolerance}"
        );
    }
}

fn assert_generated_text_roundtrips(text: &str, transform: &ElastixTransform, probes: &[[f64; 3]]) {
    let params = RegisterParams::default();
    let parsed = ElastixTransform::from_texts(&[text]).expect("generated transform parses");
    let root = std::env::temp_dir().join(format!(
        "atlas-registration-generated-transform-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("scratch directory");
    let path = root.join("TransformParameters.0.txt");
    std::fs::write(&path, text).expect("transform file");
    let read = ElastixTransform::read(&path).expect("generated transform reads from disk");
    let _ = std::fs::remove_dir_all(&root);
    for probe in probes {
        assert_eq!(
            parsed.apply(*probe, &params),
            transform.apply(*probe, &params)
        );
        assert_eq!(
            read.apply(*probe, &params),
            transform.apply(*probe, &params)
        );
    }
}

fn assert_voxels_same(left: &Voxels, right: &Voxels) {
    assert_eq!(left.dtype(), right.dtype());
    assert_eq!(left.shape(), right.shape());
    for (left, right) in left.widened().iter().zip(right.widened()) {
        assert_eq!(left, &right);
    }
}

fn assert_transform_text_error_contains(text: String, needles: &[&str]) {
    let error = ElastixTransform::from_texts(&[&text])
        .expect_err("malformed generated transform must be refused")
        .to_string();
    for needle in needles {
        assert!(error.contains(needle), "{error}");
    }
}

#[test]
fn moment_translation_moves_the_fixed_centre_to_the_moving_centre() {
    let fixed = point_image([2, 3, 1]);
    let moving = point_image([6, 1, 4]);

    let transform =
        run(&VolumeFitParams::default(), &fixed, &moving, 4).expect("registration computes");

    assert_eq!(
        transform.apply([2.0, 3.0, 1.0], &RegisterParams::default()),
        [6.0, 1.0, 4.0]
    );
}

#[test]
fn empty_images_are_refused_by_name() {
    let fixed = Voxels::F64(Array3::from_elem((0, 7, 5), 0.0));
    let moving = Voxels::F64(Array3::from_elem((0, 7, 5), 0.0));

    let error = error_text(run(&VolumeFitParams::default(), &fixed, &moving, 4));

    assert!(error.contains("empty image shape"), "{error}");
    assert!(error.contains("[0, 7, 5]"), "{error}");
}

#[test]
fn one_voxel_wide_images_are_refused_before_sampling() {
    let fixed = Voxels::F64(Array3::from_elem((1, 7, 5), 0.0));
    let moving = Voxels::F64(Array3::from_elem((1, 7, 5), 0.0));

    let error = error_text(run(&VolumeFitParams::default(), &fixed, &moving, 4));

    assert!(error.contains("too small for trilinear"), "{error}");
    assert!(error.contains("at least two voxels"), "{error}");
}

#[test]
fn incompatible_shapes_are_refused_by_name() {
    let fixed = Voxels::F64(Array3::from_elem((9, 7, 5), 1.0));
    let moving = Voxels::F64(Array3::from_elem((9, 7, 4), 1.0));

    let error = error_text(run(&VolumeFitParams::default(), &fixed, &moving, 4));

    assert!(error.contains("fixed image has shape"), "{error}");
    assert!(error.contains("moving image has shape"), "{error}");
}

#[test]
fn non_f64_images_are_refused_by_name() {
    let fixed = Voxels::F32(Array3::from_elem((9, 7, 5), 1.0));
    let moving = gaussian([3.95, 2.75, 2.4]);

    let error = error_text(run(&VolumeFitParams::default(), &fixed, &moving, 4));

    assert!(error.contains("fixed image is F32"), "{error}");
    assert!(error.contains("expected f64 volumes"), "{error}");
}

#[test]
fn non_finite_voxels_are_refused_by_name() {
    let mut fixed = gaussian([3.2, 3.1, 2.0]).view::<f64>().unwrap().to_owned();
    fixed[[2, 2, 2]] = f64::NAN;
    let moving = gaussian([3.95, 2.75, 2.4]);

    let error = error_text(run(
        &VolumeFitParams::default(),
        &Voxels::F64(fixed),
        &moving,
        4,
    ));

    assert!(
        error.contains("fixed image contains 1 non-finite"),
        "{error}"
    );
}

#[test]
fn flat_images_are_refused_as_no_metric_signal() {
    let fixed = Voxels::F64(Array3::from_elem((9, 7, 5), 1.0));
    let moving = gaussian([3.95, 2.75, 2.4]);

    let error = error_text(run(&VolumeFitParams::default(), &fixed, &moving, 4));

    assert!(error.contains("fixed image is flat"), "{error}");
    assert!(error.contains("no metric signal"), "{error}");
}

#[test]
fn images_without_positive_mass_are_refused_before_identity_can_escape() {
    let fixed = Voxels::F64(Array3::from_shape_fn((9, 7, 5), |(x, _, _)| {
        if x == 0 {
            -1.0
        } else {
            0.0
        }
    }));
    let moving = gaussian([3.95, 2.75, 2.4]);

    let error = error_text(run(&VolumeFitParams::default(), &fixed, &moving, 4));

    assert!(
        error.contains("fixed image has no positive finite mass"),
        "{error}"
    );
}

#[test]
fn no_overlap_after_initialisation_is_refused_by_name() {
    let fixed = point_image([0, 0, 0]);
    let moving = point_image([8, 6, 4]);

    let error = error_text(run(&VolumeFitParams::default(), &fixed, &moving, 4));

    assert!(error.contains("no overlapping moving samples"), "{error}");
    assert!(error.contains("current translation"), "{error}");
}

#[test]
fn impossible_optimiser_config_is_refused_before_execution() {
    let fixed = gaussian([3.2, 3.1, 2.0]);
    let moving = gaussian([3.95, 2.75, 2.4]);
    let params = VolumeFitParams {
        damping: f64::NAN,
        ..VolumeFitParams::default()
    };

    let error = error_text(run(&params, &fixed, &moving, 4));

    assert!(error.contains("damping must be finite"), "{error}");

    let params = VolumeFitParams {
        sampling: Sampling::Stride([2, 0, 2]),
        ..VolumeFitParams::default()
    };
    let error = error_text(run(&params, &fixed, &moving, 4));
    assert!(error.contains("sampling stride"), "{error}");
    assert!(error.contains("positive"), "{error}");
}

#[test]
fn diagnostics_report_the_optimiser_stopping_criterion() {
    let fixed = gaussian([3.2, 3.1, 2.0]);
    let moving = gaussian([3.95, 2.75, 2.4]);
    let params = VolumeFitParams {
        tolerance: 10.0,
        ..VolumeFitParams::default()
    };

    let diagnostics =
        run_with_diagnostics(&params, &fixed, &moving, 4, 2).expect("registration computes");

    assert_eq!(diagnostics.converged_by, FitConvergence::StepTolerance);
    assert!(diagnostics.substages >= 1, "{diagnostics:?}");
    assert!(diagnostics.final_cost.is_finite(), "{diagnostics:?}");
    assert!(
        diagnostics.final_step_norm <= params.tolerance,
        "{diagnostics:?}"
    );
    assert_point_close(
        diagnostics
            .transform
            .apply([3.2, 3.1, 2.0], &RegisterParams::default()),
        [3.95, 2.75, 2.4],
        10.0,
    );
}

#[test]
fn bad_geometry_spacing_and_direction_are_refused_by_name() {
    let spacing = ImageGeometry::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    .unwrap_err()
    .to_string();
    assert!(spacing.contains("Spacing"), "{spacing}");
    assert!(spacing.contains("zero or non-finite"), "{spacing}");

    let direction = ImageGeometry::new(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    .unwrap_err()
    .to_string();
    assert!(direction.contains("direction matrix"), "{direction}");
    assert!(direction.contains("cannot be inverted"), "{direction}");
}

#[test]
fn elastix_optimiser_settings_are_mapped_or_refused_by_name() {
    let resolutions = elastix_setting_support("NumberOfResolutions");
    assert_eq!(resolutions.status, ElastixSettingStatus::Supported);
    assert!(resolutions.native.contains("PyramidSchedule"));

    let optimizer = elastix_setting_support("Optimizer");
    assert_eq!(optimizer.status, ElastixSettingStatus::Approximation);
    assert!(optimizer.native.contains("Gauss-Newton"));
    assert!(optimizer.native.contains("StandardGradientDescent"));

    let sampler = elastix_setting_support("ImageSampler");
    assert_eq!(sampler.status, ElastixSettingStatus::Approximation);
    require_supported_elastix_setting("ImageSampler").expect("deterministic samplers are mapped");

    let bins = elastix_setting_support("NumberOfHistogramBins");
    assert_eq!(bins.status, ElastixSettingStatus::Supported);
    assert!(bins.native.contains("metric_bins"));

    let sample_budget = elastix_setting_support("NumberOfSpatialSamples");
    assert_eq!(sample_budget.status, ElastixSettingStatus::Unsupported);
    assert!(sample_budget.native.contains("deterministic"));

    let auto_parameter_estimation = elastix_setting_support("AutomaticParameterEstimation");
    assert_eq!(
        auto_parameter_estimation.status,
        ElastixSettingStatus::Unsupported
    );
    assert!(auto_parameter_estimation
        .native
        .contains("optimiser parameter estimation"));

    let fixed_interp = elastix_setting_support("FixedImageBSplineInterpolationOrder");
    assert_eq!(fixed_interp.status, ElastixSettingStatus::Unsupported);
    assert!(fixed_interp.native.contains("fixed voxels"));

    let memory_mode = elastix_setting_support("UseFastAndLowMemoryVersion");
    assert_eq!(memory_mode.status, ElastixSettingStatus::Unsupported);
    assert!(memory_mode.native.contains("memory mode"));

    let final_grid = elastix_setting_support("FinalGridSpacingInVoxels");
    assert_eq!(final_grid.status, ElastixSettingStatus::Supported);
    assert!(final_grid.native.contains("ControlGrid"));

    let unknown = require_supported_elastix_setting("NoSuchElastixSetting")
        .unwrap_err()
        .to_string();
    assert!(unknown.contains("NoSuchElastixSetting"), "{unknown}");
}

#[test]
fn common_elastix_optimiser_settings_are_deliberately_classified() {
    let relevant_settings = [
        "Transform",
        "Metric",
        "Metric0Weight",
        "Metric1Weight",
        "Optimizer",
        "NumberOfResolutions",
        "ImagePyramidSchedule",
        "FixedImagePyramid",
        "MovingImagePyramid",
        "AutomaticTransformInitialization",
        "InitialTransformParametersFileName",
        "HowToCombineTransforms",
        "AutomaticScalesEstimation",
        "AutomaticParameterEstimation",
        "MaximumNumberOfIterations",
        "NumberOfHistogramBins",
        "Interpolator",
        "BSplineInterpolationOrder",
        "FixedImageBSplineInterpolationOrder",
        "ImageSampler",
        "SampleGridSpacing",
        "NumberOfSpatialSamples",
        "NewSamplesEveryIteration",
        "UseRandomSampleRegion",
        "SampleRegionSize",
        "MaximumNumberOfSamplingAttempts",
        "MaximumStepLength",
        "LevenbergMarquardtDamping",
        "RegistrationTolerance",
        "BSplineTransformSplineOrder",
        "FinalGridSpacingInVoxels",
        "FinalBSplineInterpolationOrder",
        "UseFastAndLowMemoryVersion",
        "SP_a",
        "SP_A",
        "SP_alpha",
    ];

    for setting in relevant_settings {
        let support = elastix_setting_support(setting);
        assert_ne!(
            support.setting, "unknown",
            "{setting} must be deliberately classified"
        );
        assert!(
            !support.native.trim().is_empty(),
            "{setting} must state the native behavior or refusal reason"
        );
    }
}

#[test]
fn elastix_parameter_text_builds_native_compute_config() {
    let params = params_from_elastix_text(
        r#"
        (Registration "MultiResolutionRegistration")
        (Transform "AffineTransform")
        (Metric "AdvancedMattesMutualInformation")
        (Optimizer "AdaptiveStochasticGradientDescent")
        (FixedImagePyramid "FixedSmoothingImagePyramid")
        (MovingImagePyramid "MovingSmoothingImagePyramid")
        (Interpolator "BSplineInterpolator")
        (BSplineInterpolationOrder 1)
        (ImageSampler "Grid")
        (SampleGridSpacing 2 1 3)
        (NumberOfHistogramBins 16 32)
        (AutomaticTransformInitialization "true")
        (AutomaticScalesEstimation "true")
        (InitialTransformParametersFileName "NoInitialTransform")
        (HowToCombineTransforms "Compose")
        (NumberOfResolutions 3)
        (ImagePyramidSchedule 6 4 2 3 2 1 1 1 1)
        (MaximumNumberOfIterations 8 16 12)
        (MaximumStepLength 0.75)
        (LevenbergMarquardtDamping 0.0002)
        (RegistrationTolerance 0.003)
        (BSplineTransformSplineOrder 3)
        (WriteTransformParametersEachIteration "false")
        (CompressResultImage "false")
        (WriteResultImage "false")
        (WriteResultImageAfterEachResolution "false")
        (ShowExactMetricValue "false")
        (WriteDiffusionFiles "true")
        "#,
    )
    .expect("elastix config maps");

    assert_eq!(params.model, TransformModel::Affine);
    assert_eq!(params.metric, Metric::MutualInformation);
    assert_eq!(params.limit.substages(), 16);
    assert_eq!(params.max_step, 0.75);
    assert_eq!(params.damping, 0.0002);
    assert_eq!(params.tolerance, 0.003);
    assert_eq!(params.metric_bins, 32);
    assert_eq!(params.sampling, Sampling::Stride([2, 1, 3]));
    assert_eq!(params.interpolator, Interpolator::Linear);
    let schedule = params.pyramid.expect("pyramid schedule");
    assert_eq!(schedule.levels(), 3);
    assert_eq!(schedule.factors(), &[[3, 2, 1], [2, 2, 2]]);
}

#[test]
fn elastix_parameter_text_refuses_unsupported_settings_by_name() {
    let sampler = params_from_elastix_text(
        r#"
        (Transform "TranslationTransform")
        (Metric "AdvancedMeanSquares")
        (ImageSampler "RandomCoordinate")
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(sampler.contains("ImageSampler"), "{sampler}");
    assert!(sampler.contains("random sampling state"), "{sampler}");

    let interpolator = params_from_elastix_text(
        r#"
        (Transform "TranslationTransform")
        (Metric "AdvancedMeanSquares")
        (Interpolator "BSplineInterpolator")
        (BSplineInterpolationOrder 3)
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(interpolator.contains("Interpolator"), "{interpolator}");
    assert!(
        interpolator.contains("BSplineInterpolationOrder"),
        "{interpolator}"
    );

    let bspline_optimizer = params_from_elastix_text(
        r#"
        (Transform "BSplineTransform")
        (Metric "AdvancedMeanSquares")
        (Optimizer "StandardGradientDescent")
        (FinalGridSpacingInVoxels 2.0 2.0 1.0)
        "#,
    )
    .expect("a gradient-descent B-spline maps to the native approximation");
    assert_eq!(
        bspline_optimizer.bspline_final_grid_spacing,
        Some([2.0, 2.0, 1.0])
    );

    let optimizer = params_from_elastix_text(
        r#"
        (Transform "TranslationTransform")
        (Metric "AdvancedMeanSquares")
        (Optimizer "QuasiNewtonLBFGS")
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(optimizer.contains("Optimizer"), "{optimizer}");
    assert!(optimizer.contains("QuasiNewtonLBFGS"), "{optimizer}");

    let wrong_transform_grid_spacing = params_from_elastix_text(
        r#"
        (Transform "TranslationTransform")
        (Metric "AdvancedMeanSquares")
        (FinalGridSpacingInVoxels 25.0 25.0 25.0)
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        wrong_transform_grid_spacing.contains("FinalGridSpacingInVoxels"),
        "{wrong_transform_grid_spacing}"
    );
    assert!(
        wrong_transform_grid_spacing.contains("BSplineTransform"),
        "{wrong_transform_grid_spacing}"
    );

    let bad_grid_spacing = params_from_elastix_text(
        r#"
        (Transform "BSplineTransform")
        (Metric "AdvancedMeanSquares")
        (FinalGridSpacingInVoxels 25.0 0.0 25.0)
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        bad_grid_spacing.contains("FinalGridSpacingInVoxels"),
        "{bad_grid_spacing}"
    );
    assert!(bad_grid_spacing.contains("positive"), "{bad_grid_spacing}");

    let no_auto_init = params_from_elastix_text(
        r#"
        (Transform "AffineTransform")
        (Metric "AdvancedMeanSquares")
        (AutomaticTransformInitialization "false")
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        no_auto_init.contains("AutomaticTransformInitialization"),
        "{no_auto_init}"
    );
    assert!(no_auto_init.contains("false"), "{no_auto_init}");

    let no_auto_scales = params_from_elastix_text(
        r#"
        (Transform "AffineTransform")
        (Metric "AdvancedMeanSquares")
        (AutomaticScalesEstimation "false")
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        no_auto_scales.contains("AutomaticScalesEstimation"),
        "{no_auto_scales}"
    );
    assert!(no_auto_scales.contains("false"), "{no_auto_scales}");

    let fixed_interpolation = params_from_elastix_text(
        r#"
        (Transform "AffineTransform")
        (Metric "AdvancedMeanSquares")
        (FixedImageBSplineInterpolationOrder 3)
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        fixed_interpolation.contains("FixedImageBSplineInterpolationOrder"),
        "{fixed_interpolation}"
    );

    let spatial_samples = params_from_elastix_text(
        r#"
        (Transform "TranslationTransform")
        (Metric "AdvancedMeanSquares")
        (ImageSampler "Grid")
        (NumberOfSpatialSamples 4000)
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        spatial_samples.contains("NumberOfSpatialSamples"),
        "{spatial_samples}"
    );

    let automatic_parameter_estimation = params_from_elastix_text(
        r#"
        (Transform "TranslationTransform")
        (Metric "AdvancedMeanSquares")
        (AutomaticParameterEstimation "true")
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        automatic_parameter_estimation.contains("AutomaticParameterEstimation"),
        "{automatic_parameter_estimation}"
    );
    assert!(
        automatic_parameter_estimation.contains("unsupported"),
        "{automatic_parameter_estimation}"
    );

    let bad_fixed_pyramid = params_from_elastix_text(
        r#"
        (Transform "TranslationTransform")
        (Metric "AdvancedMeanSquares")
        (FixedImagePyramid "FixedShrinkingImagePyramid")
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        bad_fixed_pyramid.contains("FixedImagePyramid"),
        "{bad_fixed_pyramid}"
    );
    assert!(
        bad_fixed_pyramid.contains("FixedShrinkingImagePyramid"),
        "{bad_fixed_pyramid}"
    );

    let chained_initial_transform = params_from_elastix_text(
        r#"
        (Transform "TranslationTransform")
        (Metric "AdvancedMeanSquares")
        (InitialTransformParametersFileName "TransformParameters.0.txt")
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        chained_initial_transform.contains("InitialTransformParametersFileName"),
        "{chained_initial_transform}"
    );
    assert!(
        chained_initial_transform.contains("TransformParameters.0.txt"),
        "{chained_initial_transform}"
    );

    let additive_chain = params_from_elastix_text(
        r#"
        (Transform "TranslationTransform")
        (Metric "AdvancedMeanSquares")
        (InitialTransformParametersFileName "NoInitialTransform")
        (HowToCombineTransforms "Add")
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        additive_chain.contains("HowToCombineTransforms"),
        "{additive_chain}"
    );
    assert!(additive_chain.contains("Add"), "{additive_chain}");

    let unknown = params_from_elastix_text(
        r#"
        (Transform "TranslationTransform")
        (Metric "AdvancedMeanSquares")
        (NoSuchSetting 1)
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(unknown.contains("NoSuchSetting"), "{unknown}");
    assert!(unknown.contains("not mapped"), "{unknown}");

    let inconsistent = params_from_elastix_text(
        r#"
        (Transform "TranslationTransform")
        (Metric "AdvancedMeanSquares")
        (NumberOfResolutions 2)
        (ImagePyramidSchedule 4 4 4 2 2 2 1 1 1)
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        inconsistent.contains("NumberOfResolutions"),
        "{inconsistent}"
    );
    assert!(
        inconsistent.contains("ImagePyramidSchedule"),
        "{inconsistent}"
    );
}

#[test]
fn smooth_subvoxel_translation_is_optimised() {
    let fixed = gaussian([3.2, 3.1, 2.0]);
    let moving = gaussian([3.95, 2.75, 2.4]);

    let transform =
        run(&VolumeFitParams::default(), &fixed, &moving, 4).expect("registration computes");
    let moved = transform.apply([3.2, 3.1, 2.0], &RegisterParams::default());

    for (got, want) in moved.into_iter().zip([3.95, 2.75, 2.4]) {
        assert!((got - want).abs() < 0.08, "{got} vs {want}");
    }
}

#[test]
fn deterministic_grid_sampling_optimises_translation() {
    let fixed = gaussian([3.2, 3.1, 2.0]);
    let moving = gaussian([3.95, 2.75, 2.4]);
    let params = VolumeFitParams::default().with_sampling(Sampling::Stride([2, 2, 1]));

    let transform = run(&params, &fixed, &moving, 4).expect("sampled registration computes");
    let moved = transform.apply([3.2, 3.1, 2.0], &RegisterParams::default());

    for (got, want) in moved.into_iter().zip([3.95, 2.75, 2.4]) {
        assert!((got - want).abs() < 0.14, "{got} vs {want}");
    }
}

#[test]
fn normalized_correlation_handles_intensity_scaled_translation() {
    let fixed = gaussian([3.2, 3.1, 2.0]);
    let moving = scaled_gaussian([3.95, 2.75, 2.4], 3.0);

    let transform = run(
        &VolumeFitParams::default().with_metric(Metric::NormalizedCorrelation),
        &fixed,
        &moving,
        4,
    )
    .expect("registration computes");
    let moved = transform.apply([3.2, 3.1, 2.0], &RegisterParams::default());

    for (got, want) in moved.into_iter().zip([3.95, 2.75, 2.4]) {
        assert!((got - want).abs() < 0.12, "{got} vs {want}");
    }
}

#[test]
fn mutual_information_handles_intensity_scaled_translation() {
    let fixed = gaussian([3.2, 3.1, 2.0]);
    let moving = scaled_gaussian([3.95, 2.75, 2.4], 3.0);
    let params = VolumeFitParams::default()
        .with_metric(Metric::MutualInformation)
        .with_metric_bins(16)
        .expect("metric bins");

    let transform = run(&params, &fixed, &moving, 4).expect("registration computes");
    let moved = transform.apply([3.2, 3.1, 2.0], &RegisterParams::default());

    for (got, want) in moved.into_iter().zip([3.95, 2.75, 2.4]) {
        assert!((got - want).abs() < 0.16, "{got} vs {want}");
    }
}

#[test]
fn affine_mode_emits_a_working_affine_transform() {
    let fixed = asymmetric_smooth([3.2, 3.1, 2.0]);
    let moving = asymmetric_smooth([3.75, 2.85, 2.35]);

    let transform = run(&VolumeFitParams::default().affine(), &fixed, &moving, 4)
        .expect("affine registration computes");
    let moved = transform.apply([3.2, 3.1, 2.0], &RegisterParams::default());

    for (got, want) in moved.into_iter().zip([3.75, 2.85, 2.35]) {
        assert!((got - want).abs() < 0.12, "{got} vs {want}");
    }
}

#[test]
fn bspline_mode_emits_a_working_cubic_grid_transform() {
    let fixed = point_image([3, 3, 2]);
    let moving = point_image([4, 2, 2]);
    let params = VolumeFitParams::default()
        .bspline()
        .with_bspline_grid(ControlGrid::new([6, 6, 6]).expect("grid"));

    let transform = run(&params, &fixed, &moving, 4).expect("bspline registration computes");
    let moved = transform.apply([3.0, 3.0, 2.0], &RegisterParams::default());

    for (got, want) in moved.into_iter().zip([4.0, 2.0, 2.0]) {
        assert!((got - want).abs() < 0.25, "{got} vs {want}");
    }
}

#[test]
fn bspline_mode_aligns_a_voxel_intensity_probe() {
    let fixed_at = [3, 3, 2];
    let moving_at = [4, 2, 2];
    let fixed = point_image(fixed_at);
    let moving = point_image(moving_at);
    let fixed_array = fixed.view::<f64>().unwrap().to_owned();
    let moving_array = moving.view::<f64>().unwrap().to_owned();
    let params = VolumeFitParams::default()
        .bspline()
        .with_bspline_grid(ControlGrid::new([6, 6, 6]).expect("grid"))
        .with_bspline_smoothness(0.01)
        .expect("smoothness");

    let transform = run(&params, &fixed, &moving, 4).expect("bspline registration computes");
    let probe = [fixed_at[0] as f64, fixed_at[1] as f64, fixed_at[2] as f64];
    let mapped = transform.apply(probe, &RegisterParams::default());
    let fixed_value = trilinear_sample(&fixed_array, probe).expect("fixed probe in bounds");
    let moving_value = trilinear_sample(&moving_array, mapped).expect("mapped probe in bounds");

    assert_eq!(fixed_value, 10.0);
    assert!(
        moving_value >= 5.0,
        "B-spline probe {probe:?} mapped to {mapped:?}: moving intensity {moving_value}"
    );
}

#[test]
fn bspline_mode_is_stable_across_small_block_cuts() {
    let fixed_at = [3, 3, 2];
    let moving_at = [4, 2, 2];
    let fixed = point_image(fixed_at);
    let moving = point_image(moving_at);
    let fixed_array = fixed.view::<f64>().unwrap().to_owned();
    let moving_array = moving.view::<f64>().unwrap().to_owned();
    let registration_params = RegisterParams::default();
    let compute_params = VolumeFitParams::default()
        .bspline()
        .with_bspline_grid(ControlGrid::new([6, 6, 6]).expect("grid"))
        .with_bspline_smoothness(0.01)
        .expect("smoothness");
    let reference =
        run(&compute_params, &fixed, &moving, SHAPE[0]).expect("full-cut B-spline computes");

    for block in [4, 3] {
        let transform =
            run(&compute_params, &fixed, &moving, block).expect("cut B-spline computes");
        for probe in [[3.0, 3.0, 2.0], [2.7, 3.2, 2.0]] {
            assert_point_close(
                transform.apply(probe, &registration_params),
                reference.apply(probe, &registration_params),
                1.0e-9,
            );
        }

        let probe = [fixed_at[0] as f64, fixed_at[1] as f64, fixed_at[2] as f64];
        let mapped = transform.apply(probe, &registration_params);
        let fixed_value = trilinear_sample(&fixed_array, probe).expect("fixed probe in bounds");
        let moving_value =
            trilinear_sample(&moving_array, mapped).expect("mapped moving probe in bounds");
        assert_eq!(fixed_value, 10.0);
        assert!(
            moving_value >= 5.0,
            "B-spline cut {block} probe {probe:?} mapped to {mapped:?}: moving intensity \
             {moving_value}"
        );
    }
}

#[test]
fn bspline_mode_accepts_control_grid_smoothness() {
    let fixed = point_image([3, 3, 2]);
    let moving = point_image([4, 2, 2]);
    let params = VolumeFitParams::default()
        .bspline()
        .with_bspline_grid(ControlGrid::new([6, 6, 6]).expect("grid"))
        .with_bspline_smoothness(0.01)
        .expect("smoothness");

    let transform = run(&params, &fixed, &moving, 4).expect("bspline registration computes");
    let moved = transform.apply([3.0, 3.0, 2.0], &RegisterParams::default());

    for (got, want) in moved.into_iter().zip([4.0, 2.0, 2.0]) {
        assert!((got - want).abs() < 0.30, "{got} vs {want}");
    }
}

#[test]
fn computed_transform_carries_physical_spacing() {
    let fixed = point_image([2, 3, 1]);
    let moving = point_image([4, 2, 3]);
    let params = VolumeFitParams::default().with_geometry(
        ImageGeometry::new(
            [0.0, 0.0, 0.0],
            [2.0, 3.0, 5.0],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        .expect("geometry")
        .spatial_frame()
        .expect("frame"),
    );

    let transform = run(&params, &fixed, &moving, 4).expect("registration computes");

    assert_eq!(
        transform.apply([4.0, 9.0, 5.0], &RegisterParams::default()),
        [8.0, 6.0, 15.0]
    );
}

#[test]
fn generated_translation_transform_roundtrips_through_native_reader() {
    let fixed = point_image([2, 3, 1]);
    let moving = point_image([4, 2, 3]);
    let (transform, text) =
        run_with_transform_text(&VolumeFitParams::default(), &fixed, &moving, 4)
            .expect("registration computes");

    assert!(
        text.contains("(Transform \"TranslationTransform\")"),
        "{text}"
    );
    assert!(text.contains("(TransformParameters 2 -1 2)"), "{text}");
    assert_generated_text_roundtrips(
        &text,
        &transform,
        &[[2.0, 3.0, 1.0], [0.0, 0.0, 0.0], [8.0, 6.0, 4.0]],
    );
}

#[test]
fn generated_affine_transform_roundtrips_through_native_reader() {
    let centre = [8.5, 8.0, 6.0];
    let matrix = [
        [1.02, 0.025, 0.0],
        [-0.015, 0.985, 0.02],
        [0.0, -0.01, 1.015],
    ];
    let shift = [0.35, -0.25, 0.20];
    let (fixed, moving, _) = affine_medium_pair(matrix, centre, shift);
    let (transform, text) =
        run_with_transform_text(&VolumeFitParams::default().affine(), &fixed, &moving, 8)
            .expect("affine registration computes");

    assert!(text.contains("(Transform \"AffineTransform\")"), "{text}");
    assert!(text.contains("(NumberOfParameters 12)"), "{text}");
    assert!(
        text.contains("(CenterOfRotationPoint 0.0 0.0 0.0)"),
        "{text}"
    );
    assert_generated_text_roundtrips(
        &text,
        &transform,
        &[[8.5, 8.0, 6.0], [5.0, 12.0, 4.0], [13.0, 5.0, 8.0]],
    );
}

#[test]
fn generated_bspline_transform_roundtrips_through_native_reader() {
    let fixed = point_image([3, 3, 2]);
    let moving = point_image([4, 2, 2]);
    let compute_params = VolumeFitParams::default()
        .bspline()
        .with_bspline_grid(ControlGrid::new([6, 6, 6]).expect("grid"))
        .with_bspline_smoothness(0.01)
        .expect("smoothness");
    let (transform, text) =
        run_with_transform_text(&compute_params, &fixed, &moving, 4).expect("B-spline computes");

    assert!(text.contains("(Transform \"BSplineTransform\")"), "{text}");
    assert!(text.contains("(GridSize 6 6 6)"), "{text}");
    assert!(text.contains("(GridSpacing"), "{text}");
    assert!(text.contains("(GridOrigin"), "{text}");
    assert!(text.contains("(BSplineTransformSplineOrder 3)"), "{text}");
    assert_generated_text_roundtrips(
        &text,
        &transform,
        &[[3.0, 3.0, 2.0], [4.0, 3.0, 2.0], [3.0, 2.0, 2.0]],
    );
}

#[test]
fn bspline_final_grid_spacing_derives_generated_grid_size() {
    let fixed = point_image([3, 3, 2]);
    let moving = point_image([4, 2, 2]);
    let compute_params = VolumeFitParams::default()
        .bspline()
        .with_bspline_final_grid_spacing([2.0, 2.0, 1.0])
        .expect("final grid spacing")
        .with_bspline_smoothness(0.01)
        .expect("smoothness");

    let (transform, text) =
        run_with_transform_text(&compute_params, &fixed, &moving, 4).expect("B-spline computes");

    // `ops::align` derives `ceil(shape / spacing) + 3` control points per axis:
    // shape [9, 7, 5] at spacing [2, 2, 1].
    assert!(text.contains("(GridSize 8 7 8)"), "{text}");
    assert!(text.contains("(NumberOfParameters 1344)"), "{text}");
    assert_generated_text_roundtrips(
        &text,
        &transform,
        &[[3.0, 3.0, 2.0], [4.0, 3.0, 2.0], [3.0, 2.0, 2.0]],
    );
}

#[test]
fn malformed_generated_transform_files_are_refused_by_name() {
    let fixed = point_image([2, 3, 1]);
    let moving = point_image([4, 2, 3]);
    let (_, translation) = run_with_transform_text(&VolumeFitParams::default(), &fixed, &moving, 4)
        .expect("translation registration computes");
    assert_transform_text_error_contains(
        translation.replace("(NumberOfParameters 3)", "(NumberOfParameters 2)"),
        &["NumberOfParameters 2", "TransformParameters"],
    );

    let centre = [8.5, 8.0, 6.0];
    let matrix = [
        [1.02, 0.025, 0.0],
        [-0.015, 0.985, 0.02],
        [0.0, -0.01, 1.015],
    ];
    let shift = [0.35, -0.25, 0.20];
    let (fixed, moving, _) = affine_medium_pair(matrix, centre, shift);
    let (_, affine) =
        run_with_transform_text(&VolumeFitParams::default().affine(), &fixed, &moving, 8)
            .expect("affine registration computes");
    assert_transform_text_error_contains(
        affine.replace("(CenterOfRotationPoint", "(CenterOfRotationPointMissing"),
        &["CenterOfRotationPoint"],
    );

    let fixed = point_image([3, 3, 2]);
    let moving = point_image([4, 2, 2]);
    let compute_params = VolumeFitParams::default()
        .bspline()
        .with_bspline_grid(ControlGrid::new([6, 6, 6]).expect("grid"))
        .with_bspline_smoothness(0.01)
        .expect("smoothness");
    let (_, bspline) =
        run_with_transform_text(&compute_params, &fixed, &moving, 4).expect("B-spline computes");
    assert_transform_text_error_contains(
        bspline.replace("(GridOrigin", "(GridOriginMissing"),
        &["GridOrigin"],
    );
    assert_transform_text_error_contains(
        bspline.replace("(GridSize 6 6 6)", "(GridSize 5 6 6)"),
        &["GridSize"],
    );
}

#[test]
fn anisotropic_pyramid_scales_translation_parameters_between_levels() {
    let shape = [15, 9, 7];
    let fixed_at = [3.0, 4.0, 3.0];
    let moving_at = [6.0, 4.0, 3.0];
    let fixed = point_image_in(shape, [3, 4, 3]);
    let moving = point_image_in(shape, [6, 4, 3]);
    let schedule = PyramidSchedule::new(vec![[3, 1, 1]]).expect("anisotropic schedule");

    let transform = resident_pyramid(
        &VolumeFitParams::default(),
        &fixed.view::<f64>().unwrap().to_owned(),
        &moving.view::<f64>().unwrap().to_owned(),
        &schedule,
    )
    .expect("anisotropic pyramid registration computes");

    assert_point_close(
        transform.apply(fixed_at, &RegisterParams::default()),
        moving_at,
        0.001,
    );
}

#[test]
fn medium_translation_is_stable_across_three_block_cuts() {
    let fixed_at = [7.4, 8.2, 6.1];
    let moving_at = [8.15, 7.65, 6.55];
    let fixed = gaussian_in(MEDIUM, fixed_at);
    let moving = gaussian_in(MEDIUM, moving_at);
    let params = RegisterParams::default();

    for block in [MEDIUM[0], 7, 4] {
        let transform = run(&VolumeFitParams::default(), &fixed, &moving, block)
            .expect("medium translation computes");
        assert_point_close(transform.apply(fixed_at, &params), moving_at, 0.08);
        assert_point_close(
            transform.apply([4.0, 12.0, 3.0], &params),
            [4.75, 11.45, 3.45],
            0.08,
        );
    }
}

#[test]
fn medium_translation_is_stable_across_worker_counts() {
    let fixed_at = [7.4, 8.2, 6.1];
    let moving_at = [8.15, 7.65, 6.55];
    let fixed = gaussian_in(MEDIUM, fixed_at);
    let moving = gaussian_in(MEDIUM, moving_at);
    let params = RegisterParams::default();
    let reference = run_with_workers(&VolumeFitParams::default(), &fixed, &moving, 5, 1)
        .expect("one-worker registration computes");

    for workers in [2, 4] {
        let transform = run_with_workers(&VolumeFitParams::default(), &fixed, &moving, 5, workers)
            .expect("multi-worker registration computes");
        for point in [[fixed_at[0], fixed_at[1], fixed_at[2]], [4.0, 12.0, 3.0]] {
            assert_point_close(
                transform.apply(point, &params),
                reference.apply(point, &params),
                1.0e-9,
            );
        }
        assert_point_close(transform.apply(fixed_at, &params), moving_at, 0.08);
    }
}

#[test]
fn medium_translation_aligns_voxel_intensity_probes() {
    let fixed_at = [7.4, 8.2, 6.1];
    let moving_at = [8.15, 7.65, 6.55];
    let fixed = gaussian_in(MEDIUM, fixed_at);
    let moving = gaussian_in(MEDIUM, moving_at);
    let fixed_array = fixed.view::<f64>().unwrap().to_owned();
    let moving_array = moving.view::<f64>().unwrap().to_owned();
    let params = RegisterParams::default();

    let transform =
        run(&VolumeFitParams::default(), &fixed, &moving, 5).expect("medium translation computes");

    for probe in [[7.4, 8.2, 6.1], [5.8, 10.4, 5.2], [10.2, 6.9, 7.0]] {
        let fixed_value = trilinear_sample(&fixed_array, probe).expect("fixed probe in bounds");
        let mapped = transform.apply(probe, &params);
        let moving_value =
            trilinear_sample(&moving_array, mapped).expect("mapped moving probe in bounds");
        assert!(
            (fixed_value - moving_value).abs() <= 0.08,
            "probe {probe:?} mapped to {mapped:?}: fixed {fixed_value} moving {moving_value}"
        );
    }
}

#[test]
fn medium_affine_model_tracks_known_shift_under_decomposition() {
    let fixed_at = [7.4, 8.2, 6.1];
    let moving_at = [8.0, 7.9, 6.45];
    let fixed = gaussian_in(MEDIUM, fixed_at);
    let moving = gaussian_in(MEDIUM, moving_at);
    let params = RegisterParams::default();

    for block in [MEDIUM[0], 8, 5] {
        let transform = run(&VolumeFitParams::default().affine(), &fixed, &moving, block)
            .expect("medium affine computes");
        assert_point_close(transform.apply(fixed_at, &params), moving_at, 0.16);
    }
}

#[test]
fn medium_affine_model_recovers_non_translation_point_probes() {
    let centre = [8.5, 8.0, 6.0];
    let matrix = [
        [1.02, 0.025, 0.0],
        [-0.015, 0.985, 0.02],
        [0.0, -0.01, 1.015],
    ];
    let shift = [0.35, -0.25, 0.20];
    let (fixed, moving, translation) = affine_medium_pair(matrix, centre, shift);
    let params = RegisterParams::default();
    let probes = [[8.5, 8.0, 6.0], [5.0, 12.0, 4.0], [13.0, 5.0, 8.0]];

    for block in [MEDIUM[0], 8, 5] {
        let transform = run(&VolumeFitParams::default().affine(), &fixed, &moving, block)
            .expect("medium non-translation affine computes");
        for probe in probes {
            assert_point_close(
                transform.apply(probe, &params),
                apply_affine(matrix, translation, probe),
                0.35,
            );
        }
    }
}

#[test]
fn medium_affine_model_aligns_voxel_intensity_probes() {
    let centre = [8.5, 8.0, 6.0];
    let matrix = [
        [1.02, 0.025, 0.0],
        [-0.015, 0.985, 0.02],
        [0.0, -0.01, 1.015],
    ];
    let shift = [0.35, -0.25, 0.20];
    let (fixed, moving, _) = affine_medium_pair(matrix, centre, shift);
    let fixed_array = fixed.view::<f64>().unwrap().to_owned();
    let moving_array = moving.view::<f64>().unwrap().to_owned();
    let params = RegisterParams::default();

    let transform = run(&VolumeFitParams::default().affine(), &fixed, &moving, 8)
        .expect("medium affine computes");

    for probe in [[8.5, 8.0, 6.0], [5.7, 11.4, 4.5], [12.6, 5.8, 8.0]] {
        let fixed_value = trilinear_sample(&fixed_array, probe).expect("fixed probe in bounds");
        let mapped = transform.apply(probe, &params);
        let moving_value =
            trilinear_sample(&moving_array, mapped).expect("mapped moving probe in bounds");
        assert!(
            (fixed_value - moving_value).abs() <= 0.12,
            "probe {probe:?} mapped to {mapped:?}: fixed {fixed_value} moving {moving_value}"
        );
    }
}

#[test]
#[ignore = "timing: the medium B-spline decomposition check measured 40.55 s; run with --ignored"]
fn medium_bspline_model_moves_known_point_under_decomposition() {
    let fixed_at = [8, 8, 6];
    let moving_at = [9, 7, 6];
    let fixed = point_image_in(MEDIUM, fixed_at);
    let moving = point_image_in(MEDIUM, moving_at);
    let fixed_array = fixed.view::<f64>().unwrap().to_owned();
    let moving_array = moving.view::<f64>().unwrap().to_owned();
    let registration_params = RegisterParams::default();
    let compute_params = VolumeFitParams::default()
        .bspline()
        .with_bspline_grid(ControlGrid::new([6, 6, 6]).expect("grid"))
        .with_bspline_smoothness(0.01)
        .expect("smoothness");

    for block in [MEDIUM[0], 7, 4] {
        let transform =
            run(&compute_params, &fixed, &moving, block).expect("medium B-spline computes");
        assert_point_close(
            transform.apply(
                [fixed_at[0] as f64, fixed_at[1] as f64, fixed_at[2] as f64],
                &registration_params,
            ),
            [
                moving_at[0] as f64,
                moving_at[1] as f64,
                moving_at[2] as f64,
            ],
            0.35,
        );
        let probe = [fixed_at[0] as f64, fixed_at[1] as f64, fixed_at[2] as f64];
        let mapped = transform.apply(probe, &registration_params);
        let fixed_value = trilinear_sample(&fixed_array, probe).expect("fixed probe in bounds");
        let moving_value =
            trilinear_sample(&moving_array, mapped).expect("mapped moving probe in bounds");
        assert_eq!(fixed_value, 10.0);
        assert!(
            moving_value >= 5.0,
            "B-spline probe {probe:?} mapped to {mapped:?}: moving intensity {moving_value}"
        );
    }
}

#[test]
fn smooth_subvoxel_translation_can_run_coarse_to_fine() {
    let fixed = gaussian([3.2, 3.1, 2.0]);
    let moving = gaussian([3.95, 2.75, 2.4]);
    let schedule = PyramidSchedule::powers_of_two(2).expect("schedule");

    let transform = resident_pyramid(
        &VolumeFitParams::default(),
        &fixed.view::<f64>().unwrap().to_owned(),
        &moving.view::<f64>().unwrap().to_owned(),
        &schedule,
    )
    .expect("pyramid registration computes");
    let moved = transform.apply([3.2, 3.1, 2.0], &RegisterParams::default());

    for (got, want) in moved.into_iter().zip([3.95, 2.75, 2.4]) {
        assert!((got - want).abs() < 0.10, "{got} vs {want}");
    }
}

#[test]
fn resident_uses_configured_smoothed_optimisation_pyramid() {
    let fixed = gaussian([3.2, 3.1, 2.0]);
    let moving = gaussian([3.95, 2.75, 2.4]);
    let params = VolumeFitParams::default()
        .with_pyramid(PyramidSchedule::powers_of_two(2).expect("schedule"))
        .with_pyramid_smoothing(PyramidSmoothing::isotropic(0.4, 3.0).expect("smoothing"));

    let transform = resident(
        &params,
        &fixed.view::<f64>().unwrap().to_owned(),
        &moving.view::<f64>().unwrap().to_owned(),
    )
    .expect("pyramid registration computes");
    let moved = transform.apply([3.2, 3.1, 2.0], &RegisterParams::default());

    for (got, want) in moved.into_iter().zip([3.95, 2.75, 2.4]) {
        assert!((got - want).abs() < 0.12, "{got} vs {want}");
    }
}

#[test]
fn registration_pyramid_accepts_an_explicit_image_recipe() {
    let fixed = gaussian([3.2, 3.1, 2.0]);
    let moving = gaussian([3.95, 2.75, 2.4]);
    let schedule = PyramidSchedule::powers_of_two(2).expect("schedule");
    let recipe = PyramidRecipe::decimation(schedule.factors().to_vec()).expect("recipe");

    let transform = resident_pyramid_with_recipe(
        &VolumeFitParams::default(),
        &fixed.view::<f64>().unwrap().to_owned(),
        &moving.view::<f64>().unwrap().to_owned(),
        &schedule,
        &recipe,
    )
    .expect("pyramid registration computes");
    let moved = transform.apply([3.2, 3.1, 2.0], &RegisterParams::default());

    for (got, want) in moved.into_iter().zip([3.95, 2.75, 2.4]) {
        assert!((got - want).abs() < 0.10, "{got} vs {want}");
    }
}

#[test]
fn registration_pyramid_accepts_explicit_levels_without_mutating_them() {
    let fixed = gaussian([3.2, 3.1, 2.0]).view::<f64>().unwrap().to_owned();
    let moving = gaussian([3.95, 2.75, 2.4])
        .view::<f64>()
        .unwrap()
        .to_owned();
    let schedule = PyramidSchedule::powers_of_two(2).expect("schedule");
    let recipe = PyramidRecipe::decimation(schedule.factors().to_vec()).expect("recipe");
    let params = VolumeFitParams::default();
    let levels =
        pyramid_levels_from_recipe(&params, &fixed, &moving, &schedule, &recipe).expect("levels");
    let level_zero_before = levels[0].fixed.clone();
    let level_one_before = levels[1].fixed.clone();

    let from_recipe =
        resident_pyramid_with_recipe(&params, &fixed, &moving, &schedule, &recipe).expect("recipe");
    let from_explicit = resident_pyramid_levels(&params, &levels).expect("explicit levels");

    for point in [[3.2, 3.1, 2.0], [1.0, 1.0, 1.0], [6.0, 5.0, 3.0]] {
        assert_point_close(
            from_explicit.apply(point, &RegisterParams::default()),
            from_recipe.apply(point, &RegisterParams::default()),
            1.0e-9,
        );
    }
    assert_voxels_same(&levels[0].fixed, &level_zero_before);
    assert_voxels_same(&levels[1].fixed, &level_one_before);
}

#[test]
fn registration_pyramid_level_metadata_is_refused_by_name() {
    let fixed = gaussian([3.2, 3.1, 2.0]).view::<f64>().unwrap().to_owned();
    let moving = gaussian([3.95, 2.75, 2.4])
        .view::<f64>()
        .unwrap()
        .to_owned();
    let schedule = PyramidSchedule::powers_of_two(2).expect("schedule");
    let recipe = PyramidRecipe::decimation(schedule.factors().to_vec()).expect("recipe");
    let params = VolumeFitParams::default();
    let levels =
        pyramid_levels_from_recipe(&params, &fixed, &moving, &schedule, &recipe).expect("levels");

    let mut bad_geometry = levels.clone();
    bad_geometry[1] = VolumePyramidLevel::new(
        bad_geometry[1].fixed.clone(),
        bad_geometry[1].moving.clone(),
        bad_geometry[1].scale,
        SpatialFrame::new(
            [0.0, 0.0, 0.0],
            [7.0, 2.0, 2.0],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        .expect("bad but invertible geometry"),
    )
    .expect("level");
    let error = error_text(resident_pyramid_levels(&params, &bad_geometry));
    assert!(error.contains("level 1"), "{error}");
    assert!(error.contains("spacing[0]"), "{error}");

    let mut bad_shape = levels.clone();
    bad_shape[1] = VolumePyramidLevel::new(
        bad_shape[1].fixed.clone(),
        bad_shape[1].moving.clone(),
        [3, 2, 2],
        bad_shape[1].geometry.clone(),
    )
    .expect("level");
    let error = error_text(resident_pyramid_levels(&params, &bad_shape));
    assert!(error.contains("level 1"), "{error}");
    assert!(error.contains("axis 0"), "{error}");
    assert!(error.contains("scale 3"), "{error}");
}

#[test]
fn registration_pyramid_level_invariants_are_refused_by_name() {
    let params = VolumeFitParams::default();
    let empty = error_text(resident_pyramid_levels(&params, &[]));
    assert!(empty.contains("at least one level"), "{empty}");

    let fixed = gaussian([3.2, 3.1, 2.0]).view::<f64>().unwrap().to_owned();
    let moving = gaussian([3.95, 2.75, 2.4])
        .view::<f64>()
        .unwrap()
        .to_owned();
    let schedule = PyramidSchedule::powers_of_two(2).expect("schedule");
    let recipe = PyramidRecipe::decimation(schedule.factors().to_vec()).expect("recipe");
    let levels =
        pyramid_levels_from_recipe(&params, &fixed, &moving, &schedule, &recipe).expect("levels");

    let mut bad_level0 = levels.clone();
    bad_level0[0].scale = [2, 1, 1];
    let error = error_text(resident_pyramid_levels(&params, &bad_level0));
    assert!(error.contains("level 0"), "{error}");
    assert!(error.contains("[1, 1, 1]"), "{error}");

    let mut bad_dtype = levels.clone();
    bad_dtype[1].fixed = Voxels::F32(Array3::from_elem((4, 3, 2), 1.0f32));
    let error = error_text(resident_pyramid_levels(&params, &bad_dtype));
    assert!(error.contains("level 1"), "{error}");
    assert!(error.contains("dtype"), "{error}");
    assert!(error.contains("expected f64"), "{error}");

    let base_geometry = SpatialFrame::unit();
    let geometry = |scale: [usize; 3]| {
        SpatialFrame::new(
            [0.0, 0.0, 0.0],
            [scale[0] as f64, scale[1] as f64, scale[2] as f64],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        .expect("geometry")
    };
    let level0 = VolumePyramidLevel::new(
        Voxels::F64(Array3::from_elem((12, 8, 6), 1.0)),
        Voxels::F64(Array3::from_elem((12, 8, 6), 1.0)),
        [1, 1, 1],
        base_geometry,
    )
    .expect("level 0");
    let level1 = VolumePyramidLevel::new(
        Voxels::F64(Array3::from_elem((6, 4, 6), 1.0)),
        Voxels::F64(Array3::from_elem((6, 4, 6), 1.0)),
        [2, 2, 1],
        geometry([2, 2, 1]),
    )
    .expect("level 1");
    let level2 = VolumePyramidLevel::new(
        Voxels::F64(Array3::from_elem((4, 4, 6), 1.0)),
        Voxels::F64(Array3::from_elem((4, 4, 6), 1.0)),
        [3, 2, 1],
        geometry([3, 2, 1]),
    )
    .expect("level 2");
    let error = error_text(resident_pyramid_levels(&params, &[level0, level1, level2]));
    assert!(error.contains("level 2"), "{error}");
    assert!(error.contains("not a cumulative downsample"), "{error}");
}

#[test]
fn the_computed_translation_is_the_same_at_two_cuts() {
    let fixed = point_image([1, 2, 3]);
    let moving = point_image([7, 5, 1]);

    let whole =
        run(&VolumeFitParams::default(), &fixed, &moving, 16).expect("whole-volume registration");
    let cut = run(&VolumeFitParams::default(), &fixed, &moving, 3).expect("cut registration");

    let params = RegisterParams::default();
    for point in [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [8.0, 6.0, 4.0]] {
        assert_eq!(whole.apply(point, &params), cut.apply(point, &params));
    }
}

#[test]
fn the_plan_records_the_supplied_moving_image() {
    let decomposition = plan(&VolumeFitParams::default(), SHAPE, 3).expect("a plan");

    assert_eq!(decomposition.phases.len(), 1);
    assert_eq!(decomposition.phases[0].source_images.len(), 1);
    assert!(
        decomposition.phases[0].source_images[0] > decomposition.n_phases(),
        "the moving image is a supplied input, not an intermediate image"
    );
}
