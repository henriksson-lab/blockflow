// SPDX-License-Identifier: MIT
// Original work for this crate.

// The stochastic registration: the reference's shipped elastix parameter files
// read whole into `fit_stochastic` settings, settings that cannot be honoured
// refused by name, and the pipeline end to end on synthetic OME-Zarr — one
// registration, a two-step chain, and points carried forward and back through
// `PointTransform`.

use std::path::{Path, PathBuf};

use blockflow::ops::align::TransformModel;
use blockflow::voxels::Voxels;
use blockflow_atlas_registration::pipeline::{
    register, register_chain, ChainConfig, Initialisation, Input, RegisterConfig, Stage,
};
use blockflow_atlas_registration::stochastic::{
    elastix_levels, stochastic_from_elastix_map, GainSequence, ParameterScales, SampleOrder,
    StochasticFitParams, StochasticInitialisation, StochasticOptimizer,
};
use blockflow_atlas_registration::volume::write_image;
use blockflow_atlas_registration::{Frame, ParameterMap, PointTransform};
use ndarray::Array3;

/// The settings of the reference's `Resources/Alignment/align_affine.txt`
/// (its comments dropped), as shipped.
const AFFINE: &str = r#"
(FixedInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingInternalImagePixelType "float")
(MovingImageDimension 3)
(Registration "MultiResolutionRegistration")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Interpolator "BSplineInterpolator")
(Metric "AdvancedMattesMutualInformation")
(Optimizer "AdaptiveStochasticGradientDescent")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(Transform "AffineTransform")
(ErodeMask "true" )
(NumberOfResolutions 4)
(HowToCombineTransforms "Compose")
(AutomaticTransformInitialization "true")
(AutomaticScalesEstimation "true")
(WriteTransformParametersEachIteration "false")
(WriteResultImage "true")
(CompressResultImage "false")
(WriteResultImageAfterEachResolution "false")
(ShowExactMetricValue "false")
(MaximumNumberOfIterations 500 )
(NumberOfHistogramBins 32 )
(FixedLimitRangeRatio 0.0)
(MovingLimitRangeRatio 0.0)
(FixedKernelBSplineOrder 3)
(MovingKernelBSplineOrder 3)
(ImageSampler "RandomCoordinate")
(FixedImageBSplineInterpolationOrder 3)
(UseRandomSampleRegion "false")
(NumberOfSpatialSamples 4000 )
(NewSamplesEveryIteration "true")
(CheckNumberOfSamples "true")
(MaximumNumberOfSamplingAttempts 10)
(BSplineInterpolationOrder 3)
(FinalBSplineInterpolationOrder 3)
(DefaultPixelValue 0)
(SP_A 20.0 )
"#;

/// The settings of the reference's `align_bspline.txt`, as shipped.
const BSPLINE: &str = r#"
(FixedInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingInternalImagePixelType "float")
(MovingImageDimension 3)
(Registration "MultiResolutionRegistration")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Interpolator "BSplineInterpolator")
(Metric "AdvancedMattesMutualInformation")
(Optimizer "StandardGradientDescent")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(Transform "BSplineTransform")
(ErodeMask "false" )
(NumberOfResolutions 3)
(FinalGridSpacingInVoxels 25.000000 25.000000 25.000000)
(HowToCombineTransforms "Compose")
(WriteTransformParametersEachIteration "false")
(WriteResultImage "true")
(CompressResultImage "false")
(WriteResultImageAfterEachResolution "false")
(ShowExactMetricValue "false")
(WriteDiffusionFiles "true")
(UseFastAndLowMemoryVersion "true")
(MaximumNumberOfIterations 5000 )
(NumberOfHistogramBins 128 )
(FixedLimitRangeRatio 0.0)
(MovingLimitRangeRatio 0.0)
(FixedKernelBSplineOrder 3)
(MovingKernelBSplineOrder 3)
(ImageSampler "RandomCoordinate")
(FixedImageBSplineInterpolationOrder 1 )
(UseRandomSampleRegion "true")
(SampleRegionSize 50.0 50.0 50.0)
(NumberOfSpatialSamples 10000 )
(NewSamplesEveryIteration "true")
(CheckNumberOfSamples "true")
(MaximumNumberOfSamplingAttempts 10)
(BSplineInterpolationOrder 3)
(FinalBSplineInterpolationOrder 3)
(DefaultPixelValue 0)
(SP_a 10000.0 )
(SP_A 100.0 )
(SP_alpha 0.6 )
"#;

fn read(text: &str) -> blockflow::error::Result<StochasticFitParams> {
    stochastic_from_elastix_map(&ParameterMap::parse(text)?, 7)
}

fn error_text(text: &str) -> String {
    format!("{:?}", read(text).expect_err("refused"))
}

#[test]
fn the_reference_affine_file_maps_to_every_setting() {
    let params = read(AFFINE).expect("the shipped affine file is accepted whole");
    assert_eq!(params.model, TransformModel::Affine);
    assert_eq!(params.levels, elastix_levels(4, 500, None));
    assert_eq!(params.levels[0].fixed_sigma, [4.0; 3]);
    assert_eq!(params.levels[3].moving_sigma, [0.5; 3]);
    assert_eq!((params.fixed_bins, params.moving_bins), (32, 32));
    assert_eq!(
        (params.fixed_kernel_order, params.moving_kernel_order),
        (3, 3)
    );
    assert_eq!(
        (
            params.fixed_limit_range_ratio,
            params.moving_limit_range_ratio
        ),
        (0.0, 0.0)
    );
    assert_eq!(
        (params.fixed_order, params.moving_order),
        (SampleOrder::Cubic, SampleOrder::Cubic)
    );
    assert_eq!(params.samples, 4000);
    assert!(params.new_samples_every_iteration);
    assert_eq!(params.sample_region, None);
    assert_eq!(
        (params.minimum_valid_fraction, params.sampling_attempts),
        (0.25, 10)
    );
    match params.optimizer {
        StochasticOptimizer::AdaptiveAutomatic {
            big_a,
            estimation,
            initial_time,
        } => {
            assert_eq!((big_a, initial_time), (20.0, 0.0));
            // elastix's defaults, as its log of the reference run prints them.
            assert_eq!(estimation.max_step, 1.0);
            assert_eq!(estimation.gradient_measurements, 3);
            assert_eq!(estimation.jacobian_measurements, 1000);
            assert_eq!(estimation.exact_gradient_samples, 100_000);
            assert_eq!(estimation.sigmoid_scale_factor, 0.1);
        }
        other => panic!("expected automatic estimation, got {other:?}"),
    }
    assert_eq!(
        params.scales,
        ParameterScales::Automatic { samples: 10_000 }
    );
    assert_eq!(
        params.initialisation,
        StochasticInitialisation::GeometricCentre
    );
    assert_eq!(params.seed, 7);
}

#[test]
fn the_reference_bspline_file_maps_to_every_setting() {
    let params = read(BSPLINE).expect("the shipped B-spline file is accepted whole");
    assert_eq!(params.model, TransformModel::BSpline);
    assert_eq!(params.levels, elastix_levels(3, 5000, Some([25.0; 3])));
    assert_eq!(params.levels[0].grid_spacing, [100.0; 3]);
    assert_eq!(params.levels[2].grid_spacing, [25.0; 3]);
    assert_eq!((params.fixed_bins, params.moving_bins), (128, 128));
    assert_eq!(
        (params.fixed_order, params.moving_order),
        (SampleOrder::Linear, SampleOrder::Cubic)
    );
    assert_eq!(params.samples, 10_000);
    assert_eq!(params.sample_region, Some([50.0; 3]));
    assert_eq!(
        params.optimizer,
        StochasticOptimizer::Standard(GainSequence {
            a: 10_000.0,
            big_a: 100.0,
            alpha: 0.6
        })
    );
    assert_eq!(params.scales, ParameterScales::Unit);
    assert_eq!(params.initialisation, StochasticInitialisation::Identity);
}

#[test]
fn settings_that_cannot_be_honoured_are_refused_by_name() {
    let cases = [
        (
            "(Metric \"AdvancedMattesMutualInformation\")",
            "(Metric \"AdvancedNormalizedCorrelation\")",
            "Metric",
        ),
        (
            "(ImageSampler \"RandomCoordinate\")",
            "(ImageSampler \"Grid\")",
            "ImageSampler",
        ),
        (
            "(Transform \"AffineTransform\")",
            "(Transform \"EulerTransform\")",
            "EulerTransform",
        ),
        (
            "(BSplineInterpolationOrder 3)",
            "(BSplineInterpolationOrder 2)",
            "BSplineInterpolationOrder",
        ),
        (
            "(ShowExactMetricValue \"false\")",
            "(SomeSettingNobodyKnows 3)",
            "SomeSettingNobodyKnows",
        ),
        (
            "(SP_A 20.0 )",
            "(SP_A 20.0 )\n(Metric1Weight 0.5)",
            "Metric1Weight",
        ),
        (
            "(AutomaticScalesEstimation \"true\")",
            "(AutomaticScalesEstimation \"false\")",
            "Scales",
        ),
        (
            "(SP_A 20.0 )",
            "(SP_A 20.0 )\n(ASGDParameterEstimationMethod \"DisplacementDistribution\")",
            "ASGDParameterEstimationMethod",
        ),
    ];
    for (from, to, name) in cases {
        assert!(AFFINE.contains(from), "{from}");
        let error = error_text(&AFFINE.replace(from, to));
        assert!(error.contains(name), "{name}: {error}");
    }
    let error = error_text(&BSPLINE.replace(
        "(HowToCombineTransforms \"Compose\")",
        "(HowToCombineTransforms \"Add\")",
    ));
    assert!(error.contains("HowToCombineTransforms"), "{error}");
}

// ---------------------------------------------------------- end to end --

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "atlas-registration-stochastic-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch directory");
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A scene with structure at several scales, in physical units over about
/// 64 × 72 × 80.
fn scene(p: [f64; 3]) -> f64 {
    let mut value = 50.0;
    let mut state: u64 = 0x2026_1019;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let extent = [64.0, 72.0, 80.0];
    for blob in 0..24 {
        let centre: [f64; 3] = std::array::from_fn(|a| extent[a] * (0.2 + 0.6 * next()));
        let width = if blob < 4 { 14.0 } else { 3.0 + 4.0 * next() };
        let height = 200.0 + 800.0 * next();
        let d: f64 = (0..3).map(|a| ((p[a] - centre[a]) / width).powi(2)).sum();
        value += height * (-0.5 * d).exp();
    }
    value
}

/// The scene moved by `shift` (a point `p` of the written image shows the
/// scene at `p - shift`), on `shape` voxels of `spacing`.
fn write_scene(root: &Path, spacing: f64, shape: [usize; 3], shift: [f64; 3]) -> Frame {
    let frame = Frame::new([spacing; 3], [0.0; 3]).unwrap();
    let image = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(z, y, x)| {
        let p = frame.to_physical([z as f64, y as f64, x as f64]);
        scene(std::array::from_fn(|a| p[a] - shift[a])).round() as u16
    });
    write_image(root, &Voxels::from(image), &frame, Some("micrometer")).unwrap();
    frame
}

/// The affine file, shortened: fewer levels and iterations for a small image.
fn quick_affine(seed: u64) -> StochasticFitParams {
    let mut params = read(AFFINE).unwrap();
    params.levels = elastix_levels(2, 200, None);
    params.samples = 2000;
    params.fixed_bins = 48;
    params.moving_bins = 48;
    params.seed = seed;
    params
}

fn quick_translation(seed: u64) -> StochasticFitParams {
    StochasticFitParams {
        model: TransformModel::Translation,
        scales: ParameterScales::Unit,
        ..quick_affine(seed)
    }
}

fn input(path: PathBuf) -> Input {
    Input {
        path,
        channel: 0,
        level: None,
    }
}

#[test]
fn a_stochastic_registration_finds_a_known_shift_and_points_follow_it() {
    let scratch = Scratch::new("single");
    let shift = [2.5, -3.0, 4.0];
    let fixed = write_scene(&scratch.join("fixed.zarr"), 1.0, [64, 72, 80], [0.0; 3]);
    // The moving image is coarser and larger, so no sample leaves it.
    let moving = write_scene(
        &scratch.join("moving.zarr"),
        2.0,
        [40, 44, 48],
        [shift[0] + 8.0, shift[1] + 8.0, shift[2] + 8.0],
    );
    let summary = register(&RegisterConfig {
        fixed: input(scratch.join("fixed.zarr")),
        moving: input(scratch.join("moving.zarr")),
        resolution: Some([2.0; 3]),
        stages: vec![Stage::Stochastic(quick_affine(3))],
        out: scratch.join("registration"),
        warped: None,
        warp_source: None,
        nearest: false,
        block: 16,
        workers: 2,
        warped_stages: false,
        initialisation: Default::default(),
    })
    .expect("registers");
    let stage = &summary.stages[0];
    assert!(stage.similarity.unwrap().correlation > 0.95, "{stage:?}");

    // Fixed level-0 voxels to moving level-0 voxels: the scene at fixed
    // physical `p` is at moving physical `p + shift + 8`.
    let transform = PointTransform::from_registration_dir(scratch.join("registration")).unwrap();
    assert_eq!(transform.steps(), 1);
    assert_eq!(transform.sample_shape(), [64, 72, 80]);
    assert_eq!(transform.atlas_shape(), [40, 44, 48]);
    let probes = [[20.0, 30.0, 40.0], [40.0, 20.0, 60.0], [32.0, 50.0, 25.0]];
    let moved = transform.apply(&probes);
    for (probe, got) in probes.iter().zip(&moved) {
        let want = moving.to_index(std::array::from_fn(|a| {
            fixed.to_physical(*probe)[a] + shift[a] + 8.0
        }));
        for a in 0..3 {
            // Half a moving voxel is a quarter of a lattice voxel's error.
            assert!(
                (got[a] - want[a]).abs() < 0.25,
                "{probe:?} -> {got:?}, want {want:?}"
            );
        }
    }
    // And back.
    let back = transform.apply_inverse(&moved);
    for (probe, got) in probes.iter().zip(&back) {
        let got = got.expect("the inverse converges");
        for a in 0..3 {
            assert!(
                (got[a] - probe[a]).abs() < 0.01,
                "{probe:?} came back as {got:?}"
            );
        }
    }
}

#[test]
fn a_two_step_chain_composes_its_registrations() {
    let scratch = Scratch::new("chain");
    let first = [1.5, -2.0, 2.5];
    let second = [-2.0, 3.0, 1.0];
    // Signal (sample) -> autofluorescence -> atlas, each moved from the last
    // and each larger than the one before, so that no sample leaves it.
    write_scene(&scratch.join("signal.zarr"), 1.0, [64, 72, 80], [0.0; 3]);
    write_scene(
        &scratch.join("auto.zarr"),
        1.0,
        [72, 80, 88],
        first.map(|v| v + 4.0),
    );
    let total: [f64; 3] = std::array::from_fn(|a| first[a] + second[a]);
    write_scene(
        &scratch.join("atlas.zarr"),
        1.0,
        [80, 88, 96],
        total.map(|v| v + 8.0),
    );
    let summaries = register_chain(&ChainConfig {
        images: vec![
            input(scratch.join("signal.zarr")),
            input(scratch.join("auto.zarr")),
            input(scratch.join("atlas.zarr")),
        ],
        // Translations: this test is about composing the steps, and an affine
        // fit of a scene this smooth shrinks by 1 to 2 per cent (the Parzen
        // estimate of mutual information prefers it; elastix's would too).
        steps: vec![
            vec![Stage::Stochastic(quick_translation(5))],
            vec![Stage::Stochastic(quick_translation(6))],
        ],
        resolution: Some([1.0; 3]),
        out: scratch.join("chain"),
        block: 16,
        workers: 2,
        warped_stages: false,
        initialisation: Default::default(),
        warped: false,
    })
    .expect("the chain registers");
    assert_eq!(summaries.len(), 2);
    let transform = PointTransform::from_registration_dir(scratch.join("chain")).unwrap();
    assert_eq!(transform.steps(), 2);
    let probes = [[24.0, 30.0, 40.0], [36.0, 40.0, 30.0]];
    for (probe, got) in probes.iter().zip(transform.apply(&probes)) {
        for a in 0..3 {
            let want = probe[a] + total[a] + 8.0;
            assert!(
                (got[a] - want).abs() < 0.15,
                "{probe:?} -> {got:?}, axis {a} want {want}"
            );
        }
    }
    // Each step on its own is a registration directory too.
    let step = PointTransform::from_registration_dir(scratch.join("chain").join("step0")).unwrap();
    let got = step.apply_one(probes[0]);
    for a in 0..3 {
        assert!(
            (got[a] - probes[0][a] - first[a] - 4.0).abs() < 0.1,
            "{got:?}"
        );
    }
}

/// A crop of the scene at `scene_origin` (scene units), on `shape` voxels of
/// `spacing`, whose frame puts its first voxel at `translation`.
fn write_crop(
    root: &Path,
    spacing: f64,
    shape: [usize; 3],
    translation: [f64; 3],
    scene_origin: [f64; 3],
) -> Frame {
    let frame = Frame::new([spacing; 3], translation).unwrap();
    let image = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(z, y, x)| {
        let index = [z, y, x];
        scene(std::array::from_fn(|a| {
            scene_origin[a] + index[a] as f64 * spacing
        }))
        .round() as u16
    });
    write_image(root, &Voxels::from(image), &frame, Some("micrometer")).unwrap();
    frame
}

/// A partial mosaic: the fixed image is a strip near one side of the moving
/// one (as nine tiles of a whole-brain mosaic are of its autofluorescence).
/// Each initialisation starts where it says: the physical frames as they
/// stand, the two geometric centres on each other, or the frames moved by
/// the translation given; and from the frames (when the strip's frame places
/// it) or from a translation a few micrometres off (when it does not), the
/// fit lands on the strip. (This scene's few broad blobs also lead a fit
/// started at the centres to it; a real strip of brain does not.)
#[test]
fn a_partial_mosaic_is_placed_by_its_frame_or_an_initial_translation() {
    let scratch = Scratch::new("initialise");
    // the whole scene, coarser, at the origin, in a field twice its size
    // (as the autofluorescence field is larger than the brain)
    write_crop(
        &scratch.join("moving.zarr"),
        2.0,
        [48, 72, 80],
        [0.0; 3],
        [0.0; 3],
    );
    // the strip: scene z 0..64, y 8..40, x 48..76
    let strip = [0.0, 8.0, 48.0];
    let shape = [64, 32, 28];
    write_crop(&scratch.join("placed.zarr"), 1.0, shape, strip, strip);
    write_crop(&scratch.join("unplaced.zarr"), 1.0, shape, [0.0; 3], strip);
    // Per probe, the error in moving voxels of the registration from `fixed`
    // started as `initialisation`; with `fit` false, of the start itself.
    let run = |fixed: &str, initialisation: Initialisation, out: &str, fit: bool| {
        let mut params = quick_translation(7);
        if !fit {
            // one step of a vanishing gain: the start, as written
            params.levels = elastix_levels(1, 1, None);
            params.optimizer = StochasticOptimizer::Standard(GainSequence {
                a: 1e-12,
                big_a: 1.0,
                alpha: 1.0,
            });
        }
        register(&RegisterConfig {
            fixed: input(scratch.join(fixed)),
            moving: input(scratch.join("moving.zarr")),
            resolution: Some([2.0; 3]),
            stages: vec![Stage::Stochastic(params)],
            out: scratch.join(out),
            warped: None,
            warp_source: None,
            nearest: false,
            block: 16,
            workers: 2,
            warped_stages: false,
            initialisation,
        })
        .expect("registers");
        // fixed voxel -> moving voxel; the truth is (scene point) / 2
        let transform = PointTransform::from_registration_dir(scratch.join(out)).unwrap();
        let probes = [[20.0, 10.0, 10.0], [40.0, 20.0, 15.0], [32.0, 16.0, 20.0]];
        let errors: Vec<[f64; 3]> = transform
            .apply(&probes)
            .iter()
            .zip(&probes)
            .map(|(got, probe)| std::array::from_fn(|a| got[a] - (strip[a] + probe[a]) / 2.0))
            .collect();
        errors
    };
    let worst = |errors: &[[f64; 3]]| {
        errors
            .iter()
            .flat_map(|e| e.iter().map(|v| v.abs()))
            .fold(0.0, f64::max)
    };
    // 3 um off in each axis, as a stage reading might be
    let guess = [strip[0] + 3.0, strip[1] - 3.0, strip[2] + 3.0];

    // Where each starts: the frames as they stand (exact), the centres
    // (the strip's centre on the field's), the translation (3 um off).
    let start = run(
        "placed.zarr",
        Initialisation::Physical,
        "start-physical",
        false,
    );
    assert!(worst(&start) < 1e-6, "{start:?}");
    let start = run("placed.zarr", Initialisation::Centre, "start-centre", false);
    let centres: [f64; 3] = std::array::from_fn(|a| {
        let field = [96.0, 144.0, 160.0][a] - 2.0;
        let fixed = [shape[0] as f64, shape[1] as f64, shape[2] as f64][a] - 1.0;
        (field / 2.0 - (strip[a] + fixed / 2.0)) / 2.0
    });
    for e in &start {
        for a in 0..3 {
            assert!((e[a] - centres[a]).abs() < 1e-6, "{start:?} vs {centres:?}");
        }
    }
    let start = run(
        "unplaced.zarr",
        Initialisation::Translation(guess),
        "start-translation",
        false,
    );
    for e in &start {
        assert!(
            (e[0] - 1.5).abs() < 1e-6 && (e[1] + 1.5).abs() < 1e-6 && (e[2] - 1.5).abs() < 1e-6,
            "{start:?}"
        );
    }

    // And where they end: on the strip.
    let physical = worst(&run(
        "placed.zarr",
        Initialisation::Physical,
        "physical",
        true,
    ));
    let translated = worst(&run(
        "unplaced.zarr",
        Initialisation::Translation(guess),
        "translation",
        true,
    ));
    assert!(
        physical < 0.25,
        "physical frames: off by {physical} moving voxels"
    );
    assert!(
        translated < 0.25,
        "initial translation: off by {translated} moving voxels"
    );
    let sidecar: serde_json::Value = serde_json::from_slice(
        &std::fs::read(scratch.join("translation").join("registration.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        sidecar["initialisation"],
        serde_json::json!({ "translation": guess })
    );
    // the parameters' own initialisation is the default
    let parsed = Initialisation::parse("center").unwrap();
    assert_eq!(parsed, Initialisation::Centre);
    assert!(Initialisation::parse("landmarks").is_err());
}

/// The same strip with nothing placing it: the placement search finds the
/// translation from the images, the fit started there lands on the strip,
/// and the sidecar records the search. `parse("search")` selects it.
#[test]
fn a_partial_mosaic_is_placed_by_a_correlation_search() {
    let scratch = Scratch::new("search");
    write_crop(
        &scratch.join("moving.zarr"),
        2.0,
        [48, 72, 80],
        [0.0; 3],
        [0.0; 3],
    );
    let strip = [0.0, 8.0, 48.0];
    write_crop(
        &scratch.join("unplaced.zarr"),
        1.0,
        [64, 32, 28],
        [0.0; 3],
        strip,
    );
    let search = match Initialisation::parse("search").unwrap() {
        Initialisation::Search(params) => params,
        other => panic!("{other:?}"),
    };
    let placement = blockflow_atlas_registration::pipeline::place(
        &input(scratch.join("unplaced.zarr")),
        &input(scratch.join("moving.zarr")),
        Some([2.0; 3]),
        &search,
    )
    .expect("placed");
    for a in 0..3 {
        assert!(
            (placement.translation()[a] - strip[a]).abs() < 1.0,
            "{:?} vs {strip:?}",
            placement.translation()
        );
    }
    register(&RegisterConfig {
        fixed: input(scratch.join("unplaced.zarr")),
        moving: input(scratch.join("moving.zarr")),
        resolution: Some([2.0; 3]),
        stages: vec![Stage::Stochastic(quick_translation(7))],
        out: scratch.join("searched"),
        warped: None,
        warp_source: None,
        nearest: false,
        block: 16,
        workers: 2,
        warped_stages: false,
        initialisation: Initialisation::Search(search),
    })
    .expect("registers");
    let transform = PointTransform::from_registration_dir(scratch.join("searched")).unwrap();
    let probes = [[20.0, 10.0, 10.0], [40.0, 20.0, 15.0], [32.0, 16.0, 20.0]];
    for (got, probe) in transform.apply(&probes).iter().zip(&probes) {
        for a in 0..3 {
            let truth = (strip[a] + probe[a]) / 2.0;
            assert!(
                (got[a] - truth).abs() < 0.25,
                "{got:?} vs {probe:?} (truth {truth})"
            );
        }
    }
    let sidecar: serde_json::Value = serde_json::from_slice(
        &std::fs::read(scratch.join("searched").join("registration.json")).unwrap(),
    )
    .unwrap();
    let recorded = &sidecar["initialisation"]["search"];
    assert_eq!(
        recorded["translation"],
        serde_json::json!(placement.translation())
    );
    assert!(recorded["ncc"].as_f64().unwrap() > 0.9, "{recorded}");
    assert!(sidecar["timings"]["place_s"].is_number());
}

#[test]
fn a_directory_that_is_not_a_registration_is_refused() {
    let scratch = Scratch::new("empty");
    let error = PointTransform::from_registration_dir(scratch.join("")).unwrap_err();
    assert!(
        format!("{error:?}").contains("not a registration directory"),
        "{error:?}"
    );
}
