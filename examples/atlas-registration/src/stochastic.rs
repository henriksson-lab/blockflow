// SPDX-License-Identifier: MIT
// Original work for this crate.

//! elastix parameter files read into [`StochasticFitParams`]: Mattes mutual
//! information, a random-coordinate sampler and elastix's stochastic gradient
//! descent, as `blockflow::ops::align::fit_stochastic` implements them.
//!
//! Every key of a file is either mapped, accepted as having no effect on the
//! fitted transform (it configures elastix's own output), or refused by name.
//! [`STOCHASTIC_SETTING_SUPPORT`] is the table. Values elastix reads per
//! resolution may be given once (for every level) or once per level.
//!
//! **Lengths.** The reference runs elastix on images whose spacing is 1, so its
//! physical lengths (`SampleRegionSize`, `MaximumStepLength`,
//! `FinalGridSpacingInPhysicalUnits`) are voxels of the registration lattice.
//! They are read as lattice voxels here, so a parameter file means what it
//! meant there.

use blockflow::error::{Error, Result};
use blockflow::ops::align::TransformModel;
pub use blockflow::ops::align::{
    elastix_levels, fit_stochastic, AffineMap, AutomaticGain, BSplineLayout, GainSequence,
    ParameterScales, RelativeTolerance, SampleOrder, SigmoidTime, StochasticFit,
    StochasticFitParams, StochasticInitialisation, StochasticLevel, StochasticLevelReport,
    StochasticOptimizer,
};

use crate::compute::ElastixSettingStatus;
use crate::parameters::ParameterMap;

/// How one elastix setting is treated by [`stochastic_from_elastix_map`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StochasticSettingSupport {
    pub setting: &'static str,
    pub status: ElastixSettingStatus,
    pub native: &'static str,
}

const fn row(
    setting: &'static str,
    status: ElastixSettingStatus,
    native: &'static str,
) -> StochasticSettingSupport {
    StochasticSettingSupport {
        setting,
        status,
        native,
    }
}

use ElastixSettingStatus::{Approximation, Supported, Unsupported};

/// The audit surface of the stochastic mapping. `Unsupported` rows are refused
/// whatever their value; others may still refuse particular values, by name.
pub const STOCHASTIC_SETTING_SUPPORT: &[StochasticSettingSupport] = &[
    row(
        "Registration",
        Supported,
        "`MultiResolutionRegistration` only",
    ),
    row(
        "Transform",
        Supported,
        "Translation, Affine, BSpline, RecursiveBSpline",
    ),
    row(
        "Metric",
        Supported,
        "`AdvancedMattesMutualInformation` (one metric)",
    ),
    row("NumberOfHistogramBins", Supported, "fixed and moving bins"),
    row("NumberOfFixedHistogramBins", Supported, "fixed bins"),
    row("NumberOfMovingHistogramBins", Supported, "moving bins"),
    row(
        "FixedKernelBSplineOrder",
        Supported,
        "fixed Parzen window order 0..=3",
    ),
    row(
        "MovingKernelBSplineOrder",
        Supported,
        "moving Parzen window order 1..=3",
    ),
    row(
        "FixedLimitRangeRatio",
        Supported,
        "histogram range widened by the ratio",
    ),
    row(
        "MovingLimitRangeRatio",
        Supported,
        "histogram range widened by the ratio",
    ),
    row(
        "UseFastAndLowMemoryVersion",
        Supported,
        "the same derivative, computed per sample",
    ),
    row("UseJacobianPreconditioning", Supported, "`false` only"),
    row("FiniteDifferenceDerivative", Supported, "`false` only"),
    row(
        "UseMultiThreadingForMetrics",
        Supported,
        "always multi-threaded; same answer",
    ),
    row(
        "Optimizer",
        Supported,
        "AdaptiveStochasticGradientDescent, StandardGradientDescent",
    ),
    row(
        "MaximumNumberOfIterations",
        Supported,
        "iterations per level",
    ),
    row(
        "AutomaticParameterEstimation",
        Supported,
        "Klein et al. 2009, `Original` method",
    ),
    row(
        "ASGDParameterEstimationMethod",
        Supported,
        "`Original` only",
    ),
    row(
        "MaximumStepLength",
        Supported,
        "δ of the estimation, lattice voxels",
    ),
    row(
        "MaximumStepLengthRatio",
        Supported,
        "multiplies the default δ of 1 voxel",
    ),
    row("NumberOfGradientMeasurements", Supported, "0 is 3"),
    row(
        "NumberOfJacobianMeasurements",
        Supported,
        "grid points for the Jacobian terms",
    ),
    row(
        "NumberOfSamplesForExactGradient",
        Supported,
        "grid points of the exact gradient",
    ),
    row(
        "SigmoidScaleFactor",
        Supported,
        "ω factor of the estimation",
    ),
    row("SigmoidInitialTime", Supported, "initial time"),
    row("SigmoidMax", Supported, "with a fixed gain"),
    row("SigmoidMin", Supported, "with a fixed gain"),
    row("SigmoidScale", Supported, "with a fixed gain"),
    row(
        "UseAdaptiveStepSizes",
        Supported,
        "`false` makes the time the iteration count",
    ),
    row("UseConstantStep", Supported, "`false` only"),
    row("SP_a", Supported, "gain a"),
    row("SP_A", Supported, "gain A"),
    row("SP_alpha", Supported, "gain alpha"),
    row(
        "AutomaticScalesEstimation",
        Supported,
        "`true` only for translation and affine",
    ),
    row(
        "AutomaticTransformInitialization",
        Supported,
        "geometric centres, or identity",
    ),
    row(
        "AutomaticTransformInitializationMethod",
        Supported,
        "`GeometricalCenter` only",
    ),
    row("NumberOfResolutions", Supported, "levels"),
    row(
        "ImagePyramidSchedule",
        Supported,
        "per-level factors, both images",
    ),
    row("FixedImagePyramidSchedule", Supported, "per-level factors"),
    row("MovingImagePyramidSchedule", Supported, "per-level factors"),
    row(
        "FixedImagePyramid",
        Supported,
        "Smoothing exactly; Recursive approximated",
    ),
    row(
        "MovingImagePyramid",
        Supported,
        "Smoothing exactly; Recursive approximated",
    ),
    row(
        "Interpolator",
        Supported,
        "Linear, or BSpline of order 1 or 3",
    ),
    row("BSplineInterpolationOrder", Supported, "1 or 3"),
    row("FixedImageBSplineInterpolationOrder", Supported, "1 or 3"),
    row(
        "ImageSampler",
        Supported,
        "`RandomCoordinate`; `Random` approximated by it",
    ),
    row("NumberOfSpatialSamples", Supported, "samples per iteration"),
    row(
        "NewSamplesEveryIteration",
        Supported,
        "fresh or fixed samples",
    ),
    row(
        "UseRandomSampleRegion",
        Supported,
        "one random box per iteration",
    ),
    row("SampleRegionSize", Supported, "box size, lattice voxels"),
    row("CheckNumberOfSamples", Supported, "valid-sample check"),
    row(
        "RequiredRatioOfValidSamples",
        Supported,
        "its threshold (0.25)",
    ),
    row(
        "MaximumNumberOfSamplingAttempts",
        Supported,
        "draws before an error",
    ),
    row(
        "FinalGridSpacingInVoxels",
        Supported,
        "B-spline spacing, lattice voxels",
    ),
    row(
        "FinalGridSpacingInPhysicalUnits",
        Supported,
        "read as lattice voxels",
    ),
    row("GridSpacingSchedule", Supported, "per-level factors"),
    row("BSplineTransformSplineOrder", Supported, "3 only"),
    row("UseCyclicTransform", Supported, "`false` only"),
    row("PassiveEdgeWidth", Supported, "0 only"),
    row("HowToCombineTransforms", Supported, "`Compose` only"),
    row(
        "InitialTransformParametersFileName",
        Supported,
        "`NoInitialTransform` only",
    ),
    row("Metric0Weight", Unsupported, "one metric per stage"),
    row("Metric1Weight", Unsupported, "one metric per stage"),
    row("Scales", Unsupported, "explicit scales are not read"),
    row(
        "ErodeMask",
        Approximation,
        "no masks are used, so it has no effect",
    ),
];

/// Keys that configure elastix's own outputs (result images, logs, files)
/// and do not change the fitted transform.
const OUTPUT_ONLY_KEYS: &[&str] = &[
    "CompressResultImage",
    "DefaultPixelValue",
    "FinalBSplineInterpolationOrder",
    "FixedImageDimension",
    "FixedInternalImagePixelType",
    "MovingImageDimension",
    "MovingInternalImagePixelType",
    "ResampleInterpolator",
    "Resampler",
    "ResultImageFormat",
    "ResultImagePixelType",
    "ShowExactMetricValue",
    "UseDirectionCosines",
    "WriteDiffusionFiles",
    "WriteResultImage",
    "WriteResultImageAfterEachResolution",
    "WriteTransformParametersEachIteration",
    "WriteIterationInfo",
];

fn refuse(text: String) -> Error {
    Error::InvalidArgument(format!("elastix parameter file (stochastic): {text}"))
}

/// Whether a parameter file asks for what [`stochastic_from_elastix_map`]
/// implements: one of elastix's stochastic optimisers.
pub fn is_stochastic(map: &ParameterMap) -> Result<bool> {
    Ok(matches!(
        map.optional_string("Optimizer")?,
        Some("AdaptiveStochasticGradientDescent" | "StandardGradientDescent")
    ))
}

/// Registration settings from one parsed elastix parameter file, with the
/// sampler seeded by `seed`.
pub fn stochastic_from_elastix_map(map: &ParameterMap, seed: u64) -> Result<StochasticFitParams> {
    for key in map.keys() {
        if OUTPUT_ONLY_KEYS.contains(&key) {
            continue;
        }
        match STOCHASTIC_SETTING_SUPPORT.iter().find(|r| r.setting == key) {
            None => {
                return Err(refuse(format!(
                    "`{key}` is not mapped; refusing rather than ignoring it"
                )))
            }
            Some(r) if r.status == Unsupported => {
                return Err(refuse(format!("`{key}` is not supported: {}", r.native)))
            }
            Some(_) => {}
        }
    }
    let text = |key: &str| map.optional_string(key);
    let flag = |key: &str, absent: bool| map.flag(key, absent);
    let mut params = StochasticFitParams {
        seed,
        ..StochasticFitParams::default()
    };

    if let Some(registration) = text("Registration")? {
        if registration != "MultiResolutionRegistration" {
            return Err(refuse(format!(
                "`Registration` `{registration}`: one metric per stage"
            )));
        }
    }
    params.model = match text("Transform")? {
        Some("TranslationTransform") => TransformModel::Translation,
        Some("AffineTransform") => TransformModel::Affine,
        Some("BSplineTransform" | "RecursiveBSplineTransform") => TransformModel::BSpline,
        Some(other) => return Err(refuse(format!("`Transform` `{other}` is not implemented"))),
        None => return Err(refuse("`Transform` is required".to_string())),
    };
    match map.values("Metric") {
        Some([one]) if unquote(one) == "AdvancedMattesMutualInformation" => {}
        Some(values) => {
            return Err(refuse(format!(
                "`Metric` {values:?}: only `AdvancedMattesMutualInformation` is implemented"
            )))
        }
        None => return Err(refuse("`Metric` is required".to_string())),
    }
    for (key, wanted) in [
        ("UseJacobianPreconditioning", false),
        ("FiniteDifferenceDerivative", false),
        ("UseConstantStep", false),
        ("UseCyclicTransform", false),
    ] {
        if flag(key, wanted)? != wanted {
            return Err(refuse(format!("`{key}` must be \"{wanted}\"")));
        }
    }
    if let Some(combine) = text("HowToCombineTransforms")? {
        if combine != "Compose" {
            return Err(refuse(format!(
                "`HowToCombineTransforms` `{combine}`: `Compose` only"
            )));
        }
    }
    if let Some(initial) = text("InitialTransformParametersFileName")? {
        if initial != "NoInitialTransform" {
            return Err(refuse(
                "`InitialTransformParametersFileName`: stages are chained by the caller".into(),
            ));
        }
    }

    // ---- levels: count, smoothing, iterations, B-spline spacing
    let resolutions = match map.values("NumberOfResolutions") {
        Some(_) => map.usize("NumberOfResolutions")?,
        None => 3,
    };
    if resolutions == 0 {
        return Err(refuse("`NumberOfResolutions` is 0".to_string()));
    }
    for key in ["FixedImagePyramid", "MovingImagePyramid"] {
        if let Some(pyramid) = text(key)? {
            let side = &key[..key.len() - "ImagePyramid".len()];
            let known = [
                format!("{side}SmoothingImagePyramid"),
                format!("{side}RecursiveImagePyramid"),
            ];
            if !known.iter().any(|k| k == pyramid) {
                return Err(refuse(format!("`{key}` `{pyramid}` is not implemented")));
            }
        }
    }
    let default_factors: Vec<[f64; 3]> = (0..resolutions)
        .map(|level| [(1u64 << (resolutions - 1 - level)) as f64; 3])
        .collect();
    let schedule = |key: &str| -> Result<Option<Vec<[f64; 3]>>> {
        let Some(values) = map.values(key) else {
            return Ok(None);
        };
        let numbers = map.floats(key)?;
        if numbers.len() != 3 * resolutions || values.is_empty() {
            return Err(refuse(format!(
                "`{key}` has {} values; {resolutions} levels of 3 are needed",
                numbers.len()
            )));
        }
        Ok(Some(
            numbers.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
        ))
    };
    let shared = schedule("ImagePyramidSchedule")?;
    let fixed_factors = schedule("FixedImagePyramidSchedule")?
        .or_else(|| shared.clone())
        .unwrap_or_else(|| default_factors.clone());
    let moving_factors = schedule("MovingImagePyramidSchedule")?
        .or(shared)
        .unwrap_or_else(|| default_factors.clone());
    let iterations = per_level_usize(map, "MaximumNumberOfIterations", resolutions, 500)?;

    let grid_spacing = if params.model == TransformModel::BSpline {
        let final_spacing = match (
            map.values("FinalGridSpacingInVoxels"),
            map.values("FinalGridSpacingInPhysicalUnits"),
        ) {
            (Some(_), Some(_)) => {
                return Err(refuse(
                    "give one of the two `FinalGridSpacing…` keys".to_string(),
                ))
            }
            (Some(_), None) => one_or_three(map, "FinalGridSpacingInVoxels")?,
            (None, Some(_)) => one_or_three(map, "FinalGridSpacingInPhysicalUnits")?,
            (None, None) => [16.0; 3],
        };
        if final_spacing.iter().any(|s| !(*s > 0.0)) {
            return Err(refuse(format!(
                "B-spline grid spacing {final_spacing:?} must be positive"
            )));
        }
        let factors = match map.values("GridSpacingSchedule") {
            None => default_factors.clone(),
            Some(_) => {
                let numbers = map.floats("GridSpacingSchedule")?;
                match numbers.len() {
                    n if n == resolutions => numbers.iter().map(|f| [*f; 3]).collect(),
                    n if n == 3 * resolutions => {
                        numbers.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
                    }
                    n => {
                        return Err(refuse(format!(
                            "`GridSpacingSchedule` has {n} values for {resolutions} levels"
                        )))
                    }
                }
            }
        };
        if map.values("PassiveEdgeWidth").is_some() && map.usize("PassiveEdgeWidth")? != 0 {
            return Err(refuse("`PassiveEdgeWidth` must be 0".to_string()));
        }
        if map.values("BSplineTransformSplineOrder").is_some()
            && map.usize("BSplineTransformSplineOrder")? != 3
        {
            return Err(refuse(
                "`BSplineTransformSplineOrder` must be 3".to_string(),
            ));
        }
        factors
            .iter()
            .map(|f| std::array::from_fn(|a| final_spacing[a] * f[a]))
            .collect()
    } else {
        for key in [
            "FinalGridSpacingInVoxels",
            "FinalGridSpacingInPhysicalUnits",
            "GridSpacingSchedule",
        ] {
            if map.values(key).is_some() {
                return Err(refuse(format!("`{key}` applies to a B-spline transform")));
            }
        }
        vec![[0.0; 3]; resolutions]
    };
    params.levels = (0..resolutions)
        .map(|level| StochasticLevel {
            fixed_sigma: fixed_factors[level].map(|f| 0.5 * f),
            moving_sigma: moving_factors[level].map(|f| 0.5 * f),
            iterations: iterations[level],
            grid_spacing: grid_spacing[level],
        })
        .collect();

    // ---- metric
    let bins = map
        .values("NumberOfHistogramBins")
        .map(|_| map.usize("NumberOfHistogramBins"))
        .transpose()?
        .unwrap_or(32);
    params.fixed_bins = optional_usize(map, "NumberOfFixedHistogramBins")?.unwrap_or(bins);
    params.moving_bins = optional_usize(map, "NumberOfMovingHistogramBins")?.unwrap_or(bins);
    params.fixed_kernel_order = optional_usize(map, "FixedKernelBSplineOrder")?.unwrap_or(0);
    params.moving_kernel_order = optional_usize(map, "MovingKernelBSplineOrder")?.unwrap_or(3);
    params.fixed_limit_range_ratio = optional_f64(map, "FixedLimitRangeRatio")?.unwrap_or(0.01);
    params.moving_limit_range_ratio = optional_f64(map, "MovingLimitRangeRatio")?.unwrap_or(0.01);

    // ---- interpolation
    params.moving_order = match text("Interpolator")? {
        None | Some("BSplineInterpolator") | Some("BSplineInterpolatorFloat") => order(
            optional_usize(map, "BSplineInterpolationOrder")?.unwrap_or(1),
            "BSplineInterpolationOrder",
        )?,
        Some("LinearInterpolator") => SampleOrder::Linear,
        Some(other) => {
            return Err(refuse(format!(
                "`Interpolator` `{other}` is not implemented"
            )))
        }
    };
    params.fixed_order = order(
        optional_usize(map, "FixedImageBSplineInterpolationOrder")?.unwrap_or(1),
        "FixedImageBSplineInterpolationOrder",
    )?;

    // ---- sampler
    match text("ImageSampler")? {
        None | Some("RandomCoordinate") | Some("Random") => {}
        Some(other) => {
            return Err(refuse(format!(
                "`ImageSampler` `{other}`: the stochastic optimisers here draw random coordinates"
            )))
        }
    }
    params.samples = optional_usize(map, "NumberOfSpatialSamples")?.unwrap_or(5000);
    params.new_samples_every_iteration = flag("NewSamplesEveryIteration", false)?;
    params.sample_region = if flag("UseRandomSampleRegion", false)? {
        Some(one_or_three(map, "SampleRegionSize")?)
    } else {
        None
    };
    params.minimum_valid_fraction = if flag("CheckNumberOfSamples", true)? {
        optional_f64(map, "RequiredRatioOfValidSamples")?.unwrap_or(0.25)
    } else {
        0.0
    };
    params.sampling_attempts = optional_usize(map, "MaximumNumberOfSamplingAttempts")?
        .unwrap_or(0)
        .max(1);

    // ---- optimiser
    let a = optional_f64(map, "SP_a")?.unwrap_or(400.0);
    let alpha = optional_f64(map, "SP_alpha")?.unwrap_or(0.602);
    params.optimizer = match text("Optimizer")? {
        Some("StandardGradientDescent") => StochasticOptimizer::Standard(GainSequence {
            a,
            big_a: optional_f64(map, "SP_A")?.unwrap_or(50.0),
            alpha,
        }),
        Some("AdaptiveStochasticGradientDescent") => {
            let big_a = optional_f64(map, "SP_A")?.unwrap_or(20.0);
            let initial_time = optional_f64(map, "SigmoidInitialTime")?.unwrap_or(0.0);
            if let Some(method) = text("ASGDParameterEstimationMethod")? {
                if method != "Original" {
                    return Err(refuse(format!(
                        "`ASGDParameterEstimationMethod` `{method}`: `Original` only"
                    )));
                }
            }
            if !flag("UseAdaptiveStepSizes", true)? {
                StochasticOptimizer::Standard(GainSequence { a, big_a, alpha })
            } else if flag("AutomaticParameterEstimation", true)? {
                let ratio = optional_f64(map, "MaximumStepLengthRatio")?.unwrap_or(1.0);
                let measurements =
                    optional_usize(map, "NumberOfGradientMeasurements")?.unwrap_or(0);
                StochasticOptimizer::AdaptiveAutomatic {
                    big_a,
                    estimation: AutomaticGain {
                        max_step: optional_f64(map, "MaximumStepLength")?.unwrap_or(ratio),
                        gradient_measurements: if measurements == 0 { 3 } else { measurements },
                        jacobian_measurements: optional_usize(map, "NumberOfJacobianMeasurements")?
                            .unwrap_or(1000),
                        exact_gradient_samples: optional_usize(
                            map,
                            "NumberOfSamplesForExactGradient",
                        )?
                        .unwrap_or(100_000),
                        sigmoid_scale_factor: optional_f64(map, "SigmoidScaleFactor")?
                            .unwrap_or(0.1),
                    },
                    initial_time,
                }
            } else {
                StochasticOptimizer::Adaptive {
                    gain: GainSequence { a, big_a, alpha },
                    time: SigmoidTime {
                        max: optional_f64(map, "SigmoidMax")?.unwrap_or(1.0),
                        min: optional_f64(map, "SigmoidMin")?.unwrap_or(-0.8),
                        scale: optional_f64(map, "SigmoidScale")?.unwrap_or(1e-8),
                        initial_time,
                    },
                }
            }
        }
        Some(other) => return Err(refuse(format!("`Optimizer` `{other}` is not stochastic"))),
        None => return Err(refuse("`Optimizer` is required".to_string())),
    };

    // ---- scales and initialisation
    params.scales = if params.model == TransformModel::BSpline {
        if flag("AutomaticScalesEstimation", false)? {
            return Err(refuse(
                "`AutomaticScalesEstimation` for a B-spline is not implemented".to_string(),
            ));
        }
        ParameterScales::Unit
    } else if flag("AutomaticScalesEstimation", false)? {
        ParameterScales::Automatic { samples: 10_000 }
    } else if params.model == TransformModel::Translation {
        ParameterScales::Unit
    } else {
        return Err(refuse(
            "an affine without `AutomaticScalesEstimation \"true\"` uses elastix's `Scales`, \
             which is not read"
                .to_string(),
        ));
    };
    params.initialisation = if flag("AutomaticTransformInitialization", false)? {
        if let Some(method) = text("AutomaticTransformInitializationMethod")? {
            if method != "GeometricalCenter" {
                return Err(refuse(format!(
                    "`AutomaticTransformInitializationMethod` `{method}`: `GeometricalCenter` only"
                )));
            }
        }
        StochasticInitialisation::GeometricCentre
    } else {
        StochasticInitialisation::Identity
    };
    Ok(params)
}

fn unquote(value: &str) -> &str {
    value.trim_matches('"')
}

fn order(value: usize, key: &str) -> Result<SampleOrder> {
    match value {
        1 => Ok(SampleOrder::Linear),
        3 => Ok(SampleOrder::Cubic),
        other => Err(refuse(format!(
            "`{key}` {other}: orders 1 and 3 are implemented"
        ))),
    }
}

fn optional_usize(map: &ParameterMap, key: &str) -> Result<Option<usize>> {
    map.values(key).map(|_| map.usize(key)).transpose()
}

fn optional_f64(map: &ParameterMap, key: &str) -> Result<Option<f64>> {
    map.values(key)
        .map(|_| Ok(map.floats_n(key, 1)?[0]))
        .transpose()
}

fn one_or_three(map: &ParameterMap, key: &str) -> Result<[f64; 3]> {
    let values = map.floats(key)?;
    match values.as_slice() {
        [one] => Ok([*one; 3]),
        [a, b, c] => Ok([*a, *b, *c]),
        other => Err(refuse(format!(
            "`{key}` has {} values; 1 or 3 expected",
            other.len()
        ))),
    }
}

/// A value elastix reads per resolution: one for all, or one per level.
fn per_level_usize(
    map: &ParameterMap,
    key: &str,
    levels: usize,
    absent: usize,
) -> Result<Vec<usize>> {
    let Some(_) = map.values(key) else {
        return Ok(vec![absent; levels]);
    };
    let values = map.usizes(key)?;
    match values.len() {
        1 => Ok(vec![values[0]; levels]),
        n if n == levels => Ok(values),
        n => Err(refuse(format!(
            "`{key}` has {n} values for {levels} levels"
        ))),
    }
}
