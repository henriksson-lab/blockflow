// SPDX-License-Identifier: MIT
// Original work for this crate.

// Computing a registration, and writing it down as elastix parameter files.
//
// The optimiser is `blockflow::ops::align`: a block-parallel Gauss-Newton fit in
// which every iteration maps fixed/moving blocks to local evidence, reduces it
// deterministically and makes one global update. This module adds the two
// things that are about the file format rather than the fit:
//
// * **reading** an elastix registration parameter file (`align_affine.txt`
//   and the like) into `VolumeFitParams`, where every setting is either mapped,
//   approximated or refused by name — [`ELASTIX_SETTING_SUPPORT`] is the table;
// * **writing** the fitted transform as a `TransformParameters.0.txt`, in
//   physical coordinates, which [`ElastixTransform`] (and transformix) read back.
//
// The fit itself is in voxel coordinates of the full-resolution lattice; the
// geometry in `VolumeFitParams` is carried through so the written file is in
// physical units.

use blockflow::error::{Error, Result};
use blockflow::iterate::SubstageLimit;
use blockflow::ops::align;
use blockflow::pyramid::PyramidRecipe;
use blockflow::voxels::Voxels;
use ndarray::Array3;

pub use blockflow::ops::align::{
    plan, pyramid_levels_from_recipe, zarr_pyramid_levels, zarr_pyramid_levels_or_build,
    ControlGrid, FitConvergence, FittedTransform, Interpolator, Metric, PyramidSchedule,
    PyramidSmoothing, PyramidSource, Sampling, SpatialFrame, TransformModel, VolumeFitParams,
    VolumePyramidLevel, VolumePyramidStoreInput,
};

use crate::parameters::ParameterMap;
use crate::transform::ImageGeometry;
use crate::ElastixTransform;

impl ImageGeometry {
    /// The same geometry as the optimiser's frame.
    pub fn spatial_frame(&self) -> Result<SpatialFrame> {
        SpatialFrame::new(self.origin(), self.spacing(), self.direction())
    }

    pub fn from_spatial_frame(frame: &SpatialFrame) -> Result<Self> {
        Self::new(frame.origin(), frame.spacing(), frame.direction())
    }
}

/// A registration's result: the transform as a point map, the fit it came from,
/// and how the optimiser stopped.
#[derive(Debug, Clone, PartialEq)]
pub struct RegistrationDiagnostics {
    pub transform: ElastixTransform,
    pub fitted: FittedTransform,
    /// Optimiser substages completed after the dense initialiser.
    pub substages: usize,
    /// Final metric cost, including regularisation terms.
    pub final_cost: f64,
    /// Euclidean norm of the final accepted parameter update.
    pub final_step_norm: f64,
    /// Absolute cost change between the final two measured states.
    pub final_cost_delta: Option<f64>,
    pub converged_by: FitConvergence,
}

// -------------------------------------------------------------- running --

/// The fitted transform as a point map, through its own parameter file text,
/// so what a caller applies is exactly what was written down.
pub fn point_transform(fitted: &FittedTransform) -> Result<ElastixTransform> {
    ElastixTransform::from_texts(&[&transform_text(fitted)?])
}

pub fn run(
    params: &VolumeFitParams,
    fixed: &Voxels,
    moving: &Voxels,
    block: usize,
) -> Result<ElastixTransform> {
    run_with_workers(params, fixed, moving, block, 1)
}

pub fn run_with_workers(
    params: &VolumeFitParams,
    fixed: &Voxels,
    moving: &Voxels,
    block: usize,
    workers: usize,
) -> Result<ElastixTransform> {
    point_transform(&align::run_with_workers(
        params, fixed, moving, block, workers,
    )?)
}

pub fn run_with_diagnostics(
    params: &VolumeFitParams,
    fixed: &Voxels,
    moving: &Voxels,
    block: usize,
    workers: usize,
) -> Result<RegistrationDiagnostics> {
    let diagnostics = align::run_with_diagnostics(params, fixed, moving, block, workers)?;
    Ok(RegistrationDiagnostics {
        transform: point_transform(&diagnostics.transform)?,
        fitted: diagnostics.transform,
        substages: diagnostics.substages,
        final_cost: diagnostics.final_cost,
        final_step_norm: diagnostics.final_step_norm,
        final_cost_delta: diagnostics.final_cost_delta,
        converged_by: diagnostics.converged_by,
    })
}

/// Run, and return the `TransformParameters.0.txt` text alongside the point
/// map read back from it.
pub fn run_with_transform_text(
    params: &VolumeFitParams,
    fixed: &Voxels,
    moving: &Voxels,
    block: usize,
) -> Result<(ElastixTransform, String)> {
    let fitted = align::run(params, fixed, moving, block)?;
    let text = transform_text(&fitted)?;
    Ok((ElastixTransform::from_texts(&[&text])?, text))
}

pub fn resident(
    params: &VolumeFitParams,
    fixed: &Array3<f64>,
    moving: &Array3<f64>,
) -> Result<ElastixTransform> {
    point_transform(&align::resident(params, fixed, moving)?)
}

pub fn resident_pyramid(
    params: &VolumeFitParams,
    fixed: &Array3<f64>,
    moving: &Array3<f64>,
    schedule: &PyramidSchedule,
) -> Result<ElastixTransform> {
    point_transform(&align::resident_pyramid(params, fixed, moving, schedule)?)
}

pub fn resident_pyramid_with_recipe(
    params: &VolumeFitParams,
    fixed: &Array3<f64>,
    moving: &Array3<f64>,
    schedule: &PyramidSchedule,
    recipe: &PyramidRecipe,
) -> Result<ElastixTransform> {
    point_transform(&align::resident_pyramid_with_recipe(
        params, fixed, moving, schedule, recipe,
    )?)
}

pub fn resident_pyramid_levels(
    params: &VolumeFitParams,
    levels: &[VolumePyramidLevel],
) -> Result<ElastixTransform> {
    point_transform(&align::resident_pyramid_levels(params, levels)?)
}

// ----------------------------------------------------- the written file --

/// The fitted transform as elastix `TransformParameters.0.txt` text, in the
/// physical frame `fitted.geometry` describes.
///
/// The fit's parameters are in full-resolution voxels; with `M = D·S` the
/// voxel-to-physical matrix and `o` the origin, a voxel map `y = A x + t`
/// becomes `y = M A M⁻¹ x + (o + M t - M A M⁻¹ o)` physically, and B-spline
/// coefficients become `M c` on a grid at `o + M·origin` with spacing
/// `spacing ⊙ S`.
pub fn transform_text(fitted: &FittedTransform) -> Result<String> {
    let frame = &fitted.geometry;
    let origin = frame.origin();
    let spacing = frame.spacing();
    let direction = frame.direction();
    let to_physical = scaled_columns(direction, spacing);
    let params = &fitted.params;
    let geometry = geometry_block(fitted.shape, frame);
    match fitted.model {
        TransformModel::Translation => {
            let t = mat_vec(to_physical, [params[0], params[1], params[2]]);
            Ok(format!(
                "(Transform \"TranslationTransform\")\n\
                 (NumberOfParameters 3)\n\
                 (TransformParameters {} {} {})\n\
                 (InitialTransformParametersFileName \"NoInitialTransform\")\n\
                 (HowToCombineTransforms \"Compose\")\n{geometry}",
                t[0], t[1], t[2]
            ))
        }
        TransformModel::Affine => {
            let to_voxel = invert_3x3(to_physical)?;
            let voxel = [
                [params[0], params[1], params[2]],
                [params[3], params[4], params[5]],
                [params[6], params[7], params[8]],
            ];
            let matrix = mat_mul(mat_mul(to_physical, voxel), to_voxel);
            let moved = mat_vec(to_physical, [params[9], params[10], params[11]]);
            let shifted = mat_vec(matrix, origin);
            let mut values: Vec<f64> = matrix.iter().flatten().copied().collect();
            values.extend((0..3).map(|a| origin[a] + moved[a] - shifted[a]));
            Ok(format!(
                "(Transform \"AffineTransform\")\n\
                 (NumberOfParameters 12)\n\
                 (TransformParameters {})\n\
                 (InitialTransformParametersFileName \"NoInitialTransform\")\n\
                 (HowToCombineTransforms \"Compose\")\n\
                 (CenterOfRotationPoint 0.0 0.0 0.0)\n{geometry}",
                join(&values)
            ))
        }
        TransformModel::BSpline => {
            let per_block = params.len() / 3;
            let mut physical = vec![0.0; params.len()];
            for point in 0..per_block {
                let moved = mat_vec(
                    to_physical,
                    [
                        params[point],
                        params[per_block + point],
                        params[2 * per_block + point],
                    ],
                );
                for axis in 0..3 {
                    physical[axis * per_block + point] = moved[axis];
                }
            }
            let (grid_origin_voxel, grid_spacing_voxel) = match fitted.bspline_layout {
                Some(layout) => (layout.origin, layout.spacing),
                None => bspline_lattice(fitted.bspline, fitted.shape),
            };
            let offset = mat_vec(to_physical, grid_origin_voxel);
            let grid_origin: [f64; 3] = std::array::from_fn(|a| origin[a] + offset[a]);
            let grid_spacing: [f64; 3] =
                std::array::from_fn(|a| grid_spacing_voxel[a] * spacing[a]);
            let size = fitted.bspline.size;
            Ok(format!(
                "(Transform \"BSplineTransform\")\n\
                 (NumberOfParameters {})\n\
                 (TransformParameters {})\n\
                 (InitialTransformParametersFileName \"NoInitialTransform\")\n\
                 (HowToCombineTransforms \"Compose\")\n\
                 (GridSize {} {} {})\n\
                 (GridIndex 0 0 0)\n\
                 (GridSpacing {})\n\
                 (GridOrigin {})\n\
                 (GridDirection {})\n\
                 (BSplineTransformSplineOrder 3)\n{geometry}",
                physical.len(),
                join(&physical),
                size[0],
                size[1],
                size[2],
                join(&grid_spacing),
                join(&grid_origin),
                join(&column_major(direction)),
            ))
        }
    }
}

/// Point a written transform at the one it composes onto.
///
/// For a chain written as several files: `text`'s transform is applied after
/// the one in `initial`, which is elastix's `"Compose"`.
pub fn with_initial(text: &str, initial: &str) -> String {
    text.replace(
        "(InitialTransformParametersFileName \"NoInitialTransform\")",
        &format!("(InitialTransformParametersFileName \"{initial}\")"),
    )
}

/// The control grid the optimiser's B-spline lives on, in voxels.
///
/// `ops::align` does not export this, so it is restated here and must agree
/// with it: `GridSize - 3` intervals over `shape - 2` voxels, the first control
/// point one interval before voxel 0.
fn bspline_lattice(grid: ControlGrid, shape: [usize; 3]) -> ([f64; 3], [f64; 3]) {
    let mut spacing = [1.0; 3];
    let mut origin = [0.0; 3];
    for axis in 0..3 {
        let high = shape[axis].saturating_sub(2).max(1) as f64;
        spacing[axis] = high / (grid.size[axis] - 3) as f64;
        origin[axis] = -spacing[axis];
    }
    (origin, spacing)
}

fn geometry_block(size: [usize; 3], frame: &SpatialFrame) -> String {
    format!(
        "(Size {} {} {})\n\
         (Index 0 0 0)\n\
         (Spacing {})\n\
         (Origin {})\n\
         (Direction {})\n\
         (FixedImageDimension 3)\n\
         (MovingImageDimension 3)\n",
        size[0],
        size[1],
        size[2],
        join(&frame.spacing()),
        join(&frame.origin()),
        join(&column_major(frame.direction())),
    )
}

fn join(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// `(Direction …)` is written column-major.
fn column_major(matrix: [[f64; 3]; 3]) -> [f64; 9] {
    std::array::from_fn(|k| matrix[k % 3][k / 3])
}

fn scaled_columns(matrix: [[f64; 3]; 3], scale: [f64; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|col| matrix[row][col] * scale[col]))
}

fn mat_mul(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|col| (0..3).map(|k| left[row][k] * right[k][col]).sum())
    })
}

fn mat_vec(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| (0..3).map(|k| matrix[row][k] * vector[k]).sum())
}

fn invert_3x3(m: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if determinant.abs() <= f64::EPSILON || !determinant.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "registration: the voxel-to-physical matrix {m:?} has determinant {determinant} and \
             cannot be inverted"
        )));
    }
    Ok(std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            let (r0, r1) = ((column + 1) % 3, (column + 2) % 3);
            let (c0, c1) = ((row + 1) % 3, (row + 2) % 3);
            (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / determinant
        })
    }))
}

// ------------------------------------------------ the parameter file read --

/// Registration settings from the text of one elastix parameter file.
pub fn params_from_elastix_text(text: &str) -> Result<VolumeFitParams> {
    params_from_elastix_map(&ParameterMap::parse(text)?)
}

/// Registration settings from one parsed elastix parameter file.
///
/// Every key is checked against [`ELASTIX_SETTING_SUPPORT`] first, so a setting
/// that would change elastix's answer and that the native fit does not honour
/// is refused by name rather than ignored.
pub fn params_from_elastix_map(map: &ParameterMap) -> Result<VolumeFitParams> {
    validate_elastix_compute_keys(map)?;
    let mut params = VolumeFitParams::default();
    if let Some(transform) = optional_string(map, "Transform")? {
        params.model = match transform {
            "TranslationTransform" => TransformModel::Translation,
            "AffineTransform" | "EulerTransform" => TransformModel::Affine,
            "BSplineTransform" | "RecursiveBSplineTransform" => TransformModel::BSpline,
            other => {
                return Err(Error::InvalidArgument(format!(
                    "registration compute: elastix setting `Transform` value `{other}` is not \
                     supported by native registration"
                )));
            }
        };
    }
    if let Some(metric) = optional_strings(map, "Metric")? {
        if metric.len() != 1 {
            return Err(Error::InvalidArgument(format!(
                "registration compute: elastix setting `Metric` has {} values; native \
                 registration accepts one metric per run",
                metric.len()
            )));
        }
        params.metric = match metric[0].as_str() {
            "AdvancedMeanSquares" | "MeanSquares" => Metric::MeanSquares,
            "NormalizedCorrelation" | "AdvancedNormalizedCorrelation" => {
                Metric::NormalizedCorrelation
            }
            "AdvancedMattesMutualInformation"
            | "MattesMutualInformation"
            | "NormalizedMutualInformation" => Metric::MutualInformation,
            other => {
                return Err(Error::InvalidArgument(format!(
                    "registration compute: elastix setting `Metric` value `{other}` is not \
                     supported by native registration"
                )));
            }
        };
    }
    if let Some(bins) = optional_usizes(map, "NumberOfHistogramBins")? {
        let bins = bins.into_iter().max().ok_or_else(|| {
            Error::InvalidArgument(
                "registration compute: elastix setting `NumberOfHistogramBins` has no values"
                    .to_string(),
            )
        })?;
        params = params.with_metric_bins(bins)?;
    }
    if let Some(optimizer) = optional_string(map, "Optimizer")? {
        validate_optimizer_from_elastix(optimizer)?;
    }
    if let Some(pyramid) = optional_string(map, "FixedImagePyramid")? {
        validate_image_pyramid_from_elastix(
            "FixedImagePyramid",
            pyramid,
            &["FixedSmoothingImagePyramid", "FixedRecursiveImagePyramid"],
        )?;
    }
    if let Some(pyramid) = optional_string(map, "MovingImagePyramid")? {
        validate_image_pyramid_from_elastix(
            "MovingImagePyramid",
            pyramid,
            &["MovingSmoothingImagePyramid", "MovingRecursiveImagePyramid"],
        )?;
    }
    if let Some(initial) = optional_string(map, "InitialTransformParametersFileName")? {
        validate_initial_transform_from_elastix(initial)?;
    }
    if let Some(combine) = optional_string(map, "HowToCombineTransforms")? {
        validate_combine_from_elastix(combine)?;
    }
    if let Some(initialization) = optional_bool(map, "AutomaticTransformInitialization")? {
        if !initialization {
            return Err(Error::InvalidArgument(
                "registration compute: elastix setting `AutomaticTransformInitialization` \
                 value `false` is not supported by native registration; the native path uses \
                 its dense moment initialiser when no initial transform is supplied"
                    .to_string(),
            ));
        }
    }
    if let Some(scales) = optional_bool(map, "AutomaticScalesEstimation")? {
        if !scales {
            return Err(Error::InvalidArgument(
                "registration compute: elastix setting `AutomaticScalesEstimation` value \
                 `false` is not supported by native registration; pyramid parameter scaling \
                 is always applied"
                    .to_string(),
            ));
        }
    }
    if let Some(sampler) = optional_string(map, "ImageSampler")? {
        params.sampling = match sampler {
            "Full" | "FullSampler" => Sampling::All,
            "Grid" | "GridSampler" => {
                Sampling::Stride(parse_sample_grid_spacing(map)?.unwrap_or([2, 2, 2]))
            }
            "Random" | "RandomCoordinate" | "RandomSparseMask" => {
                return Err(Error::InvalidArgument(format!(
                    "registration compute: elastix setting `ImageSampler` value `{sampler}` \
                     needs random sampling state; native registration supports deterministic \
                     `Full` and `Grid` sampling"
                )));
            }
            other => {
                return Err(Error::InvalidArgument(format!(
                    "registration compute: elastix setting `ImageSampler` value `{other}` is \
                     not supported by native registration"
                )));
            }
        };
    }
    if let Some(interpolator) = optional_string(map, "Interpolator")? {
        params.interpolator = moving_interpolator_from_elastix(map, interpolator)?;
    }
    if let Some(iterations) = optional_usizes(map, "MaximumNumberOfIterations")? {
        let limit = iterations.into_iter().max().ok_or_else(|| {
            Error::InvalidArgument(
                "registration compute: elastix setting `MaximumNumberOfIterations` has no \
                 values"
                    .to_string(),
            )
        })?;
        params.limit = SubstageLimit::of(limit)?;
    }
    if let Some(step) = optional_f64(map, "MaximumStepLength")? {
        if step < 0.0 || !step.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "registration compute: elastix setting `MaximumStepLength` must be finite and \
                 non-negative, got {step}"
            )));
        }
        params.max_step = step;
    }
    if let Some(damping) = optional_f64(map, "LevenbergMarquardtDamping")? {
        if damping < 0.0 || !damping.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "registration compute: elastix setting `LevenbergMarquardtDamping` must be \
                 finite and non-negative, got {damping}"
            )));
        }
        params.damping = damping;
    }
    if let Some(tolerance) = optional_f64(map, "RegistrationTolerance")? {
        if tolerance < 0.0 || !tolerance.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "registration compute: elastix setting `RegistrationTolerance` must be finite \
                 and non-negative, got {tolerance}"
            )));
        }
        params.tolerance = tolerance;
    }
    let resolutions = optional_usize(map, "NumberOfResolutions")?;
    if let Some(schedule_values) = optional_usizes(map, "ImagePyramidSchedule")? {
        params.pyramid = Some(schedule_from_elastix(&schedule_values, resolutions)?);
    } else if let Some(levels) = resolutions {
        params.pyramid = Some(PyramidSchedule::powers_of_two(levels)?);
    }
    if let Some(order) = optional_usize(map, "BSplineTransformSplineOrder")? {
        if order != 3 {
            return Err(Error::InvalidArgument(format!(
                "registration compute: elastix setting `BSplineTransformSplineOrder` is {order}; \
                 native B-spline registration supports order 3"
            )));
        }
    }
    if let Some(spacing) = optional_f64_triple(map, "FinalGridSpacingInVoxels")? {
        if params.model != TransformModel::BSpline {
            return Err(Error::InvalidArgument(
                "registration compute: elastix setting `FinalGridSpacingInVoxels` applies to \
                 B-spline registration; native registration only accepts it with \
                 `BSplineTransform`"
                    .to_string(),
            ));
        }
        params = params
            .with_bspline_final_grid_spacing(spacing)
            .map_err(|error| {
                Error::InvalidArgument(format!(
                    "registration compute: elastix setting `FinalGridSpacingInVoxels`: {error}"
                ))
            })?;
    }
    Ok(params)
}

/// How one elastix parameter-file setting maps to native registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElastixSettingStatus {
    /// The setting has a direct native equivalent.
    Supported,
    /// The native behaviour is deliberate but not a 1:1 elastix implementation.
    Approximation,
    /// The setting is known and intentionally not implemented.
    Unsupported,
}

/// One row in the elastix-to-native optimiser mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElastixSettingSupport {
    pub setting: &'static str,
    pub status: ElastixSettingStatus,
    pub native: &'static str,
}

/// The Gauss-Newton fit's map from relevant elastix settings. Files with a
/// stochastic optimiser are read by [`crate::stochastic`] instead, which has
/// its own table.
///
/// This table is intentionally data rather than prose only: tests assert that
/// unsupported settings are named and that implemented settings have a stable
/// contract. It is not a parser for parameter files; it is the audit surface for
/// the settings common atlas-registration parameter files use.
pub const ELASTIX_SETTING_SUPPORT: &[ElastixSettingSupport] = &[
    ElastixSettingSupport {
        setting: "Transform",
        status: ElastixSettingStatus::Supported,
        native: "TransformModel::{Translation, Affine, BSpline}",
    },
    ElastixSettingSupport {
        setting: "Metric",
        status: ElastixSettingStatus::Approximation,
        native: "Metric::{MeanSquares, NormalizedCorrelation, MutualInformation}",
    },
    ElastixSettingSupport {
        setting: "NumberOfHistogramBins",
        status: ElastixSettingStatus::Supported,
        native: "VolumeFitParams::metric_bins",
    },
    ElastixSettingSupport {
        setting: "NumberOfResolutions",
        status: ElastixSettingStatus::Supported,
        native: "PyramidSchedule::levels",
    },
    ElastixSettingSupport {
        setting: "ImagePyramidSchedule",
        status: ElastixSettingStatus::Supported,
        native: "PyramidSchedule factors and explicit PyramidRecipe",
    },
    ElastixSettingSupport {
        setting: "FixedImagePyramid",
        status: ElastixSettingStatus::Approximation,
        native: "PyramidSmoothing plus resident/recipe pyramid building",
    },
    ElastixSettingSupport {
        setting: "MovingImagePyramid",
        status: ElastixSettingStatus::Approximation,
        native: "PyramidSmoothing plus resident/recipe pyramid building",
    },
    ElastixSettingSupport {
        setting: "Interpolator",
        status: ElastixSettingStatus::Supported,
        native: "trilinear moving-image sampling",
    },
    ElastixSettingSupport {
        setting: "BSplineInterpolationOrder",
        status: ElastixSettingStatus::Supported,
        native: "accepted only as order 1 when Interpolator is BSplineInterpolator",
    },
    ElastixSettingSupport {
        setting: "FinalBSplineInterpolationOrder",
        status: ElastixSettingStatus::Unsupported,
        native: "no result-image resampler is emitted by native registration",
    },
    ElastixSettingSupport {
        setting: "Optimizer",
        status: ElastixSettingStatus::Approximation,
        native: "AdaptiveStochasticGradientDescent and StandardGradientDescent map to \
                 block-reduced damped Gauss-Newton state updates",
    },
    ElastixSettingSupport {
        setting: "MaximumNumberOfIterations",
        status: ElastixSettingStatus::Supported,
        native: "VolumeFitParams::limit",
    },
    ElastixSettingSupport {
        setting: "MaximumStepLength",
        status: ElastixSettingStatus::Supported,
        native: "VolumeFitParams::max_step",
    },
    ElastixSettingSupport {
        setting: "LevenbergMarquardtDamping",
        status: ElastixSettingStatus::Supported,
        native: "VolumeFitParams::damping",
    },
    ElastixSettingSupport {
        setting: "RegistrationTolerance",
        status: ElastixSettingStatus::Supported,
        native: "VolumeFitParams::tolerance",
    },
    ElastixSettingSupport {
        setting: "AutomaticScalesEstimation",
        status: ElastixSettingStatus::Approximation,
        native: "per-resolution parameter scaling from PyramidSchedule",
    },
    ElastixSettingSupport {
        setting: "AutomaticTransformInitialization",
        status: ElastixSettingStatus::Approximation,
        native: "dense moment initialiser when no initial transform is supplied",
    },
    ElastixSettingSupport {
        setting: "InitialTransformParametersFileName",
        status: ElastixSettingStatus::Approximation,
        native: "`NoInitialTransform` maps to native initialisation; transform chains are \
                 refused",
    },
    ElastixSettingSupport {
        setting: "HowToCombineTransforms",
        status: ElastixSettingStatus::Approximation,
        native: "`Compose` is accepted for no-initial native compute imports; additive chains are \
                 refused",
    },
    ElastixSettingSupport {
        setting: "AutomaticParameterEstimation",
        status: ElastixSettingStatus::Unsupported,
        native: "native registration does not implement elastix stochastic optimiser parameter \
                 estimation",
    },
    ElastixSettingSupport {
        setting: "ImageSampler",
        status: ElastixSettingStatus::Approximation,
        native: "Sampling::{All, Stride}; random samplers are refused by value",
    },
    ElastixSettingSupport {
        setting: "SampleGridSpacing",
        status: ElastixSettingStatus::Supported,
        native: "Sampling::Stride",
    },
    ElastixSettingSupport {
        setting: "FixedImageBSplineInterpolationOrder",
        status: ElastixSettingStatus::Unsupported,
        native: "fixed-image interpolation order is only used by elastix random-coordinate \
                 sampling; native registration samples fixed voxels on its deterministic grid",
    },
    ElastixSettingSupport {
        setting: "NumberOfSpatialSamples",
        status: ElastixSettingStatus::Unsupported,
        native: "random spatial sample budgets are not used by deterministic native sampling",
    },
    ElastixSettingSupport {
        setting: "NewSamplesEveryIteration",
        status: ElastixSettingStatus::Unsupported,
        native: "native registration has deterministic per-iteration samples",
    },
    ElastixSettingSupport {
        setting: "UseRandomSampleRegion",
        status: ElastixSettingStatus::Unsupported,
        native: "native registration does not implement elastix random sample regions",
    },
    ElastixSettingSupport {
        setting: "SampleRegionSize",
        status: ElastixSettingStatus::Unsupported,
        native: "native registration does not implement elastix random sample regions",
    },
    ElastixSettingSupport {
        setting: "MaximumNumberOfSamplingAttempts",
        status: ElastixSettingStatus::Unsupported,
        native: "native registration does not retry random sample draws",
    },
    ElastixSettingSupport {
        setting: "UseFastAndLowMemoryVersion",
        status: ElastixSettingStatus::Unsupported,
        native: "native B-spline registration uses its own reduced normal equations and does not \
                 implement elastix's alternate B-spline memory mode",
    },
    ElastixSettingSupport {
        setting: "SP_a",
        status: ElastixSettingStatus::Unsupported,
        native: "native registration uses max_step, damping, and tolerance instead of stochastic \
                 gain schedules",
    },
    ElastixSettingSupport {
        setting: "SP_A",
        status: ElastixSettingStatus::Unsupported,
        native: "native registration uses max_step, damping, and tolerance instead of stochastic \
                 gain schedules",
    },
    ElastixSettingSupport {
        setting: "SP_alpha",
        status: ElastixSettingStatus::Unsupported,
        native: "native registration uses max_step, damping, and tolerance instead of stochastic \
                 gain schedules",
    },
    ElastixSettingSupport {
        setting: "FinalGridSpacingInVoxels",
        status: ElastixSettingStatus::Supported,
        native: "shape-dependent ControlGrid derived from final spacing in voxels",
    },
    ElastixSettingSupport {
        setting: "BSplineTransformSplineOrder",
        status: ElastixSettingStatus::Supported,
        native: "native B-spline registration accepts cubic order 3",
    },
    ElastixSettingSupport {
        setting: "Metric0Weight",
        status: ElastixSettingStatus::Unsupported,
        native: "single metric per native run",
    },
    ElastixSettingSupport {
        setting: "Metric1Weight",
        status: ElastixSettingStatus::Unsupported,
        native: "single metric per native run",
    },
];

pub fn elastix_setting_support(setting: &str) -> ElastixSettingSupport {
    ELASTIX_SETTING_SUPPORT
        .iter()
        .copied()
        .find(|row| row.setting == setting)
        .unwrap_or(ElastixSettingSupport {
            setting: "unknown",
            status: ElastixSettingStatus::Unsupported,
            native: "unknown elastix setting is not applied by native registration",
        })
}

pub fn require_supported_elastix_setting(setting: &str) -> Result<()> {
    let support = elastix_setting_support(setting);
    if support.status == ElastixSettingStatus::Unsupported {
        Err(Error::InvalidArgument(format!(
            "registration compute: elastix setting `{setting}` is unsupported by native \
             registration: {}",
            support.native
        )))
    } else {
        Ok(())
    }
}

fn validate_elastix_compute_keys(map: &ParameterMap) -> Result<()> {
    for key in map.keys() {
        if ignored_elastix_compute_key(key) {
            continue;
        }
        let support = elastix_setting_support(key);
        if support.setting == "unknown" {
            return Err(Error::InvalidArgument(format!(
                "registration compute: elastix setting `{key}` is not mapped to native \
                 registration"
            )));
        }
        if support.status == ElastixSettingStatus::Unsupported {
            return Err(Error::InvalidArgument(format!(
                "registration compute: elastix setting `{key}` is unsupported by native \
                 registration: {}",
                support.native
            )));
        }
    }
    Ok(())
}

fn ignored_elastix_compute_key(key: &str) -> bool {
    matches!(
        key,
        "BSplineInterpolationOrder"
            | "BSplineTransformSplineOrder"
            | "CheckNumberOfSamples"
            | "DefaultPixelValue"
            | "ErodeMask"
            | "FixedImageDimension"
            | "FixedInternalImagePixelType"
            | "FixedLimitRangeRatio"
            | "FixedKernelBSplineOrder"
            | "GridDirection"
            | "GridIndex"
            | "GridOrigin"
            | "GridSize"
            | "GridSpacing"
            | "ImagePyramidSchedule"
            | "Index"
            | "LevenbergMarquardtDamping"
            | "MaximumStepLength"
            | "MovingImageDimension"
            | "MovingInternalImagePixelType"
            | "MovingLimitRangeRatio"
            | "MovingKernelBSplineOrder"
            | "NumberOfParameters"
            | "NumberOfSamplesForExactGradient"
            | "Origin"
            | "Registration"
            | "RegistrationTolerance"
            | "CompressResultImage"
            | "ResampleInterpolator"
            | "Resampler"
            | "ResultImageFormat"
            | "ResultImagePixelType"
            | "SampleGridSpacing"
            | "ShowExactMetricValue"
            | "Size"
            | "Spacing"
            | "TransformParameters"
            | "UseDirectionCosines"
            | "WriteDiffusionFiles"
            | "WriteResultImage"
            | "WriteResultImageAfterEachResolution"
            | "WriteTransformParametersEachIteration"
    )
}

fn optional_string<'a>(map: &'a ParameterMap, key: &str) -> Result<Option<&'a str>> {
    if map.values(key).is_none() {
        return Ok(None);
    }
    map.string(key).map(Some)
}

fn optional_strings(map: &ParameterMap, key: &str) -> Result<Option<Vec<String>>> {
    let Some(values) = map.values(key) else {
        return Ok(None);
    };
    Ok(Some(
        values
            .iter()
            .map(|value| value.trim_matches('"').to_string())
            .collect(),
    ))
}

fn optional_usize(map: &ParameterMap, key: &str) -> Result<Option<usize>> {
    if map.values(key).is_none() {
        return Ok(None);
    }
    map.usize(key).map(Some)
}

fn optional_usizes(map: &ParameterMap, key: &str) -> Result<Option<Vec<usize>>> {
    if map.values(key).is_none() {
        return Ok(None);
    }
    map.usizes(key).map(Some)
}

fn optional_f64(map: &ParameterMap, key: &str) -> Result<Option<f64>> {
    if map.values(key).is_none() {
        return Ok(None);
    }
    let values = map.floats_n(key, 1)?;
    Ok(Some(values[0]))
}

fn optional_bool(map: &ParameterMap, key: &str) -> Result<Option<bool>> {
    if map.values(key).is_none() {
        return Ok(None);
    }
    map.flag(key, false).map(Some)
}

fn optional_f64_triple(map: &ParameterMap, key: &str) -> Result<Option<[f64; 3]>> {
    if map.values(key).is_none() {
        return Ok(None);
    }
    let values = map.floats_n(key, 3)?;
    Ok(Some([values[0], values[1], values[2]]))
}

fn validate_optimizer_from_elastix(optimizer: &str) -> Result<()> {
    match optimizer {
        "AdaptiveStochasticGradientDescent" | "StandardGradientDescent" => Ok(()),
        other => Err(Error::InvalidArgument(format!(
            "registration compute: elastix setting `Optimizer` value `{other}` is not supported \
             by native registration; `AdaptiveStochasticGradientDescent` and \
             `StandardGradientDescent` are approximated by block-reduced damped Gauss-Newton"
        ))),
    }
}

fn validate_image_pyramid_from_elastix(
    setting: &str,
    pyramid: &str,
    accepted: &[&str],
) -> Result<()> {
    if accepted.contains(&pyramid) {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!(
            "registration compute: elastix setting `{setting}` value `{pyramid}` is not supported \
             by native registration; supported pyramid components are {}",
            accepted.join(", ")
        )))
    }
}

fn validate_initial_transform_from_elastix(initial: &str) -> Result<()> {
    if initial == "NoInitialTransform" {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!(
            "registration compute: elastix setting `InitialTransformParametersFileName` value \
             `{initial}` is not supported by native registration compute import; transform \
             chains are not imported"
        )))
    }
}

fn validate_combine_from_elastix(combine: &str) -> Result<()> {
    if combine == "Compose" {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!(
            "registration compute: elastix setting `HowToCombineTransforms` value `{combine}` is \
             not supported by native registration compute import; only `Compose` is compatible \
             with no-initial single-run native compute"
        )))
    }
}

fn moving_interpolator_from_elastix(
    map: &ParameterMap,
    interpolator: &str,
) -> Result<Interpolator> {
    match interpolator {
        "LinearInterpolator" => Ok(Interpolator::Linear),
        "BSplineInterpolator" => {
            let Some(order) = optional_usize(map, "BSplineInterpolationOrder")? else {
                return Err(Error::InvalidArgument(
                    "registration compute: elastix setting `Interpolator` value \
                     `BSplineInterpolator` requires `BSplineInterpolationOrder` so native \
                     registration can prove it maps to trilinear sampling"
                        .to_string(),
                ));
            };
            if order == 1 {
                Ok(Interpolator::Linear)
            } else {
                Err(Error::InvalidArgument(format!(
                    "registration compute: elastix setting `Interpolator` value \
                     `BSplineInterpolator` with `BSplineInterpolationOrder` {order} is not \
                     supported by native registration; only order 1 maps to trilinear sampling"
                )))
            }
        }
        other => Err(Error::InvalidArgument(format!(
            "registration compute: elastix setting `Interpolator` value `{other}` is not \
             supported by native registration"
        ))),
    }
}

fn parse_sample_grid_spacing(map: &ParameterMap) -> Result<Option<[usize; 3]>> {
    let Some(values) = optional_usizes(map, "SampleGridSpacing")? else {
        return Ok(None);
    };
    let spacing = match values.as_slice() {
        [one] => [*one; 3],
        [x, y, z] => [*x, *y, *z],
        other => {
            return Err(Error::InvalidArgument(format!(
                "registration compute: elastix setting `SampleGridSpacing` has {} values; \
                 expected one isotropic spacing or three axis spacings",
                other.len()
            )));
        }
    };
    if spacing.contains(&0) {
        return Err(Error::InvalidArgument(format!(
            "registration compute: elastix setting `SampleGridSpacing` {spacing:?} must be \
             positive on every axis"
        )));
    }
    Ok(Some(spacing))
}

fn schedule_from_elastix(
    values: &[usize],
    declared_resolutions: Option<usize>,
) -> Result<PyramidSchedule> {
    if !values.len().is_multiple_of(3) {
        return Err(Error::InvalidArgument(format!(
            "registration compute: elastix setting `ImagePyramidSchedule` has {} values; a 3D \
             schedule needs triples",
            values.len()
        )));
    }
    let levels = values.len() / 3;
    if let Some(declared) = declared_resolutions {
        if declared != levels {
            return Err(Error::InvalidArgument(format!(
                "registration compute: elastix setting `NumberOfResolutions` is {declared}, but \
                 `ImagePyramidSchedule` has {levels} level(s)"
            )));
        }
    }
    if levels == 0 {
        return Err(Error::InvalidArgument(
            "registration compute: elastix setting `ImagePyramidSchedule` has no levels"
                .to_string(),
        ));
    }
    let mut cumulative = Vec::with_capacity(levels);
    for scale in values.as_chunks::<3>().0 {
        let scale = *scale;
        if scale.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "registration compute: elastix setting `ImagePyramidSchedule` contains zero scale \
                 in {scale:?}"
            )));
        }
        cumulative.push(scale);
    }
    cumulative.reverse();
    if cumulative[0] != [1, 1, 1] {
        return Err(Error::InvalidArgument(format!(
            "registration compute: elastix setting `ImagePyramidSchedule` finest level must be \
             [1, 1, 1], got {:?}",
            cumulative[0]
        )));
    }
    let mut factors = Vec::with_capacity(levels.saturating_sub(1));
    for pair in cumulative.windows(2) {
        let fine = pair[0];
        let coarse = pair[1];
        let mut factor = [1; 3];
        for axis in 0..3 {
            if coarse[axis] < fine[axis] || coarse[axis] % fine[axis] != 0 {
                return Err(Error::InvalidArgument(format!(
                    "registration compute: elastix setting `ImagePyramidSchedule` level {:?} is \
                     not a cumulative multiple of finer level {:?}",
                    coarse, fine
                )));
            }
            factor[axis] = coarse[axis] / fine[axis];
        }
        factors.push(factor);
    }
    PyramidSchedule::new(factors)
}
