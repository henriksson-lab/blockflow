// SPDX-License-Identifier: MIT
//
// Block-reduced fitting of a 3-D coordinate map between two scalar volumes.
//
// This is a general optimizer shape: each block maps two image regions to local
// evidence, all evidence is reduced in a deterministic order, then one global
// parameter state is updated. The transform family and sampling policy are
// parameters; file-format adapters belong in caller crates.
//
// The second half of the file is a different optimiser for the same problem:
// `fit_stochastic`, elastix's stochastic gradient descent on Mattes mutual
// information with a random-coordinate sampler (see the section header there
// for why it is resident rather than planned).

use crate::assemble::ImageId;
use crate::decomposition::Decomposition;
use crate::env::{ArrayEnvironment, BlockBuf, Environment};
use crate::error::{Error, Result};
use crate::fragment::PhaseWork;
use crate::geometry::BlockGrid;
use crate::iterate::{
    iterative_reduce_phase, IterativeReduceOp, Partial, ReduceBlock, StateUpdate, SubstageLimit,
};
use crate::op::{Chain as BlockChain, SourceInput};
use crate::ops::{Gaussian, SmoothOp};
use crate::pyramid::{LevelRecipe, PyramidRecipe};
use crate::sidecar::Lifecycle;
use crate::strategy::{execute_phases, Hints, Workflow as BlockWorkflow};
use crate::voxels::Voxels;
use crate::Dtype;
use crate::Reach;

use ndarray::{Array3, ArrayView3, Axis};
use rayon::prelude::*;
#[cfg(feature = "zarr")]
use std::path::PathBuf;

const STATE_STREAM: &str = "volume_fit_state";
fn moving_image() -> ImageId {
    ImageId::supplied(0)
}

/// Physical frame attached to voxel coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialFrame {
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: [[f64; 3]; 3],
}

impl SpatialFrame {
    pub fn new(origin: [f64; 3], spacing: [f64; 3], direction: [[f64; 3]; 3]) -> Result<Self> {
        if spacing
            .iter()
            .any(|value| *value == 0.0 || !value.is_finite())
        {
            return Err(Error::InvalidArgument(format!(
                "spatial frame spacing {spacing:?} has a zero or non-finite extent"
            )));
        }
        invert_3x3(direction)?;
        Ok(Self {
            origin,
            spacing,
            direction,
        })
    }

    pub fn unit() -> Self {
        Self {
            origin: [0.0; 3],
            spacing: [1.0; 3],
            direction: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    pub fn spacing(&self) -> [f64; 3] {
        self.spacing
    }

    pub fn direction(&self) -> [[f64; 3]; 3] {
        self.direction
    }
}

/// Parameters for volume fitting computation.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeFitParams {
    /// Transform model to optimize.
    pub model: TransformModel,
    /// Image similarity optimized by the global update.
    pub metric: Metric,
    /// Histogram bins per image axis for histogram-based metrics.
    pub metric_bins: usize,
    /// Which fixed-image voxels contribute metric/gradient evidence after the
    /// dense moment initializer.
    pub sampling: Sampling,
    /// Moving-image interpolation used by metric evaluation.
    pub interpolator: Interpolator,
    /// Backstop for the global update loop.
    pub limit: SubstageLimit,
    /// Moving-image halo in voxels. The optimizer only evaluates samples it has
    /// fetched, so this should cover the expected correction plus one
    /// interpolation cell.
    pub search_radius: usize,
    /// Largest parameter update accepted from one Gauss-Newton step.
    pub max_step: f64,
    /// Stop once the update norm is no larger than this.
    pub tolerance: f64,
    /// Diagonal Levenberg damping added to the normal equations.
    pub damping: f64,
    /// Fixed-image geometry, carried through to the emitted transform.
    ///
    /// **It is metadata here, not something the fit is expressed in.** The
    /// optimizer works entirely in voxel coordinates — `transform_point` and
    /// `parameter_jacobian` take voxel positions and return voxel positions —
    /// and [`FittedTransform`] says so: its parameters are in the
    /// full-resolution voxel frame. The spacing is validated, and a coarse
    /// pyramid level's frame is required to be this one scaled by the level's
    /// factor, but no conversion to physical coordinates happens anywhere; an
    /// anisotropic frame produces bit-identical parameters to a unit one, which
    /// `tests/align.rs` asserts. A caller who wants physical units applies this
    /// frame themselves.
    pub geometry: SpatialFrame,
    /// Optional resident coarse-to-fine schedule used by [`resident`].
    pub pyramid: Option<PyramidSchedule>,
    /// Optional Gaussian smoothing applied before each optimization pyramid
    /// decimation. This affects only derived in-memory volume fitting levels, not
    /// OME-Zarr storage pyramids.
    pub pyramid_smoothing: Option<PyramidSmoothing>,
    /// Where coarse-to-fine optimization levels should come from when a caller
    /// uses a storage-capable execution path.
    pub pyramid_source: PyramidSource,
    /// Control grid used by the native cubic B-spline model.
    pub bspline: ControlGrid,
    /// Optional final final B-spline grid spacing in voxels. This is
    /// resolved to [`bspline`](Self::bspline) once the full-resolution shape is
    /// known, before optimization starts.
    pub bspline_final_grid_spacing: Option<[f64; 3]>,
    /// Quadratic neighbour penalty on native B-spline control coefficients.
    pub bspline_smoothness: f64,
    /// Also stop once the cost change is no larger than this fraction of the
    /// cost's magnitude: a tolerance that does not depend on the metric's or
    /// the image's scale. `None` keeps the absolute tests alone.
    pub relative_tolerance: Option<f64>,
    /// Measure an affine step in **scaled** parameters for the step bound and
    /// the step tolerance: each matrix entry weighted by the root mean square
    /// of the coordinate it multiplies over the lattice, so both are in voxels
    /// of displacement rather than in raw matrix units (elastix's
    /// `AutomaticScalesEstimation`). `false` keeps the raw Euclidean norm.
    pub scaled_affine_steps: bool,
    /// Worker threads for [`resident`] and the coarse-to-fine levels; with
    /// more than one, each level is cut into blocks of 32.
    pub workers: usize,
}

/// Native optimizer stopping condition observed for a completed run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitConvergence {
    /// The parameter update norm fell below [`VolumeFitParams::tolerance`].
    StepTolerance,
    /// The absolute cost change fell below [`VolumeFitParams::tolerance`].
    CostTolerance,
}

/// Diagnostic measurements from a volume fitting run.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeFitDiagnostics {
    /// Transform produced from the final optimizer state.
    pub transform: FittedTransform,
    /// Number of optimizer substages completed after the dense initializer.
    pub substages: usize,
    /// Final metric cost, including regularization terms.
    pub final_cost: f64,
    /// Euclidean norm of the final accepted parameter update.
    pub final_step_norm: f64,
    /// Absolute cost change between the final two measured states.
    pub final_cost_delta: Option<f64>,
    /// Criterion that ended the native optimizer.
    pub converged_by: FitConvergence,
}

impl Default for VolumeFitParams {
    fn default() -> Self {
        Self {
            model: TransformModel::Translation,
            metric: Metric::MeanSquares,
            metric_bins: 32,
            sampling: Sampling::All,
            interpolator: Interpolator::Linear,
            limit: SubstageLimit::of(16).expect("a positive limit"),
            search_radius: 8,
            max_step: 2.0,
            tolerance: 1.0e-4,
            damping: 1.0e-6,
            geometry: SpatialFrame::unit(),
            pyramid: None,
            pyramid_smoothing: None,
            pyramid_source: PyramidSource::Resident,
            bspline: ControlGrid::default(),
            bspline_final_grid_spacing: None,
            bspline_smoothness: 0.0,
            relative_tolerance: None,
            scaled_affine_steps: false,
            workers: 1,
        }
    }
}

/// Source policy for optimization pyramid levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyramidSource {
    /// Derive resident optimization levels from the full-resolution arrays.
    Resident,
    /// Use already built OME-Zarr multiscale levels.
    OmeZarr,
    /// Use OME-Zarr levels when present; otherwise build the requested levels.
    OmeZarrOrBuild,
}

impl VolumeFitParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn affine(mut self) -> Self {
        self.model = TransformModel::Affine;
        self.limit = SubstageLimit::of(64).expect("a positive limit");
        self
    }

    pub fn bspline(mut self) -> Self {
        self.model = TransformModel::BSpline;
        self.limit = SubstageLimit::of(64).expect("a positive limit");
        self
    }

    pub fn with_geometry(mut self, geometry: SpatialFrame) -> Self {
        self.geometry = geometry;
        self
    }

    pub fn with_pyramid(mut self, pyramid: PyramidSchedule) -> Self {
        self.pyramid = Some(pyramid);
        self
    }

    pub fn with_pyramid_smoothing(mut self, smoothing: PyramidSmoothing) -> Self {
        self.pyramid_smoothing = Some(smoothing);
        self
    }

    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_metric_bins(mut self, bins: usize) -> Result<Self> {
        if bins < 4 {
            return Err(Error::InvalidArgument(format!(
                "volume fit: histogram metrics need at least 4 bins, got {bins}"
            )));
        }
        self.metric_bins = bins;
        Ok(self)
    }

    pub fn with_sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }

    pub fn with_interpolator(mut self, interpolator: Interpolator) -> Self {
        self.interpolator = interpolator;
        self
    }

    pub fn with_bspline_grid(mut self, grid: ControlGrid) -> Self {
        self.bspline = grid;
        self.bspline_final_grid_spacing = None;
        self
    }

    pub fn with_bspline_final_grid_spacing(mut self, spacing: [f64; 3]) -> Result<Self> {
        validate_bspline_final_grid_spacing(spacing)?;
        self.bspline_final_grid_spacing = Some(spacing);
        Ok(self)
    }

    pub fn with_bspline_smoothness(mut self, smoothness: f64) -> Result<Self> {
        if smoothness < 0.0 || !smoothness.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "volume fit: B-spline smoothness must be finite and non-negative, got \
                 {smoothness}"
            )));
        }
        self.bspline_smoothness = smoothness;
        Ok(self)
    }

    fn resolved_for_shape(&self, shape: [usize; 3]) -> Result<Self> {
        let mut resolved = self.clone();
        if resolved.model == TransformModel::BSpline {
            if let Some(spacing) = resolved.bspline_final_grid_spacing {
                resolved.bspline = bspline_grid_from_final_spacing(shape, spacing)?;
                resolved.bspline_final_grid_spacing = None;
            }
        }
        Ok(resolved)
    }
}

/// Transform model optimized by the volume fitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformModel {
    Translation,
    Affine,
    BSpline,
}

/// Cubic B-spline control grid size, in control points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlGrid {
    pub size: [usize; 3],
}

impl Default for ControlGrid {
    fn default() -> Self {
        Self { size: [4, 4, 4] }
    }
}

impl ControlGrid {
    pub fn new(size: [usize; 3]) -> Result<Self> {
        if size.iter().any(|extent| *extent < 4) {
            return Err(Error::InvalidArgument(format!(
                "cubic B-spline grid {size:?} must have at least four control points per axis"
            )));
        }
        Ok(Self { size })
    }

    fn control_points(self) -> usize {
        self.size[0] * self.size[1] * self.size[2]
    }
}

fn validate_bspline_final_grid_spacing(spacing: [f64; 3]) -> Result<()> {
    if spacing.iter().any(|axis| *axis <= 0.0 || !axis.is_finite()) {
        return Err(Error::InvalidArgument(format!(
            "final B-spline grid spacing {spacing:?} must be finite and positive"
        )));
    }
    Ok(())
}

fn bspline_grid_from_final_spacing(shape: [usize; 3], spacing: [f64; 3]) -> Result<ControlGrid> {
    validate_bspline_final_grid_spacing(spacing)?;
    let mut size = [4usize; 3];
    for axis in 0..3 {
        size[axis] = ((shape[axis].max(1) as f64 / spacing[axis]).ceil() as usize + 3).max(4);
    }
    ControlGrid::new(size)
}

/// Image similarity metric optimized by the global update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    MeanSquares,
    NormalizedCorrelation,
    MutualInformation,
}

/// Sampling policy for fixed-image voxels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampling {
    All,
    Stride([usize; 3]),
}

impl Sampling {
    fn includes(self, point: [usize; 3]) -> bool {
        match self {
            Self::All => true,
            Self::Stride(stride) => {
                point[0] % stride[0] == 0 && point[1] % stride[1] == 0 && point[2] % stride[2] == 0
            }
        }
    }
}

/// Interpolation used when sampling the moving image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolator {
    Linear,
}

/// Fitted coordinate map, expressed in the full-resolution voxel frame.
#[derive(Debug, Clone, PartialEq)]
pub struct FittedTransform {
    pub model: TransformModel,
    pub params: Vec<f64>,
    pub shape: [usize; 3],
    pub geometry: SpatialFrame,
    pub bspline: ControlGrid,
    /// The B-spline control grid's placement, when it is not the implicit one
    /// [`ControlGrid`] implies (`GridSize - 3` intervals over `shape - 2`
    /// voxels): [`fit_stochastic`] places it as elastix does.
    pub bspline_layout: Option<BSplineLayout>,
}

/// Gaussian smoothing applied while deriving optimization pyramid levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyramidSmoothing {
    pub sigma: [f64; 3],
    pub truncate: f64,
}

impl PyramidSmoothing {
    pub fn isotropic(sigma: f64, truncate: f64) -> Result<Self> {
        Self::new([sigma; 3], truncate)
    }

    pub fn new(sigma: [f64; 3], truncate: f64) -> Result<Self> {
        Gaussian::new(sigma, truncate)?;
        Ok(Self { sigma, truncate })
    }

    fn chain(self) -> Result<BlockChain> {
        Ok(BlockChain::op(SmoothOp::new(
            "volume-pyramid-smooth",
            Gaussian::new(self.sigma, self.truncate)?,
        )))
    }
}

impl TransformModel {
    fn affine_like_parameters(self) -> usize {
        match self {
            Self::Translation => 3,
            Self::Affine => 12,
            Self::BSpline => 0,
        }
    }
}

/// Coarse-to-fine volume fitting schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyramidSchedule {
    factors: Vec<[usize; 3]>,
}

impl PyramidSchedule {
    /// Per-step factors from one level to the next coarser level.
    pub fn new(factors: Vec<[usize; 3]>) -> Result<Self> {
        if factors
            .iter()
            .any(|factor| factor.contains(&0) || factor == &[1, 1, 1])
        {
            return Err(Error::InvalidArgument(format!(
                "volume pyramid factors must be positive and must shrink at least one \
                 axis, got {factors:?}"
            )));
        }
        Ok(Self { factors })
    }

    pub fn powers_of_two(levels: usize) -> Result<Self> {
        if levels == 0 {
            return Err(Error::InvalidArgument(
                "a volume pyramid with zero levels has no image to fit".to_string(),
            ));
        }
        Self::new(vec![[2, 2, 2]; levels.saturating_sub(1)])
    }

    pub fn factors(&self) -> &[[usize; 3]] {
        &self.factors
    }

    pub fn levels(&self) -> usize {
        self.factors.len() + 1
    }

    pub fn recipe(&self) -> Result<PyramidRecipe> {
        PyramidRecipe::decimation(self.factors.clone())
    }

    pub fn smoothed_recipe(&self, smoothing: PyramidSmoothing) -> Result<PyramidRecipe> {
        let levels = self
            .factors
            .iter()
            .map(|factor| LevelRecipe::new(*factor, smoothing.chain()?))
            .collect::<Result<Vec<_>>>()?;
        Ok(PyramidRecipe::new(levels))
    }
}

/// One fixed/moving optimization level.
///
/// `scale` is cumulative relative to the full-resolution lattice. A level read
/// from an OME-Zarr multiscale store can be represented here after the caller
/// attaches the storage level and reads its voxels; a level derived in memory by
/// [`pyramid_levels_from_recipe`] uses the same path.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumePyramidLevel {
    pub fixed: Voxels,
    pub moving: Voxels,
    pub scale: [usize; 3],
    pub geometry: SpatialFrame,
}

impl VolumePyramidLevel {
    pub fn new(
        fixed: Voxels,
        moving: Voxels,
        scale: [usize; 3],
        geometry: SpatialFrame,
    ) -> Result<Self> {
        if scale.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "volume pyramid level scale {scale:?} must be positive on every axis"
            )));
        }
        if fixed.shape() != moving.shape() {
            return Err(Error::InvalidArgument(format!(
                "volume pyramid level scale {scale:?} has fixed shape {:?} and moving \
                 shape {:?}; both images must be on the same lattice",
                fixed.shape(),
                moving.shape()
            )));
        }
        Ok(Self {
            fixed,
            moving,
            scale,
            geometry,
        })
    }
}

/// Storage-backed or buildable optimization pyramid inputs.
#[cfg(feature = "zarr")]
#[derive(Debug)]
pub struct VolumePyramidStoreInput<'a> {
    /// Complete stored multiscale image, when the requested levels already exist.
    pub stored: Option<&'a crate::zarr_env::MultiscaleImage>,
    /// Full-resolution resident image used to build the requested levels when
    /// `stored` is absent or too short for the requested schedule.
    pub full_resolution: &'a Voxels,
    /// Root directory for a newly built optimization pyramid.
    pub build_root: PathBuf,
}

#[cfg(feature = "zarr")]
impl<'a> VolumePyramidStoreInput<'a> {
    pub fn new(full_resolution: &'a Voxels, build_root: impl Into<PathBuf>) -> Self {
        Self {
            stored: None,
            full_resolution,
            build_root: build_root.into(),
        }
    }

    pub fn with_stored(
        stored: &'a crate::zarr_env::MultiscaleImage,
        full_resolution: &'a Voxels,
        build_root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            stored: Some(stored),
            full_resolution,
            build_root: build_root.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Moments {
    fixed_weight: f64,
    fixed_sum: [f64; 3],
    moving_weight: f64,
    moving_sum: [f64; 3],
}

#[derive(Debug, Clone)]
struct OptimizerState {
    params: Vec<f64>,
    cost: f64,
    substage: usize,
    step_norm: f64,
    cost_delta: f64,
    convergence: Option<FitConvergence>,
}

#[derive(Debug, Clone, Default)]
struct Evidence {
    moments: Moments,
    count: f64,
    cost: f64,
    fixed_sum: f64,
    fixed_square_sum: f64,
    moving_sum: f64,
    moving_square_sum: f64,
    cross_sum: f64,
    gradient: Vec<f64>,
    hessian: Vec<f64>,
    jacobian_sum: Vec<f64>,
    fixed_jacobian_sum: Vec<f64>,
    moving_jacobian_sum: Vec<f64>,
    joint_histogram: Vec<f64>,
    joint_jacobian_sum: Vec<f64>,
}

impl Evidence {
    fn combine(&mut self, other: Self) {
        self.moments.combine(other.moments);
        self.count += other.count;
        self.cost += other.cost;
        self.fixed_sum += other.fixed_sum;
        self.fixed_square_sum += other.fixed_square_sum;
        self.moving_sum += other.moving_sum;
        self.moving_square_sum += other.moving_square_sum;
        self.cross_sum += other.cross_sum;
        if self.gradient.is_empty() {
            self.gradient = vec![0.0; other.gradient.len()];
            self.hessian = vec![0.0; other.hessian.len()];
            self.jacobian_sum = vec![0.0; other.jacobian_sum.len()];
            self.fixed_jacobian_sum = vec![0.0; other.fixed_jacobian_sum.len()];
            self.moving_jacobian_sum = vec![0.0; other.moving_jacobian_sum.len()];
            self.joint_histogram = vec![0.0; other.joint_histogram.len()];
            self.joint_jacobian_sum = vec![0.0; other.joint_jacobian_sum.len()];
        }
        for axis in 0..self.gradient.len() {
            self.gradient[axis] += other.gradient[axis];
            self.jacobian_sum[axis] += other.jacobian_sum[axis];
            self.fixed_jacobian_sum[axis] += other.fixed_jacobian_sum[axis];
            self.moving_jacobian_sum[axis] += other.moving_jacobian_sum[axis];
        }
        for index in 0..self.hessian.len() {
            self.hessian[index] += other.hessian[index];
        }
        for index in 0..self.joint_histogram.len() {
            self.joint_histogram[index] += other.joint_histogram[index];
        }
        for index in 0..self.joint_jacobian_sum.len() {
            self.joint_jacobian_sum[index] += other.joint_jacobian_sum[index];
        }
    }

    fn finish_metric(
        &mut self,
        model: TransformModel,
        metric: Metric,
        bins: usize,
        damping: f64,
    ) -> Result<()> {
        match metric {
            Metric::MeanSquares => Ok(()),
            Metric::NormalizedCorrelation => self.finish_normalized_correlation(model, damping),
            Metric::MutualInformation => self.finish_mutual_information(model, bins, damping),
        }
    }

    fn finish_normalized_correlation(&mut self, model: TransformModel, damping: f64) -> Result<()> {
        let count = self.count;
        let fixed_variance = self.fixed_square_sum - self.fixed_sum * self.fixed_sum / count;
        let moving_variance = self.moving_square_sum - self.moving_sum * self.moving_sum / count;
        if fixed_variance <= f64::EPSILON
            || moving_variance <= f64::EPSILON
            || !fixed_variance.is_finite()
            || !moving_variance.is_finite()
        {
            return Err(Error::InvalidArgument(format!(
                "volume fit: normalized correlation needs non-constant overlapping \
                 samples; fixed variance {fixed_variance}, moving variance {moving_variance}"
            )));
        }
        let fixed_mean = self.fixed_sum / count;
        let moving_mean = self.moving_sum / count;
        let covariance = self.cross_sum - self.fixed_sum * self.moving_sum / count;
        let denominator = (fixed_variance * moving_variance).sqrt();
        self.cost = -covariance / denominator;
        for axis in 0..self.gradient.len() {
            let d_covariance = self.fixed_jacobian_sum[axis] - fixed_mean * self.jacobian_sum[axis];
            let d_moving_variance =
                2.0 * (self.moving_jacobian_sum[axis] - moving_mean * self.jacobian_sum[axis]);
            let d_correlation = d_covariance / denominator
                - covariance * d_moving_variance
                    / (2.0 * fixed_variance.sqrt() * moving_variance.powf(1.5));
            self.gradient[axis] = -d_correlation;
        }
        self.hessian.fill(0.0);
        let scale = (1.0 + damping.max(0.0)).max(f64::EPSILON);
        for axis in 0..self.gradient.len() {
            self.hessian[diagonal_slot(model, axis)] = scale;
        }
        Ok(())
    }

    fn finish_mutual_information(
        &mut self,
        model: TransformModel,
        bins: usize,
        damping: f64,
    ) -> Result<()> {
        if self.count <= 0.0 || self.joint_histogram.len() != bins * bins {
            return Err(Error::InvalidArgument(
                "volume fit: mutual information needs overlapping histogram samples".to_string(),
            ));
        }
        let smooth = 1.0e-6;
        let count = self.count + smooth * (bins * bins) as f64;
        let mut fixed = vec![0.0; bins];
        let mut moving = vec![0.0; bins];
        for fixed_bin in 0..bins {
            for moving_bin in 0..bins {
                let value = self.joint_histogram[joint_index(bins, fixed_bin, moving_bin)] + smooth;
                fixed[fixed_bin] += value;
                moving[moving_bin] += value;
            }
        }
        let mut score = vec![0.0; bins * bins];
        let mut mutual_information = 0.0;
        for fixed_bin in 0..bins {
            for moving_bin in 0..bins {
                let index = joint_index(bins, fixed_bin, moving_bin);
                let joint = self.joint_histogram[index] + smooth;
                let probability = joint / count;
                let ratio = joint * count / (fixed[fixed_bin] * moving[moving_bin]);
                let log_ratio = ratio.ln();
                mutual_information += probability * log_ratio;
                score[index] = log_ratio;
            }
        }
        self.cost = -mutual_information;
        self.gradient.fill(0.0);
        let width = 1.0 / bins as f64;
        for fixed_bin in 0..bins {
            for moving_bin in 0..bins {
                let lower = if moving_bin == 0 {
                    score[joint_index(bins, fixed_bin, moving_bin)]
                } else {
                    score[joint_index(bins, fixed_bin, moving_bin - 1)]
                };
                let upper = if moving_bin + 1 == bins {
                    score[joint_index(bins, fixed_bin, moving_bin)]
                } else {
                    score[joint_index(bins, fixed_bin, moving_bin + 1)]
                };
                let derivative = (upper - lower) / (2.0 * width);
                let offset = joint_index(bins, fixed_bin, moving_bin) * self.gradient.len();
                for axis in 0..self.gradient.len() {
                    self.gradient[axis] -=
                        derivative * self.joint_jacobian_sum[offset + axis] / self.count.max(1.0);
                }
            }
        }
        self.hessian.fill(0.0);
        let scale = (self.count + damping.max(0.0)).max(f64::EPSILON);
        for axis in 0..self.gradient.len() {
            self.hessian[diagonal_slot(model, axis)] = scale;
        }
        Ok(())
    }
}

impl Moments {
    fn add_fixed(&mut self, position: [f64; 3], weight: f64) {
        self.fixed_weight += weight;
        for axis in 0..3 {
            self.fixed_sum[axis] += weight * position[axis];
        }
    }

    fn add_moving(&mut self, position: [f64; 3], weight: f64) {
        self.moving_weight += weight;
        for axis in 0..3 {
            self.moving_sum[axis] += weight * position[axis];
        }
    }

    fn combine(&mut self, other: Self) {
        self.fixed_weight += other.fixed_weight;
        self.moving_weight += other.moving_weight;
        for axis in 0..3 {
            self.fixed_sum[axis] += other.fixed_sum[axis];
            self.moving_sum[axis] += other.moving_sum[axis];
        }
    }

    fn translation(self) -> Result<[f64; 3]> {
        if self.fixed_weight <= 0.0 || self.moving_weight <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "volume fit: both images need positive finite mass; fixed mass {}, \
                 moving mass {}",
                self.fixed_weight, self.moving_weight
            )));
        }
        let mut shift = [0.0; 3];
        for axis in 0..3 {
            shift[axis] = self.moving_sum[axis] / self.moving_weight
                - self.fixed_sum[axis] / self.fixed_weight;
        }
        Ok(shift)
    }
}

fn pack_state(state: OptimizerState) -> Vec<u8> {
    let mut values = state.params;
    values.push(state.cost);
    values.push(state.substage as f64);
    values.push(state.step_norm);
    values.push(state.cost_delta);
    values.push(match state.convergence {
        None => 0.0,
        Some(FitConvergence::StepTolerance) => 1.0,
        Some(FitConvergence::CostTolerance) => 2.0,
    });
    pack_f64s(&values)
}

fn parameter_count(params: &VolumeFitParams) -> usize {
    match params.model {
        TransformModel::Translation | TransformModel::Affine => {
            params.model.affine_like_parameters()
        }
        TransformModel::BSpline => 3 * params.bspline.control_points(),
    }
}

fn unpack_state(bytes: &[u8], params: &VolumeFitParams) -> Result<OptimizerState> {
    let count = parameter_count(params);
    let expected = count + 5;
    let values = unpack_f64s_dynamic(bytes, expected, "optimizer state")?;
    let state_params = values[..count].to_vec();
    let convergence = match values[count + 4] as usize {
        0 => None,
        1 => Some(FitConvergence::StepTolerance),
        2 => Some(FitConvergence::CostTolerance),
        code => {
            return Err(Error::InvalidArgument(format!(
                "volume fit: optimizer state has unknown convergence code {code}"
            )));
        }
    };
    Ok(OptimizerState {
        params: state_params,
        cost: values[count],
        substage: values[count + 1] as usize,
        step_norm: values[count + 2],
        cost_delta: values[count + 3],
        convergence,
    })
}

fn pack_evidence(evidence: Evidence) -> Vec<u8> {
    let mut values = vec![
        evidence.moments.fixed_weight,
        evidence.moments.fixed_sum[0],
        evidence.moments.fixed_sum[1],
        evidence.moments.fixed_sum[2],
        evidence.moments.moving_weight,
        evidence.moments.moving_sum[0],
        evidence.moments.moving_sum[1],
        evidence.moments.moving_sum[2],
        evidence.count,
        evidence.cost,
        evidence.fixed_sum,
        evidence.fixed_square_sum,
        evidence.moving_sum,
        evidence.moving_square_sum,
        evidence.cross_sum,
    ];
    values.extend(evidence.gradient);
    values.extend(evidence.hessian);
    values.extend(evidence.jacobian_sum);
    values.extend(evidence.fixed_jacobian_sum);
    values.extend(evidence.moving_jacobian_sum);
    values.extend(evidence.joint_histogram);
    values.extend(evidence.joint_jacobian_sum);
    pack_f64s(&values)
}

fn unpack_evidence(bytes: &[u8], params: &VolumeFitParams) -> Result<Evidence> {
    let n = parameter_count(params);
    let h = hessian_len(params.model, n);
    let joint = params.metric_bins * params.metric_bins;
    let values = unpack_f64s_dynamic(bytes, 15 + 4 * n + h + joint + joint * n, "evidence")?;
    let gradient_start = 15;
    let hessian_start = gradient_start + n;
    let jacobian_start = hessian_start + h;
    let fixed_jacobian_start = jacobian_start + n;
    let moving_jacobian_start = fixed_jacobian_start + n;
    let joint_histogram_start = moving_jacobian_start + n;
    let joint_jacobian_start = joint_histogram_start + joint;
    Ok(Evidence {
        moments: Moments {
            fixed_weight: values[0],
            fixed_sum: [values[1], values[2], values[3]],
            moving_weight: values[4],
            moving_sum: [values[5], values[6], values[7]],
        },
        count: values[8],
        cost: values[9],
        fixed_sum: values[10],
        fixed_square_sum: values[11],
        moving_sum: values[12],
        moving_square_sum: values[13],
        cross_sum: values[14],
        gradient: values[gradient_start..hessian_start].to_vec(),
        hessian: values[hessian_start..jacobian_start].to_vec(),
        jacobian_sum: values[jacobian_start..fixed_jacobian_start].to_vec(),
        fixed_jacobian_sum: values[fixed_jacobian_start..moving_jacobian_start].to_vec(),
        moving_jacobian_sum: values[moving_jacobian_start..joint_histogram_start].to_vec(),
        joint_histogram: values[joint_histogram_start..joint_jacobian_start].to_vec(),
        joint_jacobian_sum: values[joint_jacobian_start..].to_vec(),
    })
}

fn pack_f64s(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn unpack_f64s_dynamic(bytes: &[u8], count: usize, what: &str) -> Result<Vec<f64>> {
    if bytes.len() != count * 8 {
        return Err(Error::InvalidArgument(format!(
            "volume fit: {what} has {} byte(s), expected {}",
            bytes.len(),
            count * 8
        )));
    }
    let mut values = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(8) {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(chunk);
        values.push(f64::from_le_bytes(raw));
    }
    Ok(values)
}

struct FitByMoments {
    params: VolumeFitParams,
    initial: Option<Vec<f64>>,
    /// The lattice the fit runs over.
    volume: [usize; 3],
}

impl IterativeReduceOp for FitByMoments {
    fn name(&self) -> &'static str {
        "volume fitting-compute"
    }

    fn limit(&self) -> SubstageLimit {
        self.params.limit
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(
            moving_image(),
            Reach::symmetric([self.params.search_radius + 1; 3]),
        )
        .holding(Dtype::F64)]
    }

    fn initial_state(&self, _volume: [usize; 3]) -> Result<Vec<u8>> {
        Ok(pack_state(OptimizerState {
            params: self
                .initial
                .clone()
                .unwrap_or_else(|| zero_params(&self.params)),
            cost: f64::INFINITY,
            substage: 0,
            step_norm: f64::INFINITY,
            cost_delta: f64::INFINITY,
            convergence: None,
        }))
    }

    fn map_block(&self, substage: usize, state: &[u8], block: &ReduceBlock<'_>) -> Result<Vec<u8>> {
        let state = unpack_state(state, &self.params)?;
        let fixed = match block.pixels()? {
            BlockBuf::Array(array) => array.view::<f64>()?,
            BlockBuf::Accounted { .. } => {
                return Err(Error::InvalidArgument(
                    "volume fit needs real image values, not a simulated block".to_string(),
                ));
            }
        };
        let moving = block.sources().get(moving_image())?;
        let moving = moving.view::<f64>()?;
        let n = parameter_count(&self.params);
        let joint = self.params.metric_bins * self.params.metric_bins;
        let mut evidence = Evidence {
            gradient: vec![0.0; n],
            hessian: vec![0.0; hessian_len(self.params.model, n)],
            jacobian_sum: vec![0.0; n],
            fixed_jacobian_sum: vec![0.0; n],
            moving_jacobian_sum: vec![0.0; n],
            joint_histogram: vec![0.0; joint],
            joint_jacobian_sum: vec![0.0; joint * n],
            ..Evidence::default()
        };
        for gz in block.valid.start[2]..block.valid.start[2] + block.valid.shape[2] {
            for gy in block.valid.start[1]..block.valid.start[1] + block.valid.shape[1] {
                for gx in block.valid.start[0]..block.valid.start[0] + block.valid.shape[0] {
                    let local = [
                        gx - block.at.offset[0],
                        gy - block.at.offset[1],
                        gz - block.at.offset[2],
                    ];
                    let fixed_value = fixed[[local[0], local[1], local[2]]];
                    let position = [gx as f64, gy as f64, gz as f64];
                    let fixed_weight = positive_weight(fixed_value);
                    if fixed_weight > 0.0 {
                        evidence.moments.add_fixed(position, fixed_weight);
                    }
                    let moving_weight = positive_weight(moving[[local[0], local[1], local[2]]]);
                    if moving_weight > 0.0 {
                        evidence.moments.add_moving(position, moving_weight);
                    }
                    if substage > 0 && self.params.sampling.includes([gx, gy, gz]) {
                        let mapped = transform_point(
                            &self.params,
                            block.grid.volume(),
                            &state.params,
                            position,
                        );
                        let sample = [
                            mapped[0] - block.at.offset[0] as f64,
                            mapped[1] - block.at.offset[1] as f64,
                            mapped[2] - block.at.offset[2] as f64,
                        ];
                        if let Some((moving_value, gradient)) =
                            sample_value_and_gradient(self.params.interpolator, &moving, sample)
                        {
                            let residual = moving_value - fixed_value;
                            evidence.count += 1.0;
                            let jacobian = parameter_jacobian(
                                &self.params,
                                block.grid.volume(),
                                gradient,
                                position,
                            );
                            evidence.fixed_sum += fixed_value;
                            evidence.fixed_square_sum += fixed_value * fixed_value;
                            evidence.moving_sum += moving_value;
                            evidence.moving_square_sum += moving_value * moving_value;
                            evidence.cross_sum += fixed_value * moving_value;
                            match self.params.metric {
                                Metric::MeanSquares => {
                                    evidence.cost += 0.5 * residual * residual;
                                    for (slot, value) in evidence.gradient.iter_mut().zip(&jacobian)
                                    {
                                        *slot += residual * value;
                                    }
                                    add_outer_for_model(
                                        self.params.model,
                                        &mut evidence.hessian,
                                        &jacobian,
                                    );
                                }
                                Metric::NormalizedCorrelation => {
                                    for axis in 0..n {
                                        evidence.jacobian_sum[axis] += jacobian[axis];
                                        evidence.fixed_jacobian_sum[axis] +=
                                            fixed_value * jacobian[axis];
                                        evidence.moving_jacobian_sum[axis] +=
                                            moving_value * jacobian[axis];
                                    }
                                }
                                Metric::MutualInformation => {
                                    let fixed_bin =
                                        intensity_bin(fixed_value, self.params.metric_bins);
                                    let moving_bin =
                                        intensity_bin(moving_value, self.params.metric_bins);
                                    let joint_index =
                                        joint_index(self.params.metric_bins, fixed_bin, moving_bin);
                                    evidence.joint_histogram[joint_index] += 1.0;
                                    let intensity_derivative =
                                        bounded_intensity_derivative(moving_value);
                                    let offset = joint_index * n;
                                    for axis in 0..n {
                                        evidence.joint_jacobian_sum[offset + axis] +=
                                            intensity_derivative * jacobian[axis];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(pack_evidence(evidence))
    }

    fn update(&self, substage: usize, state: &[u8], partials: &[Partial]) -> Result<StateUpdate> {
        let previous = unpack_state(state, &self.params)?;
        let n = parameter_count(&self.params);
        let joint = self.params.metric_bins * self.params.metric_bins;
        let mut total = Evidence {
            gradient: vec![0.0; n],
            hessian: vec![0.0; hessian_len(self.params.model, n)],
            jacobian_sum: vec![0.0; n],
            fixed_jacobian_sum: vec![0.0; n],
            moving_jacobian_sum: vec![0.0; n],
            joint_histogram: vec![0.0; joint],
            joint_jacobian_sum: vec![0.0; joint * n],
            ..Evidence::default()
        };
        for partial in partials {
            total.combine(unpack_evidence(&partial.bytes, &self.params)?);
        }
        if substage == 0 {
            // A seeded run starts **at the seed, in full**. Reducing the seed to
            // its translation and rebuilding the parameter vector around that
            // would discard everything else it carries — the linear part of an
            // affine, every control coefficient of a B-spline — which is
            // precisely what the coarse level of a pyramid seeds the level below
            // it with. `tests/align.rs` carries the readout that measures it.
            let params = match self.initial.as_ref() {
                Some(seed) => seed.clone(),
                None => params_with_translation(&self.params, total.moments.translation()?),
            };
            let next = OptimizerState {
                params,
                cost: f64::INFINITY,
                substage: 0,
                step_norm: f64::INFINITY,
                cost_delta: f64::INFINITY,
                convergence: None,
            };
            return Ok(StateUpdate::continuing(pack_state(next)));
        }
        if total.count <= 0.0 {
            return Err(Error::InvalidArgument(
                "volume fit: no overlapping moving samples remain at the current \
                 translation"
                    .to_string(),
            ));
        }
        total.finish_metric(
            self.params.model,
            self.params.metric,
            self.params.metric_bins,
            self.params.damping,
        )?;
        total.cost += apply_bspline_smoothness(
            &self.params,
            &previous.params,
            &mut total.gradient,
            &mut total.hessian,
        );
        let step = solve_step(
            self.params.model,
            total.hessian,
            total.gradient,
            self.params.damping,
        )?;
        let weights = step_weights(&self.params, self.volume);
        let step = clamp_step_weighted(step, self.params.max_step, weights.as_deref());
        let mut params = previous.params;
        for (param, delta) in params.iter_mut().zip(&step) {
            *param += *delta;
        }
        let step_norm = weighted_norm(&step, weights.as_deref());
        let cost_delta = if previous.cost.is_finite() {
            (previous.cost - total.cost).abs()
        } else {
            f64::INFINITY
        };
        let relative = self
            .params
            .relative_tolerance
            .is_some_and(|fraction| cost_delta <= fraction * total.cost.abs());
        let convergence = if step_norm <= self.params.tolerance {
            Some(FitConvergence::StepTolerance)
        } else if cost_delta <= self.params.tolerance || relative {
            Some(FitConvergence::CostTolerance)
        } else {
            None
        };
        let next = OptimizerState {
            params,
            cost: total.cost,
            substage,
            step_norm,
            cost_delta,
            convergence,
        };
        let bytes = pack_state(next);
        if convergence.is_some() {
            Ok(StateUpdate::converged(bytes))
        } else {
            Ok(StateUpdate::continuing(bytes))
        }
    }

    fn state_stream(&self) -> &'static str {
        STATE_STREAM
    }

    fn state_lifecycle(&self) -> Lifecycle {
        Lifecycle::Persistent
    }
}

fn zero_params(params: &VolumeFitParams) -> Vec<f64> {
    match params.model {
        TransformModel::Translation => vec![0.0; 3],
        TransformModel::Affine => vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        TransformModel::BSpline => vec![0.0; 3 * params.bspline.control_points()],
    }
}

fn params_with_translation(params: &VolumeFitParams, translation: [f64; 3]) -> Vec<f64> {
    let mut values = zero_params(params);
    match params.model {
        TransformModel::Translation => values.copy_from_slice(&translation),
        TransformModel::Affine => values[9..12].copy_from_slice(&translation),
        TransformModel::BSpline => {
            seed_bspline_translation(params.bspline, &mut values, translation)
        }
    }
    values
}

fn transform_point(
    config: &VolumeFitParams,
    shape: [usize; 3],
    params: &[f64],
    point: [f64; 3],
) -> [f64; 3] {
    match config.model {
        TransformModel::Translation => [
            point[0] + params[0],
            point[1] + params[1],
            point[2] + params[2],
        ],
        TransformModel::Affine => [
            params[0] * point[0] + params[1] * point[1] + params[2] * point[2] + params[9],
            params[3] * point[0] + params[4] * point[1] + params[5] * point[2] + params[10],
            params[6] * point[0] + params[7] * point[1] + params[8] * point[2] + params[11],
        ],
        TransformModel::BSpline => {
            let displacement = bspline_displacement(config.bspline, shape, params, point);
            [
                point[0] + displacement[0],
                point[1] + displacement[1],
                point[2] + displacement[2],
            ]
        }
    }
}

fn parameter_jacobian(
    config: &VolumeFitParams,
    shape: [usize; 3],
    image_gradient: [f64; 3],
    point: [f64; 3],
) -> Vec<f64> {
    match config.model {
        TransformModel::Translation => image_gradient.to_vec(),
        TransformModel::Affine => vec![
            image_gradient[0] * point[0],
            image_gradient[0] * point[1],
            image_gradient[0] * point[2],
            image_gradient[1] * point[0],
            image_gradient[1] * point[1],
            image_gradient[1] * point[2],
            image_gradient[2] * point[0],
            image_gradient[2] * point[1],
            image_gradient[2] * point[2],
            image_gradient[0],
            image_gradient[1],
            image_gradient[2],
        ],
        TransformModel::BSpline => {
            bspline_parameter_jacobian(config.bspline, shape, image_gradient, point)
        }
    }
}

fn seed_bspline_translation(grid: ControlGrid, params: &mut [f64], translation: [f64; 3]) {
    let per_block = grid.control_points();
    for axis in 0..3 {
        for slot in &mut params[axis * per_block..(axis + 1) * per_block] {
            *slot = translation[axis];
        }
    }
}

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

fn bspline_displacement(
    grid: ControlGrid,
    shape: [usize; 3],
    params: &[f64],
    point: [f64; 3],
) -> [f64; 3] {
    let Some(weights) = bspline_weights(grid, shape, point) else {
        return [0.0; 3];
    };
    let per_block = grid.control_points();
    let mut displacement = [0.0; 3];
    for (flat, weight) in weights {
        for axis in 0..3 {
            displacement[axis] += weight * params[axis * per_block + flat];
        }
    }
    displacement
}

fn bspline_parameter_jacobian(
    grid: ControlGrid,
    shape: [usize; 3],
    image_gradient: [f64; 3],
    point: [f64; 3],
) -> Vec<f64> {
    let mut jacobian = vec![0.0; 3 * grid.control_points()];
    if let Some(weights) = bspline_weights(grid, shape, point) {
        let per_block = grid.control_points();
        for (flat, weight) in weights {
            for axis in 0..3 {
                jacobian[axis * per_block + flat] = image_gradient[axis] * weight;
            }
        }
    }
    jacobian
}

fn bspline_weights(
    grid: ControlGrid,
    shape: [usize; 3],
    point: [f64; 3],
) -> Option<Vec<(usize, f64)>> {
    let (origin, spacing) = bspline_lattice(grid, shape);
    let mut u = [0.0; 3];
    for axis in 0..3 {
        u[axis] = (point[axis] - origin[axis]) / spacing[axis];
        if u[axis] < 1.0 || u[axis] >= grid.size[axis] as f64 - 2.0 {
            return None;
        }
    }
    let mut starts = [0isize; 3];
    let mut weights = [[0.0f64; 4]; 3];
    for axis in 0..3 {
        let start = u[axis].floor() as isize - 1;
        starts[axis] = start;
        for offset in 0..4 {
            weights[axis][offset] = cubic_basis(u[axis] - (start + offset as isize) as f64);
        }
    }
    let mut out = Vec::with_capacity(64);
    for k in 0..4 {
        let gk = starts[2] + k as isize;
        let wk = weights[2][k];
        for j in 0..4 {
            let gj = starts[1] + j as isize;
            let wjk = wk * weights[1][j];
            for i in 0..4 {
                let gi = starts[0] + i as isize;
                let weight = wjk * weights[0][i];
                if weight == 0.0 {
                    continue;
                }
                let flat = gi as usize + grid.size[0] * (gj as usize + grid.size[1] * gk as usize);
                out.push((flat, weight));
            }
        }
    }
    Some(out)
}

fn cubic_basis(t: f64) -> f64 {
    let t = t.abs();
    if t < 1.0 {
        2.0 / 3.0 - t * t + t * t * t / 2.0
    } else if t < 2.0 {
        let u = 2.0 - t;
        u * u * u / 6.0
    } else {
        0.0
    }
}

fn apply_bspline_smoothness(
    config: &VolumeFitParams,
    params: &[f64],
    gradient: &mut [f64],
    hessian: &mut [f64],
) -> f64 {
    if config.model != TransformModel::BSpline || config.bspline_smoothness == 0.0 {
        return 0.0;
    }
    let weight = config.bspline_smoothness;
    let grid = config.bspline;
    let per_block = grid.control_points();
    let mut cost = 0.0;
    for axis in 0..3 {
        for z in 0..grid.size[2] {
            for y in 0..grid.size[1] {
                for x in 0..grid.size[0] {
                    let here = axis * per_block + bspline_flat(grid, x, y, z);
                    if x + 1 < grid.size[0] {
                        cost += smooth_pair(
                            weight,
                            params,
                            gradient,
                            hessian,
                            here,
                            axis * per_block + bspline_flat(grid, x + 1, y, z),
                        );
                    }
                    if y + 1 < grid.size[1] {
                        cost += smooth_pair(
                            weight,
                            params,
                            gradient,
                            hessian,
                            here,
                            axis * per_block + bspline_flat(grid, x, y + 1, z),
                        );
                    }
                    if z + 1 < grid.size[2] {
                        cost += smooth_pair(
                            weight,
                            params,
                            gradient,
                            hessian,
                            here,
                            axis * per_block + bspline_flat(grid, x, y, z + 1),
                        );
                    }
                }
            }
        }
    }
    cost
}

fn smooth_pair(
    weight: f64,
    params: &[f64],
    gradient: &mut [f64],
    hessian: &mut [f64],
    a: usize,
    b: usize,
) -> f64 {
    let delta = params[a] - params[b];
    gradient[a] += weight * delta;
    gradient[b] -= weight * delta;
    // B-spline evidence keeps the diagonal alone.
    hessian[a] += weight;
    hessian[b] += weight;
    0.5 * weight * delta * delta
}

fn bspline_flat(grid: ControlGrid, x: usize, y: usize, z: usize) -> usize {
    x + grid.size[0] * (y + grid.size[1] * z)
}

/// The Hessian evidence a model accumulates: packed symmetric `n (n + 1) / 2`
/// for translation and affine, and **the diagonal alone** for a B-spline, whose
/// update uses nothing else. (A packed Hessian per block grows with the square
/// of the control point count: 25-voxel spacing on a 25 µm hemisphere is
/// thousands of nodes and tens of gigabytes per block.)
fn hessian_len(model: TransformModel, n: usize) -> usize {
    match model {
        TransformModel::BSpline => n,
        TransformModel::Translation | TransformModel::Affine => symmetric_len(n),
    }
}

fn diagonal_slot(model: TransformModel, axis: usize) -> usize {
    match model {
        TransformModel::BSpline => axis,
        TransformModel::Translation | TransformModel::Affine => symmetric_index(axis, axis),
    }
}

fn symmetric_len(n: usize) -> usize {
    n * (n + 1) / 2
}

fn symmetric_index(row: usize, col: usize) -> usize {
    let (row, col) = if row <= col { (row, col) } else { (col, row) };
    col * (col + 1) / 2 + row
}

fn add_outer(hessian: &mut [f64], jacobian: &[f64]) {
    for col in 0..jacobian.len() {
        for row in 0..=col {
            hessian[symmetric_index(row, col)] += jacobian[row] * jacobian[col];
        }
    }
}

fn add_outer_for_model(model: TransformModel, hessian: &mut [f64], jacobian: &[f64]) {
    match model {
        TransformModel::Translation | TransformModel::Affine => add_outer(hessian, jacobian),
        TransformModel::BSpline => {
            for (axis, value) in jacobian.iter().enumerate() {
                hessian[axis] += value * value;
            }
        }
    }
}

fn solve_step(
    model: TransformModel,
    hessian: Vec<f64>,
    gradient: Vec<f64>,
    damping: f64,
) -> Result<Vec<f64>> {
    if model == TransformModel::BSpline {
        return solve_diagonal_step(hessian, gradient, damping);
    }
    let n = gradient.len();
    let mut matrix = vec![vec![0.0; n]; n];
    for col in 0..n {
        for row in 0..=col {
            let value = hessian[symmetric_index(row, col)];
            matrix[row][col] = value;
            matrix[col][row] = value;
        }
    }
    for (axis, row) in matrix.iter_mut().enumerate() {
        row[axis] += damping.max(0.0);
    }
    let rhs = gradient.into_iter().map(|value| -value).collect();
    solve_linear(matrix, rhs)
}

fn solve_diagonal_step(hessian: Vec<f64>, gradient: Vec<f64>, damping: f64) -> Result<Vec<f64>> {
    let mut step = vec![0.0; gradient.len()];
    for axis in 0..gradient.len() {
        let diagonal = hessian[axis] + damping.max(0.0);
        if diagonal <= f64::EPSILON || !diagonal.is_finite() {
            step[axis] = 0.0;
        } else {
            step[axis] = -gradient[axis] / diagonal;
        }
    }
    Ok(step)
}

fn solve_linear(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    for pivot in 0..n {
        let mut best = pivot;
        let mut best_value = matrix[pivot][pivot].abs();
        for (row, values) in matrix.iter().enumerate().skip(pivot + 1) {
            let value = values[pivot].abs();
            if value > best_value {
                best = row;
                best_value = value;
            }
        }
        if best_value <= f64::EPSILON || !best_value.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "volume fit: singular normal equations at column {pivot}"
            )));
        }
        if best != pivot {
            matrix.swap(best, pivot);
            rhs.swap(best, pivot);
        }
        let divisor = matrix[pivot][pivot];
        for col in pivot..n {
            matrix[pivot][col] /= divisor;
        }
        rhs[pivot] /= divisor;
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for col in pivot..n {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Ok(rhs)
}

/// Per-parameter weights of the step norm: `None` for the raw Euclidean
/// norm; for a scaled affine, `E[x_c^2]` over the lattice for matrix entry
/// `(r, c)` and 1 for the translation, so `sqrt(Σ w_p Δ_p²)` is a displacement
/// in voxels.
fn step_weights(params: &VolumeFitParams, volume: [usize; 3]) -> Option<Vec<f64>> {
    if !(params.scaled_affine_steps && params.model == TransformModel::Affine) {
        return None;
    }
    // Mean of x^2 over 0..n is (n - 1)(2n - 1) / 6.
    let second: [f64; 3] = volume.map(|n| {
        let n = n as f64;
        ((n - 1.0) * (2.0 * n - 1.0) / 6.0).max(1.0)
    });
    let mut weights = vec![1.0; 12];
    for r in 0..3 {
        for c in 0..3 {
            weights[3 * r + c] = second[c];
        }
    }
    Some(weights)
}

fn weighted_norm(vector: &[f64], weights: Option<&[f64]>) -> f64 {
    match weights {
        None => norm(vector),
        Some(weights) => vector
            .iter()
            .zip(weights)
            .map(|(v, w)| w * v * v)
            .sum::<f64>()
            .sqrt(),
    }
}

fn clamp_step_weighted(mut step: Vec<f64>, max_step: f64, weights: Option<&[f64]>) -> Vec<f64> {
    let length = weighted_norm(&step, weights);
    if max_step > 0.0 && length > max_step {
        for value in &mut step {
            *value *= max_step / length;
        }
    }
    step
}

fn norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn positive_weight(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

fn joint_index(bins: usize, fixed_bin: usize, moving_bin: usize) -> usize {
    fixed_bin * bins + moving_bin
}

fn intensity_bin(value: f64, bins: usize) -> usize {
    let coordinate = bounded_intensity(value);
    ((coordinate * bins as f64).floor() as usize).min(bins - 1)
}

fn bounded_intensity(value: f64) -> f64 {
    if value.is_finite() {
        value.atan() / std::f64::consts::PI + 0.5
    } else if value.is_sign_negative() {
        0.0
    } else {
        1.0
    }
}

fn bounded_intensity_derivative(value: f64) -> f64 {
    if value.is_finite() {
        1.0 / (std::f64::consts::PI * (1.0 + value * value))
    } else {
        0.0
    }
}

fn sample_value_and_gradient(
    interpolator: Interpolator,
    image: &ArrayView3<'_, f64>,
    point: [f64; 3],
) -> Option<(f64, [f64; 3])> {
    match interpolator {
        Interpolator::Linear => value_and_gradient(image, point),
    }
}

fn value_and_gradient(image: &ArrayView3<'_, f64>, point: [f64; 3]) -> Option<(f64, [f64; 3])> {
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
    let c000 = image[[x, y, z]];
    let c100 = image[[x + 1, y, z]];
    let c010 = image[[x, y + 1, z]];
    let c110 = image[[x + 1, y + 1, z]];
    let c001 = image[[x, y, z + 1]];
    let c101 = image[[x + 1, y, z + 1]];
    let c011 = image[[x, y + 1, z + 1]];
    let c111 = image[[x + 1, y + 1, z + 1]];
    let c00 = lerp(c000, c100, dx);
    let c10 = lerp(c010, c110, dx);
    let c01 = lerp(c001, c101, dx);
    let c11 = lerp(c011, c111, dx);
    let c0 = lerp(c00, c10, dy);
    let c1 = lerp(c01, c11, dy);
    let value = lerp(c0, c1, dz);
    let grad_x = lerp(
        lerp(c100 - c000, c110 - c010, dy),
        lerp(c101 - c001, c111 - c011, dy),
        dz,
    );
    let grad_y = lerp(
        lerp(c010 - c000, c110 - c100, dx),
        lerp(c011 - c001, c111 - c101, dx),
        dz,
    );
    let grad_z = lerp(
        lerp(c001 - c000, c101 - c100, dx),
        lerp(c011 - c010, c111 - c110, dx),
        dy,
    );
    Some((value, [grad_x, grad_y, grad_z]))
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn invert_3x3(matrix: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let m = matrix;
    let determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if determinant.abs() <= f64::EPSILON || !determinant.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "volume fit: direction matrix {matrix:?} has determinant {determinant} \
             and cannot be inverted"
        )));
    }
    let mut inverse = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let (r0, r1) = ((column + 1) % 3, (column + 2) % 3);
            let (c0, c1) = ((row + 1) % 3, (row + 2) % 3);
            inverse[row][column] = (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / determinant;
        }
    }
    Ok(inverse)
}

pub fn plan(params: &VolumeFitParams, shape: [usize; 3], block: usize) -> Result<Decomposition> {
    let params = params.resolved_for_shape(shape)?;
    let op = FitByMoments {
        params: params.clone(),
        initial: None,
        volume: shape,
    };
    let edge = block.max(1);
    let grid = BlockGrid::new(shape, [edge, edge, edge])?;
    Ok(Decomposition {
        volume: shape,
        dtype: Dtype::F64,
        phases: vec![iterative_reduce_phase(&op, grid)?],
        chain_reach: [0, 0, 0],
    })
}

pub fn run(
    params: &VolumeFitParams,
    fixed: &Voxels,
    moving: &Voxels,
    block: usize,
) -> Result<FittedTransform> {
    run_with_workers(params, fixed, moving, block, 1)
}

pub fn run_with_workers(
    params: &VolumeFitParams,
    fixed: &Voxels,
    moving: &Voxels,
    block: usize,
    workers: usize,
) -> Result<FittedTransform> {
    let params = params.resolved_for_shape(fixed.shape())?;
    Ok(run_with_initial(&params, fixed, moving, block, workers, None)?.0)
}

/// Run volume fitting and return convergence diagnostics alongside the
/// transform.
pub fn run_with_diagnostics(
    params: &VolumeFitParams,
    fixed: &Voxels,
    moving: &Voxels,
    block: usize,
    workers: usize,
) -> Result<VolumeFitDiagnostics> {
    run_with_seed_diagnostics(params, fixed, moving, block, workers, None)
}

pub fn run_with_seed_diagnostics(
    params: &VolumeFitParams,
    fixed: &Voxels,
    moving: &Voxels,
    block: usize,
    workers: usize,
    initial: Option<Vec<f64>>,
) -> Result<VolumeFitDiagnostics> {
    let params = params.resolved_for_shape(fixed.shape())?;
    let (transform, state) = run_with_initial(&params, fixed, moving, block, workers, initial)?;
    let final_cost_delta = if state.cost_delta.is_finite() {
        Some(state.cost_delta)
    } else {
        None
    };
    let converged_by = state.convergence.ok_or_else(|| {
        Error::InvalidArgument(
            "volume fit: final state did not record a convergence criterion".to_string(),
        )
    })?;
    Ok(VolumeFitDiagnostics {
        transform,
        substages: state.substage,
        final_cost: state.cost,
        final_step_norm: state.step_norm,
        final_cost_delta,
        converged_by,
    })
}

/// Run one optimization level over resident volumes, optionally seeded.
///
/// Returns the fitted transform together with the final [`OptimizerState`], so
/// that a pyramid level can hand its whole parameter vector — not merely the
/// translation it reduces to — to the level below it.
fn run_with_initial(
    params: &VolumeFitParams,
    fixed: &Voxels,
    moving: &Voxels,
    block: usize,
    workers: usize,
    initial: Option<Vec<f64>>,
) -> Result<(FittedTransform, OptimizerState)> {
    validate_config(params)?;
    if fixed.shape() != moving.shape() {
        return Err(Error::InvalidArgument(format!(
            "volume fit: fixed image has shape {:?} and moving image has shape {:?}; \
             this first native estimator expects both images on the same lattice",
            fixed.shape(),
            moving.shape()
        )));
    }
    if fixed.dtype() != Dtype::F64 || moving.dtype() != Dtype::F64 {
        return Err(Error::InvalidArgument(format!(
            "volume fit: fixed image is {:?} and moving image is {:?}; expected f64 \
             volumes",
            fixed.dtype(),
            moving.dtype()
        )));
    }
    let shape = fixed.shape();
    validate_images(fixed, moving, shape)?;
    let op = FitByMoments {
        params: params.clone(),
        initial,
        volume: shape,
    };
    let decomposition = plan(params, shape, block)?;
    let env = ArrayEnvironment::with_inputs(
        fixed.clone(),
        vec![moving.clone()],
        &decomposition,
        [block.max(1), block.max(1), block.max(1)],
    )?;
    let hints = Hints {
        concurrency: workers.max(1),
        ..Hints::default()
    };
    execute_phases(
        "volume fitting-compute",
        &BlockWorkflow::new(BlockChain::sequence(Vec::new()), shape, Dtype::F64),
        &decomposition,
        &hints,
        &env,
        &[],
        &[PhaseWork::IterateReduce(&op)],
    )?;
    let bytes = env
        .read_sidecar(STATE_STREAM, 0, [0, 0, 0])?
        .ok_or_else(|| {
            Error::InvalidArgument("volume fit: final state sidecar was not written".to_string())
        })?;
    let state = unpack_state(&bytes, params)?;
    let transform = FittedTransform {
        model: params.model,
        params: state.params.clone(),
        shape,
        geometry: params.geometry.clone(),
        bspline: params.bspline,
        bspline_layout: None,
    };
    Ok((transform, state))
}

pub fn validate_config(params: &VolumeFitParams) -> Result<()> {
    if params.metric_bins < 4 {
        return Err(Error::InvalidArgument(format!(
            "volume fit: histogram metrics need at least 4 bins, got {}",
            params.metric_bins
        )));
    }
    if params.max_step < 0.0 || !params.max_step.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "volume fit: max_step must be finite and non-negative, got {}",
            params.max_step
        )));
    }
    if params.tolerance < 0.0 || !params.tolerance.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "volume fit: tolerance must be finite and non-negative, got {}",
            params.tolerance
        )));
    }
    if params.damping < 0.0 || !params.damping.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "volume fit: damping must be finite and non-negative, got {}",
            params.damping
        )));
    }
    if let Some(spacing) = params.bspline_final_grid_spacing {
        validate_bspline_final_grid_spacing(spacing)?;
    }
    if let Sampling::Stride(stride) = params.sampling {
        if stride.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "volume fit: sampling stride {stride:?} must be positive on every axis"
            )));
        }
    }
    Ok(())
}

fn validate_images(fixed: &Voxels, moving: &Voxels, shape: [usize; 3]) -> Result<()> {
    if shape.contains(&0) {
        return Err(Error::InvalidArgument(format!(
            "volume fit: empty image shape {shape:?} has no samples to fit"
        )));
    }
    if shape.iter().any(|axis| *axis < 2) {
        return Err(Error::InvalidArgument(format!(
            "volume fit: image shape {shape:?} is too small for trilinear moving-image \
             sampling; every axis needs at least two voxels"
        )));
    }
    let fixed = fixed.view::<f64>()?;
    let moving = moving.view::<f64>()?;
    let fixed_stats = ImageStats::from("fixed image", fixed)?;
    let moving_stats = ImageStats::from("moving image", moving)?;
    fixed_stats.require_signal()?;
    moving_stats.require_signal()?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ImageStats {
    name: &'static str,
    finite: usize,
    non_finite: usize,
    min: f64,
    max: f64,
    positive_mass: f64,
}

impl ImageStats {
    fn from(name: &'static str, image: ArrayView3<'_, f64>) -> Result<Self> {
        let mut stats = Self {
            name,
            finite: 0,
            non_finite: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            positive_mass: 0.0,
        };
        for value in image.iter().copied() {
            if value.is_finite() {
                stats.finite += 1;
                stats.min = stats.min.min(value);
                stats.max = stats.max.max(value);
                if value > 0.0 {
                    stats.positive_mass += value;
                }
            } else {
                stats.non_finite += 1;
            }
        }
        if stats.non_finite > 0 {
            return Err(Error::InvalidArgument(format!(
                "volume fit: {} contains {} non-finite voxel(s)",
                stats.name, stats.non_finite
            )));
        }
        Ok(stats)
    }

    fn require_signal(self) -> Result<()> {
        if self.finite == 0 {
            return Err(Error::InvalidArgument(format!(
                "volume fit: {} has no finite samples",
                self.name
            )));
        }
        if self.min == self.max {
            return Err(Error::InvalidArgument(format!(
                "volume fit: {} is flat at {}; no metric signal is available",
                self.name, self.min
            )));
        }
        if self.positive_mass <= 0.0 || !self.positive_mass.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "volume fit: {} has no positive finite mass for initialization",
                self.name
            )));
        }
        Ok(())
    }
}

pub fn resident(
    params: &VolumeFitParams,
    fixed: &Array3<f64>,
    moving: &Array3<f64>,
) -> Result<FittedTransform> {
    if let Some(schedule) = params.pyramid.as_ref() {
        return resident_pyramid(params, fixed, moving, schedule);
    }
    let shape = [fixed.shape()[0], fixed.shape()[1], fixed.shape()[2]];
    let whole = shape.iter().copied().max().unwrap_or(1);
    let block = if params.workers > 1 {
        32.min(whole)
    } else {
        whole
    };
    run_with_workers(
        params,
        &fixed.clone().into(),
        &moving.clone().into(),
        block,
        params.workers.max(1),
    )
}

pub fn resident_pyramid(
    params: &VolumeFitParams,
    fixed: &Array3<f64>,
    moving: &Array3<f64>,
    schedule: &PyramidSchedule,
) -> Result<FittedTransform> {
    let recipe = match params.pyramid_smoothing {
        Some(smoothing) => schedule.smoothed_recipe(smoothing)?,
        None => schedule.recipe()?,
    };
    resident_pyramid_with_recipe(params, fixed, moving, schedule, &recipe)
}

pub fn resident_pyramid_with_recipe(
    params: &VolumeFitParams,
    fixed: &Array3<f64>,
    moving: &Array3<f64>,
    schedule: &PyramidSchedule,
    recipe: &PyramidRecipe,
) -> Result<FittedTransform> {
    if recipe.len() != schedule.levels() {
        return Err(Error::InvalidArgument(format!(
            "volume pyramid schedule has {} level(s), but the image recipe builds {}",
            schedule.levels(),
            recipe.len()
        )));
    }
    let levels = pyramid_levels_from_recipe(params, fixed, moving, schedule, recipe)?;
    resident_pyramid_levels(params, &levels)
}

pub fn pyramid_levels_from_recipe(
    params: &VolumeFitParams,
    fixed: &Array3<f64>,
    moving: &Array3<f64>,
    schedule: &PyramidSchedule,
    recipe: &PyramidRecipe,
) -> Result<Vec<VolumePyramidLevel>> {
    if recipe.len() != schedule.levels() {
        return Err(Error::InvalidArgument(format!(
            "volume pyramid schedule has {} level(s), but the image recipe builds {}",
            schedule.levels(),
            recipe.len()
        )));
    }
    let fixed_levels = recipe.build_resident(&fixed.clone().into())?;
    let moving_levels = recipe.build_resident(&moving.clone().into())?;
    let mut levels = Vec::with_capacity(schedule.levels());
    for level in 0..schedule.levels() {
        let scale = cumulative_scale(schedule, level);
        levels.push(VolumePyramidLevel::new(
            fixed_levels[level].clone(),
            moving_levels[level].clone(),
            scale,
            geometry_for_scale(&params.geometry, scale)?,
        )?);
    }
    Ok(levels)
}

#[cfg(feature = "zarr")]
pub fn zarr_pyramid_levels_or_build(
    params: &VolumeFitParams,
    fixed: VolumePyramidStoreInput<'_>,
    moving: VolumePyramidStoreInput<'_>,
    schedule: &PyramidSchedule,
    work_root: impl Into<PathBuf>,
    chunk: [usize; 3],
) -> Result<Vec<VolumePyramidLevel>> {
    use crate::zarr_env::{MultiscaleImage, PyramidSpec, ZarrEnvironment};

    let work_root = work_root.into();
    let use_stored = fixed.stored.zip(moving.stored).filter(|(fixed, moving)| {
        stored_pyramid_matches_schedule(fixed, schedule)
            && stored_pyramid_matches_schedule(moving, schedule)
    });
    let built_fixed;
    let built_moving;
    let (fixed_pyramid, moving_pyramid): (&MultiscaleImage, &MultiscaleImage) =
        if let Some((fixed_pyramid, moving_pyramid)) = use_stored {
            (fixed_pyramid, moving_pyramid)
        } else {
            let spec = PyramidSpec::new(schedule.factors().to_vec())?.with_chunk(chunk);
            built_fixed =
                ZarrEnvironment::build_multiscale(&fixed.build_root, fixed.full_resolution, &spec)?;
            built_moving = ZarrEnvironment::build_multiscale(
                &moving.build_root,
                moving.full_resolution,
                &spec,
            )?;
            (&built_fixed, &built_moving)
        };

    zarr_pyramid_levels(params, fixed_pyramid, moving_pyramid, schedule, work_root)
}

#[cfg(feature = "zarr")]
pub fn zarr_pyramid_levels(
    params: &VolumeFitParams,
    fixed: &crate::zarr_env::MultiscaleImage,
    moving: &crate::zarr_env::MultiscaleImage,
    schedule: &PyramidSchedule,
    work_root: impl Into<PathBuf>,
) -> Result<Vec<VolumePyramidLevel>> {
    use crate::zarr_env::ZarrEnvironment;

    let work_root = work_root.into();
    let mut levels = Vec::with_capacity(schedule.levels());
    for level in 0..schedule.levels() {
        let expected_scale = cumulative_scale(schedule, level);
        let fixed_scale = fixed.scale().get(level).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "volume pyramid fixed stored level {level} is absent; requested {} level(s)",
                schedule.levels()
            ))
        })?;
        let moving_scale = moving.scale().get(level).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "volume pyramid moving stored level {level} is absent; requested {} level(s)",
                schedule.levels()
            ))
        })?;
        if *fixed_scale != expected_scale || *moving_scale != expected_scale {
            return Err(Error::InvalidArgument(format!(
                "volume pyramid stored level {level} scale mismatch: fixed {:?}, moving \
                 {:?}, expected {:?}",
                fixed_scale, moving_scale, expected_scale
            )));
        }
        let env = ZarrEnvironment::attach_multiscale_level(
            work_root.join(format!("level-{level}")),
            &[fixed, moving],
            level,
        )?;
        levels.push(VolumePyramidLevel::new(
            env.image(0)?,
            env.image(ImageId::supplied(0).index())?,
            expected_scale,
            geometry_for_scale(&params.geometry, expected_scale)?,
        )?);
    }
    validate_pyramid_levels(params, &levels)?;
    Ok(levels)
}

#[cfg(feature = "zarr")]
fn stored_pyramid_matches_schedule(
    image: &crate::zarr_env::MultiscaleImage,
    schedule: &PyramidSchedule,
) -> bool {
    if image.len() < schedule.levels() {
        return false;
    }
    (0..schedule.levels()).all(|level| image.scale()[level] == cumulative_scale(schedule, level))
}

pub fn resident_pyramid_levels(
    params: &VolumeFitParams,
    levels: &[VolumePyramidLevel],
) -> Result<FittedTransform> {
    let params = if let Some(level0) = levels.first() {
        params.resolved_for_shape(level0.fixed.shape())?
    } else {
        params.clone()
    };
    validate_pyramid_levels(&params, levels)?;
    let mut initial: Option<Vec<f64>> = None;
    for level in (0..levels.len()).rev() {
        let scale = levels[level].scale;
        let seeded = initial.as_ref().map(|params_at_full_resolution| {
            scale_params_down(params.model, params_at_full_resolution, scale)
        });
        let mut level_params = params.clone();
        level_params.geometry = levels[level].geometry.clone();
        // One block per level single-threaded (the reduction order, and so the
        // answer, of the original schedule); blocks of 32 across the workers
        // otherwise.
        let whole = levels[level]
            .fixed
            .shape()
            .iter()
            .copied()
            .max()
            .unwrap_or(1);
        let block = if params.workers > 1 {
            32.min(whole)
        } else {
            whole
        };
        let (next, state_at_level) = run_with_initial(
            &level_params,
            &levels[level].fixed,
            &levels[level].moving,
            block,
            params.workers.max(1),
            seeded,
        )?;
        initial = Some(scale_params_up(params.model, &state_at_level.params, scale));
        let _ = next;
    }
    let params_at_full_resolution = initial
        .ok_or_else(|| Error::InvalidArgument("volume pyramid has no level to run".to_string()))?;
    let shape = levels[0].fixed.shape();
    Ok(FittedTransform {
        model: params.model,
        params: params_at_full_resolution,
        shape,
        geometry: params.geometry.clone(),
        bspline: params.bspline,
        bspline_layout: None,
    })
}

fn validate_pyramid_levels(params: &VolumeFitParams, levels: &[VolumePyramidLevel]) -> Result<()> {
    validate_config(params)?;
    if levels.is_empty() {
        return Err(Error::InvalidArgument(
            "volume pyramid levels: at least one level is required".to_string(),
        ));
    }
    if levels[0].scale != [1, 1, 1] {
        return Err(Error::InvalidArgument(format!(
            "volume pyramid level 0 must have scale [1, 1, 1], got {:?}",
            levels[0].scale
        )));
    }
    let full_shape = levels[0].fixed.shape();
    let mut previous_scale = [1, 1, 1];
    for (index, level) in levels.iter().enumerate() {
        if level.fixed.shape() != level.moving.shape() {
            return Err(Error::InvalidArgument(format!(
                "volume pyramid level {index} has fixed shape {:?} and moving shape {:?}; \
                 both images must be on the same lattice",
                level.fixed.shape(),
                level.moving.shape()
            )));
        }
        if level.fixed.dtype() != Dtype::F64 || level.moving.dtype() != Dtype::F64 {
            return Err(Error::InvalidArgument(format!(
                "volume pyramid level {index} has fixed dtype {:?} and moving dtype {:?}; \
                 expected f64 volumes",
                level.fixed.dtype(),
                level.moving.dtype()
            )));
        }
        if level.scale.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "volume pyramid level {index} scale {:?} must be positive on every axis",
                level.scale
            )));
        }
        if index > 0 {
            let grows = (0..3).any(|axis| level.scale[axis] > previous_scale[axis]);
            let compatible = (0..3).all(|axis| {
                level.scale[axis] >= previous_scale[axis]
                    && level.scale[axis] % previous_scale[axis] == 0
            });
            if !grows || !compatible {
                return Err(Error::InvalidArgument(format!(
                    "volume pyramid level {index} scale {:?} is not a cumulative \
                     downsample of previous scale {:?}",
                    level.scale, previous_scale
                )));
            }
        }
        for axis in 0..3 {
            let expected = full_shape[axis] / level.scale[axis];
            if expected == 0 || expected != level.fixed.shape()[axis] {
                return Err(Error::InvalidArgument(format!(
                    "volume pyramid level {index} axis {axis} has shape {} at scale {}; \
                     expected {expected} from full-resolution shape {}",
                    level.fixed.shape()[axis],
                    level.scale[axis],
                    full_shape[axis]
                )));
            }
        }
        let expected_geometry = geometry_for_scale(&params.geometry, level.scale)?;
        validate_level_geometry(index, &level.geometry, &expected_geometry)?;
        previous_scale = level.scale;
    }
    Ok(())
}

fn geometry_for_scale(base: &SpatialFrame, scale: [usize; 3]) -> Result<SpatialFrame> {
    let spacing = base.spacing();
    SpatialFrame::new(
        base.origin(),
        [
            spacing[0] * scale[0] as f64,
            spacing[1] * scale[1] as f64,
            spacing[2] * scale[2] as f64,
        ],
        base.direction(),
    )
}

fn validate_level_geometry(
    index: usize,
    got: &SpatialFrame,
    expected: &SpatialFrame,
) -> Result<()> {
    for axis in 0..3 {
        same_number(
            got.origin()[axis],
            expected.origin()[axis],
            index,
            &format!("origin[{axis}]"),
        )?;
        same_number(
            got.spacing()[axis],
            expected.spacing()[axis],
            index,
            &format!("spacing[{axis}]"),
        )?;
        for column in 0..3 {
            same_number(
                got.direction()[axis][column],
                expected.direction()[axis][column],
                index,
                &format!("direction[{axis}][{column}]"),
            )?;
        }
    }
    Ok(())
}

fn same_number(got: f64, expected: f64, index: usize, field: &str) -> Result<()> {
    if (got - expected).abs() > 1.0e-9 || !got.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "volume pyramid level {index} metadata {field} is {got}, expected {expected}"
        )));
    }
    Ok(())
}

fn cumulative_scale(schedule: &PyramidSchedule, level: usize) -> [usize; 3] {
    let mut scale = [1, 1, 1];
    for factor in schedule.factors().iter().take(level) {
        for axis in 0..3 {
            scale[axis] *= factor[axis];
        }
    }
    scale
}

fn scale_params_down(model: TransformModel, params: &[f64], scale: [usize; 3]) -> Vec<f64> {
    scale_params(model, params, scale, false)
}

fn scale_params_up(model: TransformModel, params: &[f64], scale: [usize; 3]) -> Vec<f64> {
    scale_params(model, params, scale, true)
}

fn scale_params(model: TransformModel, params: &[f64], scale: [usize; 3], up: bool) -> Vec<f64> {
    let mut next = params.to_vec();
    match model {
        TransformModel::Translation => {
            for axis in 0..3 {
                let factor = scale[axis] as f64;
                next[axis] = if up {
                    params[axis] * factor
                } else {
                    params[axis] / factor
                };
            }
        }
        TransformModel::Affine => {
            for row in 0..3 {
                for col in 0..3 {
                    let index = row * 3 + col;
                    let row_scale = scale[row] as f64;
                    let col_scale = scale[col] as f64;
                    next[index] = if up {
                        params[index] * row_scale / col_scale
                    } else {
                        params[index] * col_scale / row_scale
                    };
                }
            }
            for axis in 0..3 {
                let factor = scale[axis] as f64;
                next[9 + axis] = if up {
                    params[9 + axis] * factor
                } else {
                    params[9 + axis] / factor
                };
            }
        }
        TransformModel::BSpline => {
            let per_block = params.len() / 3;
            for axis in 0..3 {
                let factor = scale[axis] as f64;
                for index in axis * per_block..(axis + 1) * per_block {
                    next[index] = if up {
                        params[index] * factor
                    } else {
                        params[index] / factor
                    };
                }
            }
        }
    }
    next
}

// ------------------------------------------------------ stochastic fitting --
//
// A second optimiser beside the block-reduced Gauss-Newton one above: the
// stochastic gradient descent of elastix (Klein et al., "Adaptive stochastic
// gradient descent optimisation for image registration", IJCV 2009), on Mattes
// mutual information with Parzen windows (Mattes et al. 2003; Thévenaz and
// Unser 2000), with a random-coordinate sampler that draws new samples every
// iteration.
//
// **Why resident and not planned.** One iteration evaluates a few thousand
// samples scattered over the whole fixed image, and the next iteration needs
// the parameters the previous one produced. A block decomposition would hand
// each block a handful of samples per iteration and pay a plan step for it;
// the moving image would also need a halo as large as the transform can move
// a point, which for an atlas fit is the whole image. So the images are held in
// memory (an atlas at 25 µm is tens of millions of voxels) and the samples of
// one iteration are split into a **fixed** number of chunks evaluated in
// parallel and reduced in chunk order. The answer depends on the seed and not
// on the worker count.
//
// **Coordinates.** Everything is in voxel indices of the arrays as given,
// `[a0, a1, a2]`. The moving image is read at `prior(T(pre(x)))` for a fixed
// voxel `x`: `pre` and `prior` are fixed affines and `T` is fitted. With
// `prior` the map between the two images' voxel frames, `T` is a transform in
// the fixed image's own frame, which is what an elastix file written in the
// fixed image's physical frame holds. A B-spline stage after an affine `A` uses
// `pre = A`: elastix's `Compose` applies the initial transform first, so its
// B-spline lives in the space the affine maps into, and its control grid
// covers the box the affine maps the fixed image onto.

/// How an image is read between voxel centres in a stochastic fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleOrder {
    /// Trilinear.
    Linear,
    /// Cubic B-spline on prefiltered coefficients (an interpolating spline,
    /// elastix's `BSplineInterpolator` of order 3), with mirror boundaries.
    Cubic,
}

/// The step gain `a / (t + A + 1)^alpha` of a stochastic gradient descent, as
/// elastix computes it (time `t` starts at 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainSequence {
    pub a: f64,
    pub big_a: f64,
    pub alpha: f64,
}

/// The adaptive time step of elastix's adaptive stochastic gradient descent:
/// `t_{k+1} = max(0, t_k + f(-g_k · g_{k-1}))` with
/// `f(x) = min + (max - min) / (1 + exp(-x / scale))`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SigmoidTime {
    pub max: f64,
    pub min: f64,
    pub scale: f64,
    pub initial_time: f64,
}

impl Default for SigmoidTime {
    fn default() -> Self {
        Self {
            max: 1.0,
            min: -0.8,
            scale: 1.0e-8,
            initial_time: 0.0,
        }
    }
}

/// elastix's automatic estimation of the gain (`AutomaticParameterEstimation`,
/// the "Original" method): the gain is chosen so that a step moves a voxel by
/// about `max_step` voxels, discounted by the measured noise of the stochastic
/// gradient, and the sigmoid is set from the same measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutomaticGain {
    /// elastix's `MaximumStepLength`, in fixed voxels.
    pub max_step: f64,
    /// elastix's `NumberOfGradientMeasurements`.
    pub gradient_measurements: usize,
    /// elastix's `NumberOfJacobianMeasurements`.
    pub jacobian_measurements: usize,
    /// elastix's `NumberOfSamplesForExactGradient`.
    pub exact_gradient_samples: usize,
    /// elastix's `SigmoidScaleFactor`.
    pub sigmoid_scale_factor: f64,
}

impl Default for AutomaticGain {
    fn default() -> Self {
        Self {
            max_step: 1.0,
            gradient_measurements: 3,
            jacobian_measurements: 1000,
            exact_gradient_samples: 100_000,
            sigmoid_scale_factor: 0.1,
        }
    }
}

/// How the step gain evolves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StochasticOptimizer {
    /// elastix's `StandardGradientDescent`: `t = k`.
    Standard(GainSequence),
    /// elastix's `AdaptiveStochasticGradientDescent` with a fixed gain.
    Adaptive {
        gain: GainSequence,
        time: SigmoidTime,
    },
    /// elastix's `AdaptiveStochasticGradientDescent` with
    /// `AutomaticParameterEstimation`: `a`, `alpha` and the sigmoid are
    /// estimated at the start of every level; `big_a` is kept.
    AdaptiveAutomatic {
        big_a: f64,
        estimation: AutomaticGain,
        initial_time: f64,
    },
}

/// How parameters are scaled before the gradient step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterScales {
    /// Every parameter steps by the gain times its gradient.
    Unit,
    /// elastix's `AutomaticScalesEstimation`: each parameter is divided by the
    /// mean squared norm of the transform's derivative with respect to it over
    /// a grid of about `samples` fixed voxels, so a matrix entry that multiplies
    /// coordinates of hundreds of voxels steps as a translation does.
    Automatic { samples: usize },
}

/// Where the fitted map starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StochasticInitialisation {
    /// Identity in the fitted map's frame (zero B-spline coefficients).
    Identity,
    /// elastix's `AutomaticTransformInitialization` with `GeometricalCenter`:
    /// the translation takes the fixed image's centre to the moving image's,
    /// and an affine rotates about the fixed centre.
    GeometricCentre,
}

/// Stop a level once the cost has stopped moving: the mean cost over the last
/// `window` iterations differs from the window before by no more than
/// `tolerance` times its magnitude. elastix's stochastic optimisers have no
/// such test and always run their iteration count; this is an option.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeTolerance {
    pub window: usize,
    pub tolerance: f64,
}

/// One resolution of a stochastic fit, coarse to fine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticLevel {
    /// Gaussian sigma applied to the fixed image, in fixed voxels. elastix's
    /// smoothing pyramid uses `0.5 * factor`, without downsampling.
    pub fixed_sigma: [f64; 3],
    /// Gaussian sigma applied to the moving image, in moving voxels.
    pub moving_sigma: [f64; 3],
    pub iterations: usize,
    /// B-spline control point spacing at this level, in fixed voxels.
    pub grid_spacing: [f64; 3],
}

/// Settings of [`fit_stochastic`]. [`Default`] is elastix's affine defaults
/// with mutual information: 4 levels of 500 iterations, 2000 samples, 32 bins.
#[derive(Debug, Clone, PartialEq)]
pub struct StochasticFitParams {
    pub model: TransformModel,
    /// Coarsest first.
    pub levels: Vec<StochasticLevel>,
    pub fixed_bins: usize,
    pub moving_bins: usize,
    /// B-spline order of the fixed Parzen window, 0 to 3.
    pub fixed_kernel_order: usize,
    /// B-spline order of the moving Parzen window, 1 to 3 (its derivative is
    /// used).
    pub moving_kernel_order: usize,
    /// Widen each histogram's intensity range by this fraction of the image's
    /// range at both ends (elastix's `FixedLimitRangeRatio`, default 0.01).
    pub fixed_limit_range_ratio: f64,
    pub moving_limit_range_ratio: f64,
    /// Spatial samples per iteration.
    pub samples: usize,
    /// Draw new samples every iteration (elastix's `NewSamplesEveryIteration`);
    /// `false` keeps one draw per level.
    pub new_samples_every_iteration: bool,
    /// Draw each iteration's samples inside one random box of this size, in
    /// fixed voxels (elastix's `UseRandomSampleRegion` and `SampleRegionSize`).
    pub sample_region: Option<[f64; 3]>,
    pub fixed_order: SampleOrder,
    pub moving_order: SampleOrder,
    pub optimizer: StochasticOptimizer,
    pub scales: ParameterScales,
    pub initialisation: StochasticInitialisation,
    pub seed: u64,
    pub relative_tolerance: Option<RelativeTolerance>,
    /// Fewer valid samples than this fraction of `samples` draws again
    /// (elastix's `CheckNumberOfSamples`, ratio 0.25).
    pub minimum_valid_fraction: f64,
    /// Draws per iteration before giving up (elastix's
    /// `MaximumNumberOfSamplingAttempts`).
    pub sampling_attempts: usize,
    /// Worker threads; 0 is the global pool.
    pub workers: usize,
}

impl Default for StochasticFitParams {
    fn default() -> Self {
        Self {
            model: TransformModel::Affine,
            levels: elastix_levels(4, 500, None),
            fixed_bins: 32,
            moving_bins: 32,
            fixed_kernel_order: 0,
            moving_kernel_order: 3,
            fixed_limit_range_ratio: 0.0,
            moving_limit_range_ratio: 0.0,
            samples: 2000,
            new_samples_every_iteration: true,
            sample_region: None,
            fixed_order: SampleOrder::Linear,
            moving_order: SampleOrder::Linear,
            optimizer: StochasticOptimizer::AdaptiveAutomatic {
                big_a: 20.0,
                estimation: AutomaticGain::default(),
                initial_time: 0.0,
            },
            scales: ParameterScales::Automatic { samples: 10_000 },
            initialisation: StochasticInitialisation::GeometricCentre,
            seed: 121_212,
            relative_tolerance: None,
            minimum_valid_fraction: 0.25,
            sampling_attempts: 10,
            workers: 0,
        }
    }
}

/// elastix's default schedule for `resolutions` levels: smoothing sigma
/// `0.5 * 2^(n-1-l)` voxels on both images, and for a B-spline a control point
/// spacing of `final_spacing * 2^(n-1-l)`.
pub fn elastix_levels(
    resolutions: usize,
    iterations: usize,
    final_spacing: Option<[f64; 3]>,
) -> Vec<StochasticLevel> {
    (0..resolutions)
        .map(|level| {
            let factor = (1u64 << (resolutions - 1 - level)) as f64;
            StochasticLevel {
                fixed_sigma: [0.5 * factor; 3],
                moving_sigma: [0.5 * factor; 3],
                iterations,
                grid_spacing: final_spacing
                    .map(|spacing| spacing.map(|axis| axis * factor))
                    .unwrap_or([0.0; 3]),
            }
        })
        .collect()
}

/// An affine map of voxel coordinates, `y = matrix · x + offset`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineMap {
    pub matrix: [[f64; 3]; 3],
    pub offset: [f64; 3],
}

impl AffineMap {
    pub fn identity() -> Self {
        Self {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            offset: [0.0; 3],
        }
    }

    /// From the twelve parameters of a [`TransformModel::Affine`]
    /// [`FittedTransform`]: the matrix row by row, then the offset.
    pub fn from_params(params: &[f64]) -> Result<Self> {
        if params.len() != 12 {
            return Err(Error::InvalidArgument(format!(
                "affine map: expected 12 parameters, got {}",
                params.len()
            )));
        }
        Ok(Self {
            matrix: std::array::from_fn(|r| std::array::from_fn(|c| params[r * 3 + c])),
            offset: [params[9], params[10], params[11]],
        })
    }

    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|r| {
            self.matrix[r][0] * point[0]
                + self.matrix[r][1] * point[1]
                + self.matrix[r][2] * point[2]
                + self.offset[r]
        })
    }

    /// `self ∘ inner`: `inner` first.
    pub fn then_after(&self, inner: &Self) -> Self {
        let matrix = std::array::from_fn(|r| {
            std::array::from_fn(|c| (0..3).map(|k| self.matrix[r][k] * inner.matrix[k][c]).sum())
        });
        let offset = self.apply(inner.offset);
        Self { matrix, offset }
    }

    pub fn inverse(&self) -> Result<Self> {
        let matrix = invert_3x3(self.matrix)?;
        let moved: [f64; 3] =
            std::array::from_fn(|r| (0..3).map(|k| matrix[r][k] * self.offset[k]).sum::<f64>());
        Ok(Self {
            matrix,
            offset: moved.map(|v| -v),
        })
    }
}

/// An explicit B-spline control grid in fixed voxels: node `i` of axis `a` is
/// at `origin[a] + i * spacing[a]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BSplineLayout {
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
}

/// What one level of a stochastic fit did.
#[derive(Debug, Clone, PartialEq)]
pub struct StochasticLevelReport {
    pub iterations: usize,
    /// Mean cost (negative mutual information) over the first and the last
    /// tenth of the iterations.
    pub initial_cost: f64,
    pub final_cost: f64,
    /// The gain sequence used (estimated, with automatic estimation).
    pub gain: GainSequence,
    pub stopped_by_tolerance: bool,
    pub seconds: f64,
    /// B-spline control grid size at this level.
    pub grid: Option<[usize; 3]>,
}

/// The result of [`fit_stochastic`].
#[derive(Debug, Clone, PartialEq)]
pub struct StochasticFit {
    /// The fitted map `T`, in fixed voxels, in [`FittedTransform`]'s
    /// conventions: an affine as `y = A x + t`, a B-spline as coefficient blocks
    /// on [`FittedTransform::bspline_layout`].
    pub transform: FittedTransform,
    pub levels: Vec<StochasticLevelReport>,
    pub scales: Vec<f64>,
    pub seconds: f64,
}

/// Fit `T` so that `moving(prior(T(pre(x))))` matches `fixed(x)` by Mattes
/// mutual information, with elastix's stochastic optimisers.
///
/// `initial` replaces the initialisation with explicit parameters in the
/// internal layout: translation `[t]`, affine `[A row-major, t]` about the
/// centre `c = pre(fixed centre)` (`T(z) = A (z - c) + c + t`), or B-spline
/// coefficients on the first level's grid.
pub fn fit_stochastic(
    params: &StochasticFitParams,
    fixed: ArrayView3<'_, f32>,
    moving: ArrayView3<'_, f32>,
    pre: &AffineMap,
    prior: &AffineMap,
    initial: Option<Vec<f64>>,
    geometry: SpatialFrame,
) -> Result<StochasticFit> {
    validate_stochastic(params, fixed.shape(), moving.shape())?;
    let run = || fit_stochastic_inner(params, fixed, moving, pre, prior, initial, geometry);
    if params.workers > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(params.workers)
            .build()
            .map_err(|error| Error::InvalidArgument(format!("stochastic fit: {error}")))?
            .install(run)
    } else {
        run()
    }
}

fn validate_stochastic(
    params: &StochasticFitParams,
    fixed: &[usize],
    moving: &[usize],
) -> Result<()> {
    let fail = |what: String| Err(Error::InvalidArgument(format!("stochastic fit: {what}")));
    if fixed.iter().chain(moving).any(|n| *n < 4) {
        return fail(format!(
            "images of shape {fixed:?} and {moving:?} are too small; every axis needs four voxels"
        ));
    }
    if params.levels.is_empty() {
        return fail("at least one level is needed".to_string());
    }
    if params.samples < 16 {
        return fail(format!(
            "{} samples per iteration is too few",
            params.samples
        ));
    }
    if params.fixed_bins < 4 || params.moving_bins < 4 {
        return fail(format!(
            "histograms need at least 4 bins, got {} and {}",
            params.fixed_bins, params.moving_bins
        ));
    }
    if params.fixed_kernel_order > 3 || !(1..=3).contains(&params.moving_kernel_order) {
        return fail(format!(
            "Parzen kernel orders must be 0..=3 (fixed) and 1..=3 (moving), got {} and {}",
            params.fixed_kernel_order, params.moving_kernel_order
        ));
    }
    if params.fixed_bins <= 2 * (params.fixed_kernel_order / 2) + 1
        || params.moving_bins <= 2 * (params.moving_kernel_order / 2) + 1
    {
        return fail("too few bins for the Parzen kernel padding".to_string());
    }
    for (index, level) in params.levels.iter().enumerate() {
        if level.iterations == 0 {
            return fail(format!("level {index} has no iterations"));
        }
        if level
            .fixed_sigma
            .iter()
            .chain(&level.moving_sigma)
            .any(|s| *s < 0.0 || !s.is_finite())
        {
            return fail(format!("level {index} has a negative or non-finite sigma"));
        }
        if params.model == TransformModel::BSpline
            && level
                .grid_spacing
                .iter()
                .any(|s| !(*s > 0.0) || !s.is_finite())
        {
            return fail(format!(
                "level {index} needs a positive B-spline grid spacing, got {:?}",
                level.grid_spacing
            ));
        }
    }
    if let Some(region) = params.sample_region {
        if region.iter().any(|s| !(*s > 0.0) || !s.is_finite()) {
            return fail(format!("sample region {region:?} must be positive"));
        }
    }
    if [
        params.fixed_limit_range_ratio,
        params.moving_limit_range_ratio,
    ]
    .iter()
    .any(|r| !(*r >= 0.0) || !r.is_finite())
    {
        return fail("limit range ratios must be finite and non-negative".to_string());
    }
    if !(0.0..=1.0).contains(&params.minimum_valid_fraction) || params.sampling_attempts == 0 {
        return fail(
            "the valid-sample fraction must be in [0, 1] with at least one attempt".into(),
        );
    }
    match params.optimizer {
        StochasticOptimizer::Standard(gain) | StochasticOptimizer::Adaptive { gain, .. } => {
            if !(gain.a > 0.0) || gain.big_a < 0.0 || gain.alpha < 0.0 {
                return fail(format!("gain {gain:?} must have a > 0, A >= 0, alpha >= 0"));
            }
        }
        StochasticOptimizer::AdaptiveAutomatic { estimation, .. } => {
            if params.model == TransformModel::BSpline {
                return fail(
                    "automatic gain estimation is implemented for translation and affine \
                     transforms; give a B-spline stage an explicit gain (elastix SP_a, SP_A, \
                     SP_alpha) or the standard optimiser"
                        .to_string(),
                );
            }
            if !(estimation.max_step > 0.0) || estimation.gradient_measurements == 0 {
                return fail(format!(
                    "automatic gain estimation {estimation:?} is degenerate"
                ));
            }
        }
    }
    if let Some(stop) = params.relative_tolerance {
        if stop.window == 0 || !(stop.tolerance >= 0.0) {
            return fail(format!(
                "relative tolerance {stop:?} needs a window and a tolerance"
            ));
        }
    }
    Ok(())
}

/// Fixed number of sample chunks per iteration: the reduction order, and so
/// the answer, does not depend on the number of threads.
const STOCHASTIC_CHUNKS: usize = 16;

struct StochasticModel {
    kind: TransformModel,
    /// Centre of rotation of an affine.
    centre: [f64; 3],
    grid: [usize; 3],
    layout: BSplineLayout,
}

impl StochasticModel {
    fn parameter_count(&self) -> usize {
        match self.kind {
            TransformModel::Translation => 3,
            TransformModel::Affine => 12,
            TransformModel::BSpline => 3 * self.grid[0] * self.grid[1] * self.grid[2],
        }
    }

    fn per_axis(&self) -> usize {
        self.grid[0] * self.grid[1] * self.grid[2]
    }

    fn apply(&self, params: &[f64], x: [f64; 3]) -> [f64; 3] {
        match self.kind {
            TransformModel::Translation => std::array::from_fn(|r| x[r] + params[r]),
            TransformModel::Affine => {
                let d = [
                    x[0] - self.centre[0],
                    x[1] - self.centre[1],
                    x[2] - self.centre[2],
                ];
                std::array::from_fn(|r| {
                    params[3 * r] * d[0]
                        + params[3 * r + 1] * d[1]
                        + params[3 * r + 2] * d[2]
                        + self.centre[r]
                        + params[9 + r]
                })
            }
            TransformModel::BSpline => match self.support(x) {
                Some(support) => {
                    let per = self.per_axis();
                    let mut out = x;
                    support.for_each(|flat, weight| {
                        for axis in 0..3 {
                            out[axis] += weight * params[axis * per + flat];
                        }
                    });
                    out
                }
                None => x,
            },
        }
    }

    /// The cubic support of a point on the control grid, or `None` outside the
    /// region the deformation is defined on (`[1, size - 2)` in grid index, as
    /// elastix and ITK).
    fn support(&self, x: [f64; 3]) -> Option<GridSupport> {
        let mut start = [0usize; 3];
        let mut weights = [[0.0; 4]; 3];
        for axis in 0..3 {
            let u = (x[axis] - self.layout.origin[axis]) / self.layout.spacing[axis];
            if !(u >= 1.0 && u < self.grid[axis] as f64 - 2.0) {
                return None;
            }
            let base = u.floor() as usize - 1;
            start[axis] = base;
            for k in 0..4 {
                weights[axis][k] = cubic_basis(u - (base + k) as f64);
            }
        }
        Some(GridSupport {
            start,
            weights,
            grid: self.grid,
        })
    }
}

struct GridSupport {
    start: [usize; 3],
    weights: [[f64; 4]; 3],
    grid: [usize; 3],
}

impl GridSupport {
    fn for_each(&self, mut visit: impl FnMut(usize, f64)) {
        for k in 0..4 {
            let wk = self.weights[2][k];
            let gk = self.start[2] + k;
            for j in 0..4 {
                let wjk = wk * self.weights[1][j];
                let row = self.grid[0] * (self.start[1] + j + self.grid[1] * gk);
                for i in 0..4 {
                    visit(row + self.start[0] + i, wjk * self.weights[0][i]);
                }
            }
        }
    }
}

/// An image as a sampler reads it: raw values (linear) or interpolating cubic
/// coefficients, row-major `[a0, a1, a2]`.
struct SampledImage {
    data: Vec<f32>,
    shape: [usize; 3],
    order: SampleOrder,
}

impl SampledImage {
    fn new(image: Array3<f32>, order: SampleOrder) -> Self {
        let shape = [image.shape()[0], image.shape()[1], image.shape()[2]];
        let image = match order {
            SampleOrder::Linear => image,
            SampleOrder::Cubic => cubic_coefficients(image),
        };
        let data = if image.is_standard_layout() {
            image.into_raw_vec_and_offset().0
        } else {
            image.iter().copied().collect()
        };
        Self { data, shape, order }
    }

    /// Inside the buffer as ITK means it: continuous index in `[-0.5, n - 0.5)`.
    fn inside(&self, p: [f64; 3]) -> bool {
        (0..3).all(|a| p[a] >= -0.5 && p[a] < self.shape[a] as f64 - 0.5)
    }

    #[inline]
    fn at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.data[(i * self.shape[1] + j) * self.shape[2] + k] as f64
    }

    fn value(&self, p: [f64; 3]) -> f64 {
        match self.order {
            SampleOrder::Linear => self.linear(p, false).0,
            SampleOrder::Cubic => self.cubic(p, false).0,
        }
    }

    fn value_and_gradient(&self, p: [f64; 3]) -> (f64, [f64; 3]) {
        match self.order {
            SampleOrder::Linear => self.linear(p, true),
            SampleOrder::Cubic => self.cubic(p, true),
        }
    }

    fn linear(&self, p: [f64; 3], gradient: bool) -> (f64, [f64; 3]) {
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        let mut t = [0.0; 3];
        for a in 0..3 {
            let n = self.shape[a];
            let q = p[a].clamp(0.0, (n - 1) as f64);
            let base = (q.floor() as usize).min(n - 2);
            lo[a] = base;
            hi[a] = base + 1;
            t[a] = q - base as f64;
        }
        let c = |i: usize, j: usize, k: usize| self.at(i, j, k);
        let c000 = c(lo[0], lo[1], lo[2]);
        let c001 = c(lo[0], lo[1], hi[2]);
        let c010 = c(lo[0], hi[1], lo[2]);
        let c011 = c(lo[0], hi[1], hi[2]);
        let c100 = c(hi[0], lo[1], lo[2]);
        let c101 = c(hi[0], lo[1], hi[2]);
        let c110 = c(hi[0], hi[1], lo[2]);
        let c111 = c(hi[0], hi[1], hi[2]);
        let c00 = lerp(c000, c001, t[2]);
        let c01 = lerp(c010, c011, t[2]);
        let c10 = lerp(c100, c101, t[2]);
        let c11 = lerp(c110, c111, t[2]);
        let c0 = lerp(c00, c01, t[1]);
        let c1 = lerp(c10, c11, t[1]);
        let value = lerp(c0, c1, t[0]);
        if !gradient {
            return (value, [0.0; 3]);
        }
        let g0 = c1 - c0;
        let g1 = lerp(c01 - c00, c11 - c10, t[0]);
        let g2 = lerp(
            lerp(c001 - c000, c011 - c010, t[1]),
            lerp(c101 - c100, c111 - c110, t[1]),
            t[0],
        );
        (value, [g0, g1, g2])
    }

    fn cubic(&self, p: [f64; 3], gradient: bool) -> (f64, [f64; 3]) {
        let mut index = [[0usize; 4]; 3];
        let mut w = [[0.0; 4]; 3];
        let mut d = [[0.0; 4]; 3];
        for a in 0..3 {
            let base = p[a].floor() as isize - 1;
            for k in 0..4 {
                let node = base + k as isize;
                let t = p[a] - node as f64;
                index[a][k] = mirror_index(node, self.shape[a]);
                w[a][k] = cubic_basis(t);
                d[a][k] = cubic_basis_derivative(t);
            }
        }
        let mut value = 0.0;
        let mut grad = [0.0; 3];
        for i in 0..4 {
            let mut v_i = 0.0;
            let mut g1_i = 0.0;
            let mut g2_i = 0.0;
            for j in 0..4 {
                let row = (index[0][i] * self.shape[1] + index[1][j]) * self.shape[2];
                let mut v_ij = 0.0;
                let mut g2_ij = 0.0;
                for k in 0..4 {
                    let c = self.data[row + index[2][k]] as f64;
                    v_ij += w[2][k] * c;
                    g2_ij += d[2][k] * c;
                }
                v_i += w[1][j] * v_ij;
                g1_i += d[1][j] * v_ij;
                g2_i += w[1][j] * g2_ij;
            }
            value += w[0][i] * v_i;
            if gradient {
                grad[0] += d[0][i] * v_i;
                grad[1] += w[0][i] * g1_i;
                grad[2] += w[0][i] * g2_i;
            }
        }
        (value, grad)
    }
}

/// ITK's mirror boundary: `-1 → 1`, `n → n - 2`, with period `2(n - 1)`.
fn mirror_index(index: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut i = index.rem_euclid(period);
    if i >= n as isize {
        i = period - i;
    }
    i as usize
}

fn cubic_basis_derivative(t: f64) -> f64 {
    let a = t.abs();
    if a < 1.0 {
        -2.0 * t + 1.5 * t * a
    } else if a < 2.0 {
        let u = 2.0 - a;
        -0.5 * t.signum() * u * u
    } else {
        0.0
    }
}

/// The centred B-spline of order `n` (0 to 3) at `x`.
fn bspline_kernel(order: usize, x: f64) -> f64 {
    let a = x.abs();
    match order {
        0 => {
            if a < 0.5 {
                1.0
            } else if a == 0.5 {
                0.5
            } else {
                0.0
            }
        }
        1 => (1.0 - a).max(0.0),
        2 => {
            if a < 0.5 {
                0.75 - a * a
            } else if a < 1.5 {
                0.5 * (1.5 - a) * (1.5 - a)
            } else {
                0.0
            }
        }
        _ => cubic_basis(x),
    }
}

/// Its derivative, `β_n'(x) = β_{n-1}(x + 1/2) - β_{n-1}(x - 1/2)`.
fn bspline_kernel_derivative(order: usize, x: f64) -> f64 {
    bspline_kernel(order - 1, x + 0.5) - bspline_kernel(order - 1, x - 0.5)
}

/// One image's Parzen window: elastix's bin layout, with `order / 2` padding
/// bins at each end and a small margin so that the extreme values fall inside.
#[derive(Debug, Clone, Copy)]
struct Parzen {
    bins: usize,
    order: usize,
    min: f64,
    max: f64,
    bin_size: f64,
    normalised_min: f64,
}

impl Parzen {
    fn new(bins: usize, order: usize, min: f64, max: f64) -> Self {
        let padding = (order / 2) as f64;
        let intervals = bins as f64 - 2.0 * padding - 1.0;
        let range = (max - min).max(f64::MIN_POSITIVE);
        let small = 0.001 * range / intervals;
        let bin_size = (range + 2.0 * small) / intervals;
        let normalised_min = (min - small) / bin_size - padding;
        Self {
            bins,
            order,
            min,
            max,
            bin_size,
            normalised_min,
        }
    }

    /// The value limited to the image's range, and whether it was limited.
    fn term(&self, value: f64) -> (f64, bool) {
        let limited = value.clamp(self.min, self.max);
        (
            limited / self.bin_size - self.normalised_min,
            limited != value,
        )
    }

    /// The first bin of the window and the window's weights.
    fn window(&self, term: f64) -> (usize, [f64; 4]) {
        let start = (term + 0.5 - self.order as f64 / 2.0).floor() as isize;
        let mut weights = [0.0; 4];
        for (k, weight) in weights.iter_mut().enumerate().take(self.order + 1) {
            *weight = bspline_kernel(self.order, (start + k as isize) as f64 - term);
        }
        (start.max(0) as usize, weights)
    }

    fn window_derivative(&self, term: f64) -> (usize, [f64; 4]) {
        let start = (term + 0.5 - self.order as f64 / 2.0).floor() as isize;
        let mut weights = [0.0; 4];
        for (k, weight) in weights.iter_mut().enumerate().take(self.order + 1) {
            *weight = bspline_kernel_derivative(self.order, (start + k as isize) as f64 - term);
        }
        (start.max(0) as usize, weights)
    }
}

/// One level's images and everything a sample needs.
struct StochasticLevelContext<'a> {
    params: &'a StochasticFitParams,
    fixed: SampledImage,
    moving: SampledImage,
    fixed_parzen: Parzen,
    moving_parzen: Parzen,
    model: StochasticModel,
    pre: AffineMap,
    prior: AffineMap,
    /// `priorᵀ`, which carries the moving image gradient back to `T`'s output.
    prior_transpose: [[f64; 3]; 3],
    level: usize,
}

/// One sample that mapped inside the moving image.
#[derive(Clone, Copy)]
struct Sample {
    /// `pre(x)`: where `T` and its Jacobian are evaluated.
    x: [f64; 3],
    fixed_term: f64,
    moving_term: f64,
    /// `priorᵀ ∇M`, zero where the moving value was limited.
    gradient: [f64; 3],
}

/// Where the samples of one evaluation come from.
#[derive(Clone, Copy)]
enum SampleSource {
    /// Uniform random coordinates; `key` seeds the draw.
    Random { key: u64, count: usize },
    /// A regular grid of about `count` points over the fixed image.
    Grid { count: usize },
}

struct Evaluation {
    cost: f64,
    gradient: Vec<f64>,
    valid: usize,
}

/// SplitMix64: a small, well-mixed, seedable generator.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u = self.uniform().max(f64::MIN_POSITIVE);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn mix_key(parts: &[u64]) -> u64 {
    let mut state = SplitMix(0x005E_ED0F_A11C_0DE5);
    let mut key = 0u64;
    for part in parts {
        state.0 ^= part.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        key = state.next_u64();
    }
    key
}

/// Points of a regular grid with about `count` points over `shape`, spaced
/// `floor((voxels / count)^(1/3))` voxels and centred, as elastix's grid
/// sampler places them.
fn grid_points(shape: [usize; 3], count: usize) -> Vec<[f64; 3]> {
    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let step = ((voxels / count.max(1) as f64).cbrt().floor() as usize).max(1);
    let axis = |n: usize| -> Vec<f64> {
        let points = (n - 1) / step + 1;
        let offset = ((n - 1) - (points - 1) * step) / 2;
        (0..points).map(|i| (offset + i * step) as f64).collect()
    };
    let (a, b, c) = (axis(shape[0]), axis(shape[1]), axis(shape[2]));
    let mut out = Vec::with_capacity(a.len() * b.len() * c.len());
    for &x in &a {
        for &y in &b {
            for &z in &c {
                out.push([x, y, z]);
            }
        }
    }
    out
}

impl StochasticLevelContext<'_> {
    fn fixed_shape(&self) -> [usize; 3] {
        self.fixed.shape
    }

    /// The sample box of one draw: the whole fixed image, or one random region.
    fn sample_box(&self, rng: &mut SplitMix) -> ([f64; 3], [f64; 3]) {
        let shape = self.fixed_shape();
        let mut lo = [0.0; 3];
        let mut hi: [f64; 3] = std::array::from_fn(|a| (shape[a] - 1) as f64);
        if let Some(region) = self.params.sample_region {
            for a in 0..3 {
                let size = region[a].min(hi[a]);
                let corner = rng.uniform() * (hi[a] - size);
                lo[a] = corner;
                hi[a] = corner + size;
            }
        }
        (lo, hi)
    }

    fn sample_at(&self, params: &[f64], x: [f64; 3]) -> Option<Sample> {
        let (fixed_term, _) = self.fixed_parzen.term(self.fixed.value(x));
        let z = self.pre.apply(x);
        let y = self.prior.apply(self.model.apply(params, z));
        if !self.moving.inside(y) {
            return None;
        }
        let (value, gradient) = self.moving.value_and_gradient(y);
        let (moving_term, limited) = self.moving_parzen.term(value);
        let gradient = if limited {
            [0.0; 3]
        } else {
            std::array::from_fn(|r| {
                (0..3)
                    .map(|k| self.prior_transpose[r][k] * gradient[k])
                    .sum::<f64>()
            })
        };
        Some(Sample {
            x: z,
            fixed_term,
            moving_term,
            gradient,
        })
    }

    /// The samples of one evaluation, in `STOCHASTIC_CHUNKS` chunks, and the
    /// node box their B-spline support can touch (the whole grid when the
    /// samples cover the whole image).
    fn draw(&self, params: &[f64], source: SampleSource) -> (Vec<Vec<Sample>>, NodeBox) {
        match source {
            SampleSource::Random { key, count } => {
                let mut region_rng = SplitMix(mix_key(&[key, u64::MAX]));
                let (lo, hi) = self.sample_box(&mut region_rng);
                let chunks = (0..STOCHASTIC_CHUNKS)
                    .into_par_iter()
                    .map(|chunk| {
                        let wanted = count / STOCHASTIC_CHUNKS
                            + usize::from(chunk < count % STOCHASTIC_CHUNKS);
                        let mut rng = SplitMix(mix_key(&[key, chunk as u64]));
                        let mut out = Vec::with_capacity(wanted);
                        for _ in 0..wanted {
                            let x: [f64; 3] =
                                std::array::from_fn(|a| lo[a] + rng.uniform() * (hi[a] - lo[a]));
                            if let Some(sample) = self.sample_at(params, x) {
                                out.push(sample);
                            }
                        }
                        out
                    })
                    .collect();
                (chunks, self.node_box(lo, hi))
            }
            SampleSource::Grid { count } => {
                let shape = self.fixed_shape();
                let points = grid_points(shape, count);
                let per = points.len().div_ceil(STOCHASTIC_CHUNKS).max(1);
                let chunks = points
                    .par_chunks(per)
                    .map(|chunk| {
                        chunk
                            .iter()
                            .filter_map(|x| self.sample_at(params, *x))
                            .collect()
                    })
                    .collect();
                let hi = shape.map(|n| (n - 1) as f64);
                (chunks, self.node_box([0.0; 3], hi))
            }
        }
    }

    fn node_box(&self, lo: [f64; 3], hi: [f64; 3]) -> NodeBox {
        if self.model.kind != TransformModel::BSpline {
            return NodeBox {
                start: [0; 3],
                size: [0; 3],
            };
        }
        let (lo, hi) = mapped_box(&self.pre, lo, hi);
        let mut start = [0usize; 3];
        let mut size = [0usize; 3];
        for a in 0..3 {
            let u = |p: f64| (p - self.model.layout.origin[a]) / self.model.layout.spacing[a];
            let first = (u(lo[a]).floor() as isize - 1).clamp(0, self.model.grid[a] as isize - 1);
            let last = (u(hi[a]).floor() as isize + 2).clamp(0, self.model.grid[a] as isize - 1);
            start[a] = first as usize;
            size[a] = (last - first + 1) as usize;
        }
        NodeBox { start, size }
    }

    /// Negative Mattes mutual information over `source`, and its gradient
    /// with respect to `params` when asked.
    fn evaluate(&self, params: &[f64], source: SampleSource, gradient: bool) -> Result<Evaluation> {
        let (chunks, nodes) = self.draw(params, source);
        let fixed_bins = self.fixed_parzen.bins;
        let moving_bins = self.moving_parzen.bins;
        let valid: usize = chunks.iter().map(Vec::len).sum();
        if valid == 0 {
            return Ok(Evaluation {
                cost: 0.0,
                gradient: Vec::new(),
                valid: 0,
            });
        }
        // The joint histogram, chunk by chunk, summed in chunk order.
        let partials: Vec<Vec<f64>> = chunks
            .par_iter()
            .map(|samples| {
                let mut joint = vec![0.0; fixed_bins * moving_bins];
                for sample in samples {
                    self.add_to_histogram(&mut joint, sample);
                }
                joint
            })
            .collect();
        let mut joint = vec![0.0; fixed_bins * moving_bins];
        for partial in &partials {
            for (slot, value) in joint.iter_mut().zip(partial) {
                *slot += value;
            }
        }
        let total: f64 = joint.iter().sum();
        let mut fixed_marginal = vec![0.0; fixed_bins];
        let mut moving_marginal = vec![0.0; moving_bins];
        for f in 0..fixed_bins {
            for m in 0..moving_bins {
                let p = joint[f * moving_bins + m] / total;
                joint[f * moving_bins + m] = p;
                fixed_marginal[f] += p;
                moving_marginal[m] += p;
            }
        }
        let mut information = 0.0;
        // `ln(p(f, m) / p_M(m))`, the factor the derivative needs.
        let mut ratio = vec![0.0; fixed_bins * moving_bins];
        for f in 0..fixed_bins {
            for m in 0..moving_bins {
                let p = joint[f * moving_bins + m];
                if p > 0.0 {
                    information += p * (p / (fixed_marginal[f] * moving_marginal[m])).ln();
                    ratio[f * moving_bins + m] = (p / moving_marginal[m]).ln();
                }
            }
        }
        if !gradient {
            return Ok(Evaluation {
                cost: -information,
                gradient: Vec::new(),
                valid,
            });
        }
        let count = self.model.parameter_count();
        let scale = 1.0 / (total * self.moving_parzen.bin_size);
        let boxed = self.model.kind == TransformModel::BSpline;
        let local_len = if boxed { 3 * nodes.volume() } else { count };
        let partials: Vec<Vec<f64>> = chunks
            .par_iter()
            .map(|samples| {
                let mut local = vec![0.0; local_len];
                for sample in samples {
                    let factor = scale * self.derivative_weight(&ratio, sample);
                    if factor == 0.0 {
                        continue;
                    }
                    let g = sample.gradient.map(|v| v * factor);
                    self.add_jacobian(&mut local, &nodes, sample.x, g);
                }
                local
            })
            .collect();
        let mut out = vec![0.0; count];
        for partial in &partials {
            if boxed {
                nodes.scatter_add(&self.model, partial, &mut out);
            } else {
                for (slot, value) in out.iter_mut().zip(partial) {
                    *slot += value;
                }
            }
        }
        Ok(Evaluation {
            cost: -information,
            gradient: out,
            valid,
        })
    }

    fn add_to_histogram(&self, joint: &mut [f64], sample: &Sample) {
        let moving_bins = self.moving_parzen.bins;
        let (f0, fw) = self.fixed_parzen.window(sample.fixed_term);
        let (m0, mw) = self.moving_parzen.window(sample.moving_term);
        for (a, wf) in fw.iter().enumerate().take(self.fixed_parzen.order + 1) {
            let f = f0 + a;
            if *wf == 0.0 || f >= self.fixed_parzen.bins {
                continue;
            }
            for (b, wm) in mw.iter().enumerate().take(self.moving_parzen.order + 1) {
                let m = m0 + b;
                if m < moving_bins {
                    joint[f * moving_bins + m] += wf * wm;
                }
            }
        }
    }

    /// `Σ_f Σ_m β_F(f - ft) β_M'(m - mt) ln(p(f, m) / p_M(m))`.
    fn derivative_weight(&self, ratio: &[f64], sample: &Sample) -> f64 {
        let moving_bins = self.moving_parzen.bins;
        let (f0, fw) = self.fixed_parzen.window(sample.fixed_term);
        let (m0, mw) = self.moving_parzen.window_derivative(sample.moving_term);
        let mut sum = 0.0;
        for (a, wf) in fw.iter().enumerate().take(self.fixed_parzen.order + 1) {
            let f = f0 + a;
            if *wf == 0.0 || f >= self.fixed_parzen.bins {
                continue;
            }
            for (b, wm) in mw.iter().enumerate().take(self.moving_parzen.order + 1) {
                let m = m0 + b;
                if m < moving_bins {
                    sum += wf * wm * ratio[f * moving_bins + m];
                }
            }
        }
        sum
    }

    /// `gradient += Jᵀ g` at `x`, for the model's Jacobian `J = ∂T/∂μ`.
    fn add_jacobian(&self, gradient: &mut [f64], nodes: &NodeBox, x: [f64; 3], g: [f64; 3]) {
        let model = &self.model;
        match model.kind {
            TransformModel::Translation => {
                for r in 0..3 {
                    gradient[r] += g[r];
                }
            }
            TransformModel::Affine => {
                let d = [
                    x[0] - model.centre[0],
                    x[1] - model.centre[1],
                    x[2] - model.centre[2],
                ];
                for r in 0..3 {
                    for c in 0..3 {
                        gradient[3 * r + c] += g[r] * d[c];
                    }
                    gradient[9 + r] += g[r];
                }
            }
            TransformModel::BSpline => {
                if let Some(support) = model.support(x) {
                    let per = nodes.volume();
                    for k in 0..4 {
                        let gk = support.start[2] + k - nodes.start[2];
                        for j in 0..4 {
                            let gj = support.start[1] + j - nodes.start[1];
                            let wjk = support.weights[2][k] * support.weights[1][j];
                            let row = nodes.size[0] * (gj + nodes.size[1] * gk);
                            for i in 0..4 {
                                let gi = support.start[0] + i - nodes.start[0];
                                let w = wjk * support.weights[0][i];
                                let flat = row + gi;
                                gradient[flat] += w * g[0];
                                gradient[per + flat] += w * g[1];
                                gradient[2 * per + flat] += w * g[2];
                            }
                        }
                    }
                }
            }
        }
    }

    /// The model's Jacobian at fixed voxel `x` (evaluated at `pre(x)`), dense
    /// `3 × P`, for the small models the automatic estimation runs on.
    fn dense_jacobian(&self, x: [f64; 3]) -> Vec<[f64; 3]> {
        let x = self.pre.apply(x);
        let count = self.model.parameter_count();
        let mut columns = vec![[0.0; 3]; count];
        for r in 0..3 {
            let mut g = [0.0; 3];
            g[r] = 1.0;
            let mut row = vec![0.0; count];
            self.add_jacobian(
                &mut row,
                &NodeBox {
                    start: [0; 3],
                    size: [0; 3],
                },
                x,
                g,
            );
            for (p, value) in row.into_iter().enumerate() {
                columns[p][r] = value;
            }
        }
        columns
    }
}

/// The sub-box of B-spline nodes one draw's samples can touch.
struct NodeBox {
    start: [usize; 3],
    size: [usize; 3],
}

impl NodeBox {
    fn volume(&self) -> usize {
        self.size[0] * self.size[1] * self.size[2]
    }

    fn scatter_add(&self, model: &StochasticModel, local: &[f64], out: &mut [f64]) {
        let per = model.per_axis();
        let local_per = self.volume();
        for axis in 0..3 {
            for k in 0..self.size[2] {
                for j in 0..self.size[1] {
                    for i in 0..self.size[0] {
                        let l = i + self.size[0] * (j + self.size[1] * k);
                        let g = (self.start[0] + i)
                            + model.grid[0]
                                * ((self.start[1] + j) + model.grid[1] * (self.start[2] + k));
                        out[axis * per + g] += local[axis * local_per + l];
                    }
                }
            }
        }
    }
}

fn fit_stochastic_inner(
    params: &StochasticFitParams,
    fixed: ArrayView3<'_, f32>,
    moving: ArrayView3<'_, f32>,
    pre: &AffineMap,
    prior: &AffineMap,
    initial: Option<Vec<f64>>,
    geometry: SpatialFrame,
) -> Result<StochasticFit> {
    let started = std::time::Instant::now();
    let fixed_shape = [fixed.shape()[0], fixed.shape()[1], fixed.shape()[2]];
    let moving_shape = [moving.shape()[0], moving.shape()[1], moving.shape()[2]];
    let centre = pre.apply(fixed_shape.map(|n| (n - 1) as f64 / 2.0));
    let prior_inverse = prior.inverse()?;
    let prior_transpose = std::array::from_fn(|r| std::array::from_fn(|c| prior.matrix[c][r]));

    // The starting parameters, in the internal layout.
    let mut current: Vec<f64> = match (params.model, initial) {
        (_, Some(seed)) => seed,
        (TransformModel::BSpline, None) => Vec::new(),
        (model, None) => {
            let translation = match params.initialisation {
                StochasticInitialisation::Identity => [0.0; 3],
                StochasticInitialisation::GeometricCentre => {
                    let moving_centre =
                        prior_inverse.apply(moving_shape.map(|n| (n - 1) as f64 / 2.0));
                    std::array::from_fn(|a| moving_centre[a] - centre[a])
                }
            };
            match model {
                TransformModel::Translation => translation.to_vec(),
                _ => {
                    let mut values = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
                    values.extend(translation);
                    values
                }
            }
        }
    };

    let mut reports = Vec::with_capacity(params.levels.len());
    let mut scales = Vec::new();
    let mut previous_grid: Option<(StochasticModel, Vec<f64>)> = None;
    let mut model = StochasticModel {
        kind: params.model,
        centre,
        grid: [0; 3],
        layout: BSplineLayout {
            origin: [0.0; 3],
            spacing: [1.0; 3],
        },
    };
    for (level_index, level) in params.levels.iter().enumerate() {
        let level_started = std::time::Instant::now();
        // The level's images: smoothed, then (for cubic reads) prefiltered.
        let (fixed_image, moving_image) = rayon::join(
            || gaussian_smooth(fixed.to_owned(), level.fixed_sigma),
            || gaussian_smooth(moving.to_owned(), level.moving_sigma),
        );
        let widen = |(lo, hi): (f64, f64), ratio: f64| {
            let range = hi - lo;
            (lo - ratio * range, hi + ratio * range)
        };
        let (fixed_min, fixed_max) = widen(extrema(&fixed_image), params.fixed_limit_range_ratio);
        let (moving_min, moving_max) =
            widen(extrema(&moving_image), params.moving_limit_range_ratio);
        if !(fixed_max > fixed_min) || !(moving_max > moving_min) {
            return Err(Error::InvalidArgument(format!(
                "stochastic fit: level {level_index} has a flat image (fixed {fixed_min}..{fixed_max}, \
                 moving {moving_min}..{moving_max}); mutual information has no signal"
            )));
        }
        let (fixed_sampled, moving_sampled) = rayon::join(
            || SampledImage::new(fixed_image, params.fixed_order),
            || SampledImage::new(moving_image, params.moving_order),
        );
        if params.model == TransformModel::BSpline {
            let (grid, layout) = bspline_grid_after(fixed_shape, level.grid_spacing, pre);
            let next = StochasticModel {
                kind: TransformModel::BSpline,
                centre,
                grid,
                layout,
            };
            current = match previous_grid.take() {
                Some((coarse, coefficients)) => upsample_bspline(&coarse, &coefficients, &next),
                None if current.len() == next.parameter_count() => current,
                None => vec![0.0; next.parameter_count()],
            };
            model = next;
        }
        let context = StochasticLevelContext {
            params,
            fixed: fixed_sampled,
            moving: moving_sampled,
            fixed_parzen: Parzen::new(
                params.fixed_bins,
                params.fixed_kernel_order,
                fixed_min,
                fixed_max,
            ),
            moving_parzen: Parzen::new(
                params.moving_bins,
                params.moving_kernel_order,
                moving_min,
                moving_max,
            ),
            model,
            pre: *pre,
            prior: *prior,
            prior_transpose,
            level: level_index,
        };
        if current.len() != context.model.parameter_count() {
            return Err(Error::InvalidArgument(format!(
                "stochastic fit: {} initial parameter(s) for a model with {}",
                current.len(),
                context.model.parameter_count()
            )));
        }
        scales = match params.scales {
            ParameterScales::Unit => vec![1.0; current.len()],
            ParameterScales::Automatic { samples } => {
                if params.model == TransformModel::BSpline {
                    vec![1.0; current.len()]
                } else {
                    estimate_scales(&context, samples)
                }
            }
        };
        let (gain, time) = match params.optimizer {
            StochasticOptimizer::Standard(gain) => (gain, None),
            StochasticOptimizer::Adaptive { gain, time } => (gain, Some(time)),
            StochasticOptimizer::AdaptiveAutomatic {
                big_a,
                estimation,
                initial_time,
            } => {
                let (gain, time) = estimate_gain(
                    &context,
                    &current,
                    &scales,
                    big_a,
                    &estimation,
                    initial_time,
                )?;
                (gain, Some(time))
            }
        };
        let report = run_stochastic_level(
            &context,
            &mut current,
            &scales,
            gain,
            time,
            level,
            level_index,
        )?;
        reports.push(StochasticLevelReport {
            seconds: level_started.elapsed().as_secs_f64(),
            grid: (params.model == TransformModel::BSpline).then_some(context.model.grid),
            ..report
        });
        model = context.model;
        if params.model == TransformModel::BSpline {
            previous_grid = Some((
                StochasticModel {
                    kind: TransformModel::BSpline,
                    centre,
                    grid: model.grid,
                    layout: model.layout,
                },
                current.clone(),
            ));
        }
    }

    let transform = match params.model {
        TransformModel::Translation => FittedTransform {
            model: TransformModel::Translation,
            params: current,
            shape: fixed_shape,
            geometry,
            bspline: ControlGrid::default(),
            bspline_layout: None,
        },
        TransformModel::Affine => {
            // `A (x - c) + c + t = A x + (c + t - A c)`.
            let mut values = current[..9].to_vec();
            for r in 0..3 {
                let moved: f64 = (0..3).map(|k| current[3 * r + k] * centre[k]).sum();
                values.push(centre[r] + current[9 + r] - moved);
            }
            FittedTransform {
                model: TransformModel::Affine,
                params: values,
                shape: fixed_shape,
                geometry,
                bspline: ControlGrid::default(),
                bspline_layout: None,
            }
        }
        TransformModel::BSpline => FittedTransform {
            model: TransformModel::BSpline,
            params: current,
            shape: fixed_shape,
            geometry,
            bspline: ControlGrid::new(model.grid)?,
            bspline_layout: Some(model.layout),
        },
    };
    Ok(StochasticFit {
        transform,
        levels: reports,
        scales,
        seconds: started.elapsed().as_secs_f64(),
    })
}

fn run_stochastic_level(
    context: &StochasticLevelContext<'_>,
    current: &mut [f64],
    scales: &[f64],
    gain: GainSequence,
    sigmoid: Option<SigmoidTime>,
    level: &StochasticLevel,
    level_index: usize,
) -> Result<StochasticLevelReport> {
    let params = context.params;
    let mut time = sigmoid.map(|s| s.initial_time).unwrap_or(0.0);
    let mut previous: Option<Vec<f64>> = None;
    let mut costs = Vec::with_capacity(level.iterations);
    let mut stopped = false;
    let required = (params.minimum_valid_fraction * params.samples as f64).ceil() as usize;
    let root_scales: Vec<f64> = scales.iter().map(|s| s.sqrt()).collect();
    for iteration in 0..level.iterations {
        let mut evaluation = None;
        for attempt in 0..params.sampling_attempts {
            let draw = if params.new_samples_every_iteration {
                iteration as u64
            } else {
                0
            };
            let key = mix_key(&[params.seed, level_index as u64, draw, attempt as u64]);
            let result = context.evaluate(
                current,
                SampleSource::Random {
                    key,
                    count: params.samples,
                },
                true,
            )?;
            if result.valid >= required.max(1) {
                evaluation = Some(result);
                break;
            }
        }
        let evaluation = evaluation.ok_or_else(|| {
            Error::InvalidArgument(format!(
                "stochastic fit: at level {level_index}, iteration {iteration}, fewer than {required} \
                 of {} samples map inside the moving image in {} attempts; the images do not overlap \
                 under the current transform",
                params.samples, params.sampling_attempts
            ))
        })?;
        // The step in scaled parameters `y = μ ⊙ √s`: `g_y = g / √s`, and
        // `y -= a_k g_y` is `μ -= a_k g / s`.
        let scaled: Vec<f64> = evaluation
            .gradient
            .iter()
            .zip(&root_scales)
            .map(|(g, r)| g / r)
            .collect();
        // elastix's `Compute_a`: `a / (A + t + 1)^alpha`.
        let step = gain.a / (time + gain.big_a + 1.0).powf(gain.alpha);
        for ((value, g), r) in current.iter_mut().zip(&scaled).zip(&root_scales) {
            *value -= step * g / r;
        }
        match sigmoid {
            Some(sigmoid) => {
                if let Some(previous) = &previous {
                    let inner: f64 = previous.iter().zip(&scaled).map(|(a, b)| a * b).sum();
                    let f = sigmoid.min
                        + (sigmoid.max - sigmoid.min) / (1.0 + (inner / sigmoid.scale).exp());
                    time = (time + f).max(0.0);
                }
                previous = Some(scaled);
            }
            None => time += 1.0,
        }
        costs.push(evaluation.cost);
        if let Some(stop) = params.relative_tolerance {
            let n = costs.len();
            if n >= 2 * stop.window && n % stop.window == 0 {
                let mean = |range: std::ops::Range<usize>| {
                    costs[range.clone()].iter().sum::<f64>() / range.len() as f64
                };
                let last = mean(n - stop.window..n);
                let before = mean(n - 2 * stop.window..n - stop.window);
                if (last - before).abs() <= stop.tolerance * before.abs() {
                    stopped = true;
                    break;
                }
            }
        }
    }
    let tenth = (costs.len() / 10).max(1);
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    Ok(StochasticLevelReport {
        iterations: costs.len(),
        initial_cost: mean(&costs[..tenth]),
        final_cost: mean(&costs[costs.len() - tenth..]),
        gain,
        stopped_by_tolerance: stopped,
        seconds: 0.0,
        grid: None,
    })
}

/// elastix's `AutomaticScalesEstimation`: `s_p = mean_x ||∂T/∂μ_p (x)||²` over
/// a grid of about `samples` fixed voxels.
fn estimate_scales(context: &StochasticLevelContext<'_>, samples: usize) -> Vec<f64> {
    let points = grid_points(context.fixed_shape(), samples);
    let count = context.model.parameter_count();
    let mut scales = vec![0.0; count];
    for x in &points {
        for (p, column) in context.dense_jacobian(*x).iter().enumerate() {
            scales[p] += column.iter().map(|v| v * v).sum::<f64>();
        }
    }
    scales
        .into_iter()
        .map(|s| (s / points.len() as f64).max(f64::MIN_POSITIVE))
        .collect()
}

/// elastix's automatic parameter estimation for adaptive stochastic gradient
/// descent, the "Original" method of Klein et al. (2009).
///
/// With `J_j` the model Jacobian at sample `j` with columns divided by `√s`,
/// `C = mean_j J_jᵀ J_j`: the gain is `a = A δ / (σ₁ √max_j JCJ_j)` times the
/// noise factor `σ₁² / (σ₁² + σ₃²)`, where `σ₁² = E||g||² / tr C` is measured
/// from exact gradients and `σ₃² = E||g - ĝ||² / tr C` from their difference to
/// stochastic ones, at points perturbed by `δ / √max_j JJ_j`; `alpha` is 1,
/// and the sigmoid runs from `-0.99 + 0.98 · noise` to 1 with scale
/// `factor · σ₃² · √tr(C²)`.
fn estimate_gain(
    context: &StochasticLevelContext<'_>,
    current: &[f64],
    scales: &[f64],
    big_a: f64,
    estimation: &AutomaticGain,
    initial_time: f64,
) -> Result<(GainSequence, SigmoidTime)> {
    let count = context.model.parameter_count();
    let root: Vec<f64> = scales.iter().map(|s| s.sqrt()).collect();
    let points = grid_points(context.fixed_shape(), estimation.jacobian_measurements);
    let jacobians: Vec<Vec<[f64; 3]>> = points
        .iter()
        .map(|x| {
            let mut columns = context.dense_jacobian(*x);
            for (column, r) in columns.iter_mut().zip(&root) {
                *column = column.map(|v| v / r);
            }
            columns
        })
        .collect();
    let mut covariance = vec![0.0; count * count];
    for columns in &jacobians {
        for p in 0..count {
            for q in 0..count {
                covariance[p * count + q] +=
                    (0..3).map(|r| columns[p][r] * columns[q][r]).sum::<f64>();
            }
        }
    }
    for value in &mut covariance {
        *value /= jacobians.len() as f64;
    }
    let trace_c: f64 = (0..count).map(|p| covariance[p * count + p]).sum();
    let trace_cc: f64 = covariance.iter().map(|v| v * v).sum();
    let root_two_two = 2.0 * std::f64::consts::SQRT_2;
    let mut max_jj = 0.0f64;
    let mut max_jcj = 0.0f64;
    for columns in &jacobians {
        // `J Jᵀ` and `J C Jᵀ`, both 3 × 3.
        let mut jj = [[0.0; 3]; 3];
        let mut jcj = [[0.0; 3]; 3];
        let mut jc = vec![[0.0; 3]; count];
        for q in 0..count {
            for p in 0..count {
                let c = covariance[p * count + q];
                if c != 0.0 {
                    for r in 0..3 {
                        jc[q][r] += columns[p][r] * c;
                    }
                }
            }
        }
        for r in 0..3 {
            for s in 0..3 {
                jj[r][s] = (0..count).map(|p| columns[p][r] * columns[p][s]).sum();
                jcj[r][s] = (0..count).map(|q| jc[q][r] * columns[q][s]).sum();
            }
        }
        let frobenius = |m: &[[f64; 3]; 3]| m.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
        let trace = |m: &[[f64; 3]; 3]| m[0][0] + m[1][1] + m[2][2];
        max_jj = max_jj.max(trace(&jj) + root_two_two * frobenius(&jj));
        max_jcj = max_jcj.max(trace(&jcj) + root_two_two * frobenius(&jcj));
    }
    let delta = estimation.max_step;
    let sigma4 = if max_jj > 1e-14 {
        delta / max_jj.sqrt()
    } else {
        0.0
    };
    let mut rng = SplitMix(mix_key(&[
        context.params.seed,
        context.level as u64,
        0xA11C,
    ]));
    let mut exact_gg = 0.0;
    let mut diff_gg = 0.0;
    for measurement in 0..estimation.gradient_measurements {
        // Perturb in scaled parameters, `μ = (y + σ₄ ξ) / √s`.
        let perturbed: Vec<f64> = current
            .iter()
            .zip(&root)
            .map(|(value, r)| value + sigma4 * rng.normal() / r)
            .collect();
        let exact = context.evaluate(
            &perturbed,
            SampleSource::Grid {
                count: estimation.exact_gradient_samples,
            },
            true,
        )?;
        let key = mix_key(&[
            context.params.seed,
            context.level as u64,
            0xE57,
            measurement as u64,
        ]);
        let approximate = context.evaluate(
            &perturbed,
            SampleSource::Random {
                key,
                count: context.params.samples,
            },
            true,
        )?;
        if exact.valid == 0 || approximate.valid == 0 {
            return Err(Error::InvalidArgument(format!(
                "stochastic fit: level {}: no samples map inside the moving image while estimating \
                 the gain",
                context.level
            )));
        }
        for p in 0..count {
            let e = exact.gradient[p] / root[p];
            let a = approximate.gradient[p] / root[p];
            exact_gg += e * e;
            diff_gg += (e - a) * (e - a);
        }
    }
    exact_gg /= estimation.gradient_measurements as f64;
    diff_gg /= estimation.gradient_measurements as f64;
    let sigma1 = if exact_gg > 1e-14 && trace_c > 1e-14 {
        (exact_gg / trace_c).sqrt()
    } else {
        0.0
    };
    let sigma3 = if diff_gg > 1e-14 && trace_c > 1e-14 {
        (diff_gg / trace_c).sqrt()
    } else {
        0.0
    };
    let a_max = if sigma1 > 1e-14 && max_jcj > 1e-14 {
        big_a * delta / sigma1 / max_jcj.sqrt()
    } else {
        0.0
    };
    let noise = sigma1 * sigma1 / (sigma1 * sigma1 + sigma3 * sigma3 + 1e-14);
    let a = a_max * noise;
    if !(a > 0.0) || !a.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "stochastic fit: level {}: the automatic gain estimate is {a} (σ₁ {sigma1}, σ₃ \
             {sigma3}); the metric has no gradient at the starting point",
            context.level
        )));
    }
    let omega = (estimation.sigmoid_scale_factor * sigma3 * sigma3 * trace_cc.sqrt()).max(1e-14);
    Ok((
        GainSequence {
            a,
            big_a,
            alpha: 1.0,
        },
        SigmoidTime {
            max: 1.0,
            min: -0.99 + 0.98 * noise,
            scale: omega,
            initial_time,
        },
    ))
}

/// [`bspline_grid_after`] with no initial transform.
pub fn bspline_grid_for_spacing(
    shape: [usize; 3],
    spacing: [f64; 3],
) -> ([usize; 3], BSplineLayout) {
    bspline_grid_after(shape, spacing, &AffineMap::identity())
}

/// The bounding box of the box `[lo, hi]` mapped by `map`.
fn mapped_box(map: &AffineMap, lo: [f64; 3], hi: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let mut out_lo = [f64::INFINITY; 3];
    let mut out_hi = [f64::NEG_INFINITY; 3];
    for corner in 0..8 {
        let p: [f64; 3] = std::array::from_fn(|a| if corner >> a & 1 == 0 { lo[a] } else { hi[a] });
        let q = map.apply(p);
        for a in 0..3 {
            out_lo[a] = out_lo[a].min(q[a]);
            out_hi[a] = out_hi[a].max(q[a]);
        }
    }
    (out_lo, out_hi)
}

/// elastix's B-spline grid for one level (`GridScheduleComputer`) over a
/// fixed image of `shape` unit voxels: `ceil(n / spacing)` intervals centred
/// on the image's `n - 1` voxel span, plus the order, so the first node is one
/// spacing before the first interval. (Measured: a 285 × 478 × 248 image at
/// spacing 25 gets elastix's `GridSize 15 23 13`.)
///
/// With an initial transform `pre` the grid covers the bounding box of the
/// fixed image mapped by `pre`, stretched per axis so that it still has the
/// same number of intervals: the image's voxel spacing becomes
/// `(hi - lo) / (n - 1)` and the grid spacing `spacing` of those. (Measured on
/// a real affine: elastix's `GridSpacing` and `GridOrigin` to 1e-6.)
pub fn bspline_grid_after(
    shape: [usize; 3],
    spacing: [f64; 3],
    pre: &AffineMap,
) -> ([usize; 3], BSplineLayout) {
    let (lo, hi) = mapped_box(pre, [0.0; 3], shape.map(|n| (n - 1) as f64));
    let mut size = [0usize; 3];
    let mut origin = [0.0; 3];
    let mut grid_spacing = [0.0; 3];
    for a in 0..3 {
        let n = shape[a] as f64;
        let voxel = if n > 1.0 {
            (hi[a] - lo[a]) / (n - 1.0)
        } else {
            1.0
        };
        let step = spacing[a] * voxel;
        let bare = (n / spacing[a]).ceil().max(1.0);
        size[a] = bare as usize + 3;
        grid_spacing[a] = step;
        origin[a] = lo[a] - (bare * step - (n - 1.0) * voxel) / 2.0 - step;
    }
    (
        size,
        BSplineLayout {
            origin,
            spacing: grid_spacing,
        },
    )
}

/// Coefficients on `fine`'s grid that reproduce `coarse`'s deformation:
/// the coarse spline evaluated at the fine nodes (mirror boundary, as ITK's
/// resample of a coefficient image), then decomposed into cubic coefficients.
fn upsample_bspline(
    coarse: &StochasticModel,
    coefficients: &[f64],
    fine: &StochasticModel,
) -> Vec<f64> {
    let per_coarse = coarse.per_axis();
    let mut out = Vec::with_capacity(fine.parameter_count());
    for axis in 0..3 {
        let block = &coefficients[axis * per_coarse..(axis + 1) * per_coarse];
        let values =
            Array3::from_shape_fn((fine.grid[2], fine.grid[1], fine.grid[0]), |(k, j, i)| {
                let p = [
                    fine.layout.origin[0] + i as f64 * fine.layout.spacing[0],
                    fine.layout.origin[1] + j as f64 * fine.layout.spacing[1],
                    fine.layout.origin[2] + k as f64 * fine.layout.spacing[2],
                ];
                let mut sum = 0.0;
                let mut index = [[0usize; 4]; 3];
                let mut w = [[0.0; 4]; 3];
                for a in 0..3 {
                    let u = (p[a] - coarse.layout.origin[a]) / coarse.layout.spacing[a];
                    let base = u.floor() as isize - 1;
                    for t in 0..4 {
                        let node = base + t as isize;
                        index[a][t] = mirror_index(node, coarse.grid[a]);
                        w[a][t] = cubic_basis(u - node as f64);
                    }
                }
                for c in 0..4 {
                    for b in 0..4 {
                        for a in 0..4 {
                            let flat = index[0][a]
                                + coarse.grid[0] * (index[1][b] + coarse.grid[1] * index[2][c]);
                            sum += w[0][a] * w[1][b] * w[2][c] * block[flat];
                        }
                    }
                }
                sum as f32
            });
        // Stored `[k, j, i]` so that the flat index `i + nx (j + ny k)` is the
        // row-major order of this array.
        let decomposed = cubic_coefficients(values);
        out.extend(decomposed.iter().map(|v| *v as f64));
    }
    out
}

fn extrema(image: &Array3<f32>) -> (f64, f64) {
    let fold = |chunk: &[f32]| {
        chunk
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(*v as f64), hi.max(*v as f64))
            })
    };
    match image.as_slice() {
        Some(values) => values.par_chunks(1 << 16).map(fold).reduce(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(a, b), (c, d)| (a.min(c), b.max(d)),
        ),
        None => fold(&image.iter().copied().collect::<Vec<_>>()),
    }
}

/// Apply `filter` to every line of `image` along `axis`, lines in parallel.
fn filter_lines(image: &mut Array3<f32>, axis: usize, filter: &(dyn Fn(&mut [f32]) + Sync)) {
    let outer = if axis == 0 { 1 } else { 0 };
    let views: Vec<_> = image.axis_chunks_iter_mut(Axis(outer), 1).collect();
    views.into_par_iter().for_each(|mut view| {
        let mut buffer = Vec::new();
        for mut lane in view.lanes_mut(Axis(axis)) {
            buffer.clear();
            buffer.extend(lane.iter().copied());
            filter(&mut buffer);
            for (slot, value) in lane.iter_mut().zip(&buffer) {
                *slot = *value;
            }
        }
    });
}

/// Separable Gaussian smoothing, sigma in voxels per axis, kernel truncated
/// at four sigma, edge values repeated beyond the image.
fn gaussian_smooth(mut image: Array3<f32>, sigma: [f64; 3]) -> Array3<f32> {
    for axis in 0..3 {
        if sigma[axis] <= 0.0 {
            continue;
        }
        let radius = (4.0 * sigma[axis]).ceil() as isize;
        let mut kernel: Vec<f64> = (-radius..=radius)
            .map(|k| (-0.5 * (k as f64 / sigma[axis]).powi(2)).exp())
            .collect();
        let total: f64 = kernel.iter().sum();
        kernel.iter_mut().for_each(|w| *w /= total);
        filter_lines(&mut image, axis, &|line: &mut [f32]| {
            let n = line.len() as isize;
            let source: Vec<f32> = line.to_vec();
            for (i, slot) in line.iter_mut().enumerate() {
                let mut sum = 0.0;
                for (k, w) in kernel.iter().enumerate() {
                    let j = (i as isize + k as isize - radius).clamp(0, n - 1);
                    sum += w * source[j as usize] as f64;
                }
                *slot = sum as f32;
            }
        });
    }
    image
}

/// Interpolating cubic B-spline coefficients (Unser's recursive filter, pole
/// `√3 - 2`, mirror boundaries), every axis.
fn cubic_coefficients(mut image: Array3<f32>) -> Array3<f32> {
    for axis in 0..3 {
        if image.shape()[axis] > 1 {
            filter_lines(&mut image, axis, &cubic_prefilter_line);
        }
    }
    image
}

fn cubic_prefilter_line(line: &mut [f32]) {
    let n = line.len();
    if n < 2 {
        return;
    }
    let z = 3.0f64.sqrt() - 2.0;
    let lambda = (1.0 - z) * (1.0 - 1.0 / z);
    let mut c: Vec<f64> = line.iter().map(|v| *v as f64 * lambda).collect();
    // Causal initialisation on the mirrored signal, truncated where `z^k` is
    // below `1e-10`.
    let horizon = ((1.0e-10f64).ln() / z.abs().ln()).ceil() as usize;
    let first = if horizon < n {
        let mut sum = c[0];
        let mut zn = z;
        for value in c.iter().take(horizon).skip(1) {
            sum += zn * value;
            zn *= z;
        }
        sum
    } else {
        let mut zn = z;
        let iz = 1.0 / z;
        let mut z2n = z.powi(n as i32 - 1);
        let mut sum = c[0] + z2n * c[n - 1];
        z2n = z2n * z2n * iz;
        for value in c.iter().take(n - 1).skip(1) {
            sum += (zn + z2n) * value;
            zn *= z;
            z2n *= iz;
        }
        sum / (1.0 - zn * zn)
    };
    c[0] = first;
    for k in 1..n {
        c[k] += z * c[k - 1];
    }
    c[n - 1] = (z / (z * z - 1.0)) * (c[n - 1] + z * c[n - 2]);
    for k in (0..n - 1).rev() {
        c[k] = z * (c[k + 1] - c[k]);
    }
    for (slot, value) in line.iter_mut().zip(&c) {
        *slot = *value as f32;
    }
}

// ------------------------------------------------------------------ tests --

/// The numerical kernels underneath the fitter, against derivatives and
/// solutions computed here.
///
/// The linear solve, the trilinear sampler, the interpolation gradient, the
/// parameter Jacobian, the spline basis and the packed symmetric index are all
/// private, so this is where they can be reached. Everything the integration file
/// `tests/align.rs` asserts rests on them being right, and it can only see them
/// through a whole optimisation: a Jacobian with two columns swapped still
/// converges on a symmetric fixture, and a gradient scaled by a constant still
/// converges, more slowly. Neither survives a finite difference.
#[cfg(test)]
mod tests {
    use super::*;

    /// A volume that is a function of its shape alone, from a 64-bit LCG with
    /// Knuth's MMIX constants. Values in `[0.5, 1.5)`: bounded away from zero so
    /// that a relative error is a meaningful thing to quote, and of order one so
    /// that an absolute finite-difference bound and a relative one agree.
    fn lcg_volume(shape: (usize, usize, usize)) -> Array3<f64> {
        let mut state: u64 = 0x2026_0830;
        Array3::from_shape_fn(shape, |_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            0.5 + (state >> 40) as f64 / 16_777_216.0
        })
    }

    /// The step a central difference is taken with, everywhere below.
    ///
    /// **Not a compromise between truncation and rounding, because there is no
    /// truncation to trade against.** The trilinear interpolant is multilinear
    /// inside one cell, so along any one coordinate it is a *straight line* and
    /// a central difference of it is exact in exact arithmetic. What is left is
    /// cancellation: `eps * |f| / h`, which at `|f| ~ 1` and `h = 1e-6` is
    /// `2e-10`; measured over every fixture below the worst gap is `3.1e-10`.
    /// Every bound below is `1e-8`, thirty times that, and every fixture keeps
    /// its sample at least `1e-3` from a cell boundary — and off a cell face
    /// entirely — so that the two evaluations stay on the same linear piece.
    const H: f64 = 1.0e-6;

    // ------------------------------------------------------- stochastic --

    /// A smooth, asymmetric scene: a few Gaussian blobs of different sizes and
    /// heights, so that mutual information has one clear optimum.
    fn blob_scene(shape: [usize; 3], shift: [f64; 3]) -> Array3<f32> {
        blob_scene_in(shape, shape, shift)
    }

    /// The scene laid out for `scene` voxels, sampled on an array of `shape`:
    /// four large blobs and twenty small ones at pseudo-random places.
    fn blob_scene_in(shape: [usize; 3], scene: [usize; 3], shift: [f64; 3]) -> Array3<f32> {
        let mut blobs = vec![
            ([0.35, 0.40, 0.45], 0.18, 1.0),
            ([0.65, 0.55, 0.35], 0.12, 0.6),
            ([0.45, 0.70, 0.65], 0.09, 0.8),
            ([0.60, 0.30, 0.70], 0.07, 0.4),
        ];
        let mut state: u64 = 0x5EED_2026;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..20 {
            let centre = [
                0.15 + 0.7 * next(),
                0.15 + 0.7 * next(),
                0.15 + 0.7 * next(),
            ];
            blobs.push((centre, 0.03 + 0.03 * next(), 0.2 + 0.6 * next()));
        }
        Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
            let p = [
                i as f64 - shift[0],
                j as f64 - shift[1],
                k as f64 - shift[2],
            ];
            let mut value = 0.05;
            for (centre, width, height) in &blobs {
                let mut d2 = 0.0;
                for a in 0..3 {
                    let n = scene[a] as f64;
                    let d = (p[a] - centre[a] * n) / (width * n);
                    d2 += d * d;
                }
                value += height * (-0.5 * d2).exp();
            }
            value as f32
        })
    }

    fn level_context<'a>(
        params: &'a StochasticFitParams,
        fixed: &Array3<f32>,
        moving: &Array3<f32>,
        model: StochasticModel,
    ) -> StochasticLevelContext<'a> {
        let (fmin, fmax) = extrema(fixed);
        let (mmin, mmax) = extrema(moving);
        StochasticLevelContext {
            params,
            fixed: SampledImage::new(fixed.clone(), params.fixed_order),
            moving: SampledImage::new(moving.clone(), params.moving_order),
            fixed_parzen: Parzen::new(params.fixed_bins, params.fixed_kernel_order, fmin, fmax),
            moving_parzen: Parzen::new(params.moving_bins, params.moving_kernel_order, mmin, mmax),
            model,
            pre: AffineMap::identity(),
            prior: AffineMap::identity(),
            prior_transpose: AffineMap::identity().matrix,
            level: 0,
        }
    }

    /// **The Mattes gradient is the derivative of the Mattes metric**, for an
    /// affine and for a B-spline, on a fixed set of samples.
    ///
    /// Cubic moving reads and cubic Parzen windows make the metric smooth in
    /// the parameters, so a central difference converges on the analytic
    /// gradient; a sign error, a missing `1 / bin size` or a dropped `priorᵀ`
    /// is off by a factor, not by a rounding error.
    #[test]
    fn the_mattes_gradient_is_the_derivative_of_the_metric() {
        // The moving image is larger and offset by `prior`, so every fixed
        // sample stays well inside it under the probed parameters and a
        // central difference sees the same sample set on both sides.
        let shape = [20, 22, 24];
        let fixed = blob_scene(shape, [0.0; 3]);
        let moving = blob_scene_in([30, 32, 34], shape, [6.3, 4.3, 5.9]);
        let prior = AffineMap {
            matrix: AffineMap::identity().matrix,
            offset: [5.0; 3],
        };
        let params = StochasticFitParams {
            fixed_kernel_order: 3,
            moving_kernel_order: 3,
            fixed_order: SampleOrder::Cubic,
            moving_order: SampleOrder::Cubic,
            fixed_bins: 16,
            moving_bins: 16,
            ..StochasticFitParams::default()
        };
        let source = SampleSource::Grid { count: 3000 };
        let centre = shape.map(|n| (n - 1) as f64 / 2.0);
        let affine = StochasticModel {
            kind: TransformModel::Affine,
            centre,
            grid: [0; 3],
            layout: BSplineLayout {
                origin: [0.0; 3],
                spacing: [1.0; 3],
            },
        };
        let (grid, layout) = bspline_grid_for_spacing(shape, [8.0; 3]);
        let bspline = StochasticModel {
            kind: TransformModel::BSpline,
            centre,
            grid,
            layout,
        };
        let count = 3 * grid[0] * grid[1] * grid[2];
        let mut bspline_params = vec![0.0; count];
        for (index, value) in bspline_params.iter_mut().enumerate() {
            *value = 0.3 * ((index as f64) * 0.37).sin();
        }
        let cases = [
            (
                affine,
                vec![
                    1.02, 0.03, -0.01, -0.02, 0.98, 0.02, 0.01, -0.03, 1.01, 0.4, -0.3, 0.2,
                ],
                0.0,
            ),
            (bspline, bspline_params, 0.0),
        ];
        for (model, point, _) in cases {
            let kind = model.kind;
            let mut context = level_context(&params, &fixed, &moving, model);
            context.prior = prior;
            let at = context.evaluate(&point, source, true).unwrap();
            assert!(at.valid > 1000, "{kind:?}: {} valid samples", at.valid);
            let h = 1.0e-5;
            let probes: Vec<usize> = if kind == TransformModel::Affine {
                (0..12).collect()
            } else {
                (0..point.len()).step_by(point.len() / 23).collect()
            };
            let mut worst = 0.0f64;
            let scale = at.gradient.iter().map(|g| g.abs()).fold(0.0, f64::max);
            for p in probes {
                let mut plus = point.clone();
                plus[p] += h;
                let mut minus = point.clone();
                minus[p] -= h;
                let up = context.evaluate(&plus, source, false).unwrap();
                let down = context.evaluate(&minus, source, false).unwrap();
                assert_eq!(up.valid, at.valid, "{kind:?}: the sample set moved");
                let numeric = (up.cost - down.cost) / (2.0 * h);
                worst = worst.max((numeric - at.gradient[p]).abs());
            }
            assert!(
                worst <= 1.0e-3 * scale,
                "{kind:?}: worst gradient error {worst:e} against a largest component {scale:e}"
            );
        }
    }

    /// The cubic sampler interpolates: at voxel centres it returns the voxel,
    /// and its gradient is the derivative of its value.
    #[test]
    fn the_cubic_sampler_interpolates_and_differentiates() {
        let image = blob_scene([9, 10, 11], [0.0; 3]);
        let sampler = SampledImage::new(image.clone(), SampleOrder::Cubic);
        for (index, value) in image.indexed_iter() {
            let p = [index.0 as f64, index.1 as f64, index.2 as f64];
            let got = sampler.value(p);
            assert!(
                (got - *value as f64).abs() < 1.0e-5,
                "{index:?}: {got} vs {value}"
            );
        }
        for p in [[3.3, 4.7, 5.1], [0.2, 9.1, 0.4], [8.4, 0.3, 10.2]] {
            let (_, gradient) = sampler.value_and_gradient(p);
            for a in 0..3 {
                let mut up = p;
                up[a] += 1.0e-4;
                let mut down = p;
                down[a] -= 1.0e-4;
                let numeric = (sampler.value(up) - sampler.value(down)) / 2.0e-4;
                assert!(
                    (numeric - gradient[a]).abs() < 1.0e-4,
                    "{p:?} axis {a}: {numeric} vs {}",
                    gradient[a]
                );
            }
        }
    }

    /// Refining a B-spline onto the next level's grid keeps the deformation.
    #[test]
    fn upsampling_a_bspline_keeps_its_deformation() {
        let shape = [40, 44, 36];
        let (coarse_grid, coarse_layout) = bspline_grid_for_spacing(shape, [16.0; 3]);
        let (fine_grid, fine_layout) = bspline_grid_for_spacing(shape, [8.0; 3]);
        let coarse = StochasticModel {
            kind: TransformModel::BSpline,
            centre: [0.0; 3],
            grid: coarse_grid,
            layout: coarse_layout,
        };
        let fine = StochasticModel {
            kind: TransformModel::BSpline,
            centre: [0.0; 3],
            grid: fine_grid,
            layout: fine_layout,
        };
        // A smooth field over the control grid: refinement onto a grid that is
        // not nested in the coarse one is a least-error fit, not an identity, so
        // a field that wiggles at the coarse node spacing is not reproduced.
        let per = coarse.per_axis();
        let coefficients: Vec<f64> = (0..coarse.parameter_count())
            .map(|index| {
                let (axis, flat) = (index / per, index % per);
                let i = flat % coarse_grid[0];
                let j = (flat / coarse_grid[0]) % coarse_grid[1];
                let k = flat / (coarse_grid[0] * coarse_grid[1]);
                1.5 * (0.4 * i as f64 + 0.3 * j as f64 - 0.5 * k as f64 + axis as f64).sin()
            })
            .collect();
        let refined = upsample_bspline(&coarse, &coefficients, &fine);
        let mut worst = 0.0f64;
        for i in (0..shape[0]).step_by(3) {
            for j in (0..shape[1]).step_by(5) {
                for k in (0..shape[2]).step_by(4) {
                    let x = [i as f64 + 0.3, j as f64 + 0.1, k as f64 + 0.7];
                    let a = coarse.apply(&coefficients, x);
                    let b = fine.apply(&refined, x);
                    for axis in 0..3 {
                        worst = worst.max((a[axis] - b[axis]).abs());
                    }
                }
            }
        }
        assert!(worst < 5.0e-2, "refined field differs by {worst}");
    }

    /// elastix's grid for a 285-voxel axis at spacing 25: 12 intervals
    /// centred on the image, plus the order, the first node one spacing before
    /// the first interval.
    #[test]
    fn the_bspline_grid_is_placed_as_elastix_places_it() {
        let (size, layout) = bspline_grid_for_spacing([285, 478, 248], [25.0; 3]);
        assert_eq!(size, [15, 23, 13]);
        assert_eq!(layout.origin, [-33.0, -36.5, -26.5]);

        // After an affine, the grid covers the affine's image of the fixed
        // box. The affine and the expected grid are elastix's own, from a real
        // autofluorescence-to-atlas run (`TransformParameters.{0,1}.txt`).
        let matrix = [
            [1.261122, -0.172123, -0.063325],
            [0.105534, 1.092952, -0.024395],
            [0.053962, 0.008768, 0.999653],
        ];
        let centre = [142.0, 238.5, 123.5];
        let translation = [18.769453, 54.163833, 20.403858];
        let offset = std::array::from_fn(|r| {
            centre[r] + translation[r] - (0..3).map(|k| matrix[r][k] * centre[k]).sum::<f64>()
        });
        let (size, layout) =
            bspline_grid_after([285, 478, 248], [25.0; 3], &AffineMap { matrix, offset });
        assert_eq!(size, [15, 23, 13]);
        let spacing = [40.1322914792, 29.2104492227, 26.9657821803];
        let origin = [-120.1565873936, -28.6511081593, -17.8908353149];
        for a in 0..3 {
            assert!((layout.spacing[a] - spacing[a]).abs() < 1e-4, "{layout:?}");
            assert!((layout.origin[a] - origin[a]).abs() < 1e-3, "{layout:?}");
        }
    }

    /// An affine fit recovers a known map, and its answer does not depend on
    /// the number of workers.
    ///
    /// The moving image is larger than the fixed one and offset by `prior`, so
    /// no sample leaves it near the answer. 64 bins: on a scene this smooth the
    /// Parzen estimate with 32 bins is largest at a map that shrinks by 2 per
    /// cent (measured independently in numpy: 1.1400 at 0.98 against 1.1361 at
    /// 1.0), while with 128 bins it is largest at the true map. That is a
    /// property of the estimator, which elastix shares, not of the optimiser.
    #[test]
    fn a_stochastic_affine_fit_recovers_a_known_map_and_is_deterministic() {
        let shape = [40, 44, 48];
        let fixed = blob_scene(shape, [0.0; 3]);
        // Moving voxel `x + 6` holds the scene at `x - (2.5, -1.5, 3)`.
        let moving = blob_scene_in([52, 56, 60], shape, [8.5, 4.5, 9.0]);
        let prior = AffineMap {
            matrix: AffineMap::identity().matrix,
            offset: [6.0; 3],
        };
        let params = StochasticFitParams {
            levels: elastix_levels(2, 200, None),
            samples: 2000,
            fixed_bins: 64,
            moving_bins: 64,
            fixed_order: SampleOrder::Cubic,
            moving_order: SampleOrder::Cubic,
            fixed_kernel_order: 3,
            moving_kernel_order: 3,
            initialisation: StochasticInitialisation::Identity,
            workers: 1,
            ..StochasticFitParams::default()
        };
        let fit = |workers| {
            fit_stochastic(
                &StochasticFitParams {
                    workers,
                    ..params.clone()
                },
                fixed.view(),
                moving.view(),
                &AffineMap::identity(),
                &prior,
                None,
                SpatialFrame::unit(),
            )
            .unwrap()
        };
        let one = fit(1);
        let four = fit(4);
        assert_eq!(one.transform.params, four.transform.params);
        let map = AffineMap::from_params(&one.transform.params).unwrap();
        for x in [[10.0, 12.0, 30.0], [30.0, 20.0, 10.0], [20.0, 35.0, 25.0]] {
            let y = map.apply(x);
            let want = [x[0] + 2.5, x[1] - 1.5, x[2] + 3.0];
            for a in 0..3 {
                assert!(
                    (y[a] - want[a]).abs() < 0.2,
                    "{x:?} -> {y:?}, want {want:?}"
                );
            }
        }
        assert!(one
            .levels
            .iter()
            .all(|l| l.final_cost < l.initial_cost + 1e-3));
    }

    /// A scaled affine step weighs each matrix entry by the mean square of the
    /// coordinate it multiplies, so its norm is a displacement in voxels.
    #[test]
    fn scaled_affine_steps_weigh_matrix_entries_by_coordinate_size() {
        let params = VolumeFitParams {
            scaled_affine_steps: true,
            ..VolumeFitParams::default().affine()
        };
        let weights = step_weights(&params, [10, 20, 30]).unwrap();
        // E[x^2] over 0..n is (n - 1)(2n - 1) / 6.
        assert_eq!(weights[0], 9.0 * 19.0 / 6.0);
        assert_eq!(weights[4], 19.0 * 39.0 / 6.0);
        assert_eq!(weights[8], 29.0 * 59.0 / 6.0);
        assert_eq!(&weights[9..], &[1.0; 3]);
        let mut step = vec![0.0; 12];
        step[2] = 0.01;
        let norm = weighted_norm(&step, Some(&weights));
        assert!((norm - 0.01 * weights[2].sqrt()).abs() < 1e-15);
        let clamped = clamp_step_weighted(step, norm / 2.0, Some(&weights));
        assert!((weighted_norm(&clamped, Some(&weights)) - norm / 2.0).abs() < 1e-12);
        assert!(step_weights(&VolumeFitParams::default().affine(), [10, 20, 30]).is_none());
        assert!(step_weights(
            &VolumeFitParams {
                scaled_affine_steps: true,
                ..VolumeFitParams::default()
            },
            [10, 20, 30]
        )
        .is_none());
    }

    /// A relative tolerance ends the Gauss-Newton loop on a small relative
    /// cost change, where the absolute tolerances alone go on.
    #[test]
    fn a_relative_tolerance_stops_the_gauss_newton_loop() {
        let shape = [16, 18, 20];
        let fixed: Voxels = blob_scene(shape, [0.0; 3]).mapv(f64::from).into();
        let moving: Voxels = blob_scene(shape, [0.6, -0.4, 0.3]).mapv(f64::from).into();
        let base = VolumeFitParams {
            tolerance: 1.0e-9,
            limit: SubstageLimit::of(40).unwrap(),
            ..VolumeFitParams::default()
        };
        let relative = VolumeFitParams {
            relative_tolerance: Some(0.5),
            ..base.clone()
        };
        let stopped = run_with_diagnostics(&relative, &fixed, &moving, 8, 2).unwrap();
        assert_eq!(stopped.converged_by, FitConvergence::CostTolerance);
        assert!(stopped.substages <= 3, "{stopped:?}");
        let tighter = VolumeFitParams {
            relative_tolerance: Some(1.0e-12),
            ..base
        };
        let longer = run_with_diagnostics(&tighter, &fixed, &moving, 8, 2);
        if let Ok(longer) = longer {
            assert!(longer.substages > stopped.substages, "{longer:?}");
        }
    }

    // ------------------------------------------------------ solve_linear --

    /// **The solve is a solve, and it pivots.**
    ///
    /// The first system has a **zero in the leading position**, which is the
    /// case partial pivoting exists for: without the row search the first thing
    /// the elimination does is divide by zero. Its coefficients and its solution
    /// are small integers and dyadic rationals, so every operation the
    /// elimination performs is exact in binary and the answer is asserted on
    /// `==` rather than within a tolerance — an equality that would not survive
    /// a solver that had drifted.
    ///
    /// The second is a 6 x 6 symmetric positive-definite matrix, which is the
    /// shape the normal equations actually have, checked by residual because its
    /// solution has no closed form worth writing down. Its entries are of order
    /// ten and its diagonal is lifted by one, so it is well conditioned and
    /// `n * eps * cond` is a few units in the last place: the measured residual
    /// is `1.8e-15` relative and the bound is `1e-12`. The liveness for that
    /// bound is in the test — a `1e-3` perturbation of the answer moves the
    /// residual to `4.2e-3`, nine decades above it, so the residual check can
    /// see a wrong answer.
    #[test]
    fn the_linear_solve_pivots_and_its_answer_satisfies_the_system() {
        // 2y +  z = 3
        //  x + y + z = 3
        // 2x + y     = 3
        let matrix = vec![
            vec![0.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        assert_eq!(matrix[0][0], 0.0, "the leading pivot is the zero");
        let solved = solve_linear(matrix, vec![3.0, 3.0, 3.0]).expect("a non-singular system");
        assert_eq!(
            solved,
            vec![1.0, 1.0, 1.0],
            "the dyadic system is solved exactly or the elimination has drifted"
        );

        // A symmetric positive-definite system of the shape the fitter builds.
        let n = 6;
        let seed = lcg_volume((n, n, 1));
        let mut matrix = vec![vec![0.0; n]; n];
        for row in 0..n {
            for col in 0..n {
                for k in 0..n {
                    matrix[row][col] += seed[[k, row, 0]] * seed[[k, col, 0]];
                }
            }
            matrix[row][row] += 1.0;
        }
        let rhs: Vec<f64> = (0..n).map(|index| 1.0 + index as f64).collect();
        let solved = solve_linear(matrix.clone(), rhs.clone()).expect("an SPD system is solvable");
        let mut worst = 0.0f64;
        for row in 0..n {
            let product: f64 = (0..n).map(|col| matrix[row][col] * solved[col]).sum();
            worst = worst.max((product - rhs[row]).abs() / rhs[row].abs());
        }
        assert!(
            worst <= 1.0e-12,
            "the residual of the 6 x 6 solve is {worst:e} relative, so what came back is not \
             the solution"
        );

        // The liveness for the residual check: a deliberately wrong answer is
        // rejected by it. Without this the check would pass for a solver that
        // returned its own right-hand side on a matrix near the identity.
        let mut wrong = solved.clone();
        wrong[2] += 1.0e-3;
        let mut wrong_residual = 0.0f64;
        for row in 0..n {
            let product: f64 = (0..n).map(|col| matrix[row][col] * wrong[col]).sum();
            wrong_residual = wrong_residual.max((product - rhs[row]).abs() / rhs[row].abs());
        }
        assert!(
            wrong_residual > 1.0e-6,
            "perturbing the answer by 1e-3 moves the residual only to {wrong_residual:e}, so \
             the residual test above cannot see a wrong answer"
        );

        // A singular system is refused by name rather than answered.
        let singular = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(solve_linear(singular, vec![1.0, 2.0]).is_err());
        // and so is one that is singular only after the first elimination
        let rank_two = vec![
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 1.0, 2.0],
        ];
        assert!(solve_linear(rank_two, vec![1.0, 1.0, 2.0]).is_err());

        println!("the 6 x 6 SPD residual is {worst:e} relative; a 1e-3 perturbation gives {wrong_residual:e}");
    }

    /// **The Gauss-Newton step solves the damped normal equations**, which is
    /// the one thing `solve_step` adds to `solve_linear`: it unpacks the
    /// symmetric storage, adds the damping to the diagonal and negates the
    /// gradient.
    ///
    /// Checked by substituting the step back into `(H + lambda I) s = -g` with
    /// `H` reassembled here from the packed form. The damping is `0.5` rather
    /// than the shipped `1e-6` so that its presence is visible: with `1e-6` the
    /// same assertion would pass for an implementation that dropped it.
    #[test]
    fn the_gauss_newton_step_solves_the_damped_normal_equations() {
        let n = 3;
        let jacobians = [
            [1.0, 0.5, -0.25],
            [0.0, 2.0, 1.0],
            [-1.5, 0.25, 0.75],
            [0.5, -0.5, 2.0],
        ];
        let mut hessian = vec![0.0; symmetric_len(n)];
        let mut gradient = vec![0.0; n];
        for jacobian in &jacobians {
            add_outer(&mut hessian, jacobian);
            for (slot, value) in gradient.iter_mut().zip(jacobian) {
                *slot += 0.3 * value;
            }
        }
        let damping = 0.5;
        let step = solve_step(
            TransformModel::Translation,
            hessian.clone(),
            gradient.clone(),
            damping,
        )
        .expect("the normal equations are not singular here");

        let mut worst = 0.0f64;
        for row in 0..n {
            let mut product = damping * step[row];
            for (col, value) in step.iter().enumerate() {
                product += hessian[symmetric_index(row, col)] * value;
            }
            worst = worst.max((product + gradient[row]).abs());
        }
        assert!(
            worst <= 1.0e-12,
            "(H + {damping} I) s + g is {worst:e} away from zero, so {step:?} is not the \
             damped Gauss-Newton step"
        );

        // Liveness: the undamped step does *not* satisfy the damped equations,
        // so the assertion above is measuring the damping and not just the solve.
        let undamped = solve_step(
            TransformModel::Translation,
            hessian.clone(),
            gradient.clone(),
            0.0,
        )
        .expect("and neither are the undamped ones");
        let mut undamped_residual = 0.0f64;
        for row in 0..n {
            let mut product = damping * undamped[row];
            for (col, value) in undamped.iter().enumerate() {
                product += hessian[symmetric_index(row, col)] * value;
            }
            undamped_residual = undamped_residual.max((product + gradient[row]).abs());
        }
        assert!(
            undamped_residual > 1.0e-3,
            "the undamped step is only {undamped_residual:e} from satisfying the damped \
             equations, so this fixture cannot see the damping"
        );

        // The B-spline branch is the diagonal of the same equations, and it is a
        // different answer here, which is what makes it worth branching on.
        // B-spline evidence is held as the diagonal alone.
        let diagonal_only = (0..n)
            .map(|axis| hessian[symmetric_index(axis, axis)])
            .collect();
        let diagonal = solve_step(
            TransformModel::BSpline,
            diagonal_only,
            gradient.clone(),
            damping,
        )
        .expect("a diagonal solve never fails");
        for axis in 0..n {
            let expected = -gradient[axis]
                / (jacobians.iter().map(|j| j[axis] * j[axis]).sum::<f64>() + damping);
            assert!((diagonal[axis] - expected).abs() <= 1.0e-15);
        }
        assert!(
            (0..n).any(|axis| (diagonal[axis] - step[axis]).abs() > 0.01),
            "the diagonal solve and the full solve agree on this system, so the branch is \
             untested by it"
        );
    }

    // ------------------------------------------------ the trilinear sampler --

    /// **The gradient the sampler returns is the derivative of the value it
    /// returns**, to `1e-8`, at points spread over a whole cell.
    ///
    /// This is the claim the whole optimiser rests on and the one no
    /// end-to-end fit can make: the metric's descent direction is this gradient,
    /// and a gradient that is off by a constant factor, or has two axes
    /// exchanged, still converges on the symmetric fixtures a registration test
    /// tends to use. A central difference of the sampler's own value cannot be
    /// fooled by either.
    ///
    /// The offsets within the cell are asymmetric and none is `0.5`, because a
    /// half-cell sample makes the three axes' bilinear weights equal and would
    /// hide an axis exchange. Each is at least `0.1` from a face, which is a
    /// hundred thousand times [`H`].
    ///
    /// The liveness is in the same test: nowhere in this fixture do two
    /// components of the gradient come within `0.0415` of each other — asserted
    /// at `0.02`, half the measured worst — so an exchanged pair of axes is a
    /// difference six orders of magnitude above the `1e-8` the agreement is
    /// asserted at, and could not pass.
    #[test]
    fn the_trilinear_gradient_is_the_derivative_of_the_value_the_sampler_returns() {
        let volume = lcg_volume((9, 8, 7));
        let view = volume.view();
        let mut worst = 0.0f64;
        let mut smallest_spread = f64::INFINITY;
        for offset in [
            [0.1, 0.3, 0.7],
            [0.4, 0.9, 0.2],
            [0.8, 0.15, 0.45],
            [0.25, 0.6, 0.85],
        ] {
            for cell in [[2usize, 3, 1], [5, 1, 4], [1, 6, 2]] {
                let point = [
                    cell[0] as f64 + offset[0],
                    cell[1] as f64 + offset[1],
                    cell[2] as f64 + offset[2],
                ];
                let (_, gradient) = sample_value_and_gradient(Interpolator::Linear, &view, point)
                    .expect("an interior point");
                for axis in 0..3 {
                    let mut up = point;
                    let mut down = point;
                    up[axis] += H;
                    down[axis] -= H;
                    let high = value_and_gradient(&view, up).expect("still in the cell").0;
                    let low = value_and_gradient(&view, down)
                        .expect("still in the cell")
                        .0;
                    let difference = (high - low) / (2.0 * H);
                    worst = worst.max((gradient[axis] - difference).abs());
                }
                let spread = (0..3)
                    .flat_map(|a| (0..3).map(move |b| (a, b)))
                    .filter(|(a, b)| a != b)
                    .map(|(a, b)| (gradient[a] - gradient[b]).abs())
                    .fold(f64::INFINITY, f64::min);
                smallest_spread = smallest_spread.min(spread);
            }
        }
        assert!(
            worst <= 1.0e-8,
            "the returned gradient is {worst:e} from a central difference of the returned \
             value, which is far past the {:e} that cancellation can explain",
            2.0e-16 / H
        );
        assert!(
            smallest_spread > 0.02,
            "two components of the gradient come within {smallest_spread} of each other \
             somewhere in this fixture, so an exchanged pair of axes could pass it"
        );
        println!(
            "the trilinear gradient matches a central difference to {worst:e}; the closest two \
             components ever come is {smallest_spread:.3}"
        );
    }

    /// **The sampler reproduces a genuinely trilinear field exactly, and its
    /// gradient is that field's analytic derivative** — an oracle that is not a
    /// finite difference of the implementation, so the two claims are
    /// independent.
    ///
    /// `f(x, y, z) = (a0 + a1 x)(b0 + b1 y)(c0 + c1 z)` is in the span the
    /// trilinear interpolant reproduces, so `value` must return it to the last
    /// bit or two and `d f / d x = a1 (b0 + b1 y)(c0 + c1 z)` in closed form.
    /// The three per-axis slopes are different and none is zero, so the three
    /// partial derivatives are three different numbers at every sample.
    ///
    /// The bound is `1e-12` absolute, not zero: the interpolant computes the
    /// same quantity by seven `lerp`s where the closed form computes it by two
    /// multiplications, and those do not round identically. Values here reach
    /// 60, so `1e-12` is about `1e-14` relative.
    #[test]
    fn the_sampler_is_exact_on_a_trilinear_field_and_so_is_its_gradient() {
        let line = |base: f64, slope: f64, at: f64| base + slope * at;
        let (a0, a1) = (2.0, 0.75);
        let (b0, b1) = (-1.0, 0.5);
        let (c0, c1) = (3.0, -0.25);
        let volume = Array3::from_shape_fn((9, 8, 7), |(x, y, z)| {
            line(a0, a1, x as f64) * line(b0, b1, y as f64) * line(c0, c1, z as f64)
        });
        let view = volume.view();

        let mut worst_value = 0.0f64;
        let mut worst_gradient = 0.0f64;
        for point in [
            [1.3, 2.7, 1.1],
            [4.5, 0.25, 3.75],
            [6.9, 5.4, 4.2],
            [0.0, 0.0, 0.0],
        ] {
            let (value, gradient) =
                value_and_gradient(&view, point).expect("every point here is interior");
            let (fx, fy, fz) = (
                line(a0, a1, point[0]),
                line(b0, b1, point[1]),
                line(c0, c1, point[2]),
            );
            worst_value = worst_value.max((value - fx * fy * fz).abs());
            let analytic = [a1 * fy * fz, b1 * fx * fz, c1 * fx * fy];
            for axis in 0..3 {
                worst_gradient = worst_gradient.max((gradient[axis] - analytic[axis]).abs());
            }
            assert!(
                (0..3).all(|a| (0..3).all(|b| a == b || (analytic[a] - analytic[b]).abs() > 0.1)),
                "the analytic partials {analytic:?} are too close together at {point:?} for a \
                 permuted gradient to be caught"
            );
        }
        assert!(
            worst_value <= 1.0e-12,
            "the interpolant is {worst_value:e} from a field it reproduces exactly"
        );
        assert!(
            worst_gradient <= 1.0e-12,
            "the gradient is {worst_gradient:e} from the analytic derivative"
        );
        println!(
            "on a trilinear field the sampler is exact to {worst_value:e} and its gradient to \
             {worst_gradient:e}"
        );
    }

    /// **The sampler refuses a point whose interpolation cell would leave the
    /// array**, and it refuses it on the *upper* face too.
    ///
    /// The upper face is the interesting one: `floor(point) + 1` has to be a
    /// valid index, so the last voxel on each axis is not a legal base cell even
    /// though it is a legal index. An implementation that only checked
    /// `floor(point) < extent` would read one past the end there, and the moving
    /// image is fetched with a halo precisely so that this refusal is what
    /// bounds the sample set rather than a fault.
    #[test]
    fn the_sampler_refuses_a_cell_that_leaves_the_array() {
        let volume = lcg_volume((5, 4, 3));
        let view = volume.view();
        // The last legal base cell on every axis, and just past it.
        assert!(value_and_gradient(&view, [3.999, 2.999, 1.999]).is_some());
        assert!(value_and_gradient(&view, [4.0, 2.0, 1.0]).is_none());
        assert!(value_and_gradient(&view, [3.0, 3.0, 1.0]).is_none());
        assert!(value_and_gradient(&view, [3.0, 2.0, 2.0]).is_none());
        // and below zero, where the floor is negative rather than merely large
        assert!(value_and_gradient(&view, [-0.001, 1.0, 1.0]).is_none());
        assert!(value_and_gradient(&view, [0.0, 0.0, 0.0]).is_some());
    }

    // ------------------------------------------------ the parameter Jacobian --

    /// **For every one of the three models, the parameter Jacobian is the
    /// derivative of the sampled moving-image value with respect to the
    /// parameters** — the chain rule that connects `transform_point` to
    /// `parameter_jacobian`, measured against central differences.
    ///
    /// This is the one relation a converging fit cannot witness. If a column of
    /// the affine Jacobian used the wrong coordinate, or the B-spline Jacobian
    /// used the wrong control point, the optimiser would still descend — just
    /// towards something else — and every end-to-end assertion in
    /// `tests/align.rs` that lands on a symmetric fixture would still pass.
    /// Here each of the 12 affine and 192 B-spline parameters is perturbed on
    /// its own and the resulting change in the sampled value is compared with
    /// the column claimed for it.
    ///
    /// The sample point is placed at an unequal fractional offset well inside
    /// the volume, so a perturbation of `H` — which for the affine moves the
    /// mapped point by at most `H * 8`, the largest coordinate in play — stays
    /// inside one interpolation cell and inside one B-spline cell. The bound is
    /// `1e-8`, as everywhere else here.
    #[test]
    fn the_parameter_jacobian_is_the_derivative_of_the_sampled_value_for_every_model() {
        let volume = lcg_volume((16, 15, 14));
        let view = volume.view();
        let shape = [16usize, 15, 14];
        let points = [[6.3, 5.7, 4.1], [8.4, 7.15, 6.6]];

        for model in [
            TransformModel::Translation,
            TransformModel::Affine,
            TransformModel::BSpline,
        ] {
            let mut config = VolumeFitParams::new();
            config.model = model;
            let base = match model {
                // No component is chosen to land the mapped point on an integer:
                // the interpolant's slope changes at a cell face, so a central
                // difference taken across one measures two different linear
                // pieces and would fail for a reason that has nothing to do with
                // the Jacobian.
                TransformModel::Translation => vec![0.35, -0.23, 0.17],
                // Near identity, so the mapped point stays where the sampler can
                // read it, but off it on every entry so that no column of the
                // Jacobian is being evaluated at a degenerate configuration.
                TransformModel::Affine => vec![
                    1.01, 0.02, -0.01, 0.015, 0.99, 0.02, -0.02, 0.01, 1.005, 0.3, -0.25, 0.2,
                ],
                TransformModel::BSpline => {
                    let mut params = vec![0.0; 3 * config.bspline.control_points()];
                    for (index, slot) in params.iter_mut().enumerate() {
                        *slot = 0.1 * ((index % 7) as f64 - 3.0) / 3.0;
                    }
                    params
                }
            };

            let mut worst = 0.0f64;
            let mut largest = 0.0f64;
            for point in points {
                let mapped = transform_point(&config, shape, &base, point);
                let (_, image_gradient) =
                    value_and_gradient(&view, mapped).expect("the mapped point is interior");
                let jacobian = parameter_jacobian(&config, shape, image_gradient, point);
                assert_eq!(jacobian.len(), base.len());

                for index in 0..base.len() {
                    let sample = |delta: f64| {
                        let mut moved = base.clone();
                        moved[index] += delta;
                        let at = transform_point(&config, shape, &moved, point);
                        value_and_gradient(&view, at)
                            .expect("a perturbed point stays in the same cell")
                            .0
                    };
                    let difference = (sample(H) - sample(-H)) / (2.0 * H);
                    worst = worst.max((jacobian[index] - difference).abs());
                    largest = largest.max(jacobian[index].abs());
                }
            }
            assert!(
                worst <= 1.0e-8,
                "{model:?}: the parameter Jacobian is {worst:e} from a central difference of \
                 the value it is supposed to be the derivative of"
            );
            assert!(
                largest > 0.01,
                "{model:?}: every Jacobian entry is smaller than {largest:e}, so agreeing with \
                 a finite difference says nothing — an all-zero Jacobian would pass"
            );
            println!("{model:?}: {} parameters, worst finite-difference gap {worst:e}, largest entry {largest:.4}", base.len());
        }
    }

    // ---------------------------------------------------- the spline basis --

    /// **The cubic B-spline basis is a partition of unity with support two**,
    /// which is what makes a constant displacement representable — and a
    /// constant displacement is the answer
    /// `a_uniform_shift_is_recovered_as_a_uniform_bspline_displacement_field`
    /// asserts the fitter finds.
    ///
    /// The four taps that overlap any position must sum to `1` exactly, or a
    /// field with every coefficient set to `d` would not displace by `d`. The
    /// sum is asserted to `1e-15` at a spread of positions across a cell — not
    /// on `==`, because the four basis values are not dyadic — and the support
    /// is checked at and past `2`, where an implementation that used the wrong
    /// piecewise branch would leak.
    #[test]
    fn the_cubic_basis_is_a_partition_of_unity_with_support_two() {
        for step in 0..64 {
            let u = step as f64 / 64.0;
            let sum: f64 = (-1..3).map(|tap| cubic_basis(u - tap as f64)).sum();
            assert!(
                (sum - 1.0).abs() <= 1.0e-15,
                "the four taps at u = {u} sum to {sum}, so a constant field would not be a \
                 constant displacement"
            );
        }
        assert_eq!(cubic_basis(2.0), 0.0);
        assert_eq!(cubic_basis(-2.0), 0.0);
        assert_eq!(cubic_basis(3.5), 0.0);
        assert!(cubic_basis(1.999) > 0.0);
        // symmetric, and peaked at the centre
        for t in [0.0f64, 0.25, 0.75, 1.5] {
            assert_eq!(cubic_basis(t), cubic_basis(-t));
        }
        assert!(cubic_basis(0.0) > cubic_basis(0.5));
        assert!(cubic_basis(0.5) > cubic_basis(1.5));

        // and the same, through the weights the fitter actually uses: the 64
        // weights at any supported point sum to one.
        let grid = ControlGrid::default();
        let shape = [24usize, 20, 16];
        for point in [[1.0, 1.0, 1.0], [11.5, 9.25, 7.75], [20.0, 16.0, 13.0]] {
            let weights = bspline_weights(grid, shape, point).expect("a supported point");
            let sum: f64 = weights.iter().map(|(_, weight)| weight).sum();
            assert!(
                (sum - 1.0).abs() <= 1.0e-14,
                "the control weights at {point:?} sum to {sum}"
            );
        }
        // Outside the grid's supported region there are no weights at all, and
        // the displacement is zero rather than an extrapolation.
        assert!(bspline_weights(grid, shape, [-1.0, 5.0, 5.0]).is_none());
        assert_eq!(
            bspline_displacement(
                grid,
                shape,
                &vec![7.0; 3 * grid.control_points()],
                [-1.0, 5.0, 5.0]
            ),
            [0.0, 0.0, 0.0]
        );
        // and a uniform coefficient field displaces by exactly that coefficient
        let uniform = vec![2.5; 3 * grid.control_points()];
        let displacement = bspline_displacement(grid, shape, &uniform, [11.5, 9.25, 7.75]);
        for axis in 0..3 {
            assert!(
                (displacement[axis] - 2.5).abs() <= 1.0e-14,
                "a uniform 2.5 field displaces by {displacement:?}"
            );
        }
    }

    /// **The packed symmetric storage addresses each pair exactly once**, and
    /// the outer product written into it is the symmetric matrix it claims.
    ///
    /// `symmetric_index` is what both the accumulation and the solve index the
    /// Hessian through, so a collision — two different pairs landing on the same
    /// slot — would silently add one parameter's curvature to another's, and
    /// nothing downstream could tell. The check is that the map from the
    /// `n(n+1)/2` unordered pairs onto slots is a bijection, and that it is
    /// order-independent.
    #[test]
    fn the_packed_symmetric_index_is_a_bijection_onto_its_slots() {
        for n in [1usize, 2, 3, 12] {
            let mut seen = vec![0usize; symmetric_len(n)];
            for col in 0..n {
                for row in 0..=col {
                    let index = symmetric_index(row, col);
                    assert_eq!(
                        index,
                        symmetric_index(col, row),
                        "n = {n}: ({row}, {col}) and ({col}, {row}) are different slots"
                    );
                    assert!(
                        index < seen.len(),
                        "n = {n}: ({row}, {col}) is out of range"
                    );
                    seen[index] += 1;
                }
            }
            assert!(
                seen.iter().all(|count| *count == 1),
                "n = {n}: the pair-to-slot map is not a bijection: {seen:?}"
            );
        }

        // and the outer product accumulated through it is J J^T
        let jacobian = [1.5, -0.5, 2.0, 0.25];
        let mut hessian = vec![0.0; symmetric_len(4)];
        add_outer(&mut hessian, &jacobian);
        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(
                    hessian[symmetric_index(row, col)],
                    jacobian[row] * jacobian[col]
                );
            }
        }
    }
}
