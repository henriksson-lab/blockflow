// SPDX-License-Identifier: MIT
//
// Block-reduced fitting of a 3-D coordinate map between two scalar volumes.
//
// This is a general optimizer shape: each block maps two image regions to local
// evidence, all evidence is reduced in a deterministic order, then one global
// parameter state is updated. The transform family and sampling policy are
// parameters; file-format adapters belong in caller crates.

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

use ndarray::{Array3, ArrayView3};
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
    /// Fixed-image geometry written to the emitted transform and used to convert
    /// the optimized voxel-space parameters to physical coordinates.
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

    fn finish_metric(&mut self, metric: Metric, bins: usize, damping: f64) -> Result<()> {
        match metric {
            Metric::MeanSquares => Ok(()),
            Metric::NormalizedCorrelation => self.finish_normalized_correlation(damping),
            Metric::MutualInformation => self.finish_mutual_information(bins, damping),
        }
    }

    fn finish_normalized_correlation(&mut self, damping: f64) -> Result<()> {
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
            self.hessian[symmetric_index(axis, axis)] = scale;
        }
        Ok(())
    }

    fn finish_mutual_information(&mut self, bins: usize, damping: f64) -> Result<()> {
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
            self.hessian[symmetric_index(axis, axis)] = scale;
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
    let h = symmetric_len(n);
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
            hessian: vec![0.0; symmetric_len(n)],
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
            hessian: vec![0.0; symmetric_len(n)],
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
            let translation = self
                .initial_translation()
                .unwrap_or(total.moments.translation()?);
            let next = OptimizerState {
                params: params_with_translation(&self.params, translation),
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
        let step = clamp_step(step, self.params.max_step);
        let mut params = previous.params;
        for (param, delta) in params.iter_mut().zip(&step) {
            *param += *delta;
        }
        let step_norm = norm(&step);
        let cost_delta = if previous.cost.is_finite() {
            (previous.cost - total.cost).abs()
        } else {
            f64::INFINITY
        };
        let convergence = if step_norm <= self.params.tolerance {
            Some(FitConvergence::StepTolerance)
        } else if cost_delta <= self.params.tolerance {
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

impl FitByMoments {
    fn initial_translation(&self) -> Option<[f64; 3]> {
        self.initial
            .as_ref()
            .map(|params| translation_from_params(self.params.model, params))
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

fn translation_from_params(model: TransformModel, params: &[f64]) -> [f64; 3] {
    match model {
        TransformModel::Translation => [params[0], params[1], params[2]],
        TransformModel::Affine => [params[9], params[10], params[11]],
        TransformModel::BSpline => [0.0, 0.0, 0.0],
    }
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
    hessian[symmetric_index(a, a)] += weight;
    hessian[symmetric_index(b, b)] += weight;
    0.5 * weight * delta * delta
}

fn bspline_flat(grid: ControlGrid, x: usize, y: usize, z: usize) -> usize {
    x + grid.size[0] * (y + grid.size[1] * z)
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
                hessian[symmetric_index(axis, axis)] += value * value;
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
        let diagonal = hessian[symmetric_index(axis, axis)] + damping.max(0.0);
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

fn clamp_step(mut step: Vec<f64>, max_step: f64) -> Vec<f64> {
    let length = norm(&step);
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

/// Run volume fitting and return the transform parameter file text it
/// emits.
///
/// This is the compatibility surface for downstream tooling: callers that need
/// a `TransformParameters.0.txt` can write this string verbatim, and the normal
/// point-transform reader must be able to consume it again.
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
    run(
        params,
        &fixed.clone().into(),
        &moving.clone().into(),
        shape.iter().copied().max().unwrap_or(1),
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
        let block = levels[level]
            .fixed
            .shape()
            .iter()
            .copied()
            .max()
            .unwrap_or(1);
        let (next, state_at_level) = run_with_initial(
            &level_params,
            &levels[level].fixed,
            &levels[level].moving,
            block,
            1,
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
