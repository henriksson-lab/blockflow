// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Level-set segmentation and evolution.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::assemble::{ImageId, Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Chain, Slicing, SourceInput, SourceInputs};
use crate::reach::Reach;
use crate::voxels::Voxels;

use super::distance::{distance_transform, DistanceParams};
use super::shapes_agree;

/// Conservative upper bound for the explicit Euler step used here.
///
/// This is intentionally a guardrail, not a tuned CFL estimator for every force
/// combination. Callers that need larger steps should substep explicitly.
pub const MAX_EXPLICIT_LEVEL_SET_DT: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelSetReport {
    pub iterations: usize,
    pub max_delta: f64,
}

/// Parameters for geodesic active-contour evolution.
///
/// The evolved field uses the convention `phi <= 0` for the inside of the
/// contour. The edge indicator is
/// `g = 1 / (1 + edge_weight * |grad(image)|^2)`, and one explicit Euler step is
///
/// ```text
/// phi' = phi + dt * (g * (curvature + balloon) * |grad(phi)|
///                    + grad(g) dot grad(phi)).
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicLevelSetConfig {
    pub dt: f64,
    pub iterations: usize,
    pub edge_weight: f64,
    pub balloon: f64,
    pub curvature_epsilon: f64,
    pub reinitialize_every: Option<usize>,
    pub reinitialize: DistanceParams,
}

impl GeodesicLevelSetConfig {
    pub fn new(dt: f64, iterations: usize, edge_weight: f64, balloon: f64) -> Result<Self> {
        let config = Self {
            dt,
            iterations,
            edge_weight,
            balloon,
            curvature_epsilon: 1.0e-12,
            reinitialize_every: None,
            reinitialize: DistanceParams::default(),
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(self) -> Result<()> {
        if !(self.dt.is_finite() && self.dt > 0.0) {
            return Err(Error::InvalidArgument(format!(
                "level-set time step must be finite and positive, got {}",
                self.dt
            )));
        }
        if self.dt > MAX_EXPLICIT_LEVEL_SET_DT {
            return Err(Error::InvalidArgument(format!(
                "level-set time step {} is above the explicit stability guard {}",
                self.dt, MAX_EXPLICIT_LEVEL_SET_DT
            )));
        }
        if !(self.edge_weight.is_finite() && self.edge_weight >= 0.0) {
            return Err(Error::InvalidArgument(format!(
                "level-set edge weight must be finite and non-negative, got {}",
                self.edge_weight
            )));
        }
        if !self.balloon.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "level-set balloon force must be finite, got {}",
                self.balloon
            )));
        }
        if !(self.curvature_epsilon.is_finite() && self.curvature_epsilon > 0.0) {
            return Err(Error::InvalidArgument(format!(
                "level-set curvature epsilon must be finite and positive, got {}",
                self.curvature_epsilon
            )));
        }
        if self.reinitialize_every == Some(0) {
            return Err(Error::InvalidArgument(
                "level-set reinitialization cadence must be positive when it is set".to_string(),
            ));
        }
        self.reinitialize.squared_sampling()?;
        Ok(())
    }
}

impl Default for GeodesicLevelSetConfig {
    fn default() -> Self {
        Self {
            dt: 0.25,
            iterations: 1,
            edge_weight: 1.0,
            balloon: 0.0,
            curvature_epsilon: 1.0e-12,
            reinitialize_every: None,
            reinitialize: DistanceParams::default(),
        }
    }
}

/// Parameters for Chan-Vese region-based level-set evolution.
///
/// Each step computes the current inside/outside means from `phi <= 0`, then
/// applies
///
/// ```text
/// phi' = phi + dt * delta(phi)
///              * (mu * curvature(phi)
///                 - lambda_inside * (image - c_inside)^2
///                 + lambda_outside * (image - c_outside)^2).
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChanVeseLevelSetConfig {
    pub dt: f64,
    pub iterations: usize,
    pub mu: f64,
    pub lambda_inside: f64,
    pub lambda_outside: f64,
    pub smoothing_epsilon: f64,
    pub curvature_epsilon: f64,
    pub reinitialize_every: Option<usize>,
    pub reinitialize: DistanceParams,
}

impl ChanVeseLevelSetConfig {
    pub fn new(
        dt: f64,
        iterations: usize,
        mu: f64,
        lambda_inside: f64,
        lambda_outside: f64,
    ) -> Result<Self> {
        let config = Self {
            dt,
            iterations,
            mu,
            lambda_inside,
            lambda_outside,
            smoothing_epsilon: 1.0,
            curvature_epsilon: 1.0e-12,
            reinitialize_every: None,
            reinitialize: DistanceParams::default(),
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(self) -> Result<()> {
        if !(self.dt.is_finite() && self.dt > 0.0) {
            return Err(Error::InvalidArgument(format!(
                "Chan-Vese time step must be finite and positive, got {}",
                self.dt
            )));
        }
        if self.dt > MAX_EXPLICIT_LEVEL_SET_DT {
            return Err(Error::InvalidArgument(format!(
                "Chan-Vese time step {} is above the explicit stability guard {}",
                self.dt, MAX_EXPLICIT_LEVEL_SET_DT
            )));
        }
        if !(self.mu.is_finite() && self.mu >= 0.0) {
            return Err(Error::InvalidArgument(format!(
                "Chan-Vese curvature weight must be finite and non-negative, got {}",
                self.mu
            )));
        }
        if !(self.lambda_inside.is_finite() && self.lambda_inside >= 0.0) {
            return Err(Error::InvalidArgument(format!(
                "Chan-Vese inside weight must be finite and non-negative, got {}",
                self.lambda_inside
            )));
        }
        if !(self.lambda_outside.is_finite() && self.lambda_outside >= 0.0) {
            return Err(Error::InvalidArgument(format!(
                "Chan-Vese outside weight must be finite and non-negative, got {}",
                self.lambda_outside
            )));
        }
        if !(self.smoothing_epsilon.is_finite() && self.smoothing_epsilon > 0.0) {
            return Err(Error::InvalidArgument(format!(
                "Chan-Vese smoothing epsilon must be finite and positive, got {}",
                self.smoothing_epsilon
            )));
        }
        if !(self.curvature_epsilon.is_finite() && self.curvature_epsilon > 0.0) {
            return Err(Error::InvalidArgument(format!(
                "Chan-Vese curvature epsilon must be finite and positive, got {}",
                self.curvature_epsilon
            )));
        }
        if self.reinitialize_every == Some(0) {
            return Err(Error::InvalidArgument(
                "Chan-Vese reinitialization cadence must be positive when it is set".to_string(),
            ));
        }
        self.reinitialize.squared_sampling()?;
        Ok(())
    }
}

impl Default for ChanVeseLevelSetConfig {
    fn default() -> Self {
        Self {
            dt: 0.25,
            iterations: 1,
            mu: 0.25,
            lambda_inside: 1.0,
            lambda_outside: 1.0,
            smoothing_epsilon: 1.0,
            curvature_epsilon: 1.0e-12,
            reinitialize_every: None,
            reinitialize: DistanceParams::default(),
        }
    }
}

/// Build a signed-distance level set from a mask.
///
/// `true` is inside and therefore negative in the returned field.
pub fn signed_distance_level_set(
    mask: ArrayView3<'_, bool>,
    params: &DistanceParams,
) -> Result<Array3<f64>> {
    let inside = distance_transform(mask, params)?;
    let outside_mask = mask.mapv(|inside| !inside);
    let outside = distance_transform(outside_mask.view(), params)?;
    Ok(outside - inside)
}

/// Write `phi <= 0` as a boolean mask.
pub fn level_set_mask_into(
    phi: ArrayView3<'_, f64>,
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(phi.shape(), out.shape(), "level_set_mask_into")?;
    for ((i, j, k), slot) in out.indexed_iter_mut() {
        let value = phi[[i, j, k]];
        if !value.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "level-set field contains non-finite value at [{i}, {j}, {k}]"
            )));
        }
        *slot = value <= 0.0;
    }
    Ok(())
}

/// Evolve `initial_phi` for `config.iterations` explicit geodesic steps.
pub fn geodesic_level_set_into<T>(
    image: ArrayView3<'_, T>,
    initial_phi: ArrayView3<'_, f64>,
    config: GeodesicLevelSetConfig,
    out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    geodesic_level_set_reporting_into(image, initial_phi, config, out).map(|_| ())
}

/// Evolve `initial_phi`, returning how many steps ran and the largest final
/// per-voxel change observed in one step.
pub fn geodesic_level_set_reporting_into<T>(
    image: ArrayView3<'_, T>,
    initial_phi: ArrayView3<'_, f64>,
    config: GeodesicLevelSetConfig,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<LevelSetReport>
where
    T: Copy + Into<f64>,
{
    config.validate()?;
    shapes_agree(
        image.shape(),
        initial_phi.shape(),
        "geodesic_level_set_into (phi)",
    )?;
    shapes_agree(image.shape(), out.shape(), "geodesic_level_set_into (out)")?;
    validate_finite(image, "level-set image")?;
    validate_finite(initial_phi, "level-set field")?;

    if config.iterations == 0 {
        out.assign(&initial_phi);
        return Ok(LevelSetReport {
            iterations: 0,
            max_delta: 0.0,
        });
    }

    let edge = edge_indicator(image, config.edge_weight);
    let mut current = initial_phi.to_owned();
    let mut next = Array3::<f64>::zeros(current.raw_dim());
    let mut max_delta = 0.0f64;
    for iteration in 0..config.iterations {
        geodesic_step_from_edge(edge.view(), current.view(), config, next.view_mut())?;
        max_delta = max_delta.max(max_abs_difference(current.view(), next.view()));
        std::mem::swap(&mut current, &mut next);
        let before_reinit = current.clone();
        reinitialize_if_due(
            &mut current,
            iteration,
            config.reinitialize_every,
            &config.reinitialize,
        )?;
        max_delta = max_delta.max(max_abs_difference(before_reinit.view(), current.view()));
    }
    out.assign(&current);
    Ok(LevelSetReport {
        iterations: config.iterations,
        max_delta,
    })
}

/// One explicit geodesic active-contour step.
pub fn geodesic_level_set_step_into<T>(
    image: ArrayView3<'_, T>,
    phi: ArrayView3<'_, f64>,
    config: GeodesicLevelSetConfig,
    out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    config.validate()?;
    shapes_agree(
        image.shape(),
        phi.shape(),
        "geodesic_level_set_step_into (phi)",
    )?;
    validate_finite(image, "level-set image")?;
    validate_finite(phi, "level-set field")?;
    let edge = edge_indicator(image, config.edge_weight);
    geodesic_step_from_edge(edge.view(), phi, config, out)
}

/// Append one planner-visible geodesic step per requested iteration.
///
/// This is the explicit-step planner path available today. It writes an
/// intermediate image after each Euler step so the next phase can read the
/// updated `phi`, while every step sources the fixed intensity image by its
/// absolute image id.
pub fn append_geodesic_level_set_phases(
    builder: &mut PlanBuilder,
    image: impl Into<ImageId> + Copy,
    config: GeodesicLevelSetConfig,
) -> Result<Vec<Phase>> {
    config.validate()?;
    if config.reinitialize_every.is_some() {
        return Err(Error::InvalidArgument(
            "geodesic level-set planner helper cannot yet express cadence-based \
             reinitialization; run the resident helper or append the distance-transform \
             phases explicitly"
                .to_string(),
        ));
    }
    let mut phases = Vec::with_capacity(config.iterations);
    let mut step_config = config;
    step_config.iterations = 1;
    for _ in 0..config.iterations {
        let op = GeodesicLevelSetStepOp::new("geodesic-level-set-step", image, step_config)?;
        phases.push(builder.pixels(Chain::op(op))?);
    }
    Ok(phases)
}

/// Evolve `initial_phi` for `config.iterations` explicit Chan-Vese steps.
pub fn chan_vese_level_set_into<T>(
    image: ArrayView3<'_, T>,
    initial_phi: ArrayView3<'_, f64>,
    config: ChanVeseLevelSetConfig,
    out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    chan_vese_level_set_reporting_into(image, initial_phi, config, out).map(|_| ())
}

/// Evolve `initial_phi`, returning how many steps ran and the largest final
/// per-voxel change observed in one step.
pub fn chan_vese_level_set_reporting_into<T>(
    image: ArrayView3<'_, T>,
    initial_phi: ArrayView3<'_, f64>,
    config: ChanVeseLevelSetConfig,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<LevelSetReport>
where
    T: Copy + Into<f64>,
{
    config.validate()?;
    shapes_agree(
        image.shape(),
        initial_phi.shape(),
        "chan_vese_level_set_into (phi)",
    )?;
    shapes_agree(image.shape(), out.shape(), "chan_vese_level_set_into (out)")?;
    validate_finite(image, "Chan-Vese image")?;
    validate_finite(initial_phi, "Chan-Vese field")?;

    if config.iterations == 0 {
        out.assign(&initial_phi);
        return Ok(LevelSetReport {
            iterations: 0,
            max_delta: 0.0,
        });
    }

    let mut current = initial_phi.to_owned();
    let mut next = Array3::<f64>::zeros(current.raw_dim());
    let mut max_delta = 0.0f64;
    for iteration in 0..config.iterations {
        chan_vese_level_set_step_into(image, current.view(), config, next.view_mut())?;
        max_delta = max_delta.max(max_abs_difference(current.view(), next.view()));
        std::mem::swap(&mut current, &mut next);
        let before_reinit = current.clone();
        reinitialize_if_due(
            &mut current,
            iteration,
            config.reinitialize_every,
            &config.reinitialize,
        )?;
        max_delta = max_delta.max(max_abs_difference(before_reinit.view(), current.view()));
    }
    out.assign(&current);
    Ok(LevelSetReport {
        iterations: config.iterations,
        max_delta,
    })
}

/// One explicit Chan-Vese level-set step.
pub fn chan_vese_level_set_step_into<T>(
    image: ArrayView3<'_, T>,
    phi: ArrayView3<'_, f64>,
    config: ChanVeseLevelSetConfig,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    config.validate()?;
    shapes_agree(
        image.shape(),
        phi.shape(),
        "chan_vese_level_set_step_into (phi)",
    )?;
    shapes_agree(
        image.shape(),
        out.shape(),
        "chan_vese_level_set_step_into (out)",
    )?;
    validate_finite(image, "Chan-Vese image")?;
    validate_finite(phi, "Chan-Vese field")?;

    let means = chan_vese_level_set_means(image, phi)?;
    let shape = shape_of(phi);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                let value = value_at(image, at);
                let inside = value - means.inside;
                let outside = value - means.outside;
                let force = config.mu * curvature(phi, at, config.curvature_epsilon)
                    - config.lambda_inside * inside * inside
                    + config.lambda_outside * outside * outside;
                out[[i, j, k]] =
                    phi[[i, j, k]] + config.dt * smoothed_delta(phi[[i, j, k]], config) * force;
            }
        }
    }
    Ok(())
}

/// Planner-visible shell for one geodesic level-set step.
///
/// The phase input is the current `phi` field. The fixed intensity image is a
/// source input, because it is read at the same physical locations but is not
/// the evolving value threaded between iterations.
#[derive(Debug, Clone, PartialEq)]
pub struct GeodesicLevelSetStepOp {
    name: &'static str,
    image: ImageId,
    config: GeodesicLevelSetConfig,
    cost: f64,
}

impl GeodesicLevelSetStepOp {
    pub fn new(
        name: &'static str,
        image: impl Into<ImageId>,
        config: GeodesicLevelSetConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            name,
            image: image.into(),
            config,
            cost: GEODESIC_LEVEL_SET_STEP_COST,
        })
    }

    pub fn config(&self) -> GeodesicLevelSetConfig {
        self.config
    }

    pub fn image(&self) -> ImageId {
        self.image
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for GeodesicLevelSetStepOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        2
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::symmetric([2, 2, 2])
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(self.image, Reach::symmetric([2, 2, 2])).holding(Dtype::F64)]
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, _input: &Voxels, _out: &mut Voxels, _at: &Anchor) -> Result<()> {
        Err(Error::InvalidArgument(format!(
            "{}: the intensity image comes from image {}, so this op has no answer from its \
             input alone. It is applied through `apply_with`.",
            self.name,
            self.image.index()
        )))
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<()> {
        let image = sources.get(self.image)?;
        let phi = input.view::<f64>()?;
        let out = out.view_mut::<f64>()?;
        dispatch_f64_input!(
            image,
            Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; source image {} has type f16",
                self.name,
                self.image.index()
            )),
            |view| { geodesic_level_set_step_into(view, phi, self.config, out) }
        )
    }

    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (self.config.balloon == 0.0).then_some(value)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Placeholder measured cost for one explicit geodesic step, in voxelwise-map
/// units. It should be retaken through the cost harness before planner tuning.
pub const GEODESIC_LEVEL_SET_STEP_COST: f64 = 90.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChanVeseMeans {
    pub inside: f64,
    pub outside: f64,
}

/// Reduce the current Chan-Vese inside/outside means for `phi <= 0`.
///
/// Planner-visible Chan-Vese needs this reduction as an explicit barrier before
/// a later update phase can broadcast the scalars into the stencil step.
pub fn chan_vese_level_set_means<T>(
    image: ArrayView3<'_, T>,
    phi: ArrayView3<'_, f64>,
) -> Result<ChanVeseMeans>
where
    T: Copy + Into<f64>,
{
    shapes_agree(image.shape(), phi.shape(), "Chan-Vese means")?;
    validate_finite(image, "Chan-Vese image")?;
    validate_finite(phi, "Chan-Vese field")?;
    let mut inside_sum = 0.0;
    let mut inside_count = 0usize;
    let mut outside_sum = 0.0;
    let mut outside_count = 0usize;
    for ((i, j, k), &level) in phi.indexed_iter() {
        let value = image[[i, j, k]].into();
        if level <= 0.0 {
            inside_sum += value;
            inside_count += 1;
        } else {
            outside_sum += value;
            outside_count += 1;
        }
    }
    if inside_count == 0 || outside_count == 0 {
        return Err(Error::InvalidArgument(format!(
            "Chan-Vese needs both inside and outside voxels under phi <= 0; got {inside_count} \
             inside and {outside_count} outside"
        )));
    }
    Ok(ChanVeseMeans {
        inside: inside_sum / inside_count as f64,
        outside: outside_sum / outside_count as f64,
    })
}

fn smoothed_delta(phi: f64, config: ChanVeseLevelSetConfig) -> f64 {
    config.smoothing_epsilon
        / (std::f64::consts::PI * (config.smoothing_epsilon * config.smoothing_epsilon + phi * phi))
}

fn reinitialize_if_due(
    phi: &mut Array3<f64>,
    iteration: usize,
    cadence: Option<usize>,
    params: &DistanceParams,
) -> Result<()> {
    let Some(cadence) = cadence else {
        return Ok(());
    };
    if (iteration + 1) % cadence != 0 {
        return Ok(());
    }
    let mask = phi.mapv(|value| value <= 0.0);
    *phi = signed_distance_level_set(mask.view(), params)?;
    Ok(())
}

fn max_abs_difference(left: ArrayView3<'_, f64>, right: ArrayView3<'_, f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn geodesic_step_from_edge(
    edge: ArrayView3<'_, f64>,
    phi: ArrayView3<'_, f64>,
    config: GeodesicLevelSetConfig,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    shapes_agree(phi.shape(), out.shape(), "geodesic level-set step")?;
    let shape = shape_of(phi);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                let grad_phi = gradient(phi, at);
                let grad_norm = norm(grad_phi);
                let curvature = curvature(phi, at, config.curvature_epsilon);
                let grad_edge = gradient(edge, at);
                let advection = dot(grad_edge, grad_phi);
                let speed = edge[[i, j, k]] * (curvature + config.balloon) * grad_norm + advection;
                out[[i, j, k]] = phi[[i, j, k]] + config.dt * speed;
            }
        }
    }
    Ok(())
}

fn edge_indicator<T>(image: ArrayView3<'_, T>, edge_weight: f64) -> Array3<f64>
where
    T: Copy + Into<f64>,
{
    let shape = shape_of(image);
    let mut edge = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let grad = gradient(image, [i, j, k]);
                edge[[i, j, k]] = 1.0 / (1.0 + edge_weight * dot(grad, grad));
            }
        }
    }
    edge
}

fn curvature(phi: ArrayView3<'_, f64>, at: [usize; 3], epsilon: f64) -> f64 {
    let shape = shape_of(phi);
    let mut divergence = 0.0;
    for axis in 0..3 {
        let plus = step(at, shape, axis, 1);
        let minus = step(at, shape, axis, -1);
        divergence += 0.5
            * (normal_component(phi, plus, axis, epsilon)
                - normal_component(phi, minus, axis, epsilon));
    }
    divergence
}

fn normal_component(phi: ArrayView3<'_, f64>, at: [usize; 3], axis: usize, epsilon: f64) -> f64 {
    let grad = gradient(phi, at);
    grad[axis] / (dot(grad, grad) + epsilon).sqrt()
}

fn gradient<T>(field: ArrayView3<'_, T>, at: [usize; 3]) -> [f64; 3]
where
    T: Copy + Into<f64>,
{
    let shape = shape_of(field);
    let mut grad = [0.0; 3];
    for axis in 0..3 {
        let plus = step(at, shape, axis, 1);
        let minus = step(at, shape, axis, -1);
        grad[axis] = 0.5 * (value_at(field, plus) - value_at(field, minus));
    }
    grad
}

fn step(mut at: [usize; 3], shape: [usize; 3], axis: usize, delta: isize) -> [usize; 3] {
    let last = shape[axis].saturating_sub(1);
    at[axis] = if delta < 0 {
        at[axis].saturating_sub(1)
    } else {
        (at[axis] + 1).min(last)
    };
    at
}

fn value_at<T>(field: ArrayView3<'_, T>, at: [usize; 3]) -> f64
where
    T: Copy + Into<f64>,
{
    field[[at[0], at[1], at[2]]].into()
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn shape_of<T>(field: ArrayView3<'_, T>) -> [usize; 3] {
    [field.shape()[0], field.shape()[1], field.shape()[2]]
}

fn validate_finite<T>(field: ArrayView3<'_, T>, what: &str) -> Result<()>
where
    T: Copy + Into<f64>,
{
    for ((i, j, k), &value) in field.indexed_iter() {
        let value = value.into();
        if !value.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "{what} contains non-finite value at [{i}, {j}, {k}]"
            )));
        }
    }
    Ok(())
}
