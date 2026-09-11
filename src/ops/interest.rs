// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Shared response-image construction and point extraction for corners and blobs.

use std::collections::BTreeMap;
use std::sync::Arc;

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::assemble::{ImageId, Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, PhaseView,
    SeamFold, SidecarSize, SourceBlocks,
};
use crate::geometry::BlockGrid;
use crate::op::{Anchor, BlockOp, Slicing, SourceInput};
use crate::points::Point;
use crate::sidecar::Lifecycle;
use crate::table::{Column, RowBuilder, Schema, Value};
use crate::voxels::Voxels;

use super::components::Connectivity;
use super::convolve::{convolve_into, Kernel, Sense, CONVOLVE_COST_PER_TAP};
use super::detect::{
    encode_moments, merge_moments_with, owner_of, LabelRegionsOp, Moments, RegionMoments,
};
use super::regional::{append_connected as append_regional_connected, regional_maxima_with};
use super::ridge::{gaussian_smooth_into_with, hessian_at, Boundary};
use super::rows::value_at;
use super::shapes_agree;
use super::smooth::{cost_for as gaussian_cost_for, Gaussian};
use super::voxelwise::MAP_COST;

/// Which bright-blob response image to build before peak extraction.
#[derive(Debug, Clone, PartialEq)]
pub enum BlobResponse {
    /// Difference of Gaussians, `G(low_sigma) - G(high_sigma)`.
    DifferenceOfGaussians {
        low_sigma: [f64; 3],
        high_sigma: [f64; 3],
        truncate: f64,
        boundary: Boundary,
    },
    /// Negated Laplacian of Gaussian, so bright blobs are positive.
    LaplacianOfGaussian {
        sigma: [f64; 3],
        truncate: f64,
        boundary: Boundary,
    },
    /// Negated 3-D determinant of the scale-normalised Hessian.
    HessianDeterminant {
        sigma: [f64; 3],
        truncate: f64,
        gamma: f64,
        boundary: Boundary,
    },
}

/// The scale attached to a blob response.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlobScale {
    /// The nominal scale of the response.
    pub sigma: [f64; 3],
    /// The wider scale for a Difference-of-Gaussians response.
    pub upper_sigma: Option<[f64; 3]>,
}

/// A blob detection with the response value and scale that produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlobDetection {
    pub at: [usize; 3],
    pub response: f64,
    pub scale: BlobScale,
}

impl BlobDetection {
    pub fn point(self) -> Point {
        Point::weighted(self.at, self.response)
    }
}

impl BlobResponse {
    pub fn difference_of_gaussians(
        low_sigma: [f64; 3],
        high_sigma: [f64; 3],
        truncate: f64,
        boundary: Boundary,
    ) -> Result<Self> {
        dog_gaussians(low_sigma, high_sigma, truncate)?;
        Ok(Self::DifferenceOfGaussians {
            low_sigma,
            high_sigma,
            truncate,
            boundary,
        })
    }

    pub fn laplacian_of_gaussian(
        sigma: [f64; 3],
        truncate: f64,
        boundary: Boundary,
    ) -> Result<Self> {
        Gaussian::new(sigma, truncate)?;
        Ok(Self::LaplacianOfGaussian {
            sigma,
            truncate,
            boundary,
        })
    }

    pub fn hessian_determinant(
        sigma: [f64; 3],
        truncate: f64,
        gamma: f64,
        boundary: Boundary,
    ) -> Result<Self> {
        validate_hessian_scale(sigma, truncate, gamma)?;
        Ok(Self::HessianDeterminant {
            sigma,
            truncate,
            gamma,
            boundary,
        })
    }

    /// The local input reach implied by this response image.
    pub fn reach(&self, axis: usize) -> usize {
        match self {
            Self::DifferenceOfGaussians {
                low_sigma,
                high_sigma,
                truncate,
                ..
            } => gaussian_reach(low_sigma[axis], *truncate)
                .max(gaussian_reach(high_sigma[axis], *truncate)),
            Self::LaplacianOfGaussian {
                sigma, truncate, ..
            } => gaussian_reach(sigma[axis], *truncate) + 1,
            Self::HessianDeterminant {
                sigma, truncate, ..
            } => gaussian_reach(sigma[axis], *truncate) + 1,
        }
    }

    pub fn scale(&self) -> BlobScale {
        match *self {
            Self::DifferenceOfGaussians {
                low_sigma,
                high_sigma,
                ..
            } => BlobScale {
                sigma: low_sigma,
                upper_sigma: Some(high_sigma),
            },
            Self::LaplacianOfGaussian { sigma, .. } | Self::HessianDeterminant { sigma, .. } => {
                BlobScale {
                    sigma,
                    upper_sigma: None,
                }
            }
        }
    }

    pub fn cost_per_voxel(&self) -> f64 {
        match *self {
            Self::DifferenceOfGaussians {
                low_sigma,
                high_sigma,
                truncate,
                ..
            } => {
                let (low, high) = dog_gaussians(low_sigma, high_sigma, truncate)
                    .expect("BlobResponse constructors validate DoG scales");
                gaussian_cost_for(&low) + gaussian_cost_for(&high) + MAP_COST
            }
            Self::LaplacianOfGaussian {
                sigma, truncate, ..
            } => {
                let gaussian = Gaussian::new(sigma, truncate)
                    .expect("BlobResponse constructors validate LoG scales");
                gaussian_cost_for(&gaussian) + CONVOLVE_COST_PER_TAP * 7.0 + MAP_COST
            }
            Self::HessianDeterminant {
                sigma, truncate, ..
            } => {
                let gaussian = Gaussian::new(sigma, truncate)
                    .expect("BlobResponse constructors validate DoH scales");
                gaussian_cost_for(&gaussian) + CONVOLVE_COST_PER_TAP * 24.0 + MAP_COST
            }
        }
    }

    pub fn response_into<T>(
        &self,
        input: ArrayView3<'_, T>,
        out: ArrayViewMut3<'_, f64>,
    ) -> Result<()>
    where
        T: Copy + Into<f64>,
    {
        match *self {
            Self::DifferenceOfGaussians {
                low_sigma,
                high_sigma,
                truncate,
                boundary,
            } => difference_of_gaussians_response_into(
                input, low_sigma, high_sigma, truncate, boundary, out,
            ),
            Self::LaplacianOfGaussian {
                sigma,
                truncate,
                boundary,
            } => laplacian_of_gaussian_response_into(input, sigma, truncate, boundary, out),
            Self::HessianDeterminant {
                sigma,
                truncate,
                gamma,
                boundary,
            } => hessian_determinant_response_into(input, sigma, truncate, gamma, boundary, out),
        }
    }
}

/// A planner-visible one-image blob response op.
#[derive(Debug, Clone, PartialEq)]
pub struct BlobResponseOp {
    name: &'static str,
    response: BlobResponse,
    cost: f64,
}

impl BlobResponseOp {
    pub fn new(name: &'static str, response: BlobResponse) -> Self {
        let cost = response.cost_per_voxel();
        Self {
            name,
            response,
            cost,
        }
    }

    pub fn difference_of_gaussians(
        name: &'static str,
        low_sigma: [f64; 3],
        high_sigma: [f64; 3],
        truncate: f64,
        boundary: Boundary,
    ) -> Result<Self> {
        Ok(Self::new(
            name,
            BlobResponse::difference_of_gaussians(low_sigma, high_sigma, truncate, boundary)?,
        ))
    }

    pub fn laplacian_of_gaussian(
        name: &'static str,
        sigma: [f64; 3],
        truncate: f64,
        boundary: Boundary,
    ) -> Result<Self> {
        Ok(Self::new(
            name,
            BlobResponse::laplacian_of_gaussian(sigma, truncate, boundary)?,
        ))
    }

    pub fn hessian_determinant(
        name: &'static str,
        sigma: [f64; 3],
        truncate: f64,
        gamma: f64,
        boundary: Boundary,
    ) -> Result<Self> {
        Ok(Self::new(
            name,
            BlobResponse::hessian_determinant(sigma, truncate, gamma, boundary)?,
        ))
    }

    pub fn response(&self) -> &BlobResponse {
        &self.response
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for BlobResponseOp {
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.response.reach(axis)
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let out = out.view_mut::<f64>()?;
        dispatch_f64_input!(
            input,
            Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; `accepts` refuses it before a run starts",
                self.name
            )),
            |view| { self.response.response_into(view, out) }
        )
    }

    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (value == 0.0).then_some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// A named response-plus-peak extractor for bright blob detectors.
#[derive(Debug, Clone, PartialEq)]
pub struct BlobDetector {
    response: BlobResponse,
    connectivity: Connectivity,
    minimum_response: f64,
}

impl BlobDetector {
    pub fn new(
        response: BlobResponse,
        connectivity: Connectivity,
        minimum_response: f64,
    ) -> Result<Self> {
        if !minimum_response.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "a blob detector response threshold must be finite, got {minimum_response}"
            )));
        }
        Ok(Self {
            response,
            connectivity,
            minimum_response,
        })
    }

    pub fn difference_of_gaussians(
        low_sigma: [f64; 3],
        high_sigma: [f64; 3],
        truncate: f64,
        boundary: Boundary,
        connectivity: Connectivity,
        minimum_response: f64,
    ) -> Result<Self> {
        Self::new(
            BlobResponse::difference_of_gaussians(low_sigma, high_sigma, truncate, boundary)?,
            connectivity,
            minimum_response,
        )
    }

    pub fn laplacian_of_gaussian(
        sigma: [f64; 3],
        truncate: f64,
        boundary: Boundary,
        connectivity: Connectivity,
        minimum_response: f64,
    ) -> Result<Self> {
        Self::new(
            BlobResponse::laplacian_of_gaussian(sigma, truncate, boundary)?,
            connectivity,
            minimum_response,
        )
    }

    pub fn hessian_determinant(
        sigma: [f64; 3],
        truncate: f64,
        gamma: f64,
        boundary: Boundary,
        connectivity: Connectivity,
        minimum_response: f64,
    ) -> Result<Self> {
        Self::new(
            BlobResponse::hessian_determinant(sigma, truncate, gamma, boundary)?,
            connectivity,
            minimum_response,
        )
    }

    pub fn response(&self) -> &BlobResponse {
        &self.response
    }

    pub fn connectivity(&self) -> Connectivity {
        self.connectivity
    }

    pub fn minimum_response(&self) -> f64 {
        self.minimum_response
    }

    pub fn reach(&self, axis: usize) -> usize {
        self.response.reach(axis)
    }

    pub fn detect_points<T>(&self, input: ArrayView3<'_, T>) -> Result<Vec<Point>>
    where
        T: Copy + Into<f64>,
    {
        Ok(self
            .detect_features(input)?
            .into_iter()
            .map(BlobDetection::point)
            .collect())
    }

    pub fn detect_features<T>(&self, input: ArrayView3<'_, T>) -> Result<Vec<BlobDetection>>
    where
        T: Copy + Into<f64>,
    {
        let mut response = Array3::<f64>::zeros(input.raw_dim());
        self.response.response_into(input, response.view_mut())?;
        let scale = self.response.scale();
        Ok(
            response_peak_points(response.view(), self.connectivity, self.minimum_response)?
                .into_iter()
                .map(|point| BlobDetection {
                    at: point.at,
                    response: point.weight,
                    scale,
                })
                .collect(),
        )
    }
}

/// Build a bright-blob Difference-of-Gaussians response image.
///
/// The response is `G(low_sigma) - G(high_sigma)`: an impulse-like bright spot is
/// positive at its centre, and a constant image is exactly zero. Scale selection
/// and peak emission stay separate so callers can share this response with
/// `response_peak_points` or planner-visible point extraction later.
pub fn difference_of_gaussians_response_into<T>(
    input: ArrayView3<'_, T>,
    low_sigma: [f64; 3],
    high_sigma: [f64; 3],
    truncate: f64,
    boundary: Boundary,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    shapes_agree(
        input.shape(),
        out.shape(),
        "difference_of_gaussians_response_into",
    )?;
    let (low, high) = dog_gaussians(low_sigma, high_sigma, truncate)?;

    let shape = input.raw_dim();
    let mut low_smoothed = Array3::<f64>::zeros(shape);
    let mut high_smoothed = Array3::<f64>::zeros(shape);
    gaussian_smooth_into_with(input, low.kernels(), boundary, low_smoothed.view_mut())?;
    gaussian_smooth_into_with(input, high.kernels(), boundary, high_smoothed.view_mut())?;
    for ((slot, &lo), &hi) in out
        .iter_mut()
        .zip(low_smoothed.iter())
        .zip(high_smoothed.iter())
    {
        *slot = lo - hi;
    }
    Ok(())
}

fn dog_gaussians(
    low_sigma: [f64; 3],
    high_sigma: [f64; 3],
    truncate: f64,
) -> Result<(Gaussian, Gaussian)> {
    let low = Gaussian::new(low_sigma, truncate)?;
    let high = Gaussian::new(high_sigma, truncate)?;
    let mut wider = false;
    for axis in 0..3 {
        if high_sigma[axis] < low_sigma[axis] {
            return Err(Error::InvalidArgument(format!(
                "a DoG high scale must be at least the low scale on every axis; got low \
                 {low_sigma:?} and high {high_sigma:?}"
            )));
        }
        wider |= high_sigma[axis] > low_sigma[axis];
    }
    if !wider {
        return Err(Error::InvalidArgument(format!(
            "a DoG response needs at least one wider high scale; got {low_sigma:?}"
        )));
    }
    Ok((low, high))
}

fn gaussian_reach(sigma: f64, truncate: f64) -> usize {
    (sigma * truncate).ceil() as usize
}

/// Build a bright-blob Laplacian-of-Gaussian response image.
///
/// The smoothing is separable Gaussian; the Laplacian is the shared six-neighbour
/// kernel. The result is negated so bright compact peaks have positive response.
pub fn laplacian_of_gaussian_response_into<T>(
    input: ArrayView3<'_, T>,
    sigma: [f64; 3],
    truncate: f64,
    boundary: Boundary,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    shapes_agree(
        input.shape(),
        out.shape(),
        "laplacian_of_gaussian_response_into",
    )?;
    let gaussian = Gaussian::new(sigma, truncate)?;
    let shape = input.raw_dim();
    let mut smoothed = Array3::<f64>::zeros(shape);
    gaussian_smooth_into_with(input, gaussian.kernels(), boundary, smoothed.view_mut())?;
    convolve_into(
        smoothed.view(),
        &Kernel::laplace_6()?,
        Sense::Correlate,
        boundary,
        out.view_mut(),
    )?;
    for value in out.iter_mut() {
        *value = -*value;
    }
    Ok(())
}

/// Build a bright-blob determinant-of-Hessian response image.
///
/// The Hessian is taken after Gaussian smoothing and scale-normalised with
/// `sigma_axis.powf(gamma)` per differentiated axis. The determinant is negated
/// for 3-D data so a compact bright blob, whose Hessian is negative definite,
/// has a positive response.
pub fn hessian_determinant_response_into<T>(
    input: ArrayView3<'_, T>,
    sigma: [f64; 3],
    truncate: f64,
    gamma: f64,
    boundary: Boundary,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    shapes_agree(
        input.shape(),
        out.shape(),
        "hessian_determinant_response_into",
    )?;
    let gaussian = validate_hessian_scale(sigma, truncate, gamma)?;
    let shape = input.raw_dim();
    let mut smoothed = Array3::<f64>::zeros(shape);
    gaussian_smooth_into_with(input, gaussian.kernels(), boundary, smoothed.view_mut())?;
    let factor = [
        sigma[0].powf(gamma),
        sigma[1].powf(gamma),
        sigma[2].powf(gamma),
    ];
    for ((i, j, k), slot) in out.indexed_iter_mut() {
        let hessian = hessian_at(smoothed.view(), [i, j, k]);
        let scaled = [
            hessian[0] * factor[0] * factor[0],
            hessian[1] * factor[1] * factor[1],
            hessian[2] * factor[2] * factor[2],
            hessian[3] * factor[0] * factor[1],
            hessian[4] * factor[0] * factor[2],
            hessian[5] * factor[1] * factor[2],
        ];
        *slot = -hessian_determinant(scaled);
    }
    Ok(())
}

fn validate_hessian_scale(sigma: [f64; 3], truncate: f64, gamma: f64) -> Result<Gaussian> {
    if !gamma.is_finite() || gamma < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "a Hessian determinant scale normalisation exponent must be finite and \
             non-negative, got {gamma}"
        )));
    }
    Gaussian::new(sigma, truncate)
}

fn hessian_determinant(hessian: [f64; 6]) -> f64 {
    let [xx, yy, zz, xy, xz, yz] = hessian;
    xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
}

/// Emit one deterministic point for every regional maximum at or above
/// `minimum_response`.
///
/// The response image is local; the plateau rule is not. A flat-topped maximum
/// is one feature, so the representative is the lexicographically first voxel
/// in that plateau and the point weight is the plateau's response value.
pub fn response_peak_points(
    response: ArrayView3<'_, f64>,
    connectivity: Connectivity,
    minimum_response: f64,
) -> Result<Vec<Point>> {
    let maxima = regional_maxima_with(response, connectivity)?;
    let shape = [
        response.shape()[0],
        response.shape()[1],
        response.shape()[2],
    ];
    let mut seen = ndarray::Array3::<bool>::from_elem(response.raw_dim(), false);
    let mut points = Vec::new();
    let mut stack = Vec::new();

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let seed = [i, j, k];
                if !maxima[seed] || seen[seed] {
                    continue;
                }
                let value = response[seed];
                seen[seed] = true;
                stack.push(seed);
                let mut representative = seed;
                while let Some(at) = stack.pop() {
                    if at < representative {
                        representative = at;
                    }
                    for &by in connectivity.offsets() {
                        let Some(to) = offset_by(at, by, shape) else {
                            continue;
                        };
                        if seen[to] || !maxima[to] || response[to] != value {
                            continue;
                        }
                        seen[to] = true;
                        stack.push(to);
                    }
                }
                if value.is_finite() && value >= minimum_response {
                    points.push(Point::weighted(representative, value));
                }
            }
        }
    }
    Ok(points)
}

pub const RESPONSE_COLUMN: &str = "response";

pub fn response_peak_schema(response_column: impl Into<String>) -> Result<Schema> {
    Schema::new(vec![Column::f64(response_column)])
}

pub struct ResponsePeakRowsOp {
    name: &'static str,
    moments_stream: String,
    moments_phase: usize,
    rows_stream: String,
    lifecycle: Lifecycle,
    connectivity: Connectivity,
    response: ImageId,
    response_dtype: Dtype,
    minimum_response: f64,
    schema: Schema,
}

impl ResponsePeakRowsOp {
    pub fn new(
        name: &'static str,
        moments_stream: impl Into<String>,
        moments_phase: usize,
        rows_stream: impl Into<String>,
        lifecycle: Lifecycle,
        _grid: &BlockGrid,
        response: impl Into<ImageId>,
        response_dtype: Dtype,
        minimum_response: f64,
        response_column: impl Into<String>,
    ) -> Result<Self> {
        if !minimum_response.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "a response peak threshold must be finite, got {minimum_response}"
            )));
        }
        let response_column = response_column.into();
        let schema = response_peak_schema(response_column.clone())?;
        Ok(Self {
            name,
            moments_stream: moments_stream.into(),
            moments_phase,
            rows_stream: rows_stream.into(),
            lifecycle,
            connectivity: Connectivity::Faces,
            response: response.into(),
            response_dtype,
            minimum_response,
            schema,
        })
    }

    pub fn reading(
        name: &'static str,
        moments_stream: impl Into<String>,
        moments: Phase,
        rows_stream: impl Into<String>,
        lifecycle: Lifecycle,
        grid: &BlockGrid,
        response: impl Into<ImageId>,
        response_dtype: Dtype,
        minimum_response: f64,
        response_column: impl Into<String>,
    ) -> Result<Self> {
        Self::new(
            name,
            moments_stream,
            moments.index(),
            rows_stream,
            lifecycle,
            grid,
            response,
            response_dtype,
            minimum_response,
            response_column,
        )
    }

    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

impl FragmentOp for ResponsePeakRowsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn barrier(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![
            FragmentInput::own(self.moments_stream.clone(), self.moments_phase)
                .with_reach([0, 0, 0]),
        ]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.rows_stream.clone(),
            self.lifecycle,
            Coverage::EveryBlock,
        )
        .sized(SidecarSize::row_table(&self.schema, 1))]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.response).holding(self.response_dtype)]
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut reports = BTreeMap::new();
        for (key, bytes) in at.fragments(&self.moments_stream)? {
            reports.insert(key.block, RegionMoments::decode(&bytes)?);
        }
        let components =
            merge_moments_with(&reports, at.grid.blocks_per_axis(), self.connectivity)?;
        encode_moments(&components)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "response peak rows sample the response image at each representative and are applied \
             through `apply_with`."
                .to_string(),
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let response = sources.get(self.response.index())?.as_array()?;
        let components = super::detect::decode_moments(at.reduced)?;
        let mut rows = RowBuilder::new(Arc::new(self.schema.clone()));
        for component in peak_components_owned_by(&components, at.grid, at.index) {
            let Some(first) = component.first() else {
                continue;
            };
            let local = [
                first[0] - at.read.start[0],
                first[1] - at.read.start[1],
                first[2] - at.read.start[2],
            ];
            let response_value = value_at(response, local)?;
            if response_value.is_finite() && response_value >= self.minimum_response {
                rows.push(first, &[Value::F64(response_value)])?;
            }
        }
        Ok(BlockOutput::fragment(
            self.rows_stream.clone(),
            rows.encode(),
        ))
    }
}

fn peak_components_owned_by(
    components: &[Moments],
    grid: &BlockGrid,
    block: [usize; 3],
) -> Vec<Moments> {
    let mut owned: Vec<Moments> = components
        .iter()
        .filter(|moments| match moments.first() {
            None => false,
            Some(at) => owner_of(grid, at) == block,
        })
        .copied()
        .collect();
    owned.sort_by_key(|moments| moments.first());
    owned
}

pub fn append_response_peak_table_phases(
    plan: &mut PlanBuilder,
    rows_stream: impl Into<String>,
    rows_lifecycle: Lifecycle,
    connectivity: Connectivity,
    minimum_response: f64,
) -> Result<(Phase, Schema)> {
    let rows_stream = rows_stream.into();
    let response = ImageId::from(plan.n_phases());
    let response_dtype = plan.reads();
    let regional_stream = format!("{rows_stream}.plateaux");
    let moments_stream = format!("{rows_stream}.moments");
    append_regional_connected(
        plan,
        regional_stream,
        Lifecycle::DeleteOnExit,
        Dtype::Bool,
        connectivity,
    )?;
    let moments = plan.fragments(
        LabelRegionsOp::new(
            "response peak labelling",
            moments_stream.clone(),
            Lifecycle::DeleteOnExit,
        )
        .connecting(connectivity),
    )?;
    let op = ResponsePeakRowsOp::reading(
        "response peak rows",
        moments_stream,
        moments,
        rows_stream,
        rows_lifecycle,
        plan.grid(),
        response,
        response_dtype,
        minimum_response,
        RESPONSE_COLUMN,
    )?
    .connecting(connectivity);
    let schema = op.schema().clone();
    let phase = plan.fragments(op)?;
    Ok((phase, schema))
}

fn offset_by(at: [usize; 3], by: [isize; 3], shape: [usize; 3]) -> Option<[usize; 3]> {
    let mut out = [0usize; 3];
    for axis in 0..3 {
        let value = at[axis] as isize + by[axis];
        if value < 0 || value >= shape[axis] as isize {
            return None;
        }
        out[axis] = value as usize;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array3;

    #[test]
    fn difference_of_gaussians_annihilates_constants_and_marks_a_bright_spot() {
        let constant = Array3::<f64>::from_elem((5, 5, 5), 7.0);
        let mut response = Array3::<f64>::zeros((5, 5, 5));
        difference_of_gaussians_response_into(
            constant.view(),
            [0.5; 3],
            [1.0; 3],
            3.0,
            Boundary::Clamp,
            response.view_mut(),
        )
        .unwrap();
        assert!(response.iter().all(|value| value.abs() < 1e-12));

        let mut impulse = Array3::<f64>::zeros((7, 7, 7));
        impulse[[3, 3, 3]] = 1.0;
        let mut response = Array3::<f64>::zeros((7, 7, 7));
        difference_of_gaussians_response_into(
            impulse.view(),
            [0.5; 3],
            [1.2; 3],
            3.0,
            Boundary::Clamp,
            response.view_mut(),
        )
        .unwrap();
        assert!(response[[3, 3, 3]] > 0.0);
        assert_eq!(
            response_peak_points(
                response.view(),
                Connectivity::Faces,
                response[[3, 3, 3]] / 2.0
            )
            .unwrap(),
            vec![Point::weighted([3, 3, 3], response[[3, 3, 3]])]
        );
    }

    #[test]
    fn laplacian_of_gaussian_annihilates_constants_and_marks_a_bright_spot() {
        let constant = Array3::<f64>::from_elem((5, 5, 5), 7.0);
        let mut response = Array3::<f64>::zeros((5, 5, 5));
        laplacian_of_gaussian_response_into(
            constant.view(),
            [0.7; 3],
            3.0,
            Boundary::Clamp,
            response.view_mut(),
        )
        .unwrap();
        assert!(response.iter().all(|value| value.abs() < 1e-12));

        let mut impulse = Array3::<f64>::zeros((7, 7, 7));
        impulse[[3, 3, 3]] = 1.0;
        let mut response = Array3::<f64>::zeros((7, 7, 7));
        laplacian_of_gaussian_response_into(
            impulse.view(),
            [0.7; 3],
            3.0,
            Boundary::Clamp,
            response.view_mut(),
        )
        .unwrap();
        assert!(response[[3, 3, 3]] > 0.0);
    }

    #[test]
    fn hessian_determinant_annihilates_constants_and_marks_a_bright_spot() {
        let constant = Array3::<f64>::from_elem((5, 5, 5), 7.0);
        let mut response = Array3::<f64>::zeros((5, 5, 5));
        hessian_determinant_response_into(
            constant.view(),
            [0.8; 3],
            3.0,
            1.0,
            Boundary::Clamp,
            response.view_mut(),
        )
        .unwrap();
        assert!(response.iter().all(|value| value.abs() < 1e-12));

        let mut impulse = Array3::<f64>::zeros((7, 7, 7));
        impulse[[3, 3, 3]] = 1.0;
        let mut response = Array3::<f64>::zeros((7, 7, 7));
        hessian_determinant_response_into(
            impulse.view(),
            [0.8; 3],
            3.0,
            1.0,
            Boundary::Clamp,
            response.view_mut(),
        )
        .unwrap();
        assert!(response[[3, 3, 3]] > 0.0);
    }

    #[test]
    fn blob_responses_refuse_degenerate_scale_choices() {
        let image = Array3::<f64>::zeros((3, 3, 3));
        let mut response = Array3::<f64>::zeros((3, 3, 3));
        assert!(difference_of_gaussians_response_into(
            image.view(),
            [1.0; 3],
            [1.0; 3],
            3.0,
            Boundary::Clamp,
            response.view_mut(),
        )
        .is_err());
        assert!(difference_of_gaussians_response_into(
            image.view(),
            [1.0; 3],
            [0.5; 3],
            3.0,
            Boundary::Clamp,
            response.view_mut(),
        )
        .is_err());
        assert!(
            BlobResponse::hessian_determinant([1.0; 3], 3.0, f64::NAN, Boundary::Clamp).is_err()
        );
    }

    #[test]
    fn blob_detector_binds_response_threshold_and_peak_emission() {
        let mut image = Array3::<f64>::zeros((7, 7, 7));
        image[[3, 3, 3]] = 1.0;
        let detector = BlobDetector::difference_of_gaussians(
            [0.5; 3],
            [1.2; 3],
            3.0,
            Boundary::Clamp,
            Connectivity::Faces,
            0.01,
        )
        .unwrap();
        assert_eq!(detector.reach(0), 4);
        let points = detector.detect_points(image.view()).unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].at, [3, 3, 3]);
        assert!(points[0].weight >= detector.minimum_response());
        let features = detector.detect_features(image.view()).unwrap();
        assert_eq!(
            features,
            vec![BlobDetection {
                at: [3, 3, 3],
                response: points[0].weight,
                scale: BlobScale {
                    sigma: [0.5; 3],
                    upper_sigma: Some([1.2; 3]),
                },
            }]
        );

        let detector = BlobDetector::hessian_determinant(
            [0.8; 3],
            3.0,
            1.0,
            Boundary::Clamp,
            Connectivity::Faces,
            0.0001,
        )
        .unwrap();
        let points = detector.detect_points(image.view()).unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].at, [3, 3, 3]);
        let features = detector.detect_features(image.view()).unwrap();
        assert_eq!(features[0].scale.sigma, [0.8; 3]);
        assert_eq!(features[0].scale.upper_sigma, None);
    }

    #[test]
    fn blob_detector_refuses_non_finite_thresholds() {
        let response = BlobResponse::laplacian_of_gaussian([1.0; 3], 3.0, Boundary::Clamp).unwrap();
        assert!(BlobDetector::new(response, Connectivity::Faces, f64::NAN).is_err());
    }

    #[test]
    fn blob_response_op_declares_the_response_image_contract() {
        let op = BlobResponseOp::difference_of_gaussians(
            "blob.dog",
            [0.5; 3],
            [1.2; 3],
            3.0,
            Boundary::Clamp,
        )
        .unwrap();
        assert!(op.accepts(Dtype::U16));
        assert!(!op.accepts(Dtype::F16));
        assert_eq!(op.produces(Dtype::U16), Dtype::F64);
        assert_eq!(op.reach(0, 100), 4);
        assert_eq!(op.constant_maps_to(0.0), Some(0.0));
        assert_eq!(op.constant_maps_to(1.0), None);
        assert!(op.cost_per_voxel() > 0.0);

        let doh =
            BlobResponseOp::hessian_determinant("blob.doh", [0.8; 3], 3.0, 1.0, Boundary::Clamp)
                .unwrap();
        assert_eq!(doh.reach(0, 100), 4);
        assert!(doh.cost_per_voxel() > op.cost_per_voxel() / 2.0);
    }

    #[test]
    fn blob_response_op_apply_matches_the_free_response_function() {
        let mut image = Array3::<u16>::zeros((7, 7, 7));
        image[[3, 3, 3]] = 10;
        let op = BlobResponseOp::laplacian_of_gaussian("blob.log", [0.7; 3], 3.0, Boundary::Clamp)
            .unwrap();
        let voxels: Voxels = image.clone().into();
        let mut via_op = Voxels::zeros(Dtype::F64, [7, 7, 7]).unwrap();
        op.apply(&voxels, &mut via_op, &Anchor::whole([7, 7, 7]))
            .unwrap();

        let mut expected = Array3::<f64>::zeros((7, 7, 7));
        laplacian_of_gaussian_response_into(
            image.view(),
            [0.7; 3],
            3.0,
            Boundary::Clamp,
            expected.view_mut(),
        )
        .unwrap();
        assert_eq!(via_op.view::<f64>().unwrap(), expected.view());
    }

    #[test]
    fn emits_one_point_per_regional_peak_plateau() {
        let mut response = Array3::<f64>::zeros((5, 5, 5));
        response[[2, 2, 1]] = 9.0;
        response[[2, 2, 2]] = 9.0;
        response[[2, 2, 3]] = 9.0;
        response[[0, 0, 0]] = 7.0;

        let points = response_peak_points(response.view(), Connectivity::Faces, 1.0).unwrap();
        assert_eq!(
            points,
            vec![
                Point::weighted([0, 0, 0], 7.0),
                Point::weighted([2, 2, 1], 9.0)
            ]
        );
    }

    #[test]
    fn applies_threshold_and_connectivity_to_peak_emission() {
        let mut response = Array3::<f64>::zeros((4, 4, 4));
        response[[1, 1, 1]] = 3.0;
        response[[2, 2, 2]] = 3.0;
        response[[0, 3, 0]] = 5.0;

        let six = response_peak_points(response.view(), Connectivity::Faces, 4.0).unwrap();
        assert_eq!(six, vec![Point::weighted([0, 3, 0], 5.0)]);

        let full =
            response_peak_points(response.view(), Connectivity::FacesEdgesAndCorners, 1.0).unwrap();
        assert_eq!(
            full,
            vec![
                Point::weighted([0, 3, 0], 5.0),
                Point::weighted([1, 1, 1], 3.0),
            ]
        );
    }
}
