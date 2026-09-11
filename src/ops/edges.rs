// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Edge-linking helpers shared by Canny-style pipelines.

use std::collections::BTreeMap;

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::assemble::{Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::env::BlockBuf;
use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, PhaseView,
    SeamFold, SidecarSize, SourceBlocks,
};
use crate::op::{Anchor, BlockOp, Chain, Slicing, SourceInput};
use crate::reach::Reach;
use crate::sidecar::Lifecycle;
use crate::voxels::Voxels;

use super::components::{
    bytes_to_words, decode_block_flags_for, encode_block_flags, expect_end, planes_of, push_planes,
    read_header, take_planes, walk_seams_with, words_to_bytes, Connectivity, FacePlanes,
    LabelIndex, Union,
};
use super::detect::label_regions_into_with;
use super::ridge::gaussian_radius;
use super::shapes_agree;
use super::smooth::SMOOTH_COST_PER_TAP;
use super::structure_tensor::gaussian_gradient_into;

/// A Canny-style edge detector over a scalar image.
///
/// This is a named composition over the general pieces in this module:
/// Gaussian gradient components, gradient magnitude, non-maximum suppression,
/// and hysteresis thresholding. It deliberately does not add a new cost family;
/// a planner-visible form should price the same stages it expands to.
pub fn canny_edges_into<T>(
    input: ArrayView3<'_, T>,
    sigma: [f64; 3],
    truncate: f64,
    low: f64,
    high: f64,
    connectivity: Connectivity,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    let dim = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let mut thinned = Array3::<f64>::zeros(dim);
    canny_response_into(input, sigma, truncate, thinned.view_mut())?;
    hysteresis_threshold_into(thinned.view(), low, high, connectivity, out)
}

/// Compute the local Canny response image before double-threshold edge linking.
pub fn canny_response_into<T>(
    input: ArrayView3<'_, T>,
    sigma: [f64; 3],
    truncate: f64,
    out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: Copy + Into<f64>,
{
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    shapes_agree(input.shape(), out.shape(), "canny_response_into")?;
    validate_canny_scale(sigma, truncate)?;
    let dim = (shape[0], shape[1], shape[2]);
    let mut gradient = [
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
        Array3::<f64>::zeros(dim),
    ];
    {
        let [x, y, z] = &mut gradient;
        gaussian_gradient_into(
            input,
            sigma,
            truncate,
            [x.view_mut(), y.view_mut(), z.view_mut()],
        )?;
    }

    let mut magnitude = Array3::<f64>::zeros(dim);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                magnitude[at] = (gradient[0][at] * gradient[0][at]
                    + gradient[1][at] * gradient[1][at]
                    + gradient[2][at] * gradient[2][at])
                    .sqrt();
            }
        }
    }

    non_maximum_suppression_into(
        magnitude.view(),
        [gradient[0].view(), gradient[1].view(), gradient[2].view()],
        out,
    )
}

pub struct CannyResponseOp {
    name: &'static str,
    sigma: [f64; 3],
    truncate: f64,
    cost: f64,
}

impl CannyResponseOp {
    pub fn new(name: &'static str, sigma: [f64; 3], truncate: f64) -> Result<Self> {
        validate_canny_scale(sigma, truncate)?;
        Ok(Self {
            name,
            sigma,
            truncate,
            cost: canny_response_cost(sigma, truncate),
        })
    }

    pub fn sigma(&self) -> [f64; 3] {
        self.sigma
    }

    pub fn truncate(&self) -> f64 {
        self.truncate
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    pub fn reach_on(&self, axis: usize) -> usize {
        gaussian_radius(self.sigma[axis], self.truncate) + 2
    }
}

impl BlockOp for CannyResponseOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.reach_on(axis)
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::symmetric([self.reach_on(0), self.reach_on(1), self.reach_on(2)])
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
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
            |view| { canny_response_into(view, self.sigma, self.truncate, out) }
        )
    }

    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        Some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

pub fn append_canny_edges_phases(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    sigma: [f64; 3],
    truncate: f64,
    low: f64,
    high: f64,
    connectivity: Connectivity,
) -> Result<Phase> {
    validate_hysteresis_thresholds(low, high)?;
    plan.pixels(Chain::op(CannyResponseOp::new(
        "canny local response",
        sigma,
        truncate,
    )?))?;
    append_hysteresis_threshold_phases(plan, stream, lifecycle, low, high, connectivity)
}

fn validate_canny_scale(sigma: [f64; 3], truncate: f64) -> Result<()> {
    if !truncate.is_finite() || truncate <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "canny_response_into needs a positive finite truncation, got {truncate}"
        )));
    }
    for (axis, &scale) in sigma.iter().enumerate() {
        if !scale.is_finite() || scale < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "canny_response_into needs finite non-negative sigma values; axis {axis} is \
                 {scale}"
            )));
        }
    }
    Ok(())
}

fn canny_response_cost(sigma: [f64; 3], truncate: f64) -> f64 {
    let taps: usize = sigma
        .iter()
        .map(|&scale| 2 * gaussian_radius(scale, truncate) + 1)
        .sum();
    SMOOTH_COST_PER_TAP * taps as f64 + CANNY_RESPONSE_VOXEL_COST
}

const CANNY_RESPONSE_VOXEL_COST: f64 = 62.0;

/// Suppress a response unless it is a local maximum along its gradient
/// direction.
///
/// The comparison samples the magnitude image one unit forward and backward in
/// the normalised gradient direction by trilinear interpolation. Boundary
/// voxels whose comparison point leaves the array are suppressed rather than
/// compared against an invented value.
pub fn non_maximum_suppression_into(
    magnitude: ArrayView3<'_, f64>,
    gradient: [ArrayView3<'_, f64>; 3],
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    shapes_agree(
        magnitude.shape(),
        out.shape(),
        "non_maximum_suppression_into",
    )?;
    for component in &gradient {
        shapes_agree(
            magnitude.shape(),
            component.shape(),
            "non_maximum_suppression_into",
        )?;
    }

    out.fill(0.0);
    let shape = [
        magnitude.shape()[0],
        magnitude.shape()[1],
        magnitude.shape()[2],
    ];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                let value = magnitude[at];
                if !value.is_finite() || value <= 0.0 {
                    continue;
                }
                let direction = [gradient[0][at], gradient[1][at], gradient[2][at]];
                let norm = (direction[0] * direction[0]
                    + direction[1] * direction[1]
                    + direction[2] * direction[2])
                    .sqrt();
                if !norm.is_finite() || norm == 0.0 {
                    continue;
                }
                let unit = [
                    direction[0] / norm,
                    direction[1] / norm,
                    direction[2] / norm,
                ];
                let here = [i as f64, j as f64, k as f64];
                let forward = [here[0] + unit[0], here[1] + unit[1], here[2] + unit[2]];
                let backward = [here[0] - unit[0], here[1] - unit[1], here[2] - unit[2]];
                let Some(ahead) = trilinear_at(magnitude, forward) else {
                    continue;
                };
                let Some(behind) = trilinear_at(magnitude, backward) else {
                    continue;
                };
                if value >= ahead && value >= behind {
                    out[at] = value;
                }
            }
        }
    }
    Ok(())
}

fn trilinear_at(values: ArrayView3<'_, f64>, at: [f64; 3]) -> Option<f64> {
    let shape = [values.shape()[0], values.shape()[1], values.shape()[2]];
    for axis in 0..3 {
        if at[axis] < 0.0 || at[axis] > (shape[axis] - 1) as f64 {
            return None;
        }
    }

    let low = [
        at[0].floor() as usize,
        at[1].floor() as usize,
        at[2].floor() as usize,
    ];
    let high = [
        (low[0] + 1).min(shape[0] - 1),
        (low[1] + 1).min(shape[1] - 1),
        (low[2] + 1).min(shape[2] - 1),
    ];
    let fraction = [
        at[0] - low[0] as f64,
        at[1] - low[1] as f64,
        at[2] - low[2] as f64,
    ];

    let mut out = 0.0;
    for da in 0..=1 {
        for db in 0..=1 {
            for dc in 0..=1 {
                let index = [
                    if da == 0 { low[0] } else { high[0] },
                    if db == 0 { low[1] } else { high[1] },
                    if dc == 0 { low[2] } else { high[2] },
                ];
                let weight = if da == 0 {
                    1.0 - fraction[0]
                } else {
                    fraction[0]
                } * if db == 0 {
                    1.0 - fraction[1]
                } else {
                    fraction[1]
                } * if dc == 0 {
                    1.0 - fraction[2]
                } else {
                    fraction[2]
                };
                out += weight * values[index];
            }
        }
    }
    Some(out)
}

/// Keep every weak edge connected to at least one strong edge.
///
/// `high` seeds the traversal and `low` defines the passable weak-edge mask.
/// The output is true for strong edges and for weak edges reached from them.
pub fn hysteresis_threshold_into(
    response: ArrayView3<'_, f64>,
    low: f64,
    high: f64,
    connectivity: Connectivity,
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(response.shape(), out.shape(), "hysteresis_threshold_into")?;
    if !low.is_finite() || !high.is_finite() || low > high {
        return Err(Error::InvalidArgument(format!(
            "hysteresis thresholds must be finite and ordered low <= high; got low={low}, \
             high={high}"
        )));
    }

    out.fill(false);
    let shape = [
        response.shape()[0],
        response.shape()[1],
        response.shape()[2],
    ];
    let mut stack = Vec::new();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                if response[at] >= high && !out[at] {
                    out[at] = true;
                    stack.push(at);
                    while let Some(here) = stack.pop() {
                        for &by in connectivity.offsets() {
                            let Some(to) = offset_by(here, by, shape) else {
                                continue;
                            };
                            if out[to] || response[to] < low {
                                continue;
                            }
                            out[to] = true;
                            stack.push(to);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

const HYSTERESIS_REPORT_MAGIC: u32 = 0x5359_4845;
const HYSTERESIS_REPORT_VERSION: u32 = 1;
const HYSTERESIS_FLAGS_MAGIC: u32 = 0x4648_5945;
const HYSTERESIS_NOUN: &str = "a hysteresis reduction";

#[derive(Debug, Clone, PartialEq, Eq)]
struct HysteresisReport {
    labels: u32,
    strong: Vec<bool>,
    faces: FacePlanes,
}

impl HysteresisReport {
    fn empty() -> Self {
        Self {
            labels: 0,
            strong: Vec::new(),
            faces: super::components::empty_planes(),
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut words = vec![
            HYSTERESIS_REPORT_MAGIC,
            HYSTERESIS_REPORT_VERSION,
            self.labels,
        ];
        words.extend(self.strong.iter().map(|&flag| u32::from(flag)));
        push_planes(&self.faces, &mut words);
        words_to_bytes(&words)
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        const NOUN: &str = "a hysteresis weak-component fragment";
        let words = bytes_to_words(bytes, NOUN)?;
        let labels = read_header(
            &words,
            HYSTERESIS_REPORT_MAGIC,
            HYSTERESIS_REPORT_VERSION,
            NOUN,
        )?;
        let labels_len = labels as usize;
        let mut at = 3usize;
        if words.len() < at + labels_len {
            return Err(Error::InvalidArgument(format!(
                "{NOUN} ends inside its strong-label flags"
            )));
        }
        let mut strong = Vec::with_capacity(labels_len);
        for &word in &words[at..at + labels_len] {
            match word {
                0 => strong.push(false),
                1 => strong.push(true),
                _ => {
                    return Err(Error::InvalidArgument(format!(
                        "{NOUN} holds a strong-label flag {word}, not 0 or 1"
                    )));
                }
            }
        }
        at += labels_len;
        let faces = take_planes(&words, &mut at, NOUN)?;
        expect_end(&words, at, NOUN)?;
        Ok(Self {
            labels,
            strong,
            faces,
        })
    }
}

pub struct HysteresisLabelsOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    low: f64,
    high: f64,
    connectivity: Connectivity,
}

impl HysteresisLabelsOp {
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        lifecycle: Lifecycle,
        low: f64,
        high: f64,
    ) -> Result<Self> {
        validate_hysteresis_thresholds(low, high)?;
        Ok(Self {
            name,
            stream: stream.into(),
            lifecycle,
            low,
            high,
            connectivity: Connectivity::Faces,
        })
    }

    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }
}

impl FragmentOp for HysteresisLabelsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![
            FragmentOutput::new(self.stream.clone(), self.lifecycle, Coverage::EveryBlock)
                .sized(SidecarSize::component_report()),
        ]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = at.pixels()? else {
            return Ok(BlockOutput::fragment(
                self.stream.clone(),
                HysteresisReport::empty().encode(),
            ));
        };
        let response = pixels.view::<f64>()?;
        let weak = response.mapv(|value| value >= self.low);
        let mut labels = Array3::<u32>::zeros(response.raw_dim());
        let count = label_regions_into_with(weak.view(), self.connectivity, labels.view_mut())?;
        let mut strong = vec![false; count as usize];
        for (at, &label) in labels.indexed_iter() {
            if label != 0 && response[at] >= self.high {
                strong[label as usize - 1] = true;
            }
        }
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            HysteresisReport {
                labels: count,
                strong,
                faces: planes_of(labels.view()),
            }
            .encode(),
        ))
    }
}

pub struct ApplyHysteresisOp {
    name: &'static str,
    stream: String,
    phase: usize,
    low: f64,
    lattice: [usize; 3],
    connectivity: Connectivity,
    response: crate::assemble::ImageId,
    response_dtype: Dtype,
}

impl ApplyHysteresisOp {
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        phase: impl Into<Phase>,
        low: f64,
        high: f64,
        lattice: [usize; 3],
        response: impl Into<crate::assemble::ImageId>,
        response_dtype: Dtype,
    ) -> Result<Self> {
        validate_hysteresis_thresholds(low, high)?;
        Ok(Self {
            name,
            stream: stream.into(),
            phase: phase.into().index(),
            low,
            lattice,
            connectivity: Connectivity::Faces,
            response: response.into(),
            response_dtype,
        })
    }

    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }
}

impl FragmentOp for ApplyHysteresisOp {
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
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn barrier(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.phase).with_reach([0, 0, 0])]
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
        for (key, bytes) in at.fragments(&self.stream)? {
            reports.insert(key.block, HysteresisReport::decode(&bytes)?);
        }
        let index = LabelIndex::build(&reports, at.grid.blocks_per_axis(), |report| report.labels)?;
        let strong = index.gather(&reports, |report| &report.strong, false);
        let mut sets = Union::new(index.total());
        walk_seams_with(
            &reports,
            at.grid.blocks_per_axis(),
            &index,
            self.connectivity,
            |report| &report.faces,
            |a, b| sets.union(a, b),
        )?;
        let mut root_strong = vec![false; index.total()];
        for (node, &flag) in strong.iter().enumerate() {
            if flag {
                let root = sets.find(node);
                root_strong[root] = true;
            }
        }
        let keep = index.per_block(&mut sets, &root_strong);
        encode_block_flags(&keep, at.grid.blocks_per_axis(), HYSTERESIS_FLAGS_MAGIC)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "hysteresis edge linking reads the response image as a declared source input and \
             is applied through `apply_with`."
                .to_string(),
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let response = sources
            .get(self.response.index())?
            .as_array()?
            .view::<f64>()?;
        let weak = response.mapv(|value| value >= self.low);
        let mut labels = Array3::<u32>::zeros(response.raw_dim());
        label_regions_into_with(weak.view(), self.connectivity, labels.view_mut())?;
        let keep = decode_block_flags_for(
            at.reduced,
            self.lattice,
            at.index,
            HYSTERESIS_FLAGS_MAGIC,
            HYSTERESIS_NOUN,
        )?;
        let mut out = Voxels::zeros(
            Dtype::Bool,
            [labels.shape()[0], labels.shape()[1], labels.shape()[2]],
        )?;
        let mut mask = out.view_mut::<bool>()?;
        for (at, slot) in mask.indexed_iter_mut() {
            let label = labels[at];
            *slot = label != 0 && keep.get(label as usize - 1).copied().unwrap_or(false);
        }
        Ok(BlockOutput::nothing().with_pixels(BlockBuf::Array(out)))
    }
}

pub fn append_hysteresis_threshold_phases(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    low: f64,
    high: f64,
    connectivity: Connectivity,
) -> Result<Phase> {
    let stream = stream.into();
    let response = crate::assemble::ImageId::from(plan.n_phases());
    let response_dtype = plan.reads();
    let labels = plan.fragments(
        HysteresisLabelsOp::new(
            "hysteresis weak-component labelling",
            stream.clone(),
            lifecycle,
            low,
            high,
        )?
        .connecting(connectivity),
    )?;
    let lattice = plan.grid().blocks_per_axis();
    plan.fragments(
        ApplyHysteresisOp::new(
            "hysteresis edge linking",
            stream,
            labels,
            low,
            high,
            lattice,
            response,
            response_dtype,
        )?
        .connecting(connectivity),
    )
}

fn validate_hysteresis_thresholds(low: f64, high: f64) -> Result<()> {
    if !low.is_finite() || !high.is_finite() || low > high {
        return Err(Error::InvalidArgument(format!(
            "hysteresis thresholds must be finite and ordered low <= high; got low={low}, \
             high={high}"
        )));
    }
    Ok(())
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
    fn canny_without_smoothing_has_a_small_exact_step_fixture() {
        let input = Array3::from_shape_fn((1, 5, 1), |(_, j, _)| if j < 2 { 0.0 } else { 10.0 });
        let mut out = Array3::<bool>::default((1, 5, 1));

        canny_edges_into(
            input.view(),
            [0.0, 0.0, 0.0],
            3.0,
            4.0,
            4.0,
            Connectivity::Faces,
            out.view_mut(),
        )
        .unwrap();

        assert_eq!(
            out.iter().copied().collect::<Vec<_>>(),
            vec![false, true, true, false, false]
        );
    }

    #[test]
    fn canny_is_the_named_composition_of_gradient_thinning_and_hysteresis() {
        let input = Array3::from_shape_fn((1, 7, 1), |(_, j, _)| (j as f64 - 3.0).abs());
        let mut direct = Array3::<bool>::default((1, 7, 1));
        canny_edges_into(
            input.view(),
            [0.0, 0.0, 0.0],
            3.0,
            0.5,
            1.0,
            Connectivity::Faces,
            direct.view_mut(),
        )
        .unwrap();

        let mut gradient = [
            Array3::<f64>::zeros((1, 7, 1)),
            Array3::<f64>::zeros((1, 7, 1)),
            Array3::<f64>::zeros((1, 7, 1)),
        ];
        {
            let [x, y, z] = &mut gradient;
            gaussian_gradient_into(
                input.view(),
                [0.0, 0.0, 0.0],
                3.0,
                [x.view_mut(), y.view_mut(), z.view_mut()],
            )
            .unwrap();
        }
        let mut magnitude = Array3::<f64>::zeros((1, 7, 1));
        for j in 0..7 {
            let at = [0, j, 0];
            magnitude[at] = (gradient[0][at] * gradient[0][at]
                + gradient[1][at] * gradient[1][at]
                + gradient[2][at] * gradient[2][at])
                .sqrt();
        }
        let mut thinned = Array3::<f64>::zeros((1, 7, 1));
        non_maximum_suppression_into(
            magnitude.view(),
            [gradient[0].view(), gradient[1].view(), gradient[2].view()],
            thinned.view_mut(),
        )
        .unwrap();
        let mut composed = Array3::<bool>::default((1, 7, 1));
        hysteresis_threshold_into(
            thinned.view(),
            0.5,
            1.0,
            Connectivity::Faces,
            composed.view_mut(),
        )
        .unwrap();

        assert_eq!(direct, composed);
    }

    #[test]
    fn non_maximum_suppression_keeps_only_peaks_along_the_gradient() {
        let mut magnitude = Array3::<f64>::zeros((1, 5, 1));
        magnitude[[0, 1, 0]] = 1.0;
        magnitude[[0, 2, 0]] = 3.0;
        magnitude[[0, 3, 0]] = 2.0;
        let gx = Array3::<f64>::zeros((1, 5, 1));
        let gy = Array3::<f64>::from_elem((1, 5, 1), 1.0);
        let gz = Array3::<f64>::zeros((1, 5, 1));
        let mut out = Array3::<f64>::zeros((1, 5, 1));

        non_maximum_suppression_into(
            magnitude.view(),
            [gx.view(), gy.view(), gz.view()],
            out.view_mut(),
        )
        .unwrap();

        assert_eq!(
            out.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 0.0, 3.0, 0.0, 0.0]
        );
    }

    #[test]
    fn non_maximum_suppression_interpolates_diagonal_comparisons() {
        let mut magnitude = Array3::<f64>::zeros((5, 5, 1));
        magnitude[[2, 2, 0]] = 10.0;
        magnitude[[1, 1, 0]] = 9.0;
        magnitude[[3, 3, 0]] = 9.0;
        magnitude[[1, 3, 0]] = 11.0;
        let gx = Array3::<f64>::from_elem((5, 5, 1), 1.0);
        let gy = Array3::<f64>::from_elem((5, 5, 1), 1.0);
        let gz = Array3::<f64>::zeros((5, 5, 1));
        let mut out = Array3::<f64>::zeros((5, 5, 1));

        non_maximum_suppression_into(
            magnitude.view(),
            [gx.view(), gy.view(), gz.view()],
            out.view_mut(),
        )
        .unwrap();

        assert_eq!(out[[2, 2, 0]], 10.0);
        assert_eq!(out[[1, 3, 0]], 11.0);

        magnitude[[3, 3, 0]] = 30.0;
        non_maximum_suppression_into(
            magnitude.view(),
            [gx.view(), gy.view(), gz.view()],
            out.view_mut(),
        )
        .unwrap();
        assert_eq!(
            out[[2, 2, 0]],
            0.0,
            "a larger response on the gradient line must suppress the centre"
        );
    }

    #[test]
    fn hysteresis_keeps_weak_edges_connected_to_strong_edges() {
        let mut response = Array3::<f64>::zeros((1, 7, 1));
        response[[0, 1, 0]] = 0.8;
        response[[0, 2, 0]] = 0.4;
        response[[0, 3, 0]] = 0.35;
        response[[0, 5, 0]] = 0.4;
        let mut out = Array3::<bool>::default((1, 7, 1));

        hysteresis_threshold_into(
            response.view(),
            0.3,
            0.7,
            Connectivity::Faces,
            out.view_mut(),
        )
        .unwrap();

        assert_eq!(
            out.iter().copied().collect::<Vec<_>>(),
            vec![false, true, true, true, false, false, false]
        );
    }

    #[test]
    fn hysteresis_honours_the_connectivity_relation() {
        let mut response = Array3::<f64>::zeros((3, 3, 3));
        response[[1, 1, 1]] = 1.0;
        response[[2, 2, 2]] = 0.5;
        let mut six = Array3::<bool>::default((3, 3, 3));
        let mut full = Array3::<bool>::default((3, 3, 3));

        hysteresis_threshold_into(
            response.view(),
            0.4,
            0.9,
            Connectivity::Faces,
            six.view_mut(),
        )
        .unwrap();
        hysteresis_threshold_into(
            response.view(),
            0.4,
            0.9,
            Connectivity::FacesEdgesAndCorners,
            full.view_mut(),
        )
        .unwrap();

        assert!(six[[1, 1, 1]]);
        assert!(!six[[2, 2, 2]]);
        assert!(full[[1, 1, 1]]);
        assert!(full[[2, 2, 2]]);
        assert!(hysteresis_threshold_into(
            response.view(),
            2.0,
            1.0,
            Connectivity::Faces,
            six.view_mut()
        )
        .is_err());
    }
}
