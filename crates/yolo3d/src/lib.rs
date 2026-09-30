//! A separate 3D detector derived from the successful 2D microscopy YOLO
//! workflow. It uses a center heatmap because dense, mostly single-class cells
//! do not benefit from emitting six box-distance distributions at every voxel.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv3d, Conv3dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, PaddingConfig3d};
use burn::prelude::{Device, Tensor};
use burn::tensor::activation::{sigmoid, silu};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectorConfig {
    pub input_channels: usize,
    pub classes: usize,
    pub base_channels: usize,
    /// Strides for P2, P3, and P4. Keep z stride one until physical feature
    /// spacing is near the y/x spacing.
    pub strides: [[usize; 3]; 3],
    pub voxel_size: [f32; 3],
    /// Dataset median physical z/y/x extent for objects assigned to each head.
    #[serde(default = "default_size_priors")]
    pub size_priors: [[f32; 3]; 3],
}

fn default_size_priors() -> [[f32; 3]; 3] {
    [[1.0; 3]; 3]
}

impl DetectorConfig {
    pub fn for_spacing(input_channels: usize, classes: usize, voxel_size: [f32; 3]) -> Self {
        let z_to_xy = voxel_size[0] / voxel_size[1].min(voxel_size[2]).max(f32::EPSILON);
        let z_strides = if z_to_xy >= 3.0 {
            [1, 1, 2]
        } else if z_to_xy >= 1.5 {
            [1, 2, 2]
        } else {
            [2, 2, 2]
        };
        Self {
            input_channels,
            classes,
            base_channels: 32,
            strides: [
                [z_strides[0], 2, 2],
                [z_strides[1], 2, 2],
                [z_strides[2], 2, 2],
            ],
            voxel_size,
            size_priors: default_size_priors(),
        }
    }

    pub fn validate(&self) -> Result<(), Error> {
        if self.input_channels == 0 || self.classes == 0 || self.base_channels < 8 {
            return Err(Error::Config(
                "channels and classes must be positive".into(),
            ));
        }
        if self
            .voxel_size
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
            || self.strides.iter().flatten().any(|stride| *stride == 0)
            || self
                .size_priors
                .iter()
                .flatten()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::Config(
                "voxel sizes and strides must be positive".into(),
            ));
        }
        Ok(())
    }

    pub fn cumulative_strides(&self) -> [[usize; 3]; 3] {
        let mut cumulative = [1usize; 3];
        self.strides.map(|stride| {
            for axis in 0..3 {
                cumulative[axis] *= stride[axis];
            }
            cumulative
        })
    }
}

#[derive(Module, Debug)]
struct Block3d {
    conv: Conv3d,
    norm: GroupNorm,
}

impl Block3d {
    fn new(
        input: usize,
        output: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
        device: &Device,
    ) -> Self {
        let groups = (1..=8)
            .rev()
            .find(|groups| output.is_multiple_of(*groups))
            .unwrap_or(1);
        Self {
            conv: Conv3dConfig::new([input, output], kernel)
                .with_stride(stride)
                .with_padding(PaddingConfig3d::Explicit(
                    kernel[0] / 2,
                    kernel[1] / 2,
                    kernel[2] / 2,
                ))
                .with_bias(false)
                .init(device),
            norm: GroupNormConfig::new(groups, output).init(device),
        }
    }

    fn forward(&self, input: Tensor<5>) -> Tensor<5> {
        silu(self.norm.forward(self.conv.forward(input)))
    }
}

#[derive(Module, Debug)]
struct CenterHead {
    heatmap: Conv3d,
    offset: Conv3d,
    size: Conv3d,
    quality: Conv3d,
}

impl CenterHead {
    fn new(channels: usize, classes: usize, size_prior: [f32; 3], device: &Device) -> Self {
        let conv = |output| {
            Conv3dConfig::new([channels, output], [1, 1, 1])
                .with_padding(PaddingConfig3d::Valid)
                .init(device)
        };
        let mut heatmap = conv(classes);
        heatmap.bias = Some(Param::from_tensor(Tensor::full([classes], -2.19, device)));
        let mut quality = conv(1);
        quality.bias = Some(Param::from_tensor(Tensor::full([1], -2.19, device)));
        let mut size = conv(3);
        size.bias = Some(Param::from_tensor(Tensor::from_floats(
            size_prior.map(f32::ln).as_slice(),
            device,
        )));
        Self {
            heatmap,
            offset: conv(3),
            size,
            quality,
        }
    }

    fn forward(&self, features: Tensor<5>) -> ScaleOutput {
        ScaleOutput {
            heatmap: sigmoid(self.heatmap.forward(features.clone())),
            offset: sigmoid(self.offset.forward(features.clone())),
            size: self.size.forward(features.clone()).exp(),
            quality: sigmoid(self.quality.forward(features)),
        }
    }
}

/// Three-scale anisotropy-aware backbone with one center head per scale.
#[derive(Module, Debug)]
pub struct Detector {
    stem: Block3d,
    stages: Vec<Block3d>,
    heads: Vec<CenterHead>,
}

#[derive(Clone, Debug)]
pub struct ScaleOutput {
    pub heatmap: Tensor<5>,
    pub offset: Tensor<5>,
    /// Positive z/y/x extents in feature-grid units.
    pub size: Tensor<5>,
    pub quality: Tensor<5>,
}

pub struct ScaleTargets {
    pub heatmap: Tensor<5>,
    pub center_weight: Tensor<5>,
    pub offset: Tensor<5>,
    pub size: Tensor<5>,
    pub regression_weight: Tensor<5>,
}

pub struct LossOutput {
    pub total: Tensor<1>,
    pub heatmap: Tensor<1>,
    pub offset: Tensor<1>,
    pub size: Tensor<1>,
    pub quality: Tensor<1>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LossWeights {
    pub heatmap: f64,
    pub offset: f64,
    pub size: f64,
    pub quality: f64,
}

/// Dataset policy rather than a model default. Fluorescence can usually use
/// XY dihedral transforms; a fixed DIC or phase-contrast lens configuration
/// generally cannot use rotations or reflections.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AugmentationPolicy {
    pub xy_right_angle_rotations: bool,
    pub xy_reflections: bool,
    pub z_reflection: bool,
    pub arbitrary_3d_rotation: bool,
    pub intensity_scale: [f32; 2],
    pub gamma: [f32; 2],
    pub poisson_noise: bool,
    pub axial_blur: bool,
    pub slice_dropout: bool,
}

impl AugmentationPolicy {
    pub fn fluorescence() -> Self {
        Self {
            xy_right_angle_rotations: true,
            xy_reflections: true,
            z_reflection: true,
            arbitrary_3d_rotation: false,
            intensity_scale: [0.8, 1.2],
            gamma: [0.8, 1.2],
            poisson_noise: true,
            axial_blur: true,
            slice_dropout: true,
        }
    }

    pub fn fixed_transmitted_light() -> Self {
        Self {
            xy_right_angle_rotations: false,
            xy_reflections: false,
            z_reflection: false,
            arbitrary_3d_rotation: false,
            intensity_scale: [0.8, 1.2],
            gamma: [0.8, 1.2],
            poisson_noise: true,
            axial_blur: false,
            slice_dropout: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryRow {
    pub epoch: usize,
    pub train_loss: f32,
    pub validation_loss: f32,
    pub center_ap: Option<f32>,
    pub box_ap50: Option<f32>,
    pub learning_rate: f64,
    pub elapsed_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Hash or stable serialized identity of the model and target geometry.
    pub run_contract: String,
    pub rows: Vec<HistoryRow>,
}

impl TrainingHistory {
    pub fn append(&mut self, continuation: Self) -> Result<(), Error> {
        if self.run_contract != continuation.run_contract {
            return Err(Error::Config(
                "training histories have different model or dataset contracts".into(),
            ));
        }
        let next = self.rows.last().map_or(0, |row| row.epoch + 1);
        if continuation
            .rows
            .first()
            .is_some_and(|row| row.epoch != next)
        {
            return Err(Error::Config(format!(
                "continued history starts at epoch {}, expected {next}",
                continuation.rows[0].epoch
            )));
        }
        self.rows.extend(continuation.rows);
        Ok(())
    }
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            heatmap: 1.0,
            offset: 1.0,
            size: 0.1,
            quality: 0.25,
        }
    }
}

/// CenterNet-style heatmap focal loss plus weighted L1 center and size losses.
/// Teacher confidence weights every regression target and supplies the quality
/// target, so uncertain pseudo-labels have proportionally less influence.
pub fn detector_loss(
    outputs: &[ScaleOutput],
    targets: &[ScaleTargets],
    weights: LossWeights,
) -> Result<LossOutput, Error> {
    if outputs.len() != targets.len() || outputs.is_empty() {
        return Err(Error::Config(
            "loss needs one nonempty target for every detector scale".into(),
        ));
    }
    let mut heatmap_loss = Tensor::<1>::zeros([1], &outputs[0].heatmap.device());
    let mut offset_loss = heatmap_loss.clone();
    let mut size_loss = heatmap_loss.clone();
    let mut quality_loss = heatmap_loss.clone();
    let mut center_normalizer = heatmap_loss.clone();
    let mut regression_normalizer = heatmap_loss.clone();
    for (output, target) in outputs.iter().zip(targets) {
        center_normalizer = center_normalizer + target.center_weight.clone().sum();
        regression_normalizer = regression_normalizer + target.regression_weight.clone().sum();
        let probability = output.heatmap.clone().clamp(1e-6, 1.0 - 1e-6);
        let positive = target.center_weight.clone()
            * (probability.clone().neg() + 1.0).powf_scalar(2.0)
            * probability.clone().log();
        let negative = (target.heatmap.clone().neg() + 1.0).powf_scalar(4.0)
            * (target.center_weight.clone().neg() + 1.0)
            * probability.clone().powf_scalar(2.0)
            * (probability.neg() + 1.0).log();
        heatmap_loss = heatmap_loss - (positive + negative).sum();

        let regression_weight = target.regression_weight.clone();
        offset_loss = offset_loss
            + ((output.offset.clone() - target.offset.clone()).abs() * regression_weight.clone())
                .sum();
        let size_delta =
            (output.size.clone().log() - target.size.clone().clamp_min(1e-6).log()).abs();
        let size_quadratic = size_delta.clone().clamp_max(1.0);
        let smooth_size =
            size_quadratic.clone().powf_scalar(2.0) * 0.5 + size_delta - size_quadratic;
        size_loss = size_loss + (smooth_size * regression_weight.clone()).sum();
        let quality = output.quality.clone().clamp(1e-6, 1.0 - 1e-6);
        let quality_positive = target.regression_weight.clone()
            * (quality.clone().neg() + 1.0).powf_scalar(2.0)
            * quality.clone().log();
        let quality_negative = (target.regression_weight.clone().neg() + 1.0)
            * quality.clone().powf_scalar(2.0)
            * (quality.neg() + 1.0).log();
        quality_loss = quality_loss - (quality_positive + quality_negative).sum();
    }
    center_normalizer = center_normalizer.clamp_min(1.0);
    regression_normalizer = regression_normalizer.clamp_min(1.0);
    heatmap_loss = heatmap_loss / center_normalizer;
    offset_loss = offset_loss / regression_normalizer.clone();
    size_loss = size_loss / regression_normalizer.clone();
    quality_loss = quality_loss / regression_normalizer;
    let total = heatmap_loss.clone() * weights.heatmap
        + offset_loss.clone() * weights.offset
        + size_loss.clone() * weights.size
        + quality_loss.clone() * weights.quality;
    Ok(LossOutput {
        total,
        heatmap: heatmap_loss,
        offset: offset_loss,
        size: size_loss,
        quality: quality_loss,
    })
}

impl Detector {
    pub fn new(config: &DetectorConfig, device: &Device) -> Result<Self, Error> {
        config.validate()?;
        let stem = Block3d::new(
            config.input_channels,
            config.base_channels,
            [1, 3, 3],
            [1, 1, 1],
            device,
        );
        let mut stages = Vec::with_capacity(3);
        let mut heads = Vec::with_capacity(3);
        let mut input = config.base_channels;
        for (level, stride) in config.strides.iter().copied().enumerate() {
            let output = config.base_channels << level;
            let kernel = if stride[0] == 1 { [1, 3, 3] } else { [3, 3, 3] };
            stages.push(Block3d::new(input, output, kernel, stride, device));
            heads.push(CenterHead::new(
                output,
                config.classes,
                config.size_priors[level],
                device,
            ));
            input = output;
        }
        Ok(Self {
            stem,
            stages,
            heads,
        })
    }

    pub fn forward(&self, input: Tensor<5>) -> Vec<ScaleOutput> {
        let mut features = self.stem.forward(input);
        let mut output = Vec::with_capacity(3);
        for (stage, head) in self.stages.iter().zip(&self.heads) {
            features = stage.forward(features);
            output.push(head.forward(features.clone()));
        }
        output
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Box3d {
    pub center: [f32; 3],
    pub size: [f32; 3],
    pub class: usize,
    pub confidence: f32,
}

impl Box3d {
    pub fn corners(self) -> ([f32; 3], [f32; 3]) {
        let mut low = [0.0; 3];
        let mut high = [0.0; 3];
        for axis in 0..3 {
            low[axis] = self.center[axis] - self.size[axis] * 0.5;
            high[axis] = self.center[axis] + self.size[axis] * 0.5;
        }
        (low, high)
    }

    pub fn iou(self, other: Self) -> f32 {
        let (a0, a1) = self.corners();
        let (b0, b1) = other.corners();
        let intersection = (0..3)
            .map(|axis| (a1[axis].min(b1[axis]) - a0[axis].max(b0[axis])).max(0.0))
            .product::<f32>();
        let av = self.size.iter().product::<f32>();
        let bv = other.size.iter().product::<f32>();
        intersection / (av + bv - intersection).max(f32::EPSILON)
    }
}

/// Class-aware greedy 3D NMS. There is deliberately no global detection cap;
/// dense volumes are bounded spatially by the caller's blocks.
pub fn nms(mut boxes: Vec<Box3d>, threshold: f32) -> Vec<Box3d> {
    boxes.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept: Vec<Box3d> = Vec::new();
    'candidate: for candidate in boxes {
        for &accepted in &kept {
            if candidate.class == accepted.class && candidate.iou(accepted) > threshold {
                continue 'candidate;
            }
        }
        kept.push(candidate);
    }
    kept
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeacherObject {
    pub id: u64,
    pub bounds_start: [usize; 3],
    pub bounds_end: [usize; 3],
    pub class: usize,
    pub confidence: f32,
    pub truncated: bool,
}

/// Dense targets for one feature level. Regression values are meaningful only
/// where `regression_weight` is nonzero.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodedTargets {
    pub shape: [usize; 3],
    pub classes: usize,
    pub heatmap: Vec<f32>,
    pub center_weight: Vec<f32>,
    pub offset: Vec<[f32; 3]>,
    /// Object size in physical units.
    pub size: Vec<[f32; 3]>,
    pub regression_weight: Vec<f32>,
    pub collisions: usize,
}

/// Encode complete teacher objects whose centers lie in this patch.
///
/// `patch_start` and `patch_shape` are in source voxels. `stride` is the
/// feature stride in source voxels. Objects outside the patch and truncated
/// teacher objects are ignored. If two centers quantize to one feature cell,
/// the higher-confidence teacher owns regression there and `collisions`
/// records the ambiguity.
pub fn encode_targets(
    objects: &[TeacherObject],
    patch_start: [usize; 3],
    patch_shape: [usize; 3],
    stride: [usize; 3],
    voxel_size: [f32; 3],
    classes: usize,
) -> Result<EncodedTargets, Error> {
    if classes == 0 || stride.contains(&0) || patch_shape.contains(&0) {
        return Err(Error::Config(
            "target classes, patch shape, and stride must be positive".into(),
        ));
    }
    let shape = [
        patch_shape[0].div_ceil(stride[0]),
        patch_shape[1].div_ceil(stride[1]),
        patch_shape[2].div_ceil(stride[2]),
    ];
    let cells = shape.iter().product();
    let mut encoded = EncodedTargets {
        shape,
        classes,
        heatmap: vec![0.0; classes * cells],
        center_weight: vec![0.0; classes * cells],
        offset: vec![[0.0; 3]; cells],
        size: vec![[0.0; 3]; cells],
        regression_weight: vec![0.0; cells],
        collisions: 0,
    };
    for object in objects {
        let Some(target) = object.target_box(voxel_size) else {
            continue;
        };
        if object.class >= classes {
            return Err(Error::Config(format!(
                "teacher object {} has class {}, but the detector has {classes} classes",
                object.id, object.class
            )));
        }
        let mut feature = [0.0; 3];
        let mut index = [0usize; 3];
        let mut inside = true;
        for axis in 0..3 {
            let center_voxel = target.center[axis] / voxel_size[axis];
            let local = center_voxel - patch_start[axis] as f32;
            inside &= local >= 0.0 && local < patch_shape[axis] as f32;
            feature[axis] = local / stride[axis] as f32;
            index[axis] = feature[axis].floor().max(0.0) as usize;
        }
        if !inside || (0..3).any(|axis| index[axis] >= shape[axis]) {
            continue;
        }
        let flat = (index[0] * shape[1] + index[1]) * shape[2] + index[2];
        draw_center_gaussian(
            &mut encoded.heatmap,
            shape,
            cells,
            object.class,
            index,
            target.size,
            stride,
            voxel_size,
            object.confidence,
        );
        if encoded.regression_weight[flat] > object.confidence {
            encoded.collisions += 1;
            continue;
        }
        if encoded.regression_weight[flat] > 0.0 {
            encoded.collisions += 1;
        }
        encoded.offset[flat] = [
            feature[0] - index[0] as f32,
            feature[1] - index[1] as f32,
            feature[2] - index[2] as f32,
        ];
        encoded.size[flat] = target.size;
        encoded.regression_weight[flat] = object.confidence.max(f32::EPSILON);
        encoded.center_weight[object.class * cells + flat] = object.confidence;
    }
    Ok(encoded)
}

#[allow(clippy::too_many_arguments)]
fn draw_center_gaussian(
    heatmap: &mut [f32],
    shape: [usize; 3],
    cells: usize,
    class: usize,
    center: [usize; 3],
    object_size: [f32; 3],
    stride: [usize; 3],
    voxel_size: [f32; 3],
    confidence: f32,
) {
    let radius = [0, 1, 2].map(|axis| {
        ((object_size[axis] / voxel_size[axis] / stride[axis] as f32) * 0.5)
            .ceil()
            .clamp(1.0, 6.0) as isize
    });
    for dz in -radius[0]..=radius[0] {
        for dy in -radius[1]..=radius[1] {
            for dx in -radius[2]..=radius[2] {
                let delta = [dz, dy, dx];
                let mut at = [0usize; 3];
                let mut valid = true;
                let mut exponent = 0.0;
                for axis in 0..3 {
                    let coordinate = center[axis] as isize + delta[axis];
                    valid &= coordinate >= 0 && coordinate < shape[axis] as isize;
                    at[axis] = coordinate.max(0) as usize;
                    exponent += (delta[axis] as f32 / radius[axis] as f32).powi(2);
                }
                if !valid {
                    continue;
                }
                let flat = class * cells + (at[0] * shape[1] + at[1]) * shape[2] + at[2];
                heatmap[flat] = heatmap[flat].max((-0.5 * exponent).exp() * confidence);
            }
        }
    }
}

/// Decode one feature level after tensors have been copied to host memory.
/// Heatmap and quality use class-major and scalar C-order layouts; offset and
/// size use channel-major `[3,z,y,x]` layouts.
#[allow(clippy::too_many_arguments)]
pub fn decode_scale(
    heatmap: &[f32],
    offset: &[f32],
    size: &[f32],
    quality: &[f32],
    shape: [usize; 3],
    classes: usize,
    stride: [usize; 3],
    voxel_size: [f32; 3],
    threshold: f32,
) -> Result<Vec<Box3d>, Error> {
    let cells = shape.iter().product::<usize>();
    if heatmap.len() != classes * cells
        || offset.len() != 3 * cells
        || size.len() != 3 * cells
        || quality.len() != cells
    {
        return Err(Error::Config("detector output shapes do not match".into()));
    }
    let mut boxes = Vec::new();
    for class in 0..classes {
        for z in 0..shape[0] {
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    let cell = (z * shape[1] + y) * shape[2] + x;
                    let score = heatmap[class * cells + cell] * quality[cell];
                    if score < threshold || !local_maximum(heatmap, class, cells, shape, [z, y, x])
                    {
                        continue;
                    }
                    let index = [z, y, x];
                    let mut center = [0.0; 3];
                    let mut extent = [0.0; 3];
                    for axis in 0..3 {
                        center[axis] = (index[axis] as f32 + offset[axis * cells + cell])
                            * stride[axis] as f32
                            * voxel_size[axis];
                        extent[axis] = size[axis * cells + cell].max(f32::EPSILON);
                    }
                    boxes.push(Box3d {
                        center,
                        size: extent,
                        class,
                        confidence: score,
                    });
                }
            }
        }
    }
    Ok(boxes)
}

fn local_maximum(
    heatmap: &[f32],
    class: usize,
    cells: usize,
    shape: [usize; 3],
    at: [usize; 3],
) -> bool {
    let flat = (at[0] * shape[1] + at[1]) * shape[2] + at[2];
    let value = heatmap[class * cells + flat];
    for dz in -1isize..=1 {
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let delta = [dz, dy, dx];
                let mut neighbour = [0usize; 3];
                let mut valid = true;
                for axis in 0..3 {
                    let coordinate = at[axis] as isize + delta[axis];
                    valid &= coordinate >= 0 && coordinate < shape[axis] as isize;
                    neighbour[axis] = coordinate.max(0) as usize;
                }
                if valid {
                    let other = (neighbour[0] * shape[1] + neighbour[1]) * shape[2] + neighbour[2];
                    if heatmap[class * cells + other] > value {
                        return false;
                    }
                }
            }
        }
    }
    true
}

impl TeacherObject {
    pub fn target_box(&self, voxel_size: [f32; 3]) -> Option<Box3d> {
        if self.truncated || (0..3).any(|axis| self.bounds_end[axis] <= self.bounds_start[axis]) {
            return None;
        }
        let mut center = [0.0; 3];
        let mut size = [0.0; 3];
        for axis in 0..3 {
            center[axis] =
                (self.bounds_start[axis] + self.bounds_end[axis]) as f32 * 0.5 * voxel_size[axis];
            size[axis] =
                (self.bounds_end[axis] - self.bounds_start[axis]) as f32 * voxel_size[axis];
        }
        Some(Box3d {
            center,
            size,
            class: self.class,
            confidence: self.confidence,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid detector configuration: {0}")]
    Config(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_round_trip_preserves_one_anisotropic_box() {
        let teacher = TeacherObject {
            id: 7,
            bounds_start: [4, 20, 30],
            bounds_end: [8, 28, 42],
            class: 0,
            confidence: 1.0,
            truncated: false,
        };
        let voxel = [2.0, 0.5, 0.5];
        let stride = [2, 4, 4];
        let targets = encode_targets(&[teacher], [0; 3], [16, 64, 64], stride, voxel, 1).unwrap();
        let cells = targets.shape.iter().product::<usize>();
        let flat = targets
            .regression_weight
            .iter()
            .position(|weight| *weight > 0.0)
            .unwrap();
        let mut offsets = vec![0.0; 3 * cells];
        let mut sizes = vec![0.0; 3 * cells];
        for axis in 0..3 {
            offsets[axis * cells + flat] = targets.offset[flat][axis];
            sizes[axis * cells + flat] = targets.size[flat][axis];
        }
        let boxes = decode_scale(
            &targets.heatmap,
            &offsets,
            &sizes,
            &vec![1.0; cells],
            targets.shape,
            1,
            stride,
            voxel,
            0.99,
        )
        .unwrap();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].center, [12.0, 12.0, 18.0]);
        assert_eq!(boxes[0].size, [8.0, 4.0, 6.0]);
    }

    #[test]
    fn nms_is_volumetric_and_class_aware() {
        let a = Box3d {
            center: [5.0; 3],
            size: [4.0; 3],
            class: 0,
            confidence: 0.9,
        };
        let mut b = a;
        b.center[0] += 0.5;
        b.confidence = 0.8;
        let mut other_class = b;
        other_class.class = 1;
        assert_eq!(nms(vec![b, a, other_class], 0.5).len(), 2);
    }
}
