//! Blockflow-owned native SAM2 mask-decoder optimization.
//!
//! sam2-rs supplies upstream-compatible components and checkpoint names. This module owns the
//! training decision: loss weights, optimizer, steps, metrics, and eventually job cancellation.

use std::{collections::HashMap, fs, path::Path};

use sam2_candle_core::{backprop::GradStore, Device, Result as CandleResult, Tensor, Var};
use sam2_candle_nn::{loss::binary_cross_entropy_with_logit, ops::sigmoid, ParamsAdamW};
pub use sam2_rs::EncodedImage;
use sam2_rs::{
    FrozenDecoderInputs, ImagePromptBatch, MaskDecoderConfig, Sam2ImagePredictor, Sam2Variant,
    TrainableMaskDecoder,
};

use crate::sam::{load_trainable_sam1_vit_b, PointPrompt, PromptSet};
use sam1_candle_transformers::models::segment_anything::sam as sam1;

pub const SAM2_TRAINING_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sam2TrainingCheckpointMetadata {
    pub schema_version: u32,
    pub resume_key: String,
    pub completed_epochs: u32,
    pub optimizer_steps: u64,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub parameter_names: Vec<String>,
}

struct NamedAdamWVar {
    name: String,
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

/// AdamW state owned by Blockflow so native jobs can resume the exact optimization trajectory.
/// Candle's public AdamW intentionally exposes neither moment tensors nor its step counter.
struct ResumableAdamW {
    vars: Vec<NamedAdamWVar>,
    step_t: u64,
    params: ParamsAdamW,
}

impl ResumableAdamW {
    fn new(variables: &sam2_candle_nn::VarMap, params: ParamsAdamW) -> CandleResult<Self> {
        Self::new_filtered(variables, params, |_| true)
    }

    fn new_filtered(
        variables: &sam2_candle_nn::VarMap,
        params: ParamsAdamW,
        include: impl Fn(&str) -> bool,
    ) -> CandleResult<Self> {
        let data = variables
            .data()
            .lock()
            .map_err(|_| sam2_candle_core::Error::Msg("decoder variable map is poisoned".into()))?;
        let mut named = data
            .iter()
            .filter(|(name, var)| include(name) && var.dtype().is_float())
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect::<Vec<_>>();
        named.sort_by(|left, right| left.0.cmp(&right.0));
        let vars = named
            .into_iter()
            .map(|(name, var)| {
                Ok(NamedAdamWVar {
                    name,
                    first_moment: Var::zeros(var.shape(), var.dtype(), var.device())?,
                    second_moment: Var::zeros(var.shape(), var.dtype(), var.device())?,
                    var,
                })
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            vars,
            step_t: 0,
            params,
        })
    }

    fn step(&mut self, grads: &GradStore) -> CandleResult<()> {
        self.step_t = self.step_t.saturating_add(1);
        let step = i32::try_from(self.step_t).unwrap_or(i32::MAX);
        let scale_m = 1.0 / (1.0 - self.params.beta1.powi(step));
        let scale_v = 1.0 / (1.0 - self.params.beta2.powi(step));
        let shrink = 1.0 - self.params.lr * self.params.weight_decay;
        for state in &self.vars {
            let Some(gradient) = grads.get(&state.var) else {
                continue;
            };
            let next_m = ((state.first_moment.as_tensor() * self.params.beta1)?
                + (gradient * (1.0 - self.params.beta1))?)?;
            let next_v = ((state.second_moment.as_tensor() * self.params.beta2)?
                + (gradient.sqr()? * (1.0 - self.params.beta2))?)?;
            let m_hat = (&next_m * scale_m)?;
            let v_hat = (&next_v * scale_v)?;
            let adjusted = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
            let next_parameter =
                ((state.var.as_tensor() * shrink)? - (adjusted * self.params.lr)?)?;
            state.first_moment.set(&next_m)?;
            state.second_moment.set(&next_v)?;
            state.var.set(&next_parameter)?;
        }
        Ok(())
    }

    fn save(&self, path: &Path) -> CandleResult<()> {
        let mut tensors = HashMap::with_capacity(self.vars.len() * 2);
        for state in &self.vars {
            tensors.insert(
                format!("first.{}", state.name),
                state.first_moment.as_detached_tensor(),
            );
            tensors.insert(
                format!("second.{}", state.name),
                state.second_moment.as_detached_tensor(),
            );
        }
        sam2_candle_core::safetensors::save(&tensors, path)
    }

    fn restore(
        &mut self,
        path: &Path,
        metadata: &Sam2TrainingCheckpointMetadata,
    ) -> CandleResult<()> {
        let tensors = sam2_candle_core::safetensors::load(
            path,
            self.vars
                .first()
                .map(|state| state.var.device())
                .unwrap_or(&Device::Cpu),
        )?;
        let names = self
            .vars
            .iter()
            .map(|state| state.name.clone())
            .collect::<Vec<_>>();
        if names != metadata.parameter_names {
            sam2_candle_core::bail!("SAM2 optimizer checkpoint parameter names do not match")
        }
        if tensors.len() != self.vars.len() * 2 {
            sam2_candle_core::bail!("SAM2 optimizer checkpoint has unexpected tensors")
        }
        for state in &self.vars {
            let first = tensors
                .get(&format!("first.{}", state.name))
                .ok_or_else(|| {
                    sam2_candle_core::Error::Msg(format!(
                        "SAM2 optimizer checkpoint lacks first moment for {}",
                        state.name
                    ))
                })?;
            let second = tensors
                .get(&format!("second.{}", state.name))
                .ok_or_else(|| {
                    sam2_candle_core::Error::Msg(format!(
                        "SAM2 optimizer checkpoint lacks second moment for {}",
                        state.name
                    ))
                })?;
            state.first_moment.set(first)?;
            state.second_moment.set(second)?;
        }
        self.step_t = metadata.optimizer_steps;
        Ok(())
    }

    fn parameter_names(&self) -> Vec<String> {
        self.vars.iter().map(|state| state.name.clone()).collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sam2DecoderLossConfig {
    pub mask_bce_weight: f64,
    pub mask_dice_weight: f64,
    pub iou_head_weight: f64,
    pub object_head_weight: f64,
}

impl Default for Sam2DecoderLossConfig {
    fn default() -> Self {
        Self {
            mask_bce_weight: 1.0,
            mask_dice_weight: 1.0,
            iou_head_weight: 0.1,
            object_head_weight: 0.1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sam2DecoderStepMetrics {
    pub total_loss: f32,
    pub mask_bce: f32,
    pub mask_dice_loss: f32,
    pub iou_head_loss: f32,
    pub object_head_loss: f32,
    pub soft_dice: f32,
    pub hard_dice: f32,
    pub hard_iou: f32,
    /// Boundary F score at one decoder-logit pixel tolerance.
    pub boundary_f_score: f32,
}

#[derive(Clone, Debug)]
pub struct Sam2MaskComparison {
    pub width: usize,
    pub height: usize,
    pub prediction: Vec<bool>,
    pub target: Vec<bool>,
}

/// Frozen image/prompt features plus the exact corrected binary target. Encoder features are
/// detached by sam2-rs, so retaining this value across epochs neither grows an autograd graph nor
/// changes the faithful inference translation.
pub struct Sam2DecoderExample {
    pub inputs: FrozenDecoderInputs,
    target: Tensor,
}

impl Sam2DecoderExample {
    /// Approximate device storage retained by this frozen example. Tensor views may share storage,
    /// so this is deliberately conservative for admission against the job memory budget.
    pub fn estimated_bytes(&self) -> u64 {
        let tensor_bytes = |tensor: &Tensor| {
            tensor
                .elem_count()
                .saturating_mul(tensor.dtype().size_in_bytes()) as u64
        };
        tensor_bytes(&self.inputs.image_embeddings)
            .saturating_add(tensor_bytes(&self.inputs.image_pe))
            .saturating_add(tensor_bytes(&self.inputs.sparse_prompt_embeddings))
            .saturating_add(tensor_bytes(&self.inputs.dense_prompt_embeddings))
            .saturating_add(tensor_bytes(&self.inputs.high_res_features[0]))
            .saturating_add(tensor_bytes(&self.inputs.high_res_features[1]))
            .saturating_add(tensor_bytes(&self.target))
    }
}

pub struct Sam2DecoderTrainingSession {
    encoder: Sam2ImagePredictor,
    trainer: Sam2DecoderTrainer,
    device: Device,
}

/// Cached, detached SAM1 ViT-B image embedding and the geometry needed to transform prompts.
pub struct Sam1EncodedImage {
    embeddings: Tensor,
    original_width: usize,
    original_height: usize,
    resized_width: usize,
    resized_height: usize,
}

impl Sam1EncodedImage {
    pub fn estimated_bytes(&self) -> u64 {
        self.embeddings
            .elem_count()
            .saturating_mul(self.embeddings.dtype().size_in_bytes()) as u64
    }
}

pub struct Sam1DecoderExample {
    inputs: sam1::FrozenDecoderInputs,
    target: Tensor,
}

impl Sam1DecoderExample {
    pub fn estimated_bytes(&self) -> u64 {
        let tensor_bytes = |tensor: &Tensor| {
            tensor
                .elem_count()
                .saturating_mul(tensor.dtype().size_in_bytes()) as u64
        };
        tensor_bytes(&self.inputs.image_embeddings)
            .saturating_add(tensor_bytes(&self.inputs.image_pe))
            .saturating_add(tensor_bytes(&self.inputs.sparse_prompt_embeddings))
            .saturating_add(tensor_bytes(&self.inputs.dense_prompt_embeddings))
            .saturating_add(tensor_bytes(&self.target))
    }
}

/// Blockflow-owned native SAM1 decoder training. The Candle SAM translation supplies the exact
/// architecture and detached decoder inputs; optimization policy remains here.
pub struct Sam1DecoderTrainingSession {
    model: sam1::Sam,
    variables: sam2_candle_nn::VarMap,
    optimizer: ResumableAdamW,
    loss: Sam2DecoderLossConfig,
    device: Device,
    steps: u64,
}

pub enum NativeSamEncodedImage {
    Sam2(EncodedImage),
    Sam1(Sam1EncodedImage),
}

impl NativeSamEncodedImage {
    pub fn estimated_bytes(&self) -> u64 {
        match self {
            Self::Sam2(value) => value.estimated_bytes(),
            Self::Sam1(value) => value.estimated_bytes(),
        }
    }
}

pub enum NativeSamDecoderExample {
    Sam2(Sam2DecoderExample),
    Sam1(Sam1DecoderExample),
}

/// Family-neutral training surface used by Blockflow jobs. A model family is selected once from
/// the immutable parent descriptor; snapshots and job payloads remain model neutral.
pub enum NativeSamDecoderTrainingSession {
    Sam2(Sam2DecoderTrainingSession),
    Sam1(Sam1DecoderTrainingSession),
}

impl NativeSamDecoderTrainingSession {
    pub fn encode_image(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> CandleResult<NativeSamEncodedImage> {
        match self {
            Self::Sam2(session) => session
                .encode_image(rgb, width, height)
                .map(NativeSamEncodedImage::Sam2),
            Self::Sam1(session) => session
                .encode_image(rgb, width, height)
                .map(NativeSamEncodedImage::Sam1),
        }
    }

    pub fn prepare_example_from_encoded(
        &mut self,
        encoded: &NativeSamEncodedImage,
        prompts: &PromptSet,
        target: &[bool],
        target_width: usize,
        target_height: usize,
    ) -> CandleResult<NativeSamDecoderExample> {
        match (self, encoded) {
            (Self::Sam2(session), NativeSamEncodedImage::Sam2(encoded)) => session
                .prepare_example_from_encoded(encoded, prompts, target, target_width, target_height)
                .map(NativeSamDecoderExample::Sam2),
            (Self::Sam1(session), NativeSamEncodedImage::Sam1(encoded)) => session
                .prepare_example_from_encoded(encoded, prompts, target, target_width, target_height)
                .map(NativeSamDecoderExample::Sam1),
            _ => sam2_candle_core::bail!("encoded image belongs to another SAM model family"),
        }
    }

    pub fn step(
        &mut self,
        example: &NativeSamDecoderExample,
    ) -> CandleResult<Sam2DecoderStepMetrics> {
        match (self, example) {
            (Self::Sam2(session), NativeSamDecoderExample::Sam2(example)) => session.step(example),
            (Self::Sam1(session), NativeSamDecoderExample::Sam1(example)) => session.step(example),
            _ => sam2_candle_core::bail!("training example belongs to another SAM model family"),
        }
    }

    pub fn evaluate(
        &mut self,
        example: &NativeSamDecoderExample,
    ) -> CandleResult<Sam2DecoderStepMetrics> {
        match (self, example) {
            (Self::Sam2(session), NativeSamDecoderExample::Sam2(example)) => {
                session.evaluate(example)
            }
            (Self::Sam1(session), NativeSamDecoderExample::Sam1(example)) => {
                session.evaluate(example)
            }
            _ => sam2_candle_core::bail!("training example belongs to another SAM model family"),
        }
    }

    pub fn evaluate_with_prediction(
        &mut self,
        example: &NativeSamDecoderExample,
    ) -> CandleResult<(Sam2DecoderStepMetrics, Sam2MaskComparison)> {
        match (self, example) {
            (Self::Sam2(session), NativeSamDecoderExample::Sam2(example)) => {
                session.evaluate_with_prediction(example)
            }
            (Self::Sam1(session), NativeSamDecoderExample::Sam1(example)) => {
                session.evaluate_with_prediction(example)
            }
            _ => sam2_candle_core::bail!("training example belongs to another SAM model family"),
        }
    }

    pub fn save_decoder(&self, path: impl AsRef<Path>) -> CandleResult<()> {
        match self {
            Self::Sam2(session) => session.save_decoder(path),
            Self::Sam1(session) => session.save_decoder(path),
        }
    }

    pub fn steps(&self) -> u64 {
        match self {
            Self::Sam2(session) => session.steps(),
            Self::Sam1(session) => session.steps(),
        }
    }

    pub fn save_checkpoint(
        &self,
        directory: impl AsRef<Path>,
        resume_key: &str,
        completed_epochs: u32,
    ) -> CandleResult<Sam2TrainingCheckpointMetadata> {
        match self {
            Self::Sam2(session) => session.save_checkpoint(directory, resume_key, completed_epochs),
            Self::Sam1(session) => session.save_checkpoint(directory, resume_key, completed_epochs),
        }
    }

    pub fn restore_checkpoint(
        &mut self,
        directory: impl AsRef<Path>,
        expected_resume_key: &str,
    ) -> CandleResult<Sam2TrainingCheckpointMetadata> {
        match self {
            Self::Sam2(session) => session.restore_checkpoint(directory, expected_resume_key),
            Self::Sam1(session) => session.restore_checkpoint(directory, expected_resume_key),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sam2TrainingDevice {
    Cpu,
    Cuda(usize),
}

pub struct Sam2DecoderTrainer {
    decoder: TrainableMaskDecoder,
    optimizer: ResumableAdamW,
    loss: Sam2DecoderLossConfig,
    steps: u64,
}

impl Sam2DecoderTrainer {
    pub fn load(
        variant: Sam2Variant,
        checkpoint: impl AsRef<Path>,
        device: &Device,
        learning_rate: f64,
        weight_decay: f64,
        loss: Sam2DecoderLossConfig,
    ) -> CandleResult<Self> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            sam2_candle_core::bail!("SAM2 decoder learning rate must be positive")
        }
        for weight in [
            loss.mask_bce_weight,
            loss.mask_dice_weight,
            loss.iou_head_weight,
            loss.object_head_weight,
        ] {
            if !weight.is_finite() || weight < 0.0 {
                sam2_candle_core::bail!("SAM2 decoder loss weights must be finite and nonnegative")
            }
        }
        Self::load_with_mask_decoder(
            variant,
            checkpoint,
            None::<&Path>,
            device,
            learning_rate,
            weight_decay,
            loss,
        )
    }

    pub fn load_with_mask_decoder(
        variant: Sam2Variant,
        checkpoint: impl AsRef<Path>,
        mask_decoder: Option<impl AsRef<Path>>,
        device: &Device,
        learning_rate: f64,
        weight_decay: f64,
        loss: Sam2DecoderLossConfig,
    ) -> CandleResult<Self> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            sam2_candle_core::bail!("SAM2 decoder learning rate must be positive")
        }
        for weight in [
            loss.mask_bce_weight,
            loss.mask_dice_weight,
            loss.iou_head_weight,
            loss.object_head_weight,
        ] {
            if !weight.is_finite() || weight < 0.0 {
                sam2_candle_core::bail!("SAM2 decoder loss weights must be finite and nonnegative")
            }
        }
        let decoder = TrainableMaskDecoder::load_sam2_1_with_mask_decoder(
            variant,
            checkpoint,
            mask_decoder,
            device,
            MaskDecoderConfig::raw(),
        )?;
        let optimizer = ResumableAdamW::new(
            decoder.variables(),
            ParamsAdamW {
                lr: learning_rate,
                weight_decay,
                ..Default::default()
            },
        )?;
        Ok(Self {
            decoder,
            optimizer,
            loss,
            steps: 0,
        })
    }

    /// One frozen-encoder decoder update. `target` is Bx1xHxW in decoder-logit coordinates.
    pub fn step(
        &mut self,
        inputs: &FrozenDecoderInputs,
        target: &Tensor,
    ) -> CandleResult<Sam2DecoderStepMetrics> {
        let output = self.decoder.forward(inputs, false)?;
        self.step_output(output, target, true)
    }

    pub fn step_binary_mask(
        &mut self,
        inputs: &FrozenDecoderInputs,
        row_major: &[bool],
        width: usize,
        height: usize,
    ) -> CandleResult<Sam2DecoderStepMetrics> {
        let output = self.decoder.forward(inputs, false)?;
        let (_, _, output_height, output_width) = output.masks.dims4()?;
        let target = resized_binary_target(
            row_major,
            width,
            height,
            output_width,
            output_height,
            output.masks.device(),
        )?;
        self.step_output(output, &target, true)
    }

    pub fn evaluate_binary_mask(
        &mut self,
        inputs: &FrozenDecoderInputs,
        row_major: &[bool],
        width: usize,
        height: usize,
    ) -> CandleResult<Sam2DecoderStepMetrics> {
        let output = self.decoder.forward(inputs, false)?;
        let (_, _, output_height, output_width) = output.masks.dims4()?;
        let target = resized_binary_target(
            row_major,
            width,
            height,
            output_width,
            output_height,
            output.masks.device(),
        )?;
        self.step_output(output, &target, false)
    }

    pub fn evaluate_binary_mask_with_prediction(
        &mut self,
        inputs: &FrozenDecoderInputs,
        row_major: &[bool],
        width: usize,
        height: usize,
    ) -> CandleResult<(Sam2DecoderStepMetrics, Sam2MaskComparison)> {
        let output = self.decoder.forward(inputs, false)?;
        let (_, _, output_height, output_width) = output.masks.dims4()?;
        let target = resized_binary_target(
            row_major,
            width,
            height,
            output_width,
            output_height,
            output.masks.device(),
        )?;
        let prediction = output
            .masks
            .gt(0_f64)?
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<u8>()?
            .into_iter()
            .map(|value| value != 0)
            .collect();
        let target_pixels = target
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|value| value >= 0.5)
            .collect();
        let metrics = self.step_output(output, &target, false)?;
        Ok((
            metrics,
            Sam2MaskComparison {
                width: output_width,
                height: output_height,
                prediction,
                target: target_pixels,
            },
        ))
    }

    fn step_output(
        &mut self,
        output: sam2_rs::MaskDecoderOutput,
        target: &Tensor,
        optimize: bool,
    ) -> CandleResult<Sam2DecoderStepMetrics> {
        if output.masks.dims() != target.dims() {
            sam2_candle_core::bail!(
                "SAM2 decoder target shape {:?} differs from logits {:?}",
                target.dims(),
                output.masks.dims()
            )
        }
        let target = target.to_dtype(output.masks.dtype())?;
        let mask_bce = binary_cross_entropy_with_logit(&output.masks, &target)?;
        let probabilities = sigmoid(&output.masks)?;
        let intersection = probabilities.mul(&target)?.sum_all()?;
        let denominator = probabilities.sum_all()?.add(&target.sum_all()?)?;
        let soft_dice = ((intersection * 2.0)? + 1.0)?.broadcast_div(&(denominator + 1.0)?)?;
        let mask_dice_loss = soft_dice.affine(-1.0, 1.0)?;
        let prediction = output.masks.gt(0_f64)?.to_dtype(output.masks.dtype())?;
        let hard_intersection = prediction.mul(&target)?.sum_all()?;
        let prediction_area = prediction.sum_all()?;
        let target_area = target.sum_all()?;
        let hard_dice = ((&hard_intersection * 2.0)? + 1.0)?
            .broadcast_div(&((&prediction_area + &target_area)? + 1.0)?)?;
        let hard_union = ((&prediction_area + &target_area)? - &hard_intersection)?;
        let hard_iou = (&hard_intersection + 1.0)?.broadcast_div(&(hard_union + 1.0)?)?;
        let (_, _, metric_height, metric_width) = prediction.dims4()?;
        let boundary_f_score = boundary_f_score(
            &prediction.flatten_all()?.to_vec1::<f32>()?,
            &target.flatten_all()?.to_vec1::<f32>()?,
            metric_width,
            metric_height,
            1,
        );
        let iou_target = output
            .iou_predictions
            .ones_like()?
            .broadcast_mul(&soft_dice.detach())?;
        let iou_head_loss = output.iou_predictions.sub(&iou_target)?.sqr()?.mean_all()?;
        let object_target = output.object_score_logits.ones_like()?;
        let object_head_loss =
            binary_cross_entropy_with_logit(&output.object_score_logits, &object_target)?;
        let total = ((&mask_bce * self.loss.mask_bce_weight)?
            + (&mask_dice_loss * self.loss.mask_dice_weight)?)?;
        let total = (total + (&iou_head_loss * self.loss.iou_head_weight)?)?;
        let total = (total + (&object_head_loss * self.loss.object_head_weight)?)?;
        let metrics = Sam2DecoderStepMetrics {
            total_loss: total.to_scalar()?,
            mask_bce: mask_bce.to_scalar()?,
            mask_dice_loss: mask_dice_loss.to_scalar()?,
            iou_head_loss: iou_head_loss.to_scalar()?,
            object_head_loss: object_head_loss.to_scalar()?,
            soft_dice: soft_dice.to_scalar()?,
            hard_dice: hard_dice.to_scalar()?,
            hard_iou: hard_iou.to_scalar()?,
            boundary_f_score,
        };
        if optimize {
            let gradients = total.backward()?;
            self.optimizer.step(&gradients)?;
            self.steps = self.steps.saturating_add(1);
        }
        Ok(metrics)
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn save_decoder(&self, path: impl AsRef<Path>) -> CandleResult<()> {
        self.decoder.save_decoder(path)
    }

    pub fn save_checkpoint(
        &self,
        directory: impl AsRef<Path>,
        resume_key: &str,
        completed_epochs: u32,
    ) -> CandleResult<Sam2TrainingCheckpointMetadata> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        let decoder_tmp = directory.join("decoder.safetensors.tmp");
        let optimizer_tmp = directory.join("optimizer.safetensors.tmp");
        self.decoder.save_decoder(&decoder_tmp)?;
        self.optimizer.save(&optimizer_tmp)?;
        fs::rename(&decoder_tmp, directory.join("decoder.safetensors"))?;
        fs::rename(&optimizer_tmp, directory.join("optimizer.safetensors"))?;
        let metadata = Sam2TrainingCheckpointMetadata {
            schema_version: SAM2_TRAINING_CHECKPOINT_SCHEMA_VERSION,
            resume_key: resume_key.into(),
            completed_epochs,
            optimizer_steps: self.optimizer.step_t,
            learning_rate: self.optimizer.params.lr,
            weight_decay: self.optimizer.params.weight_decay,
            beta1: self.optimizer.params.beta1,
            beta2: self.optimizer.params.beta2,
            epsilon: self.optimizer.params.eps,
            parameter_names: self.optimizer.parameter_names(),
        };
        let metadata_tmp = directory.join("metadata.json.tmp");
        let metadata_bytes = serde_json::to_vec_pretty(&metadata)
            .map_err(|error| sam2_candle_core::Error::Msg(error.to_string()))?;
        fs::write(&metadata_tmp, metadata_bytes)?;
        fs::rename(metadata_tmp, directory.join("metadata.json"))?;
        Ok(metadata)
    }

    pub fn restore_checkpoint(
        &mut self,
        directory: impl AsRef<Path>,
        expected_resume_key: &str,
    ) -> CandleResult<Sam2TrainingCheckpointMetadata> {
        let directory = directory.as_ref();
        let metadata: Sam2TrainingCheckpointMetadata =
            serde_json::from_slice(&fs::read(directory.join("metadata.json"))?)
                .map_err(|error| sam2_candle_core::Error::Msg(error.to_string()))?;
        if metadata.schema_version != SAM2_TRAINING_CHECKPOINT_SCHEMA_VERSION {
            sam2_candle_core::bail!("unsupported SAM2 training checkpoint schema")
        }
        if metadata.resume_key != expected_resume_key {
            sam2_candle_core::bail!("SAM2 training checkpoint belongs to a different recipe")
        }
        let same = |left: f64, right: f64| left.to_bits() == right.to_bits();
        if !same(metadata.learning_rate, self.optimizer.params.lr)
            || !same(metadata.weight_decay, self.optimizer.params.weight_decay)
            || !same(metadata.beta1, self.optimizer.params.beta1)
            || !same(metadata.beta2, self.optimizer.params.beta2)
            || !same(metadata.epsilon, self.optimizer.params.eps)
        {
            sam2_candle_core::bail!("SAM2 training checkpoint optimizer settings do not match")
        }
        let mut variables = self.decoder.variables().clone();
        variables.load(directory.join("decoder.safetensors"))?;
        self.optimizer
            .restore(&directory.join("optimizer.safetensors"), &metadata)?;
        self.steps = metadata.optimizer_steps;
        Ok(metadata)
    }
}

impl Sam1DecoderTrainingSession {
    pub fn load_on_device(
        checkpoint: impl AsRef<Path>,
        mask_decoder: Option<&Path>,
        training_device: Sam2TrainingDevice,
        learning_rate: f64,
        weight_decay: f64,
        loss: Sam2DecoderLossConfig,
    ) -> CandleResult<Self> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            sam2_candle_core::bail!("SAM1 decoder learning rate must be positive")
        }
        let device = training_device_device(training_device)?;
        let (model, variables) = load_trainable_sam1_vit_b(checkpoint.as_ref(), &device)
            .map_err(sam2_candle_core::Error::Msg)?;
        if let Some(path) = mask_decoder {
            restore_variables_with_prefix(&variables, path, "mask_decoder.")?;
        }
        let optimizer = ResumableAdamW::new_filtered(
            &variables,
            ParamsAdamW {
                lr: learning_rate,
                weight_decay,
                ..Default::default()
            },
            |name| name.starts_with("mask_decoder."),
        )?;
        Ok(Self {
            model,
            variables,
            optimizer,
            loss,
            device,
            steps: 0,
        })
    }

    pub fn encode_image(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> CandleResult<Sam1EncodedImage> {
        if width == 0 || height == 0 || rgb.len() != width.saturating_mul(height).saturating_mul(3)
        {
            sam2_candle_core::bail!("SAM1 training image is not valid interleaved RGB8")
        }
        let scale = sam1::IMAGE_SIZE as f64 / width.max(height) as f64;
        let resized_width = ((width as f64 * scale) + 0.5).floor().max(1.0) as usize;
        let resized_height = ((height as f64 * scale) + 0.5).floor().max(1.0) as usize;
        let image = Tensor::from_vec(rgb.to_vec(), (height, width, 3), &self.device)?
            .permute((2, 0, 1))?
            .to_dtype(sam2_candle_core::DType::F32)?
            .unsqueeze(0)?
            .upsample_bilinear2d(resized_height, resized_width, false)?
            .squeeze(0)?;
        Ok(Sam1EncodedImage {
            embeddings: self.model.embeddings(&image)?.detach(),
            original_width: width,
            original_height: height,
            resized_width,
            resized_height,
        })
    }

    pub fn prepare_example_from_encoded(
        &self,
        encoded: &Sam1EncodedImage,
        prompts: &PromptSet,
        target: &[bool],
        target_width: usize,
        target_height: usize,
    ) -> CandleResult<Sam1DecoderExample> {
        if target.len() != target_width.saturating_mul(target_height) {
            sam2_candle_core::bail!("SAM1 training target dimensions do not match its pixels")
        }
        let points = prompts
            .points
            .iter()
            .map(|point| {
                (
                    f64::from(point.x) / encoded.original_width as f64,
                    f64::from(point.y) / encoded.original_height as f64,
                    point.positive,
                )
            })
            .collect::<Vec<_>>();
        let bounding_box = prompts.bounding_box.map(|bounds| {
            [
                f64::from(bounds.x0) / encoded.original_width as f64,
                f64::from(bounds.y0) / encoded.original_height as f64,
                f64::from(bounds.x1) / encoded.original_width as f64,
                f64::from(bounds.y1) / encoded.original_height as f64,
            ]
        });
        let mask_input = prompts
            .mask_input
            .as_ref()
            .map(|(values, height, width)| {
                if (*width, *height) != (256, 256) || values.len() != width.saturating_mul(*height)
                {
                    sam2_candle_core::bail!("SAM1 dense training prompt must be 256x256")
                }
                Tensor::from_vec(values.clone(), (1, 1, *height, *width), &self.device)
            })
            .transpose()?;
        let inputs = self.model.frozen_decoder_inputs(
            &encoded.embeddings,
            encoded.resized_height,
            encoded.resized_width,
            &points,
            bounding_box,
            mask_input.as_ref(),
        )?;
        let target = Tensor::from_vec(
            target
                .iter()
                .map(|value| if *value { 1_f32 } else { 0_f32 })
                .collect::<Vec<_>>(),
            (1, 1, target_height, target_width),
            &self.device,
        )?;
        Ok(Sam1DecoderExample { inputs, target })
    }

    pub fn step(&mut self, example: &Sam1DecoderExample) -> CandleResult<Sam2DecoderStepMetrics> {
        self.run_example(example, true).map(|value| value.0)
    }

    pub fn evaluate(
        &mut self,
        example: &Sam1DecoderExample,
    ) -> CandleResult<Sam2DecoderStepMetrics> {
        self.run_example(example, false).map(|value| value.0)
    }

    pub fn evaluate_with_prediction(
        &mut self,
        example: &Sam1DecoderExample,
    ) -> CandleResult<(Sam2DecoderStepMetrics, Sam2MaskComparison)> {
        let (metrics, comparison) = self.run_example(example, false)?;
        Ok((metrics, comparison.expect("prediction requested")))
    }

    fn run_example(
        &mut self,
        example: &Sam1DecoderExample,
        optimize: bool,
    ) -> CandleResult<(Sam2DecoderStepMetrics, Option<Sam2MaskComparison>)> {
        let (masks, iou_predictions) = self.model.decode_frozen(&example.inputs, false)?;
        let (_, _, output_height, output_width) = masks.dims4()?;
        let (_, _, target_height, target_width) = example.target.dims4()?;
        let target_values = example
            .target
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|value| value >= 0.5)
            .collect::<Vec<_>>();
        let target = resized_binary_target(
            &target_values,
            target_width,
            target_height,
            output_width,
            output_height,
            &self.device,
        )?
        .to_dtype(masks.dtype())?;
        let mask_bce = binary_cross_entropy_with_logit(&masks, &target)?;
        let probabilities = sigmoid(&masks)?;
        let intersection = probabilities.mul(&target)?.sum_all()?;
        let denominator = probabilities.sum_all()?.add(&target.sum_all()?)?;
        let soft_dice = ((intersection * 2.0)? + 1.0)?.broadcast_div(&(denominator + 1.0)?)?;
        let mask_dice_loss = soft_dice.affine(-1.0, 1.0)?;
        let prediction = masks.gt(0_f64)?.to_dtype(masks.dtype())?;
        let hard_intersection = prediction.mul(&target)?.sum_all()?;
        let prediction_area = prediction.sum_all()?;
        let target_area = target.sum_all()?;
        let hard_dice = ((&hard_intersection * 2.0)? + 1.0)?
            .broadcast_div(&((&prediction_area + &target_area)? + 1.0)?)?;
        let hard_union = ((&prediction_area + &target_area)? - &hard_intersection)?;
        let hard_iou = (&hard_intersection + 1.0)?.broadcast_div(&(hard_union + 1.0)?)?;
        let iou_target = iou_predictions
            .ones_like()?
            .broadcast_mul(&hard_iou.detach())?;
        let iou_head_loss = iou_predictions.sub(&iou_target)?.sqr()?.mean_all()?;
        let object_head_loss = Tensor::zeros((), masks.dtype(), masks.device())?;
        let total = ((&mask_bce * self.loss.mask_bce_weight)?
            + (&mask_dice_loss * self.loss.mask_dice_weight)?)?;
        let total = (total + (&iou_head_loss * self.loss.iou_head_weight)?)?;
        let prediction_values = prediction.flatten_all()?.to_vec1::<f32>()?;
        let target_pixels = target.flatten_all()?.to_vec1::<f32>()?;
        let boundary = boundary_f_score(
            &prediction_values,
            &target_pixels,
            output_width,
            output_height,
            1,
        );
        let metrics = Sam2DecoderStepMetrics {
            total_loss: total.to_scalar()?,
            mask_bce: mask_bce.to_scalar()?,
            mask_dice_loss: mask_dice_loss.to_scalar()?,
            iou_head_loss: iou_head_loss.to_scalar()?,
            object_head_loss: object_head_loss.to_scalar()?,
            soft_dice: soft_dice.to_scalar()?,
            hard_dice: hard_dice.to_scalar()?,
            hard_iou: hard_iou.to_scalar()?,
            boundary_f_score: boundary,
        };
        if optimize {
            let gradients = total.backward()?;
            self.optimizer.step(&gradients)?;
            self.steps = self.steps.saturating_add(1);
        }
        let comparison = (!optimize).then(|| Sam2MaskComparison {
            width: output_width,
            height: output_height,
            prediction: prediction_values
                .into_iter()
                .map(|value| value >= 0.5)
                .collect(),
            target: target_pixels
                .into_iter()
                .map(|value| value >= 0.5)
                .collect(),
        });
        Ok((metrics, comparison))
    }

    pub fn save_decoder(&self, path: impl AsRef<Path>) -> CandleResult<()> {
        save_variables_with_prefix(&self.variables, path.as_ref(), "mask_decoder.")
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn save_checkpoint(
        &self,
        directory: impl AsRef<Path>,
        resume_key: &str,
        completed_epochs: u32,
    ) -> CandleResult<Sam2TrainingCheckpointMetadata> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        let decoder_tmp = directory.join("decoder.safetensors.tmp");
        let optimizer_tmp = directory.join("optimizer.safetensors.tmp");
        self.save_decoder(&decoder_tmp)?;
        self.optimizer.save(&optimizer_tmp)?;
        fs::rename(&decoder_tmp, directory.join("decoder.safetensors"))?;
        fs::rename(&optimizer_tmp, directory.join("optimizer.safetensors"))?;
        let metadata = Sam2TrainingCheckpointMetadata {
            schema_version: SAM2_TRAINING_CHECKPOINT_SCHEMA_VERSION,
            resume_key: resume_key.into(),
            completed_epochs,
            optimizer_steps: self.optimizer.step_t,
            learning_rate: self.optimizer.params.lr,
            weight_decay: self.optimizer.params.weight_decay,
            beta1: self.optimizer.params.beta1,
            beta2: self.optimizer.params.beta2,
            epsilon: self.optimizer.params.eps,
            parameter_names: self.optimizer.parameter_names(),
        };
        fs::write(
            directory.join("metadata.json.tmp"),
            serde_json::to_vec_pretty(&metadata)
                .map_err(|error| sam2_candle_core::Error::Msg(error.to_string()))?,
        )?;
        fs::rename(
            directory.join("metadata.json.tmp"),
            directory.join("metadata.json"),
        )?;
        Ok(metadata)
    }

    pub fn restore_checkpoint(
        &mut self,
        directory: impl AsRef<Path>,
        expected_resume_key: &str,
    ) -> CandleResult<Sam2TrainingCheckpointMetadata> {
        let directory = directory.as_ref();
        let metadata: Sam2TrainingCheckpointMetadata =
            serde_json::from_slice(&fs::read(directory.join("metadata.json"))?)
                .map_err(|error| sam2_candle_core::Error::Msg(error.to_string()))?;
        validate_checkpoint_metadata(&metadata, expected_resume_key, &self.optimizer)?;
        restore_variables_with_prefix(
            &self.variables,
            &directory.join("decoder.safetensors"),
            "mask_decoder.",
        )?;
        self.optimizer
            .restore(&directory.join("optimizer.safetensors"), &metadata)?;
        self.steps = metadata.optimizer_steps;
        Ok(metadata)
    }
}

fn training_device_device(training_device: Sam2TrainingDevice) -> CandleResult<Device> {
    match training_device {
        Sam2TrainingDevice::Cpu => Ok(Device::Cpu),
        Sam2TrainingDevice::Cuda(index) => {
            #[cfg(feature = "sam-cuda")]
            {
                Device::new_cuda(index)
            }
            #[cfg(not(feature = "sam-cuda"))]
            {
                let _ = index;
                sam2_candle_core::bail!(
                    "CUDA SAM training requested, but Blockflow was built without sam-cuda"
                )
            }
        }
    }
}

fn save_variables_with_prefix(
    variables: &sam2_candle_nn::VarMap,
    path: &Path,
    prefix: &str,
) -> CandleResult<()> {
    let data = variables
        .data()
        .lock()
        .map_err(|_| sam2_candle_core::Error::Msg("model variable map is poisoned".into()))?;
    let tensors = data
        .iter()
        .filter(|(name, _)| name.starts_with(prefix))
        .map(|(name, variable)| (name.clone(), variable.as_detached_tensor()))
        .collect::<HashMap<_, _>>();
    if tensors.is_empty() {
        sam2_candle_core::bail!("model has no variables with prefix {prefix}")
    }
    sam2_candle_core::safetensors::save(&tensors, path)
}

fn restore_variables_with_prefix(
    variables: &sam2_candle_nn::VarMap,
    path: &Path,
    prefix: &str,
) -> CandleResult<()> {
    let tensors = sam2_candle_core::safetensors::load(path, &Device::Cpu)?;
    let data = variables
        .data()
        .lock()
        .map_err(|_| sam2_candle_core::Error::Msg("model variable map is poisoned".into()))?;
    let expected = data.keys().filter(|name| name.starts_with(prefix)).count();
    if tensors.len() != expected {
        sam2_candle_core::bail!("decoder overlay tensor count does not match model")
    }
    for (name, variable) in data.iter().filter(|(name, _)| name.starts_with(prefix)) {
        let tensor = tensors
            .get(name)
            .ok_or_else(|| sam2_candle_core::Error::Msg(format!("decoder overlay lacks {name}")))?
            .to_device(variable.device())?;
        variable.set(&tensor)?;
    }
    Ok(())
}

fn validate_checkpoint_metadata(
    metadata: &Sam2TrainingCheckpointMetadata,
    expected_resume_key: &str,
    optimizer: &ResumableAdamW,
) -> CandleResult<()> {
    if metadata.schema_version != SAM2_TRAINING_CHECKPOINT_SCHEMA_VERSION
        || metadata.resume_key != expected_resume_key
    {
        sam2_candle_core::bail!("training checkpoint belongs to a different recipe or schema")
    }
    let same = |left: f64, right: f64| left.to_bits() == right.to_bits();
    if !same(metadata.learning_rate, optimizer.params.lr)
        || !same(metadata.weight_decay, optimizer.params.weight_decay)
        || !same(metadata.beta1, optimizer.params.beta1)
        || !same(metadata.beta2, optimizer.params.beta2)
        || !same(metadata.epsilon, optimizer.params.eps)
        || metadata.parameter_names != optimizer.parameter_names()
    {
        sam2_candle_core::bail!("training checkpoint optimizer settings do not match")
    }
    Ok(())
}

fn boundary_f_score(
    prediction: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    tolerance: usize,
) -> f32 {
    let boundaries = |mask: &[f32]| {
        let mut result = Vec::new();
        for y in 0..height {
            for x in 0..width {
                if mask[y * width + x] < 0.5 {
                    continue;
                }
                let boundary = x == 0
                    || y == 0
                    || x + 1 == width
                    || y + 1 == height
                    || mask[y * width + x - 1] < 0.5
                    || mask[y * width + x + 1] < 0.5
                    || mask[(y - 1) * width + x] < 0.5
                    || mask[(y + 1) * width + x] < 0.5;
                if boundary {
                    result.push((x, y));
                }
            }
        }
        result
    };
    let predicted = boundaries(prediction);
    let expected = boundaries(target);
    if predicted.is_empty() && expected.is_empty() {
        return 1.0;
    }
    if predicted.is_empty() || expected.is_empty() {
        return 0.0;
    }
    let matched = |point: (usize, usize), candidates: &[(usize, usize)]| {
        candidates.iter().any(|candidate| {
            point.0.abs_diff(candidate.0) <= tolerance && point.1.abs_diff(candidate.1) <= tolerance
        })
    };
    let precision = predicted
        .iter()
        .filter(|&&point| matched(point, &expected))
        .count() as f32
        / predicted.len() as f32;
    let recall = expected
        .iter()
        .filter(|&&point| matched(point, &predicted))
        .count() as f32
        / expected.len() as f32;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

impl Sam2DecoderTrainingSession {
    pub fn load_on_device(
        variant: Sam2Variant,
        checkpoint: impl AsRef<Path>,
        mask_decoder: Option<&Path>,
        training_device: Sam2TrainingDevice,
        learning_rate: f64,
        weight_decay: f64,
        loss: Sam2DecoderLossConfig,
    ) -> CandleResult<Self> {
        let device = match training_device {
            Sam2TrainingDevice::Cpu => Device::Cpu,
            Sam2TrainingDevice::Cuda(index) => {
                #[cfg(feature = "sam-cuda")]
                {
                    Device::new_cuda(index)?
                }
                #[cfg(not(feature = "sam-cuda"))]
                {
                    let _ = index;
                    sam2_candle_core::bail!(
                        "CUDA SAM training requested, but Blockflow was built without sam-cuda"
                    )
                }
            }
        };
        let checkpoint = checkpoint.as_ref();
        let encoder = Sam2ImagePredictor::load_sam2_1_on_device(variant, checkpoint, &device)?;
        let trainer = Sam2DecoderTrainer::load_with_mask_decoder(
            variant,
            checkpoint,
            mask_decoder,
            &device,
            learning_rate,
            weight_decay,
            loss,
        )?;
        Ok(Self {
            encoder,
            trainer,
            device,
        })
    }

    pub fn load_cpu(
        variant: Sam2Variant,
        checkpoint: impl AsRef<Path>,
        learning_rate: f64,
        weight_decay: f64,
        loss: Sam2DecoderLossConfig,
    ) -> CandleResult<Self> {
        Self::load_on_device(
            variant,
            checkpoint,
            None,
            Sam2TrainingDevice::Cpu,
            learning_rate,
            weight_decay,
            loss,
        )
    }

    pub fn load_cpu_with_mask_decoder(
        variant: Sam2Variant,
        checkpoint: impl AsRef<Path>,
        mask_decoder: impl AsRef<Path>,
        learning_rate: f64,
        weight_decay: f64,
        loss: Sam2DecoderLossConfig,
    ) -> CandleResult<Self> {
        Self::load_on_device(
            variant,
            checkpoint,
            Some(mask_decoder.as_ref()),
            Sam2TrainingDevice::Cpu,
            learning_rate,
            weight_decay,
            loss,
        )
    }

    pub fn prepare_example(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
        prompts: &PromptSet,
        target: &[bool],
        target_width: usize,
        target_height: usize,
    ) -> CandleResult<Sam2DecoderExample> {
        if target.len() != target_width.saturating_mul(target_height) {
            sam2_candle_core::bail!("SAM2 training target dimensions do not match its pixels")
        }
        let encoded = self.encoder.encode_rgb8(rgb, width, height)?;
        self.prepare_example_from_encoded(&encoded, prompts, target, target_width, target_height)
    }

    pub fn encode_image(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> CandleResult<EncodedImage> {
        self.encoder.encode_rgb8(rgb, width, height)
    }

    pub fn prepare_example_from_encoded(
        &mut self,
        encoded: &EncodedImage,
        prompts: &PromptSet,
        target: &[bool],
        target_width: usize,
        target_height: usize,
    ) -> CandleResult<Sam2DecoderExample> {
        if target.len() != target_width.saturating_mul(target_height) {
            sam2_candle_core::bail!("SAM2 training target dimensions do not match its pixels")
        }
        self.encoder.set_encoded_image(encoded);
        let prompt_batch = prompt_batch(prompts, &self.device)?;
        let inputs = self.encoder.frozen_decoder_inputs(0, &prompt_batch)?;
        let target = Tensor::from_vec(
            target
                .iter()
                .map(|value| if *value { 1_f32 } else { 0_f32 })
                .collect::<Vec<_>>(),
            (1, 1, target_height, target_width),
            &self.device,
        )?;
        Ok(Sam2DecoderExample { inputs, target })
    }

    pub fn step(&mut self, example: &Sam2DecoderExample) -> CandleResult<Sam2DecoderStepMetrics> {
        let (_, _, height, width) = example.target.dims4()?;
        let values = example
            .target
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|value| value >= 0.5)
            .collect::<Vec<_>>();
        self.trainer
            .step_binary_mask(&example.inputs, &values, width, height)
    }

    pub fn evaluate(
        &mut self,
        example: &Sam2DecoderExample,
    ) -> CandleResult<Sam2DecoderStepMetrics> {
        let (_, _, height, width) = example.target.dims4()?;
        let values = example
            .target
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|value| value >= 0.5)
            .collect::<Vec<_>>();
        self.trainer
            .evaluate_binary_mask(&example.inputs, &values, width, height)
    }

    pub fn evaluate_with_prediction(
        &mut self,
        example: &Sam2DecoderExample,
    ) -> CandleResult<(Sam2DecoderStepMetrics, Sam2MaskComparison)> {
        let (_, _, height, width) = example.target.dims4()?;
        let values = example
            .target
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|value| value >= 0.5)
            .collect::<Vec<_>>();
        self.trainer
            .evaluate_binary_mask_with_prediction(&example.inputs, &values, width, height)
    }

    pub fn save_decoder(&self, path: impl AsRef<Path>) -> CandleResult<()> {
        self.trainer.save_decoder(path)
    }

    pub fn steps(&self) -> u64 {
        self.trainer.steps()
    }

    pub fn save_checkpoint(
        &self,
        directory: impl AsRef<Path>,
        resume_key: &str,
        completed_epochs: u32,
    ) -> CandleResult<Sam2TrainingCheckpointMetadata> {
        self.trainer
            .save_checkpoint(directory, resume_key, completed_epochs)
    }

    pub fn restore_checkpoint(
        &mut self,
        directory: impl AsRef<Path>,
        expected_resume_key: &str,
    ) -> CandleResult<Sam2TrainingCheckpointMetadata> {
        self.trainer
            .restore_checkpoint(directory, expected_resume_key)
    }
}

fn prompt_batch(prompts: &PromptSet, device: &Device) -> CandleResult<ImagePromptBatch> {
    let point_coords = (!prompts.points.is_empty())
        .then(|| {
            Tensor::from_vec(
                prompts
                    .points
                    .iter()
                    .flat_map(|point| [point.x, point.y])
                    .collect::<Vec<_>>(),
                (1, prompts.points.len(), 2),
                device,
            )
        })
        .transpose()?;
    let point_labels = (!prompts.points.is_empty())
        .then(|| {
            Tensor::from_vec(
                prompts
                    .points
                    .iter()
                    .map(|point| i64::from(point.positive))
                    .collect::<Vec<_>>(),
                (1, prompts.points.len()),
                device,
            )
        })
        .transpose()?;
    let boxes = prompts
        .bounding_box
        .map(|value| Tensor::new(&[[value.x0, value.y0, value.x1, value.y1]], device))
        .transpose()?;
    let mask_input = prompts
        .mask_input
        .as_ref()
        .map(|(values, height, width)| {
            Tensor::from_vec(values.clone(), (1, 1, *height, *width), device)
        })
        .transpose()?;
    Ok(ImagePromptBatch {
        point_coords,
        point_labels,
        boxes,
        mask_input,
        normalize_coords: true,
    })
}

fn resized_binary_target(
    row_major: &[bool],
    width: usize,
    height: usize,
    output_width: usize,
    output_height: usize,
    device: &Device,
) -> CandleResult<Tensor> {
    if width == 0 || height == 0 || row_major.len() != width.saturating_mul(height) {
        sam2_candle_core::bail!("SAM2 binary target dimensions are invalid")
    }
    let mut resized = Vec::with_capacity(output_width.saturating_mul(output_height));
    for y in 0..output_height {
        let source_y = ((y * height + output_height / 2) / output_height).min(height - 1);
        for x in 0..output_width {
            let source_x = ((x * width + output_width / 2) / output_width).min(width - 1);
            resized.push(if row_major[source_y * width + source_x] {
                1_f32
            } else {
                0_f32
            });
        }
    }
    Tensor::from_vec(resized, (1, 1, output_height, output_width), device)
}

/// Deterministically sample one foreground and one background click from a corrected binary mask.
/// The caller supplies a reproducible recipe/sample/epoch seed; no global RNG or Python runtime is
/// involved. Coordinates are in the input mask's pixel space.
pub fn simulate_point_prompts(
    row_major: &[bool],
    width: usize,
    height: usize,
    seed: u64,
) -> Result<PromptSet, String> {
    if width == 0 || height == 0 || row_major.len() != width.saturating_mul(height) {
        return Err("SAM2 prompt simulation mask dimensions are invalid".into());
    }
    let foreground = row_major.iter().filter(|value| **value).count();
    let background = row_major.len().saturating_sub(foreground);
    if foreground == 0 {
        return Err("SAM2 prompt simulation needs foreground pixels".into());
    }
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let positive_rank = (next() as usize) % foreground;
    let positive = nth_pixel(row_major, true, positive_rank)
        .ok_or_else(|| "SAM2 foreground prompt simulation failed".to_owned())?;
    let mut points = vec![PointPrompt {
        x: (positive % width) as f32 + 0.5,
        y: (positive / width) as f32 + 0.5,
        positive: true,
    }];
    if background > 0 {
        let negative_rank = (next() as usize) % background;
        let negative = nth_pixel(row_major, false, negative_rank)
            .ok_or_else(|| "SAM2 background prompt simulation failed".to_owned())?;
        points.push(PointPrompt {
            x: (negative % width) as f32 + 0.5,
            y: (negative / width) as f32 + 0.5,
            positive: false,
        });
    }
    Ok(PromptSet {
        points,
        bounding_box: None,
        mask_input: None,
    })
}

fn nth_pixel(mask: &[bool], value: bool, rank: usize) -> Option<usize> {
    mask.iter()
        .enumerate()
        .filter(|(_, pixel)| **pixel == value)
        .nth(rank)
        .map(|(index, _)| index)
}

#[cfg(test)]
mod optimizer_tests {
    use sam2_candle_nn::{AdamW, Init, Optimizer, VarMap};

    use super::*;

    #[test]
    fn resumable_adamw_matches_candle_updates() -> CandleResult<()> {
        let device = Device::Cpu;
        let custom_variables = VarMap::new();
        let candle_variables = VarMap::new();
        let custom_value = custom_variables.get(
            (3,),
            "weight",
            Init::Const(2.0),
            sam2_candle_core::DType::F32,
            &device,
        )?;
        let candle_value = candle_variables.get(
            (3,),
            "weight",
            Init::Const(2.0),
            sam2_candle_core::DType::F32,
            &device,
        )?;
        let parameters = ParamsAdamW {
            lr: 0.025,
            weight_decay: 0.01,
            ..Default::default()
        };
        let mut custom = ResumableAdamW::new(&custom_variables, parameters.clone())?;
        let mut candle = AdamW::new(candle_variables.all_vars(), parameters)?;
        for _ in 0..3 {
            let custom_loss = custom_value.affine(1.0, -4.0)?.sqr()?.sum_all()?;
            let candle_loss = candle_value.affine(1.0, -4.0)?.sqr()?.sum_all()?;
            custom.step(&custom_loss.backward()?)?;
            candle.backward_step(&candle_loss)?;
        }
        assert_eq!(
            custom_value.to_vec1::<f32>()?,
            candle_value.to_vec1::<f32>()?
        );
        Ok(())
    }
}
