//! SAM2 model ownership at the workflow boundary.
//!
//! The viewer supplies RGB pixels and plain prompts. This adapter owns the cached image embedding
//! and returns ordinary row-major masks, keeping Candle tensors and checkpoint details out of UI
//! and job protocol crates.

use std::path::Path;

use sam1_candle_transformers::models::segment_anything::sam as sam1;
use sam2_candle_core::{DType, Device, IndexOp, Tensor};
use sam2_candle_nn::{VarBuilder, VarMap};
use sam2_rs::{
    AutomaticMaskConfig, AutomaticMaskOutputMode, AutomaticMaskSegmentation, ImagePrediction,
    InteractiveVideoOutput, InteractiveVideoSession, Sam2AutomaticMaskGenerator,
    Sam2ImagePredictor, Sam2InteractiveVideoPredictor, VideoPredictorConfig,
};
pub use sam2_rs::{BoxPrompt, PointPrompt, Sam2Variant};

#[derive(Clone, Debug, Default)]
pub struct PromptSet {
    pub points: Vec<PointPrompt>,
    pub bounding_box: Option<BoxPrompt>,
    /// One low-resolution logit plane `(values, height, width)` used to refine a prior prediction
    /// or a drawn mask.
    pub mask_input: Option<(Vec<f32>, usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct Prediction {
    pub width: usize,
    pub height: usize,
    pub masks: Vec<Vec<f32>>,
    pub scores: Vec<f32>,
}

/// Plain RGB input owned by the caller. Ordered microscopy z slices use the same temporal axis as
/// ordinary video without exposing Candle tensors outside this adapter.
#[derive(Clone, Copy, Debug)]
pub struct VideoRgbFrame<'a> {
    pub rgb: &'a [u8],
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Debug)]
pub struct VideoMaskFrame {
    pub frame_index: usize,
    pub width: usize,
    pub height: usize,
    pub object_ids: Vec<u32>,
    /// One row-major binary mask per object ID.
    pub masks: Vec<Vec<bool>>,
}

pub struct VideoSession {
    session: InteractiveVideoSession,
    width: usize,
    height: usize,
    frames: usize,
}

pub struct VideoPredictor {
    predictor: Sam2InteractiveVideoPredictor,
}

impl VideoPredictor {
    pub fn load(variant: Sam2Variant, checkpoint: impl AsRef<Path>) -> Result<Self, String> {
        Sam2InteractiveVideoPredictor::load_sam2_1(
            variant,
            checkpoint,
            VideoPredictorConfig::default(),
        )
        .map(|predictor| Self { predictor })
        .map_err(|error| error.to_string())
    }

    pub fn init(&self, frames: &[VideoRgbFrame<'_>]) -> Result<VideoSession, String> {
        let first = frames
            .first()
            .ok_or_else(|| "SAM video/z stack is empty".to_owned())?;
        let expected = first.width.saturating_mul(first.height).saturating_mul(3);
        if first.rgb.len() != expected
            || frames.iter().any(|frame| {
                frame.width != first.width
                    || frame.height != first.height
                    || frame.rgb.len() != expected
            })
        {
            return Err("SAM video/z slices must be equally sized RGB8 frames".into());
        }
        let inputs = frames
            .iter()
            .map(|frame| (frame.rgb, frame.width, frame.height))
            .collect::<Vec<_>>();
        let session = self
            .predictor
            .init_state(&inputs)
            .map_err(|error| error.to_string())?;
        Ok(VideoSession {
            session,
            width: first.width,
            height: first.height,
            frames: frames.len(),
        })
    }

    pub fn correct_points_or_box(
        &self,
        session: &mut VideoSession,
        frame_index: usize,
        object_id: u32,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        clear_old_points: bool,
    ) -> Result<VideoMaskFrame, String> {
        self.predictor
            .add_new_points_or_box(
                &mut session.session,
                frame_index,
                object_id,
                points,
                bounding_box,
                clear_old_points,
            )
            .map_err(|error| error.to_string())
            .and_then(|output| decode_video_output(output, session.width, session.height))
    }

    /// Correct one object from a full-resolution row-major binary mask. The tensor remains an
    /// implementation detail of this adapter so viewer and workflow callers share a plain Rust
    /// contract and cannot disagree about Candle versions or device placement.
    pub fn correct_mask(
        &self,
        session: &mut VideoSession,
        frame_index: usize,
        object_id: u32,
        row_major: &[bool],
        width: usize,
        height: usize,
    ) -> Result<VideoMaskFrame, String> {
        if width != session.width
            || height != session.height
            || row_major.len() != width.saturating_mul(height)
        {
            return Err("SAM correction mask does not match the z-stack frame".into());
        }
        let values = row_major
            .iter()
            .map(|value| if *value { 1_f32 } else { 0_f32 })
            .collect::<Vec<_>>();
        let mask = sam2_candle_core::Tensor::from_vec(
            values,
            (height, width),
            &sam2_candle_core::Device::Cpu,
        )
        .map_err(|error| error.to_string())?;
        self.predictor
            .add_new_mask(&mut session.session, frame_index, object_id, &mask)
            .map_err(|error| error.to_string())
            .and_then(|output| decode_video_output(output, session.width, session.height))
    }

    /// Propagate exactly one requested slice. Calling this repeatedly lets a workflow report
    /// progress and honor cancellation between slices while retaining SAM2's temporal memory.
    pub fn propagate_frame(
        &self,
        session: &mut VideoSession,
        frame_index: usize,
        reverse: bool,
    ) -> Result<VideoMaskFrame, String> {
        if frame_index >= session.frames {
            return Err("SAM propagation frame is outside the z stack".into());
        }
        let mut outputs = self
            .predictor
            .propagate_in_video(&mut session.session, Some(frame_index), Some(0), reverse)
            .map_err(|error| error.to_string())?;
        let output = outputs
            .pop()
            .filter(|output| output.frame_index == frame_index)
            .ok_or_else(|| "SAM did not return the requested propagation frame".to_owned())?;
        decode_video_output(output, session.width, session.height)
    }
}

fn decode_video_output(
    output: InteractiveVideoOutput,
    width: usize,
    height: usize,
) -> Result<VideoMaskFrame, String> {
    let (objects, channels, mask_height, mask_width) =
        output.masks.dims4().map_err(|error| error.to_string())?;
    if objects != output.object_ids.len()
        || channels != 1
        || mask_width != width
        || mask_height != height
    {
        return Err(format!(
            "SAM video returned mask shape {:?} for {} objects at {width}x{height}",
            output.masks.dims(),
            output.object_ids.len()
        ));
    }
    let plane = width.saturating_mul(height);
    let values = output
        .masks
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| error.to_string())?;
    let masks = values
        .chunks_exact(plane)
        .map(|mask| mask.iter().map(|value| *value > 0.0).collect())
        .collect();
    Ok(VideoMaskFrame {
        frame_index: output.frame_index,
        width,
        height,
        object_ids: output.object_ids,
        masks,
    })
}

#[derive(Clone, Debug)]
pub struct AutomaticInstance {
    pub width: usize,
    pub height: usize,
    pub row_major: Vec<bool>,
    pub bbox_xywh: [f32; 4],
    pub area: usize,
    pub predicted_iou: f32,
    pub stability_score: f32,
    pub seed_xy: [f32; 2],
}

pub struct AutomaticGenerator {
    generator: Sam2AutomaticMaskGenerator,
}

impl AutomaticGenerator {
    pub fn load(
        variant: Sam2Variant,
        checkpoint: impl AsRef<Path>,
        points_per_side: usize,
        points_per_batch: usize,
        minimum_area: usize,
    ) -> Result<Self, String> {
        let config = AutomaticMaskConfig {
            points_per_side: Some(points_per_side),
            points_per_batch,
            min_mask_region_area: minimum_area,
            output_mode: AutomaticMaskOutputMode::BinaryMask,
            ..Default::default()
        };
        Sam2AutomaticMaskGenerator::load_sam2_1(variant, checkpoint, config)
            .map(|generator| Self { generator })
            .map_err(|error| error.to_string())
    }

    pub fn generate(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<AutomaticInstance>, String> {
        self.generator
            .generate_rgb8(rgb, width, height)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|mask| match mask.segmentation {
                AutomaticMaskSegmentation::Binary {
                    width,
                    height,
                    row_major,
                } => Ok(AutomaticInstance {
                    width,
                    height,
                    row_major,
                    bbox_xywh: mask.bbox,
                    area: mask.area,
                    predicted_iou: mask.predicted_iou,
                    stability_score: mask.stability_score,
                    seed_xy: mask.point,
                }),
                _ => Err("SAM automatic generator returned a non-binary mask".into()),
            })
            .collect()
    }
}

pub struct CachedPredictor {
    predictor: Sam2ImagePredictor,
    image_size: Option<(usize, usize)>,
    embedding_generation: u64,
}

/// Native SAM1 ViT-B adapter for original and micro-sam fine-tuned checkpoints. Training policy
/// stays in Blockflow; this adapter only owns faithful image preprocessing and inference tensors.
pub struct Sam1CachedPredictor {
    predictor: sam1::Sam,
    image_embeddings: Option<Tensor>,
    original_size: Option<(usize, usize)>,
    resized_size: Option<(usize, usize)>,
    embedding_generation: u64,
}

fn sam1_checkpoint_builder(
    checkpoint: &Path,
    device: &Device,
) -> Result<VarBuilder<'static>, String> {
    if checkpoint.extension().and_then(|value| value.to_str()) == Some("safetensors") {
        return unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, device) }
            .map_err(|error| error.to_string());
    }
    let select = |builder: VarBuilder<'static>| {
        if builder.contains_tensor("image_encoder.patch_embed.proj.weight") {
            builder
        } else {
            builder.rename_f(|name| format!("sam.{name}"))
        }
    };
    match VarBuilder::from_pth(checkpoint, DType::F32, device) {
        Ok(builder)
            if builder.contains_tensor("image_encoder.patch_embed.proj.weight")
                || builder.contains_tensor("sam.image_encoder.patch_embed.proj.weight") =>
        {
            Ok(select(builder))
        }
        Ok(_) | Err(_) => {
            VarBuilder::from_pth_with_state(checkpoint, DType::F32, "model_state", device)
                .map(select)
                .map_err(|error| format!("load SAM1 checkpoint state: {error}"))
        }
    }
}

/// Build a SAM1 ViT-B model backed by trainable variables and copy an original or micro-sam
/// checkpoint into it. Blockflow's trainer filters the returned map to decoder variables, leaving
/// this adapter responsible only for faithful checkpoint naming and architecture construction.
pub(crate) fn load_trainable_sam1_vit_b(
    checkpoint: &Path,
    device: &Device,
) -> Result<(sam1::Sam, VarMap), String> {
    let source = sam1_checkpoint_builder(checkpoint, device)?;
    let variables = VarMap::new();
    let model = sam1::Sam::new(
        768,
        12,
        12,
        &[2, 5, 8, 11],
        VarBuilder::from_varmap(&variables, DType::F32, device),
    )
    .map_err(|error| error.to_string())?;
    let data = variables
        .data()
        .lock()
        .map_err(|_| "SAM1 variable map is poisoned".to_owned())?;
    for (name, variable) in data.iter() {
        let tensor = source
            .get(variable.shape().clone(), name)
            .map_err(|error| format!("load SAM1 tensor {name}: {error}"))?;
        variable
            .set(&tensor)
            .map_err(|error| format!("install SAM1 tensor {name}: {error}"))?;
    }
    drop(data);
    Ok((model, variables))
}

impl Sam1CachedPredictor {
    pub fn load_vit_b(checkpoint: impl AsRef<Path>) -> Result<Self, String> {
        let checkpoint = checkpoint.as_ref();
        let device = Device::Cpu;
        let build = |builder: VarBuilder<'_>| sam1::Sam::new(768, 12, 12, &[2, 5, 8, 11], builder);
        let predictor = build(sam1_checkpoint_builder(checkpoint, &device)?)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            predictor,
            image_embeddings: None,
            original_size: None,
            resized_size: None,
            embedding_generation: 0,
        })
    }

    pub fn load_vit_b_with_mask_decoder(
        checkpoint: impl AsRef<Path>,
        decoder_checkpoint: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let device = Device::Cpu;
        let (predictor, variables) = load_trainable_sam1_vit_b(checkpoint.as_ref(), &device)?;
        let tensors = sam2_candle_core::safetensors::load(decoder_checkpoint.as_ref(), &device)
            .map_err(|error| error.to_string())?;
        let data = variables
            .data()
            .lock()
            .map_err(|_| "SAM1 variable map is poisoned".to_owned())?;
        let expected = data
            .keys()
            .filter(|name| name.starts_with("mask_decoder."))
            .count();
        if tensors.len() != expected {
            return Err("SAM1 decoder overlay tensor count does not match ViT-B".into());
        }
        for (name, variable) in data
            .iter()
            .filter(|(name, _)| name.starts_with("mask_decoder."))
        {
            let tensor = tensors
                .get(name)
                .ok_or_else(|| format!("SAM1 decoder overlay lacks {name}"))?;
            variable
                .set(tensor)
                .map_err(|error| format!("install SAM1 decoder tensor {name}: {error}"))?;
        }
        drop(data);
        Ok(Self {
            predictor,
            image_embeddings: None,
            original_size: None,
            resized_size: None,
            embedding_generation: 0,
        })
    }

    pub fn set_image(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<u64, String> {
        if width == 0 || height == 0 || rgb.len() != width.saturating_mul(height).saturating_mul(3)
        {
            return Err("SAM1 input is not a valid interleaved RGB8 image".into());
        }
        let scale = sam1::IMAGE_SIZE as f64 / width.max(height) as f64;
        let resized_width = ((width as f64 * scale) + 0.5).floor().max(1.0) as usize;
        let resized_height = ((height as f64 * scale) + 0.5).floor().max(1.0) as usize;
        let image = Tensor::from_vec(rgb.to_vec(), (height, width, 3), &Device::Cpu)
            .and_then(|value| value.permute((2, 0, 1)))
            .and_then(|value| value.to_dtype(DType::F32))
            .and_then(|value| value.unsqueeze(0))
            .and_then(|value| value.upsample_bilinear2d(resized_height, resized_width, false))
            .and_then(|value| value.squeeze(0))
            .map_err(|error| error.to_string())?;
        self.image_embeddings = Some(
            self.predictor
                .embeddings(&image)
                .map_err(|error| error.to_string())?,
        );
        self.original_size = Some((width, height));
        self.resized_size = Some((resized_width, resized_height));
        self.embedding_generation = self.embedding_generation.wrapping_add(1).max(1);
        Ok(self.embedding_generation)
    }

    pub fn predict(&self, prompts: &PromptSet, multimask: bool) -> Result<Prediction, String> {
        let embeddings = self
            .image_embeddings
            .as_ref()
            .ok_or("set an image before decoding SAM1 prompts")?;
        let (width, height) = self.original_size.ok_or("SAM1 image size is absent")?;
        let (resized_width, resized_height) = self
            .resized_size
            .ok_or("SAM1 resized image size is absent")?;
        let points = prompts
            .points
            .iter()
            .map(|point| {
                (
                    f64::from(point.x) / width as f64,
                    f64::from(point.y) / height as f64,
                    point.positive,
                )
            })
            .collect::<Vec<_>>();
        let bounding_box = prompts.bounding_box.map(|bounds| {
            [
                f64::from(bounds.x0) / width as f64,
                f64::from(bounds.y0) / height as f64,
                f64::from(bounds.x1) / width as f64,
                f64::from(bounds.y1) / height as f64,
            ]
        });
        let mask_input = prompts
            .mask_input
            .as_ref()
            .map(|(values, mask_height, mask_width)| {
                if (*mask_width, *mask_height) != (256, 256)
                    || values.len() != mask_width.saturating_mul(*mask_height)
                {
                    return Err("SAM1 dense mask prompt must be 256x256".to_owned());
                }
                Tensor::from_vec(
                    values.clone(),
                    (1, 1, *mask_height, *mask_width),
                    embeddings.device(),
                )
                .map_err(|error| error.to_string())
            })
            .transpose()?;
        let (low_resolution, scores) = self
            .predictor
            .forward_for_embeddings_with_prompts(
                embeddings,
                resized_height,
                resized_width,
                &points,
                bounding_box,
                mask_input.as_ref(),
                multimask,
            )
            .map_err(|error| error.to_string())?;
        let masks = low_resolution
            .upsample_bilinear2d(sam1::IMAGE_SIZE, sam1::IMAGE_SIZE, false)
            .and_then(|value| value.i((0, .., ..resized_height, ..resized_width)))
            .and_then(|value| value.unsqueeze(0))
            .and_then(|value| value.upsample_bilinear2d(height, width, false))
            .and_then(|value| value.squeeze(0))
            .map_err(|error| error.to_string())?;
        let candidates = masks.dim(0).map_err(|error| error.to_string())?;
        let flat = masks
            .flatten_all()
            .and_then(|value| value.to_vec1::<f32>())
            .map_err(|error| error.to_string())?;
        let plane = width.saturating_mul(height);
        Ok(Prediction {
            width,
            height,
            masks: flat
                .chunks_exact(plane)
                .take(candidates)
                .map(<[f32]>::to_vec)
                .collect(),
            scores: scores
                .flatten_all()
                .and_then(|value| value.to_vec1::<f32>())
                .map_err(|error| error.to_string())?,
        })
    }

    pub fn embedding_generation(&self) -> Option<u64> {
        self.original_size.map(|_| self.embedding_generation)
    }
}

impl CachedPredictor {
    pub fn load(variant: Sam2Variant, checkpoint: impl AsRef<Path>) -> Result<Self, String> {
        Sam2ImagePredictor::load_sam2_1(variant, checkpoint)
            .map(|predictor| Self {
                predictor,
                image_size: None,
                embedding_generation: 0,
            })
            .map_err(|error| error.to_string())
    }

    pub fn load_with_mask_decoder(
        variant: Sam2Variant,
        parent_checkpoint: impl AsRef<Path>,
        decoder_checkpoint: impl AsRef<Path>,
    ) -> Result<Self, String> {
        Sam2ImagePredictor::load_sam2_1_with_mask_decoder_on_device(
            variant,
            parent_checkpoint,
            decoder_checkpoint,
            &sam2_candle_core::Device::Cpu,
        )
        .map(|predictor| Self {
            predictor,
            image_size: None,
            embedding_generation: 0,
        })
        .map_err(|error| error.to_string())
    }

    /// Encode one interleaved RGB8 image. Prompt corrections reuse this embedding until another
    /// image is installed.
    pub fn set_image(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<u64, String> {
        if rgb.len() != width.saturating_mul(height).saturating_mul(3) {
            return Err(format!(
                "RGB input has {} bytes, expected {width}x{height}x3",
                rgb.len()
            ));
        }
        self.predictor
            .set_image(rgb, width, height)
            .map_err(|error| error.to_string())?;
        self.image_size = Some((width, height));
        self.embedding_generation = self.embedding_generation.wrapping_add(1).max(1);
        Ok(self.embedding_generation)
    }

    pub fn embedding_generation(&self) -> Option<u64> {
        self.image_size.map(|_| self.embedding_generation)
    }

    pub fn predict(&self, prompts: &PromptSet, multimask: bool) -> Result<Prediction, String> {
        let (width, height) = self
            .image_size
            .ok_or_else(|| "set an image before decoding SAM prompts".to_owned())?;
        let mask = prompts
            .mask_input
            .as_ref()
            .map(|(values, height, width)| (values.as_slice(), *height, *width));
        let prediction = self
            .predictor
            .predict_cached_prompts(&prompts.points, prompts.bounding_box, mask, multimask, true)
            .map_err(|error| error.to_string())?;
        prediction_to_rows(prediction, width, height)
    }
}

fn prediction_to_rows(
    prediction: ImagePrediction,
    expected_width: usize,
    expected_height: usize,
) -> Result<Prediction, String> {
    let dimensions = prediction
        .masks
        .dims4()
        .map_err(|error| error.to_string())?;
    let (batch, candidates, height, width) = dimensions;
    if batch != 1 || width != expected_width || height != expected_height {
        return Err(format!(
            "SAM returned mask shape {dimensions:?}, expected [1,N,{expected_height},{expected_width}]"
        ));
    }
    let flat = prediction
        .masks
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| error.to_string())?;
    let plane = width.saturating_mul(height);
    let masks = flat
        .chunks_exact(plane)
        .take(candidates)
        .map(<[f32]>::to_vec)
        .collect();
    let scores = prediction
        .low_res
        .iou_predictions
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| error.to_string())?;
    Ok(Prediction {
        width,
        height,
        masks,
        scores,
    })
}

#[derive(Clone, Copy, Debug)]
pub enum MaskRasterShape {
    Brush(f32),
    Polygon,
}

/// Rasterize ordered level-zero brush and polygon edits into SAM's 256×256 mask prompt.
/// Later edits win, and erase returns a pixel to the neutral logit.
pub fn rasterize_mask_strokes(
    image_size: [usize; 2],
    strokes: &[(MaskRasterShape, Option<bool>, Vec<[f32; 2]>)],
) -> Vec<f32> {
    const SIDE: usize = 256;
    let mut mask = vec![0.0_f32; SIDE * SIDE];
    let scale = [
        SIDE as f32 / image_size[0].max(1) as f32,
        SIDE as f32 / image_size[1].max(1) as f32,
    ];
    for (shape, positive, points) in strokes {
        let value = match positive {
            Some(true) => 8.0,
            Some(false) => -8.0,
            None => 0.0,
        };
        if matches!(shape, MaskRasterShape::Polygon) {
            paint_polygon(&mut mask, [SIDE, SIDE], points, scale, value);
            continue;
        }
        let MaskRasterShape::Brush(radius) = shape else {
            unreachable!()
        };
        let radius = (*radius * scale[0].min(scale[1])).max(0.5);
        if let Some(point) = points.first().filter(|_| points.len() == 1) {
            paint_circle(
                &mut mask,
                [SIDE, SIDE],
                [point[0] * scale[0], point[1] * scale[1]],
                radius,
                value,
            );
        }
        for segment in points.windows(2) {
            let start = segment[0];
            let end = *segment.last().unwrap_or(&start);
            let distance = (end[0] - start[0]).hypot(end[1] - start[1]);
            let steps = (distance * scale[0].max(scale[1])).ceil().max(1.0) as usize;
            for step in 0..=steps {
                let t = step as f32 / steps as f32;
                let centre = [
                    (start[0] + (end[0] - start[0]) * t) * scale[0],
                    (start[1] + (end[1] - start[1]) * t) * scale[1],
                ];
                paint_circle(&mut mask, [SIDE, SIDE], centre, radius, value);
            }
        }
    }
    mask
}

/// Apply level-zero brush/polygon corrections to an existing prediction at the predictor frame
/// resolution. Erase restores the unedited prediction beneath that stroke.
pub fn apply_binary_mask_strokes(
    base: &[bool],
    target_size: [usize; 2],
    source_size: [usize; 2],
    strokes: &[(MaskRasterShape, Option<bool>, Vec<[f32; 2]>)],
) -> Result<Vec<bool>, String> {
    let [width, height] = target_size;
    if width == 0
        || height == 0
        || source_size.contains(&0)
        || base.len() != width.saturating_mul(height)
    {
        return Err("SAM correction mask dimensions are invalid".into());
    }
    let original = base.to_vec();
    let mut corrected = original.clone();
    let scale = [
        width as f32 / source_size[0] as f32,
        height as f32 / source_size[1] as f32,
    ];
    for (shape, value, points) in strokes {
        let mut footprint = vec![0.0_f32; width * height];
        match shape {
            MaskRasterShape::Polygon => {
                paint_polygon(&mut footprint, target_size, points, scale, 1.0)
            }
            MaskRasterShape::Brush(radius) => {
                let radius = (*radius * scale[0].min(scale[1])).max(0.5);
                if let Some(point) = points.first().filter(|_| points.len() == 1) {
                    paint_circle(
                        &mut footprint,
                        target_size,
                        [point[0] * scale[0], point[1] * scale[1]],
                        radius,
                        1.0,
                    );
                }
                for segment in points.windows(2) {
                    let start = segment[0];
                    let end = segment[1];
                    let distance = (end[0] - start[0]).hypot(end[1] - start[1]);
                    let steps = (distance * scale[0].max(scale[1])).ceil().max(1.0) as usize;
                    for step in 0..=steps {
                        let t = step as f32 / steps as f32;
                        paint_circle(
                            &mut footprint,
                            target_size,
                            [
                                (start[0] + (end[0] - start[0]) * t) * scale[0],
                                (start[1] + (end[1] - start[1]) * t) * scale[1],
                            ],
                            radius,
                            1.0,
                        );
                    }
                }
            }
        }
        for (index, _) in footprint
            .iter()
            .enumerate()
            .filter(|(_, value)| **value > 0.0)
        {
            corrected[index] = value.unwrap_or(original[index]);
        }
    }
    Ok(corrected)
}

fn paint_polygon(
    mask: &mut [f32],
    size: [usize; 2],
    points: &[[f32; 2]],
    scale: [f32; 2],
    value: f32,
) {
    let [width, height] = size;
    if points.len() < 3 {
        return;
    }
    let polygon = points
        .iter()
        .map(|point| [point[0] * scale[0], point[1] * scale[1]])
        .collect::<Vec<_>>();
    let x0 = polygon
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, (width - 1) as f32) as usize;
    let x1 = polygon
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, (width - 1) as f32) as usize;
    let y0 = polygon
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, (height - 1) as f32) as usize;
    let y1 = polygon
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, (height - 1) as f32) as usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let sample = [x as f32 + 0.5, y as f32 + 0.5];
            let mut inside = false;
            let mut previous = polygon.len() - 1;
            for current in 0..polygon.len() {
                let a = polygon[current];
                let b = polygon[previous];
                if ((a[1] > sample[1]) != (b[1] > sample[1]))
                    && sample[0] < (b[0] - a[0]) * (sample[1] - a[1]) / (b[1] - a[1]) + a[0]
                {
                    inside = !inside;
                }
                previous = current;
            }
            if inside {
                mask[y * width + x] = value;
            }
        }
    }
}

fn paint_circle(mask: &mut [f32], size: [usize; 2], centre: [f32; 2], radius: f32, value: f32) {
    let [width, height] = size;
    let x0 = (centre[0] - radius).floor().max(0.0) as usize;
    let y0 = (centre[1] - radius).floor().max(0.0) as usize;
    let x1 = (centre[0] + radius).ceil().min((width - 1) as f32) as usize;
    let y1 = (centre[1] + radius).ceil().min((height - 1) as f32) as usize;
    let radius_squared = radius * radius;
    for y in y0..=y1 {
        for x in x0..=x1 {
            if (x as f32 - centre[0]).powi(2) + (y as f32 - centre[1]).powi(2) <= radius_squared {
                mask[y * width + x] = value;
            }
        }
    }
}
