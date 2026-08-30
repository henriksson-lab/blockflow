//! YOLO prediction over a local OME-Zarr through `blockflow`.
//!
//! The older `predict` command reads TIFF tiles directly. This module is the
//! Blockflow path: one fragment phase reads haloed Zarr blocks, runs YOLO, owns
//! detections by centre-in-core, and emits one table row per spot.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::op::{Chain, SourceInput};
use crate::strategy::{execute_phases, Hints, Workflow};
use crate::table::{Column, ColumnType, Row, RowBuilder, Schema, Table, Value};
use crate::{
    env::Environment, fragment_phase, AttachedImage, BlockBuf, BlockGrid, BlockOutput, BlockView,
    Coverage, Decomposition, Dtype, FragmentOp, FragmentOutput, ImageId, Lifecycle, PhaseWork,
    Reach, Region, SourceBlocks, ZarrEnvironment,
};
use anyhow::{bail, Context, Result};
use burn::prelude::*;
#[cfg(feature = "yolo-cuda")]
use burn::tensor::DeviceIndex;
#[cfg(not(feature = "yolo-cuda"))]
use burn::tensor::DeviceKind;
use image::{DynamicImage, RgbImage};

const STREAM: &str = "yolo_detections";

pub struct PredictConfig {
    pub zarr: PathBuf,
    pub level: usize,
    pub weights: PathBuf,
    pub config: PathBuf,
    pub channels: Vec<usize>,
    pub normalize_range: (f64, f64),
    pub block: usize,
    pub halo: usize,
    pub conf_threshold: f32,
    pub input_size: u32,
    pub concurrency: usize,
    pub min_separation: f64,
    pub out: PathBuf,
    pub summary: PathBuf,
    pub work: PathBuf,
}

pub fn run(config: &PredictConfig) -> Result<()> {
    if config.channels.is_empty() || config.channels.len() > 3 {
        bail!("--channels wants 1-3 channel indices");
    }

    let level_dir = config.zarr.join(config.level.to_string());
    if !level_dir.is_dir() {
        bail!(
            "{} is not a directory. `--zarr` wants the OME-Zarr root.",
            level_dir.display()
        );
    }
    let (height, width) = level_extent(&level_dir)?;
    let volume = [1, height, width];
    println!(
        "YOLO over level {}: {} x {}, channels {:?}",
        config.level, width, height, config.channels
    );

    let images: Vec<AttachedImage> = config
        .channels
        .iter()
        .map(|channel| AttachedImage::at(&level_dir).plane(*channel, [height, width]))
        .collect();
    let env = ZarrEnvironment::attach(&config.work, &images)
        .map_err(|error| anyhow::anyhow!("attaching {}: {error}", level_dir.display()))?;
    let dtype = env
        .image_dtype(0)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let detector = Arc::new(YoloBlockDetector::new(config, dtype)?);
    let grid = BlockGrid::new(volume, [1, config.block, config.block])
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let blocks = grid.n_blocks();
    let phase = fragment_phase(detector.as_ref(), grid.clone())
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let plan = Decomposition {
        volume,
        dtype,
        phases: vec![phase],
        chain_reach: [0, 0, 0],
    };
    plan.check().map_err(|error| anyhow::anyhow!("{error}"))?;

    let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, dtype);
    let hints = Hints {
        concurrency: config.concurrency.max(1),
        ..Hints::default()
    };
    println!(
        "{blocks} block(s), block {}, halo {}, {} concurrent",
        config.block, config.halo, hints.concurrency
    );

    let started = std::time::Instant::now();
    let stats = execute_phases(
        "yolo-slide",
        &workflow,
        &plan,
        &hints,
        &env,
        &[],
        &[PhaseWork::Fragments(detector.as_ref())],
    )
    .map_err(|error| anyhow::anyhow!("the run: {error}"))?;
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "Ran {blocks} block(s) in {elapsed:.1}s, {} reads, {:.1} Mpx read",
        stats.reads,
        stats.read_voxels as f64 / 1e6
    );

    let schema = detector.schema_data()?;
    let mut table = Table::new(volume, schema).map_err(|error| anyhow::anyhow!("{error}"))?;
    for core in grid.cores() {
        let bytes = env
            .read_sidecar(STREAM, 0, core.index)
            .map_err(|error| anyhow::anyhow!("{error}"))?
            .with_context(|| format!("block {:?} wrote no detection blob", core.index))?;
        table
            .write(core.index, &bytes)
            .map_err(|error| anyhow::anyhow!("{error}"))?;
    }
    table.seal().map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut detections: Vec<Detection> = table
        .query(&Region::whole(&volume))
        .map_err(|error| anyhow::anyhow!("{error}"))?
        .iter()
        .map(Detection::from_row)
        .collect();
    let before = detections.len();
    detections = deduplicate(detections, config.min_separation);
    let merged = before.saturating_sub(detections.len());
    if merged > 0 {
        println!("{merged} duplicate detection(s) merged by centre distance");
    }

    write_outputs(config, &detections)?;
    println!(
        "{} detection(s) written to {}",
        detections.len(),
        config.out.display()
    );
    Ok(())
}

struct YoloBlockDetector {
    model: Mutex<yolov11::model::model::YOLO>,
    device: Device,
    channels: usize,
    range: (f64, f64),
    halo: [usize; 3],
    conf_threshold: f32,
    input_size: u32,
    source_dtype: Dtype,
}

impl YoloBlockDetector {
    fn new(config: &PredictConfig, source_dtype: Dtype) -> Result<Self> {
        #[cfg(feature = "yolo-cuda")]
        let device = Device::cuda(DeviceIndex::Default);
        #[cfg(not(feature = "yolo-cuda"))]
        let device = Device::wgpu(DeviceKind::DefaultDevice);
        let yolo_config = yolov11::train::config::Config::load(&config.config)?;
        let num_classes = yolo_config.num_classes();
        let mut model = yolov11::model::model::yolo_v11_n(num_classes, &device);

        if config.weights.extension().and_then(|value| value.to_str()) == Some("safetensors") {
            use burn_store::ModuleSnapshot;
            let weights = config.weights.to_string_lossy();
            let mut store = burn_store::SafetensorsStore::from_file(weights.as_ref());
            model
                .load_from(&mut store)
                .map_err(|error| anyhow::anyhow!("loading safetensors weights: {error}"))?;
        } else if config
            .weights
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|ext| ext == "pt" || ext == "pth")
        {
            bail!("Direct .pt loading is not supported. Convert to .safetensors first.");
        } else {
            model = model
                .try_load_file(&config.weights)
                .map_err(|error| anyhow::anyhow!("loading weights: {error}"))?;
        }

        Ok(Self {
            model: Mutex::new(model),
            device,
            channels: config.channels.len(),
            range: config.normalize_range,
            halo: [0, config.halo, config.halo],
            conf_threshold: config.conf_threshold,
            input_size: config.input_size,
            source_dtype,
        })
    }

    fn schema_data(&self) -> crate::error::Result<Schema> {
        Schema::new(vec![
            Column::new("id", ColumnType::U64),
            Column::new("x", ColumnType::F64),
            Column::new("y", ColumnType::F64),
            Column::new("confidence", ColumnType::F64),
            Column::new("class", ColumnType::U64),
        ])
    }

    fn schema(&self) -> crate::error::Result<Arc<Schema>> {
        Ok(Arc::new(self.schema_data()?))
    }
}

impl FragmentOp for YoloBlockDetector {
    fn name(&self) -> &'static str {
        "yolo"
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.halo[axis]
    }

    fn cost_per_voxel(&self) -> f64 {
        3.0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        (1..self.channels)
            .map(|which| {
                SourceInput::new(ImageId::supplied(which - 1), Reach::symmetric(self.halo))
                    .holding(self.source_dtype)
            })
            .collect()
    }

    fn seam_fold(&self) -> Option<crate::SeamFold> {
        Some(crate::SeamFold::PerBlock)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            STREAM.to_string(),
            Lifecycle::Persistent,
            Coverage::EveryBlock,
        )
        // One row per detection, and a block cannot detect more objects than it
        //             // holds voxels.
        .sized(match self.schema() {
            Ok(schema) => crate::fragment::SidecarSize::row_table(&schema, 1),
            Err(_) => crate::fragment::SidecarSize::Unstated,
        })]
    }

    fn apply(&self, at: &BlockView<'_>) -> crate::error::Result<BlockOutput> {
        self.apply_with(at, SourceBlocks::none())
    }

    fn apply_with(
        &self,
        at: &BlockView<'_>,
        sources: SourceBlocks<'_>,
    ) -> crate::error::Result<BlockOutput> {
        let schema = self.schema()?;
        let mut rows = RowBuilder::new(schema);
        let BlockBuf::Array(primary) = at.pixels()? else {
            return Ok(BlockOutput::fragment(STREAM.to_string(), rows.encode()));
        };
        let mut channels = vec![primary.widened()];
        for which in 1..self.channels {
            let image = ImageId::supplied(which - 1);
            let BlockBuf::Array(buf) = sources.get(image.index())? else {
                return Err(crate::error::Error::InvalidArgument(format!(
                    "YOLO channel {which} arrived without values"
                )));
            };
            channels.push(buf.widened());
        }

        let rgb = rgb_from_channels(&channels, self.range)
            .map_err(|error| crate::error::Error::backend(format!("yolo: {error}")))?;
        let dyn_img = DynamicImage::ImageRgb8(rgb);
        let (letterboxed, (ratio_w, ratio_h), (pad_w, pad_h)) =
            yolov11::data::resize::resize(&dyn_img, self.input_size, false)
                .map_err(|error| crate::error::Error::backend(format!("yolo: {error}")))?;
        let tensor = image_to_tensor(&letterboxed, &self.device).unsqueeze::<4>();
        let output = {
            let model = self
                .model
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match model.forward(tensor / 255.0, false) {
                yolov11::model::model::YOLOOutput::Infer(output) => output,
                yolov11::model::model::YOLOOutput::Train(_) => {
                    return Err(crate::error::Error::backend(
                        "YOLO inference returned training output".to_string(),
                    ))
                }
            }
        };
        let detections =
            yolov11::model::nms::non_max_suppression(&output, self.conf_threshold, 0.45);

        if let Some(tile_dets) = detections.first() {
            for det in tile_dets {
                let x = (((det[0] + det[2]) * 0.5) - pad_w) as f64 / ratio_w.max(1e-6) as f64;
                let y = (((det[1] + det[3]) * 0.5) - pad_h) as f64 / ratio_h.max(1e-6) as f64;
                if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 {
                    continue;
                }
                let global_y = at.at.offset[1] as f64 + y;
                let global_x = at.at.offset[2] as f64 + x;
                let centre = [
                    0usize,
                    global_y.round().max(0.0) as usize,
                    global_x.round().max(0.0) as usize,
                ];
                if !owns(at.core, centre) {
                    continue;
                }
                let class = det[5].max(0.0) as u64;
                rows.push(
                    centre,
                    &[
                        Value::U64(detection_id(centre, at.at.volume, class)),
                        Value::F64(global_x),
                        Value::F64(global_y),
                        Value::F64(det[4] as f64),
                        Value::U64(class),
                    ],
                )?;
            }
        }

        Ok(BlockOutput::fragment(STREAM.to_string(), rows.encode()))
    }
}

fn rgb_from_channels(channels: &[ndarray::Array3<f64>], range: (f64, f64)) -> Result<RgbImage> {
    let (_, height, width) = channels[0].dim();
    for channel in channels {
        if channel.dim() != channels[0].dim() {
            bail!("attached YOLO channels have different block shapes");
        }
    }
    let mut data = vec![0u8; height * width * 3];
    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * 3;
            match channels.len() {
                1 => {
                    let value = normalise(channels[0][[0, y, x]], range);
                    data[base] = value;
                    data[base + 1] = value;
                    data[base + 2] = value;
                }
                2 => {
                    data[base] = normalise(channels[0][[0, y, x]], range);
                    data[base + 1] = normalise(channels[1][[0, y, x]], range);
                }
                3 => {
                    data[base] = normalise(channels[0][[0, y, x]], range);
                    data[base + 1] = normalise(channels[1][[0, y, x]], range);
                    data[base + 2] = normalise(channels[2][[0, y, x]], range);
                }
                _ => bail!("YOLO wants 1-3 channels"),
            }
        }
    }
    RgbImage::from_raw(width as u32, height as u32, data)
        .ok_or_else(|| anyhow::anyhow!("failed to create RGB tile"))
}

fn normalise(value: f64, (low, high): (f64, f64)) -> u8 {
    let span = (high - low).max(1e-6);
    (((value - low) / span) * 255.0).clamp(0.0, 255.0) as u8
}

fn owns(region: &Region, at: [usize; 3]) -> bool {
    (0..3).all(|axis| {
        at[axis] >= region.start[axis] && at[axis] < region.start[axis] + region.shape[axis]
    })
}

fn detection_id(at: [usize; 3], volume: [usize; 3], class: u64) -> u64 {
    1 + class * (volume[0] as u64) * (volume[1] as u64) * (volume[2] as u64)
        + (at[0] as u64) * (volume[1] as u64) * (volume[2] as u64)
        + (at[1] as u64) * (volume[2] as u64)
        + at[2] as u64
}

fn image_to_tensor(img: &image::RgbImage, device: &Device) -> Tensor<3> {
    let (w, h) = img.dimensions();
    let raw = img.as_raw();
    let hw = (h * w) as usize;
    let mut sample = vec![0.0f32; 3 * hw];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * 3;
            sample[y * w as usize + x] = raw[idx + 2] as f32;
            sample[hw + y * w as usize + x] = raw[idx + 1] as f32;
            sample[2 * hw + y * w as usize + x] = raw[idx] as f32;
        }
    }
    Tensor::<1>::from_floats(sample.as_slice(), device).reshape([3, h as usize, w as usize])
}

fn level_extent(level_dir: &Path) -> Result<(usize, usize)> {
    let text = std::fs::read_to_string(level_dir.join("zarr.json"))
        .with_context(|| format!("reading {}/zarr.json", level_dir.display()))?;
    let metadata: serde_json::Value = serde_json::from_str(&text)?;
    let shape = metadata
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .with_context(|| format!("{}/zarr.json declares no shape", level_dir.display()))?;
    if shape.len() != 3 {
        bail!(
            "{}/zarr.json is rank {}; an OME-Zarr level here is [c, y, x]",
            level_dir.display(),
            shape.len()
        );
    }
    Ok((
        shape[1].as_u64().unwrap_or(0) as usize,
        shape[2].as_u64().unwrap_or(0) as usize,
    ))
}

#[derive(Clone)]
struct Detection {
    id: u64,
    x: f64,
    y: f64,
    confidence: f64,
    class: u64,
}

impl Detection {
    fn from_row(row: &Row<'_>) -> Self {
        Self {
            id: row.u64(0).expect("id"),
            x: row.f64(1).expect("x"),
            y: row.f64(2).expect("y"),
            confidence: row.f64(3).expect("confidence"),
            class: row.u64(4).expect("class"),
        }
    }
}

fn deduplicate(mut detections: Vec<Detection>, radius: f64) -> Vec<Detection> {
    if radius <= 0.0 || detections.len() < 2 {
        detections.sort_by_key(|detection| detection.id);
        return detections;
    }
    detections.sort_by(|left, right| {
        right
            .confidence
            .partial_cmp(&left.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = vec![true; detections.len()];
    let radius_sq = radius * radius;
    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..detections.len() {
            if !keep[j] || detections[i].class != detections[j].class {
                continue;
            }
            let dx = detections[i].x - detections[j].x;
            let dy = detections[i].y - detections[j].y;
            if dx * dx + dy * dy <= radius_sq {
                keep[j] = false;
            }
        }
    }
    let mut out: Vec<_> = detections
        .into_iter()
        .enumerate()
        .filter_map(|(index, detection)| keep[index].then_some(detection))
        .collect();
    out.sort_by_key(|detection| detection.id);
    out
}

fn write_outputs(config: &PredictConfig, detections: &[Detection]) -> Result<()> {
    let mut writer = csv::Writer::from_path(&config.out)
        .with_context(|| format!("writing {}", config.out.display()))?;
    writer.write_record(["id", "x", "y", "confidence", "class"])?;
    for detection in detections {
        writer.write_record([
            detection.id.to_string(),
            format!("{:.2}", detection.x),
            format!("{:.2}", detection.y),
            format!("{:.4}", detection.confidence),
            detection.class.to_string(),
        ])?;
    }
    writer.flush()?;

    let mut summary = csv::Writer::from_path(&config.summary)
        .with_context(|| format!("writing {}", config.summary.display()))?;
    summary.write_record(["image", "spot_count"])?;
    summary.write_record([
        config.zarr.display().to_string(),
        detections.len().to_string(),
    ])?;
    summary.flush()?;
    Ok(())
}
