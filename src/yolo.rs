//! YOLO prediction over a local OME-Zarr through `blockflow`.
//!
//! The older `predict` command reads TIFF tiles directly. This module is the
//! Blockflow path: one fragment phase reads haloed Zarr blocks, runs YOLO, owns
//! detections by centre-in-core, and emits one table row per spot.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::op::SourceInput;
use crate::strategy::{execute_phases, Hints};
use crate::table::{Column, ColumnType, Row, RowBuilder, Schema, Table, Value};
use crate::{
    env::Environment, AttachedImage, BlockBuf, BlockGrid, BlockOutput, BlockView, Coverage, Dtype,
    FragmentOp, FragmentOutput, ImageId, Lifecycle, PlanBuilder, Reach, Region, SourceBlocks,
    ZarrEnvironment,
};
use anyhow::{bail, Context, Result};
use burn::prelude::*;
#[cfg(any(feature = "yolo-cuda", feature = "yolo-libtorch"))]
use burn::tensor::DeviceIndex;
#[cfg(not(any(feature = "yolo-cuda", feature = "yolo-libtorch")))]
use burn::tensor::DeviceKind;
use image::{DynamicImage, RgbImage};
use ngff_object_table::{
    ColumnRole, ColumnSpec, CoordinateColumn, DType as TableDType, SpatialIndexSpec, TableSpec,
    TableWriter,
};

const STREAM: &str = "yolo_detections";

pub struct PredictConfig {
    pub zarr: PathBuf,
    pub level: usize,
    pub weights: PathBuf,
    pub config: PathBuf,
    pub channels: Vec<usize>,
    /// Optional `[y, x, height, width]` window within the selected level.
    pub region: Option<[usize; 4]>,
    pub normalize_range: (f64, f64),
    pub block: usize,
    pub halo: usize,
    pub conf_threshold: f32,
    pub nms_iou: f32,
    pub max_detections: usize,
    pub input_size: u32,
    pub concurrency: usize,
    pub min_separation: f64,
    pub out: PathBuf,
    pub summary: PathBuf,
    pub table: Option<PathBuf>,
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
    let (level_height, level_width) = level_extent(&level_dir)?;
    let [origin_y, origin_x, height, width] =
        config.region.unwrap_or([0, 0, level_height, level_width]);
    if height == 0
        || width == 0
        || origin_y.saturating_add(height) > level_height
        || origin_x.saturating_add(width) > level_width
    {
        bail!(
            "region [{origin_y}, {origin_x}, {height}, {width}] is outside level extent \
             [{level_height}, {level_width}]"
        );
    }
    let volume = [1, height, width];
    println!(
        "YOLO over level {} region y={} x={} height={} width={}, channels {:?}",
        config.level, origin_y, origin_x, height, width, config.channels
    );

    let images: Vec<AttachedImage> = config
        .channels
        .iter()
        .map(|channel| {
            AttachedImage::at(&level_dir).window([*channel, origin_y, origin_x], [1, height, width])
        })
        .collect();
    let env = ZarrEnvironment::attach(&config.work, &images)
        .map_err(|error| anyhow::anyhow!("attaching {}: {error}", level_dir.display()))?;
    let dtype = env
        .image_dtype(0)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let detector = YoloBlockDetector::new(config, dtype, [origin_y, origin_x])?;
    detector.warmup()?;
    let schema = detector.schema_data()?;
    let grid = BlockGrid::new(volume, [1, config.block, config.block])
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let blocks = grid.n_blocks();
    let mut builder = PlanBuilder::new(volume, dtype, grid.clone());
    let phase = builder
        .fragments(detector)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let assembly = builder
        .finish()
        .map_err(|error| anyhow::anyhow!("{error}"))?;
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
        &assembly.workflow,
        &assembly.decomposition,
        &hints,
        &env,
        &[],
        &assembly.work(),
    )
    .map_err(|error| anyhow::anyhow!("the run: {error}"))?;
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "Ran {blocks} block(s) in {elapsed:.3}s, {} reads, {:.1} Mpx read",
        stats.reads,
        stats.read_voxels as f64 / 1e6
    );

    let mut table = Table::new(volume, schema).map_err(|error| anyhow::anyhow!("{error}"))?;
    for core in grid.cores() {
        let bytes = env
            .read_sidecar(STREAM, phase.index(), core.index)
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
    detections.sort_by(|left, right| {
        left.y
            .total_cmp(&right.y)
            .then(left.x.total_cmp(&right.x))
            .then(left.class.cmp(&right.class))
            .then(right.confidence.total_cmp(&left.confidence))
    });
    for (index, detection) in detections.iter_mut().enumerate() {
        detection.id = index as u64 + 1;
    }

    write_outputs(config, &detections, level_height, level_width)?;
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
    nms_iou: f32,
    max_detections: usize,
    input_size: u32,
    source_dtype: Dtype,
    origin: [usize; 2],
}

impl YoloBlockDetector {
    fn new(config: &PredictConfig, source_dtype: Dtype, origin: [usize; 2]) -> Result<Self> {
        #[cfg(feature = "yolo-libtorch")]
        let device = Device::libtorch_cuda(DeviceIndex::Default);
        #[cfg(all(feature = "yolo-cuda", not(feature = "yolo-libtorch")))]
        let device = Device::cuda(DeviceIndex::Default);
        #[cfg(not(any(feature = "yolo-cuda", feature = "yolo-libtorch")))]
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
        model = model.fuse();

        Ok(Self {
            model: Mutex::new(model),
            device,
            channels: config.channels.len(),
            range: config.normalize_range,
            halo: [0, config.halo, config.halo],
            conf_threshold: config.conf_threshold,
            nms_iou: config.nms_iou,
            max_detections: config.max_detections,
            input_size: config.input_size,
            source_dtype,
            origin,
        })
    }

    fn warmup(&self) -> Result<()> {
        let model = self
            .model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for _ in 0..3 {
            let input = Tensor::<4>::zeros(
                [1, 3, self.input_size as usize, self.input_size as usize],
                &self.device,
            );
            match model.forward(input, false) {
                yolov11::model::model::YOLOOutput::Infer(output) => {
                    let _ = yolov11::model::nms::non_max_suppression_with_options(
                        &output,
                        yolov11::model::nms::NmsOptions {
                            confidence_threshold: self.conf_threshold,
                            iou_threshold: self.nms_iou,
                            max_detections: self.max_detections,
                            ..Default::default()
                        },
                    );
                }
                yolov11::model::model::YOLOOutput::Train(_) => {
                    bail!("YOLO warmup returned training output")
                }
            }
        }
        Ok(())
    }

    fn schema_data(&self) -> crate::error::Result<Schema> {
        Schema::new(vec![
            Column::new("id", ColumnType::U64),
            Column::new("x", ColumnType::F64),
            Column::new("y", ColumnType::F64),
            Column::new("confidence", ColumnType::F64),
            Column::new("class", ColumnType::U64),
            Column::new("x1", ColumnType::F64),
            Column::new("y1", ColumnType::F64),
            Column::new("x2", ColumnType::F64),
            Column::new("y2", ColumnType::F64),
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
        // holds voxels.
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
        let profile = std::env::var_os("BLOCKFLOW_YOLO_PROFILE").is_some();
        let total_started = std::time::Instant::now();
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

        let preprocess_started = std::time::Instant::now();
        let rgb = rgb_from_channels(&channels, self.range)
            .map_err(|error| crate::error::Error::backend(format!("yolo: {error}")))?;
        let rgb_elapsed = preprocess_started.elapsed();
        let resize_started = std::time::Instant::now();
        let dyn_img = DynamicImage::ImageRgb8(rgb);
        let (letterboxed, (ratio_w, ratio_h), (pad_w, pad_h)) =
            yolov11::data::resize::resize(&dyn_img, self.input_size, false)
                .map_err(|error| crate::error::Error::backend(format!("yolo: {error}")))?;
        let resize_elapsed = resize_started.elapsed();
        let planar_started = std::time::Instant::now();
        let (sample, height, width) = image_to_chw(&letterboxed);
        let planar_elapsed = planar_started.elapsed();
        let upload_started = std::time::Instant::now();
        let tensor = Tensor::<1>::from_floats(sample.as_slice(), &self.device)
            .reshape([1, 3, height, width]);
        if profile {
            self.device
                .sync()
                .map_err(|error| crate::error::Error::backend(format!("yolo sync: {error}")))?;
        }
        let upload_elapsed = upload_started.elapsed();
        let forward_started = std::time::Instant::now();
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
        if profile {
            self.device
                .sync()
                .map_err(|error| crate::error::Error::backend(format!("yolo sync: {error}")))?;
        }
        let forward_elapsed = forward_started.elapsed();
        let nms_started = std::time::Instant::now();
        let detections = yolov11::model::nms::non_max_suppression_with_options(
            &output,
            yolov11::model::nms::NmsOptions {
                confidence_threshold: self.conf_threshold,
                iou_threshold: self.nms_iou,
                max_detections: self.max_detections,
                ..Default::default()
            },
        );
        let nms_elapsed = nms_started.elapsed();

        let rows_started = std::time::Instant::now();
        if let Some(tile_dets) = detections.first() {
            for det in tile_dets {
                let x = (((det[0] + det[2]) * 0.5) - pad_w) as f64 / ratio_w.max(1e-6) as f64;
                let y = (((det[1] + det[3]) * 0.5) - pad_h) as f64 / ratio_h.max(1e-6) as f64;
                if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 {
                    continue;
                }
                let local_y = at.at.offset[1] as f64 + y;
                let local_x = at.at.offset[2] as f64 + x;
                let centre = [
                    0usize,
                    local_y.round().max(0.0) as usize,
                    local_x.round().max(0.0) as usize,
                ];
                if !owns(at.core, centre) {
                    continue;
                }
                let global_y = local_y + self.origin[0] as f64;
                let global_x = local_x + self.origin[1] as f64;
                let global_x1 = at.at.offset[2] as f64
                    + ((det[0] - pad_w) as f64 / ratio_w.max(1e-6) as f64)
                    + self.origin[1] as f64;
                let global_y1 = at.at.offset[1] as f64
                    + ((det[1] - pad_h) as f64 / ratio_h.max(1e-6) as f64)
                    + self.origin[0] as f64;
                let global_x2 = at.at.offset[2] as f64
                    + ((det[2] - pad_w) as f64 / ratio_w.max(1e-6) as f64)
                    + self.origin[1] as f64;
                let global_y2 = at.at.offset[1] as f64
                    + ((det[3] - pad_h) as f64 / ratio_h.max(1e-6) as f64)
                    + self.origin[0] as f64;
                let class = det[5].max(0.0) as u64;
                rows.push(
                    centre,
                    &[
                        Value::U64(detection_id(centre, at.at.volume, class)),
                        Value::F64(global_x),
                        Value::F64(global_y),
                        Value::F64(det[4] as f64),
                        Value::U64(class),
                        Value::F64(global_x1),
                        Value::F64(global_y1),
                        Value::F64(global_x2),
                        Value::F64(global_y2),
                    ],
                )?;
            }
        }
        let rows_elapsed = rows_started.elapsed();

        if profile {
            eprintln!(
                "YOLO_PROFILE block={:?} rgb_us={} resize_us={} planar_us={} upload_us={} forward_us={} nms_us={} rows_us={} total_us={}",
                at.core.start,
                rgb_elapsed.as_micros(),
                resize_elapsed.as_micros(),
                planar_elapsed.as_micros(),
                upload_elapsed.as_micros(),
                forward_elapsed.as_micros(),
                nms_elapsed.as_micros(),
                rows_elapsed.as_micros(),
                total_started.elapsed().as_micros(),
            );
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

fn image_to_chw(img: &image::RgbImage) -> (Vec<f32>, usize, usize) {
    let (w, h) = img.dimensions();
    let raw = img.as_raw();
    let hw = (h * w) as usize;
    let mut sample = vec![0.0f32; 3 * hw];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * 3;
            sample[y * w as usize + x] = raw[idx] as f32;
            sample[hw + y * w as usize + x] = raw[idx + 1] as f32;
            sample[2 * hw + y * w as usize + x] = raw[idx + 2] as f32;
        }
    }
    (sample, h as usize, w as usize)
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
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
}

impl Detection {
    fn from_row(row: &Row<'_>) -> Self {
        Self {
            id: row.u64(0).expect("id"),
            x: row.f64(1).expect("x"),
            y: row.f64(2).expect("y"),
            confidence: row.f64(3).expect("confidence"),
            class: row.u64(4).expect("class"),
            x1: row.f64(5).expect("x1"),
            y1: row.f64(6).expect("y1"),
            x2: row.f64(7).expect("x2"),
            y2: row.f64(8).expect("y2"),
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
    let radius_sq = radius * radius;
    let mut accepted = Vec::<Detection>::with_capacity(detections.len());
    let mut cells = HashMap::<(u64, i64, i64), Vec<usize>>::new();
    for detection in detections {
        let cell_x = (detection.x / radius).floor() as i64;
        let cell_y = (detection.y / radius).floor() as i64;
        let mut duplicate = false;
        'neighbors: for dy in -1..=1 {
            for dx in -1..=1 {
                let key = (detection.class, cell_y + dy, cell_x + dx);
                for &index in cells.get(&key).into_iter().flatten() {
                    let previous = &accepted[index];
                    let delta_x = detection.x - previous.x;
                    let delta_y = detection.y - previous.y;
                    if delta_x * delta_x + delta_y * delta_y <= radius_sq {
                        duplicate = true;
                        break 'neighbors;
                    }
                }
            }
        }
        if !duplicate {
            let index = accepted.len();
            cells
                .entry((detection.class, cell_y, cell_x))
                .or_default()
                .push(index);
            accepted.push(detection);
        }
    }
    accepted.sort_by_key(|detection| detection.id);
    accepted
}

fn write_outputs(
    config: &PredictConfig,
    detections: &[Detection],
    height: usize,
    width: usize,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(&config.out)
        .with_context(|| format!("writing {}", config.out.display()))?;
    writer.write_record([
        "id",
        "x",
        "y",
        "confidence",
        "class",
        "x1",
        "y1",
        "x2",
        "y2",
    ])?;
    for detection in detections {
        writer.write_record([
            detection.id.to_string(),
            format!("{:.2}", detection.x),
            format!("{:.2}", detection.y),
            format!("{:.4}", detection.confidence),
            detection.class.to_string(),
            format!("{:.2}", detection.x1),
            format!("{:.2}", detection.y1),
            format!("{:.2}", detection.x2),
            format!("{:.2}", detection.y2),
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
    if let Some(table) = &config.table {
        write_object_table(table, detections, height as u64, width as u64)?;
        std::fs::copy(&config.out, table.join("table.csv"))?;
    }
    Ok(())
}

fn write_object_table(
    output: &Path,
    detections: &[Detection],
    height: u64,
    width: u64,
) -> Result<()> {
    const TILE: u64 = 1024;
    const CHUNK: usize = 2048;
    anyhow::ensure!(
        !output.exists(),
        "object-table output {} already exists",
        output.display()
    );
    let parent = output
        .parent()
        .with_context(|| format!("{} has no parent directory", output.display()))?;
    std::fs::create_dir_all(parent)?;
    let parent_metadata = parent.join("zarr.json");
    if !parent_metadata.exists() {
        let temporary = parent.join(format!(".zarr.json.{}.tmp", std::process::id()));
        std::fs::write(
            &temporary,
            b"{\n  \"zarr_format\": 3,\n  \"node_type\": \"group\",\n  \"attributes\": {}\n}\n",
        )?;
        match std::fs::rename(&temporary, &parent_metadata) {
            Ok(()) => {}
            Err(error) if parent_metadata.exists() => {
                let _ = std::fs::remove_file(&temporary);
                drop(error);
            }
            Err(error) => return Err(error.into()),
        }
    }
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .context("object-table output needs a UTF-8 file name")?;
    let staging = parent.join(format!(".{name}.partial"));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }

    let grid_shape = vec![height.div_ceil(TILE), width.div_ceil(TILE)];
    let tile_id = |detection: &Detection| {
        (detection.y.max(0.0) as u64 / TILE) * grid_shape[1] + detection.x.max(0.0) as u64 / TILE
    };
    let mut order = (0..detections.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        tile_id(&detections[left])
            .cmp(&tile_id(&detections[right]))
            .then(detections[left].y.total_cmp(&detections[right].y))
            .then(detections[left].x.total_cmp(&detections[right].x))
            .then(detections[left].id.cmp(&detections[right].id))
    });

    let columns = vec![
        ColumnSpec::new("detection_id", TableDType::U64, ColumnRole::Identity),
        ColumnSpec::new("centroid_y", TableDType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_x", TableDType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("bbox_y1", TableDType::F32, ColumnRole::Measurement),
        ColumnSpec::new("bbox_x1", TableDType::F32, ColumnRole::Measurement),
        ColumnSpec::new("bbox_y2", TableDType::F32, ColumnRole::Measurement),
        ColumnSpec::new("bbox_x2", TableDType::F32, ColumnRole::Measurement),
        ColumnSpec::new("confidence", TableDType::F32, ColumnRole::Measurement),
        ColumnSpec::new("class", TableDType::U64, ColumnRole::Category),
    ];
    let spatial_index = SpatialIndexSpec {
        coordinates: vec![
            CoordinateColumn {
                axis: "y".into(),
                column: "centroid_y".into(),
            },
            CoordinateColumn {
                axis: "x".into(),
                column: "centroid_x".into(),
            },
        ],
        tile_shape: vec![TILE, TILE],
        grid_shape: grid_shape.clone(),
        tile_order: "row_major".into(),
        within_tile_order: "lexicographic_coordinates_then_identity".into(),
    };
    let spec = TableSpec::new(
        detections.len() as u64,
        CHUNK as u64,
        "../../",
        "detection_id",
        columns,
        spatial_index,
    );
    let writer = TableWriter::create(&staging, spec)?;
    for (chunk_index, indices) in order.chunks(CHUNK).enumerate() {
        let start = (chunk_index * CHUNK) as u64;
        let rows = indices
            .iter()
            .map(|&index| &detections[index])
            .collect::<Vec<_>>();
        writer.write_u64(
            "detection_id",
            start,
            &rows.iter().map(|row| row.id).collect::<Vec<_>>(),
        )?;
        writer.write_f32(
            "centroid_y",
            start,
            &rows.iter().map(|row| row.y as f32).collect::<Vec<_>>(),
        )?;
        writer.write_f32(
            "centroid_x",
            start,
            &rows.iter().map(|row| row.x as f32).collect::<Vec<_>>(),
        )?;
        let float_columns: [(&str, Vec<f32>); 5] = [
            ("bbox_y1", rows.iter().map(|row| row.y1 as f32).collect()),
            ("bbox_x1", rows.iter().map(|row| row.x1 as f32).collect()),
            ("bbox_y2", rows.iter().map(|row| row.y2 as f32).collect()),
            ("bbox_x2", rows.iter().map(|row| row.x2 as f32).collect()),
            (
                "confidence",
                rows.iter().map(|row| row.confidence as f32).collect(),
            ),
        ];
        for (name, values) in float_columns {
            writer.write_f32(name, start, &values)?;
        }
        writer.write_u64(
            "class",
            start,
            &rows.iter().map(|row| row.class).collect::<Vec<_>>(),
        )?;
    }

    let mut identities = order
        .iter()
        .enumerate()
        .map(|(physical, &index)| (detections[index].id, physical as u64))
        .collect::<Vec<_>>();
    identities.sort_unstable();
    for (chunk_index, chunk) in identities.chunks(CHUNK).enumerate() {
        writer.write_identity_index(
            (chunk_index * CHUNK) as u64,
            &chunk.iter().map(|row| row.0).collect::<Vec<_>>(),
            &chunk.iter().map(|row| row.1).collect::<Vec<_>>(),
        )?;
    }

    let mut counts = vec![0u64; grid_shape.iter().product::<u64>() as usize];
    for &index in &order {
        counts[tile_id(&detections[index]) as usize] += 1;
    }
    let mut starts = vec![0u64; counts.len()];
    let mut next = 0u64;
    for (start, count) in starts.iter_mut().zip(&counts) {
        *start = next;
        next += count;
    }
    writer.write_spatial_index(&starts, &counts)?;
    writer.finish()?;
    std::fs::rename(&staging, output)?;
    println!("native object table written to {}", output.display());
    Ok(())
}
