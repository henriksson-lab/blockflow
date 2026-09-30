//! Blockflow adapter for the separate `yolo3d` center detector.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use burn::module::Module;
use burn::prelude::{Device, Tensor};
#[cfg(feature = "yolo3d-cuda")]
use burn::tensor::DeviceIndex;
use burn::tensor::{FloatDType, Transaction};
use yolo3d_model::{
    decode_scale, nms, Box3d, Detector, DetectorConfig, ScaleOutput, SliceFuseConfig,
    SliceFuseDetector,
};

use crate::table::{Column, ColumnType, RowBuilder, Schema, Value};
use crate::{
    BlockBuf, BlockOutput, BlockView, Coverage, Error, FragmentOp, FragmentOutput, Lifecycle,
    Region, Result, SourceBlocks,
};

pub const STREAM: &str = "yolo3d.detections";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceDevice {
    Cpu,
    Cuda(usize),
}

pub struct Yolo3dDetector {
    model: Mutex<Detector>,
    config: DetectorConfig,
    device: Device,
    halo: [usize; 3],
    range: (f32, f32),
    threshold: f32,
    nms_iou: f32,
    cost: f64,
}

impl Yolo3dDetector {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        config: DetectorConfig,
        checkpoint: &Path,
        device: InferenceDevice,
        halo: [usize; 3],
        range: (f32, f32),
        threshold: f32,
        nms_iou: f32,
    ) -> Result<Self> {
        if range.1.partial_cmp(&range.0) != Some(std::cmp::Ordering::Greater) {
            return Err(Error::invalid("YOLO3D normalization high must exceed low"));
        }
        let device = match device {
            InferenceDevice::Cpu => Device::flex(),
            InferenceDevice::Cuda(index) => {
                #[cfg(feature = "yolo3d-cuda")]
                {
                    Device::cuda(DeviceIndex::new(index))
                }
                #[cfg(not(feature = "yolo3d-cuda"))]
                {
                    return Err(Error::invalid(format!(
                        "CUDA device {index} requested, but blockflow was built without yolo3d-cuda"
                    )));
                }
            }
        };
        let model = Detector::new(&config, &device)
            .map_err(|error| Error::backend(format!("yolo3d: {error}")))?
            .try_load_file(checkpoint)
            .map_err(|error| {
                Error::backend(format!("loading {}: {error}", checkpoint.display()))
            })?;
        Ok(Self {
            model: Mutex::new(model),
            config,
            device,
            halo,
            range,
            threshold,
            nms_iou,
            cost: 3.0,
        })
    }

    #[must_use]
    pub fn with_cost_per_voxel(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    pub fn schema() -> Result<Schema> {
        Schema::new(vec![
            Column::u64("id"),
            Column::new("z", ColumnType::F64),
            Column::new("y", ColumnType::F64),
            Column::new("x", ColumnType::F64),
            Column::new("confidence", ColumnType::F64),
            Column::u64("class"),
            Column::new("z0", ColumnType::F64),
            Column::new("y0", ColumnType::F64),
            Column::new("x0", ColumnType::F64),
            Column::new("z1", ColumnType::F64),
            Column::new("y1", ColumnType::F64),
            Column::new("x1", ColumnType::F64),
        ])
    }
}

/// Blockflow adapter for YOLO-SliceFuse3D. The checkpoint contains the reused
/// 2D microscopy encoder, axial fusion layers, and 3D center heads.
pub struct YoloSliceFuse3dDetector {
    model: Arc<Mutex<SliceFuseDetector>>,
    config: SliceFuseConfig,
    device: Device,
    halo: [usize; 3],
    range: (f32, f32),
    threshold: f32,
    nms_iou: f32,
    cost: f64,
    profiler: Option<Arc<SliceFuseProfiler>>,
    feature_cache: Option<Arc<Mutex<HashMap<SliceFeatureKey, [Tensor<4>; 3]>>>>,
    precomputed_feature_cache: bool,
    transaction_download: bool,
    fp16: bool,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct SliceFeatureKey {
    z: usize,
    y: usize,
    x: usize,
    height: usize,
    width: usize,
}

#[derive(Clone, Debug, Default)]
pub struct SliceFuseProfile {
    pub blocks: u64,
    pub input_voxels: u64,
    pub tensor_downloads: u64,
    pub preprocess: Duration,
    pub upload: Duration,
    pub model_lock_wait: Duration,
    pub encoder: Duration,
    pub fusion_and_heads: Duration,
    pub download: Duration,
    pub decode: Duration,
    pub nms: Duration,
    pub rows: Duration,
    pub feature_cache_hits: u64,
    pub feature_cache_misses: u64,
    pub feature_cache_entries: u64,
}

#[derive(Debug, Default)]
pub struct SliceFuseProfiler {
    totals: Mutex<SliceFuseProfile>,
}

impl SliceFuseProfiler {
    #[must_use]
    pub fn snapshot(&self) -> SliceFuseProfile {
        self.totals
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn record(&self, block: SliceFuseProfile) {
        let mut totals = self
            .totals
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        totals.blocks += block.blocks;
        totals.input_voxels += block.input_voxels;
        totals.tensor_downloads += block.tensor_downloads;
        totals.preprocess += block.preprocess;
        totals.upload += block.upload;
        totals.model_lock_wait += block.model_lock_wait;
        totals.encoder += block.encoder;
        totals.fusion_and_heads += block.fusion_and_heads;
        totals.download += block.download;
        totals.decode += block.decode;
        totals.nms += block.nms;
        totals.rows += block.rows;
        totals.feature_cache_hits += block.feature_cache_hits;
        totals.feature_cache_misses += block.feature_cache_misses;
        totals.feature_cache_entries = totals
            .feature_cache_entries
            .max(block.feature_cache_entries);
    }
}

impl YoloSliceFuse3dDetector {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        config: SliceFuseConfig,
        checkpoint: &Path,
        device: InferenceDevice,
        halo: [usize; 3],
        range: (f32, f32),
        threshold: f32,
        nms_iou: f32,
    ) -> Result<Self> {
        if range.1.partial_cmp(&range.0) != Some(std::cmp::Ordering::Greater) {
            return Err(Error::invalid(
                "YOLO-SliceFuse3D normalization high must exceed low",
            ));
        }
        let device = inference_device(device)?;
        let model = SliceFuseDetector::uninitialized(&config, &device)
            .map_err(|error| Error::backend(format!("yolo-slicefuse3d: {error}")))?
            .try_load_file(checkpoint)
            .map_err(|error| {
                Error::backend(format!("loading {}: {error}", checkpoint.display()))
            })?;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            config,
            device,
            halo,
            range,
            threshold,
            nms_iou,
            cost: 6.0,
            profiler: None,
            feature_cache: None,
            precomputed_feature_cache: false,
            transaction_download: false,
            fp16: false,
        })
    }

    #[must_use]
    pub fn with_cost_per_voxel(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    #[must_use]
    pub fn with_profiler(mut self, profiler: Arc<SliceFuseProfiler>) -> Self {
        self.profiler = Some(profiler);
        self
    }

    /// Cache frozen 2D encoder features on the inference device. Entries use
    /// the exact XY read window and absolute Z, preserving padding and feature
    /// grid alignment while overlapping Z blocks reuse their shared slices.
    #[must_use]
    pub fn with_feature_cache(mut self) -> Self {
        self.feature_cache = Some(Arc::new(Mutex::new(HashMap::new())));
        self
    }

    /// Build an encoder pass that fills the same feature cache consumed by this
    /// detector. Run it to completion on a whole-depth grid with matching XY
    /// cores before executing the detector plan.
    #[must_use]
    pub fn with_precomputed_feature_cache(
        mut self,
        xy_core: [usize; 2],
    ) -> (YoloSliceFuse3dEncoder, Self) {
        let cache = Arc::new(Mutex::new(HashMap::new()));
        self.feature_cache = Some(cache.clone());
        self.precomputed_feature_cache = true;
        let encoder = YoloSliceFuse3dEncoder {
            model: self.model.clone(),
            device: self.device.clone(),
            xy_core,
            xy_halo: [self.halo[1], self.halo[2]],
            range: self.range,
            cache,
            fp16: self.fp16,
            profiler: self.profiler.clone(),
        };
        (encoder, self)
    }

    /// Download all head tensors in one backend transaction instead of twelve
    /// individually synchronized reads.
    #[must_use]
    pub fn with_transaction_download(mut self) -> Self {
        self.transaction_download = true;
        self
    }

    /// Store model parameters and activations as FP16 while retaining FP32
    /// host-side decoding and thresholds.
    #[must_use]
    pub fn with_fp16(mut self) -> Self {
        let mut model = self
            .model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *model = model.clone().cast_float(FloatDType::F16);
        drop(model);
        self.fp16 = true;
        self
    }

    fn forward_cached(
        &self,
        model: &SliceFuseDetector,
        tensor: Option<Tensor<5>>,
        read: &Region,
        profile: &mut SliceFuseProfile,
    ) -> Result<Vec<yolo3d_model::ScaleOutput>> {
        let [depth, height, width] = read.shape3();
        if let Some(tensor) = &tensor {
            let [batch, channels, tensor_depth, tensor_height, tensor_width] = tensor.dims();
            if batch != 1
                || channels != 1
                || [tensor_depth, tensor_height, tensor_width] != [depth, height, width]
            {
                return Err(Error::invalid(
                    "YOLO-SliceFuse3D feature cache received inconsistent pixels",
                ));
            }
        }
        let keys = (0..depth)
            .map(|local_z| SliceFeatureKey {
                z: read.start[0] + local_z,
                y: read.start[1],
                x: read.start[2],
                height,
                width,
            })
            .collect::<Vec<_>>();
        let mut cache = self
            .feature_cache
            .as_ref()
            .expect("cache path requires a cache")
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let missing = keys
            .iter()
            .enumerate()
            .filter_map(|(index, key)| (!cache.contains_key(key)).then_some(index))
            .collect::<Vec<_>>();
        profile.feature_cache_hits += (depth - missing.len()) as u64;
        profile.feature_cache_misses += missing.len() as u64;
        if !missing.is_empty() {
            let tensor = tensor.ok_or_else(|| {
                Error::backend(format!(
                    "precomputed SliceFuse cache is missing {} slices for read window {:?}",
                    missing.len(),
                    read.start
                ))
            })?;
            let slices = tensor.swap_dims(1, 2).reshape([depth, 1, height, width]);
            let started = Instant::now();
            let input = Tensor::cat(
                missing
                    .iter()
                    .map(|&index| slices.clone().slice_dim(0, index..index + 1))
                    .collect(),
                0,
            );
            let encoded = model.encode_slices(input);
            self.device.sync().map_err(|error| {
                Error::backend(format!("synchronizing cached SliceFuse features: {error}"))
            })?;
            profile.encoder += started.elapsed();
            for (batch_index, &local_z) in missing.iter().enumerate() {
                cache.insert(
                    keys[local_z],
                    std::array::from_fn(|level| {
                        encoded[level]
                            .clone()
                            .slice_dim(0, batch_index..batch_index + 1)
                    }),
                );
            }
        }
        profile.feature_cache_entries = cache.len() as u64;
        let features = std::array::from_fn(|level| {
            let slices = keys
                .iter()
                .map(|key| cache.get(key).expect("feature was populated")[level].clone())
                .collect();
            let feature = Tensor::cat(slices, 0);
            let [_, feature_channels, feature_height, feature_width] = feature.dims();
            feature
                .reshape([1, depth, feature_channels, feature_height, feature_width])
                .swap_dims(1, 2)
        });
        drop(cache);
        let started = Instant::now();
        let outputs = model.predict_encoded(features);
        if self.profiler.is_some() {
            self.device.sync().map_err(|error| {
                Error::backend(format!("synchronizing cached SliceFuse heads: {error}"))
            })?;
        }
        profile.fusion_and_heads += started.elapsed();
        Ok(outputs)
    }

    pub fn schema() -> Result<Schema> {
        Yolo3dDetector::schema()
    }
}

const FEATURE_READY_STREAM: &str = "yolo3d.features.ready";

/// First phase of the staged SliceFuse path. Each block spans the complete Z
/// axis and one detector-aligned XY window, so the 2D encoder sees every slice
/// in one efficient batch and no Z overlap is encoded twice.
pub struct YoloSliceFuse3dEncoder {
    model: Arc<Mutex<SliceFuseDetector>>,
    device: Device,
    xy_core: [usize; 2],
    xy_halo: [usize; 2],
    range: (f32, f32),
    cache: Arc<Mutex<HashMap<SliceFeatureKey, [Tensor<4>; 3]>>>,
    fp16: bool,
    profiler: Option<Arc<SliceFuseProfiler>>,
}

impl FragmentOp for YoloSliceFuse3dEncoder {
    fn name(&self) -> &'static str {
        "YOLO-SliceFuse3D encode slices"
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        match axis {
            0 => 0,
            1 => self.xy_halo[0],
            _ => self.xy_halo[1],
        }
    }

    fn cost_per_voxel(&self) -> f64 {
        6.0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            FEATURE_READY_STREAM,
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(voxels) = at.pixels()? else {
            return Ok(BlockOutput::fragment(FEATURE_READY_STREAM, Vec::new()));
        };
        let values = voxels.widened();
        let shape = values.dim();
        let volume = at.grid.volume();
        let span = (self.range.1 - self.range.0).max(f32::EPSILON);
        let started = Instant::now();
        struct Window {
            start: [usize; 3],
            input: Vec<f32>,
        }
        let mut windows: BTreeMap<(usize, usize), Vec<Window>> = BTreeMap::new();
        let core_y0 = at.core.start[1];
        let core_y1 = core_y0 + at.core.shape[1];
        let read_y0 = core_y0.saturating_sub(self.xy_halo[0]);
        let read_y1 = core_y1.saturating_add(self.xy_halo[0]).min(volume[1]);
        let mut core_x0 = at.core.start[2];
        let combined_x1 = core_x0 + at.core.shape[2];
        while core_x0 < combined_x1 {
            let core_x1 = core_x0.saturating_add(self.xy_core[1]).min(combined_x1);
            let read_x0 = core_x0.saturating_sub(self.xy_halo[1]);
            let read_x1 = core_x1.saturating_add(self.xy_halo[1]).min(volume[2]);
            let height = read_y1 - read_y0;
            let width = read_x1 - read_x0;
            let local_y0 = read_y0 - at.read.start[1];
            let local_x0 = read_x0 - at.read.start[2];
            let mut input = Vec::with_capacity(shape.0 * height * width);
            for z in 0..shape.0 {
                for y in local_y0..local_y0 + height {
                    for x in local_x0..local_x0 + width {
                        input.push(
                            ((values[[z, y, x]] as f32 - self.range.0) / span).clamp(0.0, 1.0),
                        );
                    }
                }
            }
            windows.entry((height, width)).or_default().push(Window {
                start: [at.read.start[0], read_y0, read_x0],
                input,
            });
            core_x0 = core_x1;
        }
        let preprocess = started.elapsed();
        let mut upload = Duration::ZERO;
        let mut encoder = Duration::ZERO;
        let window_count = windows.values().map(Vec::len).sum::<usize>();
        let started = Instant::now();
        let model = self
            .model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let lock_wait = started.elapsed();
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for ((height, width), group) in windows {
            let started = Instant::now();
            let tensors = group
                .iter()
                .map(|window| {
                    Tensor::<1>::from_floats(window.input.as_slice(), &self.device)
                        .reshape([shape.0, 1, height, width])
                })
                .collect();
            let tensor = Tensor::cat(tensors, 0);
            let tensor = if self.fp16 {
                tensor.cast(FloatDType::F16)
            } else {
                tensor
            };
            if self.profiler.is_some() {
                self.device.sync().map_err(|error| {
                    Error::backend(format!("SliceFuse encoder upload: {error}"))
                })?;
            }
            upload += started.elapsed();
            let started = Instant::now();
            let encoded = model.encode_slices(tensor);
            self.device.sync().map_err(|error| {
                Error::backend(format!("synchronizing SliceFuse encoder: {error}"))
            })?;
            encoder += started.elapsed();
            for (window_index, window) in group.iter().enumerate() {
                for local_z in 0..shape.0 {
                    let batch_index = window_index * shape.0 + local_z;
                    let key = SliceFeatureKey {
                        z: window.start[0] + local_z,
                        y: window.start[1],
                        x: window.start[2],
                        height,
                        width,
                    };
                    cache.insert(
                        key,
                        std::array::from_fn(|level| {
                            encoded[level]
                                .clone()
                                .slice_dim(0, batch_index..batch_index + 1)
                        }),
                    );
                }
            }
        }
        let entries = cache.len() as u64;
        drop(cache);
        drop(model);
        if let Some(profiler) = &self.profiler {
            profiler.record(SliceFuseProfile {
                input_voxels: voxels.len() as u64,
                preprocess,
                upload,
                model_lock_wait: lock_wait,
                encoder,
                feature_cache_misses: (window_count * shape.0) as u64,
                feature_cache_entries: entries,
                ..SliceFuseProfile::default()
            });
        }
        Ok(BlockOutput::fragment(FEATURE_READY_STREAM, Vec::new()))
    }
}

fn inference_device(device: InferenceDevice) -> Result<Device> {
    match device {
        InferenceDevice::Cpu => Ok(Device::flex()),
        InferenceDevice::Cuda(index) => {
            #[cfg(feature = "yolo3d-cuda")]
            {
                Ok(Device::cuda(DeviceIndex::new(index)))
            }
            #[cfg(not(feature = "yolo3d-cuda"))]
            {
                Err(Error::invalid(format!(
                    "CUDA device {index} requested, but blockflow was built without yolo3d-cuda"
                )))
            }
        }
    }
}

#[cfg(feature = "zarr")]
pub fn finalize_detections(
    env: &dyn crate::Environment,
    phase: usize,
    volume: [usize; 3],
    root: impl AsRef<Path>,
    temporary_parent: impl AsRef<Path>,
) -> Result<ngff_object_table::TableReader> {
    use crate::object_table::{
        finalize_object_table, ColumnRole, ColumnSpec, CoordinateColumn, DType, ObjectValue,
        SpatialIndexSpec, TableSpec,
    };
    let columns = vec![
        ColumnSpec::new("detection_id", DType::U64, ColumnRole::Identity),
        ColumnSpec::new("z", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("y", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("x", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("confidence", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("class", DType::U32, ColumnRole::Measurement),
        ColumnSpec::new("z0", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("y0", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("x0", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("z1", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("y1", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("x1", DType::F32, ColumnRole::Measurement),
    ];
    let tile_shape = vec![32, 256, 256];
    let spatial_index = SpatialIndexSpec {
        coordinates: ["z", "y", "x"]
            .into_iter()
            .map(|axis| CoordinateColumn {
                axis: axis.into(),
                column: axis.into(),
            })
            .collect(),
        grid_shape: (0..3)
            .map(|axis| volume[axis].div_ceil(tile_shape[axis] as usize) as u64)
            .collect(),
        tile_shape,
        tile_order: "row_major".into(),
        within_tile_order: "lexicographic_coordinates_then_identity".into(),
    };
    let spec = TableSpec::new(0, 2048, "../../", "detection_id", columns, spatial_index);
    finalize_object_table(
        env,
        STREAM,
        phase,
        volume,
        Yolo3dDetector::schema()?,
        root,
        temporary_parent,
        spec,
        |row| {
            Ok(Some(vec![
                ObjectValue::U64(row.u64(0)?),
                ObjectValue::F32(row.f64(1)? as f32),
                ObjectValue::F32(row.f64(2)? as f32),
                ObjectValue::F32(row.f64(3)? as f32),
                ObjectValue::F32(row.f64(4)? as f32),
                ObjectValue::U32(
                    u32::try_from(row.u64(5)?)
                        .map_err(|_| Error::invalid("YOLO3D class does not fit u32"))?,
                ),
                ObjectValue::F32(row.f64(6)? as f32),
                ObjectValue::F32(row.f64(7)? as f32),
                ObjectValue::F32(row.f64(8)? as f32),
                ObjectValue::F32(row.f64(9)? as f32),
                ObjectValue::F32(row.f64(10)? as f32),
                ObjectValue::F32(row.f64(11)? as f32),
            ]))
        },
    )
}

impl FragmentOp for Yolo3dDetector {
    fn name(&self) -> &'static str {
        "yolo3d"
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.halo[axis]
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<crate::SeamFold> {
        Some(crate::SeamFold::PerBlock)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        let size = Self::schema()
            .map(|schema| crate::fragment::SidecarSize::row_table(&Arc::new(schema), 1))
            .unwrap_or(crate::fragment::SidecarSize::Unstated);
        vec![FragmentOutput::new(STREAM, Lifecycle::Persistent, Coverage::EveryBlock).sized(size)]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        self.apply_with(at, SourceBlocks::none())
    }

    fn apply_with(&self, at: &BlockView<'_>, _sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let schema = Arc::new(Self::schema()?);
        let mut rows = RowBuilder::new(schema);
        let BlockBuf::Array(voxels) = at.pixels()? else {
            return Ok(BlockOutput::fragment(STREAM, rows.encode()));
        };
        let values = voxels.widened();
        let shape = values.dim();
        let span = (self.range.1 - self.range.0).max(f32::EPSILON);
        let input = values
            .iter()
            .map(|value| ((*value as f32 - self.range.0) / span).clamp(0.0, 1.0))
            .collect::<Vec<_>>();
        let tensor = Tensor::<1>::from_floats(input.as_slice(), &self.device)
            .reshape([1, 1, shape.0, shape.1, shape.2]);
        let outputs = self
            .model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .forward(tensor);
        let strides = self.config.cumulative_strides();
        let mut detections = Vec::new();
        for (level, output) in outputs.into_iter().enumerate() {
            let dims = output.heatmap.dims();
            let spatial = [dims[2], dims[3], dims[4]];
            let heatmap = tensor_values(output.heatmap)?;
            let offset = tensor_values(output.offset)?;
            let size = tensor_values(output.size)?;
            let quality = tensor_values(output.quality)?;
            detections.extend(
                decode_scale(
                    &heatmap,
                    &offset,
                    &size,
                    &quality,
                    spatial,
                    self.config.classes,
                    strides[level],
                    self.config.voxel_size,
                    self.threshold,
                )
                .map_err(|error| Error::backend(format!("yolo3d decode: {error}")))?,
            );
        }
        for detection in nms(detections, self.nms_iou) {
            emit_detection(&mut rows, detection, self.config.voxel_size, at)?;
        }
        Ok(BlockOutput::fragment(STREAM, rows.encode()))
    }
}

impl FragmentOp for YoloSliceFuse3dDetector {
    fn name(&self) -> &'static str {
        "yolo-slicefuse3d"
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.halo[axis]
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }

    fn reads_pixels(&self) -> bool {
        !self.precomputed_feature_cache
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<crate::SeamFold> {
        Some(crate::SeamFold::PerBlock)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        let size = Self::schema()
            .map(|schema| crate::fragment::SidecarSize::row_table(&Arc::new(schema), 1))
            .unwrap_or(crate::fragment::SidecarSize::Unstated);
        vec![FragmentOutput::new(STREAM, Lifecycle::Persistent, Coverage::EveryBlock).sized(size)]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        self.apply_with(at, SourceBlocks::none())
    }

    fn apply_with(&self, at: &BlockView<'_>, _sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let mut profile = SliceFuseProfile {
            blocks: 1,
            ..SliceFuseProfile::default()
        };
        let schema = Arc::new(Self::schema()?);
        let mut rows = RowBuilder::new(schema);
        let tensor = if self.precomputed_feature_cache {
            None
        } else {
            let BlockBuf::Array(voxels) = at.pixels()? else {
                return Ok(BlockOutput::fragment(STREAM, rows.encode()));
            };
            profile.input_voxels = voxels.len() as u64;
            let started = Instant::now();
            let values = voxels.widened();
            let shape = values.dim();
            let span = (self.range.1 - self.range.0).max(f32::EPSILON);
            let input = values
                .iter()
                .map(|value| ((*value as f32 - self.range.0) / span).clamp(0.0, 1.0))
                .collect::<Vec<_>>();
            profile.preprocess = started.elapsed();
            let started = Instant::now();
            let tensor = Tensor::<1>::from_floats(input.as_slice(), &self.device)
                .reshape([1, 1, shape.0, shape.1, shape.2]);
            let tensor = if self.fp16 {
                tensor.cast(FloatDType::F16)
            } else {
                tensor
            };
            if self.profiler.is_some() {
                self.device
                    .sync()
                    .map_err(|error| Error::backend(format!("yolo-slicefuse3d upload: {error}")))?;
            }
            profile.upload = started.elapsed();
            Some(tensor)
        };
        let started = Instant::now();
        let model = self
            .model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        profile.model_lock_wait = started.elapsed();
        let outputs = if self.feature_cache.is_some() {
            self.forward_cached(&model, tensor, at.read, &mut profile)?
        } else if self.profiler.is_some() {
            let tensor = tensor.expect("uncached SliceFuse inference reads pixels");
            let (outputs, times) = model
                .forward_profiled(tensor, &self.device)
                .map_err(|error| Error::backend(format!("yolo-slicefuse3d profile: {error}")))?;
            profile.encoder = times.encoder;
            profile.fusion_and_heads = times.fusion_and_heads;
            outputs
        } else {
            model.forward(tensor.expect("uncached SliceFuse inference reads pixels"))
        };
        drop(model);
        let strides = self.config.cumulative_strides();
        let mut detections = Vec::new();
        let downloaded = if self.transaction_download {
            let started = Instant::now();
            let downloaded = download_scale_outputs(outputs)?;
            profile.download += started.elapsed();
            profile.tensor_downloads += 1;
            downloaded
        } else {
            outputs
                .into_iter()
                .map(|output| {
                    let dims = output.heatmap.dims();
                    let started = Instant::now();
                    let downloaded = DownloadedScale {
                        spatial: [dims[2], dims[3], dims[4]],
                        heatmap: tensor_values(output.heatmap)?,
                        offset: tensor_values(output.offset)?,
                        size: tensor_values(output.size)?,
                        quality: tensor_values(output.quality)?,
                    };
                    profile.download += started.elapsed();
                    profile.tensor_downloads += 4;
                    Ok(downloaded)
                })
                .collect::<Result<Vec<_>>>()?
        };
        for (level, output) in downloaded.into_iter().enumerate() {
            let started = Instant::now();
            detections.extend(
                decode_scale(
                    &output.heatmap,
                    &output.offset,
                    &output.size,
                    &output.quality,
                    output.spatial,
                    self.config.classes,
                    strides[level],
                    self.config.voxel_size,
                    self.threshold,
                )
                .map_err(|error| Error::backend(format!("yolo-slicefuse3d decode: {error}")))?,
            );
            profile.decode += started.elapsed();
        }
        let started = Instant::now();
        let detections = nms(detections, self.nms_iou);
        profile.nms = started.elapsed();
        let started = Instant::now();
        for detection in detections {
            emit_detection(&mut rows, detection, self.config.voxel_size, at)?;
        }
        profile.rows = started.elapsed();
        if let Some(profiler) = &self.profiler {
            profiler.record(profile);
        }
        Ok(BlockOutput::fragment(STREAM, rows.encode()))
    }
}

fn tensor_values<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>> {
    tensor
        .cast(FloatDType::F32)
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|error| Error::backend(format!("reading yolo3d tensor: {error}")))
}

struct DownloadedScale {
    spatial: [usize; 3],
    heatmap: Vec<f32>,
    offset: Vec<f32>,
    size: Vec<f32>,
    quality: Vec<f32>,
}

fn download_scale_outputs(outputs: Vec<ScaleOutput>) -> Result<Vec<DownloadedScale>> {
    let spatial = outputs
        .iter()
        .map(|output| {
            let dims = output.heatmap.dims();
            [dims[2], dims[3], dims[4]]
        })
        .collect::<Vec<_>>();
    let mut transaction = Transaction::default();
    for output in outputs {
        transaction = transaction
            .register(output.heatmap.cast(FloatDType::F32))
            .register(output.offset.cast(FloatDType::F32))
            .register(output.size.cast(FloatDType::F32))
            .register(output.quality.cast(FloatDType::F32));
    }
    let data = transaction
        .try_execute()
        .map_err(|error| Error::backend(format!("reading SliceFuse tensors: {error}")))?;
    if data.len() != spatial.len() * 4 {
        return Err(Error::backend(format!(
            "SliceFuse transaction returned {} tensors, expected {}",
            data.len(),
            spatial.len() * 4
        )));
    }
    let mut data = data.into_iter();
    spatial
        .into_iter()
        .map(|spatial| {
            let mut convert = || {
                data.next()
                    .expect("transaction tensor count was checked")
                    .try_to_vec::<f32>()
                    .map_err(|error| {
                        Error::backend(format!("decoding SliceFuse tensor data: {error}"))
                    })
            };
            Ok(DownloadedScale {
                spatial,
                heatmap: convert()?,
                offset: convert()?,
                size: convert()?,
                quality: convert()?,
            })
        })
        .collect()
}

fn emit_detection(
    rows: &mut RowBuilder,
    detection: Box3d,
    voxel_size: [f32; 3],
    at: &BlockView<'_>,
) -> Result<()> {
    let mut center = [0.0f64; 3];
    let mut low = [0.0f64; 3];
    let mut high = [0.0f64; 3];
    let (physical_low, physical_high) = detection.corners();
    for axis in 0..3 {
        let spacing = voxel_size[axis] as f64;
        center[axis] = at.at.offset[axis] as f64 + detection.center[axis] as f64 / spacing;
        low[axis] = at.at.offset[axis] as f64 + physical_low[axis] as f64 / spacing;
        high[axis] = at.at.offset[axis] as f64 + physical_high[axis] as f64 / spacing;
    }
    let owned = center.map(|value| value.round().max(0.0) as usize);
    if !owns(at.core, owned) {
        return Ok(());
    }
    let id = detection_id(&detection, center, low, high);
    rows.push(
        owned,
        &[
            Value::U64(id),
            Value::F64(center[0]),
            Value::F64(center[1]),
            Value::F64(center[2]),
            Value::F64(detection.confidence as f64),
            Value::U64(detection.class as u64),
            Value::F64(low[0]),
            Value::F64(low[1]),
            Value::F64(low[2]),
            Value::F64(high[0]),
            Value::F64(high[1]),
            Value::F64(high[2]),
        ],
    )
}

fn owns(region: &Region, at: [usize; 3]) -> bool {
    (0..3).all(|axis| {
        at[axis] >= region.start[axis] && at[axis] < region.start[axis] + region.shape[axis]
    })
}

fn detection_id(detection: &Box3d, center: [f64; 3], low: [f64; 3], high: [f64; 3]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut add = |value: u64| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
    };
    add(detection.class as u64);
    for value in center.into_iter().chain(low).chain(high) {
        add(value.to_bits());
    }
    hash.max(1)
}
