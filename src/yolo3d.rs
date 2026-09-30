//! Blockflow adapter for the separate `yolo3d` center detector.

use std::path::Path;
use std::sync::{Arc, Mutex};

use burn::module::Module;
use burn::prelude::{Device, Tensor};
#[cfg(feature = "yolo3d-cuda")]
use burn::tensor::DeviceIndex;
use yolo3d_model::{decode_scale, nms, Box3d, Detector, DetectorConfig};

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

fn tensor_values<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>> {
    tensor
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|error| Error::backend(format!("reading yolo3d tensor: {error}")))
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
