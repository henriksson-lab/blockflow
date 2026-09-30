//! Classical, chunked 3-D bright-centre detection.

use std::sync::Arc;

use ndarray::Array3;

use crate::assemble::{Phase, PlanBuilder};
use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, PhaseView,
    SeamFold, SidecarSize,
};
use crate::ops::{
    gaussian_smooth_into_with, otsu_threshold, response_peak_points, BlobResponse, Boundary,
    Connectivity, Gaussian,
};
use crate::sidecar::Lifecycle;
use crate::table::{Column, RowBuilder, Schema, Value};

const CANDIDATE_MAGIC: u64 = 0x4343_3344_4341_4e31;
const SELECTED_MAGIC: u64 = 0x4343_3344_5345_4c31;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CenterCandidate {
    pub at: [usize; 3],
    pub response: f64,
    pub raw_intensity: f64,
    pub smoothed_intensity: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClassicalCenterResponse {
    Gaussian {
        sigma: [f64; 3],
        truncate: f64,
        boundary: Boundary,
    },
    Blob(BlobResponse),
}

impl ClassicalCenterResponse {
    pub fn gaussian(sigma: [f64; 3], truncate: f64, boundary: Boundary) -> Result<Self> {
        Gaussian::new(sigma, truncate)?;
        Ok(Self::Gaussian {
            sigma,
            truncate,
            boundary,
        })
    }

    fn reach(&self, axis: usize) -> usize {
        let response = match self {
            Self::Gaussian {
                sigma, truncate, ..
            } => (sigma[axis] * truncate).ceil() as usize,
            Self::Blob(response) => response.reach(axis),
        };
        response.saturating_add(1)
    }

    fn scale(&self) -> [f64; 3] {
        match self {
            Self::Gaussian { sigma, .. } => *sigma,
            Self::Blob(response) => response.scale().sigma,
        }
    }

    fn images(&self, input: ndarray::ArrayView3<'_, f64>) -> Result<(Array3<f64>, Array3<f64>)> {
        let mut response = Array3::zeros(input.raw_dim());
        let mut smoothed = Array3::zeros(input.raw_dim());
        match self {
            Self::Gaussian {
                sigma,
                truncate,
                boundary,
            } => {
                let gaussian = Gaussian::new(*sigma, *truncate)?;
                gaussian_smooth_into_with(
                    input,
                    gaussian.kernels(),
                    *boundary,
                    response.view_mut(),
                )?;
                smoothed.assign(&response);
            }
            Self::Blob(blob) => {
                blob.response_into(input, response.view_mut())?;
                let gaussian = Gaussian::new(blob.scale().sigma, 3.0)?;
                gaussian_smooth_into_with(
                    input,
                    gaussian.kernels(),
                    Boundary::Reflect,
                    smoothed.view_mut(),
                )?;
            }
        }
        Ok((response, smoothed))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CenterCandidatesOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    response: ClassicalCenterResponse,
    connectivity: Connectivity,
    minimum_response: f64,
}

impl CenterCandidatesOp {
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        lifecycle: Lifecycle,
        response: ClassicalCenterResponse,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
            response,
            connectivity: Connectivity::FacesEdgesAndCorners,
            minimum_response: f64::MIN_POSITIVE,
        }
    }

    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    pub fn minimum_response(mut self, value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::invalid("minimum centre response must be finite"));
        }
        self.minimum_response = value;
        Ok(self)
    }
}

impl FragmentOp for CenterCandidatesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.response.reach(axis)
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![
            FragmentOutput::new(&self.stream, self.lifecycle, Coverage::EveryBlock).sized(
                SidecarSize::Terms {
                    fixed: 16,
                    per_core_voxel: 48.0,
                    per_read_voxel: 0.0,
                    per_face_voxel: 0.0,
                    tight: false,
                },
            ),
        ]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let input = at.pixels()?.as_array()?.widened();
        let (response, smoothed) = self.response.images(input.view())?;
        let peaks =
            response_peak_points(response.view(), self.connectivity, self.minimum_response)?;
        let mut candidates = Vec::new();
        for peak in peaks {
            let global = [0, 1, 2].map(|axis| at.read.start[axis] + peak.at[axis]);
            let in_core = (0..3).all(|axis| {
                global[axis] >= at.core.start[axis]
                    && global[axis] < at.core.start[axis] + at.core.shape[axis]
            });
            if in_core {
                candidates.push(CenterCandidate {
                    at: global,
                    response: peak.weight,
                    raw_intensity: input[peak.at],
                    smoothed_intensity: smoothed[peak.at],
                });
            }
        }
        Ok(BlockOutput::fragment(
            &self.stream,
            encode_candidates(CANDIDATE_MAGIC, &candidates),
        ))
    }
}

#[derive(Debug, Clone)]
pub struct SelectCentersOp {
    name: &'static str,
    input: String,
    input_phase: usize,
    output: String,
    lifecycle: Lifecycle,
    bins: usize,
    minimum_distance_physical: f64,
    voxel_size: [f64; 3],
    scale: [f64; 3],
    schema: Schema,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CenterSelectionConfig {
    pub histogram_bins: usize,
    pub minimum_distance_physical: f64,
    pub voxel_size: [f64; 3],
    pub scale: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassicalCenterConfig {
    pub response: ClassicalCenterResponse,
    pub histogram_bins: usize,
    pub minimum_distance_physical: f64,
    pub voxel_size: [f64; 3],
}

impl SelectCentersOp {
    pub fn new(
        name: &'static str,
        input: impl Into<String>,
        input_phase: usize,
        output: impl Into<String>,
        lifecycle: Lifecycle,
        config: CenterSelectionConfig,
    ) -> Result<Self> {
        if config.histogram_bins == 0
            || !config.minimum_distance_physical.is_finite()
            || config.minimum_distance_physical < 0.0
        {
            return Err(Error::invalid(
                "centre selection needs positive histogram bins and a finite non-negative distance",
            ));
        }
        if config
            .voxel_size
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::invalid("voxel spacing must be finite and positive"));
        }
        Ok(Self {
            name,
            input: input.into(),
            input_phase,
            output: output.into(),
            lifecycle,
            bins: config.histogram_bins,
            minimum_distance_physical: config.minimum_distance_physical,
            voxel_size: config.voxel_size,
            scale: config.scale,
            schema: center_schema()?,
        })
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

impl FragmentOp for SelectCentersOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn barrier(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.input, self.input_phase)]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![
            FragmentOutput::new(&self.output, self.lifecycle, Coverage::EveryBlock)
                .sized(SidecarSize::row_table(&self.schema, 1)),
        ]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut candidates = Vec::new();
        at.stream_fragments(&self.input, &mut |_key, bytes| {
            candidates.extend(decode_candidates(CANDIDATE_MAGIC, bytes)?);
            Ok(())
        })?;
        if candidates.is_empty() {
            return Ok(encode_candidates(SELECTED_MAGIC, &[]));
        }
        let response_threshold = otsu_threshold(
            &candidates
                .iter()
                .map(|row| row.response)
                .collect::<Vec<_>>(),
            self.bins,
        )?;
        let intensity_threshold = otsu_threshold(
            &candidates
                .iter()
                .map(|row| row.smoothed_intensity)
                .collect::<Vec<_>>(),
            self.bins,
        )?;
        candidates.retain(|row| {
            row.response >= response_threshold && row.smoothed_intensity >= intensity_threshold
        });
        candidates.sort_by(|left, right| {
            right
                .response
                .total_cmp(&left.response)
                .then_with(|| left.at.cmp(&right.at))
        });
        let distance2 = self.minimum_distance_physical.powi(2);
        let mut accepted: Vec<CenterCandidate> = Vec::new();
        for candidate in candidates {
            let separated = accepted.iter().all(|other| {
                let squared = (0..3)
                    .map(|axis| {
                        let delta = (candidate.at[axis] as f64 - other.at[axis] as f64)
                            * self.voxel_size[axis];
                        delta * delta
                    })
                    .sum::<f64>();
                squared >= distance2
            });
            if separated {
                accepted.push(candidate);
            }
        }
        accepted.sort_by_key(|row| row.at);
        Ok(encode_candidates(SELECTED_MAGIC, &accepted))
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let selected = decode_candidates(SELECTED_MAGIC, at.reduced)?;
        let mut rows = RowBuilder::new(Arc::new(self.schema.clone()));
        for candidate in selected {
            if crate::ops::owner_of(at.grid, candidate.at) != at.index {
                continue;
            }
            rows.push(
                candidate.at,
                &[
                    Value::U64(center_id(candidate.at, at.volume())?),
                    Value::F64(candidate.response),
                    Value::F64(candidate.raw_intensity),
                    Value::F64(candidate.smoothed_intensity),
                    Value::F64(self.scale[0]),
                    Value::F64(self.scale[1]),
                    Value::F64(self.scale[2]),
                ],
            )?;
        }
        Ok(BlockOutput::fragment(&self.output, rows.encode()))
    }
}

pub fn append_classical_center_phases(
    plan: &mut PlanBuilder,
    candidate_stream: impl Into<String>,
    rows_stream: impl Into<String>,
    config: ClassicalCenterConfig,
) -> Result<(Phase, Schema)> {
    let candidate_stream = candidate_stream.into();
    let rows_stream = rows_stream.into();
    let scale = config.response.scale();
    let candidates = plan.fragments(CenterCandidatesOp::new(
        "classical 3d centre candidates",
        candidate_stream.clone(),
        Lifecycle::DeleteOnExit,
        config.response,
    ))?;
    let selection = SelectCentersOp::new(
        "classical 3d centre selection",
        candidate_stream,
        candidates.index(),
        rows_stream,
        Lifecycle::Persistent,
        CenterSelectionConfig {
            histogram_bins: config.histogram_bins,
            minimum_distance_physical: config.minimum_distance_physical,
            voxel_size: config.voxel_size,
            scale,
        },
    )?;
    let schema = selection.schema().clone();
    let phase = plan.fragments(selection)?;
    Ok((phase, schema))
}

pub struct CenterSeedsOp {
    name: &'static str,
    rows: String,
    rows_phase: usize,
    schema: Schema,
}

impl CenterSeedsOp {
    pub fn new(
        name: &'static str,
        rows: impl Into<String>,
        rows_phase: usize,
        schema: Schema,
    ) -> Self {
        Self {
            name,
            rows: rows.into(),
            rows_phase,
            schema,
        }
    }
}

impl FragmentOp for CenterSeedsOp {
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

    fn produces(&self, _input: crate::Dtype) -> crate::Dtype {
        crate::Dtype::U32
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.rows, self.rows_phase)]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let bytes = at.own(&self.rows).unwrap_or(&[]);
        let rows = crate::ops::merge_rows(
            at.volume(),
            self.schema.clone(),
            std::iter::once((at.index, bytes)),
        )?;
        let mut out = at.output_buffer(0.0)?;
        if let Some(array) = out.as_array_mut() {
            let mut labels = array.view_mut::<u32>()?;
            for row in rows {
                let label = match row.values.first() {
                    Some(Value::U64(value)) => u32::try_from(*value).map_err(|_| {
                        Error::invalid("classical centre seed identity does not fit u32")
                    })?,
                    _ => return Err(Error::invalid("classical centre seed row has no identity")),
                };
                let local = [0, 1, 2].map(|axis| row.at[axis] - at.read.start[axis]);
                labels[local] = label;
            }
        }
        Ok(BlockOutput::nothing().with_pixels(out))
    }
}

pub fn center_schema() -> Result<Schema> {
    Schema::new(vec![
        Column::u64("detection_id"),
        Column::f64("response"),
        Column::f64("raw_intensity"),
        Column::f64("smoothed_intensity"),
        Column::f64("scale_z"),
        Column::f64("scale_y"),
        Column::f64("scale_x"),
    ])
}

fn center_id(at: [usize; 3], volume: [usize; 3]) -> Result<u64> {
    let linear = at[0]
        .checked_mul(volume[1])
        .and_then(|value| value.checked_add(at[1]))
        .and_then(|value| value.checked_mul(volume[2]))
        .and_then(|value| value.checked_add(at[2]))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| Error::invalid("classical centre identity overflow"))?;
    u64::try_from(linear).map_err(|_| Error::invalid("classical centre identity does not fit u64"))
}

#[cfg(feature = "zarr")]
pub fn finalize_centers(
    env: &dyn crate::Environment,
    root: impl AsRef<std::path::Path>,
    temporary_parent: impl AsRef<std::path::Path>,
    config: CenterTable<'_>,
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
        ColumnSpec::new("response", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("raw_intensity", DType::F32, ColumnRole::Intensity),
        ColumnSpec::new("smoothed_intensity", DType::F32, ColumnRole::Intensity),
        ColumnSpec::new("scale_z", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("scale_y", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("scale_x", DType::F32, ColumnRole::Measurement),
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
            .map(|axis| config.index_volume[axis].div_ceil(tile_shape[axis] as usize) as u64)
            .collect(),
        tile_shape,
        tile_order: "row_major".into(),
        within_tile_order: "lexicographic_coordinates_then_identity".into(),
    };
    let spec = TableSpec::new(0, 2048, "../../", "detection_id", columns, spatial_index);
    finalize_object_table(
        env,
        config.stream,
        config.phase,
        config.volume,
        center_schema()?,
        root,
        temporary_parent,
        spec,
        |row| {
            let at = row.at();
            let global = [0, 1, 2].map(|axis| at[axis] + config.origin[axis]);
            Ok(Some(vec![
                ObjectValue::U64(row.u64(0)?),
                ObjectValue::F32(global[0] as f32),
                ObjectValue::F32(global[1] as f32),
                ObjectValue::F32(global[2] as f32),
                ObjectValue::F32(row.f64(1)? as f32),
                ObjectValue::F32(row.f64(2)? as f32),
                ObjectValue::F32(row.f64(3)? as f32),
                ObjectValue::F32(row.f64(4)? as f32),
                ObjectValue::F32(row.f64(5)? as f32),
                ObjectValue::F32(row.f64(6)? as f32),
            ]))
        },
    )
}

#[cfg(feature = "zarr")]
#[derive(Debug, Clone, Copy)]
pub struct CenterTable<'a> {
    pub stream: &'a str,
    pub phase: usize,
    pub volume: [usize; 3],
    pub origin: [usize; 3],
    pub index_volume: [usize; 3],
}

#[cfg(feature = "zarr")]
pub fn finalize_watershed_instances(
    env: &dyn crate::Environment,
    root: impl AsRef<std::path::Path>,
    temporary_parent: impl AsRef<std::path::Path>,
    config: WatershedInstanceTable<'_>,
) -> Result<ngff_object_table::TableReader> {
    use crate::object_table::{
        finalize_object_table, ColumnRole, ColumnSpec, CoordinateColumn, DType, ObjectValue,
        SpatialIndexSpec, TableSpec,
    };

    let columns = vec![
        ColumnSpec::new("label_id", DType::U64, ColumnRole::Identity),
        ColumnSpec::new("centroid_z", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_y", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_x", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("volume_voxels", DType::U64, ColumnRole::Measurement),
        ColumnSpec::new("volume_um3", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("dapi_mean", DType::F32, ColumnRole::Intensity),
        ColumnSpec::new("dapi_min", DType::F32, ColumnRole::Intensity),
        ColumnSpec::new("dapi_max", DType::F32, ColumnRole::Intensity),
    ];
    let tile_shape = vec![32, 256, 256];
    let spatial_index = SpatialIndexSpec {
        coordinates: ["z", "y", "x"]
            .into_iter()
            .map(|axis| CoordinateColumn {
                axis: axis.into(),
                column: format!("centroid_{axis}"),
            })
            .collect(),
        grid_shape: (0..3)
            .map(|axis| config.volume[axis].div_ceil(tile_shape[axis] as usize) as u64)
            .collect(),
        tile_shape,
        tile_order: "row_major".into(),
        within_tile_order: "lexicographic_coordinates_then_identity".into(),
    };
    let mut spec = TableSpec::new(0, 2048, "../../", "label_id", columns, spatial_index);
    spec.region = Some(format!("../../labels/{}", config.layer));
    finalize_object_table(
        env,
        config.stream,
        config.phase,
        config.volume,
        crate::ops::tabulation_schema(config.fixed),
        root,
        temporary_parent,
        spec,
        |row| {
            let values = crate::ops::region_values(row, config.fixed)?;
            if values.count == 0 {
                return Ok(None);
            }
            Ok(Some(vec![
                ObjectValue::U64(values.label),
                ObjectValue::F32(values.centroid[0] as f32),
                ObjectValue::F32(values.centroid[1] as f32),
                ObjectValue::F32(values.centroid[2] as f32),
                ObjectValue::U64(values.count),
                ObjectValue::F32((values.count as f64 * config.volume_per_voxel) as f32),
                ObjectValue::F32((values.sum / values.count as f64) as f32),
                ObjectValue::F32(values.min as f32),
                ObjectValue::F32(values.max as f32),
            ]))
        },
    )
}

#[cfg(feature = "zarr")]
#[derive(Debug, Clone, Copy)]
pub struct WatershedInstanceTable<'a> {
    pub stream: &'a str,
    pub phase: usize,
    pub volume: [usize; 3],
    pub fixed: crate::ops::FixedPoint,
    pub layer: &'a str,
    pub volume_per_voxel: f64,
}

fn encode_candidates(magic: u64, candidates: &[CenterCandidate]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 + candidates.len() * 48);
    bytes.extend_from_slice(&magic.to_le_bytes());
    bytes.extend_from_slice(&(candidates.len() as u64).to_le_bytes());
    for candidate in candidates {
        for coordinate in candidate.at {
            bytes.extend_from_slice(&(coordinate as u64).to_le_bytes());
        }
        for value in [
            candidate.response,
            candidate.raw_intensity,
            candidate.smoothed_intensity,
        ] {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    bytes
}

fn decode_candidates(expected: u64, bytes: &[u8]) -> Result<Vec<CenterCandidate>> {
    if bytes.len() < 16 {
        return Err(Error::invalid(
            "classical centre fragment ended before its header",
        ));
    }
    let word = |offset: usize| -> u64 {
        u64::from_le_bytes(
            bytes[offset..offset + 8]
                .try_into()
                .expect("eight-byte slice"),
        )
    };
    if word(0) != expected {
        return Err(Error::invalid(
            "classical centre fragment has the wrong magic",
        ));
    }
    let count = usize::try_from(word(8))
        .map_err(|_| Error::invalid("classical centre count does not fit usize"))?;
    if bytes.len() != 16 + count * 48 {
        return Err(Error::invalid(
            "classical centre fragment has the wrong length",
        ));
    }
    let mut out = Vec::with_capacity(count);
    for row in 0..count {
        let base = 16 + row * 48;
        let mut at = [0usize; 3];
        for axis in 0..3 {
            at[axis] = usize::try_from(word(base + axis * 8))
                .map_err(|_| Error::invalid("classical centre coordinate does not fit usize"))?;
        }
        let value = |index: usize| f64::from_bits(word(base + (3 + index) * 8));
        let candidate = CenterCandidate {
            at,
            response: value(0),
            raw_intensity: value(1),
            smoothed_intensity: value(2),
        };
        if [
            candidate.response,
            candidate.raw_intensity,
            candidate.smoothed_intensity,
        ]
        .iter()
        .any(|number| !number.is_finite())
        {
            return Err(Error::invalid(
                "classical centre fragment contains a non-finite value",
            ));
        }
        out.push(candidate);
    }
    Ok(out)
}
