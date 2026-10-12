// SPDX-License-Identifier: MIT

//! **The frame job**: segment one frame, measure it in the other channel, and
//! publish a durable row artifact.
//!
//! Two ordered Blockflow executions over ordinary rank-three `[Z, Y, X]`
//! images:
//!
//! 1. `InstanceSegment` reads the segmentation channel at frame `t` and writes
//!    final-id labels straight into plane `t` of the series' staged label
//!    stack (an output target), with ids offset by `t_index · Z·Y·X` so they
//!    are unique across the whole series without a relabel pass;
//! 2. `Measurements` reads that plane back as its label image, with the
//!    measurement channel at frame `t` as a supplied input.
//!
//! The artifact is `rows.json` plus `manifest.json`, the manifest written
//! last. **The manifest is the only evidence of completion**: a frame without
//! a valid one is rerun from a cleared plane, so nothing an interrupted run
//! wrote survives into the rerun.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::decomposition::Constraints;
use blockflow::geometry::BlockGrid;
use blockflow::model_segment::{InstanceSegment, SegmentBackend};
use blockflow::op::Chain;
use blockflow::ops::collect_rows;
use blockflow::ops::measure::{
    collect_class_a_shapes, collect_class_a_values, IntensityImage, IntensityMeasurements,
    IntensitySet, Measurements, ShapeMeasurements, ShapeSet,
};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::table::Value as Cell;
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::{Dtype, Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::params::{fnv64, Params};
use crate::segment::{frame_range, Normalized};
use crate::series::SeriesJob;

const STREAM: &str = "frame.instances";

/// One detection, as the frame job measured it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrameRow {
    pub detection_id: u64,
    pub t_index: u64,
    pub stored_t: usize,
    /// Pixels (voxels in 3-D).
    pub area: u64,
    /// Exact centroid in native voxel coordinates, `[z, y, x]`.
    pub centroid: [f64; 3],
    /// Segmentation channel, raw values.
    pub segment_mean: f64,
    pub segment_min: f64,
    pub segment_max: f64,
    /// Shape, from the label image.
    pub bbox_min: [u64; 3],
    pub bbox_max: [u64; 3],
    pub equivalent_diameter: f64,
    pub eccentricity: Option<f64>,
    pub major_axis_length: Option<f64>,
    pub minor_axis_length: Option<f64>,
    pub orientation_yx: Option<f64>,
    /// Measurement channel, raw values.
    pub measure_sum: f64,
    pub measure_mean: Option<f64>,
    pub measure_min: Option<f64>,
    pub measure_max: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrameManifest {
    pub version: u32,
    pub stored_t: usize,
    pub t_index: u64,
    pub params_hash: String,
    pub input_hash: String,
    pub rows_hash: String,
    pub plane_hash: String,
    pub rows: usize,
    pub normalization: Option<(f32, f32)>,
}

/// What a frame job did.
#[derive(Debug)]
pub enum FrameOutcome {
    Skipped(Vec<FrameRow>),
    Ran(Vec<FrameRow>),
}

impl FrameOutcome {
    pub fn rows(&self) -> &[FrameRow] {
        match self {
            Self::Skipped(rows) | Self::Ran(rows) => rows,
        }
    }
}

pub fn frame_dir(job: &SeriesJob, stored_t: usize) -> PathBuf {
    job.work.join("frames").join(format!("{stored_t:06}"))
}

/// Load a completed frame artifact, or `None` if it is absent, stale or
/// damaged. `verify_plane` re-hashes the label plane against the manifest.
pub fn load_completed(
    job: &SeriesJob,
    stored_t: usize,
    verify_plane: bool,
) -> Result<Option<Vec<FrameRow>>> {
    let dir = frame_dir(job, stored_t);
    let Ok(bytes) = fs::read(dir.join("manifest.json")) else {
        return Ok(None);
    };
    let Ok(manifest) = serde_json::from_slice::<FrameManifest>(&bytes) else {
        return Ok(None);
    };
    if manifest.version != 1
        || manifest.stored_t != stored_t
        || manifest.t_index != job.timeline[stored_t].t_index
        || manifest.params_hash != job.params_hash
        || manifest.input_hash != job.input_hash
    {
        return Ok(None);
    }
    let Ok(rows_bytes) = fs::read(dir.join("rows.json")) else {
        return Ok(None);
    };
    if format!("{:016x}", fnv64(&rows_bytes)) != manifest.rows_hash {
        return Ok(None);
    }
    let Ok(rows) = serde_json::from_slice::<Vec<FrameRow>>(&rows_bytes) else {
        return Ok(None);
    };
    if verify_plane && format!("{:016x}", job.stack.plane_hash(0, stored_t)?) != manifest.plane_hash
    {
        return Ok(None);
    }
    Ok(Some(rows))
}

/// Run (or skip) one frame.
pub fn run_frame(
    job: &SeriesJob,
    stored_t: usize,
    backend: &Arc<dyn SegmentBackend>,
    params: &Params,
) -> Result<FrameOutcome> {
    if let Some(rows) = load_completed(job, stored_t, params.verify_planes)? {
        return Ok(FrameOutcome::Skipped(rows));
    }
    let dir = frame_dir(job, stored_t);
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(Error::backend)?;
    }
    // Nothing an interrupted attempt wrote may survive: a chunk it wrote that
    // this attempt skips (as empty) would otherwise keep stale ids.
    job.stack.clear_plane(stored_t)?;
    fs::create_dir_all(&dir).map_err(Error::backend)?;
    let scratch = job.work.join("tmp").join(format!("frame-{stored_t:06}"));
    if scratch.exists() {
        fs::remove_dir_all(&scratch).map_err(Error::backend)?;
    }

    let t_index = job.timeline[stored_t].t_index;
    let range = frame_range(
        params.normalization,
        job.movie_range,
        &job.image,
        job.segment_channel,
        stored_t,
        &scratch.join("normalize"),
    )?;
    let backend: Arc<dyn SegmentBackend> = match range {
        Some((low, high)) => Arc::new(Normalized {
            inner: backend.clone(),
            low,
            high,
        }),
        None => backend.clone(),
    };

    // 1. Segment into the staged plane.
    let source = job.image.frame(0, job.segment_channel, stored_t)?;
    let (dtype, volume) = source.metadata()?;
    let voxels = (volume[0] * volume[1] * volume[2]) as u64;
    let offset = t_index
        .checked_mul(voxels)
        .ok_or_else(|| Error::invalid("detection id range overflows u64"))?;
    let halo = [
        if volume[0] > 1 { params.halo } else { 0 },
        params.halo,
        params.halo,
    ];
    let op = InstanceSegment::new(
        "segment frame",
        backend,
        halo,
        STREAM,
        Lifecycle::Persistent,
        Vec::new(),
    )
    .writing_labels()
    .with_id_offset(offset)
    .with_scale(if is_float(dtype) { 16 } else { 0 });
    let schema = op.schema()?;
    let grid = frame_grid(volume, &params.blocks)?;
    let mut builder = PlanBuilder::new(volume, dtype, grid);
    builder.fragments(op)?;
    let assembly = builder.finish()?;
    let env = ZarrEnvironment::attach(scratch.join("segment"), &[source])?
        .with_output_target(job.stack.plane(0, stored_t)?)?;
    let hints = Hints {
        concurrency: params.block_workers.max(1),
        ..Hints::default()
    };
    execute_phases(
        "segment frame",
        &assembly.workflow,
        &assembly.decomposition,
        &hints,
        &env,
        &[],
        &assembly.work(),
    )?;
    let scale = if is_float(dtype) { 65536.0 } else { 1.0 };
    let mut segmented: BTreeMap<u64, (u64, [u64; 3], [u64; 3])> = BTreeMap::new();
    for row in collect_rows(&env, STREAM, assembly.n_phases() - 1, volume, schema)? {
        let word = |i: usize| match &row.values[i] {
            Cell::U64(v) => Ok(*v),
            other => Err(Error::invalid(format!("unexpected row cell {other:?}"))),
        };
        let id = word(0)?;
        // A seam can emit an object twice; the id is a function of its voxels,
        // so the duplicate is the same object and is dropped.
        segmented.entry(id).or_insert((
            word(1)?,
            [word(2)?, word(3)?, word(4)?],
            [word(5)?, word(6)?, word(7)?],
        ));
    }
    drop(env);

    // 2. Measure the other channel over the staged plane.
    let labels = job.stack.plane(0, stored_t)?;
    let marker = job.image.frame(0, job.measure_channel, stored_t)?;
    let (marker_dtype, _) = marker.metadata()?;
    let images = [labels, marker];
    let plan = Measurements::for_labels(ImageId::from(0))
        .shape(ShapeSet::standard())
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(marker_dtype),
            IntensitySet::standard(),
        )
        .stream("frame.measure")
        .lifecycle(Lifecycle::DeleteOnExit)
        .plan_on_attached(
            &images,
            &Constraints {
                split_axes: if volume[0] > 1 {
                    vec![0, 1, 2]
                } else {
                    vec![1, 2]
                },
                block_candidates: params.blocks.clone(),
                expected_concurrency: params.block_workers.max(1),
                ..Constraints::default()
            },
        )?;
    let shape_rows = plan
        .class_a_rows()
        .ok_or_else(|| Error::invalid("measurement plan has no shape rows"))?;
    let value_rows = plan
        .class_a_intensity_rows(0)
        .ok_or_else(|| Error::invalid("measurement plan has no intensity rows"))?;
    let env = ZarrEnvironment::attach(scratch.join("measure"), &images)?;
    let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::U64);
    execute_phases(
        "measure frame",
        &workflow,
        &plan.decomposition,
        &hints,
        &env,
        &[],
        &plan.phase_work(),
    )?;
    let shapes: BTreeMap<u64, ShapeMeasurements> =
        collect_class_a_shapes(&env, &shape_rows, volume, plan.fixed)?
            .iter()
            .map(|shape| {
                let m = ShapeMeasurements::from_shape(shape);
                (m.label, m)
            })
            .collect();
    let values: BTreeMap<u64, IntensityMeasurements> =
        collect_class_a_values(&env, &value_rows, volume, plan.fixed)?
            .iter()
            .map(|v| {
                let m = IntensityMeasurements::from_values(v);
                (m.label, m)
            })
            .collect();
    drop(env);

    let mut rows = Vec::with_capacity(segmented.len());
    for (&id, &(count, sums, intensity)) in &segmented {
        if count == 0 {
            continue;
        }
        let shape = shapes.get(&id).ok_or_else(|| {
            Error::invalid(format!("frame {stored_t}: no shape row for detection {id}"))
        })?;
        let value = values.get(&id).ok_or_else(|| {
            Error::invalid(format!(
                "frame {stored_t}: no intensity row for detection {id}"
            ))
        })?;
        let axes = shape.principal_axis_lengths;
        rows.push(FrameRow {
            detection_id: id,
            t_index,
            stored_t,
            area: count,
            centroid: sums.map(|s| s as f64 / count as f64),
            segment_mean: intensity[0] as f64 / scale / count as f64,
            segment_min: intensity[1] as f64 / scale,
            segment_max: intensity[2] as f64 / scale,
            bbox_min: shape.bbox_min,
            bbox_max: shape.bbox_max,
            equivalent_diameter: if volume[0] > 1 {
                shape.equivalent_sphere_diameter
            } else {
                2.0 * (count as f64 / std::f64::consts::PI).sqrt()
            },
            // `ShapeMeasurements::eccentricity` is the 3-D one, exactly 1 for
            // any region one voxel thick; a 2-D frame wants the in-plane
            // value from the two largest axes, as `regionprops` computes it.
            eccentricity: if volume[0] > 1 {
                shape.eccentricity
            } else {
                axes.and_then(|a| {
                    (a[0] > 0.0).then(|| (1.0 - (a[1] / a[0]).powi(2)).max(0.0).sqrt())
                })
            },
            major_axis_length: axes.map(|a| a[0]),
            minor_axis_length: axes.map(|a| a[1]),
            orientation_yx: shape.orientation_yx,
            measure_sum: value.sum,
            measure_mean: value.mean,
            measure_min: value.min,
            measure_max: value.max,
        });
    }
    for id in shapes.keys() {
        if !segmented.contains_key(id) {
            return Err(Error::invalid(format!(
                "frame {stored_t}: label {id} is in the plane but has no segmentation row; \
                 the halo is smaller than an object"
            )));
        }
    }

    // 3. Publish: rows, then the manifest last.
    let rows_bytes = serde_json::to_vec(&rows).map_err(Error::backend)?;
    write_atomic(&dir.join("rows.json"), &rows_bytes)?;
    let manifest = FrameManifest {
        version: 1,
        stored_t,
        t_index,
        params_hash: job.params_hash.clone(),
        input_hash: job.input_hash.clone(),
        rows_hash: format!("{:016x}", fnv64(&rows_bytes)),
        plane_hash: format!("{:016x}", job.stack.plane_hash(0, stored_t)?),
        rows: rows.len(),
        normalization: range,
    };
    write_atomic(
        &dir.join("manifest.json"),
        &serde_json::to_vec_pretty(&manifest).map_err(Error::backend)?,
    )?;
    let _ = fs::remove_dir_all(&scratch);
    Ok(FrameOutcome::Ran(rows))
}

/// One block per frame when a candidate covers it — no halo is read twice and
/// no object meets a seam — otherwise the largest candidate.
fn frame_grid(volume: [usize; 3], blocks: &[usize]) -> Result<BlockGrid> {
    let largest = volume[1].max(volume[2]);
    let mut candidates = blocks.to_vec();
    candidates.sort_unstable();
    let edge = candidates
        .iter()
        .copied()
        .find(|&b| b >= largest)
        .or_else(|| candidates.last().copied())
        .ok_or_else(|| Error::invalid("no block candidates"))?;
    BlockGrid::new(
        volume,
        [volume[0], edge.min(volume[1]), edge.min(volume[2])],
    )
}

fn is_float(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::F16 | Dtype::F32 | Dtype::F64)
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes).map_err(Error::backend)?;
    fs::File::open(&tmp)
        .and_then(|f| f.sync_all())
        .map_err(Error::backend)?;
    fs::rename(&tmp, path).map_err(Error::backend)
}

/// The run's parameters, as recorded beside each manifest for humans.
pub fn describe(params: &Params, job: &SeriesJob) -> serde_json::Value {
    json!({
        "frame_parameters": params.frame_json(job.segment_channel, job.measure_channel),
        "params_hash": job.params_hash,
        "input_hash": job.input_hash,
    })
}
