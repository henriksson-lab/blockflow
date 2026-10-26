//! Atomic publication of sparse interactive label masks into OME-Zarr pyramids.

use std::{
    collections::{HashMap, HashSet},
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use zarrs::array::data_type;
use zarrs::array::{ArrayBuilder, ArraySubset, FillValue};
use zarrs::filesystem::FilesystemStore;

use ngff_object_table::{
    ColumnRole, ColumnSpec, CoordinateColumn, DType, SpatialIndexSpec, TableReader, TableSpec,
    TableWriter,
};

use crate::label_pyramid::refresh_label_registry;
use crate::ome_zarr::{write_label_metadata, OmeVolumePyramid};
use crate::{Error, Result};

static STAGING_SERIAL: AtomicU64 = AtomicU64::new(0);

struct TargetLock {
    path: Option<PathBuf>,
}

impl TargetLock {
    fn acquire(labels_root: &Path, name: &str, owner_job_id: Option<&str>) -> Result<Self> {
        let path = labels_root.join(format!(".{name}.rozt.lock"));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::AlreadyExists {
                    Error::invalid(format!("label result {name} is being modified"))
                } else {
                    Error::backend(error)
                }
            })?;
        let marker = serde_json::to_vec(&serde_json::json!({
            "jobId": owner_job_id,
            "pid": std::process::id(),
        }))
        .map_err(Error::backend)?;
        if let Err(error) = file.write_all(&marker).and_then(|()| file.sync_all()) {
            let _ = fs::remove_file(&path);
            return Err(Error::backend(error));
        }
        Ok(Self { path: Some(path) })
    }

    /// Leave the ownership-marked lock on disk for a prepared result that another trusted
    /// process will either publish or discard after user review.
    fn persist(&mut self) -> Result<PathBuf> {
        self.path
            .take()
            .ok_or_else(|| Error::invalid("label target lock was already persisted"))
    }
}

impl Drop for TargetLock {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = fs::remove_file(path);
        }
    }
}

#[derive(Clone, Debug)]
pub struct SparseBinaryMask<'a> {
    pub width: usize,
    pub height: usize,
    pub origin_xy: [f64; 2],
    pub extent_xy: [f64; 2],
    pub runs: &'a [u32],
}

#[derive(Clone, Debug)]
pub struct PublishedLabel {
    pub name: String,
    pub label_id: u64,
    pub foreground_voxels: u64,
}

/// A complete label pyramid retained under its job-owned hidden staging name for review.
#[derive(Clone, Debug)]
pub struct PreparedLabel {
    pub name: String,
    pub staging: PathBuf,
    pub destination: PathBuf,
    pub lock: PathBuf,
}

#[derive(Clone, Copy, Debug)]
pub struct ObjectTableMeasurement {
    /// Physical size of one level-zero voxel in XYZ order.
    pub spacing_xyz: [f64; 3],
    /// Score assigned to the object changed by this commit.
    pub selected_score: f32,
    /// Semantic annotation type. Zero means unknown for imported or legacy objects.
    pub class_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LabelEditMode {
    Add,
    Replace,
    Subtract,
}

/// Atomically edit one ID in a uint64 label pyramid. Legacy uint32 results remain editable when
/// the requested ID fits their storage type.
pub fn edit_sparse_plane_label(
    source: &OmeVolumePyramid,
    name: &str,
    z: usize,
    mask: SparseBinaryMask<'_>,
    label_id: u64,
    mode: LabelEditMode,
    provenance: Option<&serde_json::Value>,
) -> Result<PublishedLabel> {
    edit_sparse_plane_label_inner(source, name, z, mask, label_id, mode, provenance, None)
}

/// Atomically edit one label ID and replace its paired native object table from the resulting
/// level-zero labels. The table and label rollback together if either publication fails.
pub fn edit_sparse_plane_segmentation(
    source: &OmeVolumePyramid,
    name: &str,
    z: usize,
    mask: SparseBinaryMask<'_>,
    label_id: u64,
    mode: LabelEditMode,
    measurement: ObjectTableMeasurement,
    provenance: Option<&serde_json::Value>,
) -> Result<PublishedLabel> {
    edit_sparse_plane_label_inner(
        source,
        name,
        z,
        mask,
        label_id,
        mode,
        provenance,
        Some(measurement),
    )
}

fn edit_sparse_plane_label_inner(
    source: &OmeVolumePyramid,
    name: &str,
    z: usize,
    mask: SparseBinaryMask<'_>,
    label_id: u64,
    mode: LabelEditMode,
    provenance: Option<&serde_json::Value>,
    measurement: Option<ObjectTableMeasurement>,
) -> Result<PublishedLabel> {
    validate_name(name)?;
    validate_mask(&mask)?;
    if label_id == 0 {
        return Err(Error::invalid("label id zero is reserved for background"));
    }
    let labels_root = source.image_root().join("labels");
    fs::create_dir_all(&labels_root).map_err(Error::backend)?;
    let _target_lock = TargetLock::acquire(&labels_root, name, None)?;
    let destination = labels_root.join(name);
    if !destination.is_dir() {
        return Err(Error::invalid(format!(
            "label result {name} does not exist"
        )));
    }
    let serial = STAGING_SERIAL.fetch_add(1, Ordering::Relaxed);
    let staging = labels_root.join(format!(".{name}.rozt-{}-{serial}.edit", std::process::id()));
    let backup = labels_root.join(format!(
        ".{name}.rozt-{}-{serial}.backup",
        std::process::id()
    ));
    let table_destination = source.image_root().join("tables").join(name);
    let existing_rows = read_existing_object_rows(&table_destination)?;
    let old_bounds = existing_rows.as_ref().and_then(|rows| {
        rows.iter()
            .find(|row| row.id == label_id)
            .map(LabelObjectRow::bounds_zyx)
    });
    let edit_bounds = affected_level_zero_bounds(source, z, &mask, None);
    let mut affected_ids = label_ids_in_bounds(&destination, edit_bounds)?;
    affected_ids.insert(label_id);
    let measurement_bounds = existing_rows.as_ref().map_or(edit_bounds, |rows| {
        rows.iter()
            .filter(|row| affected_ids.contains(&row.id))
            .fold(edit_bounds, |bounds, row| {
                union_zyx_bounds(bounds, row.bounds_zyx())
            })
    });
    clone_tree_copy_on_write(&destination, &staging)?;
    let outcome = (|| {
        let mut changed = 0u64;
        for (level_index, level) in source.levels().iter().enumerate() {
            let scale = source.levels()[0].shape[0] as f64 / level.shape[0] as f64;
            let level_z = ((z as f64 + 0.5) / scale).floor() as usize;
            let affected = affected_level_bounds(source, level_index, z, &mask, old_bounds);
            privatize_label_chunks(&staging, level_index, affected)?;
            if mode == LabelEditMode::Replace {
                if let Some(bounds) = old_bounds {
                    let bounds = scale_zyx_bounds(bounds, source.levels()[0].shape, level.shape);
                    clear_label_id(&staging, level_index, level.shape, label_id, Some(bounds))?;
                } else {
                    clear_label_id(&staging, level_index, level.shape, label_id, None)?;
                }
            }
            changed += edit_level(
                &staging,
                source,
                level_index,
                level_z,
                &mask,
                label_id,
                mode,
            )?;
        }
        if let Some(provenance) = provenance {
            append_commit_provenance(&staging, provenance)?;
        }
        if let Some(measurement) = measurement {
            let table = stage_object_table(
                source.image_root(),
                name,
                serial,
                &staging,
                label_id,
                measurement,
                existing_rows.as_deref(),
                Some(&affected_ids),
                Some(measurement_bounds),
            )?;
            replace_label_and_table(&labels_root, &staging, &destination, &backup, table)?;
        } else {
            fs::rename(&destination, &backup).map_err(Error::backend)?;
            if let Err(error) = fs::rename(&staging, &destination) {
                let _ = fs::rename(&backup, &destination);
                return Err(Error::backend(error));
            }
            if let Err(error) = refresh_label_registry(&labels_root) {
                let _ = fs::rename(&destination, &staging);
                let _ = fs::rename(&backup, &destination);
                return Err(error);
            }
            fs::remove_dir_all(&backup).map_err(Error::backend)?;
        }
        Ok(PublishedLabel {
            name: name.to_owned(),
            label_id,
            foreground_voxels: changed,
        })
    })();
    if staging.exists() {
        let _ = fs::remove_dir_all(&staging);
    }
    if backup.exists() && !destination.exists() {
        let _ = fs::rename(&backup, &destination);
    }
    outcome
}

fn clone_tree_copy_on_write(source: &Path, destination: &Path) -> Result<()> {
    clone_tree_copy_on_write_inner(source, destination, false)
}

fn clone_tree_copy_on_write_inner(
    source: &Path,
    destination: &Path,
    inside_chunks: bool,
) -> Result<()> {
    fs::create_dir_all(destination).map_err(Error::backend)?;
    for entry in fs::read_dir(source).map_err(Error::backend)? {
        let entry = entry.map_err(Error::backend)?;
        let to = destination.join(entry.file_name());
        if entry.file_type().map_err(Error::backend)?.is_dir() {
            clone_tree_copy_on_write_inner(
                &entry.path(),
                &to,
                inside_chunks || entry.file_name() == "c",
            )?;
        } else if inside_chunks {
            if fs::hard_link(entry.path(), &to).is_err() {
                fs::copy(entry.path(), to).map_err(Error::backend)?;
            }
        } else {
            fs::copy(entry.path(), to).map_err(Error::backend)?;
        }
    }
    Ok(())
}

fn clear_label_id(
    root: &Path,
    level: usize,
    shape: [usize; 3],
    label_id: u64,
    level_bounds: Option<[usize; 6]>,
) -> Result<()> {
    let store = std::sync::Arc::new(FilesystemStore::new(root).map_err(Error::backend)?);
    let array = zarrs::array::Array::open(store, &format!("/{level}")).map_err(Error::backend)?;
    let block = [1usize, 256, 256];
    let bounds = level_bounds.unwrap_or([0, 0, 0, shape[0], shape[1], shape[2]]);
    for z in bounds[0]..bounds[3] {
        for y in (bounds[1]..bounds[4]).step_by(block[1]) {
            for x in (bounds[2]..bounds[5]).step_by(block[2]) {
                let part = [
                    1,
                    (bounds[4] - y).min(block[1]),
                    (bounds[5] - x).min(block[2]),
                ];
                let subset = ArraySubset::new_with_start_shape(
                    vec![z as u64, y as u64, x as u64],
                    part.map(|v| v as u64).to_vec(),
                )
                .map_err(Error::backend)?;
                if *array.data_type() == data_type::uint64() {
                    let mut values: Vec<u64> = array
                        .retrieve_array_subset(&subset)
                        .map_err(Error::backend)?;
                    if clear_values(&mut values, label_id) {
                        array
                            .store_array_subset(&subset, &values)
                            .map_err(Error::backend)?;
                    }
                } else if *array.data_type() == data_type::uint32() {
                    let wanted = u32::try_from(label_id).map_err(|_| {
                        Error::invalid("label ID does not fit this legacy uint32 result")
                    })?;
                    let mut values: Vec<u32> = array
                        .retrieve_array_subset(&subset)
                        .map_err(Error::backend)?;
                    if clear_values(&mut values, wanted) {
                        array
                            .store_array_subset(&subset, &values)
                            .map_err(Error::backend)?;
                    }
                } else {
                    return Err(Error::invalid("label array is not uint32 or uint64"));
                }
            }
        }
    }
    Ok(())
}

fn affected_level_zero_bounds(
    source: &OmeVolumePyramid,
    z: usize,
    mask: &SparseBinaryMask<'_>,
    old: Option<[u64; 6]>,
) -> [u64; 6] {
    let shape = source.levels()[0].shape;
    let x0 = mask.origin_xy[0].floor().max(0.0) as usize;
    let y0 = mask.origin_xy[1].floor().max(0.0) as usize;
    let x1 = (mask.origin_xy[0] + mask.extent_xy[0])
        .ceil()
        .clamp(0.0, shape[2] as f64) as usize;
    let y1 = (mask.origin_xy[1] + mask.extent_xy[1])
        .ceil()
        .clamp(0.0, shape[1] as f64) as usize;
    let edited = [z, y0, x0, (z + 1).min(shape[0]), y1, x1];
    old.map_or(edited, |old| {
        [
            old[0].min(edited[0] as u64) as usize,
            old[1].min(edited[1] as u64) as usize,
            old[2].min(edited[2] as u64) as usize,
            old[3].max(edited[3] as u64) as usize,
            old[4].max(edited[4] as u64) as usize,
            old[5].max(edited[5] as u64) as usize,
        ]
    })
    .map(|value| value as u64)
}

fn union_zyx_bounds(left: [u64; 6], right: [u64; 6]) -> [u64; 6] {
    [
        left[0].min(right[0]),
        left[1].min(right[1]),
        left[2].min(right[2]),
        left[3].max(right[3]),
        left[4].max(right[4]),
        left[5].max(right[5]),
    ]
}

fn affected_level_bounds(
    source: &OmeVolumePyramid,
    level: usize,
    z: usize,
    mask: &SparseBinaryMask<'_>,
    old: Option<[u64; 6]>,
) -> [usize; 6] {
    scale_zyx_bounds(
        affected_level_zero_bounds(source, z, mask, old),
        source.levels()[0].shape,
        source.levels()[level].shape,
    )
}

fn scale_zyx_bounds(bounds: [u64; 6], from: [usize; 3], to: [usize; 3]) -> [usize; 6] {
    let mut scaled = [0usize; 6];
    for axis in 0..3 {
        let ratio = to[axis] as f64 / from[axis].max(1) as f64;
        scaled[axis] = (bounds[axis] as f64 * ratio).floor() as usize;
        scaled[axis + 3] = (bounds[axis + 3] as f64 * ratio)
            .ceil()
            .clamp(0.0, to[axis] as f64) as usize;
    }
    scaled
}

fn privatize_label_chunks(root: &Path, level: usize, bounds: [usize; 6]) -> Result<()> {
    let metadata_path = root.join(level.to_string()).join("zarr.json");
    if !metadata_path.is_file() {
        // Zarr v2 and non-default flat chunk encodings were copied normally rather than linked.
        return Ok(());
    }
    let metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(metadata_path).map_err(Error::backend)?)
            .map_err(Error::backend)?;
    let chunk = metadata
        .pointer("/chunk_grid/configuration/chunk_shape")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::invalid("label array has no regular chunk shape"))?
        .iter()
        .map(|value| value.as_u64().and_then(|v| usize::try_from(v).ok()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| Error::invalid("label chunk shape is invalid"))?;
    if chunk.len() != 3 || chunk.contains(&0) {
        return Err(Error::invalid(
            "label chunk shape must be positive ZYX rank three",
        ));
    }
    if bounds[..3]
        .iter()
        .zip(&bounds[3..])
        .any(|(lo, hi)| lo >= hi)
    {
        return Ok(());
    }
    for cz in bounds[0] / chunk[0]..=(bounds[3] - 1) / chunk[0] {
        for cy in bounds[1] / chunk[1]..=(bounds[4] - 1) / chunk[1] {
            for cx in bounds[2] / chunk[2]..=(bounds[5] - 1) / chunk[2] {
                let path = root
                    .join(level.to_string())
                    .join("c")
                    .join(cz.to_string())
                    .join(cy.to_string())
                    .join(cx.to_string());
                if path.is_file() {
                    let temporary = path.with_extension(format!("cow-{}", std::process::id()));
                    fs::copy(&path, &temporary).map_err(Error::backend)?;
                    fs::rename(temporary, path).map_err(Error::backend)?;
                }
            }
        }
    }
    Ok(())
}

fn edit_level(
    root: &Path,
    source: &OmeVolumePyramid,
    level_index: usize,
    z: usize,
    mask: &SparseBinaryMask<'_>,
    label_id: u64,
    mode: LabelEditMode,
) -> Result<u64> {
    let level0 = &source.levels()[0];
    let level = &source.levels()[level_index];
    if z >= level.shape[0] {
        return Ok(0);
    }
    let scale = [
        level0.shape[1] as f64 / level.shape[1] as f64,
        level0.shape[2] as f64 / level.shape[2] as f64,
    ];
    let x0 = (mask.origin_xy[0] / scale[1]).floor().max(0.0) as usize;
    let y0 = (mask.origin_xy[1] / scale[0]).floor().max(0.0) as usize;
    let x1 = ((mask.origin_xy[0] + mask.extent_xy[0]) / scale[1])
        .ceil()
        .clamp(0.0, level.shape[2] as f64) as usize;
    let y1 = ((mask.origin_xy[1] + mask.extent_xy[1]) / scale[0])
        .ceil()
        .clamp(0.0, level.shape[1] as f64) as usize;
    if x0 >= x1 || y0 >= y1 {
        return Ok(0);
    }
    let shape = [1, y1 - y0, x1 - x0];
    let decoded = decode_mask(mask)?;
    let store = std::sync::Arc::new(FilesystemStore::new(root).map_err(Error::backend)?);
    let array =
        zarrs::array::Array::open(store, &format!("/{level_index}")).map_err(Error::backend)?;
    let subset = ArraySubset::new_with_start_shape(
        vec![z as u64, y0 as u64, x0 as u64],
        shape.map(|v| v as u64).to_vec(),
    )
    .map_err(Error::backend)?;
    let changed = if *array.data_type() == data_type::uint64() {
        let mut values: Vec<u64> = array
            .retrieve_array_subset(&subset)
            .map_err(Error::backend)?;
        let changed = apply_label_edit(
            &mut values,
            shape,
            [x0, y0, x1, y1],
            scale,
            mask,
            &decoded,
            label_id,
            mode,
        );
        array
            .store_array_subset(&subset, &values)
            .map_err(Error::backend)?;
        changed
    } else if *array.data_type() == data_type::uint32() {
        let label_id = u32::try_from(label_id)
            .map_err(|_| Error::invalid("label ID does not fit this legacy uint32 result"))?;
        let mut values: Vec<u32> = array
            .retrieve_array_subset(&subset)
            .map_err(Error::backend)?;
        let changed = apply_label_edit(
            &mut values,
            shape,
            [x0, y0, x1, y1],
            scale,
            mask,
            &decoded,
            label_id,
            mode,
        );
        array
            .store_array_subset(&subset, &values)
            .map_err(Error::backend)?;
        changed
    } else {
        return Err(Error::invalid("label array is not uint32 or uint64"));
    };
    Ok(changed)
}

fn clear_values<T: Copy + Eq + From<u8>>(values: &mut [T], wanted: T) -> bool {
    let mut dirty = false;
    for value in values {
        if *value == wanted {
            *value = T::from(0);
            dirty = true;
        }
    }
    dirty
}

fn apply_label_edit<T: Copy + Eq + From<u8>>(
    values: &mut [T],
    shape: [usize; 3],
    bounds: [usize; 4],
    scale: [f64; 2],
    mask: &SparseBinaryMask<'_>,
    decoded: &[bool],
    label_id: T,
    mode: LabelEditMode,
) -> u64 {
    let [x0, y0, x1, y1] = bounds;
    let mut changed = 0;
    for yy in y0..y1 {
        for xx in x0..x1 {
            let lx = (xx as f64 + 0.5) * scale[1];
            let ly = (yy as f64 + 0.5) * scale[0];
            let mx = ((lx - mask.origin_xy[0]) * mask.width as f64 / mask.extent_xy[0]).floor();
            let my = ((ly - mask.origin_xy[1]) * mask.height as f64 / mask.extent_xy[1]).floor();
            if mx >= 0.0
                && my >= 0.0
                && mx < mask.width as f64
                && my < mask.height as f64
                && decoded[my as usize * mask.width + mx as usize]
            {
                let value = &mut values[(yy - y0) * shape[2] + xx - x0];
                let next = if mode == LabelEditMode::Subtract && *value == label_id {
                    T::from(0)
                } else if mode != LabelEditMode::Subtract {
                    label_id
                } else {
                    *value
                };
                if next != *value {
                    *value = next;
                    changed += 1;
                }
            }
        }
    }
    changed
}

/// Streaming writer for a whole-image instance result. Callers may write bounded level-zero
/// patches in a deterministic order; `finish` builds the remaining levels out of core and then
/// publishes the complete group atomically.
pub struct SparseLabelVolumeWriter {
    image_root: std::path::PathBuf,
    labels_root: std::path::PathBuf,
    staging: std::path::PathBuf,
    destination: std::path::PathBuf,
    levels: Vec<crate::ome_zarr::OmeVolumeLevel>,
    source: String,
    name: String,
    serial: u64,
    finished: bool,
    target_lock: TargetLock,
}

impl SparseLabelVolumeWriter {
    pub fn create(
        source: &OmeVolumePyramid,
        name: &str,
        chunk: [usize; 3],
        owner_job_id: Option<&str>,
    ) -> Result<Self> {
        validate_name(name)?;
        let level0 = source
            .levels()
            .first()
            .ok_or_else(|| Error::invalid("source pyramid is empty"))?;
        if chunk.contains(&0) {
            return Err(Error::invalid("label chunks must be positive"));
        }
        let labels_root = source.image_root().join("labels");
        fs::create_dir_all(&labels_root).map_err(Error::backend)?;
        let target_lock = TargetLock::acquire(&labels_root, name, owner_job_id)?;
        let destination = labels_root.join(name);
        if destination.exists() {
            return Err(Error::invalid(format!(
                "label result {name} already exists"
            )));
        }
        let serial = STAGING_SERIAL.fetch_add(1, Ordering::Relaxed);
        let staging = labels_root.join(format!(
            ".{name}.rozt-{}-{serial}.partial",
            std::process::id()
        ));
        fs::create_dir_all(&staging).map_err(Error::backend)?;
        if let Some(job_id) = owner_job_id {
            fs::write(
                staging.join(".rozt-job-owner.json"),
                serde_json::to_vec(&serde_json::json!({"jobId": job_id}))
                    .map_err(Error::backend)?,
            )
            .map_err(Error::backend)?;
        }
        let store = std::sync::Arc::new(FilesystemStore::new(&staging).map_err(Error::backend)?);
        let array = ArrayBuilder::new(
            level0.shape.map(|v| v as u64).to_vec(),
            chunk.map(|v| v as u64).to_vec(),
            data_type::uint64(),
            FillValue::from(0u64),
        )
        .build(store, "/0")
        .map_err(Error::backend)?;
        array.store_metadata().map_err(Error::backend)?;
        Ok(Self {
            image_root: source.image_root().to_path_buf(),
            labels_root,
            staging,
            destination,
            levels: source.levels().to_vec(),
            source: source.label_source().to_owned(),
            name: name.to_owned(),
            serial,
            finished: false,
            target_lock,
        })
    }

    pub fn write_level_zero_patch(
        &self,
        start: [usize; 3],
        shape: [usize; 3],
        values: &[u64],
    ) -> Result<()> {
        let expected = shape
            .iter()
            .try_fold(1usize, |n, &v| n.checked_mul(v))
            .ok_or_else(|| Error::invalid("label patch size overflows"))?;
        if expected != values.len() {
            return Err(Error::invalid(format!(
                "label patch has {} values, expected {expected}",
                values.len()
            )));
        }
        for axis in 0..3 {
            if start[axis]
                .checked_add(shape[axis])
                .is_none_or(|end| end > self.levels[0].shape[axis])
            {
                return Err(Error::invalid("label patch is outside level zero"));
            }
        }
        let store =
            std::sync::Arc::new(FilesystemStore::new(&self.staging).map_err(Error::backend)?);
        let array = zarrs::array::Array::open(store, "/0").map_err(Error::backend)?;
        let subset = ArraySubset::new_with_start_shape(
            start.map(|v| v as u64).to_vec(),
            shape.map(|v| v as u64).to_vec(),
        )
        .map_err(Error::backend)?;
        array
            .store_array_subset(&subset, values)
            .map_err(Error::backend)
    }

    pub fn record_commit_provenance(&self, provenance: &serde_json::Value) -> Result<()> {
        append_commit_provenance(&self.staging, provenance)
    }

    pub fn finish(mut self, workers: usize) -> Result<PublishedLabel> {
        self.prepare_pyramid(workers)?;
        fs::rename(&self.staging, &self.destination).map_err(Error::backend)?;
        if let Err(error) = refresh_label_registry(&self.labels_root) {
            let _ = fs::rename(&self.destination, &self.staging);
            return Err(error);
        }
        self.finished = true;
        Ok(PublishedLabel {
            name: self.name.clone(),
            label_id: 0,
            foreground_voxels: 0,
        })
    }

    /// Build all pyramid levels but retain the complete result under its hidden, job-owned
    /// staging path. The target lock is deliberately persisted so another process can approve or
    /// reject the result without allowing a competing writer to claim its final name.
    pub fn prepare_for_review(mut self, workers: usize) -> Result<PreparedLabel> {
        self.prepare_pyramid(workers)?;
        let lock = self.target_lock.persist()?;
        self.finished = true;
        Ok(PreparedLabel {
            name: self.name.clone(),
            staging: self.staging.clone(),
            destination: self.destination.clone(),
            lock,
        })
    }

    /// Build the label pyramid and publish it with its native spatial table as one rollback-safe
    /// transaction. Readers never observe only one half of the pair.
    pub fn finish_with_object_table(
        mut self,
        workers: usize,
        label_id: u64,
        foreground_voxels: u64,
        measurement: ObjectTableMeasurement,
    ) -> Result<PublishedLabel> {
        if label_id == 0 {
            return Err(Error::invalid("label id zero is reserved for background"));
        }
        self.prepare_pyramid(workers)?;
        let table = stage_object_table(
            &self.image_root,
            &self.name,
            self.serial,
            &self.staging,
            label_id,
            measurement,
            None,
            None,
            None,
        )?;
        let table_staging = table.staging.clone();
        if let Err(error) =
            install_new_label_and_table(&self.labels_root, &self.staging, &self.destination, table)
        {
            if table_staging.exists() {
                let _ = fs::remove_dir_all(table_staging);
            }
            return Err(error);
        }
        self.finished = true;
        Ok(PublishedLabel {
            name: self.name.clone(),
            label_id,
            foreground_voxels,
        })
    }

    fn prepare_pyramid(&self, workers: usize) -> Result<()> {
        let shapes = self
            .levels
            .iter()
            .map(|level| level.shape)
            .collect::<Vec<_>>();
        let work = self.staging.join(".pyramid-work");
        crate::label_pyramid::build_nearest_label_pyramid_with_blocks(
            &self.staging,
            &work,
            &shapes,
            &[[1, 256, 256], [1, 128, 128], [1, 64, 64]],
            workers.max(1),
        )?;
        if work.exists() {
            fs::remove_dir_all(&work).map_err(Error::backend)?;
        }
        write_label_metadata(&self.staging, &self.levels, &self.source)?;
        Ok(())
    }
}

/// Publish propagated sparse z masks as one uint64 label volume and paired native object table.
/// All masks are sampled into level-zero coordinates before the ordinary pyramid builder runs.
pub fn publish_sparse_volume_segmentation(
    source: &OmeVolumePyramid,
    name: &str,
    masks: &[(usize, SparseBinaryMask<'_>)],
    label_id: u64,
    measurement: ObjectTableMeasurement,
    provenance: Option<&serde_json::Value>,
) -> Result<PublishedLabel> {
    if masks.is_empty() {
        return Err(Error::invalid("propagated segmentation has no masks"));
    }
    let level0 = source
        .levels()
        .first()
        .ok_or_else(|| Error::invalid("source pyramid is empty"))?;
    let writer = SparseLabelVolumeWriter::create(source, name, [1, 256, 256], None)?;
    let mut foreground_voxels = 0_u64;
    for (z, mask) in masks {
        validate_mask(mask)?;
        if *z >= level0.shape[0] {
            return Err(Error::invalid(format!(
                "z plane {z} is outside source depth {}",
                level0.shape[0]
            )));
        }
        let decoded = decode_mask(mask)?;
        let x0 = mask.origin_xy[0].floor().max(0.0) as usize;
        let y0 = mask.origin_xy[1].floor().max(0.0) as usize;
        let x1 = (mask.origin_xy[0] + mask.extent_xy[0])
            .ceil()
            .clamp(0.0, level0.shape[2] as f64) as usize;
        let y1 = (mask.origin_xy[1] + mask.extent_xy[1])
            .ceil()
            .clamp(0.0, level0.shape[1] as f64) as usize;
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        let mut values = vec![0_u64; (y1 - y0) * (x1 - x0)];
        for yy in y0..y1 {
            for xx in x0..x1 {
                let mx = ((xx as f64 + 0.5 - mask.origin_xy[0]) * mask.width as f64
                    / mask.extent_xy[0])
                    .floor();
                let my = ((yy as f64 + 0.5 - mask.origin_xy[1]) * mask.height as f64
                    / mask.extent_xy[1])
                    .floor();
                if mx >= 0.0
                    && my >= 0.0
                    && mx < mask.width as f64
                    && my < mask.height as f64
                    && decoded[my as usize * mask.width + mx as usize]
                {
                    values[(yy - y0) * (x1 - x0) + xx - x0] = label_id;
                    foreground_voxels = foreground_voxels.saturating_add(1);
                }
            }
        }
        writer.write_level_zero_patch([*z, y0, x0], [1, y1 - y0, x1 - x0], &values)?;
    }
    if let Some(provenance) = provenance {
        writer.record_commit_provenance(provenance)?;
    }
    writer.finish_with_object_table(1, label_id, foreground_voxels, measurement)
}

/// Remove unpublished and published label output owned by one cancelled job attempt.
///
/// Ownership is checked inside every directory before deletion. A result with the same name that
/// predates the job, or was produced by another attempt, is never removed.
pub fn cleanup_job_label_attempt(
    source: &OmeVolumePyramid,
    name: &str,
    job_id: &str,
) -> Result<Vec<String>> {
    validate_name(name)?;
    let labels_root = source.image_root().join("labels");
    if !labels_root.is_dir() {
        return Ok(Vec::new());
    }
    let mut removed = Vec::new();
    let hidden_prefix = format!(".{name}.rozt-");
    for entry in fs::read_dir(&labels_root).map_err(Error::backend)? {
        let entry = entry.map_err(Error::backend)?;
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        let is_attempt = file_name.starts_with(&hidden_prefix)
            && (file_name.ends_with(".partial") || file_name.ends_with(".edit"));
        if is_attempt && owned_by_job(&entry.path(), job_id)? {
            fs::remove_dir_all(entry.path()).map_err(Error::backend)?;
            removed.push(file_name.into_owned());
        }
    }
    let destination = labels_root.join(name);
    if destination.is_dir() && owned_by_job(&destination, job_id)? {
        fs::remove_dir_all(&destination).map_err(Error::backend)?;
        removed.push(name.to_owned());
        refresh_label_registry(&labels_root)?;
    }
    let lock = labels_root.join(format!(".{name}.rozt.lock"));
    if lock.is_file() && marker_owned_by_job(&lock, job_id)? {
        fs::remove_file(&lock).map_err(Error::backend)?;
        removed.push(
            lock.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
        );
    }
    Ok(removed)
}

fn owned_by_job(directory: &Path, job_id: &str) -> Result<bool> {
    marker_owned_by_job(&directory.join(".rozt-job-owner.json"), job_id)
}

fn marker_owned_by_job(path: &Path, job_id: &str) -> Result<bool> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(Error::backend(error)),
    };
    let marker: serde_json::Value = serde_json::from_slice(&bytes).map_err(Error::backend)?;
    Ok(marker.get("jobId").and_then(serde_json::Value::as_str) == Some(job_id))
}

impl Drop for SparseLabelVolumeWriter {
    fn drop(&mut self) {
        if !self.finished && self.staging.exists() {
            let _ = fs::remove_dir_all(&self.staging);
        }
    }
}

/// Publish one accepted 2D mask as a complete sparse OME-Zarr label pyramid.
///
/// Every level is created in a hidden sibling directory and contains only chunks touched by the
/// mask; unwritten chunks read as background. The complete group is renamed into place before the
/// labels registry is refreshed, so readers never observe a partly written pyramid.
pub fn publish_sparse_plane_label(
    source: &OmeVolumePyramid,
    name: &str,
    z: usize,
    mask: SparseBinaryMask<'_>,
    label_id: u64,
    provenance: Option<&serde_json::Value>,
) -> Result<PublishedLabel> {
    publish_sparse_plane_label_inner(source, name, z, mask, label_id, provenance, None)
}

/// Atomically publish a sparse label pyramid and its native spatial object table.
pub fn publish_sparse_plane_segmentation(
    source: &OmeVolumePyramid,
    name: &str,
    z: usize,
    mask: SparseBinaryMask<'_>,
    label_id: u64,
    measurement: ObjectTableMeasurement,
    provenance: Option<&serde_json::Value>,
) -> Result<PublishedLabel> {
    publish_sparse_plane_label_inner(
        source,
        name,
        z,
        mask,
        label_id,
        provenance,
        Some(measurement),
    )
}

fn publish_sparse_plane_label_inner(
    source: &OmeVolumePyramid,
    name: &str,
    z: usize,
    mask: SparseBinaryMask<'_>,
    label_id: u64,
    provenance: Option<&serde_json::Value>,
    measurement: Option<ObjectTableMeasurement>,
) -> Result<PublishedLabel> {
    validate_name(name)?;
    validate_mask(&mask)?;
    if label_id == 0 {
        return Err(Error::invalid("label id zero is reserved for background"));
    }
    let level_zero = source
        .levels()
        .first()
        .ok_or_else(|| Error::invalid("source pyramid is empty"))?;
    if z >= level_zero.shape[0] {
        return Err(Error::invalid(format!(
            "z plane {z} is outside source depth {}",
            level_zero.shape[0]
        )));
    }
    let labels_root = source.image_root().join("labels");
    fs::create_dir_all(&labels_root).map_err(Error::backend)?;
    let _target_lock = TargetLock::acquire(&labels_root, name, None)?;
    let destination = labels_root.join(name);
    if destination.exists() {
        return Err(Error::invalid(format!(
            "label result {name} already exists"
        )));
    }
    let serial = STAGING_SERIAL.fetch_add(1, Ordering::Relaxed);
    let staging = labels_root.join(format!(
        ".{name}.rozt-{}-{serial}.partial",
        std::process::id()
    ));
    if staging.exists() {
        fs::remove_dir_all(&staging).map_err(Error::backend)?;
    }
    fs::create_dir_all(&staging).map_err(Error::backend)?;
    let outcome = (|| {
        let mut foreground_voxels = 0;
        for (level_index, level) in source.levels().iter().enumerate() {
            let written = write_level(source, &staging, level_index, z, &mask, label_id)?;
            if level_index == 0 {
                foreground_voxels = written;
            }
            if level.shape.contains(&0) {
                return Err(Error::invalid("source label level has an empty axis"));
            }
        }
        if let Some(provenance) = provenance {
            append_commit_provenance(&staging, provenance)?;
        }
        write_label_metadata(&staging, source.levels(), source.label_source())?;
        if let Some(measurement) = measurement {
            let table = stage_object_table(
                source.image_root(),
                name,
                serial,
                &staging,
                label_id,
                measurement,
                None,
                None,
                None,
            )?;
            install_new_label_and_table(&labels_root, &staging, &destination, table)?;
        } else {
            fs::rename(&staging, &destination).map_err(Error::backend)?;
            if let Err(error) = refresh_label_registry(&labels_root) {
                let _ = fs::rename(&destination, &staging);
                return Err(error);
            }
        }
        Ok(PublishedLabel {
            name: name.to_owned(),
            label_id,
            foreground_voxels,
        })
    })();
    if outcome.is_err() && staging.exists() {
        let _ = fs::remove_dir_all(&staging);
    }
    outcome
}

#[derive(Clone, Debug)]
struct LabelObjectRow {
    id: u64,
    count: u64,
    sum_xyz: [u128; 3],
    /// Preserve stored f32 centroids exactly for rows unaffected by an incremental edit.
    centroid_override: Option<[f32; 3]>,
    min_xyz: [u64; 3],
    max_xyz_exclusive: [u64; 3],
    score: f32,
    class_id: u32,
}

impl LabelObjectRow {
    fn bounds_zyx(&self) -> [u64; 6] {
        [
            self.min_xyz[2],
            self.min_xyz[1],
            self.min_xyz[0],
            self.max_xyz_exclusive[2],
            self.max_xyz_exclusive[1],
            self.max_xyz_exclusive[0],
        ]
    }
}

struct StagedObjectTable {
    staging: PathBuf,
    destination: PathBuf,
    backup: PathBuf,
}

#[derive(Clone, Debug)]
pub struct PublishedObjectTable {
    pub name: String,
    pub objects: u64,
}

/// Measure an existing label result and atomically publish a native spatial object table.
/// The immutable owner marker lets a cancelled ROZT job remove only its own attempt.
pub fn publish_object_table_for_label(
    source: &OmeVolumePyramid,
    label_name: &str,
    table_name: &str,
    owner_job_id: &str,
    replace: bool,
) -> Result<PublishedObjectTable> {
    validate_name(label_name)?;
    validate_name(table_name)?;
    if owner_job_id.trim().is_empty() {
        return Err(Error::invalid("object table owner job ID is empty"));
    }
    let label_root = source.image_root().join("labels").join(label_name);
    if !label_root.is_dir() {
        return Err(Error::invalid(format!(
            "label result {label_name} does not exist"
        )));
    }
    let spacing_zyx = source.voxel_size();
    let spacing_xyz = [spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]];
    if spacing_xyz
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(Error::invalid("object-table calibration is invalid"));
    }
    let tables_root = source.image_root().join("tables");
    fs::create_dir_all(&tables_root).map_err(Error::backend)?;
    ensure_zarr_group(&tables_root)?;
    let _target_lock = TargetLock::acquire(&tables_root, table_name, Some(owner_job_id))?;
    let destination = tables_root.join(table_name);
    if destination.exists() && !replace {
        return Err(Error::invalid(format!(
            "object table {table_name} already exists"
        )));
    }
    let staging = tables_root.join(format!(".{table_name}.rozt-{owner_job_id}.partial"));
    let backup = tables_root.join(format!(".{table_name}.rozt-{owner_job_id}.backup"));
    if staging.exists() || backup.exists() {
        return Err(Error::invalid(
            "object-table transaction path already exists",
        ));
    }
    let outcome = (|| {
        let scores = existing_object_scores(&destination)?;
        let (shape_zyx, mut rows) = measure_level_zero_labels(&label_root, &scores)?;
        let object_count = rows.len() as u64;
        fs::create_dir_all(&staging).map_err(Error::backend)?;
        fs::write(
            staging.join(".rozt-job-owner.json"),
            serde_json::to_vec(&serde_json::json!({"jobId": owner_job_id}))
                .map_err(Error::backend)?,
        )
        .map_err(Error::backend)?;
        write_object_rows(&staging, label_name, shape_zyx, spacing_xyz, &mut rows)?;
        let had_destination = destination.exists();
        if had_destination {
            fs::rename(&destination, &backup).map_err(Error::backend)?;
        }
        if let Err(error) = fs::rename(&staging, &destination) {
            if had_destination {
                let _ = fs::rename(&backup, &destination);
            }
            return Err(Error::backend(error));
        }
        if had_destination {
            let _ = fs::remove_dir_all(&backup);
        }
        Ok(PublishedObjectTable {
            name: table_name.to_owned(),
            objects: object_count,
        })
    })();
    if staging.exists() {
        let _ = fs::remove_dir_all(&staging);
    }
    if backup.exists() && !destination.exists() {
        let _ = fs::rename(&backup, &destination);
    }
    outcome
}

impl Drop for StagedObjectTable {
    fn drop(&mut self) {
        if self.staging.exists() {
            let _ = fs::remove_dir_all(&self.staging);
        }
    }
}

fn stage_object_table(
    image_root: &Path,
    name: &str,
    serial: u64,
    label_staging: &Path,
    selected_id: u64,
    measurement: ObjectTableMeasurement,
    existing_rows: Option<&[LabelObjectRow]>,
    affected_ids: Option<&HashSet<u64>>,
    affected_bounds: Option<[u64; 6]>,
) -> Result<StagedObjectTable> {
    if measurement
        .spacing_xyz
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || !measurement.selected_score.is_finite()
    {
        return Err(Error::invalid(
            "object-table calibration or SAM score is invalid",
        ));
    }
    let tables_root = image_root.join("tables");
    fs::create_dir_all(&tables_root).map_err(Error::backend)?;
    ensure_zarr_group(&tables_root)?;
    let destination = tables_root.join(name);
    let staging = tables_root.join(format!(
        ".{name}.rozt-{}-{serial}.partial",
        std::process::id()
    ));
    let backup = tables_root.join(format!(
        ".{name}.rozt-{}-{serial}.backup",
        std::process::id()
    ));
    if staging.exists() || backup.exists() {
        return Err(Error::invalid(
            "object-table transaction path already exists",
        ));
    }
    let (shape_zyx, mut rows) = if let (Some(existing), Some(affected_ids), Some(bounds)) =
        (existing_rows, affected_ids, affected_bounds)
    {
        let scores = existing
            .iter()
            .map(|row| (row.id, row.score))
            .chain(std::iter::once((selected_id, measurement.selected_score)))
            .collect::<HashMap<_, _>>();
        let (shape, measured) =
            measure_labels_in_bounds(label_staging, affected_ids, bounds, &scores)?;
        let mut rows = existing
            .iter()
            .filter(|row| !affected_ids.contains(&row.id))
            .cloned()
            .collect::<Vec<_>>();
        rows.extend(measured);
        (shape, rows)
    } else {
        let mut scores = existing_object_scores(&destination)?;
        scores.insert(selected_id, measurement.selected_score);
        measure_level_zero_labels(label_staging, &scores)?
    };
    let existing_classes = existing_rows
        .unwrap_or_default()
        .iter()
        .map(|row| (row.id, row.class_id))
        .collect::<HashMap<_, _>>();
    for row in &mut rows {
        row.class_id = if row.id == selected_id {
            measurement.class_id
        } else {
            existing_classes.get(&row.id).copied().unwrap_or(0)
        };
    }
    if let Err(error) = write_object_rows(
        &staging,
        name,
        shape_zyx,
        measurement.spacing_xyz,
        &mut rows,
    ) {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    Ok(StagedObjectTable {
        staging,
        destination,
        backup,
    })
}

fn existing_object_scores(table: &Path) -> Result<HashMap<u64, f32>> {
    if !table.join("zarr.json").is_file() {
        return Ok(HashMap::new());
    }
    let reader = TableReader::open(table).map_err(Error::backend)?;
    if !reader
        .spec()
        .columns
        .iter()
        .any(|column| column.name == "sam_score" && column.dtype == DType::F32)
    {
        return Ok(HashMap::new());
    }
    let count = reader.spec().row_count;
    let ids = reader
        .read_u64(&reader.spec().identity, 0..count)
        .map_err(Error::backend)?;
    let scores = reader
        .read_f32("sam_score", 0..count)
        .map_err(Error::backend)?;
    Ok(ids.into_iter().zip(scores).collect())
}

fn read_existing_object_rows(table: &Path) -> Result<Option<Vec<LabelObjectRow>>> {
    if !table.join("zarr.json").is_file() {
        return Ok(None);
    }
    let reader = TableReader::open(table).map_err(Error::backend)?;
    let count = reader.spec().row_count;
    let ids = reader
        .read_u64(&reader.spec().identity, 0..count)
        .map_err(Error::backend)?;
    let counts = reader
        .read_u64("volume_voxels", 0..count)
        .or_else(|_| reader.read_u64("area_pixels", 0..count))
        .map_err(Error::backend)?;
    let centroid_x = reader
        .read_f32("centroid_x", 0..count)
        .map_err(Error::backend)?;
    let centroid_y = reader
        .read_f32("centroid_y", 0..count)
        .map_err(Error::backend)?;
    let centroid_z = reader
        .read_f32("centroid_z", 0..count)
        .map_err(Error::backend)?;
    let min_x = reader
        .read_u64("bbox_x0", 0..count)
        .map_err(Error::backend)?;
    let min_y = reader
        .read_u64("bbox_y0", 0..count)
        .map_err(Error::backend)?;
    let min_z = reader
        .read_u64("bbox_z0", 0..count)
        .map_err(Error::backend)?;
    let max_x = reader
        .read_u64("bbox_x1", 0..count)
        .map_err(Error::backend)?;
    let max_y = reader
        .read_u64("bbox_y1", 0..count)
        .map_err(Error::backend)?;
    let max_z = reader
        .read_u64("bbox_z1", 0..count)
        .map_err(Error::backend)?;
    let scores = reader
        .read_f32("sam_score", 0..count)
        .unwrap_or_else(|_| vec![0.0; count as usize]);
    let class_ids = reader
        .read_u32("class_id", 0..count)
        .unwrap_or_else(|_| vec![0; count as usize]);
    let lengths = [
        counts.len(),
        centroid_x.len(),
        centroid_y.len(),
        centroid_z.len(),
        min_x.len(),
        min_y.len(),
        min_z.len(),
        max_x.len(),
        max_y.len(),
        max_z.len(),
        scores.len(),
        class_ids.len(),
    ];
    if ids.len() != count as usize || lengths.iter().any(|length| *length != ids.len()) {
        return Err(Error::invalid(
            "paired object table columns have inconsistent lengths",
        ));
    }
    Ok(Some(
        (0..ids.len())
            .map(|index| LabelObjectRow {
                id: ids[index],
                count: counts[index],
                sum_xyz: [0; 3],
                centroid_override: Some([centroid_x[index], centroid_y[index], centroid_z[index]]),
                min_xyz: [min_x[index], min_y[index], min_z[index]],
                max_xyz_exclusive: [max_x[index], max_y[index], max_z[index]],
                score: scores[index],
                class_id: class_ids[index],
            })
            .collect(),
    ))
}

fn label_ids_in_bounds(label_root: &Path, bounds: [u64; 6]) -> Result<HashSet<u64>> {
    let store = std::sync::Arc::new(FilesystemStore::new(label_root).map_err(Error::backend)?);
    let array = zarrs::array::Array::open(store, "/0").map_err(Error::backend)?;
    let shape: [u64; 3] = array
        .shape()
        .try_into()
        .map_err(|_| Error::invalid("level-zero label array must have ZYX rank three"))?;
    let bounds = clamp_zyx_bounds(bounds, shape);
    let mut ids = HashSet::new();
    for z in bounds[0]..bounds[3] {
        for y in (bounds[1]..bounds[4]).step_by(256) {
            for x in (bounds[2]..bounds[5]).step_by(256) {
                let part = [1, (bounds[4] - y).min(256), (bounds[5] - x).min(256)];
                let subset = ArraySubset::new_with_start_shape(vec![z, y, x], part.to_vec())
                    .map_err(Error::backend)?;
                ids.extend(
                    read_label_values_u64(&array, &subset)?
                        .into_iter()
                        .filter(|id| *id != 0),
                );
            }
        }
    }
    Ok(ids)
}

fn measure_labels_in_bounds(
    label_root: &Path,
    selected_ids: &HashSet<u64>,
    bounds: [u64; 6],
    scores: &HashMap<u64, f32>,
) -> Result<([u64; 3], Vec<LabelObjectRow>)> {
    let store = std::sync::Arc::new(FilesystemStore::new(label_root).map_err(Error::backend)?);
    let array = zarrs::array::Array::open(store, "/0").map_err(Error::backend)?;
    let shape: [u64; 3] = array
        .shape()
        .try_into()
        .map_err(|_| Error::invalid("level-zero label array must have ZYX rank three"))?;
    let bounds = clamp_zyx_bounds(bounds, shape);
    let mut rows = HashMap::<u64, LabelObjectRow>::new();
    for z in bounds[0]..bounds[3] {
        for y in (bounds[1]..bounds[4]).step_by(256) {
            for x in (bounds[2]..bounds[5]).step_by(256) {
                let part = [1, (bounds[4] - y).min(256), (bounds[5] - x).min(256)];
                let subset = ArraySubset::new_with_start_shape(vec![z, y, x], part.to_vec())
                    .map_err(Error::backend)?;
                for (index, id) in read_label_values_u64(&array, &subset)?
                    .into_iter()
                    .enumerate()
                {
                    if !selected_ids.contains(&id) {
                        continue;
                    }
                    let xyz = [
                        x + index as u64 % part[2],
                        y + (index as u64 / part[2]) % part[1],
                        z,
                    ];
                    let row = rows.entry(id).or_insert_with(|| LabelObjectRow {
                        id,
                        count: 0,
                        sum_xyz: [0; 3],
                        centroid_override: None,
                        min_xyz: xyz,
                        max_xyz_exclusive: xyz.map(|value| value + 1),
                        score: scores.get(&id).copied().unwrap_or(0.0),
                        class_id: 0,
                    });
                    row.count += 1;
                    for axis in 0..3 {
                        row.sum_xyz[axis] += u128::from(xyz[axis]);
                        row.min_xyz[axis] = row.min_xyz[axis].min(xyz[axis]);
                        row.max_xyz_exclusive[axis] =
                            row.max_xyz_exclusive[axis].max(xyz[axis] + 1);
                    }
                }
            }
        }
    }
    Ok((shape, rows.into_values().collect()))
}

fn clamp_zyx_bounds(bounds: [u64; 6], shape: [u64; 3]) -> [u64; 6] {
    [
        bounds[0].min(shape[0]),
        bounds[1].min(shape[1]),
        bounds[2].min(shape[2]),
        bounds[3].min(shape[0]),
        bounds[4].min(shape[1]),
        bounds[5].min(shape[2]),
    ]
}

fn measure_level_zero_labels(
    label_root: &Path,
    scores: &HashMap<u64, f32>,
) -> Result<([u64; 3], Vec<LabelObjectRow>)> {
    let store = std::sync::Arc::new(FilesystemStore::new(label_root).map_err(Error::backend)?);
    let array = zarrs::array::Array::open(store, "/0").map_err(Error::backend)?;
    let shape: [u64; 3] = array
        .shape()
        .try_into()
        .map_err(|_| Error::invalid("level-zero label array must have ZYX rank three"))?;
    let mut rows = HashMap::<u64, LabelObjectRow>::new();
    for z in 0..shape[0] {
        for y in (0..shape[1]).step_by(256) {
            for x in (0..shape[2]).step_by(256) {
                let part = [1, (shape[1] - y).min(256), (shape[2] - x).min(256)];
                let subset = ArraySubset::new_with_start_shape(vec![z, y, x], part.to_vec())
                    .map_err(Error::backend)?;
                let values = read_label_values_u64(&array, &subset)?;
                for (index, id) in values.into_iter().enumerate() {
                    if id == 0 {
                        continue;
                    }
                    let xyz = [
                        x + index as u64 % part[2],
                        y + (index as u64 / part[2]) % part[1],
                        z,
                    ];
                    let row = rows.entry(id).or_insert_with(|| LabelObjectRow {
                        id,
                        count: 0,
                        sum_xyz: [0; 3],
                        centroid_override: None,
                        min_xyz: xyz,
                        max_xyz_exclusive: xyz.map(|value| value + 1),
                        score: scores.get(&id).copied().unwrap_or(0.0),
                        class_id: 0,
                    });
                    row.count += 1;
                    for axis in 0..3 {
                        row.sum_xyz[axis] += u128::from(xyz[axis]);
                        row.min_xyz[axis] = row.min_xyz[axis].min(xyz[axis]);
                        row.max_xyz_exclusive[axis] =
                            row.max_xyz_exclusive[axis].max(xyz[axis] + 1);
                    }
                }
            }
        }
    }
    Ok((shape, rows.into_values().collect()))
}

fn read_label_values_u64(
    array: &zarrs::array::Array<FilesystemStore>,
    subset: &ArraySubset,
) -> Result<Vec<u64>> {
    if *array.data_type() == data_type::uint64() {
        array.retrieve_array_subset(subset).map_err(Error::backend)
    } else if *array.data_type() == data_type::uint32() {
        let values: Vec<u32> = array
            .retrieve_array_subset(subset)
            .map_err(Error::backend)?;
        Ok(values.into_iter().map(u64::from).collect())
    } else {
        Err(Error::invalid("label array is not uint32 or uint64"))
    }
}

fn write_object_rows(
    root: &Path,
    label_name: &str,
    shape_zyx: [u64; 3],
    spacing_xyz: [f64; 3],
    rows: &mut [LabelObjectRow],
) -> Result<()> {
    const ROW_CHUNK: usize = 2048;
    const TILE_ZYX: [u64; 3] = [32, 256, 256];
    let grid_shape = shape_zyx
        .into_iter()
        .zip(TILE_ZYX)
        .map(|(extent, tile)| extent.div_ceil(tile))
        .collect::<Vec<_>>();
    let centroid = |row: &LabelObjectRow, axis: usize| {
        row.centroid_override
            .map(|centroid| centroid[axis] as f64)
            .unwrap_or_else(|| row.sum_xyz[axis] as f64 / row.count.max(1) as f64)
    };
    let tile_id = |row: &LabelObjectRow| {
        let tile = [
            centroid(row, 2) as u64 / TILE_ZYX[0],
            centroid(row, 1) as u64 / TILE_ZYX[1],
            centroid(row, 0) as u64 / TILE_ZYX[2],
        ];
        (tile[0] * grid_shape[1] + tile[1]) * grid_shape[2] + tile[2]
    };
    rows.sort_unstable_by(|left, right| {
        tile_id(left)
            .cmp(&tile_id(right))
            .then(centroid(left, 2).total_cmp(&centroid(right, 2)))
            .then(centroid(left, 1).total_cmp(&centroid(right, 1)))
            .then(centroid(left, 0).total_cmp(&centroid(right, 0)))
            .then(left.id.cmp(&right.id))
    });
    let mut columns = vec![
        ColumnSpec::new("label_id", DType::U64, ColumnRole::Identity),
        ColumnSpec::new("centroid_z", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_y", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_x", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("area_pixels", DType::U64, ColumnRole::Measurement),
        ColumnSpec::new("area_um2", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("volume_voxels", DType::U64, ColumnRole::Measurement),
        ColumnSpec::new("volume_um3", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("sam_score", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("class_id", DType::U32, ColumnRole::Category),
    ];
    for axis in ["z", "y", "x"] {
        columns.push(ColumnSpec::new(
            format!("bbox_{axis}0"),
            DType::U64,
            ColumnRole::Measurement,
        ));
        columns.push(ColumnSpec::new(
            format!("bbox_{axis}1"),
            DType::U64,
            ColumnRole::Measurement,
        ));
    }
    for column in &mut columns {
        match column.name.as_str() {
            "area_um2" => column.unit = Some("µm²".into()),
            "volume_um3" => column.unit = Some("µm³".into()),
            _ => {}
        }
    }
    let spatial = SpatialIndexSpec {
        coordinates: ["z", "y", "x"]
            .into_iter()
            .map(|axis| CoordinateColumn {
                axis: axis.into(),
                column: format!("centroid_{axis}"),
            })
            .collect(),
        tile_shape: TILE_ZYX.to_vec(),
        grid_shape: grid_shape.clone(),
        tile_order: "row_major".into(),
        within_tile_order: "lexicographic_coordinates_then_identity".into(),
        downsample: None,
    };
    let mut spec = TableSpec::new(
        rows.len() as u64,
        ROW_CHUNK as u64,
        "../../",
        "label_id",
        columns,
        spatial,
    );
    spec.region = Some(format!("../../labels/{label_name}"));
    let writer = TableWriter::create(root, spec).map_err(Error::backend)?;
    let area_per_pixel = spacing_xyz[0] * spacing_xyz[1];
    let volume_per_voxel = area_per_pixel * spacing_xyz[2];
    for (chunk_index, chunk) in rows.chunks(ROW_CHUNK).enumerate() {
        let start = (chunk_index * ROW_CHUNK) as u64;
        writer
            .write_u64(
                "label_id",
                start,
                &chunk.iter().map(|row| row.id).collect::<Vec<_>>(),
            )
            .map_err(Error::backend)?;
        for (name, axis) in [("centroid_z", 2), ("centroid_y", 1), ("centroid_x", 0)] {
            writer
                .write_f32(
                    name,
                    start,
                    &chunk
                        .iter()
                        .map(|row| centroid(row, axis) as f32)
                        .collect::<Vec<_>>(),
                )
                .map_err(Error::backend)?;
        }
        for name in ["area_pixels", "volume_voxels"] {
            writer
                .write_u64(
                    name,
                    start,
                    &chunk.iter().map(|row| row.count).collect::<Vec<_>>(),
                )
                .map_err(Error::backend)?;
        }
        for (name, scale) in [
            ("area_um2", area_per_pixel),
            ("volume_um3", volume_per_voxel),
        ] {
            writer
                .write_f32(
                    name,
                    start,
                    &chunk
                        .iter()
                        .map(|row| (row.count as f64 * scale) as f32)
                        .collect::<Vec<_>>(),
                )
                .map_err(Error::backend)?;
        }
        writer
            .write_f32(
                "sam_score",
                start,
                &chunk.iter().map(|row| row.score).collect::<Vec<_>>(),
            )
            .map_err(Error::backend)?;
        writer
            .write_u32(
                "class_id",
                start,
                &chunk.iter().map(|row| row.class_id).collect::<Vec<_>>(),
            )
            .map_err(Error::backend)?;
        for (name, axis, maximum) in [
            ("bbox_z0", 2, false),
            ("bbox_z1", 2, true),
            ("bbox_y0", 1, false),
            ("bbox_y1", 1, true),
            ("bbox_x0", 0, false),
            ("bbox_x1", 0, true),
        ] {
            writer
                .write_u64(
                    name,
                    start,
                    &chunk
                        .iter()
                        .map(|row| {
                            if maximum {
                                row.max_xyz_exclusive[axis]
                            } else {
                                row.min_xyz[axis]
                            }
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(Error::backend)?;
        }
    }
    let mut identities = rows
        .iter()
        .enumerate()
        .map(|(physical, row)| (row.id, physical as u64))
        .collect::<Vec<_>>();
    identities.sort_unstable();
    for (chunk_index, chunk) in identities.chunks(ROW_CHUNK).enumerate() {
        writer
            .write_identity_index(
                (chunk_index * ROW_CHUNK) as u64,
                &chunk.iter().map(|row| row.0).collect::<Vec<_>>(),
                &chunk.iter().map(|row| row.1).collect::<Vec<_>>(),
            )
            .map_err(Error::backend)?;
    }
    let mut counts = vec![0_u64; grid_shape.iter().product::<u64>() as usize];
    for row in rows.iter() {
        counts[tile_id(row) as usize] += 1;
    }
    let mut starts = vec![0_u64; counts.len()];
    let mut next = 0_u64;
    for (start, count) in starts.iter_mut().zip(&counts) {
        *start = next;
        next = next
            .checked_add(*count)
            .ok_or_else(|| Error::invalid("object-table row count overflow"))?;
    }
    writer
        .write_spatial_index(&starts, &counts)
        .map_err(Error::backend)?;
    writer.finish().map_err(Error::backend)?;
    Ok(())
}

fn replace_label_and_table(
    labels_root: &Path,
    label_staging: &Path,
    label_destination: &Path,
    label_backup: &Path,
    table: StagedObjectTable,
) -> Result<()> {
    fs::rename(label_destination, label_backup).map_err(Error::backend)?;
    let had_table = table.destination.exists();
    if had_table {
        if let Err(error) = fs::rename(&table.destination, &table.backup) {
            let _ = fs::rename(label_backup, label_destination);
            return Err(Error::backend(error));
        }
    }
    let install = (|| {
        fs::rename(label_staging, label_destination).map_err(Error::backend)?;
        fs::rename(&table.staging, &table.destination).map_err(Error::backend)?;
        refresh_label_registry(labels_root)
    })();
    if let Err(error) = install {
        if table.destination.exists() {
            let _ = fs::rename(&table.destination, &table.staging);
        }
        if label_destination.exists() {
            let _ = fs::rename(label_destination, label_staging);
        }
        let _ = fs::rename(label_backup, label_destination);
        if had_table {
            let _ = fs::rename(&table.backup, &table.destination);
        }
        return Err(error);
    }
    let _ = fs::remove_dir_all(label_backup);
    if had_table {
        let _ = fs::remove_dir_all(&table.backup);
    }
    Ok(())
}

fn install_new_label_and_table(
    labels_root: &Path,
    label_staging: &Path,
    label_destination: &Path,
    table: StagedObjectTable,
) -> Result<()> {
    if table.destination.exists() {
        return Err(Error::invalid(format!(
            "object table {} already exists",
            table.destination.display()
        )));
    }
    fs::rename(label_staging, label_destination).map_err(Error::backend)?;
    let install = fs::rename(&table.staging, &table.destination)
        .map_err(Error::backend)
        .and_then(|()| refresh_label_registry(labels_root));
    if let Err(error) = install {
        if table.destination.exists() {
            let _ = fs::rename(&table.destination, &table.staging);
        }
        let _ = fs::rename(label_destination, label_staging);
        return Err(error);
    }
    Ok(())
}

fn ensure_zarr_group(root: &Path) -> Result<()> {
    let path = root.join("zarr.json");
    if path.exists() {
        return Ok(());
    }
    let temporary = root.join(format!(".zarr.json.{}.tmp", std::process::id()));
    let bytes = serde_json::to_vec_pretty(&serde_json::json!({
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {}
    }))
    .map_err(Error::backend)?;
    fs::write(&temporary, bytes).map_err(Error::backend)?;
    match fs::rename(&temporary, &path) {
        Ok(()) => Ok(()),
        Err(_) if path.exists() => {
            let _ = fs::remove_file(temporary);
            Ok(())
        }
        Err(error) => Err(Error::backend(error)),
    }
}

fn append_commit_provenance(root: &Path, provenance: &serde_json::Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(provenance).map_err(Error::backend)?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join(".rozt-commits.jsonl"))
        .map_err(Error::backend)?;
    file.write_all(&bytes).map_err(Error::backend)?;
    file.sync_data().map_err(Error::backend)
}

fn write_level(
    source: &OmeVolumePyramid,
    staging: &Path,
    level_index: usize,
    z_level_zero: usize,
    mask: &SparseBinaryMask<'_>,
    label_id: u64,
) -> Result<u64> {
    let level0 = &source.levels()[0];
    let level = &source.levels()[level_index];
    let scale = [
        level0.shape[0] as f64 / level.shape[0] as f64,
        level0.shape[1] as f64 / level.shape[1] as f64,
        level0.shape[2] as f64 / level.shape[2] as f64,
    ];
    let z = ((z_level_zero as f64 + 0.5) / scale[0]).floor() as usize;
    let x0 = (mask.origin_xy[0] / scale[2]).floor().max(0.0) as usize;
    let y0 = (mask.origin_xy[1] / scale[1]).floor().max(0.0) as usize;
    let x1 = ((mask.origin_xy[0] + mask.extent_xy[0]) / scale[2])
        .ceil()
        .clamp(0.0, level.shape[2] as f64) as usize;
    let y1 = ((mask.origin_xy[1] + mask.extent_xy[1]) / scale[1])
        .ceil()
        .clamp(0.0, level.shape[1] as f64) as usize;
    let shape = [1usize, y1.saturating_sub(y0), x1.saturating_sub(x0)];
    let store = std::sync::Arc::new(FilesystemStore::new(staging).map_err(Error::backend)?);
    let chunk = [
        1u64,
        level.shape[1].clamp(1, 256) as u64,
        level.shape[2].clamp(1, 256) as u64,
    ];
    let array = ArrayBuilder::new(
        level.shape.map(|v| v as u64).to_vec(),
        chunk.to_vec(),
        data_type::uint64(),
        FillValue::from(0u64),
    )
    .build(store, &format!("/{level_index}"))
    .map_err(Error::backend)?;
    array.store_metadata().map_err(Error::backend)?;
    if shape[1] == 0 || shape[2] == 0 || z >= level.shape[0] {
        return Ok(0);
    }
    let decoded = decode_mask(mask)?;
    let mut values = vec![0u64; shape[1] * shape[2]];
    for out_y in y0..y1 {
        for out_x in x0..x1 {
            let level0_x = (out_x as f64 + 0.5) * scale[2];
            let level0_y = (out_y as f64 + 0.5) * scale[1];
            let mask_x =
                ((level0_x - mask.origin_xy[0]) * mask.width as f64 / mask.extent_xy[0]).floor();
            let mask_y =
                ((level0_y - mask.origin_xy[1]) * mask.height as f64 / mask.extent_xy[1]).floor();
            if mask_x >= 0.0
                && mask_y >= 0.0
                && mask_x < mask.width as f64
                && mask_y < mask.height as f64
                && decoded[mask_y as usize * mask.width + mask_x as usize]
            {
                values[(out_y - y0) * shape[2] + out_x - x0] = label_id;
            }
        }
    }
    let subset = ArraySubset::new_with_start_shape(
        vec![z as u64, y0 as u64, x0 as u64],
        shape.map(|v| v as u64).to_vec(),
    )
    .map_err(Error::backend)?;
    let foreground = values.iter().filter(|&&value| value != 0).count() as u64;
    array
        .store_array_subset(&subset, &values)
        .map_err(Error::backend)?;
    Ok(foreground)
}

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.starts_with('.')
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
    {
        return Err(Error::invalid(
            "label result name must be one safe path component",
        ));
    }
    Ok(())
}

fn validate_mask(mask: &SparseBinaryMask<'_>) -> Result<()> {
    if mask.width == 0
        || mask.height == 0
        || mask.extent_xy.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || mask.origin_xy.iter().any(|v| !v.is_finite())
    {
        return Err(Error::invalid("mask geometry is empty or non-finite"));
    }
    let expected = mask
        .width
        .checked_mul(mask.height)
        .ok_or_else(|| Error::invalid("mask dimensions overflow"))?;
    let actual = mask
        .runs
        .iter()
        .try_fold(0usize, |sum, &run| sum.checked_add(run as usize))
        .ok_or_else(|| Error::invalid("mask run lengths overflow"))?;
    if actual != expected {
        return Err(Error::invalid(format!(
            "mask runs cover {actual} pixels, expected {expected}"
        )));
    }
    Ok(())
}

fn decode_mask(mask: &SparseBinaryMask<'_>) -> Result<Vec<bool>> {
    validate_mask(mask)?;
    let mut decoded = Vec::with_capacity(mask.width * mask.height);
    let mut foreground = false;
    for &run in mask.runs {
        decoded.extend(std::iter::repeat_n(foreground, run as usize));
        foreground = !foreground;
    }
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRoot(PathBuf);

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn source_pyramid() -> (TestRoot, OmeVolumePyramid) {
        let root = std::env::temp_dir().join(format!(
            "blockflow-label-pair-{}-{}",
            std::process::id(),
            STAGING_SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("zarr.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "zarr_format":3,"node_type":"group","attributes":{"ome":{
                    "version":"0.5","multiscales":[{"version":"0.5",
                    "axes":[{"name":"z","type":"space","unit":"micrometer"},
                            {"name":"y","type":"space","unit":"micrometer"},
                            {"name":"x","type":"space","unit":"micrometer"}],
                    "datasets":[{"path":"0","coordinateTransformations":[{
                        "type":"scale","scale":[2.0,0.5,0.5]}]}]}]}}
            }))
            .unwrap(),
        )
        .unwrap();
        let store = std::sync::Arc::new(FilesystemStore::new(&root).unwrap());
        let array = ArrayBuilder::new(
            vec![2, 8, 8],
            vec![1, 8, 8],
            data_type::uint8(),
            FillValue::from(0_u8),
        )
        .build(store, "/0")
        .unwrap();
        array.store_metadata().unwrap();
        let pyramid = OmeVolumePyramid::open(&root, 0, 0).unwrap();
        (TestRoot(root), pyramid)
    }

    #[test]
    fn paired_sparse_publish_and_edit_keep_native_objects_exact() {
        let (_root, pyramid) = source_pyramid();
        let label_id = u64::from(u32::MAX) + 17;
        let runs = [0, 4];
        let mask = || SparseBinaryMask {
            width: 2,
            height: 2,
            origin_xy: [1.0, 2.0],
            extent_xy: [2.0, 2.0],
            runs: &runs,
        };
        publish_sparse_plane_segmentation(
            &pyramid,
            "cells",
            0,
            mask(),
            label_id,
            ObjectTableMeasurement {
                spacing_xyz: [0.5, 0.5, 2.0],
                selected_score: 0.93,
                class_id: 2,
            },
            None,
        )
        .unwrap();
        let table = TableReader::open(pyramid.image_root().join("tables/cells")).unwrap();
        assert_eq!(table.read_u64("label_id", 0..1).unwrap(), [label_id]);
        assert_eq!(table.read_u64("area_pixels", 0..1).unwrap(), [4]);
        assert_eq!(table.read_f32("area_um2", 0..1).unwrap(), [1.0]);
        assert_eq!(table.read_f32("sam_score", 0..1).unwrap(), [0.93]);

        edit_sparse_plane_segmentation(
            &pyramid,
            "cells",
            0,
            mask(),
            label_id,
            LabelEditMode::Subtract,
            ObjectTableMeasurement {
                spacing_xyz: [0.5, 0.5, 2.0],
                selected_score: 0.5,
                class_id: 2,
            },
            None,
        )
        .unwrap();
        let table = TableReader::open(pyramid.image_root().join("tables/cells")).unwrap();
        assert_eq!(table.spec().row_count, 0);
    }

    #[test]
    fn propagated_volume_publishes_one_calibrated_3d_object_transaction() {
        let (_root, pyramid) = source_pyramid();
        let runs = [0, 4];
        let masks = [0_usize, 1]
            .into_iter()
            .map(|z| {
                (
                    z,
                    SparseBinaryMask {
                        width: 2,
                        height: 2,
                        origin_xy: [1.0, 2.0],
                        extent_xy: [2.0, 2.0],
                        runs: &runs,
                    },
                )
            })
            .collect::<Vec<_>>();
        let published = publish_sparse_volume_segmentation(
            &pyramid,
            "propagated",
            &masks,
            9,
            ObjectTableMeasurement {
                spacing_xyz: [0.5, 0.5, 2.0],
                selected_score: 1.0,
                class_id: 3,
            },
            None,
        )
        .unwrap();
        assert_eq!(published.foreground_voxels, 8);
        assert!(pyramid.image_root().join("labels/propagated/0").is_dir());
        let table = TableReader::open(pyramid.image_root().join("tables/propagated")).unwrap();
        assert_eq!(table.read_u64("label_id", 0..1).unwrap(), [9]);
        assert_eq!(table.read_u64("volume_voxels", 0..1).unwrap(), [8]);
        assert_eq!(table.read_f32("volume_um3", 0..1).unwrap(), [4.0]);
    }

    #[test]
    fn measurement_job_publishes_owned_native_table_from_existing_labels() {
        let (_root, pyramid) = source_pyramid();
        let runs = [0, 4];
        publish_sparse_plane_label(
            &pyramid,
            "source-labels",
            1,
            SparseBinaryMask {
                width: 2,
                height: 2,
                origin_xy: [2.0, 3.0],
                extent_xy: [2.0, 2.0],
                runs: &runs,
            },
            42,
            None,
        )
        .unwrap();
        let result = publish_object_table_for_label(
            &pyramid,
            "source-labels",
            "measurements",
            "job-7",
            false,
        )
        .unwrap();
        assert_eq!(result.objects, 1);
        let table = TableReader::open(pyramid.image_root().join("tables/measurements")).unwrap();
        assert_eq!(table.read_u64("label_id", 0..1).unwrap(), [42]);
        assert_eq!(table.read_u64("area_pixels", 0..1).unwrap(), [4]);
        assert_eq!(table.read_f32("area_um2", 0..1).unwrap(), [1.0]);
        let marker: serde_json::Value = serde_json::from_slice(
            &fs::read(
                pyramid
                    .image_root()
                    .join("tables/measurements/.rozt-job-owner.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(marker["jobId"], "job-7");
    }
}
