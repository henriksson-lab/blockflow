// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! **Time-lapse OME-Zarr images, one frame at a time.**
//!
//! Time is a storage axis and a collection key, never a block-grid axis: a
//! frame is handed to Blockflow as an ordinary rank-three `[Z, Y, X]` image
//! (`Z = 1` for 2-D), and the three grid slots stay spatial so that time plus
//! 3-D needs no change. See `TRACKING_PLAN.md`.
//!
//! | piece | what it is |
//! |---|---|
//! | [`TimeSeriesImage`] | a `t[,c][,z],y,x` multiscale image: axes, levels, spacing, timeline, one frame as an [`AttachedImage`] |
//! | [`LabelStack`] | a `t[,z],y,x` `uint64` label pyramid under construction: one plane per frame, a stored time chunk of one |
//!
//! # Why a time chunk of one
//!
//! Frames are segmented concurrently, each writing its own plane. With one
//! time index per chunk, two frames never touch the same compressed chunk, so
//! no frame read-modify-writes another's bytes; and clearing a frame for a
//! rerun is deleting its chunks, which no other frame shares.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use zarrs::array::{Array as ZarrArray, ArraySubset};
use zarrs::filesystem::FilesystemStore;

use crate::track::Frame;
use crate::zarr_env::{Compression, SpatialAxis};
use crate::zarr_layout::StorageLayout;
use crate::{AttachedImage, Dtype, Error, Result};

/// One level of a time-series image.
#[derive(Clone, Debug)]
pub struct TimeLevel {
    pub path: String,
    /// The stored array's full shape, in source axis order.
    pub shape: Vec<usize>,
    pub coordinate_transformations: Value,
}

/// A multiscale image with a time axis, opened from its metadata.
#[derive(Clone, Debug)]
pub struct TimeSeriesImage {
    root: PathBuf,
    axes: Vec<Value>,
    names: Vec<String>,
    t: Option<usize>,
    c: Option<usize>,
    z: Option<usize>,
    y: usize,
    x: usize,
    levels: Vec<TimeLevel>,
}

impl TimeSeriesImage {
    /// Open the multiscale image group at `root`. Accepts `t`, `c` and `z`
    /// as optional axes; `y` and `x` are required, spatial axes must be in
    /// `z, y, x` order, and any other axis must have length one.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let metadata = read_json(&root.join("zarr.json"))?;
        let multiscale = metadata
            .pointer("/attributes/ome/multiscales/0")
            .or_else(|| metadata.pointer("/attributes/multiscales/0"))
            .ok_or_else(|| {
                Error::invalid(format!("{} has no OME multiscales entry", root.display()))
            })?;
        let axes = multiscale
            .get("axes")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::invalid("OME multiscales entry has no axes"))?
            .clone();
        let names = axes
            .iter()
            .map(|axis| {
                axis.as_str()
                    .or_else(|| axis.get("name").and_then(Value::as_str))
                    .map(str::to_ascii_lowercase)
                    .ok_or_else(|| Error::invalid("OME axis has no name"))
            })
            .collect::<Result<Vec<_>>>()?;
        let find = |name: &str| names.iter().position(|n| n == name);
        let (Some(y), Some(x)) = (find("y"), find("x")) else {
            return Err(Error::invalid(format!(
                "OME image {} needs y and x axes; it has {names:?}",
                root.display()
            )));
        };
        let z = find("z");
        if z.is_some_and(|z| z > y) || y > x {
            return Err(Error::invalid(format!(
                "OME spatial axes must occur in z,y,x order; got {names:?}"
            )));
        }
        let mut levels = Vec::new();
        for dataset in multiscale
            .get("datasets")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::invalid("OME multiscales entry has no datasets"))?
        {
            let path = dataset
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::invalid("multiscale dataset has no path"))?
                .to_owned();
            let array = read_json(&root.join(&path).join("zarr.json"))?;
            let shape = array
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::invalid(format!("OME level {path} has no shape")))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|v| v as usize)
                        .ok_or_else(|| Error::invalid(format!("OME level {path} has a bad shape")))
                })
                .collect::<Result<Vec<_>>>()?;
            if shape.len() != names.len() {
                return Err(Error::invalid(format!(
                    "OME level {path} has rank {} but {} axes",
                    shape.len(),
                    names.len()
                )));
            }
            for (axis, name) in names.iter().enumerate() {
                if !["t", "c", "z", "y", "x"].contains(&name.as_str()) && shape[axis] != 1 {
                    return Err(Error::invalid(format!(
                        "OME axis {name} has length {}; only t, c, z, y, x may be longer than one",
                        shape[axis]
                    )));
                }
            }
            levels.push(TimeLevel {
                path,
                shape,
                coordinate_transformations: dataset
                    .get("coordinateTransformations")
                    .cloned()
                    .unwrap_or(Value::Array(Vec::new())),
            });
        }
        if levels.is_empty() {
            return Err(Error::invalid("OME image has no levels"));
        }
        let t = find("t");
        let c = find("c");
        let first = levels[0].shape.clone();
        for level in &levels {
            for axis in [t, c].into_iter().flatten() {
                if level.shape[axis] != first[axis] {
                    return Err(Error::invalid(
                        "OME pyramid changes the t or c extent between levels",
                    ));
                }
            }
        }
        Ok(Self {
            root,
            axes,
            names,
            t,
            c,
            z,
            y,
            x,
            levels,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn levels(&self) -> &[TimeLevel] {
        &self.levels
    }

    /// Frames on the stored time axis; one if there is none.
    pub fn n_frames(&self) -> usize {
        self.t.map_or(1, |t| self.levels[0].shape[t])
    }

    /// Channels; one if there is no `c` axis.
    pub fn n_channels(&self) -> usize {
        self.c.map_or(1, |c| self.levels[0].shape[c])
    }

    /// Whether the image has a real `z` axis.
    pub fn has_z(&self) -> bool {
        self.z.is_some()
    }

    /// The rank-three `[Z, Y, X]` frame extent at `level`.
    pub fn frame_shape(&self, level: usize) -> Result<[usize; 3]> {
        let level = self.level(level)?;
        Ok([
            self.z.map_or(1, |z| level.shape[z]),
            level.shape[self.y],
            level.shape[self.x],
        ])
    }

    /// Physical voxel size `[z, y, x]` at `level`, from its scale transform;
    /// `1` for an absent axis or an absent transform.
    pub fn spacing(&self, level: usize) -> Result<[f64; 3]> {
        let scale = self.scale(level)?;
        Ok([
            self.z.map_or(1.0, |z| scale[z]),
            scale[self.y],
            scale[self.x],
        ])
    }

    /// The acquired timeline: one [`Frame`] per stored time index. Elapsed
    /// seconds come from the `t` scale and unit when the unit is a time unit.
    pub fn timeline(&self) -> Result<Vec<Frame>> {
        let seconds_per_index = match self.t {
            None => None,
            Some(t) => {
                let unit = self.axes[t].get("unit").and_then(Value::as_str);
                unit.and_then(seconds_per_unit)
                    .map(|factor| factor * self.scale(0).map(|s| s[t]).unwrap_or(1.0))
            }
        };
        Ok((0..self.n_frames() as u64)
            .map(|t_index| Frame {
                t_index,
                elapsed_seconds: seconds_per_index.map(|s| s * t_index as f64),
            })
            .collect())
    }

    /// One frame of one channel at `level`, as a rank-three image.
    pub fn frame(&self, level: usize, channel: usize, t: usize) -> Result<AttachedImage> {
        let stored = self.level(level)?;
        if channel >= self.n_channels() || t >= self.n_frames() {
            return Err(Error::invalid(format!(
                "frame (c={channel}, t={t}) is outside {} channels x {} frames",
                self.n_channels(),
                self.n_frames()
            )));
        }
        let mut fixed = Vec::new();
        for (axis, name) in self.names.iter().enumerate() {
            match name.as_str() {
                "y" | "x" | "z" => {}
                "t" => fixed.push((axis, t)),
                "c" => fixed.push((axis, channel)),
                _ => fixed.push((axis, 0)),
            }
        }
        let first = self.z.map_or(SpatialAxis::Singleton, SpatialAxis::Source);
        Ok(AttachedImage::at(self.root.join(&stored.path)).selected(
            [
                first,
                SpatialAxis::Source(self.y),
                SpatialAxis::Source(self.x),
            ],
            fixed,
        ))
    }

    /// Axis metadata for a label image of this series: the source axes with
    /// `c` and any singleton non-spatial axis dropped, so `t[,z],y,x`. `None`
    /// is a `t` the source does not have — a single-frame image — which the
    /// label stack still carries, with extent one.
    fn label_axes(&self) -> Vec<Option<usize>> {
        let mut axes: Vec<Option<usize>> = (0..self.names.len())
            .filter(|&axis| {
                Some(axis) == self.t || Some(axis) == self.z || axis == self.y || axis == self.x
            })
            .map(Some)
            .collect();
        if self.t.is_none() {
            axes.insert(0, None);
        }
        axes
    }

    fn level(&self, level: usize) -> Result<&TimeLevel> {
        self.levels
            .get(level)
            .ok_or_else(|| Error::invalid(format!("OME image has no level {level}")))
    }

    fn scale(&self, level: usize) -> Result<Vec<f64>> {
        let rank = self.names.len();
        let scale = self
            .level(level)?
            .coordinate_transformations
            .as_array()
            .and_then(|rows| {
                rows.iter()
                    .find(|row| row.get("type").and_then(Value::as_str) == Some("scale"))
            })
            .and_then(|row| row.get("scale"))
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(1.0))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![1.0; rank]);
        if scale.len() != rank {
            return Err(Error::invalid(format!(
                "scale transform has {} entries for {rank} axes",
                scale.len()
            )));
        }
        Ok(scale)
    }
}

fn seconds_per_unit(unit: &str) -> Option<f64> {
    Some(match unit {
        "second" | "s" => 1.0,
        "millisecond" | "ms" => 1e-3,
        "microsecond" => 1e-6,
        "nanosecond" => 1e-9,
        "minute" | "min" => 60.0,
        "hour" | "h" => 3600.0,
        "day" => 86400.0,
        _ => return None,
    })
}

/// A `t[,z],y,x` `uint64` label pyramid, one plane per frame.
///
/// Created hidden, filled frame by frame through [`Self::plane`] used as a
/// [`crate::ZarrEnvironment::with_output_target`], and published by renaming
/// the whole group into place when every frame and level is complete.
#[derive(Clone, Debug)]
pub struct LabelStack {
    root: PathBuf,
    has_z: bool,
    /// Stored shape per level, `[t, (z,) y, x]`.
    shapes: Vec<Vec<usize>>,
}

impl LabelStack {
    /// Create the group and every level array (metadata only; no chunk is
    /// written, so every plane reads as background until a frame writes it).
    ///
    /// Levels mirror `source`'s levels, so label level `k` lines up with image
    /// level `k` voxel for voxel and the transforms carry over unchanged.
    /// `spatial_chunk` is the `[y, x]` chunk; `z` is one chunk deep and `t`
    /// one index per chunk. Refuses an existing `root`.
    pub fn create(
        root: impl Into<PathBuf>,
        source: &TimeSeriesImage,
        spatial_chunk: [usize; 2],
        source_reference: &str,
    ) -> Result<Self> {
        let root = root.into();
        if root.exists() {
            return Err(Error::invalid(format!(
                "label stack {} already exists",
                root.display()
            )));
        }
        if spatial_chunk.contains(&0) {
            return Err(Error::invalid("label chunk must be positive"));
        }
        fs::create_dir_all(&root).map_err(Error::backend)?;
        let frames = source.n_frames();
        let keep = source.label_axes();
        let mut shapes = Vec::new();
        let mut datasets = Vec::new();
        for (index, level) in source.levels().iter().enumerate() {
            let [z, y, x] = source.frame_shape(index)?;
            let mut shape = vec![frames];
            let mut chunk = vec![1];
            if source.has_z() {
                shape.push(z);
                chunk.push(z);
            }
            shape.extend([y, x]);
            chunk.extend([spatial_chunk[0].min(y), spatial_chunk[1].min(x)]);
            let names: Vec<&str> = if source.has_z() {
                vec!["t", "z", "y", "x"]
            } else {
                vec!["t", "y", "x"]
            };
            let metadata = StorageLayout::chunked(chunk, Compression::for_dtype(Dtype::U64))
                .array_metadata(&shape, Dtype::U64, json!(0), Some(json!(names)))?;
            let dir = root.join(index.to_string());
            fs::create_dir_all(&dir).map_err(Error::backend)?;
            write_json(&dir.join("zarr.json"), &metadata)?;
            datasets.push(json!({
                "path": index.to_string(),
                "coordinateTransformations": project(&level.coordinate_transformations, &keep),
            }));
            shapes.push(shape);
        }
        let axes: Vec<Value> = keep
            .iter()
            .map(|&axis| match axis.map(|axis| &source.axes[axis]) {
                None => json!({"name": "t", "type": "time"}),
                Some(Value::String(name)) => json!({"name": name}),
                Some(other) => other.clone(),
            })
            .collect();
        let group = json!({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": {
                    "version": "0.5",
                    "multiscales": [{"axes": axes, "datasets": datasets}],
                },
                "image-label": {"version": "0.5", "source": {"image": source_reference}},
            }
        });
        write_json(&root.join("zarr.json"), &group)?;
        Ok(Self {
            root,
            has_z: source.has_z(),
            shapes,
        })
    }

    /// Open an existing stack, e.g. on resume.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let metadata = read_json(&root.join("zarr.json"))?;
        let multiscale = metadata
            .pointer("/attributes/ome/multiscales/0")
            .ok_or_else(|| Error::invalid(format!("{} is not a label stack", root.display())))?;
        let rank = multiscale
            .get("axes")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        if rank != 3 && rank != 4 {
            return Err(Error::invalid(format!(
                "{} has rank {rank}; a label stack is t,y,x or t,z,y,x",
                root.display()
            )));
        }
        let mut shapes = Vec::new();
        for index in 0.. {
            let path = root.join(index.to_string()).join("zarr.json");
            if !path.is_file() {
                break;
            }
            let array = read_json(&path)?;
            let shape: Vec<usize> = array
                .get("shape")
                .and_then(Value::as_array)
                .map(|v| {
                    v.iter()
                        .filter_map(Value::as_u64)
                        .map(|v| v as usize)
                        .collect()
                })
                .unwrap_or_default();
            let chunk_t = array
                .pointer("/chunk_grid/configuration/chunk_shape/0")
                .and_then(Value::as_u64);
            if shape.len() != rank || chunk_t != Some(1) {
                return Err(Error::invalid(format!(
                    "{} is not a label-stack level with a time chunk of one",
                    path.display()
                )));
            }
            shapes.push(shape);
        }
        if shapes.is_empty() {
            return Err(Error::invalid(format!("{} has no levels", root.display())));
        }
        Ok(Self {
            root,
            has_z: rank == 4,
            shapes,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn n_levels(&self) -> usize {
        self.shapes.len()
    }

    pub fn n_frames(&self) -> usize {
        self.shapes[0][0]
    }

    /// The `[Z, Y, X]` frame extent at `level`.
    pub fn frame_shape(&self, level: usize) -> Result<[usize; 3]> {
        let shape = self.level_shape(level)?;
        Ok(if self.has_z {
            [shape[1], shape[2], shape[3]]
        } else {
            [1, shape[1], shape[2]]
        })
    }

    /// Plane `t` of `level` as a rank-three image — to write into with
    /// [`crate::ZarrEnvironment::with_output_target`], or to read back as an
    /// input.
    pub fn plane(&self, level: usize, t: usize) -> Result<AttachedImage> {
        let shape = self.level_shape(level)?;
        if t >= shape[0] {
            return Err(Error::invalid(format!(
                "frame {t} is outside a stack of {} frames",
                shape[0]
            )));
        }
        let axes = if self.has_z {
            [
                SpatialAxis::Source(1),
                SpatialAxis::Source(2),
                SpatialAxis::Source(3),
            ]
        } else {
            [
                SpatialAxis::Singleton,
                SpatialAxis::Source(1),
                SpatialAxis::Source(2),
            ]
        };
        Ok(AttachedImage::at(self.root.join(level.to_string())).selected(axes, vec![(0, t)]))
    }

    /// Delete every stored chunk of plane `t` at every level. Safe because the
    /// time chunk is one: no other frame shares these chunks. Used before
    /// rerunning a frame that has no valid completion manifest, so nothing an
    /// interrupted run wrote can survive into the rerun.
    pub fn clear_plane(&self, t: usize) -> Result<()> {
        for level in 0..self.n_levels() {
            let dir = self
                .root
                .join(level.to_string())
                .join("c")
                .join(t.to_string());
            match fs::remove_dir_all(&dir) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(Error::backend(error)),
            }
        }
        Ok(())
    }

    /// Delete the stored chunks of plane `t` at one level only — for
    /// rebuilding a derived pyramid level without touching level 0.
    pub fn clear_level_plane(&self, level: usize, t: usize) -> Result<()> {
        self.level_shape(level)?;
        let dir = self
            .root
            .join(level.to_string())
            .join("c")
            .join(t.to_string());
        match fs::remove_dir_all(&dir) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(Error::backend(error)),
        }
    }

    /// A content hash (FNV-1a, 64 bit) of plane `t` at `level`, read in
    /// bounded `y` bands. Fill-value chunks hash as the zeros they read as, so
    /// the hash is a function of the values and not of which chunks happen to
    /// be stored.
    pub fn plane_hash(&self, level: usize, t: usize) -> Result<u64> {
        let shape = self.level_shape(level)?.to_vec();
        let dir = self.root.join(level.to_string());
        let store = std::sync::Arc::new(FilesystemStore::new(&dir).map_err(Error::backend)?);
        let array = ZarrArray::open(store, "/").map_err(Error::backend)?;
        let y_axis = shape.len() - 2;
        let band = 256usize;
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut y = 0;
        while y < shape[y_axis] {
            let rows = band.min(shape[y_axis] - y);
            let mut start = vec![t as u64];
            let mut extent = vec![1u64];
            for (axis, &length) in shape.iter().enumerate().skip(1) {
                if axis == y_axis {
                    start.push(y as u64);
                    extent.push(rows as u64);
                } else {
                    start.push(0);
                    extent.push(length as u64);
                }
            }
            let subset =
                ArraySubset::new_with_start_shape(start, extent).map_err(Error::backend)?;
            let values: Vec<u64> = array
                .retrieve_array_subset(&subset)
                .map_err(Error::backend)?;
            for value in values {
                for byte in value.to_le_bytes() {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x0100_0000_01b3);
                }
            }
            y += rows;
        }
        Ok(hash)
    }

    /// Read plane `t` of `level` whole, as `[Z, Y, X]`. Holds one frame in
    /// memory; for frames too large for that, attach [`Self::plane`] to a plan
    /// instead.
    pub fn read_plane(&self, level: usize, t: usize) -> Result<ndarray::Array3<u64>> {
        let (array, subset) = self.plane_subset(level, t)?;
        let values: Vec<u64> = array
            .retrieve_array_subset(&subset)
            .map_err(Error::backend)?;
        let [z, y, x] = self.frame_shape(level)?;
        ndarray::Array3::from_shape_vec((z, y, x), values).map_err(Error::backend)
    }

    /// Write plane `t` of `level` whole. The caller owns the plane: no other
    /// writer may touch the same `t` concurrently (a time chunk of one makes
    /// different frames independent).
    pub fn write_plane(&self, level: usize, t: usize, values: &ndarray::Array3<u64>) -> Result<()> {
        let shape = self.frame_shape(level)?;
        if values.shape() != shape {
            return Err(Error::invalid(format!(
                "plane is {:?}; level {level} frames are {shape:?}",
                values.shape()
            )));
        }
        let (array, subset) = self.plane_subset(level, t)?;
        let standard = values.as_standard_layout();
        array
            .store_array_subset(&subset, standard.as_slice().expect("standard layout"))
            .map_err(Error::backend)
    }

    fn plane_subset(
        &self,
        level: usize,
        t: usize,
    ) -> Result<(ZarrArray<FilesystemStore>, ArraySubset)> {
        let shape = self.level_shape(level)?.to_vec();
        if t >= shape[0] {
            return Err(Error::invalid(format!(
                "frame {t} is outside a stack of {} frames",
                shape[0]
            )));
        }
        let dir = self.root.join(level.to_string());
        let store = std::sync::Arc::new(FilesystemStore::new(&dir).map_err(Error::backend)?);
        let array = ZarrArray::open(store, "/").map_err(Error::backend)?;
        let mut start = vec![t as u64];
        start.extend(std::iter::repeat(0).take(shape.len() - 1));
        let mut extent = vec![1u64];
        extent.extend(shape[1..].iter().map(|&n| n as u64));
        let subset = ArraySubset::new_with_start_shape(start, extent).map_err(Error::backend)?;
        Ok((array, subset))
    }

    fn level_shape(&self, level: usize) -> Result<&[usize]> {
        self.shapes
            .get(level)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::invalid(format!("label stack has no level {level}")))
    }
}

/// Keep the entries of each scale/translation transform for `keep` axes;
/// `None` is an axis the source lacks (scale 1, translation 0).
fn project(transforms: &Value, keep: &[Option<usize>]) -> Value {
    Value::Array(
        transforms
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|transform| {
                let kind = transform.get("type")?.as_str()?;
                let key = match kind {
                    "scale" | "translation" => kind,
                    _ => return None,
                };
                let values = transform.get(key)?.as_array()?;
                let projected: Vec<Value> = keep
                    .iter()
                    .map(|&axis| {
                        axis.and_then(|axis| values.get(axis).cloned())
                            .unwrap_or(json!(if kind == "scale" { 1 } else { 0 }))
                    })
                    .collect();
                let mut object = Map::new();
                object.insert("type".into(), Value::String(kind.into()));
                object.insert(key.into(), Value::Array(projected));
                Some(Value::Object(object))
            })
            .collect(),
    )
}

pub(crate) fn read_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path)
        .map_err(|error| Error::backend(format!("reading {}: {error}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| Error::backend(format!("parsing {}: {error}", path.display())))
}

pub(crate) fn write_json(path: &Path, value: &Value) -> Result<()> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(Error::backend)?,
    )
    .map_err(|error| Error::backend(format!("writing {}: {error}", path.display())))
}
