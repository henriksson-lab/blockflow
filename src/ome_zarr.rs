//! Axis-aware access to one scalar `[z, y, x]` volume in an OME-Zarr pyramid.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::{AttachedImage, Error, Result};

#[derive(Clone, Debug)]
pub struct OmeVolumeLevel {
    pub path: String,
    pub shape: [usize; 3],
    pub coordinate_transformations: Value,
}

#[derive(Clone, Debug)]
pub struct OmeVolumePyramid {
    root: PathBuf,
    label_source: String,
    spatial_axes: [usize; 3],
    fixed: Vec<(usize, usize)>,
    levels: Vec<OmeVolumeLevel>,
}

impl OmeVolumePyramid {
    /// Select one channel and time point from the first OME multiscale image.
    /// Missing `c` or `t` axes are accepted only for index zero. Any other
    /// non-spatial axis must have length one.
    pub fn open(root: impl Into<PathBuf>, channel: usize, time: usize) -> Result<Self> {
        let container_root = root.into();
        let (root, metadata) = locate_image_group(&container_root)?;
        let image_path = root.strip_prefix(&container_root).map_err(|_| {
            Error::invalid(format!(
                "OME image group {} is outside container {}",
                root.display(),
                container_root.display()
            ))
        })?;
        let label_source = if image_path.as_os_str().is_empty() {
            "../../".to_owned()
        } else {
            format!("../../{}", image_path.to_string_lossy())
        };
        let multiscale = metadata
            .pointer("/attributes/ome/multiscales/0")
            .or_else(|| metadata.pointer("/attributes/multiscales/0"))
            .ok_or_else(|| Error::invalid("root metadata has no OME multiscales entry"))?;
        let axes = multiscale
            .get("axes")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::invalid("OME multiscales entry has no axes"))?
            .iter()
            .map(|axis| {
                axis.as_str()
                    .or_else(|| axis.get("name").and_then(Value::as_str))
                    .map(str::to_ascii_lowercase)
                    .ok_or_else(|| Error::invalid("OME axis has no name"))
            })
            .collect::<Result<Vec<_>>>()?;
        let axis = |name: &str| {
            axes.iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| Error::invalid(format!("OME image has no {name} axis")))
        };
        let spatial_axes = [axis("z")?, axis("y")?, axis("x")?];
        if !(spatial_axes[0] < spatial_axes[1] && spatial_axes[1] < spatial_axes[2]) {
            return Err(Error::invalid(format!(
                "OME spatial axes must occur in z,y,x order; got {spatial_axes:?} in {axes:?}"
            )));
        }
        let datasets = multiscale
            .get("datasets")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::invalid("OME multiscales entry has no datasets"))?;
        let mut levels = Vec::with_capacity(datasets.len());
        let mut fixed = None;
        for dataset in datasets {
            let path = dataset
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::invalid("multiscale dataset has no path"))?
                .to_owned();
            let array = read_json(&root.join(&path).join("zarr.json"))?;
            let source_shape = array
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::invalid(format!("OME level {path} has no array shape")))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| {
                            Error::invalid(format!("OME level {path} has invalid shape"))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            if source_shape.len() != axes.len() {
                return Err(Error::invalid(format!(
                    "OME level {path} has rank {}, but {} axes are declared",
                    source_shape.len(),
                    axes.len()
                )));
            }
            let selected = fixed_indices(&axes, &source_shape, spatial_axes, channel, time)?;
            if let Some(expected) = &fixed {
                if expected != &selected {
                    return Err(Error::invalid(
                        "OME pyramid changes non-spatial axes between levels",
                    ));
                }
            } else {
                fixed = Some(selected);
            }
            levels.push(OmeVolumeLevel {
                path,
                shape: spatial_axes.map(|axis| source_shape[axis]),
                coordinate_transformations: project_transforms(
                    dataset.get("coordinateTransformations"),
                    spatial_axes,
                ),
            });
        }
        Ok(Self {
            root,
            label_source,
            spatial_axes,
            fixed: fixed.unwrap_or_default(),
            levels,
        })
    }

    pub fn levels(&self) -> &[OmeVolumeLevel] {
        &self.levels
    }

    /// Relative NGFF image path from a sibling `labels/<name>` group.
    pub fn label_source(&self) -> &str {
        &self.label_source
    }

    pub fn attached_level(&self, level: usize) -> Result<AttachedImage> {
        let level = self
            .levels
            .get(level)
            .ok_or_else(|| Error::invalid(format!("OME pyramid has no level {level}")))?;
        Ok(AttachedImage::at(self.root.join(&level.path))
            .volume(self.spatial_axes, self.fixed.clone()))
    }

    pub fn voxel_volume(&self) -> f64 {
        self.voxel_size().iter().product()
    }

    /// Level-0 physical spacing in `[z, y, x]` order. Unitless arrays use one.
    pub fn voxel_size(&self) -> [f64; 3] {
        self.levels
            .first()
            .and_then(|level| level.coordinate_transformations.as_array())
            .and_then(|rows| {
                rows.iter()
                    .find(|row| row.get("type").and_then(Value::as_str) == Some("scale"))
            })
            .and_then(|row| row.get("scale"))
            .and_then(Value::as_array)
            .and_then(|scale| {
                Some([
                    scale.first()?.as_f64()?,
                    scale.get(1)?.as_f64()?,
                    scale.get(2)?.as_f64()?,
                ])
            })
            .unwrap_or([1.0; 3])
    }
}

/// Write OME-Zarr 0.5 metadata for a label pyramid whose numbered levels have
/// the same shapes and transforms as `levels`.
pub fn write_label_metadata(root: &Path, levels: &[OmeVolumeLevel], source: &str) -> Result<()> {
    let datasets = levels
        .iter()
        .enumerate()
        .map(|(index, level)| {
            json!({
                "path": index.to_string(),
                "coordinateTransformations": level.coordinate_transformations
            })
        })
        .collect::<Vec<_>>();
    let metadata = json!({
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [{
                    "version": "0.5",
                    "axes": [
                        {"name":"z", "type":"space", "unit":"micrometer"},
                        {"name":"y", "type":"space", "unit":"micrometer"},
                        {"name":"x", "type":"space", "unit":"micrometer"}
                    ],
                    "datasets": datasets
                }]
            },
            "image-label": {"version":"0.5", "source":{"image":source}}
        }
    });
    fs::write(
        root.join("zarr.json"),
        serde_json::to_vec_pretty(&metadata).map_err(Error::backend)?,
    )
    .map_err(Error::backend)
}

/// Return the image group for either a direct multiscale image or the first
/// series in the OME-Zarr 0.5 Bio-Formats layout (`OME/zarr.json`).
fn locate_image_group(container: &Path) -> Result<(PathBuf, Value)> {
    let metadata = read_json(&container.join("zarr.json"))?;
    if multiscale(&metadata).is_some() {
        return Ok((container.to_owned(), metadata));
    }

    let ome_metadata_path = container.join("OME").join("zarr.json");
    let ome_metadata = read_json(&ome_metadata_path).map_err(|_| {
        Error::invalid(format!(
            "{} has no OME multiscales entry and {} is unavailable",
            container.display(),
            ome_metadata_path.display()
        ))
    })?;
    let series = ome_metadata
        .pointer("/attributes/ome/series")
        .and_then(Value::as_array)
        .and_then(|series| series.first())
        .and_then(|series| {
            series
                .as_str()
                .or_else(|| series.get("path").and_then(Value::as_str))
        })
        .ok_or_else(|| Error::invalid("OME metadata has no image series"))?;
    let image_root = container.join(series);
    let image_metadata = read_json(&image_root.join("zarr.json"))?;
    if multiscale(&image_metadata).is_none() {
        return Err(Error::invalid(format!(
            "OME series {series} has no multiscales entry"
        )));
    }
    Ok((image_root, image_metadata))
}

fn multiscale(metadata: &Value) -> Option<&Value> {
    metadata
        .pointer("/attributes/ome/multiscales/0")
        .or_else(|| metadata.pointer("/attributes/multiscales/0"))
}

fn fixed_indices(
    axes: &[String],
    shape: &[usize],
    spatial: [usize; 3],
    channel: usize,
    time: usize,
) -> Result<Vec<(usize, usize)>> {
    let mut fixed = Vec::new();
    for (axis, (name, &length)) in axes.iter().zip(shape).enumerate() {
        if spatial.contains(&axis) {
            continue;
        }
        let index = match name.as_str() {
            "c" => channel,
            "t" => time,
            _ if length == 1 => 0,
            _ => return Err(Error::invalid(format!(
                "OME axis {name} has length {length}; only z, y, x, c, t and singleton axes can be selected"
            ))),
        };
        if index >= length {
            return Err(Error::invalid(format!(
                "selected {name} index {index} is outside axis length {length}"
            )));
        }
        fixed.push((axis, index));
    }
    if !axes.iter().any(|axis| axis == "c") && channel != 0 {
        return Err(Error::invalid(
            "--channel is nonzero but the OME image has no c axis",
        ));
    }
    if !axes.iter().any(|axis| axis == "t") && time != 0 {
        return Err(Error::invalid(
            "--time is nonzero but the OME image has no t axis",
        ));
    }
    Ok(fixed)
}

fn project_transforms(transforms: Option<&Value>, spatial: [usize; 3]) -> Value {
    Value::Array(
        transforms
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|transform| {
                let kind = transform.get("type")?.as_str()?;
                let key = match kind {
                    "scale" => "scale",
                    "translation" => "translation",
                    _ => return None,
                };
                let values = transform.get(key)?.as_array()?;
                let default = Value::from(if kind == "scale" { 1 } else { 0 });
                let projected = spatial
                    .iter()
                    .map(|&axis| values.get(axis).cloned().unwrap_or_else(|| default.clone()))
                    .collect();
                let mut object = Map::new();
                object.insert("type".into(), Value::String(kind.into()));
                object.insert(key.into(), Value::Array(projected));
                Some(Value::Object(object))
            })
            .collect(),
    )
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path).map_err(Error::backend)?).map_err(Error::backend)
}
