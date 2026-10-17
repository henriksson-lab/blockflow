//! Axis-aware access to one scalar `[z, y, x]` volume in an OME-Zarr pyramid.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::{AttachedImage, Error, Result};

/// Ensure `root` is a Zarr v3 group without replacing existing metadata.
pub fn ensure_zarr_group(root: &Path) -> Result<()> {
    fs::create_dir_all(root).map_err(Error::backend)?;
    let metadata_path = root.join("zarr.json");
    if metadata_path.is_file() {
        let metadata = read_json(&metadata_path)?;
        if metadata.get("zarr_format").and_then(Value::as_u64) != Some(3)
            || metadata.get("node_type").and_then(Value::as_str) != Some("group")
        {
            return Err(Error::invalid(format!(
                "{} is not a Zarr v3 group",
                metadata_path.display()
            )));
        }
        return Ok(());
    }
    let metadata = json!({
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {}
    });
    fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&metadata).map_err(Error::backend)?,
    )
    .map_err(Error::backend)
}

#[derive(Clone, Debug)]
pub struct OmeVolumeLevel {
    pub path: String,
    pub shape: [usize; 3],
    pub coordinate_transformations: Value,
}

#[derive(Clone, Debug)]
pub struct OmeVolumePyramid {
    root: PathBuf,
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
            spatial_axes,
            fixed: fixed.unwrap_or_default(),
            levels,
        })
    }

    pub fn levels(&self) -> &[OmeVolumeLevel] {
        &self.levels
    }

    /// Zarr group that owns the multiscale image.
    ///
    /// Derived labels and tables belong below this group. It can differ from
    /// the path passed to [`Self::open`] when that path is an image collection.
    pub fn image_root(&self) -> &Path {
        &self.root
    }

    /// Relative NGFF image path from a sibling `labels/<name>` group.
    pub fn label_source(&self) -> &str {
        "../../"
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
/// series in the transitional OME-Zarr image-collection layout
/// (`OME/zarr.json`).
pub(crate) fn locate_image_group(container: &Path) -> Result<(PathBuf, Value)> {
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

/// A label image, and optionally its object table, to install in an OME-Zarr
/// image with [`publish_label`].
pub struct LabelPublication<'a> {
    /// `labels/<name>` and `tables/<name>`.
    pub name: &'a str,
    /// A finished level-0 `uint64` label array — typically a run's
    /// `work/level{N}`, pinned with `Hints::keep_images`. It is moved, or
    /// copied when it lives on another filesystem.
    pub level0: &'a Path,
    /// Extra `image-label` attributes, e.g. `colors` and `properties`.
    pub image_label: Option<Map<String, Value>>,
    /// A finished object table directory (e.g. from
    /// `object_table::finalize_object_table`), moved to `tables/<name>`. Its
    /// spec's `region` should be `../../labels/<name>`.
    pub table: Option<&'a Path>,
    /// Replace an existing `labels/<name>` or `tables/<name>`; when `false`
    /// either existing is refused (checked again just before the swap, so one
    /// that appeared during the run is refused too).
    pub replace: bool,
    /// Block shapes tried for the label pyramid; empty means `[32, 256, 256]`.
    pub pyramid_blocks: Vec<[usize; 3]>,
}

/// Install a label image (and its table) in `pyramid`'s image: build the
/// nearest-neighbour label pyramid matching the image's levels, write the
/// label metadata, replace any previous `labels/<name>` and `tables/<name>`,
/// and refresh the `labels` registry. Everything is staged next to the
/// destination first, so a failure leaves the previous output in place.
pub fn publish_label(
    pyramid: &OmeVolumePyramid,
    publication: LabelPublication<'_>,
    workers: usize,
) -> Result<()> {
    let name = publication.name;
    if name.is_empty() || name.contains('/') || name.starts_with('.') {
        return Err(Error::invalid(format!(
            "label name {name:?} must be one plain path component"
        )));
    }
    let level0 = read_json(&publication.level0.join("zarr.json"))?;
    if level0.get("data_type").and_then(Value::as_str) != Some("uint64") {
        return Err(Error::invalid(format!(
            "{} is not a uint64 array; label pyramids here are uint64 — widen it first \
             (ops::WidenLabelsOp)",
            publication.level0.display()
        )));
    }
    let labels_dir = pyramid.image_root().join("labels");
    let tables_dir = pyramid.image_root().join("tables");
    let refuse_existing = || -> Result<()> {
        for existing in [labels_dir.join(name), tables_dir.join(name)] {
            if !publication.replace && existing.exists() {
                return Err(Error::invalid(format!(
                    "{} already exists and replacing it was not asked for",
                    existing.display()
                )));
            }
        }
        Ok(())
    };
    refuse_existing()?;
    ensure_zarr_group(&labels_dir)?;
    let staging = labels_dir.join(format!(".{name}-publish"));
    if staging.exists() {
        fs::remove_dir_all(&staging).map_err(Error::backend)?;
    }
    let layer = staging.join("layer");
    fs::create_dir_all(&layer).map_err(Error::backend)?;
    move_path(publication.level0, &layer.join("0"))?;
    let shapes: Vec<[usize; 3]> = pyramid.levels().iter().map(|l| l.shape).collect();
    crate::label_pyramid::build_nearest_label_pyramid_with_blocks(
        &layer,
        &staging.join("pyramid-work"),
        &shapes,
        if publication.pyramid_blocks.is_empty() {
            &[[32, 256, 256]]
        } else {
            &publication.pyramid_blocks
        },
        workers,
    )?;
    write_label_metadata(&layer, pyramid.levels(), pyramid.label_source())?;
    if let Some(extra) = publication.image_label {
        let path = layer.join("zarr.json");
        let mut metadata = read_json(&path)?;
        if let Some(Value::Object(label)) = metadata.pointer_mut("/attributes/image-label") {
            label.extend(extra);
        }
        fs::write(
            &path,
            serde_json::to_vec_pretty(&metadata).map_err(Error::backend)?,
        )
        .map_err(Error::backend)?;
    }
    let table = match publication.table {
        Some(table) => {
            let staged = staging.join("table");
            move_path(table, &staged)?;
            Some(staged)
        }
        None => None,
    };
    refuse_existing()?;
    replace_path(&layer, &labels_dir.join(name))?;
    if let Some(staged) = table {
        ensure_zarr_group(&tables_dir)?;
        replace_path(&staged, &tables_dir.join(name))?;
    }
    crate::label_pyramid::refresh_label_registry(&labels_dir)?;
    fs::remove_dir_all(&staging).map_err(Error::backend)
}

/// Rename, or copy and delete when `from` and `to` are on different
/// filesystems.
pub fn move_path(from: &Path, to: &Path) -> Result<()> {
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent).map_err(Error::backend)?;
    }
    if fs::rename(from, to).is_ok() {
        return Ok(());
    }
    copy_tree(from, to)?;
    fs::remove_dir_all(from).map_err(Error::backend)
}

fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    if from.is_dir() {
        fs::create_dir_all(to).map_err(Error::backend)?;
        for entry in fs::read_dir(from).map_err(Error::backend)? {
            let entry = entry.map_err(Error::backend)?;
            copy_tree(&entry.path(), &to.join(entry.file_name()))?;
        }
        Ok(())
    } else {
        fs::copy(from, to).map(|_| ()).map_err(|error| {
            Error::backend(format!(
                "copy {} to {}: {error}",
                from.display(),
                to.display()
            ))
        })
    }
}

/// Swap `staged` into `destination`, removing what was there.
fn replace_path(staged: &Path, destination: &Path) -> Result<()> {
    let old = destination.with_file_name(format!(
        ".{}-replaced",
        destination
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("old")
    ));
    if old.exists() {
        fs::remove_dir_all(&old).map_err(Error::backend)?;
    }
    if destination.exists() {
        fs::rename(destination, &old).map_err(Error::backend)?;
    }
    fs::rename(staged, destination).map_err(Error::backend)?;
    if old.exists() {
        fs::remove_dir_all(&old).map_err(Error::backend)?;
    }
    Ok(())
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

#[cfg(test)]
mod publish_tests {
    use super::*;
    use zarrs::array::{data_type, ArrayBuilder, ArraySubset, FillValue};
    use zarrs::filesystem::FilesystemStore;

    struct TestRoot(PathBuf);

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn array(
        dir: &Path,
        dtype: zarrs::array::DataType,
        fill: FillValue,
    ) -> zarrs::array::Array<FilesystemStore> {
        let store = std::sync::Arc::new(FilesystemStore::new(dir).unwrap());
        let a = ArrayBuilder::new(vec![4, 8, 8], vec![2, 8, 8], dtype, fill)
            .build(store, "/")
            .unwrap();
        a.store_metadata().unwrap();
        a
    }

    #[test]
    fn publish_installs_and_replaces_a_label_and_its_table() {
        let root = TestRoot(
            std::env::temp_dir().join(format!("blockflow-publish-{}", std::process::id())),
        );
        let image = root.0.join("image.ome.zarr");
        fs::create_dir_all(&image).unwrap();
        fs::write(
            image.join("zarr.json"),
            serde_json::to_vec_pretty(&json!({
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
        fs::create_dir_all(image.join("0")).unwrap();
        array(&image.join("0"), data_type::uint8(), FillValue::from(0u8));
        let pyramid = OmeVolumePyramid::open(&image, 0, 0).unwrap();

        for value in [7u64, 9] {
            let level0 = root.0.join("work/level3");
            fs::create_dir_all(&level0).unwrap();
            let a = array(&level0, data_type::uint64(), FillValue::from(0u64));
            a.store_array_subset(
                &ArraySubset::new_with_shape(vec![4, 8, 8]),
                &vec![value; 256],
            )
            .unwrap();
            let table = root.0.join("work/table");
            ensure_zarr_group(&table).unwrap();
            let mut style = Map::new();
            style.insert(
                "colors".into(),
                json!([{"label-value": value, "rgba": [1, 2, 3, 4]}]),
            );
            publish_label(
                &pyramid,
                LabelPublication {
                    name: "cells",
                    level0: &level0,
                    image_label: Some(style),
                    table: Some(&table),
                    replace: true,
                    pyramid_blocks: Vec::new(),
                },
                2,
            )
            .unwrap();
            assert!(!level0.exists() && !table.exists(), "inputs are moved");
            let meta = read_json(&image.join("labels/cells/zarr.json")).unwrap();
            assert_eq!(
                meta.pointer("/attributes/image-label/colors/0/label-value"),
                Some(&json!(value))
            );
            let registry = read_json(&image.join("labels/zarr.json")).unwrap();
            assert_eq!(
                registry.pointer("/attributes/ome/labels"),
                Some(&json!(["cells"]))
            );
            assert!(image.join("tables/cells/zarr.json").is_file());
            let store =
                std::sync::Arc::new(FilesystemStore::new(image.join("labels/cells")).unwrap());
            let written = zarrs::array::Array::open(store, "/0").unwrap();
            let got: Vec<u64> = written
                .retrieve_array_subset(&ArraySubset::new_with_shape(vec![4, 8, 8]))
                .unwrap();
            assert!(
                got.iter().all(|&v| v == value),
                "the second run replaces the first"
            );
        }
        // a u32 label array is refused by name
        let narrow = root.0.join("work/narrow");
        fs::create_dir_all(&narrow).unwrap();
        array(&narrow, data_type::uint32(), FillValue::from(0u32));
        let refused = publish_label(
            &pyramid,
            LabelPublication {
                name: "cells",
                level0: &narrow,
                image_label: None,
                table: None,
                replace: true,
                pyramid_blocks: Vec::new(),
            },
            1,
        );
        assert!(refused.unwrap_err().to_string().contains("WidenLabelsOp"));
        // an existing label is refused unless replacing was asked for
        let level0 = root.0.join("work/level4");
        fs::create_dir_all(&level0).unwrap();
        array(&level0, data_type::uint64(), FillValue::from(0u64));
        let kept = publish_label(
            &pyramid,
            LabelPublication {
                name: "cells",
                level0: &level0,
                image_label: None,
                table: None,
                replace: false,
                pyramid_blocks: Vec::new(),
            },
            1,
        );
        assert!(kept.unwrap_err().to_string().contains("already exists"));
    }
}
