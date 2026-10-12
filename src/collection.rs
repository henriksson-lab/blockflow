// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! **Collections of image series: plates, wells, fields.**
//!
//! A multiwell plate is a hierarchy above the pixel arrays — plate,
//! acquisition, well, field — and a *series* is one field imaged over time.
//! The hierarchy is a collection key, never a pixel axis: tracking links
//! within one series and never across two.
//!
//! Two sources describe the hierarchy:
//!
//! * an **OME-NGFF HCS plate** — `plate` metadata enumerates wells, each
//!   well's `well.images` enumerates its field images and their acquisitions;
//! * an explicit **collection manifest** for separately stored images, which
//!   records every relationship a directory scan would otherwise have to
//!   invent. A scan is never allowed to.
//!
//! Both yield [`SeriesRef`]s in a stable order.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use serde_json::Value;

use crate::time_series::read_json;
use crate::track::Frame;
use crate::{Error, Result};

/// The stable identity of one series within a collection.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeriesKey {
    pub plate_id: String,
    pub acquisition_id: Option<u64>,
    pub well_row: String,
    pub well_column: String,
    pub field_id: String,
}

impl SeriesKey {
    /// A path-safe name, e.g. `A-1-f0` or `A-1-f0-acq2`.
    pub fn slug(&self) -> String {
        let mut slug = format!("{}-{}-f{}", self.well_row, self.well_column, self.field_id);
        if let Some(acquisition) = self.acquisition_id {
            slug.push_str(&format!("-acq{acquisition}"));
        }
        slug.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }
}

/// Which channel to segment and which to measure, when a series says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelMapping {
    pub segment: usize,
    pub measure: usize,
}

/// One series: its key, where its image is, and what the collection says
/// about it beyond the image's own metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct SeriesRef {
    pub key: SeriesKey,
    /// The multiscale image group.
    pub image: PathBuf,
    /// How the collection referred to it (relative path or URI as written).
    pub source_uri: String,
    pub well_indices: Option<(u64, u64)>,
    pub channels: Option<ChannelMapping>,
    /// The acquired timeline, one entry per stored time index, when the
    /// collection states it. `None` means the stored axis is the timeline.
    pub timeline: Option<Vec<Frame>>,
}

/// Enumerate an OME-NGFF HCS plate. Accepts both the 0.5 (`attributes.ome`)
/// and the 0.4 (`attributes`) metadata placement.
pub fn open_plate(root: impl AsRef<Path>) -> Result<Vec<SeriesRef>> {
    let root = root.as_ref();
    let metadata = read_json(&root.join("zarr.json")).or_else(|_| {
        read_json(&root.join(".zattrs")).map(|a| serde_json::json!({"attributes": a}))
    })?;
    let attributes = metadata.get("attributes").cloned().unwrap_or(Value::Null);
    let plate = attributes
        .pointer("/ome/plate")
        .or_else(|| attributes.get("plate"))
        .ok_or_else(|| Error::invalid(format!("{} has no HCS plate metadata", root.display())))?;
    let plate_id = plate
        .get("name")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            root.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "plate".into())
        });
    let names = |key: &str| -> Vec<String> {
        plate
            .get(key)
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(|item| {
                        item.get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned()
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let rows = names("rows");
    let columns = names("columns");
    let acquisitions: BTreeSet<u64> = plate
        .get("acquisitions")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|a| a.get("id").and_then(Value::as_u64))
                .collect()
        })
        .unwrap_or_default();
    let wells = plate
        .get("wells")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::invalid("HCS plate metadata has no wells"))?;
    let mut series = Vec::new();
    for well in wells {
        let path = well
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::invalid("HCS well has no path"))?;
        check_relative(path)?;
        let row_index = well.get("rowIndex").and_then(Value::as_u64);
        let column_index = well.get("columnIndex").and_then(Value::as_u64);
        let (well_row, well_column) = match (row_index, column_index) {
            (Some(r), Some(c)) => (
                rows.get(r as usize).cloned().ok_or_else(|| {
                    Error::invalid(format!(
                        "well {path} names row index {r}, which is not declared"
                    ))
                })?,
                columns.get(c as usize).cloned().ok_or_else(|| {
                    Error::invalid(format!(
                        "well {path} names column index {c}, which is not declared"
                    ))
                })?,
            ),
            _ => {
                let mut parts = path.split('/');
                match (parts.next(), parts.next()) {
                    (Some(r), Some(c)) => (r.to_owned(), c.to_owned()),
                    _ => {
                        return Err(Error::invalid(format!(
                            "well path {path} is not row/column"
                        )))
                    }
                }
            }
        };
        let well_root = root.join(path);
        let well_metadata = read_json(&well_root.join("zarr.json")).or_else(|_| {
            read_json(&well_root.join(".zattrs")).map(|a| serde_json::json!({"attributes": a}))
        })?;
        let well_attributes = well_metadata
            .get("attributes")
            .cloned()
            .unwrap_or(Value::Null);
        let images = well_attributes
            .pointer("/ome/well/images")
            .or_else(|| well_attributes.pointer("/well/images"))
            .and_then(Value::as_array)
            .ok_or_else(|| Error::invalid(format!("well {path} has no well.images metadata")))?;
        for image in images {
            let field = image
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::invalid(format!("an image of well {path} has no path")))?;
            check_relative(field)?;
            let acquisition_id = image.get("acquisition").and_then(Value::as_u64);
            if let Some(id) = acquisition_id {
                if !acquisitions.is_empty() && !acquisitions.contains(&id) {
                    return Err(Error::invalid(format!(
                        "well {path} image {field} names acquisition {id}, which the plate does not declare"
                    )));
                }
            }
            let image_root = well_root.join(field);
            if !image_root.join("zarr.json").is_file() && !image_root.join(".zattrs").is_file() {
                return Err(Error::invalid(format!(
                    "field image {} does not exist",
                    image_root.display()
                )));
            }
            series.push(SeriesRef {
                key: SeriesKey {
                    plate_id: plate_id.clone(),
                    acquisition_id,
                    well_row: well_row.clone(),
                    well_column: well_column.clone(),
                    field_id: field.to_owned(),
                },
                image: image_root,
                source_uri: format!("{path}/{field}"),
                well_indices: row_index.zip(column_index),
                channels: None,
                timeline: None,
            });
        }
    }
    finish(series)
}

/// Read a collection manifest (JSON):
///
/// ```json
/// {
///   "version": 1,
///   "plate_id": "P1",
///   "series": [
///     {"path": "A1-f0.zarr", "well_row": "A", "well_column": "1", "field": "0",
///      "acquisition": 0, "channels": {"segment": 0, "measure": 1},
///      "timeline": [{"t_index": 0, "elapsed_seconds": 0.0}, ...]}
///   ]
/// }
/// ```
///
/// Paths resolve against the manifest's directory; relative paths may not
/// contain `..`. `acquisition`, `channels` and `timeline` are optional.
pub fn open_manifest(path: impl AsRef<Path>) -> Result<Vec<SeriesRef>> {
    let path = path.as_ref();
    let manifest = read_json(path)?;
    if manifest.get("version").and_then(Value::as_u64) != Some(1) {
        return Err(Error::invalid(
            "collection manifest must have \"version\": 1",
        ));
    }
    let plate_id = manifest
        .get("plate_id")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::invalid("collection manifest has no plate_id"))?
        .to_owned();
    let base = path.parent().unwrap_or(Path::new("."));
    let entries = manifest
        .get("series")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::invalid("collection manifest has no series list"))?;
    let mut series = Vec::new();
    for entry in entries {
        let text = |key: &str| -> Result<String> {
            match entry.get(key) {
                Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
                Some(Value::Number(n)) => Ok(n.to_string()),
                _ => Err(Error::invalid(format!("manifest series entry needs {key}"))),
            }
        };
        let uri = text("path")?;
        let image = if Path::new(&uri).is_absolute() {
            PathBuf::from(&uri)
        } else {
            check_relative(&uri)?;
            base.join(&uri)
        };
        if !image.join("zarr.json").is_file() {
            return Err(Error::invalid(format!(
                "manifest image {} does not exist",
                image.display()
            )));
        }
        let channels =
            match entry.get("channels") {
                None | Some(Value::Null) => None,
                Some(channels) => {
                    Some(ChannelMapping {
                        segment: channels.get("segment").and_then(Value::as_u64).ok_or_else(
                            || Error::invalid("manifest channels need a segment index"),
                        )? as usize,
                        measure: channels.get("measure").and_then(Value::as_u64).ok_or_else(
                            || Error::invalid("manifest channels need a measure index"),
                        )? as usize,
                    })
                }
            };
        let timeline = match entry.get("timeline") {
            None | Some(Value::Null) => None,
            Some(Value::Array(frames)) => Some(
                frames
                    .iter()
                    .map(|frame| {
                        Ok(Frame {
                            t_index: frame.get("t_index").and_then(Value::as_u64).ok_or_else(
                                || Error::invalid("manifest timeline entries need t_index"),
                            )?,
                            elapsed_seconds: frame.get("elapsed_seconds").and_then(Value::as_f64),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            Some(_) => return Err(Error::invalid("manifest timeline must be a list")),
        };
        series.push(SeriesRef {
            key: SeriesKey {
                plate_id: plate_id.clone(),
                acquisition_id: entry.get("acquisition").and_then(Value::as_u64),
                well_row: text("well_row")?,
                well_column: text("well_column")?,
                field_id: text("field")?,
            },
            image,
            source_uri: uri,
            well_indices: None,
            channels,
            timeline,
        });
    }
    finish(series)
}

fn finish(mut series: Vec<SeriesRef>) -> Result<Vec<SeriesRef>> {
    series.sort_by(|a, b| a.key.cmp(&b.key));
    for pair in series.windows(2) {
        if pair[0].key == pair[1].key {
            return Err(Error::invalid(format!(
                "series {:?} occurs more than once",
                pair[0].key
            )));
        }
    }
    let mut slugs = BTreeSet::new();
    for s in &series {
        if !slugs.insert(s.key.slug()) {
            return Err(Error::invalid(format!(
                "series {:?} collides with another after path sanitising",
                s.key
            )));
        }
    }
    Ok(series)
}

fn check_relative(path: &str) -> Result<()> {
    let p = Path::new(path);
    if p.is_absolute()
        || p.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::Prefix(_) | Component::RootDir
            )
        })
    {
        return Err(Error::invalid(format!(
            "collection path {path:?} must be relative and may not leave its parent"
        )));
    }
    Ok(())
}
