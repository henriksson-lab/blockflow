//! Typed, atomic image-level metadata stored beside OME-Zarr image attributes.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{Error, Result};

pub const ATTRIBUTE_KEY: &str = "rozt:imageMetadata";
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum ImageMetadataProperty {
    Categorical {
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        unit: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    String {
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        unit: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    Boolean {
        value: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        unit: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    Integer {
        value: i64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        unit: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    Float {
        value: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        unit: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImageMetadataProvenance {
    pub writer: String,
    pub updated_at_ms: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImageMetadataDocument {
    pub schema_version: u32,
    #[serde(default)]
    pub properties: BTreeMap<String, ImageMetadataProperty>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<ImageMetadataProvenance>,
}

impl Default for ImageMetadataDocument {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            properties: BTreeMap::new(),
            provenance: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageMetadataSnapshot {
    pub document: ImageMetadataDocument,
    pub revision: String,
}

pub fn read_image_metadata(image_group: &Path) -> Result<ImageMetadataSnapshot> {
    let (path, mut root, v3) = read_root(image_group)?;
    let attributes = attributes_mut(&path, &mut root, v3)?;
    let document = match attributes.get(ATTRIBUTE_KEY) {
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|error| Error::invalid(format!("parse {ATTRIBUTE_KEY}: {error}")))?,
        None => ImageMetadataDocument::default(),
    };
    validate(&document)?;
    snapshot(document)
}

pub fn replace_image_metadata(
    image_group: &Path,
    expected_revision: Option<&str>,
    replacement: &ImageMetadataDocument,
) -> Result<ImageMetadataSnapshot> {
    validate(replacement)?;
    let lock_path = image_group.join(".rozt-image-metadata.lock");
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(Error::backend)?;
    lock.lock_exclusive().map_err(Error::backend)?;
    let result = (|| {
        let current = read_image_metadata(image_group)?;
        if expected_revision.is_some_and(|expected| expected != current.revision) {
            return Err(Error::invalid(format!(
                "image metadata revision conflict: current {}",
                current.revision
            )));
        }
        let next = snapshot(replacement.clone())?;
        if next.revision == current.revision {
            return Ok(current);
        }
        let (path, mut root, v3) = read_root(image_group)?;
        let attributes = attributes_mut(&path, &mut root, v3)?;
        attributes.insert(
            ATTRIBUTE_KEY.into(),
            serde_json::to_value(replacement).map_err(Error::backend)?,
        );
        write_json_atomic(&path, &root)?;
        Ok(next)
    })();
    let _ = fs2::FileExt::unlock(&lock);
    result
}

fn snapshot(document: ImageMetadataDocument) -> Result<ImageMetadataSnapshot> {
    let bytes = serde_json::to_vec(&document).map_err(Error::backend)?;
    let mut digest = Sha256::new();
    digest.update(b"rozt.image-metadata/v1\0");
    digest.update(bytes);
    Ok(ImageMetadataSnapshot {
        document,
        revision: format!("image-metadata-sha256:{:x}", digest.finalize()),
    })
}

fn validate(document: &ImageMetadataDocument) -> Result<()> {
    if document.schema_version != SCHEMA_VERSION {
        return Err(Error::invalid(format!(
            "unsupported image metadata schema {}",
            document.schema_version
        )));
    }
    let mut folded = HashSet::new();
    for (key, property) in &document.properties {
        let valid = !key.is_empty()
            && key.as_bytes()[0].is_ascii_lowercase()
            && key.bytes().all(|b| {
                b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'_' | b'.' | b'-')
            });
        if !valid || !folded.insert(key.to_ascii_lowercase()) {
            return Err(Error::invalid(format!(
                "invalid or duplicate image metadata key {key:?}"
            )));
        }
        if let ImageMetadataProperty::Float { value, .. } = property {
            if !value.is_finite() {
                return Err(Error::invalid(format!(
                    "image metadata {key:?} is not finite"
                )));
            }
        }
    }
    Ok(())
}

fn read_root(image_group: &Path) -> Result<(PathBuf, Value, bool)> {
    let v3 = image_group.join("zarr.json");
    if v3.is_file() {
        return Ok((v3.clone(), read_json(&v3)?, true));
    }
    let v2 = image_group.join(".zattrs");
    if v2.is_file() {
        return Ok((v2.clone(), read_json(&v2)?, false));
    }
    Err(Error::invalid(format!(
        "{} is not a Zarr v2/v3 group",
        image_group.display()
    )))
}

fn attributes_mut<'a>(
    path: &Path,
    root: &'a mut Value,
    v3: bool,
) -> Result<&'a mut serde_json::Map<String, Value>> {
    let target = if v3 {
        root.get_mut("attributes")
            .ok_or_else(|| Error::invalid(format!("{} has no attributes object", path.display())))?
    } else {
        root
    };
    target
        .as_object_mut()
        .ok_or_else(|| Error::invalid(format!("{} attributes are not an object", path.display())))
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path).map_err(Error::backend)?).map_err(Error::backend)
}

fn write_json_atomic(path: &Path, value: &Value) -> Result<()> {
    let temporary = path.with_extension("rozt-image-metadata.tmp");
    let bytes = serde_json::to_vec_pretty(value).map_err(Error::backend)?;
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temporary)
        .map_err(Error::backend)?;
    use std::io::Write;
    file.write_all(&bytes).map_err(Error::backend)?;
    file.sync_all().map_err(Error::backend)?;
    fs::rename(&temporary, path).map_err(Error::backend)
}
