// SPDX-License-Identifier: MIT
// Original work for this crate.

//! A prepared atlas as one OME-Zarr image (feature `zarr`).
//!
//! ```text
//! <atlas>.ome.zarr/
//!   zarr.json              multiscale reference image; attributes.atlas
//!   0, 1, ...              reference levels (block means)
//!   labels/annotation      uint64 region ids; image-label properties + colors
//!   labels/hemispheres     uint64 hemisphere labels; image-label properties
//!   distance_to_surface/   multiscale float32 image, in voxels
//! ```
//!
//! All images share level 0's grid and physical frame (voxel size in
//! micrometres, voxel 0 at the origin). The distance to the surface is a
//! continuous field, not a label, so it is a nested multiscale image rather
//! than a `labels/` entry; open it with
//! `OmeVolumePyramid::open(root.join("distance_to_surface"), 0, 0)`. Its
//! values are in **voxels**, the reference's convention; multiply by the voxel
//! size (`attributes.atlas.voxel_size_um`, isotropic 25 µm for the Allen
//! atlas) for micrometres.
//!
//! `attributes.atlas.window` is [`PreparedWindow::to_json`]: the orientation
//! and crop that turned the source atlas into this one.

use std::fs;
use std::path::{Path, PathBuf};

use blockflow::error::{Error, Result};
use blockflow::ome_zarr::{publish_label, LabelPublication, OmeVolumePyramid};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{PhysicalFrame, PyramidSpec, ZarrEnvironment};
use serde_json::{json, Map, Value};

use crate::ontology::{Ontology, OntologyLevel};
use crate::prepare::{AtlasStack, PreparedWindow};

/// The name of the nested distance-to-surface image.
pub const DISTANCE_TO_SURFACE: &str = "distance_to_surface";
/// The label image of region ids.
pub const ANNOTATION_LABEL: &str = "annotation";
/// The label image of hemispheres.
pub const HEMISPHERES_LABEL: &str = "hemispheres";

/// Source OME-Zarr images, one per atlas volume (channel 0, time 0, level 0
/// of each is read).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AtlasSources {
    pub annotation: PathBuf,
    pub reference: Option<PathBuf>,
    pub hemispheres: Option<PathBuf>,
    pub distance_to_surface: Option<PathBuf>,
}

/// Level 0 of the OME-Zarr image at `root`, resident.
pub fn read_volume(root: &Path) -> Result<Voxels> {
    OmeVolumePyramid::open(root, 0, 0)
        .and_then(|pyramid| pyramid.attached_level(0)?.read())
        .map_err(|error| Error::invalid(format!("reading {}: {error}", root.display())))
}

/// Every source volume, resident.
pub fn read_atlas_sources(sources: &AtlasSources) -> Result<AtlasStack> {
    let optional = |path: &Option<PathBuf>| path.as_deref().map(read_volume).transpose();
    Ok(AtlasStack {
        annotation: read_volume(&sources.annotation)?,
        reference: optional(&sources.reference)?,
        hemispheres: optional(&sources.hemispheres)?,
        distance_to_surface: optional(&sources.distance_to_surface)?,
    })
}

/// How to write a prepared atlas.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteAtlasParams {
    /// Level-0 voxel size of the **prepared** atlas, `[z, y, x]` µm (orient a
    /// source voxel size with `Orientation::orient_resolution`).
    pub voxel_size_um: [f64; 3],
    /// A display name, kept in `attributes.atlas.name`.
    pub name: String,
    /// Replace an existing atlas at the destination; otherwise it is refused.
    pub replace: bool,
    /// Threads for the label pyramids.
    pub workers: usize,
}

/// What [`write_prepared_atlas`] wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct WrittenAtlas {
    pub root: PathBuf,
    pub shape: [usize; 3],
    pub levels: usize,
    /// Distinct non-zero region ids in the annotation.
    pub regions_present: usize,
}

/// Write `stack` (which must have a reference image) as one OME-Zarr atlas at
/// `root`, staged beside it and moved into place at the end.
///
/// Every id in the annotation has to be in `ontology` (0 excepted): an id the
/// tree lacks is refused, as [`crate::Annotator`] would refuse it later.
pub fn write_prepared_atlas(
    root: &Path,
    stack: &AtlasStack,
    window: &PreparedWindow,
    ontology: &Ontology,
    params: &WriteAtlasParams,
) -> Result<WrittenAtlas> {
    let reference = stack.reference.as_ref().ok_or_else(|| {
        Error::invalid("a prepared atlas is written around its reference image; none was given")
    })?;
    let shape = stack.shape();
    if window.shape != shape {
        return Err(Error::invalid(format!(
            "the window says {:?} but the volumes are {shape:?}",
            window.shape
        )));
    }
    if root.exists() && !params.replace {
        return Err(Error::invalid(format!(
            "{} exists and replacing it was not asked for",
            root.display()
        )));
    }
    let annotation = to_u64_labels(&stack.annotation, "annotation")?;
    let regions_present = check_ontology_covers(&annotation, ontology)?;

    let staging = PathBuf::from(format!("{}.partial", root.display()));
    if staging.exists() {
        fs::remove_dir_all(&staging).map_err(Error::backend)?;
    }
    let levels = pyramid_levels(shape);
    let frame = PhysicalFrame::new(params.voxel_size_um, Some("micrometer"));
    let spec = PyramidSpec::powers_of_two(levels)?.with_physical(frame.clone());
    ZarrEnvironment::build_multiscale(&staging, reference, &spec)?;
    merge_attributes(
        &staging.join("zarr.json"),
        "atlas",
        json!({
            "name": params.name,
            "voxel_size_um": params.voxel_size_um,
            "window": window.to_json(),
            "reference": ".",
            "labels": [ANNOTATION_LABEL, HEMISPHERES_LABEL],
            "distance_to_surface": stack.distance_to_surface.as_ref().map(|_| DISTANCE_TO_SURFACE),
            "distance_to_surface_unit": "voxel",
            "label_level": "tree depth from the root record (0); st_level is kept beside it",
        }),
    )?;
    let pyramid = OmeVolumePyramid::open(&staging, 0, 0)?;

    let mut image_label = Map::new();
    image_label.insert(
        "properties".into(),
        Value::Array(region_properties(ontology)?),
    );
    image_label.insert("colors".into(), Value::Array(region_colors(ontology)));
    publish_resident_label(&pyramid, ANNOTATION_LABEL, &annotation, image_label, params)?;
    drop(annotation);

    if let Some(hemispheres) = &stack.hemispheres {
        let hemispheres = to_u64_labels(hemispheres, "hemispheres")?;
        let mut values = hemispheres
            .view::<u64>()?
            .iter()
            .copied()
            .collect::<Vec<_>>();
        values.sort_unstable();
        values.dedup();
        let properties = values
            .iter()
            .map(|&value| json!({"label-value": value, "name": format!("hemisphere {value}")}))
            .collect();
        let mut image_label = Map::new();
        image_label.insert("properties".into(), Value::Array(properties));
        publish_resident_label(
            &pyramid,
            HEMISPHERES_LABEL,
            &hemispheres,
            image_label,
            params,
        )?;
    }

    if let Some(distance) = &stack.distance_to_surface {
        let distance = match distance {
            Voxels::F32(_) => distance.clone(),
            other => Voxels::F32(other.widened().mapv(|value| value as f32)),
        };
        let nested = staging.join(DISTANCE_TO_SURFACE);
        ZarrEnvironment::build_multiscale(&nested, &distance, &spec)?;
        merge_attributes(
            &nested.join("zarr.json"),
            "distance_to_surface",
            json!({
                "unit": "voxel",
                "voxel_size_um": params.voxel_size_um,
                "note": "distance from each voxel to the brain surface in voxels; multiply by \
                         voxel_size_um for micrometres",
            }),
        )?;
    }

    if root.exists() {
        fs::remove_dir_all(root).map_err(Error::backend)?;
    }
    fs::rename(&staging, root).map_err(Error::backend)?;
    Ok(WrittenAtlas {
        root: root.to_owned(),
        shape,
        levels,
        regions_present,
    })
}

/// A prepared atlas read back, all volumes resident at level 0.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedAtlas {
    pub root: PathBuf,
    pub stack: AtlasStack,
    pub window: PreparedWindow,
    pub voxel_size_um: [f64; 3],
    pub ontology: Ontology,
}

/// Read an atlas written by [`write_prepared_atlas`]. The region tree comes
/// from the annotation's `image-label` properties, with `level` as each
/// node's level.
pub fn open_prepared_atlas(root: &Path, level: OntologyLevel) -> Result<PreparedAtlas> {
    let metadata = read_json(&root.join("zarr.json"))?;
    let atlas = metadata.pointer("/attributes/atlas").ok_or_else(|| {
        Error::invalid(format!(
            "{} has no attributes.atlas; it is not a prepared atlas",
            root.display()
        ))
    })?;
    let window = PreparedWindow::from_json(
        atlas
            .get("window")
            .ok_or_else(|| Error::invalid("attributes.atlas has no window"))?,
    )?;
    let voxel_size_um = atlas
        .get("voxel_size_um")
        .and_then(Value::as_array)
        .and_then(|v| {
            Some([
                v.first()?.as_f64()?,
                v.get(1)?.as_f64()?,
                v.get(2)?.as_f64()?,
            ])
        })
        .ok_or_else(|| Error::invalid("attributes.atlas has no voxel_size_um"))?;
    let labels = root.join("labels");
    let annotation_root = labels.join(ANNOTATION_LABEL);
    let properties = read_json(&annotation_root.join("zarr.json"))?
        .pointer("/attributes/image-label/properties")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| Error::invalid("labels/annotation has no image-label properties"))?;
    let ontology = Ontology::from_image_label_properties(&properties, level)?;
    let optional = |path: PathBuf| {
        if path.join("zarr.json").is_file() {
            read_volume(&path).map(Some)
        } else {
            Ok(None)
        }
    };
    let stack = AtlasStack {
        annotation: read_volume(&annotation_root)?,
        reference: Some(read_volume(root)?),
        hemispheres: optional(labels.join(HEMISPHERES_LABEL))?,
        distance_to_surface: optional(root.join(DISTANCE_TO_SURFACE))?,
    };
    Ok(PreparedAtlas {
        root: root.to_owned(),
        stack,
        window,
        voxel_size_um,
        ontology,
    })
}

/// `image-label.properties` for every region of the tree: `label-value`,
/// `name`, `acronym`, `parent`, `level` (tree depth), `st_level` and
/// `color_hex_triplet` where known.
pub fn region_properties(ontology: &Ontology) -> Result<Vec<Value>> {
    ontology
        .ids()
        .map(|id| {
            let mut entry = Map::new();
            entry.insert("label-value".into(), json!(id));
            entry.insert("name".into(), json!(ontology.name(id)));
            entry.insert("acronym".into(), json!(ontology.acronym(id)));
            entry.insert("parent".into(), json!(ontology.parent(id)));
            entry.insert("level".into(), json!(ontology.depth(id)?));
            entry.insert("st_level".into(), json!(ontology.structure_level(id)));
            if let Some([r, g, b]) = ontology.color(id) {
                entry.insert(
                    "color_hex_triplet".into(),
                    json!(format!("{r:02X}{g:02X}{b:02X}")),
                );
            }
            Ok(Value::Object(entry))
        })
        .collect()
}

/// `image-label.colors`: opaque RGBA per region with a colour, except 0.
pub fn region_colors(ontology: &Ontology) -> Vec<Value> {
    ontology
        .ids()
        .filter(|&id| id != 0)
        .filter_map(|id| {
            ontology
                .color(id)
                .map(|[r, g, b]| json!({"label-value": id, "rgba": [r, g, b, 255]}))
        })
        .collect()
}

/// Levels of 2x reduction until an axis would fall under 16 voxels, at most 6.
fn pyramid_levels(shape: [usize; 3]) -> usize {
    let mut levels = 1;
    while levels < 6 && shape.iter().all(|&n| n >> levels >= 16) {
        levels += 1;
    }
    levels
}

fn to_u64_labels(volume: &Voxels, what: &str) -> Result<Voxels> {
    if let Voxels::U64(_) = volume {
        return Ok(volume.clone());
    }
    let widened = volume.widened_i64(what)?;
    if widened.iter().any(|&value| value < 0) {
        return Err(Error::invalid(format!(
            "the {what} volume holds negative labels; label images here are unsigned"
        )));
    }
    Ok(Voxels::U64(widened.mapv(|value| value as u64)))
}

/// The number of distinct non-zero ids, after checking the tree holds each.
fn check_ontology_covers(annotation: &Voxels, ontology: &Ontology) -> Result<usize> {
    let mut ids = annotation
        .view::<u64>()?
        .iter()
        .copied()
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    ids.retain(|&id| id != 0);
    let missing = ids
        .iter()
        .filter(|&&id| i64::try_from(id).map_or(true, |id| !ontology.contains(id)))
        .take(10)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(Error::invalid(format!(
            "annotation ids {missing:?} are not in the region tree; the annotation and the \
             structure file have to describe the same atlas"
        )));
    }
    Ok(ids.len())
}

fn publish_resident_label(
    pyramid: &OmeVolumePyramid,
    name: &str,
    labels: &Voxels,
    image_label: Map<String, Value>,
    params: &WriteAtlasParams,
) -> Result<()> {
    let level0 = pyramid.image_root().join(format!(".{name}-level0"));
    let spec = PyramidSpec::new(Vec::new())?.with_chunk([64, 64, 64]);
    ZarrEnvironment::build_multiscale(&level0, labels, &spec)?;
    publish_label(
        pyramid,
        LabelPublication {
            name,
            level0: &level0.join("0"),
            image_label: Some(image_label),
            table: None,
            replace: true,
            pyramid_blocks: Vec::new(),
        },
        params.workers.max(1),
    )?;
    fs::remove_dir_all(&level0).map_err(Error::backend)
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path)
        .map_err(|error| Error::invalid(format!("reading {}: {error}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|error| Error::invalid(format!("parsing {}: {error}", path.display())))
}

/// Set `attributes.<key>` in a `zarr.json`.
fn merge_attributes(path: &Path, key: &str, value: Value) -> Result<()> {
    let mut metadata = read_json(path)?;
    metadata
        .get_mut("attributes")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| Error::invalid(format!("{} has no attributes object", path.display())))?
        .insert(key.to_owned(), value);
    fs::write(
        path,
        serde_json::to_vec_pretty(&metadata).map_err(Error::backend)?,
    )
    .map_err(Error::backend)
}
