// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Atlas space through a registration: points of an image mapped into the
//! voxels of a prepared atlas (`atlas-regions`' `atlas-prepare` output), and
//! the atlas lookups and densities read there.
//!
//! **Frames.** Three level-0 voxel lattices are involved, each with voxel
//! centres at integers and `physical = translation + index * scale` (NGFF):
//!
//! 1. the **image** the points were found in (cells, a vessel graph, a tract
//!    map), at any resolution;
//! 2. the registration's **sample**, its first fixed image
//!    ([`PointTransform::sample_frame`]), which is what
//!    [`PointTransform::apply`] takes. It is usually not the image of 1: a
//!    25 µm resample of the signal, or the autofluorescence channel;
//! 3. the **atlas**, the registration's last moving image
//!    ([`PointTransform::atlas_frame`]), which has to be the prepared atlas the
//!    lookups read.
//!
//! 1 → 2 is the change of lattice between the two images' physical frames
//! ([`ImageToSample`]): `j = (t_i + i * s_i + o - t_s) / s_s` per axis, with
//! `o` an optional offset in µm for an image whose own metadata does not place
//! it in the sample's physical frame (a tile of a mosaic whose tiles all start
//! at 0). The scale comes from the two images' metadata, never from a
//! hand-entered factor. 2 → 3 is the registration.
//!
//! **Atlas consistency.** The atlas must be the registration's moving image:
//! its level-0 shape must be the moving image's and its voxel size the moving
//! image's level-0 scale (both always checked), and its reference must equal
//! the moving image voxel for voxel (checked when the moving image can be
//! read, unless [`OpenParams::check_reference`] is off).
//!
//! **Lookups.** A point picks its atlas voxel by [`Rounding`]: to the nearest
//! voxel centre with ties to even (the default), or down, as the reference's
//! `astype(int)`. A point outside the atlas has region 0 and hemisphere `-1`,
//! and its distance to the surface is read at the nearest edge voxel, as the
//! reference reads it. Regions collapse to an ancestor by tree depth, the
//! reference's level.
//!
//! **Distance to the surface is in atlas voxels** (`distance_to_surface_voxels`):
//! the unit the prepared atlas stores it in and the one the reference's
//! thresholds are in (the vessel classifier's minimum depth, the cell crust).
//! [`RegisteredAtlas::voxel_length_um`] converts it to micrometres on an atlas
//! with cubic voxels.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use atlas_regions::atlas_zarr::{
    open_prepared_atlas, read_volume, PreparedAtlas, ANNOTATION_LABEL, DISTANCE_TO_SURFACE,
    HEMISPHERES_LABEL,
};
use atlas_regions::{
    distance_to_surface, label_at, voxel_index, AnnotateParams, Annotator, AtlasVolumes,
    OntologyLevel, OutOfBounds, Rounding,
};
use blockflow::budget::Lease;
use blockflow::error::{Error, Result};
use blockflow::geometry::BlockGrid;
use blockflow::ops::distance::DistanceParams;
use blockflow::ops::voxelize::{ball, voxelize_into};
use blockflow::points::Point;
use blockflow::region::Region;
use blockflow::voxels::Voxels;
use ndarray::Array3;
use serde_json::{json, Value};

use crate::memory;
use crate::points::RegisteredImage;
use crate::rescale::Frame;
use crate::PointTransform;

/// An image's level-0 physical frame and the unit it is in.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageFrame {
    pub frame: Frame,
    /// The spatial unit, `None` when the image declares none.
    pub unit: Option<String>,
}

impl ImageFrame {
    pub fn new(scale: [f64; 3], translation: [f64; 3], unit: Option<&str>) -> Self {
        Self {
            frame: Frame { scale, translation },
            unit: unit.map(str::to_string),
        }
    }

    /// Level 0 of the first multiscale image at `root`, from its OME-Zarr
    /// metadata (`scale` and `translation` of dataset 0, the `x` axis' unit).
    /// An absent translation is zero; an absent scale is one.
    pub fn from_ome_zarr(root: &Path) -> Result<Self> {
        let path = root.join("zarr.json");
        let metadata = read_json(&path)?;
        let multiscale = metadata
            .pointer("/attributes/ome/multiscales/0")
            .or_else(|| metadata.pointer("/attributes/multiscales/0"))
            .ok_or_else(|| Error::invalid(format!("{} has no multiscales", root.display())))?;
        let axes: Vec<Value> = multiscale
            .get("axes")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let name = |axis: &Value| {
            axis.as_str()
                .or_else(|| axis.get("name").and_then(Value::as_str))
                .unwrap_or("")
                .to_ascii_lowercase()
        };
        let index = |wanted: &str| axes.iter().position(|axis| name(axis) == wanted);
        let spatial = [index("z"), index("y"), index("x")];
        if spatial.contains(&None) {
            return Err(Error::invalid(format!(
                "{}: the axes have no z, y and x",
                root.display()
            )));
        }
        let unit = spatial[2]
            .and_then(|at| axes[at].get("unit"))
            .and_then(Value::as_str);
        let mut scale = [1.0; 3];
        let mut translation = [0.0; 3];
        let transforms = multiscale
            .pointer("/datasets/0/coordinateTransformations")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for transform in &transforms {
            let key = match transform.get("type").and_then(Value::as_str) {
                Some("scale") => "scale",
                Some("translation") => "translation",
                _ => continue,
            };
            let Some(vector) = transform.get(key).and_then(Value::as_array) else {
                continue;
            };
            for (slot, axis) in spatial.iter().enumerate() {
                if let Some(value) = axis.and_then(|at| vector.get(at)).and_then(Value::as_f64) {
                    match key {
                        "scale" => scale[slot] = value,
                        _ => translation[slot] = value,
                    }
                }
            }
        }
        Ok(Self::new(scale, translation, unit))
    }
}

/// Whether `unit` names the micrometre.
pub fn is_micrometre(unit: &str) -> bool {
    matches!(
        unit,
        "micrometer" | "micrometre" | "µm" | "μm" | "um" | "micron"
    )
}

fn same_unit(a: &str, b: &str) -> bool {
    a == b || (is_micrometre(a) && is_micrometre(b))
}

/// `sample = image * scale + offset` per axis, in level-0 voxels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageToSample {
    pub scale: [f64; 3],
    pub offset: [f64; 3],
}

impl ImageToSample {
    /// Image level-0 voxels to the sample's, through physical space, with
    /// `offset_um` added to the image's physical positions.
    ///
    /// Refused: a scale that is not positive, an offset that is not finite,
    /// two units that differ (when both are known), and a non-zero offset on
    /// an image whose unit is known and not the micrometre.
    pub fn new(image: &ImageFrame, sample: &RegisteredImage, offset_um: [f64; 3]) -> Result<Self> {
        if let (Some(ours), Some(theirs)) = (&image.unit, &sample.unit) {
            if !same_unit(ours, theirs) {
                return Err(Error::invalid(format!(
                    "the image is in {ours:?} and the registration's sample image{} in {theirs:?}",
                    sample
                        .path
                        .as_deref()
                        .map_or(String::new(), |path| format!(" {}", path.display()))
                )));
            }
        }
        if offset_um.iter().any(|value| !value.is_finite()) {
            return Err(Error::invalid(format!(
                "the sample offset {offset_um:?} µm is not finite"
            )));
        }
        if let Some(unit) = image.unit.as_deref().filter(|unit| !is_micrometre(unit)) {
            if offset_um.iter().any(|value| *value != 0.0) {
                return Err(Error::invalid(format!(
                    "the sample offset is in µm and the image is in {unit:?}"
                )));
            }
        }
        let (from, to) = (&image.frame, &sample.frame);
        for axis in 0..3 {
            if !(from.scale[axis] > 0.0 && from.scale[axis].is_finite())
                || !(to.scale[axis] > 0.0 && to.scale[axis].is_finite())
            {
                return Err(Error::invalid(format!(
                    "level-0 scales must be positive: image {:?}, sample {:?}",
                    from.scale, to.scale
                )));
            }
        }
        Ok(Self {
            scale: std::array::from_fn(|a| from.scale[a] / to.scale[a]),
            offset: std::array::from_fn(|a| {
                (from.translation[a] + offset_um[a] - to.translation[a]) / to.scale[a]
            }),
        })
    }

    pub fn apply(&self, at: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|a| at[a] * self.scale[a] + self.offset[a])
    }
}

/// How a registered atlas is opened.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenParams {
    /// The prepared atlas; `None` takes the registration's moving image, when
    /// that is a prepared atlas.
    pub atlas: Option<PathBuf>,
    /// Compare the atlas's reference with the registration's moving image
    /// voxel for voxel (when that image can be read) and refuse a difference.
    /// The shape and voxel size are compared either way.
    pub check_reference: bool,
}

impl Default for OpenParams {
    fn default() -> Self {
        Self {
            atlas: None,
            check_reference: true,
        }
    }
}

/// How points read the atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LookupParams {
    /// Collapse regions onto their ancestor at this tree depth.
    pub level: Option<i64>,
    pub rounding: Rounding,
}

/// Region, hemisphere and distance to the surface per point, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct AtlasLookup {
    /// Region id, collapsed to [`LookupParams::level`]; 0 outside the atlas.
    pub region: Vec<i64>,
    /// Hemisphere id, `-1` outside the atlas; `None` when the atlas has no
    /// hemisphere volume.
    pub hemisphere: Option<Vec<i64>>,
    /// Distance to the brain surface in atlas voxels, read at the nearest
    /// edge voxel outside the atlas.
    pub distance_to_surface_voxels: Vec<f64>,
}

/// Points mapped from an image into the sample grid and the atlas.
#[derive(Debug, Clone, PartialEq)]
pub struct MappedPoints {
    pub sample: Vec<[f64; 3]>,
    pub atlas: Vec<[f64; 3]>,
    /// Points outside the sample's level-0 box (still mapped: outside a
    /// B-spline's grid the deformation is the identity).
    pub outside_sample: usize,
}

/// A density on the atlas grid.
#[derive(Debug, Clone)]
pub struct AtlasDensity {
    /// `F64`, the atlas's shape.
    pub voxels: Voxels,
    /// Points left out: outside the atlas, or with a non-finite position or
    /// weight.
    pub dropped: usize,
}

/// A registration and the prepared atlas it maps into.
#[derive(Debug, Clone)]
pub struct RegisteredAtlas {
    /// The registration (or chain) directory.
    pub registration: PathBuf,
    pub transform: PointTransform,
    /// The prepared atlas, resident at level 0. Its distance to the surface
    /// is moved to [`Self::distance_to_surface_voxels`].
    pub atlas: PreparedAtlas,
    /// The prepared atlas's own level-0 frame, for images written on its grid.
    pub atlas_image: ImageFrame,
    distance_voxels: Voxels,
    /// The resident volumes' bytes, leased from the process's memory budget
    /// for as long as this (or a clone, which shares it) is alive. A clone
    /// copies the volumes without leasing the copy.
    lease: Option<Arc<Lease>>,
}

impl RegisteredAtlas {
    /// Open the registration at `registration` (a `register` or `chain`
    /// directory) and its prepared atlas, and check that they belong
    /// together.
    pub fn open(registration: &Path, params: &OpenParams) -> Result<Self> {
        let transform = PointTransform::from_registration_dir(registration)?;
        let moving = transform.atlas_frame().clone();
        let atlas_root = match &params.atlas {
            Some(path) => path.clone(),
            None => match &moving.path {
                Some(path) if is_prepared_atlas(path) => path.clone(),
                _ => {
                    return Err(Error::invalid(format!(
                        "the registration's moving image ({}) is not a prepared atlas; give the \
                         prepared atlas it was made from with --atlas",
                        moving
                            .path
                            .as_deref()
                            .map_or("not recorded".into(), |p| p.display().to_string())
                    )))
                }
            },
        };
        // The volumes are leased, as stored, before they are read; a distance
        // to the surface the atlas lacks is computed in float64.
        let stored = prepared_atlas_bytes(&atlas_root)?;
        let computed_distance = if stored.distance == 0 {
            8 * memory::voxels(moving.shape)
        } else {
            0
        };
        let lease = memory::lease(
            stored.total() + computed_distance,
            &format!("the prepared atlas {}", atlas_root.display()),
        )?
        .map(Arc::new);
        let mut atlas = open_prepared_atlas(&atlas_root, OntologyLevel::TreeDepth)?;
        let shape = atlas.stack.annotation.shape();
        if shape != moving.shape {
            return Err(Error::invalid(format!(
                "the atlas {} is {shape:?} and the registration {} maps into a {:?} moving \
                 image; label with the atlas the registration used (prepared with the same \
                 orientation and crop)",
                atlas_root.display(),
                registration.display(),
                moving.shape
            )));
        }
        for axis in 0..3 {
            let (ours, theirs) = (atlas.voxel_size_um[axis], moving.frame.scale[axis]);
            if (ours - theirs).abs() > 1e-6 * ours.abs().max(1.0) {
                return Err(Error::invalid(format!(
                    "the atlas {} has {:?} µm voxels and the registration's moving image {:?}",
                    atlas_root.display(),
                    atlas.voxel_size_um,
                    moving.frame.scale
                )));
            }
        }
        if params.check_reference {
            check_reference(&atlas, &moving, &atlas_root)?;
        }
        let distance_voxels = match atlas.stack.distance_to_surface.take() {
            Some(distance) => distance,
            None => {
                let size = atlas.voxel_size_um;
                if size
                    .iter()
                    .any(|um| (um - size[0]).abs() > 1e-9 * size[0].abs())
                {
                    return Err(Error::invalid(format!(
                        "the atlas {} has no distance to the surface, and its {size:?} µm voxels \
                         are not cubic, so a distance in voxels has no meaning; prepare it with \
                         one",
                        atlas_root.display()
                    )));
                }
                // Beside the output: the mask, and the transform's own
                // float64 field.
                let _computing = memory::lease(
                    9 * memory::voxels(shape),
                    "the atlas's distance to the surface, computed",
                )?;
                distance_to_surface(&atlas.stack.annotation, &DistanceParams::default())?
            }
        };
        let atlas_image = ImageFrame::from_ome_zarr(&atlas_root)?;
        Ok(Self {
            registration: registration.to_path_buf(),
            transform,
            atlas,
            atlas_image,
            distance_voxels,
            lease,
        })
    }

    /// The bytes leased for the resident volumes (`None` when the budget
    /// could not grant them and no binding cap was stated).
    pub fn leased_bytes(&self) -> Option<u64> {
        self.lease.as_ref().map(|lease| lease.bytes())
    }

    /// The registration's sample, its first fixed image.
    pub fn sample(&self) -> &RegisteredImage {
        self.transform.sample_frame()
    }

    /// The atlas's level-0 shape, `[z, y, x]`.
    pub fn shape(&self) -> [usize; 3] {
        self.atlas.stack.annotation.shape()
    }

    /// The atlas voxel size, µm, `[z, y, x]`.
    pub fn voxel_size_um(&self) -> [f64; 3] {
        self.atlas.voxel_size_um
    }

    /// The edge of a cubic atlas voxel in µm, the factor from
    /// `distance_to_surface_voxels` to micrometres; `None` when the voxels are
    /// not cubic.
    pub fn voxel_length_um(&self) -> Option<f64> {
        let size = self.voxel_size_um();
        size.iter()
            .all(|um| (um - size[0]).abs() <= 1e-9 * size[0].abs())
            .then_some(size[0])
    }

    /// One atlas voxel's volume, mm³.
    pub fn voxel_volume_mm3(&self) -> f64 {
        self.voxel_size_um().iter().product::<f64>() * 1e-9
    }

    /// Distance to the surface on the atlas grid, in atlas voxels.
    pub fn distance_to_surface_voxels(&self) -> &Voxels {
        &self.distance_voxels
    }

    /// The change of lattice from an image's level-0 voxels to the sample's,
    /// with `offset_um` added to the image's physical positions.
    pub fn image_to_sample(
        &self,
        image: &ImageFrame,
        offset_um: [f64; 3],
    ) -> Result<ImageToSample> {
        ImageToSample::new(image, self.sample(), offset_um)
    }

    /// Image level-0 voxels to the sample grid and on to atlas voxels.
    pub fn map(&self, to_sample: &ImageToSample, points: &[[f64; 3]]) -> MappedPoints {
        let sample: Vec<[f64; 3]> = points.iter().map(|p| to_sample.apply(*p)).collect();
        let shape = self.sample().shape;
        let outside_sample = sample
            .iter()
            .filter(|p| (0..3).any(|a| !(p[a] >= -0.5 && p[a] < shape[a] as f64 - 0.5)))
            .count();
        let atlas = self.transform.apply(&sample);
        MappedPoints {
            sample,
            atlas,
            outside_sample,
        }
    }

    /// Image level-0 voxels to atlas voxels, with no offset.
    pub fn map_points(&self, image: &ImageFrame, points: &[[f64; 3]]) -> Result<Vec<[f64; 3]>> {
        Ok(self
            .map(&self.image_to_sample(image, [0.0; 3])?, points)
            .atlas)
    }

    /// A warning when an image on `image` of `shape` voxels reaches more than
    /// one sample voxel outside the sample's box: the two images are then
    /// probably not in one physical frame.
    pub fn frame_warning(&self, to_sample: &ImageToSample, shape: [usize; 3]) -> Option<String> {
        let low = to_sample.apply([0.0; 3]);
        let high = to_sample.apply(shape.map(|n| n as f64 - 1.0));
        let sample = self.sample().shape;
        let outside = (0..3).any(|axis| {
            let (a, b) = (low[axis].min(high[axis]), low[axis].max(high[axis]));
            a < -1.0 || b > sample[axis] as f64
        });
        outside.then(|| {
            format!(
                "the image spans sample voxels {low:?} to {high:?} and the registration's \
                 sample is {sample:?}; are both images in one physical frame?"
            )
        })
    }

    /// The atlas volumes on the atlas grid; the distance is in voxels.
    pub fn volumes(&self) -> AtlasVolumes<'_> {
        let volumes = AtlasVolumes::new(&self.atlas.stack.annotation)
            .with_distance_to_surface(&self.distance_voxels);
        match &self.atlas.stack.hemispheres {
            Some(hemispheres) => volumes.with_hemispheres(hemispheres),
            None => volumes,
        }
    }

    /// An annotator over the atlas's own region tree, reading as `params`
    /// says: region 0 and hemisphere `-1` outside, the distance clamped.
    pub fn annotator(&self, params: &LookupParams) -> Annotator {
        Annotator::new(
            AnnotateParams::default()
                .with_level(params.level)
                .with_rounding(params.rounding)
                .with_distance_out_of_bounds(OutOfBounds::Clamp),
        )
        .with_ontology(Arc::new(self.atlas.ontology.clone()))
    }

    /// Region, hemisphere and distance at each of `points` (atlas voxels).
    pub fn lookup(&self, points: &[[f64; 3]], params: &LookupParams) -> Result<AtlasLookup> {
        let columns = self.annotator(params).columns(self.volumes(), points)?;
        Ok(AtlasLookup {
            region: columns.annotation,
            hemisphere: self
                .atlas
                .stack
                .hemispheres
                .is_some()
                .then_some(columns.hemisphere),
            distance_to_surface_voxels: columns.distance_to_surface,
        })
    }

    /// The atlas voxel `at` (atlas voxels) falls in under `rounding`; `None`
    /// outside the atlas.
    pub fn voxel(&self, at: [f64; 3], rounding: Rounding) -> Option<[usize; 3]> {
        voxel_index(at, self.shape(), rounding, OutOfBounds::Invalid)
    }

    /// How many of `points` (atlas voxels) have no atlas voxel.
    pub fn outside(&self, points: &[[f64; 3]], rounding: Rounding) -> usize {
        points
            .iter()
            .filter(|p| self.voxel(**p, rounding).is_none())
            .count()
    }

    /// The atlas voxels of every `(region, hemisphere)` pair, regions
    /// collapsed to tree depth `level`; hemisphere `-1` without a hemisphere
    /// volume.
    pub fn region_voxels(&self, level: Option<i64>) -> Result<BTreeMap<(i64, i64), u64>> {
        let annotation = &self.atlas.stack.annotation;
        let hemispheres = self.atlas.stack.hemispheres.as_ref();
        let shape = annotation.shape();
        let mut voxels: BTreeMap<(i64, i64), u64> = BTreeMap::new();
        for z in 0..shape[0] {
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    let at = [z, y, x];
                    let region = label_at(annotation, at)?;
                    let hemisphere = match hemispheres {
                        Some(volume) => label_at(volume, at)?,
                        None => -1,
                    };
                    *voxels.entry((region, hemisphere)).or_insert(0) += 1;
                }
            }
        }
        let Some(level) = level else {
            return Ok(voxels);
        };
        let mut collapsed = BTreeMap::new();
        for ((region, hemisphere), count) in voxels {
            let region = match region {
                0 => 0,
                id => self.atlas.ontology.collapse(id, level)?,
            };
            *collapsed.entry((region, hemisphere)).or_insert(0) += count;
        }
        Ok(collapsed)
    }

    /// `points` (atlas voxels) splatted into a density on the atlas grid: each
    /// adds 1, or its entry of `weights`, to every voxel of a ball of `radius`
    /// voxels around the voxel `rounding` picks. A point is in the density
    /// exactly when [`Self::voxel`] gives it one (and its weight is finite).
    pub fn density(
        &self,
        points: &[[f64; 3]],
        weights: Option<&[f64]>,
        rounding: Rounding,
        radius: [usize; 3],
    ) -> Result<AtlasDensity> {
        if let Some(weights) = weights {
            if weights.len() != points.len() {
                return Err(Error::invalid(format!(
                    "{} weights for {} points",
                    weights.len(),
                    points.len()
                )));
            }
        }
        let shape = self.shape();
        let mut placed = Vec::with_capacity(points.len());
        let mut dropped = 0;
        for (index, point) in points.iter().enumerate() {
            let weight = weights.map_or(1.0, |weights| weights[index]);
            match self.voxel(*point, rounding) {
                Some(at) if weight.is_finite() => placed.push(Point::weighted(at, weight)),
                _ => dropped += 1,
            }
        }
        let grid = BlockGrid::whole(shape)?;
        let mut out = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
        voxelize_into(
            &[([0, 0, 0], placed)],
            &grid,
            &ball(radius),
            &Region::whole(&shape),
            out.view_mut(),
        )?;
        Ok(AtlasDensity {
            voxels: Voxels::from(out),
            dropped,
        })
    }

    /// What outputs record about the atlas space.
    pub fn describe(&self) -> Value {
        let sample = self.sample();
        json!({
            "registration": self.registration.display().to_string(),
            "registration_steps": self.transform.steps(),
            "atlas": self.atlas.root.display().to_string(),
            "atlas_shape": self.shape(),
            "atlas_voxel_size_um": self.voxel_size_um(),
            "window": self.atlas.window.to_json(),
            "sample_image": sample.path,
            "sample_shape": sample.shape,
            "sample_frame": {"scale": sample.frame.scale, "translation": sample.frame.translation},
            "sample_unit": sample.unit,
            "distance_to_surface_unit": "atlas voxel",
            "coordinates": "coordinates_atlas",
            "coordinates_unit": "atlas level-0 voxel",
        })
    }
}

/// A prepared atlas's level-0 volumes, bytes as stored.
struct PreparedBytes {
    reference: u64,
    annotation: u64,
    hemispheres: u64,
    distance: u64,
}

impl PreparedBytes {
    fn total(&self) -> u64 {
        self.reference + self.annotation + self.hemispheres + self.distance
    }
}

/// What `open_prepared_atlas` will hold resident for the atlas at `root`.
fn prepared_atlas_bytes(root: &Path) -> Result<PreparedBytes> {
    let level0 = |path: PathBuf| -> Result<u64> {
        if !path.join("zarr.json").is_file() {
            return Ok(0);
        }
        let (dtype, shape) = blockflow::ome_zarr::OmeVolumePyramid::open(&path, 0, 0)?
            .attached_level(0)?
            .metadata()?;
        Ok(memory::voxels(shape) * dtype.size_of() as u64)
    };
    let labels = root.join("labels");
    Ok(PreparedBytes {
        reference: level0(root.to_path_buf())?,
        annotation: level0(labels.join(ANNOTATION_LABEL))?,
        hemispheres: level0(labels.join(HEMISPHERES_LABEL))?,
        distance: level0(root.join(DISTANCE_TO_SURFACE))?,
    })
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path)
        .map_err(|error| Error::invalid(format!("reading {}: {error}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|error| Error::invalid(format!("parsing {}: {error}", path.display())))
}

fn is_prepared_atlas(root: &Path) -> bool {
    read_json(&root.join("zarr.json"))
        .ok()
        .is_some_and(|json| json.pointer("/attributes/atlas").is_some())
}

/// Refuse an atlas whose reference is not the registration's moving image.
fn check_reference(atlas: &PreparedAtlas, moving: &RegisteredImage, root: &Path) -> Result<()> {
    let Some(path) = moving
        .path
        .as_deref()
        .filter(|path| path.join("zarr.json").is_file())
    else {
        eprintln!(
            "atlas: the registration's moving image is not readable; the atlas is checked by \
             shape and voxel size only"
        );
        return Ok(());
    };
    let reference = atlas
        .stack
        .reference
        .as_ref()
        .ok_or_else(|| Error::invalid("the prepared atlas has no reference image"))?;
    let (dtype, shape) = blockflow::ome_zarr::OmeVolumePyramid::open(path, 0, 0)?
        .attached_level(0)?
        .metadata()?;
    let _moving = memory::lease(
        memory::voxels(shape) * dtype.size_of() as u64,
        "the registration's moving image, read to check the atlas reference",
    )?;
    let moving = read_volume(path)?;
    if moving.shape() != reference.shape() {
        return Err(Error::invalid(format!(
            "the registration's moving image {} is {:?} and the atlas reference {:?}",
            path.display(),
            moving.shape(),
            reference.shape()
        )));
    }
    // Equal as stored is equal, with nothing widened; otherwise the voxels
    // that differ are counted a plane at a time.
    if moving == *reference {
        return Ok(());
    }
    let [nz, ny, nx] = moving.shape();
    let mut differing = 0;
    for z in 0..nz {
        let plane = Region::new(&[z, 0, 0], &[1, ny, nx]);
        let (a, b) = (
            moving.slice_region(&plane)?.widened(),
            reference.slice_region(&plane)?.widened(),
        );
        differing += a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    }
    if differing > 0 {
        return Err(Error::invalid(format!(
            "the atlas reference {} differs from the registration's moving image {} at \
             {differing} voxels; label with the atlas the registration used (or pass \
             --skip-atlas-check)",
            root.display(),
            path.display()
        )));
    }
    Ok(())
}
