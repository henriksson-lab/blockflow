// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Atlas columns for detected cells, the crust filter, and region volumes.
//!
//! Two frames. Either the annotation is **in the sample's frame already**: an
//! integer volume covering the same physical box as level 0 of the image, at
//! any resolution, where a cell at image voxel `p` reads the annotation voxel
//! whose centre is nearest `(p + 0.5) * A / I - 0.5` per axis, with `A` and `I`
//! the annotation's and the image's extents ([`AtlasFrame::covering`]); the
//! distance to the surface is then in micrometres. Or the points are mapped
//! into a prepared atlas through a registration
//! (`blockflow_atlas_registration::atlas_space`) and labelled there
//! ([`annotate_registered`]); the distance is then in atlas voxels, as the
//! prepared atlas stores it.

use std::collections::BTreeMap;
use std::sync::Arc;

use atlas_regions::{
    distance_to_surface, label_at, AnnotateParams, Annotator, AtlasVolumes, Ontology, OutOfBounds,
    Rounding,
};
use blockflow::error::{Error, Result};
use blockflow::ops::distance::DistanceParams;
use blockflow::voxels::Voxels;
use blockflow_atlas_registration::atlas_space::{LookupParams, RegisteredAtlas};

use crate::detect::{CellAtlas, CellRow};

/// An annotation (and optionally hemispheres and a region tree) placed over an
/// image.
#[derive(Debug, Clone)]
pub struct AtlasFrame {
    pub annotation: Voxels,
    pub hemispheres: Option<Voxels>,
    /// Distance to the brain surface in micrometres, on the annotation's grid.
    pub distance_to_surface: Option<Voxels>,
    pub ontology: Option<Arc<Ontology>>,
    /// Collapse region ids onto their ancestor at this level (needs the tree).
    pub level: Option<i64>,
    /// `atlas = image * scale + offset`, per axis `[z, y, x]`.
    pub scale: [f64; 3],
    pub offset: [f64; 3],
    /// The annotation's voxel size in micrometres, `[z, y, x]`.
    pub voxel_size_um: [f64; 3],
    /// How a continuous annotation coordinate picks its voxel. `Nearest` for
    /// voxel centres at integers (blockflow's convention, the default);
    /// `Floor` is what the reference does (`astype(int)`).
    pub rounding: Rounding,
}

impl AtlasFrame {
    /// An annotation covering the whole of an image of `image_shape` voxels of
    /// `image_voxel_um` micrometres.
    pub fn covering(
        annotation: Voxels,
        image_shape: [usize; 3],
        image_voxel_um: [f64; 3],
    ) -> Result<Self> {
        let atlas_shape = annotation.shape();
        if atlas_shape.contains(&0) || image_shape.contains(&0) {
            return Err(Error::invalid(format!(
                "an annotation of {atlas_shape:?} cannot cover an image of {image_shape:?}"
            )));
        }
        let scale = [0, 1, 2].map(|axis| atlas_shape[axis] as f64 / image_shape[axis] as f64);
        Ok(Self {
            annotation,
            hemispheres: None,
            distance_to_surface: None,
            ontology: None,
            level: None,
            scale,
            offset: scale.map(|factor| 0.5 * factor - 0.5),
            voxel_size_um: [0, 1, 2].map(|axis| image_voxel_um[axis] / scale[axis]),
            rounding: Rounding::Nearest,
        })
    }

    pub fn with_rounding(mut self, rounding: Rounding) -> Self {
        self.rounding = rounding;
        self
    }

    pub fn with_hemispheres(mut self, hemispheres: Voxels) -> Result<Self> {
        if hemispheres.shape() != self.annotation.shape() {
            return Err(Error::invalid(format!(
                "the hemispheres are {:?} and the annotation {:?}; they must share a grid",
                hemispheres.shape(),
                self.annotation.shape()
            )));
        }
        self.hemispheres = Some(hemispheres);
        Ok(self)
    }

    pub fn with_ontology(mut self, ontology: Arc<Ontology>, level: Option<i64>) -> Self {
        self.ontology = Some(ontology);
        self.level = level;
        self
    }

    /// Build the distance-to-surface field from the annotation (`> 0` is
    /// inside), in micrometres.
    pub fn with_surface_distance(mut self) -> Result<Self> {
        let params = DistanceParams::default().with_sampling(self.voxel_size_um);
        self.distance_to_surface = Some(distance_to_surface(&self.annotation, &params)?);
        Ok(self)
    }

    /// An image voxel position in annotation voxel coordinates.
    pub fn to_atlas(&self, at: [f64; 3]) -> [f64; 3] {
        [0, 1, 2].map(|axis| at[axis] * self.scale[axis] + self.offset[axis])
    }

    /// One annotation voxel's volume, in cubic millimetres.
    pub fn voxel_volume_mm3(&self) -> f64 {
        self.voxel_size_um.iter().product::<f64>() * 1e-9
    }

    fn annotator(&self) -> Annotator {
        // The collapse is applied here rather than by the annotator, which
        // would look label 0 (outside the brain, but inside the volume) up in
        // the tree and refuse it. A point outside the annotation has no
        // distance (`NaN`), so the crust drops it.
        Annotator::new(
            AnnotateParams::default()
                .with_rounding(self.rounding)
                .with_distance_out_of_bounds(OutOfBounds::Invalid),
        )
    }

    /// Region, hemisphere and distance for each point (image voxels).
    pub fn columns(&self, points: &[[f64; 3]]) -> Result<Vec<CellAtlas>> {
        let atlas: Vec<[f64; 3]> = points.iter().map(|point| self.to_atlas(*point)).collect();
        self.columns_at(&atlas)
    }

    /// Region, hemisphere and distance for each point given in annotation
    /// voxel coordinates. A point outside the annotation has region `0` and
    /// hemisphere `-1`, as in the reference, and no distance (`NaN`).
    pub fn columns_at(&self, atlas: &[[f64; 3]]) -> Result<Vec<CellAtlas>> {
        let mut volumes = AtlasVolumes::new(&self.annotation);
        if let Some(hemispheres) = &self.hemispheres {
            volumes = volumes.with_hemispheres(hemispheres);
        }
        if let Some(distance) = &self.distance_to_surface {
            volumes = volumes.with_distance_to_surface(distance);
        }
        let columns = self.annotator().columns(volumes, atlas)?;
        (0..atlas.len())
            .map(|row| {
                Ok(CellAtlas {
                    region: self.collapse(columns.annotation[row])?,
                    hemisphere: columns.hemisphere.get(row).copied().unwrap_or(-1),
                    distance_to_surface: columns
                        .distance_to_surface
                        .get(row)
                        .copied()
                        .unwrap_or(f64::NAN),
                    atlas_at: None,
                })
            })
            .collect()
    }

    /// The volume of every `(region, hemisphere)` pair of the annotation, in
    /// cubic millimetres, with the frame's collapse applied. Hemisphere `-1`
    /// when there is no hemisphere volume.
    pub fn region_volumes(&self) -> Result<BTreeMap<(i64, i64), f64>> {
        Ok(self
            .region_voxels()?
            .into_iter()
            .map(|(key, count)| (key, count as f64 * self.voxel_volume_mm3()))
            .collect())
    }

    /// The annotation voxels of every `(region, hemisphere)` pair, with the
    /// frame's collapse applied.
    pub fn region_voxels(&self) -> Result<BTreeMap<(i64, i64), u64>> {
        let shape = self.annotation.shape();
        let mut voxels: BTreeMap<(i64, i64), u64> = BTreeMap::new();
        for z in 0..shape[0] {
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    let at = [z, y, x];
                    let region = label_at(&self.annotation, at)?;
                    let hemisphere = match &self.hemispheres {
                        Some(hemispheres) => label_at(hemispheres, at)?,
                        None => -1,
                    };
                    *voxels.entry((region, hemisphere)).or_insert(0) += 1;
                }
            }
        }
        let mut collapsed = BTreeMap::new();
        for ((region, hemisphere), count) in voxels {
            let region = self.collapse(region)?;
            *collapsed.entry((region, hemisphere)).or_insert(0) += count;
        }
        Ok(collapsed)
    }

    fn collapse(&self, region: i64) -> Result<i64> {
        match (self.level, &self.ontology) {
            (Some(level), Some(ontology)) if region > 0 => ontology.collapse(region, level),
            (Some(_), None) => Err(Error::invalid(
                "a collapse level was given without a region tree",
            )),
            _ => Ok(region),
        }
    }
}

/// Put atlas columns on `rows` and apply the crust filter.
///
/// `origin` is the processed volume's offset in the image (a crop's start),
/// so a row's image position is `at + origin`. With `crust_um > 0` a row
/// passes only if its distance to the surface is strictly above `crust_um`;
/// a cell outside the annotation has no distance and fails.
pub fn annotate_cells(
    rows: &mut [CellRow],
    frame: &AtlasFrame,
    origin: [usize; 3],
    crust_um: f64,
) -> Result<()> {
    check_crust(frame, crust_um)?;
    let points: Vec<[f64; 3]> = rows
        .iter()
        .map(|row| [0, 1, 2].map(|axis| (row.at[axis] + origin[axis]) as f64))
        .collect();
    let columns = frame.columns(&points)?;
    apply_columns(rows, columns, crust_um);
    Ok(())
}

fn apply_columns(rows: &mut [CellRow], columns: Vec<CellAtlas>, crust_um: f64) {
    for (row, atlas) in rows.iter_mut().zip(columns) {
        if crust_um > 0.0 {
            row.passes_filter &= crust_keeps(atlas.distance_to_surface, crust_um);
        }
        row.atlas = Some(atlas);
    }
}

fn check_crust(frame: &AtlasFrame, crust_um: f64) -> Result<()> {
    if !crust_um.is_finite() || crust_um < 0.0 {
        return Err(Error::invalid(format!(
            "the crust is {crust_um} µm; it must be finite and non-negative (0 turns it off)"
        )));
    }
    if crust_um > 0.0 && frame.distance_to_surface.is_none() {
        return Err(Error::invalid(
            "a crust was asked for and the atlas frame has no distance-to-surface field",
        ));
    }
    Ok(())
}

/// The crust test: strictly deeper than `crust_um`. `NaN` (no distance) fails.
pub fn crust_keeps(distance: f64, crust_um: f64) -> bool {
    distance > crust_um
}

/// The crust on the atlas grid, where the distance is in atlas voxels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AtlasCrust {
    Off,
    /// A cell passes when its distance is strictly above this many atlas
    /// voxels: the reference's `distance_from_surface_px`.
    Voxels(f64),
    /// A cell passes when its distance times the atlas voxel's edge is
    /// strictly above `crust_um` (cubic atlas voxels only).
    Micrometres {
        crust_um: f64,
        voxel_um: f64,
    },
}

impl AtlasCrust {
    /// The crust `--crust-voxels` or `--crust-um` names on `atlas`; 0 is off.
    /// A crust in micrometres needs cubic atlas voxels, since the distance is
    /// in voxels.
    pub fn new(voxels: Option<f64>, crust_um: f64, atlas: &RegisteredAtlas) -> Result<Self> {
        let given = voxels.unwrap_or(crust_um);
        if !given.is_finite() || given < 0.0 {
            return Err(Error::invalid(format!(
                "the crust is {given}; it must be finite and non-negative (0 turns it off)"
            )));
        }
        if given == 0.0 {
            return Ok(Self::Off);
        }
        if let Some(voxels) = voxels {
            return Ok(Self::Voxels(voxels));
        }
        let voxel_um = atlas.voxel_length_um().ok_or_else(|| {
            Error::invalid(format!(
                "--crust-um needs cubic atlas voxels, since the atlas's distance to the surface \
                 is in voxels; the atlas has {:?} µm voxels, so give --crust-voxels",
                atlas.voxel_size_um()
            ))
        })?;
        Ok(Self::Micrometres { crust_um, voxel_um })
    }

    /// Whether a cell `distance` atlas voxels deep passes. `NaN` fails.
    pub fn keeps(self, distance: f64) -> bool {
        match self {
            Self::Off => true,
            Self::Voxels(voxels) => distance > voxels,
            // `d * 25 > crust` and not `d > crust / 25`: the decision a
            // distance in micrometres made, exactly.
            Self::Micrometres { crust_um, voxel_um } => distance * voxel_um > crust_um,
        }
    }

    /// The crust in micrometres, when the atlas voxel has one length.
    pub fn um(self, atlas: &RegisteredAtlas) -> Option<f64> {
        match self {
            Self::Off => Some(0.0),
            Self::Voxels(voxels) => atlas.voxel_length_um().map(|um| voxels * um),
            Self::Micrometres { crust_um, .. } => Some(crust_um),
        }
    }

    /// The crust in atlas voxels.
    pub fn voxels(self) -> f64 {
        match self {
            Self::Off => 0.0,
            Self::Voxels(voxels) => voxels,
            Self::Micrometres { crust_um, voxel_um } => crust_um / voxel_um,
        }
    }
}

/// Put atlas columns on `rows` from their positions in atlas voxels (one per
/// row, in order), read through the registered atlas, recording each position
/// in the row, and apply `crust`. The distance column is in atlas voxels; a
/// point outside the atlas has region 0 and hemisphere `-1`, and the distance
/// of the nearest edge voxel.
pub fn annotate_registered(
    rows: &mut [CellRow],
    atlas: &RegisteredAtlas,
    atlas_points: &[[f64; 3]],
    lookup: &LookupParams,
    crust: AtlasCrust,
) -> Result<()> {
    if atlas_points.len() != rows.len() {
        return Err(Error::invalid(format!(
            "{} atlas points for {} cells",
            atlas_points.len(),
            rows.len()
        )));
    }
    let columns = atlas.lookup(atlas_points, lookup)?;
    for (index, (row, point)) in rows.iter_mut().zip(atlas_points).enumerate() {
        let distance = columns.distance_to_surface_voxels[index];
        row.passes_filter &= crust.keeps(distance);
        row.atlas = Some(CellAtlas {
            region: columns.region[index],
            hemisphere: columns.hemisphere.as_ref().map_or(-1, |h| h[index]),
            distance_to_surface: distance,
            atlas_at: Some(*point),
        });
    }
    Ok(())
}
