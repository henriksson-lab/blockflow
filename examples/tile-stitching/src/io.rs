// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The CLI's files: the tile-set description, reading tiles from OME-Zarr, and
//! writing the fused volume and the fitted positions.
//!
//! A tile set is a JSON file listing one OME-Zarr image per tile:
//!
//! ```json
//! {
//!   "grid_axes": ["x", "y"],
//!   "overlap": [66, 212],
//!   "tiles": [
//!     { "path": "tile_00x00.zarr", "grid": [0, 0] },
//!     { "path": "tile_01x00.zarr", "grid": [1, 0] }
//!   ]
//! }
//! ```
//!
//! `grid[d]` is the tile's index along `grid_axes[d]`; `overlap[d]` is the
//! nominal overlap in voxels along that axis. Instead of `overlap`, every tile
//! may carry `"position": [z, y, x]`, its nominal stage position in voxels.
//! Paths are relative to the JSON file.

use std::path::{Path, PathBuf};

use blockflow::error::{Error, Result};
use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{PhysicalFrame, PyramidSpec, ZarrEnvironment};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::layout::{TileLayout, TileSource};
use crate::wobbly::{source_status_name, status_name};
use crate::{Aligned, Stitched};

/// The tile-set file.
#[derive(Debug, Clone, Deserialize)]
pub struct TileSetFile {
    #[serde(default = "default_grid_axes")]
    pub grid_axes: Vec<String>,
    #[serde(default)]
    pub overlap: Option<Vec<isize>>,
    pub tiles: Vec<TileEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TileEntry {
    pub path: PathBuf,
    pub grid: Vec<isize>,
    #[serde(default)]
    pub position: Option<[isize; 3]>,
}

fn default_grid_axes() -> Vec<String> {
    vec!["x".to_string(), "y".to_string()]
}

impl TileSetFile {
    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| Error::invalid(format!("reading {}: {error}", path.display())))?;
        let mut file: Self = serde_json::from_str(&text)
            .map_err(|error| Error::invalid(format!("parsing {}: {error}", path.display())))?;
        let base = path.parent().unwrap_or(Path::new("."));
        for tile in &mut file.tiles {
            if tile.path.is_relative() {
                tile.path = base.join(&tile.path);
            }
        }
        Ok(file)
    }

    /// The grid axes as `(z, y, x)` volume axes.
    pub fn grid_axes(&self) -> Result<Vec<usize>> {
        self.grid_axes
            .iter()
            .map(|name| match name.to_ascii_lowercase().as_str() {
                "z" => Ok(0),
                "y" => Ok(1),
                "x" => Ok(2),
                other => Err(Error::invalid(format!(
                    "grid axis {other:?} is not one of z, y, x"
                ))),
            })
            .collect()
    }

    /// The nominal layout for tiles of the given shapes.
    pub fn layout(&self, shapes: &[[usize; 3]]) -> Result<TileLayout> {
        let grid_axes = self.grid_axes()?;
        let sources = self
            .tiles
            .iter()
            .zip(shapes)
            .map(|(tile, &shape)| TileSource {
                shape,
                tile_position: tile.grid.clone(),
            })
            .collect::<Vec<_>>();
        if self.tiles.iter().all(|tile| tile.position.is_some()) {
            let positions = self
                .tiles
                .iter()
                .map(|tile| tile.position.expect("checked"))
                .collect();
            return TileLayout::with_positions(sources, positions, &grid_axes);
        }
        let overlap = self.overlap.as_ref().ok_or_else(|| {
            Error::invalid(
                "the tile set gives neither an `overlap` nor a `position` for every tile",
            )
        })?;
        TileLayout::on_grid(sources, overlap, &grid_axes)
    }

    /// The nominal overlap along each grid axis, for the projection depth.
    pub fn overlaps(&self, layout: &TileLayout) -> Vec<usize> {
        match &self.overlap {
            Some(overlap) => overlap.iter().map(|&o| o.max(1) as usize).collect(),
            // From the positions: the smallest overlap between grid neighbours
            // along each axis.
            None => layout
                .grid_axes()
                .iter()
                .map(|&axis| {
                    layout
                        .alignments()
                        .iter()
                        .filter(|a| a.displacement[axis] != 0)
                        .map(|a| {
                            layout.sources()[a.pre].shape[axis] as isize
                                - a.displacement[axis].abs()
                        })
                        .min()
                        .unwrap_or(1)
                        .max(1) as usize
                })
                .collect(),
        }
    }
}

/// One tile read from an OME-Zarr image.
#[derive(Debug, Clone)]
pub struct TileImage {
    pub voxels: Voxels,
    /// The level's physical voxel size, `(z, y, x)`.
    pub voxel_size: [f64; 3],
    /// The level's physical translation, `(z, y, x)` (zero when none is
    /// recorded).
    pub translation: [f64; 3],
    /// The spatial axes' unit, when the image names one.
    pub unit: Option<String>,
}

/// One tile: a channel of one level of an OME-Zarr image, optionally cropped
/// to `z_range` (half-open, in level coordinates).
pub fn read_tile(
    path: &Path,
    channel: usize,
    level: usize,
    z_range: Option<(usize, usize)>,
) -> Result<TileImage> {
    let pyramid = OmeVolumePyramid::open(path, channel, 0)?;
    let image = pyramid.attached_level(level)?;
    let (_, shape) = image.metadata()?;
    let voxels = match z_range {
        None => image.read()?,
        Some((start, stop)) => {
            if start >= stop || stop > shape[0] {
                return Err(Error::invalid(format!(
                    "z range {start}..{stop} is outside {}'s {} planes",
                    path.display(),
                    shape[0]
                )));
            }
            image.read_region([start, 0, 0], [stop - start, shape[1], shape[2]])?
        }
    };
    let voxel_size = pyramid
        .levels()
        .get(level)
        .and_then(|level| level.coordinate_transformations.as_array())
        .and_then(|rows| rows.iter().find(|row| row["type"] == "scale"))
        .and_then(|row| row["scale"].as_array())
        .and_then(|scale| {
            Some([
                scale.first()?.as_f64()?,
                scale.get(1)?.as_f64()?,
                scale.get(2)?.as_f64()?,
            ])
        })
        .unwrap_or_else(|| pyramid.voxel_size());
    let translation = crate::tile_dir::tile_frame(path, level)
        .ok()
        .and_then(|frame| frame.translation)
        .unwrap_or([0.0; 3]);
    let unit = std::fs::read(pyramid.image_root().join("zarr.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|metadata| {
            let axes = metadata
                .pointer("/attributes/ome/multiscales/0/axes")
                .or_else(|| metadata.pointer("/attributes/multiscales/0/axes"))?
                .as_array()?
                .clone();
            axes.iter()
                .find(|axis| axis["name"] == "x")
                .and_then(|axis| axis["unit"].as_str().map(str::to_owned))
        });
    Ok(TileImage {
        voxels,
        voxel_size,
        translation,
        unit,
    })
}

/// Write the fused volume as a multiscale OME-Zarr image with physical scale
/// `voxel_size` (in `unit`, when given) at level 0.
pub fn write_volume(
    root: &Path,
    volume: &Voxels,
    voxel_size: [f64; 3],
    unit: Option<&str>,
    levels: usize,
) -> Result<()> {
    write_volume_at(root, volume, voxel_size, [0.0; 3], unit, levels)
}

/// [`write_volume`] with level 0 translated by `translation` (physical units).
pub fn write_volume_at(
    root: &Path,
    volume: &Voxels,
    voxel_size: [f64; 3],
    translation: [f64; 3],
    unit: Option<&str>,
    levels: usize,
) -> Result<()> {
    if root.exists() {
        return Err(Error::invalid(format!(
            "{} exists; remove it or choose another output",
            root.display()
        )));
    }
    let shape = volume.shape();
    // Halve y and x per level, and z too while it stays at least as finely
    // sampled as the plane.
    let mut factors = Vec::new();
    let mut extent = shape;
    for _ in 1..levels {
        if extent[1] < 64 && extent[2] < 64 {
            break;
        }
        let z = if extent[0] >= 64 { 2 } else { 1 };
        factors.push([z, 2, 2]);
        extent = [extent[0] / z, extent[1] / 2, extent[2] / 2];
    }
    let spec = PyramidSpec::new(factors)?
        .with_physical(PhysicalFrame::new(voxel_size, unit).with_origin(translation));
    ZarrEnvironment::build_multiscale(root, volume, &spec)?;
    Ok(())
}

/// The fitted positions as JSON: every tile's nominal and rigid position, its
/// per-slice placement and status, and every alignment's rigid shift and
/// per-slice measurement (in-plane displacement of `post` relative to `pre`
/// per slice from `start`, before smoothing and placement).
pub fn positions_json(
    stitched: &Stitched,
    nominal: &TileLayout,
    paths: &[PathBuf],
    z_offset: usize,
) -> Value {
    let aligned = Aligned {
        layout: stitched.layout.clone(),
        wobbly: stitched.wobbly.clone(),
        placed: stitched.placed.clone(),
        placement: stitched.placement.clone(),
    };
    aligned_positions_json(&aligned, nominal, paths, z_offset)
}

/// [`positions_json`] for an alignment that was not fused.
pub fn aligned_positions_json(
    stitched: &Aligned,
    nominal: &TileLayout,
    paths: &[PathBuf],
    z_offset: usize,
) -> Value {
    let layout = &stitched.layout;
    let tiles = (0..layout.sources().len())
        .map(|index| {
            let mut tile = json!({
                "path": paths.get(index).map(|p| p.display().to_string()),
                "grid": layout.sources()[index].tile_position,
                "shape": layout.sources()[index].shape,
                "nominal": nominal.positions()[index],
                "rigid": layout.positions()[index],
            });
            if let Some(placed) = &stitched.placed {
                tile["slices"] = json!(placed.positions[index]);
                tile["status"] = json!(placed.status[index]
                    .iter()
                    .map(|&s| source_status_name(s))
                    .collect::<Vec<_>>());
            }
            tile
        })
        .collect::<Vec<_>>();
    let alignments = (0..layout.alignments().len())
        .map(|index| {
            let alignment = &layout.alignments()[index];
            let mut entry = json!({
                "pre": alignment.pre,
                "post": alignment.post,
                "shift": nominal_shift(nominal, layout, index),
                "quality": if alignment.quality.is_finite() { json!(alignment.quality) } else { Value::Null },
            });
            if let Some(placed) = &stitched.placed {
                if let Some(report) = placed.report.resolution.get(index) {
                    entry["slices"] = json!(report.slices);
                    entry["interpolated"] = json!(report.interpolated);
                    entry["invalidated"] = json!(report.invalidated);
                }
            }
            let measured = stitched
                .wobbly
                .as_ref()
                .and_then(|wobbly| Some((wobbly, wobbly.alignments.get(index)?)));
            if let Some((wobbly, slot)) = measured {
                let start = wobbly.sources[slot.pre].position[wobbly.axis]
                    .max(wobbly.sources[slot.post].position[wobbly.axis]);
                entry["measured"] = json!({
                    "start": start,
                    "displacements": slot.track.displacements,
                    "status": slot.track.status.iter().map(|&s| status_name(s)).collect::<Vec<_>>(),
                });
            }
            entry
        })
        .collect::<Vec<_>>();
    json!({
        "axes": ["z", "y", "x"],
        "wobble_axis": stitched.placement.axis(),
        "z_offset": z_offset,
        "origin": stitched.placement.origin(),
        "shape": stitched.placement.shape().ok(),
        "tiles": tiles,
        "alignments": alignments,
    })
}

/// The rigid correction of alignment `index` relative to the nominal layout.
fn nominal_shift(nominal: &TileLayout, fitted: &TileLayout, index: usize) -> [isize; 3] {
    let alignment = &fitted.alignments()[index];
    let pre = nominal.positions()[alignment.pre];
    let post = nominal.positions()[alignment.post];
    std::array::from_fn(|axis| alignment.displacement[axis] - (post[axis] - pre[axis]))
}
