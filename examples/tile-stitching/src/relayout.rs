// SPDX-License-Identifier: MIT
// Original work for this crate.

//! A fitted layout reused: the positions a stitching run wrote, read back,
//! carried to another pyramid level, and fused over the whole volume or over
//! one box of it.
//!
//! This is how a whole sample is stitched without holding it at full
//! resolution: the alignment runs on a coarser level (every tile and the fused
//! volume fit in memory there), and the positions it found are then used to
//! fuse any box at level 0, reading only the tiles and planes that box needs.
//!
//! **Levels.** A level `L'` position is the level `L` one times
//! `f = s_L / s_L'` per axis, rounded. The wobble axis (z) must keep its voxel
//! size, since the per-slice placement is per plane. With integer `f` (going
//! to a finer level) the positions are exact multiples, so the two fused frames
//! agree exactly.
//!
//! **Frame.** The fused volume's physical frame is that of its first tile:
//! fused voxel `u` is at `t_0 + s (u - (a_0 - o) + c)` per axis, with `t_0`
//! and `s` the first tile's translation and voxel size at the fused level,
//! `a_0` that tile's anchor position (the fused position of its plane 0 along
//! z, its median per-slice position in-plane), `o` the placement's origin and
//! `c` the crop start of the stitched planes (`--z-range`). A box of the fused
//! volume starting at `r` is at that plus `s r`. Two runs over the same
//! positions at different levels therefore describe the same physical space,
//! and tiles placed by distinct stage translations keep them.

use std::path::{Path, PathBuf};

use blockflow::error::{Error, Result};
use blockflow::region::Region;
use blockflow::voxels::Voxels;
use serde_json::Value;

use crate::fuse::{fuse_region, FuseParams, PlacedSource, Placement};
use crate::io::read_tile;
use crate::wobbly::SourceStatus;

/// One tile of a stored layout.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredTile {
    pub path: PathBuf,
    pub shape: [usize; 3],
    /// Where each stitched plane goes (index `k` is tile plane `z_offset + k`),
    /// before the origin is subtracted.
    pub positions: Vec<[i64; 3]>,
    /// Per stitched plane, `SourceStatus`.
    pub status: Vec<i32>,
}

/// The positions a stitching run wrote (`<out>.positions.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct StoredLayout {
    pub tiles: Vec<StoredTile>,
    pub origin: [i64; 3],
    /// The first stitched plane of every tile.
    pub z_offset: usize,
    /// The voxel size the positions are in.
    pub voxel_size: [f64; 3],
}

fn status_from_name(name: &str) -> Result<i32> {
    Ok(match name {
        "isolated" => SourceStatus::ISOLATED,
        "invalid" => SourceStatus::INVALID,
        "valid" => SourceStatus::VALID,
        "fixed" => SourceStatus::FIXED,
        other => {
            return Err(Error::invalid(format!(
                "unknown tile slice status {other:?}"
            )))
        }
    })
}

fn triple<T>(value: &Value, what: &str, read: impl Fn(&Value) -> Option<T>) -> Result<[T; 3]> {
    let array = value
        .as_array()
        .filter(|array| array.len() == 3)
        .ok_or_else(|| Error::invalid(format!("{what}: expected three values, got {value}")))?;
    let mut items = array.iter().map(|item| {
        read(item).ok_or_else(|| Error::invalid(format!("{what}: {item} is not a number")))
    });
    Ok([
        items.next().expect("three")?,
        items.next().expect("three")?,
        items.next().expect("three")?,
    ])
}

impl StoredLayout {
    /// Read a positions file. `voxel_size` is used when the file does not
    /// record one (files written before it did).
    pub fn read(path: &Path, voxel_size: Option<[f64; 3]>) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| Error::invalid(format!("reading {}: {error}", path.display())))?;
        let json: Value = serde_json::from_str(&text)
            .map_err(|error| Error::invalid(format!("parsing {}: {error}", path.display())))?;
        Self::from_json(&json, voxel_size)
            .map_err(|error| Error::invalid(format!("{}: {error}", path.display())))
    }

    pub fn from_json(json: &Value, voxel_size: Option<[f64; 3]>) -> Result<Self> {
        if json["wobble_axis"].as_u64().unwrap_or(0) != 0 {
            return Err(Error::invalid(
                "only layouts that wobble along z are reused",
            ));
        }
        let voxel_size = match json.get("voxel_size") {
            Some(value) if !value.is_null() => triple(value, "voxel_size", Value::as_f64)?,
            _ => voxel_size.ok_or_else(|| {
                Error::invalid("the positions file records no voxel size; give one")
            })?,
        };
        let origin = triple(&json["origin"], "origin", Value::as_i64)?;
        let z_offset = json["z_offset"].as_u64().unwrap_or(0) as usize;
        let tiles = json["tiles"]
            .as_array()
            .ok_or_else(|| Error::invalid("no tiles"))?
            .iter()
            .enumerate()
            .map(|(index, tile)| {
                let path = tile["path"]
                    .as_str()
                    .map(PathBuf::from)
                    .ok_or_else(|| Error::invalid(format!("tile {index} has no path")))?;
                let shape = triple(&tile["shape"], "shape", |v| v.as_u64().map(|v| v as usize))?;
                let (positions, status) = match tile["slices"].as_array() {
                    Some(slices) => {
                        let positions = slices
                            .iter()
                            .map(|slice| triple(slice, "slice position", Value::as_i64))
                            .collect::<Result<Vec<_>>>()?;
                        let status = match tile["status"].as_array() {
                            Some(names) => names
                                .iter()
                                .map(|name| status_from_name(name.as_str().unwrap_or("")))
                                .collect::<Result<Vec<_>>>()?,
                            None => vec![SourceStatus::VALID; positions.len()],
                        };
                        (positions, status)
                    }
                    // A rigid run: every plane at the rigid position.
                    None => {
                        let rigid = triple(&tile["rigid"], "rigid", Value::as_i64)?;
                        let positions = (0..shape[0] as i64)
                            .map(|k| [rigid[0] + k, rigid[1], rigid[2]])
                            .collect::<Vec<_>>();
                        let status = vec![SourceStatus::VALID; positions.len()];
                        (positions, status)
                    }
                };
                if positions.len() != shape[0] || status.len() != shape[0] {
                    return Err(Error::invalid(format!(
                        "tile {index} has {} planes and {} positions",
                        shape[0],
                        positions.len()
                    )));
                }
                Ok(StoredTile {
                    path,
                    shape,
                    positions,
                    status,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if tiles.is_empty() {
            return Err(Error::invalid("no tiles"));
        }
        Ok(Self {
            tiles,
            origin,
            z_offset,
            voxel_size,
        })
    }

    /// The same layout for tiles of `shapes` at `voxel_size`: in-plane
    /// positions and the origin scaled by the voxel-size ratio and rounded.
    pub fn rescaled(&self, voxel_size: [f64; 3], shapes: &[[usize; 3]]) -> Result<Self> {
        if shapes.len() != self.tiles.len() {
            return Err(Error::invalid(format!(
                "the layout has {} tiles and {} shapes were given",
                self.tiles.len(),
                shapes.len()
            )));
        }
        let factor: [f64; 3] = std::array::from_fn(|a| self.voxel_size[a] / voxel_size[a]);
        if (factor[0] - 1.0).abs() > 1e-9 {
            return Err(Error::invalid(format!(
                "the layout's z voxel size is {} and the tiles' is {}; the per-slice placement \
                 is per plane, so z must keep its voxel size",
                self.voxel_size[0], voxel_size[0]
            )));
        }
        let scale = |value: i64, axis: usize| (value as f64 * factor[axis]).round() as i64;
        let tiles = self
            .tiles
            .iter()
            .zip(shapes)
            .enumerate()
            .map(|(index, (tile, &shape))| {
                if shape[0] < self.z_offset + tile.shape[0] {
                    return Err(Error::invalid(format!(
                        "tile {index} has {} planes at this level and the layout places planes \
                         {}..{}",
                        shape[0],
                        self.z_offset,
                        self.z_offset + tile.shape[0]
                    )));
                }
                Ok(StoredTile {
                    path: tile.path.clone(),
                    shape: [tile.shape[0], shape[1], shape[2]],
                    positions: tile
                        .positions
                        .iter()
                        .map(|p| [p[0], scale(p[1], 1), scale(p[2], 2)])
                        .collect(),
                    status: tile.status.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            tiles,
            origin: [
                self.origin[0],
                scale(self.origin[1], 1),
                scale(self.origin[2], 2),
            ],
            z_offset: self.z_offset,
            voxel_size,
        })
    }

    /// The fused volume's shape.
    pub fn shape(&self) -> Result<[usize; 3]> {
        // `Placement::shape` starts at min(lower, 0), which the stored origin
        // already is, so the two frames agree.
        self.placement(None)?.shape()
    }

    /// A placement of every tile's planes `range` (stitched-plane indices,
    /// all when `None`) at its positions minus the origin.
    fn placement(&self, ranges: Option<&[Option<(usize, usize)>]>) -> Result<Placement> {
        let mut sources = Vec::new();
        for (index, tile) in self.tiles.iter().enumerate() {
            let (lo, hi) = match ranges {
                None => (0, tile.shape[0]),
                Some(ranges) => match ranges[index] {
                    Some(range) => range,
                    None => continue,
                },
            };
            let positions = tile.positions[lo..hi]
                .iter()
                .map(|p| std::array::from_fn(|a| p[a] - self.origin[a]))
                .collect();
            sources.push(PlacedSource::new(
                [hi - lo, tile.shape[1], tile.shape[2]],
                0,
                positions,
                tile.status[lo..hi].to_vec(),
            )?);
        }
        Placement::new(0, sources)
    }

    /// The first tile's anchor: the fused position of its plane 0 along z, its
    /// median per-slice position in-plane (before the origin is subtracted).
    pub fn anchor(&self) -> [i64; 3] {
        anchor(&self.tiles[0].positions)
    }

    /// The stitched planes of each tile that reach into `region` (fused
    /// coordinates), as half-open stitched-plane ranges.
    fn planes_in(&self, region: &Region) -> Vec<Option<(usize, usize)>> {
        let lower: [i64; 3] = std::array::from_fn(|a| region.start[a] as i64);
        let upper: [i64; 3] = std::array::from_fn(|a| (region.start[a] + region.shape[a]) as i64);
        self.tiles
            .iter()
            .map(|tile| {
                let touching = (0..tile.shape[0]).filter(|&k| {
                    let p: [i64; 3] =
                        std::array::from_fn(|a| tile.positions[k][a] - self.origin[a]);
                    tile.status[k] >= SourceStatus::VALID
                        && (lower[0]..upper[0]).contains(&p[0])
                        && (1..3).all(|a| p[a] < upper[a] && p[a] + tile.shape[a] as i64 > lower[a])
                });
                let (mut lo, mut hi) = (usize::MAX, 0);
                for k in touching {
                    lo = lo.min(k);
                    hi = hi.max(k + 1);
                }
                (lo < hi).then_some((lo, hi))
            })
            .collect()
    }

    /// Fuse `region` of the fused volume (all of it when `None`), reading only
    /// the tiles and planes it needs from `level` of each tile. Voxels no tile
    /// covers are zero.
    pub fn fuse(
        &self,
        params: &FuseParams,
        region: Option<&Region>,
        channel: usize,
        level: usize,
    ) -> Result<Voxels> {
        let shape = self.shape()?;
        let region = match region {
            Some(region) => {
                region.check_within(&shape, "fused region")?;
                region.clone()
            }
            None => Region::new(&[0, 0, 0], &shape),
        };
        let ranges = self.planes_in(&region);
        let mut tiles = Vec::new();
        for (tile, range) in self.tiles.iter().zip(&ranges) {
            if let Some((lo, hi)) = range {
                let start = self.z_offset + lo;
                let image = read_tile(
                    &tile.path,
                    channel,
                    level,
                    Some((start, self.z_offset + hi)),
                )?;
                if image.voxels.shape()[1..] != tile.shape[1..] {
                    return Err(Error::invalid(format!(
                        "{} is {:?} at level {level} and the layout expects {:?} in-plane",
                        tile.path.display(),
                        image.voxels.shape(),
                        &tile.shape[1..]
                    )));
                }
                tiles.push(image.voxels);
            }
        }
        let region_shape = [region.shape[0], region.shape[1], region.shape[2]];
        if tiles.is_empty() {
            let dtype = read_tile(&self.tiles[0].path, channel, level, Some((0, 1)))?
                .voxels
                .dtype();
            return Voxels::zeros(dtype, region_shape);
        }
        let placement = self.placement(Some(&ranges))?;
        // The placement starts at the fused origin (every position is at or
        // above it) and ends at the last plane of the tiles read; fuse the part
        // of the region inside it.
        let covered = placement.shape()?;
        let start: [usize; 3] = std::array::from_fn(|a| region.start[a]);
        let stop: [usize; 3] =
            std::array::from_fn(|a| (region.start[a] + region.shape[a]).min(covered[a]));
        let mut out = Voxels::zeros(tiles[0].dtype(), region_shape)?;
        if (0..3).all(|a| stop[a] > start[a]) {
            let inner = Region::new(
                &start,
                &std::array::from_fn::<usize, 3, _>(|a| stop[a] - start[a]),
            );
            let fused = fuse_region(params, &tiles, &placement, &inner)?;
            out.assign_region(&Region::new(&[0, 0, 0], &inner.shape), &fused)?;
        }
        Ok(out)
    }
}

/// See [`StoredLayout::anchor`].
pub fn anchor(positions: &[[i64; 3]]) -> [i64; 3] {
    let median = |axis: usize| {
        let mut values = positions.iter().map(|p| p[axis]).collect::<Vec<_>>();
        values.sort_unstable();
        values[(values.len() - 1) / 2]
    };
    [positions[0][0], median(1), median(2)]
}

/// The fused volume's level translation (see the module documentation):
/// `t_0 + s (c - (a_0 - o)) + s r` for a box starting at `region_start`.
pub fn fused_translation(
    tile_translation: [f64; 3],
    voxel_size: [f64; 3],
    anchor: [i64; 3],
    origin: [i64; 3],
    z_offset: usize,
    region_start: [usize; 3],
) -> [f64; 3] {
    let crop = [z_offset as f64, 0.0, 0.0];
    std::array::from_fn(|a| {
        tile_translation[a]
            + voxel_size[a] * (crop[a] - (anchor[a] - origin[a]) as f64 + region_start[a] as f64)
    })
}
