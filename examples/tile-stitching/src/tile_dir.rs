// SPDX-License-Identifier: MIT
// Original work for this crate.

//! A tile set built from a directory of converted tiles instead of a
//! hand-written tile-set file.
//!
//! Every OME-Zarr image in the directory whose name carries a grid index
//! (`tile_03x04.zarr`: the last `<row> x <column>` in the name, spaces and
//! case of the `x` free) is one tile. The row runs along y and the column
//! along x. Nominal positions come from the first of these that is available:
//!
//! 1. **Physical translations.** When the tiles' level-0
//!    `coordinateTransformations` carry translations that differ between
//!    tiles, each tile's nominal position is its translation relative to the
//!    first tile's, in voxels of the stitched level.
//! 2. **An overlap.** Otherwise the grid with one nominal overlap per axis,
//!    `--overlap`, in level-0 voxels.
//!
//! Only standard OME-Zarr metadata and the file name are read; nothing
//! specific to an instrument or to the format the tiles were converted from.

use std::path::{Path, PathBuf};

use blockflow::error::{Error, Result};
use serde_json::{json, Value};

use crate::io::{TileEntry, TileSetFile};

/// What to build a tile set from a directory with.
#[derive(Debug, Clone, Default)]
pub struct TileDirOptions {
    /// The pyramid level that will be stitched; positions and overlaps are
    /// returned in its voxels.
    pub level: usize,
    /// The nominal overlap in level-0 voxels, `[y, x]`, used when the tiles
    /// carry no distinct translations.
    pub overlap: Option<[f64; 2]>,
    /// Only rows (grid y) in `start..stop`, before rebasing to zero.
    pub rows: Option<(isize, isize)>,
    /// Only columns (grid x) in `start..stop`, before rebasing to zero.
    pub columns: Option<(isize, isize)>,
}

/// A tile set derived from a directory, with a record of where every number
/// came from.
#[derive(Debug, Clone)]
pub struct DerivedTileSet {
    pub tile_set: TileSetFile,
    /// `"translation"` or `"overlap option"`.
    pub positions_from: String,
    /// Per tile and for the set: names, metadata read, notes.
    pub provenance: Value,
}

/// The level-0 and level-`L` frame of one tile, read from its group metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct TileFrame {
    pub scale0: [f64; 3],
    pub translation0: Option<[f64; 3]>,
    pub scale: [f64; 3],
    pub translation: Option<[f64; 3]>,
}

/// `[row, column]` from a tile's name: the last `<digits> x <digits>` in it
/// (spaces around the `x` allowed, either case).
pub fn grid_from_name(name: &str) -> Option<[isize; 2]> {
    let bytes = name.as_bytes();
    let mut found = None;
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() || (i > 0 && bytes[i - 1].is_ascii_digit()) {
            i += 1;
            continue;
        }
        let row_end = digits_end(bytes, i);
        let mut j = row_end;
        while j < bytes.len() && bytes[j] == b' ' {
            j += 1;
        }
        if j < bytes.len() && (bytes[j] == b'x' || bytes[j] == b'X') {
            let mut k = j + 1;
            while k < bytes.len() && bytes[k] == b' ' {
                k += 1;
            }
            if k < bytes.len() && bytes[k].is_ascii_digit() {
                let column_end = digits_end(bytes, k);
                let row = name[i..row_end].parse().ok();
                let column = name[k..column_end].parse().ok();
                if let (Some(row), Some(column)) = (row, column) {
                    found = Some([row, column]);
                }
                i = column_end;
                continue;
            }
        }
        i = row_end;
    }
    found
}

fn digits_end(bytes: &[u8], start: usize) -> usize {
    let mut end = start;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    end
}

/// The group's `multiscales[0]`, NGFF 0.5 (`attributes.ome`) or 0.4.
fn multiscale(group: &Value) -> Option<&Value> {
    group
        .pointer("/attributes/ome/multiscales/0")
        .or_else(|| group.pointer("/attributes/multiscales/0"))
        .or_else(|| group.pointer("/multiscales/0"))
}

fn read_group(root: &Path) -> Result<Value> {
    let path = if root.join("zarr.json").exists() {
        root.join("zarr.json")
    } else {
        root.join(".zattrs")
    };
    let bytes = std::fs::read(&path)
        .map_err(|error| Error::invalid(format!("reading {}: {error}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| Error::invalid(format!("parsing {}: {error}", path.display())))
}

/// The spatial `(z, y, x)` entries of one transformation of `kind` in a
/// dataset's (or multiscale's) `coordinateTransformations`, by the axes'
/// names.
fn spatial(transforms: Option<&Value>, kind: &str, axes: &[String]) -> Option<[f64; 3]> {
    let row = transforms?
        .as_array()?
        .iter()
        .find(|row| row["type"] == kind)?;
    let values = row[kind].as_array()?;
    if values.len() != axes.len() {
        return None;
    }
    let pick = |name: &str| -> Option<f64> {
        let index = axes.iter().position(|axis| axis == name)?;
        values[index].as_f64()
    };
    Some([pick("z")?, pick("y")?, pick("x")?])
}

/// A tile's frame at level 0 and at `level`, from its group metadata. A
/// multiscale-wide transformation is composed onto each dataset's.
pub fn tile_frame(root: &Path, level: usize) -> Result<TileFrame> {
    let group = read_group(root)?;
    let multiscale = multiscale(&group)
        .ok_or_else(|| Error::invalid(format!("{} has no OME-Zarr multiscales", root.display())))?;
    let axes = multiscale["axes"]
        .as_array()
        .map(|axes| {
            axes.iter()
                .map(|axis| {
                    axis["name"]
                        .as_str()
                        .or_else(|| axis.as_str())
                        .unwrap_or("")
                        .to_string()
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let datasets = multiscale["datasets"]
        .as_array()
        .ok_or_else(|| Error::invalid(format!("{} lists no datasets", root.display())))?;
    let global = multiscale.get("coordinateTransformations");
    let global_scale = spatial(global, "scale", &axes).unwrap_or([1.0; 3]);
    let global_translation = spatial(global, "translation", &axes);
    let frame_of = |index: usize| -> Result<([f64; 3], Option<[f64; 3]>)> {
        let dataset = datasets.get(index).ok_or_else(|| {
            Error::invalid(format!(
                "{} has {} levels and level {index} was asked for",
                root.display(),
                datasets.len()
            ))
        })?;
        let transforms = dataset.get("coordinateTransformations");
        let scale = spatial(transforms, "scale", &axes).ok_or_else(|| {
            Error::invalid(format!(
                "level {index} of {} has no z/y/x scale",
                root.display()
            ))
        })?;
        let translation = spatial(transforms, "translation", &axes);
        // Global after per-dataset: p = g_s * (s * i + t) + g_t.
        let composed_scale = std::array::from_fn(|a| global_scale[a] * scale[a]);
        let composed_translation = match (translation, global_translation) {
            (None, None) => None,
            (t, g) => {
                let t = t.unwrap_or([0.0; 3]);
                let g = g.unwrap_or([0.0; 3]);
                Some(std::array::from_fn(|a| global_scale[a] * t[a] + g[a]))
            }
        };
        Ok((composed_scale, composed_translation))
    };
    let (scale0, translation0) = frame_of(0)?;
    let (scale, translation) = frame_of(level)?;
    Ok(TileFrame {
        scale0,
        translation0,
        scale,
        translation,
    })
}

fn is_image(path: &Path) -> bool {
    path.is_dir() && (path.join("zarr.json").exists() || path.join(".zattrs").exists())
}

/// Build the tile set for every gridded OME-Zarr image under `dir`.
pub fn from_directory(dir: &Path, options: &TileDirOptions) -> Result<DerivedTileSet> {
    let mut entries = std::fs::read_dir(dir)
        .map_err(|error| Error::invalid(format!("listing {}: {error}", dir.display())))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| is_image(path))
        .collect::<Vec<PathBuf>>();
    entries.sort();
    let mut skipped = Vec::new();
    let mut tiles = Vec::new();
    for path in entries {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        match grid_from_name(&name) {
            Some([row, column]) => {
                let inside = |range: Option<(isize, isize)>, value: isize| {
                    range.is_none_or(|(start, stop)| (start..stop).contains(&value))
                };
                if inside(options.rows, row) && inside(options.columns, column) {
                    tiles.push((path, name, row, column));
                }
            }
            None => skipped.push(name),
        }
    }
    if tiles.is_empty() {
        return Err(Error::invalid(format!(
            "no OME-Zarr image under {} has a grid index (`RRxCC`) in its name{}",
            dir.display(),
            if options.rows.is_some() || options.columns.is_some() {
                " within the selected rows and columns"
            } else {
                ""
            }
        )));
    }
    // Row-major order, so tile 0 is the lowest row and column.
    tiles.sort_by_key(|(_, _, row, column)| (*row, *column));
    for pair in tiles.windows(2) {
        if (pair[0].2, pair[0].3) == (pair[1].2, pair[1].3) {
            return Err(Error::invalid(format!(
                "{} and {} have the same grid index {:02}x{:02}",
                pair[0].1, pair[1].1, pair[0].2, pair[0].3
            )));
        }
    }
    let row0 = tiles.iter().map(|t| t.2).min().expect("non-empty");
    let column0 = tiles.iter().map(|t| t.3).min().expect("non-empty");

    let mut frames = Vec::with_capacity(tiles.len());
    let mut provenance_tiles = Vec::with_capacity(tiles.len());
    for (path, _, row, column) in &tiles {
        let frame = tile_frame(path, options.level)?;
        provenance_tiles.push(json!({
            "path": path.display().to_string(),
            "name_grid": [row, column],
            "grid": [column - column0, row - row0],
            "scale0": frame.scale0,
            "translation0": frame.translation0,
        }));
        frames.push(frame);
    }
    if let Some(index) = frames
        .iter()
        .position(|frame| frame.scale != frames[0].scale)
    {
        return Err(Error::invalid(format!(
            "{} has voxel size {:?} at level {} and {} has {:?}; the tiles must share one",
            tiles[index].1, frames[index].scale, options.level, tiles[0].1, frames[0].scale
        )));
    }
    let scale = frames[0].scale;
    let scale0 = frames[0].scale0;

    // Translations that tell the tiles apart place them; identical ones (every
    // tile at the origin, or a pyramid's half-voxel offsets) do not.
    let translations = frames
        .iter()
        .map(|frame| frame.translation)
        .collect::<Option<Vec<_>>>();
    let distinct = translations
        .as_ref()
        .is_some_and(|all| all.iter().any(|t| t != &all[0]));
    let entries = tiles
        .iter()
        .map(|(path, _, row, column)| TileEntry {
            path: path.clone(),
            grid: vec![column - column0, row - row0],
            position: None,
        })
        .collect::<Vec<_>>();
    let (tile_set, positions_from) = if distinct {
        let translations = translations.expect("checked");
        let mut entries = entries;
        for (entry, translation) in entries.iter_mut().zip(&translations) {
            entry.position = Some(std::array::from_fn(|a| {
                ((translation[a] - translations[0][a]) / scale[a]).round() as isize
            }));
        }
        let tile_set = TileSetFile {
            grid_axes: vec!["x".into(), "y".into()],
            overlap: None,
            tiles: entries,
        };
        (tile_set, "translation".to_string())
    } else {
        let overlap0 = options.overlap.ok_or_else(|| {
            Error::invalid(format!(
                "the tiles under {} carry no translations that tell them apart; give \
                 --overlap Y,X (level-0 voxels)",
                dir.display()
            ))
        })?;
        // Into the stitched level's voxels; [x, y] for the grid axes.
        let at_level = [
            (overlap0[1] * scale0[2] / scale[2]).round() as isize,
            (overlap0[0] * scale0[1] / scale[1]).round() as isize,
        ];
        let tile_set = TileSetFile {
            grid_axes: vec!["x".into(), "y".into()],
            overlap: Some(at_level.to_vec()),
            tiles: entries,
        };
        (tile_set, "overlap option".to_string())
    };
    Ok(DerivedTileSet {
        provenance: json!({
            "directory": dir.display().to_string(),
            "level": options.level,
            "voxel_size": scale,
            "positions_from": positions_from,
            "skipped": skipped,
            "tiles": provenance_tiles,
        }),
        tile_set,
        positions_from,
    })
}

impl TileSetFile {
    /// The tile set as the JSON the `--tiles` option reads (absolute paths).
    pub fn to_json(&self) -> Value {
        let tiles = self
            .tiles
            .iter()
            .map(|tile| {
                let mut entry = json!({
                    "path": tile.path.display().to_string(),
                    "grid": tile.grid,
                });
                if let Some(position) = tile.position {
                    entry["position"] = json!(position);
                }
                entry
            })
            .collect::<Vec<_>>();
        let mut out = json!({ "grid_axes": self.grid_axes, "tiles": tiles });
        if let Some(overlap) = &self.overlap {
            out["overlap"] = json!(overlap);
        }
        out
    }
}
