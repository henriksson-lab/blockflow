// SPDX-License-Identifier: MIT
// Original work for this crate.

//! A tile set built from a directory of converted tiles instead of a
//! hand-written tile-set file.
//!
//! Every OME-Zarr image in the directory whose name carries a grid index
//! (`tile_03x04.zarr`: the last `<row> x <column>` in the name, spaces and
//! case of the `x` free) is one tile; when no name carries one, every image
//! is. The row runs along y and the column along x. Nominal positions and
//! neighbours come from the first of these that is available:
//!
//! 1. **Physical translations.** When the tiles' level-0
//!    `coordinateTransformations` carry translations that differ between
//!    tiles, each tile's nominal position is its translation relative to the
//!    first tile's, in voxels of the stitched level, and the tiles aligned
//!    are those whose boxes overlap there ([`TileLayout::from_boxes`]); grid
//!    indices are not needed.
//! 2. **The names and an overlap.** Otherwise the grid from the names with
//!    one nominal overlap per axis, `--overlap`, in level-0 voxels, and grid
//!    neighbours.
//!
//! [`TileLayout::from_boxes`]: crate::layout::TileLayout::from_boxes
//!
//! Only standard OME-Zarr metadata and the file name are read; nothing
//! specific to an instrument or to the format the tiles were converted from.

use std::path::{Path, PathBuf};

use blockflow::error::{Error, Result};
use serde_json::{json, Value};

use crate::io::{Neighbours, TileEntry, TileSetFile};

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

/// Build the tile set for the OME-Zarr images under `dir`.
///
/// The tiles are the images whose names carry a grid index, or, when no
/// name carries one, every image. Tiles whose translations tell them apart
/// are placed by them and aligned with the tiles their boxes overlap; grid
/// indices are then only recorded (and select `rows`/`columns`). Otherwise
/// the grid from the names and `overlap` place them.
pub fn from_directory(dir: &Path, options: &TileDirOptions) -> Result<DerivedTileSet> {
    let mut entries = std::fs::read_dir(dir)
        .map_err(|error| Error::invalid(format!("listing {}: {error}", dir.display())))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| is_image(path))
        .collect::<Vec<PathBuf>>();
    entries.sort();
    let named = entries
        .into_iter()
        .map(|path| {
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            let grid = grid_from_name(&name);
            (path, name, grid)
        })
        .collect::<Vec<_>>();
    let gridded = named.iter().any(|(_, _, grid)| grid.is_some());
    let selecting = options.rows.is_some() || options.columns.is_some();
    if selecting && !gridded {
        return Err(Error::invalid(format!(
            "--rows/--columns select by the grid index in the names, and no image under {} has one",
            dir.display()
        )));
    }
    let mut skipped = Vec::new();
    // (path, name, [row, column] when named)
    let mut tiles: Vec<(PathBuf, String, Option<[isize; 2]>)> = Vec::new();
    for (path, name, grid) in named {
        match grid {
            Some([row, column]) => {
                let inside = |range: Option<(isize, isize)>, value: isize| {
                    range.is_none_or(|(start, stop)| (start..stop).contains(&value))
                };
                if inside(options.rows, row) && inside(options.columns, column) {
                    tiles.push((path, name, grid));
                }
            }
            None if !gridded => tiles.push((path, name, None)),
            None => skipped.push(name),
        }
    }
    if tiles.is_empty() {
        return Err(Error::invalid(format!(
            "no OME-Zarr image under {}{}",
            dir.display(),
            if selecting {
                " has a grid index within the selected rows and columns"
            } else {
                ""
            }
        )));
    }
    if gridded {
        // Row-major order, so tile 0 is the lowest row and column.
        tiles.sort_by_key(|(_, _, grid)| grid.expect("gridded"));
        for pair in tiles.windows(2) {
            if pair[0].2 == pair[1].2 {
                let [row, column] = pair[0].2.expect("gridded");
                return Err(Error::invalid(format!(
                    "{} and {} have the same grid index {row:02}x{column:02}",
                    pair[0].1, pair[1].1
                )));
            }
        }
    }
    let row0 = tiles.iter().filter_map(|t| t.2).map(|g| g[0]).min();
    let column0 = tiles.iter().filter_map(|t| t.2).map(|g| g[1]).min();
    // [column, row] rebased to 0, for the grid axes [x, y]
    let rebased = |grid: Option<[isize; 2]>| -> Vec<isize> {
        grid.map_or_else(Vec::new, |[row, column]| {
            vec![
                column - column0.expect("gridded"),
                row - row0.expect("gridded"),
            ]
        })
    };

    let mut frames = Vec::with_capacity(tiles.len());
    let mut provenance_tiles = Vec::with_capacity(tiles.len());
    for (path, _, grid) in &tiles {
        let frame = tile_frame(path, options.level)?;
        provenance_tiles.push(json!({
            "path": path.display().to_string(),
            "name_grid": grid,
            "grid": rebased(*grid),
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
            "{} has voxel size {:?} at level {} and {} has {:?}; the tiles must share one{}",
            tiles[index].1,
            frames[index].scale,
            options.level,
            tiles[0].1,
            frames[0].scale,
            if gridded {
                ""
            } else {
                " (no name carries a grid index, so every image in the directory is a tile)"
            }
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
        .map(|(path, _, grid)| TileEntry {
            path: path.clone(),
            grid: rebased(*grid),
            position: None,
        })
        .collect::<Vec<_>>();
    let mut notes = Vec::new();
    let (tile_set, positions_from) = if distinct {
        let translations = translations.expect("checked");
        let mut entries = entries;
        for (entry, translation) in entries.iter_mut().zip(&translations) {
            entry.position = Some(std::array::from_fn(|a| {
                ((translation[a] - translations[0][a]) / scale[a]).round() as isize
            }));
        }
        for (i, a) in entries.iter().enumerate() {
            if let Some(b) = entries[i + 1..].iter().find(|b| b.position == a.position) {
                return Err(Error::invalid(format!(
                    "{} and {} have the same translation, so one of them is misplaced; fix the \
                     positions (img2omezarr set-positions), or give --overlap and remove the \
                     translations",
                    a.path.display(),
                    b.path.display()
                )));
            }
        }
        if options.overlap.is_some() {
            notes.push(json!(
                "--overlap is ignored: the tiles' translations place them"
            ));
        }
        let tile_set = TileSetFile {
            grid_axes: vec!["x".into(), "y".into()],
            overlap: None,
            neighbours: Some(Neighbours::Boxes),
            tiles: entries,
        };
        (tile_set, "translation".to_string())
    } else {
        if !gridded {
            return Err(Error::invalid(format!(
                "the images under {} carry no translations that tell them apart, and no name \
                 carries a grid index (`RRxCC`), so nothing places them",
                dir.display()
            )));
        }
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
            neighbours: Some(Neighbours::Grid),
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
            "neighbours": tile_set.neighbours().name(),
            "skipped": skipped,
            "notes": notes,
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
                let mut entry = json!({ "path": tile.path.display().to_string() });
                if !tile.grid.is_empty() {
                    entry["grid"] = json!(tile.grid);
                }
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
        if let Some(neighbours) = self.neighbours {
            out["neighbours"] = json!(neighbours.name());
        }
        out
    }
}
