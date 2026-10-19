// SPDX-License-Identifier: MIT
// Original work for this crate.

use std::path::PathBuf;
use std::time::Instant;

use blockflow::region::Region;
use blockflow::Result;
use clap::{Parser, ValueEnum};
use serde_json::json;
use tile_stitching::fuse::{Method, Weighting};
use tile_stitching::io::{aligned_positions_json, read_tile, write_volume_at, TileSetFile};
use tile_stitching::relayout::{anchor, fused_translation, StoredLayout};
use tile_stitching::rigid::RigidParams;
use tile_stitching::tile_dir::{from_directory, TileDirOptions};
use tile_stitching::{align, fuse, StitchParams};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FuseMethod {
    Interpolation,
    Mean,
    Max,
    Min,
}

/// Stitch overlapping OME-Zarr tiles into one OME-Zarr volume.
#[derive(Debug, Parser)]
#[command(name = "tile-stitching")]
struct Cli {
    /// The tile-set JSON: one OME-Zarr image per tile with its grid index and
    /// the nominal overlap (see the README).
    #[arg(long, conflicts_with_all = ["tile_dir", "layout_from"])]
    tiles: Option<PathBuf>,
    /// A directory of OME-Zarr tiles whose names carry their grid index
    /// (`tile_03x04.zarr`: row y, column x); the tile set is built from their
    /// metadata (see the README) and written to `<out>.tiles.json`.
    #[arg(long, conflicts_with = "layout_from")]
    tile_dir: Option<PathBuf>,
    /// With `--tile-dir`: the nominal overlap in level-0 voxels, `Y,X` or one
    /// value for both, instead of the metadata's.
    #[arg(long, value_parser = parse_overlap, requires = "tile_dir")]
    overlap: Option<[f64; 2]>,
    /// With `--tile-dir`: only grid rows `START:STOP` (half-open, as named).
    #[arg(long, value_parser = parse_signed_range, requires = "tile_dir")]
    rows: Option<(isize, isize)>,
    /// With `--tile-dir`: only grid columns `START:STOP` (half-open, as named).
    #[arg(long, value_parser = parse_signed_range, requires = "tile_dir")]
    columns: Option<(isize, isize)>,
    /// Fuse only, at the positions an earlier run wrote (its
    /// `<out>.positions.json`), carried to `--level`. No alignment runs.
    #[arg(long, conflicts_with = "z_range")]
    layout_from: Option<PathBuf>,
    /// With `--layout-from`: fuse only the box `Z0:Z1,Y0:Y1,X0:X1` of the fused
    /// volume (voxels of `--level`); its frame keeps the box's place.
    #[arg(long, value_parser = parse_region, requires = "layout_from")]
    region: Option<[(usize, usize); 3]>,
    /// The fused OME-Zarr image to write. Refused if it exists.
    #[arg(long)]
    out: PathBuf,
    /// The fitted positions as JSON. Defaults to `<out>.positions.json`.
    #[arg(long)]
    positions: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    channel: usize,
    /// The pyramid level of every tile to stitch.
    #[arg(long, default_value_t = 0)]
    level: usize,
    /// Stitch only planes `START:STOP` of z (half-open, level coordinates).
    #[arg(long, value_parser = parse_range)]
    z_range: Option<(usize, usize)>,
    /// The per-slice search range in voxels either side, `Y,X`, or one value
    /// for both.
    #[arg(long, value_parser = parse_shifts, default_value = "15,30")]
    max_shift: [isize; 2],
    /// The rigid step's in-plane search range in voxels either side, `Y,X`, or
    /// one value for both. Only the axis transverse to each pair's tiling axis
    /// is searched.
    #[arg(long, value_parser = parse_shifts, default_value = "30,30")]
    rigid_max_shift: [isize; 2],
    /// The rigid search range along z.
    #[arg(long, default_value_t = 20)]
    max_shift_z: isize,
    /// Fuse at the rigid positions only, without the per-slice alignment.
    #[arg(long)]
    rigid_only: bool,
    /// Measure every N-th slice of each overlap; the rest are interpolated.
    #[arg(long, default_value_t = 3)]
    slice_step: usize,
    #[arg(long, value_enum, default_value_t = FuseMethod::Interpolation)]
    fuse: FuseMethod,
    /// Align and write the positions file only: no fusion, no image. Fuse
    /// later with `--layout-from`, at any level or over a box.
    #[arg(long)]
    positions_only: bool,
    /// Pyramid levels to write, level 0 included.
    #[arg(long, default_value_t = 4)]
    out_levels: usize,
}

fn parse_signed_range(text: &str) -> std::result::Result<(isize, isize), String> {
    let (start, stop) = text
        .split_once(':')
        .ok_or_else(|| format!("expected START:STOP, got {text:?}"))?;
    let parse = |s: &str| s.trim().parse::<isize>().map_err(|e| format!("{s:?}: {e}"));
    let range = (parse(start)?, parse(stop)?);
    if range.0 >= range.1 {
        return Err(format!("{text:?} is empty"));
    }
    Ok(range)
}

/// `Z0:Z1,Y0:Y1,X0:X1`, each non-empty.
fn parse_region(text: &str) -> std::result::Result<[(usize, usize); 3], String> {
    let ranges = text
        .split(',')
        .map(parse_range)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    match ranges[..] {
        [z, y, x] if [z, y, x].iter().all(|r| r.0 < r.1) => Ok([z, y, x]),
        [_, _, _] => Err(format!("{text:?} has an empty range")),
        _ => Err(format!("expected Z0:Z1,Y0:Y1,X0:X1, got {text:?}")),
    }
}

/// `N` or `Y,X`, positive.
fn parse_overlap(text: &str) -> std::result::Result<[f64; 2], String> {
    let values = text
        .split(',')
        .map(|s| s.trim().parse::<f64>().map_err(|e| format!("{s:?}: {e}")))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let overlap = match values[..] {
        [both] => [both, both],
        [y, x] => [y, x],
        _ => return Err(format!("expected N or Y,X, got {text:?}")),
    };
    if overlap.iter().any(|&o| !(o.is_finite() && o > 0.0)) {
        return Err(format!("overlaps must be positive, got {text:?}"));
    }
    Ok(overlap)
}

fn parse_range(text: &str) -> std::result::Result<(usize, usize), String> {
    let (start, stop) = text
        .split_once(':')
        .ok_or_else(|| format!("expected START:STOP, got {text:?}"))?;
    let parse = |s: &str| s.trim().parse::<usize>().map_err(|e| format!("{s:?}: {e}"));
    Ok((parse(start)?, parse(stop)?))
}

/// `N` or `Y,X`, non-negative: a symmetric range per in-plane axis.
fn parse_shifts(text: &str) -> std::result::Result<[isize; 2], String> {
    let values = text
        .split(',')
        .map(|s| s.trim().parse::<isize>().map_err(|e| format!("{s:?}: {e}")))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let shifts = match values[..] {
        [both] => [both, both],
        [y, x] => [y, x],
        _ => return Err(format!("expected N or Y,X, got {text:?}")),
    };
    if shifts.iter().any(|&shift| shift < 0) {
        return Err(format!("search ranges must be non-negative, got {text:?}"));
    }
    Ok(shifts)
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("tile-stitching: use a release build (cargo run --release)");
        std::process::exit(2);
    }
    if let Err(error) = run(&Cli::parse()) {
        eprintln!("tile-stitching: {error}");
        std::process::exit(1);
    }
}

fn sibling(out: &std::path::Path, suffix: &str) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn write_json(path: &std::path::Path, value: &serde_json::Value) -> Result<()> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(blockflow::Error::backend)?,
    )
    .map_err(blockflow::Error::backend)
}

fn fuse_method(method: FuseMethod) -> Method {
    match method {
        FuseMethod::Interpolation => Method::Interpolation,
        FuseMethod::Mean => Method::Mean,
        FuseMethod::Max => Method::Max,
        FuseMethod::Min => Method::Min,
    }
}

fn run(cli: &Cli) -> Result<()> {
    if let Some(layout) = &cli.layout_from {
        return run_relayout(cli, layout);
    }
    let started = Instant::now();
    let tile_set = match (&cli.tiles, &cli.tile_dir) {
        (Some(path), None) => TileSetFile::read(path)?,
        (None, Some(dir)) => {
            let derived = from_directory(
                dir,
                &TileDirOptions {
                    level: cli.level,
                    overlap: cli.overlap,
                    rows: cli.rows,
                    columns: cli.columns,
                },
            )?;
            eprintln!(
                "tile set from {}: {} tiles, nominal positions from the {}",
                dir.display(),
                derived.tile_set.tiles.len(),
                derived.positions_from
            );
            for note in derived.provenance["notes"].as_array().into_iter().flatten() {
                eprintln!("  note: {}", note.as_str().unwrap_or_default());
            }
            let mut file = derived.tile_set.to_json();
            file["derived"] = derived.provenance.clone();
            write_json(&sibling(&cli.out, ".tiles.json"), &file)?;
            derived.tile_set
        }
        _ => {
            return Err(blockflow::Error::invalid(
                "give one of --tiles, --tile-dir or --layout-from",
            ))
        }
    };
    let mut tiles = Vec::with_capacity(tile_set.tiles.len());
    let (mut voxel_size, mut unit, mut tile_translation) = ([1.0; 3], None, [0.0; 3]);
    for (index, entry) in tile_set.tiles.iter().enumerate() {
        let image = read_tile(&entry.path, cli.channel, cli.level, cli.z_range)?;
        let tile = image.voxels;
        eprintln!(
            "tile {index}: {} {:?} {}",
            entry.path.display(),
            tile.shape(),
            tile.dtype().numpy_name()
        );
        if index == 0 {
            voxel_size = image.voxel_size;
            unit = image.unit;
            tile_translation = image.translation;
        }
        tiles.push(tile);
    }
    let shapes = tiles.iter().map(|tile| tile.shape()).collect::<Vec<_>>();
    let layout = tile_set.layout(&shapes)?;
    let grid_axes = tile_set.grid_axes()?;

    let mut params = StitchParams {
        rigid: RigidParams::from_overlaps(&tile_set.overlaps(&layout), &grid_axes),
        wobbly: !cli.rigid_only,
        ..StitchParams::default()
    };
    let [rigid_y, rigid_x] = cli.rigid_max_shift;
    params.rigid.max_shifts = [
        (-cli.max_shift_z, cli.max_shift_z),
        (-rigid_y, rigid_y),
        (-rigid_x, rigid_x),
    ];
    params.measure.max_shifts = cli.max_shift.map(|shift| [-shift, shift]);
    params.measure.slice_step = cli.slice_step;
    params.fuse.method = fuse_method(cli.fuse);
    params.fuse.weighting = Weighting::Pyramid;

    let stitched = align(&tiles, layout.clone(), &params)?;
    for (index, alignment) in stitched.layout.alignments().iter().enumerate() {
        eprintln!(
            "alignment {index}: tiles {} -> {} displacement {:?} quality {:.4e}",
            alignment.pre, alignment.post, alignment.displacement, alignment.quality
        );
    }
    let fused_shape = stitched.placement.shape()?;
    eprintln!(
        "aligned, fused shape {fused_shape:?}, in {:.1?}",
        started.elapsed()
    );
    let fused = if cli.positions_only {
        None
    } else {
        let fused = fuse::resident(&params.fuse, &tiles, &stitched.placement)?;
        eprintln!("fused in {:.1?}", started.elapsed());
        Some(fused)
    };
    drop(tiles);

    // The fused frame is the first tile's (see `relayout`).
    let z_offset = cli.z_range.map_or(0, |range| range.0);
    let first = match &stitched.placed {
        Some(placed) => placed.positions[0]
            .iter()
            .map(|p| [p[0] as i64, p[1] as i64, p[2] as i64])
            .collect::<Vec<_>>(),
        None => {
            let rigid = stitched.layout.positions()[0].map(|v| v as i64);
            (0..shapes[0][0] as i64)
                .map(|k| [rigid[0] + k, rigid[1], rigid[2]])
                .collect()
        }
    };
    let anchor = anchor(&first);
    let origin = stitched.placement.origin();
    let translation = fused_translation(
        tile_translation,
        voxel_size,
        anchor,
        origin,
        z_offset,
        [0; 3],
    );
    let positions_path = cli
        .positions
        .clone()
        .unwrap_or_else(|| sibling(&cli.out, ".positions.json"));
    let paths = tile_set
        .tiles
        .iter()
        .map(|tile| tile.path.clone())
        .collect::<Vec<_>>();
    let mut json = aligned_positions_json(&stitched, &layout, &paths, z_offset);
    json["level"] = json!(cli.level);
    json["channel"] = json!(cli.channel);
    json["voxel_size"] = json!(voxel_size);
    json["unit"] = json!(unit);
    json["anchor"] = json!(anchor);
    json["translation"] = json!(translation);
    // The positions first: they are small and the image write is the slow part.
    write_json(&positions_path, &json)?;
    if let Some(fused) = &fused {
        write_volume_at(
            &cli.out,
            fused,
            voxel_size,
            translation,
            unit.as_deref(),
            cli.out_levels,
        )?;
    }
    println!(
        "wrote {}{} {fused_shape:?} in {:.1?}",
        positions_path.display(),
        if fused.is_some() {
            format!(" and {}", cli.out.display())
        } else {
            String::new()
        },
        started.elapsed()
    );
    Ok(())
}

/// `--layout-from`: fuse the whole volume or `--region` of it at the stored
/// positions, carried to `--level`.
fn run_relayout(cli: &Cli, path: &std::path::Path) -> Result<()> {
    let started = Instant::now();
    let stored = StoredLayout::read(path, None)?;
    let mut shapes = Vec::with_capacity(stored.tiles.len());
    let mut first = None;
    for tile in &stored.tiles {
        let pyramid = blockflow::ome_zarr::OmeVolumePyramid::open(&tile.path, cli.channel, 0)?;
        let (_, shape) = pyramid.attached_level(cli.level)?.metadata()?;
        shapes.push(shape);
        if first.is_none() {
            first = Some(read_tile(&tile.path, cli.channel, cli.level, Some((0, 1)))?);
        }
    }
    let first = first.expect("a stored layout has tiles");
    let layout = stored.rescaled(first.voxel_size, &shapes)?;
    let shape = layout.shape()?;
    let region = cli
        .region
        .map(|ranges| Region::new(&ranges.map(|r| r.0), &ranges.map(|r| r.1 - r.0)));
    let params = tile_stitching::fuse::FuseParams {
        method: fuse_method(cli.fuse),
        weighting: Weighting::Pyramid,
    };
    let fused = layout.fuse(&params, region.as_ref(), cli.channel, cli.level)?;
    let start = region
        .as_ref()
        .map_or([0; 3], |r| [r.start[0], r.start[1], r.start[2]]);
    let translation = fused_translation(
        first.translation,
        first.voxel_size,
        layout.anchor(),
        layout.origin,
        layout.z_offset,
        start,
    );
    write_volume_at(
        &cli.out,
        &fused,
        first.voxel_size,
        translation,
        first.unit.as_deref(),
        cli.out_levels,
    )?;
    let report = json!({
        "layout_from": path.display().to_string(),
        "level": cli.level,
        "voxel_size": first.voxel_size,
        "fused_shape": shape,
        "region": region.as_ref().map(|r| json!({"start": r.start, "shape": r.shape})),
        "shape": fused.shape(),
        "translation": translation,
        "seconds": started.elapsed().as_secs_f64(),
    });
    write_json(&sibling(&cli.out, ".fuse.json"), &report)?;
    println!(
        "wrote {} {:?} (of {:?}) in {:.1?}",
        cli.out.display(),
        fused.shape(),
        shape,
        started.elapsed()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse_overlap, parse_region, parse_shifts};

    #[test]
    fn regions_and_overlaps_parse() {
        assert_eq!(parse_region("0:4,2:6,1:3"), Ok([(0, 4), (2, 6), (1, 3)]));
        assert!(parse_region("0:4,2:6").is_err());
        assert!(parse_region("4:4,2:6,1:3").is_err());
        assert_eq!(parse_overlap("120,127"), Ok([120.0, 127.0]));
        assert_eq!(parse_overlap("100"), Ok([100.0, 100.0]));
        assert!(parse_overlap("0").is_err());
    }

    #[test]
    fn a_search_range_is_one_value_or_one_per_axis() {
        assert_eq!(parse_shifts("30"), Ok([30, 30]));
        assert_eq!(parse_shifts("15,30"), Ok([15, 30]));
        assert_eq!(parse_shifts(" 15 , 30 "), Ok([15, 30]));
        assert!(parse_shifts("15,30,5").is_err());
        assert!(parse_shifts("-1").is_err());
        assert!(parse_shifts("").is_err());
    }
}
