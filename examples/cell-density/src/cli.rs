// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The `cell-density` command line: argument parsing and the run over one
//! OME-Zarr image. `main.rs` only parses and prints; tests call [`run`].

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use atlas_regions::{Ontology, Rounding};
use blockflow::ome_zarr::{
    ensure_zarr_group, move_path, publish_label, LabelPublication, OmeVolumePyramid,
};
use blockflow::ops::element::ElementShape;
use blockflow::ops::watershed::Separation;
use blockflow::voxels::Voxels;
use blockflow::{AttachedImage, Error, Result, ZarrEnvironment};
use clap::{Parser, ValueEnum};
use serde_json::{json, Map, Value};

use blockflow_atlas_registration::atlas_space::{
    ImageFrame, LookupParams, OpenParams, RegisteredAtlas,
};

use crate::atlas::{annotate_cells, annotate_registered, AtlasCrust, AtlasFrame};
use crate::collapse::{collapse, RegionCell, RegionContext};
use crate::density::{density, place_at};
use crate::detect::{CellPlan, CellRow, PlanOptions};
use crate::params::{Band, CellParams, DogScales, Maxima, Watershed};
use crate::table::{
    image_geometry, utc_timestamp, write_cell_table_named, write_density_image, write_ome_image,
    write_region_csv,
};

#[derive(Debug, Parser)]
#[command(
    about = "Detect cells in an OME-Zarr channel; write object tables, region counts and a density map"
)]
pub struct Args {
    /// OME-Zarr image; level 0 is read, never written.
    #[arg(long)]
    pub input: PathBuf,
    /// Image store to write `tables/` (and `labels/`) into, normally a copy of
    /// the input's metadata with its levels linked.
    #[arg(long, conflicts_with = "in_place")]
    pub output: Option<PathBuf>,
    /// Write into `--input` itself.
    #[arg(long)]
    pub in_place: bool,
    #[arg(long, default_value_t = 0)]
    pub channel: usize,
    #[arg(long, default_value_t = 0)]
    pub time: usize,
    /// Level-0 crop origin, z,y,x. Coordinates in the tables stay global.
    #[arg(long, value_delimiter = ',', requires = "crop_shape")]
    pub crop_start: Option<Vec<usize>>,
    /// Level-0 crop shape, z,y,x.
    #[arg(long, value_delimiter = ',', requires = "crop_start")]
    pub crop_shape: Option<Vec<usize>>,
    /// Compute block, z,y,x. Does not change any result (the watershed runs
    /// on its own tiles, `--watershed-tile`).
    #[arg(long, value_delimiter = ',', default_values_t = [64, 128, 128])]
    pub block: Vec<usize>,
    /// Worker threads (default: all).
    #[arg(long)]
    pub workers: Option<usize>,
    /// Chunk cache for the intermediate images, in MiB. Does not change any
    /// result. The environment's own default is an eighth of the machine,
    /// which on a whole tile is most of the run's memory; 2 GiB holds the halo
    /// re-reads of the blocks in flight (at 1 GiB the watershed phase took
    /// four times the CPU on a whole tile).
    #[arg(long, default_value_t = 2048)]
    pub cache_mb: u64,
    /// Memory cap in MiB: the most the run may hold resident (default:
    /// $BLOCKFLOW_MEMORY_CAP, else the cgroup's limit, else a share of the
    /// machine). The planned phases admit blocks in flight under it and the
    /// chunk cache (at most `--cache-mb`) fills what they leave. Does not
    /// change any result.
    #[arg(long)]
    pub memory_mb: Option<u64>,
    /// Table (and label) name.
    #[arg(long, default_value = "cell-density")]
    pub name: String,
    /// Also publish the basins as `labels/<name>`.
    #[arg(long)]
    pub labels: bool,
    #[arg(long)]
    pub overwrite: bool,

    // ---------------------------------------------------------- detection --
    /// Background element extent, z,y,x (flat in z: per plane).
    #[arg(long, value_delimiter = ',', default_values_t = [1, 10, 10])]
    pub background_size: Vec<usize>,
    #[arg(long, value_enum, default_value_t = Form::Disk)]
    pub background_form: Form,
    /// Difference of Gaussians, inner sigma z,y,x in voxels (with --dog-sigma2).
    #[arg(long, value_delimiter = ',', requires = "dog_sigma2")]
    pub dog_sigma: Option<Vec<f64>>,
    #[arg(long, value_delimiter = ',', requires = "dog_sigma")]
    pub dog_sigma2: Option<Vec<f64>>,
    #[arg(long, default_value_t = 4.0)]
    pub dog_truncate: f64,
    /// h-maxima prominence, in intensity units; 0 is off.
    #[arg(long, default_value_t = 0.0)]
    pub h_max: f64,
    #[arg(long, value_enum, default_value_t = MaximaKind::Window)]
    pub maxima: MaximaKind,
    /// Window maxima radius, z,y,x.
    #[arg(long, value_delimiter = ',', default_values_t = [2, 2, 2])]
    pub maxima_radius: Vec<usize>,
    /// Foreground level above the removed background: seeds and basins. The
    /// default follows the reference's shipped cell-map configuration.
    #[arg(long, default_value_t = 500.0)]
    pub threshold: f64,
    /// Let neighbouring basins touch instead of leaving a one-voxel line.
    #[arg(long)]
    pub no_watershed_line: bool,
    /// Flood per tile with a halo (bounded memory, parallel), or once over the
    /// whole volume (a barrier holding the whole volume, one thread).
    #[arg(long, value_enum, default_value_t = WatershedKind::Tiled)]
    pub watershed: WatershedKind,
    /// Watershed tile, z,y,x. Part of the answer, unlike `--block`.
    #[arg(long, value_delimiter = ',', default_values_t = [64, 128, 128])]
    pub watershed_tile: Vec<usize>,
    /// Voxels read around each watershed tile, z,y,x; part of the answer.
    #[arg(long, value_delimiter = ',', default_values_t = [16, 16, 16])]
    pub watershed_halo: Vec<usize>,
    /// Size band edges in voxels; `none` opens an edge.
    #[arg(long, default_value = "50", value_parser = edge)]
    pub size_min: Edge,
    #[arg(long, default_value = "900", value_parser = edge)]
    pub size_max: Edge,
    /// Intensity (raw max) band edges; `none` opens an edge.
    #[arg(long, default_value = "none", value_parser = edge)]
    pub intensity_min: Edge,
    #[arg(long, default_value = "none", value_parser = edge)]
    pub intensity_max: Edge,

    // -------------------------------------------------------------- atlas --
    /// Integer annotation array (a Zarr array directory, e.g.
    /// `.../labels/atlas/0`) already in the sample's frame, covering level 0.
    #[arg(long, conflicts_with = "registration")]
    pub annotation: Option<PathBuf>,
    /// A registration directory (`atlas-registration register` or `chain`):
    /// its first fixed image is the sample this image's cells are carried
    /// through, its last moving image the atlas. Cells are then labelled,
    /// filtered and voxelised in atlas space.
    #[arg(long)]
    pub registration: Option<PathBuf>,
    /// The prepared atlas (`atlas-prepare` OME-Zarr) the registration used as
    /// its moving image: annotation, hemispheres, distance and region tree
    /// (default: the moving image, when that is a prepared atlas).
    #[arg(long, requires = "registration")]
    pub atlas: Option<PathBuf>,
    /// Added to the detection image's physical positions before they are
    /// read in the registration's sample frame, z,y,x in µm (for a tile
    /// whose metadata does not place it in the sample).
    #[arg(
        long,
        value_delimiter = ',',
        requires = "registration",
        allow_hyphen_values = true
    )]
    pub sample_offset_um: Option<Vec<f64>>,
    /// How an atlas coordinate picks its voxel: nearest centre, or floor
    /// (the reference's `astype(int)`).
    #[arg(long, value_enum, default_value_t = RoundingKind::Nearest)]
    pub atlas_rounding: RoundingKind,
    /// Do not compare the atlas's reference with the registration's moving
    /// image voxel for voxel (the shape is still checked).
    #[arg(long, requires = "registration")]
    pub skip_atlas_check: bool,
    /// Hemisphere array on the annotation's grid.
    #[arg(long, requires = "annotation")]
    pub hemispheres: Option<PathBuf>,
    /// Region tree, Allen-style JSON lines.
    #[arg(long, requires = "annotation")]
    pub ontology: Option<PathBuf>,
    /// Collapse regions onto their ancestor at this level (with --ontology
    /// or --atlas; the atlas's level is the depth in the tree, root 0).
    #[arg(long)]
    pub level: Option<i64>,
    /// Drop cells within this distance of the brain surface, µm (0 is off).
    /// With --registration the distance is in atlas voxels, so this needs
    /// cubic atlas voxels: a cell passes when its distance times the voxel's
    /// edge is strictly above this.
    #[arg(long, default_value_t = 0.0)]
    pub crust_um: f64,
    /// The crust in atlas voxels, as the reference gives it
    /// (`distance_from_surface_px`): a cell passes when its distance
    /// (`distance_to_surface_voxels`) is strictly above this. With
    /// --registration.
    #[arg(long, conflicts_with = "crust_um", requires = "registration")]
    pub crust_voxels: Option<f64>,
    /// List every region of the tree in the region table, empty ones included
    /// (with --ontology or --atlas).
    #[arg(long)]
    pub all_regions: bool,

    // ------------------------------------------------------------ density --
    /// Write a density image (filtered cells) to this path.
    #[arg(long)]
    pub density: Option<PathBuf>,
    /// Density lattice, in level-0 voxels per density voxel, z,y,x (default:
    /// the annotation's grid). With --registration the lattice is the atlas.
    #[arg(long, value_delimiter = ',', conflicts_with = "registration")]
    pub density_factor: Option<Vec<usize>>,
    /// Radius of each cell's ball in the density image, µm (default 100, or
    /// with --registration 5 atlas voxels as the reference's default).
    #[arg(long, conflicts_with = "density_radius_voxels")]
    pub density_radius_um: Option<f64>,
    /// Radius of each cell's ball in density-lattice voxels.
    #[arg(long)]
    pub density_radius_voxels: Option<usize>,
    /// What each cell adds to the density: 1, or its raw or
    /// background-removed intensity (the reference's weighted voxelisation).
    #[arg(long, value_enum, default_value_t = DensityWeight::Count)]
    pub density_weight: DensityWeight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum RoundingKind {
    Nearest,
    Floor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DensityWeight {
    Count,
    Source,
    Background,
}

impl DensityWeight {
    fn of(self, row: &CellRow) -> f64 {
        match self {
            Self::Count => 1.0,
            Self::Source => row.source,
            Self::Background => row.background,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Form {
    Disk,
    Ellipsoid,
    Box,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum WatershedKind {
    Tiled,
    Whole,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum MaximaKind {
    Window,
    Regional,
}

pub type Edge = Option<f64>;

fn edge(text: &str) -> std::result::Result<Edge, String> {
    if text.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    text.parse::<f64>()
        .map(Some)
        .map_err(|_| format!("{text:?} is neither a number nor `none`"))
}

fn triple<T: Copy>(values: &[T], flag: &str) -> Result<[T; 3]> {
    match values {
        [z, y, x] => Ok([*z, *y, *x]),
        _ => Err(Error::invalid(format!(
            "{flag} takes three comma-separated values, z,y,x; got {}",
            values.len()
        ))),
    }
}

/// The detector parameters the flags name.
pub fn cell_params(args: &Args) -> Result<CellParams> {
    let params = CellParams {
        background_form: match args.background_form {
            Form::Disk => ElementShape::ExtentEllipsoid,
            Form::Ellipsoid => ElementShape::Ellipsoid,
            Form::Box => ElementShape::Box,
        },
        background_size: triple(&args.background_size, "--background-size")?,
        dog: match (&args.dog_sigma, &args.dog_sigma2) {
            (Some(sigma), Some(sigma2)) => Some(DogScales {
                sigma: triple(sigma, "--dog-sigma")?,
                sigma2: triple(sigma2, "--dog-sigma2")?,
                truncate: args.dog_truncate,
            }),
            _ => None,
        },
        h: args.h_max,
        maxima: match args.maxima {
            MaximaKind::Window => Maxima::Window {
                radius: triple(&args.maxima_radius, "--maxima-radius")?,
            },
            MaximaKind::Regional => Maxima::Regional,
        },
        threshold: args.threshold,
        separation: if args.no_watershed_line {
            Separation::Adjacent
        } else {
            Separation::Line
        },
        watershed: match args.watershed {
            WatershedKind::Tiled => Watershed::Tiled {
                tile: triple(&args.watershed_tile, "--watershed-tile")?,
                halo: triple(&args.watershed_halo, "--watershed-halo")?,
            },
            WatershedKind::Whole => Watershed::Whole,
        },
        size: Band::new(args.size_min, args.size_max),
        intensity: Band::new(args.intensity_min, args.intensity_max),
    };
    params.validate()?;
    Ok(params)
}

fn write_json_file(path: &Path, value: &Value) -> Result<()> {
    fs::write(
        path,
        serde_json::to_string_pretty(value).map_err(Error::backend)?,
    )
    .map_err(Error::backend)
}

fn prepare_destination(path: &Path, overwrite: bool) -> Result<()> {
    if path.exists() {
        if !overwrite {
            return Err(Error::invalid(format!(
                "{} exists; pass --overwrite to replace it",
                path.display()
            )));
        }
        fs::remove_dir_all(path).map_err(Error::backend)?;
    }
    Ok(())
}

fn quantiles(mut values: Vec<f64>) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    values.sort_by(f64::total_cmp);
    let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
    json!({"min": at(0.0), "p05": at(0.05), "median": at(0.5), "p95": at(0.95), "max": at(1.0)})
}

fn summary(rows: &[&CellRow]) -> Value {
    json!({
        "count": rows.len(),
        "size": quantiles(rows.iter().map(|row| row.size as f64).collect()),
        "source": quantiles(rows.iter().map(|row| row.source).collect()),
    })
}

/// Run the whole command over `args`, returning the run report.
pub fn run(args: &Args) -> Result<Value> {
    let params = cell_params(args)?;
    if args.name.is_empty() || args.name.contains(['/', '\\']) || args.name.starts_with('.') {
        return Err(Error::invalid("--name must be one plain path component"));
    }
    let block = triple(&args.block, "--block")?;
    if block.contains(&0) {
        return Err(Error::invalid("--block must be positive"));
    }
    let started = Instant::now();

    let pyramid = OmeVolumePyramid::open(&args.input, args.channel, args.time)?;
    let full = pyramid
        .levels()
        .first()
        .ok_or_else(|| Error::invalid("the image has no levels"))?
        .shape;
    let origin = args
        .crop_start
        .as_deref()
        .map(|values| triple(values, "--crop-start"))
        .transpose()?
        .unwrap_or([0; 3]);
    let shape = args
        .crop_shape
        .as_deref()
        .map(|values| triple(values, "--crop-shape"))
        .transpose()?
        .unwrap_or(full);
    if (0..3).any(|axis| shape[axis] == 0 || origin[axis] + shape[axis] > full[axis]) {
        return Err(Error::invalid(format!(
            "the crop {origin:?} + {shape:?} is not inside level 0 {full:?}"
        )));
    }
    let cropped = shape != full;
    if args.labels && cropped {
        return Err(Error::invalid(
            "--labels with a crop needs translated label metadata; crop the image into its own \
             OME-Zarr first",
        ));
    }
    let geometry = image_geometry(pyramid.image_root(), full)?;
    let rounding = match args.atlas_rounding {
        RoundingKind::Nearest => Rounding::Nearest,
        RoundingKind::Floor => Rounding::Floor,
    };
    let has_tree = args.ontology.is_some() || args.registration.is_some();
    if args.level.is_some() && !has_tree {
        return Err(Error::invalid(
            "--level needs --ontology or --registration/--atlas",
        ));
    }
    if args.all_regions && !has_tree {
        return Err(Error::invalid(
            "--all-regions needs --ontology or --registration/--atlas",
        ));
    }
    if args.crust_um != 0.0 && args.annotation.is_none() && args.registration.is_none() {
        return Err(Error::invalid(
            "--crust-um needs --annotation or --registration",
        ));
    }
    let lookup = LookupParams {
        level: args.level,
        rounding,
    };
    let registered = match &args.registration {
        Some(registration) => {
            let offset = args
                .sample_offset_um
                .as_deref()
                .map(|values| triple(values, "--sample-offset-um"))
                .transpose()?
                .unwrap_or([0.0; 3]);
            let atlas = RegisteredAtlas::open(
                registration,
                &OpenParams {
                    atlas: args.atlas.clone(),
                    check_reference: !args.skip_atlas_check,
                },
            )?;
            let image = ImageFrame::new(
                geometry.scale_zyx,
                geometry.translation_zyx,
                (geometry.unit != "voxel").then_some(geometry.unit.as_str()),
            );
            let to_sample = atlas.image_to_sample(&image, offset)?;
            if let Some(warning) = atlas.frame_warning(&to_sample, full) {
                eprintln!("atlas: {warning}");
            }
            let crust = AtlasCrust::new(args.crust_voxels, args.crust_um, &atlas)?;
            Some((atlas, to_sample, crust))
        }
        None => None,
    };

    let output = match (&args.output, args.in_place) {
        (Some(output), false) => output.clone(),
        (None, true) => args.input.clone(),
        _ => {
            return Err(Error::invalid(
                "give --output (normally a copy of the input) or say --in-place",
            ))
        }
    };
    if !args.in_place && fs::canonicalize(&output).ok() == fs::canonicalize(&args.input).ok() {
        return Err(Error::invalid(
            "--output is the input store; say --in-place to write into it",
        ));
    }
    let output_pyramid = OmeVolumePyramid::open(&output, args.channel, args.time).ok();
    if let Some(copy) = &output_pyramid {
        if copy.levels().first().map(|level| level.shape) != Some(full) {
            return Err(Error::invalid(format!(
                "{} is an image of a different level-0 shape than {}",
                output.display(),
                args.input.display()
            )));
        }
    } else if args.labels {
        return Err(Error::invalid(format!(
            "--labels needs --output to be an OME-Zarr image; {} is not one",
            output.display()
        )));
    }
    let output_root = output_pyramid
        .as_ref()
        .map(|copy| copy.image_root().to_path_buf())
        .unwrap_or_else(|| output.clone());
    let tables = output_root.join("tables");
    let all_dir = tables.join(&args.name);
    let filtered_dir = tables.join(format!("{}-filtered", args.name));
    prepare_destination(&all_dir, args.overwrite)?;
    prepare_destination(&filtered_dir, args.overwrite)?;
    if args.labels {
        prepare_destination(&output_root.join("labels").join(&args.name), args.overwrite)?;
    }
    if let Some(path) = &args.density {
        prepare_destination(path, args.overwrite)?;
    }
    let work = tables.join(format!(".{}-work", args.name));
    if work.exists() {
        fs::remove_dir_all(&work).map_err(Error::backend)?;
    }

    // ------------------------------------------------------------- detect --
    let source = if cropped {
        pyramid.attached_level(0)?.window(origin, shape)
    } else {
        pyramid.attached_level(0)?
    };
    let (dtype, volume) = source.metadata()?;
    let plan = CellPlan::assemble(
        &params,
        dtype,
        volume,
        block,
        PlanOptions {
            publish_labels: args.labels,
            ..PlanOptions::default()
        },
    )?;
    let env = ZarrEnvironment::attach(&work, std::slice::from_ref(&source))?
        .with_cache(args.cache_mb.max(1) << 20);
    let keep: Vec<_> = plan.published_image.into_iter().collect();
    let detect_started = Instant::now();
    plan.execute("cell density", &env, &plan.hints(args.workers, &keep), &[])?;
    let mut run = plan.cells(&env, &params)?;
    drop(env);
    let detect_seconds = detect_started.elapsed().as_secs_f64();
    eprintln!(
        "{} seeds, {} cells, {} passing the bands in {detect_seconds:.1} s",
        run.seeds,
        run.rows.len(),
        run.filtered().count()
    );

    // -------------------------------------------------------------- atlas --
    let voxel_size = pyramid.voxel_size();
    let mut sample_frame = None;
    if let Some(path) = &args.annotation {
        let annotation = AttachedImage::at(path).read()?;
        let mut atlas = AtlasFrame::covering(annotation, full, voxel_size)?.with_rounding(rounding);
        if let Some(path) = &args.hemispheres {
            atlas = atlas.with_hemispheres(AttachedImage::at(path).read()?)?;
        }
        if let Some(path) = &args.ontology {
            atlas = atlas.with_ontology(Arc::new(Ontology::from_jsonl_file(path)?), args.level);
        }
        let atlas = atlas.with_surface_distance()?;
        annotate_cells(&mut run.rows, &atlas, origin, args.crust_um)?;
        sample_frame = Some(atlas);
    }
    let mut mapping_report = Value::Null;
    if let Some((registered, to_sample, crust)) = &registered {
        let atlas_started = Instant::now();
        let points: Vec<[f64; 3]> = run
            .rows
            .iter()
            .map(|row| [0, 1, 2].map(|axis| (row.at[axis] + origin[axis]) as f64))
            .collect();
        let mapped = registered.map(to_sample, &points);
        let before = run.filtered().count();
        annotate_registered(&mut run.rows, registered, &mapped.atlas, &lookup, *crust)?;
        let outside_atlas = registered.outside(&mapped.atlas, rounding);
        mapping_report = json!({
            "cells_outside_sample_image": mapped.outside_sample,
            "cells_outside_atlas": outside_atlas,
            "filtered_cells_dropped_by_crust": before - run.filtered().count(),
            "seconds": atlas_started.elapsed().as_secs_f64(),
        });
    }
    let registration_provenance = match &registered {
        Some((registered, to_sample, _)) => {
            let mut described = registered.describe();
            described["detection_to_sample"] = json!({
                "scale_zyx": to_sample.scale,
                "offset_zyx": to_sample.offset,
                "rule": "sample voxel = detection voxel * scale + offset, from \
                         (t_d + i * s_d + sample_offset_um - t_s) / s_s",
            });
            described["rounding"] = json!(format!("{rounding:?}"));
            described["level"] = json!(args.level);
            described
        }
        None => Value::Null,
    };
    let filter = match &registered {
        Some((registered, _, crust)) => json!({
            "size": [params.size.min, params.size.max],
            "intensity": [params.intensity.min, params.intensity.max],
            "crust_voxels": crust.voxels(),
            "crust_um": crust.um(registered),
            "distance_to_surface_unit": "atlas voxel",
            "rule": match crust {
                AtlasCrust::Micrometres { .. } => "min <= value < max on each band; \
                    distance_to_surface_voxels * atlas voxel edge (µm) > crust_um",
                _ => "min <= value < max on each band; distance_to_surface_voxels > crust_voxels",
            },
        }),
        None => json!({
            "size": [params.size.min, params.size.max],
            "intensity": [params.intensity.min, params.intensity.max],
            "crust_um": args.crust_um,
            "distance_to_surface_unit": "micrometer",
            "rule": "min <= value < max on each band; distance_to_surface > crust_um",
        }),
    };

    // -------------------------------------------------------------- write --
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let filtered: Vec<CellRow> = run.filtered().cloned().collect();
    let provenance = |kind: &str| {
        let mut attributes = Map::new();
        attributes.insert(
            "cell_density".into(),
            json!({
                "tool": "cell-density",
                "version": env!("CARGO_PKG_VERSION"),
                "date": utc_timestamp(now),
                "command": std::env::args().collect::<Vec<_>>(),
                "input": {
                    "path": fs::canonicalize(&args.input).unwrap_or(args.input.clone()),
                    "channel": args.channel,
                    "time": args.time,
                    "level": 0,
                    "dtype": dtype.numpy_name(),
                    "level0_shape_zyx": full,
                    "crop_zyx": {"start": origin, "shape": shape},
                },
                "table": kind,
                "params": format!("{params:?}"),
                "filter": filter,
                "coordinates": "z, y, x are level-0 voxel indices of the whole image; \
                                *_um = translation + index * scale; atlas_z, atlas_y, atlas_x \
                                (with a registration) are atlas level-0 voxel indices",
                "registration": registration_provenance,
                "block_zyx": block,
                "counts": {"seeds": run.seeds, "cells": run.rows.len(), "filtered": filtered.len()},
            }),
        );
        attributes
    };
    ensure_zarr_group(&tables)?;
    let staged_all = work.join("table-all");
    let region = args.labels.then(|| format!("../../labels/{}", args.name));
    let tree: Option<&Ontology> = match &registered {
        Some((registered, _, _)) => Some(&registered.atlas.ontology),
        None => sample_frame
            .as_ref()
            .and_then(|atlas| atlas.ontology.as_deref()),
    };
    write_cell_table_named(
        &staged_all,
        &run.rows,
        origin,
        &geometry,
        region.as_deref(),
        &provenance("all"),
        tree,
    )?;
    let mut region_rows = None;
    // The region grid: the atlas's through a registration, else the
    // annotation's in the sample's frame. Cells group by their voxel on it.
    let regions_on = match (&registered, &sample_frame) {
        (Some((registered, _, _)), _) => Some((
            registered.shape(),
            registered.region_voxels(args.level)?,
            registered.voxel_volume_mm3(),
            registered.voxel_size_um(),
            json!({
                "frame": "atlas",
                "regions": registered.atlas.root.join("labels").join("annotation"),
                "hemispheres": registered.atlas.root.join("labels").join("hemispheres"),
            }),
        )),
        (None, Some(atlas)) => Some((
            atlas.annotation.shape(),
            atlas.region_voxels()?,
            atlas.voxel_volume_mm3(),
            atlas.voxel_size_um,
            json!({
                "frame": "sample",
                "regions": args.annotation,
                "hemispheres": args.hemispheres,
            }),
        )),
        (None, None) => None,
    };
    if let Some((atlas_shape, voxels, voxel_mm3, voxel_size_um, labels)) = regions_on {
        let cells: Vec<RegionCell> = filtered
            .iter()
            .map(|row| {
                let columns = row.atlas.expect("annotated above");
                // Only the grouping's blocking reads this position.
                let at = columns.atlas_at.unwrap_or_else(|| {
                    sample_frame
                        .as_ref()
                        .expect("a sample-frame annotation without atlas positions")
                        .to_atlas([0, 1, 2].map(|axis| (row.at[axis] + origin[axis]) as f64))
                });
                let at = [0, 1, 2]
                    .map(|axis| (at[axis].round().max(0.0) as usize).min(atlas_shape[axis] - 1));
                RegionCell {
                    at,
                    region: columns.region,
                    hemisphere: columns.hemisphere,
                    size: row.size as f64,
                }
            })
            .collect();
        let volumes = voxels
            .iter()
            .map(|(key, count)| (*key, *count as f64 * voxel_mm3))
            .collect();
        let rows = collapse(
            &cells,
            atlas_shape,
            atlas_shape.map(|extent| extent.min(64)),
            RegionContext {
                ontology: tree,
                volumes: Some(&volumes),
                voxels: Some(&voxels),
                all_regions: args.all_regions,
            },
        )?;
        write_region_csv(&staged_all.join("regions.csv"), &rows)?;
        write_json_file(
            &staged_all.join("regions.json"),
            &json!({
                "table": "regions.csv",
                "key": ["region", "hemisphere"],
                "key_meaning": "region = label value of the annotation (collapsed to `level` \
                                when given; 0 outside the brain and outside the atlas); \
                                hemisphere = label value of the hemisphere image (-1 outside \
                                it or without one)",
                "labels": labels,
                "level": args.level,
                "voxel_size_um_zyx": voxel_size_um,
                "volume": "volume_voxels = annotation voxels of the region on that hemisphere; \
                           volume_mm3 = volume_voxels x voxel volume",
                "cells": "filtered cells (bands and crust)",
            }),
        )?;
        region_rows = Some(rows.len());
    }
    let staged_filtered = work.join("table-filtered");
    write_cell_table_named(
        &staged_filtered,
        &filtered,
        origin,
        &geometry,
        None,
        &provenance("filtered"),
        tree,
    )?;
    if args.labels {
        let published = plan
            .published_image
            .expect("the plan widens the labels when asked to");
        let level0 = work.join(format!("level{}", published.index()));
        publish_label(
            output_pyramid.as_ref().expect("checked above"),
            LabelPublication {
                name: &args.name,
                level0: &level0,
                image_label: None,
                table: Some(&staged_all),
                replace: args.overwrite,
                pyramid_blocks: vec![block],
            },
            args.workers.unwrap_or_else(rayon::current_num_threads),
        )?;
    } else {
        move_path(&staged_all, &all_dir)?;
    }
    move_path(&staged_filtered, &filtered_dir)?;

    // ------------------------------------------------------------ density --
    let mut density_report = Value::Null;
    let weights: Vec<f64> = filtered
        .iter()
        .map(|row| args.density_weight.of(row))
        .collect();
    let weighted = (args.density_weight != DensityWeight::Count).then_some(weights.as_slice());
    if let (Some(path), Some((registered, _, _))) = (&args.density, &registered) {
        let lattice = registered.shape();
        let atlas_um = registered.voxel_size_um();
        let radius = match (args.density_radius_voxels, args.density_radius_um) {
            (Some(voxels), _) => [voxels; 3],
            (None, Some(um)) => atlas_um.map(|voxel| (um / voxel).round() as usize),
            (None, None) => [5; 3],
        };
        let points: Vec<[f64; 3]> = filtered
            .iter()
            .map(|row| {
                row.atlas
                    .and_then(|atlas| atlas.atlas_at)
                    .expect("mapped above")
            })
            .collect();
        let (placed, dropped) = place_at(&points, weighted, lookup.rounding, lattice)?;
        let image = density(
            placed,
            lattice,
            radius,
            lattice.map(|extent| extent.min(128)),
        )?;
        let mut attributes = provenance("density");
        attributes.insert(
            "density".into(),
            json!({"frame": "atlas", "radius_voxels_zyx": radius, "dropped": dropped,
                   "weight": format!("{:?}", args.density_weight).to_lowercase(),
                   "meaning": "sum over filtered cells within the radius of each atlas voxel \
                               of the cell's weight (1 for counts)"}),
        );
        let narrow = Voxels::F32(image.view::<f64>()?.mapv(|value| value as f32));
        let atlas_image = &registered.atlas_image;
        write_ome_image(
            path,
            &narrow,
            atlas_image.frame.scale,
            atlas_image.frame.translation,
            atlas_image.unit.as_deref().unwrap_or("voxel"),
            &attributes,
        )?;
        density_report = json!({"path": path, "frame": "atlas", "shape_zyx": lattice,
                                "radius_voxels_zyx": radius, "dropped": dropped});
    } else if let Some(path) = &args.density {
        let factor =
            match (&args.density_factor, &sample_frame) {
                (Some(factor), _) => triple(factor, "--density-factor")?.map(|value| value as f64),
                (None, Some(atlas)) => atlas.scale.map(|scale| 1.0 / scale),
                (None, None) => return Err(Error::invalid(
                    "--density needs --density-factor (or an --annotation to take the grid from)",
                )),
            };
        if factor.iter().any(|value| !(*value >= 1.0)) {
            return Err(Error::invalid(format!(
                "the density lattice {factor:?} is finer than level 0"
            )));
        }
        let lattice =
            [0, 1, 2].map(|axis| ((full[axis] as f64 / factor[axis]).ceil() as usize).max(1));
        let coarse_um = [0, 1, 2].map(|axis| voxel_size[axis] * factor[axis]);
        let radius = match args.density_radius_voxels {
            Some(voxels) => [voxels; 3],
            None => {
                let um = args.density_radius_um.unwrap_or(100.0);
                coarse_um.map(|coarse| (um / coarse).round() as usize)
            }
        };
        // Fine voxel `p` sits at `(p + 0.5) / factor - 0.5` on the lattice.
        let points: Vec<[f64; 3]> = filtered
            .iter()
            .map(|row| {
                [0, 1, 2]
                    .map(|axis| ((row.at[axis] + origin[axis]) as f64 + 0.5) / factor[axis] - 0.5)
            })
            .collect();
        let (placed, dropped) = place_at(&points, weighted, Rounding::Nearest, lattice)?;
        let image = density(
            placed,
            lattice,
            radius,
            lattice.map(|extent| extent.min(128)),
        )?;
        let mut attributes = provenance("density");
        attributes.insert(
            "density".into(),
            json!({"radius_voxels_zyx": radius, "factor_zyx": factor, "dropped": dropped,
                   "meaning": "number of filtered cells within the radius of each voxel"}),
        );
        write_density_image(path, &image, factor, &geometry, &attributes)?;
        density_report = json!({"path": path, "shape_zyx": lattice, "radius_voxels_zyx": radius});
    }
    fs::remove_dir_all(&work).map_err(Error::backend)?;

    let all: Vec<&CellRow> = run.rows.iter().collect();
    let passing: Vec<&CellRow> = filtered.iter().collect();
    let report = json!({
        "input": args.input,
        "crop_zyx": {"start": origin, "shape": shape},
        "seeds": run.seeds,
        "cells": summary(&all),
        "filtered": summary(&passing),
        "tables": {"all": all_dir, "filtered": filtered_dir},
        "regions": region_rows,
        "atlas_mapping": mapping_report,
        "density": density_report,
        "seconds": {"detect": detect_seconds, "total": started.elapsed().as_secs_f64()},
    });
    Ok(report)
}
