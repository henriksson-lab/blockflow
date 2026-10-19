// SPDX-License-Identifier: MIT
// Original work for this crate.

//! `atlas-prepare`: orient, crop and optionally halve an atlas given as
//! OME-Zarr volumes plus its structure file, and write it as one OME-Zarr
//! atlas (see `atlas_regions::atlas_zarr` for the layout).

use std::path::PathBuf;
use std::time::Instant;

use atlas_regions::atlas_zarr::{
    read_atlas_sources, write_prepared_atlas, AtlasSources, WriteAtlasParams,
};
use atlas_regions::{
    prepare_atlas, Crop, HemisphereSelection, Ontology, OntologyLevel, Orientation, PrepareParams,
};
use blockflow::error::{Error, Result};
use clap::Parser;
use serde_json::json;

#[derive(Parser, Debug)]
#[command(
    name = "atlas-prepare",
    about = "Orient, crop and optionally halve an atlas, and write it as one OME-Zarr image"
)]
struct Args {
    /// The prepared atlas to write (an `.ome.zarr` directory).
    output: PathBuf,
    /// Annotation (region ids) as OME-Zarr.
    #[arg(long)]
    annotation: PathBuf,
    /// Reference (grey-level) image as OME-Zarr; it becomes the atlas image.
    #[arg(long)]
    reference: PathBuf,
    /// Hemisphere labels as OME-Zarr.
    #[arg(long)]
    hemispheres: Option<PathBuf>,
    /// Distance to the brain surface as OME-Zarr, in voxels.
    #[arg(long)]
    distance_to_surface: Option<PathBuf>,
    /// The structure file: the nested JSON graph (`ABA_annotation_last.json`,
    /// the reference's default) or one record per line (`.jsonl`).
    #[arg(long)]
    ontology: PathBuf,
    /// The reference's orientation tuple in its (x, y, z) order, such as
    /// `1,-2,3`, or `left` / `right`.
    #[arg(long, default_value = "1,2,3", allow_hyphen_values = true)]
    orientation: String,
    /// Crop of the oriented atlas, `z0:z1,y0:y1,x0:x1` (Python slices).
    #[arg(long, conflicts_with = "slicing_xyz", allow_hyphen_values = true)]
    crop: Option<String>,
    /// The same crop in the reference's (x, y, z) order, `x0:x1,y0:y1,z0:z1`.
    #[arg(long, allow_hyphen_values = true)]
    slicing_xyz: Option<String>,
    /// Keep one hemisphere: the hemisphere label value to keep (0 or 255 in
    /// the shipped Allen atlas).
    #[arg(long)]
    hemisphere: Option<i64>,
    /// Voxels kept beyond the hemisphere's bounding box.
    #[arg(long, default_value_t = 0)]
    hemisphere_margin: usize,
    /// Source voxel size in µm: one value (isotropic) or `z,y,x`.
    #[arg(long, default_value = "25")]
    voxel_size_um: String,
    /// Display name stored in the atlas metadata.
    #[arg(long, default_value = "atlas")]
    name: String,
    /// Replace an existing output.
    #[arg(long)]
    replace: bool,
    /// Threads for the label pyramids (default: all).
    #[arg(long)]
    workers: Option<usize>,
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("atlas-prepare: this program requires cargo run --release");
        std::process::exit(1);
    }
    match run(&Args::parse()) {
        Ok(report) => println!("{report}"),
        Err(error) => {
            eprintln!("atlas-prepare: {error}");
            std::process::exit(1);
        }
    }
}

fn run(args: &Args) -> Result<String> {
    let started = Instant::now();
    let orientation = Orientation::parse(&args.orientation)?;
    let crop = match (&args.crop, &args.slicing_xyz) {
        (Some(zyx), _) => Crop::parse_zyx(zyx)?,
        (None, Some(xyz)) => Crop::parse_xyz(xyz)?,
        (None, None) => Crop::ALL,
    };
    let voxel_size = parse_voxel_size(&args.voxel_size_um)?;
    let ontology = Ontology::from_file(&args.ontology, OntologyLevel::TreeDepth)?;
    let sources = AtlasSources {
        annotation: args.annotation.clone(),
        reference: Some(args.reference.clone()),
        hemispheres: args.hemispheres.clone(),
        distance_to_surface: args.distance_to_surface.clone(),
    };
    let stack = read_atlas_sources(&sources)?;
    let read_seconds = started.elapsed().as_secs_f64();
    let params = PrepareParams {
        orientation,
        crop,
        hemisphere: args.hemisphere.map(|value| HemisphereSelection {
            value,
            margin: args.hemisphere_margin,
        }),
    };
    let (prepared, window) = prepare_atlas(&stack, &params)?;
    drop(stack);
    let workers = args.workers.unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    });
    let written = write_prepared_atlas(
        &args.output,
        &prepared,
        &window,
        &ontology,
        &WriteAtlasParams {
            voxel_size_um: orientation.orient_resolution(voxel_size),
            name: args.name.clone(),
            replace: args.replace,
            workers,
        },
    )?;
    let report = json!({
        "output": written.root,
        "shape_zyx": written.shape,
        "levels": written.levels,
        "regions_in_ontology": ontology.len(),
        "regions_present": written.regions_present,
        "window": window.to_json(),
        "voxel_size_um": orientation.orient_resolution(voxel_size),
        "read_seconds": read_seconds,
        "total_seconds": started.elapsed().as_secs_f64(),
    });
    serde_json::to_string_pretty(&report).map_err(Error::backend)
}

fn parse_voxel_size(spec: &str) -> Result<[f64; 3]> {
    let values = spec
        .split(',')
        .map(|v| v.trim().parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| Error::invalid(format!("--voxel-size-um {spec:?}: {error}")))?;
    let size = match values.as_slice() {
        [v] => [*v; 3],
        [z, y, x] => [*z, *y, *x],
        _ => {
            return Err(Error::invalid(format!(
                "--voxel-size-um {spec:?}: give one value or z,y,x"
            )))
        }
    };
    if size.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(Error::invalid(format!(
            "--voxel-size-um {spec:?} must be positive"
        )));
    }
    Ok(size)
}
