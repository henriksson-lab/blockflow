// SPDX-License-Identifier: MIT

//! Import an Endrov `<nuclineage>` as Blockflow tracking tables inside the
//! OME-Zarr image it annotates. See the example's `README.md`.

use std::path::PathBuf;
use std::process::ExitCode;

use blockflow::{Error, Result};
use celegans_4d::endrov::LeafEnd;
use celegans_4d::import::{run, ImportParams, PixelOrigin};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LeafEndChoice {
    /// A nucleus without children ends at its last keyframe.
    LastKeyframe,
    /// Endrov's rule: hold its last position to the end of the recording.
    Hold,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OriginChoice {
    /// Endrov integer pixel coordinates are pixel corners (Endrov's convention).
    Corner,
    /// Treat them as pixel centres.
    Centre,
}

#[derive(Debug, Parser)]
#[command(
    name = "import_endrov_lineage",
    about = "Endrov nuclineage -> Blockflow detections/tracks/links tables inside an OME-Zarr image"
)]
struct Args {
    /// The multiscale image group the lineage annotates, e.g. `rec.ome.zarr/DIC`. Its stored
    /// frames are the detection timeline. Outputs go into its `tables/`, `labels/`, `tracking/`.
    #[arg(long)]
    image: PathBuf,
    /// `rmd.ostxml`, a `.ost` directory or a `.ost.zip`. Default: `<image>/../OST/rmd.ostxml`,
    /// the copy img2omezarr keeps.
    #[arg(long)]
    ostxml: Option<PathBuf>,
    /// `<nuclineage id=..>` to import. Default: the only one with keyframes that is not
    /// Endrov's model fit `estcell`.
    #[arg(long)]
    lineage: Option<String>,
    /// Table / label / tracking name.
    #[arg(long, default_value = "endrov")]
    layer: String,
    #[arg(long, value_enum, default_value_t = LeafEndChoice::LastKeyframe)]
    leaf_end: LeafEndChoice,
    #[arg(long, value_enum, default_value_t = OriginChoice::Corner)]
    pixel_origin: OriginChoice,
    /// Keep single-keyframe nuclei without relatives (event/landmark markers) as tracks.
    #[arg(long)]
    keep_markers: bool,
    /// Also rasterise each nucleus as a sphere into `labels/<layer>`.
    #[arg(long)]
    labels: bool,
    #[arg(long, default_value_t = 256)]
    label_chunk: usize,
    /// Endpoints closer than this (physical units) to the stack edge are "near border".
    #[arg(long, default_value_t = 1.5)]
    border_distance: f64,
    /// Replace existing outputs of this layer.
    #[arg(long)]
    overwrite: bool,
    #[arg(long, default_value_t = 8)]
    jobs: usize,
}

fn main() -> ExitCode {
    match main_inner() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("import_endrov_lineage: {error}");
            ExitCode::FAILURE
        }
    }
}

fn main_inner() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::invalid("this example requires cargo run --release"));
    }
    let args = Args::parse();
    let ostxml = match args.ostxml {
        Some(path) => path,
        None => args
            .image
            .parent()
            .ok_or_else(|| Error::invalid("--image has no parent directory; pass --ostxml"))?
            .join("OST")
            .join("rmd.ostxml"),
    };
    let params = ImportParams {
        image: args.image,
        ostxml,
        lineage: args.lineage,
        layer: args.layer,
        leaf_end: match args.leaf_end {
            LeafEndChoice::LastKeyframe => LeafEnd::LastKeyframe,
            LeafEndChoice::Hold => LeafEnd::Hold,
        },
        pixel_origin: match args.pixel_origin {
            OriginChoice::Corner => PixelOrigin::Corner,
            OriginChoice::Centre => PixelOrigin::Centre,
        },
        keep_markers: args.keep_markers,
        labels: args.labels,
        label_chunk: args.label_chunk,
        border_distance: args.border_distance,
        overwrite: args.overwrite,
        jobs: args.jobs,
    };
    let (counts, summary) = run(&params)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "image": params.image,
            "layer": params.layer,
            "lineage": summary["source"]["lineage_id"],
            "counts": counts,
        }))
        .map_err(Error::backend)?
    );
    Ok(())
}
