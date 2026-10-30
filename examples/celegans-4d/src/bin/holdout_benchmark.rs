// SPDX-License-Identifier: MIT

//! Held-out-keyframe benchmark of nucleus position refiners on one Endrov
//! recording. See the example's `README.md` and `celegans_4d::benchmark`.

use std::path::PathBuf;
use std::process::ExitCode;

use blockflow::{Error, Result};
use celegans_4d::benchmark::{
    annotation_stats, evaluate, write_outputs, Linear, LinearPlaneZ, Nearest, Recording, Refiner,
    Scenario,
};
use celegans_4d::endrov::{parse_lineages, read_ostxml};
use celegans_4d::import::{choose_lineage, PixelOrigin};
use clap::{Parser, ValueEnum};
use serde_json::json;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RefinerChoice {
    /// Endrov linear interpolation (the baseline).
    Linear,
    /// Linear, z rounded to the nearest stack plane (the truth's quantisation).
    LinearPlaneZ,
    /// The nearer remaining keyframe, held.
    Nearest,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OriginChoice {
    Corner,
    Centre,
}

#[derive(Debug, Parser)]
#[command(
    name = "holdout_benchmark",
    about = "Hold out Endrov keyframes, refine the interpolated position, score against the hand-placed one"
)]
struct Args {
    /// The image group the lineage annotates, e.g. `rec.ome.zarr/DIC`.
    #[arg(long)]
    image: PathBuf,
    /// Default: `<image>/../OST/rmd.ostxml`.
    #[arg(long)]
    ostxml: Option<PathBuf>,
    #[arg(long)]
    lineage: Option<String>,
    #[arg(long, value_enum, default_value_t = RefinerChoice::Linear)]
    refiner: RefinerChoice,
    /// Stride scenarios: keep every s-th keyframe of every cell. Empty for single-holdout only.
    #[arg(long, value_delimiter = ',', default_value = "2,3,4,6,8")]
    strides: Vec<usize>,
    /// Skip the single-keyframe holdout.
    #[arg(long)]
    no_single: bool,
    #[arg(long, value_enum, default_value_t = OriginChoice::Corner)]
    pixel_origin: OriginChoice,
    /// Image channel the refiner reads.
    #[arg(long, default_value_t = 0)]
    channel: usize,
    /// Output directory: keyframes.csv, summary.json, report.md.
    #[arg(long)]
    out: PathBuf,
}

fn main() -> ExitCode {
    match main_inner() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("holdout_benchmark: {error}");
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
    let origin = match args.pixel_origin {
        OriginChoice::Corner => PixelOrigin::Corner,
        OriginChoice::Centre => PixelOrigin::Centre,
    };
    let recording = Recording::open(&args.image, origin, args.channel)?;
    let lineage = choose_lineage(
        parse_lineages(&read_ostxml(&ostxml)?)?,
        args.lineage.as_deref(),
    )?;
    let refiner: Box<dyn Refiner> = match args.refiner {
        RefinerChoice::Linear => Box::new(Linear),
        RefinerChoice::LinearPlaneZ => Box::new(LinearPlaneZ),
        RefinerChoice::Nearest => Box::new(Nearest),
    };
    let mut scenarios = Vec::new();
    if !args.no_single {
        scenarios.push(Scenario::Single);
    }
    for &s in &args.strides {
        if s < 2 {
            return Err(Error::invalid(format!(
                "stride {s} removes nothing; use 2 or more"
            )));
        }
        scenarios.push(Scenario::Stride(s));
    }
    scenarios.sort();
    scenarios.dedup();
    let rows = evaluate(&lineage, &recording, refiner.as_ref(), &scenarios)?;
    let recording_name = args
        .image
        .parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().trim_end_matches(".ome.zarr").to_owned())
        .unwrap_or_default();
    let header = json!({
        "schema": "blockflow.celegans-4d.holdout/v1",
        "recording": recording_name,
        "image": args.image,
        "ostxml": ostxml,
        "lineage_id": lineage.id,
        "refiner": refiner.name(),
        "pixel_origin": format!("{:?}", args.pixel_origin).to_lowercase(),
        "spacing_zyx": recording.mapping.scale,
        "annotation": annotation_stats(&lineage, &recording),
        "semantics": {
            "interior": "not the first or last visible keyframe of a cell (visible: inside start=/end= and before the first child)",
            "single": "one interior keyframe removed at a time; the rest of the lineage intact",
            "strideN": "in every cell at once, interior keyframes at visible index j with j % N != 0 removed; first and last kept",
            "interpolation": "Endrov Lineage.interpolatePos via celegans_4d::endrov::Nucleus::interpolate",
            "errors": "µm in Endrov world coordinates; z = |dz|, r = |dr|; ratio = median 3D error / linear median on the same rows",
            "gap_frames": "(key_after - key_before) / frame interval",
            "cells_at_frame": "annotated (non-marker) cells present at the frame, full lineage",
        },
    });
    let document = write_outputs(&args.out, header, &rows)?;
    let brief: serde_json::Map<String, serde_json::Value> = document["scenarios"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(k, v)| {
            let o = &v["overall"];
            (
                k.clone(),
                json!({
                    "n": o["n"],
                    "d3_median": o["d3"]["median"], "d3_p90": o["d3"]["p90"],
                    "xy_median": o["xy"]["median"], "xy_p90": o["xy"]["p90"],
                    "z_median": o["z"]["median"], "z_p90": o["z"]["p90"],
                    "r_median": o["r"]["median"],
                }),
            )
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"out": args.out, "refiner": refiner.name(), "overall": brief})
        )
        .map_err(Error::backend)?
    );
    Ok(())
}
