// SPDX-License-Identifier: MIT

use std::path::PathBuf;

use blockflow::collection::SeriesKey;
use blockflow::{Error, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use plate_timelapse_tracking::params::{Normalization, Params, Segmenter, Stage};
use plate_timelapse_tracking::run::{run, Input};
use plate_timelapse_tracking::score::score;
use plate_timelapse_tracking::synth::{read_truth, write_plate, SynthSpec};

#[derive(Debug, Parser)]
#[command(
    name = "plate-timelapse-tracking",
    about = "Segment, measure and track cells in multiwell-plate time-lapse OME-Zarr data"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Segment every frame, measure, link into tracks, and publish tables.
    Run(RunArgs),
    /// Write a synthetic HCS plate with ground truth.
    Synth(SynthArgs),
    /// Score one series' detections.csv against synthetic truth.
    Score {
        #[arg(long)]
        detections: PathBuf,
        #[arg(long)]
        truth: PathBuf,
        /// Match radius in pixels.
        #[arg(long, default_value_t = 4.0)]
        radius: f64,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SegmenterChoice {
    Threshold,
    Stardist,
    Cellpose,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum NormalizeChoice {
    None,
    Fixed,
    Movie,
    Frame,
}

#[derive(Debug, Args)]
struct RunArgs {
    /// An OME-NGFF HCS plate.
    #[arg(long, conflicts_with_all = ["manifest", "image"])]
    plate: Option<PathBuf>,
    /// A collection manifest (JSON) for separately stored images.
    #[arg(long, conflicts_with_all = ["plate", "image"])]
    manifest: Option<PathBuf>,
    /// One field image (a multiscale image group with a t axis).
    #[arg(long, conflicts_with_all = ["plate", "manifest"])]
    image: Option<PathBuf>,
    /// Well and field naming for --image outputs.
    #[arg(long, default_value = "A")]
    well_row: String,
    #[arg(long, default_value = "1")]
    well_column: String,
    #[arg(long, default_value = "0")]
    field: String,

    #[arg(long, value_enum, default_value_t = SegmenterChoice::Threshold)]
    segmenter: SegmenterChoice,
    /// Threshold segmenter: level, on normalised values if normalising.
    #[arg(long, default_value_t = 100.0)]
    threshold: f32,
    /// StarDist model directory, or Cellpose checkpoint.
    #[arg(long)]
    model: Option<PathBuf>,
    /// StarDist Keras weights (default MODEL/weights_best.h5).
    #[arg(long)]
    weights: Option<PathBuf>,
    #[arg(long)]
    prob_threshold: Option<f32>,
    #[arg(long)]
    nms_threshold: Option<f32>,
    /// Cellpose: CUDA device ordinal (CPU when absent).
    #[arg(long)]
    cuda_device: Option<usize>,
    #[arg(long)]
    diameter: Option<f32>,
    #[arg(long, default_value_t = 0.4)]
    flow_threshold: f32,
    #[arg(long, default_value_t = 0.0)]
    cellprob_threshold: f32,
    #[arg(long, default_value_t = 15)]
    min_size: i32,

    #[arg(long, value_enum, default_value_t = NormalizeChoice::None)]
    normalize: NormalizeChoice,
    /// Fixed normalisation range.
    #[arg(long)]
    low: Option<f32>,
    #[arg(long)]
    high: Option<f32>,
    /// Percentile normalisation (movie or frame).
    #[arg(long, default_value_t = 1.0)]
    low_percentile: f64,
    #[arg(long, default_value_t = 99.8)]
    high_percentile: f64,

    #[arg(long, default_value_t = 0)]
    segment_channel: usize,
    #[arg(long, default_value_t = 1)]
    measure_channel: usize,
    #[arg(long, default_value = "cells")]
    layer: String,
    /// Segmentation halo in pixels; at least the largest cell diameter.
    #[arg(long, default_value_t = 64)]
    halo: usize,
    #[arg(long, value_delimiter = ',', default_value = "512,1024,2048")]
    blocks: Vec<usize>,
    #[arg(long, default_value_t = 256)]
    label_chunk: usize,
    /// Executor threads inside one frame plan.
    #[arg(long, default_value_t = 1)]
    block_workers: usize,
    /// Frames processed concurrently across the whole collection.
    #[arg(long, default_value_t = 4)]
    frame_jobs: usize,
    /// Series finalised concurrently.
    #[arg(long, default_value_t = 2)]
    series_jobs: usize,

    /// Largest displacement between consecutive frames, in physical units.
    #[arg(long)]
    max_distance: Option<f64>,
    /// Largest speed, physical units per second; needs timestamps.
    #[arg(long)]
    max_speed: Option<f64>,
    /// Weight of (1 - IoU) in the link cost (0 = centroid only).
    #[arg(long, default_value_t = 0.0)]
    iou_weight: f64,
    #[arg(long, default_value_t = 0.0)]
    size_weight: f64,
    /// Fixed birth/death cost; default 1.05 x the largest link cost per frame pair.
    #[arg(long)]
    alternative_cost: Option<f64>,
    /// Segment linking: close gaps of up to --max-gap frames within this
    /// physical distance.
    #[arg(long)]
    gap_closing: Option<f64>,
    #[arg(long, default_value_t = 2)]
    max_gap: u64,
    /// Segment linking: allow a new track to split from a detection one frame
    /// earlier within this distance (divisions, segmentation splits).
    #[arg(long)]
    splitting: Option<f64>,
    /// Segment linking: allow an ending track to merge into a detection one
    /// frame later within this distance (fragments rejoining).
    #[arg(long)]
    merging: Option<f64>,
    /// Fixed alternative cost of the segment step (default 1.05 x the 90th
    /// percentile of its candidate costs).
    #[arg(long)]
    segment_alternative_cost: Option<f64>,
    /// Physical distance from the frame edge that counts as near the border.
    #[arg(long, default_value_t = 10.0)]
    border_distance: f64,
    /// Correct stage drift between frames (phase correlation of the
    /// segmentation channel) before gating links.
    #[arg(long, default_value_t = false)]
    drift_correction: bool,
    /// Pyramid level for drift estimation (default: finest level <= 1536 px).
    #[arg(long)]
    drift_level: Option<usize>,
    /// Also write labels/<layer>-tracks, pixels relabelled by track id.
    #[arg(long, default_value_t = false)]
    track_labels: bool,

    #[arg(long, default_value_t = false)]
    overwrite: bool,
    /// Re-hash completed frames' label planes when resuming.
    #[arg(long, default_value_t = false)]
    verify_planes: bool,
    /// Stop after the frame stage, keeping the work directory.
    #[arg(long, hide = true)]
    stop_after_frames: bool,
}

#[derive(Debug, Args)]
struct SynthArgs {
    #[arg(long)]
    out: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "A")]
    rows: Vec<String>,
    #[arg(long, value_delimiter = ',', default_value = "1,2")]
    columns: Vec<String>,
    #[arg(long, default_value_t = 1)]
    fields: usize,
    #[arg(long, default_value_t = 8)]
    frames: usize,
    #[arg(long, default_value_t = 160)]
    size: usize,
    #[arg(long, default_value_t = 12)]
    cells: usize,
    /// Largest per-axis step per frame, pixels.
    #[arg(long, default_value_t = 2.0)]
    step: f64,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Per-cell, per-frame division probability.
    #[arg(long, default_value_t = 0.0)]
    divide: f64,
    /// Global stage drift per frame, pixels, as dy,dx.
    #[arg(long, value_delimiter = ',', default_value = "0,0")]
    drift: Vec<f64>,
}

fn main() {
    if let Err(error) = dispatch() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn dispatch() -> Result<()> {
    match Cli::parse().command {
        Command::Run(args) => run_command(args),
        Command::Synth(args) => {
            let spec = SynthSpec {
                rows: args.rows,
                columns: args.columns,
                fields: args.fields,
                frames: args.frames,
                size: [args.size, args.size],
                cells: args.cells,
                step: args.step,
                divide: args.divide,
                seed: args.seed,
                drift: [
                    args.drift.first().copied().unwrap_or(0.0),
                    args.drift.get(1).copied().unwrap_or(0.0),
                ],
                ..SynthSpec::default()
            };
            let truth = write_plate(&args.out, &spec)?;
            println!("plate={} fields={}", args.out.display(), truth.len());
            Ok(())
        }
        Command::Score {
            detections,
            truth,
            radius,
        } => {
            let truth = read_truth(&truth)?;
            let report = score(&detections, &truth, radius)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report).map_err(Error::backend)?
            );
            Ok(())
        }
    }
}

fn run_command(args: RunArgs) -> Result<()> {
    let input = match (&args.plate, &args.manifest, &args.image) {
        (Some(plate), None, None) => Input::Plate(plate.clone()),
        (None, Some(manifest), None) => Input::Manifest(manifest.clone()),
        (None, None, Some(image)) => Input::Image {
            path: image.clone(),
            key: SeriesKey {
                plate_id: "single".into(),
                acquisition_id: None,
                well_row: args.well_row.clone(),
                well_column: args.well_column.clone(),
                field_id: args.field.clone(),
            },
        },
        _ => {
            return Err(Error::invalid(
                "give exactly one of --plate, --manifest, --image",
            ))
        }
    };
    let model = || {
        args.model
            .clone()
            .ok_or_else(|| Error::invalid("--model is required for this segmenter"))
    };
    let segmenter = match args.segmenter {
        SegmenterChoice::Threshold => Segmenter::Threshold {
            threshold: args.threshold,
        },
        SegmenterChoice::Stardist => Segmenter::Stardist {
            model: model()?,
            weights: args.weights.clone(),
            prob_threshold: args.prob_threshold,
            nms_threshold: args.nms_threshold,
        },
        SegmenterChoice::Cellpose => Segmenter::Cellpose {
            model: model()?,
            cuda_device: args.cuda_device,
            diameter: args.diameter,
            flow_threshold: args.flow_threshold,
            cellprob_threshold: args.cellprob_threshold,
            min_size: args.min_size,
        },
    };
    let normalization = match args.normalize {
        NormalizeChoice::None => Normalization::None,
        NormalizeChoice::Fixed => Normalization::Fixed {
            low: args
                .low
                .ok_or_else(|| Error::invalid("--normalize fixed needs --low"))?,
            high: args
                .high
                .ok_or_else(|| Error::invalid("--normalize fixed needs --high"))?,
        },
        NormalizeChoice::Movie => Normalization::MoviePercentile {
            low: args.low_percentile,
            high: args.high_percentile,
        },
        NormalizeChoice::Frame => Normalization::FramePercentile {
            low: args.low_percentile,
            high: args.high_percentile,
        },
    };
    let params = Params {
        segmenter,
        normalization,
        segment_channel: args.segment_channel,
        measure_channel: args.measure_channel,
        layer: args.layer,
        halo: args.halo,
        blocks: args.blocks,
        label_chunk: args.label_chunk,
        block_workers: args.block_workers,
        frame_jobs: args.frame_jobs,
        series_jobs: args.series_jobs,
        max_distance: args.max_distance,
        max_speed: args.max_speed,
        iou_weight: args.iou_weight,
        size_weight: args.size_weight,
        alternative_cost: args.alternative_cost,
        gap_closing: args.gap_closing,
        max_gap: args.max_gap,
        splitting: args.splitting,
        merging: args.merging,
        segment_alternative_cost: args.segment_alternative_cost,
        border_distance: args.border_distance,
        drift_correction: args.drift_correction,
        drift_level: args.drift_level,
        track_labels: args.track_labels,
        overwrite: args.overwrite,
        verify_planes: args.verify_planes,
        // The command publishes labels, tables and tracking metadata in each image group.
        // Separate overlay roots remain an internal regression-test facility.
        output: None,
        stop_after: args.stop_after_frames.then_some(Stage::Frames),
    };
    let report = run(&input, &params)?;
    for status in &report.series {
        let key = &status.key;
        match &status.error {
            None => println!(
                "{}{} field {}: frames={} detections={} tracks={}",
                key.well_row,
                key.well_column,
                key.field_id,
                status.frames,
                status.detections,
                status.tracks
            ),
            Some(error) => println!(
                "{}{} field {}: FAILED {error}",
                key.well_row, key.well_column, key.field_id
            ),
        }
    }
    println!(
        "frames ran={} skipped={}{}",
        report.frames_ran,
        report.frames_skipped,
        report
            .index
            .as_ref()
            .map(|p| format!(" index={}", p.display()))
            .unwrap_or_default()
    );
    if report.failed() > 0 {
        return Err(Error::invalid(format!("{} series failed", report.failed())));
    }
    Ok(())
}
