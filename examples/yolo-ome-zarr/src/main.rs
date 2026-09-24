use std::path::PathBuf;

use blockflow::yolo::{self, PredictConfig};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "yolo-ome-zarr",
    about = "Run YOLOv11 over an OME-Zarr image with Blockflow"
)]
struct Args {
    #[arg(long)]
    zarr: PathBuf,
    #[arg(long, default_value_t = 0)]
    level: usize,
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    config: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "0,1,2")]
    channels: Vec<usize>,
    /// Optional y,x,height,width window within the selected pyramid level.
    #[arg(long, value_delimiter = ',')]
    region: Option<Vec<usize>>,
    #[arg(long, default_value_t = 0.0)]
    normalize_low: f64,
    #[arg(long, default_value_t = 255.0)]
    normalize_high: f64,
    #[arg(long, default_value_t = 640)]
    block: usize,
    #[arg(long, default_value_t = 0)]
    halo: usize,
    #[arg(long, default_value_t = 0.25)]
    conf_threshold: f32,
    #[arg(long, default_value_t = 640)]
    input_size: u32,
    #[arg(long, default_value_t = 1)]
    workers: usize,
    #[arg(long, default_value_t = 0.0)]
    min_separation: f64,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    summary: PathBuf,
    #[arg(long)]
    work: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let region = args
        .region
        .map(|values| {
            <[usize; 4]>::try_from(values).map_err(|values| {
                anyhow::anyhow!("--region wants y,x,height,width; got {values:?}")
            })
        })
        .transpose()?;
    yolo::run(&PredictConfig {
        zarr: args.zarr,
        level: args.level,
        weights: args.weights,
        config: args.config,
        channels: args.channels,
        region,
        normalize_range: (args.normalize_low, args.normalize_high),
        block: args.block,
        halo: args.halo,
        conf_threshold: args.conf_threshold,
        input_size: args.input_size,
        concurrency: args.workers,
        min_separation: args.min_separation,
        out: args.out,
        summary: args.summary,
        work: args.work,
    })
}
