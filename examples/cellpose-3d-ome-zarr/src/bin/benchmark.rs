use std::path::PathBuf;
use std::time::Instant;

use cellpose::core::Device;
use cellpose::{io, CellposeModel, EvalParams, InferenceBackend};
use clap::{Parser, ValueEnum};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(about = "Benchmark direct Rust Cellpose3D inference on TIFF volumes")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, required = true)]
    image: Vec<PathBuf>,
    #[arg(long, value_enum, default_value_t = DeviceChoice::Cuda)]
    device: DeviceChoice,
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    #[arg(long, default_value_t = 1.98)]
    anisotropy: f32,
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 1)]
    runs: usize,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    mask_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DeviceChoice {
    Cpu,
    Cuda,
}

fn main() -> anyhow::Result<()> {
    anyhow::ensure!(!cfg!(debug_assertions), "benchmark requires --release");
    let args = Args::parse();
    anyhow::ensure!(
        args.runs > 0 && !args.image.is_empty(),
        "no measured inputs"
    );
    let device = match args.device {
        DeviceChoice::Cpu => Device::Cpu,
        DeviceChoice::Cuda => Device::Cuda(args.cuda_device),
    };
    let load_started = Instant::now();
    let model =
        CellposeModel::new_with_backend(&args.model, device, None, true, InferenceBackend::Candle)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let params = EvalParams {
        do_3d: true,
        z_axis: Some(0),
        anisotropy: Some(args.anisotropy),
        batch_size: args.batch_size,
        ..EvalParams::default()
    };
    let inputs = args
        .image
        .iter()
        .map(|path| Ok((path, io::imread(path)?)))
        .collect::<anyhow::Result<Vec<_>>>()?;

    for _ in 0..args.warmup {
        let _ = model.eval_3d(&inputs[0].1, &params)?;
    }

    let mut measurements = Vec::new();
    for run in 0..args.runs {
        for (path, image) in &inputs {
            let started = Instant::now();
            let result = model.eval_3d(image, &params)?;
            let seconds = started.elapsed().as_secs_f64();
            let cells = result.masks.iter().copied().max().unwrap_or(0);
            let foreground_voxels = result.masks.iter().filter(|&&label| label != 0).count();
            let mask_path = if let Some(directory) = &args.mask_dir {
                std::fs::create_dir_all(directory)?;
                let stem = path
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .unwrap_or("crop");
                let output = directory.join(format!("{stem}-run{run}-rust-masks.tif"));
                io::imsave(&output, &result.masks.mapv(|label| label as f32))?;
                Some(output)
            } else {
                None
            };
            measurements.push(json!({
                "image": path,
                "run": run,
                "seconds": seconds,
                "cells": cells,
                "foreground_voxels": foreground_voxels,
                "mask": mask_path,
            }));
        }
    }
    let report = json!({
        "implementation": "cellpose-rs",
        "scope": "eval_3d",
        "model": args.model,
        "device": format!("{:?}", args.device).to_lowercase(),
        "anisotropy": args.anisotropy,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "runs": args.runs,
        "model_load_seconds": load_seconds,
        "measurements": measurements,
    });
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
