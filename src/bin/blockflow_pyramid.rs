// SPDX-License-Identifier: MIT
//
// Add an intensity pyramid to an existing OME-Zarr image, in place.
//
//     blockflow-pyramid <image.zarr> [--method mean|max] [--target-size 256]
//         [--max-levels N] [--chunk z,y,x] [--shard z,y,x|none]
//         [--compression level0|none|gzipN|zstdN|blosc-<cname>N[-shuffle]]
//         [--workers N] [--memory-gb G] [--no-translation] [--output DIR]
//         [--overwrite] [--dry-run] [--json-progress]
//
// Existing levels are never modified; new levels are written as the next free
// numeric paths and the group's `multiscales[0].datasets` is extended last.
// See `blockflow::intensity_pyramid` for the factor rule, the translation
// convention and the memory bound.
//
// Progress goes to stderr, one line per wave; with `--json-progress` each line
// is a JSON object (`{"event":"progress",...}`) for a job runner to parse.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use blockflow::intensity_pyramid::{
    build_pyramid, human_bytes, parse_compression, plan_pyramid, Downsample, PyramidOptions,
    PyramidProgress, ShardChoice,
};

const USAGE: &str = "usage: blockflow-pyramid <image.zarr> [--method mean|max] [--target-size 256] \
[--max-levels N] [--chunk z,y,x] [--shard z,y,x|none] [--compression level0|none|gzipN|zstdN|blosc-zstd5] \
[--workers N] [--memory-gb G] [--no-translation] [--output DIR] [--overwrite] [--dry-run] [--json-progress]";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("blockflow-pyramid: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut image: Option<PathBuf> = None;
    let mut options = PyramidOptions::default();
    let mut dry_run = false;
    let mut json_progress = false;
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--method" => {
                options.method = match value("--method")?.as_str() {
                    "mean" => Downsample::Mean,
                    "max" => Downsample::Max,
                    other => return Err(format!("unknown method {other}")),
                }
            }
            "--target-size" => options.target_size = number(&value("--target-size")?)?,
            "--max-levels" => options.max_levels = Some(number(&value("--max-levels")?)?),
            "--chunk" => options.chunk = Some(shape(&value("--chunk")?)?),
            "--shard" => {
                let text = value("--shard")?;
                options.shard = Some(if text == "none" {
                    ShardChoice::None
                } else {
                    ShardChoice::Shape(shape(&text)?)
                });
            }
            "--compression" => {
                let text = value("--compression")?;
                options.compression = if text == "level0" {
                    None
                } else {
                    Some(parse_compression(&text).map_err(|e| e.to_string())?)
                };
            }
            "--workers" => options.workers = number(&value("--workers")?)?,
            "--memory-gb" => {
                let gb: f64 = value("--memory-gb")?
                    .parse()
                    .map_err(|_| "--memory-gb takes a number".to_string())?;
                options.memory_budget = (gb * (1u64 << 30) as f64) as u64;
            }
            "--no-translation" => options.half_voxel_translation = false,
            "--output" => options.output = Some(PathBuf::from(value("--output")?)),
            "--overwrite" => options.overwrite = true,
            "--dry-run" => dry_run = true,
            "--json-progress" => json_progress = true,
            other if other.starts_with('-') => {
                return Err(format!("unknown option {other}\n{USAGE}"))
            }
            other => {
                if image.replace(PathBuf::from(other)).is_some() {
                    return Err(format!("one image path only\n{USAGE}"));
                }
            }
        }
    }
    let image = image.ok_or_else(|| USAGE.to_string())?;

    let plan = plan_pyramid(&image, &options).map_err(|e| e.to_string())?;
    print!("{}", plan.describe());
    if dry_run {
        return Ok(());
    }
    if plan.new_levels.is_empty() {
        return Ok(());
    }

    let started = Instant::now();
    let report = |progress: &PyramidProgress| {
        if json_progress {
            eprintln!(
                "{}",
                serde_json::json!({
                    "event": "progress",
                    "level": progress.level + 1,
                    "levels": progress.levels,
                    "path": progress.path,
                    "units_done": progress.units_done,
                    "units_total": progress.units_total,
                    "elapsed_s": started.elapsed().as_secs_f64(),
                })
            );
        } else {
            eprintln!(
                "level {}/{} (path {}): {}/{} units, {:.1} s",
                progress.level + 1,
                progress.levels,
                progress.path,
                progress.units_done,
                progress.units_total,
                started.elapsed().as_secs_f64()
            );
        }
    };
    let written = build_pyramid(&plan, &options, &report).map_err(|e| e.to_string())?;
    let total: u64 = plan.new_levels.iter().map(|l| plan.level_bytes(l)).sum();
    println!(
        "wrote levels {:?} ({} uncompressed) in {:.1} s; metadata updated",
        written,
        human_bytes(total),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn number(text: &str) -> Result<usize, String> {
    text.parse()
        .map_err(|_| format!("expected a number, got {text}"))
}

fn shape(text: &str) -> Result<Vec<usize>, String> {
    text.split(',').map(|part| number(part.trim())).collect()
}
