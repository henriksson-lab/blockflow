// SPDX-License-Identifier: MIT

use std::fs;
use std::path::PathBuf;

use blockflow::{Error, Result};
use clap::Parser;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "opencv-compare")]
struct Config {
    #[arg(long)]
    blockflow: PathBuf,
    #[arg(long)]
    opencv: PathBuf,
    #[arg(long, default_value = "opencv-comparison.json")]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    max_object_delta: u64,
    #[arg(long, default_value_t = 0.01)]
    max_area_relative_error: f64,
}

#[derive(Debug)]
struct Summary {
    objects: u64,
    total_area: u64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = Config::parse()?;
    let blockflow = read_summary(&config.blockflow)?;
    let opencv = read_summary(&config.opencv)?;
    let object_delta = blockflow.objects.abs_diff(opencv.objects);
    let area_relative_error = relative_error(blockflow.total_area, opencv.total_area);
    let passed = object_delta <= config.max_object_delta
        && area_relative_error <= config.max_area_relative_error;
    let report = json!({
        "passed": passed,
        "blockflow": {
            "objects": blockflow.objects,
            "total_foreground_area": blockflow.total_area,
        },
        "opencv": {
            "objects": opencv.objects,
            "total_foreground_area": opencv.total_area,
        },
        "metrics": {
            "object_delta": object_delta,
            "area_relative_error": area_relative_error,
        },
        "limits": {
            "max_object_delta": config.max_object_delta,
            "max_area_relative_error": config.max_area_relative_error,
        }
    });
    let text = serde_json::to_string_pretty(&report)
        .map_err(|err| Error::invalid(format!("opencv-compare: encode report: {err}")))?;
    fs::write(&config.out, text)
        .map_err(|err| Error::invalid(format!("opencv-compare: write report: {err}")))?;
    if passed {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "opencv-compare: comparison failed; wrote {}",
            config.out.display()
        )))
    }
}

impl Config {
    fn parse() -> Result<Self> {
        let config = <Self as Parser>::parse();
        if !config.max_area_relative_error.is_finite() {
            return Err(Error::invalid(
                "opencv-compare: --max-area-relative-error must be finite",
            ));
        }

        Ok(config)
    }
}

fn read_summary(path: &PathBuf) -> Result<Summary> {
    let text = fs::read_to_string(path)
        .map_err(|err| Error::invalid(format!("opencv-compare: read {}: {err}", path.display())))?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
        Error::invalid(format!(
            "opencv-compare: parse {} as JSON: {err}",
            path.display()
        ))
    })?;
    Ok(Summary {
        objects: json
            .get("objects")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| Error::invalid("opencv-compare: missing objects"))?,
        total_area: json
            .get("total_foreground_area")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| Error::invalid("opencv-compare: missing total_foreground_area"))?,
    })
}

fn relative_error(a: u64, b: u64) -> f64 {
    let denom = a.max(b);
    if denom == 0 {
        0.0
    } else {
        a.abs_diff(b) as f64 / denom as f64
    }
}
