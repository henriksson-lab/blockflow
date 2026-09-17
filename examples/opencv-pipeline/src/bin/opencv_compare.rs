// SPDX-License-Identifier: MIT

use std::env;
use std::fs;
use std::path::PathBuf;

use blockflow::{Error, Result};
use serde_json::json;

#[derive(Debug)]
struct Config {
    blockflow: PathBuf,
    opencv: PathBuf,
    out: PathBuf,
    max_object_delta: u64,
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
        let mut blockflow = None;
        let mut opencv = None;
        let mut out = PathBuf::from("opencv-comparison.json");
        let mut max_object_delta = 0;
        let mut max_area_relative_error = 0.01;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--blockflow" => blockflow = Some(path_arg(&mut args, "--blockflow")?),
                "--opencv" => opencv = Some(path_arg(&mut args, "--opencv")?),
                "--out" => out = path_arg(&mut args, "--out")?,
                "--max-object-delta" => {
                    max_object_delta = parse_arg(&mut args, "--max-object-delta")?
                }
                "--max-area-relative-error" => {
                    max_area_relative_error = parse_arg(&mut args, "--max-area-relative-error")?
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(Error::invalid(format!(
                        "opencv-compare: unknown argument {other:?}; use --help"
                    )));
                }
            }
        }

        Ok(Self {
            blockflow: blockflow
                .ok_or_else(|| Error::invalid("opencv-compare: missing --blockflow"))?,
            opencv: opencv.ok_or_else(|| Error::invalid("opencv-compare: missing --opencv"))?,
            out,
            max_object_delta,
            max_area_relative_error,
        })
    }
}

fn path_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<PathBuf> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::invalid(format!("opencv-compare: {name} needs a path")))
}

fn parse_arg<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let raw = args
        .next()
        .ok_or_else(|| Error::invalid(format!("opencv-compare: {name} needs a value")))?;
    raw.parse::<T>().map_err(|err| {
        Error::invalid(format!(
            "opencv-compare: could not parse {name} value {raw:?}: {err}"
        ))
    })
}

fn print_help() {
    println!("opencv-compare --blockflow SUMMARY.json --opencv SUMMARY.json [--out REPORT.json]");
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
