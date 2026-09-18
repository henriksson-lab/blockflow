// SPDX-License-Identifier: MIT

use std::env;
use std::fs;
use std::path::PathBuf;

use blockflow::{Error, Result};
use serde_json::json;

#[derive(Debug)]
struct Config {
    blockflow: PathBuf,
    skimage: PathBuf,
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
    let skimage = read_summary(&config.skimage)?;
    let object_delta = blockflow.objects.abs_diff(skimage.objects);
    let area_relative_error = relative_error(blockflow.total_area, skimage.total_area);
    let passed = object_delta <= config.max_object_delta
        && area_relative_error <= config.max_area_relative_error;
    let report = json!({
        "passed": passed,
        "blockflow": {
            "objects": blockflow.objects,
            "total_foreground_area": blockflow.total_area,
        },
        "skimage": {
            "objects": skimage.objects,
            "total_foreground_area": skimage.total_area,
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
        .map_err(|err| Error::invalid(format!("skimage-compare: encode report: {err}")))?;
    fs::write(&config.out, text)
        .map_err(|err| Error::invalid(format!("skimage-compare: write report: {err}")))?;
    if passed {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "skimage-compare: comparison failed; wrote {}",
            config.out.display()
        )))
    }
}

impl Config {
    fn parse() -> Result<Self> {
        let mut blockflow = None;
        let mut skimage = None;
        let mut out = PathBuf::from("skimage-comparison.json");
        let mut max_object_delta = 0;
        let mut max_area_relative_error = 0.01;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--blockflow" => blockflow = Some(path_arg(&mut args, "--blockflow")?),
                "--skimage" => skimage = Some(path_arg(&mut args, "--skimage")?),
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
                        "skimage-compare: unknown argument {other:?}; use --help"
                    )));
                }
            }
        }

        Ok(Self {
            blockflow: blockflow
                .ok_or_else(|| Error::invalid("skimage-compare: missing --blockflow"))?,
            skimage: skimage.ok_or_else(|| Error::invalid("skimage-compare: missing --skimage"))?,
            out,
            max_object_delta,
            max_area_relative_error,
        })
    }
}

fn path_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<PathBuf> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::invalid(format!("skimage-compare: {name} needs a path")))
}

fn parse_arg<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let raw = args
        .next()
        .ok_or_else(|| Error::invalid(format!("skimage-compare: {name} needs a value")))?;
    raw.parse::<T>().map_err(|err| {
        Error::invalid(format!(
            "skimage-compare: could not parse {name} value {raw:?}: {err}"
        ))
    })
}

fn print_help() {
    println!("skimage-compare --blockflow SUMMARY.json --skimage SUMMARY.json [--out REPORT.json]");
}

fn read_summary(path: &PathBuf) -> Result<Summary> {
    let text = fs::read_to_string(path).map_err(|err| {
        Error::invalid(format!("skimage-compare: read {}: {err}", path.display()))
    })?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
        Error::invalid(format!(
            "skimage-compare: parse {} as JSON: {err}",
            path.display()
        ))
    })?;
    Ok(Summary {
        objects: json
            .get("objects")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| Error::invalid("skimage-compare: missing objects"))?,
        total_area: json
            .get("total_foreground_area")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| Error::invalid("skimage-compare: missing total_foreground_area"))?,
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
