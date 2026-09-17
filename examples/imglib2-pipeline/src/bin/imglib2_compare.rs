// SPDX-License-Identifier: MIT

use std::env;
use std::fs;
use std::path::PathBuf;

use blockflow::{Error, Result};
use serde_json::json;

#[derive(Debug)]
struct Config {
    blockflow: PathBuf,
    imglib2: PathBuf,
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
    let imglib2 = read_summary(&config.imglib2)?;
    let object_delta = blockflow.objects.abs_diff(imglib2.objects);
    let area_relative_error = relative_error(blockflow.total_area, imglib2.total_area);
    let passed = object_delta <= config.max_object_delta
        && area_relative_error <= config.max_area_relative_error;
    let report = json!({
        "passed": passed,
        "blockflow": {
            "objects": blockflow.objects,
            "total_foreground_area": blockflow.total_area,
        },
        "imglib2": {
            "objects": imglib2.objects,
            "total_foreground_area": imglib2.total_area,
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
        .map_err(|err| Error::invalid(format!("imglib2-compare: encode report: {err}")))?;
    fs::write(&config.out, text)
        .map_err(|err| Error::invalid(format!("imglib2-compare: write report: {err}")))?;
    if passed {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "imglib2-compare: comparison failed; wrote {}",
            config.out.display()
        )))
    }
}

impl Config {
    fn parse() -> Result<Self> {
        let mut blockflow = None;
        let mut imglib2 = None;
        let mut out = PathBuf::from("imglib2-comparison.json");
        let mut max_object_delta = 0;
        let mut max_area_relative_error = 0.01;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--blockflow" => blockflow = Some(path_arg(&mut args, "--blockflow")?),
                "--imglib2" => imglib2 = Some(path_arg(&mut args, "--imglib2")?),
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
                        "imglib2-compare: unknown argument {other:?}; use --help"
                    )));
                }
            }
        }

        Ok(Self {
            blockflow: blockflow
                .ok_or_else(|| Error::invalid("imglib2-compare: missing --blockflow"))?,
            imglib2: imglib2.ok_or_else(|| Error::invalid("imglib2-compare: missing --imglib2"))?,
            out,
            max_object_delta,
            max_area_relative_error,
        })
    }
}

fn path_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<PathBuf> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::invalid(format!("imglib2-compare: {name} needs a path")))
}

fn parse_arg<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let raw = args
        .next()
        .ok_or_else(|| Error::invalid(format!("imglib2-compare: {name} needs a value")))?;
    raw.parse::<T>().map_err(|err| {
        Error::invalid(format!(
            "imglib2-compare: could not parse {name} value {raw:?}: {err}"
        ))
    })
}

fn print_help() {
    println!("imglib2-compare --blockflow SUMMARY.json --imglib2 SUMMARY.json [--out REPORT.json]");
}

fn read_summary(path: &PathBuf) -> Result<Summary> {
    let text = fs::read_to_string(path).map_err(|err| {
        Error::invalid(format!("imglib2-compare: read {}: {err}", path.display()))
    })?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
        Error::invalid(format!(
            "imglib2-compare: parse {} as JSON: {err}",
            path.display()
        ))
    })?;
    Ok(Summary {
        objects: json
            .get("objects")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| Error::invalid("imglib2-compare: missing objects"))?,
        total_area: json
            .get("total_foreground_area")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| Error::invalid("imglib2-compare: missing total_foreground_area"))?,
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
