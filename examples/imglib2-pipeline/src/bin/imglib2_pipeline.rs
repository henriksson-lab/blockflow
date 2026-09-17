// SPDX-License-Identifier: MIT

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use blockflow::ops::components;
use blockflow::ops::{
    gaussian_smooth_into_with, remove_small_objects_into, Boundary, Connectivity, Gaussian,
};
use blockflow::{Error, Result};
use ndarray::Array3;

#[derive(Debug)]
struct Config {
    input: PathBuf,
    out: PathBuf,
    sigma: f64,
    min_size: u64,
}

#[derive(Debug)]
struct ObjectRow {
    label: u32,
    count: u64,
    centroid_y: f64,
    centroid_x: f64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = Config::parse()?;
    fs::create_dir_all(&config.out)
        .map_err(|err| Error::invalid(format!("imglib2-pipeline: create output dir: {err}")))?;

    let started = Instant::now();
    let input = load_luma(&config.input)?;
    let load_seconds = started.elapsed().as_secs_f64();

    let pipeline_started = Instant::now();
    let smoothed = smooth(input.view(), config.sigma)?;
    let threshold = otsu_threshold(smoothed.iter().copied());
    let raw_mask = smoothed.mapv(|value| value > threshold);
    let mut mask = Array3::<bool>::from_elem(raw_mask.raw_dim(), false);
    remove_small_objects_into(
        raw_mask.view(),
        Connectivity::Faces,
        config.min_size,
        mask.view_mut(),
    )?;
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    components::label_members_into_with(
        [mask.shape()[0], mask.shape()[1], mask.shape()[2]],
        Connectivity::Faces,
        |at| mask[[at[0], at[1], at[2]]],
        labels.view_mut(),
    )?;
    let rows = measure(labels.view());
    let pipeline_seconds = pipeline_started.elapsed().as_secs_f64();

    write_objects(&rows, &config.out.join("objects.csv"))?;
    write_summary(
        &config,
        rows.len(),
        rows.iter().map(|row| row.count).sum(),
        threshold,
        load_seconds,
        pipeline_seconds,
        &config.out.join("summary.json"),
    )?;

    println!(
        "objects={} threshold={threshold:.6} output={}",
        rows.len(),
        config.out.display()
    );
    Ok(())
}

impl Config {
    fn parse() -> Result<Self> {
        let mut input = None;
        let mut out = None;
        let mut sigma = 1.5f64;
        let mut min_size = 20;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => input = Some(path_arg(&mut args, "--input")?),
                "--out" => out = Some(path_arg(&mut args, "--out")?),
                "--sigma" => sigma = parse_arg(&mut args, "--sigma")?,
                "--min-size" => min_size = parse_arg(&mut args, "--min-size")?,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(Error::invalid(format!(
                        "imglib2-pipeline: unknown argument {other:?}; use --help"
                    )));
                }
            }
        }

        if sigma < 0.0 || !sigma.is_finite() {
            return Err(Error::invalid(
                "imglib2-pipeline: --sigma must be finite and non-negative",
            ));
        }
        if min_size == 0 {
            return Err(Error::invalid(
                "imglib2-pipeline: --min-size must be at least 1",
            ));
        }

        Ok(Self {
            input: input
                .ok_or_else(|| Error::invalid("imglib2-pipeline: missing required --input"))?,
            out: out.unwrap_or_else(|| PathBuf::from(".tmp/imglib2-pipeline/blockflow")),
            sigma,
            min_size,
        })
    }
}

fn path_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<PathBuf> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::invalid(format!("imglib2-pipeline: {name} needs a path")))
}

fn parse_arg<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let raw = args
        .next()
        .ok_or_else(|| Error::invalid(format!("imglib2-pipeline: {name} needs a value")))?;
    raw.parse::<T>().map_err(|err| {
        Error::invalid(format!(
            "imglib2-pipeline: could not parse {name} value {raw:?}: {err}"
        ))
    })
}

fn print_help() {
    println!(
        "imglib2-pipeline --input IMAGE --out DIR [--sigma 1.5] [--min-size 20]\n\
         Runs the Blockflow side of the ImgLib2 comparison benchmark."
    );
}

fn load_luma(path: &Path) -> Result<Array3<f64>> {
    let image = image::ImageReader::open(path)
        .map_err(|err| Error::invalid(format!("imglib2-pipeline: open image: {err}")))?
        .decode()
        .map_err(|err| Error::invalid(format!("imglib2-pipeline: decode image: {err}")))?
        .to_luma8();
    let (width, height) = image.dimensions();
    let width = usize::try_from(width)
        .map_err(|_| Error::invalid("imglib2-pipeline: image width does not fit usize"))?;
    let height = usize::try_from(height)
        .map_err(|_| Error::invalid("imglib2-pipeline: image height does not fit usize"))?;
    let mut out = Array3::<f64>::zeros((1, height, width));
    for (x, y, pixel) in image.enumerate_pixels() {
        out[[0, y as usize, x as usize]] = f64::from(pixel.0[0]);
    }
    Ok(out)
}

fn smooth(input: ndarray::ArrayView3<'_, f64>, sigma: f64) -> Result<Array3<f64>> {
    let gaussian = Gaussian::new([0.0, sigma, sigma], 3.0)?;
    let mut out = Array3::<f64>::zeros(input.raw_dim());
    gaussian_smooth_into_with(input, gaussian.kernels(), Boundary::Reflect, out.view_mut())?;
    Ok(out)
}

fn otsu_threshold(values: impl Iterator<Item = f64>) -> f64 {
    let mut hist = [0u64; 256];
    let mut total = 0u64;
    for value in values {
        if value.is_finite() {
            let bin = value.round().clamp(0.0, 255.0) as usize;
            hist[bin] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    let sum_total: f64 = hist
        .iter()
        .enumerate()
        .map(|(bin, &count)| bin as f64 * count as f64)
        .sum();
    let mut weight_background = 0u64;
    let mut sum_background = 0.0;
    let mut best_bin = 0usize;
    let mut best_variance = f64::NEG_INFINITY;
    for (bin, &count) in hist.iter().enumerate() {
        weight_background += count;
        if weight_background == 0 {
            continue;
        }
        let weight_foreground = total - weight_background;
        if weight_foreground == 0 {
            break;
        }
        sum_background += bin as f64 * count as f64;
        let mean_background = sum_background / weight_background as f64;
        let mean_foreground = (sum_total - sum_background) / weight_foreground as f64;
        let variance = weight_background as f64
            * weight_foreground as f64
            * (mean_background - mean_foreground).powi(2);
        if variance > best_variance {
            best_variance = variance;
            best_bin = bin;
        }
    }
    best_bin as f64
}

fn measure(labels: ndarray::ArrayView3<'_, u32>) -> Vec<ObjectRow> {
    let mut tallies = BTreeMap::<u32, (u64, u64, u64)>::new();
    for ((_, y, x), &label) in labels.indexed_iter() {
        if label == 0 {
            continue;
        }
        let tally = tallies.entry(label).or_insert((0, 0, 0));
        tally.0 += 1;
        tally.1 += y as u64;
        tally.2 += x as u64;
    }
    tallies
        .into_iter()
        .map(|(label, (count, sum_y, sum_x))| ObjectRow {
            label,
            count,
            centroid_y: sum_y as f64 / count as f64,
            centroid_x: sum_x as f64 / count as f64,
        })
        .collect()
}

fn write_objects(rows: &[ObjectRow], path: &Path) -> Result<()> {
    let file = File::create(path)
        .map_err(|err| Error::invalid(format!("imglib2-pipeline: create CSV: {err}")))?;
    let mut out = BufWriter::new(file);
    writeln!(out, "label,count,centroid_y,centroid_x").map_err(write_error)?;
    for row in rows {
        writeln!(
            out,
            "{},{},{},{}",
            row.label, row.count, row.centroid_y, row.centroid_x
        )
        .map_err(write_error)?;
    }
    out.flush().map_err(write_error)
}

fn write_summary(
    config: &Config,
    objects: usize,
    total_area: u64,
    threshold: f64,
    load_seconds: f64,
    pipeline_seconds: f64,
    path: &Path,
) -> Result<()> {
    let summary = serde_json::json!({
        "input": config.input,
        "objects": objects,
        "total_foreground_area": total_area,
        "threshold": threshold,
        "sigma": config.sigma,
        "min_size": config.min_size,
        "load_seconds": load_seconds,
        "pipeline_seconds": pipeline_seconds,
    });
    let text = serde_json::to_string_pretty(&summary)
        .map_err(|err| Error::invalid(format!("imglib2-pipeline: encode summary: {err}")))?;
    fs::write(path, text)
        .map_err(|err| Error::invalid(format!("imglib2-pipeline: write summary: {err}")))
}

fn write_error(err: std::io::Error) -> Error {
    Error::invalid(format!("imglib2-pipeline: write output: {err}"))
}
