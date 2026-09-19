// SPDX-License-Identifier: MIT

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use blockflow::ops::IntensityMeasurements;
use blockflow::{Error, Result};
use ndarray::Array3;
use serde_json::json;

const HEIGHT: usize = 96;
const WIDTH: usize = 128;
const OBJECTS_PER_IMAGE: usize = 6;

#[derive(Debug)]
struct Config {
    out: PathBuf,
    images: usize,
    threshold: f64,
}

#[derive(Debug)]
struct ObjectRow {
    image: usize,
    label: u64,
    count: u64,
    centroid_y: f64,
    centroid_x: f64,
    intensity: IntensityMeasurements,
    positive: bool,
}

#[derive(Debug, Default)]
struct IntensityTally {
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
    weighted_position: [f64; 3],
    position: [u64; 3],
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = Config::parse()?;
    fs::create_dir_all(&config.out).map_err(|err| {
        Error::invalid(format!(
            "percent-positive: create output directory {}: {err}",
            config.out.display()
        ))
    })?;

    let mut rows = Vec::new();
    for image in 0..config.images {
        let (labels, marker) = synthetic_fixture(image);
        rows.extend(measure_image(
            image,
            labels.view(),
            marker.view(),
            config.threshold,
        )?);
    }
    rows.sort_by_key(|row| (row.image, row.label));

    write_objects(&rows, &config.out.join("objects.csv"))?;
    write_summary(&config, &rows, &config.out.join("summary.json"))?;

    let positive = rows.iter().filter(|row| row.positive).count();
    println!(
        "images={} objects={} positive={} output={}",
        config.images,
        rows.len(),
        positive,
        config.out.display()
    );
    Ok(())
}

impl Config {
    fn parse() -> Result<Self> {
        let mut out = PathBuf::from(".tmp/percent-positive/blockflow");
        let mut images = 10usize;
        let mut threshold = 110.0f64;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out" => out = path_arg(&mut args, "--out")?,
                "--images" => images = parse_arg(&mut args, "--images")?,
                "--threshold" => threshold = parse_arg(&mut args, "--threshold")?,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(Error::invalid(format!(
                        "percent-positive: unknown argument {other:?}; use --help"
                    )));
                }
            }
        }

        if images == 0 {
            return Err(Error::invalid(
                "percent-positive: --images must be at least 1",
            ));
        }
        if !threshold.is_finite() {
            return Err(Error::invalid(
                "percent-positive: --threshold must be finite",
            ));
        }

        Ok(Self {
            out,
            images,
            threshold,
        })
    }
}

fn path_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<PathBuf> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::invalid(format!("percent-positive: {name} needs a path")))
}

fn parse_arg<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let raw = args
        .next()
        .ok_or_else(|| Error::invalid(format!("percent-positive: {name} needs a value")))?;
    raw.parse::<T>().map_err(|err| {
        Error::invalid(format!(
            "percent-positive: could not parse {name} value {raw:?}: {err}"
        ))
    })
}

fn print_help() {
    println!(
        "percent-positive --out DIR [--images 10] [--threshold 110]\n\
         Generates deterministic labelled-cell fixtures and classifies objects by mean marker intensity."
    );
}

fn synthetic_fixture(image: usize) -> (Array3<u32>, Array3<f64>) {
    let mut labels = Array3::<u32>::zeros((1, HEIGHT, WIDTH));
    let mut marker = Array3::<f64>::zeros((1, HEIGHT, WIDTH));
    for local in 0..OBJECTS_PER_IMAGE {
        let label = (image * 100 + local + 1) as u32;
        let (y0, x0, height, width) = object_rect(image, local);
        for y in y0..y0 + height {
            for x in x0..x0 + width {
                labels[[0, y, x]] = label;
                marker[[0, y, x]] = pixel_intensity(image, local, y, x);
            }
        }
    }
    (labels, marker)
}

fn object_rect(image: usize, local: usize) -> (usize, usize, usize, usize) {
    let row = local / 3;
    let col = local % 3;
    let y0 = 10 + row * 36 + (image % 3);
    let x0 = 12 + col * 36 + ((image + local) % 5);
    let height = 16 + ((image + local) % 4);
    let width = 18 + ((2 * image + local) % 5);
    (y0, x0, height, width)
}

fn pixel_intensity(image: usize, local: usize, y: usize, x: usize) -> f64 {
    let base = 58.0 + 9.0 * local as f64 + 4.0 * (image % 7) as f64;
    base + ((y + 2 * x + image) % 11) as f64
}

fn measure_image(
    image: usize,
    labels: ndarray::ArrayView3<'_, u32>,
    marker: ndarray::ArrayView3<'_, f64>,
    threshold: f64,
) -> Result<Vec<ObjectRow>> {
    if labels.shape() != marker.shape() {
        return Err(Error::invalid(
            "percent-positive: label/value shape mismatch",
        ));
    }

    let mut tallies = BTreeMap::<u64, IntensityTally>::new();
    for ((z, y, x), &raw_label) in labels.indexed_iter() {
        if raw_label == 0 {
            continue;
        }
        let label = u64::from(raw_label);
        let at = [z, y, x];
        let value = marker[at];
        tallies
            .entry(label)
            .and_modify(|tally| tally.add(at, value))
            .or_insert_with(|| IntensityTally::new(at, value));
    }

    let mut rows = Vec::with_capacity(tallies.len());
    for (label, tally) in tallies {
        let intensity = tally.intensity(label);
        let mean = intensity.mean.ok_or_else(|| {
            Error::invalid(format!(
                "percent-positive: object {label} has no finite marker pixels"
            ))
        })?;
        rows.push(ObjectRow {
            image,
            label,
            count: intensity.count,
            centroid_y: tally.position[1] as f64 / tally.count as f64,
            centroid_x: tally.position[2] as f64 / tally.count as f64,
            intensity,
            positive: mean >= threshold,
        });
    }
    Ok(rows)
}

impl IntensityTally {
    fn new(at: [usize; 3], value: f64) -> Self {
        let mut tally = Self {
            count: 0,
            sum: 0.0,
            min: value,
            max: value,
            weighted_position: [0.0; 3],
            position: [0; 3],
        };
        tally.add(at, value);
        tally
    }

    fn add(&mut self, at: [usize; 3], value: f64) {
        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        for axis in 0..3 {
            self.position[axis] += at[axis] as u64;
            self.weighted_position[axis] += value * at[axis] as f64;
        }
    }

    fn intensity(&self, label: u64) -> IntensityMeasurements {
        IntensityMeasurements {
            label,
            count: self.count,
            finite_count: self.count,
            nonfinite: 0,
            sum: self.sum,
            mean: Some(self.sum / self.count as f64),
            min: Some(self.min),
            max: Some(self.max),
            weighted_centroid: (self.sum != 0.0).then_some([
                self.weighted_position[0] / self.sum,
                self.weighted_position[1] / self.sum,
                self.weighted_position[2] / self.sum,
            ]),
        }
    }
}

fn write_objects(rows: &[ObjectRow], path: &PathBuf) -> Result<()> {
    let file = File::create(path)
        .map_err(|err| Error::invalid(format!("percent-positive: create CSV: {err}")))?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "image,label,count,centroid_y,centroid_x,mean_intensity,sum_intensity,positive"
    )
    .map_err(write_error)?;
    for row in rows {
        writeln!(
            out,
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{}",
            row.image,
            row.label,
            row.count,
            row.centroid_y,
            row.centroid_x,
            row.intensity.mean.unwrap_or(f64::NAN),
            row.intensity.sum,
            if row.positive { "true" } else { "false" }
        )
        .map_err(write_error)?;
    }
    Ok(())
}

fn write_summary(config: &Config, rows: &[ObjectRow], path: &PathBuf) -> Result<()> {
    let positive = rows.iter().filter(|row| row.positive).count();
    let negative = rows.len() - positive;
    let total_area: u64 = rows.iter().map(|row| row.count).sum();
    let marker_sum: f64 = rows.iter().map(|row| row.intensity.sum).sum();
    let summary = json!({
        "images": config.images,
        "objects": rows.len(),
        "threshold": config.threshold,
        "positive": positive,
        "negative": negative,
        "percent_positive": positive as f64 / rows.len() as f64,
        "total_area": total_area,
        "marker_sum": marker_sum,
    });
    fs::write(
        path,
        serde_json::to_string_pretty(&summary).expect("summary JSON must serialize") + "\n",
    )
    .map_err(|err| Error::invalid(format!("percent-positive: write summary JSON: {err}")))
}

fn write_error(err: std::io::Error) -> Error {
    Error::invalid(format!("percent-positive: write output: {err}"))
}
