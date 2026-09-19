// SPDX-License-Identifier: MIT

use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use blockflow::ops::{colocalization_measurements, ColocalizationMeasurements};
use blockflow::{Error, Result};
use ndarray::Array3;
use serde_json::json;

const HEIGHT: usize = 72;
const WIDTH: usize = 96;
const OBJECTS_PER_IMAGE: usize = 4;

#[derive(Debug)]
struct Config {
    out: PathBuf,
    images: usize,
    fixture_dir: Option<PathBuf>,
}

#[derive(Debug)]
struct ObjectRow {
    image: usize,
    measurements: ColocalizationMeasurements,
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
            "colocalization: create output directory {}: {err}",
            config.out.display()
        ))
    })?;

    let mut rows = Vec::new();
    for image in 0..config.images {
        let (labels, channel_a, channel_b) = if let Some(dir) = &config.fixture_dir {
            fixture_from_file(dir, image)?
        } else {
            synthetic_fixture(image)
        };
        let mut measurements =
            colocalization_measurements(labels.view(), channel_a.view(), channel_b.view())?;
        measurements.sort_by_key(|row| row.label);
        rows.extend(measurements.into_iter().map(|measurements| ObjectRow {
            image,
            measurements,
        }));
    }

    write_objects(&rows, &config.out.join("objects.csv"))?;
    write_summary(&config, &rows, &config.out.join("summary.json"))?;

    println!(
        "images={} objects={} pairs={} output={}",
        config.images,
        rows.len(),
        rows.iter().map(|row| row.measurements.count).sum::<u64>(),
        config.out.display()
    );
    Ok(())
}

impl Config {
    fn parse() -> Result<Self> {
        let mut out = PathBuf::from(".tmp/colocalization/blockflow");
        let mut images = 10usize;
        let mut fixture_dir = None;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out" => out = path_arg(&mut args, "--out")?,
                "--images" => images = parse_arg(&mut args, "--images")?,
                "--fixture-dir" => fixture_dir = Some(path_arg(&mut args, "--fixture-dir")?),
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(Error::invalid(format!(
                        "colocalization: unknown argument {other:?}; use --help"
                    )));
                }
            }
        }

        if images == 0 {
            return Err(Error::invalid(
                "colocalization: --images must be at least 1",
            ));
        }

        Ok(Self {
            out,
            images,
            fixture_dir,
        })
    }
}

fn path_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<PathBuf> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::invalid(format!("colocalization: {name} needs a path")))
}

fn parse_arg<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let raw = args
        .next()
        .ok_or_else(|| Error::invalid(format!("colocalization: {name} needs a value")))?;
    raw.parse::<T>().map_err(|err| {
        Error::invalid(format!(
            "colocalization: could not parse {name} value {raw:?}: {err}"
        ))
    })
}

fn print_help() {
    println!(
        "colocalization --out DIR [--images 10] [--fixture-dir DIR]\n\
         Measures labelled two-channel fixtures and reports colocalization rows."
    );
}

fn synthetic_fixture(image: usize) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let mut labels = Array3::<f64>::zeros((1, HEIGHT, WIDTH));
    let mut channel_a = Array3::<f64>::zeros((1, HEIGHT, WIDTH));
    let mut channel_b = Array3::<f64>::zeros((1, HEIGHT, WIDTH));
    for local in 0..OBJECTS_PER_IMAGE {
        let label = (image * 100 + local + 1) as f64;
        let (y0, x0, height, width) = object_rect(image, local);
        for y in y0..y0 + height {
            for x in x0..x0 + width {
                let (a, b) = channels(image, local, y, x);
                labels[[0, y, x]] = label;
                channel_a[[0, y, x]] = a;
                channel_b[[0, y, x]] = b;
            }
        }
    }
    (labels, channel_a, channel_b)
}

fn object_rect(image: usize, local: usize) -> (usize, usize, usize, usize) {
    let row = local / 2;
    let col = local % 2;
    let y0 = 9 + row * 31 + (image % 4);
    let x0 = 11 + col * 42 + ((image + local) % 5);
    let height = 19 + ((image + local) % 5);
    let width = 22 + ((2 * image + local) % 6);
    (y0, x0, height, width)
}

fn channels(image: usize, local: usize, y: usize, x: usize) -> (f64, f64) {
    let base_a = 15.0 + 5.0 * local as f64 + 2.0 * (image % 6) as f64;
    let a = base_a + ((x + 2 * y + image) % 23) as f64;
    let b = if local % 2 == 0 {
        4.0 + 1.7 * a + ((3 * x + y + image) % 7) as f64
    } else {
        140.0 - 1.2 * a + ((x + 5 * y + image) % 9) as f64
    };
    (a, b)
}

fn fixture_from_file(
    image_dir: &PathBuf,
    image: usize,
) -> Result<(Array3<f64>, Array3<f64>, Array3<f64>)> {
    let path = image_dir.join(format!("objects-{image:03}.csv"));
    let file = File::open(&path).map_err(|err| {
        Error::invalid(format!(
            "colocalization: open fixture {}: {err}",
            path.display()
        ))
    })?;
    let mut labels = Array3::<f64>::zeros((1, HEIGHT, WIDTH));
    let mut channel_a = Array3::<f64>::zeros((1, HEIGHT, WIDTH));
    let mut channel_b = Array3::<f64>::zeros((1, HEIGHT, WIDTH));
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|err| {
            Error::invalid(format!(
                "colocalization: read fixture {}: {err}",
                path.display()
            ))
        })?;
        if line_index == 0 {
            if line.trim() != "label,local,y0,x0,height,width" {
                return Err(Error::invalid(format!(
                    "colocalization: unexpected fixture header in {}",
                    path.display()
                )));
            }
            continue;
        }
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.len() != 6 {
            return Err(Error::invalid(format!(
                "colocalization: malformed fixture row {} in {}",
                line_index + 1,
                path.display()
            )));
        }
        let label: u64 = parse_field(fields[0], "label", &path)?;
        let local: usize = parse_field(fields[1], "local", &path)?;
        let y0: usize = parse_field(fields[2], "y0", &path)?;
        let x0: usize = parse_field(fields[3], "x0", &path)?;
        let height: usize = parse_field(fields[4], "height", &path)?;
        let width: usize = parse_field(fields[5], "width", &path)?;
        if y0 + height > HEIGHT || x0 + width > WIDTH {
            return Err(Error::invalid(format!(
                "colocalization: fixture row {} exceeds shape",
                line_index + 1
            )));
        }
        for y in y0..y0 + height {
            for x in x0..x0 + width {
                let (a, b) = channels(image, local, y, x);
                labels[[0, y, x]] = label as f64;
                channel_a[[0, y, x]] = a;
                channel_b[[0, y, x]] = b;
            }
        }
    }
    Ok((labels, channel_a, channel_b))
}

fn parse_field<T: std::str::FromStr>(raw: &str, name: &str, path: &PathBuf) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    raw.parse::<T>().map_err(|err| {
        Error::invalid(format!(
            "colocalization: parse {name}={raw:?} in {}: {err}",
            path.display()
        ))
    })
}

fn write_objects(rows: &[ObjectRow], path: &PathBuf) -> Result<()> {
    let file = File::create(path)
        .map_err(|err| Error::invalid(format!("colocalization: create objects CSV: {err}")))?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "image,label,count,finite_count,pearson,slope_b_on_a,overlap_coefficient,manders_m1,manders_m2,sum_a,sum_b,sum_ab"
    )
    .map_err(write_error)?;
    for row in rows {
        let m = row.measurements;
        writeln!(
            out,
            "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            row.image,
            m.label,
            m.count,
            m.finite_count,
            m.pearson().unwrap_or(f64::NAN),
            m.slope_b_on_a().unwrap_or(f64::NAN),
            m.overlap_coefficient().unwrap_or(f64::NAN),
            m.manders_m1().unwrap_or(f64::NAN),
            m.manders_m2().unwrap_or(f64::NAN),
            m.sum_a,
            m.sum_b,
            m.sum_ab
        )
        .map_err(write_error)?;
    }
    Ok(())
}

fn write_summary(config: &Config, rows: &[ObjectRow], path: &PathBuf) -> Result<()> {
    let pairs: u64 = rows.iter().map(|row| row.measurements.count).sum();
    let finite_pairs: u64 = rows.iter().map(|row| row.measurements.finite_count).sum();
    let summary = json!({
        "finite_pairs": finite_pairs,
        "images": config.images,
        "objects": rows.len(),
        "pairs": pairs,
    });
    fs::write(
        path,
        serde_json::to_string_pretty(&summary).expect("summary JSON must serialize") + "\n",
    )
    .map_err(|err| Error::invalid(format!("colocalization: write summary JSON: {err}")))
}

fn write_error(err: std::io::Error) -> Error {
    Error::invalid(format!("colocalization: write output: {err}"))
}
