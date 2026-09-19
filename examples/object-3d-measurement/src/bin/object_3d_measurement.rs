// SPDX-License-Identifier: MIT

use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use blockflow::ops::{
    object_geometry_basic_measurements_u32, ObjectGeometryBasicMeasurements, PhysicalSpacing,
};
use blockflow::{Error, Result};
use ndarray::Array3;
use serde_json::json;

const SHAPE: [usize; 3] = [32, 48, 56];
const SPACING: [f64; 3] = [1.5, 0.75, 0.5];
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
    measurements: ObjectGeometryBasicMeasurements,
}

#[derive(Debug, Default)]
struct Timings {
    fixture_load: Duration,
    label_construction: Duration,
    measurement: Duration,
    csv_write: Duration,
}

#[derive(Debug, Clone, Copy)]
struct BoxSpec {
    label: u32,
    start: [usize; 3],
    extent: [usize; 3],
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
            "object-3d-measurement: create output directory {}: {err}",
            config.out.display()
        ))
    })?;

    let spacing = PhysicalSpacing::new(SPACING)?;
    let mut rows = Vec::new();
    let mut timings = Timings::default();
    for image in 0..config.images {
        let labels = if let Some(dir) = &config.fixture_dir {
            let started = Instant::now();
            let boxes = load_fixture_boxes(dir, image)?;
            timings.fixture_load += started.elapsed();
            let started = Instant::now();
            let labels = labels_from_boxes(&boxes);
            timings.label_construction += started.elapsed();
            labels
        } else {
            let started = Instant::now();
            let labels = synthetic_labels(image);
            timings.label_construction += started.elapsed();
            labels
        };
        let started = Instant::now();
        let mut measurements = object_geometry_basic_measurements_u32(labels.view(), spacing)?;
        timings.measurement += started.elapsed();
        measurements.sort_by_key(|row| row.label);
        rows.extend(measurements.into_iter().map(|measurements| ObjectRow {
            image,
            measurements,
        }));
    }

    let started = Instant::now();
    write_objects(&rows, &config.out.join("objects.csv"))?;
    timings.csv_write = started.elapsed();
    write_summary(&config, &rows, &timings, &config.out.join("summary.json"))?;

    println!(
        "images={} objects={} voxels={} output={}",
        config.images,
        rows.len(),
        rows.iter().map(|row| row.measurements.count).sum::<u64>(),
        config.out.display()
    );
    Ok(())
}

impl Config {
    fn parse() -> Result<Self> {
        let mut out = PathBuf::from(".tmp/object-3d-measurement/blockflow");
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
                        "object-3d-measurement: unknown argument {other:?}; use --help"
                    )));
                }
            }
        }

        if images == 0 {
            return Err(Error::invalid(
                "object-3d-measurement: --images must be at least 1",
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
        .ok_or_else(|| Error::invalid(format!("object-3d-measurement: {name} needs a path")))
}

fn parse_arg<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let raw = args
        .next()
        .ok_or_else(|| Error::invalid(format!("object-3d-measurement: {name} needs a value")))?;
    raw.parse::<T>().map_err(|err| {
        Error::invalid(format!(
            "object-3d-measurement: could not parse {name} value {raw:?}: {err}"
        ))
    })
}

fn print_help() {
    println!(
        "object-3d-measurement --out DIR [--images 10] [--fixture-dir DIR]\n\
         Measures deterministic labelled 3-D objects with physical geometry."
    );
}

fn synthetic_labels(image: usize) -> Array3<u32> {
    let mut labels = Array3::<u32>::zeros((SHAPE[0], SHAPE[1], SHAPE[2]));
    for local in 0..OBJECTS_PER_IMAGE {
        let label = (image * 100 + local + 1) as u32;
        let (start, extent) = object_box(image, local);
        for z in start[0]..start[0] + extent[0] {
            for y in start[1]..start[1] + extent[1] {
                for x in start[2]..start[2] + extent[2] {
                    labels[[z, y, x]] = label;
                }
            }
        }
    }
    labels
}

fn object_box(image: usize, local: usize) -> ([usize; 3], [usize; 3]) {
    let start = [
        2 + (local % 2) * 13 + (image % 2),
        4 + (local / 2) * 20 + (image % 3),
        5 + (local % 2) * 25 + ((image + local) % 4),
    ];
    let extent = [
        5 + ((image + local) % 4),
        8 + ((2 * image + local) % 5),
        9 + ((image + 2 * local) % 6),
    ];
    (start, extent)
}

fn load_fixture_boxes(dir: &PathBuf, image: usize) -> Result<Vec<BoxSpec>> {
    let path = dir.join(format!("boxes-{image:03}.csv"));
    let file = File::open(&path).map_err(|err| {
        Error::invalid(format!(
            "object-3d-measurement: open fixture {}: {err}",
            path.display()
        ))
    })?;
    let mut boxes = Vec::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|err| {
            Error::invalid(format!(
                "object-3d-measurement: read fixture {}: {err}",
                path.display()
            ))
        })?;
        if line_index == 0 {
            if line.trim() != "label,z0,y0,x0,dz,dy,dx" {
                return Err(Error::invalid(format!(
                    "object-3d-measurement: unexpected fixture header in {}",
                    path.display()
                )));
            }
            continue;
        }
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.len() != 7 {
            return Err(Error::invalid(format!(
                "object-3d-measurement: malformed fixture row {} in {}",
                line_index + 1,
                path.display()
            )));
        }
        let label: u32 = parse_field(fields[0], "label", &path)?;
        let start: [usize; 3] = [
            parse_field(fields[1], "z0", &path)?,
            parse_field(fields[2], "y0", &path)?,
            parse_field(fields[3], "x0", &path)?,
        ];
        let extent: [usize; 3] = [
            parse_field(fields[4], "dz", &path)?,
            parse_field(fields[5], "dy", &path)?,
            parse_field(fields[6], "dx", &path)?,
        ];
        for axis in 0..3 {
            if start[axis] + extent[axis] > SHAPE[axis] {
                return Err(Error::invalid(format!(
                    "object-3d-measurement: fixture row {} exceeds shape {:?}",
                    line_index + 1,
                    SHAPE
                )));
            }
        }
        boxes.push(BoxSpec {
            label,
            start,
            extent,
        });
    }
    Ok(boxes)
}

fn labels_from_boxes(boxes: &[BoxSpec]) -> Array3<u32> {
    let mut labels = Array3::<u32>::zeros((SHAPE[0], SHAPE[1], SHAPE[2]));
    for spec in boxes {
        for z in spec.start[0]..spec.start[0] + spec.extent[0] {
            for y in spec.start[1]..spec.start[1] + spec.extent[1] {
                for x in spec.start[2]..spec.start[2] + spec.extent[2] {
                    labels[[z, y, x]] = spec.label;
                }
            }
        }
    }
    labels
}

fn parse_field<T: std::str::FromStr>(raw: &str, name: &str, path: &PathBuf) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    raw.parse::<T>().map_err(|err| {
        Error::invalid(format!(
            "object-3d-measurement: parse {name}={raw:?} in {}: {err}",
            path.display()
        ))
    })
}

fn write_objects(rows: &[ObjectRow], path: &PathBuf) -> Result<()> {
    let file = File::create(path).map_err(|err| {
        Error::invalid(format!("object-3d-measurement: create objects CSV: {err}"))
    })?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "image,label,count,bbox_min_z,bbox_min_y,bbox_min_x,bbox_max_z,bbox_max_y,bbox_max_x,physical_bbox_extent_z,physical_bbox_extent_y,physical_bbox_extent_x"
    )
    .map_err(write_error)?;
    for row in rows {
        let m = row.measurements;
        writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6}",
            row.image,
            m.label,
            m.count,
            m.bbox_min[0],
            m.bbox_min[1],
            m.bbox_min[2],
            m.bbox_max[0],
            m.bbox_max[1],
            m.bbox_max[2],
            m.physical_bbox_extent[0],
            m.physical_bbox_extent[1],
            m.physical_bbox_extent[2]
        )
        .map_err(write_error)?;
    }
    Ok(())
}

fn write_summary(
    config: &Config,
    rows: &[ObjectRow],
    timings: &Timings,
    path: &PathBuf,
) -> Result<()> {
    let total_voxels: u64 = rows.iter().map(|row| row.measurements.count).sum();
    let started = Instant::now();
    write_summary_with_seconds(config, rows, timings, path, total_voxels, 0.0)?;
    let summary_write_seconds = started.elapsed().as_secs_f64();
    write_summary_with_seconds(
        config,
        rows,
        timings,
        path,
        total_voxels,
        summary_write_seconds,
    )
}

fn write_summary_with_seconds(
    config: &Config,
    rows: &[ObjectRow],
    timings: &Timings,
    path: &PathBuf,
    total_voxels: u64,
    summary_write_seconds: f64,
) -> Result<()> {
    let summary = json!({
        "images": config.images,
        "objects": rows.len(),
        "spacing_x": SPACING[2],
        "spacing_y": SPACING[1],
        "spacing_z": SPACING[0],
        "total_voxels": total_voxels,
        "fixture_load_seconds": timings.fixture_load.as_secs_f64(),
        "label_construction_seconds": timings.label_construction.as_secs_f64(),
        "measurement_seconds": timings.measurement.as_secs_f64(),
        "csv_write_seconds": timings.csv_write.as_secs_f64(),
        "summary_write_seconds": summary_write_seconds,
    });
    fs::write(
        path,
        serde_json::to_string_pretty(&summary).expect("summary JSON must serialize") + "\n",
    )
    .map_err(|err| Error::invalid(format!("object-3d-measurement: write summary JSON: {err}")))
}

fn write_error(err: std::io::Error) -> Error {
    Error::invalid(format!("object-3d-measurement: write output: {err}"))
}
