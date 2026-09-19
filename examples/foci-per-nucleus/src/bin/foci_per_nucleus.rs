// SPDX-License-Identifier: MIT

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use blockflow::{Error, Result};
use ndarray::Array3;
use serde_json::json;

const HEIGHT: usize = 104;
const WIDTH: usize = 136;
const NUCLEI_PER_IMAGE: usize = 5;

#[derive(Debug)]
struct Config {
    out: PathBuf,
    images: usize,
}

#[derive(Debug)]
struct NucleusRow {
    image: usize,
    label: u64,
    area: u64,
    foci_count: u64,
    foci_intensity_sum: f64,
}

#[derive(Debug)]
struct FocusRow {
    image: usize,
    focus: u64,
    y: usize,
    x: usize,
    intensity: f64,
    nucleus_label: u64,
}

#[derive(Debug, Clone, Copy)]
struct FocusSpec {
    y: usize,
    x: usize,
    intensity: f64,
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
            "foci-per-nucleus: create output directory {}: {err}",
            config.out.display()
        ))
    })?;

    let mut nuclei = Vec::new();
    let mut foci = Vec::new();
    for image in 0..config.images {
        let labels = synthetic_nuclei(image);
        let specs = synthetic_foci(image);
        let (mut image_nuclei, mut image_foci) = count_foci(image, labels.view(), &specs)?;
        nuclei.append(&mut image_nuclei);
        foci.append(&mut image_foci);
    }

    write_nuclei(&nuclei, &config.out.join("nuclei.csv"))?;
    write_foci(&foci, &config.out.join("foci.csv"))?;
    write_summary(&config, &nuclei, &foci, &config.out.join("summary.json"))?;

    let assigned = foci.iter().filter(|row| row.nucleus_label != 0).count();
    println!(
        "images={} nuclei={} foci={} assigned={} output={}",
        config.images,
        nuclei.len(),
        foci.len(),
        assigned,
        config.out.display()
    );
    Ok(())
}

impl Config {
    fn parse() -> Result<Self> {
        let mut out = PathBuf::from(".tmp/foci-per-nucleus/blockflow");
        let mut images = 10usize;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out" => out = path_arg(&mut args, "--out")?,
                "--images" => images = parse_arg(&mut args, "--images")?,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(Error::invalid(format!(
                        "foci-per-nucleus: unknown argument {other:?}; use --help"
                    )));
                }
            }
        }

        if images == 0 {
            return Err(Error::invalid(
                "foci-per-nucleus: --images must be at least 1",
            ));
        }

        Ok(Self { out, images })
    }
}

fn path_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<PathBuf> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::invalid(format!("foci-per-nucleus: {name} needs a path")))
}

fn parse_arg<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let raw = args
        .next()
        .ok_or_else(|| Error::invalid(format!("foci-per-nucleus: {name} needs a value")))?;
    raw.parse::<T>().map_err(|err| {
        Error::invalid(format!(
            "foci-per-nucleus: could not parse {name} value {raw:?}: {err}"
        ))
    })
}

fn print_help() {
    println!(
        "foci-per-nucleus --out DIR [--images 10]\n\
         Generates deterministic nucleus labels and foci, then counts foci by containing nucleus."
    );
}

fn synthetic_nuclei(image: usize) -> Array3<u32> {
    let mut labels = Array3::<u32>::zeros((1, HEIGHT, WIDTH));
    for local in 0..NUCLEI_PER_IMAGE {
        let label = (image * 100 + local + 1) as u32;
        let (y0, x0, height, width) = nucleus_rect(image, local);
        for y in y0..y0 + height {
            for x in x0..x0 + width {
                labels[[0, y, x]] = label;
            }
        }
    }
    labels
}

fn nucleus_rect(image: usize, local: usize) -> (usize, usize, usize, usize) {
    let y0 = 12 + (local / 3) * 42 + (image % 4);
    let x0 = 10 + (local % 3) * 40 + ((image + local) % 6);
    let height = 24 + ((image + 2 * local) % 5);
    let width = 26 + ((2 * image + local) % 7);
    (y0, x0, height, width)
}

fn synthetic_foci(image: usize) -> Vec<FocusSpec> {
    let mut specs = Vec::new();
    for local in 0..NUCLEI_PER_IMAGE {
        let (y0, x0, height, width) = nucleus_rect(image, local);
        let count = 1 + ((image + local) % 3);
        for index in 0..count {
            let y = y0 + 3 + ((image + 5 * index + local) % (height - 6).max(1));
            let x = x0 + 4 + ((2 * image + 7 * index + local) % (width - 8).max(1));
            let intensity = 180.0 + 11.0 * index as f64 + 3.0 * local as f64 + (image % 5) as f64;
            specs.push(FocusSpec { y, x, intensity });
        }
    }
    specs.push(FocusSpec {
        y: 2 + image % 5,
        x: 3 + image % 7,
        intensity: 99.0,
    });
    let (y0, x0, _height, width) = nucleus_rect(image, 0);
    specs.push(FocusSpec {
        y: y0,
        x: x0 + width,
        intensity: 120.0,
    });
    specs
}

fn count_foci(
    image: usize,
    labels: ndarray::ArrayView3<'_, u32>,
    specs: &[FocusSpec],
) -> Result<(Vec<NucleusRow>, Vec<FocusRow>)> {
    let mut nuclei = BTreeMap::<u64, NucleusRow>::new();
    for &raw_label in labels.iter() {
        if raw_label == 0 {
            continue;
        }
        let label = u64::from(raw_label);
        nuclei
            .entry(label)
            .and_modify(|row| row.area += 1)
            .or_insert(NucleusRow {
                image,
                label,
                area: 1,
                foci_count: 0,
                foci_intensity_sum: 0.0,
            });
    }

    let mut foci = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        let label = u64::from(labels[[0, spec.y, spec.x]]);
        if label != 0 {
            let row = nuclei.get_mut(&label).ok_or_else(|| {
                Error::invalid(format!(
                    "foci-per-nucleus: focus assigned to missing nucleus label {label}"
                ))
            })?;
            row.foci_count += 1;
            row.foci_intensity_sum += spec.intensity;
        }
        foci.push(FocusRow {
            image,
            focus: (image * 1000 + index + 1) as u64,
            y: spec.y,
            x: spec.x,
            intensity: spec.intensity,
            nucleus_label: label,
        });
    }

    Ok((nuclei.into_values().collect(), foci))
}

fn write_nuclei(rows: &[NucleusRow], path: &PathBuf) -> Result<()> {
    let file = File::create(path)
        .map_err(|err| Error::invalid(format!("foci-per-nucleus: create nuclei CSV: {err}")))?;
    let mut out = BufWriter::new(file);
    writeln!(out, "image,label,area,foci_count,foci_intensity_sum").map_err(write_error)?;
    for row in rows {
        writeln!(
            out,
            "{},{},{},{},{:.6}",
            row.image, row.label, row.area, row.foci_count, row.foci_intensity_sum
        )
        .map_err(write_error)?;
    }
    Ok(())
}

fn write_foci(rows: &[FocusRow], path: &PathBuf) -> Result<()> {
    let file = File::create(path)
        .map_err(|err| Error::invalid(format!("foci-per-nucleus: create foci CSV: {err}")))?;
    let mut out = BufWriter::new(file);
    writeln!(out, "image,focus,y,x,intensity,nucleus_label").map_err(write_error)?;
    for row in rows {
        writeln!(
            out,
            "{},{},{},{},{:.6},{}",
            row.image, row.focus, row.y, row.x, row.intensity, row.nucleus_label
        )
        .map_err(write_error)?;
    }
    Ok(())
}

fn write_summary(
    config: &Config,
    nuclei: &[NucleusRow],
    foci: &[FocusRow],
    path: &PathBuf,
) -> Result<()> {
    let assigned = foci.iter().filter(|row| row.nucleus_label != 0).count();
    let summary = json!({
        "assigned_foci": assigned,
        "images": config.images,
        "nuclei": nuclei.len(),
        "total_foci": foci.len(),
        "unassigned_foci": foci.len() - assigned,
    });
    fs::write(
        path,
        serde_json::to_string_pretty(&summary).expect("summary JSON must serialize") + "\n",
    )
    .map_err(|err| Error::invalid(format!("foci-per-nucleus: write summary JSON: {err}")))
}

fn write_error(err: std::io::Error) -> Error {
    Error::invalid(format!("foci-per-nucleus: write output: {err}"))
}
