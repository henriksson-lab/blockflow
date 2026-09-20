// SPDX-License-Identifier: MIT

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::dtype::Dtype;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::measure::{
    collect_colocalization_rows, ColocalizationMeasurements, IntensityImage, Measurements,
};
use blockflow::probes::IdentityOp;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::{AttachedImage, Error, Result};
use clap::Parser;
use ndarray::Array3;
use serde_json::json;

const HEIGHT: usize = 72;
const WIDTH: usize = 96;
const OBJECTS_PER_IMAGE: usize = 4;

#[derive(Debug, Parser)]
#[command(
    name = "colocalization",
    about = "Measures labelled two-channel fixtures and reports colocalization rows."
)]
struct Config {
    #[arg(long, default_value = ".tmp/colocalization/blockflow")]
    out: PathBuf,
    #[arg(long, default_value_t = 10)]
    images: usize,
    #[arg(long)]
    fixture_dir: Option<PathBuf>,
    #[arg(long, default_value = ".tmp/colocalization/input.zarr")]
    zarr_dir: PathBuf,
    #[arg(long, value_parser = parse_chunk, default_value = "1x32x32")]
    chunk: [usize; 3],
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
        let input_zarr = ensure_colocalization_zarr(&config, image, labels, channel_a, channel_b)?;
        let mut measurements = planned_colocalization(&input_zarr, config.chunk)?;
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
        let config = <Self as Parser>::parse();

        if config.images == 0 {
            return Err(Error::invalid(
                "colocalization: --images must be at least 1",
            ));
        }
        if config.chunk.contains(&0) {
            return Err(Error::invalid(
                "colocalization: --chunk dimensions must be positive",
            ));
        }

        Ok(config)
    }
}

struct ColocalizationZarr {
    labels: PathBuf,
    channel_a: PathBuf,
    channel_b: PathBuf,
    work: PathBuf,
}

fn ensure_colocalization_zarr(
    config: &Config,
    image: usize,
    labels: Array3<f64>,
    channel_a: Array3<f64>,
    channel_b: Array3<f64>,
) -> Result<ColocalizationZarr> {
    let root = config.zarr_dir.join(format!("image-{image:03}"));
    let labels_path = ensure_array_zarr(&root.join("labels.zarr"), labels, config.chunk)?;
    let channel_a_path = ensure_array_zarr(&root.join("channel-a.zarr"), channel_a, config.chunk)?;
    let channel_b_path = ensure_array_zarr(&root.join("channel-b.zarr"), channel_b, config.chunk)?;
    Ok(ColocalizationZarr {
        labels: labels_path,
        channel_a: channel_a_path,
        channel_b: channel_b_path,
        work: root.join("work.zarr"),
    })
}

fn ensure_array_zarr(store: &Path, array: Array3<f64>, chunk: [usize; 3]) -> Result<PathBuf> {
    let path = store.join("level0");
    if path.join("zarr.json").exists() {
        let (dtype, volume) = AttachedImage::at(&path).metadata()?;
        if dtype != Dtype::F64 || volume != [1, HEIGHT, WIDTH] {
            return Err(Error::invalid(format!(
                "colocalization: prepared store {} is {dtype:?} {volume:?}, expected F64 {:?}",
                path.display(),
                [1, HEIGHT, WIDTH]
            )));
        }
        return Ok(path);
    }

    let voxels: Voxels = array.into();
    ZarrEnvironment::create(store, &voxels, chunk)?;
    Ok(path)
}

fn planned_colocalization(
    input: &ColocalizationZarr,
    chunk: [usize; 3],
) -> Result<Vec<ColocalizationMeasurements>> {
    let (_, volume) = AttachedImage::at(&input.labels).metadata()?;
    let grid = BlockGrid::new(volume, chunk)?;
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    builder.pixels(Chain::op(IdentityOp::new(
        "colocalization-label-source",
        [0, 0, 0],
    )))?;
    let base = builder.finish()?;
    let labels = ImageId::from(base.n_phases());
    let measurements = Measurements::for_labels(labels)
        .colocalization(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            IntensityImage::<1>::new(ImageId::supplied(1)).holding(Dtype::F64),
        )
        .stream("colocalization.measurements")
        .lifecycle(Lifecycle::DeleteOnExit)
        .build(base.decomposition.clone())?;
    let rows = measurements
        .colocalization_rows(0)
        .ok_or_else(|| Error::invalid("colocalization: planned rows are missing"))?;
    let env = ZarrEnvironment::attach(
        &input.work,
        &[
            AttachedImage::at(&input.labels),
            AttachedImage::at(&input.channel_a),
            AttachedImage::at(&input.channel_b),
        ],
    )?;
    let mut work = base.work();
    work.extend(measurements.phase_work());
    execute_phases(
        "colocalization planned measurement",
        &base.workflow,
        &measurements.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )?;
    collect_colocalization_rows(&env, &rows, volume)
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
    let b = if local.is_multiple_of(2) {
        4.0 + 1.7 * a + ((3 * x + y + image) % 7) as f64
    } else {
        140.0 - 1.2 * a + ((x + 5 * y + image) % 9) as f64
    };
    (a, b)
}

fn fixture_from_file(
    image_dir: &Path,
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

fn parse_field<T: std::str::FromStr>(raw: &str, name: &str, path: &Path) -> Result<T>
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
        "input_zarr_dir": config.zarr_dir.display().to_string(),
        "chunk_shape": config.chunk,
        "execution": "planned colocalization measurement over attached Zarr inputs",
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

fn parse_chunk(raw: &str) -> std::result::Result<[usize; 3], String> {
    let parts = raw
        .split(['x', 'X', ',', ':'])
        .map(str::trim)
        .collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(format!(
            "colocalization: chunk shape {raw:?} must have three dimensions"
        ));
    }
    let mut out = [0usize; 3];
    for (index, part) in parts.iter().enumerate() {
        out[index] = part
            .parse::<usize>()
            .map_err(|err| format!("colocalization: could not parse chunk shape {raw:?}: {err}"))?;
    }
    Ok(out)
}
