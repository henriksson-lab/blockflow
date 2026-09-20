// SPDX-License-Identifier: MIT

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::dtype::Dtype;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::measure::{
    collect_class_a_shapes, collect_class_a_values, IntensityImage, IntensityMeasurements,
    IntensitySet, Measurements, ShapeMeasurements, ShapeSet,
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

const HEIGHT: usize = 96;
const WIDTH: usize = 128;
const OBJECTS_PER_IMAGE: usize = 6;

#[derive(Debug, Parser)]
#[command(
    name = "percent-positive",
    about = "Generates deterministic labelled-cell fixtures and classifies objects by mean marker intensity."
)]
struct Config {
    #[arg(long, default_value = ".tmp/percent-positive/blockflow")]
    out: PathBuf,
    #[arg(long, default_value_t = 10)]
    images: usize,
    #[arg(long, default_value_t = 110.0)]
    threshold: f64,
    #[arg(long, default_value = ".tmp/percent-positive/input.zarr")]
    zarr_dir: PathBuf,
    #[arg(long, value_parser = parse_chunk, default_value = "1x32x32")]
    chunk: [usize; 3],
}

#[derive(Debug)]
struct ObjectRow {
    image: usize,
    label: u64,
    count: u64,
    shape: ShapeMeasurements,
    intensity: IntensityMeasurements,
    positive: bool,
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
        let input_zarr = ensure_percent_positive_zarr(&config, image, labels, marker)?;
        rows.extend(planned_percent_positive(
            image,
            &input_zarr,
            config.threshold,
            config.chunk,
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
        let config = <Self as Parser>::parse();

        if config.images == 0 {
            return Err(Error::invalid(
                "percent-positive: --images must be at least 1",
            ));
        }
        if !config.threshold.is_finite() {
            return Err(Error::invalid(
                "percent-positive: --threshold must be finite",
            ));
        }
        if config.chunk.contains(&0) {
            return Err(Error::invalid(
                "percent-positive: --chunk dimensions must be positive",
            ));
        }

        Ok(config)
    }
}

struct PercentPositiveZarr {
    labels: PathBuf,
    marker: PathBuf,
    work: PathBuf,
}

fn ensure_percent_positive_zarr(
    config: &Config,
    image: usize,
    labels: Array3<u32>,
    marker: Array3<f64>,
) -> Result<PercentPositiveZarr> {
    let root = config.zarr_dir.join(format!("image-{image:03}"));
    let labels = ensure_array_zarr(&root.join("labels.zarr"), labels, config.chunk)?;
    let marker = ensure_array_zarr(&root.join("marker.zarr"), marker, config.chunk)?;
    Ok(PercentPositiveZarr {
        labels,
        marker,
        work: root.join("work.zarr"),
    })
}

fn ensure_array_zarr<T>(store: &Path, array: Array3<T>, chunk: [usize; 3]) -> Result<PathBuf>
where
    T: blockflow::voxels::VoxelElement + 'static,
    Voxels: From<Array3<T>>,
{
    let path = store.join("level0");
    if path.join("zarr.json").exists() {
        let (_, volume) = AttachedImage::at(&path).metadata()?;
        if volume != [1, HEIGHT, WIDTH] {
            return Err(Error::invalid(format!(
                "percent-positive: prepared store {} is volume {volume:?}, expected {:?}",
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

fn planned_percent_positive(
    image: usize,
    input: &PercentPositiveZarr,
    threshold: f64,
    chunk: [usize; 3],
) -> Result<Vec<ObjectRow>> {
    let (_, volume) = AttachedImage::at(&input.labels).metadata()?;
    let grid = BlockGrid::new(volume, chunk)?;
    let mut builder = PlanBuilder::new(volume, Dtype::U32, grid);
    builder.pixels(Chain::op(IdentityOp::new(
        "percent-positive-label-source",
        [0, 0, 0],
    )))?;
    let base = builder.finish()?;
    let labels = ImageId::from(base.n_phases());
    let measurements = Measurements::for_labels(labels)
        .shape(ShapeSet::standard())
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            IntensitySet::standard(),
        )
        .stream("percent-positive.measurements")
        .lifecycle(Lifecycle::DeleteOnExit)
        .build(base.decomposition.clone())?;
    let shape_rows = measurements
        .class_a_rows()
        .ok_or_else(|| Error::invalid("percent-positive: planned shape rows are missing"))?;
    let intensity_rows = measurements
        .class_a_intensity_rows(0)
        .ok_or_else(|| Error::invalid("percent-positive: planned intensity rows are missing"))?;
    let env = ZarrEnvironment::attach(
        &input.work,
        &[
            AttachedImage::at(&input.labels),
            AttachedImage::at(&input.marker),
        ],
    )?;
    let mut work = base.work();
    work.extend(measurements.phase_work());
    execute_phases(
        "percent-positive planned measurement",
        &base.workflow,
        &measurements.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )?;
    let shapes = collect_class_a_shapes(&env, &shape_rows, volume, measurements.fixed)?;
    let intensities = collect_class_a_values(&env, &intensity_rows, volume, measurements.fixed)?;
    let mut intensity_by_label = intensities
        .into_iter()
        .map(|values| {
            let measurements = IntensityMeasurements::from_values(&values);
            (measurements.label, measurements)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut out = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let shape = ShapeMeasurements::from_shape(&shape);
        let intensity = intensity_by_label.remove(&shape.label).ok_or_else(|| {
            Error::invalid(format!(
                "percent-positive: no intensity row for label {}",
                shape.label
            ))
        })?;
        let mean = intensity.mean.ok_or_else(|| {
            Error::invalid(format!(
                "percent-positive: object {} has no finite marker pixels",
                shape.label
            ))
        })?;
        out.push(ObjectRow {
            image,
            label: shape.label,
            count: shape.count,
            shape,
            intensity,
            positive: mean >= threshold,
        });
    }
    Ok(out)
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
        let centroid = row.shape.centroid.unwrap_or([f64::NAN; 3]);
        writeln!(
            out,
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{}",
            row.image,
            row.label,
            row.count,
            centroid[1],
            centroid[2],
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
        "input_zarr_dir": config.zarr_dir.display().to_string(),
        "chunk_shape": config.chunk,
        "execution": "planned shape/intensity measurement over attached Zarr inputs",
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

fn parse_chunk(raw: &str) -> std::result::Result<[usize; 3], String> {
    let parts = raw
        .split(['x', 'X', ',', ':'])
        .map(str::trim)
        .collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(format!(
            "percent-positive: chunk shape {raw:?} must have three dimensions"
        ));
    }
    let mut out = [0usize; 3];
    for (index, part) in parts.iter().enumerate() {
        out[index] = part.parse::<usize>().map_err(|err| {
            format!("percent-positive: could not parse chunk shape {raw:?}: {err}")
        })?;
    }
    Ok(out)
}
