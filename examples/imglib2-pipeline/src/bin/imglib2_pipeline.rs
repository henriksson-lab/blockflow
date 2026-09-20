// SPDX-License-Identifier: MIT

#[path = "../../../support/planning.rs"]
mod example_planning;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::dtype::Dtype;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::label::{LabelComponentsOp, RelabelComponentsOp};
use blockflow::ops::{
    append_global_threshold_phases, append_remove_small_objects_phases, collect_class_a_shapes,
    Boundary, Connectivity, Gaussian, GlobalThreshold, GlobalThresholdOutput,
    GlobalThresholdSelection, Measurements, ShapeMeasurements, ShapeSet, SmoothOp, ThresholdTest,
};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::{AttachedImage, Error, Result};
use clap::Parser;
use ndarray::Array3;

#[derive(Debug, Parser)]
#[command(
    name = "imglib2-pipeline",
    about = "Runs the Blockflow side of the ImgLib2 comparison benchmark."
)]
struct Config {
    #[arg(long, required_unless_present = "input_zarr")]
    input: Option<PathBuf>,
    /// Existing rank-3 Zarr array (for an OME-Zarr pyramid, pass its level directory).
    #[arg(long)]
    input_zarr: Option<PathBuf>,
    /// Channel index when the rank-3 input array is [channel, y, x].
    #[arg(long)]
    channel: Option<usize>,
    /// Convert a fixture to Zarr and exit before processing.
    #[arg(long, default_value_t = false)]
    prepare_only: bool,
    #[arg(long, default_value = ".tmp/imglib2-pipeline/blockflow")]
    out: PathBuf,
    #[arg(long, default_value_t = 1.5)]
    sigma: f64,
    #[arg(long, default_value_t = 20)]
    min_size: u64,
    #[arg(long, default_value = ".tmp/imglib2-pipeline/input.zarr")]
    zarr_dir: PathBuf,
    #[arg(long, value_parser = parse_chunk, default_value = "1x256x256")]
    chunk: [usize; 3],
}

#[derive(Debug)]
struct ObjectRow {
    label: u64,
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
    let input_zarr = ensure_input_zarr(&config)?;
    let load_seconds = started.elapsed().as_secs_f64();
    if config.prepare_only {
        println!("prepared {}", input_zarr.display());
        return Ok(());
    }

    let pipeline_started = Instant::now();
    let rows = planned_segmentation(&config, &input_zarr)?;
    let pipeline_seconds = pipeline_started.elapsed().as_secs_f64();

    write_objects(&rows, &config.out.join("objects.csv"))?;
    write_summary(
        &config,
        rows.len(),
        rows.iter().map(|row| row.count).sum(),
        load_seconds,
        pipeline_seconds,
        &config.out.join("summary.json"),
    )?;

    println!(
        "objects={} threshold_method=otsu output={}",
        rows.len(),
        config.out.display()
    );
    Ok(())
}

impl Config {
    fn parse() -> Result<Self> {
        let config = <Self as Parser>::parse();

        if config.sigma < 0.0 || !config.sigma.is_finite() {
            return Err(Error::invalid(
                "imglib2-pipeline: --sigma must be finite and non-negative",
            ));
        }
        if config.min_size == 0 {
            return Err(Error::invalid(
                "imglib2-pipeline: --min-size must be at least 1",
            ));
        }
        if config.chunk.contains(&0) {
            return Err(Error::invalid(
                "imglib2-pipeline: --chunk dimensions must be positive",
            ));
        }

        Ok(config)
    }
}

fn ensure_input_zarr(config: &Config) -> Result<PathBuf> {
    if let Some(array) = &config.input_zarr {
        AttachedImage::at(array).metadata()?;
        return Ok(array.clone());
    }
    let array = config.zarr_dir.join("level0");
    if array.join("zarr.json").exists() {
        let (dtype, _) = AttachedImage::at(&array).metadata()?;
        if dtype != Dtype::F64 {
            return Err(Error::invalid(format!(
                "imglib2-pipeline: prepared input {} is {dtype:?}, expected F64",
                array.display()
            )));
        }
        return Ok(array);
    }

    let input = load_luma(
        config
            .input
            .as_ref()
            .ok_or_else(|| Error::invalid("--input or --input-zarr is required"))?,
    )?;
    let voxels: Voxels = input.into();
    ZarrEnvironment::create(&config.zarr_dir, &voxels, config.chunk)?;
    Ok(array)
}

fn planned_segmentation(config: &Config, input_zarr: &Path) -> Result<Vec<ObjectRow>> {
    let source = AttachedImage::at(input_zarr);
    let source = if let Some(channel) = config.channel {
        let (_, shape) = source.metadata()?;
        source.plane(channel, [shape[1], shape[2]])
    } else {
        source
    };
    let (dtype, volume) = source.metadata()?;
    let grid = BlockGrid::new(volume, config.chunk)?;
    let mut builder = PlanBuilder::new(volume, dtype, grid);
    example_planning::pixels(
        &mut builder,
        Chain::sequence(vec![Chain::op(SmoothOp::new(
            "imglib2-gaussian-smooth",
            Gaussian::new([0.0, config.sigma, config.sigma], 3.0)?.with_boundary(Boundary::Reflect),
        ))]),
    )?;
    append_global_threshold_phases(
        &mut builder,
        "imglib2-threshold",
        GlobalThresholdSelection::single(GlobalThreshold::otsu(256)?),
        GlobalThresholdOutput::Mask {
            test: ThresholdTest::Above,
        },
    )?;
    append_remove_small_objects_phases(
        &mut builder,
        "imglib2-remove-small",
        Lifecycle::DeleteOnExit,
        Connectivity::Faces,
        config.min_size,
    )?;
    let faces = builder.fragments(
        LabelComponentsOp::new(
            "imglib2-label-components",
            "imglib2.components",
            Lifecycle::DeleteOnExit,
        )
        .connecting(Connectivity::Faces),
    )?;
    builder.fragments(
        RelabelComponentsOp::reading(
            "imglib2-relabel-components",
            "imglib2.components",
            faces,
            builder.grid(),
        )
        .connecting(Connectivity::Faces),
    )?;
    let base = builder.finish()?;
    let labels = ImageId::from(base.n_phases());
    let measurements = Measurements::for_labels(labels)
        .shape(ShapeSet::basic())
        .stream("imglib2.measurements")
        .lifecycle(Lifecycle::DeleteOnExit)
        .build(base.decomposition.clone())?;
    let shape_rows = measurements
        .class_a_rows()
        .ok_or_else(|| Error::invalid("imglib2-pipeline: planned shape rows are missing"))?;
    let scratch = tempfile::tempdir_in(&config.out).map_err(Error::backend)?;
    let env = ZarrEnvironment::attach(scratch.path(), &[source])?;
    let mut work = base.work();
    work.extend(measurements.phase_work());
    execute_phases(
        "imglib2 planned segmentation",
        &base.workflow,
        &measurements.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )?;
    let mut rows = collect_class_a_shapes(&env, &shape_rows, volume, measurements.fixed)?
        .into_iter()
        .map(|shape| {
            let shape = ShapeMeasurements::from_shape(&shape);
            let centroid = shape.centroid.unwrap_or([f64::NAN; 3]);
            ObjectRow {
                label: shape.label,
                count: shape.count,
                centroid_y: centroid[1],
                centroid_x: centroid[2],
            }
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| row.label);
    Ok(rows)
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
    load_seconds: f64,
    pipeline_seconds: f64,
    path: &Path,
) -> Result<()> {
    let summary = serde_json::json!({
        "input": config.input,
        "objects": objects,
        "total_foreground_area": total_area,
        "threshold_method": "otsu",
        "sigma": config.sigma,
        "min_size": config.min_size,
        "load_seconds": load_seconds,
        "pipeline_seconds": pipeline_seconds,
        "input_zarr": config.input_zarr.clone().unwrap_or_else(|| config.zarr_dir.join("level0")),
        "channel": config.channel,
        "chunk_shape": config.chunk,
        "execution": "planned segmentation over attached Zarr input",
    });
    let text = serde_json::to_string_pretty(&summary)
        .map_err(|err| Error::invalid(format!("imglib2-pipeline: encode summary: {err}")))?;
    fs::write(path, text)
        .map_err(|err| Error::invalid(format!("imglib2-pipeline: write summary: {err}")))
}

fn write_error(err: std::io::Error) -> Error {
    Error::invalid(format!("imglib2-pipeline: write output: {err}"))
}

fn parse_chunk(raw: &str) -> std::result::Result<[usize; 3], String> {
    let parts = raw
        .split(['x', 'X', ',', ':'])
        .map(str::trim)
        .collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(format!(
            "imglib2-pipeline: chunk shape {raw:?} must have three dimensions"
        ));
    }
    let mut out = [0usize; 3];
    for (index, part) in parts.iter().enumerate() {
        out[index] = part.parse::<usize>().map_err(|err| {
            format!("imglib2-pipeline: could not parse chunk shape {raw:?}: {err}")
        })?;
    }
    Ok(out)
}
