use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::yolo3d::{finalize_detections, InferenceDevice, Yolo3dDetector};
use blockflow::{BlockGrid, Error, PlanBuilder, Result, ZarrEnvironment};
use clap::{Parser, ValueEnum};
use yolo3d::DetectorConfig;

#[derive(Debug, Parser)]
#[command(about = "Run the anisotropy-aware YOLO3D detector over an OME-Zarr volume")]
struct Args {
    #[arg(long)]
    zarr: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    /// DetectorConfig JSON saved beside the checkpoint by training.
    #[arg(long)]
    model_config: PathBuf,
    #[arg(long, default_value_t = 0)]
    channel: usize,
    #[arg(long, default_value_t = 0)]
    time: usize,
    #[arg(long, value_enum, default_value_t = DeviceChoice::Cpu)]
    device: DeviceChoice,
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    #[arg(long, default_value_t = 32)]
    block_z: usize,
    #[arg(long, default_value_t = 256)]
    block: usize,
    #[arg(long, default_value_t = 8)]
    halo_z: usize,
    #[arg(long, default_value_t = 32)]
    halo: usize,
    #[arg(long, default_value_t = 0.25)]
    threshold: f32,
    #[arg(long, default_value_t = 0.3)]
    nms_iou: f32,
    #[arg(long, default_value_t = 0.0)]
    normalize_low: f32,
    #[arg(long, default_value_t = 255.0)]
    normalize_high: f32,
    #[arg(long, default_value_t = 1)]
    workers: usize,
    #[arg(long, default_value = "yolo3d-dapi")]
    layer: String,
    #[arg(long, default_value_t = false)]
    overwrite: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DeviceChoice {
    Cpu,
    Cuda,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::invalid("this example requires cargo run --release"));
    }
    let args = Args::parse();
    if args.block_z == 0 || args.block == 0 || args.halo_z == 0 || args.halo == 0 {
        return Err(Error::invalid("block and halo dimensions must be positive"));
    }
    let started = Instant::now();
    let pyramid = OmeVolumePyramid::open(&args.zarr, args.channel, args.time)?;
    let source = pyramid.attached_level(0)?;
    let (dtype, volume) = source.metadata()?;
    let detector_config: DetectorConfig =
        serde_json::from_slice(&fs::read(&args.model_config).map_err(Error::backend)?)
            .map_err(Error::backend)?;
    let device = match args.device {
        DeviceChoice::Cpu => InferenceDevice::Cpu,
        DeviceChoice::Cuda => InferenceDevice::Cuda(args.cuda_device),
    };
    let detector = Yolo3dDetector::load(
        detector_config,
        &args.checkpoint,
        device,
        [args.halo_z, args.halo, args.halo],
        (args.normalize_low, args.normalize_high),
        args.threshold,
        args.nms_iou,
    )?;
    let schema = Yolo3dDetector::schema()?;
    let grid = BlockGrid::new(volume, [args.block_z, args.block, args.block])?;
    let mut builder = PlanBuilder::new(volume, dtype, grid);
    let phase = builder.fragments(detector)?;
    let assembly = builder.finish()?;
    let work = args
        .zarr
        .join("tables")
        .join(format!(".{}-blockflow-work", args.layer));
    let destination = args.zarr.join("tables").join(&args.layer);
    if destination.exists() {
        if !args.overwrite {
            return Err(Error::invalid(format!(
                "{} exists; pass --overwrite to replace it",
                destination.display()
            )));
        }
        fs::remove_dir_all(&destination).map_err(Error::backend)?;
    }
    if work.exists() {
        fs::remove_dir_all(&work).map_err(Error::backend)?;
    }
    let env = ZarrEnvironment::attach(&work, &[source])?;
    execute_phases(
        "YOLO3D OME-Zarr",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints {
            concurrency: args.workers.max(1),
            ..Hints::default()
        },
        &env,
        &[],
        &assembly.work(),
    )?;
    let temporary = work.join("table");
    let table = finalize_detections(&env, phase.index(), volume, &temporary, &work)?;
    let detections = table.spec().row_count;
    write_compatibility_csv(&table, &temporary.join("table.csv"))?;
    drop(table);
    drop(env);
    fs::create_dir_all(destination.parent().expect("table has parent")).map_err(Error::backend)?;
    fs::rename(&temporary, &destination).map_err(Error::backend)?;
    fs::remove_dir_all(&work).map_err(Error::backend)?;
    println!(
        "detections={} elapsed_seconds={:.3} table={} columns={}",
        detections,
        started.elapsed().as_secs_f64(),
        destination.display(),
        schema.len()
    );
    Ok(())
}

fn write_compatibility_csv(
    table: &ngff_object_table::TableReader,
    path: &std::path::Path,
) -> Result<()> {
    let rows = 0..table.spec().row_count;
    let ids = table
        .read_u64("detection_id", rows.clone())
        .map_err(Error::backend)?;
    let z = table.read_f32("z", rows.clone()).map_err(Error::backend)?;
    let y = table.read_f32("y", rows.clone()).map_err(Error::backend)?;
    let x = table.read_f32("x", rows.clone()).map_err(Error::backend)?;
    let confidence = table
        .read_f32("confidence", rows.clone())
        .map_err(Error::backend)?;
    let classes = table
        .read_u32("class", rows.clone())
        .map_err(Error::backend)?;
    let z0 = table.read_f32("z0", rows.clone()).map_err(Error::backend)?;
    let y0 = table.read_f32("y0", rows.clone()).map_err(Error::backend)?;
    let x0 = table.read_f32("x0", rows.clone()).map_err(Error::backend)?;
    let z1 = table.read_f32("z1", rows.clone()).map_err(Error::backend)?;
    let y1 = table.read_f32("y1", rows.clone()).map_err(Error::backend)?;
    let x1 = table.read_f32("x1", rows).map_err(Error::backend)?;
    let file = fs::File::create(path).map_err(Error::backend)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "detection_id,z,y,x,confidence,class,z0,y0,x0,z1,y1,x1"
    )
    .map_err(Error::backend)?;
    for row in 0..ids.len() {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            ids[row],
            z[row],
            y[row],
            x[row],
            confidence[row],
            classes[row],
            z0[row],
            y0[row],
            x0[row],
            z1[row],
            y1[row],
            x1[row]
        )
        .map_err(Error::backend)?;
    }
    writer.flush().map_err(Error::backend)
}
