use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use blockflow::ome_zarr::{ensure_zarr_group, OmeVolumePyramid};
use blockflow::strategy::{execute_phases, Hints};
use blockflow::yolo3d::{
    finalize_detections, InferenceDevice, SliceFuseProfiler, YoloSliceFuse3dDetector,
};
use blockflow::{BlockGrid, Environment, Error, PlanBuilder, Result, ZarrEnvironment};
use clap::{Parser, ValueEnum};
use yolo3d::SliceFuseConfig;

#[derive(Debug, Parser)]
#[command(about = "Run YOLO-SliceFuse3D over an OME-Zarr volume")]
struct Args {
    #[arg(long)]
    zarr: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    /// SliceFuseConfig JSON saved beside the checkpoint by training.
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
    #[arg(long, default_value = "yolo-slicefuse3d-cell-centers-v1")]
    layer: String,
    #[arg(long, default_value_t = false)]
    overwrite: bool,
    /// Synchronize CUDA stages and print a detailed timing profile. Profiling
    /// changes execution boundaries and is not a normal speed measurement.
    #[arg(long, default_value_t = false)]
    profile: bool,
    /// Retain frozen 2D encoder features on the device and reuse Z-overlap.
    #[arg(long, default_value_t = false)]
    feature_cache: bool,
    /// Precompute every detector-aligned XY window across the complete Z axis
    /// in a separate Blockflow pass before running the 3D heads.
    #[arg(long, default_value_t = false, conflicts_with = "feature_cache")]
    precompute_features: bool,
    /// Number of adjacent XY windows encoded together by the precompute pass.
    #[arg(long, default_value_t = 1, requires = "precompute_features")]
    precompute_spatial_batch: usize,
    /// Read all twelve head tensors with one backend transaction.
    #[arg(long, default_value_t = false)]
    transaction_download: bool,
    /// Run the model in FP16 while keeping host decoding in FP32.
    #[arg(long, default_value_t = false)]
    fp16: bool,
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
    if args.precompute_spatial_batch == 0 {
        return Err(Error::invalid("precompute spatial batch must be positive"));
    }
    let started = Instant::now();
    let pyramid = OmeVolumePyramid::open(&args.zarr, args.channel, args.time)?;
    let image_root = pyramid.image_root();
    let source = pyramid.attached_level(0)?;
    let (dtype, volume) = source.metadata()?;
    let detector_config: SliceFuseConfig =
        serde_json::from_slice(&fs::read(&args.model_config).map_err(Error::backend)?)
            .map_err(Error::backend)?;
    let device = match args.device {
        DeviceChoice::Cpu => InferenceDevice::Cpu,
        DeviceChoice::Cuda => InferenceDevice::Cuda(args.cuda_device),
    };
    let model_load_started = Instant::now();
    let mut detector = YoloSliceFuse3dDetector::load(
        detector_config,
        &args.checkpoint,
        device,
        [args.halo_z, args.halo, args.halo],
        (args.normalize_low, args.normalize_high),
        args.threshold,
        args.nms_iou,
    )?;
    let model_load = model_load_started.elapsed();
    let profiler = args.profile.then(|| Arc::new(SliceFuseProfiler::default()));
    if let Some(profiler) = &profiler {
        detector = detector.with_profiler(profiler.clone());
    }
    if args.transaction_download {
        detector = detector.with_transaction_download();
    }
    if args.fp16 {
        detector = detector.with_fp16();
    }
    if args.feature_cache {
        detector = detector.with_feature_cache();
    }
    let schema = YoloSliceFuse3dDetector::schema()?;
    let grid = BlockGrid::new(volume, [args.block_z, args.block, args.block])?;
    let precompute = if args.precompute_features {
        let (encoder, cached_detector) =
            detector.with_precomputed_feature_cache([args.block, args.block]);
        detector = cached_detector;
        let mut builder = PlanBuilder::new(
            volume,
            dtype,
            BlockGrid::new(
                volume,
                [
                    volume[0],
                    args.block,
                    args.block
                        .checked_mul(args.precompute_spatial_batch)
                        .ok_or_else(|| Error::invalid("precompute spatial batch is too large"))?,
                ],
            )?,
        );
        builder.fragments(encoder)?;
        Some(builder.finish()?)
    } else {
        None
    };
    let mut builder = PlanBuilder::new(volume, dtype, grid);
    let phase = builder.fragments(detector)?;
    let assembly = builder.finish()?;
    let work = image_root
        .join("tables")
        .join(format!(".{}-blockflow-work", args.layer));
    let destination = image_root.join("tables").join(&args.layer);
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
    let inference_started = Instant::now();
    if let Some(precompute) = &precompute {
        execute_phases(
            "YOLO-SliceFuse3D feature precompute",
            &precompute.workflow,
            &precompute.decomposition,
            &Hints {
                concurrency: args.workers.max(1),
                ..Hints::default()
            },
            &env,
            &[],
            &precompute.work(),
        )?;
    }
    execute_phases(
        "YOLO-SliceFuse3D OME-Zarr",
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
    let inference = inference_started.elapsed();
    let (reads, _, read_voxels, _, chunks_read, _, peak_resident_bytes) = env.counters().snapshot();
    let finalize_started = Instant::now();
    let temporary = work.join("table");
    let table = finalize_detections(&env, phase.index(), volume, &temporary, &work)?;
    let detections = table.spec().row_count;
    write_compatibility_csv(&table, &temporary.join("table.csv"))?;
    let finalize = finalize_started.elapsed();
    drop(table);
    drop(env);
    ensure_zarr_group(destination.parent().expect("table has parent"))?;
    fs::rename(&temporary, &destination).map_err(Error::backend)?;
    fs::remove_dir_all(&work).map_err(Error::backend)?;
    println!(
        "detections={} elapsed_seconds={:.3} table={} columns={}",
        detections,
        started.elapsed().as_secs_f64(),
        destination.display(),
        schema.len()
    );
    if let Some(profiler) = profiler {
        let profile = profiler.snapshot();
        let seconds = |duration: std::time::Duration| duration.as_secs_f64();
        println!(
            "profile_json={}",
            serde_json::json!({
                "model_load_seconds": seconds(model_load),
                "inference_seconds": seconds(inference),
                "finalize_seconds": seconds(finalize),
                "blocks": profile.blocks,
                "input_voxels": profile.input_voxels,
                "unique_voxels": volume.iter().product::<usize>(),
                "input_redundancy": profile.input_voxels as f64 / volume.iter().product::<usize>() as f64,
                "environment_reads": reads,
                "environment_read_voxels": read_voxels,
                "environment_chunks_read": chunks_read,
                "peak_resident_bytes": peak_resident_bytes,
                "tensor_downloads": profile.tensor_downloads,
                "preprocess_seconds": seconds(profile.preprocess),
                "upload_seconds": seconds(profile.upload),
                "model_lock_wait_seconds": seconds(profile.model_lock_wait),
                "encoder_seconds": seconds(profile.encoder),
                "fusion_and_heads_seconds": seconds(profile.fusion_and_heads),
                "download_seconds": seconds(profile.download),
                "decode_seconds": seconds(profile.decode),
                "nms_seconds": seconds(profile.nms),
                "rows_seconds": seconds(profile.rows),
                "feature_cache_hits": profile.feature_cache_hits,
                "feature_cache_misses": profile.feature_cache_misses,
                "feature_cache_entries": profile.feature_cache_entries,
            })
        );
    }
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
