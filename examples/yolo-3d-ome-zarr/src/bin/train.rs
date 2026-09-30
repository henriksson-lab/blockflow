use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::bail;
use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::{AttachedImage, Environment, ImageId, Region, ZarrEnvironment};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::{momentum::MomentumConfig, GradientsParams, SgdConfig};
use burn::prelude::{Device, Tensor};
#[cfg(feature = "cuda")]
use burn::tensor::DeviceIndex;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use yolo3d::{
    decode_scale, detector_loss, encode_targets, nms, Box3d, Detector, DetectorConfig,
    EncodedTargets, HistoryRow, LossWeights, ScaleTargets, TeacherObject, TrainingHistory,
};

#[derive(Debug, Parser)]
#[command(about = "Distill a 3D center detector from an OME-Zarr label volume")]
struct Args {
    #[arg(long)]
    zarr: PathBuf,
    /// Label layer containing the volumetric Cellpose or StarDist teacher.
    #[arg(long, default_value = "cellpose3d-dapi")]
    teacher: String,
    #[arg(long, default_value_t = 0)]
    channel: usize,
    #[arg(long, default_value_t = 0)]
    time: usize,
    /// Physical z,y,x voxel sizes in micrometers. Defaults to OME level-0 scale.
    #[arg(long, value_delimiter = ',')]
    voxel_size: Option<Vec<f32>>,
    #[arg(long, default_value_t = 32)]
    patch_z: usize,
    #[arg(long, default_value_t = 256)]
    patch: usize,
    #[arg(long, default_value_t = 4)]
    ownership_halo_z: usize,
    #[arg(long, default_value_t = 24)]
    ownership_halo: usize,
    #[arg(long, default_value_t = 20)]
    epochs: usize,
    /// Evaluate this checkpoint without training. The validation partition is
    /// used unless --evaluation-partition test is explicitly requested.
    #[arg(long)]
    evaluate_checkpoint: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = EvaluationPartition::Validation)]
    evaluation_partition: EvaluationPartition,
    /// Maximum ranked local maxima retained per ownership core for metrics.
    #[arg(long, default_value_t = 256)]
    evaluation_top_k: usize,
    #[arg(long, default_value_t = 0.3)]
    evaluation_nms_iou: f32,
    /// Additional epochs start after an existing run's last committed epoch.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Start a new history and dataset contract from model weights only.
    #[arg(long)]
    initialize_checkpoint: Option<PathBuf>,
    /// Reuse or hand-edit a split.json to select exact train, validation, and
    /// frozen test windows. Its volume, patch, and halo must match this run.
    #[arg(long)]
    split: Option<PathBuf>,
    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,
    /// Nesterov SGD momentum, matching the established 2D YOLO training path.
    #[arg(long, default_value_t = 0.937)]
    momentum: f64,
    #[arg(long, default_value_t = 10.0)]
    gradient_clip: f32,
    #[arg(long, default_value_t = 0.0)]
    normalize_low: f32,
    #[arg(long, default_value_t = 255.0)]
    normalize_high: f32,
    /// Dataset augmentation policy. Fluorescence enables D4 and z reflection;
    /// transmitted keeps lens orientation fixed.
    #[arg(long, value_enum, default_value_t = AugmentationChoice::Fluorescence)]
    augmentation: AugmentationChoice,
    #[arg(long, default_value_t = 4 << 30)]
    cache_bytes: u64,
    #[arg(long, value_enum, default_value_t = DeviceChoice::Cpu)]
    device: DeviceChoice,
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    #[arg(long, default_value = ".tmp/yolo3d-training")]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DeviceChoice {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum AugmentationChoice {
    None,
    Fluorescence,
    Transmitted,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum EvaluationPartition {
    Validation,
    Test,
}

#[derive(Clone, Debug, Serialize)]
struct DetectionMetric {
    ap: f32,
    best_threshold: f32,
    precision: f32,
    recall: f32,
    f1: f32,
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
}

#[derive(Clone, Debug, Serialize)]
struct EvaluationReport {
    partition: String,
    tiles: usize,
    teacher_objects: usize,
    raw_local_maxima: usize,
    retained_before_nms: usize,
    predictions_after_nms: usize,
    top_k_per_tile: usize,
    nms_iou: f32,
    validation_loss: f32,
    /// A center is correct within half the teacher box diagonal.
    center: DetectionMetric,
    center_operating_points: Vec<OperatingPoint>,
    box_ap50: f32,
    box_diagnostics_at_center_threshold: BoxDiagnostics,
}

#[derive(Clone, Debug, Serialize)]
struct OperatingPoint {
    threshold: f32,
    precision: f32,
    recall: f32,
    f1: f32,
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
}

#[derive(Clone, Debug, Serialize)]
struct BoxDiagnostics {
    center_matches: usize,
    mean_iou: f32,
    mean_size_ratio_zyx: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
struct Tile {
    start: [usize; 3],
    core_start: [usize; 3],
    core_end: [usize; 3],
}

#[derive(Debug, Deserialize)]
struct SplitManifest {
    volume: [usize; 3],
    patch: [usize; 3],
    ownership_halo: [usize; 3],
    train: Vec<[usize; 3]>,
    validation: Vec<[usize; 3]>,
    test: Vec<[usize; 3]>,
}

fn main() -> anyhow::Result<()> {
    if cfg!(debug_assertions) {
        bail!("this example requires cargo run --release");
    }
    let args = Args::parse();
    validate(&args)?;
    fs::create_dir_all(&args.output)?;
    let pyramid = OmeVolumePyramid::open(&args.zarr, args.channel, args.time)?;
    let image = pyramid.attached_level(0)?;
    let labels = AttachedImage::at(args.zarr.join("labels").join(&args.teacher).join("0"));
    let (_, volume) = image.metadata()?;
    let (_, label_volume) = labels.metadata()?;
    anyhow::ensure!(
        volume == label_volume,
        "image and teacher label shapes differ"
    );
    let voxel_size = args
        .voxel_size
        .as_ref()
        .map(|values| <[f32; 3]>::try_from(values.clone()))
        .transpose()
        .map_err(|values| anyhow::anyhow!("--voxel-size wants z,y,x; got {values:?}"))?
        .unwrap_or_else(|| pyramid.voxel_size().map(|value| value as f32));
    let mut detector_config = DetectorConfig::for_spacing(1, 1, voxel_size);
    let patch = [args.patch_z, args.patch, args.patch];
    let halo = [
        args.ownership_halo_z,
        args.ownership_halo,
        args.ownership_halo,
    ];
    let tiles = tiles(volume, patch, halo);
    let (train, validation, test) = if let Some(path) = &args.split {
        load_split(path, &tiles, volume, patch, halo)?
    } else {
        split_tiles(&tiles, volume, patch)
    };
    anyhow::ensure!(
        !train.is_empty() && !validation.is_empty(),
        "volume is too small for the requested spatial split and patch"
    );
    let split = serde_json::json!({
        "volume": volume,
        "patch": patch,
        "ownership_halo": halo,
        "augmentation": format!("{:?}", args.augmentation).to_lowercase(),
        "train": train.iter().map(|tile| tile.start).collect::<Vec<_>>(),
        "validation": validation.iter().map(|tile| tile.start).collect::<Vec<_>>(),
        "test": test.iter().map(|tile| tile.start).collect::<Vec<_>>(),
    });
    let split_bytes = serde_json::to_vec_pretty(&split)?;
    fs::write(args.output.join("split.json"), &split_bytes)?;
    let work = args.output.join("zarr-work");
    let env = ZarrEnvironment::attach(&work, &[image, labels])?.with_cache(args.cache_bytes);
    detector_config.size_priors = estimate_size_priors(
        &env,
        &train,
        patch,
        detector_config.cumulative_strides(),
        voxel_size,
    )?;
    fs::write(
        args.output.join("model-config.json"),
        serde_json::to_vec_pretty(&detector_config)?,
    )?;
    fs::write(
        args.output.join("optimizer-config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "name": "sgd",
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "nesterov": true,
            "gradient_clip_norm": args.gradient_clip,
        }))?,
    )?;
    let contract = format!(
        "yolo3d-v1:{}:{}",
        serde_json::to_string(&detector_config)?,
        simple_hash(&split_bytes)
    );
    let device = device(&args)?;
    let mut model = Detector::new(&detector_config, &device)?;
    let mut history = TrainingHistory {
        run_contract: contract.clone(),
        rows: Vec::new(),
    };
    if let Some(run) = &args.resume {
        let prior: TrainingHistory = serde_json::from_slice(&fs::read(run.join("history.json"))?)?;
        anyhow::ensure!(
            prior.run_contract == contract,
            "resume run uses a different model, split, or target geometry"
        );
        model = model.try_load_file(run.join("last.bpk"))?;
        history = prior;
        let prior_best = run.join("best.bpk");
        let next_best = args.output.join("best.bpk");
        if prior_best.is_file() && prior_best != next_best {
            fs::copy(prior_best, next_best)?;
        }
    }
    if let Some(checkpoint) = &args.initialize_checkpoint {
        model = model.try_load_file(checkpoint)?;
    }
    if let Some(checkpoint) = &args.evaluate_checkpoint {
        model = model.try_load_file(checkpoint)?;
        let (partition, evaluation_tiles) = match args.evaluation_partition {
            EvaluationPartition::Validation => ("validation", validation.as_slice()),
            EvaluationPartition::Test => ("test", test.as_slice()),
        };
        anyhow::ensure!(
            !evaluation_tiles.is_empty(),
            "{partition} partition is empty"
        );
        let report = evaluate(
            &model,
            &env,
            evaluation_tiles,
            patch,
            detector_config.cumulative_strides(),
            voxel_size,
            (args.normalize_low, args.normalize_high),
            &device,
            partition,
            args.evaluation_top_k,
            args.evaluation_nms_iou,
        )?;
        let report_bytes = serde_json::to_vec_pretty(&report)?;
        fs::write(
            args.output.join(format!("evaluation-{partition}.json")),
            &report_bytes,
        )?;
        println!("{}", String::from_utf8(report_bytes)?);
        return Ok(());
    }
    let start_epoch = history.rows.last().map_or(0, |row| row.epoch + 1);
    let mut optimizer = SgdConfig::new()
        .with_momentum(Some(MomentumConfig {
            momentum: args.momentum,
            dampening: 0.0,
            nesterov: true,
        }))
        .with_gradient_clipping(Some(GradientClippingConfig::Norm(args.gradient_clip)))
        .init();
    let strides = detector_config.cumulative_strides();
    let mut best_center_ap = history
        .rows
        .iter()
        .filter_map(|row| row.center_ap)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut best_validation = history
        .rows
        .iter()
        .filter(|row| row.center_ap == Some(best_center_ap))
        .map(|row| row.validation_loss)
        .fold(f32::INFINITY, f32::min);
    let training_started = Instant::now();
    for epoch in start_epoch..start_epoch + args.epochs {
        let mut train_loss = 0.0f64;
        let order = locality_order(train.len(), epoch);
        for index in order {
            let allow_geometry = (0..3).all(|axis| {
                train[index].start[axis] > 0
                    && train[index].start[axis] + patch[axis] < volume[axis]
            });
            let (input, objects) = sample(
                &env,
                train[index],
                patch,
                args.normalize_low,
                args.normalize_high,
                &device,
                transform(args.augmentation, epoch, index, allow_geometry),
            )?;
            let targets = targets(&objects, patch, strides, voxel_size, &device)?;
            let output = model.forward(input);
            let loss = detector_loss(&output, &targets, LossWeights::default())?;
            let value = scalar(&loss.total)?;
            anyhow::ensure!(
                value.is_finite(),
                "non-finite loss at epoch {epoch}, sample {index}"
            );
            let gradients = GradientsParams::from_grads(loss.total.backward(), &model);
            model = optimizer.step(args.learning_rate, model, gradients);
            train_loss += value as f64;
        }
        let validation_report = evaluate(
            &model,
            &env,
            &validation,
            patch,
            strides,
            voxel_size,
            (args.normalize_low, args.normalize_high),
            &device,
            "validation",
            args.evaluation_top_k,
            args.evaluation_nms_iou,
        )?;
        let row = HistoryRow {
            epoch,
            train_loss: (train_loss / train.len() as f64) as f32,
            validation_loss: validation_report.validation_loss,
            center_ap: Some(validation_report.center.ap),
            box_ap50: Some(validation_report.box_ap50),
            learning_rate: args.learning_rate,
            elapsed_seconds: training_started.elapsed().as_secs_f64(),
        };
        history.rows.push(row);
        fs::write(
            args.output.join("evaluation-validation.json"),
            serde_json::to_vec_pretty(&validation_report)?,
        )?;
        model.clone().save_file(args.output.join("last.bpk"))?;
        if validation_report.center.ap > best_center_ap
            || (validation_report.center.ap == best_center_ap
                && validation_report.validation_loss < best_validation)
        {
            best_center_ap = validation_report.center.ap;
            best_validation = validation_report.validation_loss;
            model.clone().save_file(args.output.join("best.bpk"))?;
            fs::write(
                args.output.join("evaluation-best-validation.json"),
                serde_json::to_vec_pretty(&validation_report)?,
            )?;
        }
        fs::write(
            args.output.join("history.json"),
            serde_json::to_vec_pretty(&history)?,
        )?;
        write_history_svg(&args.output.join("training-progress.svg"), &history)?;
        let row = history.rows.last().expect("just pushed");
        println!(
            "epoch={} train_loss={:.6} validation_loss={:.6} center_ap={:.4} box_ap50={:.4} threshold={:.4} elapsed_seconds={:.1}",
            row.epoch,
            row.train_loss,
            row.validation_loss,
            validation_report.center.ap,
            validation_report.box_ap50,
            validation_report.center.best_threshold,
            row.elapsed_seconds
        );
    }
    println!(
        "training_complete epochs={} train_tiles={} validation_tiles={} frozen_test_tiles={} output={}",
        history.rows.len(),
        train.len(),
        validation.len(),
        test.len(),
        args.output.display()
    );
    Ok(())
}

fn validate(args: &Args) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.patch_z > 2 * args.ownership_halo_z,
        "patch-z must exceed twice ownership-halo-z"
    );
    anyhow::ensure!(
        args.patch > 2 * args.ownership_halo,
        "patch must exceed twice ownership-halo"
    );
    anyhow::ensure!(
        (args.epochs > 0 || args.evaluate_checkpoint.is_some())
            && args.learning_rate > 0.0
            && (0.0..1.0).contains(&args.momentum)
            && args.gradient_clip.is_finite()
            && args.gradient_clip > 0.0,
        "epochs must be positive unless evaluating a checkpoint; learning rate and gradient clip must be positive, and momentum must be in [0, 1)"
    );
    anyhow::ensure!(
        [
            args.resume.is_some(),
            args.initialize_checkpoint.is_some(),
            args.evaluate_checkpoint.is_some(),
        ]
        .into_iter()
        .filter(|selected| *selected)
        .count()
            <= 1,
        "--resume, --initialize-checkpoint, and --evaluate-checkpoint are mutually exclusive"
    );
    anyhow::ensure!(
        args.evaluation_top_k > 0,
        "evaluation-top-k must be positive"
    );
    anyhow::ensure!(
        (0.0..=1.0).contains(&args.evaluation_nms_iou),
        "evaluation-nms-iou must be between zero and one"
    );
    anyhow::ensure!(
        args.normalize_high > args.normalize_low,
        "normalization high must exceed low"
    );
    Ok(())
}

fn device(args: &Args) -> anyhow::Result<Device> {
    let device = match args.device {
        DeviceChoice::Cpu => Device::flex(),
        DeviceChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Device::cuda(DeviceIndex::new(args.cuda_device))
            }
            #[cfg(not(feature = "cuda"))]
            bail!("CUDA training requires --features cuda")
        }
    };
    Ok(device.autodiff())
}

fn tiles(volume: [usize; 3], patch: [usize; 3], halo: [usize; 3]) -> Vec<Tile> {
    if (0..3).any(|axis| volume[axis] < patch[axis]) {
        return Vec::new();
    }
    let axes =
        [0, 1, 2].map(|axis| axis_tiles(volume[axis], patch[axis], patch[axis] - 2 * halo[axis]));
    let mut tiles = Vec::new();
    for &(z, z0, z1) in &axes[0] {
        for &(y, y0, y1) in &axes[1] {
            for &(x, x0, x1) in &axes[2] {
                let start = [z, y, x];
                tiles.push(Tile {
                    start,
                    core_start: [z0, y0, x0],
                    core_end: [z1, y1, x1],
                });
            }
        }
    }
    tiles
}

fn axis_tiles(length: usize, patch: usize, step: usize) -> Vec<(usize, usize, usize)> {
    let mut starts = (0..=length - patch).step_by(step).collect::<Vec<_>>();
    let last = length - patch;
    if starts.last().copied() != Some(last) {
        starts.push(last);
    }
    starts
        .iter()
        .enumerate()
        .map(|(index, &start)| {
            let core_start = if index == 0 {
                0
            } else {
                (starts[index - 1] + patch + start) / 2
            };
            let core_end = if index + 1 == starts.len() {
                length
            } else {
                (start + patch + starts[index + 1]) / 2
            };
            (start, core_start, core_end)
        })
        .collect()
}

fn split_tiles(
    tiles: &[Tile],
    volume: [usize; 3],
    patch: [usize; 3],
) -> (Vec<Tile>, Vec<Tile>, Vec<Tile>) {
    let validation_start = volume[1] * 70 / 100;
    let test_start = volume[1] * 86 / 100;
    let guard = patch[1];
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();
    for &tile in tiles {
        let center = tile.start[1] + patch[1] / 2;
        if center + guard <= validation_start {
            train.push(tile);
        } else if center >= validation_start + guard && center + guard <= test_start {
            validation.push(tile);
        } else if center >= test_start + guard {
            test.push(tile);
        }
    }
    (train, validation, test)
}

fn load_split(
    path: &PathBuf,
    tiles: &[Tile],
    volume: [usize; 3],
    patch: [usize; 3],
    halo: [usize; 3],
) -> anyhow::Result<(Vec<Tile>, Vec<Tile>, Vec<Tile>)> {
    let manifest: SplitManifest = serde_json::from_slice(&fs::read(path)?)?;
    anyhow::ensure!(
        manifest.volume == volume && manifest.patch == patch && manifest.ownership_halo == halo,
        "split manifest volume, patch, or ownership halo does not match this run"
    );
    let mut available = tiles
        .iter()
        .map(|tile| (tile.start, *tile))
        .collect::<BTreeMap<_, _>>();
    let mut resolve = |name: &str, starts: Vec<[usize; 3]>| -> anyhow::Result<Vec<Tile>> {
        starts
            .into_iter()
            .map(|start| {
                available.remove(&start).ok_or_else(|| {
                    anyhow::anyhow!(
                        "{name} split start {start:?} is duplicated or is not a generated tile"
                    )
                })
            })
            .collect()
    };
    let train = resolve("train", manifest.train)?;
    let validation = resolve("validation", manifest.validation)?;
    let test = resolve("test", manifest.test)?;
    ensure_partitions_do_not_overlap(&train, &validation, patch, "train", "validation")?;
    ensure_partitions_do_not_overlap(&train, &test, patch, "train", "test")?;
    ensure_partitions_do_not_overlap(&validation, &test, patch, "validation", "test")?;
    Ok((train, validation, test))
}

fn ensure_partitions_do_not_overlap(
    left: &[Tile],
    right: &[Tile],
    patch: [usize; 3],
    left_name: &str,
    right_name: &str,
) -> anyhow::Result<()> {
    for a in left {
        for b in right {
            let overlaps = (0..3).all(|axis| {
                a.start[axis] < b.start[axis] + patch[axis]
                    && b.start[axis] < a.start[axis] + patch[axis]
            });
            anyhow::ensure!(
                !overlaps,
                "{left_name} tile {:?} overlaps {right_name} tile {:?}",
                a.start,
                b.start
            );
        }
    }
    Ok(())
}

fn locality_order(length: usize, epoch: usize) -> Vec<usize> {
    const GROUP: usize = 8;
    let groups = length.div_ceil(GROUP);
    let mut group_order = (0..groups).collect::<Vec<_>>();
    let mut state = epoch as u64 ^ 0x9e37_79b9_7f4a_7c15;
    for end in (1..group_order.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        group_order.swap(end, state as usize % (end + 1));
    }
    group_order
        .into_iter()
        .flat_map(|group| {
            let start = group * GROUP;
            start..(start + GROUP).min(length)
        })
        .collect()
}

fn estimate_size_priors(
    env: &ZarrEnvironment,
    tiles: &[Tile],
    patch: [usize; 3],
    strides: [[usize; 3]; 3],
    voxel_size: [f32; 3],
) -> anyhow::Result<[[f32; 3]; 3]> {
    let volume = env.volume();
    let mut extents_by_id = BTreeMap::<u64, [usize; 3]>::new();
    for &tile in tiles {
        let region = Region::new(&tile.start, &patch);
        let labels_buf = env.read(ImageId::supplied(0).index(), &region)?;
        let labels = labels_buf.as_array()?.view::<u64>()?;
        let mut accum = BTreeMap::<u64, ([usize; 3], [usize; 3], bool)>::new();
        for z in 0..patch[0] {
            for y in 0..patch[1] {
                for x in 0..patch[2] {
                    let label = labels[[z, y, x]];
                    if label == 0 {
                        continue;
                    }
                    let entry =
                        accum
                            .entry(label)
                            .or_insert(([z, y, x], [z + 1, y + 1, x + 1], false));
                    for (axis, value) in [z, y, x].into_iter().enumerate() {
                        entry.0[axis] = entry.0[axis].min(value);
                        entry.1[axis] = entry.1[axis].max(value + 1);
                        entry.2 |= (value == 0 && tile.start[axis] > 0)
                            || (value + 1 == patch[axis]
                                && tile.start[axis] + patch[axis] < volume[axis]);
                    }
                }
            }
        }
        for (id, (start, end, truncated)) in accum {
            if !truncated {
                extents_by_id
                    .entry(id)
                    .or_insert_with(|| [0, 1, 2].map(|axis| end[axis] - start[axis]));
            }
        }
    }
    anyhow::ensure!(
        !extents_by_id.is_empty(),
        "training windows contain no complete teacher objects"
    );
    let mut by_level = [Vec::<[f32; 3]>::new(), Vec::new(), Vec::new()];
    for extent in extents_by_id.into_values() {
        let level = assigned_level(extent.map(|value| value as f32), strides);
        by_level[level].push([0, 1, 2].map(|axis| extent[axis] as f32 * voxel_size[axis]));
    }
    let all = by_level.iter().flatten().copied().collect::<Vec<_>>();
    let fallback = median_extent(&all);
    Ok(by_level.map(|values| {
        if values.is_empty() {
            fallback
        } else {
            median_extent(&values)
        }
    }))
}

fn median_extent(values: &[[f32; 3]]) -> [f32; 3] {
    [0, 1, 2].map(|axis| {
        let mut axis_values = values.iter().map(|value| value[axis]).collect::<Vec<_>>();
        axis_values.sort_by(f32::total_cmp);
        axis_values[axis_values.len() / 2]
    })
}

fn assigned_level(extent: [f32; 3], strides: [[usize; 3]; 3]) -> usize {
    strides
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            let score = |stride: &&[usize; 3]| {
                let cells = (0..3)
                    .map(|axis| extent[axis] / stride[axis] as f32)
                    .fold(0.0f32, f32::max);
                (cells.max(f32::EPSILON).log2() - 2.0).abs()
            };
            score(left).total_cmp(&score(right))
        })
        .map(|(level, _)| level)
        .unwrap_or(0)
}

fn sample(
    env: &ZarrEnvironment,
    tile: Tile,
    patch: [usize; 3],
    low: f32,
    high: f32,
    device: &Device,
    transform: Transform,
) -> anyhow::Result<(Tensor<5>, Vec<TeacherObject>)> {
    let region = Region::new(&tile.start, &patch);
    let image = env.read(0, &region)?.as_array()?.widened();
    let labels_buf = env.read(ImageId::supplied(0).index(), &region)?;
    let labels = labels_buf.as_array()?.view::<u64>()?;
    let span = (high - low).max(f32::EPSILON);
    let mut values = vec![0.0f32; patch.iter().product()];
    let mut transformed_labels = vec![0u64; values.len()];
    for z in 0..patch[0] {
        for y in 0..patch[1] {
            for x in 0..patch[2] {
                let [tz, ty, tx] = transform.map([z, y, x], patch);
                let flat = (tz * patch[1] + ty) * patch[2] + tx;
                let normalized = ((image[[z, y, x]] as f32 - low) / span).clamp(0.0, 1.0);
                values[flat] =
                    (normalized.powf(transform.gamma) * transform.intensity).clamp(0.0, 1.0);
                transformed_labels[flat] = labels[[z, y, x]];
            }
        }
    }
    let input = Tensor::<1>::from_floats(values.as_slice(), device)
        .reshape([1, 1, patch[0], patch[1], patch[2]]);
    let volume = env.volume();
    let mut accum = BTreeMap::<u64, ([usize; 3], [usize; 3], bool)>::new();
    for z in 0..patch[0] {
        for y in 0..patch[1] {
            for x in 0..patch[2] {
                let label = transformed_labels[(z * patch[1] + y) * patch[2] + x];
                if label == 0 {
                    continue;
                }
                let entry = accum
                    .entry(label)
                    .or_insert(([z, y, x], [z + 1, y + 1, x + 1], false));
                for (axis, value) in [z, y, x].into_iter().enumerate() {
                    entry.0[axis] = entry.0[axis].min(value);
                    entry.1[axis] = entry.1[axis].max(value + 1);
                    entry.2 |= (value == 0 && tile.start[axis] > 0)
                        || (value + 1 == patch[axis]
                            && tile.start[axis] + patch[axis] < volume[axis]);
                }
            }
        }
    }
    let objects = accum
        .into_iter()
        .map(|(id, (start, end, touches_boundary))| TeacherObject {
            id,
            bounds_start: start,
            bounds_end: end,
            class: 0,
            confidence: 1.0,
            truncated: touches_boundary,
        })
        .collect();
    Ok((input, objects))
}

fn targets(
    objects: &[TeacherObject],
    patch: [usize; 3],
    strides: [[usize; 3]; 3],
    voxel_size: [f32; 3],
    device: &Device,
) -> anyhow::Result<Vec<ScaleTargets>> {
    let mut assigned = [Vec::new(), Vec::new(), Vec::new()];
    for object in objects {
        let extent = [0, 1, 2]
            .map(|axis| object.bounds_end[axis].saturating_sub(object.bounds_start[axis]) as f32);
        let level = assigned_level(extent, strides);
        assigned[level].push(object.clone());
    }
    strides
        .iter()
        .enumerate()
        .map(|(level, &stride)| {
            let encoded = encode_targets(&assigned[level], [0; 3], patch, stride, voxel_size, 1)?;
            Ok(target_tensors(encoded, device))
        })
        .collect()
}

fn target_tensors(target: EncodedTargets, device: &Device) -> ScaleTargets {
    let cells = target.shape.iter().product::<usize>();
    let mut offset = vec![0.0; 3 * cells];
    let mut size = vec![1.0; 3 * cells];
    for cell in 0..cells {
        for axis in 0..3 {
            offset[axis * cells + cell] = target.offset[cell][axis];
            size[axis * cells + cell] = target.size[cell][axis].max(f32::EPSILON);
        }
    }
    ScaleTargets {
        heatmap: Tensor::<1>::from_floats(target.heatmap.as_slice(), device).reshape([
            1,
            target.classes,
            target.shape[0],
            target.shape[1],
            target.shape[2],
        ]),
        center_weight: Tensor::<1>::from_floats(target.center_weight.as_slice(), device).reshape([
            1,
            target.classes,
            target.shape[0],
            target.shape[1],
            target.shape[2],
        ]),
        offset: Tensor::<1>::from_floats(offset.as_slice(), device).reshape([
            1,
            3,
            target.shape[0],
            target.shape[1],
            target.shape[2],
        ]),
        size: Tensor::<1>::from_floats(size.as_slice(), device).reshape([
            1,
            3,
            target.shape[0],
            target.shape[1],
            target.shape[2],
        ]),
        regression_weight: Tensor::<1>::from_floats(target.regression_weight.as_slice(), device)
            .reshape([1, 1, target.shape[0], target.shape[1], target.shape[2]]),
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    model: &Detector,
    env: &ZarrEnvironment,
    tiles: &[Tile],
    patch: [usize; 3],
    strides: [[usize; 3]; 3],
    voxel_size: [f32; 3],
    range: (f32, f32),
    device: &Device,
    partition: &str,
    top_k: usize,
    nms_iou: f32,
) -> anyhow::Result<EvaluationReport> {
    let model = model.valid();
    let device = device.clone().inner();
    let mut total = 0.0f64;
    let mut predictions = Vec::new();
    let mut teachers = Vec::new();
    let mut raw_local_maxima = 0usize;
    let mut retained_before_nms = 0usize;
    for &tile in tiles {
        let (input, objects) = sample(
            env,
            tile,
            patch,
            range.0,
            range.1,
            &device,
            Transform::identity(),
        )?;
        let target = targets(&objects, patch, strides, voxel_size, &device)?;
        let output = model.forward(input);
        total += scalar(&detector_loss(&output, &target, LossWeights::default())?.total)? as f64;

        let mut tile_predictions = Vec::new();
        for (level, scale) in output.into_iter().enumerate() {
            let dims = scale.heatmap.dims();
            let shape = [dims[2], dims[3], dims[4]];
            let heatmap = tensor_values(scale.heatmap)?;
            let offset = tensor_values(scale.offset)?;
            let size = tensor_values(scale.size)?;
            let quality = tensor_values(scale.quality)?;
            let decoded = decode_scale(
                &heatmap,
                &offset,
                &size,
                &quality,
                shape,
                1,
                strides[level],
                voxel_size,
                0.0,
            )?;
            raw_local_maxima += decoded.len();
            tile_predictions.extend(decoded.into_iter().filter(|detection| {
                (0..3).all(|axis| {
                    let global_voxel =
                        tile.start[axis] as f32 + detection.center[axis] / voxel_size[axis];
                    global_voxel >= tile.core_start[axis] as f32
                        && global_voxel < tile.core_end[axis] as f32
                })
            }));
        }
        tile_predictions.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
        tile_predictions.truncate(top_k);
        retained_before_nms += tile_predictions.len();
        for mut detection in nms(tile_predictions, nms_iou) {
            for (axis, &spacing) in voxel_size.iter().enumerate() {
                detection.center[axis] += tile.start[axis] as f32 * spacing;
            }
            predictions.push(detection);
        }
        teachers.extend(objects.iter().filter_map(|object| {
            let local_center =
                [0, 1, 2].map(|axis| (object.bounds_start[axis] + object.bounds_end[axis]) / 2);
            let owned = (0..3).all(|axis| {
                let global_center = tile.start[axis] + local_center[axis];
                global_center >= tile.core_start[axis] && global_center < tile.core_end[axis]
            });
            owned
                .then(|| object.target_box(voxel_size))
                .flatten()
                .map(|mut target| {
                    for (axis, &spacing) in voxel_size.iter().enumerate() {
                        target.center[axis] += tile.start[axis] as f32 * spacing;
                    }
                    target
                })
        }));
    }
    predictions.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    let center = detection_metric(&predictions, &teachers, center_match_score);
    let center_operating_points = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        .map(|threshold| operating_point(&predictions, &teachers, threshold, center_match_score))
        .into_iter()
        .collect();
    let box_ap50 = detection_metric(&predictions, &teachers, |prediction, teacher| {
        let iou = prediction.iou(*teacher);
        (iou >= 0.5).then_some(iou)
    })
    .ap;
    let box_diagnostics_at_center_threshold =
        box_diagnostics(&predictions, &teachers, center.best_threshold);
    Ok(EvaluationReport {
        partition: partition.into(),
        tiles: tiles.len(),
        teacher_objects: teachers.len(),
        raw_local_maxima,
        retained_before_nms,
        predictions_after_nms: predictions.len(),
        top_k_per_tile: top_k,
        nms_iou,
        validation_loss: (total / tiles.len() as f64) as f32,
        center,
        center_operating_points,
        box_ap50,
        box_diagnostics_at_center_threshold,
    })
}

fn center_match_score(prediction: &Box3d, teacher: &Box3d) -> Option<f32> {
    let distance = (0..3)
        .map(|axis| (prediction.center[axis] - teacher.center[axis]).powi(2))
        .sum::<f32>()
        .sqrt();
    let tolerance = 0.5
        * teacher
            .size
            .iter()
            .map(|value| value.powi(2))
            .sum::<f32>()
            .sqrt();
    (distance <= tolerance).then_some(1.0 - distance / tolerance.max(f32::EPSILON))
}

fn operating_point(
    predictions: &[Box3d],
    teachers: &[Box3d],
    threshold: f32,
    matches: impl Fn(&Box3d, &Box3d) -> Option<f32>,
) -> OperatingPoint {
    let mut claimed = vec![false; teachers.len()];
    let mut true_positives = 0usize;
    let mut predictions_at_threshold = 0usize;
    for prediction in predictions
        .iter()
        .take_while(|prediction| prediction.confidence >= threshold)
    {
        predictions_at_threshold += 1;
        let best = teachers
            .iter()
            .enumerate()
            .filter(|(index, teacher)| !claimed[*index] && prediction.class == teacher.class)
            .filter_map(|(index, teacher)| matches(prediction, teacher).map(|score| (index, score)))
            .max_by(|left, right| left.1.total_cmp(&right.1));
        if let Some((index, _)) = best {
            claimed[index] = true;
            true_positives += 1;
        }
    }
    let precision = true_positives as f32 / predictions_at_threshold.max(1) as f32;
    let recall = true_positives as f32 / teachers.len().max(1) as f32;
    OperatingPoint {
        threshold,
        precision,
        recall,
        f1: 2.0 * precision * recall / (precision + recall).max(f32::EPSILON),
        true_positives,
        false_positives: predictions_at_threshold - true_positives,
        false_negatives: teachers.len() - true_positives,
    }
}

fn tensor_values<const D: usize>(tensor: Tensor<D>) -> anyhow::Result<Vec<f32>> {
    tensor
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|error| anyhow::anyhow!("reading detector output: {error}"))
}

fn detection_metric(
    predictions: &[Box3d],
    teachers: &[Box3d],
    matches: impl Fn(&Box3d, &Box3d) -> Option<f32>,
) -> DetectionMetric {
    if teachers.is_empty() {
        return DetectionMetric {
            ap: 0.0,
            best_threshold: 1.0,
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            true_positives: 0,
            false_positives: 0,
            false_negatives: 0,
        };
    }
    let mut claimed = vec![false; teachers.len()];
    let mut true_positive = Vec::with_capacity(predictions.len());
    for prediction in predictions {
        let best = teachers
            .iter()
            .enumerate()
            .filter(|(index, teacher)| !claimed[*index] && prediction.class == teacher.class)
            .filter_map(|(index, teacher)| matches(prediction, teacher).map(|score| (index, score)))
            .max_by(|left, right| left.1.total_cmp(&right.1));
        if let Some((index, _)) = best {
            claimed[index] = true;
            true_positive.push(true);
        } else {
            true_positive.push(false);
        }
    }

    let mut cumulative_tp = 0usize;
    let mut precision = Vec::with_capacity(predictions.len());
    let mut recall = Vec::with_capacity(predictions.len());
    let mut best_index = None;
    let mut best_f1 = 0.0f32;
    for (index, &is_tp) in true_positive.iter().enumerate() {
        cumulative_tp += usize::from(is_tp);
        let p = cumulative_tp as f32 / (index + 1) as f32;
        let r = cumulative_tp as f32 / teachers.len() as f32;
        let f1 = 2.0 * p * r / (p + r).max(f32::EPSILON);
        precision.push(p);
        recall.push(r);
        if f1 > best_f1 {
            best_f1 = f1;
            best_index = Some(index);
        }
    }

    let mut precision_envelope = precision.clone();
    for index in (0..precision_envelope.len().saturating_sub(1)).rev() {
        precision_envelope[index] = precision_envelope[index].max(precision_envelope[index + 1]);
    }
    let mut ap = 0.0f32;
    let mut prior_recall = 0.0f32;
    for index in 0..predictions.len() {
        if true_positive[index] {
            ap += (recall[index] - prior_recall) * precision_envelope[index];
            prior_recall = recall[index];
        }
    }

    let Some(index) = best_index else {
        return DetectionMetric {
            ap,
            best_threshold: predictions.first().map_or(1.0, |item| item.confidence),
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            true_positives: 0,
            false_positives: predictions.len(),
            false_negatives: teachers.len(),
        };
    };
    let tp = true_positive[..=index]
        .iter()
        .filter(|value| **value)
        .count();
    DetectionMetric {
        ap,
        best_threshold: predictions[index].confidence,
        precision: precision[index],
        recall: recall[index],
        f1: best_f1,
        true_positives: tp,
        false_positives: index + 1 - tp,
        false_negatives: teachers.len() - tp,
    }
}

fn box_diagnostics(predictions: &[Box3d], teachers: &[Box3d], threshold: f32) -> BoxDiagnostics {
    let mut claimed = vec![false; teachers.len()];
    let mut iou_sum = 0.0f32;
    let mut ratio_sum = [0.0f32; 3];
    let mut count = 0usize;
    for prediction in predictions
        .iter()
        .take_while(|prediction| prediction.confidence >= threshold)
    {
        let best = teachers
            .iter()
            .enumerate()
            .filter(|(index, teacher)| !claimed[*index] && prediction.class == teacher.class)
            .filter_map(|(index, teacher)| {
                let distance = (0..3)
                    .map(|axis| (prediction.center[axis] - teacher.center[axis]).powi(2))
                    .sum::<f32>()
                    .sqrt();
                let tolerance = 0.5
                    * teacher
                        .size
                        .iter()
                        .map(|value| value.powi(2))
                        .sum::<f32>()
                        .sqrt();
                (distance <= tolerance).then_some((index, distance))
            })
            .min_by(|left, right| left.1.total_cmp(&right.1));
        let Some((index, _)) = best else {
            continue;
        };
        claimed[index] = true;
        let teacher = teachers[index];
        iou_sum += prediction.iou(teacher);
        for (axis, ratio) in ratio_sum.iter_mut().enumerate() {
            *ratio += prediction.size[axis] / teacher.size[axis].max(f32::EPSILON);
        }
        count += 1;
    }
    BoxDiagnostics {
        center_matches: count,
        mean_iou: iou_sum / count.max(1) as f32,
        mean_size_ratio_zyx: ratio_sum.map(|sum| sum / count.max(1) as f32),
    }
}

fn scalar(value: &Tensor<1>) -> anyhow::Result<f32> {
    value
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .map(|values| values[0])
        .map_err(|error| anyhow::anyhow!("reading loss: {error}"))
}

fn simple_hash(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn write_history_svg(path: &std::path::Path, history: &TrainingHistory) -> anyhow::Result<()> {
    const WIDTH: f32 = 900.0;
    const HEIGHT: f32 = 480.0;
    const LEFT: f32 = 70.0;
    const TOP: f32 = 30.0;
    const PLOT_W: f32 = 800.0;
    const PLOT_H: f32 = 390.0;
    let maximum = history
        .rows
        .iter()
        .flat_map(|row| [row.train_loss, row.validation_loss])
        .filter(|value| value.is_finite())
        .fold(0.0f32, f32::max)
        .max(f32::EPSILON);
    let points = |validation: bool| {
        history
            .rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let x = if history.rows.len() <= 1 {
                    LEFT
                } else {
                    LEFT + PLOT_W * index as f32 / (history.rows.len() - 1) as f32
                };
                let value = if validation {
                    row.validation_loss
                } else {
                    row.train_loss
                };
                let y = TOP + PLOT_H * (1.0 - value / maximum);
                format!("{x:.2},{y:.2}")
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    let svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{bottom}" stroke="black"/>
<line x1="{LEFT}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>
<polyline points="{train}" fill="none" stroke="#1769aa" stroke-width="2"/>
<polyline points="{validation}" fill="none" stroke="#d1495b" stroke-width="2"/>
<text x="{LEFT}" y="20" font-family="sans-serif" font-size="16">YOLO3D training loss (blue) and validation loss (red)</text>
<text x="10" y="{TOP}" font-family="sans-serif" font-size="12">{maximum:.4}</text>
<text x="{right}" y="455" text-anchor="end" font-family="sans-serif" font-size="12">epoch {last}</text>
</svg>
"##,
        bottom = TOP + PLOT_H,
        right = LEFT + PLOT_W,
        train = points(false),
        validation = points(true),
        last = history.rows.last().map_or(0, |row| row.epoch),
    );
    fs::write(path, svg)?;
    Ok(())
}

#[derive(Clone, Copy)]
struct Transform {
    rotation: usize,
    reflect_x: bool,
    reflect_z: bool,
    intensity: f32,
    gamma: f32,
}

impl Transform {
    fn identity() -> Self {
        Self {
            rotation: 0,
            reflect_x: false,
            reflect_z: false,
            intensity: 1.0,
            gamma: 1.0,
        }
    }

    fn map(self, [z, y, x]: [usize; 3], shape: [usize; 3]) -> [usize; 3] {
        let z = if self.reflect_z { shape[0] - 1 - z } else { z };
        let x = if self.reflect_x { shape[2] - 1 - x } else { x };
        let (y, x) = match self.rotation % 4 {
            0 => (y, x),
            1 => (x, shape[1] - 1 - y),
            2 => (shape[1] - 1 - y, shape[2] - 1 - x),
            _ => (shape[2] - 1 - x, y),
        };
        [z, y, x]
    }
}

fn transform(
    policy: AugmentationChoice,
    epoch: usize,
    sample: usize,
    allow_geometry: bool,
) -> Transform {
    if matches!(policy, AugmentationChoice::None) {
        return Transform::identity();
    }
    let mut state = (epoch as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(sample as u64);
    let mut random = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 32) as u32) as f32 / u32::MAX as f32
    };
    // Interior ownership cores are symmetric under these transforms. Volume-edge
    // cores are asymmetric so that the outermost voxels are owned, so keep their
    // geometry fixed rather than transforming the ownership contract.
    let geometry = allow_geometry && matches!(policy, AugmentationChoice::Fluorescence);
    Transform {
        rotation: if geometry {
            (random() * 4.0) as usize
        } else {
            0
        },
        reflect_x: geometry && random() < 0.5,
        reflect_z: geometry && random() < 0.5,
        intensity: 0.8 + 0.4 * random(),
        gamma: 0.8 + 0.4 * random(),
    }
}
