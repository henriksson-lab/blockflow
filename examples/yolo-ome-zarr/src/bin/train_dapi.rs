use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use blockflow::assemble::ImageId;
use blockflow::object_table::TableReader;
use blockflow::{AttachedImage, Environment, Region, Voxels, ZarrEnvironment};
use burn::prelude::*;
use clap::{Parser, ValueEnum};
use yolov11::data::source::{BoxTarget, GeometryPolicy, RawSample, SampleStore, SharedSampleStore};
use yolov11::train::config::Config as YoloConfig;
use yolov11::train::train::{train_with_stores_and_test, EvaluationMetrics, TrainingOptions};

const CORE: usize = 512;
const HALO: usize = 64;
const INPUT: usize = CORE + 2 * HALO;
const LABEL_IMAGE: usize = ImageId::SUPPLIED_BASE;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum GeometryArg {
    None,
    D4,
}

impl From<GeometryArg> for GeometryPolicy {
    fn from(value: GeometryArg) -> Self {
        match value {
            GeometryArg::None => Self::None,
            GeometryArg::D4 => Self::D4,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Train YOLO on DAPI crops with Cellpose-derived boxes")]
struct Args {
    #[arg(long)]
    zarr: PathBuf,
    #[arg(long, default_value = "cellpose-dapi")]
    layer: String,
    #[arg(long, default_value_t = 0)]
    channel: usize,
    #[arg(long, default_value_t = 1)]
    low: u8,
    #[arg(long, default_value_t = 70)]
    high: u8,
    #[arg(long, default_value = "examples/yolo-ome-zarr/dapi_args.yaml")]
    config: PathBuf,
    #[arg(long, default_value = ".tmp/yolo-dapi-training/work")]
    work: PathBuf,
    #[arg(long, default_value = ".tmp/yolo-dapi-training/output")]
    output: PathBuf,
    #[arg(long, default_value_t = 64)]
    train_tiles: usize,
    #[arg(long, default_value_t = 16)]
    validation_tiles: usize,
    #[arg(long, default_value_t = 64)]
    test_tiles: usize,
    #[arg(long, default_value_t = 20)]
    epochs: usize,
    #[arg(long, default_value_t = 2)]
    batch_size: usize,
    #[arg(long, default_value_t = 2)]
    workers: usize,
    #[arg(long, default_value_t = 2)]
    queue_batches: usize,
    #[arg(long, default_value_t = 512)]
    cache_mib: u64,
    #[arg(long, default_value_t = 2)]
    prefetch_threads: usize,
    #[arg(long, value_enum, default_value_t = GeometryArg::D4)]
    geometry: GeometryArg,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, conflicts_with = "resume")]
    weights: Option<PathBuf>,
    /// Continue an exact staged run from this run directory.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Total epoch horizon for the learning-rate schedule. Set on the first stage.
    #[arg(long)]
    schedule_epochs: Option<usize>,
    /// Evaluate the frozen test region after this stage.
    #[arg(long)]
    finalize: bool,
    /// Keep pretrained features but initialize a new one-class output head.
    #[arg(long, requires = "weights")]
    reset_class_head: bool,
    /// Per-parameter gradient norm limit. Use 0 to disable clipping.
    #[arg(long, default_value_t = 10.0)]
    gradient_clip: f32,
    /// Validate the source, split, and generated targets without starting CUDA training.
    #[arg(long)]
    inspect_only: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Tile {
    y: usize,
    x: usize,
}

struct OmeDapiStore {
    env: Arc<ZarrEnvironment>,
    tiles: Vec<Tile>,
    centroids: Arc<HashMap<u64, [f32; 2]>>,
    height: usize,
    width: usize,
    low: u8,
    high: u8,
}

impl OmeDapiStore {
    fn new(
        env: Arc<ZarrEnvironment>,
        tiles: Vec<Tile>,
        centroids: Arc<HashMap<u64, [f32; 2]>>,
        shape: [usize; 2],
        low: u8,
        high: u8,
    ) -> Result<Self> {
        anyhow::ensure!(low < high, "normalization requires --low < --high");
        Ok(Self {
            env,
            tiles,
            centroids,
            height: shape[0],
            width: shape[1],
            low,
            high,
        })
    }

    fn region(&self, tile: Tile) -> (Region, isize, isize) {
        let nominal_y = tile.y as isize - HALO as isize;
        let nominal_x = tile.x as isize - HALO as isize;
        let y0 = nominal_y.max(0) as usize;
        let x0 = nominal_x.max(0) as usize;
        let y1 = (nominal_y + INPUT as isize).min(self.height as isize) as usize;
        let x1 = (nominal_x + INPUT as isize).min(self.width as isize) as usize;
        (
            Region::new(&[0, y0, x0], &[1, y1 - y0, x1 - x0]),
            nominal_y,
            nominal_x,
        )
    }
}

impl SampleStore for OmeDapiStore {
    fn len(&self) -> usize {
        self.tiles.len()
    }

    fn load(&self, key: usize) -> Result<RawSample> {
        let tile = *self
            .tiles
            .get(key)
            .with_context(|| format!("DAPI tile key {key} is out of range"))?;
        let (region, nominal_y, nominal_x) = self.region(tile);
        let image = self.env.read(0, &region)?;
        let labels = self.env.read(LABEL_IMAGE, &region)?;
        let image = match image.as_array()? {
            Voxels::U8(values) => values,
            other => anyhow::bail!("DAPI array is {:?}, expected uint8", other.dtype()),
        };
        let labels = match labels.as_array()? {
            Voxels::U64(values) => values,
            other => anyhow::bail!("Cellpose labels are {:?}, expected uint64", other.dtype()),
        };

        let y0 = region.start[1];
        let x0 = region.start[2];
        let read_h = region.shape[1];
        let read_w = region.shape[2];
        let dst_y = (y0 as isize - nominal_y) as usize;
        let dst_x = (x0 as isize - nominal_x) as usize;
        let mut rgb = vec![0u8; INPUT * INPUT * 3];
        let range = (self.high - self.low) as f32;
        for y in 0..read_h {
            for x in 0..read_w {
                let value = image[[0, y, x]];
                let normalized = (((value.saturating_sub(self.low)) as f32 / range) * 255.0)
                    .clamp(0.0, 255.0) as u8;
                let pixel = ((dst_y + y) * INPUT + dst_x + x) * 3;
                rgb[pixel..pixel + 3].fill(normalized);
            }
        }

        let mut boxes = HashMap::<u64, [usize; 4]>::new();
        for y in 0..read_h {
            for x in 0..read_w {
                let label = labels[[0, y, x]];
                if label == 0 || label == u64::MAX {
                    continue;
                }
                boxes
                    .entry(label)
                    .and_modify(|bbox| {
                        bbox[0] = bbox[0].min(y);
                        bbox[1] = bbox[1].min(x);
                        bbox[2] = bbox[2].max(y + 1);
                        bbox[3] = bbox[3].max(x + 1);
                    })
                    .or_insert([y, x, y + 1, x + 1]);
            }
        }

        let core_y1 = (tile.y + CORE).min(self.height) as f32;
        let core_x1 = (tile.x + CORE).min(self.width) as f32;
        let mut targets = Vec::new();
        for (label, [min_y, min_x, max_y, max_x]) in boxes {
            let Some([centroid_y, centroid_x]) = self.centroids.get(&label).copied() else {
                continue;
            };
            if centroid_y < tile.y as f32
                || centroid_y >= core_y1
                || centroid_x < tile.x as f32
                || centroid_x >= core_x1
            {
                continue;
            }
            let touches_read_edge = (min_y == 0 && y0 != 0)
                || (min_x == 0 && x0 != 0)
                || (max_y == read_h && y0 + read_h != self.height)
                || (max_x == read_w && x0 + read_w != self.width);
            if touches_read_edge {
                continue;
            }
            targets.push(BoxTarget {
                class: 0.0,
                x1: (dst_x + min_x) as f32,
                y1: (dst_y + min_y) as f32,
                x2: (dst_x + max_x) as f32,
                y2: (dst_y + max_y) as f32,
            });
        }
        targets.sort_by(|left, right| {
            left.y1
                .total_cmp(&right.y1)
                .then(left.x1.total_cmp(&right.x1))
        });

        Ok(RawSample {
            rgb,
            width: INPUT,
            height: INPUT,
            targets,
        })
    }

    fn prefetch(&self, keys: &[usize]) -> Result<()> {
        let regions = keys
            .iter()
            .filter_map(|&key| self.tiles.get(key).copied())
            .map(|tile| self.region(tile).0)
            .collect::<Vec<_>>();
        self.env.prefetch(0, &regions)?;
        self.env.prefetch(LABEL_IMAGE, &regions)?;
        Ok(())
    }
}

fn read_centroids(table_root: &Path) -> Result<HashMap<u64, [f32; 2]>> {
    let reader = TableReader::open(table_root)?;
    let rows = reader.spec().row_count;
    let ids = reader.read_u64("label_id", 0..rows)?;
    let ys = reader.read_f32("centroid_y", 0..rows)?;
    let xs = reader.read_f32("centroid_x", 0..rows)?;
    anyhow::ensure!(
        ids.len() == ys.len() && ids.len() == xs.len(),
        "table columns differ in length"
    );
    Ok(ids
        .into_iter()
        .zip(ys.into_iter().zip(xs))
        .map(|(id, (y, x))| (id, [y, x]))
        .collect())
}

fn occupied_tiles(centroids: &HashMap<u64, [f32; 2]>) -> Vec<Tile> {
    centroids
        .values()
        .map(|[y, x]| Tile {
            y: (*y as usize / CORE) * CORE,
            x: (*x as usize / CORE) * CORE,
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn intersects(a: [usize; 4], b: [usize; 4]) -> bool {
    a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
}

fn split_tiles(tiles: Vec<Tile>, shape: [usize; 2]) -> (Vec<Tile>, Vec<Tile>, Vec<Tile>) {
    let test = [
        shape[0] / 4,
        shape[1] / 4,
        3 * shape[0] / 4,
        3 * shape[1] / 4,
    ];
    let mut eligible = Vec::new();
    let mut test_tiles = Vec::new();
    for tile in tiles {
        let core = [
            tile.y,
            tile.x,
            (tile.y + CORE).min(shape[0]),
            (tile.x + CORE).min(shape[1]),
        ];
        let core_inside_test =
            core[0] >= test[0] && core[1] >= test[1] && core[2] <= test[2] && core[3] <= test[3];
        if core_inside_test {
            test_tiles.push(tile);
            continue;
        }
        let input = [
            tile.y.saturating_sub(HALO),
            tile.x.saturating_sub(HALO),
            (tile.y + CORE + HALO).min(shape[0]),
            (tile.x + CORE + HALO).min(shape[1]),
        ];
        if intersects(input, test) {
            continue;
        }
        eligible.push(tile);
    }
    let validation_cut = eligible
        .iter()
        .map(|tile| tile.y)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let validation_cut = validation_cut
        .get(validation_cut.len().saturating_mul(4) / 5)
        .copied()
        .unwrap_or(usize::MAX);
    let (validation, training): (Vec<_>, Vec<_>) = eligible
        .into_iter()
        .partition(|tile| tile.y >= validation_cut);
    (training, validation, test_tiles)
}

fn select_evenly<T>(values: Vec<T>, limit: usize) -> Vec<T> {
    if limit == 0 {
        return Vec::new();
    }
    if values.len() <= limit {
        return values;
    }
    let total = values.len();
    let mut values = values.into_iter().map(Some).collect::<Vec<_>>();
    (0..limit)
        .map(|index| {
            let source = ((2 * index + 1) * total) / (2 * limit);
            values[source]
                .take()
                .expect("evenly selected indices are unique")
        })
        .collect()
}

fn write_preview(path: &Path, sample: &RawSample) -> Result<()> {
    let mut rgb = sample.rgb.clone();
    for target in &sample.targets {
        let x1 = target.x1.max(0.0).min((sample.width - 1) as f32) as usize;
        let y1 = target.y1.max(0.0).min((sample.height - 1) as f32) as usize;
        let x2 = target.x2.max(1.0).min(sample.width as f32) as usize - 1;
        let y2 = target.y2.max(1.0).min(sample.height as f32) as usize - 1;
        for x in x1..=x2 {
            for y in [y1, y2] {
                let pixel = (y * sample.width + x) * 3;
                rgb[pixel..pixel + 3].copy_from_slice(&[255, 0, 0]);
            }
        }
        for y in y1..=y2 {
            for x in [x1, x2] {
                let pixel = (y * sample.width + x) * 3;
                rgb[pixel..pixel + 3].copy_from_slice(&[255, 0, 0]);
            }
        }
    }
    let image = image::RgbImage::from_raw(sample.width as u32, sample.height as u32, rgb)
        .context("sample RGB buffer has the wrong length")?;
    image.save(path)?;
    Ok(())
}

fn inspect_store(name: &str, store: &OmeDapiStore, output: &Path) -> Result<serde_json::Value> {
    let started = Instant::now();
    let mut target_counts = Vec::with_capacity(store.len());
    for key in 0..store.len() {
        let sample = store.load(key)?;
        if key == 0 {
            write_preview(&output.join(format!("{name}-sample.png")), &sample)?;
        }
        target_counts.push(sample.targets.len());
    }
    let targets: usize = target_counts.iter().sum();
    Ok(serde_json::json!({
        "tiles": store.len(),
        "targets": targets,
        "empty_tiles": target_counts.iter().filter(|&&count| count == 0).count(),
        "min_targets_per_tile": target_counts.iter().min().copied().unwrap_or(0),
        "max_targets_per_tile": target_counts.iter().max().copied().unwrap_or(0),
        "mean_targets_per_tile": targets as f64 / store.len().max(1) as f64,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    }))
}

fn metrics_json(metrics: EvaluationMetrics) -> serde_json::Value {
    serde_json::json!({
        "mAP": metrics.mean_ap,
        "mAP@50": metrics.map50,
        "recall": metrics.recall,
        "precision": metrics.precision,
    })
}

fn split_manifest(
    args: &Args,
    shape: [usize; 2],
    train: &[Tile],
    validation: &[Tile],
    test: &[Tile],
) -> serde_json::Value {
    let coordinates = |tiles: &[Tile]| {
        tiles
            .iter()
            .map(|tile| [tile.y, tile.x])
            .collect::<Vec<_>>()
    };
    serde_json::json!({
        "version": 1,
        "zarr": args.zarr.display().to_string(),
        "layer": &args.layer,
        "channel": args.channel,
        "normalization": [args.low, args.high],
        "shape": shape,
        "core": CORE,
        "halo": HALO,
        "train": coordinates(train),
        "validation": coordinates(validation),
        "test": coordinates(test),
    })
}

fn persist_split(output: &Path, resume: Option<&Path>, split: &serde_json::Value) -> Result<()> {
    if let Some(run) = resume {
        let path = run.join("split.json");
        let previous: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&path)
                .with_context(|| format!("reading resume split {}", path.display()))?,
        )?;
        anyhow::ensure!(
            previous == *split,
            "the source or spatial train/validation/test split changed since the resumed run"
        );
    }
    std::fs::write(output.join("split.json"), serde_json::to_vec_pretty(split)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let program_started = Instant::now();
    let args = Args::parse();
    std::fs::create_dir_all(&args.work)?;
    std::fs::create_dir_all(&args.output)?;

    let source = AttachedImage::at(args.zarr.join("0"));
    let (_, source_shape) = source.metadata()?;
    anyhow::ensure!(
        args.channel < source_shape[0],
        "channel {} is outside source shape {:?}",
        args.channel,
        source_shape
    );
    let shape = [source_shape[1], source_shape[2]];
    let source = source.plane(args.channel, shape);
    let label_path = args.zarr.join("labels").join(&args.layer).join("0");
    let labels = AttachedImage::at(&label_path);
    let (_, label_shape) = labels.metadata()?;
    anyhow::ensure!(
        label_shape == [1, shape[0], shape[1]],
        "label shape {:?} does not match image plane {:?}",
        label_shape,
        shape
    );

    let env = Arc::new(
        ZarrEnvironment::attach(&args.work, &[source, labels])?
            .with_cache(args.cache_mib * 1024 * 1024)
            .with_prefetch(args.prefetch_threads, 1)?,
    );
    let centroids = Arc::new(read_centroids(&args.zarr.join("tables").join(&args.layer))?);
    let occupied = occupied_tiles(&centroids);
    let (train_tiles, validation_tiles, test_tiles) = split_tiles(occupied, shape);
    let train_tiles = select_evenly(train_tiles, args.train_tiles);
    let validation_tiles = select_evenly(validation_tiles, args.validation_tiles);
    let test_tiles = select_evenly(test_tiles, args.test_tiles);
    anyhow::ensure!(
        !train_tiles.is_empty(),
        "spatial split produced no training tiles"
    );
    anyhow::ensure!(
        !validation_tiles.is_empty(),
        "spatial split produced no validation tiles"
    );
    anyhow::ensure!(
        !test_tiles.is_empty(),
        "spatial split produced no test tiles"
    );
    println!(
        "objects={} train_tiles={} validation_tiles={} frozen_test_tiles={} shape={}x{}",
        centroids.len(),
        train_tiles.len(),
        validation_tiles.len(),
        test_tiles.len(),
        shape[0],
        shape[1]
    );
    let split = split_manifest(&args, shape, &train_tiles, &validation_tiles, &test_tiles);
    persist_split(&args.output, args.resume.as_deref(), &split)?;

    let actual_train_tiles = train_tiles.len();
    let actual_validation_tiles = validation_tiles.len();
    let actual_test_tiles = test_tiles.len();
    let train = Arc::new(OmeDapiStore::new(
        Arc::clone(&env),
        train_tiles,
        Arc::clone(&centroids),
        shape,
        args.low,
        args.high,
    )?);
    let validation = Arc::new(OmeDapiStore::new(
        Arc::clone(&env),
        validation_tiles,
        Arc::clone(&centroids),
        shape,
        args.low,
        args.high,
    )?);
    let test = Arc::new(OmeDapiStore::new(
        Arc::clone(&env),
        test_tiles,
        Arc::clone(&centroids),
        shape,
        args.low,
        args.high,
    )?);
    if args.inspect_only {
        let started = Instant::now();
        let train_report = inspect_store("train", &train, &args.output)?;
        let validation_report = inspect_store("validation", &validation, &args.output)?;
        let test_report = inspect_store("test", &test, &args.output)?;
        env.drain_prefetch();
        let report = serde_json::json!({
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "objects": centroids.len(),
            "shape": shape,
            "train": train_report,
            "validation": validation_report,
            "test": test_report,
        });
        std::fs::write(
            args.output.join("inspection.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    let train: SharedSampleStore = train;
    let validation: SharedSampleStore = validation;
    let test: SharedSampleStore = test;
    let config = YoloConfig::load(&args.config)?;
    let options = TrainingOptions {
        input_size: INPUT,
        batch_size: args.batch_size,
        epochs: args.epochs,
        eval_interval: 1,
        loader_workers: args.workers,
        queue_batches: args.queue_batches,
        seed: args.seed,
        geometry: args.geometry.into(),
        weights: args.weights,
        reset_class_head: args.reset_class_head,
        gradient_clip: (args.gradient_clip > 0.0).then_some(args.gradient_clip),
        schedule_epochs: args.schedule_epochs,
        resume: args.resume.clone(),
        finalize: args.finalize,
        output_dir: args.output.clone(),
    };

    #[cfg(feature = "libtorch")]
    let device = Device::libtorch_cuda(burn::tensor::DeviceIndex::Default).autodiff();
    #[cfg(all(feature = "cuda", not(feature = "libtorch")))]
    let device = Device::cuda(burn::tensor::DeviceIndex::Default).autodiff();
    #[cfg(not(any(feature = "cuda", feature = "libtorch")))]
    compile_error!("yolo-dapi-train requires --features libtorch or cuda");

    let started = Instant::now();
    let summary = train_with_stores_and_test(
        &config,
        train,
        Some(validation),
        Some(test),
        &options,
        &device,
    )?;
    env.drain_prefetch();
    let elapsed = started.elapsed();
    let cache = env
        .cache_stats()
        .context("training environment has no cache")?;
    let prefetch = env
        .prefetch_stats()
        .context("training environment has no prefetcher")?;
    let report = serde_json::json!({
        "elapsed_seconds": elapsed.as_secs_f64(),
        "end_to_end_seconds": program_started.elapsed().as_secs_f64(),
        "zarr": args.zarr.display().to_string(),
        "layer": args.layer,
        "channel": args.channel,
        "normalization": [args.low, args.high],
        "config": args.config.display().to_string(),
        "weights": options.weights.as_ref().map(|path| path.display().to_string()),
        "resume": options.resume.as_ref().map(|path| path.display().to_string()),
        "reset_class_head": options.reset_class_head,
        "train_tiles": actual_train_tiles,
        "validation_tiles": actual_validation_tiles,
        "test_tiles": actual_test_tiles,
        "epochs": args.epochs,
        "stage": summary.stage,
        "completed_epochs": summary.completed_epochs,
        "schedule_epochs": summary.schedule_epochs,
        "finalized": args.finalize,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "geometry": format!("{:?}", args.geometry).to_lowercase(),
        "gradient_clip": args.gradient_clip,
        "best_epoch": summary.best_epoch,
        "best_validation": metrics_json(summary.best_validation),
        "test": summary.test.map(metrics_json),
        "cache": {
            "hits_decoded": cache.hits_decoded,
            "hits_encoded": cache.hits_encoded,
            "misses": cache.misses,
            "evictions": cache.evictions,
            "source_reads": cache.source_reads,
            "source_bytes": cache.source_bytes,
            "prefetch_issued": cache.prefetch_issued,
            "prefetch_used": cache.prefetch_used,
            "prefetch_wasted_evicted": cache.prefetch_wasted_evicted,
            "prefetch_wasted_refused": cache.prefetch_wasted_refused,
            "prefetch_declined": cache.prefetch_declined,
            "resident_bytes": cache.resident_bytes,
            "resident_chunks": cache.resident_chunks,
        },
        "prefetch": {
            "submitted": prefetch.submitted,
            "started": prefetch.started,
            "chunks": prefetch.chunks,
            "cancelled": prefetch.cancelled,
            "declined": prefetch.declined,
            "failed": prefetch.failed,
        }
    });
    let report_bytes = serde_json::to_vec_pretty(&report)?;
    std::fs::write(args.output.join("training-run.json"), &report_bytes)?;
    std::fs::write(
        args.output
            .join(format!("stage-{:04}-run.json", summary.stage)),
        report_bytes,
    )?;
    println!(
        "elapsed={:.3}s output={}",
        elapsed.as_secs_f64(),
        args.output.display()
    );
    Ok(())
}
