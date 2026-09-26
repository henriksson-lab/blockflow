#![recursion_limit = "256"]

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use blockflow::assemble::ImageId;
use blockflow::object_table::TableReader;
use blockflow::{AttachedImage, Environment, Region, Voxels, ZarrEnvironment};
use burn::prelude::*;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use yolov11::data::source::{BoxTarget, GeometryPolicy, RawSample, SampleStore, SharedSampleStore};
use yolov11::train::config::Config as YoloConfig;
use yolov11::train::train::{
    evaluate_checkpoint_with_store, train_with_stores_and_test, BatchNormPolicy, EvaluationMetrics,
    TrainableLayers, TrainingOptions,
};

const CORE: usize = 512;
const HALO: usize = 64;
const INPUT: usize = CORE + 2 * HALO;
const LABEL_IMAGE: usize = ImageId::SUPPLIED_BASE;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum GeometryArg {
    None,
    D4,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TrainableLayersArg {
    All,
    HeadAndNeck,
    HeadOnly,
}

impl From<TrainableLayersArg> for TrainableLayers {
    fn from(value: TrainableLayersArg) -> Self {
        match value {
            TrainableLayersArg::All => Self::All,
            TrainableLayersArg::HeadAndNeck => Self::HeadAndNeck,
            TrainableLayersArg::HeadOnly => Self::HeadOnly,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BatchNormArg {
    Update,
    Frozen,
}

impl From<BatchNormArg> for BatchNormPolicy {
    fn from(value: BatchNormArg) -> Self {
        match value {
            BatchNormArg::Update => Self::Update,
            BatchNormArg::Frozen => Self::Frozen,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OverfitDensity {
    Sparse,
    Dense,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TileCount {
    All,
    Count(usize),
}

impl FromStr for TileCount {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("all") {
            return Ok(Self::All);
        }
        let count = value
            .parse::<usize>()
            .map_err(|_| format!("expected a positive integer or 'all', got {value:?}"))?;
        if count == 0 {
            return Err("tile count must be positive".to_owned());
        }
        Ok(Self::Count(count))
    }
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
    /// Number of training tiles, or `all` for every eligible occupied tile.
    #[arg(long, default_value = "64")]
    train_tiles: TileCount,
    #[arg(long, default_value_t = 16)]
    validation_tiles: usize,
    #[arg(long, default_value_t = 64)]
    test_tiles: usize,
    /// Desired fraction of spatially separated background tiles in each split.
    #[arg(long, default_value_t = 0.0)]
    negative_tile_fraction: f64,
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
    #[arg(long, value_enum, default_value_t = TrainableLayersArg::All)]
    trainable_layers: TrainableLayersArg,
    #[arg(long, value_enum, default_value_t = BatchNormArg::Update)]
    batch_norm: BatchNormArg,
    /// Deliberately use the same sparse or dense tiles for training and validation.
    #[arg(
        long,
        value_enum,
        conflicts_with_all = ["sweep", "resume", "finalize"]
    )]
    overfit_density: Option<OverfitDensity>,
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
    /// Confidence threshold used while evaluating validation and test data.
    #[arg(long, default_value_t = 0.001)]
    evaluation_confidence: f32,
    /// IoU threshold used by evaluation NMS.
    #[arg(long, default_value_t = 0.65)]
    evaluation_nms_iou: f32,
    /// Maximum detections retained per image during evaluation.
    #[arg(long, default_value_t = 1000)]
    evaluation_max_detections: usize,
    /// Validate the source, split, and generated targets without starting CUDA training.
    #[arg(long, conflicts_with = "evaluate_only")]
    inspect_only: bool,
    /// Evaluate --weights on the validation split without training or using test data.
    #[arg(long, requires = "weights", conflicts_with = "inspect_only")]
    evaluate_only: bool,
    /// Run a sequential validation-ranked hyperparameter sweep from YAML.
    #[arg(
        long,
        requires = "weights",
        conflicts_with_all = ["inspect_only", "evaluate_only", "resume", "finalize"]
    )]
    sweep: Option<PathBuf>,
}

fn finish_process() -> anyhow::Result<()> {
    std::io::stdout().flush()?;
    // Long-running LibTorch autodiff clients have finished writing every
    // report and checkpoint at this point, but C++ CUDA teardown can segfault.
    // Exit after the explicit flush instead of running those atexit handlers.
    #[cfg(feature = "libtorch")]
    unsafe {
        libc::_exit(0);
    }
    #[cfg(not(feature = "libtorch"))]
    Ok(())
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SweepSpec {
    rounds: Vec<SweepRound>,
    #[serde(default)]
    max_lr: Vec<f64>,
    #[serde(default)]
    min_lr_ratio: Vec<f64>,
    #[serde(default)]
    weight_decay: Vec<f64>,
    #[serde(default)]
    gradient_clip: Vec<f32>,
    #[serde(default)]
    geometry: Vec<SweepGeometry>,
    #[serde(default)]
    trainable_layers: Vec<TrainableLayers>,
    #[serde(default)]
    batch_norm: Vec<BatchNormPolicy>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SweepRound {
    epochs: usize,
    keep: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum SweepGeometry {
    None,
    D4,
}

impl From<SweepGeometry> for GeometryPolicy {
    fn from(value: SweepGeometry) -> Self {
        match value {
            SweepGeometry::None => Self::None,
            SweepGeometry::D4 => Self::D4,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SweepTrialReport {
    id: String,
    max_lr: f64,
    min_lr: f64,
    weight_decay: f64,
    gradient_clip: Option<f32>,
    geometry: SweepGeometry,
    trainable_layers: TrainableLayers,
    batch_norm: BatchNormPolicy,
    completed_epochs: usize,
    active: bool,
    best_epoch: usize,
    best_validation: EvaluationMetrics,
    elapsed_seconds: f64,
}

#[derive(Clone)]
struct SweepTrial {
    report: SweepTrialReport,
    config: YoloConfig,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Tile {
    y: usize,
    x: usize,
}

struct OmeDapiStore {
    env: Arc<ZarrEnvironment>,
    tiles: Vec<Tile>,
    height: usize,
    width: usize,
    low: u8,
    high: u8,
}

impl OmeDapiStore {
    fn new(
        env: Arc<ZarrEnvironment>,
        tiles: Vec<Tile>,
        shape: [usize; 2],
        low: u8,
        high: u8,
    ) -> Result<Self> {
        anyhow::ensure!(low < high, "normalization requires --low < --high");
        Ok(Self {
            env,
            tiles,
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

        let mut targets = Vec::new();
        for (_label, [min_y, min_x, max_y, max_x]) in boxes {
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

    fn epoch_keys(&self, epoch: usize, seed: u64) -> Vec<usize> {
        const BLOCK_TILES: usize = 4;
        let mut grouped = std::collections::BTreeMap::<(usize, usize), Vec<usize>>::new();
        for (key, tile) in self.tiles.iter().enumerate() {
            grouped
                .entry((tile.y / (CORE * BLOCK_TILES), tile.x / (CORE * BLOCK_TILES)))
                .or_default()
                .push(key);
        }
        let mut blocks = grouped.into_values().collect::<Vec<_>>();
        let mut state = seed ^ (epoch as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        deterministic_shuffle(&mut blocks, &mut state);
        for block in &mut blocks {
            deterministic_shuffle(block, &mut state);
        }
        blocks.into_iter().flatten().collect()
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

fn deterministic_shuffle<T>(values: &mut [T], state: &mut u64) {
    for index in (1..values.len()).rev() {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let selected = ((value ^ (value >> 31)) as usize) % (index + 1);
        values.swap(index, selected);
    }
}

fn all_tiles(shape: [usize; 2]) -> Vec<Tile> {
    (0..shape[0])
        .step_by(CORE)
        .flat_map(|y| (0..shape[1]).step_by(CORE).map(move |x| Tile { y, x }))
        .collect()
}

fn background_tiles(tiles: Vec<Tile>, occupied: &BTreeSet<Tile>) -> Vec<Tile> {
    let mut blocked = BTreeSet::new();
    for tile in occupied {
        for dy in [-1isize, 0, 1] {
            for dx in [-1isize, 0, 1] {
                let Some(y) = tile.y.checked_add_signed(dy * CORE as isize) else {
                    continue;
                };
                let Some(x) = tile.x.checked_add_signed(dx * CORE as isize) else {
                    continue;
                };
                blocked.insert(Tile { y, x });
            }
        }
    }
    tiles
        .into_iter()
        .filter(|tile| !blocked.contains(tile))
        .collect()
}

fn intersects(a: [usize; 4], b: [usize; 4]) -> bool {
    a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
}

fn split_test_region(tiles: Vec<Tile>, shape: [usize; 2]) -> (Vec<Tile>, Vec<Tile>) {
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
    (eligible, test_tiles)
}

fn split_tiles(tiles: Vec<Tile>, shape: [usize; 2]) -> (Vec<Tile>, Vec<Tile>, Vec<Tile>, usize) {
    let (eligible, test_tiles) = split_test_region(tiles, shape);
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
    let (validation, mut training): (Vec<_>, Vec<_>) = eligible
        .into_iter()
        .partition(|tile| tile.y >= validation_cut);
    // Keep the full 640-pixel inputs disjoint across the train/validation
    // boundary. Without this guard, adjacent 64-pixel halos expose the same
    // tissue in both partitions.
    training.retain(|tile| tile.y.saturating_add(CORE + 2 * HALO) <= validation_cut);
    (training, validation, test_tiles, validation_cut)
}

fn split_tiles_at(
    tiles: Vec<Tile>,
    shape: [usize; 2],
    validation_cut: usize,
) -> (Vec<Tile>, Vec<Tile>, Vec<Tile>) {
    let (eligible, test_tiles) = split_test_region(tiles, shape);
    let (validation, mut training): (Vec<_>, Vec<_>) = eligible
        .into_iter()
        .partition(|tile| tile.y >= validation_cut);
    training.retain(|tile| tile.y.saturating_add(CORE + 2 * HALO) <= validation_cut);
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

fn select_with_background(
    positives: Vec<Tile>,
    backgrounds: Vec<Tile>,
    limit: TileCount,
    negative_fraction: f64,
) -> Vec<Tile> {
    let (positive_limit, negative_limit) = match limit {
        TileCount::All => {
            let negatives = if negative_fraction == 0.0 {
                0
            } else {
                ((positives.len() as f64 * negative_fraction) / (1.0 - negative_fraction)).round()
                    as usize
            };
            (positives.len(), negatives)
        }
        TileCount::Count(total) => {
            let negatives = (total as f64 * negative_fraction).round() as usize;
            (total.saturating_sub(negatives), negatives)
        }
    };
    let mut selected = select_evenly(positives, positive_limit);
    selected.extend(select_evenly(backgrounds, negative_limit));
    selected.sort_unstable();
    selected
}

fn select_overfit_tiles(
    mut tiles: Vec<Tile>,
    centroids: &HashMap<u64, [f32; 2]>,
    limit: usize,
    density: OverfitDensity,
) -> Vec<Tile> {
    let mut ownership_counts = HashMap::<Tile, usize>::new();
    for [y, x] in centroids.values() {
        *ownership_counts
            .entry(Tile {
                y: (*y as usize / CORE) * CORE,
                x: (*x as usize / CORE) * CORE,
            })
            .or_default() += 1;
    }
    if matches!(density, OverfitDensity::Sparse) {
        // Avoid nearly empty tiles: they make AP unstable and do not exercise
        // whether the detector can memorize distinct nuclei. Sixteen owned
        // objects still represents the sparse end of this dataset.
        tiles.retain(|tile| ownership_counts.get(tile).copied().unwrap_or(0) >= 16);
    }
    tiles.sort_by(|left, right| {
        let left_count = ownership_counts.get(left).copied().unwrap_or(0);
        let right_count = ownership_counts.get(right).copied().unwrap_or(0);
        let density_order = match density {
            OverfitDensity::Sparse => left_count.cmp(&right_count),
            OverfitDensity::Dense => right_count.cmp(&left_count),
        };
        density_order.then_with(|| left.cmp(right))
    });
    tiles.truncate(limit);
    tiles
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

fn box_iou(left: &BoxTarget, right: &BoxTarget) -> f32 {
    let intersection = (left.x2.min(right.x2) - left.x1.max(right.x1)).max(0.0)
        * (left.y2.min(right.y2) - left.y1.max(right.y1)).max(0.0);
    let left_area = (left.x2 - left.x1) * (left.y2 - left.y1);
    let right_area = (right.x2 - right.x1) * (right.y2 - right.y1);
    intersection / (left_area + right_area - intersection + 1e-7)
}

fn oracle_nms_count(targets: &[BoxTarget], iou_threshold: f32, max_detections: usize) -> usize {
    let mut suppressed = vec![false; targets.len()];
    let mut retained = 0;
    for index in 0..targets.len() {
        if suppressed[index] {
            continue;
        }
        if retained == max_detections {
            break;
        }
        retained += 1;
        for candidate in index + 1..targets.len() {
            if !suppressed[candidate]
                && box_iou(&targets[index], &targets[candidate]) > iou_threshold
            {
                suppressed[candidate] = true;
            }
        }
    }
    retained
}

fn percentile(sorted: &[usize], numerator: usize, denominator: usize) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let index = (sorted.len() - 1) * numerator / denominator;
    sorted[index]
}

fn assignment_geometry(targets: &[BoxTarget]) -> (Vec<usize>, usize, usize) {
    let levels = [(8.0f32, 80usize), (16.0, 40), (32.0, 20)];
    let total_locations: usize = levels.iter().map(|(_, size)| size * size).sum();
    let mut occupancy = vec![0u16; total_locations];
    let mut candidate_counts = Vec::with_capacity(targets.len());
    for target in targets {
        let mut candidates = 0;
        let mut offset = 0;
        for (stride, size) in levels {
            for y in 0..size {
                let anchor_y = (y as f32 + 0.5) * stride;
                if anchor_y <= target.y1 || anchor_y >= target.y2 {
                    continue;
                }
                for x in 0..size {
                    let anchor_x = (x as f32 + 0.5) * stride;
                    if anchor_x > target.x1 && anchor_x < target.x2 {
                        occupancy[offset + y * size + x] += 1;
                        candidates += 1;
                    }
                }
            }
            offset += size * size;
        }
        candidate_counts.push(candidates);
    }
    let contested_locations = occupancy.iter().filter(|&&count| count > 1).count();
    let contested_relations = occupancy
        .iter()
        .filter(|&&count| count > 1)
        .map(|&count| count as usize)
        .sum();
    (candidate_counts, contested_locations, contested_relations)
}

fn inspect_store(name: &str, store: &OmeDapiStore, output: &Path) -> Result<serde_json::Value> {
    let started = Instant::now();
    let mut target_counts = Vec::with_capacity(store.len());
    let mut areas = Vec::new();
    let mut oracle_at_65 = 0usize;
    let mut oracle_at_80 = 0usize;
    let mut oracle_at_95 = 0usize;
    let mut oracle_cap_300 = 0usize;
    let mut overlapping_pairs_at_65 = 0usize;
    let mut assignment_candidates = Vec::new();
    let mut contested_anchor_locations = 0usize;
    let mut contested_anchor_relations = 0usize;
    let mut csv = String::from(
        "key,tile_y,tile_x,targets,oracle_nms_065,oracle_nms_080,oracle_nms_095,cap_300\n",
    );
    for key in 0..store.len() {
        let sample = store.load(key)?;
        if key == 0 {
            write_preview(&output.join(format!("{name}-sample.png")), &sample)?;
        }
        let predicted = sample
            .targets
            .iter()
            .map(|target| {
                [
                    target.x1,
                    target.y1,
                    target.x2,
                    target.y2,
                    1.0,
                    target.class,
                ]
            })
            .collect::<Vec<_>>();
        let ground_truth = sample
            .targets
            .iter()
            .map(|target| [target.class, target.x1, target.y1, target.x2, target.y2])
            .collect::<Vec<_>>();
        let oracle_matches =
            yolov11::model::metrics::compute_metric(&predicted, &ground_truth, &[0.5, 0.75, 0.95]);
        anyhow::ensure!(
            oracle_matches
                .iter()
                .all(|thresholds| thresholds.iter().all(|matched| *matched)),
            "perfect-prediction metric oracle failed for {name} tile {key}"
        );
        for (left, target) in sample.targets.iter().enumerate() {
            areas.push(((target.x2 - target.x1) * (target.y2 - target.y1)) as usize);
            overlapping_pairs_at_65 += sample.targets[left + 1..]
                .iter()
                .filter(|right| box_iou(target, right) > 0.65)
                .count();
        }
        let (candidates, contested_locations, contested_relations) =
            assignment_geometry(&sample.targets);
        assignment_candidates.extend(candidates);
        contested_anchor_locations += contested_locations;
        contested_anchor_relations += contested_relations;
        let retained_65 = oracle_nms_count(&sample.targets, 0.65, 1000);
        let retained_80 = oracle_nms_count(&sample.targets, 0.80, 1000);
        let retained_95 = oracle_nms_count(&sample.targets, 0.95, 1000);
        let retained_cap = sample.targets.len().min(300);
        oracle_at_65 += retained_65;
        oracle_at_80 += retained_80;
        oracle_at_95 += retained_95;
        oracle_cap_300 += retained_cap;
        let tile = store.tiles[key];
        writeln!(
            csv,
            "{key},{},{},{},{retained_65},{retained_80},{retained_95},{retained_cap}",
            tile.y,
            tile.x,
            sample.targets.len()
        )?;
        target_counts.push(sample.targets.len());
    }
    std::fs::write(output.join(format!("{name}-density.csv")), csv)?;
    let targets: usize = target_counts.iter().sum();
    target_counts.sort_unstable();
    areas.sort_unstable();
    assignment_candidates.sort_unstable();
    let recall = |retained: usize| retained as f64 / targets.max(1) as f64;
    Ok(serde_json::json!({
        "tiles": store.len(),
        "targets": targets,
        "empty_tiles": target_counts.iter().filter(|&&count| count == 0).count(),
        "min_targets_per_tile": target_counts.iter().min().copied().unwrap_or(0),
        "max_targets_per_tile": target_counts.iter().max().copied().unwrap_or(0),
        "mean_targets_per_tile": targets as f64 / store.len().max(1) as f64,
        "target_count_percentiles": {
            "p50": percentile(&target_counts, 50, 100),
            "p90": percentile(&target_counts, 90, 100),
            "p95": percentile(&target_counts, 95, 100),
            "p99": percentile(&target_counts, 99, 100),
        },
        "box_area_percentiles": {
            "p01": percentile(&areas, 1, 100),
            "p50": percentile(&areas, 50, 100),
            "p99": percentile(&areas, 99, 100),
        },
        "tiles_over_300_targets": target_counts.iter().filter(|&&count| count > 300).count(),
        "overlapping_box_pairs_iou_over_065": overlapping_pairs_at_65,
        "assignment_geometry": {
            "zero_candidate_targets": assignment_candidates.iter().filter(|&&count| count == 0).count(),
            "candidate_locations_per_target": {
                "p01": percentile(&assignment_candidates, 1, 100),
                "p50": percentile(&assignment_candidates, 50, 100),
                "p99": percentile(&assignment_candidates, 99, 100),
            },
            "contested_anchor_locations": contested_anchor_locations,
            "target_anchor_relations_at_contested_locations": contested_anchor_relations,
        },
        "perfect_prediction_metric": "passed",
        "oracle_recall": {
            "max_300_without_nms": recall(oracle_cap_300),
            "nms_iou_065_max_1000": recall(oracle_at_65),
            "nms_iou_080_max_1000": recall(oracle_at_80),
            "nms_iou_095_max_1000": recall(oracle_at_95),
        },
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    }))
}

fn metrics_json(metrics: EvaluationMetrics) -> serde_json::Value {
    serde_json::json!({
        "mAP": metrics.mean_ap,
        "mAP@50": metrics.map50,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "confidence_threshold": metrics.confidence_threshold,
    })
}

fn nonempty_axis<T: Clone>(values: &[T], fallback: T) -> Vec<T> {
    if values.is_empty() {
        vec![fallback]
    } else {
        values.to_vec()
    }
}

fn write_sweep_outputs(
    output: &Path,
    baseline: EvaluationMetrics,
    reports: &[SweepTrialReport],
) -> Result<()> {
    let mut ranked = reports.to_vec();
    ranked.sort_by(|left, right| {
        right
            .best_validation
            .mean_ap
            .partial_cmp(&left.best_validation.mean_ap)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.id.cmp(&right.id))
    });
    let summary = serde_json::json!({
        "baseline": metrics_json(baseline),
        "test_evaluated": false,
        "trials": ranked,
    });
    std::fs::write(
        output.join("sweep.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    let mut csv = String::from(
        "rank,id,active,completed_epochs,max_lr,min_lr,weight_decay,gradient_clip,geometry,trainable_layers,batch_norm,best_epoch,mAP,mAP@50,recall,precision,elapsed_seconds\n",
    );
    for (rank, trial) in ranked.iter().enumerate() {
        csv.push_str(&format!(
            "{},{},{},{},{:.8},{:.8},{:.8},{},{:?},{:?},{:?},{},{:.8},{:.8},{:.8},{:.8},{:.3}\n",
            rank + 1,
            trial.id,
            trial.active,
            trial.completed_epochs,
            trial.max_lr,
            trial.min_lr,
            trial.weight_decay,
            trial
                .gradient_clip
                .map_or_else(|| "off".to_owned(), |value| value.to_string()),
            trial.geometry,
            trial.trainable_layers,
            trial.batch_norm,
            trial.best_epoch,
            trial.best_validation.mean_ap,
            trial.best_validation.map50,
            trial.best_validation.recall,
            trial.best_validation.precision,
            trial.elapsed_seconds,
        ));
    }
    std::fs::write(output.join("sweep.csv"), csv)?;
    Ok(())
}

fn run_sweep(
    args: &Args,
    base_config: &YoloConfig,
    train: SharedSampleStore,
    validation: SharedSampleStore,
    device: &Device,
) -> Result<()> {
    let sweep_path = args.sweep.as_deref().context("missing sweep path")?;
    let spec: SweepSpec = serde_yaml::from_slice(
        &std::fs::read(sweep_path)
            .with_context(|| format!("reading sweep specification {}", sweep_path.display()))?,
    )
    .with_context(|| format!("parsing sweep specification {}", sweep_path.display()))?;
    anyhow::ensure!(!spec.rounds.is_empty(), "a sweep needs at least one round");
    for (index, round) in spec.rounds.iter().enumerate() {
        anyhow::ensure!(
            round.epochs > 0,
            "sweep round {} has zero epochs",
            index + 1
        );
        anyhow::ensure!(round.keep > 0, "sweep round {} keeps no trials", index + 1);
    }

    let max_lrs = nonempty_axis(&spec.max_lr, base_config.max_lr);
    let min_lr_ratios = nonempty_axis(&spec.min_lr_ratio, base_config.min_lr / base_config.max_lr);
    let weight_decays = nonempty_axis(&spec.weight_decay, base_config.weight_decay);
    let gradient_clips = nonempty_axis(&spec.gradient_clip, args.gradient_clip);
    let geometries = nonempty_axis(
        &spec.geometry,
        match args.geometry {
            GeometryArg::None => SweepGeometry::None,
            GeometryArg::D4 => SweepGeometry::D4,
        },
    );
    let trainable_layers = nonempty_axis(&spec.trainable_layers, TrainableLayers::All);
    let batch_norms = nonempty_axis(&spec.batch_norm, BatchNormPolicy::Update);
    let trial_count = max_lrs.len()
        * min_lr_ratios.len()
        * weight_decays.len()
        * gradient_clips.len()
        * geometries.len()
        * trainable_layers.len()
        * batch_norms.len();
    anyhow::ensure!(
        trial_count <= 256,
        "sweep expands to {trial_count} trials; limit is 256"
    );
    let schedule_epochs: usize = spec.rounds.iter().map(|round| round.epochs).sum();
    let trials_dir = args.output.join("trials");
    std::fs::create_dir_all(&trials_dir)?;

    let baseline_options = TrainingOptions {
        input_size: INPUT,
        batch_size: args.batch_size,
        loader_workers: args.workers,
        queue_batches: args.queue_batches,
        seed: args.seed,
        evaluation_confidence_threshold: args.evaluation_confidence,
        evaluation_iou_threshold: args.evaluation_nms_iou,
        evaluation_max_detections: args.evaluation_max_detections,
        output_dir: args.output.clone(),
        ..TrainingOptions::default()
    };
    let weights = args
        .weights
        .as_deref()
        .context("a sweep requires --weights")?;
    let baseline = evaluate_checkpoint_with_store(
        base_config,
        weights,
        Arc::clone(&validation),
        &baseline_options,
        device,
    )?;

    let mut trials = Vec::with_capacity(trial_count);
    for max_lr in max_lrs {
        anyhow::ensure!(
            max_lr.is_finite() && max_lr > 0.0,
            "max_lr must be positive"
        );
        for min_lr_ratio in &min_lr_ratios {
            anyhow::ensure!(
                min_lr_ratio.is_finite() && *min_lr_ratio > 0.0 && *min_lr_ratio <= 1.0,
                "min_lr_ratio must be in (0, 1]"
            );
            for weight_decay in &weight_decays {
                anyhow::ensure!(
                    weight_decay.is_finite() && *weight_decay >= 0.0,
                    "weight_decay must be non-negative"
                );
                for gradient_clip in &gradient_clips {
                    anyhow::ensure!(
                        gradient_clip.is_finite() && *gradient_clip >= 0.0,
                        "gradient_clip must be non-negative"
                    );
                    for geometry in &geometries {
                        for trainable in &trainable_layers {
                            for batch_norm in &batch_norms {
                                let id = format!("trial-{:03}", trials.len());
                                let mut config = base_config.clone();
                                config.max_lr = max_lr;
                                config.min_lr = max_lr * min_lr_ratio;
                                config.weight_decay = *weight_decay;
                                let mut report = SweepTrialReport {
                                    id: id.clone(),
                                    max_lr,
                                    min_lr: config.min_lr,
                                    weight_decay: *weight_decay,
                                    gradient_clip: (*gradient_clip > 0.0).then_some(*gradient_clip),
                                    geometry: *geometry,
                                    trainable_layers: *trainable,
                                    batch_norm: *batch_norm,
                                    completed_epochs: 0,
                                    active: true,
                                    best_epoch: 0,
                                    best_validation: EvaluationMetrics::default(),
                                    elapsed_seconds: 0.0,
                                };
                                let report_path = trials_dir.join(&id).join("sweep-trial.json");
                                if report_path.is_file() {
                                    let previous: SweepTrialReport =
                                        serde_json::from_slice(&std::fs::read(&report_path)?)?;
                                    anyhow::ensure!(
                                        previous.max_lr == report.max_lr
                                            && previous.min_lr == report.min_lr
                                            && previous.weight_decay == report.weight_decay
                                            && previous.gradient_clip == report.gradient_clip
                                            && previous.geometry == report.geometry
                                            && previous.trainable_layers == report.trainable_layers
                                            && previous.batch_norm == report.batch_norm,
                                        "{} has different parameters in the existing sweep",
                                        id
                                    );
                                    report = previous;
                                    report.active = true;
                                }
                                trials.push(SweepTrial { report, config });
                            }
                        }
                    }
                }
            }
        }
    }

    let mut target_epochs = 0usize;
    for (round_index, round) in spec.rounds.iter().enumerate() {
        target_epochs += round.epochs;
        for trial in trials.iter_mut().filter(|trial| trial.report.active) {
            if trial.report.completed_epochs >= target_epochs {
                continue;
            }
            anyhow::ensure!(
                trial.report.completed_epochs == target_epochs - round.epochs,
                "{} stopped at epoch {}, which is not a sweep round boundary",
                trial.report.id,
                trial.report.completed_epochs
            );
            let trial_dir = trials_dir.join(&trial.report.id);
            std::fs::create_dir_all(&trial_dir)?;
            let resume = (trial.report.completed_epochs > 0).then_some(trial_dir.clone());
            let options = TrainingOptions {
                input_size: INPUT,
                batch_size: args.batch_size,
                epochs: round.epochs,
                eval_interval: 1,
                loader_workers: args.workers,
                queue_batches: args.queue_batches,
                seed: args.seed,
                geometry: trial.report.geometry.into(),
                weights: resume.is_none().then(|| weights.to_path_buf()),
                reset_class_head: resume.is_none() && args.reset_class_head,
                gradient_clip: trial.report.gradient_clip,
                trainable_layers: trial.report.trainable_layers,
                batch_norm: trial.report.batch_norm,
                evaluation_confidence_threshold: args.evaluation_confidence,
                evaluation_iou_threshold: args.evaluation_nms_iou,
                evaluation_max_detections: args.evaluation_max_detections,
                schedule_epochs: Some(schedule_epochs),
                resume,
                finalize: false,
                output_dir: trial_dir.clone(),
            };
            let started = Instant::now();
            let summary = train_with_stores_and_test(
                &trial.config,
                Arc::clone(&train),
                Some(Arc::clone(&validation)),
                None,
                &options,
                device,
            )?;
            trial.report.completed_epochs = summary.completed_epochs;
            trial.report.best_epoch = summary.best_epoch;
            trial.report.best_validation = summary.best_validation;
            trial.report.elapsed_seconds += started.elapsed().as_secs_f64();
            std::fs::write(
                trial_dir.join("sweep-trial.json"),
                serde_json::to_vec_pretty(&trial.report)?,
            )?;
        }

        let mut active = trials
            .iter()
            .enumerate()
            .filter(|(_, trial)| trial.report.active)
            .map(|(index, trial)| (index, trial.report.best_validation.mean_ap))
            .collect::<Vec<_>>();
        active.sort_by(|(left_index, left_map), (right_index, right_map)| {
            right_map
                .partial_cmp(left_map)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    trials[*left_index]
                        .report
                        .id
                        .cmp(&trials[*right_index].report.id)
                })
        });
        let keep = round.keep.min(active.len());
        for (rank, (index, _)) in active.into_iter().enumerate() {
            trials[index].report.active = rank < keep;
            let trial_dir = trials_dir.join(&trials[index].report.id);
            std::fs::write(
                trial_dir.join("sweep-trial.json"),
                serde_json::to_vec_pretty(&trials[index].report)?,
            )?;
        }
        let reports = trials
            .iter()
            .map(|trial| trial.report.clone())
            .collect::<Vec<_>>();
        write_sweep_outputs(&args.output, baseline, &reports)?;
        println!(
            "Sweep round {}/{} complete: {} epochs, keeping {} trial(s)",
            round_index + 1,
            spec.rounds.len(),
            target_epochs,
            keep
        );
    }
    Ok(())
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
        "version": 2,
        "zarr": args.zarr.display().to_string(),
        "layer": &args.layer,
        "channel": args.channel,
        "normalization": [args.low, args.high],
        "shape": shape,
        "core": CORE,
        "halo": HALO,
        "targets": "all-complete-visible-instances",
        "train_validation_guard": 2 * HALO,
        "negative_tile_fraction": args.negative_tile_fraction,
        "overfit_density": args.overfit_density.map(|value| format!("{value:?}").to_lowercase()),
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
    anyhow::ensure!(
        (0.0..1.0).contains(&args.negative_tile_fraction),
        "--negative-tile-fraction must be in [0, 1)"
    );
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
    let occupied_set = occupied.iter().copied().collect::<BTreeSet<_>>();
    let (train_tiles, validation_tiles, test_tiles, validation_cut) = split_tiles(occupied, shape);
    let backgrounds = background_tiles(all_tiles(shape), &occupied_set);
    let (train_backgrounds, validation_backgrounds, test_backgrounds) =
        split_tiles_at(backgrounds, shape, validation_cut);
    let (train_tiles, validation_tiles) = if let Some(density) = args.overfit_density {
        anyhow::ensure!(
            args.negative_tile_fraction == 0.0,
            "--overfit-density cannot be combined with background tiles"
        );
        let TileCount::Count(train_limit) = args.train_tiles else {
            anyhow::bail!("--overfit-density requires a numeric --train-tiles limit");
        };
        let train_tiles = select_overfit_tiles(train_tiles, &centroids, train_limit, density);
        (train_tiles.clone(), train_tiles)
    } else {
        (
            select_with_background(
                train_tiles,
                train_backgrounds,
                args.train_tiles,
                args.negative_tile_fraction,
            ),
            select_with_background(
                validation_tiles,
                validation_backgrounds,
                TileCount::Count(args.validation_tiles),
                args.negative_tile_fraction,
            ),
        )
    };
    let test_tiles = select_with_background(
        test_tiles,
        test_backgrounds,
        TileCount::Count(args.test_tiles),
        args.negative_tile_fraction,
    );
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
        "objects={} train_tiles={} ({} background) validation_tiles={} ({} background) frozen_test_tiles={} ({} background) shape={}x{}",
        centroids.len(),
        train_tiles.len(),
        train_tiles.iter().filter(|tile| !occupied_set.contains(tile)).count(),
        validation_tiles.len(),
        validation_tiles.iter().filter(|tile| !occupied_set.contains(tile)).count(),
        test_tiles.len(),
        test_tiles.iter().filter(|tile| !occupied_set.contains(tile)).count(),
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
        shape,
        args.low,
        args.high,
    )?);
    let validation = Arc::new(OmeDapiStore::new(
        Arc::clone(&env),
        validation_tiles,
        shape,
        args.low,
        args.high,
    )?);
    let test = Arc::new(OmeDapiStore::new(
        Arc::clone(&env),
        test_tiles,
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
        weights: args.weights.clone(),
        reset_class_head: args.reset_class_head,
        gradient_clip: (args.gradient_clip > 0.0).then_some(args.gradient_clip),
        trainable_layers: args.trainable_layers.into(),
        batch_norm: args.batch_norm.into(),
        evaluation_confidence_threshold: args.evaluation_confidence,
        evaluation_iou_threshold: args.evaluation_nms_iou,
        evaluation_max_detections: args.evaluation_max_detections,
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
    if args.sweep.is_some() {
        run_sweep(&args, &config, train, validation, &device)?;
        env.drain_prefetch();
        println!(
            "sweep_elapsed={:.3}s output={}",
            started.elapsed().as_secs_f64(),
            args.output.display()
        );
        return finish_process();
    }
    if args.evaluate_only {
        let weights = options
            .weights
            .as_deref()
            .expect("clap requires weights for evaluate-only");
        let metrics =
            evaluate_checkpoint_with_store(&config, weights, validation, &options, &device)?;
        env.drain_prefetch();
        let report = serde_json::json!({
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "weights": weights.display().to_string(),
            "validation_tiles": actual_validation_tiles,
            "validation": metrics_json(metrics),
        });
        std::fs::write(
            args.output.join("validation-evaluation.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return finish_process();
    }
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
        "negative_tile_fraction": args.negative_tile_fraction,
        "epochs": args.epochs,
        "stage": summary.stage,
        "completed_epochs": summary.completed_epochs,
        "schedule_epochs": summary.schedule_epochs,
        "finalized": args.finalize,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "geometry": format!("{:?}", args.geometry).to_lowercase(),
        "trainable_layers": format!("{:?}", args.trainable_layers).to_lowercase(),
        "batch_norm": format!("{:?}", args.batch_norm).to_lowercase(),
        "overfit_density": args.overfit_density.map(|value| format!("{value:?}").to_lowercase()),
        "evaluation_confidence": args.evaluation_confidence,
        "evaluation_nms_iou": args.evaluation_nms_iou,
        "evaluation_max_detections": args.evaluation_max_detections,
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
    finish_process()
}
