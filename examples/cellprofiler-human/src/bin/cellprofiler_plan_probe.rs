// SPDX-License-Identifier: MIT

#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::items_after_test_module,
    clippy::manual_is_multiple_of
)]

#[path = "../../../support/planning.rs"]
mod example_planning;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use blockflow::assemble::{ImageId, Phase, PlanBuilder};
use blockflow::dtype::Dtype;
use blockflow::env::BlockBuf;
use blockflow::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, PhaseView,
    SeamFold, SourceBlocks,
};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::op::SourceInput;
use blockflow::ops::distance;
use blockflow::ops::label::{LabelComponentsOp, RelabelComponentsOp};
use blockflow::ops::measure::{
    collect_class_a_shapes, collect_class_a_values, IntensityImage, IntensityMeasurements,
    IntensitySet, Measurements, ShapeMeasurements, ShapeSet,
};
use blockflow::ops::regional;
use blockflow::ops::{
    append_fill_label_holes_2d_by_label_phase, append_filter_labels_by_size_phases,
    append_filter_labels_touching_border_on_axes_phase, append_global_threshold_phases,
    append_remove_small_objects_phases, regional_maxima, Boundary, Connectivity, DistanceParams,
    Gaussian, GlobalThreshold, GlobalThresholdOutput, GlobalThresholdSelection, SeededWatershedOp,
    Separation, SmoothOp, ThresholdTest, VoxelwiseMapOp,
};
use blockflow::reach::Reach;
use blockflow::sidecar::Lifecycle;
use blockflow::simulate::{ExecutorOrder, Machine, Rates, Run};
use blockflow::strategy::{execute_phases, Hints};
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::{AttachedImage, Error, Result};
use clap::Parser;
use image::{ImageBuffer, Luma};
use ndarray::Array3;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "cellprofiler-human")]
struct Cli {
    #[arg(long, required_unless_present = "input_zarr")]
    input: Option<PathBuf>,
    #[arg(long)]
    input_zarr: Option<PathBuf>,
    #[arg(long)]
    ensure_input_zarr: Option<PathBuf>,
    /// Prepare the input array and exit before planning or processing.
    #[arg(long, default_value_t = false)]
    prepare_only: bool,
    #[arg(long, default_value = "cellprofiler-human.json")]
    out: PathBuf,
    #[arg(long, value_parser = parse_chunk, default_value = "1x256x256")]
    chunk: [usize; 3],
    #[arg(long, default_value_t = 1)]
    workers: usize,
    #[arg(long, default_value_t = 0)]
    cache_bytes: u64,
    #[arg(long, default_value_t = 1.0)]
    sigma: f64,
    #[arg(long, default_value_t = 1.0)]
    background_percentile: f64,
    #[arg(long)]
    no_background_subtract: bool,
    #[arg(long, value_enum, default_value = "li")]
    threshold_method: ThresholdMethod,
    #[arg(long, default_value_t = 256)]
    threshold_bins: usize,
    #[arg(long, default_value_t = 50)]
    min_size: u64,
    #[arg(long, default_value_t = 5027)]
    max_size: u64,
    #[arg(long)]
    no_max_size: bool,
    #[arg(long, default_value_t = 6.0)]
    seed_min_distance: f64,
    #[arg(long, default_value_t = 3)]
    maxima_downsample: usize,
    #[arg(long, default_value_t = 1.3488)]
    declump_sigma: f64,
    #[arg(long, value_enum, default_value = "intensity")]
    declump_method: DeclumpMethod,
    #[arg(long, default_value_t = 0)]
    merge_line_basin_pixels: usize,
    #[arg(long)]
    merge_line_max_saddle_drop: Option<f64>,
    #[arg(long, default_value_t = 256)]
    distance_block: usize,
    #[arg(long, default_value = "cellprofiler-output")]
    materialize_objects: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    materialize_repeats: usize,
}

#[derive(Debug)]
struct Config {
    input: Option<PathBuf>,
    input_zarr: Option<PathBuf>,
    ensure_input_zarr: Option<PathBuf>,
    prepare_only: bool,
    out: PathBuf,
    chunk: [usize; 3],
    workers: usize,
    cache_bytes: u64,
    sigma: f64,
    background_percentile: Option<f64>,
    threshold_method: ThresholdMethod,
    threshold_bins: usize,
    min_size: u64,
    max_size: Option<u64>,
    seed_min_distance: f64,
    maxima_downsample: usize,
    declump_sigma: f64,
    declump_method: DeclumpMethod,
    merge_line_basin_pixels: usize,
    merge_line_max_saddle_drop: Option<f64>,
    distance_block: usize,
    materialize_objects: Option<PathBuf>,
    materialize_repeats: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, clap::ValueEnum)]
enum ThresholdMethod {
    #[value(alias = "minimum-cross-entropy", alias = "minimum_cross_entropy")]
    Li,
    Otsu,
}

impl ThresholdMethod {
    fn as_str(self) -> &'static str {
        match self {
            Self::Li => "li",
            Self::Otsu => "otsu",
        }
    }

    fn global_threshold(self, bins: usize) -> Result<GlobalThreshold> {
        match self {
            Self::Li => Ok(GlobalThreshold::Li),
            Self::Otsu => GlobalThreshold::otsu(bins),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, clap::ValueEnum)]
enum DeclumpMethod {
    Intensity,
    #[value(alias = "shape")]
    Distance,
}

impl DeclumpMethod {
    fn as_str(self) -> &'static str {
        match self {
            Self::Intensity => "intensity",
            Self::Distance => "distance",
        }
    }
}

impl Config {
    fn parse() -> Result<Self> {
        let cli = Cli::parse();

        let config = Self {
            input: cli.input,
            input_zarr: cli.input_zarr,
            ensure_input_zarr: cli.ensure_input_zarr,
            prepare_only: cli.prepare_only,
            out: cli.out,
            chunk: cli.chunk,
            workers: cli.workers,
            cache_bytes: cli.cache_bytes,
            sigma: cli.sigma,
            background_percentile: (!cli.no_background_subtract)
                .then_some(cli.background_percentile),
            threshold_method: cli.threshold_method,
            threshold_bins: cli.threshold_bins,
            min_size: cli.min_size,
            max_size: (!cli.no_max_size).then_some(cli.max_size),
            seed_min_distance: cli.seed_min_distance,
            maxima_downsample: cli.maxima_downsample,
            declump_sigma: cli.declump_sigma,
            declump_method: cli.declump_method,
            merge_line_basin_pixels: cli.merge_line_basin_pixels,
            merge_line_max_saddle_drop: cli.merge_line_max_saddle_drop,
            distance_block: cli.distance_block,
            materialize_objects: cli.materialize_objects,
            materialize_repeats: cli.materialize_repeats,
        };
        config.validate()
    }

    fn validate(self) -> Result<Self> {
        if self.chunk.contains(&0) {
            return Err(Error::invalid(
                "cellprofiler-human: --chunk dimensions must be positive",
            ));
        }
        if self.input_zarr.is_some() && self.ensure_input_zarr.is_some() {
            return Err(Error::invalid(
                "cellprofiler-human: use either --input-zarr or --ensure-input-zarr, not both",
            ));
        }
        if self.ensure_input_zarr.is_some() && self.input.is_none() {
            return Err(Error::invalid(
                "cellprofiler-human: --ensure-input-zarr needs --input for fixture preparation",
            ));
        }
        if self.workers == 0 {
            return Err(Error::invalid(
                "cellprofiler-human: --workers must be positive",
            ));
        }
        if self.threshold_bins < 2 {
            return Err(Error::invalid(
                "cellprofiler-human: --threshold-bins must be at least 2",
            ));
        }
        if self.sigma < 0.0 || !self.sigma.is_finite() {
            return Err(Error::invalid(
                "cellprofiler-human: --sigma must be finite and non-negative",
            ));
        }
        if let Some(percentile) = self.background_percentile {
            if !(0.0..=100.0).contains(&percentile) || !percentile.is_finite() {
                return Err(Error::invalid(
                    "cellprofiler-human: --background-percentile must be finite and in [0, 100]",
                ));
            }
        }
        if self.distance_block == 0 {
            return Err(Error::invalid(
                "cellprofiler-human: --distance-block must be positive",
            ));
        }
        if let Some(max_size) = self.max_size {
            if max_size < self.min_size {
                return Err(Error::invalid(
                    "cellprofiler-human: --max-size must be at least --min-size",
                ));
            }
        }
        if self.seed_min_distance < 0.0 || !self.seed_min_distance.is_finite() {
            return Err(Error::invalid(
                "cellprofiler-human: --seed-min-distance must be finite and non-negative",
            ));
        }
        if self.maxima_downsample == 0 {
            return Err(Error::invalid(
                "cellprofiler-human: --maxima-downsample must be at least 1",
            ));
        }
        if self.declump_sigma < 0.0 || !self.declump_sigma.is_finite() {
            return Err(Error::invalid(
                "cellprofiler-human: --declump-sigma must be finite and non-negative",
            ));
        }
        if let Some(max_saddle_drop) = self.merge_line_max_saddle_drop {
            if !max_saddle_drop.is_finite() {
                return Err(Error::invalid(
                    "cellprofiler-human: --merge-line-max-saddle-drop must be finite",
                ));
            }
        }
        if self.materialize_repeats == 0 {
            return Err(Error::invalid(
                "cellprofiler-human: --materialize-repeats must be at least 1",
            ));
        }
        Ok(self)
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = Config::parse()?;
    let source = input_source(&config)?;
    if config.prepare_only {
        println!("prepared {}", source.array.display());
        return Ok(());
    }
    let volume = source.volume;
    let planned = build_planned_probe(volume, &config)?;
    let rates = Rates {
        chunk: config.chunk,
        chunk_bytes: chunk_bytes(config.chunk, 8),
        ..Rates::default()
    };
    let machine = Machine {
        workers: config.workers,
        cache_bytes: config.cache_bytes,
        ..Machine::default()
    };
    let mut work = planned.measurement_base.work();
    work.extend(planned.measurements.phase_work());
    let mut scheduler = ExecutorOrder::phase_major();
    let outcome = Run::new(&planned.measurements.decomposition, &work)
        .machine(machine)
        .rates(rates)
        .go(&mut scheduler)?;
    let materialized = match &config.materialize_objects {
        Some(out_dir) => Some(materialize_planned_objects_repeated(
            &planned, &source, out_dir, &config,
        )?),
        None => None,
    };

    let phase_blocks = planned
        .measurements
        .decomposition
        .phases
        .iter()
        .map(|phase| {
            json!({
                "blocks": phase.grid.n_blocks(),
                "blocks_per_axis": phase.grid.blocks_per_axis(),
                "dtype": format!("{:?}", phase.dtype.unwrap_or(Dtype::F64)),
            })
        })
        .collect::<Vec<_>>();
    let mut included = vec![
        "global Li threshold mask",
        "remove small objects",
        "regional maxima",
        "minimum-distance seed suppression",
        "maxima component labelling",
        "declump-source watershed cost",
        "seeded watershed",
        "post-declump per-label hole filling",
    ];
    if config.merge_line_basin_pixels != 0 {
        included.push("planned watershed-line basin merge");
    }
    included.push("final label border filtering");
    included.push("final object-size filtering");
    included.push("object shape tabulation");
    included.push("object intensity tabulation on the input image");
    included.push("measurement row merge");
    if config.sigma != 0.0 {
        included.insert(0, "threshold XY Gaussian smoothing");
    }
    if config.background_percentile.is_some() {
        included.insert(0, "background percentile subtraction");
    }
    match config.declump_method {
        DeclumpMethod::Intensity => included.insert(
            (usize::from(config.background_percentile.is_some())
                + usize::from(config.sigma != 0.0))
                + 3,
            "intensity declump XY Gaussian smoothing",
        ),
        DeclumpMethod::Distance => included.insert(
            (usize::from(config.background_percentile.is_some())
                + usize::from(config.sigma != 0.0))
                + 3,
            "distance transform",
        ),
    }

    let materialized_object_count = materialized
        .as_ref()
        .and_then(|value| value.get("objects"))
        .and_then(|value| value.as_u64());
    let not_included = match materialized_object_count {
        Some(1..) => Vec::<&str>::new(),
        Some(0) => vec!["validated non-empty planned CSV/table materialization"],
        None => vec!["planned CSV/table materialization"],
    };
    let report = json!({
        "status": "planned_segmentation_with_measurements_simulated",
        "scope": {
            "included": included,
            "not_included": not_included,
            "reason": "this is the primary Blockflow example path: it exposes the planned segmentation skeleton, min-distance seed suppression, optional watershed-line basin merging, final object-size filtering and shape/intensity measurement phases to the simulator, and materializes planned measurement rows when requested"
        },
        "input": source.description,
        "input_zarr": source.array.display().to_string(),
        "volume": volume,
        "requested": {
            "workers": config.workers,
            "chunk_shape": config.chunk,
            "cache_bytes": config.cache_bytes,
            "sigma": config.sigma,
            "background_percentile": config.background_percentile,
            "threshold_method": config.threshold_method.as_str(),
            "threshold_bins": config.threshold_bins,
            "min_size": config.min_size,
            "max_size": config.max_size,
            "seed_min_distance": config.seed_min_distance,
            "maxima_downsample": config.maxima_downsample,
            "declump_sigma": config.declump_sigma,
            "declump_method": config.declump_method.as_str(),
            "merge_line_basin_pixels": config.merge_line_basin_pixels,
            "merge_line_max_saddle_drop": config.merge_line_max_saddle_drop,
            "distance_block": config.distance_block,
            "materialize_repeats": config.materialize_repeats,
            "watershed_separation": "line"
        },
        "plan": {
            "skeleton_phases": planned.skeleton.n_phases(),
            "measurement_rows_phase": planned.measurements.rows_phase,
            "phases": planned.measurements.decomposition.n_phases(),
            "images": planned.measurements.decomposition.n_images(),
            "phase_blocks": phase_blocks
        },
        "simulator": {
            "scheduler": "phase-major",
            "estimated_pipeline_seconds": outcome.makespan_ns as f64 / 1.0e9,
            "makespan_ns": outcome.makespan_ns,
            "tasks_run": outcome.tasks_run,
            "peak_bytes": outcome.peak_bytes,
            "fetched_bytes": outcome.fetched_bytes,
            "written_bytes": outcome.written_bytes,
            "materialised_bytes": outcome.materialised_bytes,
            "cache_hits": outcome.cache_hits,
            "cache_misses": outcome.cache_misses,
            "prefetched_bytes": outcome.prefetched_bytes,
            "duplicated_fetches": outcome.duplicated_fetches,
            "sidecar_bytes_written": outcome.sidecar_bytes_written,
            "sidecar_gather_peak": outcome.sidecar_gather_peak,
            "phase_span_ns": outcome.phase_span_ns,
            "phase_overlap": outcome.phase_overlap()
        },
        "materialized_outputs": materialized
    });
    let text = serde_json::to_string_pretty(&report)
        .map_err(|err| Error::invalid(format!("cellprofiler-human: encode JSON: {err}")))?;
    fs::write(&config.out, format!("{text}\n")).map_err(|err| {
        Error::invalid(format!(
            "cellprofiler-human: write {}: {err}",
            config.out.display()
        ))
    })?;
    println!(
        "planned_measurement_skeleton_phases={} tasks={} simulated_seconds={:.6} output={}",
        planned.measurements.decomposition.n_phases(),
        outcome.tasks_run,
        outcome.makespan_ns as f64 / 1.0e9,
        config.out.display()
    );
    Ok(())
}

struct PlannedProbe {
    skeleton: blockflow::assemble::Assembly,
    measurement_base: blockflow::assemble::Assembly,
    measurements: blockflow::ops::measure::MeasurementPlan,
}

struct InputSource {
    array: PathBuf,
    volume: [usize; 3],
    description: String,
}

fn build_planned_probe(volume: [usize; 3], config: &Config) -> Result<PlannedProbe> {
    let skeleton = build_planned_skeleton(volume, config)?;
    let measurement_base = build_planned_skeleton(volume, config)?;
    let labels = ImageId::from(measurement_base.decomposition.n_phases());
    let measurements = Measurements::for_labels(labels)
        .shape(ShapeSet::standard())
        .intensity(IntensityImage::<0>::new(0usize), IntensitySet::standard())
        .stream("cellprofiler.objects")
        .lifecycle(Lifecycle::DeleteOnExit)
        .build(measurement_base.decomposition.clone())?;
    Ok(PlannedProbe {
        skeleton,
        measurement_base,
        measurements,
    })
}

fn materialize_planned_objects(
    planned: &PlannedProbe,
    source: &InputSource,
    out_dir: &Path,
    _config: &Config,
) -> Result<serde_json::Value> {
    fs::create_dir_all(out_dir).map_err(|err| {
        Error::invalid(format!(
            "cellprofiler-human: create materialization directory {}: {err}",
            out_dir.display()
        ))
    })?;
    let started = Instant::now();
    let scratch = tempfile::tempdir_in(out_dir).map_err(Error::backend)?;
    let env = ZarrEnvironment::attach(scratch.path(), &[AttachedImage::at(source.array.clone())])?;
    let label_image = planned.measurement_base.decomposition.n_phases();
    let mut work = planned.measurement_base.work();
    work.extend(planned.measurements.phase_work());
    let hints = Hints {
        keep_images: [ImageId::from(label_image)].into_iter().collect(),
        ..Hints::default()
    };
    execute_phases(
        "cellprofiler planned materialization",
        &planned.measurement_base.workflow,
        &planned.measurements.decomposition,
        &hints,
        &env,
        &[],
        &work,
    )?;
    let label_stats = planned_label_stats(&env, label_image)?;
    let labels_png = out_dir.join("labels.png");
    save_planned_labels(&env, label_image, &labels_png)?;
    let rows = collect_planned_object_rows(&env, planned)?;
    let objects_csv = out_dir.join("planned_objects.csv");
    write_planned_object_csv(&rows, &objects_csv)?;
    let total_area: u64 = rows.iter().map(|row| row.shape.count).sum();
    let summary = json!({
        "objects": rows.len(),
        "total_foreground_area": total_area,
        "mean_object_area": if rows.is_empty() { 0.0 } else { total_area as f64 / rows.len() as f64 },
        "label_image": label_stats,
        "labels_png": labels_png.display().to_string(),
        "objects_csv": objects_csv.display().to_string(),
        "measurement_intensity_source": "input_luma_normalized_0_1",
        "input_zarr": source.array.display().to_string(),
        "seconds": started.elapsed().as_secs_f64()
    });
    let summary_path = out_dir.join("planned-summary.json");
    let summary_text = serde_json::to_string_pretty(&summary).map_err(|err| {
        Error::invalid(format!(
            "cellprofiler-human: encode materialization summary: {err}"
        ))
    })?;
    fs::write(&summary_path, format!("{summary_text}\n")).map_err(|err| {
        Error::invalid(format!(
            "cellprofiler-human: write {}: {err}",
            summary_path.display()
        ))
    })?;
    Ok(json!({
        "objects_csv": objects_csv.display().to_string(),
        "summary_json": summary_path.display().to_string(),
        "objects": rows.len(),
        "label_image": label_stats,
        "labels_png": labels_png.display().to_string(),
        "seconds": started.elapsed().as_secs_f64()
    }))
}

fn materialize_planned_objects_repeated(
    planned: &PlannedProbe,
    source: &InputSource,
    out_dir: &Path,
    config: &Config,
) -> Result<serde_json::Value> {
    let mut summaries = Vec::with_capacity(config.materialize_repeats);
    for _ in 0..config.materialize_repeats {
        summaries.push(materialize_planned_objects(
            planned, source, out_dir, config,
        )?);
    }
    if summaries.len() == 1 {
        return summaries
            .pop()
            .ok_or_else(|| Error::invalid("cellprofiler-human: missing materialization run"));
    }

    let seconds = summaries
        .iter()
        .filter_map(|summary| summary.get("seconds").and_then(|value| value.as_f64()))
        .collect::<Vec<_>>();
    let seconds_min = seconds.iter().copied().min_by(f64::total_cmp);
    let seconds_max = seconds.iter().copied().max_by(f64::total_cmp);
    let seconds_mean =
        (!seconds.is_empty()).then(|| seconds.iter().copied().sum::<f64>() / seconds.len() as f64);
    let seconds_median = median_seconds(seconds.clone());
    let last = summaries
        .last()
        .cloned()
        .ok_or_else(|| Error::invalid("cellprofiler-human: missing materialization run"))?;
    let mut out = match last {
        serde_json::Value::Object(map) => map,
        _ => {
            return Err(Error::invalid(
                "cellprofiler-human: materialization summary was not an object",
            ));
        }
    };
    out.insert("seconds".to_string(), json!(seconds_min));
    out.insert("seconds_min".to_string(), json!(seconds_min));
    out.insert("seconds_max".to_string(), json!(seconds_max));
    out.insert("seconds_mean".to_string(), json!(seconds_mean));
    out.insert("seconds_median".to_string(), json!(seconds_median));
    out.insert("repeats".to_string(), json!(summaries.len()));
    out.insert("repeat_seconds".to_string(), json!(seconds));
    Ok(serde_json::Value::Object(out))
}

fn median_seconds(mut seconds: Vec<f64>) -> Option<f64> {
    if seconds.is_empty() {
        return None;
    }
    seconds.sort_by(f64::total_cmp);
    let mid = seconds.len() / 2;
    if seconds.len() % 2 == 0 {
        Some((seconds[mid - 1] + seconds[mid]) / 2.0)
    } else {
        Some(seconds[mid])
    }
}

fn planned_label_stats(env: &ZarrEnvironment, image: usize) -> Result<serde_json::Value> {
    let labels = env.image(image)?;
    let labels = labels.view::<u32>()?;
    let mut nonzero_voxels = 0u64;
    let mut max_label = 0u32;
    let mut labels_seen = HashSet::<u32>::new();
    for &label in &labels {
        if label == 0 {
            continue;
        }
        nonzero_voxels += 1;
        max_label = max_label.max(label);
        labels_seen.insert(label);
    }
    Ok(json!({
        "image": image,
        "nonzero_voxels": nonzero_voxels,
        "max_label": max_label,
        "distinct_labels": labels_seen.len()
    }))
}

fn save_planned_labels(env: &ZarrEnvironment, image: usize, path: &Path) -> Result<()> {
    let labels = env.image(image)?;
    let labels = labels.view::<u32>()?;
    if labels.shape()[0] != 1 {
        return Err(Error::invalid(
            "cellprofiler-human: planned label PNG export currently expects a single Z plane",
        ));
    }
    let mut image =
        ImageBuffer::<Luma<u16>, Vec<u16>>::new(labels.shape()[2] as u32, labels.shape()[1] as u32);
    for y in 0..labels.shape()[1] {
        for x in 0..labels.shape()[2] {
            let label = labels[[0, y, x]].min(u32::from(u16::MAX)) as u16;
            image.put_pixel(x as u32, y as u32, Luma([label]));
        }
    }
    image
        .save(path)
        .map_err(|err| Error::invalid(format!("cellprofiler-human: save labels: {err}")))
}

struct PlannedObjectRow {
    label: u64,
    shape: ShapeMeasurements,
    intensity: IntensityMeasurements,
}

fn collect_planned_object_rows(
    env: &ZarrEnvironment,
    planned: &PlannedProbe,
) -> Result<Vec<PlannedObjectRow>> {
    let volume = planned.measurements.decomposition.volume;
    let shape_rows = planned.measurements.class_a_rows().ok_or_else(|| {
        Error::invalid("cellprofiler-human: planned measurements have no shape rows")
    })?;
    let intensity_rows = planned
        .measurements
        .class_a_intensity_rows(0)
        .ok_or_else(|| {
            Error::invalid("cellprofiler-human: planned measurements have no intensity rows")
        })?;
    let shapes = collect_class_a_shapes(env, &shape_rows, volume, planned.measurements.fixed)?;
    let intensities =
        collect_class_a_values(env, &intensity_rows, volume, planned.measurements.fixed)?;
    let mut intensity_by_label = BTreeMap::<u64, IntensityMeasurements>::new();
    for values in intensities {
        let mut measurements = IntensityMeasurements::from_values(&values);
        measurements.sum /= 65535.0;
        measurements.mean = measurements.mean.map(|value| value / 65535.0);
        measurements.min = measurements.min.map(|value| value / 65535.0);
        measurements.max = measurements.max.map(|value| value / 65535.0);
        intensity_by_label.insert(measurements.label, measurements);
    }
    let mut rows = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let shape = ShapeMeasurements::from_shape(&shape);
        let intensity = intensity_by_label.remove(&shape.label).ok_or_else(|| {
            Error::invalid(format!(
                "cellprofiler-human: no intensity row for label {}",
                shape.label
            ))
        })?;
        rows.push(PlannedObjectRow {
            label: shape.label,
            shape,
            intensity,
        });
    }
    Ok(rows)
}

fn write_planned_object_csv(rows: &[PlannedObjectRow], path: &Path) -> Result<()> {
    let file = File::create(path)
        .map_err(|err| Error::invalid(format!("cellprofiler-human: create CSV: {err}")))?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "label,count,centroid_z,centroid_y,centroid_x,bbox_min_z,bbox_min_y,bbox_min_x,\
         bbox_max_z,bbox_max_y,bbox_max_x,equivalent_radius,equivalent_diameter,\
         principal_axis_0,principal_axis_1,principal_axis_2,eccentricity,\
         intensity_count,finite_intensity_count,intensity_sum,intensity_mean,intensity_min,intensity_max"
    )
    .map_err(write_error)?;
    for row in rows {
        let centroid = row.shape.centroid.unwrap_or([f64::NAN; 3]);
        let axes = row.shape.principal_axis_lengths.unwrap_or([f64::NAN; 3]);
        writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            row.label,
            row.shape.count,
            centroid[0],
            centroid[1],
            centroid[2],
            row.shape.bbox_min[0],
            row.shape.bbox_min[1],
            row.shape.bbox_min[2],
            row.shape.bbox_max[0],
            row.shape.bbox_max[1],
            row.shape.bbox_max[2],
            row.shape.equivalent_sphere_radius,
            row.shape.equivalent_sphere_diameter,
            axes[0],
            axes[1],
            axes[2],
            optional(row.shape.eccentricity),
            row.intensity.count,
            row.intensity.finite_count,
            row.intensity.sum,
            optional(row.intensity.mean),
            optional(row.intensity.min),
            optional(row.intensity.max)
        )
        .map_err(write_error)?;
    }
    out.flush().map_err(write_error)
}

fn build_planned_skeleton(
    volume: [usize; 3],
    config: &Config,
) -> Result<blockflow::assemble::Assembly> {
    let grid = BlockGrid::new(volume, config.chunk)?;
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    if let Some(percentile) = config.background_percentile {
        append_percentile_background_subtract_phase(
            &mut builder,
            "cellprofiler-background-subtract",
            percentile,
        )?;
    }
    let corrected_image = ImageId::from(builder.n_phases());
    if config.sigma != 0.0 {
        let input = corrected_image;
        example_planning::pixels(
            &mut builder,
            Chain::sequence(vec![
                Chain::source(input, Dtype::F64),
                Chain::op(SmoothOp::new(
                    "cellprofiler-threshold-smoothing",
                    Gaussian::new([0.0, config.sigma, config.sigma], 3.0)?
                        .with_boundary(Boundary::Reflect),
                )),
            ]),
        )?;
    }
    append_global_threshold_phases(
        &mut builder,
        "cellprofiler-threshold",
        GlobalThresholdSelection::single(
            config
                .threshold_method
                .global_threshold(config.threshold_bins)?,
        ),
        GlobalThresholdOutput::Mask {
            test: ThresholdTest::Above,
        },
    )?;
    append_remove_small_objects_phases(
        &mut builder,
        "cellprofiler-remove-small",
        Lifecycle::DeleteOnExit,
        Connectivity::Faces,
        config.min_size,
    )?;
    let mask_image = ImageId::from(builder.n_phases());
    let declump_image = match config.declump_method {
        DeclumpMethod::Intensity => {
            example_planning::pixels(
                &mut builder,
                Chain::sequence(vec![
                    Chain::source(corrected_image, Dtype::F64),
                    Chain::op(SmoothOp::new(
                        "cellprofiler-intensity-declump-smoothing",
                        Gaussian::new([0.0, config.declump_sigma, config.declump_sigma], 3.0)?
                            .with_boundary(Boundary::Reflect),
                    )),
                ]),
            )?;
            ImageId::from(builder.n_phases())
        }
        DeclumpMethod::Distance => {
            distance::append_to(
                &mut builder,
                &DistanceParams::default(),
                config.distance_block,
            )?;
            ImageId::from(builder.n_phases())
        }
    };
    if config.maxima_downsample == 1 {
        regional::append_to(
            &mut builder,
            "cellprofiler-regional-maxima",
            Lifecycle::DeleteOnExit,
            Dtype::Bool,
        )?;
        let maxima_image = ImageId::from(builder.n_phases());
        append_min_distance_seed_suppression_phases(
            &mut builder,
            "cellprofiler-seed-min-distance",
            maxima_image,
            declump_image,
            mask_image,
            config.seed_min_distance,
        )?;
    } else {
        append_downsampled_seed_suppression_phases(
            &mut builder,
            "cellprofiler-downsampled-seed-maxima",
            volume,
            declump_image,
            mask_image,
            config.maxima_downsample,
            config.seed_min_distance,
        )?;
    }
    let seed_grid = builder.grid().clone();
    let label = builder.fragments(
        LabelComponentsOp::new(
            "cellprofiler-maxima-seed-labelling",
            "cellprofiler-maxima-seeds",
            Lifecycle::DeleteOnExit,
        )
        .connecting(Connectivity::Faces),
    )?;
    builder.fragments(
        RelabelComponentsOp::reading(
            "cellprofiler-maxima-seed-relabel",
            "cellprofiler-maxima-seeds",
            label,
            &seed_grid,
        )
        .connecting(Connectivity::Faces),
    )?;
    let seeds_image = ImageId::from(builder.n_phases());
    example_planning::pixels(
        &mut builder,
        Chain::sequence(vec![
            Chain::source(declump_image, Dtype::F64),
            Chain::op(VoxelwiseMapOp::new(
                "cellprofiler-negate-declump-source",
                |value| -value,
            )),
        ]),
    )?;
    example_planning::pixels(
        &mut builder,
        Chain::op(
            SeededWatershedOp::new(
                "cellprofiler-seeded-watershed",
                seeds_image,
                Separation::Line,
            )
            .within(mask_image),
        ),
    )?;
    append_fill_label_holes_2d_by_label_phase(&mut builder)?;
    if config.merge_line_basin_pixels != 0 {
        let labels = ImageId::from(builder.n_phases());
        builder.fragments(WatershedLineMergeOp::new(
            "cellprofiler planned watershed-line basin merge",
            labels,
            declump_image,
            config.merge_line_basin_pixels,
            config.merge_line_max_saddle_drop,
        ))?;
    }
    append_filter_labels_touching_border_on_axes_phase(&mut builder, [false, true, true])?;
    append_filter_labels_by_size_phases(
        &mut builder,
        "cellprofiler-final-size-filter",
        Lifecycle::DeleteOnExit,
        config.min_size,
        config.max_size,
    )?;
    builder.finish()
}

struct WatershedLineMergeOp {
    name: &'static str,
    labels: ImageId,
    intensity: ImageId,
    min_line_pixels: usize,
    max_saddle_drop: Option<f64>,
}

impl WatershedLineMergeOp {
    fn new(
        name: &'static str,
        labels: ImageId,
        intensity: ImageId,
        min_line_pixels: usize,
        max_saddle_drop: Option<f64>,
    ) -> Self {
        Self {
            name,
            labels,
            intensity,
            min_line_pixels,
            max_saddle_drop,
        }
    }
}

impl FragmentOp for WatershedLineMergeOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![
            SourceInput::new(self.labels, Reach::all()).holding(Dtype::U32),
            SourceInput::new(self.intensity, Reach::all()).holding(Dtype::F64),
        ]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let labels = sources
            .get(self.labels.index())?
            .as_array()?
            .view::<u32>()?;
        let intensity = sources
            .get(self.intensity.index())?
            .as_array()?
            .view::<f64>()?;
        let relabelled = merge_labels_across_watershed_lines_for_probe(
            labels,
            intensity,
            self.min_line_pixels,
            self.max_saddle_drop,
        )?;
        let mut out = at.output_buffer(0.0)?;
        let BlockBuf::Array(array) = &mut out else {
            return Ok(BlockOutput::nothing().with_pixels(out));
        };
        let mut view = array.view_mut::<u32>()?;
        for ((z, y, x), slot) in view.indexed_iter_mut() {
            *slot = relabelled[[
                at.at.offset[0] + z,
                at.at.offset[1] + y,
                at.at.offset[2] + x,
            ]];
        }
        Ok(BlockOutput::nothing().with_pixels(out))
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "cellprofiler-human: watershed-line merge op reads declared sources and uses apply_with",
        ))
    }
}

fn merge_labels_across_watershed_lines_for_probe(
    labels: ndarray::ArrayView3<'_, u32>,
    intensity: ndarray::ArrayView3<'_, f64>,
    min_line_pixels: usize,
    max_saddle_drop: Option<f64>,
) -> Result<Array3<u32>> {
    if labels.shape() != intensity.shape() {
        return Err(Error::invalid(
            "cellprofiler-human: line-merge labels and intensity shape mismatch",
        ));
    }
    let mut pair_evidence = BTreeMap::<(u32, u32), LineMergeEvidence>::new();
    for ((z, y, x), &label) in labels.indexed_iter() {
        if label != 0 {
            continue;
        }
        let mut touching = BTreeSet::<u32>::new();
        let mut neighbours = Vec::<(u32, f64)>::new();
        for [dz, dy, dx] in [
            [-1isize, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ] {
            let Some(nz) = z.checked_add_signed(dz) else {
                continue;
            };
            let Some(ny) = y.checked_add_signed(dy) else {
                continue;
            };
            let Some(nx) = x.checked_add_signed(dx) else {
                continue;
            };
            if nz >= labels.shape()[0] || ny >= labels.shape()[1] || nx >= labels.shape()[2] {
                continue;
            }
            let neighbour = labels[[nz, ny, nx]];
            if neighbour != 0 {
                touching.insert(neighbour);
                neighbours.push((neighbour, intensity[[nz, ny, nx]]));
            }
        }
        if touching.len() == 2 {
            let mut touching_labels = touching.into_iter();
            let a = touching_labels.next().expect("two touching labels");
            let b = touching_labels.next().expect("two touching labels");
            let pair = if a < b { (a, b) } else { (b, a) };
            pair_evidence
                .entry(pair)
                .or_default()
                .add(intensity[[z, y, x]], neighbours);
        }
    }
    let mut parent = BTreeMap::<u32, u32>::new();
    for ((a, b), evidence) in pair_evidence {
        if evidence.line_pixels >= min_line_pixels
            && evidence
                .saddle_drop()
                .is_none_or(|drop| max_saddle_drop.is_none_or(|limit| drop <= limit))
        {
            union_probe_label(&mut parent, a, b);
        }
    }
    let mut out = Array3::<u32>::zeros(labels.raw_dim());
    for ((z, y, x), slot) in out.indexed_iter_mut() {
        let label = labels[[z, y, x]];
        if label != 0 {
            *slot = find_probe_label(&mut parent, label);
        }
    }
    Ok(out)
}

#[derive(Default)]
struct LineMergeEvidence {
    line_pixels: usize,
    line_sum: f64,
    boundary: BTreeMap<u32, (f64, usize)>,
}

impl LineMergeEvidence {
    fn add(&mut self, line_value: f64, neighbours: Vec<(u32, f64)>) {
        self.line_pixels += 1;
        self.line_sum += line_value;
        for (label, value) in neighbours {
            let entry = self.boundary.entry(label).or_default();
            entry.0 += value;
            entry.1 += 1;
        }
    }

    fn saddle_drop(&self) -> Option<f64> {
        if self.line_pixels == 0 || self.boundary.len() != 2 {
            return None;
        }
        let line_mean = self.line_sum / self.line_pixels as f64;
        let weakest_boundary_mean = self
            .boundary
            .values()
            .filter_map(|(sum, count)| (*count > 0).then_some(*sum / *count as f64))
            .min_by(f64::total_cmp)?;
        Some(weakest_boundary_mean - line_mean)
    }
}

fn find_probe_label(parent: &mut BTreeMap<u32, u32>, label: u32) -> u32 {
    let current = parent.get(&label).copied().unwrap_or(label);
    if current == label {
        parent.entry(label).or_insert(label);
        return label;
    }
    let root = find_probe_label(parent, current);
    parent.insert(label, root);
    root
}

fn union_probe_label(parent: &mut BTreeMap<u32, u32>, a: u32, b: u32) {
    let root_a = find_probe_label(parent, a);
    let root_b = find_probe_label(parent, b);
    if root_a == root_b {
        return;
    }
    let root = root_a.min(root_b);
    let other = root_a.max(root_b);
    parent.insert(other, root);
}

const BACKGROUND_SAMPLES_MAGIC: u64 = 0x4247_5341_4d50_0001;
const BACKGROUND_LEVEL_MAGIC: u64 = 0x4247_4c45_564c_0001;

struct BackgroundPercentileSamplesOp {
    name: &'static str,
    stream: String,
}

impl BackgroundPercentileSamplesOp {
    fn new(name: &'static str, stream: impl Into<String>) -> Self {
        Self {
            name,
            stream: stream.into(),
        }
    }
}

impl FragmentOp for BackgroundPercentileSamplesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.stream.clone(),
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let pixels = at.pixels()?.as_array()?;
        let values = pixels
            .view::<f64>()?
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            encode_background_samples(&values),
        ))
    }
}

struct ApplyBackgroundPercentileSubtractOp {
    name: &'static str,
    stream: String,
    samples_phase: usize,
    image: usize,
    percentile: f64,
}

impl ApplyBackgroundPercentileSubtractOp {
    fn new(
        name: &'static str,
        stream: impl Into<String>,
        samples_phase: impl Into<Phase>,
        image: impl Into<ImageId>,
        percentile: f64,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            samples_phase: samples_phase.into().index(),
            image: image.into().index(),
            percentile,
        }
    }
}

impl FragmentOp for ApplyBackgroundPercentileSubtractOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.samples_phase).with_reach([0, 0, 0])]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.image).holding(Dtype::F64)]
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut values = Vec::new();
        for (_key, bytes) in at.fragments(&self.stream)? {
            values.extend(decode_background_samples(&bytes)?);
        }
        let level = percentile_value(values, self.percentile)?;
        Ok(encode_background_level(level))
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "background percentile subtraction reads a declared image source and uses apply_with",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let level = decode_background_level(at.reduced)?;
        let pixels = sources.get(self.image)?.as_array()?.view::<f64>()?;
        let mut out = at.output_buffer(0.0)?;
        if let Some(out) = out.as_array_mut() {
            let mut out = out.view_mut::<f64>()?;
            for (slot, &value) in out.iter_mut().zip(pixels.iter()) {
                *slot = (value - level).max(0.0);
            }
        }
        Ok(BlockOutput::nothing().with_pixels(out))
    }
}

fn append_percentile_background_subtract_phase(
    builder: &mut PlanBuilder,
    stream: impl Into<String>,
    percentile: f64,
) -> Result<Phase> {
    let stream = stream.into();
    let image = ImageId::from(builder.n_phases());
    let samples = builder.fragments(BackgroundPercentileSamplesOp::new(
        "cellprofiler background percentile samples",
        stream.clone(),
    ))?;
    builder.fragments(ApplyBackgroundPercentileSubtractOp::new(
        "cellprofiler background percentile subtract",
        stream,
        samples,
        image,
        percentile,
    ))
}

fn percentile_value(values: impl IntoIterator<Item = f64>, percentile: f64) -> Result<f64> {
    let mut values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() {
        return Err(Error::invalid(
            "cellprofiler-human: background percentile needs at least one finite value",
        ));
    }
    values.sort_by(f64::total_cmp);
    let rank = ((percentile / 100.0).clamp(0.0, 1.0) * values.len().saturating_sub(1) as f64)
        .round() as usize;
    Ok(values[rank])
}

fn encode_background_samples(values: &[f64]) -> Vec<u8> {
    let mut words = Vec::with_capacity(2 + values.len());
    words.push(BACKGROUND_SAMPLES_MAGIC);
    words.push(values.len() as u64);
    words.extend(values.iter().map(|value| value.to_bits()));
    encode_words(&words)
}

fn decode_background_samples(bytes: &[u8]) -> Result<Vec<f64>> {
    let words = decode_words(bytes, "background sample fragment")?;
    if words.len() < 2 || words[0] != BACKGROUND_SAMPLES_MAGIC {
        return Err(Error::invalid(
            "cellprofiler-human: background sample fragment has the wrong magic",
        ));
    }
    let rows = usize::try_from(words[1]).map_err(|_| {
        Error::invalid("cellprofiler-human: background sample count does not fit usize")
    })?;
    if words.len() != 2 + rows {
        return Err(Error::invalid(format!(
            "cellprofiler-human: background sample fragment declares {rows} values but has {} words",
            words.len()
        )));
    }
    Ok(words[2..]
        .iter()
        .map(|word| f64::from_bits(*word))
        .collect())
}

fn encode_background_level(level: f64) -> Vec<u8> {
    encode_words(&[BACKGROUND_LEVEL_MAGIC, level.to_bits()])
}

fn decode_background_level(bytes: &[u8]) -> Result<f64> {
    let words = decode_words(bytes, "background level reduction")?;
    if words.len() != 2 || words[0] != BACKGROUND_LEVEL_MAGIC {
        return Err(Error::invalid(
            "cellprofiler-human: background level reduction has the wrong shape or magic",
        ));
    }
    Ok(f64::from_bits(words[1]))
}

const SEED_CANDIDATE_MAGIC: u64 = 0x5345_4544_4341_4e01;
const SEED_ACCEPTED_MAGIC: u64 = 0x5345_4544_4b45_4550;

#[derive(Clone, Copy)]
struct SeedCandidate {
    at: [usize; 3],
    score: f64,
}

struct SeedCandidateOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    distance: ImageId,
    mask: ImageId,
}

impl SeedCandidateOp {
    fn new(
        name: &'static str,
        stream: impl Into<String>,
        lifecycle: Lifecycle,
        distance: impl Into<ImageId>,
        mask: impl Into<ImageId>,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
            distance: distance.into(),
            mask: mask.into(),
        }
    }
}

impl FragmentOp for SeedCandidateOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            Coverage::EveryBlock,
        )]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![
            SourceInput::voxelwise(self.distance).holding(Dtype::F64),
            SourceInput::voxelwise(self.mask).holding(Dtype::Bool),
        ]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(maxima_pixels) = at.pixels()? else {
            return Ok(BlockOutput::fragment(
                self.stream.clone(),
                encode_seed_candidates(&[]),
            ));
        };
        let BlockBuf::Array(distance_pixels) = sources.get(self.distance.index())? else {
            return Ok(BlockOutput::fragment(
                self.stream.clone(),
                encode_seed_candidates(&[]),
            ));
        };
        let BlockBuf::Array(mask_pixels) = sources.get(self.mask.index())? else {
            return Ok(BlockOutput::fragment(
                self.stream.clone(),
                encode_seed_candidates(&[]),
            ));
        };
        let maxima = maxima_pixels.view::<bool>()?;
        let distance = distance_pixels.view::<f64>()?;
        let mask = mask_pixels.view::<bool>()?;
        if maxima.shape() != distance.shape() || maxima.shape() != mask.shape() {
            return Err(Error::invalid(
                "cellprofiler-human: seed candidate source shape mismatch",
            ));
        }
        let mut candidates = Vec::new();
        for ((z, y, x), &is_maximum) in maxima.indexed_iter() {
            if is_maximum && mask[[z, y, x]] {
                candidates.push(SeedCandidate {
                    at: [
                        at.at.offset[0] + z,
                        at.at.offset[1] + y,
                        at.at.offset[2] + x,
                    ],
                    score: distance[[z, y, x]],
                });
            }
        }
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            encode_seed_candidates(&candidates),
        ))
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "cellprofiler-human: seed candidate op reads declared sources and uses apply_with",
        ))
    }
}

struct ApplySeedMinDistanceOp {
    name: &'static str,
    stream: String,
    phase: usize,
    min_distance: f64,
}

impl ApplySeedMinDistanceOp {
    fn new(
        name: &'static str,
        stream: impl Into<String>,
        phase: impl Into<blockflow::assemble::Phase>,
        min_distance: f64,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            phase: phase.into().index(),
            min_distance,
        }
    }
}

impl FragmentOp for ApplySeedMinDistanceOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn barrier(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.phase).with_reach([0, 0, 0])]
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut candidates = Vec::new();
        for (_key, bytes) in at.fragments(&self.stream)? {
            candidates.extend(decode_seed_candidates(&bytes)?);
        }
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.at.cmp(&b.at)));
        let accepted = accepted_seed_points(candidates, self.min_distance);
        Ok(encode_accepted_seeds(&accepted))
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let accepted = decode_accepted_seeds(at.reduced)?;
        let mut out = at.output_buffer(0.0)?;
        let BlockBuf::Array(array) = &mut out else {
            return Ok(BlockOutput::nothing().with_pixels(out));
        };
        let mut view = array.view_mut::<bool>()?;
        for ((z, y, x), slot) in view.indexed_iter_mut() {
            let global = [
                at.at.offset[0] + z,
                at.at.offset[1] + y,
                at.at.offset[2] + x,
            ];
            *slot = accepted.contains(&global);
        }
        Ok(BlockOutput::nothing().with_pixels(out))
    }
}

fn append_min_distance_seed_suppression_phases(
    builder: &mut PlanBuilder,
    stream: impl Into<String>,
    maxima: ImageId,
    distance: ImageId,
    mask: ImageId,
    min_distance: f64,
) -> Result<()> {
    let stream = stream.into();
    let candidates = builder.fragments(SeedCandidateOp::new(
        "cellprofiler seed candidates",
        stream.clone(),
        Lifecycle::DeleteOnExit,
        distance,
        mask,
    ))?;
    builder.fragments(ApplySeedMinDistanceOp::new(
        "cellprofiler seed min-distance",
        stream,
        candidates,
        min_distance,
    ))?;
    let _ = maxima;
    Ok(())
}

const DOWNSAMPLED_SEED_BIN_MAGIC: u64 = 0x4453_4545_4442_0001;

#[derive(Clone, Copy, Debug)]
struct DownsampledSeedBin {
    low: [usize; 3],
    at: [usize; 3],
    score: f64,
    in_mask: bool,
}

struct DownsampledSeedBinOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    values: ImageId,
    mask: ImageId,
    downsample: usize,
}

impl DownsampledSeedBinOp {
    fn new(
        name: &'static str,
        stream: impl Into<String>,
        lifecycle: Lifecycle,
        values: ImageId,
        mask: ImageId,
        downsample: usize,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
            values,
            mask,
            downsample,
        }
    }
}

impl FragmentOp for DownsampledSeedBinOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            Coverage::EveryBlock,
        )]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![
            SourceInput::voxelwise(self.values).holding(Dtype::F64),
            SourceInput::voxelwise(self.mask).holding(Dtype::Bool),
        ]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let values = sources
            .get(self.values.index())?
            .as_array()?
            .view::<f64>()?;
        let mask = sources.get(self.mask.index())?.as_array()?.view::<bool>()?;
        if values.shape() != mask.shape() {
            return Err(Error::invalid(
                "cellprofiler-human: downsampled seed source shape mismatch",
            ));
        }
        let mut bins = BTreeMap::<[usize; 3], DownsampledSeedBin>::new();
        for ((z, y, x), &score) in values.indexed_iter() {
            if !score.is_finite() {
                continue;
            }
            let global = [
                at.at.offset[0] + z,
                at.at.offset[1] + y,
                at.at.offset[2] + x,
            ];
            let low = [
                global[0],
                global[1] / self.downsample,
                global[2] / self.downsample,
            ];
            let candidate = DownsampledSeedBin {
                low,
                at: global,
                score,
                in_mask: mask[[z, y, x]],
            };
            match bins.get(&low).copied() {
                Some(best) if score < best.score || (score == best.score && global >= best.at) => {}
                _ => {
                    bins.insert(low, candidate);
                }
            }
        }
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            encode_downsampled_seed_bins(bins.values().copied()),
        ))
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "cellprofiler-human: downsampled seed bin op reads declared sources and uses apply_with",
        ))
    }
}

struct ApplyDownsampledSeedsOp {
    name: &'static str,
    stream: String,
    phase: usize,
    volume: [usize; 3],
    downsample: usize,
    min_distance: f64,
}

impl ApplyDownsampledSeedsOp {
    fn new(
        name: &'static str,
        stream: impl Into<String>,
        phase: impl Into<blockflow::assemble::Phase>,
        volume: [usize; 3],
        downsample: usize,
        min_distance: f64,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            phase: phase.into().index(),
            volume,
            downsample,
            min_distance,
        }
    }
}

impl FragmentOp for ApplyDownsampledSeedsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn barrier(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.phase).with_reach([0, 0, 0])]
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut bins = BTreeMap::<[usize; 3], DownsampledSeedBin>::new();
        for (_key, bytes) in at.fragments(&self.stream)? {
            for candidate in decode_downsampled_seed_bins(&bytes)? {
                match bins.get(&candidate.low).copied() {
                    Some(best)
                        if candidate.score < best.score
                            || (candidate.score == best.score && candidate.at >= best.at) => {}
                    _ => {
                        bins.insert(candidate.low, candidate);
                    }
                }
            }
        }
        let low_shape = [
            self.volume[0],
            self.volume[1].div_ceil(self.downsample),
            self.volume[2].div_ceil(self.downsample),
        ];
        let mut low = Array3::<f64>::from_elem(low_shape, f64::NEG_INFINITY);
        for (low_at, bin) in &bins {
            low[[low_at[0], low_at[1], low_at[2]]] = bin.score;
        }
        let maxima = regional_maxima(low.view())?;
        let mut candidates = Vec::new();
        for (low_at, bin) in bins {
            if bin.in_mask && maxima[[low_at[0], low_at[1], low_at[2]]] {
                candidates.push(SeedCandidate {
                    at: bin.at,
                    score: bin.score,
                });
            }
        }
        let accepted = accepted_seed_points(candidates, self.min_distance);
        Ok(encode_accepted_seeds(&accepted))
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let accepted = decode_accepted_seeds(at.reduced)?;
        let mut out = at.output_buffer(0.0)?;
        let BlockBuf::Array(array) = &mut out else {
            return Ok(BlockOutput::nothing().with_pixels(out));
        };
        let mut view = array.view_mut::<bool>()?;
        for ((z, y, x), slot) in view.indexed_iter_mut() {
            let global = [
                at.at.offset[0] + z,
                at.at.offset[1] + y,
                at.at.offset[2] + x,
            ];
            *slot = accepted.contains(&global);
        }
        Ok(BlockOutput::nothing().with_pixels(out))
    }
}

fn append_downsampled_seed_suppression_phases(
    builder: &mut PlanBuilder,
    stream: impl Into<String>,
    volume: [usize; 3],
    values: ImageId,
    mask: ImageId,
    downsample: usize,
    min_distance: f64,
) -> Result<()> {
    let stream = stream.into();
    let bins = builder.fragments(DownsampledSeedBinOp::new(
        "cellprofiler downsampled seed bins",
        stream.clone(),
        Lifecycle::DeleteOnExit,
        values,
        mask,
        downsample,
    ))?;
    builder.fragments(ApplyDownsampledSeedsOp::new(
        "cellprofiler downsampled seed min-distance",
        stream,
        bins,
        volume,
        downsample,
        min_distance,
    ))?;
    Ok(())
}

fn squared_distance(a: [usize; 3], b: [usize; 3]) -> f64 {
    let dz = a[0] as f64 - b[0] as f64;
    let dy = a[1] as f64 - b[1] as f64;
    let dx = a[2] as f64 - b[2] as f64;
    dz * dz + dy * dy + dx * dx
}

fn accepted_seed_points(mut candidates: Vec<SeedCandidate>, min_distance: f64) -> Vec<[usize; 3]> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.at.cmp(&b.at)));
    let min_distance2 = min_distance * min_distance;
    let mut accepted = Vec::<[usize; 3]>::new();
    for candidate in candidates {
        if accepted
            .iter()
            .all(|&seed| squared_distance(candidate.at, seed) >= min_distance2)
        {
            accepted.push(candidate.at);
        }
    }
    accepted
}

fn encode_downsampled_seed_bins(bins: impl IntoIterator<Item = DownsampledSeedBin>) -> Vec<u8> {
    let bins = bins.into_iter().collect::<Vec<_>>();
    let mut words = Vec::with_capacity(2 + bins.len() * 8);
    words.push(DOWNSAMPLED_SEED_BIN_MAGIC);
    words.push(bins.len() as u64);
    for bin in bins {
        words.extend(bin.low.iter().map(|&coordinate| coordinate as u64));
        words.extend(bin.at.iter().map(|&coordinate| coordinate as u64));
        words.push(bin.score.to_bits());
        words.push(u64::from(bin.in_mask));
    }
    encode_words(&words)
}

fn decode_downsampled_seed_bins(bytes: &[u8]) -> Result<Vec<DownsampledSeedBin>> {
    let words = decode_words(bytes, "downsampled seed bin fragment")?;
    if words.len() < 2 || words[0] != DOWNSAMPLED_SEED_BIN_MAGIC {
        return Err(Error::invalid(
            "cellprofiler-human: downsampled seed bin fragment has the wrong magic",
        ));
    }
    let rows = usize::try_from(words[1]).map_err(|_| {
        Error::invalid("cellprofiler-human: downsampled seed bin count does not fit usize")
    })?;
    if words.len() != 2 + rows * 8 {
        return Err(Error::invalid(format!(
            "cellprofiler-human: downsampled seed bin fragment declares {rows} rows but has {} words",
            words.len()
        )));
    }
    let mut out = Vec::with_capacity(rows);
    for row in words[2..].chunks_exact(8) {
        let in_mask = match row[7] {
            0 => false,
            1 => true,
            _ => {
                return Err(Error::invalid(
                    "cellprofiler-human: downsampled seed bin mask flag is invalid",
                ));
            }
        };
        out.push(DownsampledSeedBin {
            low: [
                usize::try_from(row[0]).map_err(|_| {
                    Error::invalid("cellprofiler-human: downsampled seed low z does not fit usize")
                })?,
                usize::try_from(row[1]).map_err(|_| {
                    Error::invalid("cellprofiler-human: downsampled seed low y does not fit usize")
                })?,
                usize::try_from(row[2]).map_err(|_| {
                    Error::invalid("cellprofiler-human: downsampled seed low x does not fit usize")
                })?,
            ],
            at: [
                usize::try_from(row[3]).map_err(|_| {
                    Error::invalid("cellprofiler-human: downsampled seed z does not fit usize")
                })?,
                usize::try_from(row[4]).map_err(|_| {
                    Error::invalid("cellprofiler-human: downsampled seed y does not fit usize")
                })?,
                usize::try_from(row[5]).map_err(|_| {
                    Error::invalid("cellprofiler-human: downsampled seed x does not fit usize")
                })?,
            ],
            score: f64::from_bits(row[6]),
            in_mask,
        });
    }
    Ok(out)
}

fn encode_words(words: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(words));
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn decode_words(bytes: &[u8], noun: &'static str) -> Result<Vec<u64>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<u64>()) {
        return Err(Error::invalid(format!(
            "cellprofiler-human: {noun} byte length {} is not a whole number of words",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u64>())
        .map(|chunk| {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            u64::from_le_bytes(word)
        })
        .collect())
}

fn encode_seed_candidates(candidates: &[SeedCandidate]) -> Vec<u8> {
    let mut words = Vec::with_capacity(2 + candidates.len() * 4);
    words.push(SEED_CANDIDATE_MAGIC);
    words.push(candidates.len() as u64);
    for candidate in candidates {
        words.extend(candidate.at.iter().map(|&coordinate| coordinate as u64));
        words.push(candidate.score.to_bits());
    }
    encode_words(&words)
}

fn decode_seed_candidates(bytes: &[u8]) -> Result<Vec<SeedCandidate>> {
    let words = decode_words(bytes, "seed candidate fragment")?;
    if words.len() < 2 || words[0] != SEED_CANDIDATE_MAGIC {
        return Err(Error::invalid(
            "cellprofiler-human: seed candidate fragment has the wrong magic",
        ));
    }
    let rows = usize::try_from(words[1]).map_err(|_| {
        Error::invalid("cellprofiler-human: seed candidate count does not fit usize")
    })?;
    if words.len() != 2 + rows * 4 {
        return Err(Error::invalid(format!(
            "cellprofiler-human: seed candidate fragment declares {rows} rows but has {} words",
            words.len()
        )));
    }
    let mut out = Vec::with_capacity(rows);
    for row in words[2..].chunks_exact(4) {
        out.push(SeedCandidate {
            at: [
                usize::try_from(row[0]).map_err(|_| {
                    Error::invalid("cellprofiler-human: seed z coordinate does not fit usize")
                })?,
                usize::try_from(row[1]).map_err(|_| {
                    Error::invalid("cellprofiler-human: seed y coordinate does not fit usize")
                })?,
                usize::try_from(row[2]).map_err(|_| {
                    Error::invalid("cellprofiler-human: seed x coordinate does not fit usize")
                })?,
            ],
            score: f64::from_bits(row[3]),
        });
    }
    Ok(out)
}

fn encode_accepted_seeds(seeds: &[[usize; 3]]) -> Vec<u8> {
    let mut words = Vec::with_capacity(2 + seeds.len() * 3);
    words.push(SEED_ACCEPTED_MAGIC);
    words.push(seeds.len() as u64);
    for seed in seeds {
        words.extend(seed.iter().map(|&coordinate| coordinate as u64));
    }
    encode_words(&words)
}

fn decode_accepted_seeds(bytes: &[u8]) -> Result<HashSet<[usize; 3]>> {
    let words = decode_words(bytes, "accepted seed reduction")?;
    if words.len() < 2 || words[0] != SEED_ACCEPTED_MAGIC {
        return Err(Error::invalid(
            "cellprofiler-human: accepted seed reduction has the wrong magic",
        ));
    }
    let rows = usize::try_from(words[1]).map_err(|_| {
        Error::invalid("cellprofiler-human: accepted seed count does not fit usize")
    })?;
    if words.len() != 2 + rows * 3 {
        return Err(Error::invalid(format!(
            "cellprofiler-human: accepted seed reduction declares {rows} rows but has {} words",
            words.len()
        )));
    }
    let mut out = HashSet::with_capacity(rows);
    for row in words[2..].chunks_exact(3) {
        out.insert([
            usize::try_from(row[0]).map_err(|_| {
                Error::invalid("cellprofiler-human: accepted seed z coordinate overflow")
            })?,
            usize::try_from(row[1]).map_err(|_| {
                Error::invalid("cellprofiler-human: accepted seed y coordinate overflow")
            })?,
            usize::try_from(row[2]).map_err(|_| {
                Error::invalid("cellprofiler-human: accepted seed x coordinate overflow")
            })?,
        ]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_suppression_prefers_high_distance_then_coordinate_order() {
        let candidates = vec![
            SeedCandidate {
                at: [0, 0, 0],
                score: 2.0,
            },
            SeedCandidate {
                at: [0, 0, 3],
                score: 10.0,
            },
            SeedCandidate {
                at: [0, 0, 6],
                score: 10.0,
            },
            SeedCandidate {
                at: [0, 0, 8],
                score: 1.0,
            },
        ];
        assert_eq!(
            accepted_seed_points(candidates, 4.0),
            vec![[0, 0, 3], [0, 0, 8]]
        );
    }

    #[test]
    fn seed_candidate_encoding_round_trips_scores_and_coordinates() {
        let candidates = vec![
            SeedCandidate {
                at: [1, 2, 3],
                score: 4.5,
            },
            SeedCandidate {
                at: [5, 8, 13],
                score: -0.0,
            },
        ];
        let decoded = decode_seed_candidates(&encode_seed_candidates(&candidates)).unwrap();
        assert_eq!(decoded.len(), candidates.len());
        for (got, want) in decoded.iter().zip(candidates.iter()) {
            assert_eq!(got.at, want.at);
            assert_eq!(got.score.to_bits(), want.score.to_bits());
        }
    }
}

fn image_volume(path: &Path) -> Result<[usize; 3]> {
    let image = image::ImageReader::open(path)
        .map_err(|err| Error::invalid(format!("cellprofiler-human: open image: {err}")))?
        .with_guessed_format()
        .map_err(|err| Error::invalid(format!("cellprofiler-human: guess format: {err}")))?
        .decode()
        .map_err(|err| Error::invalid(format!("cellprofiler-human: decode image: {err}")))?;
    let width = usize::try_from(image.width())
        .map_err(|_| Error::invalid("cellprofiler-human: image width does not fit usize"))?;
    let height = usize::try_from(image.height())
        .map_err(|_| Error::invalid("cellprofiler-human: image height does not fit usize"))?;
    Ok([1, height, width])
}

fn input_source(config: &Config) -> Result<InputSource> {
    if let Some(array) = &config.input_zarr {
        let (_, volume) = AttachedImage::at(array).metadata()?;
        return Ok(InputSource {
            array: array.clone(),
            volume,
            description: format!("zarr:{}", array.display()),
        });
    }

    if let Some(store) = &config.ensure_input_zarr {
        let input = config.input.as_ref().ok_or_else(|| {
            Error::invalid("cellprofiler-human: --ensure-input-zarr needs --input")
        })?;
        let array = ensure_input_zarr(input, store, config.chunk)?;
        let (_, volume) = AttachedImage::at(&array).metadata()?;
        return Ok(InputSource {
            array,
            volume,
            description: format!("prepared-zarr-from:{}", input.display()),
        });
    }

    let input = config.input.as_ref().ok_or_else(|| {
        Error::invalid("cellprofiler-human: either --input or --input-zarr is required")
    })?;
    let volume = image_volume(input)?;
    let fallback_store = config
        .materialize_objects
        .as_ref()
        .map(|out| out.join("input.zarr"));
    let array = match fallback_store {
        Some(store) => ensure_input_zarr(input, &store, config.chunk)?,
        None => PathBuf::new(),
    };
    Ok(InputSource {
        array,
        volume,
        description: input.display().to_string(),
    })
}

fn ensure_input_zarr(input_path: &Path, store: &Path, chunk: [usize; 3]) -> Result<PathBuf> {
    let array = store.join("level0");
    if array.join("zarr.json").exists() {
        let (_, stored_volume) = AttachedImage::at(&array).metadata()?;
        let image_volume = image_volume(input_path)?;
        if stored_volume != image_volume {
            return Err(Error::invalid(format!(
                "cellprofiler-human: prepared input store {} has volume {:?}, but {} is {:?}; remove the stale store or choose another --ensure-input-zarr path",
                array.display(),
                stored_volume,
                input_path.display(),
                image_volume
            )));
        }
        return Ok(array);
    }

    let input = load_luma_as_volume(input_path)?;
    let voxels = input.into();
    ZarrEnvironment::create(store, &voxels, chunk)?;
    Ok(array)
}

fn chunk_bytes(chunk: [usize; 3], bytes_per_voxel: u64) -> u64 {
    chunk
        .iter()
        .copied()
        .map(|value| value as u64)
        .product::<u64>()
        .saturating_mul(bytes_per_voxel)
}

fn load_luma_as_volume(path: &Path) -> Result<Array3<f64>> {
    let image = image::ImageReader::open(path)
        .map_err(|err| Error::invalid(format!("cellprofiler-human: open image: {err}")))?
        .with_guessed_format()
        .map_err(|err| Error::invalid(format!("cellprofiler-human: guess format: {err}")))?
        .decode()
        .map_err(|err| Error::invalid(format!("cellprofiler-human: decode image: {err}")))?
        .to_luma16();
    let (width, height) = image.dimensions();
    let width = usize::try_from(width)
        .map_err(|_| Error::invalid("cellprofiler-human: image width does not fit usize"))?;
    let height = usize::try_from(height)
        .map_err(|_| Error::invalid("cellprofiler-human: image height does not fit usize"))?;
    let mut out = Array3::<f64>::zeros((1, height, width));
    for (x, y, pixel) in image.enumerate_pixels() {
        out[[0, y as usize, x as usize]] = f64::from(pixel.0[0]);
    }
    Ok(out)
}

fn optional(value: Option<f64>) -> f64 {
    value.unwrap_or(f64::NAN)
}

fn write_error(err: std::io::Error) -> Error {
    Error::invalid(format!("cellprofiler-human: write CSV: {err}"))
}

fn parse_chunk(raw: &str) -> std::result::Result<[usize; 3], String> {
    let parts = raw
        .split(['x', 'X', ',', ':'])
        .map(str::trim)
        .collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(format!(
            "cellprofiler-human: chunk shape {raw:?} must have three dimensions"
        ));
    }
    let mut out = [0usize; 3];
    for (index, part) in parts.iter().enumerate() {
        out[index] = part.parse::<usize>().map_err(|err| {
            format!("cellprofiler-human: could not parse chunk shape {raw:?}: {err}")
        })?;
    }
    Ok(out)
}
