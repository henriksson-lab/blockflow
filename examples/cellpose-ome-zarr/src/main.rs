// SPDX-License-Identifier: MIT

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use blockflow::assemble::PlanBuilder;
use blockflow::decomposition::{Constraints, Decomposition};
use blockflow::fragment::{fragment_phase, FragmentOp, PhaseWork};
use blockflow::geometry::BlockGrid;
use blockflow::label_pyramid::{build_nearest_label_pyramid, refresh_label_registry};
use blockflow::model_segment::{
    cellpose::{CellposeBackend, Device},
    finalize_instances, InstanceSegment,
};
use blockflow::op::Chain;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, predicted_phase_prices, Hints};
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::{AttachedImage, Dtype, Error, Result};
use cellpose::EvalParams;
use clap::{Parser, ValueEnum};
use serde_json::{json, Value as Json};

const STREAM: &str = "cellpose.instances";

#[derive(Debug, Parser)]
#[command(
    name = "cellpose-ome-zarr",
    about = "Segment one OME-Zarr channel with Cellpose and add a newvolim label layer"
)]
struct Config {
    /// OME-Zarr group containing the image pyramid.
    #[arg(long)]
    zarr: PathBuf,
    /// Cellpose checkpoint, for example ~/.cellpose/models/cpsam.safetensors.
    #[arg(long)]
    model: PathBuf,
    /// Inference device. CUDA requires building this example with --features cuda.
    #[arg(long, value_enum, default_value_t = DeviceChoice::Cpu)]
    device: DeviceChoice,
    /// CUDA device ordinal when --device cuda is selected.
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    /// Channel in the [c,y,x] arrays. The 2079 slide's first DAPI is channel 0.
    #[arg(long, default_value_t = 0)]
    channel: usize,
    /// Dataset-local label and measurement-table name.
    #[arg(long, default_value = "cellpose-dapi")]
    layer: String,
    /// Skip a core when all of its DAPI values are at or below this value.
    #[arg(long, default_value_t = 0.0)]
    empty_below: f64,
    /// Halo in pixels. It must cover the largest expected nucleus diameter.
    #[arg(long, default_value_t = 64)]
    halo: usize,
    /// Candidate square block edges offered to the planner.
    #[arg(long, value_delimiter = ',', default_value = "512,1024,2048")]
    blocks: Vec<usize>,
    /// Executor worker count. Cellpose inference itself is serialized per model.
    #[arg(long, default_value_t = 1)]
    workers: usize,
    /// Expected cell diameter in pixels. Omit to use the model's native scale.
    #[arg(long)]
    diameter: Option<f32>,
    /// Cell probability threshold.
    #[arg(long, default_value_t = 0.0)]
    cellprob_threshold: f32,
    /// Flow error threshold.
    #[arg(long, default_value_t = 0.4)]
    flow_threshold: f32,
    /// Smallest retained object in pixels.
    #[arg(long, default_value_t = 15)]
    min_size: i32,
    /// Cellpose inference windows processed in one batch.
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    /// Replace an existing layer and table with the same name.
    #[arg(long, default_value_t = false)]
    overwrite: bool,
    /// Write detailed per-block Cellpose stage timings as JSON.
    ///
    /// Profiling synchronizes CUDA stages and changes performance. Do not use
    /// the profiled wall time as the normal workflow benchmark.
    #[arg(long)]
    profile_json: Option<PathBuf>,
    /// Resume finalization from an existing labels/.<layer>-blockflow-work directory.
    #[arg(long, default_value_t = false, hide = true)]
    resume_work: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DeviceChoice {
    Cpu,
    Cuda,
}

#[derive(Clone)]
struct SourceLevel {
    path: String,
    shape: [usize; 3],
    transform: Json,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = Config::parse();
    let run_started = Instant::now();
    validate(&config)?;
    let levels = source_levels(&config.zarr)?;
    let level0 = levels
        .first()
        .ok_or_else(|| Error::invalid("OME-Zarr image has no multiscale levels"))?;
    if config.channel >= level0.shape[0] {
        return Err(Error::invalid(format!(
            "channel {} is outside level 0 shape {:?}",
            config.channel, level0.shape
        )));
    }

    let layer_root = config.zarr.join("labels").join(&config.layer);
    let table_root = config.zarr.join("tables").join(&config.layer);
    check_destination(&layer_root, config.overwrite)?;
    check_destination(&table_root, config.overwrite)?;
    let work = config
        .zarr
        .join("labels")
        .join(format!(".{}-blockflow-work", config.layer));

    let source = AttachedImage::at(config.zarr.join(&level0.path))
        .plane(config.channel, [level0.shape[1], level0.shape[2]]);
    let (dtype, volume) = source.metadata()?;
    let device = match config.device {
        DeviceChoice::Cpu => Device::Cpu,
        DeviceChoice::Cuda => Device::Cuda(config.cuda_device),
    };
    let params = EvalParams {
        diameter: config.diameter,
        batch_size: config.batch_size,
        cellprob_threshold: config.cellprob_threshold,
        flow_threshold: config.flow_threshold,
        min_size: config.min_size,
        ..EvalParams::default()
    };
    let backend = CellposeBackend::new(&config.model, device, params)?;
    let (backend, profile) = if config.profile_json.is_some() {
        let (backend, profile) = backend.with_profile();
        (backend, Some(profile))
    } else {
        (backend, None)
    };
    let backend = Arc::new(backend);
    let op = InstanceSegment::new(
        "cellpose-dapi",
        backend,
        [0, config.halo, config.halo],
        STREAM,
        Lifecycle::Persistent,
        Vec::new(),
    )
    .skipping_empty(Some(config.empty_below))
    .writing_labels();
    let row_schema = op.schema()?;

    let constraints = constraints(&config, volume);
    let grid = planned_fragment_grid(&op, dtype, volume, &constraints)?;
    println!("planned block {:?} over {:?}", grid.block(), volume);
    let staged_layer = work.join("layer");
    let staged_table = work.join("table");
    if config.resume_work {
        if !staged_layer.join("0").is_dir() {
            return Err(Error::invalid(format!(
                "cannot resume: {} is missing",
                staged_layer.join("0").display()
            )));
        }
    } else {
        if work.exists() {
            fs::remove_dir_all(&work).map_err(Error::backend)?;
        }
        fs::create_dir_all(&work).map_err(Error::backend)?;
    }
    let mut builder = PlanBuilder::new(volume, dtype, grid);
    builder.fragments(op)?;
    let assembly = builder.finish()?;
    let env = ZarrEnvironment::attach(&work, &[source.clone()])?;
    let phase = assembly.n_phases() - 1;
    if !config.resume_work {
        let work_kinds = assembly.work();
        let hints = Hints {
            concurrency: config.workers.max(1),
            cache_bytes: constraints.cache_bytes,
            prefetch_depth: constraints.prefetch_depth,
            prefetch_chunk_bytes: constraints.prefetch_chunk_bytes,
            ..Hints::default()
        };
        execute_phases(
            "cellpose OME-Zarr",
            &assembly.workflow,
            &assembly.decomposition,
            &hints,
            &env,
            &[],
            &work_kinds,
        )?;
        fs::create_dir_all(&staged_layer).map_err(Error::backend)?;
        move_path(
            &work.join(format!("level{}", phase + 1)),
            &staged_layer.join("0"),
        )?;
    }
    drop(env);
    let label_shapes = levels
        .iter()
        .map(|level| [1, level.shape[1], level.shape[2]])
        .collect::<Vec<_>>();
    build_nearest_label_pyramid(
        &staged_layer,
        &work,
        &label_shapes,
        &config.blocks,
        config.workers,
    )?;
    let rows_env = ZarrEnvironment::attach(&work, &[source])?;
    let table = finalize_instances(
        &rows_env,
        STREAM,
        phase,
        volume,
        row_schema,
        &staged_table,
        &work,
        &config.layer,
        pixel_area(&levels[0]),
    )?;
    drop(rows_env);
    write_label_metadata(&staged_layer, &levels)?;
    let cells = table.spec().row_count;
    drop(table);
    replace_with(&staged_layer, &layer_root, config.overwrite)?;
    replace_with(&staged_table, &table_root, config.overwrite)?;
    refresh_label_registry(&config.zarr.join("labels"))?;
    fs::remove_dir_all(&work).map_err(Error::backend)?;

    if let (Some(path), Some(profile)) = (&config.profile_json, profile) {
        write_profile(
            path,
            &config,
            run_started.elapsed().as_secs_f64(),
            profile.samples(),
        )?;
    }

    println!(
        "cells={} label={} table={}",
        cells,
        layer_root.display(),
        table_root.display()
    );
    Ok(())
}

fn write_profile(
    path: &Path,
    config: &Config,
    wall_seconds: f64,
    samples: Vec<cellpose::models::EvalTiming>,
) -> Result<()> {
    let sample_values = serde_json::to_value(&samples).map_err(Error::backend)?;
    let totals = match &sample_values {
        Json::Array(values) => sum_json_objects(values),
        _ => Json::Null,
    };
    let report = json!({
        "release_build": !cfg!(debug_assertions),
        "wall_seconds": wall_seconds,
        "blocks": samples.len(),
        "parameters": {
            "zarr": config.zarr,
            "model": config.model,
            "device": format!("{:?}", config.device),
            "cuda_device": config.cuda_device,
            "channel": config.channel,
            "halo": config.halo,
            "blocks": config.blocks,
            "workers": config.workers,
            "diameter": config.diameter,
            "cellprob_threshold": config.cellprob_threshold,
            "flow_threshold": config.flow_threshold,
            "min_size": config.min_size,
            "batch_size": config.batch_size,
        },
        "totals": totals,
        "samples": sample_values,
    });
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(Error::backend)?;
    }
    let bytes = serde_json::to_vec_pretty(&report).map_err(Error::backend)?;
    fs::write(path, bytes).map_err(Error::backend)
}

fn sum_json_objects(values: &[Json]) -> Json {
    let mut total = Json::Object(serde_json::Map::new());
    for value in values {
        add_json_numbers(&mut total, value);
    }
    total
}

fn add_json_numbers(total: &mut Json, value: &Json) {
    match (total, value) {
        (Json::Object(total), Json::Object(value)) => {
            for (key, value) in value {
                add_json_numbers(total.entry(key.clone()).or_insert(Json::Null), value);
            }
        }
        (slot @ Json::Null, Json::Number(number)) => *slot = Json::Number(number.clone()),
        (Json::Number(total), Json::Number(value)) => {
            if let (Some(total_value), Some(value)) = (total.as_f64(), value.as_f64()) {
                if let Some(sum) = serde_json::Number::from_f64(total_value + value) {
                    *total = sum;
                }
            }
        }
        (slot @ Json::Null, Json::Object(_)) => {
            *slot = Json::Object(serde_json::Map::new());
            add_json_numbers(slot, value);
        }
        _ => {}
    }
}

fn validate(config: &Config) -> Result<()> {
    if config.profile_json.is_some() && cfg!(debug_assertions) {
        return Err(Error::invalid(
            "--profile-json requires a release build; rerun with cargo run --release",
        ));
    }
    if config.halo == 0 || config.blocks.is_empty() || config.blocks.contains(&0) {
        return Err(Error::invalid(
            "--halo and every --blocks entry must be positive",
        ));
    }
    if config.layer.is_empty()
        || config.layer.contains('/')
        || config.layer.contains('\\')
        || config.layer == "."
        || config.layer == ".."
    {
        return Err(Error::invalid("--layer must be one path component"));
    }
    if !config.model.is_file() {
        return Err(Error::invalid(format!(
            "missing Cellpose checkpoint {}",
            config.model.display()
        )));
    }
    if config.batch_size == 0 || config.min_size < 0 {
        return Err(Error::invalid(
            "--batch-size must be positive and --min-size cannot be negative",
        ));
    }
    Ok(())
}

fn constraints(config: &Config, volume: [usize; 3]) -> Constraints {
    Constraints {
        expected_concurrency: config.workers.max(1),
        block_candidates: config.blocks.clone(),
        split_axes: if volume[0] == 1 {
            vec![1, 2]
        } else {
            vec![0, 1, 2]
        },
        ..Constraints::default()
    }
}

fn planned_fragment_grid(
    op: &dyn FragmentOp,
    dtype: Dtype,
    volume: [usize; 3],
    constraints: &Constraints,
) -> Result<BlockGrid> {
    let chain = Chain::sequence(Vec::new());
    let mut best: Option<(f64, usize, BlockGrid)> = None;
    for &edge in &constraints.block_candidates {
        let grid = BlockGrid::along(volume, &constraints.split_axes, edge)?;
        let mut phase = fragment_phase(op, grid.clone())?;
        phase.dtype = (op.produces(dtype) != dtype).then_some(op.produces(dtype));
        let decomposition = Decomposition {
            volume,
            dtype,
            phases: vec![phase],
            chain_reach: [0, 0, 0],
        };
        let prices = predicted_phase_prices(
            &chain,
            &decomposition,
            &[PhaseWork::Fragments(op)],
            &constraints.model,
            constraints.expected_concurrency,
        )?;
        let (cost, makespan) = &prices[0];
        if !constraints.affords_working_set(cost) {
            continue;
        }
        if best.as_ref().is_none_or(|(old, old_edge, _)| {
            (*makespan, std::cmp::Reverse(edge)) < (*old, std::cmp::Reverse(*old_edge))
        }) {
            best = Some((*makespan, edge, grid));
        }
    }
    best.map(|(_, _, grid)| grid)
        .ok_or_else(|| Error::invalid("no Cellpose block candidate fits the supplied constraints"))
}

fn source_levels(root: &Path) -> Result<Vec<SourceLevel>> {
    let metadata: Json =
        serde_json::from_slice(&fs::read(root.join("zarr.json")).map_err(Error::backend)?)
            .map_err(Error::backend)?;
    let scale = metadata
        .pointer("/attributes/ome/multiscales/0")
        .or_else(|| metadata.pointer("/attributes/multiscales/0"))
        .ok_or_else(|| Error::invalid("root metadata has no OME multiscales entry"))?;
    let datasets = scale
        .get("datasets")
        .and_then(Json::as_array)
        .ok_or_else(|| Error::invalid("OME multiscales entry has no datasets"))?;
    datasets
        .iter()
        .map(|dataset| {
            let path = dataset
                .get("path")
                .and_then(Json::as_str)
                .ok_or_else(|| Error::invalid("multiscale dataset has no path"))?
                .to_owned();
            let (_, shape) = AttachedImage::at(root.join(&path)).metadata()?;
            Ok(SourceLevel {
                path,
                shape,
                transform: dataset
                    .get("coordinateTransformations")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
            })
        })
        .collect()
}

fn write_label_metadata(root: &Path, levels: &[SourceLevel]) -> Result<()> {
    let datasets = levels
        .iter()
        .enumerate()
        .map(|(index, level)| {
            json!({"path": index.to_string(), "coordinateTransformations": level.transform})
        })
        .collect::<Vec<_>>();
    let metadata = json!({
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [{
                    "version": "0.5",
                    "axes": [
                        {"name":"z", "type":"space", "unit":"micrometer"},
                        {"name":"y", "type":"space", "unit":"micrometer"},
                        {"name":"x", "type":"space", "unit":"micrometer"}
                    ],
                    "datasets": datasets
                }]
            },
            "image-label": {"version":"0.5", "source":{"image":"../../"}}
        }
    });
    fs::write(
        root.join("zarr.json"),
        serde_json::to_vec_pretty(&metadata).map_err(Error::backend)?,
    )
    .map_err(Error::backend)
}

fn pixel_area(level: &SourceLevel) -> f64 {
    level
        .transform
        .as_array()
        .and_then(|rows| {
            rows.iter()
                .find(|row| row.get("type").and_then(Json::as_str) == Some("scale"))
        })
        .and_then(|row| row.get("scale"))
        .and_then(Json::as_array)
        .and_then(|scale| Some(scale.get(1)?.as_f64()? * scale.get(2)?.as_f64()?))
        .unwrap_or(1.0)
}

fn check_destination(path: &Path, overwrite: bool) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    if !overwrite {
        return Err(Error::invalid(format!(
            "{} already exists; pass --overwrite to replace it",
            path.display()
        )));
    }
    Ok(())
}

fn replace_with(staged: &Path, destination: &Path, overwrite: bool) -> Result<()> {
    if destination.exists() {
        if !overwrite {
            return Err(Error::invalid(format!(
                "{} appeared during the run and will not be replaced",
                destination.display()
            )));
        }
        fs::remove_dir_all(destination).map_err(Error::backend)?;
    }
    move_path(staged, destination)
}

fn move_path(from: &Path, to: &Path) -> Result<()> {
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent).map_err(Error::backend)?;
    }
    fs::rename(from, to).map_err(|error| {
        Error::backend(format!(
            "move {} to {}: {error}",
            from.display(),
            to.display()
        ))
    })
}
