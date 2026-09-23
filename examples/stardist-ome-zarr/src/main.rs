// SPDX-License-Identifier: MIT

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use blockflow::assemble::PlanBuilder;
use blockflow::decomposition::{Constraints, Decomposition};
use blockflow::fragment::{fragment_phase, FragmentOp, PhaseWork};
use blockflow::geometry::BlockGrid;
use blockflow::label_pyramid::{build_nearest_label_pyramid, refresh_label_registry};
use blockflow::model_segment::{finalize_instances, stardist::StardistBackend, InstanceSegment};
use blockflow::op::Chain;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, predicted_phase_prices, Hints};
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::{AttachedImage, Dtype, Error, Result};
use clap::Parser;
use serde_json::{json, Value as Json};

const STREAM: &str = "stardist.instances";

#[derive(Debug, Parser)]
#[command(
    name = "stardist-ome-zarr",
    about = "Segment one OME-Zarr channel with StarDist and add a newvolim label layer"
)]
struct Config {
    /// OME-Zarr group containing the image pyramid.
    #[arg(long)]
    zarr: PathBuf,
    /// StarDist model directory containing config.json and thresholds.json.
    #[arg(long)]
    model: PathBuf,
    /// Keras weights, normally MODEL/weights_best.h5.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Channel in the [c,y,x] arrays. The 2079 slide's first DAPI is channel 0.
    #[arg(long, default_value_t = 0)]
    channel: usize,
    /// Dataset-local label and measurement-table name.
    #[arg(long, default_value = "stardist-dapi")]
    layer: String,
    /// Fixed input value mapped to zero for StarDist.
    #[arg(long, default_value_t = 0.0)]
    low: f32,
    /// Fixed input value mapped to one for StarDist.
    #[arg(long, default_value_t = 255.0)]
    high: f32,
    /// Skip a core when all of its DAPI values are at or below this value.
    #[arg(long, default_value_t = 0.0)]
    empty_below: f64,
    /// Halo in pixels. It must cover the largest expected nucleus diameter.
    #[arg(long, default_value_t = 64)]
    halo: usize,
    /// Candidate square block edges offered to the planner.
    #[arg(long, value_delimiter = ',', default_value = "512,1024,2048")]
    blocks: Vec<usize>,
    /// Executor worker count. StarDist inference itself is serialized per model.
    #[arg(long, default_value_t = 1)]
    workers: usize,
    /// Optional StarDist probability threshold override.
    #[arg(long)]
    prob_threshold: Option<f32>,
    /// Optional StarDist NMS threshold override.
    #[arg(long)]
    nms_threshold: Option<f32>,
    /// Replace an existing layer and table with the same name.
    #[arg(long, default_value_t = false)]
    overwrite: bool,
    /// Resume finalization from an existing labels/.<layer>-blockflow-work directory.
    #[arg(long, default_value_t = false, hide = true)]
    resume_work: bool,
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

    let source = AttachedImage::at(config.zarr.join(&level0.path))
        .plane(config.channel, [level0.shape[1], level0.shape[2]]);
    let (dtype, volume) = source.metadata()?;
    let weights = config
        .weights
        .clone()
        .unwrap_or_else(|| config.model.join("weights_best.h5"));
    let backend = Arc::new(StardistBackend::new(
        &config.model,
        &weights,
        config.prob_threshold,
        config.nms_threshold,
        (config.low, config.high),
    )?);
    let op = InstanceSegment::new(
        "stardist-dapi",
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
            "stardist OME-Zarr",
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

    println!(
        "cells={} label={} table={}",
        cells,
        layer_root.display(),
        table_root.display()
    );
    Ok(())
}

fn validate(config: &Config) -> Result<()> {
    if !(config.high > config.low) {
        return Err(Error::invalid("--high must be greater than --low"));
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
    for required in [
        config.model.join("config.json"),
        config.model.join("thresholds.json"),
    ] {
        if !required.is_file() {
            return Err(Error::invalid(format!(
                "missing model file {}",
                required.display()
            )));
        }
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
        .ok_or_else(|| Error::invalid("no StarDist block candidate fits the supplied constraints"))
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
