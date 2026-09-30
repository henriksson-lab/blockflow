// SPDX-License-Identifier: MIT

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use blockflow::assemble::PlanBuilder;
use blockflow::decomposition::{Constraints, Decomposition};
use blockflow::fragment::{fragment_phase, FragmentOp, PhaseWork};
use blockflow::geometry::BlockGrid;
use blockflow::label_pyramid::{build_nearest_label_pyramid_with_blocks, refresh_label_registry};
use blockflow::model_segment::{
    finalize_instances_3d, stardist::Stardist3dBackend, InstanceSegment,
};
use blockflow::ome_zarr::{OmeVolumeLevel, OmeVolumePyramid};
use blockflow::op::Chain;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, predicted_phase_prices, Hints};
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::{Dtype, Error, Result};
use clap::Parser;
use serde_json::json;

const STREAM: &str = "stardist.instances";

#[derive(Debug, Parser)]
#[command(
    name = "stardist-3d-ome-zarr",
    about = "Segment one OME-Zarr volume with StarDist 3D and add a newvolim label layer"
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
    /// Time point selected from the OME t axis.
    #[arg(long, default_value_t = 0)]
    time: usize,
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
    /// Halo in z slices. It must cover the largest expected nucleus diameter.
    #[arg(long, default_value_t = 16)]
    halo_z: usize,
    /// Halo in y and x pixels.
    #[arg(long, default_value_t = 64)]
    halo: usize,
    /// Candidate y/x block edges offered to the planner.
    #[arg(long, value_delimiter = ',', default_value = "128,256")]
    blocks: Vec<usize>,
    /// Candidate z block depths offered to the planner.
    #[arg(long, value_delimiter = ',', default_value = "16,32,64")]
    z_blocks: Vec<usize>,
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
    let pyramid = OmeVolumePyramid::open(&config.zarr, config.channel, config.time)?;
    let levels = pyramid.levels();
    if levels.is_empty() {
        return Err(Error::invalid("OME-Zarr image has no multiscale levels"));
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

    let source = pyramid.attached_level(0)?;
    let (dtype, volume) = source.metadata()?;
    let weights = config
        .weights
        .clone()
        .unwrap_or_else(|| config.model.join("weights_best.h5"));
    let backend = Arc::new(Stardist3dBackend::new(
        &config.model,
        &weights,
        config.prob_threshold,
        config.nms_threshold,
        (config.low, config.high),
    )?);
    let op = InstanceSegment::new(
        "stardist-dapi",
        backend,
        [config.halo_z, config.halo, config.halo],
        STREAM,
        Lifecycle::Persistent,
        Vec::new(),
    )
    .skipping_empty(Some(config.empty_below))
    .writing_labels();
    let row_schema = op.schema()?;

    let constraints = constraints(&config, volume);
    let grid = planned_fragment_grid(&op, dtype, volume, &constraints, &config.z_blocks)?;
    println!("planned block {:?} over {:?}", grid.block(), volume);
    let mut builder = PlanBuilder::new(volume, dtype, grid);
    builder.fragments(op)?;
    let assembly = builder.finish()?;
    let env = ZarrEnvironment::attach(&work, std::slice::from_ref(&source))?;
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
    let label_shapes = levels.iter().map(|level| level.shape).collect::<Vec<_>>();
    let pyramid_blocks = volume_block_candidates(&config.z_blocks, &config.blocks);
    build_nearest_label_pyramid_with_blocks(
        &staged_layer,
        &work,
        &label_shapes,
        &pyramid_blocks,
        config.workers,
    )?;
    let rows_env = ZarrEnvironment::attach(&work, &[source])?;
    let table = finalize_instances_3d(
        &rows_env,
        STREAM,
        phase,
        volume,
        row_schema,
        &staged_table,
        &work,
        &config.layer,
        pyramid.voxel_volume(),
    )?;
    drop(rows_env);
    write_label_metadata(&staged_layer, levels, pyramid.label_source())?;
    let cells = table.spec().row_count;
    drop(table);
    replace_with(&staged_layer, &layer_root, config.overwrite)?;
    replace_with(&staged_table, &table_root, config.overwrite)?;
    refresh_label_registry(&config.zarr.join("labels"))?;
    fs::remove_dir_all(&work).map_err(Error::backend)?;

    println!(
        "cells={} elapsed_seconds={:.3} label={} table={}",
        cells,
        run_started.elapsed().as_secs_f64(),
        layer_root.display(),
        table_root.display()
    );
    Ok(())
}

fn validate(config: &Config) -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::invalid("this example requires cargo run --release"));
    }
    if config.high.partial_cmp(&config.low) != Some(std::cmp::Ordering::Greater) {
        return Err(Error::invalid("--high must be greater than --low"));
    }
    if config.halo_z == 0
        || config.halo == 0
        || config.blocks.is_empty()
        || config.z_blocks.is_empty()
        || config.blocks.contains(&0)
        || config.z_blocks.contains(&0)
    {
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
    z_blocks: &[usize],
) -> Result<BlockGrid> {
    let chain = Chain::sequence(Vec::new());
    let mut best: Option<(f64, usize, BlockGrid)> = None;
    for &z_edge in z_blocks {
        for &edge in &constraints.block_candidates {
            let grid = BlockGrid::new(volume, [z_edge, edge, edge])?;
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
            let candidate_size = z_edge.saturating_mul(edge).saturating_mul(edge);
            if best.as_ref().is_none_or(|(old, old_size, _)| {
                (*makespan, std::cmp::Reverse(candidate_size))
                    < (*old, std::cmp::Reverse(*old_size))
            }) {
                best = Some((*makespan, candidate_size, grid));
            }
        }
    }
    best.map(|(_, _, grid)| grid)
        .ok_or_else(|| Error::invalid("no StarDist block candidate fits the supplied constraints"))
}

fn volume_block_candidates(z: &[usize], xy: &[usize]) -> Vec<[usize; 3]> {
    z.iter()
        .flat_map(|&depth| xy.iter().map(move |&edge| [depth, edge, edge]))
        .collect()
}

fn write_label_metadata(root: &Path, levels: &[OmeVolumeLevel], source: &str) -> Result<()> {
    let datasets = levels
        .iter()
        .enumerate()
        .map(|(index, level)| {
            json!({"path": index.to_string(), "coordinateTransformations": level.coordinate_transformations})
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
            "image-label": {"version":"0.5", "source":{"image":source}}
        }
    });
    fs::write(
        root.join("zarr.json"),
        serde_json::to_vec_pretty(&metadata).map_err(Error::backend)?,
    )
    .map_err(Error::backend)
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
