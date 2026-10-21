// SPDX-License-Identifier: MIT
//! Run a bounded physical plane sampler on prepared Zarr input and support.
use blockflow::{
    assemble::PlanBuilder,
    geometry::BlockGrid,
    ops::plane_sample::{append_plane_sample_phase, PhysicalGrid, PlaneGrid, PlaneSampleOp},
    strategy::{execute_phases, Hints},
    zarr_env::{AttachedImage, ZarrEnvironment},
    Error, Result, Voxels,
};
use clap::{Parser, Subcommand};
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Prepare a deterministic analytic Zarr fixture, outside measured execution.
    Prepare {
        #[arg(long)]
        out: PathBuf,
    },
    /// Sample image and support independently on the same plane and depth.
    Sample {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}
#[derive(Serialize, Deserialize)]
struct Config {
    input: PathBuf,
    support: PathBuf,
    origin_zyx: [f64; 3],
    voxel_spacing_zyx: [f64; 3],
    unit: String,
    center_zyx: [f64; 3],
    row_tangent_zyx: [f64; 3],
    column_tangent_zyx: [f64; 3],
    normal_zyx: [f64; 3],
    shape_rc: [usize; 2],
    spacing_rc: [f64; 2],
    depth: f64,
    block_rc: [usize; 2],
}
fn fail(e: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(e.to_string())
}
fn prepare(out: &Path) -> Result<()> {
    if out.exists() {
        return Err(fail("fixture destination exists; choose a new directory"));
    }
    std::fs::create_dir_all(out).map_err(fail)?;
    let out = out.canonicalize().map_err(fail)?;
    let shape = [32, 48, 56];
    let image = Array3::from_shape_fn(shape, |q| (2 * q.0 + 3 * q.1 + q.2) as f32 + 0.125);
    let support = Array3::from_shape_fn(shape, |q| q.1 >= 15 && q.2 >= 17);
    // Fixture preparation may hold these small analytic arrays; Sample never
    // reads a whole source array. Separate stores keep support independent.
    ZarrEnvironment::create(out.join("image"), &Voxels::from(image), [8; 3])?;
    ZarrEnvironment::create(out.join("support"), &Voxels::from(support), [8; 3])?;
    let s = 0.5_f64.sqrt();
    let config = Config {
        input: out.join("image/level0"),
        support: out.join("support/level0"),
        origin_zyx: [10., -5., 8.],
        voxel_spacing_zyx: [2., 0.7, 1.3],
        unit: "example_units".into(),
        center_zyx: [40., 9., 34.],
        row_tangent_zyx: [s, s, 0.],
        column_tangent_zyx: [0., 0., 1.],
        normal_zyx: [s, -s, 0.],
        shape_rc: [25, 31],
        spacing_rc: [0.8, 1.1],
        depth: 1.7,
        block_rc: [7, 9],
    };
    std::fs::write(
        out.join("plane.json"),
        serde_json::to_vec_pretty(&config).map_err(fail)?,
    )
    .map_err(fail)?;
    println!(
        "{}",
        json!({"config":out.join("plane.json"),"scope":"analytic software fixture; not biological validation"})
    );
    Ok(())
}
fn sample(config_path: &Path, out: &Path) -> Result<()> {
    if out.exists() {
        return Err(fail("output exists; choose a new directory"));
    }
    let config: Config =
        serde_json::from_slice(&std::fs::read(config_path).map_err(fail)?).map_err(fail)?;
    if config.unit.is_empty() {
        return Err(fail("supply a physical unit label"));
    }
    let grid = PhysicalGrid::new(config.origin_zyx, config.voxel_spacing_zyx)?;
    let plane = PlaneGrid::new(
        config.center_zyx,
        config.row_tangent_zyx,
        config.column_tangent_zyx,
        config.normal_zyx,
        config.shape_rc,
        config.spacing_rc,
    )?;
    let output_shape = [1, config.shape_rc[0], config.shape_rc[1]];
    let op = PlaneSampleOp::new(grid, plane, config.depth)?;
    let mut records = vec![];
    let mut input_shape = None;
    for (name, path) in [("intensity", &config.input), ("support", &config.support)] {
        let image = AttachedImage::at(path);
        let (dtype, shape) = image.metadata()?;
        if input_shape.is_some_and(|s| s != shape) {
            return Err(fail(
                "intensity/support grids must have equal shape and the configured physical frame",
            ));
        }
        input_shape = Some(shape);
        let mut builder = PlanBuilder::new(shape, dtype, BlockGrid::whole(shape)?);
        append_plane_sample_phase(
            &mut builder,
            op.clone(),
            shape,
            BlockGrid::new(output_shape, [1, config.block_rc[0], config.block_rc[1]])?,
        )?;
        let assembly = builder.finish()?;
        let max_source_voxels = assembly.decomposition.phases[0]
            .blocks
            .iter()
            .map(|b| b.source.voxels())
            .max()
            .unwrap_or(0);
        let env = ZarrEnvironment::attach(out.join(name), &[image])?;
        let start = Instant::now();
        execute_phases(
            "physical-plane-zarr",
            &assembly.workflow,
            &assembly.decomposition,
            &Hints {
                concurrency: 2,
                ..Hints::default()
            },
            &env,
            &[],
            &assembly.work(),
        )?;
        records.push(json!({"component":name,"input":path,"output":out.join(name).join("level1"),"input_shape":shape,"output_shape":output_shape,"blocks":assembly.decomposition.phases[0].blocks.len(),"max_source_voxels_per_block":max_source_voxels,"planned_seconds":start.elapsed().as_secs_f64()}));
    }
    let report = json!({"config":config,"outputs":records,"output_axes":["singleton","row","column"],"invalid":"NaN outside voxel-center domain or at nonfinite contributing values","support":"sampled separately without intensity division or validity threshold","coordinate_map":"origin + index * voxel_spacing in z,y,x; plane basis and signed depth are in config","scope":"bounded planned Zarr sampling, no registration or biological accuracy claim"});
    std::fs::write(
        out.join("sampling.json"),
        serde_json::to_vec_pretty(&report).map_err(fail)?,
    )
    .map_err(fail)?;
    println!("{}", serde_json::to_string(&report).map_err(fail)?);
    Ok(())
}
fn main() {
    if cfg!(debug_assertions) {
        eprintln!("use cargo run --release");
        std::process::exit(2);
    }
    let result = match Cli::parse().command {
        Command::Prepare { out } => prepare(&out),
        Command::Sample { config, out } => sample(&config, &out),
    };
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
