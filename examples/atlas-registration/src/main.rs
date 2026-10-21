// SPDX-License-Identifier: MIT
// Original work for this crate.

use std::path::PathBuf;

use blockflow::error::{Error, Result};
use blockflow::iterate::SubstageLimit;
use blockflow_atlas_registration::compute::{
    params_from_elastix_map, ControlGrid, Metric, PyramidSchedule, Sampling, VolumeFitParams,
};
use blockflow_atlas_registration::pipeline::{
    register, register_chain, transform_points, ChainConfig, Initialisation, Input, PointSource,
    PointsConfig, RegisterConfig, RegisterSummary, Stage, Units,
};
use blockflow_atlas_registration::stochastic::{
    is_stochastic, stochastic_from_elastix_map, RelativeTolerance,
};
use blockflow_atlas_registration::{ParameterMap, PointSpace};
use clap::{Args, Parser, Subcommand, ValueEnum};

/// Register OME-Zarr volumes and carry points between them.
#[derive(Parser)]
#[command(name = "atlas-registration")]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// Memory cap in MiB for the planned volume phases and their chunk caches
    /// (default: $BLOCKFLOW_MEMORY_CAP, else the cgroup's limit, else a share
    /// of the machine). The registration's own resident volumes are not
    /// planned and are not bounded by it.
    #[arg(long, global = true)]
    memory_mb: Option<u64>,
}

#[derive(Subcommand)]
enum Command {
    /// Fit a transform from the fixed volume to the moving one.
    Register(Box<RegisterArgs>),
    /// Register a chain of images, each (fixed) to the next (moving): the
    /// sample's signal channel, its autofluorescence, the atlas.
    Chain(ChainArgs),
    /// Carry a CSV point table through a registration.
    TransformPoints(PointsArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum ModelArg {
    Translation,
    Affine,
    Bspline,
    /// An affine, then a B-spline on the affinely warped moving image.
    AffineBspline,
}

#[derive(Clone, Copy, ValueEnum)]
enum MetricArg {
    MeanSquares,
    Correlation,
    MutualInformation,
}

#[derive(Args)]
struct RegisterArgs {
    /// Fixed OME-Zarr image (the sample, to carry its points into an atlas).
    #[arg(long)]
    fixed: PathBuf,
    #[arg(long, default_value_t = 0)]
    fixed_channel: usize,
    /// Fixed pyramid level; default: the coarsest no coarser than --resolution.
    #[arg(long)]
    fixed_level: Option<usize>,
    /// Moving OME-Zarr image (the atlas reference, typically).
    #[arg(long)]
    moving: PathBuf,
    #[arg(long, default_value_t = 0)]
    moving_channel: usize,
    #[arg(long)]
    moving_level: Option<usize>,
    /// Registration voxel size, `s` or `z,y,x`, in the images' unit; default:
    /// the fixed level as stored.
    #[arg(long, value_parser = parse_triple)]
    resolution: Option<[f64; 3]>,
    /// Output directory for TransformParameters.N.txt and registration.json.
    #[arg(long)]
    out: PathBuf,
    /// How the first stage starts (stochastic stages): `parameters` (as the
    /// file's AutomaticTransformInitialization says), `physical` (the images'
    /// physical frames as they stand, elastix's "false"), or `centre` (their
    /// geometric centres, elastix's "true" with GeometricalCenter).
    #[arg(
        long,
        default_value = "parameters",
        conflicts_with = "initial_translation"
    )]
    initialise: String,
    /// Start the first stage from the physical frames with the fixed image
    /// moved by `Z,Y,X` (the images' unit): its point p starts at the moving
    /// image's point p + t (stochastic translation or affine first stage).
    #[arg(long, value_parser = parse_triple, allow_hyphen_values = true)]
    initial_translation: Option<[f64; 3]>,
    #[arg(long, value_enum, default_value_t = ModelArg::Affine)]
    model: ModelArg,
    #[arg(long, value_enum, default_value_t = MetricArg::Correlation)]
    metric: MetricArg,
    /// elastix registration parameter file, one per stage; replaces --model
    /// and --metric. Files with a stochastic optimiser (the reference's
    /// `align_affine.txt`, `align_bspline.txt`) run the stochastic fit.
    #[arg(long)]
    parameters: Vec<PathBuf>,
    #[command(flatten)]
    stochastic: StochasticArgs,
    /// Gauss-Newton iterations per stage.
    #[arg(long, default_value_t = 64)]
    iterations: usize,
    /// Stop a stage once its step norm or cost change is no larger than this.
    /// Absolute: suits correlation, whose cost is in [-1, 1]; a sum of squared
    /// differences over a large lattice rarely changes this little.
    #[arg(long, default_value_t = 1e-2)]
    tolerance: f64,
    /// Coarse-to-fine levels (2x per level); 1 fits the lattice only.
    #[arg(long, default_value_t = 1)]
    levels: usize,
    /// B-spline control grid size, `n` or `z,y,x` control points.
    #[arg(long, value_parser = parse_grid, default_value = "6")]
    grid: [usize; 3],
    /// B-spline control-coefficient smoothness penalty.
    #[arg(long, default_value_t = 0.0)]
    smoothness: f64,
    /// Fit on every n-th voxel per axis, `n` or `z,y,x`.
    #[arg(long, value_parser = parse_grid)]
    stride: Option<[usize; 3]>,
    /// Write the moving image (or --warp-source) warped onto the fixed lattice.
    #[arg(long)]
    warped: Option<PathBuf>,
    /// Image to warp instead of the moving one, on the moving image's frame.
    #[arg(long)]
    warp_source: Option<PathBuf>,
    /// Nearest-neighbour warp, for label images.
    #[arg(long)]
    nearest: bool,
    #[arg(long, default_value_t = 64)]
    block: usize,
}

/// Options of the stochastic fit (elastix parameter files).
#[derive(Args, Clone)]
struct StochasticArgs {
    /// Seed of the random samplers; the result depends on it and not on the
    /// number of threads.
    #[arg(long, default_value_t = 121_212)]
    seed: u64,
    /// Stop a level early once the mean cost over 50 iterations changes by no
    /// more than this fraction (elastix always runs every iteration).
    #[arg(long)]
    relative_tolerance: Option<f64>,
    /// Write the moving image warped after each stage, as
    /// `stage{N}_warped.ome.zarr` in the output directory.
    #[arg(long)]
    warped_stages: bool,
}

#[derive(Args)]
struct ChainArgs {
    /// The images, first (the sample) to last (the atlas): each is the fixed
    /// image of one step and the moving image of the step before it.
    #[arg(long = "image", required = true, num_args = 1)]
    images: Vec<PathBuf>,
    /// `STEP=FILE`: an elastix parameter file for step STEP (0-based), in
    /// order; e.g. `0=align_affine.txt 1=align_affine.txt 1=align_bspline.txt`.
    #[arg(long = "parameters", required = true, num_args = 1)]
    parameters: Vec<String>,
    /// Registration voxel size, `s` or `z,y,x`.
    #[arg(long, value_parser = parse_triple)]
    resolution: Option<[f64; 3]>,
    /// `STEP=MODE`: how step STEP's first stage starts, `parameters`
    /// (default), `physical` or `centre` (see `register --initialise`).
    #[arg(long = "initialise", num_args = 1)]
    initialise: Vec<String>,
    /// `STEP=Z,Y,X`: start step STEP from the physical frames with its fixed
    /// image moved by this translation (see `register
    /// --initial-translation`).
    #[arg(long = "initial-translation", num_args = 1, allow_hyphen_values = true)]
    initial_translation: Vec<String>,
    /// Output directory: `step{k}/` per step and `chain.json`.
    #[arg(long)]
    out: PathBuf,
    /// Write each step's moving image warped onto its fixed lattice.
    #[arg(long)]
    warped: bool,
    #[command(flatten)]
    stochastic: StochasticArgs,
    #[arg(long, default_value_t = 64)]
    block: usize,
}

#[derive(Clone, Copy, ValueEnum)]
enum UnitsArg {
    Voxel,
    Physical,
}

#[derive(Clone, Copy, ValueEnum)]
enum SpaceArg {
    Physical,
    Index,
}

#[derive(Args)]
struct PointsArgs {
    /// A directory `register` wrote.
    #[arg(
        long,
        conflicts_with = "transform",
        required_unless_present = "transform"
    )]
    registration: Option<PathBuf>,
    /// A bare elastix transform file instead.
    #[arg(long)]
    transform: Option<PathBuf>,
    /// Point space for a bare transform file.
    #[arg(long, value_enum, default_value_t = SpaceArg::Physical)]
    space: SpaceArg,
    /// Input CSV with a header row.
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// The coordinate columns, `z,y,x`.
    #[arg(long, default_value = "z,y,x")]
    columns: String,
    #[arg(long, default_value = "moving_")]
    prefix: String,
    /// Input and output units with --registration: fixed level-0 voxels in,
    /// moving level-0 voxels out; or physical in and out.
    #[arg(long, value_enum, default_value_t = UnitsArg::Voxel)]
    units: UnitsArg,
    /// Reverse each point around a bare transform file written in x,y,z order.
    #[arg(long)]
    reverse_axes: bool,
    /// From the last moving image (the atlas) back to the first fixed image
    /// (the sample), by inverting the transform numerically.
    #[arg(long)]
    inverse: bool,
}

fn parse_triple(text: &str) -> std::result::Result<[f64; 3], String> {
    let values = text
        .split(',')
        .map(|v| v.trim().parse::<f64>().map_err(|e| format!("`{v}`: {e}")))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    match values.as_slice() {
        [one] => Ok([*one; 3]),
        [z, y, x] => Ok([*z, *y, *x]),
        _ => Err(format!("expected one value or z,y,x, got `{text}`")),
    }
}

fn parse_grid(text: &str) -> std::result::Result<[usize; 3], String> {
    let values = parse_triple(text)?;
    if values.iter().any(|v| v.fract() != 0.0 || *v < 1.0) {
        return Err(format!("expected positive integers, got `{text}`"));
    }
    Ok(values.map(|v| v as usize))
}

/// One stage from an elastix parameter file: stochastic for elastix's
/// stochastic optimisers, the native Gauss-Newton fit otherwise.
fn stage_from_file(path: &PathBuf, options: &StochasticArgs) -> Result<Stage> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| Error::invalid(format!("{}: {error}", path.display())))?;
    let map = ParameterMap::parse(&text)?;
    let named = |error: Error| Error::invalid(format!("{}: {error}", path.display()));
    if is_stochastic(&map)? {
        let mut params = stochastic_from_elastix_map(&map, options.seed).map_err(named)?;
        params.relative_tolerance = options
            .relative_tolerance
            .map(|tolerance| RelativeTolerance {
                window: 50,
                tolerance,
            });
        Ok(Stage::Stochastic(params))
    } else {
        Ok(Stage::Native(params_from_elastix_map(&map).map_err(named)?))
    }
}

fn stages(args: &RegisterArgs) -> Result<Vec<Stage>> {
    if !args.parameters.is_empty() {
        return args
            .parameters
            .iter()
            .map(|path| stage_from_file(path, &args.stochastic))
            .collect();
    }
    let metric = match args.metric {
        MetricArg::MeanSquares => Metric::MeanSquares,
        MetricArg::Correlation => Metric::NormalizedCorrelation,
        MetricArg::MutualInformation => Metric::MutualInformation,
    };
    let mut base = VolumeFitParams::default().with_metric(metric);
    base.limit = SubstageLimit::of(args.iterations)?;
    base.tolerance = args.tolerance;
    if args.levels > 1 {
        base = base.with_pyramid(PyramidSchedule::powers_of_two(args.levels)?);
    }
    if let Some(stride) = args.stride {
        base = base.with_sampling(Sampling::Stride(stride));
    }
    // Affine parameters are in voxels and the matrix entries multiply
    // coordinates of hundreds of voxels: a step bound that suits a translation
    // throws an affine off the image.
    let mut affine = base.clone().affine();
    affine.limit = base.limit;
    affine.max_step = 0.05;
    let mut bspline = base
        .clone()
        .bspline()
        .with_bspline_grid(ControlGrid::new(args.grid)?)
        .with_bspline_smoothness(args.smoothness)?;
    bspline.limit = base.limit;
    bspline.max_step = 0.5;
    Ok(match args.model {
        ModelArg::Translation => vec![base.into()],
        ModelArg::Affine => vec![affine.into()],
        ModelArg::Bspline => vec![bspline.into()],
        ModelArg::AffineBspline => vec![affine.into(), bspline.into()],
    })
}

fn print_summary(summary: &RegisterSummary) {
    println!(
        "lattice={:?} spacing={:?} fixed_level={} moving_level={}",
        summary.lattice_shape,
        summary.lattice_frame.scale,
        summary.fixed_level,
        summary.moving_level
    );
    for (index, stage) in summary.stages.iter().enumerate() {
        println!(
            "stage {index}: {:?} substages={:?} cost={:?} converged_by={:?} seconds={:?} \
             similarity={:?}",
            stage.model,
            stage.substages,
            stage.final_cost,
            stage.converged_by,
            stage.seconds,
            stage.similarity
        );
    }
    println!("transform={}", summary.transform_file.display());
    if let Some(path) = &summary.warped {
        println!("warped={}", path.display());
    }
}

fn run_chain(args: &ChainArgs) -> Result<()> {
    let steps_count = args.images.len().saturating_sub(1);
    let mut steps: Vec<Vec<Stage>> = vec![Vec::new(); steps_count];
    for entry in &args.parameters {
        let (step, file) = entry
            .split_once('=')
            .ok_or_else(|| Error::invalid(format!("--parameters `{entry}`: expected STEP=FILE")))?;
        let step: usize = step
            .trim()
            .parse()
            .map_err(|error| Error::invalid(format!("--parameters `{entry}`: {error}")))?;
        if step >= steps_count {
            return Err(Error::invalid(format!(
                "--parameters `{entry}`: {} images make {steps_count} step(s)",
                args.images.len()
            )));
        }
        steps[step].push(stage_from_file(
            &PathBuf::from(file.trim()),
            &args.stochastic,
        )?);
    }
    if let Some(empty) = steps.iter().position(Vec::is_empty) {
        return Err(Error::invalid(format!(
            "chain: step {empty} has no parameter file"
        )));
    }
    let initialisation =
        chain_initialisation(steps_count, &args.initialise, &args.initial_translation)?;
    let summaries = register_chain(&ChainConfig {
        images: args
            .images
            .iter()
            .map(|path| Input {
                path: path.clone(),
                channel: 0,
                level: None,
            })
            .collect(),
        steps,
        resolution: args.resolution,
        out: args.out.clone(),
        block: args.block,
        workers: rayon::current_num_threads(),
        warped_stages: args.stochastic.warped_stages,
        warped: args.warped,
        initialisation,
    })?;
    for (index, summary) in summaries.iter().enumerate() {
        println!("step {index}:");
        print_summary(summary);
    }
    println!("chain={}", args.out.display());
    Ok(())
}

/// Per step, from `STEP=MODE` and `STEP=Z,Y,X` entries; one entry per step
/// at most.
fn chain_initialisation(
    steps: usize,
    modes: &[String],
    translations: &[String],
) -> Result<Vec<Initialisation>> {
    let mut out: Vec<Option<Initialisation>> = vec![None; steps];
    let entries = modes.iter().map(|entry| (entry, "--initialise")).chain(
        translations
            .iter()
            .map(|entry| (entry, "--initial-translation")),
    );
    for (entry, flag) in entries {
        let (step, value) = entry
            .split_once('=')
            .ok_or_else(|| Error::invalid(format!("{flag} `{entry}`: expected STEP=VALUE")))?;
        let step: usize = step
            .trim()
            .parse()
            .map_err(|error| Error::invalid(format!("{flag} `{entry}`: {error}")))?;
        if step >= steps {
            return Err(Error::invalid(format!(
                "{flag} `{entry}`: the chain has {steps} step(s)"
            )));
        }
        let parsed = if flag == "--initialise" {
            Initialisation::parse(value)?
        } else {
            Initialisation::Translation(
                parse_triple(value)
                    .map_err(|error| Error::invalid(format!("{flag} `{entry}`: {error}")))?,
            )
        };
        if out[step].replace(parsed).is_some() {
            return Err(Error::invalid(format!(
                "step {step} is given more than one initialisation"
            )));
        }
    }
    Ok(out.into_iter().map(Option::unwrap_or_default).collect())
}

fn run_register(args: &RegisterArgs) -> Result<()> {
    let config = RegisterConfig {
        fixed: Input {
            path: args.fixed.clone(),
            channel: args.fixed_channel,
            level: args.fixed_level,
        },
        moving: Input {
            path: args.moving.clone(),
            channel: args.moving_channel,
            level: args.moving_level,
        },
        resolution: args.resolution,
        stages: stages(args)?,
        out: args.out.clone(),
        warped: args.warped.clone(),
        warp_source: args.warp_source.clone().map(|path| Input {
            path,
            channel: 0,
            level: None,
        }),
        nearest: args.nearest,
        block: args.block,
        workers: rayon::current_num_threads(),
        warped_stages: args.stochastic.warped_stages,
        initialisation: match args.initial_translation {
            Some(t) => Initialisation::Translation(t),
            None => Initialisation::parse(&args.initialise)?,
        },
    };
    let summary = register(&config)?;
    print_summary(&summary);
    Ok(())
}

fn run_points(args: &PointsArgs) -> Result<()> {
    let columns: Vec<String> = args
        .columns
        .split(',')
        .map(|c| c.trim().to_string())
        .collect();
    let [z, y, x] = <[String; 3]>::try_from(columns)
        .map_err(|_| Error::invalid("--columns needs three names, z,y,x"))?;
    let source = match (&args.registration, &args.transform) {
        (Some(directory), _) => PointSource::Registration(directory.clone()),
        (None, Some(file)) => PointSource::Transform {
            file: file.clone(),
            space: match args.space {
                SpaceArg::Physical => PointSpace::Physical,
                SpaceArg::Index => PointSpace::Index,
            },
        },
        (None, None) => return Err(Error::invalid("--registration or --transform is needed")),
    };
    let rows = transform_points(&PointsConfig {
        source,
        input: args.input.clone(),
        output: args.output.clone(),
        columns: [z, y, x],
        prefix: args.prefix.clone(),
        units: match args.units {
            UnitsArg::Voxel => Units::Voxel,
            UnitsArg::Physical => Units::Physical,
        },
        reverse_axes: args.reverse_axes,
        inverse: args.inverse,
    })?;
    println!("points={rows} output={}", args.output.display());
    Ok(())
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("atlas-registration: use a release build (cargo run --release)");
        std::process::exit(2);
    }
    let cli = Cli::parse();
    blockflow::budget::set_process_cap(blockflow::budget::MemoryCap::resolve(
        cli.memory_mb.map(|mib| mib << 20),
    ));
    let result = match &cli.command {
        Command::Register(args) => run_register(args),
        Command::Chain(args) => run_chain(args),
        Command::TransformPoints(args) => run_points(args),
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
