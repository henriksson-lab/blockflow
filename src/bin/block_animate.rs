// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// One run, one log, one movie.
//
//     cargo run --release --bin block_animate -- --grid 5,5,5 --render
//
// What this is, next to `block_progress_export`
// ---------------------------------------------
// `block_progress_export` writes *two* logs — the same workflow and the same
// decomposition under two schedules — because the pair is the evidence that the
// picture follows the schedule rather than sweeping generically. That remains,
// and the two-panel view still draws it.
//
// This binary is the other job: take **one** run and show it as what it is, a
// lattice of blocks in three dimensions with the camera moving around it. It is
// for presentation and intuition. Serious diagnosis wants a live view polling
// `LatestOpPerChunk` during the run, which is a different program.
//
// The two halves, and which one is optional
// -----------------------------------------
// Exporting is Rust and always runs. Rendering shells out to the bundled
// renderer and only happens if asked for with `--render`; see `animate.rs` for
// why the seam is where it is. With no Python on the machine this binary still
// writes a usable log and says so.
//
// Options
// -------
// ```text
//   --out PATH          log to write            (default ./block_flight.json)
//   --grid NX,NY,NZ     blocks per axis         (default 4,4,4)
//   --block N           block edge, voxels      (default 64)
//   --schedule S        block-major|phase-major (default block-major)
//   --concurrency N     tasks in flight         (default 4)
//   --render            draw the movie after exporting
//   --view 3d|2d        which picture           (default 3d)
//   --output NAME       movie base name         (default block_flight)
//   --quality l|m|h     frame size              (default l)
//   --max-steps N       ceiling on rendered steps (default 40)
//   --seconds-per-step  pace                    (default 0.15)
//   --media-dir DIR     where manim writes      (default ./media)
//   --allow-large       render past the practical block ceiling
// ```

use blockflow::animate::{
    render, within_volume3d_ceiling, Quality, RenderRequest, View, VOLUME3D_BLOCK_CEILING,
};
use blockflow::decomposition::{groups_for, summarise_slots, Decomposition, PhaseDecomposition};
use blockflow::env::AccountingEnvironment;
use blockflow::export::{write_order_log_json, ExportMeta};
use blockflow::geometry::BlockGrid;
use blockflow::probes::IdentityOp;
use blockflow::strategy::{execute_observed, Hints, SchedulePriority, Workflow};
use blockflow::{Chain, Dtype, Error, Result};

/// The command line, parsed by hand: this crate's dependency list is short on
/// purpose and one binary does not justify adding an argument parser to it.
struct Options {
    out: std::path::PathBuf,
    grid: [usize; 3],
    block: usize,
    schedule: (&'static str, SchedulePriority),
    concurrency: usize,
    render: bool,
    request: RenderRequest,
}

fn parse_grid(text: &str) -> Result<[usize; 3]> {
    let parts: Vec<&str> = text.split(',').collect();
    let numbers: Vec<usize> = match parts.len() {
        // A single number is the same count on every axis, which is the common
        // case and the one worth a shorthand.
        1 => {
            let n = parse_number(parts[0])?;
            vec![n, n, n]
        }
        3 => parts
            .iter()
            .map(|part| parse_number(part))
            .collect::<Result<Vec<_>>>()?,
        _ => {
            return Err(Error::InvalidArgument(format!(
                "--grid wants N or NX,NY,NZ, got {text:?}"
            )))
        }
    };
    if numbers.contains(&0) {
        return Err(Error::InvalidArgument(
            "--grid: a lattice with no blocks on an axis has nothing to draw".to_string(),
        ));
    }
    Ok([numbers[0], numbers[1], numbers[2]])
}

fn parse_number(text: &str) -> Result<usize> {
    text.trim()
        .parse()
        .map_err(|_| Error::InvalidArgument(format!("expected a number, got {text:?}")))
}

fn parse_float(text: &str) -> Result<f64> {
    text.trim()
        .parse()
        .map_err(|_| Error::InvalidArgument(format!("expected a number, got {text:?}")))
}

fn parse_options() -> Result<Options> {
    let mut options = Options {
        out: std::path::PathBuf::from("block_flight.json"),
        grid: [4, 4, 4],
        block: 64,
        schedule: ("block_major", SchedulePriority::BlockMajor),
        concurrency: 4,
        render: false,
        request: RenderRequest::new("block_flight.json"),
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        // `--flag=value` and `--flag value` both, because typing either and
        // being told off is a waste of everyone's time.
        let (flag, inline) = match argument.split_once('=') {
            Some((flag, value)) => (flag.to_string(), Some(value.to_string())),
            None => (argument.clone(), None),
        };
        let mut value = || -> Result<String> {
            match &inline {
                Some(value) => Ok(value.clone()),
                None => arguments
                    .next()
                    .ok_or_else(|| Error::InvalidArgument(format!("{flag} wants a value"))),
            }
        };
        match flag.as_str() {
            "--out" => options.out = std::path::PathBuf::from(value()?),
            "--grid" => options.grid = parse_grid(&value()?)?,
            "--block" => options.block = parse_number(&value()?)?,
            "--schedule" => {
                options.schedule = match value()?.as_str() {
                    "block-major" | "block_major" => ("block_major", SchedulePriority::BlockMajor),
                    "phase-major" | "phase_major" => ("phase_major", SchedulePriority::PhaseMajor),
                    other => {
                        return Err(Error::InvalidArgument(format!(
                            "--schedule wants block-major or phase-major, got {other:?}"
                        )))
                    }
                }
            }
            "--concurrency" => options.concurrency = parse_number(&value()?)?.max(1),
            "--render" => options.render = true,
            "--view" => {
                options.request.view = match value()?.as_str() {
                    "3d" => View::Volume3d,
                    "2d" => View::Grid2d,
                    other => {
                        return Err(Error::InvalidArgument(format!(
                            "--view wants 3d or 2d, got {other:?}"
                        )))
                    }
                }
            }
            "--output" => options.request.output = value()?,
            "--quality" => {
                options.request.quality = match value()?.as_str() {
                    "l" => Quality::Low,
                    "m" => Quality::Medium,
                    "h" => Quality::High,
                    other => {
                        return Err(Error::InvalidArgument(format!(
                            "--quality wants l, m or h, got {other:?}"
                        )))
                    }
                }
            }
            "--max-steps" => options.request.max_steps = parse_number(&value()?)?.max(1),
            "--seconds-per-step" => options.request.seconds_per_step = parse_float(&value()?)?,
            "--fps" => options.request.fps = Some(parse_number(&value()?)? as u32),
            "--media-dir" => options.request.media_dir = Some(std::path::PathBuf::from(value()?)),
            "--python" => options.request.python = Some(std::path::PathBuf::from(value()?)),
            "--allow-large" => options.request.allow_large = true,
            "-h" | "--help" => {
                println!("{}", HELP);
                std::process::exit(0);
            }
            other => {
                return Err(Error::InvalidArgument(format!(
                    "unknown option {other:?}. --help lists them."
                )))
            }
        }
    }
    options.request.logs = vec![options.out.clone()];
    options.request.verbose = true;
    Ok(options)
}

const HELP: &str = "\
block_animate — run a workflow, export its order log, optionally draw it.

    block_animate [--out LOG.json] [--grid N|NX,NY,NZ] [--block VOXELS]
                  [--schedule block-major|phase-major] [--concurrency N]
                  [--render] [--view 3d|2d] [--output NAME] [--quality l|m|h]
                  [--max-steps N] [--seconds-per-step S] [--fps N]
                  [--media-dir DIR] [--python PATH] [--allow-large]

Exporting needs nothing but this binary. Rendering needs Python with manim, and
happens only with --render.";

/// Every failure here is a message meant to be read — a missing interpreter, a
/// lattice too large to draw — so it is printed with `Display` and not with the
/// `Debug` of an `Err` returned from `main`, which escapes the newlines out of
/// exactly the messages that need them.
fn main() {
    if let Err(error) = run() {
        eprintln!("block_animate: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let options = parse_options()?;

    // Voxels are notional: the run goes through the accounting environment, so
    // nothing is allocated and the lattice can be as large as the log consumer
    // will tolerate.
    let shape = [
        options.grid[0] * options.block,
        options.grid[1] * options.block,
        options.grid[2] * options.block,
    ];
    let chain = Chain::sequence(vec![
        Chain::op(IdentityOp::new("median", [2, 2, 2]).with_cost(1.0)),
        Chain::op(IdentityOp::new("background", [4, 6, 6]).with_cost(2.0)),
        Chain::op(IdentityOp::new("threshold", [0, 0, 0]).with_cost(0.5)),
    ]);
    let ops: Vec<(usize, String)> = chain
        .slots()
        .iter()
        .enumerate()
        .map(|(slot, sub)| (slot, sub.display_name()))
        .collect();
    let workflow = Workflow::new(chain, shape, Dtype::U16);

    // The decomposition is stated rather than planned, for the same reason the
    // two-schedule export states it: what varies here is the *schedule*, and a
    // planner free to choose a different grid per schedule would confound that.
    let slots = workflow.chain.slots();
    let phases: Vec<PhaseDecomposition> = groups_for(0b11, slots.len())
        .iter()
        .map(|group| {
            let (reach, _, names, _) = summarise_slots(&slots, group, shape)?;
            // Split on all three axes: a lattice one block deep is a picture
            // that would have been better drawn flat.
            let grid = BlockGrid::along(shape, &[0, 1, 2], options.block)?;
            Ok(PhaseDecomposition::derive(
                group.clone(),
                names,
                reach.clone(),
                reach,
                grid,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let decomposition = Decomposition {
        volume: shape,
        dtype: workflow.dtype,
        phases,
        chain_reach: workflow.chain.reach3(&shape),
    };
    decomposition.check()?;

    let hints = Hints {
        priority: options.schedule.1,
        concurrency: options.concurrency,
        ..Hints::default()
    };
    let env = AccountingEnvironment::new(shape, [options.block, options.block, options.block], 2);
    let stats = execute_observed(
        options.schedule.0,
        &workflow,
        &decomposition,
        &hints,
        &env,
        &[],
    )?;

    let meta = ExportMeta::new(options.schedule.0, shape, decomposition.n_phases()).with_ops(ops);
    write_order_log_json(&stats.log, &meta, &options.out)?;
    let blocks: usize = options.grid.iter().product();
    println!(
        "{:<12} grid {:?}  {} blocks  {} tasks  {} events  -> {}",
        options.schedule.0,
        options.grid,
        blocks,
        stats.tasks,
        stats.log.len(),
        options.out.display()
    );

    if !options.render {
        println!(
            "not rendering (pass --render). The log alone is enough for \
             tools/animate_block_progress.py, and for anything else that reads \
             the schema."
        );
        return Ok(());
    }
    if options.request.view == View::Volume3d
        && !within_volume3d_ceiling(options.grid)
        && !options.request.allow_large
    {
        return Err(Error::InvalidArgument(format!(
            "{blocks} blocks is past the practical ceiling of \
             {VOLUME3D_BLOCK_CEILING} for the three-dimensional view — the \
             render time is per block per frame and this would run for hours. \
             Use a smaller --grid, or --allow-large if you meant it. The log \
             itself is already written and has no such limit."
        )));
    }
    let movie = render(&options.request)?;
    println!("movie: {}", movie.display());
    Ok(())
}
