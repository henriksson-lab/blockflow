// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Watch a block schedule in a browser — a recorded one, or one happening now.
//
//     block_gui run.json                 # replay an exported order log
//     block_gui --demo                   # run a synthetic workflow and watch it live
//     block_gui run.json --bind 127.0.0.1:9000
//
// `--demo` is not only a toy. It is the shortest complete example of attaching
// the server to a run — five lines, marked below — and somebody wiring this
// into their own program should read that function rather than the module docs.
//
// On a compute node
// -----------------
// Start it there, bound to loopback, and forward the port:
//
//     node$   block_gui --demo --bind 127.0.0.1:8731
//     laptop$ ssh -N -L 8731:127.0.0.1:8731 user@node
//     laptop$ open http://127.0.0.1:8731/
//
// The default bind is already loopback, so the flag above is only there to show
// the shape. A non-loopback bind needs `--allow-public`; read what that flag
// says before using it.

use std::sync::Arc;

use blockflow::decomposition::{groups_for, summarise_slots, Decomposition, PhaseDecomposition};
use blockflow::env::AccountingEnvironment;
use blockflow::export::ExportMeta;
use blockflow::geometry::BlockGrid;
use blockflow::gui::{serve, LiveSource, Options, ReplaySource};
use blockflow::probes::IdentityOp;
use blockflow::strategy::{execute_observed, Hints, SchedulePriority, Workflow};
use blockflow::{Chain, Dtype, Error, Result};

const USAGE: &str = "\
block_gui — watch a block schedule in a browser.

  block_gui <order-log.json>     replay an exported order log
  block_gui --demo               run a synthetic workflow and watch it live

Options:
  --bind ADDR        address to listen on (default 127.0.0.1:8731). Port 0 asks
                     the system for a free one.
  --allow-public     permit a bind that is not loopback. THIS PUBLISHES THE RUN:
                     there is no authentication, so anyone who can reach this
                     machine on the network can watch it, and on a shared
                     compute node that is everyone. The usual answer is to keep
                     the default and forward the port over SSH instead:
                         ssh -N -L 8731:127.0.0.1:8731 <user>@<node>
  --assets DIR       where the built browser files are (default: webui/dist
                     beside the crate, or $BLOCKFLOW_GUI_ASSETS)
  --blocks N         --demo only: blocks per side (default 8)
  --workers N        --demo only: concurrent tasks (default 4)
  --phase-major      --demo only: finish each phase before starting the next
  -h, --help         this
";

struct Args {
    log: Option<String>,
    demo: bool,
    bind: String,
    allow_public: bool,
    assets: Option<String>,
    blocks: usize,
    workers: usize,
    phase_major: bool,
}

fn parse() -> std::result::Result<Args, String> {
    let mut args = Args {
        log: None,
        demo: false,
        bind: format!("127.0.0.1:{}", blockflow::gui::DEFAULT_PORT),
        allow_public: false,
        assets: None,
        blocks: 8,
        workers: 4,
        phase_major: false,
    };
    let mut rest = std::env::args().skip(1);
    while let Some(arg) = rest.next() {
        let mut value = |name: &str| -> std::result::Result<String, String> {
            rest.next().ok_or_else(|| format!("{name} needs a value"))
        };
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--demo" => args.demo = true,
            "--allow-public" => args.allow_public = true,
            "--phase-major" => args.phase_major = true,
            "--bind" => args.bind = value("--bind")?,
            "--assets" => args.assets = Some(value("--assets")?),
            "--blocks" => {
                args.blocks = value("--blocks")?
                    .parse()
                    .map_err(|_| "--blocks must be a number".to_string())?
            }
            "--workers" => {
                args.workers = value("--workers")?
                    .parse()
                    .map_err(|_| "--workers must be a number".to_string())?
            }
            other if other.starts_with('-') => return Err(format!("unknown option {other}")),
            other => args.log = Some(other.to_string()),
        }
    }
    if args.demo == args.log.is_some() {
        return Err("give exactly one of an order-log path or --demo".to_string());
    }
    Ok(args)
}

fn options(args: &Args) -> Result<Options> {
    let bind = args
        .bind
        .parse()
        .map_err(|err| Error::invalid(format!("--bind {}: {err}", args.bind)))?;
    let mut options = Options::default().bind(bind);
    options.allow_public = args.allow_public;
    if let Some(assets) = &args.assets {
        options = options.assets(assets);
    }
    Ok(options)
}

fn main() {
    let args = match parse() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("block_gui: {message}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(&args) {
        eprintln!("block_gui: {error}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<()> {
    match &args.log {
        Some(path) => replay(path, args),
        None => demo(args),
    }
}

fn replay(path: &str, args: &Args) -> Result<()> {
    let source = Arc::new(ReplaySource::from_path(path)?);
    let log = source.log();
    println!(
        "replaying {path}: strategy {}, grid {:?}, {} phases, {} events{}",
        log.strategy,
        log.grid,
        log.phases,
        log.events.len(),
        if log.unknown > 0 {
            format!(" ({} of an unfamiliar type, skipped)", log.unknown)
        } else {
            String::new()
        }
    );
    let server = serve(source, options(args)?)?;
    println!("open {}", server.url());
    println!("ctrl-c to stop");
    park();
    Ok(())
}

/// A live run with the server attached. **This is the whole integration**; the
/// five marked lines are what a real program adds.
fn demo(args: &Args) -> Result<()> {
    let side = args.blocks.max(1);
    // Said before it happens, because planning a large grid takes a visible
    // moment and the server does not exist until it is done — a browser pointed
    // at the port before this finishes gets connection refused, which looks like
    // a broken tool rather than a busy one.
    println!("planning a {side}x{side} grid...");
    let shape = [64, side * 64, side * 64];
    let chain = Chain::sequence(vec![
        Chain::op(IdentityOp::new("smooth", [0, 2, 2]).with_cost(1.0)),
        Chain::op(IdentityOp::new("background", [0, 6, 6]).with_cost(2.0)),
        Chain::op(IdentityOp::new("threshold", [0, 0, 0]).with_cost(0.5)),
    ]);
    let ops: Vec<(usize, String)> = chain
        .slots()
        .iter()
        .enumerate()
        .map(|(slot, sub)| (slot, sub.display_name()))
        .collect();
    let workflow = Workflow::new(chain, shape, Dtype::U16);

    let slots = workflow.chain.slots();
    let phases: Vec<PhaseDecomposition> = groups_for(0b11, slots.len())
        .iter()
        .map(|group| {
            let (reach, _, names, _) =
                summarise_slots(&slots, group, shape).expect("the demo chain's slots summarise");
            let grid = BlockGrid::along(shape, &[1, 2], 64).unwrap();
            PhaseDecomposition::derive(group.clone(), names, reach.clone(), reach, grid)
        })
        .collect();
    let decomposition = Decomposition {
        volume: shape,
        dtype: workflow.dtype,
        phases,
        chain_reach: workflow.chain.reach3(&shape),
    };
    decomposition.check()?;
    let grid = decomposition.phases[0].grid.blocks_per_axis();

    // ---- the integration, in full ----------------------------------------
    let meta = ExportMeta::new("live", shape, decomposition.n_phases()).with_ops(ops);
    let live = Arc::new(LiveSource::new(meta, grid).with_timeline());
    let server = serve(live.clone(), options(args)?)?;
    let listeners = live.listeners();
    // ... execute_observed(..., &listeners) ... then live.finished();
    // ----------------------------------------------------------------------

    println!("open {}", server.url());
    println!(
        "grid {grid:?} = {} blocks, {} phases, {} workers",
        grid[0] * grid[1] * grid[2],
        decomposition.n_phases(),
        args.workers
    );
    println!("the run starts now; reload if the page was open before it");

    let hints = Hints {
        priority: if args.phase_major {
            SchedulePriority::PhaseMajor
        } else {
            SchedulePriority::BlockMajor
        },
        concurrency: args.workers.max(1),
        ..Hints::default()
    };
    // The accounting environment allocates no arrays, so the grid can be as
    // large as you like and the run takes as long as the schedule says rather
    // than as long as the memory bandwidth does. It is a schedule to watch, not
    // a computation.
    let env = AccountingEnvironment::new(shape, [32, 64, 64], 2);
    let stats = execute_observed("live", &workflow, &decomposition, &hints, &env, &listeners)?;
    live.finished();

    // `applications` rather than `ops_applied`: this demo drives a chain, where
    // the two are equal, but the figure a reader wants from a line labelled
    // "applications" is "did anything run", and `ops_applied` answers that with
    // zero for a plan whose phases are fragment phases. `blocks` is beside it
    // for the same reason — it is the admitted count, which every kind of phase
    // contributes to.
    println!(
        "done: {} tasks, {} applications over {} blocks, {} events, {} listener faults",
        stats.tasks,
        stats.applications(),
        stats.blocks_admitted,
        stats.log.len(),
        stats.listener_faults
    );
    println!("the view stays up; ctrl-c to stop");
    park();
    Ok(())
}

/// Wait until the process is interrupted. No signal handling: ctrl-c ending the
/// process is the expected way out, and a handler would only add a way for the
/// server to be left running with nothing to serve.
fn park() {
    loop {
        std::thread::park();
    }
}
