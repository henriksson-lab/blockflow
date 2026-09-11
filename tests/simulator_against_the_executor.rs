// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The simulator, checked against the executor that it models.**
//
// `simulate` had no consumer in `src/` — only a ranking suite that compares its
// figures against each other. A model with nothing to disagree with drifts from
// the thing it models, and every fidelity item this file was written alongside
// was an instance of that having already happened: a phase reading two images
// charged as if it read one, a chunk grid built from the wrong extent, a store
// costing nothing at all.
//
// # What is compared, and why none of it is a duration
//
// The two do not share a clock and must not be asked to. What they do share is
// **arithmetic on the plan**, and every quantity below is deterministic in both:
//
// | | executor | simulator |
// |---|---|---|
// | admission order | `ExecutionLog::visit_order`, whose doc calls `TaskAdmitted` order *the schedule* | the sequence of picks a `Scheduler` made |
// | tasks | `Stats::tasks` | `Outcome::tasks_run` |
// | chunks fetched | `Event::RegionRead::chunks`, summed | `Outcome::cache_misses + cache_hits` |
// | bytes stored | `Event::RegionWritten::bytes`, summed | `Outcome::written_bytes + materialised_bytes` |
//
// A disagreement in any of them is a disagreement about what the run *does*,
// which is exactly the class of defect a ranking suite cannot see: both arms of
// a comparison are wrong in the same direction and the ranking survives.
//
// # Why the simulator's order is read through a `Scheduler`
//
// `Outcome` is `Copy` and carries scalars; threading a trace through it would
// make every caller pay for one test. A scheduler already sees every choice as
// it is made and is the supported way to observe one, so `Recording` wraps
// `ExecutorOrder` — the scheduler that shares `strategy::priority_key` with the
// real dispatcher — and writes down what it picked. That the two orders agree is
// therefore *nearly* a tautology, and deliberately so: `priority_key` is shared
// precisely to make it one. What the assertion catches is the loop around it,
// which is not shared — readiness, barriers, and which phase a task belongs to.

use std::cell::RefCell;
use std::rc::Rc;

use blockflow::assemble::{Assembly, PlanBuilder};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::log::Event;
use blockflow::op::Chain;
use blockflow::probes::IdentityOp;
use blockflow::simulate::{Decision, ExecutorOrder, Machine, Rates, Run, Scheduler};
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

/// The chunk shape both halves are told about, so that "chunks touched" is one
/// question. The executor learns it from the environment, the simulator from
/// `Rates`, and a test that let them differ would be comparing two lattices.
const CHUNK: [usize; 3] = [8, 8, 8];

/// A scheduler that records what it picked, delegating the decision itself.
struct Recording {
    inner: Box<dyn Scheduler>,
    picked: Rc<RefCell<Vec<usize>>>,
}

impl Scheduler for Recording {
    fn name(&self) -> &'static str {
        "recording"
    }

    fn pick(&mut self, decision: &Decision<'_>) -> usize {
        let slot = self.inner.pick(decision);
        self.picked.borrow_mut().push(decision.ready[slot]);
        slot
    }
}

/// What both halves said about one plan.
struct Both {
    /// `(phase, block index)` in the order each admitted them.
    executor_order: Vec<(usize, [usize; 3])>,
    simulator_order: Vec<(usize, [usize; 3])>,
    executor_tasks: usize,
    simulator_tasks: u64,
    executor_chunks_read: u64,
    simulator_chunks_read: u64,
    executor_bytes_written: u64,
    simulator_bytes_written: u64,
    simulator_prefetched_bytes: u64,
    simulator_duplicated_fetches: u64,
}

fn run_both(assembly: Assembly, volume: [usize; 3]) -> Both {
    run_both_with(
        assembly,
        volume,
        Hints {
            concurrency: 1,
            ..Hints::default()
        },
        Machine {
            workers: 1,
            cache_bytes: 0,
            prefetch_depth: 0,
            ..Machine::default()
        },
        Rates::default(),
        || Box::new(ExecutorOrder::phase_major()),
    )
}

fn run_both_with(
    assembly: Assembly,
    volume: [usize; 3],
    hints: Hints,
    machine: Machine,
    rates: Rates,
    make_scheduler: impl FnOnce() -> Box<dyn Scheduler>,
) -> Both {
    let plan = &assembly.decomposition;

    // --- the executor ---------------------------------------------------
    let input = Voxels::F64(ndarray::Array3::from_elem(volume, 1.0));
    let env = ArrayEnvironment::for_decomposition(input, plan, CHUNK).expect("an environment");
    let workflow = &assembly.workflow;
    // **`execute_phases` and not `execute`.** `execute` hands every phase
    // `PhaseWork::Pixels` — so a fragment phase reached through it is never
    // applied, and the block is read and written as if the phase were a chain.
    // The first version of this file used `execute`, and the "divergence" it
    // then reported on the fragment fixture was that mistake and not the
    // executor's.
    let stats = execute_phases(
        "differential",
        workflow,
        plan,
        &hints,
        &env,
        &[],
        &assembly.work(),
    )
    .expect("a run");

    let mut executor_order = Vec::new();
    let mut executor_chunks_read = 0u64;
    let mut executor_bytes_written = 0u64;
    for event in stats.log.events() {
        match event {
            Event::TaskAdmitted { phase, index } => executor_order.push((phase, index)),
            Event::RegionRead { chunks, .. } => executor_chunks_read += chunks,
            Event::RegionWritten { bytes, .. } => executor_bytes_written += bytes,
            _ => {}
        }
    }

    // --- the simulator --------------------------------------------------
    let picked = Rc::new(RefCell::new(Vec::new()));
    let mut scheduler = Recording {
        inner: make_scheduler(),
        picked: picked.clone(),
    };
    let outcome = Run::new(plan, &assembly.work())
        // The cache is irrelevant to the demand count compared here: the
        // executor's `RegionRead::chunks` counts chunks a demand fetch touches,
        // and the simulator's hit/miss split is reduced back to that below.
        // Prefetch fills are a separate simulator counter and deliberately not
        // part of the executor-facing demand read comparison.
        .machine(machine)
        .rates(Rates {
            chunk: CHUNK,
            chunk_bytes: (CHUNK.iter().product::<usize>() * 8) as u64,
            ..rates
        })
        .go(&mut scheduler)
        .expect("a simulable plan");

    let graph = blockflow::graph::TaskGraph::build(plan);
    let simulator_order = picked
        .borrow()
        .iter()
        .map(|&id| {
            let task = &graph.tasks[id];
            (task.phase, task.index)
        })
        .collect();

    let simulated_chunks_touched = outcome.cache_misses + outcome.cache_hits;
    let prefetched_chunks = outcome.prefetched_bytes / (CHUNK.iter().product::<usize>() * 8) as u64;

    Both {
        executor_order,
        simulator_order,
        executor_tasks: stats.tasks,
        simulator_tasks: outcome.tasks_run,
        executor_chunks_read,
        simulator_chunks_read: simulated_chunks_touched.saturating_sub(prefetched_chunks),
        executor_bytes_written,
        simulator_bytes_written: outcome.written_bytes + outcome.materialised_bytes,
        simulator_prefetched_bytes: outcome.prefetched_bytes,
        simulator_duplicated_fetches: outcome.duplicated_fetches,
    }
}

fn assert_stores_agree(what: &str, both: &Both) {
    assert_eq!(
        both.simulator_bytes_written, both.executor_bytes_written,
        "{what}: the simulator stored {} bytes against the executor's {}. Both are \
         `valid.voxels() x dtype_at(phase + 1)` summed over blocks, so a disagreement means one \
         of them is writing a different extent or a different element type.",
        both.simulator_bytes_written, both.executor_bytes_written
    );
}

fn assert_agrees(what: &str, both: &Both) {
    assert_counts_agree(what, both);
    assert_eq!(
        both.simulator_order, both.executor_order,
        "{what}: the two admitted blocks in different orders. `priority_key` is shared, so the \
         difference is in the loop around it — readiness, a barrier, or which phase a task was \
         thought to belong to."
    );
}

fn assert_counts_agree(what: &str, both: &Both) {
    assert_eq!(
        both.simulator_tasks as usize, both.executor_tasks,
        "{what}: the simulator ran {} tasks against the executor's {}. This is a property of the \
         plan, so a disagreement is about which tasks exist rather than about scheduling.",
        both.simulator_tasks, both.executor_tasks
    );
    assert_eq!(
        both.simulator_chunks_read, both.executor_chunks_read,
        "{what}: the simulator fetched {} chunks against the executor's {}. Both count the \
         chunks a block's fetch region touches, over every image the phase reads, on that \
         image's own grid — the three things the fidelity work had to correct.",
        both.simulator_chunks_read, both.executor_chunks_read
    );
}

/// The plain case: three pixel phases, one lattice, a halo on each.
#[test]
fn an_all_pixel_chain_agrees() {
    let volume = [16, 16, 16];
    let grid = BlockGrid::new(volume, [8, 8, 8]).expect("a grid");
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    for name in ["first", "second", "third"] {
        builder
            .pixels(Chain::op(IdentityOp::new(name, [1, 1, 1])))
            .expect("a pixel phase");
    }
    let both = run_both(builder.finish().expect("an assembly"), volume);
    assert_agrees("an all-pixel chain", &both);
    assert_stores_agree("an all-pixel chain", &both);
}

/// A reach of zero, so a block's fetch region is its core and the chunk count is
/// the smallest it can be — the arm where an off-by-one in the fetch extent has
/// nowhere to hide.
#[test]
fn a_reachless_chain_agrees() {
    let volume = [16, 16, 16];
    let grid = BlockGrid::new(volume, [8, 8, 8]).expect("a grid");
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    for name in ["first", "second"] {
        builder
            .pixels(Chain::op(IdentityOp::new(name, [0, 0, 0])))
            .expect("a pixel phase");
    }
    let both = run_both(builder.finish().expect("an assembly"), volume);
    assert_agrees("a reachless chain", &both);
    assert_stores_agree("a reachless chain", &both);
}

/// The single-worker differential above checks the exact admission sequence.
/// With several workers, prefetch, contention or a handout scheduler, the order
/// is intentionally a scheduling question rather than a conservation law. The
/// executor and simulator still have to agree on the plan arithmetic: how many
/// tasks exist, how many chunks demand reads touch, and how many bytes the plan
/// stores.
#[test]
fn machine_terms_and_handout_scheduling_preserve_executor_arithmetic() {
    use blockflow::distributed::handout::HandoutPolicy;
    use blockflow::simulate::Handout;

    enum SimScheduler {
        PhaseMajor,
        NearestHandout,
    }

    struct MachineCase {
        name: &'static str,
        hints: Hints,
        machine: Machine,
        rates: Rates,
        scheduler: SimScheduler,
    }

    impl SimScheduler {
        fn make(&self) -> Box<dyn Scheduler> {
            match self {
                Self::PhaseMajor => Box::new(ExecutorOrder::phase_major()),
                Self::NearestHandout => Box::new(Handout::new(HandoutPolicy::NearestFirst)),
            }
        }
    }

    fn assembly(volume: [usize; 3]) -> Assembly {
        let grid = BlockGrid::new(volume, [8, 8, 8]).expect("a grid");
        let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
        for name in ["wide", "middle", "narrow"] {
            builder
                .pixels(Chain::op(IdentityOp::new(name, [1, 1, 1])))
                .expect("a pixel phase");
        }
        builder.finish().expect("an assembly")
    }

    let volume = [32, 32, 32];
    let cases = [
        MachineCase {
            name: "cache and prefetch",
            hints: Hints {
                concurrency: 4,
                prefetch_depth: 2,
                ..Hints::default()
            },
            machine: Machine {
                workers: 4,
                cache_bytes: 1 << 20,
                prefetch_depth: 2,
                io_channels: 8,
                ..Machine::default()
            },
            rates: Rates {
                compute_ns_per_voxel: 1_000.0,
                ..Rates::default()
            },
            scheduler: SimScheduler::PhaseMajor,
        },
        MachineCase {
            name: "wave dispatch and contention",
            hints: Hints {
                concurrency: 4,
                ..Hints::default()
            },
            machine: Machine {
                workers: 4,
                wave_synchronous: true,
                contention: blockflow::simulate::MEASURED_CONTENTION,
                ..Machine::default()
            },
            rates: Rates::default(),
            scheduler: SimScheduler::PhaseMajor,
        },
        MachineCase {
            name: "distributed handout",
            hints: Hints {
                concurrency: 6,
                ..Hints::default()
            },
            machine: Machine {
                nodes: 2,
                workers: 6,
                cache_bytes: 1 << 18,
                cache_shared: false,
                candidate_window: 24,
                ..Machine::default()
            },
            rates: Rates::default(),
            scheduler: SimScheduler::NearestHandout,
        },
    ];

    let mut saw_prefetch = false;
    let mut saw_duplicated_fetch = false;
    for case in cases {
        let both = run_both_with(
            assembly(volume),
            volume,
            case.hints,
            case.machine,
            case.rates,
            || case.scheduler.make(),
        );
        assert_counts_agree(case.name, &both);
        assert_stores_agree(case.name, &both);
        saw_prefetch |= both.simulator_prefetched_bytes > 0;
        saw_duplicated_fetch |= both.simulator_duplicated_fetches > 0;
    }
    assert!(
        saw_prefetch,
        "the prefetch arm never issued a prefetch, so it did not exercise the machine term"
    );
    assert!(
        saw_duplicated_fetch,
        "the distributed handout arm never duplicated a fetch, so it did not exercise the \
         multi-cache machine term"
    );
}

/// A **dtype change**, so that the images are not all `f64` and the per-image
/// byte arithmetic has something to get wrong. Both halves size a fetch as
/// `voxels x dtype_at(image)`, and an implementation that folded the input's
/// type over every image would agree with itself and disagree here.
#[test]
fn a_chain_that_changes_element_type_agrees() {
    let volume = [16, 16, 16];
    let grid = BlockGrid::new(volume, [8, 8, 8]).expect("a grid");
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    builder
        .pixels(Chain::op(
            blockflow::ops::voxelwise::VoxelwiseMaskOp::threshold("threshold", 0.5),
        ))
        .expect("a thresholding phase");
    builder
        .pixels(Chain::op(IdentityOp::new("after", [1, 1, 1])))
        .expect("a pixel phase");
    let both = run_both(builder.finish().expect("an assembly"), volume);
    assert_agrees("a chain that changes element type", &both);
    assert_stores_agree("a chain that changes element type", &both);
}

/// A **fragment phase**, which writes a sidecar rather than an image.
///
/// It is here for the four quantities the table above compares, and because a
/// fragment phase is where `writes_an_image` is asked of the work rather than
/// assumed — a branch no all-pixel plan exercises. The **sidecar** payload is
/// deliberately outside the comparison: the simulator keeps it in
/// `Outcome::sidecar_bytes_written` and the executor's `RegionWritten` events
/// never carry it, so the two are not counting the same thing and agreeing
/// about it would mean nothing.
#[test]
fn a_plan_with_a_fragment_phase_agrees() {
    let volume = [16, 16, 16];
    let grid = BlockGrid::new(volume, [8, 8, 8]).expect("a grid");
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    builder
        .pixels(Chain::op(IdentityOp::new("before", [1, 1, 1])))
        .expect("a pixel phase");
    builder
        .fragments(blockflow::probes::BlockSummaryOp::new(
            "summary",
            "summary",
            blockflow::sidecar::Lifecycle::DeleteOnExit,
        ))
        .expect("a fragment phase");
    let both = run_both(builder.finish().expect("an assembly"), volume);
    assert_agrees("a plan with a fragment phase", &both);

    // The stores agree too, now that the fragment op is actually applied. The
    // simulator additionally accounts the phase's **sidecar** payload, which is
    // a separate counter precisely so that this comparison stays about images.
    assert_stores_agree("a plan with a fragment phase", &both);
    assert!(
        both.simulator_bytes_written > 0,
        "the pixel phase before the fragment one stores an image, so a zero here would mean \
         the fixture stopped exercising the comparison"
    );
}
