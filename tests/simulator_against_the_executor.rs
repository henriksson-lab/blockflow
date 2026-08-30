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
use std::collections::BTreeSet;
use std::rc::Rc;

use blockflow::assemble::{Assembly, PlanBuilder};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::log::Event;
use blockflow::op::Chain;
use blockflow::probes::IdentityOp;
use blockflow::simulate::{simulate, Decision, ExecutorOrder, Machine, PerPhase, Rates, Scheduler};
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

/// The chunk shape both halves are told about, so that "chunks touched" is one
/// question. The executor learns it from the environment, the simulator from
/// `Rates`, and a test that let them differ would be comparing two lattices.
const CHUNK: [usize; 3] = [8, 8, 8];

/// A scheduler that records what it picked, delegating the decision itself.
struct Recording {
    inner: ExecutorOrder,
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
}

fn run_both(assembly: Assembly, volume: [usize; 3]) -> Both {
    let plan = &assembly.decomposition;

    // --- the executor ---------------------------------------------------
    let input = Voxels::F64(ndarray::Array3::from_elem(volume, 1.0));
    let env = ArrayEnvironment::for_decomposition(input, plan, CHUNK).expect("an environment");
    let workflow = &assembly.workflow;
    // One worker, so the admitted order is a sequence rather than an
    // interleaving that only a clock could reproduce.
    let hints = Hints {
        concurrency: 1,
        ..Hints::default()
    };
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
        inner: ExecutorOrder::phase_major(),
        picked: picked.clone(),
    };
    let outcome = simulate(
        plan,
        &assembly.work(),
        // The cache is irrelevant to what is compared: the executor's
        // `RegionRead::chunks` counts the chunks a fetch *touches*, and the
        // simulator's `cache_misses + cache_hits` is the same count with the
        // hit/miss split thrown away. Comparing against misses alone was the
        // first version of this test and it was wrong by 21 of 192 —
        // `ModelledCache::new(0, ..)` has a capacity of **one**, not zero, so
        // "no cache" still hits. The sum has no such assumption in it.
        &Machine {
            workers: 1,
            cache_bytes: 0,
            prefetch_depth: 0,
            ..Machine::default()
        },
        &Rates {
            chunk: CHUNK,
            chunk_bytes: (CHUNK.iter().product::<usize>() * 8) as u64,
            ..Rates::default()
        },
        &BTreeSet::new(),
        &BTreeSet::new(),
        PerPhase::default(),
        &mut scheduler,
    )
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

    Both {
        executor_order,
        simulator_order,
        executor_tasks: stats.tasks,
        simulator_tasks: outcome.tasks_run,
        executor_chunks_read,
        simulator_chunks_read: outcome.cache_misses + outcome.cache_hits,
        executor_bytes_written,
        simulator_bytes_written: outcome.written_bytes + outcome.materialised_bytes,
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
    assert_eq!(
        both.simulator_tasks as usize, both.executor_tasks,
        "{what}: the simulator ran {} tasks against the executor's {}. This is a property of the \
         plan, so a disagreement is about which tasks exist rather than about scheduling.",
        both.simulator_tasks, both.executor_tasks
    );
    assert_eq!(
        both.simulator_order, both.executor_order,
        "{what}: the two admitted blocks in different orders. `priority_key` is shared, so the \
         difference is in the loop around it — readiness, a barrier, or which phase a task was \
         thought to belong to."
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

/// A **fragment phase**, which writes a sidecar rather than an image and is the
/// one phase kind whose bytes neither half prices yet.
///
/// It is here for the other three quantities — the task count, the admitted
/// order and the chunks its blocks read — and because a fragment phase is where
/// `writes_an_image` is asked of the work rather than assumed, which is a branch
/// no all-pixel plan exercises. The sidecar bytes it writes are the subject of
/// the `Sidecar traffic and the barrier gather` item and are deliberately not
/// asserted here: neither side counts them, so agreeing about them would mean
/// nothing.
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
