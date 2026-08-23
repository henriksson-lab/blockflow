// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The barrier and the hoisted reduction, executed.** `docs/design/barriers.md`
// specifies both; this is what runs them.
//
// The shape under test is the one four shipped ops have — `ops::fill`,
// `ops::regional`, `ops::detect`, `ops::label` — and it is a fragment-and-join:
// phase 0 emits one fragment per block, phase 1 needs *every* block's fragment
// before any of its own blocks may run. Before `FragmentOp::barrier` the only
// way to say that was to fetch the whole volume in every block, because a
// dependency in this crate is a region intersection and nothing else; before
// `FragmentOp::reduce` the merge had nowhere to live but `apply`, which is per
// block, so it ran once per block.
//
// Four arms, and what separates them
// ----------------------------------
// One op, `GlobalOffsetOp`, with two booleans. Every other line of the four arms
// is the same line, so a difference in the counters is attributable to the
// declaration and to nothing else:
//
// | arm | `barrier()` | reduction | halo | the merge runs |
// |---|---|---|---|---|
// | **in-plan** — what the framework admitted before this | `false` | in `apply` | the whole volume | once per block |
// | **barrier alone** | `true` | in `apply` | zero | once per block |
// | **barrier, reduction hoisted** | `true` | in `reduce` | zero | once |
// | **out of plan** — the merge between two `execute_phases` calls | n/a | in the caller | zero | once |
//
// What is asserted, and why each part of it
// -----------------------------------------
// * **The same answer, at every lattice, byte for byte.** The four arms and a
//   whole-volume reference agree exactly. That is the acceptance bar and the
//   grids are asserted to be genuinely distinct — a sweep that had quietly
//   decayed to one grid, or to a grid of one block, would pass while meaning
//   nothing.
// * **The counters, not a model.** `EnvCounters` is the framework's own
//   instrument, bumped inside `Environment::read` and inside the sidecar store.
//   Nothing below multiplies a block count by a volume and calls it traffic; the
//   *predictions* are stated and then compared against what was counted.
// * **The merge count.** The CPU half of `barriers.md` §7.2 is not in any byte
//   column: the fold is small per invocation and there are `blocks` invocations.
//   Both arms count their folds, so the multiplier is measured rather than
//   argued.
// * **A liveness control beside each claim.** Every assertion here has a
//   negative arm — the same program with one declaration changed — because an
//   assertion that no mutant fails is not a test.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, BlockBuf, Environment};
use blockflow::fragment::{
    append_fragment_phase, fragment_phase, pack_u64, unpack_u64, BlockOutput, BlockView, Coverage,
    FragmentInput, FragmentOp, FragmentOutput, PhaseView, PhaseWork, SeamFold,
};
use blockflow::geometry::BlockGrid;
use blockflow::graph::TaskGraph;
use blockflow::log::{Event, ExecutionLog};
use blockflow::op::Chain;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, SchedulePriority, Workflow};
use blockflow::Voxels;

const STREAM: &str = "sums";
const WIDE: &str = "wide";

// ------------------------------------------------------------- the volume --

const VOLUME: [usize; 3] = [16, 16, 16];

fn scene() -> Array3<f64> {
    // A field with no symmetry a block boundary could hide: the global sum is a
    // number no arm can guess from one block.
    let mut array = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for (index, value) in array.indexed_iter_mut() {
        let (i, j, k) = index;
        *value = ((i * 7 + j * 13 + k * 3) % 11) as f64 + (i * j * k % 5) as f64;
    }
    array
}

fn total_of(array: &Array3<f64>) -> f64 {
    array.iter().sum()
}

// ----------------------------------------------------------------- phase 0 --

/// `volume -> volume + fragments`: carry the pixels on and emit this block's
/// sum. `ops::label`'s local labelling has this shape and so does `ops::fill`'s
/// first phase.
struct BlockSumOp {
    name: &'static str,
}

impl FragmentOp for BlockSumOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            STREAM.to_string(),
            Lifecycle::Persistent,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput, blockflow::Error> {
        let mut buffer = at.output_buffer(0.0)?;
        let BlockBuf::Array(pixels) = at.pixels()? else {
            return Ok(BlockOutput::fragment(STREAM, pack_u64(&[0])).with_pixels(buffer));
        };
        let values = pixels.view::<f64>()?;
        let sum: f64 = values.iter().sum();
        let BlockBuf::Array(out) = &mut buffer else {
            unreachable!("the environment gave data for the input and none for the output");
        };
        out.view_mut::<f64>()?.assign(&values);
        Ok(BlockOutput::fragment(STREAM, pack_u64(&[sum.to_bits()])).with_pixels(buffer))
    }
}

// ----------------------------------------------------------------- phase 1 --

/// `fragments + volume -> volume`: add the **global** sum to every voxel.
///
/// The answer depends on every block, which is the whole point: a block that ran
/// before its neighbours had written their fragments would add a smaller number
/// and produce a complete, well-formed, wrong volume.
struct GlobalOffsetOp {
    name: &'static str,
    from_phase: usize,
    lattice: [usize; 3],
    barrier: bool,
    hoisted: bool,
    /// How many times the fold over the fragment set ran. The quantity
    /// `barriers.md` §7.2 is about, and it is in no byte column.
    folds: Arc<AtomicUsize>,
}

impl GlobalOffsetOp {
    fn new(name: &'static str, from_phase: usize, lattice: [usize; 3]) -> Self {
        Self {
            name,
            from_phase,
            lattice,
            barrier: false,
            hoisted: false,
            folds: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn with_barrier(mut self, barrier: bool) -> Self {
        self.barrier = barrier;
        self
    }

    fn hoisting(mut self, hoisted: bool) -> Self {
        self.hoisted = hoisted;
        self
    }

    fn folds(&self) -> usize {
        self.folds.load(Ordering::SeqCst)
    }

    /// The fold itself, over whatever fragment set it is handed. One function so
    /// that the hoisted and the per-block arms cannot drift into computing two
    /// different things.
    fn fold(
        &self,
        seen: &mut dyn FnMut(&mut dyn FnMut(&[u8])) -> Result<(), blockflow::Error>,
    ) -> Result<f64, blockflow::Error> {
        let mut total = 0.0f64;
        let mut failed: Option<blockflow::Error> = None;
        seen(&mut |bytes| match unpack_u64(bytes) {
            Ok(words) if words.len() == 1 => total += f64::from_bits(words[0]),
            Ok(words) => {
                failed = Some(blockflow::Error::InvalidArgument(format!(
                    "a block sum is one word; this one is {}",
                    words.len()
                )))
            }
            Err(err) => failed = Some(err),
        })?;
        self.folds.fetch_add(1, Ordering::SeqCst);
        match failed {
            Some(err) => Err(err),
            None => Ok(total),
        }
    }
}

impl FragmentOp for GlobalOffsetOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Nothing this block is authoritative for reaches past its core. The
    /// whole-lattice reach in [`Self::inputs`] is a **fragment** reach.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn barrier(&self) -> bool {
        self.barrier
    }

    fn gathers(&self) -> bool {
        false
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        // **The declaration the hoisting buys.** With the fold in `apply`, every
        // block needs every fragment and says so. With it in `reduce`, the block
        // needs none of them — the phase does — so the reach drops to zero and
        // the set stops being transmitted once per block.
        let reach = if self.hoisted {
            [0, 0, 0]
        } else {
            self.lattice
        };
        vec![FragmentInput::own(STREAM.to_string(), self.from_phase).with_reach(reach)]
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>, blockflow::Error> {
        if !self.hoisted {
            return Ok(Vec::new());
        }
        let total = self.fold(&mut |visit| {
            at.stream_fragments(STREAM, &mut |_, bytes| {
                visit(bytes);
                Ok(())
            })?;
            Ok(())
        })?;
        Ok(pack_u64(&[total.to_bits()]))
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput, blockflow::Error> {
        let total = if self.hoisted {
            let words = unpack_u64(at.reduced)?;
            if words.len() != 1 {
                return Err(blockflow::Error::InvalidArgument(format!(
                    "the phase reduction is one word; this one is {}. An empty reduction is \
                     what a block is handed when the phase has no barrier, which is why the \
                     plan refuses that pair.",
                    words.len()
                )));
            }
            f64::from_bits(words[0])
        } else {
            self.fold(&mut |visit| {
                at.stream_fragments(STREAM, &mut |_, bytes| {
                    visit(bytes);
                    Ok(())
                })?;
                Ok(())
            })?
        };

        let mut buffer = at.output_buffer(0.0)?;
        let BlockBuf::Array(pixels) = at.pixels()? else {
            return Ok(BlockOutput::nothing().with_pixels(buffer));
        };
        let values = pixels.view::<f64>()?;
        let BlockBuf::Array(out) = &mut buffer else {
            unreachable!("the environment gave data for the input and none for the output");
        };
        let mut view = out.view_mut::<f64>()?;
        // **Only the core**, because with the fold in `apply` the fragment reach
        // is also the halo and the buffer holds the whole volume.
        let offset = [
            at.core.start[0] - at.read.start[0],
            at.core.start[1] - at.read.start[1],
            at.core.start[2] - at.read.start[2],
        ];
        let extent = at.core.shape.clone();
        let window = ndarray::s![
            offset[0]..offset[0] + extent[0],
            offset[1]..offset[1] + extent[1],
            offset[2]..offset[2] + extent[2],
        ];
        let mut target = view.slice_mut(window);
        target.assign(&values.slice(window));
        target.mapv_inplace(|value| value + total);
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

// ------------------------------------------------------------- the harness --

fn hints(priority: SchedulePriority, concurrency: usize) -> Hints {
    Hints {
        priority,
        concurrency,
        ..Hints::default()
    }
}

fn empty_workflow() -> Workflow {
    Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::F64)
}

/// What one arm moved, from the framework's own counters.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Traffic {
    read_bytes: u64,
    write_bytes: u64,
    fragment_bytes: u64,
}

impl Traffic {
    fn of(env: &dyn Environment) -> Self {
        let counters = env.counters();
        Self {
            read_bytes: counters.read_bytes.load(Ordering::Relaxed),
            write_bytes: counters.write_bytes.load(Ordering::Relaxed),
            fragment_bytes: counters.sidecar_bytes_written.load(Ordering::Relaxed)
                + counters.sidecar_bytes_read.load(Ordering::Relaxed),
        }
    }

    fn total(self) -> u64 {
        self.read_bytes + self.write_bytes + self.fragment_bytes
    }
}

struct Run {
    output: Array3<f64>,
    traffic: Traffic,
    folds: usize,
    halo: [usize; 3],
    /// Edges from the barrier phase down to the phase below it.
    edges: usize,
    /// What the graph records about the phase, which is the whole declaration.
    barrier_recorded: bool,
    log: Arc<ExecutionLog>,
}

/// The two-phase plan, and the one run of it. `barrier` and `hoisted` are the
/// only things that vary between arms.
fn run_in_plan(
    block: [usize; 3],
    barrier: bool,
    hoisted: bool,
    priority: SchedulePriority,
    concurrency: usize,
) -> Run {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let lattice = grid.blocks_per_axis();
    let summarise = BlockSumOp { name: "sum" };
    let offset = GlobalOffsetOp::new("offset", 0, lattice)
        .with_barrier(barrier)
        .hoisting(hoisted);

    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![fragment_phase(&summarise, grid).expect("phase 0")],
        chain_reach: [0, 0, 0],
    };
    plan = append_fragment_phase(plan, &offset).expect("phase 1");
    plan.check().expect("the plan tiles");

    // Per side, per axis: the quantity `fragment_phase` sets and
    // `barriers.md` §3.2.3 is about.
    let granted = plan.phases[1].halo.in_voxels(plan.phases[1].grid.block());
    let halo = [
        granted.axis(0).bound(VOLUME[0]).0,
        granted.axis(1).bound(VOLUME[1]).0,
        granted.axis(2).bound(VOLUME[2]).0,
    ];
    let graph = TaskGraph::build(&plan);
    let edges: usize = graph
        .tasks_in_phase(1)
        .iter()
        .map(|task| task.deps.len())
        .sum();
    let barrier_recorded = graph.is_barrier(1);

    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(scene()),
        &plan,
        [VOLUME[0], VOLUME[1], VOLUME[2]],
    )
    .expect("environment");
    let log = Arc::new(ExecutionLog::new());
    execute_phases(
        "barrier-arm",
        &empty_workflow(),
        &plan,
        &hints(priority, concurrency),
        &env,
        &[log.clone()],
        &[
            PhaseWork::Fragments(&summarise),
            PhaseWork::Fragments(&offset),
        ],
    )
    .expect("a run");

    let traffic = Traffic::of(&env);
    let output = env.output().view::<f64>().expect("f64 out").to_owned();
    Run {
        output,
        traffic,
        folds: offset.folds(),
        halo,
        edges,
        barrier_recorded,
        log,
    }
}

/// The reference: one block, no seams, nothing to reconcile.
fn reference() -> Array3<f64> {
    let scene = scene();
    let total = total_of(&scene);
    scene.mapv(|value| value + total)
}

/// Lattices that are genuinely distinct, asserted rather than assumed.
///
/// The decay this guards against is real and has happened on this project: a
/// sweep that shrank to two grids, one of them a single block, kept passing and
/// had stopped meaning anything.
fn grids() -> Vec<[usize; 3]> {
    let blocks = [[16, 16, 16], [8, 16, 16], [8, 8, 16], [4, 4, 8]];
    let counts: Vec<usize> = blocks
        .iter()
        .map(|block| {
            BlockGrid::new(VOLUME, *block)
                .expect("a lattice")
                .n_blocks()
        })
        .collect();
    assert_eq!(counts, vec![1, 2, 4, 32], "the sweep's lattices moved");
    let mut distinct = counts.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), counts.len(), "two grids are the same grid");
    assert!(
        counts.iter().filter(|&&n| n > 1).count() >= 3,
        "a sweep of single-block lattices asserts nothing about decomposition"
    );
    blocks.to_vec()
}

// ------------------------------------------------------------ the answers --

/// **The acceptance bar.** Every arm, at every lattice, byte for byte against a
/// whole-volume reference.
#[test]
fn every_arm_agrees_with_the_whole_volume_reference_at_every_lattice() {
    let want = reference();
    for block in grids() {
        for (barrier, hoisted, arm) in [
            (false, false, "in-plan"),
            (true, false, "barrier alone"),
            (true, true, "barrier, hoisted"),
        ] {
            let run = run_in_plan(block, barrier, hoisted, SchedulePriority::BlockMajor, 4);
            assert_eq!(
                run.output, want,
                "arm {arm:?} disagreed with the reference at block {block:?}"
            );
        }
    }
}

/// **The halo, which is where the pixel amplification lives.**
///
/// Without a barrier the whole-lattice fragment reach forces a whole-volume
/// halo, so every block fetches the whole image and the read traffic is the
/// block count times the volume. With one, `halo = reach = 0`.
///
/// The liveness control is the first row of the table: the same op, the same
/// plan, the same everything, with `barrier()` answering `false`.
#[test]
fn a_barrier_relieves_the_halo_and_the_reads_that_follow_it() {
    let voxels = (VOLUME[0] * VOLUME[1] * VOLUME[2]) as u64;
    for block in grids() {
        let blocks = BlockGrid::new(VOLUME, block).expect("a lattice").n_blocks() as u64;
        let without = run_in_plan(block, false, false, SchedulePriority::PhaseMajor, 1);
        let with = run_in_plan(block, true, false, SchedulePriority::PhaseMajor, 1);

        if blocks > 1 {
            assert_eq!(
                without.halo,
                [VOLUME[0], VOLUME[1], VOLUME[2]],
                "without a barrier the fragment reach must still force a whole-volume halo \
                 at block {block:?}"
            );
        }
        assert_eq!(
            with.halo,
            [0, 0, 0],
            "a barrier phase's halo is its own reach, which is zero here"
        );

        // Predicted, then compared: phase 0 reads the volume once, and phase 1
        // reads it `blocks` times without a barrier and once with.
        let predicted_without = (1 + blocks) * voxels * 8;
        let predicted_with = 2 * voxels * 8;
        assert_eq!(
            without.traffic.read_bytes, predicted_without,
            "the in-plan arm's reads did not decompose at block {block:?}"
        );
        assert_eq!(
            with.traffic.read_bytes, predicted_with,
            "the barrier arm's reads did not decompose at block {block:?}"
        );
    }
}

/// **A barrier is a phase-level fact, and the edges stay a sum.**
///
/// This asserted the opposite until the cost of the opposite was measured. A
/// barrier phase's tasks used to depend on *every* task of the phase below —
/// `blocks x blocks` edges, correct, and free for both schedulers because their
/// indegree machinery enforced it with no new code. What that costs is in
/// [`a_barrier_at_a_large_block_count_is_priced`], and it is a product where
/// every other edge in the graph is a sum, at exactly the block counts a barrier
/// exists for.
///
/// So the edges are now the ordinary region-derived ones — with the halo
/// relieved, each block's own core — and the barrier is one bool per phase on
/// `TaskGraph::barriers` that every scheduler gates on. What that costs is that
/// a scheduler *has* to gate; `no_block_of_a_barrier_phase_starts_before_the_phase_below_has_finished`
/// and `the_coordinator_gates_a_barrier_phase_on_the_phase_below` are the two
/// that show the two schedulers do.
#[test]
fn a_barrier_is_a_phase_level_fact_and_the_edges_stay_a_sum() {
    for block in grids() {
        let blocks = BlockGrid::new(VOLUME, block).expect("a lattice").n_blocks();
        let with = run_in_plan(block, true, true, SchedulePriority::PhaseMajor, 1);
        assert_eq!(
            with.edges, blocks,
            "one edge per task — its own core — at {blocks} block(s)"
        );
        assert!(
            with.barrier_recorded,
            "and the barrier is recorded on the graph"
        );

        // The liveness control, and it is the same plan with `barrier()`
        // answering `false`: the halo comes back, so every block fetches the
        // whole volume and the edges *are* the product — by the ordinary region
        // rule, which is `barriers.md` §1.2's coupling seen from the other side.
        // The two arms therefore reach opposite edge counts for one reason, and
        // the halo is asserted beside them so it is visible which.
        let without = run_in_plan(block, false, false, SchedulePriority::PhaseMajor, 1);
        assert_eq!(without.edges, blocks * blocks);
        assert!(!without.barrier_recorded);
        if blocks > 1 {
            assert_ne!(without.halo, with.halo);
            assert!(
                without.edges > with.edges,
                "at {blocks} block(s) the two arms should differ in edges"
            );
        }
    }
}

/// **The CPU half, which is in no byte column.**
///
/// `barriers.md` §7.2: the fold is small per invocation and there are `blocks`
/// invocations, and nobody had multiplied. Hoisted, it runs once.
#[test]
fn hoisting_runs_the_reduction_once_instead_of_once_per_block() {
    for block in grids() {
        let blocks = BlockGrid::new(VOLUME, block).expect("a lattice").n_blocks();
        let per_block = run_in_plan(block, true, false, SchedulePriority::PhaseMajor, 1);
        let hoisted = run_in_plan(block, true, true, SchedulePriority::PhaseMajor, 1);
        assert_eq!(
            per_block.folds, blocks,
            "the per-block arm folded the fragment set {} time(s) at {blocks} block(s)",
            per_block.folds
        );
        assert_eq!(
            hoisted.folds, 1,
            "the hoisted arm folded the fragment set {} time(s) at {blocks} block(s)",
            hoisted.folds
        );
    }
}

/// **The fragment traffic, and the multiplier hoisting removes.**
///
/// Per-block, the set is transmitted `1 + blocks` times — written once, read
/// once per block. Hoisted, exactly twice, at every lattice.
#[test]
fn hoisting_transmits_the_fragment_set_twice_instead_of_once_per_block() {
    for block in grids() {
        let blocks = BlockGrid::new(VOLUME, block).expect("a lattice").n_blocks() as u64;
        let per_block = run_in_plan(block, true, false, SchedulePriority::PhaseMajor, 1);
        let hoisted = run_in_plan(block, true, true, SchedulePriority::PhaseMajor, 1);
        // One fragment per block, one 8-byte word each: `F` is `blocks * 8`.
        let set = blocks * 8;
        assert_eq!(
            per_block.traffic.fragment_bytes,
            (1 + blocks) * set,
            "the per-block arm's fragment traffic did not decompose at {blocks} block(s)"
        );
        assert_eq!(
            hoisted.traffic.fragment_bytes,
            2 * set,
            "the hoisted arm transmits the set twice — written once, read once"
        );
    }
}

/// **The whole thing, priced.** `barriers.md` §7.1's table, at this scale.
///
/// What transfers from here to that table and what does not, stated so the two
/// are not confused. The **pixel** half transfers exactly: it is `1 + blocks`
/// transmissions of the image without a barrier and two with, which is a
/// property of the halo and not of the data, and it is asserted term by term
/// here and predicted from the same formula there. The **fragment** half does
/// not transfer in absolute terms: a fragment is eight bytes here and a block
/// *face* there, so `F` is a rounding error at this scale and 33 % of the total
/// at that one. What does transfer is its multiplier — `1 + blocks` against 2 —
/// which is what hoisting removes and which is asserted separately.
#[test]
fn the_four_arms_are_measured_against_each_other() {
    let voxels = (VOLUME[0] * VOLUME[1] * VOLUME[2]) as u64;
    eprintln!(
        "{:>6} | {:<18} | {:>12} | {:>8} | {:>10} | {:>12} | {:>6} | {:>6}",
        "blocks", "arm", "read", "write", "fragments", "total", "folds", "edges"
    );
    for block in [[16, 16, 16], [8, 8, 16], [4, 4, 8], [2, 2, 4], [2, 2, 2]] {
        let blocks = BlockGrid::new(VOLUME, block).expect("a lattice").n_blocks();
        let mut totals = Vec::new();
        for (name, barrier, hoisted) in [
            ("in-plan", false, false),
            ("barrier alone", true, false),
            ("barrier, hoisted", true, true),
        ] {
            let run = run_in_plan(block, barrier, hoisted, SchedulePriority::PhaseMajor, 1);
            eprintln!(
                "{blocks:>6} | {name:<18} | {:>12} | {:>8} | {:>10} | {:>12} | {:>6} | {:>6}",
                run.traffic.read_bytes,
                run.traffic.write_bytes,
                run.traffic.fragment_bytes,
                run.traffic.total(),
                run.folds,
                run.edges,
            );
            totals.push(run);
        }
        let in_plan = totals[0].traffic.total();
        let alone = totals[1].traffic.total();
        let hoisted = totals[2].traffic.total();
        eprintln!(
            "{blocks:>6} | {:<18} | in-plan {:.2}x, barrier alone {:.2}x, hoisted 1.00x",
            "ratios",
            in_plan as f64 / hoisted as f64,
            alone as f64 / hoisted as f64,
        );

        // Predicted, then compared, so that the table above is a decomposition
        // rather than a reading.
        let n = blocks as u64;
        assert_eq!(totals[0].traffic.read_bytes, (1 + n) * voxels * 8);
        assert_eq!(totals[1].traffic.read_bytes, 2 * voxels * 8);
        assert_eq!(totals[2].traffic.read_bytes, 2 * voxels * 8);
        assert_eq!(totals[0].traffic.fragment_bytes, (1 + n) * n * 8);
        assert_eq!(totals[1].traffic.fragment_bytes, (1 + n) * n * 8);
        assert_eq!(totals[2].traffic.fragment_bytes, 2 * n * 8);
        assert_eq!(totals[0].folds, blocks);
        assert_eq!(totals[1].folds, blocks);
        assert_eq!(totals[2].folds, 1);
        if blocks > 1 {
            assert!(
                in_plan > alone && alone > hoisted,
                "the three arms should be strictly ordered at {blocks} block(s); \
                 got {in_plan}, {alone}, {hoisted}"
            );
        }
    }
}

/// **What a barrier costs the graph, which is now nothing.**
///
/// Inverted. It asserted a product; it asserts a sum. Kept rather than deleted
/// because the product is what the first implementation shipped and the reason
/// it did not survive is the number this test holds: the graph is the same size
/// whether or not a phase declares a barrier, and the declaration costs one bool
/// per phase.
#[test]
fn a_barrier_costs_the_graph_nothing() {
    for block in [[8, 8, 16], [4, 4, 8], [2, 2, 2]] {
        let blocks = BlockGrid::new(VOLUME, block).expect("a lattice").n_blocks();
        let with = run_in_plan(block, true, true, SchedulePriority::PhaseMajor, 1);

        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        let plain = PhaseDecomposition::derive(
            vec![0],
            vec!["one".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            grid.clone(),
        );
        let ordinary = Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases: vec![plain.clone(), plain],
            chain_reach: [0, 0, 0],
        };
        let edges: usize = TaskGraph::build(&ordinary)
            .tasks_in_phase(1)
            .iter()
            .map(|task| task.deps.len())
            .sum();
        assert_eq!(edges, blocks, "a zero-reach phase depends on its own block");
        assert_eq!(
            with.edges, edges,
            "and a barrier phase on the same lattice costs the graph the same"
        );
    }
}

// ------------------------------------------------------------- the refusals --

/// A reduction without a barrier is refused at plan time, by name.
#[test]
fn a_reduction_without_a_barrier_is_refused() {
    let grid = BlockGrid::new(VOLUME, [8, 8, 16]).expect("a lattice");
    let lattice = grid.blocks_per_axis();
    let summarise = BlockSumOp { name: "sum" };
    let offset = GlobalOffsetOp::new("offset", 0, lattice)
        .with_barrier(false)
        .hoisting(true);
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![fragment_phase(&summarise, grid).expect("phase 0")],
        chain_reach: [0, 0, 0],
    };
    plan = append_fragment_phase(plan, &offset).expect("phase 1");
    let err = blockflow::fragment::check_phase_work(
        &plan,
        &[
            PhaseWork::Fragments(&summarise),
            PhaseWork::Fragments(&offset),
        ],
    )
    .expect_err("a reduction with no moment to run at is not a plan");
    let text = err.to_string();
    assert!(text.contains("computes a phase reduction"), "{text}");
    assert!(text.contains("barrier() == false"), "{text}");

    // The liveness control: the same op with the barrier declared is accepted.
    let offset = GlobalOffsetOp::new("offset", 0, lattice)
        .with_barrier(true)
        .hoisting(true);
    let grid = BlockGrid::new(VOLUME, [8, 8, 16]).expect("a lattice");
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![fragment_phase(&summarise, grid).expect("phase 0")],
        chain_reach: [0, 0, 0],
    };
    plan = append_fragment_phase(plan, &offset).expect("phase 1");
    blockflow::fragment::check_phase_work(
        &plan,
        &[
            PhaseWork::Fragments(&summarise),
            PhaseWork::Fragments(&offset),
        ],
    )
    .expect("a barrier gives the reduction a moment to run at");
}

/// A plan whose record disagrees with its op is refused, in both directions.
#[test]
fn a_plan_that_disagrees_with_its_op_about_the_barrier_is_refused() {
    let grid = BlockGrid::new(VOLUME, [8, 8, 16]).expect("a lattice");
    let lattice = grid.blocks_per_axis();
    let summarise = BlockSumOp { name: "sum" };

    for (declared, recorded) in [(true, false), (false, true)] {
        let offset = GlobalOffsetOp::new("offset", 0, lattice).with_barrier(declared);
        let mut plan = Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases: vec![fragment_phase(&summarise, grid.clone()).expect("phase 0")],
            chain_reach: [0, 0, 0],
        };
        plan = append_fragment_phase(plan, &offset).expect("phase 1");
        plan.phases[1] = plan.phases[1].clone().with_barrier(recorded);
        let err = blockflow::fragment::check_phase_work(
            &plan,
            &[
                PhaseWork::Fragments(&summarise),
                PhaseWork::Fragments(&offset),
            ],
        )
        .expect_err("a plan that waits for one thing and fetches another is not a plan");
        assert!(err.to_string().contains("declares `barrier() =="), "{err}");
    }
}

/// A barrier recorded on a phase that runs no fragment op is refused: nothing
/// could have asked for it, and what it does is serialise two phases silently.
#[test]
fn a_barrier_on_a_phase_no_op_could_have_declared_it_for_is_refused() {
    let grid = BlockGrid::new(VOLUME, [8, 8, 16]).expect("a lattice");
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![
            PhaseDecomposition::derive(
                vec![0],
                vec!["first".to_string()],
                [0, 0, 0],
                [0, 0, 0],
                grid.clone(),
            ),
            PhaseDecomposition::derive(
                vec![1],
                vec!["second".to_string()],
                [0, 0, 0],
                [0, 0, 0],
                grid,
            ),
        ],
        chain_reach: [0, 0, 0],
    };
    // The liveness control first: without the barrier the same plan passes.
    blockflow::fragment::check_phase_work(&plan, &[PhaseWork::Pixels, PhaseWork::Pixels])
        .expect("two pixel phases are a plan");
    plan.phases[1] = plan.phases[1].clone().with_barrier(true);
    let err = blockflow::fragment::check_phase_work(&plan, &[PhaseWork::Pixels, PhaseWork::Pixels])
        .expect_err("a barrier nothing declared is a barrier nothing asked for");
    assert!(err.to_string().contains("records a barrier"), "{err}");
}

// -------------------------------------------------------------- the plan --

/// The barrier is in the binding half of the plan: it changes the fingerprint
/// and it survives the wire.
#[test]
fn the_barrier_is_part_of_the_plan_and_not_a_hint() {
    let grid = BlockGrid::new(VOLUME, [8, 8, 16]).expect("a lattice");
    let plain = PhaseDecomposition::derive(
        vec![0],
        vec!["first".to_string()],
        [0, 0, 0],
        [0, 0, 0],
        grid.clone(),
    );
    let one = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![plain.clone(), plain.clone()],
        chain_reach: [0, 0, 0],
    };
    let mut two = one.clone();
    two.phases[1] = two.phases[1].clone().with_barrier(true);
    assert_ne!(
        one.fingerprint(),
        two.fingerprint(),
        "two plans that wait for different things must not hash the same"
    );

    // And a plan with no barrier fingerprints exactly as it did before the
    // field existed — the frozen fingerprints every op is pinned by are unmoved.
    let mut unchanged = one.clone();
    unchanged.phases[1] = unchanged.phases[1].clone().with_barrier(false);
    assert_eq!(one.fingerprint(), unchanged.fingerprint());
}

/// **Nobody may run a block of a reducing phase on its own.** The blob is
/// computed for the phase and there is nothing in a single task's arguments to
/// carry it, so the pair is refused rather than handed an empty table.
#[test]
fn a_single_task_of_a_reducing_phase_is_refused_rather_than_given_nothing() {
    let grid = BlockGrid::new(VOLUME, [8, 8, 16]).expect("a lattice");
    let lattice = grid.blocks_per_axis();
    let summarise = BlockSumOp { name: "sum" };
    let hoisting = GlobalOffsetOp::new("offset", 0, lattice)
        .with_barrier(true)
        .hoisting(true);
    let plain = GlobalOffsetOp::new("offset", 0, lattice)
        .with_barrier(true)
        .hoisting(false);

    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![fragment_phase(&summarise, grid).expect("phase 0")],
        chain_reach: [0, 0, 0],
    };
    plan = append_fragment_phase(plan, &hoisting).expect("phase 1");
    let graph = TaskGraph::build(&plan);
    let task = graph.tasks_in_phase(1)[0].clone();
    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(scene()),
        &plan,
        [VOLUME[0], VOLUME[1], VOLUME[2]],
    )
    .expect("environment");
    let chain = Chain::sequence(Vec::new());
    let err = blockflow::strategy::execute_task_of(
        &chain,
        &plan,
        &task,
        &PhaseWork::Fragments(&hoisting),
        &env,
        &[],
    )
    .expect_err("a hoisted reduction cannot travel one task at a time");
    assert!(
        err.to_string().contains("computes a phase reduction"),
        "{err}"
    );

    // The liveness control: a barrier *without* a reduction distributes, because
    // the barrier is an ordering and the block is an ordinary block.
    blockflow::strategy::execute_task_of(
        &chain,
        &plan,
        &task,
        &PhaseWork::Fragments(&plain),
        &env,
        &[],
    )
    .expect("a barrier alone is an ordering, and a block of one is an ordinary block");
}

// ------------------------------------------------------------ the schedule --

/// **What a barrier gives up, measured**: no task of the barrier phase is
/// admitted until every task below it has written its output.
///
/// The negative control is the same run with `barrier()` answering `false` and
/// the halo relieved by hand — which is what an op that hoisted a reduction
/// without declaring a barrier would be, and which `check_phase_work` refuses
/// for exactly this reason. Here it is built anyway, to show what the refusal
/// is protecting against: under `BlockMajor` a block of phase 1 is admitted
/// while phase 0 still has blocks outstanding.
#[test]
fn no_block_of_a_barrier_phase_starts_before_the_phase_below_has_finished() {
    let block = [4, 4, 8];
    let blocks = BlockGrid::new(VOLUME, block).expect("a lattice").n_blocks();
    // Serial, so that the event order is the schedule rather than a race.
    let run = run_in_plan(block, true, true, SchedulePriority::BlockMajor, 1);
    assert_eq!(outstanding_at_first_admission(&run.log, 1, blocks), Some(0));

    // The control: the same lattice, no barrier, and the fetch derived edges —
    // which under a whole-volume halo are also every task, so this arm too has
    // to wait. The point of asserting it is that the *reason* differs and the
    // price does not: `barriers.md` §4's "in the case that motivated it, nothing
    // is given up at all".
    let control = run_in_plan(block, false, false, SchedulePriority::BlockMajor, 1);
    assert_eq!(
        outstanding_at_first_admission(&control.log, 1, blocks),
        Some(0)
    );

    // And the arm that shows the schedule can interleave at all, so that the
    // assertion above is not vacuous: a plain two-phase pixel plan under
    // `BlockMajor` admits phase 1 with phase 0 outstanding.
    let interleaved = plain_pixel_run(block);
    assert!(
        outstanding_at_first_admission(&interleaved, 1, blocks).is_some_and(|left| left > 0),
        "the executor did not interleave two ordinary phases, so the barrier assertion \
         above is asserting nothing"
    );
}

/// How many blocks of `phase - 1` had not yet written when the first block of
/// `phase` was admitted. `None` if the phase was never admitted.
///
/// `TaskAdmitted` is emitted before any of a task's work and `BlockWritten`
/// after all of it, so on a serial run the two orders together are the schedule.
fn outstanding_at_first_admission(log: &ExecutionLog, phase: usize, below: usize) -> Option<usize> {
    let mut written = 0usize;
    for event in log.events() {
        match event {
            Event::BlockWritten { phase: at, .. } if at == phase - 1 => written += 1,
            Event::TaskAdmitted { phase: at, .. } if at == phase => {
                return Some(below - written);
            }
            _ => {}
        }
    }
    None
}

/// A two-phase plan of ordinary pixel phases, run under `BlockMajor`, purely as
/// evidence that the executor interleaves phases when nothing stops it.
fn plain_pixel_run(block: [usize; 3]) -> Arc<ExecutionLog> {
    use blockflow::probes::IdentityOp;
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![
            PhaseDecomposition::derive(
                vec![0],
                vec!["one".to_string()],
                [0, 0, 0],
                [0, 0, 0],
                grid.clone(),
            ),
            PhaseDecomposition::derive(
                vec![1],
                vec!["two".to_string()],
                [0, 0, 0],
                [0, 0, 0],
                grid,
            ),
        ],
        chain_reach: [0, 0, 0],
    };
    let workflow = Workflow::new(
        Chain::sequence(vec![
            Chain::op(IdentityOp::new("one", [0, 0, 0])),
            Chain::op(IdentityOp::new("two", [0, 0, 0])),
        ]),
        VOLUME,
        Dtype::F64,
    );
    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(scene()),
        &plan,
        [VOLUME[0], VOLUME[1], VOLUME[2]],
    )
    .expect("environment");
    let log = Arc::new(ExecutionLog::new());
    execute_phases(
        "interleaved",
        &workflow,
        &plan,
        &hints(SchedulePriority::BlockMajor, 1),
        &env,
        &[log.clone()],
        &[PhaseWork::Pixels, PhaseWork::Pixels],
    )
    .expect("a run");
    log
}

// -------------------------------------------------------------- the reduce --

/// `reduce` is handed the **whole** fragment set, even though the op declares a
/// fragment reach of zero. That separation is where the money is.
#[test]
fn the_reduction_sees_every_block_while_the_blocks_see_none() {
    let seen: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    struct CountingOp {
        lattice: [usize; 3],
        seen: Arc<AtomicUsize>,
    }
    impl FragmentOp for CountingOp {
        fn name(&self) -> &'static str {
            "counting"
        }
        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }
        fn reads_pixels(&self) -> bool {
            true
        }
        fn writes_pixels(&self) -> bool {
            true
        }
        fn barrier(&self) -> bool {
            true
        }
        fn gathers(&self) -> bool {
            false
        }
        fn inputs(&self) -> Vec<FragmentInput> {
            vec![FragmentInput::own(STREAM.to_string(), 0).with_reach([0, 0, 0])]
        }
        fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>, blockflow::Error> {
            let count = at.fragments(STREAM)?.len();
            self.seen.store(count, Ordering::SeqCst);
            assert_eq!(
                at.blocks().len(),
                self.lattice[0] * self.lattice[1] * self.lattice[2]
            );
            Ok(pack_u64(&[count as u64]))
        }
        fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput, blockflow::Error> {
            // The block itself gathers nothing: the reach is zero and `gathers`
            // is false, so `fragments` is empty and the answer comes from the
            // phase's blob.
            assert!(at.fragments(STREAM).is_empty());
            assert_eq!(unpack_u64(at.reduced).expect("a blob").len(), 1);
            Ok(BlockOutput::nothing().with_pixels(at.output_buffer(0.0)?))
        }
    }

    let block = [4, 4, 8];
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let lattice = grid.blocks_per_axis();
    let blocks = grid.n_blocks();
    let summarise = BlockSumOp { name: "sum" };
    let counting = CountingOp {
        lattice,
        seen: seen.clone(),
    };
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![fragment_phase(&summarise, grid).expect("phase 0")],
        chain_reach: [0, 0, 0],
    };
    plan = append_fragment_phase(plan, &counting).expect("phase 1");
    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(scene()),
        &plan,
        [VOLUME[0], VOLUME[1], VOLUME[2]],
    )
    .expect("environment");
    execute_phases(
        "counting",
        &empty_workflow(),
        &plan,
        &hints(SchedulePriority::BlockMajor, 4),
        &env,
        &[],
        &[
            PhaseWork::Fragments(&summarise),
            PhaseWork::Fragments(&counting),
        ],
    )
    .expect("a run");
    assert_eq!(
        seen.load(Ordering::SeqCst),
        blocks,
        "the reduction saw {} of {blocks} fragment(s)",
        seen.load(Ordering::SeqCst)
    );
}

/// **A reduction that is genuinely per-block loses nothing.** An op that takes
/// the default `reduce` is handed an empty blob and is unchanged in every
/// respect, whether or not it declares a barrier.
#[test]
fn an_op_that_reduces_nothing_is_handed_an_empty_blob() {
    struct LookingOp {
        barrier: bool,
        saw: Arc<AtomicUsize>,
    }
    impl FragmentOp for LookingOp {
        fn name(&self) -> &'static str {
            "looking"
        }
        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }
        fn reads_pixels(&self) -> bool {
            true
        }
        fn writes_pixels(&self) -> bool {
            true
        }
        fn barrier(&self) -> bool {
            self.barrier
        }
        fn gathers(&self) -> bool {
            false
        }
        fn inputs(&self) -> Vec<FragmentInput> {
            vec![FragmentInput::own(STREAM.to_string(), 0).with_reach([0, 0, 0])]
        }
        fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput, blockflow::Error> {
            self.saw.fetch_add(at.reduced.len(), Ordering::SeqCst);
            Ok(BlockOutput::nothing().with_pixels(at.output_buffer(0.0)?))
        }
    }

    for barrier in [false, true] {
        let saw = Arc::new(AtomicUsize::new(0));
        let block = [8, 8, 16];
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        let summarise = BlockSumOp { name: "sum" };
        let looking = LookingOp {
            barrier,
            saw: saw.clone(),
        };
        let mut plan = Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases: vec![fragment_phase(&summarise, grid).expect("phase 0")],
            chain_reach: [0, 0, 0],
        };
        plan = append_fragment_phase(plan, &looking).expect("phase 1");
        let env = ArrayEnvironment::for_decomposition(
            Voxels::from(scene()),
            &plan,
            [VOLUME[0], VOLUME[1], VOLUME[2]],
        )
        .expect("environment");
        execute_phases(
            "looking",
            &empty_workflow(),
            &plan,
            &hints(SchedulePriority::PhaseMajor, 1),
            &env,
            &[],
            &[
                PhaseWork::Fragments(&summarise),
                PhaseWork::Fragments(&looking),
            ],
        )
        .expect("a run");
        assert_eq!(
            saw.load(Ordering::SeqCst),
            0,
            "an op that overrides nothing was handed a blob (barrier: {barrier})"
        );
    }
}

/// **What the product would have cost, kept as the measurement that removed it.**
///
/// `#[ignore]`d because it allocates hundreds of megabytes to prove a point
/// about allocating hundreds of megabytes. Run it with
/// `--release --ignored --nocapture` when the number is wanted.
///
/// The first implementation of `barrier` gave every task of the phase an edge to
/// every task below it — `blocks x blocks` — which both schedulers enforced for
/// free through machinery they already had. This builds that cross product by
/// hand, beside the graph that ships, so the trade is a number rather than a
/// preference. `graph.rs`'s own `producers_of` header rejected `O(blocks^2)` at
/// 6 700 blocks as making the DAG cost more to build than the work it schedules;
/// that argument was made about a different feature, before this one existed,
/// and it applies here unchanged.
#[test]
#[ignore = "allocates hundreds of megabytes; run with --release --ignored --nocapture"]
fn a_barrier_at_a_large_block_count_is_priced() {
    use std::time::Instant;
    for edge in [16usize, 8, 4] {
        let volume = [256, 256, 256];
        let grid = BlockGrid::new(volume, [edge * 4, edge * 4, edge * 4]).expect("a lattice");
        let blocks = grid.n_blocks();
        let plain = PhaseDecomposition::derive(
            vec![0],
            vec!["one".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            grid.clone(),
        );
        let mut plan = Decomposition {
            volume,
            dtype: Dtype::F64,
            phases: vec![plain.clone(), plain],
            chain_reach: [0, 0, 0],
        };
        plan.phases[1] = plan.phases[1].clone().with_barrier(true);

        let started = Instant::now();
        let graph = TaskGraph::build(&plan);
        let shipped_build = started.elapsed().as_secs_f64();
        let shipped_edges: usize = graph.tasks.iter().map(|task| task.deps.len()).sum();
        let started = Instant::now();
        let shipped_held: usize = graph.dependents().iter().map(Vec::len).sum();
        let shipped_fan = started.elapsed().as_secs_f64();
        assert!(graph.is_barrier(1), "and it is still a barrier");

        // The cross product, built by hand: what `deps` and `dependents` would
        // have held for this same plan.
        let started = Instant::now();
        let (from, to) = graph.phase_ranges[0];
        let product: Vec<Vec<usize>> = (0..blocks).map(|_| (from..to).collect()).collect();
        let product_edges: usize = product.iter().map(Vec::len).sum();
        let fan: Vec<Vec<usize>> = (0..blocks).map(|_| (0..blocks).collect()).collect();
        let product_held: usize = fan.iter().map(Vec::len).sum();
        let product_build = started.elapsed().as_secs_f64();
        assert_eq!(product_edges, blocks * blocks);

        eprintln!(
            "{blocks:>6} blocks | shipped {shipped_edges:>10} edges + {shipped_held:>10} \
             dependents in {:>6.3} s | product {product_edges:>10} + {product_held:>10} in \
             {product_build:>6.3} s, about {} MiB of usize",
            shipped_build + shipped_fan,
            (product_edges + product_held) * std::mem::size_of::<usize>() / (1024 * 1024)
        );
    }
}

// ------------------------------------------------------ the other scheduler --

/// **The coordinator gates it too, and this is what makes the phase-level fact
/// safe.**
///
/// The barrier used to be `blocks x blocks` edges, which both schedulers
/// enforced for free through indegree machinery they already had. Removing the
/// edges removes that, so the distributed coordinator has to gate explicitly —
/// and if it did not, a worker would compute a barrier phase's block from an
/// incomplete fragment set and report a plausible wrong answer.
///
/// Driven through the coordinator's own public surface — `pull` and `completed`
/// — rather than through its internals, because that is what a worker does. It
/// needs no feature: `distributed`'s state machine compiles and is tested with
/// none, and only the HTTP server is behind the flag.
///
/// Two arms, and the second is the liveness control: the same plan, the same
/// lattice, the same pulls, with the barrier not declared. There phase 1 is
/// handed out while phase 0 still has work, which is what the gate is preventing
/// in the first arm — and it is what `barriers.md` §4 means by giving pipelining
/// up at the phase's own start edge.
#[test]
fn the_coordinator_gates_a_barrier_phase_on_the_phase_below() {
    use blockflow::distributed::coordinator::Job;
    use blockflow::distributed::protocol::Handout;
    use blockflow::distributed::spec::{probe_job, ChainSpec};

    for barrier in [false, true] {
        let (spec, mut plan) = probe_job(4, 2, ChainSpec::identity());
        assert!(plan.n_phases() >= 2, "this test is about a phase boundary");
        let last = plan.n_phases() - 1;
        plan.phases[last] = plan.phases[last].clone().with_barrier(barrier);
        let below = plan.phases[last - 1].blocks.len();
        assert!(below > 1, "a one-block phase below cannot show a gate");

        let mut job = Job::new(spec, plan).expect("a job");
        // Pull and report one task at a time, and watch for the first moment a
        // task of the barrier phase is offered. `handed_early` is the whole
        // measurement: was one offered while the phase below still had work?
        let mut done_below = 0usize;
        let mut handed_early = false;
        while done_below < below {
            match job.pull("w") {
                Handout::Task(assignment) if assignment.phase == last => {
                    handed_early = true;
                    break;
                }
                Handout::Task(assignment) => {
                    job.completed("w", assignment.task).expect("a completion");
                    done_below += 1;
                }
                other => panic!("expected work with {done_below} of {below} done, got {other:?}"),
            }
        }

        if barrier {
            assert!(
                !handed_early,
                "a barrier phase was handed out while the phase below still had \
                 a task outstanding"
            );
            assert_eq!(done_below, below, "the phase below ran to completion first");
            // and then it is handed out, so the gate opens rather than jamming
            let Handout::Task(assignment) = job.pull("w") else {
                panic!("the gate never opened");
            };
            assert_eq!(assignment.phase, last);
        } else {
            assert!(
                handed_early,
                "without a barrier the coordinator pipelines across the phase \
                 boundary, which is the concurrency a barrier gives up — if it \
                 does not, the arm above is asserting nothing"
            );
        }
    }
}

/// And the gate survives a reissue: a worker lost mid-phase must not let the
/// barrier phase through on a completion count that was never earned.
#[test]
fn the_coordinator_gate_is_not_opened_by_a_reissued_task() {
    use blockflow::distributed::coordinator::Job;
    use blockflow::distributed::protocol::Handout;
    use blockflow::distributed::spec::{probe_job, ChainSpec};

    let (spec, mut plan) = probe_job(4, 2, ChainSpec::identity());
    let last = plan.n_phases() - 1;
    plan.phases[last] = plan.phases[last].clone().with_barrier(true);
    let below = plan.phases[last - 1].blocks.len();
    let mut job = Job::new(spec, plan).expect("a job");

    // Every task of the phase below is handed out; all but one are reported,
    // and the last one's worker is lost. The phase is *not* complete, and a
    // count that treated a reissue as progress would say it was.
    let mut handed = Vec::new();
    for _ in 0..below {
        let Handout::Task(assignment) = job.pull("w") else {
            panic!("expected work");
        };
        handed.push(assignment.task);
    }
    for &task in handed.iter().take(below - 1) {
        job.completed("w", task).expect("a completion");
        // A duplicate report of the same task must not count twice either.
        job.completed("w", task).expect("an idempotent completion");
    }
    job.failed("w", *handed.last().expect("one held"), "lost");

    let Handout::Task(assignment) = job.pull("w") else {
        panic!("expected the reissued task");
    };
    assert_eq!(
        assignment.phase,
        last - 1,
        "the reissued task, not a barrier phase task"
    );
}

/// **"Every earlier phase", not "the phase below".**
///
/// A `FragmentInput` names the phase that wrote the stream, and that phase may
/// be further back than `p - 1`. The gate is stated over every earlier phase for
/// exactly this case: here phase 2 declares a barrier and reduces over the
/// stream **phase 0** wrote, with an unrelated phase 1 in between.
///
/// **It does not tell the two formulations apart, and that is worth stating
/// rather than implying.** Mutating the executor's gate to `p - 1` alone leaves
/// this test passing, because the induction that made the narrow form correct is
/// real: every task has a non-empty valid region, those regions tile, so all of
/// `p-1` done implies all of `p-2` done. What this test does show is the
/// property that matters — a reduction over a stream written two phases back
/// sees every fragment of it — which no other test here covers, since every
/// other plan in this file is two phases.
#[test]
fn a_reduction_over_a_stream_from_two_phases_back_sees_all_of_it() {
    let seen: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    struct FarOp {
        seen: Arc<AtomicUsize>,
    }
    impl FragmentOp for FarOp {
        fn name(&self) -> &'static str {
            "far"
        }
        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }
        fn reads_pixels(&self) -> bool {
            true
        }
        fn writes_pixels(&self) -> bool {
            true
        }
        fn barrier(&self) -> bool {
            true
        }
        fn gathers(&self) -> bool {
            false
        }
        fn inputs(&self) -> Vec<FragmentInput> {
            // Written by phase 0, read by phase 2.
            vec![FragmentInput::own(STREAM.to_string(), 0).with_reach([0, 0, 0])]
        }
        fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>, blockflow::Error> {
            let count = at.fragments(STREAM)?.len();
            self.seen.store(count, Ordering::SeqCst);
            Ok(pack_u64(&[count as u64]))
        }
        fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput, blockflow::Error> {
            Ok(BlockOutput::nothing().with_pixels(at.output_buffer(0.0)?))
        }
    }
    /// The phase in between, which writes nothing anyone reads.
    struct MiddleOp;
    impl FragmentOp for MiddleOp {
        fn name(&self) -> &'static str {
            "middle"
        }
        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }
        fn reads_pixels(&self) -> bool {
            true
        }
        fn writes_pixels(&self) -> bool {
            true
        }
        fn outputs(&self) -> Vec<FragmentOutput> {
            vec![FragmentOutput::new(
                "middle".to_string(),
                Lifecycle::DeleteOnExit,
                Coverage::EveryBlock,
            )]
        }
        fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput, blockflow::Error> {
            Ok(BlockOutput::fragment("middle", pack_u64(&[0])).with_pixels(at.output_buffer(0.0)?))
        }
    }

    let block = [4, 4, 8];
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let blocks = grid.n_blocks();
    let summarise = BlockSumOp { name: "sum" };
    let middle = MiddleOp;
    let far = FarOp { seen: seen.clone() };
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![fragment_phase(&summarise, grid).expect("phase 0")],
        chain_reach: [0, 0, 0],
    };
    plan = append_fragment_phase(plan, &middle).expect("phase 1");
    plan = append_fragment_phase(plan, &far).expect("phase 2");
    assert!(plan.phases[2].barrier);
    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(scene()),
        &plan,
        [VOLUME[0], VOLUME[1], VOLUME[2]],
    )
    .expect("environment");
    execute_phases(
        "far",
        &empty_workflow(),
        &plan,
        &hints(SchedulePriority::BlockMajor, 4),
        &env,
        &[],
        &[
            PhaseWork::Fragments(&summarise),
            PhaseWork::Fragments(&middle),
            PhaseWork::Fragments(&far),
        ],
    )
    .expect("a run");
    assert_eq!(
        seen.load(Ordering::SeqCst),
        blocks,
        "the reduction saw {} of {blocks} fragment(s) from two phases back",
        seen.load(Ordering::SeqCst)
    );
}

/// **A reduction that does not associate is caught, exactly as a block's fold
/// is.**
///
/// `SeamFold::Unordered` is the claim that the answer is a function of the *set*
/// of fragments rather than of their order. A hoisted reduction walks the
/// lattice in row-major order, which is one order out of many, and two different
/// lattices walk two different ones — so an `f64` accumulation makes the phase's
/// answer a property of how the volume was cut. The executor reduces a second
/// time with the lattice reversed and requires the same bytes.
///
/// The payload is crafted so that the two orders genuinely differ: `1e18`,
/// `-1e18` and then six `1.0`s sum to `6.0` forwards — the large terms cancel
/// before the small ones arrive — and to `0.0` backwards, because `1e18 + 1.0`
/// is `1e18`.
#[test]
fn a_reduction_that_does_not_associate_is_refused() {
    struct DriftingOp {
        associates: bool,
    }
    impl FragmentOp for DriftingOp {
        fn name(&self) -> &'static str {
            "drifting"
        }
        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }
        fn reads_pixels(&self) -> bool {
            true
        }
        fn writes_pixels(&self) -> bool {
            true
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
        fn inputs(&self) -> Vec<FragmentInput> {
            vec![FragmentInput::own(WIDE.to_string(), 0).with_reach([0, 0, 0])]
        }
        fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>, blockflow::Error> {
            if self.associates {
                // The liveness control, and it is the same program with the
                // accumulator changed: an integer fold associates, so the two
                // walks agree and the phase is decomposition-invariant.
                let mut total = 0u64;
                at.stream_fragments(WIDE, &mut |_, bytes| {
                    total = total.wrapping_add(unpack_u64(bytes)?[1]);
                    Ok(())
                })?;
                return Ok(pack_u64(&[total]));
            }
            let mut total = 0.0f64;
            at.stream_fragments(WIDE, &mut |_, bytes| {
                total += f64::from_bits(unpack_u64(bytes)?[0]);
                Ok(())
            })?;
            Ok(pack_u64(&[total.to_bits()]))
        }
        fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput, blockflow::Error> {
            Ok(BlockOutput::nothing().with_pixels(at.output_buffer(0.0)?))
        }
    }

    /// One fragment per block carrying a value chosen so that an `f64` fold is
    /// order-dependent, and an integer beside it that is not.
    struct WideningOp;
    impl FragmentOp for WideningOp {
        fn name(&self) -> &'static str {
            "widening"
        }
        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }
        fn reads_pixels(&self) -> bool {
            true
        }
        fn writes_pixels(&self) -> bool {
            true
        }
        fn outputs(&self) -> Vec<FragmentOutput> {
            vec![FragmentOutput::new(
                WIDE.to_string(),
                Lifecycle::Persistent,
                Coverage::EveryBlock,
            )]
        }
        fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput, blockflow::Error> {
            let flat = at.index[0] * 4 + at.index[1] * 2 + at.index[2];
            // `1e18` has an ulp of 128, so `1e18 + 1.0` is `1e18` exactly.
            // Forwards the two large terms cancel first and the six ones
            // survive; backwards the ones are absorbed and the answer is zero.
            let value: f64 = match flat {
                0 => 1e18,
                1 => -1e18,
                _ => 1.0,
            };
            let buffer = at.output_buffer(0.0)?;
            Ok(
                BlockOutput::fragment(WIDE, pack_u64(&[value.to_bits(), flat as u64]))
                    .with_pixels(buffer),
            )
        }
    }

    let block = [8, 8, 8];
    for associates in [false, true] {
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        assert!(grid.n_blocks() >= 3, "an f64 fold of two terms associates");
        let widen = WideningOp;
        let drifting = DriftingOp { associates };
        let mut plan = Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases: vec![fragment_phase(&widen, grid).expect("phase 0")],
            chain_reach: [0, 0, 0],
        };
        plan = append_fragment_phase(plan, &drifting).expect("phase 1");
        let env = ArrayEnvironment::for_decomposition(
            Voxels::from(scene()),
            &plan,
            [VOLUME[0], VOLUME[1], VOLUME[2]],
        )
        .expect("environment");
        let outcome = execute_phases(
            "drifting",
            &empty_workflow(),
            &plan,
            &hints(SchedulePriority::PhaseMajor, 1),
            &env,
            &[],
            &[
                PhaseWork::Fragments(&widen),
                PhaseWork::Fragments(&drifting),
            ],
        );
        if associates {
            outcome.expect("an integer fold associates and the two walks agree");
        } else {
            let err = outcome.expect_err("an f64 fold over three terms does not associate");
            assert!(err.to_string().contains("SeamFold::Unordered"), "{err}");
            assert!(err.to_string().contains("walking the lattice"), "{err}");
        }
    }
}

/// **A barrier on phase 0**, which waits for a phase that does not exist.
///
/// Degenerate but declarable: an op that always says `barrier() == true` may end
/// up first in a plan. The gate — every earlier phase complete — is vacuously
/// satisfied, so the phase runs; there is nothing to deadlock on and nothing for
/// a reduction to read, because `check_phase_work` already refuses a fragment
/// input naming this phase or a later one.
#[test]
fn a_barrier_on_the_first_phase_waits_for_nothing_and_runs() {
    struct FirstOp;
    impl FragmentOp for FirstOp {
        fn name(&self) -> &'static str {
            "first"
        }
        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }
        fn reads_pixels(&self) -> bool {
            true
        }
        fn writes_pixels(&self) -> bool {
            true
        }
        fn barrier(&self) -> bool {
            true
        }
        fn outputs(&self) -> Vec<FragmentOutput> {
            vec![FragmentOutput::new(
                STREAM.to_string(),
                Lifecycle::DeleteOnExit,
                Coverage::EveryBlock,
            )]
        }
        fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput, blockflow::Error> {
            assert!(at.reduced.is_empty(), "there is nothing to have reduced");
            Ok(BlockOutput::fragment(STREAM, pack_u64(&[0])).with_pixels(at.output_buffer(0.0)?))
        }
    }

    let block = [4, 4, 8];
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let blocks = grid.n_blocks();
    let first = FirstOp;
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![fragment_phase(&first, grid).expect("phase 0")],
        chain_reach: [0, 0, 0],
    };
    assert!(plan.phases[0].barrier);
    let graph = TaskGraph::build(&plan);
    assert!(graph.is_barrier(0));
    for task in graph.tasks_in_phase(0) {
        assert!(task.deps.is_empty(), "there is no phase below phase 0");
    }
    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(scene()),
        &plan,
        [VOLUME[0], VOLUME[1], VOLUME[2]],
    )
    .expect("environment");
    let stats = execute_phases(
        "first",
        &empty_workflow(),
        &plan,
        &hints(SchedulePriority::BlockMajor, 4),
        &env,
        &[],
        &[PhaseWork::Fragments(&first)],
    )
    .expect("a run");
    assert_eq!(stats.tasks, blocks);
}
