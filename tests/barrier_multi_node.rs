// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A barrier phase with a hoisted reduction, across real worker processes.**
//
// `docs/design/barriers.md` §7.5 specified `FragmentOp::reduce` as a blob
// computed once for the phase, and §8.7 recorded that it was **refused** in a
// distributed run because nothing carried the blob to a worker. This is the
// file that closes that, and the answer turned out not to be a transport.
//
// What is under test, and why each part of it needs real processes
// ----------------------------------------------------------------
// * **The blob is derived, not observed.** Every worker computes it itself, from
//   the fragment set on shared storage, in an order that is a function of the
//   plan. So every worker must reach *byte-identical* bytes with nothing sent
//   between them — no election, no upload, no download, and nothing added to a
//   coordinator whose design is that it holds no data. Two processes are the
//   minimum that can disagree, which is why this cannot be an in-process test.
// * **The barrier gate is what makes the set whole.** A worker reduces on the
//   first task of a barrier phase it is handed, and the coordinator does not
//   hand one out until every earlier phase has been *reported* complete. A
//   worker writes its fragments before it reports. That chain is three
//   processes long and only a real run exercises it.
// * **Once per phase per worker, not once per task.** That is the entire claim
//   `reduce` makes, and the worker report counts it so it is measured rather
//   than asserted from the design.
//
// What it costs, measured against the alternative
// -----------------------------------------------
// Every worker reading the whole fragment set costs `nodes x F` instead of the
// `2 x F` a single-process run pays, and `nodes` folds instead of one. The
// arithmetic that matters is which multiplier it is: **the case against
// re-deriving the reduction per block was that the multiplier was `blocks`,
// which a caller raises to make a stage fit in memory. This one is `nodes`,
// which a caller sets from the machines they have.** It does not move when the
// lattice does, and `the_cost_of_reducing_on_every_node_is_per_node_not_per_block`
// is where that is shown rather than argued.

#![cfg(feature = "distributed")]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use blockflow::decomposition::Decomposition;
use blockflow::distributed::local::{self, Binaries, LocalOptions};
use blockflow::distributed::shared_volume::SharedVolumes;
use blockflow::distributed::spec::{
    probe_job_over, ChainSpec, FragmentPhaseSpec, HoistedReduceOp, JobSpec, StoreSpec,
};
use blockflow::distributed::HandoutPolicy;
use blockflow::sidecar::Lifecycle;
use ndarray::Array3;
use serde_json::Value;

const BLOCKS: usize = 16;

fn binaries() -> Binaries {
    Binaries {
        coordinator: PathBuf::from(env!("CARGO_BIN_EXE_blockflow-coordinator")),
        worker: PathBuf::from(env!("CARGO_BIN_EXE_blockflow-worker")),
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "blockflow-barrier-multinode-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn ramp(shape: [usize; 3]) -> Array3<f64> {
    let mut array = Array3::zeros((shape[0], shape[1], shape[2]));
    for (flat, value) in array.iter_mut().enumerate() {
        *value = flat as f64;
    }
    array
}

fn options(dir: &Path, workers: usize) -> LocalOptions {
    let mut options = LocalOptions::new(dir, workers).expect("local options");
    options.binaries = binaries();
    options.timeout = Duration::from_secs(120);
    options
}

/// A summary phase followed by a **barrier** phase whose op hoists its fold.
///
/// `hoisted` is the arm under test; `hoisted == false` builds the same job with
/// `FragmentPhaseSpec::reduce`, which is the same answer computed the way the
/// framework admitted it before a barrier existed — whole-lattice fragment
/// reach, the fold re-derived in every block. Keeping both here is what makes
/// the comparison one program with one thing changed.
fn barrier_job(dir: &Path, hoisted: bool) -> (JobSpec, Decomposition) {
    let volumes = dir.join("volumes");
    let (mut spec, pixels_only) = probe_job_over(
        BLOCKS,
        1,
        ChainSpec::identity(),
        StoreSpec::Files {
            dir: volumes.clone(),
        },
    );
    spec.policy = HandoutPolicy::NearestFirst;
    spec.lease = None;
    let summary_phase = pixels_only.n_phases();
    let lattice = pixels_only.phases[summary_phase - 1].grid.blocks_per_axis();
    spec.workflow.fragment_phases = vec![
        FragmentPhaseSpec::summary("summary", "fragments", Lifecycle::DeleteOnExit),
        if hoisted {
            FragmentPhaseSpec::hoisted("hoisted", "fragments", summary_phase)
        } else {
            FragmentPhaseSpec::reduce("reduce", "fragments", summary_phase, lattice)
        },
    ];
    let decomposition =
        blockflow::distributed::spec::decompose(&spec, 1).expect("a barrier job decomposes");
    let store = SharedVolumes::create(
        &volumes,
        spec.workflow.shape,
        spec.workflow.chunk,
        decomposition.n_phases(),
    )
    .expect("image files");
    store
        .write_image(0, &ramp(spec.workflow.shape))
        .expect("an input");
    (spec, decomposition)
}

fn output_bytes(dir: &Path, spec: &JobSpec, decomposition: &Decomposition) -> Vec<u8> {
    SharedVolumes::open(
        &dir.join("volumes"),
        spec.workflow.shape,
        spec.workflow.chunk,
        decomposition.n_phases(),
    )
    .expect("the volumes")
    .image_bytes(decomposition.n_phases())
    .expect("the output image")
}

/// Where two images first differ, and by how much — rather than the two whole
/// byte vectors.
///
/// An `assert_eq!` on the images themselves is correct and unreadable: it prints
/// megabytes of little-endian `f64` on failure, which is the shape of a message
/// nobody reads and therefore nearly as bad as no message. This says the same
/// thing in one line.
fn first_difference(left: &[u8], right: &[u8]) -> Option<String> {
    if left.len() != right.len() {
        return Some(format!(
            "different sizes: {} byte(s) against {}",
            left.len(),
            right.len()
        ));
    }
    let at = left.iter().zip(right).position(|(a, b)| a != b)?;
    let voxel = at / std::mem::size_of::<f64>();
    let word = |bytes: &[u8]| {
        let start = voxel * 8;
        bytes
            .get(start..start + 8)
            .and_then(|slice| slice.try_into().ok())
            .map(f64::from_le_bytes)
    };
    let differing = left.iter().zip(right).filter(|(a, b)| a != b).count();
    Some(format!(
        "first differ at byte {at} (voxel {voxel}): {:?} against {:?}; {differing} of {} \
         byte(s) differ",
        word(left),
        word(right),
        left.len()
    ))
}

fn worker_field(reports: &[Value], field: &str) -> Vec<u64> {
    reports
        .iter()
        .map(|report| report.get(field).and_then(Value::as_u64).unwrap_or(0))
        .collect()
}

// ------------------------------------------------------------ the headline --

/// **N workers produce byte-identical output to a single-node run**, over a
/// phase whose every block's answer is a reduction of every other block's
/// fragment.
///
/// This is the acceptance bar and it is the one assertion that would fail if the
/// per-worker blobs differed by a single byte: the op writes the reduction into
/// every voxel, so a worker that reduced over a partial set — or in a different
/// order — writes a different volume and the comparison catches it. That is why
/// the test compares *bytes* rather than counting reductions and calling it
/// agreement.
#[test]
fn a_hoisted_reduction_is_byte_identical_however_many_workers_run_it() {
    let reference_dir = scratch("hoisted-reference");
    let (reference_spec, reference_plan) = barrier_job(&reference_dir, true);
    let reference_run = local::run(
        &options(&reference_dir, 1),
        &reference_spec,
        &reference_plan,
    )
    .expect("a single-worker run");
    assert_eq!(reference_run.status.done, reference_run.status.tasks);
    let reference = output_bytes(&reference_dir, &reference_spec, &reference_plan);
    assert!(!reference.is_empty(), "the reference wrote nothing");

    // **How many workers actually reduced, across the sweep.** The agreement
    // claim is only exercised by a run where more than one worker reduced, and
    // whether that happens is the handout policy's business, not this test's —
    // a locality-aware coordinator may hand every block of a phase to one
    // worker. So it is measured and asserted at the end rather than assumed per
    // configuration, and without that assertion this test could pass while never
    // once comparing two processes' bytes.
    let mut most_reducers = 0usize;
    for workers in [2usize, 3, 5] {
        let dir = scratch(&format!("hoisted-{workers}"));
        let (spec, plan) = barrier_job(&dir, true);
        let run = local::run(&options(&dir, workers), &spec, &plan)
            .unwrap_or_else(|error| panic!("{workers} workers: {error}"));
        assert_eq!(run.status.done, run.status.tasks);
        if let Some(what) = first_difference(&output_bytes(&dir, &spec, &plan), &reference) {
            panic!(
                "{workers} workers disagree with one node over a hoisted reduction — which \
                 means two workers reduced to different bytes, since the reduction is what \
                 every voxel holds. {what}"
            );
        }

        // The blob was computed on every worker that ran a block of the phase,
        // and **once** on each. Not once per task: that is the claim.
        let reductions = worker_field(&run.workers, "reductions");
        let bytes = worker_field(&run.workers, "reduced_bytes");
        assert!(
            reductions.iter().all(|&count| count <= 1),
            "{workers} workers: a worker reduced more than once per phase: {reductions:?}"
        );
        assert!(
            reductions.iter().sum::<u64>() >= 1,
            "{workers} workers: nobody reduced, so this run proves nothing: {reductions:?}"
        );
        // Every worker that reduced produced the same number of bytes, which is
        // the cheapest cross-process check on the blob short of comparing the
        // output — and the output is compared above.
        let sizes: BTreeSet<u64> = reductions
            .iter()
            .zip(&bytes)
            .filter(|(&count, _)| count > 0)
            .map(|(_, &size)| size)
            .collect();
        assert_eq!(
            sizes.len(),
            1,
            "{workers} workers: the reductions are different sizes: {bytes:?}"
        );
        most_reducers = most_reducers.max(reductions.iter().filter(|&&count| count > 0).count());
        println!(
            "{workers} worker(s): reductions per worker {reductions:?}, {} byte(s) each",
            sizes.iter().next().expect("one size")
        );
        std::fs::remove_dir_all(&dir).ok();
    }
    assert!(
        most_reducers >= 2,
        "no configuration had two workers reduce, so nothing above compared two processes' \
         blobs and this test asserts nothing about cross-node agreement"
    );
    std::fs::remove_dir_all(&reference_dir).ok();
}

/// **The hoisted arm and the in-plan arm agree on the answer, and the worker
/// reports say which one it is.**
///
/// The liveness control for the test above: if `HoistedReduceOp` computed
/// something other than what `FragmentReduceOp` computes, the first test would
/// still pass — every worker would agree on the same wrong number. This is the
/// oracle, and it is the shape the framework admitted before a barrier existed.
///
/// It is also where the hoisting is *measured* across processes rather than
/// argued from the design. `Stats::sidecar_reads` makes the same distinction
/// in-process — `O(blocks)` hoisted against `O(blocks²)` per-block, for the same
/// plan and the same answer — and `WorkerReport::sidecar_reads` is what carries
/// it here. Summed over the workers of a job it is the figure one node would
/// have reported, and the two arms below produce identical volumes and
/// wildly different ones.
#[test]
fn the_hoisted_arm_and_the_in_plan_arm_write_the_same_volume() {
    let mut answers = Vec::new();
    let mut reads = Vec::new();
    let mut applications = Vec::new();
    for hoisted in [false, true] {
        let dir = scratch(if hoisted {
            "agree-hoisted"
        } else {
            "agree-in-plan"
        });
        let (spec, plan) = barrier_job(&dir, hoisted);
        assert_eq!(
            plan.phases[plan.n_phases() - 1].barrier,
            hoisted,
            "only the hoisted arm declares a barrier"
        );
        let run = local::run(&options(&dir, 3), &spec, &plan).expect("a three-worker run");
        assert_eq!(run.status.done, run.status.tasks);
        answers.push(output_bytes(&dir, &spec, &plan));
        reads.push(
            worker_field(&run.workers, "sidecar_reads")
                .iter()
                .sum::<u64>(),
        );
        applications.push(
            worker_field(&run.workers, "fragment_applications")
                .iter()
                .sum::<u64>(),
        );
        std::fs::remove_dir_all(&dir).ok();
    }
    if let Some(what) = first_difference(&answers[0], &answers[1]) {
        panic!("the barrier changes when the answer is computed, never what it is. {what}");
    }

    // **The applications are the control on the reads.** Both arms apply the op
    // to every block of both fragment phases and to nothing else, so the work
    // done per block is the same in the two arms and the read counts below are
    // a comparison of *traffic* rather than of two different amounts of work.
    assert_eq!(
        applications[0], applications[1],
        "the arms apply the op to the same blocks; only where the fold sits differs. \
         In-plan {} against hoisted {}",
        applications[0], applications[1]
    );
    assert_eq!(
        applications[1],
        2 * BLOCKS as u64,
        "two fragment phases, one application per block each, summed over the workers that \
         ran them"
    );

    // And the traffic, which is the whole point of the barrier. The in-plan arm
    // hands every block the whole lattice's fragments; the hoisted arm reads the
    // set once per worker that reduced, plus each block's own fragment.
    assert!(
        reads[0] > reads[1],
        "the in-plan arm must read more fragments than the hoisted one, or the barrier bought \
         nothing: in-plan {} against hoisted {}",
        reads[0],
        reads[1]
    );
    assert!(
        reads[0] >= (BLOCKS * BLOCKS) as u64,
        "the in-plan arm reads the whole set in every block, so its reads are at least \
         blocks²  = {}: measured {}",
        BLOCKS * BLOCKS,
        reads[0]
    );
    assert!(
        reads[1] < (BLOCKS * BLOCKS) as u64,
        "the hoisted arm's multiplier is nodes, not blocks, so with three workers it must \
         stay well under blocks² = {}: measured {}",
        BLOCKS * BLOCKS,
        reads[1]
    );
    println!(
        "{BLOCKS} blocks, 3 workers: in-plan {} fragment reads, hoisted {}; both applied {} \
         fragment ops",
        reads[0], reads[1], applications[1]
    );
}

// --------------------------------------------------------------- the price --

/// **What a barrier buys the plan, and what reducing on every node costs.**
///
/// Two numbers, both from the run rather than from the design:
///
/// * the barrier phase's **halo**, which is the whole volume without one and
///   zero with one — that is where the pixel amplification `barriers.md` §2.1
///   measures at 67.4 GiB goes;
/// * the reduction's **blob size against the fragment set it was derived from**,
///   which is what shipping it would have saved and what deriving it costs.
#[test]
fn the_cost_of_reducing_on_every_node_is_per_node_not_per_block() {
    let dir = scratch("hoisted-price");
    let (spec, plan) = barrier_job(&dir, true);
    let last = plan.n_phases() - 1;
    let blocks = plan.phases[last].blocks.len();
    let volume = plan.volume;

    let (in_plan_dir, in_plan_plan) = {
        let dir = scratch("in-plan-price");
        let (_, plan) = barrier_job(&dir, false);
        (dir, plan)
    };
    let granted = |plan: &Decomposition| {
        let phase = &plan.phases[plan.n_phases() - 1];
        let halo = phase.halo.in_voxels(phase.grid.block());
        [
            halo.axis(0).bound(plan.volume[0]).0,
            halo.axis(1).bound(plan.volume[1]).0,
            halo.axis(2).bound(plan.volume[2]).0,
        ]
    };
    assert_eq!(
        granted(&plan),
        [0, 0, 0],
        "a barrier phase's halo is its own reach, which is zero here"
    );
    assert_eq!(
        granted(&in_plan_plan),
        volume,
        "without a barrier the whole-lattice fragment reach still forces a whole-volume halo"
    );
    std::fs::remove_dir_all(&in_plan_dir).ok();

    let workers = 4usize;
    let run = local::run(&options(&dir, workers), &spec, &plan).expect("a four-worker run");
    assert_eq!(run.status.done, run.status.tasks);
    let reductions: u64 = worker_field(&run.workers, "reductions").iter().sum();
    let blob: u64 = worker_field(&run.workers, "reduced_bytes")
        .iter()
        .copied()
        .max()
        .unwrap_or(0);

    // **The multiplier, which is the whole argument.** The reduction ran once
    // per worker that saw a block of the phase — never once per block.
    assert!(
        reductions <= workers as u64,
        "{reductions} reduction(s) across {workers} worker(s): the multiplier is nodes, not \
         tasks"
    );
    assert!(
        (reductions as usize) < blocks,
        "{reductions} reduction(s) against {blocks} block(s): if these were equal the \
         reduction would not have been hoisted at all"
    );
    println!(
        "{blocks} blocks, {workers} workers: {reductions} reduction(s) of {blob} byte(s) each; \
         halo {:?} with a barrier against {:?} without",
        granted(&plan),
        volume
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// **A reduction over an incomplete fragment set is refused, not answered.**
///
/// The one failure mode this design has: if a sidecar store is not in fact
/// shared between nodes, every worker reduces over its own fragments and answers
/// plausibly and differently on each machine. `strategy::reduce_phase` verifies
/// the completeness the barrier promises rather than trusting it, for every
/// declared input stream whose producer said `Coverage::EveryBlock`.
///
/// Provoked in-process, because provoking it across processes would mean
/// building a deliberately-unshared store — and the guard is the same code on
/// both paths. The fragment set is emptied between the two phases, which is what
/// an unshared store looks like from a worker that wrote none of it.
#[test]
fn a_reduction_over_an_incomplete_fragment_set_is_refused() {
    use blockflow::distributed::spec::ProbeWorkflows;
    use blockflow::distributed::WorkflowFactory;
    use blockflow::fragment::PhaseWork;
    use blockflow::strategy::reduce_phase;

    let dir = scratch("incomplete");
    let (spec, plan) = barrier_job(&dir, true);
    let last = plan.n_phases() - 1;
    let factory = ProbeWorkflows;
    let ops = factory
        .fragment_ops(&spec.workflow, 0)
        .expect("the fragment ops");
    let pixel_phases = plan.n_phases() - ops.len();
    let work: Vec<PhaseWork> = (0..plan.n_phases())
        .map(|phase| match phase.checked_sub(pixel_phases) {
            None => PhaseWork::Pixels,
            Some(index) => PhaseWork::Fragments(ops[index].as_ref()),
        })
        .collect();
    let env = factory
        .environment(&spec.workflow, plan.n_phases())
        .expect("an environment");
    env.prepare(&plan).expect("prepared");
    for entry in &work {
        if let PhaseWork::Fragments(op) = entry {
            for output in op.outputs() {
                env.declare_sidecar(&output.stream, output.lifecycle)
                    .expect("declared");
            }
        }
    }

    // Nothing has been written, which is what a worker on an unshared store
    // would see: the barrier says the set is whole and it is empty.
    let err = reduce_phase(&plan, last, &work, env.as_ref())
        .expect_err("an empty fragment set is not a complete one");
    let text = err.to_string();
    assert!(text.contains("is not complete"), "{text}");
    assert!(text.contains("not actually shared between nodes"), "{text}");

    // The liveness control: with the set whole, the same call succeeds. Written
    // through the store directly, so the control differs from the arm above in
    // exactly one thing — whether the fragments are there.
    let grid = &plan.phases[last].grid;
    for core in grid.cores() {
        env.write_sidecar(
            "fragments",
            last - 1,
            core.index,
            &blockflow::fragment::pack_u64(&[
                core.index[0] as u64,
                core.index[1] as u64,
                core.index[2] as u64,
                core.core.voxels() as u64,
                0f64.to_bits(),
                0,
            ]),
        )
        .expect("a fragment");
    }
    let blob = reduce_phase(&plan, last, &work, env.as_ref()).expect("a complete set reduces");
    assert_eq!(
        HoistedReduceOp::read(&blob).expect("a hoisted blob"),
        plan.volume.iter().product::<usize>() as u64,
        "the reduction is the summed voxel count of the whole volume"
    );
    std::fs::remove_dir_all(&dir).ok();
}
