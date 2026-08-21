// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// `docs/design/BLOCK_OPS.md` §"Images have a lifetime, and iteration has
// substages", the executable half. One phase, an unknown number of substages,
// two operands at every one of them.
//
// The probe is `probes::CappedSpreadOp`: `g <- min(spread(g), f)`, where
// `spread` is the neighbourhood maximum along axis 0 and `f` is the phase's input
// read pointwise. It is written for this suite rather than borrowed from
// `src/ops/`, which is the honest way to test a framework feature: converting a
// real op would change that op's reach, which is parity-visible and is a separate
// review.
//
// What each property is worth, since none of them is obvious from the code:
//
// 1. **The external reach is one substage's, whatever the count.** This is the
//    entire reason the shape exists. A forty-six-substage run has the reach of a
//    one-substage run, where `Chain::sequence` of forty-six one-substage ops would
//    have reach forty-six and `ops/deconvolve.rs`'s in-op loop has `2rn`.
// 2. **Live storage is two buffers, whatever the count.** Measured the way
//    `tests/image_lifetime.rs` measures image residency — from the environment's
//    own counter, not argued.
// 3. **Decomposition invariance**, against a whole-volume reference that runs the
//    same kernel in a loop. The bar every op in this crate meets.
// 4. **The fixed operand is there at every substage**, with the wrong algorithm
//    computed alongside so the test fails if it is supplied only at substage 0.
//    Without this the framework would silently compute something else, which is
//    exactly what `ops/deconvolve.rs`'s header warns about.
// 5. **Convergence terminates and the count follows the data**, not the plan.
// 6. **The limit fires**, names the op and the count, and writes nothing.
// 7. **The halo guard still fires**, because an iterative phase is not exempt
//    from the check the whole crate rests on.

use ndarray::Array3;

use blockflow::decomposition::Decomposition;
use blockflow::env::{AccountingEnvironment, ArrayEnvironment};
use blockflow::fragment::PhaseWork;
use blockflow::geometry::BlockGrid;
use blockflow::iterate::{iterative_phase, substage_reach, IterativeOp, Substage, SubstageLimit};
use blockflow::log::Stats;
use blockflow::op::{Anchor, Chain};
use blockflow::probes::{CappedSpreadOp, IdentityOp};
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [48, 4, 4];
/// Everything is a parameter: the value above which a voxel seeds the spread is
/// the caller's, not the op's.
const SEED_ABOVE: f64 = 50.0;

/// A high value at one end of axis 0, a path of moderate values `length` voxels
/// long, a gap of zeros, and then a region the spread can never reach.
///
/// The high value spreads one voxel per substage and is capped at the path's own
/// value, so the iteration takes `length + 2` substages: one to seed, `length` to
/// travel, and one to observe that nothing moved. Changing `length` changes the
/// count and nothing else, which is what makes it the right input for property 5.
///
/// **The unreachable region is what makes the answer differ from the input.** An
/// earlier version of this file had a path that ran to the end of the volume, and
/// the fixed point was then the input itself — so an iteration that did nothing at
/// all would have passed the invariance test. The gap stops the spread and the
/// tail is never seeded, so the answer is zero there where the input is not, and
/// the comparison has something to say.
fn input(length: usize) -> Voxels {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, _, _)| {
        if i == 0 {
            100.0
        } else if i <= length {
            5.0
        } else if i <= length + 2 {
            0.0
        } else {
            7.0
        }
    })
    .into()
}

fn op(limit: SubstageLimit) -> CappedSpreadOp {
    CappedSpreadOp::new("spread", SEED_ABOVE, limit)
}

fn generous() -> SubstageLimit {
    CappedSpreadOp::bound_for(VOLUME)
}

/// The whole-volume reference: the same kernel, in a loop, over the whole array
/// with no blocks and no halo at all.
///
/// The loop is the executor's own, written out: the running operand starts as the
/// input, the fixed operand is the input at every substage, and the iteration
/// stops when a substage changes nothing. Nothing about it can be wrong in the
/// way a decomposition can be wrong, which is why it is the reference.
fn reference(op: &CappedSpreadOp, source: &Voxels) -> (Voxels, usize) {
    let volume = source.shape();
    let at = Anchor::whole(volume);
    let fixed = source.clone();
    let mut current = source.clone();
    let mut ran = 0usize;
    loop {
        let mut out = Voxels::zeros(Dtype::F64, volume).expect("a buffer");
        {
            let operands: [&Voxels; 2] = [&current, &fixed];
            op.substage(&Substage::new(ran, &operands, &at), &mut out)
                .expect("a substage");
        }
        ran += 1;
        let changed = out != current;
        current = out;
        if !changed {
            break;
        }
    }
    (current, ran)
}

/// The **wrong** algorithm: the fixed operand is supplied at substage 0 and the
/// running estimate is handed in its place afterwards.
///
/// This is what a framework that read the second operand once would compute, and
/// it is well-formed and plausible — `min(spread(g), g)` is `g`, so it freezes at
/// the seed. Computed here so that property 4 is a comparison against a real
/// alternative rather than a hope.
fn reference_without_the_fixed_operand(op: &CappedSpreadOp, source: &Voxels) -> Voxels {
    let volume = source.shape();
    let at = Anchor::whole(volume);
    let fixed = source.clone();
    let mut current = source.clone();
    let mut ran = 0usize;
    loop {
        let mut out = Voxels::zeros(Dtype::F64, volume).expect("a buffer");
        {
            let second: &Voxels = if ran == 0 { &fixed } else { &current };
            let operands: [&Voxels; 2] = [&current, second];
            op.substage(&Substage::new(ran, &operands, &at), &mut out)
                .expect("a substage");
        }
        ran += 1;
        let changed = out != current;
        current = out;
        if !changed || ran > VOLUME[0] + 4 {
            break;
        }
    }
    current
}

/// One iterative phase over a lattice cut on axis 0.
fn plan(op: &CappedSpreadOp, block: usize) -> Decomposition {
    let grid = BlockGrid::along(VOLUME, &[0], block).expect("a lattice");
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![iterative_phase(op, grid).expect("an iterative phase")],
        chain_reach: [1, 0, 0],
    }
}

/// An iterative phase owns no chain slot, so the workflow it runs under has none
/// either — the same position a fragment-only run is in.
fn empty_workflow() -> Workflow {
    Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::F64)
}

fn run_plan(
    plan: &Decomposition,
    source: Voxels,
    work: &[PhaseWork<'_>],
    workflow: &Workflow,
) -> blockflow::Result<(Stats, ArrayEnvironment)> {
    let env = ArrayEnvironment::for_decomposition(source, plan, [8, 4, 4]).expect("an environment");
    let stats = execute_phases(
        "iterate",
        workflow,
        plan,
        &Hints::default(),
        &env,
        &[],
        work,
    )?;
    Ok((stats, env))
}

fn run(op: &CappedSpreadOp, source: Voxels, block: usize) -> (Stats, ArrayEnvironment) {
    let plan = plan(op, block);
    run_plan(&plan, source, &[PhaseWork::Iterate(op)], &empty_workflow()).expect("a run")
}

fn values(voxels: &Voxels) -> Array3<f64> {
    voxels.view::<f64>().expect("f64").to_owned()
}

// ------------------------------------------------------------ 1. reach --

/// **The headline property.** Forty-six substages and one substage have the same
/// external reach, because the depth is paid in private round trips and not in
/// halo.
#[test]
fn the_external_reach_does_not_grow_with_the_substage_count() {
    let op = op(generous());
    let plan = plan(&op, 12);

    // Stated by the op, derived into the plan, and neither of them mentions a
    // substage count — there is nothing in either for one to multiply.
    assert_eq!(substage_reach(&op), [1, 0, 0]);
    assert_eq!(plan.phases[0].reach, [1, 0, 0]);
    assert_eq!(plan.phases[0].halo, [1, 0, 0]);

    // The same plan, twice, over data that makes it iterate very different
    // numbers of times.
    let brief = run_plan(
        &plan,
        input(1),
        &[PhaseWork::Iterate(&op)],
        &empty_workflow(),
    )
    .expect("a run");
    let deep = run_plan(
        &plan,
        input(44),
        &[PhaseWork::Iterate(&op)],
        &empty_workflow(),
    )
    .expect("a run");

    assert_eq!(brief.0.substages, vec![3], "seed, one step, one quiet pass");
    assert_eq!(deep.0.substages, vec![46]);
    assert!(
        deep.0.substages[0] >= 40,
        "the deep run must actually be deep for this test to say anything"
    );

    // The plan is one object and its fingerprint is one number: the substage
    // count is in neither, which is the design's claim about where it belongs.
    assert_eq!(
        brief.0.decomposition_fingerprint,
        deep.0.decomposition_fingerprint
    );

    // And the comparison that makes the number mean something: a `Chain::sequence`
    // of one op per substage would need a halo of one per stage — forty-six
    // voxels, wider than the twelve-voxel block this phase is cut into, so no such
    // plan is even runnable at this block size. This one reaches one.
    let per_substage = 1usize;
    assert_eq!(plan.phases[0].halo, [per_substage, 0, 0]);
    assert!(
        deep.0.substages[0] * per_substage > plan.phases[0].grid.block()[0],
        "the stacked-chain reach would exceed a whole block; this phase's does not"
    );
}

// ---------------------------------------------------------- 2. storage --

/// Two private buffers, whatever `N` is.
///
/// Measured from the environment's own residency counter, the way
/// `tests/image_lifetime.rs` counts resident images: a run that iterated
/// forty-six times must peak at exactly what a run that iterated three times
/// peaked at, because the buffer written at substage `k` is the one holding
/// substage `k-2`.
#[test]
fn live_storage_is_constant_in_the_substage_count() {
    let op = op(generous());
    let (brief, brief_env) = run(&op, input(1), 12);
    let (deep, deep_env) = run(&op, input(44), 12);

    assert_eq!(brief.substages, vec![3]);
    assert_eq!(deep.substages, vec![46]);
    assert_eq!(
        brief.peak_resident_bytes, deep.peak_resident_bytes,
        "peak residency moved with the substage count; the ping-pong is not a ping-pong"
    );

    // Not merely equal but small: one whole volume is 6144 voxels at eight bytes,
    // and the phase holds two of them plus a handful of blocks. One buffer per
    // substage would be forty-six.
    let volume_bytes = (VOLUME[0] * VOLUME[1] * VOLUME[2] * std::mem::size_of::<f64>()) as u64;
    assert!(
        deep.peak_resident_bytes < 4 * volume_bytes,
        "peak {} against a volume of {volume_bytes}",
        deep.peak_resident_bytes
    );
    assert!(deep.peak_resident_bytes > 2 * volume_bytes);

    // The images themselves are unaffected: input and output, as always.
    assert_eq!(brief_env.resident_images(), 2);
    assert_eq!(deep_env.resident_images(), 2);
}

// ----------------------------------------------------- 3. invariance --

/// The answer is the whole-volume answer, and it is byte-identical across cuts.
#[test]
fn an_iterative_phase_is_decomposition_invariant() {
    let op = op(generous());
    let source = input(44);
    let (expected, expected_substages) = reference(&op, &source);
    assert_eq!(expected_substages, 46);

    // The reference is not the input: an iteration that did nothing would pass a
    // comparison against itself.
    assert_ne!(values(&expected), values(&source));

    let mut answers = Vec::new();
    for block in [48usize, 24, 16, 12, 8, 6] {
        let (stats, env) = run(&op, source.clone(), block);
        let got = values(&env.output());
        assert_eq!(
            got,
            values(&expected),
            "block {block} disagrees with the whole-volume reference"
        );
        assert_eq!(
            stats.substages,
            vec![expected_substages],
            "block {block} took a different number of substages than the whole volume"
        );
        answers.push(got);
    }
    for answer in &answers[1..] {
        assert_eq!(answer, &answers[0], "two decompositions disagree");
    }
}

// -------------------------------------------------- 4. the second operand --

/// The fixed operand is read at **every** substage, and this test fails if it is
/// read only at substage 0.
#[test]
fn the_fixed_operand_is_available_at_every_substage() {
    let op = op(generous());
    let source = input(44);

    let (right, substages) = reference(&op, &source);
    let wrong = reference_without_the_fixed_operand(&op, &source);

    // The two algorithms really do differ, and they differ *late*: the wrong one
    // freezes at the seed after one substage, so the divergence is at step 5 and
    // at every step after it, not at step 0.
    assert_ne!(
        values(&right),
        values(&wrong),
        "the two algorithms agree, so this test proves nothing"
    );
    assert!(substages > 5);
    let frozen = values(&wrong);
    assert_eq!(frozen[[10, 0, 0]], 0.0, "the wrong iteration never travels");
    assert_eq!(values(&right)[[10, 0, 0]], 5.0);

    let (stats, env) = run(&op, source, 12);
    assert_eq!(values(&env.output()), values(&right));
    assert_ne!(values(&env.output()), frozen);
    assert_eq!(stats.substages, vec![substages]);
}

// ------------------------------------------------------ 5. convergence --

/// One plan, two datasets, two substage counts, both answers right.
#[test]
fn the_substage_count_is_a_function_of_the_data_and_not_of_the_plan() {
    let op = op(generous());
    let plan = plan(&op, 12);

    let mut seen = Vec::new();
    for length in [1usize, 5, 17, 44] {
        let source = input(length);
        let (expected, expected_substages) = reference(&op, &source);
        let (stats, env) =
            run_plan(&plan, source, &[PhaseWork::Iterate(&op)], &empty_workflow()).expect("a run");
        assert_eq!(stats.substages, vec![expected_substages], "length {length}");
        assert_eq!(values(&env.output()), values(&expected), "length {length}");
        // Same plan every time; only the trace moved.
        assert_eq!(stats.decomposition_fingerprint, plan.fingerprint());
        seen.push(expected_substages);
    }
    assert_eq!(seen, vec![3, 7, 19, 46]);
}

// ------------------------------------------------------------ 6. limit --

/// The runaway guard fires, names the op and the count, and writes nothing.
#[test]
fn the_limit_fires_by_name_and_returns_no_partial_result() {
    let op = op(SubstageLimit::of(3).expect("a positive limit"));
    let plan = plan(&op, 12);
    let source = input(44);
    let env =
        ArrayEnvironment::for_decomposition(source, &plan, [8, 4, 4]).expect("an environment");
    let error = execute_phases(
        "iterate",
        &empty_workflow(),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Iterate(&op)],
    )
    .expect_err("this iteration cannot converge in three substages");
    let message = error.to_string();
    assert!(message.contains("spread"), "{message}");
    assert!(message.contains("3 substage(s)"), "{message}");
    assert!(message.contains("deliberately not written"), "{message}");

    // A partially converged volume is plausible, well-formed and wrong, so none
    // of it is there: the output image is still the unwritten sentinel.
    let out = values(&env.output());
    assert!(
        out.iter().all(|value| value.is_nan()),
        "a truncated answer reached the output image"
    );
}

/// A limit of zero is a limit that means "do not run", and is refused rather than
/// honoured.
#[test]
fn a_zero_limit_is_refused_where_it_is_stated() {
    assert!(SubstageLimit::of(0).is_err());
}

// -------------------------------------------------------- 7. halo guard --

/// An iterative phase is not exempt from the check the crate rests on.
#[test]
fn the_halo_guard_fires_for_an_iterative_phase_with_a_short_halo() {
    let op = op(generous());
    let plan = plan(&op, 12).with_forced_halo([0, 0, 0]);
    let error = plan
        .check()
        .expect_err("a halo of zero under a reach of one leaves a hole");
    let message = error.to_string();
    assert!(message.contains("tile"), "{message}");
    assert!(message.contains("lost part of their core"), "{message}");

    // And the executor refuses it too, rather than running a plan its own guard
    // rejects.
    let env =
        ArrayEnvironment::for_decomposition(input(44), &plan, [8, 4, 4]).expect("an environment");
    assert!(execute_phases(
        "iterate",
        &empty_workflow(),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Iterate(&op)],
    )
    .is_err());
}

// ------------------------------------------------- the surrounding plan --

/// An iterative phase in the middle of a chain: the tasks before it feed it, the
/// tasks after it read what it wrote, and the image it read is freed when it is
/// done with it.
#[test]
fn an_iterative_phase_sits_between_ordinary_phases() {
    let op = op(generous());
    let chain = Chain::sequence(vec![
        Chain::op(IdentityOp::new("before", [0, 0, 0])),
        Chain::op(IdentityOp::new("after", [0, 0, 0])),
    ]);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let slots = workflow.chain.slots();
    let grid = BlockGrid::along(VOLUME, &[0], 12).expect("a lattice");
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![
            blockflow::decomposition::PhaseDecomposition::derive(
                vec![0],
                vec![slots[0].display_name()],
                [0, 0, 0],
                [0, 0, 0],
                grid.clone(),
            ),
            iterative_phase(&op, grid.clone()).expect("an iterative phase"),
            blockflow::decomposition::PhaseDecomposition::derive(
                vec![1],
                vec![slots[1].display_name()],
                [0, 0, 0],
                [0, 0, 0],
                grid,
            ),
        ],
        chain_reach: [1, 0, 0],
    };
    plan.check().expect("a plan");

    let source = input(44);
    let (expected, expected_substages) = reference(&op, &source);
    let env =
        ArrayEnvironment::for_decomposition(source, &plan, [8, 4, 4]).expect("an environment");
    let stats = execute_phases(
        "iterate",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Iterate(&op),
            PhaseWork::Pixels,
        ],
    )
    .expect("a run");

    assert_eq!(values(&env.output()), values(&expected));
    // One entry per phase, and only the iteration has a count.
    assert_eq!(stats.substages, vec![0, expected_substages, 0]);
    assert_eq!(stats.tasks, 3 * grid_blocks());
    // The image the iteration read is internal and is freed the moment the
    // iteration finishes, exactly as any other phase's input is.
    assert!(env.is_discarded(1));
    assert!(env.is_discarded(2));
    assert_eq!(env.resident_images(), 2);
}

fn grid_blocks() -> usize {
    BlockGrid::along(VOLUME, &[0], 12)
        .expect("a lattice")
        .n_blocks()
}

// -------------------------------------------------------------- guards --

/// An iterative phase owns no slot of the chain, and a plan that gives it one is
/// refused before a block runs rather than producing an image nobody wrote.
#[test]
fn an_iterative_phase_that_is_given_a_chain_slot_is_refused() {
    let op = op(generous());
    let mut plan = plan(&op, 12);
    plan.phases[0].slots = vec![0];
    plan.phases[0].names = vec!["identity".to_string()];
    let workflow = Workflow::new(
        Chain::op(IdentityOp::new("identity", [0, 0, 0])),
        VOLUME,
        Dtype::F64,
    );
    let env =
        ArrayEnvironment::for_decomposition(input(4), &plan, [8, 4, 4]).expect("an environment");
    let message = execute_phases(
        "iterate",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Iterate(&op)],
    )
    .expect_err("a phase cannot both iterate and apply a slot")
    .to_string();
    assert!(message.contains("owns no slot of the chain"), "{message}");
}

/// **A finding, recorded as a test.** An iteration runs to convergence, and
/// whether anything changed is a question about values — so an environment that
/// holds no data cannot run one, and says so rather than inventing a count.
///
/// That is a real limitation: `AccountingEnvironment` is how this crate validates
/// an executor at full scale without a byte of input, and an iterative phase is
/// outside what it can simulate today. Simulating one needs a *stated* substage
/// count, which is a different quantity from the one a real run discovers and is
/// deliberately not invented here.
#[test]
fn an_environment_that_holds_no_data_refuses_to_iterate_rather_than_guessing() {
    let op = op(generous());
    let plan = plan(&op, 12);
    let env = AccountingEnvironment::new(VOLUME, [8, 4, 4], 8);
    let message = execute_phases(
        "iterate",
        &empty_workflow(),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Iterate(&op)],
    )
    .expect_err("convergence is a question about values")
    .to_string();
    assert!(message.contains("holds no data"), "{message}");
    assert!(message.contains("spread"), "{message}");
}

// --------------------------------------- 9. the substage wave, at any width --

/// **The blocks of one substage may run together, and it is the same answer.**
///
/// The independence is structural — every block of a substage reads `current`,
/// which nothing writes until the substage ends, and writes its own core of
/// `next`, and the cores are disjoint because they tile — so the wave is safe for
/// the same reason the executor's ready wave is. What is *not* safe, and what
/// this pins from the other side, is running the substages together: block `b` at
/// substage `k + 1` reads cores its neighbours wrote at `k`, so the boundary is a
/// real exchange point and the loop over substages stays serial.
///
/// A run that lost that barrier would not fail loudly. It would read a mixture of
/// substage `k` and `k + 1` across a seam and converge to a plausible volume in a
/// plausible number of substages, so the assertion that catches it is the whole
/// triple: the same values, the **same substage count**, and the same amount
/// changed in each of them.
#[test]
fn a_substage_wave_is_the_same_answer_at_any_width() {
    let op = op(generous());
    let source = input(44);
    let (expected, expected_substages) = reference(&op, &source);

    // A cut fine enough that the wave has something to schedule; one block would
    // make every concurrency below the same run.
    let plan = plan(&op, 6);
    assert!(
        plan.n_tasks() >= 8,
        "the cut leaves {} block(s), too few for a wave",
        plan.n_tasks()
    );

    let mut answers = Vec::new();
    for concurrency in [1usize, 2, 4, 8] {
        let env = ArrayEnvironment::for_decomposition(source.clone(), &plan, [8, 4, 4])
            .expect("an environment");
        let hints = Hints {
            concurrency,
            ..Hints::default()
        };
        let stats = execute_phases(
            "iterate",
            &empty_workflow(),
            &plan,
            &hints,
            &env,
            &[],
            &[PhaseWork::Iterate(&op)],
        )
        .expect("a run");
        assert_eq!(
            values(&env.output()),
            values(&expected),
            "concurrency {concurrency} disagrees with the whole-volume reference"
        );
        assert_eq!(
            stats.substages,
            vec![expected_substages],
            "concurrency {concurrency} took a different number of substages"
        );
        answers.push(stats.substage_changes);
    }
    for (concurrency, changes) in [1usize, 2, 4, 8].iter().zip(&answers) {
        assert_eq!(
            changes, &answers[0],
            "concurrency {concurrency} changed a different amount per substage, which is a \
             substage boundary that stopped being a barrier"
        );
    }
}

/// **What each substage changed, as a sequence rather than as a bit.**
///
/// The reduction the loop already takes, reported. It is worth having because
/// the count alone cannot distinguish an iteration that is converging from one
/// that is about to hit its limit, and the shape of the sequence can: this one
/// decays to zero, and the zero is the last entry exactly because a phase ends on
/// the substage that changed nothing.
///
/// The liveness half is the middle assertion. Without it a run that changed
/// nothing at all — the identity, which is a well-formed iteration — would
/// satisfy the first and the last.
#[test]
fn the_substage_changes_are_the_reduction_the_loop_stops_on() {
    let op = op(generous());
    let (_, env_substages) = reference(&op, &input(44));
    let (stats, _) = run(&op, input(44), 12);

    let changes = &stats.substage_changes[0];
    assert_eq!(
        changes.len(),
        env_substages,
        "one entry per substage: {changes:?}"
    );
    assert_eq!(
        *changes.last().expect("a substage"),
        0,
        "a phase ends on the substage that changed nothing: {changes:?}"
    );
    assert!(
        changes[..changes.len() - 1].iter().all(|count| *count > 0),
        "every substage before the last changed something, or the loop would have stopped \
         earlier: {changes:?}"
    );
    assert!(
        changes.iter().sum::<u64>() > 0,
        "an iteration that changed nothing anywhere is the identity, and would satisfy every \
         other assertion here"
    );
}
