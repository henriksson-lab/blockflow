// SPDX-License-Identifier: MIT
//
// The acceptance criterion for `blockflow::statistics`, end to end, through the
// executor rather than through a mock.
//
// The unit tests in `src/statistics.rs` prove the arithmetic — that a machine
// key partitions, that a store round-trips, that an empty snapshot returns the
// caller's model bit for bit. What they cannot prove is the only claim that
// matters:
//
// > **A coefficient measured by a run predicts the next run better than the
// > constant shipped with the crate did.**
//
// That needs a real chain over a real array with a real environment, because it
// is a claim about the binary that did the work. So this file runs one, records
// what it cost, plans again against the recording, runs again, and asserts that
// the second plan's prediction is closer to what the second run actually spent.
//
// What is deliberately *not* asserted, and what had to be re-baselined
// --------------------------------------------------------------------
// **No test here asserts a duration.** Not "the run took under a second", not
// "the calibrated plan was faster". Every assertion below is a relationship
// between two recorded numbers — a prediction against an observation, a
// fingerprint against a fingerprint.
//
// **The sentence that used to follow that one was wrong, and it is the reason
// `a_measured_coefficient_predicts_better_than_the_shipped_seed` was flaky.** It
// claimed such a relationship is "exactly as true on a loaded machine as on an
// idle one". It is not. A prediction fitted to one run and judged against
// another is judged against a *second* stopwatch, and if the machine's speed
// moves between the two the calibration is charged for a change it did not
// cause. Sustained load is harmless — it slows both runs and the relationship
// survives — so the failure needs load that *changes*, which is exactly what a
// box with six other workers on it produces. Measured below.
//
// **The second wrong claim was about magnitudes.** It said predicted cost
// "spans orders of magnitude" between an uncalibrated model (denominated in
// voxelwise maps) and a calibrated one (denominated in nanoseconds). On this
// machine it spans **2.4x**: `9.87e6` against `2.41e7` for the same plan, with
// the observation at `2.31e7`. The shipped seed is therefore *accidentally
// nearly right* here — one voxelwise map happens to cost about 2.3 ns of the
// work this chain does — and that accident, not the calibration, is what sets
// how much room the comparison has. It is why the criterion below can resolve a
// model that is 10x wrong and cannot resolve one that is 2x too cheap: a model
// 2x too cheap really is closer to the truth than the seed. See
// `the_comparison_rejects_a_model_that_is_ten_times_wrong` for the measured
// resolution in both directions.
//
// Both error measures are kept, because measurement showed they catch different
// things. The **ratio** is scale-free and is the one that means what it says.
// The **difference** is the literal statement of the property, and is the only
// one of the two that rejects a model 2x too dear.

use ndarray::Array3;

use blockflow::decomposition::{predicted_cost, Constraints, CostModel, Decomposition};
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::listener::EventListener;
use blockflow::log::Stats;
use blockflow::op::Chain;
use blockflow::ops::element::{ElementShape, StructuringElement};
use blockflow::ops::rank::RankFilterOp;
use blockflow::ops::smooth::{Gaussian, SmoothOp};
use blockflow::ops::voxelwise::VoxelwiseMapOp;
use blockflow::statistics::{
    observed_nanos, MachineKey, PlanIdentity, Provenance, Recorder, Snapshot, Statistics, Term,
    REPRODUCTIONS,
};
use blockflow::strategy::{Enumerating, Strategy, Workflow};
use blockflow::voxels::Voxels;

use std::sync::Arc;

const VOLUME: [usize; 3] = [32, 32, 64];

/// A chain with three different cost shapes in it, on purpose: a flat voxelwise
/// map, a separable convolution priced per tap, and a dense neighbourhood priced
/// per element voxel. One coefficient has to serve all three, which is what
/// makes `Snapshot::family_spread` say something.
fn chain() -> Chain {
    let element = StructuringElement::from_size(ElementShape::Box, [3, 3, 3]).expect("an element");
    Chain::sequence(vec![
        Chain::op(VoxelwiseMapOp::new("scale", |value| value * 0.5)),
        Chain::op(SmoothOp::new(
            "smooth",
            Gaussian::isotropic(1.0, 3.0).expect("a gaussian"),
        )),
        Chain::op(RankFilterOp::median("median", element)),
    ])
}

fn workflow() -> Workflow {
    Workflow::new(chain(), VOLUME, Dtype::F64)
}

fn input() -> Voxels {
    let mut array = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for (flat, value) in array.iter_mut().enumerate() {
        *value = ((flat * 7919) % 1013) as f64;
    }
    array.into()
}

/// Small blocks on the one splittable axis, so the plan has several of them and
/// the search has something to choose between.
fn constraints(model: CostModel) -> Constraints {
    Constraints {
        block_candidates: vec![16, 32],
        model,
        ..Constraints::default()
    }
}

/// The plan a model would choose. **A planning decision, and nothing below lets
/// a stopwatch reach it** — see [`paired_trial`].
fn plan_for(model: CostModel) -> Decomposition {
    Enumerating::default()
        .decompose(&workflow(), &constraints(model.clone()))
        .expect("a plan")
}

/// What a stated plan is predicted to cost under a stated model.
///
/// The plan is an argument rather than something derived here, which is the
/// whole shape of the repair described in [`paired_trial`]: pricing is a
/// function of `(chain, plan, model)` and asking two models about **one** plan
/// is the only way to compare them that a timing cannot perturb.
fn price(plan: &Decomposition, model: &CostModel) -> f64 {
    predicted_cost(&workflow().chain, plan, &[], model).expect("a price")
}

/// What one run of a stated plan observed.
struct Outcome {
    observed: f64,
    stats: Stats,
    recorder: Arc<Recorder>,
}

/// Run a stated plan. No planning happens here and no model is consulted.
fn run(plan: &Decomposition) -> Outcome {
    let workflow = workflow();
    let env =
        ArrayEnvironment::for_decomposition(input(), plan, [8, 8, 8]).expect("an environment");
    let recorder = Arc::new(Recorder::new(&workflow.chain, plan));
    let listeners: [Arc<dyn EventListener>; 1] = [recorder.clone()];
    let stats = Enumerating::default()
        .run_observed(&workflow, plan, &env, &listeners)
        .expect("a run");
    let observed = observed_nanos(&stats.log);
    Outcome {
        observed,
        stats,
        recorder,
    }
}

/// Plan under a model and run what it chose.
///
/// For the tests that want a whole cycle and hand it the *seeded* model, so the
/// plan is a function of a constant. [`paired_trial`] deliberately does not use
/// this.
fn plan_and_run(model: CostModel) -> Outcome {
    run(&plan_for(model.clone()))
}

/// How wrong a prediction is, scale-free. `1.0` is exact.
fn error(predicted: f64, observed: f64) -> f64 {
    assert!(predicted > 0.0 && observed > 0.0, "nothing was measured");
    // `total_cmp` rather than `f64::max`/`f64::min`: those absorb a NaN and
    // would hand back a ratio for a pair that has no ratio, which in an error
    // measure reads as a very good prediction.
    match predicted.total_cmp(&observed) {
        std::cmp::Ordering::Less => observed / predicted,
        _ => predicted / observed,
    }
}

// ------------------------------------------------------------------------
// The property the whole module exists for.
// ------------------------------------------------------------------------

/// How wrong the negative control's model is made, **as a multiple of the error
/// the shipped seed has on the very run being judged**.
///
/// # Why the factor is derived and not stated
///
/// The criterion here is "is the calibrated prediction closer to this
/// observation than the seed's is", so what it can *resolve* is bounded by how
/// far off the seed already is — and that is a property of the machine at the
/// moment of the run, not a constant. On a quiet box one voxelwise map happens
/// to cost about 2.2 ns of this chain's work, so the seed is off by 2.2x. With
/// eight copies of this test file running at once it is off by **10 to 12x**.
///
/// A control stated as a bare "10x too dear" therefore stops being a control
/// exactly when the box gets busy: measured, `observed 9.910e7`, seed predicting
/// `9.466e6` and so off by `10.47x`, against a 10x-miscalibrated model
/// predicting `1.027e9` and off by `10.36x` — **marginally better than the
/// seed**, so the control won the comparison it exists to lose, 11 times out of
/// 11. That is not the criterion failing. It is a model 10x too dear genuinely
/// being no worse than a seed 10x too cheap.
///
/// Expressing the control in multiples of `seeded_error` puts it in the only
/// coordinates the criterion has. `TooDear(4.0)` is *four times further from the
/// truth than the seed is*, whatever the truth turned out to be, so it must lose
/// on any machine in any mood — and `TooCheap(0.5)` is half as far off as the
/// seed, so it must **win**, which is the resolution limit stated as an
/// assertion rather than as a caveat.
#[derive(Debug, Clone, Copy)]
enum Wrongness {
    /// The honest fitted model.
    Honest,
    /// Over-priced until its error is this multiple of the seed's.
    TooDear(f64),
    /// Under-priced until its error is this multiple of the seed's.
    TooCheap(f64),
}

impl Wrongness {
    /// What to multiply every fitted coefficient by.
    ///
    /// A model priced at `k x` the truth is off by `k`; the honest fitted model
    /// is already close to the truth, so scaling it by `seeded_error * multiple`
    /// lands its error at that multiple of the seed's, and dividing lands the
    /// same distance on the cheap side.
    fn factor(self, seeded_error: f64) -> f64 {
        match self {
            Wrongness::Honest => 1.0,
            Wrongness::TooDear(multiple) => seeded_error * multiple,
            Wrongness::TooCheap(multiple) => 1.0 / (seeded_error * multiple),
        }
    }
}

/// What one paired trial decided.
struct Trial {
    /// Whether the calibrated model beat the seed on the scale-free measure.
    ratio_win: bool,
    /// Whether it beat the seed on the literal difference.
    absolute_win: bool,
    report: String,
}

/// One fit, one held-out run of **one fixed plan**, and both models judged
/// against it.
///
/// # The plan is a fixture, and that is the repair
///
/// The property is "the measured coefficient predicts this run better than the
/// seed did". That is one piece of work, two predictions and one stopwatch — it
/// says nothing about planning, and nothing here plans. `plan` is chosen once by
/// the caller from the *seeded* model and handed in; both models are asked to
/// **price** it, and the same plan is what gets run. A coefficient can therefore
/// be as far off as the machine's mood makes it without changing what is being
/// predicted.
///
/// **The version this replaces re-planned under the fitted model and asserted
/// the plan had not moved. That assertion was wrong twice over.**
///
/// It was wrong about the crate. The argument for it was that
/// `Enumerating::default()`'s `concurrency: 1` is that type's own documented
/// negative control, so the objective is the serial work total and falls
/// monotonically as the block grows — every candidate list answers with its
/// largest entry whatever the coefficients are. That is true of the **block
/// edge** and says nothing about the **phase partition**: where a chain is cut
/// is a function of the coefficients at any concurrency, because a cut trades a
/// materialisation against a halo and the model prices both.
///
/// **The mechanism, measured.** 1080 fits of this chain from nine concurrent
/// threads at a load average of 53, each one re-planned under what it fitted:
///
/// ```text
///   materialise ns/voxel | min 1.95 | p50 2.23 | p90 2.72 | p99 63.68 | max 82.76
/// ```
///
/// Four of the 1080 re-planned, every one of them **2 phases into 1**, and every
/// one with a materialise coefficient in the 63-83 band. That distribution is
/// not drift, it is bimodal: the bulk is a 1.4x spread around 2.2 and about one
/// fit in 250 lands 30x above it. `Term::Materialise` is fitted here from
/// **65 536** units — one intermediate write, the smallest evidence base of any
/// term in this chain, against 139 264 for reads and 9 195 766 for compute — so
/// a single write that stalls for a few milliseconds is the whole coefficient.
/// And a materialisation priced 30x too dear makes a cut unaffordable, so the
/// planner fuses. The term with the least evidence behind it is the one that
/// decides whether a chain is cut.
///
/// The fingerprints four workers reported are exactly these two:
/// `47593583722462630` (`00a9161cbb5d7da6`, the seeded plan) against
/// `10336899902409553960` (`8f740bb1a9e43828`, the fused one).
///
/// It was also wrong about the subject. A model fitted to a measurement choosing
/// a *different plan* is not an anomaly to be asserted away — it is the whole
/// reason a store is worth keeping, and it is measurable. A consumer of this
/// crate fed a `Snapshot` recorded from one workload to a stage that does other
/// work, and the objective moved its chosen block edge up one rung; a stopwatch
/// on the two rungs put the new one **1.06x slower**. An assertion forbidding
/// the plan to move asserted the opposite of what had been established.
///
/// So: anything that wants to say something about calibration **and** planning
/// at once has to **state** its coefficients rather than race the machine for
/// them. A snapshot built from literals is a fact about the objective; a
/// snapshot built from a stopwatch is a random variable, and a plan is not
/// allowed to be one in an assertion.
///
/// # Why both models are judged against the same observation
///
/// The earlier form also compared the seed against the fit run's own stopwatch
/// and the calibration against the held-out run's, so the two errors had
/// different denominators and any change in the machine's speed between them
/// landed entirely on the calibration. The seed is a constant: `predicted_cost`
/// of one plan under one model is one number, so nothing is lost by asking it
/// about the run actually being predicted, and the confound goes with it.
///
/// `wrongness` is the negative control's one changed thing. See [`Wrongness`]
/// for why it is a multiple of the seed's own error rather than a stated factor,
/// and [`the_comparison_rejects_a_model_wronger_than_the_seed`] for the control
/// itself.
fn paired_trial(plan: &Decomposition, wrongness: Wrongness) -> Trial {
    let seeded = CostModel::default();

    // The fit. **`REPRODUCTIONS` runs, because that is what it now takes to be
    // believed** — a store that has seen a term once calibrates nothing, by
    // design, and a fit built from one run would be testing the refusal rather
    // than the calibration. See `blockflow::statistics::REPRODUCTIONS`.
    let mut store = Statistics::new();
    for _ in 0..REPRODUCTIONS {
        store.record(&run(plan).recorder.observations());
    }
    let snapshot = store.snapshot_here();
    assert!(
        !snapshot.is_empty(),
        "one run produced no coefficients:\n{}",
        snapshot.describe()
    );
    let mut calibrated = snapshot.calibrate(&seeded);
    assert_ne!(
        calibrated,
        seeded,
        "a snapshot with evidence left the model untouched:\n{}",
        snapshot.describe()
    );

    // The held-out run: the same plan again. Everything from here is a function
    // of one observation and two prices of one plan.
    let observed = run(plan).observed;
    let seeded_price = price(plan, &seeded);
    let seeded_error = error(seeded_price, observed);

    // The control's factor, derived *now* because it is a multiple of the error
    // the seed happens to have on this run. See [`Wrongness`].
    let factor = wrongness.factor(seeded_error);
    if factor != 1.0 {
        calibrated.read_cost_per_voxel *= factor;
        calibrated.write_cost_per_voxel *= factor;
        calibrated.materialise_cost_per_voxel *= factor;
        calibrated.compute_scale *= factor;
    }

    let calibrated_price = price(plan, &calibrated);
    let calibrated_error = error(calibrated_price, observed);
    let seeded_gap = (seeded_price - observed).abs();
    let calibrated_gap = (calibrated_price - observed).abs();
    let report = format!(
        "plan {:016x}, observed {observed:.3e}\n\
         seeded:     predicted {seeded_price:.3e}, off by {seeded_error:.2}x, gap {seeded_gap:.3e}\n\
         calibrated: predicted {calibrated_price:.3e}, off by {calibrated_error:.2}x, gap \
         {calibrated_gap:.3e}\n\
         {wrongness:?} -> factor {factor:.3}\n{}",
        plan.fingerprint(),
        snapshot.describe()
    );
    Trial {
        ratio_win: calibrated_error < seeded_error,
        absolute_win: calibrated_gap < seeded_gap,
        report,
    }
}

/// How many paired trials one verdict is taken over.
///
/// **A sign test, and the count is derived from a measured failure rate rather
/// than picked.** One trial is a coin the machine can flip: the fit and the
/// held-out run are milliseconds apart, and if the box's load changes across
/// that gap the calibration is judged against a machine that is not the one it
/// was fitted to. Under a synthetic 30-thread load switched on and off every two
/// seconds — the shape a box with six other workers on it actually has — **1080
/// paired trials** gave a per-trial failure rate of **0.09%** on the ratio
/// measure and **2.3%** on the difference, with the worst block of nine scoring
/// 9/9 and 7/9 respectively. Under *sustained* load of 30 the rate was **0/200
/// on both**: steady contention slows the fit and the held-out run alike and the
/// relationship survives it, which is why widening a tolerance would have been
/// the wrong repair — there is no tolerance to widen, only a coincidence in time
/// to average out.
///
/// Eleven, with a bare majority required, so the verdict survives five bad
/// coins. At the measured rates that is a failure probability under `1e-8`, and
/// it costs about 80 ms. It is not a weakened assertion: each individual trial's
/// criterion is still "strictly closer", and a control that is actually wrong
/// loses nearly every trial rather than a few — measured in
/// [`the_comparison_rejects_a_model_wronger_than_the_seed`].
///
/// **The condition that actually broke this file was not external load, it was
/// eight copies of it at once**, which is what a full `cargo test` sweep is. The
/// three repairs — the plan held fixed, one observation judging both models, and
/// the control's factor derived from the seed's own error — were each measured
/// against that condition on a 40-core host at a load average of 44 to 52, 158
/// GB free:
///
/// ```text
///   before  8 concurrent whole-file runs x 5 rounds | 22 of 40 failed
///   after   8 concurrent whole-file runs x 8 rounds |  0 of 64
///   after  16 concurrent whole-file runs x 4 rounds |  0 of 64
/// ```
///
/// That is about 4200 trials with nothing to report. **Nothing in it is a
/// widened tolerance**: every individual comparison is still strict, and the
/// two controls below still have to flip.
const TRIALS: usize = 11;

/// The verdict of one block of trials.
struct Verdict {
    ratio_wins: usize,
    absolute_wins: usize,
    last: String,
}

impl Verdict {
    /// Whether the calibrated model won the scale-free comparison in most
    /// trials. **The measure that means what it says**: it is a ratio between a
    /// prediction and an observation, so it is invariant to how fast the machine
    /// happened to be.
    fn ratio_majority(&self) -> bool {
        self.ratio_wins * 2 > TRIALS
    }

    /// The same for the literal difference. Kept because it is the plain
    /// statement of the property and because it is the only one of the two that
    /// rejects a model over-priced by less than the seed is under-priced — but
    /// it is **not** scale-free, so nothing that has to hold under load rests on
    /// it alone.
    fn absolute_majority(&self) -> bool {
        self.absolute_wins * 2 > TRIALS
    }
}

fn verdict_over(plan: &Decomposition, wrongness: Wrongness) -> Verdict {
    let mut verdict = Verdict {
        ratio_wins: 0,
        absolute_wins: 0,
        last: String::new(),
    };
    for _ in 0..TRIALS {
        let trial = paired_trial(plan, wrongness);
        verdict.ratio_wins += usize::from(trial.ratio_win);
        verdict.absolute_wins += usize::from(trial.absolute_win);
        verdict.last = trial.report;
    }
    verdict
}

/// A measured coefficient beats a stale constant.
///
/// Fit on one run, judged on the next — not on the run it was fitted to, which
/// would be a tautology — and both models judged against that one run's
/// stopwatch, pricing one plan that neither of them was allowed to choose. A run
/// is discarded before any of it is measured, so that neither side is paying for
/// cold code and first-touch page faults while the other is not: the store's
/// premise is *repeated* runs, and comparing a cold run against a warm one would
/// measure the warm-up rather than the calibration.
#[test]
fn a_measured_coefficient_predicts_better_than_the_shipped_seed() {
    // One plan, chosen from the shipped constants, and then held fixed for every
    // trial below. A stopwatch never reaches it.
    let plan = plan_for(CostModel::default());

    // Warm-up, recorded by nobody.
    let _ = run(&plan);

    let verdict = verdict_over(&plan, Wrongness::Honest);
    assert!(
        verdict.ratio_majority(),
        "calibration lost the scale-free comparison in {} of {TRIALS} trials\n{}",
        TRIALS - verdict.ratio_wins,
        verdict.last
    );
    assert!(
        verdict.absolute_majority(),
        "calibration lost the absolute comparison in {} of {TRIALS} trials\n{}",
        TRIALS - verdict.absolute_wins,
        verdict.last
    );
}

/// The liveness test beside the one above: the same program, with the fitted
/// coefficients pushed **further from the truth than the shipped seed is**, and
/// the verdict has to flip.
///
/// **Without this, the test above would pass against a calibration that had
/// stopped working**, because almost any number in the right decade beats a
/// constant denominated in a different unit.
///
/// **Both directions, and the scale-free measure is the one asserted.** A model
/// four times further off than the seed has a ratio error of `4 x seeded_error`
/// against the seed's `seeded_error`, whichever side of the truth it is on, so
/// the margin is exactly four and does not depend on the machine. The difference
/// measure is *not* symmetric — a prediction that is too cheap can never be more
/// than the observation away from it, while one that is too dear is unbounded —
/// so it rejects `TooDear` emphatically and `TooCheap` by a margin that narrows
/// as the seed's own error grows. Asserting it in both directions would be
/// asserting arithmetic that is only true on a quiet machine, which is the
/// mistake this whole file has now made once. It is reported instead.
#[test]
fn the_comparison_rejects_a_model_wronger_than_the_seed() {
    let plan = plan_for(CostModel::default());
    let _ = run(&plan);

    for wrongness in [Wrongness::TooDear(4.0), Wrongness::TooCheap(4.0)] {
        let verdict = verdict_over(&plan, wrongness);
        assert!(
            !verdict.ratio_majority(),
            "a model four times further from the truth than the seed still won the scale-free \
             comparison {} times in {TRIALS} ({wrongness:?}); it also won the absolute one {} \
             times\n{}",
            verdict.ratio_wins,
            verdict.absolute_wins,
            verdict.last
        );
    }
}

/// The other half of liveness, and the one that stops the control above from
/// being satisfied by a criterion that rejects everything: a model **half** as
/// far from the truth as the seed must still win.
///
/// This is the resolution limit stated as an assertion. A criterion that
/// rejected this would be asserting something false — a model half as wrong
/// really is better — and it is worth pinning because the obvious way to make
/// [`the_comparison_rejects_a_model_wronger_than_the_seed`] pass is to make
/// nothing ever win.
///
/// `TooCheap` rather than `TooDear` for the reason that test's header gives: on
/// the cheap side both measures agree at any load, and on the dear side the
/// difference measure does not.
#[test]
fn the_comparison_keeps_a_model_less_wrong_than_the_seed() {
    let plan = plan_for(CostModel::default());
    let _ = run(&plan);

    let verdict = verdict_over(&plan, Wrongness::TooCheap(0.5));
    assert!(
        verdict.ratio_majority(),
        "a model half as far from the truth as the seed lost the scale-free comparison {} times \
         in {TRIALS}. The criterion is rejecting things it should keep, which would also make \
         `the_comparison_rejects_a_model_wronger_than_the_seed` pass for the wrong reason.\n{}",
        TRIALS - verdict.ratio_wins,
        verdict.last
    );
}

// ------------------------------------------------------------------------
// Empty is today.
// ------------------------------------------------------------------------

/// The non-negotiable one: with nothing recorded, the planner sees exactly the
/// model it was handed, and plans exactly the plan it planned before this module
/// existed.
#[test]
fn an_absent_store_plans_exactly_as_the_shipped_constants_do() {
    let workflow = workflow();
    let strategy = Enumerating::default();
    let seeded = CostModel::default();

    let untouched = strategy
        .decompose(&workflow, &constraints(seeded.clone()))
        .expect("a plan");

    for snapshot in [
        Snapshot::default(),
        Snapshot::empty(MachineKey::detect()),
        // a store that exists but holds nothing for this machine
        Statistics::new().snapshot_here(),
    ] {
        assert!(snapshot.is_empty());
        let model = snapshot.calibrate(&seeded);
        assert_eq!(model.clone(), seeded, "an empty snapshot changed the model");
        let planned = strategy
            .decompose(&workflow, &constraints(model.clone()))
            .expect("a plan");
        assert_eq!(planned.fingerprint(), untouched.fingerprint());
        assert_eq!(planned, untouched);
    }
}

/// A store written by another machine is on disk, is read back, and is still not
/// evidence about this one.
#[test]
fn a_store_from_a_different_machine_does_not_change_this_machine_s_plan() {
    let here = MachineKey::detect();
    let there = MachineKey {
        host: format!("{}-somewhere-else", here.host),
        ..here.clone()
    };

    // A run's worth of real observations, relabelled as another machine's.
    let observed = plan_and_run(CostModel::default()).recorder.observations();
    let mut foreign = observed.clone();
    foreign.machine = there.clone();
    assert!(!foreign.is_empty());

    let path = scratch("foreign");
    let mut store = Statistics::new();
    store.record(&foreign);
    store.save(&path).expect("a save");

    let reloaded = Statistics::load(&path).expect("a load");
    assert_eq!(reloaded.machines(), vec![there.clone()]);
    let here_snapshot = reloaded.snapshot(&here);
    assert!(
        here_snapshot.is_empty(),
        "a foreign store was used as evidence here:\n{}",
        here_snapshot.describe()
    );
    assert_eq!(
        here_snapshot.calibrate(&CostModel::default()),
        CostModel::default()
    );
    // and it is perfectly good evidence about the machine it is actually about
    assert!(!reloaded.snapshot(&there).is_empty());
    let _ = std::fs::remove_file(&path);
}

// ------------------------------------------------------------------------
// Reproducibility.
// ------------------------------------------------------------------------

/// Two plans of one workflow against one snapshot are the same plan, and the
/// identity says so. A *different* snapshot is a different identity whether or
/// not the plan moved — which is the point: the question "why did these two runs
/// plan differently" has to stay answerable, and it is answered by the pair.
#[test]
fn one_snapshot_plans_one_way_and_the_identity_shows_it() {
    let observations = plan_and_run(CostModel::default()).recorder.observations();
    let mut store = Statistics::new();
    store.record(&observations);
    let snapshot = store.snapshot_here();

    let workflow = workflow();
    let strategy = Enumerating::default();
    let model = snapshot.calibrate(&CostModel::default());

    let first = strategy
        .decompose(&workflow, &constraints(model.clone()))
        .expect("a plan");
    let second = strategy
        .decompose(&workflow, &constraints(model.clone()))
        .expect("a plan");
    assert_eq!(first, second);
    assert_eq!(
        PlanIdentity::of(&first, &snapshot),
        PlanIdentity::of(&second, &snapshot)
    );

    // Taking the snapshot again off an unchanged store gives the same identity:
    // freezing is a value, not an event.
    assert_eq!(store.snapshot_here().fingerprint(), snapshot.fingerprint());
    assert_eq!(
        PlanIdentity::of(&first, &store.snapshot_here()),
        PlanIdentity::of(&first, &snapshot)
    );

    // More evidence is a different snapshot, and therefore a different identity
    // for the very same decomposition. The plan may or may not move; the
    // identity records that the evidence did.
    store.record(&observations);
    let later = store.snapshot_here();
    assert_ne!(later.fingerprint(), snapshot.fingerprint());
    let same_plan = PlanIdentity::of(&first, &later);
    assert_eq!(same_plan.decomposition, first.fingerprint());
    assert_ne!(same_plan, PlanIdentity::of(&first, &snapshot));
    assert_ne!(
        same_plan.digest(),
        PlanIdentity::of(&first, &snapshot).digest()
    );
}

// ------------------------------------------------------------------------
// What a run actually yields.
// ------------------------------------------------------------------------

/// The recorder derives every coefficient the cost model can hold from the
/// events a run already emits, and nothing it derives is a timing of one op at
/// one filter size.
#[test]
fn a_run_yields_the_coefficients_the_model_is_made_of() {
    let outcome = plan_and_run(CostModel::default());
    let mut store = Statistics::new();
    store.record(&outcome.recorder.observations());
    let snapshot = store.snapshot_here();
    let report = snapshot.describe();

    for term in [Term::Read, Term::Compute, Term::ReadBytes] {
        let coefficient = snapshot
            .coefficient(&term)
            .unwrap_or_else(|| panic!("no {term:?} in\n{report}"));
        assert!(coefficient.nanos_per_unit > 0.0, "{report}");
        assert!(coefficient.units > 0.0, "{report}");
        assert!(coefficient.total_nanos > 0.0, "{report}");
        assert_eq!(coefficient.runs, 1);
        // **Derived, and not yet believed.** One run yields every coefficient
        // the model is made of — which is what this test is about — and
        // `Snapshot::calibrate` still uses none of them, because a measurement
        // that has not been reproduced is a fact about one run. The two are
        // separate claims and this asserts both.
        assert_eq!(
            snapshot.provenance(&term),
            Provenance::Unreproduced { runs: 1 },
            "{report}"
        );
        assert!(snapshot.believable(&term).is_none(), "{report}");
    }
    assert_eq!(
        snapshot.calibrate(&CostModel::default()),
        CostModel::default(),
        "a single run calibrated the model:\n{report}"
    );

    // The plan has more than one phase or it does not; either way exactly one of
    // the two write destinations must have been used, and whichever it is, the
    // other keeps its seeded ratio rather than a number nobody measured.
    let wrote_output = snapshot.coefficient(&Term::Write).is_some();
    let wrote_intermediate = snapshot.coefficient(&Term::Materialise).is_some();
    assert!(wrote_output, "a run wrote no output at all:\n{report}");
    assert_eq!(
        wrote_intermediate,
        outcome.stats.phases > 1,
        "materialisation and phase count disagree:\n{report}"
    );

    // Every op in the chain contributed to a family, and the families are what
    // the single `compute_scale` has to blend.
    let families = snapshot.families();
    for name in ["scale", "smooth", "median"] {
        assert!(families.contains_key(name), "no family {name} in\n{report}");
    }
    assert!(snapshot.family_spread().is_some(), "{report}");
}

/// The coefficient generalises over the parameter, which is the reason it is a
/// coefficient and not a timing: a 27-voxel element and a 125-voxel one declare
/// 27 and 125 units, so the *same* number is expected to predict both.
#[test]
fn one_coefficient_covers_two_filter_sizes() {
    let shape = [24, 24, 32];
    let mut store = Statistics::new();
    let mut declared = Vec::new();
    for edge in [3usize, 5] {
        let element =
            StructuringElement::from_size(ElementShape::Box, [edge, edge, edge]).expect("element");
        let chain = Chain::op(RankFilterOp::median("median", element));
        declared.push(chain.cost_per_voxel());
        let workflow = Workflow::new(chain, shape, Dtype::F64);
        let strategy = Enumerating::default();
        let decomposition = strategy
            .decompose(&workflow, &constraints(CostModel::default()))
            .expect("a plan");
        let mut array = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
        for (flat, value) in array.iter_mut().enumerate() {
            *value = ((flat * 7919) % 1013) as f64;
        }
        let env = ArrayEnvironment::for_decomposition(array.into(), &decomposition, [8, 8, 8])
            .expect("an environment");
        let recorder = Arc::new(Recorder::new(&workflow.chain, &decomposition));
        let listeners: [Arc<dyn EventListener>; 1] = [recorder.clone()];
        strategy
            .run_observed(&workflow, &decomposition, &env, &listeners)
            .expect("a run");
        store.record(&recorder.observations());
    }

    // The two ops declare very different costs, and that difference is the
    // element size — it is stated by the op, not learned from the store.
    assert!(declared[1] > 3.0 * declared[0], "{declared:?}");

    // Both runs fed one coefficient, keyed by the family rather than by the
    // filter size, and the store holds one entry rather than two.
    let snapshot = store.snapshot_here();
    let families = snapshot.families();
    assert_eq!(
        families.keys().collect::<Vec<_>>(),
        vec!["median"],
        "{}",
        snapshot.describe()
    );
    let coefficient = families["median"];
    assert_eq!(coefficient.runs, 2, "{}", snapshot.describe());
    assert!(coefficient.nanos_per_unit > 0.0);
}

// ------------------------------------------------------------------------
// Persistence.
// ------------------------------------------------------------------------

/// Write a store, read it back, identical coefficients — through a real file,
/// with coefficients that came off a real run rather than out of a literal.
#[test]
fn a_store_round_trips_through_a_file() {
    let outcome = plan_and_run(CostModel::default());
    let mut store = Statistics::new();
    store.record(&outcome.recorder.observations());

    let path = scratch("round-trip");
    let _ = std::fs::remove_file(&path);
    store.save(&path).expect("a save");
    let reloaded = Statistics::load(&path).expect("a load");

    assert_eq!(reloaded, store);
    let before = store.snapshot_here();
    let after = reloaded.snapshot_here();
    assert_eq!(before.fingerprint(), after.fingerprint());
    for key in [
        Term::Read,
        Term::Write,
        Term::Materialise,
        Term::Compute,
        Term::ReadBytes,
        Term::WriteBytes,
    ] {
        assert_eq!(
            before.coefficient(&key),
            after.coefficient(&key),
            "{key:?} moved across the file"
        );
    }
    assert_eq!(before.families(), after.families());
    // and the model built from either is the same model
    assert_eq!(
        before.calibrate(&CostModel::default()),
        after.calibrate(&CostModel::default())
    );
    let _ = std::fs::remove_file(&path);
}

/// The numbers behind the assertions, printed rather than asserted.
///
/// Ignored for the reason `ops::cost`'s table is: a figure printed in a test
/// suite is a measurement of the machine's mood. It is here because the
/// assertions above are deliberately about *relationships*, and somebody
/// reading them will want to see the magnitudes at least once:
///
/// ```text
/// cargo test --release --test statistics -- --ignored --nocapture
/// ```
#[test]
#[ignore = "a measurement, not an assertion"]
fn print_what_calibration_changed() {
    let seeded = CostModel::default();
    let plan = plan_for(seeded.clone());
    let _ = run(&plan);

    // One run first, to show what a store that cannot yet be believed looks
    // like, and then enough to be believed. The difference between the two
    // tables is the mechanism `REPRODUCTIONS` describes.
    let mut store = Statistics::new();
    let fit = run(&plan);
    store.record(&fit.recorder.observations());
    println!("after one run:\n{}", store.snapshot_here().describe());
    println!(
        "  calibrates to {:?}\n",
        store.snapshot_here().calibrate(&seeded)
    );

    for _ in 1..REPRODUCTIONS {
        store.record(&run(&plan).recorder.observations());
    }
    let snapshot = store.snapshot_here();
    let calibrated = snapshot.calibrate(&seeded);
    let observed = run(&plan).observed;

    println!("after {REPRODUCTIONS} runs:\n{}", snapshot.describe());
    println!("seeded model     {seeded:?}");
    println!("calibrated model {calibrated:?}");
    println!(
        "plan {:016x}, {} phase(s)",
        plan.fingerprint(),
        fit.stats.phases
    );
    println!("held-out observation {observed:.4e} ns");
    for (name, model) in [
        ("seeded    ", seeded.clone()),
        ("calibrated", calibrated.clone()),
    ] {
        let predicted = price(&plan, &model);
        println!(
            "{name}: predicted {predicted:.4e}  off by {:.2}x  gap {:.4e}",
            error(predicted, observed),
            (predicted - observed).abs()
        );
    }
    println!("family spread {:?}", snapshot.family_spread());

    // The plan the fitted model *would* have chosen, printed and not asserted.
    // It is allowed to differ, and `paired_trial`'s header is about why an
    // earlier version of this file asserted that it could not.
    let replanned = plan_for(calibrated.clone());
    println!(
        "the fitted model would plan {:016x} ({})",
        replanned.fingerprint(),
        if replanned == plan {
            "the same plan"
        } else {
            "a different plan"
        }
    );
}

fn scratch(what: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "blockflow-statistics-{what}-{}.json",
        std::process::id()
    ))
}
