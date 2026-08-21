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

use blockflow::decomposition::{predicted_cost, Constraints, CostModel};
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

/// Plan, run, and hand back what the run observed together with what the plan
/// predicted under the model it was planned with.
struct Outcome {
    predicted: f64,
    observed: f64,
    stats: Stats,
    recorder: Arc<Recorder>,
    /// Which decomposition was run. Carried because
    /// [`paired_trial`] judges two *models* against one measurement, which is
    /// only a fair comparison while both models plan the same thing.
    plan: u64,
}

fn plan_and_run(model: CostModel) -> Outcome {
    let workflow = workflow();
    let strategy = Enumerating::default();
    let decomposition = strategy
        .decompose(&workflow, &constraints(model))
        .expect("a plan");
    let predicted = predicted_cost(&workflow.chain, &decomposition, &[], &model).expect("a price");
    let env = ArrayEnvironment::for_decomposition(input(), &decomposition, [8, 8, 8])
        .expect("an environment");
    let recorder = Arc::new(Recorder::new(&workflow.chain, &decomposition));
    let listeners: [Arc<dyn EventListener>; 1] = [recorder.clone()];
    let stats = strategy
        .run_observed(&workflow, &decomposition, &env, &listeners)
        .expect("a run");
    let observed = observed_nanos(&stats.log);
    Outcome {
        predicted,
        observed,
        stats,
        recorder,
        plan: decomposition.fingerprint(),
    }
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

/// What one paired trial decided.
struct Trial {
    /// Whether the calibrated model beat the seed on the scale-free measure.
    ratio_win: bool,
    /// Whether it beat the seed on the literal difference.
    absolute_win: bool,
    report: String,
}

/// One fit, one held-out run, and both models judged against it.
///
/// **Why both models are judged against the *same* observation.** The property
/// is "the measured coefficient predicts this run better than the seed did", and
/// that is one run and two predictions, not two runs. The earlier form compared
/// the seed against the fit run's own stopwatch and the calibration against the
/// held-out run's, so the two errors had different denominators and a change in
/// the machine's speed between them landed entirely on the calibration. The seed
/// is a constant: it predicts the held-out run exactly as well as it predicted
/// anything else, and `predicted_cost` of the same plan under the same model is
/// the same number, so nothing is lost by asking it about the run that is
/// actually being predicted.
///
/// That is sound **only while both models plan the same decomposition**, and it
/// is asserted rather than assumed. Today no model can move it here: `constraints`
/// builds on `Enumerating::default()`, whose `concurrency` of `1` is that type's
/// own documented negative control — the objective collapses to the serial work
/// total, which falls monotonically as the block grows, so every candidate list
/// answers with its largest entry whatever the coefficients are. The assertion is
/// there because that is a property of the fixture rather than of calibration,
/// and a fixture that gained a pool would silently invalidate the comparison.
///
/// `miscalibration` is the negative control's one changed thing: `1.0` is the
/// honest measurement, anything else is the same program with the fitted
/// coefficients scaled off the truth by that factor. See
/// [`the_comparison_rejects_a_model_that_is_ten_times_wrong`].
fn paired_trial(miscalibration: f64) -> Trial {
    let seeded = CostModel::default();

    // The fit. Its prediction under the seeded model is also the seed's
    // prediction of the held-out run, because the plan is the same one.
    let fit = plan_and_run(seeded);
    let mut store = Statistics::new();
    store.record(&fit.recorder.observations());
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
    if miscalibration != 1.0 {
        calibrated.read_cost_per_voxel *= miscalibration;
        calibrated.write_cost_per_voxel *= miscalibration;
        calibrated.materialise_cost_per_voxel *= miscalibration;
        calibrated.compute_scale *= miscalibration;
    }

    // The held-out run, under the calibrated plan.
    let evaluated = plan_and_run(calibrated);
    assert_eq!(
        fit.plan, evaluated.plan,
        "calibration moved this chain's plan, so one observation no longer \
         judges both models; see `paired_trial`"
    );

    let observed = evaluated.observed;
    let seeded_error = error(fit.predicted, observed);
    let calibrated_error = error(evaluated.predicted, observed);
    let seeded_gap = (fit.predicted - observed).abs();
    let calibrated_gap = (evaluated.predicted - observed).abs();
    let report = format!(
        "observed {observed:.3e}\n\
         seeded:     predicted {:.3e}, off by {seeded_error:.2}x, gap {seeded_gap:.3e}\n\
         calibrated: predicted {:.3e}, off by {calibrated_error:.2}x, gap {calibrated_gap:.3e}\n\
         miscalibration {miscalibration}\n{}",
        fit.predicted,
        evaluated.predicted,
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
/// held-out run are about 3 ms apart, and if the box's load changes across that
/// gap the calibration is judged against a machine that is not the one it was
/// fitted to. Under a synthetic 30-thread load switched on and off every two
/// seconds — the shape a box with six other workers on it actually has —
/// **1080 paired trials** gave a per-trial failure rate of **0.09%** on the
/// ratio measure and **2.3%** on the difference, with the worst block of nine
/// scoring 9/9 and 7/9 respectively. Under *sustained* load of 30 the rate was
/// **0/200 on both**: steady contention slows the fit and the held-out run alike
/// and the relationship survives it, which is why widening a tolerance would
/// have been the wrong repair — there is no tolerance to widen, only a
/// coincidence in time to average out.
///
/// Eleven, with a bare majority required, so the verdict survives five bad
/// coins. At the measured rates that is a failure probability under `1e-8`, and
/// it costs about 80 ms. It is not a weakened assertion: each individual trial's
/// criterion is still "strictly closer", and a calibration that is actually
/// broken loses nearly every trial rather than a few — measured in
/// [`the_comparison_rejects_a_model_that_is_ten_times_wrong`].
const TRIALS: usize = 11;

/// A measured coefficient beats a stale constant.
///
/// Fit on one run, judged on the next — not on the run it was fitted to, which
/// would be a tautology. A run is discarded before any of it is measured, so
/// that neither side is paying for cold code and first-touch page faults while
/// the other is not: the store's premise is *repeated* runs, and comparing a
/// cold run against a warm one would measure the warm-up rather than the
/// calibration.
#[test]
fn a_measured_coefficient_predicts_better_than_the_shipped_seed() {
    // Warm-up, recorded by nobody.
    let _ = plan_and_run(CostModel::default());

    let mut ratio_wins = 0;
    let mut absolute_wins = 0;
    let mut last = String::new();
    for _ in 0..TRIALS {
        let trial = paired_trial(1.0);
        ratio_wins += usize::from(trial.ratio_win);
        absolute_wins += usize::from(trial.absolute_win);
        last = trial.report;
    }
    assert!(
        ratio_wins * 2 > TRIALS,
        "calibration lost the scale-free comparison in {} of {TRIALS} trials\n{last}",
        TRIALS - ratio_wins
    );
    assert!(
        absolute_wins * 2 > TRIALS,
        "calibration lost the absolute comparison in {} of {TRIALS} trials\n{last}",
        TRIALS - absolute_wins
    );
}

/// The liveness test beside the one above: the same program, with the fitted
/// coefficients moved off the truth by one factor, and the verdict has to flip.
///
/// **Without this, the test above would pass against a calibration that had
/// stopped working**, because the shipped seed is only 2.4x off on this machine
/// and almost any number in the right decade beats it.
///
/// **The measured resolution, in both directions, over 20 blocks of nine trials
/// each.** Wins per block, worst case:
///
/// ```text
///   factor | ratio wins | absolute wins | rejected by
///    0.1   |   0-4 / 9  |    0-4 / 9    | both
///    0.5   |   9   / 9  |    9   / 9    | neither — and correctly so
///    2     |   8-9 / 9  |    0-4 / 9    | the difference only
///   10     |   0-3 / 9  |    0   / 9    | both
/// ```
///
/// So the two measures are kept because they catch different things, and the
/// factors asserted here are the ones both reject. **`0.5` is not a gap in the
/// test.** A model half the truth genuinely *is* closer to a `2.31e7` ns
/// observation than a seed predicting `9.87e6`, so a criterion that rejected it
/// would be asserting something false. The resolution this criterion has is
/// about a factor of two, which is where `CostModel` itself runs out: two
/// identical voxelwise maps at different positions in one chain measure
/// **1.6x** apart, and nothing denominated in voxelwise maps can claim better.
#[test]
fn the_comparison_rejects_a_model_that_is_ten_times_wrong() {
    let _ = plan_and_run(CostModel::default());

    for miscalibration in [10.0_f64, 0.1] {
        let mut ratio_wins = 0;
        let mut absolute_wins = 0;
        let mut last = String::new();
        for _ in 0..TRIALS {
            let trial = paired_trial(miscalibration);
            ratio_wins += usize::from(trial.ratio_win);
            absolute_wins += usize::from(trial.absolute_win);
            last = trial.report;
        }
        assert!(
            ratio_wins * 2 <= TRIALS,
            "a model {miscalibration}x off the measurement still won the \
             scale-free comparison {ratio_wins} times in {TRIALS}\n{last}"
        );
        assert!(
            absolute_wins * 2 <= TRIALS,
            "a model {miscalibration}x off the measurement still won the \
             absolute comparison {absolute_wins} times in {TRIALS}\n{last}"
        );
    }
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
        .decompose(&workflow, &constraints(seeded))
        .expect("a plan");

    for snapshot in [
        Snapshot::default(),
        Snapshot::empty(MachineKey::detect()),
        // a store that exists but holds nothing for this machine
        Statistics::new().snapshot_here(),
    ] {
        assert!(snapshot.is_empty());
        let model = snapshot.calibrate(&seeded);
        assert_eq!(model, seeded, "an empty snapshot changed the model");
        let planned = strategy
            .decompose(&workflow, &constraints(model))
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
        .decompose(&workflow, &constraints(model))
        .expect("a plan");
    let second = strategy
        .decompose(&workflow, &constraints(model))
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
        assert_eq!(coefficient.runs, 1);
        assert_eq!(
            snapshot.provenance(&term),
            Provenance::Measured {
                runs: 1,
                units: coefficient.units
            }
        );
    }

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
    let _ = plan_and_run(seeded);
    let first = plan_and_run(seeded);
    let mut store = Statistics::new();
    store.record(&first.recorder.observations());
    let snapshot = store.snapshot_here();
    let calibrated = snapshot.calibrate(&seeded);
    let second = plan_and_run(calibrated);
    println!("{}", snapshot.describe());
    println!("seeded model     {seeded:?}");
    println!("calibrated model {calibrated:?}");
    println!(
        "seeded:     predicted {:.4e}  observed {:.4e}  off by {:.1}x",
        first.predicted,
        first.observed,
        error(first.predicted, first.observed)
    );
    println!(
        "calibrated: predicted {:.4e}  observed {:.4e}  off by {:.1}x",
        second.predicted,
        second.observed,
        error(second.predicted, second.observed)
    );
    println!("family spread {:?}", snapshot.family_spread());
    println!("phases {} -> {}", first.stats.phases, second.stats.phases);
}

fn scratch(what: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "blockflow-statistics-{what}-{}.json",
        std::process::id()
    ))
}
