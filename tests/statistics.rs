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
// What is deliberately *not* asserted
// -----------------------------------
// **No test here asserts on wall-clock time.** Not "the run took under a
// second", not "the calibrated plan was faster". Measurements are the subject
// of these tests, never the assertion: everything below is a relationship
// between two recorded numbers — a prediction against an observation, a
// fingerprint against a fingerprint — which is exactly as true on a loaded
// machine as on an idle one.
//
// The error measure is the **ratio**, not the difference, and that is not a
// weakening. Predicted cost spans orders of magnitude between an uncalibrated
// model (denominated in voxelwise maps) and a calibrated one (denominated in
// nanoseconds), and an absolute difference between two such numbers is
// dominated by whichever is larger rather than by which is more nearly right.
// The difference form is asserted too, since it is the literal statement of the
// property, but the ratio is the one that means what it says.

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
}

fn plan_and_run(model: CostModel) -> Outcome {
    let workflow = workflow();
    let strategy = Enumerating::default();
    let decomposition = strategy
        .decompose(&workflow, &constraints(model))
        .expect("a plan");
    let predicted = predicted_cost(&workflow.chain, &decomposition, &model).expect("a price");
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
    }
}

/// How wrong a prediction is, scale-free. `1.0` is exact.
fn error(predicted: f64, observed: f64) -> f64 {
    assert!(predicted > 0.0 && observed > 0.0, "nothing was measured");
    predicted.max(observed) / predicted.min(observed)
}

// ------------------------------------------------------------------------
// The property the whole module exists for.
// ------------------------------------------------------------------------

/// A measured coefficient beats a stale constant.
///
/// Fit on one run, evaluated against another — not against the run it was fitted
/// to, which would be a tautology. The first run is discarded before either is
/// measured, so that neither side is paying for cold code and first-touch page
/// faults while the other is not: the store's premise is *repeated* runs, and
/// comparing a cold run against a warm one would measure the warm-up rather than
/// the calibration.
#[test]
fn a_measured_coefficient_predicts_better_than_the_shipped_seed() {
    let seeded = CostModel::default();

    // Warm-up, recorded by nobody.
    let _ = plan_and_run(seeded);

    // The seeded plan, and what it really cost.
    let first = plan_and_run(seeded);

    // The evidence, and the plan it produces.
    let mut store = Statistics::new();
    store.record(&first.recorder.observations());
    let snapshot = store.snapshot_here();
    assert!(
        !snapshot.is_empty(),
        "one run produced no coefficients:\n{}",
        snapshot.describe()
    );
    let calibrated = snapshot.calibrate(&seeded);
    assert_ne!(
        calibrated,
        seeded,
        "a snapshot with evidence left the model untouched:\n{}",
        snapshot.describe()
    );

    let second = plan_and_run(calibrated);

    let before = error(first.predicted, first.observed);
    let after = error(second.predicted, second.observed);
    let report = format!(
        "seeded:     predicted {:.3e} against observed {:.3e}, off by {before:.1}x\n\
         calibrated: predicted {:.3e} against observed {:.3e}, off by {after:.1}x\n{}",
        first.predicted,
        first.observed,
        second.predicted,
        second.observed,
        snapshot.describe()
    );
    assert!(after < before, "calibration did not help\n{report}");
    assert!(
        (second.predicted - second.observed).abs() < (first.predicted - first.observed).abs(),
        "the calibrated prediction is not closer in absolute terms\n{report}"
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
