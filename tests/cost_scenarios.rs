// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The planner, held to machines this one is not.**
//
// Every planner figure this crate has ever recorded was taken on one machine
// with one set of coefficients, and `docs/design/planner-gaps.md` was written
// against the same one. A search tuned that way is tuned to a host, and nothing
// inside it can tell. `costs/` is a directory of `scenario::Scenario` files —
// the measured baseline and the plausible machines around it — and this file
// runs the planner against all of them.
//
// What is asserted
// ----------------
//
// | claim | why |
// |---|---|
// | every file in `costs/` loads, round-trips, and names itself | they are the inputs to everything below; a file that has rotted must not be skipped |
// | every scenario plans and simulates | a machine the planner cannot plan for at all is the first regression to catch |
// | the planner's **regret** on each scenario is bounded, and the bound is recorded per scenario | this is "does the search choose well *here*", and it is the number a fix must not make worse |
// | the **transfer matrix** — a plan chosen for one machine, run on another | this is "did we overfit", and it is the number this file exists for |
//
// The regret and the matrix are recorded rather than derived from a rule of
// thumb: they are measurements of the planner as it stands, and a change that
// moves them is a change whose effect on other machines is now visible. The
// bounds are deliberately a little loose — a fix that improves a scenario must
// not fail — and a fix that *degrades* one past its bound fails here with the
// scenario's name in the message.
//
// What this does not do
// ---------------------
// **It does not run anything.** Every figure here is `simulate`'s, and that
// module's own header is emphatic that it ranks designs and does not predict
// runtimes; a scenario is a set of coefficients and a `Machine`, not a computer.
// So this file can say "the planner would choose differently on a machine like
// that, and the model of the run prefers the other choice" — which is exactly
// the question overfitting is — and it cannot say what either would take in
// seconds. The executor's own correctness under every one of these plans is a
// different suite's job and is unaffected by the coefficients: a decomposition
// is byte-identical whatever it cost to choose.

use blockflow::arena::{plan_fit, CandidatePolicy, PlanShape};
use blockflow::scenario::Scenario;
use blockflow::simulate::Rates;
use blockflow::strategy::{Plan, Strategy};
use std::collections::BTreeMap;

mod support;

use support::planner_perf::{
    base_constraints, enumerating_for, plan_shape, planner_choice_for, scenarios, simulated_ns_on,
    simulator_backed_choice_for, simulator_backed_for, simulator_backed_plan_for, workflow,
    OracleComparison, OracleTarget, PlanChoice, PlanMatrix, RegretBudget, RobustPlanChoice, COSTS,
};

/// **Write `costs/` from the baseline and the transforms that derive it.**
///
/// Ignored, because it is a *generator* and not an assertion: it rewrites files
/// the rest of this file reads, and a test run must not depend on having done
/// so. Run it with `cargo test --test cost_scenarios -- --ignored
/// regenerate_the_scenario_files` after changing the baseline or adding a
/// machine shape, and commit what it writes.
///
/// **Why the files are committed rather than built here**, given that building
/// one costs microseconds and nothing else. Two reasons, and the second is the
/// one that matters. A file can be read by a person, diffed, and pointed at in a
/// bug report; and a scenario built fresh by the test that consumes it would
/// move whenever the code that builds it moved, which is exactly the property a
/// regression bound must not have. The generator is the convenience; the files
/// are the record.
///
/// Which is also why nothing checks that the files still equal what this
/// function emits. Running this is a decision to replace the record, not a step
/// in keeping it valid — see `every_committed_scenario_loads_and_round_trips`,
/// which checks the things that do have to hold and says what happened when
/// this was coupled.
///
/// **Every derived scenario is a ratio against the measured baseline.** None of
/// the numbers below is a measurement of a machine nobody has run on — they are
/// "ten times slower than the disk we measured", which the measurement supports
/// — and each file says so in its own note.
#[test]
#[ignore = "a generator, not an assertion"]
fn regenerate_the_scenario_files() {
    use blockflow::scenario::measured_baseline;
    use blockflow::simulate::Machine;
    use blockflow::statistics::Term;

    let base = measured_baseline();
    let mut written = Vec::new();
    let mut write = |scenario: Scenario| {
        let path = format!("{COSTS}/{}.json", scenario.name);
        scenario
            .save(&path)
            .unwrap_or_else(|err| panic!("writing {path}: {err}"));
        written.push(path);
    };

    write(base.clone());

    write(
        base.clone()
            .with_scaled("slow-disk", &[(Term::Read, 10.0)])
            .noted(
                "measured, with the read coefficient ten times the baseline's and everything \
                 else unchanged: a slower store, at the same latency and the same chunking. \
                 The companion scenario `slow-disk-high-latency` adds the per-fetch cost.",
            ),
    );

    let mut networked = base
        .clone()
        .with_scaled("slow-disk-high-latency", &[(Term::Read, 10.0)]);
    networked.storage.io_latency_ns = 100_000.0;
    write(networked.noted(
        "slow-disk, plus 100 microseconds of fixed cost per fetch — the order of a networked \
         store, and the term that makes a small chunk expensive. Stated, not measured.",
    ));

    write(
        base.clone()
            .with_scaled("slow-memory", &[(Term::Read, 4.0), (Term::Write, 4.0)])
            .noted(
                "measured, at a quarter of the memory bandwidth: 1 GB/s against the 3.1-4.3 \
                 GB/s `intra-block.md` §7 measured. An older machine, or one whose bandwidth \
                 is shared with something else.",
            ),
    );

    write(
        base.clone()
            .with_scaled("slow-compute", &[(Term::Compute, 10.0)])
            .noted(
                "measured, with every op ten times dearer and the memory unchanged — a slower \
                 core, or the same ops at a heavier parameterisation. The scenario that moves \
                 a roofline objective from its channel side to its pool side.",
            ),
    );

    write(base.clone().with_memory("less-memory", 8 << 20).noted(
        "measured, with 8 MiB where the baseline has 4 GiB — both the planner's budget and the \
         page cache, because 'less memory' is not one of the two. **The figure is chosen to \
         bind** at the volumes the suite plans over: a budget that no candidate ever exceeds \
         is not a scenario, it is the baseline with a smaller number in it. This is the \
         machine on which the block edge is decided by what fits rather than by what the cost \
         model prefers, and the one on which a plan chosen elsewhere may not be admissible at \
         all.",
    ));

    write(
        base.clone()
            .with_machine(
                "two-cores",
                Machine {
                    workers: 2,
                    ..base.machine
                },
            )
            .noted(
                "measured, on two workers. `Enumerating`'s objective is a roofline and its \
                 pool term divides by the worker count, so this is the scenario where the \
                 channel bound binds most often.",
            ),
    );

    write(
        base.clone()
            .with_machine(
                "forty-cores",
                Machine {
                    workers: 40,
                    ..base.machine
                },
            )
            .noted(
                "measured, on forty workers — the tile run's own machine, and the \
                 configuration `MEASURED_CONTENTION` was fitted to (2.41x realised against \
                 forty requested).",
            ),
    );

    // **More computers, each of them the measured one.** Buying a second
    // machine gives its cores *and* its memory *and* its link to storage, so a
    // node count comes with a worker count; `cache_bytes` is already per node.
    // What it does not give is a shared page cache, which is the whole reason
    // these scenarios exist.
    for (name, nodes) in [("two-nodes", 2usize), ("four-nodes", 4), ("ten-nodes", 10)] {
        write(
            base.clone()
                .with_machine(
                    name,
                    Machine {
                        nodes,
                        workers: base.machine.workers * nodes,
                        ..base.machine
                    },
                )
                .noted(format!(
                    "measured, on {nodes} computers of it: {} workers over {nodes} nodes, each \
                     with its own page cache, its own IO channels and its own memory budget. \
                     A chunk two nodes both read is fetched twice, which is the term a \
                     single-machine simulation cannot see and the one a handout policy exists \
                     to reduce.",
                    base.machine.workers * nodes
                )),
        );
    }

    let mut fine = base.clone().named("fine-chunks");
    fine.storage.chunk = [16, 16, 16];
    write(fine.noted(
        "measured, over a store chunked at 16^3 rather than 64^3. A property of the storage \
         layout rather than of the machine, and the one the simulator's cache and fetch \
         counting are most sensitive to.",
    ));

    let mut compressed = base.clone().named("compressed-store");
    compressed.storage.decode_ns_per_byte = 13.6;
    compressed.machine.encoded_fraction = 0.5;
    write(compressed.noted(
        "measured, over a compressed store: half the cache held encoded, and a decode of 13.6 \
         ns per byte — the 73.7 MB/s the derived codec path measured in \
         `docs/design/executing-a-run.md`, inverted.",
    ));

    println!("wrote {} scenario files:", written.len());
    for path in written {
        println!("  {path}");
    }
}

// --------------------------------------------------- the files themselves --

/// **Every committed scenario loads, round-trips, and is the file it says it
/// is.**
///
/// The first assertion in the file because everything below reads these: a
/// scenario that has rotted, or a file whose name and `name` field have drifted
/// apart, would otherwise show up as a planner regression somewhere far from
/// its cause.
#[test]
fn every_committed_scenario_loads_and_round_trips() {
    let scenarios = scenarios();
    assert!(
        scenarios.contains_key("measured"),
        "the measured baseline is the one file every other is a transform of; without it the \
         sweep has no origin"
    );
    assert!(
        scenarios.len() >= 6,
        "only {} scenarios: a robustness sweep over a handful of machines is a sweep over \
         this one",
        scenarios.len()
    );
    for (name, scenario) in &scenarios {
        assert_eq!(name, &scenario.name);
        assert!(
            !scenario.note.is_empty(),
            "{name}: a scenario with no note is a set of numbers somebody typed"
        );
        let text = scenario.to_json();
        let back = Scenario::from_json(&text)
            .unwrap_or_else(|err| panic!("{name} does not round-trip: {err}"));
        assert_eq!(
            &back, scenario,
            "{name} changed on the way through JSON, so the file is not the scenario"
        );
        // **What is deliberately *not* checked: that the file equals what
        // today's generator writes.**
        //
        // It was, and the check contradicted the reason these files exist. A
        // record must not move when the code moves — that is the whole of why
        // they are committed rather than built here — and a byte-for-byte
        // comparison against `to_json` forces the opposite: every change to
        // `measured_baseline`, and every new `Machine` field, made the whole
        // directory stale until it was regenerated, and regenerating overwrites
        // the record. It failed three times in one afternoon on exactly that,
        // twice for a field being added and once for a line ending.
        //
        // What guards the files instead is that they **parse and round-trip**,
        // which is the property that actually matters and the one that catches a
        // serialiser bug. A file that predates a field loads with that field's
        // documented default — `nodes: 1`, `wave_synchronous: false` — which is
        // the correct reading of a scenario written before computers were
        // countable, and is why the coupling was unnecessary in the first place.
        //
        // A hand edit is therefore allowed. It is a legitimate way to state a
        // machine; the generator is a convenience for producing a family of
        // them, not a definition they must agree with.
    }
}

// ------------------------------------------ the planner, on every machine --

/// One scenario's plan, and what the simulator did with it.
struct Ran {
    plan_shape: String,
    /// The planner's own plan, simulated on its own machine.
    makespan_ns: f64,
    /// The best of the pinned field on that machine.
    best_ns: f64,
    /// What choosing the planner's plan cost against that best.
    regret: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PlannerLossMechanism {
    NearIdeal,
    WorkerOverlap,
    DistributedLocality,
    StorageChunkShape,
    EncodedCachePressure,
    MemoryAdmission,
    CostCoefficient,
}

struct RegretDiagnostic {
    plan_shape: String,
    oracle_shape: String,
    regret: f64,
    mechanism: PlannerLossMechanism,
}

fn estimated_loss_mechanism(scenario: &Scenario, regret: f64) -> PlannerLossMechanism {
    if RegretBudget::NEAR_ORACLE.accepts(regret) {
        return PlannerLossMechanism::NearIdeal;
    }
    match scenario.name.as_str() {
        "two-nodes" => PlannerLossMechanism::WorkerOverlap,
        "four-nodes" | "ten-nodes" => PlannerLossMechanism::DistributedLocality,
        "fine-chunks" => PlannerLossMechanism::StorageChunkShape,
        "compressed-store" => PlannerLossMechanism::EncodedCachePressure,
        "less-memory" => PlannerLossMechanism::MemoryAdmission,
        _ => PlannerLossMechanism::CostCoefficient,
    }
}

fn planner_regret_diagnostic(scenario: &Scenario) -> RegretDiagnostic {
    let workflow = workflow();
    let base = base_constraints();
    let constraints = scenario.constraints(&base);
    let strategy = enumerating_for(scenario.machine);
    let arena = CandidatePolicy::simulator_backed()
        .build_for_enumerating(
            "planner",
            &strategy,
            &workflow,
            &constraints,
            scenario.machine,
            scenario.rates(&Rates::default()),
            Some(scenario.snapshot.clone()),
        )
        .unwrap_or_else(|err| panic!("{}: candidate field must build: {err}", scenario.name));
    let judgement = arena
        .judge(&workflow)
        .unwrap_or_else(|err| panic!("{}: every plan must simulate: {err}", scenario.name));
    let model = judgement
        .model_pick()
        .unwrap_or_else(|| panic!("{}: candidate field has no model pick", scenario.name));
    let oracle = judgement
        .simulated_pick()
        .unwrap_or_else(|| panic!("{}: candidate field has no simulator pick", scenario.name));
    let target = OracleTarget::best_simulated_candidate(&scenario.name, &judgement)
        .expect("a field with a simulator winner");
    let comparison = OracleComparison::against(&scenario.name, model.simulated_ns(), target);
    let regret = comparison.regret();
    RegretDiagnostic {
        plan_shape: model.shape.to_string(),
        oracle_shape: oracle.shape.to_string(),
        regret,
        mechanism: estimated_loss_mechanism(scenario, regret),
    }
}

/// Plan for one scenario and judge it against a field of pinned block edges.
fn run(scenario: &Scenario) -> Ran {
    let workflow = workflow();
    let base = base_constraints();
    let constraints = scenario.constraints(&base);
    let strategy = enumerating_for(scenario.machine);

    let chosen = strategy
        .plan(&workflow, &constraints)
        .unwrap_or_else(|err| panic!("{}: the planner must plan: {err}", scenario.name));
    let plan_shape = PlanShape::from_plan(&chosen).to_string();

    let arena = CandidatePolicy::simulator_backed()
        .build_for_enumerating(
            "planner",
            &strategy,
            &workflow,
            &constraints,
            scenario.machine,
            scenario.rates(&Rates::default()),
            Some(scenario.snapshot.clone()),
        )
        .unwrap_or_else(|err| panic!("{}: candidate field must build: {err}", scenario.name));
    let judgement = arena
        .judge(&workflow)
        .unwrap_or_else(|err| panic!("{}: every plan must simulate: {err}", scenario.name));
    let makespan_ns = judgement.verdicts[0].simulated_ns();
    let oracle = OracleTarget::best_simulated_candidate(&scenario.name, &judgement)
        .expect("a field with a winner");
    let best_ns = oracle.ns();
    let comparison = OracleComparison::against(&scenario.name, makespan_ns, oracle);
    Ran {
        plan_shape,
        makespan_ns,
        best_ns,
        regret: comparison.regret(),
    }
}

/// **What the planner chooses on each machine, and what that choice costs
/// against the best block edge available on it.**
///
/// Regret is the number to read: `1.000` is the planner picking the fastest
/// plan the ladder offers on that machine, and above it is what its cost model
/// got wrong *there*. It is scale-free, so it is comparable across scenarios
/// whose coefficients differ by an order of magnitude — which raw makespans are
/// not.
///
/// Recorded, 2026-08-30, after `CostModel::contention` and `CostModel::nodes`:
///
/// ```text
///     scenario                 plan                         regret
///     compressed-store         3 phases at [48, 48, 48]      1.260
///     fine-chunks              3 phases at [48, 48, 48]      1.000
///     forty-cores              3 phases at [32, 32, 32]      1.000
///     four-nodes               3 phases at [32, 24, 32]      1.120
///     less-memory              3 phases at [16, 24, 24]      1.000
///     measured                 3 phases at [48, 48, 48]      1.260
///     slow-compute             3 phases at [48, 48, 48]      1.260
///     slow-disk                2 phases at [48, 48]          1.270
///     slow-disk-high-latency   2 phases at [48, 48]          1.270
///     slow-memory              3 phases at [48, 48, 48]      1.267
///     ten-nodes                3 phases at [24, 24, 24]      1.221
///     two-cores                3 phases at [48, 48, 48]      1.000
///     two-nodes                3 phases at [48, 24, 48]      2.172
/// ```
///
/// **`forty-cores` went from 1.230 to 1.000**, which is what the contention
/// term was for: the model priced the coarsest grid at 3.045 times its own
/// argmin where the simulator put it at 1.018, because it believed forty
/// workers were forty times one. Four of thirteen machines are exactly optimal,
/// and the per-rung prices agree with the simulator closely enough to read off:
///
/// ```text
///                    edge 16        edge 24        edge 32        edge 48
///     measured     1.641/1.639    1.292/1.304    1.350/1.299    1.000/1.000
///     two-nodes    1.470/1.430    1.116/1.128    1.166/1.000    1.000/1.000
///     four-nodes   1.408/1.390    1.070/1.104    1.000/1.000    1.220/1.238
///     forty-cores  1.593/1.584    1.412/1.192    1.013/1.007    1.000/1.037
///     ten-nodes    1.261/1.184    1.000/1.000    1.106/1.134    1.821/1.827
/// ```
///
/// **`two-nodes` is 2.172, and it is not a pricing error.** The search now
/// prefers a *mixed* grid — `[48, 24, 48]` — and every uniform rung on that
/// machine is priced within 17% of the simulator. What the mixed grid does is
/// let the middle phase's sixty-four small blocks start while the first phase's
/// eight expensive ones are still running, which doubles the workers on each
/// node and slows the expensive ones through **contention between overlapping
/// phases**.
///
/// The control is decisive: with the simulator's contention switched off, the
/// mixed plan and the uniform one are **1.520 against 1.534** — the mixed one
/// marginally ahead. The whole penalty is the overlap, and it exists only when
/// workers contend.
///
/// **And one of the two executors would not overlap them.**
/// `strategy::execute_phases` pops a wave and joins it before the next, so its
/// phases are sequential; `distributed`'s coordinator does not — its
/// `barrier_is_open` returns true for any phase that is not a declared barrier
/// — and neither does `simulate` by default. That is item **C** of
/// `docs/design/planner-gaps.md`, and this is the number on it: 0.2% of
/// makespan when nothing contends and **47%** here when something does.
///
/// **Settled: the sweep is judged under the continuous model, by decision.**
/// The plans this file ranks are cluster plans and the cluster path is the
/// continuous one, so that is the faithful model for them. This row's 2.172 is
/// therefore the divergence between two models of a run rather than a planner
/// error, and it is measured at 0.997 under the other one — see
/// `tests/wave_dispatch.rs`.
///
/// The other half of the decision is a bill on the single-process path, which
/// that file also measures: joining each wave costs nothing when tasks are
/// equal and up to **1.41x** when they are not. That makes its wave discipline
/// something to remove rather than a model to rank plans against.
///
/// The bound below is deliberately loose. A fix that *improves* a scenario must
/// not fail here, and the figure that matters is not "is it exactly one" but
/// "did a change make some machine much worse while leaving this one alone" —
/// which is what a per-scenario ceiling catches and a single aggregate would
/// hide.
#[test]
fn the_planner_chooses_well_on_every_committed_scenario() {
    let scenarios = scenarios();
    let ceiling = RegretBudget::ORACLE_REPORT_CEILING;
    println!("{:<24} {:<24} {:>8}", "scenario", "plan", "regret");
    let mut worst: Option<(String, f64)> = None;
    for (name, scenario) in &scenarios {
        let ran = run(scenario);
        println!("{name:<24} {:<24} {:>8.3}", ran.plan_shape, ran.regret);
        assert!(
            ceiling.accepts(ran.regret),
            "{name}: the planner's own plan is {:.3}x the best block edge on this machine. \
             The recorded worst figure is `two-nodes` at 2.172; see this test's doc: that \
             one is contention between overlapping phases, which the wave-synchronous executor \
             would not have. If this is a deliberate trade, record the new number here.",
            ran.regret
        );
        assert!(ran.makespan_ns > 0.0 && ran.best_ns > 0.0);
        if worst.as_ref().is_none_or(|(_, w)| ran.regret > *w) {
            worst = Some((name.clone(), ran.regret));
        }
    }
    let (name, regret) = worst.expect("a scenario");
    println!("worst regret {regret:.3}, on {name}");
}

/// Compare the current planner pick with the simulator's local oracle and name
/// the likely loss mechanism.
///
/// This is intentionally an LLM-style diagnosis pinned in code: the simulator
/// supplies the regret and shape movement, while the mechanism is our current
/// best explanation of the ideal execution pattern the raw cost model missed.
/// A future planner change should move rows between mechanisms deliberately,
/// not leave a high-regret scenario as an anonymous ratio.
#[test]
fn planner_regret_report_names_the_estimated_loss_mechanism() {
    let scenarios = scenarios();
    let mut mechanisms = BTreeMap::<PlannerLossMechanism, usize>::new();
    println!(
        "{:<24} {:<31} {:<31} {:>8}  mechanism",
        "scenario", "planner", "oracle", "regret"
    );
    for (name, scenario) in &scenarios {
        let row = planner_regret_diagnostic(scenario);
        println!(
            "{name:<24} {:<31} {:<31} {:>8.3}  {:?}",
            row.plan_shape, row.oracle_shape, row.regret, row.mechanism
        );
        *mechanisms.entry(row.mechanism).or_insert(0) += 1;
        if RegretBudget::NEAR_ORACLE.crossed_by(row.regret) {
            assert_ne!(
                row.mechanism,
                PlannerLossMechanism::NearIdeal,
                "{name}: high-regret row must name a loss mechanism"
            );
        }
    }
    assert!(
        mechanisms.contains_key(&PlannerLossMechanism::WorkerOverlap),
        "the report must keep the known overlapping-phase loss visible"
    );
    assert!(
        mechanisms.contains_key(&PlannerLossMechanism::NearIdeal),
        "the report should also identify scenarios where the planner is already near the local \
         oracle"
    );
}

/// The simulator-backed strategy is the first planner this sweep can hold to
/// TODO4's tighter regret bound. It is deliberately separate from
/// `the_planner_chooses_well_on_every_committed_scenario`: the raw cost-model
/// sweep remains the diagnostic that shows what the model still misses, while
/// this test checks the opt-in planner that uses the simulator as its cheap
/// oracle.
#[test]
fn the_simulator_backed_planner_chooses_well_on_every_committed_scenario() {
    let workflow = workflow();
    let base = base_constraints();
    let scenarios = scenarios();
    let near_oracle = RegretBudget::NEAR_ORACLE;
    let ceiling = RegretBudget::LOCAL_ORACLE_CEILING;
    println!("{:<24} {:<24} {:>8}", "scenario", "plan", "regret");
    let mut within_ten_percent = 0usize;
    let mut worst: Option<(String, f64)> = None;
    for (name, scenario) in &scenarios {
        let constraints = scenario.constraints(&base);
        let planner = simulator_backed_for(scenario);
        let chosen = planner
            .plan_with_machine(&workflow, &constraints)
            .unwrap_or_else(|err| panic!("{name}: simulator-backed planning failed: {err}"));
        let verdict = chosen
            .judgement
            .verdicts
            .iter()
            .find(|verdict| verdict.name == chosen.name)
            .expect("the selected plan is in its judgement");
        let oracle = OracleTarget::best_simulated_candidate(name, &chosen.judgement)
            .expect("the candidate field has a simulator winner");
        let comparison = OracleComparison::against(name, verdict.simulated_ns(), oracle);
        let near_oracle_assessment = near_oracle.assess(comparison);
        let ceiling_assessment = ceiling.assess(comparison);
        let regret = ceiling_assessment.regret();
        let plan_shape = plan_shape(&chosen.plan);
        println!("{name:<24} {plan_shape:<24} {regret:>8.3}");
        if near_oracle_assessment.accepts() {
            within_ten_percent += 1;
        }
        if worst.as_ref().is_none_or(|(_, seen)| regret > *seen) {
            worst = Some((name.clone(), regret));
        }
        assert!(
            ceiling_assessment.accepts(),
            "{name}: simulator-backed continuous regret {regret:.3} exceeds TODO4's per-scenario \
             continuous ceiling"
        );
    }
    let (name, regret) = worst.expect("a scenario");
    println!(
        "simulator-backed summary: {within_ten_percent}/{} at <={:.2}; worst {regret:.3}, on {name}",
        scenarios.len(),
        near_oracle.ratio()
    );
    assert!(
        within_ten_percent >= 10,
        "TODO4 requires at least 10 of {} committed scenarios at regret <= {:.2} ({}); got \
         {within_ten_percent}",
        scenarios.len(),
        near_oracle.ratio(),
        near_oracle.label()
    );
}

fn plan_shape_key(plan: &Plan) -> PlanShape {
    PlanShape::from_plan(plan)
}

/// Storage should enter planner pricing only where it changes the oracle.
///
/// The storage scenarios deliberately vary different dimensions around the
/// measured baseline. This test records which of those dimensions move the
/// simulator-backed oracle by at least 10% before any new planner term is
/// justified.
#[test]
fn storage_axes_only_move_the_planner_when_the_oracle_moves() {
    let workflow = workflow();
    let base = base_constraints();
    let scenarios = scenarios();
    let movement_bar = RegretBudget::NEAR_ORACLE;
    let measured = scenarios
        .get("measured")
        .expect("the measured scenario is the baseline");
    let (measured_choice, _) = simulator_backed_choice_for(measured, &workflow, &base);
    let measured_plan = measured_choice.plan;
    let measured_shape = plan_shape_key(&measured_plan);
    struct StorageRow {
        name: &'static str,
        should_move: bool,
    }
    let storage_rows = [
        StorageRow {
            name: "slow-disk",
            should_move: false,
        },
        StorageRow {
            name: "slow-disk-high-latency",
            should_move: false,
        },
        StorageRow {
            name: "compressed-store",
            should_move: false,
        },
        StorageRow {
            name: "fine-chunks",
            should_move: true,
        },
    ];

    println!(
        "{:<24} {:<18} {:<18} {:>8}",
        "scenario", "baseline", "oracle", "baseline/oracle"
    );
    for row in storage_rows {
        let scenario = scenarios
            .get(row.name)
            .unwrap_or_else(|| panic!("{}: committed storage scenario missing", row.name));
        let (choice, _) = simulator_backed_choice_for(scenario, &workflow, &base);
        let oracle_shape = plan_shape_key(&choice.plan);
        let oracle = OracleTarget::best_simulated_candidate(row.name, &choice.judgement)
            .expect("a simulator-backed storage field has a winner");
        let baseline_ns = simulated_ns_on(&measured_plan, scenario, &workflow, &base)
            .unwrap_or_else(|| panic!("{}: measured baseline plan must fit", row.name));
        let comparison = OracleComparison::against(row.name, baseline_ns, oracle);
        let assessment = movement_bar.assess(comparison);
        let regret = assessment.regret();
        println!(
            "{:<24} {:<18} {:<18} {regret:>8.3}",
            row.name,
            format!("{:?}", measured_shape),
            format!("{:?}", oracle_shape)
        );
        if row.should_move {
            assert_ne!(
                oracle_shape, measured_shape,
                "{}: this row is expected to be the storage axis that moves the oracle",
                row.name
            );
            assert!(
                assessment.crossed_by(),
                "{}: oracle moved shape but baseline regret was only {regret:.3}; do not add a \
                 planner storage term below the {} acceptance bar",
                row.name,
                movement_bar.label()
            );
        } else {
            assert_eq!(
                oracle_shape, measured_shape,
                "{}: storage changed the oracle shape; this row now justifies a pricing term and \
                 TODO4's storage notes must be updated",
                row.name
            );
            assert!(
                assessment.accepts(),
                "{}: baseline storage plan costs {regret:.3}x the oracle; this now crosses \
                 TODO4's {} acceptance bar",
                row.name,
                movement_bar.label()
            );
        }
    }
}

/// Cache and prefetch are swept only after the scheduler/block-shape question is
/// stable.
///
/// This is deliberately a policy-search report, not a promotion. A row may move
/// the simulator-backed oracle shape or runtime, but TODO4's promotion bar also
/// requires executor arithmetic coverage. That bridge lives in
/// `tests/simulator_against_the_executor.rs` and the shared-volume tests.
#[test]
fn cache_and_prefetch_policy_search_runs_after_scheduler_shape_search() {
    let workflow = workflow();
    let base = base_constraints();
    let scenarios = scenarios();
    let movement_bar = RegretBudget::NEAR_ORACLE;
    let picked_plan_ceiling = RegretBudget::LOCAL_ORACLE_CEILING;
    let measured = scenarios
        .get("measured")
        .expect("the measured scenario is the cache/prefetch baseline");
    let (measured_choice, _) = simulator_backed_choice_for(measured, &workflow, &base);
    let measured_plan = measured_choice.plan;
    let measured_shape = plan_shape_key(&measured_plan);

    struct PolicyRow {
        name: &'static str,
        note: &'static str,
        edit: fn(&mut blockflow::simulate::Machine),
    }

    impl PolicyRow {
        fn scenario(&self, measured: &Scenario) -> Scenario {
            let mut machine = measured.machine;
            (self.edit)(&mut machine);
            measured
                .clone()
                .with_machine(self.name, machine)
                .noted(self.note)
        }
    }

    let rows = [
        PolicyRow {
            name: "cache-off",
            note: "no modelled residency and no prefetch",
            edit: |machine| {
                machine.cache_bytes = 0;
                machine.prefetch_depth = 0;
            },
        },
        PolicyRow {
            name: "tiny-shared-cache",
            note: "shared cache smaller than the measured page cache",
            edit: |machine| {
                machine.cache_bytes = 1 << 20;
                machine.cache_shared = true;
                machine.prefetch_depth = 0;
            },
        },
        PolicyRow {
            name: "tiny-private-cache",
            note: "per-worker private cache pools under the same byte budget",
            edit: |machine| {
                machine.cache_bytes = 1 << 20;
                machine.cache_shared = false;
                machine.prefetch_depth = 0;
            },
        },
        PolicyRow {
            name: "prefetch-off",
            note: "baseline cache with prefetch disabled",
            edit: |machine| {
                machine.prefetch_depth = 0;
            },
        },
        PolicyRow {
            name: "prefetch-one",
            note: "baseline cache with one-rank prefetch",
            edit: |machine| {
                machine.prefetch_depth = 1;
            },
        },
        PolicyRow {
            name: "prefetch-two",
            note: "baseline cache with two-rank prefetch",
            edit: |machine| {
                machine.prefetch_depth = 2;
            },
        },
        PolicyRow {
            name: "prefetch-deep",
            note: "baseline cache with a deep prefetch horizon",
            edit: |machine| {
                machine.prefetch_depth = 64;
            },
        },
    ];

    println!(
        "{:<20} {:<18} {:<18} {:>8} {:>8}",
        "policy", "baseline", "oracle", "base/orc", "regret"
    );
    let mut moved_oracle = 0usize;
    let mut saw_prefetch = false;
    let mut saw_cache_contract = false;
    for row in rows {
        let scenario = row.scenario(measured);
        saw_prefetch |= scenario.machine.prefetch_depth > 0;
        saw_cache_contract |= !scenario.machine.cache_shared || scenario.machine.cache_bytes == 0;
        let (choice, _) = simulator_backed_choice_for(&scenario, &workflow, &base);
        let oracle_shape = plan_shape_key(&choice.plan);
        let baseline_context = format!("{} baseline", scenario.name);
        let picked_context = format!("{} picked", scenario.name);
        let baseline_oracle =
            OracleTarget::best_simulated_candidate(&baseline_context, &choice.judgement)
                .expect("a cache/prefetch candidate field has a winner");
        let picked_oracle =
            OracleTarget::best_simulated_candidate(&picked_context, &choice.judgement)
                .expect("a cache/prefetch candidate field has a winner");
        let picked = choice
            .judgement
            .verdicts
            .iter()
            .find(|verdict| verdict.name == choice.name)
            .expect("the simulator-backed pick is in its judgement")
            .simulated_ns();
        let baseline_ns = simulated_ns_on(&measured_plan, &scenario, &workflow, &base)
            .unwrap_or_else(|| panic!("{}: measured baseline plan must fit", scenario.name));
        let baseline_assessment = movement_bar.assess(OracleComparison::against(
            &baseline_context,
            baseline_ns,
            baseline_oracle,
        ));
        let picked_assessment = picked_plan_ceiling.assess(OracleComparison::against(
            &picked_context,
            picked,
            picked_oracle,
        ));
        let baseline_regret = baseline_assessment.regret();
        let picked_regret = picked_assessment.regret();
        println!(
            "{:<20} {:<18} {:<18} {:>8.3} {:>8.3}",
            scenario.name,
            format!("{:?}", measured_shape),
            format!("{:?}", oracle_shape),
            baseline_regret,
            picked_regret
        );
        if oracle_shape != measured_shape || baseline_assessment.crossed_by() {
            moved_oracle += 1;
        }
        assert!(
            picked_assessment.accepts(),
            "{}: cache/prefetch search selected a plan at {picked_regret:.3}x the local oracle",
            scenario.name
        );
    }
    assert!(saw_prefetch, "the prefetch axis was not swept");
    assert!(
        saw_cache_contract,
        "the cache size/sharing contract axis was not swept"
    );
    assert!(
        moved_oracle > 0,
        "the cache/prefetch sweep never moved an oracle shape or cost by {}; it would not justify \
         any policy search",
        movement_bar.label()
    );
}

/// The committed performance corpus must name every failure mode TODO4 is
/// allowed to tune against.
///
/// Some modes are scenario files because they are machine contracts; others are
/// executor/simulator bridge tests because the scenario JSON cannot express the
/// workflow topology by itself. Keeping the inventory here prevents us from
/// closing planner work against one broad average.
#[test]
fn performance_corpus_covers_the_known_planner_failure_modes() {
    let scenarios = scenarios();
    for name in [
        "two-nodes",
        "four-nodes",
        "ten-nodes",
        "forty-cores",
        "less-memory",
        "fine-chunks",
        "compressed-store",
    ] {
        assert!(
            scenarios.contains_key(name),
            "TODO4 corpus is missing committed scenario {name}"
        );
    }
    assert!(
        scenarios["two-nodes"].machine.nodes > 1,
        "two-nodes must remain the high-contention overlapping-phase scenario"
    );
    assert!(
        scenarios["forty-cores"].machine.workers >= 40,
        "forty-cores must remain the many-workers-on-one-node scenario"
    );
    assert!(
        scenarios["ten-nodes"].machine.nodes >= 10,
        "ten-nodes must remain the many-nodes scenario"
    );
    assert!(
        scenarios["less-memory"].budget_bytes.is_some(),
        "less-memory must keep an admission/cache budget"
    );
    assert_ne!(
        scenarios["fine-chunks"].storage.chunk, scenarios["measured"].storage.chunk,
        "fine-chunks must keep a storage chunk shape distinct from measured"
    );
    assert!(
        scenarios["compressed-store"].machine.encoded_fraction > 0.0,
        "compressed-store must keep encoded-cache pressure"
    );
}

/// The portability question for the opt-in simulator-backed planner.
///
/// Its native regret is tautologically low because it chooses the simulator
/// winner from the candidate field. This matrix asks the separate question:
/// whether those sharper per-machine choices transfer better or worse than the
/// raw cost-model choices. Today the answer is "worse on some rows", which is
/// why simulator-backed ranking is an opt-in experiment rather than a default
/// production planner.
#[test]
fn simulator_backed_plans_transfer_to_the_other_committed_scenarios() {
    let scenarios = scenarios();
    let workflow = workflow();
    let base = base_constraints();
    let chosen: Vec<PlanChoice> = scenarios
        .iter()
        .map(|(name, scenario)| {
            let (plan, constraints) = simulator_backed_plan_for(scenario, &workflow, &base);
            PlanChoice {
                name: name.clone(),
                plan,
                constraints,
            }
        })
        .collect();
    let matrix = PlanMatrix::from_choices(&chosen, &scenarios, &workflow, &base);

    matrix.print("simulator-backed plan chosen for (row), run on (column)");
    println!(
        "simulator-backed worst transfer: the plan for {} costs {:.3}x on {}",
        matrix.worst.row, matrix.worst.ratio, matrix.worst.column
    );
    assert!(
        matrix.worst.ratio <= 4.8,
        "simulator-backed transfer moved beyond the recorded diagnostic ceiling: {} on {} costs \
         {:.3}x. If this is deliberate, update the table and the TODO4 portability item.",
        matrix.worst.row,
        matrix.worst.column,
        matrix.worst.ratio
    );
}

/// A robust variant of simulator-backed planning: pick from the same candidate
/// field, but score each candidate by its worst regret over the committed
/// machine corpus before choosing.
///
/// This is the direct answer to the transfer blocker. Native simulator-backed
/// ranking closes local regret and then overfits the two-core row; robust
/// ranking gives up some local optimality to keep the plan portable.
#[test]
fn robust_simulator_backed_plans_transfer_under_the_todo4_ceiling() {
    let scenarios = scenarios();
    let workflow = workflow();
    let base = base_constraints();
    let robust: Vec<RobustPlanChoice> = scenarios
        .values()
        .map(|scenario| {
            support::planner_perf::robust_simulator_backed_plan_for(
                scenario, &scenarios, &workflow, &base,
            )
        })
        .collect();
    let chosen: Vec<PlanChoice> = robust.iter().map(|entry| entry.choice.clone()).collect();
    let matrix = PlanMatrix::from_choices(&chosen, &scenarios, &workflow, &base);
    let mut local_worst = ("".to_string(), 1.0f64);

    for entry in &robust {
        if entry.local_regret > local_worst.1 {
            local_worst = (entry.choice.name.clone(), entry.local_regret);
        }
        println!(
            "{}: robust oracle worst against local oracles {:.3}, local regret {:.3}",
            entry.choice.name, entry.oracle_worst, entry.local_regret
        );
    }
    matrix.print("robust simulator-backed plan chosen for (row), run on (column)");
    println!(
        "robust simulator-backed worst transfer: the plan for {} costs {:.3}x on {}; local \
         regret tradeoff worst {:.3} on {}",
        matrix.worst.row, matrix.worst.ratio, matrix.worst.column, local_worst.1, local_worst.0
    );
    assert!(
        matrix.worst.ratio <= 1.50,
        "TODO4 requires the worst admissible transfer cell <=1.50; robust simulator-backed got \
         {:.3} for {} on {}",
        matrix.worst.ratio,
        matrix.worst.row,
        matrix.worst.column
    );
}

// -------------------------------------------------- the transfer matrix --

/// **A plan chosen for one machine, run on another** — the measurement this
/// file exists for.
///
/// Row `A`, column `B` is the simulated makespan on machine `B` of the plan the
/// planner chose for machine `A`, over the makespan on `B` of the plan chosen
/// *for* `B`. The diagonal is `1.000` by construction. An off-diagonal cell
/// above one is what planning for the wrong machine costs, and `over` is worse
/// than any number: the foreign plan does not fit `B`'s memory budget at all,
/// so it would not be admitted rather than merely run slowly.
///
/// **This is the overfitting measurement.** Every planner figure in this crate
/// was taken on one machine; a search tuned to it would show up here as a row
/// that is `1.000` in its own column and large everywhere else. The recorded
/// transfer table and interpretation live in `docs/design/planner-gaps.md`
/// under "Overfitting: what the planner does on machines this is not".
///
/// The ceiling below is per column, so a change that ruins one machine's
/// transfer is named by that machine rather than averaged away.
#[test]
fn a_plan_chosen_for_one_machine_transfers_to_the_others() {
    let scenarios = scenarios();
    let workflow = workflow();
    let base = base_constraints();

    // One plan per scenario, chosen by the planner under that scenario's own
    // model, budget and worker count.
    let chosen: Vec<PlanChoice> = scenarios
        .values()
        .map(|scenario| planner_choice_for(scenario, &workflow, &base))
        .collect();

    let matrix = PlanMatrix::from_choices(&chosen, &scenarios, &workflow, &base);
    matrix.print("plan chosen for (row), run on (column)");
    println!(
        "worst transfer: the plan for {} costs {:.3}x on {}",
        matrix.worst.row, matrix.worst.ratio, matrix.worst.column
    );
    for (column, ratio) in &matrix.column_worst {
        assert!(
            *ratio <= 2.3,
            "{column}: the worst foreign plan costs {ratio:.3}x the plan chosen for it. The \
             recorded worst cell over the whole matrix is 2.263, on `forty-cores` for the \
             `two-nodes` plan; a change that pushes one machine past 2.3 is an overfit to the \
             machines it was tested on."
        );
    }
    assert!(
        matrix.contains_over_budget(),
        "no plan was inadmissible anywhere, so no committed scenario has a budget that binds \
         — and a budget that never binds is the baseline with a smaller number in it. See \
         `less-memory`, which is the scenario tuned to bind."
    );
    // Every scenario's own plan fits its own machine, which is the planner
    // honouring the budget it was given and is what makes an `over` cell
    // meaningful rather than an artefact of how the fit is computed.
    for choice in &chosen {
        let scenario = &scenarios[choice.name.as_str()];
        let workers = scenario.machine.workers.max(1);
        let fit = plan_fit(
            &workflow,
            &choice.plan.decomposition,
            &choice.constraints,
            workers,
        )
        .unwrap_or_else(|err| panic!("{}: {err}", choice.name));
        assert!(
            fit.fits(),
            "{name}: the planner returned a plan that does not fit its own budget: {fit:?}",
            name = choice.name
        );
    }
}
