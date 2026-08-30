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

use std::collections::BTreeMap;

use blockflow::arena::Arena;
use blockflow::decomposition::{Constraints, CostModel};
use blockflow::op::Chain;
use blockflow::probes::{AffineOp, IdentityOp};
use blockflow::scenario::Scenario;
use blockflow::simulate::Rates;
use blockflow::strategy::{Enumerating, Strategy, Workflow};
use blockflow::Dtype;

/// Where the committed scenario files live, relative to the crate root.
const COSTS: &str = "costs";

const VOLUME: [usize; 3] = [96, 96, 96];

/// The block edges every scenario's planner may choose between, and the field
/// its choice is judged against.
///
/// **The caller's knob, not the scenario's** — see `Scenario::constraints`. It
/// is the same ladder for every machine so that what varies across a row of the
/// table is the machine and not the search space.
const LADDER: [usize; 4] = [16, 24, 32, 48];

/// A chain with a real decision in it: a wide-reach op that wants coarse blocks
/// and a cheap narrow one that does not, so where the phase boundary goes and
/// how big the blocks are both matter.
fn chain() -> Chain {
    Chain::sequence(vec![
        Chain::op(IdentityOp::new("smooth", [4, 4, 4]).with_cost(2.0)),
        Chain::op(AffineOp::new("combine", 1.5, 0.5, [1, 1, 1]).with_cost(1.0)),
        Chain::op(IdentityOp::new("skeletonize", [2, 2, 2]).with_cost(8.0)),
    ])
}

fn workflow() -> Workflow {
    Workflow::new(chain(), VOLUME, Dtype::F64)
}

fn base_constraints() -> Constraints {
    Constraints {
        block_candidates: LADDER.to_vec(),
        split_axes: vec![0, 1, 2],
        model: CostModel::default(),
        ..Default::default()
    }
}

fn scenarios() -> BTreeMap<String, Scenario> {
    Scenario::load_dir(COSTS).unwrap_or_else(|err| {
        panic!("the committed scenarios must load: {err}");
    })
}

/// **Write `costs/` from the baseline and the transforms that derive it.**
///
/// Ignored, because it is a *generator* and not an assertion: it rewrites files
/// the rest of this file reads, and a test run must not depend on having done
/// so. Run it with `cargo test --test cost_scenarios -- --ignored
/// regenerate_the_scenario_files` after changing the baseline or adding a
/// machine shape, and commit what it writes.
///
/// **Why the files are committed rather than built here.** Two reasons, and the
/// second is the one that matters. A file can be read by a person, diffed, and
/// pointed at in a bug report; and a scenario built fresh by the test that
/// consumes it would move whenever the code that builds it moved, which is
/// exactly the property a regression bound must not have. The generator is the
/// convenience; the files are the record.
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
        // and the file on disk is what the generator would write, so a hand
        // edit that the generator would undo is caught here rather than at the
        // next regeneration.
        let path = format!("{COSTS}/{name}.json");
        let on_disk = std::fs::read_to_string(&path).expect("the file this came from");
        // The first differing line, rather than two thousand characters of
        // `assert_eq!`. It is how the `preserve_order` difference below was
        // found: every line differed, which said "the key order moved" where a
        // whole-string diff said only "not equal".
        if let Some((line, (a, b))) = on_disk
            .lines()
            .zip(text.lines())
            .enumerate()
            .find(|(_, (a, b))| a != b)
        {
            panic!(
                "{path} line {line} is not what `to_json` writes.\n  on disk:   {a}\n  \
                 generated: {b}\nRegenerate with `cargo test --test cost_scenarios -- \
                 --ignored regenerate_the_scenario_files`."
            );
        }
        assert_eq!(
            on_disk, text,
            "{path} and `to_json` agree line by line and differ in length"
        );
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

/// Plan for one scenario and judge it against a field of pinned block edges.
fn run(scenario: &Scenario) -> Ran {
    let workflow = workflow();
    let base = base_constraints();
    let constraints = scenario.constraints(&base);
    let workers = scenario.machine.workers.max(1);
    let strategy = |constraints: &Constraints| {
        Enumerating {
            concurrency: workers,
            ..Enumerating::default()
        }
        .plan(&workflow, constraints)
    };

    let chosen = strategy(&constraints)
        .unwrap_or_else(|err| panic!("{}: the planner must plan: {err}", scenario.name));
    let plan_shape = format!(
        "{} phase(s) at {:?}",
        chosen.decomposition.n_phases(),
        chosen
            .decomposition
            .phases
            .iter()
            .map(|phase| phase.grid.block()[0])
            .collect::<Vec<_>>()
    );

    let mut arena = Arena::new(scenario.machine, scenario.rates(&Rates::default()))
        .with_snapshot(scenario.snapshot.clone());
    arena
        .enter_plan("planner".to_string(), chosen, constraints.clone())
        .expect("a plan the arena can hold");
    for edge in LADDER {
        let pinned = Constraints {
            block_candidates: vec![edge],
            ..constraints.clone()
        };
        // A rung the budget refuses is not an entrant; the planner's own
        // refusal to use it is the point of the budget.
        if let Ok(plan) = strategy(&pinned) {
            arena
                .enter_plan(format!("edge {edge}"), plan, pinned)
                .expect("a plan the arena can hold");
        }
    }
    let judgement = arena
        .judge(&workflow)
        .unwrap_or_else(|err| panic!("{}: every plan must simulate: {err}", scenario.name));
    let makespan_ns = judgement.verdicts[0].simulated_ns();
    let best_ns = judgement
        .simulated_pick()
        .expect("a field with a winner")
        .simulated_ns();
    Ran {
        plan_shape,
        makespan_ns,
        best_ns,
        regret: makespan_ns / best_ns,
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
///     compressed-store         3 phases at [48, 48, 48]      1.000
///     fine-chunks              3 phases at [48, 48, 48]      1.000
///     forty-cores              3 phases at [48, 32, 32]      1.000
///     four-nodes               3 phases at [32, 24, 32]      1.047
///     less-memory              3 phases at [32, 24, 24]      1.000
///     measured                 3 phases at [48, 48, 48]      1.000
///     slow-compute             3 phases at [48, 48, 48]      1.000
///     slow-disk                2 phases at [48, 48]          1.000
///     slow-disk-high-latency   2 phases at [48, 48]          1.000
///     slow-memory              3 phases at [48, 48, 48]      1.000
///     ten-nodes                3 phases at [24, 24, 24]      1.000
///     two-cores                3 phases at [48, 48, 48]      1.000
///     two-nodes                3 phases at [48, 24, 48]      1.467
/// ```
///
/// **`forty-cores` went from 1.230 to 1.000**, which is what the contention
/// term was for: the model priced the coarsest grid at 3.045 times its own
/// argmin where the simulator put it at 1.018, because it believed forty
/// workers were forty times one. Ten of thirteen machines are now exactly
/// optimal, and the per-rung prices agree with the simulator closely enough to
/// read off:
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
/// **`two-nodes` is 1.467, and it is not a pricing error.** The search now
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
/// **And the executor would not overlap them.** `strategy::execute` pops a wave
/// and joins it before the next, so its phases are sequential; `simulate`
/// dispatches continuously. That is item **C** of `docs/design/planner-gaps.md`
/// — "the simulator and the executor have different concurrency models, and
/// neither states it" — and this is the first number on it: measured at 0.2% of
/// makespan when nothing contends, and **47%** here when something does. So the
/// 1.467 is the divergence between the two models of a run, and which of them
/// to believe about a mixed grid is a question this crate has not settled.
///
/// The bound below is deliberately loose. A fix that *improves* a scenario must
/// not fail here, and the figure that matters is not "is it exactly one" but
/// "did a change make some machine much worse while leaving this one alone" —
/// which is what a per-scenario ceiling catches and a single aggregate would
/// hide.
#[test]
fn the_planner_chooses_well_on_every_committed_scenario() {
    let scenarios = scenarios();
    println!("{:<24} {:<24} {:>8}", "scenario", "plan", "regret");
    let mut worst: Option<(String, f64)> = None;
    for (name, scenario) in &scenarios {
        let ran = run(scenario);
        println!("{name:<24} {:<24} {:>8.3}", ran.plan_shape, ran.regret);
        assert!(
            ran.regret >= 1.0,
            "{name}: a regret below one is arithmetically impossible"
        );
        assert!(
            ran.regret <= 1.55,
            "{name}: the planner's own plan is {:.3}x the best block edge on this machine. \
             The recorded figures are 1.000 everywhere but `four-nodes` (1.047) and \
             `two-nodes` (1.467, and see this test's doc: that one is contention between \
             overlapping phases, which the wave-synchronous executor would not have). If \
             this is a deliberate trade, record the new number here.",
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
/// that is `1.000` in its own column and large everywhere else. What the table
/// actually shows is milder and worth knowing:
///
/// * the `measured` row transfers cleanly to every other machine, so the plan
///   this crate's own evidence produces is not a plan for this host alone;
/// * the whole matrix is driven by the **worker count**, which is the axis
///   every other measurement this session landed on: the `forty-cores` row and
///   column are where the cells depart from one, and the disk, memory and
///   chunk scenarios transfer to each other almost exactly.
///
/// Recorded, 2026-08-30 (after the contention term). Rows are the machine
/// planned for, columns the machine run on; `over` is a plan that does not fit
/// that machine's budget. Trimmed to the columns that are not near-constant;
/// the full table is what the test prints.
///
/// ```text
///                          forty-c  four-no  less-me  measure  ten-nod  two-nod
///     compressed-store       1.037    1.183     over    1.000    1.827    0.681
///     forty-cores            1.000    0.976     over    1.110    1.245    0.676
///     four-nodes             1.025    1.000    1.013    1.278    1.178    0.755
///     less-memory            1.128    1.033    1.000    1.263    0.966    0.740
///     measured               1.037    1.183     over    1.000    1.827    0.681
///     slow-disk              1.043    1.184     over    1.006    1.807    0.684
///     ten-nodes              1.192    1.055    1.028    1.304    1.000    0.769
///     two-cores              1.037    1.183     over    1.000    1.827    0.681
///     two-nodes              1.606    1.755     over    0.998    1.796    1.000
/// ```
///
/// Four things it says:
///
/// * **a single-machine plan still costs 1.83x on ten computers**, and that is
///   the largest cell. The contention term did not touch it, because what it is
///   made of is not contention: ten page caches that cannot lend each other a
///   chunk fetch the same volume many times over — measured at 3.26x the bytes
///   in `tests/multiple_computers.rs` — and **nothing in `CostModel` prices
///   that**. It is the next item;
/// * **the `less-memory` column is `over` for every foreign plan.** A plan
///   chosen on a machine with memory is not slow on one without it — it is
///   inadmissible, which is the strongest form the overfitting question has.
///   That column is the reason the fit is checked rather than only the duration;
/// * **the `two-nodes` column is below one for almost every foreign plan**, and
///   its own row is the worst traveller in the table (1.61 and 1.76 on the two
///   busiest machines). Both are the same fact seen twice: that row's plan is
///   the mixed grid whose penalty is contention between overlapping phases —
///   see the regret test's doc, and the control that switches contention off and
///   watches the penalty vanish;
/// * **the machines that differ only in their storage still transfer at
///   1.000.** Disk speed, chunk shape and compression choose one plan among
///   them, exactly as before the contention term. What moves a plan is the slot
///   count, the node count and the budget.
///
/// The ceiling below is per column, so a change that ruins one machine's
/// transfer is named by that machine rather than averaged away.
#[test]
fn a_plan_chosen_for_one_machine_transfers_to_the_others() {
    use blockflow::arena::working_set_bytes;

    let scenarios = scenarios();
    let workflow = workflow();
    let base = base_constraints();

    // One plan per scenario, chosen by the planner under that scenario's own
    // model, budget and worker count.
    let chosen: Vec<(String, blockflow::strategy::Plan, Constraints)> = scenarios
        .iter()
        .map(|(name, scenario)| {
            let constraints = scenario.constraints(&base);
            let plan = Enumerating {
                concurrency: scenario.machine.workers.max(1),
                ..Enumerating::default()
            }
            .plan(&workflow, &constraints)
            .unwrap_or_else(|err| panic!("{name}: {err}"));
            (name.clone(), plan, constraints)
        })
        .collect();

    let names: Vec<&str> = chosen.iter().map(|(name, _, _)| name.as_str()).collect();
    println!("plan chosen for (row), run on (column)");
    print!("{:<24}", "");
    for name in &names {
        print!("{:>10.10}", name);
    }
    println!();

    let mut worst = (String::new(), String::new(), 1.0f64);
    let mut column_worst: BTreeMap<&str, f64> = BTreeMap::new();
    let mut rows: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (row_name, plan, _) in &chosen {
        let mut cells = Vec::new();
        for (column_name, _, _) in &chosen {
            let scenario = &scenarios[column_name.as_str()];
            let constraints = scenario.constraints(&base);
            let workers = scenario.machine.workers.max(1);

            // Does it fit at all on this machine?
            let bytes = working_set_bytes(&workflow, &plan.decomposition, &constraints, workers)
                .unwrap_or_else(|err| panic!("{row_name} on {column_name}: {err}"));
            if scenario
                .budget_bytes
                .is_some_and(|budget| bytes > budget as f64)
            {
                cells.push("over".to_string());
                continue;
            }

            let native = &chosen
                .iter()
                .find(|(name, _, _)| name == column_name)
                .expect("every column is a scenario that was planned for")
                .1;
            let mut arena = Arena::new(scenario.machine, scenario.rates(&Rates::default()))
                .with_snapshot(scenario.snapshot.clone());
            // Entrant 0 is the foreign plan and entrant 1 the native one, which
            // is what the ratio below reads. On the diagonal they are the same
            // plan and the cell is 1.000 by construction.
            for (name, plan) in [("foreign", plan), ("native", native)] {
                arena
                    .enter_plan(name.to_string(), plan.clone(), constraints.clone())
                    .expect("a plan the arena can hold");
            }
            let judgement = arena
                .judge(&workflow)
                .unwrap_or_else(|err| panic!("{row_name} on {column_name}: {err}"));
            let ratio = judgement.verdicts[0].simulated_ns() / judgement.verdicts[1].simulated_ns();
            cells.push(format!("{ratio:.3}"));
            let entry = column_worst.entry(column_name.as_str()).or_insert(1.0);
            *entry = entry.max(ratio);
            if ratio > worst.2 {
                worst = (row_name.clone(), column_name.clone(), ratio);
            }
        }
        rows.insert(row_name.as_str(), cells);
    }

    for (name, cells) in &rows {
        print!("{name:<24}");
        for cell in cells {
            print!("{cell:>10}");
        }
        println!();
    }
    println!(
        "worst transfer: the plan for {} costs {:.3}x on {}",
        worst.0, worst.2, worst.1
    );
    for (column, ratio) in &column_worst {
        assert!(
            *ratio <= 2.0,
            "{column}: the worst foreign plan costs {ratio:.3}x the plan chosen for it. The \
             recorded worst cell over the whole matrix is 1.827, on `ten-nodes`, and it is \
             cross-node fetch duplication that nothing prices yet; a change that pushes one \
             machine past twice is an overfit to the machines it was tested on."
        );
    }
    assert!(
        rows.values().flatten().any(|cell| cell == "over"),
        "no plan was inadmissible anywhere, so no committed scenario has a budget that binds \
         — and a budget that never binds is the baseline with a smaller number in it. See \
         `less-memory`, which is the scenario tuned to bind."
    );
    // Every scenario's own plan fits its own machine, which is the planner
    // honouring the budget it was given and is what makes an `over` cell
    // meaningful rather than an artefact of how the fit is computed.
    for (name, plan, constraints) in &chosen {
        let scenario = &scenarios[name.as_str()];
        let workers = scenario.machine.workers.max(1);
        let bytes = working_set_bytes(&workflow, &plan.decomposition, constraints, workers)
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        if let Some(budget) = scenario.budget_bytes {
            assert!(
                bytes <= budget as f64,
                "{name}: the planner returned a plan needing {bytes:.0} bytes against its own \
                 {budget} byte budget"
            );
        }
    }
}
