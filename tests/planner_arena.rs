// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The planner arena, exercised: plans from every strategy, judged by the
// simulator, beside the price the planner's own model puts on them.**
//
// `docs/design/planner-gaps.md` opens on the finding this file answers —
// *"Nothing has ever fed a `Strategy`-produced `Decomposition` into
// `simulate`"* — and calls the missing harness **G1**, the prerequisite for
// accepting any other change on its list. `src/arena.rs` is the harness; this
// file is what says it works and what it found.
//
// What is asserted, and in what order
// -----------------------------------
//
// | claim | why it comes first |
// |---|---|
// | the arena's price is the search's own objective | everything below compares against it; if the arena priced plans differently from the planner, a disagreement would be the arena's |
// | every strategy's plan reaches the simulator, and simulates to the plan's own task count | the path itself, which is the deliverable |
// | the two judges are two judges | a field where they cannot disagree measures nothing, and a rank correlation of one would be indistinguishable from a bug that read one number twice |
// | the arena is deterministic | it is going to be used to adjudicate changes, so a figure that moves between runs is worse than no figure |
//
// The first is the load-bearing one and it is not a tautology: `price_plan`
// prices a plan phase by phase from the plan itself, while the search prices
// candidate *runs* and keeps an argmin over a DP. They meet at
// `strategy::phase_price`, which both call — so the test that the arena's
// argmin over a ladder is the edge the search chose is a check on the
// composition around that function, which is not shared.

use blockflow::arena::{price_plan, Arena};
use blockflow::decomposition::{Constraints, CostModel};
use blockflow::op::Chain;
use blockflow::probes::{AffineOp, IdentityOp};
use blockflow::simulate::{Decision, ExecutorOrder, Machine, Rates, Scheduler};
use blockflow::strategy::{
    Enumerating, Greedy, Materialising, PartitionSearch, Strategy, Trivial, Workflow,
};
use blockflow::Dtype;

/// Long enough on the split axis for the ladder below to give genuinely
/// different grids, and small enough that a simulated run is instant.
const VOLUME: [usize; 3] = [64, 64, 64];

/// The chunk both the rates and any cache question are stated in.
const CHUNK: [usize; 3] = [16, 16, 16];

/// A chain with something for a planner to decide: three ops with different
/// reaches, so cutting between them trades a materialisation against a halo,
/// and one of them expensive enough that where it lands matters.
fn chain() -> Chain {
    Chain::sequence(vec![
        Chain::op(IdentityOp::new("wide", [4, 4, 4]).with_cost(2.0)),
        Chain::op(AffineOp::new("scale", 1.5, 0.5, [1, 1, 1]).with_cost(8.0)),
        Chain::op(IdentityOp::new("narrow", [0, 0, 0]).with_cost(1.0)),
    ])
}

fn workflow() -> Workflow {
    Workflow::new(chain(), VOLUME, Dtype::F64)
}

fn constraints(candidates: Vec<usize>) -> Constraints {
    Constraints {
        block_candidates: candidates,
        split_axes: vec![0, 1, 2],
        model: CostModel::default(),
        ..Default::default()
    }
}

fn machine(workers: usize) -> Machine {
    Machine {
        workers,
        // A cache that holds something, so that an ordering can hit in it and
        // two plans over the same volume are not trivially equal in IO.
        cache_bytes: 1 << 22,
        prefetch_depth: 0,
        ..Machine::default()
    }
}

fn rates() -> Rates {
    Rates {
        chunk: CHUNK,
        chunk_bytes: (CHUNK.iter().product::<usize>() * 8) as u64,
        ..Rates::default()
    }
}

// ------------------------------- claim 1: the arena prices what the search --

/// **The arena's objective is the search's**, checked where the two can differ:
/// the plan the search returns must be the cheapest thing the arena can price
/// out of the same ladder.
///
/// Both sides call `strategy::phase_price`, which is the point — but only the
/// coefficients are shared by that. What is not shared is everything around it:
/// the search prices contiguous *runs* of slots inside a DP and takes an argmin
/// per run, while `price_plan` walks a finished plan's phases and sums them. If
/// those two compositions ever part, this fails.
///
/// **The comparison is `<=` and not `==` on purpose.** The search chooses the
/// partition and the edge together and may give two phases two edges; pinning
/// the ladder to one rung takes that freedom away, so a pinned plan is one the
/// search could have returned and not necessarily the one it did. What must
/// hold — and what would break the moment the arena priced anything
/// differently from the search — is that the search's answer is no more
/// expensive than any of them.
#[test]
fn the_arena_prices_the_search_s_own_answer_lowest() {
    let workflow = workflow();
    let ladder = vec![8usize, 16, 32, 64];
    let full = constraints(ladder.clone());
    let chosen = Enumerating::default()
        .plan(&workflow, &full)
        .expect("a plan over the whole ladder");
    let chosen_price =
        price_plan(&workflow, &chosen.decomposition, &full, 1).expect("the chosen plan prices");
    println!(
        "the search chose {:?} over {} phases, priced {chosen_price:.1}",
        chosen
            .decomposition
            .phases
            .iter()
            .map(|phase| phase.grid.block())
            .collect::<Vec<_>>(),
        chosen.decomposition.n_phases()
    );

    let mut priced: Vec<(usize, f64)> = Vec::new();
    for &edge in &ladder {
        let pinned = constraints(vec![edge]);
        let plan = Enumerating::default()
            .plan(&workflow, &pinned)
            .unwrap_or_else(|err| panic!("edge {edge}: {err}"));
        // Priced against the **full** ladder's constraints, because a
        // `Constraints` carries the cost model and a price taken under a
        // different model is a price on a different ruler. Only the candidate
        // list differs between the two, and the candidate list is not a
        // coefficient.
        let price = price_plan(&workflow, &plan.decomposition, &full, 1)
            .unwrap_or_else(|err| panic!("edge {edge}: {err}"));
        priced.push((edge, price));
        println!(
            "edge {edge}: {} phases, blocks {:?}, priced {price:.1}",
            plan.decomposition.n_phases(),
            plan.decomposition
                .phases
                .iter()
                .map(|phase| phase.grid.n_blocks())
                .collect::<Vec<_>>()
        );
    }

    for (edge, price) in &priced {
        assert!(
            chosen_price <= *price,
            "the search returned a plan the arena prices at {chosen_price}, above the {price} \
             it prices the plan pinned to edge {edge} at. One of the two is not the objective \
             the other thinks it is."
        );
    }

    // and the ladder must actually spread, or the comparison above is a
    // comparison of one number wearing four names
    let spread = priced
        .iter()
        .map(|(_, price)| *price)
        .fold(f64::NEG_INFINITY, f64::max)
        / priced
            .iter()
            .map(|(_, price)| *price)
            .fold(f64::INFINITY, f64::min);
    assert!(
        spread > 1.05,
        "the ladder prices within {spread:.4} of each other, so this test would pass on a \
         constant"
    );
    println!("the ladder spreads by {spread:.3}x");
}

// ------------------------------------ claim 2: every strategy reaches the --
// ------------------------------------------------------ simulator --------

/// **The path G1 says does not exist**: a `Strategy`, a `Workflow` and a
/// `Constraints` in, a simulated outcome out, for every strategy the crate
/// ships.
///
/// The conservation law is the assertion that makes it more than a smoke test.
/// `Outcome::tasks_run` is a property of the *plan* — the simulator's own doc
/// calls it the invariant to check a scheduler against — so it must equal the
/// decomposition's own task count for every entrant. A path that silently ran a
/// different plan, or ran the same plan twice, fails here rather than producing
/// a plausible ranking.
#[test]
fn every_strategy_produces_a_plan_the_simulator_runs() {
    let workflow = workflow();
    let constraints = constraints(vec![16, 32, 64]);
    let mut arena = Arena::new(machine(4), rates());
    arena
        .enter("trivial", &Trivial, &workflow, &constraints)
        .expect("the trivial strategy plans");
    arena
        .enter("greedy", &Greedy::default(), &workflow, &constraints)
        .expect("the greedy strategy plans");
    arena
        .enter(
            "materialising",
            &Materialising::default(),
            &workflow,
            &constraints,
        )
        .expect("the materialising strategy plans");
    for concurrency in [1usize, 4] {
        arena
            .enter(
                format!("enumerating, {concurrency} workers"),
                &Enumerating {
                    concurrency,
                    ..Enumerating::default()
                },
                &workflow,
                &constraints,
            )
            .expect("the enumerating strategy plans");
    }

    let judgement = arena.judge(&workflow).expect("every plan simulates");
    println!("{}", judgement.table());
    assert_eq!(judgement.verdicts.len(), 5);

    for (verdict, entrant) in judgement.verdicts.iter().zip(arena.entrants()) {
        assert_eq!(
            verdict.outcome.tasks_run as usize,
            entrant.plan.decomposition.n_tasks(),
            "{}: the simulator ran a different number of tasks than the plan has",
            verdict.name
        );
        assert!(
            verdict.outcome.makespan_ns > 0,
            "{}: a plan that takes no time is a plan that did nothing",
            verdict.name
        );
        assert!(
            verdict.priced_ns > 0.0,
            "{}: the planner priced it at nothing",
            verdict.name
        );
        assert!(
            verdict.outcome.written_bytes > 0,
            "{}: nothing reached the output",
            verdict.name
        );
    }

    // Every entrant writes the same output volume, which is what makes them
    // entrants in one competition rather than four different runs.
    let written: Vec<u64> = judgement
        .verdicts
        .iter()
        .map(|verdict| verdict.outcome.written_bytes)
        .collect();
    assert!(
        written.windows(2).all(|pair| pair[0] == pair[1]),
        "the entrants wrote different totals to the output: {written:?}"
    );
}

// -------------------------------------- claim 3: what the arena measured --

/// **The two judges agree at one worker and part at four.** The arena's first
/// finding, recorded here.
///
/// The field is four plans over one chain, differing only in the block edge the
/// ladder was pinned to — the one degree of freedom the search's inner sweep
/// has. Both rankings are taken at one worker and at four:
///
/// ```text
///     1 worker                          4 workers
///     plan     priced  simulated        plan     priced  simulated
///     edge 8    3.068      7.506        edge 8    2.530      4.663
///     edge 16   1.760      3.641        edge 16   1.457      2.285
///     edge 32   1.326      2.448        edge 32   1.000      1.000
///     edge 64   1.000      1.000        edge 64   2.660      2.479
///     kendall tau 1.0                   kendall tau 0.667
///     no discordant pair                edge 8 against edge 64
/// ```
///
/// At one worker the two orderings are **identical**, which is what
/// `Enumerating::concurrency`'s own doc predicts when it calls `1` the negative
/// control: the objective collapses to serial work, which is monotone in the
/// edge, and so is the simulator here. At four they disagree about the two
/// extremes — the cost model prices the 1024-block plan *below* the
/// single-block one, and the simulator makes it nearly twice as slow. Both
/// still pick edge 32, so the regret is `1.000`: **the model's argmin survives
/// and its ordering does not.**
///
/// **This is item C of `docs/design/planner-gaps.md` with a number on it** —
/// "the simulator and the executor have different concurrency models, and
/// neither states it" — reached from the other side: the *cost model* and the
/// simulator differ, `rounds()` divides a whole per-block cost by the pool
/// against a simulator that dispatches continuously, and the difference is
/// invisible at the shipped default concurrency of one. It is not the chunk
/// grid: the same pair is discordant at chunk edges 8, 16, 32 and 64, and the
/// tau at one worker is 1.0 at all four.
///
/// **When this changes, that is a finding and not a failure.** A cost model
/// that acquires a per-block term, or a simulator that stops dispatching
/// continuously, moves these numbers; the assertions below are here so that it
/// moves them *visibly*, with a place to write down what the new figures are.
#[test]
fn the_two_judges_agree_at_one_worker_and_part_at_four() {
    let workflow = workflow();
    let edges = [8usize, 16, 32, 64];
    let mut taus = Vec::new();
    for workers in [1usize, 4] {
        let mut arena = Arena::new(machine(workers), rates());
        for edge in edges {
            arena
                .enter(
                    format!("edge {edge}"),
                    &Enumerating {
                        concurrency: workers,
                        ..Enumerating::default()
                    },
                    &workflow,
                    &constraints(vec![edge]),
                )
                .unwrap_or_else(|err| panic!("edge {edge}: {err}"));
        }
        let judgement = arena.judge(&workflow).expect("every plan simulates");
        println!("{}", judgement.table());

        // Neither judge may be constant, or a rank correlation over it is a
        // statement about a tie rather than about an ordering.
        let priced: Vec<f64> = judgement
            .verdicts
            .iter()
            .map(|verdict| verdict.priced_ns)
            .collect();
        let simulated: Vec<f64> = judgement
            .verdicts
            .iter()
            .map(|verdict| verdict.simulated_ns())
            .collect();
        assert!(
            priced.windows(2).all(|pair| pair[0] != pair[1]),
            "{workers} workers: the model priced two entrants alike: {priced:?}"
        );
        assert!(
            simulated.windows(2).all(|pair| pair[0] != pair[1]),
            "{workers} workers: the simulator timed two entrants alike: {simulated:?}"
        );

        let tau = judgement.kendall_tau().expect("an ordered field");
        let regret = judgement.regret().expect("a field with a winner");
        assert!(
            regret >= 1.0,
            "a regret below one is arithmetically impossible: {regret}"
        );
        println!(
            "{workers} workers: kendall tau {tau:.3}, regret {regret:.3}, discordant {:?}",
            judgement.discordant_pairs()
        );
        taus.push((workers, tau, regret, judgement));
    }

    let (_, tau_one, regret_one, one) = &taus[0];
    assert_eq!(
        *tau_one,
        1.0,
        "at the negative control the two judges ordered the field differently, which is new: \
         {:?}",
        one.discordant_pairs()
    );
    assert_eq!(*regret_one, 1.0);

    let (_, tau_four, regret_four, four) = &taus[1];
    assert!(
        *tau_four < 1.0,
        "at four workers the two judges now agree on the whole ordering. That is a finding — \
         the recorded measurement is a tau of 0.667 with edge 8 against edge 64 — and this \
         test is where to write down what changed."
    );
    assert_eq!(
        four.discordant_pairs(),
        vec![("edge 8", "edge 64")],
        "the recorded disagreement is between the extremes of the ladder"
    );
    assert_eq!(
        *regret_four, 1.0,
        "the model's argmin used to survive the disagreement; now it does not, which is a \
         bigger finding than the disagreement itself"
    );
    assert_eq!(
        four.simulated_pick().map(|verdict| verdict.name.as_str()),
        Some("edge 32")
    );
}

// ------------------------------ claim 4: when the phases of a plan overlap --

/// **G2, measured — and the answer is not the one the scouting report
/// predicted.**
///
/// `strategy::phase_makespan` prices one phase alone and the partition search
/// adds the phases up, so the objective is a sum over phases: the wall clock
/// only if no two phases ever run at once. `docs/design/planner-gaps.md` lists
/// that as **G2** and calls overlapping phases *"the worst of these"*, on the
/// reasoning that *"`TaskGraph` makes them pipeline"* — a block of phase
/// `p + 1` depends on the blocks of phase `p` covering its read extent and on
/// nothing else, so the graph permits a worker to start the next phase long
/// before this one ends.
///
/// **The graph permits it; the shipped scheduler does not do it; and where a
/// scheduler does, it buys nothing.** `Outcome::phase_overlap` is the sum of
/// the phases' own spans over the makespan — `1.0` for a strictly sequential
/// run, above it for a pipelined one. On this file's chain at four workers:
///
/// ```text
///     plan      phases   blocks   phase-major   block-major   makespan x   peak x
///     edge 8         2     1024         1.003         1.894        0.999    1.023
///     edge 16        2      128         1.030         1.644        1.002    1.049
///     edge 32        1        8         1.000         1.000        1.000    1.000
///     edge 64        1        1         1.000         1.000        1.000    1.000
/// ```
///
/// Three things, in the order they matter:
///
/// 1. under `SchedulePriority::PhaseMajor` — `Hints::default()`, the shipped
///    policy, and the order `strategy::execute`'s heap really pops — the phases
///    are **sequential to within 3%**, so the cost model's assumption is not a
///    bias there. It is a correct description of what that dispatcher does;
/// 2. under `BlockMajor`, whose own doc calls it *"fusion, and the smaller
///    working set"*, the same plans run at **1.6 to 1.9 phase-spans to the
///    makespan** — the pipelining the report predicted, and it is real;
/// 3. and it changes the makespan by **0.2%** and the peak bytes by **2 to 5%**.
///    With four workers saturated the total work is the same either way, so
///    overlapping the phases moves when the work happens and not how much.
///
/// So the bias G2 names exists, is confined to a policy the default does not
/// use, and is worth about a fifth of a percent of wall clock where it does
/// apply. On this evidence it belongs well below G3 rather than at the top of
/// the list — which is what "measure the bias first" was for.
///
/// **What this is not evidence about.** One chain, one volume, four workers,
/// `contention = 0.0`, and a pool that never starves. Pipelining pays when a
/// phase cannot fill the pool — a late phase with few blocks, a barrier, a
/// phase whose blocks are unequal — and none of those is in this field. The
/// figure to distrust here is the makespan column, not the overlap one.
///
/// The single-phase plans are the control: a run with one phase has nothing to
/// overlap, and its ratio is `1.000` under both policies by construction.
#[test]
fn the_phases_overlap_only_under_the_policy_that_fuses() {
    let workflow = workflow();
    let mut arena = Arena::new(machine(4), rates());
    for edge in [8usize, 16, 32, 64] {
        arena
            .enter(
                format!("edge {edge}"),
                &Enumerating {
                    concurrency: 4,
                    ..Enumerating::default()
                },
                &workflow,
                &constraints(vec![edge]),
            )
            .unwrap_or_else(|err| panic!("edge {edge}: {err}"));
    }
    let sequential = arena
        .judge_with(&workflow, &mut || Box::new(ExecutorOrder::phase_major()))
        .expect("every plan simulates");
    let fused = arena
        .judge_with(&workflow, &mut || Box::new(ExecutorOrder::block_major()))
        .expect("every plan simulates");

    let mut multi_phase = 0usize;
    let mut single_phase = 0usize;
    println!(
        "{:<12} {:>7} {:>8} {:>13} {:>13} {:>14} {:>9}",
        "plan", "phases", "blocks", "phase-major", "block-major", "makespan x", "peak x"
    );
    for (one, other) in sequential.verdicts.iter().zip(fused.verdicts.iter()) {
        let sequential = one.outcome.phase_overlap().expect("a run of some length");
        let fused = other.outcome.phase_overlap().expect("a run of some length");
        println!(
            "{:<12} {:>7} {:>8} {:>13.3} {:>13.3} {:>14.3} {:>9.3}",
            one.name,
            one.phases,
            one.blocks.iter().sum::<usize>(),
            sequential,
            fused,
            other.simulated_ns() / one.simulated_ns(),
            other.outcome.peak_bytes as f64 / one.outcome.peak_bytes.max(1) as f64
        );
        if one.phases == 1 {
            single_phase += 1;
            assert_eq!(
                (sequential, fused),
                (1.0, 1.0),
                "{}: a plan with one phase has nothing to overlap, so its span is its \
                 makespan under any policy. A ratio away from one means the span is not \
                 measuring what it says.",
                one.name
            );
            continue;
        }
        multi_phase += 1;
        assert!(
            sequential < 1.05,
            "{}: the phase-major policy now overlaps its phases at {sequential:.3} spans to \
             the makespan. That is a finding — the cost model's sum over phases stops being \
             exact for the shipped dispatcher — and this test is where to write down what \
             changed.",
            one.name
        );
        assert!(
            fused > 1.5,
            "{}: the block-major policy ran at {fused:.3} spans to the makespan, so it did \
             not fuse. Either the task graph stopped letting a block run ahead of its phase — \
             which is what that policy is for — or the span accounting is wrong.",
            one.name
        );
    }
    assert!(
        multi_phase >= 2 && single_phase >= 1,
        "the field needs both kinds to say anything: {multi_phase} multi-phase and \
         {single_phase} single-phase"
    );
}

// ----------------------------- claim 5: the price is the planner's, still --

/// **The arena's price and the plan evaluator's agree, on a chain that
/// binarizes half way through** — which is the chain they used to disagree
/// about.
///
/// Two statements of one quantity, from two directions. `arena::price_plan`
/// walks a plan's phases and calls `strategy::phase_price`, the function the
/// partition search itself minimises with; `strategy::predicted_makespan` walks
/// the same phases through `Decomposition`'s own accounting, with
/// `phase_traffic` and `phase_compute_per_voxel`. Neither is a summary of the
/// other and both are `sum over phases of phase_makespan`, so they have to come
/// out equal.
///
/// **They did not, and the chain below is why.** The search handed
/// `workflow.dtype` to every phase while `predicted_cost` read the plan's own
/// `dtype_at`, so any chain that changed element type mid-way was priced two
/// ways by two parts of one planner — `docs/design/planner-gaps.md`'s G3, and a
/// defect `Materialising` had already fixed for itself while naming it. The
/// first op here turns `f64` into `bool`, so the second phase reads one byte a
/// voxel where the workflow's own type is eight: an **8x** disagreement on
/// every term built from the byte count.
///
/// The plain chain beside it is the control. The two prices agreed on it before
/// the fold as well, so a test that used only that chain would have passed
/// throughout and said nothing.
#[test]
fn the_arena_prices_a_binarising_chain_as_the_plan_evaluator_does() {
    use blockflow::fragment::PhaseWork;
    use blockflow::probes::NonZeroOp;
    use blockflow::strategy::predicted_makespan;

    let constraints = constraints(vec![16, 32, 64]);
    let cases: Vec<(&str, Chain)> = vec![
        (
            "binarizing",
            Chain::sequence(vec![
                Chain::op(NonZeroOp::new("binarize", [1, 1, 1])),
                Chain::op(IdentityOp::new("after", [2, 2, 2]).with_cost(4.0)),
                Chain::op(IdentityOp::new("last", [1, 1, 1]).with_cost(8.0)),
            ]),
        ),
        ("plain", chain()),
    ];
    for (name, chain) in cases {
        let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
        let plan = Enumerating {
            concurrency: 4,
            ..Enumerating::default()
        }
        .plan(&workflow, &constraints)
        .unwrap_or_else(|err| panic!("{name}: {err}"));
        let phases = plan.decomposition.n_phases();
        let dtypes: Vec<blockflow::Dtype> = (0..phases)
            .map(|index| plan.decomposition.dtype_at(index))
            .collect();

        let arena = price_plan(&workflow, &plan.decomposition, &constraints, 4)
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        let evaluator = predicted_makespan(
            &workflow.chain,
            &plan.decomposition,
            &vec![PhaseWork::Pixels; phases],
            &constraints.model,
            4,
        )
        .unwrap_or_else(|err| panic!("{name}: {err}"));

        println!("{name}: {phases} phases reading {dtypes:?}, priced {arena:.1} / {evaluator:.1}");
        if name == "binarizing" {
            assert!(
                dtypes.contains(&Dtype::Bool),
                "the fixture must actually change element type, or it is the control twice"
            );
        }
        assert_eq!(
            arena, evaluator,
            "{name}: the search's objective and the plan evaluator's price the same plan \
             differently"
        );
    }
}

/// **A chain that hands an op an element type it does not accept is refused
/// when the plan is made**, which is a consequence of the same fold.
///
/// The search folds `Chain::produces` along the slots to learn what each run
/// reads, and that fold is the check: `produces` refuses an op handed a type it
/// does not accept. So a chain that binarizes and then applies an `f64`-only op
/// no longer reaches `check_dtypes` at execute time — it is refused by the
/// planner, naming both the op and the type.
///
/// Worth its own test because it is a **new refusal**: before the fold the
/// search never asked, and the same chain planned happily and failed later.
#[test]
fn the_search_refuses_a_chain_that_narrows_into_an_op_that_cannot_take_it() {
    use blockflow::probes::NonZeroOp;

    let workflow = Workflow::new(
        Chain::sequence(vec![
            Chain::op(NonZeroOp::new("binarize", [1, 1, 1])),
            // `f64` only, by its own `accepts`.
            Chain::op(AffineOp::new("scale", 1.5, 0.5, [1, 1, 1])),
        ]),
        VOLUME,
        Dtype::F64,
    );
    let err = Enumerating::default()
        .plan(&workflow, &constraints(vec![32]))
        .expect_err("an op handed a type it does not accept must be refused")
        .to_string();
    assert!(
        err.contains("scale") && err.contains("bool"),
        "the refusal must name the op and the type: {err}"
    );
}

// ------------------------ claim 6: a planner change, adjudicated by the sim --

/// **G3, adjudicated: the per-family compute corrections change the plan, and
/// the simulator prefers the plan they choose.**
///
/// This is what the arena was built for. `docs/design/planner-gaps.md` opens on
/// the finding that no `Strategy`-produced plan had ever reached `simulate`, so
/// every planner change in this crate's history was accepted or rejected by
/// argument and by the cost model's own opinion of itself. This one is not.
///
/// **The setup.** Two ops with *equal declared cost* and different reaches —
/// a wide-reach one and a narrow-reach one — and a snapshot saying their true
/// rates are 64x apart. Both judges are told the same measurements: the planner
/// through `CostModel::compute_of`, filled by `Snapshot::calibrate`; the
/// simulator through `PerPhase::ns_per_voxel`, filled by
/// `simulate::phase_rates_from_snapshot`. The uncorrected model sees two ops it
/// believes cost the same, so it fuses them and gives both one grid; the
/// corrected model can see that the dear op wants a fine grid and the cheap one
/// a coarse grid, and cuts between them.
///
/// ```text
///     workers   plain plan                     corrected plan                  simulated
///        1      1 phase,  block 64             1 phase,  block 64              1.00x
///        4      1 phase,  block 32             2 phases, blocks 64 and 32      1.28x
///       40      1 phase,  block 32             2 phases, blocks 64 and 16      2.86x
/// ```
///
/// **2.86x at forty workers**, and nothing at one — the negative control, where
/// `Enumerating`'s objective collapses to serial work and the per-phase freedom
/// it is given is freedom on paper. That is the same boundary every other
/// measurement in this file falls on, and it is the reason the shipped default
/// of `concurrency = 1` hid all of this.
///
/// **What it is not.** Evidence that the corrections are *right* — the snapshot
/// here is constructed, not measured, and the simulator is told the same
/// numbers the planner is, so this cannot check whether either matches a
/// machine. What it checks is that the planner can now *act* on evidence it has
/// been recording for years and could not use, and that acting on it is not
/// harmful in the model of the run that is independent of the cost model.
#[test]
fn the_per_family_corrections_change_the_plan_and_the_simulator_prefers_it() {
    use blockflow::decomposition::CostModel;
    use blockflow::statistics::{Coefficient, Snapshot, Term};

    /// A coefficient stated outright, reproduced often enough to be believed.
    fn coefficient(nanos_per_unit: f64) -> Coefficient {
        Coefficient {
            nanos_per_unit,
            runs: 8,
            units: 1000.0,
            total_nanos: nanos_per_unit * 1000.0,
        }
    }

    // Equal declared costs, so the uncorrected model cannot tell them apart;
    // different reaches, so which grid suits which op is a real question.
    let workflow = Workflow::new(
        Chain::sequence(vec![
            Chain::op(IdentityOp::new("cheap", [4, 4, 4]).with_cost(1.0)),
            Chain::op(IdentityOp::new("dear", [1, 1, 1]).with_cost(1.0)),
        ]),
        VOLUME,
        Dtype::F64,
    );
    let snapshot = Snapshot::default()
        .with(Term::Compute, coefficient(10.0))
        .with(Term::ComputeOf("cheap".to_string()), coefficient(1.0))
        .with(Term::ComputeOf("dear".to_string()), coefficient(640.0));
    let plain = CostModel::default();
    let corrected = snapshot.calibrate(&plain);
    assert_eq!(corrected.correction_for("cheap"), 0.1);
    assert_eq!(corrected.correction_for("dear"), 64.0);

    let ladder = vec![8usize, 16, 32, 64];
    let mut differed = 0usize;
    println!(
        "{:>8} {:<34} {:<34} {:>10}",
        "workers", "plain plan", "corrected plan", "simulated"
    );
    for workers in [1usize, 4, 40] {
        let mut arena = Arena::new(machine(workers), rates()).with_snapshot(snapshot.clone());
        let mut shapes = Vec::new();
        for (name, model) in [("plain", plain.clone()), ("corrected", corrected.clone())] {
            let pinned = Constraints {
                block_candidates: ladder.clone(),
                split_axes: vec![0, 1, 2],
                model,
                ..Default::default()
            };
            let plan = Enumerating {
                concurrency: workers,
                ..Enumerating::default()
            }
            .plan(&workflow, &pinned)
            .unwrap_or_else(|err| panic!("{workers} workers, {name}: {err}"));
            shapes.push(format!(
                "{} phase(s), blocks {:?}",
                plan.decomposition.n_phases(),
                plan.decomposition
                    .phases
                    .iter()
                    .map(|phase| phase.grid.block()[0])
                    .collect::<Vec<_>>()
            ));
            arena
                .enter_plan(name.to_string(), plan, pinned)
                .expect("a plan the arena can hold");
        }
        let judgement = arena.judge(&workflow).expect("both plans simulate");
        let plain_ns = judgement.verdicts[0].simulated_ns();
        let corrected_ns = judgement.verdicts[1].simulated_ns();
        let speedup = plain_ns / corrected_ns;
        println!(
            "{workers:>8} {:<34} {:<34} {speedup:>9.2}x",
            shapes[0], shapes[1]
        );

        if workers == 1 {
            assert_eq!(
                shapes[0], shapes[1],
                "at the negative control the objective is monotone in the edge and the \
                 corrections cannot move it; if they now do, this test is where the new \
                 behaviour gets written down"
            );
            assert_eq!(plain_ns, corrected_ns, "the same plan is the same run");
            continue;
        }

        assert_ne!(
            shapes[0], shapes[1],
            "{workers} workers: the corrections changed no plan, so nothing here is being \
             adjudicated"
        );
        differed += 1;
        assert!(
            speedup > 1.0,
            "{workers} workers: the plan the corrections chose is {speedup:.3}x the plain \
             one — the simulator says acting on the measurements made the run slower, which \
             is a finding and not a passing test"
        );
    }
    assert_eq!(
        differed, 2,
        "both worker counts above the control must move"
    );
}

// ------------------------------------------------- claim 4: deterministic --

/// The same field judged twice is the same table.
///
/// The arena exists to adjudicate planner changes, so a figure that moves
/// between two runs of one binary is worse than no figure at all: it would let
/// a change be accepted or rejected by whichever run someone quoted.
#[test]
fn judging_the_same_field_twice_gives_the_same_table() {
    let workflow = workflow();
    let build = || {
        let mut arena = Arena::new(machine(2), rates());
        for (name, search) in [
            ("dp", PartitionSearch::Dp),
            ("exhaustive", PartitionSearch::Exhaustive),
        ] {
            arena
                .enter(
                    name,
                    &Enumerating {
                        concurrency: 2,
                        search,
                        ..Enumerating::default()
                    },
                    &workflow,
                    &constraints(vec![16, 32]),
                )
                .expect("a plan");
        }
        arena.judge(&workflow).expect("a judgement")
    };
    let first = build();
    let second = build();
    assert_eq!(first, second);
    println!("{}", first.table());
}

// -------------------------------------- the cost of the dispatch loop --

/// A scheduler that takes the first ready task without looking at the rest.
///
/// Not a scheduler anybody should use — it is the **`O(1)` control** for the
/// measurement below, and the only way to separate what the dispatch loop costs
/// from what asking a `Scheduler` costs.
struct FirstReady;

impl Scheduler for FirstReady {
    fn name(&self) -> &'static str {
        "first-ready"
    }

    fn pick(&mut self, _decision: &Decision<'_>) -> usize {
        0
    }
}

/// **What the maintained ready set cost, and where the rest of the time went** —
/// the second half of G1, and the reason it was on that list at all.
///
/// `simulate` used to rebuild its ready set with a
/// `(0..graph.tasks.len()).filter(..)` at **every dispatch**, so a run of `T`
/// tasks evaluated the readiness predicate `O(T^2)` times.
/// `docs/design/planner-gaps.md` calls that *"fine for a 4^3 fixture and not
/// for an arena"*. It is now maintained incrementally — admitted when a last
/// dependency completes or a barrier clears — and the scan survives only as the
/// `debug_assertions` oracle the maintained set is compared against.
///
/// Measured on this machine, release, best of three, a three-phase pixel chain
/// on a `128^3` volume at four workers:
///
/// ```text
///      tasks   scan (ms)   maintained (ms)   scan us/task   maintained us/task
///         24         2.5               2.7          103.3                113.6
///        192         4.3               4.1           22.2                 21.5
///      1 536        28.1              13.9           18.3                  9.0
///     12 288       897.2             424.4           73.0                 34.5
///     98 304    79 207.1          36 298.9          805.7                369.3
/// ```
///
/// The `scan` column was taken with the old body reinstated in place of the
/// maintained set; the rest of the file was the same. The two small rows are
/// noise — at 24 tasks the run is dominated by building the task graph — and
/// the claim is the two large ones.
///
/// **2.2x at 98 304 tasks, and still quadratic** — which is the finding, not a
/// disappointment. The readiness scan was one of *two* `O(T^2)` terms in the
/// loop and the smaller one. The other is the `Scheduler` itself: it is handed
/// `Decision::ready` as a slice and every scheduler in the crate walks all of
/// it, so a dispatch costs `O(ready)` however cheap the loop around it is. The
/// `FirstReady` column of the same run is that term isolated:
///
/// ```text
///      tasks   ExecutorOrder (ms)   FirstReady (ms)   share in the scheduler
///      1 536                 13.9               9.0                    35 %
///     12 288                424.4              35.6                    92 %
///     98 304             36 298.9             567.2                    98 %
/// ```
///
/// At the scale an arena sweep works at, **98% of a simulation is the scheduler
/// scanning the ready set**, and fixing that is a change to the `Scheduler`
/// trait rather than to this loop: `CacheAware` and the handout policies want
/// the whole set by design. Recorded in `docs/design/planner-gaps.md` as the
/// next thing in the way.
///
/// **What was done about it**: `Machine::candidate_window` bounds how much of
/// the ready set a scheduler is shown, which caps the term rather than making
/// any one scheduler cheaper. `tests/candidate_window.rs` is what it costs —
/// 49x at 98 304 tasks with a bit-identical schedule for `ExecutorOrder`, and
/// not free for a policy that looks past the front of the set. The window is
/// `0` by default, so every figure above is still the figure this loop
/// produces.
///
/// **Ignored, because it is a measurement.** It also has to be run
/// `--release`: the maintained set carries a `debug_assertions` oracle that
/// runs the old scan on every dispatch and compares, so a debug build times
/// both implementations at once. That oracle is why no separate correctness
/// test appears here — every simulation in the suite already runs both and
/// compares them.
#[test]
#[ignore = "a measurement, not an assertion"]
fn print_the_cost_of_the_dispatch_loop() {
    use blockflow::assemble::PlanBuilder;
    use blockflow::geometry::BlockGrid;
    use blockflow::probes::IdentityOp;
    use blockflow::simulate::{simulate, ExecutorOrder, PerPhase};
    use std::collections::BTreeSet;
    use std::time::Instant;

    let volume = [128usize, 128, 128];
    println!(
        "{:>8} {:>10} {:>14} {:>14} {:>16} {:>12}",
        "edge", "tasks", "wall (ms)", "us/task", "first-ready (ms)", "share"
    );
    for edge in [64usize, 32, 16, 8, 4] {
        let grid = BlockGrid::new(volume, [edge, edge, edge]).expect("a grid");
        let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
        for name in ["first", "second", "third"] {
            builder
                .pixels(Chain::op(IdentityOp::new(name, [1, 1, 1])))
                .expect("a pixel phase");
        }
        let assembly = builder.finish().expect("an assembly");
        let tasks = assembly.decomposition.n_tasks();
        let run = |scheduler: &mut dyn Scheduler, repetitions: usize| -> f64 {
            let mut best = f64::INFINITY;
            for _ in 0..repetitions {
                let started = Instant::now();
                let outcome = simulate(
                    &assembly.decomposition,
                    &assembly.work(),
                    &machine(4),
                    &rates(),
                    &BTreeSet::new(),
                    &BTreeSet::new(),
                    PerPhase::default(),
                    scheduler,
                )
                .expect("a simulable plan");
                assert_eq!(outcome.tasks_run as usize, tasks);
                best = best.min(started.elapsed().as_secs_f64());
            }
            best
        };
        let best = run(&mut ExecutorOrder::phase_major(), 3);
        // Once: it is the cheap arm, and the point is the ratio.
        let control = run(&mut FirstReady, 1);
        println!(
            "{edge:>8} {tasks:>10} {:>14.1} {:>16.1} {:>16.1} {:>11.0}%",
            best * 1e3,
            best * 1e6 / tasks as f64,
            control * 1e3,
            100.0 * (1.0 - control / best)
        );
    }
}
