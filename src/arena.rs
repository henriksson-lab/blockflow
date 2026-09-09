// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! **The planner arena: a competition between plans, judged by the simulator.**
//!
//! # The gap this closes
//!
//! `docs/design/planner-gaps.md` opens on it: *"Nothing has ever fed a
//! `Strategy`-produced `Decomposition` into `simulate`."* Before this file,
//! nothing in `src/` called [`simulate`](crate::simulate::simulate) at all, and
//! the two test suites that did built every plan by hand with `PlanBuilder`.
//! The [`Scheduler`] trait picks among *ready tasks*, so the simulator ranked
//! **schedulers over one plan**; nothing ranked **plans**. That is the wrong way
//! round for a crate whose planner has a search in it and whose cost model is
//! the only thing that has ever adjudicated the search.
//!
//! So the arena is the mirror image of the suite that already exists:
//!
//! | | varies | held fixed | judge |
//! |---|---|---|---|
//! | `tests/simulate_ranks.rs` | the scheduler | the plan | the simulator |
//! | **this file** | **the plan** | the scheduler | the simulator |
//!
//! # The two judges, and why both are here
//!
//! Every entrant is scored twice, and the finding is the *disagreement*:
//!
//! * [`Verdict::priced_ns`](crate::arena::Verdict::priced_ns) — the **planner's own objective**, the sum over
//!   phases of [`phase_price`](crate::strategy::phase_price)'s makespan. This is not a re-implementation of
//!   it: `PhasePricer` calls the same function while it sweeps candidates, so a
//!   change to the objective moves both sides of this comparison at once. What
//!   the arena adds is the ability to apply it to a plan the search did *not*
//!   produce;
//! * [`Verdict::outcome`](crate::arena::Verdict::outcome) — what [`simulate`](crate::simulate::simulate) did
//!   with the same plan: a makespan built from a task graph, a cache, an IO
//!   channel and a scheduler, none of which the cost model has.
//!
//! [`Judgement::regret`](crate::arena::Judgement::regret) is the number to read. It is the simulated makespan of
//! the plan the **cost model** would pick, over the best simulated makespan in
//! the field: *what trusting the planner costs, in the simulator's units*. A
//! regret of `1.0` says the model picked the simulator's winner. Nothing here
//! asserts that it does.
//!
//! Each [`Verdict`](crate::arena::Verdict) also states whether the entrant was
//! admissible under its own constraints. Inadmissible entrants keep their
//! prices and simulated outcomes in the table, but winner selection skips them,
//! so a faster-looking plan can be shown and rejected in the same report.
//!
//! # What this is not
//!
//! **Not a runtime prediction.** `simulate`'s own header says it ranks designs
//! and does not predict runtimes, and every limit listed there is inherited
//! whole: workers do not contend unless
//! [`Machine::contention`](crate::simulate::Machine::contention) says so. The
//! simulator default dispatches continuously because that is what the
//! distributed coordinator does;
//! [`Machine::wave_synchronous`](crate::simulate::Machine::wave_synchronous) is
//! the explicit model for the current in-process executor. A regret figure is evidence that
//! two rankings differ, not a measurement of seconds anybody will wait.
//!
//! **Not a search.** The arena judges the entrants it is handed. Turning a
//! disagreement into a better planner is the work `planner-gaps.md` lists after
//! this one; the arena is what makes that work adjudicable rather than
//! arguable.
//!
//! **Pixel phases only, and that is a limit of the *price*, not of the
//! simulator.** [`Strategy::decompose`] partitions chain slots, so every phase
//! it produces is
//! [`PhaseWork::Pixels`](crate::fragment::PhaseWork::Pixels) — and the
//! planner's objective is defined over exactly that: a run of chain slots, one
//! traversal per image read, writing the image after it. A fragment phase owns
//! no slots and reads no image that way, so pricing one with the pixel rule
//! would be inventing a number for the half of the comparison that is supposed
//! to be the planner's. [`price_plan`](crate::arena::price_plan) refuses such a
//! phase rather than charging it something. `simulate` itself is happy with
//! fragment and iterative work — `tests/simulate_ranks.rs` runs both — so what
//! this file would need to judge one is an objective to compare against, which
//! the planner does not have.
//!
//! [`Scheduler`]: crate::simulate::Scheduler
//! [`Machine::contention`]: crate::simulate::Machine::contention
//! [`Strategy::decompose`]: crate::strategy::Strategy::decompose

use std::collections::BTreeSet;
use std::fmt;

use crate::decomposition::{
    images_read_by, resident_buffers_of, Constraints, Decomposition, PhaseTraffic,
};
use crate::error::{Error, Result};
use crate::fragment::PhaseWork;
use crate::geometry::BlockGrid;
use crate::simulate::{
    phase_rates_from_snapshot, simulate, ExecutorOrder, Machine, Outcome, PerPhase, Rates,
    Scheduler,
};
use crate::statistics::Snapshot;
use crate::strategy::{phase_price, Enumerating, PartitionSearch, Plan, Strategy, Workflow};

/// One plan entered into the competition, and what it was planned under.
///
/// The `Constraints` travel with the plan because the cost model's coefficients
/// are in them: two entrants planned under two models are two plans *and* two
/// objectives, and pricing them both against one of the two would be scoring
/// one entrant with the other's ruler.
pub struct Entrant {
    /// How this plan was produced, in the caller's words. Printed, and used to
    /// name a winner.
    pub name: String,
    pub plan: Plan,
    pub constraints: Constraints,
}

/// The structural shape of a plan, independent of cost and scheduling.
///
/// This is intentionally more than a printed edge list. The planner arena uses
/// shape as evidence and as a duplicate key; recording only one block axis can
/// merge distinct anisotropic plans.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PlanShape {
    pub phases: Vec<PhaseShape>,
}

impl PlanShape {
    pub fn from_plan(plan: &Plan) -> Self {
        Self::from_decomposition(&plan.decomposition)
    }

    pub fn from_decomposition(decomposition: &Decomposition) -> Self {
        Self {
            phases: decomposition
                .phases
                .iter()
                .map(|phase| PhaseShape {
                    slots: phase.slots.clone(),
                    block: phase.grid.block(),
                    blocks: phase.grid.n_blocks(),
                })
                .collect(),
        }
    }

    /// Transitional convenience for reports that still print the old shape.
    pub fn first_axis_edges(&self) -> Vec<usize> {
        self.phases.iter().map(|phase| phase.block[0]).collect()
    }
}

impl fmt::Display for PlanShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let blocks: Vec<[usize; 3]> = self.phases.iter().map(|phase| phase.block).collect();
        write!(f, "{} phase(s) at {:?}", self.phases.len(), blocks)
    }
}

/// One phase's contribution to [`PlanShape`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhaseShape {
    pub slots: Vec<usize>,
    pub block: [usize; 3],
    pub blocks: usize,
}

/// What both judges said about one entrant.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub name: String,
    /// Full phase/block shape. Use this for equality, deduplication and reports.
    pub shape: PlanShape,
    /// Whether this plan fits the admission contract it was entered with.
    pub admissible: bool,
    /// **The planner's objective**: the sum over phases of [`phase_price`]'s
    /// makespan, at the arena's worker count.
    pub priced_ns: f64,
    /// **The simulator's answer** for the same plan.
    pub outcome: Outcome,
}

impl Verdict {
    /// The simulated makespan, which is the arena's ranking key.
    pub fn simulated_ns(&self) -> f64 {
        self.outcome.makespan_ns as f64
    }
}

/// The field, judged.
#[derive(Debug, Clone, PartialEq)]
pub struct Judgement {
    pub verdicts: Vec<Verdict>,
    /// The worker count both judges were told about. Stated here because the
    /// two used to be able to differ silently — `planner-gaps.md`, item E.
    pub workers: usize,
}

impl Judgement {
    /// The entrant with the lowest **priced** cost: the plan a planner that
    /// enumerated this field would return. Ties go to the earlier entrant, which
    /// is the order the caller entered them in.
    pub fn model_pick(&self) -> Option<&Verdict> {
        self.verdicts
            .iter()
            .filter(|verdict| verdict.admissible)
            .min_by(|a, b| a.priced_ns.total_cmp(&b.priced_ns))
    }

    /// The entrant with the lowest **simulated** makespan.
    pub fn simulated_pick(&self) -> Option<&Verdict> {
        self.verdicts
            .iter()
            .filter(|verdict| verdict.admissible)
            .min_by(|a, b| a.simulated_ns().total_cmp(&b.simulated_ns()))
    }

    /// **What trusting the cost model costs here**: the simulated makespan of
    /// the model's pick over the best simulated makespan in the field.
    ///
    /// `1.0` exactly when the two judges agree on the winner or when the
    /// model's pick ties it. Always at least `1.0`, because the denominator is a
    /// minimum over the same set. `None` for an empty field, and for one whose
    /// best simulated makespan is zero — a field of plans that do nothing is not
    /// a field a ratio says anything about.
    pub fn regret(&self) -> Option<f64> {
        let picked = self.model_pick()?.simulated_ns();
        let best = self.simulated_pick()?.simulated_ns();
        (best > 0.0).then(|| picked / best)
    }

    /// Pairs the two judges order differently, as `(cheaper by the model,
    /// cheaper in the simulator)`.
    ///
    /// The detail behind [`Self::regret`]: a field can have a regret of `1.0` —
    /// the model picked the winner — and still order everything below it wrong,
    /// which matters as soon as the winner is unaffordable for a reason neither
    /// judge holds.
    pub fn discordant_pairs(&self) -> Vec<(&str, &str)> {
        let mut out = Vec::new();
        for (left_index, left) in self.verdicts.iter().enumerate() {
            for right in &self.verdicts[left_index + 1..] {
                let model = left.priced_ns.total_cmp(&right.priced_ns);
                let simulated = left.simulated_ns().total_cmp(&right.simulated_ns());
                if model.is_eq() || simulated.is_eq() {
                    continue;
                }
                if model != simulated {
                    out.push((left.name.as_str(), right.name.as_str()));
                }
            }
        }
        out
    }

    /// Kendall's tau over the two rankings: `1.0` for identical orders, `-1.0`
    /// for reversed, `0.0` for unrelated. Tied pairs count as neither, which is
    /// tau-a; a field with many ties therefore reports a tau below one without
    /// any pair being ordered wrongly, and [`Self::discordant_pairs`] is the
    /// place to look when it does.
    pub fn kendall_tau(&self) -> Option<f64> {
        let mut concordant = 0i64;
        let mut discordant = 0i64;
        for (left_index, left) in self.verdicts.iter().enumerate() {
            for right in &self.verdicts[left_index + 1..] {
                let model = left.priced_ns.total_cmp(&right.priced_ns);
                let simulated = left.simulated_ns().total_cmp(&right.simulated_ns());
                if model.is_eq() || simulated.is_eq() {
                    continue;
                }
                if model == simulated {
                    concordant += 1;
                } else {
                    discordant += 1;
                }
            }
        }
        let total = concordant + discordant;
        (total > 0).then(|| (concordant - discordant) as f64 / total as f64)
    }

    /// The field as a table, ordered as entered.
    ///
    /// Ratios rather than absolutes in the last two columns, because neither
    /// judge's units mean anything on their own: the priced figure is in the
    /// cost model's nanoseconds and the simulated one in `Rates`'.
    pub fn table(&self) -> String {
        let best_priced = self
            .verdicts
            .iter()
            .map(|verdict| verdict.priced_ns)
            .fold(f64::INFINITY, f64::min);
        let best_simulated = self
            .verdicts
            .iter()
            .map(Verdict::simulated_ns)
            .fold(f64::INFINITY, f64::min);
        let mut out = format!(
            "planner arena, {} workers\n{:<34} {:>7} {:>5} {:>9} {:>10} {:>10} {:>12}\n",
            self.workers, "plan", "phases", "fit", "blocks", "priced", "simulated", "fetched MiB"
        );
        for verdict in &self.verdicts {
            let blocks: usize = verdict.shape.phases.iter().map(|phase| phase.blocks).sum();
            out.push_str(&format!(
                "{:<34} {:>7} {:>5} {:>9} {:>10.3} {:>10.3} {:>12.1}\n",
                verdict.name,
                verdict.shape.phases.len(),
                if verdict.admissible { "yes" } else { "no" },
                blocks,
                verdict.priced_ns / best_priced.max(f64::MIN_POSITIVE),
                verdict.simulated_ns() / best_simulated.max(f64::MIN_POSITIVE),
                verdict.outcome.fetched_bytes as f64 / (1024.0 * 1024.0),
            ));
        }
        if let (Some(model), Some(simulated)) = (self.model_pick(), self.simulated_pick()) {
            out.push_str(&format!(
                "the model picks {}; the simulator picks {}; regret {:.3}\n",
                model.name,
                simulated.name,
                self.regret().unwrap_or(f64::NAN)
            ));
        }
        out
    }
}

/// One execution case in a robust simulator-backed comparison.
///
/// `baseline_ns` is the local oracle for this case: candidate makespans are
/// divided by it to produce regret. The arena does not prescribe how that
/// baseline was chosen, which keeps the robust selector usable with committed
/// scenarios, ad hoc machines, or a caller's own acceptance corpus.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustCase {
    pub name: String,
    pub machine: Machine,
    pub rates: Rates,
    pub constraints: Constraints,
    pub snapshot: Option<Snapshot>,
    pub baseline_ns: f64,
    pub local: bool,
}

impl RobustCase {
    pub fn new(
        name: impl Into<String>,
        machine: Machine,
        rates: Rates,
        constraints: Constraints,
        baseline_ns: f64,
    ) -> Self {
        Self {
            name: name.into(),
            machine,
            rates,
            constraints,
            snapshot: None,
            baseline_ns,
            local: false,
        }
    }

    pub fn with_snapshot(mut self, snapshot: Snapshot) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    pub fn local(mut self) -> Self {
        self.local = true;
        self
    }
}

/// The entrant selected by robust simulator-backed scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustPick {
    pub name: String,
    pub plan: Plan,
    pub worst_regret: f64,
    pub local_regret: f64,
    pub fit_cases: usize,
}

/// The competition: a machine, a set of rates, and the plans entered so far.
///
/// The machine and the rates are held here and not per entrant on purpose. A
/// competition in which two plans are judged on two machines ranks nothing, and
/// making that unrepresentable is cheaper than checking for it.
pub struct Arena {
    pub machine: Machine,
    pub rates: Rates,
    /// Measurements the **simulator** is told about, per phase.
    ///
    /// `None` charges every phase [`Rates::compute_ns_per_voxel`], which is one
    /// number against a measured 57x spread and is the simulator's own stated
    /// weakest point. With a snapshot, each entrant's phases are priced by
    /// [`phase_rates_from_snapshot`] — `sum over slots of declared x measured`.
    ///
    /// **The point of having it here is that the other judge can be told the
    /// same thing.** `CostModel::compute_of` carries the same evidence to the
    /// planner, through `Snapshot::calibrate`; a field judged with a snapshot
    /// here and priced under a model calibrated from it is the two judges given
    /// one set of measurements, which is the only arrangement in which their
    /// disagreement is about the *models* rather than about what each was told.
    snapshot: Option<Snapshot>,
    entrants: Vec<Entrant>,
}

impl Arena {
    pub fn new(machine: Machine, rates: Rates) -> Self {
        Self {
            machine,
            rates,
            snapshot: None,
            entrants: Vec::new(),
        }
    }

    /// Tell the simulator what a run measured; see [`Self::snapshot`].
    pub fn with_snapshot(mut self, snapshot: Snapshot) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    /// Ask a strategy for a plan and enter it.
    ///
    /// The whole of the path `planner-gaps.md` says does not exist: a
    /// `Strategy`, a `Workflow` and a `Constraints` go in, and a plan the
    /// simulator will run comes out.
    pub fn enter(
        &mut self,
        name: impl Into<String>,
        strategy: &dyn Strategy,
        workflow: &Workflow,
        constraints: &Constraints,
    ) -> Result<()> {
        let plan = strategy.plan(workflow, constraints)?;
        self.enter_plan(name, plan, constraints.clone())
    }

    /// Enter a plan **nobody's strategy produced** — a hand-built one, or one
    /// from a strategy configured in a way no caller would ship.
    ///
    /// This is what makes the arena a measuring instrument rather than a
    /// regression test on the search: the interesting question is whether the
    /// search's argmin is the simulator's, and asking it means entering the
    /// plans the search rejected.
    pub fn enter_plan(
        &mut self,
        name: impl Into<String>,
        plan: Plan,
        constraints: Constraints,
    ) -> Result<()> {
        self.entrants.push(Entrant {
            name: name.into(),
            plan,
            constraints,
        });
        Ok(())
    }

    pub fn entrants(&self) -> &[Entrant] {
        &self.entrants
    }

    /// Price every entrant with the planner's objective, run every entrant
    /// through the simulator, and report both.
    ///
    /// The scheduler is [`ExecutorOrder::phase_major`] for every entrant — the
    /// shipped default, `Hints::default()`'s own policy, and the one that shares
    /// `strategy::priority_key` with the real dispatcher — so that what varies
    /// between entrants is the plan and not the order a scheduler happens to
    /// like. The arena ranks plans; `tests/simulate_ranks.rs` ranks schedulers.
    /// [`Self::judge_with`] states a different one.
    pub fn judge(&self, workflow: &Workflow) -> Result<Judgement> {
        self.judge_with(workflow, &mut || Box::new(ExecutorOrder::phase_major()))
    }

    /// [`Self::judge`] with the scheduler stated.
    ///
    /// A factory rather than a scheduler, because a `Scheduler` is `&mut` for
    /// the length of a run and every entrant must be judged by a fresh one: a
    /// scheduler carrying state from the previous plan would make an entrant's
    /// figure depend on what was entered before it.
    ///
    /// **The scheduler is a lever of the machine, not of the plan.** Holding it
    /// fixed is what makes the field a competition between plans; varying it —
    /// which is what `the_phases_overlap_only_under_the_policy_that_fuses` does
    /// — asks a different question, and one the arena is the right instrument
    /// for only because it can hold everything else still.
    pub fn judge_with(
        &self,
        workflow: &Workflow,
        make_scheduler: &mut dyn FnMut() -> Box<dyn Scheduler>,
    ) -> Result<Judgement> {
        let mut verdicts = Vec::with_capacity(self.entrants.len());
        for entrant in &self.entrants {
            let decomposition = &entrant.plan.decomposition;
            // A plan that does not tile is not a plan. The executor refuses it
            // and so does this, rather than reporting a number for a run that
            // could not happen.
            decomposition.check()?;
            let priced_ns = price_plan(
                workflow,
                decomposition,
                &entrant.constraints,
                self.machine.workers,
            )?;
            let admissible = admissible_plan(workflow, decomposition, &entrant.constraints)?;
            // Every phase a `Strategy` produces is a run of chain slots; see the
            // module header for why the arena holds no other kind.
            let work = vec![PhaseWork::Pixels; decomposition.n_phases()];
            // Per-phase compute rates, where a run has measured them. Derived
            // per entrant because a rate is per *phase* and two entrants are two
            // partitions — the same measurement, folded onto different phases.
            let phase_rates: Vec<f64> = match &self.snapshot {
                Some(snapshot) => phase_rates_from_snapshot(
                    snapshot,
                    decomposition,
                    &workflow.chain.slots(),
                    self.rates.compute_ns_per_voxel,
                ),
                None => Vec::new(),
            };
            let mut scheduler = make_scheduler();
            let outcome = simulate(
                decomposition,
                &work,
                &self.machine,
                &self.rates,
                &entrant.plan.hints.release_images,
                &entrant.plan.hints.keep_images,
                PerPhase {
                    ns_per_voxel: &phase_rates,
                    ..PerPhase::default()
                },
                scheduler.as_mut(),
            )?;
            let shape = PlanShape::from_decomposition(decomposition);
            verdicts.push(Verdict {
                name: entrant.name.clone(),
                shape,
                admissible,
                priced_ns,
                outcome,
            });
        }
        Ok(Judgement {
            verdicts,
            workers: self.machine.workers,
        })
    }

    /// Pick the entrant with the lowest worst regret across `cases`.
    ///
    /// Ties go to the lower regret on the case marked `local`, then to the
    /// entrant order. Plans that do not fit a case are skipped for that case;
    /// a candidate must fit the local case and at least one case overall.
    pub fn robust_pick_with(
        &self,
        workflow: &Workflow,
        cases: &[RobustCase],
        make_scheduler: &mut dyn FnMut() -> Box<dyn Scheduler>,
    ) -> Result<RobustPick> {
        if cases.is_empty() {
            return Err(Error::InvalidArgument(
                "robust simulator-backed: no execution cases".into(),
            ));
        }
        if !cases.iter().any(|case| case.local) {
            return Err(Error::InvalidArgument(
                "robust simulator-backed: no local execution case".into(),
            ));
        }
        for case in cases {
            if case.baseline_ns <= 0.0 {
                return Err(Error::InvalidArgument(format!(
                    "robust simulator-backed: case {} has non-positive baseline {}",
                    case.name, case.baseline_ns
                )));
            }
        }

        let mut best: Option<RobustPick> = None;
        for entrant in &self.entrants {
            let mut worst_regret = 1.0f64;
            let mut local_regret = None;
            let mut fit_cases = 0usize;
            for case in cases {
                match plan_fit(
                    workflow,
                    &entrant.plan.decomposition,
                    &case.constraints,
                    case.machine.workers.max(1),
                )? {
                    PlanFit::Fits { .. } => {}
                    PlanFit::OverBudget { .. } => continue,
                }

                let mut arena = Arena::new(case.machine, case.rates);
                if let Some(snapshot) = &case.snapshot {
                    arena = arena.with_snapshot(snapshot.clone());
                }
                arena.enter_plan(
                    entrant.name.clone(),
                    entrant.plan.clone(),
                    case.constraints.clone(),
                )?;
                let judgement = arena.judge_with(workflow, make_scheduler)?;
                let simulated_ns = judgement.verdicts[0].simulated_ns();
                let regret = simulated_ns / case.baseline_ns;
                worst_regret = worst_regret.max(regret);
                fit_cases += 1;
                if case.local {
                    local_regret = Some(regret);
                }
            }

            let Some(local_regret) = local_regret else {
                continue;
            };
            if fit_cases == 0 {
                continue;
            }
            let pick = RobustPick {
                name: entrant.name.clone(),
                plan: entrant.plan.clone(),
                worst_regret,
                local_regret,
                fit_cases,
            };
            if best.as_ref().is_none_or(|current| {
                pick.worst_regret < current.worst_regret
                    || (pick.worst_regret == current.worst_regret
                        && pick.local_regret < current.local_regret)
            }) {
                best = Some(pick);
            }
        }

        best.ok_or_else(|| {
            Error::InvalidArgument("robust simulator-backed: no candidate fits local case".into())
        })
    }
}

/// Builds an [`Arena`] candidate field while making duplicate plans
/// unrepresentable.
///
/// Candidate fields are planner evidence: reports, simulator-backed ranking and
/// transfer tests must all agree on which plans were considered. This helper is
/// the shared path for entering strategy plans, pinned block-edge variants and
/// mixed per-phase edge variants.
pub struct CandidateFieldBuilder {
    arena: Arena,
    seen: BTreeSet<PlanSignature>,
}

impl CandidateFieldBuilder {
    pub fn new(machine: Machine, rates: Rates) -> Self {
        Self {
            arena: Arena::new(machine, rates),
            seen: BTreeSet::new(),
        }
    }

    pub fn with_snapshot(mut self, snapshot: Snapshot) -> Self {
        self.arena = self.arena.with_snapshot(snapshot);
        self
    }

    pub fn enter_strategy(
        &mut self,
        name: impl Into<String>,
        strategy: &dyn Strategy,
        workflow: &Workflow,
        constraints: &Constraints,
    ) -> Result<bool> {
        let plan = strategy.plan(workflow, constraints)?;
        self.enter_plan(name, plan, constraints.clone())
    }

    pub fn enter_plan(
        &mut self,
        name: impl Into<String>,
        plan: Plan,
        constraints: Constraints,
    ) -> Result<bool> {
        if !self.seen.insert(PlanSignature(PlanShape::from_plan(&plan))) {
            return Ok(false);
        }
        self.arena.enter_plan(name, plan, constraints)?;
        Ok(true)
    }

    pub fn enter_pinned_edges(
        &mut self,
        label: impl Fn(usize) -> String,
        strategy: &dyn Strategy,
        workflow: &Workflow,
        constraints: &Constraints,
    ) -> Result<()> {
        for &edge in &constraints.block_candidates {
            let pinned = Constraints {
                block_candidates: vec![edge],
                ..constraints.clone()
            };
            if let Ok(plan) = strategy.plan(workflow, &pinned) {
                self.enter_plan(label(edge), plan, pinned)?;
            }
        }
        Ok(())
    }

    pub fn enter_mixed_edges(
        &mut self,
        label: impl Fn(usize, &[usize]) -> String,
        edges: &[usize],
        constraints: &Constraints,
    ) -> Result<()> {
        let seeds: Vec<Plan> = self
            .arena
            .entrants()
            .iter()
            .map(|entrant| entrant.plan.clone())
            .collect();
        for (seed_index, seed) in seeds.iter().enumerate() {
            for (chosen, plan) in mixed_edge_plans(seed, edges)? {
                self.enter_plan(label(seed_index, &chosen), plan, constraints.clone())?;
            }
        }
        Ok(())
    }

    pub fn finish(self) -> Arena {
        self.arena
    }
}

/// Which plan variants enter a simulator-backed candidate field.
///
/// The policy owns field construction, not the scoring objective. Robust and
/// minimax selection stay explicit on [`Arena`] so callers can see whether they
/// are choosing a local simulator winner or a cross-scenario compromise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidatePolicy {
    pub include_strategy: bool,
    pub include_partition_variants: bool,
    pub include_pinned_edges: bool,
    pub include_mixed_edges: bool,
}

impl CandidatePolicy {
    pub const fn simulator_backed() -> Self {
        Self {
            include_strategy: true,
            include_partition_variants: false,
            include_pinned_edges: true,
            include_mixed_edges: false,
        }
    }

    pub const fn simulator_backed_mixed() -> Self {
        Self {
            include_mixed_edges: true,
            ..Self::simulator_backed()
        }
    }

    pub const fn oracle_report() -> Self {
        Self {
            include_strategy: true,
            include_partition_variants: true,
            include_pinned_edges: true,
            include_mixed_edges: true,
        }
    }

    pub fn build_for_strategy(
        &self,
        strategy_label: impl Into<String>,
        strategy: &dyn Strategy,
        workflow: &Workflow,
        constraints: &Constraints,
        machine: Machine,
        rates: Rates,
        snapshot: Option<Snapshot>,
    ) -> Result<Arena> {
        if self.include_partition_variants {
            return Err(Error::InvalidArgument(
                "candidate policy: partition variants require an Enumerating strategy".into(),
            ));
        }
        let mut field = self.builder(machine, rates, snapshot);
        self.enter_strategy(&mut field, strategy_label, strategy, workflow, constraints)?;
        self.enter_pinned_edges(&mut field, strategy, workflow, constraints)?;
        self.enter_mixed_edges(&mut field, constraints)?;
        Ok(field.finish())
    }

    pub fn build_for_enumerating(
        &self,
        strategy_label: impl Into<String>,
        strategy: &Enumerating,
        workflow: &Workflow,
        constraints: &Constraints,
        machine: Machine,
        rates: Rates,
        snapshot: Option<Snapshot>,
    ) -> Result<Arena> {
        let mut field = self.builder(machine, rates, snapshot);
        self.enter_strategy(&mut field, strategy_label, strategy, workflow, constraints)?;
        if self.include_partition_variants {
            for (label, search) in [
                ("search-dp", PartitionSearch::Dp),
                ("search-exhaustive", PartitionSearch::Exhaustive),
                ("search-single", PartitionSearch::SingleGroup),
            ] {
                let variant = Enumerating {
                    concurrency: strategy.concurrency,
                    search,
                    ..Enumerating::default()
                };
                if let Ok(plan) = variant.plan(workflow, constraints) {
                    field.enter_plan(label.to_string(), plan, constraints.clone())?;
                }
            }
        }
        self.enter_pinned_edges(&mut field, strategy, workflow, constraints)?;
        self.enter_mixed_edges(&mut field, constraints)?;
        Ok(field.finish())
    }

    fn builder(
        &self,
        machine: Machine,
        rates: Rates,
        snapshot: Option<Snapshot>,
    ) -> CandidateFieldBuilder {
        let mut field = CandidateFieldBuilder::new(machine, rates);
        if let Some(snapshot) = snapshot {
            field = field.with_snapshot(snapshot);
        }
        field
    }

    fn enter_strategy(
        &self,
        field: &mut CandidateFieldBuilder,
        strategy_label: impl Into<String>,
        strategy: &dyn Strategy,
        workflow: &Workflow,
        constraints: &Constraints,
    ) -> Result<()> {
        if self.include_strategy {
            field.enter_strategy(strategy_label, strategy, workflow, constraints)?;
        }
        Ok(())
    }

    fn enter_pinned_edges(
        &self,
        field: &mut CandidateFieldBuilder,
        strategy: &dyn Strategy,
        workflow: &Workflow,
        constraints: &Constraints,
    ) -> Result<()> {
        if self.include_pinned_edges {
            field.enter_pinned_edges(
                |edge| format!("edge-{edge}"),
                strategy,
                workflow,
                constraints,
            )?;
        }
        Ok(())
    }

    fn enter_mixed_edges(
        &self,
        field: &mut CandidateFieldBuilder,
        constraints: &Constraints,
    ) -> Result<()> {
        if self.include_mixed_edges {
            field.enter_mixed_edges(
                |seed_index, chosen| format!("mixed-{seed_index}-{chosen:?}"),
                &constraints.block_candidates,
                constraints,
            )?;
        }
        Ok(())
    }
}

/// An opt-in strategy wrapper that lets the simulator choose among candidates.
///
/// The wrapped strategy still builds every candidate. This wrapper only changes
/// the judge: it enters the wrapped strategy's normal plan plus the same
/// strategy pinned to each stated block candidate, runs that field through
/// [`Arena`], and returns the simulator winner.
///
/// This is intentionally separate from the default planner. It is a measuring
/// and experimentation strategy for closing planner regret, not a claim that
/// simulator-backed ranking should be the production default everywhere.
pub struct SimulatorBacked<S, F>
where
    S: Strategy,
    F: Fn() -> Box<dyn Scheduler> + Sync,
{
    pub strategy: S,
    pub machine: Machine,
    pub machine_variants: Vec<(String, Machine)>,
    pub rates: Rates,
    pub snapshot: Option<Snapshot>,
    pub make_scheduler: F,
    pub candidate_policy: CandidatePolicy,
}

/// The simulator-backed winner with the execution contract it was judged under.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulatedPlan {
    pub name: String,
    pub plan: Plan,
    pub machine_name: String,
    pub machine: Machine,
    pub regret: f64,
    pub judgement: Judgement,
}

/// How [`SimulatorBacked`] chooses across named machine contracts.
///
/// `FastestOracle` asks "which candidate field produced the shortest simulated
/// runtime?" `LowestPlannerRegret` asks "which execution contract makes the
/// planner's model pick closest to the simulator pick?" The second form is for
/// explicit policy decisions such as wave-synchronous dispatch, where closing a
/// planner/executor mismatch may be more important than selecting the fastest
/// sampled continuous-overlap oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationObjective {
    FastestOracle,
    LowestPlannerRegret,
}

impl<S, F> SimulatorBacked<S, F>
where
    S: Strategy,
    F: Fn() -> Box<dyn Scheduler> + Sync,
{
    pub fn new(strategy: S, machine: Machine, rates: Rates, make_scheduler: F) -> Self {
        Self {
            strategy,
            machine,
            machine_variants: Vec::new(),
            rates,
            snapshot: None,
            make_scheduler,
            candidate_policy: CandidatePolicy::simulator_backed(),
        }
    }

    pub fn with_snapshot(mut self, snapshot: Snapshot) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    /// Add every per-phase combination of the stated scalar block candidates,
    /// preserving each seed plan's phase partition.
    ///
    /// This is explicit because it can grow quickly: a three-phase plan and a
    /// four-rung ladder adds `4^3` candidates for that partition.
    pub fn with_mixed_edges(mut self) -> Self {
        self.candidate_policy = CandidatePolicy::simulator_backed_mixed();
        self
    }

    /// Add a named machine contract to the simulator-backed choice.
    ///
    /// The default `machine` passed to [`Self::new`] is always considered. This
    /// adds alternatives such as a wave-synchronous dispatch model without
    /// hiding that the returned plan was chosen under a different execution
    /// contract.
    pub fn with_machine_variant(mut self, name: impl Into<String>, machine: Machine) -> Self {
        self.machine_variants.push((name.into(), machine));
        self
    }

    fn candidate_field(
        &self,
        workflow: &Workflow,
        constraints: &Constraints,
        machine: Machine,
    ) -> Result<Arena> {
        self.candidate_policy.build_for_strategy(
            "strategy",
            &self.strategy,
            workflow,
            constraints,
            machine,
            self.rates,
            self.snapshot.clone(),
        )
    }

    /// Build the default-machine candidate arena without choosing a winner.
    ///
    /// This is for reports and robust-selection experiments that need to judge
    /// the same candidate field under several machines before committing to a
    /// plan.
    pub fn candidate_arena(&self, workflow: &Workflow, constraints: &Constraints) -> Result<Arena> {
        self.candidate_field(workflow, constraints, self.machine)
    }

    /// Choose the simulator winner and state the machine contract it was judged
    /// under.
    pub fn plan_with_machine(
        &self,
        workflow: &Workflow,
        constraints: &Constraints,
    ) -> Result<SimulatedPlan> {
        self.plan_with_machine_by(workflow, constraints, SimulationObjective::FastestOracle)
    }

    /// Choose a simulator-backed plan under the requested cross-machine
    /// objective.
    pub fn plan_with_machine_by(
        &self,
        workflow: &Workflow,
        constraints: &Constraints,
        objective: SimulationObjective,
    ) -> Result<SimulatedPlan> {
        let mut cases = Vec::with_capacity(1 + self.machine_variants.len());
        cases.push(("default".to_string(), self.machine));
        cases.extend(self.machine_variants.iter().cloned());

        let mut best: Option<(SimulatedPlan, f64, f64)> = None;
        for (machine_name, machine) in cases {
            let arena = self.candidate_field(workflow, constraints, machine)?;
            let judgement = arena.judge_with(workflow, &mut || (self.make_scheduler)())?;
            let winner = judgement.simulated_pick().ok_or_else(|| {
                Error::InvalidArgument("simulator-backed: no candidate fit".into())
            })?;
            let simulated_ns = winner.simulated_ns();
            let name = winner.name.clone();
            let regret = judgement.regret().unwrap_or(f64::NAN);
            let plan = arena
                .entrants()
                .iter()
                .find(|entrant| entrant.name == name)
                .map(|entrant| entrant.plan.clone())
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "simulator-backed: simulator picked unknown entrant {name}"
                    ))
                })?;
            let candidate = SimulatedPlan {
                name,
                plan,
                machine_name,
                machine,
                regret,
                judgement,
            };
            let score = match objective {
                SimulationObjective::FastestOracle => simulated_ns,
                SimulationObjective::LowestPlannerRegret => regret,
            };
            if best
                .as_ref()
                .is_none_or(|(_, best_score, best_simulated_ns)| {
                    score < *best_score
                        || (score == *best_score && simulated_ns < *best_simulated_ns)
                })
            {
                best = Some((candidate, score, simulated_ns));
            }
        }
        best.map(|(candidate, _, _)| candidate)
            .ok_or_else(|| Error::InvalidArgument("simulator-backed: no machine cases".into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PlanSignature(PlanShape);

fn mixed_edge_plans(seed: &Plan, edges: &[usize]) -> Result<Vec<(Vec<usize>, Plan)>> {
    if seed
        .decomposition
        .phases
        .iter()
        .any(|phase| phase.reads_across_grids())
    {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut chosen = vec![0usize; seed.decomposition.n_phases()];
    fn visit(
        at: usize,
        seed: &Plan,
        edges: &[usize],
        chosen: &mut [usize],
        out: &mut Vec<(Vec<usize>, Plan)>,
    ) -> Result<()> {
        if at == chosen.len() {
            let mut phases = Vec::with_capacity(seed.decomposition.phases.len());
            for (phase, edge) in seed.decomposition.phases.iter().zip(chosen.iter().copied()) {
                let grid = BlockGrid::new(phase.volume(), [edge; 3])?;
                phases.push(phase.regrid_preserving_metadata(grid));
            }
            out.push((
                chosen.to_vec(),
                Plan {
                    decomposition: Decomposition {
                        volume: seed.decomposition.volume,
                        dtype: seed.decomposition.dtype,
                        phases,
                        chain_reach: seed.decomposition.chain_reach,
                    },
                    hints: seed.hints.clone(),
                },
            ));
            return Ok(());
        }
        for &edge in edges {
            chosen[at] = edge;
            visit(at + 1, seed, edges, chosen, out)?;
        }
        Ok(())
    }
    visit(0, seed, edges, &mut chosen, &mut out)?;
    Ok(out)
}

impl<S, F> Strategy for SimulatorBacked<S, F>
where
    S: Strategy,
    F: Fn() -> Box<dyn Scheduler> + Sync,
{
    fn name(&self) -> &'static str {
        "simulator-backed"
    }

    fn decompose(&self, workflow: &Workflow, constraints: &Constraints) -> Result<Decomposition> {
        self.plan(workflow, constraints)
            .map(|plan| plan.decomposition)
    }

    fn plan(&self, workflow: &Workflow, constraints: &Constraints) -> Result<Plan> {
        self.plan_with_machine(workflow, constraints)
            .map(|chosen| chosen.plan)
    }
}

/// Whether a plan fits the arena admission contract, with the first refusal
/// stated in the same adjusted bytes [`Constraints::affords_working_set`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanFit {
    Fits { bytes: u64 },
    OverBudget { bytes: u64, budget: u64 },
}

impl PlanFit {
    pub fn fits(self) -> bool {
        matches!(self, Self::Fits { .. })
    }

    pub fn with_admitted_value<T>(self, value: impl FnOnce() -> T) -> PlanAdmission<T> {
        match self {
            Self::Fits { bytes } => PlanAdmission::Fits {
                bytes,
                value: value(),
            },
            Self::OverBudget { bytes, budget } => PlanAdmission::OverBudget { bytes, budget },
        }
    }

    pub fn refused<T>(self) -> Option<PlanAdmission<T>> {
        match self {
            Self::Fits { .. } => None,
            Self::OverBudget { bytes, budget } => Some(PlanAdmission::OverBudget { bytes, budget }),
        }
    }
}

/// A value produced only if a plan passed the admission contract.
///
/// Simulator-backed reports often need to carry a measured value such as
/// simulated nanoseconds or transfer regret, but an over-budget plan must keep
/// the adjusted bytes that explain the refusal. This type keeps those two facts
/// on the same path as [`PlanFit`] instead of erasing the refusal into a bool.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanAdmission<T> {
    Fits { bytes: u64, value: T },
    OverBudget { bytes: u64, budget: u64 },
}

impl<T> PlanAdmission<T> {
    pub fn fits(&self) -> bool {
        matches!(self, Self::Fits { .. })
    }

    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Fits { value, .. } => Some(value),
            Self::OverBudget { .. } => None,
        }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> PlanAdmission<U> {
        match self {
            Self::Fits { bytes, value } => PlanAdmission::Fits {
                bytes,
                value: f(value),
            },
            Self::OverBudget { bytes, budget } => PlanAdmission::OverBudget { bytes, budget },
        }
    }
}

/// Whether a plan fits the admission contract it was entered with.
pub fn admissible_plan(
    workflow: &Workflow,
    decomposition: &Decomposition,
    constraints: &Constraints,
) -> Result<bool> {
    Ok(plan_fit(
        workflow,
        decomposition,
        constraints,
        constraints.expected_concurrency,
    )?
    .fits())
}

/// Whether a plan fits at `workers` concurrent block tasks.
pub fn plan_fit(
    workflow: &Workflow,
    decomposition: &Decomposition,
    constraints: &Constraints,
    workers: usize,
) -> Result<PlanFit> {
    let admission = Constraints {
        expected_concurrency: workers.max(1),
        ..constraints.clone()
    };
    let priced = phase_prices(workflow, decomposition, &admission, workers)?;
    let mut peak = 0;
    for (cost, _) in &priced {
        let bytes = admission.admission_demand_bytes(cost);
        peak = peak.max(bytes);
        if let Some(budget) = admission.budget_bytes {
            let budget = admission.admission_budget_bytes(budget);
            if bytes > budget {
                return Ok(PlanFit::OverBudget { bytes, budget });
            }
        }
    }
    Ok(PlanFit::Fits { bytes: peak })
}

/// **The planner's objective, applied to a plan the planner did not have to
/// produce**: the sum over phases of [`phase_price`]'s predicted makespan.
///
/// Everything here is the rule `Enumerating` searches under, and each one is a
/// decision worth naming rather than a detail:
///
/// * **the sum**, in phase order, because the search's DP is
///   `best[j] + price(j..i)` and that adds groups left to right. Phases are
///   charged as if they ran one after another, which `planner-gaps.md` records
///   as G2 — the `TaskGraph` makes them pipeline. The arena inherits the bias
///   deliberately: a re-priced plan has to be priced the way the planner prices
///   one, or the comparison is between two objectives rather than between an
///   objective and a simulation;
/// * **every phase at the element type it reads**, which is what the search
///   does since G3 and what `Decomposition::predicted_cost` always did. It is
///   read off the plan with `dtype_at` rather than folded again here: the plan
///   is where the fold's answer was recorded, and a second fold would be a
///   second opinion about it. Before G3 the search priced every phase at
///   `workflow.dtype` and this reproduced that, defect and all, because the
///   arena's job is to price what the planner prices;
/// * **materialised except the last**, which is what a phase boundary *is*;
/// * **`workers`** is the arena's, not the strategy's. A plan is chosen under
///   the concurrency its strategy was configured with and judged at the machine
///   the arena states — and those two being separately settable with nothing
///   reconciling them is item E of the same report. Passing one number to both
///   judges is this file's answer to it.
pub fn price_plan(
    workflow: &Workflow,
    decomposition: &Decomposition,
    constraints: &Constraints,
    workers: usize,
) -> Result<f64> {
    Ok(phase_prices(workflow, decomposition, constraints, workers)?
        .iter()
        .map(|(_, makespan)| makespan)
        .sum())
}

/// **The largest working set any phase of this plan demands**, in bytes, on the
/// same arithmetic `Constraints::affords_working_set` tests a candidate with:
/// one block's resident bytes times the concurrency.
///
/// **What the makespan cannot say.** A plan that is fast on the machine it was
/// planned for may not *fit* on another one at all, and a transfer sweep that
/// reported only durations would rank an impossible plan against feasible ones.
/// `tests/cost_scenarios.rs` uses this to mark those cells rather than time
/// them.
pub fn working_set_bytes(
    workflow: &Workflow,
    decomposition: &Decomposition,
    constraints: &Constraints,
    workers: usize,
) -> Result<f64> {
    Ok(phase_prices(workflow, decomposition, constraints, workers)?
        .iter()
        .map(|(cost, _)| cost.working_set_bytes_per_block * workers.max(1) as f64)
        .fold(0.0, f64::max))
}

/// Every phase's cost and predicted makespan, in phase order.
fn phase_prices(
    workflow: &Workflow,
    decomposition: &Decomposition,
    constraints: &Constraints,
    workers: usize,
) -> Result<Vec<(crate::decomposition::PhaseCost, f64)>> {
    let slots = workflow.chain.slots();
    let phases = decomposition.n_phases();
    let mut priced = Vec::with_capacity(phases);
    for (index, phase) in decomposition.phases.iter().enumerate() {
        for &slot in &phase.slots {
            if slot >= slots.len() {
                return Err(Error::InvalidArgument(format!(
                    "arena: phase {index} owns slot {slot} and the workflow's chain has {}. The \
                     plan and the workflow are not the same work, so pricing it against this \
                     chain would be pricing something else.",
                    slots.len()
                )));
            }
        }
        if phase.slots.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "arena: phase {index} owns no chain slot. The planner's objective is defined \
                 over a run of slots — one traversal per image read, writing the image after \
                 it — and a fragment or iterative phase is neither, so there is no price to \
                 put on it rather than one to invent."
            )));
        }
        let traffic = PhaseTraffic {
            images_read: images_read_by(&slots, &phase.slots, workflow.shape)?,
            // A run of chain slots is a pixel phase, and a pixel phase writes
            // the image after it.
            writes_an_image: true,
            repeats: 1,
            // What the run's own slots hold between them; see
            // `resident_buffers_of`.
            chain_buffers: resident_buffers_of(&slots, &phase.slots),
        };
        // The distinct traversal orders the run's ops prefer, which is what the
        // search's `GroupFold` accumulates and what `price_phase` charges a
        // conflict for.
        let mut orders: Vec<[usize; 3]> = Vec::new();
        for &slot in &phase.slots {
            for order in slots[slot].preferred_iterations() {
                if !orders.contains(&order) {
                    orders.push(order);
                }
            }
        }
        priced.push(phase_price(
            &slots,
            &phase.slots,
            &phase.grid,
            &phase.halo,
            decomposition.dtype_at(index).size_of() as f64,
            orders.len(),
            index + 1 < phases,
            traffic,
            constraints,
            workers,
        ));
    }
    Ok(priced)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A verdict with only the two numbers the arithmetic below reads. The rest
    /// is filled with what an empty plan would carry, so that a test of the
    /// ranking cannot accidentally depend on a plan.
    fn verdict(name: &str, priced_ns: f64, makespan_ns: u64) -> Verdict {
        Verdict {
            name: name.to_string(),
            shape: PlanShape {
                phases: vec![PhaseShape {
                    slots: Vec::new(),
                    block: [1, 1, 1],
                    blocks: 1,
                }],
            },
            admissible: true,
            priced_ns,
            outcome: Outcome {
                makespan_ns,
                ..Outcome::default()
            },
        }
    }

    fn judged(verdicts: Vec<Verdict>) -> Judgement {
        Judgement {
            verdicts,
            workers: 1,
        }
    }

    fn plan_with_block(block: [usize; 3]) -> Plan {
        Plan {
            decomposition: Decomposition {
                volume: [16, 16, 16],
                dtype: crate::Dtype::F64,
                phases: vec![crate::decomposition::PhaseDecomposition::derive(
                    vec![0],
                    vec!["only".to_string()],
                    [0, 0, 0],
                    [0, 0, 0],
                    BlockGrid::new([16, 16, 16], block).unwrap(),
                )],
                chain_reach: [0, 0, 0],
            },
            hints: crate::strategy::Hints::default(),
        }
    }

    #[test]
    fn plan_shape_distinguishes_anisotropic_blocks_with_the_same_first_edge() {
        let wide_y = PlanShape::from_plan(&plan_with_block([8, 16, 8]));
        let wide_z = PlanShape::from_plan(&plan_with_block([8, 8, 16]));

        assert_ne!(
            wide_y, wide_z,
            "candidate deduplication must not collapse plans that only agree on block()[0]"
        );
        assert_eq!(
            wide_y.first_axis_edges(),
            wide_z.first_axis_edges(),
            "this is the collision the old scalar signature could not see"
        );
    }

    #[test]
    fn plan_fit_reports_the_adjusted_admission_budget() {
        let workflow = Workflow::new(
            crate::op::Chain::op(crate::probes::IdentityOp::new("only", [0, 0, 0])),
            [16, 16, 16],
            crate::Dtype::F64,
        );
        let plan = plan_with_block([16, 16, 16]);
        let constraints = Constraints {
            budget_bytes: Some(10_000),
            cache_bytes: 9_000,
            expected_concurrency: 1,
            ..Constraints::default()
        };

        let fit = plan_fit(&workflow, &plan.decomposition, &constraints, 1).unwrap();

        let PlanFit::OverBudget { bytes, budget } = fit else {
            panic!("expected the cache reservation to make the plan over budget, got {fit:?}");
        };
        assert_eq!(budget, 1_000);
        assert!(bytes > budget);
    }

    #[test]
    fn plan_admission_carries_value_only_for_fitting_plans() {
        let admitted = PlanFit::Fits { bytes: 64 }.with_admitted_value(|| "simulated");
        assert_eq!(
            admitted,
            PlanAdmission::Fits {
                bytes: 64,
                value: "simulated"
            }
        );
        assert_eq!(admitted.value(), Some(&"simulated"));

        let refused: PlanAdmission<&'static str> = PlanFit::OverBudget {
            bytes: 200,
            budget: 100,
        }
        .with_admitted_value(|| panic!("refused plans must not evaluate a simulated value"));
        assert_eq!(
            refused,
            PlanAdmission::OverBudget {
                bytes: 200,
                budget: 100
            }
        );
        assert!(!refused.fits());
        assert_eq!(refused.value(), None);
    }

    /// The two rankings agreeing is tau `1.0` and regret `1.0`; the two exactly
    /// reversed is tau `-1.0` and a regret that is the whole spread.
    ///
    /// Both ends, because a correlation with one sign wired wrong reads as
    /// perfect agreement on the field the author happened to try.
    #[test]
    fn the_two_ends_of_the_rank_correlation() {
        let agreeing = judged(vec![
            verdict("a", 1.0, 10),
            verdict("b", 2.0, 20),
            verdict("c", 3.0, 30),
        ]);
        assert_eq!(agreeing.kendall_tau(), Some(1.0));
        assert_eq!(agreeing.regret(), Some(1.0));
        assert!(agreeing.discordant_pairs().is_empty());
        assert_eq!(agreeing.model_pick().map(|v| v.name.as_str()), Some("a"));
        assert_eq!(
            agreeing.simulated_pick().map(|v| v.name.as_str()),
            Some("a")
        );

        let reversed = judged(vec![
            verdict("a", 1.0, 30),
            verdict("b", 2.0, 20),
            verdict("c", 3.0, 10),
        ]);
        assert_eq!(reversed.kendall_tau(), Some(-1.0));
        // The model picks `a`, which the simulator makes three times the best.
        assert_eq!(reversed.regret(), Some(3.0));
        assert_eq!(
            reversed.discordant_pairs(),
            vec![("a", "b"), ("a", "c"), ("b", "c")]
        );
    }

    /// **A regret of one is not agreement.** The model picks the simulator's
    /// winner and orders everything below it wrongly, which is the case that
    /// makes `discordant_pairs` worth having beside the ratio.
    #[test]
    fn the_argmin_can_survive_an_ordering_that_does_not() {
        let field = judged(vec![
            verdict("winner", 1.0, 10),
            verdict("second", 2.0, 40),
            verdict("third", 3.0, 20),
        ]);
        assert_eq!(field.regret(), Some(1.0));
        assert_eq!(field.discordant_pairs(), vec![("second", "third")]);
        assert!(field.kendall_tau().unwrap() < 1.0);
    }

    /// Ties count as neither concordant nor discordant, and a field that is all
    /// ties has no correlation to report rather than a zero.
    #[test]
    fn a_field_of_ties_reports_no_correlation_rather_than_zero() {
        let field = judged(vec![verdict("a", 1.0, 10), verdict("b", 1.0, 99)]);
        assert_eq!(field.kendall_tau(), None);
        assert_eq!(
            field.regret(),
            Some(1.0),
            "the tie is broken by entry order"
        );

        let empty = judged(Vec::new());
        assert_eq!(empty.kendall_tau(), None);
        assert_eq!(empty.regret(), None);
        assert!(empty.model_pick().is_none());
    }

    /// **The two plans the price refuses**, which are the two ways a plan can
    /// fail to be the work the workflow describes.
    ///
    /// A phase with no slots is a fragment or iterative phase, which the
    /// planner's objective says nothing about; a phase naming a slot the chain
    /// does not have is a plan for some other chain. Both would otherwise
    /// produce a number — the first by charging a pixel phase's traffic to work
    /// that does none, the second by panicking on an index — and a number is
    /// the one thing an arena must not invent.
    #[test]
    fn the_price_refuses_a_plan_that_is_not_this_workflows_work() {
        use crate::decomposition::PhaseDecomposition;
        use crate::geometry::BlockGrid;
        use crate::op::Chain;
        use crate::probes::IdentityOp;
        use crate::reach::Reach;
        use crate::Dtype;

        let volume = [16usize, 16, 16];
        let workflow = Workflow::new(
            Chain::op(IdentityOp::new("only", [1, 1, 1])),
            volume,
            Dtype::F64,
        );
        let grid = BlockGrid::new(volume, [8, 8, 8]).expect("a lattice");
        let phase = |slots: Vec<usize>| {
            PhaseDecomposition::derive(
                slots,
                vec!["p".to_string()],
                Reach::symmetric([1, 1, 1]),
                Reach::symmetric([1, 1, 1]),
                grid.clone(),
            )
        };
        let plan = |slots: Vec<usize>| Decomposition {
            volume,
            dtype: Dtype::F64,
            phases: vec![phase(slots)],
            chain_reach: [1, 1, 1],
        };
        let constraints = Constraints::default();

        let err = price_plan(&workflow, &plan(Vec::new()), &constraints, 1)
            .expect_err("a phase with no slots has no price")
            .to_string();
        assert!(err.contains("owns no chain slot"), "{err}");

        let err = price_plan(&workflow, &plan(vec![0, 1]), &constraints, 1)
            .expect_err("a phase naming a slot the chain has not")
            .to_string();
        assert!(err.contains("slot 1"), "{err}");

        // and the plan that *is* this workflow's work prices.
        let priced = price_plan(&workflow, &plan(vec![0]), &constraints, 1).expect("a price");
        assert!(priced > 0.0);
    }

    /// A field whose best simulated makespan is zero has no ratio, and saying
    /// so is better than dividing by it.
    #[test]
    fn a_field_that_takes_no_time_has_no_regret() {
        let field = judged(vec![verdict("a", 1.0, 0), verdict("b", 2.0, 5)]);
        assert_eq!(field.regret(), None);
    }

    #[test]
    fn inadmissible_verdicts_are_reported_but_not_picked() {
        let mut impossible = verdict("too-large", 0.1, 1);
        impossible.admissible = false;
        let field = judged(vec![impossible, verdict("fits", 1.0, 10)]);

        assert_eq!(field.model_pick().map(|v| v.name.as_str()), Some("fits"));
        assert_eq!(
            field.simulated_pick().map(|v| v.name.as_str()),
            Some("fits")
        );
        assert!(
            field.table().contains("too-large"),
            "the rejected entrant should still be visible"
        );
        assert!(
            field.table().contains(" no "),
            "the table should expose why the faster-looking entrant was skipped"
        );
    }
}
