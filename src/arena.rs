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
//! # What this is not
//!
//! **Not a runtime prediction.** `simulate`'s own header says it ranks designs
//! and does not predict runtimes, and every limit listed there is inherited
//! whole: workers do not contend unless [`Machine::contention`] says so, and the
//! executor is wave-synchronous where the simulator dispatches continuously
//! (`planner-gaps.md`, item C). A regret figure is evidence that two rankings
//! differ, not a measurement of seconds anybody will wait.
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

use crate::decomposition::{images_read_by, Constraints, Decomposition, PhaseTraffic};
use crate::error::{Error, Result};
use crate::fragment::PhaseWork;
use crate::simulate::{simulate, ExecutorOrder, Machine, Outcome, PerPhase, Rates, Scheduler};
use crate::strategy::{phase_price, Plan, Strategy, Workflow};

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

/// What both judges said about one entrant.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub name: String,
    pub phases: usize,
    /// Blocks per phase, in phase order.
    pub blocks: Vec<usize>,
    /// The block extent each phase chose.
    pub edges: Vec<[usize; 3]>,
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
            .min_by(|a, b| a.priced_ns.total_cmp(&b.priced_ns))
    }

    /// The entrant with the lowest **simulated** makespan.
    pub fn simulated_pick(&self) -> Option<&Verdict> {
        self.verdicts
            .iter()
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
            "planner arena, {} workers\n{:<34} {:>7} {:>9} {:>10} {:>10} {:>12}\n",
            self.workers, "plan", "phases", "blocks", "priced", "simulated", "fetched MiB"
        );
        for verdict in &self.verdicts {
            let blocks: usize = verdict.blocks.iter().sum();
            out.push_str(&format!(
                "{:<34} {:>7} {:>9} {:>10.3} {:>10.3} {:>12.1}\n",
                verdict.name,
                verdict.phases,
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

/// The competition: a machine, a set of rates, and the plans entered so far.
///
/// The machine and the rates are held here and not per entrant on purpose. A
/// competition in which two plans are judged on two machines ranks nothing, and
/// making that unrepresentable is cheaper than checking for it.
pub struct Arena {
    pub machine: Machine,
    pub rates: Rates,
    entrants: Vec<Entrant>,
}

impl Arena {
    pub fn new(machine: Machine, rates: Rates) -> Self {
        Self {
            machine,
            rates,
            entrants: Vec::new(),
        }
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
            // Every phase a `Strategy` produces is a run of chain slots; see the
            // module header for why the arena holds no other kind.
            let work = vec![PhaseWork::Pixels; decomposition.n_phases()];
            let mut scheduler = make_scheduler();
            let outcome = simulate(
                decomposition,
                &work,
                &self.machine,
                &self.rates,
                &entrant.plan.hints.release_images,
                &entrant.plan.hints.keep_images,
                PerPhase::default(),
                scheduler.as_mut(),
            )?;
            verdicts.push(Verdict {
                name: entrant.name.clone(),
                phases: decomposition.n_phases(),
                blocks: decomposition
                    .phases
                    .iter()
                    .map(|phase| phase.grid.n_blocks())
                    .collect(),
                edges: decomposition
                    .phases
                    .iter()
                    .map(|phase| phase.grid.block())
                    .collect(),
                priced_ns,
                outcome,
            });
        }
        Ok(Judgement {
            verdicts,
            workers: self.machine.workers,
        })
    }
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
/// * **every phase at `workflow.dtype`**, which is what the search does and
///   which `planner-gaps.md` names as a defect (a chain that binarizes halfway
///   is priced as if the second half still moved eight bytes a voxel). Same
///   argument: reproduce it, do not quietly fix it here;
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
    let slots = workflow.chain.slots();
    let phases = decomposition.n_phases();
    let bytes = workflow.dtype.size_of() as f64;
    let mut total = 0.0;
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
        let (_, makespan) = phase_price(
            &slots,
            &phase.slots,
            &phase.grid,
            &phase.halo,
            bytes,
            orders.len(),
            index + 1 < phases,
            traffic,
            constraints,
            workers,
        );
        total += makespan;
    }
    Ok(total)
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
            phases: 1,
            blocks: vec![1],
            edges: vec![[1, 1, 1]],
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
}
