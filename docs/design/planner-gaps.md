# Planner and executor — what the optimiser does not yet consider

A scouting report, written before the work rather than after it. The long-term
goal it is written for: **improve the planner and the executor, and use
`simulate` as the test bed for all of it.**

> **Provenance and how far to trust it.** Produced on **2026-08-30** by a read of
> `decomposition.rs`, `strategy.rs`, `simulate.rs`, `graph.rs`, `budget.rs`,
> `prefetch.rs`, `cache.rs`, `slab.rs` and `distributed/`, against the design
> docs. Every `file:line` was read rather than inferred, but **line numbers drift
> and none of this has been re-checked since**; treat a citation as a pointer to
> a place, not a promise about a line. **Nothing here has been implemented or
> measured** — where a number appears it is quoted from a measurement the crate
> already recorded, and it says which. Claims that are the author's inference or
> guess are labelled as such in the text.

## The overarching finding

**Nothing has ever fed a `Strategy`-produced `Decomposition` into `simulate`.**
Nothing in `src/` calls `simulate`; only `tests/simulate_ranks.rs` and
`tests/simulator_against_the_executor.rs` do, and neither mentions
`Enumerating`, `Greedy`, `Materialising` or `Trivial` — every simulated plan is
hand-built with `PlanBuilder`. The `Scheduler` trait picks among *ready tasks*,
so it ranks **schedulers over one plan**; nothing ranks **plans**.

The crate has already decided that the way to settle a planner question is a
competition — `BlockLadder::ALL`, `SlabPolicy::ALL` and `PartitionSearch`'s three
variants all exist so that "a competition that enumerates cannot silently stop
covering a variant", and the refined-ladder decision was explicitly deferred to
one. **That harness does not exist**, and it gates every other item below.

## The search space as it actually is

`Strategy::decompose` takes no `Environment` and is contractually data-blind, so
a plan is a function of the chain, an `[usize; 3]`, one `Dtype` and
`Constraints`. `Enumerating` decides two things: where the phase cuts go (a DP
over contiguous prefixes, `O(n^2)` priced runs) and the block edge per phase (an
inner sweep over `block_candidates`).

Fixed **outside** the search: which axes may be cut (default `vec![2]`, z only);
block *shape* — the candidates are scalar edges; the worker count (and there are
two of them, unreconciled — G7); traversal order; element type and volume per
phase; and everything about storage.

And the shipped default of the flagship strategy is `Enumerating::concurrency =
1`, which its own doc calls **"the negative control"**: at one worker the
objective collapses to serial work, which is monotone in the edge, so the sweep
answers *"the largest candidate that fits"* every time. **The default
configuration cannot exercise the one degree of freedom the search has.**

## Not modelled at all

| | state |
|---|---|
| **op reordering / algebraic rewriting** | Forbidden by contract — `execute_phases` refuses a decomposition that reorders. Out of scope by decision, and the report agrees with the decision. |
| **algorithm selection (`Chain::Alternative`)** | Machinery exists (`choose_branches`, `choose_paths`) and **has no caller in `src/`**. See G6. |
| **precision / dtype selection** | Ops declare, nothing chooses — and the search prices every phase at `workflow.dtype`, so a chain that binarizes halfway is priced as if the second half still moved 8 bytes a voxel. `Materialising` fixes this for itself and names the defect. |
| **chunk shape, storage layout** | Derived downstream from the block by a fixed rule, never priced. The *simulator* already has chunk shape as a lever with a measured interior optimum. See G9. |
| **compression per intermediate** | A fixed table by dtype. The ratios are measured (2.09x raw uint16, 19.7x bool) and never reach the planner: one `materialise_cost_per_voxel` for every intermediate, whose own doc admits it "over-values fusing late stages and under-values fusing early ones". `ChunkCache::encoded_ratio()` measures the real thing at run time and nothing consumes it. |
| **anisotropic blocks** | Measured and deliberately declined: a finer *scalar* ladder was worth up to 2.7x, full per-axis freedom at most 1.4x more. But the same doc records an open, quantified defect: **52.4 GiB of fragment traffic at 482 slab blocks against 136.8 GiB at 512 cube blocks** — near-identical block counts, 2.6x apart — and `price_phase` has no term that grows with the block count. |
| **GPU placement** | Entirely absent from the plan, the price, the DAG and the handout. `placement.rs` scopes it out and lists the three missing declarations. |
| **overlapping phases** | See G2 — the worst of these, and not written down anywhere before this report. |
| **recompute vs materialise per block** | Per phase only. No boundary-block-materialises/interior-fuses mechanism. |
| **intra-block slab count** | A rule of thumb in no cost model, and `SlabCut::amplification()` — the exact closed form, whose doc calls it "the number a planner has to weigh" — has no caller outside its own tests. See G8. |

## Cost-model defects the crate has not written down

The ones it *has* written down are good and should be read first — `PhaseCost`'s
header names the class better than this report could: *"A term that is wrong by
an amount which varies with the candidate is not a conservative approximation. It
is a bias."* Five instances found and fixed, each with a measurement. Still open
and recorded there: `working_set_bytes_per_block` charging one image where a
phase reads three (`budget.rs` puts it at 1.00x–3.56x); one `read_cost_per_voxel`
for halo and core against a measured 0.38–4 spread and a 42x cold-vs-warm term;
`order_conflict_penalty` defaulting to zero and uncalibratable; `Term::ComputeOf`
"recorded and reported, not used".

New in this report:

* **A. One global compute scalar against a measured 57x per-phase spread.**
  `combine` 3.541, `smooth` 98.329, `skeletonize` 201.397 ns/voxel. The
  simulator consumes per-phase rates already; the planner has nowhere to put
  them. Because the objective is a roofline, a 57x error on one phase flips which
  side of the max binds — and `assemble.rs` measured exactly that: sweeping
  declared compute over `1e-3..1e3` walks the argmin from the coarsest candidate
  to the finest at every `workers > 1`.
* **B. `rounds()` assumes identical tasks; the executor is bulk-synchronous and
  the tasks are not.** The executor pops a wave, runs it, and **joins the whole
  wave** before the next, so the real cost is `sum over waves of max(task)`, not
  `ceil(n/P) x mean`. Edge blocks are clamped and cheaper; a short-circuited
  block costs ~0. A wave with one live block and 39 constant ones costs a full
  block time.
* **C. The simulator and the executor have different concurrency models, and
  neither states it.** `simulate` dispatches continuously as workers free; the
  executor is wave-synchronous. So the simulator **cannot see** the
  wave-straggler cost — which is precisely the cost a scheduling change would try
  to remove. The differential test compares order, counts, chunks and bytes, all
  invariant to this.
* **D. `admission_bytes` has no caller.** `budget.rs` derives a measured 3.6x
  admission margin over the plan-only working-set figure, with a test asserting
  it is the smallest tenth covering every measured shape — and
  `affords_working_set` compares the raw figure.
* **E. Two concurrency numbers nothing reconciles.** `Strategy::plan` copies
  `slab_policy` from the constraints into the hints and *not* the concurrency. A
  caller who sets `Enumerating { concurrency: 40 }` and leaves
  `expected_concurrency` at its default gets a plan priced for 40, budgeted for
  1, executed at 40. Both default to 1, so it is silent until someone raises one.
* **F. Prefetch depth means three different things** — threads in
  `prefetch::Prefetcher`, blocks-ahead-in-plan-rank in `Machine`, and a boolean
  in the executor, which declares a whole phase at once. The simulator's headline
  "depth 1 optimal, deep is a cliff" finding is therefore about a quantity no
  component actually has.
* **G. The cache knows the future and uses LRU.** `prefetch.rs`'s founding
  argument is "Here it is known. The block plan is enumerated up front", and
  `ChunkCache::evict_one` is pure recency. `Entry` records `tier`,
  `from_prefetch` and `used`; eviction consults none of them. With the reference
  string known, Belady's MIN is computable.
* **H. The distributed submission path plans as a single-worker run** —
  hardcoded `block_candidates: vec![8]`, `split_axes: vec![0]`,
  `Enumerating::default()`. Honestly read, it is a probe harness; it is also the
  only one.

## The gaps, prioritised

Full argument, proposed approach, expected payoff, validation plan and cost for
each is in the report this file summarises. In short:

- **G1 — no planner→simulator path.** The arena. Prerequisite for accepting any
  other change. ~1 week, including making `simulate`'s ready set incremental
  (it is an `O(T^2)` full scan per dispatch today, fine for a 4^3 fixture and not
  for an arena).
- **G2 — `phase_makespan` assumes phases are sequential; `TaskGraph` makes them
  pipeline.** Measure the bias first (cheap, falls out of G1). It also exposes
  two under-charges in the budget, and under-charging is the one direction this
  crate says it may not be wrong in.
- **G3 — per-phase compute rates, plus the dtype/volume prefix fold.** Tier-1
  wiring: coefficients measured, consumer exists, additivity already proven to
  survive, effect on the argmin already measured.
- **G4 — greedy dispatch instead of wave-synchronous.** Measure in the simulator
  before touching the executor; `PerPhase::constant_fraction` already exists to
  make the fixture.
- **G5 — Belady replacement, since the plan is known.** Measure the LRU–OPT gap
  first; the measurement is the deliverable.
- **G6 — resolve `Chain::Alternative`.** One line to call the shipped machinery;
  the real work is scoring a branch at its own reach rather than the folded max.
- **G7 — reconcile the two concurrency numbers; apply the admission margin.**
- **G8 — price the slab count.** Call `amplification()`.
- **G9 — storage as a plan decision.** Largest potential payoff, far too large to
  start before the arena can score it. "Data-blind" and "storage-blind" are
  different claims and the crate conflates them; a `StorageModel` in
  `Constraints` would be data-blind, deterministic and hashable.

## If we can do three things

1. **Build the planner arena (G1).**
2. **Per-phase compute rates plus the dtype/volume prefix fold (G3).** Highest
   evidence-to-effort ratio here, and it would be the first planner change ever
   adjudicated by the simulator.
3. **Settle the sequential-phases assumption (G2), and the two budget
   under-charges it exposes.**

Deliberately **not** in the top three: GPU placement (nothing to build on),
anisotropic blocks (measured, correctly declined), op reordering (forbidden by
contract, rightly), and storage-as-a-decision (too large before the arena exists).
