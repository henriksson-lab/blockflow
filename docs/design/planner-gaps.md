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

> **Update, 2026-08-30.** The first sentence of this section is no longer true:
> `src/arena.rs` calls `simulate`, and `Arena::enter` takes a `&dyn Strategy`.
> The rest of the paragraph still stands — the `Scheduler` trait still ranks
> schedulers over one plan — and the arena is the other axis rather than a
> replacement for it. See **G1** and **What the arena found first**.

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

- **G1 — no planner→simulator path. BUILT, 2026-08-30.** `src/arena.rs`, held
  by `tests/planner_arena.rs`. A `Strategy`, a `Workflow` and a `Constraints` go
  in; a plan comes out priced twice — by the planner's own objective and by
  `simulate` — with the disagreement between the two rankings as the finding.
  See **What the arena found first**, below. The ready set is now maintained
  rather than rescanned (2.2x at 98 304 tasks, and the old scan survives as a
  `debug_assertions` oracle that every simulating test in the suite runs), which
  exposed the **larger** `O(T^2)` underneath it — see **The next thing in the
  way**.
- **G2 — `phase_makespan` assumes phases are sequential; `TaskGraph` makes them
  pipeline. MEASURED, 2026-08-30, and it is smaller than this report thought.**
  Under `SchedulePriority::PhaseMajor` — the shipped default and the order
  `execute`'s heap pops — the phases of a two-phase plan are sequential to
  within 3%, so the sum-over-phases objective is a correct description of that
  dispatcher rather than a bias. Under `BlockMajor` they overlap at 1.6–1.9
  phase-spans to the makespan, which *is* the pipelining predicted — and it
  moves the makespan by **0.2%** and the peak bytes by 2–5%, because four
  saturated workers do the same work either way. On that evidence this belongs
  below G3, not above it. Caveat, stated in the test: one chain, one volume,
  four workers, no contention, a pool that never starves. Pipelining pays where
  a phase *cannot* fill the pool, and no such phase is in the fixture yet. The
  two budget under-charges are untouched and still open.
- **G3 — per-phase compute rates, plus the dtype prefix fold. DONE, 2026-08-30,
  and adjudicated by the arena.** `CostModel::compute_of` holds one
  dimensionless correction per op family; `Snapshot::calibrate` fills it from the
  `Term::ComputeOf` coefficients this crate has recorded for years under a doc
  saying they were "recorded and reported, **not used**". An empty table is the
  old model bit for bit, so nothing recorded under it moved.

  **The adjudication.** Two ops with equal *declared* cost, different reaches,
  and a snapshot saying their true rates are 64x apart; both judges told the same
  measurements — the planner through `compute_of`, the simulator through
  `PerPhase::ns_per_voxel`:

  | workers | plain plan | corrected plan | simulated |
  |---|---|---|---|
  | 1 | 1 phase, block 64 | 1 phase, block 64 | 1.00x |
  | 4 | 1 phase, block 32 | 2 phases, 64 and 32 | **1.28x** |
  | 40 | 1 phase, block 32 | 2 phases, 64 and 16 | **2.86x** |

  The uncorrected model believes the two ops cost the same, so it fuses them and
  gives both one grid; the corrected one sees that the dear op wants a fine grid
  and the cheap one a coarse grid, and cuts. Nothing moves at one worker, which
  is where every other measurement in this file also stops.
  `the_per_family_corrections_change_the_plan_and_the_simulator_prefers_it`.

  **The dtype half** is folded too: `Enumerating` walks `Chain::produces` along
  the slots and prices each run at the type it reads, where it used to hand
  `workflow.dtype` to every phase — an 8x error on every byte-derived term of a
  chain that binarizes half way. `Materialising` had already fixed it for itself
  while naming the defect, and `predicted_cost` had always read `dtype_at`; the
  search was the last place holding one number, and the two prices now agree on
  a binarising chain where they used to differ. A side effect worth knowing: the
  fold *is* a type check, so a chain that narrows into an op which cannot accept
  the narrower type is now refused when the plan is made rather than when a
  block reaches it.

  **The volume half is not done.** `Chain::output_shape` folds the same way, but
  a per-run volume means each phase's `BlockGrid` is built on its own extent,
  which is a change to what the search *produces* rather than to how it prices.
  Left, and named.
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

## What the arena found first

One chain, four plans differing only in the block edge, judged at one worker and
at four. `priced` is the planner's objective and `simulated` the simulator's
makespan, each as a ratio to the best in its own column:

| | 1 worker priced | 1 worker simulated | 4 workers priced | 4 workers simulated |
|---|---|---|---|---|
| edge 8 (1024 blocks) | 3.068 | 7.506 | 2.530 | 4.663 |
| edge 16 (128 blocks) | 1.760 | 3.641 | 1.457 | 2.285 |
| edge 32 (8–16 blocks) | 1.326 | 2.448 | **1.000** | **1.000** |
| edge 64 (1 block) | **1.000** | **1.000** | 2.660 | 2.479 |
| Kendall tau | 1.000 | | 0.667 | |

**At one worker the two judges order the field identically. At four they do
not** — the cost model prices the 1024-block plan *below* the single-block one
and the simulator makes it nearly twice as slow. Both still choose edge 32, so
the regret (the simulated cost of trusting the model) is 1.000: the model's
argmin survives and its ordering does not.

That is **item C above with a number on it**, reached from the other side: the
cost model's `rounds()` divides a whole per-block cost by the pool, the
simulator dispatches continuously, and the difference is invisible at the
shipped default concurrency of one — which is also the configuration
`Enumerating` calls its own negative control. It is not the chunk grid: the same
pair is discordant at chunk edges 8, 16, 32 and 64, and the tau at one worker is
1.000 at all four.

The figures are pinned in
`the_two_judges_agree_at_one_worker_and_part_at_four`, which says in its own
doc that a change to either judge is expected to move them and that the test is
where the new ones get written down.

## The next thing in the way

Making the ready set incremental moved `simulate` from 79.2 s to 36.3 s on a
98 304-task plan and left it quadratic, so the scan was the *smaller* of two
`O(T^2)` terms. The other is the **`Scheduler` interface itself**: `pick` is
handed `Decision::ready` as a slice and every scheduler in the crate walks all
of it, so a dispatch costs `O(ready)` however cheap the loop around it is.
Measured against an `O(1)` control that takes the first ready task:

| tasks | `ExecutorOrder` | first-ready | share inside the scheduler |
|---|---|---|---|
| 1 536 | 13.9 ms | 9.0 ms | 35 % |
| 12 288 | 424.4 ms | 35.6 ms | 92 % |
| 98 304 | 36 298.9 ms | 567.2 ms | 98 % |

At the scale an arena sweep works at, **98% of a simulation is the scheduler
scanning the ready set.** Fixing it is a change to the trait, not to the loop:
`ExecutorOrder` only ever wants the minimum of a fixed key and could be served
by a heap the loop maintains, but `CacheAware`, `WarmestFirst` and the handout
policies want the whole set by design — their key depends on cache state that
changes under them. A plausible shape is an optional `fn key(&self, task) ->
Option<[usize; 5]>` that a stateless scheduler implements and the loop uses to
keep a heap, falling back to the scan for the ones that cannot. Not attempted
here; measured, so that it can be.

## If we can do three things

1. **Build the planner arena (G1).**
2. ~~**Per-phase compute rates plus the dtype/volume prefix fold (G3)**~~ —
   **done**, and it was the first planner change ever adjudicated by the
   simulator: 2.86x at forty workers. See G3 above. What is left of the item is
   the *volume* half of the fold.
3. ~~**Settle the sequential-phases assumption (G2)**~~ — **done, and it
   settled small**: 0.2% of makespan, and only under a policy the default does
   not use. See G2 above. The two budget under-charges it was expected to expose
   are still open and are now the whole of what is left of that item. The
   vacancy is best filled by **the `Scheduler`'s `O(T^2)`** above, which now
   gates how large a field the arena can judge.

Deliberately **not** in the top three: GPU placement (nothing to build on),
anisotropic blocks (measured, correctly declined), op reordering (forbidden by
contract, rightly), and storage-as-a-decision (too large before the arena exists).
