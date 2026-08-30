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
  invariant to this. **Measured, 2026-08-30, and it is not benign:** the
  divergence is worth 0.2% of makespan when nothing contends and **47%** when
  something does. `costs/two-nodes` is where it shows — the search's mixed grid
  `[48, 24, 48]` lets a phase's small blocks start while the previous phase's
  expensive ones are still running, which doubles the workers per node and slows
  them through contention. Switch the simulator's contention off and the penalty
  vanishes (1.520 against the uniform grid's 1.534).

  **`Machine::wave_synchronous` now models the executor's own dispatch**, and it
  cost nothing to add: the ready set already knew how to hold a task until an
  earlier phase finished, because a barrier phase does exactly that. Under it,
  on the machine and the plan where this was found:

  | contention | mixed / uniform, continuous | mixed / uniform, in waves |
  |---|---|---|
  | 0.00 | 0.991 | 0.996 |
  | 0.40 | **1.467** | **0.997** |

  The two models **rank the two plans oppositely**, and only once workers
  contend — at `contention: 0.0`, where every figure recorded before today was
  taken, they agree to within 1%. The simulator is the pessimistic one, which is
  the direction that matters: a planner tuned against it avoids a plan the
  executor would run well. So `costs/two-nodes`' 1.467 "regret" is this
  divergence and not a planner error.

  **What is still open is which model an arena should judge under.** The sweep
  runs on the continuous one, because that is what every recorded number used
  and `Machine::default` still says; flipping it would move all of them. It is
  no longer an unnoticed difference between two modules, which is what item C
  was about. `tests/wave_dispatch.rs` carries the measurement.
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

## Overfitting: what the planner does on machines this is not

Every figure above was taken on one machine, which is how a search gets tuned to
a host without anyone being able to tell. `costs/` is now a directory of
scenario files — the measured baseline and nine plausible neighbours of it
(slower disk, slower memory, less memory, two cores, forty cores, fine chunks, a
compressed store) — and `tests/cost_scenarios.rs` runs the planner against all of
them. Both judges are told the same coefficients: the planner through
`Snapshot::calibrate`, the simulator through `Rates::from_snapshot`.

**Two findings, and neither is the one that was expected.**

1. **The plan does not depend on the disk, the memory speed, the chunk shape or
   the compression.** All six of those scenarios choose the *same* plan — two
   phases at edge 48 — and transfer to each other at exactly `1.000`. What moves
   the search's answer is the **worker count** and the **budget**, and nothing
   else in the sweep. So the crate is not overfitted to this host's storage; it
   is close to blind to storage. Worth knowing before anyone calibrates a read
   coefficient expecting the plan to move.
2. **Where it is sensitive, it is wrong.** On `forty-cores` the planner's own
   choice is **1.230x** the best block edge available there, and the same defect
   shows in the transfer matrix from the other side: a plan chosen for *any
   other* machine runs **0.828x** — 17% faster — on forty cores than the plan
   chosen for it. Every other column is at or below 1.007.

And one operational finding: on `less-memory` every plan chosen elsewhere is
**inadmissible**, not merely slow. A transfer sweep that compared only durations
would have ranked an impossible plan against feasible ones, which is why the
harness checks the working set against the budget before it times anything.

The tables are recorded in the tests' own docs, with per-scenario ceilings, so a
change that degrades one machine while leaving this one alone fails with that
machine's name in the message.

## More than one computer

`simulate` modelled a worker pool: `workers` slots, one page cache, one set of
IO channels, one contention coefficient over all of them. That is a thread pool
on one machine, and it is not what `distributed` runs. `Machine::nodes` is the
field that says how many computers there are — `worker % nodes` is the topology
— and three of the model's quantities now stop at the boundary: **the page
cache**, **the IO channel** and **memory bandwidth**. At one node every
expression is the one it was, which `one_node_is_the_machine_this_crate_has_
always_simulated` asserts over both shipped schedulers.

`Outcome::duplicated_fetches` changed meaning in the same edit, and had to.
It counted any chunk fetched twice by anyone, so a pool going back for something
it had evicted was indistinguishable from a second machine's copy — on the
fixture below, a single node with a small cache reported **270** "duplicated"
fetches, which would have made two thirds of the two-node figure eviction rather
than duplication. It now counts a fetch by a pool that another pool had already
fetched, which is the sentence its own doc always claimed.

**What separate caches cost**, with the worker count held at eight and only
their distribution varied (`tests/multiple_computers.rs`):

| nodes | misses | duplicated | fetched MiB | hits |
|---|---|---|---|---|
| 1 | 462 | 0 | 14.4 | 2538 |
| 2 | 805 | 421 | 25.2 | 2195 |
| 4 | 1195 | 900 | 37.3 | 1805 |
| 8 | 1508 | 1290 | 47.1 | 1492 |

**3.26x the bytes off storage at eight computers for the same tasks.** Against
it, each node brings its own link: with the cache off, so that every arrangement
fetches the same chunks, the channel wait falls from 636 ms on one node to 96 ms
on eight.

**What it costs the planner.** `costs/` now carries `two-nodes`, `four-nodes`
and `ten-nodes` — the measured machine, two, four and ten times over — and the
transfer matrix says a plan chosen for **one** machine costs **1.86x** on ten.
That is the largest cell in the table and it is a term no `CostModel` can
express: the node count and the slot count are one axis to this planner, and
they differ in precisely the thing that matters, which is whether a fetch can be
shared. `Constraints` has no `nodes`, `price_phase` has no duplication term, and
the budget is per node only because `Scenario::constraints` divides the
concurrency by hand.

**Where spreading the computers apart pays, and where it inverts.** The obvious
proposal about a cluster — start the machines at different corners rather than
all at one — is right, and `HandoutPolicy::NearestFirst` (farthest-point
seeding, then nearest-unclaimed) is already the coordinator's default. What was
not known is the boundary. Against plan order, four computers over a `96^3`
volume in `16^3` chunks:

| threads per computer | 2 MiB cache | 8 MiB | 32 MiB |
|---|---|---|---|
| 1 | **1.239x** | 1.034 | 1.035 |
| 2 | **1.149x** | 1.016 | 1.024 |
| 4 | 1.071 | 1.017 | 1.017 |
| 10 | **0.875x** | 0.998 | 1.008 |

The ordering quantity is **cache per thread**. With room for a thread's working
set, separating the machines is pure saving — 1.24x and two thirds of the
traffic. With a fifth of a megabyte each it **inverts**: a machine's own threads
thrash their shared pool, and plan order, which marches every worker through
adjacent blocks together, keeps a tighter joint working set and wins despite
duplicating more across machines. Give them room and it comes back — and
simultaneously flattens the gain at one thread, because a pool that holds
everything leaves a policy nothing to save.

So the rule is **spread the computers apart, provided each computer's cache can
hold about one read extent per thread it runs** — not "spread the workers out".

A prediction that failed, recorded because giving it up was not obvious:
`Handout` seeded from every *other worker's* anchor, so with ten threads per
machine it separated threads that share a cache as eagerly as machines that do
not. Keying the seeds and the view anchor by node instead — `Decision::
node_anchors`, right on its own terms and kept — changed the inverted case by
nothing. The axis is the cache, not the seeding target.

**A policy that reads the pool instead of following a route.**
`HandoutPolicy::Coalescing` scores candidates by chunks rather than walking a
prescribed curve — a Hilbert or Morton order is the textbook answer and the
wrong shape here, because a route marches into the holes left by stolen blocks,
short-circuited blocks and neighbours that finish early. The score is two counts
and no weight: chunks this block needs that my node's pool does not hold, minus
those a task my node **already has in flight** is bringing in. Distance breaks
ties and only ties, because it has no cost units and cannot be added to a chunk
count without a constant.

That second term is the one no existing policy has, and it is what fixes the
inversion. Against plan order, four computers on `96^3` in `16^3` chunks:

| threads | cache | `nearest-first` | `coalescing` |
|---|---|---|---|
| 1 | 2 MiB | 1.239 | 1.277 |
| 2 | 2 MiB | 1.149 | 1.219 |
| 4 | 2 MiB | 1.071 | 1.171 |
| 10 | 2 MiB | **0.875** | **1.093** |
| 10 | 32 MiB | 1.008 | 1.022 |

Better in every cell, and the harmful case is gone. Nothing was prescribed to
make a node's threads converge: a block a neighbour is already fetching for
scores cheap, which is a fact the policy reads rather than a rule it follows.

**A repulsion term, built twice and rejected both times.** Charging a candidate
for standing near another computer is the obvious third term. Built once as a
tuned penalty and once *derived* — two regions growing at equal rate meet at the
perpendicular bisector, the chunks fetched twice are those within a halo of it,
so the expected duplication is the block's chunk count times a risk running 0 to
1 across a shell of width `2H`. The derivation is sound and dissolves the weight,
since the result is in chunks like everything else. **Both measured worse than
no term**, and worst in the cell the policy exists for: 1.093 without it, 1.003
tuned, 0.986 derived. The reason is double-counting — a block another computer
works near is a block whose chunks are not in my pool and not in my node's
in-flight set, so it is already expensive by the warmth term, which measures the
fact directly instead of inferring it from geometry. Repulsion belongs at
*seeding*, where there is no warmth to read yet, and the seeding rule already
does it.

**What it costs**: 122.8 microseconds a handout against `nearest-first`'s 8.1,
from the chunk-key walk over every ready candidate. Free on a real coordinator —
a block is tens to hundreds of milliseconds — and not free in this simulator,
where the scheduler's scan is already 98% of a large run. The bound, if it ever
matters, is to score a shortlist rather than the whole ready set, which is the
same fix the dispatch loop wants.

Not selectable by a caller: it is held at the same admission boundary
`cache-modelled` is, and for the same reason — it scores against a modelled
cache, and `cache::ChunkCache` has no non-test construction site.

**And placement starts paying.** `distributed::handout`'s policies could only
be ranked under `cache_shared: false` — one cache per *slot*, the pessimistic
reading and a machine nobody has. With a pool per computer they have something
real to save: on four nodes, `NearestFirst` duplicates **342** fetches against
`Naive`'s **900**, the same plan and the same tasks either way. On one node both
duplicate nothing, which is what says the saving is the boundary and not the
policy.

### The node-aware terms that exist now

`CostModel` gained two fields, both machine properties rather than coefficients
and both inert at their defaults:

* **`contention`**, the same Amdahl coefficient `simulate::Machine::contention`
  carries. `phase_makespan`'s pool term said forty workers were forty times one;
  the measured figure is 2.41. On `costs/forty-cores` the model priced the
  coarsest grid at **3.045** times its own argmin where the simulator put it at
  **1.018**, and chose a finer grid on the strength of workers that do not
  exist. With the term, that scenario's regret went **1.230 → 1.000**, and the
  per-rung prices now track the simulator within a few percent on every machine
  in `costs/`;
* **`nodes`**, which the contention term needs (workers contend with the ones on
  their own machine) and which divides the channel bound (each computer has its
  own link).

**What it did not fix**, and the sweep says so: the largest cell in the transfer
matrix is still a single-machine plan costing **1.83x** on ten computers, and
that is fetch duplication, not contention.

### A node-aware read term, declined on the evidence

The obvious next move was a cost term for the chunks two machines both fetch —
3.26x the bytes at eight nodes, and nothing prices it. **The measurement says
not to.** On `costs/ten-nodes` the model's per-rung prices already track the
simulator's:

| | edge 16 | edge 24 | edge 32 | edge 48 |
|---|---|---|---|---|
| priced | 1.261 | 1.000 | 1.106 | 1.821 |
| simulated | 1.184 | 1.000 | 1.134 | 1.827 |

— including the 1.82 penalty on the coarse grid, which is exactly where the
duplication bites, and the planner's regret there is **1.000**. A duplication
term could only perturb a ranking that is already right. The 1.83x cell in the
transfer matrix is not a planner error either: it is the cost of *reusing* a
plan across machines, which the planner does not do when it is asked to plan for
the machine it is on.

Declined for now, in the same spirit as the anisotropic blocks item above:
measured, and not warranted on the evidence. What would warrant it is a case
where the *ranking* on a cluster is wrong, and none of the thirteen committed
scenarios produces one.

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
