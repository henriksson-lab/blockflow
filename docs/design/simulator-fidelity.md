# Simulator fidelity — a work plan

What `src/simulate.rs` would need in order to be *realistic*, organised by the
question that actually decides how each item is built: **where does the number
come from?**

*Companion files: [`cache-and-prefetch.md`](cache-and-prefetch.md),
[`executing-a-run.md`](executing-a-run.md),
[`dimensions-and-modules.md`](dimensions-and-modules.md).*

> **Where this came from, and how far to trust it.** A read of `src/simulate.rs`
> end to end on **2026-08-30**, against the modules it is a model *of* —
> `cache.rs`, `distributed/cache_model.rs`, `statistics.rs`, `log.rs`,
> `decomposition.rs`, `iterate.rs`, `fragment.rs`. Every item carries the
> `file:line` its claim rests on, and [What was checked](#what-was-checked) says
> which were read in the code and which are inference. **Nothing here has been
> implemented or measured.** The sizes are guesses; the acceptance criteria are
> the part worth arguing about, because they are what would make each item
> checkable rather than plausible.
>
> *Restructured the same day, after the first version's section split (executor
> fidelity against physics) turned out to answer the wrong question. The tiers
> below answer the one that decides the work: whether an item is wiring, a
> number, or a model.*

## The four tiers, and why they are the organising principle

The first thing to know about any item is not how wrong it is — it is **where
its number comes from**, because that decides whether you are wiring, passing a
scalar, or building a model. Four answers:

* **Tier 1 — already declared, data-blind.** The op or the plan states it and
  the simulator simply does not read it. Pure wiring, no new API.
* **Tier 2 — a function of the data, but invariant to every lever.** One
  measured number per phase, supplied the way `phase_ns_per_voxel` already is.
  **No stand-in op, and no trait method** — a trait method here would have to
  either lie or run the op.
* **Tier 3 — a function of the data *and* of the lever being compared.** No
  scalar transfers between arms of a comparison, so this needs a model of the
  **data** — not a second implementation of the op.
* **Machine** — nothing to do with the data. A property of the hardware or the
  storage, measured on the machine.

**The rule that puts an item in tier 2 rather than on a trait.** Op traits in
this crate carry only data-blind declarations. `IterativeOp` is the worked
example and it is deliberate: it *requires* `limit() -> SubstageLimit`
(`src/iterate.rs:275`), a runaway **bound**, and it has no method for the
substage **count**, because the count needs data. Declare the bound; measure the
actual. That pattern is the answer to most of what follows, and the sidecar
suggestion at the end is the same pattern applied again.

## Three ways not to drift, and the rule for the simulator

Every number in the tiers above reaches the simulator through one of exactly
three channels, and the crate already uses all three. Which one an item gets is
part of the item.

**1. Share the definition — drift *impossible*.** The thing is a pure function of
declared data, so there is one implementation and both callers call it.
`strategy::priority_key` is public *for the simulator*, and its doc
(`src/strategy.rs:2765`) states the principle:

> a simulator claiming to model the executor's dispatch order must not carry its
> own copy of that order. The alternative was a transcribed key and a test
> comparing the two, which is a drift detector where sharing the definition is a
> drift *impossibility*.

`Decomposition::images_dead_after` (`src/decomposition.rs:648`), called by the
executor at `src/strategy.rs:868`, is the same move.

**2. Check the declaration against the run — drift *detected, loudly*.** The
thing must be stated ahead of time but is observable afterwards, so something
compares the two on every run. `check_fragment_coverage`
(`src/fragment.rs:2074`) walks the store after a fragment phase and checks what
landed against what was declared — its own words at `src/fragment.rs:209`: "what
is checked is what landed rather than what the plan promised".
`IterativeOp::limit` is the same shape: a **bound** the runaway guard fires on,
so it cannot be silently wrong in the dangerous direction. So is
`check_output_shapes`, and `dependencies_cover_reads`.

> **This is the condition on every declaration this plan proposes adding.** A
> declaration nobody checks is exactly the drift it was meant to prevent, moved
> somewhere harder to see. If a quantity cannot be checked against a run, do not
> declare it — measure it, and say that is what you did.

**3. Measure and feed back — drift *repaired*.** The thing cannot be declared at
all. `statistics::Snapshot::calibrate` (`src/statistics.rs:1245`) refits from
real runs and `Provenance` (`:1065`) reports how much evidence is behind each
coefficient.

**The rule: the simulator is never a fourth channel.** Every number it uses
arrives by share, check, or measure — never as a constant typed into
`src/simulate.rs` or into a test. Item 13 is mechanism 2 applied to the
simulator itself; item 4 is mechanism 3. `TILE_PHASE_RATES`
(`tests/simulate_ranks.rs:84`) is today's violation of the rule, and item 4 is
its removal.

## How to read this

The numbering is stable and is the identifier — item 5 stays item 5 as the order
changes. Tick a box when the item's **acceptance** holds, not when the code is
written. Sizes: **S** ≈ an afternoon, **M** ≈ a day or two, **L** ≈ a week or
more, mostly because the answer is not known in advance.

## Order of work

1. **Wave 0 — instrumentation:** 13, 14. Until these exist there is no way to
   tell which of the rest matter, and 13 will find divergences this list does
   not contain.
2. **Wave 1 — tier 1, all wiring:** 17 **first** — it removes a live drift
   risk from the very walk item 1 then modifies — then 1, 5, 6, 7.
3. **Wave 2 — tier 2, the numbers:** 4, 15.
4. **Wave 3 — the cache, which is one story:** 3 (decide), then 2, 10.
5. **Wave 4 — machine terms:** 8, 9.
6. **Wave 5 — tier 3, the models:** 16, 11.
7. **Wave 6 — the distributed half:** 12.
8. **Found along the way:** 18, whenever the fragment write path is next opened;
   19, whenever `PhaseCost` is next opened.

---

## Tier 0 — method, before any modelling

- [x] **13. Differential-test the simulator against the real executor.** **M**

  **What.** Run one plan through `strategy::execute_observed` and through
  `simulate` with the matching scheduler, and assert on the quantities that are
  deterministic in both. No durations, so no wall clock and no flakiness.

  **Why now.** Nothing in `src/` calls `simulate` — only
  `tests/simulate_ranks.rs`. A model with no consumer drifts from the thing it
  models, and every tier-1 item below is an instance of that having happened.

  **What it can lean on, unchanged.** `ExecutionLog` implements `EventListener`
  (`src/listener.rs:85`); `execute_observed` takes `&[Arc<dyn EventListener>]`
  (`src/strategy.rs:415`); `ExecutionLog::visit_order(phase)`,
  `blocks_admitted()` and `check_coverage_and_order()` already exist
  (`src/log.rs:429`, `:417`, `:456`). The only new API is on the simulator side:
  `Outcome` records no order.

  **Note on scope.** Ordering is already drift-*impossible* through
  `priority_key`, so this test's value is in the **quantities** — bytes, chunks,
  tasks, short-circuits — not the order.

  **Acceptance.** On three plans (an all-pixel chain, one with a fragment phase,
  one with a shape change) the two agree on admission order, `tasks_run`, the
  set of chunks touched per phase, and bytes read. Disagreements are recorded
  here as new items rather than papered over.

- [x] **14. Sweep rate-space instead of adding noise.** **S**

  **What.** Run every ranking comparison over a grid of compute:IO ratios and
  report **the region in which the ranking holds**.

  **Why.** The acceptance bar is "a change known to be an improvement must rank
  as one" (`tests/simulate_ranks.rs:1-11`), asserted today at a single point —
  `TILE_PHASE_RATES` (`tests/simulate_ranks.rs:84`) and one `io_ns_per_byte` —
  while the header states the byte coefficient spans `0.31` to above `4` across
  the corpus. A ranking that holds only at one ratio inside an order-of-magnitude
  spread is not a ranking.

  **Acceptance.** Each comparison reports the fraction of the swept grid on
  which its conclusion holds, and the suite fails when a conclusion that used to
  hold everywhere stops holding somewhere. A partial one is *documented*, not
  deleted.

---

## Tier 1 — already declared; the simulator just does not read it

Pure wiring. No new public API in any of these.

- [x] **17. Share the image-freeable predicate instead of transcribing it.** **S**
      *(new; do this before item 1, which modifies the same walk)*

  **What.** The freeing decision has two halves. *Which images are dead after
  this phase* is **shared** — `Decomposition::images_dead_after`
  (`src/decomposition.rs:648`), called by the executor at
  `src/strategy.rs:868`. *Whether a dead image is freeable* —
  `(Internal || released) && !kept` — exists **three times**, and every site
  documents that it is a copy:

  | | |
  |---|---|
  | `src/strategy.rs:877` | the executor — the truth |
  | `src/decomposition.rs:1090` | in `peak_image_bytes_with`, labelled "**Word for word the executor's rule** at `strategy.rs`'s `images_dead_after` loop" |
  | `src/simulate.rs:767` | the simulator, "applies the executor's rule with them, **exactly as** `Decomposition::peak_image_bytes_with` does" |

  **Why now.** There is already a hairline crack. The shared
  `images_dead_after` handles an image nobody reads with
  `None => image > 0 && image - 1 == phase` (`src/decomposition.rs:652`); both
  transcriptions instead say `None => false` (`src/decomposition.rs:1098`,
  `src/simulate.rs:776`). Those are not the same rule. They agree today only
  because an image enters the live set when its writer starts, so the differing
  case is unreachable — **agreement contingent on an invariant stated somewhere
  else entirely**, which is the failure mode transcription produces. Item 1
  rewrites the simulator's residency walk, so this is the moment.

  **Change.** Extract `Decomposition::image_freeable(image, released, kept)` and
  have all three call it; ideally fold it into `images_dead_after` so the two
  halves travel together and a caller cannot take one without the other.

  **Acceptance.** No `Visibility::Internal` comparison outside the one function,
  asserted by a grep test in the manner of `tests/no_domain_vocabulary.rs`. The
  `None` arms are reconciled deliberately, with the chosen answer stated, rather
  than left agreeing by accident.

- [x] **1. Charge for writes.** **M**

  **What.** The loop charges read misses and compute (`src/simulate.rs:820-841`).
  Nothing charges for writing the output block back.

  **Why it matters.** `CostModel` (`src/decomposition.rs:1478`) carries **two**
  write terms and `statistics::Term` (`src/statistics.rs:271`) measures both:
  `Write` for the workflow output and `Materialise` for an intermediate, split
  because "an intermediate compresses differently from an output, so one number
  over-values fusing late stages" (`src/statistics.rs:278`). The executor
  measures writes, the planner prices writes, the simulator ignores them — so
  every decision whose payoff is *fewer writes* is invisible: fusion against
  materialisation, keep against release, block-major against phase-major.
  `strategy::Materialising` is a shipped strategy the simulator cannot price.

  **What it needs.** `BlockGeometry.core`/`.valid` are `Region`s with
  `.voxels()`; `decomposition.dtype_at(phase + 1)`; `writes_an_image` is already
  computed at `src/simulate.rs:711` and used only to decide allocation. Bytes
  written is arithmetic on values already in scope.

  **Acceptance.** A plan that materialises an intermediate ranks worse than the
  fused equivalent at equal compute, and the gap moves with the intermediate's
  size. `tasks_run` stays invariant across schedulers.

- [x] **5. Do not prefetch data that does not exist yet.** **S**

  **What.** The ahead-loop skips only tasks already *started* —
  `indegree[ahead.id] == usize::MAX` at `src/simulate.rs:880` — and never checks
  that the ahead task's dependencies are met.

  **Why it matters.** `TaskGraph::build` (`src/graph.rs:148`) lays tasks out
  phase-major, so at depth ≥ 2 the prefetcher warms chunks of an intermediate
  whose producing phase has not run. It pays the channel time, so the depth
  cliff stands for the right reason, but it banks hits it could not have had.

  **Change.** Gate on `indegree[ahead.id] == 0`.

  **Acceptance.** `prefetched_bytes` contains no chunk of an image whose
  producing phase is incomplete at issue time. The depth-one gain and the
  deep-prefetch cliff both survive.

- [x] **6. Price a phase that reads more than one image.** **M**

  **What.** `block_bytes[id] = read_voxels * bytes_per_voxel * 2`
  (`src/simulate.rs:838`) — "a phase reading three images is charged as if it
  read one", as its own comment says.

  **Why it matters.** It understates working set and fetched bytes for exactly
  the multi-input fused phases a scheduler would otherwise prefer. The same gap
  is recorded on `PhaseCost::working_set_bytes_per_block`
  (`src/decomposition.rs:2463`).

  **What it needs.** `PhaseDecomposition::source_images` is public
  (`src/decomposition.rs:121`), `Task::source_deps` carries one entry per source
  image (`src/graph.rs:71`), `PhaseTraffic::images_read` already counts them —
  and **`ChunkGrid::keys(image, region)` is already image-keyed**
  (`src/distributed/cache_model.rs:124`). The simulator only ever asks about one.

  **Acceptance.** A two-input phase reports strictly more `fetched_bytes` and a
  strictly larger `peak_bytes` than the one-input equivalent on the same volume.

- [x] **7. One chunk grid per image, not one per plan.** **S**

  **What.** `ChunkGrid::new(decomposition.volume, rates.chunk)`
  (`src/simulate.rs:677`) is built once from the *full* volume, while `bytes_of`
  correctly uses `volume_at(image)` (`:681`).

  **Why it matters.** Any plan with a resample or a pyramid level keys its
  chunks against the wrong grid. Chunk shape is per-array in Zarr, not global.

  **Change.** A `Vec<ChunkGrid>` held by the simulator, from `volume_at`.
  Extending `ChunkGrid` itself would reach into `placement`; holding a vec does
  not.

  **Acceptance.** On a plan with a downsampling phase, the chunk count for the
  downsampled image matches what `zarr_env` would create for that extent.

---

## Tier 2 — a function of the data, invariant to the lever

One measured number per phase, carried beside `phase_ns_per_voxel`. **No
placeholder ops, and nothing new on an op trait.**

- [x] **4. Close the calibration loop.** **M**

  **What.** `Snapshot::calibrate` (`src/statistics.rs:1245`) fits a `CostModel`
  from real runs, but nothing converts a `Snapshot` into `Rates` +
  `phase_ns_per_voxel`. The suite's rates are three numbers a person typed out
  of a run dated 2026-08-23 (`tests/simulate_ranks.rs:84`).

  **What it needs, and how little.** `Coefficient` is
  `{nanos_per_unit, runs, units, nanos}` (`src/statistics.rs:1035`).
  **`Term::ReadBytes` is already nanoseconds per byte read** — that *is*
  `io_ns_per_byte`, currently marked diagnostic-only. `Term::ComputeOf(String)`
  is per-family compute, marked "recorded and reported, **not used**". So this is
  a `Rates::from_snapshot` constructor over measurements that already exist. The
  only design work is mapping op family to phase when a phase's chain has
  several.

  **Acceptance.** `tests/simulate_ranks.rs` takes its rates from a recorded
  `Statistics` file rather than a literal, and a stale file fails loudly. Re-run
  14 on the measured rates and record whether any ranking moved.

- [x] **15. Price a fixpoint phase at its measured substage count.** **M**
      *(new; not in the first version of this plan)*

  **What.** An iterative phase is charged one pass — by the simulator
  (`src/simulate.rs:841`, one `phase_rates[phase]` per task) **and by the
  planner**.

  **Why it is tier 2 and not a placeholder op.** This is the case that looks
  like it needs a faithful stand-in and does not, and the crate has already
  measured why. `src/iterate.rs:314`:

  > **the substage count does not vary with the block edge.** Swept over
  > thirteen lattices including `[1, 1, 1]` — where every step of the
  > propagation is a halo exchange — four reaches and two data shapes including
  > a serpentine forcing a long geodesic, the count is the whole-volume count
  > every time.

  So `S` is constant across every block edge, ordering and worker count in a
  comparison — it cancels between arms exactly as `Rates`'s wrong coefficient
  does. And **every real run already reports it**: `Stats::substages: Vec<usize>`
  (`src/log.rs:657`), described at `:650` as "**the one figure of a run that is a
  function of the data**", with `Stats::substage_changes` (`:678`) carrying the
  per-substage decay shape beside it. A stand-in op that iterates faithfully
  would be an expensive way to recompute a number that is measured invariant,
  and it is the transcription-and-drift pattern `priority_key` rejects.

  **Why the planner is in scope.** `src/iterate.rs:296` says the planner prices
  **one** substage, that this was believed to change only the predicted duration,
  and that the belief is "measured false — it changes the shape too, through the
  block edge". The true shape is `S x (read + compute) + write`; the one-substage
  choice costs up to **1.125x** the right one (`src/iterate.rs:310`). One change
  fixes planner and simulator together, which is the argument against a
  simulator-only path.

  **Change, and what was actually done.** `PerPhase::substages` beside
  `PerPhase::ns_per_voxel`, sourced from `Stats::substages`; the count folds into
  the rate so that `S` substages cost exactly `S` times the rate, and the fetch
  and the store still happen once each. `SubstageLimit` stays what it is — a
  bound, for refusing a runaway — and did not become an estimate.
  
  **The planner half is item 19 and was not done here.** `PhaseCost` cannot take
  a count without first learning that a phase has repeating and non-repeating
  terms, which `phase_compute_per_voxel`'s own doc says in as many words. That is
  a change to what the planner *chooses* on and wants its own measurement, so it
  is split out rather than bolted on.

  **Acceptance, as met.** `a_substage_count_multiplies_the_compute_and_nothing_else`
  asserts the identity — two substages cost exactly what twice the rate costs, to
  the nanosecond — and that the fetch, the store and the byte counters do not
  move with it. The block-edge half of the original acceptance belongs to item
  19, which is where the planner learns the shape.

---

- [x] **19. Give `PhaseCost` a repeats-versus-once split.** **L**
      *(new; the planner half of item 15, separated because it is a change to
      what the planner **chooses** on)*

  **What.** Item 15 gave the simulator a measured substage count. The planner
  still prices an iterative phase at `S == 1`, and cannot do otherwise with the
  shape it has.

  **Why it is its own item.** `phase_compute_per_voxel`'s own doc
  (`src/decomposition.rs:2966`) says what is missing and why a count alone will
  not do it:

  > what is wanted is a statement of **which terms of a phase repeat and which
  > happen once**, which is a fact about `strategy::run_iterative_phase` rather
  > than about any builder.

  The shape is `S x (read + compute) + write`: pricing at `S == 1` over-weights
  the store against the rest by a residual that varies with the block edge, and
  the resulting choice was measured at up to **1.125x** the right one
  (`src/iterate.rs:310`). So this is not "multiply a number" — it is `PhaseCost`
  learning that a phase has repeating and non-repeating terms, which every
  strategy's pricing then has to respect.

  **Acceptance.** At `S > 1` the chosen block edge departs from the
  one-substage choice in the direction `tests/phase_pricing.rs` sweeps, and
  `tests/iterative_block_choice.rs` records which edge each count picks.

## Tier 3 — a function of the data *and* of the lever

Here no scalar transfers between arms, so a model is genuinely needed — a model
of the **data**, not a second implementation of the op. `synthetic::Scene` /
`SceneSpec` is the crate's existing vocabulary for structured volumes with known
properties, and is where such a model belongs.

- [x] **16. Model short-circuited blocks.** **M**
      *(new; missed in the first audit)*

  **What.** `BlockOp::constant_maps_to` (`src/op.rs:1009`) lets the executor skip
  a block whose input is uniform. `Event::BlockShortCircuited` (`src/log.rs:132`)
  and `Stats::tasks_short_circuited` (`src/log.rs:593`) record it. **The
  simulator models none of it** — it charges every task in full.

  **Why it is tier 3.** A finer cut produces more uniform blocks, so the skipped
  fraction is a function of the grid *and* the data. One measured figure does not
  transfer from one decomposition to another, which is exactly the lever a block
  ladder sweeps. So the simulator cannot see the thing `constant_maps_to` exists
  to buy, and it over-charges finer cuts systematically.

  **Change.** An occupancy model — the fraction of blocks that are constant, as
  a function of block edge — supplied per phase, and the executor's own skip rule
  applied to it. Not an op model.

  **Acceptance.** On a fixture with a known constant fraction, the simulated
  skip count matches a real run's `tasks_short_circuited` at two different block
  edges — two, because agreeing at one is what a scalar would also do.

- [x] **11. Sidecar traffic and the barrier gather.** **L**
      *(re-scoped: this is the fragment instance of tier 3, not a fixpoint
      problem)*

  **What.** Barriers are pure ordering in the simulator; a fragment phase costs
  no bytes.

  **Why it matters.** See [Sidecars — a design
  suggestion](#sidecars--a-design-suggestion) below, which is the substance of
  this item. In short: sidecars are fully **measured**, entirely **undeclared**,
  and entirely **unplanned** — `Decomposition::exact_read_voxels` says outright
  it "has never counted those" (`src/decomposition.rs:852`) and `Residency`
  (`src/decomposition.rs:2470`) has no term for them.

  **Acceptance.** A barrier's cost rises with the fragment payload; a zero-reach
  fragment phase still costs its writes; and the simulated gather peak matches a
  real run's `Stats::sidecar_bytes_written` for the same plan.

---

- [x] **18. ~~A fragment phase stores an image its work says it does not.~~
      Retracted: the differential test was calling the wrong entry point.** **S**
      *(raised by item 13, and then withdrawn by item 13 — which is the loop
      working)*

  **What was reported.** On a two-phase plan ending in a fragment phase, the
  executor appeared to store a full image for a phase whose
  `PhaseWork::writes_an_image()` is `false`, and the figure was pinned in
  `tests/simulator_against_the_executor.rs` as an open divergence between the
  executor and the planner's own declaration.

  **What it actually was.** The test called `strategy::execute`, and
  `execute` hands every phase `PhaseWork::Pixels` (`src/strategy.rs:423`). So the
  fragment op was never applied at all — `Stats::fragment_applications` was `0`
  and `sidecar_writes` was `0` — and the block was read and written as if the
  phase were a chain. The executor was doing exactly what it was asked. With
  `execute_phases` and the assembly's own `work()`, the two agree.

  **What is worth keeping.** Two things, both now written into the tests that
  found them. `execute` is a convenience for all-pixel plans and is silently
  wrong for any other — worth knowing before writing the next harness. And a
  block whose read extent is uniform **short-circuits before `apply`**, so a
  fixture built on constant data tests the short circuit and reports that
  nothing else happened; `writing_past_the_declared_sidecar_bound_is_refused`
  says so where the next person will hit it.

## Machine terms

Nothing to do with the data. Each needs a measurement of the hardware before it
can be parameterised, and each lands as a field on `Rates` (measured constants)
or `Machine` (planner levers) — the split those two structs already document.

- [x] **8. A CPU contention term.** **M**

  Compute is a fixed duration however many workers run (`src/simulate.rs:841`).
  The header already names the number (`src/simulate.rs:66`): the tile run
  measured realised concurrency of **`2.41x` against forty requested**, against
  the simulator's near-linear scaling — about a 16x error on the one axis a
  distributed scheduler is judged on. Not a NUMA model: one stated scaling law,
  `t(1) x f(running)`, fitted to that figure.

  **Acceptance.** At forty workers the simulated speed-up is within a stated
  factor of `2.41x`, and the header's "do not read this axis" becomes a bound.

- [x] **9. IO latency and IO parallelism.** **M**

  One serial channel at bytes x rate (`src/simulate.rs:827`), no per-request
  cost, no concurrency. On a filesystem or object store, latency dominates for
  small chunks and concurrency is how bandwidth is reached — so as modelled,
  small chunks are free and concurrency never helps. Add `io_latency_ns` per
  request and a channel count; `io_free_at` becomes one free-at time per channel.

  **Acceptance.** A chunk-size sweep has an interior optimum rather than
  improving monotonically toward zero, and the depth cliff moves with channel
  count in the direction the storage does.

- [x] **10. Charge decompression on the miss path.** **S**

  A fetched chunk costs transfer only; `zarr_env` reads gzip, and the decode is
  CPU work proportional to bytes, between the transfer and the compute. One
  `Rates` field.

  **Acceptance.** At the compression ratio where the corpus says deflate stops
  paying, the simulator agrees it stops paying.

- [x] **2. Two cache tiers, and a decode cost on an encoded hit.** **M**

  A hit costs nothing (`src/simulate.rs:823`). The real cache has two tiers
  (`src/cache.rs:160`) and `src/cache.rs:145` records an encoded hit at
  **962 us — ~100x a decoded hit**, still ~40x cheaper than storage. So a
  cache-size sweep is monotone by construction while the real curve has a knee:
  more capacity buys more *encoded* residency and hits get two orders of
  magnitude dearer. Reuse `cache::Tier` and `cache::ArrayPolicy` as the
  vocabulary rather than inventing one.

  **Open.** Extend `ModelledCache` (shared with `placement::entitled` and
  `HandoutPolicy::CacheModelled` — real blast radius) or give the simulator its
  own. Tier accounting does not violate `cache_model.rs`'s standing rule, which
  prohibits state fed by *worker reports*, not extra derived state.

  **Depends on 3.**

- [x] **3. Decide which cache is being modelled.** **L**

  `src/distributed/cache_model.rs:31` states the problem: the LRU "was written as
  a model of `cache::ChunkCache` — which has **no non-test construction site**,
  so no `Environment::read` is served from one. What can physically serve a
  re-read on a node is the page cache, sized by free RAM." So the simulator's
  central mechanism is parameterised by a cache nobody constructs, and
  `Machine::cache_bytes` is an axis that does not exist on the machine.

  **Change — mostly a decision; both paths are existing API.** Either wire
  `ChunkCache` into `Environment::read` (`ChunkCache::new(MemoryBudget, capacity)`
  and `CachingSource::attach` exist), or model the page cache using `budget.rs`'s
  free-RAM rule — in which case it is not a planner lever at all.

  **Acceptance.** A real run's re-read count is reproduced within a stated
  tolerance through the mechanism the model claims. Until then every
  cache-sized conclusion carries a caveat naming this item.

- [x] **12. The distributed dimension.** **L**

  `workers` are slots in one address space sharing one cache; `Machine::cache_bytes`
  calls that sharing "the optimistic reading". `distributed/` has per-worker
  caches, a handout policy, `placement::entitled` and shared-volume IO, and
  `HandoutPolicy::CacheModelled` (`src/distributed/handout.rs:95`) is refused at
  `select` partly because the cache model is not trusted — which is a question
  the simulator should be able to settle and cannot.

  **Reuse.** `handout::choose` is a free function (`src/distributed/handout.rs:229`)
  and `HandoutPolicy` a plain enum, so a simulator `Scheduler` can **call the
  real policy** rather than copy it — the `priority_key` precedent again.

  **Acceptance.** `nearest_first` ranks above naive pull by a duplicated-fetch
  margin of the same *sign* as `src/distributed/tests.rs:238` measures.

  **Depends on 3.**

---

## Still open, and named rather than implied

Item 11 built the **machinery** — `SidecarSize`, the bound checked at the write
site, the simulator's per-block charge and gather peak — and declared a size on
exactly one stream, `probes::BlockSummaryOp`'s, which is `PerBlock(48)` and
therefore tight enough for the check to bite. **The twelve shipped streams across
ten op modules are still `SidecarSize::Unstated`**, which is the honest state:
each needs a reading of its own encoder to say which variant it is, and guessing
would put a plausible number where a bound belongs.

- [x] **20. Declare a `SidecarSize` for each shipped stream.** **M**

  `ops/{adjacency,coordinates,detect,fill,label,regional,rows,tabulate,walk}.rs`
  and `points.rs`. The variants are already sized to the shapes that occur:
  `components`-family block flags and the label/plateau merges are `PerItem`;
  `coordinates` and `walk` look like `PerCoreVoxel`; `tabulate` and `rows` emit
  row tables and are `PerItem` with a ceiling that refuses nothing — which is
  the case the enum exists to make *visible* rather than to solve.

  **Acceptance, as met.** `every_shipped_fragment_stream_declares_a_size` in
  `tests/one_freeing_rule.rs` scans `src/` outside test modules and fails on any
  `FragmentOutput::new` without a `.sized(..)`. Every tight bound is enforced at
  the write site.

  **What it taught, which was not what the plan expected.** The enum shipped
  with a variant per shape and ended as **one general form** — a header plus
  terms over the core, the read extent, and the read extent's boundary — because
  `detect`'s report is two shapes at once and a variant per encoder was
  unsustainable. And **four of the bounds were wrong on the first reading**, each
  caught by the write-site guard against a real run rather than by review:

  | | declared | written | what was missed |
  |---|---|---|---|
  | `regional.faces` | 124 | 144 | a surface term is not a function of a volume |
  | `regional.faces` | 236 | 256 | the planes are over the **read** extent, not the core |
  | `moments` | 1 212 476 | 1 649 260 | the stream is faces **and** moments |
  | `partials` | 8 | 72 | three words a region, not one a block |

  That is the point of item 20 rather than an embarrassment in it: a declaration
  nobody checks is a second copy of a truth, and this one was wrong four times in
  an afternoon. The bound now comes from each encoder's own arithmetic, in one
  constructor per shape.

- [x] **21. Give `Residency` a `sidecar_bytes` term.** **M**

  `Outcome::sidecar_gather_peak` records the barrier gather, deliberately
  *beside* `peak_bytes` rather than inside it, because `Residency` still has no
  term for sidecars and a figure the byte budget does not know about must not
  silently move the number strategies are compared on. Closing that is a change
  to `Constraints::affords_working_set`, and wants item 20 first: a budget built
  on `Unstated` would be a budget built on zero.

## Sidecars — a design suggestion

Item 11's substance. Sidecars are not a corner case: **12 declaration sites
across 10 op modules** (`ops/{adjacency,coordinates,detect,fill,label,regional,
rows,tabulate,walk}.rs`, `points.rs`), and for a tabulating or row-emitting op
the payload can rival the image.

### The state today, in four columns

| | sidecars |
|---|---|
| **measured** | **completely.** `Event::SidecarWritten` / `SidecarRead` / `SidecarDiscarded` (`src/log.rs:162`, `:172`, `:185`); `Stats::sidecar_reads` / `sidecar_writes` / `sidecar_bytes_read` / `sidecar_bytes_written` (`src/log.rs:728-731`); `Sidecars::bytes()` and `fragments()` running totals. |
| **declared** | **not at all.** `FragmentOutput` is `{stream, lifecycle, coverage}` (`src/fragment.rs:246`). `Coverage` (`:207`) says *whether* every block writes, never how much. |
| **planned** | **not at all.** `Decomposition::exact_read_voxels` says it "has never counted those" (`src/decomposition.rs:852`); `Residency` (`:2470`) is `image_bytes + working_set_bytes` and has no third term, so `Constraints::affords_working_set` (`:2055`) cannot see a gather. |
| **simulated** | **not at all.** Barriers are ordering only. |

### Why the gap has teeth, and which way it points

At a barrier the gather holds **every contributing block's fragment at once**.
Under `Coverage::EveryBlock` that is `n_blocks x payload` resident at one
instant, with no budget term. And the payload's block-count scaling runs the
*wrong way*: a per-block header, or any fixed-size-per-block component, totals
**more** as the cut gets finer — while the two terms the budget does model,
`image_bytes` and `working_set_bytes`, both fall. So the planner is free to
choose a finer cut to shrink the working set and thereby grow an unbudgeted
peak, with nothing anywhere to notice.

### The suggestion: declare a bound, check it, measure the actual

The crate has solved this shape of problem before — `IterativeOp::limit` with
`Stats::substages`, `Coverage` with the guard that walks the store — and the
shape has three parts, not two: a **bound** that is always true, a **check**
comparing it against what the run did, and a **measurement** for the quantity
the bound is too loose to serve. The bound and the measurement go to different
consumers; the check is what stops the bound becoming a second copy of a truth.
Reusing a shape duplicates no code, which is the whole reason it is a shape
worth reusing.

**0. The declaration must be checked, or it must not exist.** This is the
condition from [Three ways not to drift](#three-ways-not-to-drift-and-the-rule-for-the-simulator),
and it is what makes the pattern safe rather than merely tidy — a size nobody
checks is a second copy of a truth, and it will disagree with the bytes the op
actually writes. The check is nearly free, because the walk already exists:
`check_fragment_coverage` (`src/fragment.rs:2074`) already iterates
`env.sidecar_keys(&output.stream)` for every declared output after the phase,
and never looks at a length. Comparing bytes against the declared bound there
costs one field, and an op that writes more than it declared then fails by name,
on the run that does it, naming the stream. **If a stream's size cannot be
checked this way, it should not be declared at all** — measure it instead, and
say so.

**1. `FragmentOutput` grows a size declaration**, required and with no default,
for the same reason `IterativeOp::limit()` has none: a silent zero is the defect
the requirement exists to prevent.

```rust
pub struct FragmentOutput {
    pub stream: String,
    pub lifecycle: Lifecycle,
    pub coverage: Coverage,
    pub size: SidecarSize,   // new, no default
}
```

**2. `SidecarSize` names the shapes that actually occur**, and makes "cannot be
declared" a variant rather than a plausible constant:

* `Fixed(u64)` — one object of known size whatever the block. `IterateReduce`'s
  final state.
* `PerBlock(u64)` — a header, a per-block scalar. **The variant that makes a
  finer cut cost more**, and the one the budget most needs.
* `PerCoreVoxel(f64)` — bounded by construction: a per-voxel flag list, a
  coordinate list.
* `PerItem { bytes: u64, at_most: ItemBound }` — labels, objects, rows.
  `ItemBound::CoreVoxels` is the honest ceiling and is enormously loose.

**3. Two consumers, two numbers, and they must not be confused.** The **declared
bound** is what a *refusal* rests on — the budget, `Constraints` — because a
bound is always true. The **measured figure**, from `Stats::sidecar_bytes_written`
through a new `Term::SidecarBytes`, is what a *choice* rests on — the cost model
and the simulator — because a choice only has to be right relative to another
choice. `Provenance::{Seeded, Unreproduced, Measured}` (`src/statistics.rs:1065`)
already expresses which of the two you are holding, and should be reported rather
than averaged away.

**4. `Residency` gains a third term**, `sidecar_bytes`: the gather at the worst
barrier, from the declaration, so `affords_working_set` becomes answerable for a
fragment phase.

**5. `exact_read_voxels` gets a sibling, not a change** — `exact_sidecar_bytes`.
Its own header defines it as the voxel figure to compare against a run's `read`,
and a run counts fragments separately; folding sidecar bytes in would break the
comparison it exists for.

**6. The simulator then needs no new concept**: charge sidecar writes per block
on the channel, and the gather as one read plus a residency spike at the barrier,
using the declaration for the data-blind part and `Term::SidecarBytes` for the
measured part.

### What this does not solve, and should say so

`ops/rows.rs` and `ops/tabulate.rs` emit row tables: bytes per object, with a
ceiling of one object per voxel. For those streams the bound refuses nothing
useful and the measured figure is the only usable number. The right outcome is
that they declare `PerItem` with the loose bound, carry a `Provenance` that says
"measured, or nothing", and that the looseness is **visible** — which is strictly
better than a plausible constant that no one can tell is wrong.

---

## What is not on this list

Deliberately, and the module header's reasoning is the reason:

* **Noise, jitter, distributions.** A distribution trades a defensible ranking
  tool for an indefensible predictor. The honest form of that concern is item
  14's sweep: *over what region does this ranking hold*, not *what is the
  variance*.
* **Rates that change over time**, thermal or otherwise.
* **Crashes, retries, stragglers, workers leaving.** The reissue machinery is
  tested in `tests/local_multi_node.rs`; simulating failure is a different tool.
* **Storage seek modelling, readahead heuristics.** Item 9's latency and
  parallelism is the part that changes rankings.
* **A parallel `SimOp` trait, or placeholder ops.** Tiers 1 and 2 need none, and
  tier 3 needs a model of the *data* rather than a second op. See the two rules
  at the top.
* **Const generics.** Every quantity here is data- or machine-determined at run
  time; a const generic is a compile-time constant. Encoding a substage count or
  a payload size in a type would bake a measurement into the type system in a
  crate whose discipline is to declare facts as data and check them.

---

## What was checked

Read in the code on 2026-08-30, and quoted above from the file named:

* No write accounting: `src/simulate.rs:820-841` charges `transfer` and
  `compute` only; the sole use of `writes_an_image` is allocation at `:811`.
* A cache hit costs no time: `src/simulate.rs:823`. Prefetch skips only started
  tasks: `:880`. `block_bytes` is two tiles of one image: `:838`. One
  `ChunkGrid` from the full volume: `:677`, against `bytes_of`'s `volume_at` at
  `:681`. `2.41x` against forty: `:66`.
* `ChunkGrid::keys` takes an **image**: `src/distributed/cache_model.rs:124`.
  `ModelledCache`'s standing rule: `:31`.
* Two cache tiers and the `962 us` encoded hit: `src/cache.rs:160`, `:145`.
* `Term::Write` / `Term::Materialise` and why they are separate:
  `src/statistics.rs:271-282`. `Coefficient`: `:1035`. `Provenance`: `:1065`.
  `families()` "recorded and reported, not used": `:1174`. `calibrate`: `:1245`.
  `CostModel`: `src/decomposition.rs:1478`.
* `IterativeOp::limit` is required and is a bound: `src/iterate.rs:275`. The
  planner prices one substage: `:296`. The `1.125x`: `:310`. The
  block-edge-invariance sweep: `:314`.
* `Stats::substages` and "the one figure of a run that is a function of the
  data": `src/log.rs:657`, `:650`. `substage_changes`: `:678`.
* `BlockOp::constant_maps_to`: `src/op.rs:1009`. `Event::BlockShortCircuited`:
  `src/log.rs:132`. `Stats::tasks_short_circuited`: `:593`.
* Sidecar events: `src/log.rs:162`, `:172`, `:185`. Sidecar counters: `:728-731`.
  `FragmentOutput`: `src/fragment.rs:246`. `Coverage`: `:207`. `BlockOutput`:
  `:264`. "has never counted those": `src/decomposition.rs:852`. `Residency`:
  `:2470`. `affords_working_set`: `:2055`.
* `ExecutionLog` is an `EventListener`: `src/listener.rs:85`. `execute_observed`
  takes listeners: `src/strategy.rs:415`. `priority_key` is public for the
  simulator: `src/strategy.rs:2765`.
* The coverage guard walks the store after the phase and already enumerates
  every fragment key without reading a length: `src/fragment.rs:2074`, and the
  claim it enforces is stated at `:209`.
* The freeable predicate exists three times — `src/strategy.rs:877`,
  `src/decomposition.rs:1090`, `src/simulate.rs:767` — and the middle one calls
  itself "word for word the executor's rule". The half that *is* shared is
  `Decomposition::images_dead_after`: `src/decomposition.rs:648`, called at
  `src/strategy.rs:868`. The differing `None` arms: `src/decomposition.rs:652`
  against `:1098` and `src/simulate.rs:776`.
* `handout::choose` is a free function: `src/distributed/handout.rs:229`;
  `HandoutPolicy`: `:95`.
* `PhaseDecomposition::source_images` is public: `src/decomposition.rs:121`.
  `Task::source_deps`: `src/graph.rs:71`.
* Nothing in `src/` calls `simulate`: only `tests/simulate_ranks.rs`.

**Inference, not checked**

* That item 1 would change the ranking of `Materialising` against a fused plan.
  It follows from the cost model having a write term and the simulator not
  having one; no run has been made either way.
* That item 5 inflates the measured prefetch benefit. The mechanism is in the
  code; the magnitude is unmeasured and may be small at depth 1, which is where
  the existing test asserts a gain.
* **The whole of the sidecar section's "teeth" argument** — that a finer cut
  grows the gather peak — is arithmetic on the declaration shapes, not a
  measurement. `Stats::sidecar_bytes_written` at two block edges would settle
  it, and that measurement should be the first thing item 11 does.
* That item 16's skipped fraction actually moves enough to change a plan choice.
  It is data-dependent by construction; whether it matters on real volumes is
  unknown.
* That the three copies of the freeable predicate **agree today**. The `None`
  arms differ; the argument that the difference is unreachable rests on an image
  entering the live set only when its writer starts, which was read in all three
  walks but not tested. Item 17 should begin by trying to construct the case
  that separates them — if one exists, it is a bug and not a tidy-up.
* Every **size** estimate.
