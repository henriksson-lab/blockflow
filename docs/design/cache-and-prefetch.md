# Wiring the cache and the prefetcher

*A specification, not an implementation. Nothing in `src/` was changed to write
it. Everything in §0 was established by enumerating call sites and by two
measurements taken from a prebuilt binary; everything after §0 is a design, and
where a design decision has no measurement behind it this note says so instead of
choosing.*

*Read §0 before any recommendation below. The whole note turns on the fact that
`ChunkCache` and `Prefetcher` are built, documented, tested — and on no path.*

---

## 0. What is true today

### 0.1 Neither type is constructed outside a `#[cfg(test)]` module

Enumerated, not inferred: every `ChunkCache::new` and every `Prefetcher::new` in
the crate is in `src/cache_tests.rs`, which `lib.rs` declares `#[cfg(test)]`.
**No `Environment::read` is served from a cache, and nothing anywhere reads
ahead.** Two consequences have been measured rather than argued:

* **`warm/cold` under `Gzip(1)` — the crate's default for every integer dtype —
  is `0.78`–`1.00`.** A second pass over data entirely in the page cache is
  barely faster than the first, because the inflate is re-paid and there is no
  in-process chunk cache to skip it. Uncompressed, where there is nothing to
  re-decode, the page cache is worth about `2x` (`0.41`–`0.61`) — the only place
  in that table where "a halo is warm and therefore cheap" holds.
  `tests/halo_io_cost.rs`.
* **Cold is `42x` warm at one block** on a flat file (12.13 s against 0.289 s).
  `docs/design/intra-block.md` §8. Whether the data is resident matters an order
  of magnitude more than how much halo is re-read — and residency is the one
  thing nothing in this crate currently manages.

### 0.2 The warning this stage is written around

The coordinator has a **`ModelledCache`**: an LRU replayed from assignments,
which until this pass described itself as modelling *"this crate's own chunk
cache"* — a cache that does not exist. It drives two consumers, and measuring
them produced the rule this note exists to obey.

`distributed::tests::the_two_policies_are_indistinguishable_until_the_model_evicts`,
32 tasks, 4 workers, 64 distinct chunks, counted and not timed:

| arm | modelled capacity | duplicated fetches | redundancy |
|---|--:|--:|--:|
| naive pull | — | 62 | 1.969 |
| **nearest-first** | — | **6** | **1.094** |
| cache-modelled | 1 chunk | **22** | **1.344** |
| cache-modelled | 2 chunks | **22** | **1.344** |
| cache-modelled | 4 chunks | 6 | 1.094 |
| cache-modelled | 512 chunks | 6 | 1.094 |

`HandoutPolicy::CacheModelled` claimed that *"with an empty or useless model this
degrades exactly to `NearestFirst`"*. The **empty** half holds: an empty model
misses every key of every candidate, the primary sort term is constant, and
distance decides. The **useless** half is false, because `handout::nearest` sorts
on `(misses, distance, task)` and a thrashing model returns a miss count that
**varies without carrying information**, so it dominates the term that was
measured rather than deferring to it:

> **A useless model degrades toward noise, not toward the policy it claims to
> fall back on. A cache term consulted before it is calibrated is worse than
> none, because it outranks a term that was measured.**

That policy is now refused at `HandoutPolicy::select`. **Every recommendation
below is shaped by it**: a cache quantity may enter a decision as a *constraint*
or as a *tie-break under a measured term*, and may not enter as a ranking key
above one, until it has a hit rate somebody has counted.

### 0.3 The half that is already right

`MemoryBudget` in `src/budget.rs` already integrates a cache correctly **at run
time**: `Class::Opportunistic` is granted only from slack, never blocks, and is
refused outright while any `Class::Reserved` request is queueing, so a cache
cannot starve compute. `ChunkCache` takes one opportunistic lease per entry and
degrades to a pass-through when refused; `CacheStats::refusals` counts it.

**Nothing in this note proposes changing that.** The gap is entirely at *plan*
time — §2.

---

## 1. Where a real cache sits

### 1.1 The seam the cache was built for does not meet the executor

`CachingSource::attach` is the read-through seam, and it is written against
`RegionSource<T>`. **No `Environment` reads through `RegionSource`.** Traced
through all four:

| environment | how `read` gets its bytes | reaches `RegionSource`? |
|---|---|---|
| `ZarrEnvironment` | `array.retrieve_array_subset(&subset)` — `zarrs`, directly | **no** |
| `distributed::SharedVolumes` | `read_exact_at` on an `f64` file, one row run at a time | **no** |
| `ArrayEnvironment` | a slice copy out of a resident `Array3` | **no** |
| `AccountingEnvironment` | fabricates voxels from a region | **no** |

The `RegionSource` implementors are `synthetic::{IntensitySource, LabelSource}`,
`npy::NpySource`, `region::ArrayRegionSource` and `observed_io::ObservedSource` —
**none of which is an environment.** The application this crate was extracted
from recorded the same fact from the other side — *"`blockflow` has
`RegionSource`/`RegionSink` traits that are the right shape and no `Environment`
built over them"* — in a tile-scale test outside this repository, named here only
so the provenance is followable and cited by heading rather than by line, as
`docs/design/intra-block.md` §7 does for the same reason.

**So the first item of work is not "turn the cache on". It is that the cache's
designed seam and the executor's read path have never met.**

### 1.2 Two placements, and the difference is exactly what the budget can see

| | **below `Voxels`** — per array, typed | **above `Voxels`** — per `Environment::read`, erased |
|---|---|---|
| key | `(ArrayId, chunk lattice index)` | `(image, Region)` |
| what it caches | decoded or encoded **chunks** | whole **block buffers** |
| element type | `T: CacheElement`, static | `Dtype` tag on `BlockBuf` |
| two overlapping reads | hit the same entries, by construction | **different keys over the same data** |
| a halo re-read | serves the neighbour's chunks | serves nothing — the extents differ |
| bytes it holds | `resident_bytes`, a counter | one buffer per distinct extent asked for |

**Below `Voxels` is the only placement that answers the question the stage
exists for.** A cache above it is keyed by the extent a caller asked for, and
`src/cache.rs`'s own header already dismisses that shape by name: *"two callers
wanting overlapping boxes produce different keys over the same data, entries
duplicate, hit rate collapses, and a request for one chunk cannot be served from
a cached span that contains it."* A halo re-read is precisely two overlapping
boxes. A block-buffer cache would hit **only** when the identical extent were
asked for twice, which a block plan never does.

**And it changes what the budget can see, which is the load-bearing half.** A
chunk cache's residency is `resident_bytes` — one number, bounded by
`capacity_bytes`, reported by `CacheStats`, and leased from `MemoryBudget`. A
block-buffer cache's residency is a function of which extents happened to be
asked for, which is a property of the traversal and not of any setting, so **its
size would not be a number the planner could subtract.** §2 needs a number.

**Recommendation: the cache sits below `Voxels`, at the per-array chunk lattice,
exactly where `ChunkCache` already is.** What must be built is the adapter from
each environment's storage call to `RegionSource<T>`, so that
`CachingSource::attach` has something to attach to.

### 1.3 What that costs, stated because it is not free

* **`ZarrEnvironment` would fetch through the cache's chunk lattice rather than
  through `zarrs`'s subset retrieval.** The two must be given the *same* lattice
  or the cache decodes units the store does not store. `register` already takes
  `chunk` as a parameter for this reason, and `ZarrEnvironment` already knows the
  array's own chunk shape (`chunk_shape()`), so the value exists; it is the
  wiring that does not.
* **`SharedVolumes` reads rows, not chunks.** It is a flat `f64` file with a
  row-run `pread`, so giving it a chunk lattice is inventing one. That is
  legitimate — `env.rs` already calls `ArrayEnvironment`'s chunk grid *"an
  accounting fiction"* — but the fiction must be the same one `chunks_touched`
  already charges against, or the counters and the cache disagree about what a
  chunk is.
* **`ArrayEnvironment` should not be cached at all**, and this is a refusal
  rather than an omission. Its `read` is a slice copy out of a volume that is
  *already entirely resident*; a cache over it would hold a second copy of bytes
  the process has, which is the opposite of the stage's purpose. `tests/halo_cost.rs`
  measured what warmth is worth there — a warm source costs `0.309x` a cold one
  at `32³` regions and `0.687x` at `64³`, bounded below by the allocation and the
  destination write — and that discount is the **CPU** cache's, which no software
  cache can add to.
* **`AccountingEnvironment` must stay uncached**, for a sharper reason: it
  fabricates. A cache in front of it would report hits against data that was
  never read, and the distributed locality harness (§0.2) would start measuring
  the cache instead of the handout.

### 1.4 The wire

`WorkflowSpec::cache_bytes` **already crosses the wire** and is already
documented as *"the per-worker cache budget the coordinator models — not an
instruction to the worker"*. Once a worker has a real cache, that sentence should
become false deliberately rather than by accident: the coordinator's model is
worthless if the two differ, and the field exists precisely so they need not be
guessed separately.

**The precedent for the worker side is `WorkerOptions::threads`** — *"how many
cores a node has and how many worker processes share them are facts about the
node, not the job"*. A node's memory is the same kind of fact. So the honest
shape is: **`cache_bytes` in the job states what the coordinator's model
assumes; a `WorkerOptions::cache_bytes` states what the node will actually give,
defaulting to the job's figure**, and the worker reports which it used so a
disagreement is visible rather than silent.

---

## 2. What the budget subtracts, and when

### 2.1 `admission_bytes` cannot carry it, and the reason is dimensional

`admission_bytes(figure: FrameworkFigure) -> u64` charges **one block**. Its two
margins are fitted per-block quantities — `UNOBSERVED_SHAPE_MARGIN = 3.6` over
the widest measured chain shape, `UNOBSERVED_OP_MARGIN = 2.1` over the widest
measured op. The comparison that uses it is, at both sites in `strategy.rs`:

```text
cost.working_set_bytes_per_block * expected_concurrency  <=  budget_bytes
```

A cache is **one reservation for the whole run**, not a per-block cost. Folding
it into the left-hand side would multiply it by `expected_concurrency`, charging
the cache once per concurrent block. **So the subtraction belongs on the
right-hand side and nowhere else.**

### 2.2 Two spellings, and the second is the one to build

**(a) The caller subtracts before setting `Constraints::budget_bytes`.** Zero
code, zero enforcement — and it is what happens today, in the sense that nobody
subtracts anything and no type says they should. A caller who forgets admits a
plan against memory that does not exist, with no diagnostic.

**(b) `Constraints` carries the reservation and the two comparison sites
subtract it.** The right-hand side becomes

```text
budget_bytes  -  cache_bytes  -  prefetch_depth * chunk_bytes
```

**Build (b).** Three arguments, in increasing force:

1. **It is hashable and it belongs in the fingerprint.** `Constraints` is what
   `decompose` is a deterministic function of. A budget that the caller
   pre-subtracted produces the same plan as one that did not, from a
   `Constraints` that cannot tell the two apart — so two runs at different cache
   sizes would carry the same `decomposition_fingerprint`. That is the exact
   defect `barriers.md` §10.5 records for `reduce`: *"a fingerprint is not
   evidence about which of them ran."*
2. **The prefetcher spends the same budget.** `depth × chunk` is held against a
   future that may not arrive, and §2.3's rule applies to it too: memory the
   plan-time budget cannot see is memory the run-time budget will refuse.
3. **It is wrong in the cheap direction.** Subtracting a reservation shrinks the
   admissible block, and a smaller block is the conservative outcome —
   `budget.rs` already argues exactly this when rejecting a separate refusal for
   fine cuts: *"admission takes the largest block that fits, and residency grows
   with block size, so the conservative move is a smaller block."*

### 2.3 The two budgets must be told the same number

There are **two** budgets and they are not the same object:

| | what it is | who checks it | knows about the cache? |
|---|---|---|---|
| `Constraints::budget_bytes` | a plan-time feasibility filter | `strategy`, at two sites | **no** |
| `budget::MemoryBudget` | a run-time lease, two classes | every acquire | **yes**, already (§0.3) |

A plan admitted against the first will be executed against the second. If the
first does not subtract what the second has already leased to a cache, the plan
is admissible and the run is not. **The whole of §2's work is making the
plan-time budget agree with a run-time budget that is already correct.**

---

## 3. How the planner learns the size, and on what principle it is chosen

### 3.1 How it arrives: `Hints::slab_policy` is the precedent, exactly

`SlabPolicy` took the route this should take, and its own record says why:
`Constraints::slab_policy` is the caller's statement, `Strategy::plan` copies it
into `Hints::slab_policy` — *"the one line that carries `Constraints::slab_policy`
into the run"* — and the executor reads it where it reads every other performance
decision. **A caller states it once, at the constraint.**

So: `Constraints::cache_bytes` and `Constraints::prefetch_depth`, copied into
`Hints`. `Hints::prefetch_depth` **already exists** — *"Reserved for
`MULTISLAB_IO.md` §4's hint-driven prefetcher. Recorded so a strategy can express
it before there is a prefetcher to consume it."* It is set in seven places and
**read in none.** Half of §4's plumbing is therefore already built and inert, in
the same way and for the same reason as everything else in this note.

### 3.2 The principle: derived, not chosen — and how far that reaches

The outside plan's *robust, not optimal* heading forbids a constant fitted on
this box — *"a mechanism needing a fitted constant per machine is not robust"*.
The precedent for obeying it is in `budget.rs`, and it is a good one: `UNOBSERVED_SHAPE_MARGIN` is **the smallest
tenth that covers every shape measured**, asserted in both directions, with the
tenth itself load-bearing because the widest op measured `2.0002x` and a whole
number would have bought fifty per cent of headroom with two ten-thousandths of
evidence.

Applying that rule here gives **a floor, a ceiling, and no optimum**, and the
three have different standing.

**The floor is derivable today, from the plan alone.** A cache that cannot hold
the halo a neighbouring block is about to re-read buys nothing at all — and §0.2
measured what a cache below that floor does: **at one task's read set it is not
merely useless, it is `3.7x` worse than ignoring it.** The plan holds every term:

```text
frontier_bytes = (mean_read_voxels - mean_core_voxels)   # halo voxels per block
               * bytes_per_voxel
               * live_frontier(visit_order, block_grid)  # blocks whose halo a
                                                         # later block re-reads
```

`mean_read_voxels` and `mean_core_voxels` are `BlockGrid`'s, exact, and already
what `price_phase` charges. `visit_order` is `Hints::visit_order`, a plan output.
**So the floor is a function of things the planner decides and already holds.**

**The ceiling is `budget_bytes` minus what the blocks need** — §2's subtraction
read the other way. Both bounds are arithmetic over quantities already exact.

**The optimum is not derivable and must not be invented.** Size against hit rate
is a curve, and nothing has ever measured a point on it, because there has never
been a cache to measure. §0.2 is the reason this matters more than it looks:
*a fitted middle would be a constant chosen on this box, ranked above a term that
was measured.*

**Recommendation.** Default `Constraints::cache_bytes` to the derived floor,
state its domain, and let it be swept — which is `PartitionSearch`/`BlockLadder`/
`SlabPolicy`'s own rule: *anything added should arrive as a named setting with a
default and a stated domain, never an operator and never a constant chosen on
this box.* The floor is defensible from arithmetic and from §0.2's measurement;
anything above it is a search, and the search is somebody else's pass.

### 3.3 And it cannot be sized without the traversal order — which is a finding

Stage 5's acceptance asks explicitly: *"If the cache or the depth cannot be sized
usefully without knowing the traversal order, say so — that is a finding about
the planner's own ordering."* **It cannot, and the `live_frontier` term above is
where that shows.** Whether a neighbour's chunk is still resident when the
neighbour runs is a function of how many other blocks ran in between, which is
the visit order and nothing else.

**The good news is that this is not a blocker but a dependency**, and it points
the same way the outside plan's bounded-horizon heading does: `visit_order` is a *plan output*,
so the planner is the one component that can size the cache, and it can only do
so *after* it has chosen an order. **Cache sizing therefore belongs downstream of
ordering, not beside it** — which makes Stage 4 a prerequisite for the useful
half of Stage 5, not merely an earlier item on a list.

---

## 4. What drives the prefetcher

### 4.1 It needs three things and the executor already holds all three

`Prefetcher::new(cache: Arc<ChunkCache>, depth: usize)` and
`submit(&dyn AccessPlan)`, where `AccessPlan::requests() -> Vec<RegionRequest>`
and a `RegionRequest` is `(ArrayId, Region, rank: u32)`. `BlockPlan::in_order`
already builds one from an array and an ordered sequence of regions.

| the prefetcher needs | where the executor already has it |
|---|---|
| an `ArrayId` | whatever `ChunkCache::register` returned for that image (§1.2) |
| a `Region` per future read | `graph.tasks[t].geometry.source` — **the same field** `placement::read_keys` and `handout::nearest` already use |
| a `rank` | the block's index in the plan's own visit order |
| a depth | `Hints::prefetch_depth`, which exists and is read by nothing (§3.1) |

**Nothing has to be predicted.** `src/prefetch.rs`'s header is right that this is
scheduling rather than prediction — *"the block plan is enumerated up front, so a
worker can declare its future reads and this becomes a scheduling problem with an
exact input"* — and the exactness is not a claim, it is `Decomposition`'s
`exact_read_voxels`, which was measured predicting a real tile-scale run **to the
voxel**: 3 549 686 208 predicted, 3 549 686 208 asked for.

**Rank is the plan's index and must not be recency.** The prefetcher's own header
makes the argument and the outside plan makes it again from the
scheduler's side: a prefetcher ranking on the plan is immune to a greedy compute
scheduler's myopia, because it is not consulting the throughput criterion at all.
That separation is what makes a greedy compute scheduler safe, and it is already
built.

### 4.2 What `PrefetchWasted` must be non-zero against

*"Waste is the cost of depth and is what tells you the depth is wrong; nothing
else in the system will."* The counters exist —
`CacheStats::{prefetch_issued, prefetch_used, prefetch_wasted_evicted,
prefetch_wasted_refused, prefetch_declined}` and
`PrefetchStats::{submitted, started, chunks, cancelled, declined, failed}` — and
have never been non-zero outside `cache_tests.rs`.

**"Waste must be non-zero somewhere in the suite" is the right instinct and the
wrong assertion**, because a run that wastes nothing may simply have a cache
large enough, which is a *good* outcome. The assertable form is a **sweep**, and
it is the same shape as §0.2's:

1. **Depth 0 is the control.** `prefetch_issued == 0`, and the run's answer is
   byte-identical to every other depth. Without this, a sweep measures the
   prefetcher against nothing.
2. **`prefetch_used > 0` at the shallow end.** A prefetcher whose reads are never
   consumed is fetching the wrong things, which is a different defect from
   fetching too many.
3. **Waste rises with depth**, at a cache size held fixed and below the plan's
   total footprint. If it does not, the depth is too shallow to be reaching past
   what the cache would have held anyway — which is exactly the claim the
   original instinct was reaching for, stated so that it can fail.
4. **A liveness control that fails if the sweep never reaches the regime.**
   §0.2's sweep needed one — its roomy end could not evict, and that is precisely
   why the earlier measurement said nothing. A prefetch sweep whose every depth
   fits in the cache is the same failure wearing different clothes.

---

## 5. The counters that would calibrate all of it

**In the discipline this stage has already used: set arithmetic over what
actually happened, no clocks, assertable rather than printable.** The precedent
is `ZarrEnvironment::unaligned_reads`, which settled the halo question from a
counter after two clocks disagreed in sign.

| claim to be settled | counters | assertable form |
|---|---|---|
| a halo re-read is priced as a hit when the cache can hold the neighbour and a miss when it cannot | `CacheStats::{hits, misses}` at two `capacity_bytes` | the same plan, two sizes: hits strictly greater at the larger, `source_reads` strictly fewer, **answer byte-identical** |
| the cache holds what it was told to | `resident_bytes`, `capacity()` | `resident_bytes <= capacity` at every sample — an equality the lease already enforces, so a violation is a lease bug |
| the budget really subtracted it | `Constraints::budget_bytes` against `MemoryBudget` high-water | a plan admitted at budget `B` with cache `C` never exceeds `B`; without the subtraction it exceeds by up to `C` |
| coalescing is working | `source_reads` against `misses` | `source_reads <= misses`, strictly less wherever `max_coalesce > 1` and the plan is contiguous |
| the encoded tier earns its place | `encoded_ratio()` | measured against the store's own codec ratio — `19.7x` on `bool`, `2.09x` on `uint16`, `1.36x` on synthetic `uint16` at `gzip1` |
| depth is not too shallow | §4.2's sweep | waste rises with depth; `prefetch_used > 0` at the shallow end |
| the cache never starves compute | `CacheStats::refusals`, `prefetch_declined` | both non-zero under a deliberately tight budget, and the run still completes |

### 5.1 One trap, and it would destroy the crate's best validation

`EnvCounters::chunks_read` is computed **geometrically** — `chunks_touched(region,
chunk)` — and is not a measurement of IO at all. Today that is harmless, because
every touched chunk is genuinely fetched. **A cache makes the two diverge**, and
if `chunks_read` is quietly repurposed to mean "chunks actually fetched" then
`exact_read_voxels`'s agreement with the run — the one end-to-end validation of
the cost accounting this crate has, verified on a real tile at `2.000x` the
volume — stops holding, and it will look like a planning defect.

> **`chunks_touched` must stay geometric. Actual fetches are `CacheStats::
> source_reads`, and the two must be reported side by side.** Their ratio *is*
> the hit rate, expressed in the currency the planner already prices in.

---

## 6. Does a wired cache make the induced-IO penalty estimable?

**The quantity: yes. The price: no, and that is the finding to surface now.**

The residency plan kept outside this repository — not part of this crate, named
only so the provenance is followable — asks under its heading *"The objective:
bounded-horizon throughput, not a global optimum"* for a penalty on *induced IO*,
the bytes an ordering causes to be re-read, and argues that it should need *"bytes
and chunk counts, which the crate already counts"* rather than a per-voxel time.

### 6.1 The quantity is reachable, and most of the code exists

Given a plan, a visit order, a chunk lattice and a cache size, "how many chunks
does this ordering cause to be re-fetched" is a replay, and the coordinator
**already has the replay**: `ChunkGrid::keys(image, region)` and
`ModelledCache::misses(&keys)`. What made that model untrustworthy in §0.2 was
never the arithmetic — it was that the size it modelled corresponded to no real
cache. **A `Constraints::cache_bytes` that a run actually honours removes exactly
that objection**, and the same replay becomes an honest estimate.

So the prerequisite is §2 and §3's field, not a further measurement.

### 6.2 The price is not reachable, and no cache fixes it

A penalty must be comparable with the thing it penalises. That objective is
throughput — **voxels per unit time** — so a penalty in *chunks* has to be
converted, and this project has refused that conversion twice, with measurements
on both sides of it:

| path | what a re-read costs, relative to a first read |
|---|---|
| in memory | `0.31` at `32³` regions, `0.72` at edge 128 — and the discount is itself a function of block shape |
| flat file, cold | `0.38` — `3.48x` the bytes for `1.32x` the time, **non-monotone** |
| flat file, warm | `0.86` |
| chunked store, default codec | **above `4`** — a halo of 4 costs `5.33x` the time for `1.308x` the voxels |

**Not one-signed, spanning a factor of ten, and `Strategy::decompose` takes no
`Environment` and is documented `BINDING: deterministic, hashable, data-blind`.**
The plan is made before the storage path exists. G20 in `docs/ops-survey/README.md`
refuses a halo weight for exactly this reason and the refusal transfers unchanged.

### 6.3 What that means for the greedy direction

> **The induced-IO penalty is estimable as a chunk count and not as a time, so it
> can enter a greedy scheduler as a *constraint* or as a *tie-break beneath* the
> throughput term — and not as a term added into the objective.**

That is a real limit on the greedy-with-horizon direction and it is better to
have it now. Three readings, and
the third is the useful one:

* Adding chunks to voxels-per-second needs the coefficient above, which is not
  one-signed and which the planner cannot see. A fitted middle would be **a
  constant chosen on this box ranked above a measured term** — §0.2's failure
  exactly, one level up.
* As a **tie-break** it is safe and it is nearly free: between two orderings of
  equal predicted throughput, prefer the one that re-fetches fewer chunks. That
  is `Stage 4`'s *"order so that dead data can die"* in the currency the crate
  counts, and it needs no conversion at all because nothing is being added.
* As a **constraint** it is safer still: refuse an ordering whose replayed
  re-fetch count exceeds some multiple of the plan's own `exact_read_voxels`.
  A refusal needs no price either.

**And there is a cheaper win in the same place.** The replay's *input* is the
chunk lattice, and §1.2's control row says the cost driver on a chunked store is
alignment rather than chunk count — `unaligned_reads` at 64 of 64 for halos of 4,
8 and 16 and **0** for a halo of a whole chunk. A penalty counting *unaligned*
re-fetches rather than re-fetches would track the measured cost far better, at
the same plan-time cost, and it is the one read-side lever the crate counts and
never prices.

---

## 7. What this does not fix

* **It does not make a halo cheap.** §0.1: cold is `42x` warm at one block, and a
  cache changes which reads are cold, not what a cold read costs.
* **It does not touch `ArrayEnvironment`'s residency**, which is where the
  tile-scale problem measured: every image at whole volume whatever the cut,
  unpriced working buffers at `59.6`–`65.4` bytes a voxel. A chunk cache is
  orthogonal to all of it, and a cache landing before those are fixed would be
  **calibrated against a system that still holds dead memory** — the owner's
  sequencing, and the reason this note is a specification.
* **It does not size itself.** §3.2 gives a floor and a ceiling and refuses an
  optimum.
* **It does not make the plan-time cache and the run-time cache the same
  object.** They will be two numbers that must agree, and §5's third row is the
  only thing that would notice if they stopped.
* **It says nothing about a write-side cache.** `produced` in the coordinator's
  model is about written chunks; nothing here proposes retaining them.

---

## 8. What would make this note obsolete

* **A hit rate.** Every "not derivable" above is not-derivable *because nothing
  has measured a point on the size-against-hits curve*. One sweep at three sizes
  on a real plan retires §3.2's refusal to name an optimum.
* **An `Environment` built over `RegionSource`.** §1.1 is the whole of the
  structural work; if that seam is closed for another reason, the cache attaches
  with `CachingSource::attach` and no new type.
* **A measured price for a re-read on the path a run uses.** §6.2 is a refusal
  resting on four numbers with opposite signs. A per-layout coefficient the
  environment could supply at execute time — not the planner at plan time —
  would turn §6.3's tie-break into a term.
* **Fetch and compute overlapping.** §4's whole value rests on the block read
  being a serial prefix, measured at `1.19x`–`1.50x` for block parallelism over
  intra-block threading at equal arithmetic. If `Environment::read` ever streamed
  into a compute that had already started, the prefetcher's case would need
  re-deriving — `intra-block.md` §11 says the same thing about its own ordering.
