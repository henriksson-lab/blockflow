# Barriers

*A specification, not an implementation.* It is written by the work that
measured the cost of not having one — `tests/label_materialisation_cost.rs`, and
`docs/ops-survey/README.md`'s **G7** row, which this note is the long form of.
Figures are cross-referenced there rather than restated, except where an
argument needs one in the line.

The standing rule on this project is to stop when the design does not permit
what is needed and say so, rather than build around it. This is that call. What
follows is what cannot be said, what not saying it costs, what saying it would
have to look like, what it would give up, and what would make this note
obsolete.

---

## 1. What cannot be said today

Two rules are involved and they are **different rules**. Conflating them is the
easy error and it is the reason the first attempt at this analysis reached the
wrong conclusion about what closing G7 would buy.

### 1.1 Rule A — the image-numbering rule

`Decomposition::n_images() == n_phases() + 1`. Image 0 is the run's input and
image `p + 1` is written by phase `p`. A fragment op that declares
`writes_pixels() == false` writes no image, so image `p + 1` does not exist, and
`fragment::check_phase_work` refuses the plan by name:

> phase `i` reads image `i`, which phase `i-1` did not write: it runs a fragment
> op that declares `writes_pixels() == false`. A phase that writes only fragments
> is terminal as far as images go.

This is what blocks the natural **label, merge, relabel** shape — the one
`ops::fill`'s header names as the program these ops are — as *three* phases. A
`fragments -> fragments` middle phase is terminal, so nothing can follow it.

**Rule A is not what this note asks to change.** §4 explains why: with Rule B
answered, the two-phase fold is cheap enough that the third phase stops being
worth its cost, and Rule A is expensive to move — the register's §10 records an
attempt at renumbering images that turned out to be unbuildable, the executor
addressing images positionally at some fifteen sites.

### 1.2 Rule B — the dependency rule, and this is the one

**A task's dependencies are computed by region intersection and by nothing
else.** `TaskGraph::build` gives task `(p, b)` the previous phase's tasks whose
*valid* regions intersect `b`'s `geometry.source` — the region it fetches — and
`TaskGraph::dependencies_cover_reads` then requires those valid regions to cover
that fetch exactly, by area.

That is a good rule. It is why two phases on different lattices have an edge at
all when they have no index correspondence, and it is why a wrong halo costs
reads rather than correctness. But it has one consequence that is not a design
choice anybody made:

> **The region a phase fetches and the set of tasks it waits for are the same
> number.** There is no way to widen the second without widening the first.

So a phase that must see *every* block's contribution before any of its own
blocks may run can only say so by **fetching the whole volume in every block**.
`fragment_phase` makes this explicit: `halo = max(reach, fragment reach x block
edge)`, so a whole-lattice fragment reach produces a whole-volume halo, which
produces a whole-volume `source`, which produces an edge to every task of the
previous phase — *and* a whole-volume read. `ops::fill`'s header already states
the coupling and calls it load-bearing, and it is right: with a zero halo the
plan would be **refused**, correctly, by `dependencies_cover_reads`, because
block `b` would be reading fragments nobody had written yet.

### 1.3 Why this is not a scheduling problem

It is worth ruling out the obvious answer. `Hints::priority` already offers
`SchedulePriority::PhaseMajor` — "every block through phase 1, then phase 2" —
and it is the default. It does not help, and the reason says exactly what the
gap is: **priority re-ranks the ready set; it does not create or remove edges,
and it does not change what a task fetches.** The halo is in the *plan*, the
fetch is derived from the halo, and the bytes are charged when the fetch
happens. Running phase-major moves the same reads to a different moment.

### 1.4 The executor already has a barrier, for one phase kind

This is the strongest evidence that what is being asked for is small. An
**iterative** phase already runs as a unit. `strategy.rs` holds its tasks out of
the ready heap, counts how many have become ready, runs the whole phase when all
of them have, and says so in its own comment: *"it is a barrier by
construction"*. The deadlock argument is written down there and is already
general, not special to iteration:

> A task of phase `p` waits only on phase `p-1`, so holding phase `p`'s tasks
> back blocks nothing that phase `p`'s tasks need; phase 0's tasks are ready from
> the start, and the property carries forward.

What an iterative phase does **not** do is relieve the halo — it does not need
to, because its substages are reach-0 and its dependencies are already satisfied
block by block. It gets its barrier free. A fragment reduction cannot: it needs
"all of phase `p-1` is finished" as an *edge*, and today the only spelling of
that edge is a whole-volume fetch.

---

## 2. The price of not saying it

Measured, on a recorded volume, with a real downstream consumer, at four
lattices, by `tests/label_materialisation_cost.rs`. The whole table and the
machine state are there; what matters here is the shape.

The same operation — a globally consistent label volume — was built twice. Once
with the merge **inside** the plan, as `ops::fill`'s two-phase fold, which is
the only shape the framework admits. Once with the merge **outside** any plan,
run between two `execute_phases` calls, which is a barrier obtained by not being
in the plan at all. Same volume, same consumer, same lattices.

### 2.1 It decomposes to the byte, and there are two amplifications

The in-plan arm's pixel reads are exactly `mask + blocks x (u32 volume) +
consumer`, and its fragment traffic is exactly `(1 + blocks) x F(blocks)`, where
`F` is what all the fragments weigh together. Both were predicted from those
formulae and both matched the counters. **Nothing here is modelled.** The
consumer's own traffic is identical in the two arms, so it cancels.

| at 256 blocks | in-plan merge | merge outside the plan |
|---|---|---|
| pixel reads | 67.427 GiB | 0.393 GiB |
| fragment traffic | 34.864 GiB | 0.271 GiB |
| writes | 0.524 GiB | 0.262 GiB |
| **total** | **106.1 GiB** | **4.2 GiB** |

**25.4x the traffic**, and the answer being computed is a 148.7 KB table.

The pixel half is `blocks x` the label image — **linear in the block count per
block, so quadratic in bytes moved**. The fragment half is `(1 + blocks) x
F(blocks)`, and `F` itself grows with the cut, because fragments are the blocks'
*faces* and cutting more finely makes more face.

**How much faster than linear that is depends on the shape of the lattice, and
it was worth sweeping rather than asserting** — §7.6 does, and finds `F` is the
total face area, dominated by cuts of the volume's **shortest** axis. On a
cubically-cut lattice `F` goes as `blocks^(1/3)` and the fragment total as
`blocks^(4/3)`, which is *sub*-quadratic; on a lattice that never divides the
short axis `F` barely moves. Earlier drafts of this note called it "worse than
quadratic". That was arithmetic on two points and it is wrong.

### 2.2 The linearity is the part that matters

Both amplifications rise with the block count. So the toll is worst **exactly
where cutting finely is most wanted** — which is the direction this project has
to go, since cutting is what makes a stage fit in memory at all. A cost that is
absent on a toy lattice and 25x on a real one is not a coefficient anybody will
notice before they need it.

### 2.3 The correct answer today requires the caller to know something

This is the part that no type enforces and it is the strongest argument for a
declaration.

The cheap arm works because the caller **split its own pipeline** into two
`execute_phases` calls with the merge between them. That is a barrier, and it is
a barrier maintained by the caller's discipline. Nothing checks it. In
particular a caller who instead tried the natural thing — one plan, a consumer
phase reading the label image, the table built lazily on first read — would get
a **plausible wrong answer or an intermittent failure**, because phases
pipeline: the consumer's blocks may begin before every producer block has
written its fragment, and the merge would then be built from an incomplete
fragment set. Whether it fails depends on the schedule.

So the state today is: the expressible shape is 25x too expensive, and the
affordable shape is correct only by an invariant living in the caller's head.
That is the definition of something the framework should be able to state.

---

## 3. What a barrier would have to be

The narrowest change that closes Rule B. Stated as a specification because
`src/strategy.rs`, `src/decomposition.rs` and `src/fragment.rs` belong to
another worker.

### 3.1 The declaration

A phase declares that its dependency on the previous phase is satisfied by
**completion** rather than by region coverage. Following the crate's own
convention that an op states what it needs and the plan records it:

* `FragmentOp::barrier() -> bool`, defaulted `false`, so nothing that ships
  today changes;
* `fragment_phase` records it on `PhaseDecomposition`, beside `halo`, exactly as
  it records `reads_pixels`;
* it is part of the plan, therefore parity-visible, therefore hashed into the
  fingerprint and carried over the distributed wire, for the same reason
  `source_images` is: a plan whose record disagrees with its ops is a plan that
  fetches one thing and waits for another.

**It must be a declaration and not a hint.** `Hints` is the advisory half — "any
strategy may ignore, override or recompute all of it and still be correct" — and
this is not advisory. A barrier ignored is a wrong answer.

### 3.2 The three consequences

1. **`TaskGraph::build`**: for a barrier phase, `deps` is *every* task of phase
   `p - 1`, rather than `producers_of(...)` by region intersection. The edge
   stops being derived from the fetch.
2. **`TaskGraph::dependencies_cover_reads`**: the coverage check as written asks
   whether the deps' valid regions cover the fetch. For a barrier phase they
   cover the whole previous volume, so the check passes for any fetch and needs
   no special case — but it should be *stated* that it passes for that reason,
   or the next person to read it will think the guard has been weakened.
3. **`fragment_phase`**: a barrier phase's halo is no longer forced up by its
   fragment reach. `halo = reach`, and a fragment reduction that reads only its
   own core declares `reach = 0` and fetches its core. **This is where the 67.4
   GiB goes.**

Note what is *not* on this list: the ready heap, the worker pool, the event
stream, the cache, the distributed placement. The executor's readiness loop is
already indegree-driven and already handles a phase held back as a unit.

### 3.3 What it does not fix, and this is the honest half

**A barrier does not remove the fragment amplification.** With `halo = 0` the
relabelling phase still declares a whole-lattice *fragment* reach, and the
executor still gathers every fragment for every block, because the merge is
still re-run per block. From the table above:

| at 256 blocks | today | with a barrier | merge outside the plan |
|---|---|---|---|
| pixel reads | 67.427 | 0.655 | 0.393 |
| fragment traffic | 34.864 | 34.864 | 0.271 |
| **total, GiB** | **106.1** | **39.3** | **4.2** |
| ratio to the cheap arm | 25.4x | **9.4x** | 1.0 |

So a barrier is worth about **2.7x** here and leaves **9.4x** on the table. That
is a real improvement and it is not the whole answer, and a note that claimed
otherwise would be selling something.

> **Both figures in this table were inferences when it was written and are now
> measurements.** `tests/label_materialisation_cost.rs` runs four arms rather
> than two: the in-plan shape, a barrier alone, a barrier with the reduction
> hoisted, and the merge outside the plan. The middle column above was composed
> by hand from two other columns; the harness now builds it — the same run with
> the merge re-derived once per block instead of once — and it comes out at
> **39.29 GiB, 9.41x**, against the 39.3 and 9.4x predicted. The right-hand
> column of §7's table is likewise measured, at **4.70 GiB, 1.13x**. §7 is the
> specification for it, and it revises one sentence of this section: the
> remaining 9.4x is **not** where the largest part of the cost is. See §7.2.

---

## 4. What a barrier would give up

Pipelining across phases is a **real property the executor has now**, not an
aspiration. Readiness is per-task and indegree-driven; a task of phase `p`
becomes ready the moment the phase-`p-1` tasks covering its fetch are done, and
`SchedulePriority::BlockMajor` exists precisely to exploit that — "advance one
block as far through the phases as its dependencies allow", which is fusion and
the smaller working set.

A barrier phase gives that up **at its own two edges**, and nowhere else:

* nothing of the barrier phase starts until all of the previous phase is done,
  so the previous phase's tail is not overlapped with the barrier phase's head;
* the barrier phase's own output image is complete only when the phase is, which
  is already true of any phase a later one reads wholly.

What it does **not** give up: the phases before the barrier still pipeline among
themselves, the phases after it still pipeline among themselves, and blocks
*within* the barrier phase still run concurrently. A barrier is a constraint on
when a phase may **start**, not on how it runs or on what its neighbours do with
each other.

**And in the case that motivated it, nothing is given up at all.** The phase
this is for already has a whole-volume halo, so it already waits for every block
of the previous phase; the barrier changes only *how it says so*. It gives up
pipelining that a whole-lattice reduction never had. That is the strongest form
of the argument and it is also the narrowest: it holds for exactly the phases
that would declare it.

The cost is therefore not "the executor loses pipelining". It is: **a new way to
be wrong exists** — an op that declares a barrier it does not need serialises
two phases that could have overlapped, and nothing will tell it so, exactly as
nothing tells an op that declares too large a halo. That is the same class of
mistake the crate already accepts for reach, and `Decomposition::check` cannot
catch either, because both are statements about what an op needs and only the op
knows.

---

## 5. What becomes possible beyond this one op

If the answer were "only the label volume", that would weaken the case, and it
should be said plainly. It is not.

Every **fragment-and-join** op in the crate is the same program and pays the same
toll: `ops::fill`, `ops::regional`, `ops::detect`, and now `ops::label`'s
component labelling. `src/ops/components.rs` exists because that program has more
than one instance. All four have a phase whose answer depends on every block, and
all four express it as a whole-lattice fragment reach, and therefore all four
carry a whole-volume halo. `ops::fill`'s header names the barrier as the way out
and calls it the open architectural question; `ops::detect`'s phase 1 avoids the
pixel half of the toll, and the reason is worth being precise about because it is
the one existing escape: it declares `reads_pixels() == false`, and the executor
then performs **no pixel IO at all** for that phase — no `Environment::read`, no
counters. Its whole-volume halo costs it nothing in pixels because it fetches
none. That escape is available only to a reduction whose answer is not a volume,
so it is open to `detect` and closed to the other three; and it does nothing
about the fragment half, which `detect` pays in full.

The register reaches the same conclusion from two other directions and names
G7 the **single largest cost lever in family B**, with A's and D's global
reductions paying the same toll by the same route. Concretely, and beyond
labelling: a global auto-threshold's level, any whole-volume histogram or
statistic that a later phase consumes, and the reduce-then-broadcast half of
contrast stretching and equalisation. Each is a reduction whose answer is a
handful of numbers and whose current cost is a whole-volume read per block.

**G8** is this with a loop around it — a substage consuming a scalar reduced over
the previous substage's whole output — and the register calls it the
highest-leverage item in family B's own document. It is not closed by a barrier,
but an iterative phase is already run as a unit by the executor, so a barrier's
declaration is plausibly the same declaration G8 needs one level down. That is a
guess and is labelled as one.

So: **four shipped ops, one named gap that is the largest in its family, and one
adjacent gap that may share the mechanism.** Not one op.

---

## 6. What would make this note obsolete

Stated because a design note whose own conclusion has no expiry date is one that
will be quoted after it stops being true.

1. **If a barrier lands**, the decorate-versus-materialise recommendation in
   `ops::label` moves from 25.4x to about 9.4x on the traffic measured here.
   Still a clear answer, no longer a dramatic one — and the recommendation should
   be re-measured rather than re-derived, because the wall-clock half of it was
   never the trustworthy half.
2. **If a barrier lands *and* a barrier phase's reduction is allowed to run
   once**, the gap closes to **1.13x**, measured (§7.1) rather than projected —
   one extra read and one extra write of the label volume, 8 bytes a voxel — and
   the recommendation becomes genuinely marginal. At that point materialising is
   the better default, because a materialised volume is remapped once and a
   decorated one is remapped per reader, and the whole decorated design's
   advantage was never the write it avoids.
3. **If Rule A is ever moved** and `fragments -> fragments` becomes expressible
   between two pixel phases, §3.3's remaining 9.4x closes by the direct route and
   this note's §1.1 should be inverted rather than deleted.
4. **If the measurement is repeated on storage rather than in memory**, every
   ratio here is a lower bound on the gap: the amplified reads are whole-volume
   and would come from a store rather than from an array, while the fragment
   traffic is small objects. Nothing in this note has been measured against
   `ZarrEnvironment`, and it should not be quoted as if it had.

The one thing that would **not** make it obsolete is a faster machine. Every
ratio above is in bytes, the byte counts reproduced to the digit across runs at
load 32 and load 40, and the wall-clock column is the one that moved.

---

## 7. The larger half: running the reduction once

§3.3 named this and did not specify it, on the grounds that it wanted its own
measurement. It has one now, and the measurement moved the argument twice.

### 7.1 The prize, measured

Four arms, same recorded volume, same consumer, same lattices. At **256
blocks**, total bytes moved — pixels in, pixels out, fragments both ways:

| arm | total | over the cheapest |
|---|---|---|
| in-plan, as the framework admits it today | 106.07 GiB | 25.4x |
| a barrier alone | 39.29 GiB | 9.41x |
| **a barrier, reduction hoisted** | **4.70 GiB** | **1.13x** |
| the merge outside the plan entirely | 4.18 GiB | 1.00 |

All four agree on the answer — same component count, same rows — at every
lattice, and every byte column reproduces to the digit across runs. The third
row is the prize: **1.13x**, which is the corrected expiry condition of §6.2
arrived at by measurement rather than by arithmetic.

### 7.2 The revision to §3.3, and it is not the one expected

§3.3 said a barrier is worth 2.7x and leaves 9.4x on the table, and both halves
are confirmed. What it did not say, because it was reasoning in bytes only, is
that **the dominant cost of re-deriving the reduction per block is not traffic
at all — it is CPU.**

Re-deriving the merge once per block, timed directly: **0.10 s at 1 block, 0.14
s at 4, 2.47 s at 32, 33.67 s at 256** — serial, so divide by the concurrency
for a wall-clock contribution, and it is CPU-seconds either way on a shared
machine. The whole hoisted arm's run is 3.80 s. So at 256 blocks the redundant
union-find alone costs **roughly nine times the entire hoisted pipeline**, in a
currency no byte column shows.

This contradicts a sentence that has stood unchallenged in `ops::fill`'s header
and in earlier drafts of this note: *the union-find is over face labels rather
than voxels, so it is small next to the pixels, but it is N times redundant and
it is not nothing.* **It is not small.** It is small *per invocation* — about
0.13 s — and there are `blocks` invocations, and nobody had multiplied.

So the case for hoisting is stronger than §3.3 made it and rests on a different
quantity than §3.3 used.

### 7.3 Why the merge is re-run per block at all

Not because anybody chose it. `check_phase_work` (§1.1, Rule A) makes a
`fragments -> fragments` phase terminal, so the merge cannot be its own phase
between two pixel phases; it folds into the relabelling phase, and a phase is a
function applied **per block**. There is nowhere in `FragmentOp` to put a result
that belongs to the phase rather than to a block. The op is `&dyn FragmentOp`
shared across workers and `apply` takes `&self` and a `BlockView`; every
quantity it can produce is per-block by construction.

### 7.4 Is this the same change as `barrier()`?

**No — it is a second one, and it needs the first.**

`barrier()` is a **precondition** and not a substitute.

* Without a barrier, hoisting is **not well defined**. Blocks of phase `p`
  become ready individually, so there is no moment at which "the fragment set is
  complete" exists to hoist to. A reduction hoisted without a barrier would be
  computed from whatever had been written, which is a schedule-dependent wrong
  answer — the same hazard §2.3 describes on the caller's side.
* With a barrier, the moment exists and is exactly the phase's start.
* But a barrier does not by itself let the op *say* what to compute at that
  moment. The executor cannot hoist what it cannot see, and `apply` is opaque to
  it.

So: two declarations, one dependent on the other. §3.3 was right to separate
them, and this section is not a correction of that.

### 7.5 What it would have to be

The narrowest object-safe shape, following the crate's own conventions:

```rust
/// Computed once for the phase, after the barrier, before any block runs.
/// Empty by default, so nothing that ships today changes.
fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> { Ok(Vec::new()) }
```

with `BlockView` gaining a `reduced: &[u8]` field, and a plan-time refusal for an
op that declares `reduce` without `barrier`.

Four points, each of which is a decision rather than an obvious step:

1. **`Vec<u8>` and not an associated type.** `FragmentOp` is used as
   `&dyn FragmentOp` throughout; an associated type is not object-safe. Bytes are
   also the currency fragments already travel in — every op on this machinery
   already has `encode`/`decode` — so the round trip is not new machinery, it is
   the machinery. For the label case it costs an encode and decode of 92-149 KB
   against the 34.6 GiB it removes.
2. **`PhaseView` gives the whole fragment set, not the pixels.** What a reduction
   needs is every block's fragments and the lattice; if it needed pixels it would
   not be a reduction over fragments and it should stay a per-block op.
3. **One blob for the phase, not one per block.** A per-block result would be a
   `fragments -> fragments` phase wearing a different hat, and that needs Rule A.
   The whole-phase blob costs `blocks x table` in transmission — 256 x 148.7 KB =
   0.036 GiB at the finest lattice measured — against the 34.6 GiB it removes.
   **This is the version that does not need Rule A, and that is why it is the one
   specified.**
4. **Not a `OnceLock` inside the op**, which is the tempting version that needs no
   framework change at all: with a barrier, an op could fill a `OnceLock<T>` on
   its first `apply` and every later block would find it. It is wrong for a
   reason the register already recorded once. G10's `apply_side` held a per-block
   map keyed by buffer offset and it was deleted, because a method should be *a
   total function of its operands* rather than of its operands plus what an
   earlier call left behind. A `OnceLock` makes the op single-use: run the same op
   over a second lattice and it answers with the first lattice's table, silently.

### 7.6 Is `F(blocks)` growing intrinsic, or an artefact?

Two questions, separated, and both swept —
`the_fragment_set_grows_with_the_cut_and_how_fast_depends_on_the_shape_of_it`.

**Must the fragment set be that big? Yes, and it is geometry.** `F` is the total
face area of the cut, and it obeys one line:

> `F = sum over axes of (cuts on that axis) x (area of the face perpendicular to
> it)`

Measured on `[16, 1304, 3369]`, with the divisions of axis 0 in brackets:
`0.0335 (1)`, `0.0668 (2)`, `0.1334 (4)`, `0.1357 (4)`, `0.2667 (8)` GiB —
linear in the cuts of the shortest axis to three figures. Any scheme that tells
one block about another's boundary pays this. It is not an artefact and hoisting
does not remove it.

**Must it be re-transmitted per block? No.** That is the `(1 + blocks)`
multiplier and it is entirely an artefact of the reduction being per-block. The
hoisted arm transmits the set exactly twice — written once, read once — at every
lattice: 0.067, 0.068, 0.136, 0.271 GiB against the in-plan arm's 0.067, 0.170,
2.241, **34.864**.

So the growth is intrinsic and the multiplier is not, and the multiplier is
where all of the money is.

**Two things this sweep overturned**, both of the shape the pricing work has been
warned about — a quantity assumed not to vary, which varied:

* *The prediction about lattice shape was backwards.* The expectation was that a
  slab cut carries more face than a cube cut at equal block count, reasoning
  about seam planes in a cube. This volume is not a cube: at 64 blocks the
  **cube** cut carries **3.08x** the fragments, because it divides the 16-voxel
  axis and the slab cut never touches it. The lesson is not about slabs; it is
  that every other figure in this analysis varies the block **count** while
  holding the lattice **shape** fixed, and the shape is worth 3x.
* *"The fragments are small next to the pixels" is false at fine cuts.* At `[8,
  8, 8]` the fragment set is **101.9%** of the label image: one transmission of
  the fragments costs more than one transmission of the whole volume. That
  sentence is in `ops::fill`'s header and was in this note.

**A planning consequence, and it belongs to somebody else.** In-plan fragment
traffic is `(1 + blocks) x F`, and dividing the shortest axis raises *both*
factors, so it is doubly expensive: at 482 slab blocks `(1 + n) F` is 52.4 GiB
and at 512 cube blocks it is **136.8**. The partition search prices neither
fragment traffic nor merge CPU — its cost model has `cost_per_block` and
`cost_per_voxel` and nothing that grows with the block *count* per block. A
planner choosing a lattice on pixel cost alone will choose the expensive one.
That is adjacent to G4 and is not this note's to fix; it is recorded because the
sweep found it.

### 7.7 What hoisting does not fix, and what it gives up

**Does not fix:**

* the intrinsic `F` above — 0.271 GiB at the finest lattice measured, which is
  the floor;
* the residual 1.13x over the fully out-of-plan arm. That is one extra read and
  one extra write of the label volume — 8 bytes a voxel — and it is the price of
  materialising at all, not of anything in this note. It does not go away and
  should not: §6.2 is where it is argued that at that point materialising becomes
  the better default anyway;
* **residency.** `FragmentOp::gathers() == false` with
  `BlockView::stream_fragments` already exists and bounds how much of the
  fragment set is resident at once; it moves the same bytes. Hoisting bounds the
  *traffic* and makes the resident set the reduction's own working set, which is
  a different quantity and could be larger. Nothing here measured it.

**Gives up:**

* **A second way to be wrong, and it is worse than the barrier's.** An op that
  computes the wrong thing in `reduce` gets a plausible answer in every block,
  and unlike a wrong halo there is no guard that could catch it: the executor
  cannot know what the reduction was supposed to be. The mitigation is the one
  the crate already uses for fragments — the blob is the op's own encoding, so
  the op's own `decode` is where a mismatch surfaces, with a magic and a version.
* **Nothing in scheduling.** Unlike the barrier, hoisting removes no concurrency:
  the reduction happens at a moment the barrier has already serialised. It is
  strictly work removed.
* **A reduction that is genuinely per-block loses nothing either**, because
  `reduce` is defaulted empty and an op that does not override it is unchanged in
  every respect.
