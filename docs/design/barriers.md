# Barriers

*Sections 1-7 were a specification and are now the specification of something
that exists. §8 is written by the implementation of it; §9 by the work that
closed the one thing §8 left open; §10 by the work that migrated the shipped ops
onto it, which is where the cost §2 measures is actually collected. Nothing in §1-7 has been edited and §8 is
edited only where §9 supersedes it, by name — a note that quietly agrees with
itself afterwards is not evidence, so where a later section contradicts an
earlier one it says so by number.*

This note is written by the work that measured the cost of not having one —
`tests/label_materialisation_cost.rs`, and `docs/ops-survey/README.md`'s **G7**
row, which this note is the long form of. Figures are cross-referenced there
rather than restated, except where an argument needs one in the line.

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

---

## 8. What was built, and what the specification got wrong

*Written by the implementation, and the only part of this note that was added
after there was one.*

Both changes landed, in §7.4's order. `FragmentOp::barrier` and
`FragmentOp::reduce` are both defaulted, so no shipped op changes; the record is
`PhaseDecomposition::barrier`, hashed into the fingerprint when set and written
to the wire when set, so a plan with no barrier in it fingerprints and serialises
exactly as it did before. `tests/barrier_phase.rs` is the executable half.

### 8.1 §3.2 named three consequences and there are five

The second and third are as specified and are built; the first was built as
specified, measured, and then replaced by something that says the same thing for
free — §8.2. Two more consequences were found only by building it, and both are
guards that would otherwise have refused the very plan the barrier exists to make
expressible:

4. **`fragment::check_phase_work`'s halo-against-fragment-reach guard.** It
   refuses a phase whose halo is narrower than its fragment reach, and its stated
   reason is exactly Rule B: *"the halo is what makes the neighbours' tasks
   dependencies of this one, so a short halo would read fragments nobody has
   written yet."* A barrier phase has `halo = 0` and a whole-lattice fragment
   reach, so it fails this guard by construction. It is skipped for a barrier
   phase, and skipped for the reason the guard is about rather than by exemption:
   the barrier states the edge directly, so the fragments exist whatever the halo
   is.
5. **The record must be checked against the op.** §3.1 says a plan whose record
   disagrees with its ops fetches one thing and waits for another; nothing in §3
   says where that is checked. It is `check_phase_work`, in both directions, plus
   a refusal of a barrier recorded on a phase that runs no fragment op at all —
   nothing could have declared that one, and what it does is serialise two phases
   silently.

§3.2's second consequence — that `dependencies_cover_reads` passes without a
special case — is confirmed, and by a shorter route than §3.2 expected. §3.2
reasoned that a barrier task's deps are the whole previous phase, whose valid
regions tile, so the areas sum for any fetch. Since §8.2 the deps are the
ordinary region-derived ones, so the guard passes for the ordinary reason and
there is nothing special to explain at all — the phase fetches its own core and
the tasks covering that core cover it.

That turns out to be the better arrangement for a second reason §3.2 did not
raise: **it keeps the guard's full force on the thing that shrank.** Widening a
barrier phase's deps to the whole phase below would have made the coverage check
pass for *any* fetch, including a wrong one, at exactly the moment the halo was
relieved. It now cannot.

### 8.2 §3.2's *first* consequence did not survive, and was replaced

> "for a barrier phase, `deps` is *every* task of phase `p - 1`"

**Built exactly as written, measured, and then withdrawn.** It was correct and it
was free for both schedulers — the in-process executor and the distributed
coordinator both drive readiness off `Task::n_dependencies` and
`TaskGraph::dependents`, so the cross product enforced the barrier with no new
code anywhere. What §3.2 does not price is that it is `blocks(p) x blocks(p-1)`
edges, **a product where every other edge in the graph is a sum**:

`a_barrier_at_a_large_block_count_is_priced` builds both, side by side, on one
plan — `deps` plus `dependents`, which are the two things a scheduler holds:

| blocks | shipped (edges + dependents) | in time | withdrawn (edges + dependents) | in time | usize held by the withdrawn form |
|---|---|---|---|---|---|
| 64 | 64 + 64 | 0.000 s | 4 096 + 4 096 | 0.000 s | < 1 MiB |
| 512 | 512 + 512 | 0.003 s | 262 144 + 262 144 | 0.018 s | 4 MiB |
| 4 096 | 4 096 + 4 096 | 0.027 s | 16 777 216 + 16 777 216 | 0.421 s | **256 MiB** |

Extrapolating to the 6 700 blocks `graph.rs` names: some 45 M edges, around 720
MiB and over a second, to express one bit per phase — against a whole hoisted
pipeline that runs in 3.80 s.

Two things decide it against the edges:

* **`graph.rs` had already rejected `O(blocks^2)` at that scale**, in
  `producers_of`'s own header, on the grounds that it makes the DAG cost more to
  build than the work it schedules. That argument was made before this feature
  existed, about a different one, and it applies here unchanged.
* **A barrier exists *for* fine cuts.** §2.2 is that the toll it removes is worst
  precisely where cutting finely is most wanted. Paying a quadratic at the block
  counts the feature is for is the wrong side of that trade.

**What replaced it.** "Every task of `p` waits for every task of `p-1`" is a
statement about two *phases*, so it is stated once: `TaskGraph::barriers` is one
bool per phase, `Task::deps` stays the ordinary region-derived edges for every
phase, and each scheduler gates. A barrier now costs the graph **nothing** —
`a_barrier_costs_the_graph_nothing` asserts the graph is the same size with and
without one.

This changes what a `TaskGraph` *is*, and that is said rather than slipped in:
**the edges are no longer the whole ordering.** A consumer that runs a task once
its `deps` are done is correct for every phase except a barrier phase, whose
blocks fetch their own cores and whose deps therefore clear long before the phase
below has finished. `TaskGraph::is_barrier` is the method such a consumer must
consult, and its documentation names the two schedulers that do and the shape
they share.

**What it costs when wrong, in each direction.** With the edges: a quadratic
paid unconditionally for a property only a distributed scheduler needs
*spelled*, at the block counts the feature targets. Without them: a scheduler
that ignores `barriers` starts a block early and the op answers from an
incomplete fragment set — a plausible wrong answer, schedule-dependent, and in a
distributed run a different one on each machine. The second is worse per
occurrence, which is why it is guarded by tests on both schedulers and why each
gate is mutation-tested: removing either fails the correctness arm, not only an
ordering assertion.

### 8.3 The barrier is enforced once, in the scheduler, and that is now said

The executor holds a barrier phase's tasks out of the ready heap and releases
them together, exactly as an iterative phase's are held. §3 said the ready heap
was **not** on the list of things to touch. It had to be, for two reasons that
turned out to be independent:

* **`reduce` needs a place to stand** — a moment after the phase below has
  finished and before any block of this phase starts. That moment is not a task,
  so no per-task hook could have been it.
* **Since §8.2, it is the whole enforcement.** While the barrier was edges, the
  hold-back was redundant with the indegree; mutation-testing found that with the
  edges removed and the hold-back kept, every correctness test still passed. That
  redundancy is gone deliberately: *a correctness property enforced in one place
  that reads as though it is enforced in two is worse than one honestly enforced
  once.* Both `strategy.rs` and `coordinator.rs` say in the code that the gate is
  the only thing holding the property.

**The release condition is stated rather than derived.** Both gates ask *has
every earlier phase finished*, not *has every task of this phase become ready*.
The second is the same moment in any plan whose valid regions tile — the
induction is real, and mutating the executor's gate from "every earlier phase" to
"phase `p-1` only" leaves even a three-phase plan reducing over phase 0's stream
passing — but it arrives there through an invariant proved in another file. The
broad form costs one comparison per phase per wave and buys a line that says the
property. If the tiling invariant is ever relaxed, it does not quietly become
wrong.

**Neither gate can deadlock**, on the argument `strategy.rs` already wrote for an
iterative phase and which this does not change: a task of phase `p` waits only on
earlier phases, so holding phase `p` back blocks nothing phase `p` needs; phase 0
is ready from the start, and the property carries forward.

### 8.4 The plan-time refusal of `reduce` without `barrier`

§7.5 requires it and does not say how, and there is a real obstacle: **a trait
method's being overridden cannot be observed.** Three shapes were available and
the third was taken.

* *A third boolean* — `fn reduces(&self) -> bool` — is the crate's usual
  convention, and it is the one an op author can override `reduce` and forget,
  leaving the reduction silently never run and every block answering from an
  empty blob.
* *Nothing at plan time*, relying on the op's own decode to fail on an empty
  blob. That is a run-time failure and a loud one, but it is not what §7.5 asks
  for.
* **A probe.** `check_phase_work` calls `reduce` on every non-barrier fragment
  phase with a `PhaseView` that holds no environment and refuses every accessor.
  The default answers `Ok(&[])` for free; an override either touches the view,
  which errors, or returns bytes, which is equally an answer. Either way the plan
  is refused by name.

What the probe misses: an override that ignores the view and returns empty, which
is the default by another spelling. What it costs, with the count named because
the caller controls it: one virtual call per **non-barrier fragment phase** per
`check_phase_work`, and `check_phase_work` runs once per `execute_phases` — a
call the conformance sweep makes thousands of times. That is affordable only
because the default answers `Ok(Vec::new())` and an empty `Vec` allocates
nothing. An override pays whatever it does before it touches the view at that
same multiplicity, and one that unwraps rather than propagating panics there
instead of erroring. All of it is visible; none of it is silent.

**Re-examined after §8.2 and unchanged.** Moving the barrier from edges to a
phase-level fact changes how the *schedule* is expressed and touches nothing
about how an op *declares*, so it opened no cleaner spelling here. This is a
decision §7.5 did not make and it should be revisited if a fourth shape appears —
the most likely being something the compiler could see, which Rust does not offer
for a defaulted trait method.

### 8.5 A reduction is order-checked, which §7 did not ask for and should have

*The price this section quotes is corrected by §10.2: the check costs one extra
reduction **and one extra pass over the whole fragment set**, because `PhaseView`
reads from the store on every walk.*

`SeamFold::Unordered` is the claim that an op's answer is a function of the
**set** of fragments it is handed and not of their order, and the executor
already checks it per block by applying the block a second time with the
neighbourhood reversed. A hoisted reduction has exactly the same exposure and §7
does not mention it: `PhaseView` walks the lattice row-major, which is one order
out of many, and **two different lattices walk two different ones** — so an `f64`
accumulation in `reduce` makes the phase's answer a property of how the volume
was cut, which is the one property this crate is arranged around.

So the same check is applied: an op declaring `SeamFold::Unordered` has its
reduction run a second time over the reversed lattice and the two must be
byte-identical. It costs one extra reduction for an op that opted in, against the
one extra `apply` **per block** the same declaration already costs — so hoisting
makes this check cheaper too, by the same multiplier it makes everything else
cheaper by. `a_reduction_that_does_not_associate_is_refused` is the executable
half, with an integer fold beside it as the liveness control.

This is not a guard on whether the reduction computes the *right* thing. §7.7 is
right that nothing could be: the executor cannot know what the reduction was
supposed to be. It is a guard on whether the answer is a function of the
decomposition, which is a different question and is checkable.

### 8.6 §7.5.3's transmission cost does not exist in this executor

§7.5 point 3 prices the whole-phase blob at `blocks x table` in transmission —
0.036 GiB at the finest lattice measured — and argues that this is affordable
against the 34.6 GiB it removes. **In-process it is not transmitted at all.**
The executor holds one `Vec<u8>` per phase and every block is handed a `&[u8]`
borrowed from it, so the count is one encode and zero copies whatever the block
count is. The argument was right and the number it was defending against turns
out to be zero.

It becomes real the moment the blob has to travel to another process, which is
exactly §8.7's gap. So §7.5.3's arithmetic should be read as the price of
*distributing* a hoisted reduction rather than of having one, and it is still the
right arithmetic for that.

### 8.7 A hoisted reduction was refused in a distributed run — see §9

`strategy::execute_task_of` is a single-task entry point that takes no blob, so
a worker calling it for a barrier phase would hand the op an empty
`BlockView::reduced` and get the plausible-in-every-block wrong answer §7.7 says
no guard could catch. It is refused there, by the same probe, with a message
naming the gap.

**A barrier without a reduction distributes perfectly well** — it is an ordering
the scheduler enforces and the block itself is an ordinary block — so only the
pair was refused.

*This section is superseded by §9, which closed it. The refusal on
`execute_task_of` remains and is still right: that entry point has no blob. What
changed is that there is now `execute_task_with_reduction`, which does, and that
nothing has to be shipped to fill it.*

### 8.8 The measurement, and what transfers from it

`tests/barrier_phase.rs` builds the three in-plan arms out of **one op with two
booleans**, so that every other line is the same line and a difference in the
counters is attributable to the declaration. Volume `[16, 16, 16]`, `f64`, the
framework's own `EnvCounters`, every column predicted from a formula and then
compared:

| blocks | arm | read | write | fragments | total | folds |
|---|---|---|---|---|---|---|
| 32 | in-plan | 1 081 344 | 65 536 | 8 448 | 1 155 328 | 32 |
| 32 | barrier alone | 65 536 | 65 536 | 8 448 | 139 520 | 32 |
| 32 | barrier, hoisted | 65 536 | 65 536 | 512 | 131 584 | **1** |
| 256 | in-plan | 8 421 376 | 65 536 | 526 336 | 9 013 248 | 256 |
| 256 | barrier alone | 65 536 | 65 536 | 526 336 | 657 408 | 256 |
| 256 | barrier, hoisted | 65 536 | 65 536 | 4 096 | 135 168 | **1** |
| 512 | in-plan | 16 809 984 | 65 536 | 2 101 248 | 18 976 768 | 512 |
| 512 | barrier alone | 65 536 | 65 536 | 2 101 248 | 2 232 320 | 512 |
| 512 | barrier, hoisted | 65 536 | 65 536 | 8 192 | 139 264 | **1** |

Ratios to the hoisted arm: **66.7x / 4.9x / 1.00** at 256 blocks, **136.3x /
16.0x / 1.00** at 512.

**What transfers to §7.1's table and what does not.** The *structure* transfers
and every term of it is asserted rather than read off:

* pixel reads are `(1 + blocks) x volume` without a barrier and `2 x volume` with
  — §1.2's coupling and §3.2.3's relief, term for term;
* fragment traffic is `(1 + blocks) x F` per block and `2 x F` hoisted — §7.6's
  multiplier, and the hoisted arm transmits the set exactly twice at every
  lattice, as §7.6 predicted;
* the fold runs `blocks` times per block and **once** hoisted, which is §7.2's
  quantity and the one no byte column shows.

**The absolute ratios do not transfer, and the reason is worth stating because it
is the same lesson §7.6 recorded.** A fragment is eight bytes here and a block
*face* there, so `F` is a rounding error at this scale and a third of the total at
that one. That is why the barrier-alone arm is worth 4.9x here and 2.7x there,
and why the residual after a barrier is 4.9x here and 9.4x there. The direction,
the decomposition and the formulae are the same; the constants are a property of
what a fragment weighs, and quoting one table's constants against the other's
volume would be exactly the error §7.6 caught twice.

The residual **1.13x** of §7.1 — the fully out-of-plan arm — does not appear
here, and should not: it is the price of materialising at all, which all three
in-plan arms pay identically.

### 8.9 §6's expiry conditions, revisited

*Closed by §10.3 and §10.4: the measurement this section records as not having
been run has been run.*

§6.1 and §6.2 both fire. A barrier has landed and so has the hoisting, so the
decorate-versus-materialise recommendation in `ops::label` moves — but the
figures it moves to are §7.1's, and **`tests/label_materialisation_cost.rs` has
not been re-run against this implementation**, because it reads a machine-local
recording that is not on the machine this was built on. The four arms there are
still the out-of-plan simulations they were written as. Re-running them with
`ops::label` migrated onto `barrier` and `reduce` is the measurement that would
close §6.1 and §6.2 properly, and it is not this change.

§6.3 is untouched: Rule A is where it was, and §7.5's whole-phase blob is
specifically the version that does not need it.

§6.4 is untouched and is worth repeating: nothing here has been measured against
`ZarrEnvironment`, and every ratio above is in-memory.

### 8.10 What is not built

*The first bullet is inverted by §10.1 — all four ops now declare both halves.*

* **No shipped op declares a barrier.** `ops::fill`, `ops::regional`,
  `ops::detect` and `ops::label` are all still the in-plan shape; migrating them
  is where the 25.4x is actually collected, and it is a change to files this note
  does not own.
* ~~**The blob is not on the distributed wire**~~ — **closed by §9**, and not the
  way this line expected: the blob never goes on the wire because every node
  derives it from a fragment set they all read. A barrier *alone* already
  distributed, through `distributed::coordinator::Job::ready`'s gate; a hoisted
  reduction now does too, through `strategy::reduce_phase` on each worker.
* **The planner still prices neither fragment traffic nor merge CPU**, which is
  §7.6's parting observation and is unaffected by anything here: a barrier makes
  the cheap lattice cheaper without making the cost model able to see it.
* **Residency is unmeasured**, exactly as §7.7 said. `PhaseView` offers a
  streaming accessor for the reduction's own working set and a gathering one
  beside it; which a reduction needs is the reduction's business and nothing here
  bounds it.
* **G8 is untouched.** §5 guessed that a barrier's declaration might be the same
  declaration a *substage* reduction needs one level down. Nothing here tests
  that and `src/iterate.rs` is unchanged; the guess is still a guess.


---

## 9. The blob does not go on the wire

*Written by the work that closed §8.7. The section it closes framed the remaining
gap as "putting the blob on the wire", and that framing is the thing that did not
survive.*

### 9.1 The question, and why the obvious answer is the wrong one

§7.5 makes `reduce` produce one `Vec<u8>` for the phase, consumed by every block.
In-process that is one allocation and a borrowed `&[u8]` per block — §8.6 records
that §7.5.3's `blocks x table` transmission cost turns out to be zero there.
Across processes it looked as though the cost must become real, and the work was
framed as a transport: one node reduces, the blob travels, the others consume it.

**It does not have to travel, because it is derived rather than observed.** Three
facts, each already true of this crate before the question was asked:

1. **The fragment set is on storage every node reads.** `SharedVolume` puts
   sidecars under the job's own directory through `FileSidecars`, and
   `fragments_written_by_several_worker_processes_are_readable_by_one_merging_reader`
   has asserted for some time that a fragment written by one process is readable
   by another.
2. **`PhaseView` walks the lattice in an order that is a function of the plan**,
   not of the schedule — `grid.cores()`, row-major. Two workers therefore feed
   their `reduce` the identical sequence of identical bytes.
3. **The barrier already tells a worker when the set is whole.** The coordinator's
   gate (§8.3) does not hand out a task of a barrier phase until every earlier
   phase has been *reported* complete, and a worker writes its fragments before it
   reports. `FileSidecars::put` is write-then-rename, so there is no prefix for a
   peer to read.

So every worker computes the blob itself, on the first task of a barrier phase it
is handed, and reaches byte-identical bytes with **nothing sent between them**.

### 9.2 The two arms, and what each costs when wrong

| | every node reduces | one node reduces, blob shipped |
|---|---|---|
| fragment reads | `nodes x F` | `F` |
| folds | `nodes` | 1 |
| bytes on the wire | **0** | `nodes x table` |
| wall clock at the barrier | one fold, in parallel | one fold, then a fetch, serialised |
| new protocol state | none | upload, download, an election, a reducer-death path |
| new state in a component that holds no data | none | all of it |

The right-hand column is cheaper in bytes — `table` is kilobytes and `F` is the
whole fragment set, so shipping wins on traffic by three orders of magnitude per
node. **It loses on everything else**, and two of those are not tradeable:

* **The coordinator holds no data.** That is not an accident of the
  implementation; it is written into `coordinator.rs` — the cache model *"is never
  authoritative about anything"*, the completions are its whole accounting, and
  events are observation. Making it a data plane for one feature is a larger
  change to this system than the feature is.
* **Shipping adds a failure mode with no existing answer.** If the reducing worker
  dies between computing the blob and posting it, every other worker is blocked on
  a fetch that will never complete, and the job's recovery story — reissue the
  task — does not cover it, because the reduction is not a task.

Deriving costs `nodes x F` reads and `nodes` folds, and **the count is the whole
argument**. The case against re-deriving per block was never that re-deriving is
expensive; it is that the multiplier was `blocks`, which a caller raises to make a
stage fit in memory — §7.2's 33.67 s at 256 blocks is 0.13 s multiplied by
something the caller controls in the direction of *more*. This multiplier is
`nodes`, which a caller sets from the machines they have. **It does not move when
the lattice does.** At the finest lattice §7.6 measured, one transmission of the
fragment set is 0.1355 GiB, so eight workers pay about 0.95 GiB more than a
single-process run — against the 34.6 GiB the hoisting removed, and flat in the
block count.

**When each is wrong.** Deriving is wrong if the sidecar store is not in fact
shared between nodes: every worker then reduces over its own fragments and answers
plausibly and differently on each machine. That is the failure this design has, it
is the exact shape §7.7 says no guard could catch *afterwards*, and so it is
guarded *before*: `strategy::reduce_phase` verifies the completeness the barrier
promises, for every declared input stream whose producer said
`Coverage::EveryBlock`, and refuses by name. Shipping is wrong if the reducer
dies, and that has no guard at all — it is a liveness failure in a protocol that
does not model the reduction as work.

**A side effect worth naming.** Nothing ran `check_fragment_coverage` in a
distributed run before this: `execute_task_of` is per task and a worker has no
end-of-phase moment. Putting the check in `reduce_phase` closes that gap for
barrier phases as well as guarding the reduction.

### 9.3 Determinism has two axes and only one of them is `SeamFold`

§8.5 added an order check because `PhaseView` walks row-major and **two lattices
are two orders**, so an `f64` fold in `reduce` is as decomposition-dependent as one
in `apply`. A distributed run looks as though it adds a second axis, and it does
not:

* **Across nodes**, the lattice is the same, so the walk is the same, so any
  *deterministic* `reduce` agrees — associative or not. `SeamFold` does not enter.
* **Across lattices**, the walk differs, and that is exactly what
  `SeamFold::Unordered` declares and what the reversed-lattice check tests.

So there is no new declaration and no new check. What there is instead is a test
that two processes agree, and it agrees on the **output volume** rather than on a
reported blob size, because the op writes the reduction into every voxel: a worker
that reduced differently writes a different volume.
`a_hoisted_reduction_is_byte_identical_however_many_workers_run_it` is that test,
and it is mutation-tested — making `reduce` depend on the process id makes it fail
with `12288 of 65536 byte(s) differ`, which is the disagreement it exists to
catch.

### 9.4 What landed

* `strategy::reduce_phase` — public, one implementation, called by
  `execute_phases` when its gate opens and by `distributed::worker` on the first
  task of a barrier phase. It carries the completeness guard and the
  reversed-lattice order check, so the two schedulers cannot drift.
* `strategy::execute_task_with_reduction` — the single-task entry point that takes
  a blob. `execute_task_of` delegates to it with `&[]` and keeps its refusal,
  which is still right: **that** entry point has none.
* `distributed::worker` holds one blob per barrier phase for the life of the job —
  on the worker, not in the op, for `reduce`'s own reason: an op that cached its
  answer would answer a second lattice with the first lattice's table.
* `WorkerReport::reductions` and `reduced_bytes`, reported per worker, so "once per
  phase, not once per task" is measured across processes rather than asserted from
  the design.
* `distributed::spec::HoistedReduceOp`, the probe that exercises it, with a magic
  and a version in its blob — the mitigation `reduce` names, since a blob is the
  op's own encoding and the op's own decode is the only place a mismatch surfaces.

Measured, on a 16-block lattice with four workers: **one reduction per worker that
saw a block of the phase, 16 bytes each, halo `[0, 0, 0]` with a barrier against
the whole volume without one.** Three workers reducing independently produce the
same volume as one worker, byte for byte.

### 9.5 What is still refused, and why

`execute_task_of` still refuses a reducing op, and should: it is the entry point
that carries no blob, and handing one an empty slice is the wrong answer in every
block. A caller with a barrier phase calls `reduce_phase` and then
`execute_task_with_reduction` — which needs no transport, so the refusal costs
nobody anything they cannot have.

**A worker whose store is not shared is refused rather than answered**, by name,
naming the unshared store as the likely cause. That is the honest form of the one
thing this design depends on.

---

## 10. Collected

*Written by the work that migrated the four shipped ops onto §8's mechanism.
§8.10's first line was that no op declared either half; this section is that line
inverted, and the three corrections the migration produced. Nothing above is
edited; where this contradicts an earlier section it says so by number.*

### 10.1 §8.10's first bullet is inverted

> "**No shipped op declares a barrier.** `ops::fill`, `ops::regional`,
> `ops::detect` and `ops::label` are all still the in-plan shape; migrating them
> is where the 25.4x is actually collected."

All four now declare `barrier()` and a hoisted `reduce()`, and
`grep -rn "fn barrier(&self)" src/` returns them, the trait default and the
distributed probe. The 25.4x is collected: §10.3 has the recorded-volume
measurement and the per-op numbers are in each op's own header.

`ops::components` is where the shared half landed — `Merge`, which selects the
shape, and `encode_block_flags`/`decode_block_flags_for`, which are the blob.
That the blob is *one* encoding across four ops is not a tidiness argument: every
one of these merges produces one flag per label per block, which is what made
§7.5's `Vec<u8>` cheap rather than a compromise.

### 10.2 §8.5 undercounts, and the missing term is a whole transmission of the fragment set

§8.5 prices the reversed-lattice order check at **"one extra reduction for an op
that opted in"**. That is right about the fold and silent about the reads.
`PhaseView` reads its fragments out of the store on every walk, so a second
reduction over the reversed lattice **re-reads the whole fragment set**. The
hoisted arm of an op declaring `SeamFold::Unordered` therefore transmits `F`
**three** times — written once, read by the reduction, read again by the check —
and not twice.

This corrects two sentences elsewhere by name:

* **§7.6**: *"the hoisted arm transmits the set exactly twice — written once,
  read once — at every lattice"*. True of an op that does not declare
  `Unordered`; three times for one that does.
* **§8.8**: the fragment column of the hoisted rows is `2 x F`, and it is right
  for the op measured there, which declares no `seam_fold`. It is not the general
  figure.

**It is the whole of why §7.1's projected 1.13x came out at 1.16x**, which is the
most useful thing about it: the difference is `0.407` against `0.271` GiB of
fragment traffic on the recorded volume, exactly one `F = 0.136 GiB`, and it was
attributable only because every other column reproduced to the digit.

The check is still worth it and the arithmetic is the same shape as before: 0.14
GiB against the 34.5 removed, and it is checking the one property this crate is
arranged around. What changes is that the price should be quoted as *one extra
reduction and one extra pass over the fragment set*, so that an op whose `F` is
large can see it.

### 10.3 §8.9's open measurement is closed

§8.9 recorded that `tests/label_materialisation_cost.rs` "has not been re-run
against this implementation", because it reads a machine-local recording. It has
been. Same recording, same `D = 16`, same four lattices; the left-hand column is
the shipped ops and therefore moved when they were migrated, and every other
column reproduced to the digit, which is what makes the one that moved
attributable:

| blocks | in-plan, as shipped now | barrier | barrier + hoisted | merge outside the plan |
|---|---|---|---|---|
| 1 | 1.26 | 1.26 | 1.26 | 0.74 |
| 4 | 1.33 | 1.40 | 1.30 | 0.77 |
| 32 | 1.74 | 3.78 | 1.67 | 1.15 |
| 256 | **4.84** | 39.29 | 4.70 | 4.18 |
| | **1.16x** | 9.41x | 1.13x | 1.00 |

**106.07 GiB became 4.84, a factor of 21.9**, with 23 627 components agreeing at
every lattice in all four arms. The merge CPU re-timed on that machine is 0.08 /
0.12 / 3.00 / **48.86** s, and the shipped arm's entire run at 256 blocks is 4.45
s against the barrier-alone arm's 53.47.

### 10.4 §6.1 and §6.2 have fired

Both were expiry conditions on the decorate-versus-materialise recommendation and
both are now spent. §6.2's is the one that matters — *"if a barrier lands **and** a
barrier phase's reduction is allowed to run once, the gap closes to 1.13x"* — and
it closed to **1.16x**, measured, for §10.2's reason. Its conclusion stands
unchanged: at that gap materialising is the better default, because a
materialised volume is remapped once and a decorated one per reader.

§6.3 is untouched — Rule A is exactly where it was, and see §10.5. §6.4 is
untouched and still worth repeating: nothing here has been measured against
`ZarrEnvironment`.

### 10.5 Four things the migration found that this note did not predict

1. **"A barrier alone is worth about a third of the gap" is not a property of the
   mechanism.** §3.3 measured 2.7x of 25.4x on the recorded volume and §8.8
   measured 4.9x of 66.7x on the fixture, and both are quoted as though they
   describe the barrier. They describe a **ratio of `F` to the volume**. A barrier
   removes the pixel half and leaves the fragment half, so what it is worth is
   whatever fraction the pixels were: on `ops::fill` over `[16, 32, 32]` at 256
   blocks it is **1.56x of 91.3x**, because at that cut the fragment set is 175%
   of the label image — the regime §7.6 predicted and nothing had landed in. On
   `ops::detect` it is **1.00x**, exactly, at every lattice, because that op
   fetches no pixels at any halo. The number must be measured per op and per
   lattice and never quoted bare.
2. **The plan cannot tell a hoisted phase from a non-hoisted one.** `barrier` is
   recorded on `PhaseDecomposition`, hashed and carried over the wire, for §3.1's
   reason. `reduce` is not, and neither is a `FragmentInput`'s reach — both are
   read off the op by `check_phase_work` on every run, which is where a
   disagreement surfaces. So the two barriered shapes **fingerprint identically**.
   That is correct rather than a gap, since they compute the same answer and
   differ only in cost, but a fingerprint is not evidence about which of them ran.
3. **A wrong fragment reach is invisible once `gathers() == false`.** §4 says the
   cost of a reach declared too wide is reads. That does not cover a hoisted op:
   restoring the whole-lattice reach on one moves **no byte** and changes **no
   answer**, because nothing gathers it — but it takes the per-block neighbourhood
   above one element, which turns `SeamFold::Unordered`'s per-block order check
   back on, so every block applies twice. The currency is CPU and no counter shows
   it. The ops that hoist assert their own `inputs()` reach and `gathers()`
   directly, and count applications, for exactly this reason.
4. **"Barrier" now means two different things one call apart.**
   `reach::Reach::is_barrier` means *this reach spans a whole axis*, which is a
   statement about fusion; `FragmentOp::barrier` means *this phase waits for all of
   the phase below*, which is a statement about ordering. Every op migrated here
   used to be the first and is now the second and is the first no longer, so a
   reader meeting both in one file has every reason to conflate them. Renaming is
   a change to `src/reach.rs` and is not this note's; naming the collision is.

---

## 11. The completeness check, priced

*Written by the work that closed the three deferrals recorded in the session
log kept outside this repository — `clearmap-rs/forme2.md` §24, which is not
part of this crate and is named here only so the provenance is followable.
Nothing above is edited.*

### 11.1 The check ran once per input stream, and should run once per phase

`check_fragment_coverage` lists **every** output stream of the op it is handed, so
one call already covers all of a producing phase's streams. `reduce_phase` ran it
once per declared *input*, so a barrier joining two streams of one producer ran
the whole check twice and listed each of that producer's streams twice. The
duplicated cost was the product of two figures the caller sets — how many streams
the op joins, and how finely the volume is cut, since a listing returns one key
per block. It is now grouped by producing phase and runs once per phase.
`tests/fragment_coverage_listings.rs` measures it, with the liveness control
beside it: a hole in the *second* declared stream is still refused, which is what
separates deduplicating the check from deleting it.

### 11.2 The single-node repetition is kept, and here is what it costs

`execute_phases` runs the same check on a fragment phase's outputs when that
phase's last task completes, and every producer `reduce_phase` names is an
earlier phase — so in-process the check at the barrier is the second one on the
same stream. That repetition is **kept**, and the two ways to remove it are both
worse:

* **Move it to the barrier.** `execute_phases` checks every fragment phase,
  including those no barrier ever reduces over, and it checks at the phase that
  made the hole rather than at whatever runs next. Deferring it lets a doomed run
  continue through the phases in between and reports the failure somewhere else.
* **Let the caller say what is already verified.** That turns a guard against a
  plausible-wrong-answer into something a caller disables by getting one argument
  wrong — on the distributed path, where nothing else runs it at all.

The bound is what makes keeping it defensible rather than a shrug. The extra cost
is one listing per producing phase, `O(blocks)` keys and no bytes, against a
phase that irreducibly writes `blocks` fragments and reads at least `blocks` more.
The ratio is fixed: it does not move when a caller cuts more finely, which is the
question `PhaseCost`'s standing rule asks of any per-item reassurance.
`Stats::sidecar_listings` and `sidecar_keys_listed` report both figures and
`tests/fragment_stats.rs` pins them.

### 11.3 A fragment phase stays out of `ops_applied`, and the other question gets its own field

`ops_applied` and `blocks_visited` are structurally zero for a fragment-only plan,
and that is now a decision rather than a deferral. Making a fragment phase emit
`Event::OpApplied` would close it, and the event is the wrong shape: it carries a
chain **slot index** and the region the op was computed **over**, and
`ExecutionLog::recomputed_margin_voxels` sums those regions against what was kept
to measure a *chain's* halo redundancy — the figure that says whether a phase
split bought anything. A fragment op has no slot to name and no halo of that kind,
so emitting the event would put invented slot numbers in the exported log and move
a number whose whole purpose is a different measurement.

The conflation was the actual defect: *how many chain ops ran* and *how many
blocks did this plan touch* are two questions and only the first had a field.
`Stats::blocks_admitted` is the second, derived from `Event::TaskAdmitted`, which
every kind of phase emits. It is what the exported document's block table has
always counted, so a consumer comparing the two now has a field that matches it on
every plan instead of only on chain-only ones.

### 11.4 The hoisting check is available multi-node

`WorkerReport` gained `sidecar_reads` and `fragment_applications` beside
`fragments`, which counted only writes. Summed over a job's workers they are what
one node would have reported, so the in-process discriminator works across
processes: measured on a 16-block lattice with three workers, the in-plan arm
reads **256** fragments — `blocks²` exactly — and the hoisted arm **96**, with
both arms applying the op **32** times and writing identical volumes.
