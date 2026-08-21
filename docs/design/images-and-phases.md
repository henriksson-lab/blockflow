<!-- SPDX-License-Identifier: MIT -->

# Images and phases

**Status: executed.** `image` is the name in the code, in both crates, and
`level` survives only where it is a **different word** — an intensity level, a
frozen format's key, a pyramid or compression level. This note was the
specification for the rename, written first so that whoever did it was executing
a decision rather than making one, and nothing in `src/` was touched to write it.
It is now the record of a decision that has landed, corrected where it predicted
wrongly.

> **Updated twice, and both updates are worth reading before the rest.**
>
> **First, while the rename was still pending:** the part of this note that was
> not a rename landed on its own, pushed by G5 rather than by anybody executing
> this document — the three-kind enum existed, `Visibility` was derived from it,
> and the environment's discard refused a supplied input for exactly the reason
> the middle column below gives. Two things this note predicted were wrong and
> are corrected in place: what closing G5 would do to the numbering, and
> `n_images() == n_phases() + 1`.
>
> **Second: the rename has been executed**, in `src/` and `tests/` of both
> crates. Every row of the table at the end is done. The two rows that table
> listed as open now have **answers rather than intentions**, and they are
> recorded there and in "Three kinds of image": `Visibility` was **kept**, not
> removed, and `image_visibility` still returns it. This note's own vocabulary
> has been swept with the code; what has deliberately **not** been swept is the
> left column of the rename table, the collision counts, and every sentence
> below that is about the word `level` rather than about the thing.

**One thing about this directory, stated once so nobody rediscovers it.**
`docs/design/BLOCK_OPS.md` is cited **45 times across 28 files** under `src/` and
`tests/` and **does not exist in the tree**. This file is the first thing in
`docs/design/`. Writing `BLOCK_OPS.md` is somebody's job and it is not this
note's; the dangling citations are recorded here as a known state, not repaired,
and nothing below cites that file as though it were readable.

> *Later, by another worker.* The `README.md` split that produced
> [`writing-an-op.md`](writing-an-op.md) checked whether the material it was
> moving was what those 45 citations want. **It is not.** The moved sections
> succeed `BLOCK_OPS.md` on the diamond — what it recorded as "`Chain` has no
> fan-in" now ships as `Chain::Parallel` — and reproduce exactly one of its
> section titles, *"A full-reach op is a planning barrier"*. The planning,
> costing and execution sections the other citations quote (§"The planning
> problem is NP-hard", §"Emit a dependency graph, do not inline the loops",
> §"Workflow -> Planner -> Executor", §"Simulating strategies", §"Estimating
> from noisy samples", §"Images are a DAG", §"Block size may differ per phase",
> §"Step 1/2/3") are in no file in this directory. The gap stands; it has now
> been looked at rather than only recorded.

---

## The decision

**`level` becomes `image`. `phase` is kept.**

Both words were put to the same test — *does the word imply an ordering that the
structure does not have?* — and they answer differently.

**`phase` survives it.** A phase is a unit of work over the whole block grid.
"These two phases run in parallel" reads correctly: two units of work, both
happening. The word carries a sense of *stage in a process* without asserting
that stage 2 is downstream of stage 1, and it is already the crate's word in
`Phase`, `PhaseDecomposition`, `PhaseWork`, `SchedulePriority::PhaseMajor`. It
stays.

**`level` fails it twice.**

1. **It implies rank.** "Level 3" reads as above "level 2". The sentence that
   breaks is *"parallel levels"* — it reads as a category error, and it is the
   sentence a multi-armed plan needs to be able to say.
2. **In an image-processing library the word is already taken.** `level` means
   resolution, or a pyramid level, to every reader who arrives from that
   direction. This is not hypothetical inside this crate: `ops::normalise`
   already has a `LevelCorrectionOp`, whose doc comment reads *"Estimate a
   slowly varying level on a sample lattice and remove it"* — a level in the
   **intensity** sense, 30 mentions, in a file that has nothing to do with plan
   levels. The crate is already using one word for two unrelated things.

### The collision counts

The alternatives were scored by how much existing text a rename would collide
with. Counts are **lines matching the word as a whole word, case-insensitive,
over `src/` only**, measured on the working tree at the time of writing:

| candidate | lines | verdict |
|---|---|---|
| `image` | **76** | **chosen.** Almost unused, and it is what the thing is |
| `node` | 165 | taken by the *distributed* sense — a machine |
| `buffer` | 440 | taken; and a buffer is a `BlockBuf`, a per-block thing |
| `slot` | 473 | taken by chain slots, the op positions inside a phase |
| `task` | 572 | taken by `TaskGraph` — one block of one phase |
| `array` | 906 | too general, and already the loose word for any of them |
| `level` | 1379 | the incumbent |
| `volume` | 2585 | taken: `volume` is the **extent**, `[usize; 3]` |
| `phase` | 2010 | the other half of the pair, and kept |

*Discrepancy recorded rather than smoothed:* the counts I was given to check
against were `image` 77, `volume` 2784, `array` 1008, `node` 177, `slot` 563,
`task` 625, `buffer` 456, `level` 1566. Mine are the same ordering and the same
decision, systematically 5–15% lower. Two workers are editing `src/` while this
was written, so the tree has moved; I could not reproduce the exact figures under
any counting convention I tried (occurrences vs. lines, `src/` vs. `src/`+`tests/`,
whole-word vs. substring). The gap does not touch the conclusion: `image` at 76
against every alternative in the hundreds or thousands.

---

## The structure is a DAG, and the number is an address

A plan is a **directed acyclic graph over stored arrays**. Linearising it into
numbered slots is an *encoding* of that graph, not a description of it — in the
same spirit as arena allocation in Rust, where an index into the arena is an
**address** and says nothing about order.

The crate already knows this and says so in two places:

* `src/graph.rs`, on `Task::source_deps`: *"**Explicit rather than inferred from
  the phase order.** A source image is written by a phase that has already run,
  so one can argue that the transitive dependency is there anyway — but only
  while every image between the two is on the same lattice, and a phase that
  resamples breaks that argument without breaking any test."*
* `src/graph.rs`, on the test that pins it: *"An image read by a source leaf is an
  edge in the graph, to the phase that wrote it — **not an assumption that the
  phase order made it available**."*

(The sentence I was handed is a splice of those two: they are 380 lines apart, on
a field doc and a test doc respectively. Both are verbatim.)

`src/assemble.rs` calls the structure by its name in passing — *"a chain in the
middle of an image DAG can be priced without the planner learning about the
DAG"*.

The worked example is `binarize` in the sibling application: a fan-out into
**seven arms** over 12 phases and 13 images, arms branching from a shared prefix
and rejoining at a sink. Arms read arrays named by an **earlier** phase through
`Chain::Source`, not only the array their immediate predecessor wrote. The
numbering there is an arena index and nothing else.

*Unverified:* I did not confirm the stronger claim that **no two** of the seven
arms sit in a line — the recorded chain has serial segments inside arms, and the
per-arm topology is in the sibling application, not here.

### The index should be a newtype — and half of it already is

`src/assemble.rs` already defines the newtype and argues for it — quoted here
as it reads now, which is the rename applied to it:

> `Phase` and `ImageId` are both a `usize` and they are deliberately not
> interchangeable. A phase index and an image number are different quantities
> that are numerically close — phase `p` writes image `p + 1` — so the mistake
> worth making impossible is not "a number out of range" but "the *other*
> number, which is also in range".

That is the right argument and it is already made. What is missing is **reach**:
the newtype exists at the builder boundary only. `Chain::source` takes an
`impl Into<ImageId>` and then immediately stores the index as a bare `usize`,
and every plan-side API downstream of it — `Chain::Source { image: usize }`,
`SourceInput::image`, `PhaseDecomposition::source_images`, `keep_images`,
`images_dead_after`, `image_visibility`, `discard_image_after` — is raw `usize`
again. The rename is the moment to push `ImageId` inward, so the type says
*address* everywhere and not just at the door.

> **Half done, and the half that is not done has a reason this note did not
> have.** The **declared** side is typed all the way in: `Chain::Source { image:
> ImageId }`, `Chain::source(impl Into<ImageId>, …)`, `SourceInput::image` and
> `Hints::keep_images` all carry `ImageId`. The **recorded** side is still a bare
> `usize` — `PhaseDecomposition::source_images`, and with it `volume_at`,
> `readers_of_image`, `images_dead_after`, `image_visibility`, `image_kind` and
> `supplied_input_images`.
>
> `assemble.rs` states why at `is_supplied_image`: `source_images` and
> `supplied_dtypes` are **serialised to the distributed wire and hashed into
> `Decomposition::fingerprint`**, so their element type is part of a *format*.
> Pushing the newtype through them is a format change, not a rename — which is
> also why `is_supplied_image` and `describe_image` exist as free functions over
> `usize` beside the `ImageId` methods that answer the same questions.

---

## Three kinds of image

This is the part of the decision with the longest life, because it is not a
rename: the current enum **loses information** and the split recovers it.

`src/decomposition.rs` as it read when this note was written:

```rust
/// Whether a level survives the run.
pub enum Visibility {
    /// Level 0 and the workflow output. Somebody outside the run reads these,
    /// so they exist when it ends.
    Published,
    /// Written by one phase, read by exactly one phase, then dead.
    Internal,
}
```

**Checked, and it says what it was reported to say.** `Published`'s doc comment
does lump the input in with the output — *"Level 0 **and** the workflow output"* —
on the single shared ground that somebody outside the run reads both. It gets the
*behaviour* for image 0 right (never freed) for a reason that is not the real one.

A second thing worth noticing while the file is open: `Internal`'s *"read by
exactly one phase"* has quietly gone stale. `Decomposition::readers_of_image` was
generalised when source leaves arrived and its own doc says so — *"A source leaf
is a second reader, so the general statement is the one the design record asks
for: an image dies after its last reader."* The enum did not follow.

> **The enum has since followed**, in the same pass as the rename.
> `Visibility::Internal` now reads *"Written by one phase, read by its readers,
> then dead"*, and names the plural: *"a source leaf is a second reader, so the
> general statement is the one `readers_of_image` makes — an image dies after
> its **last** reader."* This paragraph is kept because the register cites it.

The three-way split, as I would finally state it — with the last column split in
two, which is the one change I would make to the table I was given:

| | produced by a phase? | recomputable by this run? | must exist when the run ends? |
|---|---|---|---|
| **input image** | no — given to the run | **never, at any price** | n/a — it existed before the run and is not the run's to free |
| **intermediate image** | yes | **yes** | no |
| **output image** | yes | yes | **yes** |

Why split the last column: "recomputable" and "must be there at the end" are two
different facts, and the original phrasing (*"yes, but must exist at the end"*)
welds them. An output image is as recomputable as an intermediate — the
difference is a **materialisation obligation**, not a property of the
computation. Keeping them apart is what lets a scheduler say "I may drop this and
rebuild it later, but I must rebuild it before I finish", which is a real and
useful state and is unsayable in one column.

**The middle column is the point.** An intermediate image may be dropped and
recomputed — the classic memory-for-recomputation trade. An input image may not
be recomputed at any price: there is no phase that produces it. Today the
scheduler cannot tell those apart, because `Published` covers both ends and
`image_visibility` is `image == 0 || image + 1 >= n_images()`. Any future policy
that trades residency for recomputation has to be able to ask this question, and
right now there is nothing to ask.

> **Landed, in substance, and the middle column is why.** `ImageKind { Input,
> Intermediate, Output }` is in `src/decomposition.rs` with this table in its own
> doc comment, and `Visibility` is now **derived** from it —
> `Input | Output => Published`, `Intermediate => Internal` — so nothing that
> reads visibility today moved.
>
> **It did not land because somebody executed this note.** It landed because
> `Environment::discard_image` had to refuse a supplied input when G5 arrived,
> and the refusal needed a question to ask. The comment at the refusal is this
> paragraph's argument, arrived at from the other end: *an intermediate may be
> dropped and rebuilt by re-running the phase that wrote it; an input cannot be
> rebuilt at any price, because no phase produces it.* A policy question this
> note framed turned out to be a **correctness** question first — freeing an
> input is an image that is gone and unrecoverable — which is a stronger reason
> for the split than the one argued here, and it is the reason that got it
> built.
>
> **The spelling has since landed and the removal has not — and the removal is
> now a decision rather than a leftover.** The enum is `ImageKind`, beside
> `Visibility` rather than in place of it, and `image_visibility` still returns
> `Visibility`. Three things settled it, and they are recorded here because the
> earlier intent was the opposite:
>
> * **Live reads in the plural**, not the couple of call sites the removal was
>   priced against. Folding `Visibility` away makes every one of them restate
>   `Input | Output` for itself. *Discrepancy recorded rather than smoothed:* the
>   enum's own doc comment says *"nine call sites across this crate and its
>   consumer"*; counting statements that actually read a `Visibility` I find
>   **eight** in this crate — `zarr_env.rs`, `strategy.rs`, and six assertions
>   across `tests/image_lifetime.rs` and `tests/supplied_inputs.rs` — and **none**
>   in the consumer. Nothing here turns on which figure is right.
> * **It answers a different question.** `ImageKind` says what an image *is*;
>   `Visibility` says what the run must *leave behind*. That is exactly the split
>   between this table's own "recomputable by this run?" and "must exist when the
>   run ends?" columns — the split this note argued for, arriving as two types
>   instead of one column.
> * **It is derived**, `Input | Output => Published` and `Intermediate =>
>   Internal`, so the two cannot disagree. A kept enum that could drift would be
>   a different proposition; this one is the same fact asked more narrowly.
>
> See the rename table.

### G5 falls out

`docs/ops-survey/README.md`'s gap register defines **G5** in terms of the
numbering: *"image numbering in which images `0..k` are inputs and phase `p`
writes image `k + p`, plus constructors taking a list."* Under the three kinds,
that clause disappears. There is no convention about which indices are
privileged, because *input image* is a **kind** and not an index range. A
multi-root DAG needs `k` input images and no special rule at all. The rest of G5
— constructors taking a list, and the environment work behind it — is untouched;
what goes away is the numbering convention, which was the awkward half.

G5's own note that *"`Chain::Source`, `SourceInput`, `check_source_images`, image
lifetimes and the byte accounting already work above image 0"* stays true and
becomes the reason this is cheap.

> **G5 has closed, and this section was right about the conclusion and wrong
> about one step.** Both halves, kept apart because they came out differently.
>
> **Right: the numbering convention went away, and it went away because it could
> not be built.** `0..k` failed twice over. The executor addresses images
> positionally — `env.read(task.phase, …)`, `env.write(task.phase + 1, …)`, at
> some fifteen sites in `strategy.rs` — so `k + p` cannot be reached without
> rewriting it. And independently, a caller needs an input's address *before* it
> builds the ops that read it, while the phase count is not known until
> `finish`, so `0..k` is unreachable from the builder as well. The register's
> §10 records both objections.
>
> **Wrong: "there is no convention about which indices are privileged".** There
> is one, and it is the opposite of the one this note was arguing against: a
> **disjoint high range**, `ImageId::SUPPLIED_BASE = usize::MAX / 2 + 1`, with
> `ImageId::supplied(i)` addressing the `i`th array handed to the run. It was
> chosen for the property this note cares about most — **adding an input
> renumbers nothing**, where `0..k` would have silently re-pointed every
> `Chain::source(3)` in the program — and for one this note could not have
> known: the address is a constant, available before a single phase exists,
> which is the order a builder runs in.
>
> **The two are not in conflict, and the distinction is this note's own.** The
> *range* says where an array lives; the *kind* says what it is. `ImageKind` is
> what says "input", and it is the kind, not the address, that answers the
> question this section actually asked — *may this be freed, and could anything
> rebuild it*. A multi-root DAG needs `k` input images and no special rule about
> what they mean; it turned out to need one about where they sit.

---

## The open question this note exists to frame

**The optimiser may not reorder, and today that is a correctness guarantee rather
than a policy.** Both quotations are where they were said to be:

* `src/assemble.rs:291`, on `PlanBuilder` — *"Every method appends. There is no
  way to insert, reorder or remove a phase, and that is not an omission: a
  `Phase` handed out earlier records an index, and an insertion would silently
  renumber it."*
* `src/assemble.rs:497` and `src/strategy.rs:307`/`:790` — *"A decomposition may
  partition; it may never reorder or drop an op."* It is an enforced refusal,
  checked against `slot_order()` in both places and pinned by a test at
  `src/strategy.rs:3269`.

So **block size and fusion are searched; the topological order is not.** The
order of the DAG is whatever sequence of builder calls the caller emitted, and no
strategy may touch it.

Two measured consequences, both from this project's own work on the sibling
application:

**1. The implicit predecessor hides a real cost.** `PlanBuilder::pixels` hands a
phase *the array below it* — phase `p` reads image `p`, writes image `p + 1` —
and any other array it reads must be named explicitly through `Chain::Source`.
That makes it easy to build a phase far from the array it actually reads, paying
for an input it never touches. One phase in the sibling application's `binarize`
was built after the rest of the program while reading an array written much
earlier, and the array it was charged for went unused: **15.7% of every voxel the
stage read**, 11 856 432 wasted voxels at one lattice and 88 842 624 at another.
Three instances of the same defect, all closed by the same move — *build a phase
next to the phase that wrote the array it reads, and spend the input it is
charged for anyway* — came to **19.7% / 22.0% / 22.2%** of whole-stage read
amplification at `[64,64,12]`, `[32,32,12]` and `[32,32,6]`, with every count
byte-identical. **The fix was pure reordering, done by hand in the builder.
Nothing in the framework could have found it.** *(The exact distance — "six
phases away" — I could not confirm; what is recorded is the move and the
percentages.)*

**2. Peak residency is a function of the order.** At tile scale the same stage
prices **67.8 GiB of images alive at one phase** (measured `VmHWM` 124.9 GiB —
plan-side residency is a lower bound, because ops' working buffers belong to no
image), and it is the one stage that cannot run. That number is not a property of
the DAG; it is a property of the *linearisation* of it. Independent arms
interleaved hold far more live images than arms completed one at a time. This is
a **pebbling problem**, and today it is unreachable by construction — no strategy
may propose a different order, so no strategy may propose a cheaper one.

*A note added later, since it is the same number.* The 67.8 GiB was computed by
hand, and so is every other figure of its kind: the image-lifetime walk that
produces it is open-coded in three consumer test files, while
`readers_of_image`, `images_dead_after`, `image_visibility` and now `image_kind`
are all public and all already agree with each other. The ops survey's register
carries that as **G16** — `Decomposition::peak_image_bytes()`, derived rather
than stored, on the same argument as everything else derived here. It is not a
prerequisite for the search below; it is the objective's first term, and there
is currently no way to ask the plan for it.

### What the open item actually is

**A search over topological orders, with peak live-image residency in the
objective, alongside the block size and fusion already searched — and a contract
change to permit it.** Not designed here. Named, with what it would need:

* **A DAG the planner can see.** `Strategy::decompose` takes a `Workflow` —
  *"one chain, one input, one output, a linear pipeline"* — so a multi-image
  stage has nothing to hand it and today groups its own phases by hand. `assemble`
  already records this as the reason *"every performance figure this crate has for
  such a stage is measured on a partition nobody chose."*
* **The three kinds**, so the objective can distinguish an image that may be
  dropped and rebuilt from one that may not. **Available now** — `image_kind`
  answers it — and this is the one prerequisite on the list that has stopped
  being one.
* **`ImageId` as an address**, so a reordering pass renumbers slots without
  renumbering meaning — which is precisely the failure `PlanBuilder`'s
  append-only rule was defending against.
* **A relaxed contract**, replacing "may never reorder" with a checked condition.

**What would have to be proven safe:** that *an op's output does not depend on
when it ran* — that the value written into an image is a function of the images
it reads and its parameters, and of nothing else. That is exactly what "may never
reorder or drop an op" guarantees today **by fiat**: it is cheap and total, and it
is why the check exists in two places and is pinned by a test. Anything that
relaxes it has to replace fiat with proof, and the honest form of the proof is a
property an op **declares** and the plan checks — not one a planner assumes.
Ops that would fail it are the ones with state outside their image arguments:
`iterate`'s running operand, anything reading a fragment stream by phase index,
anything whose element type or lattice is inferred from a neighbour. The refusal
must name them; a reordering that silently changes a voxel is the one outcome not
available.

---

## The viewer: scrolling the DAG

A linearised DAG **confuses a reader looking at data across images**. A numbered
list says "image 7", and the question a reader has is "which arm is that on, and
what wrote it" — which the number does not answer, and, worse, actively
mis-answers by suggesting image 7 is downstream of image 6.

The crate has a GUI (`src/gui/`, and the `block_gui` and `block_animate`
binaries), so this is a real consumer and not a hypothetical. `gui/mod.rs`
already treats replay and live as one thing because splitting them made two
programs that *"eventually disagree about what a block's state even means"*; the
same argument applies to a plan view that renders a DAG as a list. **A viewer
needs a way to scroll the DAG** — to move along edges rather than along indices.
Recorded here as a requirement the rename makes statable; not designed.

---

## The rename, as a list

**Executed, in `src/` and `tests/` of both crates.** The left column is kept
verbatim — it is what the names *were*, and it is what makes this table readable
as a map. Every row is done. The two rows this table listed as open are marked,
and both are now answers rather than intentions.

| before | after |
|---|---|
| `assemble::Level` | `ImageId` — and pushed inward past the builder boundary |
| `Chain::Source { level, dtype }` | `Chain::Source { image, dtype }` |
| `Chain::source(level, dtype)` | `Chain::source(image, dtype)` |
| `SourceInput::level` | `SourceInput::image` |
| `PhaseDecomposition::source_levels` | `source_images` |
| `with_source_levels` / `declare_source_levels` / `check_source_levels` | `..._source_images` |
| `Hints::keep_levels` | `keep_images` |
| `Environment::discard_level` / `discard_level_after` | `discard_image` / `discard_image_after` |
| `Decomposition::levels_dead_after(phase)` | `images_dead_after(phase)` |
| `Decomposition::readers_of_level` | `readers_of_image` |
| `Decomposition::n_levels` / `volume_at(level)` | `n_images` / `volume_at(image)` — **and `n_supplied_inputs` with them**, which is the other half of the count and did not exist when this table was written |
| `Decomposition::level_visibility` | `image_visibility` — ~~and its return type is the three-kind enum~~. **The name landed; the return type did not, and that half of the row is now withdrawn rather than pending.** `image_visibility` still returns `Visibility`, `image_kind` is the accessor that returns the three kinds, and both are public. See the row below |
| ~~`Visibility::{Published, Internal}`~~ → **`LevelKind::{Input, Intermediate, Output}`** | `ImageKind` — **done as a rename, and the removal it implied is withdrawn.** The enum is `decomposition::ImageKind`, *beside* `Visibility` rather than replacing it, with `Visibility` derived from it (`Input \| Output => Published`, `Intermediate => Internal`) so nothing that reads visibility moved. `Visibility` is **kept**: live reads in the plural rather than the couple of call sites the removal was priced against, a genuinely different question — what an image *is* against what the run must *leave behind* — and derivation, which is what stops the two from disagreeing. The argument is in "Three kinds of image" above, where the strikethrough in this row's left column was first proposed |
| `Level::SUPPLIED_BASE` / `Level::supplied(i)` / `Level::supplied_index` / `is_supplied_level` / `describe_level` | `ImageId::…` and `is_supplied_image` / `describe_image` — **added to this table after it was written**, by G5. `describe_image` is the one with a reason to exist beyond the word: a supplied address prints as a nineteen-digit number nobody typed, so every diagnostic goes through it. `is_supplied_image` and `describe_image` stayed **free functions over `usize`** rather than becoming `ImageId` methods only, because the recorded half of the plan is a serialised format and cannot carry the newtype |
| `Decomposition::supplied_input_levels` / `n_supplied_inputs` / `level_kind` | `supplied_input_images` / `n_supplied_inputs` / `image_kind` — the middle one keeps its name, because "input" is already the word the three kinds use |
| `Phase::level()` / `Phase::writes()` | `Phase::image()` / `Phase::writes()` |
| `LevelStore`, `level_shape`, `level_path`, `level_dtype`, `allocated_levels`, `resident_levels` | `Image…` / `image_…` |

**Do not rename**, and this is the whole reason the word had to go:
`ops::normalise::LevelCorrectionOp` and its vocabulary. That `level` is an
intensity level and always was.

> **The do-not-rename list turned out to have four entries, not one, and this is
> the part of the specification a later reader is most likely to break.** Every
> one of them is `level` in a *different* sense, and three of them are not even a
> naming question:
>
> * **Intensity.** `LevelCorrectionOp` above, and `Threshold::level` — a public
>   `f64` field. A blanket replace destroyed it and the compiler caught it at
>   three consumer call sites, which is why the sweep was redone from a backup
>   after a line-by-line audit rather than patched forward. Prose about a
>   threshold's level, an adaptive threshold's estimated level or a plateau's
>   value stays.
> * **Formats, frozen by being formats.** The order-log document is a **versioned
>   schema** — `"version": 1`, bumped only on a breaking change — with a recording
>   checked in at `tools/sample_block_progress.json`; its `"level"` key and its
>   `"source": "level 0"` values are still `level`. So are the on-disk Zarr node
>   names `root/level<n>`, the `level-{n}.f64` files, and the distributed wire's
>   `"source_levels"` field, which `wire.rs` reads in documents written before a
>   feature existed. **Renaming any of these is a version bump, not a rename.**
> * **Other senses:** the deflate/gzip compression level, the region tree's
>   `st_level`, a graph's raw/cleaned/reduced levels and their recorded fixture
>   paths, `RadiusLevel`, and the tie-group "shells" of an offset sequence.
> * **English.** "-level" and "at the X level" — volume-level, byte-level,
>   caller-level, top-level, levels of detail. Left alone everywhere.

~~`n_images() == n_phases() + 1` becomes false the moment G5 lands, so the rename
is the right time to stop deriving the count and start storing it.~~

> **Corrected — G5 landed and it did not become false.** `n_images()` still
> returns `phases.len() + 1`, and what changed is what it *means*: **the number
> of images the run writes into**. `n_supplied_inputs()` is the other half, and
> the two are deliberately added by nobody — every caller of `n_images` wants
> what the plan fills in (a chunk list, an image table, a bound on an image a
> phase may name), and none of them wants a total that includes arrays the run
> was handed. The supplied inputs are addressed in a disjoint high range, so they
> are not in `0..n_images()` and no loop over it has to learn about them.
>
> **So the advice reverses.** The count should stay **derived**, on the same
> argument this note makes for `image_visibility` being derived: a stored count
> is a field that can disagree with the arithmetic, and this one now has two
> arithmetics to disagree with. What the rename should carry across is the
> *doc comment*, because the identity is no longer self-explanatory — the trap
> is a reader who takes `n_images()` for "how many arrays this run touches",
> which it is not and has not been since G5.
>
> **Carried across.** `n_images()`'s doc comment now opens *"This is not the
> number of arrays a run touches"* and names `n_supplied_inputs` as the other
> half, added by nobody, deliberately.

---

## What is unverified

* The exact collision counts differ from the ones I was checking against by
  5–15%, in the same direction for every word (see above). The tree is moving.
* "A phase built six phases away from the array it reads" — the 15.7% and the
  19.7/22.0/22.2% figures are recorded and confirmed; the distance is not.
* "Seven arms, no two in a line" — seven arms is recorded; the no-two-in-a-line
  topology is not, and the shared prefix in the recorded chain argues against it.
* Every quotation above was read out of the working tree while two workers were
  editing `src/env.rs`, `src/voxels.rs`, `src/assemble.rs`, `src/strategy.rs` and
  `src/decomposition.rs`. Line numbers may have moved; the sentences were
  verbatim when read.
