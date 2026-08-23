# Writing an op

*Split out of `README.md`. Companion files:
[`dimensions-and-modules.md`](dimensions-and-modules.md),
[`executing-a-run.md`](executing-a-run.md),
[`images-and-phases.md`](images-and-phases.md).*

> **Where this came from, and how far to trust it.** These sections were split
> out of the crate's `README.md`, where they sat below a line reading
> *"Below are ramblings for the LLM. info might not be accurate"*. **That caveat
> travels with the text.** Moving prose into `docs/` does not promote it to a
> specification, and nothing here has been rewritten to sound more settled than
> it was — the sections are the README's own words, in the README's own order,
> with the README's own emphasis.
>
> What *has* happened is a spot-check of the claims that other files depend on.
> [What was checked](#what-was-checked) at the foot of this file lists, by name,
> what was verified against the code, what was corrected, and what is still
> unverified. **A claim not listed there was not checked.** Treat an unchecked
> paragraph as a design intention that was written down, not as a description of
> what the code does today.

> **This file is not `BLOCK_OPS.md`.** Module headers throughout `src/` and
> `tests/` cite `docs/design/BLOCK_OPS.md`, which does not exist in this
> repository. That document, judged by what its citations quote, is an earlier
> and much broader planning-and-execution design — §"The planning problem is
> NP-hard", §"Emit a dependency graph, do not inline the loops",
> §"Workflow -> Planner -> Executor", §"Simulating strategies",
> §"Estimating from noisy samples", §"Images are a DAG", §"Block size may differ
> per phase", §"Step 1/2/3", §"`Chain` has no fan-in".
> This file reproduces exactly one of those section titles and *succeeds* the
> document on one more: what `BLOCK_OPS.md` recorded as "`Chain` has no fan-in"
> is described below as a shipped `Chain::Parallel`. **Reading this file closes
> 1 of the 45 citations outright and answers about 8 of them substantively; the
> other 37 are untouched.** See
> [The 45 dangling citations](#the-45-dangling-citations).


## Branching: three shapes, and why `max` is not enough

A `Chain` is `Sequence`, `Alternative` and `Parallel`.

```rust
Chain::sequence(vec![
    Chain::op(median),
    Chain::parallel(
        vec![arm_a, arm_b],                        // both run, same input
        Box::new(LogicCombine::new("or", Logic::Or)),
    )?,
])
```

`Alternative` is a choice — reaches take the **max**, one branch runs.
`Parallel` is a fan-in — reaches take the **max and the combine's adds**, and
**every** branch runs. Those two fold reach identically, which is not a
coincidence and is worth stating plainly, because it is how a real diamond was
once modelled as an alternation and passed 903 reach comparisons: *the max is
the correct budget whether one branch runs or all of them.* Reach cannot tell
them apart.

What tells them apart is what executes and what is produced, and there the two
are opposites:

| fold | `Alternative` | `Parallel` |
|---|---|---|
| `reach` | max | max, plus the combine's |
| `apply` | the live branch | every branch, then the combine |
| `side_outputs` | the live branch's | **the union** |
| `cost_per_voxel` | max | sum, plus the combine's |
| `constant_maps_to` | the live branch's | every branch's, then the combine's |

The `side_outputs` row is the one that cannot go wrong quietly. Over-declaring
an *output* is not safe the way over-declaring a *reach* is: a reach that is too
large costs reads, an output that is not produced is a hole that fails the
coverage guard. So an alternation declares only what its live branch writes, and
a fan-in declares every branch's — and structurally it has no choice, since the
variant carries no "which branch" to consult.

Two things a `Parallel` deliberately does not do:

* **It does not become several phases.** It is one indivisible slot. A phase
  reads one image and writes one image, so a cut between the branches and the
  combine would need an image per branch and a phase with several inputs, neither
  of which a `Decomposition` can state. Branch results are transient buffers
  inside one task — allocated where a `Sequence`'s intermediates are — so they
  add no task, no DAG edge and no materialisation.
* **It is not a merge.** A merge is a reduction over blocks, which a full reach
  already expresses; see the bullet under *Output that is not an image*.

`Combine` is a second trait rather than a wider `BlockOp` because the arity is
the difference: every question a combine answers — which element types it takes,
what shape it produces, what a constant folds to — is a question about a *list*.
Branch results need not agree with each other on element type; they must be
acceptable to the combine, which is checked when the plan is made.

### Two arrays: a branch that reads instead of computing

A phase reads the image it is handed. An operation needing a *second* array —
measuring one array against another, masking, seeding a reconstruction from
somewhere other than its own mask — used to get it by holding it whole
(`CombineOp`), which is one full copy of the array resident for the length of
the run.

`Chain::Source` is a leaf that reads a stored image instead of computing one,
so the second arm of a fan-in can be an array in storage:

```rust
Chain::parallel(
    vec![computed_arm, Chain::source(image, Dtype::F64)],
    Box::new(LogicCombine::new("xor", Logic::Xor)),
)?
```

It reuses the fan-in machinery entirely: one buffer per branch, joined by the
combine, with every fold above applying unchanged. Three things are new, and
none of them is assumed:

* **reach 0**, exactly. It reads the block's own read extent and nothing
  around it, so it never widens the halo of the arm beside it.
* **the image is in the plan.** Which image an arm reads changes voxels, so it
  is recorded in `PhaseDecomposition::source_images`, fingerprinted, and sent
  over the wire. `check_source_images` compares it against the chain and
  refuses, *by name and when the plan is made*, an image that does not exist, a
  forward reference to one a later phase writes, one on a different lattice,
  and one whose element type is not what the leaf declared.
* **an image dies after its last reader.** An image with a second reader is not
  freed when the first one finishes. `Decomposition::readers_of_image` is the
  refcount; with no source leaf it answers one phase and the old rule falls out
  of the new one unchanged.

Image 0 is the case of this that always existed: an image with no producing
phase. `Chain::source(0, dtype)` says so explicitly — and it is the one form
that is valid under *every* partition, because image 0 is below every phase
whatever the planner does with the chain. A leaf naming an intermediate names an
image number, so it constrains where the phase boundaries may fall; the shipped
planners do not yet place a boundary to satisfy one, they are refused by
`check_source_images` if they do not.

## Reach: what an op reads, and in what units

`BlockOp::reach(axis, volume_len) -> usize` is the statement most ops want and
it has not changed: symmetric, the same for every block, in the phase's own
voxels. It is required and has no default, because it is the one number a silent
zero would turn into a complete, well-formed, wrong volume.

`BlockOp::reach_spec(volume) -> Reach` is the full statement, defaulted to
lifting the above, so an op that has nothing more to say says nothing more. Four
things it can say that a triple cannot, each of which was costing something
measurable:

| form | says |
|---|---|
| `Bounded { lo, hi }` | asymmetric — a one-sided dependency declared on both sides fetched **3.27x** what was needed |
| `PerBlock(table)` | one `(lo, hi)` per block index along the axis, for a lattice whose voxel footprint differs per block |
| `All` | the whole axis, so a **planning barrier is a type** rather than a comparison somebody has to remember to make |
| `Space` | which volume the numbers are measured against, in what unit, in which axis order |

`Space` is the one that caused most of the loss: the same dependency is `2` in a
lattice's index space and `255` in voxels. It carries a **frame** (this phase's
volume, or the image below's — which decides whether a read clamped at the
phase's own edge may be trusted), a **unit** (voxels, whole blocks of this
phase's lattice, or steps of the image below's lattice) and an **axis order**.
Conversion happens in `PhaseDecomposition::derive`, the first place a grid
exists; before that a reach stays symbolic, because the planner is comparing
candidate grids and a reach that changed with the grid could not be compared
against anything.

The same type is the **halo**, because a halo is a granted reach and the guard
this crate is built on is the comparison between the two. That the halo may
differ per block and per side is what lets an op mandate an input extent *and*
reach: the cores are cut `extent - (lo + hi)` wide and the window slides inward
at the volume's ends rather than being clipped, so every block is handed the
extent it asked for (`Reach::window`, `BlockConstraint::lattice`).

A symmetric triple converts into a `Reach`, compares equal to one, **hashes as
one** and is the same three numbers on the wire, so a plan that says nothing new
is the plan it always was — fingerprint included.

## Slicing: whether the answer survives a cut inside one block

`BlockOp::slicing() -> Slicing` is the second declaration and it is **not**
derivable from the first. It is defaulted to `Slicing::UNDECLARED`, which is
today's behaviour exactly — one task per block, on one thread — and an op author
who says nothing loses a speedup that was never there. `reach` has no default
because a forgotten zero is a wrong answer; this one costs performance, and a
default is affordable on the second where it is not on the first.

**A reach says what an op reads; it does not say the answer is a function of
what was read.** That sentence is the whole content of the declaration, and the
demonstration is in the crate: `ops::sliding` computes the *same statistic* as
`ops::rank`'s median, with the same reach and the same output shape, by carrying
a histogram along the scan — so where the scan began is in the answer, and a cut
moves where the scan began. No signal available to the framework separates the
two. **Nothing here may derive a slicing; it may only read one**, and that
prohibition is the condition on which the default being safe rests at all.

Declare `Slicing::Stencil` when all three hold:

1. the output at `v` is a function of the input within `reach_spec` of `v` and
   of nothing else — not the block's extent, not an accumulator over it, not an
   identifier handed out in traversal order;
2. the output lattice is the input lattice, so a slab's core is the same index
   range in both buffers;
3. the answer is **bit-identical** whether or not the block was cut, which for a
   kernel accumulating in floating point means no sum is reassociated.

**A `Combine` declares separately, and a `Parallel` node is only as sliceable as
its narrowest part.** Three declared arms under a sink that says nothing is a
refused node, and the arms cannot speak for it. This is the shape that kept
every fan-in in this crate unsliceable long after its filters were declared.

**The bar is not the declaration.** An op is declared when somebody has put it in
`tests/intra_block_slicing.rs` and watched bit-identity hold uncut against cut at
every thread count, on a fixture that can *see its halo* — the file's three-way
perturbation probe is there because a fixture whose fold saturated once left a
test green under a halo-of-zero mutant. The bar is stated per element type and
refuses the rest **by name** rather than widening to `f64`, so an op writing a
type it has no arm for is a message and not a lossy comparison; the arm is the
edit, never the relaxation. See `docs/design/intra-block.md` for the measurement,
what the cut costs, and when a planner asks for one.

## Output that is not an image

Some steps produce, per block, a **fragment** that a later global step merges —
incident lists, component boundaries, per-block displacement estimates a
least-squares solve consumes. None of those is a pixel region, so the executor
had nowhere to put them and they were held in memory, which is what pins such a
stage to one node.

`sidecar` is that place. It is keyed by the decomposition's own unit and takes
bytes:

```rust
env.declare_sidecar("fragments", Lifecycle::DeleteOnExit)?;   // no default
env.write_sidecar("fragments", phase, block, &my_bytes)?;     // per block
...
for (key, bytes) in env.sidecar_fragments("fragments")? { /* merge */ }
let removed = env.discard_sidecars()?;                        // says what went
```

Three things it deliberately does not do:

* **It imposes no format.** Not serde, not JSON. The writer knows what it wrote
  and the merge knows what it reads; supporting *bytes* is what makes supporting
  arbitrary types possible at all.
* **It is not an array.** One plain object per `(stream, phase, block)`, so it
  ports to an object store without a design. Chunking, codecs and partial reads
  buy nothing for a blob whose length is whatever the writer decided.
* **It does not merge, and does not need a fan-in node.** A merge is a global
  reduction, so the task DAG has no node for it — and none is needed. Either run
  the block phase and reduce afterwards (`fragment::fold_fragments` streams one
  fragment at a time; `sidecar_fragments` holds them all and is the wrong tool at
  scale), or express the reduction as a `FragmentOp` whose reach is the whole
  volume, which the existing geometry already handles: every block reads
  everything, a short halo collapses the valid regions and fails the tiling
  check, and the cost model prices it towards one block. Multi-node needs nothing
  extra either way, because the fragments are on shared storage.
* **A full-reach op is a planning barrier**, and both planners segment there
  rather than fusing across it: per-block cost stops being local, so the
  infinite-grid cost model has no standing to trade the cut away. The predicate
  is `decomposition::is_planning_barrier` — `AxisReach::All` by type, or an
  exact `reach >= extent` for an op that states a number, never a threshold,
  because a *bounded* reach is not a barrier however large it is.

## Ops that are not `region -> region`

`sidecar` is storage. What *produces and consumes* fragments is `fragment`, and
it is a second trait rather than a wider `BlockOp::apply`:

```rust
impl FragmentOp for MyOp {
    fn name(&self) -> &'static str { "my_op" }
    fn reads_pixels(&self) -> bool { true }
    fn inputs(&self) -> Vec<FragmentInput> {          // stream, phase, reach in BLOCKS
        vec![FragmentInput::own("fragments", 1).with_reach([1, 0, 0])]
    }
    fn outputs(&self) -> Vec<FragmentOutput> {        // no defaults on either field
        vec![FragmentOutput::new("merged", Lifecycle::DeleteOnExit, Coverage::EveryBlock)]
    }
    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> { /* ... */ }
}
```

Three things worth knowing:

* **A reach in blocks becomes a task dependency.** The phase's halo is set to
  `reach * block_edge`, so the DAG — which is built from read-extent overlap —
  gives block `b` its neighbours as dependencies, single-node and distributed
  alike. A zero-reach phase reads exactly one fragment per stream per block, and
  that is measurable from outside the op because the executor does the gathering.
* **No pixel IO unless the op asks for it.** A `fragments -> fragments` phase
  moves no voxel and touches no read, write or chunk counter.
* **The tiling guard cannot see fragments, so there is a second guard.** A
  fragment phase's valid regions are its cores whatever it wrote, so the exact-
  tiling check passes vacuously. `Coverage::EveryBlock` makes the executor check
  the *store* after the phase; a phase that writes no image and declares no
  every-block stream is refused, because nothing about it would be checkable.

A stream is declared `DeleteOnExit` or `Persistent` and `Lifecycle` has no
`Default`, because keeping intermediate output deliberately and cleaning it up
are both legitimate and drifting into either is not. `discard_sidecars` removes
the delete-on-exit streams, **returns what it removed and emits it as an event**,
fails rather than swallows a removal error, and confirms the path is gone
afterwards — a cleanup that silently does nothing is indistinguishable from one
that worked, which is exactly how an earlier one went unnoticed.

---

## The 45 dangling citations

`docs/design/BLOCK_OPS.md` is cited 45 times across 28 files under `src/` and
`tests/` and does not exist. That gap is recorded in
[`images-and-phases.md`](images-and-phases.md) and in four documents under
`docs/ops-survey/`. Before naming this file, the citations were read to see
whether the material moved out of `README.md` is what they want.

**It is not, and this file is deliberately not called `BLOCK_OPS.md`.** Naming
it that would make 45 citations resolve to a document that answers almost none
of them, which is worse than a dangling link: a reader following
`src/env.rs:5`'s §"Simulating strategies" to a file that has no such section
learns nothing and cannot tell whether the section was lost or never existed.

What this file *does* satisfy:

* **`src/decomposition.rs:1231`** — §"A full-reach op is a planning barrier".
  The only exact section-title match. The bullet under *Output that is not an
  image* below gives the same three reasons (fusion impossible, per-block cost
  stops being local, cache does not survive) that the code's own header gives.

What it answers substantively, without being the cited source — in each case
because it restates the same anecdote the header does, from the same lost
original, so it adds no evidence:

* `src/op.rs:1159` and `tests/fan_in.rs:7` — the diamond modelled as an
  `Alternative` that passed 903 comparisons.
* `src/op.rs:2061` and `tests/fan_in.rs:377` — §"Step 2"'s contrast between a
  hole and an over-wide reach.
* `src/ops/background.rs:55` — what happens when a fan-in is modelled as an
  alternation.
* `src/ops/voxelwise.rs:7` and `tests/patch_grid_shape.rs:740` — §"`Chain` has
  no fan-in". **Superseded rather than supplied**: the shape those headers say
  cannot be expressed is `Chain::Parallel`, which ships. The headers are stale,
  not merely dangling.
* `src/ops/element.rs:33` — a reach fed by the configured halo. The *Reach*
  section below makes the same point about the guard being the comparison
  between a reach and a halo; it is not the same sentence.

**What remains unwritten**, and is in no file under `docs/`: §"The planning
problem is NP-hard", §"Emit a dependency graph, do not inline the loops",
§"Workflow -> Planner -> Executor, with the contract in the types",
§"Simulating strategies", §"Estimating from noisy samples", §"Images are a DAG",
§"Block size may differ per phase", §"The combined pass", §"Step 1", §"Step 3",
"explicit edges in the binding plan", the seven-merge-step measurement
`src/decomposition.rs:1166` attributes to it, and the cross-grid-fetch wall
`tests/op_constraints.rs:347` and `tests/several_outputs.rs:428` quote. That is
roughly 37 of the 45. Writing `BLOCK_OPS.md` is still somebody's job.

---

## What was checked

These are the claims other files rely on. Everything else in this file is
unchecked and carries the caveat at the top.

**Verified**

* Every API named in the *Branching* section exists with the shape shown:
  `Chain::op` (`src/op.rs:1248`), `Chain::sequence` (`:1253`),
  `Chain::alternative(branches, taken)` (`:1383`),
  `Chain::parallel(branches, combine) -> Result<Chain>` (`:1406`),
  `Chain::source(image: impl Into<ImageId>, dtype: Dtype)` (`:1287`),
  `LogicCombine::new` and `Logic` (`src/ops/voxelwise.rs:123`).
* **The fold table is the one in the code.** `src/op.rs:1150` carries the same
  five-row table with the same readings, including `side_outputs` folding by
  union for `Parallel` and by `taken` for `Alternative`, and `side_outputs`
  (`src/op.rs:2061`) is implemented that way. The reason given here for the
  asymmetry — an undeclared output is a hole the coverage guard reports, an
  over-wide reach only costs reads — is the reason given at `tests/fan_in.rs:377`.
* **The 903-comparison diamond.** Recorded at `src/op.rs:1159` and
  `tests/fan_in.rs:7`, in both cases attributed to `BLOCK_OPS.md`. The claim in
  this file is the same claim; it is not independent confirmation of it.
* **A `Parallel` is one indivisible slot**, for the stated reason — a phase reads
  one image and writes one image, so a cut between the branches and the combine
  would need a phase with several inputs. `Chain::slots` (`src/op.rs:2427`)
  states it in the same words.
* `PhaseDecomposition::source_images`, `check_source_images` and
  `Decomposition::readers_of_image` (`src/decomposition.rs:531`) all exist, and
  `check_source_images` is re-exported from `src/lib.rs:196`.
* **The reach forms.** `AxisReach` (`src/reach.rs:228`) has `Bounded { lo, hi }`,
  `PerBlock(Vec<(usize, usize)>)` and `All`; `Space` is a struct at
  `src/reach.rs:115`; `Reach::is_whole_axis` (`:630`) keys off the variant.
  `Reach::window` (`:768`) and `BlockConstraint::lattice` (`src/op.rs:239`)
  exist, and `lattice`'s own doc comment describes the inward-sliding window
  this file describes.
* **The 3.27x figure.** Pinned in the module header at `src/reach.rs:16`, in the
  same table row, for the same reason.
* **Conversion happens in `PhaseDecomposition::derive`.** `src/decomposition.rs:183`
  gives the identical argument — a reach in whole blocks or a permuted axis order
  is symbolic until a grid exists, because the planner is comparing candidate
  grids.
* **The sidecar API**, exactly as written: `declare_sidecar`, `write_sidecar`,
  `sidecar_fragments`, `discard_sidecars` (`src/env.rs:706`–`:778`), `Lifecycle`
  (`src/sidecar.rs:102`) with no `Default`, and `fragment::fold_fragments`
  (`src/fragment.rs:1633`).
* **`decomposition::is_planning_barrier` is an exact comparison, not a
  threshold** — `src/decomposition.rs:1225` is `extent > 1 && lo + hi >= extent`,
  and the header above it (`:1231`) gives the same fusion / non-local-cost /
  cache-does-not-survive argument this file gives.
* **The `FragmentOp` shape.** `trait FragmentOp: Send + Sync`
  (`src/fragment.rs:606`) with `name`, `reads_pixels` (default `false`),
  `inputs`, `outputs` and `fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput>`
  (`:721`); `FragmentInput::own(stream, phase)` and `.with_reach([_;3])`
  (`:169`, `:177`); `FragmentOutput::new(stream, lifecycle, coverage)` (`:253`)
  with all three fields required; `Coverage::EveryBlock` (`:226`).

**Stale, fixed**

* *Two arrays*: "a leaf naming an intermediate names **a** image number" →
  "**an** image number". A leftover from the `level → image` rename, which
  swept the README but left the article. The only edit made to the moved text.

**Flagged, not fixed**

* **The `FragmentOp` sketch is missing two methods that now exist.** The trait
  has grown `writes_pixels` (`src/fragment.rs:637`) and
  `produces(&self, input: Dtype) -> Dtype` (`:657`), and an `apply_with(at,
  sources)` (`:743`) for a fragment op that declares stored source images. The
  code block below is therefore a *valid* impl but no longer a complete picture
  of the trait, and the section's bullet "no pixel IO unless the op asks for it"
  is now understated: a fragment phase can also *write* an image, which the
  section does not mention at all.
* **The last sentence of *Two arrays* does not parse.** "the shipped planners do
  not yet place a boundary to satisfy one, they are refused by
  `check_source_images` if they do not" is a comma splice whose second clause has
  no clear subject. The intended claim is presumably that a chain whose source
  leaf needs a boundary the planner will not place is refused at plan time. Left
  as written rather than guessed at.
* **The *Reach* section says `BlockOp::reach` "is required and has no default".**
  True of `BlockOp` (`src/op.rs:608`, no body). Not true of `FragmentOp::reach`
  (`src/fragment.rs:618`), which defaults to `0` — the exact silent zero the
  paragraph argues against. Whether that is a considered difference or a drift
  was not established.
* The `Combine` element-type argument ("branch results need not agree with each
  other on element type; they must be acceptable to the combine, which is checked
  when the plan is made") was not traced to the check that does it.
* Nothing in the *Output that is not an image* section's claims about what
  `discard_sidecars` reports, emits and re-confirms was exercised; only the
  function's existence and signature were confirmed.
