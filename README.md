# blockflow

**not yet ready to be used**

This is an **experimental** crate for processing of large scale imaging data.
Modern microscopes are able to churn out TB-scale datasets, making it impossible
to load them into memory at once and using old bases. The solution is
(1) "out of core processing", where only a part of the data is present in memory
at once. Furthermore, (2) multithreading, GPUs and computing on multiple computers
in parallel is required.

Figuring out the optimal compute order is hard (likely NP-hard). The following factors
need to be taken into account:

* How much memory is available?
* How many threads are available, and what CPU/how much cache memory?
* How many computers are available?
* Is a GPU available? And if so, what type, and which compute nodes have them?
* How much time does it take to read data?
* How much time does it take to write data?
* How much time does it take to compress data?
* How well does the data compress?
* What operations are performed, and in which order?
* What precision does the data need to be stored in?

This crate aims to resolve the problem using the following ingredients:

* Operations are represented as a DAG (direct acyclic graph), representing dependencies
* Borrowing from database query planners, statistics about compute times are gathered during execution 
* A 4d scheduler figures out the best order and adapts in realtime based on statistics
* Designed for multiple compute nodes, GPUs and heterogenous compute environments from day one
* OME-Zarr is used to enable distributed computing on chunks of image data

This crate is not yet ready for general consumption.



**Below are ramblings for the LLM. info might not be accurate**

## Why it is its own crate

It was extracted from `clearmap-rs`, where it had grown to fifteen files under
`parallel_processing/block_ops/`. The reason for the boundary is **dependency
direction**, not packaging. Inside one crate, `use crate::image_processing::…`
is frictionless, so coupling accumulates silently — the parent repository has a
documented history of exactly that. Across a crate boundary every dependency is
deliberate, visible in `Cargo.toml`, and one-way:

> `blockflow` must not depend on `clearmap-rs`. `clearmap-rs` depends on
> `blockflow`.

The intended direction of travel is multi-node, out-of-core execution of general
image-processing pipelines. This crate is the part of that which is not specific
to any one pipeline.

## Two rules for anything added here

**1. Everything is a parameter.** Filter sizes, sigmas, thresholds, spacings,
structuring-element shapes — supplied by the caller, never baked in. An
application's *values* are its domain knowledge; the op only knows how to apply
a filter of a given size. This is what separates an op's logic from the problem
it happens to be used for, and a parameter that exists only because one caller
needs a particular number is a leak that will show up as an awkward interface
long before it shows up as anything else.

**2. No domain vocabulary in names or documentation.** Nothing here should
mention vessels, arteries, brains, or the application it was extracted from.
Where a name is domain-flavoured, the general equivalent exists and is the
honest name anyway:

| domain-flavoured | general |
|---|---|
| `tubify` | tubeness / vesselness enhancement (Frangi/Sato-style) |
| `vessel_background` | background estimation |
| `lightsheet_correction` | stripe / illumination correction |

**The naming test**, which is a cheap and surprisingly reliable filter for what
belongs where: *if an op cannot be named without domain terms, it is domain
logic and belongs in the application crate.* Apply it while writing.

Rule 2 is enforced — `tests/no_domain_vocabulary.rs` greps the crate for a list
of domain terms and fails on a hit. Rule 1 cannot be checked mechanically and is
a review matter.

Beyond licensing, the reason for both: a crate with no domain knowledge is
independently testable, reusable outside the project that produced it, and
forced to have an honest interface.

## Dimensionality: 3-D, with 2-D as the special case

Volumes are 3-D. There are no separate 2-D paths, and there should never be
one: a 2-D problem passes a volume of depth 1, blocks span the single plane, and
the z halo is 0 — the planner already handles a zero-reach axis, so nothing
special is needed. `reach(axis, volume_len)` is per-axis and stays that way.

Nothing needs 4-D. Channels are separate arrays, and a combining step reads two
sources rather than one 4-D array; time is not in scope.

## What can live here, and what cannot

This crate is MIT. `clearmap-rs` is a translation of ClearMap, which is
GPL-3.0, and **a translated op is a derivative work of ClearMap**. Moving such
a file into this crate would not relicense it; relicensing is not available to
us at all. So:

| | where it lives |
|---|---|
| the framework — ops, chains, geometry, decomposition, the DAG, the executor, the event stream, the cache, the prefetcher | **here**, MIT |
| an op **written from scratch** | **here**, MIT |
| an op **translated from ClearMap** (or from any GPL source) | **`clearmap-rs`**, GPL, as an adapter implementing `blockflow::BlockOp` |

That still gets the architecture the eventual vision wants — this crate defines
the interface; pipeline-specific implementations live outside it — but the
translated code itself does not migrate, ever. If you are tempted to move an op
across "because it is generic", check its provenance header first. A file whose
header names an upstream module is not eligible.

The first adapter is `clearmap_rs::dataflow::binarize`, which implements
`BlockOp` over ClearMap's binarize kernels. It is the worked example of the
boundary: this crate never learns what binarization is, and the kernels never
learn what a block is.

## Layout

| module | what it owns |
|---|---|
| `error` | The crate's own `Error`/`Result`. Small on purpose; callers convert at the boundary. |
| `dtype` | The element-type tag the cost model and the cache need. Not an IO abstraction. |
| `region` | `Region`, `RegionSource`, `RegionSink`, and the in-memory implementations. |
| `tiling` | The exact-tiling predicate the whole design's correctness guard rests on. |
| `budget` | One byte-denominated pool. `Reserved` compute may wait; `Opportunistic` cache and prefetch may not. |
| `op` | `BlockOp`, `Combine` and `Chain`. Reach, execution, traversal preference, constant algebra; sequences, exclusive alternatives, and a fan-in whose branches all run. |
| `fragment` | `FragmentOp`: the shapes `region -> region` cannot express — `volume -> fragments`, `fragments -> fragments`, `fragments -> volume`. One executor, and a coverage guard on the fragment side. |
| `geometry` | Read extent, trustworthy extent, `valid = core ∩ trustworthy`. |
| `decomposition` | The binding plan — parity-visible, deterministic, data-blind, hashable — and its cost model. |
| `graph` | `(block, phase)` tasks with explicit dependencies. |
| `env` | The injected environment: real arrays, or a loader that only accumulates cost. |
| `sidecar` | Per-block output that is **not** a pixel region: `(stream, phase, block) -> bytes`, plain objects, a declared lifecycle, a deletion that reports itself. |
| `strategy` | One `Strategy` trait, and the single executor every strategy shares. |
| `log` | The `Event` stream and the log the acceptance criterion is asserted from. |
| `statistics` | Measured cost coefficients, accumulated from real runs and persisted. Nanoseconds per unit of *declared* cost, keyed by machine; an absent store leaves the shipped constants exactly where they are. |
| `listener` | `EventListener`, the dispatch set, and the built-in listeners. |
| `observed_io` | Source/sink decorators, so IO outside the executor emits through the same trait. |
| `export` | The order log as JSON, with a cross-language schema. |
| `animate` | The seam to the bundled renderer. Opt-in, and no dependency of the crate. |
| `probes` | Synthetic ops that prove the framework without a real kernel. |
| `synthetic` | A generated volume with its ground truth: intensities, an exact label volume, an object table. Objects are placed in global coordinates and rendered by region, so a block equals the whole volume's cut bit for bit. |
| `agreement` | How a produced labelling relates to a known-correct one — matched, split, merged, missed, spurious — matched by overlap, because label ids never agree. |
| `cache` | One `(array, chunk)`-keyed LRU with a decoded and an encoded tier, over one byte budget. |
| `prefetch` | A scheduler, not a predictor: it reads the plan the caller already has. |
| `net` | One bind policy, shared by both servers, and how a coordinator decides what address to publish. |
| `distributed` | A coordinator, workers that pull, four rendezvous backends, and a local multi-node mode that runs all of it as separate processes. |
| `gui` | *(feature `gui`)* An HTTP server over the progress listener, and the browser view it feeds. |
| `zarr_env` | *(feature `zarr`)* Levels as Zarr v3 arrays on a filesystem store — the `Environment` that actually moves bytes. Compression is per level and derived from the level's own element type. |

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
  reads one level and writes one level, so a cut between the branches and the
  combine would need a level per branch and a phase with several inputs, neither
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

A phase reads the level it is handed. An operation needing a *second* array —
measuring one array against another, masking, seeding a reconstruction from
somewhere other than its own mask — used to get it by holding it whole
(`CombineOp`), which is one full copy of the array resident for the length of
the run.

`Chain::Source` is a leaf that reads a stored level instead of computing one,
so the second arm of a fan-in can be an array in storage:

```rust
Chain::parallel(
    vec![computed_arm, Chain::source(level, Dtype::F64)],
    Box::new(LogicCombine::new("xor", Logic::Xor)),
)?
```

It reuses the fan-in machinery entirely: one buffer per branch, joined by the
combine, with every fold above applying unchanged. Three things are new, and
none of them is assumed:

* **reach 0**, exactly. It reads the block's own read extent and nothing
  around it, so it never widens the halo of the arm beside it.
* **the level is in the plan.** Which level an arm reads changes voxels, so it
  is recorded in `PhaseDecomposition::source_levels`, fingerprinted, and sent
  over the wire. `check_source_levels` compares it against the chain and
  refuses, *by name and when the plan is made*, a level that does not exist, a
  forward reference to one a later phase writes, one on a different lattice,
  and one whose element type is not what the leaf declared.
* **a level dies after its last reader.** A level with a second reader is not
  freed when the first one finishes. `Decomposition::readers_of_level` is the
  refcount; with no source leaf it answers one phase and the old rule falls out
  of the new one unchanged.

Level 0 is the case of this that always existed: a level with no producing
phase. `Chain::source(0, dtype)` says so explicitly — and it is the one form
that is valid under *every* partition, because level 0 is below every phase
whatever the planner does with the chain. A leaf naming an intermediate names a
level number, so it constrains where the phase boundaries may fall; the shipped
planners do not yet place a boundary to satisfy one, they are refused by
`check_source_levels` if they do not.

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
volume, or the level below's — which decides whether a read clamped at the
phase's own edge may be trusted), a **unit** (voxels, whole blocks of this
phase's lattice, or steps of the level below's lattice) and an **axis order**.
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

## Storage

Until `zarr_env` existed, every claim this crate made was a claim about arrays
already in memory: `ArrayEnvironment` holds whole volumes, `AccountingEnvironment`
holds nothing and prices what it would have held. That is a real gap in a crate
whose subject is out-of-core execution, and `ZarrEnvironment` closes it. Level
`l` is the array at `root/level<l>`; `prepare` creates levels 1..n at each
phase's own volume and element type; all eleven element types a `Voxels` can hold
map to a Zarr v3 data type, and `float16` is **refused by name** rather than
widened, because this crate has no buffer that can hold one.

The acceptance criterion is stated as a negative and asserted in
`tests/zarr_env.rs`: **the storage layer is invisible to the answer.** Every op
family, run over `synthetic::Scene` data through Zarr arrays on a disk, produces
what the same op produces through `ArrayEnvironment` — at several block edges,
several chunk shapes, and up to eight threads.

### The one thing a storage backend here has to get right

`zarrs` 0.23.13 loses data on concurrent partial-chunk writes: its
`store_chunk_subset` decodes the chunk, patches the sub-box, re-encodes and
overwrites, with the per-chunk lock its own source scaffolded still commented
out. Two threads writing two halves of one chunk both read the old chunk and both
write a whole one. Measured here: **39–40 of 40 trials lost a half-chunk**
unguarded, **0 of 40** guarded.

The guard is per-chunk serialisation over a fixed stripe array, taken **only for
the chunks a write covers in part** — a chunk covered edge to edge is a blind
overwrite with nothing to lose, so a chunk-aligned write takes no locks at all.
`ZarrEnvironment::serialised_writes` says how often the slow path was taken, so a
caller who can align their blocks to the chunk grid can see that it worked. The
answer is the same either way, which is this crate's founding principle applied
to the write side: *a mistake about chunks costs performance, never correctness.*

### Compression, chosen per level and derived from the element type

An out-of-core framework writes and re-reads its intermediates constantly, and
one of them is a `bool` mask that should cost almost nothing to keep. So a level
carries a codec, and — because the levels of one plan are not one kind of data —
**it is chosen per level, not per run**:

```rust
// The default. Every level gets `Compression::for_dtype` of its own type.
let env = ZarrEnvironment::create(root, &input, [64, 64, 64])?;

// Or say it. `uniform` speaks for the run; `with_level` overrides one level.
let policy = CompressionPolicy::derived()
    .with_level(2, Compression::Gzip(6));      // the mask, harder
let env = ZarrEnvironment::create_with_compression(root, &input, chunk, policy)?;

env.compression_at(2)?;      // what was built, read off the array
env.stored_bytes(2)?;        // and what it cost on the disk
```

The default is *derived* rather than configured: levels already carry their own
element type (`Decomposition::dtype_at`), and the element type is the best single
predictor of whether deflate will pay. **`bool` and the integers compress at
level 1; `float32` and `float64` are left raw.**

The evidence is a test, not a claim — `compression_pays_for_bool_and_not_for_float`
prints this table on every run. A 64³ `synthetic::Scene`, chunk 32³, release
build, level 0 `float64` and level 1 a `bool` mask:

| policy | level 0 (`float64`) | level 1 (`bool`) | run | break-even |
|---|---|---|---|---|
| no compression | 2 097 152 B | 262 144 B | 7 ms | — |
| `gzip1` everywhere | 1 978 908 B (1.06x) | 22 845 B (11.5x) | 42 ms | 10.4 MB/s |
| `gzip9` everywhere | 1 982 399 B (1.06x) | 10 188 B (25.7x) | 92 ms | 4.3 MB/s |
| **derived (the default)** | 2 097 152 B (1.00x) | 22 845 B (11.5x) | **11 ms** | **73.7 MB/s** |
| derived, `bool` at `gzip6` | 2 097 152 B | 10 685 B (24.5x) | 25 ms | 14.5 MB/s |
| derived, `bool` at `gzip9` | 2 097 152 B | 10 188 B (25.7x) | 64 ms | 4.5 MB/s |

**break-even** is bytes saved over CPU seconds spent saving them: the store speed
below which compressing is faster end to end. It is the number the defaults are
picked on. Compressing the `float64` level — the step from the derived row to
`gzip1` everywhere — costs 31 ms to save 118 kB, a break-even of **3.8 MB/s**,
slower than any store this will meet, so the floats are left alone. Compressing
the `bool` level costs 4 ms to save 239 kB — **73.7 MB/s**, which most network and
shared storage is slower than. Turning `bool` up to level 6 doubles the ratio
again, but the *incremental* trade is 12 kB for 14 ms — 0.9 MB/s, worse than the
`float64` case just refused, which is why the default is level 1 and the higher
levels are a caller's decision. `uint16` is the weakest of the three and is
stated as such: 1.36x here, against **2.09x** measured on real acquisitions and
**19.7x** on real `bool` intermediates, the pair `cache::DeflateCodec` was
chosen on.

`gzip` is the only codec offered, and the reason is the dependency graph rather
than the ratio: `zarrs`'s `gzip` feature is exactly `dep:flate2`, and `flate2` is
already here for `cache::DeflateCodec`, so enabling it adds **no package** — the
`zarr` feature costs the same as it did. `zstd` and `blosc` would each add crates
and a C build to buy a ratio, and the place the ratio matters most is `bool`
data, where deflate is already near the ceiling.

**What compression does to the guard above**, because it changes a trade and not
just a constant. A fully covered chunk is still a blind overwrite with no lock,
but the write is now an *encode*. A partly covered chunk was decode-patch-encode
and is now **decompress-patch-recompress, inside the lock** — the serialised
section becomes the dominant cost of such a write rather than a rounding error on
it. That does not weaken the guard; it makes the alignment advice matter more,
and `serialised_writes` / `unaligned_reads` still point at exactly the writes and
reads that are paying for it. The race test runs both ways and the compressed arm
loses *more* without the guard than the raw one does — a longer critical section
is an easier one to lose data in — and zero with it.

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
  the *store* after the phase; a phase that writes no level and declares no
  every-block stream is refused, because nothing about it would be checkable.

A stream is declared `DeleteOnExit` or `Persistent` and `Lifecycle` has no
`Default`, because keeping intermediate output deliberately and cleaning it up
are both legitimate and drifting into either is not. `discard_sidecars` removes
the delete-on-exit streams, **returns what it removed and emits it as an event**,
fails rather than swallows a removal error, and confirms the path is gone
afterwards — a cleanup that silently does nothing is indistinguishable from one
that worked, which is exactly how an earlier one went unnoticed.

## Seeing a schedule

A schedule is much easier to believe once you have watched one. `tools/
animate_block_progress.py` draws an exported order log, and there is a CLI over
both halves:

```
cargo run --release --bin block_animate -- --grid 5,5,5 --out run.json --render
```

That writes the log and then, only because `--render` asked for it, the movie:
one cube per block, coloured by how far through the chain the block is, with the
camera orbiting the lattice. `--view 2d` draws the grid flat instead, and takes
two logs side by side — which is what `block_progress_export` produces, and the
reason it exists: two schedules over one decomposition must look different, and
seeing that they do is what says the picture follows the schedule.

**Rendering is optional and Python is not a dependency.** Exporting a log needs
nothing but `cargo`, and the log is written before rendering is attempted. If
manim is missing, that log is still on disk and the message names what to install
rather than what went wrong internally. `blockflow::animate::render` is the same
capability as a library call.

## Watching a schedule, live or replayed

The animation above is for showing somebody. The `gui` feature is for *you*,
while a job is running or afterwards:

```
cargo run --release --features gui --bin block_gui -- run.json     # replay a log
cargo run --release --features gui --bin block_gui -- --demo       # watch a run
```

Then open `http://127.0.0.1:8731/`. The page is a grid of blocks coloured by how
far through the chain each has got, a legend, a timeline, and — for a replay —
play, step and scrub.

**The browser does not know which it is looking at.** A replay is a recorded
stream: its events are decoded back into `Event`s and fed to the same
`LatestOpPerChunk` a live run feeds, so both answer the same four endpoints
(`/api/meta`, `/api/state`, `/api/events`, `/api/control`) and the page branches
on one capability flag, never on a mode.

**Watching must not disturb.** The server runs on its own threads, a snapshot
takes the progress listener's *shared* guard, and a floor on how often a
snapshot is taken means no number of clients polling at any rate can cost the
run more than twenty snapshots a second. `gui::tests` asserts that a run with a
client polling flat out does identical work to a run with no server, and reports
what it costs in wall time.

**On a compute node, keep the default bind and forward the port.** The server
binds `127.0.0.1` and refuses anything else unless `--allow-public` is passed,
because a shared node with an unauthenticated HTTP server on `0.0.0.0` publishes
somebody's run to everybody:

```
node$   block_gui --demo
laptop$ ssh -N -L 8731:127.0.0.1:8731 user@node
```

**Building the page.** The browser half is `webui/`, a separate crate compiled
to WebAssembly by [trunk](https://trunkrs.dev), not by cargo:

```
cargo install trunk && (cd webui && trunk build --release)
```

The server reads the result from `webui/dist` (or `--assets DIR`, or
`$BLOCKFLOW_GUI_ASSETS`). With the page unbuilt the server still runs and still
serves the endpoints, and `/` explains how to build it.

## Running on many nodes

The task DAG is the point at which multi-node stops needing a redesign: tasks
are independent given their inputs, and **a task never needs a peer's in-memory
output** — it reads storage. So coordination traffic is metadata only, and the
shape is a **coordinator that is its own program** plus **workers that pull**.

```
blockflow-coordinator --rendezvous "$RDV" --job job.json --exit-when-done &
<launcher> blockflow-worker --rendezvous "$RDV"
wait
```

Not MPI, and the reason is not taste: there is no rank-to-rank transfer here to
use collectives or RDMA on, so what MPI would add is a C dependency and a static
rank model that fights a greedy adaptive scheduler.

**Pull rather than static assignment**, because a large and data-dependent
fraction of blocks are empty, so a `block % N` split hands one worker a dense
region and another a sparse one.

**Nodes do not die** (decided 2026-08-17). The deployment is 10-20 cooperating
nodes on AWS and SLURM, where losing one is a major event and not a routine one
— and where a lost node held blocks in memory, so re-running the tasks it had
*claimed* restores the claim table and not the position. So a claim has **no
expiry**: it is held until it completes. A worker that goes takes the job with
it — the run aborts, naming the worker and every task it was holding, and what
to do next is decided by the batch script or the orchestrator that started it.
Detection is a signal and not a timeout: whoever launched the worker sees the
process exit and says so. The reissue machinery is still there, still tested,
and a job opts in by setting `JobSpec::lease`. See `src/distributed/mod.rs`.

**Handout is locality-biased, not territorial.** Workers are seeded far apart
and then take the nearest unclaimed task, so they grow compact regions towards
each other. Territories would balance badly, because empty regions cluster.
Measured over a real decomposition with four workers: naive global pull
duplicates 62 of 64 chunks across workers (1.97 fetches per distinct chunk);
nearest-first duplicates 6 (1.09).

**The coordinator models each worker's cache and is never told.** It assigned
every task, so it can replay the same eviction policy against what it handed
out. Nothing about cache state may introduce a point where one side waits for
the other — see `distributed::cache_model`.

### Local multi-node mode

One command, a coordinator and N workers, **as separate processes**:

```
cargo build --release --features distributed
blockflow-local --workers 4 --blocks 32 --phases 2
blockflow-local --workers 3 --kill 0:4                  # a worker dies: the job aborts
blockflow-local --workers 3 --kill 0:4 --lease-ms 400   # opt in: the claims are reissued
```

Processes rather than threads, deliberately: real HTTP, real separate caches and
memory budgets, real process boundaries. A thread-based fake would share one
address space and would validate almost nothing that matters — not a flush
missed before a dependent read, not two processes writing one file, not a work
list that only stays ahead because the "network" was a function call.

This is how the distribution claims are checked (`tests/local_multi_node.rs`):
N workers produce byte-identical output to a single-node run over several
worker counts; the merged event stream satisfies the same
`check_coverage_and_order` a single-node run is asserted with; a killed worker
aborts the job promptly with a message naming it and its claims; the same kill
under an explicit lease has those claims reissued with the output still
byte-identical; and no worker's work list ever runs empty while the coordinator
has work.

### Finding each other

Four backends behind one trait, because the mechanism differs and only the
mechanism does: a file on a shared filesystem keyed by the job id; an
environment variable, where the scheduler already told every node the main
node's address; an object in a store, polled; or an address on the command line.

A coordinator has to be reachable by other nodes, so it is exactly the
`--allow-public` path the progress view refuses by default — and that stays a
conscious choice, because there is no authentication. What is *advertised* is
separate from what is *bound* (`--advertise`), because clusters have management
and fabric interfaces with different names for one host.

## Testing

```
cargo test
cargo test --features gui,distributed
```

## License

MIT (AI generated code)
