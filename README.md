# blockflow

Out-of-core block processing, as a library: an op trait and a chain algebra, a
reach-derived valid region, a deterministic block decomposition, a task DAG, a
strategy contract with one shared executor, an event stream with pluggable
listeners, and a shared chunk cache with a hint-driven prefetcher.

MIT. Original work.

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
| `op` | `BlockOp` and `Chain`. Reach, execution, traversal preference, constant algebra. |
| `fragment` | `FragmentOp`: the shapes `region -> region` cannot express — `volume -> fragments`, `fragments -> fragments`, `fragments -> volume`. One executor, and a coverage guard on the fragment side. |
| `geometry` | Read extent, trustworthy extent, `valid = core ∩ trustworthy`. |
| `decomposition` | The binding plan — parity-visible, deterministic, data-blind, hashable — and its cost model. |
| `graph` | `(block, phase)` tasks with explicit dependencies. |
| `env` | The injected environment: real arrays, or a loader that only accumulates cost. |
| `sidecar` | Per-block output that is **not** a pixel region: `(stream, phase, block) -> bytes`, plain objects, a declared lifecycle, a deletion that reports itself. |
| `strategy` | One `Strategy` trait, and the single executor every strategy shares. |
| `log` | The `Event` stream and the log the acceptance criterion is asserted from. |
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
| `zarr_env` | *(feature `zarr`)* Levels as Zarr v3 arrays on a filesystem store — the `Environment` that actually moves bytes. |

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
  is `decomposition::is_planning_barrier` — an exact `reach >= extent`, never a
  threshold, because a *bounded* reach is not a barrier however large it is.

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
use collectives or RDMA on, so what MPI would add is a C dependency, a static
rank model that fights a greedy adaptive scheduler, and fault intolerance — one
rank dies, the job dies.

**Pull rather than static assignment**, because a large and data-dependent
fraction of blocks are empty, so a `block % N` split hands one worker a dense
region and another a sparse one. Pull also gives fault tolerance for free: a
claim unhonoured past its lease is reissued.

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
blockflow-local --workers 3 --kill 0:4 --lease-ms 400   # a worker is killed
```

Processes rather than threads, deliberately: real HTTP, real separate caches and
memory budgets, real process boundaries. A thread-based fake would share one
address space and would validate almost nothing that matters — not a flush
missed before a dependent read, not two processes writing one file, not a work
list that only stays ahead because the "network" was a function call.

This is how the distribution claims are checked (`tests/local_multi_node.rs`):
N workers produce byte-identical output to a single-node run over several
worker counts; the merged event stream satisfies the same
`check_coverage_and_order` a single-node run is asserted with; a killed worker's
claims are reissued and the output is still byte-identical; and no worker's work
list ever runs empty while the coordinator has work.

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
