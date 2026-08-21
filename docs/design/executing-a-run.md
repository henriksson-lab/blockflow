# Executing a run

Where a run's bytes live, how to look at the schedule it produced, and how the
same run spreads over many machines.

*Split out of `README.md`. Companion files:
[`dimensions-and-modules.md`](dimensions-and-modules.md),
[`writing-an-op.md`](writing-an-op.md),
[`images-and-phases.md`](images-and-phases.md),
[`barriers.md`](barriers.md).*

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

## Storage

Until `zarr_env` existed, every claim this crate made was a claim about arrays
already in memory: `ArrayEnvironment` holds whole volumes, `AccountingEnvironment`
holds nothing and prices what it would have held. That is a real gap in a crate
whose subject is out-of-core execution, and `ZarrEnvironment` closes it. Image
`l` is the array at `root/level<l>`; `prepare` creates images 1..n at each
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

### Compression, chosen per image and derived from the element type

An out-of-core framework writes and re-reads its intermediates constantly, and
one of them is a `bool` mask that should cost almost nothing to keep. So an image
carries a codec, and — because the images of one plan are not one kind of data —
**it is chosen per image, not per run**:

```rust
// The default. Every image gets `Compression::for_dtype` of its own type.
let env = ZarrEnvironment::create(root, &input, [64, 64, 64])?;

// Or say it. `uniform` speaks for the run; `with_image` overrides one image.
let policy = CompressionPolicy::derived()
    .with_image(2, Compression::Gzip(6));      // the mask, harder
let env = ZarrEnvironment::create_with_compression(root, &input, chunk, policy)?;

env.compression_at(2)?;      // what was built, read off the array
env.stored_bytes(2)?;        // and what it cost on the disk
```

The default is *derived* rather than configured: images already carry their own
element type (`Decomposition::dtype_at`), and the element type is the best single
predictor of whether deflate will pay. **`bool` and the integers compress at
level 1; `float32` and `float64` are left raw.**

The evidence is a test, not a claim — `compression_pays_for_bool_and_not_for_float`
prints this table on every run. A 64³ `synthetic::Scene`, chunk 32³, release
build, image 0 `float64` and image 1 a `bool` mask:

| policy | image 0 (`float64`) | image 1 (`bool`) | run | break-even |
|---|---|---|---|---|
| no compression | 2 097 152 B | 262 144 B | 7 ms | — |
| `gzip1` everywhere | 1 978 908 B (1.06x) | 22 845 B (11.5x) | 42 ms | 10.4 MB/s |
| `gzip9` everywhere | 1 982 399 B (1.06x) | 10 188 B (25.7x) | 92 ms | 4.3 MB/s |
| **derived (the default)** | 2 097 152 B (1.00x) | 22 845 B (11.5x) | **11 ms** | **73.7 MB/s** |
| derived, `bool` at `gzip6` | 2 097 152 B | 10 685 B (24.5x) | 25 ms | 14.5 MB/s |
| derived, `bool` at `gzip9` | 2 097 152 B | 10 188 B (25.7x) | 64 ms | 4.5 MB/s |

**break-even** is bytes saved over CPU seconds spent saving them: the store speed
below which compressing is faster end to end. It is the number the defaults are
picked on. Compressing the `float64` image — the step from the derived row to
`gzip1` everywhere — costs 31 ms to save 118 kB, a break-even of **3.8 MB/s**,
slower than any store this will meet, so the floats are left alone. Compressing
the `bool` image costs 4 ms to save 239 kB — **73.7 MB/s**, which most network and
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

---

## What was checked

These are the claims other files rely on. Everything else in this file is
unchecked and carries the caveat at the top.

**Verified**

* **`root/level<l>` is right, and is not a `level → image` leftover.** The
  *concept* was renamed to image; the *on-disk prefix* was deliberately not.
  `src/zarr_env.rs:1317` is `format!("/level{image}")`, and the module header at
  `:27` states the same sentence this file states. Do not "fix" it.
* **`zarrs` is on 0.23.13** (`Cargo.toml:224`, `Cargo.lock:1326`), so the version
  the concurrency bug is attributed to is the version in the tree.
* **"All eleven element types … and `float16` is refused by name."** `Dtype`
  (`src/dtype.rs:22`) has twelve variants; `F16` is refused rather than widened,
  with the refusal naming `float16` and `float32` in its message
  (`src/zarr_env.rs:236`, asserted at `:1908`). Twelve minus one is eleven — the
  count is consistent, not off by one.
* **`tests/zarr_env.rs` exists**, and the compression evidence table is printed by
  a real test: `compression_pays_for_bool_and_not_for_float`
  (`tests/zarr_env.rs:488`), which is the name `src/zarr_env.rs:313` cites as its
  evidence.
* **The compression API is as written**: `Compression::for_dtype`
  (`src/zarr_env.rs:383`), `CompressionPolicy::derived` (`:456`), `.with_image`
  (`:471`), `create_with_compression` (`:889`), `compression_at` (`:1149`),
  `stored_bytes` (`:1166`), `serialised_writes` (`:1066`), `unaligned_reads`
  (`:1075`), `Decomposition::dtype_at` (`src/decomposition.rs:495`),
  `cache::DeflateCodec` (`src/cache.rs:248`).
* **The race test really runs 40 trials both ways** and reports guarded and
  unguarded counts separately (`src/zarr_env.rs:2456`, `:2465`), so the
  "39–40 of 40 / 0 of 40" figures come from a runnable measurement.
* **The `gzip`-only argument holds at the dependency level.** `Cargo.toml:215`
  records the same reasoning — `zarrs`'s `gzip` feature is exactly `dep:flate2`,
  and `flate2` is already a dependency — and `zarrs` is pulled in with
  `default-features = false`.
* **Both binaries and both tools exist**: `src/bin/block_animate.rs`,
  `src/bin/block_progress_export.rs`, `tools/animate_block_progress.py`, and
  `blockflow::animate::render` (`src/animate.rs:292`). `--view 2d` is accepted in
  the spaced form the example uses (`src/bin/block_animate.rs:151`).
* **The GUI details.** Default bind `127.0.0.1:8731` (`src/bin/block_gui.rs:45`);
  `--allow-public` is the only way past the loopback check
  (`src/net.rs:45`); all four endpoints are served by one dispatch
  (`src/gui/server.rs:313`–`:336`); the same `LatestOpPerChunk` feeds live and
  replay (`src/gui/mod.rs:19`, `src/gui/live.rs:53`); assets come from
  `webui/dist`, `--assets DIR` or `$BLOCKFLOW_GUI_ASSETS`
  (`src/gui/server.rs:380`).
* **"No more than twenty snapshots a second"** is a real floor, not a figure of
  speech: `DEFAULT_MIN_SNAPSHOT_INTERVAL` is 50 ms (`src/gui/live.rs:142`).
* **Four rendezvous backends behind one trait**, matching the four described:
  `FileRendezvous`, `EnvRendezvous`, `ObjectRendezvous`, `DirectRendezvous`
  (`src/distributed/rendezvous.rs:171`, `:230`, `:382`, `:282`), and
  `advertised_addr` is the separate advertised/bound quantity (`src/net.rs:68`).
* **The claim expiry decision.** `JobSpec::lease` is `None` unless a caller sets
  it, and `src/distributed/mod.rs:42` gives the same reasoning in the same terms.
* **The 62-vs-6 locality figure** is pinned in a module header
  (`src/distributed/placement.rs:20`) and referenced again at
  `src/distributed/tests.rs:459`.
* `tests/local_multi_node.rs` exists; `log::check_coverage_and_order`
  (`src/log.rs:431`) is the predicate it is said to reuse;
  `distributed::cache_model` is a real module (`src/distributed/mod.rs:118`).
* `blockflow-local`'s flags are as shown, including `--kill I:N` and `--lease-ms N`
  (`src/bin/blockflow_local.rs:45`, `:51`).

**Flagged, not fixed**

* **The compression table's numbers were not re-measured for this move.** They
  are printed by `compression_pays_for_bool_and_not_for_float` on every run, so
  they are checkable — but they are a release-build timing on one machine, and
  the millisecond columns and the break-even figures derived from them will
  differ elsewhere. The *ordering* is what the defaults rest on.
* **The 1.97 / 1.09 redundancy ratios were not found as literals.** They are
  computed and printed by `nearest_first_handout_costs_fewer_duplicated_fetches_than_naive_pull`
  (`src/distributed/tests.rs:218`), which asserts only the ordering. The 62-vs-6
  pair behind them is pinned; the ratios are a report, not a pin.
* **The 2.09x and 19.7x figures** attributed to real acquisitions and real `bool`
  intermediates are quoted from `src/zarr_env.rs:358`, which quotes them in turn.
  Neither this file nor that header is the measurement.
* **"Nodes do not die (decided 2026-08-17)"** is a dated design decision about a
  specific deployment, not a property of the code. The reissue machinery it
  describes as opt-in is present and tested; the decision itself can go stale
  without anything failing.
