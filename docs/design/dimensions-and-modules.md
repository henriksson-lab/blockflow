# Dimensions and modules

*Split out of `README.md`. Companion files: [`writing-an-op.md`](writing-an-op.md),
[`executing-a-run.md`](executing-a-run.md), [`images-and-phases.md`](images-and-phases.md).*

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

## Dimensionality: 3-D, with 2-D as the special case

Volumes are 3-D. There are no separate 2-D paths, and there should never be
one: a 2-D problem passes a volume of depth 1, blocks span the single plane, and
the z halo is 0 — the planner already handles a zero-reach axis, so nothing
special is needed. `reach(axis, volume_len)` is per-axis and stays that way.

Nothing needs 4-D. Channels are separate arrays, and a combining step reads two
sources rather than one 4-D array; time is not in scope.

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
| `zarr_env` | *(feature `zarr`)* Images as Zarr v3 arrays on a filesystem store — the `Environment` that actually moves bytes. Compression is per image and derived from the image's own element type. |

---

## What was checked

Claims below were read against the code at the time of the split. Everything
else in this file is unchecked and carries the caveat at the top.

**Verified**

* `reach(axis, volume_len)` is per-axis and takes exactly those two arguments —
  `BlockOp::reach` (`src/op.rs:608`) and `FragmentOp::reach` (`src/fragment.rs`)
  are both `fn reach(&self, axis: usize, volume_len: usize) -> usize`.
* Every one of the 28 module names in the layout table names a module that
  exists in `src/lib.rs`.
* The `sidecar` row's key shape. `FragmentKey` is `{ stream, phase, block }`
  (`src/sidecar.rs:135`), which is the `(stream, phase, block) -> bytes` the row
  claims.
* The `zarr_env` row's "compression is per image and derived from the image's own
  element type" — `Compression::for_dtype` and `CompressionPolicy::derived`
  (`src/zarr_env.rs:383`, `:456`).

**Flagged, not fixed**

* **The layout table is incomplete.** Seven public modules have no row:
  `assemble`, `iterate`, `ops`, `points`, `reach`, `table`, `voxels`. Four of
  them are load-bearing for the sections in
  [`writing-an-op.md`](writing-an-op.md) — `reach` owns `Reach`, `AxisReach` and
  `Space`; `ops` is the whole shipped op library; `iterate` owns the
  many-substage phase; `table` and `points` own non-image output. A reader using
  this table as a map will not find them.
* **The `op` row is out of date on one word.** It claims `op` owns "Reach";
  `Reach`, `AxisReach`, `Space` and `Reach::window` live in `src/reach.rs`.
  `BlockOp::reach` and `BlockOp::reach_spec` are still declared in `op`, so the
  row is not simply wrong — it is a summary that predates the split of the type
  out of the module. Left as written rather than rewritten.
* The claim that a 2-D problem needs nothing special (depth 1, blocks spanning
  the single plane, z halo 0) was not exercised. It is an argument, not a
  measurement, and no test was found that pins it.
