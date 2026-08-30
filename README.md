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


## Design notes

The longer design material that used to sit below this line is now in
[`docs/design/`](docs/design/) — [dimensions and modules](docs/design/dimensions-and-modules.md),
[writing an op](docs/design/writing-an-op.md), [executing a run](docs/design/executing-a-run.md),
[images and phases](docs/design/images-and-phases.md) — where it keeps the
accuracy caveat it was written under, and each file ends with a list of which of
its claims have been checked against the code.

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

## Testing

```
cargo test
cargo test --features gui,distributed,zarr,model-segment
```

Both are what CI runs, and both take about a minute — `[profile.dev]` compiles
this crate at `opt-level = 1` and its dependencies at `2`, which is the
difference between a suite of 2002 tests that takes **63 s** and one that takes
**622 s**. The manifest has the measurements.

The suite that asserts is the suite that runs. The 39 `#[ignore]`d tests are
**measurements** — tables of nanoseconds per voxel, of resident bytes, of how
far repetitions moved — and they print rather than assert, because nothing in
this crate asserts on a duration. Run them deliberately, on a quiet machine:

```
cargo test --release -- --ignored --nocapture
```

The features CI does not cover are the ones a hosted runner cannot: `fftw`
wants a system `libfftw3` (Linux and macOS jobs install one; there is no
Windows job), and everything `*-cuda` wants a device.

## License

MIT (AI generated code)
