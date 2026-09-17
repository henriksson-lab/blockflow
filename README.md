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

## CellProfiler-Style Benchmark

One current benchmark is the CellProfiler ExampleHuman HT29 image set. To keep
startup overhead from dominating, this benchmark duplicates the small
three-channel example into 10-image and 50-image batches. The Blockflow rows
below are the release `cellprofiler-human` binary running the current DAPI
nuclei path over the DAPI images. The CellProfiler rows are
`cellprofiler/cellprofiler:4.2.8` running the downloaded `ExampleHuman.cppipe`
pipeline in Docker over the same three-channel image sets.

Measured on 2026-09-17 on an Intel Xeon Gold 6138 machine. Compile time is not
included. The CellProfiler run uses a warm local Docker image and reports peak
container memory sampled with `docker stats`; Blockflow RSS is Linux
`/usr/bin/time -v` max RSS.

| runner | scope | wall time | max RSS / peak memory | output |
|---|---:|---:|---:|---:|
| Blockflow `target/release/cellprofiler-human` | 10 DAPI nuclei runs | 1.09 s | 23,708 KiB / 23.2 MiB | 2,880 nuclei |
| CellProfiler 4.2.8 Docker | 10 full ExampleHuman image sets | 54.93 s | 504.4 MiB | 2,890 nuclei |
| Blockflow `target/release/cellprofiler-human` | 50 DAPI nuclei runs | 5.17 s | 23,928 KiB / 23.4 MiB | 14,400 nuclei |
| CellProfiler 4.2.8 Docker | 50 full ExampleHuman image sets | 238.13 s | 671.2 MiB | 14,450 nuclei |

That is about a 50x wall-time difference on 10 image sets and 46x on 50 image
sets, with about 22x lower peak memory on 10 image sets and 29x lower peak
memory on 50 image sets for the current Blockflow path. This is a
pipeline-reference benchmark, not a claim of exact operation-for-operation
parity: the CellProfiler pipeline also
identifies secondary/tertiary objects and exports more measurements. The
current semantic comparison is documented in `CELLPROFILER.md`; the remaining
one-object difference is treated as expected reference drift for this milestone.

Reproduction commands:

```sh
cargo build -p blockflow-cellprofiler-human --release --bin cellprofiler-human

N=50
bench=target/cellprofiler-readme-bench-${N}
mkdir -p "$bench/images"
for n in $(seq 0 $((N - 1))); do
  id=$(printf "%02d" "$n")
  cp .tmp/cellprofiler-human/examples-master/ExampleHuman/images/AS_09125_050116030001_D03f00d0.tif \
    "$bench/images/AS_09125_050116${id}_D03f00d0.tif"
  cp .tmp/cellprofiler-human/examples-master/ExampleHuman/images/AS_09125_050116030001_D03f00d1.tif \
    "$bench/images/AS_09125_050116${id}_D03f00d1.tif"
  cp .tmp/cellprofiler-human/examples-master/ExampleHuman/images/AS_09125_050116030001_D03f00d2.tif \
    "$bench/images/AS_09125_050116${id}_D03f00d2.tif"
done

/usr/bin/time -v bash -lc 'set -euo pipefail
  bench=target/cellprofiler-readme-bench-50
  i=0
  for img in "$bench"/images/*d0.tif; do
    target/release/cellprofiler-human \
      --input "$img" \
      --out "$bench/blockflow-output/run-${i}" \
      --min-size 50 --max-size 5027 \
      --sigma 1.0 --declump-sigma 1.3488 \
      --threshold-method li --threshold-bins 256 \
      --seed-min-distance 6 --maxima-downsample 3 \
      --declump-method intensity \
      --merge-line-basin-pixels 16 --merge-line-max-saddle-drop 0 >/dev/null
    i=$((i + 1))
  done'
```

The CellProfiler measurement used a non-hidden fixture directory because the
pipeline excludes hidden directories during image discovery:

```sh
cp .tmp/cellprofiler-human/examples-master/ExampleHuman/ExampleHuman.cppipe \
  target/cellprofiler-readme-bench-50/ExampleHuman.cppipe

docker run --rm \
  -v "$PWD/target/cellprofiler-readme-bench-50:/bench" \
  -w /bench \
  cellprofiler/cellprofiler:4.2.8 \
  -c -r -p ExampleHuman.cppipe -i images -o cellprofiler-output
```

## ImgLib2-Style Benchmark

The ImgLib2 comparison lives in `examples/imglib2-pipeline/`. It uses
deterministic 8-bit BMP fixtures and runs a generic image-processing pipeline:
Gaussian smoothing, Otsu thresholding, connected components, size filtering and
area/centroid export. The Java reference uses ImgLib2 core containers and
accessors plus Java ImageIO; it is intentionally not an ImageJ/CellProfiler
workflow clone.

Measured on 2026-09-17 on the same Intel Xeon Gold 6138 machine. The table
reports script wall time and summed in-process pipeline time from the per-image
JSON summaries. The comparison requires exact object count and foreground area
agreement.

| runner | batch | wall time | pipeline time sum | output |
|---|---:|---:|---:|---:|
| Blockflow `imglib2-pipeline` | 10 images | 0.155 s | 0.065 s | 49 objects / 23,232 px |
| ImgLib2 Java reference | 10 images | 2.924 s | 0.889 s | 49 objects / 23,232 px |
| Blockflow `imglib2-pipeline` | 50 images | 0.766 s | 0.343 s | 248 objects / 115,888 px |
| ImgLib2 Java reference | 50 images | 14.966 s | 4.616 s | 248 objects / 115,888 px |

Reproduction commands:

```sh
cargo build -p blockflow-imglib2-pipeline --release
examples/imglib2-pipeline/reference-imglib2/build.sh

examples/imglib2-pipeline/scripts/run_benchmark.sh 10 .tmp/imglib2-pipeline/bench-10
examples/imglib2-pipeline/scripts/run_benchmark.sh 50 .tmp/imglib2-pipeline/bench-50
```

## OpenCV-Style Benchmark

The OpenCV comparison lives in `examples/opencv-pipeline/`. It uses the same
deterministic BMP fixture family and runs a classical OpenCV-style workflow:
optional affine warp, Gaussian smoothing, Otsu thresholding, 3x3 open/close,
connected components, size filtering and area/centroid export. The OpenCV
reference is a C++ program built with CMake against OpenCV 4.5.4.

Measured on 2026-09-18 on the same Intel Xeon Gold 6138 machine. The table
reports script wall time and summed in-process pipeline time from per-image JSON
summaries. The comparison requires exact object count and allows up to 2%
foreground-area drift because the Gaussian and morphology implementations are
not bit-identical.

| mode | runner | batch | wall time | pipeline time sum | output |
|---|---|---:|---:|---:|---:|
| segment | Blockflow `opencv-pipeline` | 10 images | 0.216 s | 0.112 s | 49 objects / 23,203 px |
| segment | OpenCV C++ reference | 10 images | 1.806 s | 0.115 s | 49 objects / 22,959 px |
| segment | Blockflow `opencv-pipeline` | 50 images | 0.979 s | 0.570 s | 248 objects / 115,752 px |
| segment | OpenCV C++ reference | 50 images | 7.383 s | 0.482 s | 248 objects / 114,482 px |
| transform | Blockflow `opencv-pipeline` | 10 images | 0.193 s | 0.113 s | 49 objects / 24,112 px |
| transform | OpenCV C++ reference | 10 images | 1.525 s | 0.092 s | 49 objects / 23,885 px |
| transform | Blockflow `opencv-pipeline` | 50 images | 0.931 s | 0.551 s | 248 objects / 120,351 px |
| transform | OpenCV C++ reference | 50 images | 7.678 s | 0.505 s | 248 objects / 119,202 px |

Reproduction commands:

```sh
cargo build -p blockflow-opencv-pipeline --release
examples/opencv-pipeline/reference-opencv/build.sh

examples/opencv-pipeline/scripts/run_benchmark.sh 10 .tmp/opencv-pipeline/bench-10 segment
examples/opencv-pipeline/scripts/run_benchmark.sh 50 .tmp/opencv-pipeline/bench-50 segment
examples/opencv-pipeline/scripts/run_benchmark.sh 10 .tmp/opencv-pipeline/bench-10-transform transform
examples/opencv-pipeline/scripts/run_benchmark.sh 50 .tmp/opencv-pipeline/bench-50-transform transform
```


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
