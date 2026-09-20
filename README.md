# blockflow

**Ready to be used by early adopters. Have your code depend on a particular git commit as API is not yet frozen**

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
* A 6d scheduler figures out the best order and adapts in realtime based on statistics
* Designed for multiple compute nodes, GPUs and heterogenous compute environments from day one
* OME-Zarr is used to enable distributed computing on chunks of image data

## Benchmarks

In the [current example benchmarks](BENCHMARKS.md), Blockflow's planned Zarr
pipelines were approximately:

* **5× faster than ImgLib2** on 10-image segmentation batches
* **4.5× faster than OpenCV** on the same batches
* **9× faster than scikit-image** on the same batches
* **3× faster than Dask-image** on two larger images

These are single-run wall-time ratios for the specified workloads, excluding
fixture preparation and compilation. See [BENCHMARKS.md](BENCHMARKS.md) for
the measurements, output checks, and reproduction commands. Earlier results
are archived in [OLD_BENCHMARKS.md](OLD_BENCHMARKS.md).

## Design notes

Details about the design are located in [`docs/design/`](docs/design) ; docs need cleaning

* [dimensions and modules](docs/design/dimensions-and-modules.md)
* [writing an op](docs/design/writing-an-op.md)
* [executing a run](docs/design/executing-a-run.md)
* [images and phases](docs/design/images-and-phases.md) 


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

MIT (but note that the code is AI generated so no guarantees about provenance)
