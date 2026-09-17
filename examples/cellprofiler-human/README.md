# CellProfiler Human Example

This example package contains the CellProfiler-style HT29 nuclei benchmark and
the comparison helpers used by the root README benchmark.

The code here is scenario wiring: CLIs, fixture scripts, CellProfiler reference
runs, sweeps and benchmark reporting. General image-processing operations and
measurement primitives should stay in the root `blockflow` crate.

## Binaries

- `cellprofiler-human` runs the resident DAPI nuclei path.
- `cellprofiler-compare` compares Blockflow object tables with a CellProfiler
  reference table by semantic columns.
- `cellprofiler-plan-probe` builds and simulates the planned segmentation path.

Build them from the workspace root:

```sh
cargo build -p blockflow-cellprofiler-human --release
```

Run the benchmark helper:

```sh
examples/cellprofiler-human/scripts/run_cellprofiler_benchmark.sh \
  .tmp/cellprofiler-human/examples-master/ExampleHuman/images/AS_09125_050116030001_D03f00d0.tif \
  .tmp/cellprofiler-human/blockflow-d0
```

## Fixture Policy

Downloaded CellProfiler data and generated CSV/PNG/JSON output are not kept in
this directory. Use `scripts/fetch_cellprofiler_human.sh` to populate
`.tmp/cellprofiler-human/` from the CellProfiler examples archive.
