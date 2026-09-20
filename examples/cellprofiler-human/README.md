# CellProfiler Human Example

This example package contains the CellProfiler-style HT29 nuclei benchmark and
the comparison helpers used by the root benchmark notes.

The user-facing Blockflow example path is planned execution. The resident binary
is kept only as a local reference/debug implementation for checking algorithmic
differences; it is not the usage pattern this crate should teach.

## Binaries

- `cellprofiler-human` builds, simulates and materializes the planned DAPI
  nuclei path. This is the primary Blockflow example binary.
- `cellprofiler-compare` compares Blockflow object tables with a CellProfiler
  reference table by semantic columns.
- `cellprofiler-resident-reference` runs the old resident DAPI nuclei path for internal
  debugging only.

For normal use, pass a prepared rank-3 `f64` Zarr array directly. The command
materializes labels and object rows by default:

```sh
cargo run -p blockflow-cellprofiler-human --bin cellprofiler-human --release -- \
  --input-zarr input.zarr/level0 --out run.json \
  --materialize-objects results
```

The benchmark uses this command after preparing its fixture array.

Build them from the workspace root:

```sh
cargo build -p blockflow-cellprofiler-human --release
```

Run the benchmark helper. It prepares or reuses a Zarr fixture at
`OUTPUT_DIR/input.zarr/level0` before the measured planned materialization,
writes the planned Blockflow object table to
`OUTPUT_DIR/blockflow/planned_objects.csv`, planned labels to
`OUTPUT_DIR/blockflow/labels.png`, and simulator/accounting output to
`OUTPUT_DIR/plan-probe.json`.

```sh
examples/cellprofiler-human/scripts/run_cellprofiler_benchmark.sh \
  .tmp/cellprofiler-human/examples-master/ExampleHuman/images/AS_09125_050116030001_D03f00d0.tif \
  .tmp/cellprofiler-human/blockflow-d0
```

## Fixture Policy

Downloaded CellProfiler data and generated CSV/PNG/JSON output are not kept in
this directory. Use `scripts/fetch_cellprofiler_human.sh` to populate
`.tmp/cellprofiler-human/` from the CellProfiler examples archive.

The planned benchmark path should run from prepared Zarr storage. Set
`BF_INPUT_ZARR=/path/to/store.zarr` to share a prepared fixture between runs; by
default the benchmark helper prepares `OUTPUT_DIR/input.zarr` from the input
image if it is missing.

To prepare the fixture explicitly before benchmarking:

```sh
examples/cellprofiler-human/scripts/prepare_input_zarr.sh \
  .tmp/cellprofiler-human/examples-master/ExampleHuman/images/AS_09125_050116030001_D03f00d0.tif \
  .tmp/cellprofiler-human/input.zarr
```
