# Foci Per Nucleus Example

This example counts bright puncta/foci inside segmented nuclei. It is a
benchmark-oriented workflow example, not a new library API: the point is to see
whether containment and child-count glue remains readable at the example layer.
The Blockflow side prepares per-image labels and focus images as Zarr and runs
planned shape/intensity reductions over those attached stores.

## Build

From the workspace root:

```sh
cargo build -p blockflow-foci-per-nucleus --release
```

## Run

Run the Blockflow side:

```sh
examples/foci-per-nucleus/scripts/run_blockflow.sh \
  .tmp/foci-per-nucleus/blockflow \
  10 \
  .tmp/foci-per-nucleus/input.zarr
```

Run the Python reference:

```sh
examples/foci-per-nucleus/scripts/run_reference.sh \
  .tmp/foci-per-nucleus/reference \
  10
```

Run both and compare:

```sh
examples/foci-per-nucleus/scripts/run_benchmark.sh 10
examples/foci-per-nucleus/scripts/run_benchmark.sh 50
```

Generated outputs live under `.tmp/foci-per-nucleus/`.
Benchmark runs place prepared Zarr inputs under the benchmark directory as
`input.zarr`.
The fixture arrays are prepared before the planned measurement run. Labels and
intensities are measured directly from attached arrays, with a planner-selected
compute grid; `--chunk` controls fixture storage chunks.
