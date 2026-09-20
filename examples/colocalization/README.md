# Colocalization Example

This example attaches three prepared Zarr arrays and runs planned Blockflow
colocalization measurement on deterministic labelled two-channel fixtures. It reports
primitive sums plus derived Pearson, overlap and Manders values and compares
them with a NumPy/scikit-image-style Python reference.

## Build

```sh
cargo build -p blockflow-colocalization --release
```

## Run

```sh
examples/colocalization/scripts/run_blockflow.sh .tmp/colocalization/blockflow 10
examples/colocalization/scripts/run_reference.sh .tmp/colocalization/reference 10
examples/colocalization/scripts/run_benchmark.sh 10
examples/colocalization/scripts/run_benchmark.sh 50
```

Generated outputs live under `.tmp/colocalization/`.
`--zarr-dir` can point to an existing set of per-image arrays. The benchmark
prepares fixture arrays before timing the same processing command.
The measurement planner chooses the compute grid for the attached arrays;
`--chunk` controls storage chunks when preparing fixtures.
