# Colocalization Example

This example exercises the resident `blockflow` colocalization measurement
surface on deterministic labelled two-channel fixture files. It reports
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
