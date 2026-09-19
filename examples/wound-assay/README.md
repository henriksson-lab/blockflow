# Wound Assay Example

This example measures open wound area from deterministic scratch-assay-like PGM
fixtures. It emits image-level summary rows and an aggregate per-column profile
that can be compared with scikit-image/imageio and OpenCV references.

The implementation is intentionally example-local. A general axis-profile helper
should only move into `blockflow` if another workflow needs the same operation.

## Build

```sh
cargo build -p blockflow-wound-assay --release
```

## Run

```sh
examples/wound-assay/scripts/run_blockflow.sh .tmp/wound-assay/blockflow 10
examples/wound-assay/scripts/run_reference.sh .tmp/wound-assay/reference 10
examples/wound-assay/scripts/run_benchmark.sh 10
examples/wound-assay/scripts/run_benchmark.sh 50
```

Generated outputs live under `.tmp/wound-assay/`.
