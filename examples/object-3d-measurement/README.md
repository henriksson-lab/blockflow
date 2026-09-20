# 3-D Object Measurement Example

This example measures labelled 3-D objects with physical voxel spacing. It uses
the planned Blockflow basic object-geometry measurement surface and compares
count, bounding boxes, physical extents, and voxel totals with a scikit-image
`regionprops` reference over the same fixture files.

Feret-like measurements are intentionally not part of this benchmark. Exact
voxel-pair Feret and directional Feret estimates have different costs and
semantics, so they should be measured as separate benchmark rows.

## Build

```sh
cargo build -p blockflow-object-3d-measurement --release
```

## Run

```sh
examples/object-3d-measurement/scripts/run_blockflow.sh .tmp/object-3d-measurement/blockflow 10
examples/object-3d-measurement/scripts/run_reference.sh .tmp/object-3d-measurement/reference 10
examples/object-3d-measurement/scripts/run_benchmark.sh 10
examples/object-3d-measurement/scripts/run_benchmark.sh 50
```

Generated outputs live under `.tmp/object-3d-measurement/`.
`--zarr-dir` can point to prepared per-image label arrays. The benchmark
prepares these arrays before timing the normal processing command.
The measurement planner chooses the compute grid for attached labels;
`--chunk` controls storage chunks when preparing fixtures.
