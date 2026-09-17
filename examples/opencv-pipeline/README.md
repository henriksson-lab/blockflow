# OpenCV Pipeline Example

This example benchmarks a small generic image-processing pipeline against an
OpenCV C++ implementation:

1. load an 8-bit microscopy-like image,
2. optionally apply a nearest-neighbour affine translation in `transform` mode,
3. Gaussian smooth,
4. Otsu threshold,
5. 3x3 binary open/close,
6. connected-components with size filtering,
7. export object count, area and centroid rows.

The purpose is different from `examples/cellprofiler-human/`: this compares
library abstractions and execution overhead, not CellProfiler workflow parity.

## Build

From the workspace root:

```sh
cargo build -p blockflow-opencv-pipeline --release
```

The OpenCV reference is a CMake C++ project:

```sh
examples/opencv-pipeline/reference-opencv/build.sh
```

## Run

Generate a deterministic 10-image fixture and run the Blockflow side:

```sh
examples/opencv-pipeline/scripts/fetch_fixture.sh 10
examples/opencv-pipeline/scripts/run_blockflow.sh \
  .tmp/opencv-pipeline/images \
  .tmp/opencv-pipeline/blockflow
```

Run the OpenCV side once the Java reference has been compiled:

```sh
examples/opencv-pipeline/scripts/run_opencv.sh \
  .tmp/opencv-pipeline/images \
  .tmp/opencv-pipeline/opencv
```

The combined benchmark helper runs both sides and compares their summary CSVs:

```sh
examples/opencv-pipeline/scripts/run_benchmark.sh 10
examples/opencv-pipeline/scripts/run_benchmark.sh 10 .tmp/opencv-pipeline/bench-10-transform transform
```

`BF_MODE=segment` is the default. Use `BF_MODE=transform` or pass `transform`
as the third `run_benchmark.sh` argument to include the affine warp step.

## Fixture Policy

Generated BMP fixtures and benchmark outputs live under `.tmp/opencv-pipeline/`
and are not committed.
