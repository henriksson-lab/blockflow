# ImgLib2 Pipeline Example

This example benchmarks a small generic image-processing pipeline against an
ImgLib2 Java implementation:

1. load an 8-bit microscopy-like image,
2. Gaussian smooth,
3. Otsu threshold,
4. face-connected component labelling,
5. size filtering,
6. export object count, area and centroid rows.

The purpose is different from `examples/cellprofiler-human/`: this compares
library abstractions and execution overhead, not CellProfiler workflow parity.

## Build

From the workspace root:

```sh
cargo build -p blockflow-imglib2-pipeline --release
```

The Java reference is a Maven project:

```sh
examples/imglib2-pipeline/reference-imglib2/build.sh
```

## Run

Generate a deterministic 10-image fixture and run the Blockflow side:

```sh
examples/imglib2-pipeline/scripts/fetch_fixture.sh 10
examples/imglib2-pipeline/scripts/run_blockflow.sh \
  .tmp/imglib2-pipeline/images \
  .tmp/imglib2-pipeline/blockflow
```

Run the ImgLib2 side once the Java reference has been compiled:

```sh
examples/imglib2-pipeline/scripts/run_imglib2.sh \
  .tmp/imglib2-pipeline/images \
  .tmp/imglib2-pipeline/imglib2
```

The combined benchmark helper runs both sides and compares their summary CSVs:

```sh
examples/imglib2-pipeline/scripts/run_benchmark.sh 10
```

## Fixture Policy

Generated BMP fixtures and benchmark outputs live under `.tmp/imglib2-pipeline/`
and are not committed.
