# Dask-image Pipeline Example

This example benchmarks a larger chunked image-processing pipeline against a
Dask-image/SciPy Python implementation:

1. load an 8-bit microscopy-like image,
2. optionally apply a nearest-neighbour affine translation in `transform` mode,
3. Gaussian smooth,
4. Otsu threshold,
5. 3x3 binary open/close,
6. connected-components with size filtering,
7. export object count, area and centroid rows.

The purpose is different from `examples/skimage-pipeline/`: this compares
chunked/lazy execution overhead and chunk-size effects, not just Python library
overhead on small image batches.

## Build

From the workspace root:

```sh
cargo build -p blockflow-dask-image-pipeline --release
```

The Dask-image reference is `reference-dask-image/dask_image_pipeline.py` and
uses `dask`, `dask-image`, `numpy`, `scipy`, `scikit-image` and `imageio`.
The benchmark scripts default to `DASK_IMAGE_DEPS=.tmp/dask-image-deps`, where
the dependencies can be installed with:

```sh
python3 -m pip install --target .tmp/dask-image-deps dask dask-image
```

## Run

The Blockflow binary reads a Zarr array directly with `--input-zarr`. For a
multiscale OME-Zarr store, pass the selected rank-3 level directory, such as
`image.zarr/0`, and use `--channel N` for a `[channel, y, x]` array. `--zarr-dir` names the prepared input store when converting a fixture. The benchmark
script converts BMP fixtures with `--prepare-only` before timing the normal
`--input-zarr` run.

Generate a deterministic two-image, 1024x1024 fixture and run the Blockflow
side:

```sh
examples/dask-image-pipeline/scripts/fetch_fixture.sh 2
examples/dask-image-pipeline/scripts/run_blockflow.sh \
  .tmp/dask-image-pipeline/images \
  .tmp/dask-image-pipeline/blockflow
```

Run the Dask-image side:

```sh
examples/dask-image-pipeline/scripts/run_dask_image.sh \
  .tmp/dask-image-pipeline/images \
  .tmp/dask-image-pipeline/dask-image
```

The combined benchmark helper runs both sides and compares their summary CSVs:

```sh
examples/dask-image-pipeline/scripts/run_benchmark.sh 2 .tmp/dask-image-pipeline/bench-2-c256 segment 256x256
examples/dask-image-pipeline/scripts/run_benchmark.sh 2 .tmp/dask-image-pipeline/bench-2-c512 segment 512x512
```

`BF_MODE=segment` is the default. Use the fourth `run_benchmark.sh` argument or
`BF_CHUNK` to choose the Dask chunk shape.

## Fixture Policy

Generated BMP fixtures and benchmark outputs live under `.tmp/dask-image-pipeline/`
and are not committed.
