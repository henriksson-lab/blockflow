# Benchmarks

These benchmark results are development measurements, not a stable public
performance contract. They compare current example pipelines against common
image-analysis frameworks and record the commands used to reproduce them.

## CellProfiler-Style Benchmark

One current benchmark is the CellProfiler ExampleHuman HT29 image set. To keep
startup overhead from dominating, this benchmark duplicates the small
three-channel example into 10-image and 50-image batches. The Blockflow rows
below are the release `cellprofiler-human` binary running the current DAPI
nuclei path over the DAPI images. The CellProfiler rows are
`cellprofiler/cellprofiler:4.2.8` running the downloaded `ExampleHuman.cppipe`
pipeline in Docker over the same three-channel image sets.

Measured on 2026-09-17 on an Intel Xeon Gold 6138 machine. Compile time is not
included. The CellProfiler run uses a warm local Docker image and reports peak
container memory sampled with `docker stats`; Blockflow RSS is Linux
`/usr/bin/time -v` max RSS.

| runner | scope | wall time | max RSS / peak memory | output |
|---|---:|---:|---:|---:|
| Blockflow `target/release/cellprofiler-human` | 10 DAPI nuclei runs | 1.09 s | 23,708 KiB / 23.2 MiB | 2,880 nuclei |
| CellProfiler 4.2.8 Docker | 10 full ExampleHuman image sets | 54.93 s | 504.4 MiB | 2,890 nuclei |
| Blockflow `target/release/cellprofiler-human` | 50 DAPI nuclei runs | 5.17 s | 23,928 KiB / 23.4 MiB | 14,400 nuclei |
| CellProfiler 4.2.8 Docker | 50 full ExampleHuman image sets | 238.13 s | 671.2 MiB | 14,450 nuclei |

That is about a 50x wall-time difference on 10 image sets and 46x on 50 image
sets, with about 22x lower peak memory on 10 image sets and 29x lower peak
memory on 50 image sets for the current Blockflow path. This is a
pipeline-reference benchmark, not a claim of exact operation-for-operation
parity: the CellProfiler pipeline also identifies secondary/tertiary objects
and exports more measurements. The current semantic comparison is documented in
`CELLPROFILER.md`; the remaining one-object difference is treated as expected
reference drift for this milestone.

Reproduction commands:

```sh
cargo build -p blockflow-cellprofiler-human --release --bin cellprofiler-human

N=50
bench=target/cellprofiler-readme-bench-${N}
mkdir -p "$bench/images"
for n in $(seq 0 $((N - 1))); do
  id=$(printf "%02d" "$n")
  cp .tmp/cellprofiler-human/examples-master/ExampleHuman/images/AS_09125_050116030001_D03f00d0.tif \
    "$bench/images/AS_09125_050116${id}_D03f00d0.tif"
  cp .tmp/cellprofiler-human/examples-master/ExampleHuman/images/AS_09125_050116030001_D03f00d1.tif \
    "$bench/images/AS_09125_050116${id}_D03f00d1.tif"
  cp .tmp/cellprofiler-human/examples-master/ExampleHuman/images/AS_09125_050116030001_D03f00d2.tif \
    "$bench/images/AS_09125_050116${id}_D03f00d2.tif"
done

/usr/bin/time -v bash -lc 'set -euo pipefail
  bench=target/cellprofiler-readme-bench-50
  i=0
  for img in "$bench"/images/*d0.tif; do
    target/release/cellprofiler-human \
      --input "$img" \
      --out "$bench/blockflow-output/run-${i}" \
      --min-size 50 --max-size 5027 \
      --sigma 1.0 --declump-sigma 1.3488 \
      --threshold-method li --threshold-bins 256 \
      --seed-min-distance 6 --maxima-downsample 3 \
      --declump-method intensity \
      --merge-line-basin-pixels 16 --merge-line-max-saddle-drop 0 >/dev/null
    i=$((i + 1))
  done'
```

The CellProfiler measurement used a non-hidden fixture directory because the
pipeline excludes hidden directories during image discovery:

```sh
cp .tmp/cellprofiler-human/examples-master/ExampleHuman/ExampleHuman.cppipe \
  target/cellprofiler-readme-bench-50/ExampleHuman.cppipe

docker run --rm \
  -v "$PWD/target/cellprofiler-readme-bench-50:/bench" \
  -w /bench \
  cellprofiler/cellprofiler:4.2.8 \
  -c -r -p ExampleHuman.cppipe -i images -o cellprofiler-output
```

## ImgLib2-Style Benchmark

The ImgLib2 comparison lives in `examples/imglib2-pipeline/`. It uses
deterministic 8-bit BMP fixtures and runs a generic image-processing pipeline:
Gaussian smoothing, Otsu thresholding, connected components, size filtering and
area/centroid export. The Java reference uses ImgLib2 core containers and
accessors plus Java ImageIO; it is intentionally not an ImageJ/CellProfiler
workflow clone.

Measured on 2026-09-17 on the same Intel Xeon Gold 6138 machine. The table
reports script wall time and summed in-process pipeline time from the per-image
JSON summaries. The comparison requires exact object count and foreground area
agreement.

| runner | batch | wall time | pipeline time sum | output |
|---|---:|---:|---:|---:|
| Blockflow `imglib2-pipeline` | 10 images | 0.155 s | 0.065 s | 49 objects / 23,232 px |
| ImgLib2 Java reference | 10 images | 2.924 s | 0.889 s | 49 objects / 23,232 px |
| Blockflow `imglib2-pipeline` | 50 images | 0.766 s | 0.343 s | 248 objects / 115,888 px |
| ImgLib2 Java reference | 50 images | 14.966 s | 4.616 s | 248 objects / 115,888 px |

Reproduction commands:

```sh
cargo build -p blockflow-imglib2-pipeline --release
examples/imglib2-pipeline/reference-imglib2/build.sh

examples/imglib2-pipeline/scripts/run_benchmark.sh 10 .tmp/imglib2-pipeline/bench-10
examples/imglib2-pipeline/scripts/run_benchmark.sh 50 .tmp/imglib2-pipeline/bench-50
```

## OpenCV-Style Benchmark

The OpenCV comparison lives in `examples/opencv-pipeline/`. It uses the same
deterministic BMP fixture family and runs a classical OpenCV-style workflow:
optional affine warp, Gaussian smoothing, Otsu thresholding, 3x3 open/close,
connected components, size filtering and area/centroid export. The OpenCV
reference is a C++ program built with CMake against OpenCV 4.5.4.

Measured on 2026-09-18 on the same Intel Xeon Gold 6138 machine. The table
reports script wall time and summed in-process pipeline time from per-image JSON
summaries. The comparison requires exact object count and allows up to 2%
foreground-area drift because the Gaussian and morphology implementations are
not bit-identical.

| mode | runner | batch | wall time | pipeline time sum | output |
|---|---|---:|---:|---:|---:|
| segment | Blockflow `opencv-pipeline` | 10 images | 0.216 s | 0.112 s | 49 objects / 23,203 px |
| segment | OpenCV C++ reference | 10 images | 1.806 s | 0.115 s | 49 objects / 22,959 px |
| segment | Blockflow `opencv-pipeline` | 50 images | 0.979 s | 0.570 s | 248 objects / 115,752 px |
| segment | OpenCV C++ reference | 50 images | 7.383 s | 0.482 s | 248 objects / 114,482 px |
| transform | Blockflow `opencv-pipeline` | 10 images | 0.193 s | 0.113 s | 49 objects / 24,112 px |
| transform | OpenCV C++ reference | 10 images | 1.525 s | 0.092 s | 49 objects / 23,885 px |
| transform | Blockflow `opencv-pipeline` | 50 images | 0.931 s | 0.551 s | 248 objects / 120,351 px |
| transform | OpenCV C++ reference | 50 images | 7.678 s | 0.505 s | 248 objects / 119,202 px |

Reproduction commands:

```sh
cargo build -p blockflow-opencv-pipeline --release
examples/opencv-pipeline/reference-opencv/build.sh

examples/opencv-pipeline/scripts/run_benchmark.sh 10 .tmp/opencv-pipeline/bench-10 segment
examples/opencv-pipeline/scripts/run_benchmark.sh 50 .tmp/opencv-pipeline/bench-50 segment
examples/opencv-pipeline/scripts/run_benchmark.sh 10 .tmp/opencv-pipeline/bench-10-transform transform
examples/opencv-pipeline/scripts/run_benchmark.sh 50 .tmp/opencv-pipeline/bench-50-transform transform
```

## scikit-image/SciPy Benchmark

The scikit-image comparison lives in `examples/skimage-pipeline/`. It uses the
same deterministic BMP fixture family and runs the same segment/transform
workflow as the OpenCV example: optional affine translation, Gaussian smoothing,
Otsu thresholding, 3x3 open/close, connected components, size filtering and
area/centroid export. The reference is a Python script using NumPy 2.4.2,
SciPy 1.15.2, scikit-image 0.26.0 and imageio 2.37.4.

Measured on 2026-09-18 on the same Intel Xeon Gold 6138 machine. The table
reports script wall time and summed in-process pipeline time from per-image JSON
summaries. The comparison requires exact object count and allows up to 2%
foreground-area drift because the Gaussian and morphology implementations are
not bit-identical.

| mode | runner | batch | wall time | pipeline time sum | output |
|---|---|---:|---:|---:|---:|
| segment | Blockflow `skimage-pipeline` | 10 images | 0.219 s | 0.130 s | 49 objects / 23,203 px |
| segment | scikit-image/SciPy reference | 10 images | 5.052 s | 0.186 s | 49 objects / 23,319 px |
| segment | Blockflow `skimage-pipeline` | 50 images | 1.017 s | 0.603 s | 248 objects / 115,752 px |
| segment | scikit-image/SciPy reference | 50 images | 25.256 s | 0.917 s | 248 objects / 116,670 px |
| transform | Blockflow `skimage-pipeline` | 10 images | 0.214 s | 0.129 s | 49 objects / 24,112 px |
| transform | scikit-image/SciPy reference | 10 images | 5.086 s | 0.182 s | 49 objects / 24,044 px |
| transform | Blockflow `skimage-pipeline` | 50 images | 0.968 s | 0.572 s | 248 objects / 120,351 px |
| transform | scikit-image/SciPy reference | 50 images | 25.218 s | 0.904 s | 248 objects / 120,012 px |

Reproduction commands:

```sh
cargo build -p blockflow-skimage-pipeline --release

examples/skimage-pipeline/scripts/run_benchmark.sh 10 .tmp/skimage-pipeline/bench-10 segment
examples/skimage-pipeline/scripts/run_benchmark.sh 50 .tmp/skimage-pipeline/bench-50 segment
examples/skimage-pipeline/scripts/run_benchmark.sh 10 .tmp/skimage-pipeline/bench-10-transform transform
examples/skimage-pipeline/scripts/run_benchmark.sh 50 .tmp/skimage-pipeline/bench-50-transform transform
```

## Dask-image Benchmark

The Dask-image comparison lives in `examples/dask-image-pipeline/`. It uses the
same pipeline shape as the scikit-image benchmark, but on larger 1024x1024
fixtures and with explicit Dask chunk sizes. The reference script uses Dask
2026.8.0, dask-image 2026.5.0, NumPy 2.5.3 and SciPy 1.18.1 installed into
`.tmp/dask-image-deps`.

Measured on 2026-09-18 on the same Intel Xeon Gold 6138 machine. The table
reports script wall time and summed in-process pipeline time from per-image JSON
summaries. The comparison requires exact object count and allows up to 2%
foreground-area drift because chunked Gaussian/morphology paths are not
bit-identical.

| runner | batch | chunk | wall time | pipeline time sum | output |
|---|---:|---:|---:|---:|---:|
| Blockflow `dask-image-pipeline` | 2 x 1024x1024 | n/a | 0.271 s | 0.202 s | 160 objects / 73,837 px |
| Dask-image reference | 2 x 1024x1024 | 256x256 | 2.738 s | 0.946 s | 160 objects / 74,992 px |
| Blockflow `dask-image-pipeline` | 2 x 1024x1024 | n/a | 0.254 s | 0.205 s | 160 objects / 73,837 px |
| Dask-image reference | 2 x 1024x1024 | 512x512 | 2.306 s | 0.580 s | 160 objects / 74,992 px |

Reproduction commands:

```sh
python3 -m pip install --target .tmp/dask-image-deps dask dask-image
cargo build -p blockflow-dask-image-pipeline --release

examples/dask-image-pipeline/scripts/run_benchmark.sh 2 .tmp/dask-image-pipeline/bench-2-c256 segment 256x256
examples/dask-image-pipeline/scripts/run_benchmark.sh 2 .tmp/dask-image-pipeline/bench-2-c512 segment 512x512
```

## 3-D Object Measurement Benchmark

The 3-D object measurement comparison lives in
`examples/object-3d-measurement/`. It uses deterministic labelled 3-D box
fixtures stored as CSV box specifications. Blockflow measures the label volumes
with `object_geometry_basic_measurements_u32`; the scikit-image reference
reconstructs the same label volumes and uses `regionprops`. This benchmark is
now intentionally a cheap-geometry benchmark: object count, bbox, physical bbox
extent, and voxel count. Feret-like measurements should be benchmarked as
separate rows because exact voxel-pair Feret is quadratic and directional Feret
has different estimate semantics.

Measured on 2026-09-18 on the same Intel Xeon Gold 6138 machine. Compile time is
not included in the timed rows. Wall time and RSS are from `/usr/bin/time -v`.
The compute column is Blockflow `measurement_seconds` and scikit-image
`pipeline_seconds` from each summary JSON. The comparison requires exact
agreement for shared CSV columns.

| runner | batch | wall time | compute time | max RSS | output |
|---|---:|---:|---:|---:|---:|
| Blockflow `object-3d-measurement` u32 basic geometry | 10 images | 0.04 s | 0.0119 s | 2,880 KiB | 40 objects / 29,802 voxels |
| scikit-image/SciPy `regionprops` basic geometry | 10 images | 0.94 s | 0.4408 s | 70,580 KiB | 40 objects / 29,802 voxels |
| Blockflow `object-3d-measurement` u32 basic geometry | 50 images | 0.04 s | 0.0309 s | 2,880 KiB | 200 objects / 149,502 voxels |
| scikit-image/SciPy `regionprops` basic geometry | 50 images | 0.60 s | 0.3141 s | 70,720 KiB | 200 objects / 149,502 voxels |

Reproduction commands:

```sh
examples/object-3d-measurement/scripts/run_benchmark.sh 10 .tmp/object-3d-measurement/u32-streaming-10
examples/object-3d-measurement/scripts/run_benchmark.sh 50 .tmp/object-3d-measurement/final-object-geometry-perf-50
```

## Colocalization Benchmark

The colocalization comparison lives in `examples/colocalization/`. It uses
deterministic labelled two-channel fixture specifications. Blockflow computes
the rows with `colocalization_measurements`; the Python reference computes the
same labelled reductions with NumPy-style loops.

Measured on 2026-09-18 on the same Intel Xeon Gold 6138 machine. Compile time is
not included in the timed rows. Wall time and RSS are from `/usr/bin/time -v`.
The comparison requires exact CSV agreement after fixed decimal formatting.

| runner | batch | wall time | max RSS | output |
|---|---:|---:|---:|---:|
| Blockflow `colocalization` | 10 images | 0.03 s | 2,880 KiB | 40 objects / 20,508 pixel pairs |
| NumPy/scikit-image-style reference | 10 images | 0.09 s | 13,440 KiB | 40 objects / 20,508 pixel pairs |
| Blockflow `colocalization` | 50 images | 0.10 s | 2,880 KiB | 200 objects / 102,906 pixel pairs |
| NumPy/scikit-image-style reference | 50 images | 0.17 s | 13,760 KiB | 200 objects / 102,906 pixel pairs |

Reproduction commands:

```sh
examples/colocalization/scripts/run_benchmark.sh 10 .tmp/colocalization/p102-final-10
examples/colocalization/scripts/run_benchmark.sh 50 .tmp/colocalization/p102-final-50
```

## Wound Assay Benchmark

The wound assay comparison lives in `examples/wound-assay/`. It uses
deterministic 8-bit PGM scratch-assay fixtures. Blockflow, scikit-image/imageio
and OpenCV all apply the same threshold-only open-area/profile measurement.

Measured on 2026-09-18 on the same Intel Xeon Gold 6138 machine. Compile time is
not included in the timed rows. Wall time and RSS are from `/usr/bin/time -v`.
The comparison requires exact image-summary and profile CSV agreement.

| runner | batch | wall time | max RSS | output |
|---|---:|---:|---:|---:|
| Blockflow `wound-assay` | 10 images | 0.01 s | 2,880 KiB | 29,370 open pixels |
| scikit-image/imageio reference | 10 images | 0.26 s | 38,080 KiB | 29,370 open pixels |
| OpenCV reference | 10 images | 0.27 s | 60,160 KiB | 29,370 open pixels |
| Blockflow `wound-assay` | 50 images | 0.02 s | 2,880 KiB | 148,421 open pixels |
| scikit-image/imageio reference | 50 images | 0.28 s | 37,760 KiB | 148,421 open pixels |
| OpenCV reference | 50 images | 0.32 s | 59,440 KiB | 148,421 open pixels |

Reproduction commands:

```sh
examples/wound-assay/scripts/run_benchmark.sh 10 .tmp/wound-assay/p102-final-10
examples/wound-assay/scripts/run_benchmark.sh 50 .tmp/wound-assay/p102-final-50
```
