# Benchmarks

These results were measured on 2026-09-20 on an Intel Xeon Gold 6138 host.
They replace the results in [OLD_BENCHMARKS.md](OLD_BENCHMARKS.md). Rows are
single runs unless a median is stated; small wall-time differences may change
on another run or machine. Release builds and fixture conversion were completed
before timing Blockflow. The Blockflow commands read prepared rank-3 Zarr
arrays, build plans, execute them, and write their reported results.
References used OpenCV C++ 4.5.4, OpenJDK 19 for ImgLib2, NumPy 2.4.2,
SciPy 1.15.2, scikit-image 0.26.0, imageio 2.37.4, and OpenCV Python 5.0.0.
The separate Dask environment used Dask 2026.8.0, dask-image 2026.5.0,
NumPy 2.5.3, and SciPy 1.18.1.

## Image-processing pipelines

The scripts below generated deterministic BMP fixtures, prepared Blockflow's
Zarr inputs, ran each compiled binary over the batch, then checked object count
and foreground area against the reference. **Wall** is the elapsed time for
the per-image processing loop, including process startup and output writes but
excluding compilation and fixture preparation. **Pipeline** is the sum of the
in-process `pipeline_seconds` values in the per-image summaries. The two
implementations' pipeline timers are useful for diagnosis; their work scopes
are not guaranteed to be identical.

| Pipeline and mode | Images | Blockflow wall / pipeline | Reference wall / pipeline | Objects (both) | Foreground pixels (Blockflow / reference) |
|---|---:|---:|---:|---:|---:|
| ImgLib2, segment | 10 × 256² | 0.660 / 0.300 s | 3.267 / 0.893 s | 49 | 23,352 / 23,232 |
| OpenCV, segment | 10 × 256² | 0.570 / 0.338 s | 2.576 / 0.133 s | 49 | 23,319 / 22,959 |
| OpenCV, transform | 10 × 256² | 0.592 / 0.401 s | 2.126 / 0.131 s | 49 | 24,044 / 23,885 |
| scikit-image, segment | 10 × 256² | 0.607 / 0.341 s | 5.502 / 0.194 s | 49 | 23,319 / 23,319 |
| scikit-image, transform | 10 × 256² | 0.638 / 0.450 s | 6.548 / 0.272 s | 49 | 24,044 / 24,044 |
| Dask-image, 256² reference chunks | 2 × 1024² | 0.895 / 0.798 s | 2.701 / 0.880 s | 160 | 74,992 / 74,992 |
| Dask-image, 512² reference chunks | 2 × 1024² | 0.787 / 0.732 s | 2.231 / 0.548 s | 160 | 74,992 / 74,992 |

The ImgLib2 comparison allows at most 1% area difference; OpenCV and
scikit-image allow 2%. The Dask-image rows match area exactly. Every comparison
passed its configured object-count and area checks. The Dask reference chunk
size changes only the Dask side; Blockflow's prepared Zarr chunks and planned
grid are the same in both Dask rows.

Reproduce these rows from the repository root:

```sh
examples/imglib2-pipeline/scripts/run_benchmark.sh 10 .tmp/current-bench/imglib2-10
examples/opencv-pipeline/scripts/run_benchmark.sh 10 .tmp/current-bench/opencv-10 segment
examples/opencv-pipeline/scripts/run_benchmark.sh 10 .tmp/current-bench/opencv-transform-10 transform
examples/skimage-pipeline/scripts/run_benchmark.sh 10 .tmp/current-bench/skimage-10 segment
examples/skimage-pipeline/scripts/run_benchmark.sh 10 .tmp/current-bench/skimage-transform-10 transform
examples/dask-image-pipeline/scripts/run_benchmark.sh 2 .tmp/current-bench/dask-2 segment 256x256
examples/dask-image-pipeline/scripts/run_benchmark.sh 2 .tmp/current-bench/dask-512-2 segment 512x512
```

The Java reference needs Maven and ImgLib2 dependencies, the OpenCV reference
needs OpenCV development libraries, and the Dask reference uses the packages
under `.tmp/dask-image-deps` (override with `DASK_IMAGE_DEPS`). Each script
writes its batch summaries and comparison result under the named directory.

## Measurement workflows

These workflows read previously prepared Zarr arrays through
`ZarrEnvironment::attach`. The labelled measurement examples use
planner-selected measurement grids; the wound assay plans its pixel phase.
The Blockflow **wall** and **RSS** figures below are from `/usr/bin/time -v`
around the release executable, after the corresponding `run_benchmark.sh`
prepared its inputs and checked the outputs. Reference wall and RSS are from
the Python reference scripts. These small runs include process startup and CSV
writes. RSS is the maximum resident set size in KiB.

| Workflow | Images | Blockflow wall / RSS | Reference wall / RSS | Checked output |
|---|---:|---:|---:|---|
| 3-D object geometry / scikit-image | 10 | 0.34 s / 10,916 KiB | 0.47 s / 70,888 KiB | 40 objects; 29,802 voxels |
| Colocalization / Python | 10 | 0.10 s / 9,280 KiB | 0.09 s / 13,440 KiB | 40 objects; 20,508 pixel pairs |
| Wound assay / scikit-image | 10 | 0.04 s / 8,320 KiB | 0.26 s / 38,080 KiB | 29,370 open pixels |
| Wound assay / OpenCV | 10 | 0.04 s / 8,320 KiB | 0.27 s / 59,840 KiB | 29,370 open pixels |
| Percent positive / Python | 10 | 0.09 s / 9,600 KiB | 0.09 s / 13,120 KiB | 60 objects; 13 positive |
| Foci per nucleus / Python | 10 | 0.27 s / 10,240 KiB | 0.08 s / 13,760 KiB | 50 nuclei; 119 foci |

Object geometry, wound, percent positive, and foci CSVs matched their
references exactly. Colocalization matched image, label, and count fields
exactly; floating columns were checked with an absolute tolerance of
`2e-5` because the planned reductions accumulate in a different order.

Reproduce the preparation and correctness checks with:

```sh
examples/object-3d-measurement/scripts/run_benchmark.sh 10 .tmp/current-bench/object-10
examples/colocalization/scripts/run_benchmark.sh 10 .tmp/current-bench/coloc-10
examples/wound-assay/scripts/run_benchmark.sh 10 .tmp/current-bench/wound-10
examples/percent-positive/scripts/run_benchmark.sh 10 .tmp/current-bench/percent-10
examples/foci-per-nucleus/scripts/run_benchmark.sh 10 .tmp/current-bench/foci-10
```

For the Blockflow wall and RSS entries, time the release executable directly
after preparation. For example:

```sh
/usr/bin/time -v -o .tmp/current-bench/object-10/blockflow-direct-time.txt \
  target/release/object-3d-measurement \
  --out .tmp/current-bench/object-10/blockflow-direct --images 10 \
  --fixture-dir .tmp/current-bench/object-10/fixtures \
  --zarr-dir .tmp/current-bench/object-10/input.zarr
```

The other direct commands use the same `--images`, `--out`, and `--zarr-dir`
arguments; colocalization and wound assay also use `--fixture-dir`. The
percent-positive and foci Python reference figures were timed separately with
`/usr/bin/time -v` around their `scripts/run_reference.sh` commands.
The foci executable also constructs deterministic focus specifications for its
`foci.csv` while measuring the prepared arrays.

## CellProfiler ExampleHuman

Ten copies of the ExampleHuman HT29 three-channel image set were used. The
Blockflow run reads the prepared DAPI Zarr array, executes its planned nuclei
workflow, and materializes labels and object rows. The CellProfiler 4.2.8
Docker run executes the full supplied three-channel pipeline, including
secondary and tertiary objects and additional measurements. These are
**different scopes**; the times are recorded for reproducibility rather than
as a like-for-like speed ratio.

| Runner | Timed scope | Wall | DAPI nuclei |
|---|---|---:|---:|
| Blockflow `cellprofiler-human` | 10 planned executions and materializations | 4.04 s | 2,880 |
| CellProfiler 4.2.8 | 10 full three-channel pipeline runs in Docker | 60.66 s | 2,890 |

Blockflow's ten materialization summaries total 3.762 s inside the 4.04 s
process-loop wall time, which also includes plan simulation. The CellProfiler
count is the number of data rows in
`Nuclei.csv`. The ten-object difference is an output difference, so this row
does not claim exact semantic parity. The Blockflow process peaked at 38,624
KiB RSS; `/usr/bin/time` around `docker run` measures the Docker client rather
than container memory, so no memory comparison is made.

The batch was prepared from the ExampleHuman files fetched by
`examples/cellprofiler-human/scripts/fetch_cellprofiler_human.sh`. After building
`cellprofiler-human --release`, make the ten-image batch and prepare its Zarr
inputs before starting the timer:

```sh
bench=target/current-bench/cellprofiler-10
source_dir=.tmp/cellprofiler-human/examples-master/ExampleHuman/images
mkdir -p "$bench/images"
for n in $(seq 0 9); do
  id=$(printf '%02d' "$n")
  for channel in 0 1 2; do
    cp "$source_dir/AS_09125_050116030001_D03f00d${channel}.tif" \
      "$bench/images/AS_09125_050116${id}_D03f00d${channel}.tif"
  done
  target/release/cellprofiler-human \
    --input "$bench/images/AS_09125_050116${id}_D03f00d0.tif" \
    --ensure-input-zarr "$bench/prepared/run-${n}/input.zarr" \
    --prepare-only --out "$bench/prepared/run-${n}/prepare.json"
done
cp .tmp/cellprofiler-human/examples-master/ExampleHuman/ExampleHuman.cppipe \
  "$bench/ExampleHuman.cppipe"
```

The exact Blockflow processing command for each prepared array was:

```sh
target/release/cellprofiler-human \
  --input-zarr target/current-bench/cellprofiler-10/prepared/run-0/input.zarr/level0 \
  --out target/current-bench/cellprofiler-10/blockflow/run-0/plan.json \
  --chunk 1x256x256 --workers 1 --cache-bytes 0 \
  --min-size 50 --max-size 5027 --sigma 1.0 --declump-sigma 1.3488 \
  --threshold-method li --threshold-bins 256 \
  --seed-min-distance 6 --maxima-downsample 3 --declump-method intensity \
  --merge-line-basin-pixels 16 --merge-line-max-saddle-drop 0 \
  --materialize-objects target/current-bench/cellprofiler-10/blockflow/run-0
```

The same command was run for `run-0` through `run-9`, timed together with
`/usr/bin/time -v` as follows:

```sh
/usr/bin/time -v -o "$bench/blockflow-time.txt" bash -c '
  for n in $(seq 0 9); do
    target/release/cellprofiler-human \
      --input-zarr "target/current-bench/cellprofiler-10/prepared/run-${n}/input.zarr/level0" \
      --out "target/current-bench/cellprofiler-10/blockflow/run-${n}/plan.json" \
      --chunk 1x256x256 --workers 1 --cache-bytes 0 \
      --min-size 50 --max-size 5027 --sigma 1.0 --declump-sigma 1.3488 \
      --threshold-method li --threshold-bins 256 \
      --seed-min-distance 6 --maxima-downsample 3 --declump-method intensity \
      --merge-line-basin-pixels 16 --merge-line-max-saddle-drop 0 \
      --materialize-objects "target/current-bench/cellprofiler-10/blockflow/run-${n}" \
      >/dev/null || exit
  done'
```

The CellProfiler command was:

```sh
/usr/bin/time -v -o "$bench/cellprofiler-time.txt" docker run --rm \
  -v "$PWD/target/current-bench/cellprofiler-10:/bench" -w /bench \
  cellprofiler/cellprofiler:4.2.8 \
  -c -r -p ExampleHuman.cppipe -i images -o cellprofiler-output
```

## Cellpose CUDA annotation

This comparison used a 2048 x 6144 DAPI subset from the 2079 OME-Zarr slide,
CP-SAM, CUDA, batch size 8, and Cellpose's default thresholds. Blockflow ran the
normal reader, planner, executor, label writer, and linked-table writer. Python
Cellpose 4.1.1 read the same pixels as one TIFF and returned its normal in-memory
outputs. Compilation and fixture conversion were outside the timers.

| Runner | Wall | Ratio | Peak host RSS | Cells |
|---|---:|---:|---:|---:|
| Blockflow, release, masks only, 1 worker | **48.92 s median** | **1.20× faster** | 1.44 GiB | 962 |
| Python Cellpose 4.1.1 | 58.58 s | 1.00× | 2.79 GiB | 1,145 |

The Blockflow median is from 48.92, 48.54, and 52.98 second runs. Python was one
run and spent 47.92 seconds inside `model.eval`. Blockflow evaluates three
overlapping outer blocks, normalizes each block independently, assigns objects
by centroid, and writes the annotation. Python evaluates the strip as one
image, so the cell counts are stable outputs for each runner rather than an
exact semantic-parity assertion.

The masks-only Cellpose API produced byte-identical Blockflow labels and an
identical table compared with the full-output API. Its 48.92 second median was
4.7% faster than the matched full-output median of 51.34 seconds. Exploratory
batch-size 16 and 32 runs took 48.76 and 49.30 seconds, so batch size 8 remains
the default. Two-worker runs took 48.29, 53.57, and 56.39 seconds: their 53.57
second median was 9.5% slower and their median peak host RSS rose to 1.57 GiB.
One worker remains the default because one model and CUDA stream serialize
inference.

Detailed release profiling attributed 40.35 of 50.04 Cellpose seconds to the
network forward pass. Upload and prediction download totaled 0.27 seconds,
while unused flow-color rendering took 3.00 seconds. These measurements support
the masks-only change and do not support adding a pinned-memory copy pipeline or
multiple model instances on this GPU.
