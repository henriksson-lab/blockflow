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

## YOLOv11 CUDA inference

Measured on 2026-09-24 on the Quadro RTX 5000. Sixteen COCO validation images
were losslessly letterboxed to 640 x 640 and repeated twice. The original read
the 32 prepared PNG entries; Blockflow read the same pixels from 32 aligned,
gzip-compressed OME-Zarr blocks. Both used batch size one, FP32, confidence
0.25, IoU 0.45, three untimed warmup forwards, PyTorch/LibTorch 2.9 with CUDA
12.8, and the same converted weights. Blockflow used four workers to overlap
reading and preprocessing with GPU execution.
The timed regions include input decoding, host preprocessing, GPU transfer,
forward inference, and NMS. Blockflow additionally writes fragment rows during
its timed phase. Model loading, warmup, final table assembly, and fixture
preparation are outside both timers.

| Runner | Median | Throughput | Ratio | Detections |
|---|---:|---:|---:|---:|
| Blockflow YOLO, release LibTorch CUDA | 0.758 s | 42.22 images/s | **1.11× faster** | 158 |
| Original `jahongir7174/YOLOv11-pt`, FP32 CUDA | 0.838 s | 38.18 images/s | 1.00× | 158 |

Blockflow runs were 0.767, 0.758, and 0.734 seconds. Original runs were 0.806,
0.838, and 0.839 seconds. All 158 detections matched by class and centre within
0.006 pixels; mean centre distance was 0.0038 pixels and maximum confidence
difference was `4.85e-5`. The original's normal FP16 path took 0.852 seconds in
one run and produced 156 detections, so FP16 did not improve throughput at batch
size one on this GPU and moved two predictions across the confidence threshold.

The initial CubeCL implementation took 2.200 seconds median. Nsight attributed
71.4% of its GPU kernel time to direct convolution. Switching the same Burn
model graph to LibTorch/cuDNN and fusing Conv plus BatchNorm reduced the model
forward from 27-29 ms to about 7.5 ms per image. Four workers were optimal on
this workload; one took 1.359 seconds, two took 0.782 seconds, and eight took
0.986 seconds.

Setting up this comparison also found and fixed an RGB-to-BGR reversal in the
Blockflow adapter. Both runners perform three forwards before their timed
regions. The fixture generator, original runner, agreement checker, release
commands, and scope are documented in
[`examples/yolo-ome-zarr`](examples/yolo-ome-zarr/README.md).

### Zero-shot DAPI baseline

The same release LibTorch path was run over the central 25% by area of the 2079
level-0 DAPI plane: `y=39360..118080`, `x=16512..49536`. DAPI was replicated to
RGB. A 512 pixel core and 64 pixel halo produced 640 pixel model inputs, and
four workers processed 10,010 blocks. The fragment phase took **209.62 s** and
the whole process took **211.90 s**, or 47.75 model inputs/s during execution.
Peak host RSS was 3.73 GiB.

This standard COCO checkpoint is not useful as a zero-shot cell detector. It
returned 2,766 detections while the existing Cellpose table contains 323,028
cells in the same window, a count ratio of **0.86%**. Detection centres landed
inside 2,100 Cellpose masks but touched only 2,026 distinct cells, an upper-bound
recall of **0.63%**. The three most common predicted COCO classes were `clock`
(1,743), `spoon` (714), and `pizza` (236). These numbers establish the
untrained baseline; they do not measure the potential of a YOLO model trained
from the Cellpose annotations.

### Trained DAPI model over the full image

Measured on 2026-09-26 on the same Quadro RTX 5000. The model selected after
80 epochs on Cellpose-derived nucleus boxes was applied to the complete
157,440 x 66,048 level-0 DAPI plane. This is a 10.40-gigapixel image. The
release LibTorch CUDA executable used 512 pixel cores, 64 pixel halo, four
workers, confidence `0.42642644` selected on the validation partition, IoU
`0.65`, and a 1,000-detection per-window limit.

| Scope | Time | Throughput |
|---|---:|---:|
| Planned fragment execution, 39,732 model inputs | **666.03 s** | **59.65 inputs/s** |
| Complete process and native table materialization | **675.32 s** | **15.40 source MP/s** |

The full workflow finished in **11 minutes 15.32 seconds**. Work outside the
fragment executor added 9.29 seconds, including model startup, planning, seam
deduplication, CSV output, and indexed NGFF object-table materialization. Peak
host RSS was 12,793,472 KiB, about 12.2 GiB.

It produced 468,345 detections after merging 999 duplicates at window seams,
or about 694 final detections per second end to end. The existing full-image
Cellpose table contains 490,379 objects, so the YOLO count is **95.5%** of the
Cellpose count. This count ratio is descriptive: it does not show whether the
same nuclei were detected or whether Cellpose errors were corrected. The
frozen spatial test split measured mAP `0.6046`, mAP50 `0.8552`, recall
`0.7755`, and precision `0.8329` against Cellpose-derived boxes.

The native table contains 468,345 verified rows with stable IDs, centroids,
bounding boxes, confidence, class, spatial indexes, and an occupancy pyramid.
newvolim discovered every row through the current CSV compatibility path and
returned 893 objects for a representative 1,000 x 1,000 viewport. The exact
release command and output location are recorded in
[`examples/yolo-ome-zarr`](examples/yolo-ome-zarr/README.md#trained-dapi-model-over-the-full-image).

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

## 3D segmentation CUDA kernels

Measured on 2026-09-29 on the Quadro RTX 5000. These direct comparisons use
the same two 16 x 64 x 64 crops from the PBMC volume, one crowded and one with
an isolated cell. Both implementations used GPU inference, one warmup,
anisotropy 1.98, batch size 8 for Cellpose, and release Rust binaries. Model
loading is excluded.

| Cellpose3D crop | Python 4.1.1 | Rust | Rust/Python time | Objects Python/Rust | Foreground Dice |
|---|---:|---:|---:|---:|---:|
| Isolated | 20.625 s | 21.785 s | **1.06x** | 1 / 1 | 0.9876 |
| Crowded | 20.686 s | 22.152 s | **1.07x** | 8 / 8 | 0.9874 |

All eight crowded-crop objects matched above 0.5 IoU; their mean matched IoU
was 0.9238. The small numerical differences are consistent with converted
weights and different GPU backends. The 6–7% timing difference does not justify
Cellpose3D optimization before larger quality evaluation.

| StarDist3D crop | Python total | Rust total | Rust/Python time | Rust raw-inference speed | Labels |
|---|---:|---:|---:|---:|---|
| Isolated | 0.160 s | 0.172 s | **1.08x** | **1.45x faster** | exact |
| Crowded | 0.237 s | 0.191 s | **0.81x** | **1.37x faster** | exact |

StarDist used the bundled `3D_demo` model. Total time includes sparse prediction
and polyhedron instance construction. The native labels matched Python exactly,
so no StarDist3D speed work is indicated by these crops. The model found only
two objects where Cellpose found eight in the crowded crop; this particular
model is therefore a format and performance reference rather than the selected
teacher.

The normal Blockflow crop workflows, including OME-Zarr reading, planning,
label writing, pyramid construction, and indexed table finalization, took
21.430 s for Cellpose (8 objects) and 0.567 s for StarDist (2 objects). The
fixed crop coordinates and reproduction commands are documented in the two 3D
example directories.

### Full PBMC volume

The release CUDA Cellpose3D example processed the complete
60 x 1,592 x 3,333 level-0 volume as one outer Blockflow block. CP-SAM used
anisotropy 1.98, batch size 8, and one worker. The complete workflow took
**9,738.827 s (2 h 42 min 18.8 s)** and published 115 instances. Peak host RSS
was 61,089,056 KiB (58.3 GiB). GPU inference stayed near full utilization; the
whole-volume mode avoids repeating Cellpose's three sets of internally tiled
orthogonal planes across overlapping outer blocks.

The result is a five-level 34 MiB label pyramid at
`labels/cellpose3d-cpsam` plus a 392 KiB indexed object table at
`tables/cellpose3d-cpsam`. The label source correctly resolves the Bio-Formats
series as `../../0`, and the temporary work tree was removed after publication.
This timing is accepted for the current project; Cellpose3D and StarDist3D
inference optimization is closed unless a future matched benchmark shows a
material regression.

### YOLO3D distilled center detector

The separate native 3D detector was trained from the Cellpose3D labels with
24 x 192 x 192 input patches, 6 x 24 x 24 ownership halos, CUDA, and release
builds. Its leakage-safe split grouped overlapping positive windows and added
two empty background windows per positive window. The final split contained
36 training, 57 validation, and 66 frozen test windows.

The AP-selected checkpoint came from epoch 2 of an eight-epoch run. The full
run took 7m29.69s including validation after every epoch. Dataset-derived size
priors were essential: without them, matched predicted extents oscillated from
less than half to more than 18 times teacher size under momentum.

| Partition | Complete teacher objects | Center AP | Precision | Recall | Box AP50 |
|---|---:|---:|---:|---:|---:|
| Validation, threshold 0.01961601 | 18 | **0.7298** | 0.7647 | 0.7222 | 0.0556 |
| Frozen test, ranked curve | 15 | **0.6233** | 0.6087 | 0.9333 | 0.0000 |

The test precision and recall are reported at the test curve's diagnostic best
point; the deployment threshold remained fixed from validation. These are
agreement metrics against the Cellpose teacher, not manually reviewed
biological accuracy. The low box AP means the current model is a center
detector with approximate extents.

Full-volume Blockflow inference at the validation-selected threshold processed
the 60 x 1,592 x 3,333 volume in **283.572s application time** and **286.83s
wall time** with 761,708 KiB peak host RSS. It published 74 detections in a
432 KiB spatially indexed object table. Relative to the 2h42m18.8s Cellpose3D
teacher run, end-to-end YOLO3D inference was **34.4x faster**. The count is
64.3% of the 115-object teacher count, consistent with held-out recall and not
a claim of label parity. The compatibility CSV was discovered by release
newvolim alongside the Cellpose label, and a full-volume viewport query returned
all 74 YOLO rows.

## StarDist CUDA annotation

This comparison, measured on 2026-09-22, used the same 2048 x 6144 DAPI subset
as the Cellpose comparison, the official `2D_versatile_fluo` model, CUDA, fixed
normalization from 1 to 70, and the model's default probability and NMS
thresholds. Blockflow used three 2048 pixel blocks with a 64 pixel halo and ran
the normal reader, planner, executor, label writer, and linked-table writer.
Python StarDist 0.9.2 read the same Zarr pixels and returned its normal in-memory
label and polygon outputs for one image.

| Runner | Wall | Ratio | Peak host RSS | Cells |
|---|---:|---:|---:|---:|
| Blockflow, release, 1 worker | **6.57 s median** | **5.30× faster** | 746 MiB | 15 |
| Python StarDist 0.9.2 | 34.80 s median | 1.00× | 2.93 GiB | 15 |

Blockflow runs took 6.66, 6.57, and 6.52 seconds. Python runs took 54.61,
34.71, and 34.80 seconds; its median `predict_instances` time was 28.05
seconds. Giving Python a comparable `n_tiles=(1, 3)` took 40.80 seconds and
2.64 GiB in one probe, so it did not narrow the gap. Both runners reported 15
objects. Matching counts are a useful sanity check, not a label-equivalence
claim, because the two runners use different outer tiling and ownership rules.

The result does not justify StarDist speed optimization before a whole-slide
run. Segmentation quality on representative tissue should be reviewed first,
because performance headroom does not establish that the model and thresholds
are biologically suitable for this slide.

An Nsight Systems follow-up on one release run measured 1.56 seconds of GPU
kernels and 0.323 seconds of GPU copies. Device-to-host copies moved 432.5 MB
in 0.315 seconds; host-to-device copies moved 58.1 MB in 0.008 seconds. All
kernels and copies used one CUDA stream. Pure copy overlap can therefore save
at most about 5% of the 6.57 second median. The trace also showed 1.34 second
host gaps between one block's dense distance download and the next block's
upload. If StarDist is optimized later, the first candidate should threshold
probabilities and gather only selected ray distances on the GPU, then pipeline
CPU instance construction with the next inference. Pinning or asynchronously
copying the current dense buffers alone has a much smaller ceiling.

The Python timing entry point is
`examples/stardist-ome-zarr/scripts/benchmark_python.py`. For example:

```sh
/usr/bin/time -v .tmp/stardist-python/bin/python \
  examples/stardist-ome-zarr/scripts/benchmark_python.py \
  --zarr .tmp/cellpose-bench-2048x6144.zarr \
  --model .tmp/stardist-models --low 1 --high 70 \
  --output .tmp/stardist-python.json
```

### Full 2079 slide completion

The StarDist example was also run over the complete 157,440 x 66,048 DAPI
plane. It published 404,695 cells as a ten-level, 1.2 GiB label pyramid at
`labels/stardist-dapi` and a 35 MiB linked table at
`tables/stardist-dapi/table.csv`. The existing Cellpose label and table remain
beside it, so newvolim can display either annotation on the same image.

The elapsed time from starting segmentation to publishing the repaired output
was **1:28:06**. The two process segments totaled **1:24:04**. This is an
end-to-end completion record, not a benchmark of the current implementation:
the run exposed an old StarDist-only pyramid builder that repeatedly read level
0. It was stopped after staging levels 0 through 7 at 157.4 GiB peak host RSS,
then resumed after both annotation examples were moved to the shared
planner-driven label-pyramid builder. The release resume built levels 8 and 9,
collected the table, and published the result in 12.95 seconds at 412 MiB peak
host RSS.

A future small-data experiment should compare one and two StarDist workers.
The current CUDA path uses one model and stream, so this must be measured rather
than assumed to improve GPU occupancy.
