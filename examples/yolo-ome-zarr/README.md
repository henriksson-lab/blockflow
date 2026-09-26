# YOLOv11 over OME-Zarr

The proposed training architecture, including Cellpose-derived targets and
reuse of Blockflow's cache and prefetcher, is described in
[`YOLO_TRAINING.md`](../../YOLO_TRAINING.md).

The implemented Cellpose-to-YOLO transfer workflow, release command, and first
DAPI experiment are documented in [`DAPI_TRAINING.md`](DAPI_TRAINING.md).

This is the normal Blockflow entry point for the YOLO fragment operation. It
reads an OME-Zarr level through `ZarrEnvironment`, assembles the fragment phase
with `PlanBuilder`, runs one model invocation per configured image block, owns
detections by their centre, merges fragment rows, and writes an indexed NGFF
object table. A CSV can also be written for tools that have not yet adopted the
native table reader.

The fastest GPU path uses Burn's LibTorch backend. It requires PyTorch 2.9 with
CUDA and uses four workers to overlap Zarr reading and preparation with model
inference. Build and run it in release mode:

```bash
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cargo run --release -p blockflow-yolo-ome-zarr \
  --no-default-features --features libtorch -- \
  --zarr image.zarr \
  --weights model.safetensors \
  --config default_args.yaml \
  --block 640 --halo 0 --workers 4 \
  --table image.zarr/tables/yolo-cells \
  --out detections.csv --summary summary.csv --work work
```

Use `--region y,x,height,width` to process a window without copying it into a
separate Zarr. Output coordinates remain in the coordinate system of the whole
selected level.

## Zero-shot DAPI baseline

The unmodified COCO checkpoint is not a useful cell detector. A release CUDA
run over the central 25% by area of the 2079 DAPI plane used one input channel,
replicated to RGB, and the normal windowed reader, planner, four-worker
executor, and fragment table path:

```bash
target/release/yolo-ome-zarr \
  --zarr /husky/otherdataset/teresa/2079_merged_registered.zarr \
  --weights /path/to/YOLOv11-rs/weights/model.safetensors \
  --config /path/to/YOLOv11-rs/default_args.yaml \
  --channels 0 --region 39360,16512,78720,33024 \
  --block 512 --halo 64 --workers 4 --conf-threshold 0.25 \
  --out detections.csv --summary summary.csv --work work
```

The 10,010 tile fragment phase took 209.62 seconds and the whole process took
211.90 seconds. YOLO returned 2,766 COCO-class detections where Cellpose found
323,028 cells. The YOLO centres touched 2,026 distinct Cellpose masks, giving an
upper-bound recall of 0.63%. A cell-specific model is required before YOLO can
be compared with Cellpose for segmentation quality.

The pure CubeCL CUDA backend remains available as `--features cuda`.

## Trained DAPI model over the full image

The Cellpose-box transfer model was run over level 0 of the complete
`2079_merged_registered` image with the confidence threshold selected on the
validation partition:

```bash
target/release/yolo-ome-zarr \
  --zarr /husky/otherdataset/teresa/2079_merged_registered.zarr \
  --weights .tmp/yolo-dapi-training/full-all-e80-lr5e-4/best.bpk \
  --config examples/yolo-ome-zarr/dapi_corrected.yaml \
  --channels 0 --normalize-low 1 --normalize-high 70 \
  --block 512 --halo 64 --workers 4 \
  --conf-threshold 0.42642644 --nms-iou 0.65 \
  --max-detections 1000 --min-separation 4 \
  --table /husky/otherdataset/teresa/2079_merged_registered.zarr/tables/yolo-dapi \
  --out .tmp/yolo-dapi-full/detections.csv \
  --summary .tmp/yolo-dapi-full/summary.csv \
  --work .tmp/yolo-dapi-full/work
```

The run processed 39,732 blocks and produced 468,345 detections after merging
999 duplicates at window seams. CUDA fragment execution took 666.03 seconds;
the complete process took 675.32 seconds (11 minutes 15.32 seconds) and reached
about 12.2 GiB resident memory. The native table stores stable IDs, centroids,
bounding boxes, confidence, class, spatial indexes, and an occupancy pyramid.
Its `table.csv` compatibility copy lets the current newvolim reader display the
same detections. A 1000 by 1000 pixel viewport query returned 893 objects,
confirming that the layer can be loaded selectively by the viewer.

This model predicts boxes learned from Cellpose instance masks. It supports
visual comparison and counting, but it does not reproduce Cellpose masks and
is not yet suitable for per-cell intensity measurements in crowded regions.

## Matched GPU benchmark

The translated model comes from `jahongir7174/YOLOv11-pt`, so that repository
is the reference implementation. The benchmark prepares the same lossless,
letterboxed 640 px images as individual PNG files for the original and as
chunk-aligned RGB OME-Zarr blocks for Blockflow:

```bash
examples/yolo-ome-zarr/benchmark/run_benchmark.sh \
  /path/to/COCO/images/val2017 /path/to/YOLOv11-rs .tmp/yolo-benchmark
```

The script performs three release CUDA runs per implementation, verifies the
detections, and writes `summary.json`. Its individual steps are:

```bash
python examples/yolo-ome-zarr/benchmark/prepare.py \
  --source /path/to/COCO/images/val2017 \
  --output .tmp/yolo-benchmark --images 16 --repeats 2

export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cargo run --release -p blockflow-yolo-ome-zarr \
  --no-default-features --features libtorch -- \
  --zarr .tmp/yolo-benchmark/input.zarr \
  --weights /path/to/model.safetensors \
  --config /path/to/default_args.yaml \
  --block 640 --halo 0 --workers 4 \
  --out .tmp/yolo-benchmark/blockflow.csv \
  --summary .tmp/yolo-benchmark/blockflow-summary.csv \
  --work .tmp/yolo-benchmark/work

python examples/yolo-ome-zarr/benchmark/original.py \
  --original /path/to/YOLOv11-pt \
  --weights /path/to/YOLOv11-pt/weights/best.pt \
  --images .tmp/yolo-benchmark/images.txt \
  --precision fp32 --batch-size 1 \
  --output .tmp/yolo-benchmark/original.json
```

Model construction and three warmup forwards are outside both timed regions.
Blockflow reports fragment execution separately from model loading and final CSV
materialisation. Compare that with `seconds` from the original runner.
Whole-process timings answer a different question and should be recorded
separately.

On the Quadro RTX 5000, 32 matched FP32 invocations took 0.758 seconds median
through Blockflow and 0.838 seconds through the original: Blockflow was **1.11×
faster**. All 158 detections matched by class and within 0.006 pixels. See
[`BENCHMARKS.md`](../../BENCHMARKS.md#yolov11-cuda-inference) for the individual
runs, measured scope, and FP16 probe.
