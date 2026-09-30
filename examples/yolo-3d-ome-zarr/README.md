# YOLO3D distillation over OME-Zarr

This is a separate 3D detector derived from the 2D microscopy YOLO workflow.
It leaves the proven 2D model unchanged. The reusable `yolo3d` crate provides:

- an anisotropy-aware Conv3D backbone with P2, P3, and P4 center heads;
- center heatmap, sub-voxel offset, physical size, and quality predictions;
- confidence-weighted targets from connected 3D teacher labels;
- dataset-derived physical size priors for stable box-head initialization;
- 3D IoU and class-aware NMS without a small whole-volume detection cap;
- resumable training histories, validation AP, threshold calibration, and an
  SVG progress plot.

The Blockflow binaries demand-load OME-Zarr patches through the decoded chunk
cache. Training shuffles groups of eight nearby patches, then reads within each
group in spatial order. This keeps reuse without holding a capped resident
training subset.

## Train from Cellpose3D labels

Always use a release build. CUDA is the intended training path:

```sh
cargo run --release -p blockflow-yolo-3d-ome-zarr --features cuda \
  --bin yolo-3d-train -- \
  --zarr /path/to/image.zarr \
  --teacher cellpose3d-dapi \
  --channel 0 --time 0 \
  --patch-z 24 --patch 192 \
  --ownership-halo-z 6 --ownership-halo 24 \
  --split /path/to/split.json \
  --epochs 20 --device cuda \
  --output .tmp/yolo3d-training/run-1
```

The run writes `model-config.json`, `optimizer-config.json`, `split.json`,
`history.json`, `training-progress.svg`, `last.bpk`, the AP-selected `best.bpk`,
and validation reports. `best.bpk` is selected by center AP, with validation
loss used only to break ties.
Continue training without discarding history:

```sh
cargo run --release -p blockflow-yolo-3d-ome-zarr --features cuda \
  --bin yolo-3d-train -- \
  --zarr /path/to/image.zarr \
  --teacher cellpose3d-dapi \
  --resume .tmp/yolo3d-training/run-1 \
  --epochs 20 --device cuda \
  --output .tmp/yolo3d-training/run-2
```

Resume is refused if the model, split, target geometry, or augmentation policy
changed. Resume restores model weights and concatenates history; it currently
starts fresh optimizer momentum. The default `fluorescence` policy uses XY right-angle rotations and
reflections, Z reflection, intensity scaling, and gamma. Use
`--augmentation transmitted` for DIC or phase contrast acquired with a fixed
lens configuration; it keeps orientation fixed. Dataset-specific choices are
stored in `split.json`.

The one-volume fallback split uses large contiguous Y bands and a full patch
guard between training, validation, and frozen test regions. With multiple
specimens, use whole specimens as the split unit before relying on the fallback.
The frozen test coordinates are recorded but training does not inspect them.

For a dataset with a special crowded region, edit a generated `split.json` and
pass it back with `--split /path/to/split.json`. This permits deliberate crowded
train, validation, and test subregions plus randomized spatial groups elsewhere.
Every listed start must belong to the generated tile grid. The trainer rejects
duplicates and windows whose input extents overlap across partitions, preventing
the same object or image context from leaking through overlapping patches.

For the clustered PBMC volume, `scripts/prepare_component_split.py` assigns
each teacher object to exactly one ownership core, joins positive tiles whose
input windows overlap, and keeps each resulting component in one partition.
The largest component contains the crowded center and is kept intact in
training. Independent exterior components are shuffled and balanced between
validation and frozen test. This is preferable to trying to cut the roughly
250 x 301 pixel central cluster into multiple 256 pixel windows. The generator
also adds two truly empty input windows per positive ownership tile by default.
These background windows are assigned without any cross-partition input
overlap. Omitting them produces severe whole-volume false positives.

Training targets include every complete object visible in a patch, including
objects centered in its halo. Ownership cores are used only to deduplicate
metrics and inference output. Treating visible halo objects as background gives
the model contradictory supervision.

The trainer estimates median physical object extents per feature scale from the
training labels. It stores the priors in `model-config.json` and initializes the
size heads at those values. The detector uses CenterNet focal normalization,
smooth L1 log-size regression, Nesterov momentum, and gradient clipping.

Evaluate an exact checkpoint without changing it:

```sh
cargo run --release -p blockflow-yolo-3d-ome-zarr --features cuda \
  --bin yolo-3d-train -- \
  --zarr /path/to/image.zarr --teacher cellpose3d-dapi \
  --patch-z 24 --patch 192 \
  --ownership-halo-z 6 --ownership-halo 24 \
  --split /path/to/split.json --device cuda \
  --evaluate-checkpoint /path/to/best.bpk \
  --output .tmp/yolo3d-evaluation
```

The report includes center AP, box AP50, the maximum-F1 validation threshold,
fixed threshold operating points, and size/IoU diagnostics. Select the
deployment threshold on validation data before inspecting the frozen test set.

## Full-volume inference

```sh
cargo run --release -p blockflow-yolo-3d-ome-zarr --features cuda \
  --bin yolo-3d-ome-zarr -- \
  --zarr /path/to/image.zarr \
  --checkpoint .tmp/yolo3d-training/run-1/best.bpk \
  --model-config .tmp/yolo3d-training/run-1/model-config.json \
  --channel 0 --time 0 --device cuda \
  --block-z 12 --block 144 --halo-z 6 --halo 24 \
  --threshold 0.01961601 \
  --layer yolo3d-dapi
```

Inference uses the Blockflow planner, halo reads, cache, centroid ownership,
and a spatially indexed NGFF object table. Rows contain `z,y,x`, confidence,
class, and the six 3D box bounds, which newvolim can load partially by spatial
tile. Boxes are suitable for navigation and counting. Quantitative per-cell
channel measurements still require an instance mask or a later ROI refinement
head.

## Clustered PBMC validation

The release CUDA workflow was validated on
`clustered-pbmcs.ome.zarr` (`60 x 1,592 x 3,333`, spacing
`0.5 x 0.253335 x 0.251327` micrometers) using the 115-object Cellpose3D CP-SAM
annotation as teacher. The leakage-safe split contains 36 training windows,
57 validation windows, and 66 frozen test windows; respectively 24, 38, and 44
are empty background windows.

The selected checkpoint was epoch 2 of an eight-epoch run. On 18 complete
validation objects it reached center AP 0.7298, precision 0.7647, recall 0.7222,
and mean matched-box IoU 0.3705 at threshold 0.01961601. Frozen test center AP
was 0.6233; 14 of 15 complete test objects were recoverable at the test curve's
best point. Box AP50 remained low (0.0556 validation and 0 test), so use this
model as a center detector with approximate boxes.

Full-volume inference used the validation-selected threshold and finished in
283.572 application seconds (286.83 seconds wall) with 744 MiB peak host RSS.
It published 74 rows at `tables/yolo3d-cell-centers-v1`. The spatially indexed
NGFF object table is 432 KiB, contains all 12 center/confidence/box columns, and
left no temporary work tree. The example also emits `table.csv` for the current
newvolim compatibility path. A release newvolim server discovered the Cellpose
label layer and the 74-row YOLO object layer, then returned all 74 objects from
a full-volume viewport query. The lower count than the 115-object teacher is
consistent with held-out recall. A threshold of 0.01 did not improve validation
recall, while 0.005 increased recall to 0.8333 at only 0.2586 precision.

The detailed rationale and literature review are in
[`YOLO3D_RESEARCH.md`](../../YOLO3D_RESEARCH.md).
