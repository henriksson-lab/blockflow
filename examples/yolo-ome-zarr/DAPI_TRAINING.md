# DAPI transfer-training experiment

This example trains YOLO directly from an OME-Zarr DAPI plane and a Cellpose
label image. It reads both arrays through one `ZarrEnvironment`, derives boxes
from the label pixels in each haloed crop, and uses the native object table's
centroids to assign every object to one 512 x 512 ownership tile. No PNG or
YOLO-text dataset is materialized.

The split reserves the central quarter of the image as a frozen test region.
Training and validation inputs cannot overlap that region. The bounded pilot
selects tiles evenly across the remaining spatial partitions.

## Inspect the generated samples

Always build and run this example in release mode:

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --release \
  -p blockflow-yolo-ome-zarr --features libtorch --bin yolo-dapi-train -- \
  --zarr /path/to/image.zarr \
  --train-tiles 64 --validation-tiles 16 \
  --inspect-only
```

This writes `inspection.json`, `train-sample.png`, and
`validation-sample.png` under `.tmp/yolo-dapi-training/output` by default.

## Run the transfer pilot

The supplied configuration uses a lower learning rate for the dense DAPI
pseudo-labels. Gradient norm clipping is enabled by this example and can be
disabled with `--gradient-clip 0` for controlled comparisons.

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --release \
  -p blockflow-yolo-ome-zarr --features libtorch --bin yolo-dapi-train -- \
  --zarr /path/to/image.zarr \
  --weights /path/to/yolov11n.safetensors \
  --reset-class-head \
  --train-tiles 256 --validation-tiles 64 --test-tiles 64 \
  --epochs 40 --schedule-epochs 40 --finalize \
  --batch-size 16 --workers 2 --queue-batches 2 \
  --cache-mib 8192 \
  --geometry d4
```

`--reset-class-head` loads compatible pretrained feature and box tensors while
leaving the one-class output convolutions newly initialized. D4 is suitable for
this fluorescence dataset. Geometry remains an explicit dataset policy because
instrument-specific DIC and phase-contrast data may not support rotations or
reflections.

The output directory contains `last.bpk`, `best.bpk`, `step.csv`,
`training-progress.svg`, `split.json`, `resume.json`, per-stage checkpoints,
and JSON run reports. The SVG is regenerated after every completed stage and
contains separate panels for training losses and validation metrics. The latest
JSON report records training time, end-to-end time, inputs, selected policies,
and cache/prefetch counters.

An existing history can be rendered without running training:

```bash
cargo run --release --manifest-path /path/to/YOLOv11-rs/Cargo.toml \
  --bin yolov11 -- plot \
  --history /path/to/run/step.csv \
  --output /path/to/run/training-progress.svg
```

Reload and score a checkpoint on the recorded validation split without touching
the frozen test partition:

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --release \
  -p blockflow-yolo-ome-zarr --features libtorch --bin yolo-dapi-train -- \
  --zarr /path/to/image.zarr \
  --weights /path/to/run/best.bpk --output /path/to/run \
  --train-tiles 256 --validation-tiles 64 --test-tiles 64 \
  --batch-size 16 --workers 2 --queue-batches 2 \
  --cache-mib 8192 --geometry d4 --evaluate-only
```

This writes `validation-evaluation.json`. A completed training stage also
reloads `best.bpk` internally and requires its recall, precision, mAP@50, and
mAP to match the in-memory best model before committing `resume.json`.

The best validation checkpoint is evaluated exactly once against the test
partition only when `--finalize` is passed. Test metrics are stored separately
in `training-run.json`.

## Hyperparameter sweeps

`--sweep` runs trials sequentially on one GPU. Every trial uses the same
in-memory stores, decoded Zarr cache, spatial split, and seed. A sweep never
evaluates the test partition. Its YAML defines Cartesian axes and
successive-halving rounds; omitted axes retain the ordinary training setting.

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --release \
  -p blockflow-yolo-ome-zarr --features libtorch --bin yolo-dapi-train -- \
  --zarr /path/to/image.zarr \
  --weights /path/to/dapi-best.bpk \
  --output /path/to/sweep \
  --train-tiles 512 --validation-tiles 128 --test-tiles 128 \
  --batch-size 16 --workers 2 --queue-batches 2 --cache-mib 16384 \
  --sweep examples/yolo-ome-zarr/dapi_sweep.yaml
```

The supplied sweep varies peak learning rate, trainable layers, and BatchNorm
policy. `head-and-neck` freezes DarkNet and trains the feature pyramid plus
detection head. `head-only` also freezes the feature pyramid. Each trial has a
resumable directory under `trials/`. `sweep.csv` and `sweep.json` are rewritten
after every round and rank trials by validation mAP. Re-running the same command
resumes trials at completed round boundaries. `dapi_head_sweep.yaml` is the
narrow follow-up that compares those two freezing policies and weight decay.

Only D4 or no geometry is active in the source-neutral OME-Zarr loader. The
HSV, translation, scale, mosaic, and probabilistic flip fields retained in the
model YAML belong to the legacy file loader and are not sweep axes here.

## Staged training and exact resume

Loading `best.bpk` through `--weights` is a warm start, not an exact resume.
It restores model weights but resets SGD momentum, EMA state, the learning-rate
schedule, and epoch numbering. This is useful for an intentional fine-tuning
fork, but it is a new run.

Choose the full schedule horizon in the first stage:

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --release \
  -p blockflow-yolo-ome-zarr --features libtorch --bin yolo-dapi-train -- \
  --zarr /path/to/image.zarr --output /path/to/run \
  --weights /path/to/yolov11n.safetensors --reset-class-head \
  --train-tiles 256 --validation-tiles 64 --test-tiles 64 \
  --epochs 20 --schedule-epochs 160 --batch-size 16 --cache-mib 8192
```

Continue it in 20-epoch stages. The original schedule horizon is read from the
checkpoint, so it does not need to be repeated:

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --release \
  -p blockflow-yolo-ome-zarr --features libtorch --bin yolo-dapi-train -- \
  --zarr /path/to/image.zarr --output /path/to/run \
  --resume /path/to/run \
  --train-tiles 256 --validation-tiles 64 --test-tiles 64 \
  --epochs 20 --batch-size 16 --cache-mib 8192
```

Pass `--finalize` on the last stage to evaluate the held-out test partition.
Each completed stage stores the raw model, EMA model and update count, SGD
momentum, optimizer position, global epoch, fixed schedule horizon, and best
validation result. `step.csv` is appended with stage and global epoch columns.
If a process stops during a stage, resume trims uncommitted history rows back
to the last completed stage.

`--resume RUN` verifies the exact tile coordinates, source settings, batch and
input sizes, seed, geometry, validation interval, optimizer, loss, learning
rate, clipping, and class configuration. `--weights CHECKPOINT` starts a fresh
warm-start run.

If a later stage increases `--train-tiles`, it is a child run because the
training distribution changed. The current even-rank sampler does not guarantee
that the 256 selected tiles are a subset of the 1,024 selected tiles.

## Current spatial split

The split operates on 512 by 512 ownership tiles containing at least one
Cellpose centroid:

1. The central half of each image axis, which is the central quarter by area,
   is reserved for test data. A test ownership core must fit wholly inside it.
2. Any other 640 by 640 model input whose 64-pixel halo intersects that central
   rectangle is discarded as a guard band.
3. Among the remaining outer tiles, the highest 20% of occupied `y` rows form
   one contiguous validation band. With image coordinates increasing downward,
   this is the lower outer band. The other 80% form training candidates.
4. Requested bounded samples are selected deterministically at evenly spaced
   ranks from each `(y, x)`-sorted candidate list.

This gives spatial separation and reproducibility, but it is not an IID random
split. It deliberately measures transfer to other parts of the slide. Only
occupied tiles are currently sampled, so there are no pure-negative training
crops. Future multi-slide training should split by whole slide or acquisition
group and add a recorded fraction of negative tiles. The exact coordinate lists
are written to the run manifest so changes to the resulting split cannot
silently change a resumed run.

For a full spatial training pass, `--train-tiles all` selects every occupied
ownership tile in the eligible training region. `--negative-tile-fraction 0.1`
adds deterministic background tiles whose 3 by 3 ownership-tile neighborhood
contains no labeled centroid. The same fraction is applied to bounded
validation and test selections. Samples stream from OME-Zarr through the
bounded cache; they are grouped into small spatial blocks and the blocks and
samples are shuffled deterministically each epoch. This retains I/O locality
without keeping a smaller resident training subset.

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --release \
  -p blockflow-yolo-ome-zarr --features libtorch --bin yolo-dapi-train -- \
  --zarr /path/to/image.zarr --output /path/to/run \
  --weights /path/to/best.bpk \
  --train-tiles all --validation-tiles 256 --test-tiles 256 \
  --negative-tile-fraction 0.1 \
  --epochs 20 --schedule-epochs 80 --batch-size 16 --cache-mib 8192
```

## Recorded experiments

Dataset: `/husky/otherdataset/teresa/2079_merged_registered.zarr`, DAPI channel
0, with `labels/cellpose-dapi` and `tables/cellpose-dapi`.

- 490,366 pseudo-labeled nuclei were available.
- Inspection selected 256 training, 64 validation, and 64 test crops. They
  contained 19,895, 3,951, and 4,373 boxes respectively.
- The retained run used batch 16, two loader workers, D4, gradient norm limit
  10, and learning rates 0.00001 to 0.0002 for 40 epochs.
- Training plus the single final test evaluation took 322.99 seconds; process
  wall time was 324.77 seconds.
- The validation-selected checkpoint reached mAP 0.0589 and mAP@50 0.1576.
- On the untouched central test partition it reached mAP 0.1018, mAP@50
  0.2355, recall 0.3844, and precision 0.3150.
- The 8 GiB decoded cache recorded 189,001 hits, 4,266 misses, and no
  evictions. It read each selected source chunk once and retained the 7.41 GB
  working set across epochs.
- Batch 16 used about 8.3 GB of GPU memory. Active samples reached 65-92% SM
  utilization; batch 8 typically reached 47-75%.

A follow-up warm-start stage used the retained checkpoint with 512 training,
128 validation, and 128 frozen test tiles, batch 16, two loader workers, D4,
and a 0.00001 to 0.00005 learning-rate range. It ran 20 epochs of a planned
120-epoch schedule in 372.92 seconds end to end. Validation mAP was 0.05649 at
epoch 1 and fell to 0.029 by epoch 20; the unchanged warm-start checkpoint
scored 0.05829 on the same 128 validation tiles. The reloaded epoch-1 best
checkpoint reproduced mAP 0.05649 and mAP@50 0.15101 exactly. The frozen test
partition was not evaluated. Do not continue this schedule unchanged.

The first successive-halving sweep screened 12 configurations on the same
512/128 split. It varied peak learning rates `0.000005`, `0.00001`, and
`0.00002`, DarkNet freezing, and BatchNorm policy. All six configurations that
updated BatchNorm ranked below all six that kept its population statistics
fixed. The selected trial froze DarkNet and BatchNorm and used peak LR
`0.000005`; its epoch-2 mAP was 0.05843 versus 0.05829 for the unchanged input
checkpoint, a 0.25% relative difference. Its later epochs declined to mAP
0.053 by epoch 10. Trial training consumed 1,694.37 seconds in total. This is
not evidence of a useful quality improvement, and the test partition remained
untouched.

The narrower eight-trial follow-up compared neck-and-head against head-only
training, weight decay `0` against `0.0005`, and peak LR `0.000005` against
`0.00001`, always with frozen BatchNorm. It completed in 600.56 seconds wall
time. Neck-and-head at peak LR `0.000005` won again; zero decay and `0.0005`
differed by only 0.00000002 mAP, while head-only trailed by 0.000054. The
selected mAP remained 0.05843 at epoch 2 and declined afterward. These results
do not justify another sweep over freezing policy or weight decay.

This establishes a working end-to-end training and held-out evaluation path and
produces loadable Burn checkpoints. The result remains well below Cellpose as a
cell detector. More spatially balanced training data and a longer stable
schedule are needed before comparing runtime at a fixed quality target.

The earlier learning-rate experiments and weak overfit results were produced
with a reversed SiLU backward call in the local Burn libtorch backend. Those
runs remain useful as records of the data and runtime path, but they do not
constrain the corrected optimizer settings. After fixing SiLU backward, the
same 16 dense tiles improved from best mAP 0.119 to 0.473 in 100 updates using
the conservative `5e-6` to `5e-5` schedule. Learning-rate selection must be
repeated before choosing a full microscopy training schedule.

The corrected follow-up used 256 training, 64 validation, and 64 frozen test
tiles, batch 16, two loader workers, frozen BatchNorm, D4 geometry, and an
80-epoch schedule with learning rates from `0.0005` down to `0.00005`. It ran
as four exactly resumed 20-epoch stages. Best validation mAP was `0.4419` at
epoch 72, with mAP50 `0.7164`, recall `0.6534`, and precision `0.7553`. The
single final evaluation of the held-out test region reached mAP `0.5457`,
mAP50 `0.8114`, recall `0.7376`, and precision `0.8094`. The successful stages
took about 837 seconds of process wall time in total. The last epochs had
plateaued, so this pilot should not be extended past its fixed horizon.

The subsequent all-tiles inspection selected 2,004 training tiles, including
200 conservative background tiles, plus 256 validation and 256 frozen test
tiles with 26 backgrounds in each. The training split contains 209,318 visible
targets. Its median, 90th percentile, and maximum target counts per tile are
46, 300, and 540. The 1,000-detection evaluation limit retains every oracle
prediction. A stride-8 assignment check found 6,585 targets, 3.1%, with no
candidate location; retain this result for the later P2 comparison.

The first 20 epochs of the all-tiles run took 1,444.8 seconds. Epoch 20 was
best, with validation mAP `0.4804`, mAP50 `0.7671`, recall `0.6797`, and
precision `0.7737`. The bounded decoded cache ended at about 8.6 GB while
streaming and evicting chunks across the full spatial dataset.

All four stages completed their 80-epoch horizon in about 5,014 seconds of
process wall time. Epoch 80 was best on validation: mAP `0.5295`, mAP50
`0.7973`, recall `0.7037`, and precision `0.8004`. The single final evaluation
of the frozen 256-tile test split reached mAP `0.6046`, mAP50 `0.8552`, recall
`0.7755`, and precision `0.8329`. The run artifacts are in
`.tmp/yolo-dapi-training/full-all-e80-lr5e-4`.

This detector is trained against boxes derived from Cellpose instance masks.
The reported metrics quantify Cellpose-box agreement rather than independent
biological correctness. Before production use, review a small stratified set
for missed nuclei, false objects, merges, and splits. Box detections support
localization and counting; quantitative per-cell color measurements require
mask prediction or a local segmentation refinement around each detection.

## Full-image inference

The validation split selected confidence threshold `0.42642644` by maximum F1.
Using that threshold, the validation-selected epoch-80 checkpoint was applied
to all of level 0 with 512 pixel blocks, 64 pixel halo, and four CUDA workers.
The 39,732 fragment tasks took 666.03 seconds. End-to-end runtime, including
model loading, planning, duplicate merging, CSV output, and native table
materialization, was 675.32 seconds. Peak resident memory was about 12.2 GiB.

The completed output contains 468,345 detections after 999 seam duplicates were
merged. It is stored at
`/husky/otherdataset/teresa/2079_merged_registered.zarr/tables/yolo-dapi` as an
indexed NGFF object table. A `table.csv` compatibility file in the same folder
makes it visible to the current newvolim object reader. Viewer discovery found
all 468,345 rows, and a 1000 by 1000 pixel viewport returned 893 rows without
triggering its density guard.
