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

The best validation checkpoint is evaluated exactly once against the test
partition only when `--finalize` is passed. Test metrics are stored separately
in `training-run.json`.

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

This establishes a working end-to-end training and held-out evaluation path and
produces loadable Burn checkpoints. The result remains well below Cellpose as a
cell detector. More spatially balanced training data and a longer stable
schedule are needed before comparing runtime at a fixed quality target.

The reference learning rate range (0.0001 to 0.01) diverged on these dense
pseudo-labels without clipping. A 0.001 peak stayed finite with clipping but
damaged the model after warmup: a 256-tile run selected epoch 3 at validation
mAP 0.009 and scored zero on the held-out test partition. The 0.0002 cap is
therefore the current DAPI example default.
