# YOLO-SliceFuse3D inference optimization plan

## Objective

Reduce end-to-end CUDA inference time for YOLO-SliceFuse3D while preserving the
quality of the current best model. Test every proposed optimization separately,
then combine the changes that pass the quality gates.

The final implementation must continue to use the normal OME-Zarr reader,
Blockflow planner and executor, and `ngff-object-table` publication path. All
builds and measurements use `--release`.

## Implemented result — 2026-10-01

The selected FP32 path uses two normal Blockflow plans over one attached
OME-Zarr source. The first reads each detector-aligned XY window over the whole
60-plane Z axis and runs the frozen 2D encoder once. It retains the P3/P4/P5
features on the GPU. The second uses the unchanged `12 x 144 x 144` core and
`6 x 24 x 24` halo for axial fusion, heads, ownership, and NMS, without reading
the source pixels again. Twelve output tensors are submitted in one Burn
transaction per detector block. Temporary completion fragments use
`DeleteOnExit`; the in-memory feature store disappears with the process.

The selected viewer layer is:

`0/tables/yolo-slicefuse3d-cell-centers-fp32-opt-v1`

Fresh release `newvolim-server` discovery reports it as a reviewable 3D object
layer with 159 objects and all 12 columns.

### Measured ablations

All entries are full-volume release CUDA runs on the fixed PBMC dataset. A
profiled run includes deliberate synchronization and is not substituted for the
normal wall-time result. The implementation is based on Git revision
`8b26b82eee9441579ffdb9c90efd4307ae5b051a`; the optimization changes were still
in the working tree during measurement.

| Configuration | Wall time | Rows | Decision |
|---|---:|---:|---|
| Original recorded FP32 reference | 149.94s | 159 | Historical comparison |
| Current direct FP32 baseline | 88.60s | 159 | Current code baseline |
| Profiled direct baseline | 93.86s | 159 | Encoder 55.36s of 83.54s inference |
| Core 24, halo Z 6 | 83.31s | 161 | Reject: boundary-dependent output |
| Core 60, halo Z 6 | 65.43s | 164 | Reject: 7 added and 2 missing centers |
| Halo Z 2 | 86.04s | 162 | Reject: insufficient ownership context |
| Core `24 x 256 x 256`, halo `2 x 24 x 24` | 111.94s | 155 | Reject: slower and changed output |
| Opportunistic per-slice feature cache | 90.08s | 159 | Reject: small encoder batches erase reuse gain |
| Transaction download only, profiled | 92.96s | 159 | Accept: exact, download 2.78s to 2.23s |
| Whole-Z staged cache, duplicate detector read | 92.08s | 159 | Reject implementation: unnecessary second read |
| Whole-Z staged cache, spatial batch 1 | 77.58s profiled | 159 | Accept |
| Staged cache, spatial batch 2 | 87.53s profiled | 159 | Reject: encoder 48.09s |
| Staged cache, spatial batch 4 | 88.22s profiled | 159 | Reject: encoder 47.85s |
| Staged cache, spatial batch 8 | 86.55s profiled | 159 | Reject: encoder 47.75s |
| Full FP16 | >300s, incomplete | — | Reject: over 3.2x profiled FP32 before completion |
| Selected FP32, normal warm median | **76.12s** | 159 | Accept |
| Selected FP32, client-cache cold | **77.15s** | 159 | Accept |

The three normal warm wall times were 77.51, 76.12, and 76.08 seconds. The
selected median is **1.16x faster** than the current 88.60-second direct path
and **1.97x faster** than the original recorded 149.94-second run. The cold run
evicted 2,651 level-0 files totaling 31,112,443 compressed bytes with
`POSIX_FADV_DONTNEED`; the NFS server cache could not be evicted.

Peak host RSS was 1,261,052 KiB. A 319-sample telemetry run measured 4,884 MiB
peak GPU memory, 22.7% mean whole-run utilization, 58% p95, and 65% maximum.
This includes model loading, source reads, finalization, and table publication,
so it is not a kernel occupancy figure.

### Output and trace checks

- Reference and selected outputs have the same 12-column schema and 159 rows.
- Maximum one-to-one center displacement is 0.00007 voxel; maximum box
  coordinate difference is 0.00004 voxel; maximum confidence difference is
  1.2e-8. There are no unmatched detections.
- Because the selected output is equivalent at this tolerance, validation and
  frozen-test center AP retain the checkpoint results of 0.6469 and 0.6261.
- Nsight Systems reports 515,650 CUDA kernel launches and 29.5s aggregate GPU
  kernel time for the direct baseline, versus 258,193 launches and 17.0s for
  the selected path. Direct convolution remains dominant.
- Device-to-host copies occupy 0.128s in the baseline trace and 0.133s in the
  selected trace. Transaction submission removes most host synchronization but
  cannot remove the backend's individual copies. GPU threshold/local-maximum
  kernels were rejected for this version: even eliminating every dense output
  copy would save under 0.2% of end-to-end time, while adding tie-order risk.
- Full FP16 accepted the dtype conversion but did not select a useful tensor
  core path. The run was terminated at 300 seconds. Mixed subgraph modes were
  not pursued because extra casts cannot recover a full-model regression of
  this magnitude on the same convolution backend.

The raw Nsight reports and SQLite exports were removed after their statistics
were recorded because the repository filesystem was full.

## Fixed reference

- Dataset:
  `/husky/otherdataset/teresa/single/clustered-pbmcs.ome.zarr`
- Image group and level: `0/0`
- Shape: `60 x 1,592 x 3,333`, `uint8`
- Checkpoint: `.tmp/yolo-slicefuse3d-pbmc/pilot-1/best.bpk`
- Model configuration:
  `.tmp/yolo-slicefuse3d-pbmc/pilot-1/model-config.json`
- Probability threshold: `0.0052330033`
- NMS IoU: retain the value from the successful production command
- Reference output:
  `0/tables/yolo-slicefuse3d-cell-centers-v1`
- Reference detections: 159
- Reference application time: 149.718 seconds
- Reference wall time: 149.94 seconds
- Reference peak host RSS: 905,136 KiB
- Reference geometry: core `12 x 144 x 144`, halo `6 x 24 x 24`, one worker

The level-0 array occupies 31,112,443 compressed bytes and expands to
318,368,160 bytes. A targeted client-cold release read and Zstd decode took
4.08--6.35 seconds. Storage is therefore a small part of the current inference
time, although every end-to-end benchmark must still report it honestly.

## Measurement protocol

### Build and runtime

- [x] Build every executable with `cargo build --release`.
- [x] Run binaries from `target/release` or use `cargo run --release`.
- [x] Use the same checkpoint, model configuration, normalization, threshold,
      NMS setting, CUDA device and dataset for every comparable run.
- [x] Record the exact Git revision and command.
- [x] Synchronize CUDA before starting and stopping every internal GPU timer.
- [x] Record both application elapsed time and `/usr/bin/time` wall time.
- [x] Record peak host RSS and peak GPU memory.
- [x] Record model load, OME-Zarr read and normalization, encoder, axial fusion,
      head, device transfer, decode/NMS, finalization and table-writing time
      separately.
- [x] Record block count, source voxels requested, unique source voxels, and
      their ratio.
- [x] Capture GPU utilization and an Nsight Systems trace for the baseline and
      final candidate. Use traces for diagnosis, not as timing results.

`/husky` is NFS. Before a cold-cache run, apply
`POSIX_FADV_DONTNEED` to this array's files. This evicts the client page cache
but cannot evict the NFS server cache. Report that limitation with the result.

### Staged evaluation

Use one fixed representative subvolume containing both the crowded center and
sparse exterior cells for rapid experiments. Keep its coordinates in every
result. Do not tune against a different crop for each change.

1. Run one untimed warmup for CUDA compilation and autotuning.
2. Measure at least three repetitions on the fixed subvolume.
3. Evaluate each change against the frozen validation and test partitions.
4. Run the whole volume only for candidates that pass the quality gate.
5. Measure the selected combined configuration at least three times on the full
   volume if runtime and machine availability allow.

Cold-cache end-to-end time and warm steady-state inference answer different
questions, so retain both rather than averaging them together.

## Quality gates

Optimizations that should be numerically equivalent, such as block geometry,
feature reuse and batching, must satisfy all of these:

- [x] The output has the same 12-column object-table schema.
- [x] Every reference center has a one-to-one match within 0.5 voxel on each
      axis, apart from a documented floating-point tie.
- [x] No additional unmatched detections appear.
- [x] Confidence and box coordinates agree within a documented small tolerance.
- [x] There is no increased error rate near block boundaries.
- [x] Validation and frozen-test center AP do not regress by more than 0.002.

FP16 and later model changes are allowed small numerical differences. They must:

- [ ] Retain at least 99% of reference detections under one-to-one spatial
      matching and introduce at most 1% unmatched detections.
- [ ] Recompute the precision-recall curve instead of judging only the fixed
      low threshold.
- [ ] Lose no more than 0.01 center AP on validation or frozen test.
- [ ] Produce a separately named full-volume viewer layer for visual review.

If the reference itself contains false positives, matching it is a regression
test rather than evidence of biological correctness. Preserve the manually
reviewable output and compare quality separately.

## Phase 0: establish the profile

- [ ] Add synchronized timers around encoder, axial fusion and heads.
- [ ] Time each of the twelve per-block tensor downloads separately.
- [ ] Time CPU decoding, NMS, row encoding and final table construction.
- [ ] Record CUDA kernel time, copy time, launch gaps and GPU occupancy.
- [ ] Confirm whether the model mutex leaves CPU preparation idle while the GPU
      runs.
- [ ] Reproduce the 149.718-second baseline before changing execution.

This profile decides how much of the remaining work is encoder arithmetic,
kernel-launch overhead, device synchronization or CPU postprocessing.

## Phase 1: block size and halo ablation

The reference decomposition contains 1,440 blocks and requests
1,007,546,688 source voxels, or **3.1647 times** the unique image volume.

Test block geometry without retraining:

| Core ZYX | Halo ZYX | Approximate full-volume blocks | Source-read ratio |
|---|---|---:|---:|
| `12 x 144 x 144` | `6 x 24 x 24` | 1,440 | 3.1647x |
| `24 x 256 x 256` | `2 x 24 x 24` | determine | determine |
| `24 x 384 x 384` | `2 x 24 x 24` | 135 | 1.4163x |
| `24 x 512 x 512` | `2 x 24 x 24` | 84 | 1.3426x |
| `60 x 384 x 384` | `2 x 24 x 24` | 45 | 1.2497x |

- [ ] Derive the required halo from the actual network receptive field and
      ownership/NMS rules. Do not reduce it solely because a faster run looks
      plausible.
- [ ] Verify that the single `3 x 1 x 1` axial fusion layer requires only one
      neighboring feature slice at each side.
- [ ] Test Z halos 1, 2 and 6 on seam-focused inputs.
- [ ] Increase XY size progressively: 256, 384, then 512 if GPU memory permits.
- [ ] Test Z cores 24 and whole-depth 60.
- [ ] Record out-of-memory boundaries rather than treating them as failed model
      configurations.
- [ ] Select the fastest geometry that passes the exact-output gate.

Target: below 90 seconds on the full volume without changing weights or numeric
precision.

## Phase 2: cache the shared 2D encoder features

The shared YOLO encoder depends on XY pixels within each slice and does not mix
Z. Its P3/P4/P5 tensors should be computed once and reused by overlapping 3D
windows.

### Design

- [ ] Split inference into an encoder phase and a fusion/head phase.
- [ ] Define a typed multiscale feature-store interface containing P3, P4 and
      P5 tensors plus their source coordinates.
- [ ] First implement an FP32 cache so cached and uncached inference can be
      compared exactly.
- [ ] Test a bounded in-memory rolling cache ordered for adjacent XY and Z
      windows.
- [ ] Also test a staged feature pyramid when features must survive process
      boundaries or repeated head/threshold experiments.
- [ ] Keep staged feature metadata tied to source identity, source level,
      normalization, checkpoint hash, model revision and dtype. Reject stale
      entries rather than silently reusing them.
- [ ] Make cache accounting visible to the planner: hits, misses, evictions,
      bytes retained and encoder pixels avoided.
- [ ] Ensure failed runs leave no apparently complete feature cache.

The estimated complete feature volume is roughly 2.2 GiB in FP32 and 1.1 GiB
in FP16. Measure actual sizes and compression. Prefer a rolling in-memory cache
for one inference pass if staging costs more than the encoder work it saves.

### Tests

- [ ] Compare uncached and cached FP32 feature tensors on interior and boundary
      tiles.
- [ ] Compare final detections using several block traversal orders.
- [ ] Measure first-run time, cache-write time and reuse time separately.
- [ ] Confirm that repeated thresholds or heads can reuse the same compatible
      feature cache.

## Phase 3: spatial microbatching

The model already treats Z slices as a 2D batch. It still invokes the model once
per spatial block, producing many small CUDA launches.

- [ ] Add a batch-capable inference interface accepting several equally shaped
      spatial blocks.
- [ ] Bucket boundary blocks by shape or pad them with explicit valid extents.
- [ ] Test spatial batch sizes 1, 2, 4 and 8.
- [ ] Avoid holding the model mutex while reading or normalizing the next batch.
- [ ] Use one model instance and one deliberate GPU queue before testing
      multiple model instances.
- [ ] Pipeline CPU read/normalization of batch N+1 with GPU inference of batch N
      only after measuring that preparation is large enough to matter.
- [ ] Compare batching alone and batching combined with the feature cache.

Do not count two Blockflow workers as GPU batching. The current mutex serializes
model calls; duplicating the model may consume memory without increasing GPU
throughput.

## Phase 4: FP16 and mixed-precision inference

- [ ] Confirm which Burn/CubeCL CUDA convolutions use FP16 tensor-core kernels
      on the Quadro RTX 5000.
- [ ] Implement FP16 feature tensors and convolution weights without changing
      the stored FP32 checkpoint format.
- [ ] Test these modes independently:
  - FP16 encoder with FP32 fusion and heads.
  - FP16 encoder and fusion with FP32 logits/decode.
  - Full FP16 inference with FP32 postprocessing.
- [ ] Keep threshold comparison, coordinate decoding and final NMS in FP32
      initially because the deployed threshold is low.
- [ ] Measure conversion overhead; reject mixed precision if conversions erase
      the kernel gain.
- [ ] Re-run validation and frozen-test precision-recall curves for every mode.

Report kernel selection and measured speed rather than assuming tensor cores
were used because tensors were stored in FP16.

## Phase 5: GPU filtering and sparse download

The current implementation downloads heatmap, offset, size and quality tensors
for P3, P4 and P5 separately. This creates up to twelve synchronized downloads
per block.

Implement this incrementally:

- [ ] Concatenate compatible head outputs so each scale needs one transfer.
- [x] Combine scale transfer submission with a Burn transaction and measure
      synchronization gaps.
- [ ] Apply sigmoid, quality combination and thresholding on the GPU.
- [ ] Perform local-maximum selection on the GPU.
- [ ] Gather coordinates, confidence, class, offset and size only for retained
      candidates.
- [ ] Download one compact candidate buffer per spatial batch.
- [ ] Retain final global NMS on the CPU initially so behavior stays easy to
      compare.
- [ ] Consider GPU NMS only if profiling still shows CPU NMS or transfer as a
      meaningful fraction of the optimized run.

Test candidate ordering and equal-score ties explicitly. Stable identity
assignment must not depend on nondeterministic GPU emission order.

## Phase 6: combined configurations

Maintain an ablation table with at least these rows:

| Configuration | Subvolume time | Full time | Peak GPU | Center AP | Reference matches | Notes |
|---|---:|---:|---:|---:|---:|---|
| Original reference | — | 149.94s wall | — | 0.6261 test | 159/159 | FP32 |
| Current direct reference | — | 88.60s wall | — | 0.6261 test | 159/159 | FP32 |
| Geometry only, whole Z | — | 65.43s wall | — | not accepted | 157/159 | 7 additions |
| Opportunistic feature cache | — | 90.08s wall | — | 0.6261 test | 159/159 | No speedup |
| Staged feature cache, batch 1 | — | 77.58s profiled | — | 0.6261 test | 159/159 | Accepted |
| Spatial batch 2 / 4 / 8 | — | 87.53 / 88.22 / 86.55s | — | 0.6261 test | 159/159 | Slower |
| FP16 only | — | >300s incomplete | — | not evaluated | — | Rejected for speed |
| Transaction download only | — | 92.96s profiled | — | 0.6261 test | 159/159 | Small exact gain |
| GPU sparse decode | — | upper bound <0.2% | — | — | — | Rejected by trace |
| Best exact FP32 combination | — | **76.12s warm median** | **4,884 MiB** | **0.6261 test** | **159/159** | Selected |
| Best accepted mixed precision | — | none | — | — | — | FP16 rejected |

- [x] Combine only individually validated changes.
- [x] Re-profile after every combination; speedups need not multiply.
- [x] Compare cold end-to-end, warm end-to-end and GPU-only time.
- [x] Publish final candidates under distinct names, for example:
  - `yolo-slicefuse3d-cell-centers-fp32-opt-v1`
  - `yolo-slicefuse3d-cell-centers-fp16-opt-v1`
- [ ] Open both beside the current reference in newvolim.
- [x] Preserve the current output until visual and metric review is complete.

Targets:

- Exact FP32 path: below 60 seconds if feature reuse and batching behave as
  expected.
- Mixed-precision path: measure a stretch target below 40 seconds, without
  relaxing the stated quality gate.

Targets guide prioritization and are not results until measured.

## Phase 7: only if the architecture remains compute-bound

After the execution changes above, consider model changes as separate trained
models rather than inference optimizations:

- [ ] Measure the contribution and accuracy of P3, P4 and P5.
- [ ] Train and evaluate a reduced-scale head if one feature level contributes
      little.
- [ ] Distill a narrower encoder from the current SliceFuse model.
- [ ] Evaluate structured channel pruning followed by fine-tuning.
- [ ] Consider INT8 only with a representative calibration set and a separate
      quality report.

Do not start these while redundant encoder work, launch count or dense output
downloads remain dominant.

## Completion criteria

- [x] Every proposed optimization category has an independent measured result or
      a measured upper bound that rejects implementation.
- [x] Rejected changes have a recorded reason: quality, memory, lack of speedup,
      complexity, or nondeterminism.
- [x] The selected combination passes the frozen validation/test metric gate.
- [ ] User visual review of the whole-volume layer.
- [x] `BENCHMARKS.md` contains release-build wall times and simple ratios against
      the 149.94-second reference.
- [x] The example README contains a normal user command with the selected
      defaults, without requiring internal Blockflow APIs.
- [x] Temporary feature stores and work directories are cleaned after successful
      publication and are clearly recoverable after interruption.
- [x] The final object table remains under the image group at
      `0/tables/<layer>` and is visible in newvolim after a fresh discovery.
