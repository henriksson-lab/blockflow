# YOLO DAPI training plan

## Current evidence

The current detector is a fast baseline trained from Cellpose-derived DAPI
annotations. The best validation mAP from the focused sweep was `0.058433`,
compared with `0.058288` for the unchanged warm-start checkpoint. Head-only
training, weight decay, and a higher learning rate did not improve it. Leading
trials peaked after one or two epochs and then degraded. The frozen test
partition has not been used to choose parameters.

This is too little improvement to justify a broader optimizer sweep. The next
work should establish whether the model and target representation can learn the
task, and whether dense nuclei violate assumptions in the detector or its
postprocessing.

## Current implementation findings

The present Rust postprocessor contains limits that can directly invalidate a
dense-cell evaluation:

- It returns at most **300 detections per image** (`MAX_DET`), while sampled
  DAPI crops already reach 287 targets. Any tile with more than 300 true cells
  has a recall ceiling of `300 / target_count`, even with perfect model output,
  and current dense tiles leave almost no allowance for duplicate predictions.
- It retains at most 30,000 candidates before NMS. This does not bind for the
  present one-class model, which emits 8,400 candidates.
- CPU greedy NMS is quadratic in the retained candidate count.
- NMS has one wall-clock deadline for the whole batch (`0.5 + 0.05 * batch`
  seconds). Once exceeded, it stops processing and leaves all remaining batch
  entries empty. Dense early images can therefore reduce apparent recall for
  unrelated later images.
- Evaluation currently uses confidence `0.001` and NMS IoU `0.65`. The low
  confidence threshold can send many candidates into the quadratic CPU path.
- Training uses task-aligned assignment with ten candidate anchors per target.
  At the actual 640-pixel model input, the stride 8, 16, and 32 levels expose
  8,400 locations (`80*80 + 40*40 + 20*20`), including 6,400 locations on the
  small-object level. Every location can ultimately be assigned to only one
  ground-truth object. Crowded objects compete for nearby fine-stride locations
  well before the raw 8,400-location bound is reached.
- Each sample image is 640 by 640 pixels: a 512-pixel ownership core plus a
  64-pixel halo on each side. The current loader labels only objects whose
  centroids fall inside the 512-pixel core, but the YOLO loss treats every
  unmatched location across the full 640-pixel image as background. The halo
  occupies 36% of the input area, so many visible nuclei are systematically
  presented as false-negative background. Ownership deduplication is useful
  when assembling final slide output; it should not remove visible labels from
  a training crop. Include every sufficiently complete visible halo object in
  training and evaluation targets, or add an ignored-anchor mask for unlabeled
  halo regions.

Before the density report, fix or experimentally disable the halo-label issue.
Then compare target counts with the 300-detection cap and record whether the NMS
deadline is reached. Make both limits explicit configuration, remove the
batch-wide early exit during correctness evaluation, and record pre-NMS recall
before interpreting the current mAP as model quality.

The legacy centroid table shows how close this dataset is to the output cap:
occupied 512-pixel ownership tiles have mean 73.2 targets, p90 195, p95 214,
p99 275, and maximum 381. Of 6,701 occupied tiles, 25 exceed 300 targets. The
current maximum therefore has a best possible recall of 78.7% after the
300-detection cap, independent of model quality. Use a configurable microscopy
default of at least 1,000 while measuring correctness.

Task-aligned assignment allocates at most ten positive candidates to each
object and resolves a contested location to one object. Instrument candidate
locations per target, positives remaining after conflict resolution,
zero-positive target fraction, and conflict rate by density and object size. At
batch 16, 381 ground truths and 8,400 locations make one dense assignment tensor
about 205 MB; several such tensors coexist, so density also has a real GPU
memory and compute cost.

The current spatial bands also have no guard band. Adjacent samples can expose
the same tissue through their 64-pixel halos across a train/validation boundary.
Split by larger spatial blocks or add a guard at least as wide as the halo.

## What the detector literature predicts

Dense nuclei are not categorically beyond modern YOLO, but this workload is far
from its common benchmark distribution. COCO contains roughly 7.6 annotated
instances per image, while occupied DAPI tiles average 73 and reach 381. The
original YOLO paper called out grouped small objects as a limitation. Modern
anchor-free heads remove the original fixed two-box-per-cell rule, but they
still emit one box per feature-map location and assign a contested location to
one target.

Task-Aligned Learning (TAL) is intended to align classification and localization
when selecting positive samples. Its top-k assignment still creates competition
when many similar small targets share nearby candidate locations. Crowded-object
research identifies ambiguous assignment and suppression by NMS as distinct
failure modes, which is why pre-NMS and post-NMS recall must be measured
separately.

The median nucleus mask area is about 424 pixels, corresponding to a circular
diameter near 23 pixels, while the first percentile is about 38 pixels, or a
7-pixel diameter. A median nucleus spans approximately three stride-8 cells, so
the existing P3 head is plausible. The small tail may need a P2 stride-4 head or
greater source magnification. Test those only after correcting labels and NMS.

Axis-aligned boxes are intrinsically problematic in crowded microscopy: two
distinct masks may have strongly overlapping boxes, causing valid objects to
suppress one another. A YOLO segmentation head still relies on box assignment
and box NMS, so masks alone do not necessarily remove this ceiling. A center,
flow, or polygon representation may fit the problem better.

Primary references:

- [YOLO: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [TOOD: Task-aligned One-stage Object Detection](https://arxiv.org/abs/2108.07755)
- [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
- [Cell Detection with Star-convex Polygons](https://arxiv.org/abs/1806.03535)
- [Density-based Object Detection in Crowded Scenes](https://arxiv.org/abs/2504.09819)
- [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://arxiv.org/abs/2202.06934)
- [Ultralytics YOLO11 model definition](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml)
- [Ultralytics task-aligned assigner](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py)

## 1. Prove that the target and evaluation pipeline is correct

Before more training, add a diagnostic command that operates on the exact
post-resize tensors and targets supplied to the model.

1. Render Cellpose-derived boxes on 50 representative model inputs, including
   dense, sparse, edge, bright, and dim tiles.
2. Record per-tile object counts, box widths and heights, area, aspect ratio,
   nearest-neighbour distance, pairwise overlap, and the fraction of objects
   clipped by a tile boundary.
3. Pass the ground-truth boxes through the evaluation path as perfect
   predictions. Recall, precision, and mAP should be approximately 1.0.
4. Round-trip targets through every coordinate conversion, resize, padding,
   batching, and decoding stage and compare coordinates numerically.
5. Verify that the validation checkpoint is evaluated with the same class and
   geometry conventions used during training.

A failure here means subsequent training results cannot be interpreted.

## 2. Run a deliberate tiny-set overfit test

Use 16 to 32 representative tiles and evaluate on those same tiles. Disable D4
and all intensity augmentation. Keep BatchNorm frozen initially and use a
learning rate low enough to remain finite, but train long enough to fit the
examples. Save training loss, decoded predictions, and mAP regularly.

Interpretation:

- High training mAP shows that the translated model, loss, target assignment,
  and decoder can represent the examples. Continue with sampling and
  generalization work.
- Falling loss with persistently low mAP suggests a mismatch in decoding,
  evaluation, confidence calibration, or box geometry.
- Failure to reduce training loss suggests a problem in gradient flow, target
  assignment, loss normalization, frozen parameters, or the translated model.
- Good performance on sparse tiles but poor performance on dense tiles points
  to a density or small-object capacity limit.

Run the same experiment on a sparse subset and a dense subset. This isolates
object count from image appearance without committing to a large training run.

## 3. Measure the effect of object density directly

Object count is a primary hypothesis, not just a dataset statistic. Stratify
validation tiles by target count and report recall, precision, mAP, unmatched
targets, and retained detections for each stratum. Also report these metrics by
object size and nearest-neighbour distance.

Inspect the following possible limits:

- The number of candidate locations available at each detection stride.
- How many targets compete for the same locations during task-aligned matching.
- Any fixed top-k candidates per target or per feature level.
- Loss normalization as the number of foreground targets grows.
- Decoder confidence filtering and any fixed maximum number of detections.
- Class-agnostic NMS suppression among tightly packed or overlapping nuclei.
- Loss of small-object information at the finest detection stride.

Test the hypothesis with controlled evaluation changes before retraining:

1. Feed ground-truth boxes through NMS and measure its maximum attainable
   recall before involving the model.
2. Raise the maximum detections well above the largest target count per tile.
3. Require identical metrics for evaluation batch sizes 1 and 16; a difference
   exposes the current batch-wide NMS deadline.
4. Sweep confidence and NMS IoU thresholds on validation predictions.
5. Compare ordinary NMS, soft NMS, and no NMS for recall and duplicate rate.
6. Plot recall against objects per tile and nearest-neighbour distance.
7. Evaluate predictions before NMS to distinguish model capacity from
   postprocessing suppression.

If pre-NMS recall remains good while post-NMS recall collapses with density,
fix postprocessing. If pre-NMS recall also collapses, investigate assignment,
feature resolution, and model capacity.

## 4. Decide whether bounding boxes are an adequate target

Cellpose supplies instance masks, while the current YOLO workflow reduces them
to bounding boxes. Dense nuclei can have strongly overlapping boxes even when
their masks remain distinguishable. Boxes also provide a poor region for later
per-cell channel quantification.

Compare three output representations on the same tiles:

1. Bounding boxes from the current detector.
2. Center or point detections followed by local mask reconstruction.
3. A YOLO instance-segmentation head trained from the Cellpose masks.

If box overlap is the main cause of suppression or ambiguous assignment, prefer
instance segmentation or center prediction rather than spending more time on
box-detector hyperparameters. Preserve stable object identifiers and mask
geometry in the NGFF object-table output so later channel measurements can be
attached to the same objects.

## 5. Improve sampling after the model passes the overfit test

Build a deterministic, spatially balanced sampler that includes dense, medium,
sparse, and 10% to 20% empty regions. Keep validation and test partitions spatially
separated from training. Use overlapping source windows for training, but assign
objects consistently so that border objects are not duplicated or repeatedly
presented as truncated targets.

The decoded OME-Zarr cache and source-neutral sample store can be reused. The
planner is useful for deterministic coverage and cache-aware ordering; epoch
sampling and augmentation should remain in the training loader.

## 6. Test microscopy-specific input normalization

Run controlled comparisons rather than combining several changes:

- Dataset-wide DAPI percentile normalization.
- Per-tile percentile normalization.
- Log or gamma compression.
- Replicated grayscale input versus a learned single-channel input stem.

Record intensity statistics by spatial region. Per-tile normalization can erase
biologically meaningful density or background differences, while global
normalization may underuse the available range in dim regions.

## 7. Add conservative fluorescence augmentation

D4 geometry is valid for the current fluorescence data. Add intensity scale,
gamma, background offset, and mild Poisson or Gaussian noise as separately
controllable policies. Avoid mosaic initially because artificial seams and
changed cell density can interact badly with dense-target assignment.

Augmentation policy belongs to the dataset configuration. DIC and phase
contrast data should default to no rotation or reflection for a fixed optical
configuration, while users training across lens configurations may opt into
those transforms deliberately.

## 8. Compare speed only at a useful quality point

Once a YOLO output reaches an agreed recall or segmentation-quality threshold,
compare end-to-end runtime with original Cellpose and blockflow Cellpose on the
same regions. Report preprocessing, GPU inference, postprocessing, annotation
writing, and total wall time separately. A faster detector is useful only if its
output quality supports the intended visualization and per-cell measurements.

## Immediate execution order

1. Perfect-prediction evaluation and coordinate round-trip checks.
2. Correct halo supervision by including visible instances or masking ignored
   regions.
3. Density, overlap, and border-target report for the existing split.
4. Remove the batch-wide NMS timeout and raise the 300-detection cap for
   correctness evaluation.
5. Sparse-versus-dense tiny-set overfit runs.
6. Pre-NMS versus post-NMS density analysis.
7. If corrected P3 training can overfit, compare one P2 stride-4 head and one
   higher-magnification input experiment on the same fixed split.
8. Decide between repairing box detection and adding an instance-segmentation
   or center-prediction path.
9. Only then test sampling, normalization, and fluorescence augmentation.

## Execution update (2026-09-26)

Steps 1 through 5 have now been exercised far enough to expose the next
correctness issue.

- Halo supervision now includes every complete visible object in the 640 by
  640 input. A halo-width guard separates the training and validation bands.
- Ground-truth predictions pass the metric path, and validation metrics are
  invariant between evaluation batch sizes 1 and 16.
- Evaluation NMS has explicit confidence, IoU, candidate, and detection limits.
  The batch-wide wall-clock abort was removed. Microscopy evaluation uses a
  1,000-detection limit.
- The corrected 512/128/128 train/validation/test inspection contains 60,262,
  10,705, and 14,982 targets. Sixty training tiles exceed 300 targets; the old
  cap limited aggregate training recall to 94.64%. Ground-truth box NMS at IoU
  0.65, 0.80, and 0.95 retained all objects when the cap was raised.
- Corrected validation of the retained checkpoint is mAP 0.06992, mAP50
  0.19674, recall 0.21709, and precision 0.36820.
- Sparse and dense 16-tile overfit sets contain 431 and 7,555 visible targets.
  Their retained-checkpoint baselines are mAP 0.0194 and 0.1143. Initial 100
  update runs failed to overfit: best mAP was about 0.023 sparse and 0.119
  dense, followed by degradation.
- Actual task-aligned assignment profiling found zero selected positives for
  74/431 sparse targets and 176/7,555 dense targets. The stride-8 floor is
  material for the sparse small-object tail, but does not explain the dense
  overfit failure.
- Direct Rust/PyTorch synthetic training isolated the translated loss. The
  optimizer and general autodiff path reduce a simple output-sum objective, and
  DFL moves in the expected direction. Classification and box-only traces were
  unstable at learning rates where PyTorch decreased smoothly.
- The tensor assigner already detached predictions internally, as required by
  PyTorch's `no_grad` assigner. The call boundary now makes this invariant
  explicit as well.
- Rust computed CIoU and DFL for every anchor and multiplied background results
  by zero afterward. PyTorch selects foreground anchors before either loss.
  The Rust path now constructs one foreground index tensor and computes both
  losses only on selected anchors, avoiding invalid background CIoU derivatives
  and reducing work. The focused release loss suite passes 17/17 tests.

The foreground-only synthetic rerun did not fix the training direction: at
batch 8, 32 targets per image, and learning rate `1e-5`, combined loss moved
from 18.935 to 19.305 over ten updates. Steady-state loss calculation remained
about 9 to 12 ms, so foreground selection did not create a meaningful runtime
regression. Do not spend another dense overfit run on this unchanged behavior.
The next gate is to record parameter and gradient norms for each head branch
and compare a single fixed-assignment update with PyTorch before changing the
architecture or training hyperparameters.

### Training-backward correction

The fixed-output loss-gradient fixture subsequently showed that the Rust and
PyTorch box, classification, DFL, and combined losses were already equivalent:
losses agreed within `4.3e-7`, per-scale gradient cosines were effectively
`1.0`, and the largest gradient-element error was `2.9e-8`. The earlier
synthetic optimizer comparison was invalid because it compared independently
initialized networks.

A matched-weight full-model fixture then isolated the failure below the loss.
The forward outputs had cosines above `0.99999`, but the Rust input gradient had
L2 norm `28,740.6` versus PyTorch's `596.5`, with cosine `0.00925`. The local
Burn libtorch SiLU override called `grad.silu_backward(input)`; libtorch expects
`input.silu_backward(grad)`. Reversing those arguments restored full-model
gradient parity. With fused BatchNorm, input-gradient cosine is
`0.9999999996` and L2 norms are `0.986398` versus `0.986425`. With training
BatchNorm, cosine is `0.99954`; the remaining small difference is numerical and
does not change the gradient direction.

The corrected 16-tile dense overfit run used the same retained checkpoint,
targets, frozen BatchNorm, no augmentation, batch 8, gradient clipping at 10,
and `5e-6` to `5e-5` schedule as the broken run. In 100 optimizer updates it
reached best mAP `0.473`, compared with `0.119` before the fix. Wall time was
203.9 seconds. Losses continued downward, so the deliberately conservative
transfer learning rate is now the likely limit. The next bounded experiment is
a tenfold-higher `5e-5` to `5e-4` dense overfit run, followed by the sparse
overfit comparison.

Both follow-ups passed. The higher-rate dense run reached best mAP `0.570` in
204.8 seconds. The matching sparse run reached best mAP `0.547` in 116.3
seconds, compared with `0.023` before the backward fix. This proves that the
existing P3 detector, target representation, loss, and optimizer can learn both
density regimes. A P2 head may still improve the small-object tail, but it is no
longer a prerequisite for starting a corrected spatially separated training
run. The next run uses 256 training tiles, 64 validation tiles, D4, frozen
BatchNorm, batch 16, and an 80-epoch schedule executed in 20-epoch stages. The
held-out test partition remains untouched until model selection is complete.

### Corrected spatial pilot result

The corrected 256/64/64 spatial run completed its fixed 80-epoch schedule in
four exactly resumed 20-epoch stages. Validation mAP rose from `0.3714` at
epoch 20 to `0.4126` at epoch 40, `0.4337` at epoch 60, and a best of `0.4419`
at epoch 72. The selected checkpoint had validation mAP50 `0.7164`, recall
`0.6534`, and precision `0.7553`. The curve plateaued over the last eight
epochs, so extending this schedule is not the next useful experiment.

After model selection, the frozen central test partition was evaluated once.
It reached mAP `0.5457`, mAP50 `0.8114`, recall `0.7376`, and precision
`0.8094`. The four successful stages took about 837 seconds of process wall
time in total, including validation every epoch and the final test evaluation.
The run, checkpoints, exact split, concatenated history, and SVG are in
`.tmp/yolo-dapi-training/corrected-pilot-256-e20-lr5e-4`.

This is sufficient evidence that the corrected trainer learns on a spatial
split. The next quality experiment should increase the number and diversity of
training tiles while preserving a new frozen test split. Architecture work on
a stride-4 P2 head remains a targeted follow-up for the smallest nuclei rather
than a prerequisite for using the training path.

### Full-image streaming baseline

The next run uses all eligible occupied training tiles instead of keeping a
fixed small subset in memory. Ten percent of each partition is made from
conservative background tiles whose neighboring ownership tiles also contain
no labeled centroid. OME-Zarr samples remain demand-loaded through the bounded
cache. Each epoch shuffles four-by-four spatial blocks and then shuffles within
each block deterministically, preserving useful read locality without fixing
the SGD order.

Inspection selected 2,004 training tiles with 209,318 targets, including 200
background tiles. Validation and frozen test each contain 256 tiles, including
26 backgrounds. A total of 6,585 training targets, 3.1%, have no stride-8
assignment candidate. The full P3 baseline should run first; this measurement
then gives the P2 experiment a specific small-object target.

The first 20-epoch stage of that baseline completed in 1,444.8 seconds. Its
best result was the final epoch: validation mAP `0.4804`, mAP50 `0.7671`,
recall `0.6797`, and precision `0.7737`. This improves on the warm-start
checkpoint over the expanded validation split, so exact resumed training
continues.

The full 80-epoch run completed in four stages and about 5,014 seconds of
process wall time. The final epoch was the validation-selected best: mAP
`0.5295`, mAP50 `0.7973`, recall `0.7037`, and precision `0.8004`. Its single
evaluation on the frozen 256-tile test split reached mAP `0.6046`, mAP50
`0.8552`, recall `0.7755`, and precision `0.8329`. The result is a useful P3
baseline for the later stride-4 comparison.

These metrics measure imitation of Cellpose-derived nucleus boxes. The current
model receives the axis-aligned box of each complete Cellpose instance and
does not reconstruct its pixel mask. Isolated boundary noise can be averaged
away by the detector, while systematic missed, merged, split, or false
Cellpose instances are reinforced by the supervised loss. Production-quality
biological evaluation therefore needs a small independently reviewed set.

### Full-image result

The selected epoch-80 model was applied to the complete level-0 DAPI plane at
the validation-calibrated confidence threshold `0.42642644`. It processed
39,732 blocks in 666.03 seconds of fragment execution and 675.32 seconds
end-to-end. The merge removed 999 window-seam duplicates and retained 468,345
detections. Peak resident memory was about 12.2 GiB.

The durable output is the indexed object table at
`/husky/otherdataset/teresa/2079_merged_registered.zarr/tables/yolo-dapi`.
newvolim currently reads its `table.csv` compatibility copy; it discovered all
rows and returned 893 objects for a representative 1000 by 1000 pixel viewport.
The native table remains the intended path for partial analysis once the viewer
connects its object-table reader.

### PBMC watershed-teacher YOLO3D comparison

For `single/clustered-pbmcs.ome.zarr`, train a second YOLO3D model from the
classical seeded watershed pseudo-labels. Keep it separate from the existing
Cellpose-teacher YOLO3D model in checkpoints, histories, configuration, and
published layer names. After validation, apply the selected watershed-teacher
model to the full level-0 volume and publish its indexed detections as
`yolo3d-watershed-centers-v1` or another explicitly watershed-derived name.

The viewer must be able to show that layer beside the existing
`yolo3d-cell-centers-v1` layer for visual comparison. Use the same inference
tiling and duplicate-merge policy for both models so visible differences are
attributable to the models and their teachers rather than to postprocessing.
Keep spatial train, validation, and test partitions disjoint, and exclude
crop-truncated watershed instances from supervision.
