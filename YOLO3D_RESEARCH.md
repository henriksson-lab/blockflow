# 3D YOLO distillation research and plan

Date: 2026-09-29.

Primary external references:

- [nnDetection paper](https://arxiv.org/abs/2106.00817) and
  [official repository](https://github.com/MIC-DKFZ/nnDetection)
- [MedYOLO paper](https://arxiv.org/abs/2312.07729) and
  [official repository](https://github.com/JDSobek/MedYOLO)
- [YOLO2U-Net paper](https://arxiv.org/abs/2207.06215)
- [I3D paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)
- [StarDist3D paper](https://arxiv.org/abs/1908.03636)

## Recommendation

Create a **separate derived 3D detector** in the YOLO workspace and a separate
normal Blockflow example such as `yolo-3d-ome-zarr`. Keep the proven 2D model
and its API stable. Share configuration, checkpoint metadata, training history,
optimizer and scheduler code, OME-Zarr sample loading, cache policy, and output
table machinery where the contracts are genuinely common.

The current implementation is concretely 2D: rank-4 tensors, `Conv2d`, four
box distances, `[cx,cy,w,h]` targets, 2D assignment, IoU, augmentation, metrics,
and NMS. Making every component generic over dimension would make the stable 2D
path harder to understand while still requiring separate 3D kernels and target
logic.

## Implementation status

The separate `yolo3d` crate and `yolo-3d-ome-zarr` example now implement the
anisotropy-aware Conv3D center detector described below. Training demand-loads
the complete OME-Zarr volume through Blockflow's bounded decoded cache, records
contiguous spatial splits, excludes incomplete teacher instances, saves the
validation-selected checkpoint, resumes compatible histories, and writes an
SVG loss plot. Inference uses planned overlapping blocks, centroid ownership,
physical-coordinate 3D NMS within each block, and a spatially indexed object
table.

Incoming-data validation, cross-block duplicate measurement, detection
metrics, mixed precision, and 2D checkpoint inflation remain subsequent work.

## What to build

### Model

Use an anisotropy-aware 3D backbone and feature pyramid derived from YOLOv11,
with P2/P3/P4 features for small dense nuclei. Early kernels and strides should
reflect voxel spacing:

- use `(1,3,3)` kernels and `(1,2,2)` strides while Z is coarsely sampled;
- introduce Z convolution and downsampling only once feature spacing is
  reasonably balanced;
- keep patch depth, height, and width independent instead of forcing cubes.

Start with a center-based head:

- one center heatmap per class;
- three sub-voxel center offsets;
- three object dimensions in physical or voxel coordinates;
- optional predicted localization quality or teacher confidence.

Local heatmap maxima suit a dense, mostly single-class cell population and
avoid producing many equivalent six-distance distributions at every feature
location. Keep a six-distance 3D DFL head as a later controlled comparison if
YOLOv11 parity becomes valuable.

This follows lessons from [nnDetection](https://arxiv.org/abs/2106.00817),
which derives kernels, strides, patches, and anchors from spacing and object
size. Avoid mandatory cubic resampling: [MedYOLO](https://arxiv.org/abs/2312.07729)
reports distortion, XY resolution loss, and high memory use from converting
anisotropic scans to fixed cubes.

### Initialization

Inflate the successful 2D DAPI weights:

- map a `k x k` kernel to `kz x k x k` by repeating along Z and dividing by
  `kz`;
- map `1 x 1` layers to `1 x 1 x 1`;
- copy normalization parameters;
- initialize the 3D output head anew.

This preserves the learned fluorescent-nucleus features at initialization.
2D-to-3D inflation is established by the
[I3D work](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html).
Train scratch and inflated variants as an explicit ablation rather than assuming
the transfer helps.

## Targets and retained teacher data

Generate targets from connected 3D Cellpose, StarDist, or u-Segment3D
instances. Store:

- stable teacher instance ID and teacher method/version;
- center in voxel and physical coordinates;
- full 3D bounding box and object extent;
- teacher confidence or agreement score where available;
- whether an instance is complete or truncated at a sampled patch boundary;
- the original instance mask reference.

Train on every complete object visible in the input patch and ignore objects
truncated by its boundary. Use ownership cores to deduplicate validation and
inference output. Restricting training targets to the ownership core labels
visible halo cells as background and creates contradictory supervision across
overlapping patches. Retaining masks allows an auxiliary foreground or
center-distance loss and a later ROI segmentation head.

[YOLO2U-Net](https://arxiv.org/abs/2207.06215) provides microscopy precedent
for detection-guided 3D segmentation: detections from orthogonal 2D views form
3D proposals which a 3D U-Net refines. Its evidence is synthetic, so it is a
useful design reference rather than a production-quality benchmark.

## Blockflow training and inference

Reuse Blockflow for volumetric sample reads and cache locality, not for forcing
training into independent image-processing tasks.

Training should:

- demand-load `[z,y,x]` patches from OME-Zarr;
- shuffle small spatial groups, then samples within them, to keep the cache warm
  without making SGD order fixed;
- retain a bounded decoded cache rather than imposing a cap on training tiles;
- sample backgrounds and density regimes deliberately;
- record exact spatial splits and concatenate histories across resumed stages;
- use dataset-specific augmentation policies.

Inference should use planned volumetric blocks with independent Z and XY sizes,
a halo covering the upper tail of cell radii, center ownership, 3D spatial
hashing, and physical-coordinate 3D NMS or weighted box fusion. Do not impose a
small global detection cap. Bound candidates per feature level or per spatial
cell, then merge spatially.

## Anisotropy and augmentation

Fluorescence biology may be orientation-independent while the acquisition is
not. Axial spacing, point spread, attenuation with depth, and missing slices can
make Z transformations invalid even for isotropic cells.

Keep these controls separate per dataset:

- XY flips and right-angle rotations;
- Z reflection;
- arbitrary 3D rotations, enabled only after suitable isotropic resampling;
- intensity scaling, gamma, Poisson/read noise, blur, and axial blur;
- Z dropout or missing-slice simulation;
- scale jitter in physical rather than raw voxel units.

## Splits and evaluation

Millions of teacher objects from one volume do not provide millions of
independent biological examples. Split in this order:

1. hold out complete specimens or acquisition sessions;
2. if only one volume exists, reserve large contiguous 3D regions with guard
   bands wider than the patch plus halo;
3. stratify by density and signal while keeping adjacent patches together;
4. keep a small human-reviewed 3D test set independent of the teacher.

Report teacher agreement separately from biological accuracy. The student may
smooth isolated teacher errors, but supervised loss reinforces systematic
misses, merges, splits, and false objects unless corrected labels or another
signal contradict them.

Useful first metrics are center average precision at physical distance
thresholds, 3D box AP, count error by density, recall by object size, and seam
stability under two tilings. When masks are refined, add 3D instance IoU and
boundary metrics.

### Clustered PBMC volume

The central crowded patch is a distinct difficulty regime and needs a grouped
spatial split. Do not randomly split overlapping training windows there. Assign
whole teacher objects and all windows that can see them to one partition, with
a guard at least one patch plus halo wide between partitions. Prefer a complete
central subregion as the frozen test set and a separate central subregion for
validation; keep the remaining central objects in training. Choose the exact
bounds from the completed Cellpose object distribution so each partition has
enough crowded cells.

Outside the central patch, randomized splitting is appropriate after grouping
nearby windows into non-overlapping spatial macrotiles. Stratify those groups
by object count, size, and signal, then assign groups randomly. This preserves
the variety of scattered blobs without leaking the same object or overlapping
context between train, validation, and test. Report metrics separately for the
central crowded region and the scattered region.

## Baselines before a full 3D model

1. **Slice linking:** run the existing 2D model on XY slices and link boxes or
   masks through Z.
2. **2.5D:** present neighboring slices as input channels while retaining a 2D
   head.
3. **Orthoview consensus:** run 2D inference on useful XY/XZ/YZ views and fuse
   them with u-Segment3D-style consensus.
4. **Native teachers:** compare Cellpose3D and StarDist3D on manually reviewed
   crops.

These determine whether true 3D context earns its memory and engineering cost.
They are also useful fallback paths for volumes whose Z sampling is too coarse
for native 3D convolution.

## Approximate engineering effort

| Deliverable | Estimate |
|---|---:|
| XY slice-linking baseline | 3–5 working days |
| 2.5D training and evaluation baseline | 1–2 weeks |
| True 3D model, targets, losses, NMS, training, and Blockflow inference MVP | 3–6 weeks |
| Mixed precision, memory optimization, viewer work, refinement, and controlled ablations | another 3–6 weeks |

These estimates assume Burn's Conv3D, resize, pooling, normalization, and
autodiff paths meet the required performance. Check them with small release
microbenchmarks before committing to the model implementation.

## Validation and extension sequence

Cellpose3D and StarDist3D inference optimization is complete for this project.
Their matched GPU speed is accepted; do not divert the YOLO3D validation plan
back into teacher-runtime optimization without evidence of a material
regression.

1. Produce a reviewed teacher comparison on the incoming volume.
2. Preserve the chosen teacher masks and create a reusable 3D target table.
3. Train the implemented scratch model and establish slice-linking and 2.5D
   baselines.
4. Add a 2D checkpoint inflation importer as a controlled initialization
   option.
5. Compare scratch versus inflation and 2.5D versus true 3D on frozen spatial
   splits.
6. Run full-volume detection only after the small-volume quality and memory
   gates pass.
7. Add optional per-box mask refinement for accurate cell shape and channel
   quantification.

## Production direction

For compact fluorescent nuclei, a fast 3D detector is a plausible production
localizer and counter. Boxes alone remain inadequate for crowded-cell intensity
measurement. The durable product should therefore retain both outputs: fast
YOLO detections for navigation and proposals, plus instance masks from the
teacher or a learned local refinement stage for quantitative analysis.

## Clustered PBMC implementation result

The first native Conv3D implementation is now validated on the complete PBMC
volume. Its data split keeps overlapping inputs in one partition, holds the
single crowded central component in training, balances independent exterior
components across validation and test, and adds deterministic empty windows.
The empty windows were necessary: the first positive-only run generated
millions of background candidates during full inference.

The final trainer uses class-specific CenterNet peaks, focal normalization over
all scales, smooth L1 regression of log physical extent, Nesterov momentum,
gradient clipping, and median physical size priors estimated from complete
training objects for each feature scale. It selects checkpoints by center AP
and reports fixed threshold operating points. The selected model reached center
AP 0.7298 on validation and 0.6233 on the untouched test partition against the
Cellpose teacher.

Full release CUDA inference produced the indexed table
`0/tables/yolo3d-cell-centers-v1` in 283.572 seconds, 34.4 times faster than the
full Cellpose3D teacher workflow by application elapsed time. The 74 detections
are useful for navigation, but validation recall of 0.7222 and low box AP50 show
that this checkpoint is not yet a replacement for the 115-instance Cellpose
mask annotation. The next quality work should use manual review and additional
independent volumes rather than tuning the frozen test partition.
