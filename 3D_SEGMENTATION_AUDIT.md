# 3D fluorescence segmentation audit

Date: 2026-09-29, updated with real-data runs on 2026-09-30.

## Answer

Blockflow now has end-to-end 3D StarDist and Cellpose examples. The source audit
below records the state before implementation; the implementation status table
has been updated so the document remains useful as a checklist.

| Layer | StarDist | Cellpose |
|---|---|---|
| Pinned upstream crate | Has `Config3D`, `StarDist3D`, 3D networks, sparse and dense 3D NMS, polyhedron rasterization, and 3D training types | Has `eval_3d`, orthogonal-view inference, anisotropy handling, 3D flow dynamics, and slice-stitching mode |
| Blockflow backend | `Stardist3dBackend` uses the native Candle CUDA 3D network, rays, NMS, and renderer | `Cellpose3dBackend` enables orthoview 3D inference and anisotropy |
| OME-Zarr example | `stardist-3d-ome-zarr` selects named axes and plans independent Z/XY blocks | `cellpose-3d-ome-zarr` uses the same shared path on CPU or CUDA |
| Output finalization | 3D label pyramid and `centroid_z/y/x`, voxel/physical volume table | Same shared 3D finalizer |
| End-to-end 3D support now | **Validated on a real PBMC crop** | **Validated on a real crop and the complete PBMC volume** |

## What is already reusable

Blockflow's execution model is rank 3 and already treats a 2D image as a
one-voxel-deep volume. The following pieces are dimensionally suitable:

- `BlockGrid`, planning, cache, prefetch, and execution over `[z,y,x]`;
- `InstanceSegment` with a three-axis halo, centroid ownership, stable IDs,
  volumetric voxel accumulation, and a dense `u64` label output;
- the nearest-neighbor label pyramid builder, including downsampling on all
  three spatial axes;
- `ngff-object-table`, whose spatial-index rank is dynamic and can store
  `z,y,x` coordinates and 3D occupancy pyramids;
- OME label metadata already declares spatial axes `z,y,x`.

This means that the planner and scheduler do not need a separate 3D design.
Model invocation, source slicing, final table columns, and seam behavior do.

## StarDist status

The pinned `stardist-rs` revision contains real StarDist3D support:

- 3D U-Net or ResNet configuration and anisotropic kernels, pooling, grid, and
  ray geometry;
- Burn and Candle 3D networks;
- sparse candidate prediction and 3D non-maximum suppression;
- star-convex polyhedron labels and 3D training data structures.

The original Blockflow adapter remains the explicit 2D path. The new
`Stardist3dBackend` keeps the distinct 3D model, output layout, NMS, and
polyhedron geometry contract visible to callers.

The main performance risk is postprocessing. Notes in the pinned upstream
checkout report roughly 0.10 seconds for raw CUDA inference on its 3D fixture,
0.17 seconds for sparse prediction, and about 172 seconds for polyhedron
postprocessing. Those figures are not a Blockflow benchmark, but they make 3D
NMS and polyhedron rendering the first component to profile before a large run.

There is no official general pretrained StarDist3D model. The official FAQ
says 3D is supported, including anisotropic data, but training data scarcity
has prevented a general pretrained release. A suitable BioImage Model Zoo
model or dataset-specific training is therefore needed. See the
[StarDist repository](https://github.com/stardist/stardist),
[StarDist FAQ](https://github.com/stardist/stardist-docs/blob/main/docs/faq.md),
and the [StarDist3D paper](https://arxiv.org/abs/1908.03636).

## Cellpose status

The pinned `cellpose-rs` revision contains both of Cellpose's volumetric modes:

1. `eval_3d` runs the 2D network in three orthogonal orientations, combines
   the predicted components into a 3D flow field, and performs 3D dynamics.
2. `eval_stitch_stack` segments XY slices independently and links masks between
   adjacent slices using an IoU threshold.

It also contains anisotropic rescaling, optional 3D flow smoothing, 3D mask
dynamics, hole filling, and volumetric minimum-size filtering. This follows the
official Cellpose approach; it is not a native Conv3D network. The
[Cellpose 3D documentation](https://cellpose.readthedocs.io/en/latest/do3d.html)
describes the same orthoview and anisotropy controls.

The original Blockflow adapter remains the explicit 2D path. The separate
`Cellpose3dBackend` enables `do_3d` while preserving the model parameters and
anisotropy setting. Slice stitching remains a separate upstream mode because
its seam and ownership semantics differ from 3D flow reconstruction.

## Implemented shared infrastructure

### Select a channel from a volumetric OME-Zarr

`AttachedImage::volume` now maps a rank-3 to rank-5 stored array to `[z,y,x]`,
and `OmeVolumePyramid` resolves the named OME axes, channel, time point, level
shapes, physical spacing, and projected spatial transforms once for all three
examples.

The axis-aware stored-array view:

- reads OME axis names rather than assuming their positions;
- fixes non-spatial axes such as `t` and `c` at caller-selected indices;
- exposes spatial axes in Blockflow order `[z,y,x]` without copying;
- maps block regions back to the source array and retains chunk accounting.

This is implemented once in `zarr_env` and shared by StarDist, Cellpose, YOLO,
and later measurement examples.

### Finalize a 3D object table

`finalize_instances_3d` now writes:

- `centroid_z`, `centroid_y`, `centroid_x`;
- `volume_voxels` and `volume_um3` using all three physical scales;
- a 3D spatial index and occupancy pyramid;
- optional bounding box columns for detection teachers and students;
- the same stable identity as the dense label volume.

The storage crate already accepts an arbitrary nonzero spatial rank, so this is
a Blockflow producer change rather than a format redesign.

### Define volumetric seams

Centroid ownership and a halo at least as large as the largest object extent
generalize to 3D. The practical problem is that a full Cellpose or StarDist
object near a tile face must be reconstructed consistently in both overlapping
tiles. The first implementation should report seam duplicate and truncated
object counts, compare two block decompositions, and keep model-internal tiling
disabled when Blockflow owns the outer subdivision.

## Validation order

1. Confirm axes, spacing, channel selection, and model weights on a small crop.
2. Compare Cellpose3D and StarDist3D on manually reviewed subvolumes.
3. Profile StarDist inference, candidate NMS, and polyhedron rendering
   separately.
4. Compare two block decompositions and report seam duplicates and truncated
   objects.
5. Choose and preserve a teacher before distillation.

## Real-data validation

On the anisotropic PBMC data, the release CUDA Cellpose path reproduced the
Python object count on both compact validation crops and reached foreground
Dice 0.9874–0.9876. The release StarDist path reproduced the Python StarDist
labels exactly. Timings and ratios are in `BENCHMARKS.md`.

The bundled StarDist `3D_demo` model found two objects in the crowded crop,
while CP-SAM found eight. This does not establish general method quality, but it
does rule out that demo checkpoint as the teacher for this dataset without
further training or review.

The complete Cellpose3D run found 115 objects in the 60 x 1,592 x 3,333
volume and finished in 9,738.827 seconds. Its five-level label pyramid and
indexed table are registered in the source OME-Zarr for direct viewer use.

The YOLO3D distillation workflow was also validated end to end. A grouped split
kept the crowded central component in training, assigned independent exterior
components to validation and frozen test, and added leakage-safe empty windows
to every partition. The selected checkpoint reached center AP 0.7298 on
validation and 0.6233 on frozen test against the Cellpose teacher. Its precise
box quality remains weak, so the output is classified as centers with
approximate extents.

Full YOLO3D inference completed in 283.572 seconds and published 74 detections
to `0/tables/yolo3d-cell-centers-v1`. This is a valid spatially indexed viewer
table beside the 115-instance Cellpose label/table output. Retain Cellpose masks
for shape-sensitive review and future per-cell channel quantification.

## Current decision

Cellpose is the current first teacher because its pretrained baseline passed the
crop parity check and found the expected crowded objects. StarDist3D is
architecturally native 3D and
attractive for compact fluorescent nuclei, but it needs a suitable model and
dataset-specific quality validation before whole-volume use.

## Optimization decision

**Closed on 2026-09-30:** Cellpose3D and StarDist3D speed is acceptable. Do not
return to optimizing either inference path as part of the current 3D work. The
matched GPU benchmarks found Cellpose3D within 7% of original Python and
StarDist3D between 1.24x faster and 1.08x slower depending on crop density.
Those differences are below the threshold for more optimization work.

Continue with output review, dataset-specific model quality, YOLO3D
distillation, and downstream measurements. Reopen inference optimization only
if a future matched release benchmark demonstrates a material regression or a
new production dataset fails an explicit throughput requirement.

The related method and detector research is in
[`USEGMENT3D_RESEARCH.md`](USEGMENT3D_RESEARCH.md) and
[`YOLO3D_RESEARCH.md`](YOLO3D_RESEARCH.md).
