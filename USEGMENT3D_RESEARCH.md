# u-Segment3D research notes

Date: 2026-09-29. Sources:

- [Nature Methods paper](https://www.nature.com/articles/s41592-025-02887-w)
- [DanuserLab repository](https://github.com/DanuserLab/u-segment3D)

## What it contributes

u-Segment3D converts stacks of 2D instance segmentations into one 3D instance
segmentation. It accepts one, two, or all three orthogonal views: XY, XZ, and
YZ. The views may come from different 2D segmenters.

It offers two routes:

- **Direct:** consume probability and gradient fields from a method such as
  Cellpose.
- **Indirect:** consume ordinary 2D instance masks and derive distance-transform
  gradients from them.

The core algorithm is:

1. segment slices in the available orthoviews;
2. represent each 2D instance by a distance-transform gradient toward a medial
   axis attractor;
3. fuse foreground probabilities and gradient components into a consensus 3D
   field, downweighting locally inconsistent views;
4. advect foreground voxels along that field with suppressed gradient descent;
5. cluster the final attractors spatially and assign the original voxels to
   those object IDs;
6. optionally refine boundaries with guide-image-aware label diffusion and
   recover protrusions.

The authors evaluated more than 70,000 cells across 11 datasets and report that
the consensus method can match native 3D approaches and can do better for some
crowded or complex morphologies. The useful principle for this project is that
strong 2D models can bootstrap 3D labels without a fully annotated 3D training
set.

## What maps well to Blockflow

### Orthoview inference is naturally a set of planned phases

XY, XZ, and YZ inference can be represented as three views over the same
stored volume. Each phase can use chunk-aware traversal and write probability,
gradient, or mask arrays. This avoids holding the volume or all slice outputs
in memory.

For anisotropic data, all three views should not be enabled automatically. The
paper explicitly notes that XZ and YZ can be worse when axial sampling or image
quality differs from XY. A dataset configuration should select views and their
weights after reading physical voxel scales.

### Consensus is a volumetric block operation

Content-based fusion uses local neighborhoods, so it has bounded reach and can
be tiled with a halo. Gradient smoothing has the same property. These are good
Blockflow operations.

### Gradient descent needs overlapping subvolumes and global coordinates

For a 194 x 5,440 x 4,792 tissue volume, the paper used 128 x 256 x 256
subvolumes with 25% overlap and 250 iterations. The complete published pipeline
took about seven hours: roughly two hours for preprocessing and three-view
Cellpose, two hours for initial consensus segmentation, one hour for filtering,
and two hours for label diffusion. It used 32 physical CPU cores, 72 threads,
1.5 TB RAM, and one 40 GB A100 for 2D inference. Running gradient descent as one
full volume was reported to be more than 20 times slower.

This strongly supports Blockflow's planned overlapping blocks. The important
difference is ownership: u-Segment3D carries globally indexed foreground
coordinates so overlap shares the same attractor and avoids a later stitching
pass. A Blockflow implementation should preserve global coordinates during
advection rather than independently labeling each block and trying to reconcile
IDs afterward.

### Output matches the existing annotation direction

The result is a dense 3D instance label volume. It can use the same stable-ID
label layer and `ngff-object-table` pairing planned for native Cellpose3D and
StarDist3D. Measurements over other fluorescence channels can then be computed
from the final masks without rerunning segmentation.

## Ideas to borrow

These ideas are useful independently of a literal port:

- keep the number of orthoviews configurable and quality-aware;
- use physical voxel spacing in resampling, gradients, smoothing, distance,
  minimum volume, and clustering;
- support direct model fields and indirect instance masks through one consensus
  interface;
- weight view consensus by local consistency rather than taking an unweighted
  vote;
- use suppressed gradient descent with a decaying step size;
- retain global coordinates in overlapped blocks so border objects converge to
  the same attractor;
- filter candidate objects by volume and gradient consistency before expensive
  boundary refinement;
- make label diffusion an optional refinement stage driven by the original
  fluorescence image;
- permit XY-only translation for strongly anisotropic acquisitions.

The repository is GPL-3.0. Blockflow is MIT, so implementation should be based
on the published method and independently written. Code should not be copied
from the Python package into Blockflow without making an explicit licensing
decision.

## Relationship to Cellpose, StarDist, and YOLO

u-Segment3D is a strong **teacher-construction** option. It can turn existing
2D Cellpose, StarDist, or YOLO-derived slice masks into 3D pseudo labels. It is
also an independent production candidate when training a 3D model is not
justified.

For the planned YOLO distillation:

- boxes derived from consensus objects can train a fast 3D detector;
- retain the consensus masks even when the first student predicts only boxes;
- store teacher provenance and confidence per object;
- use the masks later for an ROI refinement model and fluorescence
  quantification;
- compare the student both to teacher labels and to a small independently
  reviewed test set, because distillation alone cannot reveal systematic
  teacher errors.

## Suggested experiment on incoming data

1. Inspect voxel spacing, channel axes, depth, object diameter in physical
   units, and axial point-spread quality.
2. Run the existing 2D detector or Cellpose on a small set of XY, XZ, and YZ
   slices. Do not assume XZ/YZ are useful merely because they exist.
3. Build three teacher candidates on a small volume:
   native Cellpose 3D, XY-only consensus, and multi-view consensus.
4. Review missed cells, merges, splits, and axial continuity manually.
5. Choose the teacher or ensemble before generating the full pseudo-label set.
6. Retain raw fields/masks and object metadata so a later detector does not
   become the only surviving representation of the teacher.

## Scope boundary

The immediate target is nuclei or compact fluorescent cells. The paper also
handles branched cells, vasculature, and protrusions, but those cases weaken the
value of axis-aligned YOLO boxes. For irregular morphologies, consensus masks
or StarDist polyhedra should remain the primary representation, with detection
used only to index or propose local refinement regions.
