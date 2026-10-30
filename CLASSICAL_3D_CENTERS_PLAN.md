# Classical 3D Cell Center Detection Plan

## Purpose

Add a normal Blockflow OME-Zarr example that detects bright cell or nucleus
centers with classical image processing. The first target is the DAPI channel in
`/husky/otherdataset/teresa/single/clustered-pbmcs.ome.zarr`.

This provides an independent result while the Cellpose3D output is reviewed.
Cellpose and YOLO3D may be displayed beside it, but neither is ground truth for
this method.

The output is a native, spatially indexed NGFF point annotation under
`tables/<layer>`, following the working YOLO3D viewer contract. It is not an
instance-label image: center detection does not establish a defensible cell
boundary, and creating one merely to resemble Cellpose would add an unmeasured
segmentation assumption. The table should be useful immediately in newvolim
and leave enough measurements in each row to support later tuning and per-cell
refinement.

Cellpose3D and StarDist3D publish linked `labels/<layer>` and `tables/<layer>`
artifacts because they produce instance masks. This example shares their native
object-table format and YOLO3D's point-annotation form. If watershed refinement
is requested, the example additionally publishes the label pyramid and linked
instance table under the same layer name.

## Proposed method

### Minimal baseline

1. Open the selected channel and time point with `OmeVolumePyramid` and use
   level 0.
2. Read the physical voxel spacing from the OME metadata.
3. Smooth the volume with an anisotropic 3D Gaussian. Specify scale in physical
   units and convert it to `[z, y, x]` voxel sigmas.
4. Find 3D regional maxima with 26-connectivity.
5. Reject maxima outside a global foreground mask made by Otsu thresholding the
   smoothed DAPI intensity.
6. Reject weak maxima by a response or prominence threshold.
7. Apply score-ordered minimum-distance suppression in physical units so one
   broad nucleus does not produce several centers.
8. Publish the surviving points as a spatially indexed NGFF object table and a
   `table.csv` compatibility view.
9. With `--watershed`, give every accepted center a stable seed label and run a
   foreground-constrained 3D watershed over the negative smoothed intensity or
   negative blob response. Publish its label pyramid and instance table using
   the same contract as Cellpose3D and StarDist3D.

### Preferred first variant

Use a Difference of Gaussians response in place of the single blurred image.
It suppresses slowly varying background and already exists as
`BlobResponseOp`. Start with two scales tied to an expected nucleus radius,
rather than arbitrary voxel scales. Keep the Gaussian-maxima path as a CLI
option because it is the most direct test of the proposed classical method.

Laplacian of Gaussian and determinant of Hessian are already implemented and
can be compared later without changing the pipeline shape.

### Threshold policies to compare

Otsu is plausible, but the population given to Otsu matters:

- **Smoothed intensity Otsu:** creates a foreground mask and is the safest
  initial use. It rejects peaks in dark background.
- **Positive response-volume Otsu:** separates weak and strong DoG or LoG
  response voxels. Background voxels may dominate it and drive the threshold
  too low.
- **Candidate-score Otsu:** runs Otsu only over regional-maxima response values.
  This directly implements “keep strong maxima,” but may be unstable if true
  and false peaks do not form two score populations.
- **Explicit response threshold or percentile:** provides a predictable fallback
  and is needed for reproducible parameter sweeps.

The example should report the selected thresholds and candidate counts before
and after every filter. It must not silently fall back when a histogram is
degenerate.

## Existing Blockflow pieces

| Need | Existing code | Assessment |
|---|---|---|
| Axis aware OME-Zarr channel selection | `OmeVolumePyramid` | Reuse |
| Physical voxel spacing | `OmeVolumePyramid::voxel_size` | Reuse |
| Anisotropic Gaussian smoothing | `Gaussian` and `SmoothOp` | Reuse |
| DoG, LoG and Hessian blob response | `BlobResponseOp` | Reuse |
| Chunk safe 3D regional maxima | `ops::regional` | Reuse |
| One deterministic row per maximum plateau | `ResponsePeakRowsOp` and `append_response_peak_table_phases` | Reuse |
| Sampling another image at point rows | `GatherRowsOp` | Reuse for raw intensity, smoothed intensity and foreground mask |
| Static row filtering | `FilterRowsOp` | Reuse where the cutoff is supplied by CLI |
| Global Otsu and related rules | `ops::threshold` | Reuse the scalar algorithms; improve the distributed collection path |
| Typed spatially indexed output | `finalize_object_table` and `ngff-object-table` | Reuse with a classical-center projection |
| Seed image construction | `VoxelizeOp` / label point operations | Reuse or generalize for stable center IDs |
| Foreground-constrained watershed | `ScalableWatershedOp` | Reuse exact bounded components and blockwise minimax fallback for oversized components |
| Label pyramid and registry | `build_nearest_label_pyramid_with_blocks`, label metadata, `refresh_label_registry` | Reuse |
| Instance measurements | `finalize_instances_3d` | Reuse |

The regional-maxima implementation already joins plateaus across block seams.
The representative coordinate is deterministic, so another block-local maxima
implementation is unnecessary.

## Operations to add or generalize

### 1. Memory-bounded global histogram reduction

The current planner-visible global threshold path collects every finite voxel
as an `f64` sample. That is too large for a full microscopy volume. Add a
distributed histogram path:

1. reduce finite count, minimum and maximum;
2. build fixed-bin block histograms over the common range;
3. merge counts and run the existing Otsu, Yen, triangle or minimum rule;
4. expose the chosen scalar to the following phase.

For bounded integer inputs, allow a one-pass exact histogram over the dtype
range. For floating response images, use the two-pass observed range. Record
the range, bin count and resulting threshold in run metadata.

### 2. Global threshold over a row column

Add a row reduction/filter operation that selects a global threshold from one
numeric column, then retains rows according to that threshold. This supports
candidate-score Otsu without materializing a mask volume. A possible API is:

```text
append_global_column_threshold_filter_phases(
    plan, rows, column, GlobalThreshold::Otsu { bins }, ThresholdTest::AtOrAbove
)
```

It must preserve row ownership and schema, use bounded histograms, and publish
the selected scalar in run metadata.

### 3. Physical-radius point suppression

Promote the score-ordered seed suppression currently embedded in the
CellProfiler planning example into a reusable row operation. It should:

- sort candidates by descending score with coordinate order as the tie-breaker;
- compare distances in physical `[z, y, x]` units;
- work across block boundaries;
- retain the original row and columns;
- allow zero radius to disable suppression.

A spatial bin grid with cells at least the suppression radius avoids an
all-pairs comparison. The reduction can first be global for correctness, then
be distributed if candidate volume makes that necessary.

### 4. Classical point-table projection

Use the existing generic `finalize_object_table` path rather than adding a new
storage implementation. Add a thin classical-center projection that maps the
row schema to stable IDs, `z/y/x`, response and intensity columns, uses the
standard 3D spatial tiles and occupancy indexes, and optionally emits
`table.csv` for the current viewer.

### 5. Optional prominence measurement

Absolute peak height can retain bright background structure and reject dim
cells. Add a local prominence or center-to-shell contrast operation if the
first comparison shows that response magnitude is insufficient. The shell
radii must be expressed in physical units and its halo declared to the planner.
This is a second iteration item, not a prerequisite for the first full run.

### 6. Optional multiscale peak merge

Cells may vary enough in diameter that one scale is inadequate. A later
multiscale operation should attach the winning scale to each candidate and
suppress duplicates in joint position/scale space. First determine whether one
physical scale works; do not pay for a scale pyramid before that measurement.

### 7. Optional watershed publication

Make watershed part of the initial example as an opt-in mode, not a future
placeholder. It should:

- convert the final center rows to stable, nonzero `u32` seed labels;
- use the Otsu foreground mask as the watershed domain;
- use the known foreground threshold during queue initialization, so
  below-threshold voxels never enter the watershed queue;
- label foreground connected components first, discard components containing no
  accepted seed, and watershed each remaining component within its bounding
  box. A masked watershed cannot cross between foreground components, so this
  is equivalent to the global flood and enables parallel work and early
  stopping on background-heavy images;
- default to negative smoothed DAPI as the flooding cost, with negative blob
  response available for comparison;
- use an explicit watershed-line policy;
- remove foreground components with no accepted seed rather than assigning
  them to an unrelated distant seed;
- write level 0 labels, build the nearest-neighbor label pyramid, write OME
  image-label metadata, refresh `labels/zarr.json`, and finalize the linked
  `tables/<layer>` instance table;
- preserve the center score and seed coordinate either in the linked table or
  in a companion center table so the pseudo-mask provenance is recoverable.

`SeededWatershedOp` correctly declares whole-volume reach. Independent block
watersheds are not equivalent. The optimized path must partition by globally
joined foreground connected components, which are mathematically independent,
rather than by arbitrary storage blocks. Retain a whole-volume reference mode
for crop-level equivalence checks.

## Example design

Create `examples/classical-3d-centers-ome-zarr` with a release-only executable.
It must use the OME reader, `PlanBuilder`, planner-selected blocks,
`execute_phases`, and the shared table writer. It should not invoke low-level
array functions over a manually loaded full volume.

Suggested CLI:

```text
cargo run --release -p classical-3d-centers-ome-zarr -- \
  --zarr /husky/otherdataset/teresa/single/clustered-pbmcs.ome.zarr \
  --channel 0 --time 0 \
  --method dog \
  --radius-um <estimated-radius> \
  --foreground-threshold otsu \
  --response-threshold candidate-otsu \
  --min-distance-um <estimated-diameter-fraction> \
  --watershed \
  --watershed-cost smoothed \
  --layer classical-3d-centers-v1
```

Also expose block shape, workers, histogram bins, Gaussian truncation, explicit
threshold overrides, and `--overwrite`. Print total elapsed time and the count
at each filtering stage.

Suggested output columns:

| Column | Meaning |
|---|---|
| `detection_id` | Stable row identity |
| `z`, `y`, `x` | Level-0 voxel center used by the viewer |
| `physical_z`, `physical_y`, `physical_x` | Physical center coordinates |
| `response` | Gaussian, DoG, LoG or Hessian response at the peak |
| `raw_intensity` | Original DAPI value at the peak |
| `smoothed_intensity` | Gaussian intensity at the peak |
| `scale_z`, `scale_y`, `scale_x` | Winning voxel scale |
| `method` | Numeric method identifier described in table metadata |

Do not invent cell extents for the default table. The annotation represents
centers. In watershed mode, measured mask extents and volumes belong in the
linked instance table.

## Implementation order

1. Add the bounded distributed histogram and verify it against the existing
   scalar threshold functions on small arrays.
2. Add global row-column thresholding.
3. Extract physical minimum-distance row suppression from the CellProfiler
   example into the library.
4. Add the classical-center projection over the existing generic object-table
   finalizer.
5. Build the OME-Zarr example using existing smoothing/blob and maxima phases.
6. Add the optional seed, watershed, label-pyramid and instance-table path.
7. Run small sparse and crowded crops to select a useful scale and compare the
   threshold policies.
8. Check watershed masks on the same crops and keep its parameters distinct
   from center-detection parameters.
9. Run the selected center configuration over the full volume, recording wall time,
   peak host memory, thresholds and stage counts.
10. Run the optional watershed over the full volume only if its whole-volume
    working set fits, then open both forms in newvolim beside Cellpose3D and
    YOLO3D.

Only focused tests are needed: histogram merge equivalence, seam-spanning peak
ownership, physical suppression across a block boundary, and table index
validity. Use release builds for all Rust builds, runs and tests.

## Parameter selection and validation

Choose several small review regions before the full run:

- sparse blobs away from the middle cluster;
- the dense middle cluster where neighboring cells are closest;
- dim blobs near the apparent detection limit;
- empty or background-heavy regions for false-positive assessment;
- volume borders and block seams.

Start with a small grid over two or three physical radii and the three useful
threshold policies. Keep the foreground rule fixed initially. For each result,
record candidates before thresholding, after foreground gating, after response
thresholding, and after distance suppression.

Manual point review is the acceptance reference. Measure precision and recall
within a physical matching radius on the reviewed regions, plus duplicate rate
and nearest-neighbor distance. Separately report agreement with Cellpose3D and
YOLO3D to reveal systematic differences. Do not optimize parameters against
the full volume or count agreement alone.

Manual whole-volume sanity count supplied on 2026-09-30 for
`single/clustered-pbmcs.ome.zarr`:

- sparse outer parts: about 37 cells, possibly 40;
- dense center cluster: about 40--50 cells;
- implied whole-volume range: roughly 77--90 cells.

These are approximate visual counts for spotting large under- or over-detection,
not object-level ground truth. Crop counts are not directly comparable with
these whole-volume totals.

The first full-volume run is justified when one setting behaves acceptably in
both the sparse regions and the dense middle cluster. Preserve all parameters
and selected thresholds in table metadata so the result can be reproduced.

## Validation status (2026-09-30)

The release example is implemented and the DoG configuration above has been
published as `0/tables/classical3d-dog-v1`. It detected 80 centres over the full
volume in 266.93 seconds of application time (4:26.99 wall time, 591 MiB peak
RSS). Of these, 40 fall inside the reviewed crowded crop box and 40 outside,
matching the approximate manual ranges above.

The optional watershed path was validated end to end on the crowded
`60 x 512 x 512` crop. It produced 41 centre rows and 34 foreground-constrained
instances in 13.57 seconds with 735 MiB peak RSS. The difference is retained in
the companion centre table for review: seven accepted centres did not survive
the independently selected watershed foreground mask.

The bounded three-phase Otsu path and masked queue exclusion are implemented.
The watershed now labels the foreground components, runs bounded component
boxes independently in parallel, and routes boxes above
`--watershed-exact-max-bbox-voxels` through a deterministic two-pass minimax
fallback. Its first pass settles the least barrier; its second settles shortest
optimal-barrier paths. Both retain one queue per block and wake only blocks that
receive an improved boundary candidate. The planner-facing component discovery
still declares whole-volume reach and holds dense state arrays. The fallback's
barrier and ownership state uses 24 bytes per voxel on 64-bit targets, down from
about 56 bytes with optional padded keys. This removes the single unbounded
global queue but is not yet an out-of-core watershed. A
full-volume watershed run is therefore still outside this validation record.

## Later extension toward cell measurements

The implemented watershed option supplies an initial pseudo-mask for later YOLO
training and per-channel intensity quantification. More selective local
watershed, random walker, or learned refinement can replace it later while
keeping the center detector and output contract stable. The center-only mode
remains independently usable and evaluable.

## Watershed-to-YOLO3D distillation

Train a second, separately identified YOLO3D model using the classical seeded
watershed output as its teacher. This model must remain distinct from the
existing YOLO3D model distilled from Cellpose so that teacher provenance is
never ambiguous.

1. Review and freeze the watershed teacher parameters first. Keep the center
   companion table so seeds rejected by the foreground mask remain auditable.
2. Convert complete watershed instances into the same 3D YOLO target contract
   used by the Cellpose-distilled model. Exclude crop-truncated instances from
   training targets.
3. Preserve the established spatial train, validation, and test separation;
   do not allow overlapping windows from one spatial neighborhood to cross
   partitions.
4. Train and select the watershed-derived model with its own checkpoint,
   history, plot, and immutable teacher/configuration metadata.
5. Apply the selected model to the entire level-0 PBMC volume, merge window
   duplicates, and publish a spatially indexed annotation table under a new
   layer name such as `yolo3d-watershed-centers-v1`.
6. Keep the existing Cellpose-derived full-volume layer, currently
   `yolo3d-cell-centers-v1`, alongside it. Both layers must load independently
   in newvolim for direct visual comparison.

The comparison should report counts and spatial agreement, but visual review
remains the acceptance reference because both models inherit systematic errors
from different pseudo-label teachers.
