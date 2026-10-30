# Object Measurement Speedup Plan

## Purpose

Improve the object-measurement benchmark and implementation after the Feret and
basic-geometry findings.

The target is not to win by changing benchmark setup or by comparing different
semantics. The target is to make the normal measurement path correct by design:
stream the data needed for each requested measurement, avoid materializing
object points unless a requested measurement really needs them, use integer
labels as the fast path, and benchmark processing time separately from setup.

## Current Interpretation

The remaining performance gap should not be explained as "Python uses C loops"
or by benchmark startup/setup effects. The fair comparison is the resident
processing path after fixture loading and label construction are separated.

For basic geometry, Blockflow should be faster or at least close because it can
scan integer labels once and maintain per-object tallies. If it is not, that is
actionable.

For Feret-like measurements, the comparison must be split by semantics:

- cheap object geometry: count, bbox, physical extent, optional moments,
- directional Feret estimate: support widths over named direction sets,
- exact voxel-pair Feret: explicit quadratic legacy path,
- future hull/candidate Feret: explicit approximate or exact method.

Bounding-box diagonal is not a Feret substitute. It may be a useful descriptor,
but it must be named as a bbox-derived measurement.

## Plan

### 1. Keep Benchmark Timing Honest

Keep `object-3d-measurement` reporting separate timings for:

- fixture loading,
- label construction,
- measurement,
- CSV writing,
- summary writing.

Use measurement-only time when discussing algorithmic performance. Keep
end-to-end wall time and RSS in `BENCHMARKS.md`, but do not use them as the
main explanation for slower processing.

Acceptance:

- `BENCHMARKS.md` reports both compute-only and wall/RSS rows.
- Reproduction commands still verify exact CSV agreement for shared columns.

### 2. Remove Avoidable Basic-Geometry Work

Basic object geometry should be a streaming tally:

```text
label -> count, bbox_min, bbox_max
```

It should not construct `ObjectPointSet`, `Vec<VoxelPoint>`, or any equivalent
per-voxel coordinate list.

Acceptance:

- Resident `object_geometry_basic_measurements_u32` uses integer labels and no
  point materialization.
- Planned `Measurements::object_geometry_basic` uses a basic-geometry partial
  tally, not `ObjectPointTallyOp`.
- Tests cover non-contiguous labels, background `0`, anisotropic spacing, and
  decomposition invariance.

### 3. Make Integer Labels The Fast Path

The benchmark and planned/resident measurement paths should treat integer
segmentation labels as the normal representation. `f64` labels are a
compatibility path for older APIs or image pipelines, not the representation to
optimize around.

Start with `u32` because it is the practical default for labelled segmentation
images. Add more label types only when needed.

Acceptance:

- The object-3D benchmark constructs `Array3<u32>`.
- The public API has a clear `u32` resident entry point.
- Float-label validation remains tested, but does not appear on the benchmark
  fast path.

### 4. Stream Directional Feret

Directional Feret should be computed as mergeable support-width tallies:

```text
label -> count, bbox_min, bbox_max, min_dot[k], max_dot[k]
```

This supports block-local partials and cheap merging. It also gives the
simulator a realistic cost model: `O(voxels * directions)` with bounded memory
per object and direction, not object-point materialization.

Acceptance:

- Planned directional Feret no longer uses `ObjectPointTallyOp`.
- Resident and planned outputs match for axes, icosahedral, and Fibonacci
  direction sets.
- Tests include anisotropic spacing and singleton objects.

### 5. Fuse Compatible Object Tallies

When a request asks for both basic geometry and directional Feret, one scan
should feed both outputs. The public row tables should remain separate; fusion
is an execution optimization.

Do not fuse paths that require different candidate data. Exact voxel-pair
Feret, hull Feret, boundary collection, and custom point-based measurements can
remain separate.

Acceptance:

- A combined basic-geometry plus directional-Feret request does not scan the
  label image twice unless the planner has a documented reason.
- Tests verify both result handles from the fused plan.

### 6. Profile Before Larger API Changes

After the streaming and integer-label paths are in place, profile the remaining
slow cases. Do not infer the cause from framework language or benchmark setup.

Questions to answer with profiling:

- Is time in label scanning, map updates, allocation, row encoding, CSV output,
  or plan overhead?
- Are hash/B-tree lookups dominating because labels are dense and could use a
  dense object table?
- Is bbox/count tallying memory-bound, and does block layout hurt locality?
- Are we computing points or physical coordinates earlier than needed?

Acceptance:

- There is a profile-backed note before adding a larger abstraction or
  specialized dense-label backend.

### 7. Decide On Dense-Label Optimization

If profiling shows map updates dominate for dense labels, add a dense-label
tally backend:

```text
Vec<Option<Tally>> indexed by label
```

Use it only when the max label is reasonable relative to the number of observed
objects. Keep sparse maps for large or sparse label spaces.

Acceptance:

- Dense and sparse backends produce identical rows.
- The backend choice is explicit and testable.
- Large sparse labels do not allocate huge vectors.

## Stop Conditions

Stop this speedup push when all are true:

- basic resident and planned geometry do not materialize object points,
- directional Feret planned execution does not materialize object points,
- integer labels are the benchmark fast path,
- basic geometry plus directional Feret can share a scan,
- benchmark tables report compute-only time separately from wall/RSS,
- the remaining performance gap has a profile-backed explanation.

If the profiled implementation is still slower for a fair apples-to-apples
case, decide whether that case is important enough to specialize. Do not keep
changing abstractions without a measured bottleneck.
