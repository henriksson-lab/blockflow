# Object Geometry Performance Plan

## Problem

The `object-3d-measurement` benchmark now asks for basic object geometry, but
the implementation still goes through `collect_object_point_sets`. That means
the basic path stores every foreground voxel coordinate per object even though
the requested output only needs:

- label,
- voxel count,
- bounding-box minimum,
- bounding-box maximum,
- physical bounding-box extent.

This is not an apples-to-apples implementation for a basic-geometry benchmark.
It is the old materialized-point object abstraction with the exact Feret column
removed.

There is also label dtype debt: many resident measurement APIs accept
`ArrayView3<'_, f64>` labels and validate/convert each foreground value through
the float label path. Segmentation labels should normally be integer labels.
Float labels should be compatibility input, not the default fast path.

## Goal

Make the basic object-geometry path fast by design:

- no point materialization for basic geometry,
- integer label arrays as the primary benchmark path,
- compute-only timing separated from fixture setup and CSV writing,
- public measurement APIs that let the implementation choose the cheapest
  object representation needed by the requested measurements.

## Phase 1: Measure The Current Cost Split

Status: implemented for `examples/object-3d-measurement`. The Blockflow
summary JSON now reports fixture load, label construction, measurement, CSV
write, and summary write seconds.

Add timing fields to `examples/object-3d-measurement` summary output:

- `fixture_load_seconds`,
- `label_construction_seconds`,
- `measurement_seconds`,
- `csv_write_seconds`,
- `summary_write_seconds`.

Keep `/usr/bin/time -v` wall/RSS for end-to-end reporting, but use
`measurement_seconds` for compute-path comparisons.

Acceptance:

- `run_benchmark.sh` still compares Blockflow and scikit-image CSVs exactly.
- `BENCHMARKS.md` reports both end-to-end wall time and measurement-only time,
  or explicitly states which one is being discussed.

## Phase 2: Streaming Basic Geometry Resident API

Status: implemented for resident `f64` compatibility labels and resident `u32`
labels. Both paths use a streaming count/bbox tally and do not materialize
object point vectors.

Implement a resident streaming tally for basic geometry:

```text
label -> {
  count,
  bbox_min,
  bbox_max
}
```

It should scan labels once and never allocate per-voxel point vectors.

Recommended API shape:

```rust
pub fn object_geometry_basic_measurements_u32(
    labels: ArrayView3<'_, u32>,
    spacing: PhysicalSpacing,
) -> Result<Vec<ObjectGeometryBasicMeasurements>>;
```

Also consider a generic internal helper over a small sealed trait:

```rust
trait LabelVoxel {
    fn to_label(self, at: [usize; 3]) -> Result<u64>;
}
```

Then expose specific public entry points for the label types we actually want to
support, starting with `u32` and the existing `f64` compatibility path.

Acceptance:

- Existing `object_geometry_basic_measurements(f64)` remains compatible.
- New `u32` path has equivalent rows on the benchmark fixtures.
- Tests cover background label `0`, multiple labels, non-contiguous labels,
  anisotropic spacing, and invalid float labels for the compatibility path.

## Phase 3: Update Object-3D Benchmark To Use Integer Labels

Status: implemented. The benchmark now builds `Array3<u32>` labels and calls
`object_geometry_basic_measurements_u32`. `BENCHMARKS.md` includes updated
end-to-end and measurement-only timings.

Change `examples/object-3d-measurement` to build `Array3<u32>` labels and call
the new `object_geometry_basic_measurements_u32` path.

Keep the scikit-image reference unchanged unless it needs dtype clarification;
it already uses integer labels.

Acceptance:

- 10-image and 50-image benchmark CSVs still match the scikit-image reference.
- Summary output includes timing split from Phase 1.
- `BENCHMARKS.md` is updated with the new numbers.

## Phase 4: Planned Basic Geometry Without ObjectPointTallyOp

Status: implemented. Planned basic geometry uses
`ObjectGeometryBasicTallyOp`, `ObjectGeometryBasicPartialLayout`, and
`MergeObjectGeometryBasicOp`. It no longer routes through
`ObjectPointTallyOp`.

Add a planned streaming tally for basic geometry instead of routing
`Measurements::object_geometry_basic` through `ObjectPointTallyOp`.

Suggested pieces:

- `ObjectGeometryBasicTally` with count and bbox.
- `ObjectGeometryBasicTallyOp` scanning label blocks.
- A compact partial layout for count and bbox rows.
- `MergeObjectGeometryBasicTallyOp` producing existing
  `ObjectGeometryBasicMeasurements` rows.

This should not affect exact geometry, custom object measures, convex hull, or
other paths that really need points.

Acceptance:

- Planned basic geometry matches resident basic geometry.
- Decomposition-invariance tests pass.
- Simulator cost no longer treats basic geometry as a point-materializing
  object measure.

## Phase 5: Streaming Directional Feret

Status: implemented. Resident directional Feret and planned directional Feret
now use streaming support-width tallies. Planned directional Feret uses
`ObjectDirectionalFeretTallyOp`, `ObjectDirectionalFeretPartialLayout`, and
`MergeObjectDirectionalFeretOp`; it no longer materializes object point vectors.

Replace the current planned directional Feret implementation with a streaming
tally:

```text
label -> {
  count,
  bbox_min,
  bbox_max,
  per direction: min_dot, max_dot
}
```

The output remains `ObjectDirectionalFeretMeasurements`.

Acceptance:

- Resident and planned directional Feret match for axes, icosahedral, and
  Fibonacci direction sets.
- Tests include anisotropic spacing and singleton objects.
- No point vectors are allocated for directional Feret.

## Phase 6: Fuse Compatible Object Tallies

Status: implemented for `object_geometry_basic` plus
`object_directional_feret`. In that combined request, directional Feret owns
the label scan and basic geometry adds a merge-only row phase derived from the
same directional Feret partials. Public row handles remain separate.

If a plan requests both basic geometry and directional Feret, one label scan
should be able to produce both partials or one combined partial that can emit
both row tables.

Do not fuse incompatible paths prematurely. Exact Feret, convex hull, boundary
candidate collection, and custom point-based measures can still use materialized
points.

Acceptance:

- Requesting `object_geometry_basic` plus `object_directional_feret` does not
  double-scan labels unless the planner has a clear reason.
- Public row tables stay separate.
- Tests verify both row handles collect correctly from the fused plan.

## Phase 7: Decide Whether To Generalize Labels Further

Status: decided for this push. Keep `u32` as the resident fast path and keep
the existing `f64` resident API as a compatibility path. Do not add `u16`,
`u64`, or `usize` public resident wrappers until a benchmark, fixture, or user
workflow needs them. Planned measurement still accepts the framework label
dtypes through `TypedSource` and validates labels through the existing
`label_value` path.

After the `u32` fast path is measured, decide whether to add `u16`, `u64`, and
possibly `usize` resident paths.

Guidance:

- `u32` is likely the right default for segmentation labels.
- `u64` is useful for row compatibility and very large label spaces.
- Float labels should remain supported only where needed for compatibility with
  existing APIs or pipelines.

Avoid forcing all resident measurement APIs through generics unless it reduces
duplication without complicating call sites. A small sealed trait plus thin
public wrappers is likely enough.

## Stop Conditions

Stop this performance push when all of these are true:

- basic geometry resident path no longer materializes object points,
- object-3D benchmark uses integer labels,
- benchmark reports compute-only and end-to-end timing,
- planned basic geometry no longer uses `ObjectPointTallyOp`,
- directional Feret no longer materializes points in the planned default path,
- the 50-image basic-geometry compute time is no longer dominated by avoidable
  point-vector allocation or float-label conversion.

If the benchmark remains slower after those changes, use profiling before
adding more abstractions.
