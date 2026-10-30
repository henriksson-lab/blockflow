# Feret Diameter Plan

## Summary

The current `object_geometry_measurements` path computes
`max_voxel_feret_diameter` by checking every pair of labelled voxel centers.
That is exact for the voxel-center point set, but it is `O(n^2)` per object and
should not be a default geometry measurement.

Bounding-box diagonal is not a replacement for Feret. It is cheap and sometimes
useful, but it is an axis-aligned descriptor and should be reported as
`bbox_diagonal` or `bbox_extent_diagonal`, not as Feret.

The right API shape is:

- `object_geometry_basic`: cheap geometry such as count, bbox, centroid, volume,
  equivalent diameter, and moment-derived axes.
- `object_feret(method)`: explicit Feret-like measurements, with the method
  named in the request and output.

## Current Code Status

- Brute voxel-center Feret:
  `ObjectPointSet::max_voxel_feret_diameter` in `src/ops/measure.rs`.
- Legacy exact voxel-pair geometry is still available as
  `object_geometry_measurements` and `Measurements::object_geometry`, through
  `MergeObjectGeometryOp`. This route is explicit and retains the
  `max_voxel_feret_diameter` column.
- Cheap geometry is available as `object_geometry_basic_measurements` and
  `Measurements::object_geometry_basic`. It reports count, bbox, and physical
  bbox extent without computing any Feret-like diameter.
- 3-D directional support-width Feret estimates are available as
  `object_directional_feret_measurements` and
  `Measurements::object_directional_feret`, using `DirectionSet3::{axes,
  icosahedral,fibonacci}` or caller-supplied normalized directions.
- Planned cheap geometry and directional Feret currently reuse
  `ObjectPointTallyOp`, so object points are still materialized before merge.
  This avoids the `O(n^2)` pairwise distance but is not yet the final streaming
  `min_dot/max_dot` tally.
- Existing convex-hull APIs are `object_convex_hull_measurements` and
  `object_voxel_face_convex_hull_measurements`, but their current 3-D hull
  builder enumerates triples of points and checks all points against each
  candidate plane. That can be worse than the brute Feret path and must not be
  made the default.

## Next Four Implementation Items

1. Replace the planned directional Feret backend with a streaming tally that
   keeps per-object count, bbox, and per-direction `min_dot/max_dot` values.
   This removes the object-point materialization for the default estimate and
   gives the simulator a better cost model.
2. Add a fused object-geometry tally path so `object_geometry_basic` and
   `object_directional_feret` can share one label pass when both are requested.
   The public rows should remain separate; only the internal scan should fuse.
3. Add 2-D projected Feret support based on the existing projected-convex path:
   boundary or projected object points, convex hull, then rotating calipers.
   Keep it named separately from 3-D directional Feret.
4. Add benchmark cases that report cheap geometry, directional Feret, and legacy
   voxel-pair exact Feret separately on small, elongated, and anisotropic 3-D
   objects. The goal is to catch accidental reintroduction of quadratic default
   work.

## Recommended Defaults

### 2-D

Use exact convex-hull Feret with rotating calipers.

Pipeline:

1. Extract boundary or contour points.
2. Build a convex hull.
3. Run rotating calipers over hull vertices.

Complexity:

- Boundary/contour extraction: `O(n)`.
- Hull: usually `O(b log b)`, or `O(b)` when input order makes that possible.
- Calipers after hull: `O(h)`.

This is the standard solution for 2-D farthest pair and matches the direction
taken by ImageJ/OpenCV-style geometry tools.

### 3-D

Do not default to exact Feret. Use directional support widths as the practical
default.

For each direction `u`, track:

```text
min_dot = min(dot(p, u))
max_dot = max(dot(p, u))
width(u) = max_dot - min_dot
```

The reported estimate is:

```text
max_directional_feret_diameter = max(width(u) for u in directions)
```

Complexity:

- `O(n * k)` for `n` points and `k` directions.
- Mergeable by keeping per-direction minima and maxima.
- Does not require materializing all object points if implemented as a tally.

Suggested direction sets:

- `DirectionSet3::axes()`: very cheap, coarse.
- `DirectionSet3::icosahedral()`: useful default.
- `DirectionSet3::fibonacci(count)`: tunable accuracy/cost.

This is a lower-bound estimate of maximum caliper extent. Accuracy improves with
direction density. It should be named as an estimate, not as exact Feret.

## Method Names

Suggested request API:

```rust
pub enum ObjectFeretMethod {
    Directional { directions: DirectionSet3 },
    VoxelPairExact,
    HullVertexPairExact { max_vertices: Option<usize> },
}
```

Suggested outputs:

- `max_directional_feret_diameter`
- `max_voxel_pair_diameter` or `max_voxel_feret_exact`
- `max_hull_feret_diameter`

Avoid using one column name for multiple semantics.

## Future Exact 3-D Hull Feret

This is the right long-term exact measurement, but it should be implemented as a
separate project rather than by reusing the current `convex_hull_3d` routine.

### Definition

For a finite point set, the maximum Feret diameter / point-set diameter is the
maximum Euclidean distance between any two points. Interior points cannot be the
unique farthest pair, so the answer is attained by points on the convex hull.
For voxel objects there are two useful exact variants:

- Voxel-center hull diameter: hull over voxel centers.
- Voxel-face hull diameter: hull over continuous voxel face/corner geometry.

The second is usually closer to a physical object caliper measurement, while the
first is closer to current `max_voxel_feret_diameter` semantics.

### Required Replacement

Replace the current `convex_hull_3d` implementation before promoting this path.
The current implementation is acceptable as a small reference but not as a
production default because it tests all point triples and all points.

A better hull builder should be one of:

- A Quickhull-style implementation for 3-D.
- An incremental 3-D hull with conflict lists.
- A dependency-backed hull if a small, maintained Rust geometry crate meets the
  project constraints.

Expected practical cost should be near `O(b log b)` or output-sensitive for
ordinary biological objects, where `b` is boundary/candidate points and `h` is
hull vertices.

### Candidate Reduction

Do not feed every object voxel into the 3-D hull path by default.

Candidate sets, in increasing physical fidelity:

- Boundary voxel centers.
- Exposed voxel face corners.
- Deduplicated surface mesh vertices after marching cubes, when marching cubes
  is available.

Boundary extraction is `O(n)` and usually reduces candidates substantially.

### Diameter After Hull

Initial exact implementation:

```text
max distance over hull vertices by pairwise scan: O(h^2)
```

This is simple, testable, and often much cheaper than `O(n^2)` because `h << n`.
It should still have a guard such as `max_vertices` or `max_pairs` so one
pathological object cannot make a normal measurement request explode.

Possible later optimization:

- Implement a true 3-D antipodal/rotating-calipers-style hull diameter
  algorithm.
- Use a practical exact/approximate diameter algorithm over hull vertices.

Subquadratic exact algorithms for 3-D diameter exist, but they are much more
complex than the 2-D rotating-calipers case. They should not be the first
implementation unless there is a clear crate-quality reason to trust them.

### Approximation Before Exact

Before `HullVertexPairExact`, implement a hull or boundary based approximation:

1. Build or collect candidate boundary points.
2. Run repeated farthest-point sweeps from deterministic seeds.
3. Combine with directional support widths.

This costs roughly:

```text
O(r * c + k * c)
```

where `c` is candidate count, `r` is farthest-point rounds/seeds, and `k` is
direction count.

This provides a strong lower bound and is suitable as a default when named as an
estimate.

### Acceptance Criteria

Exact hull Feret is ready when:

- The current `O(p^4)` hull builder is replaced or bypassed.
- The API separates exact and approximate methods.
- Large-object cost has an explicit guard.
- Tests compare:
  - simple boxes,
  - elongated diagonal rods,
  - hollow/shell objects,
  - anisotropic spacing,
  - small degenerate objects with 1, 2, or 3 points.
- Benchmarks report cheap geometry separately from Feret methods.

## Benchmark Implications

The current 3-D object benchmark should not call `object_geometry_measurements`
if the benchmark is meant to represent cheap geometry. It should request basic
geometry only, then optionally run a separate Feret benchmark for:

- directional estimate,
- voxel-pair exact,
- future hull-vertex exact.

This will make benchmark ratios meaningful: cheap measurements are compared to
cheap measurements, and expensive shape descriptors are compared separately.

## References Checked

- scikit-image `regionprops` documents `feret_diameter_max` as using a convex
  hull contour: https://scikit-image.org/docs/stable/api/skimage.measure.html
- ImageJ documents Feret as the longest distance between selection boundary
  points and uses convex hull endpoints:
  https://imagej.net/ij/docs/guide/146-30.html
- ImageJ source notes Feret endpoints are convex hull vertices:
  https://github.com/imagej/ImageJ/blob/master/ij/gui/Roi.java
- Princeton `FarthestPair` describes 2-D convex hull plus rotating calipers:
  https://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/FarthestPair.java.html
- OpenCV exposes convex hull and rotating-calipers-backed shape routines:
  https://docs.opencv.org/doc/doxygen/html/d9/ded/group__geometry__shape.html
