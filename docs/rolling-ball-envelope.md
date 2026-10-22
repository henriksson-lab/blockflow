# Rolling-ball tissue/object envelopes

`blockflow::ops::envelope` supplies a physical-scale envelope of a bool mask.
This is useful for defining a broad object boundary while bridging narrow
clefts. Keep the original mask separately: filled envelope regions are not
measured material or inferred mechanical connections.

```rust
use blockflow::ops::envelope::{
    RollingBallEnvelope, rolling_ball_envelope,
    append_rolling_ball_envelope_phases,
};
use blockflow::ops::components::Connectivity;
use blockflow::sidecar::Lifecycle;

// Radius and pitch use the same length unit, here micrometres; axes are z,y,x.
let config = RollingBallEnvelope::new(320.0, [80.0, 78.8, 78.9])?
    .connecting(Connectivity::Faces);
// Resident: use on a deliberately bounded/coarse image.
let envelope = rolling_ball_envelope(mask.view(), config)?;
// Planned: builder currently reads Bool, all images on the same lattice.
let result = append_rolling_ball_envelope_phases(
    &mut builder, config, 64, "tissue.envelope", Lifecycle::DeleteOnExit,
)?;
```

## Definition and finite-domain behavior

1. Compute the exact physical distance to the input foreground voxel centers.
2. Mark centers whose distance is strictly greater than the radius as available
   for a ball. Keep those connected to the array exterior by the selected
   connectivity (the ball-center connectivity, not tissue connectivity).
3. Remove voxels within the radius of an exterior-connected center. The
   complement is the envelope. Ball/tissue tangency is excluded; center-to-query
   distance equal to the radius is removed.

The planned construction is a composition, not independent per-block floods:
complement, exact distance sweeps, blocked-center threshold, global hole fill,
second exact distance sweeps, final threshold (13 phases). Distance transforms
use Infinite for absent seeds, avoiding a phantom outside seed. There is no
whole-volume pixel reduction; distance sweeps retain whole lanes, and filling
uses the existing component-fragment seam merge. The caller still budgets
whole-axis windows and the global component state; this does not promise
constant total memory independent of volume or component count. Sweep and
fill phases use the requested transverse block edge. The helper leaves the
builder on that cubic block grid.

Radius zero equals hole filling. Empty and full nonsingleton arrays remain
empty and full. All foreground input voxel centers remain inside the result.
Interior cavities fill even at zero radius. Objects may remain separate;
connectivity, convexity, a smooth manifold and continuous-ball accessibility
are not guaranteed. This is a sampled-geometry construction. Singleton axes
are not exterior exits, allowing 2-D planes; an all-singleton/empty shape is
rejected.

Ball centers exist only in the finite array. Pad by more than the radius with
known background for a free-space enclosure. Tissue near/touching a crop edge
or insufficient padding can leave no exterior centers near a face, causing
an envelope to reach that face. Do not silently reinterpret cropped data as
known exterior. A large radius is not a request for the convex hull.

## Existing alternatives and scope

Blockflow already supplies `ops::level_set` balloon/geodesic and resident
Chan-Vese evolution, `ops::morphology`, `ops::fill`, `ops::distance` and
`ops::star_convex`. This addition makes scale-dependent exterior accessibility
available as a reusable pipeline using those existing global primitives.

It does not implement CGAL alpha wrapping, alpha complexes or explicit
convexity/k-convexity priors. Those have different guarantees and costs.
[Alpha wrapping](https://www.cgal.org/2022/05/18/alpha_wrap/index.html) is a
possible later geometry backend if a watertight enclosing surface is needed.
For tissue stitching an envelope is only a face-proposal aid; deciding which
boundary patches are actual matching knife cuts remains an application task.

## Validation

`cargo test --offline --release -p blockflow --no-default-features --test rolling_ball_envelope`
checks an independent exhaustive-distance/BFS/ball-union oracle, anisotropic
pitch, narrow/wide cleft access, containment, empty/full/cavity behavior,
parameter rejection and planned agreement across block seams. Release-only
real-data smoke evidence is maintained by the consuming stitching project's
`research/ahlgren/chunkiness/REPORT.md`; it is not anatomical ground truth.
