# Shell Objects 3D Plan

## Purpose

Port the non-learning cyst detector prototyped in `imaging_nedal` (crate
`cystfind`) to a planned, out-of-core blockflow pipeline. The target objects are
round, thick-walled shells (~30 µm, ~60 voxels across at level 0) in the
nedal_data micro-CT-like volumes, e.g.
`/husky/otherdataset/teresa/nedal_data/ND_WT_5_s1.ome.zarr`.

The prototype publishes `labels/cystfind` + `tables/cystfind` into the image and
on ND_WT_5_s1 accepts 46 objects, about 40 of them real, with 29 of 30
independently eye-labelled objects recalled. That result is the acceptance
reference for the port: the same objects, centres and radii within a stated
tolerance.

Rule for the port: **no algorithm is duplicated**. What blockflow already does is
called; what it nearly does is generalised in place; only what it cannot express
is new.

## Method (as prototyped)

1. **Coarse scan.** An isotropic 4× grid (a stored level plus block averaging);
   normalised cross-correlation with a radial template of the shell only (r/R in
   0.65–1.4, the interior ignored) at six radii; the maximum over radii; local
   maxima; suppression within 0.75 × the smaller radius.
2. **Template bootstrap.** The template starts from one exemplar's radial
   profile and is replaced by the mean z-scored profile of the 12 best fitted
   objects, then the scan reruns. Host-side, small.
3. **Fit at level 0, per candidate.** A local centre search for the sphere that
   collects the most wall evidence, then 500 star-convex rays: per ray the
   boundary is the dark-band-to-wall step, regularised against neighbouring
   rays, moved out to the wall peak plus a pad, and the centre re-fitted by a
   least-squares sphere. Features: `strong`, `ring_depth`, coverage, roughness.
4. **Accept.** Feature thresholds (two-tier on `strong` by size) and duplicate
   removal: an object centred inside a better-scoring one (d < 0.9 × the larger
   radius) is part of it.
5. **Labels.** Each fit rasterised from its per-ray radii; overlapping voxels go
   to the object they are relatively deepest in.

The stored z spacing of this data is wrong (the shells are spheres of equal voxel
extent on every axis), so the pipeline takes spacing as a parameter and never
derives anything from `voxel_size()` unless asked.

## What is reused

| need | blockflow |
|---|---|
| FFT correlation with an arbitrary kernel | `ops::convolve::TransformConvolveOp` (overlap-save, f64; kernel from `StructuringElement::from_offsets`; `Sense::Correlate`, `Boundary::Clamp`) |
| NCC per radius | a composition, not an op: `Chain::parallel` of three `TransformConvolveOp` branches (template, mask, mask over the squared input) joined by `FnCombine` |
| maximum over radii | `ArithmeticCombine::Maximum`; no argmax is needed because the fit searches the radius itself |
| peaks to rows, score cut | `ops::interest::append_response_peak_table_phases`, `FilterRowsOp` |
| smoothing before the fit | `SmoothOp` as its own phase; the fit op reads that image |
| feature cuts | `FilterRowsOp` / `ColumnTest` (the fit emits a precomputed margin for the two-tier rule) |
| table + labels into the image | `object_table::finalize_object_table` with its own `TableSpec`, then `ome_zarr::publish_label` |
| window around a row, owned by one block | the `FragmentOp` + `SourceInput` reach + `owner_of` pattern of `examples/bead-layer-distance` |

## What is generalised

1. **Point suppression as a reusable barrier row op** (`ops::suppress`). Planned
   item 3 of `CLASSICAL_3D_CENTERS_PLAN.md`, never built; today
   `classical_centers::suppress_within_distance` takes one fixed distance. The op
   orders rows by a priority column and drops a row within
   `factor × pair(r_kept, r_row)` of a kept one, where the radius is fixed or a
   column and `pair` is the smaller, the larger or the sum. The scan's peak
   suppression and the duplicate rule are both instances; `classical_centers`
   calls the same kernel.
2. **Box-mean downsampling in a plan.** `ResampleOp` point-samples only and the
   shape-changing phase builder is crate-private. Add a mean reduction and a
   public phase builder, so a coarse isotropic grid comes from a stored level
   without a host-side pass.
3. **Shared geometry.** A public Fibonacci sphere (one exists privately in
   `ops::label_energy`), a public trilinear sampler (private in `ops::edges`)
   and a least-squares sphere fit (new, small).
4. **Later, if measured to matter:** a multi-kernel FFT op that transforms its
   input once for all kernels. `TransformConvolveOp` re-transforms per kernel,
   18 transforms here where the prototype needs 2 spectra.

## What is new

5. **`RayShellFitOp`**, a `FragmentOp`: reads the smoothed level-0 image through
   a declared window, fits every candidate row its block owns, emits centre,
   summary radii and features. The edge criterion is a parameter, so it is a
   star-convex boundary fit rather than a cyst detector. Per-ray radii go to a
   companion stream, `DeleteOnExit` by default and persistent on request.
6. **`StarConvexLabelsOp`**, a `u64` painter in the shape of `BeadLabelsOp`:
   reads fitted rows from neighbouring blocks, paints each object from its rays,
   and resolves overlaps by relative depth. Requires object radius ≤ block edge.

## Example

`examples/shell-objects-3d-ome-zarr`:

- plan 1 on a coarse level: downsample, NCC, maximum, peaks, suppression;
- plan 2 on level 0: candidates as rows, smoothing, fit, feature filter,
  duplicate suppression, painting, `finalize_object_table`, `publish_label`;
- host-side: the template from an exemplar or a saved template file, the
  bootstrap rescan, contact sheets, and a `POST /v1/datasets/rescan` to the
  viewer after publishing (its feature cache is not invalidated otherwise).

## Validation

- Focused tests: suppression equivalence with the old fixed-distance function
  and across block seams; box-mean against a resident reference; geometry
  helpers; ray fit and painter on a synthetic shell crossing block seams.
- ND_WT_5_s1 against the prototype's published result: objects matched by
  centre, radius differences reported.
- One more image with the saved template, no new exemplar.
- Wall time, peak memory and block sizes, release builds only.

## Order

1. `ops::suppress` and its use in `classical_centers`.
2. Box-mean resampling and its phase builder.
3. Geometry helpers.
4. `RayShellFitOp`, `StarConvexLabelsOp`.
5. The example, then validation.

## Status (2026-10-18)

Done on branch `shell-objects-3d`:

- `ops::suppress` (`suppress_points`, `SuppressRowsOp`): fixed or per-row radius,
  smaller/larger/sum pairing, positions from the row's voxel or from columns;
  `classical_centers::suppress_within_distance` is its fixed case.
- `ops::resample::{append_resample_phase, append_block_mean_phases, block_mean_kernel}`:
  a block mean as a box correlation followed by a nearest resample.
- `sphere` (`fibonacci_sphere`, `fit_sphere`, `RaySet`); `ops::edges::trilinear_at` public.
- `ops::template_match`: radial templates and the NCC composition.
- `ops::star_convex`: `fit_shell`, `ShellFitOp`, `StarConvexLabelsOp`.
- `examples/shell-objects-3d-ome-zarr`, validated against the prototype; see its README.

`ops::label_energy` keeps its own Fibonacci spiral: its weights depend on that
spiral's axis order. Open: the shared-spectrum transform op (item 4), skipping
the smoothing where there are no candidates, a second image with the saved
template.

## Speed (2026-10-18, second round)

Measured first (perf, per-block timings), then changed. Every change below
left the output byte-identical (CSV, contact sheets) on ND_WT_5_s1.

| where the time went | fix | where |
|---|---|---|
| the scan ran as one block on one thread, and re-transformed the input per kernel (36 transforms per tile) | `TransformBankOp`: inputs transformed once per tile, one inverse per kernel, groups combined per voxel and folded; tiles of a block computed in parallel (`convolve::run_tiles`, shared with `TransformConvolveOp`) | `ops::transform_bank`, `ops::convolve` |
| a transform op computed every tile its buffer touched, halo included: up to 27 tiles per core tile | only tiles that can hold a valid voxel (outside the kernel's own side of a non-face buffer edge) are computed | `ops::convolve::run_tiles` |
| `PlanBuilder::pixels` built phases from `reach3()`, the symmetric worst case, so the tile-aligned halo discount (`AxisReach::Aligned`) reached the pricing and never the run | phases take `reach_spec()` | `assemble.rs` |
| the level-0 smoothing was a full-volume f64 phase written to and read back from the work store, and each fit block re-read a 4.7x window of it | `ShellFitOp::with_prefilter`: reads the raw image and smooths only the 32^3 tiles some candidate's fit samples, each with its full Gaussian reach (same bits as `SmoothOp`); no smoothed volume exists | `ops::star_convex` |
| the painter's depth buffer, and the executor's order check of `SeamFold::Unordered` ops, covered the read extent, which a fragment reach of one block makes 27 blocks | depth over the core; the order check compares pixels only over the region that is written | `ops::star_convex`, `strategy.rs` |
| a second fit pass ran even without a bootstrap | one pass, which paints | the example |

Scan 18–26 s → 3.2 s; fit pass 37–72 s → 17 s (painting pass ~21 s); whole run
3:11 → 71 s at 32 workers (prototype 83 s); peak RSS 11.9 → 7.3 GB.

No laziness hint was needed: every saving is an op not doing work whose result
nothing reads, decided from what the op itself holds (its rows, its declared
sides), so the plan, its halos and its pricing are unchanged.

Open, in blockflow rather than here:

- A fragment input's reach in whole blocks still widens a non-barrier phase's
  pixel read extent (`fragment_phase`), so a painter that reads no pixels gets a
  27-block output buffer and residency charge for a one-block core.
- Plateau labelling (`append_response_peak_table_phases`) is single-threaded
  per block; with one coarse block it was 1.5 s of the scan.
- Read and compute of a fragment phase's source windows do not overlap
  (`Hints::prefetch_depth` is reserved, not implemented).
