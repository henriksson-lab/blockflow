# Atlas registration

Registers a moving OME-Zarr volume to a fixed one, writes the transform as
elastix `TransformParameters.N.txt` files, optionally warps the moving volume
(or a label image on its frame) onto the fixed lattice as OME-Zarr, and carries
CSV point tables from the fixed volume's voxels to the moving volume's — through
one registration or a chain of them (signal → autofluorescence → atlas). It
also reads transforms elastix wrote, with every convention measured against
transformix.

There are two fits, both in blockflow's `ops::align`:

- **stochastic** (`fit_stochastic`): elastix's registration as the reference
  runs it — Mattes mutual information with Parzen windows, a seeded
  random-coordinate sampler drawing new samples every iteration, elastix's
  adaptive stochastic gradient descent with its automatic gain estimation, or
  standard gradient descent, multi-resolution smoothing pyramids, automatic
  affine parameter scales, and B-spline grids placed and refined as elastix
  places them. elastix parameter files with a stochastic optimiser — the
  reference's shipped `align_affine.txt` and `align_bspline.txt` among them —
  run this fit, unmodified.
- **Gauss-Newton** (`run`, `resident`): the block-planned fit, for the
  `--model`/`--metric` options and for parameter files with other optimisers.

Originally developed as part of a ClearMap reimplementation; the reference
named below is ClearMap 3.1.

## Pipeline (stochastic)

1. **Frames.** Each OME-Zarr level's `scale` and `translation` give a voxel
   frame (`p = scale * i + translation`, voxel centres at integers).
2. **Lattices.** Both images are put on lattices of voxel size `--resolution`
   (default: the fixed level as stored), **each over its own physical box**:
   the coarsest pyramid level no coarser than the lattice is read and resampled
   (trilinear, in parallel slabs; a level that is the lattice is used as read).
3. **Fit.** Every stage is `fit_stochastic` from fixed lattice voxels to moving
   lattice voxels. A stage after an affine is fitted after it, as elastix's
   `Compose` chains its files: the B-spline lives in the space the affine maps
   into, and its control grid covers the box the affine maps the fixed image
   onto.
4. **Write.** Each stage is converted to the fixed lattice's physical frame and
   written as `TransformParameters.N.txt`, stage `N` naming `N - 1` as its
   initial transform. After each stage the moving image is warped through the
   files written so far and compared with the fixed one (correlation and mutual
   information, in `registration.json`).
5. **Warp** (optional) and **points** as below.

**Direction.** As in elastix, the transform maps *fixed* coordinates to
*moving* coordinates. To carry detections from a sample into an atlas, register
with the sample as `--fixed` and the atlas as `--moving`; the warped output is
then the atlas (or, with `--warp-source annotation.zarr --nearest`, its labels)
in the sample's space. The reference does the same: its 3.1 configuration
registers with the atlas as the moving image, and its cell pipeline carries
points with `transformix -def` through signal → autofluorescence, then
autofluorescence → atlas.

## Point transforms

`PointTransform` is the library entry point for other examples:

```rust
use blockflow_atlas_registration::PointTransform;

let transform = PointTransform::from_registration_dir("registration/")?; // or a chain directory
let atlas_voxels: Vec<[f64; 3]> = transform.apply(&sample_voxels);
let back: Vec<Option<[f64; 3]>> = transform.apply_inverse(&atlas_voxels);
```

- `from_registration_dir(dir: impl AsRef<Path>) -> Result<PointTransform>`:
  `dir` is what `register` wrote (`registration.json`) or what `chain` wrote
  (`chain.json`, whose steps are applied in order).
- `apply(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]>`: **sample level-0 voxel
  indices `[z, y, x]` → atlas level-0 voxel indices `[z, y, x]`** (continuous;
  voxel centres at integers). "Sample" is the first fixed image, at level 0
  whatever level the registration ran on; "atlas" is the last moving image.
  Outside a B-spline's grid the deformation is the identity (as in elastix);
  no point is dropped. Points are mapped in parallel and returned in order.
- `apply_one(&self, point: [f64; 3]) -> [f64; 3]`, the same for one point.
- `apply_inverse(&self, points: &[[f64; 3]]) -> Vec<Option<[f64; 3]>>`: atlas
  voxels → sample voxels by Newton's method on the forward map (to 1e-3
  voxels); `None` where it does not converge.
- `steps()`, `sample_shape()`, `atlas_shape()`.
- `sample_frame(&self) -> &RegisteredImage` and `atlas_frame(&self) ->
  &RegisteredImage`: the first fixed image and the last moving image as the
  sidecars record them, `RegisteredImage { path: Option<PathBuf>, shape:
  [usize; 3], frame: Frame /* level-0 scale, translation */, unit:
  Option<String> }`. The unit is the sidecar's `unit` (written since
  2026-10-19), else the one the image's OME-Zarr declares, else `None`.

## Atlas space

`atlas_space::RegisteredAtlas` is a registration plus the prepared atlas
(`atlas-regions`' `atlas-prepare` output) it registered to. vessel-graph uses
it; cell-density's own copy (`src/registered.rs`) is to move onto it:

```rust
use blockflow_atlas_registration::atlas_space::{ImageFrame, LookupParams, OpenParams, RegisteredAtlas};

let registered = RegisteredAtlas::open(dir, &OpenParams { atlas: Some(atlas), check_reference: true })?;
let to_sample = registered.image_to_sample(&ImageFrame::from_ome_zarr(image)?, offset_um)?;
let mapped = registered.map(&to_sample, &image_voxels);       // .sample, .atlas, .outside_sample
let columns = registered.lookup(&mapped.atlas, &LookupParams { level, rounding })?;
let density = registered.density(&mapped.atlas, weights, rounding, radius)?;
```

- **Open.** The atlas defaults to the registration's moving image when that is
  a prepared atlas. Refused: a shape other than the moving image's, a voxel
  size other than its level-0 scale, and (unless `check_reference` is off) a
  reference that differs from the moving image at any voxel.
- **Frames.** `ImageToSample` is `sample = image * scale + offset`, from the
  image's and the sample's physical frames, with an offset in µm added to the
  image's physical positions. Units must agree when both are known.
  `frame_warning` flags an image that reaches more than a sample voxel outside
  the sample's box.
- **Lookups.** `Rounding::Nearest` (ties to even) or `Rounding::Floor`. Outside
  the atlas: region 0, hemisphere -1, the distance read at the nearest edge
  voxel. `level` collapses by tree depth. `annotator()` and `volumes()` give
  the `atlas-regions` pieces for callers with their own point sets.
- **Distance to the surface is in atlas voxels**
  (`AtlasLookup::distance_to_surface_voxels`): the prepared atlas's field, or,
  if it has none, one computed from the annotation (refused for non-cubic
  voxels). `voxel_length_um()` converts to µm on cubic voxels.
- **Density.** Each point adds 1, or its weight, to a ball of `radius` atlas
  voxels; points with no atlas voxel or a non-finite weight are dropped and
  counted. `region_voxels(level)` and `voxel_volume_mm3()` give region volumes.
- `describe()` is the JSON outputs record about the atlas space.

In a chain, step `k`'s moving image must be step `k + 1`'s fixed image (same
level-0 shape and frame, which is checked); points are handed on in its
level-0 voxels.

## Input

- OME-Zarr images, one channel each, any dtype. Axes must include `z, y, x`;
  the spatial unit is whatever the metadata says, the same for all images.
- elastix registration parameter files (`--parameters`, one per stage). For a
  stochastic file every key is mapped, accepted as output-only (result-image
  settings), or refused by name — `stochastic::STOCHASTIC_SETTING_SUPPORT` is
  the table. elastix's physical lengths (`SampleRegionSize`,
  `MaximumStepLength`, `FinalGridSpacingInPhysicalUnits`) are read as lattice
  voxels, which is what they are when the reference runs elastix on its
  unit-spacing images. Other files map to the Gauss-Newton fit
  (`compute::ELASTIX_SETTING_SUPPORT`).
- For `transform-points`: a CSV with a header row and three coordinate columns
  (`--columns`, default `z,y,x`).

**Orientation.** The images' frames are taken at face value. A sample in a
different orientation from the atlas must be reoriented first, or the atlas
prepared to the sample's orientation (as the reference's
`prepare_annotation_files` does).

## Run

```text
cargo build --release -p blockflow-atlas-registration
A=…/ClearMap/Resources/Alignment

# autofluorescence -> atlas, the reference's files
target/release/atlas-registration register \
    --fixed autofluorescence.zarr --moving reference_25um.zarr --resolution 25 \
    --parameters $A/align_affine.txt --parameters $A/align_bspline.txt \
    --out registration/ \
    --warped registration/annotation_in_sample.zarr \
    --warp-source annotation_25um.zarr --nearest

# signal -> autofluorescence -> atlas, as the reference's cell pipeline
target/release/atlas-registration chain \
    --image signal.zarr --image autofluorescence.zarr --image reference_25um.zarr \
    --resolution 25 --out chain/ \
    --parameters 0=$A/align_affine.txt \
    --parameters 1=$A/align_affine.txt --parameters 1=$A/align_bspline.txt

# where a part of a mosaic lies in its autofluorescence (no fit)
target/release/atlas-registration place \
    --fixed signal.zarr --moving autofluorescence.zarr --resolution 25 --out place.json

target/release/atlas-registration transform-points \
    --registration chain/ --input cells.csv --output cells_atlas.csv \
    --columns centroid_z,centroid_y,centroid_x          # add --inverse for atlas -> sample

target/release/atlas-registration register \
    --fixed sample.zarr --moving atlas.zarr --resolution 25 \
    --model affine-bspline --out registration/          # the Gauss-Newton fit
```

Stochastic options: `--seed` (default 121212; the result depends on it and not
on the thread count), `--relative-tolerance` (stop a level once the mean cost
over 50 iterations changes by less than this fraction; elastix has no such test
and runs every iteration), `--warped-stages` (write the moving image warped
after each stage).

### Memory cap

`--memory-mb` (any subcommand; else `$BLOCKFLOW_MEMORY_CAP`, else the
cgroup's limit, else a share of the machine). The registration's volumes are
not planned, but each is leased from the process's budget
(`blockflow::budget::process_budget`, through `memory::lease`) before it is
allocated and given back once it is freed: both lattices in `float64` for the
whole step, the fit's `float32` copies and each stage's smoothed copies of
them, each image read whole (as stored and widened) while it is resampled or
warped, each stage's warped image, the placement search's pyramids and
correlations, and a prepared atlas opened with `RegisteredAtlas` (its volumes
as stored, for as long as it is held). Under a stated or cgroup cap a lease
that does not fit stops the run, naming what needed it and the cap that would
hold it; under the machine's own share it is a warning. Under a binding cap
the allocator maps every buffer of 1 MiB or more, so freed chunk buffers go
back to the system rather than staying in the workers' heaps.

Measured 2026-10-23 on the osf-sa3x8 chain of the lightsheet pipeline
(stitched signal → autofluorescence → ABA 2017 25 µm, `--initialise
0=search`, 40 threads, load 32–50), before and after alternated; wall s / CPU
s / peak RSS:

| | before | after |
|---|---|---|
| uncapped | 84 / 1190 / 2376 MiB; 88 / 1198 / 2347 MiB | 81 / 1181 / 1422 MiB; 89 / 1180 / 1422 MiB |
| `--memory-mb 1280` | | 95 / 1205 / 1201 MiB |
| cgroup `MemoryMax=1280M` | | 104 / 1233 / 1199 MiB |
| `--memory-mb 1024` | | refused in step 1, naming the resampling read |

The transform files and `registration.json` (timings aside) are the same in
every run. The peak fell because the lattice warp writes its planes straight
into the output a row at a time: it used to collect slabs and then copy them,
with three `float64` coordinate volumes per slab on every worker.

### Initialisation

Where the first stage starts, i.e. where the moving image is taken to lie
against the fixed one before the fit moves it (`pipeline::Initialisation`;
later stages start from the stages before them):

| `register` | `chain` | start | elastix |
|---|---|---|---|
| `--initialise parameters` (default) | (default) | as the stage's file says | `AutomaticTransformInitialization` as written |
| `--initialise physical` | `--initialise K=physical` | the two images' physical frames as they stand | `AutomaticTransformInitialization "false"` |
| `--initialise centre` | `--initialise K=centre` | the fixed image's geometric centre on the moving image's | `"true"` with `AutomaticTransformInitializationMethod "GeometricalCenter"` |
| `--initial-translation Z,Y,X` | `--initial-translation K=Z,Y,X` | the physical frames, the fixed image moved by `t`: its point `p` starts at the moving image's `p + t` (the images' unit) | `-t0` with a translation transform, except that here `t` is the first stage's starting value, fitted with it and written into its file |
| `--initialise search` | `--initialise K=search` | as `--initial-translation`, with the `t` a placement search finds on the two lattices (below) | none |

The reference's `align_affine.txt` says `"true"`, so by default a registration
starts from the centres. That places a whole sample in its atlas, or a whole
mosaic in its autofluorescence, but not a part of a mosaic: its centre is not
the field's. When the frames place the part (tiles converted with their stage
positions, an autofluorescence image with its own), `physical` starts it
there; when they do not, `--initial-translation` does. An initialisation other
than `parameters` applies to stochastic stages only, and a translation needs a
translation or affine first stage. `registration.json` records it
(`initialisation`; for a search, the whole search under `search`, and its
time as `timings.place_s`). `--initial-translation` overrides a search.

### Placement search

`--initialise search`, or `atlas-registration place --fixed F --moving M
--resolution 25 [--out place.json]` on its own (`pipeline::place`,
`place::search_translation`), finds where the fixed image lies in the moving
one, translation only, on the lattices the fit uses:

1. **Pyramid**: 2x block means of both lattices, down to where the fixed
   image is still 16 voxels across on every axis (at most 8x the lattice;
   `--search-levels` to choose), intensities as `ln(1 + v)` (`--search-linear`
   to keep them).
2. **Exhaustive at the coarsest level**: the masked normalised
   cross-correlation at every integer translation that overlaps at least half
   of the smaller image (`--search-min-overlap`), from six FFT correlations
   (`ops::fft::RealTransform3`): box, image and squared image of each side,
   so means and variances are the overlap's own. The working set is bounded
   (1 GiB, a quarter of the memory cap, or what the cap leaves beside the
   volumes already held, whichever is least; coarser if needed) and leased
   from the process budget, as are the two pyramids.
3. **Candidates**: the 8 best local maxima, each refined level by level by
   hill-climbing the NCC from where the level above put it; candidates 0.15
   below the best are dropped, ones that meet are merged; a parabola per
   axis gives the sub-voxel peak at the lattice.
4. **Verdict**: refused when the best NCC is below 0.2
   (`--search-min-peak`), or when a distinct runner-up comes within 0.05
   (`--search-min-margin`) at the finest level that still compared one, with
   the candidates listed and a pointer to `--initial-translation`.

Rotation and scale are left to the fit, so on real pairs whose frames differ
by a few per cent in scale the peak NCC is well below 1.

## Output

- `TransformParameters.N.txt`: elastix parameter files in **`z, y, x` array
  order** (elastix's "x" is array axis 0), physical units of the fixed lattice
  frame. The highest `N` is the outermost link; transformix reads them as
  written.
- `registration.json`: both images' paths, levels, level-0 shapes and frames,
  both lattices, and per stage: per-level iterations, first- and last-tenth
  mean cost, the gain used (estimated, for the adaptive optimiser), seconds, the
  B-spline grid, and the correlation and mutual information of the fixed
  lattice against the moving image warped so far; wall-time breakdown.
- `chain.json` (`chain`): the step directories in order.
- `--warped`: an OME-Zarr 0.5 image on the lattice frame, `float32`, or
  `uint64` with `--nearest` on an integer source (labels).
- `transform-points`: the input CSV with three columns appended
  (`--prefix`, default `moving_`).

## Validation

Release build, `cargo test --release -p blockflow-atlas-registration`; the
`ops::align` kernels in `cargo test --release --features zarr --lib ops::align`.

**Placement search**: `place`'s unit tests (the FFT field against the direct
NCC at every tested shift to 1e-6; a crop found within a quarter voxel; two
identical copies refused as ambiguous; an unrelated pattern refused), and in
`tests/register_stochastic.rs` the unplaced strip below found within a voxel
and registered within a quarter of a moving voxel from it. On real data
(`tests/real_placement.rs`, ignored; 25 µm lattices, load 30-50 on 40 cores):

| pair | found (µm) | independent | NCC / margin | search |
|---|---|---|---|---|
| osf-sa3x8 nine-tile strip in its autofluorescence | (-11, 3404, 4541) | (0, 3420, 4535) by hand (NCC at 50 / 30 µm) | 0.558 / 0.20 | 1.0-4.3 s |
| osf-cztq2 whole mosaic in its autofluorescence | (247, 135, -369) | — | 0.712 / 0.39 | 1.8-2.2 s |

The osf-sa3x8 chain from the found translation and from the hand one put the
vessel graph's vertices a median 3 µm apart in the atlas (lightsheet-pipeline
README). On osf-cztq2 the affine started from the search equals the one
started from the centres (the reference file's default) to 1e-10 in every
parameter; outputs of the other initialisations are unchanged by the search
code (the same transform files, to the bit).

**Initialisation** (`tests/register_stochastic.rs`, a strip near one side of a
field twice the scene's size): each mode starts where it says, to 1e-6
voxels (the physical frames; the centres on each other; the frames moved by
the translation given), and from the frames, or from a translation 3 µm off
on every axis, the fit lands within a quarter of a moving voxel. On
osf-sa3x8's nine-tile strip in its autofluorescence (lightsheet-pipeline
README) the centres collapse the affine and the frames do not overlap; the
measured or searched translation registers it.

**Against the reference's elastix run** (`reference-clearmap/`, full numbers
in its README): osf-cztq2 autofluorescence resampled to 25 µm (248 × 478 × 285)
→ ABA 2017 25 µm reference prepared for this sample (256 × 528 × 320), both
sides with `align_affine.txt` then `align_bspline.txt`, unmodified:

| | elastix | native |
|---|---|---|
| correlation, fixed vs warped reference, after affine / B-spline | 0.659 / 0.655 | 0.663 / 0.657 |
| mutual information (64 bins), after affine / B-spline | 0.490 / 0.528 | 0.493 / 0.537 |
| fit time, affine / B-spline, 40 threads | 75 s / 436 s | 6.6 s / 53 s |
| whole command (with warps and outputs) | 532 s | 76 s |

Warped annotation, native against elastix, Dice per major region after the
B-spline: 0.90 (fiber tracts) to 0.98 (midbrain, striatum, cerebellum), whole
brain 0.994 (elastix's own affine against its B-spline: 0.44 to 0.90). Point grid (31 497 points inside the sample) through both: median
16 µm, 95th percentile 51 µm apart — the same as two native runs with
different seeds (16 and 49 µm), so within the method's own sampling noise; the
B-spline itself moves points by a median of 173 µm. Through the two-step chain
(signal → autofluorescence → atlas): median 17 µm, 95th percentile 50 µm.

**Synthetic** (`tests/register_stochastic.rs`): the reference's two files read
setting by setting; refused settings named; a known shift recovered by the
stochastic affine through the pipeline within a quarter of a moving voxel and
inverted back to 0.01 voxel; a two-step chain composed within 0.15 voxel.
`ops::align`'s tests check the Mattes gradient against central differences for
an affine and a B-spline, the cubic sampler, B-spline grid refinement,
elastix's grid placement against a real elastix file, and that a fit is the
same with 1 and 4 workers.

The files the stochastic fit writes (an affine and a B-spline on an explicit,
elastix-placed grid, chained) give the same points through transformix 4.9 as
through `transform-points --units physical`, to transformix's printed
precision (run from the registration directory: the initial-transform name is
relative, and this crate's reader resolves it against the file's directory).

**Point transforms against transformix 4.900** (unchanged): 15 recorded
cases, 147 points — affine in both point spaces with an asymmetric direction
and anisotropic spacing, Euler under both `ComputeZYX`, both chain combination
rules, B-spline coefficient layout, support and valid region, index rounding at
negative halves — reproduced within transformix's printed precision (`1e-6`).
**Negative controls** (`tests/register_controls.rs`): an inverted transform, a
transposed matrix, a dropped index/physical round trip and a swapped chain each
move all 1000 probe points by a measured mean (25.41, 5.14, 14.04, 5.1),
against 11.12 for not registering at all.

## Limitations

- **The Parzen estimate of mutual information prefers a slight shrink on very
  smooth images** (about 2 % on a scene of broad blobs with 32 bins, measured
  independently in numpy). elastix's estimator is the same one; on the real
  atlas pair the native and elastix results agree to the noise above.
- **Stochastic fit, not planned.** Both lattices are held in memory (an atlas
  at 25 µm is tens of millions of voxels, about 1.1 GB with the fit's copies
  on the osf-sa3x8 chain), leased under the cap but not made smaller by it;
  each iteration's samples are split into a fixed number of chunks evaluated in
  parallel and reduced in order.
- **Not implemented** (refused by name): masks, multi-metric registration,
  Euler/similarity transforms, explicit `Scales`, `Full`/`Grid` samplers with
  the stochastic optimisers, automatic gain estimation for a B-spline, a stage
  after a B-spline, Gauss-Newton and stochastic stages in one run.
  `RecursiveImagePyramid` is approximated by smoothing without downsampling,
  and `Random` (voxel) sampling by random coordinates.
- **The Gauss-Newton fit** keeps the B-spline Hessian as its diagonal (so dense
  grids fit in memory), measures affine steps in scaled parameters with
  `scaled_affine_steps`, stops on a relative cost change with
  `relative_tolerance`, and runs coarse-to-fine levels on `workers` threads;
  its mutual information is still the bounded-intensity approximation, and it
  is not what the parameter files above run.
- `transform-points --reverse-axes` is for bare elastix files written in
  `x, y, z` order and is refused with a registration directory.
