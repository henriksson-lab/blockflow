# Plan: whole-organ light-sheet analysis at ClearMap's level

Written 2026-10-18. Follows `LIGHTSHEET_MIGRATION_PLAN.md`.

## Goal

Software that does what ClearMap 3.1 does, with similar results and similar
speed, using similar algorithms.
- Bit-for-bit parity is **not** a goal.
- Agreement within stated tolerances **is** a goal, measured against ClearMap
  on real data.
- The GUI is out of scope. Long term, newvolim
  (`/home/mahogny/github/claude/newvolim`) replaces ClearMap's GUI, so the
  outputs here should be things newvolim can show (see section H).

ClearMap's source, for reference only:
`/data/henriksson/github/claude/clearmap-rs/ClearMap-py/ClearMap` (3.1.0).

## Where we stand

The per-sample image processing is in place, mostly with ClearMap's own
algorithms, in `examples/{tile-stitching, atlas-registration, cell-density,
vessel-graph}` and `crates/atlas-regions`. Specifically:
- stitching: rigid, wobbly, placement and fusion;
- cell detection: background, maxima, watershed shape, size filter, crust,
  region counts;
- vessel binarisation, all five arms;
- skeleton (PK12), graph build, clean, reduce, radii, artery/vein classification;
- tract maps;
- point transforms read from elastix files.

What is missing is mostly *around* those stages:
- atlas orientation;
- atlas-space outputs, and the links between the stages;
- group statistics;
- batch processing;
- building the tile set from converted tiles' metadata (conversion itself is `img2omezarr`'s job);
- whole-brain memory scale.

On top of that, the registration fit is weaker than elastix.

Algorithm substitutions that may change results materially:
- **registration metric and optimiser** (highest risk);
- **no deep vessel filling** (on by default in ClearMap 3.1);
- **one clip range for all vessel channels**;
- **region volumes measured in sample space instead of atlas space**.

## Phases

### A. One sample, end to end, in atlas space (first)

Outcome: stitch → register → cells, vessels and tracts in atlas coordinates,
labelled, with densities on the atlas grid. This is what everything after
depends on.

1. **Orientation and atlas preparation** (ClearMap `Alignment/orientation.py`,
   `Annotation.prepare_annotation_files`).
   - Signed axis permutations for volumes, points and resolutions.
   - Atlas crop and hemisphere selection.
   - These go into `crates/atlas-regions`, or a sibling crate if they grow.
2. **Atlas resources.** Read ClearMap's shipped ABA files (annotation,
   reference, hemispheres, ontology JSON) from a user-given directory, and
   convert them once to OME-Zarr with region names and hierarchy. The atlas
   files themselves are not committed.
3. **Registration as good as elastix on ClearMap's own parameter files** (`ops::align` in core):
   - Mattes MI with a random-coordinate sampler;
   - stochastic gradient descent;
   - affine parameter scaling and relative stopping tolerances;
   - a sparse or diagonal B-spline Hessian, so grids at 25-voxel spacing fit;
   - parallel coarse-to-fine fitting.

   Also: accept `Resources/Alignment/align_{affine,bspline}.txt`, and validate
   against elastix on several brains, not one pair.
4. **Two-step registration chains** (signal → autofluorescence →
   atlas), with composed point transforms.
5. **Atlas-space outputs.**
   - `cell-density --registration <dir>` maps cells into atlas voxels, labels
     them there, applies the crust filter in ClearMap's units, voxelises on the
     atlas grid (optionally weighted), and takes region volumes from the atlas.
   - `vessel-graph graph --registration <dir>` runs transform, annotate,
     distance-to-surface and branch density (the library has these; the CLI
     doesn't call them).
   - The same for tract points.
6. **Per-channel vessel parameters**: ClearMap 3.1's separate clip ranges and
   thresholds for vessels and arteries, and a separate artery mask for
   classification.
7. **A comparison harness against ClearMap**, as `reference-clearmap/`
   scripts in each example, following the other examples' `reference-*`
   convention. Run ClearMap and the example on the same sample and report the
   agreement. Proposed tolerances, to be fixed after a first run:

   | stage | measure |
   |---|---|
   | stitching | per-tile position difference in voxels |
   | registration | landmark or label overlap (Dice per major region) |
   | cells | detected-cell matching within 2 voxels; region counts within a few per cent |
   | vessels | mask Dice; total vessel length per region |
   | runtime | wall time and peak memory, same machine |

### B. Group statistics (ClearMap `Analysis/Statistics`)

A new example, e.g. `examples/group-statistics`, over the atlas-space outputs
of phase A:
- voxel-wise t-tests and p-value maps, including signed / coloured p maps;
- region-count t-tests;
- Benjamini–Hochberg correction and q-values;
- summary tables.

Distribution tests (Anderson–Darling, Cramér–von Mises) come after those.

### C. Whole-organ scale

These stages currently hold whole volumes in memory, where ClearMap works from
disk. Make them planned (out-of-core) and benchmark them against ClearMap on a
whole brain:
- **cell watershed**: switch `cell-density` from `ScalableWatershedOp` to
  core's `TiledWatershedOp` with global seeds.
  - `ScalableWatershedOp` avoids one global priority queue, but still declares
    `Reach::all`, so its phase holds cost, mask, seeds and labels for the whole
    volume (about 18 B/voxel). Its doc's own plan is to move component discovery
    into fragment phases.
  - `TiledWatershedOp` holds one tile plus its halo, and keeps global seed ids
    across seams. Its answer depends on the tile grid by contract, but ClearMap
    also does shape detection per block with overlap, and cells (50–900 voxels)
    are far smaller than a halo of a few cell diameters.
  - Measure the change against the current whole-volume answer when switching;
    no new core op is needed;
- **Requirement (user, 2026-10-19):** every conversion to block-by-block steps must keep or improve wall time, CPU time and peak memory. Measure each stage before and after on the same data, and the outputs must not change.
- **stitching**: fusion as a planned phase over several supplied inputs, plus a parallel pyramid write. Today fusing at level 2 takes 472 s, mostly the single-threaded write;
- **vessel graph**: radius walk, signal reads, branch density;
- **atlas lookups**.

### D. Remaining algorithm gaps

- **Deep vessel filling CNN** (on by default in ClearMap 3.1): port the model
  to burn/candle as the other model-backed examples do. Weights from ClearMap's
  resources, if their licence allows.
- **Vessel quantification**: physical edge length, vessel length density per
  region, radius statistics, orientation (radial/planar).
- **Check the binary-smoothing table**: compare our derived 2^27 table with
  ClearMap's `Smoothing.npy` directly.
- **Optional ClearMap compositions** where ours differ, offered as options and
  not defaults: cv2-style background opening, histogram percentiles.
- **Smaller items**:
  - object-level colocalization between two cell tables;
  - flatfield / illumination correction;
  - CellMap equalisation;
  - inverse transforms and deformation fields;
  - graph corrections.

### E. Input: OME-Zarr only

Every example takes OME-Zarr and nothing else. Converting raw TIFF tiles, file-name patterns or OME-TIFF layouts is the job of `/home/mahogny/github/claude/img2omezarr`, not of anything here.

What remains on this side is small: `tile-stitching` should build its tile set from the converted tiles' own metadata (grid position, physical position) where `img2omezarr` records it, instead of needing a hand-written tile-set JSON.

### F. Batch processing

One driver that runs phase A for every sample directory from one parameter
file. Its output is a fixed atlas-space layout per sample, which phase B
consumes.

### G. Landmark registration

Registration takes paired fixed/moving landmarks: elastix `.pts`, or a table
of `landmark_id, fixed_z/y/x, moving_z/y/x`. This pairs with newvolim's
landmark annotation (section H).

### H. Outputs newvolim can show

newvolim (ROZT) already displays OME-Zarr images and labels, `ngff-object-table`
cell tables (with gating and filters), atlas names, and per-region counts. It
depends on blockflow's `ngff-object-table`, `ngff-label-map` and
`blockflow-label-edit`.

**blockflow side**

- Vessel graphs as two NGFF object tables (vertices; edges with
  source/target, radius, length, class, region, polyline) instead of, or
  beside, JSON.
- Atlas labels warped into the sample frame (`labels/atlas`), with
  `id, name, acronym, parent, level` in `image-label.properties` or an ontology
  table; hemisphere as a second label image.
- Per-region results as a label-keyed table.
- Each example registered as a newvolim job definition with a JSON-Schema
  payload, including a crop/ROI field for tuning on a subset.

**newvolim side** (that repo's plan, listed here so the contracts line up)

- Paired registration landmarks, and linked cursors across two datasets.
- An atlas structure tree: hierarchy in `AtlasEntry`, pick, isolate.
- A polyline/graph layer with column colouring and the existing filters.
- ROI annotation → job payload.
- Optional: NGFF displacement-field transforms.

## Progress

### A1/A2 (done 2026-10-19)
- `atlas_regions::Orientation` matches the reference's orientation functions for 12 orientations; preparation matches voxel for voxel.
- Full ABA 2017 25 µm atlas as OME-Zarr: `/husky/henriksson/lightsheet-reference/atlas/ABA_25um_2017.ome.zarr`.
  - Annotation labels carry the ontology; hemispheres are 0/255, so 0 is a real hemisphere.

### A3/A4 (done 2026-10-19)
- `ops::align::fit_stochastic` uses Mattes MI, a random sampler and elastix-style adaptive SGD. It accepts the reference's `align_{affine,bspline}.txt` unmodified.
- osf-cztq2 autofluorescence → atlas, ours against elastix:
  - correlation 0.657 vs 0.655, MI 0.537 vs 0.528;
  - warped-annotation Dice per major region 0.90–0.98;
  - point difference median 16 µm, 95th percentile 51 µm, the same as ours against itself with another seed;
  - whole command 76 s vs 532 s.
- Two-step chains, with points composed through `PointTransform`.
- On this sample the atlas orientation is `(1, 2, 3)`, not the `(1, -2, 3)` used in the reference's example script.
- Open items:
  - validated on one brain only;
  - masks, Euler/similarity transforms and multi-metric registration are refused;
  - transformix must run from inside the registration directory.

### C, cell watershed (done 2026-10-19)
- `cell-density` now uses `TiledWatershedOp` with global seeds by default: 64×128×128 tiles, 16-voxel halo. The old path stays behind `--watershed whole`.
- A basin whose seed lies beyond the halo is cleared (`DropEdgeBasinsOp`), not merged.
- The result is deterministic across worker counts and block grids.
- Agreement:
  - against one tile over the whole crop: 34 248 of 34 251 cells have the same size;
  - against the old path: 98.5% the same. Most differences come from `ScalableWatershedOp`'s blockwise fallback for components over 16 M voxels, which changes basins.
- Whole cFos tile (1.04 G voxels): 28.7 min and 41 GB, against 34 min and 60 GB before.
- Open items:
  - outside the watershed, peak memory is still about 40 B/voxel, and the tabulation merges are slow;
  - each tile searches its whole buffer for edge seeds.

### A7, comparison harness (stitching and cells, 2026-10-19)
Scripts are in `examples/{tile-stitching,cell-density}/reference-clearmap/`; outputs are under `/husky/henriksson/lightsheet-reference/{stitching,cells}/`.

**Stitching** (3×3 tiles of osf-sa3x8)
- Rigid: identical to the reference, with a tile position difference of 0 voxels.
- The reference's wobbly stage has a bug. `stitching_wobbly._connected_components` overwrites `displacements` before reading it, so its per-slice shifts are never applied. The harness can run it with that line corrected.
- Per-slice, against the corrected reference with the same ±30 search: median 1 voxel, and 68–78% of planes within 1 voxel. The reference's default searches ±15 in y; per-axis ranges are being added.
- Speed: 35–41 s against 59–145 s.

**Cells** (osf-cztq2 cFos tile 03x02)
- Detections match at 97.7% precision and 96.9% recall.
- After the size filter: 84 424 cells against 91 687. Most of that gap comes from the reference's 16-plane z blocks; run as one block, recall is 96%.
- The reference's shipped config uses threshold 500, against our default of 700.
- **Performance gap, to fix:** a whole tile takes 3 491 s and 48 GB, against 837 s and 6.2 GB for the reference. CPU time is 9× the reference's. Something besides the watershed still holds the whole volume.

### A5, vessels and tracts (done 2026-10-19)
- `vessel-graph graph|tract --registration <dir> [--atlas <prepared.ome.zarr>]` maps points into atlas space, then labels them there and voxelises on the atlas grid.
- Frames:
  - the image goes to the registration's fixed image through the two OME-Zarr physical frames;
  - the fixed image goes to the atlas through `PointTransform`;
  - an atlas whose shape or reference differs from the registration's moving image is refused.
- `segment`/`tract --crop` writes a crop that keeps the source's physical frame.
- The osf-sa3x8 atlas fits best in orientation `(1, -2, 3)` (MI 0.547), while osf-cztq2 fits best in `(1, 2, 3)`. Orientation is per sample.
- **Gap: the chain from tiles to the atlas.** A converted tile sits at the physical origin, not at its place in the brain, so a registered single-tile crop lands in an atlas corner. Fix: stitch, then register the stitched signal → autofluorescence → atlas, as the reference does, and run cells and vessels on the stitched frame. This is the remaining A item, and it is also the core of F.
- Core candidate: `RegisteredAtlas` in `vessel-graph/src/atlas_space.rs` (registration plus prepared atlas plus frame mapping). Cells need the same, so it should move to a shared crate. Also wanted: a `PointTransform::sample_frame()` accessor.

### A5, cells (done 2026-10-19)
- `cell-density --registration <dir> --atlas <prepared.ome.zarr>` maps, labels, applies the crust filter, counts and voxelises in atlas space.
  - Region volumes come from the atlas.
  - `regions.json` describes the label-keyed `regions.csv` (item H).
- The default threshold is now 500, from the reference's shipped `cell_map.yml`.
- On osf-cztq2, a 512×640×640 crop of tile 01x02 (22k cells), ours against the reference's elastix and its labelling:
  - same finest region 90.3%, same major region 97.0%, same hemisphere 100%;
  - crust decisions agree on 99.9% of cells, region volumes are identical, and density correlates at 0.991;
  - the finest-label agreement is limited by the registration's 25 µm median difference. Changing only the rounding gives 90.2%.
- The atlas step adds 5.5 s and 2.5 GB.
- Shared helper: `blockflow_atlas_registration::atlas_space::RegisteredAtlas`, plus `PointTransform::sample_frame()` / `atlas_frame()`.
  - vessel-graph uses it.
  - Distance to surface is in atlas voxels, the reference's unit.
  - cell-density uses it too. Outputs on the real crop are identical, except that the column is now `distance_to_surface_voxels`.

### Phase A status (2026-10-19)
Done:
- one sample end to end in atlas space (osf-cztq2 cells);
- comparison harnesses for stitching, registration and cells.

Members, release:

| crate | pass | ignored |
|---|---|---|
| atlas-regions | 39 | 2 |
| tile-stitching | 72 | 4 |
| atlas-registration | 123 | 7 |
| cell-density | 51 | 5 |
| vessel-graph | 99 | 6 |
| core lib (zarr) | 1487 | 25 |

All fail 0, and clippy `--no-deps` is clean.

Still open in A:
- validating registration on more brains;
- osf-sa3x8 vessels in the atlas (partial mosaic; needs an initial transform or G);
- a comparison against the reference's vessel graph.

### Stitching search ranges (done 2026-10-19)
- `--max-shift Y,X` (per slice, default `15,30`) and `--rigid-max-shift Y,X` (default `30,30`), following the reference's `stitching.yml` v3.1.
- The reference has a second defect: `shifts_from_tracing` breaks ties by reading errors at the measurement index, not at the slice.
- With both of the reference's defects patched, against ours, seam offsets have a median of 0. 97% (y) and 93% (x) of planes fall within 1 voxel, and the fused images correlate at r 0.99.
- What is left: the reference smooths each tile's positions over the whole stack, where we smooth over the tile's own slices.

### Whole chain on real data (osf-cztq2, done 2026-10-19)
- Stitch 40 tiles (rigid plus wobbly), then register the fused signal → autofluorescence → atlas, then run `cell-density --registration` on a level-0 box of the fused image. No hand-given offset is needed.
- Times:

  | stage | wall | peak memory |
  |---|---|---|
  | align | 243 s | 21 GB |
  | fuse at level 2 | 472 s | 14 GB (mostly a single-threaded pyramid write) |
  | chain registration | 90 s | 3.2 GB |
  | cells on the box | 52 s | 8.9 GB |

- Against the earlier hand-offset run: atlas positions differ by a median of 20 µm; 92.3% of cells fall in the same finest region and 100% in the same hemisphere. The (−5, −7, +12) µm systematic shift is explained by the old mosaic's column pitch and by the reference's half-voxel resample origin.
- Driver: `examples/lightsheet-pipeline/run.sh` runs stitch → register → cells → vessels into a fixed per-sample layout (see its README).
- `tile-stitching --tile-dir` takes the grid from the tile names. `--layout-from` plus `--region` fuses a box at stored positions on any level.
- `--tile-dir` reads only standard OME-Zarr metadata and the file name. It reads no instrument or vendor metadata; that is the converter's job.
  - Grid index: from the name (`…03x04…`).
  - Positions: from distinct level-0 NGFF translations, or else from `--overlap Y,X`.
- **For img2omezarr** (item E): write each tile's nominal position as a level-0 NGFF `translation` (physical units, z/y/x, stage signs resolved). Today every tile sits at the origin, so osf-cztq2 needs `--overlap 120,120`.
- **osf-sa3x8 vessels in the atlas: blocked.** The conversion holds only 9 tiles (a 1.7×5.4 mm strip of a 6.6×11.5 mm field). Registering signal → auto by centring cannot place it. This needs the whole mosaic, an initial-transform option in `atlas-registration`, or landmarks (G).

### C, cell performance (done 2026-10-19)
Whole cFos tile (1.04 G voxels), measured back to back with the reference at load 5–11:

| | wall | CPU | peak |
|---|---|---|---|
| reference | 81 s | 779 s | 7.0 GB |
| ours, now | 99 s | 807 s | 5.6 GB |
| ours, before | 1 767 s | 11 282 s | 44 GB |

The cell tables are byte-identical before and after.

Fixes:
1. Tabulation merges were quadratic in the block count. `MergeTabulationOp::folded_once` (core) folds them once instead.
2. Seed numbering now happens once, then each block gets its own slice.
3. The chunk cache is capped by `--cache-mb`, default 2048, and malloc arenas are limited.
4. `ExtremeFilterOp` (running extremes) replaces tap-by-tap rank gathers on halos.
5. The tiled watershed searches for edge seeds only from cut faces (core).
6. Intermediates are stored as u16/bool, not f64.

The remaining wall-time gap is gzip I/O in the label phases.

Core candidates:
- the same quadratic merge in `classical-3d-centers` and `ops::measure`;
- `RegionPointsOp` decodes every component in every block;
- `ops::rank`'s halo path (with `extreme.rs` as the template);
- the default zarr chunk cache, an eighth of RAM.

### Tile positions and osf-sa3x8 vessels in the atlas (done 2026-10-19)
- **img2omezarr** writes tile positions as NGFF translations (`TILE_POSITIONS_PLAN.md` steps 1–3).
  - Its LaVision source has a default axis mapping of `x,-y,-z`, verified against the stitched cFos layout: pitch scale 0.98, max residual 12 µm.
  - `set-positions --from-originals` reads the original files.
  - bioformats-rs is back to Bio-Formats behaviour. img2omezarr's lock still pins the old bioformats-rs commit.
- **tile-stitching**: neighbours come from translated boxes, so names are optional. `--layout-from` accepts a finer z.
- **atlas-registration**: `--initialise parameters|physical|centre` and `--initial-translation Z,Y,X` (per step in `chain`).
- **osf-sa3x8**:
  - Its autofluorescence has no stage position, so the signal→auto initial translation `0,3420,4535` µm was found by a log-intensity cross-correlation (`lightsheet-pipeline/checks/find_translation.py`, outside the pipeline).
  - The vessel graph (58 788 vertices) lands 99.99% in isocortex. Layer depths measured from the sample surface are monotonic: L1 96, L2/3 314, L4 512 and L5 645 µm.
- Open:
  - make the initial-translation search a pipeline step (or landmarks, G);
  - osf-cztq2 row 07 still has no valid stage positions;
  - compare the vessel graph against the reference's TubeMap.

### D, deep vessel filling (done 2026-10-19)
- `vessel-graph` feature `deep-fill` (burn, CPU): the reference's CNN as a block-planned `BlockOp` with a 64-voxel halo (128 at resample 2).
  - On by default with the feature, as in the reference's 3.1 config.
  - Weights come from `--deep-fill-weights` / `$VESSEL_FILL_WEIGHTS` and are never committed. The code is GPLv3; the weights' licence is unclear (the `setup.py` says MIT, the LICENSE says GPLv3).
- osf-sa3x8 crop:
  - the network output matches the reference network within 5.6e-4;
  - the filled mask has Dice 0.966 (vessel) and 0.974 (artery) against the reference's eval-mode `fill_vessels`;
  - CPU time is the same as the reference's, at 0.73 GB against 8.1 GB.
- The reference never calls `eval()`, so dropout stays active at inference. Its fill is nondeterministic (run-to-run Dice 0.958) and adds about 50% more voxels. It also zeroes the last planes of each axis. Ours is deterministic.
- Graph on the crop: 425 mm total length, against 406 mm for the reference in eval mode and 387 mm as shipped.
- Open: a GPU path (burn CUDA), and validating on a whole tile.

### C, stitching fusion (done 2026-10-19)
- `--layout-from` fusion is now planned slab by slab, reading only the tile boxes each slab needs, with a parallel pyramid writer (`tile-stitching/src/{planned,pyramid}.rs`). Every output is byte-identical to before.

  | run | wall | CPU | peak |
  |---|---|---|---|
  | osf-cztq2, level 2 | 809 → 50 s | 798 → 342 s | 13.9 → 5.3 GB |
  | osf-sa3x8, level 1 | 313 → 79 s | 405 → 278 s | 11.0 → 2.5 GB |

  The machine was at load 45–95, so wall times are approximate.
- Core candidates:
  - a phase input on its own lattice, with per-block read boxes, which would make fusion a core phase;
  - `build_multiscale` encodes one chunk at a time and re-reads each level, which slows every core user;
  - `chunk_for_block` gives one-row chunks for prime extents.
- The writer copies two private core pieces, the unwritten-fill rule and the multiscales metadata writer, and could drift from them.
- Open: fusion during alignment (without `--positions-only`) still holds the fused volume.

### C, vessel graph block by block (done 2026-10-19)

- Outputs are byte-identical before and after, on the crop and on the fused osf-sa3x8 region (67 M voxels), with and without deep fill. Runs and loads are in `/husky/henriksson/lightsheet-reference/vessel-planned/runs/summary.txt`; the full table is in the example README's "Out of core" section.
- `graph`: the masks are packed into one byte a voxel, and radii, signals and neighbours are measured per block in parallel; the JSON is streamed. Region: 45 → 27 s, CPU 43 → 36 s, peak 3.85 → 2.07 GB.
- `tract`: the points are streamed one row of blocks at a time. Region peak 6.07 → 4.49 GB; wall and CPU are unchanged within noise.
- `segment`: a stored skeleton thinning, one round per pass over disk, gives the same skeleton. It is about twice as slow in wall time, so it only switches on above `--thin-memory-mb` (default 8192). Two malloc arenas cut about 0.7 GB. Region: 250 → ~160 s, peak 9.7 → 9.0 GB.
- Still in memory: the graph itself, at about 430 bytes per skeleton voxel plus about 450 bytes per edge point while the tables are written. At the crop's density a whole brain would need several hundred GB. This is the next limit for whole brains.
- Core candidates:
  - a stored or tiled iterative phase, which would make the thinning workaround unnecessary;
  - a columnar object-table writer;
  - a typed `ops::rows::value_at` path (26% of `graph`'s CPU);
  - the mask-packing op;
  - a better default chunk cache size.
- Not done: a whole-tile run; `--density-level 0` branch density and `--annotation-label` stay in memory.

### Benchmark suite against ClearMap (scripts done 2026-10-20)

- `examples/lightsheet-pipeline/benchmark/` (`bench.py list|prepare|run|summarize`) runs seven workloads on both sides with matched parameters: two stitchings, registration, cells, vessels, vessels with deep fill, and end to end. It measures the whole process tree (wall, CPU, summed RSS and PSS, I/O) and records load and provenance. The README lists every difference between the two sides.
- Results go in `/husky/henriksson/lightsheet-reference/benchmark/results/`.
- Only the smoke sizes have run, under load, so there are no numbers yet. The full run takes 10–14 h on a quiet machine.
- The vessels speed gap seen in the smoke run (blockflow ~117 s against the reference's ~91 s, equal CPU) is fixed, and segment output is byte-identical (2026-10-20). Smoke segment: 123 → 31 s, CPU 900 → 270 s, peak 4.04 → 3.74 GB. Region: 174 → 54 s, CPU 1752 → 458 s, peak 8.84 → 8.14 GB.
  - The rank filters use `SlidingHistogramOp`.
  - Thinning visits border lists, and an iterative op can now declare itself `stationary` to get a per-block dirty set.
  - Core chooses the block grid with `Host::detect` and `choose_grid`; on smoke it picked z-slabs. `Cursor::partition` now offers only the axes the grid cuts.
  - `tract`'s smoke peak rose from 2.0 to 2.2–2.5 GB while it got 3× faster. The user accepted that, provided the ops respect memory caps, which is now done (2026-10-22):
    - The cap comes from `--memory-mb`, then `BLOCKFLOW_MEMORY_CAP`, then the cgroup limit, then the machine rule.
    - The planner treats the cap as a hard limit on the active set (blocks in flight × per-block bytes) and prices the cache as a cost.
    - Caches lease from the same budget.
    - The executor admits blocks against the cap.
    - Capped runs stay under the cap, including under a real cgroup limit, with identical outputs and an unchanged uncapped speed. Details in `/husky/henriksson/lightsheet-reference/vessel-speed/memcap/`.
    - Open: waves are synchronous, so tight caps cost wall time (admitting a block as each one finishes would recover it); tile-stitching and the in-memory graph stages aren't cap-aware. Details in `examples/vessel-graph/README.md` "Speed" and `/husky/henriksson/lightsheet-reference/vessel-speed/`.
- The steady 20 µm end-to-end atlas offset is explained: the reference resamples images centre-aligned (`cv2.resize`) but points from the corner (`p * f`), then labels by truncation. Carried in voxel centres it agrees to 4.8 µm and 97.3% same region. We keep NGFF voxel centres throughout; `checks.py e2e` reports both conventions (benchmark README, "Atlas positions and the voxel convention").

### Round 3 (2026-10-22/23)

Six parallel agents; reports in `/husky/henriksson/lightsheet-reference/round3/{A..F}/`. On the combined tree, core (1831), vessel-graph, tile-stitching, cell-density and atlas-registration tests pass.

- **A, vessel graph memory:** the graph takes 62–77 B per skeleton voxel, down from ~430 (ClearMap is about 60–90). A whole brain needs about 53–69 GB. Added core `ObjectTableBuilder` (spills to disk) and `ReadAsF64`. The graph leases its bytes against the cap. Outputs are byte-identical.
- **B, graph against ClearMap:** graph construction is identical given the same skeleton. The differences come from thinning axis order (z,y,x against x,y,z; a convention, kept) and our 1.9% larger mask. Region: branch F1 0.79 and 4562 against 4482 mm/mm³ overall; 0.88 with the thinning matched. Scripts are in `vessel-graph/reference-clearmap/compare_graph.*`.
- **C, executor, pyramid, stitching:** streaming admission; core `MultiscaleWriter`, which tile-stitching now uses; stitching reads boxes and respects `--memory-mb`. osf-cztq2 L1 align: 254 → 22 s, 21.3 → 7 GB. `chunk_for_extent` / `chunk_for_grid` fix one-row chunks, so fused chunk files differ while their voxels are identical. `BlockOp::max_in_flight`.
  - Worse: osf-sa3x8 align uses +15% CPU, and its fuse peaks +0.5 GB (from the chunk fix).
  - Open: barriers wait for every earlier phase, not only the phases they read.
- **D, window statistics:** core `ops::lattice_samples` computes each lattice point once. Smoke CPU 311 → 248 s. Core clippy is clean.
- **E, placement and positions:**
  - `atlas-registration --initialise search` and `place` find osf-sa3x8's placement within one atlas voxel.
  - bioformats-rs gives plane position units, and img2omezarr uses them.
  - osf-cztq2 row 07 is placed through `set-positions` on a grid; the original `zarr.json` files are backed up.
- **F, GPU deep fill:** the `deep-fill-cuda` and `deep-fill-wgpu` features and `--deep-fill-device`. Deep fill matches the CPU to 1e-6 and runs about 6× faster. Segment with deep fill on the region: 102–143 → 59–80 s.
- **Integration:**
  - Fixed one compile error in deep fill's `max_in_flight` (`self.model.on_gpu()`).
  - Fixed a cap breach: the lattice-samples interpolate phase held about 46 B a voxel against 6 B charged. Ops now declare `FragmentOp::scratch_bytes`, the interpolate phase works plane by plane, and admission gives unmeasured phases 2× headroom and stops admitting when resident memory exceeds the leases. Tract at a 512 MiB cap: 975 → 307 MiB; every capped run stays under its cap, outputs are identical, and the uncapped speed is unchanged. Regression test: `tests/admission_underestimate.rs`.

### Round 4 (2026-10-23)

Reports in `/husky/henriksson/lightsheet-reference/round4/{G,H,I}/`. Core with `zarr`: 1841 tests pass.

- **G, round 3's regressions:** all fixed; every run matches or beats both HEAD and round 3 on wall, CPU and peak.
  - sa3x8 L1 align CPU 75 / 84 → 64 s: one box per pair, shared between the rigid and per-slice steps under a lease.
  - sa3x8 `--layout-from` peak 2.6 / 3.1 → 1.9 GB: one chunk row per slab.
  - Fused pyramid peak 2.51 / 2.85 → 2.4 GB.
  - `fuse::resident` and `stitch` go through planned fusion.
  - Positions are identical, and fused voxels are identical.
- **H, registration memory:** the resident volumes lease from the budget (`atlas-registration/src/memory.rs`). The sa3x8 chain peaks at 2.36 → 1.42 GB with identical transforms, and capped and cgroup runs stay under the cap. The vessel-graph README's "Speed" and "Memory cap" sections are rewritten.
  - Open: `RegisteredAtlas::density` is unleased.
  - Open: `memory::lease` duplicates the vessel graph's `lease_resident`, and both could move into `budget`.
- **I, benchmark smoke:** all 7 workloads ran on both sides and the checks passed.
  - Fixed: the overnight command's argument parsing; the elastix and transformix thread pool, which inflated the reference's registration CPU about 6.6×; one failure ending the run; the order of the load wait and the warm-up.
  - Estimate: 7–10 h.
  - Open: our cells smoke is slower than the reference (I/O on NFS?).

## Order

**A → B** gives the first complete ClearMap-equivalent study (per-sample
atlas-space results, then group comparisons).

**C** is needed before full brains run comfortably, and **F** makes many brains
practical. **E** is only the small tile-set item. **D**, **G** and **H** follow.

Within A, the registration work (A3) is the largest item and the main risk to
"similar results". A1, A2, A5 and A6 are mostly wiring and can proceed in
parallel with it.
