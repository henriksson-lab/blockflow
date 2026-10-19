# cell-density

Detect cells (nuclei, cFos-positive cells) in one channel of a whole-organ
light-sheet OME-Zarr image, and write them as NGFF object tables attached to
that image. Given a registration into an atlas (`--registration`, `--atlas`)
it maps the cells into atlas voxels and labels, filters, counts and
voxelises them there; given instead an annotation already registered into the
sample's frame (`--annotation`) it does the same in the sample's frame. Either
way it adds region, hemisphere and distance-to-surface columns, drops cells in
a surface crust, counts cells per region, and renders a density map.

Originally developed as part of a ClearMap reimplementation.

## Pipeline

One planned run (`detect::CellPlan`), all in the image's `[z, y, x]` order,
in voxels and raw intensity units:

| step | blockflow |
|---|---|
| background removal: the volume minus its opening by a flat 10 x 10 disk per plane, clamped at 0 | `ops::background::background_estimate`'s opening, with each pass by the crate's `extreme::ExtremeFilterOp` (running extremes over the whole buffer), `DifferenceCombine`; the opening runs in `uint16`, and the result is stored in the input's unsigned type when the default configuration allows (no band-pass, no h-maxima, window maxima) |
| optional band-pass (`--dog-sigma`, `--dog-sigma2`) | `ops::interest::BlobResponseOp`, clamped at 0 |
| optional prominence (`--h-max`) | `ops::reconstruct::HExtremaOp`, skipped at 0 |
| maxima: a window maximum of radius 2 (default) or the regional maxima | `extreme::ExtremeFilterOp` (separable for the box) or `ops::regional`; the mask is stored as `bool` |
| foreground gate: filtered value above `--threshold` (default 500, the reference's shipped cell-map configuration; 700 is only its code default) | `VoxelwiseMapOp::threshold`, `LogicCombine` |
| one seed per maximum (26-connected), numbered in position order | `ops::detect::LabelRegionsOp`, then the crate's `SeedRowsOp`: a barrier that merges the components across blocks and numbers their seeds once, and hands each block its own |
| seed image | `classical_centers::CenterSeedsOp` |
| basins: a seeded watershed of the negated filtered volume, inside the foreground, with a one-voxel line between basins, flooded per tile of `--watershed-tile` (default `64,128,128`) with a halo of `--watershed-halo` (default 16 voxels) from the global seeds | `ops::tiled_watershed::TiledWatershedOp`, in one phase with the negation and the crate's `DropEdgeBasinsOp`; `--watershed whole` keeps the old `ops::watershed::ScalableWatershedOp` barrier |
| `size` (basin voxels), `source` (raw maximum), `background` (background-removed maximum) | `ops::tabulate`, twice, each merge folded once in a barrier (`MergeTabulationOp::folded_once`) |
| size band `[50, 900)` and optional intensity band, as a `passes_filter` flag | rows |

The watershed tile is part of the answer, as the block is in the reference
implementation, which detects shapes per `z` slab of 50 to 100 planes
overlapping by 32 (16 planes either side of a seam) with whole planes in
`y, x`. Here every axis is cut, with a 16-voxel halo, about three diameters
of the largest cell in the size band. A seed's basin keeps the seed's
`label_id` across tile seams; a basin part whose seed is beyond the halo is
flooded from an edge seed and then cleared, so every row is a seed.

Then, outside the plan, with `--registration` (atlas space, through the shared
`blockflow_atlas_registration::atlas_space`, which vessel-graph uses too;
`atlas::annotate_registered` puts its lookups on the rows):

- **mapping**: each cell's level-0 voxel goes to the registration's sample
  grid and through `PointTransform` into atlas level-0 voxels (see Frames);
- **atlas columns**: region, hemisphere and distance to the surface
  (`distance_to_surface_voxels`, in atlas voxels, as the prepared atlas
  stores it) read on the prepared atlas at that position (the voxel nearest the continuous
  position, ties to even; `--atlas-rounding floor` takes the reference's
  `astype(int)` instead); optional collapse to `--level` (depth in the tree,
  root 0, the reference's level). Outside the atlas: region 0, hemisphere
  -1, and the distance of the nearest edge voxel;
- **crust**: `--crust-voxels N` is the reference's `distance_from_surface_px`
  (its `remove_crust`): the atlas's shipped distance field, in atlas voxels,
  must be strictly above `N`. `--crust-um C` keeps a cell when its distance
  times the atlas voxel's edge is strictly above `C`, so it needs cubic atlas
  voxels and is refused otherwise; at 25 µm `--crust-um 75` and
  `--crust-voxels 3` make the same decisions (`d * 25` and `N * 25` are
  exact). The tables' `filter` provenance records both crusts and
  `distance_to_surface_unit: "atlas voxel"`;
- **per-region counts**: as below, with volumes counted on the atlas
  annotation (`volume_voxels`, and `volume_mm3` = voxels × voxel volume), as
  the reference's `get_lateralised_volume_map`;
- **density**: on the atlas grid, a ball of `--density-radius-voxels`
  (default 5, the reference's CellMap default) per filtered cell, adding 1
  or, with `--density-weight source|background`, the cell's intensity (the
  reference's weighted voxelisation). This is cell-density's own planned,
  blockwise `density::density`, not the shared helper's whole-grid one; the
  cells pick their voxel by the same `--atlas-rounding`.

Or, with `--annotation` (sample frame):

- **atlas columns** (`atlas::annotate_cells`): region, hemisphere and distance
  to the surface per cell, via `atlas_regions::Annotator`, with the distance
  field from `atlas_regions::distance_to_surface` in micrometres; optional
  collapse to a hierarchy level;
- **crust** (`--crust-um`): a cell passes only if strictly deeper than this;
  a cell outside the annotation has no depth and fails;
- **per-region counts** (`collapse::collapse`): `ops::rows`' grouping phases
  over the filtered cells, keyed by `(region, hemisphere)`, with names from
  the tree and volumes and densities from the annotation;
- **density** (`density::density`): `ops::voxelize` with a ball of
  `--density-radius-um` per filtered cell, on a coarser lattice.

The defaults are one parameter point for 16-bit nuclear/cFos data at a few
micrometres per voxel. Everything is a flag; `--help` lists them.

## Frames (with `--registration`)

Three level-0 lattices, each with voxel centres at integers and
`physical = translation + index * scale` (NGFF):

1. the **detection image** (`--input`, e.g. a cFos tile at 6 × 1.625 ×
   1.625 µm);
2. the registration's **sample**, its first fixed image (`registration.json`
   → `fixed.level0_frame`; a chain's first step), e.g. the signal at 25 µm.
   `PointTransform::apply` maps *these* voxels;
3. the **atlas**, the registration's last moving image.

Detection voxel `i` goes to sample voxel
`j = (t_d + i * s_d + o - t_s) / s_s` per axis: `s_d, t_d` from the input's
level-0 metadata, `s_s, t_s` as the registration recorded the sample, and
`o` = `--sample-offset-um` (default 0). The scale is never entered by hand.
`o` is for a detection image whose own metadata does not place it in the
sample's physical frame, such as one tile of a mosaic whose converted tiles
all sit at the origin. Then `j` goes through the registration to atlas
voxels. The table keeps the detection coordinates (`z, y, x`, `*_um`) and
adds the atlas ones (`atlas_z, atlas_y, atlas_x`).

`--sample-offset-um` is in µm; a non-zero offset on an image whose unit is
known and not the micrometre is refused, as are an image and sample of
different units. A warning is printed when the image reaches more than a
sample voxel outside the sample's box.

The atlas must be the prepared atlas (`atlas-prepare`) the registration used
as its moving image: its shape must be the moving image's level-0 shape and
its voxel size the moving image's level-0 scale, or the run is refused, and
its reference is compared with the moving image voxel for voxel when the
registration's moving image is readable (`--skip-atlas-check` skips that).
`--atlas` defaults to the moving image when that is itself a prepared atlas.
For osf-cztq2 that is ABA 2017 25 µm, orientation `(1, 2, 3)`, `z 0:256`:
`/husky/henriksson/lightsheet-reference/atlas/ABA_25um_2017_1_2_3_z0-256.ome.zarr`.

## Input

- An OME-Zarr image with `z`, `y`, `x` axes (optionally `c`, `t`). Level 0 of
  `--channel` and `--time` is read, or a `--crop-start`/`--crop-shape`
  window of it. Any integer or float dtype; `uint16` keeps the background
  opening narrow.
- Optional `--registration DIR [--atlas ATLAS.ome.zarr]`: an
  `atlas-registration register` or `chain` directory and the prepared atlas
  (annotation, hemispheres, distance to the surface, region tree in the label
  properties). `--annotation` and `--registration` exclude each other.
- Optional `--annotation`: an integer Zarr array (a directory such as
  `.../labels/atlas/0`) **already in the sample's frame**, covering the same
  physical box as level 0 at any resolution. Registration is not this
  example's job (see `examples/atlas-registration`). A cell at image voxel
  `p` reads the annotation voxel nearest `(p + 0.5) * A / I - 0.5` per axis.
  `--hemispheres` is an array on the same grid; `--ontology` an Allen-style
  JSON-lines structure file; `--level` collapses regions onto that level.

## Run

```sh
cargo build --release -p blockflow-cell-density

target/release/cell-density --input image.ome.zarr --output copy.ome.zarr \
    [--crop-start z,y,x --crop-shape z,y,x] [--block 64,128,128] [--workers N] [--cache-mb 2048] \
    [--labels] [--threshold 500] [--size-min 50 --size-max 900] \
    [--watershed tiled|whole --watershed-tile 64,128,128 --watershed-halo 16,16,16] \
    [--annotation atlas/0 --hemispheres hemi/0 --ontology tree.jsonl \
     --crust-um 75 --all-regions] \
    [--density density.ome.zarr --density-radius-um 100 [--density-factor 8,16,16]]

# in atlas space
target/release/cell-density --input tile.zarr --output out/ \
    --registration registration/chain --atlas ABA_25um_2017_1_2_3_z0-256.ome.zarr \
    [--sample-offset-um=-9,1462.8125,2712.8125] [--level 6] [--crust-voxels 3] \
    [--atlas-rounding nearest|floor] [--all-regions] \
    [--density density.ome.zarr [--density-radius-voxels 5] [--density-weight count|source|background]]
```

`--output` is normally a copy of the input's metadata with its levels linked;
`--in-place` writes into the input. Existing outputs are refused without
`--overwrite`. `--labels` needs an uncropped run. The block size, worker count and
chunk cache (`--cache-mb`) do not change any result; the watershed tile and halo do (see Limitations). Stdout gets a JSON report (counts, size and intensity
quantiles, seconds).

Tests: `cargo test --release -p blockflow-cell-density`; the real-data and
timing tests are ignored (`-- --ignored`, see Validation).

## Output

| path | contents |
|---|---|
| `tables/<name>/` | every basin, with `passes_filter`; paired with `labels/<name>` when `--labels` |
| `tables/<name>/regions.csv` | with `--annotation` or `--registration`: one row per `(region, hemisphere)` over the filtered cells |
| `tables/<name>/regions.json` | how `regions.csv` is keyed: `region` is the label value of `labels.regions` (the atlas's `labels/annotation`, or the `--annotation` array), `hemisphere` that of `labels.hemispheres`; frame, level, voxel size |
| `tables/<name>-filtered/` | only the rows that pass the bands and the crust |
| `labels/<name>/` | with `--labels`: the basins as an NGFF `uint64` label image (`label value = label_id`) with the image's pyramid |
| `--density` | a one-level OME-Zarr `float32` image: filtered cells (or their summed weight) within the radius of each voxel; on the atlas grid and in its frame with `--registration` |

Default `<name>` is `cell-density`. Each table is a native
`ngff-object-table` with a `table.csv` beside it holding the same rows, and a
`cell_density` provenance attribute (command, input, crop, parameters, filter
rule, counts).

| column | dtype | meaning |
|---|---|---|
| `label_id` | u64 | the seed's rank in `[z, y, x]` order, from 1; independent of the block grid; the label value |
| `z`, `y`, `x` | f64 | the seed voxel (the rounded centroid of its maximum), level-0 voxels of the whole image (crop origin added) |
| `z_um`, `y_um`, `x_um` | f64 | `translation + index * scale` from level 0 |
| `size` | u64 | basin voxels |
| `source` | f64 | maximum of the raw input over the basin |
| `background` | f64 | maximum of the background-removed volume over the basin |
| `passes_filter` | u8 | 1 when the row passes the size and intensity bands and the crust |
| `region`, `hemisphere` | i64 | with `--annotation` or `--registration`; `0` and `-1` outside it |
| `distance_to_surface` | f64 | with `--annotation`: µm to the brain surface, computed from the annotation, `NaN` outside its grid |
| `distance_to_surface_voxels` | f64 | with `--registration` instead: atlas voxels to the brain surface, the prepared atlas's field (computed from its annotation when it has none, cubic voxels only), the nearest edge voxel's outside the atlas |
| `atlas_z`, `atlas_y`, `atlas_x` | f64 | with `--registration`: atlas level-0 voxels (continuous, centres at integers) |

`table.csv` also has `region_acronym` and `region_name` when there is a
region tree (the object table holds numbers only).

`regions.csv` has `region, hemisphere, acronym, name, cells, mean_size,
volume_voxels, volume_mm3, density_per_mm3`; with `--all-regions` every region of the tree
appears on every hemisphere seen, with zero cells where there are none.

## Validation

- **Ground truth** (`tests/cells_ground_truth.rs`): on eight separated
  synthetic objects every object is found once, in the voxel its centre
  falls in, and each basin is its object voxel for voxel. On sixteen objects
  over noise and a gradient, the regional maxima without a prominence return
  more than twice as many seeds as objects; `--h-max` at a quarter of the
  brightness leaves no spurious seed and keeps at least four fifths of the
  objects.
- **Decomposition** (`tests/cells_block_sweep.rs`, `tests/end_to_end.rs`):
  identical rows, label ids included, over six block grids with touching
  objects, for the default, the regional-plus-prominence and the band-pass
  variants; `uint16` input gives the same rows as its `f64` widening; two
  grids publish byte-identical tables.
- **Atlas** (`tests/cells_crust.rs`, `tests/collapse_regions.rs`): region,
  hemisphere and depth on a synthetic two-region ball; the crust filters,
  is strict, drops cells outside, commutes with the bands and does not depend
  on chunking; region counts, means, volumes and densities on constructed
  tables, independent of the cut and the row order.
- **Atlas space, synthetic** (`tests/cells_registered.rs`): a registration
  whose expected atlas coordinate is known in closed form (detection 4 × 2 ×
  2 µm at the origin; sample 8 µm translated by 4 µm; a `--sample-offset-um`;
  an elastix affine into a 16³ atlas of 10 µm voxels): every cell's
  `atlas_*` matches `(A (i s_d + o) + t) / 10` to 1e-9; region, hemisphere
  and depth (in atlas voxels) are the atlas's at that voxel, 0/-1 and the
  edge depth outside; the crust in voxels is strict, and the crust in µm is
  the voxel depth times the voxel edge, strictly above it; region voxel counts and volumes equal the
  atlas's; the density sits on the atlas grid and its frame and sums the
  intensity weights; `--level` collapses a child; a two-step chain composes
  and floor rounding picks the floor voxel; an atlas of another shape, or
  whose reference is not the moving image, is refused.
- **Atlas space, real** (`tests/cells_registered_real.rs`, ignored;
  `reference-atlas/compare_atlas.py` for the reference side). osf-cztq2
  cFos `tile_01x02`, window `512 × 640 × 640` from `256,192,192` (210 M
  voxels), defaults (threshold 500): 329 459 basins, **21 997 cells in the
  size band**, through the native chain signal mosaic → autofluorescence →
  atlas (`registration/ours/chain`). The chain's signal is a nominal-grid
  mosaic of the cFos tiles at 25 µm, so the tile is placed at its mosaic
  offset: tile `(r, c) = (1, 2)` starts at mosaic voxel
  `round([0, r · 1471, c · 1368] / 25) = [0, 59, 109]`, holding the tile's
  level 4 (translation `[9, 12.1875, 12.1875]` µm) resampled to 25 µm, so
  `--sample-offset-um = 25 · start − t4 = -9,1462.8125,2712.8125`. (The
  mosaic's resample stretches a tile by under 1%, at most half a mosaic voxel
  at its far edge; both sides below use the same sample coordinates.) The same
  cells through the reference's own `Elastix.transform_points` with its
  elastix transforms (`registration/clearmap/`, signal → auto → atlas) and
  labelled its way (truncation):
  - atlas positions 25 µm apart (median; 95th percentile 49 µm), the same as
    the two registrations' grid points in that box (21 µm median, the same
    mean offset), so the difference is the registration's;
  - **same finest region for 90.3% of cells**, same major region 97.0%,
    same hemisphere 100%; for scale, the reference's own coordinates
    labelled by nearest voxel instead of truncation keep 90.2%, so the
    finest labels move with any half-voxel change;
  - per-region counts: summed |difference| 6.1% of cells over 61 regions
    (5.6% with `--atlas-rounding floor`); major regions Isocortex −0.8%,
    STR +3.4%, CTXsp −0.2%, fiber tracts +4.9%, VS 0%, OLF −14% (344 vs
    401 cells);
  - crust at 3 voxels: the same keep/drop decision for 99.9% of cells;
  - region volumes: identical voxel counts for all 60 rows;
  - density (radius 5) against the reference's `Voxelization.voxelize`:
    correlation 0.991, totals within 0.1%.
  - Time: the whole run 101 s wall, 16 GB peak (detection 88-90 s); the atlas
    part (atlas and reference check, mapping of 329 459 cells, lookups,
    region volumes, density) adds about 5.5 s and 2.5 GB, measured on a small
    crop with and without `--registration`; the reference's two transformix
    calls took 1.2 s for the 21 997 filtered cells (it maps only those).
- **Real crop** (ignored; `CELL_DENSITY_TILE` overrides the default
  `/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles/tile_05x06.zarr`):
  Brain-39L tile 05x06, `z 1792..2048, y 640..1024, x 16..400` (37.7 M
  voxels, `uint16`). Defaults at the earlier foreground (`--threshold 700`;
  the numbers in this and the next items were measured at 700; at the
  current default of 500 the crop gives 54 526 basins, 2 434 in the band):
  **34 251 basins, 1 394 in the size band**,
  byte-identical tables at 40 workers with `--block 64,128,128` and at one
  worker with `--block 32,96,128`. Against the recorded output of the
  original Python implementation on the same crop (34 442 basins): 96% of
  our basins sit on one of its voxels, and 89% of those have the same size;
  of its 1 282 in-band cells we also find, the median size differs by 1.8%
  (p90 34%). Most of the size differences come from the background: here it
  is a true opening (the maximum over the reflected element), where the
  original took the maximum over the element itself, which slides the
  estimate by a voxel on even element sizes. With that composition
  substituted the raw counts agree to 34 441 of 34 442.
- **Tiled against whole-volume watershed**
  (`tests/cells_tiled_watershed.rs`). Synthetic, 300 touching objects in
  48 x 96 x 80 with 16 x 32 x 32 tiles: rows and basins identical over five
  block grids at one and four workers; one tile over the volume is the
  whole-volume flood exactly; at a 6/8-voxel halo 372 of 379 sizes agree and
  every voxel that differs was cleared (an edge-seed part), none given to
  another seed; at a 15-voxel halo all agree. On the real crop (ignored),
  against one flood over the whole crop: at the default (64 x 128 x 128,
  halo 16) 34 248 of 34 251 sizes agree, 1 392 of 1 394 in-band cells (worst
  2 voxels; the worst overall, 695, is one large out-of-band basin); 697
  voxels cleared and 2 relabelled. A halo of 8 gives 34 233 (in band 1 378,
  worst 170), a halo of 32 gives 34 246 (worst 13); tiles of 32 x 64 x 64
  match the default. The old barrier (`--watershed whole`) is further from
  both: 33 753 of 34 251 sizes and 973 of 1 393 in-band cells agree with the
  tiled answer, because foreground components with a bounding box over 16 M
  voxels leave its exact flood for a blockwise minimax fallback (with the
  exact flood forced it agrees with the one-tile answer on 34 250 of 34 251).
  That also puts the old path further from the original: median in-band
  size difference 3.2% (p90 58%) against the tiled 1.8% (34%).
- **Larger regions** (cFos sample `osf-cztq2/cFos/tile_01x02.zarr`, CLI,
  `/usr/bin/time`, 40 threads on a machine at load 40-80 from other jobs, so
  the times are loose): a `512 x 640 x 640` window from `256,192,192` (210 M
  voxels, 5.6x the crop) gives 184 706 basins either way, 184 704 of the
  same size (worst 12), 12 462 in band with no flag changed; tiled 108 s and
  10.9 GB peak, whole 95 s and 12.4 GB. The whole tile, `1030 x 1007 x 1004`
  (1.04 G voxels, 27x): 648 058 basins either way; tiled **28.7 min, 41.4 GB
  peak**, whole 34.0 min, 60.5 GB peak; 630 112 sizes and 52 454 of 61 748
  in-band cells agree, 2 372 flags differ, the large-component fallback of
  the old path again. Doubling the halo to 32 moves 16 of the 648 058 sizes
  (worst 10 voxels) and no flag, so the default halo has converged
  (44 GB peak, 49 min on the loaded machine).
- **Time and memory**, against the reference on the same machine, run back
  to back by `reference-clearmap/run.sh` (2026-10-19, 40 threads against its
  10 processes, load average 5-11; outputs in
  `/husky/henriksson/lightsheet-reference/cells-perf/`):

  | | wall | user + sys | peak (tree PSS) |
  |---|---|---|---|
  | crop, reference | 7.2 s | 64 s | 2.2 GB |
  | crop, ours | 6.4 s | 37 s | 2.0 GB |
  | whole tile, reference | 81 s | 779 s | 7.0 GB |
  | whole tile, ours | 99 s | 807 s | 5.6 GB |
  | whole tile, ours before the changes below | 1 767 s | 11 282 s | 44 GB |

  The tables, crop and whole tile, are byte-identical to those of the
  binary before the changes (`ours-before/`; only the provenance's date and
  paths differ). What the whole tile spent its time
  and memory on, from `tests/cells_op_profile.rs` (per-phase wall, CPU and
  resident set), and what changed:
  - **The two tabulation merges, 9 260 s of CPU and most of the memory.**
    Every block folded every block's partials (1 088 blocks, a million
    labels), so the work went with the square of the block count, and each
    worker held an accumulator for every label. They are now folded once,
    in a barrier (`ops::tabulate::MergeTabulationOp::folded_once`): 8 s.
  - **Region points, 478 s.** `ops::detect::RegionPointsOp` hands every
    block the whole component list to find its own; `SeedRowsOp` now merges
    the components and numbers the seeds in one barrier and gives each block
    its slice: 2 s.
  - **The chunk cache**, by default an eighth of the machine (23 GB here),
    filled up; `--cache-mb` (default 2 GiB) holds the halo re-reads (at
    1 GiB the watershed phase took four times the CPU). And glibc's eight
    malloc arenas per core kept about 3 GB of freed block buffers resident;
    the binary limits them to two (`MALLOC_ARENA_MAX` overrides), at no cost
    in time.
  - **The rank filters' boundary path**: `RankFilterOp` takes running
    extremes only where the whole element fits in the buffer and gathers the
    element tap by tap in every block's halo, which was 40% of the remaining
    CPU. `extreme::ExtremeFilterOp` slides every row (and, for a box, every
    axis) over the whole buffer: the opening went from 530 s to 165 s and
    the window maxima from 420 s to 170 s.
  - **The watershed's edge seeds**: each tile took the regional minima of
    its whole buffer to find the few that touch a cut face; the face search
    in `ops::tiled_watershed` walks only plateaux reached from a face. The
    watershed op went from 492 s to 69 s.
  - **Intermediate bytes**: the background-removed volume is stored as the
    input's `uint16` and the maxima as `bool`, not `f64`, which matters
    because the work directory sits beside the output on NFS here (with it
    on a local disk the whole tile took 75 s). The label scan of the
    tabulation reads integer labels as stored.

  What is left is reading and writing: the phases that read the labels
  (`uint32`, gzip) or the mask run at 5-12 cores.

## Limitations

- **The watershed's answer depends on its tile.** Each tile is flooded with
  its halo from the global seeds it holds. Where a basin reaches further
  from its seed than the halo, the far part is cleared rather than given to
  a neighbouring seed, so that cell comes out smaller (and a `--labels`
  image has a hole there). At the default halo this touched three of
  34 251 basins on the real crop. The block grid and the worker count do not
  change the answer; `--watershed-tile` and `--watershed-halo` do, and are
  recorded in the provenance.
- **`--watershed whole`** is a barrier over the whole volume (60 GB on a
  whole tile) and runs that phase on one thread; it is kept for comparison.
- One tile at a time; cells in tile overlaps are not merged across tiles.
- With `--annotation`, it must already be in the sample's frame. Either way
  the region volumes, distance field and lookups are computed in memory on
  the annotation's grid (cheap at atlas resolution; the prepared atlas with
  its distance field is about 2.5 GB at 25 µm).
- The detection image must be placed in the registration's sample frame by
  its metadata or `--sample-offset-um`; a pure offset cannot express a
  mosaic whose tiles were resampled with a stretch (under 1% here).
- Atlas lookups are nearest-voxel, no interpolation; the finest labels are
  sensitive to half a voxel (see Validation), the major regions much less.
- All parameters are in voxels and raw counts: on strongly anisotropic data
  the window of five spans a different physical extent in `z`, and
  `--threshold` and the size band need adjusting for other stains or gains.
- The density counts cells per ball and loses mass within one radius of a
  face; it is not normalised to cells per volume.
