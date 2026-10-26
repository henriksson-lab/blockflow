# tile-stitching

Stitches overlapping light-sheet tiles into one OME-Zarr volume: a rigid
alignment of each pair of neighbouring tiles, a per-slice ("wobbly") alignment
that follows the in-plane drift of the tiles along the stack, and a
weighted-mean fusion. Originally developed as part of a ClearMap
reimplementation.

## Pipeline

```text
tiles + grid + nominal overlap
  layout   nominal positions from the grid                     src/layout.rs
  rigid    per neighbouring pair: max-project the facing ends  src/rigid.rs
           (depth = nominal overlap), mean squared difference
           over every shift via blockflow::ops::fft; each pair
           reads its two facing slabs, or with them the boxes
           the per-slice step will cut, kept (src/tiles.rs)
  layout   least squares over the pair graph -> one position   src/layout.rs
           per tile, lowered to the origin
  wobbly   per pair and slice of the overlap: landscape over   src/wobbly.rs
           in-plane shifts (ops::fft), local minima, linked
           across slices by optimal assignment; unmeasured
           slices interpolated
  place    per slice: least squares within each connected      src/place.rs
           component; then one shift per component over the
           whole stack so tiles move as little as possible
           between slices; smoothing
  fuse     per output voxel: weighted mean of the covering     src/fuse.rs
           tiles, weight = 1 + distance to the tile's nearest
           in-plane face; planned slab by slab, chunks in      src/planned.rs
           parallel, over views of resident tiles or boxes
           read from the tiles on disk (the CLI, after
           alignment or with --layout-from)
  write    multiscale OME-Zarr, slab by slab, chunk-parallel   blockflow::zarr_env::MultiscaleWriter
```

The rigid step corrects only the axes transverse to the tiling axis (the
tiling axis is projected away); the per-slice step corrects the in-plane axes,
including the tiling axis. Volumes are `(z, y, x)`, tiled in x and y, wobbling
along z; every module takes the axes as parameters.

The library API is `tile_stitching::stitch(tiles, layout, params)`, with the
steps also available one by one (`align`, `align_rigid`, `measure_wobbly`,
`place_wobbly`, `wobbly::measure_layout`, `place::place_layout`,
`fuse::resident`). For tiles left on disk it is `align_stored` (on
`tiles::StoredTiles`) and `planned::fuse_to_pyramid`, which the CLI runs. Both
paths run the same code under the memory cap (see [Memory](#memory)): the
alignment reads boxes from `tiles::TileBoxes` (views of resident tiles, or
boxes read from stored ones) and the fusion is the planned one over the same
boxes, so `fuse::resident` holds only the volume it returns and one slab
beside the tiles it was given.

## Input

One OME-Zarr image per tile (any element type; all tiles the same), described
by a JSON tile-set file. Paths are relative to the file.

```json
{
  "grid_axes": ["x", "y"],
  "overlap": [66, 212],
  "tiles": [
    { "path": "tile_03x04.zarr", "grid": [0, 0] },
    { "path": "tile_03x05.zarr", "grid": [1, 0] },
    { "path": "tile_04x04.zarr", "grid": [0, 1] }
  ]
}
```

`grid[d]` is the tile's index along `grid_axes[d]`; `overlap[d]` is the
nominal overlap in voxels along that axis, and is also the depth of the rigid
step's projection. Instead of `overlap`, every tile may give
`"position": [z, y, x]`, its nominal position in voxels. Which tiles are
aligned (`"neighbours"`):

- `"grid"`: tiles one step apart on the grid (the default when every tile has
  a `grid`);
- `"boxes"`: tiles whose boxes, at their positions, overlap in the plane
  (needs a `position` for every tile; `grid` may then be left out, and is the
  default when some tile has none). Two tiles pair along x (or y) when their
  boxes overlap along it and their offset across it is under half a tile and
  relatively smaller, so a regular grid gives its row and column seams and no
  diagonals, a jittered or partial grid the same, and a staggered row pairs
  with both tiles it straddles (`TileLayout::from_boxes`).

### From a directory of converted tiles (`--tile-dir`)

`--tile-dir DIR` builds the tile set from the tiles themselves
(`src/tile_dir.rs`) and writes it, with a record of where each number came
from, to `<out>.tiles.json` (which `--tiles` reads back):

- **Tiles.** Every OME-Zarr image in `DIR` whose name carries a grid index,
  the last `<row> x <column>` in it (`tile_03x04.zarr`,
  `sample[03 x 04]_c0.ome.zarr`), is a tile; other images (an
  `autofluorescence.zarr` beside the tiles) are skipped. When no name carries
  one, every image in `DIR` is a tile. The row runs along y and the column
  along x. `--rows A:B` and `--columns A:B` (half-open, as named) select a
  sub-grid by the names; grid indices are rebased to 0.
- **Positions and neighbours**, the first of:
  1. *physical translations*: when the tiles' level-0
     `coordinateTransformations` carry translations that differ between tiles,
     each tile's position is its translation relative to the first tile's, in
     voxels of `--level`, and the neighbours are found from the boxes
     (`"neighbours": "boxes"`). Grid indices are not needed, and `--overlap`
     is ignored (with a note). Two tiles at the same translation are refused:
     one of them is misplaced (osf-cztq2's vendor metadata repeats row 00 as
     row 07, which `img2omezarr set-positions` refuses already);
  2. the grid from the names and `--overlap Y,X` (level-0 voxels, one value
     for both allowed), scaled to `--level`, with grid neighbours.

  With neither, the directory is refused.

Only standard OME-Zarr metadata and the file name are read. Nothing about the
instrument or the format the tiles were converted from is used; that is the
converter's job (`img2omezarr`, whose `set-positions` writes the translations
into existing conversions: from a table, a grid, or the originals' LaVision
stage positions).

The conversions under `/husky/otherdataset/teresa/clearmap_em` themselves carry
no translations, so read directly they need `--overlap` (osf-cztq2: `120,120`,
measured from the whole-sample stitch). Their metadata-only mirrors under
`/husky/henriksson/lightsheet-reference/positions/` carry them: osf-sa3x8's
nine tiles the LaVision stage positions (`positions/osf-sa3x8/tiles`, placed
and paired with no `--overlap`), osf-cztq2's 40 a grid of 887 x 884 voxels
(`positions/osf-cztq2`; its LaVision positions are refused, row 07 repeating
row 00).

### Fused frame

The fused image's level-0 frame is the first tile's: fused voxel `u` is at
`t_0 + s (u - (a_0 - o) + c)`, with `t_0`, `s` the first tile's translation
and voxel size at the stitched level, `a_0` its anchor (the fused position of
its plane 0 along z, its median per-slice position in-plane), `o` the
placement's origin and `c` the `--z-range` start (`src/relayout.rs`). Tiles
at the origin (all current conversions) therefore give a fused image whose
first tile keeps its own frame, and tiles with distinct translations keep
them. The positions file records `level`, `voxel_size`, `unit`, `anchor` and
`translation`. Registration (`atlas-registration`, which starts from the
images' geometric centres, their physical frames, or a given translation:
`--initialise`, `--initial-translation`) and the detectors (`cell-density`, `vessel-graph`,
which read each image's level-0 scale and translation) take the frame as it
is.

### Reusing a layout (`--layout-from`)

`--layout-from <out>.positions.json` skips the alignment and fuses at the
stored positions, carried to `--level`: in-plane positions are scaled by the
ratio of the voxel sizes (exact from a coarser level to a finer one by an
integer factor). The per-slice placement is per plane, so z must keep its
voxel size or, for a pyramid that halves z too (osf-sa3x8's), be finer by a
whole factor: each stored plane then stands for the planes it covers, at
that factor times its z position plus 0, 1, ... and its in-plane position.
`--region Z0:Z1,Y0:Y1,X0:X1`
(fused voxels of `--level`) fuses only that box, reading only the tiles and
planes that reach into it; its frame is the box's place in the fused frame,
so it lines up with a whole-sample fusion at another level. This is how a
whole sample is stitched without holding it at full resolution: align and
fuse at level 1 (or 2) for registration, then fuse level-0 boxes for
detection. Writes `<out>.fuse.json` (layout, level, region, translation,
seconds).

This fusion is planned and out of core (`src/planned.rs`). Before anything is
read, each tile's box for the region is known: the planes usable there and,
per plane, the in-plane part inside the region at that plane's own
position. The region is cut into slabs of whole output chunk rows along the
axis the output's chunks cut most often (`pyramid::sweep_axis`; for
osf-cztq2 at level 2, whose chunks are `149 x 76 x 143`, that is y). Each
tile is held as one buffer covering what the current slab needs. The next
slab keeps the rows still needed and reads the rest up to the tile's next
stored chunk boundary, so every stored tile chunk is decoded once; a tile's
buffer is dropped once no later slab needs it. The next
slab's reads overlap the current slab's fusion and writing. A slab's chunks
are fused in parallel by the same kernel as the resident fusion, which reads
the same tile voxels in the same order, so the output is identical. Each
finished slab goes straight to the pyramid writer (`src/pyramid.rs`). As soon
as a level has a whole row of chunks, the writer encodes and stores it, chunk
by chunk in parallel. As soon as the windows for the next level's rows are
present, it computes those rows. No level is ever read back. The writer is
core's (`blockflow::zarr_env::MultiscaleWriter`, the one behind
`ZarrEnvironment::build_multiscale`), so the image is the one
`build_multiscale` writes for the resident volume.

There is no automatic choice between this and a resident fusion, because the
planned path is the faster one at every size measured, a 512^3-voxel box
included (see Validation). `StoredLayout::fuse_resident` (read every touching
plane whole, fuse on one thread) is kept as the reference the tests compare
against.

## Run

```sh
cargo build --release -p blockflow-tile-stitching
target/release/tile-stitching --tiles tiles.json --out fused.zarr --z-range 1200:1320

# a whole sample: align and fuse at level 1, then a level-0 box at those positions
target/release/tile-stitching --tile-dir cFos/ --overlap 120,120 --level 1 \
    --max-shift 8,15 --rigid-max-shift 15,15 --positions-only --out align
target/release/tile-stitching --layout-from align.positions.json --level 2 --out fused_l2.ome.zarr
target/release/tile-stitching --layout-from align.positions.json --level 0 \
    --region 256:768,1100:1740,2000:2640 --out box.ome.zarr
```

Search ranges are in voxels of `--level`, so halve the reference's for
level 1.

Options: `--tiles FILE` | `--tile-dir DIR [--overlap Y,X] [--rows A:B]
[--columns A:B]` | `--layout-from POSITIONS [--region Z0:Z1,Y0:Y1,X0:X1]`,
`--positions-only` (align and write the positions file, no fusion),
`--channel`, `--level` (pyramid level of the tiles), `--z-range
START:STOP`, `--max-shift Y,X` (per-slice search, default `15,30`),
`--rigid-max-shift Y,X` (rigid in-plane search, default `30,30`),
`--max-shift-z` (rigid search along z, default 20), `--slice-step` (measure
every N-th slice, default 3), `--rigid-only`, `--fuse
interpolation|mean|max|min`, `--out-levels`, `--positions`, `--memory-mb`
(see [Memory](#memory)). Both in-plane range options also take one value for
both axes; ranges are in voxels either side.

### Memory

The memory cap is `--memory-mb`, else `$BLOCKFLOW_MEMORY_CAP`, else the
cgroup's limit, else a share of the machine (`blockflow::budget`; the first
stderr line says which). The CLI never holds a tile whole. The alignment
reads boxes (`src/tiles.rs`): a rigid pair its two facing slabs, a per-slice
pair the two boxes its overlap cuts span, in the tiles' own element type,
widening one plane at a time while it measures. The two overlap: for narrow
tiles (osf-sa3x8 is 202 voxels across x) the cuts take most of the slabs'
stored chunks. So the rigid step reads, for each pair it can, the bounding box
of its slabs and of the cuts the per-slice step will take at any rigid
position within its search range (plus two voxels; a pair placed further is
read again), where that decodes fewer stored chunks than reading the two
apart, projects its slabs from it, and keeps it, leased from the budget, until
the per-slice step has measured that pair. What is kept is at most three
quarters of what the per-slice step's boxes take with a pair on every worker,
and half the cap. A rigid pair that keeps nothing reads its slabs a few stored
chunk rows at a time. Pairs run in parallel, each
holding a lease of what it will hold from the process budget, so as many run
at once as the cap leaves room for (and never more than holding every tile
would take); one pair is the least a run can hold. The fusion, after the
alignment or with `--layout-from`, is the planned one (below): its slabs are
sized so that a slab's tile buffers and fused rows take at most half of what
the cap leaves, its tile buffers lease from the budget while held, and the
pyramid writer encodes as many chunks at once as the cap leaves room for.
Under a binding cap the allocator's mmap threshold is fixed at 4 MiB, so the
large buffers it frees are handed back rather than kept in its arenas. The
results do not depend on the cap.

The search defaults are the reference's shipped ones (ClearMap 3.1's
`config/defaults/v3.1/stitching.yml`): rigid +-30 in x and y and +-20 in z;
per slice +-30 in x and +-15 in y. The library's `MeasureParams::default()`
and `RigidParams::default()` match.

## Output

- `fused.zarr`: an OME-Zarr 0.5 multiscale image `(z, y, x)` of the tiles'
  element type, with the first tile's physical voxel size and unit. Voxels no
  tile covers are zero.
- `fused.zarr.tiles.json` (with `--tile-dir`): the derived tile set and its
  provenance (`derived`: per tile the name's grid index, level-0 scale and
  translation).
- `fused.zarr.positions.json`: the level, voxel size, unit, anchor and
  frame translation (see [Fused frame](#fused-frame)); per tile its grid index, shape, nominal and
  rigid position, and (unless `--rigid-only`) its position and status in every
  slice; per alignment the rigid shift relative to the nominal layout, its
  quality, and how many slices were interpolated or invalidated. Positions
  are in voxels of the fused volume's frame before `origin` is subtracted;
  `z_offset` is the start of `--z-range`. Each alignment also carries its
  per-slice measurement (`measured`: `start` slice, in-plane `[y, x]`
  displacement of `post` relative to `pre` per slice, and status), before
  smoothing and placement.

## Validation

`cargo test --release -p blockflow-tile-stitching` (all synthetic):

- Tiles cropped from one synthetic scene at known offsets: a pair offset by
  `(2, 3, -4)` voxels from the nominal grid is recovered exactly (z and y by the
  rigid step, x by the per-slice step) and the fused volume equals the scene
  voxel for voxel; a per-slice drift along z is followed plane by plane; a
  2 x 2 grid with a different offset per tile is placed exactly; the same
  through OME-Zarr files on disk and back (`tests/stitch_composition.rs`).
- Rigid correlation: planted shifts, clipping and normalisation where they
  decide the answer, refusal of a dark pair, least squares versus tree walk on
  an inconsistent cycle, decomposition of the max projection
  (`tests/stitch_rigid.rs`).
- Per-slice measurement and placement: planted drifts under both shift
  methods, the landscape against a direct padding-free walk, the resolution
  and isolation rules (`tests/stitch_wobbly.rs`, `tests/stitch_measure_layout.rs`).
- Tile directories and reused layouts (`tests/stitch_tile_dir.rs`): grid
  indices from names; a two-tile directory without translations is refused
  without `--overlap`, and with it gives the hand-written set and recovers the
  planted offset; the overlap is scaled to the level;
  row/column selection; distinct translations give the nominal positions and
  the fused frame keeps the first tile's; neighbours from boxes are the grid
  neighbours on a regular, jittered or partial grid, with no diagonals, and
  pair a staggered row with both tiles it straddles; tiles without grid names
  placed by translations alone stitch exactly as the same tiles named and
  given the overlap; duplicated translations are refused; a positions file read back fuses
  the same volume, any box equals that crop, and level-1 positions carried to
  level 0 put every tile voxel where the scene has it, in the frame the
  level-1 volume describes, also from a level that halves z (each plane
  standing for two), and a z ratio that is not whole is refused; the CLI end to end (`--tile-dir`, then
  `--layout-from --region`, frames compared).
- Fusion: the weighted mean against its definition on a three-tile plane,
  the other methods, and identical output across block lattices that cut the
  seam (`tests/stitch_fuse.rs`).

Real data (`--ignored`; tiles from `TILE_STITCHING_TILES`, default the OSF
`sa3x8` conversion under `/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles`),
release build, 2026-10-18:

| run | result |
|---|---|
| tiles `04x05`/`04x06`, z 1200..1440 (240 x 1304 x 404 each), rigid | displacement `(1, 0, 338)`, quality finite |
| same pair, per-slice measurement | 239 overlapping slices, 80 measured, 80 aligned, 1.3 s |
| same pair, end to end | fused `241 x 1304 x 762` in 7.4 s |
| CLI, 3 x 3 grid, z 1200..1320 | fused `124 x 3504 x 1127` in 32 s, 2.8 GB peak RSS |
| 2026-10-19: `--tile-dir` on the LaVision-positioned mirror (`positions/osf-sa3x8/tiles`), no `--overlap`, level 1, whole stacks, `--positions-only` | 9 tiles placed from their translations, 12 pairs from their boxes (the grid's), 77 s, 4.8 GB peak RSS; then the whole fusion at level 1, `1686 x 1839 x 570`, 188 s, 11 GB |

On that run the stitched tile positions (per-slice medians) against the stage
positions: y pitch the stage's (fitted scale 1.001), x pitch 2.7% shorter
(0.973; osf-cztq2's vendor pitch is 2% long on both axes), residuals with the
scale fitted 2.2 µm (y) and 6.7 µm (x) at most.

Planned fusion and the slab-by-slab writer (`tests/stitch_planned.rs`):
over a wobbly 2 x 3 grid with unusable planes and small stored chunks, the
planned fusion equals the resident one for every method, box, sweep axis,
slab size and chunk shape. A thin box reads under a quarter of the tile
voxels the whole volume reads. The planned image, and `write_volume_at`
fed any slab sizes along any axis, are the same files byte for byte as
`build_multiscale` writes (u8, u16, f32 and bool; prime extents; one level
only).

Measured 2026-10-19, before (the resident fusion and `build_multiscale`)
against after, on the same machine (40 cores, shared, load average 45-95
throughout). In every case, every file of the output is byte-identical:

| run | wall | user+sys | peak RSS |
|---|---|---|---|
| osf-cztq2, 40 tiles, level 2 whole (`1043 x 1811 x 1142`) | 809 s -> 50 s | 798 s -> 342 s | 13.9 -> 5.3 GB |
| osf-cztq2, level-0 box `512 x 640 x 640` | 39.5 s -> 9.2 s | 33.9 s -> 28.1 s | 2.33 -> 1.43 GB |
| osf-sa3x8, 9 tiles, level 1 whole (`1686 x 1839 x 570`) | 313 s -> 79 s | 405 s -> 278 s | 11.0 -> 2.5 GB |
| osf-sa3x8, level-0 box `512 x 512 x 256`, 1 level | 22.3 s -> 3.9 s | 21.0 s -> 11.5 s | 2.88 -> 0.87 GB |
| osf-sa3x8 `--tile-dir`, level 1, z 600..840, align + resident fuse | 16.4 s -> 15.3 s | 70.6 s -> 43.7 s | 1.39 -> 1.39 GB |

Against ClearMap 3.1 on a 3 x 3 grid (`reference-clearmap/`, both at the
shipped search ranges, the reference with two of its defects corrected): rigid
positions identical; per-slice measurements identical in 919 of 921 aligned
slices; seam offsets within a voxel in 97% (y) and 93% (x) of planes, median
0; fused Pearson r 0.99.

Measured 2026-10-22 (machine load 10-45, so wall times are approximate):
before is the CLI that held every tile, and after the alignment the fused
volume. Positions are byte for byte those of before in every run, and every
fused image has the same voxels in every level (its chunks are
`chunk_for_extent`'s now, see Limitations).

| run | wall | user+sys | peak RSS |
|---|---|---|---|
| osf-cztq2, 40 tiles, level 1, `--positions-only` | 254 -> 22 s | 388 -> 374-382 s | 21.3 -> 6.8-7.7 GB |
| the same, `--memory-mb 4096` | 28 s | 364 s | 3.2 GB |
| osf-sa3x8, 9 tiles, level 1, `--positions-only` | 28-35 -> 10 s | 66-70 -> 79-82 s | 4.8 -> 2.1-2.4 GB |
| osf-sa3x8, level 1, align and fuse | 97-130 -> 23 s | 343-363 -> 308 s | 8.2 -> 3.2 GB |
| the same, `--memory-mb 2048` (and under a 2048 MiB cgroup) | 25-26 s | 332-337 s | 1.9 GB |
| osf-cztq2, level 2, align and fuse | 83 -> 47 s | 419 -> 426 s | 9.9 -> 5.0 GB |
| osf-cztq2, level 2, `--layout-from` | 49 -> 23 s | 348 -> 303 s | 5.1 -> 3.0 GB |
| the same, `--memory-mb 2048` | 28 s | 354 s | 1.2 GB |
| osf-sa3x8, level 1, `--layout-from` | 39 -> 23 s | 322 -> 224 s | 2.6 -> 3.1 GB |

Measured 2026-10-23 (load 5-35), alternating runs: before is HEAD, round 3
the table above, after adds the shared reads, the pieced rigid slabs, one-row
fused slabs fused in place, and the writer's held-row and resident-volume
changes. Positions are byte for byte those of before; fused images are byte
for byte those of round 3, and the same voxels as before.

| run | wall (before / round 3 / after) | user+sys | peak RSS |
|---|---|---|---|
| osf-sa3x8, level 1, `--positions-only` | 25 / 6.0 / 4.1 s | 75 / 84 / 64 s | 4.8 / 2.2-2.4 / 1.9-2.0 GB |
| osf-cztq2, level 1, `--positions-only` | 183 / 20-23 / 17-19 s | 394 / 365-376 / 283-287 s | 21.3 / 7.3-8.0 / 6.5-7.0 GB |
| the same, `--memory-mb 4096` | - / 26-28 / 22-24 s | - / 378 / 321-331 s | - / 3.2-3.3 / 2.8 GB |
| osf-sa3x8, level 1, `--layout-from` | 31-61 / 16-19 / 14-16 s | 338 / 226 / 223 s | 2.6-2.9 / 3.1 / 1.9 GB |
| osf-sa3x8, level 1, align and fuse | 54 / 22 / 18 s | 368 / 309 / 288 s | 8.2 / 3.2 / 2.0 GB |
| the same, `--memory-mb 2048` | - / 24-25 / 19 s | - / 337 / 305 s | - / 1.9 / 1.6 GB |
| osf-cztq2, level 2, `--layout-from` | 28-34 / 22 / 20 s | 356 / 305 / 293 s | 5.1 / 3.0 / 2.3 GB |
| the same, `--memory-mb 2048` | - / 27 / 23-25 s | - / 357 / 343 s | - / 1.22 / 1.18 GB |
| osf-cztq2, level 2, align and fuse | 66 / 31-32 / 27 s | 407 / 426 / 403 s | 9.9 / 4.9 / 4.1-4.2 GB |

The osf-sa3x8 fusion had a higher peak in round 3 because a slab was two
rows of its 141-plane output chunks. It is one row now: a slab holds every
tile's rows for it, so one row is half the tile buffers (two rows held 3.0
GB).

## Limitations

- A per-slice pair holds both of its overlap boxes, whole, in the tiles'
  element type while it is measured (the stack's mean and norm need every
  plane twice), so one pair is the least the alignment holds: about 3 GB for
  osf-cztq2 at level 1. A cap under that runs one pair at a time over it.
- The planned fusion is planned here, not through a `blockflow` plan. A
  phase's supplied inputs must share the plan's volume and lattice, and each
  tile has its own shape and sits at its own translation in every plane.
- The output's chunks are `zarr_env::chunk_for_extent`'s (2026-10-22): a
  prime extent is halved rather than cut to one row, so osf-cztq2 at level 2
  is chunked `149 x 76 x 143` instead of `1043 x 1 x 1142`. The voxels are
  those of before; the chunk files are not. A fused slab is at least one row
  of output chunks (as few rows as give every worker a chunk to fuse).
- The rigid step does not correct the tiling axis; that is left to the
  per-slice step, so `--rigid-only` keeps the nominal spacing along it.
- The whole-stack placement is a dense solve with one unknown per tile cluster
  per slice; long stacks of many tiles make it slow.
