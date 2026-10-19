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
           over every shift via blockflow::ops::fft
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
           in-plane face; blocks in parallel on a BlockGrid
```

The rigid step corrects only the axes transverse to the tiling axis (the
tiling axis is projected away); the per-slice step corrects the in-plane axes,
including the tiling axis. Volumes are `(z, y, x)`, tiled in x and y, wobbling
along z; every module takes the axes as parameters.

The library API is `tile_stitching::stitch(tiles, layout, params)`, with the
steps also available one by one (`align_rigid`, `measure_wobbly`, `place_wobbly`,
`wobbly::measure_layout`, `place::place_layout`, `fuse::resident`).

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
`"position": [z, y, x]`, its nominal position in voxels; alignments are still
between grid neighbours.

### From a directory of converted tiles (`--tile-dir`)

`--tile-dir DIR` builds the tile set from the tiles themselves
(`src/tile_dir.rs`) and writes it, with a record of where each number came
from, to `<out>.tiles.json` (which `--tiles` reads back):

- **Tiles and grid.** Every OME-Zarr image in `DIR` whose name carries a grid
  index, the last `<row> x <column>` in it (`tile_03x04.zarr`,
  `sample[03 x 04]_c0.ome.zarr`), is a tile; the row runs along y and the
  column along x. Other images (an `autofluorescence.zarr` beside the tiles)
  are skipped. `--rows A:B` and `--columns A:B` (half-open, as named) select
  a sub-grid; grid indices are rebased to 0.
- **Nominal positions**, the first of:
  1. *physical translations*: when the tiles' level-0
     `coordinateTransformations` carry translations that differ between tiles,
     each tile's position is its translation relative to the first tile's, in
     voxels of `--level`;
  2. `--overlap Y,X` (level-0 voxels, one value for both allowed), scaled to
     `--level`.

  With neither, the directory is refused.

Only standard OME-Zarr metadata and the file name are read. Nothing about the
instrument or the format the tiles were converted from is used; that is the
converter's job (`img2omezarr`).

What a converter should record so that no `--overlap` is needed: each tile's
nominal position as a level-0 NGFF `translation`, in the image's own physical
units and axes (z, y, x), with the stage-to-image axis signs already resolved.
The grid index in the file name is then still needed for which tiles are
neighbours. The conversions at hand under `/husky/otherdataset/teresa/clearmap_em`
have no translations, so they need `--overlap` (osf-cztq2: `120,120`, measured
from the whole-sample stitch).

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
images' geometric centres) and the detectors (`cell-density`, `vessel-graph`,
which read each image's level-0 scale and translation) take the frame as it
is.

### Reusing a layout (`--layout-from`)

`--layout-from <out>.positions.json` skips the alignment and fuses at the
stored positions, carried to `--level`: in-plane positions are scaled by the
ratio of the voxel sizes (exact from a coarser level to a finer one by an
integer factor), and z must keep its voxel size. `--region Z0:Z1,Y0:Y1,X0:X1`
(fused voxels of `--level`) fuses only that box, reading only the tiles and
planes that reach into it; its frame is the box's place in the fused frame,
so it lines up with a whole-sample fusion at another level. This is how a
whole sample is stitched without holding it at full resolution: align and
fuse at level 1 (or 2) for registration, then fuse level-0 boxes for
detection. Writes `<out>.fuse.json` (layout, level, region, translation,
seconds).

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
interpolation|mean|max|min`, `--out-levels`, `--positions`. Both in-plane
range options also take one value for both axes; ranges are in voxels either
side.

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
  the fused frame keeps the first tile's; a positions file read back fuses
  the same volume, any box equals that crop, and level-1 positions carried to
  level 0 put every tile voxel where the scene has it, in the frame the
  level-1 volume describes; the CLI end to end (`--tile-dir`, then
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

Against ClearMap 3.1 on a 3 x 3 grid (`reference-clearmap/`, both at the
shipped search ranges, the reference with two of its defects corrected): rigid
positions identical; per-slice measurements identical in 919 of 921 aligned
slices; seam offsets within a voxel in 97% (y) and 93% (x) of planes, median
0; fused Pearson r 0.99.

## Limitations

- Resident: every tile (or its `--z-range`) and the fused volume are held in
  memory. For a whole sample, align at a coarser level with
  `--positions-only` and fuse with `--layout-from` (only the tiles and planes a
  `--region` needs are read); osf-cztq2's 40 tiles at level 1 peak at about
  40 GB. The fusion is block-local and parallel, but it is driven on a
  `BlockGrid` directly rather than through a `blockflow` plan, because no phase
  can read several images each at its own (per-slice) translation.
- The rigid step does not correct the tiling axis; that is left to the
  per-slice step, so `--rigid-only` keeps the nominal spacing along it.
- The whole-stack placement is a dense solve with one unknown per tile cluster
  per slice; long stacks of many tiles make it slow.
