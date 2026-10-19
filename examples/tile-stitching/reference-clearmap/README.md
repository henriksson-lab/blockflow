# tile-stitching against the reference implementation

Stitches the same tiles with ClearMap 3.1 (the reference) and with
`tile-stitching`, and compares positions, fused volumes, time and memory.

## Files

| file | runs with | does |
|---|---|---|
| `prepare_tiles.py` | a Python with `zarr` 3, numpy | extracts channel 0 of each OME-Zarr tile over the z-range as `npy/tile_YYxXX.npy` in the reference's `(x, y, z)` order, and writes `tiles.json` for `tile-stitching` (pointing at the OME-Zarr tiles) |
| `clearmap_stitch.py` | the reference's env (`clearmap-gt2`) | rigid alignment and placement, wobbly alignment, placement and fusion, as the reference's stitching orchestrator runs them with the `config/defaults/v3.1/stitching.yml` parameters and `n_processes: -1`; optionally with the defects below patched |
| `measure.py` | any Python 3 | runs a command under `/usr/bin/time -v` and samples the summed RSS/PSS of its process tree (the reference runs worker pools, which `time` does not add up) |
| `compare.py` | `zarr` 3, numpy | the comparison tables below |
| `run.sh` | bash | `prepare`, `clearmap`, `ours`, `compare` or `all` |

Environment, with defaults: `STITCH_TILES`
(`/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles`), `STITCH_OUT`
(`/husky/henriksson/lightsheet-reference/stitching`), `STITCH_Z`
(`1200:1440`), `STITCH_VARIANTS` (which reference runs, by directory
name; default all), `STITCH_ROWS` / `STITCH_COLS` (`3,4,5` / `4,5,6`, the first and
second index of `tile_YYxXX`), `STITCH_OVERLAP` (`66,212`, x and y, what the
reference's `parse_overlaps` reads from the original Brain-39L OME-TIFF;
the other tiles' TIFFs have no valid OME metadata), `CLEARMAP_PY`, `PY`.

The reference is run five ways (`run.sh clearmap`), each patch a textual
edit of one function's source (see `clearmap_stitch.py`):

| directory | placement fix | tracing fix | smoothing over own slices | wobbly y search |
|---|---|---|---|---|
| `clearmap/` | | | | 15 |
| `clearmap-patched/` | yes | | | 15 |
| `clearmap-patched-y30/` | yes | | | 30 |
| `clearmap-corrected/` (the main comparison) | yes | yes | | 15 |
| `clearmap-corrected-span/` | yes | yes | yes | 15 |

and ours three ways: `ours/` (defaults: wobbly search y +-15, x +-30, as the
reference's), `ours-y30/` (`--max-shift 30`) and `ours-rigid/`
(`--rigid-only`).

## Search ranges

The reference's shipped defaults search, in voxels either side, as follows.
They come from `config/defaults/v3.1/stitching.yml`, the one stitching
config in 3.1: `cell_map.yml` and `vasculature.yml` have no stitching
section. The v3.0 and v2.1 configs and `Scripts/TubeMap.py` use the same
values.

| stage | x | y | z |
|---|---|---|---|
| rigid | 30 | 30 | 20 |
| wobbly (per slice) | 30 | 15 | 0 |

`tile-stitching` now has the same defaults: `--max-shift 15,30` (per slice,
`Y,X`), `--rigid-max-shift 30,30` (rigid, in plane) and `--max-shift-z 20`
(rigid, z). Both range options accept one value for both axes.

## Defects in the reference

**Its wobbly placement is a no-op.** In ClearMap 3.1,
`stitching_wobbly._connected_components(positions, alignment_pairs,
displacements)` rebinds `displacements = []` inside its loop before zipping
over it, so every component gets no alignments, `_place_slice` returns the
rigid positions and `place_layout` writes them back for every slice. The
per-slice displacements are measured (here about `+13` voxels along x for
every x-neighbour pair, e.g. 349..352 against the rigid 338), then
discarded: the "wobbly" result is the rigid layout, and the fused volume is
the rigid fusion (1084 x 3489 x 244 here, exactly the rigid extent).
`STITCH_PATCH_PLACEMENT=1` renames that list and nothing else.

**Its tracer breaks ties between trajectories by the wrong landscape.** In
`shifts_from_tracing`, when several trajectories are equally long, the one
with the lowest summed error wins, but each point's error is read from
`errors[l]`, where `l` is the point's index among the measured slices of the
stretch, not its slice. With every third slice measured, that is the
landscape of a slice a third of the way along, and most of those were never
measured (all zero), so the tie is broken by chance. `tile-stitching` reads
each point's own slice (`select_trajectories`, tested by
`the_greedy_pick_reads_each_points_own_slice`). `STITCH_PATCH_TRACING=1`
reads `errors[s + measured_se[l]]`. This is most of the earlier gap: of the
919 slices both sides aligned, the per-slice measurements (before smoothing
and placement) are identical in 803 (87%) against the reference with the
defect and in 919 (100%) with it corrected. The x-neighbour pairs (overlap
66 x 1304) were identical either way; all the differences were in the
y-neighbour pairs (overlap 212 x 404), whose landscapes have several
minima of similar depth, so ties are common.

**Its position smoothing reaches outside each tile** (a choice more than a
defect). `place_layout` smooths each tile's optimised positions over every
slice of the layout, including the slices outside the tile and its isolated
slices. Those hold the tile's *rigid* position, not an optimised one, and
the binary pass then sets the first and last 11 slices to their median, so
the ends of every stack are pulled towards the rigid layout. It also lowers
the layout to its origin over those values, which here moves the whole
frame by 6 voxels in y. `tile-stitching` smooths each tile over its own
slices, after filling its isolated slices from its neighbours.
`STITCH_PATCH_SMOOTHING=1` does the former.

## Results

OSF `sa3x8` (Brain-39L), 3 x 3 tiles `03..05 x 04..06`, z `1200..1440`
(each tile 240 x 1304 x 404 `uint16`), 2026-10-19, release build. The
machine was shared, so wall times are indicative; the reference's own wall
time varied 35-145 s between identical runs.

**Rigid**: every pairwise displacement and quality is identical to the
reference's, and the placed positions are identical for all nine tiles
(maximum difference `[0, 0, 0]` voxels).

**Rigid fusion** (`ours-rigid` against `clearmap` as shipped, which is rigid
in effect): Pearson r 0.9986 in single-tile voxels (99.8% identical), 0.9996
in overlaps (50.9% identical: the fusion weights differ slightly).

**Per slice**, ours against the reference. *Pair displacement* is, for each
neighbouring pair and plane, the in-plane offset of the second tile relative
to the first (what the seams see; free of each side's choice of per-slice
gauge), ours minus reference, over planes valid on both sides (n = 2867).
*Tile position* is relative to tile 0's rigid position, so it includes the
gauge. Each cell is median / p95 / max absolute difference in voxels, and
the fraction within 1 voxel.

| ours | reference | pair y | pair x | tile position y | tile position x |
|---|---|---|---|---|---|
| `ours` | `clearmap-corrected` | 0 / 1 / 5, 97.3% | 0 / 2 / 10, 92.6% | 1 / 2 / 5, 87.0% | 0 / 2 / 15, 94.1% |
| `ours` | `clearmap-corrected-span` | 0 / 1 / 3, 97.4% | 0 / 1 / 5, 97.7% | 6 / 7 / 7, 0% (gauge) | 0 / 1 / 5, 98.5% |
| `ours` | `clearmap-patched` | 1 / 4 / 8, 66.4% | 1 / 5 / 10, 59.9% | 2 / 4 / 7, 47.9% | 1 / 5 / 16, 52.2% |
| `ours-y30` | `clearmap-patched-y30` | 1 / 4 / 11, 77.8% | 1 / 5 / 14, 68.3% | 1 / 3 / 6, 80.1% | 3 / 7 / 18, 12.1% |

Before the search ranges matched (ours +-30 in y against the reference's
15), the pair displacements were 4 / 13 / 20, 23% (y) and 3 / 13 / 31, 30%
(x) against `clearmap-patched`.

**Fused volume**, Pearson r between the two fused volumes; *volume frame*
maps the reference onto ours by one offset (tile 0's first plane),
*per-plane frame* by the median over tiles of the per-plane offset, which
removes the gauge:

| comparison | frame | overlap voxels | single-tile voxels | all |
|---|---|---|---|---|
| `ours` vs `clearmap-corrected` | per plane | 0.9935 | 0.9889 | 0.9898 |
| `ours` vs `clearmap-corrected` | volume | 0.9780 | 0.9623 | 0.9653 |
| `ours` vs `clearmap-corrected-span` | per plane | 0.9956 | 0.9921 | 0.9928 |
| `ours` vs `clearmap-corrected-span` | volume | 0.9954 | 0.9920 | 0.9927 |
| `ours` vs `clearmap-patched` | per plane | 0.9682 | 0.9395 | 0.9450 |
| `ours-y30` vs `clearmap-patched-y30` | per plane | 0.9492 | 0.9372 | 0.9395 |
| `ours` vs `clearmap` (as shipped) | per plane | 0.8321 | 0.6570 | 0.6933 |
| `ours-rigid` vs `clearmap` (as shipped) | volume | 0.9996 | 0.9986 | 0.9988 |

Vessel images decorrelate within a voxel or two of shift, so these are
sensitive; a one-voxel disagreement in a plane is enough to bring r well
below 1.

**What is left.** With the measurements identical, the rest is placement
arithmetic. Against `clearmap-corrected`, the planes off by more than a
voxel are mostly the first and last 10 of the stack (the smoothing above)
and planes 100-120, where several y-neighbour alignments failed tracing and
the per-slice graph loses edges. The remaining one-voxel differences are
rounding: both sides round each slice's least-squares positions and each
cluster's whole-stack shift to integers, and the reference rounds `lstsq`
output whose exact half-integers carry 1e-13 residuals, where ours snaps to
the half and rounds half to even (`place::round_snapped`). Applying that
rounding to the reference as well moves the pair agreement by under half a
percentage point, so it is not worth a variant. No change to
`tile-stitching`'s placement is indicated: the reference's choices that
differ are the defect and the out-of-tile smoothing above.

**Time and memory** (`measure.py`; peak RSS of the largest process, and the
summed PSS of the whole tree):

| run | wall | user + sys | peak RSS | tree PSS |
|---|---|---|---|---|
| reference as shipped, 39 processes | 59-145 s (rigid 3 s, wobbly align 27 s, place 0.7 s, fuse 27 s in the fastest run) | 300-340 s | 1.2 GB | 10.5 GB |
| reference, corrected | 35-94 s (rigid 0.6 s, wobbly align 13 s, place 0.5 s, fuse 21 s in the fastest run) | 190-400 s | 1.2 GB | 10.2-10.9 GB |
| ours | 40-45 s | 212 s | 4.7 GB | 4.7 GB |
| ours, `--rigid-only` | 26-33 s | 162-183 s | 4.1 GB | 4.1 GB |

The reference's times exclude its tile conversion to `.npy` (done by
`prepare_tiles.py`, about 30 s); ours include reading the OME-Zarr tiles.

## Other differences worth knowing

- Fused extents (z, y, x): ours 244 x 3499 x 1135; `clearmap-corrected`
  244 x 3500 x 1135; reference as shipped 244 x 3489 x 1084, the rigid
  extent. The fusion stage is most of the reference's time variation.
- `compare.py` maps the reference's stitched array from its
  `origin_wobbly`, `min(lower, 0)` per axis; the out-of-tile smoothing
  variant is the first whose `lower` is not zero.
