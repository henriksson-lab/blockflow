# lightsheet-pipeline

One light-sheet sample, end to end in atlas space, as the reference's
CellMap/TubeMap scripts run it:

1. **stitch** the signal tiles (rigid, then per slice) into one fused volume;
2. **register** the fused signal → autofluorescence → atlas as a chain
   (affine, then affine + B-spline, the reference's parameter files);
3. **detect** cells (`cell-density`) and/or vessels (`vessel-graph`) on
   level-0 boxes of the fused volume, with `--registration <chain>`, so atlas
   positions come out of the frames with no hand-given offset.

A driver script over the examples' binaries, not a crate yet (the start of
item F of `LIGHTSHEET_FEATURE_PLAN.md`). Originally developed as part of a
ClearMap reimplementation; "the reference" is ClearMap 3.1.

## Run

```sh
cargo build --release -p blockflow-tile-stitching -p blockflow-atlas-registration \
    -p blockflow-cell-density -p blockflow-vessel-graph      # or BUILD=1 below
examples/lightsheet-pipeline/run.sh SAMPLE_DIR PARAMS [STAGE...]
```

- `SAMPLE_DIR` holds the converted inputs (OME-Zarr only; convert TIFFs with
  `img2omezarr`): a directory of signal tiles named by grid index
  (`tile_RRxCC.zarr`) and the autofluorescence image.
- `PARAMS` is a bash file of `KEY=value` lines; `params/osf-cztq2.env` is a
  worked example and `run.sh` lists every key with its default.
- `STAGE`s: `align fuse register cells vessels` (default all; `cells` and
  `vessels` also need `CELLS=1` / `VESSELS=1`). A stage whose output exists
  is skipped unless `FORCE=1`.
- Needs `jq`, `/usr/bin/time` and a `python3` (`PY`) with numpy and, for
  vessels, `zarr`.

| key | meaning |
|---|---|
| `SAMPLE_ID`, `OUT_ROOT` | output goes to `$OUT_ROOT/$SAMPLE_ID` |
| `SIGNAL_TILES`, `SIGNAL_CHANNEL` | tile directory (relative to the sample) and channel |
| `AUTOFLUORESCENCE` | the autofluorescence image (relative to the sample) |
| `ATLAS` | a prepared atlas (`atlas-prepare`) in the sample's orientation |
| `ALIGNMENT_DIR`, `SIGNAL_TO_AUTO`, `AUTO_TO_ATLAS` | elastix parameter files per step |
| `STITCH_LEVEL`, `STITCH_OVERLAP`, `STITCH_ROWS`, `STITCH_COLUMNS`, `STITCH_MAX_SHIFT`, `STITCH_RIGID_MAX_SHIFT`, `STITCH_MAX_SHIFT_Z`, `STITCH_RIGID_ONLY`, `STITCH_ARGS` | `tile-stitching --tile-dir` (search ranges in voxels of `STITCH_LEVEL`) |
| `FUSE_LEVEL` | level of the whole fused volume, the registration's signal |
| `REGISTRATION_RESOLUTION`, `REGISTRATION_ARGS` | `atlas-registration chain` |
| `CELLS`, `CELL_REGION` or `CELL_TILE` + `CELL_TILE_BOX`, `CELL_ARGS` | the cell box (fused level-0 voxels, or a box in one tile's own voxels) and extra `cell-density` flags |
| `VESSELS`, `VESSEL_CHANNELS`, `VESSEL_ROLES`, `VESSEL_REGION` / `VESSEL_TILE` + `VESSEL_TILE_BOX`, `VESSEL_SEGMENT_ARGS`, `VESSEL_GRAPH_ARGS` | the vessel box and `vessel-graph` flags |

## Stages

| stage | command | output |
|---|---|---|
| `align` | `tile-stitching --tile-dir … --level $STITCH_LEVEL --positions-only` | tile positions |
| `fuse` | `tile-stitching --layout-from … --level $FUSE_LEVEL` | whole fused volume |
| `register` | `atlas-registration chain --image fused --image autofluorescence --image atlas` | chain |
| `cells-fuse`, `cells` | `tile-stitching --layout-from … --level 0 --region …`; `cell-density --registration … --atlas … --density …` | cell tables in atlas space |
| `vessels-fuse-<c>`, `vessels-stack`, `vessels-segment`, `vessels-graph` | one fused box per channel, stacked (`stack_channels.py`), `vessel-graph segment`, `graph --registration … --atlas …` | vessel graph in atlas space |

The alignment runs on a coarser level (all tiles and the placement fit in
memory there) and every fusion reuses its positions, so the full-resolution
volume is never held or written whole: the registration reads a level-2 fusion,
and detection reads level-0 boxes. The fused frames all describe the same
physical space (`tile-stitching` README, "Fused frame"), so a box's cells land
in the registration's sample frame through their metadata alone.

## Output layout (`$OUT_ROOT/$SAMPLE_ID/`)

This is the contract phase B (group statistics) reads.

```text
manifest.json                 schema 1: sample, commit, binaries' build times, per-stage
                              {wall_s, peak_rss_gb, disk_gb, started, command}
params.env                    the parameter file as run
logs/<stage>.{log,time,json}  stdout/stderr, /usr/bin/time -v, the stage summary
stitch/
  align.positions.json        tile positions at STITCH_LEVEL (per tile, per slice), frame
  align.tiles.json            the tile set derived from the tile metadata, with provenance
  fused.ome.zarr              whole sample at FUSE_LEVEL, OME-Zarr 0.5, frame of tile 0
  fused.ome.zarr.fuse.json    level, shape, translation, seconds
registration/
  chain.json                  steps in order: fused signal -> autofluorescence -> atlas
  step0/, step1/              registration.json + TransformParameters.N.txt per step
cells/                        (CELLS=1)
  region.txt                  the box, Z0:Z1,Y0:Y1,X0:X1 in fused level-0 voxels
  region.ome.zarr             that box at level 0, frame = its place in the fused frame
  detect/tables/cell-density          every basin (NGFF object table + table.csv)
  detect/tables/cell-density-filtered cells in the size band: z,y,x (box voxels),
                                      *_um (physical), atlas_z/y/x (atlas voxels),
                                      region, hemisphere, distance_to_surface_voxels
                                      (atlas voxels), names
  detect/tables/cell-density/regions.{csv,json}
                              per-(region, hemisphere) counts over the filtered cells,
                              volumes from the atlas, densities; regions.json the keying
  density.ome.zarr            cell density on the atlas grid
  report.json                 cell-density's JSON report
vessels/                      (VESSELS=1)
  region.txt, channel-<c>.ome.zarr, region.ome.zarr (stacked, with labels/)
  graph.json, tables          vessel graph in atlas space (vessel-graph README)
  branch-density.ome.zarr     branch density on the atlas grid
```

Atlas-space values (`atlas_*`, `region`, `hemisphere`, densities) are on the
prepared atlas named by `ATLAS`, the registration chain's last image.

## osf-cztq2 (cFos)

Run 2026-10-19 (blockflow `780f2cf` plus the uncommitted examples; binaries
built 02:11-02:37 the same day), `params/osf-cztq2.env`, 40 tiles of
1030 x 1007 x 1004, outputs in
`/husky/henriksson/lightsheet-reference/pipeline/osf-cztq2/`:

| stage | what | wall | peak RSS | disk |
|---|---|---|---|---|
| align | 40 tiles at level 1 (3.25 µm), 67 pairs, rigid + per slice | 243 s | 21.3 GB | 9 MB |
| fuse | whole sample at level 2, 1043 x 1811 x 1142 | 472 s | 13.9 GB | 3.3 GB |
| register | fused signal -> autofluorescence (affine) -> atlas (affine + B-spline), 25 µm | 90 s | 3.2 GB | <1 MB |
| cells-fuse | level-0 box 512 x 640 x 640 | 19 s | 2.3 GB | 0.34 GB |
| cells | `cell-density --registration`, 329 514 basins, 22 002 cells | 52 s | 8.9 GB | 0.67 GB |

Of the fusion's 472 s most is writing the OME-Zarr pyramid (the fusion itself
is parallel; the write is not). Registration similarity: signal -> auto
correlation 0.459, MI 0.570 (the earlier nominal-grid mosaic: 0.487 / 0.655);
auto -> atlas 0.660 / 0.543 after the B-spline (earlier 0.657 / 0.537).

**Against the earlier hand-offset run** (`cells-atlas/`: the same window of
tile 01x02, placed by `--sample-offset-um` in a nominal-grid 25 µm mosaic
registered through its own chain), `checks/compare_hand_offset.py`, output in
`cells/comparison-hand-offset.json`:

- detection: 21 813 of 21 997 old cells (99.2%) match a new one within 2
  voxels, with zero median offset: the box (`CELL_TILE=01x02`,
  `CELL_TILE_BOX=256:768,192:832,192:832`) lies inside the tile, away from
  its overlaps, so it holds the tile's voxels up to the per-slice wobble;
- atlas positions: **median 20 µm, 95th percentile 43 µm, max 62 µm**, within
  the registration's noise (two seeds of one registration differ by a median
  of 16 µm); same finest region 92.3%, same hemisphere 100%;
- a systematic shift of (-5, -7, +12) µm (z, y, x), under a quarter of an
  atlas voxel. Mapping the same cells through each chain's steps separately:
  - signal -> autofluorescence contributes a median 19 µm, mean (-2, 0, +19)
    µm, mostly along x: the earlier mosaic placed the columns 1368 µm apart (to
    span the autofluorescence) where the stitch finds about 1440 µm, so its
    affine had to absorb a 5% stretch along x;
  - autofluorescence -> atlas contributes a median 18 µm, mean (-4, -11, -11)
    µm: the earlier chain registered the reference's 25 µm resample of the
    autofluorescence, which spans the stack's extent, so its voxel 0 centre
    sits half a 25 µm voxel minus half a raw voxel from the raw image's, which
    is registered here: (25 - 5) / 2 = 10 µm in y and x, as seen (z, -4 µm,
    less so).
- Against the reference's own elastix chain and labelling of the old cells
  (`clearmap/tile_01x02_crop/cells_reference.csv`): median 36 µm, 95th
  percentile 53 µm, mean shift (-19, -9, +1) µm; same finest region 89.3% (the
  hand-offset run against it: median 25 µm, 90.3%). The extra distance is the same frame
  difference as above plus the reference's own registration noise.

## osf-sa3x8 (vessels): not run

The conversion holds 9 tiles (rows 03-05, columns 04-06) of a whole-brain
mosaic: about 1.7 x 5.4 mm of a 6.6 x 11.5 mm autofluorescence field. A
signal -> autofluorescence affine started from the two images' geometric
centres, as elastix's `AutomaticTransformInitialization` (and ours) does, has
no way to find where that strip sits, and the tiles' metadata cannot place it (it
carries no translations), and there is no initial transform option in
`atlas-registration`. The vessel stages are written but not run
(`stack_channels.py` was smoke-tested on two copies of the cell box); running them needs either the
whole mosaic or an initial transform (a landmark registration, plan item G).

## Limitations

- One sample per call; the batch over samples (item F proper) is a loop over
  this script.
- Detection runs on boxes, not the whole sample at level 0: cells and
  vessels are not merged across boxes.
- The registration's signal is fused at level 2 from level-1 positions, so
  in-plane positions are rounded to 6.5 µm there (half a voxel at most);
  level-0 boxes reuse the level-1 positions doubled, exactly.
- The stitching overlap of osf-cztq2 has to be given (`STITCH_OVERLAP`): the
  tiles' metadata gives no usable one (tile-stitching README).
