# cell-density against the reference implementation

Detects cells on the same voxels with ClearMap 3.1's `detect_cells` (the
reference) and with `cell-density`, matches the detections, and compares
counts, sizes, intensities, time and memory.

## Files

| file | runs with | does |
|---|---|---|
| `prepare_crop.py` | a Python with `zarr` 3, numpy | extracts level 0 of the OME-Zarr image over the crop as `input.npy` in the reference's `(x, y, z)` order; `cell-density` reads the same voxels from the OME-Zarr image with `--crop-*` |
| `clearmap_cells.py` | the reference's env (`clearmap-gt2`) | `detect_cells` and the size filter as the reference's cell-map orchestrator runs them with `config/defaults/v3.1/cell_map.yml`: background disk (10, 10), maxima shape 5, no h-max, shape threshold 500, intensity measure `source`, size band [50, 900), z blocks of 10..16 planes with overlap 5, 10 processes |
| `measure.py` | any Python 3 | `/usr/bin/time -v` plus the summed RSS/PSS of the process tree; `MEASURE_KILL_GB` stops a run that outgrows a shared machine |
| `compare.py` | numpy, scipy, pandas | the tables below |
| `run.sh` | bash | `prepare`, `clearmap`, `ours`, `compare` or `all`; `ours` rebuilds and records the source state (`ours/source-state.txt`: time, HEAD, git status, source file times, binary hash) |

Environment, with defaults: `CELLS_TILE`
(`/husky/otherdataset/teresa/clearmap_em/osf-cztq2/cFos/tile_03x02.zarr`),
`CELLS_CROP_START` (`400,248,246`), `CELLS_CROP_SHAPE` (`256,512,512`; 0 runs
to the end of an axis), `CELLS_OUT`
(`/husky/henriksson/lightsheet-reference/cells/crop`), `CELLS_THRESHOLD`
(`500`), `CELLS_OURS_ARGS`, `CELLS_OURS_DIR` (`ours`), `CELLS_REF_DIR`
(`clearmap`), `CLEARMAP_PROCESSES` (`10`), `CLEARMAP_BLOCK` (`10,16,5`:
size_min, size_max, overlap), `CELLS_MATCH_RADIUS` (`2`), `CLEARMAP_PY`, `PY`.

The full tile: `CELLS_OUT=/husky/henriksson/lightsheet-reference/cells/tile_03x02
CELLS_CROP_START=0,0,0 CELLS_CROP_SHAPE=0,0,0 run.sh all`.

Two adjustments to make the reference run, neither touching its algorithm:
the environment lacks `pyarrow`, so the Feather files it passes per-block
tables through are swapped for pickles; and its input is the `.npy` crop
(what its pipeline reads after stitching).

`cell-density` is run with `--threshold 500` (the reference's 3.1 default
shape threshold; `cell-density`'s own default is 700) and its other
defaults, which match the reference's: flat 10 x 10 disk per plane, window
maxima of radius 2, size band [50, 900), watershed with a one-voxel line.

## Matching

One-to-one, closest pairs first, within 2 voxels (Euclidean, voxel units).
Precision is the matched fraction of our detections, recall of the
reference's. `source` is the raw maximum over the basin in both (the
reference measures over its shape labels when shape detection runs), so it
differs only where the basins differ.

## Results

OSF `cztq2` cFos, tile `03x02` (1030 x 1007 x 1004 `uint16`, voxels
6 x 1.625 x 1.625 µm), 2026-10-19, release build, on a shared machine (load
average 45-120 on 40 cores, the full tile partly in swap), so wall times
are indicative and CPU times more so.

### Crop `z 400..656, y 248..760, x 246..758` (67.1 M voxels)

| set | ours | reference | matched | precision | recall | same voxel | size r (Pearson / Spearman), identical | source r, identical |
|---|---|---|---|---|---|---|---|---|
| all detections | 53 964 | 54 358 | 53 256 | 98.7% | 98.0% | 98.4% | 0.942 / 0.991, 88.6% | 0.987 / 1.000, 99.4% |
| size band [50, 900) | 1 002 | 1 110 | 981 | 97.9% | 88.4% | 98.9% | 0.854 / 0.941, 39.6% | 0.999 / 0.995, 97.5% |

99.4% of our filtered cells are detections of the reference at some size,
and 99.7% of its filtered cells are ours: the band disagreements are size
disagreements, not detection disagreements.

The reference's own answer depends on its z blocks. With one block over the
crop (`CLEARMAP_BLOCK=256,256,0`, `clearmap-oneblock/`) it finds 54 359
detections and **1 020** filtered cells instead of 1 110, and against that:

| set | ours | reference (one block) | matched | precision | recall | size r, identical | source r |
|---|---|---|---|---|---|---|---|
| all detections | 53 964 | 54 359 | 53 256 | 98.7% | 98.0% | 0.995 / 0.992, 88.8% | 1.000 |
| size band | 1 002 | 1 020 | 980 | 97.8% | 96.1% | 0.983 / 0.983, 42.0% | 1.000 |

So most of the filtered-count gap at the 3.1 defaults (1 002 against
1 110) is the reference's 16-plane blocks with a 5-plane overlap, which cut
cells along z. What remains is small, systematic size differences
(identical sizes for 42% of filtered matches, ours smaller more often than
larger): the background opening's handling of even element sizes, as the
example's README describes.

`--watershed whole` gives byte-identical tables to the default tiled
watershed on this crop (`ours-whole/`).

| side | wall | user + sys | peak RSS (largest process) | tree PSS | M voxels/s (wall) |
|---|---|---|---|---|---|
| reference, 10 processes | 9.1 s | 71 s | 0.35 GB | 2.2 GB | 7.4 |
| reference, one block | 108 s | 78 s | 3.0 GB | 3.1 GB | 0.6 |
| ours, 40 threads | 16.5-34 s | 129-130 s | 4.9 GB | 4.1-4.9 GB | 2.0-4.1 |

### Full tile (1 041 M voxels)

| set | ours | reference | matched | precision | recall | same voxel | size r, identical | source r, identical |
|---|---|---|---|---|---|---|---|---|
| all detections | 982 947 | 991 578 | 960 501 | 97.7% | 96.9% | 97.7% | 0.803 / 0.989, 80.6% | 0.972 / 0.999, 98.5% |
| size band [50, 900) | 84 424 | 91 687 | 82 749 | 98.0% | 90.3% | 98.1% | 0.831 / 0.917, 34.7% | 0.980 / 0.989, 96.3% |

Of our filtered cells 99.4% are reference detections; of its, 98.5% are
ours.

| side | wall | user + sys | peak RSS | tree PSS | M voxels/s (wall) |
|---|---|---|---|---|---|
| reference, 10 processes | 837 s | 1 168 s | 0.95 GB | 6.2 GB | 1.24 |
| ours, 40 threads | 3 491 s | 10 646 s | 48.2 GB | 46.1 GB | 0.30 |

Our full-tile run used the binary built at 00:26 on 2026-10-19
(`detect.rs` of 23:16 the day before); `detect.rs` changed again at 00:39,
and the crop rerun with that later binary (01:29) gave identical tables.

## What looks wrong on our side

- **Memory on a whole tile.** 48 GB peak for 1.04 G voxels (about 46 B per
  voxel) with the default tiled watershed, against 6 GB summed over the
  reference's ten workers. The tiled watershed should bound memory to a
  tile and its halo; something else in the plan still holds the whole
  volume (or several copies).
- **CPU time.** 9x the reference's CPU on the full tile (10 646 s against
  1 168 s) and 1.8x on the crop (130 s against 71 s). The full-tile run
  spent long stretches with its threads in uninterruptible sleep at about
  3-4 cores, consistent with paging under the memory above on a loaded
  machine, so part of that is the memory problem; the crop number is not.
- **Default threshold.** `cell-density` defaults to `--threshold 700`; the
  reference's 3.1 configuration uses 500 (700 is only its code default).
