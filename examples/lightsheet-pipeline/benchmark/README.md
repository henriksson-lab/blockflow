# Light-sheet benchmark: blockflow against the reference

One entry point that runs each light-sheet stage, and the whole chain, on
both sides — blockflow's examples and the reference they were developed
against (ClearMap 3.1, "the reference") — on identical inputs with matched
parameters, and records wall time, CPU time and memory of the whole process
tree, with enough provenance to defend or discard a number later.

It lives here, beside the pipeline driver, because the end-to-end workload is
`../run.sh` itself and the stage workloads reuse the per-stage harnesses in
`examples/*/reference-clearmap/` (their reference scripts, their inputs and
their `compare.py`); nothing in it is a library.

## Quick start

```sh
cd examples/lightsheet-pipeline/benchmark
python3 bench.py list                         # workloads, sizes, expected durations, differences
python3 bench.py run --smoke --repeats 1      # every workload at its smoke size (~40 min in all)
python3 bench.py summarize --max-load 8
```

**The full benchmark, when the machine is quiet** (about 10–14 hours at
`--repeats 3`; see the durations below; run it in `tmux`):

```sh
cd examples/lightsheet-pipeline/benchmark
python3 bench.py run --build \
    stitch-sa3x8:harness stitch-cztq2:l2 register:full cells:tile \
    vessels:region vessels-deepfill:region e2e:full \
    --threads 40 --repeats 3 --warm run --max-load 4 --wait-max-s 3600
python3 bench.py summarize --max-load 4 --exclude-loaded
```

- `--build` runs `cargo build --release -p` for the four packages
  (`vessel-graph` with `--features deep-fill`) first; the binaries' hashes and
  build times are recorded either way.
- `--warm run` does one discarded run of each side before the measured
  repeats (the reference's first process after a while pays several seconds,
  up to tens, for imports and its compiled extensions; the default `--warm
  read` only reads the inputs into the page cache).
- `--max-load 4 --wait-max-s 3600`: before each run, wait up to an hour for
  the 1-minute load to fall to 4; a run that starts above it anyway is
  flagged, and `--exclude-loaded` leaves it out of the summary.
- For a scaling point add a second pass with `--threads 16`; the summary
  groups by thread count.
- Bigger sizes (`stitch-sa3x8:l1`, `stitch-cztq2:l1`) prepare 4 and 21 GB of
  `.npy` and take much longer; run them separately.

Each `run` ends with the summary. Interrupting with Ctrl-C stops the run in
progress (the signal reaches the whole tree); rows already written stay.

## Files

| file | runs with | does |
|---|---|---|
| `bench.py` | any Python 3 (standard library) | the driver: `list`, `prepare`, `run`, `summarize`; workload definitions |
| `measure.py` | any Python 3 (standard library) | runs one command and measures its process tree (below) |
| `prepare.py` | Python with numpy, zarr 3 | untimed extraction of the reference's `.npy` inputs from the OME-Zarr data |
| `checks.py` | numpy + zarr (vessels), pandas + scipy (e2e), the reference's Python (registration) | output agreement per repeat |
| `clearmap/stitch.py` | the reference's env (`clearmap-gt2`) | rigid + wobbly stitching, as `tile-stitching/reference-clearmap/clearmap_stitch.py`, with the search ranges and process count given |
| `clearmap/register.py` | the reference's env | `Elastix.align` with its shipped parameter files |
| `clearmap/vessels.py` | the reference's env | its TubeMap steps from two raw channels to the reduced graph |
| `clearmap/e2e.py` | the reference's env | stitch, resample, register, detect, transform and label: its CellMap chain |
| `clearmap/compat.py` | imported by the above | two adaptations that let the reference run in parallel here (below) |
| `ours/vessels.sh` | bash | `vessel-graph segment` then `graph`, with per-step times |

The cell workload runs `cell-density/reference-clearmap/clearmap_cells.py`
unchanged; the stitching and cell checks are those harnesses' `compare.py`.

## Workloads

`bench.py list` prints all of this with the parameters. Durations are per
side at 16–40 threads, from the smoke run and earlier harness numbers.

| workload | sizes | data | expected per side |
|---|---|---|---|
| `stitch-sa3x8` | `smoke` (z 1200:1260), `harness` (z 1200:1440, the stitching harness's), `l1` (level 1, whole z) | osf-sa3x8, 3 x 3 tiles, channel 0 | smoke under a minute; harness 1–2.5 min; l1 5–15 min |
| `stitch-cztq2` | `smoke` (level 3, rows 0:2, columns 0:2), `l3`, `l2`, `l1` (whole 8 x 5 mosaic) | osf-cztq2 cFos | smoke under a minute; l3 2–5 min; l2 5–20 min; l1 15–60 min |
| `register` | `smoke` (affine), `full` (affine + B-spline) | osf-cztq2 autofluorescence -> ABA 2017 25 µm | elastix 2 / 9 min; blockflow 10 s / 1.5 min |
| `cells` | `smoke` (crop 256 x 512 x 512), `tile` (whole tile 03x02, 1.04 G voxels) | osf-cztq2 cFos | smoke 10–50 s; tile 1.5–15 min |
| `vessels` | `smoke` (crop 128 x 512 x 404 of tile 05x05), `region` (fused box 512 x 512 x 256) | osf-sa3x8, channels 0 + 1 | smoke 1–2.5 min; region 3–6 min |
| `vessels-deepfill` | as `vessels` | the same, with the CNN deep filling | smoke 1.5–2.5 min; region 4–10 min |
| `e2e` | `smoke` (level 3, affine only, box 128 x 256 x 256), `full` (level 2, affine + B-spline, the pipeline's 512 x 640 x 640 box) | osf-cztq2 | smoke: reference 5 min, blockflow 1 min; full: reference 20–40 min, blockflow 5–10 min |

Inputs are read-only: the OME-Zarr data under `/husky/otherdataset/teresa/
clearmap_em/`, the registration harness's 25 µm images and prepared atlas
(`/husky/henriksson/lightsheet-reference/registration/`), the vessel crops of
`vessel-planned/data/` (copied once into `inputs/`) and the prepared atlas
OME-Zarr. Everything written goes under `--root`
(`/husky/henriksson/lightsheet-reference/benchmark/`): `inputs/` (prepared
once, kept), `work/<workload>/<size>/<stamp>-r<k>/{clearmap,ours}/` (each a
fresh directory; the large outputs of all but the last repeat are removed
unless `--keep`) and `results/`.

## What is measured (`measure.py`)

Per run, for the command and every process it starts:

- **wall** from fork to the last process of the tree being reaped;
- **CPU** user + system from `getrusage(RUSAGE_CHILDREN)`. The wrapper is a
  child subreaper, so orphaned grandchildren are reaped and counted too;
  elastix, transformix and worker pools count;
- **tree RSS peak**: the summed `VmRSS` of the tree, sampled every 50 ms
  (`--interval`). It counts pages shared between processes once per
  process — an upper bound for fork-based worker pools;
- **tree PSS peak**: the summed PSS (shared pages split between users),
  sampled every second (`--pss-interval`): the fairest single number for a
  worker pool, and equal to RSS for one process;
- **tree anon peak** (`RssAnon`, without file-backed pages such as a
  memory-mapped `.npy`) and the **max single-process RSS** (`ru_maxrss`, what
  `/usr/bin/time` reports — meaningless as a total for a worker pool);
- the largest process and thread counts, and I/O (`rchar`/`wchar`,
  `read_bytes`/`write_bytes` summed over the tree's processes at their last
  sample, plus `ru_inblock`/`ru_oublock`; on NFS `read_bytes` is what the
  client fetched);
- the machine: load averages before and after, the 1-minute load each
  second (`load1_max`, `load1_mean`) and `other_cores_mean`: all CPUs' busy
  time from `/proc/stat` minus this tree's own CPU, per second of wall — how
  many cores other jobs used meanwhile.

Recorded once per invocation and stored in every row (`provenance`): host,
CPU model, `nproc`, sockets, RAM, available memory at the start, the
filesystems of the inputs and of `--root`, blockflow's commit with the number
of uncommitted paths and a hash of the diff over the packages involved, each
binary's hash and build time, the reference's checkout commit and dirty
state, its elastix version, both Pythons with numpy/scipy/torch/graph-tool/
zarr/pandas versions. Per row: the exact command, the thread environment and
any extra environment, CPU pinning, the warm-up and page-cache state, the
load wait, and each side's own per-step times (`side_timing`: the reference
scripts' `timing.json`, `registration.json` timings, `run.sh`'s per-stage
manifest).

## Fairness controls

- **Threads**: `--threads N` (default all cores) sets `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
  `NUMEXPR_MAX_THREADS`, `VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS` and
  `RAYON_NUM_THREADS` for both sides, and each tool's own flag: every
  `processes`/`n_processes` of the reference, elastix/transformix `-threads`,
  `torch.set_num_threads`, `cell-density --workers`, `vessel-graph --workers`.
  The reference's worker pools are fork-based and each worker's OpenMP/BLAS
  may also use N threads, as in a real run of it; no single pool exceeds N.
  `--cpus LIST` pins both sides with `taskset`.
- **Same filesystem**: all outputs go under `--root` (NFS here, as the
  inputs); both sides write there.
- **Fresh outputs**: every run gets a new directory (the driver refuses an
  existing one); untimed setup (copying the vessel image that `segment`
  writes into, writing the pipeline's parameter file) happens before the
  measurement.
- **Repeats, alternated**: `--repeats R`; the side that goes first alternates
  (reference first in even repeats, blockflow first in odd ones).
- **Load**: `--max-load L` flags runs that start above it; `--wait-max-s S`
  waits for the load to drop first. The load before, during and after, and
  the other jobs' cores, are in every row.
- **Page cache**: without root the caches cannot be dropped. `--warm read`
  (default) reads each side's inputs once right before its run (the row says
  whether they fit in half the available memory: if not, the cache state is
  "partial"); `--warm run` adds one discarded run of each side; `--warm
  none` does neither and says so.
- **Checks**: after each repeat (outside the measurement) the outputs are
  compared, so a fast but wrong run is visible: the stitching harness's
  `compare.py` (positions per slice, fused-volume correlation), the cell
  harness's `compare.py` (precision, recall), the registration point grid
  through both transforms (median distance), the vessel masks' Dice and
  skeleton/graph sizes, and the e2e cell counts and same-region fraction.

## Results

`results/` under `--root`:

- `runs.jsonl`: one JSON object per run with everything above;
- `runs.csv`: one row per run, the main columns (`run_id, workload, size,
  side, repeat, position, threads, variant, exit, wall_s, user_s, sys_s,
  cpu_s, tree_rss_peak_gb, tree_pss_peak_gb, tree_anon_peak_gb,
  max_rss_single_gb, processes_peak, load1_before, load1_after, load1_max,
  other_cores_mean, over_max_load, cache, started, blockflow_commit,
  blockflow_dirty, clearmap_commit, out`);
- `checks.jsonl`: one record per checked repeat (the run ids it covers);
- `summary.md`, `summary.json`: written by `summarize` — per workload, size,
  thread count and side, the median and range of wall time and the medians
  of CPU, tree RSS, tree PSS and the largest process, the median load before,
  the number of flagged runs, and the reference/blockflow ratios of the
  medians (above 1: blockflow faster or smaller), then the latest check per
  workload. `--workloads`, `--max-load` and `--exclude-loaded` filter.

Use `--results DIR` to keep a campaign apart from the smoke runs.

## Differences between the two sides

What a reader of the numbers must know; `bench.py list` prints the same per
workload.

**Everywhere**

- The reference reads `.npy` stacks in its `(x, y, z)` order (TIFF for
  elastix), which its own pipeline would make while converting the
  microscope's files; `prepare.py` extracts them from the same OME-Zarr
  voxels, untimed. blockflow reads the OME-Zarr (zstd chunks) inside its
  measurement. This favours the reference.
- The reference's shipped process counts are 10 (cell detection, vessel
  binarisation) or all but one; here every count is `--threads`.
- `clearmap/compat.py` adapts two things in the reference's checkout, no
  algorithm: (1) its block workers reopen a memory-mapped sink in the mode it
  was created in, `w+`, which truncates and re-creates the file while other
  workers write into it, so `vasculature.binarize` with a file sink dies of
  SIGBUS (`BrokenProcessPool`) with more than two processes; workers now
  reopen it `r+`. (2) The rank filters' compiled code takes `const char`
  masks, handed over as uint8/bool (as `binarize_roles.py` does).
- The reference's results depend on its process count through its block
  splitting (`optimization=True` adapts the blocks to the pool): the cell
  crop gave 1 080 filtered cells at 16 processes against 1 110 at 10; the
  vessel binarisation Dice 0.995 between 1 and 8 processes.

**Stitching**

- Reference variant `corrected` by default: its wobbly-placement defect and
  its tracing tie-break defect patched (the stitching harness's main
  comparison). `--variant shipped` runs it as shipped, which measures the
  per-slice displacements and then discards them (its result is the rigid
  layout); `patched` corrects only the placement.
- Search ranges: the reference's `stitching.yml` v3.1 level-0 values (rigid
  x, y 30, z 20; per slice x 30, y 15) divided by the level's factors and
  rounded up, on both sides.
- The reference smooths each tile's positions over the whole stack (including
  slices outside the tile), blockflow over the tile's own slices.
- blockflow writes one level (`--out-levels 1`), as the reference writes one
  array. osf-cztq2 needs `--overlap 120,120` (its conversion carries no tile
  translations); the reference gets the same overlap.

**Registration**

- The inputs are the registration harness's: the reference's own 25 µm
  resample of the autofluorescence and its prepared atlas (orientation (1, 2,
  3), z 0:256), as TIFF for elastix and as OME-Zarr for blockflow; resampling
  is not timed on either side here (it is in `e2e`).
- Both are stochastic; seed-to-seed noise of blockflow alone is about 16 µm
  median, so the check's median distance should be of that size.
- elastix writes its warped result image after each stage; blockflow warps
  the reference after each stage to measure similarity but writes only the
  transform and `registration.json`.

**Cells**

- Parameters match: the reference's `cell_map.yml` v3.1 and `cell-density
  --threshold 500` (its default threshold is also 500 now).
- The reference cuts the volume into z blocks of 10–16 planes with overlap 5,
  which split cells along z: most of its extra filtered cells (cell-density
  harness README). blockflow's tiled watershed gives the same answer on every
  block grid.
- Its environment lacks pyarrow, so its per-block Feather tables are pickles.

**Vessels**

- Reference: its TubeMap steps called one by one (`clearmap/vessels.py`) with
  `vasculature.yml` v3.1: per channel binarize (z blocks of 10–16, overlap
  5), smooth, fill (and deep fill); OR; smooth + fill; skeletonize;
  `graph_from_skeleton`; radii; artery binary and intensity per vertex;
  `clean_graph`; `reduce_graph`. Every intermediate is a `.npy` file, as its
  workspace has them. blockflow: `vessel-graph segment` then `graph
  --all-components` (every component, as the reference's clean + reduce
  keep); its `graph` also classifies arteries and computes lengths.
- No atlas step on either side.
- Known algorithm differences (vessel-graph README): clip rounding, ceiling
  window percentiles, the tubeness filter; per-channel binarisation Dice
  about 0.98.
- `vessels-deepfill`: both on the CPU (blockflow has no GPU path; the
  reference's orchestrator would use CUDA). The reference runs its network
  with dropout active, as shipped (nondeterministic, about half again as many
  filled voxels); `--deep-fill-eval` switches dropout off. Its blocks (500,
  1000 at resample 2, overlap 50/100) are smaller than the network's reach;
  blockflow plans 256 x 512 x 512 blocks with a 64 (128) voxel halo.

**End to end** (`e2e`)

- blockflow: `../run.sh` stages `align fuse register cells` with
  `STITCH_LEVEL = FUSE_LEVEL` = the size's level (the pipeline's own default
  aligns at level 1 and fuses at 2); its fuse stage writes a 4-level pyramid.
  Reference: stitch and fuse at that level, resample the fused signal and
  the full-resolution autofluorescence to 25 µm (`Resampling.resample`;
  blockflow reads a pyramid level of the autofluorescence instead), elastix
  signal -> autofluorescence (affine) and autofluorescence -> atlas.
- Cells: blockflow fuses the level-0 box from the level-L positions and
  detects there; the reference cannot fuse a level-0 box from coarser
  positions, so it detects on the same box of tile 01x02's own voxels
  (prepared untimed) and places it at the tile's median wobbly position. Its
  real pipeline would first fuse the whole sample at level 0, which is not
  measured: the reference side is faster here than in practice.
- Into the atlas: the reference's `resample_points` and `transform_points`
  twice (transformix), then the annotation at `astype(int)`; blockflow's
  `cell-density --registration --atlas` (nearest-centre rounding).
- `smoke` registers autofluorescence -> atlas with the affine only.
- Atlas positions are compared twice, as shipped and with the reference
  carried in the voxel-centre convention: see the next section.

## Atlas positions and the voxel convention

**Our choice.** Every image's level-0 voxel `i` has its centre at
`translation + i * scale` (NGFF). Pyramid levels shift their translation by
half a voxel per factor 2, so a coarse voxel's centre is the centroid of its
window. The OME-Zarr inputs are written that way too (osf-cztq2's
autofluorescence: level 1 `scale [12, 10, 10]`, `translation [3, 2.5, 2.5]`).
Points move between images through those physical frames, never through a
ratio of shapes, and an atlas voxel is picked by its nearest centre. Each step
is then exact for a point at a voxel centre, and a point keeps its place on
the image content whatever resolution it passes through. elastix uses the
same convention (an index is a voxel centre). This is what we defend.

**What the reference does.** It mixes two conventions:
- Its images are resampled with `cv2.resize`, which aligns voxel centres:
  source `(i + 0.5) / f - 0.5`.
- Its points are resampled with `resample_points`, which scales from the
  corner, `p * f`, with `f` the ratio of the discretised shapes. A point
  therefore lands `0.5 (1 - f)` resampled voxels off the image content it was
  found in: about half a 25 µm voxel per axis (12 µm) when `f` is small.
- It then labels with `astype(int)`, which truncates a centre-convention
  coordinate, a half-voxel bias with the opposite sign. On average this
  undoes part of the point shift for labelling, but not for the positions it
  writes (`xt, yt, zt`).

**Evidence** (e2e smoke, 548 matched cells, run 20261020T160505-r0;
`checks.py e2e` reports both):

| reference carried | ours − reference, atlas voxels (z, y, x) | distance, median / p95 | same finest region |
|---|---|---|---|
| as shipped: `p * f`, `astype(int)` | −0.33, −0.64, −0.35 | 20.1 / 20.9 µm | 95.4% |
| voxel centres: `(p + 0.5) f − 0.5`, nearest | 0.08, −0.14, 0.10 | 4.8 / 5.6 µm | 97.3% |

The as-shipped transform reproduces the reference's `xt, yt, zt` exactly
(within 3e-14 voxels), so the only change is the convention. The spread is
0.02 voxels per axis, so the 20 µm was a constant offset. The remaining 5 µm
is registration agreement: the two sides register independently. Mixing
conventions on the reference side (centres for positions, `astype(int)` for
labels) gives 89.6% region agreement, so labels and positions must be
judged in the same convention.

**The right comparison.**
- Compare positions and regions with the reference carried in voxel centres
  (`voxel_centres` in `checks.py e2e`). That separates registration
  differences from bookkeeping.
- Report the as-shipped numbers as the reference's own output, with the
  12 µm-per-axis offset named as a convention effect, not an error of ours.
- For region counts against the reference, its shipped numbers (corner
  points plus truncation) are what users of it get; they agree with ours to
  95% on this box, and 97% once the conventions match.

## Smoke run (2026-10-20)

Every workload at its smoke size, `--threads 16`, one repeat (cells two),
`--warm read`, on the shared 40-core machine with other jobs running (1-minute
load 8–45 at the starts, other jobs using 2–19 cores during the runs; see
`other_cores_mean` in `runs.jsonl`). **These numbers show that the scripts
work end to end; they are not measurements.** Results in
`/husky/henriksson/lightsheet-reference/benchmark/results/`, outputs of the
last repeat under `work/`.

| workload:size | side | wall s | CPU s | tree PSS GB | load before | check |
|---|---|---|---|---|---|---|
| stitch-sa3x8:smoke | reference | 13.4 | 79 | 2.74 | 23.9 | rigid positions identical; per slice as the harness |
| | blockflow | 9.5 | 32 | 1.09 | 22.7 | |
| stitch-cztq2:smoke | reference | 4.5 | 17 | 0.30 | 20.3 | rigid identical; pair offsets within 1 voxel on all 512 planes |
| | blockflow | 2.2 | 4 | 0.26 | 20.1 | |
| register:smoke | reference | 108.7 | 1790 | 2.51 | 20.7 | point grid: median 15.2 µm, p95 39.9 µm (the harness's affine numbers) |
| | blockflow | 10.0 | 111 | 1.36 | 44.6 | |
| cells:smoke | reference | 48.0, 6.7 | 105, 75 | 3.2 | 16.0, 19.4 | precision 98.7%, recall 98.0% (filtered 98.0% / 90.9%) |
| | blockflow | 8.2, 6.8 | 34, 32 | 1.4 | 20.7, 20.6 | |
| vessels:smoke | reference | 83.0 | 945 | 2.90 | 11.0 | final mask Dice 0.981; skeleton 242 k / 251 k voxels; graph 30.4 k / 31.7 k vertices |
| | blockflow | 102.8 | 788 | 3.61 | 20.6 | |
| vessels-deepfill:smoke | reference | 77.7 | 864 | 7.64 | 11.2 | Dice 0.912 (the reference's dropout fills more) |
| | blockflow | 123.2 | 922 | 3.63 | 13.3 | |
| e2e:smoke | reference | 306.5 | 3922 | 3.10 | 10.3 | 548 of 562 cells matched; same finest region 95.4%; atlas positions 20 µm apart (median) |
| | blockflow | 44.9 | 253 | 2.47 | 8.2 | |

What the smoke run showed about the method:

- The reference's first process in a while is slow (cells: 48 s, then 6.7 s
  for the same work): use `--warm run` and at least three repeats.
- Tree RSS overstates fork-based pools badly (reference vessels: RSS 9.3 GB,
  PSS 2.9 GB); quote PSS for the tree.
- elastix keeps its threads busy (1790 s of CPU in 109 s at 16 threads);
  its CPU time is not work done.
- The e2e reference spends 76 s of its 82 s stitching in the fusion
  (`stitch_layout`) at level 3; the per-step times are in each row's
  `side_timing`.

Fixed while making the smoke run work: the reference's vessel binarisation
needs the `compat.py` sink fix to run with more than two processes, and
transformix needs its `bin/` directory on `LD_LIBRARY_PATH`. The first
registration, vessel and e2e checks of this run failed on bugs in `checks.py`
(since fixed) and were rerun on the same outputs; the corrected records are
appended to `checks.jsonl` with a `note`.
