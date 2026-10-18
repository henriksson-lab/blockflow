# Plate time-lapse tracking

Segment cells in every frame of a multiwell-plate 2-D time-lapse, measure them
in a second channel, link them from frame to frame into tracks, and publish
labels and tables from which each cell's measurements can be plotted over time.
The design and its reasoning are in [`TRACKING_PLAN.md`](../../TRACKING_PLAN.md).

**Time is never a Blockflow grid axis.** Each frame is an ordinary rank-three
`[1, Y, X]` image (`[Z, Y, X]` later), segmented and measured by ordinary plans.
What crosses frames is rows. `t` exists only in storage (labels are `[t, y, x]`)
and in the tables.

## Pipeline

For every series (one field imaged over time), with all frames of all series
on one bounded pool:

1. **Frame job**, per stored time index:
   - `InstanceSegment` (threshold, StarDist or Cellpose) writes final-id labels
     straight into plane `t` of a hidden `[t, y, x]` label stack. Ids are offset
     by `t_index · Y·X`, so they are unique across the series with no relabel
     pass.
   - `Measurements` (shape, plus intensity in the measurement channel) reads
     that plane back.
   - Rows go into `frames/<t>/rows.json`, and `manifest.json` is written last.
2. **Series stage**:
   - Gated candidates between consecutive acquired frames.
   - `FrameLapLinker` (frame-to-frame LAP), producing a validated `TrackGraph`
     with tracks and observational events.
   - Label pyramids built frame by frame, downsampling space only.
   - Optional track-coloured labels.
   - Tables and exports.
   - Publish by rename, with `tracking/<layer>/tracking-summary.json` last.
3. **Collection index** at the plate root.

## Input

Three ways to name what to process:

- `--plate P.zarr`: an OME-NGFF HCS plate. Wells come from `plate`, fields
  and acquisitions from each well's `well.images`.
- `--manifest M.json`: separately stored images, with every relationship
  stated explicitly:

  ```json
  {"version": 1, "plate_id": "P1", "series": [
    {"path": "A1-f0.zarr", "well_row": "A", "well_column": "1", "field": "0",
     "acquisition": 0, "channels": {"segment": 0, "measure": 1},
     "timeline": [{"t_index": 0, "elapsed_seconds": 0}, {"t_index": 1, "elapsed_seconds": 300}]}
  ]}
  ```

  `timeline` maps stored frames to acquisition indices. A missing index is a
  gap, and version 1 ends tracks before it and starts new ones after it.
- `--image F.zarr`: one field.

Each field image is a multiscale `t,c,y,x` (or `t,c,z,y,x`) image. Elapsed
time comes from the `t` scale when its unit is a time unit.

Results are written into each field image under `labels/`, `tables/` and
`tracking/`. The command does not create a separate overlay dataset. This keeps
the image and its derived tracking data together in the OME-Zarr hierarchy.

## Run

Synthetic data with ground truth:

```bash
cargo run --release -p blockflow-plate-timelapse-tracking -- \
  synth --out /tmp/plate.zarr --rows A,B --columns 1,2 --frames 20 --size 256 --cells 60

cargo run --release -p blockflow-plate-timelapse-tracking -- run \
  --plate /tmp/plate.zarr --threshold 100 --max-distance 6 \
  --halo 16 --blocks 256,512 --label-chunk 128 --track-labels

cargo run --release -p blockflow-plate-timelapse-tracking -- score \
  --detections /tmp/plate.zarr/A/1/0/tracking/cells/detections.csv \
  --truth /tmp/plate.zarr/truth/A-1-f0-acq0.csv
```

StarDist and Cellpose are build features:

```bash
cargo run --release -p blockflow-plate-timelapse-tracking --features stardist -- run \
  --plate P.zarr --segmenter stardist --model MODEL_DIR \
  --normalize movie --low-percentile 1 --high-percentile 99.8 \
  --segment-channel 0 --measure-channel 1 --max-distance 15

cargo run --release -p blockflow-plate-timelapse-tracking --features cellpose-cuda -- run \
  --plate P.zarr --segmenter cellpose --model cpsam.safetensors --cuda-device 0 \
  --normalize movie --max-distance 15
```

Normalisation is explicit and recorded. The options are:

- `fixed --low --high`;
- `movie`: percentiles over every frame of the series, computed once and
  shared by all frames;
- `frame`: per-frame percentiles, robust to bleaching but not comparable
  across frames.

The network is loaded once, configured for `[0, 1]` input, and each frame is
mapped there with its own range.

Main tracking options:

| option | meaning |
|---|---|
| `--max-distance` | gate, physical units per frame |
| `--max-speed` | gate as speed × elapsed time; needs timestamps |
| `--iou-weight` | add `w·(1 − IoU)` to the cost; overlap is read from the two label planes |
| `--alternative-cost` | fixed birth/death cost (default 1.05 × the largest link cost of each frame pair) |
| `--border-distance` | physical distance under which an endpoint is `near_border` |
| `--drift-correction` | estimate stage drift between frames and gate links on drift-corrected positions |
| `--drift-level` | pyramid level for the image estimate (default: finest level ≤ 1536 px) |

### Lineage: divisions, splits, merges, gaps

By default linking is frame-to-frame and one-to-one (`frame-lap`). Any of the
following switches on segment linking (`segment-lap`, the second step of
Jaqaman et al. 2008, as in TrackMate and `laptrack`):

| option | meaning |
|---|---|
| `--gap-closing D` | an ending track may join a track starting up to `--max-gap` (default 2) frame indices later, within `D` |
| `--splitting D` | a new track may split from any detection one frame earlier within `D` |
| `--merging D` | an ending track may merge into any detection one frame later within `D` |
| `--segment-alternative-cost C` | fixed alternative cost (default 1.05 × the 90th percentile of the segment step's candidate costs) |

Segment linking takes the frame-to-frame segments and solves one global
assignment over all their ends and starts.

A division is a split. So is a cell that Cellpose cut in two for a frame, and
when the pieces rejoin a merge follows. The lineage then shows a small
split-then-merge **bubble**. This is deliberate: the graph records what
linkage saw before any correction, so segmentation errors are visible rather
than silently resolved. Collapsing bubbles into one cell is a separate, later
step.

Where the lineage is recorded:

- `tables/<layer>-links`: `kind` is 0 continue, 1 divide, 2 merge, 3 gap;
- `tables/<layer>-tracks`: `n_parents`, `parent_track_id` and `n_children`;
- `tracking/<layer>/graph.json`: napari's parent map, which allows several
  parents per track;
- the CTC export is skipped (with a note) when there are merges, because CTC
  allows one parent per track.

On four synthetic fields with divisions, the full edge set — frame links,
splits, merges and gap closings — was identical to `laptrack`'s with the same
cutoffs and fixed costs (`scripts/compare_laptrack.py --splitting ...`).

### Stage drift

A plate reader revisits each well between frames, and the stage does not
return exactly. On the first real plate (Tecan, row A), the field moved about
17 µm between frames 0 and 1 and about 30 µm over the movie. That is more than
any sensible gate, so without correction most cells broke their tracks there.

`--drift-correction` estimates the drift in two steps:

1. Phase correlation between consecutive frames of the segmentation channel,
   at a coarse level, with low-pass weighting.
2. Refinement from the detections: every cell votes for its displacement to
   nearby cells in the next frame, and the most common vote wins.

The per-frame offsets are subtracted before gating and recorded in the
summary (`tracking.drift_offsets_physical`). Tables keep native pixel
positions. Overlap costs (`--iou-weight`) compare native pixels, so they are
refused together with drift correction.

## Output, per field

| path | content |
|---|---|
| `labels/<layer>/` | `[t, y, x]` `uint64` detection ids, multiscale; only `y, x` downsampled |
| `labels/<layer>-tracks/` | (`--track-labels`) the same pixels relabelled by `track_id` |
| `tables/<layer>/` | one row per detection: `detection_id, t, y, x, track_id, t_index, elapsed_seconds, area_pixels, area_physical, equivalent_diameter, eccentricity, major/minor_axis_length, orientation_yx, segment_{mean,min,max}, measure_{sum,mean,min,max}` |
| `tables/<layer>-tracks/` | one row per maximal unbranched track: start/end frame and ids, parents/children, `start_reason`, `end_reason`, border flags |
| `tables/<layer>-links/` | one row per edge: `from_id, to_id, dt, elapsed_seconds, cost, kind` |
| `tracking/<layer>/identity-state.json` | revisioned durable detection, track, lineage, and link identities used by interactive editors |
| `tracking/<layer>/tracking-summary.json` | counts, events, every parameter and hash, linker name and capabilities, code tables |
| `tracking/<layer>/detections.csv` | the detection table as CSV, for quick plotting |
| `tracking/<layer>/tracks.csv`, `graph.json` | napari `Tracks` layer data and parent graph |
| `tracking/<layer>/ctc/res_track.txt` | Cell Tracking Challenge track file, plus `labels.csv` mapping CTC labels to track ids |

At the collection root, `tracking/<layer>-index.{json,csv}` has one row per
series: its key, result path, counts, parameter hash and status.

Tables are native object tables (`crates/ngff-object-table`):

- each column is a Zarr array at `tables/<layer>/columns/<name>`;
- the spatial index is `(t, y, x)` with a time tile of one, and its occupancy
  pyramid never sums across frames (`spatial_index.downsample = [false, true,
  true]`).

**The downstream contract:** group `tables/<layer>` by `track_id`, order by
`t_index`, and plot any measurement column. Join fields through the series key
in the collection index. Lineage lives in the tracks and links tables, so a
future division-aware linker only adds rows.

```python
import pandas as pd
d = pd.read_csv("A/1/0/tracking/cells/detections.csv")
for track, rows in d.sort_values("t_index").groupby("track_id"):
    plt.plot(rows.elapsed_seconds, rows.measure_mean)
```

Events are **observations**:

- `start_reason`: `first_frame`, `unmatched`, `frame_gap`, `division` or
  `merge`;
- `end_reason`: `last_frame`, `unmatched`, `frame_gap`, `division` or
  `merge`;
- whether each endpoint is near the border.

Death is not inferred: linkage alone cannot tell it from a cell leaving the
field or a missed segmentation.

## Resume

A run that stops part-way keeps `labels/.<layer>-blockflow-work/`. The next run
with the same parameters and input:

- skips every frame whose `manifest.json` is valid, meaning the parameter hash,
  input hash and rows hash all match (`--verify-planes` also re-hashes the
  label plane);
- reruns every other frame from a cleared plane.

Chunk presence is never treated as completion, because background chunks are
not stored. Changing any segmentation or measurement parameter invalidates
every frame. Tracking parameters do not, so relinking needs no
re-segmentation.

## Comparing with `laptrack`

`scripts/compare_laptrack.py` re-links one series' detections with Python
`laptrack` (frame linking only) using the same cost, gate and fixed birth/death
cost. Run Blockflow with a fixed cost so both use the same model:

```bash
plate-timelapse-tracking run --plate P.zarr --max-distance 14 --alternative-cost 120 ...
python scripts/compare_laptrack.py P.zarr/A/1/0/tracking/cells/detections.csv \
  --max-distance 14 --alternative-cost 120 --spacing 0.65 0.65
```

On four synthetic fields of 20 frames, in which 57% of detections had more than
one candidate inside the gate, the two link sets were identical, and both
matched the ground truth exactly. Agreement is expected only when no two
assignments tie on cost.

## Limitations of version 1

- **No gap closing, division or merging.** `capabilities()` says so, and the
  graph validator enforces it.
- **CTC masks are not written.** Use `res_track.txt` together with
  `labels/<layer>-tracks`.
- **Percentile normalisation reads whole frames.** Fine for plate fields; not
  for whole-slide frames.
- **Float input images must be non-negative.** `InstanceSegment` accumulates
  in fixed point.
- **Overlap costs compare native pixels.** Do not combine them with drift
  correction once it exists.
