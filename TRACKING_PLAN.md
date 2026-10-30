# Plate Time-Lapse Tracking: Plan (rev. 5)

Goal: add `examples/plate-timelapse-tracking/` for multiwell plates containing
many 2-D time-lapse fields, usually with two channels. Fields and frames are
processed with bounded parallelism. The example should:

1. discover every field in the plate without losing its plate/well/acquisition
   relationship,
2. segment cells in every frame with StarDist or Cellpose,
3. link detections into one independent track graph per field,
4. measure every detection in a second channel with the existing
   CellProfiler-style `Measurements` builder, and
5. write labels, tables and collection metadata from which a downstream tool
   can plot each cell's measurements over time.

Version 1 assumes that cells do not divide. Its graph and output schemas must
already be able to represent later division, merge and gap-aware linkers.
Biological events such as death must not be inferred from linkage alone.
Tracking sits behind a trait so more algorithms can be added; version 1
implements one.

## Implementation status (2026-10-02)

Implemented and tested against synthetic data; waiting for representative
plate data.

| plan item | where | status |
|---|---|---|
| A1 collection descriptors, HCS plate, manifest | `src/collection.rs` | done; `tests/collection.rs` |
| A2 frame views, virtual singleton `z` | `zarr_env::SpatialAxis`, `time_series::TimeSeriesImage` | done; `tests/time_series.rs` |
| A3 output target into a staged plane, read-back, clear, hash | `ZarrEnvironment::with_output_target`, `time_series::LabelStack` | done |
| A3 `InstanceSegment` id offset | `InstanceSegment::with_id_offset` | done; `tests/model_segment_labels.rs` |
| A3 durable frame artifacts, resume | example `frame.rs` | done |
| A4 per-frame label pyramids, space only | `label_pyramid::downsample_label_plane` | done |
| B bounded scheduling across series | example `pool.rs`, `run.rs` | done (frames, then series; no pipelining) |
| C1–C5 tracking module | `src/track/` | done; 11 unit tests |
| D tables, per-axis occupancy downsampling | `ngff-object-table` `spatial_index.downsample`, `object_table::write_object_rows`, example `tables.rs` | done |
| D exports | `track::export`, example `series.rs` | CTC `res_track.txt` and napari CSV; **CTC masks and GEFF not written** |
| D collection index | example `run.rs` | done (`tracking/<layer>-index.{json,csv}`) |
| E per-frame measurements, normalisation | example `frame.rs`, `segment.rs` | done (fixed, movie and frame percentiles) |
| drift correction | `track::drift` (phase correlation + point-vote refinement), example `--drift-correction` | done; needed on the first real plate (~30 µm drift) |
| gap closing, splitting, merging (lineage) | `track::segment::SegmentLapLinker`; example `--gap-closing/--splitting/--merging` | done; edge sets identical to `laptrack` on synthetic plates with divisions |
| sparse LAP solver | `track::lap::assign_sparse` | done; replaced per-component dense solves after a 4,400-node component took minutes on real data |
| `laptrack` comparison | `scripts/compare_laptrack.py` | done; identical links on 4 ambiguous synthetic fields |
| CTC real-data scoring | none | **not run** (needs the CTC tools and data) |

## Data model: plate, series and frame

There is a collection hierarchy above the pixel arrays:

```text
plate
  acquisition (optional)
    well (row, column)
      field / image / series
        frame t
          channel c
            spatial image [1, Y, X] now; [Z, Y, X] later
```

Here **series** means one field imaged repeatedly over time. It is a collection
key, not another pixel-array axis. Tracking never links different fields or
wells unless a future workflow explicitly asks for that. The stable series key
is `(plate_id, acquisition_id?, well_row, well_column, field_id)`; a frame key
adds `t`.

The preferred representation is one OME-NGFF HCS plate group:

- plate metadata enumerates wells,
- each well's `well.images` enumerates its field image groups and optional
  acquisition IDs, and
- each field image is normally a `t,c,y,x` multiscale image (or `t,c,z,y,x`
  later).

This preserves the relationships without adding a `series` dimension to each
array. OME-NGFF's optional `OME/series` list is a flat catalog of image paths;
it may help discover non-HCS collections or mirror HCS images for readers that
do not understand plates, but it is not a pixel axis and does not replace the
plate/well hierarchy.

Separately stored Zarr images are also supported through an explicit collection
manifest. The manifest records a collection/plate ID and, for every field, its
source URI or relative path, acquisition, well row/column, field ID and channel
mapping. Relative paths are preferred for portability. A directory scan is not
allowed to invent these relationships. When conversion is possible, writing a
conforming HCS plate is preferable; otherwise the manifest is the authoritative
relationship record and is copied into the result provenance.

## Time is not a Blockflow grid axis

Rev. 1 put `t` into slot 0 of the rank-3 block grid. That was the wrong
abstraction:

- Segmentation and measurement are per frame. No pixel op in version 1 reads
  across time, so temporal reach buys nothing.
- Linking works on detection rows rather than a pixel grid.
- Treating axis 0 as time would make spatial connectivity, perimeter, Feret,
  moments, distance and pyramid operations silently wrong unless every one
  special-cased it.
- A field is modest in size (about 2k x 2k). Scale comes from
  wells x fields x frames x channels, so it is a collection scheduling problem.
- Future time plus 3-D data needs all three Blockflow axes for `z,y,x`.

A frame is always exposed to Blockflow as rank-three `[1,Y,X]` for 2-D or
`[Z,Y,X]` for 3-D. Cross-frame information flows as durable rows and pair
artifacts at the series level, never through pixel reach.

`t` remains a storage/table axis. Per-field labels are stored as `[t,y,x]`
(later `[t,z,y,x]`), and detection tables carry `t`. If a future pixel
algorithm really needs temporal neighbours, supply neighbouring frames to a
frame operation explicitly or add a distinct temporal abstraction; do not
reinterpret the spatial grid.

| level | unit | machinery |
|---|---|---|
| spatial | one frame, `[1,Y,X]` or `[Z,Y,X]` | existing Blockflow plans and ops |
| series | ordered frames for one field | frame artifacts, pair stages and tracking |
| collection | acquisitions, wells and fields | HCS metadata or explicit manifest; bounded scheduling and indexes |

## Existing pieces

| need | existing piece |
|---|---|
| per-frame tiled segmentation | `model_segment::InstanceSegment` with StarDist and Cellpose backends |
| measurements in another channel | `ops::measure::Measurements::for_labels(..).intensity(..)` |
| selecting `c` and `t` when a real `z` axis exists | `zarr_env::VolumeSelection` / `OmeVolumePyramid::open` |
| object-table storage | `object_table`, `crates/ngff-object-table` |
| translation drift estimation | `ops::align` |

Selection without a real `z` axis, writable time-plane publication,
collection enumeration and multi-run row assembly are new work; the current
APIs must not be described as already supporting them.

## A. Collection I/O (`ome_zarr`, library)

### A1. Collection descriptors

Add a common read model:

```rust
struct SeriesKey {
    plate_id: String,
    acquisition_id: Option<u64>,
    well_row: String,
    well_column: String,
    field_id: String,
}

struct SeriesRef {
    key: SeriesKey,
    image: ImageLocation,
    channels: ChannelMapping,
    frames: Vec<FrameDescriptor>,
}

struct FrameDescriptor {
    t_index: u64,
    elapsed_seconds: Option<f64>,
    channel_planes: ChannelPlanes,
}
```

`Plate::open(root)` reads OME-NGFF `plate` and `well.images` metadata, validates
referenced paths and acquisition IDs, and returns `SeriesRef`s in a stable
order. It must retain both well names and numeric indices where available.

`CollectionManifest::open(path)` produces the same descriptors for separate
Zarr stores. Validate uniqueness of `SeriesKey`, existence/reachability of each
image, channel mappings and safe relative paths. Do not merge acquisitions
merely because they share a well and field number.

Frame descriptors make the acquired timeline explicit rather than assuming
that every channel has the same dense `0..T` range. Require strictly increasing
`t_index` and, when present, finite increasing timestamps. Version 1 requires
the segmentation and requested measurement channels for every processed frame;
it fails validation rather than silently pairing unequal channel sequences.
An explicit later policy may allow missing measurement values.

Uneven time intervals are supported: displacement gating may be specified as a
maximum speed times elapsed time, and the elapsed interval is recorded on each
edge. A missing whole-frame index is a gap. Because version 1 has no gap
closing, it ends tracks before the gap and starts new tracks after it; a later
gap-capable linker may connect them with `dt > 1`.

### A2. Frame views and virtual singleton z

`FrameRef` identifies `(SeriesRef, channel, t)` and attaches one rank-three
spatial image. Existing selection requires three distinct real source axes, so
support for `(t,c,y,x)` needs an explicit mapping such as:

```text
[VirtualSingleton, Source(y), Source(x)]
```

rather than a fake source-axis number. Test `tcyx`, `tczyx`, `cyx` and `zyx`,
including projected physical transforms and rejection of unsupported axes.

### A3. Direct staged plane output and durable frame artifacts

Create one hidden, field-owned staging label group with final shape `[t,y,x]`
(later `[t,z,y,x]`) and a stored time chunk of one. Each frame job writes its
final-ID label pixels directly into its selected time plane. The completed
group is renamed into place only after all frames, pyramid levels and metadata
validate, following the existing hidden-work/publish pattern used by the
OME-Zarr segmentation examples.

This requires two narrow library changes:

- `InstanceSegment` accepts a checked ID offset which is applied identically to
  emitted rows and nonzero label pixels; frame `t` uses `t * Z * Y * X`, and
- an output selection can bind a rank-three Blockflow result to one plane of a
  pre-created higher-rank staging array. It is an output owned by the run, not
  an attached input made writable.

Thus labels are generated once; there is no local label array followed by a
full copy/relabel pass. Since the time chunk is one, concurrently processed
frames never update the same compressed label chunk.

A frame job contains two ordered Blockflow executions:

1. segmentation writes final-ID labels into the selected staging plane and
   completes all of that plane's writes;
2. measurement attaches that same staged plane back as a read-only
   `[1,Y,X]`/`[Z,Y,X]` label source, attaches channel 2 for the same frame, and
   runs `Measurements::for_labels(..)`.

The selected-output facility therefore needs a matching selected-input view of
the still-hidden staging group. The measurement execution starts only after
the segmentation execution has completed and its writes are visible; it never
reads a plane while that plane is being written. Other frame jobs may safely
write or read different `t` chunks concurrently. Join segmentation rows and
measurement rows by their already-final `detection_id` before publishing the
frame row artifact.

Each successful frame job also atomically publishes a private row artifact:

- per-frame detection and measurement rows, already using final field IDs, in
  a stable intermediate format,
- source identity, series key, `t_index`, dimensions, physical transforms,
- segmentation/measurement configuration and parameter hash, and
- a completion manifest written last.

Resume checks validate the completion manifest and parameter/input hashes;
the existence of a directory or some Zarr chunks is not completion. Temporary
artifacts are renamed into place only after validation. The series stage reads
these artifacts, so an interrupted plate run does not depend on rows retained
in memory.

On resume, a valid completion manifest is the only skip condition. It is
written last, after the plane's writes and the row artifact are complete, and
records a content hash of the frame's label plane; a resume may verify that
hash. Chunk presence is never used as evidence of completion, because Zarr may
omit chunks that equal the fill value, so a correctly completed frame can have
missing chunks.

A frame without a valid manifest is rerun from a cleared plane: delete every
stored chunk of that `t` plane (and its partial row artifact) before
segmentation writes again. Otherwise a chunk written by the interrupted run
but not rewritten by the rerun (skipped as empty, or different because inputs
or a nondeterministic backend changed) would leave stale pixels. Clearing is
safe because the time chunk is one, so no other frame shares those chunks.
Publication must be idempotent and reject incompatible existing staging
metadata. Attached source arrays remain read-only.

### A4. Collection-aware label pyramids

Do not pass `[t,y,x]` to the existing rank-three spatial label-pyramid builder,
because it would treat `t` as `z` and write `z,y,x` metadata. Instead, reuse the
selected-input and selected-output plane views from A3: for each frame and
level, read plane `t` of level `k-1` in the staging group and write plane `t` of
level `k`, with nearest-neighbour downsampling of the spatial axes only. There
is no separate per-frame pyramid followed by a publish/copy step. Level
metadata preserves the source axis names (`t` included), units and coordinate
transformations, with `t` scale unchanged at every level. Frames are
independent here too and can run concurrently; level `k` of frame `t` depends
only on level `k-1` of the same frame.

## B. Series and collection execution

Keep orchestration in the example until another collection workflow needs it.
Version 1 uses a bounded worker pool rather than a general job-DAG framework:

1. enumerate all series in stable key order;
2. submit frame jobs across all wells and fields with configurable global
   concurrency; model backends are loaded once and shared through their `Arc`s;
3. publish durable frame artifacts independently;
4. build adjacent-frame pair artifacts when both frames are complete;
5. fold each field's artifacts into its independent `TrackGraph` and outputs;
6. build the collection index and summary after every requested field is
   complete.

Parallelism across fields and frames is expected to dominate for small images,
but the scheduler must bound total inference concurrency and memory rather than
starting the whole plate at once. Pair and series tasks may pipeline later; a
custom DAG is warranted only after measurement shows a benefit. Distributed
execution can eventually hand out frame or field keys using the same durable
artifact contract. Blockflow's existing `(block,phase)` `TaskGraph` is not
reused for collection scheduling.

### Detection identity

`InstanceSegment::identify` returns a local ID in `1..=Z*Y*X`; zero remains
background. Convert it with checked `u64` arithmetic:

```text
detection_id = checked_mul(t, Z*Y*X) + local_id
```

Pass that value as the frame's `InstanceSegment` ID offset so its rows and
staged label plane already contain final field IDs. Within a field this is
deterministic and needs no counter. Across a collection the full identity is
`(SeriesKey, detection_id)`; do not pretend the numeric ID alone is
plate-global. Reject dimensions/frame counts whose identity range overflows
`u64`.

## C. Tracking module (`src/track/`, library)

Tracking is pure rows-in, graph-out code:

```text
track/
  mod.rs         Detection, CandidateSet, Linker trait, TrackGraph
  candidates.rs  gated candidate edges between frames
  lap.rs         v1 frame-to-frame LAP linker
  graph.rs       validation, chains and observational events
  export.rs      tables, CTC and napari-style CSV (GEFF later)
```

### C1. Detection and coordinate spaces

`Detection` holds `detection_id`, `t_index`, optional `elapsed_seconds`, native
position `[f64;3]`, registered position `[f64;3]`, size and configured optional
features. For 2-D, `z=0`. Native coordinates address labels and tables;
registered coordinates are in a common physical series space and are used for
distance gating. Carry physical spacing per series; anisotropic `z` then needs
no tracker change.

If drift correction is disabled, registered equals native. If enabled, store
the transform for every frame. Overlap costs must compare labels in the same
registered coordinate system; until shifted-label overlap exists, reject the
combination of drift correction and overlap cost rather than mixing coordinate
systems.

### C2. Candidate edges

For every detection at `t`, find detections at the next consecutive frame index
within `max_distance` in registered physical space. With timestamps, gating may
instead use `max_speed * elapsed_interval`. Version 1 does not create candidates
across a missing frame index. Later linkers may request through
`t+1+max_gap`. Use a deterministic spatial grid or k-d tree and sort candidate
edges by IDs before solving.

Optional overlap features (`IoU`, shared pixels) are pair-stage artifacts. For
small fields read the two planes whole; for large fields compute them with a
spatial Blockflow operation using the second plane as a supplied input. Time
still does not become grid reach.

### C3. Linker extension point

```rust
pub trait Linker: Send + Sync {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> LinkerCapabilities;
    fn link(
        &self,
        detections: &DetectionSet,
        candidates: &CandidateSet,
    ) -> Result<TrackGraph>;
}
```

Future implementations may add gap closing, division/merge LAP, overlap
greedy, min-cost flow or Bayesian tracking.

### C4. Version-1 `FrameLapLinker`

- Cost is squared registered physical displacement, optionally plus weighted
  `1-IoU` and relative size change. Validate that every cost and parameter is
  finite and nonnegative; gate beyond `max_distance`.
- Birth and death have explicit alternative costs in compatible units.
- Split the gated bipartite candidate graph into connected components. Solve
  unmatched alternatives locally so dummy birth/death nodes do not reconnect
  otherwise independent components. Handle isolated detections directly.
- Prefer a tested permissively licensed Hungarian/Jonker-Volgenant
  implementation. Write one only if available crates fail the required API or
  determinism contract.
- Sort nodes and edges and use a solver with deterministic behavior for a fixed
  ordered cost matrix. Promise that identical inputs, parameters and solver
  version produce identical output; do not claim a lexicographically minimal
  assignment among equal optima. If cross-solver tie equivalence later matters,
  add a proved integer quantisation/secondary-objective scheme rather than an
  ad-hoc floating epsilon.
- Version 1 has no gaps, division or merge; `capabilities()` reports that.

### C5. Track graph and events

Nodes are detections; edges contain `from_id, to_id, dt`, optional
`elapsed_seconds`, `cost` and `kind = continue | divide | merge | gap`.
Validate that the graph is acyclic, time increases on every edge, IDs exist,
and in/out degrees conform to linker capabilities.

Tracks are maximal unbranched chains with `track_id, t_start, t_end, first_id,
last_id, n_detections, parent_track_ids, child_track_ids`. `track_id` is the
first detection ID and is stable within its `SeriesKey`.

Events record observations rather than unsupported biological conclusions:

- `start_reason = first_frame | unmatched | frame_gap | division | merge`,
- `end_reason = last_frame | unmatched | frame_gap | division | merge`, and
- `border_state = near_border | interior` at the start/end.

Every track ending immediately before a missing frame index gets
`end_reason=frame_gap`; every track beginning immediately after it gets
`start_reason=frame_gap`. These are observational markers, not links, and let a
future gap-closing linker find the eligible endpoints without confusing them
with ordinary unmatched detections.

`likely_entered_fov` and `likely_left_fov` may be derived annotations from the
border rule. Record the border-distance threshold and its physical unit in the
tracking parameters. Death, true appearance/disappearance and segmentation
misses remain unknown unless a separate classifier supplies evidence.

## D. Outputs

### Per field/image group

| path | content |
|---|---|
| `labels/<layer>/` | `[t,y,x]` (later `[t,z,y,x]`) `uint64` detection IDs; only spatial axes are downsampled |
| `labels/<layer>-tracks/` | optional pixels relabelled by `track_id` through a per-frame LUT |
| `tables/<layer>/` | one row per detection with `detection_id,t_index,elapsed_seconds,z,y,x,size,track_id,...measurements` |
| `tables/<layer>-tracks/` | one row per maximal unbranched track |
| `tables/<layer>-links/` | one row per graph edge |
| `tracking-summary.json` | counts, timings, source identity, transforms and complete parameters |

Frame artifacts are private resumability data, not the public table layout.
The series stage reads all completed frame row artifacts, whose IDs are already
final within the field, adds the timeline columns and writes one final detection
table per field. The current
`finalize_object_table` consumes one execution environment, so version 1 should
write the final table from the stable intermediate rows directly. Generalize
fragment finalization only if another use case needs multi-run assembly.

The object-table schema currently has no arbitrary provenance extension.
Either add a namespaced extension field to its metadata or keep complete
linker/capability/parameter metadata in `tracking-summary.json` and link it
from the table. Do not silently add undeclared attributes.

Index the detection table by `(t_index,y,x)` with time tile size one and the
usual spatial tile sizes. The current `SpatialIndexSpec` accepts arbitrary axis
names, but its occupancy pyramid halves and sums **every** axis at each level.
Used unchanged, its coarse levels would mix counts from neighbouring frames.

Before calling `(t,y,x)` indexing viewer-safe, extend the object-table index
settings with per-axis downsampling flags (backward-compatible default: all
axes downsample). This table uses `[false,true,true]`: occupancy levels retain
their full `t` extent while halving only `y,x`, just as label pyramids preserve
time. Update occupancy shape generation, summation, metadata validation,
round-trip tests and region reads accordingly. A viewer can then fetch one
frame's detections and occupancy at every spatial level without scanning the
whole field table or mixing time points.

The format has no secondary index on `track_id`, so plotting a single track
scans the field table in version 1. That is acceptable at the expected scale
of roughly one million rows per field; add a track-to-row index only if
measurements show it is needed.

Downstream contract: within a field, group the detection table by `track_id`,
order by `t_index`, and plot a measurement column. Join fields through `SeriesKey`
from the collection index. Lineage lives in the tracks and links tables, so a
division-aware linker adds graph rows without changing detection identity.

### Collection root

Write `tables/<layer>-index/` (or an equivalent manifest-backed index) with one
row per series:

```text
plate_id, acquisition_id, well_row, well_column, field_id,
source_uri, result_path, n_frames, n_tracks, n_detections, params_hash, status
```

For an HCS input, derived labels/tables should live under each field image when
the destination is writable, and the plate-root index summarizes them. For
read-only or separately stored inputs, write a distinct result collection with
the copied input manifest plus relative links/paths to per-series results. Do
not modify source data merely to manufacture a common root.

Exports are CTC (`res_track.txt` plus masks) for TRA/DET scoring and a
napari-style CSV. Check the current GEFF specification before adding GEFF.

## E. Per-frame measurements

After the segmentation plan has completed its selected-plane write, start a
separate measurement plan with the staged label plane attached read-only as
`labels_t` and the matching measurement channel as another input. Use
`Measurements::for_labels(labels_t).intensity(channel2_t, ...)`; these are the
full CellProfiler-style measurements, not only the sum/min/max accumulated by
`InstanceSegment`. Optional background subtraction uses `ops::background`.

Expose fixed movie-wide normalization and per-frame percentiles, recording the
choice and exact values. Movie-wide values require a deterministic prepass (or
explicit user-provided values) whose completed artifact is shared by every
frame; they must not be recomputed independently by workers.

## End-to-end pipeline

1. Open one explicit field for the core gate, or open an HCS plate/collection
   manifest and enumerate stable `SeriesRef`s at the collection gate.
2. Validate axes, channels, dimensions, physical units, frame counts and output
   identity ranges before starting expensive work.
3. Optionally compute deterministic per-series normalization and drift
   transforms.
4. Process frames across all wells/fields with bounded concurrency; atomically
   publish frame artifacts.
5. Build pair artifacts and candidate edges within each series only.
6. Run `FrameLapLinker`, validate `TrackGraph`, classify observational events
   and publish per-field labels/tables/pyramids atomically.
7. Publish the plate/collection index and summary last.

## Delivery scope

The architecture supports plates and manifest-backed collections, but those
readers are not prerequisites for proving tracking. Deliver in two gates:

- **Core gate:** one explicitly selected real field, its complete time series,
  segmentation, measurements, final-ID staged labels, tracking and tables.
- **Collection gate:** HCS enumeration, manifest-backed separate stores,
  multiple acquisitions, plate-wide scheduling/indexing and exhaustive
  interruption testing. Start this only when representative input establishes
  which layout is actually required.

The core gate still tests a basic interrupted resume because it defines the
staging contract. Broad failure injection across many wells and publication
points belongs to the collection gate.

## Validation

1. **Synthetic rows:** trajectories, births, unmatched ends, border cases and
   later branch/merge graph fixtures. Most tracker tests need no pixels.
2. **Synthetic collections (collection gate):** process multiple wells and
   fields together; assert that no cross-series edges appear and that serial
   and parallel runs are identical.
3. **Timeline validation:** reject mismatched required channels; test uneven
   timestamps and verify that missing frame indices split tracks with matching
   `frame_gap` end/start reasons in version 1.
4. **Storage layouts (collection gate):** equivalent `tcyx` and `tczyx` HCS
   fields plus separate Zarr stores joined by a manifest produce equivalent
   logical outputs.
5. **Resume/failure injection:** at the core gate, stop and resume one frame,
   including a rerun whose output has fewer stored chunks than the interrupted
   write (stale-chunk check) and a completed frame with omitted fill-value
   chunks (must be skipped, not rerun);
   at the collection gate, inject failures during frame writes, pair stages and
   series publication, then compare with a clean run.
6. **Invariance:** identical results across block sizes, worker counts and
   frame concurrency.
7. **Table occupancy:** with a `(t,y,x)` index, verify every occupancy level
   preserves the `t` extent and never sums objects from neighbouring frames.
8. **Measurement staging:** compare measurements read back from a staged time
   plane with the same measurements on a standalone label frame.
9. **Reference:** compare frame-LAP output against Python `laptrack` on
   identical detections without equal-optimum ties. Test tie cases separately
   for repeatability of this implementation, not equality with `laptrack`.
10. **Real data:** use Cell Tracking Challenge 2-D data such as
   Fluo-N2DH-GOWT1; use dividing Fluo-N2DL-HeLa to document the version-1
   limitation, and score via the CTC export.

## Order of work

1. Implement `Detection`, candidates, validated `TrackGraph` and frame LAP
   against synthetic rows.
2. Define the durable row-artifact schema, parameter/input hashing and checked
   `InstanceSegment` ID offsets.
3. Implement virtual-singleton-z reads and selected output into a hidden
   field-level `[t,y,x]` staging group.
4. Run segmentation then measurement by reading the staged plane back; run one
   synthetic time-lapse field end to end, including interrupted resume.
5. Add per-axis object-table occupancy downsampling and then final
   detection/track/link tables and provenance.
6. Add per-frame spatial label pyramids published into time-bearing storage.
7. Run the core gate on one representative real field; add CTC and `laptrack`
   comparisons.
8. When real collection layout requires it, add HCS enumeration and/or
   collection-manifest input, bounded plate scheduling and the collection
   index.
9. Add multiwell failure-injection and layout-equivalence tests at the
   collection gate.

## Open questions requiring representative data

- Is the source conforming OME-NGFF HCS, a flat NGFF collection, or separate
  Zarr stores requiring a manifest/conversion?
- Does one field image contain the complete time axis, or does acquisition
  produce one Zarr store per frame?
- Are there multiple acquisitions per well, and are repeated field IDs stable
  across them?
- What are the frame count, interval and typical displacement relative to cell
  size? This decides whether centroid-only linkage is sufficient.
- Is segmentation performed on one channel and measurement on the other?
- Is there stage drift, and must overlap costs be supported when it is enabled?
- Which measurement families are required: intensity only, or also shape and
  texture?
