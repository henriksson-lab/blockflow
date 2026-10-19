# Vessel graph

Light-sheet vasculature to a vessel graph. The **segmentation** half turns one
or more vessel channels of an OME-Zarr image into a binary mask and its
skeleton, and also makes tract maps (a fibre-tract channel to a mask and a
point density). The **graph** half turns the skeleton into an annotated vessel
graph. Every step that reads a volume is planned over blocks with blockflow
(the skeleton's thinning holds the volume, two bytes a voxel, only where that
fits `--thin-memory-mb`); see [Out of core](#out-of-core).

Originally developed as part of a ClearMap reimplementation.

## Pipeline

### Segmentation (`src/segment`)

One plan per run:

```text
channel k:  raw -> binarize -> smooth -> fill [-> deep fill] --\
                                                                +-> OR -> smooth -> fill -> mask -> skeleton
channel j:  raw -> binarize -> smooth -> fill [-> deep fill] --/
```

With one channel the channel's own filled mask is the answer; the second
smooth-and-fill runs only after a union. The deep fill (a CNN that fills
hollow vessel lumens, [below](#deep-vessel-filling)) needs the `deep-fill`
cargo feature and is on by default only in a build that has it.

#### Channel roles and parameters

Each channel has a **role**, which picks its binarisation defaults; they are
the reference pipeline's vasculature configuration (`vasculature.yml`, version
3.1):

| role | clip range | deconvolve threshold | equalize, vesselize arms |
|---|---|---|---|
| `vessel` (all-vessel stain) | `(350, 1000)` | 750 | on |
| `artery`, `vein` (large vessels) | `(1000, 8000)` | 450 | off |

Lightsheet, median, adaptive and the smoothing/fill after each channel are the
same for every role. `--role vessel,artery` states the roles in `--channel`
order; by default the first channel is `vessel` and every other one `artery`.
`BinarizeParams::default()` is the reference's *library* default, `(400,
60000)` with every arm on, which no role uses. The roles also set the deep
filling's resampling (`vessel` 1, `artery` and `vein` 2, the reference's
`resample_factor`); it is on for every role in a `deep-fill` build and off
otherwise.

Overrides apply in this order: the role's defaults, then a JSON parameter file
(`--params`), then the command-line flags (which apply to every channel). The
file has one entry per `--channel`, every key optional:

```json
{
  "channels": [
    {"role": "vessel", "clip_range": [350, 1000]},
    {"role": "artery", "clip_range": [1000, 8000], "deconvolve_threshold": 450,
     "equalize": false, "vesselize": false, "smooth_passes": 6, "fill": true,
     "deep_fill": true, "deep_fill_resample": 2, "deep_fill_threshold": 0.5}
  ],
  "combined": {"smooth_passes": 6, "fill": true}
}
```

Channel keys: `role`, `clip_range`, `bins`, the arm switches `lightsheet`,
`median`, `deconvolve`, `adaptive`, `equalize`, `vesselize` (an arm switched on
starts at its defaults), `deconvolve_threshold`, `equalize_threshold`,
`vesselize_threshold` (refused for an arm that is off), `smooth_passes` (0 is
none), `fill`, and `deep_fill`, `deep_fill_resample`, `deep_fill_threshold`
(the last two refused where the deep filling is off). Unknown keys are
refused. In the library:
`ChannelRole::{binarize, channel}`, `SegmentParams::for_roles`, and
`SegmentOverrides::{from_json, read, roles, params}` (`src/segment/params.rs`).

The binarisation of a channel is a fan-out and fan-in: a clip, two corrections,
and five arms OR-ed into one sink.

| step | what | blockflow pieces |
|---|---|---|
| clip | raw `(400, 60000)` spread over 4096 bins, rounded; three masks of the raw range: saturated (`>= high`, seeds the sink), not-low (the median's population), in-range (the population of the window estimates) | `voxelwise` (`WidenOp`, `VoxelwiseMapOp`, `NarrowOp`, `VoxelwiseMaskOp`) |
| lightsheet | `v - min(v, min(p25 along a 150-voxel x line, 2 x p25 of a 200 x 200 window))`, both estimates over the in-range voxels; the window on a 25-voxel lattice, stride 2 | `rank::MaskedRankFilterOp`, `local::LocalStatisticOp` (masked, sampled), `voxelwise::FnCombine` |
| median | 3 x 3 x 3 over the not-low voxels; dark voxels left at zero | `MaskedRankFilterOp` |
| deconvolve arm | the sink's voxels blurred (σ 10 in y, x; flat in z) and subtracted outside the sink; residual `> 750` | `smooth::SmoothOp` (reflecting boundary), `FnCombine`, planned with `PlanBuilder::partition` |
| adaptive arm | the deconvolved volume against a local isodata threshold (3 x 250 x 250 window, lattice 3 x 50 x 50) | `local::AdaptiveThresholdOp`, `Statistic::Isodata` |
| equalize arm | `v * min(1 / max(p40, 1), 1.5 / p97.5)` over a 5 x 200 x 200 window of in-range voxels (lattice 5 x 50 x 50); `> 1.1` | `LocalStatisticOp` (masked), `normalise::bounded_gain_value` |
| vesselize arm | the equalized volume clamped at 1.1 and rescaled to the bins; minus its median over a 30 x 30 disk of in-range voxels; Hessian tubeness at σ 1; `> 120` | `MaskedRankFilterOp`, `ridge::RidgeFilterOp` with `RatioResponse`, planned with `partition` |
| smooth | six passes of a topology-preserving 3 x 3 x 3 configuration rule (spurs removed, pinholes filled), derived here from its definition | `configuration::ConfigurationPassOp` |
| fill | background not reaching the outside of the volume becomes foreground | `fill::append_connected` (two fragment phases, merge in a barrier) |
| union | OR of the channels | `voxelwise::LogicCombine` |
| skeleton | the volume's faces cleared, then directional thinning to a fixed point, one pass per round, rounds exchanged through a fixed halo | `directional::directional_passes_into` inside an `IterativeOp` |

Three arms change what a later arm reads: the lightsheet correction is under
the median (so under everything), deconvolution replaces the volume the
adaptive arm thresholds, and equalisation replaces the volume the vessel arm
filters. Turning one off reroutes the arms after it to the median. Every arm
and both corrections can be turned off.

### Deep vessel filling (`src/segment/deep_fill`)

The reference's 3.1 vasculature pipeline runs a trained 3-D CNN after each
channel's binarise, smooth and fill (`deep_fill`, on by default in its
`vasculature.yml`): it fills the hollow lumens that large vessels leave in a
binary mask, whose walls stain and whose centres do not. This is that network
(`VesselFillingNetwork`, 276k parameters), loading the reference's trained
weights, with the reference's inference:

- the answer is the mask OR (probability >= 0.5), the reference's
  `max(cnn, mask) >= threshold`;
- `resample` max-pools the mask before the network and upsamples the
  probability back (trilinear), 1 for the all-vessel channel and 2 for the
  large-vessel channels, as in the reference's configuration;
- the network runs on the mask in the reference's own axis order (its arrays
  are `[x, y, z]`; the kernels' spatial axes are reversed at load).

The architecture: max pool, `conv1` (7^3, stride 2), max pool, `conv2` (5^3),
a depthwise-separable 3^3, `convbin1` (3^3), trilinear x2, the transposed
`conv3` (7^3, stride 2), trilinear x2, the input concatenated back, `conv4`
(3^3), `sigmoid(4 (c - 0.5))` and a two-way softmax. The encoder runs on burn's
CPU backend (`flex`, as the `yolo3d` examples); the two decoder layers, which
are most of the arithmetic and which burn's convolutions handle badly (a
column buffer of 9.7 GB per block for the transposed convolution, 73 s for the
last one on the crop below), are a polyphase convolution as one burn matrix
product per batch of rows and a direct convolution over z planes
(`deep_fill/network.rs` says what was measured).

**Planned per block** (`DeepFillOp`, a `BlockOp`): a halo of 64 voxels per side
(128 at resample 2), from the network's measured reach (an output voxel
depends on the input within 55 voxels, 112 at resample 2) plus the slack
for starting the network at a multiple of its stride (8, 16), which makes the
answer depend on a voxel's position modulo the stride. Every block starts at
such a multiple, so every block computes what one pass over the whole volume
would, and the result does not depend on the grid (to 1e-7 in the
probability, voxel for voxel in the mask, on seeded random weights:
`deep_fill_blocks`). The deep-fill phase has its own block grid
(`--deep-fill-block`, default 256,512,512) and blocks run through the network
one at a time, each on every core. Where the reference ORs overlapping blocks
of up to 500 voxels (1000 at resample 2) with an overlap of 50 (100), less
than the network's reach, this is the single-pass answer. At the volume's far
faces the network's output is up to 7 voxels short (15 at resample 2), as the
reference's is; those voxels keep the mask, where the reference sets them to
zero.

**Dropout.** The reference never puts its network in evaluation mode, so its
three dropout layers (p 0.15) stay on at inference: its filled mask changes
from run to run, and it fills about half again as many voxels as the network
in evaluation mode does (see [Validation](#deep-vessel-filling-against-the-reference)).
This port is the network in evaluation mode, deterministic.

**The weights** are not part of this repository. They are the reference's
`ClearMap/ImageProcessing/machine_learning/vessel_filling/resources/cnn_binary_vessel_filling.pth`
(a PyTorch state dict, 1.1 MB), which the reference ships inside its GPLv3
repository with no separate licence or terms for the weights (its `setup.py`
says MIT, its `LICENSE` GPLv3); with that unclear for a trained model
distributed in an MIT crate, they are read at run time from a path you give.
Convert them once to safetensors (needs torch):

```sh
python reference-clearmap/convert_weights.py \
  .../vessel_filling/resources/cnn_binary_vessel_filling.pth cnn_binary_vessel_filling.safetensors
```

The script checks that the file reads back equal to the state dict. The
reference's checkout may also hold a `.safetensors` next to the `.pth`; it has
the same tensors.

**Build and run.**

```sh
cargo build --release -p blockflow-vessel-graph --features deep-fill
target/release/vessel-graph segment --zarr tile.ome.zarr --channel 0,1 --role vessel,artery \
  --deep-fill-weights cnn_binary_vessel_filling.safetensors
```

In a `deep-fill` build every role deep-fills by default, as the reference
does: give the weights with `--deep-fill-weights` or `$VESSEL_FILL_WEIGHTS`, or
pass `--no-deep-fill`; a run with neither is refused, naming both. Without the
feature the deep filling is off by default and asking for it
(`--deep-fill-weights`, or `"deep_fill": true` in a parameter file) is refused.
`--deep-fill-threshold` sets the threshold for every channel. The per-channel
mask labels (`<name>-<role>-mask`) are the deep-filled masks. Library:
`segment::deep_fill::{DeepFillParams, DeepFillSetup, DeepFillOutput,
append_deep_fill, deep_fill}` (the last over an in-memory mask, returning the
filled mask or the probability).

### Tract maps (`src/segment/tract.rs`)

Two runs: the first samples the raw channel (one voxel in 1000 by linear index)
and a barrier reduces the sample to the 70th and 99.999th percentiles, which
become the clip range; the second binarises with the clip, lightsheet, median
and vessel arm (threshold 1), turns the mask into points and voxelises them
into a density (points per radius-5 ball).

Pieces: `threshold::SampledValuesOp`, `GlobalThreshold::Percentile`,
`voxelize::VoxelizeOp`.

With `--registration` the mask's points are also mapped into a prepared atlas
(see [Atlas space](#atlas-space-through-a-registration)), labelled there
(region, hemisphere; region 0 outside the atlas) and voxelised on the atlas
grid, as the reference's tract map does (`transform_coordinates`,
`label_coordinates`, `voxelize` with radius 5 atlas voxels).

### Graph (`src/graph`)

Skeleton to an annotated vessel graph. The `graph` subcommand runs stages
1–3 planned over blocks of the stored labels and channels (see
[Out of core](#out-of-core)); the graph itself, from stage 4 on, is resident:

1. **build** — skeleton voxels become vertices and their 26-neighbour pairs
   edges (`ops::coordinates`, `ops::adjacency`);
2. **radii** — per vertex, the distance to the nearest background voxel of the
   vessel mask, by an ordered offset walk (`ops::walk`, `ops::rows::Limit`;
   search limit 50 voxels);
3. **signals** — optional artery/vein intensity and mask read around each
   vertex;
4. **clean** — cliques of branch voxels collapse to one vertex, carrying every
   vertex column;
5. **reduce** — chains of degree-2 vertices contract into edges that keep their
   polylines;
6. **largest component** (`ops::components`' union-find), optional;
7. **length** — per edge, the sum of its polyline's segment lengths: `length`
   in voxels and `length_units` in µm (the voxel spacing per axis);
8. **annotate** — atlas region, hemisphere and distance to surface per vertex
   and edge (`atlas_regions::Annotator`), at `coordinates_atlas`: an atlas
   label already in the sample frame, or a prepared atlas in its own space
   through a registration (`--registration`, with the radii also as `_atlas`
   twins in atlas voxels);
9. **classify** — artery/vein edges by tracing from the edges the artery mask
   covers, over the line graph;
10. **branch density** — branch points voxelised into a density volume
   (`ops::voxelize`), on the atlas grid when there is an atlas.

Columns live at three levels — vertex, edge and polyline point — and travel
through clean and reduce. `VesselGraph::rescale_to_atlas`,
`transform_coordinates` and `scale_radii` move a graph into atlas units.
`transform_coordinates(name, &map)` takes any `PointMap`: an `Affine` between
two lattices, or any closure `Fn(&[[f64; 3]]) -> Result<Vec<[f64; 3]>>`, which
is how a registration's point transform plugs in. The graph is written as NGFF
object tables (below) or JSON (`"format": "vessel-graph", "version": 1`).

## Out of core

Every stage whose memory grows with the volume runs block by block; what is
left resident grows with the vessels (the graph, the tract points of one row
of blocks) or with the atlas.

**`graph`, stages 1–3** (`graph::planned`, `graph::vessel_graph_stored`): one
plan over the published labels and channels, attached as stored arrays.

1. *pack* (`PackMasksOp`, blocks of `--pack-block`, default `64,64,128`, the
   labels' chunk): the `uint64` vessel mask, skeleton and artery/vein masks
   narrowed to one `u8` image, a bit each, reading each label chunk once;
2. *vertex measure* (`VertexMeasureOp`, a fragment phase on blocks of
   `--block`, default `128,256,256`): the packed masks and the channels read
   over the block's core grown by one reach — the radius walk's search limit
   (50) or the widest ball a signal can be read over (the search radius of the
   longest stop: 53 at the reference's 3-voxel artery shift), whichever is
   larger. For every skeleton voxel of the core: the walk
   (`RadiusSearch::stop_at`, the same `ops::walk::walk_from`), each signal's
   maximum over the vertex's ball (`signal::Ball`), and the core's adjacent
   pairs (`ops::adjacency::encode_adjacent_pairs`). Neither the walk nor a
   ball can leave the window inside the volume, so each vertex gets the
   whole-volume answer on every grid (`tests/graph_planned.rs`: one block,
   dividing and non-dividing grids, blocks smaller than the reach, two search
   limits). The rows are merged in coordinate order (the vertex list); the
   pairs by `ops::adjacency::collect_pairs`.

`--workers`, `--cache-mb` (default 256) and `--work` (scratch, removed) do
not change any result. The ball's maximum over a `bool`, `u8` or `u16` window
has a fast path (no per-offset check where the ball's box is inside the
volume and the window), and the ball's offset list holds only offsets within
its radius, sorted once; the answers are the same.

**`segment`, the thinning** (`segment::thin`): the in-plan fixed point is one
iterative phase whose two private buffers are the whole volume (two bytes a
voxel). Over `--thin-memory-mb` (default 2048, a volume of 1 G voxels) the
thinning runs over stored images instead: each round is a pixel phase
(`ThinRoundOp`, one directional pass at the pass's own reach), eight rounds a
plan, then a fragment phase compares the last two images (`ChangedOp`) and
the run stops when the last round changed nothing — the iterative phase's
test, so the skeleton is the same image (`tests/segment_thin_stored.rs`, and
`segment_cli` with `--thin-memory-mb 0`). A block whose window is the one the
previous round read copies its core instead of thinning. Each round is a
read and a write of the stored image and a barrier, which on the osf-sa3x8
crop cost more wall time than the resident phase (see the measurements), so
the resident phase stays the default for volumes it fits.

**`tract`, the points** (`segment::tract::for_each_point_slab`): the mask's
points are streamed one row of blocks at a time (each row sorted, rows in z
order: the whole list's order), mapped and labelled a million at a time, and
the CSV is written as they come. The atlas density is accumulated as a count
per atlas voxel and rendered once, one ball of weight `c` per voxel holding
`c` points, which is the sum of their unit balls exactly (integer counts in
`f64`).

**The rest.** The JSON is streamed (`VesselGraph::write_json`, the same text
as `to_json`), and the tables' rows are reserved at their width. The atlas
volumes (`--registration`, `--annotation-label`) and the branch density are
on the atlas or density lattice, not the image's: a 25 µm atlas is fixed, and
the density's default lattice is the coarsest pyramid level. They stay
resident.

`VESSEL_GRAPH_PROFILE=1` prints, per stage and per planned phase, the wall
and CPU time and the resident set and its peak (it resets the peak through
`/proc/self/clear_refs`, so a profiled run's overall peak is the largest
stage peak, not `time`'s figure).

## Input

An OME-Zarr image with axes `t, c, z, y, x` (or `z, y, x`), `uint16` (or any
type blockflow attaches). The channels of one segmentation must share a shape.
The image must be a local copy: label images are written into it.

## Run

```sh
# vessel mask and skeleton of channels 0 and 1
cargo run --release -p blockflow-vessel-graph -- segment \
  --zarr /path/to/image.ome.zarr --channel 0,1

# a tract map of channel 2, with its points rescaled to an atlas shape
cargo run --release -p blockflow-vessel-graph -- tract \
  --zarr /path/to/image.ome.zarr --channel 2 \
  --points-csv points.csv --scale-to 320,528,456

# the same with a parameter file of per-channel overrides
cargo run --release -p blockflow-vessel-graph -- segment \
  --zarr /path/to/image.ome.zarr --channel 0,1 --role vessel,artery --params params.json

# the vessel graph of the published skeleton and masks, classified with the
# artery mask `segment` published and the artery channel 1's intensity,
# written as tables into the image (and as JSON)
cargo run --release -p blockflow-vessel-graph -- graph \
  --zarr /path/to/image.ome.zarr --artery-channel 1 --out graph.json

# in atlas space through a registration (sample -> prepared atlas), with the
# branch density on the atlas grid
cargo run --release -p blockflow-vessel-graph -- graph \
  --zarr /path/to/image.ome.zarr --artery-channel 1 \
  --registration registration/ --atlas ABA_25um_2017_1_2_3_z0-256.ome.zarr \
  --branch-density branch-density-atlas.ome.zarr

# the tract map's points in atlas space, labelled, and their atlas density
cargo run --release -p blockflow-vessel-graph -- tract \
  --zarr /path/to/image.ome.zarr --channel 2 \
  --registration registration/ --atlas ABA_25um_2017_1_2_3_z0-256.ome.zarr \
  --points-csv tract-points.csv --atlas-density tract-density-atlas.ome.zarr

# segmentation of a crop, copied (in the source's physical frame) to crop.ome.zarr
cargo run --release -p blockflow-vessel-graph -- segment \
  --zarr tile.ome.zarr --channel 0,1 --crop 1600:1728,400:912,: --crop-out crop.ome.zarr

# annotated from an atlas already warped into the sample frame (label images
# in the same OME-Zarr), with the branch density as an image
cargo run --release -p blockflow-vessel-graph -- graph \
  --zarr /path/to/image.ome.zarr --artery-channel 1 \
  --annotation-label atlas --hemisphere-label hemispheres \
  --ontology ABA_annotation_last.jsonl --branch-density branch-density.ome.zarr
```

`segment --help` lists `--role`, `--params`, and the flags that override every
channel: the clip range and bins, `--no-<arm>` switches for every arm, the arm
thresholds, smoothing passes, `--no-fill`, `--keep-border`, and the deep
filling's `--no-deep-fill`, `--deep-fill-weights`, `--deep-fill-threshold` and
`--deep-fill-block`, and `--thin-memory-mb`;
`tract --help` the percentiles, sampling, density radius and the CSV options;
`graph --help` also `--block`, `--pack-block`, `--workers`, `--cache-mb` and
`--work` for its planned stages.
Both take `--block z,y,x` (default `64,256,256`), `--workers`, `--time`,
`--work` (scratch directory, removed afterwards) and `--crop
z0:z1,y0:y1,x0:x1` with `--crop-out <path>`: the crop of the channels is
copied into a new one-level `c, z, y, x` OME-Zarr image (channels renumbered
from 0 in `--channel` order, the source's scale, the translation moved to the
crop's first voxel, so physical positions and any registration of the source
still apply), the run happens on the copy, and the labels are published there;
`graph --zarr <crop-out>` then reads them.

## Output

Into the image's `labels` group, replacing any previous run, each with a label
pyramid matching the image's levels:

| subcommand | label image | values |
|---|---|---|
| `segment` | `<name>-mask` (default `vessel-mask`) | 1 vessel, 0 background |
| `segment` | `<name>-skeleton` | 1 skeleton voxel |
| `segment`, two channels or more | `<name>-<role>-mask` per channel (`<name>-<role><channel>-mask` when two share a role) | that channel's own mask after its smooth and fill |
| `tract` | `<name>-mask` (default `tract-mask`) | 1 tract |
| `tract` | `<name>-density` | mask points within the ball around each voxel |

`tract --points-csv` also writes the mask's voxels as `z,y,x` rows, optionally
scaled to `--scale-to`. The library API (`segment::segment`,
`segment::binarize_volume`, `segment::mask_steps`, `segment::tract_map`) runs
the same plans over arrays in memory, and `segment::segment_plan` /
`binarize_plan` assemble them without running.

### Graph

`graph` reads `labels/<name>-skeleton`, `labels/<name>-mask` and, where they
exist, `labels/<name>-artery-mask` and `labels/<name>-vein-mask` (or the labels
`--artery-mask` / `--vein-mask` name), as the `artery_binary` and `vein_binary`
columns. `--artery-channel` / `--vein-channel` add the raw intensities
(`artery_raw`, `vein_raw`). With an artery mask the edges are classified
(`artery`, `vein`). It writes:

- **object tables** (unless `--no-tables`) under the image's `tables/`, in the
  `ngff-object-table` format, replacing any of the same name:

  | table | identity | rows | columns |
  |---|---|---|---|
  | `<name>-vertices` | `vertex_id` | one per vertex | `z, y, x` (level-0 voxels, spatially indexed), `z_um, y_um, x_um`, `degree`, every vertex column (`radii`, `radius_units`, `artery_binary`, `annotation`, `coordinates_atlas_{z,y,x}`, ...) |
  | `<name>-edges` | `edge_id` | one per edge | `z, y, x` (the polyline's middle point), `source`, `target` (vertex ids), `point_start`, `point_count`, `length` (voxels), `length_units` (µm), `radii`, `radius_units`, `artery`, `vein`, `distance_to_surface`, every edge column |
  | `<name>-edge-points` | `point_id` | one per polyline point | `z, y, x`, `edge_id`, `index` (along the edge, from `source`), every polyline column |

  An object table holds one scalar per row and column, so the polyline is the
  points table: edge `e`'s points are `point_id` `point_start .. point_start +
  point_count`, in order. A per-axis graph column becomes `<name>_z`, `_y`,
  `_x`; region and hemisphere ids are `int64`, flags `uint8`, the rest
  `float64`. `coordinate_space` is `../../` (the image), and each table names
  the other two in `extensions["vessel_graph"]` (`{"version": 1, "role":
  "vertices" | "edges" | "edge_points", "vertices": "../<name>-vertices",
  ...}`);
- **JSON** with `--out`;
- **annotation** with `--annotation-label <label>`: an atlas annotation already
  in the sample frame, as a label image of the same OME-Zarr on any lattice of
  the image's physical frame. The graph's voxels are mapped onto the label's
  lattice (`coordinates_atlas`) and looked up: `annotation`,
  `hemisphere` (`--hemisphere-label`, on the same lattice) and
  `distance_to_surface` (`--distance-label`, or computed from the annotation in
  units of `--distance-unit` µm, default 25, the reference atlas voxel the
  classifier's thresholds are in); `--ontology` and `--level` collapse the
  regions. Annotation runs before classification, which reads the edges'
  distance to the surface;
- **atlas space** with `--registration <dir>` (and `--atlas <prepared
  atlas>`), instead of `--annotation-label`: `coordinates_atlas` are atlas
  level-0 voxels through the registration, `annotation`, `hemisphere` and
  `distance_to_surface` (in atlas voxels: the prepared atlas's field, or,
  when it has none, computed from its annotation in voxels; `--distance-unit`
  does not apply) are read from the prepared atlas, `--level` collapses on the atlas's own tree by tree depth
  (the reference's level), and every radius column gets an `_atlas` twin in
  atlas voxels (µm over the sample voxel, the reference's `_scale`). The
  tables record the atlas space as `extensions["vessel_graph"]["atlas"]`; see
  [Atlas space](#atlas-space-through-a-registration);
- **branch density** with `--branch-density <path>`: a one-level `float32`
  OME-Zarr image of the vertices' balls (`--density-radius`, default 15 voxels)
  on the atlas grid with `--registration` (the reference's branch density),
  on the annotation's lattice with `--annotation-label`, or on the image's
  pyramid level `--density-level` (default the coarsest), with that lattice's
  scale and translation.

### Atlas space through a registration

`graph --registration <dir>` and `tract --registration <dir>`
(`blockflow_atlas_registration::atlas_space::RegisteredAtlas`, shared with
cell-density) take a directory `atlas-registration register` or `chain`
wrote, registered with the sample as the (first) fixed image and the atlas as
the (last) moving image, and the prepared atlas (`atlas-prepare`) the
registration's moving image was. Three lattices are involved:

| frame | what | from |
|---|---|---|
| image | level 0 of `--zarr`, the voxels the graph or tract map is in | the image's OME-Zarr `scale`, `translation` |
| sample | level 0 of the registration's first fixed image (e.g. a 25 µm resample of the signal, or the autofluorescence at its own resolution) — what `PointTransform::apply` takes, whatever level the registration ran on | `PointTransform::sample_frame()`: `registration.json` → `fixed.level0_frame` (the first step's, for a chain) |
| atlas | level 0 of the last moving image = the prepared atlas | `PointTransform::atlas_frame()` |

Image → sample is the change of lattice between the two images' frames
(`physical = translation + index * scale`, voxel centres at integers), so no
scale is entered by hand; it needs both images to carry their physical scales
and to share a physical frame (their translations place them in one space; a
`--crop` copy keeps its source's frame). When the two cover the same box from
one origin this is the reference's `index * resampled_shape / original_shape`
up to the rounding of the resampled shape. A warning is printed when the image
reaches more than a sample voxel outside the sample's box. Sample → atlas is
`PointTransform::apply`.

The atlas has to be the registration's moving image: a shape other than
`PointTransform::atlas_shape()`, or a voxel size other than the moving image's
level-0 scale, is refused, and its reference is compared
voxel for voxel with the moving image named in `registration.json` (refused
if any voxel differs; `--skip-atlas-check` skips the comparison, and an
unreadable moving image is checked by shape and voxel size only). The image's
unit, when it declares one, must be the sample's. `--atlas` defaults to the
moving image when that is itself a prepared atlas. Points mapped outside the
atlas get region 0 and hemisphere -1, as in the reference; the distance to the
surface is read at the nearest edge voxel, as the reference clamps it. The
tables' `atlas` extension records `distance_to_surface_unit: "atlas voxel"`.

For the osf-cztq2 registration
(`/husky/henriksson/lightsheet-reference/registration/ours/auto_to_reference`),
the matching atlas is
`/husky/henriksson/lightsheet-reference/atlas/ABA_25um_2017_1_2_3_z0-256.ome.zarr`:

```sh
S=/husky/henriksson/lightsheet-reference/atlas/source
target/release/atlas-prepare ABA_25um_2017_1_2_3_z0-256.ome.zarr \
  --annotation $S/ABA_25um_2017_annotation.ome.zarr \
  --reference $S/ABA_25um_2017_reference.ome.zarr \
  --hemispheres $S/ABA_25um_2017_hemispheres.ome.zarr \
  --distance-to-surface $S/ABA_25um_2017_distance_to_surface.ome.zarr \
  --ontology ClearMap/Resources/Atlas/ABA_annotation_last.json \
  --orientation 1,2,3 --crop 0:256,:,:
```

(16.6 s, 2.7 GB peak); its reference, annotation and hemispheres equal the
registration's moving `reference_25um`, `annotation_25um` and
`hemispheres_25um` voxel for voxel.

The library entry point is `graph::vessel_graph`, with each stage also callable
on its own (`graph::{build, radii, clean, reduce, largest_component, classify,
annotate, branch_density, write_graph_tables}`).

## Validation

`cargo test --release -p blockflow-vessel-graph` runs on synthetic data
(`segment::synthetic`: eight tubes of several brightnesses in a noisy
background, one with a saturated core):

- **block-grid invariance** — the binarisation (every arm on) and the whole
  segmentation are bit-identical at five block grids, including ones that do
  not divide the volume (`segment_intra_block`);
- **arms** — each arm on its own finds 700–2500 vessel voxels, and 90–99 % of
  what it adds is vessel; removing the adaptive, equalize or vessel arm
  from the full chain loses 324, 2425 and 1997 of its 5097 voxels; the
  deconvolve arm's own voxels are covered by the others on this fixture
  (`segment_arms`);
- **composition** — fill closes a sealed cavity, the skeleton lies inside the
  mask at under a fifth of it, the mask covers more than 85 % of the tubes'
  axis voxels, a two-channel union contains each channel, and the tract
  density is the points spread over the ball (`segment_composition`);
- **plans** — a `uint16` source plans the same phases and reads as an `f64` one;
  turning an arm off removes its phases; the segmentation plans at the size of
  a light-sheet tile (3369 x 1304 x 404), reading 93x, 51x and 36x the volume at
  blocks of 64/128/256 in z (`segment_partition`, `segment_composition_tile`);
- **fill traffic** — fragment reads are linear in the block count
  (`segment_fragment_traffic`);
- **graph** — raw graph, clique collapse, chain reduction, radii and branch
  density on synthetic skeletons, and the whole chain (classify, annotate,
  scale, edge lengths, a closure as point map, JSON round trip) on a two-tube T
  (`graph_*`);
- **parameters** — the roles' defaults, a parameter file overriding only its
  channels, and refusals of unknown keys and of a threshold for an arm that is
  off (`segment::params` unit tests);
- **CLI** — `segment` (two channels, a parameter file) and `tract` publish
  exactly the in-memory answers, including the artery channel's own mask;
  `graph` over the published labels builds the in-memory graph, finds the
  artery mask and classifies, and its three tables read back as the graph
  (every edge's polyline is its points' rows, by identity); with an atlas
  label at half the resolution and a translation, every vertex's region and
  hemisphere are the ones its physical position falls in, and the branch
  density lands on the atlas lattice (`segment_cli`);
- **atlas space** — with a synthetic prepared atlas and a hand-written
  registration (an affine scale and translation in physical units, sample on a
  lattice twice as coarse as the image), every vertex's and polyline point's
  `coordinates_atlas` are the stated map of its physical position to 1e-6
  voxels, its region and hemisphere are the atlas voxel's (0 outside the
  atlas), the distance to the surface is the atlas's (clamped), `_atlas` radii
  are µm over the sample voxel, the branch density is on the atlas grid and
  the tables record the atlas; `tract` writes the same mapping and labels per
  point and an atlas density of exactly the points inside; a two-step chain
  gives the one-step answer (the sample frame is the first step's); an atlas
  of another shape, or whose reference differs from the moving image in one
  voxel, is refused; a `--crop` copy is the source's voxels at the crop's
  physical origin, and its graph maps as the source's voxels would
  (`graph_registered`).

On a recorded 32 x 96 x 96 vasculature crop of a whole-brain scan
(`VESSEL_GRAPH_FIXTURES/vasc-binarize`, ignored test), the binarisation at the
defaults agrees with the original pipeline's recorded mask at a Dice
coefficient of 0.923 (37 160 voxels against 37 681). `segment_attribution` and
`segment_residency` (ignored) time the arms and measure the resident cost per
voxel.

Per channel role, against the reference's own binarisation at each role's
defaults (`reference-clearmap/run.sh`, outputs in
`/husky/henriksson/lightsheet-reference/vessels`; the ignored test
`segment_roles` compares):

| volume | role | ours | reference | Dice | wall time, ours / reference |
|---|---|---|---|---|---|
| recorded crop, 32 x 96 x 96 | vessel | 55 565 | 55 424 | 0.984 | 2.4 s / 4.1 s |
| recorded crop | artery | 784 | 793 | 0.971 | 0.14 s / 0.51 s |
| tile 05x05 z 1600–1664, 64 x 256 x 256, channel 0 | vessel | 666 043 | 664 797 | 0.982 | 22.6 s / 63.0 s |
| same region, channel 1 | artery | 83 008 | 82 428 | 0.979 | 4.0 s / 8.7 s |

The binarisation alone on both sides (no smoothing or filling); the
reference ran serially (its in-memory sink needs `processes="serial"`), ours on
every core in one block. The recorded crop is a vessel-channel crop, so its
artery row checks the large-vessel parameter path rather than an artery
stain.

### Deep vessel filling against the reference

`reference-clearmap/run_deep_fill.sh` (outputs in
`/husky/henriksson/lightsheet-reference/vessel-filling`; the ignored tests in
`tests/deep_fill_reference.rs` compare) fills the same masks on both sides: our
`segment --no-deep-fill` of tile 05x05 of osf-sa3x8, z 1600–1728, y 400–912,
all x (128 x 512 x 404, 1.6 x 1.625 x 1.625 µm), channel 0 as `vessel` and
channel 1 as `artery`, each channel's mask after its smooth and fill. The
reference runs `fill_vessels` with its TubeMap parameters (`reference-clearmap/fill_vessels.py`).

The network's probability, over the whole crop as one block in evaluation mode:

| channel | max abs. difference | correlation | voxels on the other side of 0.5 |
|---|---|---|---|
| vessel (resample 1) | 5.6e-4 | 0.9999999 | 2 107 of 25.6 M |
| artery (resample 2) | 1.8e-4 | 1.0000000 | 20 of 24.9 M |

(on a 62 x 126 x 126 crop: 4.8e-7.) The filled masks (input 4 974 142 and
678 190 voxels):

| | vessel: voxels added | Dice with ours | artery: voxels added | Dice with ours |
|---|---|---|---|---|
| ours | 1 893 238 | | 70 147 | |
| reference network as one block, evaluation mode | | 0.99989 * | | 0.99999 * |
| reference `fill_vessels`, evaluation mode, CPU | 1 739 556 | 0.966 (0.983 **) | 31 762 | 0.974 (1.000 **) |
| reference `fill_vessels` as shipped (dropout), CPU, two runs | 2 620 080, 2 638 131 | 0.911, 0.910 (0.925 **) | 47 502, 44 986 | 0.963, 0.965 |
| reference as shipped, GPU (float16) | 2 634 042 | 0.910 | 39 706 | 0.968 |

\* where the reference's output reaches; \*\* without the last 8 (16) planes
on each axis's far side. What the reference's `fill_vessels` adds beyond the
network: at the far faces its sink is left at zero (215 764 and 38 370 mask
voxels dropped here), and its blocks (500 voxels, overlap 50, so two along y
here) see less than the network's reach at the seam. Its dropout, on at
inference, adds about half again as many voxels and changes from run to run
(two CPU runs: Dice 0.958 with each other on the vessel channel).

The graph of each pair of filled masks (united, smoothed and filled,
skeletonised, `graph` defaults with the largest component):

| filled masks | mask voxels | vertices | edges | total length |
|---|---|---|---|---|
| none (deep fill off) | 5 324 291 | 23 678 | 33 493 | 508.0 mm |
| ours | 7 225 860 | 19 270 | 25 970 | 425.0 mm |
| reference, evaluation mode | 7 054 578 | 18 361 | 24 697 | 405.6 mm |
| reference as shipped, CPU (two runs) | 7 938 544, 7 954 797 | 16 976, 17 000 | 22 752, 22 720 | 386.8 mm, 386.8 mm |
| reference as shipped, GPU | 7 952 128 | 17 025 | 22 792 | 387.9 mm |

`segment ... --deep-fill-weights` and `graph` on the crop give ours exactly.
Filling removes 19 % of the vertices and 16 % of the length: the hollow
vessels' double walls skeletonise to parallel strands that the filled lumen
replaces with one centre line.

Time and peak memory of the filling alone (same machine, 40 cores, loaded by
other jobs; the reference with torch 2.6, without `no_grad` as it runs):

| | vessel | artery |
|---|---|---|
| ours (one block, 40 threads) | 7.2–7.7 s, 0.73 GB | 1.1–1.2 s |
| reference CPU, 20 / 40 threads | 7.8 / 7.1 s, 8.1 GB | 0.78 / 0.69 s, 1.8 GB |
| reference GPU (Quadro RTX 5000, float16) | 0.88 s, 2.9 GB on the GPU | 2.1 s, 0.44 GB |

The whole `segment` on the crop: 104.9 s and 5.6 GB with the deep filling,
91–93 s and 5.3 GB without. Ours has no GPU path.

## Limitations

- Agreement with the original implementation is close, not exact: the clip
  rounds instead of truncating, the window percentiles are ceiling percentiles
  rather than linear-interpolated ones, the sample lattices use blockflow's
  default interpolation, and the tubeness is `ops::ridge`'s Hessian.
- The adaptive, equalize and vessel windows (up to 250 voxels) are large: on a
  small block every block reads a halo of most of its neighbours. The planner
  may split the deconvolve and vessel arms into phases; the other arms are
  fixed phases.
- The equalize threshold is relative to the local 40th percentile; on a
  background that is flat after the lightsheet correction it selects the top of
  the noise. Real data, and the synthetic fixture, have enough vessel in every
  window for the cap to bind.
- A crop (`--crop`) is a copy: the published labels must match the image's
  level 0, so a run on part of a volume runs on a copy of that part.
- The tract label density is at the source resolution; the atlas-space
  density needs `--registration`. The tract points are streamed a row of
  blocks at a time, so a row's points are resident (12 bytes each).
- The smoothing runs a stated number of passes; there is no fixed-point option.
- The graph is resident (see [Out of core](#out-of-core) for what that costs
  per vertex and edge), as are the atlas volumes and the branch density on
  their own lattices. With `--density-level 0` on a one-level image the
  density is the image's lattice and is held whole (12 bytes a voxel).
- The stored thinning (volumes over `--thin-memory-mb`) is a read, a write
  and a barrier per round; on the crop it took about twice the wall time of
  the resident phase for the same CPU time.
- The deep filling runs on the CPU only (the reference runs it on a GPU),
  and only in a build with the `deep-fill` feature; its weights are not part
  of the crate. It is the network in evaluation mode, which adds about 70 %
  of what the reference's as shipped (dropout on) adds.
- Atlas space through `--registration` needs the image and the registration's
  sample to share a physical frame. Raw light-sheet tiles converted one by one
  sit at the origin, not at their place in the brain, so a tile (or a crop of
  one) can only be mapped once it is stitched into the sample's frame.
- The atlas volumes are resident (a 25 µm atlas is about 1 GB with its
  distance field); the point mapping is resident too.
