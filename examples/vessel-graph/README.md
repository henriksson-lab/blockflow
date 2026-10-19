# Vessel graph

Light-sheet vasculature to a vessel graph. The **segmentation** half turns one
or more vessel channels of an OME-Zarr image into a binary mask and its
skeleton, and also makes tract maps (a fibre-tract channel to a mask and a
point density). The **graph** half turns the skeleton into an annotated vessel
graph. Every step is planned over blocks with blockflow; nothing loads a whole
volume.

Originally developed as part of a ClearMap reimplementation.

## Pipeline

### Segmentation (`src/segment`)

One plan per run:

```text
channel k:  raw -> binarize -> smooth -> fill --\
                                                 +-> OR -> smooth -> fill -> mask -> skeleton
channel j:  raw -> binarize -> smooth -> fill --/
```

With one channel the channel's own filled mask is the answer; the second
smooth-and-fill runs only after a union.

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
60000)` with every arm on, which no role uses.

Overrides apply in this order: the role's defaults, then a JSON parameter file
(`--params`), then the command-line flags (which apply to every channel). The
file has one entry per `--channel`, every key optional:

```json
{
  "channels": [
    {"role": "vessel", "clip_range": [350, 1000]},
    {"role": "artery", "clip_range": [1000, 8000], "deconvolve_threshold": 450,
     "equalize": false, "vesselize": false, "smooth_passes": 6, "fill": true}
  ],
  "combined": {"smooth_passes": 6, "fill": true}
}
```

Channel keys: `role`, `clip_range`, `bins`, the arm switches `lightsheet`,
`median`, `deconvolve`, `adaptive`, `equalize`, `vesselize` (an arm switched on
starts at its defaults), `deconvolve_threshold`, `equalize_threshold`,
`vesselize_threshold` (refused for an arm that is off), `smooth_passes` (0 is
none) and `fill`. Unknown keys are refused. In the library:
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

Skeleton to an annotated vessel graph, resident in memory:

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
thresholds, smoothing passes, `--no-fill`, `--keep-border`;
`tract --help` the percentiles, sampling, density radius and the CSV options.
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
  density needs `--registration`. The registered tract points are mapped and
  labelled resident (a whole brain's tract mask is many millions of points).
- The smoothing runs a stated number of passes; there is no fixed-point option.
- The graph stages are resident: the radius walk, the signal reads, the atlas
  labels and the branch density hold whole volumes.
- No deep vessel filling (the reference's CNN step, on by default there): the
  per-channel masks are binarised, smoothed and filled only.
- Atlas space through `--registration` needs the image and the registration's
  sample to share a physical frame. Raw light-sheet tiles converted one by one
  sit at the origin, not at their place in the brain, so a tile (or a crop of
  one) can only be mapped once it is stitched into the sample's frame.
- The atlas volumes are resident (a 25 µm atlas is about 1 GB with its
  distance field); the point mapping is resident too.
