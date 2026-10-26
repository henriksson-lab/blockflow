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
| lightsheet | `v - min(v, min(p25 along a 150-voxel x line, 2 x p25 of a 200 x 200 window))`, both estimates over the in-range voxels; the window on a 25-voxel lattice, stride 2 | `rank::MaskedRankFilterOp`, `local::LocalStatisticOp` (masked, sampled) or `lattice_samples` (below), `voxelwise::FnCombine` |
| median | 3 x 3 x 3 over the not-low voxels; dark voxels left at zero | `MaskedRankFilterOp` |
| deconvolve arm | the sink's voxels blurred (σ 10 in y, x; flat in z) and subtracted outside the sink; residual `> 750` | `smooth::SmoothOp` (reflecting boundary), `FnCombine`, planned with `PlanBuilder::partition` |
| adaptive arm | the deconvolved volume against a local isodata threshold (3 x 250 x 250 window, lattice 3 x 50 x 50) | `local::AdaptiveThresholdOp` or `lattice_samples`, `Statistic::Isodata` |
| equalize arm | `v * min(1 / max(p40, 1), 1.5 / p97.5)` over a 5 x 200 x 200 window of in-range voxels (lattice 5 x 50 x 50); `> 1.1` | `LocalStatisticOp` (masked) or `lattice_samples`, `normalise::bounded_gain_value` |
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

The three window statistics on a lattice (lightsheet background, equalize
percentiles, adaptive threshold) are evaluated one of two ways, with the same
bytes out. Fused, each block evaluates every lattice point its halo brackets,
so where halos overlap a point is evaluated several times — about three
times for the equalize window on an 8-plane z slab. Evaluated once
(`ops::lattice_samples`), a fragment phase computes each point in the block
whose core holds it, at the window's own reach; a fragment-only phase hands
each block the points its voxels bracket; and a third phase interpolates them
and combines them with the voxels they are compared against. The choice is per statistic, against the plan's grid: once
when the fused form would evaluate each point 1.5 times or more
(`LocalStatistic::evaluations_per_sample`). `VESSEL_GRAPH_WINDOWS=fused` or
`=once` forces one form, for measuring.

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
DeepFillDevice, append_deep_fill, deep_fill}` (the last over an in-memory
mask, returning the filled mask or the probability).

**On a GPU.** `--features deep-fill-cuda` (or `deep-fill-wgpu`, through
Vulkan, Metal or DX12) adds the same network on burn's GPU backend
(`deep_fill/gpu.rs`), and `--deep-fill-device cpu|gpu|auto` (alias
`--device`; default `auto`) chooses where it runs: `auto` is the GPU when the
build has a runtime and a device is reachable, else the CPU; `gpu` refuses a
run that has neither. Neither feature needs a CUDA toolkit to build (the
driver and NVRTC are loaded at run time). The whole forward pass runs on the
device, `conv3` as eight convolutions (one per output parity) and `conv4`
composed with the upsampling before it into convolutions at half resolution
(the outermost plane of each face recomputed the direct way, where conv4's
zero padding differs); `gpu.rs` says what each saved. Blocks queue for one
thread that drives the device, one block on it at a time. The device memory
is its own cap, `blockflow::gpu`'s budget (three quarters of the device;
`BLOCKFLOW_GPU_MEMORY` sets it, `BLOCKFLOW_GPU=off` hides the device): a
forward pass holds at most 144 bytes per network-input voxel (measured: 4.7
GB for 320 x 512 x 256, 8.3 GB for 512 x 512 x 256), leases that from the
budget while it runs, and the deep-fill block is halved along its longest axis
until its read extent fits. Host memory stays the planner's. The first pass
of a process compiles the device's kernels (a few seconds; the GPU thread
warms small shapes while the earlier phases run); a `cubecl.toml` in the
working directory with `[compilation] cache = true` keeps them across runs.

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
   `--block`, default `128,256,256`; see the block grid below): the packed
   masks and the channels read
   over the block's core grown by one reach — the radius walk's search limit
   (50) or the widest ball a signal can be read over (the search radius of the
   longest stop: 53 at the reference's 3-voxel artery shift), whichever is
   larger. For every skeleton voxel of the core: the walk
   (`RadiusSearch::stop_at`, the same `ops::walk::walk_from`), each signal's
   maximum over the vertex's ball (`signal::Ball`), and a bit for each of its
   lexicographically later neighbours (`ops::adjacency::forward_offsets`)
   that is skeleton too: the adjacent pairs, in the order
   `ops::adjacency::collect_pairs` lists them (`RawGraph::from_steps`). Neither
   the walk nor a ball can leave the window inside the volume, so each vertex
   gets the whole-volume answer on every grid (`tests/graph_planned.rs`: one
   block, dividing and non-dividing grids, blocks smaller than the reach, two
   search limits, the default grid). The rows are merged in coordinate order
   (the vertex list); a second stream counts them per block, so the graph's
   size is known (and its memory leased) before any row is read.

`--workers`, `--cache-mb` (default 256), `--block`, `--pack-block` and
`--work` (scratch, removed) do not change any result. The ball's maximum over
a `bool`, `u8` or `u16` window has a fast path (no per-offset check where the
ball's box is inside the volume and the window), and the ball's offset list
holds only offsets within its radius, sorted once; the radius walk reads its
window through a typed slice (`ops::walk::walk_from` matches the image's type
once, not at every offset as `ops::rows::value_at` does), about a quarter
less CPU in the vertex measure; the answers are the same.

The graph's two plans keep their measured blocks (above) unless one block of
them does not fit the memory budget, and then run on the planner's grid
(`strategy::choose_grid`). The planner's own choice is not the default: its
cost model does not know that the vertex measure works through one block on
every worker (a plane each), so it plans a block per worker and holds that
many halo windows (2.9 GB at 24 s on the region, against 1.8 GB at 21 s on
the default grid), and it counts the pack, which is bound by decoding the
labels, as one read channel.

**The block grid.** Without `--block`, `segment` and `tract` build their plan
on each candidate grid of `blockflow::strategy::grid_candidates` (every axis
cut into a power-of-two number of pieces, fewest blocks first) and keep the
one the cost model prices fastest on `--workers` and the machine's memory
(`strategy::choose_grid`, `Host::detect`); planning takes under a second. The
heavy arms reach far in the image plane and at most two voxels across it
(windows of 200–250 in `y, x`, a 150-voxel line along `x`, a flat blur and a
one-plane disk), so the choice on these data is a stack of slabs: `[8, 512,
404]` on the smoke crop at 16 workers. The thinning, which reaches 12 voxels
on every axis, gets its own priced grid (`PlanBuilder::iterate_priced`; the
stored rounds a grid of their own from `choose_grid`), and the partitioned
arms are offered the plan grid's edges along the axes it cuts. The answer
does not depend on the grid; the published labels' chunk shape does, since a
chunk divides the block that wrote it. `--block` fixes one grid for every
phase, as before.

**`segment`, the thinning** (`segment::thin`): the in-plan fixed point is one
iterative phase whose two private buffers are the whole volume (two bytes a
voxel). It declares itself stationary (`IterativeOp::stationary`), so a round
revisits only the blocks a change in the previous round can reach (the
executor's dirty set), and a pass visits only its border voxels, held as a
list (`ops::directional`), rather than every voxel thirteen times. Over `--thin-memory-mb` (default 8192, a volume of 4 G voxels: a
whole tile stays resident, a whole brain does not) the thinning runs over
stored images instead: each round is a pixel phase
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
as `to_json`, each value read as it is written), and so are the tables: each
row is made from the graph's columns and pushed to
`blockflow::object_table::ObjectTableBuilder`, which holds a word a value and
sorts and spills runs past its buffer (what the memory budget has free, up to
the whole table), the same files whatever the buffer. The atlas volumes
(`--registration`, `--annotation-label`) and the branch density are on the
atlas or density lattice, not the image's: a 25 µm atlas is fixed, and the
density's default lattice is the coarsest pyramid level. They stay resident;
the density's `f64` volume is let go once narrowed, before it is written.

**The graph's own memory.** The graph is held as flat arrays, nothing per row
(`graph::vessel`, `graph::store`): the raw graph is its coordinates (`u32` x
3) and edges (`u32` pairs), built from a neighbour bit per vertex row rather
than a list of pairs; the cleaned graph is a raw id per surviving vertex and
its edges, the coordinates read off the raw graph's; reduce walks an
incidence of edge ids. The columns measured on the raw graph are not copied
to the vertex and polyline levels, which hold a cleaned-vertex id per row and
read the raw graph's arrays through it (a clique centre's folded values
apart): the four radius arrays are one `u32` per raw vertex, the walk's stop;
an artery-channel maximum of a `uint16` channel an `f32`; a mask a byte.
Columns set later are kept in the narrowest store that gives back every
value's bits (`f32`, `i32`, flags), and the `_atlas` radius twins and the
lattice changes are read through the columns they scale. Measured with a
counting allocator (`tests/graph_memory.rs`), bytes per raw vertex at each
stage's peak, on the region with the deep filling and the registration
(487 172 raw vertices, 537 860 polyline points; heap above the 908 MB the
atlas holds, constants such as the 21 MB radius search included) and on the
crop (250 631 raw vertices, no atlas; 84 of each "after" figure there is the
radius search's 21 MB):

| stage | region before | region after | crop before | crop after |
|---|---|---|---|---|
| collect the vertex rows | 721 | 477 (43 of it the rows, the rest the chunk cache) | 757 | 425 (idem) |
| raw graph | 413 | 73 | 552 | 113 |
| clean | 274 | 97 | 272 | 136 |
| reduce | 188 | 105 | 187 | 144 |
| assemble | 274 | 99 | 276 | 138 |
| largest component, lengths | 408 | 100 | 395 | 136 |
| atlas: map and annotate | 259 | 143 | — | — |
| tables and JSON | 862 | 382 (every row buffered) | 539 | 280 (idem) |

On a synthetic lattice of 37 296 raw vertices the stages from clean to the
lengths hold 62 bytes a raw vertex, the raw graph and its columns included,
and the graph at the end 51; the atlas columns add about 37 on the region
(`coordinates_atlas` as three `f64`s a point, the rest narrowed). The
reference's graph-tool graph is roughly 60–90 bytes a raw vertex by its
source. At the crop's density with the deep filling (1.9 M raw vertices a
mm³, an upper bound) a 450 mm³ brain has about 0.86 G raw vertices: about
53 GB for the graph stages (62 bytes a raw vertex), 69 GB in atlas space
(80), against 370 GB before, and the tables in the writer's buffer (runs on
disk past it), against 340 GB more before.

**Memory cap** (`--memory-mb`, any subcommand; else `$BLOCKFLOW_MEMORY_CAP`,
else the cgroup's `memory.max`, else `min(MemTotal / 2, MemAvailable * 3 /
4)`; the first stderr line says which). An eighth of a stated or cgroup cap
(at least 128 MiB) is held back; the rest is shared by the blocks in flight
and the chunk caches (`blockflow::budget::process_budget`), and a block that
does not fit beside the caches makes them give bytes back.

- **What a block costs.** Its phase's buffers (input, output, source images,
  the buffers its chain holds between them), plus what its op says it
  allocates beyond them (`FragmentOp::scratch_bytes`: the lattice-sample
  phases declare their planes, brackets and sample boxes). The grid choice
  sizes each phase's blocks in flight under the cap on that figure (the
  `segment:`/`tract:` line reports the widest active set;
  `VESSEL_GRAPH_PROFILE=1` lists them per phase). An op that can only use
  one block at a time says so (`BlockOp::max_in_flight`; the deep filling on
  a GPU, whose blocks queue for one device thread), and its phase is planned
  and admitted at one block.
- **Streaming admission.** The executor admits a block as soon as one in
  flight finishes and the next one's charge fits; there are no waves. A block
  is charged the plan's figure with the op margin (2.1x). A phase whose full
  set of blocks at that charge would take more than half the budget is
  measured: its first blocks start from a moment with nothing in flight, alone
  and at twice the charge, and its later blocks are charged the most one of
  them was seen to hold plus a quarter (never less than the plan's figure
  without the margin), a figure that only rises. Nothing is learned while two phases are in flight
  together.
- **The resident check.** While near the cap the executor samples the
  resident set. What no lease accounts for (the binary, the heap the
  allocator kept) is leased at each moment with nothing in flight, and when
  the resident set is more than a sixteenth of the budget over everything
  leased, no block is admitted until it comes back.

Below the least feasible cap the run stops before reading anything and names
that cap (about 330 MiB for `tract` on the smoke crop: the label phases are
on their own fixed grid). The resident thinning's ceiling
(`--thin-memory-mb`) defaults to 8 GiB or a quarter of the cap's usable
bytes, whichever is less. `graph` counts the raw vertices (a count per block
beside the rows) and leases the graph's bytes before it reads a row
(`planned::GRAPH_BYTES_PER_VERTEX`, 72, plus `ATLAS_BYTES_PER_VERTEX`, 48, in
atlas space), held as long as the graph; the branch density leases 16 bytes
a voxel of its lattice while it is made, and each table's row buffer is what
the budget has free. Under a stated or cgroup cap a lease that does not fit
stops the run naming the cap that would hold it; under the machine's own
share it is a warning. Results do not depend on the cap.

`graph`, measured 2026-10-22 (load 27–37), every output byte for byte the
uncapped run's: on the crop without a branch density, uncapped 486–519 MB,
512 MiB 279 MB, 256 MiB 215 MB, 160 MiB 135 MB (the vertex measure on the
planner's grid, the tables spilled) at 5–10 s; with the branch density
(404 MiB of its own on the crop's single level) 1024 MiB 386 MB against
559 MB before, and 512 MiB refused naming about 550 MiB. On the region in
atlas space the registration's own loading peaked at 1.74 GB (not leased
then; the prepared atlas's volumes are leased since 2026-10-23);
the graph stages stay under 1.51 GB at a 1536 MiB cap.

`tract` and `segment`, measured 2026-10-23 at `--workers 16` (load 6–23),
one run each: peak RSS (`/usr/bin/time`), wall s, CPU s. Every label, the
tract CSV and the graph JSON are byte for byte those of the uncapped run;
the cgroup rows run under `systemd-run --user --scope -p MemoryMax=`.

| run | cap | peak | wall | CPU |
|---|---|---|---|---|
| `tract`, smoke crop | none | 1897 MiB | 9.4 | 67 |
| | 1024 MiB | 678 MiB | 16 | 72 |
| | 1024 MiB (cgroup) | 652 MiB | 16 | 73 |
| | 512 MiB | 307 MiB | 29 | 91 |
| `segment`, smoke crop | none | 3045 MiB | 24 | 190 |
| | 2048 MiB | 1188 MiB | 39 | 201 |
| | 1024 MiB | 897 MiB | 45 | 198 |
| | 1024 MiB (cgroup) | 804 MiB | 46 | 197 |
| `segment`, region | 2048 MiB | 1807 MiB | 66 | 420 |

Before the lattice-sample phases declared their scratch, the interpolating
phase held about 46 bytes a voxel against the 6 it was charged, and `tract`
at 512 MiB peaked at 975 MiB. The CPU time moves little with the cap; the
wall time rises because a phase that fits fewer blocks runs fewer at once.
Streaming admission against the waves it replaced (2026-10-22, load 7–19,
before → after): `segment` on the region uncapped 55.6 → 50.5 s, 486 →
465 s CPU, 8.1 → 7.5 GB; at 2048 MiB 72 → 62 s, 2.0 → 1.76 GB. At the
tightest caps (`tract` at 512 MiB, `segment` on the crop at 1024 MiB) the
wall time did not move (48 → 49 s, 59 → 57 s): the dominant phases fit one
or two blocks there, so there was no waiting on a wave to recover.

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
Both take `--block z,y,x` (default: the grid the planner chooses, below),
`--workers`, `--time`,
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
91–93 s and 5.3 GB without.

**On the GPU** (Quadro RTX 5000, 16 GB, CUDA 12.8 driver; `deep-fill-cuda`,
float32; load averages 24–31; `tests/deep_fill_reference.rs`
`crop_gpu_against_cpu` and `reference-clearmap/fill_vessels.py --device cuda`):

| | vessel | artery |
|---|---|---|
| ours, CPU (40 threads) | 9.3–12.1 s | 1.35–1.73 s |
| ours, GPU, warm | 0.92–1.0 s | 0.52–0.55 s |
| ours, GPU, first pass of the process (device opened, kernels compiled) | 3.8 s | |
| reference GPU, float16, as shipped (dropout) | 1.1–1.5 s, 2.7 GB on the GPU | 0.67–0.68 s, 0.41 GB |
| reference GPU, float16, evaluation mode | 0.97–0.98 s, 2.6 GB | 0.75–0.84 s, 0.40 GB |

Ours on the GPU against ours on the CPU, over the crop: probability max
difference 7.8e-7 (vessel) and 4.2e-7 (artery), mean 2.8e-8 and 1.1e-8;
the filled masks are identical. Against the reference the GPU's numbers are
the CPU's above (Dice 0.9660 vessel, 0.9737 artery with its `fill_vessels` in
evaluation mode). The whole `segment` with deep filling, alternating runs
(wall s / CPU s / peak GB; load averages 30–60, shared with other jobs; outputs
under `/husky/henriksson/lightsheet-reference/round3/F`):

| | crop | region |
|---|---|---|
| CPU | 51 / 397 / 3.04; 52 / 395 / 3.01; 60 / 395 / 2.95 | 143 / 823 / 6.28; 105 / 734 / 6.16; 102 / 790 / 6.06 |
| GPU | 33 / 353 / 3.64; 43 / 350 / 3.63; 39 / 358 / 3.32 | 61 / 645 / 6.65; 80 / 633 / 6.39; 59 / 608 / 6.43 |
| deep-fill phases, CPU (vessel; artery wall s) | 12.3–18.3; 2.2–2.5 | 16.2–38.3; 3.0–5.2 |
| deep-fill phases, GPU | 2.3–2.6; 0.50–0.72 | 2.2–2.9; 0.68–1.66 |

Every label of the crop is identical between the two; on the region two
voxels of the vessel channel's mask differ (of 67 M; rounding at the
threshold), one of the united mask, and the skeleton is identical. The
GPU's peak host memory is 0.3–0.6 GB above the CPU's: the CUDA context and
runtime compiler.

### Graph against the reference

`reference-clearmap/compare_graph.sh WORK` compares our graph with the
reference's TubeMap graph (`benchmark/clearmap/vessels.py`, `vasculature.yml`
v3.1, no deep filling) on the same two-channel image, and runs each side's
graph half on the other side's skeleton and mask so the differences can be
placed: `clearmap_graph.py` is the reference's graph half on any skeleton
(`graph_from_skeleton`, radii, artery signals, `clean_graph`,
`reduce_graph`, its classifier) and exports its graphs as arrays;
`compare_graph.py` matches skeleton voxels, branch and end points (one to
one, nearest first), edges (by their matched ends, then polyline distance),
radii, lengths, tortuosity and artery labels, and in atlas space carries
both graphs through the same registration. Outputs for the two volumes below
are in `/husky/henriksson/lightsheet-reference/round3/B/{smoke,region}/compare`.

Volumes: the benchmark's smoke crop (tile 05x05 of osf-sa3x8, 128 x 512 x
404) and the fused region (512 x 512 x 256, z 160:672, y 896:1408, x 320:576
of the fused osf-sa3x8, 1.6 x 1.625 x 1.625 µm), with the pipeline's
registration (fused -> autofluorescence -> ABA 2017 25 µm, `(1,-2,3)`) for
the atlas part. "a / b" is ours / the reference.

| region volume | 1 full | 2 graph half, our skeleton | 3 graph half, its skeleton | 4 thinning, our mask | 5 mask only |
|---|---|---|---|---|---|
| a vs b | ours vs reference | ours vs its graph half | ours vs its graph half | ours vs its thinning | its chain on our mask vs on its mask |
| vertices | 61 904 / 59 953 | identical | identical | 61 904 / 60 639 | 60 639 / 59 953 |
| edges | 89 491 / 86 987 | identical | identical | 89 491 / 88 278 | 88 278 / 86 987 |
| total length | 1293 / 1271 mm | identical | identical | 1293 / 1287 mm | 1287 / 1271 mm |
| skeleton voxels, exact F1 (within 3 µm) | 0.517 (0.863) | | | 0.574 (0.893) | 0.772 (0.923) |
| branch points within 5 µm, F1 (10 µm) | 0.793 (0.842) | 1 | 1 | 0.831 (0.870) | 0.875 (0.899) |
| matched branch points of equal degree | 0.680 | 1 | 1 | 0.730 | 0.814 |
| edges matched by ends, F1 | 0.523 | 1 | 1 | 0.592 | 0.707 |
| polyline length within 3 µm of the other's | 0.920 / 0.928 | 1 / 1 | 1 / 1 | 0.946 / 0.946 | 0.947 / 0.955 |
| `artery_binary` / `artery` kappa, matched edges | 0.927 / 0.882 | 0.9992 / 0.9988 | 0.9995 / 0.9996 | 0.933 / 0.896 | 0.963 / 0.938 |

The smoke crop gives the same picture (full: 31 667 / 30 253 vertices,
567 / 549 mm, skeleton exact F1 0.522, branch F1 0.776, edges F1 0.530; on
one skeleton both graph halves agree exactly). What each step contributes:

- **Graph half: none.** On the same skeleton and mask, both directions give
  the same vertices, edges, polylines, lengths (`length` is the sum of
  26-neighbour steps in µm on both sides, a clique centre being the
  per-axis median of its members, the reference's
  `medoid_vertex_coordinates`), voxel radii and distances to the surface,
  and differ only in the radius walk's **tie order**: both walk the ellipsoid
  of offsets by voxel length and stop at the first background voxel
  (`radius_measurements.search_indices_sphere`, ours
  `RadiusSearch::new`), but within a set of equally long offsets the
  reference takes whichever numpy's unstable `argsort` puts first and ours
  the lexicographic first. The voxel radius is the same; the µm radius
  differs by up to 0.03 µm on 17 % of edges (z is 1.6 µm, y and x 1.625),
  and the per-axis offset, from which both sides take the artery search
  radius (`floor(max(radius_units_axial + shift) / spacing)`,
  `tube_map._set_vertex_vessel_type_expression`, ours
  `signal::search_radii`), on 2.4 % of vertices, which moves `artery_raw`
  on 20 % of them and `artery_binary` on 10 of 61 904. After
  classification: 11 of 86 987 edges, in atlas space too (pair 7, with the
  distance to the surface identical on every edge). Defensible either way;
  the reference's choice depends on numpy's sort.
- **Thinning: the largest single difference, and only an axis convention.**
  Both are the Palágyi–Kuba 12-subiteration thinning with the border set
  taken once per pass (the reference's `PK12.skeletonize_index`, ours
  `ops::directional`), but the reference applies its rotations and lookup
  table to `(x, y, z)` arrays and ours to `[z, y, x]`. The reference's
  thinning of our mask with the axes handed over reversed **is our skeleton
  voxel for voxel** (both volumes; `compare/skeleton-in-our-frame.json`). In
  its own frame it keeps a different, equally valid medial voxel about half
  the time: exact F1 0.57, 0.89 within 3 µm, and 1–1.5 % more branch
  points on ours. Neither frame is more correct, but thinning in the reference's
  frame (the orientation table permuted, or the volume transposed) would make
  this step exact; `directional_pass_with_sources_into` already takes the
  table.
- **Mask: the rest.** Our final mask is a 1.9 % larger near-superset of the
  reference's (Dice 0.982; on the smoke crop 146 k voxels only ours, 47 k
  only theirs, mostly the vessel channel after smoothing and filling: 135 k
  / 46 k there, against +0.2 % for the binarisation alone on the tile region above). Through
  the reference's own thinning and graph half that becomes +1.2 % length,
  +1.4 % branch points and +0.6 % end points (pair 5), and it is what
  remains once the thinning frame matches.

Graph-level statistics (region; smoke in brackets): length density 4562 /
4482 mm/mm³ (5071 / 4906), branch density 155 700 / 152 100 per mm³
(158 100 / 151 500), components 1368 / 1336 (2857 / 2829), the largest
holding 99.0 % / 99.0 % of the length, degree histograms of the same shape
(degree 3: 28 640 / 28 169, 4: 8 863 / 8 553, 5: 3 506 / 3 352), self-loops
903 / 897, median edge length 12.8 µm on both, median tortuosity 1.125 on
both. On matched edges (full pair) the median polyline distance is 1.05 µm,
the length difference median 0, and the edge radius (`radius_units`, the
maximum along the edge on both sides) median difference 0 (Pearson 0.94).
No edge is a vein on either side in this cortex box: no edge reaches the
12.5 µm vein radius with an artery intensity at or below 2500.

In atlas space (both graphs through the same registration; every 8th voxel
of the box mapped for the region volumes; ours labels identically to its own
`annotation` column), length and branch density per region:

| region | volume in box | length density, mm/mm³ | branch points per mm³ |
|---|---|---|---|
| SSs2/3 | 0.095 mm³ | 6525 / 6425 (+1.6 %) | 219 900 / 215 500 (+2.0 %) |
| SSs1 | 0.051 mm³ | 7314 / 7192 (+1.7 %) | 247 900 / 243 900 (+1.7 %) |
| SSs4 | 0.034 mm³ | 7596 / 7420 (+2.4 %) | 270 500 / 260 000 (+4.0 %) |
| SSp-n2/3 | 0.003 mm³ | 6601 / 6421 (+2.8 %) | 224 600 / 217 900 (+3.1 %) |
| SSs (all layers) | 0.182 mm³ | 6939 / 6818 (+1.8 %) | 236 900 / 231 400 (+2.4 %) |

and by depth (distance to the surface, 2-voxel = 50 µm bins inside the
brain) ours is 0.8–3.9 % above the reference in length density in every bin
with the same profile (3596 / 3556 at the surface, a peak of 9168 / 9013 at
50–100 µm, 5890 / 5774 at 200–250 µm, 7848 / 7669 at 400–450 µm). The
reference's classifier needs the atlas's distance to the surface (it indexes
it unconditionally); without an atlas, `clearmap_graph.py classify` passes
that test everywhere, as `graph` does.

`benchmark/checks.py vessels` reports a summary of the same comparison per
benchmark repeat (`graph`: skeleton, branch, end point and edge F1, length
coverage, densities; about 30–50 s on the smoke crop).

### Out of core: before and after

The same data, the same machine (40 cores, shared with other jobs: load
averages 20–75 over the runs), the binaries before and after these changes run
alternately (`/usr/bin/time -v`); outputs under
`/husky/henriksson/lightsheet-reference/vessel-planned` (`runs/summary.txt`
has every run with its load averages). The crop is the osf-sa3x8 tile 05x05
crop above (128 x 512 x 404, 26.5 M voxels); the region the fused osf-sa3x8
box `160:672,896:1408,320:576` (512 x 512 x 256, 67 M voxels), with
`graph`/`tract --registration` into the atlas. `segment` takes channels 0 and
1 as `vessel,artery`; `graph` adds `--artery-channel 1 --branch-density`;
`tract` runs on channel 0 with `--points-csv` (and `--atlas-density` on the
region). Wall s / CPU (user + sys) s / peak GB, each run:

| | before | after |
|---|---|---|
| crop `segment` | 132 / 1263 / 5.24; 103 / 1155 / 5.35; 107 / 1191 / 5.30 | 136 / 1264 / 4.52; 115 / 1148 / 4.60; 88 / 1022 / 4.59; 137 / 1171 / 4.62; 104 / 1135 / 4.58 |
| crop `segment`, deep fill | 184 / 1366 / 5.25; 129 / 1227 / 5.23 | 110 / 1170 / 4.78; 118 / 1220 / 4.50 |
| crop `graph` | 20 / 18.0 / 1.06; 14 / 14.9 / 1.07; 18 / 16.7 / 1.06 | 9.4 / 13.6 / 0.53 |
| crop `graph`, deep fill | 22 / 19.9 / 0.99; 15 / 17.1 / 0.97 | 9.7 / 15.2 / 0.52 |
| crop `tract` | 49 / 337 / 3.16; 39 / 313 / 3.03 | 29 / 254 / 2.02; 34 / 270 / 2.05 |
| region `segment` | 267 / 2266 / 9.72; 184 / 2343 / 9.87; 250 / 2324 / 9.63 | 146 / 2067 / 9.16; 175 / 2275 / 8.93 |
| region `segment`, deep fill | 304 / 2537 / 9.85; 187 / 2301 / 10.35; 250 / 2503 / 10.56 | 204 / 2448 / 9.13; 241 / 2501 / 9.09 |
| region `graph` | 58 / 47.9 / 3.85; 30 / 37.8 / 3.85; 45 / 43.4 / 3.85 | 26 / 36.8 / 2.09; 28 / 35.8 / 2.06 |
| region `graph`, deep fill | 84 / 54.2 / 3.50; 34 / 41.1 / 3.51; 36 / 41.0 / 3.50 | 22 / 42.1 / 2.10; 22 / 40.0 / 1.90 |
| region `tract` | 111 / 802 / 6.11; 72 / 752 / 6.05; 67 / 721 / 6.07 | 70 / 775 / 4.49; 74 / 773 / 4.49 |

Every label, table (vertices, edges, edge points, with every column: radii,
classes, atlas columns), graph JSON, branch density, tract CSV and tract atlas
density is byte for byte the same before and after, on both sizes, with and
without the deep filling.

What moved:

- `graph`: the volumes are no longer read whole (the crop's read stage peaked
  at 0.90 GB, 34 bytes a voxel, the region's at 2.40 GB with the atlas), the
  radius walk and the signal reads run on every block in parallel instead of
  in one serial loop (crop: 2.9 s to 0.8 s of wall for radii and signals), the
  ball maximum has a typed path, the JSON is streamed (the region's tables and
  JSON stage peaked at 3.76 GB before, 1.54 GB after), and the heap the plans
  freed is handed back before the resident stages;
- `tract`: the points are streamed, and the atlas density is rendered from a
  count per atlas voxel (the region's mapping, labelling, CSV and density:
  16.1 s and 16.8 s of CPU before, 5.3 s and 5.9 s after); the remaining peak
  and CPU time are the plan's;
- `segment`: at these sizes the plan is unchanged (the thinning stays resident
  under `--thin-memory-mb`), so wall and CPU time are within the run-to-run
  spread (±20 % on this machine; the same binary with glibc's default arenas,
  `MALLOC_ARENA_MAX=320`, against its own two: 101 / 94 / 111 s and 1091 /
  1057 / 1161 s CPU against 88 / 137 / 104 s and 1022 / 1171 / 1135 s). The peak
  falls by 0.7 GB from the two malloc arenas `main` now asks for (as
  `cell-density` does). With the stored thinning forced (`--thin-memory-mb 0`):
  crop 99 / 1045 / 4.55, region 207 / 2038 / 8.14, the same skeleton, 48
  rounds either way. The thinning alone on the crop's mask
  (`segment_thin_stored`, ignored test): resident 7–20 s of wall and 86–175 s
  of CPU, stored 14–39 s and 74–105 s, depending on the grid and the load.

### Speed

Measured 2026-10-20 at `--workers 16` on the 40-core shared machine (one-minute
load 10–60 during the runs), `segment` with profiling off, before and after
alternated; medians of three runs on the smoke crop (`128 x 512 x 404`, two
channels) and two on the osf-sa3x8 region (`512 x 512 x 256`). Wall s / CPU s
(user + system) / peak RSS GB:

| `segment` | before | after |
|---|---|---|
| smoke crop | 123 / 900 / 4.04 | 31 / 270 / 3.74 |
| smoke crop, deep fill | 123 / 962 / 4.08 | 40 / 334 / 3.86 |
| region | 174 / 1752 / 8.84 | 54 / 458 / 8.14 |
| region, deep fill | 201 / 2023 / 8.89 | 92 / 702 / 8.29 |

`tract` on channel 0: crop 35 / 222 / 1.98 before, 12 / 85 / 2.24–2.54 after
(the peak is in the tubeness phase, on twice the blocks in flight); region 62 /
558 / 3.78 before, 20 / 163 / 3.66 after. Every label (voxels and stored
bytes), the graph JSON and the tract CSV are byte for byte the same.

What moved, then and since:

- **The sliding histogram.** The lightsheet line (150 voxels along `x`) and
  the vessel-background disk (709 offsets) are
  `ops::sliding::SlidingHistogramOp` on the clip's bins rather than a dense
  gather (crop phases 3 and 11: 173 and 439 s of CPU before, 29 and 30 s
  after).
- **The thinning's border list.** A pass visits its border voxels, held as a
  list, instead of scanning every voxel thirteen times; the block copies are
  flat; and the iterative phase reruns only the blocks a change can reach
  (crop: 84 s of CPU before, 21 s after; the stored rounds,
  `--thin-memory-mb 0`, 64 s to 30 s).
- **Planner-chosen grids** (above). On the crop `[8, 512, 404]`: the
  plane-wide windows read no halo in `y, x`, and sixteen blocks keep sixteen
  workers busy where `[64, 256, 256]` gave eight; on the region `[32, 512,
  256]`.
- **A heap release at every phase boundary** (`profile::Listening`), without
  which the region's peak was 9.1 GB on the chosen grid.
- **Lattice samples** (2026-10-22). The three window statistics on a lattice
  are evaluated once per lattice point (`ops::lattice_samples`, see
  "Segmentation") where the fused form would evaluate a point 1.5 times or
  more. Medians of four runs, fused → once (load 35–52): smoke crop 34.0 →
  32.4 s, 311 → 248 s CPU, 2.89 → 2.49 GB; region 78.2 → 76.2 s, 551 →
  525 s CPU, 5.91 → 5.70 GB; with the deep filling, crop 365 → 311 s CPU
  and region 794 → 760 s. The labels are identical.
- **Streaming admission** (2026-10-22; "Memory cap" above). Uncapped, on the
  region: 55.6 → 50.5 s, 486 → 465 s CPU, 8.1 → 7.5 GB; `tract` on the crop
  10.0 → 9.8 s, 76 → 69 s CPU, 2.57 → 2.06 GB.
- **GPU deep fill** (2026-10-22; "Deep vessel filling"). The whole `segment`
  with deep filling, CPU → GPU: crop 51–60 → 33–43 s, region 102–143 →
  59–80 s; the deep-fill phases themselves 12.3–18.3 → 2.3–2.6 s (crop,
  vessel). One block is on the device at a time (`BlockOp::max_in_flight`),
  so the phase does not park workers and their buffers in the device's
  queue.

## Limitations

- Agreement with the original implementation is close, not exact: the clip
  rounds instead of truncating, the window percentiles are ceiling percentiles
  rather than linear-interpolated ones, the sample lattices use blockflow's
  default interpolation, and the tubeness is `ops::ridge`'s Hessian.
- The adaptive, equalize and vessel windows (up to 250 voxels) are large: on a
  block cut in the image plane every block reads a halo of most of its
  neighbours. The chosen grid cuts across the plane instead; a `--block` that
  cuts `y` or `x` pays that halo. The planner may split the deconvolve and
  vessel arms into phases; the other arms are fixed phases.
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
  and a barrier per round, 48 rounds on these data: on the crop's mask it took
  about twice the wall time of the resident phase for less CPU time, so it is
  the default only past the resident phase's memory.
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
