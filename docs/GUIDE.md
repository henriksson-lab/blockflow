# Building a pipeline on blockflow — a map

A short orientation for people and agents who want to *use* blockflow: write an
op, assemble a plan, run it over an OME-Zarr image, and publish labels and
object tables. It points at the code rather than restating it. The long design
notes in [`design/`](design/) explain *why*; this file says *where* and *how*.

**Status.** Written 2026-10-17 against commit `11d31fc`; §5 and the new ops in
§4.7–4.8 added 2026-10-18 on branch `shell-objects-3d`. Every function and type
named here was checked to exist at that commit, and the signatures quoted were
copied from the source. When this file and the code disagree, the code wins —
please fix this file.

## 1. The model in five sentences

1. Work is a **plan** of **phases**; each phase runs one op (or a chain of ops)
   over every block of a **block grid** and writes one image and/or byte
   **fragments** (rows).
2. An op declares what it needs — its **reach** (halo), extra **source
   images**, whether it is a **barrier** with a one-off **reduce** — and the
   planner derives block reads, ordering and memory from those declarations.
3. Images are numbered: `ImageId 0` is the input, phase `p` writes image
   `p + 1`, and `ImageId::supplied(i)` is the i-th extra array handed to the run.
4. An **environment** (`ZarrEnvironment` for files) provides storage; the
   executor (`strategy::execute_phases`) runs the plan with bounded memory.
5. Anything global (a threshold, a fitted model, a height map) is computed in a
   barrier phase's `reduce` and handed to every block of that phase as bytes —
   never by loading the whole volume into an op.

If an op you write needs the whole volume in memory, it will not scale; see
§4.6 for the sanctioned exception and its cost.

## 2. Where things are

| need | look at |
|---|---|
| op traits | `src/op.rs` (`BlockOp`, `SourceInput`, `Combine`), `src/fragment.rs` (`FragmentOp`, `BlockView`, `BlockOutput`, `FragmentInput/Output`), `src/iterate.rs` |
| plan assembly | `src/assemble.rs` (`PlanBuilder`, `Phase`, `ImageId`, `Chain`), `src/reach.rs` (`Reach`) |
| running | `src/strategy.rs` (`execute_phases`, `Hints`), `src/zarr_env.rs` (`ZarrEnvironment`) |
| OME-Zarr in/out | `src/ome_zarr.rs` (`OmeVolumePyramid`, `write_label_metadata`, `ensure_zarr_group`), `src/label_pyramid.rs`, `src/object_table.rs`, `crates/ngff-object-table` |
| rows | `src/table.rs` (`Schema`, `Column`, `RowBuilder`, `Row`, `Table`), `src/ops/rows.rs` |
| ops by topic | `src/ops/mod.rs` — its header table lists every module and what it makes expressible |
| finished pipelines | `examples/*` (§7) |
| design rationale | `docs/design/*.md` (note the trust caveat at the top of `writing-an-op.md`) |
| known gaps | `docs/design/planner-gaps.md`, `docs/ops-survey/` |

## 3. Ops

### `BlockOp` — region in, region out (`src/op.rs`)

Required: `name()`, `reach(axis, volume_len) -> usize` (deliberately no
default), `apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor)`.
Commonly overridden:

- `accepts(Dtype) -> bool` — **the default accepts only `F64`**. An op fed
  `u16` or `f32` input is refused at plan time unless it says otherwise.
- `produces(Dtype) -> Dtype` — output element type.
- `reach_spec(volume) -> Reach` — `Bounded{lo,hi}`, `PerBlock`, `All`, `Space`.
- `source_inputs(volume) -> Vec<SourceInput>` with `apply_with(input, sources, out, at)`
  for extra images; `SourceInput::voxelwise(image)` reads with no halo, and a
  supplied image must declare its dtype.

Put the computation in a free function generic over the element type and keep
the `BlockOp` a thin wrapper; that is the convention in `src/ops/`.

### `FragmentOp` — blocks that emit or consume rows (`src/fragment.rs`)

- `inputs()` / `outputs()` declare fragment streams
  (`FragmentInput::own(stream, phase)`, `FragmentOutput::new(stream, Lifecycle, Coverage)`).
- `reads_pixels`, `writes_pixels`, `produces`, `reach` (default 0),
  `source_inputs`, `seam_fold`.
- `apply(&BlockView) -> Result<BlockOutput>`; `BlockOutput::fragment(stream, bytes)`
  or `BlockOutput::nothing()`.
- `BlockView` carries `grid`, `core` (the region this block **owns**), `read`
  (core plus halo), `valid`, `at`, `dtype` and `reduced`.
- `barrier() -> bool` + `reduce(&PhaseView) -> Result<Vec<u8>>`: when the phase
  is a barrier, `reduce` runs **once**, after all earlier phases finish and
  before any block of this phase; every block sees the result as
  `BlockView::reduced`. Declaring `reduce` without `barrier` is refused at plan
  time. Give the encoding a magic and a version.

### Chains

`Chain::op`, `Chain::sequence`, `Chain::parallel` (fan-in through a `Combine`),
`Chain::alternative`, `Chain::source(ImageId, Dtype)` (start a chain from
another image instead of the phase input).

## 4. Patterns

### 4.1 Blockwise filters

```rust
let mut b = PlanBuilder::new(volume, dtype, BlockGrid::new(volume, [32, 256, 256])?);
let smoothed = b.pixels(Chain::op(SmoothOp::new("smooth", gaussian)))?;
let edges = b.pixels(Chain::sequence(vec![
    Chain::source(smoothed.image()?, Dtype::F64),
    Chain::op(my_edge_op),
]))?;
let assembly = b.finish()?;
let env = ZarrEnvironment::attach(&work, &[source])?;
execute_phases("name", &assembly.workflow, &assembly.decomposition,
               &Hints { concurrency: workers, ..Hints::default() }, &env, &[], &assembly.work())?;
```

`PlanBuilder` only appends: `pixels(chain)`, `fragments(op)`, `iterate(op)`,
`partition(..)`, `regrid(grid)` (later phases use the new lattice), then
`finish()`. `Phase::index()` and `Phase::image()` name what a phase wrote.

### 4.2 Several channels

Open one pyramid per channel and attach the extras as supplied images; they
must share shape and lattice.

```rust
let a = OmeVolumePyramid::open(&zarr, 0, t)?.attached_level(0)?;   // ImageId 0
let b = OmeVolumePyramid::open(&zarr, 1, t)?.attached_level(0)?;   // ImageId::supplied(0)
let env = ZarrEnvironment::attach(&work, &[a, b])?;
```

In the plan, read `b` with `Chain::source(ImageId::supplied(0), dtype)` or as an
op's `SourceInput`.

### 4.3 A global value used blockwise (threshold, model, small map)

Sample per block into a fragment stream, then a barrier op `reduce`s the samples
once and applies the result per block. The canonical code is
`src/ops/threshold.rs`:

```rust
pub fn append_global_threshold_phases(plan: &mut PlanBuilder, stream: impl Into<String>,
    selection: impl Into<GlobalThresholdSelection>, output: GlobalThresholdOutput) -> Result<Phase>
pub fn append_bounded_otsu_threshold_phases(plan: &mut PlanBuilder, image: ImageId,
    stream: impl Into<String>, bins: usize, output: GlobalThresholdOutput) -> Result<Phase>
```

`GlobalThresholdSelection` is `Single(GlobalThreshold)` (Otsu, Yen, triangle,
minimum, `RobustSigma` = median + k·1.4826·MAD, `LogOtsu`) or `MultiOtsu`.
To threshold a sample of any image — on a grid-independent lattice
(`ValueSampling`), optionally inside a mask — emit it with `SampledValuesOp`
and reduce it in your own barrier op with `sampled_thresholds(at, stream, rule)`.
`GlobalThresholdOutput::KeepAtOrAbove` writes the value where it reaches the
level and `-inf` elsewhere, which turns a data-dependent level into a fixed
one for ops that take a constant minimum (e.g. the peak table below). The bounded variant keeps O(blocks × bins) state, not
O(voxels). The same barrier-and-reduce pattern serves anything small and
global; a 2D height map is a reduce result like any other.

The alternative is two runs: compute the small result with one
`execute_phases`, then pass it as a parameter to the ops of a second plan
(`src/ops/slice_normalise.rs` does this).

### 4.4 Rows and object tables

- Rows: `Schema::new(vec![Column::u64(..), Column::f64(..)])`,
  `RowBuilder::new(schema).push(at, &values)`, `.encode()` into a fragment.
- **Ownership across block seams:** emit an object only from the block whose
  `core` contains its anchor point (centroid, peak, or first voxel);
  `ops::detect::owner_of(grid, at)` gives the owning block. A halo at least as
  large as the object avoids losing objects that straddle seams.
- IDs that survive reruns: derive them from a voxel position (linear index + 1),
  as `classical_centers` and `model_segment` do.
- Finalise into a spatially indexed table (feature `zarr`):
  `object_table::finalize_object_table(env, stream, phase, volume, schema, out_root, tmp_parent, TableSpec, project)`,
  which streams fragments through bounded external sort. For rows that are
  already in memory and few, `object_table::write_object_rows(root, spec, rows)`
  writes the identical layout.
- Worked examples: `classical_centers::{append_classical_center_phases, finalize_centers}`,
  `model_segment::finalize_instances_3d`.

### 4.5 Connected components and per-object measurement

```rust
let faces = builder.fragments(LabelComponentsOp::new("label", "x.components", Lifecycle::DeleteOnExit)
    .connecting(Connectivity::Faces))?;
builder.fragments(RelabelComponentsOp::reading("relabel", "x.components", faces, builder.grid())
    .connecting(Connectivity::Faces))?;
```

This is out-of-core labelling with a global merge across seams (`src/ops/label.rs`,
union-find in `src/ops/components.rs`). Planned measurements on a label image:
`ops::measure::Measurements::for_labels(labels)...build(decomposition)`; see
`examples/skimage-pipeline`. Points to regions: `ops::detect::{LabelRegionsOp, RegionPointsOp, detect_phases}`.
Painting points into labels: `ops::label::LabelPointsOp`; densities:
`ops::voxelize::VoxelizeOp`.

### 4.6 Whole-volume ops (the exception)

An op with `Reach::all()` placed after `builder.regrid(BlockGrid::new(volume, volume))`
sees the entire volume as one block (`ops::watershed::ScalableWatershedOp`
does this). It is planned and correct, but memory grows with the volume, so it
caps the image size. Prefer §4.3: reduce the volume to the small global
structure the algorithm actually needs, solve that in `reduce`, and apply it
blockwise.
`ops::surfaces` is a worked example: an exact global min cut over stacked
height-field surfaces, where blocks pool their own columns, a barrier `reduce`
solves the small graph, and blocks paint the region labels. The solve phase
also publishes the solved columns (`SurfaceColumns`) on `<stream>.heights`,
which later barrier ops read in their `reduce` (`SurfaceColumns::from_stream`)
— e.g. `SurfaceBandOp`, a mask between two shifted surfaces.

### 4.7 Peaks and points

- Response images: `ops::interest::BlobResponseOp` (DoG, LoG, Hessian determinant).
- Planned exact regional maxima as rows:
  `ops::interest::append_response_peak_table_phases(plan, rows_stream, lifecycle, connectivity, minimum_response)`.
  Only a fixed threshold is supported.
- `classical_centers` adds candidate selection: per-block candidates, then a
  barrier `SelectCentersOp` whose `reduce` thresholds them (Otsu on response
  and on smoothed intensity) and does greedy minimum-distance suppression in
  physical units (`minimum_distance_physical`, strongest first); `apply` emits
  each centre from its owning block. **The thresholds are hard-coded Otsu**,
  which fails when bright objects saturate.
- Suppression as its own phase: `ops::suppress::SuppressRowsOp` (barrier) keeps
  rows by a priority column and drops one within `factor × pair(r_kept, r_row)`
  of a kept row — fixed or per-row radius, smaller/larger/sum pairing,
  positions from the row voxel or from columns. `suppress_points` is the
  resident kernel; `classical_centers::suppress_within_distance` calls it.
- Template matching: `ops::template_match` (radial templates, NCC at several
  radii as one `ops::transform_bank::TransformBankOp`); a window fit per row
  that emits shapes: `ops::star_convex` (`ShellFitOp`, `StarConvexLabelsOp`).
  `examples/shell-objects-3d-ome-zarr` uses all of these.
- `ops::interest::response_peak_points` is the resident (whole-array) helper,
  for tests and small arrays.

### 4.8 Different resolutions

Every image in one plan shares one lattice; a source on a different lattice is
refused. To work at a coarser level, either attach `pyramid.attached_level(k)`
in a separate run and pass the small result on (§4.3 two-run form), or resample
inside the plan with `ops::resample::{ResampleOp, resample_phase}`
(`append_resample_phase` on a `PlanBuilder`; `append_block_mean_phases` for a
box mean to a coarser grid).

### 4.9 Iterating to a fixed point

`src/iterate.rs`, added to a plan with `PlanBuilder::iterate(op)`.

- `IterativeOp`: the running estimate is an **image**. Each substage reads
  its operands — `Operand::Running` (the previous substage's output),
  `Operand::Fixed` (the phase input) or `Operand::Source(ImageId)` — with a
  per-substage reach, and neighbours exchange their previous outputs through
  that halo. The phase's reach therefore does not grow with the number of
  substages, and the loop stops when nothing changes (a `SubstageLimit` is a
  required runaway guard). Examples: `ops/reconstruct.rs` (`HExtremaOp`),
  `ops/configuration.rs`.
- `IterativeReduceOp`: the running estimate is a **small global state**:
  `state[k+1] = update(state[k], reduce(map(block, state[k])))`, with the
  converged state written to a fragment stream. Example: `ops/align.rs`
  (`FitByMoments`). For fitting a few parameters to data that is not all in
  memory. `PlanBuilder` has no method for it: build the phase with
  `iterate::iterative_reduce_phase(&op, grid)` into a `Decomposition` and run
  it with `PhaseWork::IterateReduce(op)`, as `ops/align.rs` and
  `tests/iterative_reduce_phase.rs` do.

Both default `accepts` to `F64` only, like `BlockOp`.

## 5. Doing less work

What the framework already skips, and what an op has to skip itself. Measured
on `examples/shell-objects-3d-ome-zarr` (SHELL_OBJECTS_3D_PLAN.md, "Speed").

**Free, if declared:**

- **Uniform blocks.** The executor skips a block whose read extent (halo
  included) is one value when every op in the phase declares
  `constant_maps_to(value)` (`strategy.rs`, "The short circuit"). The block is
  still read and scanned, and a constant output block is still written.
- **Known-empty chunks.** The chunk cache does not fetch a chunk its backend
  reports `is_known_empty` (`cache.rs`, `ChunkFetcher`/`RegionSource`); the
  block buffer is left zero there. Only synthetic sources implement it — the
  Zarr backend does not, and a Zarr array written by a run has the *unwritten*
  sentinel as fill value (`zarr_env::unwritten_fill`: `MAX` for integers, NaN
  for floats), so an all-zero label chunk is written and read back in full.
- **Tile-aligned halos.** `TransformConvolveOp`/`TransformBankOp` declare
  `AxisReach::Aligned`: on a block edge that is a whole number of tiles the
  halo is the kernel's reach, otherwise it adds `tile - 1`. Pick block edges
  that are multiples of the tile.

**Up to the op** (none of these change a plan, a halo or a price):

- Compute only outputs someone reads. A transform op computes only tiles that
  can hold a valid voxel (`convolve::run_tiles`); a separable smoothing of a
  sub-region uses `ridge::gaussian_smooth_region_into`, bit-identical to
  smoothing everything and cropping.
- Do not materialise a pre-filter a sparse consumer reads: `ShellFitOp::with_prefilter`
  reads the raw image and smooths only the tiles its rows' fits can reach. A
  full `SmoothOp` phase there cost a written and re-read f64 volume.
- Several kernels on one input: `TransformBankOp` transforms each input once per
  tile instead of once per kernel.
- Threads inside a block (`run_tiles`, rayon in row ops) pay when there are
  fewer blocks than cores; with many blocks in flight they share one rayon pool
  with the executor's workers.

**Costs that are easy to miss:**

- **Read amplification.** A block reads `core + 2 × reach` per axis; at a reach
  of ~90 and a 160×256×256 block that is ~5× the volume per pass. A block edge
  of a few times the reach (the shell example uses 4×) fixed it there.
- **`SeamFold::Unordered` fragment ops are applied twice per block** (fragments
  reversed) whenever they gather more than one fragment, and the outputs
  compared (pixels only over the written region). Keep per-application work
  proportional to the core.
- **A fragment input's reach in whole blocks widens the phase's pixel read
  extent** (`fragment::fragment_phase`) unless the op is a barrier: a painter
  reading neighbours' rows gets a `BlockView::output_buffer` of 3×3×3 blocks.
  Size scratch buffers by `BlockView::core`, not `read`.
- Profile before guessing: `perf record -g` on a release build, and per-phase
  CPU against wall time (`cpu.rs`); a phase on one core is usually one block.

## 6. Reading inputs and publishing outputs

- `OmeVolumePyramid::open(root, channel, time)`: `levels()`, `voxel_size()`
  (z, y, x), `voxel_volume()`, `image_root()`, `label_source()`,
  `attached_level(k)` (optionally `.window(start, shape)`).
- `ZarrEnvironment::attach(work, &[images])`: image 0 then supplied images.
  A phase's output is stored under `work/level{p+1}`; examples move that
  directory into place rather than copying it.
- Labels: write level 0 as **uint64** (the label pyramid builder requires it),
  then `label_pyramid::build_nearest_label_pyramid_with_blocks`,
  `ome_zarr::write_label_metadata`, and `label_pyramid::refresh_label_registry`
  on the `labels` group. Extra NGFF `image-label` keys (`colors`, `properties`)
  can be added to the written `zarr.json`; newvolim reads both.
- Label + table pairs for newvolim: `labels/<name>` and `tables/<name>`, table
  `region = "../../labels/<name>"`, `coordinate_space = "../../"`, identity
  column `label_id` equal to the label values, coordinates `centroid_z/y/x` in
  level-0 voxels. A `deleted` (and `toggled`) column, if present, is read as the
  default review state.
- `ome_zarr::publish_label` installs a planned label image and its table: it
  moves a run's `work/level{N}` (pin it with `Hints::keep_images`; it must be
  `uint64` — `ops::WidenLabelsOp` converts), builds the pyramid and metadata,
  swaps in label and table, and refreshes the registry; `replace: false`
  refuses existing output. `bead-layer-distance`, `classical-3d-centers`,
  `cellpose-3d` and `stardist-3d` use it; the 2D cellpose and stardist
  examples still hand-roll the steps for their 2D levels. `label_publish.rs` is the transactional writer for
  sparse masks edited in the UI.

## 7. Anatomy of an example

Examples are workspace members (root `Cargo.toml`) under `examples/<name>/` and
follow [`OMEZARR_EXAMPLES.md`](../OMEZARR_EXAMPLES.md): attach the Zarr, run
planned.

| part | where |
|---|---|
| `Cargo.toml` | `blockflow` with feature `zarr`, `clap` (derive), `ngff-object-table` as needed |
| `src/main.rs` | refuse debug builds; parse CLI; open pyramids; assemble plan(s); `execute_phases`; finalise tables; publish |
| `src/lib.rs` + modules | when there is enough to test (`examples/plate-timelapse-tracking`) |
| `tests/` | end-to-end on synthetic data (`plate-timelapse-tracking/tests/end_to_end.rs`) |
| `README.md` | pipeline, input, run, output, validation numbers, limitations |
| `scripts/` | comparisons against reference tools |

Reusable ops and `append_*_phases` / `finalize_*` helpers go in blockflow's
`src/`; CLI, configuration, publishing and reports stay in the example. Good
references: `classical-3d-centers-ome-zarr` (planned detection, watershed,
tables), `cellpose-3d-ome-zarr` (instances, grid choice, resume),
`plate-timelapse-tracking` (library-shaped example with tests),
`bead-layer-distance` (several channels in one plan, global surfaces, sampled
thresholds, components, rows from barrier ops, `publish_label`,
`finalize_object_table`, a synthetic end-to-end test across block grids).

## 8. Conventions

- Release builds only, for runs, tests and benchmarks (`AGENTS.md`).
- New dependencies: only what the framework needs goes in the root crate;
  heavy ones are feature-gated (see the comment block in `Cargo.toml`).
- Tests sit inline (`#[cfg(test)]`) in most `src/ops/*.rs`, plus `tests/`.
- Errors name the cause and the fix; doc comments explain why, with measured
  numbers where performance is claimed.
- A new ops module gets a row in the table at the top of `src/ops/mod.rs`.

## 9. Traps

- A barrier whose `seam_fold` is `SeamFold::Unordered` has its `reduce` run
  **twice**, walking the blocks forwards and backwards, and the run is refused
  if the answers differ; a non-barrier one has its `apply` run twice per block
  that gathers more than one fragment (§5). Budget for it, and do not log from
  `reduce`/`apply` expecting one line.
- An intermediate image is freed after its last reader. To read one after the
  run (e.g. from `work/level{N}` or `env.image`), pin it with
  `Hints { keep_images, .. }`.

- `BlockOp::accepts` defaults to `F64` only (see §3).
- Label pyramids need `uint64` labels.
- `component_label_phases` and `detect_phases` return a standalone
  `Decomposition`; inside a `PlanBuilder`, add `LabelComponentsOp` and
  `RelabelComponentsOp` with `builder.fragments` as in §4.5.
- Reading an intermediate image by path (`work/level{N}`) works, but couples you
  to the layout; prefer chaining with `Chain::source` / `SourceInput`, or
  `env.image(i)` for small images.
- Module headers cite `docs/design/BLOCK_OPS.md`, which does not exist.
- `graph_cut::CutGraph` is an exact in-memory min cut. For surfaces use
  `ops::surfaces::append_layered_surface_phases` (§4.6).
