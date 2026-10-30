# Plan: dissolve clearmap-ng into blockflow

Decided 2026-10-18. Source crate: `/data/henriksson/github/claude/clearmap-ng`
(backups of it and of `clearmap-rs` exist).

## Decisions

- **Parity with the original ClearMap is no longer a goal.** The oracle, the
  `clearmap-rs` dev-dependency and every parity test go.
- **No ClearMap naming.** Code gets neutral names for what it does.
- **Copyright is cleared.** No licence concern unless a file says otherwise.
- **Reuse before adding.** Prefer an existing blockflow op. Add a core op only
  if it is genuinely general. An op that is too specific stays in the example
  that uses it.
- **Examples live in blockflow's workspace**, so a blockflow refactor that breaks
  them is caught. That requires CI to build the workspace (step 1).

## State at the start

- clearmap-ng: ~64k lines in `src/`, 80 test files (~73k lines). Last commit
  `40379f4` (2026-08-28), plus uncommitted cell-CLI work from 2026-10-02
  (`cell_run.rs`, `cell_table.rs`, `bin/clearmap_ng_cells.rs`, `docs/`).
- Library and binaries build against current blockflow. Six test files do not:
  `Decomposition::residency` gained a `&Chain` argument, and
  `Constraints`/`CostModel` gained fields (`CostModel` is no longer `Copy`).
  Affected: `smooth_residency_plan`, `fill_residency_plan`, `residency_bar`,
  `skeletonize_working_set`, `planner_machine`, `planner_calibration`.
- blockflow CI runs `cargo build`/`test`/`clippy` at the root without
  `--workspace` and has no `default-members`, so **examples and `crates/` are
  never built or tested by CI**.
- Parity runs through every stage, not just `src/oracle`: each stage implements
  `oracle::Stage`, and parameters implement `params::Parameters`
  (`clearmap()`, `regime()`, `axes()`) with `Tuned<T>`/`Axis`-wrapped fields.

## What goes where

### Delete

- `src/oracle/`, `src/params.rs`.
- The 33 test files that use `clearmap_rs`, plus `tier_labelling`. Salvage the
  oracle-free tests mixed into `cells_crust`, `graph_radius`, `stitch_rigid`
  and `histogram_percentile_cost`.
- The orchestration monolith: `workflow.rs`, `config/` (including
  `config/text.rs`) and `workspace.rs`. They hard-wire all three ClearMap
  workflows together. Each example gets its own small CLI that builds a plan and
  calls `execute_phases`, as the existing examples do.

### Replace with existing blockflow ops

| clearmap-ng | blockflow |
|---|---|
| `stages/smooth` (3x3x3 lookup) | `ops::configuration` |
| `stages/fill` | `ops::fill` |
| `stages/skeletonize` (PK12) | `ops::skeleton` / `ops::directional` |
| `stages/clip` | `ops::voxelwise` |
| `stages/resample` | `ops::Resample` |
| `stages/distance` (EDT) | `ops::distance` |
| `stages/cells` (background, DoG, h-maxima, points) | `ops::background`, `ops::interest`, `ops::reconstruct`, `ops::detect`, `ops::suppress` |
| `stages/cells/{shape,intensity}` | `ops::watershed`, `ops::measure`, `ops::intensity_statistics` |
| `equalize`, `histogram_percentile` | `ops::exposure`, `ops::normalise`, `ops::local`/`rank` (if close enough) |
| `vesselize` (tubeness) | `ops::ridge` (if close enough) |
| `deconvolve` | `ops::deconvolve`, `ops::smooth` |
| `stages/machine` (planner glue) | `examples/support/planning.rs` (check the overlap) |
| `stages/voxelize` | `ops::voxelize` |

### Candidates for core

Only if general, and only if no existing op covers them:

- a crop op (blockflow can only open a window on the input today)
- a histogram-based local percentile, if it clearly beats `local`/`rank`
- skeleton → graph extraction (`stages/graph`)
- a tubeness filter, if `ridge` cannot express it

### New examples (`examples/<name>`, package `blockflow-<name>`)

| example | from clearmap-ng |
|---|---|
| `tile-stitching` | `stages/stitch/{layout,rigid,wobbly,place,fuse}` |
| `atlas-registration` | `stages/register/{compute,parameters,transform,report}`, `bin/register_report`, resample wiring, `tract::Rescale` |
| `cell-density` | `stages/cells/*`, `collapse`, `cell_run`, `cell_table`, the cells CLI, voxelize wiring, crust filter |
| `vessel-graph` | `binarize`, `combine`, `graph`, `clean`, `reduce`, `component`, `radius`, `properties`, `scale`, `classify`, `voxelize/branch`, `tract` |

### Shared crate

`crates/atlas-regions`: `annotate::{Ontology, AtlasVolumes}` and the
distance-to-surface mask (`mask_from_annotation`). Cells, vessels and
registration all use them.

## Order

1. **blockflow CI builds the workspace.** *Done 2026-10-18.*
   - New `members` job: `cargo build --workspace --all-targets` and `cargo test --workspace`, excluding the root and the model-backed examples.
   - The `stardist` and `cellpose+yolo` rows also build their examples with `-p`.
   - All existing non-model members built and passed their tests.
   - Found, not fixed: the core crate itself fails `cargo clippy -D warnings` (7 errors) and `cargo fmt --check` (100+ files) on the local stable toolchain.
2. **Strip parity, on a clearmap-ng branch** (`dissolve-into-blockflow`). The 2026-10-02 work is committed there as `9a805eb`.
   - In practice the strip happens *while porting* each module, not as a separate pass over clearmap-ng.
   - Reason: every stage would otherwise be edited twice. The orchestration layer and six broken planner tests are deleted rather than fixed.
3. **Swap in blockflow ops stage by stage**, per the table above, done in the same pass as the port.
4. **Port into blockflow.** Stub crates and workspace entries were created by the coordinator; agents fill them using a shared brief.
   - First wave, in parallel: `crates/atlas-regions`, `examples/tile-stitching`, `examples/atlas-registration`.
   - Second wave, needing `atlas-regions`: `examples/cell-density`, `examples/vessel-graph`.
   - Per-example CLIs replace the `workflow`/`config` layer.
5. **Promote whatever turned out general to core.**
6. **Retire the clearmap-ng repo** once nothing is left in it.

## Outcome (2026-10-18)

**Steps 1–6 are done.** Nothing is committed in blockflow yet: another session works in it, so the changes are left in the working tree for review.

### Steps 2–4: ported

| crate | tests (release) |
|---|---|
| `crates/atlas-regions` | 27, 1 ignored |
| `examples/tile-stitching` | 65, 3 ignored |
| `examples/atlas-registration` | 110, 5 ignored |
| `examples/cell-density` | 49, 4 ignored |
| `examples/vessel-graph` (`segment`, `tract`, `graph` subcommands) | 84, 4 ignored |

- Together: 335 pass, 17 ignored (real `/husky` data, an external `transformix`, or timing only), 0 fail, in both the release and dev profiles.
- `clippy --no-deps -D warnings` and `cargo fmt --check` are clean for all five.
- No ClearMap naming remains, beyond one origin line per README and `/husky` data paths.

### Step 3: replaced by existing ops

Most stages now call existing ops: `configuration`, `fill`, `directional`, `voxelwise`, `transform`, `distance`, `background`, `interest`, `reconstruct`, `watershed`, `tabulate`, `voxelize`, `rank`, `ridge` (`RatioResponse` expresses the tubeness filter), `smooth`, `fft`, `align`, `coordinates`, `adjacency`, `walk`, `components`.

These were deleted rather than kept: `histogram_percentile`, `equalize` (replaced by `Rank::ceiling_percentile`), `crop` (a window on the input is enough), `stages/resample` (replaced by `ops::transform`) and `stages/machine`.

### Step 5: added to core

- **Bug fix:** `zarr_env::write_multiscales_metadata` wrote each level's scale reversed under `z, y, x` axes, which shows on any anisotropic pyramid. Regression test: `tests/zarr_env.rs::an_anisotropic_pyramid_writes_its_scale_in_axis_order`.
- **New option:** `zarr_env::PhysicalFrame` and `PyramidSpec::with_physical`. A pyramid can now carry physical scale, translation (half-block offsets for coarser levels) and unit.
  - Three examples had each rewritten the metadata themselves; all three now use this.
  - Test: `a_physical_pyramid_writes_its_scale_translation_and_unit`.

### Left in their examples

Each has one consumer today, so by the "too specific stays in the example" rule they stay. They're worth promoting when a second consumer appears.

- **vessel-graph:**
  - skeleton → graph extraction, clique collapse and chain contraction (`graph::{raw, clean, reduce}`);
  - a per-row reduction over a variable-radius ball (`graph::signal`);
  - directional thinning run to a fixed point (`segment::skeleton`);
  - a mask → point-blob adapter for `VoxelizeOp`;
  - the topological smoothing table as a `configuration` preset;
  - a `PlanBuilder` "input-or-source" cursor (`segment::plan`).
- **cell-density:**
  - `SeedRowsOp`, which numbers a point stream `1..n` independently of the grid;
  - fragment-phase component discovery for `ScalableWatershedOp`, which today needs a single whole-volume block;
  - integer-preserving background removal.
- **tile-stitching:**
  - a max-projection reduction to a plane;
  - fusion as a planned phase reading several supplied images, each at its own translation;
  - the grounded-Laplacian placement solve.

### Improvements to existing core ops

Found during the port, not done:

- `ops::align`: affine parameters are in voxel units with absolute stopping tolerances, so a pure affine fit performs badly on real data. The example works around this with its defaults.
- `ops::align`: the B-spline Hessian is stored dense per block.
- `ops::align`: coarse-to-fine fitting is single-threaded.
- `ops::align`: the B-spline grid layout should be exposed on `FittedTransform`.
- `threshold::decode_finite_samples` is `pub(crate)`, so sampled thresholds can't be read outside a reduce.
- **Planner:** it picks odd lattices for partitioned arms when the given block is thin in z (about 50× reads on that phase).

### Found, not fixed

- Core fails `cargo clippy -D warnings` (7 lints) and `cargo fmt --check` on the current stable toolchain. This was already the case before this work.
- Note: `cargo fmt -- <file>` at the blockflow root reformats the whole crate. Use `cargo fmt -p <member>`.

### Step 6: clearmap-ng retired

- Branch `dissolve-into-blockflow`, three commits:
  - `9a805eb`: the 2026-10-02 work, committed as-is.
  - `c05490a`: `MOVED.md`, which maps each old module to its new home.
  - `3ac77aa`: removes the source, tests, docs and manifest.
- The tree now holds only `README.md`, `MOVED.md`, `DATA_MOVED.md` (the real-data locations) and `LICENSE`.
- The code survives in history at `9a805eb`, and in the owner's backups.

## Notes

- Many tests read fixtures from `/husky/henriksson/clearmap-img/outputs`
  (`CLEARMAP_NG_FIXTURES`). Zarr conversions moved to
  `/husky/otherdataset/teresa/clearmap_em/` on 2026-10-02 (see clearmap-ng's
  `DATA_MOVED.md`). Examples should prefer synthetic data in `tests/` and keep
  real-data runs as `#[ignore]` measurements.
- blockflow conventions apply to anything moved in: `--release` for timings,
  British spelling, `XxxOp` / `xxx_into` / `append_xxx_phases`, reach derived
  from parameters, measured `cost_per_voxel`, a row in the `src/ops/mod.rs`
  table for any new ops module, clippy `-D warnings`.
