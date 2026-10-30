# Example Workflow Implementation Plan

Goal: add realistic biology image-analysis examples that double as benchmark
workloads. Treat examples as executable pressure tests for the library API. Do
not add core library helpers speculatively; promote only repeated or awkward
example code that is hard to write correctly at the example layer.

## Principles

- Keep scenario-specific orchestration in `examples/*`.
- Keep reusable image-processing kernels, measurement primitives and correctness
  contracts in the root `blockflow` crate.
- Prefer deterministic synthetic fixtures first, then optional real/reference
  data when available.
- Every example should emit machine-readable output that can be compared across
  runs: CSV for object rows, JSON for summary/benchmark metadata.
- Every example should have a script that runs enough images to make startup
  overhead visible separately from per-image pipeline cost.
- Promote code from examples into the library only after at least two examples
  need the same helper, or one example needs a helper whose correctness is easy
  to get subtly wrong.

## Example Set

### 1. Percent-Positive Scoring

Purpose: model a common CellProfiler/ImageJ workflow where segmented cells or
nuclei are classified by marker intensity.

Initial package: `examples/percent-positive/`.

Pipeline:

1. Generate or load a label image and one marker-intensity channel.
2. Measure per-object intensity with the existing measurement builder.
3. Classify each object as positive/negative by a configured intensity feature
   and threshold.
4. Emit object CSV and summary JSON with positive count, negative count, total
   count and percent positive.

Benchmark/reference angle:

- Compare against a small Python NumPy/scikit-image reference script.
- Require exact object IDs/counts and exact classification on deterministic
  fixtures.

Library-promotion trigger:

- If thresholded table classification is duplicated in another example, add a
  small table-summary helper to `blockflow`.

### 2. Foci Per Nucleus

Purpose: count puncta/foci/speckles inside segmented nuclei.

Initial package: `examples/foci-per-nucleus/`.

Pipeline:

1. Generate or load a nucleus-label image and a spot-intensity image.
2. Enhance spots with an existing filter/chain where possible.
3. Detect spot objects or points.
4. Relate spot objects/points to containing nucleus labels.
5. Emit per-nucleus foci counts and optional per-focus intensity rows.

Benchmark/reference angle:

- Compare against a Python reference over deterministic synthetic images.
- Make the fixture include border cases: foci on label boundaries, outside any
  nucleus, and multiple foci in one nucleus.

Library-promotion trigger:

- If containment/child-count wiring is awkward despite existing overlap
  relationship machinery, add a typed relation wrapper rather than duplicating
  row-table glue in every example.

### 3. Wound / Scratch Assay

Purpose: measure open wound area fraction from a binary or intensity image.

Initial package: `examples/wound-assay/`.

Pipeline:

1. Generate or load an image with cell-covered and open regions.
2. Threshold, clean with morphology and label connected regions.
3. Report total open area, covered area, open fraction and optional per-column
   wound-width profile.
4. Emit summary JSON and optional profile CSV.

Benchmark/reference angle:

- Compare against a NumPy/scikit-image reference.
- Use 10-image and 50-image batches like the existing benchmarks.

Library-promotion trigger:

- If profile extraction becomes a reusable row operation, consider adding a
  general axis-profile helper. Keep simple summaries in the example.

### 4. Colocalization Workflow

Purpose: exercise the existing colocalization measurement surface as an example
rather than a hidden test-only capability.

Initial package: `examples/colocalization/`.

Pipeline:

1. Generate or load two intensity channels and optional object labels.
2. Compute image-level and object-level colocalization summaries already
   supported by `ops::measure`.
3. Emit summary rows for Pearson/Manders/Costes-style measurements where
   available.

Benchmark/reference angle:

- Compare against a Python reference for the basic scalar measures on
  deterministic fixtures.

Library-promotion trigger:

- If the example needs only naming/output glue, keep it outside the library.

### 5. 3-D Object Measurement

Purpose: show physical-spacing-aware 3-D measurements and stress row outputs.

Initial package: `examples/object-3d-measurement/`.

Pipeline:

1. Generate labelled 3-D objects with known physical spacing.
2. Measure volume, boundary/surface-derived values, radius summaries, moments
   and selected descriptors already present in the measurement API.
3. Emit object CSV and summary JSON.

Benchmark/reference angle:

- Use analytic fixtures where expected volume/centroid/bounding boxes are known.
- Add Python reference only for measures that are easy to express exactly.

Library-promotion trigger:

- If feature selection or output naming is the only hard part, add only small
  typed feature-selection helpers.

## Shared Structure

Each example package should follow the current benchmark examples:

- `examples/<name>/Cargo.toml`
- `examples/<name>/README.md`
- `examples/<name>/src/bin/<name>.rs`
- `examples/<name>/scripts/run_blockflow.sh`
- `examples/<name>/scripts/run_reference.sh` when a reference is practical
- `examples/<name>/scripts/run_benchmark.sh`
- optional `examples/<name>/reference-*/`

Generated fixtures and outputs must live under `.tmp/<name>/` and stay out of
git.

## Implementation Order

1. Add `percent-positive` because it is the smallest table-level workflow and
   should reveal whether simple derived summaries belong in examples or the
   library.
2. Add `foci-per-nucleus` because it exercises object relationships and will
   show whether existing overlap/child-count APIs are ergonomic enough.
3. Add `wound-assay` because it is a compact segmentation-plus-summary
   benchmark with clear image-level outputs.
4. Add `colocalization` to expose existing advanced measurement support through
   a runnable example.
5. Add `object-3d-measurement` after the 2-D examples establish the package and
   benchmark conventions.

## Implementation Status

Implemented example packages:

- `examples/percent-positive/`: deterministic labelled objects, marker
  intensity classification, object CSV, summary JSON and Python reference.
- `examples/foci-per-nucleus/`: deterministic nucleus labels, foci containment
  counts, nuclei/foci CSVs, summary JSON and Python reference.
- `examples/wound-assay/`: deterministic scratch-assay images, open-area
  summaries, per-column profile CSV and Python reference.
- `examples/colocalization/`: deterministic labelled two-channel fixtures using
  `blockflow::ops::colocalization_measurements`, object CSV, summary JSON and
  Python reference.
- `examples/object-3d-measurement/`: deterministic labelled 3-D objects using
  `blockflow::ops::object_geometry_measurements`, object CSV, summary JSON and
  Python reference.

No shared library helper was promoted during this pass. The example-local glue
remained small enough that the current promotion rule did not trigger.

## Evidence Gates

For each example:

- `cargo fmt --check`
- `cargo build -p <example-package> --release`
- example benchmark script on the deterministic small fixture
- comparison script passes when a reference exists

After adding or promoting shared library code:

- focused tests for touched root `blockflow` code
- `cargo clippy --all-targets`
- feature clippy if feature-gated code was touched

## Stop Point For This Goal

Stop this example pass when all five example packages exist, build, and have
documented run/benchmark commands, or when two consecutive examples show the
same missing library abstraction and the next useful step is a library API
change rather than more example wiring.
