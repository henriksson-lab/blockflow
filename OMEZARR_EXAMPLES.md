# OME-Zarr Example Conversion

Goal: every user-facing Blockflow example should teach planned execution over a
prepared Zarr/OME-Zarr-style fixture. Resident in-memory binaries may remain as
debug or reference code, but benchmark scripts should not present them as the
normal library path.

## Policy

- Prepare benchmark inputs as Zarr before measured Blockflow processing.
- Attach prepared arrays with `ZarrEnvironment::attach` for planned execution.
- Keep original TIFF/PNG/CSV fixture downloads outside the measured path.
- Report the prepared store path in benchmark JSON so reruns are auditable.
- Use resident direct-op code only as an internal comparator or fixture
  generator, not as the primary example.

## Status

- `examples/cellprofiler-human`: fixture/materialization path converted. The
  planned probe can ensure or attach a prepared rank-3 Zarr input and
  materializes through `ZarrEnvironment`; `scripts/prepare_input_zarr.sh` gives
  benchmark jobs an explicit preparation step.
- `examples/opencv-pipeline`: converted. The Blockflow side prepares the input
  image as Zarr, attaches it, and runs planned optional translation, smoothing,
  Otsu thresholding, morphology open/close, small-object filtering, connected
  components and shape measurement.
- `examples/skimage-pipeline`: converted. The Blockflow side prepares the input
  image as Zarr, attaches it, and runs planned optional translation, smoothing,
  Otsu thresholding, morphology open/close, small-object filtering, connected
  components and shape measurement.
- `examples/dask-image-pipeline`: converted. The Blockflow side prepares the
  input image as Zarr, attaches it, and runs planned optional translation,
  smoothing, Otsu thresholding, morphology open/close, small-object filtering,
  connected components and shape measurement.
- `examples/imglib2-pipeline`: converted. The Blockflow side prepares the input
  image as Zarr, attaches it, and runs planned smoothing, Otsu thresholding,
  small-object filtering, connected-component labelling and shape measurement.
- `examples/foci-per-nucleus`: converted. The binary prepares per-image nucleus
  labels plus focus-count and focus-intensity Zarr arrays, attaches them, and
  runs planned shape/intensity reductions to produce nucleus area, foci counts
  and foci intensity sums.
- `examples/percent-positive`: converted. The binary prepares per-image label
  and marker Zarr arrays, attaches marker as a supplied input, runs planned
  shape/intensity measurements, and classifies positivity from planned rows.
- `examples/object-3d-measurement`: converted. The user-facing binary prepares
  per-image label Zarr fixtures, attaches each one, and runs planned
  object-geometry measurement phases before writing the existing CSV/summary
  outputs.
- `examples/colocalization`: converted. The binary prepares label, channel A
  and channel B Zarr arrays per image, attaches them as one primary and two
  supplied inputs, then runs planned colocalization measurement phases.
- `examples/wound-assay`: converted. The binary prepares per-image scratch assay
  images as Zarr, attaches each one, runs a planned fixed-threshold open-wound
  mask, and derives area/profile CSVs from the planned mask output.

## Verification Notes

- Converted examples now expose explicit Zarr/OME-Zarr-style input locations in
  their Blockflow-side binaries or benchmark scripts.
- Comparison examples still keep external framework reference code resident
  because those scripts measure the comparison frameworks, not Blockflow's user
  path.
- Several examples now duplicate small image-to-Zarr helpers. A shared helper is
  reasonable once the example surface settles, but it was not required for the
  conversion.

## Follow-ups

- Consider a small example-support crate if new examples repeat the same
  `ensure_array_zarr` and chunk parser code again.
- Promote axis profile or point-to-label joins into library ops only after a
  second workflow needs the same operation.
