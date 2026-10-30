# Cellpose optimization plan

## Objective

Reduce end-to-end CUDA time for the normal Blockflow Cellpose workflow while
preserving its masks, label ownership, table rows, and OME-Zarr output. Keep one
model per GPU initially. Add concurrency only where measurements show that it
can overlap useful CPU or transfer work with GPU execution.

The optimization target is the release build of
`examples/cellpose-ome-zarr`, using the real DAPI subset and the same CP-SAM
checkpoint and segmentation parameters as the baseline below.

## Baseline and diagnosis

Measured on 2026-09-22 on a Quadro RTX 5000 with the 2048 x 6144 DAPI subset:

| Runner | Workers | End-to-end wall | Peak host RSS | Cells |
|---|---:|---:|---:|---:|
| Python Cellpose 4.1.1 | 1 | 58.58 s | 2.79 GiB | 1,145 |
| Blockflow release | 1 | 49.64 s | 1.44 GiB | 962 |
| Blockflow release | 2 | 53.23 s | 1.69 GiB | 962 |

The Python runner evaluates the entire strip as one image and does not write a
label layer or table. Blockflow uses three overlapping outer blocks, normalizes
each block independently, owns objects by centroid, and writes the normal
annotation products. Cell counts therefore measure each runner's stable output;
they are not expected to match between the two tiling schemes.

Two Blockflow workers are 7.2% slower than one. The current backend locks the
`CellposeModel` around the complete `eval` call. Cellpose-rs then performs CPU
preparation, upload, inference, synchronous downloads, dynamics, cleanup, and
flow rendering inside that call. Transfers and network execution use one CUDA
stream. A second executor worker can prepare Blockflow's input array and handle
the returned labels, but cannot overlap the next upload or Cellpose CPU work
with the current inference.

GPU capacity is not the immediate constraint: the card has 16 GiB and the
observed model workload uses roughly 3 GiB. Active inference already occupies
the GPU heavily, so duplicating the entire model is not the first optimization.

## Measurement rules

1. Build and run Rust code with `--release`. Never use debug timings.
2. Use the same source pixels, checkpoint, thresholds, halo, block size, and
   batch size when comparing revisions.
3. Record end-to-end wall time, Cellpose stage timings, cell count, peak host
   RSS, and peak GPU memory.
4. Run the final candidate at least three times after one untimed warm-up and
   report the median. Short exploratory measurements may be single runs.
5. Compare one and two Blockflow workers. Do not enable two workers by default
   unless the median improves without changing output.
6. Require identical Blockflow label arrays and table rows before and after an
   optimization. The Python count remains contextual rather than a parity
   assertion because its outer tiling differs.
7. Keep each benchmark batch below three minutes where practical. Use the full
   slide only after the subset demonstrates a material improvement.

## Phase 1: expose actionable timings

Add opt-in profiling to the Blockflow Cellpose adapter and example using
cellpose-rs's existing `EvalTiming` and `RunNetTiming` structures.

- Add a backend profiling mode that calls `eval_with_timing` and aggregates
  timings safely across blocks.
- Report at least conversion, normalization, stack construction, tile
  preparation, tensor upload, forward pass, prediction download, style
  download, tile averaging, output extraction, dynamics, cleanup, and flow RGB.
- Keep profiling disabled by default because device synchronization used for
  precise stage timings can perturb execution.
- Write one machine-readable JSON summary from the example, including the
  complete command parameters and end-to-end wall time.
- Run one profiled release pass on the subset and use its dominant avoidable
  stages to select Phase 2 work.

Deliverable: a timing report whose stage totals explain the major gaps between
GPU inference bursts. This instrumentation remains available for later
regression work.

## Phase 2: add a masks-only Cellpose path

Blockflow consumes only instance masks. The general Cellpose API currently
also materializes flows, flow RGB, cell probability, and styles.

Implement this first in `cellpose-rs`, then update Blockflow's pinned
cellpose-rs revision.

- Add a public masks-only evaluation API rather than changing the behavior of
  the Python-compatible `eval` API.
- Avoid style download and aggregation when the caller does not request it.
- Avoid flow RGB construction and final public flow/cell-probability copies.
- Retain the flow and cell-probability data required internally by dynamics and
  mask cleanup.
- Return the same mask dtype, shape, IDs, and errors as the existing API.
- Switch `CellposeBackend` to the masks-only API.

Verification:

- Focused cellpose-rs tests compare masks-only results byte for byte with
  `eval(...).masks` on representative small CPU fixtures.
- One CUDA integration case compares the two paths when a device is available.
- Blockflow's subset label arrays and table rows remain identical.
- Record the release wall-time and memory change. Keep the change even if its
  speedup is small when it removes substantial unused allocation and work.

## Phase 3: reduce serialized CPU work

Use Phase 1 timings to move expensive CPU-only preparation and postprocessing
outside the model-wide critical section.

Preferred design in cellpose-rs:

1. Split masks-only evaluation into explicit preparation, network execution,
   and mask finalization values.
2. Make preparation and finalization independent of mutable model state.
3. Lock only the network/device submission portion in Blockflow, or expose a
   cellpose-rs runner that owns this narrow serialization internally.
4. Allow worker N+1 to normalize and prepare tiles while worker N runs the
   network, and allow worker N-1 to finalize masks at the same time.
5. Bound prepared and pending results to one item on each side of inference so
   additional workers cannot cause unbounded host memory growth.

Do not expose partially processed Cellpose internals directly through the
generic `SegmentBackend` trait. Put the pipeline behind a Cellpose-specific
backend implementation or add a general staged-backend abstraction only if a
second backend needs the same lifecycle.

Verification:

- Threaded and single-worker results are byte-identical.
- A failure in any stage cancels the bounded pipeline and returns the original
  error without deadlock.
- Two workers must improve the three-run median by at least 5% before becoming
  the documented recommendation. Otherwise retain one worker as the default.

## Phase 4: tune batching and copies

After unused work and broad serialization are removed:

- Benchmark batch sizes 8, 16, and 32 on the subset. Include peak GPU memory and
  mask equality in the result.
- Remove avoidable ndarray layout copies found by the profiler, while keeping
  the ndarray 0.16/0.17 boundary localized in the adapter.
- Reuse bounded host work buffers where profiles show meaningful allocation or
  copy cost.
- Select a new default batch size only when it improves the median and remains
  comfortably within the 16 GiB test GPU.

## Phase 5: asynchronous CUDA transfer pipeline

Attempt this only if Phase 1 timings still show material transfer or
serialization gaps after Phases 2–4.

- Add pinned host buffers for tile input and prediction output.
- Use separate CUDA copy and compute streams with explicit events.
- Double-buffer batches so upload N+1 can overlap compute N and download N-1.
- Keep at most two batches resident beyond the active batch and account for
  their GPU memory in the batch-size limit.
- Preserve a synchronous CPU path and a synchronous CUDA fallback.
- Use Nsight Systems or equivalent event tracing to verify actual overlap;
  executor worker count alone is not evidence of overlap.

This work belongs primarily in cellpose-rs/Candle integration. Do not duplicate
CUDA stream management in the OME-Zarr example.

## Decision on multiple model instances

Test two model instances only after the staged single-model pipeline. Although
two models should fit in 16 GiB, concurrent forwards may compete for an already
busy GPU and double persistent weights and activation memory. Adopt a model
pool only if profiling shows unused compute capacity and its three-run median
beats the single-model pipeline with identical results.

## Completion criteria

The plan is complete when:

- Blockflow uses the cellpose-rs masks-only API.
- Profiling attributes the important CPU, transfer, network, and dynamics time.
- The optimized one-worker workflow preserves its pre-optimization label array
  and table exactly.
- The subset is benchmarked with release builds using the measurement rules.
- The chosen worker count and batch size are documented from measured results.
- `BENCHMARKS.md` records the final Blockflow/Python ratio and the Blockflow
  worker comparison without presenting the differing cell counts as semantic
  parity.
- The whole-slide example documentation uses the selected defaults.

An end-to-end median improvement of 10% is the target. If the measured safe
improvements do not reach 10%, document the stage breakdown and stop before
adding complex asynchronous CUDA code whose expected gain is smaller than its
maintenance cost.

## Implementation result

Implemented on 2026-09-22:

- Blockflow can write release-only per-block and aggregate Cellpose timing JSON.
- Cellpose-rs provides `eval_masks` and `eval_masks_with_timing`; its Candle path
  skips style downloads and Blockflow no longer builds unused flows, cell
  probability output, styles, or flow RGB.
- The optimized output is byte-identical to the prior Blockflow level-0 labels
  and its linked table is identical. A focused CUDA integration test also
  compares masks-only and full evaluation directly.
- The matched full-output median was 51.34 seconds and the masks-only median was
  48.92 seconds, a 4.7% improvement. End-to-end Blockflow is 1.20x faster than
  the recorded 58.58 second Python Cellpose run.
- Batch sizes 16 and 32 did not improve over 8. Two workers had a 53.57 second
  median, 9.5% slower than one worker, and used more host memory.
- Detailed timings show 40.35 of 50.04 seconds in network forward, 0.27 seconds
  in upload plus prediction download, and 3.00 seconds in the flow RGB work now
  removed.

The measured improvement did not reach 10%. Sharing the immutable model without
the coarse lock was also tested: it did not improve wall time and increased
host memory. The transfer ceiling is too small to justify pinned buffers,
multiple model instances, or a separate CUDA copy stream on this GPU, so Phases
3 and 5 stop at their measurement gates.
