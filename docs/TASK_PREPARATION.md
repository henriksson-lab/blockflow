# Metadata-first task execution

`FragmentOp::prepares()` opts a phase into `prepare(&PreparationView)`.
Existing operations default to dense execution. Preparation runs only after a
task's declared dependencies are ready, before pixel prefetch and large task
leases. The local executor prepares at most one wave ahead and reuses its metadata
at execution. Standalone task execution (also used by workers) prepares through
the same helper. Prepared Run windows use `Environment::prefetch_prepared` with stable static-task
ranks; completed tasks submit no prefetches. The Zarr environment implements this
without restarting the rank at each wave. The static DAG and source declarations remain conservative;
preparation never adds a source or widens a declared extent.

Return one of:

* `Run`: read the original extents and apply the existing kernel.
* `ReadWindows`: replace auxiliary-source reads with specified windows. The
  primary pixel input, if any, keeps its full halo. Every window must lie in the
  static source footprint. Include every source the kernel needs.
* `Complete`: emit properly encoded fragments and an optional constant image.
  A pixel writer must supply a finite constant. Every-block streams must still
  have a fragment. Normal stream, size, coverage and task-completion validation
  applies. Background-valued cores avoid allocation/writes when the environment
  guarantees that unwritten storage returns that background; other constants
  are written normally.

Preparation cannot access pixel buffers or allocate output arrays. Metadata
selectors distinguish missing/unknown from empty. `own_rows_empty` decodes the
actual row-table encoding. `complete(stream)` checks that the whole requested
neighbourhood was found; a painter cannot infer an empty core from its owned
fragment alone. `known_uniform` asks authoritative storage metadata, never scans
pixels. Streaming ops can use that metadata but have no gathered fragments.
Unordered folds have both their preparation decisions and normal outputs checked
with reversed metadata order.

`SourceBlocks::covering(image, region, fallback)` returns the containing buffer
and its actual global extent. Use that origin to index cropped reads; `get`
refuses ambiguous multiple-window sources. `coalesce_read_windows` merges
intersecting boxes by image; disjoint windows stay separate. The executor also
merges windows sharing storage chunks when the merged box costs at most twice
their combined voxels; chunk caches handle the remaining shared chunks. Object windows must include their complete
filtering and normalization context, including any required content outside the
mask. Mask gating does not zero the input to a filter or truncate its halo.

`ops::activity::ActivityOp` generates versioned occupancy flags for a mask or
label image, one per core, once in the run's DAG. Any nonzero value is active.
Its stream can feed multiple consumers. `TabulateValuesOp::with_activity` uses a
summary of its own label source on the same grid and emits an encoded empty
partial for inactive blocks. Recompute summaries with each run/source generation;
do not substitute a persistent sidecar from an earlier image version. A known
zero label block can also skip directly from storage metadata.

The initial consumers are `PartitionOp` (owned rows, cropped object windows),
`RunLabelsOp` and `PartitionEvidenceOp` (all neighbouring runs), and
`TabulateValuesOp` (mask activity or known-zero labels). `Unprepared(op)` preserves
the same kernel and static plan while disabling preparation for a dense reference.

Completed tasks lease metadata/output memory rather than image/compute scratch.
For active tasks, only exact avoided source-buffer bytes are subtracted from the
conservative phase charge; compute scratch is retained. Workers executing a
single task make the same read/output decisions; remote admission and the static
planner continue using conservative charges. This does not promise a hard RSS
limit for user-defined preparation metadata, which must remain small.

`TaskPrepared` records preparation duration, completion decisions, original read
voxels and requested read voxels; `ExecutionLog::preparation_stats` totals them.
`FragmentApplied` records compute time, including unordered verification. Existing
read/write events retain requested bytes, chunks and durations. These count
requested regions, not physical disk traffic or cache misses; use cache events for
those. Summed stage times are worker time, not elapsed pipeline time. Both new
events round-trip through JSON export and the worker event transport.

Validation: `tests/task_preparation.rs` compares dense/prepared execution across
worker counts and grids, checks unknown metadata fallback, runtime bounds,
neighbour-owned runs, background cores, fresh mask summaries and disjoint windows.
Its ignored `benchmark_sparse_and_dense` measures both sparse and fully occupied
masks in release builds. Existing interior tests cover exact parent ownership and
grid-independent territories. The static planner is deliberately not being taught
speculative occupancy estimates in this change.


## Release measurements (2026-10-21)

The ignored synthetic benchmark uses a 128-cubed label/value fixture, 32-voxel
blocks, four workers and five repetitions per case. Median end-to-end times:

| Mask | Dense reference | Prepared | Requested voxels, dense / prepared |
| --- | ---: | ---: | ---: |
| One foreground voxel | 13.05 ms | 8.62 ms | 6,291,456 / 2,162,688 |
| Fully occupied | 38.43 ms | 37.00 ms | 6,291,456 / 6,291,456 |

Both paths include the occupancy-summary phase; it scans the mask once when
storage cannot establish uniformity. The fully occupied case checks that metadata
gating leaves dense reads intact. Timings are machine-specific.

A real 625x824x811 WT_5 image, four workers and unchanged segmentation settings,
requested 39.34 GB with dense reads versus 0.817 GB with prepared object windows
(97.92% less). Executor time was 123.86 vs 76.02 s; total including validation but
excluding PNGs was 137.31 vs 88.62 s. These are indicative single runs on a shared
machine; the reference overlapped release compilation. Partition labels, boundary
evidence and all three CSV tables were byte-identical: 72 parents, 872 territories,
4,671,909 interior voxels. Nothing was republished. Detailed results are in the
imaging_nedal repository's
`analysis/partition_review_2026-10-21/sparse_execution_benchmark/`.

The full default release library suite passed (1,478 tests; 25 ignored), all 12
new executor tests passed, and distributed/Zarr all-target release checks passed.
