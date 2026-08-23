// SPDX-License-Identifier: MIT
//
// Original work. Extracted from `clearmap-rs`, where this lived as
// `parallel_processing/block_ops/` and had grown to fifteen files.
//
// > A mistake in which chunks we fetch should cost **performance**, never
// > **correctness**.
//
// The IO layer already had that property. The provisioning layer did not: a
// block read with too small a halo produces a complete, well-formed, wrong
// volume, with no error and no diagnostic. This crate removes that failure mode
// for block-processed chains by inverting the assertion — instead of checking
// `halo >= reach`, it *derives* what is trustworthy from reach and lets an
// exact-tiling check (`tiling`) be the guard.
//
// Why it is a crate and not a directory
// -------------------------------------
// **Dependency direction, not packaging.** Inside one crate `use
// crate::image_processing::…` is frictionless, so coupling accumulates
// silently. Across a crate boundary every dependency is deliberate, visible in
// `Cargo.toml`, and one-way: `blockflow` must not depend on `clearmap-rs`;
// `clearmap-rs` depends on `blockflow`. The intended direction of travel is
// multi-node, out-of-core execution of general image-processing pipelines, and
// this crate is the part of that which belongs to no particular pipeline.
//
// What that boundary cost, and how each cost was paid
// ---------------------------------------------------
// | outward dependency, before | now |
// |---|---|
// | `clearmap_rs::{ClearMapError, Result}` | `error::Error` — three variants, because the framework only ever raised two of ClearMap's thirty, plus one for foreign backends. `clearmap-rs` converts both ways at the boundary. |
// | `parallel_processing::region_io` | `region` — the box, the two traits and the in-memory implementations moved; the Zarr backend stayed and implements this crate's traits. The npy backend stayed too, and that decision was later reversed and counted: twenty private re-implementations of one array format. It is `npy` now. |
// | `io::source::DataType` | `dtype::Dtype` — a byte-width tag, which is all the cost model and the cache ever wanted from it. |
// | `block_processing::valid_boxes_tile_volume` | `tiling::boxes_tile_exactly` — **reimplemented, not moved**: the original lives in a GPL-translated file. See `tiling`'s header; `clearmap-rs` carries a test that the two agree. |
//
// What could **not** come, and never can
// --------------------------------------
// This crate is MIT; `clearmap-rs` is a translation of GPL-3.0 ClearMap. A
// translated op is a derivative work, and relocating it does not relicense it.
// So the adapters stayed behind: `clearmap_rs::dataflow::binarize` implements
// `BlockOp` over that crate's translated kernels, and depends on five of its
// image-processing modules — which is precisely the boundary this extraction
// drew. An op written from scratch may live here. An op translated from
// anything GPL may not, however generic it looks. See `README.md`.
//
// The layers, and what each is allowed to decide
// ----------------------------------------------
// | module | what it owns |
// |---|---|
// | `error` | This crate's `Error`/`Result`. |
// | `dtype` | The byte width of an element. |
// | `region` | `Region`, `RegionSource`, `RegionSink`, in-memory implementations. |
// | `sidecar` | Per-block output that is not a pixel region: `(stream, phase, block) -> bytes`, plain objects, a declared lifecycle, and a deletion that reports itself. |
// | `tiling` | The exact-tiling predicate the correctness argument rests on. |
// | `op` | `BlockOp` and `Chain`. Reach, execution, output shape, element type, traversal preference and constant algebra on one type, folded over one tree. |
// | `fragment` | `FragmentOp`: the shapes `region -> region` cannot express — `volume -> fragments`, `fragments -> fragments`, `fragments -> volume`. A separate trait, one executor, and a coverage guard on the fragment side because the tiling guard cannot reach it. |
// | `table` | A row store keyed by position, with typed columns. Two states, one query method, two interchangeable indexes, and an order that is a function of the row set rather than of the cut. |
// | `points` | The four-column case of `table`: a position and one weight. Keeps the point's type and its headerless encoding; the store is `table`'s. |
// | `geometry` | The inversion: read extent, trustworthy extent, `valid = core ∩ trustworthy`. |
// | `decomposition` | The **binding** plan — parity-visible, deterministic, data-blind, hashable — plus the cost model used to choose it. |
// | `assemble` | Writing one down: a slot cursor, the names, the per-phase reach, the element type a fragment phase writes, what each phase runs, and the source images. Assembly only — it decides nothing the plan records, and hands back handles so a phase is never addressed by a literal. |
// | `graph` | `(block, phase)` tasks with explicit dependencies. Thousands of nodes, never per-voxel. |
// | `voxels` | What a block *is*: rank 3, element type as a run-time tag. Plus the dynamic-rank buffer a side output goes to, which is a different question. |
// | `env` | The injected environment: real arrays, or a loader that only accumulates cost. |
// | `strategy` | One `Strategy` trait with `decompose` (binding) and `run` (dynamic), and the single executor every strategy shares. |
// | `log` | The `Event` stream — scheduling, op, IO and cache layers — and the log the acceptance criterion is asserted from. |
// | `listener` | `EventListener`, the dispatch set, and two listeners: the order log and a live per-block progress view. |
// | `npy` | The `.npy` file format: header, element type, memory order, and both traits over it. Both orders read, neither transposed. |
// | `observed_io` | `RegionSource`/`RegionSink` decorators, so IO outside the executor emits through the same trait. |
// | `export` | The order log as JSON, with the cross-language schema documented in its header. |
// | `animate` | The seam to the bundled renderer, which turns an exported log into a movie. Opt-in; the crate does not depend on it. |
// | `probes` | Synthetic ops that prove the framework without a real kernel. |
// | `synthetic` | A generated volume with its ground truth: intensities, exact labels, an object table. Placed in global coordinates, rendered by region. |
// | `agreement` | How a produced labelling relates to a known-correct one — matched, split, merged, missed, spurious — by overlap, because ids never match. |
// | `gui` | *(feature `gui`)* An HTTP server over the progress listener, and the browser view it feeds. A replay and a live run answer the same endpoints. |
// | `cache` | One `(array, chunk)`-keyed LRU, two tiers, one byte budget taken opportunistically. |
// | `net` | One bind policy, shared by both servers, and how a coordinator decides what address to publish. |
// | `distributed` | A coordinator, workers that pull, four rendezvous backends, and a local multi-node mode that runs all of it as separate processes. |
// | `prefetch` | A scheduler over declared future reads, not a predictor. |
//
// **The framework was proven before a kernel was attached**, and that ordering
// is the point: `probes`, `env`, `strategy` and `tests` import nothing
// translated, so a failure there is a framework failure. After the extraction
// that is enforced by the compiler rather than by discipline — there is no
// translated code in this crate to import.

/// How a produced labelling relates to a known-correct one. Overlap-based,
/// because label ids never agree between two labellings of the same volume.
pub mod agreement;
pub mod animate;
/// Assembling a plan: the slot cursor, the names, the per-phase reach, the
/// element type a fragment phase writes, what each phase runs, and the images a
/// phase reads besides its own input. **Assembly, not planning** — it decides
/// nothing a `Decomposition` records, and everything it produces goes through
/// the same five guards the executor runs.
pub mod assemble;
pub mod budget;
pub mod cache;
pub mod cpu;
pub mod decomposition;
/// Multi-node execution: a coordinator, workers that pull, and pluggable
/// rendezvous. The HTTP *server* half is behind the `distributed` feature; the
/// protocol, the coordinator's state machine, the handout policy, the cache
/// model, the client and the worker loop are not.
pub mod distributed;
pub mod dtype;
pub mod env;
pub mod error;
pub mod export;
/// Ops whose input or output is not a pixel region: `volume -> fragments`,
/// `fragments -> fragments` and `fragments -> volume`, scheduled by the same
/// executor over the same task DAG. A second trait beside `BlockOp`, because
/// they are a different shape rather than a wider signature.
pub mod fragment;
pub mod geometry;
pub mod graph;
/// An HTTP server and a browser view over the event stream — live, or replayed
/// from an exported order log. Behind the `gui` feature; with it off this crate
/// is unchanged and pulls no extra dependency.
#[cfg(feature = "gui")]
pub mod gui;
/// A blocking HTTP/1.1 server on ordinary threads: one thread per connection,
/// taken at accept. Not behind a feature — it has no dependency and the
/// no-feature configuration is the one everything else is measured against —
/// and see its header for the measured defect it exists instead of.
pub mod http;
/// One phase that runs an unknown number of substages, with more than one
/// operand available at every substage. The shape an iteration takes when its
/// step depends on both a running estimate and something fixed — which
/// `Chain::Sequence` cannot express, because a sequence hands each step only its
/// predecessor's output. The phase's external reach is **one substage's**, and
/// live storage is two private buffers whatever the substage count turns out to
/// be.
pub mod iterate;
pub mod listener;
pub mod log;
pub mod net;
/// The `.npy` array file format: a header, an element type, a memory order and
/// a flat contiguous buffer. Whole arrays of any rank, `Voxels` for the rank-3
/// case, and `RegionSource`/`RegionSink` implementations that read and write a
/// box without holding the volume. **Both memory orders are handled and neither
/// is transposed**; a caller that can take only one says so and gets a refusal
/// naming the order it found.
pub mod npy;
pub mod observed_io;
pub mod op;
/// Image-processing operations: voxelwise combination, rank filtering,
/// morphology, windowed statistics on a globally anchored sample lattice, and
/// thresholding against one. Each is a free function generic over the element
/// type as far as its algorithm allows, with a thin `BlockOp` over it; every
/// `reach` is derived from the op's own parameters and none can be configured.
pub mod ops;
/// A set of positions, written per block and read by region. The one node whose
/// partitioning would otherwise be "whichever block emitted it", which is a fact
/// about the run rather than about the data; this stores by position instead,
/// with a canonical order that is a function of the point set alone.
pub mod points;
pub mod prefetch;
pub mod probes;
/// What an operation reads beyond what it writes, and in which coordinate
/// space it is counted. See [`reach::Reach`].
pub mod reach;
pub mod region;
/// Block-keyed output that is not a pixel region: `(stream, phase, block) ->
/// bytes`, on the environment beside the region writes. Storage only; what
/// produces and consumes fragments is `fragment`.
pub mod sidecar;
/// Cutting one block into slabs run on separate threads: the mechanism below
/// the block, and the arithmetic that says what a cut costs. Not a policy —
/// nothing here decides when to slice. See `docs/design/intra-block.md`.
pub mod slab;
/// Measured cost coefficients, accumulated from real runs and fed back into
/// planning. Nanoseconds per unit of *declared* cost, keyed by machine and
/// persisted across runs; an absent or empty store leaves the shipped constants
/// — which are a seed, not a fallback — exactly where they are.
pub mod statistics;
pub mod strategy;
/// A generated volume whose answer is known by construction: intensities, an
/// exact label volume and an object table, from a seed and a shape. Objects are
/// placed in global coordinates and rendered by region, so a block and the whole
/// volume agree bit for bit.
pub mod synthetic;
/// A row per object, keyed by position, with typed columns — a size, a total, a
/// measure — written per block and read by region. The node kind a point set
/// was a one-column special case of; `points` is now exactly that special case.
pub mod table;
pub mod tiling;
/// A block's data: rank 3, element type carried as a tag. Also the
/// dynamic-rank buffer a side output goes to, which is a different question and
/// says so.
pub mod voxels;
/// Images as Zarr v3 arrays on a filesystem store: the `Environment` that moves
/// bytes. Behind the `zarr` feature; with it off this crate is unchanged and
/// pulls no extra dependency.
#[cfg(feature = "zarr")]
pub mod zarr_env;

#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod observability_tests;
#[cfg(test)]
mod tests;

pub use agreement::{compare_labels, Agreement, Matched, Merged, Split};
pub use animate::{render, RenderRequest, View};
pub use assemble::{Assembly, ImageId, Phase, PlanBuilder};
pub use budget::{Class, Lease, MemoryBudget};
pub use cache::{
    ArrayId, ArrayPolicy, CacheElement, CacheStats, CachingSource, ChunkCache, ChunkFetcher,
    ChunkKey, Codec, DeflateCodec, RegionSourceFetcher, Tier,
};
pub use decomposition::{
    check_block_constraints, check_chunk_exclusive_writes, check_output_shapes,
    check_source_images, constraint_for, cuttable_axes, halo_spans_axis, is_planning_barrier,
    predicted_cost, reaches_whole_axis, refined_ladder, splittable_axes, BlockLadder, Constraints,
    CostModel, Decomposition, PhaseDecomposition, PhaseTraffic,
};
pub use distributed::{
    Assignment, ChunkGrid, Coordinator, Handout, HandoutPolicy, JobSpec, JobStatus, ModelledCache,
    Rendezvous, SharedVolumes, WorkerOptions, WorkflowFactory, WorkflowSpec,
};
pub use dtype::Dtype;
pub use env::{AccountingEnvironment, ArrayEnvironment, BlockBuf, Environment};
pub use error::{Error, Result};
pub use export::{
    event_from_json, event_json, order_log_to_json, write_order_log_json, ExportMeta,
};
pub use fragment::{
    append_fragment_phase, check_fragment_coverage, check_phase_work, fold_fragments,
    fragment_only, fragment_phase, neighbourhood, neighbourhood_size, BlockOutput, BlockView,
    Coverage, FragmentInput, FragmentOp, FragmentOutput, PhaseWork, SeamFold, SourceBlocks,
    StreamCoverage,
};
pub use geometry::{BlockCore, BlockGeometry, BlockGrid};
pub use graph::{SourceDep, Task, TaskGraph};
pub use iterate::{
    check_iterative, iterative_phase, substage_reach, IterativeOp, Operand, Substage,
    SubstageLimit, SubstageOperand,
};
pub use listener::{BlockProgress, EventListener, LatestOpPerChunk, OrderLog, ProgressKind};
pub use log::{Event, ExecutionLog, Stats};
pub use npy::{
    descr_of, read_array, read_array_file, read_array_file_as, read_array_from, read_array_mapped,
    read_array_mapped_file, read_array_mapped_from, read_elements, read_elements_file,
    read_elements_from, read_header_file, read_voxels, read_voxels_file, read_voxels_from,
    write_array, write_array_file, write_array_to, write_elements, write_elements_file,
    write_elements_to, write_voxels, write_voxels_file, write_voxels_to, Elements, ElementsVariant,
    Endian, Header, NpyElement, NpySink, NpySource, Order, OrderPolicy,
};
pub use observed_io::{ObservedSink, ObservedSource};
pub use op::{
    Anchor, BlockConstraint, BlockOp, Chain, Combine, Geometry, InputMap, Output, Placement,
    SideBlock, SourceInputs,
};
pub use points::{
    decode_points, encode_points, Layout, Point, PointIndex, PointStore, State, WORDS_PER_POINT,
};
pub use prefetch::{AccessPlan, BlockPlan, PlanHandle, PrefetchStats, Prefetcher, RegionRequest};
pub use probes::{
    region_of, AffineOp, BlockSummaryOp, CappedSpreadOp, DecimateOp, DriftingSumOp,
    FragmentReduceOp, IdentityOp, MandatedExtentOp, NeighbourFoldOp, NonZeroOp, OpaqueOp,
    RegionMergeOp, RegionSumOp, SideOutputOp, SpreadLatticeOp, WindowSumOp,
};
pub use reach::{AxisReach, Frame, Reach, Space, Units};
pub use region::{ArrayRegionSink, ArrayRegionSource, Region, RegionSink, RegionSource};
pub use sidecar::{
    Discarded, FileSidecars, FragmentKey, Lifecycle, MemorySidecars, SidecarBackend, Sidecars,
    StreamRemoval,
};
pub use statistics::{
    observed_nanos, Coefficient, MachineKey, Observed, PlanIdentity, Provenance, Recorder,
    RunObservations, Snapshot, Statistics, Term,
};
pub use strategy::{
    execute, execute_observed, execute_phases, execute_task, execute_task_of, ArrayRef,
    Enumerating, Greedy, Hints, Materialising, PartitionSearch, Plan, SchedulePriority, Strategy,
    TaskOutcome, Trivial, Workflow,
};
pub use synthetic::{
    IntensitySource, LabelSource, Object, ObjectRecord, Rendered, Scene, SceneSpec,
};
pub use table::{
    encoded_schema, Column, ColumnType, Row, RowBuilder, Schema, Table, TableIndex, Value,
};
pub use tiling::boxes_tile_exactly;
pub use voxels::{SideBuf, VoxelElement, Voxels, VoxelsMut};
#[cfg(feature = "zarr")]
pub use zarr_env::{chunk_for_block, zarr_data_type, AttachedImage, Window, ZarrEnvironment};
