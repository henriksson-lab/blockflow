# Paged physical feature index

`feature_index::{Writer, Reader}` stores bounded locally retained physical
features using existing `Table` rows. It complements the resident point store;
it does not collect an organ-wide point vector or triangle table.

## Build and binding

Create a NEW root with `Writer::create(root, geometry, binding_sha256,
per_cell_limit, memory.clone())`. Supply each intended retention cell once,
in strictly increasing ZYX order, through `write_cell(cell, features, budget)`.
Feature iteration within a cell may have any order and streams through a
bounded strongest-K selector. Duplicate anchor/scale identities retain their
best weight, with deterministic physical-coordinate ties. Retention cells
are a fixed policy lattice; changing image IO chunks need not change them.

A feature contains a native integer `at_zyx` anchor, physical `position_xyz`,
finite weight and positive scale. Geometry carries the source shape, cell
shape, physical origin/pitch/units and maximum subvoxel displacement from the
anchor (0 through 1 voxel). The position must agree with that anchor bound.
The meaning of weight and scale belongs to the extracting application.

The caller supplies a 64-hex binding for immutable source/content version,
channel, time, selected level, calibration, detector and retention policy.
The store compares bindings; it cannot independently authenticate image
content. It has no page checksums or concurrent mutation support. Keep sealed
stores immutable. Stable `FeatureKey {cell_zyx, row}` values identify rows
within the SAME sealed index, not across changed extraction/retention policies.

Call consuming `finish()` only after supplying ALL intended cells. The writer
cannot know the caller's intended sparse coverage. An interrupted build,
resource stop, invalid cell order or failed write must never be published as
complete. The initial manifest is unsealed; `Reader::open` refuses it.
Pages and final manifest use temporary-file replacement and file fsync;
this is not a resumable transactional store or a power-loss durability proof.

## Query

`Reader::open(root, expected_binding, memory)` validates a bounded manifest and
occupied-cell directory length. `next_cell(budget)` streams occupied keys in
canonical order; it does not enumerate all empty space. `read_cell` and
`neighbors(position_xyz, radius, limit, budget)` access bounded Table pages.
The directory is binary searched with an independent file handle, preserving
the streaming cursor. Missing indexed pages are errors, not empty regions.

Radius neighborhoods include permitted anchor offsets and filter actual
physical Euclidean distances. Results are nearest-first with deterministic
FeatureKey ties. `truncated` and `matching_points` expose discarded neighbors;
a work/resource stop returns `Outcome::Stopped` and discards partial query
results. An exhausted query is not an empty neighborhood. Callers must carry
incomplete/truncated retrieval evidence into downstream decisions.

## Resource contracts and costs

Use ONE shared `MemoryBudget` for simultaneously live caller payload and
index decoding/query payload. The index leases conservative bounded Table
and output workspaces. Budget stats report cumulative cell visits, attempted
file reads/writes, charged payload bytes and work units; limits are sticky.
They exclude manifest-open overhead, filesystem metadata operations and OS
read amplification, and do not measure physical device IO. Leases are payload
allowances, not verified allocator/process RSS ceilings. Caller-owned feature
vectors after return must be budgeted by the caller while retained.

Files are `manifest.json`, streaming `cells.bin` (24 bytes per occupied key),
and `pages/z/y/x.rows`. Each page uses the existing Table schema/encoding,
with native anchor plus five F64 values (physical XYZ, weight, scale).
`Reader::open` retains the uncached path. Opt-in
`Reader::open_cached(root, binding, memory, ReadCacheConfig)` adds a byte-capped
LRU over directory blocks and decoded feature pages. It uses opportunistic
leases and registers with the existing `MemoryBudget` / `budget::Reclaim` interface;
`try_acquire_reclaiming` can evict it for compute. Cache insertion never waits.
Cache bookkeeping and payload are charged conservatively, with live decode
and caller-output workspace admitted separately. The cache is local to one
immutable reader; mutations require a new reader/binding. A cached hit need not
recheck a file already validated by that reader.

`cache_stats()` reports hits, eviction, declined admission and charged cache
bytes; `clear_cache()` releases entries. Directory block prefetch falls back to
a single key if it would exceed the remaining read-byte allowance. Logical
cell visits and point-matching work remain charged on hits. Only actual file
fetches consume read-call/read-byte limits; directory probes, directory fetches,
page fetches and decodes now have separate counters. Bounded reuse can expose
a different limiting resource; it is not a reason to relax matching criteria.
Dense neighborhoods and tiny cells can still be IO intensive. Profile before
adding parallelism or a secondary descriptor index. Local retention bounds RAM but not total disk
size or global search work; the application still needs cumulative budgets.

## Verification

`cargo test --release --offline --test feature_index` checks deterministic
retention/bytes, duplicate handling, subvoxel cell seams, stable query order,
query-versus-stream cursor independence, tiny-radius empty intervals, growing
empty domain extent, memory/work/IO stops, unsealed publication, wrong bindings,
invalid cell order and missing/oversized pages. It does not establish codec or
whole-process memory bounds, or registration recall on microscopy.

Additional cache checks cover identical ordered radius results and streaming
cursor behavior, bounded eviction, zero-read warm queries, explicit cache
release/shared compute reclamation and directory-prefetch byte limits. Caches
assume sealed input immutability, not concurrent editing of page files.

## Partitioned selection before retention

`Reader::neighbors_by(position, radius, partitions, per_partition, selector,
budget)` visits radius points in bounded pages and applies a deterministic,
bounded-cost caller selector before top-K retention. Return
`Some((partition_index, finite_priority))`, smaller priority first, or `None` to
exclude. FeatureKey breaks ties. Capacities are at most 256 partitions / 4096
total points. Partitions retain independently, with no unused-capacity transfer.
No full in-radius point vector is needed. The caller owns the interpretation
and any additional nonconstant selector work; application geometry is not part
of the storage API.

Results expose `radius_points`, eligible `matching_points`,
`partition_counts` and per-partition truncation through `truncated`. Excluded
points still consume work. Invalid returned bins or priorities are errors;
resource stops discard partial results. The original `neighbors` is a
one-partition distance-ranked wrapper; focused tests check exact output and
ledger equivalence. Selection and radius queries share the existing page cache
and compute admission. This supports spatially distributed or predicate-filtered
neighbors without an organ-wide point/descriptor table.
