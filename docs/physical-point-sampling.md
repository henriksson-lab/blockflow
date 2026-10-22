# Sampling arbitrary physical points

`sampling::sample_points` evaluates typed chunk-backed sources at batches of XYZ physical positions and returns trilinear values and physical gradients. This complements `ops::plane_sample`, which plans regular planar output through the executor. Array/chunk shapes and chunk starts are ZYX; physical coordinates, calibration and gradients are XYZ. Integer indices denote pixel centres. The caller supplies consistent units.

Implement `SampleSource` for one immutable calibrated image. Its layout declares shape, chunk lattice, actual dtype, origin, spacing and additional source workspace. Return exact clipped typed chunks and optional validity masks. `EnvironmentSampleSource` adapts the existing typed `Environment::read` and its cache, avoiding a second storage/cache implementation. Accounting-only environments cannot supply numerical samples.

The sampler groups all interpolation corners by source chunk and reads each chunk once per batch. It reserves eight corner requests per point and one clipped chunk plus optional mask. The sort is in place. Values and gradients have independent validity: an exact observed pixel remains a valid value when a masked neighbor prevents its derivative. Singleton axes have zero derivative. Upper-boundary derivatives use the inward interpolation cell. Nonfinite coordinates are outside-domain samples; nonfinite contributing pixels are invalid data. Arithmetic overflow is an error.

`SamplingBudget` caps planned sampler/source workspace, per-batch chunk reads and cumulative decoded IO charge. Memory/IO preflight failures occur before reading source chunks. An IO/budget error must not be interpreted as absent tissue. Discard output after any error. Empty batches require no chunk buffers or reads.

The source workspace declaration must include cache, encoded data, decoder scratch and copies beyond the returned typed chunk/mask. This is a contract supplied by the source, not codec verification or measured process RSS. Caller-owned coordinates/results, metadata, allocator overhead and unrelated data remain outside kernel accounting. Concurrent callers need shared admission outside this API. IO charges include a mask byte even when the source omits its optional mask; cache hits remain logical chunk reads, not filesystem traffic. No source-content identity or persistent feature-index policy is introduced here.

Integer samples convert numerically to f64; uint64/int64 values outside its exact integer range lose precision. F16 is refused because `Voxels` does not have a native half-precision variant.

`tests/sampling.rs` covers typed calibration, seams, edges, masks, separate derivative validity, singleton axes, empty batches, preflight failures and constant planned memory over enormous procedural volumes. Storage adapters must additionally verify their actual decoder/cache allocations and IO behavior.

`sampling_working_bytes(layout, point_count)` exposes the same checked allocation
plan used by `sample_points`, without allocating or reading chunks. Consumers
can reserve their own positions/results and other live sources' workspace before
admitting a batch. An empty batch reports zero; invalid layouts and arithmetic
overflow remain errors. This does not turn a declared decoder reservation into a
verified codec or process memory ceiling.
