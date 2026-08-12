// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written against the `zarrs` API; nothing here is
// derived from any other consumer of it.
//
// A **storage** environment: levels are Zarr v3 arrays on a filesystem store.
//
// Why this exists
// ---------------
// The three environments this crate shipped with are `ArrayEnvironment` (real
// values, whole volumes in memory), `AccountingEnvironment` (no values at all,
// costs only) and the multi-node `SharedVolumes`. Every correctness claim the
// crate makes was therefore made about arrays that had already been allocated in
// full — which is exactly the case out-of-core execution exists to avoid. Until
// something here could read a level off a disk and write the next one back,
// "out-of-core block processing" was a description of the scheduling, not of the
// data.
//
// This is the missing half, and its acceptance criterion is stated as a
// negative: **the storage layer must be invisible to the answer.** A chain run
// through this environment must produce, byte for byte, what the same chain
// produces through `ArrayEnvironment`. `tests/zarr_env.rs` asserts exactly that
// over `synthetic::Scene` data, at several block sizes.
//
// Levels are arrays
// -----------------
// Level `l` is the array at `root/level<l>`. Level 0 is the workflow input and
// is created by [`ZarrEnvironment::create`]; [`Environment::prepare`] creates
// levels 1..=n from the plan, each at **its own** `volume_at` and `dtype_at`,
// because a phase owns its volume and its element type and an environment that
// allocated one shape for every level could not host a plan that changes either.
//
// The fill value is `Voxels::unwritten`'s sentinel — NaN for the floats, the
// maximum for the integers — so that a level read back before it was written is
// loud in storage for the same reason it is loud in memory. Zarr gives this for
// free: a chunk nobody wrote is not a file, and reads back as the fill value.
//
// The bug this is arranged against
// --------------------------------
// `zarrs` 0.23.13 **loses data on concurrent partial-chunk writes**. In
// `array_sync_readable_writable.rs` the author scaffolded a per-chunk lock and
// left it commented out (`// Lock the chunk`, four lines, at the top of
// `store_chunk_subset_opt`'s slow branch). The path underneath it is a literal
// read-modify-write: decode the whole chunk, patch the requested sub-box into
// the decoded buffer, re-encode and overwrite the whole chunk. Two threads doing
// that to two halves of one chunk both read the old chunk and both write a
// whole one; the loser's half is gone, with no error and no diagnostic.
//
// That matters more here than in most places, because this crate's executor
// writes blocks from a rayon pool **by design**, and a block's valid region is
// the plan's business, not the chunk grid's — so partial-chunk writes are the
// ordinary case, not an edge case. `tests` below provokes the race and watches
// it fail without the guard.
//
// What the guard is, and what it costs
// ------------------------------------
// **Serialise per chunk, and only where read-modify-write is what will happen.**
// Before a write, the chunks the region touches are split into those it covers
// *entirely* and those it covers only in part:
//
// * A chunk covered entirely is written by `store_chunk` — a blind overwrite,
//   no decode, nothing to lose — and, under this trait's disjoint-write
//   contract, no other writer can be touching it. **No lock is taken.** That is
//   what makes a chunk-aligned write the fast path in a way that is load-bearing
//   rather than advisory.
// * A chunk covered in part is the read-modify-write above. Its lock is taken
//   for the duration of the store.
//
// The locks are a fixed stripe array rather than a map keyed by chunk, so
// nothing is allocated or evicted on the write path and there is no map lock to
// contend on. Two unrelated chunks may collide on a stripe; the cost of a
// collision is that two writes which could have run at once do not, which is
// performance, never correctness — the same trade this crate makes everywhere
// else. Stripes are locked in ascending index order after deduplication, which
// is a total order, so a write that touches several chunks cannot deadlock
// against another that touches an overlapping set.
//
// The alternative was to require block writes to be chunk-disjoint. It was
// rejected because it is not this layer's decision to make: the block grid comes
// from the plan and the chunk grid from the storage, and refusing every plan
// whose blocks straddle a chunk would refuse most plans. What is kept from that
// idea is the accounting — [`ZarrEnvironment::serialised_writes`] says how often
// the slow path was taken, so a caller who *can* align their blocks can see that
// it worked.
//
// **What the guard does not cover**, stated rather than implied: a read
// concurrent with a write of the same array. Reads take no locks. The contract
// this environment is written to is the executor's — a level is written by one
// phase, sealed by `finish`, and read by the next — and `RegionSink` already
// documents writes as disjoint. A caller who reads an array while another thread
// is writing it is outside that contract and this guard does not rescue them.
//
// Compression, and why it is per level
// ------------------------------------
// [`Compression`] says how a level's chunks are encoded; [`CompressionPolicy`]
// says which level gets which. Per level rather than per environment, because
// the levels of one plan are not one kind of data: a `bool` mask and the
// `float64` it was thresholded out of sit in the same run and want opposite
// answers, and a single switch would have to be wrong for one of them.
//
// The default is **derived, not configured**. Levels already carry their own
// element type — `Decomposition::dtype_at` — and the element type is the single
// best predictor of whether deflate will pay, so [`Compression::for_dtype`]
// reads the plan rather than asking the caller. What that derivation is, and the
// evidence for it, is on `Compression::for_dtype`. An explicit override is one
// call away and does not have to fight the default to be heard.
//
// What compression does to the guard above is the part worth stating, because it
// changes a trade rather than only a constant:
//
// * A **fully covered** chunk is still a blind overwrite and still takes no
//   lock. What was a `memcpy` into a buffer is now an *encode* — deflate over
//   the whole chunk — so the fast path got slower in absolute terms.
// * A **partly covered** chunk was already decode-patch-encode. Under
//   compression the decode is a *decompress* and the encode is a *recompress*,
//   both over the whole chunk, and both happen **inside the lock**. The
//   serialised section is now the dominant cost of such a write rather than a
//   rounding error on it.
//
// So compression does not weaken the guard — it makes it matter more, and it
// makes the *alignment* advice matter more still. Before, a straddling write
// paid a decode and a re-encode of bytes it did not change; now it pays a
// decompress and a recompress of them, under a lock. The counter that reports
// this is unchanged and means the same thing: [`Self::serialised_writes`] counts
// writes that took that path, and it is the number a caller who can align their
// block grid to the chunk grid should be watching. `unaligned_reads` likewise —
// a partial *read* now pays a decompress of the whole chunk to keep a corner of
// it.
//
// None of that touches the answer. `tests/zarr_env.rs` asserts the same chains
// through a compressed store, an uncompressed store and `ArrayEnvironment` agree
// voxel for voxel, which is the only claim compression is allowed to make.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock, RwLockWriteGuard};

use ndarray::{Array3, ArrayD, IxDyn};

use zarrs::array::codec::api::BytesToBytesCodecTraits;
use zarrs::array::codec::GzipCodec;
use zarrs::array::data_type;
use zarrs::array::{
    Array as ZarrArray, ArrayBuilder, ArraySubset, DataType, Element, ElementOwned, FillValue,
};
use zarrs::filesystem::FilesystemStore;

use crate::decomposition::Decomposition;
use crate::dtype::Dtype;
use crate::env::{block_shape, BlockBuf, EnvCounters, Environment};
use crate::error::{Error, Result};
use crate::geometry::{chunks_touched, region_within};
use crate::op::{Anchor, Chain, Output};
use crate::region::Region;
use crate::sidecar::{FileSidecars, Sidecars};
use crate::voxels::{SideBuf, VoxelElement, Voxels};

/// The store this environment is written against.
type Store = FilesystemStore;

/// One Zarr array: a level, or one of the arrays an op writes beside its result.
type Stored = ZarrArray<Store>;

// ------------------------------------------------------------- data types --

/// The Zarr v3 data type an element type is stored as.
///
/// **All eleven `Voxels` variants have one**, and the mapping is the identity on
/// names: `Dtype::numpy_name` and the Zarr v3 data-type name are the same string
/// for every one of them, which is not a coincidence — Zarr v3 took its names
/// from NumPy's.
///
/// The one refusal is [`Dtype::F16`], and it is refused for the reason
/// [`Voxels::filled`] refuses it rather than for a reason about Zarr: Zarr *has*
/// `float16`, and a `float16` array could be created here quite happily. What
/// could not happen is a block being read out of it, because Rust has no native
/// 16-bit float and this crate's dependency list is deliberately short. Creating
/// an array nothing can read would be worse than saying so.
///
/// Refusing by name rather than widening to `float32` is the whole point. A
/// silent widening writes a file whose element type is not the one the plan
/// declared, and every consumer downstream — including a reader in another
/// language — would be right to believe the file.
pub fn zarr_data_type(dtype: Dtype) -> Result<DataType> {
    Ok(match dtype {
        Dtype::Bool => data_type::bool(),
        Dtype::U8 => data_type::uint8(),
        Dtype::U16 => data_type::uint16(),
        Dtype::U32 => data_type::uint32(),
        Dtype::U64 => data_type::uint64(),
        Dtype::I8 => data_type::int8(),
        Dtype::I16 => data_type::int16(),
        Dtype::I32 => data_type::int32(),
        Dtype::I64 => data_type::int64(),
        Dtype::F32 => data_type::float32(),
        Dtype::F64 => data_type::float64(),
        Dtype::F16 => {
            return Err(Error::InvalidArgument(
                "half-precision is refused rather than widened: Zarr has a float16 data type, but \
                 this crate has no buffer that can hold one — Rust has no native 16-bit float and \
                 the dependency list is deliberately short — so an array created at float16 would \
                 be one nothing here could read a block out of. Widening it to float32 would \
                 write a file whose declared element type is not the plan's, which every reader \
                 downstream would be right to believe."
                    .to_string(),
            ))
        }
    })
}

/// The value a voxel nobody wrote reads back as.
///
/// The same sentinel [`Voxels::unwritten`] uses, and for the same argument: an
/// unwritten voxel must be loud. Zarr makes this cheaper than memory does — a
/// chunk nobody wrote is not a file at all, and reads back as this.
fn unwritten_fill(dtype: Dtype) -> Result<FillValue> {
    Ok(match dtype {
        Dtype::Bool => FillValue::from(true),
        Dtype::U8 => FillValue::from(u8::MAX),
        Dtype::U16 => FillValue::from(u16::MAX),
        Dtype::U32 => FillValue::from(u32::MAX),
        Dtype::U64 => FillValue::from(u64::MAX),
        Dtype::I8 => FillValue::from(i8::MAX),
        Dtype::I16 => FillValue::from(i16::MAX),
        Dtype::I32 => FillValue::from(i32::MAX),
        Dtype::I64 => FillValue::from(i64::MAX),
        Dtype::F32 => FillValue::from(f32::NAN),
        Dtype::F64 => FillValue::from(f64::NAN),
        Dtype::F16 => return zarr_data_type(Dtype::F16).map(|_| unreachable!()),
    })
}

// ------------------------------------------------------------ compression --

/// How one array's chunks are encoded on the way to the store.
///
/// A **bytes-to-bytes** choice: it sits after the `bytes` codec, so it changes
/// how many bytes a chunk occupies and nothing else. The element type, the
/// element order, the fill value and the chunk grid are all unaffected, which is
/// what makes the byte-identity claim in `tests/zarr_env.rs` a claim about
/// storage rather than about arithmetic.
///
/// Two variants and no more, deliberately. Zarr v3 has a long list of codecs and
/// `zarrs` can build most of them; what this environment exposes is the subset
/// that is a **core** Zarr codec (so a reader in another language is not being
/// asked to implement an extension), **lossless** (so the byte-identity claim
/// survives), and already paid for in this crate's dependency graph. `gzip` is
/// the only one of those, because `zarrs`'s `gzip` feature is `dep:flate2` and
/// `flate2` is already here for `cache::DeflateCodec` — enabling it adds **no
/// package** to the graph. `zstd`, `blosc` and `bz2` each add crates and a C
/// build for a ratio that would have to be argued rather than assumed; the shape
/// of this enum leaves room for them and this version does not spend the
/// dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    /// The `bytes` codec alone: elements in native byte order, verbatim.
    ///
    /// The right answer for data whose low bits are noise — see
    /// [`Compression::for_dtype`].
    None,
    /// The Zarr v3 `gzip` codec at a deflate level of 0..=9, clamped on the way
    /// in so a caller cannot build an array `zarrs` will refuse.
    Gzip(u32),
}

impl Compression {
    /// **The derived default: compress the integers and the `bool`s, leave the
    /// floats alone.**
    ///
    /// | element type | default | why |
    /// |---|---|---|
    /// | `bool` | `Gzip(1)` | one byte per voxel holding one bit of information, in long runs. This is the case compression exists for. |
    /// | `uint8`…`int64` | `Gzip(1)` | images and label volumes: bounded range, spatial correlation, and — for labels — long runs. |
    /// | `float32`, `float64` | `None` | an IEEE mantissa over continuous data is close to incompressible byte-wise, and deflate charges full price to discover that. |
    ///
    /// The evidence, in three parts and in increasing order of how much it
    /// should be trusted:
    ///
    /// 1. **The shape of the data.** `bool` is the only element type here that
    ///    is *guaranteed* to waste seven eighths of every byte, and a mask is
    ///    runs by construction. Nothing about a `float64` intermediate is
    ///    similarly guaranteed.
    /// 2. **This crate's own prior measurement**, recorded when the block cache
    ///    was designed: **19.7x** on `bool` intermediates and **2.09x** on raw
    ///    `uint16`, deflate, on real volumes. `cache::DeflateCodec` was chosen
    ///    on those numbers and defaults to level 1 on the same argument this
    ///    does — past level 1 the extra ratio on run-heavy data is small and the
    ///    CPU cost is not.
    /// 3. **The measurement in this crate's own tests.**
    ///    `compression_pays_for_bool_and_not_for_float` in `tests/zarr_env.rs`
    ///    writes a `synthetic::Scene` volume and its threshold through this
    ///    environment and prints stored bytes and elapsed time per level, both
    ///    ways. It asserts the direction of each default rather than a ratio: a
    ///    number is a fact about one volume, but "the `bool` level shrinks a
    ///    great deal and the `float64` level barely moves" is the claim the
    ///    defaults are actually making.
    ///
    /// The float refusal is the one that surprises people, so: it is not that
    /// floats never compress — a float volume that is mostly one value
    /// compresses fine, and a caller who has one should say
    /// `Compression::Gzip(1)` and get it. It is that deflate over a *byte*
    /// stream cannot see that the interesting bits of an `f64` are in byte 7 and
    /// the noise is in byte 0. The codec that fixes that is a byte shuffle, and
    /// the shuffle codec `zarrs` carries is marked experimental and outside Zarr
    /// v3 core — which is a defensible thing for a caller to opt into and not a
    /// defensible thing for a library to write into files by default.
    ///
    /// `float16` has no case here at all: it is refused before an array exists
    /// (see [`zarr_data_type`]), and this returns [`Compression::None`] for it
    /// only so that this function is total.
    pub fn for_dtype(dtype: Dtype) -> Self {
        match dtype {
            // A byte per bit, in runs.
            Dtype::Bool => Self::Gzip(1),
            // Bounded range and spatial correlation; labels are runs outright.
            Dtype::U8
            | Dtype::U16
            | Dtype::U32
            | Dtype::U64
            | Dtype::I8
            | Dtype::I16
            | Dtype::I32
            | Dtype::I64 => Self::Gzip(1),
            // Mantissa noise: deflate pays full price to find nothing.
            Dtype::F32 | Dtype::F64 | Dtype::F16 => Self::None,
        }
    }

    /// The codecs `zarrs` should put after the `bytes` codec.
    ///
    /// Empty for [`Compression::None`], which is exactly the pipeline this
    /// environment wrote before compression existed — so an array built with
    /// `None` is byte-for-byte the array the previous version built, metadata
    /// included.
    fn bytes_to_bytes(self) -> Result<Vec<Arc<dyn BytesToBytesCodecTraits>>> {
        Ok(match self {
            Self::None => Vec::new(),
            Self::Gzip(level) => {
                let codec = GzipCodec::new(level.min(9)).map_err(Error::backend)?;
                vec![Arc::new(codec)]
            }
        })
    }

    /// What the metadata will call it, for the counters and for a message.
    pub fn name(self) -> String {
        match self {
            Self::None => "bytes".to_string(),
            Self::Gzip(level) => format!("gzip{}", level.min(9)),
        }
    }
}

/// Which [`Compression`] each level gets.
///
/// Three ways to say it, in the order they should be reached for:
///
/// * [`CompressionPolicy::derived`] — the default. Every level gets
///   [`Compression::for_dtype`] of **its own** element type, which the plan
///   already knows. Nothing to configure and nothing to keep in step with the
///   plan when the plan changes.
/// * [`CompressionPolicy::uniform`] — one answer for the whole run. For a
///   caller who is measuring, or who knows something about their data that the
///   element type does not say.
/// * [`CompressionPolicy::with_level`] — an override for one level, on top of
///   either of the above. This is the knob the per-level design exists for: the
///   level a plan reads once and deletes can be left raw while the mask beside
///   it is compressed, without the caller having to restate the other levels.
///
/// Levels are numbered as they are everywhere else here: level 0 is the input,
/// level `p+1` is what phase `p` wrote. An override naming a level the plan does
/// not have is not an error — it simply never applies, because a policy is
/// written before the plan is known and refusing it would make the two orders of
/// construction disagree.
#[derive(Clone, Debug, Default)]
pub struct CompressionPolicy {
    /// `None` means "derive from the element type", which is the default.
    everywhere: Option<Compression>,
    at_level: BTreeMap<usize, Compression>,
}

impl CompressionPolicy {
    /// Every level gets [`Compression::for_dtype`] of its own element type.
    pub fn derived() -> Self {
        Self::default()
    }

    /// Every level gets the same thing, whatever it holds.
    pub fn uniform(compression: Compression) -> Self {
        Self {
            everywhere: Some(compression),
            at_level: BTreeMap::new(),
        }
    }

    /// One level, said explicitly, overriding whatever this policy would
    /// otherwise have chosen for it.
    #[must_use]
    pub fn with_level(mut self, level: usize, compression: Compression) -> Self {
        self.at_level.insert(level, compression);
        self
    }

    /// What level `level`, holding `dtype`, is stored as.
    pub fn at(&self, level: usize, dtype: Dtype) -> Compression {
        match self.at_level.get(&level) {
            Some(&explicit) => explicit,
            None => self.everywhere.unwrap_or_else(|| Compression::for_dtype(dtype)),
        }
    }

    /// What a side output holding `dtype` is stored as.
    ///
    /// A side output is not a level and has no level number, so a per-level
    /// override cannot name one — but a `uniform` policy is a statement about
    /// the run and does apply. Otherwise it is derived from the declared element
    /// type, exactly as a level is.
    pub fn for_side(&self, dtype: Dtype) -> Compression {
        self.everywhere
            .unwrap_or_else(|| Compression::for_dtype(dtype))
    }
}

/// `match` over every element type with one body, binding `$element` to the Rust
/// type that holds it.
///
/// The dispatch happens once per read and once per write rather than once per
/// voxel, which is why a tag is affordable at all. `Dtype::F16` is not an arm:
/// it is refused by [`zarr_data_type`] before an array exists, and by
/// [`Voxels::filled`] before a buffer does.
macro_rules! by_dtype {
    ($dtype:expr, |$element:ident| $body:expr) => {{
        let dtype = $dtype;
        match dtype {
            Dtype::Bool => {
                type $element = bool;
                $body
            }
            Dtype::U8 => {
                type $element = u8;
                $body
            }
            Dtype::U16 => {
                type $element = u16;
                $body
            }
            Dtype::U32 => {
                type $element = u32;
                $body
            }
            Dtype::U64 => {
                type $element = u64;
                $body
            }
            Dtype::I8 => {
                type $element = i8;
                $body
            }
            Dtype::I16 => {
                type $element = i16;
                $body
            }
            Dtype::I32 => {
                type $element = i32;
                $body
            }
            Dtype::I64 => {
                type $element = i64;
                $body
            }
            Dtype::F32 => {
                type $element = f32;
                $body
            }
            Dtype::F64 => {
                type $element = f64;
                $body
            }
            Dtype::F16 => return zarr_data_type(Dtype::F16).map(|_| unreachable!()),
        }
    }};
}

// ------------------------------------------------------------ chunk locks --

/// Which chunks a region covers only in part.
///
/// Those are the chunks a write will read-modify-write, and therefore the only
/// ones that need serialising. A chunk covered *entirely* is overwritten
/// wholesale, so two disjoint writes cannot both be inside it.
///
/// The comparison is against the **full** chunk extent, not the extent clipped
/// to the array — which is what `zarrs` compares against too. So the last chunk
/// on an axis is partial unless the array's extent is a multiple of the chunk's,
/// and that is the correct answer: `zarrs` will decode and re-encode it.
///
/// Rank-generic, because a side output has its own rank.
fn partly_covered_chunks(region: &Region, chunk: &[usize]) -> Vec<Vec<u64>> {
    let rank = region.ndim();
    if rank == 0 || region.voxels() == 0 {
        return Vec::new();
    }
    // Per axis: the chunk indices the region touches, and which of them it
    // covers from edge to edge.
    let mut spans: Vec<(u64, u64)> = Vec::with_capacity(rank);
    let mut fully: Vec<(u64, u64)> = Vec::with_capacity(rank);
    for axis in 0..rank {
        let edge = chunk.get(axis).copied().unwrap_or(1).max(1) as u64;
        let lo = region.start[axis] as u64;
        let hi = lo + region.shape[axis] as u64;
        spans.push((lo / edge, (hi - 1) / edge + 1));
        // The first chunk is fully covered when the region starts on its edge;
        // the last when the region ends on one.
        let first_full = lo.div_ceil(edge);
        let last_full = hi / edge;
        fully.push((first_full, last_full.max(first_full)));
    }

    let mut found = Vec::new();
    let mut index: Vec<u64> = spans.iter().map(|&(lo, _)| lo).collect();
    loop {
        let whole = (0..rank).all(|axis| {
            let (lo, hi) = fully[axis];
            index[axis] >= lo && index[axis] < hi
        });
        if !whole {
            found.push(index.clone());
        }
        // Odometer over the touched chunk box.
        let mut axis = rank;
        loop {
            if axis == 0 {
                return found;
            }
            axis -= 1;
            index[axis] += 1;
            if index[axis] < spans[axis].1 {
                break;
            }
            index[axis] = spans[axis].0;
        }
    }
}

/// A fixed set of locks, one stripe per hash of `(array, chunk)`.
///
/// Fixed rather than a map keyed by chunk: nothing is allocated on the write
/// path, there is no map lock to contend on, and there is nothing to evict. The
/// price is that two unrelated chunks may share a stripe and serialise when they
/// need not have, which costs a write's worth of latency and no correctness.
#[derive(Debug)]
struct ChunkLocks {
    stripes: Vec<RwLock<()>>,
}

/// Enough that collisions are rare at the thread counts this crate schedules
/// (measured runs use up to 40), and small enough to be a rounding error at
/// construction.
const STRIPES: usize = 1024;

impl ChunkLocks {
    fn new() -> Self {
        Self {
            stripes: (0..STRIPES).map(|_| RwLock::new(())).collect(),
        }
    }

    /// FNV-1a over the array id and the chunk index. Any spreading hash does;
    /// what matters is that it is a pure function of the key, so two threads
    /// naming one chunk always name one stripe.
    fn stripe(array: u64, chunk: &[u64]) -> usize {
        let mut hash = 0xcbf29ce484222325u64;
        for value in std::iter::once(array).chain(chunk.iter().copied()) {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        (hash % STRIPES as u64) as usize
    }

    /// Hold every stripe a partial write of `region` will read-modify-write.
    ///
    /// Sorted and deduplicated before acquisition, so the acquisition order is a
    /// total order over stripes and two writes with overlapping chunk sets
    /// cannot deadlock against one another.
    fn hold(&self, array: u64, region: &Region, chunk: &[usize]) -> Vec<RwLockWriteGuard<'_, ()>> {
        let mut wanted: Vec<usize> = partly_covered_chunks(region, chunk)
            .iter()
            .map(|index| Self::stripe(array, index))
            .collect();
        wanted.sort_unstable();
        wanted.dedup();
        wanted
            .into_iter()
            .map(|stripe| {
                self.stripes[stripe]
                    .write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
            })
            .collect()
    }
}

// --------------------------------------------------------------- the array --

/// One array in the store, with everything needed to read and write it.
struct StoredArray {
    array: Stored,
    /// A stable id for the lock stripes. Distinct per array, and stable for the
    /// life of the environment.
    id: u64,
    dtype: Dtype,
    shape: Vec<usize>,
    chunk: Vec<usize>,
    /// What this array's chunks are encoded with. Kept so that
    /// [`ZarrEnvironment::compression_at`] can answer without re-reading the
    /// metadata, and so a message can name it.
    compression: Compression,
}

impl StoredArray {
    fn create(
        store: &Arc<Store>,
        path: &str,
        id: u64,
        dtype: Dtype,
        shape: &[usize],
        chunk: &[usize],
        compression: Compression,
    ) -> Result<Self> {
        let zarr_shape: Vec<u64> = shape.iter().map(|&value| value as u64).collect();
        let zarr_chunk: Vec<u64> = chunk
            .iter()
            .map(|&value| value.max(1) as u64)
            .collect::<Vec<_>>();
        let mut builder = ArrayBuilder::new(
            zarr_shape,
            zarr_chunk,
            zarr_data_type(dtype)?,
            unwritten_fill(dtype)?,
        );
        // Left unset for `Compression::None`, rather than set to an empty list:
        // an array built with `None` must be the array this environment built
        // before compression existed, metadata document included.
        let codecs = compression.bytes_to_bytes()?;
        if !codecs.is_empty() {
            builder.bytes_to_bytes_codecs(codecs);
        }
        let array = builder.build(store.clone(), path).map_err(Error::backend)?;
        array.store_metadata().map_err(Error::backend)?;
        Ok(Self {
            array,
            id,
            dtype,
            shape: shape.to_vec(),
            chunk: chunk.iter().map(|&value| value.max(1)).collect(),
            compression,
        })
    }

    fn subset(&self, region: &Region) -> Result<ArraySubset> {
        ArraySubset::new_with_start_shape(
            region.start.iter().map(|&value| value as u64).collect(),
            region.shape.iter().map(|&value| value as u64).collect(),
        )
        .map_err(Error::backend)
    }

    /// Read `region` as `T`.
    ///
    /// One call, both alignments. `zarrs`'s own `retrieve_array_subset` already
    /// dispatches a region that lands exactly on one chunk to a whole-chunk
    /// decode, a region inside one chunk to a partial decode, and a multi-chunk
    /// region to a parallel per-chunk assembly — so a chunk-aligned read is the
    /// fast path without a second implementation of it here, and a second
    /// implementation is exactly the shape of the off-by-one this crate exists
    /// to remove. What is added here is the *visible* half:
    /// [`ZarrEnvironment::unaligned_reads`] counts the reads that took the
    /// partial-decode path, so a caller who can align their blocks can see
    /// whether they did.
    fn read_as<T: ElementOwned + VoxelElement>(&self, region: &Region) -> Result<Voxels> {
        let subset = self.subset(region)?;
        let data: Vec<T> = self
            .array
            .retrieve_array_subset(&subset)
            .map_err(Error::backend)?;
        let shape = block_shape(region)?;
        let array = Array3::from_shape_vec((shape[0], shape[1], shape[2]), data).map_err(|_| {
            Error::ShapeMismatch {
                expected: region.shape.clone(),
                got: vec![],
            }
        })?;
        Ok(T::wrap(array))
    }

    /// Write `data` — already in C order, `region`'s extent — into `region`.
    ///
    /// `locks` is `None` only in the test that watches the race happen. There is
    /// no way to reach that from outside this module, deliberately: a guard that
    /// can be switched off in production is a guard that will be.
    ///
    /// The guards are held across `store_array_subset` and therefore across
    /// whatever `zarrs` does inside it — which, on a compressed array, includes
    /// a **decompress and a recompress of every partly covered chunk**. That is
    /// the critical section growing, not the guard weakening: the same chunks
    /// are serialised, for longer, and the write that avoids the section
    /// entirely is still the one that covers its chunks from edge to edge.
    fn write_as<T: Element>(
        &self,
        region: &Region,
        data: &[T],
        locks: Option<&ChunkLocks>,
    ) -> Result<bool> {
        let subset = self.subset(region)?;
        let guards = locks.map(|locks| locks.hold(self.id, region, &self.chunk));
        let serialised = guards.as_ref().is_some_and(|held| !held.is_empty());
        let outcome = self
            .array
            .store_array_subset(&subset, data)
            .map_err(Error::backend);
        drop(guards);
        outcome.map(|()| serialised)
    }
}

// ---------------------------------------------------------- the environment --

/// Levels as Zarr v3 arrays under one root.
///
/// The storage counterpart of `ArrayEnvironment`, and held to the same standard:
/// `tests/zarr_env.rs` runs the same chains over the same data through both and
/// asserts the outputs agree voxel for voxel.
pub struct ZarrEnvironment {
    root: PathBuf,
    store: Arc<Store>,
    volume: [usize; 3],
    chunk: [usize; 3],
    /// Level `l` at index `l`. Level 0 exists from construction; `prepare` adds
    /// the rest, because only the plan knows their shapes and element types.
    levels: RwLock<Vec<Arc<StoredArray>>>,
    /// The arrays ops write beside their primary result, by declared name.
    ///
    /// A `BTreeMap` rather than a `HashMap` so that iteration is declaration
    /// order, matching `ArrayEnvironment`.
    side: RwLock<BTreeMap<String, Arc<StoredArray>>>,
    /// Which level is stored with which codec. Consulted by `prepare`, which is
    /// where every level but level 0 is created.
    compression: CompressionPolicy,
    locks: ChunkLocks,
    /// Writes that had to serialise on at least one chunk, and reads that could
    /// not be served by whole-chunk decodes. Diagnostics, not correctness: both
    /// are zero for a decomposition whose blocks land on the chunk grid.
    serialised_writes: AtomicU64,
    unaligned_reads: AtomicU64,
    counters: EnvCounters,
    /// On disk, to match the levels. An environment whose volumes are shared
    /// between processes must not offer fragments that are not.
    sidecars: Sidecars,
}

impl ZarrEnvironment {
    /// Create the store at `root` and write `input` as level 0.
    ///
    /// `root` is created if it does not exist. The remaining levels are not
    /// created here — [`Environment::prepare`] does that, from the plan, because
    /// a phase owns its volume and its element type and this constructor knows
    /// neither.
    ///
    /// Compression is [`CompressionPolicy::derived`]: every level is stored as
    /// [`Compression::for_dtype`] of its own element type. See
    /// [`Self::create_with_compression`] to say otherwise.
    pub fn create(root: impl Into<PathBuf>, input: &Voxels, chunk: [usize; 3]) -> Result<Self> {
        Self::create_with_compression(root, input, chunk, CompressionPolicy::derived())
    }

    /// [`Self::create`], with the codec of every level said explicitly.
    ///
    /// The policy is kept for the life of the environment, because
    /// [`Environment::prepare`] creates levels 1..=n later and has to ask it the
    /// same question this constructor asks it about level 0.
    pub fn create_with_compression(
        root: impl Into<PathBuf>,
        input: &Voxels,
        chunk: [usize; 3],
        compression: CompressionPolicy,
    ) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(Error::backend)?;
        let store = Arc::new(FilesystemStore::new(&root).map_err(Error::backend)?);
        let volume = input.shape();
        let level0 = StoredArray::create(
            &store,
            &level_path(0),
            0,
            input.dtype(),
            &volume,
            &chunk,
            compression.at(0, input.dtype()),
        )?;
        let locks = ChunkLocks::new();
        let whole = Region::whole(&volume);
        by_dtype!(input.dtype(), |Element| {
            let view = input.view::<Element>()?;
            // `Cow`, so a block that is already contiguous — which every
            // `Voxels` this crate builds is — costs nothing here.
            let standard = view.as_standard_layout();
            let data = standard.as_slice().expect("standard layout is contiguous");
            level0.write_as::<Element>(&whole, data, Some(&locks))?;
        });
        let sidecars = Sidecars::new(FileSidecars::at(root.join("sidecars"))?);
        Ok(Self {
            root,
            store,
            volume,
            chunk,
            levels: RwLock::new(vec![Arc::new(level0)]),
            side: RwLock::new(BTreeMap::new()),
            compression,
            locks,
            serialised_writes: AtomicU64::new(0),
            unaligned_reads: AtomicU64::new(0),
            counters: EnvCounters::default(),
            sidecars,
        })
    }

    /// Where the arrays live.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// How many writes had to serialise on at least one chunk.
    ///
    /// Zero says every write landed on the chunk grid and took the lock-free
    /// path. A large number says the block grid and the chunk grid disagree,
    /// which is a *performance* fact about the plan — the answer is the same
    /// either way, which is the whole design of this layer.
    pub fn serialised_writes(&self) -> u64 {
        self.serialised_writes.load(Ordering::SeqCst)
    }

    /// How many reads asked for something other than whole chunks.
    ///
    /// Such a read is correct — `zarrs` decodes the chunk and crops — but it
    /// pays a decode for data it discards, which is the over-read the block
    /// planner's alignment advice exists to avoid.
    pub fn unaligned_reads(&self) -> u64 {
        self.unaligned_reads.load(Ordering::SeqCst)
    }

    fn levels_guard(&self) -> std::sync::RwLockReadGuard<'_, Vec<Arc<StoredArray>>> {
        self.levels
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// One level, or an error saying how many there are.
    ///
    /// The guard is taken **once**. Reading an `RwLock` twice in one expression
    /// — which is what an `ok_or_else` closure that reaches for the length
    /// again would do — is a recursive read, and `std` is explicit that a
    /// recursive read may deadlock against a waiting writer.
    fn level_array(&self, level: usize) -> Result<Arc<StoredArray>> {
        let levels = self.levels_guard();
        match levels.get(level) {
            Some(array) => Ok(array.clone()),
            None => Err(Error::InvalidArgument(format!(
                "this environment holds {} level(s) and level {level} was asked for; \
                 `Environment::prepare` creates the levels a plan writes, and is called once \
                 before any task",
                levels.len()
            ))),
        }
    }

    /// The shape of one level, which is not `volume()` once a phase changes it.
    pub fn level_shape(&self, level: usize) -> Result<[usize; 3]> {
        let array = self.level_array(level)?;
        block_shape(&Region::whole(&array.shape))
    }

    /// The element type of one level.
    pub fn level_dtype(&self, level: usize) -> Result<Dtype> {
        Ok(self.level_array(level)?.dtype)
    }

    /// What one level's chunks are actually stored as.
    ///
    /// Read off the array rather than recomputed from the policy, so it answers
    /// what was built and not what would be built now.
    pub fn compression_at(&self, level: usize) -> Result<Compression> {
        Ok(self.level_array(level)?.compression)
    }

    /// How many bytes one level occupies on the disk.
    ///
    /// **The measurement compression exists to move**, and an accessor rather
    /// than a number in a document because it is the only honest way to compare
    /// two policies: run both and weigh the directories. Chunk files only — the
    /// `zarr.json` metadata document is excluded, because it is a fixed few
    /// hundred bytes that would swamp a small array and vanish in a large one,
    /// and either way it is not what a codec changes.
    ///
    /// A chunk nobody wrote is not a file, so this counts what was written and
    /// not what was declared. That is the same convention `EnvCounters`'
    /// `write_bytes` uses for the *uncompressed* side, which makes the pair of
    /// them a ratio without any further bookkeeping.
    pub fn stored_bytes(&self, level: usize) -> Result<u64> {
        // Named to check the level exists, so a typo is an error and not a zero.
        let _ = self.level_array(level)?;
        Ok(directory_bytes(&self.root.join(format!("level{level}"))))
    }

    /// How many bytes one side output occupies on the disk, or `None` if nothing
    /// declared that name. Same accounting as [`Self::stored_bytes`].
    pub fn stored_side_bytes(&self, name: &str) -> Option<u64> {
        let known = self
            .side
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(name);
        known.then(|| directory_bytes(&self.root.join("side").join(name)))
    }

    /// One whole level, read back.
    ///
    /// For inspection and for the byte-identity comparison; it deliberately does
    /// **not** touch the counters, which are the run's, not the assertion's.
    pub fn level(&self, level: usize) -> Result<Voxels> {
        let array = self.level_array(level)?;
        let whole = Region::whole(&array.shape);
        by_dtype!(array.dtype, |Element| array.read_as::<Element>(&whole))
    }

    /// The last level: the workflow output.
    pub fn output(&self) -> Result<Voxels> {
        let last = self.levels_guard().len().saturating_sub(1);
        self.level(last)
    }

    /// One side output, by declared name, or `None` if nothing declared it.
    ///
    /// Widened to `f64` on the way out, matching `SideBuf`'s own element type
    /// and `ArrayEnvironment::side_output`'s signature. What is *stored* is the
    /// declared type — see [`Self::put_side`].
    pub fn side_output(&self, name: &str) -> Result<Option<ArrayD<f64>>> {
        let Some(array) = self
            .side
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(name)
            .cloned()
        else {
            return Ok(None);
        };
        let whole = Region::whole(&array.shape);
        let subset = array.subset(&whole)?;
        let widened = by_dtype!(array.dtype, |Element| {
            let data: Vec<Element> = array
                .array
                .retrieve_array_subset(&subset)
                .map_err(Error::backend)?;
            data.into_iter()
                .map(VoxelElement::into_f64)
                .collect::<Vec<f64>>()
        });
        ArrayD::from_shape_vec(IxDyn(&array.shape), widened)
            .map(Some)
            .map_err(Error::backend)
    }

    /// Every declared side output's name, in declaration order.
    pub fn side_output_names(&self) -> Vec<String> {
        self.side
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .cloned()
            .collect()
    }

    /// Whether `region` covers every chunk it touches from edge to edge.
    fn aligned(&self, region: &Region, chunk: &[usize]) -> bool {
        partly_covered_chunks(region, chunk).is_empty()
    }
}

/// Every chunk file under `at`, added up.
///
/// Recursive, because the default chunk key encoding nests one directory per
/// axis. `zarr.json` is skipped wherever it appears — see [`ZarrEnvironment::stored_bytes`].
/// A directory that does not exist is zero rather than an error: an array whose
/// chunks were all left unwritten is a legitimate state, and it stores nothing.
fn directory_bytes(at: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(at) else {
        return 0;
    };
    let mut total = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            total += directory_bytes(&path);
        } else if path.file_name().is_some_and(|name| name != "zarr.json") {
            total += entry.metadata().map(|meta| meta.len()).unwrap_or(0);
        }
    }
    total
}

/// `root/level0`, `root/level1`, …
///
/// A leading `/` because a Zarr node path is absolute within its store, and the
/// store's own root is the directory.
fn level_path(level: usize) -> String {
    format!("/level{level}")
}

/// `root/side/<name>`. The name is the op's declaration and is used verbatim, so
/// that what a reader finds on disk is what the plan called it.
fn side_path(name: &str) -> String {
    format!("/side/{name}")
}

/// A chunk shape for an array of arbitrary rank.
///
/// A side output's rank and extent are its own — one row per object, one score
/// per class per position — so the level chunk shape does not apply and there is
/// nothing to inherit. 256 per axis is a chunk of a few megabytes at eight bytes
/// an element, which is the size the rest of this crate is tuned around, capped
/// at the extent so a small array is one chunk.
fn side_chunk(shape: &[usize]) -> Vec<usize> {
    shape.iter().map(|&dim| dim.clamp(1, 256)).collect()
}

impl Environment for ZarrEnvironment {
    fn volume(&self) -> [usize; 3] {
        self.volume
    }

    /// Create levels 1..=n at the plan's per-phase volume and element type, and
    /// check that level 0 is what the plan says it read.
    ///
    /// Idempotent in the only sense that matters: called twice with the same
    /// plan it rebuilds the same arrays over the same metadata. It is called
    /// once, before any task.
    fn prepare(&self, decomposition: &Decomposition) -> Result<()> {
        let held = self.level_shape(0)?;
        let wanted = decomposition.volume_at(0);
        if held != wanted {
            return Err(Error::InvalidArgument(format!(
                "zarr environment holds level 0 as {held:?} and the decomposition reads it as \
                 {wanted:?}"
            )));
        }
        let held = self.level_dtype(0)?;
        let wanted = decomposition.dtype_at(0);
        if held != wanted {
            return Err(Error::InvalidArgument(format!(
                "zarr environment holds level 0 as {} and the decomposition reads it as {}",
                held.numpy_name(),
                wanted.numpy_name()
            )));
        }
        let mut levels = Vec::with_capacity(decomposition.n_levels());
        levels.push(self.level_array(0)?);
        for level in 1..decomposition.n_levels() {
            // The plan's own element type for this level, asked twice: once for
            // what to store and once for how. Deriving the codec here rather
            // than at construction is the whole reason the policy is per level
            // — level 0 is the only one whose type this environment knew before
            // it saw a plan.
            let dtype = decomposition.dtype_at(level);
            levels.push(Arc::new(StoredArray::create(
                &self.store,
                &level_path(level),
                level as u64,
                dtype,
                &decomposition.volume_at(level),
                &self.chunk,
                self.compression.at(level, dtype),
            )?));
        }
        *self
            .levels
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = levels;
        Ok(())
    }

    fn read(&self, level: usize, region: &Region) -> Result<BlockBuf> {
        let array = self.level_array(level)?;
        region_within(region, &array.shape, "block-op read")?;
        if !self.aligned(region, &array.chunk) {
            self.unaligned_reads.fetch_add(1, Ordering::SeqCst);
        }
        let block = by_dtype!(array.dtype, |Element| array.read_as::<Element>(region))?;
        self.counters.reads.fetch_add(1, Ordering::SeqCst);
        self.counters
            .read_voxels
            .fetch_add(block.len() as u64, Ordering::SeqCst);
        self.counters
            .read_bytes
            .fetch_add(block.bytes(), Ordering::SeqCst);
        self.counters
            .chunks_read
            .fetch_add(chunks_touched(region, &self.chunk), Ordering::SeqCst);
        self.counters.add_resident(block.bytes());
        Ok(BlockBuf::Array(block))
    }

    /// Identical to `ArrayEnvironment::apply`, and identical on purpose: where
    /// the bytes live has nothing to do with what an op computes, and the moment
    /// this diverges the byte-identity claim stops being about storage.
    fn apply(&self, slot: &Chain, input: &BlockBuf, at: &Anchor) -> Result<BlockBuf> {
        let array = input.as_array()?;
        let mut out = Voxels::zeros(
            slot.produces(array.dtype())?,
            slot.output_shape(array.shape())?,
        )?;
        slot.apply(array, &mut out, at)?;
        self.counters.ops_applied.fetch_add(1, Ordering::SeqCst);
        self.counters.estimated_work.fetch_add(
            (array.len() as f64 * slot.cost_per_voxel()) as u64,
            Ordering::SeqCst,
        );
        self.counters.add_resident(out.bytes());
        Ok(BlockBuf::Array(out))
    }

    fn write(&self, level: usize, within: &Region, valid: &Region, buf: &BlockBuf) -> Result<()> {
        let block = buf.as_array()?;
        let array = self.level_array(level)?;
        region_within(valid, &array.shape, "block-op write")?;
        if valid.voxels() == 0 {
            return Ok(());
        }
        if block.dtype() != array.dtype {
            return Err(Error::InvalidArgument(format!(
                "level write: this block holds {} and level {level} holds {}",
                block.dtype().numpy_name(),
                array.dtype.numpy_name()
            )));
        }
        let source = block.slice_region(within)?;
        if source.shape() != [valid.shape[0], valid.shape[1], valid.shape[2]] {
            return Err(Error::ShapeMismatch {
                expected: valid.shape.clone(),
                got: source.shape().to_vec(),
            });
        }
        let serialised = by_dtype!(array.dtype, |Element| {
            let view = source.view::<Element>()?;
            let standard = view.as_standard_layout();
            let data = standard.as_slice().expect("standard layout is contiguous");
            array.write_as::<Element>(valid, data, Some(&self.locks))
        })?;
        if serialised {
            self.serialised_writes.fetch_add(1, Ordering::SeqCst);
        }
        self.counters.writes.fetch_add(1, Ordering::SeqCst);
        self.counters
            .write_voxels
            .fetch_add(valid.voxels() as u64, Ordering::SeqCst);
        self.counters
            .write_bytes
            .fetch_add(source.bytes(), Ordering::SeqCst);
        Ok(())
    }

    /// Create the array, filled with the unwritten sentinel, and refuse a second
    /// declaration that disagrees with the first.
    ///
    /// Two ops writing one name is legitimate — a phase and a later phase may
    /// both contribute — but only if they agree about what the array *is*.
    fn declare_side_output(&self, output: &Output) -> Result<()> {
        let mut outer = self
            .side
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = outer.get(&output.name) {
            if existing.shape != output.shape {
                return Err(Error::InvalidArgument(format!(
                    "side output {:?} was declared as {:?} and is now declared as {:?}",
                    output.name, existing.shape, output.shape
                )));
            }
            if existing.dtype != output.dtype {
                return Err(Error::InvalidArgument(format!(
                    "side output {:?} was declared as {} and is now declared as {}",
                    output.name,
                    existing.dtype.numpy_name(),
                    output.dtype.numpy_name()
                )));
            }
            return Ok(());
        }
        // Ids above the level range, so a side array and a level can never
        // collide on the lock stripes for the wrong reason.
        let id = 1u64 << 32 | outer.len() as u64;
        let array = StoredArray::create(
            &self.store,
            &side_path(&output.name),
            id,
            output.dtype,
            &output.shape,
            &side_chunk(&output.shape),
            self.compression.for_side(output.dtype),
        )?;
        outer.insert(output.name.clone(), Arc::new(array));
        Ok(())
    }

    /// Store one block's slice of a side output, **at the declared element
    /// type**.
    ///
    /// `SideBuf` is `f64` — it is the one buffer in the crate whose rank is
    /// unknown, and `f64` is the crate's numeric lingua franca — but the
    /// `Output` that declared the array says what the array *is*, and that is
    /// also what `write_side` charges for. Storing it at anything else would
    /// make the accounting and the file disagree. The narrowing is Rust's
    /// saturating cast, the same one [`Voxels::filled`] documents.
    fn put_side(
        &self,
        output: &Output,
        _phase: usize,
        region: &Region,
        buf: &SideBuf,
    ) -> Result<()> {
        let Some(values) = buf.as_array() else {
            return Err(Error::InvalidArgument(
                "side buffer holds no data: this is a simulated run".to_string(),
            ));
        };
        let array = self
            .side
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&output.name)
            .cloned()
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "side output {:?} was written before it was declared; the executor declares \
                     every output of every phase before it runs a task",
                    output.name
                ))
            })?;
        region.check_within(&array.shape, "side-output write")?;
        if region.voxels() == 0 {
            return Ok(());
        }
        if values.shape() != region.shape.as_slice() {
            return Err(Error::ShapeMismatch {
                expected: region.shape.clone(),
                got: values.shape().to_vec(),
            });
        }
        let standard = values.as_standard_layout();
        let source = standard
            .as_slice()
            .expect("as_standard_layout is contiguous");
        by_dtype!(array.dtype, |Element| {
            let data: Vec<Element> = source
                .iter()
                .map(|&value| <Element as VoxelElement>::from_f64(value))
                .collect();
            array.write_as::<Element>(region, &data, Some(&self.locks))
        })?;
        Ok(())
    }

    fn uniform(&self, buf: &BlockBuf) -> Option<f64> {
        match buf {
            BlockBuf::Array(array) => array.uniform(),
            BlockBuf::Accounted { uniform, .. } => *uniform,
        }
    }

    fn constant(&self, dtype: Dtype, region: &Region, value: f64) -> Result<BlockBuf> {
        let shape = block_shape(region)?;
        let array = Voxels::filled(dtype, shape, value)?;
        self.counters.add_resident(array.bytes());
        Ok(BlockBuf::Array(array))
    }

    fn release(&self, buf: &BlockBuf) {
        self.counters.drop_resident(buf.bytes());
    }

    /// Everything written to `level` is durable.
    ///
    /// A `FilesystemStore` write is a completed `write` to a file by the time
    /// `store_array_subset` returns, so there is nothing here to flush. It is
    /// still an override point rather than an omission: a store with a write-back
    /// cache would need one, and a caller must be able to write the barrier
    /// without knowing which store it has.
    fn finish(&self, _level: usize) -> Result<()> {
        Ok(())
    }

    fn counters(&self) -> &EnvCounters {
        &self.counters
    }

    fn chunk_shape(&self) -> [usize; 3] {
        self.chunk
    }

    fn sidecars(&self) -> Option<&Sidecars> {
        Some(&self.sidecars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A directory nobody else is using, removed by the caller.
    fn scratch(name: &str) -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = NEXT.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "blockflow-zarr-{}-{name}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    const EVERY_TYPE: [Dtype; 11] = [
        Dtype::Bool,
        Dtype::U8,
        Dtype::U16,
        Dtype::U32,
        Dtype::U64,
        Dtype::I8,
        Dtype::I16,
        Dtype::I32,
        Dtype::I64,
        Dtype::F32,
        Dtype::F64,
    ];

    /// A block whose values are a function of position, so a misplaced voxel is
    /// visible rather than plausible. Kept inside every type's range.
    fn ramp(dtype: Dtype, shape: [usize; 3]) -> Voxels {
        fill_ramp(dtype, shape).expect("every tested type has a buffer")
    }

    fn fill_ramp(dtype: Dtype, shape: [usize; 3]) -> Result<Voxels> {
        let mut block = Voxels::zeros(dtype, shape)?;
        by_dtype!(dtype, |Element| {
            let mut view = block.view_mut::<Element>()?;
            for (flat, value) in view.iter_mut().enumerate() {
                *value = <Element as VoxelElement>::from_f64((flat % 100) as f64);
            }
        });
        Ok(block)
    }

    // ------------------------------------------------------- data types --

    #[test]
    fn every_element_type_a_block_can_hold_has_a_zarr_data_type() {
        for dtype in EVERY_TYPE {
            assert!(zarr_data_type(dtype).is_ok(), "{dtype:?}");
            assert!(unwritten_fill(dtype).is_ok(), "{dtype:?}");
        }
    }

    #[test]
    fn half_precision_is_refused_by_name_rather_than_widened() {
        let err = zarr_data_type(Dtype::F16).unwrap_err().to_string();
        assert!(err.contains("half-precision"), "got: {err}");
        assert!(
            err.contains("float16") && err.contains("float32"),
            "got: {err}"
        );
    }

    // ------------------------------------------------------- compression --

    /// The derived default, spelled out: the small-alphabet types compress, the
    /// floats do not.
    ///
    /// A table test rather than a comment, because the defaults are the part of
    /// this that a reader has to take on trust — this at least makes the claim
    /// checkable and makes a change to it visible in a diff.
    #[test]
    fn the_derived_default_compresses_the_integers_and_bool_and_leaves_floats_raw() {
        for dtype in [
            Dtype::Bool,
            Dtype::U8,
            Dtype::U16,
            Dtype::U32,
            Dtype::U64,
            Dtype::I8,
            Dtype::I16,
            Dtype::I32,
            Dtype::I64,
        ] {
            assert_eq!(
                Compression::for_dtype(dtype),
                Compression::Gzip(1),
                "{dtype:?}"
            );
        }
        for dtype in [Dtype::F16, Dtype::F32, Dtype::F64] {
            assert_eq!(
                Compression::for_dtype(dtype),
                Compression::None,
                "{dtype:?}"
            );
        }
    }

    /// The three ways to say it, and which one wins where.
    #[test]
    fn a_per_level_override_beats_a_uniform_policy_which_beats_the_derivation() {
        let derived = CompressionPolicy::derived();
        assert_eq!(derived.at(0, Dtype::F64), Compression::None);
        assert_eq!(derived.at(1, Dtype::Bool), Compression::Gzip(1));

        // A `uniform` policy speaks for every level, including the ones the
        // derivation would have left alone.
        let everywhere = CompressionPolicy::uniform(Compression::Gzip(6));
        assert_eq!(everywhere.at(0, Dtype::F64), Compression::Gzip(6));
        assert_eq!(everywhere.at(7, Dtype::Bool), Compression::Gzip(6));

        // And a level named explicitly wins over either.
        let mixed = CompressionPolicy::derived()
            .with_level(0, Compression::Gzip(9))
            .with_level(2, Compression::None);
        assert_eq!(mixed.at(0, Dtype::F64), Compression::Gzip(9));
        assert_eq!(mixed.at(1, Dtype::Bool), Compression::Gzip(1));
        assert_eq!(mixed.at(2, Dtype::Bool), Compression::None);
        // A level the plan does not have is not an error; it simply never fires.
        assert_eq!(
            CompressionPolicy::derived()
                .with_level(99, Compression::Gzip(9))
                .at(3, Dtype::F32),
            Compression::None
        );

        // Side outputs have no level number, so an override cannot name one, but
        // a statement about the whole run does apply to them.
        assert_eq!(mixed.for_side(Dtype::F64), Compression::None);
        assert_eq!(everywhere.for_side(Dtype::F64), Compression::Gzip(6));
    }

    /// A level out of `zarrs`'s range is clamped rather than refused, and the
    /// array is still built.
    ///
    /// Clamped and not refused because the number is a dial, not an identity: a
    /// caller who writes 12 meant "as hard as it goes", and failing the whole
    /// run over it would be pedantry. What is *not* clamped anywhere is an
    /// element type — see [`zarr_data_type`] — because that one changes what the
    /// file means.
    #[test]
    fn an_out_of_range_compression_level_is_clamped_rather_than_refused() {
        assert_eq!(Compression::Gzip(12).name(), "gzip9");
        assert!(Compression::Gzip(12).bytes_to_bytes().is_ok());
        let root = scratch("clamped");
        let input = ramp(Dtype::U8, [4, 4, 4]);
        let env = ZarrEnvironment::create_with_compression(
            &root,
            &input,
            [2, 2, 2],
            CompressionPolicy::uniform(Compression::Gzip(12)),
        )
        .unwrap();
        assert_eq!(env.level(0).unwrap(), input);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The metadata says `gzip` when it should and says nothing when it should
    /// not.
    ///
    /// This is the half of the claim `stored_bytes` cannot make: a reader in
    /// another language opens `zarr.json`, and what it finds there has to be the
    /// truth about the bytes beside it.
    #[test]
    fn the_stored_metadata_names_the_codec_and_omits_it_when_there_is_none() {
        for (compression, expect_gzip) in [
            (Compression::None, false),
            (Compression::Gzip(4), true),
        ] {
            let root = scratch("metadata");
            let input = ramp(Dtype::U16, [4, 4, 4]);
            ZarrEnvironment::create_with_compression(
                &root,
                &input,
                [2, 2, 2],
                CompressionPolicy::uniform(compression),
            )
            .unwrap();
            let json = std::fs::read_to_string(root.join("level0").join("zarr.json")).unwrap();
            assert_eq!(
                json.contains("gzip"),
                expect_gzip,
                "{} metadata: {json}",
                compression.name()
            );
            let _ = std::fs::remove_dir_all(&root);
        }
    }

    /// Compression is a fact about the disk and not about the counters.
    ///
    /// `EnvCounters::write_bytes` is what the plan moved; `stored_bytes` is what
    /// landed. Keeping them separate is what lets a caller divide one by the
    /// other, and it is also why compression cannot quietly change the budget or
    /// the cost model — those read the counters.
    #[test]
    fn compression_moves_stored_bytes_and_leaves_the_counters_alone() {
        let shape = [16, 16, 16];
        let mut sizes = Vec::new();
        let mut counted = Vec::new();
        for compression in [Compression::None, Compression::Gzip(1)] {
            let root = scratch("stored-bytes");
            // Every voxel the same: the most compressible thing there is, so the
            // direction of the comparison cannot be in doubt.
            let input = Voxels::filled(Dtype::U8, shape, 3.0).unwrap();
            let env = ZarrEnvironment::create_with_compression(
                &root,
                &input,
                [8, 8, 8],
                CompressionPolicy::uniform(compression),
            )
            .unwrap();
            sizes.push(env.stored_bytes(0).unwrap());
            counted.push(env.counters().byte_snapshot());
            assert_eq!(env.level(0).unwrap(), input, "{}", compression.name());
            let _ = std::fs::remove_dir_all(&root);
        }
        assert_eq!(sizes[0], 16 * 16 * 16, "raw uint8 is one byte a voxel");
        // Eight rather than the ~14x this actually gets: the gzip header and
        // trailer are 18 bytes a chunk and the chunks here are 512 bytes, so the
        // ratio at this size is dominated by framing. The assertion is about the
        // direction; the *measurement* is in `tests/zarr_env.rs`, at a size
        // where framing is a rounding error.
        assert!(
            sizes[1] * 8 < sizes[0],
            "a constant uint8 volume stored {} bytes compressed against {} raw",
            sizes[1],
            sizes[0]
        );
        assert_eq!(
            counted[0], counted[1],
            "the counters report the bytes the plan moved, which compression does not change"
        );
    }

    // -------------------------------------------------------- round trip --

    /// Every supported element type, several chunk shapes, **both codecs**, and
    /// reads that both land on the chunk grid and deliberately do not.
    ///
    /// The codec axis is the point of the compression work: an element that came
    /// back wrong under `gzip` and right under `bytes` would be a decode bug,
    /// and it would show here for every type at once rather than in whichever
    /// integration test happened to use that type.
    #[test]
    fn every_element_type_round_trips_at_several_chunk_shapes_aligned_and_not() {
        let shape = [8, 6, 10];
        for dtype in EVERY_TYPE {
            for chunk in [[8, 6, 10], [4, 3, 5], [2, 2, 2], [3, 4, 7], [16, 16, 16]] {
                for compression in [Compression::None, Compression::Gzip(1), Compression::Gzip(9)] {
                let root = scratch("round-trip");
                let input = ramp(dtype, shape);
                let env = ZarrEnvironment::create_with_compression(
                    &root,
                    &input,
                    chunk,
                    CompressionPolicy::uniform(compression),
                )
                .unwrap();
                assert_eq!(env.compression_at(0).unwrap(), compression);

                // The whole volume, which is chunk-aligned by construction.
                let whole = env.level(0).unwrap();
                assert_eq!(whole, input, "{dtype:?} at chunk {chunk:?}");

                // A read that lands on the grid where the grid divides it, and a
                // read chosen to straddle chunks on every axis.
                for region in [
                    Region::new(&[0, 0, 0], &[4, 3, 5]),
                    Region::new(&[1, 1, 1], &[5, 4, 7]),
                    Region::new(&[3, 2, 4], &[1, 1, 1]),
                    Region::new(&[7, 5, 9], &[1, 1, 1]),
                ] {
                    let got = env.read(0, &region).unwrap();
                    let want = input.slice_region(&region).unwrap();
                    assert_eq!(
                        got.as_array().unwrap(),
                        &want,
                        "{dtype:?} at chunk {chunk:?}, region {region:?}, {}",
                        compression.name()
                    );
                }
                let _ = std::fs::remove_dir_all(&root);
                }
            }
        }
    }

    /// A misaligned write is still exactly the box it was given: nothing outside
    /// it moves, including inside the chunks it half-covers.
    #[test]
    fn a_misaligned_write_touches_exactly_its_own_box() {
        let root = scratch("misaligned-write");
        let shape = [8, 8, 8];
        let input = ramp(Dtype::U16, shape);
        let env = ZarrEnvironment::create(&root, &input, [4, 4, 4]).unwrap();

        let region = Region::new(&[1, 2, 3], &[3, 3, 3]);
        let patch = Voxels::filled(Dtype::U16, [3, 3, 3], 7.0).unwrap();
        env.write(
            0,
            &Region::whole(&[3, 3, 3]),
            &region,
            &BlockBuf::Array(patch),
        )
        .unwrap();
        assert!(env.serialised_writes() > 0, "this write straddles chunks");

        let got = env.level(0).unwrap();
        let got = got.view::<u16>().unwrap();
        let want = input.view::<u16>().unwrap();
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    let inside = (1..4).contains(&i) && (2..5).contains(&j) && (3..6).contains(&k);
                    let expected = if inside { 7 } else { want[[i, j, k]] };
                    assert_eq!(got[[i, j, k]], expected, "at {i},{j},{k}");
                }
            }
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// An unwritten voxel reads back as the sentinel a `Voxels` would hold, not
    /// as a convincing zero.
    #[test]
    fn an_unwritten_level_reads_back_as_the_unwritten_sentinel() {
        let root = scratch("unwritten");
        let input = Voxels::zeros(Dtype::F64, [4, 4, 4]).unwrap();
        let env = ZarrEnvironment::create(&root, &input, [2, 2, 2]).unwrap();
        // A level the plan creates and nothing writes.
        let array = StoredArray::create(
            &env.store,
            &level_path(9),
            9,
            Dtype::F64,
            &[4, 4, 4],
            &[2, 2, 2],
            Compression::for_dtype(Dtype::F64),
        )
        .unwrap();
        let block = array.read_as::<f64>(&Region::whole(&[4, 4, 4])).unwrap();
        assert!(block
            .view::<f64>()
            .unwrap()
            .iter()
            .all(|value| value.is_nan()));
        let _ = std::fs::remove_dir_all(&root);
    }

    // ------------------------------------------------------ chunk cover --

    #[test]
    fn a_region_on_the_grid_covers_every_chunk_it_touches() {
        assert!(partly_covered_chunks(&Region::new(&[0, 0, 0], &[4, 4, 4]), &[4, 4, 4]).is_empty());
        assert!(partly_covered_chunks(&Region::new(&[4, 8, 0], &[8, 4, 4]), &[4, 4, 4]).is_empty());
        // One voxel short on one axis and the two end chunks of that axis go.
        assert_eq!(
            partly_covered_chunks(&Region::new(&[0, 0, 0], &[3, 4, 4]), &[4, 4, 4]).len(),
            1
        );
        // A box wholly inside one chunk is that one chunk, partial.
        assert_eq!(
            partly_covered_chunks(&Region::new(&[1, 1, 1], &[2, 2, 2]), &[4, 4, 4]),
            vec![vec![0, 0, 0]]
        );
        // Rank-generic, for the side outputs.
        assert!(partly_covered_chunks(&Region::new(&[0, 0], &[6, 4]), &[3, 4]).is_empty());
    }

    // ------------------------------------------------------- the race --

    /// Two threads, two halves of one chunk, forty times.
    ///
    /// This is the measurement the guard exists for. `written` is what the two
    /// threads between them wrote; `readable` is what came back. Without the
    /// guard the second writer's decode-patch-encode overwrites the first's
    /// whole chunk and a half of the data is simply gone — no error, no
    /// diagnostic, a complete well-formed wrong array.
    ///
    /// The two arms run **the same code**, differing only in whether the locks
    /// are passed, so what is being compared is the guard and not two
    /// implementations.
    ///
    /// `compression` is the second axis, and it is here because compression
    /// changes the *shape* of the thing being raced: decode-patch-encode becomes
    /// decompress-patch-recompress, which is a longer window and, if anything,
    /// an easier one to lose data in. A guard verified only on raw arrays would
    /// not have been verified where it now matters most.
    fn partial_chunk_race(guarded: bool, compression: Compression) -> usize {
        const TRIALS: usize = 40;
        let root = scratch(&format!(
            "race-{}-{}",
            if guarded { "guarded" } else { "open" },
            compression.name()
        ));
        let store = Arc::new(FilesystemStore::new(&root).unwrap());
        let locks = ChunkLocks::new();
        let mut lost = 0;

        for trial in 0..TRIALS {
            // One chunk, exactly. Two disjoint halves of it, written at once.
            let shape = [16, 16, 16];
            let array = StoredArray::create(
                &store,
                &format!("/race{trial}"),
                trial as u64,
                Dtype::U32,
                &shape,
                &shape,
                compression,
            )
            .unwrap();
            let halves = [
                (Region::new(&[0, 0, 0], &[8, 16, 16]), 1u32),
                (Region::new(&[8, 0, 0], &[8, 16, 16]), 2u32),
            ];
            // A barrier, so the two decode-patch-encode windows actually
            // overlap. Without one the second thread's spawn latency often
            // lets the first finish, and the test would under-report a race
            // that is entirely real — the failure this whole module exists
            // for is precisely one that looks fine when it is not provoked.
            let start = std::sync::Barrier::new(halves.len());
            std::thread::scope(|scope| {
                for (region, value) in &halves {
                    let array = &array;
                    let locks = &locks;
                    let start = &start;
                    scope.spawn(move || {
                        let data = vec![*value; region.voxels()];
                        start.wait();
                        array
                            .write_as::<u32>(region, &data, guarded.then_some(locks))
                            .unwrap();
                    });
                }
            });

            let back: Voxels = array.read_as::<u32>(&Region::whole(&shape)).unwrap();
            let back = back.view::<u32>().unwrap();
            let lower_kept = back[[0, 0, 0]] == 1;
            let upper_kept = back[[8, 0, 0]] == 2;
            if !(lower_kept && upper_kept) {
                lost += 1;
            }
        }
        let _ = std::fs::remove_dir_all(&root);
        lost
    }

    /// **The guard, watched failing.** Without it, concurrent partial-chunk
    /// writes lose a whole half; with it, none of forty trials does.
    ///
    /// The unguarded arm asserts `> 0` rather than a rate: it is a race, and a
    /// scheduler that happened to serialise all forty would make a fixed
    /// threshold a flaky test rather than a stronger one. What is not negotiable
    /// is the guarded arm, which must be exactly zero.
    ///
    /// **Run on both a raw and a compressed array**, and the compressed arm is
    /// not a formality: it is the configuration this environment now writes by
    /// default for an integer level, so a guard that had only ever been measured
    /// on raw arrays would be a guard measured somewhere nobody runs.
    #[test]
    fn concurrent_partial_chunk_writes_lose_data_without_the_guard_and_not_with_it() {
        for compression in [Compression::None, Compression::Gzip(1)] {
            let what = compression.name();
            let guarded = partial_chunk_race(true, compression);
            assert_eq!(
                guarded, 0,
                "{guarded} of 40 guarded trials lost a half-chunk on a {what} array; the guard is \
                 not closing the race"
            );

            let open = partial_chunk_race(false, compression);
            // Printed rather than only asserted: the number is the evidence, and
            // a reader who wants to know how badly it fails should not have to
            // edit the test to find out. `cargo test -- --nocapture`.
            eprintln!(
                "partial-chunk race ({what}): {open}/40 trials lost a half-chunk unguarded, \
                 {guarded}/40 guarded"
            );
            assert!(
                open > 0,
                "40 unguarded trials on a {what} array lost nothing. Either `zarrs` has grown the \
                 chunk lock its source still has commented out, in which case this test has done \
                 its job and the guard can be reconsidered on evidence — or the two writers are \
                 not actually overlapping a chunk any more, in which case this test has stopped \
                 measuring anything and must be fixed before it is trusted."
            );
        }
    }

    /// A chunk-aligned write takes no locks at all, which is what makes
    /// alignment worth advising.
    #[test]
    fn an_aligned_write_is_lock_free_and_a_straddling_one_is_not() {
        let root = scratch("aligned");
        let input = Voxels::zeros(Dtype::U8, [8, 8, 8]).unwrap();
        let env = ZarrEnvironment::create(&root, &input, [4, 4, 4]).unwrap();
        assert_eq!(
            env.serialised_writes(),
            0,
            "the input write covers the grid"
        );

        let block = Voxels::filled(Dtype::U8, [4, 4, 4], 1.0).unwrap();
        env.write(
            0,
            &Region::whole(&[4, 4, 4]),
            &Region::new(&[4, 4, 4], &[4, 4, 4]),
            &BlockBuf::Array(block),
        )
        .unwrap();
        assert_eq!(env.serialised_writes(), 0);

        let block = Voxels::filled(Dtype::U8, [2, 2, 2], 2.0).unwrap();
        env.write(
            0,
            &Region::whole(&[2, 2, 2]),
            &Region::new(&[1, 1, 1], &[2, 2, 2]),
            &BlockBuf::Array(block),
        )
        .unwrap();
        assert_eq!(env.serialised_writes(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    // ------------------------------------------------------- accounting --

    /// The counters are filled the way the other environments fill them, which
    /// is what keeps the budget and the cost model working across environments.
    #[test]
    fn the_counters_agree_with_the_in_memory_environment() {
        use crate::env::ArrayEnvironment;

        let root = scratch("counters");
        let input = ramp(Dtype::U16, [8, 8, 8]);
        let region = Region::new(&[1, 1, 1], &[4, 4, 4]);

        let memory = ArrayEnvironment::new(input.clone(), 1, [4, 4, 4]).unwrap();
        let buf = memory.read(0, &region).unwrap();
        memory
            .write(1, &Region::whole(&[4, 4, 4]), &region, &buf)
            .unwrap();

        let storage = ZarrEnvironment::create(&root, &input, [4, 4, 4]).unwrap();
        // One phase, so one level beside the input, at the same shape and type.
        let extra = StoredArray::create(
            &storage.store,
            &level_path(1),
            1,
            Dtype::U16,
            &[8, 8, 8],
            &[4, 4, 4],
            Compression::for_dtype(Dtype::U16),
        )
        .unwrap();
        storage.levels.write().unwrap().push(Arc::new(extra));
        let buf = storage.read(0, &region).unwrap();
        storage
            .write(1, &Region::whole(&[4, 4, 4]), &region, &buf)
            .unwrap();

        assert_eq!(memory.counters().snapshot(), storage.counters().snapshot());
        assert_eq!(
            memory.counters().byte_snapshot(),
            storage.counters().byte_snapshot()
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A level allocated at one type and written at another is refused, rather
    /// than silently converted — the same guard `ArrayEnvironment` carries.
    #[test]
    fn writing_a_level_in_the_wrong_element_type_is_refused_by_name() {
        let root = scratch("wrong-type");
        let input = Voxels::zeros(Dtype::F64, [2, 2, 2]).unwrap();
        let env = ZarrEnvironment::create(&root, &input, [2, 2, 2]).unwrap();
        let region = Region::whole(&[2, 2, 2]);
        let wrong = BlockBuf::Array(Voxels::zeros(Dtype::U16, [2, 2, 2]).unwrap());
        let err = env
            .write(0, &region, &region, &wrong)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("uint16") && err.contains("float64"),
            "got: {err}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
