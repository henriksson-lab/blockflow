// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The injected environment. `docs/design/BLOCK_OPS.md` §"Simulating strategies
// to choose between them" makes this a hard requirement rather than a
// convenience: sources, sinks and ops must be **supplied**, never reached for,
// because
//
// * a strategy that reaches for real IO cannot be simulated, and
// * a simulation that exercises a different code path measures something other
//   than what will run.
//
// So `execute` in `strategy.rs` moves no bytes itself. It reads, applies and
// writes through this trait, and the *same strategy code* therefore runs
// against real arrays (`ArrayEnvironment`) and against a loader that only
// accumulates cost (`AccountingEnvironment`). The simulator and the executor
// are one program with two environments, which is the only arrangement in
// which "A beats B" means anything.
//
// Why `BlockBuf` is a concrete enum and not an associated type
// -----------------------------------------------------------
// An associated buffer type would make `Environment` non-object-safe, and the
// conformance suite is written as `for strategy in [Trivial, Enumerating,
// Greedy] { ... }` over trait objects. A two-variant enum that only the
// environments construct costs one match arm each and keeps every strategy on
// one code path. The strategy never inspects a `BlockBuf`; it hands it back.
//
// The **element type** is carried the same way and for the same reason, one
// layer in: `BlockBuf::Array` holds a `voxels::Voxels`, whose variant is the
// element type. See that module's header for the three pieces of evidence
// against a generic parameter. Both variants of `BlockBuf` know their dtype,
// because the simulated one has to price bytes and a byte is a width.
//
// Levels
// ------
// Level 0 is the workflow input. Level `p` (1..=n_phases) is the output of
// phase `p-1`, so the last level is the workflow output and levels in between
// are intermediates. That numbering is what makes "is this write a
// materialisation?" a comparison rather than a flag.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use ndarray::{ArrayD, Axis, IxDyn, Slice};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::region::Region;
use crate::sidecar::{Discarded, FragmentKey, Lifecycle, Sidecars};
use crate::voxels::{SideBuf, Voxels};

use super::decomposition::Decomposition;
use super::geometry::{chunks_touched, region_within};
use super::op::{Anchor, Chain, Output, SideBlock};

/// A block's worth of data, or a stand-in for one.
///
/// The simulated variant carries no data at all — that is the point. It lets a
/// strategy be exercised over volumes far larger than could be allocated, at
/// the cost of proving nothing about geometry (which the array variant proves,
/// at small scale).
#[derive(Debug, Clone, PartialEq)]
pub enum BlockBuf {
    Array(Voxels),
    Accounted {
        region: Region,
        /// The element type the block would hold. Carried because the *cost* of
        /// a simulated block is bytes and a byte is `voxels * width`, which is
        /// the whole reason the element type stopped being `f64`.
        dtype: Dtype,
        /// What is known about uniformity. `Some(v)` means every voxel is `v`.
        uniform: Option<f64>,
    },
}

impl BlockBuf {
    pub fn voxels(&self) -> usize {
        match self {
            BlockBuf::Array(array) => array.len(),
            BlockBuf::Accounted { region, .. } => region.voxels(),
        }
    }

    /// The element type this block holds, real or simulated.
    pub fn dtype(&self) -> Dtype {
        match self {
            BlockBuf::Array(array) => array.dtype(),
            BlockBuf::Accounted { dtype, .. } => *dtype,
        }
    }

    /// Bytes the block occupies decoded, which is the figure residency is
    /// counted in and the one an element type narrower than `f64` shrinks.
    pub fn bytes(&self) -> u64 {
        self.voxels() as u64 * self.dtype().size_of() as u64
    }

    pub fn as_array(&self) -> Result<&Voxels> {
        match self {
            BlockBuf::Array(array) => Ok(array),
            BlockBuf::Accounted { .. } => Err(Error::InvalidArgument(
                "block buffer holds no data: this is a simulated run".to_string(),
            )),
        }
    }

    /// The buffer's data, to be filled in.
    ///
    /// For an op that *produces* a block rather than transforming one — see
    /// `fragment::FragmentOp` — which gets its buffer from
    /// `Environment::constant` so that the same op runs under a simulated
    /// environment. `None` rather than an error, so that filling in a buffer
    /// that holds no data is a no-op the op writes as `if let`, not a failure it
    /// has to special-case.
    pub fn as_array_mut(&mut self) -> Option<&mut Voxels> {
        match self {
            BlockBuf::Array(array) => Some(array),
            BlockBuf::Accounted { .. } => None,
        }
    }
}

/// Counters an environment accumulates, whether it moves bytes or only counts
/// them.
#[derive(Debug, Default)]
pub struct EnvCounters {
    pub reads: AtomicU64,
    pub writes: AtomicU64,
    pub read_voxels: AtomicU64,
    pub write_voxels: AtomicU64,
    /// Bytes of level data moved, **decoded**.
    ///
    /// Beside the voxel counts rather than derived from them, because a run
    /// whose levels have different element types has no single bytes-per-voxel
    /// and a caller multiplying by one would get a number that means nothing.
    /// This is the figure the element type change is measured in: the same
    /// chain over the same volume moves an eighth of these bytes as `bool` as it
    /// does as `f64`.
    pub read_bytes: AtomicU64,
    pub write_bytes: AtomicU64,
    pub chunks_read: AtomicU64,
    pub ops_applied: AtomicU64,
    pub estimated_work: AtomicU64,
    pub resident_bytes: AtomicU64,
    pub peak_resident_bytes: AtomicU64,
    /// Sidecar traffic — per-block output that is not a pixel region. Counted
    /// separately from `reads`/`writes` rather than folded into them, because
    /// the two are priced completely differently: a region write is voxels
    /// through a chunk grid, a fragment is one small object.
    pub sidecar_writes: AtomicU64,
    pub sidecar_reads: AtomicU64,
    pub sidecar_bytes_written: AtomicU64,
    pub sidecar_bytes_read: AtomicU64,
    /// Traffic to the arrays an op writes **beside** its primary result. Counted
    /// separately from `writes`/`write_voxels` for the same reason sidecars are:
    /// a side output has its own element type and its own rank, so its voxels
    /// are not commensurable with the level's and adding them would produce one
    /// number that means nothing. `side_bytes_written` is the commensurable one,
    /// and it is the figure that was missing — the framework counted 95.2 MB of
    /// a run that wrote 158.6.
    pub side_writes: AtomicU64,
    pub side_elements: AtomicU64,
    pub side_bytes_written: AtomicU64,
}

impl EnvCounters {
    pub fn add_resident(&self, bytes: u64) {
        let now = self.resident_bytes.fetch_add(bytes, Ordering::SeqCst) + bytes;
        self.peak_resident_bytes.fetch_max(now, Ordering::SeqCst);
    }

    /// Saturating, because residency is bookkeeping and must never be the
    /// thing that panics a run. An unbalanced pair is a bug in the caller, and
    /// `peak_resident_bytes` going wrong is a wrong *number*, not wrong data.
    pub fn drop_resident(&self, bytes: u64) {
        let _ = self
            .resident_bytes
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                Some(current.saturating_sub(bytes))
            });
    }

    /// `(writes, reads, bytes written, bytes read)` for the sidecar store.
    ///
    /// Its own accessor rather than four more slots in `snapshot`, whose tuple
    /// several callers destructure positionally.
    pub fn sidecar_snapshot(&self) -> (u64, u64, u64, u64) {
        (
            self.sidecar_writes.load(Ordering::SeqCst),
            self.sidecar_reads.load(Ordering::SeqCst),
            self.sidecar_bytes_written.load(Ordering::SeqCst),
            self.sidecar_bytes_read.load(Ordering::SeqCst),
        )
    }

    /// `(bytes read, bytes written)` for level data, decoded.
    ///
    /// Its own accessor rather than two more slots in `snapshot`, whose tuple
    /// several callers destructure positionally.
    pub fn byte_snapshot(&self) -> (u64, u64) {
        (
            self.read_bytes.load(Ordering::SeqCst),
            self.write_bytes.load(Ordering::SeqCst),
        )
    }

    /// `(writes, elements, bytes written)` for the side outputs.
    ///
    /// Its own accessor rather than three more slots in `snapshot`, whose tuple
    /// several callers destructure positionally.
    pub fn side_snapshot(&self) -> (u64, u64, u64) {
        (
            self.side_writes.load(Ordering::SeqCst),
            self.side_elements.load(Ordering::SeqCst),
            self.side_bytes_written.load(Ordering::SeqCst),
        )
    }

    pub fn snapshot(&self) -> (u64, u64, u64, u64, u64, f64, u64) {
        (
            self.reads.load(Ordering::SeqCst),
            self.writes.load(Ordering::SeqCst),
            self.read_voxels.load(Ordering::SeqCst),
            self.write_voxels.load(Ordering::SeqCst),
            self.chunks_read.load(Ordering::SeqCst),
            self.estimated_work.load(Ordering::SeqCst) as f64,
            self.peak_resident_bytes.load(Ordering::SeqCst),
        )
    }
}

/// Where data comes from, where it goes, and who applies the ops.
///
/// Object-safe on purpose: strategies are exercised as trait objects in the
/// conformance suite.
pub trait Environment: Sync {
    fn volume(&self) -> [usize; 3];

    /// Create whatever intermediates the decomposition implies. Called once,
    /// before any task runs.
    fn prepare(&self, decomposition: &Decomposition) -> Result<()>;

    fn read(&self, level: usize, region: &Region) -> Result<BlockBuf>;

    /// Apply one slot of the chain over the whole buffer.
    ///
    /// Routing this through the environment rather than calling
    /// `Chain::apply` directly is what lets the simulated environment skip the
    /// arithmetic while the strategy stays byte-identical.
    ///
    /// `at` is where the buffer sits in the volume. The executor knows it (it
    /// is the read extent it just asked for) and the op may need it; see
    /// [`Anchor`]. It is threaded rather than recomputed so the two cannot
    /// disagree.
    fn apply(&self, slot: &Chain, input: &BlockBuf, at: &Anchor) -> Result<BlockBuf>;

    /// Write the sub-box `within` of `buf` to absolute position `valid`.
    fn write(&self, level: usize, within: &Region, valid: &Region, buf: &BlockBuf) -> Result<()>;

    // -------------------------------------------------- side outputs --
    //
    // A level is one array of one element type, which is the shape of every
    // op whose result is a volume. An op that produces several results has to
    // put the rest somewhere, and where it used to put them was outside the
    // framework — a custom environment writing on the side, with no term
    // anywhere for what it moved. Measured: 95.2 MB counted against 158.6 MB
    // written, short by 1.67x.
    //
    // These are beside `write` rather than a wider `write`, on the same
    // argument the sidecar block below is made with: a side output has its own
    // element type, its own rank and its own coordinate space, so it is
    // addressed by name and by a region of its own rather than by a level and
    // a region of the level's.
    //
    // `write_side` is the executor's entry point and is **not** an override
    // point: it counts, then stores. An environment supplies storage by
    // overriding `put_side` and gets the accounting it cannot then forget,
    // which is exactly the failure this exists to close.

    /// Create the array a side output goes to. Called once per declared output,
    /// before any task, by the executor — the same arrangement as
    /// [`Self::declare_sidecar`], and for the same reason: a block must not be
    /// the thing that creates the array it writes into.
    ///
    /// The default does nothing, which is right for an environment that only
    /// counts.
    fn declare_side_output(&self, _output: &Output) -> Result<()> {
        Ok(())
    }

    /// Produce one slot's side outputs for a block.
    ///
    /// Routed through the environment for the same reason [`Self::apply`] is:
    /// a simulated run must skip the arithmetic while the strategy stays
    /// byte-identical. The default does both — it dispatches on what the buffers
    /// actually hold rather than on which environment it is, so an environment
    /// that wraps another inherits the right behaviour instead of a plausible
    /// one.
    ///
    /// `block.regions` is one per declared output, from
    /// [`crate::op::BlockOp::side_region`]; the buffers come back in the same
    /// order and with those shapes, checked here.
    fn apply_side(
        &self,
        slot: &Chain,
        input: &BlockBuf,
        primary: &BlockBuf,
        block: &SideBlock<'_>,
    ) -> Result<Vec<SideBuf>> {
        let regions = block.regions;
        // Allocated through `side_constant` so that residency is booked the way
        // this environment books it, and `release_side` on the way out balances.
        let mut buffers: Vec<SideBuf> = regions
            .iter()
            .map(|region| self.side_constant(region))
            .collect();
        let (BlockBuf::Array(input), BlockBuf::Array(result)) = (input, primary) else {
            return Ok(buffers);
        };
        let produced = slot.apply_side(input, result, block)?;
        if produced.len() != regions.len() {
            return Err(Error::InvalidArgument(format!(
                "op {:?} declares {} side output(s) and produced {}. The declaration is what \
                 says an array exists and what shape it has, so a block that produces a \
                 different number of them has nowhere to put the difference.",
                slot.display_name(),
                regions.len(),
                produced.len()
            )));
        }
        for ((buffer, region), array) in buffers.iter_mut().zip(regions).zip(produced) {
            let Some(target) = buffer.as_array_mut() else {
                continue;
            };
            if array.shape() != region.shape.as_slice() {
                return Err(Error::ShapeMismatch {
                    expected: region.shape.clone(),
                    got: array.shape().to_vec(),
                });
            }
            target.assign(&array);
        }
        Ok(buffers)
    }

    /// Write one block's slice of a side output. **Counts, then stores.**
    ///
    /// Not an override point; see [`Self::put_side`].
    fn write_side(
        &self,
        output: &Output,
        phase: usize,
        region: &Region,
        buf: &SideBuf,
    ) -> Result<()> {
        self.put_side(output, phase, region, buf)?;
        self.counters().side_writes.fetch_add(1, Ordering::SeqCst);
        self.counters()
            .side_elements
            .fetch_add(region.voxels() as u64, Ordering::SeqCst);
        self.counters().side_bytes_written.fetch_add(
            region.voxels() as u64 * output.dtype.size_of() as u64,
            Ordering::SeqCst,
        );
        Ok(())
    }

    /// Where a side output's bytes go. The override point.
    ///
    /// The default discards them, which is the honest thing for an environment
    /// that holds no data: the *cost* is still counted by `write_side`, which is
    /// the whole point, and a simulated run is meant to be free.
    fn put_side(
        &self,
        _output: &Output,
        _phase: usize,
        _region: &Region,
        _buf: &SideBuf,
    ) -> Result<()> {
        Ok(())
    }

    /// A zero-filled side-output buffer of `region`'s shape, at `region`'s rank.
    ///
    /// The override point an environment that holds no data uses to allocate
    /// nothing. Separate from [`Self::constant`] because a side output has its
    /// own rank and its own coordinate space, and a level buffer has neither —
    /// merging them would put a `Vec<usize>` shape back into the type the rank
    /// was just pinned out of.
    fn side_constant(&self, region: &Region) -> SideBuf {
        self.counters()
            .add_resident(region.voxels() as u64 * std::mem::size_of::<f64>() as u64);
        SideBuf::zeros(region)
    }

    /// Release a side-output buffer's accounted residency.
    fn release_side(&self, buf: &SideBuf) {
        self.counters()
            .drop_resident(buf.elements() as u64 * std::mem::size_of::<f64>() as u64);
    }

    /// Is every voxel the same value? `None` means "not known", which disables
    /// the short circuit — the safe default, matching `constant_maps_to`.
    fn uniform(&self, buf: &BlockBuf) -> Option<f64>;

    /// A block of `region`'s shape and `dtype`, every element `value`.
    ///
    /// `dtype` is an argument rather than an implicit for the same reason
    /// `reach` takes `volume_len`: the caller is the only thing that knows which
    /// level this block is destined for, and a buffer allocated at the wrong
    /// width is a buffer the write refuses.
    fn constant(&self, dtype: Dtype, region: &Region, value: f64) -> Result<BlockBuf>;

    /// Release a buffer's accounted residency.
    fn release(&self, buf: &BlockBuf);

    /// Stage barrier: everything written to `level` is durable.
    fn finish(&self, level: usize) -> Result<()>;

    fn counters(&self) -> &EnvCounters;

    fn chunk_shape(&self) -> [usize; 3] {
        [1, 1, 1]
    }

    // ------------------------------------------------------- sidecars --
    //
    // Per-block output that is **not** a pixel region, beside the region
    // writes rather than somewhere else, so that it inherits the same event
    // stream and the same accounting and is available under every environment
    // — including the simulated one, which is what keeps a strategy that
    // produces fragments simulatable.
    //
    // The key is `(stream, phase, block)` and the value is bytes; see
    // `sidecar` for why both, and for why merging is the caller's job.
    //
    // The methods below are defaults over `sidecars()` on purpose: an
    // environment supplies a *store* and gets the declaration check, the
    // counters and the events without writing them, so three environments
    // cannot get them three subtly different ways.

    /// This environment's sidecar store, if it has one.
    fn sidecars(&self) -> Option<&Sidecars> {
        None
    }

    /// The store, or an error naming the environment that has none.
    fn require_sidecars(&self) -> Result<&Sidecars> {
        self.sidecars().ok_or_else(|| {
            Error::InvalidArgument(
                "this environment has no sidecar store, so there is nowhere to put per-block \
                 output that is not a pixel region. An environment that supports fragments \
                 returns one from `Environment::sidecars`."
                    .to_string(),
            )
        })
    }

    /// Create a stream, saying now whether its fragments survive the run.
    fn declare_sidecar(&self, stream: &str, lifecycle: Lifecycle) -> Result<()> {
        self.require_sidecars()?.declare(stream, lifecycle)
    }

    /// Write one block's fragment. The sidecar counterpart of [`Self::write`].
    fn write_sidecar(
        &self,
        stream: &str,
        phase: usize,
        block: [usize; 3],
        bytes: &[u8],
    ) -> Result<()> {
        self.require_sidecars()?
            .write(stream, phase, block, bytes)?;
        self.counters()
            .sidecar_writes
            .fetch_add(1, Ordering::SeqCst);
        self.counters()
            .sidecar_bytes_written
            .fetch_add(bytes.len() as u64, Ordering::SeqCst);
        Ok(())
    }

    /// One block's fragment, or `None` if that block wrote none.
    fn read_sidecar(
        &self,
        stream: &str,
        phase: usize,
        block: [usize; 3],
    ) -> Result<Option<Vec<u8>>> {
        let found = self.require_sidecars()?.read(stream, phase, block)?;
        self.counters().sidecar_reads.fetch_add(1, Ordering::SeqCst);
        if let Some(bytes) = &found {
            self.counters()
                .sidecar_bytes_read
                .fetch_add(bytes.len() as u64, Ordering::SeqCst);
        }
        Ok(found)
    }

    /// Every key in `stream`. Empty for a stream nobody wrote to.
    fn sidecar_keys(&self, stream: &str) -> Result<Vec<FragmentKey>> {
        self.require_sidecars()?.keys(stream)
    }

    /// Every fragment in `stream`, in key order.
    ///
    /// **Materialises the whole stream**, which for anything at scale
    /// reintroduces the residency problem per-block fragments exist to avoid —
    /// every block's output in memory at once, in the reader instead of in the
    /// producer. Prefer `fragment::fold_fragments`, which does the same walk
    /// over `sidecar_keys` + `read_sidecar` with one fragment resident.
    ///
    /// Kept because a small stream read whole is the convenient thing and this
    /// crate's own tests are exactly that case.
    fn sidecar_fragments(&self, stream: &str) -> Result<Vec<(FragmentKey, Vec<u8>)>> {
        let fragments = self.require_sidecars()?.fragments(stream)?;
        let bytes: u64 = fragments.iter().map(|(_, value)| value.len() as u64).sum();
        self.counters()
            .sidecar_reads
            .fetch_add(fragments.len() as u64, Ordering::SeqCst);
        self.counters()
            .sidecar_bytes_read
            .fetch_add(bytes, Ordering::SeqCst);
        Ok(fragments)
    }

    /// Remove every delete-on-exit stream, and say what went.
    ///
    /// The "exit" is this call. A `Drop` would have nowhere to report to and
    /// no way to fail loudly, which is the shape of the cleanup bug this is
    /// arranged against rather than the fix for it.
    fn discard_sidecars(&self) -> Result<Discarded> {
        self.require_sidecars()?.discard()
    }
}

// ------------------------------------------------------------------ real --

/// Real volumes held in memory, one per level, **each at its own element type**.
///
/// This is the geometry oracle's environment: small volumes, real values, so
/// identity ops must reproduce the input exactly and a window-sum op must
/// reproduce its whole-volume answer. It is deliberately *not* the streaming
/// path — that is `region_io`'s job, and a Zarr environment would go through
/// `ZarrRegionSink` because `zarrs` 0.23.13 loses data on concurrent partial
/// chunk writes.
pub struct ArrayEnvironment {
    volume: [usize; 3],
    levels: Vec<RwLock<Voxels>>,
    chunk: [usize; 3],
    /// The arrays ops write beside their primary result, by declared name.
    ///
    /// Allocated on declaration rather than on first write, and NaN-filled on
    /// the same argument the levels are: a voxel nobody wrote must be loud. The
    /// map is behind one lock because declaration is once-per-run and writes go
    /// through the inner lock; a `BTreeMap` rather than a `HashMap` so that
    /// iteration order is the declaration order a diagnostic reports in.
    ///
    /// Still `ArrayD<f64>`: a side output's **rank** is its own, which is the
    /// whole reason it is not a level, and its dtype is carried by the `Output`
    /// that declared it and charged by `write_side`.
    side: RwLock<BTreeMap<String, RwLock<ArrayD<f64>>>>,
    counters: EnvCounters,
    /// In-memory, to match the levels: an environment whose volumes are not
    /// shared cannot offer shared fragments either, and pretending otherwise
    /// would make a single-node test pass for a reason a distributed run does
    /// not have.
    sidecars: Sidecars,
}

impl ArrayEnvironment {
    /// `n_phases` levels are written; level 0 holds `input`.
    ///
    /// Every level gets `input`'s element type, which is right for the plans
    /// that keep one and is all this constructor can know. A plan whose phases
    /// change type — or shape — needs [`Self::for_decomposition`], which reads
    /// both off the plan.
    pub fn new(input: Voxels, n_phases: usize, chunk: [usize; 3]) -> Result<Self> {
        let volume = input.shape();
        let dtype = input.dtype();
        let mut levels = Vec::with_capacity(n_phases + 1);
        levels.push(RwLock::new(input));
        for _ in 0..n_phases {
            levels.push(RwLock::new(Voxels::unwritten(dtype, volume)?));
        }
        Ok(Self {
            volume,
            levels,
            chunk,
            side: RwLock::new(BTreeMap::new()),
            counters: EnvCounters::default(),
            sidecars: Sidecars::in_memory(),
        })
    }

    /// Levels shaped **and typed** by the plan: level `p+1` gets phase `p`'s
    /// volume and phase `p`'s element type.
    ///
    /// [`Self::new`] gives every level the input's shape and type, which is
    /// right for every plan that keeps one, and is the only thing that could be
    /// done while a plan *had* one of each. A phase that changes either needs
    /// its output level allocated at what it writes, and the decomposition is
    /// what says which — so both are read from the decomposition rather than
    /// passed alongside it, where they could disagree.
    pub fn for_decomposition(
        input: Voxels,
        decomposition: &Decomposition,
        chunk: [usize; 3],
    ) -> Result<Self> {
        let volume = input.shape();
        if volume != decomposition.volume {
            return Err(Error::InvalidArgument(format!(
                "array environment: the input is {volume:?} and the decomposition reads level \
                 0 as {:?}",
                decomposition.volume
            )));
        }
        if input.dtype() != decomposition.dtype {
            return Err(Error::InvalidArgument(format!(
                "array environment: the input holds {} and the decomposition reads level 0 as \
                 {}",
                input.dtype().numpy_name(),
                decomposition.dtype.numpy_name()
            )));
        }
        let mut levels = Vec::with_capacity(decomposition.n_levels());
        levels.push(RwLock::new(input));
        for (index, phase) in decomposition.phases.iter().enumerate() {
            levels.push(RwLock::new(Voxels::unwritten(
                decomposition.dtype_at(index + 1),
                phase.volume(),
            )?));
        }
        Ok(Self {
            volume,
            levels,
            chunk,
            side: RwLock::new(BTreeMap::new()),
            counters: EnvCounters::default(),
            sidecars: Sidecars::in_memory(),
        })
    }

    fn level_guard(&self, level: usize) -> std::sync::RwLockReadGuard<'_, Voxels> {
        self.levels[level]
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The shape of one level, which is not `volume()` once a phase changes it.
    pub fn level_shape(&self, level: usize) -> [usize; 3] {
        self.level_guard(level).shape()
    }

    /// The element type of one level, which is not the workflow's once a phase
    /// changes it.
    pub fn level_dtype(&self, level: usize) -> Dtype {
        self.level_guard(level).dtype()
    }

    /// The last level: the workflow output.
    pub fn output(&self) -> Voxels {
        self.levels
            .last()
            .expect("at least one level")
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn level(&self, level: usize) -> Voxels {
        self.level_guard(level).clone()
    }

    /// One side output, by declared name, or `None` if nothing declared it.
    ///
    /// Any rank: what came back from [`Output::shape`] is what was allocated.
    pub fn side_output(&self, name: &str) -> Option<ArrayD<f64>> {
        let outer = self
            .side
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        outer.get(name).map(|array| {
            array
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
        })
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
}

impl Environment for ArrayEnvironment {
    fn volume(&self) -> [usize; 3] {
        self.volume
    }

    fn prepare(&self, decomposition: &Decomposition) -> Result<()> {
        if decomposition.n_phases() + 1 != self.levels.len() {
            return Err(Error::InvalidArgument(format!(
                "array environment built for {} phases, decomposition has {}",
                self.levels.len() - 1,
                decomposition.n_phases()
            )));
        }
        // Per level, because a phase owns its volume and its element type. This
        // is the check that used to be `Decomposition::check` refusing the plan
        // outright: an environment which allocated one shape for every level
        // cannot host a phase that changes shape, and that is a fact about the
        // environment, not about the plan. `for_decomposition` builds one that
        // can.
        for level in 0..self.levels.len() {
            let held = self.level_shape(level);
            let wanted = decomposition.volume_at(level);
            if held != wanted {
                return Err(Error::InvalidArgument(format!(
                    "array environment holds level {level} as {held:?} and the decomposition \
                     writes it as {wanted:?}. `ArrayEnvironment::for_decomposition` allocates \
                     each level at the shape its phase gives it."
                )));
            }
            let held = self.level_dtype(level);
            let wanted = decomposition.dtype_at(level);
            if held != wanted {
                return Err(Error::InvalidArgument(format!(
                    "array environment holds level {level} as {} and the decomposition writes \
                     it as {}. `ArrayEnvironment::for_decomposition` allocates each level at \
                     the element type its phase gives it.",
                    held.numpy_name(),
                    wanted.numpy_name()
                )));
            }
        }
        Ok(())
    }

    fn read(&self, level: usize, region: &Region) -> Result<BlockBuf> {
        region_within(region, &self.level_shape(level), "block-op read")?;
        let array = self.level_guard(level).slice_region(region)?;
        self.counters.reads.fetch_add(1, Ordering::SeqCst);
        self.counters
            .read_voxels
            .fetch_add(array.len() as u64, Ordering::SeqCst);
        self.counters
            .read_bytes
            .fetch_add(array.bytes(), Ordering::SeqCst);
        self.counters
            .chunks_read
            .fetch_add(chunks_touched(region, &self.chunk), Ordering::SeqCst);
        self.counters.add_resident(array.bytes());
        Ok(BlockBuf::Array(array))
    }

    fn apply(&self, slot: &Chain, input: &BlockBuf, at: &Anchor) -> Result<BlockBuf> {
        let array = input.as_array()?;
        // Allocated from what the chain **declares**, not from what it was
        // handed. That one line is the difference between a phase that may
        // translate its read and a phase that may resize it.
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
        // The caller releases `input`; releasing it here too would double-count.
        self.counters.add_resident(out.bytes());
        Ok(BlockBuf::Array(out))
    }

    fn write(&self, level: usize, within: &Region, valid: &Region, buf: &BlockBuf) -> Result<()> {
        let array = buf.as_array()?;
        region_within(valid, &self.level_shape(level), "block-op write")?;
        if valid.voxels() == 0 {
            return Ok(());
        }
        let source = array.slice_region(within)?;
        let mut guard = self.levels[level]
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.assign_region(valid, &source)?;
        self.counters.writes.fetch_add(1, Ordering::SeqCst);
        self.counters
            .write_voxels
            .fetch_add(valid.voxels() as u64, Ordering::SeqCst);
        self.counters
            .write_bytes
            .fetch_add(source.bytes(), Ordering::SeqCst);
        Ok(())
    }

    /// Allocate the array, NaN-filled, and refuse a second declaration that
    /// disagrees with the first.
    ///
    /// Two ops writing one name is legitimate — a phase and a later phase may
    /// both contribute — but only if they agree about what the array *is*. Two
    /// disagreeing declarations are a plan whose outputs depend on which op ran
    /// last, which is the class of silent wrongness this crate exists to remove.
    fn declare_side_output(&self, output: &Output) -> Result<()> {
        let mut outer = self
            .side
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = outer.get(&output.name) {
            let held = existing
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if held.shape() != output.shape.as_slice() {
                return Err(Error::InvalidArgument(format!(
                    "side output {:?} was declared as {:?} and is now declared as {:?}",
                    output.name,
                    held.shape(),
                    output.shape
                )));
            }
            return Ok(());
        }
        outer.insert(
            output.name.clone(),
            RwLock::new(ArrayD::from_elem(IxDyn(&output.shape), f64::NAN)),
        );
        Ok(())
    }

    fn put_side(
        &self,
        output: &Output,
        _phase: usize,
        region: &Region,
        buf: &SideBuf,
    ) -> Result<()> {
        let Some(array) = buf.as_array() else {
            return Err(Error::InvalidArgument(
                "side buffer holds no data: this is a simulated run".to_string(),
            ));
        };
        let outer = self
            .side
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let held = outer.get(&output.name).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "side output {:?} was written before it was declared; the executor declares \
                 every output of every phase before it runs a task",
                output.name
            ))
        })?;
        let mut guard = held
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        region_within(region, guard.shape(), "side-output write")?;
        if region.voxels() == 0 {
            return Ok(());
        }
        let mut target = guard.view_mut();
        for (axis, (&start, &len)) in region.start.iter().zip(region.shape.iter()).enumerate() {
            target.slice_axis_inplace(Axis(axis), Slice::from(start..start + len));
        }
        target.assign(array);
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

/// A level region's extent as the rank-3 array it is, or an error saying so.
pub(crate) fn block_shape(region: &Region) -> Result<[usize; 3]> {
    if region.shape.len() != 3 {
        return Err(Error::InvalidArgument(format!(
            "a level's blocks are 3-D, got a region of rank {}",
            region.shape.len()
        )));
    }
    Ok([region.shape[0], region.shape[1], region.shape[2]])
}

// ------------------------------------------------------------- simulated --

/// Accumulates cost; holds no data.
///
/// This is the design's "noop loader": voxels and chunks read, bytes moved,
/// ops applied, peak resident bytes, all counted with nothing allocated. It is
/// what makes a scheduling assertion affordable on a volume far larger than
/// anything worth allocating, and what turns the noop machinery from a test
/// harness into a simulator.
///
/// `emptiness` simulates the measured 37-41 % all-empty block fraction, which
/// no static model can know in advance and which the greedy runtime half is
/// there to exploit. It is a *deterministic* function of the region so that a
/// simulated run is reproducible.
pub struct AccountingEnvironment {
    volume: [usize; 3],
    chunk: [usize; 3],
    bytes_per_voxel: u64,
    /// The element type the **ops** see.
    ///
    /// Deliberately independent of `bytes_per_voxel`, which is the *storage*
    /// width this run is pricing, and they answer different questions: this
    /// crate's callers already pass `2` here for a `u16` volume while running an
    /// `f64` chain over it, because a narrow volume decoded to a wider element
    /// is the ordinary case and the IO cost is the narrow one. So this defaults
    /// to `f64` — the type every op in this crate accepted before `accepts`
    /// existed — and [`Self::with_dtype`] states it where a caller means
    /// something else.
    dtype: Dtype,
    /// Fraction of blocks reported uniform at `fill_value`, in [0, 1].
    emptiness: f64,
    fill_value: f64,
    counters: EnvCounters,
    /// Real fragments, in a simulated environment, and deliberately so. A
    /// voxel can be fabricated from a region; a fragment's bytes are the
    /// *caller's* and cannot be. A store that counted and discarded them would
    /// make a simulated run of a strategy that reads its own fragments back
    /// take a code path the real run does not.
    sidecars: Sidecars,
}

impl AccountingEnvironment {
    pub fn new(volume: [usize; 3], chunk: [usize; 3], bytes_per_voxel: u64) -> Self {
        Self {
            volume,
            chunk,
            bytes_per_voxel,
            dtype: Dtype::F64,
            emptiness: 0.0,
            fill_value: 0.0,
            counters: EnvCounters::default(),
            sidecars: Sidecars::in_memory(),
        }
    }

    /// State the element type the ops see. `bytes_per_voxel` is untouched: see
    /// the field's own documentation for why the two are separate.
    pub fn with_dtype(mut self, dtype: Dtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Report a deterministic `emptiness` fraction of blocks as uniformly
    /// `fill_value`, halo included.
    pub fn with_emptiness(mut self, emptiness: f64, fill_value: f64) -> Self {
        self.emptiness = emptiness.clamp(0.0, 1.0);
        self.fill_value = fill_value;
        self
    }

    /// Deterministic from the region's lower corner, so two runs of the same
    /// decomposition see the same empty blocks.
    fn looks_empty(&self, region: &Region) -> bool {
        if self.emptiness <= 0.0 {
            return false;
        }
        let mut hash = 0xcbf29ce484222325u64;
        for &value in region.start.iter().chain(region.shape.iter()) {
            hash ^= value as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        ((hash % 1_000_000) as f64 / 1_000_000.0) < self.emptiness
    }
}

impl Environment for AccountingEnvironment {
    fn volume(&self) -> [usize; 3] {
        self.volume
    }

    fn prepare(&self, decomposition: &Decomposition) -> Result<()> {
        // One volume, for every level: this environment counts regions against a
        // single extent and has no per-level shape to check a read against. A
        // plan whose phases change shape is refused *here*, by the environment
        // that cannot host it, rather than by the plan's own guard.
        let uniform = decomposition.uniform_volume().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "accounting environment has one volume {:?}, and this decomposition changes \
                 shape between levels: {:?}",
                self.volume,
                (0..decomposition.n_levels())
                    .map(|level| decomposition.volume_at(level))
                    .collect::<Vec<_>>()
            ))
        })?;
        if uniform != self.volume {
            return Err(Error::InvalidArgument(format!(
                "accounting environment volume {:?} disagrees with decomposition {:?}",
                self.volume, decomposition.volume
            )));
        }
        Ok(())
    }

    fn read(&self, _level: usize, region: &Region) -> Result<BlockBuf> {
        region_within(region, &self.volume, "simulated read")?;
        self.counters.reads.fetch_add(1, Ordering::SeqCst);
        self.counters
            .read_voxels
            .fetch_add(region.voxels() as u64, Ordering::SeqCst);
        self.counters.read_bytes.fetch_add(
            region.voxels() as u64 * self.bytes_per_voxel,
            Ordering::SeqCst,
        );
        self.counters
            .chunks_read
            .fetch_add(chunks_touched(region, &self.chunk), Ordering::SeqCst);
        self.counters
            .add_resident(region.voxels() as u64 * self.bytes_per_voxel);
        Ok(BlockBuf::Accounted {
            region: region.clone(),
            dtype: self.dtype,
            uniform: self.looks_empty(region).then_some(self.fill_value),
        })
    }

    fn apply(&self, slot: &Chain, input: &BlockBuf, _at: &Anchor) -> Result<BlockBuf> {
        let BlockBuf::Accounted {
            region,
            dtype,
            uniform,
        } = input
        else {
            return Err(Error::InvalidArgument(
                "accounting environment was handed a real array".to_string(),
            ));
        };
        self.counters.ops_applied.fetch_add(1, Ordering::SeqCst);
        self.counters.estimated_work.fetch_add(
            (region.voxels() as f64 * slot.cost_per_voxel()) as u64,
            Ordering::SeqCst,
        );
        // The output's extent and element type come from the chain's own
        // declaration, exactly as they do in the real environment — a simulated
        // run of a resizing phase must count the resized block, or it is
        // simulating a different plan. The region's *start* is kept because
        // nothing here reads it; only its extent is priced.
        let produced = slot.output_shape(block_shape(region)?)?;
        let region = Region::new(&region.start, &produced);
        let dtype = slot.produces(*dtype)?;
        // The only thing a data-free run can say about the output's uniformity
        // is what the op has *declared*, which is exactly the algebra the short
        // circuit is licensed by.
        let uniform = uniform.and_then(|value| slot.constant_maps_to(value));
        self.counters
            .add_resident(region.voxels() as u64 * self.bytes_per_voxel);
        Ok(BlockBuf::Accounted {
            region,
            dtype,
            uniform,
        })
    }

    fn write(&self, _level: usize, _within: &Region, valid: &Region, buf: &BlockBuf) -> Result<()> {
        self.counters.writes.fetch_add(1, Ordering::SeqCst);
        self.counters
            .write_voxels
            .fetch_add(valid.voxels() as u64, Ordering::SeqCst);
        let _ = buf;
        self.counters.write_bytes.fetch_add(
            valid.voxels() as u64 * self.bytes_per_voxel,
            Ordering::SeqCst,
        );
        Ok(())
    }

    fn uniform(&self, buf: &BlockBuf) -> Option<f64> {
        match buf {
            BlockBuf::Accounted { uniform, .. } => *uniform,
            BlockBuf::Array(_) => None,
        }
    }

    fn constant(&self, dtype: Dtype, region: &Region, value: f64) -> Result<BlockBuf> {
        self.counters
            .add_resident(region.voxels() as u64 * self.bytes_per_voxel);
        Ok(BlockBuf::Accounted {
            region: region.clone(),
            dtype,
            uniform: Some(value),
        })
    }

    /// Nothing, which is what a simulated run allocates. The residency is still
    /// booked, because the *cost* of a side output is real even where its bytes
    /// are not.
    fn side_constant(&self, region: &Region) -> SideBuf {
        self.counters
            .add_resident(region.voxels() as u64 * std::mem::size_of::<f64>() as u64);
        SideBuf::Accounted {
            elements: region.voxels(),
        }
    }

    fn release(&self, buf: &BlockBuf) {
        self.counters
            .drop_resident(buf.voxels() as u64 * self.bytes_per_voxel);
    }

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
    use crate::probes::{AffineOp, IdentityOp};
    use ndarray::Array3;

    #[test]
    fn the_array_environment_round_trips_a_region() {
        let mut input = Array3::<f64>::zeros((4, 4, 4));
        for (flat, value) in input.iter_mut().enumerate() {
            *value = flat as f64;
        }
        let env = ArrayEnvironment::new(input.clone().into(), 1, [2, 2, 2]).unwrap();
        let region = Region::new(&[1, 1, 1], &[2, 2, 2]);
        let buf = env.read(0, &region).unwrap();
        env.write(1, &Region::new(&[0, 0, 0], &[2, 2, 2]), &region, &buf)
            .unwrap();
        let out = env.output();
        let out = out.view::<f64>().unwrap();
        assert_eq!(out[[1, 1, 1]], input[[1, 1, 1]]);
        assert_eq!(out[[2, 2, 2]], input[[2, 2, 2]]);
        // nothing else was written, and the sentinel says so
        assert!(out[[0, 0, 0]].is_nan());
    }

    #[test]
    fn uniformity_is_detected_in_real_data_and_declared_in_simulated_data() {
        let env =
            ArrayEnvironment::new(Array3::from_elem((4, 4, 4), 7.0).into(), 1, [4, 4, 4]).unwrap();
        let buf = env.read(0, &Region::whole(&[4, 4, 4])).unwrap();
        assert_eq!(env.uniform(&buf), Some(7.0));

        let sim = AccountingEnvironment::new([4, 4, 4], [4, 4, 4], 8).with_emptiness(1.0, 0.0);
        let buf = sim.read(0, &Region::whole(&[4, 4, 4])).unwrap();
        assert_eq!(sim.uniform(&buf), Some(0.0));
        // and the declared algebra propagates
        let doubled = sim
            .apply(
                &Chain::op(AffineOp::new("d", 2.0, 1.0, [0, 0, 0])),
                &buf,
                &Anchor::whole([4, 4, 4]),
            )
            .unwrap();
        assert_eq!(sim.uniform(&doubled), Some(1.0));
    }

    #[test]
    fn a_simulated_run_allocates_nothing_for_a_full_scale_read() {
        // 2094 x 13316 x 3369 is the motivating dataset. Reading it as one
        // region would be 93.9 Gvoxel; the accounting environment counts it
        // instantly.
        let sim = AccountingEnvironment::new([2094, 13316, 3369], [128, 128, 128], 2);
        let buf = sim.read(0, &Region::whole(&[2094, 13316, 3369])).unwrap();
        assert_eq!(buf.voxels(), 2094 * 13316 * 3369);
        assert_eq!(
            sim.counters().read_voxels.load(Ordering::SeqCst),
            2094u64 * 13316 * 3369
        );
        sim.apply(
            &Chain::op(IdentityOp::new("noop", [0, 0, 0])),
            &buf,
            &Anchor::whole([2094, 13316, 3369]),
        )
        .unwrap();
    }

    /// The measurement the element-type change exists for, taken through the
    /// environment's own counter rather than argued: the identical chain over
    /// the identical extent moves an eighth of the bytes.
    #[test]
    fn a_narrow_level_reads_an_eighth_of_the_bytes_a_wide_one_does() {
        let extent = [16, 16, 16];
        let region = Region::whole(&extent);

        let wide = ArrayEnvironment::new(Array3::from_elem((16, 16, 16), 0.0f64).into(), 1, extent)
            .unwrap();
        wide.read(0, &region).unwrap();
        let (wide_bytes, _) = wide.counters().byte_snapshot();

        let narrow =
            ArrayEnvironment::new(Array3::from_elem((16, 16, 16), false).into(), 1, extent)
                .unwrap();
        narrow.read(0, &region).unwrap();
        let (narrow_bytes, _) = narrow.counters().byte_snapshot();

        assert_eq!(wide_bytes, 16 * 16 * 16 * 8);
        assert_eq!(narrow_bytes, 16 * 16 * 16);
        assert_eq!(wide_bytes, narrow_bytes * 8);
        // and the residency the budget is stated in moved with it
        assert_eq!(
            wide.counters().peak_resident_bytes.load(Ordering::SeqCst),
            narrow.counters().peak_resident_bytes.load(Ordering::SeqCst) * 8
        );
    }

    /// A level allocated at one type and written at another is refused, rather
    /// than silently converted. This is the guard `prepare` gained when a level
    /// stopped being `f64` by definition.
    #[test]
    fn writing_a_level_in_the_wrong_element_type_is_refused_by_name() {
        let env = ArrayEnvironment::new(Array3::from_elem((2, 2, 2), 0.0f64).into(), 1, [2, 2, 2])
            .unwrap();
        let region = Region::whole(&[2, 2, 2]);
        let wrong = BlockBuf::Array(Voxels::zeros(Dtype::U16, [2, 2, 2]).unwrap());
        let err = env
            .write(1, &region, &region, &wrong)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("uint16") && err.contains("float64"),
            "got: {err}"
        );
    }
}
