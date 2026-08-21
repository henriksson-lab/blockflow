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
// Images
// ------
// Image 0 is the workflow input. Image `p` (1..=n_phases) is the output of
// phase `p-1`, so the last image is the workflow output and images in between
// are intermediates. That numbering is what makes "is this write a
// materialisation?" a comparison rather than a flag.
//
// A run may also be handed **more arrays than one**, and those are images too:
// read through the same `read`, priced by the same counters, named by the same
// `source_images`. They are addressed in a disjoint high range —
// `assemble::ImageId::supplied(i)` — so that adding one renumbers nothing, and
// because a caller has to know the address before it builds the ops that read
// it, which is before the plan knows how many phases it has. See
// `ArrayEnvironment::with_inputs`. The one behavioural difference is that they
// are never written and never freed: no phase produces one, so nothing could
// rebuild it, which is the distinction `decomposition::ImageKind` records and
// `Visibility` cannot.
//
// An image is **allocated when something first writes to it**, not when the
// environment is built. See [`ImageStore`] for the measurement that is for; the
// short form is that a plan's images are alive over *lifetimes*, and an
// environment that allocated all of them at once cost the sum of the lifetimes
// rather than their largest overlap.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;

use ndarray::{ArrayD, Axis, IxDyn, Slice};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::region::Region;
use crate::sidecar::{Discarded, FragmentKey, Lifecycle, Sidecars};
use crate::voxels::{SideBuf, Voxels};

use crate::assemble::{describe_image, is_supplied_image, ImageId};

use super::decomposition::Decomposition;
use super::geometry::{chunks_touched, region_within};
use super::op::{Anchor, Chain, Output, Placement, SideBlock, SourceInputs};

/// Unwrap the executor's `(image, buffer)` list into the `(image, &Voxels)` form
/// [`SourceInputs`] holds.
///
/// A free function rather than a method on `SourceInputs` because the borrow has
/// to outlive the call: the `Vec` it returns owns the references, and the caller
/// keeps it alive for as long as it holds the `SourceInputs`.
pub(crate) fn as_source_arrays(
    sources: &[(usize, BlockBuf)],
) -> Result<Vec<(crate::assemble::ImageId, &Voxels)>> {
    sources
        .iter()
        .map(|(image, buf)| Ok(((*image).into(), buf.as_array()?)))
        .collect()
}

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
    /// Bytes of image data moved, **decoded**.
    ///
    /// Beside the voxel counts rather than derived from them, because a run
    /// whose images have different element types has no single bytes-per-voxel
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
    /// are not commensurable with the image's and adding them would produce one
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

    /// `(bytes read, bytes written)` for image data, decoded.
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

    fn read(&self, image: usize, region: &Region) -> Result<BlockBuf>;

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
    ///
    /// `sources` is the same arrangement for the images the slot's
    /// [`Chain::Source`] leaves read: one `(image, buffer)` per image in
    /// `PhaseDecomposition::source_images`, each holding **the same extent as
    /// `input`**, read by the executor through [`Self::read`] so that its bytes
    /// are counted exactly where every other read's are. Empty for every slot
    /// with no source leaf, which is every slot this crate shipped before they
    /// existed.
    ///
    /// **A parameter rather than a second method.** A defaulted
    /// `apply_with_sources` would leave the four existing implementations
    /// silently ignoring the operands, and "silently ignoring an operand" is the
    /// precise shape of the wrong answer this whole change exists to remove —
    /// a complete, well-formed volume combined against nothing.
    fn apply(
        &self,
        slot: &Chain,
        input: &BlockBuf,
        sources: &[(usize, BlockBuf)],
        at: &Placement,
    ) -> Result<BlockBuf>;

    /// Write the sub-box `within` of `buf` to absolute position `valid`.
    fn write(&self, image: usize, within: &Region, valid: &Region, buf: &BlockBuf) -> Result<()>;

    // -------------------------------------------------- side outputs --
    //
    // An image is one array of one element type, which is the shape of every
    // op whose result is a volume. An op that produces several results has to
    // put the rest somewhere, and where it used to put them was outside the
    // framework — a custom environment writing on the side, with no term
    // anywhere for what it moved. Measured: 95.2 MB counted against 158.6 MB
    // written, short by 1.67x.
    //
    // These are beside `write` rather than a wider `write`, on the same
    // argument the sidecar block below is made with: a side output has its own
    // element type, its own rank and its own coordinate space, so it is
    // addressed by name and by a region of its own rather than by an image and
    // a region of the image's.
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
    ///
    /// `sources` is the list [`Self::apply`] was handed for this block, in the
    /// same form and for the same reason: a side output may be a function of
    /// every array the phase read and not only of the one the op was applied to.
    /// It is unwrapped **after** the simulated arm returns, because an accounted
    /// block holds no array and asking one for a buffer before the arithmetic is
    /// known to be happening would turn a counting run into a refusal.
    fn apply_side(
        &self,
        slot: &Chain,
        input: &BlockBuf,
        sources: &[(usize, BlockBuf)],
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
        let stored = as_source_arrays(sources)?;
        let produced = slot.apply_side(input, SourceInputs::new(&stored), result, block)?;
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
    /// own rank and its own coordinate space, and an image buffer has neither —
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
    /// image this block is destined for, and a buffer allocated at the wrong
    /// width is a buffer the write refuses.
    fn constant(&self, dtype: Dtype, region: &Region, value: f64) -> Result<BlockBuf>;

    /// Release a buffer's accounted residency.
    fn release(&self, buf: &BlockBuf);

    // ------------------------------------------------ buffer arithmetic --
    //
    // Three operations on buffers the *executor* holds, rather than on images.
    // They exist for the iterative phase (`crate::iterate`), whose two private
    // ping-pong buffers are whole-volume blocks the executor allocates through
    // `constant` and owns for the length of the phase — never images, because
    // nothing outside the phase can see them and the plan allocates no image for
    // them.
    //
    // They are on this trait, and not free functions over `BlockBuf`, for the
    // reason `apply` is: a simulated run must skip the arithmetic while the
    // executor stays byte-identical, and residency must be booked the way the
    // environment books it. Each is a **default** that dispatches on what the
    // buffer actually holds rather than on which environment it is, so an
    // environment that wraps another inherits the right behaviour instead of a
    // plausible one — the same arrangement `apply_side` uses.
    //
    // Every region here is stated in the **volume's** coordinates, with the
    // buffer saying which part of that volume it holds. Absolute throughout, so
    // there is one origin to get wrong instead of two.

    /// The part of `buf` — which holds `holds` — covering `region`.
    fn slice(&self, buf: &BlockBuf, holds: &Region, region: &Region) -> Result<BlockBuf> {
        let within = relative(region, holds, "buffer slice")?;
        match buf {
            BlockBuf::Array(array) => {
                let taken = array.slice_region(&within)?;
                self.counters().add_resident(taken.bytes());
                Ok(BlockBuf::Array(taken))
            }
            BlockBuf::Accounted { dtype, .. } => {
                self.counters()
                    .add_resident(region.voxels() as u64 * dtype.size_of() as u64);
                Ok(BlockBuf::Accounted {
                    region: region.clone(),
                    dtype: *dtype,
                    // Nothing is claimed about a sub-box of a buffer whose
                    // uniformity was a property of the whole. `None` disables the
                    // short circuit, which is the safe default everywhere else in
                    // this file too.
                    uniform: None,
                })
            }
        }
    }

    /// Copy `source`, which covers `region`, into `target`, which holds `holds`.
    ///
    /// `&mut` rather than interior mutability because the executor owns these
    /// buffers outright. **This is where a distributed run needs a barrier**: the
    /// blocks of one substage write disjoint cores and could run concurrently,
    /// but every block of substage `k+1` reads cores its neighbours wrote at `k`,
    /// so the substage boundary is a real exchange point. In one process that is
    /// the end of a loop body; across processes it is a barrier and a shared
    /// private buffer, and neither is built here.
    fn place(
        &self,
        target: &mut BlockBuf,
        holds: &Region,
        region: &Region,
        source: &BlockBuf,
    ) -> Result<()> {
        let within = relative(region, holds, "buffer placement")?;
        match target {
            BlockBuf::Array(array) => array.assign_region(&within, source.as_array()?),
            // Nothing to move, and nothing to charge: a write into a buffer that
            // holds no data is the simulated case, and the *cost* of it was
            // already booked when the buffer was allocated.
            BlockBuf::Accounted { .. } => Ok(()),
        }
    }

    /// Do these two buffers hold the same values?
    ///
    /// `None` means the environment cannot tell, which is the honest answer for
    /// one that holds no data — and the reason an iterative phase refuses to run
    /// under such an environment rather than guessing a substage count. See
    /// `crate::iterate`.
    ///
    /// Equality is the element type's own, so **a NaN never equals itself** and an
    /// iteration that produces one never converges. That is the behaviour worth
    /// having: it ends at the runaway limit, naming the op, rather than settling
    /// on a volume with a hole in it.
    fn same(&self, left: &BlockBuf, right: &BlockBuf) -> Option<bool> {
        match (left, right) {
            (BlockBuf::Array(left), BlockBuf::Array(right)) => Some(left == right),
            _ => None,
        }
    }

    /// Run one substage of an iterative op over one block.
    ///
    /// Routed through the environment for exactly [`Self::apply`]'s reason. The
    /// output is the operands' shape and element type: an iterative phase neither
    /// resizes nor retypes, because the running operand of substage `k+1` is the
    /// output of substage `k` and a step that changed either could not be
    /// iterated.
    fn apply_substage(
        &self,
        op: &dyn crate::iterate::IterativeOp,
        index: usize,
        operands: &[BlockBuf],
        at: &Anchor,
    ) -> Result<BlockBuf> {
        let first = operands.first().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "iterative op {:?} was run with no operands; `check_iterative` refuses such an \
                 op when the plan is built",
                op.name()
            ))
        })?;
        self.counters().ops_applied.fetch_add(1, Ordering::SeqCst);
        self.counters().estimated_work.fetch_add(
            (first.voxels() as f64 * op.cost_per_voxel()) as u64,
            Ordering::SeqCst,
        );
        let arrays: Option<Vec<&Voxels>> = operands
            .iter()
            .map(|buf| match buf {
                BlockBuf::Array(array) => Some(array),
                BlockBuf::Accounted { .. } => None,
            })
            .collect();
        let Some(arrays) = arrays else {
            // Simulated: the shape and the width are what a run would have cost,
            // and there is no value to compute.
            let BlockBuf::Accounted { region, dtype, .. } = first else {
                return Err(Error::InvalidArgument(format!(
                    "iterative op {:?} was handed a mixture of real and simulated operands",
                    op.name()
                )));
            };
            self.counters()
                .add_resident(region.voxels() as u64 * dtype.size_of() as u64);
            return Ok(BlockBuf::Accounted {
                region: region.clone(),
                dtype: *dtype,
                uniform: None,
            });
        };
        let mut out = Voxels::zeros(arrays[0].dtype(), arrays[0].shape())?;
        op.substage(&crate::iterate::Substage::new(index, &arrays, at), &mut out)?;
        self.counters().add_resident(out.bytes());
        Ok(BlockBuf::Array(out))
    }

    /// Stage barrier: everything written to `image` is durable.
    fn finish(&self, image: usize) -> Result<()>;

    /// This image is dead: nothing will read it again, so free it.
    ///
    /// **The default is to keep it**, which is what every environment did before
    /// this existed. Freeing is an optimisation and forgetting to free costs
    /// space; freeing something still wanted costs an answer, so the safe
    /// default is the one that does nothing.
    ///
    /// Called by the executor when the phase that reads `image` has finished
    /// every one of its tasks, and only for an image the plan calls
    /// [`Visibility::Internal`](crate::decomposition::Visibility::Internal) and
    /// the caller has not pinned. An image is read by exactly one phase, so that
    /// moment is unambiguous.
    ///
    /// Reading a discarded image afterwards must **fail**, not return zeros. A
    /// freed image that reads as data is the same class of defect as an
    /// unwritten image that reads as zeros, and this crate fills those with NaN
    /// for the same reason.
    fn discard_image(&self, _image: usize) -> Result<()> {
        Ok(())
    }

    /// The same, and **which phase's completion freed it**.
    ///
    /// A separate method with a default that forwards, rather than a sixth
    /// argument to `discard_image`, so that every environment written before it
    /// existed keeps compiling and keeps behaving identically. What it buys is
    /// the diagnostic: "image 4 was discarded" is a fact a reader can do nothing
    /// with, and "image 4 was freed after phase 3, which the plan says is its
    /// last reader" tells them which `Hints::keep_images` entry they wanted. An
    /// environment that does not record the phase loses only the second half of
    /// that sentence.
    fn discard_image_after(&self, image: usize, _phase: usize) -> Result<()> {
        self.discard_image(image)
    }

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

/// One image's storage: **what it will hold, and whether it holds it yet.**
///
/// The declared element type and shape are known from the moment the
/// environment is built — that is what `prepare` checks and what a read of an
/// unwritten image is shaped by — but the buffer behind them is not allocated
/// until the first write.
///
/// Why, measured
/// -------------
/// An image costs `volume x sizeof(dtype)` bytes for as long as it is allocated,
/// and a plan says exactly how long that is: image `p + 1` is written by phase
/// `p` and freed after its last reader, which is what `discard_image` already
/// did at the far end. Allocating every image in the constructor made the *near*
/// end wrong in the same way the far end used to be, and the two are not
/// symmetric in cost: freeing late costs the tail of a chain, allocating early
/// costs **all of it at once, before the first phase runs**.
///
/// On the 12-phase, 13-image plan this was measured against — `404 x 1304 x
/// 3369`, 1.775 Gvoxel — the images' largest simultaneous total is 67.8 GiB and
/// their sum is 138.8 GiB. Constructing the environment paid the second figure,
/// and paid it in *touched* pages rather than reservations, because an unwritten
/// image is sentinel-filled rather than zeroed: `Voxels::unwritten` writes every
/// element. A 1/64 scale model of that plan measured 1.96 GiB of resident set
/// from the constructor alone, against a priced peak of 1.06 GiB — 1.85x, before
/// a phase had run. Deferring the allocation to the first write makes the
/// environment cost what the plan prices it at.
///
/// **No voxel moves.** A read of an image nobody has written yields
/// [`Voxels::unwritten`] over the read region, which is byte-for-byte the slice
/// it would have taken out of a fully sentinel-filled image; the first write
/// materialises the whole image at the sentinel and then assigns into it, which
/// is what a write into a fully sentinel-filled image always did.
struct ImageStore {
    /// What this image holds, whether or not it holds it yet. Kept beside the
    /// buffer rather than read off it, because the whole point is that there
    /// may be no buffer to read it off.
    dtype: Dtype,
    shape: [usize; 3],
    /// `None` before the first write, and again after a discard.
    data: Option<Voxels>,
}

impl ImageStore {
    /// An image nothing has written yet.
    ///
    /// Fallible for one reason and it is the constructor's: `Dtype::F16` has no
    /// buffer variant, and refusing it here — against a zero-element probe,
    /// which allocates nothing — keeps that refusal at the moment the
    /// environment is built rather than moving it to whichever write happened to
    /// come first.
    fn pending(dtype: Dtype, shape: [usize; 3]) -> Result<Self> {
        Voxels::zeros(dtype, [0, 0, 0])?;
        Ok(Self {
            dtype,
            shape,
            data: None,
        })
    }

    /// An image that already holds something: image 0, which is the input.
    fn held(data: Voxels) -> Self {
        Self {
            dtype: data.dtype(),
            shape: data.shape(),
            data: Some(data),
        }
    }

    /// Does this image hold a buffer? The measurable form of the deferral.
    fn is_allocated(&self) -> bool {
        self.data.is_some()
    }

    /// The whole image as it reads: the buffer, or the sentinel it would have
    /// been filled with.
    fn whole(&self) -> Result<Voxels> {
        match &self.data {
            Some(data) => Ok(data.clone()),
            None => Voxels::unwritten(self.dtype, self.shape),
        }
    }

    /// `region` of the image as it reads. The unwritten case allocates
    /// `region`'s worth and not the image's, which is the point.
    fn slice(&self, region: &Region) -> Result<Voxels> {
        match &self.data {
            Some(data) => data.slice_region(region),
            None => Voxels::unwritten(self.dtype, block_shape(region)?),
        }
    }

    /// The buffer to write into, allocated at the sentinel if this is the first
    /// write.
    fn buffer_mut(&mut self) -> Result<&mut Voxels> {
        if self.data.is_none() {
            self.data = Some(Voxels::unwritten(self.dtype, self.shape)?);
        }
        Ok(self.data.as_mut().expect("just allocated"))
    }

    /// Free the buffer, keeping what the image *is*. Reads are refused by the
    /// discard flag rather than by the absence, which is the arrangement
    /// `discard_image` already documented: the flag is what carries the meaning.
    fn release(&mut self) {
        self.data = None;
    }
}

/// Real volumes held in memory, one per image, **each at its own element type**
/// and each allocated when something first writes to it.
///
/// This is the geometry oracle's environment: small volumes, real values, so
/// identity ops must reproduce the input exactly and a window-sum op must
/// reproduce its whole-volume answer. It is deliberately *not* the streaming
/// path — that is `region_io`'s job, and a Zarr environment would go through
/// `ZarrRegionSink` because `zarrs` 0.23.13 loses data on concurrent partial
/// chunk writes.
///
/// Being resident is not the same as being resident *all at once*: an image's
/// lifetime runs from the phase that writes it to its last reader, and this
/// environment now holds an image over exactly that interval. See [`ImageStore`]
/// for the near end and [`Self::discard_image`] for the far one.
pub struct ArrayEnvironment {
    volume: [usize; 3],
    /// Every array, addressed by slot: `0..n_written` are the images the plan
    /// writes and the rest are the arrays the caller handed the run, in the
    /// order they were handed over.
    ///
    /// One vector rather than two because a supplied input is an image in every
    /// way that this environment cares about — it is read through `read`, it is
    /// priced by the same counters, it holds a shape and an element type — and
    /// the only thing that is different about it is its **address**, which
    /// [`Self::slot`] translates in one place. Two vectors would put the
    /// translation in every method instead.
    images: Vec<RwLock<ImageStore>>,
    /// How many of `images` the plan writes: image 0 plus one per phase.
    ///
    /// The boundary between the two address ranges, and what `output()` counts
    /// back from — it used to be `images.last()`, which stopped being the output
    /// the moment something was appended after it.
    n_written: usize,
    /// Set when an image has been discarded. Kept beside the images rather than
    /// inferred from a placeholder's shape, because "this was freed" and "this
    /// happens to be small" are different facts and only one of them is an
    /// error to read.
    discarded: Vec<AtomicBool>,
    /// The phase after whose completion each image was freed, or `usize::MAX`
    /// for an image nobody has freed or one freed through `discard_image`, which
    /// does not say.
    ///
    /// Beside `discarded` rather than folded into it because they answer
    /// different questions — "may I read this" and "who took it away" — and the
    /// first must stay a single relaxed load on a path the executor walks.
    freed_after: Vec<AtomicUsize>,
    chunk: [usize; 3],
    /// The arrays ops write beside their primary result, by declared name.
    ///
    /// Allocated on declaration rather than on first write, and NaN-filled on
    /// the same argument the images are: a voxel nobody wrote must be loud. The
    /// map is behind one lock because declaration is once-per-run and writes go
    /// through the inner lock; a `BTreeMap` rather than a `HashMap` so that
    /// iteration order is the declaration order a diagnostic reports in.
    ///
    /// Still `ArrayD<f64>`: a side output's **rank** is its own, which is the
    /// whole reason it is not an image, and its dtype is carried by the `Output`
    /// that declared it and charged by `write_side`.
    side: RwLock<BTreeMap<String, RwLock<ArrayD<f64>>>>,
    counters: EnvCounters,
    /// In-memory, to match the images: an environment whose volumes are not
    /// shared cannot offer shared fragments either, and pretending otherwise
    /// would make a single-node test pass for a reason a distributed run does
    /// not have.
    sidecars: Sidecars,
}

impl ArrayEnvironment {
    /// `n_phases` images are written; image 0 holds `input`.
    ///
    /// Every image gets `input`'s element type, which is right for the plans
    /// that keep one and is all this constructor can know. A plan whose phases
    /// change type — or shape — needs [`Self::for_decomposition`], which reads
    /// both off the plan.
    pub fn new(input: Voxels, n_phases: usize, chunk: [usize; 3]) -> Result<Self> {
        let volume = input.shape();
        let dtype = input.dtype();
        let mut images = Vec::with_capacity(n_phases + 1);
        images.push(RwLock::new(ImageStore::held(input)));
        for _ in 0..n_phases {
            images.push(RwLock::new(ImageStore::pending(dtype, volume)?));
        }
        Ok(Self {
            discarded: (0..images.len()).map(|_| AtomicBool::new(false)).collect(),
            freed_after: (0..images.len())
                .map(|_| AtomicUsize::new(usize::MAX))
                .collect(),
            volume,
            n_written: images.len(),
            images,
            chunk,
            side: RwLock::new(BTreeMap::new()),
            counters: EnvCounters::default(),
            sidecars: Sidecars::in_memory(),
        })
    }

    /// Images shaped **and typed** by the plan: image `p+1` gets phase `p`'s
    /// volume and phase `p`'s element type.
    ///
    /// [`Self::new`] gives every image the input's shape and type, which is
    /// right for every plan that keeps one, and is the only thing that could be
    /// done while a plan *had* one of each. A phase that changes either needs
    /// its output image allocated at what it writes, and the decomposition is
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
                "array environment: the input is {volume:?} and the decomposition reads image \
                 0 as {:?}",
                decomposition.volume
            )));
        }
        if input.dtype() != decomposition.dtype {
            return Err(Error::InvalidArgument(format!(
                "array environment: the input holds {} and the decomposition reads image 0 as \
                 {}",
                input.dtype().numpy_name(),
                decomposition.dtype.numpy_name()
            )));
        }
        let mut images = Vec::with_capacity(decomposition.n_images());
        images.push(RwLock::new(ImageStore::held(input)));
        for (index, phase) in decomposition.phases.iter().enumerate() {
            images.push(RwLock::new(ImageStore::pending(
                decomposition.dtype_at(index + 1),
                phase.volume(),
            )?));
        }
        Ok(Self {
            discarded: (0..images.len()).map(|_| AtomicBool::new(false)).collect(),
            freed_after: (0..images.len())
                .map(|_| AtomicUsize::new(usize::MAX))
                .collect(),
            volume,
            n_written: images.len(),
            images,
            chunk,
            side: RwLock::new(BTreeMap::new()),
            counters: EnvCounters::default(),
            sidecars: Sidecars::in_memory(),
        })
    }

    /// The plan's images, **and the arrays the run was handed besides its
    /// input**.
    ///
    /// `input` is image 0 and is what the first phase reads. `supplied` is
    /// everything else the run was given: `supplied[i]` is
    /// [`ImageId::supplied(i)`], and it is an image like any other — read through
    /// the same `Environment::read` at the reading block's own fetch region,
    /// charged the same bytes, named by a `Chain::Source` leaf or a
    /// `BlockOp::source_input` — with one difference, which is the whole
    /// distinction the [`ImageKind`] split records: **no phase writes it and
    /// nothing can rebuild it**, so it is never freed and a write to it is
    /// refused by name.
    ///
    /// Every one of them must be in image 0's shape, because a source leaf has
    /// reach 0 and is handed the reading block's fetch region: across shapes the
    /// same integers would name different voxels. Their element types are their
    /// own, and each is checked against what the plan's readers declared.
    ///
    /// [`ImageId::supplied(i)`]: crate::assemble::ImageId::supplied
    /// [`ImageKind`]: crate::decomposition::ImageKind
    pub fn with_inputs(
        input: Voxels,
        supplied: Vec<Voxels>,
        decomposition: &Decomposition,
        chunk: [usize; 3],
    ) -> Result<Self> {
        let mut env = Self::for_decomposition(input, decomposition, chunk)?;
        let wanted = decomposition.supplied_input_images();
        if supplied.len() != wanted.len() {
            return Err(Error::InvalidArgument(format!(
                "array environment was handed {} array(s) besides its input and the plan reads                  {}: {}. An array nothing reads is not an image of this plan, and an image nothing                  supplied has nothing to be fetched from.",
                supplied.len(),
                wanted.len(),
                wanted
                    .iter()
                    .map(|&image| describe_image(image))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        for (which, array) in supplied.into_iter().enumerate() {
            let image = ImageId::supplied(which).index();
            if !wanted.contains(&image) {
                return Err(Error::InvalidArgument(format!(
                    "array environment was handed {} and the plan does not read it; it reads {}.                      Supplied inputs are addressed from zero in the order they are handed over,                      so a gap in the list moves every array after it.",
                    describe_image(image),
                    wanted
                        .iter()
                        .map(|&image| describe_image(image))
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
            let held = array.shape();
            let volume = decomposition.volume_at(image);
            if held != volume {
                return Err(Error::InvalidArgument(format!(
                    "array environment: {} is {held:?} and the plan reads it as {volume:?}. A                      supplied input is read at the reading block's own fetch region, so it is in                      image 0's coordinate space and no other.",
                    describe_image(image)
                )));
            }
            let held = array.dtype();
            let wants = decomposition.dtype_at(image);
            if held != wants {
                return Err(Error::InvalidArgument(format!(
                    "array environment: {} holds {} and the plan's readers declare {}. Nothing                      folds an element type for an array no phase wrote, so the declaration is                      the only statement there is and it has to be the array's own.",
                    describe_image(image),
                    held.numpy_name(),
                    wants.numpy_name()
                )));
            }
            env.images.push(RwLock::new(ImageStore::held(array)));
            env.discarded.push(AtomicBool::new(false));
            env.freed_after.push(AtomicUsize::new(usize::MAX));
        }
        Ok(env)
    }

    /// Where `image` sits in `self.images`.
    ///
    /// The one place the two address ranges meet. An image the plan writes is at
    /// its own number; a supplied input is at `n_written` plus its index, which
    /// is the order it was handed over in.
    fn slot(&self, image: usize) -> usize {
        match ImageId::from(image).supplied_index() {
            Some(which) => self.n_written + which,
            None => image,
        }
    }

    /// Whether `image` has been freed.
    pub fn is_discarded(&self, image: usize) -> bool {
        self.discarded
            .get(self.slot(image))
            .map(|flag| flag.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    /// How many images still hold their data.
    ///
    /// The measurable form of the whole point: a twenty-phase chain used to end
    /// with twenty-one of these and now ends with three.
    ///
    /// **"Still", and therefore about discarding only.** An image nothing has
    /// written yet is counted here, because this answers "what has the run given
    /// back" and an unwritten image was never taken. What is actually allocated
    /// at this moment is [`Self::allocated_images`], and the two differ during a
    /// run for exactly the length of the chain that has not run yet.
    pub fn resident_images(&self) -> usize {
        (0..self.images.len())
            .filter(|&image| !self.is_discarded(image))
            .count()
    }

    /// How many images hold a buffer **right now**.
    ///
    /// The near end of an image's lifetime, as [`Self::resident_images`] is the
    /// far end: an image is allocated by its first write and freed by its
    /// discard, so before the run this is 1 — the input — however many phases
    /// the plan has. That is the difference this environment used to pay in
    /// full: see [`ImageStore`] for the 138.8 GiB against 67.8 GiB it was worth
    /// on the plan it was measured against.
    pub fn allocated_images(&self) -> usize {
        (0..self.images.len())
            .filter(|&image| self.image_guard(image).is_allocated())
            .count()
    }

    /// Bytes the images hold **right now**: what a resident environment costs at
    /// this moment, stated the way a residency budget is.
    ///
    /// Only allocated images count, because only they occupy anything. An image
    /// the plan has not reached yet contributes nothing, which is the whole
    /// claim and is why it is reported rather than argued.
    pub fn allocated_image_bytes(&self) -> u64 {
        (0..self.images.len())
            .filter_map(|image| {
                let guard = self.image_guard(image);
                guard.data.as_ref().map(|data| data.bytes())
            })
            .sum()
    }

    /// Which phase's completion freed `image`, when that was recorded.
    ///
    /// `None` for a live image and for one freed through `discard_image`, which
    /// carries no phase; the two are distinguished by [`Self::is_discarded`].
    pub fn freed_after(&self, image: usize) -> Option<usize> {
        let phase = self
            .freed_after
            .get(self.slot(image))?
            .load(Ordering::SeqCst);
        (phase != usize::MAX).then_some(phase)
    }

    fn refuse_if_discarded(&self, image: usize, what: &str) -> Result<()> {
        if !self.is_discarded(image) {
            return Ok(());
        }
        // The phase is named where it is known, because it is the whole of what
        // a reader needs: `keep_images` takes an image number, and the question
        // that gets somebody there is "who freed it and why did the plan think
        // it could". Without it the message says only that something happened.
        let freed = match self.freed_after(image) {
            Some(phase) => format!(
                "was discarded after phase {phase}, whose completion the plan says is the last \
                 read of it"
            ),
            None => "was discarded".to_string(),
        };
        Err(Error::InvalidArgument(format!(
            "{what}: {} {freed}, so the plan says nothing wants it again. Pin it with \
             `Hints::keep_images` if something does.",
            describe_image(image)
        )))
    }

    fn image_guard(&self, image: usize) -> std::sync::RwLockReadGuard<'_, ImageStore> {
        self.images[self.slot(image)]
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The shape of one image, which is not `volume()` once a phase changes it.
    ///
    /// **Declared**, so it is the same answer before and after the image is
    /// written, and after it is discarded. It used to be read off the buffer,
    /// which meant a discarded image reported the one-voxel placeholder that
    /// stood in for it; the flag is what says an image is gone
    /// ([`Self::is_discarded`]), and this now says only what the image is.
    pub fn image_shape(&self, image: usize) -> [usize; 3] {
        self.image_guard(image).shape
    }

    /// The element type of one image, which is not the workflow's once a phase
    /// changes it. Declared, on [`Self::image_shape`]'s argument.
    pub fn image_dtype(&self, image: usize) -> Dtype {
        self.image_guard(image).dtype
    }

    /// The last image: the workflow output.
    ///
    /// A workflow whose last phase never ran hands back the unwritten sentinel
    /// over the whole image, which is what a caller reading an unwritten output
    /// always got.
    pub fn output(&self) -> Voxels {
        self.images[self.n_written - 1]
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .whole()
            .expect("an image's element type was checked when it was declared")
    }

    /// One image's data, or the refusal that names why it is not there.
    ///
    /// **The checked form, and the one to use.** A discarded image holds a
    /// one-voxel placeholder — see [`Self::discard_image`] — so reading one
    /// through [`Self::image`] used to hand back a `1 x 1 x 1` array that then
    /// failed, somewhere else, as a shape mismatch against the volume. That is
    /// the same class of defect as a freed image reading as zeros: the error
    /// names a symptom, and the fact worth reporting — this image was freed, by
    /// this phase, and `Hints::keep_images` is how to keep it — is nowhere in
    /// it.
    pub fn try_image(&self, image: usize) -> Result<Voxels> {
        self.refuse_if_discarded(image, "image")?;
        self.image_guard(image).whole()
    }

    /// One image's data.
    ///
    /// **Panics on a discarded image**, with [`Self::try_image`]'s message. The
    /// signature cannot become a `Result` without changing every caller of it,
    /// and handing back the placeholder is the one behaviour that is definitely
    /// wrong — so the accessor keeps its shape and states its precondition the
    /// way an index does. A caller who does not know whether an image survived
    /// the run is asking a question, and [`Self::try_image`] is the form that
    /// answers it.
    pub fn image(&self, image: usize) -> Voxels {
        self.try_image(image)
            .unwrap_or_else(|refusal| panic!("{refusal}"))
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
        if decomposition.n_phases() + 1 != self.n_written {
            return Err(Error::InvalidArgument(format!(
                "array environment built for {} phases, decomposition has {}",
                self.n_written - 1,
                decomposition.n_phases()
            )));
        }
        let supplied = decomposition.supplied_input_images();
        if supplied.len() != self.images.len() - self.n_written {
            return Err(Error::InvalidArgument(format!(
                "array environment holds {} supplied input(s) and the decomposition reads {}. \
                 `ArrayEnvironment::with_inputs` is what seeds them.",
                self.images.len() - self.n_written,
                supplied.len()
            )));
        }
        // Per image, because a phase owns its volume and its element type. This
        // is the check that used to be `Decomposition::check` refusing the plan
        // outright: an environment which allocated one shape for every image
        // cannot host a phase that changes shape, and that is a fact about the
        // environment, not about the plan. `for_decomposition` builds one that
        // can.
        for image in (0..self.n_written).chain(supplied.iter().copied()) {
            let held = self.image_shape(image);
            let wanted = decomposition.volume_at(image);
            if held != wanted {
                return Err(Error::InvalidArgument(format!(
                    "array environment holds {} as {held:?} and the decomposition writes it \
                     as {wanted:?}. `ArrayEnvironment::for_decomposition` allocates each image \
                     at the shape its phase gives it.",
                    describe_image(image)
                )));
            }
            let held = self.image_dtype(image);
            let wanted = decomposition.dtype_at(image);
            if held != wanted {
                return Err(Error::InvalidArgument(format!(
                    "array environment holds {} as {} and the decomposition writes it as {}. \
                     `ArrayEnvironment::for_decomposition` allocates each image at the element \
                     type its phase gives it.",
                    describe_image(image),
                    held.numpy_name(),
                    wanted.numpy_name()
                )));
            }
        }
        // **`check_chunk_exclusive_writes` is deliberately not called here**,
        // and its absence is a decision rather than an oversight. That invariant
        // exists for two storage reasons — a store that loses bytes when two
        // tasks read-modify-write one chunk, and a cache that cannot give a
        // shared chunk a lifetime — and this environment has neither. Its
        // `chunk` is an accounting fiction, used by `chunks_touched` to price IO
        // that is not happening; an image here is one `Array3` and a write is a
        // slice assignment into disjoint indices, which no chunk grid mediates.
        // Enforcing it would refuse plans that are perfectly well defined in
        // memory, for a hazard that is not present, and would make the in-memory
        // oracle less general than the storage it is the oracle for.
        Ok(())
    }

    fn read(&self, image: usize, region: &Region) -> Result<BlockBuf> {
        if self.slot(image) >= self.images.len() {
            return Err(Error::InvalidArgument(format!(
                "block-op read: this environment does not hold {}. It holds {} image(s) the plan \
                 writes and {} supplied input(s).",
                describe_image(image),
                self.n_written,
                self.images.len() - self.n_written
            )));
        }
        self.refuse_if_discarded(image, "block-op read")?;
        region_within(region, &self.image_shape(image), "block-op read")?;
        let array = self.image_guard(image).slice(region)?;
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

    fn apply(
        &self,
        slot: &Chain,
        input: &BlockBuf,
        sources: &[(usize, BlockBuf)],
        at: &Placement,
    ) -> Result<BlockBuf> {
        let array = input.as_array()?;
        let stored = as_source_arrays(sources)?;
        // Allocated from what the chain **declares**, not from what it was
        // handed. That one line is the difference between a phase that may
        // translate its read and a phase that may resize it.
        let mut out = Voxels::zeros(
            slot.produces(array.dtype())?,
            slot.placed_output_shape(array.shape(), at)?,
        )?;
        slot.apply_placed(array, SourceInputs::new(&stored), &mut out, at)?;
        self.counters.ops_applied.fetch_add(1, Ordering::SeqCst);
        self.counters.estimated_work.fetch_add(
            (array.len() as f64 * slot.cost_per_voxel()) as u64,
            Ordering::SeqCst,
        );
        // The caller releases `input`; releasing it here too would double-count.
        self.counters.add_resident(out.bytes());
        Ok(BlockBuf::Array(out))
    }

    fn write(&self, image: usize, within: &Region, valid: &Region, buf: &BlockBuf) -> Result<()> {
        if is_supplied_image(image) {
            return Err(Error::InvalidArgument(format!(
                "block-op write: {} was handed to the run. An input is not the run's to \
                 overwrite, and a plan that wrote one would change what a re-run reads.",
                describe_image(image)
            )));
        }
        self.refuse_if_discarded(image, "block-op write")?;
        let array = buf.as_array()?;
        region_within(valid, &self.image_shape(image), "block-op write")?;
        if valid.voxels() == 0 {
            return Ok(());
        }
        let source = array.slice_region(within)?;
        let mut guard = self.images[self.slot(image)]
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // The first write to an image is what allocates it, sentinel-filled — so
        // a partial write leaves the rest unwritten exactly as it did when every
        // image was allocated in the constructor.
        guard.buffer_mut()?.assign_region(valid, &source)?;
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

    fn finish(&self, _image: usize) -> Result<()> {
        Ok(())
    }

    /// Free the array and remember that it was freed.
    ///
    /// There is no placeholder any more, and there does not need to be one: a
    /// image knows its own element type and shape whether or not it holds a
    /// buffer ([`ImageStore`]), so freeing is dropping the buffer, and the flag
    /// carries the meaning as it always did. What that changes for a reader is
    /// that [`Self::image_shape`] on a discarded image now reports the shape the
    /// image *is* rather than the one-voxel stand-in — reads and writes are
    /// still refused by name, which is the behaviour anything depends on.
    fn discard_image(&self, image: usize) -> Result<()> {
        // **The one thing the three-kind split is for.** An intermediate may be
        // dropped and rebuilt by re-running the phase that wrote it; an input
        // cannot be rebuilt at any price, because no phase produces it. Freeing
        // one would be an image that is gone and unrecoverable, so it is refused
        // here rather than left to `image_visibility` to avoid asking for.
        if is_supplied_image(image) {
            return Err(Error::InvalidArgument(format!(
                "cannot discard {}: it was handed to the run and no phase writes it, so nothing \
                 could rebuild it. Only an image the run produces may be freed.",
                describe_image(image)
            )));
        }
        let Some(flag) = self.discarded.get(self.slot(image)) else {
            return Err(Error::InvalidArgument(format!(
                "cannot discard image {image}; this environment holds {}",
                self.n_written
            )));
        };
        if flag.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        self.images[self.slot(image)]
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .release();
        Ok(())
    }

    /// The same, recording the phase so that a later reader can be told who
    /// took the image away rather than only that somebody did.
    ///
    /// Recorded **before** the discard, so that an image which is observably
    /// discarded is never observably discarded-by-nobody: the flag is what a
    /// concurrent reader tests, and the phase must already be there when it
    /// flips.
    fn discard_image_after(&self, image: usize, phase: usize) -> Result<()> {
        if let Some(slot) = self.freed_after.get(self.slot(image)) {
            slot.store(phase, Ordering::SeqCst);
        }
        self.discard_image(image)
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

/// `region`, restated in `holds`'s own coordinates.
///
/// Both are in the volume's coordinates; a buffer holding `holds` indexes from
/// its own lower corner, and this is the one place that subtraction happens. It
/// fails rather than saturating: a region outside the buffer is a caller that has
/// the wrong buffer, and a clamped answer would be a well-formed wrong one.
pub(crate) fn relative(region: &Region, holds: &Region, what: &str) -> Result<Region> {
    if region.ndim() != holds.ndim() {
        return Err(Error::InvalidArgument(format!(
            "{what}: a region of rank {} against a buffer of rank {}",
            region.ndim(),
            holds.ndim()
        )));
    }
    let mut start = vec![0usize; region.ndim()];
    for axis in 0..region.ndim() {
        let lo = region.start[axis]
            .checked_sub(holds.start[axis])
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "{what}: {:?}+{:?} starts before the buffer, which holds {:?}+{:?}",
                    region.start, region.shape, holds.start, holds.shape
                ))
            })?;
        if lo + region.shape[axis] > holds.shape[axis] {
            return Err(Error::InvalidArgument(format!(
                "{what}: {:?}+{:?} leaves the buffer, which holds {:?}+{:?}",
                region.start, region.shape, holds.start, holds.shape
            )));
        }
        start[axis] = lo;
    }
    Ok(Region::new(&start, &region.shape))
}

/// An image region's extent as the rank-3 array it is, or an error saying so.
pub(crate) fn block_shape(region: &Region) -> Result<[usize; 3]> {
    if region.shape.len() != 3 {
        return Err(Error::InvalidArgument(format!(
            "an image's blocks are 3-D, got a region of rank {}",
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
        // One volume, for every image: this environment counts regions against a
        // single extent and has no per-image shape to check a read against. A
        // plan whose phases change shape is refused *here*, by the environment
        // that cannot host it, rather than by the plan's own guard.
        let uniform = decomposition.uniform_volume().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "accounting environment has one volume {:?}, and this decomposition changes \
                 shape between images: {:?}",
                self.volume,
                (0..decomposition.n_images())
                    .map(|image| decomposition.volume_at(image))
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

    fn read(&self, _image: usize, region: &Region) -> Result<BlockBuf> {
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

    /// **`sources` is ignored, and that is the honest answer here.** A simulated
    /// run holds no data, so a stored operand has nothing to contribute to a
    /// result that is itself an extent and an element type. Its *cost* is not
    /// lost: the executor reads it through [`Environment::read`], which is where
    /// this environment counts every byte it is asked for, and where a source
    /// arm's reads have to be counted for the simulation to be of the right
    /// plan.
    fn apply(
        &self,
        slot: &Chain,
        input: &BlockBuf,
        _sources: &[(usize, BlockBuf)],
        at: &Placement,
    ) -> Result<BlockBuf> {
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
        let produced = slot.placed_output_shape(block_shape(region)?, at)?;
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

    fn write(&self, _image: usize, _within: &Region, valid: &Region, buf: &BlockBuf) -> Result<()> {
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

    fn finish(&self, _image: usize) -> Result<()> {
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
                &[],
                &Placement::same(Anchor::whole([4, 4, 4])),
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
            &[],
            &Placement::same(Anchor::whole([2094, 13316, 3369])),
        )
        .unwrap();
    }

    /// The measurement the element-type change exists for, taken through the
    /// environment's own counter rather than argued: the identical chain over
    /// the identical extent moves an eighth of the bytes.
    #[test]
    fn a_narrow_image_reads_an_eighth_of_the_bytes_a_wide_one_does() {
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

    /// **The negative control for the representation.** Every element type's own
    /// extremes, through an image and back out, compared bit for bit.
    ///
    /// An image that went through `f64` on its way in or out would pass every
    /// fixture that never reaches an extreme and fail exactly here:
    /// `u64::MAX` and `i64::MIN + 1` are not representable in an `f64` and would
    /// come back rounded, `-0.0` would come back as `0.0` under any comparison
    /// that used `==` on the way, and a NaN would come back a different NaN — or
    /// as a zero, if anything on the path had used `f64::min`/`max` rather than
    /// `total_cmp`. So the assertion is on the *bits* for the float types, which
    /// is the only comparison that can see any of that.
    #[test]
    fn every_element_type_round_trips_its_extremes_through_an_image() {
        /// One column of values in, the same column out.
        macro_rules! round_trip {
            ($type:ty, $values:expr, $eq:expr) => {{
                let values: Vec<$type> = $values;
                let extent = [values.len(), 1, 1];
                let input =
                    Array3::<$type>::from_shape_vec((values.len(), 1, 1), values.clone()).unwrap();
                let env = ArrayEnvironment::new(Voxels::from(input), 1, [1, 1, 1]).unwrap();
                let region = Region::whole(&extent);
                let buf = env.read(0, &region).unwrap();
                env.write(1, &Region::new(&[0, 0, 0], &extent), &region, &buf)
                    .unwrap();
                let out = env.output();
                assert_eq!(out.dtype(), <$type as crate::voxels::VoxelElement>::DTYPE);
                let got = out.view::<$type>().unwrap();
                for (index, wanted) in values.iter().enumerate() {
                    let held = got[[index, 0, 0]];
                    let same: bool = $eq(held, *wanted);
                    assert!(
                        same,
                        "{} lost {:?} at index {index}: got {:?}",
                        <$type as crate::voxels::VoxelElement>::DTYPE.numpy_name(),
                        wanted,
                        held
                    );
                }
            }};
        }
        round_trip!(bool, vec![true, false, true], |held, wanted| held == wanted);
        round_trip!(u8, vec![0, 1, u8::MAX], |held, wanted| held == wanted);
        round_trip!(u16, vec![0, 1, u16::MAX, u16::MAX - 1], |held, wanted| held
            == wanted);
        round_trip!(u32, vec![0, 1, u32::MAX, u32::MAX - 1], |held, wanted| held
            == wanted);
        // Beyond `f64`'s 53 bits of mantissa: `u64::MAX as f64 as u64` saturates
        // and `(u64::MAX - 1) as f64 as u64` does too, so an image that had been
        // through an `f64` could not tell these three apart.
        round_trip!(
            u64,
            vec![0, 1, u64::MAX, u64::MAX - 1, (1u64 << 53) + 1],
            |held, wanted| held == wanted
        );
        round_trip!(i8, vec![i8::MIN, -1, 0, i8::MAX], |held, wanted| held
            == wanted);
        round_trip!(i16, vec![i16::MIN, -1, 0, i16::MAX], |held, wanted| held
            == wanted);
        round_trip!(i32, vec![i32::MIN, -1, 0, i32::MAX], |held, wanted| held
            == wanted);
        round_trip!(
            i64,
            vec![i64::MIN, i64::MIN + 1, -1, 0, i64::MAX, i64::MAX - 1],
            |held, wanted| held == wanted
        );
        // Bit equality, so `-0.0` is not `0.0` and a NaN is the NaN it was.
        round_trip!(
            f32,
            vec![
                -0.0,
                0.0,
                f32::NAN,
                -f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::MIN,
                f32::MAX,
                f32::MIN_POSITIVE,
                f32::EPSILON,
            ],
            |held: f32, wanted: f32| held.to_bits() == wanted.to_bits()
        );
        round_trip!(
            f64,
            vec![
                -0.0,
                0.0,
                f64::NAN,
                -f64::NAN,
                f64::from_bits(0x7ff8_0000_dead_beef),
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::MIN,
                f64::MAX,
                f64::MIN_POSITIVE,
                f64::EPSILON,
            ],
            |held: f64, wanted: f64| held.to_bits() == wanted.to_bits()
        );
    }

    /// The control above, inverted: the comparison it uses can *see* a lost
    /// extreme. Without this the round trip would pass just as well against a
    /// comparison that could not fail.
    #[test]
    fn the_round_trip_control_would_notice_a_value_that_went_through_an_f64() {
        // `u64::MAX` itself is not the witness: Rust's float-to-integer cast
        // saturates, so it comes back unchanged for the wrong reason. Its
        // neighbour is the witness, and so is the first integer above the
        // mantissa.
        assert_ne!(
            u64::MAX - 1,
            (u64::MAX - 1) as f64 as u64,
            "`u64::MAX - 1` survives a trip through `f64`, so the control above proves nothing"
        );
        assert_ne!(
            (1u64 << 53) + 1,
            ((1u64 << 53) + 1) as f64 as u64,
            "the first integer above `f64`'s mantissa survives a trip through it"
        );
        assert_ne!(
            i64::MIN + 1,
            (i64::MIN + 1) as f64 as i64,
            "`i64::MIN + 1` survives a trip through `f64`"
        );
        assert_ne!(
            (-0.0f64).to_bits(),
            0.0f64.to_bits(),
            "the float comparison cannot tell `-0.0` from `0.0`"
        );
        assert_ne!(
            f64::NAN.to_bits(),
            f64::from_bits(0x7ff8_0000_dead_beef).to_bits(),
            "the float comparison cannot tell two NaNs apart"
        );
    }

    /// **An image is allocated by its first write, not by the constructor.**
    ///
    /// The measurement in [`ImageStore`]'s header, as a fact about one
    /// environment rather than about a run: an eight-phase plan holds one image
    /// before anything has run, where it used to hold nine.
    #[test]
    fn an_image_costs_nothing_until_something_writes_to_it() {
        let extent = [8, 8, 8];
        let phases = 8;
        let input = Voxels::zeros(Dtype::F64, extent).unwrap();
        let env = ArrayEnvironment::new(input, phases, [8, 8, 8]).unwrap();

        assert_eq!(env.allocated_images(), 1, "only the input");
        assert_eq!(
            env.allocated_image_bytes(),
            (8 * 8 * 8 * 8) as u64,
            "one f64 image's worth"
        );
        // …and the images are all there, at their declared shape and type.
        assert_eq!(env.resident_images(), phases + 1);
        for image in 0..=phases {
            assert_eq!(env.image_shape(image), extent);
            assert_eq!(env.image_dtype(image), Dtype::F64);
        }

        // The liveness half: writing is what allocates, so writing everything
        // reaches the figure the constructor used to start at.
        let region = Region::whole(&extent);
        let buf = env.read(0, &region).unwrap();
        for image in 1..=phases {
            env.write(image, &region, &region, &buf).unwrap();
        }
        assert_eq!(env.allocated_images(), phases + 1);
        assert_eq!(
            env.allocated_image_bytes(),
            (phases as u64 + 1) * (8 * 8 * 8 * 8)
        );

        // …and discarding gives it back, which is the far end of the same
        // lifetime.
        env.discard_image(1).unwrap();
        assert_eq!(env.allocated_images(), phases);
        assert_eq!(env.resident_images(), phases);
    }

    /// **Deferring the allocation moves no voxel.** An image nobody has written
    /// reads exactly what a sentinel-filled image would have handed back — which
    /// is what it *was*, before the allocation was deferred.
    #[test]
    fn an_unwritten_image_reads_as_the_sentinel_it_would_have_been_filled_with() {
        let extent = [4, 4, 4];
        let region = Region::new(&[1, 1, 1], &[2, 2, 2]);
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
            Dtype::F32,
            Dtype::F64,
        ] {
            let env =
                ArrayEnvironment::new(Voxels::zeros(dtype, extent).unwrap(), 1, extent).unwrap();
            let whole = Voxels::unwritten(dtype, extent).unwrap();

            let read = env.read(1, &region).unwrap();
            let read = read.as_array().unwrap();
            let cut = whole.slice_region(&region).unwrap();
            assert_eq!(read.dtype(), dtype);
            assert_eq!(read.shape(), [2, 2, 2]);
            let held = env.try_image(1).unwrap();
            assert_eq!(held.dtype(), dtype);
            assert_eq!(held.shape(), extent);
            // `NaN != NaN`, so the float types are compared through the property
            // the sentinel is *defined* by rather than through equality — the
            // same reason `Voxels::uniform` says a NaN-filled image is not
            // uniform.
            match dtype {
                Dtype::F32 => {
                    assert!(read.view::<f32>().unwrap().iter().all(|v| v.is_nan()));
                    assert!(held.view::<f32>().unwrap().iter().all(|v| v.is_nan()));
                }
                Dtype::F64 => {
                    assert!(read.view::<f64>().unwrap().iter().all(|v| v.is_nan()));
                    assert!(held.view::<f64>().unwrap().iter().all(|v| v.is_nan()));
                }
                _ => {
                    assert_eq!(
                        read, &cut,
                        "{dtype:?}: a partial read of an unwritten image"
                    );
                    assert_eq!(held, whole, "{dtype:?}: the whole unwritten image");
                }
            }
        }
    }

    /// A partial write allocates the image and leaves the rest unwritten, which
    /// is what a write into a fully sentinel-filled image always did. The
    /// deferral must not turn "nobody wrote this corner" into a zero.
    #[test]
    fn a_partial_write_leaves_the_rest_of_a_deferred_image_unwritten() {
        let extent = [4, 4, 4];
        let env =
            ArrayEnvironment::new(Voxels::filled(Dtype::U16, extent, 7.0).unwrap(), 1, extent)
                .unwrap();
        let corner = Region::new(&[0, 0, 0], &[2, 2, 2]);
        let buf = env.read(0, &corner).unwrap();
        env.write(1, &Region::new(&[0, 0, 0], &[2, 2, 2]), &corner, &buf)
            .unwrap();

        let image = env.try_image(1).unwrap();
        let view = image.view::<u16>().unwrap();
        assert_eq!(view[[0, 0, 0]], 7, "what was written");
        assert_eq!(
            view[[3, 3, 3]],
            u16::MAX,
            "what nobody wrote keeps the sentinel rather than becoming a zero"
        );
    }

    /// An image allocated at one type and written at another is refused, rather
    /// than silently converted. This is the guard `prepare` gained when an image
    /// stopped being `f64` by definition.
    #[test]
    fn writing_an_image_in_the_wrong_element_type_is_refused_by_name() {
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
