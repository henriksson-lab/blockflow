// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Ops whose input or output is **not a pixel region**.
//
// Why this is a second trait and not a wider `BlockOp`
// ---------------------------------------------------
// `BlockOp::apply(input, out, at)` is `region -> region`: one buffer in, one
// buffer of the same shape out. Three useful shapes do not fit it.
//
// * `volume -> fragments`. A block reads pixels and produces a *summary* — a
//   handful of bytes whose size has nothing to do with the block's voxel count.
//   There is no `out` buffer to write.
// * `fragments -> fragments`. A block reads its own fragment, and where it says
//   so its neighbours' fragments, and produces another fragment. There is no
//   pixel array involved **at either end**.
// * `fragments -> volume`. The reduction, written back as pixels. Declared with
//   a reach of the whole volume, which is not a special case in the geometry —
//   see "The global step is a full-reach phase" below.
//
// Widening `apply` to cover them would put an environment handle, a stream list
// and an optional output buffer on the one signature every ordinary op
// implements, to serve ops that are a different shape. So this file adds
// [`FragmentOp`] beside `BlockOp` and changes neither `BlockOp` nor `Chain`.
//
// The one thing the two kinds share is the **decomposition**: the same blocks,
// the same lattice, the same task DAG. That is deliberate and is what makes a
// fragment phase schedulable by the executor that already exists rather than by
// a second one.
//
// How a fragment reach becomes a dependency
// -----------------------------------------
// A fragment op may declare a reach *in blocks*: "to compute block b's output I
// need blocks b-1 .. b+1 of stream s". The executor's DAG is built from
// **read-extent overlap** (`graph.rs`), so the way to make that reach produce
// the right dependencies is to widen the phase's read extent by the same
// neighbourhood, in voxels: `halo = reach * block_edge`. Then
//
// * `TaskGraph::build` gives the phase's task for block b exactly the previous
//   phase's tasks for blocks b-1 .. b+1 as dependencies, single-node and
//   distributed alike, with no new scheduling code at all;
// * `Decomposition::check` still passes when the op's own reach is smaller than
//   the halo, because `valid == core` and the valid regions still tile.
//
// The widened read extent is a scheduling statement, not a promise to read
// pixels: an op that declares `reads_pixels() == false` does no pixel IO at all
// and the executor performs none on its behalf.
//
// The global step is a full-reach phase, not a fan-in
// ---------------------------------------------------
// Reducing a whole stream to one answer looks like it needs a node the task DAG
// does not have. It does not. It is an op that reads everything, and the
// existing geometry already says what that means (`geometry.rs:185-244`):
//
// * `trust_lo = read_lo + reach` unless the read starts at 0, and
//   `trust_hi = read_hi - reach` unless it ends at the volume;
// * so with `reach == volume`, any block whose read extent is not the whole
//   axis has `lo >= hi`, collapses to a zero-extent valid region, and the
//   tiling check reports the hole. **A short halo on a global op fails loudly**,
//   which is the property this crate exists for;
// * the only configuration that survives is `halo == volume` — every block
//   reads everything, `valid == core`, and the blocks tile exactly. Correct, and
//   N-fold redundant, so the cost model drives the planner to one block.
//
// What follows from that, and is worth saying out loud: **a full-reach phase is
// a reboot.** Nothing fuses across it, the infinite-grid costing assumption in
// `price_phase` stops holding because the per-block cost is no longer local, and
// no cache state survives it.
//
// The guard, on the side the output is actually on
// ------------------------------------------------
// `Decomposition::check` and the executor's post-run check assert that a
// phase's *valid regions* tile the volume. For a phase whose output is
// fragments that assertion is about an image nobody wrote: `valid == core` by
// construction, cores tile by construction, and the check passes without
// constraining the fragments at all. A guard that cannot fail is worse than no
// guard, because it is trusted.
//
// So a fragment stream declares its [`Coverage`], with no default, and
// [`check_fragment_coverage`] runs after the phase's last task — against the
// *store*, so that what is checked is what landed. A phase that writes no pixel
// image and declares no every-block stream is refused at plan time, because
// nothing about it would be checkable at all.
//
// What is still the caller's
// --------------------------
// Deciding what the bytes mean, and choosing whether the reduction runs as a
// full-reach phase or as a plain loop after the run. [`fold_fragments`] serves
// the second, and it *streams*: one fragment resident at a time, by key.
// `Environment::sidecar_fragments` materialises a whole stream and is the wrong
// tool for anything at scale.
//
// The second array, and where the operand is needed
// -------------------------------------------------
// A `volume -> fragments` op summarising one array against *another* — per
// region of a label array, a quantity read out of an intensity array — had no
// way to say so: a fragment op read the one image it was handed. `BlockOp`
// already had the shape for it ([`BlockOp::source_inputs`] plus
// [`BlockOp::apply_with`]), so this file takes that shape rather than inventing
// a second one, down to the defaulted `apply_with` **erroring** when an operand
// was declared. `env.rs`' argument for that is the whole reason it exists:
// *silently ignoring an operand is the precise shape of the wrong answer this
// change exists to remove.*
//
// **The emit path needs it; the merge path does not, and is not refused one.**
// A `volume -> fragments` op reads pixels, so a second pixel array is the same
// kind of thing at the same block extent. A `fragments -> fragments` merge has
// no pixel array at either end — that is what `reads_pixels() == false` buys —
// so it has nothing to combine an operand against. It is not *forbidden* one,
// because the two declarations are independent (`reads_pixels` is about image
// `p`, `source_inputs` is about every other image, and each is read and counted
// on its own), and a merge that genuinely wants to look at a stored array can
// say so without also paying for an image it does not want. What it may not do
// is take one silently.
//
// The seam, which is the part that is not plumbing
// ------------------------------------------------
// A per-region reduction over a second array *straddles seams*: a region cut by
// a block boundary is summed in pieces and the pieces are added in the merge.
// If the accumulator is `f64`, that addition does not associate, so the answer
// depends on the order the blocks merged in — and a plan cut differently gives a
// different number. `ops/detect.rs` deferred a weighted centroid on exactly
// this, and named the honest answers: a fixed-point accumulator, or a stated
// tolerance.
//
// This file does not pick one. What it removes is the *silence*: [`SeamFold`]
// has no default, an op that reads a second array (or that folds fragments
// derived from one) must state which of the three it is, and the one that claims
// order-independence is **checked** rather than believed — the executor applies
// the block a second time with the neighbourhood reversed and requires the same
// bytes. See [`SeamFold`] for what each variant forbids.
//
// [`BlockOp::source_inputs`]: crate::op::BlockOp::source_inputs
// [`BlockOp::apply_with`]: crate::op::BlockOp::apply_with

use std::collections::{BTreeMap, BTreeSet};

use crate::assemble::{describe_image, is_supplied_image};
use crate::decomposition::{Decomposition, PhaseDecomposition};
use crate::dtype::Dtype;
use crate::env::{BlockBuf, Environment};
use crate::error::{Error, Result};
use crate::geometry::BlockGrid;
use crate::op::{Anchor, SourceInput};
use crate::region::Region;
use crate::sidecar::{check_stream_name, FragmentKey, Lifecycle};

/// A fragment stream an op reads, and how far it reaches for it.
///
/// `phase` is part of the address, not an implementation detail: fragments are
/// keyed `(stream, phase, block)`, so "the fragments of stream `s`" is not a
/// well-formed request — a stream written by two phases holds two generations.
/// Stating it makes an op that reads the wrong generation a wrong *number* the
/// caller wrote down rather than a silent default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentInput {
    pub stream: String,
    /// The phase whose blocks wrote the fragments.
    pub phase: usize,
    /// Neighbouring blocks read on each axis, **in block units**. `[0, 0, 0]`
    /// means this block's fragment and nothing else, and the executor then
    /// fetches exactly one fragment per stream per block.
    pub reach: [usize; 3],
}

impl FragmentInput {
    /// This block's own fragment, from `phase`, and no neighbour.
    pub fn own(stream: impl Into<String>, phase: usize) -> Self {
        Self {
            stream: stream.into(),
            phase,
            reach: [0, 0, 0],
        }
    }

    pub fn with_reach(mut self, reach: [usize; 3]) -> Self {
        self.reach = reach;
        self
    }

    /// Every block of the lattice, whatever this block's index. What a global
    /// reduction declares.
    pub fn whole(stream: impl Into<String>, phase: usize, grid: &BlockGrid) -> Self {
        Self {
            stream: stream.into(),
            phase,
            reach: grid.blocks_per_axis(),
        }
    }
}

/// Which blocks of the phase's lattice must appear in a stream.
///
/// **This is the fragment side of the tiling guard, and it exists because
/// without it there is none.** `Decomposition::check` and the executor's
/// post-run check both assert that the phase's *valid regions* tile the volume.
/// For a phase that writes a pixel image that is a statement about the output.
/// For a phase whose output is fragments it is a statement about an image nobody
/// wrote: `valid == core` by construction, cores tile by construction, and the
/// check passes while asserting nothing at all about the fragments — which are
/// the actual output. A guard that cannot fail is worse than no guard, because
/// it is trusted.
///
/// So a stream says which of the two it is, and there is no default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coverage {
    /// Every block of the phase's lattice writes exactly one fragment. The
    /// executor checks it against the *store* after the phase, so what is
    /// checked is what landed rather than what the plan promised.
    ///
    /// This is nearly always the right answer, including for a block with
    /// nothing to say: a zero-length fragment is present, and "present and
    /// empty" is a different fact from "absent", which is exactly the
    /// distinction a merge needs.
    EveryBlock,
    /// Blocks may write or not. The coverage guard then cannot constrain
    /// anything, and only containment is checked — every key names a block of
    /// this lattice.
    Sparse,
}

impl Coverage {
    pub fn as_str(self) -> &'static str {
        match self {
            Coverage::EveryBlock => "every-block",
            Coverage::Sparse => "sparse",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "every-block" => Some(Coverage::EveryBlock),
            "sparse" => Some(Coverage::Sparse),
            _ => None,
        }
    }
}

/// A fragment stream an op writes.
///
/// Both fields are decisions with no default. The lifecycle says whether the
/// fragments survive the run; the coverage says what the guard is allowed to
/// assert about them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentOutput {
    pub stream: String,
    pub lifecycle: Lifecycle,
    pub coverage: Coverage,
}

impl FragmentOutput {
    pub fn new(stream: impl Into<String>, lifecycle: Lifecycle, coverage: Coverage) -> Self {
        Self {
            stream: stream.into(),
            lifecycle,
            coverage,
        }
    }
}

/// What one block of a fragment phase produced.
#[derive(Default)]
pub struct BlockOutput {
    /// `(stream, bytes)`, keyed by the executor as `(stream, this phase, this
    /// block)`. Every stream named must be one the op declared.
    pub fragments: Vec<(String, Vec<u8>)>,
    /// The block's pixels, over its **read** extent, for an op that declares
    /// `writes_pixels`. The executor slices the valid sub-box out of it, exactly
    /// as it does for a `BlockOp`.
    pub pixels: Option<BlockBuf>,
}

impl BlockOutput {
    pub fn nothing() -> Self {
        Self::default()
    }

    pub fn fragment(stream: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            fragments: vec![(stream.into(), bytes)],
            pixels: None,
        }
    }

    pub fn with_fragment(mut self, stream: impl Into<String>, bytes: Vec<u8>) -> Self {
        self.fragments.push((stream.into(), bytes));
        self
    }

    pub fn with_pixels(mut self, pixels: BlockBuf) -> Self {
        self.pixels = Some(pixels);
        self
    }
}

/// What one block of a fragment phase is handed.
///
/// Fragments inside the declared reach are **gathered by the executor**, not
/// pulled by the op. That is what makes "a zero-reach phase reads no neighbour"
/// a property of the framework that can be measured from the outside rather
/// than a promise each op makes separately. An op that would rather not have
/// them all resident says so with [`FragmentOp::gathers`] and streams them with
/// [`BlockView::stream_fragments`]; the neighbourhood is the same either way,
/// because the executor computed it from the declaration.
pub struct BlockView<'a> {
    /// The phase this block is running, which is also the phase its output
    /// fragments are keyed under.
    pub phase: usize,
    /// The block's index in this phase's grid.
    pub index: [usize; 3],
    /// This phase's lattice.
    pub grid: &'a BlockGrid,
    /// The block owns this region of the volume.
    pub core: &'a Region,
    /// What the phase's read extent covers. Wider than the core where a reach
    /// or a fragment reach was declared.
    pub read: &'a Region,
    /// The region this block's output is authoritative for.
    pub valid: &'a Region,
    /// Where `read` sits in the volume, for an op anchored to the global grid.
    pub at: Anchor,
    /// The element type of the image this phase **writes**, which is what
    /// [`Self::output_buffer`] allocates at. Carried rather than assumed: a
    /// buffer of the wrong width is a buffer the write refuses, and the executor
    /// is the only thing that knows which image this block is destined for.
    pub dtype: Dtype,
    env: &'a dyn Environment,
    pixels: Option<&'a BlockBuf>,
    /// Stream -> (producing phase, the blocks the declaration asks for).
    wanted: BTreeMap<String, (usize, Vec<[usize; 3]>)>,
    gathered: BTreeMap<String, Vec<(FragmentKey, Vec<u8>)>>,
}

impl<'a> BlockView<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        phase: usize,
        index: [usize; 3],
        grid: &'a BlockGrid,
        core: &'a Region,
        read: &'a Region,
        valid: &'a Region,
        at: Anchor,
        dtype: Dtype,
        env: &'a dyn Environment,
        pixels: Option<&'a BlockBuf>,
        wanted: BTreeMap<String, (usize, Vec<[usize; 3]>)>,
        gathered: BTreeMap<String, Vec<(FragmentKey, Vec<u8>)>>,
    ) -> Self {
        Self {
            phase,
            index,
            grid,
            core,
            read,
            valid,
            at,
            dtype,
            env,
            pixels,
            wanted,
            gathered,
        }
    }

    /// The block's pixels, or an error naming the declaration that would have
    /// produced them.
    ///
    /// An op that declared `reads_pixels() == false` and then asks for pixels
    /// is a bug in the op, and it is a *loud* one: the alternative is an op
    /// that quietly computes something else on the fragment-only path.
    pub fn pixels(&self) -> Result<&BlockBuf> {
        self.pixels.ok_or_else(|| {
            Error::InvalidArgument(
                "this fragment op asked for its block's pixels, but it declares \
                 `reads_pixels() == false`, so the executor read none. An op that needs \
                 pixels says so; a phase whose op says no does no pixel IO at all."
                    .to_string(),
            )
        })
    }

    /// Whether the executor read this block's pixels.
    pub fn has_pixels(&self) -> bool {
        self.pixels.is_some()
    }

    /// The volume this block is anchored in — the phase's own coordinate space,
    /// and the one [`FragmentOp::reach`] and [`FragmentOp::source_inputs`] state
    /// their answers over.
    pub fn volume(&self) -> [usize; 3] {
        self.at.volume
    }

    /// A buffer shaped like this block's read extent, filled with `value`.
    ///
    /// Routed through the environment on purpose: it is the one construction an
    /// op needs that differs between a run that holds data and a run that only
    /// counts, and going through `Environment::constant` is what lets one op
    /// serve both. A `writes_pixels` op that built an array directly would be
    /// unsimulatable.
    ///
    /// The element type is the one the image this phase writes holds; see
    /// [`Self::dtype`].
    pub fn output_buffer(&self, value: f64) -> Result<BlockBuf> {
        self.env.constant(self.dtype, self.read, value)
    }

    /// The blocks the declaration asks for on `stream`, clamped to the lattice.
    /// Empty for a stream this op did not declare.
    pub fn wanted(&self, stream: &str) -> &[[usize; 3]] {
        self.wanted
            .get(stream)
            .map(|(_, blocks)| blocks.as_slice())
            .unwrap_or_default()
    }

    /// Every fragment gathered for `stream`, in key order, and nothing outside
    /// the declared reach.
    ///
    /// A block that wrote no fragment simply does not appear; a hole is a
    /// normal outcome, not an error, and the op decides what it means. Empty
    /// when the op declared `gathers() == false`.
    pub fn fragments(&self, stream: &str) -> &[(FragmentKey, Vec<u8>)] {
        self.gathered
            .get(stream)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// This block's own fragment in `stream`.
    pub fn own(&self, stream: &str) -> Option<&[u8]> {
        self.fragments(stream)
            .iter()
            .find(|(key, _)| key.block == self.index)
            .map(|(_, bytes)| bytes.as_slice())
    }

    /// Every gathered fragment in `stream` except this block's own.
    pub fn neighbours(&self, stream: &str) -> impl Iterator<Item = &(FragmentKey, Vec<u8>)> {
        let index = self.index;
        self.fragments(stream)
            .iter()
            .filter(move |(key, _)| key.block != index)
    }

    /// Visit the declared neighbourhood of `stream` one fragment at a time.
    ///
    /// For the op that reaches over the whole lattice: gathering there would put
    /// every fragment of every block in memory at once, once per block, which is
    /// the residency problem per-block fragments exist to avoid. Streaming costs
    /// the same reads and keeps one fragment resident.
    pub fn stream_fragments(
        &self,
        stream: &str,
        visit: &mut dyn FnMut(&FragmentKey, &[u8]) -> Result<()>,
    ) -> Result<usize> {
        let Some((phase, blocks)) = self.wanted.get(stream) else {
            return Err(Error::InvalidArgument(format!(
                "this op did not declare stream {stream:?} as an input, so the executor \
                 computed no neighbourhood for it. Declare it in `inputs`."
            )));
        };
        let mut seen = 0usize;
        for &block in blocks {
            if let Some(bytes) = self.env.read_sidecar(stream, *phase, block)? {
                visit(&FragmentKey::new(stream, *phase, block), &bytes)?;
                seen += 1;
            }
        }
        Ok(seen)
    }
}

/// The stored images a fragment op declared, keyed by image.
///
/// The fragment side of [`SourceInputs`](crate::op::SourceInputs), with the same
/// contract: one entry per image rather than one per declaration, each holding
/// **the extent the block was read at**, and [`Self::get`] erroring rather than
/// returning an `Option` — a missing operand is a block that would compute
/// against nothing, which is the class of quiet wrong answer this crate is
/// arranged against.
///
/// **A second type rather than the same one**, and for one reason: a fragment op
/// is handed [`BlockBuf`], not `Voxels`. [`BlockView::pixels`] already is,
/// because a simulated run holds no array at all and a fragment op that could
/// not be simulated would be a phase the planner cannot price. Nothing else
/// about it differs, and nothing about it is a second *mechanism* — the
/// declaration is `BlockOp`'s [`SourceInput`], the plan record is the same
/// `PhaseDecomposition::source_images`, and the executor reads it through the
/// same `Environment::read`.
#[derive(Clone, Copy)]
pub struct SourceBlocks<'a> {
    entries: &'a [(usize, &'a BlockBuf)],
}

impl<'a> SourceBlocks<'a> {
    /// Nothing stored: what an op declaring no source input is applied with.
    pub const fn none() -> Self {
        Self { entries: &[] }
    }

    pub fn new(entries: &'a [(usize, &'a BlockBuf)]) -> Self {
        Self { entries }
    }

    /// The images supplied, in the order the executor read them.
    pub fn images(&self) -> Vec<usize> {
        self.entries.iter().map(|(image, _)| *image).collect()
    }

    /// The block of `image`, or an error naming what was supplied.
    pub fn get(&self, image: usize) -> Result<&'a BlockBuf> {
        self.entries
            .iter()
            .find(|(named, _)| *named == image)
            .map(|(_, buf)| *buf)
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "a fragment op reads image {image} and the executor supplied [{}]. The \
                     images a phase reads besides its own input are recorded in the plan \
                     (`PhaseDecomposition::source_images`) and read there; an op naming one \
                     the plan does not list has nothing to be handed.",
                    self.entries
                        .iter()
                        .map(|(named, _)| named.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
    }
}

/// What an op does with values that straddle a block seam, and therefore
/// whether its answer is a function of the decomposition.
///
/// **There is no default, on [`Coverage`]'s argument.** A per-region reduction
/// over a second array is summed in pieces — one per block the region touches —
/// and the pieces are combined in a merge. Combine them with `f64` addition and
/// the result depends on the order the blocks merged in, so the same data cut
/// two ways gives two answers, neither of them wrong-looking. `ops/detect.rs`
/// deferred a weighted centroid on precisely this. A silent default here would
/// be a promise nobody made, about the one property this crate exists to keep.
///
/// So the obligation is stated, and it is stated **where the values come from**:
/// [`check_phase_work`] requires a declaration from any fragment op that reads a
/// second array, and from any op that folds — with a non-zero fragment reach —
/// a stream carrying values that came from one. The obligation follows the
/// operand through the fragment graph rather than stopping at the op that read
/// it, because the op that reads the array is usually not the op that adds
/// across the seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeamFold {
    /// **Nothing crosses a seam here.** This block's fragment is a function of
    /// this block alone, so there is no order to depend on. What an emitting
    /// `volume -> fragments` op says.
    ///
    /// Checked, not believed: an op declaring this may not declare a fragment
    /// input with a non-zero reach, because a non-zero reach *is* a fold over
    /// more than one block's values.
    PerBlock,
    /// **A function of the set of fragments, not of their order.** Integer
    /// addition, min, max, bitwise or, a fixed-point accumulator, a union-find
    /// over labels — anything associative and commutative in the type it is
    /// accumulated in.
    ///
    /// Checked, not believed: the executor applies the block a second time with
    /// the declared neighbourhood **reversed** and requires byte-identical
    /// output. An `f64` sum over three or more fragments fails that, which is
    /// the hazard this variant exists to catch.
    ///
    /// What it costs: one extra application of the op per block, and — for an
    /// op that declared `gathers() == false` and streams — one extra pass of
    /// sidecar reads, which the counters show. What it forbids is claiming
    /// order-independence for free.
    Unordered,
    /// **The fold is order-dependent, and the op says so.** Allowed, and it is
    /// the honest declaration for a real `f64` accumulation.
    ///
    /// What it forbids is the assumption everything else in this crate is
    /// arranged around: a phase declaring this is **not decomposition-invariant**
    /// — the same volume cut into different blocks may give a different answer,
    /// within the rounding of the accumulator — so its output must not be
    /// compared byte-for-byte across plans, and a run of it is reproducible only
    /// against a fixed decomposition. Nothing is checked, because there is
    /// nothing to check; what is bought is that a reader of the op can see it.
    OrderDependent,
}

impl SeamFold {
    pub fn as_str(self) -> &'static str {
        match self {
            SeamFold::PerBlock => "per-block",
            SeamFold::Unordered => "unordered",
            SeamFold::OrderDependent => "order-dependent",
        }
    }
}

/// One operation whose input or output is a fragment rather than a pixel
/// region.
///
/// `Send + Sync` for the same reason [`crate::op::BlockOp`] is: the executor
/// runs blocks concurrently and shares one reference across workers.
pub trait FragmentOp: Send + Sync {
    fn name(&self) -> &'static str;

    /// Voxels read beyond the voxel written along `axis`, in the coordinate
    /// system the op is anchored to — the same contract as `BlockOp::reach`,
    /// and it governs the same thing: what is shrunk off the read extent to get
    /// the region this block's output may be trusted for.
    ///
    /// Compute it independently of any configured halo, or the guard in
    /// `decomposition.rs` compares a number against itself and cannot fire. An
    /// op whose answer depends on every block returns `volume_len`, and see the
    /// module header for exactly what the geometry then does.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    /// Does this op read its block's pixels?
    ///
    /// `false` — the default — means the executor performs **no pixel IO** for
    /// this phase: no `Environment::read`, no read counters, no chunk counters.
    /// That is the `fragments -> fragments` case, and it is the default because
    /// an op that says nothing should cost nothing.
    fn reads_pixels(&self) -> bool {
        false
    }

    /// Does this op write a pixel region?
    ///
    /// `true` makes the phase produce an image like any other, so an ordinary
    /// pixel phase may follow it. `false` means the phase writes fragments and
    /// nothing else, and is therefore terminal as far as images are concerned.
    fn writes_pixels(&self) -> bool {
        false
    }

    /// What element type this op writes, given the one it reads.
    ///
    /// The counterpart of [`BlockOp::produces`](crate::op::BlockOp::produces),
    /// and it exists for the same reason: `check_dtypes` folds a plan's element
    /// types from image 0 and refuses an image allocated at a width its producer
    /// does not write. That fold runs over the *chain*, and a fragment phase owns
    /// no slot of the chain — so before this method existed there was nothing to
    /// fold and a fragment op that changed the width was refused with a message
    /// about ops it does not have.
    ///
    /// The default hands the type on unchanged, which is right for every
    /// fragment op this crate shipped before the method existed and is the safe
    /// default in the same way `BlockOp`'s is: an op that says nothing keeps
    /// exactly the contract it already had. Only consulted when
    /// [`Self::writes_pixels`] is true; a phase that writes no image has no
    /// image whose width could be wrong.
    fn produces(&self, input: Dtype) -> Dtype {
        input
    }

    /// Fragment streams this op reads, with the phase they were written by and
    /// the reach in blocks. Empty is the `volume -> fragments` case.
    fn inputs(&self) -> Vec<FragmentInput> {
        Vec::new()
    }

    /// Fragment streams this op writes, and what becomes of them. May be empty
    /// for a `fragments -> volume` op, which writes pixels instead.
    fn outputs(&self) -> Vec<FragmentOutput> {
        Vec::new()
    }

    /// Should the executor gather the declared neighbourhood before calling
    /// [`Self::apply`]?
    ///
    /// `true` by default because it is what makes the reach checkable from
    /// outside the op. An op that reaches over the whole lattice should say
    /// `false` and use [`BlockView::stream_fragments`]: the reads are the same,
    /// the residency is one fragment instead of all of them.
    fn gathers(&self) -> bool {
        true
    }

    /// Stored images this op reads **besides** the image it is handed, each with
    /// its own reach.
    ///
    /// The same declaration [`BlockOp::source_inputs`] makes, in the same units,
    /// recorded in the same `PhaseDecomposition::source_images` and read by the
    /// executor through the same `Environment::read`. Empty by default, which is
    /// every fragment op this crate shipped before the method existed and is the
    /// honest answer for all of them.
    ///
    /// **Independent of [`Self::reads_pixels`].** That method is about image
    /// `p`, the one the phase is handed; this is about every other image. An op
    /// may read a second array without reading its own input — a merge that
    /// consults a stored array says so here and still declares
    /// `reads_pixels() == false`, and pays for one array rather than two.
    ///
    /// `volume` is the phase's own anchoring volume, the same one
    /// [`Self::reach`] is given an axis of.
    ///
    /// [`BlockOp::source_inputs`]: crate::op::BlockOp::source_inputs
    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        Vec::new()
    }

    /// What this op does with values that straddle a seam. See [`SeamFold`].
    ///
    /// `None` means "has not said", and it is the default so that nothing that
    /// shipped before this method changes. It is not a licence: an op that reads
    /// a second array, or that folds fragments carrying a second array's values
    /// over a non-zero reach, is refused by name at plan time until it answers.
    fn seam_fold(&self) -> Option<SeamFold> {
        None
    }

    /// Produce this block's output.
    ///
    /// Returning no fragment at all is legitimate — a block with nothing to say
    /// writes nothing, which is exactly what an absent key means to a reader.
    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput>;

    /// [`Self::apply`], with the stored images [`Self::source_inputs`] declared.
    ///
    /// Each buffer holds that image over the **block's own fetch region** — the
    /// same extent [`BlockView::pixels`] would be, and the same extent
    /// `Chain::Source` leaves are handed — because that is what the plan
    /// records and the executor reads.
    ///
    /// **The default refuses rather than falling through**, exactly as
    /// [`BlockOp::apply_with`] does, and for the reason `env.rs:279` gives for
    /// it: *silently ignoring an operand is the precise shape of the wrong
    /// answer this whole change exists to remove — a complete, well-formed
    /// volume combined against nothing.* An op that declares an operand and
    /// forgets to override this would emit a fragment summarising one array
    /// while claiming to summarise two, which is a plausible number and
    /// therefore the expensive kind of wrong.
    ///
    /// It hands off to [`Self::apply`] when nothing was declared, which is the
    /// case that must stay free.
    ///
    /// [`BlockOp::apply_with`]: crate::op::BlockOp::apply_with
    fn apply_with(&self, at: &BlockView<'_>, _sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let declared = self.source_inputs(at.volume());
        if !declared.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "fragment op {:?} declares {} source input(s) (image(s) {}) and does not \
                 override `apply_with`, so the operands it asked the plan to fetch would be \
                 dropped on the floor and the block would be summarised from one array while \
                 claiming to summarise two.",
                self.name(),
                declared.len(),
                declared
                    .iter()
                    .map(|input| input.image.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        self.apply(at)
    }
}

/// What a phase runs.
///
/// The decomposition says how a phase is *blocked*; this says what a block of
/// it does. They are separate because the decomposition is the binding,
/// serialisable, integer half — it travels to a worker over a wire — and an op
/// is not serialisable at all. A distributed worker is given the decomposition
/// and rebuilds the work locally, exactly as it already does for `Chain`.
#[derive(Clone, Copy)]
pub enum PhaseWork<'a> {
    /// The chain slots the decomposition assigned to this phase. `region ->
    /// region`, and the only kind that existed before this file.
    Pixels,
    /// A fragment op. The phase must own no chain slots.
    Fragments(&'a dyn FragmentOp),
    /// An iteration run to a fixed point inside this one phase. The phase must
    /// own no chain slots.
    ///
    /// `region -> region` like `Pixels`, and it reads and writes an image the same
    /// way; what differs is that a block of it is visited an unknown number of
    /// times before the image is written, and that every visit is handed more
    /// than one operand. See `crate::iterate` for why that cannot be a
    /// `Chain::sequence`.
    Iterate(&'a dyn crate::iterate::IterativeOp),
}

impl PhaseWork<'_> {
    pub fn is_fragments(&self) -> bool {
        matches!(self, PhaseWork::Fragments(_))
    }

    pub fn is_iterative(&self) -> bool {
        matches!(self, PhaseWork::Iterate(_))
    }

    /// Whether this phase produces the image after it.
    pub fn writes_an_image(&self) -> bool {
        match self {
            PhaseWork::Pixels | PhaseWork::Iterate(_) => true,
            PhaseWork::Fragments(op) => op.writes_pixels(),
        }
    }

    /// Whether this phase reads the image before it.
    pub fn reads_an_image(&self) -> bool {
        match self {
            PhaseWork::Pixels | PhaseWork::Iterate(_) => true,
            PhaseWork::Fragments(op) => op.reads_pixels(),
        }
    }

    /// A name for diagnostics.
    pub fn describe(&self) -> String {
        match self {
            PhaseWork::Pixels => "pixels".to_string(),
            PhaseWork::Fragments(op) => format!("fragments({})", op.name()),
            PhaseWork::Iterate(op) => format!("iterate({})", op.name()),
        }
    }
}

// -------------------------------------------------------------- geometry --

/// The blocks within `reach` of `index` on a lattice of `counts` blocks, in
/// row-major order.
///
/// Clamped at the lattice edge, so an edge block sees fewer neighbours than an
/// interior one — the same asymmetry a clamped halo has, and for the same
/// reason. Public because it is also the *analytic* answer a test checks the
/// measured fetch count against.
pub fn neighbourhood(index: [usize; 3], reach: [usize; 3], counts: [usize; 3]) -> Vec<[usize; 3]> {
    let mut bounds = [(0usize, 0usize); 3];
    for axis in 0..3 {
        let last = counts[axis].saturating_sub(1);
        bounds[axis] = (
            index[axis].saturating_sub(reach[axis]),
            (index[axis] + reach[axis]).min(last),
        );
    }
    let mut out = Vec::new();
    for i in bounds[0].0..=bounds[0].1 {
        for j in bounds[1].0..=bounds[1].1 {
            for k in bounds[2].0..=bounds[2].1 {
                out.push([i, j, k]);
            }
        }
    }
    out
}

/// How many blocks [`neighbourhood`] returns, without building it.
pub fn neighbourhood_size(index: [usize; 3], reach: [usize; 3], counts: [usize; 3]) -> usize {
    (0..3)
        .map(|axis| {
            let last = counts[axis].saturating_sub(1);
            let lo = index[axis].saturating_sub(reach[axis]);
            let hi = (index[axis] + reach[axis]).min(last);
            hi + 1 - lo
        })
        .product()
}

/// The decomposition phase a fragment op runs as.
///
/// The halo is the widest of three things: the op's own reach, every fragment
/// input's reach converted to voxels, and every [`FragmentOp::source_inputs`]
/// reach. Where the op's reach is smaller than the halo the
/// valid regions equal the cores and tile the volume; where it is the whole
/// volume, only `halo == volume` survives the tiling check, which is the
/// module header's point about a global step.
///
/// **The source reach widens the halo and not the reach**, which is the same
/// arrangement a fragment input's reach already has. The halo is what is
/// *fetched*, so an operand consulted over a window needs it; the reach is what
/// is *shrunk off the read extent to get the trusted region*, and a fragment is
/// emitted for the block's core whatever window the operand was read over. An op
/// whose **pixel** output at a voxel depends on the operand's window is making a
/// statement about trust and says it in [`FragmentOp::reach`]; nothing here can
/// say it on its behalf.
///
/// The images the source inputs name are recorded on the phase, which is what
/// makes the executor fetch them, the DAG depend on their producers, the image
/// lifetimes keep them alive, `exact_read_voxels` count them and the fingerprint
/// hash them. A fragment phase built any other way would read a second array off
/// the books.
///
/// The op's [`FragmentOp::reads_pixels`] is recorded too, for the other half of
/// the same books: an op that declines its own image is not charged for it by
/// `Decomposition::exact_read_voxels`, because `strategy::run_fragment_task`
/// does not fetch it. A phase reading a second array and not its own then
/// predicts one array's worth rather than two, which is what the run performs.
///
/// `names` is left empty on purpose: `Decomposition::op_names_in_order` pairs
/// with `slot_order`, which the acceptance criterion is asserted against, and a
/// fragment phase occupies no slot. Naming it there would put a name in that
/// list with no `OpApplied` event to match it and break the check for every
/// mixed decomposition.
pub fn fragment_phase(op: &dyn FragmentOp, grid: BlockGrid) -> Result<PhaseDecomposition> {
    let volume = grid.volume();
    let edge = grid.block();
    let mut reach = [0usize; 3];
    for (axis, value) in reach.iter_mut().enumerate() {
        *value = op.reach(axis, volume[axis]);
    }
    let mut halo = reach;
    for input in op.inputs() {
        check_stream_name(&input.stream)?;
        for axis in 0..3 {
            halo[axis] = halo[axis].max(input.reach[axis].saturating_mul(edge[axis]));
        }
    }
    if op.outputs().is_empty() && !op.writes_pixels() {
        return Err(Error::InvalidArgument(format!(
            "fragment op {:?} declares no output stream and writes no pixels, so running it \
             would produce nothing at all.",
            op.name()
        )));
    }
    for output in op.outputs() {
        check_stream_name(&output.stream)?;
    }
    let mut images = Vec::new();
    let mut supplied = Vec::new();
    for input in op.source_inputs(volume) {
        let wanted = input.reach.in_voxels(edge);
        for (axis, value) in halo.iter_mut().enumerate() {
            let (lo, hi) = wanted.axis(axis).bound(volume[axis]);
            *value = (*value).max(lo).max(hi);
        }
        // A supplied input is produced by no phase, so there is no fold that
        // could say what is in it and the op's own declaration is the only
        // statement there is. Recorded here for the same reason
        // `Decomposition::declare_source_images` records the chain half's: this
        // is the only thing holding the op.
        if input.image.is_supplied() {
            let dtype = input.dtype.ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "fragment op {:?} reads {}, and nothing says what it holds. An image the run                      writes has its element type in the fold of the chain that wrote it; a                      supplied input is produced by no phase, so the reader is the only                      statement there is — declare it with `SourceInput::holding`.",
                    op.name(),
                    describe_image(input.image.index())
                ))
            })?;
            supplied.push((input.image.index(), dtype));
        }
        images.push(input.image.index());
    }
    Ok(
        PhaseDecomposition::derive(Vec::new(), Vec::new(), reach, halo, grid)
            .with_source_images(images)
            .with_supplied_dtypes(supplied)
            .reading_input_image(op.reads_pixels()),
    )
}

/// Add a fragment phase to the end of a decomposition, on the last phase's
/// lattice.
///
/// The lattice is inherited rather than chosen because fragments are keyed by
/// block index: a phase that read another phase's fragments on a *different*
/// lattice would be addressing blocks that do not correspond to anything.
pub fn append_fragment_phase(
    mut plan: Decomposition,
    op: &dyn FragmentOp,
) -> Result<Decomposition> {
    let grid = plan
        .phases
        .last()
        .ok_or_else(|| {
            Error::InvalidArgument(
                "a fragment phase can only be appended to a decomposition that has one".to_string(),
            )
        })?
        .grid
        .clone();
    plan.phases.push(fragment_phase(op, grid)?);
    Ok(plan)
}

/// A decomposition made of fragment phases and nothing else.
///
/// **The honest statement about a run with no pixels in it.** A `Decomposition`
/// is expressed over a volume and a block lattice, and there is no way to say
/// "a lattice of N blocks over nothing" — `decomposition.rs` derives every
/// block's core, read extent and valid region from a volume, and
/// `Decomposition::check` asserts that the valid regions tile that volume. So
/// this takes a volume, and the volume is doing one job: **it is the coordinate
/// system the block lattice is cut from.** No phase built here reads a pixel of
/// it unless its op says it does.
///
/// When the fragments came from an earlier pixel run, that is exact rather than
/// nominal: pass the volume they were produced over and the block indices line
/// up by construction, which is what [`append_fragment_phase`] does. For a run
/// whose fragments have no volume behind them at all, the volume is a stated
/// lattice size and nothing more — a placeholder, and named as one here rather
/// than dressed up.
pub fn fragment_only(
    volume: [usize; 3],
    block: [usize; 3],
    dtype: Dtype,
    ops: &[&dyn FragmentOp],
) -> Result<Decomposition> {
    if ops.is_empty() {
        return Err(Error::InvalidArgument(
            "a fragment-only decomposition needs at least one op; `Decomposition::check` \
             rejects a plan with no phases"
                .to_string(),
        ));
    }
    let grid = BlockGrid::new(volume, block)?;
    let mut phases = Vec::with_capacity(ops.len());
    for op in ops {
        phases.push(fragment_phase(*op, grid.clone())?);
    }
    let plan = Decomposition {
        volume,
        dtype,
        phases,
        chain_reach: [0, 0, 0],
    };
    plan.check()?;
    Ok(plan)
}

/// The guard for a `(decomposition, work)` pair, run once per execution.
///
/// Everything it refuses is a plan that would run and produce something wrong
/// or unreadable, which is the only kind of check this crate adds.
pub fn check_phase_work(plan: &Decomposition, work: &[PhaseWork<'_>]) -> Result<()> {
    if work.len() != plan.n_phases() {
        return Err(Error::InvalidArgument(format!(
            "the decomposition has {} phase(s) and {} were described. Every phase says what \
             it runs; there is no default.",
            plan.n_phases(),
            work.len()
        )));
    }
    // Image 0 is the workflow input and always exists; image p+1 exists iff
    // phase p wrote it. A phase that reads no pixels needs neither.
    let mut image_written: Vec<Option<usize>> = vec![None; work.len() + 1];
    // Streams holding values that came out of a **second array**, addressed the
    // way a `FragmentInput` addresses them: `(stream, the phase that wrote it)`.
    //
    // The obligation to state a [`SeamFold`] follows the operand along these
    // edges rather than stopping at the op that read the array, because the op
    // that reads the array is not in general the op that adds across the seam:
    // an emitting op produces one partial per block and a later merge combines
    // them, and it is the *merge* whose accumulator decides whether the answer
    // is decomposition-invariant. A plan with no source input anywhere never
    // marks a stream, so nothing that shipped before this is asked anything.
    let mut carries_operand: BTreeSet<(String, usize)> = BTreeSet::new();
    for (index, entry) in work.iter().enumerate() {
        let phase = &plan.phases[index];
        if entry.reads_an_image() && index > 0 && image_written[index].is_none() {
            return Err(Error::InvalidArgument(format!(
                "phase {index} reads image {index}, which phase {} did not write: it runs a \
                 fragment op that declares `writes_pixels() == false`. A phase that writes \
                 only fragments is terminal as far as images go; an op that hands pixels on \
                 says `writes_pixels`.",
                index - 1
            )));
        }
        if entry.writes_an_image() {
            image_written[index + 1] = Some(index);
        }
        if let PhaseWork::Iterate(op) = entry {
            // Re-run when the plan is *used*, not only when it is built, on
            // exactly `check_block_constraints`' argument: a plan may arrive from
            // any strategy or off a wire.
            crate::iterate::check_iterative(*op)?;
            if !phase.slots.is_empty() {
                return Err(Error::InvalidArgument(format!(
                    "phase {index} runs iterative op {:?} but the decomposition gives it chain \
                     slots {:?}. An iterative phase owns no slot of the chain — see \
                     `iterate::iterative_phase`, which builds one.",
                    op.name(),
                    phase.slots
                )));
            }
            // The private ping-pong buffers are the phase's own volume, and the
            // running operand of substage 0 is the image below read through the
            // same block geometry. A phase that resized or re-gridded between the
            // two would be handing substage 1 an operand of a different shape
            // than substage 0 produced, so the iteration would not close.
            let below = plan.volume_at(index);
            if phase.volume() != below || phase.reads_across_grids() {
                return Err(Error::InvalidArgument(format!(
                    "phase {index} runs iterative op {:?} and reads a {below:?} image to write a \
                     {:?} one. An iteration feeds its own output back in, so its substages must \
                     agree on one extent and one lattice; a phase that changes either is a \
                     transformation and belongs before or after the iteration, not inside it.",
                    op.name(),
                    phase.volume()
                )));
            }
        }
        // **The half of `check_source_images` a chain cannot make.** That guard
        // folds the *slots* of a phase, so it can only speak for a phase that
        // owns some, and it skips the ones that do not. The two kinds that own
        // none are the two here: a `Pixels` phase with an empty slot list reads
        // nothing besides its input, and an iterative phase's second operand is
        // `Operand::Fixed`, which is its own input image and not a second one.
        // Either recording a source image is a plan that would fetch an array
        // nothing consumes, and be priced for it.
        if phase.slots.is_empty() && !entry.is_fragments() && !phase.source_images.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "decomposition phase {index} ({}) owns no chain slot and records that it also \
                 reads image(s) {:?}. Only a fragment op can read a second image without a \
                 chain slot to say so, and this phase runs {}. The recorded list is what the \
                 executor fetches and what the fingerprint hashes, so an entry nothing reads \
                 is a plan priced for an array no block consumes.",
                phase.names.join(">"),
                phase.source_images,
                entry.describe()
            )));
        }
        let PhaseWork::Fragments(op) = entry else {
            continue;
        };
        if !phase.slots.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "phase {index} runs fragment op {:?} but the decomposition gives it chain \
                 slots {:?}. A fragment phase owns no slot of the chain — see \
                 `fragment_phase`, which builds one.",
                op.name(),
                phase.slots
            )));
        }
        if op.outputs().is_empty() && !op.writes_pixels() {
            return Err(Error::InvalidArgument(format!(
                "phase {index}: fragment op {:?} produces nothing at all",
                op.name()
            )));
        }
        // The gap this closes: a phase that writes no pixel image is not
        // constrained by the tiling check at all — its valid regions equal its
        // cores by construction, so the check passes whatever the fragments do.
        // If nothing about such a phase is checkable, say so at plan time
        // rather than running it and calling the vacuous check a guard.
        if !op.writes_pixels()
            && !op
                .outputs()
                .iter()
                .any(|output| output.coverage == Coverage::EveryBlock)
        {
            return Err(Error::InvalidArgument(format!(
                "phase {index}: fragment op {:?} writes no pixel image and declares no \
                 every-block stream, so nothing about its output can be checked. The \
                 tiling guard passes for such a phase whatever it wrote — its valid \
                 regions are its cores by construction — so a run of it would be entirely \
                 unguarded. Declare at least one output with `Coverage::EveryBlock`; a \
                 block with nothing to say writes a zero-length fragment, which is present \
                 and therefore checkable.",
                op.name()
            )));
        }
        let edge = phase.grid.block();

        // ------------------------------------------------ the second array --
        //
        // Everything `check_source_images` asserts about a chain's source
        // leaves, asserted here about a fragment op's, because this is the only
        // place holding both the plan and the op. The list is deliberately the
        // same list, in the same order, so that the two guards cannot drift into
        // meaning different things about one field.
        let read_volume = plan.volume_at(index);
        let declared_sources = op.source_inputs(read_volume);
        if !declared_sources.is_empty()
            && (phase.volume() != read_volume || phase.reads_across_grids())
        {
            return Err(Error::InvalidArgument(format!(
                "phase {index}: fragment op {:?} reads a second image and the phase reads a \
                 {read_volume:?} image to work in {:?}. A source image is fetched at the \
                 block's own fetch region, so an op that reads one must be on one lattice in \
                 one coordinate space; across grids the same integers name different voxels, \
                 and the op is asked for its declaration in a volume that is not the one it \
                 would be handed.",
                op.name(),
                phase.volume()
            )));
        }
        {
            let granted = phase.halo.in_voxels(edge);
            for input in &declared_sources {
                // **The equal-reach limit binds here too, and is not loosened.**
                // The executor reads a source image at the block's own fetch
                // region, so an operand wanting more than the phase fetches
                // would be handed a buffer narrower than its kernel walks. What
                // differs from the chain case is only who widens the halo:
                // `fragment_phase` folds this reach into it, so a phase built
                // that way satisfies the limit by construction and this check is
                // for the plan that arrived from somewhere else.
                let wanted = input.reach.in_voxels(edge);
                for axis in 0..3 {
                    let (want_lo, want_hi) = wanted.axis(axis).bound(read_volume[axis]);
                    let (have_lo, have_hi) = granted.axis(axis).bound(read_volume[axis]);
                    if want_lo > have_lo || want_hi > have_hi {
                        return Err(Error::InvalidArgument(format!(
                            "phase {index}: fragment op {:?} reads image {} with a reach of \
                             {want_lo}+{want_hi} on axis {axis}, and the phase is granted a \
                             halo of {have_lo}+{have_hi} there. A source image is read at the \
                             block's own fetch region, so an operand reaching further than the \
                             phase does would be handed a buffer its kernel walks past the end \
                             of. Build the phase with `fragment_phase`, which folds this reach \
                             into the halo.",
                            op.name(),
                            input.image
                        )));
                    }
                }
            }
            let mut named: Vec<usize> = declared_sources
                .iter()
                .map(|input| input.image.index())
                .collect();
            named.sort_unstable();
            named.dedup();
            if named != phase.source_images {
                return Err(Error::InvalidArgument(format!(
                    "phase {index}: fragment op {:?} reads image(s) {named:?} besides its own, \
                     and the decomposition records {:?}. The recorded list is what the executor \
                     fetches, what the DAG takes its edges from and what the fingerprint \
                     hashes, so a plan whose record disagrees with its op would price one image \
                     and read another.",
                    op.name(),
                    phase.source_images
                )));
            }
            for &image in &named {
                if is_supplied_image(image) {
                    // Handed to the run: no producing phase, so neither the
                    // bound, nor the forward reference, nor "which phase wrote
                    // it" is a question that can be asked of it. What is left is
                    // the extent, which is the same question for every image.
                    let stored = plan.volume_at(image);
                    if stored != read_volume {
                        return Err(Error::InvalidArgument(format!(
                            "phase {index}: fragment op {:?} reads {}, which is {stored:?},                              beside image {index}, which is {read_volume:?}. A supplied input is                              read at the block's own fetch region, so it has to be in the same                              coordinate space as the image the phase reads, and that space is                              image 0's.",
                            op.name(),
                            describe_image(image)
                        )));
                    }
                    continue;
                }
                if image >= plan.n_images() {
                    return Err(Error::InvalidArgument(format!(
                        "phase {index}: fragment op {:?} reads image {image}, and this plan has \
                         {} image(s), numbered 0 to {}.",
                        op.name(),
                        plan.n_images(),
                        plan.n_images() - 1
                    )));
                }
                if image > index {
                    return Err(Error::InvalidArgument(format!(
                        "phase {index}: fragment op {:?} reads image {image}, but image {image} \
                         is written by phase {}, which runs after it. Phases run in order, so a \
                         second image may only be one at or below the image this phase is \
                         handed — image {index} here.",
                        op.name(),
                        image - 1
                    )));
                }
                // **The check the chain half has no way to need.** Every phase
                // of a pixel plan writes the image above it, so an image at or
                // below the current one exists by construction. A fragment phase
                // writes one only if it says `writes_pixels`, so a plan can
                // name an image that no phase ever wrote — an array with no
                // producer, which the executor would read as whatever `prepare`
                // left behind.
                if image > 0 && image_written[image].is_none() {
                    return Err(Error::InvalidArgument(format!(
                        "phase {index}: fragment op {:?} reads image {image}, which phase {} \
                         did not write: it runs a fragment op that declares `writes_pixels() \
                         == false`. An image nothing wrote is not a second array, it is whatever \
                         `prepare` allocated.",
                        op.name(),
                        image - 1
                    )));
                }
                let stored = plan.volume_at(image);
                if stored != read_volume {
                    return Err(Error::InvalidArgument(format!(
                        "phase {index}: fragment op {:?} reads image {image}, which is \
                         {stored:?}, beside image {index}, which is {read_volume:?}. A source \
                         image is read at the block's own fetch region, so the two images have \
                         to be in one coordinate space; across grids the same integers would \
                         name different voxels.",
                        op.name()
                    )));
                }
            }
        }

        // ------------------------------------------------------- the seam --
        //
        // See `SeamFold`. The obligation is on any op that reads a second array,
        // and on any op that folds — over a non-zero fragment reach, which is
        // what makes a value cross a seam — a stream carrying such an array's
        // values.
        let inputs = op.inputs();
        let reads_operand = inputs
            .iter()
            .any(|input| carries_operand.contains(&(input.stream.clone(), input.phase)));
        let folds_operand = inputs.iter().any(|input| {
            input.reach != [0, 0, 0]
                && carries_operand.contains(&(input.stream.clone(), input.phase))
        });
        if !declared_sources.is_empty() || reads_operand {
            for output in op.outputs() {
                carries_operand.insert((output.stream, index));
            }
        }
        match op.seam_fold() {
            None if !declared_sources.is_empty() || folds_operand => {
                let because = if declared_sources.is_empty() {
                    "folds fragments carrying a second array's values over a non-zero reach"
                } else {
                    "reads a second array"
                };
                return Err(Error::InvalidArgument(format!(
                    "phase {index}: fragment op {:?} {because} and does not say what it does \
                     with values that straddle a seam. A region cut by a block boundary is \
                     summed in pieces and the pieces are combined in a merge; combine them \
                     with `f64` addition and the answer depends on the order the blocks merged \
                     in, so the same data cut two ways gives two numbers and neither looks \
                     wrong. Override `seam_fold` with `SeamFold::PerBlock` (nothing crosses a \
                     seam here), `SeamFold::Unordered` (the fold is associative in the type it \
                     accumulates in — the executor checks it) or `SeamFold::OrderDependent` \
                     (it is not, and this phase is therefore not decomposition-invariant). \
                     There is no default, for `Coverage`'s reason.",
                    op.name()
                )));
            }
            Some(SeamFold::PerBlock) if inputs.iter().any(|input| input.reach != [0, 0, 0]) => {
                let (stream, reach) = inputs
                    .iter()
                    .find(|input| input.reach != [0, 0, 0])
                    .map(|input| (input.stream.clone(), input.reach))
                    .expect("just found one");
                return Err(Error::InvalidArgument(format!(
                    "phase {index}: fragment op {:?} declares `SeamFold::PerBlock` — this \
                     block's fragment is a function of this block alone — and reaches {reach:?} \
                     block(s) for stream {stream:?}. A non-zero fragment reach *is* a fold over \
                     more than one block's values, so the two statements cannot both be true. \
                     An op that folds says `Unordered` or `OrderDependent`.",
                    op.name()
                )));
            }
            _ => {}
        }

        for input in op.inputs() {
            check_stream_name(&input.stream)?;
            if input.phase >= index {
                return Err(Error::InvalidArgument(format!(
                    "phase {index}: fragment op {:?} reads stream {:?} from phase {}, which \
                     is this phase or a later one. A fragment is read after it is written, \
                     and the DAG only orders earlier phases before later ones.",
                    op.name(),
                    input.stream,
                    input.phase
                )));
            }
            // **The other half of the address, and the half nothing checked.**
            // A stream written by two phases holds two generations, which is why
            // an input names a phase as well as a stream. The consequence is that
            // a wrong phase number is not out of range and does not fail: it
            // reads real fragments of the wrong generation, and the run answers
            // differently with no diagnostic anywhere. The forward reference
            // above is refused because it cannot possibly work; this refuses the
            // one that merely does not do what it says.
            //
            // **Scoped to the case this can be sure about**, and the scope is the
            // interesting part. It fires when *some* phase of this plan writes
            // the stream and the phase named is not one of them — the address
            // exists and points at the wrong generation, which is unambiguous. It
            // does not fire when no phase writes it at all, because that is not
            // by itself an error: `Environment::write_sidecar` is public, so a
            // stream may be seeded from outside the plan entirely, and a check
            // that demanded an in-plan producer would refuse a legitimate run to
            // catch a mistake that fails loudly at the first block anyway.
            //
            // It is here rather than in `Decomposition::check` for the same
            // reason `check_dtypes` is: a plan records op *names*, and only the
            // `(plan, work)` pair knows what each phase writes.
            let writers: Vec<usize> = work
                .iter()
                .enumerate()
                .filter(|(_, entry)| match entry {
                    PhaseWork::Fragments(producer) => producer
                        .outputs()
                        .iter()
                        .any(|out| out.stream == input.stream),
                    _ => false,
                })
                .map(|(at, _)| at)
                .collect();
            if !writers.is_empty() && !writers.contains(&input.phase) {
                let named = match work.get(input.phase) {
                    Some(PhaseWork::Fragments(producer)) => format!(
                        "runs fragment op {:?}, which writes {:?}",
                        producer.name(),
                        producer
                            .outputs()
                            .iter()
                            .map(|out| out.stream.clone())
                            .collect::<Vec<_>>()
                    ),
                    Some(PhaseWork::Pixels) => {
                        "runs chain slots, which write an image and no stream".to_string()
                    }
                    Some(PhaseWork::Iterate(producer)) => format!(
                        "runs iterative op {:?}, which writes an image and no stream",
                        producer.name()
                    ),
                    None => "is not in the work list at all".to_string(),
                };
                return Err(Error::InvalidArgument(format!(
                    "phase {index}: fragment op {:?} reads stream {:?} from phase {}, but phase \
                     {} {named}. {:?} is written by phase(s) {:?}. The phase is half the address \
                     — a stream written by two phases holds two generations — so naming the \
                     wrong one is a read of the wrong generation, which produces an answer \
                     rather than an error.",
                    op.name(),
                    input.stream,
                    input.phase,
                    input.phase,
                    input.stream,
                    writers
                )));
            }
            let source = &plan.phases[input.phase].grid;
            if source.volume() != phase.grid.volume() || source.block() != edge {
                return Err(Error::InvalidArgument(format!(
                    "phase {index}: fragment op {:?} reads stream {:?} written by phase {} on \
                     a different lattice ({:?}/{:?} against {:?}/{:?}). Fragments are keyed \
                     by block index, so two phases that exchange them must be cut the same \
                     way.",
                    op.name(),
                    input.stream,
                    input.phase,
                    source.volume(),
                    source.block(),
                    phase.grid.volume(),
                    edge
                )));
            }
            // The narrowest side granted on the worst block, because the
            // question is whether every task can see its neighbours' fragments —
            // a halo that is generous somewhere else does not answer it.
            let granted = phase
                .halo
                .in_voxels(edge)
                .granted_everywhere(phase.grid.volume());
            for axis in 0..3 {
                let counts = phase.grid.blocks_per_axis()[axis];
                let blocks = input.reach[axis].min(counts.saturating_sub(1));
                let wanted = blocks
                    .saturating_mul(edge[axis])
                    .min(phase.grid.volume()[axis]);
                if granted[axis] < wanted {
                    return Err(Error::InvalidArgument(format!(
                        "phase {index}: fragment op {:?} reaches {} block(s) on axis {axis} \
                         for stream {:?}, which is {wanted} voxel(s), but the phase halo is \
                         {}. The halo is what makes the neighbours' tasks dependencies of \
                         this one, so a short halo would read fragments nobody has written \
                         yet. Build the phase with `fragment_phase`.",
                        op.name(),
                        input.reach[axis],
                        input.stream,
                        granted[axis]
                    )));
                }
            }
        }
    }
    Ok(())
}

// -------------------------------------------------------- coverage guard --

/// What one stream of a fragment phase actually holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamCoverage {
    pub stream: String,
    pub phase: usize,
    pub declared: Coverage,
    /// Distinct blocks that wrote a fragment.
    pub blocks: usize,
    /// Blocks in the phase's lattice.
    pub lattice: usize,
}

impl StreamCoverage {
    pub fn describe(&self) -> String {
        format!(
            "{:?} phase {}: {} of {} block(s), declared {}",
            self.stream,
            self.phase,
            self.blocks,
            self.lattice,
            self.declared.as_str()
        )
    }
}

/// The guard on a fragment phase's **output**, run once after its last task.
///
/// Checked against the store rather than against the executor's own record of
/// what it wrote, for the same reason `execute_phases` re-checks the tiling of
/// what was actually written: a plan that promised a fragment per block and an
/// executor that wrote something else would otherwise agree with each other.
///
/// Two things are checked, and the second is the one an area-only coverage
/// check is missing — the defect that lets two disjoint boxes summing to the
/// right total pass while one of them sits outside the volume:
///
/// * **coverage** — for a [`Coverage::EveryBlock`] stream, the block set in the
///   store equals the lattice, and the first missing block is named;
/// * **containment** — every key names a block of *this* lattice. Without it a
///   stream could hold the right *number* of fragments while some sat outside
///   the grid, which is exactly how a total-only check passes on wrong data.
pub fn check_fragment_coverage(
    env: &dyn Environment,
    plan: &Decomposition,
    phase: usize,
    op: &dyn FragmentOp,
) -> Result<Vec<StreamCoverage>> {
    let grid = &plan
        .phases
        .get(phase)
        .ok_or_else(|| {
            Error::InvalidArgument(format!(
                "phase {phase} of a decomposition with {}",
                plan.n_phases()
            ))
        })?
        .grid;
    let counts = grid.blocks_per_axis();
    let lattice: std::collections::BTreeSet<[usize; 3]> =
        grid.cores().into_iter().map(|core| core.index).collect();
    let mut report = Vec::new();
    for output in op.outputs() {
        let mut written = std::collections::BTreeSet::new();
        for key in env.sidecar_keys(&output.stream)? {
            if key.phase != phase {
                continue;
            }
            if (0..3).any(|axis| key.block[axis] >= counts[axis]) {
                return Err(Error::InvalidArgument(format!(
                    "stream {:?} holds a fragment for block {:?} of phase {phase}, which is \
                     outside that phase's lattice of {counts:?} blocks. A fragment keyed \
                     outside the grid is not merely extra — it means the block index a task \
                     wrote under is not the index the plan gave it.",
                    output.stream, key.block
                )));
            }
            written.insert(key.block);
        }
        if output.coverage == Coverage::EveryBlock && written != lattice {
            let missing: Vec<[usize; 3]> = lattice.difference(&written).copied().collect();
            return Err(Error::InvalidArgument(format!(
                "stream {:?} declares every-block coverage for phase {phase} but holds {} of \
                 {} fragment(s); {} block(s) wrote none, first {:?}. This is the fragment \
                 side of the tiling guard: the phase's valid regions tile the volume by \
                 construction and say nothing about the fragments, so a hole here is a hole \
                 the pixel check cannot see.",
                output.stream,
                written.len(),
                lattice.len(),
                missing.len(),
                missing.first()
            )));
        }
        report.push(StreamCoverage {
            stream: output.stream.clone(),
            phase,
            declared: output.coverage,
            blocks: written.len(),
            lattice: lattice.len(),
        });
    }
    Ok(report)
}

// -------------------------------------------------------------- payloads --

/// A fragment as little-endian `u64`s.
///
/// Offered because every probe in this crate wants the same shape and writing
/// it four times would be four chances to get it subtly different — **not**
/// because the store knows this encoding. A fragment is bytes; what they mean
/// is the caller's, which is the whole point of taking bytes.
pub fn pack_u64(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// The other half of [`pack_u64`]. A length that is not a multiple of eight is
/// a truncated fragment and says so.
pub fn unpack_u64(bytes: &[u8]) -> Result<Vec<u64>> {
    if bytes.len() % 8 != 0 {
        return Err(Error::InvalidArgument(format!(
            "a packed fragment is a whole number of 8-byte words; this one is {} byte(s)",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|word| u64::from_le_bytes(word.try_into().expect("eight bytes")))
        .collect())
}

// ----------------------------------------------------------------- merge --

/// Visit every fragment of `stream`, one at a time, in key order.
///
/// **This is the streaming merge, and streaming is the point.**
/// `Environment::sidecar_fragments` returns the whole stream as a `Vec`, which
/// makes every fragment of every block resident at once — the same
/// whole-answer-in-memory problem that per-block fragments exist to avoid, just
/// moved to the reader. This lists the keys and reads one fragment at a time,
/// so a reduction's residency is its accumulator plus one fragment.
///
/// For a reduction that wants to be a *phase* rather than a loop after the run,
/// see the module header: it is an op with a full-volume reach, and
/// [`BlockView::stream_fragments`] is this function's counterpart inside one.
pub fn fold_fragments(
    env: &dyn Environment,
    stream: &str,
    visit: &mut dyn FnMut(&FragmentKey, &[u8]) -> Result<()>,
) -> Result<usize> {
    let mut seen = 0usize;
    for key in env.sidecar_keys(stream)? {
        let Some(bytes) = env.read_sidecar(&key.stream, key.phase, key.block)? else {
            // Listed and then gone. The same coordination bug `Sidecars`
            // refuses to paper over, refused the same way.
            return Err(Error::Backend(format!(
                "sidecar fragment {key:?} was listed and then could not be read. A stream \
                 must not be discarded while it is being merged."
            )));
        };
        visit(&key, &bytes)?;
        seen += 1;
    }
    Ok(seen)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Nothing;

    impl FragmentOp for Nothing {
        fn name(&self) -> &'static str {
            "nothing"
        }

        fn outputs(&self) -> Vec<FragmentOutput> {
            vec![FragmentOutput::new(
                "out",
                Lifecycle::DeleteOnExit,
                Coverage::EveryBlock,
            )]
        }

        fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
            Ok(BlockOutput::nothing())
        }
    }

    struct Reaching(usize, [usize; 3]);

    impl FragmentOp for Reaching {
        fn name(&self) -> &'static str {
            "reaching"
        }

        fn inputs(&self) -> Vec<FragmentInput> {
            vec![FragmentInput::own("in", self.0).with_reach(self.1)]
        }

        fn outputs(&self) -> Vec<FragmentOutput> {
            vec![FragmentOutput::new(
                "out",
                Lifecycle::DeleteOnExit,
                Coverage::EveryBlock,
            )]
        }

        fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
            Ok(BlockOutput::nothing())
        }
    }

    /// Reads everything and writes pixels: the global step, declared.
    struct Global;

    impl FragmentOp for Global {
        fn name(&self) -> &'static str {
            "global"
        }

        fn reach(&self, _axis: usize, volume_len: usize) -> usize {
            volume_len
        }

        fn writes_pixels(&self) -> bool {
            true
        }

        fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
            Ok(BlockOutput::nothing().with_pixels(at.output_buffer(0.0)?))
        }
    }

    #[test]
    fn a_zero_reach_neighbourhood_is_the_block_itself() {
        assert_eq!(neighbourhood([2, 0, 0], [0, 0, 0], [8, 1, 1]), [[2, 0, 0]]);
        assert_eq!(neighbourhood_size([2, 0, 0], [0, 0, 0], [8, 1, 1]), 1);
    }

    #[test]
    fn a_neighbourhood_clamps_at_the_lattice_edge() {
        assert_eq!(
            neighbourhood([0, 0, 0], [1, 0, 0], [4, 1, 1]),
            [[0, 0, 0], [1, 0, 0]]
        );
        assert_eq!(
            neighbourhood([3, 0, 0], [1, 0, 0], [4, 1, 1]),
            [[2, 0, 0], [3, 0, 0]]
        );
        assert_eq!(
            neighbourhood([2, 0, 0], [1, 0, 0], [4, 1, 1]),
            [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
        );
        for index in 0..4 {
            assert_eq!(
                neighbourhood_size([index, 0, 0], [1, 0, 0], [4, 1, 1]),
                neighbourhood([index, 0, 0], [1, 0, 0], [4, 1, 1]).len()
            );
        }
    }

    #[test]
    fn a_fragment_reach_becomes_a_halo_of_whole_blocks() {
        let grid = BlockGrid::new([32, 4, 4], [8, 4, 4]).unwrap();
        let phase = fragment_phase(&Reaching(0, [1, 0, 0]), grid).unwrap();
        assert_eq!(phase.halo, [8, 0, 0]);
        // and the phase's *reach* stays zero, so the valid regions are the cores
        assert_eq!(phase.reach, [0, 0, 0]);
        for block in &phase.blocks {
            assert_eq!(block.valid, block.core);
        }
    }

    #[test]
    fn a_zero_reach_phase_gets_no_halo_at_all() {
        let grid = BlockGrid::new([32, 4, 4], [8, 4, 4]).unwrap();
        let phase = fragment_phase(&Nothing, grid).unwrap();
        assert_eq!(phase.halo, [0, 0, 0]);
        for block in &phase.blocks {
            assert_eq!(block.read, block.core);
        }
    }

    /// The derivation the module header states, checked rather than asserted in
    /// prose: a full reach with a short halo loses every interior block's core.
    #[test]
    fn a_full_reach_phase_only_tiles_when_it_reads_everything() {
        let volume = [32usize, 4, 4];
        let short = PhaseDecomposition::derive(
            Vec::new(),
            Vec::new(),
            volume,
            [8, 4, 4],
            BlockGrid::new(volume, [8, 4, 4]).unwrap(),
        );
        assert_eq!(short.blocks_missing_valid_core().len(), 4);

        let whole = fragment_phase(&Global, BlockGrid::new(volume, [8, 4, 4]).unwrap()).unwrap();
        assert_eq!(whole.reach, volume);
        assert_eq!(whole.halo, volume);
        assert!(whole.blocks_missing_valid_core().is_empty());
        for block in &whole.blocks {
            assert_eq!(block.read, Region::whole(&volume));
            assert_eq!(block.valid, block.core);
        }
    }

    #[test]
    fn a_fragment_only_plan_tiles_its_lattice() {
        let plan = fragment_only([32, 4, 4], [8, 4, 4], Dtype::F64, &[&Nothing]).unwrap();
        assert_eq!(plan.n_phases(), 1);
        assert_eq!(plan.n_tasks(), 4);
        plan.check().unwrap();
    }

    #[test]
    fn a_phase_may_not_read_a_stream_from_itself_or_later() {
        // reading phase 0 from phase 1 is fine
        let backwards = Reaching(0, [0, 0, 0]);
        let plan =
            fragment_only([32, 4, 4], [8, 4, 4], Dtype::F64, &[&Nothing, &backwards]).unwrap();
        check_phase_work(
            &plan,
            &[
                PhaseWork::Fragments(&Nothing),
                PhaseWork::Fragments(&backwards),
            ],
        )
        .unwrap();

        // reading phase 1 *from* phase 1 is not
        let itself = Reaching(1, [0, 0, 0]);
        let error = check_phase_work(
            &plan,
            &[
                PhaseWork::Fragments(&Nothing),
                PhaseWork::Fragments(&itself),
            ],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("this phase or a later one"), "{error}");
    }

    #[test]
    fn a_phase_may_not_follow_one_that_wrote_no_image() {
        let plan = fragment_only([32, 4, 4], [8, 4, 4], Dtype::F64, &[&Nothing, &Nothing]).unwrap();
        let error = check_phase_work(&plan, &[PhaseWork::Fragments(&Nothing), PhaseWork::Pixels])
            .unwrap_err()
            .to_string();
        assert!(error.contains("did not write"), "{error}");
        // but a fragment op that hands pixels on is allowed to be followed
        let onward = fragment_only([32, 4, 4], [8, 4, 4], Dtype::F64, &[&Global, &Global]).unwrap();
        check_phase_work(&onward, &[PhaseWork::Fragments(&Global); 2]).unwrap();
    }

    #[test]
    fn packing_round_trips_and_a_truncated_fragment_is_refused() {
        let packed = pack_u64(&[1, 2, 3, u64::MAX]);
        assert_eq!(packed.len(), 32);
        assert_eq!(unpack_u64(&packed).unwrap(), vec![1, 2, 3, u64::MAX]);
        assert!(unpack_u64(&packed[..31]).is_err());
    }
}
