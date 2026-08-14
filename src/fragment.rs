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
// fragments that assertion is about a level nobody wrote: `valid == core` by
// construction, cores tile by construction, and the check passes without
// constraining the fragments at all. A guard that cannot fail is worse than no
// guard, because it is trusted.
//
// So a fragment stream declares its [`Coverage`], with no default, and
// [`check_fragment_coverage`] runs after the phase's last task — against the
// *store*, so that what is checked is what landed. A phase that writes no pixel
// level and declares no every-block stream is refused at plan time, because
// nothing about it would be checkable at all.
//
// What is still the caller's
// --------------------------
// Deciding what the bytes mean, and choosing whether the reduction runs as a
// full-reach phase or as a plain loop after the run. [`fold_fragments`] serves
// the second, and it *streams*: one fragment resident at a time, by key.
// `Environment::sidecar_fragments` materialises a whole stream and is the wrong
// tool for anything at scale.

use std::collections::BTreeMap;

use crate::decomposition::{Decomposition, PhaseDecomposition};
use crate::dtype::Dtype;
use crate::env::{BlockBuf, Environment};
use crate::error::{Error, Result};
use crate::geometry::BlockGrid;
use crate::op::Anchor;
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
/// For a phase that writes a pixel level that is a statement about the output.
/// For a phase whose output is fragments it is a statement about a level nobody
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
    /// The element type of the level this phase **writes**, which is what
    /// [`Self::output_buffer`] allocates at. Carried rather than assumed: a
    /// buffer of the wrong width is a buffer the write refuses, and the executor
    /// is the only thing that knows which level this block is destined for.
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

    /// A buffer shaped like this block's read extent, filled with `value`.
    ///
    /// Routed through the environment on purpose: it is the one construction an
    /// op needs that differs between a run that holds data and a run that only
    /// counts, and going through `Environment::constant` is what lets one op
    /// serve both. A `writes_pixels` op that built an array directly would be
    /// unsimulatable.
    ///
    /// The element type is the one the level this phase writes holds; see
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
    /// `true` makes the phase produce a level like any other, so an ordinary
    /// pixel phase may follow it. `false` means the phase writes fragments and
    /// nothing else, and is therefore terminal as far as levels are concerned.
    fn writes_pixels(&self) -> bool {
        false
    }

    /// What element type this op writes, given the one it reads.
    ///
    /// The counterpart of [`BlockOp::produces`](crate::op::BlockOp::produces),
    /// and it exists for the same reason: `check_dtypes` folds a plan's element
    /// types from level 0 and refuses a level allocated at a width its producer
    /// does not write. That fold runs over the *chain*, and a fragment phase owns
    /// no slot of the chain — so before this method existed there was nothing to
    /// fold and a fragment op that changed the width was refused with a message
    /// about ops it does not have.
    ///
    /// The default hands the type on unchanged, which is right for every
    /// fragment op this crate shipped before the method existed and is the safe
    /// default in the same way `BlockOp`'s is: an op that says nothing keeps
    /// exactly the contract it already had. Only consulted when
    /// [`Self::writes_pixels`] is true; a phase that writes no level has no
    /// level whose width could be wrong.
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

    /// Produce this block's output.
    ///
    /// Returning no fragment at all is legitimate — a block with nothing to say
    /// writes nothing, which is exactly what an absent key means to a reader.
    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput>;
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
    /// `region -> region` like `Pixels`, and it reads and writes a level the same
    /// way; what differs is that a block of it is visited an unknown number of
    /// times before the level is written, and that every visit is handed more
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

    /// Whether this phase produces the level after it.
    pub fn writes_a_level(&self) -> bool {
        match self {
            PhaseWork::Pixels | PhaseWork::Iterate(_) => true,
            PhaseWork::Fragments(op) => op.writes_pixels(),
        }
    }

    /// Whether this phase reads the level before it.
    pub fn reads_a_level(&self) -> bool {
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
/// The halo is the widest of the op's own reach and every input's fragment
/// reach converted to voxels. Where the op's reach is smaller than the halo the
/// valid regions equal the cores and tile the volume; where it is the whole
/// volume, only `halo == volume` survives the tiling check, which is the
/// module header's point about a global step.
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
    Ok(PhaseDecomposition::derive(
        Vec::new(),
        Vec::new(),
        reach,
        halo,
        grid,
    ))
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
    // Level 0 is the workflow input and always exists; level p+1 exists iff
    // phase p wrote it. A phase that reads no pixels needs neither.
    let mut level_written: Vec<Option<usize>> = vec![None; work.len() + 1];
    for (index, entry) in work.iter().enumerate() {
        let phase = &plan.phases[index];
        if entry.reads_a_level() && index > 0 && level_written[index].is_none() {
            return Err(Error::InvalidArgument(format!(
                "phase {index} reads level {index}, which phase {} did not write: it runs a \
                 fragment op that declares `writes_pixels() == false`. A phase that writes \
                 only fragments is terminal as far as levels go; an op that hands pixels on \
                 says `writes_pixels`.",
                index - 1
            )));
        }
        if entry.writes_a_level() {
            level_written[index + 1] = Some(index);
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
            // running operand of substage 0 is the level below read through the
            // same block geometry. A phase that resized or re-gridded between the
            // two would be handing substage 1 an operand of a different shape
            // than substage 0 produced, so the iteration would not close.
            let below = plan.volume_at(index);
            if phase.volume() != below || phase.reads_across_grids() {
                return Err(Error::InvalidArgument(format!(
                    "phase {index} runs iterative op {:?} and reads a {below:?} level to write a \
                     {:?} one. An iteration feeds its own output back in, so its substages must \
                     agree on one extent and one lattice; a phase that changes either is a \
                     transformation and belongs before or after the iteration, not inside it.",
                    op.name(),
                    phase.volume()
                )));
            }
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
        // The gap this closes: a phase that writes no pixel level is not
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
                "phase {index}: fragment op {:?} writes no pixel level and declares no \
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
                        "runs chain slots, which write a level and no stream".to_string()
                    }
                    Some(PhaseWork::Iterate(producer)) => format!(
                        "runs iterative op {:?}, which writes a level and no stream",
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
    fn a_phase_may_not_follow_one_that_wrote_no_level() {
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
