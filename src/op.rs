// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The arrangement this replaces is a block-processed chain written as one
// function, whose halo is a hand-maintained formula sitting in a different
// place from the code it describes. Two things then drift apart with nobody
// noticing: the stencil an op actually uses, and the number somebody wrote down
// for it. The abstraction here puts reach, execution and traversal preference
// on one type, and derives the chain's reach by folding over the same tree the
// executor walks — so the formula *is* the code.
//
// The invariant this file exists to make structural
// -------------------------------------------------
// You cannot add an op to execution and forget it in reach, because there is
// **one** structure, not two. `Chain::reach` and `Chain::apply` recurse over
// the same `Chain` value; a node that contributes to one necessarily
// contributes to the other.
//
// Nothing here calls a real kernel. `BlockOp::apply` is the seat a thin adapter
// over a translated kernel will occupy later; the framework is proven first, so
// that a seam failure can be attributed to the framework or the kernel rather
// than to "somewhere in the pipeline".

use ndarray::ArrayD;

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::geometry::BlockGeometry;
use crate::reach::Reach;
use crate::region::Region;
use crate::voxels::Voxels;

/// Where the buffer handed to [`BlockOp::apply`] sits inside the volume.
///
/// **Why this is a parameter and not an implicit.** `reach` already takes
/// `volume_len` "rather than an implicit so the caller must decide which, and a
/// reviewer can see the decision" — an op anchored to the global grid answers
/// differently from one anchored to its block. Execution has exactly the same
/// question and had no way to ask it: `apply(&input, &mut out)` can only see the
/// buffer, so an op whose arithmetic depends on the global grid (the sampled
/// stages of a chain: adaptive thresholding, histogram equalization, sampled
/// background estimation) would silently re-anchor to the block and produce a
/// complete, well-formed, wrong volume — the failure mode this module exists to
/// remove. Making it an argument forces every op to answer, and the answer is
/// visible at the call site.
///
/// A position-independent op simply ignores it. That is the common case and
/// costs nothing; what it no longer does is *assume* it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Anchor {
    /// Lower corner of the buffer inside the volume.
    pub offset: [usize; 3],
    /// Shape of the whole volume.
    pub volume: [usize; 3],
}

impl Anchor {
    /// The buffer is the whole volume.
    pub fn whole(volume: [usize; 3]) -> Self {
        Self {
            offset: [0, 0, 0],
            volume,
        }
    }

    pub fn new(offset: [usize; 3], volume: [usize; 3]) -> Self {
        Self { offset, volume }
    }

    /// The anchor of a buffer holding `region` of a `volume`-shaped array.
    pub fn of_region(region: &Region, volume: [usize; 3]) -> Result<Self> {
        if region.start.len() != 3 {
            return Err(Error::InvalidArgument(format!(
                "block ops are 3-D, got a region of rank {}",
                region.start.len()
            )));
        }
        Ok(Self {
            offset: [region.start[0], region.start[1], region.start[2]],
            volume,
        })
    }

    /// Whether a buffer of `shape` anchored here *is* the whole volume, in
    /// which case a global-grid op and a block-local one agree by definition.
    pub fn is_whole(&self, shape: &[usize]) -> bool {
        self.offset == [0, 0, 0] && shape == self.volume
    }
}

/// One array a workflow writes.
///
/// **Why an op declares more than one.** A `Workflow` names a single output of a
/// single element type, and the executor writes one buffer to one level. An
/// operation that produces several results — a labelling plus the scores it was
/// thresholded from, a segmentation plus its boundaries — then has nowhere to
/// put the rest, so they are written on the side by whatever is holding the
/// storage, and the framework's own byte figure is short by however much they
/// came to. Measured on the patch-lattice harness: **95.2 MB counted against
/// 158.6 MB written**, a factor of 1.67. Declaring them here puts them back in
/// the accounting, in the guard and in the event stream.
///
/// **Rank is a `Vec`, not `[usize; 3]`, and that is the point.** The array an op
/// writes beside a volume is often not a volume: one row per object, one score
/// per class per output position. [`Region`] and
/// [`crate::tiling::boxes_tile_exactly`] are both rank-generic already, so a
/// side output of rank 2 or 5 costs nothing extra to place or to check. What is
/// still rank 3 is the *block geometry* — the lattice the phase is cut from —
/// and that is a different question, owned by a different change.
///
/// **The name is a `String` rather than an `ArrayRef`.** `ArrayRef` belongs with
/// the workflow, and an op must be able to declare what it writes without
/// depending on one; the environment resolves both the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output {
    pub name: String,
    pub dtype: Dtype,
    /// The whole array's shape. Any rank.
    pub shape: Vec<usize>,
}

impl Output {
    pub fn new(name: impl Into<String>, dtype: Dtype, shape: &[usize]) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    /// Bytes the whole array occupies, decoded.
    pub fn bytes(&self) -> u64 {
        self.shape.iter().product::<usize>() as u64 * self.dtype.size_of() as u64
    }
}

/// Everything a block's side outputs are positioned by.
///
/// One struct rather than four arguments, because these are all answers to the
/// same question — *which part of what I was handed goes where* — and because
/// the set will grow when a coordinate space becomes something a reach carries.
///
/// `within` is the piece that cannot be derived. The buffer an op is handed is
/// the **read** extent, halo included; a side output is written once per output
/// voxel, so the op must know which sub-box of its buffer is the trustworthy
/// one. `at` gives the buffer's position and `regions` gives the destinations,
/// but neither says where `valid` sits inside `read` — that is a fact about the
/// halo, which belongs to the plan.
#[derive(Debug, Clone, Copy)]
pub struct SideBlock<'a> {
    /// Where the buffer sits in the volume it was read from.
    pub at: &'a Anchor,
    /// The trustworthy sub-box of the buffer, **relative to the buffer**.
    pub within: &'a Region,
    /// One per declared side output, in declaration order: where it lands, in
    /// that output's own coordinate space.
    pub regions: &'a [Region],
}

/// What an op requires of the blocks it is handed.
///
/// **Verified, not constructed.** A hook that returned a [`BlockGrid`] would
/// support exactly the easy case: `BlockGrid::cores` builds `start = index *
/// block`, so it can only describe a lattice that is disjoint and evenly
/// strided. Real mandated lattices are not always that — a lattice spread evenly
/// across an extent is *overlapping* and *unevenly spaced*, and no `BlockGrid`
/// will ever produce one. So this is a predicate over the regions a plan
/// actually hands the op, which can state both kinds; whether a plan satisfying
/// it can be *built* is then a separate, visible question rather than a silently
/// unsupported one.
///
/// **What is constrained is the region the op is handed** — `BlockGeometry`'s
/// `source`, which the executor already requires to be the shape of `read`. Not
/// the core: a plan can satisfy a constraint on cores and still grow a halo and
/// hand the op a different extent, which is the silent kind of wrongness this
/// crate is arranged against. The two coincide exactly when the phase's reach is
/// zero, which is the case a mandated input extent arises in.
///
/// [`BlockGrid`]: crate::geometry::BlockGrid
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockConstraint {
    /// Every block must be handed exactly this extent.
    ///
    /// An **anisotropic** shape, which is the form a mandate actually takes and
    /// which `Constraints::block_candidates` — a list of scalar edges — cannot
    /// express at all. A planner that honours this builds
    /// `BlockGrid::new(volume, extent)` directly instead of choosing from the
    /// menu.
    Extent([usize; 3]),
    /// The blocks must be exactly these regions, in this order.
    ///
    /// The escape hatch for a lattice that is not a grid. Nothing that derives
    /// its blocks from a `BlockGrid` can satisfy it — cores are `index * block`
    /// and must tile — so every shipped strategy **refuses** rather than
    /// producing a plan that passes its own guard and cannot run. What *can*
    /// satisfy it is a phase whose blocks carry an explicit `source`
    /// (`PhaseDecomposition::with_sources`): the block lattice stays the unit
    /// grid of an index space and the overlap lives in the mapping, which is the
    /// representation such a lattice has anyway.
    Regions(Vec<Region>),
}

impl BlockConstraint {
    /// The grid that satisfies this constraint over `volume`, if one exists.
    ///
    /// `None` is not a failure to try: it says no `BlockGrid` can produce these
    /// blocks, which is a fact about the constraint.
    pub fn grid(&self, volume: [usize; 3]) -> Option<crate::geometry::BlockGrid> {
        match self {
            Self::Extent(extent) => crate::geometry::BlockGrid::new(volume, *extent).ok(),
            Self::Regions(_) => None,
        }
    }

    /// The grid **and the halo** that satisfy this constraint over `volume` for
    /// an operation that also reaches.
    ///
    /// **This is where "the extent I accept" stops being "the extent I need
    /// around it".** With one symmetric integer per axis they were the same
    /// number and the two demands were not jointly satisfiable: a block at a
    /// volume edge had its read clamped and was handed something narrower than
    /// an interior block, and a block in the middle was handed its core plus
    /// twice the reach rather than the extent asked for. Both are fixed by
    /// separating the two quantities:
    ///
    /// * the **cores** are cut small enough to leave room — `extent` less what
    ///   the reach needs on each side — so an interior block's read is the
    ///   extent exactly;
    /// * the **halo** is a per-block window ([`Reach::window`]) that slides
    ///   inward at the ends instead of being clipped, so an edge block is handed
    ///   the extent too. It writes only its own core, so nothing is written
    ///   twice and the tiling check is unaffected.
    ///
    /// `Ok(None)` says no `BlockGrid` can produce these blocks, which is a fact
    /// about the constraint rather than a failure to try. `Err` says the extent
    /// cannot be met over this volume at all.
    pub fn lattice(
        &self,
        volume: [usize; 3],
        reach: &Reach,
    ) -> Result<Option<(crate::geometry::BlockGrid, Reach)>> {
        match self {
            Self::Regions(_) => Ok(None),
            Self::Extent(extent) => {
                let reach = reach.in_voxels(*extent);
                let mut block = [0usize; 3];
                for axis in 0..3 {
                    let (lo, hi) = reach.axis(axis).bound(volume[axis]);
                    block[axis] = extent[axis]
                        .checked_sub(lo + hi)
                        .filter(|&edge| edge > 0)
                        .ok_or_else(|| {
                            Error::InvalidArgument(format!(
                                "this op accepts exactly {extent:?} and reaches {lo}+{hi} on axis \
                                 {axis}, which leaves no voxel it could be trusted for. An \
                                 operation that reads its whole input to produce nothing is not a \
                                 block operation.",
                            ))
                        })?;
                }
                let grid = crate::geometry::BlockGrid::new(volume, block)?;
                let halo = Reach::window(&grid, *extent)?;
                Ok(Some((grid, halo)))
            }
        }
    }

    /// Are these the blocks the op accepts?
    ///
    /// `what` names the plan under test, so a refusal says which phase of which
    /// decomposition failed rather than only that something did.
    pub fn check(&self, blocks: &[BlockGeometry], what: &str) -> Result<()> {
        match self {
            Self::Extent(extent) => {
                for block in blocks {
                    if block.source.shape != extent {
                        return Err(Error::InvalidArgument(format!(
                            "{what}: block {:?} is handed {:?} and this op accepts exactly \
                             {extent:?}. A block shape is not a preference here — an op that \
                             states one cannot run on anything else, so a plan offering \
                             something else is refused rather than run.",
                            block.index, block.source.shape
                        )));
                    }
                }
                Ok(())
            }
            Self::Regions(regions) => {
                if blocks.len() != regions.len() {
                    return Err(Error::InvalidArgument(format!(
                        "{what}: the plan has {} blocks and this op mandates {}. The mandated \
                         lattice is not a block grid — it is stated region by region — so a \
                         plan whose blocks come from a `BlockGrid` can only match it by \
                         carrying an explicit `source` per block.",
                        blocks.len(),
                        regions.len()
                    )));
                }
                for (block, wanted) in blocks.iter().zip(regions) {
                    if &block.source != wanted {
                        return Err(Error::InvalidArgument(format!(
                            "{what}: block {:?} is handed {:?}+{:?} and this op mandates \
                             {:?}+{:?}. The mandated lattice is stated region by region and \
                             matched in block order.",
                            block.index,
                            block.source.start,
                            block.source.shape,
                            wanted.start,
                            wanted.shape
                        )));
                    }
                }
                Ok(())
            }
        }
    }
}

/// One operation in a block-processed chain.
///
/// `Send + Sync` is required rather than incidental: the executor runs blocks
/// concurrently and shares one `&dyn BlockOp` across workers.
pub trait BlockOp: Send + Sync {
    fn name(&self) -> &'static str;

    /// Voxels read beyond the voxel written, along `axis`.
    ///
    /// `volume_len` is the extent of the axis in the **coordinate system the op
    /// is anchored to** — global under global grid anchoring, not the block's.
    /// It is a parameter rather than an implicit so the caller must decide
    /// which, and a reviewer can see the decision.
    ///
    /// This must be computed *independently of the configured halo*. If the
    /// plan's halo feeds the reach, the guard in `decomposition.rs` compares a number
    /// against itself and cannot fire.
    ///
    /// This is the **degenerate** statement — symmetric, the same for every
    /// block, in the phase's own voxels — and it has no default for the reason
    /// above: it is the one place a silent zero would produce a complete,
    /// well-formed, wrong volume. An operation with something more exact to say
    /// says it in [`Self::reach_spec`], and that is what the plan uses.
    fn reach(&self, axis: usize, volume_len: usize) -> usize;

    /// The full statement of what this operation reads: one-sided, per-block,
    /// whole-axis, and in a named coordinate space.
    ///
    /// **Defaulted, and the default is what [`Self::reach`] says.** Most
    /// operations mean exactly the symmetric per-axis integer — `src/ops/`
    /// derives several of them tight to a single voxel from the element they are
    /// stated over — and none of them changed when this method appeared. What
    /// changes is what an operation *can* say when the symmetric integer is a
    /// lie in the safe direction: a lattice dependency that is one-sided costs
    /// **3.27x** in fetches when it is declared on both sides, an unevenly
    /// spread lattice has a different voxel footprint per block, and "reaches
    /// everything" was indistinguishable from "reaches 4096".
    ///
    /// An implementation that overrides this must keep [`Self::reach`] a valid
    /// symmetric bound on it; `Chain::reach_spec` checks that rather than
    /// trusting it, because two statements of one quantity are two statements
    /// that can drift.
    fn reach_spec(&self, volume: [usize; 3]) -> Reach {
        Reach::symmetric([
            self.reach(0, volume[0]),
            self.reach(1, volume[1]),
            self.reach(2, volume[2]),
        ])
    }

    /// Compute the op over the whole of `input`, writing into `out`.
    ///
    /// `out` is [`Self::output_shape`] of `input`'s shape and holds
    /// [`Self::produces`] of `input`'s element type — the caller allocates it
    /// from what this op *declared*, so the two cannot disagree. Values near the
    /// array edge are the op's own business: it sees a clamped read extent at a
    /// real volume boundary and must be defined there.
    ///
    /// `at` says where `input` sits in the volume; see [`Anchor`] for why it is
    /// an argument. A position-independent op ignores it.
    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()>;

    // ------------------------------------------------- what it can handle --
    //
    // An op that cannot work in the element type it is handed must be refused
    // when the plan is *made*, not discovered when a block reaches it. The two
    // methods below are the same arrangement `block_constraint` uses: declared
    // by the op, folded by the `Chain`, consulted by `decompose` and re-checked
    // by `execute`, because a plan may arrive from any strategy or off a wire.

    /// Can this op be handed a block of `dtype`?
    ///
    /// **The default is `f64` and only `f64`**, which is what every op in this
    /// crate was before the element type became a tag — so an op that says
    /// nothing keeps exactly the contract it already had. The alternative
    /// default, "anything", would let an op that only handles one type pass the
    /// plan-time check and fail at run time, which is the failure this method
    /// exists to remove.
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    /// What this op writes, given what it reads.
    ///
    /// Only ever called for a `dtype` this op [`accepts`](Self::accepts). The
    /// default hands the type on unchanged, which is right for every op whose
    /// output is the same kind of thing as its input; a thresholding op that
    /// narrows to `bool`, or a statistic that widens to `f64`, says so here and
    /// the level it writes is allocated at that width.
    fn produces(&self, input: Dtype) -> Dtype {
        input
    }

    /// The shape of the block this op produces, from the shape it is handed.
    ///
    /// **Declared rather than assumed**, which is the point. Before this
    /// existed, `Environment::apply` allocated the output as the input's own
    /// shape, so a phase could translate its read but never resize it, and
    /// `run_task` had to refuse every plan whose fetch extent was not its write
    /// extent (`strategy.rs`, the "cross-grid read that resizes" refusal). The
    /// executor now compares what the phase *says* it produces against the read
    /// extent the plan derived, so a decimating or upsampling phase is a plan
    /// that checks rather than a plan that is turned away.
    ///
    /// A function of the shape and nothing else — never of the data — for the
    /// reason a `Decomposition` is data-blind: two datasets must not produce two
    /// different plans.
    fn output_shape(&self, input: [usize; 3]) -> [usize; 3] {
        input
    }

    /// Axes ordered slowest- to fastest-varying, as this op prefers to be
    /// traversed. Purely **advisory**: a wrong answer costs locality, never
    /// correctness. `None` means "no preference".
    fn preferred_iteration(&self) -> Option<[usize; 3]> {
        None
    }

    /// If every voxel the op would read is `value`, is the output a known
    /// constant? `None` means "not known", which disables the short circuit.
    ///
    /// The default is `None`, so an op that says nothing is never skipped. That
    /// asymmetry is deliberate: the short circuit fires only where an op has
    /// *stated* the mapping, so a skipped block produces exactly what computing
    /// it would have.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        None
    }

    /// Relative compute cost per voxel processed, in whatever unit the
    /// `CostModel` uses for its read and write weights.
    ///
    /// **This must be measured, not guessed.** `docs/design/BLOCK_OPS.md` is
    /// explicit that the planner will confidently return the optimal schedule
    /// *for its cost model*, and that the model being wrong is the risk rather
    /// than the search. The default of 1.0 is a placeholder that says "one unit
    /// of work", not a measurement.
    fn cost_per_voxel(&self) -> f64 {
        1.0
    }

    // ------------------------------------------------- several outputs --
    //
    // The primary result is what `apply` writes and what the next op in a
    // fused phase consumes. The arrays below are **terminal**: nothing reads
    // them back inside the chain, because there is no fan-in. That asymmetry
    // is the reason they are declared separately rather than `apply` returning
    // a list — it is a real property of the shape, not a convenience.

    /// Arrays this op writes beside its primary result.
    ///
    /// `volume` is the extent the op is anchored to, on the same argument as
    /// [`Self::reach`]'s `volume_len`: an output sized from the volume is the
    /// common case and the op must be made to say so rather than assume it.
    ///
    /// The default is none, so an op that says nothing writes one array — which
    /// is every op this crate shipped before the method existed.
    fn side_outputs(&self, _volume: [usize; 3]) -> Vec<Output> {
        Vec::new()
    }

    /// Where this block's slice of side output `which` lands, in **that
    /// output's own** coordinate space.
    ///
    /// A function of the block's valid region and the volume, never of the data,
    /// for the same reason a `Decomposition` is data-blind: two datasets would
    /// otherwise land in two different places and no parity figure would carry.
    ///
    /// The default is `valid` itself, which is right for a side output that is a
    /// second array of the phase's own shape. It is safe to default because it
    /// is **checked and not assumed**: the executor requires the regions a phase
    /// produces to tile the declared output exactly, so a mapping of the wrong
    /// rank, or one that leaves a hole, fails the run rather than half-filling
    /// an array.
    fn side_region(&self, _which: usize, valid: &Region, _volume: [usize; 3]) -> Result<Region> {
        Ok(valid.clone())
    }

    /// This block's slice of each side output, in declaration order.
    ///
    /// Called once per block, with the buffer the op was given **and the primary
    /// result it produced**, so an op deriving a side output from its own answer
    /// does not recompute it and one that does not can ignore `primary`.
    ///
    /// Each array must have the shape of the corresponding [`SideBlock::regions`]
    /// entry; the environment checks that rather than trusting it.
    /// The arrays are `ArrayD<f64>` and stay so: a side output's **rank** is its
    /// own — see [`Output`] — and pinning it to 3 the way a level now is would
    /// delete the case the type exists for.
    fn apply_side(
        &self,
        _input: &Voxels,
        _primary: &Voxels,
        _block: &SideBlock<'_>,
    ) -> Result<Vec<ArrayD<f64>>> {
        Ok(Vec::new())
    }

    // ------------------------------------- constraining the decomposition --

    /// What this op requires of the blocks it is handed. `None` — the default —
    /// means "any".
    ///
    /// `volume` is the extent of the **level the phase reads**, because that is
    /// the space a block's `source` region is in and `source` is what the op is
    /// handed. For every phase whose output grid is its input grid the two are
    /// the same; where they differ, a lattice laid over an array is a lattice
    /// over that array's extent and not over the one being written.
    ///
    /// **Consulted by `decompose` and re-checked in `execute`.** A constraint
    /// honoured only at planning time is not a constraint: the whole point of
    /// this crate's one-trait design is that `Greedy::run` may be handed
    /// `Trivial::decompose`'s plan, and a plan from anywhere must be refused if
    /// it does not fit. The re-check lives in the executor rather than in
    /// `Decomposition::check` because the executor is the first place that holds
    /// both the plan and the ops — a `Decomposition` records names, not
    /// implementations.
    ///
    /// The failure this removes was measured: with a mandated block shape absent
    /// from the candidate list, both shipped strategies returned a decomposition
    /// that `check`s clean and cannot run.
    fn block_constraint(&self, _volume: [usize; 3]) -> Option<BlockConstraint> {
        None
    }
}

/// What joins the branches of a [`Chain::Parallel`] back into one result.
///
/// **Why a second trait rather than a wider `BlockOp`.** A `BlockOp` is handed
/// one buffer, and every one of the ~dozen implementations in this crate and in
/// its callers is written to that signature. A combine is handed *several*, one
/// per branch, and every one of its questions — which element types it accepts,
/// what shape it produces, what a constant folds to — is a question about a
/// **list**. Widening `BlockOp` to take a slice would rewrite every op to say
/// "I take exactly one" and would let a one-input op be placed where a fan-in
/// is meant. The arity is the difference, so the arity is in the type.
///
/// **What it deliberately does not have.** No side outputs and no
/// `preferred_iteration`. Side outputs are terminal (see [`BlockOp`]'s
/// "several outputs" block) and a branch that wants one declares it on the op
/// that computes it; a traversal preference is advisory and a combine sits
/// under whatever order its branches asked for. Both can be added later without
/// changing what is written here; neither can be removed once callers rely on
/// it.
pub trait Combine: Send + Sync {
    fn name(&self) -> &'static str;

    /// Voxels read beyond the voxel written, along `axis`, by the combine
    /// **itself** — not by its branches.
    ///
    /// No default, for the same reason [`BlockOp::reach`] has none: this is the
    /// number the halo is derived from and the one place a silent zero would
    /// produce a complete, well-formed, wrong volume. A voxelwise combine
    /// answers `0` and says so.
    fn reach(&self, axis: usize, volume_len: usize) -> usize;

    /// The full statement, on exactly [`BlockOp::reach_spec`]'s argument and
    /// with the same default: a combine that says one integer per axis means the
    /// symmetric form, and the vast majority do, because a combine that is
    /// voxelwise reaches nothing at all.
    fn reach_spec(&self, volume: [usize; 3]) -> Reach {
        Reach::symmetric([
            self.reach(0, volume[0]),
            self.reach(1, volume[1]),
            self.reach(2, volume[2]),
        ])
    }

    /// Can this combine be handed branch results of exactly these element
    /// types, in branch order?
    ///
    /// The arity is part of the question: a combine that joins two branches
    /// must refuse a list of three rather than silently ignoring one.
    fn accepts(&self, inputs: &[Dtype]) -> bool;

    /// What it writes, given what the branches wrote. Only ever called for a
    /// list this combine [`accepts`](Self::accepts).
    fn produces(&self, inputs: &[Dtype]) -> Dtype;

    /// The shape it produces from the branches' shapes.
    ///
    /// Fallible where [`BlockOp::output_shape`] is not, and that is the point:
    /// branch shapes are not required to agree *by the node* — see
    /// [`Chain::output_shape`] — so the combine is the one thing that knows
    /// whether the list it was given is joinable, and it must be able to say no.
    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]>;

    /// Compute the combine over every branch result, writing into `out`.
    ///
    /// `inputs` is in branch order and `at` is the anchor of the **shared**
    /// input the branches were handed, which is also the anchor of their
    /// results: every branch of a fan-in reads the same buffer at the same
    /// position, which is what makes it a fan-in rather than two chains.
    fn apply(&self, inputs: &[Voxels], out: &mut Voxels, at: &Anchor) -> Result<()>;

    /// If branch `i` would produce the constant `values[i]` everywhere, is the
    /// result a known constant? `None` — the default — disables the short
    /// circuit, on exactly [`BlockOp::constant_maps_to`]'s argument.
    fn constant_maps_to(&self, _values: &[f64]) -> Option<f64> {
        None
    }

    /// Relative compute cost per voxel produced, given how many branches are
    /// being joined.
    ///
    /// The arity is an argument because a pairwise combine folded over `n`
    /// branches does `n - 1` pairs' worth of work, and a cost that could not
    /// see `n` would charge a three-branch join what it charges a two-branch
    /// one. Measured, not guessed; the default of `1.0` is a placeholder.
    fn cost_per_voxel(&self, _branches: usize) -> f64 {
        1.0
    }
}

/// The stored levels a chain's [`Chain::Source`] leaves are handed, keyed by
/// level.
///
/// **One entry per level, not one per leaf.** Two leaves naming the same level
/// read the same voxels of the same array at the same extent, so giving them
/// one buffer is not a cache — it is the statement that a level is one thing.
/// It also removes the only ordering question a positional list would have had:
/// a `Chain::Alternative` whose live branch skips a leaf would consume a
/// different number of entries than the tree contains, and every fold in this
/// file would then have to agree on a traversal order that nothing else needs.
///
/// Each buffer holds **exactly the extent the block was read at** — the phase's
/// fetch region — because a source leaf has reach 0 and produces the shape it
/// was handed. The executor reads it; nothing here fetches.
#[derive(Clone, Copy)]
pub struct SourceInputs<'a> {
    entries: &'a [(usize, &'a Voxels)],
}

impl<'a> SourceInputs<'a> {
    /// Nothing stored: what a chain with no source leaf is applied with, and
    /// what [`Chain::apply`] passes.
    pub const fn none() -> Self {
        Self { entries: &[] }
    }

    pub fn new(entries: &'a [(usize, &'a Voxels)]) -> Self {
        Self { entries }
    }

    /// The buffer for `level`, or an error naming what was supplied.
    ///
    /// An error rather than an `Option` because there is no sensible thing to
    /// do without it: a missing operand is a block that would combine against
    /// nothing, which is the class of quiet wrong answer this crate is arranged
    /// against.
    pub fn get(&self, level: usize) -> Result<&'a Voxels> {
        self.entries
            .iter()
            .find(|(named, _)| *named == level)
            .map(|(_, buf)| *buf)
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "a source leaf reads level {level} and the executor supplied [{}]. The \
                     levels a phase reads besides its own input are recorded in the plan \
                     (`PhaseDecomposition::source_levels`) and read there; a leaf naming one \
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

/// The structure of a chain: what is sequential, what is exclusive, what
/// branches and rejoins.
///
/// `Alternative` carries `taken` in addition to the branches, which is the one
/// place this departs from the sketch in `docs/design/BLOCK_OPS.md`. The sketch
/// writes `Alternative(Vec<Chain>)`, which defines `reach` (the max, so the
/// plan is valid whichever branch runs) but leaves `apply` undefined — there is
/// no way to walk "mutually exclusive branches" without knowing which one is
/// live. `taken` is the minimum needed to make execution total, and it keeps
/// the asymmetry the design wants: **reach is budgeted for every branch,
/// execution runs one.** Over-declaring reach is safe and merely costs halo.
///
/// `Parallel` is the other reading of the same `max`, and the reason it was
/// missing for so long. `docs/design/BLOCK_OPS.md` records a real diamond
/// modelled as an `Alternative` that passed **903** comparisons: reach folds as
/// the max whether one branch runs or all of them, so no reach test can tell
/// the two apart. What tells them apart is what executes and what is produced —
/// and there the two variants are opposites, which is why every fold below
/// states which reading it is taking:
///
/// | fold | `Alternative` | `Parallel` |
/// |---|---|---|
/// | `reach` | max | max, plus the combine's |
/// | `apply` | `taken` | every branch, then the combine |
/// | `side_outputs` | `taken`'s | **the union**, in branch order |
/// | `cost_per_voxel` | max | sum, plus the combine's |
/// | `constant_maps_to` | `taken`'s | every branch's, then the combine's |
///
/// The `side_outputs` row is the one that cannot be got wrong quietly. Step 2
/// of the combined pass put it exactly: *over-declaring an output is not safe
/// the way over-declaring reach is — a reach that is too large costs reads, an
/// output that is not produced is a hole.* `Alternative` therefore declares
/// only what `taken` writes. `Parallel` must declare all of them, because all
/// of them are written, and an undeclared one has no array to land in.
/// **Structurally, there is nothing else it could do**: the variant carries no
/// `taken`, so the fold has no index to select with and the union is its only
/// total definition.
pub enum Chain {
    /// Applied in order: reaches ADD.
    Sequence(Vec<Chain>),
    /// Mutually exclusive branches: reaches take the MAX; `taken` runs.
    Alternative {
        branches: Vec<Chain>,
        taken: usize,
    },
    /// Concurrent branches over the same input, rejoined by `combine`: reaches
    /// take the MAX and the combine's ADDS. **Every branch runs.**
    Parallel {
        branches: Vec<Chain>,
        combine: Box<dyn Combine>,
    },
    Op(Box<dyn BlockOp>),
    /// A leaf that **reads** a stored level instead of computing one.
    ///
    /// Every other node in this tree is a function of the buffer handed to it.
    /// This one ignores that buffer and answers with a level of the run, read
    /// at the block's own read extent. It is what makes a two-array operation
    /// expressible without holding the second array whole:
    /// `Chain::parallel([computed, Chain::source(level, dtype)], combine)` is a
    /// diamond whose second arm is an array on disk, and every fold below folds
    /// it exactly as it folds a computed arm.
    ///
    /// **Why a leaf of the chain rather than a second input to a phase.**
    /// `Combine::produces` and `Combine::accepts` already take **lists**, the
    /// `Parallel` node already allocates one buffer per branch and joins them,
    /// and `tests/fan_in.rs` already proves that machinery. A second input on
    /// `PhaseDecomposition` would have needed a parallel set of folds — reach,
    /// element type, shape, cost — for the same shape of question. So the edge
    /// the design record asks for (`docs/design/BLOCK_OPS.md`, *"Levels are a
    /// DAG"*) is added where the DAG already existed.
    ///
    /// **Reach 0, and that is exact rather than conservative.** It reads the
    /// extent it is asked for and nothing around it, so it adds nothing to the
    /// phase's halo. A source arm therefore never widens the read of the arm
    /// beside it — which is the whole reason this is cheaper than the array it
    /// replaces.
    ///
    /// **The element type is declared and then checked.** `produces` has only
    /// an input element type to work from and a source leaf's answer does not
    /// depend on it, so the leaf carries the answer. What the level actually
    /// holds is known to the plan, not to the chain, so
    /// [`check_source_levels`](crate::decomposition::check_source_levels)
    /// compares the two — the same arrangement as every other quantity this
    /// crate states twice.
    ///
    /// **`level` is a level of the plan, so a chain carrying one constrains the
    /// plans it is valid for.** That is not a leak: which level is read is
    /// parity-visible, it is recorded in `PhaseDecomposition::source_levels`,
    /// and a partition that renumbers the level out from under a leaf is
    /// refused by name at plan time rather than discovered at the first block.
    Source {
        /// The level read. Must be at or below the level its phase reads, so
        /// that the phase that wrote it has run; a forward reference is refused
        /// by `check_source_levels`.
        level: usize,
        /// The element type that level holds.
        dtype: Dtype,
    },
}

impl Chain {
    /// A single op, boxed.
    pub fn op<T: BlockOp + 'static>(op: T) -> Chain {
        Chain::Op(Box::new(op))
    }

    /// A sequence of ops, each boxed.
    pub fn sequence(children: Vec<Chain>) -> Chain {
        Chain::Sequence(children)
    }

    /// A leaf that reads level `level`, which holds `dtype`.
    ///
    /// Infallible here on purpose: whether the level exists, whether it is a
    /// forward reference and whether it really holds `dtype` are all questions
    /// about a *plan*, and this constructor has none. They are answered by
    /// [`check_source_levels`](crate::decomposition::check_source_levels), in
    /// one place, at plan time.
    pub fn source(level: usize, dtype: Dtype) -> Chain {
        Chain::Source { level, dtype }
    }

    /// Every level named by a source leaf anywhere in the subtree, ascending
    /// and without repeats.
    ///
    /// **Every branch of an `Alternative` counts, not just `taken`.** This is
    /// the `reach` reading rather than the `side_outputs` reading, and for
    /// `reach`'s reason: the level has to be *there* whichever branch is live,
    /// so it must be kept alive and read for all of them. Over-declaring costs
    /// a read; under-declaring is a branch with no operand.
    pub fn source_levels(&self) -> Vec<usize> {
        let mut seen = Vec::new();
        self.collect_source_levels(&mut seen);
        seen.sort_unstable();
        seen.dedup();
        seen
    }

    fn collect_source_levels(&self, seen: &mut Vec<usize>) {
        match self {
            Chain::Op(_) => {}
            Chain::Source { level, .. } => seen.push(*level),
            Chain::Sequence(children)
            | Chain::Alternative {
                branches: children, ..
            }
            | Chain::Parallel {
                branches: children, ..
            } => {
                for child in children {
                    child.collect_source_levels(seen);
                }
            }
        }
    }

    /// Mutually exclusive branches, of which `taken` is live.
    pub fn alternative(branches: Vec<Chain>, taken: usize) -> Result<Chain> {
        if branches.is_empty() {
            return Err(Error::InvalidArgument(
                "Chain::alternative needs at least one branch".to_string(),
            ));
        }
        if taken >= branches.len() {
            return Err(Error::InvalidArgument(format!(
                "Chain::alternative taken={taken} out of {} branches",
                branches.len()
            )));
        }
        Ok(Chain::Alternative { branches, taken })
    }

    /// Branches that **all** run over the same input, rejoined by `combine`.
    ///
    /// **Two branches is the minimum and it is checked.** A one-branch fan-in
    /// is a `Sequence` wearing a different name: it would fold every quantity
    /// below to the branch's own answer plus the combine's, so it can only
    /// mislead a reader into thinking something forks. Zero branches is a
    /// combine with nothing to combine — the malformed case that would
    /// otherwise reach `apply` and index an empty slice.
    pub fn parallel(branches: Vec<Chain>, combine: Box<dyn Combine>) -> Result<Chain> {
        if branches.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "Chain::parallel for combine {:?} was given no branches. A fan-in joins results \
                 that something produced; with none there is nothing to join and nothing to \
                 allocate the combine's inputs from.",
                combine.name()
            )));
        }
        if branches.len() == 1 {
            return Err(Error::InvalidArgument(format!(
                "Chain::parallel for combine {:?} was given one branch. A single branch is a \
                 `Sequence` — every fold here would reduce to that branch's own answer — and \
                 writing it as a fan-in says the chain forks where it does not.",
                combine.name()
            )));
        }
        Ok(Chain::Parallel { branches, combine })
    }

    /// A name for logs and diagnostics.
    pub fn display_name(&self) -> String {
        match self {
            Chain::Op(op) => op.name().to_string(),
            Chain::Source { level, .. } => format!("source(level {level})"),
            Chain::Parallel { branches, combine } => format!(
                "par({})>{}",
                branches
                    .iter()
                    .map(Chain::display_name)
                    .collect::<Vec<_>>()
                    .join("&"),
                combine.name()
            ),
            Chain::Alternative { branches, taken } => format!(
                "alt[{}]({})",
                taken,
                branches
                    .iter()
                    .map(Chain::display_name)
                    .collect::<Vec<_>>()
                    .join("|")
            ),
            Chain::Sequence(children) => format!(
                "seq({})",
                children
                    .iter()
                    .map(Chain::display_name)
                    .collect::<Vec<_>>()
                    .join(">")
            ),
        }
    }

    /// Voxels read beyond the voxel written, along `axis`, by the whole
    /// subtree: sequential reaches **add**, exclusive reaches take the **max**,
    /// concurrent reaches take the max and the combine's adds.
    ///
    /// **Why `Parallel` folds as max-then-add, and why it is exact as a fold.**
    /// To produce the combine's output at voxel `v` the combine reads its
    /// branches' results within `r_combine` of `v`; branch `i` produces a
    /// result at voxel `u` from input within `r_i` of `u`. So the input needed
    /// for `v` lies within `max_i r_i + r_combine` — the same arithmetic
    /// `Sequence` does over a stage whose reach is the max of the branches',
    /// which is what a fan-in *is* when you look only at what it reads.
    ///
    /// **It is an upper bound on the chain, not a tight per-voxel figure**, and
    /// that is true of every reach this crate folds: `docs/design/BLOCK_OPS.md`
    /// records that a chain's reach is an upper bound because three ops would
    /// have to hit their individual worst cases at the same seam for it to be
    /// attained. `Parallel` adds one more way to be loose — the widest branch
    /// and the combine's own widest offset need not line up either — and
    /// nothing about that is new in kind. Over-declaring a reach costs reads;
    /// the guard in `decomposition.rs` fires on under-declaration, which this
    /// fold cannot produce.
    pub fn reach(&self, axis: usize, volume_len: usize) -> usize {
        match self {
            Chain::Op(op) => op.reach(axis, volume_len),
            // Zero, exactly. A source leaf reads the extent it is handed and
            // nothing around it, so it contributes nothing to its phase's halo
            // and never widens the arm beside it.
            Chain::Source { .. } => 0,
            Chain::Sequence(children) => children
                .iter()
                .map(|child| child.reach(axis, volume_len))
                .sum(),
            Chain::Alternative { branches, .. } => branches
                .iter()
                .map(|branch| branch.reach(axis, volume_len))
                .max()
                .unwrap_or(0),
            Chain::Parallel { branches, combine } => {
                branches
                    .iter()
                    .map(|branch| branch.reach(axis, volume_len))
                    .max()
                    .unwrap_or(0)
                    + combine.reach(axis, volume_len)
            }
        }
    }

    /// Reach on every axis of a 3-D volume, as the symmetric triple.
    pub fn reach3(&self, volume: &[usize]) -> [usize; 3] {
        [
            self.reach(0, volume[0]),
            self.reach(1, volume[1]),
            self.reach(2, volume[2]),
        ]
    }

    /// The full statement of the whole subtree's reach, folded over the same
    /// tree [`Self::reach`] walks and by the same rules — sequential reaches
    /// **add**, exclusive and concurrent branches take the **wider**, a
    /// combine's own reach adds on top — now per side, per block and per
    /// coordinate space.
    ///
    /// **This is what the plan is built from.** `Chain::reach` remains the
    /// symmetric bound, and is what a caller holding one integer per axis must
    /// use; everything that derives geometry uses this.
    ///
    /// Two ways it can fail, and both are facts about the chain rather than
    /// about the caller:
    ///
    /// * two ops state their reaches in **different coordinate spaces**, which
    ///   cannot be folded without a grid to convert with. A planner turns that
    ///   into "these two cannot share a phase", which is the right answer: a
    ///   change of coordinate space is a phase boundary.
    /// * an op's `reach_spec` is **wider than its own `reach`**. The two are one
    ///   quantity stated twice, and the whole crate is arranged so that a
    ///   quantity stated twice is checked rather than assumed. The bound is the
    ///   one the halo used to be derived from, so an op that quietly widened
    ///   past it would be under-halo'd by every plan built before it changed.
    pub fn reach_spec(&self, volume: [usize; 3]) -> Result<Reach> {
        let spec = self.fold_reach_spec(volume)?;
        for axis in 0..3 {
            let bound = self.reach(axis, volume[axis]);
            let (lo, hi) = spec.axis(axis).bound(volume[axis]);
            let widest = if spec.is_whole_axis(axis, volume[axis]) && bound >= volume[axis] {
                // `All` and "a number at least the extent" agree; the bound is
                // met by construction and the numbers need not.
                bound
            } else {
                lo.max(hi)
            };
            if widest > bound {
                return Err(Error::InvalidArgument(format!(
                    "{}: its full reach says {widest} on axis {axis} and its symmetric reach says \
                     {bound}. The two are one quantity stated twice, and the symmetric one is \
                     what every plan built before the full form existed derived its halo from, so \
                     it has to remain a bound on it.",
                    self.display_name()
                )));
            }
        }
        Ok(spec)
    }

    fn fold_reach_spec(&self, volume: [usize; 3]) -> Result<Reach> {
        match self {
            Chain::Op(op) => Ok(op.reach_spec(volume)),
            Chain::Source { .. } => Ok(Reach::none()),
            Chain::Sequence(children) => fold_specs(children, volume, Reach::add),
            Chain::Alternative { branches, .. } => fold_specs(branches, volume, Reach::max),
            Chain::Parallel { branches, combine } => {
                let widest = fold_specs(branches, volume, Reach::max)?;
                widest.add(&combine.reach_spec(volume))
            }
        }
    }

    /// What the subtree writes, given what it reads, or an error naming the op
    /// that refuses the type it would be handed.
    ///
    /// Sequential ops **compose**: each is handed what the one before produced,
    /// which is the same walk `apply` takes. Exclusive branches must **agree** —
    /// and that is the one place this departs from `reach`, deliberately. A
    /// reach may be over-declared because a wider halo is merely wasteful; an
    /// element type cannot, because the level is allocated at one width and the
    /// plan is binding. Two branches that write different types are a phase
    /// whose output type depends on which arm ran, so they are refused here
    /// rather than producing a plan that is right for one of them.
    ///
    /// **Concurrent branches are required to agree with the *combine*, not with
    /// each other** — the same rule stated from the other side. An
    /// `Alternative`'s branches are candidates for one level, so they must
    /// match each other. A `Parallel`'s branch results never become a level:
    /// they are transient buffers, allocated and consumed inside one slot, and
    /// only the combine's answer is written. Requiring them to match each other
    /// would forbid the case the node exists for — an image branch and a mask
    /// branch joined into a masked image is three element types by
    /// construction. So what is checked here is `Combine::accepts` against the
    /// list the branches produce, at plan time, in the same place and for the
    /// same reason as `BlockOp::accepts`.
    pub fn produces(&self, input: Dtype) -> Result<Dtype> {
        match self {
            Chain::Op(op) => {
                if !op.accepts(input) {
                    return Err(Error::InvalidArgument(format!(
                        "op {:?} does not accept {}. An op states the element types it can be \
                         handed, and a plan that would hand it another is refused when the plan \
                         is made rather than when a block reaches it.",
                        op.name(),
                        input.numpy_name()
                    )));
                }
                Ok(op.produces(input))
            }
            // **Accepts anything and answers what it holds.** A source leaf is
            // not a function of the buffer it was handed — it ignores it — so
            // refusing an element type here would be refusing something it
            // never looks at. The declaration is checked against the level it
            // names by `check_source_levels`, which is the only place that
            // knows what the level holds.
            Chain::Source { dtype, .. } => Ok(*dtype),
            Chain::Sequence(children) => {
                let mut current = input;
                for child in children {
                    current = child.produces(current)?;
                }
                Ok(current)
            }
            Chain::Alternative { branches, .. } => {
                let mut agreed: Option<Dtype> = None;
                for branch in branches {
                    let produced = branch.produces(input)?;
                    match agreed {
                        None => agreed = Some(produced),
                        Some(existing) if existing == produced => {}
                        Some(existing) => {
                            return Err(Error::InvalidArgument(format!(
                                "{:?} writes {} on one branch and {} on another. A level is \
                                 allocated at one width and a decomposition is binding, so an \
                                 element type that depends on which branch is live is not a plan.",
                                self.display_name(),
                                existing.numpy_name(),
                                produced.numpy_name()
                            )))
                        }
                    }
                }
                agreed.ok_or_else(|| {
                    Error::InvalidArgument(
                        "Chain::alternative needs at least one branch".to_string(),
                    )
                })
            }
            Chain::Parallel { branches, combine } => {
                let produced = branches
                    .iter()
                    .map(|branch| branch.produces(input))
                    .collect::<Result<Vec<Dtype>>>()?;
                if !combine.accepts(&produced) {
                    return Err(Error::InvalidArgument(format!(
                        "combine {:?} does not accept [{}], which is what the {} branches of \
                         {:?} write from {}. Every branch of a fan-in runs and every result \
                         reaches the combine, so a type it cannot join is refused when the plan \
                         is made rather than when a block reaches it.",
                        combine.name(),
                        produced
                            .iter()
                            .map(|dtype| dtype.numpy_name())
                            .collect::<Vec<_>>()
                            .join(", "),
                        produced.len(),
                        self.display_name(),
                        input.numpy_name()
                    )));
                }
                Ok(combine.produces(&produced))
            }
        }
    }

    /// The shape the subtree produces from `input`, folded the same way.
    ///
    /// Exclusive branches must agree, on the same argument as [`Self::produces`]:
    /// the level's extent is in the plan and the plan does not know which branch
    /// runs.
    ///
    /// Concurrent branches are **not** required to agree here; the combine is
    /// asked whether the shapes it was handed are joinable, which is why
    /// [`Combine::output_shape`] is fallible where [`BlockOp::output_shape`] is
    /// not. A voxelwise combine answers by requiring agreement and naming the
    /// two branches that disagree; a combine that stacks or reduces answers
    /// something else, and neither is this node's business.
    pub fn output_shape(&self, input: [usize; 3]) -> Result<[usize; 3]> {
        match self {
            Chain::Op(op) => Ok(op.output_shape(input)),
            // The extent it was asked for, which is the whole of what "reach 0"
            // means for a leaf that reads rather than computes. It is also what
            // makes a source arm joinable with a computed arm that keeps its
            // extent, and what makes it *refused* — by the combine, naming both
            // shapes — beside one that resizes.
            Chain::Source { .. } => Ok(input),
            Chain::Sequence(children) => {
                let mut current = input;
                for child in children {
                    current = child.output_shape(current)?;
                }
                Ok(current)
            }
            Chain::Alternative { branches, .. } => {
                let mut agreed: Option<[usize; 3]> = None;
                for branch in branches {
                    let produced = branch.output_shape(input)?;
                    match agreed {
                        None => agreed = Some(produced),
                        Some(existing) if existing == produced => {}
                        Some(existing) => {
                            return Err(Error::InvalidArgument(format!(
                                "{:?} produces {existing:?} on one branch and {produced:?} on \
                                 another from an input of {input:?}. The level's extent is in the \
                                 plan, and the plan does not know which branch is live.",
                                self.display_name()
                            )))
                        }
                    }
                }
                agreed.ok_or_else(|| {
                    Error::InvalidArgument(
                        "Chain::alternative needs at least one branch".to_string(),
                    )
                })
            }
            Chain::Parallel { branches, combine } => {
                let produced = branches
                    .iter()
                    .map(|branch| branch.output_shape(input))
                    .collect::<Result<Vec<[usize; 3]>>>()?;
                combine.output_shape(&produced)
            }
        }
    }

    /// The same walk as `reach`, over the same tree, for a subtree that reads
    /// nothing but its input.
    ///
    /// `out` must be what the subtree declared it produces, in both shape and
    /// element type. It is checked rather than assumed because this is the seam
    /// where a wrong declaration would otherwise become a wrong volume.
    ///
    /// A subtree containing a [`Chain::Source`] fails here, naming the level:
    /// this entry point has no stored operands to hand it, and producing a
    /// buffer without one would be the quiet wrong answer rather than the loud
    /// one. Use [`Self::apply_with`].
    pub fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        self.apply_with(input, SourceInputs::none(), out, at)
    }

    /// [`Self::apply`], with the stored levels this subtree's source leaves
    /// read.
    ///
    /// `sources` is threaded down unchanged, exactly as `at` is and for the
    /// same reason: a leaf deep inside a `Parallel` branch must see the same
    /// buffer the executor read, not one re-derived on the way down.
    pub fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        at: &Anchor,
    ) -> Result<()> {
        let wanted_shape = self.output_shape(input.shape())?;
        let wanted_dtype = self.produces(input.dtype())?;
        if out.shape() != wanted_shape {
            return Err(Error::ShapeMismatch {
                expected: wanted_shape.to_vec(),
                got: out.shape().to_vec(),
            });
        }
        if out.dtype() != wanted_dtype {
            return Err(Error::InvalidArgument(format!(
                "{:?} reads {} and writes {}, and was handed an output holding {}",
                self.display_name(),
                input.dtype().numpy_name(),
                wanted_dtype.numpy_name(),
                out.dtype().numpy_name()
            )));
        }
        match self {
            Chain::Op(op) => op.apply(input, out, at),
            // The one node that answers from something other than `input`. The
            // buffer holds the block's read extent of the level, so this is a
            // copy and not a slice: the executor already asked for exactly the
            // extent, at reach 0, and anything else would mean the plan and the
            // read disagreed.
            Chain::Source { level, dtype } => {
                let stored = sources.get(*level)?;
                if stored.dtype() != *dtype {
                    return Err(Error::InvalidArgument(format!(
                        "a source leaf declares level {level} holds {} and the buffer read from \
                         it holds {}. The declaration is what every fold of this chain was built \
                         from, so the two have to be one fact.",
                        dtype.numpy_name(),
                        stored.dtype().numpy_name()
                    )));
                }
                if stored.shape() != out.shape() {
                    return Err(Error::ShapeMismatch {
                        expected: out.shape().to_vec(),
                        got: stored.shape().to_vec(),
                    });
                }
                out.assign(stored)
            }
            Chain::Alternative { branches, taken } => {
                branches[*taken].apply_with(input, sources, out, at)
            }
            // Every branch, over the **same** buffer at the **same** anchor,
            // then the combine over all of their results. The shared input is
            // what makes this a fan-in rather than two chains that happen to be
            // adjacent, and passing `at` down unchanged is what makes a
            // position-dependent branch see the position it actually has.
            //
            // The results live here, in this frame, for the length of this
            // call: `branches.len()` buffers of the branch's own declared shape
            // and element type, allocated exactly the way `Sequence` allocates
            // its intermediates a few lines below. They never reach a level and
            // never reach the environment, because a level is a phase's output
            // and this whole node is one slot of one phase.
            Chain::Parallel { branches, combine } => {
                let mut results = Vec::with_capacity(branches.len());
                for branch in branches {
                    let mut result = Voxels::zeros(
                        branch.produces(input.dtype())?,
                        branch.output_shape(input.shape())?,
                    )?;
                    branch.apply_with(input, sources, &mut result, at)?;
                    results.push(result);
                }
                combine.apply(&results, out, at)
            }
            Chain::Sequence(children) => match children.len() {
                0 => out.assign(input),
                1 => children[0].apply_with(input, sources, out, at),
                n => {
                    let mut current = input.clone();
                    for (position, child) in children.iter().enumerate() {
                        if position + 1 == n {
                            return child.apply_with(&current, sources, out, at);
                        }
                        let mut next = Voxels::zeros(
                            child.produces(current.dtype())?,
                            child.output_shape(current.shape())?,
                        )?;
                        child.apply_with(&current, sources, &mut next, at)?;
                        current = next;
                    }
                    Ok(())
                }
            },
        }
    }

    /// Every side output the subtree writes, in the order they are written.
    ///
    /// **`Alternative` consults `taken`, not every branch**, which is the
    /// opposite of `reach` and for a stated reason: over-declaring a reach costs
    /// halo, while over-declaring an output allocates an array nobody writes and
    /// then fails the coverage guard for a hole that is not a bug. Only the live
    /// branch writes, so only the live branch declares — the same rule
    /// `constant_maps_to` follows.
    ///
    /// **`Parallel` folds by union, and that is the whole distinction between
    /// the two variants.** Read the sentence above in reverse: only the branch
    /// that *writes* may declare, and in a fan-in every branch writes. Folding
    /// a `Parallel` by consulting one branch would leave the others' arrays
    /// undeclared, so the executor would never create them and the block
    /// results would have nowhere to land — the hole that
    /// `docs/design/BLOCK_OPS.md` §"Step 2" contrasts with an over-wide reach,
    /// which merely costs reads. There is also nothing to consult with: the
    /// variant carries no `taken`.
    pub fn side_outputs(&self, volume: [usize; 3]) -> Vec<Output> {
        match self {
            Chain::Op(op) => op.side_outputs(volume),
            // A leaf that reads writes nothing.
            Chain::Source { .. } => Vec::new(),
            Chain::Sequence(children) => children
                .iter()
                .flat_map(|child| child.side_outputs(volume))
                .collect(),
            Chain::Alternative { branches, taken } => branches[*taken].side_outputs(volume),
            Chain::Parallel { branches, .. } => branches
                .iter()
                .flat_map(|branch| branch.side_outputs(volume))
                .collect(),
        }
    }

    /// Where this block's slice of side output `which` of the subtree lands.
    pub fn side_region(&self, which: usize, valid: &Region, volume: [usize; 3]) -> Result<Region> {
        match self {
            Chain::Op(op) => op.side_region(which, valid, volume),
            Chain::Source { level, .. } => Err(Error::InvalidArgument(format!(
                "side output {which} of a leaf reading level {level}, which declares none"
            ))),
            Chain::Alternative { branches, taken } => {
                branches[*taken].side_region(which, valid, volume)
            }
            // Routed by counting declarations in branch order, exactly as
            // `Sequence` routes by counting them in child order — because
            // `side_outputs` concatenated them in that order and this is the
            // inverse of that concatenation.
            Chain::Parallel { branches, .. } | Chain::Sequence(branches) => {
                let mut remaining = which;
                for child in branches {
                    let count = child.side_outputs(volume).len();
                    if remaining < count {
                        return child.side_region(remaining, valid, volume);
                    }
                    remaining -= count;
                }
                Err(Error::InvalidArgument(format!(
                    "side output {which} of {:?}, which declares {}",
                    self.display_name(),
                    self.side_outputs(volume).len()
                )))
            }
        }
    }

    /// The same walk as [`Self::side_outputs`], producing the arrays.
    ///
    /// The `Sequence` arm re-derives the intermediates, because a child's side
    /// outputs are a function of *its* input and *its* result rather than the
    /// sequence's. That cost is never paid by the executor, which calls this per
    /// [`Self::slots`] entry and a slot is never a `Sequence`; it is paid only by
    /// a caller applying a whole chain by hand, which is the whole-volume
    /// reference case and is meant to be simple rather than fast.
    ///
    /// **Side outputs and source leaves do not compose yet, and the failure is
    /// loud.** Re-deriving an intermediate calls [`Self::apply`] rather than
    /// [`Self::apply_with`], so a subtree that both declares a side output and
    /// contains a [`Chain::Source`] fails naming the level. The combination the
    /// executor actually meets is safe: a `Parallel` skips every branch that
    /// declares no side output, so a source arm beside a side-output arm is
    /// never re-derived. Threading the stored operands through here as well
    /// would widen [`crate::env::Environment::apply_side`] for a case nothing
    /// exercises, and a widening nothing tests is worse than a refusal that
    /// says what is missing.
    pub fn apply_side(
        &self,
        input: &Voxels,
        primary: &Voxels,
        block: &SideBlock<'_>,
    ) -> Result<Vec<ArrayD<f64>>> {
        match self {
            Chain::Op(op) => op.apply_side(input, primary, block),
            Chain::Source { .. } => Ok(Vec::new()),
            Chain::Alternative { branches, taken } => {
                branches[*taken].apply_side(input, primary, block)
            }
            // Every branch's side outputs, in branch order, matching what
            // `side_outputs` declared.
            //
            // **Each branch's own primary has to be recomputed and none of them
            // is `primary`.** In a `Sequence` the last child's result *is* the
            // sequence's, so that one is passed through; here `primary` is the
            // *combine's* answer and no branch produced it. A branch declaring
            // no side output is skipped rather than recomputed — it would have
            // nothing to be asked for.
            Chain::Parallel { branches, .. } => {
                let mut produced = Vec::new();
                for branch in branches {
                    let taken = branch.side_outputs(block.at.volume).len();
                    if taken == 0 {
                        continue;
                    }
                    let mut result = Voxels::zeros(
                        branch.produces(input.dtype())?,
                        branch.output_shape(input.shape())?,
                    )?;
                    branch.apply(input, &mut result, block.at)?;
                    let regions = &block.regions[produced.len()..produced.len() + taken];
                    produced.extend(branch.apply_side(
                        input,
                        &result,
                        &SideBlock { regions, ..*block },
                    )?);
                }
                Ok(produced)
            }
            Chain::Sequence(children) => {
                let mut produced = Vec::new();
                let mut current = input.clone();
                for (position, child) in children.iter().enumerate() {
                    let result = if position + 1 == children.len() {
                        primary.clone()
                    } else {
                        let mut next = Voxels::zeros(
                            child.produces(current.dtype())?,
                            child.output_shape(current.shape())?,
                        )?;
                        child.apply(&current, &mut next, block.at)?;
                        next
                    };
                    let taken = child.side_outputs(block.at.volume).len();
                    let regions = &block.regions[produced.len()..produced.len() + taken];
                    produced.extend(child.apply_side(
                        &current,
                        &result,
                        &SideBlock { regions, ..*block },
                    )?);
                    current = result;
                }
                Ok(produced)
            }
        }
    }

    /// What the subtree requires of the blocks it is handed, or an error naming
    /// the two ops that cannot be fused because they disagree.
    ///
    /// Disagreement is an **error rather than a conjunction**: two mandated
    /// extents in one phase are not jointly satisfiable, and reporting that as
    /// "no plan fits" would hide the fact that cutting between them does fit.
    /// `Enumerating` turns it into an infeasible partition and keeps searching;
    /// `Greedy` cuts on it.
    pub fn block_constraint(&self, volume: [usize; 3]) -> Result<Option<BlockConstraint>> {
        match self {
            Chain::Op(op) => Ok(op.block_constraint(volume)),
            // Nothing. A source leaf takes the extent it is given, so it can
            // never be the branch that makes a fan-in unsatisfiable.
            Chain::Source { .. } => Ok(None),
            Chain::Alternative { branches, taken } => branches[*taken].block_constraint(volume),
            // **Concurrent branches must agree, and unlike a `Sequence` there
            // is no cut that would resolve a disagreement.** A `Parallel` is
            // one indivisible slot (see [`Self::slots`]), so two branches
            // mandating different extents is not "these cannot be fused" — it
            // is a node that cannot run at all, and the caller has to change
            // the chain rather than the partition. The check is the same fold;
            // only the remedy differs, so only the message does.
            Chain::Parallel { branches, .. } => {
                let mut found: Option<BlockConstraint> = None;
                for branch in branches {
                    let Some(constraint) = branch.block_constraint(volume)? else {
                        continue;
                    };
                    match &found {
                        None => found = Some(constraint),
                        Some(existing) if existing == &constraint => {}
                        Some(existing) => {
                            return Err(Error::InvalidArgument(format!(
                                "{:?} mandates {existing:?} on one branch and {:?} mandates \
                                 {constraint:?} on another. Every branch of a fan-in is handed \
                                 the same block and the node cannot be cut between them, so \
                                 unlike a sequence there is no partition that satisfies both.",
                                self.display_name(),
                                branch.display_name(),
                            )))
                        }
                    }
                }
                Ok(found)
            }
            Chain::Sequence(children) => {
                let mut found: Option<BlockConstraint> = None;
                for child in children {
                    let Some(constraint) = child.block_constraint(volume)? else {
                        continue;
                    };
                    match &found {
                        None => found = Some(constraint),
                        Some(existing) if existing == &constraint => {}
                        Some(existing) => {
                            return Err(Error::InvalidArgument(format!(
                                "{:?} mandates {existing:?} and {:?} mandates {constraint:?}; \
                                 one phase hands every one of its ops the same block, so these \
                                 two cannot be fused",
                                self.display_name(),
                                child.display_name(),
                            )))
                        }
                    }
                }
                Ok(found)
            }
        }
    }

    /// Relative compute cost per voxel of the whole subtree.
    ///
    /// Exclusive branches take the **max**, matching `reach`: the plan must be
    /// affordable whichever branch runs.
    ///
    /// Concurrent branches **sum**, plus the combine's, and this is the fold
    /// where the two variants visibly part company. Reach folds the same way
    /// for both — which is exactly why a diamond could hide as an alternation —
    /// but work does not: an alternation does one branch's work and a fan-in
    /// does all of it. A `Parallel` priced at the max would tell the planner a
    /// three-branch node costs what its widest branch costs.
    pub fn cost_per_voxel(&self) -> f64 {
        match self {
            Chain::Op(op) => op.cost_per_voxel(),
            // Zero *compute*. What a source arm costs is a read, and a read is
            // priced where every other read is — by voxels fetched from a level,
            // through `Environment::read` and `Decomposition::exact_read_voxels`
            // — not by a compute figure that would then be counted twice.
            Chain::Source { .. } => 0.0,
            Chain::Sequence(children) => children.iter().map(Chain::cost_per_voxel).sum(),
            Chain::Alternative { branches, .. } => branches
                .iter()
                .map(Chain::cost_per_voxel)
                .fold(0.0_f64, f64::max),
            Chain::Parallel { branches, combine } => {
                branches.iter().map(Chain::cost_per_voxel).sum::<f64>()
                    + combine.cost_per_voxel(branches.len())
            }
        }
    }

    /// Fold `constant_maps_to` down the subtree: sequential ops compose, and a
    /// single `None` anywhere collapses the whole subtree to `None`.
    ///
    /// Exclusive branches consult **`taken`** rather than every branch, because
    /// only `taken` runs. That is the one place `taken` changes an answer
    /// rather than merely selecting work, and it is sound for the same reason:
    /// the value produced is the value the live branch would have produced.
    ///
    /// A `Parallel` folds only if **every** branch folds *and* the combine
    /// folds on those results — one `None` anywhere collapses the node, the
    /// same rule `Sequence` follows and for the same reason. All three parts
    /// are needed: a branch that declines leaves the combine an unknown
    /// operand, and a combine that declines has said its answer is not
    /// determined by its operands' values alone. Note this is a *conjunction*
    /// where `side_outputs` is a *union* and `Alternative` is a *selection* —
    /// three different folds over the same branch list, which is why each one
    /// is written out rather than shared.
    pub fn constant_maps_to(&self, value: f64) -> Option<f64> {
        match self {
            Chain::Op(op) => op.constant_maps_to(value),
            // **`None`, always.** The short circuit's premise is that a uniform
            // input determines the output, and a source leaf's output is
            // determined by data nobody has looked at. So a phase with a source
            // arm never short circuits — which is the correct answer and not a
            // missed optimisation: the arm it did not read is exactly the thing
            // that could have made the block non-uniform.
            Chain::Source { .. } => None,
            Chain::Alternative { branches, taken } => branches[*taken].constant_maps_to(value),
            Chain::Parallel { branches, combine } => {
                let values = branches
                    .iter()
                    .map(|branch| branch.constant_maps_to(value))
                    .collect::<Option<Vec<f64>>>()?;
                combine.constant_maps_to(&values)
            }
            Chain::Sequence(children) => {
                let mut current = value;
                for child in children {
                    current = child.constant_maps_to(current)?;
                }
                Some(current)
            }
        }
    }

    /// Every distinct traversal preference declared anywhere in the subtree,
    /// in first-seen order.
    ///
    /// More than one entry is the signal `docs/design/BLOCK_OPS.md` calls a
    /// **candidate phase boundary**: ops that want opposite traversals are
    /// telling the planner where fusing stops paying.
    pub fn preferred_iterations(&self) -> Vec<[usize; 3]> {
        let mut seen = Vec::new();
        self.collect_preferred(&mut seen);
        seen
    }

    fn collect_preferred(&self, seen: &mut Vec<[usize; 3]>) {
        match self {
            Chain::Op(op) => {
                if let Some(order) = op.preferred_iteration() {
                    if !seen.contains(&order) {
                        seen.push(order);
                    }
                }
            }
            // No preference. A traversal order is a claim about locality of the
            // work, and this node does none; the order it is read in is the one
            // the arm beside it asked for.
            Chain::Source { .. } => {}
            // Every branch of a fan-in runs, so every branch's preference is
            // real and a disagreement between two of them is exactly the
            // "candidate phase boundary" signal — with the caveat that this one
            // cannot be cut on, because the node is one slot. Seeing it is
            // still worth more than not seeing it: it is the measurement that
            // says a fan-in is paying an order conflict.
            Chain::Parallel { branches, .. } | Chain::Sequence(branches) => {
                for child in branches {
                    child.collect_preferred(seen);
                }
            }
            Chain::Alternative { branches, taken } => branches[*taken].collect_preferred(seen),
        }
    }

    /// The chain flattened into the units the planner may cut between.
    ///
    /// Nested `Sequence`s flatten (sequencing is associative, and finer
    /// granularity gives the planner more cut points). An `Alternative` is one
    /// indivisible slot: cutting *inside* an exclusive branch would produce a
    /// schedule whose phases depend on which branch is live, which is exactly
    /// the decomposition-dependent behaviour this framework exists to prevent.
    ///
    /// A `Parallel` is one indivisible slot too, and for a blunter reason: a
    /// phase reads **one** level and writes **one** level
    /// (`Decomposition::dtype_at`, `Environment::read`), so a cut placed after
    /// the branches and before the combine would need a level per branch and a
    /// phase with several inputs, neither of which a `Decomposition` can state.
    /// `docs/design/BLOCK_OPS.md` names cutting inside the diamond as what
    /// fan-in would eventually *let* the planner do; this change makes the
    /// diamond expressible and executable, and leaves that cut where it was.
    pub fn slots(&self) -> Vec<&Chain> {
        let mut out = Vec::new();
        self.collect_slots(&mut out);
        out
    }

    fn collect_slots<'a>(&'a self, out: &mut Vec<&'a Chain>) {
        match self {
            Chain::Sequence(children) => {
                for child in children {
                    child.collect_slots(out);
                }
            }
            _ => out.push(self),
        }
    }
}

/// Fold a list of branches' full reaches with `combine`, which is `Reach::add`
/// for a sequence and `Reach::max` for branches.
///
/// An empty list folds to nothing read, which is what an empty subtree does; the
/// constructors already refuse an empty `Alternative` or `Parallel`, so this is
/// reachable only through `Chain::Sequence(vec![])`.
fn fold_specs(
    branches: &[Chain],
    volume: [usize; 3],
    combine: impl Fn(&Reach, &Reach) -> Result<Reach>,
) -> Result<Reach> {
    let mut folded: Option<Reach> = None;
    for branch in branches {
        let stated = branch.fold_reach_spec(volume)?;
        folded = Some(match folded {
            Some(so_far) => combine(&so_far, &stated)?,
            None => stated,
        });
    }
    Ok(folded.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Logic, LogicCombine};
    use crate::probes::{AffineOp, DecimateOp, IdentityOp, NonZeroOp, OpaqueOp, SideOutputOp};

    fn or() -> Box<dyn Combine> {
        Box::new(LogicCombine::new("or", Logic::Or))
    }

    fn chain() -> Chain {
        Chain::sequence(vec![
            Chain::op(IdentityOp::new("a", [1, 2, 3])),
            Chain::alternative(
                vec![
                    Chain::op(IdentityOp::new("b", [10, 0, 0])),
                    Chain::sequence(vec![
                        Chain::op(IdentityOp::new("c", [2, 2, 2])),
                        Chain::op(IdentityOp::new("d", [3, 0, 0])),
                    ]),
                ],
                0,
            )
            .unwrap(),
            Chain::op(IdentityOp::new("e", [0, 0, 5])),
        ])
    }

    #[test]
    fn sequence_reaches_add_and_alternative_reaches_take_the_max() {
        let chain = chain();
        // axis 0: 1 + max(10, 2 + 3) + 0 = 11
        assert_eq!(chain.reach(0, 1000), 11);
        // axis 1: 2 + max(0, 2 + 0) + 0 = 4
        assert_eq!(chain.reach(1, 1000), 4);
        // axis 2: 3 + max(0, 2 + 0) + 5 = 10
        assert_eq!(chain.reach(2, 1000), 10);
    }

    #[test]
    fn slots_flatten_sequences_but_never_split_an_alternative() {
        let chain = chain();
        let slots = chain.slots();
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].display_name(), "a");
        assert_eq!(slots[1].display_name(), "alt[0](b|seq(c>d))");
        assert_eq!(slots[2].display_name(), "e");
        // The alternative slot still reports the conservative max reach.
        assert_eq!(slots[1].reach(0, 1000), 10);
    }

    #[test]
    fn one_op_without_a_declared_constant_collapses_the_whole_chain() {
        let with_constants = Chain::sequence(vec![
            Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
            Chain::op(AffineOp::new("plus_one", 1.0, 1.0, [0, 0, 0])),
        ]);
        assert_eq!(with_constants.constant_maps_to(3.0), Some(7.0));

        let with_an_opaque_op = Chain::sequence(vec![
            Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
            Chain::op(OpaqueOp::new("opaque", [0, 0, 0])),
            Chain::op(AffineOp::new("plus_one", 1.0, 1.0, [0, 0, 0])),
        ]);
        assert_eq!(with_an_opaque_op.constant_maps_to(3.0), None);
    }

    #[test]
    fn apply_walks_the_same_tree_reach_folds_over() {
        let chain = Chain::sequence(vec![
            Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
            Chain::alternative(
                vec![
                    Chain::op(AffineOp::new("plus_ten", 1.0, 10.0, [0, 0, 0])),
                    Chain::op(AffineOp::new("plus_one", 1.0, 1.0, [0, 0, 0])),
                ],
                1,
            )
            .unwrap(),
        ]);
        let input: Voxels = ndarray::Array3::from_elem((2, 2, 2), 3.0).into();
        let mut out = Voxels::zeros(Dtype::F64, [2, 2, 2]).unwrap();
        chain
            .apply(&input, &mut out, &Anchor::whole([2, 2, 2]))
            .unwrap();
        assert!(out.view::<f64>().unwrap().iter().all(|&value| value == 7.0));
        // and the taken branch is what `constant_maps_to` reports
        assert_eq!(chain.constant_maps_to(3.0), Some(7.0));
    }

    // ----------------------------------------------------------- fan-in --

    /// A stem, a fan-in of two arms with a nested sequence in one of them, and
    /// a tail. The same shape the tree test above uses for `Alternative`, so
    /// the two folds can be compared directly.
    fn diamond() -> Chain {
        Chain::sequence(vec![
            Chain::op(IdentityOp::new("a", [1, 2, 3])),
            Chain::parallel(
                vec![
                    Chain::op(IdentityOp::new("left", [10, 0, 0])),
                    Chain::sequence(vec![
                        Chain::op(IdentityOp::new("right1", [2, 2, 2])),
                        Chain::op(IdentityOp::new("right2", [3, 0, 0])),
                    ]),
                ],
                or(),
            )
            .unwrap(),
            Chain::op(IdentityOp::new("e", [0, 0, 5])),
        ])
    }

    /// The max across branches, then the combine's, then the sequence adds —
    /// which is numerically what `Alternative` gives for the same branches
    /// because this combine reaches zero.
    ///
    /// That coincidence is the point rather than an accident of the fixture: it
    /// is exactly why a diamond could be written as an alternation and pass 903
    /// reach comparisons. The tests below are the ones that can tell them apart.
    #[test]
    fn parallel_reaches_take_the_max_across_branches_and_the_combine_adds() {
        let chain = diamond();
        // axis 0: 1 + (max(10, 2 + 3) + 0) + 0 = 11
        assert_eq!(chain.reach(0, 1000), 11);
        // axis 1: 2 + (max(0, 2 + 0) + 0) + 0 = 4
        assert_eq!(chain.reach(1, 1000), 4);
        // axis 2: 3 + (max(0, 2 + 0) + 0) + 5 = 10
        assert_eq!(chain.reach(2, 1000), 10);

        // A combine that reaches on its own account adds on top of the max,
        // rather than being folded into it.
        let with_a_reaching_combine = Chain::parallel(
            vec![
                Chain::op(IdentityOp::new("left", [10, 0, 0])),
                Chain::op(IdentityOp::new("right", [4, 0, 0])),
            ],
            Box::new(ReachingCombine),
        )
        .unwrap();
        assert_eq!(with_a_reaching_combine.reach(0, 1000), 10 + 6);
    }

    /// A combine that is not voxelwise, so the `+ combine` term is visible.
    struct ReachingCombine;

    impl Combine for ReachingCombine {
        fn name(&self) -> &'static str {
            "reaching"
        }
        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            6
        }
        fn accepts(&self, inputs: &[Dtype]) -> bool {
            inputs.len() == 2
        }
        fn produces(&self, inputs: &[Dtype]) -> Dtype {
            inputs[0]
        }
        fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
            Ok(inputs[0])
        }
        fn apply(&self, inputs: &[Voxels], out: &mut Voxels, _at: &Anchor) -> Result<()> {
            out.assign(&inputs[0])
        }
    }

    #[test]
    fn slots_flatten_sequences_but_never_split_a_parallel() {
        let chain = diamond();
        let slots = chain.slots();
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].display_name(), "a");
        assert_eq!(slots[1].display_name(), "par(left&seq(right1>right2))>or");
        assert_eq!(slots[2].display_name(), "e");
        assert_eq!(slots[1].reach(0, 1000), 10);
    }

    /// The distinction the 903 passing comparisons could not see, in the one
    /// fold where getting it wrong leaves a hole rather than costing reads.
    ///
    /// Same two branches, same declarations, two node types: an `Alternative`
    /// declares only what its live branch writes, and a `Parallel` declares
    /// every branch's, because every branch writes.
    #[test]
    fn a_parallel_declares_every_branchs_side_outputs_where_an_alternative_declares_one() {
        let volume = [4usize, 4, 4];
        let arms = || {
            vec![
                Chain::op(SideOutputOp::new("left", [0, 0, 0]).with_side("m", Dtype::I32, 0, 1)),
                Chain::op(SideOutputOp::new("right", [0, 0, 0]).with_side("m", Dtype::I32, 0, 1)),
            ]
        };

        let exclusive = Chain::alternative(arms(), 0).unwrap();
        let names: Vec<String> = exclusive
            .side_outputs(volume)
            .into_iter()
            .map(|output| output.name)
            .collect();
        assert_eq!(names, vec!["left.m".to_string()]);

        let concurrent = Chain::parallel(arms(), or()).unwrap();
        let names: Vec<String> = concurrent
            .side_outputs(volume)
            .into_iter()
            .map(|output| output.name)
            .collect();
        assert_eq!(names, vec!["left.m".to_string(), "right.m".to_string()]);

        // and the routing is the inverse of that concatenation
        let valid = Region::new(&[0, 0, 0], &volume);
        assert!(concurrent.side_region(0, &valid, volume).is_ok());
        assert!(concurrent.side_region(1, &valid, volume).is_ok());
        assert!(concurrent.side_region(2, &valid, volume).is_err());
    }

    /// Work sums where an alternation maxes, which is the other fold where the
    /// two readings of `max` give different answers.
    #[test]
    fn a_parallel_costs_every_branch_where_an_alternative_costs_the_widest() {
        let arms = || {
            vec![
                Chain::op(IdentityOp::new("left", [0, 0, 0]).with_cost(3.0)),
                Chain::op(IdentityOp::new("right", [0, 0, 0]).with_cost(5.0)),
            ]
        };
        assert_eq!(Chain::alternative(arms(), 1).unwrap().cost_per_voxel(), 5.0);
        // 3 + 5 branches, plus one pair's worth of combine at the measured
        // 0.49.
        assert_eq!(
            Chain::parallel(arms(), or()).unwrap().cost_per_voxel(),
            3.0 + 5.0 + 0.49
        );
    }

    /// Every branch runs, so every branch's result is combined — asserted on
    /// values that make dropping either one visible.
    #[test]
    fn apply_runs_every_branch_and_combines_them() {
        // Under this module's mask convention `Or` is set where either arm is
        // non-zero. Left sets nothing, right sets everything, so a fan-in that
        // ran only the left arm would produce all zeros.
        let chain = Chain::parallel(
            vec![
                Chain::op(AffineOp::new("zero", 0.0, 0.0, [0, 0, 0])),
                Chain::op(AffineOp::new("one", 0.0, 1.0, [0, 0, 0])),
            ],
            or(),
        )
        .unwrap();
        let input: Voxels = ndarray::Array3::from_elem((2, 2, 2), 3.0).into();
        let mut out = Voxels::zeros(Dtype::F64, [2, 2, 2]).unwrap();
        chain
            .apply(&input, &mut out, &Anchor::whole([2, 2, 2]))
            .unwrap();
        assert!(out.view::<f64>().unwrap().iter().all(|&value| value == 1.0));
        // and the constant algebra reports the same answer the run produced
        assert_eq!(chain.constant_maps_to(3.0), Some(1.0));
    }

    /// A conjunction over branches *and* the combine, unlike the union
    /// `side_outputs` folds and the selection `Alternative` folds.
    #[test]
    fn a_parallel_folds_a_constant_only_when_every_branch_and_the_combine_do() {
        let opaque_branch = Chain::parallel(
            vec![
                Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
                Chain::op(OpaqueOp::new("opaque", [0, 0, 0])),
            ],
            or(),
        )
        .unwrap();
        assert_eq!(opaque_branch.constant_maps_to(3.0), None);

        let silent_combine = Chain::parallel(
            vec![
                Chain::op(AffineOp::new("double", 2.0, 0.0, [0, 0, 0])),
                Chain::op(AffineOp::new("plus_one", 1.0, 1.0, [0, 0, 0])),
            ],
            Box::new(ReachingCombine),
        )
        .unwrap();
        assert_eq!(silent_combine.constant_maps_to(3.0), None);
    }

    // -------------------------------------------- the guards, seen firing --

    #[test]
    fn a_parallel_of_fewer_than_two_branches_is_refused() {
        let refusal = |branches: Vec<Chain>| match Chain::parallel(branches, or()) {
            Ok(_) => panic!("a fan-in with fewer than two branches was accepted"),
            Err(err) => err.to_string(),
        };
        let empty = refusal(Vec::new());
        assert!(empty.contains("no branches"), "got: {empty}");

        let one = refusal(vec![Chain::op(IdentityOp::new("a", [0, 0, 0]))]);
        assert!(one.contains("one branch"), "got: {one}");
    }

    /// A combine handed an element type it cannot join is refused when the
    /// chain is asked what it produces — which is plan time, not run time.
    #[test]
    fn a_combine_that_cannot_accept_a_branchs_dtype_is_refused() {
        let chain = Chain::parallel(
            vec![
                Chain::op(NonZeroOp::new("mask", [0, 0, 0])),
                Chain::op(IdentityOp::new("image", [0, 0, 0])),
            ],
            or(),
        )
        .unwrap();
        let err = chain.produces(Dtype::F64).unwrap_err().to_string();
        assert!(
            err.contains("does not accept [bool, float64]"),
            "got: {err}"
        );

        // and `apply` cannot get past it either, because it asks the same
        // question first
        let input: Voxels = ndarray::Array3::from_elem((2, 2, 2), 1.0).into();
        let mut out = Voxels::zeros(Dtype::F64, [2, 2, 2]).unwrap();
        assert!(chain
            .apply(&input, &mut out, &Anchor::whole([2, 2, 2]))
            .is_err());
    }

    /// Branch shapes are the combine's business, and this combine refuses two
    /// extents it cannot pair voxel for voxel.
    #[test]
    fn a_voxelwise_combine_refuses_branches_of_different_extents() {
        let chain = Chain::parallel(
            vec![
                Chain::op(IdentityOp::new("whole", [0, 0, 0])),
                Chain::op(DecimateOp::new("half", 2)),
            ],
            or(),
        )
        .unwrap();
        let err = chain.output_shape([8, 4, 4]).unwrap_err().to_string();
        assert!(
            err.contains("branch 0 produced [8, 4, 4] and branch 1 produced [4, 4, 4]"),
            "got: {err}"
        );
    }
}
