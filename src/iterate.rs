// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// `docs/design/BLOCK_OPS.md` §"Images have a lifetime, and iteration has
// substages". **One phase that internally runs an unknown number of substages,
// with more than one operand available at every substage.**
//
// What this is for
// ----------------
// An iteration whose step depends on two operands — a running estimate and
// something fixed for the whole loop — has had nowhere to live. `Chain::
// Sequence` hands each child exactly one buffer, its predecessor's output, so a
// chain of `n` one-step ops feeds step `k` its own predecessor's estimate *as the
// fixed operand*, which is a different and much worse algorithm;
// `Chain::Parallel` shares one input across branches and joins them once, which
// is a diamond and not a recurrence. `ops/deconvolve.rs` states the problem in
// exactly those terms and resolves it by carrying the loop inside one op, at a
// reach of `2 * radius * iterations` — 72 voxels at twelve iterations.
//
// The third shape is this one, and the reason it is worth building is not
// tidiness:
//
// **The phase's external reach is the per-substage reach, whatever the substage
// count turns out to be.** Substage 0 reads the phase's input image over
// core ⊕ r and writes its core of a private buffer; substage `k` reads the
// private buffer substage `k-1` wrote, over core ⊕ r, and its neighbours *wrote*
// their cores of it. The depth is paid in private round trips, not in halo,
// because each substage is a real exchange point: blocks read each other's
// **outputs** rather than re-deriving them from an ever-wider read of the
// original. A forty-substage phase has the reach of a one-substage phase, and
// `tests/iterative_phase.rs` asserts exactly that.
//
// Two private buffers, not `N`
// ----------------------------
// The buffer written at substage `k` already holds the output of substage `k-2`,
// which nothing will read again — so two alternating buffers suffice and live
// storage is `O(1)` in the substage count. That is what makes the shape
// affordable, and it is also what will later make a skipped write free: a block
// that did not change can leave the buffer holding `k-2`'s value, which is its
// own unchanged value. The skip is not built here. Correctness first, and a
// trivial executor is the oracle the optimisation will be tested against.
//
// Where the loop stops
// --------------------
// **At convergence, not at a stated count.** The count appears in no reach, no
// image allocation and no phase structure, so there is nothing in the binding
// half of a plan for it to be: the same plan is shipped to a worker before any
// data is seen and compared against alternatives exactly as before, and
// determinism is untouched — same plan, same data, same substage count, same
// answer. What is given up is that the *execution trace* is a function of
// metadata alone, which was never one of the guarantees.
//
// The implementation here is the trivial, obviously-correct one: every block
// runs every substage, and after each substage the executor asks whether
// anything changed anywhere. In a single process that is a loop over the blocks
// and needs no barrier primitive. The per-block skip, the frontier and the dirty
// set are the optimisation and they come later.
//
// **The runaway limit is required and is a guard, not a parameter**, on exactly
// `ops::skeleton::PassLimit`'s argument: exceeding it is an error naming the op
// and the count, never a truncated answer, because a partially converged volume
// is plausible, well-formed and wrong. Its *derivation* belongs with the op — an
// op that peels from the surface inward is bounded by half the shortest axis, one
// that spreads along paths by the longest path the data permits — so it is
// returned by the op rather than computed centrally. That is also why this is a
// separate type from `PassLimit` rather than the same one: `PassLimit::
// for_volume` carries the peeling derivation, and a framework type offering it
// would be offering one op's bound to every op.

use crate::decomposition::PhaseDecomposition;
use crate::dtype::Dtype;
use crate::env::BlockBuf;
use crate::error::{Error, Result};
use crate::geometry::BlockGrid;
use crate::op::{Anchor, SourceInput, SourceInputs};
use crate::region::Region;
use crate::sidecar::Lifecycle;
use crate::voxels::Voxels;

/// Where one of a substage's operands comes from.
///
/// **An enum rather than "the first one is special".** An op declares a list and
/// the executor loops over it, so an iteration with three operands costs the
/// same code as one with two; nothing here special-cases the arity. What the
/// variants say is *which array* an operand is a view of, which is the part that
/// cannot be derived from a position in a list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    /// The running estimate: substage `k` is handed what substage `k-1` wrote,
    /// and substage 0 is handed the phase's input image.
    ///
    /// Exactly one operand must be this, because it is what the iteration
    /// iterates and it is what the two private buffers ping-pong between. An op
    /// declaring none would compute the same answer forever; an op declaring two
    /// would need two ping-pongs, which is a different shape and is not this one.
    Running,
    /// The phase's input image, re-read at **every** substage.
    ///
    /// The motivating case is a step of the form `g <- min(dilate(g), f)`, where
    /// `f` is fixed for the whole iteration and read pointwise. Supplying it only
    /// at substage 0 would silently compute a different algorithm — which is
    /// precisely the failure `ops/deconvolve.rs`'s header warns about — so
    /// `tests/iterative_phase.rs` carries a test that fails if it is.
    ///
    /// Today every `Fixed` operand is a view of the same array, because a phase
    /// has one input image. When images become a DAG this variant is where the
    /// image number goes, and nothing else about the interface moves.
    ///
    /// **A second image is now expressible, and this variant has not moved to
    /// it.** [`Chain::Source`](crate::op::Chain::Source) is a leaf that reads a
    /// stored image at the block's read extent, and a phase records which images
    /// it reads in `PhaseDecomposition::source_images`. That is the general
    /// form of what this variant does narrowly — but an iterative phase owns no
    /// chain slot, so it has no leaf to carry the number and would need the
    /// image on `SubstageOperand` instead. Adding it is a change to this enum
    /// and to `run_iterative_phase`'s operand gathering, and nothing else; it is
    /// left undone rather than guessed at, because no op has asked for it yet
    /// and an untested variant of a two-array iteration is exactly the kind of
    /// plausible thing that would be wrong.
    Fixed,
}

/// One operand of a substage, and what the substage reads of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubstageOperand {
    pub operand: Operand,
    /// Voxels read beyond the voxel written, per axis, **per substage**.
    ///
    /// This is the number the phase's halo comes from — the widest over the
    /// operands — and it is stated here rather than in a separate `reach` method
    /// on the op because everything an op reads it reads from an operand, so a
    /// second statement of the same quantity could only drift from this one.
    /// `Chain::reach` and `Chain::apply` are folded over one tree for the same
    /// reason.
    pub reach: [usize; 3],
}

impl SubstageOperand {
    pub fn running(reach: [usize; 3]) -> Self {
        Self {
            operand: Operand::Running,
            reach,
        }
    }

    pub fn fixed(reach: [usize; 3]) -> Self {
        Self {
            operand: Operand::Fixed,
            reach,
        }
    }
}

/// How many substages an iteration is allowed before it is declared broken.
///
/// **A guard, not a parameter**, and the distinction is the one
/// `ops::skeleton::PassLimit` settles: an op asked for exactly `n` steps has `n`
/// in its answer, whereas running to convergence has no `n` in its definition at
/// all and any number attached to it is there only to stop a non-terminating
/// iteration from running forever with no diagnostic.
///
/// So exceeding it is an **error naming the op and the count**, never a silent
/// truncation. A partially converged volume is plausible, well-formed and wrong,
/// which is the failure mode this crate is arranged against; a loud one says
/// either that the iteration does not converge or that the limit was set below
/// what this data needs, and both are things a caller can act on.
///
/// There is deliberately no `for_volume` here. The bound depends on what the
/// iteration *does* — an op that peels from the surface inward is bounded by half
/// the shortest axis, one that spreads along paths by the longest path the data
/// permits, up to the volume's diameter — so the derivation belongs with the op
/// and a framework-supplied one would be one op's bound offered to every op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubstageLimit(usize);

impl SubstageLimit {
    pub fn of(substages: usize) -> Result<Self> {
        if substages == 0 {
            return Err(Error::InvalidArgument(
                "a substage limit of zero would refuse before doing anything; the limit is a \
                 backstop, and one that fires immediately is a limit that means \"do not run\""
                    .to_string(),
            ));
        }
        Ok(Self(substages))
    }

    pub fn substages(self) -> usize {
        self.0
    }
}

/// What one substage of one block is handed.
///
/// **Every operand is handed the same extent** — the block's read extent, core
/// plus the phase's halo — even where its own declared reach is narrower. That is
/// a deliberate simplification with a stated cost: one coordinate origin for the
/// whole substage, so an op indexes all of its operands the same way and cannot
/// mix two origins up, against re-reading a pointwise operand over a halo it does
/// not use. Per-operand extents would save that IO and are the thing to build if
/// it ever shows up in a measurement; nothing in the declaration has to change
/// for it, because the reaches are already stated per operand.
pub struct Substage<'a> {
    index: usize,
    operands: &'a [&'a Voxels],
    at: &'a Anchor,
}

impl<'a> Substage<'a> {
    /// Public so that a caller holding the whole volume can run the same kernel
    /// in a loop and get the reference answer the block-decomposed run is checked
    /// against. That comparison is the bar every op in this crate meets, and the
    /// machinery has to meet it too.
    pub fn new(index: usize, operands: &'a [&'a Voxels], at: &'a Anchor) -> Self {
        Self {
            index,
            operands,
            at,
        }
    }

    /// Which substage this is, counting from zero.
    ///
    /// Most ops ignore it. One that initialises its running estimate from the
    /// input image — which is what substage 0's `Running` operand *is* — reads it.
    pub fn index(&self) -> usize {
        self.index
    }

    /// The operands, in the order [`IterativeOp::operands`] declared them.
    pub fn operands(&self) -> &[&'a Voxels] {
        self.operands
    }

    /// One operand, by declaration position.
    pub fn operand(&self, which: usize) -> Result<&'a Voxels> {
        self.operands.get(which).copied().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "substage operand {which} was asked for and {} were declared",
                self.operands.len()
            ))
        })
    }

    /// Where the buffers sit in the volume they were read from. A
    /// position-independent op ignores it; see [`Anchor`].
    pub fn at(&self) -> &Anchor {
        self.at
    }
}

/// An operation that runs to a fixed point inside one phase.
///
/// Deliberately **not** a wider [`crate::op::BlockOp`]. A block op is handed one
/// buffer and every implementation in this crate and its callers is written to
/// that signature; a substage is handed several, and is asked a question a block
/// op is never asked — *has anything changed*. The arity and the loop are the
/// difference, so they are in the type, on exactly the argument
/// [`crate::op::Combine`] is a separate trait.
///
/// `Send + Sync` for the reason `BlockOp` is: the executor shares one
/// `&dyn IterativeOp` across a phase.
pub trait IterativeOp: Send + Sync {
    fn name(&self) -> &'static str;

    /// The operands one substage is handed, and what it reads of each.
    ///
    /// No default: this is the only statement of what the op reads, so a silent
    /// empty list would be the same class of defect as a silent zero reach.
    fn operands(&self) -> Vec<SubstageOperand>;

    /// The runaway guard. See [`SubstageLimit`] for why it is required and why
    /// its derivation is the op's.
    fn limit(&self) -> SubstageLimit;

    /// Can this op be handed blocks of `dtype`?
    ///
    /// The default is `f64` and only `f64`, matching [`crate::op::BlockOp::accepts`].
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    /// Compute one substage over the whole of the block, writing into `out`.
    ///
    /// `out` has the operands' shape and their element type. Only the block's
    /// valid region is kept — the rest was computed from a halo that may itself
    /// have been computed from a shorter halo — so an op may write the whole
    /// buffer and need not know which part survives.
    fn substage(&self, at: &Substage<'_>, out: &mut Voxels) -> Result<()>;

    /// Relative compute cost per voxel per **substage**. Measured, not guessed;
    /// the default of 1.0 is a placeholder, as it is on `BlockOp`.
    ///
    /// A phase whose substage count is unknown cannot be priced as a whole. The
    /// planner prices **one** substage — and the second half of what this doc
    /// used to say, *that this changes the predicted duration and not the plan's
    /// shape*, is measured false. It changes the shape too, through the block
    /// edge.
    ///
    /// A constant multiplier is neutral only if it multiplies the **whole**
    /// price, and `S` substages do not. A substage reads and computes; the image
    /// is written **once**, at the fixed point, because the substages ping-pong
    /// two private buffers and `strategy::run_iterative_phase` writes only after
    /// the loop. The true shape is `S x (read + compute) + write`, so ranking at
    /// `S == 1` weighs the write `S` times too heavily against the rest, and the
    /// residual `(S - 1) x (read + compute)` is a function of the block edge
    /// through the read amplification. Measured: the choice departs from the
    /// one-substage choice at counts as ordinary as two and four, never below
    /// three workers, and the one-substage choice costs up to **1.125x** the
    /// right one.
    ///
    /// What *is* true, and now has evidence rather than an argument: **the
    /// substage count does not vary with the block edge.** Swept over thirteen
    /// lattices including `[1, 1, 1]` — where every step of the propagation is a
    /// halo exchange — four reaches and two data shapes including a serpentine
    /// forcing a long geodesic, the count is the whole-volume count every time.
    /// The halo is one substage's reach wide and holds the neighbours' cores from
    /// the previous substage, so a seam is crossed at exactly the rate the inside
    /// of a block is, which is what this module's header claims.
    fn cost_per_voxel(&self) -> f64 {
        1.0
    }
}

/// What one substage reads beyond the voxel it writes: the widest of the
/// operands' reaches, per axis.
///
/// **This is the phase's whole external reach**, and the headline property of
/// the shape: it does not grow with the substage count, because the depth is paid
/// in private round trips rather than in halo.
pub fn substage_reach(op: &dyn IterativeOp) -> [usize; 3] {
    let mut reach = [0usize; 3];
    for operand in op.operands() {
        for axis in 0..3 {
            reach[axis] = reach[axis].max(operand.reach[axis]);
        }
    }
    reach
}

/// Everything about an iterative op that can be checked without data.
///
/// Called from [`iterative_phase`] when a plan is built and again from
/// `check_phase_work` when one is run, on exactly the argument
/// `check_block_constraints` is re-run in the executor: a plan may arrive from
/// any strategy or off a wire, and one that satisfied a rule when it was chosen
/// is not thereby a plan that satisfies it now.
pub fn check_iterative(op: &dyn IterativeOp) -> Result<()> {
    let operands = op.operands();
    if operands.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "iterative op {:?} declares no operands, so a substage would read nothing and the \
             iteration would have nothing to converge.",
            op.name()
        )));
    }
    let running = operands
        .iter()
        .filter(|operand| operand.operand == Operand::Running)
        .count();
    if running != 1 {
        return Err(Error::InvalidArgument(format!(
            "iterative op {:?} declares {running} `Operand::Running` operand(s); exactly one is \
             required. The running operand is what the iteration iterates and what the two \
             private buffers alternate between: none would recompute one answer forever, and two \
             would be a different shape than this one.",
            op.name()
        )));
    }
    Ok(())
}

/// The decomposition phase an iterative op runs as.
///
/// The reach **and** the halo are one substage's, which is the whole point; there
/// is no factor of the substage count anywhere in this function, and there is
/// nothing for one to multiply. The phase owns no chain slot, on exactly
/// [`crate::fragment::fragment_phase`]'s argument: `Decomposition::
/// op_names_in_order` pairs with `slot_order`, and a name there with no
/// `OpApplied` event to match it breaks the check for every mixed decomposition.
pub fn iterative_phase(op: &dyn IterativeOp, grid: BlockGrid) -> Result<PhaseDecomposition> {
    check_iterative(op)?;
    let reach = substage_reach(op);
    Ok(PhaseDecomposition::derive(
        Vec::new(),
        Vec::new(),
        reach,
        reach,
        grid,
    ))
}

/// One block's view of an iterative map-reduce substage.
///
/// The state is deliberately not stored here: every block is handed the same
/// byte slice by `IterativeReduceOp::map_block`, so the common operand is in
/// the method signature rather than hidden in the view. The view is the block's
/// geometry and, when the op asks for it, the pixels read for that geometry.
pub struct ReduceBlock<'a> {
    pub phase: usize,
    pub index: [usize; 3],
    pub grid: &'a BlockGrid,
    pub core: &'a Region,
    pub read: &'a Region,
    pub valid: &'a Region,
    pub at: Anchor,
    pixels: Option<&'a BlockBuf>,
    sources: SourceInputs<'a>,
}

impl<'a> ReduceBlock<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        phase: usize,
        index: [usize; 3],
        grid: &'a BlockGrid,
        core: &'a Region,
        read: &'a Region,
        valid: &'a Region,
        at: Anchor,
        pixels: Option<&'a BlockBuf>,
        sources: SourceInputs<'a>,
    ) -> Self {
        Self {
            phase,
            index,
            grid,
            core,
            read,
            valid,
            at,
            pixels,
            sources,
        }
    }

    pub fn pixels(&self) -> Result<&BlockBuf> {
        self.pixels.ok_or_else(|| {
            Error::InvalidArgument(
                "this iterative reduce op asked for pixels, but it declares \
                 `reads_pixels() == false`, so the executor read none. An op that needs \
                 pixels says so."
                    .to_string(),
            )
        })
    }

    pub fn sources(&self) -> &SourceInputs<'a> {
        &self.sources
    }
}

/// The result of one global update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateUpdate {
    pub state: Vec<u8>,
    pub converged: bool,
}

impl StateUpdate {
    pub fn continuing(state: Vec<u8>) -> Self {
        Self {
            state,
            converged: false,
        }
    }

    pub fn converged(state: Vec<u8>) -> Self {
        Self {
            state,
            converged: true,
        }
    }
}

/// One block's partial contribution to an iterative global update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partial {
    pub index: [usize; 3],
    pub bytes: Vec<u8>,
}

/// An iteration whose carried value is a small global state, not an image.
///
/// Each substage has the shape:
///
/// ```text
/// state[k + 1] = update(state[k], reduce(map(block, state[k])))
/// ```
///
/// The executor owns the barriers: every block sees the same state, partials
/// are reduced in lattice order, and the next state is visible only after the
/// whole set has been reduced.
pub trait IterativeReduceOp: Send + Sync {
    fn name(&self) -> &'static str;

    fn limit(&self) -> SubstageLimit;

    fn reach(&self) -> [usize; 3] {
        [0, 0, 0]
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        Vec::new()
    }

    fn initial_state(&self, volume: [usize; 3]) -> Result<Vec<u8>>;

    fn map_block(&self, substage: usize, state: &[u8], block: &ReduceBlock<'_>) -> Result<Vec<u8>>;

    fn update(&self, substage: usize, state: &[u8], partials: &[Partial]) -> Result<StateUpdate>;

    /// Where the converged state is written.
    ///
    /// One sidecar fragment is written, under block index `[0, 0, 0]`. A stream
    /// rather than `Stats` makes the result data, not telemetry, and lets a
    /// later phase or caller read it through the same sidecar store as other
    /// small objects.
    fn state_stream(&self) -> &'static str;

    fn state_lifecycle(&self) -> Lifecycle {
        Lifecycle::Persistent
    }

    fn cost_per_voxel(&self) -> f64 {
        1.0
    }
}

pub fn check_iterative_reduce(op: &dyn IterativeReduceOp) -> Result<()> {
    if op.state_stream().is_empty() {
        return Err(Error::InvalidArgument(format!(
            "iterative reduce op {:?} declares an empty state stream; the final state would be \
             unreachable.",
            op.name()
        )));
    }
    crate::sidecar::check_stream_name(op.state_stream())?;
    Ok(())
}

pub fn iterative_reduce_phase(
    op: &dyn IterativeReduceOp,
    grid: BlockGrid,
) -> Result<PhaseDecomposition> {
    check_iterative_reduce(op)?;
    let volume = grid.volume();
    let edge = grid.block();
    let reach = op.reach();
    let mut halo = reach;
    let mut images = Vec::new();
    let mut supplied = Vec::new();
    for input in op.source_inputs(volume) {
        let wanted = input.reach.in_voxels(edge);
        for (axis, value) in halo.iter_mut().enumerate() {
            let (lo, hi) = wanted.axis(axis).bound(volume[axis]);
            *value = (*value).max(lo).max(hi);
        }
        if input.image.is_supplied() {
            let dtype = input.dtype.ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "iterative reduce op {:?} reads {}, and nothing says what it holds. An image \
                     the run writes has its element type in the fold of the chain that wrote it; \
                     a supplied input is produced by no phase, so the reader is the only \
                     statement there is.",
                    op.name(),
                    crate::assemble::describe_image(input.image.index())
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

#[cfg(test)]
mod tests {
    use super::*;

    struct Declaring(Vec<SubstageOperand>);

    impl IterativeOp for Declaring {
        fn name(&self) -> &'static str {
            "declaring"
        }

        fn operands(&self) -> Vec<SubstageOperand> {
            self.0.clone()
        }

        fn limit(&self) -> SubstageLimit {
            SubstageLimit::of(4).expect("a positive limit")
        }

        fn substage(&self, at: &Substage<'_>, out: &mut Voxels) -> Result<()> {
            out.assign(at.operand(0)?)
        }
    }

    #[test]
    fn the_phase_reach_is_the_widest_operand_and_not_their_sum() {
        let op = Declaring(vec![
            SubstageOperand::running([2, 1, 0]),
            SubstageOperand::fixed([0, 3, 0]),
        ]);
        assert_eq!(substage_reach(&op), [2, 3, 0]);
    }

    #[test]
    fn an_op_with_no_running_operand_is_refused_by_name() {
        let op = Declaring(vec![SubstageOperand::fixed([1, 1, 1])]);
        let message = check_iterative(&op).unwrap_err().to_string();
        assert!(message.contains("declaring"), "{message}");
        assert!(message.contains("exactly one is required"), "{message}");
    }

    #[test]
    fn two_running_operands_are_refused_because_there_is_one_ping_pong() {
        let op = Declaring(vec![
            SubstageOperand::running([1, 0, 0]),
            SubstageOperand::running([1, 0, 0]),
        ]);
        assert!(check_iterative(&op).is_err());
    }

    #[test]
    fn a_zero_limit_is_refused_rather_than_meaning_do_not_run() {
        assert!(SubstageLimit::of(0).is_err());
        assert_eq!(SubstageLimit::of(1).unwrap().substages(), 1);
    }
}
