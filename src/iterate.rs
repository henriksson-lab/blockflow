// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// `docs/design/BLOCK_OPS.md` §"Levels have a lifetime, and iteration has
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
// count turns out to be.** Substage 0 reads the phase's input level over
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
// level allocation and no phase structure, so there is nothing in the binding
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
use crate::error::{Error, Result};
use crate::geometry::BlockGrid;
use crate::op::Anchor;
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
    /// and substage 0 is handed the phase's input level.
    ///
    /// Exactly one operand must be this, because it is what the iteration
    /// iterates and it is what the two private buffers ping-pong between. An op
    /// declaring none would compute the same answer forever; an op declaring two
    /// would need two ping-pongs, which is a different shape and is not this one.
    Running,
    /// The phase's input level, re-read at **every** substage.
    ///
    /// The motivating case is a step of the form `g <- min(dilate(g), f)`, where
    /// `f` is fixed for the whole iteration and read pointwise. Supplying it only
    /// at substage 0 would silently compute a different algorithm — which is
    /// precisely the failure `ops/deconvolve.rs`'s header warns about — so
    /// `tests/iterative_phase.rs` carries a test that fails if it is.
    ///
    /// Today every `Fixed` operand is a view of the same array, because a phase
    /// has one input level. When levels become a DAG this variant is where the
    /// level number goes, and nothing else about the interface moves.
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
    /// input level — which is what substage 0's `Running` operand *is* — reads it.
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
    /// planner prices one substage and records that the phase is unbounded; since
    /// nothing can fuse across it — self-contained, external reach one substage's
    /// worth — that changes the predicted duration and not the plan's shape.
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
