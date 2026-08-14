// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Assembly, not planning.** Every number a `Decomposition` holds is
// parity-visible: change a slot range, a halo, a per-phase element type or which
// level an arm reads and the output changes. So a builder that *decided* any of
// them would be a planner wearing a convenience's name, and the two must not be
// the same object — `strategy::Strategy::decompose` is where a plan is chosen.
// What this module removes is the **bookkeeping** between a caller who already
// knows what the phases are and a `Decomposition` that records them, and nothing
// else. Everything it produces goes through `Decomposition::check`,
// `check_dtypes`, `check_source_levels`, `check_block_constraints` and
// `check_phase_work` unchanged, and `PlanBuilder::finish` runs all five so that
// a plan that would be refused at execution is refused where it was written.
//
// Where it came from
// ------------------
// The first mixed-kind multi-phase plan built on this crate was hand-assembled,
// and its author kept a list of what that cost. Six of the eight entries were
// one thing each: **the slot cursor, the names, the per-phase reach, the
// fragment phase's element type, the `PhaseWork` list, and remembering to call
// `declare_source_levels`.** Those six are this builder's whole remit. The list
// is worth reading as a design constraint rather than as history, because each
// entry is a place where writing the *wrong* number compiles, runs, and produces
// a different answer with no error at all:
//
// | what was hand-maintained | how it went wrong | what it is now |
// |---|---|---|
// | a phase index passed to an op's constructor | a literal that is off by one reads a *different generation* of a stream — a wrong answer, not an error | [`Phase`], which can only come from the builder that made the phase |
// | slot ranges and names, as two parallel lists with three cursors | a slice off by one names the wrong op in every log and event | derived here from the chain fragment itself |
// | a phase's reach, taken from a fragment before it was moved into the sequence | had to be taken *early*, which is the easiest ordering in the file to get wrong | taken here, in the one place that holds the fragment |
// | `phase.dtype = Some(Dtype::U32)` after `fragment_phase` | forgetting it is refused, but by a message about a level's width | asked of the op, which is the only thing that knows |
// | a second list of `PhaseWork`, tied to the phase list by a `debug_assert_eq!` on its length | a length check catches the count, never the order | **not a second list**: the builder records the kind *with* the phase |
// | `declare_source_levels`, called by hand at the end | forgetting it does not fail `check()`; it fails as a level freed before its second reader | part of [`PlanBuilder::finish`] |
//
// What is a type here rather than a check, and why that is the point
// ------------------------------------------------------------------
// [`Phase`] and [`Level`] are both a `usize` and they are deliberately not
// interchangeable. A phase index and a level number are different quantities
// that are numerically close — phase `p` writes level `p + 1` — so the mistake
// worth making impossible is not "a number out of range" but "the *other*
// number, which is also in range". `Chain::source` takes a [`Level`] and an op
// that reads another phase's stream takes a [`Phase`], so passing one where the
// other belongs is a compile error rather than a plan that runs and answers
// differently.
//
// The `PhaseWork` list is the sharper case. It used to be two lists that a
// length assertion compared; here the kind is stored *in* the phase list, so
// there is no second list to disagree — `Assembly::work` derives the borrowed
// view on demand. That is the difference between a check and a shape.
//
// What it deliberately does not absorb
// ------------------------------------
// * **The block lattice.** A grid is a planning decision with a cost model
//   behind it, so the caller supplies it; [`PlanBuilder::regrid`] is how a plan
//   that changes lattice mid-way says so, and it is a statement rather than an
//   inference.
// * **Per-block source regions** (`PhaseDecomposition::with_sources`). A
//   cross-grid mapping is the operation, not bookkeeping about it.
// * **Anything a `Hints` holds.** The builder produces the binding half only.
//
// The cost of the cut, stated
// ---------------------------
// The builder owns the ops of its non-pixel phases, because a `PhaseWork`
// borrows them and something has to outlive the run. That means an op cannot be
// inspected by the caller after it goes in. In exchange the caller no longer
// holds an op *and* the phase index it was built for as two fields that can
// disagree, which was the failure this is here to remove.

use crate::decomposition::{
    check_block_constraints, check_dtypes, check_source_levels, Decomposition, PhaseDecomposition,
};
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::fragment::{check_phase_work, fragment_phase, FragmentOp, PhaseWork};
use crate::geometry::BlockGrid;
use crate::iterate::{iterative_phase, IterativeOp};
use crate::op::Chain;
use crate::strategy::Workflow;

/// A level of the plan: level 0 is the input, level `p + 1` is what phase `p`
/// wrote.
///
/// A newtype over the index and not an alias, because the whole reason it exists
/// is to be a *different type* from [`Phase`]. `From<usize>` is implemented so
/// that every caller who already writes a literal keeps working; what it buys is
/// that a caller holding a phase handle cannot pass it where a level belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Level(usize);

impl Level {
    /// The level number, for the places that index by it.
    pub fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for Level {
    fn from(level: usize) -> Self {
        Level(level)
    }
}

impl From<Level> for usize {
    fn from(level: Level) -> Self {
        level.0
    }
}

impl std::fmt::Display for Level {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A phase of the plan under assembly: where it landed, and the level it writes.
///
/// **The handle is the whole of item one.** An op that reads another phase's
/// fragment stream needs that phase's index, and the index is only known once
/// the phase exists — so hand-assembly forced an ordering (build the producing
/// phase, note its number, build the op, build the consuming phase) with nothing
/// enforcing it and a literal always available as the wrong shortcut. A wrong
/// literal is not an error: a stream written by two phases holds two
/// generations, so it reads real fragments from the wrong one.
///
/// This type can only be produced by [`PlanBuilder`], and it is produced by
/// exactly the call that creates the phase. There is no constructor from a
/// number, deliberately: a number is what went wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Phase {
    index: usize,
    /// `None` for a phase that writes fragments and no pixels — which is
    /// terminal as far as levels go, and is a fact about the op rather than
    /// about the builder.
    writes: Option<Level>,
}

impl Phase {
    /// Where this phase sits in the plan.
    pub fn index(self) -> usize {
        self.index
    }

    /// The level this phase writes, if it writes one.
    pub fn writes(self) -> Option<Level> {
        self.writes
    }

    /// The level this phase writes, or the refusal that names why there is none.
    ///
    /// The fallible form exists because whether a fragment phase writes pixels
    /// is the op's answer, not something a signature can promise. For a pixel or
    /// an iterative phase it cannot fail, and the `?` costs nothing.
    pub fn level(self) -> Result<Level> {
        self.writes.ok_or_else(|| {
            Error::InvalidArgument(format!(
                "phase {} writes fragments and no pixels, so there is no level {} for anything \
                 to read. A phase that hands pixels on says so with `FragmentOp::writes_pixels`.",
                self.index,
                self.index + 1
            ))
        })
    }
}

/// What a phase runs, owned rather than borrowed.
///
/// The owning half of [`PhaseWork`], and the reason there is no second list:
/// this sits beside the `PhaseDecomposition` it belongs to, so a phase and its
/// kind are appended by one statement and cannot be reordered relative to each
/// other.
enum Work {
    Pixels,
    Fragments(Box<dyn FragmentOp>),
    Iterate(Box<dyn IterativeOp>),
}

impl Work {
    fn borrow(&self) -> PhaseWork<'_> {
        match self {
            Work::Pixels => PhaseWork::Pixels,
            Work::Fragments(op) => PhaseWork::Fragments(&**op),
            Work::Iterate(op) => PhaseWork::Iterate(&**op),
        }
    }
}

/// Assemble a `Decomposition`, the `Workflow` it partitions, and what each phase
/// runs, in one pass and in phase order.
///
/// Every method appends. There is no way to insert, reorder or remove a phase,
/// and that is not an omission: a [`Phase`] handed out earlier records an index,
/// and an insertion would silently renumber it. A plan that needs a different
/// order is a different sequence of calls.
pub struct PlanBuilder {
    volume: [usize; 3],
    dtype: Dtype,
    grid: BlockGrid,
    /// The chain, in fragments, in chain order. Assembled by `finish` — held
    /// apart until then only because each fragment's slot count and reach have
    /// to be taken from it before it is moved.
    fragments: Vec<Chain>,
    /// Responsibility one: the slot cursor.
    slots_used: usize,
    /// Responsibilities two to four, per phase, and five beside them.
    phases: Vec<PhaseDecomposition>,
    work: Vec<Work>,
    /// The element type of the level the *next* phase will read. Folded here for
    /// the same reason `Decomposition::declare_dtypes` folds it: a fragment
    /// phase owns no slot, so nothing downstream can recover what it was handed.
    reads: Dtype,
}

impl PlanBuilder {
    /// A plan over `volume`, whose level 0 holds `dtype`, cut on `grid`.
    ///
    /// The grid is the caller's because choosing one is planning: it trades
    /// halo against task count against residency, and this builder has no cost
    /// model and must not grow one.
    pub fn new(volume: [usize; 3], dtype: Dtype, grid: BlockGrid) -> Self {
        Self {
            volume,
            dtype,
            grid,
            fragments: Vec::new(),
            slots_used: 0,
            phases: Vec::new(),
            work: Vec::new(),
            reads: dtype,
        }
    }

    /// The lattice the next phase will be cut on.
    ///
    /// `pub` because an op that reads another phase's fragments has to be built
    /// against the same lattice — fragments are keyed by block index — and the
    /// caller would otherwise have to keep its own copy of a grid the builder
    /// already holds.
    pub fn grid(&self) -> &BlockGrid {
        &self.grid
    }

    /// The element type the next phase will read.
    pub fn reads(&self) -> Dtype {
        self.reads
    }

    /// How many phases have been appended.
    pub fn n_phases(&self) -> usize {
        self.phases.len()
    }

    /// Cut every phase from here on a different lattice.
    ///
    /// A statement rather than an inference: re-blocking at a phase boundary is
    /// free — the boundary is already a materialisation — but *which* boundary
    /// to do it at is a planning decision, and the builder has no business
    /// making one. Fragment-exchanging phases must share a lattice, and
    /// `check_phase_work` says so by name if this is called between them.
    pub fn regrid(&mut self, grid: BlockGrid) {
        self.grid = grid;
    }

    /// Append a `Pixels` phase running one fragment of the chain.
    ///
    /// This is responsibilities one to three in one call: the fragment is
    /// appended to the workflow's sequence, its slot range is derived from the
    /// cursor, its names are taken from its own slots, and its reach is taken
    /// from the fragment *before* it is moved — which is the ordering that had
    /// to be got right by hand, three times, against three cursors.
    ///
    /// The halo is set equal to the reach, which is what a planner that is not
    /// over-fetching does. A plan that grants a wider halo is expressible and is
    /// not expressible here: granting one is a decision about IO, so it belongs
    /// to whoever is making it.
    pub fn pixels(&mut self, chain: Chain) -> Result<Phase> {
        let slots = chain.slots();
        if slots.is_empty() {
            return Err(Error::InvalidArgument(
                "a pixel phase with no chain slots would read a level and write it back \
                 unchanged, at the price of a full materialisation. If that copy is wanted, \
                 say so with an identity op; if it is not, the phase is not a phase."
                    .to_string(),
            ));
        }
        let volume = self.grid.volume();
        let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
        let reach = chain.reach3(&volume);
        // Fold the element type over exactly the slots this phase owns, by the
        // same rule `Decomposition::declare_dtypes` uses, and declare it only
        // where it changed — so a plan whose chain never retypes fingerprints
        // exactly as it did before this builder existed.
        let mut produced = self.reads;
        for slot in &slots {
            produced = slot.produces(produced)?;
        }
        let declared = (produced != self.reads).then_some(produced);
        let n = slots.len();
        drop(slots);

        let mut phase = PhaseDecomposition::derive(
            (self.slots_used..self.slots_used + n).collect(),
            names,
            reach,
            reach,
            self.grid.clone(),
        );
        phase.dtype = declared;
        self.slots_used += n;
        self.fragments.push(chain);
        self.reads = produced;
        Ok(self.push(phase, Work::Pixels, true))
    }

    /// Append an `Iterate` phase.
    ///
    /// The reach is one substage's, which is `iterative_phase`'s whole point and
    /// is left to it. The op is taken by value because a `PhaseWork` borrows it
    /// and something has to own it until the run is over.
    pub fn iterate(&mut self, op: impl IterativeOp + 'static) -> Result<Phase> {
        let phase = iterative_phase(&op, self.grid.clone())?;
        // An iteration feeds its own output back in, so it hands the element
        // type on unchanged; `check_dtypes` asserts the op accepts what it is
        // handed rather than assuming it.
        Ok(self.push(phase, Work::Iterate(Box::new(op)), true))
    }

    /// Append a `Fragments` phase.
    ///
    /// **Responsibility four is here**, and it is the one the hand-built plan's
    /// author called the biggest: a fragment phase owns no chain slot, so
    /// `check_dtypes` has nothing to fold and asks the op instead — and the plan
    /// has to have allocated the level at the width the op says it writes.
    /// Stating that by hand is a line that is easy to omit and whose omission is
    /// refused by a message about a level's width rather than about the missing
    /// line. Here it is asked of the op, which is the only thing that knows.
    pub fn fragments(&mut self, op: impl FragmentOp + 'static) -> Result<Phase> {
        self.fragments_boxed(Box::new(op))
    }

    /// The same, for a caller that already holds a boxed op.
    ///
    /// The phase is derived here rather than taken as an argument, deliberately:
    /// a signature that accepted both a `PhaseDecomposition` and the op it was
    /// derived from would let the two disagree, which is the shape of mistake
    /// this whole module exists to remove.
    pub fn fragments_boxed(&mut self, op: Box<dyn FragmentOp>) -> Result<Phase> {
        let mut phase = fragment_phase(&*op, self.grid.clone())?;
        let writes = op.writes_pixels();
        if writes {
            let produced = op.produces(self.reads);
            phase.dtype = (produced != self.reads).then_some(produced);
            self.reads = produced;
        }
        Ok(self.push(phase, Work::Fragments(op), writes))
    }

    /// The one place a phase and its kind are appended, so that there is no
    /// order for them to disagree about.
    fn push(&mut self, phase: PhaseDecomposition, work: Work, writes_a_level: bool) -> Phase {
        let index = self.phases.len();
        self.phases.push(phase);
        self.work.push(work);
        Phase {
            index,
            writes: writes_a_level.then(|| Level(index + 1)),
        }
    }

    /// Close the plan: assemble the chain, derive what a chain can derive, and
    /// run every guard the executor would.
    ///
    /// **Responsibility six is `declare_source_levels`**, and it is here rather
    /// than offered as a step because forgetting it is not refused by
    /// `Decomposition::check` — it surfaces later as a level freed before its
    /// second reader, or as an executor refusing a dependency it was never told
    /// about. A construction step that can be forgotten and whose omission is
    /// diagnosed somewhere else is not a step, it is a trap.
    ///
    /// The five checks are the executor's own, called unchanged. Running them
    /// here does not make the executor's copies redundant — a plan may arrive
    /// from any strategy or off a wire — it means a plan written by hand fails
    /// at the line that wrote it.
    pub fn finish(self) -> Result<Assembly> {
        if self.phases.is_empty() {
            return Err(Error::InvalidArgument(
                "a plan with no phases: nothing was appended to this builder.".to_string(),
            ));
        }
        let chain = Chain::sequence(self.fragments);
        let chain_reach = chain.reach3(&self.volume);
        let workflow = Workflow::new(chain, self.volume, self.dtype);
        let mut decomposition = Decomposition {
            volume: self.volume,
            dtype: self.dtype,
            phases: self.phases,
            chain_reach,
        };
        decomposition.declare_source_levels(&workflow.chain)?;

        let assembly = Assembly {
            workflow,
            decomposition,
            work: self.work,
        };
        // The executor's own five, in the executor's own order, called
        // unchanged. Running them here does not make its copies redundant — a
        // plan may arrive from any strategy or off a wire — it means a plan
        // written by hand fails at the line that wrote it rather than at the
        // first run of it.
        {
            let work = assembly.work();
            assembly.decomposition.check()?;
            check_phase_work(&assembly.decomposition, &work)?;
            check_block_constraints(&assembly.workflow.chain, &assembly.decomposition)?;
            check_dtypes(&assembly.workflow.chain, &assembly.decomposition, &work)?;
            check_source_levels(&assembly.workflow.chain, &assembly.decomposition)?;
        }
        Ok(assembly)
    }
}

/// A finished plan and everything needed to run it.
///
/// The three are one object because they cannot be separated: `PhaseWork`
/// borrows the ops, so the ops must outlive the `execute_phases` call, and the
/// ops carry plan coordinates so an op built without knowing where it landed is
/// an op addressing the wrong generation of a stream. Handing back the
/// decomposition alone would be handing back half a plan.
pub struct Assembly {
    pub workflow: Workflow,
    pub decomposition: Decomposition,
    work: Vec<Work>,
}

impl Assembly {
    /// What each phase runs, borrowed from the ops this owns.
    ///
    /// Derived on demand rather than stored, which is what makes item four
    /// unrepresentable rather than checked: there is no second list to fall out
    /// of order with the phases, because the kinds *are* in the phase list.
    pub fn work(&self) -> Vec<PhaseWork<'_>> {
        self.work.iter().map(Work::borrow).collect()
    }

    /// The number of phases, from the one list that has them.
    pub fn n_phases(&self) -> usize {
        self.work.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probes::{AffineOp, IdentityOp};

    fn grid() -> BlockGrid {
        BlockGrid::new([16, 8, 8], [8, 8, 8]).expect("a lattice")
    }

    /// The builder's plan and the hand-built one, compared as plans.
    #[test]
    fn two_pixel_phases_assemble_to_the_hand_built_decomposition() {
        let volume = [16usize, 8, 8];
        let mut plan = PlanBuilder::new(volume, Dtype::F64, grid());
        plan.pixels(Chain::op(IdentityOp::new("first", [1, 0, 0])))
            .expect("a pixel phase");
        plan.pixels(Chain::op(AffineOp::new("second", 2.0, 1.0, [0, 2, 0])))
            .expect("a pixel phase");
        let built = plan.finish().map_err(|e| e.to_string()).expect("a plan");

        let first = Chain::op(IdentityOp::new("first", [1, 0, 0]));
        let second = Chain::op(AffineOp::new("second", 2.0, 1.0, [0, 2, 0]));
        let (n_first, first_reach) = (first.slots().len(), first.reach3(&volume));
        let (n_second, second_reach) = (second.slots().len(), second.reach3(&volume));
        let chain = Chain::sequence(vec![first, second]);
        let names: Vec<String> = chain
            .slots()
            .iter()
            .map(|slot| slot.display_name())
            .collect();
        let mut hand = Decomposition {
            volume,
            dtype: Dtype::F64,
            phases: vec![
                PhaseDecomposition::derive(
                    (0..n_first).collect(),
                    names[0..n_first].to_vec(),
                    first_reach,
                    first_reach,
                    grid(),
                ),
                PhaseDecomposition::derive(
                    (n_first..n_first + n_second).collect(),
                    names[n_first..n_first + n_second].to_vec(),
                    second_reach,
                    second_reach,
                    grid(),
                ),
            ],
            chain_reach: chain.reach3(&volume),
        };
        hand.declare_source_levels(&chain).expect("source levels");

        assert_eq!(built.decomposition, hand);
        assert_eq!(built.decomposition.fingerprint(), hand.fingerprint());
    }

    /// A phase handle knows which level its phase wrote, and the two numbers are
    /// different types so that they cannot be swapped.
    #[test]
    fn a_phase_handle_carries_the_level_it_wrote() {
        let mut plan = PlanBuilder::new([16, 8, 8], Dtype::F64, grid());
        let first = plan
            .pixels(Chain::op(IdentityOp::new("first", [1, 0, 0])))
            .expect("a pixel phase");
        assert_eq!(first.index(), 0);
        assert_eq!(first.level().expect("a level"), Level::from(1));
    }

    #[test]
    fn a_pixel_phase_with_no_slots_is_refused() {
        let mut plan = PlanBuilder::new([16, 8, 8], Dtype::F64, grid());
        let message = plan
            .pixels(Chain::sequence(Vec::new()))
            .expect_err("no slots")
            .to_string();
        assert!(message.contains("no chain slots"), "{message}");
    }

    #[test]
    fn a_plan_with_no_phases_is_refused() {
        let plan = PlanBuilder::new([16, 8, 8], Dtype::F64, grid());
        let message = plan.finish().err().expect("no phases").to_string();
        assert!(message.contains("no phases"), "{message}");
    }
}
