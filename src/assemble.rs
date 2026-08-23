// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Assembly, not planning.** Every number a `Decomposition` holds is
// parity-visible: change a slot range, a halo, a per-phase element type or which
// image an arm reads and the output changes. So a builder that *decided* any of
// them would be a planner wearing a convenience's name, and the two must not be
// the same object — `strategy::Strategy::decompose` is where a plan is chosen.
// What this module removes is the **bookkeeping** between a caller who already
// knows what the phases are and a `Decomposition` that records them, and nothing
// else. Everything it produces goes through `Decomposition::check`,
// `check_dtypes`, `check_source_images`, `check_block_constraints` and
// `check_phase_work` unchanged, and `PlanBuilder::finish` runs all five so that
// a plan that would be refused at execution is refused where it was written.
//
// Where it came from
// ------------------
// The first mixed-kind multi-phase plan built on this crate was hand-assembled,
// and its author kept a list of what that cost. Six of the eight entries were
// one thing each: **the slot cursor, the names, the per-phase reach, the
// fragment phase's element type, the `PhaseWork` list, and remembering to call
// `declare_source_images`.** Those six are this builder's whole remit. The list
// is worth reading as a design constraint rather than as history, because each
// entry is a place where writing the *wrong* number compiles, runs, and produces
// a different answer with no error at all:
//
// | what was hand-maintained | how it went wrong | what it is now |
// |---|---|---|
// | a phase index passed to an op's constructor | a literal that is off by one reads a *different generation* of a stream — a wrong answer, not an error | [`Phase`], which can only come from the builder that made the phase |
// | slot ranges and names, as two parallel lists with three cursors | a slice off by one names the wrong op in every log and event | derived here from the chain fragment itself |
// | a phase's reach, taken from a fragment before it was moved into the sequence | had to be taken *early*, which is the easiest ordering in the file to get wrong | taken here, in the one place that holds the fragment |
// | `phase.dtype = Some(Dtype::U32)` after `fragment_phase` | forgetting it is refused, but by a message about an image's width | asked of the op, which is the only thing that knows |
// | a second list of `PhaseWork`, tied to the phase list by a `debug_assert_eq!` on its length | a length check catches the count, never the order | **not a second list**: the builder records the kind *with* the phase |
// | `declare_source_images`, called by hand at the end | forgetting it does not fail `check()`; it fails as an image freed before its second reader | part of [`PlanBuilder::finish`] |
//
// What is a type here rather than a check, and why that is the point
// ------------------------------------------------------------------
// [`Phase`] and [`ImageId`] are both a `usize` and they are deliberately not
// interchangeable. A phase index and an image number are different quantities
// that are numerically close — phase `p` writes image `p + 1` — so the mistake
// worth making impossible is not "a number out of range" but "the *other*
// number, which is also in range". `Chain::source` takes an [`ImageId`] and an op
// that reads another phase's stream takes a [`Phase`], so passing one where the
// other belongs is a compile error rather than a plan that runs and answers
// differently.
//
// The `PhaseWork` list is the sharper case. It used to be two lists that a
// length assertion compared; here the kind is stored *in* the phase list, so
// there is no second list to disagree — `Assembly::work` derives the borrowed
// view on demand. That is the difference between a check and a shape.
//
// Asking a planner is not becoming one
// ------------------------------------
// [`PlanBuilder::pixels`] makes **one** phase of the chain it is handed, because
// a caller who knows the phases must be able to say so. That left the partition
// planner with no way in: `strategy::Strategy::decompose` takes a `Workflow` —
// one chain, one input, one output, a linear pipeline — and a stage that needs
// several images has none of that to offer, so it grouped its own phases by
// hand and the planner never got a vote. Every performance figure this crate
// has for such a stage is therefore measured on a partition nobody chose.
//
// [`PlanBuilder::partition`] is the way in, and it stays on this module's side
// of the line for one reason: **it decides nothing.** The caller names the
// strategy and the [`crate::decomposition::Constraints`] it may spend; the
// builder contributes the coordinate space the chain runs in and the slot
// cursor to renumber the answer by. What comes back is as many phases as the
// planner chose, which may be one. That is the same relationship this module
// already has with `iterative_phase` and `fragment_phase` — the shape of the
// phase is asked of the thing that knows it, and the bookkeeping around it is
// what is removed here.
//
// What it does deliberately do
// ----------------------------
// * **Synthesise the `Workflow` per call**, over the sub-chain, the current
//   image's volume and the element type the next phase reads. `decompose` reads
//   exactly those three fields, so a chain in the middle of an image DAG can be
//   priced without the planner learning about the DAG.
// * **Renumber the slots.** The planner numbers a chain from zero; a plan under
//   assembly is at whatever the cursor says.
// * **Leave the source leaves alone.** [`crate::op::Chain::Source`] names an
//   absolute image of the whole plan, and cutting a chain into phases never
//   changes one — see [`PlanBuilder::partition`] for the argument.
//
// What it deliberately does not absorb
// ------------------------------------
// * **The block lattice.** A grid is a planning decision with a cost model
//   behind it, so the caller supplies it; [`PlanBuilder::regrid`] is how a plan
//   that changes lattice mid-way says so, and it is a statement rather than an
//   inference. A planned run of phases is cut on the lattices the *planner*
//   chose, per phase, and the builder's own stays what the caller last said —
//   [`Partition::grid`] hands back the last of them for a caller who wants to
//   carry on where the planner left off, which is again a statement rather than
//   an inference.
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
    check_block_constraints, check_dtypes, check_source_images, cuttable_axes, phase_traffic,
    price_phase, Constraints, Decomposition, PhaseCost, PhaseDecomposition,
};
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::fragment::{check_phase_work, fragment_phase, FragmentOp, PhaseWork};
use crate::geometry::BlockGrid;
use crate::iterate::{iterative_phase, substage_reach, IterativeOp};
use crate::op::Chain;
use crate::reach::Reach;
use crate::strategy::{phase_makespan, CandidateTally, Strategy, Workflow};

/// An image of the plan: image 0 is the input, image `p + 1` is what phase `p`
/// wrote, and an address at or above [`ImageId::SUPPLIED_BASE`] is one of the
/// arrays the caller handed the run.
///
/// A newtype over the index and not an alias, because the whole reason it exists
/// is to be a *different type* from [`Phase`]. `From<usize>` is implemented so
/// that every caller who already writes a literal keeps working; what it buys is
/// that a caller holding a phase handle cannot pass it where an image belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageId(usize);

impl ImageId {
    /// The first address in the **supplied-input** range.
    ///
    /// An image below this is written by the run: image 0 by whoever seeded it,
    /// image `p + 1` by phase `p`. An image at or above it is an array the caller
    /// handed the run, which no phase produces and nothing can recompute.
    ///
    /// **Why a disjoint high range rather than `0..k` at the bottom.** A caller
    /// has to know a supplied input's address *before* it builds the ops that
    /// read it — `Chain::source` takes the number, and a `BlockOp` that reads a
    /// second array stores the number inside itself — and a plan does not know
    /// how many phases it has until it is finished, because
    /// [`PlanBuilder::partition`] lets a strategy choose. Numbering the inputs
    /// `0..k` and the phase outputs `k + p` would fix that, at the price of
    /// renumbering **every phase-written image** the moment a second input is
    /// added: an op holding `Chain::source(3)` because it meant "phase 2's
    /// output" would silently mean something else. This range renumbers nothing,
    /// and it is stable under a partition that adds phases.
    ///
    /// It is the high bit of a `usize`, so the test is one instruction and no
    /// plan can reach it by counting phases.
    pub const SUPPLIED_BASE: usize = usize::MAX / 2 + 1;

    /// The address of the `which`th array handed to the run, counted from zero.
    ///
    /// Available before a single phase has been appended, which is the whole
    /// point: the ops that read it are built first.
    pub fn supplied(which: usize) -> Self {
        assert!(
            which < Self::SUPPLIED_BASE,
            "supplied input {which} is outside the address range"
        );
        ImageId(Self::SUPPLIED_BASE + which)
    }

    /// The image number, for the places that index by it.
    pub fn index(self) -> usize {
        self.0
    }

    /// Whether this address names an array handed to the run.
    pub fn is_supplied(self) -> bool {
        is_supplied_image(self.0)
    }

    /// Which supplied array this is, or `None` for an image the run writes.
    pub fn supplied_index(self) -> Option<usize> {
        self.is_supplied().then(|| self.0 - Self::SUPPLIED_BASE)
    }
}

/// Whether `image` addresses an array handed to the run rather than one it
/// writes.
///
/// A free function as well as an [`ImageId`] method because the *recorded* half
/// of the plan still carries a bare `usize` — `PhaseDecomposition::source_images`
/// and `supplied_dtypes` are serialised to the distributed wire and hashed into
/// `Decomposition::fingerprint`, so their element type is part of a format — and
/// the question has to be askable there without a round trip through the
/// newtype. The *declared* half (`Chain::Source`, `SourceInput::image`) is an
/// [`ImageId`] and asks it as [`ImageId::is_supplied`].
pub fn is_supplied_image(image: usize) -> bool {
    image >= ImageId::SUPPLIED_BASE
}

/// How to name `image` in a message: an index for an image the run writes, and
/// "supplied input `i`" for one it was handed.
///
/// Without this every diagnostic about a supplied input prints a
/// nineteen-digit number, which is an address nobody typed and nobody can read
/// back.
pub fn describe_image(image: usize) -> String {
    match is_supplied_image(image) {
        true => format!("supplied input {}", image - ImageId::SUPPLIED_BASE),
        false => format!("image {image}"),
    }
}

impl From<usize> for ImageId {
    fn from(image: usize) -> Self {
        ImageId(image)
    }
}

impl From<ImageId> for usize {
    fn from(image: ImageId) -> Self {
        image.0
    }
}

impl std::fmt::Display for ImageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.supplied_index() {
            Some(which) => write!(f, "supplied input {which}"),
            None => write!(f, "{}", self.0),
        }
    }
}

/// A phase of the plan under assembly: where it landed, and the image it writes.
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
    /// terminal as far as images go, and is a fact about the op rather than
    /// about the builder.
    writes: Option<ImageId>,
}

impl Phase {
    /// Where this phase sits in the plan.
    pub fn index(self) -> usize {
        self.index
    }

    /// The image this phase writes, if it writes one.
    pub fn writes(self) -> Option<ImageId> {
        self.writes
    }

    /// The image this phase writes, or the refusal that names why there is none.
    ///
    /// The fallible form exists because whether a fragment phase writes pixels
    /// is the op's answer, not something a signature can promise. For a pixel or
    /// an iterative phase it cannot fail, and the `?` costs nothing.
    pub fn image(self) -> Result<ImageId> {
        self.writes.ok_or_else(|| {
            Error::InvalidArgument(format!(
                "phase {} writes fragments and no pixels, so there is no image {} for anything \
                 to read. A phase that hands pixels on says so with `FragmentOp::writes_pixels`.",
                self.index,
                self.index + 1
            ))
        })
    }
}

/// How many substages an iterative phase is expected to run, where anybody knows.
///
/// **Not [`crate::iterate::SubstageLimit`]**, and the distinction is the one that
/// type exists to make: the limit is a runaway guard, deliberately generous, and
/// pricing against it would price the backstop rather than the work. This is a
/// *count*, and the only honest source for one is a run that already happened —
/// `Stats::substages` reports it per phase, which is why this can be asked for
/// without inventing a number.
///
/// # Why it is asked for at all
///
/// `IterativeOp::cost_per_voxel` is per substage and the planner prices one, on
/// the argument that the count is a positive constant common to every candidate
/// and so cannot move an argmin. **The first half of that is measured and true;
/// the second half is false.** The count really is independent of the lattice —
/// `tests/iterative_block_choice.rs` runs the executor over nineteen grids
/// including one block per voxel, where every propagation step crosses a seam,
/// and the count never moves — but it multiplies only *part* of the price. A
/// phase runs `S` substages of read-and-compute and writes its image **once**,
/// so the true cost is `S * (read + compute) + write`, and ranking on `S == 1`
/// weighs the write `S` times too heavily. The error is
/// `(S - 1) * (read + compute)`, which is a function of the block edge through
/// the read amplification — the same family of mistake as the two before it.
///
/// Swept, the chosen edge departs from the one-substage choice at counts as
/// ordinary as `2` and `4` rather than only at extreme ones, in a small minority
/// of configurations, and never below three workers. The regret — the price of
/// the one-substage choice under the objective the phase really has — reaches
/// `1.125x` over the sweep `tests/iterative_block_choice.rs` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Substages {
    /// Nobody has run it. One substage is priced, which is what the planner has
    /// always done and is byte-identical to it.
    Unknown,
    /// A count from a previous run's `Stats::substages` for this phase.
    Measured(usize),
}

impl Substages {
    /// The multiplier the repeated half of the price is charged at.
    ///
    /// Fallible on zero for [`crate::iterate::SubstageLimit::of`]'s reason: a
    /// phase that ran no substages wrote nothing, so a zero here is a
    /// transcription error rather than a measurement, and silently treating it
    /// as one would price the phase at its write alone.
    fn factor(self, op: &str) -> Result<f64> {
        match self {
            Substages::Unknown => Ok(1.0),
            Substages::Measured(0) => Err(Error::InvalidArgument(format!(
                "iterative op {op:?} was priced against a measured substage count of zero. An \
                 iteration that ran no substages wrote no image, so zero is a mis-transcribed \
                 measurement rather than one; `Substages::Unknown` is how a caller says it does \
                 not have the number."
            ))),
            Substages::Measured(count) => Ok(count as f64),
        }
    }
}

/// What becomes of the image a phase writes, which is what its write is charged
/// at.
///
/// **A caller's statement, not an inference**, and the reason is measured
/// rather than argued: see [`PlanBuilder::iterate_priced`], which found the
/// choice of block edge sensitive to it. `predicted_makespan` reads the same
/// fact off a finished plan as `index + 1 < n_phases`; a builder pricing a
/// phase it is in the middle of appending has no finished plan to read it off,
/// and the two weights are equal under
/// [`CostModel::default`](crate::decomposition::CostModel::default), so an
/// assumption here would have been invisible in every default-model test and
/// wrong for exactly the caller who calibrated a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Materialisation {
    /// Another phase will read it: charged `materialise_cost_per_voxel`.
    ///
    /// The case to reach for when in doubt. The default model's note gives the
    /// reason — it "assumes poor compression, which biases towards fusing — the
    /// cheaper mistake" — and a phase appended mid-plan is more often followed
    /// than not.
    Intermediate,
    /// It is the plan's output: charged `write_cost_per_voxel`.
    Output,
}

/// Whether one candidate's working set fits the budget, at the concurrency the
/// caller expects to run at.
///
/// Lifted out of the sweep rather than inlined because it is the partition
/// search's own rule, word for word (`PhasePricer::affordable`), and a second
/// copy that drifted would let this builder accept a lattice the search would
/// have refused. `working_set_bytes_per_block` is computed from the *clamped*
/// read extent for exactly this reason: a budget checked against an
/// over-charged read invents infeasibility.
fn affordable(cost: &PhaseCost, constraints: &Constraints) -> bool {
    constraints.budget_bytes.is_none_or(|budget| {
        cost.working_set_bytes_per_block * constraints.expected_concurrency.max(1) as f64
            <= budget as f64
    })
}

/// The phase one [`PlanBuilder::iterate_priced`] call made, the lattice it was
/// priced onto, and what the sweep looked at on the way.
///
/// **The tally is not decoration.** A sweep that silently dropped every
/// candidate but one reads exactly like a sweep that considered them all and
/// preferred that one, and the difference decides whether a caller should widen
/// its budget or its candidate list. It is the same [`CandidateTally`] the
/// partition search folds into its own account, so the two read the same way.
#[derive(Debug, Clone)]
pub struct PricedPhase {
    phase: Phase,
    grid: BlockGrid,
    makespan: f64,
    ranked: f64,
    tally: CandidateTally,
}

impl PricedPhase {
    /// The phase, for addressing what it wrote.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// The image the phase wrote.
    pub fn image(&self) -> ImageId {
        self.phase
            .writes()
            .expect("an iterative phase writes an image")
    }

    /// The lattice the sweep settled on.
    ///
    /// Offered rather than applied, on [`Partition::grid`]'s argument: whether
    /// the phases appended *next* should sit on it is a second planning
    /// decision and belongs to whoever is making it. A caller who wants it says
    /// [`PlanBuilder::regrid`] with this.
    pub fn grid(&self) -> &BlockGrid {
        &self.grid
    }

    /// What the winning candidate was predicted to take, **for one substage**.
    ///
    /// One substage because that is what can be priced without data; see
    /// [`PlanBuilder::iterate_priced`] for why the ranking is nevertheless the
    /// whole phase's. In the model's units — a bare number under
    /// [`crate::decomposition::CostModel::default`], nanoseconds under one
    /// calibrated from a [`crate::statistics::Snapshot`].
    pub fn makespan(&self) -> f64 {
        self.makespan
    }

    /// What the sweep actually minimised.
    ///
    /// Equal to [`Self::makespan`] under [`Substages::Unknown`], and `S` times
    /// the repeated half of it plus one write under a measured count. It is
    /// reported rather than kept private because a caller comparing two
    /// candidate plans has to know which of the two numbers it is comparing —
    /// and because a ranking done on a quantity nobody can read back is the
    /// thing [`crate::strategy::predicted_makespan`] exists to prevent.
    pub fn ranked_makespan(&self) -> f64 {
        self.ranked
    }

    /// What the sweep was offered and what it dropped.
    pub fn tally(&self) -> CandidateTally {
        self.tally
    }
}

/// The phases one [`PlanBuilder::partition`] call made, in plan order.
///
/// **A run of phases is not a phase, and the difference is worth a type.** A
/// caller carries on building from the *last* image a call produced, so that is
/// what [`Self::image`] and [`Self::last`] answer; but "how many did the planner
/// make" is the question the whole entry exists to be able to ask — a bridge
/// that silently always made one would pass every other test — so the count is
/// here rather than left to be inferred by subtracting two `n_phases()` readings
/// around the call.
///
/// Never empty: a partition of a chain with at least one slot has at least one
/// group, and a chain with no slots is refused before any of this.
#[derive(Debug, Clone)]
pub struct Partition {
    phases: Vec<Phase>,
    grid: BlockGrid,
}

impl Partition {
    /// The last phase, which is the one a caller keeps building from.
    pub fn last(&self) -> Phase {
        *self
            .phases
            .last()
            .expect("a partition holds at least one phase")
    }

    /// The image the last phase wrote.
    ///
    /// Infallible where [`Phase::image`] is not, and for a reason rather than
    /// for convenience: every phase a partition holds is a `Pixels` phase over a
    /// run of chain slots, and such a phase always writes the image below it
    /// plus one. There is no fragment op here to answer otherwise.
    pub fn image(&self) -> ImageId {
        self.last()
            .writes()
            .expect("a pixel phase writes the image above it")
    }

    /// How many phases the planner made of the chain.
    ///
    /// One means it declined to cut, which is an answer and not a failure.
    pub fn n_phases(&self) -> usize {
        self.phases.len()
    }

    /// Every phase it made, in plan order.
    pub fn phases(&self) -> &[Phase] {
        &self.phases
    }

    /// The lattice the last phase was cut on.
    ///
    /// Offered because the planner chose it and the builder did not: a caller
    /// who wants the phases it appends next to sit on the lattice the planner
    /// settled on says [`PlanBuilder::regrid`] with this, and a caller who wants
    /// its own lattice back does nothing. Neither is inferred.
    pub fn grid(&self) -> &BlockGrid {
        &self.grid
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
    /// The element type of the image the *next* phase will read. Folded here for
    /// the same reason `Decomposition::declare_dtypes` folds it: a fragment
    /// phase owns no slot, so nothing downstream can recover what it was handed.
    reads: Dtype,
}

impl PlanBuilder {
    /// A plan over `volume`, whose image 0 holds `dtype`, cut on `grid`.
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
                "a pixel phase with no chain slots would read an image and write it back \
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

    /// Append **as many `Pixels` phases as `strategy` makes** of one chain.
    ///
    /// The counterpart of [`Self::pixels`], and the two are the two answers to
    /// one question. `pixels` is for a caller who knows the phase; this is for a
    /// caller who knows the *chain* and wants the partition priced. Both stay,
    /// because "I want exactly this phase" and "choose for me" are different
    /// statements and a builder that could only make one of them would be
    /// deciding on the caller's behalf either way.
    ///
    /// # How a `Workflow`-shaped planner is reached from an image DAG
    ///
    /// `Strategy::decompose` takes a [`Workflow`], which is one chain over one
    /// volume in one element type. It reads exactly those three fields, so the
    /// workflow is **synthesised here, per call**, from the chain, from
    /// `self.grid().volume()` — the space the image this chain reads lives in,
    /// which is not `Decomposition::volume` once a phase has changed shape — and
    /// from [`Self::reads`]. The `input` and `output` names it carries are the
    /// defaults and mean nothing mid-plan; nothing in a partition search looks at
    /// them. No narrower entry point into the search is needed for that.
    ///
    /// One term of the price *is* affected by the synthesis and cannot be fixed
    /// from here: a group is charged `materialise_cost_per_voxel` when it is not
    /// the last of the chain, and the last group of a mid-plan chain writes an
    /// intermediate that this workflow has no way to mention. It is priced as an
    /// output, so the final cut is very slightly under-charged. Both weights
    /// default to `1.0`, where the distinction disappears.
    ///
    /// # What happens to the image a source leaf names
    ///
    /// Nothing, and that is the property to check rather than to assume.
    /// [`Chain::source`] names an image of the *whole plan*; the planner never
    /// sees an image number, only slots, so it cannot rewrite one. What cutting
    /// changes is which phase a leaf ends up in: with one phase every slot sits
    /// at phase `p` and reads image `p`, and with `k` phases a slot may sit as
    /// late as `p + k - 1`. The rule
    /// [`crate::decomposition::check_source_images`] enforces is that a leaf's
    /// image is at most its phase's own — so a leaf that was legal fused is
    /// legal cut, because cutting only ever moves it later. The reverse is what
    /// would be unsound, and the reverse is not a thing this can do.
    ///
    /// The one direction in which a cut can genuinely refuse is an op declaring
    /// a *reach* over the image it sources: the granted halo is the phase's, and
    /// a phase holding fewer slots may grant less than the fused one did. That
    /// is refused by name at [`Self::finish`], with the image and both reaches
    /// in the message, rather than run.
    ///
    /// # Where the numbers come from
    ///
    /// `constraints` is the caller's, whole. The block candidates in particular
    /// are **not** taken from `self.grid()`: a grid is one lattice a caller
    /// stated, and a candidate list is a set of scalar edges a cost model is to
    /// choose between — the builder's lattice is not a candidate list with one
    /// entry, it is a different kind of thing. What the builder contributes is
    /// the volume, which is not a choice at all.
    pub fn partition(
        &mut self,
        chain: Chain,
        strategy: &dyn Strategy,
        constraints: &Constraints,
    ) -> Result<Partition> {
        let slots = chain.slots();
        if slots.is_empty() {
            return Err(Error::InvalidArgument(
                "a planned run of pixel phases with no chain slots: there is nothing to \
                 partition. A phase that reads an image and writes it back unchanged is an \
                 identity op, said out loud."
                    .to_string(),
            ));
        }
        let n = slots.len();
        // The element type, folded over the whole chain by exactly the rule
        // `pixels` folds it by. The planner declares it per phase from the same
        // fold over the same slots, started at the same type, so the two agree
        // phase by phase and this is only the value the *next* phase reads.
        let mut produced = self.reads;
        for slot in &slots {
            produced = slot.produces(produced)?;
        }
        drop(slots);

        let workflow = Workflow::new(chain, self.grid.volume(), self.reads);
        let planned = strategy.decompose(&workflow, constraints)?;
        // A decomposition may partition; it may never reorder or drop an op —
        // the executor's own words, checked here because a strategy is an open
        // trait and a plan that renumbered the chain would be renumbered again
        // by the cursor below into something nobody could read back.
        let order = planned.slot_order();
        if order != (0..n).collect::<Vec<_>>() {
            return Err(Error::InvalidArgument(format!(
                "strategy {:?} partitioned a {n}-slot chain into slot order {order:?}, which is \
                 not the chain's own order 0..{n}. A decomposition may partition; it may never \
                 reorder or drop an op.",
                strategy.name()
            )));
        }
        let base = self.slots_used;
        let mut phases = Vec::with_capacity(planned.phases.len());
        let mut grid = self.grid.clone();
        for mut phase in planned.phases {
            // The one renumbering. The planner counted this chain's slots from
            // zero; the plan under assembly is at the cursor. Everything else it
            // recorded — the reach, the halo, the grid, the element type it
            // declares — is a fact about the run of slots and travels unchanged,
            // and `source_images` is absolute already and is re-derived over the
            // whole chain by `finish` in any case.
            for slot in &mut phase.slots {
                *slot += base;
            }
            grid = phase.grid.clone();
            phases.push(self.push(phase, Work::Pixels, true));
        }
        self.slots_used += n;
        self.fragments.push(workflow.chain);
        self.reads = produced;
        Ok(Partition { phases, grid })
    }

    /// Append an `Iterate` phase **on the lattice the builder is holding**.
    ///
    /// The reach is one substage's, which is `iterative_phase`'s whole point and
    /// is left to it. The op is taken by value because a `PhaseWork` borrows it
    /// and something has to own it until the run is over.
    ///
    /// The grid is the previous phase's, unpriced, and that is a statement and
    /// not an oversight — the same statement [`Self::new`] makes about the
    /// caller's opening lattice. What was an oversight was that it was the
    /// *only* thing on offer: a caller who wanted the edge chosen rather than
    /// inherited had nothing to call. [`Self::iterate_priced`] is that, and the
    /// two stand in the same relation as [`Self::pixels`] and
    /// [`Self::partition`] — "I want exactly this phase" against "choose for
    /// me", both kept because they are different statements.
    pub fn iterate(&mut self, op: impl IterativeOp + 'static) -> Result<Phase> {
        let phase = iterative_phase(&op, self.grid.clone())?;
        // An iteration feeds its own output back in, so it hands the element
        // type on unchanged; `check_dtypes` asserts the op accepts what it is
        // handed rather than assuming it.
        Ok(self.push(phase, Work::Iterate(Box::new(op)), true))
    }

    /// Append an `Iterate` phase **on the cheapest lattice `constraints` offers**.
    ///
    /// The counterpart of [`Self::iterate`], and the reason it exists is that
    /// `iterate` inherits `self.grid()` with no pricing at all. Every other
    /// phase kind in this builder either has its edge chosen by a search
    /// ([`Self::partition`]) or has it stated by a caller who is choosing
    /// ([`Self::pixels`]); an iterative phase had neither, and inheriting the
    /// grid of the phase before it is not a choice, it is the absence of one.
    ///
    /// # What the sweep is
    ///
    /// Each edge in `constraints.block_candidates` is turned into a grid by the
    /// same two steps the partition search uses — [`cuttable_axes`] takes the
    /// reach-derived floor off the axes, [`BlockGrid::along`] builds the
    /// lattice — the phase is derived on it, priced by [`price_phase`], dropped
    /// if the working set exceeds `budget_bytes`, and scored by
    /// [`phase_makespan`]. The winner is the lowest makespan, ties broken
    /// towards the larger edge, which is `PhasePricer`'s rule and is here so a
    /// caller reading two plans side by side is not comparing two tie-breaks.
    ///
    /// **The number this reports is the number the finished plan reports.** It
    /// is built from the same calls in the same order as
    /// [`crate::strategy::predicted_makespan`]'s per-phase term, so a caller can
    /// price the whole plan afterwards and find this phase's contribution
    /// unchanged. `tests/iterative_block_choice.rs` asserts that equality rather
    /// than trusting it — a sweep minimising a quantity the plan does not report
    /// would be choosing on a number nobody can check — and asserts it for both
    /// values of `materialisation`, which is the argument that makes it true.
    ///
    /// # What moves the answer, measured rather than assumed
    ///
    /// The objective is a roofline, `max(pool, channel)` — see
    /// [`phase_makespan`] — and **which side binds is the whole of it.** Every
    /// statement below is about that.
    ///
    /// * **`workers` decides whether there is a choice at all.** At
    ///   `workers == 1` the pool term is the channel term plus the compute and
    ///   the conflict, so the pool always binds, the objective is the phase's
    ///   serial work, and that is monotone in the edge: the sweep answers "the
    ///   largest candidate that fits" for every op, every reach and every
    ///   compute figure. That is not a defect — it is
    ///   [`crate::strategy::Enumerating`]'s own account of what `concurrency ==
    ///   1` means — but a caller who leaves `workers` at one has bought a sweep
    ///   that cannot move and would do as well with [`Self::iterate`] and the
    ///   coarsest grid.
    /// * **The declared compute moves the argmin, and moves it a long way.**
    ///   This corrects a claim carried in from the pricing work: that scaling
    ///   `IterativeOp::cost_per_voxel` leaves the choice fixed because compute
    ///   is charged over the same extent as the read. That holds only where the
    ///   pool binds *for every candidate*, which is `workers == 1`. Above it,
    ///   compute appears in the pool and not in the channel, so raising it
    ///   walks the phase from bandwidth-bound to compute-bound and the argmin
    ///   walks with it — over `1e-3 .. 1e3` on this file's probe, from the
    ///   coarsest candidate to the finest. Measured at every `workers > 1`
    ///   swept and at no `workers == 1`.
    /// * **The halo**, through the read amplification `(edge + lo + hi) / edge`,
    ///   which is the term that grows without bound as the edge falls and is
    ///   the whole reason a fine grid is ever refused. It moves the *price* at
    ///   every candidate; whether it moves the *argmin* depends on which side
    ///   binds, which is why it is not on its own a lever a caller can reason
    ///   about.
    ///
    /// `workers` is an argument because the builder has no other source for it:
    /// it is [`crate::strategy::Enumerating::concurrency`], which reaches the
    /// executor through [`crate::strategy::Strategy::hints`] and is not
    /// recoverable from a `Constraints`.
    ///
    /// # Why `materialisation` is asked for rather than assumed
    ///
    /// A phase's write is charged at `write_cost_per_voxel` if it is the plan's
    /// output and at `materialise_cost_per_voxel` if another phase will read it,
    /// and `predicted_makespan` decides that by position — `index + 1 <
    /// n_phases`. A builder cannot: when this is called the phase *is* the last
    /// one, and whether it stays last is a fact about calls the caller has not
    /// made yet.
    ///
    /// Assuming it was the first version of this method, on the argument that
    /// the two weights enter the pool as `core * write * ceil(n / workers)` and
    /// the channel as `mean_core * n * write`, the second of which is the volume
    /// exactly at every candidate — so the write would shift every candidate by
    /// the same amount and could not move a ranking. **The argument is wrong and
    /// the sweep says so.** It is right in the channel bound and false in the
    /// pool bound, where `ceil(n / workers) / n` is a function of the block
    /// count: while `n < workers` it is `1 / n`, so a one-block candidate carries
    /// the whole write and a forty-block one carries a fortieth of it. Swept over
    /// `1e-6 .. 1e6` against reach, compute, split axes and `workers`, the chosen
    /// edge moves in about a quarter of the configurations — at **every**
    /// `workers > 1` tried and at **no** `workers == 1`, which is the same
    /// dividing line every other lever in this method falls on.
    /// `tests/iterative_block_choice.rs` holds the sweep.
    ///
    /// This was the third time this file had priced something with an error term
    /// that was itself a function of the candidate — `price_phase`'s core charge
    /// and its reach charge were the first two — so the rule is worth writing
    /// down: an approximation is admissible in a *price* only when it is
    /// constant across the things being ranked, and that is a measurement and
    /// never an argument. `substages` below is the fourth, found by applying it.
    ///
    /// # Why `substages` is asked for rather than left out
    ///
    /// The first version of this method excluded the substage count on a
    /// two-part argument: `IterativeOp::cost_per_voxel` is per substage, and the
    /// count is a positive constant common to every candidate, so it cannot move
    /// an argmin.
    ///
    /// **The first part survives a hard sweep and the second does not.** The
    /// count really is a property of the data and the op and not of the lattice:
    /// the executor was run over nineteen grids — including one block per voxel,
    /// where every step of the propagation crosses a seam, and ragged grids that
    /// divide no axis evenly — across five substage reaches and two data shapes,
    /// one of them a one-voxel-wide serpentine that forces a long geodesic
    /// through the volume, and the count is the whole-volume count every time.
    /// That is the halo doing its job: it is one substage's reach wide and holds
    /// the neighbours' cores from the previous substage, so information crosses
    /// a seam at exactly the rate it crosses the inside of a block.
    ///
    /// But a constant multiplier is only neutral if it multiplies the *whole*
    /// price, and this one does not. A phase runs `S` substages of read and
    /// compute and writes its image **once**, at the fixed point — the substages
    /// ping-pong two private buffers and touch no image — so the true shape is
    /// `S * (read + compute) + write`, and ranking at `S == 1` weighs the write
    /// `S` times too heavily relative to the rest. The residual is
    /// `(S - 1) * (read + compute)`, a function of the block edge through the
    /// read amplification. Swept, the choice departs from the one-substage
    /// choice at counts as ordinary as `2` and `4` — not only at extreme ones —
    /// in a small minority of configurations, never below three workers, and the
    /// one-substage choice costs up to `1.125x` the chosen one under the
    /// objective the phase really has.
    ///
    /// A uniform multiplier on the *whole* price really would be neutral, and
    /// `tests/iterative_block_choice.rs` asserts that too — beside the
    /// repeated-half version that is not — because a correction that scaled
    /// everything by the count would look like a correction and do nothing.
    ///
    /// So the count is asked for where a caller has it, and
    /// [`Substages::Unknown`] is byte-identical to not asking. It is a
    /// *measurement*, from `Stats::substages` of a previous run, and explicitly
    /// not `IterativeOp::limit()`, which is a runaway guard and would price the
    /// backstop.
    ///
    /// **The ranking and the reported price are then two different numbers**,
    /// deliberately: [`PricedPhase::makespan`] stays the one-substage figure the
    /// finished plan reports, because `predicted_makespan` prices one substage
    /// and nothing downstream of the phase knows the count, and
    /// [`PricedPhase::ranked_makespan`] is what the sweep minimised. Under
    /// `Substages::Unknown` they are the same number.
    ///
    /// The *shape* of that correction — which terms of a phase repeat and which
    /// happen once — is a fact about `strategy::run_iterative_phase` rather than
    /// about this builder, and would be better stated beside `price_phase` where
    /// `predicted_makespan` could use it too. It is here because this is the
    /// only door an iterative phase's block edge can come through: a phase with
    /// no chain slot is not a member of the partition search.
    pub fn iterate_priced(
        &mut self,
        op: impl IterativeOp + 'static,
        constraints: &Constraints,
        workers: usize,
        materialisation: Materialisation,
        substages: Substages,
    ) -> Result<PricedPhase> {
        if constraints.block_candidates.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "iterate_priced: iterative op {:?} was asked to be priced against an empty \
                 `block_candidates`, so there is nothing to choose between. A caller who wants \
                 the lattice this builder is already holding says `iterate` and says it out loud.",
                op.name()
            )));
        }
        let volume = self.grid.volume();
        let reach: Reach = substage_reach(&op).into();
        let bytes = self.reads.size_of() as f64;
        // The phase's own index, which is where it will land: `push` appends.
        let index = self.phases.len();
        // The caller's, because the builder cannot know it and the sweep is not
        // indifferent to it. See the doc above for the measurement that settles
        // which of those two facts is the binding one.
        let is_materialised = matches!(materialisation, Materialisation::Intermediate);
        // The repeated half of the price, charged as many times as the phase
        // will repeat it. The write is charged once whatever this is: an
        // iterative phase writes its image at the fixed point and never at a
        // substage — `run_iterative_phase` ping-pongs two private buffers and
        // writes the image once at the end — so the write is not part of what
        // repeats. See `Substages` for the measurement that made this worth
        // having.
        let repeats = substages.factor(op.name())?;
        let ranking_model = Constraints {
            model: crate::decomposition::CostModel {
                read_cost_per_voxel: constraints.model.read_cost_per_voxel * repeats,
                ..constraints.model
            },
            ..constraints.clone()
        };
        let ranking_model = &ranking_model.model;

        let mut tally = CandidateTally::default();
        let mut chosen: Option<(f64, usize, PhaseDecomposition)> = None;
        for &edge in &constraints.block_candidates {
            tally.offered += 1;
            // Per candidate, not hoisted: an axis is cut only where the cut
            // narrows what a block reads, and that depends on the edge.
            let axes = cuttable_axes(&constraints.split_axes, &reach, volume, edge);
            let Ok(grid) = BlockGrid::along(volume, &axes, edge) else {
                tally.no_grid += 1;
                continue;
            };
            // Derived rather than assembled by hand, so that the thing priced is
            // the thing appended. `iterative_phase` also runs `check_iterative`,
            // which is why an op that could never be a phase is refused here at
            // the first candidate rather than after a sweep.
            let phase = iterative_phase(&op, grid)?;
            let work = PhaseWork::Iterate(&op);
            let traffic = phase_traffic(index, &phase, Some(&work))?;
            let cost = price_phase(
                &phase.grid,
                // The halo, not the reach. They are equal for an iterative
                // phase — `iterative_phase` sets both to one substage's — but
                // the argument that picks between them is `price_phase`'s and
                // is not this file's to re-decide.
                &phase.halo,
                op.cost_per_voxel() * repeats,
                // A slotless phase has no `preferred_iteration` to conflict
                // with, and this is the number `predicted_makespan` passes for
                // it. Anything else here and the sweep would minimise a
                // quantity the finished plan does not report.
                0,
                is_materialised,
                bytes,
                ranking_model,
                ranking_model.materialise_cost_per_voxel,
                traffic,
            );
            if !affordable(&cost, constraints) {
                tally.over_budget += 1;
                continue;
            }
            tally.priced += 1;
            // The per-voxel write charge, derived from `traffic` rather than
            // from `materialisation` alone, which is `predicted_makespan`'s own
            // line: the channel bound has to count the bytes `price_phase`
            // charged for, and a phase that writes no image was charged for
            // none. An iterative phase always writes one — so this is a branch
            // that never takes its first arm today — and it is written this way
            // so that it goes on agreeing with the plan's own price if that
            // ever stops being true.
            let write_cost = if !traffic.writes_an_image {
                0.0
            } else if is_materialised {
                constraints.model.materialise_cost_per_voxel
            } else {
                constraints.model.write_cost_per_voxel
            };
            let makespan = phase_makespan(&cost, &phase.grid, workers, ranking_model, write_cost);
            let better = match &chosen {
                None => true,
                Some((best, best_edge, _)) => {
                    (makespan, std::cmp::Reverse(edge)) < (*best, std::cmp::Reverse(*best_edge))
                }
            };
            if better {
                chosen = Some((makespan, edge, phase));
            }
        }

        let Some((ranked, _, phase)) = chosen else {
            return Err(Error::InvalidArgument(format!(
                "iterate_priced: iterative op {:?} over volume {volume:?} has no affordable \
                 lattice. Of the {} candidate edge(s) {:?}, {} produced no grid at all once the \
                 reach-derived floor had taken the axes off {:?}, and {} exceeded the byte \
                 budget {:?} at a concurrency of {}. A wider budget, a coarser candidate or a \
                 narrower substage reach than {reach} is what changes that.",
                op.name(),
                tally.offered,
                constraints.block_candidates,
                tally.no_grid,
                constraints.split_axes,
                tally.over_budget,
                constraints.budget_bytes,
                constraints.expected_concurrency.max(1),
            )));
        };
        // The winner, re-priced at **one** substage under the caller's own model,
        // because that is the number the finished plan reports:
        // `predicted_makespan` prices one substage and cannot do otherwise —
        // nothing downstream of the phase knows the count. So the sweep ranks on
        // the phase's real shape and reports the plan's own figure, and the two
        // are separate fields rather than one number that is neither.
        let traffic = phase_traffic(index, &phase, Some(&PhaseWork::Iterate(&op)))?;
        let plan_cost = price_phase(
            &phase.grid,
            &phase.halo,
            op.cost_per_voxel(),
            0,
            is_materialised,
            bytes,
            &constraints.model,
            constraints.model.materialise_cost_per_voxel,
            traffic,
        );
        let plan_write = if !traffic.writes_an_image {
            0.0
        } else if is_materialised {
            constraints.model.materialise_cost_per_voxel
        } else {
            constraints.model.write_cost_per_voxel
        };
        let makespan = phase_makespan(
            &plan_cost,
            &phase.grid,
            workers,
            &constraints.model,
            plan_write,
        );
        let grid = phase.grid.clone();
        let phase = self.push(phase, Work::Iterate(Box::new(op)), true);
        Ok(PricedPhase {
            phase,
            grid,
            makespan,
            ranked,
            tally,
        })
    }

    /// Append a `Fragments` phase.
    ///
    /// **Responsibility four is here**, and it is the one the hand-built plan's
    /// author called the biggest: a fragment phase owns no chain slot, so
    /// `check_dtypes` has nothing to fold and asks the op instead — and the plan
    /// has to have allocated the image at the width the op says it writes.
    /// Stating that by hand is a line that is easy to omit and whose omission is
    /// refused by a message about an image's width rather than about the missing
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
    fn push(&mut self, phase: PhaseDecomposition, work: Work, writes_an_image: bool) -> Phase {
        let index = self.phases.len();
        self.phases.push(phase);
        self.work.push(work);
        Phase {
            index,
            writes: writes_an_image.then(|| ImageId(index + 1)),
        }
    }

    /// Close the plan: assemble the chain, derive what a chain can derive, and
    /// run every guard the executor would.
    ///
    /// **Responsibility six is `declare_source_images`**, and it is here rather
    /// than offered as a step because forgetting it is not refused by
    /// `Decomposition::check` — it surfaces later as an image freed before its
    /// second reader, or as an executor refusing a dependency it was never told
    /// about. A construction step that can be forgotten and whose omission is
    /// diagnosed somewhere else is not a step, it is a trap.
    ///
    /// The five checks are the executor's own, called unchanged. Running them
    /// here does not make the executor's copies redundant — a plan may arrive
    /// from any strategy or off a wire — it means a plan written by hand fails
    /// at the line that wrote it.
    ///
    /// # What it costs, and why that is worth a paragraph here
    ///
    /// This is on a partition search's inner loop: a search prices a candidate
    /// grid by building the plan it implies and closing it, so `finish` runs
    /// once per candidate and the fine candidates are the ones with the most
    /// blocks. It is therefore the one method in this file whose complexity is
    /// a feature rather than an implementation detail.
    ///
    /// It is **linear in the block count**, and every term is: the chain is
    /// assembled once, `declare_source_images` walks each phase's slots once,
    /// and of the five checks only `Decomposition::check` looks at blocks at
    /// all, once each. That was not always true — `check` asks
    /// [`crate::tiling::boxes_tile_exactly`] whether a phase's blocks cover
    /// their volume once each, and that predicate used to compare every pair of
    /// blocks, which made closing a plan quadratic and made it the *whole* cost
    /// of pricing a candidate: `445 ms` at `8192` blocks against `11 ms` to
    /// build the phases themselves, and half a minute at the block counts a
    /// fine grid asks for. `tiling`'s header has the algorithm that replaced it
    /// and the before/after; `tests/tiling_scaling.rs` pins the scaling with a
    /// step counter rather than a stopwatch.
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
        decomposition.declare_source_images(&workflow.chain)?;

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
            check_source_images(&assembly.workflow.chain, &assembly.decomposition)?;
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
        hand.declare_source_images(&chain).expect("source images");

        assert_eq!(built.decomposition, hand);
        assert_eq!(built.decomposition.fingerprint(), hand.fingerprint());
    }

    /// A phase handle knows which image its phase wrote, and the two numbers are
    /// different types so that they cannot be swapped.
    #[test]
    fn a_phase_handle_carries_the_image_it_wrote() {
        let mut plan = PlanBuilder::new([16, 8, 8], Dtype::F64, grid());
        let first = plan
            .pixels(Chain::op(IdentityOp::new("first", [1, 0, 0])))
            .expect("a pixel phase");
        assert_eq!(first.index(), 0);
        assert_eq!(first.image().expect("an image"), ImageId::from(1));
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

/// **What the bridge to the partition planner has to be true for.**
///
/// Four questions, and each of them is a way for this entry to look like it
/// works while being useless or wrong:
///
/// * **the answer must not change.** A chain the planner cut into several
///   phases has to compute what the same chain forced into one computes, bit
///   for bit. `pixels` still exists, so both plans are buildable and the
///   comparison is on voxels rather than on an argument;
/// * **the planner must actually be consulted.** A bridge that quietly always
///   made one phase would pass every other test here, so one chain is asserted
///   to come back cut and one to come back whole;
/// * **a source leaf must still name the image it meant.** This is the failure
///   that is silent: a leaf reads an image *by index*, and an entry that renumbers
///   images under one would read real voxels from the wrong array;
/// * **and the cut has to be worth making**, on the shape the gap was found on:
///   a large-reach op with reach-0 work either side of it, which is what fusing
///   makes expensive and what nobody was pricing.
#[cfg(test)]
mod planned_phases {
    use super::*;
    use crate::decomposition::{Constraints, CostModel};
    use crate::env::ArrayEnvironment;
    use crate::ops::background::DifferenceCombine;
    use crate::ops::smooth::{Gaussian, SmoothOp};
    use crate::ops::voxelwise::VoxelwiseMapOp;
    use crate::strategy::{execute_phases, Enumerating, Greedy, Hints, Strategy};
    use crate::voxels::Voxels;

    /// Small enough to run every plan it appears in, and still large enough for
    /// the blur's halo to be the thing that decides the block edge.
    const VOLUME: [usize; 3] = [32, 32, 8];
    /// A radius of 8 on x and y and nothing on z: the shape of the blur the
    /// consumer fuses — flat on z, wide in plane — at a size a test can run.
    const SIGMA: f64 = 2.0;
    /// A volume the consumer's own scale is worth measuring at. Nothing runs at
    /// this size; the plans are built and their geometry is read, which is where
    /// a read amplification comes from anyway.
    const WIDE: [usize; 3] = [128, 128, 32];
    /// The scale the consumer ships, whose radius is 40.
    const WIDE_SIGMA: f64 = 10.0;
    const TRUNCATE: f64 = 4.0;
    /// The fingerprint of the two-phase `pixels` plan below.
    ///
    /// A literal, and it is checkable rather than magic: the plan it belongs to
    /// is the one `tests::two_pixel_phases_assemble_to_the_hand_built_decomposition`
    /// builds a second time by hand and compares field for field, so the number
    /// is pinned to a decomposition written out in full a few hundred lines
    /// above rather than to whatever this module happens to produce.
    const PIXELS_FINGERPRINT: u64 = 6813685505909348196;

    /// The chain the gap was found on, at a size a test can run.
    ///
    /// A large-reach op between two reach-0 fan-ins, each of which reads an image
    /// back through a source leaf. Both halves matter: the reach-0 work is what
    /// pays the blur's halo when it is fused with it, and the source leaves are
    /// what make that expensive twice over — an image a phase sources is fetched
    /// at the block's own read extent, so a fused phase reads *three* images at
    /// the blur's halo where a cut one reads one.
    fn arm(values: ImageId, sink: ImageId, sigma: f64) -> Chain {
        Chain::sequence(vec![
            Chain::parallel(
                vec![
                    Chain::op(VoxelwiseMapOp::new("masked", |value| value * 0.5)),
                    Chain::source(values, Dtype::F64),
                ],
                Box::new(DifferenceCombine::new("masked source")),
            )
            .expect("a fan-in"),
            Chain::op(SmoothOp::new(
                "blur",
                Gaussian::new([sigma, sigma, 0.0], TRUNCATE).expect("a kernel"),
            )),
            Chain::parallel(
                vec![
                    Chain::op(VoxelwiseMapOp::new("blurred", |value| value)),
                    Chain::source(sink, Dtype::F64),
                ],
                Box::new(DifferenceCombine::new("residual")),
            )
            .expect("a fan-in"),
        ])
    }

    /// A chain with nothing to gain from a cut: three reach-0 maps.
    fn voxelwise_chain() -> Chain {
        Chain::sequence(vec![
            Chain::op(VoxelwiseMapOp::new("scale", |value| value * 2.0)),
            Chain::op(VoxelwiseMapOp::new("floor", |value| value.max(0.0))),
            Chain::op(VoxelwiseMapOp::new("shift", |value| value + 1.0)),
        ])
    }

    /// A budget the blur's halo actually binds against, and a candidate list to
    /// spend it on. The caller's, in full: see `PlanBuilder::partition`.
    fn constraints(budget: u64, candidates: Vec<usize>) -> Constraints {
        Constraints {
            budget_bytes: Some(budget),
            expected_concurrency: 1,
            model: CostModel::default(),
            block_candidates: candidates,
            split_axes: vec![0, 1, 2],
            ..Default::default()
        }
    }

    fn small_constraints() -> Constraints {
        constraints(64 << 10, vec![4, 8, 16, 32])
    }

    /// The two phases the arm reads back through its source leaves. Both plans
    /// below start with these, so everything the comparison sees is the arm's.
    fn preamble(plan: &mut PlanBuilder) -> (ImageId, ImageId) {
        let values = plan
            .pixels(Chain::op(VoxelwiseMapOp::new("values", |value| {
                value + 1.0
            })))
            .expect("a pixel phase")
            .image()
            .expect("an image");
        let sink = plan
            .pixels(Chain::op(VoxelwiseMapOp::new("sink", |value| value * 0.25)))
            .expect("a pixel phase")
            .image()
            .expect("an image");
        (values, sink)
    }

    /// The plan a caller who groups its own phases builds: one `pixels` call for
    /// the whole arm, on the largest block edge that fits the budget.
    ///
    /// The edge is `Greedy`'s rather than one this test picked, because "fuse
    /// everything and take the largest block that fits" is exactly what a caller
    /// grouping by hand is doing, and a number invented here would make the
    /// comparison a statement about the number.
    fn fused(volume: [usize; 3], sigma: f64, constraints: &Constraints) -> Assembly {
        let mut plan = PlanBuilder::new(volume, Dtype::F64, whole(volume));
        let (values, sink) = preamble(&mut plan);
        let workflow = Workflow::new(arm(values, sink, sigma), volume, Dtype::F64);
        let one = Greedy { concurrency: 1 }
            .decompose(&workflow, constraints)
            .expect("a fused plan");
        assert_eq!(
            one.n_phases(),
            1,
            "the fused baseline has to be fused, or it is measuring something else"
        );
        plan.regrid(one.phases[0].grid.clone());
        plan.pixels(arm(values, sink, sigma))
            .expect("one pixel phase");
        plan.finish().expect("a plan")
    }

    /// The same arm, partitioned by the planner.
    fn planned(volume: [usize; 3], sigma: f64, constraints: &Constraints) -> (Assembly, usize) {
        let mut plan = PlanBuilder::new(volume, Dtype::F64, whole(volume));
        let (values, sink) = preamble(&mut plan);
        let made = plan
            .partition(
                arm(values, sink, sigma),
                &Enumerating::default(),
                constraints,
            )
            .expect("a planned run of phases");
        let n = made.n_phases();
        assert_eq!(made.last().index(), plan.n_phases() - 1);
        (plan.finish().expect("a plan"), n)
    }

    /// The caller's own lattice for the two leading phases: one block, so that
    /// nothing about them can be confused with what the planner chose.
    fn whole(volume: [usize; 3]) -> BlockGrid {
        BlockGrid::new(volume, volume).expect("a lattice")
    }

    fn input(volume: [usize; 3]) -> Voxels {
        let n: usize = volume.iter().product();
        let values: Vec<f64> = (0..n)
            .map(|index| ((index * 37) % 101) as f64 / 101.0)
            .collect();
        ndarray::Array3::from_shape_vec(volume, values)
            .expect("a volume")
            .into()
    }

    /// Run a plan and hand back the image it wrote last.
    fn run(assembly: &Assembly) -> Voxels {
        let env = ArrayEnvironment::for_decomposition(
            input(assembly.decomposition.volume),
            &assembly.decomposition,
            [8, 8, 8],
        )
        .expect("an environment");
        execute_phases(
            "partition",
            &assembly.workflow,
            &assembly.decomposition,
            &Hints::default(),
            &env,
            &[],
            &assembly.work(),
        )
        .expect("a run");
        env.image(assembly.decomposition.n_phases())
    }

    /// Voxels read over the arm's phases only — the two leading phases are the
    /// same in both plans and are not what is being compared.
    fn arm_reads(assembly: &Assembly) -> usize {
        assembly.decomposition.exact_read_voxels()[2..].iter().sum()
    }

    fn arm_blocks(assembly: &Assembly) -> Vec<usize> {
        assembly.decomposition.phases[2..]
            .iter()
            .map(|phase| phase.grid.n_blocks())
            .collect()
    }

    // ------------------------------------------------ one: the answer --

    /// **The safety property, on bits.** A chain the planner cut computes what
    /// the same chain forced into one computes.
    ///
    /// Not "close": every voxel's `f64` compared by its bit pattern, because a
    /// halo one voxel short of the reach is worth about an ulp at the seams and
    /// would pass any tolerance somebody chose.
    #[test]
    fn a_planned_run_of_phases_computes_what_the_fused_one_computes() {
        let constraints = small_constraints();
        let fused = fused(VOLUME, SIGMA, &constraints);
        let (planned, phases) = planned(VOLUME, SIGMA, &constraints);
        assert_eq!(fused.n_phases(), 3, "two leading phases and the fused arm");
        assert!(
            phases > 1,
            "the planner declined to cut, so this proves nothing"
        );

        let from_fused = run(&fused);
        let from_planned = run(&planned);
        assert_eq!(from_fused.shape(), from_planned.shape());
        let fused_view = from_fused.view::<f64>().expect("f64");
        let planned_view = from_planned.view::<f64>().expect("f64");
        let differing = fused_view
            .iter()
            .zip(planned_view.iter())
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        assert_eq!(differing, 0, "{differing} voxels differ between the plans");
        // and it is an answer rather than two empty volumes agreeing
        assert!(
            fused_view.iter().any(|value| *value != 0.0),
            "the fixture computed nothing at all"
        );
    }

    // --------------------------------------- two: the planner was asked --

    /// The chain that wants cutting comes back cut, and the phases are the
    /// planner's own — same count, same lattices, same slots, renumbered.
    #[test]
    fn the_planner_decides_how_many_phases_a_chain_becomes() {
        let constraints = small_constraints();
        let (assembly, phases) = planned(VOLUME, SIGMA, &constraints);
        assert!(
            phases > 1,
            "the planner was not consulted: one call, one phase, every time"
        );

        // The same partition the planner returns when it is asked directly, so
        // that what arrived in the plan is its answer and not a rounding of it.
        let workflow = Workflow::new(arm(ImageId(1), ImageId(2), SIGMA), VOLUME, Dtype::F64);
        let direct = Enumerating::default()
            .decompose(&workflow, &constraints)
            .expect("a plan");
        assert_eq!(direct.n_phases(), phases);
        for (index, phase) in direct.phases.iter().enumerate() {
            let landed = &assembly.decomposition.phases[2 + index];
            assert_eq!(landed.grid, phase.grid, "phase {index} lattice");
            assert_eq!(landed.reach, phase.reach, "phase {index} reach");
            assert_eq!(landed.halo, phase.halo, "phase {index} halo");
            // renumbered by the cursor, and by nothing else
            let shifted: Vec<usize> = phase.slots.iter().map(|slot| slot + 2).collect();
            assert_eq!(landed.slots, shifted, "phase {index} slots");
        }
    }

    /// And a chain with nothing to gain from a cut comes back whole. The other
    /// half of the same assertion: an entry that always cut would be as wrong as
    /// one that never did.
    #[test]
    fn a_chain_the_planner_will_not_cut_stays_one_phase() {
        let constraints = small_constraints();
        let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, whole(VOLUME));
        let made = plan
            .partition(voxelwise_chain(), &Enumerating::default(), &constraints)
            .expect("a planned run of phases");
        assert_eq!(made.n_phases(), 1);
        assert_eq!(made.image(), ImageId(1));
        let assembly = plan.finish().expect("a plan");
        assert_eq!(assembly.decomposition.n_phases(), 1);
        assert_eq!(assembly.decomposition.phases[0].slots, vec![0, 1, 2]);
    }

    // ------------------------------------- three: the source leaf's image --

    /// **The silent failure, checked.** A source leaf names an image of the whole
    /// plan; cutting the chain around it must not move the image it means.
    ///
    /// Cutting can only ever move a slot *later*, and the rule is that a leaf's
    /// image is at most its phase's own — so the direction cutting moves in is
    /// the safe one. This asserts the outcome rather than the argument: the
    /// images recorded against the phases are the images the chain named, in
    /// both plans, whatever the partition did.
    #[test]
    fn cutting_a_chain_does_not_renumber_the_images_its_source_leaves_name() {
        let constraints = small_constraints();
        let fused = fused(VOLUME, SIGMA, &constraints);
        let (planned, phases) = planned(VOLUME, SIGMA, &constraints);
        assert!(phases > 1);

        // Fused: one phase, both leaves, both images.
        assert_eq!(fused.decomposition.phases[2].source_images, vec![1, 2]);
        // Cut: the same two images, now in the phases the slots landed in, and
        // every one of them at or below its own phase's input image.
        let named: Vec<usize> = planned.decomposition.phases[2..]
            .iter()
            .flat_map(|phase| phase.source_images.iter().copied())
            .collect();
        assert_eq!(named, vec![1, 2]);
        for (index, phase) in planned.decomposition.phases.iter().enumerate() {
            for &image in &phase.source_images {
                assert!(
                    image <= index,
                    "phase {index} sources image {image}, which it runs before"
                );
            }
        }
        // `finish` ran `check_source_images` over both, which is the guard this
        // is the readable form of.
    }

    // ------------------------------------------ four: the cut is worth it --

    /// **The measurement, on the shape the gap was found on.** Reported as
    /// exact voxels read — the clamped geometry, not a model — for the fused
    /// plan and for the planned one.
    #[test]
    fn fusing_the_large_reach_op_with_its_neighbours_reads_the_volume_far_more_times() {
        let constraints = small_constraints();
        let fused = fused(VOLUME, SIGMA, &constraints);
        let (planned, phases) = planned(VOLUME, SIGMA, &constraints);
        let voxels: usize = VOLUME.iter().product();

        let fused_reads = arm_reads(&fused);
        let planned_reads = arm_reads(&planned);
        println!(
            "arm fused:   {:?} block(s), {fused_reads} voxels read, {:.3}x the volume",
            arm_blocks(&fused),
            fused_reads as f64 / voxels as f64
        );
        println!(
            "arm planned: {:?} block(s), {planned_reads} voxels read, {:.3}x the volume",
            arm_blocks(&planned),
            planned_reads as f64 / voxels as f64
        );
        assert_eq!(phases, 3);
        assert!(
            planned_reads * 3 < fused_reads * 2,
            "the planned partition has to be worth making: {planned_reads} against {fused_reads}"
        );
    }

    /// The same measurement at the consumer's own numbers: a radius-40 blur on a
    /// `[128, 128, 32]` volume against a 4 MiB block budget.
    ///
    /// Nothing is run here — a read amplification comes from the plan's clamped
    /// geometry, which is the same figure whether or not anybody executes it —
    /// so this is the size the gap was actually found at rather than the size a
    /// test can afford to run.
    #[test]
    fn the_same_holds_at_the_radius_and_volume_the_gap_was_found_at() {
        let constraints = constraints(4 << 20, vec![16, 32, 64, 128]);
        let fused = fused(WIDE, WIDE_SIGMA, &constraints);
        let (planned, phases) = planned(WIDE, WIDE_SIGMA, &constraints);
        let voxels: usize = WIDE.iter().product();

        let fused_reads = arm_reads(&fused);
        let planned_reads = arm_reads(&planned);
        println!(
            "wide arm fused:   {:?} block(s), {fused_reads} voxels read, {:.3}x the volume",
            arm_blocks(&fused),
            fused_reads as f64 / voxels as f64
        );
        println!(
            "wide arm planned: {:?} block(s), {planned_reads} voxels read, {:.3}x the volume",
            arm_blocks(&planned),
            planned_reads as f64 / voxels as f64
        );
        assert_eq!(phases, 3);
        assert!(
            planned_reads * 2 < fused_reads,
            "{planned_reads} against {fused_reads}"
        );
    }

    // ------------------------------------ and what the entry does not touch --

    /// **`pixels` is untouched, and the fingerprint says so.**
    ///
    /// A pinned literal rather than a comparison against a plan built here,
    /// because the thing worth catching is the whole assembly path moving under
    /// a plan somebody already has a parity figure for. This is the number a
    /// two-phase `pixels` plan has always had.
    #[test]
    fn a_plan_built_with_pixels_fingerprints_as_it_always_has() {
        let volume = [16usize, 8, 8];
        let grid = BlockGrid::new(volume, [8, 8, 8]).expect("a lattice");
        let mut plan = PlanBuilder::new(volume, Dtype::F64, grid);
        plan.pixels(Chain::op(crate::probes::IdentityOp::new(
            "first",
            [1, 0, 0],
        )))
        .expect("a pixel phase");
        plan.pixels(Chain::op(crate::probes::AffineOp::new(
            "second",
            2.0,
            1.0,
            [0, 2, 0],
        )))
        .expect("a pixel phase");
        let built = plan.finish().expect("a plan");
        assert_eq!(built.decomposition.n_phases(), 2);
        assert_eq!(built.decomposition.fingerprint(), PIXELS_FINGERPRINT);
    }

    /// A run of phases has to hold at least one, and a chain with no slots holds
    /// none — refused where it is written, in the words `pixels` refuses it in.
    #[test]
    fn a_planned_run_over_a_chain_with_no_slots_is_refused() {
        let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, whole(VOLUME));
        let message = plan
            .partition(
                Chain::sequence(Vec::new()),
                &Enumerating::default(),
                &small_constraints(),
            )
            .expect_err("no slots")
            .to_string();
        assert!(message.contains("nothing to partition"), "{message}");
    }
}
