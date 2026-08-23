// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **What the barrier and the hoisted reduction are worth to `ops::label` and
// `ops::detect`, measured on the shipped ops.**
//
// `tests/barrier_phase.rs` measures the mechanism on a probe op built for the
// purpose — one op with two booleans, `[16, 16, 16]`, a fragment that weighs
// eight bytes. `docs/design/barriers.md` §8.8 is explicit that the *structure*
// of that table transfers and the *constants* do not, because a fragment there
// is eight bytes and a fragment in these two ops is a block **face**. This file
// is the same arms over the two ops that actually ship one, so that the
// constants are measured rather than carried over.
//
// The arms, and why they are built here rather than switched on
// -------------------------------------------------------------
// The shipped ops declare one thing. To measure what the declaration is worth,
// the other shapes have to exist beside it, and they are built here — as
// [`RelabelArm`] and [`RegionArm`], which call the **same public merge** the
// shipped ops call and differ from them only in what they declare:
//
// | arm | `barrier()` | the fold | fragment reach | `SeamFold` | folds |
// |---|---|---|---|---|---|
// | in-plan | `false` | in `apply` | whole lattice | not said | once per block |
// | barrier alone | `true` | in `apply` | whole lattice | not said | once per block |
// | hoisted, unchecked | `true` | in `reduce` | zero | not said | once |
// | hoisted, checked | `true` | in `reduce` | zero | `Unordered` | twice |
// | **shipped** | what `ops::label` and `ops::detect` declare | — | — | — | — |
//
// **The fourth arm exists because the order check is not free, and the design
// note says it nearly is.** `barriers.md` §8.5 prices the reversed-lattice check
// at "one extra reduction for an op that opted in, against the one extra `apply`
// per block the same declaration already costs" — which is the right comparison
// for an op that *already* declared `SeamFold::Unordered`. Neither of these two
// did. So for them it is not a saving that got smaller, it is a new cost, and it
// is not only CPU: the second reduction re-reads the whole fragment set out of
// the store, so the hoisted arm transmits the set **three** times rather than
// twice — written once, read twice. `the_order_check_costs_one_more_pass_over_
// the_fragment_set` measures it, and it is why the two hoisted arms are run
// separately rather than the shipped declaration being assumed free.
//
// The in-plan arm is character for character what the two ops did before this
// change, so it is the **liveness control** the migration needs: an op whose
// barrier is withheld must produce the same answer, or the barrier is doing
// something other than what it claims. That control is not a mutation of the
// shipped op — it is a fourth op declaring the old thing, run beside it, and its
// bytes are compared against the shipped op's at every lattice.
//
// A knob on the shipped op was the other way to get it, and it was rejected: a
// production op does not need a public way to ask for the shape that costs 25x,
// and a control that is a mode of the thing under test is a control that shares
// its bugs.
//
// What is asserted and what is only reported
// -------------------------------------------
// **Asserted**: every arm agrees byte for byte with every other at every
// lattice; the fold count is the block count without the hoisting and one or two
// with it; the shipped ops' traffic is exactly the checked hoisted arm's, which
// is what pins the shipped declaration to an arm whose folds were counted; the
// traffic ordering `in-plan >= barrier alone >= hoisted` holds, with the
// *equality* in `detect`'s first pair asserted as an equality rather than
// tolerated, because it is a prediction of the design note and not a wobble.
//
// **Reported only**: seconds, and the merge CPU. Wall clock on a shared machine
// is the least trustworthy signal here and every byte column is exact.
//
// The prediction this file exists to test, for `ops::detect`
// -----------------------------------------------------------
// `ops::detect`'s phase 1 declares `reads_pixels() == false`, so it pays no
// pixel re-reads at all — and a barrier's whole traffic contribution is the
// relief of the pixel re-reads. So the prediction is that **a barrier alone is
// worth exactly nothing** to `detect`: not "a little", not "less than for
// `label`", but zero bytes, with everything it pays sitting in the hoisting.
// [`a_barrier_alone_is_worth_nothing_at_all_to_detect`] asserts it as an
// equality.
//
// What it found, on this fixture
// -------------------------------
// `[16, 128, 128]`, 73 275 set voxels, total bytes moved at 256 blocks:
//
// | | in-plan | barrier alone | hoisted, unchecked | as shipped |
// |---|---|---|---|---|
// | `ops::label` | 500.94 MiB | 245.94 | 4.31 | **6.08** |
// | `ops::detect` | 659.62 MiB | **659.62** | 6.51 | **9.07** |
//
// `ops::detect`'s second column is the prediction, and the two figures are the
// same number and not merely close.
//
// **These constants are this fixture's.** `tests/label_materialisation_cost.rs`
// is the same measurement on a recorded volume for `ops::label`, and its header
// carries what the migration did there: 106.07 GiB to 4.84 at 256 blocks, with
// every other arm reproducing to the digit.
//
// What is not covered here
// ------------------------
// **A distributed run of either op.** `barriers.md` §9 establishes that a
// hoisted reduction distributes — every worker derives the same blob from the
// shared fragment set with nothing sent — and `tests/barrier_multi_node.rs`
// exercises it across real processes. It does so with
// `distributed::spec::HoistedReduceOp`, a probe: `FragmentPhaseSpec::kind` names
// a fixed set of probe ops, so there is no way from a `JobSpec` to say
// "`ops::detect`". Nothing here or there runs *these* ops on more than one node.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array3;

use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, BlockBuf, Environment};
use blockflow::fragment::{
    fragment_phase, BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput,
    PhaseView, PhaseWork, SeamFold,
};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::components::{core_within_read, Connectivity};
use blockflow::ops::detect::{
    decode_moments, encode_moments, merge_moments_with, moments_owned_by, Emission, LabelRegionsOp,
    Moments, RegionMoments, RegionPointsOp,
};
use blockflow::ops::label::{ComponentFaces, GlobalLabels, LabelComponentsOp, RelabelComponentsOp};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [16, 128, 128];
const FACES: &str = "faces";
const MOMENTS: &str = "moments";
const POINTS: &str = "points";

/// One, four, thirty-two and two hundred and fifty-six blocks — the lattices
/// `barriers.md` §8.8 tabulates, so the two tables can be read against each
/// other.
fn blockings() -> Vec<[usize; 3]> {
    vec![VOLUME, [16, 64, 64], [8, 32, 32], [4, 16, 16]]
}

/// A sparse pseudorandom mask, `tests/global_labels.rs`' fixture at this volume
/// and for its reason: many small components in no arranged position, so some of
/// them cross a seam at every lattice and none of them was placed to.
fn mask() -> Array3<bool> {
    let mut out = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                out[[i, j, k]] = (state >> 33) % 100 < 28;
            }
        }
    }
    out
}

// ------------------------------------------------------------- what an arm is --

/// Which of the three shapes an arm declares. `Shipped` is not one of them: it
/// is the real op, and which shape it turns out to be is what is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    InPlan,
    BarrierOnly,
    Hoisted,
    HoistedChecked,
}

/// Every arm, coarsest declaration first.
const ARMS: [Arm; 4] = [
    Arm::InPlan,
    Arm::BarrierOnly,
    Arm::Hoisted,
    Arm::HoistedChecked,
];

impl Arm {
    fn barrier(self) -> bool {
        self != Arm::InPlan
    }

    fn hoisted(self) -> bool {
        self == Arm::Hoisted || self == Arm::HoistedChecked
    }

    /// Whether the arm declares `SeamFold::Unordered`, which is what makes the
    /// executor run the reduction a second time over the reversed lattice.
    fn order_checked(self) -> bool {
        self == Arm::HoistedChecked
    }

    fn name(self) -> &'static str {
        match self {
            Arm::InPlan => "in-plan",
            Arm::BarrierOnly => "barrier alone",
            Arm::Hoisted => "hoisted, unchecked",
            Arm::HoistedChecked => "hoisted, checked",
        }
    }
}

/// Every byte an arm moved: pixels in, pixels out, fragments both ways.
///
/// Summed rather than compared column by column, for
/// `tests/label_materialisation_cost.rs`' reason: the arms trade a pixel read
/// for a fragment gather, so a ratio on one column would flatter whichever
/// design is cheap in it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Traffic {
    read_bytes: u64,
    write_bytes: u64,
    fragment_bytes: u64,
}

impl Traffic {
    fn of(env: &dyn Environment) -> Self {
        let counters = env.counters();
        Self {
            read_bytes: counters.read_bytes.load(Ordering::Relaxed),
            write_bytes: counters.write_bytes.load(Ordering::Relaxed),
            fragment_bytes: counters.sidecar_bytes_written.load(Ordering::Relaxed)
                + counters.sidecar_bytes_read.load(Ordering::Relaxed),
        }
    }

    fn total(self) -> u64 {
        self.read_bytes + self.write_bytes + self.fragment_bytes
    }
}

/// One run of one arm: what it moved, how long it took, how many times it folded
/// the fragment set, and what it answered.
#[derive(Debug, Clone)]
struct Run {
    traffic: Traffic,
    seconds: f64,
    folds: usize,
    answer: Vec<u8>,
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

// ----------------------------------------------------- ops::label, the arms --

/// `ops::label`'s phase 1, in whichever of the three shapes it is asked for.
///
/// Everything below the declarations is [`RelabelComponentsOp`]'s own body,
/// calling the same public [`GlobalLabels::merge`] and the same
/// [`GlobalLabels::remap_block`], so a difference between this and the shipped
/// op is a difference in what was *declared* and nothing else.
struct RelabelArm {
    arm: Arm,
    lattice: [usize; 3],
    folds: Arc<AtomicUsize>,
}

impl RelabelArm {
    fn new(arm: Arm, grid: &BlockGrid) -> Self {
        Self {
            arm,
            lattice: grid.blocks_per_axis(),
            folds: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn merge(
        &self,
        reports: &BTreeMap<[usize; 3], ComponentFaces>,
        grid: &BlockGrid,
    ) -> Result<GlobalLabels, blockflow::Error> {
        let table = GlobalLabels::merge(reports, grid, Connectivity::Faces)?;
        self.folds.fetch_add(1, Ordering::SeqCst);
        Ok(table)
    }
}

impl FragmentOp for RelabelArm {
    fn name(&self) -> &'static str {
        "relabel-arm"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn barrier(&self) -> bool {
        self.arm.barrier()
    }

    fn gathers(&self) -> bool {
        !self.arm.hoisted()
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        let reach = if self.arm.hoisted() {
            [0, 0, 0]
        } else {
            self.lattice
        };
        vec![FragmentInput::own(FACES.to_string(), 0).with_reach(reach)]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        self.arm.order_checked().then_some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>, blockflow::Error> {
        if !self.arm.hoisted() {
            return Ok(Vec::new());
        }
        let mut reports = BTreeMap::new();
        for (key, bytes) in at.fragments(FACES)? {
            reports.insert(key.block, ComponentFaces::decode(&bytes)?);
        }
        Ok(self.merge(&reports, at.grid)?.encode())
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput, blockflow::Error> {
        let mut buffer = at.output_buffer(0.0)?;
        let BlockBuf::Array(pixels) = at.pixels()? else {
            return Ok(BlockOutput::nothing().with_pixels(buffer));
        };
        let labels = pixels.view::<u32>()?;
        let global = if self.arm.hoisted() {
            GlobalLabels::decode(at.reduced)?
        } else {
            let mut reports = BTreeMap::new();
            for (key, bytes) in at.fragments(FACES) {
                reports.insert(key.block, ComponentFaces::decode(bytes)?);
            }
            self.merge(&reports, at.grid)?
        };
        let (offset, extent) = core_within_read(at)?;
        let window = ndarray::s![
            offset[0]..offset[0] + extent[0],
            offset[1]..offset[1] + extent[1],
            offset[2]..offset[2] + extent[2],
        ];
        let BlockBuf::Array(out) = &mut buffer else {
            unreachable!("the environment gave data for the input and none for the output");
        };
        let mut view = out.view_mut::<u32>()?;
        global.remap_block(at.index, labels.slice(window), view.slice_mut(window))?;
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

/// The two-phase plan, whichever op is phase 1.
fn label_plan(grid: BlockGrid, phase1: &dyn FragmentOp) -> (Decomposition, LabelComponentsOp) {
    let label = LabelComponentsOp::new("label", FACES, Lifecycle::DeleteOnExit);
    let volume = grid.volume();
    let mut labelling = fragment_phase(&label, grid.clone()).expect("phase 0");
    labelling.dtype = Some(Dtype::U32);
    let mut relabelling = fragment_phase(phase1, grid).expect("phase 1");
    relabelling.dtype = Some(Dtype::U32);
    let plan = Decomposition {
        volume,
        dtype: Dtype::Bool,
        phases: vec![labelling, relabelling],
        chain_reach: [0, 0, 0],
    };
    plan.check().expect("a plan");
    (plan, label)
}

fn run_label(mask: &Array3<bool>, block: [usize; 3], arm: Option<Arm>) -> Run {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let shipped = RelabelComponentsOp::new("relabel", FACES, 0, &grid);
    let control = arm.map(|arm| RelabelArm::new(arm, &grid));
    let phase1: &dyn FragmentOp = match &control {
        Some(arm) => arm,
        None => &shipped,
    };
    let (plan, label) = label_plan(grid, phase1);
    let input: Voxels = mask.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, [4, 16, 16]).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::Bool);
    let started = Instant::now();
    execute_phases(
        "relabel",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(phase1)],
    )
    .expect("a run");
    let seconds = started.elapsed().as_secs_f64();
    let answer = env
        .output()
        .view::<u32>()
        .expect("a label volume")
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    Run {
        traffic: Traffic::of(&env),
        seconds,
        folds: control
            .map(|arm| arm.folds.load(Ordering::SeqCst))
            .unwrap_or(0),
        answer,
    }
}

// ---------------------------------------------------- ops::detect, the arms --

/// `ops::detect`'s phase 1, in whichever of the three shapes it is asked for.
///
/// [`RegionPointsOp`]'s own body below the declarations, calling the same public
/// [`merge_moments_with`] and the same [`moments_owned_by`].
struct RegionArm {
    arm: Arm,
    lattice: [usize; 3],
    folds: Arc<AtomicUsize>,
}

impl RegionArm {
    fn new(arm: Arm, grid: &BlockGrid) -> Self {
        Self {
            arm,
            lattice: grid.blocks_per_axis(),
            folds: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn merge(
        &self,
        reports: &BTreeMap<[usize; 3], RegionMoments>,
        counts: [usize; 3],
    ) -> Result<Vec<Moments>, blockflow::Error> {
        let totals = merge_moments_with(reports, counts, Connectivity::Faces)?;
        self.folds.fetch_add(1, Ordering::SeqCst);
        Ok(totals)
    }
}

impl FragmentOp for RegionArm {
    fn name(&self) -> &'static str {
        "region-arm"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn barrier(&self) -> bool {
        self.arm.barrier()
    }

    fn gathers(&self) -> bool {
        !self.arm.hoisted()
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        let reach = if self.arm.hoisted() {
            [0, 0, 0]
        } else {
            self.lattice
        };
        vec![FragmentInput::own(MOMENTS.to_string(), 0).with_reach(reach)]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        self.arm.order_checked().then_some(SeamFold::Unordered)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            POINTS.to_string(),
            Lifecycle::Persistent,
            Coverage::EveryBlock,
        )]
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>, blockflow::Error> {
        if !self.arm.hoisted() {
            return Ok(Vec::new());
        }
        let mut reports = BTreeMap::new();
        for (key, bytes) in at.fragments(MOMENTS)? {
            reports.insert(key.block, RegionMoments::decode(&bytes)?);
        }
        let totals = self.merge(&reports, at.grid.blocks_per_axis())?;
        encode_moments(&totals)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput, blockflow::Error> {
        let components = if self.arm.hoisted() {
            decode_moments(at.reduced)?
        } else {
            let mut reports = BTreeMap::new();
            for (key, bytes) in at.fragments(MOMENTS) {
                reports.insert(key.block, RegionMoments::decode(bytes)?);
            }
            self.merge(&reports, at.grid.blocks_per_axis())?
        };
        let mine = moments_owned_by(&components, at.grid, at.index);
        Ok(BlockOutput::fragment(
            POINTS.to_string(),
            Emission::Point.encode(&mine)?,
        ))
    }
}

fn detect_plan(grid: BlockGrid, phase1: &dyn FragmentOp) -> (Decomposition, LabelRegionsOp) {
    let label = LabelRegionsOp::new("label", MOMENTS, Lifecycle::DeleteOnExit);
    let volume = grid.volume();
    let labelling = fragment_phase(&label, grid.clone()).expect("phase 0");
    let detecting = fragment_phase(phase1, grid).expect("phase 1");
    let plan = Decomposition {
        volume,
        dtype: Dtype::Bool,
        phases: vec![labelling, detecting],
        chain_reach: [0, 0, 0],
    };
    plan.check().expect("a plan");
    (plan, label)
}

fn run_detect(mask: &Array3<bool>, block: [usize; 3], arm: Option<Arm>) -> Run {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let shipped = RegionPointsOp::new("points", MOMENTS, 0, POINTS, Lifecycle::Persistent, &grid);
    let control = arm.map(|arm| RegionArm::new(arm, &grid));
    let phase1: &dyn FragmentOp = match &control {
        Some(arm) => arm,
        None => &shipped,
    };
    let (plan, label) = detect_plan(grid.clone(), phase1);
    let input: Voxels = mask.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, [4, 16, 16]).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::Bool);
    let started = Instant::now();
    execute_phases(
        "detect",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(phase1)],
    )
    .expect("a run");
    let seconds = started.elapsed().as_secs_f64();

    // The answer is every block's point blob, in lattice order. Reading it back
    // out of the store rather than out of the op, because the store is what a
    // consumer reads.
    let counts = grid.blocks_per_axis();
    let mut answer = Vec::new();
    for i in 0..counts[0] {
        for j in 0..counts[1] {
            for k in 0..counts[2] {
                let bytes = env
                    .read_sidecar(POINTS, 1, [i, j, k])
                    .expect("the store answers")
                    .unwrap_or_else(|| panic!("block {:?} wrote no points", [i, j, k]));
                answer.extend_from_slice(&bytes);
            }
        }
    }
    Run {
        traffic: Traffic::of(&env),
        seconds,
        folds: control
            .map(|arm| arm.folds.load(Ordering::SeqCst))
            .unwrap_or(0),
        answer,
    }
}

// ------------------------------------------------------------ the fixture --

/// **The lattices are genuinely distinct and genuinely cut**, run before
/// anything uses them.
///
/// The standing lesson: a decomposition-invariance sweep here had decayed to two
/// grids, one of them a single block, and passed while measuring nothing.
#[test]
fn the_lattices_are_distinct_and_the_fixture_has_seams_to_cross() {
    let mut seen: Vec<[usize; 3]> = Vec::new();
    let mut single = 0;
    let mut all_three = 0;
    for block in blockings() {
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        let counts = grid.blocks_per_axis();
        assert!(
            !seen.contains(&counts),
            "{block:?} is the {counts:?} lattice a previous block edge already produced"
        );
        seen.push(counts);
        if grid.n_blocks() == 1 {
            single += 1;
        }
        if counts.iter().all(|&n| n > 1) {
            all_three += 1;
        }
    }
    assert_eq!(single, 1, "the no-seam case is worth having exactly once");
    assert!(
        all_three >= 2,
        "at least two lattices must cut all three axes, so lattice edges and corners are \
         crossed and not only faces: {seen:?}"
    );

    // The merge is load-bearing: a mask whose components never crossed a seam
    // would make every arm agree for a reason that is not the barrier.
    let mask = mask();
    let set = mask.iter().filter(|&&on| on).count();
    assert!(
        set > VOLUME.iter().product::<usize>() / 10,
        "{set} set voxel(s) is too sparse to make components that cross"
    );
    let coarse = run_label(&mask, VOLUME, Some(Arm::InPlan));
    let fine = run_label(&mask, [4, 16, 16], Some(Arm::InPlan));
    assert_eq!(
        coarse.answer, fine.answer,
        "the control arm is not decomposition-invariant, so nothing below means anything"
    );
    assert!(
        fine.folds > coarse.folds,
        "the fine lattice folded no more often than the single block"
    );
}

// ------------------------------------------------------- the three arms --

/// **Every arm answers the same bytes, and the shipped op is one of them.**
///
/// The liveness control the migration needs: withhold the barrier and the answer
/// must not move. If it did, the barrier would be doing something other than
/// what it claims.
#[test]
fn the_shipped_ops_answer_what_the_pre_barrier_shape_answered_at_every_lattice() {
    let mask = mask();
    for block in blockings() {
        let control = run_label(&mask, block, Some(Arm::InPlan));
        for arm in ARMS.into_iter().skip(1) {
            let run = run_label(&mask, block, Some(arm));
            assert_eq!(
                run.answer,
                control.answer,
                "ops::label, {block:?}: the {} arm disagrees with the pre-barrier shape",
                arm.name()
            );
        }
        let shipped = run_label(&mask, block, None);
        assert_eq!(
            shipped.answer, control.answer,
            "ops::label, {block:?}: the shipped op disagrees with the pre-barrier shape"
        );

        let control = run_detect(&mask, block, Some(Arm::InPlan));
        for arm in ARMS.into_iter().skip(1) {
            let run = run_detect(&mask, block, Some(arm));
            assert_eq!(
                run.answer,
                control.answer,
                "ops::detect, {block:?}: the {} arm disagrees with the pre-barrier shape",
                arm.name()
            );
        }
        let shipped = run_detect(&mask, block, None);
        assert_eq!(
            shipped.answer, control.answer,
            "ops::detect, {block:?}: the shipped op disagrees with the pre-barrier shape"
        );
    }
}

/// **The fold runs once per block without the hoisting and once with it.**
///
/// `barriers.md` §7.2's quantity, and the one no byte column shows: the merge is
/// 0.13 s a call on the recorded volume and there are `blocks` calls, which is
/// the multiplier nobody had taken.
#[test]
fn hoisting_folds_the_fragment_set_once_instead_of_once_per_block() {
    let mask = mask();
    for block in blockings() {
        let blocks = BlockGrid::new(VOLUME, block).expect("a lattice").n_blocks();
        for arm in [Arm::InPlan, Arm::BarrierOnly] {
            assert_eq!(
                run_label(&mask, block, Some(arm)).folds,
                blocks,
                "ops::label, {block:?}, {}: the merge must run once per block",
                arm.name()
            );
            assert_eq!(
                run_detect(&mask, block, Some(arm)).folds,
                blocks,
                "ops::detect, {block:?}, {}: the merge must run once per block",
                arm.name()
            );
        }
        assert_eq!(
            run_label(&mask, block, Some(Arm::Hoisted)).folds,
            1,
            "ops::label, {block:?}: the hoisted arm must fold once"
        );
        assert_eq!(
            run_detect(&mask, block, Some(Arm::Hoisted)).folds,
            1,
            "ops::detect, {block:?}: the hoisted arm must fold once"
        );

        // With `SeamFold::Unordered` — which is what the shipped ops declare —
        // once more, over the reversed lattice, and **only** once more. Two is
        // flat in the block count and that is the whole of the claim: the
        // multiplier that was `blocks` is gone, not made smaller. A one-block
        // lattice has no order, so the executor skips the second pass and the
        // count is one there.
        let checked = if blocks == 1 { 1 } else { 2 };
        assert_eq!(
            run_label(&mask, block, Some(Arm::HoistedChecked)).folds,
            checked,
            "ops::label, {block:?}: the checked hoisted arm must fold {checked} time(s)"
        );
        assert_eq!(
            run_detect(&mask, block, Some(Arm::HoistedChecked)).folds,
            checked,
            "ops::detect, {block:?}: the checked hoisted arm must fold {checked} time(s)"
        );
    }
}

/// **The shipped ops move exactly the checked hoisted arm's bytes.**
///
/// This is what makes the fold count above evidence about the shipped ops: they
/// are not instrumented, and this equality pins their declaration to the arm
/// whose folds were counted. A shipped op that had kept the whole-lattice reach
/// would move the in-plan arm's fragment bytes and this would say so.
#[test]
fn the_shipped_ops_are_the_checked_hoisted_arm() {
    let mask = mask();
    for block in blockings() {
        assert_eq!(
            run_label(&mask, block, None).traffic,
            run_label(&mask, block, Some(Arm::HoistedChecked)).traffic,
            "ops::label, {block:?}: the shipped op is not the checked hoisted arm"
        );
        assert_eq!(
            run_detect(&mask, block, None).traffic,
            run_detect(&mask, block, Some(Arm::HoistedChecked)).traffic,
            "ops::detect, {block:?}: the shipped op is not the checked hoisted arm"
        );
    }
}

/// **What the two shipped ops declare, asserted directly rather than inferred
/// from what they moved.**
///
/// This test exists because a mutation survived without it. Putting the
/// whole-lattice fragment reach back on [`RelabelComponentsOp::inputs`] changed
/// no byte and no answer, because `gathers() == false` means the executor
/// fetches nothing on the op's behalf and the op's `apply` asks for nothing —
/// so the reach became a statement with no consequence in traffic. It is not
/// without consequence: a non-zero reach makes the neighbourhood more than one
/// fragment, which turns the executor's `SeamFold::Unordered` check back on
/// **per block**, so every block applies twice. That is CPU and no byte column
/// shows it, which is exactly the class of cost `barriers.md` §7.2 is about.
///
/// So the declarations are asserted as declarations: what the op says, and what
/// the plan records from it.
#[test]
fn the_two_ops_declare_a_barrier_a_reduction_and_no_reach() {
    let grid = BlockGrid::new(VOLUME, [4, 16, 16]).expect("a lattice");
    let relabel = RelabelComponentsOp::new("relabel", FACES, 0, &grid);
    let points = RegionPointsOp::new("points", MOMENTS, 0, POINTS, Lifecycle::Persistent, &grid);

    for (name, op) in [
        ("ops::label", &relabel as &dyn FragmentOp),
        ("ops::detect", &points as &dyn FragmentOp),
    ] {
        assert!(op.barrier(), "{name}: the phase must declare a barrier");
        assert!(
            !op.gathers(),
            "{name}: a hoisted reduction leaves a block nothing to gather"
        );
        assert_eq!(
            op.seam_fold(),
            Some(blockflow::fragment::SeamFold::Unordered),
            "{name}: an op that folds must say which kind of fold it is"
        );
        let inputs = op.inputs();
        assert_eq!(inputs.len(), 1, "{name}: one stream");
        assert_eq!(
            inputs[0].reach,
            [0, 0, 0],
            "{name}: the fragment reach is the phase's, not a block's — a non-zero one here \
             turns the per-block order check back on and every block applies twice"
        );

        // And the plan records it, in both directions: the halo the barrier
        // relieves, and the barrier itself.
        let phase = fragment_phase(op, grid.clone()).expect("a phase");
        assert!(phase.barrier, "{name}: the plan must record the barrier");
        assert_eq!(phase.halo, [0, 0, 0], "{name}: the halo is the reach");
        for block in &phase.blocks {
            assert_eq!(
                block.read.ranges(),
                block.core.ranges(),
                "{name}: with the edge stated as a barrier each block fetches its own core"
            );
        }
    }
}

/// **The order check costs one more pass over the fragment set**, and this is
/// the sentence of `barriers.md` §8.5 that did not survive being applied to
/// these two ops.
///
/// §8.5 prices the reversed-lattice check at one extra reduction, against the
/// one extra `apply` per block the same declaration already costs. That is the
/// right comparison for an op that had already declared `SeamFold::Unordered`;
/// neither of these had, so for them the check is a **new** cost rather than a
/// saving that got smaller — and it is not only the fold. `PhaseView` reads its
/// fragments out of the store, so the second reduction re-reads the whole set:
/// the shipped shape transmits it written once, read twice.
///
/// What is asserted here is that the cost is exactly that and no more — no pixel
/// traffic moves, the fragment traffic rises, and at a single block, which has
/// no order to reverse, the two arms are the same run.
#[test]
fn the_order_check_costs_one_more_pass_over_the_fragment_set() {
    let mask = mask();
    for block in blockings() {
        let blocks = BlockGrid::new(VOLUME, block).expect("a lattice").n_blocks();
        for (op, runner) in [
            (
                "ops::label",
                run_label as fn(&Array3<bool>, [usize; 3], Option<Arm>) -> Run,
            ),
            ("ops::detect", run_detect),
        ] {
            let unchecked = runner(&mask, block, Some(Arm::Hoisted));
            let checked = runner(&mask, block, Some(Arm::HoistedChecked));
            assert_eq!(
                (checked.traffic.read_bytes, checked.traffic.write_bytes),
                (unchecked.traffic.read_bytes, unchecked.traffic.write_bytes),
                "{op}, {block:?}: the order check reads no pixels and writes none"
            );
            if blocks == 1 {
                assert_eq!(
                    checked.traffic, unchecked.traffic,
                    "{op}, {block:?}: a single block has no order, so the check must be skipped"
                );
            } else {
                assert!(
                    checked.traffic.fragment_bytes > unchecked.traffic.fragment_bytes,
                    "{op}, {block:?}: the second reduction must re-read the fragment set — \
                     {} against {}",
                    checked.traffic.fragment_bytes,
                    unchecked.traffic.fragment_bytes
                );
                println!(
                    "{op:>12} | {blocks:>4} blocks | order check costs {:.3} MiB of fragments, \
                     {:.2}x the unchecked hoisted arm's",
                    mib(checked.traffic.fragment_bytes - unchecked.traffic.fragment_bytes),
                    checked.traffic.fragment_bytes as f64 / unchecked.traffic.fragment_bytes as f64
                );
            }
        }
    }
}

/// **A barrier alone is worth nothing at all to `ops::detect`**, and this is an
/// equality rather than a bound.
///
/// The whole traffic contribution of a barrier is that it relieves the halo, and
/// the halo costs pixel reads. `ops::detect`'s phase 1 declares
/// `reads_pixels() == false`, so the executor performs no pixel IO for it at any
/// halo and there is nothing for the relief to relieve. Everything this op pays
/// — the fragment set once per block, and the merge once per block — is in the
/// hoisting.
///
/// `ops::label` is the other side of the same statement and is asserted here
/// beside it: its phase 1 *does* read pixels, so a barrier moves its bytes.
#[test]
fn a_barrier_alone_is_worth_nothing_at_all_to_detect() {
    let mask = mask();
    for block in blockings() {
        let blocks = BlockGrid::new(VOLUME, block).expect("a lattice").n_blocks();
        let in_plan = run_detect(&mask, block, Some(Arm::InPlan));
        let alone = run_detect(&mask, block, Some(Arm::BarrierOnly));
        assert_eq!(
            in_plan.traffic, alone.traffic,
            "ops::detect, {block:?} ({blocks} block(s)): a barrier moved bytes it cannot move"
        );

        let in_plan = run_label(&mask, block, Some(Arm::InPlan));
        let alone = run_label(&mask, block, Some(Arm::BarrierOnly));
        if blocks == 1 {
            assert_eq!(
                in_plan.traffic, alone.traffic,
                "ops::label, {block:?}: one block has no halo to relieve"
            );
        } else {
            assert!(
                alone.traffic.read_bytes < in_plan.traffic.read_bytes,
                "ops::label, {block:?} ({blocks} block(s)): a barrier must relieve the pixel \
                 re-reads — {} against {}",
                alone.traffic.read_bytes,
                in_plan.traffic.read_bytes
            );
            assert_eq!(
                alone.traffic.fragment_bytes, in_plan.traffic.fragment_bytes,
                "ops::label, {block:?}: a barrier alone must not touch the fragment half"
            );
        }
    }
}

/// **The whole table, reported**, and the ordering asserted.
///
/// Printed rather than compared against a constant, because the constants are a
/// property of this volume and this fixture; what is asserted is the ordering
/// and the two equalities the design predicts.
#[test]
fn the_four_arms_are_tabulated_at_every_lattice() {
    let mask = mask();
    println!(
        "\nvolume {VOLUME:?}, {} set voxel(s)\n",
        mask.iter().filter(|&&on| on).count()
    );
    for (op, runner) in [
        (
            "ops::label",
            run_label as fn(&Array3<bool>, [usize; 3], Option<Arm>) -> Run,
        ),
        ("ops::detect", run_detect),
    ] {
        println!(
            "{op:>12} | {:>6} | {:<19} | {:>10} | {:>10} | {:>10} | {:>10} | {:>6} | {:>7}",
            "blocks", "arm", "read MiB", "write MiB", "frag MiB", "total MiB", "folds", "s"
        );
        for block in blockings() {
            let blocks = BlockGrid::new(VOLUME, block).expect("a lattice").n_blocks();
            let mut totals = Vec::new();
            for arm in ARMS {
                let run = runner(&mask, block, Some(arm));
                println!(
                    "{:>12} | {blocks:>6} | {:<19} | {:>10.3} | {:>10.3} | {:>10.3} | {:>10.3} | {:>6} | {:>7.2}",
                    "",
                    arm.name(),
                    mib(run.traffic.read_bytes),
                    mib(run.traffic.write_bytes),
                    mib(run.traffic.fragment_bytes),
                    mib(run.traffic.total()),
                    run.folds,
                    run.seconds
                );
                totals.push(run.traffic.total());
            }
            println!(
                "{:>12} | {blocks:>6} | {:<19} | in-plan {:.2}x, barrier alone {:.2}x, \
                 hoisted unchecked 1.00x, as shipped {:.2}x",
                "",
                "ratio",
                totals[0] as f64 / totals[2] as f64,
                totals[1] as f64 / totals[2] as f64,
                totals[3] as f64 / totals[2] as f64
            );
            assert!(
                totals[0] >= totals[1] && totals[1] >= totals[2],
                "{op}, {block:?}: the arms must not get more expensive as the declaration \
                 improves — {totals:?}"
            );
            if blocks > 1 {
                assert!(
                    totals[1] > totals[2],
                    "{op}, {block:?}: the hoisting must remove something at {blocks} blocks"
                );
                assert!(
                    totals[3] < totals[1],
                    "{op}, {block:?}: the order check must not cost back what the hoisting \
                     removed — {totals:?}"
                );
            }
        }
        println!();
    }
}
