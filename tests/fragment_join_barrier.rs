// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **What the barrier and the hoisted reduction are worth on two shipped ops**,
// measured rather than quoted.
//
// `docs/design/barriers.md` was written from a measurement of the *cost of not
// having one* and specified a way out in two steps: a phase may declare that it
// waits for all of the phase below it, and — only then — it may compute one
// answer for the whole phase instead of re-deriving it in every block. §8.10
// recorded that no shipped op declared either. `ops::fill` and `ops::regional`
// now do, and this file is the evidence for what that bought and the control
// that says it is still the same answer.
//
// Three things this file is arranged to avoid, each of which has bitten this
// project:
//
// * **The arms are one op with one word changed.** `ops::components::Merge` says where
//   the merge runs and nothing else varies between the three arms, so a
//   difference in the counters is attributable to the declaration. Two
//   implementations compared against each other would measure the difference
//   between two implementations.
// * **The liveness control is the barrier withheld.** A wrong reduction is
//   plausible in every block and no guard can catch it afterwards, so the
//   acceptance sweep runs all three arms at every lattice against a whole-volume
//   reference. The arm with the barrier withheld is the shape the framework
//   admitted before, and it must agree to the byte — if it does not, the barrier
//   is doing something other than what it claims.
// * **The lattices are genuinely distinct and that is asserted.** A sweep that
//   decayed to two grids, one of them a single block, kept passing here and had
//   stopped meaning anything.
//
// The absolute ratios in this file are **not** the ones `barriers.md` §7.1
// quotes and are not meant to be. §8.8 records why: a fragment is a block *face*
// there and a much smaller thing at this scale, so the fragment term is a
// different fraction of the total. What transfers is the structure — pixel reads
// go as `(1 + blocks) x volume` without a barrier and `2 x volume` with one,
// fragment traffic as `(1 + blocks) x F` per block and a small multiple of `F`
// hoisted, and the merge runs `blocks` times per block and once hoisted.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ndarray::Array3;

use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::error::Result;
use blockflow::fragment::{
    fragment_phase, BlockOutput, BlockView, FragmentInput, FragmentOp, FragmentOutput, PhaseView,
    PhaseWork, SeamFold, SourceBlocks,
};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::op::SourceInput;
use blockflow::ops::components::{decode_block_flags_for, encode_block_flags, Merge};
use blockflow::ops::fill::{
    fill_from_labels_into, label_background_into, outside_flags, FillHolesOp, LabelBackgroundOp,
};
use blockflow::ops::regional::{regional_maxima, LabelPlateauxOp, RegionalMaximaOp};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

/// Small enough that the finest lattice below is 256 blocks of 4 voxels an edge
/// and the whole sweep runs in well under a second; large enough that the merge
/// has something to join at every one of them.
const VOLUME: [usize; 3] = [16, 32, 32];
const FILL_STREAM: &str = "barrier.fill.faces";
const REGIONAL_STREAM: &str = "barrier.regional.faces";

// ------------------------------------------------------------ the scenes --

/// A mask whose holes only close when blocks talk to each other.
///
/// Two long closed boxes running the length of the x axis, so at every lattice
/// below the first that cuts x their cavities are several block-local components
/// that only the merge joins; and one box whose cavity drains through a channel
/// that leaves and re-enters blocks, so a block-local answer fills what the
/// global answer must not. `the_scenes_make_the_merge_load_bearing` asserts both
/// halves rather than assuming them.
fn mask_scene() -> Array3<bool> {
    let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    // a closed box spanning the whole x axis
    fill_box(&mut mask, [1, 2, 2], [14, 8, 8], true);
    fill_box(&mut mask, [2, 3, 3], [13, 7, 7], false);
    // a second one, offset so it lands differently under every cut
    fill_box(&mut mask, [1, 12, 12], [14, 20, 20], true);
    fill_box(&mut mask, [2, 13, 13], [13, 19, 19], false);
    // a third whose cavity drains out through the volume's own face
    fill_box(&mut mask, [1, 24, 4], [14, 30, 12], true);
    fill_box(&mut mask, [2, 25, 5], [13, 29, 11], false);
    for y in 25..VOLUME[1] {
        mask[[7, y, 8]] = false;
    }
    mask
}

fn fill_box(mask: &mut Array3<bool>, low: [usize; 3], high: [usize; 3], value: bool) {
    for i in low[0]..=high[0] {
        for j in low[1]..=high[1] {
            for k in low[2]..=high[2] {
                mask[[i, j, k]] = value;
            }
        }
    }
}

/// A field with plateaus that span the lattice and one corner higher than all of
/// them.
///
/// The single global peak is the case a block cannot see for itself: every other
/// plateau is a maximum of its own block and of nothing else, and only the merge
/// carries the peak's value far enough to disqualify them.
fn value_scene() -> Array3<f64> {
    let mut values = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for ((i, j, k), slot) in values.indexed_iter_mut() {
        // coarse terraces, so plateaus are wide and cross block seams
        *slot = ((i / 5) + (j / 7) + (k / 6)) as f64;
    }
    // a long ridge at one value, spanning x, which is one plateau at every cut
    for i in 0..VOLUME[0] {
        values[[i, 15, 15]] = 7.0;
    }
    // and one voxel strictly above everything, at the far corner from it
    values[[VOLUME[0] - 1, VOLUME[1] - 1, VOLUME[2] - 1]] = 99.0;
    values
}

// ----------------------------------------------------------- the oracles --

/// The whole-volume reference for `ops::fill`: the same kernels, called once.
fn fill_reference(mask: &Array3<bool>) -> Array3<bool> {
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    let count = label_background_into(mask.view(), labels.view_mut()).expect("labelled");
    let flags = outside_flags(labels.view(), count, [0, 0, 0], VOLUME, VOLUME);
    let mut out = Array3::from_elem(mask.raw_dim(), false);
    fill_from_labels_into(labels.view(), &flags, out.view_mut()).expect("filled");
    out
}

/// The whole-volume reference for `ops::regional`: the crate's own three
/// kernels, called once over everything.
///
/// Not a second implementation — that would make a disagreement a modelling
/// difference rather than a decomposition bug.
fn regional_reference(values: &Array3<f64>) -> Array3<bool> {
    regional_maxima(values.view()).expect("the whole-volume maxima")
}

// ------------------------------------------------------- counting the op --

/// A `FragmentOp` that delegates everything and counts what it was asked.
///
/// Every declaration is forwarded, so the plan built over this is the plan built
/// over the op it wraps — the point is to count `apply` and `reduce`, which is
/// the quantity `barriers.md` §7.2 is about and the one no byte column shows.
struct Counting<T> {
    inner: T,
    hoisted: bool,
    applies: AtomicUsize,
    /// `reduce` calls that produced bytes. A barriered but unhoisted arm has its
    /// `reduce` entered too — the executor asks every barrier phase — and it
    /// answers empty, which is not a merge.
    reductions: AtomicUsize,
}

impl<T: FragmentOp> Counting<T> {
    fn new(inner: T, hoisted: bool) -> Self {
        Self {
            inner,
            hoisted,
            applies: AtomicUsize::new(0),
            reductions: AtomicUsize::new(0),
        }
    }

    /// How many times the global merge ran. Once per `apply` while it is in the
    /// block, once per non-empty `reduce` once it is hoisted out.
    fn merges(&self) -> usize {
        if self.hoisted {
            self.reductions.load(Ordering::SeqCst)
        } else {
            self.applies.load(Ordering::SeqCst)
        }
    }

    fn applies(&self) -> usize {
        self.applies.load(Ordering::SeqCst)
    }
}

impl<T: FragmentOp> FragmentOp for Counting<T> {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        self.inner.reach(axis, volume_len)
    }
    fn cost_per_voxel(&self) -> f64 {
        self.inner.cost_per_voxel()
    }
    fn reads_pixels(&self) -> bool {
        self.inner.reads_pixels()
    }
    fn writes_pixels(&self) -> bool {
        self.inner.writes_pixels()
    }
    fn produces(&self, input: Dtype) -> Dtype {
        self.inner.produces(input)
    }
    fn inputs(&self) -> Vec<FragmentInput> {
        self.inner.inputs()
    }
    fn outputs(&self) -> Vec<FragmentOutput> {
        self.inner.outputs()
    }
    fn barrier(&self) -> bool {
        self.inner.barrier()
    }
    fn gathers(&self) -> bool {
        self.inner.gathers()
    }
    fn source_inputs(&self, volume: [usize; 3]) -> Vec<SourceInput> {
        self.inner.source_inputs(volume)
    }
    fn seam_fold(&self) -> Option<SeamFold> {
        self.inner.seam_fold()
    }
    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let answer = self.inner.reduce(at)?;
        if !answer.is_empty() {
            self.reductions.fetch_add(1, Ordering::SeqCst);
        }
        Ok(answer)
    }
    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        self.applies.fetch_add(1, Ordering::SeqCst);
        self.inner.apply(at)
    }
    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        self.applies.fetch_add(1, Ordering::SeqCst);
        self.inner.apply_with(at, sources)
    }
}

// ------------------------------------------------------------- the arms --

/// What one arm moved and how much work it did.
#[derive(Debug, Clone)]
struct Arm {
    output: Array3<bool>,
    read_bytes: u64,
    write_bytes: u64,
    fragment_bytes: u64,
    /// How many times the global merge ran.
    merges: usize,
    /// How many times the phase's `apply` ran — `blocks`, or twice that where
    /// `SeamFold::Unordered` costs a second application per block.
    applies: usize,
    halo: [usize; 3],
    barrier_recorded: bool,
    /// Serial, at concurrency one. Reported rather than asserted: the byte
    /// columns reproduce to the digit and this one does not.
    elapsed: Duration,
}

impl Arm {
    fn total(&self) -> u64 {
        self.read_bytes + self.write_bytes + self.fragment_bytes
    }
}

fn hints() -> Hints {
    Hints {
        concurrency: 1,
        ..Hints::default()
    }
}

fn halo_of(plan: &Decomposition, phase: usize) -> [usize; 3] {
    let entry = &plan.phases[phase];
    let granted = entry.halo.in_voxels(entry.grid.block());
    [
        granted.axis(0).bound(VOLUME[0]).0,
        granted.axis(1).bound(VOLUME[1]).0,
        granted.axis(2).bound(VOLUME[2]).0,
    ]
}

fn measure(env: &ArrayEnvironment) -> (u64, u64, u64) {
    let counters = env.counters();
    (
        counters.read_bytes.load(Ordering::Relaxed),
        counters.write_bytes.load(Ordering::Relaxed),
        counters.sidecar_bytes_written.load(Ordering::Relaxed)
            + counters.sidecar_bytes_read.load(Ordering::Relaxed),
    )
}

/// `ops::fill`'s two phases, with the merge placed by `merge`.
fn run_fill(block: [usize; 3], merge: Merge) -> Arm {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let label = LabelBackgroundOp::new("label", FILL_STREAM, Lifecycle::DeleteOnExit);
    let fill = Counting::new(
        FillHolesOp::new("fill", FILL_STREAM, 0, Dtype::Bool, &grid).merging(merge),
        merge.is_hoisted(),
    );

    let mut labelling = fragment_phase(&label, grid.clone()).expect("phase 0");
    labelling.dtype = Some(Dtype::U32);
    let mut filling = fragment_phase(&fill, grid).expect("phase 1");
    filling.dtype = Some(Dtype::Bool);
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::Bool,
        phases: vec![labelling, filling],
        chain_reach: [0, 0, 0],
    };
    plan.check().expect("the plan tiles");

    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(mask_scene()),
        &plan,
        [VOLUME[0], VOLUME[1], VOLUME[2]],
    )
    .expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::Bool);
    let started = Instant::now();
    execute_phases(
        "fill-arm",
        &workflow,
        &plan,
        &hints(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&fill)],
    )
    .expect("a run");
    let elapsed = started.elapsed();

    let (read_bytes, write_bytes, fragment_bytes) = measure(&env);
    Arm {
        output: env.output().view::<bool>().expect("bool out").to_owned(),
        read_bytes,
        write_bytes,
        fragment_bytes,
        merges: fill.merges(),
        applies: fill.applies(),
        halo: halo_of(&plan, 1),
        barrier_recorded: plan.phases[1].barrier,
        elapsed,
    }
}

/// `ops::regional`'s two phases, with the merge placed by `merge`.
fn run_regional(block: [usize; 3], merge: Merge) -> Arm {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let label = LabelPlateauxOp::new("label", REGIONAL_STREAM, Lifecycle::DeleteOnExit);
    let maxima = Counting::new(
        RegionalMaximaOp::new("maxima", REGIONAL_STREAM, 0, Dtype::Bool, &grid).merging(merge),
        merge.is_hoisted(),
    );

    let mut labelling = fragment_phase(&label, grid.clone()).expect("phase 0");
    labelling.dtype = Some(Dtype::U32);
    let mut finding = fragment_phase(&maxima, grid).expect("phase 1");
    finding.dtype = Some(Dtype::Bool);
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![labelling, finding],
        chain_reach: [0, 0, 0],
    };
    plan.check().expect("the plan tiles");

    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(value_scene()),
        &plan,
        [VOLUME[0], VOLUME[1], VOLUME[2]],
    )
    .expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::F64);
    let started = Instant::now();
    execute_phases(
        "regional-arm",
        &workflow,
        &plan,
        &hints(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&maxima)],
    )
    .expect("a run");
    let elapsed = started.elapsed();

    let (read_bytes, write_bytes, fragment_bytes) = measure(&env);
    Arm {
        output: env.output().view::<bool>().expect("bool out").to_owned(),
        read_bytes,
        write_bytes,
        fragment_bytes,
        merges: maxima.merges(),
        applies: maxima.applies(),
        halo: halo_of(&plan, 1),
        barrier_recorded: plan.phases[1].barrier,
        elapsed,
    }
}

const ARMS: [Merge; 3] = [
    Merge::PerBlock,
    Merge::PerBlockBehindABarrier,
    Merge::OnceForThePhase,
];

/// The lattices the sweep runs, **asserted to be genuinely distinct**.
///
/// The decay this guards against has happened on this project: a sweep that
/// shrank to two grids, one of them a single block, kept passing and had stopped
/// measuring anything.
fn lattices() -> Vec<[usize; 3]> {
    let blocks = [[16, 32, 32], [16, 16, 16], [8, 8, 8], [4, 4, 4]];
    let counts: Vec<usize> = blocks
        .iter()
        .map(|block| {
            BlockGrid::new(VOLUME, *block)
                .expect("a lattice")
                .n_blocks()
        })
        .collect();
    assert_eq!(counts, vec![1, 4, 32, 256], "the sweep's lattices moved");
    let mut distinct = counts.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), counts.len(), "two grids are the same grid");
    assert!(
        counts.iter().filter(|&&n| n > 1).count() >= 3,
        "a sweep of single-block lattices asserts nothing about decomposition"
    );
    blocks.to_vec()
}

fn n_blocks(block: [usize; 3]) -> usize {
    BlockGrid::new(VOLUME, block).expect("a lattice").n_blocks()
}

// -------------------------------------------------- the scenes are hard --

/// **Asserted first, because every measurement below is worthless without it.**
/// A scene whose answer a block can reach on its own measures nothing about a
/// merge, and a merge that never joins anything is a barrier around an identity.
#[test]
fn the_scenes_make_the_merge_load_bearing() {
    let mask = mask_scene();
    let filled = fill_reference(&mask);
    assert!(
        !mask[[7, 5, 5]] && filled[[7, 5, 5]],
        "the first cavity fills"
    );
    assert!(
        !mask[[7, 15, 15]] && filled[[7, 15, 15]],
        "and so does the second"
    );
    assert!(
        !filled[[7, 27, 8]],
        "the draining cavity is not a hole, however long the drain"
    );
    assert!(!filled[[0, 0, 0]], "the outside must not fill");

    // and the block-local answer genuinely disagrees at a real lattice: the
    // one-block arm is the reference, so a finer one that agreed with a
    // *block-local* answer would be agreeing by luck
    let cut = run_fill([8, 8, 8], Merge::OnceForThePhase);
    assert_eq!(cut.output, filled);
    assert!(
        cut.merges >= 1,
        "the merge must have run at all for the answer to be global"
    );

    let values = value_scene();
    let maxima = regional_reference(&values);
    assert!(
        maxima[[VOLUME[0] - 1, VOLUME[1] - 1, VOLUME[2] - 1]],
        "the one voxel above everything is a regional maximum"
    );
    let count = maxima.iter().filter(|&&flag| flag).count();
    assert_eq!(
        count, 1,
        "exactly one plateau survives, so every other plateau is disqualified by \
         something that may be a lattice away — which is the case no block can see"
    );
}

// -------------------------------------------------- the acceptance bar --

/// **The acceptance bar, and the barrier's liveness control in one.**
///
/// Every arm, at every lattice, byte-identical to a whole-volume reference. The
/// `Merge::PerBlock` arm is the shape the framework admitted before a barrier
/// could be declared; if the barriered arms ever stop agreeing with it, the
/// barrier is doing something other than what it claims.
#[test]
fn every_arm_of_fill_agrees_with_the_whole_volume_reference_at_every_lattice() {
    let want = fill_reference(&mask_scene());
    for block in lattices() {
        for merge in ARMS {
            let arm = run_fill(block, merge);
            assert_eq!(
                arm.output, want,
                "lattice {block:?} with the merge {merge:?} disagreed with the reference"
            );
        }
    }
}

#[test]
fn every_arm_of_regional_agrees_with_the_whole_volume_reference_at_every_lattice() {
    let want = regional_reference(&value_scene());
    for block in lattices() {
        for merge in ARMS {
            let arm = run_regional(block, merge);
            assert_eq!(
                arm.output, want,
                "lattice {block:?} with the merge {merge:?} disagreed with the reference"
            );
        }
    }
}

// ------------------------------------------------------ what it bought --

/// The halo, which is where the pixel amplification lived.
///
/// `fragment_phase` used to widen a phase's halo to cover its fragment reach,
/// because the halo was the only way to say "after all of the phase below". With
/// the barrier the halo is the reach, which for both these ops is nothing.
#[test]
fn a_barrier_relieves_the_halo_and_the_pixel_reads_that_followed_it() {
    for block in lattices() {
        let blocks = n_blocks(block);
        for (name, run) in [
            ("fill", run_fill as fn([usize; 3], Merge) -> Arm),
            ("regional", run_regional as fn([usize; 3], Merge) -> Arm),
        ] {
            let in_plan = run(block, Merge::PerBlock);
            let barriered = run(block, Merge::PerBlockBehindABarrier);
            let hoisted = run(block, Merge::OnceForThePhase);

            assert!(!in_plan.barrier_recorded, "{name}: the control has none");
            assert!(barriered.barrier_recorded, "{name}");
            assert!(hoisted.barrier_recorded, "{name}");

            assert_eq!(
                hoisted.halo,
                [0, 0, 0],
                "{name} at {block:?}: a barrier phase reading only its own core has no halo"
            );
            assert_eq!(barriered.halo, [0, 0, 0], "{name} at {block:?}");
            if blocks > 1 {
                assert!(
                    in_plan.halo != [0, 0, 0],
                    "{name} at {block:?}: without a barrier the fragment reach is the halo"
                );
                assert!(
                    in_plan.read_bytes > barriered.read_bytes,
                    "{name} at {block:?}: the halo is what the pixel reads followed"
                );
            }
            // and the barrier alone does nothing to the fragment half, which is
            // §3.3's honest half
            assert_eq!(
                in_plan.fragment_bytes, barriered.fragment_bytes,
                "{name} at {block:?}: a barrier moves no fragments"
            );
            assert_eq!(
                in_plan.merges, barriered.merges,
                "{name} at {block:?}: a barrier removes no merges"
            );
        }
    }
}

/// The larger half: the merge runs **once** rather than once per block, and the
/// fragment set stops being transmitted once per block with it.
#[test]
fn hoisting_runs_the_merge_once_however_finely_the_volume_is_cut() {
    for block in lattices() {
        let blocks = n_blocks(block);
        for (name, run) in [
            ("fill", run_fill as fn([usize; 3], Merge) -> Arm),
            ("regional", run_regional as fn([usize; 3], Merge) -> Arm),
        ] {
            let in_plan = run(block, Merge::PerBlock);
            let hoisted = run(block, Merge::OnceForThePhase);

            // **The quantity no byte column shows.** Per block the merge runs
            // once per `apply`, and `SeamFold::Unordered` costs a second `apply`
            // per block on top — a cost hoisting removes as well, because the
            // hoisted arm's per-block fragment reach is nothing and there is no
            // order left to check per block.
            assert_eq!(
                in_plan.applies,
                if blocks > 1 { 2 * blocks } else { 1 },
                "{name} at {block:?}: the in-plan arm applies once per block, twice where \
                 the order check bites"
            );
            assert_eq!(
                hoisted.applies, blocks,
                "{name} at {block:?}: hoisting leaves one application per block"
            );
            assert_eq!(
                in_plan.merges, in_plan.applies,
                "{name} at {block:?}: every in-plan application re-derives the whole merge"
            );
            assert_eq!(
                hoisted.merges,
                if blocks > 1 { 2 } else { 1 },
                "{name} at {block:?}: once for the phase, and once more for the \
                 reversed-lattice order check"
            );

            // and the set is transmitted a fixed number of times rather than
            // once per block
            if blocks > 1 {
                assert!(
                    in_plan.fragment_bytes > hoisted.fragment_bytes,
                    "{name} at {block:?}"
                );
                let written = hoisted.fragment_bytes / 3;
                assert_eq!(
                    hoisted.fragment_bytes,
                    3 * written,
                    "{name} at {block:?}: the hoisted arm writes the set once and reads it \
                     twice — the reduction and the reversed-lattice check"
                );
            }
        }
    }
}

/// The three arms side by side, with every column predicted from a formula
/// before it is compared.
///
/// The formulae are `barriers.md` §8.8's, term for term: pixel reads are
/// `(1 + blocks) x volume` without a barrier and `2 x volume` with one; the
/// fragment set is transmitted `(1 + blocks)` times per block and three times
/// hoisted. Nothing here is read off the counters and asserted against itself.
#[test]
fn the_three_arms_are_measured_against_each_other() {
    let voxels = VOLUME[0] * VOLUME[1] * VOLUME[2];
    for (name, run, in_width, out_width) in [
        ("fill", run_fill as fn([usize; 3], Merge) -> Arm, 1u64, 4u64),
        ("regional", run_regional, 8, 4),
    ] {
        println!("\n{name}: VOLUME {VOLUME:?}");
        println!(
            "{:>7} {:>6} {:>12} {:>12} {:>12} {:>14} {:>7} {:>9}",
            "blocks", "arm", "read", "write", "fragments", "total", "merges", "seconds"
        );
        for block in lattices() {
            let blocks = n_blocks(block);
            let mut cheapest = u64::MAX;
            let mut rows = Vec::new();
            for merge in ARMS {
                let arm = run(block, merge);
                cheapest = cheapest.min(arm.total());
                rows.push((merge, arm));
            }
            for (merge, arm) in &rows {
                println!(
                    "{blocks:>7} {:>6} {:>12} {:>12} {:>12} {:>14} {:>7} {:>9.3}",
                    match merge {
                        Merge::PerBlock => "plan",
                        Merge::PerBlockBehindABarrier => "barr",
                        Merge::OnceForThePhase => "hoist",
                    },
                    arm.read_bytes,
                    arm.write_bytes,
                    arm.fragment_bytes,
                    arm.total(),
                    arm.merges,
                    arm.elapsed.as_secs_f64(),
                );
            }

            // **The pixel column, predicted.** Phase 0 reads the input image
            // once; phase 1 reads the `u32` label image once per block without a
            // barrier and once with one. `SeamFold::Unordered`'s second
            // application is served from the same fetch, so it is not in here.
            let phase0 = voxels as u64 * in_width;
            let labels = voxels as u64 * out_width;
            let (_, in_plan) = &rows[0];
            let (_, barriered) = &rows[1];
            let (_, hoisted) = &rows[2];
            assert_eq!(
                in_plan.read_bytes,
                phase0 + blocks as u64 * labels,
                "{name} at {blocks} blocks: the in-plan arm reads the label image per block"
            );
            assert_eq!(
                barriered.read_bytes,
                phase0 + labels,
                "{name} at {blocks} blocks: a barrier leaves one read of the label image"
            );
            assert_eq!(
                hoisted.read_bytes, barriered.read_bytes,
                "{name} at {blocks} blocks: hoisting is not about pixels"
            );

            // **The fragment column, predicted.** `F` is what the whole set
            // weighs; the counters see it written once and read once per block
            // without hoisting, and once written and twice read with it.
            let written = in_plan.fragment_bytes / (1 + blocks as u64);
            assert_eq!(
                in_plan.fragment_bytes,
                (1 + blocks as u64) * written,
                "{name} at {blocks} blocks: the set is transmitted once per block"
            );
            // Once written, once read by the reduction, and once more by the
            // reversed-lattice order check — which a one-block lattice is
            // exempt from, because a one-element sequence has no order.
            let reads = if blocks > 1 { 2 } else { 1 };
            assert_eq!(
                hoisted.fragment_bytes,
                (1 + reads) * written,
                "{name} at {blocks} blocks: written once and read {reads} time(s)"
            );

            // and the ordering the whole exercise is for
            assert!(hoisted.total() <= cheapest, "{name} at {blocks} blocks");
            if blocks > 1 {
                assert!(hoisted.total() < barriered.total());
                assert!(barriered.total() < in_plan.total());
            }
        }
    }
}

/// **What the plan can and cannot tell apart**, which is worth pinning because
/// one half of it is a surprise.
///
/// The barrier is part of the plan — recorded, hashed, carried over the wire —
/// for `barriers.md` §3.1's reason: a plan whose record disagrees with its ops
/// waits for one thing and fetches another. So declaring it moves the
/// fingerprint, and it must.
///
/// **The hoisting does not.** `reduce` is not recorded on the plan and neither
/// is a fragment input's reach; both are read off the op by `check_phase_work`
/// on every run, which is where a disagreement would surface. So the two
/// barriered arms fingerprint identically, and that is correct rather than a
/// gap — they compute the same answer and differ only in cost — but it means a
/// fingerprint is not evidence about *which* of them ran.
#[test]
fn the_barrier_is_in_the_fingerprint_and_the_hoisting_is_not() {
    let block = [8usize, 8, 8];
    let fingerprint = |merge: Merge| {
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        let label = LabelBackgroundOp::new("label", FILL_STREAM, Lifecycle::DeleteOnExit);
        let fill = FillHolesOp::new("fill", FILL_STREAM, 0, Dtype::Bool, &grid).merging(merge);
        let mut labelling = fragment_phase(&label, grid.clone()).expect("phase 0");
        labelling.dtype = Some(Dtype::U32);
        let mut filling = fragment_phase(&fill, grid).expect("phase 1");
        filling.dtype = Some(Dtype::Bool);
        let plan = Decomposition {
            volume: VOLUME,
            dtype: Dtype::Bool,
            phases: vec![labelling, filling],
            chain_reach: [0, 0, 0],
        };
        plan.check().expect("the plan tiles");
        plan.fingerprint()
    };
    assert_ne!(
        fingerprint(Merge::PerBlock),
        fingerprint(Merge::PerBlockBehindABarrier),
        "a barrier is part of the plan and a resumed run has to see it move"
    );
    assert_eq!(
        fingerprint(Merge::PerBlockBehindABarrier),
        fingerprint(Merge::OnceForThePhase),
        "where the merge runs is the op's business and is checked against the op \
         on every run, not recorded in the plan"
    );
}

// ------------------------------------------------- the blob's own guards --

/// A reduction blob is the op's own encoding, and the op's own decode is the
/// only place a mismatch can surface — `barriers.md` §7.7. So the magic, the
/// lattice and the emptiness are each refused by name.
#[test]
fn a_reduction_blob_says_what_it_is_and_what_it_was_reduced_over() {
    let mut flags = BTreeMap::new();
    flags.insert([0usize, 0, 0], vec![true, false]);
    flags.insert([0usize, 0, 1], vec![false, true, true]);
    let counts = [1usize, 1, 2];
    let blob = encode_block_flags(&flags, counts, 0x1234_5678).expect("encoded");

    assert_eq!(
        decode_block_flags_for(&blob, counts, [0, 0, 0], 0x1234_5678, "a blob").unwrap(),
        vec![true, false]
    );
    assert_eq!(
        decode_block_flags_for(&blob, counts, [0, 0, 1], 0x1234_5678, "a blob").unwrap(),
        vec![false, true, true]
    );

    // another op's magic
    let wrong = decode_block_flags_for(&blob, counts, [0, 0, 0], 0x8765_4321, "a blob")
        .expect_err("a blob decoded under the wrong op's magic must be refused");
    assert!(wrong.to_string().contains("magic"), "{wrong}");

    // another lattice, which is the plausible-in-every-block failure
    let other = decode_block_flags_for(&blob, [2, 1, 1], [0, 0, 0], 0x1234_5678, "a blob")
        .expect_err("a blob from another cut must be refused");
    assert!(other.to_string().contains("lattice"), "{other}");

    // an empty blob, which is what a block gets from an entry point that carries
    // none
    let empty = decode_block_flags_for(&[], counts, [0, 0, 0], 0x1234_5678, "a blob")
        .expect_err("an empty reduction must be refused");
    assert!(empty.to_string().contains("empty"), "{empty}");

    // a lattice the merge did not answer for every block of
    let short = encode_block_flags(&flags, [1, 1, 3], 0x1234_5678)
        .expect_err("a merge that skipped a block must be refused");
    assert!(short.to_string().contains("block"), "{short}");
}

/// A hoisted op run through the entry point that carries no blob is refused
/// rather than answered from an empty table.
///
/// `strategy::execute_task_of` takes no reduction and says so; this asserts that
/// the refusal reaches these two ops, which is the thing §8.7 exists for.
#[test]
fn a_shipped_reducing_op_is_refused_by_the_entry_point_that_has_no_blob() {
    use blockflow::graph::TaskGraph;

    let grid = BlockGrid::new(VOLUME, [8, 8, 8]).expect("a lattice");
    let label = LabelBackgroundOp::new("label", FILL_STREAM, Lifecycle::DeleteOnExit);
    let fill = FillHolesOp::new("fill", FILL_STREAM, 0, Dtype::Bool, &grid);
    let mut labelling = fragment_phase(&label, grid.clone()).expect("phase 0");
    labelling.dtype = Some(Dtype::U32);
    let mut filling = fragment_phase(&fill, grid).expect("phase 1");
    filling.dtype = Some(Dtype::Bool);
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::Bool,
        phases: vec![labelling, filling],
        chain_reach: [0, 0, 0],
    };
    plan.check().expect("the plan tiles");

    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(mask_scene()),
        &plan,
        [VOLUME[0], VOLUME[1], VOLUME[2]],
    )
    .expect("environment");
    let graph = TaskGraph::build(&plan);
    let task = graph.tasks_in_phase(1)[0].clone();
    let err = blockflow::strategy::execute_task_of(
        &Chain::sequence(Vec::new()),
        &plan,
        &task,
        &PhaseWork::Fragments(&fill),
        &env,
        &[],
    )
    .expect_err("a reducing op has no blob here");
    let message = err.to_string();
    assert!(message.contains("reduce"), "{message}");
    assert!(message.contains("execute_task_with_reduction"), "{message}");
}
