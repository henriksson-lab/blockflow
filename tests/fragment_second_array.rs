// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A `FragmentOp` reading a **second array**, and the seam that comes with it.
//
// What this is for
// ----------------
// `FragmentOp` read the one image its phase was handed. The shape that did not
// fit was the useful one: summarise a *label* array against an *intensity*
// array, per region, where the regions are not the blocks and a region may be
// cut in half by a block seam. `BlockOp` already had the mechanism —
// `source_inputs` plus `apply_with`, with the default `apply_with` **erroring**
// rather than dropping an operand — so `FragmentOp` takes the same one.
//
// The hazard, which is the content
// --------------------------------
// A per-region sum straddling a seam is computed in pieces and the pieces are
// added in a merge. In `f64` that addition does not associate, so the total
// depends on the order the blocks merged in and therefore on how the volume was
// cut. `ops/detect.rs` deferred a weighted centroid on exactly this.
//
// This crate does not solve it. It removes the silence:
//
// * `SeamFold` has **no default**, and an op that reads a second array — or that
//   folds fragments carrying a second array's values over a non-zero reach — is
//   refused by name until it says which of the three it is;
// * `SeamFold::Unordered` is **checked**: the executor applies the block a second
//   time with the neighbourhood reversed and requires the same bytes;
// * `SeamFold::OrderDependent` is allowed and says what it costs.
//
// The four assertions the change is accepted on are the four sections below.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::{
    append_fragment_phase, BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp,
    FragmentOutput, PhaseWork, SeamFold, SourceBlocks,
};
use blockflow::geometry::BlockGrid;
use blockflow::log::{Event, ExecutionLog};
use blockflow::op::{Chain, SourceInput};
use blockflow::probes::{
    region_of, BlockSummaryOp, DriftingSumOp, IdentityOp, NeighbourFoldOp, RegionMergeOp,
    RegionSumOp,
};
use blockflow::reach::Reach;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use ndarray::Array3;

// ------------------------------------------------------------- fixtures --

const VOLUME: [usize; 3] = [12, 4, 4];
/// Region edge. Coprime with every block edge tried below, so a region boundary
/// at 5 and 10 never lines up with a block boundary at 2, 3, 4 or 6 — which is
/// the point: the sums being merged are sums of *pieces*.
const REGION: usize = 5;

fn ramp(volume: [usize; 3]) -> blockflow::Voxels {
    let mut array = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));
    for (flat, value) in array.iter_mut().enumerate() {
        *value = flat as f64;
    }
    array.into()
}

fn identity_workflow(volume: [usize; 3]) -> Workflow {
    Workflow::new(
        Chain::op(IdentityOp::new("identity", [0, 0, 0])),
        volume,
        Dtype::F64,
    )
}

/// One pixel phase over `block`, as the image the fragment phases are appended
/// after. Built by hand so the block count is stated rather than chosen.
fn one_pixel_phase(volume: [usize; 3], block: [usize; 3]) -> Decomposition {
    Decomposition {
        volume,
        dtype: Dtype::F64,
        phases: vec![PhaseDecomposition::derive(
            vec![0],
            vec!["identity".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            BlockGrid::new(volume, block).expect("a lattice"),
        )],
        chain_reach: [0, 0, 0],
    }
}

/// What the per-region sums *are*, computed over the whole volume with no
/// blocking anywhere. The oracle the decomposed answers are compared against, so
/// that "every block size agrees" is not merely "every block size is wrong the
/// same way".
fn reference_totals(volume: [usize; 3]) -> BTreeMap<u64, (u64, u64)> {
    let mut totals: BTreeMap<u64, (u64, u64)> = BTreeMap::new();
    let mut flat = 0u64;
    for i in 0..volume[0] {
        for j in 0..volume[1] {
            for k in 0..volume[2] {
                let entry = totals
                    .entry(region_of([i, j, k], REGION, volume))
                    .or_insert((0, 0));
                entry.0 += 1;
                entry.1 += flat;
                flat += 1;
            }
        }
    }
    totals
}

// ------------------------------------------- 1. additive: nothing moved --

/// The fingerprints are pinned, and they are the values the crate produced
/// **before** `source_inputs` existed on `FragmentOp`.
///
/// `Decomposition::fingerprint` hashes `source_images` only when the list is
/// non-empty, so a plan whose ops declare no second array hashes exactly as it
/// did — but that is an argument, and the argument is worth a number. These two
/// were read off the pre-change library.
const ONE_FRAGMENT_PHASE: u64 = 9371778399568127708;
const TWO_FRAGMENT_PHASES: u64 = 7876609532266632336;

#[test]
fn a_fragment_plan_that_reads_no_second_array_is_the_plan_it_always_was() {
    let volume = [32, 4, 4];
    let base = one_pixel_phase(volume, [8, 4, 4]);
    let summary = BlockSummaryOp::new("summary", "blocks", Lifecycle::DeleteOnExit);
    let plan = append_fragment_phase(base, &summary).expect("a fragment phase");
    assert_eq!(
        plan.fingerprint(),
        ONE_FRAGMENT_PHASE,
        "adding declared source inputs to `FragmentOp` moved a plan that declares none"
    );
    let fold = NeighbourFoldOp::new(
        "fold",
        "blocks",
        1,
        [1, 0, 0],
        "folded",
        Lifecycle::DeleteOnExit,
    );
    let plan = append_fragment_phase(plan, &fold).expect("a second fragment phase");
    assert_eq!(plan.fingerprint(), TWO_FRAGMENT_PHASES);

    // And the mechanism behind the numbers, stated so a later reader can see
    // why they did not move: an op that says nothing records nothing.
    for phase in &plan.phases {
        assert!(
            phase.source_images.is_empty(),
            "a phase whose op declares no second array recorded one"
        );
    }
}

// ------------------------------ 2. decomposition invariance, over a seam --

/// The three-phase plan the invariance test runs: identity pixels, then per-
/// region partials read out of **image 0**, then the merge.
///
/// The partial phase declares `reads_pixels() == false`, so the array it reads
/// is genuinely a second one — it never touches the image it was handed.
fn region_sum_plan(block: [usize; 3]) -> (Decomposition, RegionSumOp, RegionMergeOp) {
    let base = one_pixel_phase(VOLUME, block);
    let lattice = base.phases[0].grid.blocks_per_axis();
    let partials = RegionSumOp::new("partials", 0, REGION, "partials", Lifecycle::DeleteOnExit);
    let merge = RegionMergeOp::new(
        "merge",
        "partials",
        1,
        lattice,
        "totals",
        Lifecycle::DeleteOnExit,
    );
    let plan = append_fragment_phase(base, &partials).expect("a partial phase");
    let plan = append_fragment_phase(plan, &merge).expect("a merge phase");
    (plan, partials, merge)
}

/// Run the plan at `block` and return the merged fragment's bytes, the raw
/// per-block partials, and the event log.
fn run_at(block: [usize; 3]) -> (Vec<u8>, BTreeMap<[usize; 3], Vec<u8>>, Arc<ExecutionLog>) {
    let (plan, partials, merge) = region_sum_plan(block);
    let env =
        ArrayEnvironment::new(ramp(VOLUME), plan.n_phases(), [4, 4, 4]).expect("an environment");
    let log = Arc::new(ExecutionLog::new());
    execute_phases(
        "region-sums",
        &identity_workflow(VOLUME),
        &plan,
        &Hints::default(),
        &env,
        &[log.clone()],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&partials),
            PhaseWork::Fragments(&merge),
        ],
    )
    .expect("a run");

    let mut per_block = BTreeMap::new();
    for index in plan.phases[1]
        .grid
        .cores()
        .iter()
        .enumerate()
        .map(|(n, _)| {
            let counts = plan.phases[1].grid.blocks_per_axis();
            [
                n / (counts[1] * counts[2]),
                (n / counts[2]) % counts[1],
                n % counts[2],
            ]
        })
    {
        if let Some(bytes) = env.read_sidecar("partials", 1, index).expect("a read") {
            per_block.insert(index, bytes);
        }
    }
    let totals = env
        .read_sidecar("totals", 2, [0, 0, 0])
        .expect("a read")
        .expect("block [0,0,0] wrote the merged totals");
    (totals, per_block, log)
}

#[test]
fn a_per_region_sum_over_a_second_array_is_byte_identical_across_block_sizes() {
    let sizes: [[usize; 3]; 5] = [[12, 4, 4], [6, 4, 4], [4, 4, 4], [3, 4, 4], [2, 4, 4]];
    let mut answers: Vec<(String, Vec<u8>)> = Vec::new();
    for block in sizes {
        let (totals, per_block, _) = run_at(block);

        // The regions really are cut. Without this the byte-identity below
        // would be the identity of five runs that never merged anything.
        let mut contributors: BTreeMap<u64, BTreeSet<[usize; 3]>> = BTreeMap::new();
        for (index, bytes) in &per_block {
            for (region, _, _) in RegionSumOp::read(bytes).expect("a partial") {
                contributors.entry(region).or_default().insert(*index);
            }
        }
        let straddled = contributors.values().filter(|who| who.len() > 1).count();
        if block != [12, 4, 4] {
            assert!(
                straddled > 0,
                "block {block:?} put no region across a seam, so this run merges nothing"
            );
        }

        // And the answer is the whole-volume one, not merely a stable one.
        let merged: BTreeMap<u64, (u64, u64)> = RegionMergeOp::read(&totals)
            .expect("the merged totals")
            .into_iter()
            .map(|(region, voxels, sum)| (region, (voxels, sum)))
            .collect();
        assert_eq!(
            merged,
            reference_totals(VOLUME),
            "block {block:?} did not reproduce the unblocked per-region sums"
        );
        answers.push((format!("{block:?}"), totals));
    }

    let (first_name, first) = &answers[0];
    for (name, bytes) in &answers[1..] {
        assert_eq!(
            bytes, first,
            "the merged fragment differs between block {first_name} and block {name}; a \
             per-region reduction over a second array must not be a function of the plan"
        );
    }
}

#[test]
fn the_second_array_is_read_and_counted_as_a_read_of_that_image() {
    // Phase 0 reads image 0 once per block, phase 1 reads image 0 once per
    // block *as its operand*, and nothing reads image 1 at all — phase 1
    // declares `reads_pixels() == false` and phase 2 is fragments to fragments.
    // A second array that cost nothing in the counters would make every
    // measurement of this feature a measurement of the wrong plan.
    let block = [3, 4, 4];
    let (_, _, log) = run_at(block);
    let mut per_image: BTreeMap<usize, usize> = BTreeMap::new();
    for event in log.events() {
        if let Event::RegionRead { image, .. } = event {
            *per_image.entry(image).or_default() += 1;
        }
    }
    let blocks = VOLUME[0] / block[0];
    assert_eq!(
        per_image.get(&0).copied().unwrap_or(0),
        2 * blocks,
        "image 0 is read once per block by the pixel phase and once per block as the operand"
    );
    assert_eq!(
        per_image.get(&1).copied().unwrap_or(0),
        0,
        "nothing reads the image the fragment phase was handed"
    );

    // The plan says so too, which is what makes the image survive to be read.
    let (plan, _, _) = region_sum_plan(block);
    assert_eq!(plan.phases[1].source_images, vec![0]);
    assert_eq!(plan.phases[0].source_images, Vec::<usize>::new());
    assert_eq!(plan.phases[2].source_images, Vec::<usize>::new());
}

// -------------------------- 3. the silent operand is not expressible --

/// Declares a second array and never overrides `apply_with`. The whole of the
/// defect: it would emit a fragment summarising one array while claiming to
/// summarise two.
struct ForgetfulOp;

impl FragmentOp for ForgetfulOp {
    fn name(&self) -> &'static str {
        "forgetful"
    }

    /// Nothing crosses on this op's own account. The second array it declares
    /// is fetched over the reach on its `SourceInput`, which is a separate
    /// declaration — and the one this test exists to see refused.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(0)]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            "forgotten",
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, _at: &BlockView<'_>) -> blockflow::error::Result<BlockOutput> {
        Ok(BlockOutput::fragment("forgotten", Vec::new()))
    }
}

#[test]
fn an_op_that_declares_a_second_array_and_takes_the_default_is_refused_by_name() {
    let op = ForgetfulOp;
    let plan = append_fragment_phase(one_pixel_phase(VOLUME, [4, 4, 4]), &op).expect("a plan");
    let env = ArrayEnvironment::new(ramp(VOLUME), plan.n_phases(), [4, 4, 4]).expect("an env");
    let failed = execute_phases(
        "forgetful",
        &identity_workflow(VOLUME),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .expect_err("an op that drops its operand must not run")
    .to_string();
    assert!(failed.contains("forgetful"), "{failed}");
    assert!(failed.contains("apply_with"), "{failed}");
    assert!(failed.contains("image(s) 0"), "{failed}");
}

// -------------------------------- 4. the order-dependence hazard --

/// The `f64` fold, at two block sizes, with values chosen so that the two
/// bracketings genuinely differ: `1 + 1 + 2^60 - 2^60` is `0` added left to
/// right one voxel at a time and `2` when the pairs are summed first.
fn drifting_plan(
    block: [usize; 3],
    fold: SeamFold,
) -> (Decomposition, BlockSummaryOp, DriftingSumOp) {
    let volume = [4, 1, 1];
    let base = one_pixel_phase(volume, block);
    let lattice = base.phases[0].grid.blocks_per_axis();
    let summary = BlockSummaryOp::new("summary", "blocks", Lifecycle::DeleteOnExit);
    let drift = DriftingSumOp::new(
        "drift",
        "blocks",
        1,
        lattice,
        "total",
        Lifecycle::DeleteOnExit,
        fold,
    );
    let plan = append_fragment_phase(base, &summary).expect("a summary phase");
    let plan = append_fragment_phase(plan, &drift).expect("a fold phase");
    (plan, summary, drift)
}

fn wide_magnitudes() -> blockflow::Voxels {
    let big = (1u64 << 60) as f64;
    Array3::from_shape_vec((4, 1, 1), vec![1.0, 1.0, big, -big])
        .expect("four voxels")
        .into()
}

fn run_drift(block: [usize; 3], fold: SeamFold) -> blockflow::error::Result<f64> {
    let (plan, summary, drift) = drifting_plan(block, fold);
    let env = ArrayEnvironment::new(wide_magnitudes(), plan.n_phases(), [1, 1, 1])
        .expect("an environment");
    execute_phases(
        "drift",
        &identity_workflow([4, 1, 1]),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&summary),
            PhaseWork::Fragments(&drift),
        ],
    )?;
    let bytes = env
        .read_sidecar("total", 2, [0, 0, 0])?
        .expect("block [0,0,0] wrote the total");
    DriftingSumOp::read(&bytes)
}

#[test]
fn an_f64_fold_claiming_to_be_unordered_is_caught_by_the_executor() {
    let failed = run_drift([1, 1, 1], SeamFold::Unordered)
        .expect_err("an `f64` sum over four fragments is not order-independent")
        .to_string();
    assert!(failed.contains("drift"), "{failed}");
    assert!(failed.contains("SeamFold::Unordered"), "{failed}");
    assert!(failed.contains("opposite order"), "{failed}");
    // And it says what to do instead, in both directions.
    assert!(failed.contains("fixed-point"), "{failed}");
    assert!(failed.contains("SeamFold::OrderDependent"), "{failed}");
}

#[test]
fn the_same_fold_declared_order_dependent_runs_and_is_not_decomposition_invariant() {
    // This is what the declaration buys and what it forbids: the run proceeds,
    // and the answer is a property of the plan rather than of the data. Nobody
    // may compare two of these across decompositions, and here is why.
    let one_voxel_blocks = run_drift([1, 1, 1], SeamFold::OrderDependent).expect("a run");
    let two_voxel_blocks = run_drift([2, 1, 1], SeamFold::OrderDependent).expect("a run");
    assert_eq!(one_voxel_blocks, 0.0);
    assert_eq!(two_voxel_blocks, 2.0);
    assert_ne!(
        one_voxel_blocks, two_voxel_blocks,
        "the probe was supposed to demonstrate the hazard and did not"
    );
}

/// Reads a stream that carries a second array's values, folds it over a
/// non-zero reach, and says nothing about the seam.
struct SilentMergeOp;

impl FragmentOp for SilentMergeOp {
    fn name(&self) -> &'static str {
        "silent-merge"
    }

    /// Nothing crosses as pixels. The block this op folds across arrives on the
    /// `partials` stream over the **fragment** reach in `inputs`, which is the
    /// distinction the phase reach must not absorb.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own("partials", 1).with_reach([1, 0, 0])]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            "merged",
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, _at: &BlockView<'_>) -> blockflow::error::Result<BlockOutput> {
        Ok(BlockOutput::fragment("merged", Vec::new()))
    }
}

#[test]
fn the_obligation_to_state_a_seam_fold_follows_the_operand_downstream() {
    // `SilentMergeOp` declares no source input of its own. What makes it owe an
    // answer is that it folds a stream written by an op that does — the op that
    // reads the array is not in general the op that adds across the seam.
    let block = [4, 4, 4];
    let base = one_pixel_phase(VOLUME, block);
    let partials = RegionSumOp::new("partials", 0, REGION, "partials", Lifecycle::DeleteOnExit);
    let silent = SilentMergeOp;
    let plan = append_fragment_phase(base, &partials).expect("a partial phase");
    let plan = append_fragment_phase(plan, &silent).expect("a merge phase");
    let env = ArrayEnvironment::new(ramp(VOLUME), plan.n_phases(), [4, 4, 4]).expect("an env");
    let failed = execute_phases(
        "silent",
        &identity_workflow(VOLUME),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&partials),
            PhaseWork::Fragments(&silent),
        ],
    )
    .expect_err("a silent fold over a second array's values must not run")
    .to_string();
    assert!(failed.contains("silent-merge"), "{failed}");
    assert!(failed.contains("straddle a seam"), "{failed}");
    assert!(failed.contains("SeamFold::Unordered"), "{failed}");
}

/// Claims nothing crosses a seam, and reaches a block for it.
struct ContradictoryOp;

impl FragmentOp for ContradictoryOp {
    fn name(&self) -> &'static str {
        "contradictory"
    }

    /// Nothing crosses as pixels; the block it reaches for is a **fragment**
    /// reach, declared on the stream in `inputs`.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own("partials", 1).with_reach([1, 0, 0])]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            "merged",
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, _at: &BlockView<'_>) -> blockflow::error::Result<BlockOutput> {
        Ok(BlockOutput::fragment("merged", Vec::new()))
    }
}

#[test]
fn per_block_is_checked_against_the_reach_rather_than_believed() {
    let block = [4, 4, 4];
    let base = one_pixel_phase(VOLUME, block);
    let partials = RegionSumOp::new("partials", 0, REGION, "partials", Lifecycle::DeleteOnExit);
    let op = ContradictoryOp;
    let plan = append_fragment_phase(base, &partials).expect("a partial phase");
    let plan = append_fragment_phase(plan, &op).expect("a merge phase");
    let env = ArrayEnvironment::new(ramp(VOLUME), plan.n_phases(), [4, 4, 4]).expect("an env");
    let failed = execute_phases(
        "contradictory",
        &identity_workflow(VOLUME),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&partials),
            PhaseWork::Fragments(&op),
        ],
    )
    .expect_err("`PerBlock` and a non-zero reach cannot both be true")
    .to_string();
    assert!(failed.contains("contradictory"), "{failed}");
    assert!(failed.contains("PerBlock"), "{failed}");
}

// ----------------------------------------- the equal-reach limit binds --

/// Wants its operand over a one-voxel window, which is more than a zero halo
/// fetches.
struct WindowedOperandOp;

impl FragmentOp for WindowedOperandOp {
    fn name(&self) -> &'static str {
        "windowed-operand"
    }

    /// Nothing crosses on the phase's own account. The one-voxel window belongs
    /// to the operand and is declared on the `SourceInput`; the test below
    /// asserts exactly that separation — the halo widens to `[1, 1, 1]` while
    /// the phase reach stays zero.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(0, Reach::symmetric([1, 1, 1]))]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            "windowed",
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, _at: &BlockView<'_>) -> blockflow::error::Result<BlockOutput> {
        unreachable!("applied through `apply_with`")
    }

    fn apply_with(
        &self,
        _at: &BlockView<'_>,
        _sources: SourceBlocks<'_>,
    ) -> blockflow::error::Result<BlockOutput> {
        Ok(BlockOutput::fragment("windowed", Vec::new()))
    }
}

#[test]
fn an_operand_reaching_past_the_phase_halo_is_refused_rather_than_planned() {
    let op = WindowedOperandOp;

    // Built by `fragment_phase`, the reach is folded into the halo and the
    // phase fetches what the operand walks.
    let planned = append_fragment_phase(one_pixel_phase(VOLUME, [4, 4, 4]), &op).expect("a plan");
    assert_eq!(planned.phases[1].halo, [1, 1, 1]);
    assert_eq!(
        planned.phases[1].reach,
        [0, 0, 0],
        "the halo widened, not the trust"
    );
    let env = ArrayEnvironment::new(ramp(VOLUME), planned.n_phases(), [4, 4, 4]).expect("an env");
    execute_phases(
        "windowed",
        &identity_workflow(VOLUME),
        &planned,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .expect("a phase built by `fragment_phase` satisfies the limit by construction");

    // Built with a zero halo — the plan a different strategy, or a wire, could
    // hand over — it is refused by name rather than discovered as a read past
    // the end of a buffer.
    let mut short = planned.clone();
    short.phases[1] = PhaseDecomposition::derive(
        Vec::new(),
        Vec::new(),
        [0, 0, 0],
        [0, 0, 0],
        short.phases[1].grid.clone(),
    )
    .with_source_images([0]);
    let env = ArrayEnvironment::new(ramp(VOLUME), short.n_phases(), [4, 4, 4]).expect("an env");
    let failed = execute_phases(
        "windowed",
        &identity_workflow(VOLUME),
        &short,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .expect_err("an operand wanting more than the phase fetches must be refused")
    .to_string();
    assert!(failed.contains("windowed-operand"), "{failed}");
    assert!(failed.contains("halo"), "{failed}");
    assert!(failed.contains("image 0"), "{failed}");
}
