// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A stress test of the framework against a shape it was not designed for, using
// only noop ops and the accounting loader: **three grids, two transitions**, an
// operation that dictates its own patch lattice from the global extent, and one
// input producing several outputs of differing dtype and rank plus a per-block
// non-pixel fragment.
//
// The question these tests answer is not whether any arithmetic is right —
// nothing computes a pixel here. It is whether the planner, the geometry, the
// tiling check, the task graph and the executor can **express and execute** the
// shape. Each test is written to record one answer, and several of them record
// an answer of "no" by asserting the exact error the framework raises.
//
// Read `patch_grid/mod.rs` first: it holds the lattice, the environment that
// carries the coordinate mapping the framework cannot, and why that mapping had
// to go there.

mod patch_grid;

use blockflow::{
    execute, BlockGrid, Chain, Constraints, Decomposition, Dtype, Enumerating, Environment, Greedy,
    Hints, PhaseDecomposition, Reach, Region, Space, Strategy, TaskGraph, Workflow,
};
use patch_grid::{
    one_phase, CountedOp, ExtraOutput, Inbound, Lattice, PatchGeometry, RegridEnvironment,
};

/// The 2-D case, spelled as this project spells it: a 3-D volume with a
/// degenerate z extent, not a second code path.
const VOLUME: [usize; 3] = [1, 1408, 1408];
const EDGE: usize = 256;
const CLASSES: usize = 3;

fn geometry() -> PatchGeometry {
    PatchGeometry::new(VOLUME, EDGE, CLASSES).expect("the extent carries the fixed patch edge")
}

// ------------------------------------------------- the lattice itself --

/// The lattice positions depend on the extent it was built over, so a block is
/// not a small copy of the volume.
///
/// This is the whole reason the operation cannot simply be handed a block. It
/// is the defect class `XY_BLOCK_SPLITTING.md` records for an unanchored sample
/// grid, and the sentence that matters there applies unchanged here: **a halo
/// does not fix it.** Widening the read does not move the lattice back.
#[test]
fn the_patch_lattice_moves_when_it_is_built_over_a_block_instead_of_the_volume() {
    let global = Lattice::new(1408, EDGE).unwrap();
    let block = Lattice::new(704, EDGE).unwrap();

    // Same edge, different positions — and not merely offset: a different count
    // and a different step.
    assert_eq!(global.count(), 11);
    assert_eq!(block.count(), 6);
    assert_eq!(global.start(1), 115);
    assert_eq!(block.start(1), 90);
    // Rebuilt over the block, patch 1 would start at 90 within the block, i.e.
    // at 90 or 794 globally — neither of which is a global patch position.
    let global_starts: Vec<usize> = (0..global.count()).map(|j| global.start(j)).collect();
    assert!(!global_starts.contains(&90));
    assert!(!global_starts.contains(&794));
}

/// The lattice covers everything and its reach in patch-index units is small.
#[test]
fn the_lattice_covers_the_extent_and_its_index_reach_is_bounded() {
    let geometry = geometry();
    for lattice in &geometry.lattice {
        assert!(lattice.covers_everything());
        // Evenly spread with `count = ceil(2 len / edge)` gives a step just
        // under `edge / 2`, so up to **three** patches cover a position and the
        // reach in index units is 2 — not the 1 that a clean `edge / 2` stride
        // would give. Recorded because the exact number is what an op has to
        // declare.
        assert_eq!(lattice.index_reach(), 2);
    }
    assert_eq!(geometry.patch_reach(), [2, 2, 0]);
    assert_eq!(geometry.patch_volume(), [11, 11, 196_608]);
}

// ------------------------------ transition 1: spatial -> patch grid --

/// **The tiling invariant holds on the patch-grid array**, exactly as argued.
///
/// Block `(j, i)` writes slot `[j, i, ..]`; those slots are disjoint and cover
/// the patch-grid array once each, so `boxes_tile_exactly` passes — with a
/// **zero** reach and a zero halo, because in patch-index space there is no
/// overlap to have. The overlap has not gone anywhere: it is entirely inside the
/// mapping between the two grids, which is priced below and which the tiling
/// check never sees.
#[test]
fn the_patch_slot_writes_tile_the_patch_grid_exactly_with_no_halo() {
    let geometry = geometry();
    let volume = geometry.patch_volume();
    let chain = Chain::op(CountedOp::new("gather", [0, 0, 0]));
    let grid = BlockGrid::new(volume, [1, 1, geometry.payload()]).unwrap();
    assert_eq!(grid.n_blocks(), 121);

    let decomposition = one_phase(
        &chain,
        volume,
        Dtype::F32,
        grid.clone(),
        [0, 0, 0],
        [0, 0, 0],
    );
    // The guard the correctness argument rests on, run over the patch grid.
    decomposition.check().unwrap();
    for block in &decomposition.phases[0].blocks {
        assert!(block.valid_covers_core());
        assert_eq!(block.valid, block.read);
    }

    let workflow = Workflow::new(chain, volume, Dtype::F32);
    let env = RegridEnvironment::new(volume, [1, 1, geometry.payload()], Dtype::F32, grid)
        .with_inbound(Inbound::SpatialUnderPatches(geometry.clone()));
    let stats = execute(
        "harness",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();

    // Every block touched, once, by the one op — asserted from the log, and
    // independently from what the environment saw.
    stats
        .log
        .check_coverage_and_order(&[(0, "gather".to_string())], 121)
        .unwrap();
    assert!(stats.log.duplicate_applications().is_empty());
    assert_eq!(env.touched_blocks().len(), 121);
    assert_eq!(stats.tasks, 121);

    // The overlap, priced. The framework's own read counter sees only the patch
    // grid; the real fetch is 4x the spatial volume, and nothing in the plan
    // says so.
    assert_eq!(stats.read_voxels, 121 * geometry.payload() as u64);
    assert_eq!(env.foreign_reads(), 121);
    let spatial_voxels = (VOLUME[0] * VOLUME[1] * VOLUME[2]) as u64;
    assert_eq!(env.foreign_union_elements(), 121 * (EDGE * EDGE) as u64);
    assert_eq!(
        env.foreign_gathered_elements(),
        env.foreign_union_elements()
    );
    let redundancy = env.foreign_gathered_elements() as f64 / spatial_voxels as f64;
    assert!(
        (redundancy - 4.0).abs() < 0.01,
        "the patch overlap re-reads the spatial array {redundancy:.2}x, and the \
         decomposition's own reach is zero, so the cost model cannot see any of it"
    );
}

/// The redundancy above is invisible to the cost model, stated as an equality.
///
/// `price_phase` derives redundancy from `reach` on the phase's own grid. The
/// phase's reach is zero, so the model predicts exactly the core — while the
/// run moves four times the spatial volume through a mapping the model has no
/// term for.
#[test]
fn the_cost_model_prices_the_patch_grid_and_not_the_array_the_data_comes_from() {
    let geometry = geometry();
    let volume = geometry.patch_volume();
    let chain = Chain::op(CountedOp::new("gather", [0, 0, 0]));
    let grid = BlockGrid::new(volume, [1, 1, geometry.payload()]).unwrap();
    let decomposition = one_phase(&chain, volume, Dtype::F32, grid, [0, 0, 0], [0, 0, 0]);

    let predicted: usize = decomposition.exact_read_voxels().iter().sum();
    assert_eq!(predicted, 121 * geometry.payload());
    // Nothing in the decomposition mentions the spatial array at all.
    assert_eq!(decomposition.volume, volume);
    assert_eq!(decomposition.chain_reach, [0, 0, 0]);
}

// ------------------------------ transition 2: patch grid -> spatial --

/// A bounded reach in patch-index units **is** statable through
/// `reach(axis, volume_len)` — but only while the phase's grid is over the patch
/// array, because `reach` is always in the units of the volume the phase is
/// decomposed over.
#[test]
fn a_bounded_reach_in_patch_index_units_is_statable_over_the_patch_grid() {
    let geometry = geometry();
    let volume = geometry.patch_volume();
    let reach = geometry.patch_reach();
    assert_eq!(reach, [2, 2, 0]);

    let chain = Chain::op(CountedOp::new("blend", reach));
    let grid = BlockGrid::new(volume, [3, 3, geometry.payload()]).unwrap();
    let decomposition = one_phase(&chain, volume, Dtype::F32, grid.clone(), reach, reach);
    decomposition.check().unwrap();
    for block in &decomposition.phases[0].blocks {
        assert!(block.valid_covers_core());
    }

    let workflow = Workflow::new(chain, volume, Dtype::F32);
    let env = RegridEnvironment::new(volume, [1, 1, geometry.payload()], Dtype::F32, grid);
    let stats = execute(
        "harness",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();
    assert_eq!(stats.tasks, 16);
    assert!(stats.read_voxels > stats.write_voxels);
    // The units are exactly what the op meant: two patch indices, not two
    // voxels of anything.
    assert_eq!(decomposition.phases[0].reach, [2, 2, 0]);
}

/// Restating the same bounded reach for a phase decomposed over the **spatial**
/// array costs a factor of nearly three in patches fetched.
///
/// Two independent over-declarations, both forced by the API:
///
/// * `reach` is one number per axis, applied to **both** sides of the block
///   (`geometry.rs` `BlockGeometry::derive`), while the real dependency is
///   asymmetric — an output pixel needs patches that *start* up to `edge - 1`
///   before it and none that start after it;
/// * a dependency that is exact in patch indices has to be rounded up into
///   voxels, because the only coordinate space the phase has is the spatial one.
#[test]
fn restating_the_patch_reach_in_spatial_units_over_declares_it() {
    let geometry = geometry();
    let reach = geometry.spatial_reach();
    assert_eq!(reach, [0, 255, 255]);

    let chain = Chain::op(CountedOp::new("combine", reach));
    let grid = BlockGrid::along(VOLUME, &[1, 2], 256).unwrap();
    assert_eq!(grid.n_blocks(), 36);
    let decomposition = one_phase(&chain, VOLUME, Dtype::F32, grid.clone(), reach, reach);
    decomposition.check().unwrap();

    let workflow = Workflow::new(chain, VOLUME, Dtype::F32);
    let env = RegridEnvironment::new(VOLUME, [1, 256, 256], Dtype::F32, grid)
        .with_inbound(Inbound::PatchesUnderSpatial(geometry.clone()));
    let stats = execute(
        "harness",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();
    assert_eq!(stats.tasks, 36);

    let fetched = env.foreign_gathered_elements() / geometry.payload() as u64;

    // What the blend genuinely needs, computed straight off the lattice: the
    // patches covering each block's *core*.
    let mut needed = 0u64;
    for block in &decomposition.phases[0].blocks {
        let mut patches = 1u64;
        for axis in 0..2 {
            let lattice = &geometry.lattice[axis];
            let start = block.core.start[axis + 1];
            let end = start + block.core.shape[axis + 1];
            let (first, last) = lattice.patches_over(start, end);
            patches *= (last - first + 1) as u64;
        }
        needed += patches;
    }

    assert_eq!(needed, 441);
    assert_eq!(fetched, 1444);
    assert!(
        fetched as f64 / needed as f64 > 3.2,
        "fetched {fetched} patch payloads where {needed} would do"
    );
}

/// **The measurement this change exists for.** Stating the same dependency
/// one-sided cuts the over-fetch, and stating it in the space it is exact in
/// removes the rest.
///
/// Three statements of one dependency, over the same volume, the same grid and
/// the same lattice, with what each fetches counted through the environment's
/// own accounting or off the plan it produced:
///
/// | statement | patch payloads fetched | over what is needed |
/// |---|---|---|
/// | symmetric `[0, 255, 255]` — the only form there was | 1444 | **3.27x** |
/// | one-sided `(255, 0)` per axis — the dependency as it actually is | 900 | 2.04x |
/// | the patch lattice's own index space, as a per-block fetch region | 441 | **1.00x** |
///
/// The first two are measured by running; the third is counted off a plan that
/// checks, because the mapping it needs is a per-block fetch region and this
/// harness's environment is the thing that would have to serve it.
///
/// What each column is: the symmetric form is one number applied to both sides,
/// and the low side is the one that is real — a pixel needs patches that
/// *start* before it and none that start after. Halving that is the asymmetric
/// row. What is left over is the **unit**: a dependency that is two patch
/// indices becomes `edge - 1` voxels when the only space the phase has is the
/// spatial one, and no amount of asymmetry recovers the difference between "two
/// patches" and "255 voxels".
#[test]
fn a_one_sided_reach_cuts_the_over_fetch_and_the_index_space_removes_the_rest() {
    let geometry = geometry();
    let grid = BlockGrid::along(VOLUME, &[1, 2], 256).unwrap();

    // What the blend genuinely needs: the patches covering each block's core.
    let needed = |blocks: &[blockflow::BlockGeometry]| -> u64 {
        blocks
            .iter()
            .map(|block| {
                let mut patches = 1u64;
                for axis in 0..2 {
                    let lattice = &geometry.lattice[axis];
                    let start = block.core.start[axis + 1];
                    let end = start + block.core.shape[axis + 1];
                    let (first, last) = lattice.patches_over(start, end);
                    patches *= (last - first + 1) as u64;
                }
                patches
            })
            .sum()
    };

    // ---- one-sided, in the phase's own voxels, run and counted ----
    //
    // The dependency as it is: `edge - 1` voxels below each output pixel and
    // none above it. `derive` shrinks the trustworthy extent by `lo` at the
    // bottom and `hi` at the top, so the valid regions still cover the cores and
    // the tiling check still passes — this is a tighter *true* statement, not a
    // weaker one.
    let one_sided = Reach::asymmetric([(0, 0), (255, 0), (255, 0)]);
    let chain = Chain::op(CountedOp::new("combine", [0, 255, 255]).with_reach(one_sided.clone()));
    let decomposition = one_phase(
        &chain,
        VOLUME,
        Dtype::F32,
        grid.clone(),
        one_sided.clone(),
        one_sided,
    );
    decomposition.check().unwrap();
    for block in &decomposition.phases[0].blocks {
        assert!(block.valid_covers_core(), "block {:?}", block.index);
    }
    let workflow = Workflow::new(chain, VOLUME, Dtype::F32);
    let env = RegridEnvironment::new(VOLUME, [1, 256, 256], Dtype::F32, grid.clone())
        .with_inbound(Inbound::PatchesUnderSpatial(geometry.clone()));
    let stats = execute(
        "harness",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();
    assert_eq!(stats.tasks, 36);
    let one_sided_fetched = env.foreign_gathered_elements() / geometry.payload() as u64;
    let want = needed(&decomposition.phases[0].blocks);
    assert_eq!(want, 441);
    assert_eq!(one_sided_fetched, 900);

    // ---- the index space, as a per-block fetch region, off the plan ----
    //
    // Level 0 is the patch array; the phase is over the spatial one; each block
    // fetches exactly the patch slots covering its core, and the dependency is
    // declared in that array's frame rather than restated in voxels of this one.
    // The permutation is real: the phase's axes are `(z, y, x)` and the patch
    // array's are `(ty, tx, payload)`.
    let index_space = Reach::from([2, 2, 0]).in_space(
        Space::source_index()
            .with_axes([1, 2, 0])
            .expect("a permutation"),
    );
    let indexed =
        Chain::op(CountedOp::new("combine", [0, 255, 255]).with_reach(index_space.clone()));
    let mut plan = one_phase(
        &indexed,
        geometry.patch_volume(),
        Dtype::F32,
        grid,
        index_space,
        [0, 0, 0],
    );
    plan.phases[0] = plan.phases[0].clone().with_sources(|block| {
        let mut start = vec![0usize; 3];
        let mut shape = vec![geometry.payload(); 3];
        for axis in 0..2 {
            let lattice = &geometry.lattice[axis];
            let at = block.core.start[axis + 1];
            let (first, last) = lattice.patches_over(at, at + block.core.shape[axis + 1]);
            start[axis] = first;
            shape[axis] = last - first + 1;
        }
        Region::new(&start, &shape)
    });
    plan.check().unwrap();
    let indexed_fetched: u64 = plan.phases[0]
        .blocks
        .iter()
        .map(|block| (block.source.shape[0] * block.source.shape[1]) as u64)
        .sum();
    assert_eq!(indexed_fetched, 441);

    // The symmetric figure this replaces is measured in the test above; the
    // ratios are what the change is worth.
    let symmetric = 1444u64;
    assert!(
        (symmetric as f64 / want as f64) > 3.2
            && (one_sided_fetched as f64 / want as f64) < 2.1
            && indexed_fetched == want,
        "symmetric {symmetric}, one-sided {one_sided_fetched}, indexed {indexed_fetched}, \
         needed {want}"
    );
}

// -------------------------------------------- one input, many outputs --

/// One pass producing several outputs of differing dtype and rank, plus a
/// per-block non-pixel fragment.
///
/// What works: the fragment. `(stream, phase, block) -> bytes` carries anything,
/// and the environment writes one per block with the run's own accounting behind
/// it.
///
/// What does not: the extra **arrays**. A `Workflow` names one output of one
/// dtype and `Environment::write` writes one buffer to one level, so the other
/// two outputs are written by the environment on the side. The framework's own
/// byte figure is then wrong by the ratio this test pins.
#[test]
fn several_outputs_of_differing_dtype_and_rank_leave_the_frameworks_accounting_short() {
    let geometry = geometry();
    let volume = geometry.patch_volume();
    let chain = Chain::op(CountedOp::new("gather", [0, 0, 0]));
    let grid = BlockGrid::new(volume, [1, 1, geometry.payload()]).unwrap();
    let decomposition = one_phase(
        &chain,
        volume,
        Dtype::F32,
        grid.clone(),
        [0, 0, 0],
        [0, 0, 0],
    );

    // The workflow's own output is the f32 field, whose folded rank is 3 and
    // whose real rank is 5: (ty, tx, classes, edge, edge).
    let extras = vec![
        ExtraOutput {
            name: "label",
            // one element per output position, i.e. one per class channel
            elements_per_valid_voxel: 1.0 / CLASSES as f64,
            dtype: Dtype::I32,
            rank: 4,
        },
        ExtraOutput {
            name: "score",
            elements_per_valid_voxel: 1.0 / CLASSES as f64,
            dtype: Dtype::F32,
            rank: 4,
        },
    ];
    let env = RegridEnvironment::new(volume, [1, 1, geometry.payload()], Dtype::F32, grid)
        .with_inbound(Inbound::SpatialUnderPatches(geometry.clone()))
        .with_extras(extras)
        .with_sidecar("block_summary", 1024)
        .unwrap();

    let workflow = Workflow::new(chain, volume, Dtype::F32);
    let stats = execute(
        "harness",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();

    // The fragment stream: one per block, keyed by the block index the
    // environment had to re-derive from the write region.
    let keys = env.sidecar_keys("block_summary").unwrap();
    assert_eq!(keys.len(), 121);
    assert_eq!(env.counters().sidecar_snapshot().0, 121);
    assert_eq!(env.counters().sidecar_snapshot().2, 121 * 1024);

    // The region accounting, which is what a plan is checked against, counts
    // only the one output the workflow names.
    let framework_bytes = stats.write_voxels * Dtype::F32.size_of() as u64;
    let real_bytes = framework_bytes + env.extra_output_bytes();
    assert_eq!(framework_bytes, 121 * 196_608 * 4);
    assert!(
        real_bytes as f64 / framework_bytes as f64 > 1.6,
        "the run wrote {real_bytes} bytes and the framework counted {framework_bytes}"
    );
}

// ------------------------------------------- what the framework refuses --

/// The two transitions **are** two phases of one decomposition, once the second
/// says where it reads.
///
/// This test recorded the opposite for as long as a `Decomposition` had one
/// volume: `check` refused any phase whose grid was over a different one, so the
/// pipeline could only be two decompositions with no dependency edge between
/// them. A phase owns its volume now, and what took the refusal's place is a
/// check with something to say rather than a shape assertion:
///
/// * a phase reading level `p` must fetch from **inside** level `p`, and a plan
///   that changes shape without saying how is caught by exactly that — the
///   second phase's blocks are cut from the spatial array and default to reading
///   their own extent, which is not a region of the patch array;
/// * given the mapping, the plan checks, the task graph joins the two phases by
///   region intersection **in the patch array's own space**, and the read
///   accounting counts what will actually be fetched.
///
/// The mapping itself is `Lattice::patches_over`, which is the harness's model
/// of the operation — it is a function of the block index, as a binding plan
/// requires, and never of the data.
#[test]
fn a_phase_boundary_may_change_the_shape_when_it_says_where_it_reads() {
    let geometry = geometry();
    let patch_volume = geometry.patch_volume();
    let payload = geometry.payload();
    let gather = Chain::op(CountedOp::new("gather", [0, 0, 0]));
    let combine = Chain::op(CountedOp::new("combine", [0, 0, 0]));

    let gather_phase = PhaseDecomposition::derive(
        vec![0],
        vec![gather.display_name()],
        [0, 0, 0],
        [0, 0, 0],
        BlockGrid::new(patch_volume, [1, 1, payload]).unwrap(),
    );
    let combine_phase = PhaseDecomposition::derive(
        vec![1],
        vec![combine.display_name()],
        [0, 0, 0],
        [0, 0, 0],
        BlockGrid::along(VOLUME, &[1, 2], EDGE).unwrap(),
    );

    // Without the mapping: the second phase's blocks would read the spatial
    // regions they write, out of an array that is 11 x 11 x payload.
    let unmapped = Decomposition {
        volume: patch_volume,
        dtype: Dtype::F32,
        phases: vec![gather_phase.clone(), combine_phase.clone()],
        chain_reach: [0, 0, 0],
    };
    let error = unmapped.check().unwrap_err().to_string();
    assert!(
        error.contains("reads from level 1") && error.contains("region axis"),
        "{error}"
    );

    // With it: every block of the combine fetches the patches covering its own
    // core, in patch-index space.
    let lattice = geometry.lattice.clone();
    let mapped = combine_phase.with_sources(|block| {
        let (first_y, last_y) = lattice[0].patches_over(
            block.core.start[1],
            block.core.start[1] + block.core.shape[1],
        );
        let (first_x, last_x) = lattice[1].patches_over(
            block.core.start[2],
            block.core.start[2] + block.core.shape[2],
        );
        Region::new(
            &[first_y, first_x, 0],
            &[last_y - first_y + 1, last_x - first_x + 1, payload],
        )
    });
    let across = Decomposition {
        volume: patch_volume,
        dtype: Dtype::F32,
        phases: vec![gather_phase, mapped],
        chain_reach: [0, 0, 0],
    };
    across.check().unwrap();

    // The seam is a real edge now: every task of the combine depends on the
    // gather tasks that produced the patches it reads, and the check that says
    // so is not vacuous.
    let graph = TaskGraph::build(&across);
    graph.dependencies_cover_reads(&across).unwrap();
    let combine_tasks: Vec<_> = graph.tasks_in_phase(1).to_vec();
    assert_eq!(combine_tasks.len(), VOLUME[1].div_ceil(EDGE).pow(2));
    assert!(
        combine_tasks.iter().all(|task| task.deps.len() >= 4),
        "a spatial block spans several patches, so it must depend on several \
         gather tasks: {:?}",
        combine_tasks.iter().map(|task| task.deps.len()).min()
    );

    // And the plan's own read accounting is over what is fetched, so the
    // cross-grid traffic is a number in the plan rather than a surprise in the
    // run: a block that spans four patches reads four payloads.
    let read = across.exact_read_voxels();
    let fetched: usize = combine_tasks
        .iter()
        .map(|task| task.geometry.source.voxels())
        .sum();
    assert_eq!(read[1], fetched);
    assert!(
        read[1] > combine_tasks.len() * payload,
        "a lattice with overlap reads more than one payload per block: {} against {}",
        read[1],
        combine_tasks.len() * payload
    );
}

/// And the executor insists on the same single volume three ways over.
#[test]
fn the_executor_requires_one_volume_for_the_workflow_the_plan_and_the_environment() {
    let geometry = geometry();
    let patch_volume = geometry.patch_volume();
    let chain = Chain::op(CountedOp::new("combine", [0, 0, 0]));
    let grid = BlockGrid::along(VOLUME, &[1, 2], 256).unwrap();
    let decomposition = one_phase(
        &chain,
        VOLUME,
        Dtype::F32,
        grid.clone(),
        [0, 0, 0],
        [0, 0, 0],
    );

    // A workflow that reads the patch grid and writes the spatial array is not
    // expressible: `Workflow` has one `shape`.
    let workflow = Workflow::new(chain, patch_volume, Dtype::F32);
    let env = RegridEnvironment::new(VOLUME, [1, 256, 256], Dtype::F32, grid);
    let error = execute(
        "harness",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("volume disagreement"), "{error}");
}

/// Nothing lets the operation constrain the block shape, so the planner is free
/// to hand it one it cannot run — and every guard passes.
///
/// The patch lattice mandates a block of exactly one patch. `Constraints` offers
/// a list of scalar block edges and the strategy picks whichever prices best;
/// `BlockOp` has no method that says "these are the shapes I accept". So a
/// candidate set that does not contain the mandated shape produces a
/// decomposition that tiles, whose dependencies cover its reads, and which is
/// unrunnable.
#[test]
fn the_planner_may_choose_a_block_shape_the_operation_cannot_accept() {
    let geometry = geometry();
    let volume = geometry.patch_volume();
    let chain = Chain::op(CountedOp::new("gather", [0, 0, 0]));
    let workflow = Workflow::new(chain, volume, Dtype::F32);
    let constraints = Constraints {
        // The mandated edge — one patch — is deliberately absent.
        block_candidates: vec![2, 4],
        split_axes: vec![0, 1],
        ..Constraints::default()
    };

    for strategy in [
        &Enumerating::default() as &dyn Strategy,
        &Greedy::default() as &dyn Strategy,
    ] {
        let decomposition = strategy.decompose(&workflow, &constraints).unwrap();
        let block = decomposition.phases[0].grid.block();
        assert_ne!(
            block[0],
            1,
            "{}: nothing told the planner the block must be one patch",
            strategy.name()
        );
        // And every guard the framework has is satisfied by it.
        decomposition.check().unwrap();
    }
}

/// A block-index key has to be inverted out of the write region, because
/// `Environment::write` is handed regions and no index.
///
/// This is what the sidecar path costs today for anything keyed by block: it
/// works, and it works only because the write grid is a regular lattice.
#[test]
fn keying_anything_by_block_means_inverting_the_geometry_from_the_write_region() {
    let geometry = geometry();
    let volume = geometry.patch_volume();
    let chain = Chain::op(CountedOp::new("gather", [0, 0, 0]));
    let grid = BlockGrid::new(volume, [1, 1, geometry.payload()]).unwrap();
    let decomposition = one_phase(
        &chain,
        volume,
        Dtype::F32,
        grid.clone(),
        [0, 0, 0],
        [0, 0, 0],
    );
    let env = RegridEnvironment::new(volume, [1, 1, geometry.payload()], Dtype::F32, grid)
        .with_sidecar("block_summary", 8)
        .unwrap();
    let workflow = Workflow::new(chain, volume, Dtype::F32);
    execute(
        "harness",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();

    let recovered = env.touched_blocks();
    let planned: Vec<[usize; 3]> = decomposition.phases[0]
        .blocks
        .iter()
        .map(|block| block.index)
        .collect();
    let mut sorted = planned.clone();
    sorted.sort_unstable();
    assert_eq!(recovered, sorted);
    // The fragment keys agree with the plan's own block indices.
    let keyed: Vec<[usize; 3]> = env
        .sidecar_keys("block_summary")
        .unwrap()
        .into_iter()
        .map(|key| key.block)
        .collect();
    assert_eq!(keyed, sorted);
}

// -------------------------------- what the 3-D form would additionally need --

/// Three passes over one array, combined, is not expressible: `Chain` has no
/// fan-in, and `Alternative` runs exactly one branch.
///
/// This is the same gap `BLOCK_OPS.md` records for the diamond — and it stays
/// hidden the same way, because `max` over branches folds the right *reach* for
/// both readings. Asserted through the op counter, which is the one place the
/// difference between "one branch ran" and "three did" is visible.
#[test]
fn three_passes_over_one_array_run_as_one_because_a_chain_has_no_fan_in() {
    let volume = [1, 256, 256];
    let branches = vec![
        Chain::op(CountedOp::new("pass_a", [0, 4, 4]).with_order([0, 1, 2])),
        Chain::op(CountedOp::new("pass_b", [4, 0, 4]).with_order([1, 0, 2])),
        Chain::op(CountedOp::new("pass_c", [4, 4, 0]).with_order([2, 0, 1])),
    ];
    let chain = Chain::alternative(branches, 0).unwrap();
    // The reach is folded as the max over branches, which is the right budget
    // whichever branch runs — and would also be the right budget if all three
    // ran. That coincidence is why the gap hides.
    assert_eq!(chain.reach3(&volume), [4, 4, 4]);

    let reach = chain.reach3(&volume);
    let grid = BlockGrid::along(volume, &[1, 2], 64).unwrap();
    let decomposition = one_phase(&chain, volume, Dtype::F32, grid.clone(), reach, reach);
    decomposition.check().unwrap();
    let workflow = Workflow::new(chain, volume, Dtype::F32);
    let env = RegridEnvironment::new(volume, [1, 64, 64], Dtype::F32, grid);
    let stats = execute(
        "harness",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();

    // One application per block, not three: only `taken` ran.
    assert_eq!(stats.ops_applied, stats.tasks);
    assert_eq!(stats.tasks, 16);
}

/// Three conflicting traversal preferences over one array yield no hint at all,
/// and the disagreement heuristic cuts the chain into three phases.
///
/// Both halves are the current behaviour rather than a defect, and both are
/// costs the 3-D form would pay: no visit order to rank prefetch by, and three
/// materialisations of the same volume where the passes are independent and
/// could in principle share one read.
#[test]
fn three_conflicting_traversal_orders_produce_no_hint_and_three_phases() {
    let volume = [64, 64, 64];
    let chain = Chain::sequence(vec![
        Chain::op(CountedOp::new("pass_a", [0, 2, 2]).with_order([0, 1, 2])),
        Chain::op(CountedOp::new("pass_b", [2, 0, 2]).with_order([1, 0, 2])),
        Chain::op(CountedOp::new("pass_c", [2, 2, 0]).with_order([2, 0, 1])),
    ]);
    assert_eq!(chain.preferred_iterations().len(), 3);

    let workflow = Workflow::new(chain, volume, Dtype::F32);
    let constraints = Constraints {
        block_candidates: vec![16, 32],
        split_axes: vec![0, 1, 2],
        ..Constraints::default()
    };
    let greedy = Greedy::default();
    let decomposition = greedy.decompose(&workflow, &constraints).unwrap();
    assert_eq!(decomposition.n_phases(), 3);
    // No consensus, so no order to rank visits or prefetch by.
    assert_eq!(greedy.hints(&workflow, &decomposition).visit_order, None);
    // And the only lever the cost model has for this is off by default.
    assert_eq!(constraints.model.order_conflict_penalty, 0.0);
}

// ---------------------------------------------------- the whole shape --

/// Both transitions, end to end, as the framework can actually run them: two
/// decompositions over two volumes, joined by a mapping it cannot see.
///
/// Everything inside each half works unchanged — geometry, tiling check, task
/// graph, scheduler, event log, cost accounting. What is missing is any
/// relationship *between* the halves: two task graphs rather than one, no
/// dependency edge across the transition, and therefore no
/// `dependencies_cover_reads` over the mapping that carries all the overlap.
#[test]
fn the_whole_shape_runs_as_two_decompositions_with_nothing_joining_them() {
    let geometry = geometry();
    let patch_volume = geometry.patch_volume();

    // Transition 1.
    let gather = Chain::op(CountedOp::new("gather", [0, 0, 0]));
    let gather_grid = BlockGrid::new(patch_volume, [1, 1, geometry.payload()]).unwrap();
    let gather_plan = one_phase(
        &gather,
        patch_volume,
        Dtype::F32,
        gather_grid.clone(),
        [0, 0, 0],
        [0, 0, 0],
    );
    let gather_workflow = Workflow::new(gather, patch_volume, Dtype::F32);
    let gather_env = RegridEnvironment::new(
        patch_volume,
        [1, 1, geometry.payload()],
        Dtype::F32,
        gather_grid,
    )
    .with_inbound(Inbound::SpatialUnderPatches(geometry.clone()))
    .with_sidecar("block_summary", 1024)
    .unwrap();
    let first = execute(
        "harness",
        &gather_workflow,
        &gather_plan,
        &Hints::default(),
        &gather_env,
    )
    .unwrap();

    // Transition 2.
    let spatial_reach = geometry.spatial_reach();
    let combine = Chain::op(CountedOp::new("combine", spatial_reach));
    let combine_grid = BlockGrid::along(VOLUME, &[1, 2], 256).unwrap();
    let combine_plan = one_phase(
        &combine,
        VOLUME,
        Dtype::F32,
        combine_grid.clone(),
        spatial_reach,
        spatial_reach,
    );
    let combine_workflow = Workflow::new(combine, VOLUME, Dtype::F32);
    let combine_env = RegridEnvironment::new(VOLUME, [1, 256, 256], Dtype::F32, combine_grid)
        .with_inbound(Inbound::PatchesUnderSpatial(geometry.clone()));
    let second = execute(
        "harness",
        &combine_workflow,
        &combine_plan,
        &Hints::default(),
        &combine_env,
    )
    .unwrap();

    // Each half is internally sound.
    assert_eq!(first.tasks, 121);
    assert_eq!(second.tasks, 36);
    assert_eq!(first.phases, 1);
    assert_eq!(second.phases, 1);
    assert_eq!(gather_env.touched_blocks().len(), 121);
    assert_eq!(combine_env.touched_blocks().len(), 36);

    // Two fingerprints, two graphs, no edge between them. A run is reproducible
    // half by half and there is nothing that identifies the pair.
    assert_ne!(
        first.decomposition_fingerprint,
        second.decomposition_fingerprint
    );

    // The mapping's cost is real and lives entirely outside both plans.
    let spatial_voxels = (VOLUME[0] * VOLUME[1] * VOLUME[2]) as u64;
    assert!(gather_env.foreign_gathered_elements() > spatial_voxels * 3);
    assert!(
        combine_env.foreign_gathered_elements()
            > (patch_volume[0] * patch_volume[1] * patch_volume[2]) as u64
    );
    let predicted: usize = combine_plan.exact_read_voxels().iter().sum();
    assert!(
        (predicted as u64) < combine_env.foreign_gathered_elements(),
        "the plan predicts {predicted} voxels read, and the run moved {} elements \
         from an array the plan never mentions",
        combine_env.foreign_gathered_elements()
    );
}

/// The first phase of a decomposition has no dependencies by construction, so
/// the graph check that would catch a short halo has nothing to check at a grid
/// transition.
#[test]
fn the_dependency_check_has_nothing_to_say_about_a_grid_transition() {
    let geometry = geometry();
    let volume = geometry.patch_volume();
    let chain = Chain::op(CountedOp::new("gather", [0, 0, 0]));
    let grid = BlockGrid::new(volume, [1, 1, geometry.payload()]).unwrap();
    let decomposition = one_phase(&chain, volume, Dtype::F32, grid, [0, 0, 0], [0, 0, 0]);

    let graph = blockflow::TaskGraph::build(&decomposition);
    assert!(graph.tasks.iter().all(|task| task.deps.is_empty()));
    // Which passes, vacuously.
    graph.dependencies_cover_reads(&decomposition).unwrap();
}

/// A short halo on the spatial side of transition 2 still fails loudly, so the
/// existing guard is not weakened by the unusual coordinate space — it simply
/// guards the half of the shape it can see.
#[test]
fn the_existing_halo_guard_still_fires_on_the_spatial_side() {
    let geometry = geometry();
    let reach = geometry.spatial_reach();
    let chain = Chain::op(CountedOp::new("combine", reach));
    let grid = BlockGrid::along(VOLUME, &[1, 2], 256).unwrap();
    let good = one_phase(&chain, VOLUME, Dtype::F32, grid, reach, reach);
    good.check().unwrap();

    let bad = good.with_forced_halo([0, 16, 16]);
    let error = bad.check().unwrap_err().to_string();
    assert!(error.contains("do not tile the volume exactly"), "{error}");
    assert!(error.contains("lost part of their core"), "{error}");
}

/// The patch-grid array's own region arithmetic is rank-3 because everything in
/// the geometry is `[usize; 3]`, so the natural rank-6 shape has to be folded.
///
/// Recorded as an equality rather than an argument: the two patch-index axes and
/// the payload use all three available axes, which is why the 2-D case fits
/// exactly and leaves nothing for a z extent above 1.
#[test]
fn the_patch_grid_array_is_folded_to_rank_three_and_uses_every_axis() {
    let geometry = geometry();
    let volume = geometry.patch_volume();
    assert_eq!(volume.len(), 3);
    assert_eq!(volume[0], geometry.lattice[0].count());
    assert_eq!(volume[1], geometry.lattice[1].count());
    assert_eq!(volume[2], geometry.payload());
    // The payload is the whole of what a patch carries, including the
    // degenerate z extent — there is no fourth axis to put it on.
    assert_eq!(geometry.payload(), VOLUME[0] * CLASSES * EDGE * EDGE);

    // `Region` itself is rank-generic, and so is the tiling predicate: the limit
    // is the geometry and the decomposition, not the box type.
    let rank_five = Region::new(&[0, 0, 0, 0, 0], &[1, 11, 11, 3, 256]);
    assert_eq!(rank_five.ndim(), 5);
    blockflow::boxes_tile_exactly(&[rank_five.ranges()], &[1, 11, 11, 3, 256]).unwrap();
}

/// A bespoke `Strategy` **can** produce the mandated grid, and a foreign `run`
/// honours it.
///
/// So the seam is adequate and the gap is narrower than "the framework cannot
/// plan this": what is missing is a way for the *op* to say what it needs.
/// `PatchStrategy` is told the lattice at construction, and nothing in the chain
/// it plans for could have told it.
#[test]
fn a_bespoke_strategy_produces_the_mandated_grid_and_a_foreign_run_honours_it() {
    let geometry = geometry();
    let volume = geometry.patch_volume();
    let chain = Chain::op(CountedOp::new("gather", [0, 0, 0]));
    let workflow = Workflow::new(chain, volume, Dtype::F32);
    let strategy = patch_grid::PatchStrategy::new(geometry.clone());

    let decomposition = strategy
        .decompose(&workflow, &Constraints::default())
        .unwrap();
    assert_eq!(
        decomposition.phases[0].grid.block(),
        [1, 1, geometry.payload()]
    );
    assert_eq!(decomposition.n_tasks(), 121);

    // `Greedy::run` against a decomposition it did not choose — the property the
    // one-trait design would otherwise erode — over this shape.
    let grid = decomposition.phases[0].grid.clone();
    let env = RegridEnvironment::new(volume, [1, 1, geometry.payload()], Dtype::F32, grid)
        .with_inbound(Inbound::SpatialUnderPatches(geometry));
    let stats = Greedy::default()
        .run(&workflow, &decomposition, &env)
        .unwrap();
    assert_eq!(stats.tasks, 121);
    assert_eq!(stats.decomposition_fingerprint, decomposition.fingerprint());
}
