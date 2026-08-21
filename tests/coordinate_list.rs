// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `ops::coordinates`: a mask in, the coordinate of
// every set voxel out, in the order a single-block walk would produce them.
//
// **The order is the specification, so the order is what this suite is about.**
// Which voxels come back is the trivial half and one assertion covers it. The
// half that can go wrong is which order they come back in, because the consumers
// of this list address its entries by position — an entry's index is the only
// name it has — so a permuted list is a different answer that is still a
// well-formed list of the right coordinates.
//
// That shapes every test here in the same way: **nothing is compared as a set.**
// Every comparison is `assert_eq!` on a whole `Vec`, in order. A suite that
// sorted both sides before comparing would pass on exactly the failure this op
// exists to prevent.
//
// The failure with the sharpest edge, and the fixture that can see it
// -------------------------------------------------------------------
// The mechanism to reach for first is a base index per block: sum the per-block
// counts, give each block a contiguous range, write. That produces a
// **block-major** order — every entry of one block before every entry of the
// next — and block-major order is the walk order only on a lattice cut so that
// the blocks do not interleave. `ops::coordinates` says exactly when that is;
// this suite's job is to make sure the fixtures can tell the difference.
//
// They can only tell the difference if the scene is **asymmetric across the
// seams**. A mask whose set voxels happen to be arranged so that every block's
// entries really are contiguous in the walk order agrees with block-major order
// by accident, and a suite built on one would pass with the ordering entirely
// wrong. So `the_scene_separates_block_major_order_from_the_walk` asserts the
// two orders **disagree** on this scene under the interleaving cuts — the
// fixture is pinned as a discriminator rather than assumed to be one, and if a
// change ever makes them agree this suite says it has stopped testing the thing
// it is for.

use std::collections::BTreeMap;

use ndarray::Array3;

use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::PhaseWork;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::coordinates::{
    block_base_indices, blocks_concatenate_in_order, collect_coordinates, coordinate_schema,
    merge_coordinates, set_voxel_rows, set_voxels, set_voxels_phase, SetVoxelsOp,
};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::table::encoded_schema;
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [12, 12, 8];
const STREAM: &str = "coordinates.set";
const CHUNK: [usize; 3] = [4, 4, 4];

// -------------------------------------------------------------- the scene --

/// A mask arranged so that no cut of it is easy.
///
/// 1. **A dense 4-cube** at 2..=5 on every axis. Every cut below that is finer
///    than the volume passes a seam through the middle of it, so a block that
///    holds part of a dense run is the normal case here rather than a corner.
/// 2. **Five corners and faces**, including both opposite corners of the volume,
///    so that nothing treats the volume boundary as special.
/// 3. **A crooked line**, `[i, 11 - i, (3 * i) % 8]` for every `i`. It runs the
///    wrong way on axis 1 against axis 0, which is what makes the scene
///    asymmetric across a seam on axis 1 — the property
///    `the_scene_separates_block_major_order_from_the_walk` depends on.
/// 4. **A bar along axis 2** at `x = 9, y = 1`, filling the axis, so that a cut
///    on the fastest axis has something to cut.
///
/// And one thing deliberately **absent**: nothing is set anywhere in
/// `x in 8..12, y in 4..8, z in 4..8`. Under the `[4, 4, 4]` cut that is exactly
/// block `[2, 1, 1]`, which therefore finds nothing at all.
/// `a_block_that_finds_nothing_writes_a_blob_and_shifts_nobody` asserts that
/// rather than trusting this comment.
fn scene() -> Array3<bool> {
    let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);

    for i in 2..=5 {
        for j in 2..=5 {
            for k in 2..=5 {
                mask[[i, j, k]] = true;
            }
        }
    }

    for at in [[0, 0, 0], [11, 0, 0], [0, 11, 7], [0, 0, 7], [11, 11, 7]] {
        mask[at] = true;
    }

    for i in 0..VOLUME[0] {
        mask[[i, 11 - i, (3 * i) % 8]] = true;
    }

    for k in 0..VOLUME[2] {
        mask[[9, 1, k]] = true;
    }

    mask
}

/// The cuts the suite sweeps, and why each is here.
///
/// | cut | lattice | what it is for |
/// |---|---|---|
/// | `[12, 12, 8]` | 1 | one block: the reference path, run through the framework |
/// | `[4, 12, 8]` | 3 x 1 x 1 | slabs on the slowest axis — the one lattice where block-major *is* the walk |
/// | `[12, 4, 8]` | 1 x 3 x 1 | cut on axis 1 alone: blocks interleave maximally |
/// | `[12, 12, 3]` | 1 x 1 x 3 | cut on the fastest axis, and ragged (8 = 3 + 3 + 2) |
/// | `[4, 4, 4]` | 3 x 3 x 2 | all three axes, through the dense cube, and one block finds nothing |
/// | `[5, 5, 3]` | 3 x 3 x 3 | ragged on every axis, so no seam lands on a round number |
/// | `[6, 6, 8]` | 2 x 2 x 1 | seams exactly through the middle of the dense cube |
/// | `[1, 12, 8]` | 12 x 1 x 1 | one voxel thick: the finest slab lattice |
const CUTS: [[usize; 3]; 8] = [
    [12, 12, 8],
    [4, 12, 8],
    [12, 4, 8],
    [12, 12, 3],
    [4, 4, 4],
    [5, 5, 3],
    [6, 6, 8],
    [1, 12, 8],
];

// ------------------------------------------------------------- the harness --

fn shape_of(mask: &Array3<bool>) -> [usize; 3] {
    [mask.shape()[0], mask.shape()[1], mask.shape()[2]]
}

/// Run the one phase and hand back the environment, which is where the answer
/// is: this op writes no image, so there is nothing to read out of the image
/// stack and everything to read out of the store.
fn run(mask: &Array3<bool>, block: [usize; 3]) -> (ArrayEnvironment, Decomposition) {
    let volume = shape_of(mask);
    let grid = BlockGrid::new(volume, block).expect("a grid");
    let op = SetVoxelsOp::new("set voxels", STREAM, Lifecycle::Persistent);
    let plan = set_voxels_phase(grid, Dtype::Bool, &op).expect("a plan");
    let input: Voxels = mask.clone().into();
    let chunk = [
        CHUNK[0].min(volume[0]),
        CHUNK[1].min(volume[1]),
        CHUNK[2].min(volume[2]),
    ];
    let env = ArrayEnvironment::for_decomposition(input, &plan, chunk).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::Bool);
    execute_phases(
        "coordinates",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&op)],
    )
    .expect("a run");
    (env, plan)
}

/// Every block's blob, in **block-index order**.
///
/// Every block is asked for one, and a block that wrote none is a failure rather
/// than a hole: the stream declares every-block coverage, so an absent blob is
/// the thing the coverage guard exists to catch and this says so a second time
/// where the bytes are read.
fn blobs(env: &ArrayEnvironment, plan: &Decomposition) -> Vec<([usize; 3], Vec<u8>)> {
    plan.phases[0]
        .grid
        .cores()
        .into_iter()
        .map(|core| {
            let bytes = env
                .read_sidecar(STREAM, 0, core.index)
                .expect("the store answers")
                .unwrap_or_else(|| panic!("block {:?} wrote no blob", core.index));
            (core.index, bytes)
        })
        .collect()
}

/// The list a run produced, in the walk order.
fn listed(mask: &Array3<bool>, block: [usize; 3]) -> Vec<[usize; 3]> {
    let (env, _) = run(mask, block);
    collect_coordinates(&env, STREAM, 0, shape_of(mask)).expect("the merge")
}

/// One block's own entries, in that block's walk order.
///
/// A single-blob merge, which within one block is the same lexicographic order
/// the block wrote in — so this is the block's list and not a re-ordering of it.
fn entries_of(volume: [usize; 3], block: [usize; 3], bytes: &[u8]) -> Vec<[usize; 3]> {
    merge_coordinates(volume, [(block, bytes)]).expect("a blob merges")
}

/// The list a **base index per block** would produce: every block's entries,
/// concatenated in block-index order.
fn block_major(volume: [usize; 3], blobs: &[([usize; 3], Vec<u8>)]) -> Vec<[usize; 3]> {
    blobs
        .iter()
        .flat_map(|(block, bytes)| entries_of(volume, *block, bytes))
        .collect()
}

// ------------------------------------------------------- the fixture's teeth --

/// The scene holds what it says it holds, counted rather than inspected.
#[test]
fn the_scene_is_the_one_described() {
    let mask = scene();
    let found = set_voxels(mask.view());
    // 64 in the cube, 5 corners, 12 on the line, 8 in the bar; the line passes
    // through none of the others and neither does the bar, so nothing overlaps.
    assert_eq!(found.len(), 64 + 5 + 12 + 8);
    assert_eq!(mask.iter().filter(|set| **set).count(), found.len());
    // and nothing at all in the octant the empty-block test needs
    for i in 8..12 {
        for j in 4..8 {
            for k in 4..8 {
                assert!(!mask[[i, j, k]], "[{i}, {j}, {k}] should be clear");
            }
        }
    }
}

/// **The guard on the guard.** A base index per block gives block-major order;
/// this asserts that on this scene, under the cuts that interleave, block-major
/// order is *not* the walk order — and that under the slab cuts it is.
///
/// Without this the ordering tests below could all pass on a fixture where the
/// two orders coincide, which is exactly what a symmetric scene does. If a
/// change ever makes the two agree here, the ordering tests have stopped
/// testing anything and this is what says so.
#[test]
fn the_scene_separates_block_major_order_from_the_walk() {
    let mask = scene();
    let want = set_voxels(mask.view());
    for cut in CUTS {
        let (env, plan) = run(&mask, cut);
        let grid = &plan.phases[0].grid;
        let concatenated = block_major(VOLUME, &blobs(&env, &plan));
        // Either way it is the same multiset: block-major and the walk differ
        // only in order, which is what makes the difference invisible to any
        // set-shaped check.
        let mut sorted = concatenated.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, want, "cut {cut:?} lost or duplicated an entry");

        if blocks_concatenate_in_order(grid) {
            assert_eq!(
                concatenated, want,
                "cut {cut:?} does not interleave, so block-major order should be the walk"
            );
        } else {
            assert_ne!(
                concatenated, want,
                "cut {cut:?} interleaves, so this scene must separate the two orders — if it \
                 does not, every ordering test in this file is passing by accident"
            );
        }
    }
}

// ------------------------------------------------------------ the acceptance --

/// **The bar.** The same coordinates in the same order under every cut, as a
/// whole list.
#[test]
fn every_decomposition_produces_the_same_list_in_the_same_order() {
    let mask = scene();
    let want = set_voxels(mask.view());
    for cut in CUTS {
        assert_eq!(listed(&mask, cut), want, "cut {cut:?}");
    }
}

/// The same claim at the byte level: re-encoding the merged list gives the blob
/// a single-block walk would have written, byte for byte.
#[test]
fn the_merged_list_re_encodes_to_the_single_block_blob() {
    let mask = scene();
    let want = set_voxel_rows(mask.view()).expect("the reference blob");
    for cut in CUTS {
        let (env, plan) = run(&mask, cut);
        // One block's blob under the finest cut is not the reference blob; the
        // *merge* of them all is, and that is the whole claim.
        let merged = merge_coordinates(
            VOLUME,
            blobs(&env, &plan)
                .iter()
                .map(|(block, bytes)| (*block, bytes.as_slice()))
                .collect::<Vec<_>>(),
        )
        .expect("the merge");
        let mut rebuilt =
            blockflow::table::RowBuilder::new(std::sync::Arc::new(coordinate_schema()));
        for at in &merged {
            rebuilt.push(*at, &[]).expect("a coordinate row");
        }
        assert_eq!(rebuilt.encode(), want, "cut {cut:?}");
    }
}

/// **The ordering hazard, driven directly.**
///
/// The failure this op exists to prevent is an order that comes from *when a
/// block finished* rather than from *where its voxels are*. Nothing in a normal
/// run makes the two visible apart, because the executor happens to hand blobs
/// over in block order — which is why this feeds them over in three different
/// orders and demands one answer.
///
/// A merge that concatenated in arrival order would pass the block-order case
/// and fail the other two. A merge that sorted only within a blob would fail all
/// three against the reference. Only an order that is a function of the
/// coordinates passes.
#[test]
fn merging_is_insensitive_to_the_order_blobs_arrive() {
    let mask = scene();
    let want = set_voxels(mask.view());
    for cut in CUTS {
        let (env, plan) = run(&mask, cut);
        let held = blobs(&env, &plan);
        let forwards: Vec<([usize; 3], &[u8])> = held
            .iter()
            .map(|(block, bytes)| (*block, bytes.as_slice()))
            .collect();

        let mut backwards = forwards.clone();
        backwards.reverse();

        // A third order that is neither, so that "insensitive" is not two data
        // points on a line: the blobs rotated by a third of the lattice.
        let mut rotated = forwards.clone();
        rotated.rotate_left(forwards.len() / 3);

        for (what, order) in [
            ("block order", &forwards),
            ("reversed", &backwards),
            ("rotated", &rotated),
        ] {
            let merged = merge_coordinates(VOLUME, order.iter().copied()).expect("the merge");
            assert_eq!(merged, want, "cut {cut:?}, blobs handed over in {what}");
        }
    }
}

// ------------------------------------------------------------- empty cases --

#[test]
fn an_all_clear_volume_produces_an_empty_list() {
    let mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    assert!(set_voxels(mask.view()).is_empty());
    for cut in CUTS {
        let (env, plan) = run(&mask, cut);
        // Every block still writes a blob — present and empty is a different
        // fact from absent, and it is the one the coverage guard can check.
        for (block, bytes) in blobs(&env, &plan) {
            assert_eq!(
                encoded_schema(&bytes).expect("a table blob"),
                coordinate_schema(),
                "cut {cut:?}, block {block:?}"
            );
            assert!(entries_of(VOLUME, block, &bytes).is_empty());
        }
        assert!(listed(&mask, cut).is_empty(), "cut {cut:?}");
    }
}

#[test]
fn an_all_set_volume_produces_every_coordinate_in_the_walk_order() {
    let mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), true);
    let want = set_voxels(mask.view());
    assert_eq!(want.len(), VOLUME[0] * VOLUME[1] * VOLUME[2]);
    assert_eq!(want.first(), Some(&[0, 0, 0]));
    assert_eq!(want.last(), Some(&[11, 11, 7]));
    for cut in CUTS {
        assert_eq!(listed(&mask, cut), want, "cut {cut:?}");
    }
}

#[test]
fn a_single_voxel_is_one_entry_wherever_the_seams_fall() {
    for at in [[0, 0, 0], [7, 5, 3], [11, 11, 7]] {
        let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
        mask[at] = true;
        for cut in CUTS {
            assert_eq!(listed(&mask, cut), vec![at], "voxel {at:?}, cut {cut:?}");
        }
    }
}

/// A block with nothing in it contributes nothing **and moves nobody**.
///
/// The second half is the one worth testing, and it is the half a prefix sum
/// makes into a case: here there is no base index for an empty block to have got
/// wrong, so the property holds by construction — and this asserts it anyway,
/// with the empty block found by counting rather than assumed from the scene's
/// description.
#[test]
fn a_block_that_finds_nothing_writes_a_blob_and_shifts_nobody() {
    let mask = scene();
    let (env, plan) = run(&mask, [4, 4, 4]);
    let held = blobs(&env, &plan);
    let empty: Vec<[usize; 3]> = held
        .iter()
        .filter(|(block, bytes)| entries_of(VOLUME, *block, bytes).is_empty())
        .map(|(block, _)| *block)
        .collect();
    assert!(
        empty.contains(&[2, 1, 1]),
        "the scene leaves block [2, 1, 1] clear; empty blocks were {empty:?}"
    );
    assert_eq!(listed(&mask, [4, 4, 4]), set_voxels(mask.view()));
}

// ------------------------------------------------- the prefix sum, guarded --

/// The cheap merge, where it applies: on a lattice that does not interleave, a
/// base index per block plus the rank within the block **is** the position in
/// the walk — and the guarded prefix sum is what says which lattices those are.
#[test]
fn a_base_index_reproduces_the_walk_exactly_where_it_is_allowed_to() {
    let mask = scene();
    let want = set_voxels(mask.view());
    let mut checked = 0;
    for cut in CUTS {
        let (env, plan) = run(&mask, cut);
        let grid = &plan.phases[0].grid;
        let held = blobs(&env, &plan);
        let counts: BTreeMap<[usize; 3], usize> = held
            .iter()
            .map(|(block, bytes)| (*block, entries_of(VOLUME, *block, bytes).len()))
            .collect();

        if !blocks_concatenate_in_order(grid) {
            let error = block_base_indices(grid, &counts)
                .expect_err("an interleaving lattice has no base index")
                .to_string();
            assert!(error.contains("cannot express the walk order"), "{error}");
            continue;
        }

        let bases = block_base_indices(grid, &counts).expect("bases");
        for (block, bytes) in &held {
            let base = bases[block];
            for (rank, at) in entries_of(VOLUME, *block, bytes).into_iter().enumerate() {
                assert_eq!(want[base + rank], at, "cut {cut:?}, block {block:?}");
            }
        }
        checked += 1;
    }
    // Both halves must actually have run: a sweep where every cut took the same
    // branch would assert half of what this test claims.
    assert!(checked > 0, "no cut in the sweep concatenates");
    assert!(checked < CUTS.len(), "no cut in the sweep interleaves");
}
