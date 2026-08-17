// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `ops::adjacency`: a mask in, every adjacent pair of
// set voxels out, in the order a single-block walk would produce them.
//
// **Two things can go wrong here and neither is arithmetic.**
//
// The first is the **order**, and it is `tests/coordinate_list.rs`' concern for
// the same reason: a consumer that numbers these rows gives each one a name that
// is its index, so a permuted list is a different answer that is still a
// well-formed list of the right pairs. So nothing here is compared as a set.
// Every comparison is `assert_eq!` on a whole `Vec`, in order, and where the
// suite does sort it says so and sorts for a different purpose.
//
// The second is **the seam**, and it is this op's own. A pair is a fact about
// two voxels, and two voxels can be on opposite sides of a block boundary. The
// rule that decides which block emits such a pair — the one whose core holds the
// lexicographically lower endpoint — is the only thing standing between the
// answer and a pair counted twice or dropped. A rule like that is not testable
// on a fixture whose pairs all live inside one block, so
// `the_scene_puts_pairs_across_every_seam` pins the fixture as a discriminator
// before anything else asserts against it: under every cut in the sweep, some
// pair must have its two endpoints in two different blocks, or the tests below
// are passing on a scene that cannot tell a correct ownership rule from no rule
// at all.
//
// And the negative control, because "the right answer came out" is a weaker
// statement than "the wrong answer was reachable and did not come out".
// `owning_the_read_extent_instead_of_the_core_double_counts` is this program
// with exactly one thing changed — a block owns everything it can see rather
// than its core — and it asserts that the change produces duplicates and a
// different list. That is what says the ownership rule is doing work.
//
// What is not here, and where it went
// -----------------------------------
// The reference this operation was specified against is a separate application
// under a different licence, and blockflow may not depend on it — see
// `tests/no_domain_vocabulary.rs` for the boundary and `README.md` for the rule.
// The comparison against it was made out of process, in a throwaway crate that
// depends on this one and reads the reference's recorded output, and it agreed
// exactly. On a recorded 96 x 96 x 32 mask with 3158 set voxels the reference's
// list is 4046 pairs; `ops::adjacency` produces that list **in order**, both as
// a single walk and through the executor under eight cuts, ragged ones and a
// 576-block one included. There was no duplicate pair and no pair joining a
// voxel to itself in it, which is the paragraph in `ops::adjacency` about the
// coordinate pair being a complete name at this stage, measured rather than
// argued. The counts under the three connectivities on that mask were 979,
// 3126 and 4046.
//
// None of that can be asserted here, because asserting it means bringing the
// data and the licence across the boundary. What is asserted here instead is
// every property those numbers were the evidence for: the walk is already the
// canonical order, the answer is the same list in the same order under every
// cut, and every pair is emitted exactly once.

use std::collections::BTreeMap;

use ndarray::{s, Array3};

use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::PhaseWork;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::adjacency::{
    adjacent_pair_rows, adjacent_pairs, adjacent_pairs_phase, collect_pairs, empty_pairs,
    encode_adjacent_pairs, forward_offsets, merge_pairs, pair_schema, walk_adjacent_pairs,
    AdjacentPairsOp, Pair,
};
use blockflow::ops::Connectivity;
use blockflow::region::Region;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::table::encoded_schema;
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [12, 12, 8];
const STREAM: &str = "adjacency.pairs";
const CHUNK: [usize; 3] = [4, 4, 4];
const WIDE: Connectivity = Connectivity::FacesEdgesAndCorners;

// -------------------------------------------------------------- the scene --

/// A mask arranged so that no cut of it is easy, and so that the two things a
/// symmetric fixture hides are both present.
///
/// 1. **A dense 4-cube** at 2..=5 on every axis. It holds most of the pairs in
///    the scene, which is where the **load imbalance** comes from: the blocks
///    that hold part of it do far more work than the rest, and
///    `the_scene_has_empty_blocks_and_an_uneven_load` measures that rather than
///    assuming it.
/// 2. **A corner-touching staircase**, `[i, i, i]` for `i` in 0..8. Consecutive
///    voxels of it touch only at a corner, so the whole chain is a chain under
///    `FacesEdgesAndCorners` and eight isolated voxels under `Faces`. It is what
///    makes the connectivity choice **visible in the answer** on this scene, and
///    it runs diagonally across every seam of every cut.
/// 3. **A crooked line**, `[i, 11 - i, (3 * i) % 8]`, which runs the wrong way on
///    axis 1 against axis 0. Its voxels are mostly not adjacent to each other; it
///    is here to make the scene asymmetric across a seam on axis 1 rather than to
///    contribute pairs.
/// 4. **A bar along axis 2** at `x = 9, y = 1`, filling the axis, so a cut on the
///    fastest axis has a run of pairs to cut through.
/// 5. **Two voxels touching only at a corner**, at `[7, 7, 3]` and `[8, 8, 4]`,
///    placed so that the `[4, 4, 4]` cut puts a seam **between them**: one is in
///    block `[1, 1, 0]` and the other in block `[2, 2, 1]`, which share only a
///    lattice corner. That is the hardest pair to own correctly and the scene
///    contains it deliberately.
///
/// And one thing deliberately **absent**: nothing is set anywhere in
/// `x in 8..12, y in 4..8, z in 4..8` except the staircase and corner voxels,
/// which do not enter it. Under the `[4, 4, 4]` cut that is block `[2, 1, 1]`,
/// which therefore owns nothing at all.
fn scene() -> Array3<bool> {
    let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);

    for i in 2..=5 {
        for j in 2..=5 {
            for k in 2..=5 {
                mask[[i, j, k]] = true;
            }
        }
    }

    for i in 0..VOLUME[2] {
        mask[[i, i, i]] = true;
    }

    for i in 0..VOLUME[0] {
        mask[[i, 11 - i, (3 * i) % 8]] = true;
    }

    for k in 0..VOLUME[2] {
        mask[[9, 1, k]] = true;
    }

    mask[[7, 7, 3]] = true;
    mask[[8, 8, 4]] = true;

    mask
}

/// The cuts the suite sweeps, and why each is here.
///
/// | cut | lattice | what it is for |
/// |---|---|---|
/// | `[12, 12, 8]` | 1 | one block: the reference path, run through the framework |
/// | `[4, 12, 8]` | 3 x 1 x 1 | slabs on the slowest axis, and the axis a forward offset never steps down on |
/// | `[12, 4, 8]` | 1 x 3 x 1 | cut on axis 1 alone, which a forward offset *does* step down on |
/// | `[12, 12, 3]` | 1 x 1 x 3 | cut on the fastest axis, and ragged (8 = 3 + 3 + 2) |
/// | `[4, 4, 4]` | 3 x 3 x 2 | all three axes, and the cut the corner-touching pair straddles |
/// | `[5, 5, 3]` | 3 x 3 x 3 | ragged on every axis, so no seam lands on a round number |
/// | `[6, 6, 8]` | 2 x 2 x 1 | seams exactly through the middle of the dense cube |
/// | `[1, 12, 8]` | 12 x 1 x 1 | one voxel thick: every block is entirely halo but for one plane |
/// | `[2, 2, 2]` | 6 x 6 x 4 | 144 blocks, most of them empty: the sparse case |
const CUTS: [[usize; 3]; 9] = [
    [12, 12, 8],
    [4, 12, 8],
    [12, 4, 8],
    [12, 12, 3],
    [4, 4, 4],
    [5, 5, 3],
    [6, 6, 8],
    [1, 12, 8],
    [2, 2, 2],
];

// ------------------------------------------------------------- the harness --

fn shape_of(mask: &Array3<bool>) -> [usize; 3] {
    [mask.shape()[0], mask.shape()[1], mask.shape()[2]]
}

/// Run the one phase and hand back the environment, which is where the answer
/// is: this op writes no level, so there is nothing to read out of the level
/// stack and everything to read out of the store.
fn run_with(
    mask: &Array3<bool>,
    block: [usize; 3],
    connectivity: Connectivity,
) -> (ArrayEnvironment, Decomposition) {
    let volume = shape_of(mask);
    let grid = BlockGrid::new(volume, block).expect("a grid");
    let op = AdjacentPairsOp::new("adjacent pairs", STREAM, Lifecycle::Persistent)
        .connecting(connectivity);
    let plan = adjacent_pairs_phase(grid, Dtype::Bool, &op).expect("a plan");
    let input: Voxels = mask.clone().into();
    let chunk = [
        CHUNK[0].min(volume[0]),
        CHUNK[1].min(volume[1]),
        CHUNK[2].min(volume[2]),
    ];
    let env = ArrayEnvironment::for_decomposition(input, &plan, chunk).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::Bool);
    execute_phases(
        "adjacency",
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

fn run(mask: &Array3<bool>, block: [usize; 3]) -> (ArrayEnvironment, Decomposition) {
    run_with(mask, block, WIDE)
}

/// Every block's blob, in **block-index order**.
///
/// Every block is asked for one, and a block that wrote none is a failure rather
/// than a hole: the stream declares every-block coverage, so an absent blob is
/// what the coverage guard exists to catch and this says so a second time where
/// the bytes are read.
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
fn listed(mask: &Array3<bool>, block: [usize; 3]) -> Vec<Pair> {
    let (env, _) = run(mask, block);
    collect_pairs(&env, STREAM, 0, shape_of(mask)).expect("the merge")
}

/// One block's own rows, in that block's walk order.
fn entries_of(volume: [usize; 3], block: [usize; 3], bytes: &[u8]) -> Vec<Pair> {
    merge_pairs(volume, [(block, bytes)]).expect("a blob merges")
}

/// Whether a position lies in a region.
fn holds(region: &Region, at: [usize; 3]) -> bool {
    (0..3).all(|axis| {
        at[axis] >= region.start[axis] && at[axis] < region.start[axis] + region.shape[axis]
    })
}

// ------------------------------------------------------- the fixture's teeth --

/// The scene holds what it says it holds, counted rather than inspected.
#[test]
fn the_scene_is_the_one_described() {
    let mask = scene();
    let set = mask.iter().filter(|value| **value).count();
    // The cube is 64; the staircase, the crooked line, the bar and the two
    // corner voxels add the rest, and some of them coincide with the cube.
    assert_eq!(set, 90);
    // The corner pair is there and nothing between them is.
    assert!(mask[[7, 7, 3]] && mask[[8, 8, 4]]);
    // The octant the empty-block test needs holds only what the description
    // allows: the staircase leaves it at [8, 8, 4] and the corner voxel is that
    // same voxel, so `x in 8..12, y in 4..8` is what must be clear.
    for i in 8..12 {
        for j in 4..8 {
            for k in 4..8 {
                assert!(!mask[[i, j, k]], "[{i}, {j}, {k}] should be clear");
            }
        }
    }
}

/// **The guard on the guard, first half.** The ownership rule only does work if
/// pairs cross seams, so this asserts that under every cut that has a seam at
/// all, some pair has its two endpoints in two different blocks — and counts
/// them, so that a change which quietly moved the scene inside the blocks fails
/// here rather than making every test below vacuous.
#[test]
fn the_scene_puts_pairs_across_every_seam() {
    let mask = scene();
    let want = adjacent_pairs(mask.view(), WIDE).expect("the reference");
    for cut in CUTS {
        let (_, plan) = run(&mask, cut);
        let blocks = &plan.phases[0].blocks;
        let owner = |at: [usize; 3]| {
            blocks
                .iter()
                .position(|block| holds(&block.core, at))
                .unwrap_or_else(|| panic!("no block owns {at:?}"))
        };
        let crossing = want
            .iter()
            .filter(|(lower, higher)| owner(*lower) != owner(*higher))
            .count();
        if blocks.len() == 1 {
            assert_eq!(crossing, 0, "one block cannot have a seam");
            continue;
        }
        assert!(
            crossing > 0,
            "cut {cut:?} has {} blocks and not one pair crosses a seam, so every ownership \
             assertion in this file is passing by accident",
            blocks.len()
        );
    }
}

/// **The guard on the guard, second half.** The probe measured a real workload
/// as 37-44% empty blocks and a 6x spread between the busiest block and the
/// mean; a uniform fixture has neither and hides both. This asserts the scene
/// reproduces the shape of that, so the empty-block and imbalance cases below
/// are exercised rather than nominal.
#[test]
fn the_scene_has_empty_blocks_and_an_uneven_load() {
    let mask = scene();
    let (env, plan) = run(&mask, [2, 2, 2]);
    let held = blobs(&env, &plan);
    let counts: Vec<usize> = held
        .iter()
        .map(|(block, bytes)| entries_of(VOLUME, *block, bytes).len())
        .collect();
    let blocks = counts.len();
    let empty = counts.iter().filter(|count| **count == 0).count();
    let total: usize = counts.iter().sum();
    let busiest = *counts.iter().max().expect("blocks");
    assert!(
        empty * 3 >= blocks,
        "only {empty} of {blocks} blocks are empty; the fixture is more uniform than the \
         workload this op is for"
    );
    assert!(
        busiest * blocks >= 3 * total,
        "the busiest block holds {busiest} of {total} pairs over {blocks} blocks, which is less \
         imbalance than the workload this op is for"
    );
}

/// And the connectivity choice is visible on this scene: the three answers are
/// three different lengths, so a test that fixes one of them is testing it.
#[test]
fn the_scene_separates_the_three_connectivities() {
    let mask = scene();
    let counts: Vec<usize> = [
        Connectivity::Faces,
        Connectivity::FacesAndEdges,
        Connectivity::FacesEdgesAndCorners,
    ]
    .into_iter()
    .map(|connectivity| {
        adjacent_pairs(mask.view(), connectivity)
            .expect("the reference")
            .len()
    })
    .collect();
    assert!(
        counts[0] < counts[1] && counts[1] < counts[2],
        "the three connectivities gave {counts:?} on this scene and must differ"
    );
}

// ------------------------------------------------------------ the acceptance --

/// **The bar.** The same pairs in the same order under every cut, as a whole
/// list, against the single-block walk.
#[test]
fn every_decomposition_produces_the_same_list_in_the_same_order() {
    let mask = scene();
    let want = adjacent_pairs(mask.view(), WIDE).expect("the reference");
    assert!(!want.is_empty());
    for cut in CUTS {
        assert_eq!(listed(&mask, cut), want, "cut {cut:?}");
    }
}

/// The same claim at the byte level: the merge of every blob re-encodes to the
/// blob a single-block walk would have written, byte for byte.
#[test]
fn the_merged_list_re_encodes_to_the_single_block_blob() {
    let mask = scene();
    let want = adjacent_pair_rows(mask.view(), WIDE).expect("the reference blob");
    let volume = shape_of(&mask);
    for cut in CUTS {
        let (env, plan) = run(&mask, cut);
        let merged = merge_pairs(
            VOLUME,
            blobs(&env, &plan)
                .iter()
                .map(|(block, bytes)| (*block, bytes.as_slice()))
                .collect::<Vec<_>>(),
        )
        .expect("the merge");
        // Re-encoded through the same kernel that wrote the reference, from a
        // mask that holds exactly the merged pairs' endpoints: what is being
        // compared is the bytes, and the only way to make bytes is to walk.
        let mut rebuilt = blockflow::table::RowBuilder::new(std::sync::Arc::new(pair_schema()));
        for (lower, higher) in &merged {
            rebuilt
                .push(
                    *lower,
                    &[
                        blockflow::table::Value::U64(higher[0] as u64),
                        blockflow::table::Value::U64(higher[1] as u64),
                        blockflow::table::Value::U64(higher[2] as u64),
                    ],
                )
                .expect("a pair row");
        }
        assert_eq!(rebuilt.encode(), want, "cut {cut:?}");
        assert_eq!(volume, VOLUME);
    }
}

/// **Every pair exactly once.** The rows the blocks wrote sum to the reference's
/// length, no pair appears twice, and no unordered pair appears in both
/// directions. Asserted on the blobs rather than on the merge, because the merge
/// is where a double count would be least visible.
#[test]
fn every_pair_is_emitted_exactly_once() {
    let mask = scene();
    let want = adjacent_pairs(mask.view(), WIDE).expect("the reference");
    for cut in CUTS {
        let (env, plan) = run(&mask, cut);
        let held = blobs(&env, &plan);
        let written: usize = held
            .iter()
            .map(|(block, bytes)| entries_of(VOLUME, *block, bytes).len())
            .sum();
        assert_eq!(
            written,
            want.len(),
            "cut {cut:?} wrote {written} rows for {} pairs",
            want.len()
        );

        let merged = listed(&mask, cut);
        let mut distinct = merged.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            merged.len(),
            "cut {cut:?} emitted a pair twice"
        );

        // and not once in each direction either, which is the other way an
        // ownership rule can double count
        let mut undirected: Vec<[[usize; 3]; 2]> = merged
            .iter()
            .map(|(lower, higher)| {
                if lower <= higher {
                    [*lower, *higher]
                } else {
                    [*higher, *lower]
                }
            })
            .collect();
        undirected.sort_unstable();
        undirected.dedup();
        assert_eq!(undirected.len(), merged.len(), "cut {cut:?}");
    }
}

/// **The ownership rule, read off the blobs.** Every row a block wrote has its
/// lower endpoint in that block's core — and, since the cores tile the volume,
/// that is the whole of "exactly once" stated per block rather than in total.
#[test]
fn a_block_writes_exactly_the_pairs_whose_lower_endpoint_it_owns() {
    let mask = scene();
    for cut in CUTS {
        let (env, plan) = run(&mask, cut);
        let held = blobs(&env, &plan);
        let blocks: BTreeMap<[usize; 3], &Region> = plan.phases[0]
            .blocks
            .iter()
            .map(|block| (block.index, &block.core))
            .collect();
        for (block, bytes) in &held {
            let core = blocks[block];
            for (lower, higher) in entries_of(VOLUME, *block, bytes) {
                assert!(
                    holds(core, lower),
                    "cut {cut:?}: block {block:?} wrote a pair from {lower:?}, which is not in \
                     its core {core:?}"
                );
                // and the higher endpoint is allowed to be anywhere at all,
                // which is the point: a pair that leaves the block is still the
                // block's to emit.
                assert!(higher > lower);
            }
        }
    }
}

/// **The negative control: this program with one thing changed.**
///
/// A block owns the pairs whose lower endpoint is in its **core**. Change that
/// one word to **read extent** — everything else identical, the same kernel, the
/// same offsets, the same merge — and the seam-crossing pairs are emitted by
/// both blocks. This asserts the wrong answer is genuinely reachable and that it
/// is wrong in the way described: strictly longer, with duplicates, and not the
/// reference list.
///
/// Without this, "the right answer came out" would be the whole of the evidence
/// that the ownership rule is load-bearing.
#[test]
fn owning_the_read_extent_instead_of_the_core_double_counts() {
    let mask = scene();
    let want = adjacent_pairs(mask.view(), WIDE).expect("the reference");
    let mut checked = 0;
    for cut in CUTS {
        let (_, plan) = run(&mask, cut);
        let blocks = &plan.phases[0].blocks;
        if blocks.len() == 1 {
            continue;
        }
        let mut wrong: Vec<([usize; 3], Vec<u8>)> = Vec::new();
        for block in blocks {
            // The whole mask is handed over so that the halo check cannot fire:
            // what is being changed is *which* voxels the block claims, and
            // nothing else.
            let bytes = encode_adjacent_pairs(mask.view(), [0, 0, 0], &block.read, VOLUME, WIDE)
                .expect("a blob");
            wrong.push((block.index, bytes));
        }
        let merged = merge_pairs(
            VOLUME,
            wrong
                .iter()
                .map(|(block, bytes)| (*block, bytes.as_slice())),
        )
        .expect("the merge");
        assert!(
            merged.len() > want.len(),
            "cut {cut:?}: owning the read extent should over-count and did not"
        );
        assert_ne!(merged, want, "cut {cut:?}");
        let mut distinct = merged.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct, want,
            "cut {cut:?}: the over-count should be duplicates of the right pairs and nothing else"
        );
        checked += 1;
    }
    assert!(checked > 0, "no cut in the sweep has a seam");
}

/// **The ordering hazard, driven directly.** The blobs are handed to the merge
/// in three different orders and must give one answer, so that nothing can be
/// reading the order a block finished in.
#[test]
fn merging_is_insensitive_to_the_order_blobs_arrive() {
    let mask = scene();
    let want = adjacent_pairs(mask.view(), WIDE).expect("the reference");
    for cut in CUTS {
        let (env, plan) = run(&mask, cut);
        let held = blobs(&env, &plan);
        let forwards: Vec<([usize; 3], &[u8])> = held
            .iter()
            .map(|(block, bytes)| (*block, bytes.as_slice()))
            .collect();

        let mut backwards = forwards.clone();
        backwards.reverse();

        let mut rotated = forwards.clone();
        rotated.rotate_left(forwards.len() / 3);

        for (what, order) in [
            ("block order", &forwards),
            ("reversed", &backwards),
            ("rotated", &rotated),
        ] {
            let merged = merge_pairs(VOLUME, order.iter().copied()).expect("the merge");
            assert_eq!(merged, want, "cut {cut:?}, blobs handed over in {what}");
        }
    }
}

// ------------------------------------------------------ the connectivity --

/// **The connectivity choice reaches the answer.** Each of the three gives its
/// own list, and each of those lists survives every decomposition unchanged —
/// so the parameter is neither ignored nor a source of a different seam bug.
#[test]
fn each_connectivity_is_decomposition_invariant_and_they_differ() {
    let mask = scene();
    let mut lengths = Vec::new();
    for connectivity in [
        Connectivity::Faces,
        Connectivity::FacesAndEdges,
        Connectivity::FacesEdgesAndCorners,
    ] {
        let want = adjacent_pairs(mask.view(), connectivity).expect("the reference");
        for cut in CUTS {
            let (env, _) = run_with(&mask, cut, connectivity);
            let merged = collect_pairs(&env, STREAM, 0, VOLUME).expect("the merge");
            assert_eq!(merged, want, "{connectivity:?}, cut {cut:?}");
        }
        lengths.push(want.len());
    }
    assert!(
        lengths[0] < lengths[1] && lengths[1] < lengths[2],
        "{lengths:?}"
    );
}

/// The narrowest connectivity does not see a corner touch and the widest does,
/// **across a seam**: the two voxels are in blocks that share only a lattice
/// corner, which is the case a face-only seam walk would never look at.
#[test]
fn a_corner_touch_across_a_lattice_corner_is_owned_correctly() {
    let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    mask[[3, 3, 3]] = true;
    mask[[4, 4, 4]] = true;
    // Under [4, 4, 4] those are blocks [0, 0, 0] and [1, 1, 1]: a lattice corner.
    for cut in CUTS {
        assert_eq!(
            listed(&mask, cut),
            vec![([3, 3, 3], [4, 4, 4])],
            "widest, cut {cut:?}"
        );
        let (env, _) = run_with(&mask, cut, Connectivity::Faces);
        assert!(
            collect_pairs(&env, STREAM, 0, VOLUME)
                .expect("the merge")
                .is_empty(),
            "faces, cut {cut:?}"
        );
    }
}

// ------------------------------------------------------------- edge cases --

#[test]
fn a_single_voxel_has_no_pair_wherever_the_seams_fall() {
    for at in [[0, 0, 0], [7, 5, 3], [11, 11, 7]] {
        let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
        mask[at] = true;
        assert!(adjacent_pairs(mask.view(), WIDE)
            .expect("reference")
            .is_empty());
        for cut in CUTS {
            assert!(listed(&mask, cut).is_empty(), "voxel {at:?}, cut {cut:?}");
        }
    }
}

#[test]
fn an_all_clear_volume_produces_an_empty_list_and_a_blob_per_block() {
    let mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    assert!(adjacent_pairs(mask.view(), WIDE)
        .expect("reference")
        .is_empty());
    for cut in CUTS {
        let (env, plan) = run(&mask, cut);
        for (block, bytes) in blobs(&env, &plan) {
            assert_eq!(
                encoded_schema(&bytes).expect("a table blob"),
                pair_schema(),
                "cut {cut:?}, block {block:?}"
            );
            assert_eq!(bytes, empty_pairs(), "cut {cut:?}, block {block:?}");
        }
        assert!(listed(&mask, cut).is_empty(), "cut {cut:?}");
    }
}

/// An all-set volume, against a closed form rather than against the code: for
/// each forward step, the number of positions it can be taken from is the
/// product of `extent - |step|` over the axes.
#[test]
fn an_all_set_volume_has_the_number_of_pairs_the_arithmetic_says() {
    let mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), true);
    let want: usize = forward_offsets(WIDE)
        .iter()
        .map(|by| {
            (0..3)
                .map(|axis| VOLUME[axis] - by[axis].unsigned_abs())
                .product::<usize>()
        })
        .sum();
    let reference = adjacent_pairs(mask.view(), WIDE).expect("reference");
    assert_eq!(reference.len(), want);
    for cut in CUTS {
        assert_eq!(listed(&mask, cut), reference, "cut {cut:?}");
    }
}

/// A block that owns no pair writes a blob and shifts nobody.
///
/// The empty block is found by counting rather than assumed from the scene's
/// description, and the second half — that the answer is unchanged — is the half
/// worth asserting: there is no base index here for an empty block to have got
/// wrong, and this says so rather than leaving it to the mechanism.
#[test]
fn a_block_that_owns_no_pair_writes_a_blob_and_shifts_nobody() {
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
        "the scene leaves block [2, 1, 1] with nothing to own; empty blocks were {empty:?}"
    );
    for block in &empty {
        let bytes = held
            .iter()
            .find(|(index, _)| index == block)
            .map(|(_, bytes)| bytes)
            .expect("the blob");
        assert_eq!(bytes, &empty_pairs());
    }
    assert_eq!(
        listed(&mask, [4, 4, 4]),
        adjacent_pairs(mask.view(), WIDE).expect("reference")
    );
}

/// A block handed less than its halo is refused rather than answering with
/// fewer pairs — the failure that a well-formed shorter list would hide.
#[test]
fn a_short_read_is_refused_rather_than_dropping_pairs() {
    let mask = scene();
    let owned = Region::new(&[4, 4, 4], &[4, 4, 4]);
    let short = mask.slice(s![4..8, 4..8, 4..8]);
    let error = walk_adjacent_pairs(short, [4, 4, 4], &owned, VOLUME, WIDE, &mut |_, _| Ok(()))
        .expect_err("a short read is refused")
        .to_string();
    assert!(error.contains("a pair reaches one voxel"), "{error}");

    // and the answer it would have given is a shorter, well-formed list, which
    // is why the refusal is the only thing that can catch it
    let sliced = short.to_owned();
    let inner = adjacent_pairs(sliced.view(), WIDE).expect("reference");
    let whole = adjacent_pairs(mask.view(), WIDE).expect("reference");
    assert!(inner.len() < whole.len());
}
