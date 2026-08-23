// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `ops::components::Connectivity` — the choice of
// which of the twenty-six voxels around one count as adjacent, and the merge
// that has to agree with it across a block seam.
//
// What is hard about this, and why the obvious fixture proves nothing
// -------------------------------------------------------------------
// A blocked connected-component labelling has a failure mode a local op does
// not: **it can be right on easy data and wrong on hard data at the same block
// size.** For connectivity the easy data is anything whose parts touch face to
// face, because a face join is made by every connectivity there is. A fixture
// built out of boxes and blobs gives the same answer under 6, 18 and 26, and
// would pass with the parameter ignored entirely, with the wider seam pairs
// never generated, and with the twelve lattice-edge and eight lattice-corner
// meetings absent from the walk.
//
// So every fixture here is chosen for what it can *discriminate*, and the
// discriminating shape is a **diagonal**: two voxels that touch only at a
// corner are one component under 26 and two under 6 and 18; two that touch only
// along an edge are one under 18 and 26 and two under 6. Each fixture below
// states which of the three it separates, and the suite asserts the separation
// rather than assuming it — if a fixture ever stops discriminating, the
// assertion that it does fails and says so instead of passing quietly.
//
// The three ways a join can cross a seam, all of which are exercised
// -----------------------------------------------------------------
// A wider connectivity does not only widen the pairing *within* a seam. It also
// adds seams, because two blocks that share only a lattice edge or only a
// lattice corner now have voxels that touch. All three are here, and which one
// a fixture exercises is a function of the block size rather than of the mask,
// which is why the same masks are run at block sizes cutting one, two and three
// axes:
//
// | blocks cut | the pair `(3,3,3)-(4,4,4)` crosses | what the walk must do |
// |---|---|---|
// | axis 0 only | one face seam | pair a voxel with a 3x3 window of the opposite face |
// | axes 0 and 1 | a lattice edge | read a *row* of a face, and pair along the free axis |
// | axes 0, 1 and 2 | a lattice corner | read a single *entry* of a face |
//
// The reference
// -------------
// `REFERENCE_*` below are `scipy.ndimage.label` on the same mask under
// `generate_binary_structure(3, 1 | 2 | 3)`, taken out of process and pinned
// here as literals — the numbering canonicalised the way this crate's is, by
// where each component's lowest voxel sits in a row-major scan. It is out of
// process because there is no in-process reference to reach: `scipy` is not a
// dependency of this crate and must not become one.

use std::collections::BTreeMap;

use ndarray::{s, Array3};

use blockflow::dtype::Dtype;
use blockflow::fragment::neighbourhood;
use blockflow::geometry::BlockGrid;
use blockflow::ops::components::{
    label_members_into, label_members_into_with, planes_of, walk_seams, walk_seams_with,
    Connectivity, FacePlanes, LabelIndex, Union,
};
use blockflow::ops::detect::{detect_phases, label_regions_into, LabelRegionsOp, RegionPointsOp};
use blockflow::ops::fill::{fill_phases, label_background_into, FillHolesOp, LabelBackgroundOp};
use blockflow::ops::regional::{regional_phases, LabelPlateauxOp, RegionalMaximaOp};
use blockflow::sidecar::Lifecycle;

const EVERY: [Connectivity; 3] = [
    Connectivity::Faces,
    Connectivity::FacesAndEdges,
    Connectivity::FacesEdgesAndCorners,
];

// ------------------------------------------------------------ the machine --

/// What one block tells the merge: how many components it found, and its six
/// face planes. The same pair `ops::fill` and `ops::detect` carry inside their
/// own fragment types, with none of their per-label payload — the payload is
/// the op's question and this suite is about the geometry.
struct Report {
    labels: u32,
    faces: FacePlanes,
}

/// The whole-volume answer: one array, one labelling, one pass, and the thing
/// every blocked answer below is measured against.
fn whole(mask: &Array3<bool>, connectivity: Connectivity) -> Array3<u32> {
    let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    let mut out = Array3::<u32>::zeros(mask.raw_dim());
    label_members_into_with(shape, connectivity, |at| mask[at], out.view_mut()).expect("a shape");
    out
}

/// The blocked answer: label each block on its own, close the local labels
/// across every seam, and write the components back out.
///
/// Returns the labelling and how many block-local components there were before
/// the merge, which is what makes "the merge is load-bearing" an assertion
/// rather than a hope.
fn blocked(
    mask: &Array3<bool>,
    block: [usize; 3],
    connectivity: Connectivity,
) -> (Array3<u32>, usize) {
    let volume = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    let grid = BlockGrid::new(volume, block).expect("a lattice");
    let counts = grid.blocks_per_axis();
    let edge = grid.block();

    let mut local = Array3::<u32>::zeros(mask.raw_dim());
    let mut reports: BTreeMap<[usize; 3], Report> = BTreeMap::new();
    for core in grid.cores() {
        let start = core.core.start.clone();
        let shape = core.core.shape.clone();
        let inside = mask.slice(s![
            start[0]..start[0] + shape[0],
            start[1]..start[1] + shape[1],
            start[2]..start[2] + shape[2],
        ]);
        let mut labels = Array3::<u32>::zeros((shape[0], shape[1], shape[2]));
        let found = label_members_into_with(
            [shape[0], shape[1], shape[2]],
            connectivity,
            |at| inside[at],
            labels.view_mut(),
        )
        .expect("a shape");
        reports.insert(
            core.index,
            Report {
                labels: found,
                faces: planes_of(labels.view()),
            },
        );
        local
            .slice_mut(s![
                start[0]..start[0] + shape[0],
                start[1]..start[1] + shape[1],
                start[2]..start[2] + shape[2],
            ])
            .assign(&labels);
    }

    let index = LabelIndex::build(&reports, counts, |report| report.labels).expect("every block");
    let before = index.total();
    let mut sets = Union::new(before);
    walk_seams_with(
        &reports,
        counts,
        &index,
        connectivity,
        |report| &report.faces,
        |a, b| sets.union(a, b),
    )
    .expect("one lattice");

    let mut out = Array3::<u32>::zeros(mask.raw_dim());
    for i in 0..volume[0] {
        for j in 0..volume[1] {
            for k in 0..volume[2] {
                let label = local[[i, j, k]];
                if label == 0 {
                    continue;
                }
                let at = [i / edge[0], j / edge[1], k / edge[2]];
                out[[i, j, k]] = sets.find(index.node(at, label)) as u32 + 1;
            }
        }
    }
    (canonical(&out), before)
}

/// Renumber components by where their lowest voxel is in a row-major scan,
/// which is the numbering `label_members_into_with` promises. Two labellings
/// that describe the same partition are then the same bytes.
fn canonical(labels: &Array3<u32>) -> Array3<u32> {
    let mut seen: BTreeMap<u32, u32> = BTreeMap::new();
    let mut out = Array3::<u32>::zeros(labels.raw_dim());
    for i in 0..labels.shape()[0] {
        for j in 0..labels.shape()[1] {
            for k in 0..labels.shape()[2] {
                let label = labels[[i, j, k]];
                if label == 0 {
                    continue;
                }
                let next = seen.len() as u32 + 1;
                out[[i, j, k]] = *seen.entry(label).or_insert(next);
            }
        }
    }
    out
}

fn components(labels: &Array3<u32>) -> u32 {
    labels.iter().copied().max().unwrap_or(0)
}

fn mask_of(shape: [usize; 3], voxels: &[[usize; 3]]) -> Array3<bool> {
    let mut mask = Array3::from_elem((shape[0], shape[1], shape[2]), false);
    for &at in voxels {
        mask[at] = true;
    }
    mask
}

/// The labelling as one digit per voxel in row-major order, which is the form
/// the reference below is pinned in.
fn digits(labels: &Array3<u32>) -> String {
    labels
        .iter()
        .map(|&label| {
            assert!(label < 10, "the reference fixture stays in one digit");
            char::from_digit(label, 10).expect("a digit")
        })
        .collect()
}

// ---------------------------------------------------- the face-connected path --

/// **The pairs a face-connected walk hands to `meet` have not moved.**
///
/// Not "the answer is the same" — the *sequence*, in order, against an oracle
/// written here as the two zipped face planes and nothing else. That is the
/// whole of what `walk_seams` was, so if the generalised walk emits it exactly
/// then no op built on the face-connected path can tell that the parameter
/// exists. The lattice is 3 x 2 x 2 with distinct labels everywhere, so every
/// seam carries traffic and a dropped or duplicated pair shows up as a length
/// mismatch rather than as a coincidence.
#[test]
fn the_face_connected_seam_walk_emits_exactly_the_pairs_it_always_did() {
    const LABELS: u32 = 5;
    let counts = [3usize, 2, 2];
    let block = [3usize, 4, 5];
    let mut dense: BTreeMap<[usize; 3], Report> = BTreeMap::new();
    for i in 0..counts[0] {
        for j in 0..counts[1] {
            for k in 0..counts[2] {
                // Labels 1..=5 spread over the block, offset per block so that
                // no two blocks put the same label in the same place, plus a
                // scattering of unlabelled voxels so that the "a pair with no
                // component is no pair" rule is exercised rather than assumed.
                let skew = i + 2 * j + 3 * k;
                let mut labels = Array3::<u32>::zeros((block[0], block[1], block[2]));
                for (slot, cell) in labels.iter_mut().enumerate() {
                    *cell = if (slot + skew) % 7 == 3 {
                        0
                    } else {
                        ((slot + skew) % LABELS as usize) as u32 + 1
                    };
                }
                dense.insert(
                    [i, j, k],
                    Report {
                        labels: LABELS,
                        faces: planes_of(labels.view()),
                    },
                );
            }
        }
    }
    let index = LabelIndex::build(&dense, counts, |report| report.labels).expect("every block");

    let mut oracle: Vec<(usize, usize)> = Vec::new();
    for &at in index.order() {
        for axis in 0..3 {
            let mut ahead = at;
            ahead[axis] += 1;
            if ahead[axis] >= counts[axis] {
                continue;
            }
            let here = &dense[&at].faces[axis * 2 + 1];
            let there = &dense[&ahead].faces[axis * 2];
            for (&a, &b) in here.1.iter().zip(there.1.iter()) {
                if a == 0 || b == 0 {
                    continue;
                }
                oracle.push((index.node(at, a), index.node(ahead, b)));
            }
        }
    }
    assert!(
        oracle.len() > 100,
        "the fixture must actually generate traffic, and generated {}",
        oracle.len()
    );

    let mut bare: Vec<(usize, usize)> = Vec::new();
    walk_seams(
        &dense,
        counts,
        &index,
        |r| &r.faces,
        |a, b| bare.push((a, b)),
    )
    .expect("one lattice");
    let mut stated: Vec<(usize, usize)> = Vec::new();
    walk_seams_with(
        &dense,
        counts,
        &index,
        Connectivity::Faces,
        |r| &r.faces,
        |a, b| stated.push((a, b)),
    )
    .expect("one lattice");

    assert_eq!(bare, oracle, "the bare walk is the two zipped faces");
    assert_eq!(stated, oracle, "and so is the walk at its default");

    // The wider walks are supersets of it and are strictly wider on this
    // lattice, which is what says the parameter reaches the walk at all.
    let mut wide: Vec<(usize, usize)> = Vec::new();
    walk_seams_with(
        &dense,
        counts,
        &index,
        Connectivity::FacesEdgesAndCorners,
        |r| &r.faces,
        |a, b| wide.push((a, b)),
    )
    .expect("one lattice");
    assert!(wide.len() > oracle.len() * 3, "26 pairs far more widely");
    for pair in &oracle {
        assert!(
            wide.contains(pair),
            "{pair:?} is missing from the wide walk"
        );
    }
}

/// The ops built on this module are still six-connected, asked in the way that
/// can tell: a corner-touching pair.
#[test]
fn the_ops_built_on_this_module_are_unmoved_and_still_face_connected() {
    let mask = mask_of([4, 4, 4], &[[1, 1, 1], [2, 2, 2]]);
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    assert_eq!(
        label_regions_into(mask.view(), labels.view_mut()).expect("a shape"),
        2,
        "`detect` labels a corner-touching pair as two regions"
    );

    // and `fill`'s background labelling likewise: a mask that is set everywhere
    // except two corner-touching voxels leaves two background components.
    let mut solid = Array3::from_elem((4, 4, 4), true);
    solid[[1, 1, 1]] = false;
    solid[[2, 2, 2]] = false;
    let mut background = Array3::<u32>::zeros(solid.raw_dim());
    assert_eq!(
        label_background_into(solid.view(), background.view_mut()).expect("a shape"),
        2
    );

    // the bare labelling agrees with the parameterised one at its default
    let mut bare = Array3::<u32>::zeros(mask.raw_dim());
    label_members_into([4, 4, 4], |at| mask[at], bare.view_mut()).expect("a shape");
    assert_eq!(bare, whole(&mask, Connectivity::Faces));
}

/// The plans of the three ops built on this module, pinned.
///
/// A fingerprint is what a resumed run compares against, so a plan that moves
/// silently is a run that cannot be resumed. The **connectivity** is consumed at
/// execution time and appears in no declaration, so it moves none of these
/// numbers, and that is what this test was written to check.
///
/// **All three moved once**, when the ops migrated onto `FragmentOp::barrier`:
/// a barrier is part of the plan and is hashed, and relieving the halo changes
/// the phase's geometry, so the fingerprints *should* have moved and a test that
/// had not noticed would have been the defect. They are pinned separately rather
/// than as one tuple, so that the next op to move says which one it was.
#[test]
fn the_plans_of_the_ops_built_on_this_module_fingerprint_as_they_did() {
    const VOLUME: [usize; 3] = [24, 16, 12];
    const BLOCK: [usize; 3] = [8, 8, 6];
    const STREAM: &str = "faces";
    const POINTS: &str = "points";

    let grid = BlockGrid::new(VOLUME, BLOCK).expect("a lattice");
    let label = LabelBackgroundOp::new("label", STREAM, Lifecycle::DeleteOnExit);
    let fill = FillHolesOp::new("fill", STREAM, 0, Dtype::Bool, &grid);
    let filling = fill_phases(grid, Dtype::Bool, &label, &fill).expect("a plan");

    let grid = BlockGrid::new(VOLUME, BLOCK).expect("a lattice");
    let plateaux = LabelPlateauxOp::new("label", STREAM, Lifecycle::DeleteOnExit);
    let maxima = RegionalMaximaOp::new("maxima", STREAM, 0, Dtype::Bool, &grid);
    let regional = regional_phases(grid, Dtype::F64, &plateaux, &maxima).expect("a plan");

    let grid = BlockGrid::new(VOLUME, BLOCK).expect("a lattice");
    let regions = LabelRegionsOp::new("label", STREAM, Lifecycle::DeleteOnExit);
    let points = RegionPointsOp::new("points", STREAM, 0, POINTS, Lifecycle::Persistent, &grid);
    let detecting = detect_phases(grid, Dtype::Bool, &regions, &points).expect("a plan");

    assert_eq!(
        filling.fingerprint(),
        FILL_FINGERPRINT,
        "ops::fill's plan moved"
    );
    assert_eq!(
        regional.fingerprint(),
        REGIONAL_FINGERPRINT,
        "ops::regional's plan moved"
    );
    assert_eq!(
        detecting.fingerprint(),
        DETECT_FINGERPRINT,
        "ops::detect's plan moved"
    );

    // and the three are genuinely different plans, so an accidental constant
    // that happened to match one of them would not match all three
    let all: Vec<u64> = vec![
        filling.fingerprint(),
        regional.fingerprint(),
        detecting.fingerprint(),
    ];
    let mut sorted = all.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), all.len());
}

const FILL_FINGERPRINT: u64 = 1_507_721_539_040_108_903;
const REGIONAL_FINGERPRINT: u64 = 4_161_838_864_743_738_177;
const DETECT_FINGERPRINT: u64 = 15_254_762_488_441_970_327;

// --------------------------------------------------------- discrimination --

/// **The discriminating case, on both sides of a seam and within one block.**
///
/// Two voxels diagonal to each other are one component under 26 and two under
/// 6. Under a block size that puts them in the same block that is a question
/// about the labelling; under one that puts them in different blocks it is a
/// question about the merge, and the two must answer alike.
#[test]
fn two_voxels_diagonal_across_a_seam_are_one_component_under_twenty_six_and_two_under_six() {
    let mask = mask_of([4, 4, 4], &[[1, 1, 1], [2, 2, 2]]);

    // within one block
    assert_eq!(components(&whole(&mask, Connectivity::Faces)), 2);
    assert_eq!(components(&whole(&mask, Connectivity::FacesAndEdges)), 2);
    assert_eq!(
        components(&whole(&mask, Connectivity::FacesEdgesAndCorners)),
        1
    );

    // and across a seam, cut every way the pair can be cut
    for (block, crossing) in [
        ([2usize, 4, 4], "one face seam"),
        ([2, 2, 4], "a lattice edge"),
        ([2, 2, 2], "a lattice corner"),
    ] {
        let (six, _) = blocked(&mask, block, Connectivity::Faces);
        let (eighteen, _) = blocked(&mask, block, Connectivity::FacesAndEdges);
        let (twenty_six, _) = blocked(&mask, block, Connectivity::FacesEdgesAndCorners);
        assert_eq!(components(&six), 2, "6 keeps them apart across {crossing}");
        assert_eq!(components(&eighteen), 2, "so does 18, across {crossing}");
        assert_eq!(
            components(&twenty_six),
            1,
            "26 joins them across {crossing}"
        );
        for connectivity in EVERY {
            assert_eq!(
                blocked(&mask, block, connectivity).0,
                whole(&mask, connectivity),
                "{connectivity:?} across {crossing} disagreed with the whole volume"
            );
        }
    }
}

/// The same, for the join that separates 18 from 6: two voxels sharing only an
/// edge.
///
/// Present because without it the offset predicate is only ever asked "exactly
/// one step?" or "anything at all?" and is blind at its own boundary — which is
/// the argument for offering eighteen at all.
#[test]
fn two_voxels_sharing_an_edge_across_a_seam_join_under_eighteen_but_not_under_six() {
    let mask = mask_of([4, 4, 4], &[[1, 1, 0], [2, 2, 0]]);

    assert_eq!(components(&whole(&mask, Connectivity::Faces)), 2);
    assert_eq!(components(&whole(&mask, Connectivity::FacesAndEdges)), 1);

    for (block, crossing) in [
        ([2usize, 4, 4], "one face seam"),
        ([2, 2, 4], "a lattice edge"),
        ([2, 2, 2], "a lattice edge, with the third axis cut too"),
    ] {
        assert_eq!(
            components(&blocked(&mask, block, Connectivity::Faces).0),
            2,
            "6 keeps them apart across {crossing}"
        );
        assert_eq!(
            components(&blocked(&mask, block, Connectivity::FacesAndEdges).0),
            1,
            "18 joins them across {crossing}"
        );
        for connectivity in EVERY {
            assert_eq!(
                blocked(&mask, block, connectivity).0,
                whole(&mask, connectivity),
                "{connectivity:?} across {crossing} disagreed with the whole volume"
            );
        }
    }
}

// ------------------------------------------------- decomposition invariance --

/// Two chains of single voxels, each stepping diagonally, and nothing that
/// touches face to face at all.
///
/// * a **corner chain** at `(t, t, t)`: consecutive voxels share only a corner,
///   so it is one component under 26 and twelve under 18 and 6;
/// * an **edge chain** at `(t, t, 20)`: consecutive voxels share only an edge,
///   so it is one component under 18 and 26 and twelve under 6.
///
/// A chain rather than a pair because a chain crosses a seam *whatever the
/// block size*, so the same fixture exercises every cut without being built for
/// one. The two are nine voxels apart on the last axis, which is more than any
/// connectivity here can span, so neither can leak into the other and the
/// counts below are a statement about each separately.
fn chains() -> Array3<bool> {
    let mut voxels: Vec<[usize; 3]> = Vec::new();
    for t in 0..12 {
        voxels.push([t, t, t]);
        voxels.push([t, t, 20]);
    }
    mask_of([12, 12, 24], &voxels)
}

#[test]
fn the_chains_discriminate_all_three_connectivities() {
    assert_eq!(
        components(&whole(&chains(), Connectivity::Faces)),
        24,
        "6 joins nothing: twelve voxels in each chain"
    );
    assert_eq!(
        components(&whole(&chains(), Connectivity::FacesAndEdges)),
        13,
        "18 joins the edge chain and leaves the corner chain in twelve"
    );
    assert_eq!(
        components(&whole(&chains(), Connectivity::FacesEdgesAndCorners)),
        2,
        "26 joins both chains"
    );
}

/// **Byte-identical to the whole-volume answer under every decomposition**,
/// for all three connectivities.
///
/// The block sizes are chosen to cut one, two and three axes, to divide the
/// volume evenly and to divide it unevenly, and to go down to a two-voxel block
/// where almost every join is a merge rather than a labelling.
#[test]
fn the_wider_connectivities_are_decomposition_invariant() {
    let mask = chains();
    let blocks: [[usize; 3]; 8] = [
        [12, 12, 24],
        [4, 12, 24],
        [4, 4, 24],
        [4, 4, 8],
        [3, 3, 6],
        [5, 5, 7],
        [7, 5, 3],
        [2, 2, 2],
    ];
    for connectivity in EVERY {
        let reference = whole(&mask, connectivity);
        assert_eq!(
            canonical(&reference),
            reference,
            "the whole-volume numbering is already by scan order"
        );
        for block in blocks {
            let (answer, before) = blocked(&mask, block, connectivity);
            assert_eq!(
                answer, reference,
                "{connectivity:?} at block {block:?} is not the whole-volume answer"
            );
            // And the merge is load-bearing, asserted rather than hoped for.
            // Under 6 the chains have no join at all — that is what they are
            // for — so the block-local count is the answer and *must* be; under
            // the wider two every block size above must leave the merge
            // strictly more local components than there are answers.
            let answer = components(&reference) as usize;
            if connectivity == Connectivity::Faces {
                assert_eq!(
                    before, answer,
                    "6 joins nothing in this fixture, at block {block:?}"
                );
            } else if block != [12, 12, 24] {
                assert!(
                    before > answer,
                    "{connectivity:?} at block {block:?} merged nothing, so this case \
                     asserts nothing about the merge"
                );
            }
        }
    }
}

/// A sparse pseudorandom mask, over every connectivity and every cut.
///
/// The hand-built fixtures above are each aimed at one join, which is what
/// makes them able to discriminate — and it is also what makes them thin: they
/// touch a few of the places a face plane can be indexed and none of the
/// combinations. This one is aimed at nothing and hits everything, and the
/// density is chosen so that it can fail in **both** directions: sparse enough
/// that the whole-volume answer is hundreds of components rather than one blob,
/// so a pair joined that should not be is as visible as a pair missed.
///
/// Not a substitute for the fixtures above and not a replacement for them: a
/// disagreement here says a block size is wrong somewhere and names no join,
/// which is why the discriminating cases stay.
#[test]
fn a_sparse_pseudorandom_volume_is_decomposition_invariant_under_every_connectivity() {
    let volume = [17usize, 13, 11];
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut mask = Array3::from_elem((volume[0], volume[1], volume[2]), false);
    for cell in mask.iter_mut() {
        // xorshift64*, written out so the fixture is a function of this file
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        // About 4% set. Higher and 26-connectivity percolates: at 15% this
        // volume collapses to 26 components and the test stops being able to
        // see an over-join, which is the thing that was measured rather than
        // guessed here.
        *cell = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 56) < 10;
    }

    for connectivity in EVERY {
        let reference = whole(&mask, connectivity);
        assert!(
            components(&reference) > 40,
            "{connectivity:?} left {} components, which is too few for this to \
             fail in both directions",
            components(&reference)
        );
        for block in [
            [17usize, 13, 11],
            [9, 13, 11],
            [9, 7, 11],
            [9, 7, 6],
            [4, 4, 4],
            [5, 3, 7],
            [2, 2, 2],
            [16, 12, 10],
        ] {
            assert_eq!(
                blocked(&mask, block, connectivity).0,
                reference,
                "{connectivity:?} at block {block:?}"
            );
        }
    }
}

// ------------------------------------------------------------ the reference --

/// `scipy.ndimage.label` on `reference_mask`, canonicalised, one digit per
/// voxel in row-major order. See the module header for how these were taken.
const REFERENCE_FACES: &str = concat!(
    "100000000000000000000000000002000002000000030000000000000000000000000000",
    "000000000000004000000000000000000000000000000000000000000000000000000000",
    "500000000000000000000000000000000000000000600000000000000000000000000007",
);
const REFERENCE_FACES_AND_EDGES: &str = concat!(
    "100000000000000000000000000002000002000000030000000000000000000000000000",
    "000000000000004000000000000000000000000000000000000000000000000000000000",
    "500000000000000000000000000000000000000000500000000000000000000000000006",
);
const REFERENCE_EVERYTHING: &str = concat!(
    "100000000000000000000000000002000002000000010000000000000000000000000000",
    "000000000000001000000000000000000000000000000000000000000000000000000000",
    "300000000000000000000000000000000000000000300000000000000000000000000004",
);

/// The mask the reference was taken on: a corner chain, an edge pair, a face
/// pair and one voxel on its own, no two of which are within a step of each
/// other except as listed.
///
/// Four shapes rather than one because the three connectivities must give three
/// *different* answers on it — 7, 6 and 4 components — so no pair of them can
/// be confused for the other and a fixture that lost its power would fail the
/// counts before it failed the labels.
fn reference_mask() -> Array3<bool> {
    mask_of(
        [6, 6, 6],
        &[
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [4, 0, 0],
            [5, 1, 0],
            [0, 5, 5],
            [0, 4, 5],
            [5, 5, 5],
        ],
    )
}

#[test]
fn the_labelling_is_byte_identical_to_the_reference_implementation() {
    let mask = reference_mask();
    for (connectivity, expected, count) in [
        (Connectivity::Faces, REFERENCE_FACES, 7u32),
        (Connectivity::FacesAndEdges, REFERENCE_FACES_AND_EDGES, 6),
        (Connectivity::FacesEdgesAndCorners, REFERENCE_EVERYTHING, 4),
    ] {
        let reference = whole(&mask, connectivity);
        assert_eq!(
            expected.len(),
            6 * 6 * 6,
            "the reference is one whole volume"
        );
        assert_eq!(components(&reference), count, "{connectivity:?} count");
        assert_eq!(digits(&reference), expected, "{connectivity:?} labelling");

        // and the blocked answer is the same bytes, under every cut
        for block in [[3usize, 6, 6], [3, 3, 6], [3, 3, 3], [2, 2, 2], [4, 4, 4]] {
            assert_eq!(
                digits(&blocked(&mask, block, connectivity).0),
                expected,
                "{connectivity:?} at block {block:?}"
            );
        }
    }
}

// ------------------------------------------------------- what the fragment is --

/// **The fragment did not have to grow, and the framework did not have to
/// change.**
///
/// Two facts, both checkable. The first is geometric: a block's six face planes
/// are its whole boundary shell, so every voxel that can meet another block's is
/// already in the fragment — asserted here by counting the voxels a 26-step can
/// leave the block from and finding them all on a face.
///
/// The second is about `FragmentOp`: a merge under 26 needs the fragments of
/// the blocks that share only a lattice edge or corner, and a declared reach of
/// one block already gathers them, because `neighbourhood` is a box rather than
/// a cross. So the wider merge is expressible with the reach vocabulary that
/// exists, and no second-array input or new declaration is needed for it.
#[test]
fn the_six_planes_are_the_whole_boundary_and_a_reach_of_one_block_gathers_it() {
    let shape = [5usize, 4, 3];
    let mut on_a_face = 0usize;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                let escapes = Connectivity::FacesEdgesAndCorners
                    .offsets()
                    .iter()
                    .any(|by| {
                        (0..3).any(|axis| {
                            let moved = at[axis] as isize + by[axis];
                            moved < 0 || moved >= shape[axis] as isize
                        })
                    });
                let bounds = (0..3).any(|axis| at[axis] == 0 || at[axis] == shape[axis] - 1);
                assert_eq!(escapes, bounds, "{at:?}");
                if bounds {
                    on_a_face += 1;
                }
            }
        }
    }
    let interior = (shape[0] - 2) * (shape[1] - 2) * (shape[2] - 2);
    assert_eq!(on_a_face, shape[0] * shape[1] * shape[2] - interior);

    // the gathered neighbourhood of a reach of one block is the full box
    let gathered = neighbourhood([1, 1, 1], [1, 1, 1], [3, 3, 3]);
    assert_eq!(gathered.len(), 27);
    assert!(
        gathered.contains(&[0, 0, 0]),
        "the lattice corner is gathered"
    );
    assert!(gathered.contains(&[2, 2, 0]), "and the lattice edge");
}

/// Two blocks whose shared extent disagrees are refused rather than merged
/// wrongly, on a diagonal meeting as well as on an axis one.
#[test]
fn fragments_from_two_different_lattices_are_refused() {
    let counts = [2usize, 2, 1];
    let mut reports: BTreeMap<[usize; 3], Report> = BTreeMap::new();
    for i in 0..2 {
        for j in 0..2 {
            let edge = if [i, j] == [1, 1] { 3 } else { 4 };
            let labels = Array3::<u32>::from_elem((2, 2, edge), 1);
            reports.insert(
                [i, j, 0],
                Report {
                    labels: 1,
                    faces: planes_of(labels.view()),
                },
            );
        }
    }
    let index = LabelIndex::build(&reports, counts, |report| report.labels).expect("every block");
    let mut sets = Union::new(index.total());
    let refused = walk_seams_with(
        &reports,
        counts,
        &index,
        Connectivity::FacesEdgesAndCorners,
        |report| &report.faces,
        |a, b| sets.union(a, b),
    );
    let message = refused.expect_err("a lattice mismatch").to_string();
    assert!(
        message.contains("two different lattices"),
        "unexpected message: {message}"
    );
}
