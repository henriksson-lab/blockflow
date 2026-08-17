// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `ops::directional`: the twelve-sub-iteration,
// template-driven thinning rule.
//
// What is asserted, and why each one is here
// ------------------------------------------
// 1. **The rule is the published one, on bits.** Two structured fixtures are
//    pinned voxel by voxel and eleven more by count and digest, against outputs
//    obtained from an independent implementation of the same published
//    algorithm. This is the only property in the file that cannot be checked
//    from inside the crate — every other one is a property of *this* code, and
//    a wrong rule would satisfy all of them.
// 2. **The order of the twelve orientations is part of the answer.** Swapping a
//    quarter turn for a three-quarter turn about one axis leaves the *set* of
//    twelve rotations unchanged and reorders it, and the result is still a
//    connected, topology-preserving, one-voxel-wide curve skeleton — just a
//    different one, displaced by about a voxel. Nothing but a direct comparison
//    catches it, so there is a direct comparison.
// 3. **The border set is stale within a pass, deliberately.** Recomputing it
//    between sub-iterations is a one-line "fix" that produces a different
//    algorithm and a plausible answer; the test builds the fixed version and
//    measures how far it drifts.
// 4. **Decomposition invariance at the declared halo, and failure below it.** A
//    generous halo that works proves less than a pair: byte-identity at the
//    declared reach across block sizes and split axes, *and* a short halo that
//    is caught by the guard, *and* an understated reach that tiles perfectly and
//    is wrong. The last is what makes the declaration a claim.
// 5. **Topology is preserved and the result is one voxel wide.** Measured by a
//    route that knows nothing about the templates: all three Betti numbers
//    before and against after, and a thickness test that looks for a full 2x2
//    square or 2x2x2 cube anywhere in the answer.
//
// What is *not* asserted, and why
// -------------------------------
// Thickness is asserted on the fixtures whose topology permits it. A volume of
// dense random noise has thousands of tunnels and hundreds of cavities that this
// rule may not destroy, and preserving them **forces** thick residue: a cavity
// bounded by a one-voxel sheet cannot be thinned to a curve without opening it.
// Those fixtures are still compared on bits and still checked for topology; they
// are excluded from the thickness assertion, and the exclusion is measured
// rather than assumed — `dense_noise_is_topologically_stuck_and_says_so` shows
// the residue and shows the reason for it.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::BlockOp;
use blockflow::ops::directional::{
    border_mask, clear_faces, directional_pass, directional_pass_into,
    directional_pass_with_sources_into, directional_passes_into, directional_reach,
    directional_sub_iteration_into, directional_thin, directional_to_fixed_point, faces_are_clear,
    sub_iteration_sources, DirectionalPassOp, SUB_ITERATIONS,
};
use blockflow::ops::skeleton::{betti_numbers, connected_components, Adjacency, PassLimit};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::Dtype;

// ------------------------------------------------------------- fixtures --

/// A small deterministic generator, so a fixture is a function of its seed and
/// of nothing else. Its digests are pinned below, which is what makes the
/// comparison with an outside implementation meaningful: a fixture that drifted
/// would silently compare a different volume.
struct Bits(u64);

impl Bits {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn digest(volume: &Array3<bool>) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &value in volume.iter() {
        hash ^= u64::from(value);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn set_count(volume: &Array3<bool>) -> usize {
    volume.iter().filter(|value| **value).count()
}

fn coordinates(volume: &Array3<bool>) -> Vec<[usize; 3]> {
    let shape = volume.shape().to_vec();
    volume
        .iter()
        .enumerate()
        .filter(|(_, value)| **value)
        .map(|(index, _)| {
            [
                index / (shape[1] * shape[2]),
                (index / shape[2]) % shape[1],
                index % shape[2],
            ]
        })
        .collect()
}

/// A solid circular column along the last axis, clear of the volume's faces.
fn column(shape: [usize; 3], radius: f64) -> Array3<bool> {
    let mut volume = Array3::from_elem(shape, false);
    let centre = [shape[0] as f64 / 2.0, shape[1] as f64 / 2.0];
    for i in 1..shape[0] - 1 {
        for j in 1..shape[1] - 1 {
            for k in 4..shape[2] - 4 {
                let dx = i as f64 + 0.5 - centre[0];
                let dy = j as f64 + 0.5 - centre[1];
                volume[[i, j, k]] = dx * dx + dy * dy <= radius * radius;
            }
        }
    }
    volume
}

/// Three solid columns meeting at the middle, one along each axis: a junction
/// with three arms, which is the smallest fixture where a thinning rule can put
/// the branch point in the wrong place.
fn three_arms(shape: [usize; 3], radius: isize) -> Array3<bool> {
    let mut volume = Array3::from_elem(shape, false);
    let c = [
        shape[0] as isize / 2,
        shape[1] as isize / 2,
        shape[2] as isize / 2,
    ];
    for i in 1..shape[0] as isize - 1 {
        for j in 1..shape[1] as isize - 1 {
            for k in 1..shape[2] as isize - 1 {
                let along_0 = (j - c[1]).pow(2) + (k - c[2]).pow(2) <= radius * radius && i >= c[0];
                let along_1 = (i - c[0]).pow(2) + (k - c[2]).pow(2) <= radius * radius && j >= c[1];
                let along_2 = (i - c[0]).pow(2) + (j - c[1]).pow(2) <= radius * radius && k >= c[2];
                if along_0 || along_1 || along_2 {
                    volume[[i as usize, j as usize, k as usize]] = true;
                }
            }
        }
    }
    volume
}

/// A slanted solid slab: no rotational symmetry at all, which is what the
/// orientation-order test needs and what the circular column cannot give it.
fn slanted_slab(shape: [usize; 3]) -> Array3<bool> {
    let mut volume = Array3::from_elem(shape, false);
    for i in 1..shape[0] - 1 {
        for j in 1..shape[1] - 1 {
            for k in 1..shape[2] - 1 {
                let plane = i as isize * 2 + j as isize - k as isize;
                volume[[i, j, k]] =
                    (10..30).contains(&plane) && (6..30).contains(&i) && (6..30).contains(&j);
            }
        }
    }
    volume
}

/// Independent random voxels at a stated density, clear of the faces.
fn noise(shape: [usize; 3], density: f64, seed: u64) -> Array3<bool> {
    let mut bits = Bits(seed | 1);
    let mut volume = Array3::from_elem(shape, false);
    for i in 1..shape[0] - 1 {
        for j in 1..shape[1] - 1 {
            for k in 1..shape[2] - 1 {
                let sample = (bits.next() >> 11) as f64 / (1u64 << 53) as f64;
                volume[[i, j, k]] = sample < density;
            }
        }
    }
    volume
}

/// A union of small spheres at random places: thick structure worth thinning,
/// unlike speckle, and with several components and a few tunnels.
fn spheres(shape: [usize; 3], count: usize, seed: u64) -> Array3<bool> {
    let mut bits = Bits(seed | 1);
    let mut volume = Array3::from_elem(shape, false);
    for _ in 0..count {
        let centre = [
            5 + (bits.next() as usize) % (shape[0] - 10),
            5 + (bits.next() as usize) % (shape[1] - 10),
            5 + (bits.next() as usize) % (shape[2] - 10),
        ];
        let radius = 2 + (bits.next() as usize) % 4;
        for i in 1..shape[0] - 1 {
            for j in 1..shape[1] - 1 {
                for k in 1..shape[2] - 1 {
                    let distance = (i as isize - centre[0] as isize).pow(2)
                        + (j as isize - centre[1] as isize).pow(2)
                        + (k as isize - centre[2] as isize).pow(2);
                    if distance <= (radius * radius) as isize {
                        volume[[i, j, k]] = true;
                    }
                }
            }
        }
    }
    volume
}

/// One pinned case: the fixture, the digest of the fixture (so a drifting
/// generator is caught rather than silently compared), and the answer.
struct Case {
    name: &'static str,
    volume: Array3<bool>,
    input_digest: u64,
    /// Voxels the published algorithm leaves.
    expected_count: usize,
    /// Digest of the volume it leaves.
    expected_digest: u64,
    /// Whether the fixture's topology permits a one-voxel-wide answer.
    thinnable: bool,
}

/// The pinned cases.
///
/// **Where the expected values come from.** Each was taken from an independent
/// implementation of the same published algorithm, run over the same fixture at
/// the same fixed point, and compared voxel by voxel — not by count and not by
/// digest, which are checked here only because a test file cannot hold thirteen
/// whole volumes. The two structured cases *are* held whole, in
/// `the_column_thins_to_its_own_axis` and `the_three_arms_keep_their_junction`,
/// and those are the ones a reader can check against a picture.
fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "column r=8",
            volume: column([40, 40, 40], 8.0),
            input_digest: 0x62b8_5461_10ea_0525,
            expected_count: 17,
            expected_digest: 0xf7ac_7f51_d712_baf2,
            thinnable: true,
        },
        Case {
            name: "column r=12",
            volume: column([40, 40, 40], 12.0),
            input_digest: 0x0e1a_0e5a_d5ff_e325,
            expected_count: 9,
            expected_digest: 0x7236_4ada_ae39_e62a,
            thinnable: true,
        },
        Case {
            name: "three arms r=5",
            volume: three_arms([40, 40, 40], 5),
            input_digest: 0x8cd9_52d2_3695_f821,
            expected_count: 34,
            expected_digest: 0xe8d6_cd21_8f52_4907,
            thinnable: true,
        },
        Case {
            name: "noise 32^3 p=0.35 seed 1",
            volume: noise([32, 32, 32], 0.35, 7919),
            input_digest: 0x034f_f180_b17a_e876,
            expected_count: 6708,
            expected_digest: 0x583d_d74a_5a82_b963,
            thinnable: false,
        },
        Case {
            name: "noise 32^3 p=0.35 seed 2",
            volume: noise([32, 32, 32], 0.35, 15838),
            input_digest: 0x98a4_a31c_8993_79fd,
            expected_count: 6902,
            expected_digest: 0x51a2_ee54_bfce_7bc5,
            thinnable: false,
        },
        Case {
            name: "noise 32^3 p=0.35 seed 3",
            volume: noise([32, 32, 32], 0.35, 23757),
            input_digest: 0x7aed_d791_08bd_96f3,
            expected_count: 6863,
            expected_digest: 0xe591_8e30_4199_0b88,
            thinnable: false,
        },
        Case {
            name: "noise 32^3 p=0.35 seed 4",
            volume: noise([32, 32, 32], 0.35, 31676),
            input_digest: 0x6219_7e31_60f0_d61a,
            expected_count: 6768,
            expected_digest: 0x258e_8acb_cc23_5c45,
            thinnable: false,
        },
        Case {
            name: "noise 24^3 p=0.6 seed 1",
            volume: noise([24, 24, 24], 0.6, 104729),
            input_digest: 0x1fdc_172d_dae6_e02d,
            expected_count: 4266,
            expected_digest: 0x7d28_27b7_dbb8_47af,
            thinnable: false,
        },
        Case {
            name: "noise 24^3 p=0.6 seed 2",
            volume: noise([24, 24, 24], 0.6, 209458),
            input_digest: 0x4121_120a_9d5a_6b0b,
            expected_count: 4238,
            expected_digest: 0xe2ee_518f_4c3c_a793,
            thinnable: false,
        },
        Case {
            name: "noise 24^3 p=0.6 seed 3",
            volume: noise([24, 24, 24], 0.6, 314187),
            input_digest: 0xeb34_8475_57b6_3c40,
            expected_count: 4139,
            expected_digest: 0x01d4_e399_4976_9376,
            thinnable: false,
        },
        Case {
            name: "spheres seed 1",
            volume: spheres([40, 40, 40], 25, 15485863),
            input_digest: 0xb4cf_10f9_dd46_7d1e,
            expected_count: 101,
            expected_digest: 0x71f6_ff9e_3b25_97c0,
            thinnable: true,
        },
        Case {
            name: "spheres seed 2",
            volume: spheres([40, 40, 40], 25, 30971726),
            input_digest: 0x1664_25b8_a98c_a92e,
            expected_count: 91,
            expected_digest: 0x225f_5d23_bd00_b8a8,
            thinnable: true,
        },
        Case {
            name: "spheres seed 3",
            volume: spheres([40, 40, 40], 25, 46457589),
            input_digest: 0x9473_d8b3_04ad_2d09,
            expected_count: 77,
            expected_digest: 0x7c0c_986f_cd3b_5d4c,
            thinnable: true,
        },
    ]
}

fn fixed_point(volume: &Array3<bool>) -> Array3<bool> {
    let shape = [volume.shape()[0], volume.shape()[1], volume.shape()[2]];
    directional_to_fixed_point(volume.view(), PassLimit::for_volume(shape))
        .expect("thinning terminates")
        .0
}

// ------------------------------------------------- 1. the rule, on bits --

/// **Property 1.** Byte-identical to an independent implementation of the same
/// published algorithm, on structured fixtures and on random ones.
#[test]
fn the_answer_is_the_published_algorithms_answer() {
    for case in cases() {
        assert_eq!(
            digest(&case.volume),
            case.input_digest,
            "{}: the fixture itself has drifted, so the pinned answer below is \
             an answer to a different question",
            case.name
        );
        assert!(
            faces_are_clear(case.volume.view()),
            "{}: the rule's precondition is that the object misses the volume's faces",
            case.name
        );
        let got = fixed_point(&case.volume);
        assert_eq!(
            set_count(&got),
            case.expected_count,
            "{}: {} voxels in, {} left, {} expected",
            case.name,
            set_count(&case.volume),
            set_count(&got),
            case.expected_count
        );
        assert_eq!(
            digest(&got),
            case.expected_digest,
            "{}: the right number of voxels in the wrong places",
            case.name
        );
    }
}

/// The two structured answers held whole rather than as digests, so that what is
/// being asserted is legible.
///
/// A solid column thins to **its own axis**: one voxel per slice, all in one
/// line, with no transverse structure whatsoever. That is worth stating
/// explicitly because it is the property a thinning rule most easily gets
/// almost-right — a rule that froze alternate slabs would leave a ladder here,
/// with the axis intact and a rung every second slice, and would pass a
/// component count, a topology check and a casual look.
#[test]
fn the_column_thins_to_its_own_axis() {
    let got = fixed_point(&column([40, 40, 40], 8.0));
    let voxels = coordinates(&got);
    let expected: Vec<[usize; 3]> = (11..=27).map(|k| [19, 20, k]).collect();
    assert_eq!(
        voxels, expected,
        "the column did not thin to a straight line"
    );

    // and the same statement in a form that does not name coordinates: exactly
    // one voxel in every slice it occupies, and nothing off the line.
    let occupied: Vec<usize> = voxels.iter().map(|v| v[2]).collect();
    let mut unique = occupied.clone();
    unique.dedup();
    assert_eq!(
        occupied, unique,
        "two voxels in one slice is a rung, not an axis"
    );
}

/// The junction fixture, held whole for the same reason: this is where a
/// thinning rule puts the branch point, and a rule that put it somewhere else
/// would still return three arms.
#[test]
fn the_three_arms_keep_their_junction() {
    let got = fixed_point(&three_arms([40, 40, 40], 5));
    assert_eq!(
        coordinates(&got),
        vec![
            [20, 20, 26],
            [20, 20, 27],
            [20, 20, 28],
            [20, 20, 29],
            [20, 20, 30],
            [20, 20, 31],
            [20, 20, 32],
            [20, 20, 33],
            [20, 21, 24],
            [20, 21, 25],
            [20, 26, 20],
            [20, 27, 20],
            [20, 28, 20],
            [20, 29, 20],
            [20, 30, 20],
            [20, 31, 20],
            [20, 32, 20],
            [20, 33, 20],
            [21, 22, 23],
            [21, 25, 20],
            [22, 21, 22],
            [22, 23, 21],
            [22, 24, 20],
            [23, 22, 21],
            [24, 21, 21],
            [25, 21, 20],
            [26, 21, 20],
            [27, 20, 20],
            [28, 20, 20],
            [29, 20, 20],
            [30, 20, 20],
            [31, 20, 20],
            [32, 20, 20],
            [33, 20, 20],
        ]
    );
    assert_eq!(
        connected_components(got.view(), true, Adjacency::TwentySix),
        1,
        "the three arms must still meet"
    );
}

// -------------------------------------- 2. the order of the orientations --

/// **Property 2, and the failure that looks like success.**
///
/// A quarter turn and a three-quarter turn about one axis are each other's
/// inverse, so exchanging them in the orientation table yields *the same twelve
/// rotations in a different order*. The test asserts, in this order:
///
/// * the transposed table is a permutation of the right one — same set, so no
///   check on the individual rotations can tell them apart;
/// * the answer it produces is nonetheless **different**;
/// * and the difference is small and plausible: the same number of components,
///   the same topology, still one voxel wide, displaced by about a voxel.
///
/// It is the third bullet that makes the first two worth a test. If the wrong
/// order produced rubbish, any assertion at all would catch it.
///
/// **The circular column is deliberately absent from the fixture list**, and
/// that absence is the sharpest thing here: it is invariant under a quarter turn
/// about its own axis, so the transposition is a *symmetry of that fixture* and
/// the wrong order returns the identical volume. A suite that tested this
/// property only on symmetric objects would pass under the error. The fixtures
/// below are the asymmetric ones.
#[test]
fn the_order_of_the_twelve_orientations_is_part_of_the_answer() {
    let right = *sub_iteration_sources();
    let mut transposed = right;
    // The entries built from a quarter turn about the last axis and those built
    // from a three-quarter turn about it, exchanged.
    transposed.swap(2, 8);
    transposed.swap(4, 10);

    let mut sorted_right: Vec<_> = right.iter().collect();
    let mut sorted_transposed: Vec<_> = transposed.iter().collect();
    sorted_right.sort();
    sorted_transposed.sort();
    assert_eq!(
        sorted_right, sorted_transposed,
        "the transposition must leave the set of twelve rotations alone; if it does \
         not, this test is measuring something cruder than the failure it is for"
    );
    assert_ne!(right, transposed, "the transposition must reorder them");

    // The column is invariant under the transposition and would pass under the
    // error; asserted here so that the omission above is a measurement.
    let symmetric = column([40, 40, 40], 8.0);
    let mut under_error = symmetric.clone();
    for _ in 0..200 {
        let mut next = under_error.clone();
        directional_pass_with_sources_into(under_error.view(), &transposed, next.view_mut())
            .unwrap();
        if next == under_error {
            break;
        }
        under_error = next;
    }
    assert_eq!(
        under_error,
        fixed_point(&symmetric),
        "the circular column was expected to be blind to the transposition; if it is not, \
         the note above about symmetric fixtures is wrong"
    );

    for (name, volume) in [
        ("three arms r=5", three_arms([40, 40, 40], 5)),
        ("slanted slab", slanted_slab([40, 40, 40])),
        ("spheres seed 1", spheres([40, 40, 40], 25, 15485863)),
        ("spheres seed 2", spheres([40, 40, 40], 25, 30971726)),
    ] {
        let want = fixed_point(&volume);

        // The same fixed-point loop, over the transposed table.
        let mut current = volume.clone();
        for _ in 0..200 {
            let mut next = current.clone();
            directional_pass_with_sources_into(current.view(), &transposed, next.view_mut())
                .unwrap();
            if next == current {
                break;
            }
            current = next;
        }

        let differing = want
            .iter()
            .zip(current.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            differing > 0,
            "{name}: the transposed orientation order produced the same volume, so this \
             suite would not notice the transposition"
        );

        // And it is the *plausible* wrong answer, which is the point.
        assert_eq!(
            betti_numbers(current.view()),
            betti_numbers(want.view()),
            "{name}: the wrong order is expected to preserve topology just as well — if it \
             does not, this test is passing for the wrong reason"
        );
        println!(
            "{name}: the transposed orientation order differs in {differing} voxels \
             ({} against {}), with identical topology",
            set_count(&current),
            set_count(&want)
        );
    }
}

// ----------------------------------------- 3. the border set is stale --

/// **Property 3.** The border set is taken once per pass and reused, and that is
/// load-bearing.
///
/// The test builds the "fixed" version — the border set recomputed before every
/// sub-iteration, which is what a reader who had not been told would write — and
/// shows it is a different algorithm. It also shows the direction: recomputing
/// admits voxels that became border voxels part-way through the pass, so the
/// fixed version deletes *more*.
#[test]
fn the_border_set_is_stale_within_a_pass() {
    let sources = sub_iteration_sources();
    for (name, volume) in [
        ("column r=8", column([40, 40, 40], 8.0)),
        ("three arms r=5", three_arms([40, 40, 40], 5)),
        ("noise 24^3 p=0.6", noise([24, 24, 24], 0.6, 104729)),
    ] {
        let mut stale = Array3::from_elem(volume.raw_dim(), false);
        directional_pass_into(volume.view(), stale.view_mut()).unwrap();

        let mut fresh = volume.clone();
        for source in sources.iter() {
            let border = border_mask(fresh.view());
            let step = fresh.clone();
            directional_sub_iteration_into(step.view(), border.view(), source, fresh.view_mut())
                .unwrap();
        }

        let differing = stale
            .iter()
            .zip(fresh.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            differing > 0,
            "{name}: recomputing the border set between sub-iterations changed nothing, so \
             nothing here pins the staleness"
        );
        assert!(
            set_count(&fresh) < set_count(&stale),
            "{name}: recomputing the border set admits voxels the pass had not admitted, so \
             it should delete more, not fewer"
        );
        println!(
            "{name}: one pass leaves {} voxels; with the border set recomputed per \
             sub-iteration it leaves {}, differing in {differing}",
            set_count(&stale),
            set_count(&fresh)
        );
    }
}

// ---------------------------------------- 4. blocking and the declared halo --

/// Deliberately small. A pass reaches twelve on every axis, so a block's read
/// region is its own edge plus twenty-four; a volume much larger than this makes
/// every blocked run in the sweep read most of the volume anyway, and buys
/// nothing but minutes.
const VOLUME: [usize; 3] = [24, 20, 18];

/// The blocked fixture: solid structure worth thinning plus speckle, entirely
/// clear of the volume's faces.
fn blocked_fixture() -> Array3<bool> {
    let mut bits = Bits(31337);
    let mut volume = Array3::from_elem(VOLUME, false);
    for i in 1..VOLUME[0] - 1 {
        for j in 1..VOLUME[1] - 1 {
            for k in 1..VOLUME[2] - 1 {
                volume[[i, j, k]] = bits.next() % 100 < 22;
            }
        }
    }
    for i in 4..16 {
        for j in 5..15 {
            for k in 5..13 {
                volume[[i, j, k]] = true;
            }
        }
    }
    volume
}

fn plan_with_reach(
    workflow: &Workflow,
    block: usize,
    split_axes: &[usize],
    reach: [usize; 3],
) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(VOLUME, split_axes, block).unwrap();
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid)
        .with_dtype(Dtype::Bool);
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::Bool,
        phases: vec![phase],
        chain_reach: reach,
    }
}

fn plan(workflow: &Workflow, block: usize, split_axes: &[usize]) -> Decomposition {
    plan_with_reach(workflow, block, split_axes, workflow.chain.reach3(&VOLUME))
}

fn run_blocked(
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Array3<bool>,
) -> Array3<bool> {
    let env = ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4])
        .expect("an environment");
    execute(
        "directional",
        workflow,
        decomposition,
        &Hints::default(),
        &env,
    )
    .expect("a run");
    env.output().view::<bool>().unwrap().to_owned()
}

fn whole_volume(input: &Array3<bool>, passes: usize) -> Array3<bool> {
    let mut out = Array3::from_elem(input.raw_dim(), false);
    directional_passes_into(input.view(), passes, out.view_mut()).expect("a reference");
    out
}

/// **The declared reach is what the chain folds**, in both directions: one pass
/// reaches `SUB_ITERATIONS`, `n` passes reach `n` times that, and
/// [`directional_reach`] agrees with `Chain::reach` rather than being a second
/// number kept by hand.
#[test]
fn the_declared_reach_is_the_reach_the_chain_computes() {
    assert_eq!(directional_pass().reach(0, 64), SUB_ITERATIONS);
    for passes in 0..4 {
        let chain = directional_thin(passes);
        for axis in 0..3 {
            assert_eq!(chain.reach(axis, 64), directional_reach(passes));
        }
    }
    assert_eq!(directional_reach(1), 12);
}

/// **Property 4a. Decomposition invariance at the declared halo**, byte-identical
/// against the whole-volume answer, over several block edges and split axes, for
/// one pass and for two.
#[test]
fn the_block_runs_agree_with_the_whole_volume_answer() {
    let input = blocked_fixture();
    for (passes, blocks, axis_sets) in [
        (
            1usize,
            &[5usize, 8, 12][..],
            &[&[0usize][..], &[2][..], &[0, 1][..], &[0, 1, 2][..]][..],
        ),
        (2, &[8usize, 12][..], &[&[0usize][..], &[0, 1][..]][..]),
    ] {
        let want = whole_volume(&input, passes);
        let moved = input
            .iter()
            .zip(want.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            moved > 500,
            "at {passes} pass(es) only {moved} voxels moved, which is too few for the \
             comparisons below to mean anything"
        );

        let workflow = Workflow::new(directional_thin(passes), VOLUME, Dtype::Bool);
        assert_eq!(
            workflow.chain.reach3(&VOLUME),
            [directional_reach(passes); 3]
        );
        let mut ran = 0;
        for &block in blocks {
            for &axes in axis_sets {
                let decomposition = plan(&workflow, block, axes);
                decomposition
                    .check()
                    .expect("an honest plan must tile the volume");
                let got = run_blocked(&workflow, &decomposition, &input);
                assert_eq!(
                    got,
                    want,
                    "{passes} pass(es), block {block}, axes {axes:?}: {} voxels differ from \
                     the whole-volume answer",
                    got.iter().zip(want.iter()).filter(|(a, b)| a != b).count()
                );
                ran += 1;
            }
        }
        assert_eq!(ran, blocks.len() * axis_sets.len(), "the sweep did not run");
        println!(
            "{passes} pass(es): {ran} decompositions, all byte-identical, {moved} voxels moved"
        );
    }
}

/// **Property 4b. The guard, seen firing.** A halo one voxel short of the
/// declared reach must stop the valid regions tiling, and the executor must
/// refuse the plan for the same reason.
#[test]
fn a_halo_short_of_the_declared_reach_is_refused() {
    let input = blocked_fixture();
    let workflow = Workflow::new(directional_pass(), VOLUME, Dtype::Bool);
    let honest = plan(&workflow, 8, &[0]);
    honest.check().expect("the honest plan tiles");

    let forced = honest.with_forced_halo([SUB_ITERATIONS - 1, 0, 0]);
    let err = forced
        .check()
        .expect_err("a short halo must not check out")
        .to_string();
    assert!(
        err.contains("do not tile the volume exactly"),
        "expected the tiling guard, got: {err}"
    );

    let env = ArrayEnvironment::new(input.into(), 1, [4, 4, 4]).unwrap();
    let err = execute("short", &workflow, &forced, &Hints::default(), &env)
        .expect_err("the executor must refuse a short halo")
        .to_string();
    assert!(err.contains("do not tile the volume exactly"), "got {err}");
}

/// **Property 4c, and the half a generous halo cannot show.** A phase that
/// *understates* its reach tiles perfectly, runs without complaint, and produces
/// a wrong volume.
///
/// The declared reach is the derivation — twelve, one per sub-iteration — and
/// the smallest halo that happens to be exact on any given data is smaller than
/// that. So this test does not assert that eleven breaks; it searches downward
/// for the largest understatement that shows, **prints it**, and asserts only
/// that some understatement does. A test that pinned the measured number would
/// be pinning a property of this fixture rather than of the rule.
#[test]
fn an_understated_reach_tiles_perfectly_and_is_wrong() {
    let input = blocked_fixture();
    let want = whole_volume(&input, 1);
    let workflow = Workflow::new(directional_pass(), VOLUME, Dtype::Bool);

    let mut largest_that_shows: Option<usize> = None;
    for reach in (0..SUB_ITERATIONS).rev() {
        let mut wrong_anywhere = false;
        for block in [6usize, 8] {
            for axes in [&[0usize][..], &[0, 1, 2][..]] {
                let plan = plan_with_reach(&workflow, block, axes, [reach; 3]);
                plan.check()
                    .expect("an understated reach is self-consistent and tiles");
                if run_blocked(&workflow, &plan, &input) != want {
                    wrong_anywhere = true;
                }
            }
        }
        if wrong_anywhere {
            largest_that_shows = Some(reach);
            break;
        }
    }

    let reach = largest_that_shows.expect(
        "no halo below the declared twelve produced a wrong volume on this fixture, which \
         would mean the declared reach is unfalsifiable here rather than derived",
    );
    println!(
        "the largest understated halo that is measurably wrong on this fixture is {reach}; \
         the declared reach is {SUB_ITERATIONS}, which is the derivation rather than the \
         measurement"
    );
    assert!(reach < SUB_ITERATIONS);
}

/// An all-set block is not constant under this rule, so nothing may be declared
/// for it — and the test asserts both halves, that the declaration is withheld
/// and that it had to be.
#[test]
fn an_all_set_block_is_not_constant_and_nothing_is_declared_for_it() {
    let op = DirectionalPassOp::default();
    assert_eq!(op.constant_maps_to(0.0), Some(0.0));
    assert_eq!(op.constant_maps_to(1.0), None);

    let solid = Array3::from_elem([16usize, 16, 16], true);
    let mut out = Array3::from_elem(solid.raw_dim(), false);
    directional_pass_into(solid.view(), out.view_mut()).unwrap();
    assert!(
        set_count(&out) < solid.len(),
        "a solid block whose faces see the outside as background must lose voxels, or the \
         withheld declaration would be over-cautious"
    );
}

/// The faces of a volume are a whole-volume concern, and the two functions that
/// say so behave as a pair.
#[test]
fn the_faces_of_a_volume_are_cleared_once_and_checked_once() {
    let mut volume = Array3::from_elem([8usize, 7, 6], true);
    assert!(!faces_are_clear(volume.view()));
    clear_faces(&mut volume);
    assert!(faces_are_clear(volume.view()));
    assert_eq!(set_count(&volume), 6 * 5 * 4);
    // Idempotent, which is what makes it safe to impose before every round.
    let once = volume.clone();
    clear_faces(&mut volume);
    assert_eq!(volume, once);
}

// -------------------------------- 5. topology, and how thin the answer is --

/// How many full 2x2x2 cubes and full 2x2 squares (in any of the three axis
/// planes) the volume contains. Both are zero exactly when the object is one
/// voxel wide in the strong sense: no two-by-two patch of it lies flat anywhere.
fn thickness(volume: &Array3<bool>) -> (usize, usize) {
    let shape = volume.shape().to_vec();
    let mut cubes = 0;
    let mut squares = 0;
    for i in 0..shape[0] - 1 {
        for j in 0..shape[1] - 1 {
            for k in 0..shape[2] - 1 {
                let at = |a: usize, b: usize, c: usize| volume[[i + a, j + b, k + c]];
                if (0..8).all(|corner| at(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1)) {
                    cubes += 1;
                }
                let faces = [
                    [at(0, 0, 0), at(0, 1, 0), at(0, 0, 1), at(0, 1, 1)],
                    [at(0, 0, 0), at(1, 0, 0), at(0, 0, 1), at(1, 0, 1)],
                    [at(0, 0, 0), at(1, 0, 0), at(0, 1, 0), at(1, 1, 0)],
                ];
                squares += faces
                    .iter()
                    .filter(|face| face.iter().all(|value| *value))
                    .count();
            }
        }
    }
    (cubes, squares)
}

/// **Property 5.** All three Betti numbers survive every fixture, measured by a
/// route that knows nothing about the deletion rule.
#[test]
fn the_topology_of_every_fixture_survives() {
    for case in cases() {
        let before = betti_numbers(case.volume.view());
        let after = betti_numbers(fixed_point(&case.volume).view());
        assert_eq!(
            before, after,
            "{}: thinning changed the topology, which this rule may never do",
            case.name
        );
        println!("{}: betti {before:?} preserved", case.name);
    }
}

/// **Property 5, the other half.** The answer is one voxel wide, on every
/// fixture whose topology permits it: not one full 2x2 square anywhere, let
/// alone a 2x2x2 cube.
#[test]
fn the_answer_is_one_voxel_wide_wherever_topology_permits() {
    let mut checked = 0;
    for case in cases().into_iter().filter(|case| case.thinnable) {
        let got = fixed_point(&case.volume);
        let (cubes, squares) = thickness(&got);
        assert_eq!(
            (cubes, squares),
            (0, 0),
            "{}: the answer still contains {cubes} solid 2x2x2 cube(s) and {squares} flat \
             2x2 square(s), so it is not one voxel wide",
            case.name
        );
        // and it is a real answer rather than an empty one
        assert!(set_count(&got) > 0, "{}: nothing left at all", case.name);
        checked += 1;
    }
    assert!(
        checked >= 5,
        "only {checked} fixtures were thickness-checked"
    );
}

/// The exclusion above, measured rather than asserted: dense noise keeps
/// hundreds of cavities that the rule may not destroy, and a cavity has to be
/// bounded by something.
#[test]
fn dense_noise_is_topologically_stuck_and_says_so() {
    let volume = noise([24, 24, 24], 0.6, 104729);
    let before = betti_numbers(volume.view());
    let got = fixed_point(&volume);
    let (cubes, squares) = thickness(&got);
    assert!(
        before[2] > 100,
        "the fixture is here for its cavities and has {} of them",
        before[2]
    );
    assert_eq!(betti_numbers(got.view()), before);
    assert!(
        cubes > 0,
        "dense noise was expected to leave thick residue; if it no longer does, the \
         thickness assertion above should stop excluding it"
    );
    println!(
        "dense noise: betti {before:?} preserved, and the price is {cubes} solid 2x2x2 \
         cube(s) and {squares} flat 2x2 square(s) of residue"
    );
}

/// The rule is **translation invariant**: nothing in it reads the anchor, so
/// shifting the object inside the volume shifts the answer and changes nothing
/// else. This is the property that makes a block's answer a function of its
/// buffer, and it is worth a test because the neighbouring op in this crate does
/// not have it.
#[test]
fn the_answer_moves_with_the_object_and_does_nothing_else() {
    let shape = [40usize, 40, 40];
    let base = spheres(shape, 12, 999331);
    let thinned = fixed_point(&base);
    for shift in [[1usize, 0, 0], [0, 1, 0], [0, 0, 1], [2, 3, 1]] {
        let mut moved = Array3::from_elem(shape, false);
        let mut expected = Array3::from_elem(shape, false);
        let mut fits = true;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let to = [i + shift[0], j + shift[1], k + shift[2]];
                    if (0..3).any(|axis| to[axis] >= shape[axis]) {
                        fits &= !base[[i, j, k]];
                        continue;
                    }
                    moved[to] = base[[i, j, k]];
                    expected[to] = thinned[[i, j, k]];
                }
            }
        }
        assert!(fits, "the shifted fixture must still fit in the volume");
        assert_eq!(
            fixed_point(&moved),
            expected,
            "shifting by {shift:?} changed more than the position of the answer"
        );
    }
}
